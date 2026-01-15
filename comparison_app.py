"""
Wellness AI Side-by-Side Comparison Application

A FastAPI application that streams responses from both PydanticAI and OpenAI Agents SDK
wellness assistants side-by-side for real-time comparison.

Usage:
    python comparison_app.py

Then open http://localhost:8080 in your browser.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

# Import PydanticAI components
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Tuple

# Import Tavily for PydanticAI web search
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# Import OpenAI Agents SDK components
from openai import OpenAI
from agents import Agent as OpenAIAgent, Runner, FileSearchTool, WebSearchTool
from openai.types.responses import ResponseTextDeltaEvent

# Import shared utilities
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase

import nest_asyncio
nest_asyncio.apply()


# =============================================================================
# PydanticAI Pipeline Components
# =============================================================================

@dataclass
class WellnessDeps:
    """Dependencies for the PydanticAI wellness assistant."""
    vector_db: VectorDatabase
    tavily_api_key: str | None = None
    max_results: int = 4
    enable_web_search: bool = True
    web_search_max_results: int = 3


class WellnessResponse(BaseModel):
    """Structured response from the PydanticAI wellness assistant."""
    answer: str
    confidence: str
    sources_used: int
    used_knowledge_base: bool = False
    used_web_search: bool = False
    disclaimer: str = "Please consult a healthcare professional for medical advice."


# Create PydanticAI agent for streaming (no structured output for streaming compatibility)
pydantic_agent = Agent(
    "openai:gpt-4.1-mini",
    deps_type=WellnessDeps,
    instructions="""You are a helpful personal wellness assistant that answers health and wellness questions.

You have access to two information sources:
1. **Knowledge Base (search_wellness_knowledge)**: Your primary source - a curated wellness guide
2. **Web Search (search_web)**: For current research, news, or topics not in the knowledge base

Instructions:
- ALWAYS search the knowledge base first for wellness questions
- Use web search for: current research/studies, recent health news, topics not fully covered in knowledge base
- Be accurate and cite your sources
- Keep responses detailed but focused
- Include a reminder that users should consult healthcare professionals for medical advice

Provide helpful, accurate wellness information based on the sources you find.""",
)


@pydantic_agent.tool
async def search_wellness_knowledge(ctx: RunContext[WellnessDeps], query: str) -> str:
    """Search the wellness knowledge base for relevant information."""
    results: List[Tuple[str, float]] = ctx.deps.vector_db.search_by_text(
        query, k=ctx.deps.max_results
    )
    if not results:
        return "No relevant information found in the wellness knowledge base."

    formatted_results = []
    for i, (content, score) in enumerate(results, 1):
        formatted_results.append(f"[Source {i}] (relevance: {score:.3f})\n{content}")
    return "\n\n---\n\n".join(formatted_results)


@pydantic_agent.tool
async def search_web(ctx: RunContext[WellnessDeps], query: str) -> str:
    """Search the web for current health and wellness information."""
    if not TAVILY_AVAILABLE or not ctx.deps.enable_web_search or not ctx.deps.tavily_api_key:
        return "Web search is not available."

    try:
        client = TavilyClient(api_key=ctx.deps.tavily_api_key)
        response = client.search(
            query=f"{query} health wellness",
            max_results=ctx.deps.web_search_max_results,
            search_depth="basic",
        )
        results = response.get("results", [])
        if not results:
            return "No relevant web results found."

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"[Web Source {i}]\nTitle: {result.get('title', 'N/A')}\n"
                f"URL: {result.get('url', 'N/A')}\nSummary: {result.get('content', 'N/A')}"
            )
        return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        return f"Web search error: {str(e)}"


# =============================================================================
# OpenAI Agents Pipeline Components
# =============================================================================

def setup_openai_vector_store(client: OpenAI, file_path: str) -> str:
    """Create a vector store and upload a file for semantic search."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    with open(file_path, "rb") as f:
        file_upload = client.files.create(file=f, purpose="assistants")

    vector_store = client.vector_stores.create(name="wellness-comparison-kb")
    client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store.id,
        file_id=file_upload.id
    )
    return vector_store.id


def create_openai_agent(vector_store_id: str, max_results: int = 4) -> OpenAIAgent:
    """Create an OpenAI Agents wellness assistant."""
    instructions = """You are a helpful personal wellness assistant that answers health and wellness questions.

You have access to two information sources:
1. **Knowledge Base (FileSearchTool)**: Your primary source - a curated wellness guide
2. **Web Search (WebSearchTool)**: For current research, news, or topics not in the knowledge base

Instructions:
- ALWAYS search the knowledge base first for wellness questions
- Use web search for: current research/studies, recent health news, topics not in the knowledge base
- Be accurate and cite your sources (knowledge base vs web)
- Keep responses detailed but focused
- Include a gentle reminder that users should consult healthcare professionals for medical advice"""

    return OpenAIAgent(
        name="Wellness Assistant",
        instructions=instructions,
        model="gpt-4.1-mini",
        tools=[
            FileSearchTool(
                max_num_results=max_results,
                vector_store_ids=[vector_store_id],
                include_search_results=True,
            ),
            WebSearchTool(),
        ],
    )


# =============================================================================
# Global State (initialized on startup)
# =============================================================================

class AppState:
    """Application state holding initialized pipelines."""
    pydantic_deps: WellnessDeps | None = None
    openai_agent: OpenAIAgent | None = None
    openai_client: OpenAI | None = None
    openai_vector_store_id: str | None = None
    initialized: bool = False


state = AppState()


# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize both pipelines on startup."""
    import sys
    print("Initializing wellness assistants...", flush=True)
    sys.stdout.flush()

    document_path = "data/HealthWellnessGuide.txt"

    try:
        # Initialize PydanticAI pipeline
        print("  - Building PydanticAI vector database...", flush=True)
        loader = TextFileLoader(document_path)
        documents = loader.load_documents()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_texts(documents)
        vector_db = VectorDatabase()
        vector_db = await vector_db.abuild_from_list(chunks)

        tavily_key = os.getenv("TAVILY_API_KEY")
        state.pydantic_deps = WellnessDeps(
            vector_db=vector_db,
            tavily_api_key=tavily_key,
            max_results=4,
            enable_web_search=TAVILY_AVAILABLE and tavily_key is not None,
        )
        print(f"    Built with {len(vector_db.vectors)} vectors", flush=True)

        # Initialize OpenAI Agents pipeline (run in thread to avoid blocking)
        print("  - Setting up OpenAI Agents vector store...", flush=True)
        import asyncio
        state.openai_client = OpenAI()

        # Run synchronous OpenAI setup in a thread pool
        loop = asyncio.get_event_loop()
        state.openai_vector_store_id = await loop.run_in_executor(
            None, setup_openai_vector_store, state.openai_client, document_path
        )
        state.openai_agent = create_openai_agent(state.openai_vector_store_id)
        print(f"    Created vector store: {state.openai_vector_store_id}", flush=True)

        state.initialized = True
        print("Both wellness assistants initialized!", flush=True)
        print("Open http://localhost:8080 in your browser", flush=True)

    except Exception as e:
        print(f"Error during initialization: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    yield

    # Cleanup on shutdown
    if state.openai_vector_store_id and state.openai_client:
        try:
            state.openai_client.vector_stores.delete(state.openai_vector_store_id)
            print("Cleaned up OpenAI vector store")
        except Exception:
            pass


app = FastAPI(title="Wellness AI Comparison", lifespan=lifespan)


# =============================================================================
# SSE Streaming Endpoints
# =============================================================================

async def pydantic_stream_generator(question: str):
    """Generate SSE events for PydanticAI streaming response."""
    start_time = time.time()
    tools_used = {"knowledge_base": False, "web_search": False}

    try:
        yield {"event": "start", "data": json.dumps({"status": "starting"})}

        if state.pydantic_deps is None:
            yield {"event": "error", "data": json.dumps({"error": "Pipeline not initialized"})}
            return

        async with pydantic_agent.run_stream(question, deps=state.pydantic_deps) as response:
            async for text in response.stream_text():
                yield {"event": "text", "data": json.dumps({"text": text})}

            # Check which tools were used by examining the messages
            for msg in response.all_messages():
                msg_str = str(msg)
                if "search_wellness_knowledge" in msg_str:
                    tools_used["knowledge_base"] = True
                if "search_web" in msg_str:
                    tools_used["web_search"] = True

            elapsed = time.time() - start_time

            yield {
                "event": "complete",
                "data": json.dumps({
                    "confidence": "high" if tools_used["knowledge_base"] else "medium",
                    "sources_used": sum(tools_used.values()),
                    "used_knowledge_base": tools_used["knowledge_base"],
                    "used_web_search": tools_used["web_search"],
                    "disclaimer": "Please consult a healthcare professional for medical advice.",
                    "elapsed_seconds": round(elapsed, 2),
                })
            }
    except Exception as e:
        yield {"event": "error", "data": json.dumps({"error": str(e)})}


async def openai_stream_generator(question: str):
    """Generate SSE events for OpenAI Agents streaming response."""
    start_time = time.time()
    full_text = ""

    try:
        yield {"event": "start", "data": json.dumps({"status": "starting"})}

        result = Runner.run_streamed(state.openai_agent, input=question)

        async for event in result.stream_events():
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseTextDeltaEvent):
                    delta = event.data.delta
                    full_text += delta
                    yield {"event": "delta", "data": json.dumps({"delta": delta, "text": full_text})}

        elapsed = time.time() - start_time
        yield {
            "event": "complete",
            "data": json.dumps({
                "elapsed_seconds": round(elapsed, 2),
                "final_text": full_text,
            })
        }
    except Exception as e:
        yield {"event": "error", "data": json.dumps({"error": str(e)})}


@app.get("/api/stream/pydantic")
async def stream_pydantic(q: str):
    """Stream PydanticAI wellness assistant response."""
    if not state.initialized:
        return {"error": "Not initialized"}
    return EventSourceResponse(pydantic_stream_generator(q))


@app.get("/api/stream/openai")
async def stream_openai(q: str):
    """Stream OpenAI Agents wellness assistant response."""
    if not state.initialized:
        return {"error": "Not initialized"}
    return EventSourceResponse(openai_stream_generator(q))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if state.initialized else "initializing",
        "pydantic_ready": state.pydantic_deps is not None,
        "openai_ready": state.openai_agent is not None,
    }


# =============================================================================
# Frontend
# =============================================================================

FRONTEND_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wellness AI Comparison</title>
    <style>
        :root {
            --bg-dark: #0a0a0a;
            --bg-card: #141414;
            --bg-input: #1a1a1a;
            --border-color: #2a2a2a;
            --text-primary: #ffffff;
            --text-secondary: #888888;
            --text-muted: #555555;
            --pydantic-color: #e92063;
            --pydantic-bg: rgba(233, 32, 99, 0.1);
            --openai-color: #10a37f;
            --openai-bg: rgba(16, 163, 127, 0.1);
            --accent-blue: #3b82f6;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 24px;
        }

        /* Header */
        .header {
            text-align: center;
            margin-bottom: 32px;
        }

        .header h1 {
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 8px;
            background: linear-gradient(135deg, var(--pydantic-color), var(--openai-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 14px;
        }

        /* Input Section */
        .input-section {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
        }

        .input-wrapper {
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
        }

        .input-wrapper input {
            flex: 1;
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px 20px;
            font-size: 16px;
            color: var(--text-primary);
            outline: none;
            transition: border-color 0.2s;
        }

        .input-wrapper input:focus {
            border-color: var(--accent-blue);
        }

        .input-wrapper input::placeholder {
            color: var(--text-muted);
        }

        .submit-btn {
            background: linear-gradient(135deg, var(--pydantic-color), var(--openai-color));
            border: none;
            border-radius: 12px;
            padding: 16px 32px;
            font-size: 16px;
            font-weight: 600;
            color: white;
            cursor: pointer;
            transition: opacity 0.2s, transform 0.1s;
        }

        .submit-btn:hover {
            opacity: 0.9;
        }

        .submit-btn:active {
            transform: scale(0.98);
        }

        .submit-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Sample Questions */
        .samples-label {
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        .sample-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .sample-chip {
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 8px 16px;
            font-size: 13px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
        }

        .sample-chip:hover {
            border-color: var(--accent-blue);
            color: var(--text-primary);
        }

        /* Comparison Grid */
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }

        @media (max-width: 1024px) {
            .comparison-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Panel */
        .panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            min-height: 500px;
        }

        .panel.pydantic {
            border-top: 3px solid var(--pydantic-color);
        }

        .panel.openai {
            border-top: 3px solid var(--openai-color);
        }

        .panel-header {
            padding: 20px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .panel-title {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .panel-title h2 {
            font-size: 18px;
            font-weight: 600;
        }

        .panel.pydantic .panel-title h2 {
            color: var(--pydantic-color);
        }

        .panel.openai .panel-title h2 {
            color: var(--openai-color);
        }

        .badge {
            font-size: 11px;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 500;
        }

        .panel.pydantic .badge {
            background: var(--pydantic-bg);
            color: var(--pydantic-color);
        }

        .panel.openai .badge {
            background: var(--openai-bg);
            color: var(--openai-color);
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: var(--text-muted);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-muted);
        }

        .status-dot.streaming {
            background: #22c55e;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Panel Content */
        .panel-content {
            flex: 1;
            padding: 24px;
            overflow-y: auto;
            font-size: 15px;
            line-height: 1.8;
            color: var(--text-primary);
        }

        .panel-content.empty {
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            font-style: italic;
        }

        .panel-content p {
            margin-bottom: 16px;
        }

        .cursor {
            display: inline-block;
            width: 2px;
            height: 1.2em;
            background: var(--text-primary);
            margin-left: 2px;
            animation: blink 1s infinite;
            vertical-align: text-bottom;
        }

        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }

        /* Panel Footer */
        .panel-footer {
            padding: 16px 24px;
            border-top: 1px solid var(--border-color);
            background: var(--bg-input);
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            font-size: 13px;
        }

        .meta-item {
            display: flex;
            align-items: center;
            gap: 6px;
            color: var(--text-secondary);
        }

        .meta-item strong {
            color: var(--text-primary);
        }

        .meta-item.success strong {
            color: #22c55e;
        }

        .meta-item.warning strong {
            color: #f59e0b;
        }

        /* Loading Spinner */
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border-color);
            border-top-color: var(--accent-blue);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Wellness AI Side-by-Side Comparison</h1>
            <p>Compare PydanticAI and OpenAI Agents SDK responses in real-time</p>
        </header>

        <section class="input-section">
            <div class="input-wrapper">
                <input type="text" id="questionInput" placeholder="Ask a wellness question..." />
                <button class="submit-btn" id="submitBtn" onclick="submitQuestion()">
                    Compare
                </button>
            </div>
            <div class="samples-label">Sample Questions</div>
            <div class="sample-chips">
                <span class="sample-chip" onclick="selectSample(this)">What exercises help with lower back pain?</span>
                <span class="sample-chip" onclick="selectSample(this)">Natural remedies for improving sleep quality?</span>
                <span class="sample-chip" onclick="selectSample(this)">Latest research on intermittent fasting?</span>
                <span class="sample-chip" onclick="selectSample(this)">How can I manage stress through lifestyle changes?</span>
                <span class="sample-chip" onclick="selectSample(this)">Benefits of meditation for mental health?</span>
            </div>
        </section>

        <div class="comparison-grid">
            <!-- PydanticAI Panel -->
            <div class="panel pydantic">
                <div class="panel-header">
                    <div class="panel-title">
                        <h2>PydanticAI</h2>
                        <span class="badge">Tavily Search</span>
                    </div>
                    <div class="status-indicator" id="pydanticStatus">
                        <span class="status-dot" id="pydanticDot"></span>
                        <span id="pydanticStatusText">Ready</span>
                    </div>
                </div>
                <div class="panel-content empty" id="pydanticContent">
                    Ask a question to see the response
                </div>
                <div class="panel-footer" id="pydanticFooter" style="display: none;">
                    <div class="meta-item" id="pydanticConfidence">
                        <span>Confidence:</span>
                        <strong>-</strong>
                    </div>
                    <div class="meta-item" id="pydanticSources">
                        <span>Sources:</span>
                        <strong>-</strong>
                    </div>
                    <div class="meta-item" id="pydanticKB">
                        <span>Knowledge Base:</span>
                        <strong>-</strong>
                    </div>
                    <div class="meta-item" id="pydanticWeb">
                        <span>Web Search:</span>
                        <strong>-</strong>
                    </div>
                    <div class="meta-item" id="pydanticTime">
                        <span>Time:</span>
                        <strong>-</strong>
                    </div>
                </div>
            </div>

            <!-- OpenAI Agents Panel -->
            <div class="panel openai">
                <div class="panel-header">
                    <div class="panel-title">
                        <h2>OpenAI Agents SDK</h2>
                        <span class="badge">WebSearchTool</span>
                    </div>
                    <div class="status-indicator" id="openaiStatus">
                        <span class="status-dot" id="openaiDot"></span>
                        <span id="openaiStatusText">Ready</span>
                    </div>
                </div>
                <div class="panel-content empty" id="openaiContent">
                    Ask a question to see the response
                </div>
                <div class="panel-footer" id="openaiFooter" style="display: none;">
                    <div class="meta-item" id="openaiTime">
                        <span>Time:</span>
                        <strong>-</strong>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let pydanticSource = null;
        let openaiSource = null;

        function selectSample(el) {
            document.getElementById('questionInput').value = el.textContent;
        }

        function submitQuestion() {
            const input = document.getElementById('questionInput');
            const question = input.value.trim();
            if (!question) return;

            // Close existing connections
            if (pydanticSource) pydanticSource.close();
            if (openaiSource) openaiSource.close();

            // Reset UI
            resetPanel('pydantic');
            resetPanel('openai');

            // Disable button
            const btn = document.getElementById('submitBtn');
            btn.disabled = true;
            btn.textContent = 'Streaming...';

            // Start both streams
            startPydanticStream(question);
            startOpenAIStream(question);
        }

        function resetPanel(panel) {
            const content = document.getElementById(`${panel}Content`);
            const footer = document.getElementById(`${panel}Footer`);
            const dot = document.getElementById(`${panel}Dot`);
            const statusText = document.getElementById(`${panel}StatusText`);

            content.innerHTML = '';
            content.classList.remove('empty');
            footer.style.display = 'none';
            dot.classList.remove('streaming');
            statusText.textContent = 'Connecting...';
        }

        function startPydanticStream(question) {
            const content = document.getElementById('pydanticContent');
            const footer = document.getElementById('pydanticFooter');
            const dot = document.getElementById('pydanticDot');
            const statusText = document.getElementById('pydanticStatusText');

            pydanticSource = new EventSource(`/api/stream/pydantic?q=${encodeURIComponent(question)}`);

            pydanticSource.addEventListener('start', (e) => {
                dot.classList.add('streaming');
                statusText.textContent = 'Streaming...';
            });

            pydanticSource.addEventListener('text', (e) => {
                const data = JSON.parse(e.data);
                content.innerHTML = formatText(data.text) + '<span class="cursor"></span>';
            });

            pydanticSource.addEventListener('complete', (e) => {
                const data = JSON.parse(e.data);

                // Remove cursor
                const cursor = content.querySelector('.cursor');
                if (cursor) cursor.remove();

                // Update status
                dot.classList.remove('streaming');
                statusText.textContent = 'Complete';

                // Update footer
                footer.style.display = 'flex';
                updateMeta('pydanticConfidence', data.confidence, data.confidence === 'high' ? 'success' : 'warning');
                updateMeta('pydanticSources', data.sources_used);
                updateMeta('pydanticKB', data.used_knowledge_base ? 'Yes' : 'No', data.used_knowledge_base ? 'success' : '');
                updateMeta('pydanticWeb', data.used_web_search ? 'Yes' : 'No', data.used_web_search ? 'success' : '');
                updateMeta('pydanticTime', `${data.elapsed_seconds}s`);

                pydanticSource.close();
                checkComplete();
            });

            pydanticSource.addEventListener('error', (e) => {
                dot.classList.remove('streaming');
                statusText.textContent = 'Error';
                content.innerHTML = '<span style="color: #ef4444;">An error occurred</span>';
                pydanticSource.close();
                checkComplete();
            });
        }

        function startOpenAIStream(question) {
            const content = document.getElementById('openaiContent');
            const footer = document.getElementById('openaiFooter');
            const dot = document.getElementById('openaiDot');
            const statusText = document.getElementById('openaiStatusText');

            openaiSource = new EventSource(`/api/stream/openai?q=${encodeURIComponent(question)}`);

            openaiSource.addEventListener('start', (e) => {
                dot.classList.add('streaming');
                statusText.textContent = 'Streaming...';
            });

            openaiSource.addEventListener('delta', (e) => {
                const data = JSON.parse(e.data);
                content.innerHTML = formatText(data.text) + '<span class="cursor"></span>';
            });

            openaiSource.addEventListener('complete', (e) => {
                const data = JSON.parse(e.data);

                // Remove cursor
                const cursor = content.querySelector('.cursor');
                if (cursor) cursor.remove();

                // Update status
                dot.classList.remove('streaming');
                statusText.textContent = 'Complete';

                // Update footer
                footer.style.display = 'flex';
                updateMeta('openaiTime', `${data.elapsed_seconds}s`);

                openaiSource.close();
                checkComplete();
            });

            openaiSource.addEventListener('error', (e) => {
                dot.classList.remove('streaming');
                statusText.textContent = 'Error';
                content.innerHTML = '<span style="color: #ef4444;">An error occurred</span>';
                openaiSource.close();
                checkComplete();
            });
        }

        function updateMeta(id, value, className = '') {
            const el = document.getElementById(id);
            const strong = el.querySelector('strong');
            strong.textContent = value;
            el.className = 'meta-item' + (className ? ` ${className}` : '');
        }

        function formatText(text) {
            // Convert markdown-like formatting to HTML
            return text
                .replace(/\n\n/g, '</p><p>')
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.+?)\*/g, '<em>$1</em>')
                .replace(/^/, '<p>')
                .replace(/$/, '</p>');
        }

        function checkComplete() {
            const pydanticStatus = document.getElementById('pydanticStatusText').textContent;
            const openaiStatus = document.getElementById('openaiStatusText').textContent;

            if ((pydanticStatus === 'Complete' || pydanticStatus === 'Error') &&
                (openaiStatus === 'Complete' || openaiStatus === 'Error')) {
                const btn = document.getElementById('submitBtn');
                btn.disabled = false;
                btn.textContent = 'Compare';
            }
        }

        // Allow Enter key to submit
        document.getElementById('questionInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') submitQuestion();
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend."""
    return FRONTEND_HTML


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)

    if not os.getenv("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY not set. PydanticAI web search will be disabled.")

    print("Starting Wellness AI Comparison Server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
