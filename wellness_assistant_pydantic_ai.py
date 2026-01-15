# %% [markdown]
# # Personal Wellness Assistant - PydanticAI Implementation
#
# This is an alternate implementation of the RAG-based Wellness Assistant using
# **PydanticAI** - a type-safe, Pythonic agent framework from the creators of Pydantic.
#
# ## Key Differences from DIY Implementation:
# - Type-safe responses using Pydantic models
# - Clean dependency injection pattern
# - `@agent.tool` decorator for retrieval
# - Web search via Tavily for current information
# - Reuses our existing VectorDatabase (keeps the learning!)
#
# ## Why PydanticAI?
# - Stays Pythonic (no framework "magic")
# - Type safety catches errors early
# - Patterns transfer to other frameworks
# - You already know Pydantic from FastAPI, data validation, etc.
#
# ## Prerequisites:
# ```bash
# uv add pydantic-ai "pydantic-ai-slim[tavily]"
# ```
#
# ## References:
# - [PydanticAI Documentation](https://ai.pydantic.dev/)
# - [PydanticAI RAG Example](https://ai.pydantic.dev/examples/rag/)
# - [Tools Documentation](https://ai.pydantic.dev/tools/)
# - [Common Tools - Tavily](https://ai.pydantic.dev/common-tools/)
# - [Tavily Integration](https://docs.tavily.com/documentation/integrations/pydantic-ai)

# %%
import asyncio
import os
from dataclasses import dataclass
from typing import List, Tuple

# Note: Install with `uv add pydantic-ai "pydantic-ai-slim[tavily]"`
try:
    from pydantic_ai import Agent, RunContext
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "PydanticAI not installed. Run: uv add pydantic-ai"
    )

# Optional: Tavily for web search (more reliable than DuckDuckGo)
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("Note: Tavily not installed. Web search disabled. Run: uv add 'pydantic-ai-slim[tavily]'")

# Reuse our existing aimakerspace library!
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase

# %%
import nest_asyncio
nest_asyncio.apply()

# %% [markdown]
# ## Step 1: Define Dependencies
#
# PydanticAI uses **dependency injection** to pass data to tools.
# This is cleaner than global variables and makes testing easier.
#
# We'll inject our VectorDatabase so the retrieval tool can access it.

# %%
@dataclass
class WellnessDeps:
    """
    Dependencies for the wellness assistant agent.

    Using a dataclass keeps dependencies explicit and type-checked.
    This pattern is common in production Python (FastAPI, pytest fixtures, etc.)
    """
    vector_db: VectorDatabase
    tavily_api_key: str | None = None  # Required for web search
    max_results: int = 4
    enable_web_search: bool = True
    web_search_max_results: int = 3


# %% [markdown]
# ## Step 2: Define Response Model (Optional but Recommended)
#
# Type-safe responses catch errors early and provide better IDE support.
# The agent will return structured data matching this schema.

# %%
class WellnessResponse(BaseModel):
    """Structured response from the wellness assistant."""
    answer: str
    confidence: str  # "high", "medium", "low"
    sources_used: int
    used_knowledge_base: bool = False
    used_web_search: bool = False
    disclaimer: str = "Please consult a healthcare professional for medical advice."


# %% [markdown]
# ## Step 3: Create the Agent
#
# The agent is configured with:
# - **System prompt**: Defines behavior (similar to RAG_SYSTEM_TEMPLATE)
# - **deps_type**: Type hint for dependency injection
# - **output_type**: Optional structured output type (Pydantic model)

# %%
# Create the agent with system instructions
wellness_agent = Agent(
    "openai:gpt-4.1-mini",
    deps_type=WellnessDeps,
    output_type=WellnessResponse,  # Structured output
    instructions="""You are a helpful personal wellness assistant that answers health and wellness questions.

You have access to two information sources:
1. **Knowledge Base (search_wellness_knowledge)**: Your primary source - a curated wellness guide
2. **Web Search (search_web)**: For current research, news, or topics not in the knowledge base

Instructions:
- ALWAYS search the knowledge base first for wellness questions
- Use web search for: current research/studies, recent health news, topics not fully covered in knowledge base
- Be accurate and cite your sources
- Keep responses detailed but focused
- Set used_knowledge_base=True and/or used_web_search=True based on which tools you used

For each response, assess your confidence:
- "high": Sources directly answer the question
- "medium": Sources are related but not comprehensive
- "low": Little or no relevant content found""",
)


# %% [markdown]
# ## Step 4: Define the Retrieval Tool
#
# The `@agent.tool` decorator registers a function as a tool the agent can call.
# The agent decides when to use it based on the user's question.
#
# Notice how `ctx.deps` gives us access to our injected dependencies!

# %%
@wellness_agent.tool
async def search_wellness_knowledge(
    ctx: RunContext[WellnessDeps],
    query: str
) -> str:
    """
    Search the wellness knowledge base for relevant information.

    Args:
        ctx: Runtime context with injected dependencies
        query: Search query to find relevant wellness content

    Returns:
        Retrieved context as formatted string
    """
    # Access our VectorDatabase through dependency injection
    results: List[Tuple[str, float]] = ctx.deps.vector_db.search_by_text(
        query,
        k=ctx.deps.max_results
    )

    if not results:
        return "No relevant information found in the wellness knowledge base."

    # Format results with source numbers (similar to original pipeline)
    formatted_results = []
    for i, (content, score) in enumerate(results, 1):
        formatted_results.append(f"[Source {i}] (relevance: {score:.3f})\n{content}")

    return "\n\n---\n\n".join(formatted_results)


# %% [markdown]
# ## Step 4b: Define the Web Search Tool
#
# Web search provides current information not in our static knowledge base.
# This uses Tavily - a search API designed for AI agents.

# %%
@wellness_agent.tool
async def search_web(
    ctx: RunContext[WellnessDeps],
    query: str
) -> str:
    """
    Search the web for current health and wellness information.

    Use this for: recent research, current health news, topics not in the knowledge base.

    Args:
        ctx: Runtime context with injected dependencies
        query: Search query for current wellness information

    Returns:
        Web search results as formatted string
    """
    if not TAVILY_AVAILABLE:
        return "Web search is not available. Install with: uv add 'pydantic-ai-slim[tavily]'"

    if not ctx.deps.enable_web_search:
        return "Web search is disabled for this session."

    if not ctx.deps.tavily_api_key:
        return "Web search requires TAVILY_API_KEY environment variable."

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

        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"[Web Source {i}]\n"
                f"Title: {result.get('title', 'N/A')}\n"
                f"URL: {result.get('url', 'N/A')}\n"
                f"Summary: {result.get('content', 'N/A')}"
            )

        return "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        return f"Web search error: {str(e)}"


# %% [markdown]
# ## Step 5: Build the Knowledge Base
#
# We reuse our existing aimakerspace classes - this keeps the learning
# from the DIY approach while showing how to integrate with PydanticAI.

# %%
def build_vector_database(
    document_path: str = "data/HealthWellnessGuide.txt",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> VectorDatabase:
    """
    Build the vector database from source documents.

    This is identical to the original notebook - we're reusing our learning!
    """
    # Load documents
    loader = TextFileLoader(document_path)
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} document(s)")

    # Split into chunks
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_texts(documents)
    print(f"Split into {len(chunks)} chunks")

    # Build vector database (embeds all chunks)
    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(chunks))
    print(f"Built vector database with {len(vector_db.vectors)} vectors")

    return vector_db


# %% [markdown]
# ## Step 6: Run the Pipeline
#
# Putting it all together. Compare how clean this is vs. the DIY approach
# while still maintaining full control over the retrieval logic.

# %%
class PydanticAIWellnessPipeline:
    """
    Wellness Assistant using PydanticAI.

    This is the PydanticAI equivalent of RetrievalAugmentedQAPipeline.
    Key benefits:
    - Type-safe responses (WellnessResponse model)
    - Clean dependency injection (WellnessDeps)
    - Reuses existing VectorDatabase code
    - Agent decides when to search (knowledge base + web)
    """

    def __init__(
        self,
        document_path: str = "data/HealthWellnessGuide.txt",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_results: int = 4,
        enable_web_search: bool = True,
        tavily_api_key: str | None = None,
    ):
        self.document_path = document_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_results = max_results
        self.enable_web_search = enable_web_search
        self.tavily_api_key = tavily_api_key
        self.vector_db: VectorDatabase | None = None
        self.deps: WellnessDeps | None = None

    def setup(self) -> "PydanticAIWellnessPipeline":
        """Initialize the vector database and dependencies."""
        self.vector_db = build_vector_database(
            self.document_path,
            self.chunk_size,
            self.chunk_overlap,
        )
        self.deps = WellnessDeps(
            vector_db=self.vector_db,
            tavily_api_key=self.tavily_api_key,
            max_results=self.max_results,
            enable_web_search=self.enable_web_search,
        )
        return self

    async def ask(self, question: str) -> WellnessResponse:
        """
        Ask a wellness question and get a structured response.

        Args:
            question: User's health/wellness question

        Returns:
            WellnessResponse with answer, confidence, sources_used, disclaimer
        """
        if not self.deps:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")

        result = await wellness_agent.run(question, deps=self.deps)
        return result.output

    def ask_sync(self, question: str) -> WellnessResponse:
        """Synchronous version of ask()."""
        return asyncio.run(self.ask(question))


# %% [markdown]
# ## Example Usage

# %%
async def main():
    """Demo the PydanticAI wellness assistant."""

    # Get Tavily API key from environment
    tavily_key = os.getenv("TAVILY_API_KEY")
    web_search_enabled = TAVILY_AVAILABLE and tavily_key is not None

    # Initialize pipeline with web search enabled
    pipeline = PydanticAIWellnessPipeline(
        document_path="data/HealthWellnessGuide.txt",
        max_results=3,
        enable_web_search=web_search_enabled,
        tavily_api_key=tavily_key,
    )
    pipeline.setup()

    # Test questions - mix of knowledge base and web search
    questions = [
        # Knowledge base questions
        "What exercises help with lower back pain?",
        "What are some natural remedies for improving sleep quality?",
        # Web search question (current research not in static knowledge base)
        "What are the latest research findings on intermittent fasting for health?",
    ]

    print("=" * 60)
    print("PydanticAI - Wellness Assistant (Tavily Web Search)")
    print(f"Web Search: {'Enabled' if web_search_enabled else 'Disabled (set TAVILY_API_KEY)'}")
    print("=" * 60)

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)

        response = await pipeline.ask(question)

        # Structured response - type-safe access!
        print(f"Answer: {response.answer[:400]}...")
        print(f"Confidence: {response.confidence}")
        print(f"Sources Used: {response.sources_used}")
        print(f"Used Knowledge Base: {response.used_knowledge_base}")
        print(f"Used Web Search: {response.used_web_search}")
        print(f"Disclaimer: {response.disclaimer}")
        print()


# %%
if __name__ == "__main__":
    # Ensure API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        from getpass import getpass
        os.environ["OPENAI_API_KEY"] = getpass("OpenAI API Key: ")

    if not os.getenv("TAVILY_API_KEY"):
        print("Note: TAVILY_API_KEY not set. Web search will be disabled.")

    asyncio.run(main())


# %% [markdown]
# ## Comparison: DIY vs PydanticAI
#
# | Aspect | DIY (Original) | PydanticAI |
# |--------|----------------|------------|
# | **Retrieval** | Always retrieves | Agent decides when |
# | **Response Type** | `dict` | `WellnessResponse` (typed) |
# | **Dependencies** | Global/closure | Explicit injection |
# | **Tool Definition** | Manual in prompt | `@agent.tool` decorator |
# | **Type Safety** | None | Full Pydantic validation |
# | **Testability** | Harder | Easy (inject mock deps) |
# | **Vector DB** | Our VectorDatabase | Our VectorDatabase (reused!) |
#
# ## What We Kept from DIY:
# - TextFileLoader, CharacterTextSplitter (document processing)
# - VectorDatabase with cosine similarity (retrieval)
# - Same chunking parameters (1000 chars, 200 overlap)
# - Same embedding model (text-embedding-3-small)
#
# ## What PydanticAI Added:
# - Type-safe responses (catch errors early)
# - Clean dependency injection (better testing, clearer code)
# - Agentic behavior (model decides when to search)
# - Structured output validation
#
# ## When to Use Each:
# - **DIY**: Learning fundamentals, maximum control, custom metrics
# - **PydanticAI**: Production apps, type safety, cleaner architecture
