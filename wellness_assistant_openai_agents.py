# %% [markdown]
# # Personal Wellness Assistant - OpenAI Agents SDK Implementation
#
# This is an alternate implementation of the RAG-based Wellness Assistant using the
# **OpenAI Agents SDK** (`openai-agents`). This approach introduces agentic patterns
# that will be covered in depth in Session 3: The Agent Loop.
#
# ## Key Differences from DIY Implementation:
# - Uses OpenAI's managed vector stores (automatic chunking, embedding, retrieval)
# - Agent decides when/how to search the knowledge base
# - Built-in tracing and observability
# - Simpler code, more abstraction
#
# ## Prerequisites:
# ```bash
# uv add openai-agents
# ```
#
# ## References:
# - [OpenAI Agents SDK Documentation](https://github.com/openai/openai-agents-python)
# - [FileSearchTool API](https://github.com/openai/openai-agents-python/blob/main/docs/tools.md)

# %%
import asyncio
import os
from pathlib import Path

from openai import OpenAI

# Note: Install with `uv add openai-agents`
try:
    from agents import Agent, Runner, FileSearchTool, WebSearchTool, trace
except ImportError:
    raise ImportError(
        "OpenAI Agents SDK not installed. Run: uv add openai-agents"
    )

# %% [markdown]
# ## Step 1: Setup OpenAI Client and Load Document
#
# Unlike the DIY approach where we manually chunk and embed, the OpenAI Agents SDK
# handles this automatically via Vector Stores.

# %%
def setup_vector_store(client: OpenAI, file_path: str, store_name: str = "wellness-kb") -> str:
    """
    Create a vector store and upload a file for semantic search.

    OpenAI automatically:
    - Parses the file
    - Chunks it (default: 800 tokens, 400 overlap)
    - Embeds with text-embedding-3-large (256 dimensions)
    - Indexes for hybrid search (semantic + keyword)

    Args:
        client: OpenAI client
        file_path: Path to the document to index
        store_name: Name for the vector store

    Returns:
        vector_store_id: ID of the created vector store
    """
    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    # Upload file
    with open(file_path, "rb") as f:
        file_upload = client.files.create(file=f, purpose="assistants")

    # Create vector store
    vector_store = client.vector_stores.create(name=store_name)

    # Add file to vector store and wait for processing
    client.vector_stores.files.create_and_poll(
        vector_store_id=vector_store.id,
        file_id=file_upload.id
    )

    print(f"Created vector store: {vector_store.id}")
    print(f"Indexed file: {file_path}")

    return vector_store.id


# %% [markdown]
# ## Step 2: Create the Wellness Assistant Agent
#
# The agent is configured with:
# - **Instructions**: System prompt defining behavior (similar to RAG_SYSTEM_TEMPLATE)
# - **FileSearchTool**: Retrieves from our wellness knowledge base
# - **max_num_results**: Controls how many chunks to retrieve (similar to `k` parameter)

# %%
def create_wellness_agent(
    vector_store_id: str,
    max_results: int = 4,
    enable_web_search: bool = True,
) -> Agent:
    """
    Create a wellness assistant agent with file search and optional web search.

    Args:
        vector_store_id: ID of the vector store containing wellness documents
        max_results: Maximum number of chunks to retrieve per search
        enable_web_search: Whether to enable web search for current information

    Returns:
        Configured Agent instance
    """
    instructions = """You are a helpful personal wellness assistant that answers health and wellness questions.

You have access to two information sources:
1. **Knowledge Base (FileSearchTool)**: Your primary source - a curated wellness guide
2. **Web Search (WebSearchTool)**: For current research, news, or topics not in the knowledge base

Instructions:
- ALWAYS search the knowledge base first for wellness questions
- Use web search for: current research/studies, recent health news, topics not in the knowledge base
- Be accurate and cite your sources (knowledge base vs web)
- Keep responses detailed but focused
- Include a gentle reminder that users should consult healthcare professionals for medical advice
- Clearly distinguish between information from your curated knowledge base vs web sources

When answering, explain which source(s) you used and why."""

    tools = [
        FileSearchTool(
            max_num_results=max_results,
            vector_store_ids=[vector_store_id],
            include_search_results=True,
        )
    ]

    if enable_web_search:
        tools.append(WebSearchTool())

    agent = Agent(
        name="Wellness Assistant",
        instructions=instructions,
        model="gpt-4.1-mini",
        tools=tools,
    )

    return agent


# %% [markdown]
# ## Step 3: Run the Assistant
#
# The `Runner` class handles the agent loop:
# 1. Send user query to agent
# 2. Agent decides to use FileSearchTool
# 3. Tool retrieves relevant chunks from vector store
# 4. Agent synthesizes response from retrieved content

# %%
async def ask_wellness_question(agent: Agent, question: str) -> dict:
    """
    Ask a question to the wellness assistant.

    Args:
        agent: Configured wellness assistant agent
        question: User's health/wellness question

    Returns:
        Dictionary with response and metadata
    """
    with trace("Wellness Assistant Query"):
        result = await Runner.run(agent, question)

        return {
            "question": question,
            "response": result.final_output,
            "items": [str(item) for item in result.new_items],
        }


# %% [markdown]
# ## Step 4: Main Pipeline
#
# Putting it all together - this mirrors the original `RetrievalAugmentedQAPipeline`
# but with OpenAI's managed infrastructure.

# %%
class OpenAIAgentsWellnessPipeline:
    """
    Wellness Assistant using OpenAI Agents SDK.

    This is the OpenAI Agents equivalent of RetrievalAugmentedQAPipeline.
    Key differences:
    - Vector store managed by OpenAI (no manual chunking/embedding)
    - Agent decides when to search (agentic behavior)
    - Web search for current information
    - Built-in tracing for observability
    """

    def __init__(
        self,
        document_path: str = "data/HealthWellnessGuide.txt",
        max_results: int = 4,
        enable_web_search: bool = True,
    ):
        self.client = OpenAI()
        self.document_path = document_path
        self.max_results = max_results
        self.enable_web_search = enable_web_search
        self.vector_store_id = None
        self.agent = None

    def setup(self) -> "OpenAIAgentsWellnessPipeline":
        """Initialize vector store and agent."""
        self.vector_store_id = setup_vector_store(
            self.client,
            self.document_path,
            store_name="wellness-assistant-kb"
        )
        self.agent = create_wellness_agent(
            self.vector_store_id,
            max_results=self.max_results,
            enable_web_search=self.enable_web_search,
        )
        return self

    async def ask(self, question: str) -> dict:
        """Ask a wellness question."""
        if not self.agent:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")
        return await ask_wellness_question(self.agent, question)

    def ask_sync(self, question: str) -> dict:
        """Synchronous version of ask()."""
        return asyncio.run(self.ask(question))

    def cleanup(self):
        """Delete the vector store (optional cleanup)."""
        if self.vector_store_id:
            self.client.vector_stores.delete(self.vector_store_id)
            print(f"Deleted vector store: {self.vector_store_id}")


# %% [markdown]
# ## Example Usage

# %%
async def main():
    """Demo the OpenAI Agents wellness assistant."""

    # Initialize pipeline
    pipeline = OpenAIAgentsWellnessPipeline(
        document_path="data/HealthWellnessGuide.txt",
        max_results=3
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
    print("OpenAI Agents SDK - Wellness Assistant")
    print("=" * 60)

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)

        result = await pipeline.ask(question)
        print(f"Response: {result['response'][:500]}...")
        print()

    # Optional: cleanup vector store
    # pipeline.cleanup()


# %%
if __name__ == "__main__":
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        from getpass import getpass
        os.environ["OPENAI_API_KEY"] = getpass("OpenAI API Key: ")

    asyncio.run(main())

# %% [markdown]
# ## Comparison: DIY vs OpenAI Agents SDK
#
# | Aspect | DIY (Original) | OpenAI Agents SDK |
# |--------|----------------|-------------------|
# | **Chunking** | Manual (`CharacterTextSplitter`) | Automatic (800 tokens, 400 overlap) |
# | **Embedding** | `text-embedding-3-small` (1536d) | `text-embedding-3-large` (256d) |
# | **Search** | Cosine similarity only | Hybrid (semantic + keyword) |
# | **Retrieval** | Always retrieves | Agent decides when to search |
# | **Code Lines** | ~200 | ~50 |
# | **Control** | Full control | Abstracted |
# | **Cost** | Pay per embedding call | Pay per vector store + agent run |
#
# ## When to Use Each:
# - **DIY**: Learning, full control, cost optimization, custom metrics
# - **OpenAI Agents**: Rapid prototyping, production apps, agentic workflows
