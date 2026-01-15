# Key Concepts: LLM APIs and Agentic Architecture

> **A comprehensive guide for AI Engineers** — From RAG fundamentals to production agent systems
>
> *Last updated: January 15, 2026*

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [OpenAI API Evolution](#2-openai-api-evolution)
3. [Anthropic API Architecture](#3-anthropic-api-architecture)
4. [The API Philosophy War](#4-the-api-philosophy-war)
5. [Embeddings Deep Dive](#5-embeddings-deep-dive)
6. [RAG Architecture Patterns](#6-rag-architecture-patterns)
7. [Code Execution in Production](#7-code-execution-in-production)
8. [Framework Integration](#8-framework-integration)
9. [Agent Skills: The 2026 Standard](#9-agent-skills-the-2026-standard)
10. [MCP: The Integration Layer](#10-mcp-the-integration-layer)
11. [Decision Framework](#11-decision-framework)
12. [References](#12-references)

---

## 1. The Big Picture

### The Evolution of LLM Applications (2023-2026)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLM APPLICATION MATURITY MODEL                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  2023          2024              2025              2026                     │
│    │             │                 │                 │                      │
│    ▼             ▼                 ▼                 ▼                      │
│  ┌───┐        ┌─────┐          ┌───────┐        ┌─────────┐                │
│  │LLM│   →    │ RAG │     →    │ Agent │   →    │ Skills  │                │
│  └───┘        └─────┘          └───────┘        └─────────┘                │
│    │             │                 │                 │                      │
│  Single       Retrieval        Tool Use +        Modular +                  │
│  Prompt       Augmented        Reasoning         Composable                 │
│               Generation       Loops             Capabilities               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Formula

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   RAG = Dense Vector Retrieval + In-Context Learning           │
│                                                                │
│   Agents = RAG + Tool Use + Reasoning Loop (ReAct)             │
│                                                                │
│   Skills = Agents + Progressive Disclosure + Modularity        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. OpenAI API Evolution

### Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OPENAI API TIMELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Mar 2023         Mar 2025              Aug 2025           Aug 2026         │
│     │                │                     │                  │             │
│     ▼                ▼                     ▼                  ▼             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐      ┌──────────┐        │
│  │  Chat    │    │  Responses   │    │Assistants│      │Assistants│        │
│  │Completions│   │    API       │    │Deprecated│      │ Sunset   │        │
│  └──────────┘    └──────────────┘    └──────────┘      └──────────┘        │
│       │                │                                                    │
│       │                │                                                    │
│       ▼                ▼                                                    │
│   SUPPORTED        RECOMMENDED                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Chat Completions API

The foundational conversational interface—stable, flexible, developer-controlled:

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is RAG?"}
    ]
)
print(response.choices[0].message.content)
```

**Characteristics:**
| Feature | Description |
|---------|-------------|
| **Primitive** | Message array `[{role, content}]` |
| **State** | Stateless—developer manages history |
| **Tools** | Manual implementation required |
| **Best For** | Simple chatbots, maximum control |

### Responses API

The agentic-first interface—server-managed loops, built-in tools:

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input="Search for recent news about RAG architectures",
    tools=[
        {"type": "web_search"},
        {"type": "file_search", "vector_store_ids": ["vs_abc123"]}
    ]
)
print(response.output)
```

**Characteristics:**
| Feature | Description |
|---------|-------------|
| **Primitive** | Universal `Item` object (state/text/tools) |
| **State** | Server-managed threads and memory |
| **Tools** | Built-in: `web_search`, `file_search`, `code_interpreter` |
| **Best For** | Agents, rapid prototyping, agentic RAG |

### API Status (January 2026)

| API | Status | Action |
|-----|--------|--------|
| **Chat Completions** | ✅ Supported | Use for simple/controlled workflows |
| **Responses API** | ⭐ Recommended | Use for new projects |
| **Assistants API** | ⛔ Deprecated | **Firm sunset: August 26, 2026** |

> **Warning:** The Assistants API sunset is non-negotiable. Transitioning to the Responses API or OpenAI Agents SDK is no longer optional for production builds. Plan your migration now.

---

## 3. Anthropic API Architecture

### The Unified Messages API

Unlike OpenAI's bifurcation, Anthropic maintains **one API** with opt-in capabilities:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ANTHROPIC: UNIFIED API MODEL                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌──────────────────┐                                │
│                         │   Messages API   │                                │
│                         │   /v1/messages   │                                │
│                         └────────┬─────────┘                                │
│                                  │                                          │
│              ┌───────────────────┼───────────────────┐                      │
│              │                   │                   │                      │
│              ▼                   ▼                   ▼                      │
│      ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                │
│      │  Tool Use    │   │    Code      │   │    Files     │                │
│      │  (tools=[])  │   │  Execution   │   │     API      │                │
│      └──────────────┘   └──────────────┘   └──────────────┘                │
│              │                   │                   │                      │
│              └───────────────────┼───────────────────┘                      │
│                                  │                                          │
│                                  ▼                                          │
│                         ┌──────────────────┐                                │
│                         │  MCP Connector   │                                │
│                         │ (Open Standard)  │                                │
│                         └──────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Basic Usage

```python
import anthropic

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is RAG?"}
    ]
)
print(message.content[0].text)
```

### With Tool Use

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    tools=[
        {
            "name": "search_knowledge_base",
            "description": "Search the wellness knowledge base",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    ],
    tool_choice={"type": "auto"},
    messages=[{"role": "user", "content": "What helps with sleep?"}]
)

# Handle tool_use blocks
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}, Input: {block.input}")
```

### Agentic Capabilities (2025-2026)

| Capability | Description | Use Case |
|------------|-------------|----------|
| **Code Execution** | Server-side Python/Bash sandbox | Data analysis, calculations |
| **Files API** | Persistent storage via `file_id` | Large document handling |
| **MCP Connector** | Connect to external systems | Database, API integrations |
| **Extended Caching** | Up to 1-hour prompt caching | Cost optimization |

---

## 4. The API Philosophy War

### Managed Autonomy vs Developer Control

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE TWO PHILOSOPHIES OF 2025-2026                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│         OPENAI                              ANTHROPIC                       │
│    "Managed Autonomy"                   "Developer Control"                 │
│                                                                             │
│    ┌─────────────────┐                  ┌─────────────────┐                │
│    │   User Query    │                  │   User Query    │                │
│    └────────┬────────┘                  └────────┬────────┘                │
│             │                                    │                          │
│             ▼                                    ▼                          │
│    ┌─────────────────┐                  ┌─────────────────┐                │
│    │  Responses API  │                  │  Messages API   │                │
│    │   ┌─────────┐   │                  └────────┬────────┘                │
│    │   │ Think   │   │                           │                          │
│    │   └────┬────┘   │                           ▼                          │
│    │        │        │                  ┌─────────────────┐                │
│    │   ┌────▼────┐   │                  │  Your Code      │                │
│    │   │ Tool    │   │          ┌──────►│  (The Loop)     │◄──────┐        │
│    │   └────┬────┘   │          │       └────────┬────────┘       │        │
│    │        │        │          │                │                │        │
│    │   ┌────▼────┐   │          │                ▼                │        │
│    │   │ Result  │   │          │       ┌─────────────────┐       │        │
│    │   └────┬────┘   │          │       │   Tool Call     │───────┘        │
│    │        │        │          │       └────────┬────────┘                │
│    │   ┌────▼────┐   │          │                │                          │
│    │   │ Respond │   │          └────────────────┘                          │
│    │   └─────────┘   │                                                      │
│    └────────┬────────┘                                                      │
│             │                                    │                          │
│             ▼                                    ▼                          │
│    ┌─────────────────┐                  ┌─────────────────┐                │
│    │  Final Answer   │                  │  Final Answer   │                │
│    │  (Black Box)    │                  │  (Full Audit)   │                │
│    └─────────────────┘                  └─────────────────┘                │
│                                                                             │
│    Best for:                            Best for:                           │
│    • Rapid prototyping                  • Enterprise compliance             │
│    • Startups                           • Audit requirements                │
│    • Less code                          • Full visibility                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Architectural Comparison

| Aspect | OpenAI (Responses) | Anthropic (Messages) |
|--------|-------------------|---------------------|
| **Core Primitive** | Items (universal object) | Messages (structured turns) |
| **Loop Management** | API-managed | Developer-managed |
| **Tool Execution** | Server-side hosted | Hybrid: native + MCP |
| **State/Memory** | Persistent Threads | Explicit via Files API |
| **Integrations** | Proprietary built-ins | MCP (Open Standard) |
| **Transparency** | Final result only | Full intermediate visibility |

---

## 5. Embeddings Deep Dive

### What Are Embeddings?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EMBEDDINGS: TEXT → VECTORS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   "The cat sat on the mat"                                                  │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────────┐                                                   │
│   │  Embedding Model    │                                                   │
│   │  (text-embedding-   │                                                   │
│   │   3-small)          │                                                   │
│   └──────────┬──────────┘                                                   │
│              │                                                              │
│              ▼                                                              │
│   [0.023, -0.041, 0.089, ..., 0.012]  ← 1,536 dimensions                    │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  SEMANTIC SPACE                                                     │   │
│   │                                                                     │   │
│   │           "cat on mat" •                                            │   │
│   │                          \                                          │   │
│   │                           \  High Similarity                        │   │
│   │                            \                                        │   │
│   │     "feline resting" •─────•  "kitten sleeping"                     │   │
│   │                                                                     │   │
│   │                                                                     │   │
│   │              "car engine" •              Low Similarity             │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### OpenAI Embedding Models (2026)

| Model | Dimensions | Context | Best For |
|-------|------------|---------|----------|
| `text-embedding-3-small` | 1,536 (default) | 8,192 tokens | Cost-effective, general use |
| `text-embedding-3-large` | 3,072 (default) | 8,192 tokens | Maximum accuracy |

### Dimension Reduction (Matryoshka)

OpenAI's `text-embedding-3` models use **Matryoshka Representation Learning**—embeddings can be shortened by truncating from the end while preserving semantic properties:

```python
from openai import OpenAI

client = OpenAI()

# Request smaller dimensions directly
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="What helps with sleep?",
    dimensions=512  # Reduced from 1,536
)

embedding = response.data[0].embedding  # len() = 512
```

**Trade-offs:**
| Dimensions | Storage | Speed | Accuracy |
|------------|---------|-------|----------|
| 1,536 | 100% | Baseline | Best |
| 512 | 33% | ~3x faster | Good |
| 256 | 17% | ~6x faster | Acceptable |

### Similarity Metrics

```python
import numpy as np

def cosine_similarity(a, b):
    """Measures angle between vectors (most common)"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    """Measures straight-line distance"""
    return -np.linalg.norm(np.array(a) - np.array(b))  # Negative: closer = higher

def dot_product(a, b):
    """Measures magnitude and direction"""
    return np.dot(a, b)
```

---

## 6. RAG Architecture Patterns

### The 5-Step RAG Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THE RAG PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        INDEXING PHASE (Offline)                      │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                     │    │
│  │   Documents        Chunks           Embeddings        Vector DB     │    │
│  │   ┌──────┐        ┌──────┐         ┌──────┐          ┌──────┐      │    │
│  │   │ .txt │   →    │chunk1│    →    │[0.2,]│     →    │  DB  │      │    │
│  │   │ .pdf │        │chunk2│         │[0.1,]│          │      │      │    │
│  │   │ .md  │        │chunk3│         │[0.3,]│          │      │      │    │
│  │   └──────┘        └──────┘         └──────┘          └──────┘      │    │
│  │       │               │                │                  │        │    │
│  │       └───────────────┴────────────────┴──────────────────┘        │    │
│  │                    STEP 1: Create Database                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        QUERY PHASE (Online)                          │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                     │    │
│  │   User Query        Query Embedding      Similarity Search          │    │
│  │   ┌──────────┐      ┌──────────┐        ┌──────────────────┐       │    │
│  │   │"What     │  →   │ [0.25,   │   →    │ chunk2: 0.95     │       │    │
│  │   │ helps    │      │  0.12,   │        │ chunk1: 0.82     │       │    │
│  │   │ sleep?"  │      │  ...]    │        │ chunk3: 0.71     │       │    │
│  │   └──────────┘      └──────────┘        └────────┬─────────┘       │    │
│  │        │                 │                       │                 │    │
│  │        └─────────────────┘                       │                 │    │
│  │           STEP 2 & 3: Embed & Search             │                 │    │
│  │                                                  │                 │    │
│  │                                                  ▼                 │    │
│  │   Augmented Prompt                      Generated Response         │    │
│  │   ┌────────────────────┐               ┌────────────────────┐     │    │
│  │   │ Context: {chunk2}  │       →       │ "Based on the      │     │    │
│  │   │ Question: "What    │               │  wellness guide,   │     │    │
│  │   │  helps sleep?"     │               │  sleep hygiene..." │     │    │
│  │   └────────────────────┘               └────────────────────┘     │    │
│  │           │                                      │                │    │
│  │           └──────────────────────────────────────┘                │    │
│  │                    STEP 4 & 5: Augment & Generate                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### RAG Evolution: Monolithic to Agentic

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **2-Step RAG** | Retrieve → Generate (single LLM call) | Simple queries, fast response |
| **Agentic RAG** | Agent decides when/what to retrieve | Complex queries, multi-hop reasoning |
| **Modular RAG** | Skills-based, domain-specific retrieval | Enterprise, multi-domain |

### Chunking Strategies

```python
# Session 2 Default
splitter = CharacterTextSplitter(
    chunk_size=1000,     # Characters per chunk
    chunk_overlap=200    # Overlap for context continuity
)

# Experiment with different configurations
configs = [
    {"chunk_size": 500,  "chunk_overlap": 50},   # Fine-grained
    {"chunk_size": 1000, "chunk_overlap": 200},  # Balanced (default)
    {"chunk_size": 1500, "chunk_overlap": 400},  # Coarse, high overlap
]
```

---

## 7. Code Execution in Production

> NOTE:  if these concepts feel heavy at this point come back to them in a couple of weeks.  A little deep for this point of the course but I found this additional content helps set the stage when factoring in the essential aspects of building production solutions that are both safe and secure.

### The Security Maturity Curve

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CODE EXECUTION SECURITY LEVELS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Level 1              Level 2               Level 3                        │
│   LOCAL               SANDBOXED             AUDITABLE                       │
│                                                                             │
│   ┌─────────┐         ┌─────────┐          ┌─────────┐                     │
│   │ Your    │         │ Isolated│          │ Logged  │                     │
│   │ Machine │         │Container│          │Sandbox  │                     │
│   └─────────┘         └─────────┘          └─────────┘                     │
│       │                   │                    │                            │
│       ▼                   ▼                    ▼                            │
│   ┌─────────┐         ┌─────────┐          ┌─────────┐                     │
│   │ Risk:   │         │ Risk:   │          │ Risk:   │                     │
│   │  HIGH   │         │  LOW    │          │ MINIMAL │                     │
│   │         │         │         │          │         │                     │
│   │ LLM can │         │ No disk │          │ Full    │                     │
│   │ access  │         │ No net  │          │ audit   │                     │
│   │ files   │         │         │          │ trail   │                     │
│   └─────────┘         └─────────┘          └─────────┘                     │
│                                                                             │
│   Session 1            Session 2+           Enterprise                      │
│   (Learning)           (Production)         (Compliance)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Anthropic Code Execution Tool

```python
import anthropic

client = anthropic.Anthropic()

# Sample RAG performance data
data_payload = """
Date,Query_Latency_ms,Embed_Similarity
2026-01-01,120,0.85
2026-01-01,145,0.78
2026-01-02,110,0.92
2026-01-02,95,0.88
"""

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    betas=["code-execution-2025-05-22"],  # Required for code execution
    tools=[
        {
            "type": "code_execution",
            "name": "python_interpreter"
        }
    ],
    messages=[
        {
            "role": "user",
            "content": f"Analyze this data and calculate average latency per day:\n{data_payload}"
        }
    ]
)

# Full visibility into execution
for block in response.content:
    if block.type == "text":
        print(f"Reasoning: {block.text}")
    elif block.type == "tool_use":
        print(f"Code Executed:\n{block.input['code']}")
    elif block.type == "tool_result":
        print(f"Output: {block.content}")
```

### Comparison

| Feature | Local (`exec()`) | Anthropic Native |
|---------|------------------|------------------|
| **Security** | High risk—filesystem access | Isolated sandbox |
| **Libraries** | Full environment | Pre-installed (Pandas, NumPy, etc.) |
| **Debugging** | Local stack trace | `stderr` in response |
| **Auditability** | Manual logging | Built-in visibility |
| **Cost** | Free (CPU) | ~$0.05/hour |

> **Important: Ephemeral Sandbox**
>
> Anthropic's native code execution runs in an **ephemeral sandbox**. Any files created during a session are **deleted when the session ends** unless explicitly saved back to a persistent MCP-connected drive or returned via the Files API. Design your workflows accordingly—extract results before session termination.

---

## 8. Framework Integration

### How Frameworks Wrap Provider APIs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FRAMEWORK ABSTRACTION LAYER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        YOUR APPLICATION                                     │
│                              │                                              │
│                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    FRAMEWORK LAYER                                  │   │
│   │                                                                     │   │
│   │    ┌──────────────┐              ┌──────────────┐                  │   │
│   │    │  LangChain   │              │  PydanticAI  │                  │   │
│   │    │  ChatOpenAI  │              │    Agent     │                  │   │
│   │    └──────┬───────┘              └──────┬───────┘                  │   │
│   │           │                             │                          │   │
│   └───────────┼─────────────────────────────┼──────────────────────────┘   │
│               │                             │                              │
│               ▼                             ▼                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      SDK LAYER                                      │   │
│   │                                                                     │   │
│   │    ┌──────────────┐              ┌──────────────┐                  │   │
│   │    │    openai    │              │  anthropic   │                  │   │
│   │    │    Python    │              │    Python    │                  │   │
│   │    └──────┬───────┘              └──────┬───────┘                  │   │
│   │           │                             │                          │   │
│   └───────────┼─────────────────────────────┼──────────────────────────┘   │
│               │                             │                              │
│               ▼                             ▼                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       API LAYER                                     │   │
│   │                                                                     │   │
│   │    ┌──────────────┐              ┌──────────────┐                  │   │
│   │    │   OpenAI     │              │  Anthropic   │                  │   │
│   │    │  /responses  │              │  /messages   │                  │   │
│   │    │  /chat/...   │              │              │                  │   │
│   │    └──────────────┘              └──────────────┘                  │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LangChain

```python
from langchain_openai import ChatOpenAI

# Default: Chat Completions API
model = ChatOpenAI(model="gpt-4o-mini")

# Opt-in: Responses API (langchain-openai >= 0.4.5)
model = ChatOpenAI(model="gpt-4o-mini", use_responses_api=True)

# With built-in tools via Responses API
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini",
    use_responses_api=True
)
# Now supports web_search, file_search, etc.
```

### PydanticAI

```python
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Direct OpenAI client injection
client = AsyncOpenAI(max_retries=3)
model = OpenAIChatModel('gpt-5', provider=OpenAIProvider(openai_client=client))
agent = Agent(model)

# With type-safe responses
from pydantic import BaseModel

class WellnessResponse(BaseModel):
    advice: str
    confidence: float
    sources: list[str]

agent = Agent(model, output_type=WellnessResponse)
```

### Benefits of Framework Abstraction

| Benefit | Description |
|---------|-------------|
| **Provider-agnostic** | Swap OpenAI ↔ Anthropic with minimal changes |
| **Standardized messages** | Consistent format across providers |
| **Middleware support** | Tracing, caching, rate limiting |
| **Structured output** | Type-safe responses (PydanticAI) |

---

## 9. Agent Skills: The 2026 Standard

### What Are Agent Skills?

Agent Skills is an open standard (launched December 18, 2025) that treats capabilities as **version-controlled folders** rather than prompts or hard-coded functions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGENT SKILLS ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   .github/skills/                    # Anthropic (Claude SDK)               │
│   .codex/skills/                     # OpenAI (Agents SDK)                  │
│   │                                                                         │
│   └── wellness-assistant/                                                   │
│       ├── SKILL.md                   # Metadata + Instructions              │
│       ├── scripts/                                                          │
│       │   ├── search_knowledge.py    # Retrieval logic                      │
│       │   └── calculate_metrics.py   # Analysis logic                       │
│       └── references/                                                       │
│           ├── HealthWellnessGuide.txt                                       │
│           └── sleep_research.pdf                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The `SKILL.md` File

```markdown
---
name: wellness-assistant
description: Use this skill when the user asks about health, nutrition, sleep, stress management, or exercise recommendations.
---

# Instructions

1. Search the knowledge base using `scripts/search_knowledge.py`
2. If the query involves metrics or comparisons, run `scripts/calculate_metrics.py`
3. Reference `references/HealthWellnessGuide.txt` for authoritative answers
4. Format responses with actionable advice and source citations
```

### Progressive Disclosure: 3-Tier Loading

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PROGRESSIVE DISCLOSURE MODEL                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TIER 1: METADATA                    ~40 tokens per skill                  │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  Loaded at session start                                             │  │
│   │  Agent sees: name + description only                                 │  │
│   │                                                                      │  │
│   │  Skills: [wellness-assistant, drug-interaction, clinical-trials]     │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              │ User: "What helps with sleep?"               │
│                              ▼                                              │
│   TIER 2: FULL INSTRUCTIONS           ~1k-5k tokens                         │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  Loaded when task matches description                                │  │
│   │  Agent reads: complete SKILL.md instructions                         │  │
│   │                                                                      │  │
│   │  → wellness-assistant instructions injected into context             │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              │ Agent reaches Step 1                         │
│                              ▼                                              │
│   TIER 3: JUST-IN-TIME RESOURCES      Variable                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  Loaded only when instruction step requires it                       │  │
│   │                                                                      │  │
│   │  → scripts/search_knowledge.py executed                              │  │
│   │  → references/HealthWellnessGuide.txt opened                         │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   BENEFIT: 100 skills × 40 tokens = 4,000 tokens (not 500,000!)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Industry Adoption

The Agent Skills standard has been adopted by:
- **Anthropic**: Claude.ai, Claude Code, Claude Agent SDK
- **OpenAI**: Codex, Agents SDK
- **Microsoft**: GitHub, VS Code
- **Others**: Cursor, Figma, Atlassian, Notion, Stripe

> **Directory Compatibility Note**
>
> The `.github/skills/` directory is the vendor-neutral standard for 2026. GitHub Copilot (Agent Mode) also supports `.claude/skills/` for backward compatibility, but `.github/skills/` is recommended for cross-platform portability.

### Student Checklist: Converting RAG to Agent Skills

- [ ] **Standardize:** Move RAG retrieval logic into `.github/skills/<skill-name>/scripts/`
- [ ] **Describe:** Ensure `SKILL.md` description is "trigger-heavy" (e.g., "Use this when the user asks about...")
- [ ] **Audit:** Use Anthropic's intermediate `tool_use` blocks to log exactly how the agent queries the RAG database
- [ ] **Test:** Verify skill triggers correctly by testing edge-case queries
- [ ] **Version:** Commit skill folders to git for reproducibility

### From Session 2 to Session 3

| Session 2 (Monolithic RAG) | Session 3 (Modular Skills) |
|----------------------------|----------------------------|
| One prompt + one vector search | Domain-specific skill folders |
| All context loaded upfront | Progressive disclosure |
| Single retrieval strategy | Skill-specific retrieval + scripts |
| Hard to maintain/extend | Version-controlled, testable |

---

## 10. MCP: The Integration Layer

### What Is MCP?

**Model Context Protocol** (MCP) is an open standard for connecting AI systems to external tools and data sources. Donated to the Linux Foundation (December 2025), it's now the industry standard for tool connectivity.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MCP: MODEL CONTEXT PROTOCOL                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌──────────────┐                                    │
│                         │  LLM Agent   │                                    │
│                         └──────┬───────┘                                    │
│                                │                                            │
│                                ▼                                            │
│                    ┌───────────────────────┐                                │
│                    │     MCP Protocol      │                                │
│                    │   (Open Standard)     │                                │
│                    └───────────┬───────────┘                                │
│                                │                                            │
│          ┌─────────────────────┼─────────────────────┐                      │
│          │                     │                     │                      │
│          ▼                     ▼                     ▼                      │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                │
│   │  Database   │      │    APIs     │      │ File System │                │
│   │   Server    │      │   Server    │      │   Server    │                │
│   └─────────────┘      └─────────────┘      └─────────────┘                │
│          │                     │                     │                      │
│          ▼                     ▼                     ▼                      │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                │
│   │  PostgreSQL │      │   Stripe    │      │   Local     │                │
│   │   Neo4j     │      │   GitHub    │      │   Docs      │                │
│   │   etc.      │      │   etc.      │      │             │                │
│   └─────────────┘      └─────────────┘      └─────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MCP vs Proprietary Tools

| Aspect | MCP (Anthropic) | Built-in Tools (OpenAI) |
|--------|-----------------|------------------------|
| **Standard** | Open (Linux Foundation) | Proprietary |
| **Hosting** | Self-hosted or cloud | OpenAI-hosted |
| **Customization** | Full control | Limited to API parameters |
| **Data Privacy** | Data stays in your infra | Data sent to OpenAI |

### LangChain MCP Integration

```python
from langchain_mcp_adapters import MultiServerMCPClient

# Connect to MCP servers
mcp_client = MultiServerMCPClient({
    "database": {"command": "mcp-server-postgres", "args": ["--connection-string", "..."]},
    "filesystem": {"command": "mcp-server-filesystem", "args": ["--root", "/data"]}
})

# Get tools for agent
tools = mcp_client.get_tools()
```

---

## 11. Decision Framework

### Choosing the Right API

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DECISION FLOWCHART                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        What are you building?                               │
│                               │                                             │
│               ┌───────────────┼───────────────┐                             │
│               ▼               ▼               ▼                             │
│        ┌──────────┐    ┌──────────┐    ┌──────────┐                        │
│        │  Simple  │    │   RAG    │    │  Agent   │                        │
│        │ Chatbot  │    │   App    │    │  System  │                        │
│        └────┬─────┘    └────┬─────┘    └────┬─────┘                        │
│             │               │               │                               │
│             ▼               │               │                               │
│      Chat Completions       │               │                               │
│      or Messages API        │               │                               │
│                             ▼               │                               │
│                    Need audit trail?        │                               │
│                      │         │            │                               │
│                     YES        NO           │                               │
│                      │         │            │                               │
│                      ▼         ▼            │                               │
│                 Anthropic  OpenAI           │                               │
│                 Messages   Responses        │                               │
│                                             ▼                               │
│                                   Need full control?                        │
│                                     │         │                             │
│                                    YES        NO                            │
│                                     │         │                             │
│                                     ▼         ▼                             │
│                               Anthropic    OpenAI                           │
│                               + Claude     + Agent                          │
│                               Agent SDK    SDK                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Quick Reference

| Use Case | Recommended Stack |
|----------|-------------------|
| **Learning/Prototyping** | Chat Completions + DIY RAG |
| **Simple RAG** | Either API + LangChain/PydanticAI |
| **Agentic RAG** | Responses API + built-in tools |
| **Enterprise RAG** | Messages API + MCP + Agent Skills |
| **Maximum Control** | Messages API + Claude Agent SDK |
| **Maximum Speed** | Responses API + OpenAI Agent SDK |

### For This Course

| Session | Recommended Approach |
|---------|---------------------|
| **Session 2 (RAG)** | Chat Completions—learn the fundamentals |
| **Session 3 (Agents)** | Add tool use—understand the loop |
| **Production** | Evaluate: Responses API or Messages + Skills |

---

## 12. References

### Official Documentation

#### OpenAI
- [Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat) — Core API documentation
- [Responses API Migration Guide](https://platform.openai.com/docs/guides/migrate-to-responses) — Migration from Chat Completions
- [Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) — Embedding models and dimension reduction
- [API Deprecations](https://platform.openai.com/docs/deprecations) — Assistants API sunset timeline

#### Anthropic
- [Messages API Reference](https://platform.claude.com/docs/en/api/messages/create) — Core API documentation
- [Tool Use Guide](https://platform.claude.com/docs/en/build-with-claude/tool-use) — Implementing tool calling
- [Agent Capabilities Announcement](https://www.anthropic.com/news/agent-capabilities-api) — Code execution, Files API
- [Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — Building agents

### Standards & Protocols

#### Agent Skills
- [Agent Skills Specification](https://agentskills.io/specification) — Official specification
- [GitHub: anthropics/skills](https://github.com/anthropics/skills/blob/main/spec/agent-skills-spec.md) — Source specification
- [Anthropic Engineering Blog](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — Design rationale

#### Model Context Protocol (MCP)
- [MCP Specification](https://modelcontextprotocol.io/specification/2025-11-25/index) — Protocol specification
- [MCP Example Servers](https://modelcontextprotocol.io/examples) — Reference implementations
- [LangChain MCP Adapters](https://docs.langchain.com/oss/javascript/langchain/mcp) — Framework integration

### Framework Documentation

#### LangChain
- [ChatOpenAI Integration](https://docs.langchain.com/oss/python/integrations/chat/openai) — OpenAI wrapper
- [Retrieval Guide](https://docs.langchain.com/oss/python/langchain/retrieval) — RAG patterns
- [Build a RAG Agent](https://docs.langchain.com/oss/python/langchain/rag) — Complete tutorial

#### PydanticAI
- [OpenAI Models](https://ai.pydantic.dev/models/openai/) — Provider integration
- [Agent Documentation](https://ai.pydantic.dev/) — Framework overview

### Academic Foundations

| Paper | Year | Key Concept |
|-------|------|-------------|
| [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | 2020 | Original RAG architecture |
| [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | 2020 | In-context learning |
| [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | 2022 | Step-by-step reasoning |
| [ReAct: Reasoning and Acting](https://arxiv.org/abs/2210.03629) | 2022 | Agent loop pattern |
| [Sentence-BERT](https://arxiv.org/abs/1908.10084) | 2019 | Sentence embeddings |

---

*This document are the result of my ramblings with Claude Code, and not part of the official curriculum.  Consider this as a milestone when Claude started aligning with MY vibe.*

*Last updated: January 15, 2026*
