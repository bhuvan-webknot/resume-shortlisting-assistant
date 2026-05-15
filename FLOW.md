# AI Resume Shortlister — Flow Documentation

## Overview

The **AI Resume Shortlister** is a CLI tool that screens resumes against a job description. It uses **LangGraph** to orchestrate a multi-step pipeline: extract skills from a JD, retrieve relevant resumes via RAG, score candidates, and produce a ranked shortlist.

---

## Architecture at a Glance

```
                         ┌─────────────────┐
                         │   main.py        │
                         │  (CLI Entry)     │
                         └────────┬────────┘
                                  │
                                  ▼
                   ┌───────────────────────────┐
                   │    LangGraph StateGraph    │
                   │  (agent_graph.py)          │
                   │                            │
                   │ classify ──► extract_skills│
                   │                  │         │
                   │                  ▼         │
                   │              decide        │
                   │              /    \        │
                   │        retrieve  direct    │
                   │             |       |      │
                   │             ▼       ▼      │
                   │           respond          │
                   │              │             │
                   │              ▼             │
                   │             END            │
                   └───────────┬───────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
            rag_system.py          Google Gemini
            (ChromaDB +            (LLM for skills
             embeddings)            & summaries)
```

---

## LangGraph for LangChain Users

If you know **LangChain** but not **LangGraph**, here's the mapping:

| LangChain Concept | LangGraph Equivalent | In This Project |
|---|---|---|
| `Chain` (linear pipeline) | `StateGraph` (graph of nodes) | `agent_graph.py:build_graph()` |
| `RunnableSequence` (a \| b \| c) | `add_edge("a", "b")` | Edges connect nodes into a flow |
| `RunnableBranch` (if/else routing) | `add_conditional_edges()` | `decide` node routes to `retrieve` or `respond` |
| `chain.invoke(input)` | `graph.invoke(state)` | `main.py:33` — passes a typed state dict |
| State passing via dict | Typed `AgentState` (a `TypedDict`) | Shared state object: `agent_graph.py:17` |

**Key difference:** LangGraph models workflows as a **directed graph** instead of a linear chain. Each node is a function that reads and mutates a shared typed state. Edges define the order of execution. This makes branching, looping, and conditional routing explicit.

---

## Data Models (`schemas.py`)

| Model | Fields | Purpose |
|---|---|---|
| `IntentClassification` | `intent`, `confidence`, `reasoning` | Classifies the user's request intent |
| `DecisionOutput` | `action`, `reasoning` | Routes to next node |
| `RankedCandidate` | `candidate_name`, `match_score`, `matched_skills`, `missing_skills`, `summary` | One scored candidate |
| `FinalResponse` | `jd_summary`, `total_resumes_processed`, `shortlisted_candidates`, `shortlist_criteria` | Final structured output |

## The Agent State (`AgentState`)

This is the central data object that every node reads and writes:

```python
class AgentState(TypedDict):
    jd_content: str              # Raw JD text (set in main.py, read by extract/retrieve)
    resume_dir: str              # Path to resume directory (set in main.py)
    jd_skills: List[str]         # Extracted skills (written by extract_skills, read by respond)
    intent: IntentClassification # Classified intent (written by classify, read by decide)
    decision: DecisionOutput     # Routing decision (written by decide, read by should_retrieve)
    retrieved_docs: List[str]    # Retrieved resume texts (written by retrieve, read by respond)
    ranked_candidates: List[...] # Scored candidates (written by respond)
    final_response: FinalResponse# Final output (written by respond, read by main.py)
```

Data flows through the graph like a baton being passed between nodes, with each node adding or updating its piece of the state.

---

## Component Deep Dive

### 1. `main.py` — CLI Entry Point

**Role:** Reads CLI arguments, loads the JD file, invokes the graph, and prints results.

```python
graph = build_graph()                                         # Compile the LangGraph
initial_state = AgentState(jd_content=jd, resume_dir=dir, ...)
result = graph.invoke(initial_state)                          # Run the entire pipeline
response = result["final_response"]                           # Extract final output
```

The user runs:
```bash
python main.py sample_jd.txt sample_resumes/
```

### 2. `agent_graph.py` — The LangGraph Workflow

This defines the entire pipeline as a `StateGraph` with 5 nodes. Here's each node:

#### Node 1: `classify_intent`
- **What it does:** Sets the intent to `"match"` (hardcoded for this use case).
- **Why:** The graph was designed for extensibility — you could plug in an LLM call here to classify different intents (e.g., "summarize", "compare", "match").
- **Writes to state:** `state["intent"]`

#### Node 2: `extract_jd_skills`
- **What it does:** Uses a **LangChain PromptTemplate + Gemini LLM** chain to extract required skills from the JD as a JSON list.
- **Prompt:** Asks Gemini to return `{"required_skills": ["Python", "SQL", ...]}`.
- **Post-processing:** Strips markdown code fences, parses JSON, cleans skills (1-3 words max, limit 15).
- **LLM call:** `(prompt | llm).invoke({"jd": jd_content})` — a standard LangChain `RunnableSequence`.
- **Writes to state:** `state["jd_skills"]`

#### Node 3: `decide_action`
- **What it does:** Routes based on intent. If `"match"`, `"rank"`, or `"filter"` → route to `retrieve`. Otherwise → route to `direct`.
- **Writes to state:** `state["decision"]`

#### Conditional Edge: `should_retrieve`
- Reads `state["decision"].action` and returns the node name to route to (`"retrieve"` or `"respond"`).

#### Node 4: `retrieve_resumes`
- **What it does:** Instantiates `ResumeRAG`, loads and indexes all resumes into ChromaDB, then retrieves the top-5 most relevant resume chunks via similarity search.
- **Writes to state:** `state["retrieved_docs"]`

#### Node 5: `generate_response`
- **What it does:** The scoring engine.
  1. For each retrieved resume: substring-matches each required skill against the resume text (case-insensitive).
  2. Computes `match_score = matched_skills / total_skills`.
  3. Creates `RankedCandidate` objects with matched/missing skills.
  4. Sorts candidates by score descending.
  5. Calls Gemini to generate a shortlist summary paragraph.
  6. Assembles the final `FinalResponse`.
- **Writes to state:** `state["ranked_candidates"]`, `state["final_response"]`

#### Graph Wiring

```python
workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_intent)
workflow.add_node("extract_skills", extract_jd_skills)
workflow.add_node("decide", decide_action)
workflow.add_node("retrieve", retrieve_resumes)
workflow.add_node("respond", generate_response)

workflow.set_entry_point("classify")
workflow.add_edge("classify", "extract_skills")
workflow.add_edge("extract_skills", "decide")
workflow.add_conditional_edges("decide", should_retrieve, {
    "retrieve": "retrieve",
    "direct": "respond"
})
workflow.add_edge("retrieve", "respond")
workflow.add_edge("respond", END)

return workflow.compile()
```

### 3. `rag_system.py` — Resume RAG System

**Role:** Loads resumes, chunks them, creates a ChromaDB vector store, and retrieves relevant chunks.

| Step | Method | Detail |
|---|---|---|
| Load resumes | `load_resumes(dir)` | Reads `.txt` and `.pdf` files, wraps each as a LangChain `Document` |
| Chunk | `text_splitter.split_documents()` | `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=100) |
| Embed | `HuggingFaceEmbeddings` | Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local inference, no API key) |
| Index | `Chroma.from_documents()` | Creates a persistent ChromaDB at `chroma_resumes/` |
| Retrieve | `similarity_search(query, k=5)` | Cosine similarity search, returns top-5 chunks |

**Important:** The vector store is rebuilt from scratch on every run (deletes existing `chroma_resumes/`).

---

## Step-by-Step Data Flow

```
User runs: python main.py sample_jd.txt sample_resumes/
```

### Step 1: main.py loads the JD
- Reads `sample_jd.txt` into `jd_content` (a string).
- Creates initial `AgentState` with `jd_content`, `resume_dir`, empty lists for everything else.

### Step 2: classify_intent
- Sets `state["intent"]` to `IntentClassification(intent="match", confidence=0.95)`.
- No LLM call — hardcoded.

### Step 3: extract_jd_skills
- **Prompt:** "Extract all technical skills, tools, programming languages, frameworks..."
- **Gemini returns:** `{"required_skills": ["Python", "SQL", "Databases", "AWS", "Docker", ...]}`
- **Parsed & stored in:** `state["jd_skills"]`

### Step 4: decide_action
- `state["intent"].intent` is `"match"`, so `action = "retrieve"`.
- Conditional edge routes to the `retrieve` node.

### Step 5: retrieve_resumes
- `ResumeRAG` loads all `.txt`/`.pdf` files from `sample_resumes/`.
- Splits into chunks, embeds with MiniLM, stores in ChromaDB.
- Queries with `jd_content` → returns top-5 chunks.
- Stored in: `state["retrieved_docs"]`

### Step 6: generate_response
- For each of the 5 retrieved chunks:
  - Extracts candidate name from first line.
  - Checks each required skill: does it appear in the text? (substring match, case-insensitive)
  - Example: if skills = `["Python", "SQL", "AWS"]`, and resume says `"Experienced in Python and AWS"`, then `matched = ["Python", "AWS"]`, `missing = ["SQL"]`, `score = 2/3 = 0.67`.
- Sorts candidates by score (highest first).
- Calls Gemini for a shortlist criteria summary.
- Stores `FinalResponse` in `state["final_response"]`.

### Step 7: main.py prints results
- Reads `result["final_response"]` and prints JD summary, skills, ranked candidates with scores.

---

## Visual Data Flow

```
State at each step (key fields only):

START
┌─ jd_content: "We are looking for a Python developer..."
└─ resume_dir: "sample_resumes/"

  │
  ▼ classify_intent
┌─ intent: {intent: "match", confidence: 0.95}
│
▼ extract_jd_skills (LLM call)
┌─ jd_skills: ["Python", "SQL", "Databases", "AWS", "Docker", ...]
│
▼ decide_action
┌─ decision: {action: "retrieve"}
│
▼ retrieve_resumes (RAG pipeline)
┌─ retrieved_docs: [
│    "John Doe\nPython Developer\n...",
│    "Jane Smith\nSoftware Engineer\n...",
│    ... (5 docs)
│  ]
│
▼ generate_response (scoring + LLM summary)
┌─ ranked_candidates: [
│    {name: "John Doe", score: 0.88, matched: [...], missing: [...]},
│    {name: "Mike Johnson", score: 0.75, ...},
│    ...
│  ]
├─ final_response: {jd_summary, total, candidates, criteria}
│
▼ END
  Return to main.py → print results
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key in .env
echo "GOOGLE_API_KEY=your_key_here" > .env

# Run the shortlister
python main.py sample_jd.txt sample_resumes/
```

---

## Extending the App

Since you know LangChain, extending this is straightforward:

- **Swap the LLM:** Change `ChatGoogleGenerativeAI` to `ChatOpenAI`, `ChatAnthropic`, etc.
- **Change the embedder:** Swap `HuggingFaceEmbeddings` for `OpenAIEmbeddings`.
- **Add intent classification:** Make `classify_intent` call an LLM prompt instead of hardcoding.
- **Add more nodes:** Add new functions and `add_node`/`add_edge` calls to the graph.
- **Make it a loop:** Add a conditional edge back to an earlier node for iterative refinement.
