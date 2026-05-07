# AI Resume Shortlister

Agentic AI system that automates resume screening using LangGraph + RAG.

## Setup

```bash
pip install -r requirements.txt
```

Set your Gemini API key in a `.env` file:
```
GOOGLE_API_KEY=your-api-key-here
```

## Usage

```bash
python main.py <job_description_file> <resume_directory>
```

Example:
```bash
python main.py jd.txt ./resumes/
```

## How it works

1. **Intent Classification** - Understands the task type
2. **Skill Extraction** - Gemini extracts required skills from JD
3. **RAG Retrieval** - Finds relevant resumes using vector search
4. **Candidate Scoring** - Simple string match scoring against JD skills
5. **Response** - Gemini generates a structured shortlist summary

## Project Structure

- `agent_graph.py` - LangGraph workflow
- `rag_system.py` - Resume RAG system
- `schemas.py` - Structured output schemas
- `main.py` - CLI entry point
