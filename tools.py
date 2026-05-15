import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from schemas import (
    RetrievalToolInput, RetrievalToolOutput,
    ScoringToolInput, ScoringToolOutput,
    SearchToolInput, SearchToolOutput,
    ToolOutput
)

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs) -> ToolOutput:
        pass

    @abstractmethod
    def input_schema(self) -> dict:
        pass

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema()}


class RetrievalTool(BaseTool):
    name = "retrieval"
    description = "Search and retrieve relevant resume documents from the vector database"

    def __init__(self, rag_system):
        self.rag = rag_system

    def input_schema(self) -> dict:
        return RetrievalToolInput.model_json_schema()

    def run(self, query: str, k: int = 5, **kwargs) -> RetrievalToolOutput:
        logger.info(f"RetrievalTool.run(query={query[:50]}..., k={k})")
        try:
            docs = self.rag.retrieve_relevant(query, k=k)
            documents = [d.page_content for d in docs]
            scores = [d.metadata.get("score", 0.0) for d in docs]
            sources = [d.metadata.get("source", "unknown") for d in docs]
            return RetrievalToolOutput(tool_name=self.name, success=True, result={
                "documents": documents, "scores": scores, "sources": sources
            }, documents=documents, scores=scores, sources=sources)
        except Exception as e:
            logger.error(f"RetrievalTool failed: {e}")
            return RetrievalToolOutput(tool_name=self.name, success=False, result={},
                documents=[], scores=[], sources=[], error=str(e))


class ScoringTool(BaseTool):
    name = "scoring"
    description = "Score and rank candidates against required job skills"

    def __init__(self, llm=None):
        self.llm = llm

    def input_schema(self) -> dict:
        return ScoringToolInput.model_json_schema()

    def run(self, required_skills: List[str], candidate_docs: List[str], **kwargs) -> ScoringToolOutput:
        logger.info(f"ScoringTool.run(skills={len(required_skills)}, candidates={len(candidate_docs)})")
        try:
            candidates = []
            for i, doc_content in enumerate(candidate_docs):
                lines = doc_content.split('\n')
                name = lines[0].strip() if lines else f"Candidate {i+1}"
                doc_lower = doc_content.lower()
                matched = [s for s in required_skills if s.lower() in doc_lower]
                missing = [s for s in required_skills if s.lower() not in doc_lower]
                score = len(matched) / len(required_skills) if required_skills else 0.0
                candidates.append({
                    "candidate_name": name,
                    "match_score": round(score, 2),
                    "matched_skills": matched,
                    "missing_skills": missing,
                    "summary": doc_content[:200]
                })

            candidates = sorted(candidates, key=lambda x: x["match_score"], reverse=True)
            summary = f"Scored {len(candidates)} candidates. "
            if candidates:
                summary += f"Top: {candidates[0]['candidate_name']} ({candidates[0]['match_score']:.0%})"

            return ScoringToolOutput(tool_name=self.name, success=True, result={
                "ranked_candidates": candidates, "summary": summary
            }, ranked_candidates=candidates, summary=summary)
        except Exception as e:
            logger.error(f"ScoringTool failed: {e}")
            return ScoringToolOutput(tool_name=self.name, success=False, result={},
                ranked_candidates=[], summary="", error=str(e))


class SearchTool(BaseTool):
    name = "search"
    description = "Search the web for additional context about companies, technologies, or candidates"

    def __init__(self):
        self._cache = {}

    def input_schema(self) -> dict:
        return SearchToolInput.model_json_schema()

    def run(self, query: str, max_results: int = 3, **kwargs) -> SearchToolOutput:
        logger.info(f"SearchTool.run(query={query}, max_results={max_results})")
        if query in self._cache:
            return SearchToolOutput(tool_name=self.name, success=True, result=self._cache[query],
                results=self._cache[query])
        try:
            import urllib.parse, urllib.request, json
            encoded = urllib.parse.quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            results = []
            if data.get("AbstractText"):
                results.append({"title": data.get("Heading", ""), "snippet": data["AbstractText"]})
            if data.get("RelatedTopics"):
                for topic in data["RelatedTopics"][:max_results - len(results)]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append({"title": topic.get("Text", "").split(" - ")[0], "snippet": topic["Text"]})
            self._cache[query] = results
            return SearchToolOutput(tool_name=self.name, success=True, result=results, results=results)
        except Exception as e:
            logger.warning(f"SearchTool failed (non-critical): {e}")
            return SearchToolOutput(tool_name=self.name, success=False, result=[],
                results=[], error=str(e))


TOOL_REGISTRY: Dict[str, BaseTool] = {}


def register_tool(tool: BaseTool):
    TOOL_REGISTRY[tool.name] = tool
    logger.info(f"Registered tool: {tool.name}")


def get_tool(name: str) -> Optional[BaseTool]:
    return TOOL_REGISTRY.get(name)


def list_tools() -> List[dict]:
    return [t.to_dict() for t in TOOL_REGISTRY.values()]
