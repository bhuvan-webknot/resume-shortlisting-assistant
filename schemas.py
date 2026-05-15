from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class IntentClassification(BaseModel):
    intent: str = Field(description="One of: match, summarize, compare, direct_query")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class DecisionOutput(BaseModel):
    action: str = Field(description="One of: retrieve, tool, direct")
    tool_name: Optional[str] = Field(default=None, description="Tool to invoke if action=tool")
    tool_input: Optional[Dict[str, Any]] = Field(default=None, description="Input params for the tool")
    reasoning: str


class ToolInput(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = None


class ToolOutput(BaseModel):
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None


class RetrievalToolInput(BaseModel):
    query: str = Field(description="Search query to find relevant resumes")
    k: int = Field(default=5, ge=1, le=20)


class RetrievalToolOutput(ToolOutput):
    documents: List[str] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class ScoringToolInput(BaseModel):
    required_skills: List[str] = Field(description="Skills required for the job")
    candidate_docs: List[str] = Field(description="Candidate resume texts to score")


class ScoringToolOutput(ToolOutput):
    ranked_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    summary: str = ""


class SearchToolInput(BaseModel):
    query: str = Field(description="Web search query")
    max_results: int = Field(default=3, ge=1, le=10)


class SearchToolOutput(ToolOutput):
    results: List[Dict[str, str]] = Field(default_factory=list)


class RankedCandidate(BaseModel):
    candidate_name: str
    match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    summary: str


class FinalResponse(BaseModel):
    jd_summary: str
    total_resumes_processed: int
    shortlisted_candidates: List[RankedCandidate]
    shortlist_criteria: str
    decisions_log: List[str] = Field(default_factory=list, description="Trace of agent decisions")
