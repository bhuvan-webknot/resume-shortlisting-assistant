from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class IntentClassification(BaseModel):
    intent: str = Field(description="One of: match, summarize, compare, direct_query")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class DecisionOutput(BaseModel):
    action: str = Field(description="One of: retrieve, direct")
    reasoning: str


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
