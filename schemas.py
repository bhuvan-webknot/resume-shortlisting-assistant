from pydantic import BaseModel
from typing import List, Optional

class IntentClassification(BaseModel):
    intent: str
    confidence: float
    reasoning: str

class DecisionOutput(BaseModel):
    action: str  # "retrieve", "tool", "direct"
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
