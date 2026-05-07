import json
import re
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from schemas import IntentClassification, DecisionOutput, RankedCandidate, FinalResponse

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize LLM with Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.0)

class AgentState(TypedDict):
    jd_content: str
    resume_dir: str
    jd_skills: List[str]
    intent: IntentClassification
    decision: DecisionOutput
    retrieved_docs: List[str]
    ranked_candidates: List[RankedCandidate]
    final_response: FinalResponse

def classify_intent(state: AgentState) -> AgentState:
    """Classify intent - default to match for resume shortlisting"""
    print("[DEBUG] Setting intent to match...")
    state["intent"] = IntentClassification(
        intent="match",
        confidence=0.95,
        reasoning="Resume shortlisting task - default to match intent"
    )
    return state

def extract_jd_skills(state: AgentState) -> AgentState:
    """Extract required skills from JD using LLM"""
    print("[DEBUG] Extracting skills from JD using LLM...")
    jd_content = state["jd_content"]

    prompt = PromptTemplate.from_template(
        "Extract all technical skills, tools, programming languages, frameworks, and technologies "
        "required from this job description.\n"
        "Return ONLY a JSON object with key 'required_skills' containing an array of strings "
        "(each skill should be 1-3 words max).\n\n"
        "Job Description:\n{jd}\n\n"
        "JSON:"
    )

    try:
        chain = prompt | llm
        result = chain.invoke({"jd": jd_content})

        # Gemini often wraps JSON in markdown blocks
        if hasattr(result, 'content'):
            content = result.content
        else:
            content = str(result)
            
        content = re.sub(r'```json|```', '', content).strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            skills = parsed.get("required_skills", [])
            # Clean and filter skills
            cleaned = [s.strip() for s in skills if 1 <= len(s.strip().split()) <= 3]
            state["jd_skills"] = cleaned[:15]
        else:
            state["jd_skills"] = []
    except Exception as e:
        print(f"[ERROR] LLM skills extraction failed: {e}, using empty list")
        state["jd_skills"] = []

    print(f"[DEBUG] Extracted skills: {state['jd_skills']}")
    return state

def decide_action(state: AgentState) -> AgentState:
    """Decide action based on intent"""
    action = "retrieve" if state["intent"].intent in ["match", "rank", "filter"] else "direct"
    state["decision"] = DecisionOutput(action=action, reasoning=f"Intent: {state['intent'].intent}")
    return state

def retrieve_resumes(state: AgentState) -> AgentState:
    """Retrieve relevant resumes using RAG"""
    print("[DEBUG] Starting resume retrieval...")
    from rag_system import ResumeRAG
    rag = ResumeRAG()
    rag.load_and_index(state["resume_dir"])
    if rag.vector_store is None:
        state["retrieved_docs"] = []
        return state
    docs = rag.retrieve_relevant(state["jd_content"], k=5)
    state["retrieved_docs"] = [d.page_content for d in docs]
    print(f"[DEBUG] Retrieved {len(state['retrieved_docs'])} docs")
    return state

def generate_response(state: AgentState) -> AgentState:
    """Generate final response using LLM with simple candidate scoring"""
    print("[DEBUG] Generating final response...")

    required_skills = state.get("jd_skills", [])
    candidates = []

    for i, doc_content in enumerate(state["retrieved_docs"]):
        lines = doc_content.split('\n')
        name = lines[0] if lines else f"Candidate {i+1}"
        doc_lower = doc_content.lower()

        matched = [s for s in required_skills if s.lower() in doc_lower]
        missing = [s for s in required_skills if s.lower() not in doc_lower]
        score = len(matched) / len(required_skills) if required_skills else 0.5

        candidates.append(RankedCandidate(
            candidate_name=name,
            match_score=round(score, 2),
            matched_skills=matched,
            missing_skills=missing,
            summary=doc_content[:200]
        ))

    candidates = sorted(candidates, key=lambda x: x.match_score, reverse=True)[:3]
    state["ranked_candidates"] = candidates

    try:
        response_prompt = PromptTemplate.from_template(
            "Summarize these top candidates for the job.\n\n"
            "JD: {jd_summary}\n"
            "Skills: {skills}\n"
            "Top:\n{candidates}\n\n"
            "Summary:"
        )
        result = (response_prompt | llm).invoke({
            "jd_summary": state["jd_content"][:200],
            "skills": ", ".join(state.get("jd_skills", [])),
            "candidates": "\n".join([
                f"- {c.candidate_name}: {c.match_score} - Matched: {', '.join(c.matched_skills)}"
                for c in candidates
            ])
        })

        reasoning = result.content if hasattr(result, 'content') else str(result)
        criteria = reasoning[:300]
    except Exception as e:
        print(f"[WARN] LLM response failed: {e}")
        criteria = "Skills match"

    state["final_response"] = FinalResponse(
        jd_summary=state["jd_content"][:200],
        total_resumes_processed=len(state["retrieved_docs"]),
        shortlisted_candidates=candidates,
        shortlist_criteria=criteria
    )
    return state

def should_retrieve(state: AgentState) -> str:
    return state["decision"].action

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("classify", classify_intent)
    workflow.add_node("extract_skills", extract_jd_skills)
    workflow.add_edge("classify", "extract_skills")
    workflow.add_node("decide", decide_action)
    workflow.add_edge("extract_skills", "decide")
    workflow.add_node("retrieve", retrieve_resumes)
    workflow.add_node("respond", generate_response)

    workflow.set_entry_point("classify")
    workflow.add_conditional_edges("decide", should_retrieve, {"retrieve": "retrieve", "direct": "respond"})
    workflow.add_edge("retrieve", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()