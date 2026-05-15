import json
import re
import logging
from typing import Dict, Any, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate

from schemas import (
    IntentClassification, DecisionOutput, RankedCandidate, FinalResponse, ToolOutput
)
from tools import (
    RetrievalTool, ScoringTool, SearchTool,
    register_tool, get_tool, TOOL_REGISTRY
)

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.0)


class AgentState(TypedDict):
    jd_content: str
    resume_dir: str
    jd_skills: List[str]
    intent: Optional[IntentClassification]
    decision: Optional[DecisionOutput]
    retrieved_docs: List[str]
    ranked_candidates: List[RankedCandidate]
    final_response: Optional[FinalResponse]
    tool_output: Optional[ToolOutput]
    query: str
    decisions_log: List[str]


def _clean_llm_json(content: str) -> str:
    content = re.sub(r'```json|```', '', content).strip()
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    return json_match.group() if json_match else content


def classify_intent(state: AgentState) -> AgentState:
    prompt = PromptTemplate.from_template(
        'You are an intent classifier for a resume shortlisting system. '
        'Given the user\'s query and job description, classify the intent.\n\n'
        'Choose one intent:\n'
        '- match: User wants to match candidates against a job description\n'
        '- summarize: User wants a summary of resumes or job description\n'
        '- compare: User wants to compare specific candidates\n'
        '- direct_query: User has a general question\n\n'
        'Return JSON: {{"intent": "...", "confidence": 0.0-1.0, "reasoning": "..."}}\n\n'
        'Query: {query}\n\n'
        'Job Description (first 300 chars): {jd}\n\n'
        'JSON:'
    )
    try:
        result = (prompt | llm).invoke({
            "query": state.get("query", "Shortlist candidates for this job"),
            "jd": state["jd_content"][:300]
        })
        content = _clean_llm_json(result.content if hasattr(result, 'content') else str(result))
        parsed = json.loads(content)
        state["intent"] = IntentClassification(**parsed)
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}, defaulting to match")
        state["intent"] = IntentClassification(intent="match", confidence=0.9, reasoning="Default fallback")

    state["decisions_log"].append(f"Intent: {state['intent'].intent} ({state['intent'].confidence})")
    logger.info(f"Intent: {state['intent'].model_dump()}")
    return state


def extract_jd_skills(state: AgentState) -> AgentState:
    logger.info("Extracting skills from JD using LLM...")
    prompt = PromptTemplate.from_template(
        "Extract all technical skills, tools, programming languages, frameworks, and technologies "
        "required from this job description.\n"
        "Return ONLY a JSON object with key 'required_skills' containing an array of strings "
        "(each skill should be 1-3 words max).\n\n"
        "Job Description:\n{jd}\n\n"
        "JSON:"
    )
    try:
        result = (prompt | llm).invoke({"jd": state["jd_content"]})
        content = _clean_llm_json(result.content if hasattr(result, 'content') else str(result))
        parsed = json.loads(content)
        skills = parsed.get("required_skills", [])
        state["jd_skills"] = [s.strip() for s in skills if 1 <= len(s.strip().split()) <= 3][:15]
    except Exception as e:
        logger.error(f"Skills extraction failed: {e}")
        state["jd_skills"] = []

    logger.info(f"Extracted {len(state['jd_skills'])} skills")
    return state


def decide_action(state: AgentState) -> AgentState:
    tools_desc = "\n".join([
        f"- {name}: {t.description} (input: {t.input_schema()})"
        for name, t in TOOL_REGISTRY.items()
    ])

    prompt = PromptTemplate.from_template(
        'You are a decision router for a resume shortlisting agent. '
        'Decide what action to take next.\n\n'
        'Intent: {intent}\n'
        'JD Skills: {skills}\n'
        'Available Tools:\n{tools}\n\n'
        'Choose action:\n'
        '- direct: Answer directly from LLM knowledge (for summaries, general questions)\n'
        '- retrieve: Retrieve resumes from the vector database (for finding/screening candidates)\n'
        '- tool: Use a specific tool (for scoring, web search)\n\n'
        'If action is \'tool\', specify which tool_name and the tool_input parameters.\n\n'
        'Return JSON: {{'
        '"action": "direct|retrieve|tool", '
        '"tool_name": null or "retrieval|scoring|search", '
        '"tool_input": {{}} or {{"query": "...", "k": 5, "required_skills": [...], "candidate_docs": [...]}}, '
        '"reasoning": "..."'
        '}}\n\n'
        'JSON:'
    )
    try:
        result = (prompt | llm).invoke({
            "intent": state["intent"].intent,
            "skills": ", ".join(state.get("jd_skills", [])),
            "tools": tools_desc
        })
        content = _clean_llm_json(result.content if hasattr(result, 'content') else str(result))
        parsed = json.loads(content)
        state["decision"] = DecisionOutput(**parsed)
    except Exception as e:
        logger.warning(f"Decision failed: {e}, defaulting to retrieve")
        state["decision"] = DecisionOutput(action="retrieve", reasoning="Fallback to retrieve")

    state["decisions_log"].append(f"Decision: {state['decision'].action} -> {state['decision'].tool_name or 'N/A'}")
    logger.info(f"Decision: {state['decision'].model_dump()}")
    return state


def retrieve_resumes(state: AgentState) -> AgentState:
    logger.info("Retrieving resumes via RAG...")
    from rag_system import ResumeRAG
    rag = ResumeRAG()
    rag.load_and_index(state["resume_dir"])
    if rag.vector_store is None:
        state["retrieved_docs"] = []
        return state
    docs = rag.retrieve_relevant(state["jd_content"], k=5, use_hybrid=True, rerank=False)
    state["retrieved_docs"] = [d.page_content for d in docs]
    logger.info(f"Retrieved {len(state['retrieved_docs'])} documents")
    return state


def execute_tool(state: AgentState) -> AgentState:
    tool_name = state["decision"].tool_name
    tool_input = state["decision"].tool_input or {}
    logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

    tool = get_tool(tool_name)
    if not tool:
        logger.error(f"Unknown tool: {tool_name}")
        state["tool_output"] = ToolOutput(tool_name=tool_name, success=False, result={}, error=f"Unknown tool: {tool_name}")
        return state

    if tool_name == "retrieval":
        from rag_system import ResumeRAG
        rag = ResumeRAG()
        rag.load_and_index(state["resume_dir"])
        if rag.vector_store:
            state["retrieved_docs"] = [d.page_content for d in rag.retrieve_relevant(
                tool_input.get("query", state["jd_content"]),
                k=tool_input.get("k", 5),
                use_hybrid=True
            )]
        tool_output = tool.run(query=tool_input.get("query", state["jd_content"]),
                               k=tool_input.get("k", 5))
    elif tool_name == "scoring":
        docs = tool_input.get("candidate_docs", state.get("retrieved_docs", []))
        skills = tool_input.get("required_skills", state.get("jd_skills", []))
        tool_output = tool.run(required_skills=skills, candidate_docs=docs)
        if tool_output.success:
            state["ranked_candidates"] = [
                RankedCandidate(**c) for c in tool_output.ranked_candidates
            ]
    elif tool_name == "search":
        tool_output = tool.run(query=tool_input.get("query", ""),
                                max_results=tool_input.get("max_results", 3))
    else:
        tool_output = ToolOutput(tool_name=tool_name, success=False, result={}, error="Unimplemented")

    state["tool_output"] = tool_output
    state["decisions_log"].append(
        f"Tool result: {tool_name} {'succeeded' if tool_output.success else 'failed'}"
    )
    logger.info(f"Tool {tool_name} result: success={tool_output.success}")
    return state


def generate_response(state: AgentState) -> AgentState:
    logger.info("Generating final response...")

    if not state.get("ranked_candidates") and state.get("retrieved_docs"):
        required_skills = state.get("jd_skills", [])
        candidates = []
        for i, doc_content in enumerate(state["retrieved_docs"]):
            lines = doc_content.split('\n')
            name = lines[0].strip() if lines else f"Candidate {i+1}"
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
        candidates = sorted(candidates, key=lambda x: x.match_score, reverse=True)
        state["ranked_candidates"] = candidates

    candidates = state.get("ranked_candidates", [])

    try:
        response_prompt = PromptTemplate.from_template(
            "Summarize the shortlisting results.\n\n"
            "JD summary: {jd_summary}\n"
            "Required skills: {skills}\n"
            "Decision trace: {decisions_log}\n\n"
            "Top candidates:\n{candidates}\n\n"
            "Provide a brief shortlist summary:"
        )
        result = (response_prompt | llm).invoke({
            "jd_summary": state["jd_content"][:200],
            "skills": ", ".join(state.get("jd_skills", [])),
            "decisions_log": "; ".join(state.get("decisions_log", [])),
            "candidates": "\n".join([
                f"- {c.candidate_name}: {c.match_score:.0%} - Matched: {', '.join(c.matched_skills)}"
                for c in candidates[:5]
            ])
        })
        criteria = result.content if hasattr(result, 'content') else str(result)
        criteria = criteria[:400]
    except Exception as e:
        logger.warning(f"LLM summary failed: {e}")
        criteria = "Skills match scoring"

    state["final_response"] = FinalResponse(
        jd_summary=state["jd_content"][:200],
        total_resumes_processed=len(state.get("retrieved_docs", [])),
        shortlisted_candidates=candidates[:5],
        shortlist_criteria=criteria,
        decisions_log=state.get("decisions_log", [])
    )
    return state


def should_retrieve(state: AgentState) -> str:
    return state["decision"].action


def build_graph():
    register_tool(RetrievalTool(None))
    register_tool(ScoringTool(llm=llm))
    register_tool(SearchTool())

    workflow = StateGraph(AgentState)
    workflow.add_node("classify", classify_intent)
    workflow.add_node("extract_skills", extract_jd_skills)
    workflow.add_node("decide", decide_action)
    workflow.add_node("retrieve", retrieve_resumes)
    workflow.add_node("tool_node", execute_tool)
    workflow.add_node("respond", generate_response)

    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "extract_skills")
    workflow.add_edge("extract_skills", "decide")

    workflow.add_conditional_edges(
        "decide", should_retrieve, {
            "retrieve": "retrieve",
            "tool": "tool_node",
            "direct": "respond"
        }
    )

    workflow.add_edge("retrieve", "respond")
    workflow.add_edge("tool_node", "respond")
    workflow.add_edge("respond", END)

    graph = workflow.compile()

    retrieval_tool = get_tool("retrieval")
    if retrieval_tool:
        from rag_system import ResumeRAG
        retrieval_tool.rag = ResumeRAG()

    return graph
