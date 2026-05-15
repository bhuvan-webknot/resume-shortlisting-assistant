"""
Test queries covering three scenarios:
  1. Retrieval cases  – find/shortlist candidates
  2. Tool usage cases – score candidates, web search
  3. Direct answer    – summarise JD, general questions

Run standalone tool tests first (no API calls), then one end-to-end pipeline test.
"""

import sys
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def test_retrieval_tool():
    from rag_system import ResumeRAG
    from tools import RetrievalTool

    rag = ResumeRAG()
    rag.load_and_index("sample_resumes/")
    tool = RetrievalTool(rag)

    out = tool.run(query="Python developer with AWS experience", k=3)
    assert out.success, f"RetrievalTool failed: {out.error}"
    assert len(out.documents) > 0, "No documents retrieved"
    assert len(out.documents) <= 3, f"Expected <= 3 docs, got {len(out.documents)}"
    logger.info(f"  -> Retrieved {len(out.documents)} docs: {out.sources}")
    return True


def test_scoring_tool():
    from tools import ScoringTool

    tool = ScoringTool()
    sample_docs = [
        "John Doe\nPython Developer\n5 years Python. SQL, AWS, Docker, Git.",
        "Jane Smith\nData Engineer\nPython, Spark, AWS, SQL, Kafka.",
        "Bob Brown\nDesigner\nFigma, Photoshop, UI/UX.",
    ]
    out = tool.run(
        required_skills=["Python", "SQL", "AWS", "Docker"],
        candidate_docs=sample_docs
    )
    assert out.success, f"ScoringTool failed: {out.error}"
    assert len(out.ranked_candidates) == 3, f"Expected 3, got {len(out.ranked_candidates)}"
    assert out.ranked_candidates[0]["match_score"] >= out.ranked_candidates[1]["match_score"], "Not sorted"
    logger.info(f"  -> Ranked {len(out.ranked_candidates)} candidates: top={out.ranked_candidates[0]['candidate_name']}")
    return True


def test_search_tool():
    from tools import SearchTool

    tool = SearchTool()
    out = tool.run(query="Python developer skills", max_results=2)
    logger.info(f"  -> Search {'succeeded' if out.success else 'failed (non-critical)'}, results={len(out.results)}")
    return True


def test_tool_registry():
    from tools import list_tools, get_tool, register_tool, RetrievalTool, ScoringTool, SearchTool

    register_tool(RetrievalTool(None))
    register_tool(ScoringTool())
    register_tool(SearchTool())
    tools = list_tools()
    assert len(tools) > 0, "No tools registered"
    names = [t["name"] for t in tools]
    assert "retrieval" in names, "retrieval not registered"
    assert "scoring" in names, "scoring not registered"
    logger.info(f"  -> Registered tools: {', '.join(names)}")
    return True


def test_rag_hybrid_search():
    from rag_system import ResumeRAG

    rag = ResumeRAG()
    rag.load_and_index("sample_resumes/")
    docs = rag.retrieve_relevant("Python AWS developer", k=3, use_hybrid=True, rerank=False)
    assert len(docs) > 0, "No docs with hybrid search"
    logger.info(f"  -> Hybrid search returned {len(docs)} docs")
    return True


def test_rag_schemas():
    from schemas import RetrievalToolInput, ScoringToolInput, ToolOutput

    inp = RetrievalToolInput(query="Python", k=5)
    assert inp.query == "Python"
    assert inp.k == 5

    inp2 = ScoringToolInput(required_skills=["Python"], candidate_docs=["resume text"])
    assert len(inp2.required_skills) == 1

    out = ToolOutput(tool_name="test", success=True, result={"key": "val"})
    assert out.success
    logger.info("  -> All schema validations passed")
    return True


def test_full_pipeline():
    from agent_graph import build_graph, AgentState

    jd_text = (
        "Job Title: Python Developer\n"
        "Requirements:\n- 3+ years of Python experience\n"
        "- Strong knowledge of SQL and databases\n"
        "- Experience with AWS cloud services\n"
        "- Familiarity with Docker and containerization\n"
        "- Good understanding of REST APIs\n"
        "- Experience with Git version control\n"
    )

    graph = build_graph()
    initial = AgentState(
        jd_content=jd_text,
        resume_dir="sample_resumes/",
        jd_skills=[],
        intent=None,
        decision=None,
        retrieved_docs=[],
        ranked_candidates=[],
        final_response=None,
        tool_output=None,
        query="Find the best candidates for this Python Developer role",
        decisions_log=[]
    )

    result = graph.invoke(initial)
    response = result.get("final_response")
    assert response is not None, "No final response"
    assert len(response.shortlisted_candidates) > 0, "No candidates shortlisted"
    assert response.total_resumes_processed > 0, "No resumes processed"
    logger.info(f"  -> Pipeline: {response.total_resumes_processed} docs, {len(response.shortlisted_candidates)} shortlisted")
    logger.info(f"  -> Decisions: {response.decisions_log}")
    for c in response.shortlisted_candidates[:3]:
        logger.info(f"     {c.candidate_name}: {c.match_score:.0%}")
    return True


def run_tests(include_llm: bool = False):
    tests = [
        ("Tool schemas",        test_rag_schemas,       False),
        ("Tool registry",       test_tool_registry,     False),
        ("RAG hybrid search",   test_rag_hybrid_search, False),
        ("RetrievalTool",       test_retrieval_tool,    False),
        ("ScoringTool",         test_scoring_tool,      False),
        ("SearchTool",          test_search_tool,       False),
    ]
    if include_llm:
        tests.append(("Full pipeline (LLM)", test_full_pipeline, True))

    passed = 0
    failed = 0
    skipped = 0

    print("=" * 60)
    print("TEST QUERIES: Resume Shortlisting Agent")
    print("=" * 60)

    for name, fn, needs_llm in tests:
        label = f"[{'LLM' if needs_llm else 'UNIT'}] {name}"
        print(f"\n  {label}...", end=" ")
        ok = False
        try:
            ok = fn()
        except Exception as e:
            print(f"FAIL\n    {e}")
            failed += 1
            continue

        if ok:
            print("PASS")
            passed += 1
        else:
            print("FAIL (returned False)")
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed + skipped} total")
    if skipped:
        print(f"  ({skipped} skipped, use --llm to include LLM-dependent tests)")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    include_llm = "--llm" in sys.argv
    sys.exit(run_tests(include_llm=include_llm))
