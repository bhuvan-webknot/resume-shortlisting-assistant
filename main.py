import sys
import os
import logging
from agent_graph import build_graph, AgentState

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _is_gdrive_input(path: str) -> bool:
    if os.path.exists(path):
        return False
    return (
        path.startswith("http") or
        path.startswith("drive.google.com") or
        (len(path) >= 25 and "/" not in path and "\\" not in path)
    )


def resolve_resume_source(resume_input: str) -> str:
    if _is_gdrive_input(resume_input):
        logger.info("Input looks like a Google Drive folder — downloading resumes...")
        from google_drive import download_resumes_from_drive
        return download_resumes_from_drive(resume_input)
    return resume_input


def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <jd_file_path> <resume_dir_or_gdrive_url> [query]")
        print("  <resume_dir_or_gdrive_url> can be:")
        print("    - A local directory path")
        print("    - A Google Drive folder URL or folder ID")
        sys.exit(1)

    jd_path = sys.argv[1]
    resume_input = sys.argv[2]
    query = sys.argv[3] if len(sys.argv) > 3 else "Shortlist the best candidates for this job"

    resume_dir = resolve_resume_source(resume_input)
    logger.info(f"Resume source: {resume_dir}")

    if jd_path.endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(jd_path)
        jd_content = loader.load()[0].page_content
    else:
        with open(jd_path, 'r') as f:
            jd_content = f.read()

    graph = build_graph()
    initial_state = AgentState(
        jd_content=jd_content,
        resume_dir=resume_dir,
        jd_skills=[],
        intent=None,
        decision=None,
        retrieved_docs=[],
        ranked_candidates=[],
        final_response=None,
        query=query,
        decisions_log=[]
    )

    result = graph.invoke(initial_state)
    response = result["final_response"]

    intent = result.get("intent")
    intent_label = intent.intent if intent else "match"

    print("\n" + "=" * 60)
    if intent_label == "compare":
        print("CANDIDATE COMPARISON RESULTS")
    else:
        print("RESUME SHORTLISTING RESULTS")
    print("=" * 60)
    print(f"\nQuery: {query}")
    print(f"Intent: {intent_label}")
    print(f"JD Summary: {response.jd_summary}")
    print(f"Total Resumes Processed: {response.total_resumes_processed}")
    print(f"\nExtracted JD Skills: {', '.join(result.get('jd_skills', []))}")

    print(f"\nShortlist Criteria: {response.shortlist_criteria}")
    if intent_label == "compare":
        print("\nCandidates Compared:")
    else:
        print("\nTop Candidates:")
    for i, c in enumerate(response.shortlisted_candidates, 1):
        print(f"\n  {i}. {c.candidate_name}")
        print(f"     Match Score: {c.match_score:.2f}")
        print(f"     Matched Skills: {', '.join(c.matched_skills) if c.matched_skills else 'None'}")
        print(f"     Missing Skills: {', '.join(c.missing_skills) if c.missing_skills else 'None'}")
        print(f"     Summary: {c.summary[:100]}...")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
