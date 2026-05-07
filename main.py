import sys
from agent_graph import build_graph, AgentState
from schemas import FinalResponse

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <jd_file_path> <resume_dir_path>")
        sys.exit(1)

    jd_path = sys.argv[1]
    resume_dir = sys.argv[2]

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
        final_response=None
    )

    result = graph.invoke(initial_state)

    response = result["final_response"]
    print("\n" + "="*60)
    print("RESUME SHORTLISTING RESULTS")
    print("="*60)
    print(f"\nJD Summary: {response.jd_summary}")
    print(f"Total Resumes Processed: {response.total_resumes_processed}")
    print(f"\nExtracted JD Skills: {', '.join(result.get('jd_skills', []))}")
    print(f"\nShortlist Criteria: {response.shortlist_criteria}")
    print("\nTop Candidates:")
    for i, c in enumerate(response.shortlisted_candidates, 1):
        print(f"\n{i}. {c.candidate_name}")
        print(f"   Match Score: {c.match_score:.2f}")
        print(f"   Matched Skills: {', '.join(c.matched_skills) if c.matched_skills else 'None'}")
        print(f"   Missing Skills: {', '.join(c.missing_skills) if c.missing_skills else 'None'}")
        print(f"   Summary: {c.summary[:100]}...")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
