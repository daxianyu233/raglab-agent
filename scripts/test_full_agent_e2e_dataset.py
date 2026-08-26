import json
from raglab.settings import PROJECT_ROOT
DATASET=PROJECT_ROOT/"raglab"/"evaluation"/"datasets"/"full_agent_e2e_v1.json"
def main():
    d=json.loads(DATASET.read_text(encoding="utf-8")); cases=d["cases"]
    ids=[c["case_id"] for c in cases]
    assert len(ids)==len(set(ids)) and len(cases)==20
    required={"direct_no_tool","knowledge_retrieval","github_semantic_retrieval","structured_query","previous_answer_reuse","previous_tool_evidence","recent_context","historical_context","long_term_memory","retrieval_forbidden","tool_minimality","dynamic_capability","hitl_reject","hitl_approve","event_persistence","thread_isolation","tool_error_recovery","max_steps","archive_reconciliation","working_memory_safety"}
    assert required=={c["category"] for c in cases}
    for c in cases:
        assert c["user_input"].strip() and c["assertions"]
    print("Full-Agent E2E Dataset v1 语义一致性测试通过")
    print("Cases：",len(cases))
    print("Categories：",len(required))
    print("接口绑定：无（通过 FullAgentE2EAdapter 解耦）")
if __name__=="__main__": main()
