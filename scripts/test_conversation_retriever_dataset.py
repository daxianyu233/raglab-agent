import json
from raglab.settings import PROJECT_ROOT

DATASET=PROJECT_ROOT/"raglab"/"evaluation"/"datasets"/"conversation_retriever_v1.json"

def main():
    d=json.loads(DATASET.read_text(encoding="utf-8"))
    ids=[x["turn_id"] for x in d["corpus_turns"]]
    assert len(ids)==len(set(ids))
    assert d["current_turn_id"] in ids
    cases=d["historical_cases"]+d["scope_cases"]
    case_ids=[x["case_id"] for x in cases]
    assert len(case_ids)==len(set(case_ids))
    corpus=set(ids)
    for c in d["historical_cases"]:
        assert c["relevant_turn_ids"]
        assert set(c["relevant_turn_ids"]) <= corpus
        assert d["current_turn_id"] not in c["relevant_turn_ids"]
    required={"exact_entity","paraphrase","distractor","multi_relevant","tool_aware","conceptual"}
    assert required <= {c["category"] for c in d["historical_cases"]}
    assert d["historical_turn_limit"]==3
    assert d["recent_turn_limit"]==3
    print("Conversation Retriever Dataset v1 语义一致性测试通过")
    print("Corpus Turns：",len(d["corpus_turns"]))
    print("Historical Cases：",len(d["historical_cases"]))
    print("Scope Cases：",len(d["scope_cases"]))

if __name__=="__main__": main()
