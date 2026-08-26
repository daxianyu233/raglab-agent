import json
from raglab.settings import PROJECT_ROOT

DATASET = PROJECT_ROOT / "raglab" / "evaluation" / "datasets" / "context_efficiency_v1.json"

def main():
    d = json.loads(DATASET.read_text(encoding="utf-8"))
    selection = d["selection_cases"]
    compression = d["compression_cases"]
    ids = [x["case_id"] for x in selection + compression]
    assert len(ids) == len(set(ids))
    assert {x["category"] for x in selection} >= {
        "no_history", "previous_turn", "recent_turns", "historical_search", "raw_tool_evidence"
    }
    assert {x["category"] for x in compression} >= {
        "historical_tool_result", "drop_low_priority_turn", "current_tool_result",
        "thread_summary", "long_term_memory", "protected_current_human"
    }
    for x in compression:
        assert x["expected_outcome"] in {"fit", "fail_closed"}
        assert x["model_context_limit_tokens"] > x["reserved_output_tokens"] + x["safety_margin_tokens"]
    print("Context Efficiency Dataset v1 语义一致性测试通过")
    print("Selection Cases：", len(selection))
    print("Compression Cases：", len(compression))

if __name__ == "__main__":
    main()
