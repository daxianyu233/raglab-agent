from scripts.evaluate_context_efficiency import aggregate

def main():
    selection = [
        {"full_history_tokens":1000,"selected_context_tokens":200,"token_reduction_rate":0.8,"required_turn_recall":1.0,"tool_pair_integrity_ok":True,"latency_ms":1.0},
        {"full_history_tokens":1000,"selected_context_tokens":400,"token_reduction_rate":0.6,"required_turn_recall":1.0,"tool_pair_integrity_ok":True,"latency_ms":1.0},
    ]
    compression = [
        {"expected_outcome":"fit","actual_outcome":"fit","final_fits":True,"passed":True,"compression_rate":0.5,"tool_pair_integrity_ok":True,"current_human_preserved":True,"required_history_retained":True,"latency_ms":2.0},
        {"expected_outcome":"fail_closed","actual_outcome":"fail_closed","final_fits":False,"passed":True,"compression_rate":None,"tool_pair_integrity_ok":True,"current_human_preserved":True,"required_history_retained":True,"latency_ms":2.0},
    ]
    m = aggregate(selection, compression)
    assert abs(m["selection"]["mean_token_reduction_rate"] - 0.7) < 1e-9
    assert abs(m["selection"]["weighted_token_reduction_rate"] - 0.7) < 1e-9
    assert m["compression"]["pass_rate"] == 1.0
    assert m["compression"]["fail_closed_correct_rate"] == 1.0
    print("Context Efficiency Evaluation Harness Phase 8C 纯逻辑回归测试通过")

if __name__ == "__main__":
    main()
