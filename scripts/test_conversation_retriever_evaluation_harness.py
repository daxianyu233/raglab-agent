from raglab.evaluation.retriever_evaluator import rank_metrics,aggregate

def main():
    h1,h3,rr,r3=rank_metrics({"turn:a"},["turn:x","turn:a"])
    assert h1==0.0 and h3==1.0 and rr==0.5 and r3==1.0
    hist=[
        {"category":"a","hit_at_1":1.0,"hit_at_3":1.0,"mrr_at_3":1.0,"recall_at_3":1.0,"retrieved_turn_ids":["turn:a"],"latency_ms":1.0},
        {"category":"b","hit_at_1":0.0,"hit_at_3":1.0,"mrr_at_3":0.5,"recall_at_3":1.0,"retrieved_turn_ids":["turn:x","turn:b"],"latency_ms":2.0},
    ]
    scope=[{"passed":True,"actual_turn_ids":[],"latency_ms":0.5}]
    m=aggregate(hist,scope)
    assert m["historical"]["hit_at_1"]==0.5
    assert m["historical"]["hit_at_3"]==1.0
    assert m["historical"]["mrr_at_3"]==0.75
    print("Conversation Retriever Evaluation Harness Phase 8B 纯逻辑回归测试通过")

if __name__=="__main__": main()
