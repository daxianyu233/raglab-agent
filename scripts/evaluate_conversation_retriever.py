import argparse,json,time
from pathlib import Path
from raglab.evaluation.retriever_evaluator import load_dataset,run_benchmark
from raglab.settings import PROJECT_ROOT

DEFAULT=PROJECT_ROOT/"raglab"/"evaluation"/"datasets"/"conversation_retriever_v1.json"
REPORTS=PROJECT_ROOT/"reports"/"evaluation"

def pct(x): return "N/A" if x is None else f"{x*100:.2f}%"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,default=DEFAULT)
    p.add_argument("--show-all",action="store_true")
    a=p.parse_args()
    d=load_dataset(a.dataset)
    hist,scope,m=run_benchmark(d)
    h=m["historical"]
    print("="*88)
    print("Phase 8B Conversation Retriever Evaluation")
    print("="*88)
    print("Dataset：",a.dataset)
    print("Corpus Turns：",len(d["corpus_turns"]))
    print("Historical Cases：",len(hist))
    print("Scope Cases：",len(scope))
    print()
    print("Historical Ranking：")
    print("  Hit@1：",pct(h["hit_at_1"]))
    print("  Hit@3：",pct(h["hit_at_3"]))
    print("  MRR@3：",f"{h['mrr_at_3']:.4f}")
    print("  Recall@3：",pct(h["recall_at_3"]))
    print("  No Result：",f"{h['no_result_count']}/{len(hist)}")
    print()
    print("Per-category：")
    for k,v in h["per_category"].items():
        print(f"  {k}: Hit@1={pct(v['hit_at_1'])}, Hit@3={pct(v['hit_at_3'])}, MRR@3={v['mrr_at_3']:.4f}, Recall@3={pct(v['recall_at_3'])}")
    s=m["scope"]
    print()
    print("Scope / Event Selection：")
    print("  Pass：",f"{s['pass_count']}/{len(scope)} ({pct(s['pass_rate'])})")
    print("  Current Turn Exclusion：",pct(s["current_turn_exclusion_rate"]))
    misses=[x for x in hist if x["hit_at_3"]<1.0 or x["recall_at_3"]<1.0]
    if misses:
        print()
        print("Historical Misses / Partial Recall：")
        for x in misses:
            print(f"  [{x['case_id']}] {x['category']}")
            print("    Query：",x["query"])
            print("    Relevant：",x["relevant_turn_ids"])
            print("    Retrieved：",x["retrieved_turn_ids"])
            print("    Recall@3：",pct(x["recall_at_3"]))
    bad=[x for x in scope if not x["passed"]]
    if bad:
        print()
        print("Scope Failures：")
        for x in bad:
            print(f"  [{x['case_id']}] {x['mismatches']}")
            print("    Expected Turns：",x["expected_turn_ids"])
            print("    Actual Turns：",x["actual_turn_ids"])
            if x["expected_event_signatures"]:
                print("    Expected Events：",x["expected_event_signatures"])
                print("    Actual Events：",x["actual_event_signatures"])
    if a.show_all:
        print()
        print("All Historical Cases：")
        for x in hist:
            print(f"  {x['case_id']}: H1={x['hit_at_1']:.0f}, H3={x['hit_at_3']:.0f}, RR={x['mrr_at_3']:.3f}, R3={x['recall_at_3']:.3f}, Top={x['retrieved_turn_ids']}")
    print()
    print("Local Retrieval Latency：")
    print("  mean：",f"{m['latency_ms']['mean']:.3f} ms")
    print("  max：",f"{m['latency_ms']['max']:.3f} ms")
    REPORTS.mkdir(parents=True,exist_ok=True)
    path=REPORTS/f"conversation_retriever_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"evaluation":"conversation_retriever","dataset":str(a.dataset),"metrics":m,"historical_results":hist,"scope_results":scope},ensure_ascii=False,indent=2),encoding="utf-8")
    print()
    print("="*88)
    print("Evaluation report：",path)
    print("="*88)

if __name__=="__main__": main()
