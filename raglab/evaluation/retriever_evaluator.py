from __future__ import annotations
import json, tempfile, time, statistics
from pathlib import Path
from collections import defaultdict
from raglab.agent.context_plan import ContextPlan
from raglab.agent.conversation_event_store import ConversationEventStore
from raglab.agent.conversation_retriever import ConversationRetriever

def tid(x): return "turn:" + str(x)

def load_dataset(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def plan(scope, query=None, previous=False, raw=False):
    return ContextPlan(
        task_intent="evaluation", response_goal="evaluate retrieval",
        history_required=(scope!="none"), history_scope=scope, history_query=query,
        previous_answer_required=previous, raw_tool_evidence_required=raw,
        external_retrieval_required=False, external_retrieval_allowed=False,
        long_term_memory_required=False, long_term_memory_query=None,
        referenced_entities=[], temporal_scope=None, confidence=1.0,
    )

def append(store, thread, turn, suffix, role, text, payload=None, tool_call_id=None, tool_name=None):
    store.append_event(
        user_id="eval-user", thread_id=thread, turn_id=turn, event_type="message",
        role=role, message_id=f"{turn}:{suffix}", content_text=text,
        payload=payload or {}, metadata={"evaluation":True},
        tool_call_id=tool_call_id, tool_name=tool_name,
    )

def populate(store, data, thread):
    for x in data["corpus_turns"]:
        turn=tid(x["turn_id"])
        append(store,thread,turn,"human","human",x["human"])
        tool=x.get("tool_name")
        if tool:
            call=f"call:{x['turn_id']}"
            append(store,thread,turn,"ai-call","assistant","",
                   {"tool_calls":[{"id":call,"name":tool,"args":{},"type":"tool_call"}]},call,tool)
            append(store,thread,turn,"tool","tool",x.get("tool_result",""),{},call,tool)
        append(store,thread,turn,"ai-final","assistant",x["assistant"],{"tool_calls":[]})

def sig(event):
    if event.role=="assistant":
        calls=event.payload.get("tool_calls",[])
        return "assistant_tool_call" if isinstance(calls,list) and calls else "assistant_final"
    return event.role

def rank_metrics(relevant,retrieved):
    rel=set(relevant); top1=retrieved[:1]; top3=retrieved[:3]
    hit1=float(any(x in rel for x in top1))
    hit3=float(any(x in rel for x in top3))
    rr=0.0
    for i,x in enumerate(top3,1):
        if x in rel: rr=1/i; break
    recall=len(rel & set(top3))/len(rel)
    return hit1,hit3,rr,recall

def run_benchmark(data):
    thread="evaluation-thread"
    current=tid(data["current_turn_id"])
    with tempfile.TemporaryDirectory() as tmp:
        store=ConversationEventStore(Path(tmp)/"events.sqlite3")
        try:
            populate(store,data,thread)
            retriever=ConversationRetriever(
                store=store,
                recent_turn_limit=int(data.get("recent_turn_limit",3)),
                historical_turn_limit=int(data.get("historical_turn_limit",3)),
            )
            hist=[]
            for c in data["historical_cases"]:
                p=plan("historical_search",c["query"])
                t0=time.perf_counter()
                r=retriever.retrieve(thread_id=thread,plan=p,current_turn_id=current)
                ms=(time.perf_counter()-t0)*1000
                relevant=[tid(x) for x in c["relevant_turn_ids"]]
                got=list(r.selected_turn_ids)
                h1,h3,rr,rec=rank_metrics(relevant,got)
                hist.append({
                    "case_id":c["case_id"],"category":c["category"],"query":c["query"],
                    "relevant_turn_ids":relevant,"retrieved_turn_ids":got,
                    "hit_at_1":h1,"hit_at_3":h3,"mrr_at_3":rr,"recall_at_3":rec,
                    "latency_ms":ms,
                    "matched_terms_by_turn":{t.turn_id:list(t.matched_terms) for t in r.turns},
                })
            scope=[]
            for c in data["scope_cases"]:
                p=plan(c["history_scope"],c.get("history_query"),
                       c.get("previous_answer_required",False),
                       c.get("raw_tool_evidence_required",False))
                t0=time.perf_counter()
                r=retriever.retrieve(thread_id=thread,plan=p,current_turn_id=current)
                ms=(time.perf_counter()-t0)*1000
                got=list(r.selected_turn_ids)
                expected=[tid(x) for x in c.get("expected_turn_ids",[])]
                actual_sigs=[sig(e) for turn in r.turns for e in turn.selected_events]
                expected_sigs=c.get("expected_event_signatures",[])
                mismatches=[]
                if got!=expected: mismatches.append("turn_ids")
                if expected_sigs and actual_sigs!=expected_sigs: mismatches.append("event_signatures")
                if current in got: mismatches.append("current_turn_not_excluded")
                scope.append({
                    "case_id":c["case_id"],"category":c["category"],"passed":not mismatches,
                    "expected_turn_ids":expected,"actual_turn_ids":got,
                    "expected_event_signatures":expected_sigs,"actual_event_signatures":actual_sigs,
                    "mismatches":mismatches,"latency_ms":ms,
                })
            return hist,scope,aggregate(hist,scope)
        finally:
            store.close()

def avg(xs): return statistics.fmean(xs) if xs else None

def aggregate(hist,scope):
    cats=defaultdict(list)
    for x in hist: cats[x["category"]].append(x)
    per={}
    for k,v in cats.items():
        per[k]={
            "case_count":len(v),
            "hit_at_1":avg([x["hit_at_1"] for x in v]),
            "hit_at_3":avg([x["hit_at_3"] for x in v]),
            "mrr_at_3":avg([x["mrr_at_3"] for x in v]),
            "recall_at_3":avg([x["recall_at_3"] for x in v]),
        }
    lat=[x["latency_ms"] for x in hist+scope]
    return {
        "historical_case_count":len(hist),
        "scope_case_count":len(scope),
        "historical":{
            "hit_at_1":avg([x["hit_at_1"] for x in hist]),
            "hit_at_3":avg([x["hit_at_3"] for x in hist]),
            "mrr_at_3":avg([x["mrr_at_3"] for x in hist]),
            "recall_at_3":avg([x["recall_at_3"] for x in hist]),
            "no_result_count":sum(not x["retrieved_turn_ids"] for x in hist),
            "per_category":per,
        },
        "scope":{
            "pass_count":sum(x["passed"] for x in scope),
            "pass_rate":avg([float(x["passed"]) for x in scope]),
            "current_turn_exclusion_rate":avg([float(tid("t31") not in x["actual_turn_ids"]) for x in scope]),
        },
        "latency_ms":{"mean":avg(lat),"max":max(lat) if lat else None},
    }
