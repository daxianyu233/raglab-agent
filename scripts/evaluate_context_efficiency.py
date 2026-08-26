from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from raglab.agent.context_assembler import ContextAssembler
from raglab.agent.context_budget import ContextBudgetConfig, ContextBudgetManager
from raglab.agent.context_compression import ContextBudgetExceededError, ContextCompressor
from raglab.agent.context_plan import ContextPlan
from raglab.agent.conversation_event_store import ConversationEvent
from raglab.agent.conversation_retriever import ConversationRetrievalResult, RetrievedConversationTurn
from raglab.settings import PROJECT_ROOT

DATASET = PROJECT_ROOT / "raglab" / "evaluation" / "datasets" / "context_efficiency_v1.json"
REPORTS = PROJECT_ROOT / "reports" / "evaluation"
SYSTEM_PROMPT = "You are the RAG-LAB evaluation assistant. Use only supplied context."


def event(turn_id, seq, role, content, suffix, tool_call_id=None, tool_name=None, tool_calls=None):
    payload = {"content": content, "id": f"{turn_id}:{suffix}"}
    if role == "assistant":
        payload["tool_calls"] = list(tool_calls or [])
    if role == "tool":
        payload["tool_call_id"] = tool_call_id
        payload["name"] = tool_name
    return ConversationEvent(
        event_id=f"evt:{turn_id}:{suffix}", user_id="eval-user", thread_id="eval-thread",
        turn_id=turn_id, sequence_no=seq, event_type="message", role=role,
        message_id=f"{turn_id}:{suffix}", tool_call_id=tool_call_id, tool_name=tool_name,
        content_text=content, payload=payload, metadata={"evaluation": True},
        created_at="2026-08-19T00:00:00+00:00",
    )


def answer_turn(short_id, human, answer, score=1.0):
    turn_id = f"turn:{short_id}"
    events = [
        event(turn_id, 1, "human", human, "human"),
        event(turn_id, 2, "assistant", answer, "final", tool_calls=[]),
    ]
    return RetrievedConversationTurn(turn_id, events, list(events), score, [])


def tool_turn(short_id, human, tool_result, answer, score=1.0, selection_mode="both"):
    turn_id = f"turn:{short_id}"
    call_id = f"call:{short_id}"
    tool_name = "search_knowledge_base"
    events = [
        event(turn_id, 1, "human", human, "human"),
        event(turn_id, 2, "assistant", "", "tool-call", call_id, tool_name,
              [{"id": call_id, "name": tool_name, "args": {"query": human}, "type": "tool_call"}]),
        event(turn_id, 3, "tool", tool_result, "tool", call_id, tool_name),
        event(turn_id, 4, "assistant", answer, "final", tool_calls=[]),
    ]
    selected = events[:3] if selection_mode == "raw" else ([events[0], events[3]] if selection_mode == "answer" else list(events))
    return RetrievedConversationTurn(turn_id, events, selected, score, [])


def retrieval(turns, strategy="evaluation"):
    return ConversationRetrievalResult(
        thread_id="eval-thread", history_scope="historical_search" if turns else "none",
        history_query="evaluation" if turns else None, selected_turn_ids=[x.turn_id for x in turns],
        turns=turns, selected_event_count=sum(len(x.selected_events) for x in turns),
        candidate_turn_count=len(turns), retrieval_strategy=strategy,
    )


def copy_mode(turn, mode):
    events = list(turn.events)
    if mode == "answer":
        selected = [e for e in events if e.role == "human" or (e.role == "assistant" and not e.payload.get("tool_calls"))]
    elif mode == "raw":
        selected = [e for e in events if e.role in {"human", "tool"} or (e.role == "assistant" and e.payload.get("tool_calls"))]
    else:
        selected = events
    return RetrievedConversationTurn(turn.turn_id, events, selected, turn.retrieval_score, list(turn.matched_terms))


def assemble(retrieval_result, current, summary="", ltm=""):
    return ContextAssembler().assemble(
        system_prompt=SYSTEM_PROMPT, current_messages=current,
        conversation_retrieval=retrieval_result, thread_summary=summary,
        long_term_memory_text=ltm,
    )


def selection_corpus():
    result = {}
    for i in range(1, 11):
        sid = f"t{i:02d}"
        if sid == "t05":
            result[sid] = tool_turn(
                sid, "查询一次较早的知识库原始证据。", "原始工具结果。" * 450,
                "根据工具证据给出较早结论。", selection_mode="both"
            )
        else:
            result[sid] = answer_turn(
                sid, f"第 {i} 轮用户讨论不同的 Agent 工程主题。" + "背景信息。" * 30,
                f"第 {i} 轮最终回答。" + "实现细节与结论。" * 55,
            )
    return result


def eval_selection(case):
    corpus = selection_corpus()
    full_turns = [copy_mode(corpus[f"t{i:02d}"], "both") for i in range(1, 11)]
    selected_turns = [copy_mode(corpus[sid], case.get("selection_modes", {}).get(sid, "answer")) for sid in case["selected_turns"]]
    current = [HumanMessage(content="这是当前用户请求，只需要完成本轮任务。", id=f"current:{case['case_id']}")]
    t0 = time.perf_counter()
    full = assemble(retrieval(full_turns, "full_history_baseline"), current)
    selected = assemble(retrieval(selected_turns, "selected_history") if selected_turns else None, current)
    ms = (time.perf_counter() - t0) * 1000
    full_tokens = full.estimated_message_tokens
    selected_tokens = selected.estimated_message_tokens
    saved = full_tokens - selected_tokens
    required = {f"turn:{x}" for x in case.get("expected_required_turns", [])}
    retained = set(selected.source_turn_ids)
    recall = len(required & retained) / len(required) if required else 1.0
    return {
        "case_id": case["case_id"], "category": case["category"],
        "full_history_tokens": full_tokens, "selected_context_tokens": selected_tokens,
        "tokens_saved": saved, "token_reduction_rate": saved / full_tokens if full_tokens else 0.0,
        "full_history_turn_count": len(full_turns), "selected_turn_count": len(selected.source_turn_ids),
        "required_turn_recall": recall, "tool_pair_integrity_ok": selected.tool_pair_integrity_ok,
        "latency_ms": ms,
    }


def plan():
    return ContextPlan(
        task_intent="evaluation", response_goal="evaluate compression", history_required=True,
        history_scope="historical_search", history_query="evaluation", previous_answer_required=False,
        raw_tool_evidence_required=False, external_retrieval_required=False,
        external_retrieval_allowed=False, long_term_memory_required=False,
        long_term_memory_query=None, referenced_entities=[], temporal_scope=None, confidence=1.0,
    )


def human_snapshot(messages):
    return [(getattr(m, "id", None), str(m.content)) for m in messages if isinstance(m, HumanMessage)]


def compression_fixture(scenario):
    summary = ""; ltm = ""; rr = None
    current = [HumanMessage(content="当前问题必须完整保留。", id=f"human:{scenario}")]
    required = []
    if scenario == "historical_tool":
        rr = retrieval([tool_turn("toolhist", "恢复这一轮原始工具证据。", "历史 Tool Result 大字段。" * 900,
                                  "历史最终回答。", score=1.0, selection_mode="raw")])
        required = ["turn:toolhist"]
    elif scenario == "multi_history":
        rr = retrieval([
            answer_turn("relevant", "真正需要的历史。", "关键结论。" * 120, 1.0),
            answer_turn("medium", "相关性一般的历史。", "普通历史。" * 500, 0.6),
            answer_turn("low", "低相关历史。", "低优先级历史。" * 500, 0.2),
        ])
        required = ["turn:relevant"]
    elif scenario == "current_tool":
        call_id = "call:current"
        current = [
            HumanMessage(content="请使用当前工具结果回答。", id="human:current_tool"),
            AIMessage(content="", id="ai:current_tool_call", tool_calls=[
                {"id": call_id, "name": "search_knowledge_base", "args": {"query": "evaluation"}, "type": "tool_call"}
            ]),
            ToolMessage(content="当前 Tool Result 大字段。" * 900, tool_call_id=call_id,
                        name="search_knowledge_base", id="tool:current"),
        ]
    elif scenario == "thread_summary":
        summary = "线程工作摘要。" * 1100
    elif scenario == "long_term_memory":
        ltm = "长期记忆中的稳定事实。" * 900
    elif scenario == "impossible_current_human":
        current = [HumanMessage(content="当前用户原始输入绝对不能截断。" * 650, id="human:impossible")]
    else:
        raise ValueError(scenario)
    return rr, current, summary, ltm, required


def eval_compression(case):
    rr, current, summary, ltm, fixture_required = compression_fixture(case["scenario"])
    assembly = assemble(rr, current, summary, ltm)
    config = ContextBudgetConfig(
        model_context_limit_tokens=int(case["model_context_limit_tokens"]),
        reserved_output_tokens=int(case["reserved_output_tokens"]), tool_schema_tokens=0,
        safety_margin_tokens=int(case["safety_margin_tokens"]),
    )
    manager = ContextBudgetManager()
    initial = manager.evaluate(assembly=assembly, config=config)
    before_human = human_snapshot(assembly.messages)
    compressor = ContextCompressor(budget_manager=manager)
    expected = case["expected_outcome"]
    t0 = time.perf_counter()
    try:
        result = compressor.compress_to_fit(assembly=assembly, budget_config=config, plan=plan())
        ms = (time.perf_counter() - t0) * 1000
        required = {f"turn:{x}" for x in case.get("required_turns_after", [])} | set(fixture_required)
        retained = set(result.assembly.source_turn_ids)
        human_ok = before_human == human_snapshot(result.assembly.messages)
        history_ok = required <= retained
        passed = expected == "fit" and result.final_budget.fits and result.assembly.tool_pair_integrity_ok and human_ok and history_ok
        return {
            "case_id": case["case_id"], "category": case["category"], "expected_outcome": expected,
            "actual_outcome": "fit", "passed": passed,
            "initial_message_tokens": result.initial_budget.estimated_message_tokens,
            "final_message_tokens": result.final_budget.estimated_message_tokens,
            "tokens_saved": result.tokens_saved,
            "compression_rate": result.tokens_saved / result.initial_budget.estimated_message_tokens,
            "initial_fits": initial.fits, "final_fits": result.final_budget.fits,
            "tool_pair_integrity_ok": result.assembly.tool_pair_integrity_ok,
            "current_human_preserved": human_ok, "required_history_retained": history_ok,
            "retained_turn_ids": list(result.assembly.source_turn_ids),
            "removed_turn_ids": list(result.removed_turn_ids),
            "actions": [asdict(a) for a in result.actions], "latency_ms": ms,
        }
    except ContextBudgetExceededError as exc:
        ms = (time.perf_counter() - t0) * 1000
        return {
            "case_id": case["case_id"], "category": case["category"], "expected_outcome": expected,
            "actual_outcome": "fail_closed", "passed": expected == "fail_closed",
            "initial_message_tokens": initial.estimated_message_tokens, "final_message_tokens": None,
            "tokens_saved": None, "compression_rate": None, "initial_fits": initial.fits,
            "final_fits": False, "tool_pair_integrity_ok": assembly.tool_pair_integrity_ok,
            "current_human_preserved": True, "required_history_retained": True,
            "retained_turn_ids": list(assembly.source_turn_ids), "removed_turn_ids": [],
            "actions": [], "latency_ms": ms, "error": str(exc),
        }


def avg(values):
    values = [x for x in values if x is not None]
    return statistics.fmean(values) if values else None


def aggregate(selection, compression):
    total_full = sum(x["full_history_tokens"] for x in selection)
    total_selected = sum(x["selected_context_tokens"] for x in selection)
    fit = [x for x in compression if x["expected_outcome"] == "fit"]
    fc = [x for x in compression if x["expected_outcome"] == "fail_closed"]
    latency = [x["latency_ms"] for x in selection + compression]
    return {
        "selection": {
            "case_count": len(selection),
            "mean_token_reduction_rate": avg([x["token_reduction_rate"] for x in selection]),
            "weighted_token_reduction_rate": (total_full - total_selected) / total_full,
            "total_full_history_tokens": total_full,
            "total_selected_context_tokens": total_selected,
            "required_turn_recall": avg([x["required_turn_recall"] for x in selection]),
            "tool_pair_integrity_rate": avg([float(x["tool_pair_integrity_ok"]) for x in selection]),
        },
        "compression": {
            "case_count": len(compression), "pass_rate": avg([float(x["passed"]) for x in compression]),
            "fit_success_rate": avg([float(x["actual_outcome"] == "fit" and x["final_fits"]) for x in fit]),
            "mean_compression_rate_fit_cases": avg([x["compression_rate"] for x in fit]),
            "tool_pair_integrity_rate": avg([float(x["tool_pair_integrity_ok"]) for x in compression]),
            "current_human_preservation_rate": avg([float(x["current_human_preserved"]) for x in compression]),
            "required_history_retention_rate": avg([float(x["required_history_retained"]) for x in compression]),
            "fail_closed_correct_rate": avg([float(x["actual_outcome"] == "fail_closed" and x["passed"]) for x in fc]),
        },
        "local_latency_ms": {"mean": avg(latency), "max": max(latency) if latency else None},
    }


def pct(x):
    return "N/A" if x is None else f"{x*100:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()
    d = json.loads(args.dataset.read_text(encoding="utf-8"))
    selection = [eval_selection(x) for x in d["selection_cases"]]
    compression = [eval_compression(x) for x in d["compression_cases"]]
    metrics = aggregate(selection, compression)

    print("="*88); print("Phase 8C Context Efficiency Evaluation"); print("="*88)
    print("Dataset：", args.dataset)
    print("Selection Cases：", len(selection)); print("Compression Cases：", len(compression))
    sm = metrics["selection"]
    print(); print("Selection Efficiency（estimated tokens）：")
    print("  Mean Token Reduction：", pct(sm["mean_token_reduction_rate"]))
    print("  Weighted Token Reduction：", pct(sm["weighted_token_reduction_rate"]))
    print("  Full-history Tokens Total：", sm["total_full_history_tokens"])
    print("  Selected-context Tokens Total：", sm["total_selected_context_tokens"])
    print("  Required Turn Recall：", pct(sm["required_turn_recall"]))
    print("  Tool Pair Integrity：", pct(sm["tool_pair_integrity_rate"]))
    cm = metrics["compression"]
    print(); print("Compression / Safety：")
    print("  Overall Pass：", pct(cm["pass_rate"]))
    print("  Fit Success：", pct(cm["fit_success_rate"]))
    print("  Mean Compression Rate：", pct(cm["mean_compression_rate_fit_cases"]))
    print("  Tool Pair Integrity：", pct(cm["tool_pair_integrity_rate"]))
    print("  Current Human Preservation：", pct(cm["current_human_preservation_rate"]))
    print("  Required History Retention：", pct(cm["required_history_retention_rate"]))
    print("  Fail-Closed Correct：", pct(cm["fail_closed_correct_rate"]))

    bad = [x for x in compression if not x["passed"]]
    if bad:
        print(); print("Compression Failures：")
        for x in bad:
            print(f"  [{x['case_id']}] expected={x['expected_outcome']} actual={x['actual_outcome']}")
            print("    initial：", x["initial_message_tokens"], "final：", x["final_message_tokens"])
            print("    retained：", x["retained_turn_ids"], "removed：", x["removed_turn_ids"])

    if args.show_all:
        print(); print("All Selection Cases：")
        for x in selection:
            print(f"  {x['case_id']}: full={x['full_history_tokens']}, selected={x['selected_context_tokens']}, saved={x['tokens_saved']} ({pct(x['token_reduction_rate'])}), recall={pct(x['required_turn_recall'])}")
        print(); print("All Compression Cases：")
        for x in compression:
            print(f"  {x['case_id']}: outcome={x['actual_outcome']}, pass={x['passed']}, initial={x['initial_message_tokens']}, final={x['final_message_tokens']}, saved={x['tokens_saved']}, actions={len(x['actions'])}")
            if x["removed_turn_ids"]:
                print("    removed_turns：", x["removed_turn_ids"])
            if x["actions"]:
                print("    action_types：", [a["action"] for a in x["actions"]])

    lm = metrics["local_latency_ms"]
    print(); print("Local Context Processing Latency：")
    print("  mean：", f"{lm['mean']:.3f} ms"); print("  max：", f"{lm['max']:.3f} ms")
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = REPORTS / f"context_efficiency_{time.strftime('%Y%m%d-%H%M%S')}.json"
    report.write_text(json.dumps({
        "evaluation":"context_efficiency",
        "token_note":"estimated tokens from RAG-LAB estimator; not provider-exact usage",
        "dataset":str(args.dataset), "metrics":metrics,
        "selection_results":selection, "compression_results":compression,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(); print("="*88); print("Evaluation report：", report); print("="*88)

if __name__ == "__main__":
    main()
