"""Phase 7D：真实 DeepSeek + 正常 Secure Agent 多轮对话验收。

运行：
    python -m scripts.test_context_real_dialogue

说明：
- 使用真实 configs/agent.yaml
- 使用真实 build_secure_agent()
- 使用真实 DeepSeek Planner
- 使用真实主 Agent LLM
- 使用真实 Tool / Event Store / Policy / Context Pipeline
- 不启动 Scheduler，避免把 Scheduler 行为混入 Context 验收
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from raglab.application.secure_agent_factory import build_secure_agent
from raglab.settings import CONFIG_DIR


RETRIEVAL_TOOL_NAMES = {
    "search_knowledge_base",
    "get_github_intelligence_schema",
    "get_github_daily_report",
    "search_github_intelligence",
    "query_github_intelligence_sql",
}


def short_text(value: Any, limit: int = 500) -> str:
    text = str(value if value is not None else "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def get_plan(result: Any) -> dict[str, Any]:
    state = dict(getattr(result, "final_state", {}) or {})
    raw = state.get("context_plan", {})
    return dict(raw) if isinstance(raw, dict) else {}


def get_retrieval(result: Any) -> dict[str, Any]:
    state = dict(getattr(result, "final_state", {}) or {})
    raw = state.get("context_retrieval", {})
    return dict(raw) if isinstance(raw, dict) else {}


def get_agent_model_traces(result: Any) -> list[dict[str, Any]]:
    traces = list(getattr(result, "model_trace", []) or [])
    return [
        dict(trace)
        for trace in traces
        if isinstance(trace, dict)
        and str(trace.get("node", "")) == "agent"
    ]


def current_tool_names(result: Any) -> list[str]:
    traces = list(getattr(result, "tool_trace", []) or [])
    names: list[str] = []
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        name = str(trace.get("name", "") or "").strip()
        if name:
            names.append(name)
    return names


def latest_exposure(result: Any) -> dict[str, Any]:
    traces = get_agent_model_traces(result)
    if not traces:
        return {}
    trace = traces[-1]
    return {
        "active_tool_names": list(trace.get("active_tool_names", []) or []),
        "exposed_tool_names": list(trace.get("exposed_tool_names", []) or []),
        "hidden_tool_names": list(trace.get("hidden_tool_names", []) or []),
        "tool_exposure": dict(trace.get("tool_exposure", {}) or {}),
        "context_pipeline": dict(trace.get("context_pipeline", {}) or {}),
    }


def print_turn_report(
    *,
    title: str,
    question: str,
    result: Any,
    event_count: int,
) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)

    print("用户：")
    print(question)

    print()
    print("Agent：")
    print(short_text(getattr(result, "answer", ""), 1000))

    print()
    print("ContextPlan：")
    print(json_text(get_plan(result)))

    retrieval = get_retrieval(result)

    print()
    print("Conversation Retrieval：")
    print(
        json_text(
            {
                "history_scope": retrieval.get("history_scope"),
                "history_query": retrieval.get("history_query"),
                "selected_turn_ids": retrieval.get("selected_turn_ids", []),
                "selected_event_count": retrieval.get("selected_event_count", 0),
                "candidate_turn_count": retrieval.get("candidate_turn_count", 0),
                "retrieval_strategy": retrieval.get("retrieval_strategy"),
            }
        )
    )

    exposure = latest_exposure(result)

    print()
    print("Tool Exposure：")
    print(
        json_text(
            {
                "active": exposure.get("active_tool_names", []),
                "exposed": exposure.get("exposed_tool_names", []),
                "hidden": exposure.get("hidden_tool_names", []),
                "decision": exposure.get("tool_exposure", {}),
            }
        )
    )

    print()
    print("本轮实际 Tool Calls：", current_tool_names(result))
    print("Event Store 当前事件数：", event_count)


class Acceptance:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if condition:
            print(f"[PASS] {message}")
            return
        print(f"[FAIL] {message}")
        self.failures.append(message)


def main() -> None:
    print("=" * 88)
    print("Phase 7D：真实 DeepSeek + 正常 Secure Agent 多轮上下文验收")
    print("=" * 88)

    config_path = CONFIG_DIR / "agent.yaml"
    print("配置：", config_path)

    agent = build_secure_agent(config_path.resolve())

    suffix = (
        time.strftime("%Y%m%d-%H%M%S")
        + "-"
        + uuid.uuid4().hex[:6]
    )
    user_id = "phase7d-user-" + suffix
    thread_id = "phase7d-thread-" + suffix

    print("user_id：", user_id)
    print("thread_id：", thread_id)

    acceptance = Acceptance()

    def run_turn(title: str, question: str) -> Any:
        result = agent.run(
            question,
            thread_id=thread_id,
            user_id=user_id,
        )

        events = (
            agent
            .conversation_event_store
            .list_thread_events(thread_id=thread_id)
        )

        print_turn_report(
            title=title,
            question=question,
            result=result,
            event_count=len(events),
        )
        return result

    # Case 1
    q1 = (
        "请查询当前已经保存的 GitHub 技术情报中 ai-memory 项目的资料，"
        "概括它主要解决什么问题。"
        "这是一个已有情报查询，请使用相应的检索工具获取事实，"
        "不要仅凭模型记忆回答。"
    )
    r1 = run_turn(
        "Case 1：新事实查询 -> 应允许并实际使用 Retrieval",
        q1,
    )

    p1 = get_plan(r1)
    tools1 = set(current_tool_names(r1))

    acceptance.check(
        p1.get("external_retrieval_allowed") is True,
        "Planner 允许本轮外部 Retrieval",
    )
    acceptance.check(
        bool(tools1 & RETRIEVAL_TOOL_NAMES),
        "主 Agent 本轮实际执行了 Retrieval Tool",
    )

    # Case 2
    q2 = (
        "不要重新搜索。请直接使用你上一轮查询 ai-memory 时"
        "已经得到的原始 Tool Evidence，而不是只改写上一轮最终答案，"
        "再说明这个项目的核心价值。"
    )
    r2 = run_turn(
        "Case 2：恢复上一轮原始 Tool Evidence -> 禁止新 Retrieval",
        q2,
    )

    p2 = get_plan(r2)
    retrieval2 = get_retrieval(r2)
    exposure2 = latest_exposure(r2)
    tools2 = set(current_tool_names(r2))

    acceptance.check(
        p2.get("raw_tool_evidence_required") is True,
        "Planner 判断需要历史原始 Tool Evidence",
    )
    acceptance.check(
        p2.get("external_retrieval_allowed") is False,
        "Planner 禁止重新 Retrieval",
    )
    acceptance.check(
        bool(retrieval2.get("selected_turn_ids", [])),
        "Conversation Retriever 找回了历史 Turn",
    )
    acceptance.check(
        not (
            set(exposure2.get("exposed_tool_names", []))
            & RETRIEVAL_TOOL_NAMES
        ),
        "Retrieval Tool Schemas 没有暴露给主 LLM",
    )
    acceptance.check(
        not (tools2 & RETRIEVAL_TOOL_NAMES),
        "本轮没有实际执行新的 Retrieval Tool",
    )

    # Case 3
    q3 = (
        "把你刚才的回答压缩成三点，每点一句话。"
        "不要重新搜索。"
    )
    r3 = run_turn(
        "Case 3：改写上一轮答案 -> Previous Answer",
        q3,
    )

    p3 = get_plan(r3)
    tools3 = set(current_tool_names(r3))

    acceptance.check(
        p3.get("previous_answer_required") is True,
        "Planner 判断需要上一轮最终回答",
    )
    acceptance.check(
        p3.get("external_retrieval_allowed") is False,
        "改写任务禁止重新 Retrieval",
    )
    acceptance.check(
        not (tools3 & RETRIEVAL_TOOL_NAMES),
        "改写任务没有执行 Retrieval Tool",
    )

    # Case 4
    q4 = "把这句话翻译成英文：可靠的系统需要清晰的模块边界。"
    r4 = run_turn(
        "Case 4：完全独立任务 -> 不应依赖旧历史或 Retrieval",
        q4,
    )

    p4 = get_plan(r4)
    tools4 = set(current_tool_names(r4))

    acceptance.check(
        p4.get("history_required") is False,
        "独立翻译任务不要求会话历史",
    )
    acceptance.check(
        not (tools4 & RETRIEVAL_TOOL_NAMES),
        "独立翻译任务没有执行 Retrieval Tool",
    )

    # Filler turns
    run_turn("Filler A", "只回复：阶段七测试A")
    run_turn("Filler B", "只回复：阶段七测试B")

    # Case 5
    q5 = (
        "请从较早的会话历史中搜索我们最开始讨论 ai-memory 的那一轮。"
        "不要重新查询 GitHub 数据源。"
        "告诉我当时那轮查询的主题是什么；"
        "如果需要证据，请使用当时已经保存的历史 Tool Evidence。"
    )
    r5 = run_turn(
        "Case 5：较早历史 -> historical_search",
        q5,
    )

    p5 = get_plan(r5)
    retrieval5 = get_retrieval(r5)
    tools5 = set(current_tool_names(r5))

    acceptance.check(
        p5.get("history_scope") == "historical_search",
        "Planner 选择 historical_search",
    )
    acceptance.check(
        bool(retrieval5.get("selected_turn_ids", [])),
        "Historical Retriever 找到了较早 Turn",
    )
    acceptance.check(
        p5.get("external_retrieval_allowed") is False,
        "历史回找任务禁止新的 Retrieval",
    )
    acceptance.check(
        not (tools5 & RETRIEVAL_TOOL_NAMES),
        "历史回找期间没有重新执行 Retrieval Tool",
    )

    print()
    print("=" * 88)

    if acceptance.failures:
        print("Phase 7D 真实对话验收：存在失败项")
        for failure in acceptance.failures:
            print(" -", failure)
        print("=" * 88)
        raise AssertionError(
            "Phase 7D 有 "
            f"{len(acceptance.failures)} "
            "个验收项未通过。"
        )

    print("Context Pipeline Phase 7D 真实 Agent 对话验收通过")
    print("=" * 88)


if __name__ == "__main__":
    main()