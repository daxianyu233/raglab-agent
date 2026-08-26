"""Phase 7D Runtime Wiring Diagnostic.

运行：
    python -m scripts.diagnose_context_runtime_wiring

只跑一个最小真实对话，用来定位：
1. build_secure_agent 实际返回什么对象；
2. Secure Runtime / Base Agent 是否共享 Event Store；
3. context_tool_policy_resolver 是否真的注入；
4. result.messages 是否可归档；
5. _archive_result_messages 实际生成了什么报告。
"""

from __future__ import annotations

import inspect
import time
import uuid

from raglab.application.secure_agent_factory import (
    build_secure_agent,
)
from raglab.settings import CONFIG_DIR


def describe_callable(value):
    if value is None:
        return None

    target = getattr(
        value,
        "__func__",
        value,
    )

    return {
        "repr": repr(value),
        "module": getattr(
            target,
            "__module__",
            None,
        ),
        "qualname": getattr(
            target,
            "__qualname__",
            None,
        ),
        "source_file": (
            inspect.getsourcefile(
                target
            )
            if callable(target)
            else None
        ),
    }


def main() -> None:
    print("=" * 88)
    print("Phase 7D Runtime Wiring Diagnostic")
    print("=" * 88)

    agent = build_secure_agent(
        (
            CONFIG_DIR
            / "agent.yaml"
        ).resolve()
    )

    base = getattr(
        agent,
        "base_agent",
        None,
    )

    print()
    print("1. 对象类型")
    print("agent type：", type(agent))
    print("agent module：", type(agent).__module__)
    print("base_agent type：", type(base))
    print(
        "base_agent module：",
        (
            type(base).__module__
            if base is not None
            else None
        ),
    )

    print()
    print("2. run() 实际来自哪里")
    print(
        "agent.run：",
        describe_callable(
            getattr(
                agent,
                "run",
                None,
            )
        ),
    )

    print(
        "base.run：",
        describe_callable(
            getattr(
                base,
                "run",
                None,
            )
        ),
    )

    print()
    print("3. Event Store 接线")

    secure_store = getattr(
        agent,
        "conversation_event_store",
        None,
    )

    base_store = getattr(
        base,
        "conversation_event_store",
        None,
    )

    pipeline = getattr(
        base,
        "context_pipeline",
        None,
    )

    pipeline_store = getattr(
        pipeline,
        "event_store",
        None,
    )

    retriever = getattr(
        pipeline,
        "retriever",
        None,
    )

    retriever_store = getattr(
        retriever,
        "store",
        None,
    )

    print(
        "secure store：",
        secure_store,
    )
    print(
        "base store：",
        base_store,
    )
    print(
        "pipeline store：",
        pipeline_store,
    )
    print(
        "retriever store：",
        retriever_store,
    )

    print(
        "secure is base：",
        secure_store
        is base_store,
    )
    print(
        "secure is pipeline：",
        secure_store
        is pipeline_store,
    )
    print(
        "secure is retriever：",
        secure_store
        is retriever_store,
    )

    print(
        "store database_path：",
        getattr(
            secure_store,
            "database_path",
            None,
        ),
    )

    print()
    print("4. Tool Policy Resolver 接线")

    resolver = getattr(
        base,
        "context_tool_policy_resolver",
        None,
    )

    print(
        "base has resolver attr：",
        hasattr(
            base,
            "context_tool_policy_resolver",
        ),
    )

    print(
        "resolver callable：",
        callable(
            resolver
        ),
    )

    print(
        "resolver：",
        describe_callable(
            resolver
        ),
    )

    print()
    print("5. 执行一个最小真实 Turn")

    suffix = (
        time.strftime(
            "%Y%m%d-%H%M%S"
        )
        + "-"
        + uuid.uuid4().hex[:6]
    )

    user_id = (
        "phase7d-diagnostic-user-"
        + suffix
    )

    thread_id = (
        "phase7d-diagnostic-thread-"
        + suffix
    )

    before = (
        secure_store
        .list_thread_events(
            thread_id=thread_id
        )
        if secure_store
        is not None
        else []
    )

    print(
        "before events：",
        len(before),
    )

    result = agent.run(
        "只回复：运行时接线诊断",
        thread_id=thread_id,
        user_id=user_id,
    )

    print()
    print("6. Result Messages")

    messages = list(
        getattr(
            result,
            "messages",
            [],
        )
        or []
    )

    print(
        "message count：",
        len(messages),
    )

    for index, message in enumerate(
        messages
    ):
        print(
            f"[{index}]",
            type(message).__name__,
            "id=",
            getattr(
                message,
                "id",
                None,
            ),
            "tool_call_id=",
            getattr(
                message,
                "tool_call_id",
                None,
            ),
            "content_preview=",
            str(
                getattr(
                    message,
                    "content",
                    "",
                )
            )[:160],
        )

    print()
    print("7. Archive Report")

    report = getattr(
        agent,
        "last_conversation_archive_report",
        None,
    )

    print(
        "report：",
        report,
    )

    if report is not None:
        try:
            print(
                "report dict：",
                vars(report),
            )
        except Exception:
            pass

    after = (
        secure_store
        .list_thread_events(
            thread_id=thread_id
        )
        if secure_store
        is not None
        else []
    )

    print()
    print(
        "after events：",
        len(after),
    )

    for event in after:
        print(
            "event：",
            getattr(
                event,
                "event_type",
                None,
            ),
            getattr(
                event,
                "role",
                None,
            ),
            getattr(
                event,
                "turn_id",
                None,
            ),
            getattr(
                event,
                "message_id",
                None,
            ),
            str(
                getattr(
                    event,
                    "content_text",
                    "",
                )
            )[:160],
        )

    print()
    print("=" * 88)
    print("Diagnostic 完成")
    print("=" * 88)


if __name__ == "__main__":
    main()