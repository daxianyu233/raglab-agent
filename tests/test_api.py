"""FastAPI adapter contract tests.

These tests inject a stateful fake secure runtime. They exercise the HTTP
boundary without loading retrieval indexes, calling an LLM, or writing Agent
checkpoints.
"""

from __future__ import annotations

import threading
import re
import subprocess
import sys
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from raglab.api.app import AgentApiRuntime, AgentExecutionCancelled, _execute, app
from raglab.agent.conversation_event_store import ConversationEventStore
from raglab.application.agent_factory import GITHUB_DAILY_REPORT_ROUTING_PROMPT
from raglab.observability.runtime_events import (
    bind_runtime_event_callback,
    emit_runtime_event,
)
from raglab.observability.execution_control import (
    finish_execution_context,
    register_current_subprocess,
    set_current_execution_id,
    terminate_execution_subprocess,
    unregister_subprocess,
)


def make_result(
    *,
    thread_id: str,
    answer: str,
    tool_trace: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    current_tool_trace = list(tool_trace or [])
    return SimpleNamespace(
        thread_id=thread_id,
        answer=answer,
        turn_llm_call_count=2,
        turn_tool_call_count=len(current_tool_trace),
        turn_summary_call_count=1,
        summary_updated=True,
        stopped_by_max_steps=False,
        total_latency_ms=12.5,
        model_trace=[{"model": "fake-model", "status": "success"}],
        tool_trace=current_tool_trace,
        final_state={
            "context_pipeline": {"enabled": True, "plan": "fake-plan"},
            "memory": {"source": "fake-memory"},
        },
    )


class FakeSecureRuntime:
    def __init__(self, conversation_event_store: ConversationEventStore) -> None:
        self.conversation_event_store = conversation_event_store
        self.cleared_threads: list[str] = []
        self.base_agent = SimpleNamespace(clear_thread=self._clear_thread)
        self.active_tools = ["search_knowledge_base", "load_skill"]
        self.loaded_skills = ["github-intelligence-update"]
        self.pending: dict[str, dict[str, Any]] = {}
        self.turns: dict[str, list[str]] = defaultdict(list)
        self.calls: list[dict[str, str]] = []
        self.pending_query_error: Exception | None = None

    def _clear_thread(self, thread_id: str) -> None:
        self.cleared_threads.append(thread_id)
        self.pending.pop(thread_id, None)

    def get_active_tool_names(self) -> list[str]:
        return list(self.active_tools)

    def get_loaded_skill_ids(self) -> list[str]:
        return list(self.loaded_skills)

    def get_pending_approval(self, thread_id: str) -> dict[str, Any] | None:
        if self.pending_query_error is not None:
            raise self.pending_query_error
        return self.pending.get(thread_id)

    def run(self, question: str, *, thread_id: str, user_id: str) -> SimpleNamespace:
        emit_runtime_event(
            "graph_started",
            {"thread_id": thread_id, "message": "LangGraph 开始执行 Agent 循环。"},
        )
        self.calls.append(
            {"question": question, "thread_id": thread_id, "user_id": user_id}
        )

        if question in {"/approve", "/reject"}:
            if thread_id not in self.pending:
                raise RuntimeError("当前 thread 没有等待中的 HITL interrupt。")
            self.pending.pop(thread_id)
            decision = "批准" if question == "/approve" else "拒绝"
            return make_result(thread_id=thread_id, answer=f"操作已{decision}。")

        if question == "trigger-hitl":
            self.pending[thread_id] = {
                "thread_id": thread_id,
                "interrupts": [
                    {
                        "type": "TOOL_APPROVAL_REQUIRED",
                        "tool_name": "update_github_intelligence",
                        "requires_approval": True,
                        "args": {"target_date": "2026-08-22"},
                    }
                ],
            }
            raise RuntimeError("聊天模型返回了空答案。")

        if question == "trigger-hitl-with-result":
            self.pending[thread_id] = {
                "thread_id": thread_id,
                "interrupts": [
                    {
                        "type": "TOOL_APPROVAL_REQUIRED",
                        "tool_name": "update_github_intelligence",
                        "requires_approval": True,
                        "args": {"target_date": "2026-08-22"},
                    }
                ],
            }
            return make_result(thread_id=thread_id, answer="")

        if question == "runtime-conflict":
            raise RuntimeError(
                "当前 thread 存在等待中的 Tool Approval，不能继续追加新的普通消息。"
            )

        if question == "unexpected-error":
            raise ValueError("internal details must not be exposed")

        self.turns[thread_id].append(question)
        emit_runtime_event(
            "tools_started",
            {"tool_names": ["search_knowledge_base"], "message": "正在执行工具：search_knowledge_base"},
        )
        trace = [{"name": "search_knowledge_base", "status": "success"}]
        return make_result(
            thread_id=thread_id,
            answer=f"thread={thread_id}; turn={len(self.turns[thread_id])}",
            tool_trace=trace,
        )


@pytest.fixture
def fake_runtime(tmp_path) -> FakeSecureRuntime:
    return FakeSecureRuntime(ConversationEventStore(tmp_path / "conversation.sqlite3"))


@pytest.fixture
def client(fake_runtime: FakeSecureRuntime):
    previous_runtime = getattr(app.state, "agent_runtime", None)
    app.state.agent_runtime = AgentApiRuntime(
        runtime=fake_runtime,  # type: ignore[arg-type]
        execution_lock=threading.RLock(),
    )
    test_client = TestClient(app, raise_server_exceptions=False)
    yield test_client
    test_client.close()
    app.state.agent_runtime = previous_runtime
    fake_runtime.conversation_event_store.close()


def test_health(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "raglab-agent-api"}


def test_web_ui_and_static_assets_are_served(client: TestClient) -> None:
    page = client.get("/")
    script = client.get("/static/app.js")
    stylesheet = client.get("/static/styles.css")

    assert page.status_code == 200
    assert "RAGLab Agent" in page.text
    assert "/static/app.js" in page.text
    assert script.status_code == 200
    assert 'const API = "/api/v1"' in script.text
    assert "activeThreadId = null;" in script.text
    assert "sessionStorage.getItem(ACTIVE_THREAD_KEY)" in script.text
    assert "sessionStorage.setItem(ACTIVE_THREAD_KEY" in script.text
    assert "localStorage.removeItem(ACTIVE_THREAD_KEY)" in script.text
    assert "await createSession();" in script.text
    assert "let streamPending = null;" in script.text
    assert "正在提交${actionText}决定" in script.text
    assert "controller.abort(), 5000" in script.text
    assert "function updateInteractionState()" in script.text
    assert "const inputLocked = busy || pending" in script.text
    assert "elements.question.readOnly = inputLocked" in script.text
    assert "addUserMessage: false" in script.text
    assert "allowPending: true" in script.text
    assert "const pendingExecutionId = session.pending.execution_id" in script.text
    assert "session.pending = null;" in script.text
    assert "executionId: pendingExecutionId" in script.text
    assert "await Promise.all([" in script.text
    assert "refreshPending(session)" in script.text
    assert "const INTERNAL_EVENT_DISPLAY_MS = 500;" in script.text
    assert "await wait(INTERNAL_EVENT_DISPLAY_MS);" in script.text
    assert "let expectedExecutionId = null;" in script.text
    assert "async function refreshExecution" in script.text
    assert "async function pollExecution" in script.text
    assert "isExecutionActive(session?.execution)" in script.text
    assert "const hasLiveProgress" in script.text
    assert "&& !hasLiveProgress" in script.text
    assert "function executionStepLabel" in script.text
    assert "/events?user_id=" in script.text
    assert "after_sequence=" in script.text
    assert "last_sequence" in script.text
    assert "cancelCurrentExecution" in script.text
    assert "/cancel?user_id=" in script.text
    assert 'includes(execution?.status)' in script.text
    assert 'elements.sendButton.classList.toggle("hidden", recoveredRunning)' in script.text
    assert 'elements.cancelButton.classList.toggle("hidden", !recoveredRunning)' in script.text
    assert 'session?.execution?.status === "CANCELLING"' in script.text
    assert 'session.execution.status = "CANCELLING"' in script.text
    assert stylesheet.status_code == 200
    assert ".approval-card" in stylesheet.text


def test_daily_report_prompt_closes_missing_today_data_loop() -> None:
    prompt = GITHUB_DAILY_REPORT_ROUTING_PROMPT
    assert "如果今天的数据不存在" in prompt
    assert "load_skill" in prompt
    assert "update_github_intelligence" in prompt
    assert "Tool Policy 与 HITL" in prompt
    assert "批准更新后必须继续完成" in prompt


def test_runtime_status(client: TestClient) -> None:
    response = client.get("/api/v1/runtime")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "active_tools": ["search_knowledge_base", "load_skill"],
        "loaded_skills": ["github-intelligence-update"],
    }


def test_runtime_uses_one_lock_per_thread(fake_runtime: FakeSecureRuntime) -> None:
    api_runtime = AgentApiRuntime(
        runtime=fake_runtime,  # type: ignore[arg-type]
        execution_lock=threading.RLock(),
    )
    assert api_runtime.lock_for("thread-a") is api_runtime.lock_for("thread-a")
    assert api_runtime.lock_for("thread-a") is not api_runtime.lock_for("thread-b")


def test_runtime_event_routes_across_worker_thread() -> None:
    received: list[tuple[str, dict[str, Any]]] = []
    with bind_runtime_event_callback(
        lambda event, data: received.append((event, data)),
        thread_id="thread-routed",
    ):
        worker = threading.Thread(
            target=lambda: emit_runtime_event(
                "model_started",
                {"thread_id": "thread-routed", "message": "started"},
            )
        )
        worker.start()
        worker.join()
    assert received == [
        ("model_started", {"thread_id": "thread-routed", "message": "started"})
    ]


def test_runtime_endpoint_returns_503_when_runtime_is_unavailable(
    client: TestClient,
) -> None:
    current_runtime = app.state.agent_runtime
    app.state.agent_runtime = None
    try:
        response = client.get("/api/v1/runtime")
    finally:
        app.state.agent_runtime = current_runtime
    assert response.status_code == 503
    assert response.json()["detail"] == "Agent Runtime 尚未初始化。"


def test_create_thread_returns_unique_session_ids(client: TestClient) -> None:
    first = client.post("/api/v1/threads")
    second = client.post("/api/v1/threads")
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["thread_id"].startswith("session-")
    assert first.json()["thread_id"] != second.json()["thread_id"]


def test_thread_list_and_message_history_are_persistent(
    client: TestClient,
    fake_runtime: FakeSecureRuntime,
) -> None:
    created = client.post("/api/v1/threads", params={"user_id": "api-user"})
    thread_id = created.json()["thread_id"]
    store = fake_runtime.conversation_event_store
    store.append_event(
        user_id="api-user",
        thread_id=thread_id,
        turn_id="turn-1",
        event_type="message",
        role="human",
        content_text="请总结 GitHub Intelligence",
    )
    store.append_event(
        user_id="api-user",
        thread_id=thread_id,
        turn_id="turn-1",
        event_type="message",
        role="assistant",
        content_text="这是一个技术情报 Agent。",
    )
    store.append_event(
        user_id="api-user",
        thread_id=thread_id,
        turn_id="turn-1",
        event_type="message",
        role="tool",
        content_text="raw tool evidence",
    )

    thread_list = client.get("/api/v1/threads", params={"user_id": "api-user"})
    assert thread_list.status_code == 200
    summary = thread_list.json()["threads"][0]
    assert summary["thread_id"] == thread_id
    assert summary["title"] == "请总结 GitHub Intelligence"
    assert summary["message_count"] == 2

    history = client.get(
        f"/api/v1/threads/{thread_id}/messages",
        params={"user_id": "api-user"},
    )
    assert history.status_code == 200
    assert [(item["role"], item["content"]) for item in history.json()["messages"]] == [
        ("user", "请总结 GitHub Intelligence"),
        ("assistant", "这是一个技术情报 Agent。"),
    ]


def test_thread_history_is_isolated_by_user(client: TestClient) -> None:
    created = client.post("/api/v1/threads", params={"user_id": "owner"})
    thread_id = created.json()["thread_id"]
    response = client.get(
        f"/api/v1/threads/{thread_id}/messages",
        params={"user_id": "another-user"},
    )
    assert response.status_code == 404


def test_users_are_listed_from_persistent_threads(client: TestClient) -> None:
    client.post("/api/v1/threads", params={"user_id": "alice"})
    client.post("/api/v1/threads", params={"user_id": "bob"})
    response = client.get("/api/v1/users")
    assert response.status_code == 200
    assert set(response.json()["users"]) == {"alice", "bob"}


def test_delete_thread_removes_events_and_checkpoint(
    client: TestClient,
    fake_runtime: FakeSecureRuntime,
) -> None:
    created = client.post("/api/v1/threads", params={"user_id": "owner"})
    thread_id = created.json()["thread_id"]
    fake_runtime.conversation_event_store.append_event(
        user_id="owner",
        thread_id=thread_id,
        turn_id="turn-delete",
        event_type="message",
        role="human",
        content_text="需要删除的消息",
    )
    response = client.delete(
        f"/api/v1/threads/{thread_id}",
        params={"user_id": "owner"},
    )
    assert response.status_code == 200
    assert response.json() == {"thread_id": thread_id, "deleted": True}
    assert fake_runtime.cleared_threads == [thread_id]
    assert fake_runtime.conversation_event_store.count_events(thread_id=thread_id) == 0
    assert client.get("/api/v1/threads", params={"user_id": "owner"}).json()["threads"] == []


def test_delete_thread_rejects_different_user(
    client: TestClient,
    fake_runtime: FakeSecureRuntime,
) -> None:
    created = client.post("/api/v1/threads", params={"user_id": "owner"})
    thread_id = created.json()["thread_id"]
    response = client.delete(
        f"/api/v1/threads/{thread_id}",
        params={"user_id": "intruder"},
    )
    assert response.status_code == 404
    assert fake_runtime.cleared_threads == []


def test_chat_maps_result_stats_and_runtime_trace(client: TestClient) -> None:
    response = client.post(
        "/api/v1/chat",
        json={
            "question": "search docs",
            "user_id": "api-user",
            "thread_id": "thread-a",
            "include_tool_trace": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["execution_status"] == "completed"
    assert body["answer"] == "thread=thread-a; turn=1"
    assert body["stats"] == {
        "llm_calls": 2,
        "tool_calls": 1,
        "summary_calls": 1,
        "summary_updated": True,
        "stopped_by_max_steps": False,
        "latency_ms": 12.5,
    }
    assert body["tool_trace"][0]["name"] == "search_knowledge_base"
    assert body["runtime_trace"]["context_pipeline"]["enabled"] is True
    assert body["runtime_trace"]["memory_trace"] == {"source": "fake-memory"}
    assert body["runtime_trace"]["loaded_skills"] == [
        "github-intelligence-update"
    ]
    assert body["pending_approval"] is None
    assert body["request_id"].startswith("req-")
    assert body["execution_id"].startswith("exec-")

    execution = client.get(
        f"/api/v1/executions/{body['execution_id']}",
        params={"user_id": "api-user"},
    )
    assert execution.status_code == 200
    assert execution.json()["status"] == "SUCCEEDED"
    assert execution.json()["current_step"] == "completed"
    assert execution.json()["thread_id"] == "thread-a"


def test_execution_status_is_isolated_by_user(client: TestClient) -> None:
    body = client.post(
        "/api/v1/chat",
        json={
            "question": "search docs",
            "user_id": "execution-owner",
            "thread_id": "thread-owned-execution",
        },
    ).json()
    response = client.get(
        f"/api/v1/executions/{body['execution_id']}",
        params={"user_id": "another-user"},
    )
    assert response.status_code == 404


def test_running_execution_can_be_marked_for_cooperative_cancellation(
    client: TestClient,
    fake_runtime: FakeSecureRuntime,
) -> None:
    thread_id = "thread-cancel"
    execution_id = "exec-cancel-test"
    fake_runtime.conversation_event_store.ensure_thread(
        user_id="cancel-owner", thread_id=thread_id,
    )
    fake_runtime.conversation_event_store.start_execution(
        execution_id=execution_id,
        user_id="cancel-owner",
        thread_id=thread_id,
    )
    cancellation_event = app.state.agent_runtime.register_execution(execution_id)

    response = client.post(
        f"/api/v1/executions/{execution_id}/cancel",
        params={"user_id": "cancel-owner"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "CANCELLING"
    assert response.json()["current_step"] == "cancellation_requested"
    assert cancellation_event.is_set()
    audit_events = fake_runtime.conversation_event_store.list_execution_events(
        execution_id=execution_id,
    )
    assert [event.event_type for event in audit_events] == [
        "user_cancellation_requested"
    ]
    app.state.agent_runtime.finish_execution(execution_id)


def test_cancelled_execution_suppresses_stale_checkpoint_interrupt(
    client: TestClient,
    fake_runtime: FakeSecureRuntime,
) -> None:
    thread_id = "thread-stale-interrupt-after-cancel"
    execution_id = "exec-stale-interrupt-after-cancel"
    fake_runtime.conversation_event_store.ensure_thread(
        user_id="local-user", thread_id=thread_id,
    )
    fake_runtime.conversation_event_store.start_execution(
        execution_id=execution_id,
        user_id="local-user",
        thread_id=thread_id,
    )
    fake_runtime.conversation_event_store.update_execution(
        execution_id,
        status="CANCELLED",
        current_step="cancelled",
    )
    fake_runtime.pending[thread_id] = {
        "thread_id": thread_id,
        "interrupts": [{"tool_name": "update_github_intelligence"}],
    }
    response = client.get("/api/v1/hitl/pending", params={"thread_id": thread_id})

    assert response.status_code == 200
    assert response.json()["pending_approval"] is None
    assert response.json()["execution_id"] is None
    assert fake_runtime.get_pending_approval(thread_id) is None
    event_types = [
        event.event_type
        for event in fake_runtime.conversation_event_store.list_execution_events(
            execution_id=execution_id,
        )
    ]
    assert "cancel_checkpoint_resolved_late" in event_types


def test_cancellation_consumes_stale_hitl_checkpoint(
    fake_runtime: FakeSecureRuntime,
) -> None:
    thread_id = "thread-cancel-checkpoint-cleanup"
    execution_id = "exec-cancel-checkpoint-cleanup"
    fake_runtime.pending[thread_id] = {
        "thread_id": thread_id,
        "interrupts": [{"tool_name": "update_github_intelligence"}],
    }
    fake_runtime.conversation_event_store.ensure_thread(
        user_id="local-user", thread_id=thread_id,
    )
    api_runtime = AgentApiRuntime(
        runtime=fake_runtime,  # type: ignore[arg-type]
        execution_lock=threading.RLock(),
    )

    def cancel_on_next_event(_event: str, _data: dict[str, Any]) -> None:
        raise AgentExecutionCancelled()

    result = _execute(
        api_runtime=api_runtime,
        question="resume-then-cancel",
        thread_id=thread_id,
        user_id="local-user",
        execution_id=execution_id,
        event_callback=cancel_on_next_event,
    )

    assert result.execution_status == "cancelled"
    assert fake_runtime.get_pending_approval(thread_id) is None
    execution = fake_runtime.conversation_event_store.get_execution(execution_id)
    assert execution is not None
    assert execution.status == "CANCELLED"
    assert execution.current_step == "cancelled"
    event_types = [
        event.event_type
        for event in fake_runtime.conversation_event_store.list_execution_events(
            execution_id=execution_id,
        )
    ]
    assert "cancel_checkpoint_resolved" in event_types
    assert "execution_cancelled" in event_types


def test_cancellation_control_signal_is_not_swallowed_by_observability() -> None:
    def cancel_callback(_event: str, _data: dict[str, Any]) -> None:
        raise AgentExecutionCancelled()

    with bind_runtime_event_callback(cancel_callback, thread_id="cancel-thread"):
        with pytest.raises(AgentExecutionCancelled):
            emit_runtime_event("model_completed", {"thread_id": "cancel-thread"})


def test_registered_update_subprocess_can_be_terminated_immediately() -> None:
    execution_id = "exec-subprocess-cancel"
    set_current_execution_id(execution_id)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert register_current_subprocess(process) == execution_id
        assert terminate_execution_subprocess(execution_id) is True
        process.wait(timeout=5)
        assert process.returncode is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        unregister_subprocess(execution_id, process)
        finish_execution_context(execution_id)


def test_chat_stream_emits_status_tool_trace_and_result(client: TestClient) -> None:
    response = client.post(
        "/api/v1/chat/stream",
        json={"question": "search docs", "thread_id": "thread-stream"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: accepted" in response.text
    assert "event: status" in response.text
    assert "event: graph_started" in response.text
    assert "event: tools_started" in response.text
    assert '"execution_id": "exec-' in response.text
    assert '"sequence_no":' in response.text
    assert "event: tool_trace" in response.text
    assert "search_knowledge_base" in response.text
    assert "event: result" in response.text
    assert "event: done" in response.text

    execution_match = re.search(r'"execution_id": "(exec-[^"]+)"', response.text)
    assert execution_match is not None
    execution_id = execution_match.group(1)
    events = client.get(
        f"/api/v1/executions/{execution_id}/events",
        params={"user_id": "local-user", "after_sequence": 0},
    )
    assert events.status_code == 200
    event_body = events.json()["events"]
    assert [item["sequence_no"] for item in event_body] == sorted(
        item["sequence_no"] for item in event_body
    )
    assert {item["event_type"] for item in event_body} >= {
        "runtime_waiting",
        "runtime_acquired",
        "graph_started",
        "tools_started",
    }
    after_first = event_body[0]["sequence_no"]
    remaining = client.get(
        f"/api/v1/executions/{execution_id}/events",
        params={"user_id": "local-user", "after_sequence": after_first},
    ).json()["events"]
    assert all(item["sequence_no"] > after_first for item in remaining)


def test_chat_stream_emits_pending_approval(client: TestClient) -> None:
    response = client.post(
        "/api/v1/chat/stream",
        json={"question": "trigger-hitl", "thread_id": "thread-stream-hitl"},
    )
    assert response.status_code == 200
    assert "event: pending_approval" in response.text
    assert "update_github_intelligence" in response.text
    assert '"execution_status": "pending_approval"' in response.text


@pytest.mark.parametrize(
    ("command", "expected_answer"),
    [("/approve", "操作已批准。"), ("/reject", "操作已拒绝。")],
)
def test_hitl_decision_can_resume_through_chat_stream(
    client: TestClient,
    command: str,
    expected_answer: str,
) -> None:
    thread_id = f"thread-stream-{command.removeprefix('/')}"
    client.post(
        "/api/v1/chat",
        json={"question": "trigger-hitl", "thread_id": thread_id},
    )

    response = client.post(
        "/api/v1/chat/stream",
        json={"question": command, "thread_id": thread_id},
    )

    assert response.status_code == 200
    assert "event: result" in response.text
    assert expected_answer in response.text
    assert '"execution_status": "completed"' in response.text
    assert client.get(
        "/api/v1/hitl/pending", params={"thread_id": thread_id}
    ).json()["pending_approval"] is None


def test_chat_can_hide_compatibility_tool_trace(client: TestClient) -> None:
    response = client.post(
        "/api/v1/chat",
        json={
            "question": "search docs",
            "thread_id": "thread-a",
            "include_tool_trace": False,
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["tool_trace"] == []
    assert body["runtime_trace"]["tool_trace"] != []


def test_chat_preserves_turns_per_thread(client: TestClient) -> None:
    first = client.post(
        "/api/v1/chat",
        json={"question": "first", "thread_id": "thread-a"},
    )
    second = client.post(
        "/api/v1/chat",
        json={"question": "second", "thread_id": "thread-a"},
    )
    isolated = client.post(
        "/api/v1/chat",
        json={"question": "first", "thread_id": "thread-b"},
    )
    assert first.json()["answer"].endswith("turn=1")
    assert second.json()["answer"].endswith("turn=2")
    assert isolated.json()["answer"].endswith("turn=1")


def test_chat_generates_thread_and_uses_default_user(
    client: TestClient,
    fake_runtime: FakeSecureRuntime,
) -> None:
    response = client.post("/api/v1/chat", json={"question": "hello"})
    body = response.json()
    assert response.status_code == 200
    assert body["thread_id"].startswith("session-")
    assert body["user_id"] == "local-user"
    assert fake_runtime.calls[-1]["thread_id"] == body["thread_id"]
    assert fake_runtime.calls[-1]["user_id"] == "local-user"


@pytest.mark.parametrize("field", ["question", "user_id", "thread_id"])
def test_chat_rejects_whitespace_only_fields(
    client: TestClient,
    field: str,
) -> None:
    payload = {
        "question": "hello",
        "user_id": "api-user",
        "thread_id": "thread-a",
    }
    payload[field] = "   "
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 422


def test_hitl_pause_is_a_successful_chat_response(client: TestClient) -> None:
    response = client.post(
        "/api/v1/chat",
        json={
            "question": "trigger-hitl",
            "user_id": "api-user",
            "thread_id": "thread-hitl",
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["execution_status"] == "pending_approval"
    assert body["pending_approval"]["interrupts"][0]["tool_name"] == (
        "update_github_intelligence"
    )
    assert "等待人工审批" in body["answer"]


def test_hitl_pause_is_detected_when_runtime_returns_normally(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/chat",
        json={
            "question": "trigger-hitl-with-result",
            "user_id": "api-user",
            "thread_id": "thread-hitl-result",
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["execution_status"] == "pending_approval"
    assert body["pending_approval"]["interrupts"][0]["tool_name"] == (
        "update_github_intelligence"
    )


def test_pending_query_and_reject_resume_flow(client: TestClient) -> None:
    started = client.post(
        "/api/v1/chat",
        json={"question": "trigger-hitl", "thread_id": "thread-hitl"},
    )
    execution_id = started.json()["execution_id"]
    pending = client.get(
        "/api/v1/hitl/pending", params={"thread_id": "thread-hitl"}
    )
    assert pending.status_code == 200
    assert pending.json()["pending_approval"] is not None
    assert pending.json()["execution_id"] == execution_id

    rejected = client.post(
        "/api/v1/reject",
        json={"thread_id": "thread-hitl", "user_id": "local-user"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["execution_status"] == "completed"
    assert rejected.json()["answer"] == "操作已拒绝。"
    assert rejected.json()["execution_id"] == execution_id

    cleared = client.get(
        "/api/v1/hitl/pending", params={"thread_id": "thread-hitl"}
    )
    assert cleared.json()["pending_approval"] is None
    assert cleared.json()["execution_id"] is None


def test_execution_status_persists_across_hitl_resume(
    client: TestClient,
    fake_runtime: FakeSecureRuntime,
) -> None:
    thread_id = "thread-execution-status"
    paused = client.post(
        "/api/v1/chat",
        json={"question": "trigger-hitl", "thread_id": thread_id},
    ).json()
    execution_id = paused["execution_id"]
    waiting = fake_runtime.conversation_event_store.get_execution(execution_id)
    assert waiting is not None
    assert waiting.status == "WAITING_HITL"
    assert waiting.current_step == "hitl_interrupt"
    active = client.get(
        f"/api/v1/threads/{thread_id}/executions/active",
        params={"user_id": "local-user"},
    )
    assert active.status_code == 200
    assert active.json()["execution"]["execution_id"] == execution_id
    assert active.json()["execution"]["status"] == "WAITING_HITL"

    resumed = client.post(
        "/api/v1/reject",
        json={
            "thread_id": thread_id,
            "user_id": "local-user",
            "execution_id": execution_id,
        },
    ).json()
    assert resumed["execution_id"] == execution_id
    completed = fake_runtime.conversation_event_store.get_execution(execution_id)
    assert completed is not None
    assert completed.status == "SUCCEEDED"
    assert completed.current_step == "completed"
    assert completed.finished_at is not None
    active_after = client.get(
        f"/api/v1/threads/{thread_id}/executions/active",
        params={"user_id": "local-user"},
    )
    assert active_after.json()["execution"] is None


def test_approve_resumes_pending_thread(client: TestClient) -> None:
    client.post(
        "/api/v1/chat",
        json={"question": "trigger-hitl", "thread_id": "thread-hitl"},
    )
    response = client.post(
        "/api/v1/approve",
        json={"thread_id": "thread-hitl", "user_id": "local-user"},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "操作已批准。"


@pytest.mark.parametrize("endpoint", ["/api/v1/approve", "/api/v1/reject"])
def test_hitl_decision_without_pending_returns_conflict(
    client: TestClient,
    endpoint: str,
) -> None:
    response = client.post(
        endpoint,
        json={"thread_id": "thread-without-pending", "user_id": "api-user"},
    )
    assert response.status_code == 409
    assert "没有等待中的 HITL interrupt" in response.json()["detail"]["message"]
    assert response.json()["detail"]["request_id"].startswith("req-")
    assert response.json()["detail"]["execution_id"].startswith("exec-")


def test_pending_thread_rejects_new_message_with_conflict(client: TestClient) -> None:
    response = client.post(
        "/api/v1/chat",
        json={"question": "runtime-conflict", "thread_id": "thread-hitl"},
    )
    assert response.status_code == 409
    assert "Tool Approval" in response.json()["detail"]["message"]


def test_unexpected_agent_error_returns_safe_500(client: TestClient) -> None:
    response = client.post(
        "/api/v1/chat",
        json={"question": "unexpected-error", "thread_id": "thread-a"},
    )
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["message"] == "Agent 执行失败。"
    assert detail["request_id"].startswith("req-")
    assert detail["execution_id"].startswith("exec-")
    assert "internal details" not in response.text


def test_pending_query_error_returns_safe_500(
    client: TestClient,
    fake_runtime: FakeSecureRuntime,
) -> None:
    fake_runtime.pending_query_error = RuntimeError("database details")
    response = client.get(
        "/api/v1/hitl/pending", params={"thread_id": "thread-a"}
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "HITL pending 状态查询失败。"
    assert "database details" not in response.text
