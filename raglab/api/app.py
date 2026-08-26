"""HTTP adapter for the RAGLab secure Agent runtime.

Architecture: HTTP -> FastAPI -> SecureAgentRuntime -> Agent -> LangGraph.
This module validates transport data and maps existing runtime results; Agent,
memory, skill, scheduler, and security behavior stays in its owning layer.
"""

from __future__ import annotations

import logging
import json
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from raglab.agent.conversation_event_store import ConversationEventStore
from raglab.application.secure_agent_factory import build_secure_agent
from raglab.control.runtime_security import SecureAgentRuntime
from raglab.observability.runtime_events import (
    RuntimeEventCallback,
    bind_runtime_event_callback,
    emit_runtime_event,
)
from raglab.settings import CONFIG_DIR

logger = logging.getLogger("raglab.api")
DEFAULT_CONFIG_PATH = CONFIG_DIR / "agent.yaml"
STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass
class AgentApiRuntime:
    """Process-wide Runtime；执行锁按 thread_id 隔离。"""

    runtime: SecureAgentRuntime
    # 保留该字段兼容现有构造代码；新请求不再用它串行化所有用户。
    execution_lock: threading.RLock
    _thread_locks: dict[str, threading.RLock] = field(default_factory=dict)
    _thread_locks_guard: threading.RLock = field(default_factory=threading.RLock)

    def lock_for(self, thread_id: str) -> threading.RLock:
        normalized = str(thread_id).strip()
        with self._thread_locks_guard:
            lock = self._thread_locks.get(normalized)
            if lock is None:
                lock = threading.RLock()
                self._thread_locks[normalized] = lock
            return lock


class CreateThreadResponse(BaseModel):
    thread_id: str


class ThreadSummary(BaseModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    last_message_preview: str


class ThreadListResponse(BaseModel):
    user_id: str
    threads: list[ThreadSummary]


class ConversationMessageResponse(BaseModel):
    event_id: str
    turn_id: str
    sequence_no: int
    role: str
    content: str
    created_at: str


class ThreadMessagesResponse(BaseModel):
    user_id: str
    thread_id: str
    messages: list[ConversationMessageResponse]


class UserListResponse(BaseModel):
    users: list[str]


class DeleteThreadResponse(BaseModel):
    thread_id: str
    deleted: bool


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    user_id: str = Field(default="local-user", min_length=1, max_length=200)
    thread_id: str | None = Field(default=None, max_length=200)
    include_tool_trace: bool = True


class HitlDecisionRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=200)
    user_id: str = Field(default="local-user", min_length=1, max_length=200)


class AgentExecutionStats(BaseModel):
    llm_calls: int
    tool_calls: int
    summary_calls: int
    summary_updated: bool
    stopped_by_max_steps: bool
    latency_ms: float


class RuntimeTrace(BaseModel):
    """Observable runtime details returned by every Agent execution."""

    context_pipeline: dict[str, Any]
    memory_trace: dict[str, Any]
    loaded_skills: list[str]
    model_trace: list[dict[str, Any]]
    tool_trace: list[dict[str, Any]]


class ChatResponse(BaseModel):
    request_id: str
    user_id: str
    thread_id: str
    execution_status: str
    answer: str
    stats: AgentExecutionStats
    tool_trace: list[dict[str, Any]]
    runtime_trace: RuntimeTrace
    pending_approval: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    service: str


class RuntimeStatusResponse(BaseModel):
    status: str
    active_tools: list[str]
    loaded_skills: list[str]


class PendingApprovalResponse(BaseModel):
    thread_id: str
    pending_approval: dict[str, Any] | None


def create_thread_id() -> str:
    return "session-" + uuid.uuid4().hex[:8]


def create_request_id() -> str:
    return "req-" + uuid.uuid4().hex[:12]


def normalize_text(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空。")
    return normalized


def get_runtime(request: Request) -> AgentApiRuntime:
    runtime = getattr(request.app.state, "agent_runtime", None)
    if not isinstance(runtime, AgentApiRuntime):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent Runtime 尚未初始化。",
        )
    return runtime


def get_conversation_store(request: Request) -> ConversationEventStore:
    store = getattr(get_runtime(request).runtime, "conversation_event_store", None)
    if not isinstance(store, ConversationEventStore):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation Event Store 尚未初始化。",
        )
    return store


def _runtime_trace(runtime: SecureAgentRuntime, result: Any) -> RuntimeTrace:
    final_state = getattr(result, "final_state", None)
    final_state = final_state if isinstance(final_state, dict) else {}
    context_pipeline = final_state.get("context_pipeline", {})
    memory_trace = final_state.get(
        "memory_trace",
        final_state.get("memory", {}),
    )
    return RuntimeTrace(
        context_pipeline=context_pipeline if isinstance(context_pipeline, dict) else {},
        memory_trace=memory_trace if isinstance(memory_trace, dict) else {},
        loaded_skills=list(runtime.get_loaded_skill_ids()),
        model_trace=list(getattr(result, "model_trace", []) or []),
        tool_trace=list(getattr(result, "tool_trace", []) or []),
    )


def _is_runtime_conflict(error: RuntimeError) -> bool:
    """Identify expected thread/HITL state conflicts without hiding other faults."""

    message = str(error)
    return (
        "等待中的 Tool Approval" in message
        or "没有等待中的 HITL interrupt" in message
    )


def _to_response(
    *, request_id: str, user_id: str, thread_id: str, result: Any,
    runtime: SecureAgentRuntime, include_tool_trace: bool = True,
) -> ChatResponse:
    tool_trace = list(getattr(result, "tool_trace", []) or [])
    return ChatResponse(
        request_id=request_id,
        user_id=user_id,
        thread_id=thread_id,
        execution_status="completed",
        answer=str(getattr(result, "answer", "")),
        stats=AgentExecutionStats(
            llm_calls=int(getattr(result, "turn_llm_call_count", 0)),
            tool_calls=int(getattr(result, "turn_tool_call_count", 0)),
            summary_calls=int(getattr(result, "turn_summary_call_count", 0)),
            summary_updated=bool(getattr(result, "summary_updated", False)),
            stopped_by_max_steps=bool(getattr(result, "stopped_by_max_steps", False)),
            latency_ms=float(getattr(result, "total_latency_ms", 0.0)),
        ),
        tool_trace=tool_trace if include_tool_trace else [],
        runtime_trace=_runtime_trace(runtime, result),
    )


def _pending_response(
    *,
    request_id: str,
    user_id: str,
    thread_id: str,
    runtime: SecureAgentRuntime,
    pending_approval: dict[str, Any],
) -> ChatResponse:
    """Map a normal LangGraph HITL pause to a successful API response."""

    return ChatResponse(
        request_id=request_id,
        user_id=user_id,
        thread_id=thread_id,
        execution_status="pending_approval",
        answer="高风险 Tool 调用已暂停，等待人工审批。",
        stats=AgentExecutionStats(
            llm_calls=0,
            tool_calls=0,
            summary_calls=0,
            summary_updated=False,
            stopped_by_max_steps=False,
            latency_ms=0.0,
        ),
        tool_trace=[],
        runtime_trace=RuntimeTrace(
            context_pipeline={},
            memory_trace={},
            loaded_skills=list(runtime.get_loaded_skill_ids()),
            model_trace=[],
            tool_trace=[],
        ),
        pending_approval=pending_approval,
    )


def _execute(
    *, api_runtime: AgentApiRuntime, question: str, thread_id: str,
    user_id: str, include_tool_trace: bool = True,
    event_callback: RuntimeEventCallback | None = None,
) -> ChatResponse:
    request_id = create_request_id()
    logger.info("Agent request started. request_id=%s user_id=%s thread_id=%s", request_id, user_id, thread_id)
    try:
        with bind_runtime_event_callback(event_callback, thread_id=thread_id):
            emit_runtime_event(
                "runtime_waiting",
                {"thread_id": thread_id, "message": "正在等待当前会话的执行锁。"},
            )
            with api_runtime.lock_for(thread_id):
                emit_runtime_event(
                    "runtime_acquired",
                    {"thread_id": thread_id, "message": "已获得当前会话执行锁。"},
                )
                result = api_runtime.runtime.run(
                    question, thread_id=thread_id, user_id=user_id
                )
                pending = api_runtime.runtime.get_pending_approval(thread_id)
    except RuntimeError as error:
        if "聊天模型返回了空答案" in str(error):
            with api_runtime.lock_for(thread_id):
                pending = api_runtime.runtime.get_pending_approval(thread_id)
            if pending is not None:
                logger.info(
                    "Agent request paused for approval. request_id=%s user_id=%s thread_id=%s",
                    request_id, user_id, thread_id,
                )
                return _pending_response(
                    request_id=request_id,
                    user_id=user_id,
                    thread_id=thread_id,
                    runtime=api_runtime.runtime,
                    pending_approval=pending,
                )
        if _is_runtime_conflict(error):
            logger.warning(
                "Agent request conflicted. request_id=%s user_id=%s thread_id=%s error=%s",
                request_id, user_id, thread_id, error,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"message": str(error), "request_id": request_id},
            ) from error
        logger.exception("Agent request failed. request_id=%s user_id=%s thread_id=%s", request_id, user_id, thread_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Agent 执行失败。", "request_id": request_id},
        ) from error
    except Exception as error:
        logger.exception("Agent request failed. request_id=%s user_id=%s thread_id=%s", request_id, user_id, thread_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Agent 执行失败。", "request_id": request_id},
        ) from error
    if pending is not None:
        logger.info(
            "Agent request paused for approval. request_id=%s user_id=%s thread_id=%s",
            request_id, user_id, thread_id,
        )
        return _pending_response(
            request_id=request_id,
            user_id=user_id,
            thread_id=thread_id,
            runtime=api_runtime.runtime,
            pending_approval=pending,
        )
    logger.info(
        "Agent request completed. request_id=%s user_id=%s thread_id=%s latency_ms=%.2f llm_calls=%d tool_calls=%d",
        request_id, user_id, thread_id,
        float(getattr(result, "total_latency_ms", 0.0)),
        int(getattr(result, "turn_llm_call_count", 0)),
        int(getattr(result, "turn_tool_call_count", 0)),
    )
    return _to_response(
        request_id=request_id, user_id=user_id, thread_id=thread_id,
        result=result, runtime=api_runtime.runtime,
        include_tool_trace=include_tool_trace,
    )


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """把 Python 数据编码成一条标准 SSE 消息。

    SSE 文本由 ``event`` 和 ``data`` 两部分组成；末尾的两个换行符
    表示这一条事件结束。这里返回字符串而不是普通 JSON Response，
    是因为 StreamingResponse 会把多个这样的字符串依次写进同一条
    HTTP 响应连接。
    """

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _chat_event_stream(
    *,
    api_runtime: AgentApiRuntime,
    question: str,
    thread_id: str,
    user_id: str,
    include_tool_trace: bool,
) -> Iterator[str]:
    """在线程中执行同步 Agent，并将 FastAPI 可确认的外围状态映射为 SSE。

    这里目前不观察 Agent 内部的 LangGraph 节点。能够确认的只有：
    Worker 已启动、Worker 尚未返回、最终 Tool Trace、HITL、结果和错误。

    这是一个生成器函数。每次执行 ``yield`` 时，会把一段数据交给
    StreamingResponse，但函数不会结束；下次 FastAPI 继续取数据时，
    函数会从上一次 yield 的下一行继续运行。
    """

    # Queue 是 Worker 线程和 SSE 生成器之间的线程安全通信通道。
    # Worker 使用 events.put(...) 放入结果；下面的 while 循环使用
    # events.get(...) 取出结果。Queue 自己不会监控 Agent。
    events: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
    run_id = "run-" + uuid.uuid4().hex[:12]
    sequence_lock = threading.Lock()
    sequence_no = 0

    def publish(event: str, data: dict[str, Any]) -> None:
        """给内部事件附加运行身份与严格递增序号后写入本请求 Queue。"""

        nonlocal sequence_no
        with sequence_lock:
            sequence_no += 1
            current_sequence = sequence_no
        events.put(
            (
                event,
                {
                    **data,
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "sequence_no": current_sequence,
                },
            )
        )

    def execute() -> None:
        """在后台线程中完成原有同步 Agent 调用。"""

        try:
            # _execute() 内部仍然调用 runtime.run()。这一步可能等待
            # LLM、检索或 Tool 很长时间，所以不能放在 SSE 生成器主循环里，
            # 否则生成器会被阻塞，页面也就收不到等待状态和心跳。
            result = _execute(
                api_runtime=api_runtime,
                question=question,
                thread_id=thread_id,
                user_id=user_id,
                include_tool_trace=include_tool_trace,
                event_callback=publish,
            )
            body = result.model_dump(mode="json")

            # tool_trace 是 Agent 完成后返回的真实轨迹，而不是工具开始时
            # 产生的实时事件。当前阶段只在执行完成后把它单独推给页面。
            tool_trace = body.get("runtime_trace", {}).get("tool_trace", [])
            if tool_trace:
                events.put(("tool_trace", {"tools": tool_trace}))

            # pending_approval 来自最新 LangGraph Checkpoint。如果执行停在
            # interrupt()，先通知页面显示批准/拒绝卡片，再发送完整结果。
            if body.get("pending_approval") is not None:
                events.put(("pending_approval", body["pending_approval"]))
            events.put(("result", body))
        except HTTPException as error:
            # StreamingResponse 开始发送后，已经不能再把响应整体改成普通
            # 409/500 JSON，因此把错误转换成 SSE 的 error 事件交给前端。
            detail = error.detail
            message = (
                detail.get("message", str(detail))
                if isinstance(detail, dict)
                else str(detail)
            )
            events.put(("error", {"status_code": error.status_code, "message": message}))
        except Exception:
            logger.exception("Streaming Agent request failed. thread_id=%s", thread_id)
            events.put(("error", {"status_code": 500, "message": "Agent 执行失败。"}))
        finally:
            # done 是内部的“流结束标记”。无论成功还是失败都必须放入，
            # 否则 SSE 循环会一直等待，浏览器连接也不会正常结束。
            events.put(("done", {}))

    # SSE 生成器负责网络输出，Worker 负责阻塞式 Agent 执行。
    # daemon=True 表示应用退出时不会因为这个后台线程阻止进程结束。
    worker = threading.Thread(
        target=execute,
        name=f"raglab-stream-{thread_id}",
        daemon=True,
    )
    started_at = time.monotonic()
    worker.start()

    # 下面两个 yield 会在 Agent 完成前立即到达浏览器。
    # yield 与 return 不同：发送数据后函数暂停，但不会结束。
    yield _sse_event(
        "accepted",
        {"run_id": run_id, "thread_id": thread_id, "message": "请求已接收。"},
    )
    yield _sse_event(
        "status",
        {"run_id": run_id, "thread_id": thread_id, "message": "正在执行 Agent…"},
    )

    while True:
        try:
            # 最多等待 Worker 两秒。若 Worker 放入了事件就立即取出；
            # 若两秒内没有消息，则进入 queue.Empty 分支发送心跳。
            event, data = events.get(timeout=2.0)
        except queue.Empty:
            # 心跳只说明 Worker 尚未返回，并不表示 Agent 当前一定正在
            # 执行某个具体节点或工具。
            yield _sse_event(
                "heartbeat",
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "elapsed_seconds": round(time.monotonic() - started_at, 1),
                    "message": "Agent 仍在执行，正在等待模型或工具返回…",
                },
            )
            continue

        # 将 Worker 放进 Queue 的事件写入同一条 HTTP 响应连接。
        yield _sse_event(event, data)
        if event == "done":
            # 生成器结束后 StreamingResponse 关闭本次 HTTP 连接。
            break


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = Path(DEFAULT_CONFIG_PATH).resolve()
    logger.info("Initializing RAGLab secure Agent Runtime. config=%s", config_path)
    app.state.agent_runtime = AgentApiRuntime(
        runtime=build_secure_agent(config_path),
        execution_lock=threading.RLock(),
    )
    logger.info("RAGLab secure Agent Runtime initialized.")
    yield
    logger.info("Shutting down RAGLab API.")
    app.state.agent_runtime = None


app = FastAPI(
    title="RAGLab AI Agent API",
    description="GitHub Intelligence、RAG、Dynamic Skill 与安全 Tool 执行服务。",
    version="0.2.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def web_ui() -> FileResponse:
    """Serve the dependency-free local Agent chat demo."""

    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="raglab-agent-api")


@app.get("/api/v1/runtime", response_model=RuntimeStatusResponse, tags=["system"])
def runtime_status(request: Request) -> RuntimeStatusResponse:
    runtime = get_runtime(request).runtime
    return RuntimeStatusResponse(
        status="ready",
        active_tools=list(runtime.get_active_tool_names()),
        loaded_skills=list(runtime.get_loaded_skill_ids()),
    )


@app.post(
    "/api/v1/threads", response_model=CreateThreadResponse,
    status_code=status.HTTP_201_CREATED, tags=["threads"],
)
def create_thread(request: Request, user_id: str = "local-user") -> CreateThreadResponse:
    try:
        normalized_user_id = normalize_text(user_id, field_name="user_id")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    thread_id = create_thread_id()
    get_conversation_store(request).ensure_thread(
        user_id=normalized_user_id,
        thread_id=thread_id,
    )
    return CreateThreadResponse(thread_id=thread_id)


@app.get(
    "/api/v1/threads", response_model=ThreadListResponse, tags=["threads"],
)
def list_threads(
    request: Request,
    user_id: str = "local-user",
    limit: int = 50,
) -> ThreadListResponse:
    try:
        normalized_user_id = normalize_text(user_id, field_name="user_id")
        threads = get_conversation_store(request).list_threads(
            user_id=normalized_user_id,
            limit=min(max(int(limit), 1), 100),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return ThreadListResponse(
        user_id=normalized_user_id,
        threads=[ThreadSummary(**thread.__dict__) for thread in threads],
    )


@app.get("/api/v1/users", response_model=UserListResponse, tags=["users"])
def list_users(request: Request, limit: int = 100) -> UserListResponse:
    try:
        users = get_conversation_store(request).list_user_ids(
            limit=min(max(int(limit), 1), 100),
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return UserListResponse(users=users)


@app.delete(
    "/api/v1/threads/{thread_id}",
    response_model=DeleteThreadResponse,
    tags=["threads"],
)
def delete_thread(
    thread_id: str,
    request: Request,
    user_id: str = "local-user",
) -> DeleteThreadResponse:
    try:
        normalized_user_id = normalize_text(user_id, field_name="user_id")
        normalized_thread_id = normalize_text(thread_id, field_name="thread_id")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    api_runtime = get_runtime(request)
    store = get_conversation_store(request)
    if store.get_thread(
        user_id=normalized_user_id,
        thread_id=normalized_thread_id,
    ) is None:
        raise HTTPException(status_code=404, detail="会话不存在或不属于当前用户。")
    clear_thread = getattr(api_runtime.runtime.base_agent, "clear_thread", None)
    if not callable(clear_thread):
        raise HTTPException(status_code=501, detail="当前 Checkpointer 不支持删除会话。")
    try:
        with api_runtime.lock_for(normalized_thread_id):
            clear_thread(normalized_thread_id)
            deleted = store.delete_thread(
                user_id=normalized_user_id,
                thread_id=normalized_thread_id,
            )
    except Exception as error:
        logger.exception("Thread deletion failed. thread_id=%s", normalized_thread_id)
        raise HTTPException(status_code=500, detail="会话删除失败。") from error
    return DeleteThreadResponse(thread_id=normalized_thread_id, deleted=deleted)


@app.get(
    "/api/v1/threads/{thread_id}/messages",
    response_model=ThreadMessagesResponse,
    tags=["threads"],
)
def list_thread_messages(
    thread_id: str,
    request: Request,
    user_id: str = "local-user",
    limit: int = 500,
) -> ThreadMessagesResponse:
    try:
        normalized_user_id = normalize_text(user_id, field_name="user_id")
        normalized_thread_id = normalize_text(thread_id, field_name="thread_id")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    store = get_conversation_store(request)
    if store.get_thread(user_id=normalized_user_id, thread_id=normalized_thread_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在或不属于当前用户。")
    events = store.list_thread_events(
        thread_id=normalized_thread_id,
        limit=min(max(int(limit), 1), 1000),
    )
    messages = [
        ConversationMessageResponse(
            event_id=event.event_id,
            turn_id=event.turn_id,
            sequence_no=event.sequence_no,
            role="user" if event.role == "human" else "assistant",
            content=event.content_text,
            created_at=event.created_at,
        )
        for event in events
        if event.role in {"human", "assistant"} and event.content_text.strip()
    ]
    return ThreadMessagesResponse(
        user_id=normalized_user_id,
        thread_id=normalized_thread_id,
        messages=messages,
    )


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["agent"])
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    try:
        question = normalize_text(payload.question, field_name="question")
        user_id = normalize_text(payload.user_id, field_name="user_id")
        thread_id = normalize_text(payload.thread_id, field_name="thread_id") if payload.thread_id else create_thread_id()
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    get_conversation_store(request).ensure_thread(
        user_id=user_id,
        thread_id=thread_id,
    )
    return _execute(
        api_runtime=get_runtime(request), question=question,
        thread_id=thread_id, user_id=user_id,
        include_tool_trace=payload.include_tool_trace,
    )


@app.post("/api/v1/chat/stream", tags=["agent"])
def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    """建立一次 POST SSE 连接并持续返回 Agent 外围运行状态。"""

    try:
        question = normalize_text(payload.question, field_name="question")
        user_id = normalize_text(payload.user_id, field_name="user_id")
        thread_id = (
            normalize_text(payload.thread_id, field_name="thread_id")
            if payload.thread_id
            else create_thread_id()
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    api_runtime = get_runtime(request)
    get_conversation_store(request).ensure_thread(
        user_id=user_id,
        thread_id=thread_id,
    )
    # 普通接口会等待 _execute() 完成后返回一个 JSON；这里把生成器交给
    # StreamingResponse。FastAPI 每从生成器取得一段 yield 数据，就立即
    # 尝试写给浏览器，因此一次请求可以收到多条事件。
    return StreamingResponse(
        _chat_event_stream(
            api_runtime=api_runtime,
            question=question,
            thread_id=thread_id,
            user_id=user_id,
            include_tool_trace=payload.include_tool_trace,
        ),
        media_type="text/event-stream",
        headers={
            # 禁止客户端或反向代理缓存/聚合流数据，否则事件可能攒到最后
            # 才一起显示，失去流式反馈的意义。
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _hitl_decision(
    payload: HitlDecisionRequest, request: Request, *, command: str,
) -> ChatResponse:
    try:
        thread_id = normalize_text(payload.thread_id, field_name="thread_id")
        user_id = normalize_text(payload.user_id, field_name="user_id")
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    return _execute(
        api_runtime=get_runtime(request), question=command,
        thread_id=thread_id, user_id=user_id,
    )


@app.post("/api/v1/approve", response_model=ChatResponse, tags=["hitl"])
def approve(payload: HitlDecisionRequest, request: Request) -> ChatResponse:
    return _hitl_decision(payload, request, command="/approve")


@app.post("/api/v1/reject", response_model=ChatResponse, tags=["hitl"])
def reject(payload: HitlDecisionRequest, request: Request) -> ChatResponse:
    return _hitl_decision(payload, request, command="/reject")


@app.get(
    "/api/v1/hitl/pending", response_model=PendingApprovalResponse, tags=["hitl"],
)
def pending_approval(thread_id: str, request: Request) -> PendingApprovalResponse:
    try:
        thread_id = normalize_text(thread_id, field_name="thread_id")
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    api_runtime = get_runtime(request)
    try:
        with api_runtime.lock_for(thread_id):
            pending = api_runtime.runtime.get_pending_approval(thread_id)
    except Exception as error:
        logger.exception("Pending approval query failed. thread_id=%s", thread_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="HITL pending 状态查询失败。",
        ) from error
    return PendingApprovalResponse(thread_id=thread_id, pending_approval=pending)
