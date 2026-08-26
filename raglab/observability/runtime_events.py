"""Agent 运行期事件回调。

使用 ContextVar 将当前请求的回调绑定到执行上下文，避免把 callback
参数穿透 Runtime、Graph Node 和 ToolNode 的所有函数签名。未绑定回调时
emit_runtime_event() 是无操作，因此 CLI、Benchmark 和普通 API 保持原行为。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import threading
from typing import Any, Callable, Iterator


RuntimeEventCallback = Callable[[str, dict[str, Any]], None]

_current_callback: ContextVar[RuntimeEventCallback | None] = ContextVar(
    "raglab_runtime_event_callback",
    default=None,
)
_callbacks_by_thread: dict[str, RuntimeEventCallback] = {}
_callbacks_lock = threading.RLock()


@contextmanager
def bind_runtime_event_callback(
    callback: RuntimeEventCallback | None,
    *,
    thread_id: str | None = None,
) -> Iterator[None]:
    """在当前上下文及指定 thread 路由中临时安装事件回调。"""

    token = _current_callback.set(callback)
    normalized_thread_id = str(thread_id or "").strip()
    if callback is not None and normalized_thread_id:
        with _callbacks_lock:
            _callbacks_by_thread[normalized_thread_id] = callback
    try:
        yield
    finally:
        if callback is not None and normalized_thread_id:
            with _callbacks_lock:
                if _callbacks_by_thread.get(normalized_thread_id) is callback:
                    _callbacks_by_thread.pop(normalized_thread_id, None)
        _current_callback.reset(token)


def emit_runtime_event(event: str, data: dict[str, Any] | None = None) -> None:
    """向已绑定的观察者发送经过简化的运行事件。

    可观测性代码不能破坏 Agent 主流程，因此回调自身异常会被忽略。
    """

    payload = dict(data or {})
    callback = _current_callback.get()
    if callback is None:
        thread_id = str(payload.get("thread_id", "") or "").strip()
        if thread_id:
            with _callbacks_lock:
                callback = _callbacks_by_thread.get(thread_id)
    if callback is None:
        return
    try:
        callback(str(event), payload)
    except Exception:
        return
