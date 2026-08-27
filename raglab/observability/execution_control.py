"""跨 Agent 线程关联 execution_id，并管理可取消的外部子进程。"""

from __future__ import annotations

import subprocess
import threading
from contextvars import ContextVar


_current_execution_id: ContextVar[str | None] = ContextVar(
    "raglab_current_execution_id", default=None,
)
_subprocesses: dict[str, subprocess.Popen[str]] = {}
_active_executions: set[str] = set()
_subprocesses_lock = threading.RLock()


def set_current_execution_id(execution_id: str) -> None:
    """在当前 Agent/Tool 工作线程中记录所属执行。"""

    normalized = str(execution_id or "").strip()
    if normalized:
        _current_execution_id.set(normalized)
        with _subprocesses_lock:
            _active_executions.add(normalized)


def register_current_subprocess(process: subprocess.Popen[str]) -> str | None:
    """将工具启动的子进程关联到当前 execution；CLI 调用时保持无操作。"""

    execution_id = _current_execution_id.get()
    if not execution_id:
        return None
    with _subprocesses_lock:
        if execution_id not in _active_executions:
            return None
        _subprocesses[execution_id] = process
    return execution_id


def unregister_subprocess(execution_id: str | None, process: subprocess.Popen[str]) -> None:
    if not execution_id:
        return
    with _subprocesses_lock:
        if _subprocesses.get(execution_id) is process:
            _subprocesses.pop(execution_id, None)


def terminate_execution_subprocess(execution_id: str) -> bool:
    """及时终止一次执行当前登记的外部流水线进程。"""

    with _subprocesses_lock:
        process = _subprocesses.get(str(execution_id).strip())
        if process is None or process.poll() is not None:
            return False
        try:
            process.terminate()
        except OSError:
            return False
        return True


def finish_execution_context(execution_id: str) -> None:
    """执行结束后清除登记，避免线程复用时使用陈旧 execution_id。"""

    normalized = str(execution_id or "").strip()
    with _subprocesses_lock:
        _active_executions.discard(normalized)
        _subprocesses.pop(normalized, None)
