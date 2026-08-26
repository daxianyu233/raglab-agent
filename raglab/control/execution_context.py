"""External Effect 执行上下文。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import (
    ContextVar,
    Token,
)

from dataclasses import dataclass
from typing import Iterator


@dataclass(
    frozen=True,
)
class EffectExecutionContext:
    """一次 Agent Graph 执行的外部副作用上下文。"""

    thread_id: str

    user_id: str

    # normal / replay
    mode: str

    # Replay 从哪个 checkpoint 开始。
    replay_from_checkpoint_id: (
        str
        | None
    ) = None

    # Replay 开始之前，
    # 原 Branch 的 Head。
    #
    # 即使 Replay 中间 interrupt，
    # 恢复后仍然需要这个值进行
    # Branch Reconciliation。
    replay_old_head_checkpoint_id: (
        str
        | None
    ) = None


_current_context: ContextVar[
    EffectExecutionContext
    | None
] = ContextVar(
    "raglab_effect_execution_context",
    default=None,
)


def get_effect_execution_context(
) -> EffectExecutionContext | None:
    """获取当前 Effect Execution Context。"""

    return _current_context.get()


@contextmanager
def effect_execution_scope(
    *,
    thread_id: str,
    user_id: str,
    mode: str,
    replay_from_checkpoint_id: (
        str
        | None
    ) = None,
    replay_old_head_checkpoint_id: (
        str
        | None
    ) = None,
) -> Iterator[
    EffectExecutionContext
]:
    """建立一次 Graph 执行上下文。"""

    normalized_thread_id = str(
        thread_id
    ).strip()

    normalized_user_id = str(
        user_id
    ).strip()

    normalized_mode = str(
        mode
    ).strip().lower()

    if not normalized_thread_id:

        raise ValueError(
            "thread_id 不能为空。"
        )

    if not normalized_user_id:

        raise ValueError(
            "user_id 不能为空。"
        )

    if normalized_mode not in {
        "normal",
        "replay",
    }:

        raise ValueError(
            "mode 只能是 "
            "'normal' 或 'replay'。"
        )

    context = EffectExecutionContext(

        thread_id=(
            normalized_thread_id
        ),

        user_id=(
            normalized_user_id
        ),

        mode=(
            normalized_mode
        ),

        replay_from_checkpoint_id=(
            str(
                replay_from_checkpoint_id
            ).strip()
            if replay_from_checkpoint_id
            else None
        ),

        replay_old_head_checkpoint_id=(
            str(
                replay_old_head_checkpoint_id
            ).strip()
            if replay_old_head_checkpoint_id
            else None
        ),
    )

    token: Token = (
        _current_context.set(
            context
        )
    )

    try:

        yield context

    finally:

        _current_context.reset(
            token
        )