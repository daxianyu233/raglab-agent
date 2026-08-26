"""External Effect Ledger 管理 CLI。"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any

from raglab.application.policy_agent_factory import (
    build_agent,
)

from raglab.control.compensation import (
    ExternalEffectCompensationManager,
)

from raglab.control.external_effect import (
    ExternalEffectStatus,
)

from raglab.control.external_effect_repository import (
    ExternalEffectRepository,
)

from raglab.settings import (
    CONFIG_DIR,
)


DEFAULT_CONFIG_PATH = (
    CONFIG_DIR
    / "generation.yaml"
)


def build_parser(
) -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "查看和管理 External Effect Ledger。"
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    # --------------------------------------------------------
    # list
    # --------------------------------------------------------

    list_parser = (
        subparsers.add_parser(
            "list",
        )
    )

    list_parser.add_argument(
        "--limit",
        type=int,
        default=30,
    )

    list_parser.add_argument(
        "--thread-id",
        default=None,
    )

    list_parser.add_argument(
        "--tool",
        default=None,
    )

    list_parser.add_argument(
        "--status",
        choices=[
            item.value
            for item
            in ExternalEffectStatus
        ],
        default=None,
    )

    # --------------------------------------------------------
    # show
    # --------------------------------------------------------

    show_parser = (
        subparsers.add_parser(
            "show",
        )
    )

    show_parser.add_argument(
        "effect_id"
    )

    # --------------------------------------------------------
    # stats
    # --------------------------------------------------------

    subparsers.add_parser(
        "stats"
    )

    # --------------------------------------------------------
    # compensate
    # --------------------------------------------------------

    compensate_parser = (
        subparsers.add_parser(
            "compensate",
        )
    )

    compensate_parser.add_argument(
        "effect_id"
    )

    compensate_parser.add_argument(
        "--config",
        type=str,
        default=str(
            DEFAULT_CONFIG_PATH
        ),
    )

    return parser


def print_effect(
    effect: Any,
) -> None:

    data = effect.to_dict()

    print()
    print("=" * 80)
    print(
        "External Effect"
    )
    print("=" * 80)

    for key in (
        "effect_id",
        "thread_id",
        "user_id",
        "checkpoint_id",
        "replay_from_checkpoint_id",
        "execution_mode",
        "tool_name",
        "tool_call_id",
        "effect_type",
        "replay_policy",
        "status",
        "compensation_tool",
        "created_at",
        "execution_started_at",
        "succeeded_at",
        "compensated_at",
    ):

        print(
            f"{key}："
            f"{data.get(key)}"
        )

    print()
    print("args_json：")
    print(
        data.get(
            "args_json",
            "",
        )
    )

    print()
    print("result_text：")
    print(
        data.get(
            "result_text",
            "",
        )
        or ""
    )

    print()
    print("error_text：")
    print(
        data.get(
            "error_text",
            "",
        )
        or ""
    )

    print()
    print(
        "compensation_result_text："
    )

    print(
        data.get(
            "compensation_result_text",
            "",
        )
        or ""
    )

    print()
    print(
        "compensation_error_text："
    )

    print(
        data.get(
            "compensation_error_text",
            "",
        )
        or ""
    )


def print_effects(
    effects: list[Any],
) -> None:

    if not effects:

        print(
            "External Effect Ledger 为空。"
        )

        return

    print()

    print(
        f"{'Effect ID':34}"
        f"{'Tool':32}"
        f"{'Status':24}"
        f"{'Mode':12}"
        f"{'Thread'}"
    )

    print(
        "-" * 140
    )

    for effect in effects:

        print(
            f"{effect.effect_id:34}"
            f"{effect.tool_name:32}"
            f"{effect.status.value:24}"
            f"{effect.execution_mode:12}"
            f"{effect.thread_id}"
        )


def close_agent(
    agent: Any,
) -> None:
    """尽量关闭原 Agent SQLite Backend。"""

    backend = getattr(
        agent,
        "persistence_backend",
        None,
    )

    close_method = getattr(
        backend,
        "close",
        None,
    )

    if callable(
        close_method
    ):

        close_method()


def main(
) -> None:

    args = (
        build_parser()
        .parse_args()
    )

    repository = (
        ExternalEffectRepository()
    )

    repository.setup()

    if args.command == "list":

        effects = (
            repository.list_recent(
                limit=(
                    args.limit
                ),
                thread_id=(
                    args.thread_id
                ),
                status=(
                    args.status
                ),
                tool_name=(
                    args.tool
                ),
            )
        )

        print_effects(
            effects
        )

        return

    if args.command == "show":

        effect = repository.get(
            args.effect_id
        )

        if effect is None:

            print(
                "Effect 不存在："
                f"{args.effect_id}"
            )

            return

        print_effect(
            effect
        )

        return

    if args.command == "stats":

        print(
            json.dumps(
                repository.statistics(),
                ensure_ascii=False,
                indent=2,
            )
        )

        return

    if args.command == "compensate":

        effect = repository.get(
            args.effect_id
        )

        if effect is None:

            print(
                "Effect 不存在："
                f"{args.effect_id}"
            )

            return

        print_effect(
            effect
        )

        print()

        confirmation = input(
            "确认执行真实补偿请输入 COMPENSATE："
        ).strip()

        if confirmation != "COMPENSATE":

            print(
                "已取消补偿。"
            )

            return

        agent = build_agent(
            Path(
                args.config
            )
        )

        try:

            manager = (
                ExternalEffectCompensationManager(

                    agent=agent,

                    policy_repository=(
                        agent
                        .tool_policy_repository
                    ),

                    effect_repository=(
                        agent
                        .external_effect_repository
                    ),
                )
            )

            result = (
                manager.compensate(
                    args.effect_id
                )
            )

            print()
            print("=" * 80)
            print("Compensation 完成")
            print("=" * 80)

            print(
                "effect_id："
                f"{result.effect_id}"
            )

            print(
                "original_tool："
                f"{result.original_tool}"
            )

            print(
                "compensation_tool："
                f"{result.compensation_tool}"
            )

            print(
                "status："
                f"{result.status}"
            )

            print(
                "result："
                f"{result.result_text}"
            )

        finally:

            close_agent(
                agent
            )

        return

    raise RuntimeError(
        f"未知命令：{args.command}"
    )


if __name__ == "__main__":
    main()