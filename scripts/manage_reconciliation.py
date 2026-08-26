"""Replay Branch Reconciliation 管理 CLI。"""

from __future__ import annotations

import argparse

from pathlib import Path
from typing import Any

from raglab.application.policy_agent_factory import (
    build_agent,
)

from raglab.control.branch_reconciliation import (
    BranchReconciliationManager,
    ReconciliationDisposition,
)

from raglab.control.compensation import (
    ExternalEffectCompensationManager,
)

from raglab.settings import (
    CONFIG_DIR,
)


DEFAULT_CONFIG_PATH = (
    CONFIG_DIR
    / "generation.yaml"
)


# ============================================================
# Parser
# ============================================================


def build_parser(
) -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "管理 Replay Branch "
            "External Effect Reconciliation。"
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
            help="查看补偿计划。",
        )
    )

    list_parser.add_argument(
        "--thread-id",
        default=None,
    )

    list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    # --------------------------------------------------------
    # show
    # --------------------------------------------------------

    show_parser = (
        subparsers.add_parser(
            "show",
            help="查看一个计划。",
        )
    )

    show_parser.add_argument(
        "plan_id"
    )

    # --------------------------------------------------------
    # latest
    # --------------------------------------------------------

    latest_parser = (
        subparsers.add_parser(
            "latest",
            help="查看某线程最新计划。",
        )
    )

    latest_parser.add_argument(
        "--thread-id",
        required=True,
    )

    # --------------------------------------------------------
    # apply
    # --------------------------------------------------------

    apply_parser = (
        subparsers.add_parser(
            "apply",
            help="执行计划中的自动补偿项。",
        )
    )

    apply_parser.add_argument(
        "plan_id"
    )

    apply_parser.add_argument(
        "--config",
        type=str,
        default=str(
            DEFAULT_CONFIG_PATH
        ),
    )

    return parser


# ============================================================
# Output
# ============================================================


def print_plan(
    plan: Any,
) -> None:

    print()
    print("=" * 80)
    print(
        "Replay Branch Reconciliation Plan"
    )
    print("=" * 80)

    print(
        f"plan_id：{plan.plan_id}"
    )

    print(
        f"thread_id：{plan.thread_id}"
    )

    print(
        "Replay Checkpoint："
        f"{plan.replay_checkpoint_id}"
    )

    print(
        "Old Head："
        f"{plan.old_head_checkpoint_id}"
    )

    print(
        "New Head："
        f"{plan.new_head_checkpoint_id}"
    )

    print(
        f"status：{plan.status.value}"
    )

    print()

    print(
        "统计："
        f"KEEP={plan.keep_count}, "
        f"NEW={plan.new_effect_count}, "
        f"COMPENSATE={plan.compensate_count}, "
        f"MANUAL={plan.manual_review_count}"
    )

    if not plan.items:

        print()
        print(
            "该 Replay 前后没有检测到"
            "需要对账的外部 Effect。"
        )

        return

    print()
    print("-" * 120)

    for index, item in enumerate(
        plan.items,
        start=1,
    ):

        print(
            f"[{index}] "
            f"{item.tool_name}"
        )

        print(
            "    effect_id："
            f"{item.effect_id}"
        )

        print(
            "    disposition："
            f"{item.disposition.value}"
        )

        print(
            "    status："
            f"{item.status.value}"
        )

        print(
            "    原因："
            f"{item.reason}"
        )

        print()


def print_plan_list(
    plans: list[Any],
) -> None:

    if not plans:

        print(
            "没有 Reconciliation Plan。"
        )

        return

    print()

    print(
        f"{'Plan ID':34}"
        f"{'Status':12}"
        f"{'Comp':8}"
        f"{'Manual':8}"
        f"{'Thread'}"
    )

    print(
        "-" * 110
    )

    for plan in plans:

        print(
            f"{plan.plan_id:34}"
            f"{plan.status.value:12}"
            f"{plan.compensate_count:<8}"
            f"{plan.manual_review_count:<8}"
            f"{plan.thread_id}"
        )


# ============================================================
# Close
# ============================================================


def close_agent(
    agent: Any,
) -> None:

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


# ============================================================
# Main
# ============================================================


def main(
) -> None:

    args = (
        build_parser()
        .parse_args()
    )

    # --------------------------------------------------------
    # list/show/latest 不需要真正启动 LLM Agent。
    #
    # 但 Manager 需要 graph 来解析 Checkpoint；
    # 当前实现统一通过现有 Factory 构造，
    # 后面再把纯查询 Repository 单独拆开。
    # --------------------------------------------------------

    if args.command in {
        "list",
        "show",
        "latest",
        "apply",
    }:

        config_path = Path(
            getattr(
                args,
                "config",
                DEFAULT_CONFIG_PATH,
            )
        )

        agent = build_agent(
            config_path
        )

    else:

        raise RuntimeError(
            f"未知命令：{args.command}"
        )

    try:

        manager: BranchReconciliationManager = (
            agent
            .branch_reconciliation_manager
        )

        # ----------------------------------------------------
        # list
        # ----------------------------------------------------

        if args.command == "list":

            plans = (
                manager.list_plans(
                    thread_id=(
                        args.thread_id
                    ),
                    limit=(
                        args.limit
                    ),
                )
            )

            print_plan_list(
                plans
            )

            return

        # ----------------------------------------------------
        # show
        # ----------------------------------------------------

        if args.command == "show":

            plan = manager.get_plan(
                args.plan_id
            )

            if plan is None:

                print(
                    "Plan 不存在："
                    f"{args.plan_id}"
                )

                return

            print_plan(
                plan
            )

            return

        # ----------------------------------------------------
        # latest
        # ----------------------------------------------------

        if args.command == "latest":

            plans = (
                manager.list_plans(
                    thread_id=(
                        args.thread_id
                    ),
                    limit=1,
                )
            )

            if not plans:

                print(
                    "该 Thread 没有 "
                    "Reconciliation Plan。"
                )

                return

            print_plan(
                plans[0]
            )

            return

        # ----------------------------------------------------
        # apply
        # ----------------------------------------------------

        if args.command == "apply":

            plan = manager.get_plan(
                args.plan_id
            )

            if plan is None:

                print(
                    "Plan 不存在："
                    f"{args.plan_id}"
                )

                return

            print_plan(
                plan
            )

            if (
                plan.compensate_count
                == 0
            ):

                print()
                print(
                    "当前 Plan 没有可自动"
                    "执行的 COMPENSATE 项。"
                )

                if (
                    plan.manual_review_count
                    > 0
                ):

                    print(
                        "存在 MANUAL_REVIEW 项，"
                        "不能自动处理。"
                    )

                return

            print()
            print(
                "即将真正修改外部系统。"
            )

            print(
                "只有 disposition=COMPENSATE "
                "的项目会自动执行。"
            )

            print(
                "MANUAL_REVIEW "
                "不会自动执行。"
            )

            print()

            confirmation = input(
                "确认采用新 Branch 并执行补偿，"
                "请输入 APPLY："
            ).strip()

            if confirmation != "APPLY":

                print(
                    "已取消。"
                )

                return

            compensation_manager = (
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
                manager.apply_plan(

                    plan_id=(
                        args.plan_id
                    ),

                    compensation_manager=(
                        compensation_manager
                    ),
                )
            )

            print()
            print(
                "补偿执行完成。"
            )

            print_plan(
                result
            )

            return

    finally:

        close_agent(
            agent
        )


if __name__ == "__main__":
    main()