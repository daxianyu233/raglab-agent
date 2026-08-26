from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class E2ESetupItem:
    """
    E2E 测试环境初始化配置。

    对应 json 中:
    
    setup:
    [
        {...}
    ]
    """

    role: str

    payload: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class E2EAssertion:
    """
    单个 E2E Case 的期望行为。

    不负责执行。
    只描述期望。
    """

    raw: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class E2ECase:
    """
    Full Agent E2E Benchmark Case。

    与生产 Agent 解耦。
    """

    case_id: str

    category: str

    user_input: str


    setup: tuple[E2ESetupItem, ...] = ()


    assertion: E2EAssertion = field(
        default_factory=E2EAssertion
    )


    tags: tuple[str, ...] = ()



    @staticmethod
    def from_dict(
        data: dict[str, Any]
    ) -> "E2ECase":

        setup_items = []

        for item in data.get(
            "setup",
            []
        ):

            role = item.get(
                "role",
                ""
            )

            payload = {
                k: v
                for k, v in item.items()
                if k != "role"
            }

            setup_items.append(
                E2ESetupItem(
                    role=role,
                    payload=payload,
                )
            )


        return E2ECase(

            case_id=data["case_id"],

            category=data.get(
                "category",
                ""
            ),

            user_input=data.get(
                "user_input",
                ""
            ),

            setup=tuple(
                setup_items
            ),

            assertion=E2EAssertion(

                raw=data.get(
                    "assertions",
                    {}
                )

            ),

            tags=tuple(
                data.get(
                    "tags",
                    []
                )
            ),

        )