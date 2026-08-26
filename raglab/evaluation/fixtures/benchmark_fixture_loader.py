from __future__ import annotations

import json

from pathlib import Path

from typing import Any


class BenchmarkFixtureLoader:
    """
    Full Agent Benchmark Fixture 加载器。

    负责：
    1. 加载 benchmark dataset
    2. 提供 case setup 信息
    3. 后续扩展测试环境注入

    不修改生产 Agent。
    """


    def __init__(
        self,
        dataset_path: Path,
    ):
        self.dataset_path = dataset_path

        self.dataset = self._load()


    def _load(
        self,
    ) -> dict[str, Any]:

        if not self.dataset_path.exists():

            raise FileNotFoundError(
                f"Benchmark dataset 不存在: "
                f"{self.dataset_path}"
            )


        with open(
            self.dataset_path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)



    def get_cases(
        self,
    ) -> list[dict[str, Any]]:

        return (
            self.dataset
            .get(
                "cases",
                []
            )
        )



    def get_case(
        self,
        case_id: str,
    ) -> dict[str, Any]:

        for case in self.get_cases():

            if (
                case.get("case_id")
                ==
                case_id
            ):

                return case


        raise KeyError(
            f"不存在测试 case: {case_id}"
        )



    def get_fixture_catalog(
        self,
    ) -> dict[str, Any]:

        return (
            self.dataset
            .get(
                "fixture_catalog",
                {}
            )
        )