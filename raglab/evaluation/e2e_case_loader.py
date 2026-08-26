from __future__ import annotations

import json

from pathlib import Path

from raglab.evaluation.models.e2e_case import (
    E2ECase,
)


def load_e2e_cases(
    path: Path,
) -> list[E2ECase]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)


    cases = []


    for item in data["cases"]:

        cases.append(

            E2ECase(

                case_id=item["case_id"],

                category=item["category"],

                user_input=item["user_input"],

                setup=item.get(
                    "setup",
                    [],
                ),

                real_data_dependencies=item.get(
                    "real_data_dependencies",
                    [],
                ),

                assertions=item.get(
                    "assertions",
                    {},
                ),
            )

        )


    return cases