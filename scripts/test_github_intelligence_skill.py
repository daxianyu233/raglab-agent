from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from raglab.agent.github_intelligence_skill import (
    SKILL_ID,
    execute_github_intelligence_update,
    get_github_intelligence_skill_prompt,
    get_github_intelligence_tools,
)

from raglab.agent.skill_loader import (
    discover_skills,
    render_skill_catalog,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "检查 GitHub 技术情报 Skill；"
            "默认只做静态检查，"
            "--run 才执行真实流水线。"
        )
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help=(
            "执行真实 GitHub 技术情报更新。"
            "该操作会访问 GitHub、调用 DeepSeek "
            "并重建索引。"
        ),
    )

    return parser.parse_args()


def print_json(
    value: Any,
) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> int:
    arguments = parse_arguments()

    print(
        "=" * 80
    )

    print(
        "GitHub 技术情报 Skill 检查"
    )

    print(
        "=" * 80
    )

    skills = discover_skills()

    if SKILL_ID not in skills:
        print(
            "错误：没有发现目标 Skill："
            f"{SKILL_ID}",
            file=sys.stderr,
        )

        return 1

    skill = skills[
        SKILL_ID
    ]

    print(
        f"Skill id：{skill.skill_id}"
    )

    print(
        f"名称：{skill.name}"
    )

    print(
        f"工具：{skill.tool_name}"
    )

    print(
        f"版本：{skill.version}"
    )

    print(
        f"文件：{skill.source_path}"
    )

    tools = (
        get_github_intelligence_tools()
    )

    tool_names = [
        str(
            getattr(
                tool,
                "name",
                "",
            )
        )
        for tool in tools
    ]

    if skill.tool_name not in tool_names:
        print(
            "错误：Skill 声明的工具没有注册："
            f"{skill.tool_name}",
            file=sys.stderr,
        )

        return 1

    prompt = (
        get_github_intelligence_skill_prompt()
    )

    if not prompt.strip():
        print(
            "错误：Skill 提示词为空。",
            file=sys.stderr,
        )

        return 1

    print()

    print(
        "工具注册检查：通过"
    )

    print(
        "Skill 提示词检查：通过"
    )

    print()

    print(
        "Skill 目录预览："
    )

    print(
        render_skill_catalog(
            skills.values()
        )
    )

    if not arguments.run:
        print()

        print(
            "静态检查完成。"
        )

        print(
            "本次没有执行 GitHub 获取，"
            "没有调用 DeepSeek。"
        )

        return 0

    print()

    print(
        "开始执行真实 GitHub 技术情报更新……"
    )

    result = (
        execute_github_intelligence_update()
    )

    print_json(
        result
    )

    if result.get(
        "status"
    ) == "success":
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )