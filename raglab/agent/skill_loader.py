from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ============================================================
# 项目路径
# ============================================================


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


DEFAULT_SKILLS_ROOT = (
    PROJECT_ROOT
    / "skills"
)


# ============================================================
# Skill 数据结构
# ============================================================


@dataclass(
    frozen=True,
)
class SkillDefinition:
    """
    从一个 SKILL.md 中读取出的 Skill 定义。

    这里保存的是 Skill 的“静态定义”，
    不负责记录 Skill 当前是否已经加载。

    例如：

        skills/
        └─ github-intelligence-update/
           └─ SKILL.md

    会被解析为一个 SkillDefinition。
    """

    skill_id: str

    name: str

    description: str

    tool_names: tuple[
        str,
        ...,
    ]

    version: str

    instructions: str

    source_path: Path

    @property
    def tool_name(
        self,
    ) -> str:
        """
        兼容当前只有一个 Tool 的旧代码。

        当前 github_intelligence_skill.py 中仍然会访问：

            skill.tool_name

        因此保留这个属性。

        如果未来一个 Skill 包含多个 Tool，
        新代码应优先使用：

            skill.tool_names
        """

        if not self.tool_names:
            raise ValueError(
                f"Skill 没有声明 Tool："
                f"{self.skill_id}"
            )

        return self.tool_names[0]


# ============================================================
# 基础文本处理
# ============================================================


def _strip_quotes(
    value: str,
) -> str:
    """
    去除简单 YAML 字符串两侧的单引号或双引号。
    """

    text = str(
        value
    ).strip()

    if (
        len(text) >= 2
        and text[0] == text[-1]
        and text[0] in {
            "'",
            '"',
        }
    ):
        return text[1:-1]

    return text


def _normalize_skill_id(
    value: str,
) -> str:
    """
    规范 Skill id。
    """

    normalized = str(
        value
    ).strip()

    if not normalized:
        raise ValueError(
            "Skill id 不能为空。"
        )

    if any(
        character.isspace()
        for character in normalized
    ):
        raise ValueError(
            "Skill id 不能包含空白字符："
            f"{normalized!r}"
        )

    return normalized


def _parse_tool_names(
    metadata: dict[
        str,
        str,
    ],
    *,
    source_path: Path,
) -> tuple[
    str,
    ...,
]:
    """
    解析 Skill 所声明的 Tool。

    为了保持当前简单 Frontmatter，
    不依赖 PyYAML。

    支持两种写法。

    单 Tool：

        tool: update_github_intelligence

    多 Tool：

        tools: search_x, update_x

    如果同时存在 tool 和 tools，
    视为配置错误。
    """

    single_tool = str(
        metadata.get(
            "tool",
            "",
        )
    ).strip()

    multiple_tools = str(
        metadata.get(
            "tools",
            "",
        )
    ).strip()

    if (
        single_tool
        and multiple_tools
    ):
        raise ValueError(
            "Skill Frontmatter 不能同时声明 "
            "'tool' 和 'tools'："
            f"{source_path}"
        )

    if single_tool:
        raw_tools = [
            single_tool
        ]

    elif multiple_tools:
        raw_tools = [
            item.strip()
            for item in (
                multiple_tools.split(
                    ","
                )
            )
            if item.strip()
        ]

    else:
        raise ValueError(
            "Skill Frontmatter 必须声明 "
            "'tool' 或 'tools'："
            f"{source_path}"
        )

    normalized_tools: list[
        str
    ] = []

    seen: set[
        str
    ] = set()

    for raw_tool_name in raw_tools:
        tool_name = str(
            raw_tool_name
        ).strip()

        if not tool_name:
            continue

        if any(
            character.isspace()
            for character in tool_name
        ):
            raise ValueError(
                "Tool 名称不能包含空白字符："
                f"{tool_name!r}；"
                f"文件：{source_path}"
            )

        if tool_name in seen:
            continue

        seen.add(
            tool_name
        )

        normalized_tools.append(
            tool_name
        )

    if not normalized_tools:
        raise ValueError(
            "Skill 没有声明有效 Tool："
            f"{source_path}"
        )

    return tuple(
        normalized_tools
    )


# ============================================================
# Frontmatter 解析
# ============================================================


def _parse_frontmatter(
    text: str,
    source_path: Path,
) -> tuple[
    dict[str, str],
    str,
]:
    """
    解析 SKILL.md 顶部的简单 YAML Frontmatter。

    当前只支持：

        key: value

    不支持嵌套 YAML。

    示例：

        ---
        id: github-intelligence-update
        name: GitHub 技术情报更新
        description: 更新 GitHub 技术情报
        tool: update_github_intelligence
        version: 1.0.0
        ---

        # GitHub 技术情报更新

        ...
    """

    normalized = str(
        text
    ).replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    lines = normalized.split(
        "\n"
    )

    if (
        not lines
        or lines[0].strip()
        != "---"
    ):
        raise ValueError(
            "SKILL.md 缺少 YAML Frontmatter "
            "起始标记 '---'："
            f"{source_path}"
        )

    closing_index: (
        int
        | None
    ) = None

    for index in range(
        1,
        len(lines),
    ):
        if (
            lines[index].strip()
            == "---"
        ):
            closing_index = (
                index
            )

            break

    if closing_index is None:
        raise ValueError(
            "SKILL.md 缺少 YAML Frontmatter "
            "结束标记 '---'："
            f"{source_path}"
        )

    metadata: dict[
        str,
        str,
    ] = {}

    for raw_line in (
        lines[
            1:closing_index
        ]
    ):
        line = raw_line.strip()

        if (
            not line
            or line.startswith(
                "#"
            )
        ):
            continue

        if ":" not in line:
            raise ValueError(
                "Skill Frontmatter 当前只支持 "
                "简单 key: value 格式："
                f"{source_path}\n"
                f"错误行：{raw_line}"
            )

        key, value = (
            line.split(
                ":",
                1,
            )
        )

        normalized_key = (
            key.strip()
        )

        if not normalized_key:
            raise ValueError(
                "Skill Frontmatter "
                "出现空键："
                f"{source_path}"
            )

        if (
            normalized_key
            in metadata
        ):
            raise ValueError(
                "Skill Frontmatter "
                "出现重复字段："
                f"{normalized_key}；"
                f"文件：{source_path}"
            )

        metadata[
            normalized_key
        ] = _strip_quotes(
            value
        )

    body = "\n".join(
        lines[
            closing_index + 1:
        ]
    ).strip()

    return (
        metadata,
        body,
    )


# ============================================================
# 单个 Skill 加载
# ============================================================


def load_skill(
    skill_path: Path,
) -> SkillDefinition:
    """
    从一个 SKILL.md 中加载 SkillDefinition。
    """

    resolved_path = Path(
        skill_path
    ).resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            "Skill 文件不存在："
            f"{resolved_path}"
        )

    text = (
        resolved_path
        .read_text(
            encoding="utf-8",
        )
    )

    (
        metadata,
        instructions,
    ) = _parse_frontmatter(
        text,
        resolved_path,
    )

    required_fields = (
        "id",
        "name",
        "description",
    )

    missing_fields = [
        field
        for field
        in required_fields
        if not str(
            metadata.get(
                field,
                "",
            )
        ).strip()
    ]

    if missing_fields:
        raise ValueError(
            "Skill Frontmatter 缺少字段："
            f"{', '.join(missing_fields)}；"
            f"文件：{resolved_path}"
        )

    if not instructions:
        raise ValueError(
            "Skill 正文不能为空："
            f"{resolved_path}"
        )

    skill_id = (
        _normalize_skill_id(
            metadata[
                "id"
            ]
        )
    )

    name = str(
        metadata[
            "name"
        ]
    ).strip()

    description = str(
        metadata[
            "description"
        ]
    ).strip()

    version = (
        str(
            metadata.get(
                "version",
                "",
            )
        ).strip()
        or "1.0.0"
    )

    tool_names = (
        _parse_tool_names(
            metadata,
            source_path=(
                resolved_path
            ),
        )
    )

    return SkillDefinition(
        skill_id=skill_id,
        name=name,
        description=description,
        tool_names=tool_names,
        version=version,
        instructions=(
            instructions
        ),
        source_path=(
            resolved_path
        ),
    )


# ============================================================
# Skill 发现
# ============================================================


def discover_skills(
    skills_root: Path = (
        DEFAULT_SKILLS_ROOT
    ),
) -> dict[
    str,
    SkillDefinition,
]:
    """
    扫描：

        skills/*/SKILL.md

    返回：

        {
            skill_id: SkillDefinition,
            ...
        }

    注意：

    这里仅仅是“发现 Skill”。

    discover_skills() 不代表 Skill
    已经被 Agent 加载。
    """

    root = Path(
        skills_root
    ).resolve()

    if not root.exists():
        return {}

    if not root.is_dir():
        raise NotADirectoryError(
            "Skills 根路径不是目录："
            f"{root}"
        )

    discovered: dict[
        str,
        SkillDefinition,
    ] = {}

    skill_paths = sorted(
        root.glob(
            "*/SKILL.md"
        ),
        key=lambda path: (
            str(
                path
            ).casefold()
        ),
    )

    for skill_path in (
        skill_paths
    ):
        skill = load_skill(
            skill_path
        )

        if (
            skill.skill_id
            in discovered
        ):
            previous_path = (
                discovered[
                    skill.skill_id
                ]
                .source_path
            )

            raise ValueError(
                "发现重复的 Skill id："
                f"{skill.skill_id}\n"
                f"第一次："
                f"{previous_path}\n"
                f"第二次："
                f"{skill.source_path}"
            )

        discovered[
            skill.skill_id
        ] = skill

    return discovered


# ============================================================
# Skill 获取
# ============================================================


def get_skill(
    skill_id: str,
    skills_root: Path = (
        DEFAULT_SKILLS_ROOT
    ),
) -> SkillDefinition:
    """
    按 Skill id 获取 Skill。

    这里仍然只是读取磁盘定义，
    不表示运行时已经加载。
    """

    normalized_skill_id = (
        _normalize_skill_id(
            skill_id
        )
    )

    skills = discover_skills(
        skills_root
    )

    try:
        return skills[
            normalized_skill_id
        ]

    except KeyError as exc:
        available_ids = sorted(
            skills.keys()
        )

        available_text = (
            ", ".join(
                available_ids
            )
            if available_ids
            else "无"
        )

        raise KeyError(
            "没有找到 Skill："
            f"{normalized_skill_id}；"
            "当前可用 Skill："
            f"{available_text}"
        ) from exc


# ============================================================
# Skill Catalog
# ============================================================


def render_skill_catalog(
    skills: Iterable[
        SkillDefinition
    ],
) -> str:
    """
    生成“Skill 目录”。

    目录只给模型看：

    - Skill id；
    - 名称；
    - 简介；
    - 版本。

    不在目录阶段暴露完整 instructions，
    也不把 Skill 专属 Tool 直接暴露为
    当前可调用 Tool。

    这正是：

        Discover

    阶段。
    """

    ordered = sorted(
        skills,
        key=lambda skill: (
            skill.skill_id
            .casefold()
        ),
    )

    if not ordered:
        return (
            "# 可用 Skills\n\n"
            "当前没有发现任何 Skill。"
        )

    lines = [
        "# 可用 Skills",
        "",
        (
            "以下只是当前系统发现的 Skill 目录。"
        ),
        (
            "Skill 出现在这里，不代表它已经加载，"
            "也不代表它的专属 Tool 已经可调用。"
        ),
        (
            "当用户请求明确匹配某个 Skill 时，"
            "应先通过 Skill 加载机制加载该 Skill。"
        ),
        "",
    ]

    for skill in ordered:
        lines.extend(
            [
                (
                    f"## {skill.name}"
                ),
                (
                    f"- id："
                    f"{skill.skill_id}"
                ),
                (
                    f"- 描述："
                    f"{skill.description}"
                ),
                (
                    f"- 版本："
                    f"{skill.version}"
                ),
                "",
            ]
        )

    return "\n".join(
        lines
    ).strip()


# ============================================================
# Skill 完整 Instructions
# ============================================================


def render_skill_instructions(
    skill: SkillDefinition,
) -> str:
    """
    生成单个已加载 Skill 的完整运行时指令。

    只有真正 Load Skill 后，
    这部分内容才应该进入 Agent 上下文。
    """

    tools_text = ", ".join(
        skill.tool_names
    )

    return (
        f"# 已加载 Skill："
        f"{skill.name}\n\n"

        f"Skill id："
        f"{skill.skill_id}\n\n"

        f"版本："
        f"{skill.version}\n\n"

        f"该 Skill 可使用的专属 Tool："
        f"{tools_text}\n\n"

        f"{skill.instructions}"
    ).strip()