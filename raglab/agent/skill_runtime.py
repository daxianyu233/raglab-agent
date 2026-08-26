from __future__ import annotations

import importlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from langchain.tools import tool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from raglab.agent.skill_loader import (
    DEFAULT_SKILLS_ROOT,
    SkillDefinition,
    discover_skills,
    render_skill_catalog,
    render_skill_instructions,
)


# ============================================================
# Skill 实现注册表
# ============================================================


# 当前 SKILL.md 负责描述：
#
#     “这个 Skill 是什么、什么时候使用、有哪些 Tool”
#
# 但是磁盘上的 Markdown 本身并不能告诉 Python：
#
#     “应该 import 哪个模块来得到这些 Tool”
#
# 因此 Runtime 需要维护一个非常轻量的实现注册表。
#
# 后面增加新 Skill 时，只需要再增加一项。
#
# 格式：
#
#     skill_id:
#         "python模块路径:返回Tool列表的函数名"
#
DEFAULT_SKILL_IMPLEMENTATIONS: dict[
    str,
    str,
] = {
    "github-intelligence-update": (
        "raglab.agent.github_intelligence_skill:"
        "get_github_intelligence_tools"
    ),
}


# ============================================================
# Tool 输入模型
# ============================================================


class LoadSkillInput(
    BaseModel
):
    """
    load_skill Tool 的输入。
    """

    skill_id: str = Field(
        description=(
            "需要加载的 Skill id。"
            "必须使用 Skill Catalog 中存在的完整 id，"
            "例如 github-intelligence-update。"
        )
    )


# ============================================================
# 已加载 Skill
# ============================================================


@dataclass(
    frozen=True,
)
class LoadedSkill:
    """
    一个已经进入当前 Agent Runtime 的 Skill。

    definition：
        从 SKILL.md 读取出的定义。

    tools：
        这个 Skill 加载后向 Agent 开放的 Tool。
    """

    definition: SkillDefinition

    tools: tuple[
        BaseTool,
        ...,
    ]


# ============================================================
# Skill Runtime
# ============================================================


class SkillRuntime:
    """
    Agent 的 Skill 运行时。

    它负责区分三个阶段：

        Discover
            ↓
        Load
            ↓
        Execute

    Discover：
        扫描 skills/*/SKILL.md，
        只知道系统里有哪些 Skill。

    Load：
        根据 skill_id 加载完整 Skill instructions，
        并解析它对应的 Python Tool。

    Execute：
        已加载 Skill 的 Tool 才会进入
        Agent 当前可调用 Tool 集合。

    注意：

    SkillRuntime 本身不负责调用 LLM，
    也不负责执行 LangGraph。

    它只负责维护：

        available_skills
        loaded_skills
        control_tools
        active_skill_tools
    """

    def __init__(
        self,
        *,
        skills_root: Path = (
            DEFAULT_SKILLS_ROOT
        ),
        implementation_registry: (
            dict[
                str,
                str,
            ]
            | None
        ) = None,
    ) -> None:

        self.skills_root = Path(
            skills_root
        ).resolve()

        self._implementation_registry = dict(
            DEFAULT_SKILL_IMPLEMENTATIONS
            if implementation_registry
            is None
            else implementation_registry
        )

        self._lock = (
            threading.RLock()
        )

        # ----------------------------------------
        # Discover
        # ----------------------------------------

        self._available_skills: dict[
            str,
            SkillDefinition,
        ] = discover_skills(
            self.skills_root
        )

        # ----------------------------------------
        # Load 状态
        # ----------------------------------------

        self._loaded_skills: dict[
            str,
            LoadedSkill,
        ] = {}

        # ----------------------------------------
        # Runtime 控制 Tool
        # ----------------------------------------

        self._control_tools = (
            self._create_control_tools()
        )

    # ========================================================
    # Discover
    # ========================================================

    def refresh(
        self,
    ) -> dict[
        str,
        SkillDefinition,
    ]:
        """
        重新扫描 skills 目录。

        主要用于开发过程中：

            新增 Skill
            修改 SKILL.md
            删除 Skill

        已加载 Skill 不会因为 refresh()
        自动卸载。

        但已经加载的 Skill 在当前 Runtime
        中继续使用加载时的定义，
        避免运行过程中配置突然变化。
        """

        discovered = discover_skills(
            self.skills_root
        )

        with self._lock:
            self._available_skills = (
                discovered
            )

        return dict(
            discovered
        )

    def available_skills(
        self,
    ) -> tuple[
        SkillDefinition,
        ...,
    ]:
        """
        返回当前发现的所有 Skill。
        """

        with self._lock:
            skills = list(
                self._available_skills
                .values()
            )

        return tuple(
            sorted(
                skills,
                key=lambda skill: (
                    skill.skill_id
                    .casefold()
                ),
            )
        )

    def available_skill_ids(
        self,
    ) -> tuple[
        str,
        ...,
    ]:
        """
        返回当前发现的 Skill id。
        """

        return tuple(
            skill.skill_id
            for skill
            in self.available_skills()
        )

    def get_skill_definition(
        self,
        skill_id: str,
    ) -> SkillDefinition:
        """
        从当前 Discover Catalog 中取得 Skill。
        """

        normalized_skill_id = str(
            skill_id
        ).strip()

        if not normalized_skill_id:
            raise ValueError(
                "skill_id 不能为空。"
            )

        with self._lock:
            skill = (
                self._available_skills
                .get(
                    normalized_skill_id
                )
            )

        if skill is not None:
            return skill

        available = (
            ", ".join(
                self.available_skill_ids()
            )
            or "无"
        )

        raise KeyError(
            "没有发现 Skill："
            f"{normalized_skill_id}；"
            "当前可用 Skill："
            f"{available}"
        )

    # ========================================================
    # Skill Python 实现解析
    # ========================================================

    def _resolve_tool_provider(
        self,
        skill_id: str,
    ) -> Callable[
        [],
        Sequence[
            BaseTool
        ],
    ]:
        """
        找到 Skill 对应的 Python Tool Provider。

        例如：

            github-intelligence-update

        对应：

            raglab.agent.github_intelligence_skill:
            get_github_intelligence_tools

        最终返回：

            get_github_intelligence_tools
        """

        provider_path = (
            self._implementation_registry
            .get(
                skill_id
            )
        )

        if not provider_path:
            raise KeyError(
                "Skill 已经被发现，但没有注册 "
                "Python 实现："
                f"{skill_id}"
            )

        if ":" not in provider_path:
            raise ValueError(
                "Skill 实现注册格式错误："
                f"{skill_id} -> "
                f"{provider_path!r}；"
                "正确格式应为 "
                "'module.path:function_name'"
            )

        (
            module_name,
            function_name,
        ) = provider_path.split(
            ":",
            1,
        )

        module_name = (
            module_name.strip()
        )

        function_name = (
            function_name.strip()
        )

        if (
            not module_name
            or not function_name
        ):
            raise ValueError(
                "Skill 实现注册格式错误："
                f"{skill_id} -> "
                f"{provider_path!r}"
            )

        try:
            module = (
                importlib.import_module(
                    module_name
                )
            )

        except ImportError as exc:
            raise ImportError(
                "无法导入 Skill 实现模块："
                f"{module_name}；"
                f"Skill：{skill_id}"
            ) from exc

        provider = getattr(
            module,
            function_name,
            None,
        )

        if not callable(
            provider
        ):
            raise TypeError(
                "Skill Tool Provider "
                "不存在或不可调用："
                f"{provider_path}"
            )

        return provider

    def _resolve_skill_tools(
        self,
        skill: SkillDefinition,
    ) -> tuple[
        BaseTool,
        ...,
    ]:
        """
        加载一个 Skill 真正对应的 Tool，
        并验证 Python 实现与 SKILL.md 声明一致。
        """

        provider = (
            self._resolve_tool_provider(
                skill.skill_id
            )
        )

        raw_tools = list(
            provider()
        )

        if not raw_tools:
            raise ValueError(
                "Skill Tool Provider "
                "没有返回任何 Tool："
                f"{skill.skill_id}"
            )

        tools: list[
            BaseTool
        ] = []

        seen_names: set[
            str
        ] = set()

        for current_tool in raw_tools:

            if not isinstance(
                current_tool,
                BaseTool,
            ):
                raise TypeError(
                    "Skill Tool Provider "
                    "只能返回 BaseTool；"
                    f"Skill：{skill.skill_id}；"
                    "实际类型："
                    f"{type(current_tool)!r}"
                )

            tool_name = str(
                current_tool.name
            ).strip()

            if not tool_name:
                raise ValueError(
                    "Skill 中存在空 Tool 名称："
                    f"{skill.skill_id}"
                )

            if tool_name in seen_names:
                raise ValueError(
                    "Skill Tool Provider "
                    "返回了重复 Tool："
                    f"{tool_name}"
                )

            seen_names.add(
                tool_name
            )

            tools.append(
                current_tool
            )

        declared_names = set(
            skill.tool_names
        )

        actual_names = set(
            seen_names
        )

        if (
            declared_names
            != actual_names
        ):
            raise ValueError(
                "SKILL.md 声明的 Tool "
                "与 Python 实现不一致。\n"
                f"Skill：{skill.skill_id}\n"
                "SKILL.md："
                f"{sorted(declared_names)}\n"
                "Python："
                f"{sorted(actual_names)}"
            )

        return tuple(
            tools
        )

    # ========================================================
    # Load
    # ========================================================

    def load(
        self,
        skill_id: str,
    ) -> LoadedSkill:
        """
        加载 Skill。

        这是 Skill 生命周期中的：

            Load

        阶段。

        加载后：

        1. 完整 SKILL.md instructions
           可以进入 Agent Prompt；

        2. Skill 专属 Tool
           可以进入 Agent Tool 候选集合。
        """

        normalized_skill_id = str(
            skill_id
        ).strip()

        if not normalized_skill_id:
            raise ValueError(
                "skill_id 不能为空。"
            )

        with self._lock:

            existing = (
                self._loaded_skills
                .get(
                    normalized_skill_id
                )
            )

            if existing is not None:
                return existing

            skill = (
                self.get_skill_definition(
                    normalized_skill_id
                )
            )

            tools = (
                self._resolve_skill_tools(
                    skill
                )
            )

            loaded = LoadedSkill(
                definition=skill,
                tools=tools,
            )

            self._loaded_skills[
                normalized_skill_id
            ] = loaded

            return loaded

    def is_loaded(
        self,
        skill_id: str,
    ) -> bool:
        """
        判断 Skill 是否已经加载。
        """

        normalized_skill_id = str(
            skill_id
        ).strip()

        with self._lock:
            return (
                normalized_skill_id
                in self._loaded_skills
            )

    def loaded_skills(
        self,
    ) -> tuple[
        LoadedSkill,
        ...,
    ]:
        """
        返回当前已经加载的 Skill。
        """

        with self._lock:
            loaded = list(
                self._loaded_skills
                .values()
            )

        return tuple(
            sorted(
                loaded,
                key=lambda item: (
                    item.definition
                    .skill_id
                    .casefold()
                ),
            )
        )

    def loaded_skill_ids(
        self,
    ) -> tuple[
        str,
        ...,
    ]:
        """
        返回所有已加载 Skill id。
        """

        return tuple(
            loaded.definition
            .skill_id
            for loaded
            in self.loaded_skills()
        )

    # ========================================================
    # Active Skill Tools
    # ========================================================

    def get_active_skill_tools(
        self,
    ) -> list[
        BaseTool
    ]:
        """
        返回已经加载 Skill 所开放的 Tool。

        未加载 Skill 的 Tool
        不会出现在这里。
        """

        tools: list[
            BaseTool
        ] = []

        seen_names: set[
            str
        ] = set()

        for loaded in (
            self.loaded_skills()
        ):
            for current_tool in (
                loaded.tools
            ):

                tool_name = str(
                    current_tool.name
                ).strip()

                if (
                    tool_name
                    in seen_names
                ):
                    raise ValueError(
                        "多个已加载 Skill "
                        "提供了同名 Tool："
                        f"{tool_name}"
                    )

                seen_names.add(
                    tool_name
                )

                tools.append(
                    current_tool
                )

        return tools

    # ========================================================
    # Prompt
    # ========================================================

    def render_catalog_prompt(
        self,
    ) -> str:
        """
        生成 Discover 阶段给 Agent 看的 Skill Catalog。
        """

        return render_skill_catalog(
            self.available_skills()
        )

    def render_loaded_instructions(
        self,
    ) -> str:
        """
        把所有已经加载 Skill 的完整 instructions
        拼接成运行时 Prompt。

        没加载的 Skill 正文不会出现在这里。
        """

        loaded = (
            self.loaded_skills()
        )

        if not loaded:
            return (
                "# 当前已加载 Skills\n\n"
                "当前没有已加载 Skill。"
            )

        blocks = [
            "# 当前已加载 Skills",
            "",
        ]

        for loaded_skill in loaded:

            blocks.append(
                render_skill_instructions(
                    loaded_skill.definition
                )
            )

            blocks.append(
                ""
            )

        return "\n\n".join(
            blocks
        ).strip()

    def render_runtime_prompt(
        self,
    ) -> str:
        """
        生成 Agent 每次模型调用需要看到的
        Skill Runtime 信息。

        包含：

        1. 可用 Skill Catalog；
        2. 当前已加载 Skill；
        3. 已加载 Skill 的完整 instructions。

        因为这个方法每轮动态执行，
        load_skill 之后下一次 Agent 节点
        就能看到新的 Skill Instructions。
        """

        return (
            "# Skill Runtime\n\n"
            "系统采用按需 Skill 加载机制。\n\n"
            "规则：\n"
            "1. Skill 出现在 Catalog 中，"
            "不代表已经加载；\n"
            "2. 用户请求明确匹配某个未加载 "
            "Skill 时，应先调用 load_skill；\n"
            "3. Skill 加载完成后，"
            "它的完整 instructions 和专属 Tool "
            "才会进入运行时；\n"
            "4. 不要绕过 load_skill 假设 "
            "未加载 Skill 的 Tool 可用；\n"
            "5. 仅查询有哪些 Skill 时，"
            "可以使用 list_skills。\n\n"
            f"{self.render_catalog_prompt()}\n\n"
            f"{self.render_loaded_instructions()}"
        ).strip()

    # ========================================================
    # Runtime 状态
    # ========================================================

    def status(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        """
        返回当前 Skill Runtime 状态。
        """

        available = []

        loaded_ids = set(
            self.loaded_skill_ids()
        )

        for skill in (
            self.available_skills()
        ):
            available.append(
                {
                    "id": (
                        skill.skill_id
                    ),
                    "name": (
                        skill.name
                    ),
                    "description": (
                        skill.description
                    ),
                    "version": (
                        skill.version
                    ),
                    "loaded": (
                        skill.skill_id
                        in loaded_ids
                    ),
                }
            )

        return {
            "skills_root": str(
                self.skills_root
            ),
            "available_count": len(
                available
            ),
            "loaded_count": len(
                loaded_ids
            ),
            "available_skills": (
                available
            ),
            "loaded_skill_ids": sorted(
                loaded_ids
            ),
            "active_skill_tools": [
                current_tool.name
                for current_tool
                in (
                    self.get_active_skill_tools()
                )
            ],
        }

    # ========================================================
    # Runtime 控制 Tools
    # ========================================================

    def _create_control_tools(
        self,
    ) -> tuple[
        BaseTool,
        BaseTool,
    ]:
        """
        创建始终提供给 Agent 的两个 Skill 控制 Tool：

            list_skills
            load_skill

        它们本身不是某个业务 Skill 的专属 Tool。

        它们属于 Skill Runtime。
        """

        runtime = self

        @tool(
            "list_skills"
        )
        def list_skills() -> str:
            """
            查看当前 Agent 可以发现的 Skill，
            以及哪些 Skill 已经加载。

            当用户询问系统有哪些 Skill、
            某个 Skill 是否可用、
            当前加载了哪些 Skill 时使用。

            本工具只查看 Skill Catalog 和加载状态，
            不执行任何业务 Skill。
            """

            return json.dumps(
                runtime.status(),
                ensure_ascii=False,
                indent=2,
            )

        @tool(
            "load_skill",
            args_schema=LoadSkillInput,
        )
        def load_skill_tool(
            skill_id: str,
        ) -> str:
            """
            按需加载一个已经存在于 Skill Catalog 中的 Skill。

            当用户请求明确匹配某个 Skill，
            但该 Skill 尚未加载时使用。

            加载成功后，该 Skill 的完整 instructions
            和专属 Tool 将在下一次 Agent 模型节点中生效。

            本工具只加载 Skill，
            不直接执行 Skill 的业务任务。
            """

            normalized_skill_id = str(
                skill_id
            ).strip()

            if not normalized_skill_id:
                return json.dumps(
                    {
                        "status": (
                            "failed"
                        ),
                        "message": (
                            "skill_id "
                            "不能为空。"
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            already_loaded = (
                runtime.is_loaded(
                    normalized_skill_id
                )
            )

            try:
                loaded = (
                    runtime.load(
                        normalized_skill_id
                    )
                )

            except Exception as exc:
                return json.dumps(
                    {
                        "status": (
                            "failed"
                        ),
                        "skill_id": (
                            normalized_skill_id
                        ),
                        "message": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                        "available_skill_ids": (
                            runtime
                            .available_skill_ids()
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            return json.dumps(
                {
                    "status": (
                        "already_loaded"
                        if already_loaded
                        else "success"
                    ),
                    "skill_id": (
                        loaded.definition
                        .skill_id
                    ),
                    "name": (
                        loaded.definition
                        .name
                    ),
                    "version": (
                        loaded.definition
                        .version
                    ),
                    "tools": [
                        current_tool.name
                        for current_tool
                        in loaded.tools
                    ],
                    "message": (
                        "Skill 已经加载，"
                        "下一次 Agent 决策时"
                        "完整 Skill instructions "
                        "和专属 Tool 将生效。"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )

        return (
            list_skills,
            load_skill_tool,
        )

    def get_control_tools(
        self,
    ) -> list[
        BaseTool
    ]:
        """
        返回 Skill Runtime 控制 Tool。

        这两个 Tool 始终存在：

            list_skills
            load_skill
        """

        return list(
            self._control_tools
        )