"""RAGLab Agent Tool 安全策略。

本模块只负责描述：

    一个 Tool 具有怎样的副作用性质。

它暂时不负责：

1. 真正拦截 Tool；
2. Effect Ledger 持久化；
3. 幂等键生成；
4. Compensation；
5. Human-in-the-Loop。

这些能力将在后续阶段建立在本模块之上。

核心原则：

    Tool 的“安全属性”不能由 LLM 自己判断，
    而必须由系统代码显式声明。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


# ============================================================
# Tool 副作用类型
# ============================================================


class ToolEffectType(str, Enum):
    """Tool 对系统状态产生的副作用类型。"""

    # --------------------------------------------------------
    # 纯读取
    # --------------------------------------------------------
    #
    # 例如：
    #
    # search_knowledge_base
    # search_github_intelligence
    #
    # 可以在 Replay 中重新执行，
    # 不需要 Compensation。
    # --------------------------------------------------------

    READ_ONLY = "read_only"

    # --------------------------------------------------------
    # 幂等写操作
    # --------------------------------------------------------
    #
    # 重复执行不会导致最终状态继续变化。
    #
    # 例如：
    #
    # UPSERT status=active
    #
    # 执行一次：
    #
    #     status = active
    #
    # 执行十次：
    #
    #     status = active
    #
    # 注意：
    #
    # 只有真正具有幂等保证的 Tool
    # 才能使用这个类型。
    # --------------------------------------------------------

    IDEMPOTENT_WRITE = "idempotent_write"

    # --------------------------------------------------------
    # 可补偿写操作
    # --------------------------------------------------------
    #
    # 操作本身有副作用，
    # 但系统存在明确的 Compensation。
    #
    # 例如：
    #
    # create_draft()
    #
    # 对应：
    #
    # delete_draft()
    #
    # --------------------------------------------------------

    COMPENSATABLE_WRITE = (
        "compensatable_write"
    )

    # --------------------------------------------------------
    # 不可逆写操作
    # --------------------------------------------------------
    #
    # 当前系统没有可靠的自动撤销方案。
    #
    # 例如：
    #
    # send_email()
    # publish_post()
    #
    # 或者：
    #
    # 虽然理论上可以恢复，
    # 但当前工程中还没有实现
    # 幂等 / Compensation 的 Tool。
    #
    # 对这种操作采用保守策略。
    # --------------------------------------------------------

    IRREVERSIBLE_WRITE = (
        "irreversible_write"
    )


# ============================================================
# Tool Policy
# ============================================================


@dataclass(
    frozen=True
)
class ToolPolicy:
    """单个 Tool 的安全策略。"""

    # LangChain Tool name。
    tool_name: str

    # Tool 的副作用类型。
    effect_type: ToolEffectType

    # 是否修改 LangGraph State 之外的状态。
    #
    # 注意：
    #
    # 本地 SQLite / 文件
    # 也属于 Graph 外部状态。
    has_external_side_effect: bool

    # 给开发者阅读的说明。
    description: str

    @property
    def is_read_only(
        self,
    ) -> bool:
        """是否为纯只读 Tool。"""

        return (
            self.effect_type
            == ToolEffectType.READ_ONLY
        )

    @property
    def is_write(
        self,
    ) -> bool:
        """是否属于写操作。"""

        return not self.is_read_only

    @property
    def replay_safe_without_guard(
        self,
    ) -> bool:
        """Replay 时是否允许无保护重新执行。

        第一版采用保守策略：

        只有 READ_ONLY Tool
        可以完全无保护 Replay。

        即使是 IDEMPOTENT_WRITE，
        后续也应该经过 Effect Ledger /
        query-before-write 等保护层。
        """

        return self.is_read_only

    @property
    def requires_effect_record(
        self,
    ) -> bool:
        """是否应该写入 External Effect Ledger。"""

        return (
            self.has_external_side_effect
            and self.is_write
        )

    @property
    def requires_replay_guard(
        self,
    ) -> bool:
        """Replay 时是否需要额外保护。"""

        return (
            self.has_external_side_effect
            and self.is_write
        )

    @property
    def compensation_expected(
        self,
    ) -> bool:
        """该 Tool 是否声明应该存在 Compensation。"""

        return (
            self.effect_type
            == ToolEffectType.COMPENSATABLE_WRITE
        )

    @property
    def requires_human_confirmation(
        self,
    ) -> bool:
        """以后是否应该默认进入人工确认。

        当前只把不可逆写操作默认视为
        必须人工确认。

        后续可以进一步做成可配置策略。
        """

        return (
            self.effect_type
            == ToolEffectType.IRREVERSIBLE_WRITE
        )


# ============================================================
# 默认 Tool Policy Registry
# ============================================================
#
# 这一层是系统代码维护的安全策略，
# 不允许 LLM 自己声明。
#
# 动态 Skill Tool 也提前在这里声明，
# 即使它启动时还不是 Active Tool。
# ============================================================


DEFAULT_TOOL_POLICIES: dict[
    str,
    ToolPolicy,
] = {

    # --------------------------------------------------------
    # PDF RAG
    # --------------------------------------------------------

    "search_knowledge_base": ToolPolicy(

        tool_name=(
            "search_knowledge_base"
        ),

        effect_type=(
            ToolEffectType.READ_ONLY
        ),

        has_external_side_effect=False,

        description=(
            "只读取本地 PDF 知识库索引，"
            "不修改持久化状态。"
        ),
    ),

    # --------------------------------------------------------
    # GitHub Intelligence Semantic Search
    # --------------------------------------------------------

    "search_github_intelligence": ToolPolicy(

        tool_name=(
            "search_github_intelligence"
        ),

        effect_type=(
            ToolEffectType.READ_ONLY
        ),

        has_external_side_effect=False,

        description=(
            "只读取已经建立的 GitHub "
            "技术情报索引。"
        ),
    ),

    # --------------------------------------------------------
    # GitHub Intelligence Text-to-SQL
    # --------------------------------------------------------

    "get_github_intelligence_schema": ToolPolicy(

        tool_name=(
            "get_github_intelligence_schema"
        ),

        effect_type=(
            ToolEffectType.READ_ONLY
        ),

        has_external_side_effect=False,

        description=(
            "只读取允许暴露给 Agent 的"
            " GitHub 情报数据库 Schema。"
        ),
    ),

    "query_github_intelligence_sql": ToolPolicy(

        tool_name=(
            "query_github_intelligence_sql"
        ),

        effect_type=(
            ToolEffectType.READ_ONLY
        ),

        has_external_side_effect=False,

        description=(
            "只允许执行受安全策略约束的"
            " SQLite SELECT / WITH 查询。"
        ),
    ),

    # --------------------------------------------------------
    # Skill Runtime Control Tools
    # --------------------------------------------------------

    "list_skills": ToolPolicy(

        tool_name="list_skills",

        effect_type=(
            ToolEffectType.READ_ONLY
        ),

        has_external_side_effect=False,

        description=(
            "只读取当前 Skill Catalog "
            "与加载状态。"
        ),
    ),

    "load_skill": ToolPolicy(

        tool_name="load_skill",

        effect_type=(
            ToolEffectType.IDEMPOTENT_WRITE
        ),

        # load_skill 会改变当前 Runtime，
        # 但不会修改 Graph 外部持久化业务系统。
        has_external_side_effect=False,

        description=(
            "修改当前 Agent 进程中的 "
            "Skill Runtime 状态；"
            "重复加载同一 Skill 应保持幂等。"
        ),
    ),

    # --------------------------------------------------------
    # GitHub Intelligence Update Skill
    # --------------------------------------------------------

    "update_github_intelligence": ToolPolicy(

        tool_name=(
            "update_github_intelligence"
        ),

        # ----------------------------------------------------
        # 为什么目前不是 IDEMPOTENT_WRITE？
        #
        # 虽然当前更新流程已经具有：
        #
        # - 同日检查；
        # - 差异检测；
        # - 部分结果复用；
        #
        # 但我们还没有建立统一：
        #
        # - idempotency key；
        # - External Effect Ledger；
        # - Compensation；
        # - Replay Guard。
        #
        # 所以从 Agent Runtime 的安全角度，
        # 现在必须保守地按不可逆写操作处理。
        #
        # 等后续这些机制完成以后，
        # 再根据真实保证降级为：
        #
        # IDEMPOTENT_WRITE
        #
        # 或：
        #
        # COMPENSATABLE_WRITE。
        # ----------------------------------------------------

        effect_type=(
            ToolEffectType.IRREVERSIBLE_WRITE
        ),

        has_external_side_effect=True,

        description=(
            "会修改 GitHub 技术情报数据库、"
            "文件和索引。"
            "当前尚未建立统一幂等与补偿保证，"
            "因此暂按高风险写操作管理。"
        ),
    ),
}


# ============================================================
# Registry 查询函数
# ============================================================


def normalize_tool_name(
    tool_name: str,
) -> str:
    """规范化 Tool name。"""

    normalized = str(
        tool_name
    ).strip()

    if not normalized:
        raise ValueError(
            "tool_name 不能为空。"
        )

    return normalized


def get_tool_policy(
    tool_name: str,
) -> ToolPolicy | None:
    """查询 Tool Policy。

    未注册 Tool 返回 None。

    当前阶段先不自动报错，
    方便我们下一步做 Active Tool
    完整性检查。
    """

    normalized = (
        normalize_tool_name(
            tool_name
        )
    )

    return DEFAULT_TOOL_POLICIES.get(
        normalized
    )


def require_tool_policy(
    tool_name: str,
) -> ToolPolicy:
    """读取 Tool Policy。

    如果 Tool 没有声明安全策略，
    直接失败。

    后续 Agent 真正执行 Tool 前，
    会使用这一类 fail-closed 行为。
    """

    normalized = (
        normalize_tool_name(
            tool_name
        )
    )

    policy = (
        DEFAULT_TOOL_POLICIES.get(
            normalized
        )
    )

    if policy is None:
        raise KeyError(
            "Tool 尚未声明安全策略："
            f"{normalized}"
        )

    return policy


def list_tool_policies(
) -> list[ToolPolicy]:
    """返回全部 Tool Policy。"""

    return [
        DEFAULT_TOOL_POLICIES[
            tool_name
        ]
        for tool_name
        in sorted(
            DEFAULT_TOOL_POLICIES
        )
    ]


def find_undeclared_tool_names(
    tool_names: Iterable[str],
) -> list[str]:
    """检查一组 Tool 是否缺少安全策略。"""

    undeclared: set[str] = set()

    for tool_name in tool_names:

        normalized = (
            normalize_tool_name(
                tool_name
            )
        )

        if (
            normalized
            not in DEFAULT_TOOL_POLICIES
        ):
            undeclared.add(
                normalized
            )

    return sorted(
        undeclared
    )


def validate_tool_policy_coverage(
    tool_names: Iterable[str],
) -> None:
    """确认所有 Tool 均已声明安全策略。

    后续会在 Agent Tool Registry
    刷新时调用这个函数。

    这样动态 Skill 如果新增 Tool，
    但开发者忘了定义安全属性，
    系统会 Fail Closed，
    而不是默认相信它安全。
    """

    undeclared = (
        find_undeclared_tool_names(
            tool_names
        )
    )

    if undeclared:

        raise ValueError(
            "以下 Tool 尚未声明安全策略："
            + ", ".join(
                undeclared
            )
        )


__all__ = [
    "DEFAULT_TOOL_POLICIES",
    "ToolEffectType",
    "ToolPolicy",
    "find_undeclared_tool_names",
    "get_tool_policy",
    "list_tool_policies",
    "normalize_tool_name",
    "require_tool_policy",
    "validate_tool_policy_coverage",
]