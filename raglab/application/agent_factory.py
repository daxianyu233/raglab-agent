"""RAGLab Agent application factory.

负责组装可被不同入口复用的 Agent Runtime。

入口层（CLI、FastAPI、后续 Scheduler/Worker）只调用本模块的
``build_agent``，不再各自维护一套 Agent 构建逻辑。

当前迁移阶段仍复用 scripts.ask_rag 中已经验证过的基础构建 helper，
以及 scripts.chat_retrieval_agent 中的基础 system prompt。
下一阶段可继续把这些通用能力下沉到 raglab 包内，彻底消除
application -> scripts 的依赖。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from raglab.agent.automatic_long_term_memory_agent import (
    AutomaticLongTermMemoryAgent,
)
from raglab.agent.conversation_event_store import (
    ConversationEventStore,
)
from raglab.agent.skill_runtime import (
    SkillRuntime,
)
from raglab.agent.tools import (
    create_agent_tools,
    get_tool_names,
)

from raglab.persistence.sqlite_backend import (
    create_sqlite_agent_persistence,
)

# 迁移阶段临时复用：这些函数原来已经被 CLI 验证通过。
from scripts.ask_rag import (
    build_bm25_index,
    create_deepseek_model,
    load_yaml_config,
    require_mapping,
    require_string,
    resolve_project_path,
)
from scripts.chat_retrieval_agent import (
    CONVERSATIONAL_AGENT_SYSTEM_PROMPT,
)


# ============================================================
# GitHub Intelligence Text-to-SQL 路由规则
# ============================================================


GITHUB_INTELLIGENCE_SQL_ROUTING_PROMPT = """
# GitHub Intelligence 结构化查询规则

系统提供一个本地 GitHub Intelligence SQLite 数据库，
但数据库的具体 Table、View 和 Column 不会固定写入 System Prompt。

当用户的问题需要精确的结构化数据时，例如：

- 精确数量；
- 日期或时间范围；
- 排名；
- 分组统计；
- COUNT、AVG、MIN、MAX、SUM 等聚合；
- Star、Fork、Issue 等数值；
- 项目的首次或最近出现时间；
- 某项目出现或入选的次数；
- 多个项目之间的结构化比较；
- 多张数据库表之间的 JOIN；
- 其他需要数据库精确计算才能回答的问题；

必须遵循下面的顺序：

1. 先调用 get_github_intelligence_schema。

   该工具会动态返回当前 Agent 被允许访问的
   SQLite Table、View、Column、类型、外键和业务说明。

2. 不要凭记忆、猜测或以前的 Tool Result
   假设数据库当前存在某张表或某个字段。

3. 获取 Schema 后，根据用户当前问题和返回的真实 Schema
   生成 SQLite 只读 SQL。

4. SQL 只能使用 Schema 中明确展示的 Table、View 和 Column。

5. 只生成 SELECT 或 WITH ... SELECT 查询。

6. 避免 SELECT *，明确列出真正需要查询的字段。

7. 将生成的 SQL 交给 query_github_intelligence_sql 执行。

8. 根据 SQL Tool 返回的真实结果回答用户，
   不要自行编造数据库中不存在的数据。

get_github_intelligence_schema 只返回数据库结构，
不返回 GitHub 项目的实际业务记录。

query_github_intelligence_sql 负责执行受权限控制的只读 SQL，
不能用于修改、删除、插入或创建数据库内容。

如果用户询问的是：

- 项目功能；
- 技术方案；
- 项目介绍；
- README 内容；
- 技术特点；
- 技术趋势解释；
- 热点含义；
- 项目摘要；
- 非结构化技术语义；

优先使用 search_github_intelligence，
不要为了这类语义问题机械调用 SQL Schema Tool。

如果一个问题同时包含：

- 精确结构化统计；
- 技术语义解释；

可以先通过 Schema + SQL 得到准确的项目、数量、
日期、排名或其他结构化结果，
再根据实际需要使用 search_github_intelligence
补充相关项目的技术语义信息。

不要对每一个 GitHub 问题机械调用
get_github_intelligence_schema。
只有问题确实需要结构化数据库查询时才调用。
""".strip()


# ============================================================
# GitHub 日报读取 / 更新路由规则
# ============================================================


GITHUB_DAILY_REPORT_ROUTING_PROMPT = """
# GitHub RAG Metadata-aware Retrieval 与日报路由规则

必须区分“结构化硬约束”和“语义软条件”。
日期、文档类型、完整仓库名等确定性条件不能只写进 query
交给 BM25 猜，而应优先传入 search_github_intelligence 的
metadata 参数，在检索前先过滤候选集。

## 1. 日期必须作为 metadata 硬约束

当用户明确指定日期或使用相对日期时：

- 今天 / 今日 -> snapshot_date="today"
- 昨天 / 昨日 -> snapshot_date="yesterday"
- 明确日期 -> snapshot_date="YYYY-MM-DD"

Tool 会在执行时把 today / yesterday 解析成当前机器本地日期，
因此不要根据历史消息或旧检索结果猜“今天是哪一天”。

## 2. 内容类型必须映射到 doc_types

- 日报 / 技术速报 / daily brief
  -> doc_types=["daily_brief"]

- 热点 / 热点主题
  -> doc_types=["daily_hotspot"]

- 项目分析 / 项目摘要 / 某日值得关注的项目
  -> doc_types=["repository_summary"]

如果用户要求跨多种内容类型，可以传多个 doc_types。
如果用户询问跨日期的长期趋势，不要机械添加 snapshot_date。

## 3. 典型调用

用户：给我今天的日报

应调用：
search_github_intelligence(
    query="GitHub 技术情报日报",
    snapshot_date="today",
    doc_types=["daily_brief"],
    top_k=5
)

用户：给我昨天的日报

应调用：
search_github_intelligence(
    query="GitHub 技术情报日报",
    snapshot_date="yesterday",
    doc_types=["daily_brief"],
    top_k=5
)

用户：8 月 17 日有哪些热点？

应调用：
search_github_intelligence(
    query="GitHub 技术热点",
    snapshot_date="2026-08-17",
    doc_types=["daily_hotspot"],
    top_k=5
)

不要采用下面这种错误方式：

search_github_intelligence(
    query="2026-08-17 GitHub 技术情报日报 热点项目",
    top_k=5
)

因为这仍然是在全库中让不同日期、不同 doc_type 的 Chunk
竞争 BM25 分数，正确文档可能在 Top-K 之前被截掉。

## 4. 日报内容请求默认是只读查询

“生成今日的日报”“给我今天的日报”“查看某日日报”如果没有
明确的更新、刷新、重新采集或重新运行语义，都表示读取当前
RAG 索引中已有的日报，不是重新执行更新流水线。

读取已有日报时：

- 不需要先调用 list_skills；
- 不需要调用 load_skill；
- 直接调用带 metadata 过滤条件的 search_github_intelligence。

## 5. 只有明确写操作意图才允许更新

只有用户明确要求更新、刷新、重新采集、同步最新数据、重新运行
GitHub 情报流水线或强制重建日报时，才允许进入
``github-intelligence-update`` Skill，并调用
``update_github_intelligence``。

即使该 Skill 已加载，也不能因为 Tool 可用就主动更新。

## 6. metadata 查询没有结果时

如果带 snapshot_date / doc_types 等硬约束的只读查询没有结果：

1. 明确告诉用户该日期 / 类型在当前 RAG 索引中没有匹配文档；
2. 不得自动删除过滤条件后改搜其他日期；
3. 不得拿最近历史日报冒充目标日期日报；
4. 不得因为只读查询失败就自行升级为写操作；
5. 可以提示用户如需重新采集，可明确要求更新或刷新。
""".strip()


# ============================================================
# 配置辅助
# ============================================================


def read_optional_mapping(
    config: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """读取可选的字典配置节点。"""

    value = config.get(
        key,
        {},
    )

    if value is None:
        return {}

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"配置节点 {key} 必须是字典。"
        )

    return dict(
        value
    )


# ============================================================
# Agent Factory
# ============================================================


def build_agent(
    config_path: Path,
) -> AutomaticLongTermMemoryAgent:
    """根据配置创建可被多个入口复用的 Agent。

    这个函数只负责“组装 Agent”，不负责：

    - 命令行 input / print；
    - HTTP Request / Response；
    - 定时任务触发；
    - 后台 Worker 调度。

    这样 CLI、FastAPI 以及后续其他入口都可以复用同一套
    Agent 构建逻辑。
    """

    config = load_yaml_config(
        config_path
    )

    retrieval_config = require_mapping(
        config,
        "retrieval",
    )

    model_config = require_mapping(
        config,
        "model",
    )

    tool_config = require_mapping(
        config,
        "tool",
    )

    agent_config = require_mapping(
        config,
        "agent",
    )

    github_config = (
        read_optional_mapping(
            config,
            "github_intelligence",
        )
    )

    # --------------------------------------------------------
    # PDF Retrieval
    # --------------------------------------------------------

    retrieval_type = require_string(
        retrieval_config,
        "type",
    ).lower()

    if retrieval_type != "bm25":
        raise ValueError(
            "当前普通 PDF 知识库只支持 BM25，"
            f"实际配置：{retrieval_type}"
        )

    bm25_config_path = (
        resolve_project_path(
            require_string(
                retrieval_config,
                "config_path",
            )
        )
    )

    # --------------------------------------------------------
    # 通用 Tool 配置
    # --------------------------------------------------------

    default_top_k = int(
        tool_config.get(
            "default_top_k",
            5,
        )
    )

    maximum_top_k = int(
        tool_config.get(
            "maximum_top_k",
            10,
        )
    )

    max_characters_per_document = int(
        tool_config.get(
            "max_characters_per_document",
            1500,
        )
    )

    # --------------------------------------------------------
    # GitHub Intelligence Tool 配置
    # --------------------------------------------------------

    include_github_search = bool(
        github_config.get(
            "enable_search_tool",
            True,
        )
    )

    include_github_schema = bool(
        github_config.get(
            "enable_schema_tool",
            True,
        )
    )

    include_github_sql = bool(
        github_config.get(
            "enable_sql_tool",
            True,
        )
    )

    # 当前设计要求：
    #
    # Text-to-SQL 必须先通过 Schema Tool
    # 获得当前真实、经过权限过滤的 Schema。
    #
    # 因此不允许：
    #
    # SQL Tool 开启
    # +
    # Schema Tool 关闭
    #
    # 否则模型只能猜数据库结构。
    if (
        include_github_sql
        and not include_github_schema
    ):
        raise ValueError(
            "启用 GitHub Intelligence SQL Tool 时，"
            "必须同时启用 Schema Tool。"
        )

    github_default_top_k = int(
        github_config.get(
            "default_top_k",
            default_top_k,
        )
    )

    github_maximum_top_k = int(
        github_config.get(
            "maximum_top_k",
            maximum_top_k,
        )
    )

    github_max_characters = int(
        github_config.get(
            "max_characters_per_document",
            1800,
        )
    )

    # --------------------------------------------------------
    # PDF BM25
    # --------------------------------------------------------

    bm25_index, build_info = (
        build_bm25_index(
            bm25_config_path
        )
    )

    print(
        "PDF BM25 索引构建完成："
        f"{build_info['chunk_count']} "
        "个 Chunk"
    )

    # --------------------------------------------------------
    # Chat Model
    # --------------------------------------------------------

    chat_model = create_deepseek_model(
        model_config
    )

    # --------------------------------------------------------
    # Skill Runtime
    # --------------------------------------------------------
    #
    # 启动时只 Discover Skill。
    # github-intelligence-update 尚未加载，
    # update_github_intelligence 也不是 Active Tool。
    # --------------------------------------------------------

    skill_runtime = SkillRuntime()

    skill_catalog_prompt = (
        skill_runtime.render_catalog_prompt()
    )

    # --------------------------------------------------------
    # Dynamic System Prompt
    # --------------------------------------------------------
    #
    # 注意：
    #
    # 这里不注入完整 SQLite Schema。
    #
    # LLM 只有在真正需要结构化查询时，
    # 才通过 get_github_intelligence_schema
    # 动态获取当前可访问 Schema。
    # --------------------------------------------------------

    prompt_parts: list[str] = [
        CONVERSATIONAL_AGENT_SYSTEM_PROMPT,
    ]

    if (
        include_github_schema
        and include_github_sql
    ):
        prompt_parts.append(
            GITHUB_INTELLIGENCE_SQL_ROUTING_PROMPT
        )

    # --------------------------------------------------------
    # 日报读取 / 更新路由
    # --------------------------------------------------------
    #
    # 这条规则优先解决自然语言中“生成日报”的歧义：
    # 默认读取已有日报；只有明确要求更新/刷新/重新采集
    # 时才允许进入 update_github_intelligence。

    prompt_parts.append(
        GITHUB_DAILY_REPORT_ROUTING_PROMPT
    )

    prompt_parts.append(
        (
            "# Skill 按需加载规则\n\n"
            "系统支持 Skill Runtime。"
            "Skill 出现在 Catalog 中只表示它可以被发现，"
            "不表示它已经加载。\n\n"

            "当用户请求明确匹配某个尚未加载的 Skill 时，"
            "先调用 load_skill，并使用 Catalog 中的完整 skill id。"
            "load_skill 成功后不要立即假设任务已经完成，"
            "而应在下一次 Agent 决策中调用该 Skill "
            "新开放的业务 Tool。\n\n"

            "只有用户询问有哪些 Skill、"
            "当前 Skill 状态等问题时，"
            "才需要调用 list_skills；"

            "执行普通业务任务时不要求先机械调用 "
            "list_skills。\n\n"

            f"{skill_catalog_prompt}"
        )
    )

    dynamic_system_prompt = (
        "\n\n".join(
            part.strip()
            for part in prompt_parts
            if str(
                part
            ).strip()
        )
    )

    # --------------------------------------------------------
    # Agent Tools
    # --------------------------------------------------------

    agent_tools = create_agent_tools(
        bm25_index=bm25_index,

        default_top_k=(
            default_top_k
        ),

        maximum_top_k=(
            maximum_top_k
        ),

        max_characters_per_document=(
            max_characters_per_document
        ),

        include_github_search=(
            include_github_search
        ),

        include_github_schema=(
            include_github_schema
        ),

        include_github_sql=(
            include_github_sql
        ),

        skill_runtime=(
            skill_runtime
        ),

        github_default_top_k=(
            github_default_top_k
        ),

        github_maximum_top_k=(
            github_maximum_top_k
        ),

        github_max_characters_per_document=(
            github_max_characters
        ),
    )

    print(
        "启动时基础工具："
        + ", ".join(
            get_tool_names(
                agent_tools
            )
        )
    )

    # --------------------------------------------------------
    # Skill Runtime Status
    # --------------------------------------------------------

    available_skill_ids = (
        skill_runtime.available_skill_ids()
    )

    print(
        "发现的 Skills："
        + (
            ", ".join(
                available_skill_ids
            )
            if available_skill_ids
            else "无"
        )
    )

    print(
        "启动时已加载 Skills：无"
    )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    # --------------------------------------------------------
    # SQLite Persistence
    # --------------------------------------------------------

    persistence = (
        create_sqlite_agent_persistence()
    )

    print(
        "Agent SQLite 持久化："
        f"{persistence.database_path}"
    )

    conversation_event_store = (
        ConversationEventStore()
    )

    print(
        "Conversation Event Store："
        f"{conversation_event_store.database_path}"
    )

    context_window_tokens = int(
        agent_config.get(
            "context_window_tokens",
            model_config.get(
                "context_window_tokens",
                32768,
            ),
        )
    )

    reserved_output_tokens = int(
        agent_config.get(
            "reserved_output_tokens",
            4096,
        )
    )

    context_safety_margin_tokens = int(
        agent_config.get(
            "context_safety_margin_tokens",
            1024,
        )
    )

    print(
        "Context Budget："
        f"window={context_window_tokens}, "
        f"output_reserve={reserved_output_tokens}, "
        f"safety_margin={context_safety_margin_tokens}"
    )

    agent = AutomaticLongTermMemoryAgent(
        chat_model=chat_model,
        tools=agent_tools,

        max_steps=int(
            agent_config.get(
                "max_steps",
                4,
            )
        ),

        system_prompt=(
            dynamic_system_prompt
        ),

        skill_runtime=(
            skill_runtime
        ),

        # ---------------------------------------------
        # Persistent Thread State
        # ---------------------------------------------

        checkpointer=(
            persistence.checkpointer
        ),

        # ---------------------------------------------
        # Persistent User Long-Term Memory
        # ---------------------------------------------

        store=(
            persistence.store
        ),

        keep_recent_turns=int(
            agent_config.get(
                "keep_recent_turns",
                4,
            )
        ),

        summarize_trigger_turns=int(
            agent_config.get(
                "summarize_trigger_turns",
                7,
            )
        ),

        minimum_memory_confidence=float(
            agent_config.get(
                "minimum_memory_confidence",
                0.80,
            )
        ),

        # ---------------------------------------------
        # Context Pipeline Phase 7A
        # ---------------------------------------------
        conversation_event_store=(
            conversation_event_store
        ),

        context_pipeline_enabled=bool(
            agent_config.get(
                "context_pipeline_enabled",
                True,
            )
        ),

        context_window_tokens=(
            context_window_tokens
        ),

        reserved_output_tokens=(
            reserved_output_tokens
        ),

        context_safety_margin_tokens=(
            context_safety_margin_tokens
        ),

        context_recent_turn_limit=int(
            agent_config.get(
                "context_recent_turn_limit",
                3,
            )
        ),

        context_historical_turn_limit=int(
            agent_config.get(
                "context_historical_turn_limit",
                3,
            )
        ),
    )

    # 保存 Persistence Bundle 引用，
    # 后续 CLI / FastAPI 可以显式关闭 SQLite Connection。
    agent.persistence_backend = (
        persistence
    )

    return agent