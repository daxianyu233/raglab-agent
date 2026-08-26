"""RAGLab 的轻量 Intent + Context Planner。"""

from __future__ import annotations

import json
import re
import time

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from raglab.agent.context_plan import (
    ContextPlan,
    NavigationContext,
    compact_navigation_context,
    validate_context_plan,
)
from raglab.generation.rag_chain import (
    ChatModelProtocol,
    extract_answer_text,
    extract_usage_metadata,
)


CONTEXT_PLANNER_SYSTEM_PROMPT = """你是 RAGLab 的轻量 Context Planner。

你的任务不是回答用户，而是判断：
“主 Agent 为了完成当前用户请求，本轮需要哪些上下文信息？”

你不会看到历史正文。
你只能知道当前线程是否存在上一轮、近期历史或更早历史。
因此：
- 不要猜测历史里具体出现了哪些实体；
- 不要把历史主题自动继承到当前任务；
- 当前用户输入中的“刚才、之前、那个、第几个、继续”等未解析引用，
  只用于判断需要哪一类历史；
- 具体历史内容由后续 Context Retriever 根据你的 ContextPlan 再读取。

必须遵守：

1. 只输出一个 JSON 对象，不输出解释、Markdown 或代码块。
2. task_intent 是开放式短标签，不要强行从固定业务类别中挑选。
3. 不要生成具体 Tool Call，也不要决定 Tool 参数。
4. history_scope 只能是：
   - none
   - previous_turn
   - recent_turns
   - historical_search
5. 用户只要求改写/润色上一轮回答时：
   - 通常 previous_answer_required=true
   - raw_tool_evidence_required=false
   - external_retrieval_required=false
6. 用户要求“根据刚才查到的原始资料重新分析/整理”时：
   - raw_tool_evidence_required=true
   - 如果明确说不要重新搜索：
     external_retrieval_allowed=false
     external_retrieval_required=false
7. 用户引用很多轮以前的信息时：
   - history_scope=historical_search
   - 给出简短 history_query
8. 判断 history_required 时，必须先判断“当前用户请求是否自包含”：

   - 如果仅凭 current_user_input 本身就能明确知道用户要做什么、
     查询什么、比较什么或生成什么，则默认它是自包含请求。
   - recent_turns / history_candidates 与当前请求主题相同、
     实体相似、属于同一业务领域，均不能单独证明当前任务依赖历史。
   - “刚才、之前、上面、那个、继续、其中、第几个、按照前面的方案”
     等指代只是常见历史依赖信号；真正判断标准是：
     如果移除历史信息，当前请求是否会缺少完成任务所必需的信息。
   - 只有答案为“会缺少必要信息”时，history_required=true。

9. 对自包含的新请求：
   - history_required=false
   - history_scope=none
   - previous_answer_required=false
   - raw_tool_evidence_required=false
   - 不要把 recent_turns / history_candidates 中的实体、
     用户偏好、任务焦点自动继承到 response_goal 或 referenced_entities。
   - referenced_entities 主要来自 current_user_input。
   - 如果该请求需要新的外部知识/数据库/RAG，
     external_retrieval_required=true。

10. recent_turns 和 history_candidates 只是“可供导航的候选目录”，
    不是当前任务的默认上下文。除非当前任务确实依赖它们，
    否则必须忽略其中的实体和结果。

11. 只有确实需要跨线程用户事实、偏好或长期信息时，
    long_term_memory_required=true。

12. 不要因为系统“可以”检索某类信息，就自动认为“需要”它。

13. confidence 是 0 到 1 的数值。

必须输出这些字段：
task_intent
response_goal
history_required
history_scope
history_query
previous_answer_required
raw_tool_evidence_required
external_retrieval_required
external_retrieval_allowed
long_term_memory_required
long_term_memory_query
referenced_entities
temporal_scope
confidence
""".strip()


DEFAULT_CAPABILITY_CATALOG = [
    "conversation_history: 可恢复当前线程历史 Human/AI/Tool 证据",
    "long_term_memory: 可检索跨线程用户事实、偏好和长期记忆",
    "knowledge_rag: 可检索 PDF/知识库语义资料",
    "github_rag: 可检索已有 GitHub 项目、热点和日报语义资料",
    "github_sql: 可查询 GitHub Intelligence 结构化数据库",
    "skills: 主 Agent 可按需加载动态 Skill；Planner 不直接加载",
]


@dataclass(frozen=True)
class ContextPlannerResult:
    """一次 Planner 调用的结果和成本信息。"""

    plan: ContextPlan
    latency_ms: float
    usage_metadata: dict[str, Any]
    navigation_characters: int
    raw_model_output: str


class ContextPlanner:
    """基于现有 ChatModelProtocol 的轻量 Context Planner。"""

    def __init__(
        self,
        *,
        chat_model: ChatModelProtocol,
        maximum_summary_characters: int = 1200,
        maximum_recent_turns: int = 3,
        maximum_history_candidates: int = 6,
    ) -> None:
        self.chat_model = chat_model

        self.maximum_summary_characters = int(
            maximum_summary_characters
        )

        self.maximum_recent_turns = int(
            maximum_recent_turns
        )

        self.maximum_history_candidates = int(
            maximum_history_candidates
        )

        if self.maximum_summary_characters <= 0:
            raise ValueError(
                "maximum_summary_characters 必须大于 0。"
            )

        if self.maximum_recent_turns <= 0:
            raise ValueError(
                "maximum_recent_turns 必须大于 0。"
            )

        if self.maximum_history_candidates <= 0:
            raise ValueError(
                "maximum_history_candidates 必须大于 0。"
            )

    def plan(
        self,
        navigation_context: NavigationContext,
    ) -> ContextPlannerResult:
        """根据轻量 Navigation Context 生成 ContextPlan。"""

        if not (
            navigation_context
            .current_user_input
            .strip()
        ):
            raise ValueError(
                "current_user_input 不能为空。"
            )

        context = compact_navigation_context(
            navigation_context,
            maximum_summary_characters=(
                self.maximum_summary_characters
            ),
            maximum_recent_turns=(
                self.maximum_recent_turns
            ),
            maximum_history_candidates=(
                self.maximum_history_candidates
            ),
        )

        if not context.capability_catalog:
            context.capability_catalog = list(
                DEFAULT_CAPABILITY_CATALOG
            )

        # ----------------------------------------------------
        # Planner Input Boundary
        # ----------------------------------------------------
        #
        # Planner 只判断“需要哪类上下文”，不读取历史正文。
        # 这样可以避免“同主题历史”污染一个本来已经自包含的新请求。
        planner_payload = {
            "current_user_input": (
                context.current_user_input
            ),
            "context_availability": {
                "previous_turn_available": bool(
                    context.recent_turns
                ),
                "recent_history_available": bool(
                    context.recent_turns
                ),
                "historical_archive_available": bool(
                    context.history_candidates
                    or context.thread_summary
                ),
                "long_term_memory_available": (
                    "long_term_memory"
                    in " ".join(
                        context.capability_catalog
                        or DEFAULT_CAPABILITY_CATALOG
                    )
                ),
            },
            "capability_catalog": (
                context.capability_catalog
                or list(
                    DEFAULT_CAPABILITY_CATALOG
                )
            ),
            "runtime_notes": (
                context.runtime_notes
            ),
        }

        navigation_json = json.dumps(
            planner_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        user_prompt = (
            "请根据下面的 Navigation Context "
            "生成本轮 ContextPlan。\n\n"
            f"{navigation_json}"
        )

        start_time = time.perf_counter()

        response = self.chat_model.invoke(
            [
                SystemMessage(
                    content=(
                        CONTEXT_PLANNER_SYSTEM_PROMPT
                    )
                ),
                HumanMessage(
                    content=user_prompt
                ),
            ]
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        if not isinstance(
            response,
            AIMessage,
        ):
            raise TypeError(
                "Context Planner 必须返回 AIMessage，"
                f"实际类型：{type(response)!r}"
            )

        raw_output = (
            extract_answer_text(
                response
            )
            .strip()
        )

        if not raw_output:
            raise ValueError(
                "Context Planner 返回了空内容。"
            )

        parsed = self._parse_json_object(
            raw_output
        )

        plan = ContextPlan.model_validate(
            parsed
        )

        plan = validate_context_plan(
            plan
        )

        return ContextPlannerResult(
            plan=plan,
            latency_ms=latency_ms,
            usage_metadata=dict(
                extract_usage_metadata(
                    response
                )
                or {}
            ),
            navigation_characters=len(
                navigation_json
            ),
            raw_model_output=raw_output,
        )

    @staticmethod
    def _parse_json_object(
        text: str,
    ) -> dict[str, Any]:
        """容忍模型偶尔返回 ```json ... ```。"""

        normalized = str(
            text
        ).strip()

        fenced = re.fullmatch(
            r"```(?:json)?\s*(\{.*\})\s*```",
            normalized,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        if fenced:
            normalized = fenced.group(
                1
            )

        try:
            value = json.loads(
                normalized
            )
        except json.JSONDecodeError:
            start = normalized.find(
                "{"
            )

            end = normalized.rfind(
                "}"
            )

            if start < 0 or end < start:
                raise ValueError(
                    "Context Planner 输出中"
                    "没有可解析的 JSON 对象。"
                )

            value = json.loads(
                normalized[
                    start:
                    end + 1
                ]
            )

        if not isinstance(
            value,
            dict,
        ):
            raise ValueError(
                "Context Planner 顶层输出"
                "必须是 JSON object。"
            )

        return value