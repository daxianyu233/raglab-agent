"""RAG Agent 使用的基础 LangChain 工具。

当前基础层提供三项业务查询能力：

1. search_knowledge_base
   搜索原来的 PDF 学习知识库。

2. search_github_intelligence
   搜索已经采集并建立索引的 GitHub 技术情报，
   适合项目摘要、热点、日报和技术趋势等语义查询。

3. query_github_intelligence_sql
   只读查询 GitHub Intelligence SQLite 数据库，
   适合数量、日期、排序、聚合、历史记录和多表关联等
   精确结构化查询。

此外，当 create_agent_tools() 接收到 SkillRuntime 时，
会加入两个 Skill Runtime 控制工具：

4. list_skills
   查看系统发现的 Skill 以及当前加载状态。

5. load_skill
   按需加载 Skill。

Skill 专属业务 Tool 不在本文件中静态注册。
例如 update_github_intelligence 只有在
github-intelligence-update Skill 被加载后，
才由 SkillRuntime 动态加入 Agent 的 Active Tools。

职责必须严格区分：

- 查询 PDF 学习资料：
  使用 search_knowledge_base。

- 查询已有 GitHub 项目摘要、每日热点、日报和技术语义：
  使用 search_github_intelligence。

- 查询 GitHub 本地数据库中的数量、日期、排序、聚合、
  历史出现次数或其他精确结构化信息：
  使用 query_github_intelligence_sql。

- 用户明确要求更新、刷新、重新采集或同步 GitHub 信息：
  先加载 github-intelligence-update Skill，
  再使用该 Skill 开放的 update_github_intelligence。

- 非 GitHub 网站的信息更新：
  不得加载 github-intelligence-update Skill。
"""

from __future__ import annotations

import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, Sequence

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from raglab.agent.github_intelligence_sql_tool import (
    DEFAULT_DATABASE_PATH as DEFAULT_GITHUB_INTELLIGENCE_DATABASE_PATH,
    create_github_intelligence_schema_tool,
    create_github_intelligence_sql_tool,
)
from raglab.agent.skill_runtime import SkillRuntime
from raglab.intelligence.retriever import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_DOCUMENTS_PATH,
    DEFAULT_BM25_K1,
    DEFAULT_BM25_PICKLE_PATH,
    DEFAULT_MANIFEST_PATH,
    load_or_rebuild_bm25_index,
    search_bm25_with_metadata_filters,
)


# ============================================================
# 通用类型
# ============================================================


class BM25IndexProtocol(Protocol):
    """BM25 索引需要满足的最小接口。"""

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> Sequence[Any]:
        """执行 BM25 检索。"""


# ============================================================
# Tool 输入模型
# ============================================================


class KnowledgeBaseSearchInput(
    BaseModel
):
    """普通 PDF 知识库检索工具的输入参数。"""

    query: str = Field(
        description=(
            "用于 PDF 学习知识库检索的完整、独立问题或关键词。"
            "不要使用“它”“这个”“上述方法”等"
            "依赖对话历史的模糊表达。"
        )
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description=(
            "需要返回的候选资料数量。"
            "一般使用 3 到 5；"
            "只有信息可能分散在多处时才增加。"
        ),
    )


class GitHubIntelligenceSearchInput(
    BaseModel
):
    """GitHub 技术情报检索工具的输入参数。"""

    query: str = Field(
        description=(
            "用于 GitHub 技术情报库检索的完整、独立问题。"
            "query 负责表达技术语义；日期、文档类型、项目等"
            "确定性硬约束应优先填写到下面的 metadata 参数中，"
            "不要只把硬约束写进 query 让 BM25 自己猜。"
        )
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description=(
            "metadata 过滤之后，需要返回的候选资料数量。"
            "通常使用 3 到 5。"
        ),
    )

    snapshot_date: str | None = Field(
        default=None,
        description=(
            "可选的日期硬约束。支持 YYYY-MM-DD、today、"
            "yesterday，以及中文的今天/今日/昨天/昨日。"
            "例如查询今日日报时填写 today；"
            "不要仅把日期写进 query。"
        ),
    )

    doc_types: list[str] = Field(
        default_factory=list,
        description=(
            "可选的文档类型硬约束。当前常用值："
            "daily_brief（日 报）、daily_hotspot（热点主题）、"
            "repository_summary（项目分析）。"
            "如果用户明确指定内容类型，应填写这里。"
        ),
    )

    repository: str | None = Field(
        default=None,
        description=(
            "可选的 GitHub 仓库完整名硬约束，例如 "
            "Panniantong/Agent-Reach。只有明确知道完整仓库名时"
            "才填写；模糊项目名仍放在 query 中。"
        ),
    )

    topic_name: str | None = Field(
        default=None,
        description=(
            "可选的热点主题精确名称硬约束。只有明确知道"
            "完整 topic_name 时才填写。"
        ),
    )


# ============================================================
# GitHub metadata 过滤参数
# ============================================================


SUPPORTED_GITHUB_DOC_TYPES = {
    "daily_brief",
    "daily_hotspot",
    "repository_summary",
}


def resolve_snapshot_date_filter(
    value: str | None,
) -> str | None:
    """将日期过滤条件规范为 YYYY-MM-DD。

    相对日期在 Tool 执行时解析，因此不会依赖 LLM 猜测
    当前日期。
    """

    if value is None:
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    normalized = text.casefold()

    local_today = (
        datetime.now()
        .astimezone()
        .date()
    )

    if normalized in {
        "today",
        "今天",
        "今日",
    }:
        return local_today.isoformat()

    if normalized in {
        "yesterday",
        "昨天",
        "昨日",
    }:
        return (
            local_today
            - timedelta(
                days=1
            )
        ).isoformat()

    # YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD
    full_date_match = re.fullmatch(
        r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})",
        text,
    )

    if full_date_match is not None:
        year, month, day = (
            int(part)
            for part in full_date_match.groups()
        )

        return datetime(
            year,
            month,
            day,
        ).date().isoformat()

    chinese_date_match = re.fullmatch(
        r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})(?:日|号)?",
        text,
    )

    if chinese_date_match is not None:
        year_text, month_text, day_text = (
            chinese_date_match.groups()
        )

        year = (
            local_today.year
            if year_text is None
            else int(
                year_text
            )
        )

        return datetime(
            year,
            int(
                month_text
            ),
            int(
                day_text
            ),
        ).date().isoformat()

    month_day_match = re.fullmatch(
        r"(\d{1,2})[-/.](\d{1,2})",
        text,
    )

    if month_day_match is not None:
        month, day = (
            int(part)
            for part in month_day_match.groups()
        )

        return datetime(
            local_today.year,
            month,
            day,
        ).date().isoformat()

    raise ValueError(
        "snapshot_date 无法解析："
        f"{value!r}。"
        "请使用 YYYY-MM-DD、today、yesterday、"
        "今天、今日、昨天或昨日。"
    )


def normalize_doc_type_filters(
    values: Sequence[str] | None,
) -> list[str]:
    """规范并校验 GitHub RAG doc_type 过滤条件。"""

    normalized: list[str] = []

    for value in (
        values
        or []
    ):
        text = str(
            value
        ).strip().casefold()

        if not text:
            continue

        if (
            text
            not in SUPPORTED_GITHUB_DOC_TYPES
        ):
            raise ValueError(
                "不支持的 GitHub doc_type："
                f"{value!r}。"
                "当前支持："
                + ", ".join(
                    sorted(
                        SUPPORTED_GITHUB_DOC_TYPES
                    )
                )
            )

        if text not in normalized:
            normalized.append(
                text
            )

    return normalized


def build_applied_github_filters(
    *,
    snapshot_date: str | None,
    doc_types: Sequence[str] | None,
    repository: str | None,
    topic_name: str | None,
) -> dict[str, Any]:
    """构造便于 Tool 输出和调试的 metadata 过滤摘要。"""

    filters: dict[str, Any] = {}

    if snapshot_date:
        filters[
            "snapshot_date"
        ] = snapshot_date

    if doc_types:
        filters[
            "doc_types"
        ] = list(
            doc_types
        )

    normalized_repository = str(
        repository
        or ""
    ).strip()

    if normalized_repository:
        filters[
            "repository"
        ] = normalized_repository

    normalized_topic_name = str(
        topic_name
        or ""
    ).strip()

    if normalized_topic_name:
        filters[
            "topic_name"
        ] = normalized_topic_name

    return filters


# ============================================================
# 通用文本与结果处理
# ============================================================


def normalize_text(
    text: str,
) -> str:
    """清理工具输出中的多余空白。"""

    return " ".join(
        str(text).split()
    ).strip()


def safe_float(
    value: Any,
) -> float | None:
    """尝试将值转换为浮点数。"""

    if value is None:
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def normalize_search_result(
    item: Any,
) -> tuple[
    Document,
    float | None,
]:
    """统一不同检索结果的数据格式。

    支持：

    1. Document；
    2. tuple[Document, score]；
    3. 具有 document 和 score 属性的对象。
    """

    if isinstance(
        item,
        Document,
    ):
        metadata_score = safe_float(
            item.metadata.get(
                "retrieval_raw_score"
            )
        )

        return (
            item,
            metadata_score,
        )

    if (
        isinstance(
            item,
            tuple,
        )
        and len(item) >= 1
        and isinstance(
            item[0],
            Document,
        )
    ):
        score = (
            safe_float(
                item[1]
            )
            if len(item) >= 2
            else None
        )

        return (
            item[0],
            score,
        )

    document = getattr(
        item,
        "document",
        None,
    )

    if isinstance(
        document,
        Document,
    ):
        score = safe_float(
            getattr(
                item,
                "score",
                None,
            )
        )

        if score is None:
            score = safe_float(
                getattr(
                    item,
                    "raw_score",
                    None,
                )
            )

        return (
            document,
            score,
        )

    raise TypeError(
        "无法识别 BM25 检索结果格式："
        f"type={type(item)!r}, "
        f"value={item!r}"
    )


def append_metadata_part(
    parts: list[str],
    *,
    label: str,
    value: Any,
) -> None:
    """向标题中追加非空元数据。"""

    if value is None:
        return

    normalized = str(
        value
    ).strip()

    if not normalized:
        return

    parts.append(
        f"{label}：{normalized}"
    )


def format_document_header(
    document: Document,
    reference_index: int,
    score: float | None,
) -> str:
    """生成普通知识库检索资料标题。"""

    metadata = document.metadata

    parts = [
        f"[资料{reference_index}]",
    ]

    append_metadata_part(
        parts,
        label="标题",
        value=metadata.get(
            "title"
        ),
    )

    append_metadata_part(
        parts,
        label="文档ID",
        value=metadata.get(
            "doc_id"
        )
        or metadata.get(
            "document_id"
        ),
    )

    append_metadata_part(
        parts,
        label="Chunk ID",
        value=metadata.get(
            "chunk_id"
        ),
    )

    append_metadata_part(
        parts,
        label="页码",
        value=metadata.get(
            "page_number"
        ),
    )

    if score is not None:
        parts.append(
            f"BM25分数：{score:.6f}"
        )

    return " | ".join(
        parts
    )


def format_github_document_header(
    document: Document,
    reference_index: int,
    score: float | None,
) -> str:
    """生成 GitHub 技术情报资料标题。"""

    metadata = document.metadata

    parts = [
        f"[GitHub资料{reference_index}]",
    ]

    append_metadata_part(
        parts,
        label="标题",
        value=metadata.get(
            "title"
        ),
    )

    append_metadata_part(
        parts,
        label="类型",
        value=metadata.get(
            "doc_type"
        ),
    )

    append_metadata_part(
        parts,
        label="日期",
        value=metadata.get(
            "snapshot_date"
        ),
    )

    append_metadata_part(
        parts,
        label="项目",
        value=metadata.get(
            "repository"
        ),
    )

    append_metadata_part(
        parts,
        label="主题",
        value=metadata.get(
            "topic_name"
        ),
    )

    append_metadata_part(
        parts,
        label="文档ID",
        value=metadata.get(
            "document_id"
        )
        or metadata.get(
            "doc_id"
        ),
    )

    append_metadata_part(
        parts,
        label="Chunk ID",
        value=metadata.get(
            "chunk_id"
        ),
    )

    if score is not None:
        parts.append(
            f"BM25分数：{score:.6f}"
        )

    return " | ".join(
        parts
    )


def format_search_results(
    query: str,
    raw_results: Sequence[Any],
    *,
    max_characters_per_document: int,
) -> str:
    """将普通 BM25 检索结果转换成模型可阅读文本。"""

    if not raw_results:
        return (
            "PDF 知识库检索没有返回任何资料。\n"
            f"检索问题：{query}"
        )

    blocks: list[str] = []

    for rank, item in enumerate(
        raw_results,
        start=1,
    ):
        document, score = (
            normalize_search_result(
                item
            )
        )

        text = normalize_text(
            document.page_content
        )

        if (
            len(text)
            > max_characters_per_document
        ):
            text = (
                text[
                    :max_characters_per_document
                ]
                .rstrip()
                + "..."
            )

        header = format_document_header(
            document=document,
            reference_index=rank,
            score=score,
        )

        blocks.append(
            f"{header}\n{text}"
        )

    joined_results = "\n\n".join(
        blocks
    )

    return (
        "检索来源：PDF 学习知识库\n"
        f"检索问题：{query}\n"
        f"返回资料数：{len(blocks)}\n\n"
        f"{joined_results}"
    )


def format_github_search_results(
    query: str,
    raw_results: Sequence[Any],
    *,
    max_characters_per_document: int,
    index_status: dict[str, Any],
    applied_filters: dict[str, Any] | None = None,
    candidate_count: int | None = None,
) -> str:
    """将 GitHub BM25 检索结果转换成模型可阅读文本。"""

    normalized_filters = dict(
        applied_filters
        or {}
    )

    filter_text = (
        str(
            normalized_filters
        )
        if normalized_filters
        else "无"
    )

    if not raw_results:
        return (
            "GitHub 技术情报库没有返回任何资料。\n"
            f"检索问题：{query}\n"
            f"metadata 过滤：{filter_text}\n"
            "如果设置了 metadata 硬约束，这表示当前 RAG 索引中"
            "没有满足这些约束的文档；不得自动改用其他日期或"
            "其他 doc_type 的资料冒充结果。"
        )

    blocks: list[str] = []

    for rank, item in enumerate(
        raw_results,
        start=1,
    ):
        document, score = (
            normalize_search_result(
                item
            )
        )

        text = normalize_text(
            document.page_content
        )

        if (
            len(text)
            > max_characters_per_document
        ):
            text = (
                text[
                    :max_characters_per_document
                ]
                .rstrip()
                + "..."
            )

        header = (
            format_github_document_header(
                document=document,
                reference_index=rank,
                score=score,
            )
        )

        blocks.append(
            f"{header}\n{text}"
        )

    joined_results = "\n\n".join(
        blocks
    )

    load_method = index_status.get(
        "load_method",
        "unknown",
    )

    document_count = index_status.get(
        "document_count",
        "unknown",
    )

    candidate_count_text = (
        "unknown"
        if candidate_count is None
        else str(
            candidate_count
        )
    )

    return (
        "检索来源：GitHub 技术情报持久化索引\n"
        f"检索问题：{query}\n"
        f"metadata 过滤：{filter_text}\n"
        f"过滤后候选 Chunk 数：{candidate_count_text}\n"
        f"返回资料数：{len(blocks)}\n"
        f"索引加载方式：{load_method}\n"
        f"索引文档数：{document_count}\n\n"
        f"{joined_results}"
    )


# ============================================================
# 普通 PDF 知识库 Tool
# ============================================================


def create_bm25_search_tool(
    bm25_index: BM25IndexProtocol,
    *,
    default_top_k: int = 5,
    maximum_top_k: int = 10,
    max_characters_per_document: int = 1500,
) -> BaseTool:
    """根据现有 BM25 索引创建 PDF 知识库搜索工具。"""

    if not hasattr(
        bm25_index,
        "search",
    ):
        raise TypeError(
            "bm25_index 必须实现 search()。"
        )

    if default_top_k <= 0:
        raise ValueError(
            "default_top_k 必须大于 0。"
        )

    if maximum_top_k <= 0:
        raise ValueError(
            "maximum_top_k 必须大于 0。"
        )

    if (
        default_top_k
        > maximum_top_k
    ):
        raise ValueError(
            "default_top_k 不能大于 "
            "maximum_top_k。"
        )

    if (
        max_characters_per_document
        <= 0
    ):
        raise ValueError(
            "max_characters_per_document "
            "必须大于 0。"
        )

    @tool(
        "search_knowledge_base",
        args_schema=(
            KnowledgeBaseSearchInput
        ),
    )
    def search_knowledge_base(
        query: str,
        top_k: int = default_top_k,
    ) -> str:
        """搜索本地 PDF 学习知识库。

        当用户询问 PDF 语料中的技术概念、实现方法、
        实验结果、配置、步骤或结论时，使用该工具。

        查询 GitHub 项目、GitHub 每日热点或 GitHub 日报时，
        应使用 search_github_intelligence，而不是本工具。

        用户要求重新采集或更新 GitHub 信息时，
        应先加载 github-intelligence-update Skill，
        再使用该 Skill 开放的 update_github_intelligence。
        """

        normalized_query = str(
            query
        ).strip()

        if not normalized_query:
            return (
                "工具调用失败："
                "检索问题不能为空。"
            )

        requested_top_k = int(
            top_k
        )

        effective_top_k = max(
            1,
            min(
                requested_top_k,
                maximum_top_k,
            ),
        )

        raw_results = (
            bm25_index.search(
                query=normalized_query,
                top_k=effective_top_k,
            )
        )

        return format_search_results(
            query=normalized_query,
            raw_results=list(
                raw_results
            ),
            max_characters_per_document=(
                max_characters_per_document
            ),
        )

    return search_knowledge_base


# ============================================================
# GitHub 情报 BM25 自动重载
# ============================================================


class ReloadableGitHubBM25Index:
    """按索引文件变化自动重新加载 GitHub BM25。

    对话系统启动时不会立即加载索引。

    第一次调用 search_github_intelligence 时才加载。

    update_github_intelligence 重建索引后，
    pickle、documents.jsonl 或 manifest 的文件状态会变化，
    下一次搜索将自动重新加载，无需重启对话程序。
    """

    def __init__(
        self,
        *,
        pickle_path: Path = (
            DEFAULT_BM25_PICKLE_PATH
        ),
        documents_path: Path = (
            DEFAULT_BM25_DOCUMENTS_PATH
        ),
        manifest_path: Path = (
            DEFAULT_MANIFEST_PATH
        ),
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
    ) -> None:
        self.pickle_path = Path(
            pickle_path
        ).resolve()

        self.documents_path = Path(
            documents_path
        ).resolve()

        self.manifest_path = Path(
            manifest_path
        ).resolve()

        self.k1 = float(
            k1
        )

        self.b = float(
            b
        )

        self._index: Any | None = None

        self._loaded_signature: tuple[
            tuple[bool, int, int],
            tuple[bool, int, int],
            tuple[bool, int, int],
        ] | None = None

        self._load_method = "not_loaded"
        self._document_count = 0
        self._last_candidate_count = 0
        self._last_applied_filters: dict[str, Any] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _file_signature(
        path: Path,
    ) -> tuple[
        bool,
        int,
        int,
    ]:
        """读取文件存在状态、修改时间和大小。"""

        try:
            stat = path.stat()

        except OSError:
            return (
                False,
                0,
                0,
            )

        return (
            True,
            int(
                stat.st_mtime_ns
            ),
            int(
                stat.st_size
            ),
        )

    def _current_signature(
        self,
    ) -> tuple[
        tuple[bool, int, int],
        tuple[bool, int, int],
        tuple[bool, int, int],
    ]:
        """计算当前 GitHub 情报索引签名。"""

        return (
            self._file_signature(
                self.pickle_path
            ),
            self._file_signature(
                self.documents_path
            ),
            self._file_signature(
                self.manifest_path
            ),
        )

    def _load_index(
        self,
    ) -> None:
        """加载或重建 GitHub BM25 索引。"""

        (
            index,
            load_method,
            document_count,
        ) = load_or_rebuild_bm25_index(
            pickle_path=(
                self.pickle_path
            ),
            documents_path=(
                self.documents_path
            ),
            k1=self.k1,
            b=self.b,
        )

        if not callable(
            getattr(
                index,
                "search",
                None,
            )
        ):
            raise TypeError(
                "GitHub BM25 索引没有可调用的 "
                "search() 方法。"
            )

        self._index = index
        self._load_method = str(
            load_method
        )
        self._document_count = int(
            document_count
        )

        self._loaded_signature = (
            self._current_signature()
        )

    def _ensure_loaded(
        self,
    ) -> None:
        """在首次搜索或索引变化后加载索引。"""

        current_signature = (
            self._current_signature()
        )

        if (
            self._index is not None
            and self._loaded_signature
            == current_signature
        ):
            return

        with self._lock:
            current_signature = (
                self._current_signature()
            )

            if (
                self._index is not None
                and self._loaded_signature
                == current_signature
            ):
                return

            self._load_index()

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        snapshot_date: str | None = None,
        doc_types: Sequence[str] | None = None,
        repository: str | None = None,
        topic_name: str | None = None,
    ) -> Sequence[Any]:
        """搜索最新 GitHub 情报 BM25 索引。

        如果提供 metadata 硬约束，则先过滤候选文档，再在
        过滤后的候选集内部执行 BM25；否则保持原来的全库
        BM25 行为。
        """

        normalized_query = str(
            query
        ).strip()

        if not normalized_query:
            return []

        requested_top_k = int(
            top_k
        )

        if requested_top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0。"
            )

        self._ensure_loaded()

        if self._index is None:
            raise RuntimeError(
                "GitHub BM25 索引加载后仍为空。"
            )

        applied_filters = (
            build_applied_github_filters(
                snapshot_date=(
                    snapshot_date
                ),
                doc_types=doc_types,
                repository=repository,
                topic_name=topic_name,
            )
        )

        self._last_applied_filters = dict(
            applied_filters
        )

        if applied_filters:
            (
                results,
                candidate_count,
            ) = search_bm25_with_metadata_filters(
                self._index,
                normalized_query,
                top_k=requested_top_k,
                k1=self.k1,
                b=self.b,
                snapshot_date=(
                    snapshot_date
                ),
                doc_types=doc_types,
                repository=repository,
                topic_name=topic_name,
            )

            self._last_candidate_count = int(
                candidate_count
            )

            return results

        self._last_candidate_count = int(
            self._document_count
        )

        search_method = getattr(
            self._index,
            "search",
            None,
        )

        if not callable(
            search_method
        ):
            raise TypeError(
                "GitHub BM25 索引没有可调用的 "
                "search() 方法。"
            )

        try:
            return search_method(
                query=normalized_query,
                top_k=requested_top_k,
            )

        except TypeError:
            return search_method(
                normalized_query,
                requested_top_k,
            )

    def status(
        self,
    ) -> dict[str, Any]:
        """返回当前缓存索引状态。"""

        return {
            "loaded": (
                self._index is not None
            ),
            "load_method": (
                self._load_method
            ),
            "document_count": (
                self._document_count
            ),
            "last_candidate_count": (
                self._last_candidate_count
            ),
            "last_applied_filters": dict(
                self._last_applied_filters
            ),
            "pickle_path": str(
                self.pickle_path
            ),
            "documents_path": str(
                self.documents_path
            ),
            "manifest_path": str(
                self.manifest_path
            ),
        }


def create_github_intelligence_search_tool(
    *,
    default_top_k: int = 5,
    maximum_top_k: int = 10,
    max_characters_per_document: int = 1800,
    pickle_path: Path = (
        DEFAULT_BM25_PICKLE_PATH
    ),
    documents_path: Path = (
        DEFAULT_BM25_DOCUMENTS_PATH
    ),
    manifest_path: Path = (
        DEFAULT_MANIFEST_PATH
    ),
) -> BaseTool:
    """创建 GitHub 技术情报搜索 Tool。"""

    if default_top_k <= 0:
        raise ValueError(
            "GitHub default_top_k 必须大于 0。"
        )

    if maximum_top_k <= 0:
        raise ValueError(
            "GitHub maximum_top_k 必须大于 0。"
        )

    if (
        default_top_k
        > maximum_top_k
    ):
        raise ValueError(
            "GitHub default_top_k 不能大于 "
            "maximum_top_k。"
        )

    if (
        max_characters_per_document
        <= 0
    ):
        raise ValueError(
            "GitHub max_characters_per_document "
            "必须大于 0。"
        )

    reloadable_index = (
        ReloadableGitHubBM25Index(
            pickle_path=pickle_path,
            documents_path=(
                documents_path
            ),
            manifest_path=(
                manifest_path
            ),
        )
    )

    @tool(
        "search_github_intelligence",
        args_schema=(
            GitHubIntelligenceSearchInput
        ),
    )
    def search_github_intelligence(
        query: str,
        top_k: int = default_top_k,
        snapshot_date: str | None = None,
        doc_types: list[str] | None = None,
        repository: str | None = None,
        topic_name: str | None = None,
    ) -> str:
        """搜索已经采集的 GitHub 技术情报。

        适用于：

        1. 查询已收录的 GitHub 项目；
        2. 查询某天的 GitHub 热点；
        3. 比较多个项目或多个日期；
        4. 查询 GitHub 日报、项目摘要和技术趋势；
        5. 在更新完成后读取新生成的索引。

        对日期、文档类型、仓库名等确定性条件，优先使用
        snapshot_date、doc_types、repository、topic_name 做
        metadata pre-filter；不要只把这些硬约束写进 query。

        本工具只读取已有索引，不访问 GitHub 网站，
        也不会触发重新采集。

        用户明确要求更新、刷新、同步或重新采集 GitHub 信息时，
        应先加载 github-intelligence-update Skill，
        再使用该 Skill 开放的 update_github_intelligence。
        """

        normalized_query = str(
            query
        ).strip()

        if not normalized_query:
            return (
                "工具调用失败："
                "GitHub 情报检索问题不能为空。"
            )

        requested_top_k = int(
            top_k
        )

        effective_top_k = max(
            1,
            min(
                requested_top_k,
                maximum_top_k,
            ),
        )

        try:
            resolved_snapshot_date = (
                resolve_snapshot_date_filter(
                    snapshot_date
                )
            )

            normalized_doc_types = (
                normalize_doc_type_filters(
                    doc_types
                )
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            return (
                "GitHub 技术情报 metadata 过滤参数错误："
                f"{exc}"
            )

        applied_filters = (
            build_applied_github_filters(
                snapshot_date=(
                    resolved_snapshot_date
                ),
                doc_types=(
                    normalized_doc_types
                ),
                repository=repository,
                topic_name=topic_name,
            )
        )

        try:
            raw_results = (
                reloadable_index.search(
                    query=normalized_query,
                    top_k=effective_top_k,
                    snapshot_date=(
                        resolved_snapshot_date
                    ),
                    doc_types=(
                        normalized_doc_types
                    ),
                    repository=repository,
                    topic_name=topic_name,
                )
            )

        except FileNotFoundError as exc:
            return (
                "GitHub 技术情报索引尚不存在。\n"
                "请先加载 github-intelligence-update Skill，\n"
                "再调用该 Skill 开放的 "
                "update_github_intelligence 完成采集和索引构建。\n"
                f"详细信息：{exc}"
            )

        except Exception as exc:
            return (
                "GitHub 技术情报检索失败："
                f"{type(exc).__name__}：{exc}"
            )

        return format_github_search_results(
            query=normalized_query,
            raw_results=list(
                raw_results
            ),
            max_characters_per_document=(
                max_characters_per_document
            ),
            index_status=(
                reloadable_index.status()
            ),
            applied_filters=(
                applied_filters
            ),
            candidate_count=(
                reloadable_index.status().get(
                    "last_candidate_count"
                )
            ),
        )

    return search_github_intelligence


# ============================================================
# 完整 Agent 工具列表
# ============================================================


def validate_unique_tool_names(
    tools: Sequence[BaseTool],
) -> None:
    """检查工具列表中是否存在重名工具。"""

    seen_names: set[str] = set()
    duplicate_names: set[str] = set()

    for current_tool in tools:
        if not isinstance(
            current_tool,
            BaseTool,
        ):
            raise TypeError(
                "工具列表中只能包含 BaseTool："
                f"{type(current_tool)!r}"
            )

        tool_name = str(
            current_tool.name
        ).strip()

        if not tool_name:
            raise ValueError(
                "工具名称不能为空。"
            )

        if tool_name in seen_names:
            duplicate_names.add(
                tool_name
            )

        seen_names.add(
            tool_name
        )

    if duplicate_names:
        raise ValueError(
            "工具列表存在重名工具："
            + ", ".join(
                sorted(
                    duplicate_names
                )
            )
        )


def create_agent_tools(
    bm25_index: BM25IndexProtocol,
    *,
    default_top_k: int = 5,
    maximum_top_k: int = 10,
    max_characters_per_document: int = 1500,

    # GitHub 技术情报 RAG。
    include_github_search: bool = True,

    # GitHub SQLite Schema 查询。
    include_github_schema: bool = True,

    # GitHub SQLite 只读 SQL 查询。
    include_github_sql: bool = True,

    github_database_path: Path = (
        DEFAULT_GITHUB_INTELLIGENCE_DATABASE_PATH
    ),

    skill_runtime: SkillRuntime | None = None,

    github_default_top_k: int | None = None,
    github_maximum_top_k: int | None = None,

    github_max_characters_per_document: int = 1800,
) -> list[BaseTool]:
    """创建供 Agent 启动时使用的基础 Tool 列表。

    默认业务 Tool：

    1. search_knowledge_base；
       用于 PDF 学习知识库语义检索。

    2. search_github_intelligence；
       用于 GitHub 技术情报 RAG 语义检索。

    3. get_github_intelligence_schema；
       用于按需获取当前 Agent 可访问的
       GitHub Intelligence SQLite Schema。

    4. query_github_intelligence_sql；
       用于执行基于当前可见 Schema 生成的
       SQLite 只读查询。

    如果传入 skill_runtime，则额外加入：

    5. list_skills；
    6. load_skill。

    Skill 专属 Tool 不在这里静态注册。

    例如 update_github_intelligence 只有在
    github-intelligence-update Skill 被加载后，
    才进入 Active Tools。
    """

    tools: list[BaseTool] = [
        create_bm25_search_tool(
            bm25_index=bm25_index,
            default_top_k=default_top_k,
            maximum_top_k=maximum_top_k,
            max_characters_per_document=(
                max_characters_per_document
            ),
        )
    ]

    if include_github_schema:
        tools.append(
            create_github_intelligence_schema_tool(
                database_path=(
                    github_database_path
                )
            )
        )
        
    if include_github_search:
        tools.append(
            create_github_intelligence_search_tool(
                default_top_k=(
                    default_top_k
                    if github_default_top_k
                    is None
                    else int(
                        github_default_top_k
                    )
                ),
                maximum_top_k=(
                    maximum_top_k
                    if github_maximum_top_k
                    is None
                    else int(
                        github_maximum_top_k
                    )
                ),
                max_characters_per_document=(
                    github_max_characters_per_document
                ),
            )
        )

    if include_github_sql:
        tools.append(
            create_github_intelligence_sql_tool(
                database_path=(
                    github_database_path
                )
            )
        )

    if skill_runtime is not None:
        tools.extend(
            skill_runtime.get_control_tools()
        )

    validate_unique_tool_names(
        tools
    )

    return tools


def get_tool_names(
    tools: Sequence[BaseTool],
) -> list[str]:
    """返回工具名称，便于日志输出和测试。"""

    return [
        str(
            current_tool.name
        )
        for current_tool in tools
    ]