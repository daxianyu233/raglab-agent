from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

from raglab.agent.skill_loader import (
    get_skill,
    render_skill_instructions,
)
from raglab.scheduler.github_update_job_coordinator import (
    GithubUpdateJobCoordinator,
)
from raglab.scheduler.job_repository import (
    ScheduledJobRepository,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

TIMEZONE_NAME = "Asia/Shanghai"

SKILL_ID = (
    "github-intelligence-update"
)

PIPELINE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "run_daily_intelligence.py"
)

DEEP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "intelligence"
    / "deep"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "intelligence_runs"
)

INDEX_MANIFEST_PATH = (
    PROJECT_ROOT
    / "storage"
    / "intelligence"
    / "rag_index_manifest.json"
)

LOCK_PATH = (
    PROJECT_ROOT
    / "storage"
    / "intelligence"
    / "github_intelligence_update.lock"
)

LOCK_STALE_SECONDS = (
    6 * 60 * 60
)

PIPELINE_TIMEOUT_SECONDS = (
    2 * 60 * 60
)

OUTPUT_TAIL_LINE_COUNT = 80

ANSI_PATTERN = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"
)


# ============================================================
# Job Coordinator
# ============================================================

_GITHUB_UPDATE_JOB_REPOSITORY = (
    ScheduledJobRepository()
)

_GITHUB_UPDATE_JOB_COORDINATOR = (
    GithubUpdateJobCoordinator(
        repository=(
            _GITHUB_UPDATE_JOB_REPOSITORY
        )
    )
)


# ============================================================
# Exceptions
# ============================================================

class PipelineBusyError(
    RuntimeError,
):
    """已经存在 GitHub Intelligence Pipeline。"""


# ============================================================
# Generic Helpers
# ============================================================

def _beijing_now() -> datetime:

    return datetime.now(
        ZoneInfo(
            TIMEZONE_NAME
        )
    )


def _read_json(
    path: Path | None,
) -> Any:

    if (
        path is None
        or not path.is_file()
    ):
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )

    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None


def _resolve_project_path(
    value: Any,
    reference_path: (
        Path
        | None
    ) = None,
) -> Path | None:

    if not value:
        return None

    candidate = Path(
        str(value)
    )

    if candidate.is_absolute():
        return candidate.resolve()

    if reference_path is not None:

        relative_candidate = (
            reference_path.parent
            / candidate
        ).resolve()

        if relative_candidate.exists():
            return relative_candidate

    return (
        PROJECT_ROOT
        / candidate
    ).resolve()


def _strip_ansi(
    text: str,
) -> str:

    return ANSI_PATTERN.sub(
        "",
        text,
    )


def _tail_lines(
    text: str,
    maximum_lines: int = (
        OUTPUT_TAIL_LINE_COUNT
    ),
) -> str:

    clean_text = _strip_ansi(
        text
    )

    lines = clean_text.splitlines()

    return "\n".join(
        lines[
            -maximum_lines:
        ]
    ).strip()


def _extract_final_path(
    output: str,
    label: str,
) -> Path | None:
    """从 Pipeline 最终输出读取文件路径。

    支持：

    日志：xxx
    摘要：xxx
    """

    prefix = (
        f"{label}："
    )

    for line in reversed(
        _strip_ansi(
            output
        ).splitlines()
    ):

        stripped = line.strip()

        if not stripped.startswith(
            prefix
        ):
            continue

        raw_path = stripped[
            len(prefix):
        ].strip()

        if raw_path:
            return Path(
                raw_path
            ).resolve()

    return None


def _find_latest_report_file(
    run_date: str,
    suffix: str,
    started_timestamp: float,
) -> Path | None:

    run_directory = (
        REPORT_ROOT
        / run_date
    )

    if not run_directory.is_dir():
        return None

    candidates = sorted(
        run_directory.glob(
            f"daily_intelligence_*{suffix}"
        ),
        key=lambda path: (
            path.stat().st_mtime
        ),
        reverse=True,
    )

    for candidate in candidates:

        try:
            modified_time = (
                candidate
                .stat()
                .st_mtime
            )

        except OSError:
            continue

        if (
            modified_time
            >= started_timestamp - 5
        ):
            return candidate.resolve()

    return (
        candidates[0].resolve()
        if candidates
        else None
    )


def _pick(
    data: Any,
    *keys: str,
    default: Any = None,
) -> Any:

    if not isinstance(
        data,
        dict,
    ):
        return default

    for key in keys:

        if key not in data:
            continue

        value = data[
            key
        ]

        if value is not None:
            return value

    return default


def _as_dict(
    value: Any,
) -> dict[str, Any]:

    return (
        value
        if isinstance(
            value,
            dict,
        )
        else {}
    )


# ============================================================
# Bottom-level File Lock
# ============================================================

def _acquire_lock(
) -> dict[str, Any]:
    """使用 O_EXCL 原子创建底层 Pipeline Lock。"""

    LOCK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Stale Lock Recovery
    # --------------------------------------------------------

    if LOCK_PATH.exists():

        try:
            age_seconds = (
                time.time()
                - LOCK_PATH
                .stat()
                .st_mtime
            )

        except OSError:
            age_seconds = 0

        if (
            age_seconds
            >= LOCK_STALE_SECONDS
        ):
            try:
                LOCK_PATH.unlink()

            except OSError:
                pass

    lock_payload = {
        "pid": os.getpid(),

        "started_at": (
            _beijing_now()
            .isoformat()
        ),

        "project_root": str(
            PROJECT_ROOT
        ),
    }

    try:

        file_descriptor = os.open(
            str(
                LOCK_PATH
            ),
            (
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
            ),
        )

    except FileExistsError as exc:

        existing_lock = (
            _read_json(
                LOCK_PATH
            )
        )

        raise PipelineBusyError(
            "GitHub 技术情报更新正在运行，"
            f"锁文件：{LOCK_PATH}；"
            f"锁信息：{existing_lock}"
        ) from exc

    try:

        os.write(
            file_descriptor,
            json.dumps(
                lock_payload,
                ensure_ascii=False,
                indent=2,
            ).encode(
                "utf-8"
            ),
        )

    finally:

        os.close(
            file_descriptor
        )

    return lock_payload


def _release_lock() -> None:

    try:

        LOCK_PATH.unlink(
            missing_ok=True
        )

    except OSError:
        pass


# ============================================================
# Structured Result
# ============================================================

def _collect_structured_result(
    *,
    run_date: str,
    return_code: int,
    stdout: str,
    stderr: str,
    started_timestamp: float,
    elapsed_seconds: float,
) -> dict[str, Any]:

    log_path = (
        _extract_final_path(
            stdout,
            "日志",
        )
    )

    summary_path = (
        _extract_final_path(
            stdout,
            "摘要",
        )
    )

    if log_path is None:

        log_path = (
            _find_latest_report_file(
                run_date,
                ".log",
                started_timestamp,
            )
        )

    if summary_path is None:

        summary_path = (
            _find_latest_report_file(
                run_date,
                ".json",
                started_timestamp,
            )
        )

    pipeline_summary = (
        _read_json(
            summary_path
        )
    )

    latest_pointer_path = (
        DEEP_ROOT
        / run_date
        / "latest_collection.json"
    )

    latest_pointer = (
        _as_dict(
            _read_json(
                latest_pointer_path
            )
        )
    )

    collection_directory = (
        _resolve_project_path(
            latest_pointer.get(
                "collection_directory"
            ),
            latest_pointer_path,
        )
    )

    if (
        collection_directory
        is None
    ):

        analysis_material_path = (
            _resolve_project_path(
                latest_pointer.get(
                    "analysis_material_path"
                ),
                latest_pointer_path,
            )
        )

        if (
            analysis_material_path
            is not None
        ):
            collection_directory = (
                analysis_material_path
                .parent
            )

    decisions_path = (
        DEEP_ROOT
        / run_date
        / (
            "repository_update_"
            "decisions.json"
        )
    )

    decisions = (
        _as_dict(
            _read_json(
                decisions_path
            )
        )
    )

    decisions_summary = (
        _as_dict(
            decisions.get(
                "summary"
            )
        )
    )

    analysis_run_path = (
        _resolve_project_path(
            latest_pointer.get(
                (
                    "repository_llm_"
                    "analysis_run_path"
                )
            ),
            latest_pointer_path,
        )
    )

    if (
        analysis_run_path is None
        and collection_directory
        is not None
    ):
        analysis_run_path = (
            collection_directory
            / (
                "repository_llm_"
                "analysis_run.json"
            )
        )

    analysis_run = (
        _as_dict(
            _read_json(
                analysis_run_path
            )
        )
    )

    hotspot_run_path = None

    if (
        collection_directory
        is not None
    ):
        hotspot_run_path = (
            collection_directory
            / (
                "daily_hotspot_"
                "analysis_run.json"
            )
        )

    hotspot_run = (
        _as_dict(
            _read_json(
                hotspot_run_path
            )
        )
    )

    brief_path = None
    report_path = None

    if (
        collection_directory
        is not None
    ):

        brief_path = (
            collection_directory
            / "github_daily_brief.md"
        )

        report_path = (
            collection_directory
            / "github_daily_report.md"
        )

    index_manifest = (
        _as_dict(
            _read_json(
                INDEX_MANIFEST_PATH
            )
        )
    )

    status = (
        "success"
        if return_code == 0
        else "failed"
    )

    project_usage = (
        _as_dict(
            _pick(
                analysis_run,
                (
                    "total_usage_"
                    "this_run"
                ),
                "total_usage",
                default={},
            )
        )
    )

    hotspot_usage = (
        _as_dict(
            _pick(
                hotspot_run,
                "usage",
                "total_usage",
                "token_usage",
                default={},
            )
        )
    )

    result = {
        "status": status,

        "tool": (
            "update_github_intelligence"
        ),

        "run_date": run_date,

        "return_code": (
            return_code
        ),

        "elapsed_seconds": (
            round(
                elapsed_seconds,
                3,
            )
        ),

        "message": (
            "GitHub 技术情报更新完成。"
            if status == "success"
            else (
                "GitHub 技术情报更新失败。"
            )
        ),

        "change_detection": {
            "repository_count": (
                _pick(
                    decisions_summary,
                    "repository_count",
                    default=0,
                )
            ),

            "llm_required_count": (
                _pick(
                    decisions_summary,
                    "llm_required_count",
                    default=0,
                )
            ),

            "llm_reuse_count": (
                _pick(
                    decisions_summary,
                    "llm_reuse_count",
                    default=0,
                )
            ),

            "status_counts": (
                _pick(
                    decisions_summary,
                    "status_counts",
                    default={},
                )
            ),
        },

        "project_analysis": {
            "planned_llm_count": (
                _pick(
                    analysis_run,
                    "planned_llm_count",
                    default=0,
                )
            ),

            "planned_reuse_count": (
                _pick(
                    analysis_run,
                    "planned_reuse_count",
                    default=0,
                )
            ),

            "new_success_count": (
                _pick(
                    analysis_run,
                    "new_success_count",
                    default=0,
                )
            ),

            "reused_success_count": (
                _pick(
                    analysis_run,
                    "reused_success_count",
                    default=0,
                )
            ),

            "reuse_fallback_llm_count": (
                _pick(
                    analysis_run,
                    (
                        "reuse_fallback_"
                        "llm_count"
                    ),
                    default=0,
                )
            ),

            "llm_failure_count": (
                _pick(
                    analysis_run,
                    "llm_failure_count",
                    default=0,
                )
            ),

            "stored_success_count": (
                _pick(
                    analysis_run,
                    "stored_success_count",
                    default=0,
                )
            ),

            "token_usage": (
                project_usage
            ),
        },

        "hotspots": {
            "status": (
                _pick(
                    hotspot_run,
                    "status",
                    default=None,
                )
            ),

            "topic_count": (
                _pick(
                    hotspot_run,
                    "hotspot_topic_count",
                    "topic_count",
                    "hotspot_count",
                    default=None,
                )
            ),

            "important_repository_count": (
                _pick(
                    hotspot_run,
                    (
                        "important_"
                        "repository_count"
                    ),
                    (
                        "focus_repository_"
                        "count"
                    ),
                    default=None,
                )
            ),

            "token_usage": (
                hotspot_usage
            ),
        },

        "rag": {
            "document_count": (
                _pick(
                    index_manifest,
                    "document_count",
                    "documents_count",
                    default=None,
                )
            ),

            "chunk_count": (
                _pick(
                    index_manifest,
                    "chunk_count",
                    "chunks_count",
                    default=None,
                )
            ),

            "chroma_path": (
                _pick(
                    index_manifest,
                    "chroma_path",
                    "chroma_directory",
                    default=None,
                )
            ),

            "bm25_path": (
                _pick(
                    index_manifest,
                    "bm25_path",
                    "bm25_directory",
                    default=None,
                )
            ),

            "index_manifest_path": (
                str(
                    INDEX_MANIFEST_PATH
                )
            ),
        },

        "paths": {
            "log": (
                str(
                    log_path
                )
                if log_path is not None
                else None
            ),

            "pipeline_summary": (
                str(
                    summary_path
                )
                if summary_path
                is not None
                else None
            ),

            "latest_collection": (
                str(
                    latest_pointer_path
                )
            ),

            "collection_directory": (
                str(
                    collection_directory
                )
                if collection_directory
                is not None
                else None
            ),

            "update_decisions": (
                str(
                    decisions_path
                )
            ),

            "project_analysis_run": (
                str(
                    analysis_run_path
                )
                if analysis_run_path
                is not None
                else None
            ),

            "daily_report": (
                str(
                    report_path
                )
                if (
                    report_path is not None
                    and report_path.is_file()
                )
                else None
            ),

            "daily_brief": (
                str(
                    brief_path
                )
                if (
                    brief_path is not None
                    and brief_path.is_file()
                )
                else None
            ),
        },

        "pipeline_summary": (
            pipeline_summary
            if isinstance(
                pipeline_summary,
                dict,
            )
            else None
        ),

        "stdout_tail": (
            _tail_lines(
                stdout
            )
        ),

        "stderr_tail": (
            _tail_lines(
                stderr
            )
        ),
    }

    return result


# ============================================================
# Raw Pipeline Execution
# ============================================================

def execute_github_intelligence_update(
) -> dict[str, Any]:
    """同步执行 GitHub 技术情报总流水线。

    这一层仍然保留原来的底层文件锁。

    上层：
        Job Single-Flight

    下层：
        File Lock

    第一版双层保护同时保留。
    """

    run_started_at = (
        _beijing_now()
    )

    run_date = (
        run_started_at
        .date()
        .isoformat()
    )

    started_timestamp = (
        time.time()
    )

    if not PIPELINE_SCRIPT.is_file():

        return {
            "status": "failed",

            "tool": (
                "update_github_intelligence"
            ),

            "run_date": run_date,

            "return_code": None,

            "message": (
                "总流水线脚本不存在："
                f"{PIPELINE_SCRIPT}"
            ),
        }

    # --------------------------------------------------------
    # Bottom-level Lock
    # --------------------------------------------------------

    try:

        _acquire_lock()

    except PipelineBusyError as exc:

        return {
            "status": "busy",

            "tool": (
                "update_github_intelligence"
            ),

            "run_date": run_date,

            "return_code": None,

            "message": str(
                exc
            ),

            "paths": {
                "lock": str(
                    LOCK_PATH
                )
            },
        }

    environment = (
        os.environ.copy()
    )

    environment.setdefault(
        "PYTHONUTF8",
        "1",
    )

    environment.setdefault(
        "PYTHONIOENCODING",
        "utf-8",
    )

    try:

        completed = (
            subprocess.run(
                [
                    sys.executable,
                    str(
                        PIPELINE_SCRIPT
                    ),
                ],

                cwd=str(
                    PROJECT_ROOT
                ),

                env=environment,

                capture_output=True,

                text=True,

                encoding="utf-8",

                errors="replace",

                timeout=(
                    PIPELINE_TIMEOUT_SECONDS
                ),

                check=False,
            )
        )

        elapsed_seconds = (
            time.time()
            - started_timestamp
        )

        return (
            _collect_structured_result(
                run_date=run_date,

                return_code=(
                    completed.returncode
                ),

                stdout=(
                    completed.stdout
                    or ""
                ),

                stderr=(
                    completed.stderr
                    or ""
                ),

                started_timestamp=(
                    started_timestamp
                ),

                elapsed_seconds=(
                    elapsed_seconds
                ),
            )
        )

    except subprocess.TimeoutExpired as exc:

        elapsed_seconds = (
            time.time()
            - started_timestamp
        )

        stdout = (
            exc.stdout.decode(
                "utf-8",
                errors="replace",
            )
            if isinstance(
                exc.stdout,
                bytes,
            )
            else (
                exc.stdout
                or ""
            )
        )

        stderr = (
            exc.stderr.decode(
                "utf-8",
                errors="replace",
            )
            if isinstance(
                exc.stderr,
                bytes,
            )
            else (
                exc.stderr
                or ""
            )
        )

        return {
            "status": "timeout",

            "tool": (
                "update_github_intelligence"
            ),

            "run_date": run_date,

            "return_code": None,

            "elapsed_seconds": (
                round(
                    elapsed_seconds,
                    3,
                )
            ),

            "message": (
                "GitHub 技术情报更新"
                "超过最大等待时间。"
            ),

            "paths": {
                "lock": str(
                    LOCK_PATH
                )
            },

            "stdout_tail": (
                _tail_lines(
                    stdout
                )
            ),

            "stderr_tail": (
                _tail_lines(
                    stderr
                )
            ),
        }

    except Exception as exc:

        return {
            "status": "failed",

            "tool": (
                "update_github_intelligence"
            ),

            "run_date": run_date,

            "return_code": None,

            "message": (
                "执行 GitHub 技术情报更新时"
                "出现异常："
                f"{exc}"
            ),

            "paths": {
                "lock": str(
                    LOCK_PATH
                )
            },
        }

    finally:

        _release_lock()


# ============================================================
# LangChain Tool
# ============================================================

@tool(
    "update_github_intelligence",
)
def update_github_intelligence() -> str:
    """更新 GitHub 技术情报。

    仅在用户明确要求更新、刷新、重新采集或同步
    GitHub 技术情报时使用。

    完整流程包括：

    - 候选采集；
    - 深度采集；
    - 同日与跨日重复检测；
    - 项目级 DeepSeek 分析或历史摘要复用；
    - 跨项目热点归纳；
    - 日报生成；
    - BM25 / Chroma 索引更新。

    用户只查询已有 GitHub 情报时不要调用。

    所有执行统一经过 GithubUpdateJobCoordinator，
    从而使：

    Scheduler
    Manual Agent
    Future API

    共享同一个 Job Single-Flight。
    """

    result = (
        _GITHUB_UPDATE_JOB_COORDINATOR
        .execute_tool_call(
            execute_update=(
                execute_github_intelligence_update
            )
        )
    )

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Skill Runtime Integration
# ============================================================

def get_github_intelligence_tools(
) -> list[Any]:
    """返回该 Skill 需要注册的业务 Tool。"""

    return [
        update_github_intelligence
    ]


def get_github_intelligence_skill_prompt(
) -> str:
    """返回 Skill 的完整执行说明。"""

    skill = get_skill(
        SKILL_ID
    )

    return render_skill_instructions(
        skill
    )