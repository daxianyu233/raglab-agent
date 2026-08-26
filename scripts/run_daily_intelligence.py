from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[assignment]


# ============================================================
# 配置区
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# None 表示使用北京时间当天。
# 需要补跑历史日期时，可以写成：
# RUN_DATE = "2026-08-03"
RUN_DATE: str | None = None

# False：每次执行完整流水线。
# True：如果某一步的结果已经完整存在，则跳过该步骤。
SKIP_COMPLETED_STEPS = False

# 是否在最后重建持久化 BM25 + Chroma 索引。
REBUILD_RAG_INDEXES = True


REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "intelligence_runs"
)

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "intelligence"
    / "raw"
)

DEEP_ROOT = (
    PROJECT_ROOT
    / "data"
    / "intelligence"
    / "deep"
)

RAG_ROOT = (
    PROJECT_ROOT
    / "data"
    / "intelligence"
    / "rag_documents"
)

INDEX_MANIFEST = (
    PROJECT_ROOT
    / "storage"
    / "intelligence"
    / "rag_index_manifest.json"
)

BM25_DOCUMENTS = (
    PROJECT_ROOT
    / "storage"
    / "bm25"
    / "intelligence"
    / "documents.jsonl"
)

CHROMA_DIR = (
    PROJECT_ROOT
    / "storage"
    / "chroma"
    / "intelligence"
)


Validator = Callable[
    [str],
    tuple[bool, str],
]


@dataclass(frozen=True)
class Step:
    name: str
    script: str | None
    required_env: tuple[str, ...]
    validator: Validator
    enabled: bool = True


# ============================================================
# 日期
# ============================================================


def beijing_now() -> datetime:
    """
    获取北京时间。

    如果当前环境没有 tzdata，则使用固定 UTC+8。
    """
    if ZoneInfo is not None:
        try:
            return datetime.now(
                ZoneInfo(
                    "Asia/Shanghai"
                )
            )
        except Exception:
            pass

    return datetime.now(
        timezone(
            timedelta(
                hours=8
            )
        )
    )


def resolve_run_date() -> str:
    """
    获取本次流水线使用的日期。
    """
    if RUN_DATE:
        datetime.strptime(
            RUN_DATE,
            "%Y-%m-%d",
        )

        return RUN_DATE

    return beijing_now().strftime(
        "%Y-%m-%d"
    )


# ============================================================
# 永久环境变量
# ============================================================


def load_user_environment(
    variable_name: str,
) -> str:
    """
    读取环境变量。

    优先级：

    1. 当前 Python 进程；
    2. Windows 当前用户永久环境变量。

    因此即使 VS Code 是设置密钥之前打开的，
    也不需要每天手动重新执行 $env:XXX=...。
    """
    value = os.getenv(
        variable_name,
        "",
    ).strip()

    if (
        not value
        and os.name == "nt"
    ):
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                "Environment",
            ) as key:
                stored_value, _ = (
                    winreg.QueryValueEx(
                        key,
                        variable_name,
                    )
                )

            value = str(
                stored_value
            ).strip()

            if value:
                os.environ[
                    variable_name
                ] = value

        except (
            FileNotFoundError,
            OSError,
        ):
            value = ""

    if not value:
        raise RuntimeError(
            f"未找到环境变量 {variable_name}。"
        )

    if not value.isascii():
        raise RuntimeError(
            f"{variable_name} 包含非 ASCII 字符，"
            "很可能保存了中文占位符。"
        )

    return value


# ============================================================
# 子脚本参数识别
# ============================================================


def supports_date_argument(
    script_path: Path,
) -> bool:
    """
    检查脚本是否支持 --date。

    这里不执行 --help，避免某些没有使用 argparse
    的脚本直接进入实际运行流程。
    """
    source = script_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    return bool(
        re.search(
            r"""["']--date["']""",
            source,
        )
    )


# ============================================================
# 子脚本执行
# ============================================================


def run_script(
    script_path: Path,
    run_date: str,
    log_file,
) -> int:
    """
    执行一个 Python 子脚本，并同时输出到：

    1. 当前终端；
    2. 本次运行日志。
    """
    command = [
        sys.executable,
        str(
            script_path
        ),
    ]

    if supports_date_argument(
        script_path
    ):
        command.extend(
            [
                "--date",
                run_date,
            ]
        )

    display_command = " ".join(
        (
            f'"{part}"'
            if " " in part
            else part
        )
        for part in command
    )

    print(
        f"执行：{display_command}"
    )

    log_file.write(
        f"\n执行：{display_command}\n"
    )

    log_file.flush()

    child_environment = (
        os.environ.copy()
    )

    child_environment[
        "PYTHONUTF8"
    ] = "1"

    child_environment[
        "PYTHONIOENCODING"
    ] = "utf-8"

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(
            line,
            end="",
        )

        log_file.write(
            line
        )

        log_file.flush()

    return process.wait()


# ============================================================
# 文件搜索工具
# ============================================================


def files_named(
    root: Path,
    *names: str,
) -> list[Path]:
    """
    在指定目录中递归查找特定文件名。
    """
    if not root.exists():
        return []

    wanted_names = set(
        names
    )

    return sorted(
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.name in wanted_names
        )
    )


# ============================================================
# 各步骤结果校验
# ============================================================


def validate_raw(
    run_date: str,
) -> tuple[bool, str]:
    """
    检查候选项目原始采集结果。
    """
    folder = (
        RAW_ROOT
        / run_date
    )

    if folder.exists():
        files = [
            path
            for path in folder.rglob("*")
            if path.is_file()
        ]
    else:
        files = []

    if not files:
        return (
            False,
            f"未生成原始采集文件：{folder}",
        )

    return (
        True,
        f"原始采集文件数：{len(files)}",
    )


def validate_deep(
    run_date: str,
) -> tuple[bool, str]:
    """
    检查深度采集入口。
    """
    latest_collection_path = (
        DEEP_ROOT
        / run_date
        / "latest_collection.json"
    )

    if not latest_collection_path.exists():
        return (
            False,
            "缺少深度采集入口："
            f"{latest_collection_path}",
        )

    return (
        True,
        "深度采集入口："
        f"{latest_collection_path}",
    )


def validate_update_detection(
    run_date: str,
) -> tuple[bool, str]:
    """
    检查仓库重复与变化检测结果。

    该文件由 detect_github_repository_updates.py 生成，
    后续逐项目 LLM 分析应依据其中的 llm_action 和
    llm_should_run 决定重新分析还是复用历史摘要。
    """
    decision_path = (
        DEEP_ROOT
        / run_date
        / "repository_update_decisions.json"
    )

    if not decision_path.is_file():
        return (
            False,
            "缺少仓库变化检测结果："
            f"{decision_path}",
        )

    try:
        payload = json.loads(
            decision_path.read_text(
                encoding="utf-8",
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        return (
            False,
            "仓库变化检测结果无法读取："
            f"{decision_path}；{exc}",
        )

    if not isinstance(payload, dict):
        return (
            False,
            "仓库变化检测结果根节点不是对象："
            f"{decision_path}",
        )

    payload_date = str(
        payload.get("snapshot_date")
        or ""
    ).strip()

    if payload_date != run_date:
        return (
            False,
            "仓库变化检测日期不匹配："
            f"期望 {run_date}，实际 {payload_date or '缺失'}",
        )

    repositories = payload.get(
        "repositories"
    )

    if not isinstance(
        repositories,
        list,
    ):
        return (
            False,
            "仓库变化检测结果缺少 repositories 列表："
            f"{decision_path}",
        )

    summary = payload.get(
        "summary"
    )

    if not isinstance(
        summary,
        dict,
    ):
        return (
            False,
            "仓库变化检测结果缺少 summary："
            f"{decision_path}",
        )

    expected_count = summary.get(
        "repository_count"
    )

    if (
        isinstance(expected_count, int)
        and expected_count != len(repositories)
    ):
        return (
            False,
            "仓库变化检测数量不一致："
            f"summary={expected_count}，"
            f"repositories={len(repositories)}",
        )

    llm_required_count = sum(
        1
        for repository in repositories
        if isinstance(repository, dict)
        and bool(
            repository.get(
                "llm_should_run"
            )
        )
    )

    llm_reuse_count = (
        len(repositories)
        - llm_required_count
    )

    return (
        True,
        "仓库变化检测完成："
        f"项目 {len(repositories)} 个，"
        f"需调用 LLM {llm_required_count} 个，"
        f"可复用摘要 {llm_reuse_count} 个",
    )


def validate_project_analysis(
    run_date: str,
) -> tuple[bool, str]:
    """
    检查逐项目 LLM 分析结果。
    """
    files = files_named(
        DEEP_ROOT
        / run_date,
        "repository_llm_summaries.json",
    )

    if not files:
        return (
            False,
            "未生成 "
            "repository_llm_summaries.json",
        )

    return (
        True,
        f"项目摘要：{files[-1]}",
    )


def validate_hotspots(
    run_date: str,
) -> tuple[bool, str]:
    """
    检查跨项目热点和完整日报。
    """
    files = files_named(
        DEEP_ROOT
        / run_date,
        "daily_hotspots.json",
        "github_daily_report.md",
    )

    existing_names = {
        path.name
        for path in files
    }

    required_names = {
        "daily_hotspots.json",
        "github_daily_report.md",
    }

    missing_names = (
        required_names
        - existing_names
    )

    if missing_names:
        return (
            False,
            "热点归纳缺少："
            + ", ".join(
                sorted(
                    missing_names
                )
            ),
        )

    return (
        True,
        "热点 JSON 和完整日报已生成",
    )


def validate_brief(
    run_date: str,
) -> tuple[bool, str]:
    """
    检查精简日报。
    """
    files = files_named(
        DEEP_ROOT
        / run_date,
        "github_daily_brief.json",
        "github_daily_brief.md",
    )

    existing_names = {
        path.name
        for path in files
    }

    required_names = {
        "github_daily_brief.json",
        "github_daily_brief.md",
    }

    missing_names = (
        required_names
        - existing_names
    )

    if missing_names:
        return (
            False,
            "精简日报缺少："
            + ", ".join(
                sorted(
                    missing_names
                )
            ),
        )

    return (
        True,
        "精简日报 JSON 和 Markdown 已生成",
    )


def validate_rag_documents(
    run_date: str,
) -> tuple[bool, str]:
    """
    检查当天统一 RAG 文档源。
    """
    folder = (
        RAG_ROOT
        / run_date
    )

    if folder.exists():
        existing_names = {
            path.name
            for path in folder.glob(
                "*.jsonl"
            )
        }
    else:
        existing_names = set()

    required_names = {
        "repository_summaries.jsonl",
        "daily_hotspots.jsonl",
        "daily_brief.jsonl",
    }

    missing_names = (
        required_names
        - existing_names
    )

    if missing_names:
        return (
            False,
            "RAG 文档缺少："
            + ", ".join(
                sorted(
                    missing_names
                )
            ),
        )

    return (
        True,
        f"RAG 文档已保存：{folder}",
    )


def validate_indexes(
    run_date: str,
) -> tuple[bool, str]:
    """
    检查持久化 BM25 和 Chroma 索引。
    """
    missing_paths: list[
        str
    ] = []

    if not INDEX_MANIFEST.exists():
        missing_paths.append(
            str(
                INDEX_MANIFEST
            )
        )

    if not BM25_DOCUMENTS.exists():
        missing_paths.append(
            str(
                BM25_DOCUMENTS
            )
        )

    if not CHROMA_DIR.exists():
        missing_paths.append(
            str(
                CHROMA_DIR
            )
        )

    if missing_paths:
        return (
            False,
            "持久化索引缺少："
            + "; ".join(
                missing_paths
            ),
        )

    return (
        True,
        "BM25、Chroma 和索引清单已更新",
    )


# ============================================================
# 主流程
# ============================================================


def main() -> int:
    run_date = resolve_run_date()

    github_token = (
        load_user_environment(
            "GITHUB_TOKEN"
        )
    )

    deepseek_key = (
        load_user_environment(
            "DEEPSEEK_API_KEY"
        )
    )

    print(
        "密钥检查通过："
        f"GITHUB_TOKEN 长度="
        f"{len(github_token)}，"
        f"DEEPSEEK_API_KEY 长度="
        f"{len(deepseek_key)}"
    )

    steps = [
        Step(
            name="GitHub 候选项目采集",
            script=(
                "collect_github_intelligence.py"
            ),
            required_env=(
                "GITHUB_TOKEN",
            ),
            validator=validate_raw,
        ),

        Step(
            name="GitHub 仓库深度采集",
            script=(
                "collect_github_repository_details.py"
            ),
            required_env=(
                "GITHUB_TOKEN",
            ),
            validator=validate_deep,
        ),

        Step(
            name="仓库重复与变化检测",
            script=(
                "detect_github_repository_updates.py"
            ),
            required_env=(),
            validator=(
                validate_update_detection
            ),
        ),

        Step(
            name="逐项目 DeepSeek 分析",
            script=(
                "analyze_github_projects.py"
            ),
            required_env=(
                "DEEPSEEK_API_KEY",
            ),
            validator=(
                validate_project_analysis
            ),
        ),

        Step(
            name="跨项目热点归纳",
            script=(
                "summarize_github_daily_hotspots.py"
            ),
            required_env=(
                "DEEPSEEK_API_KEY",
            ),
            validator=validate_hotspots,
        ),

        Step(
            name="生成精简日报",
            script=(
                "generate_github_daily_brief.py"
            ),
            required_env=(),
            validator=validate_brief,
        ),

        Step(
            name="检查当天 RAG 文档",
            script=None,
            required_env=(),
            validator=(
                validate_rag_documents
            ),
        ),

        Step(
            name="重建持久化 RAG 索引",
            script=(
                "build_intelligence_indexes.py"
            ),
            required_env=(),
            validator=validate_indexes,
            enabled=(
                REBUILD_RAG_INDEXES
            ),
        ),
    ]

    started_at = beijing_now()

    run_folder = (
        REPORT_ROOT
        / run_date
    )

    run_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        started_at.strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    log_path = (
        run_folder
        / (
            "daily_intelligence_"
            f"{timestamp}.log"
        )
    )

    summary_path = (
        run_folder
        / (
            "daily_intelligence_"
            f"{timestamp}.json"
        )
    )

    results: list[
        dict
    ] = []

    enabled_steps = [
        step
        for step in steps
        if step.enabled
    ]

    print()
    print(
        "=" * 80
    )

    print(
        "GitHub 每日技术情报自动流水线"
    )

    print(
        f"日期：{run_date}"
    )

    print(
        f"日志：{log_path}"
    )

    print(
        "=" * 80
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:

        for index, step in enumerate(
            enabled_steps,
            start=1,
        ):
            print()

            print(
                f"[{index}/"
                f"{len(enabled_steps)}] "
                f"{step.name}"
            )

            print(
                "-" * 80
            )

            for variable_name in (
                step.required_env
            ):
                load_user_environment(
                    variable_name
                )

            if SKIP_COMPLETED_STEPS:
                complete, message = (
                    step.validator(
                        run_date
                    )
                )

                if complete:
                    print(
                        "复用已有结果："
                        f"{message}"
                    )

                    results.append(
                        {
                            "name": (
                                step.name
                            ),

                            "status": (
                                "success"
                            ),

                            "message": (
                                "复用已有结果："
                                + message
                            ),
                        }
                    )

                    continue

            step_started_at = (
                time.perf_counter()
            )

            if step.script:
                script_path = (
                    PROJECT_ROOT
                    / "scripts"
                    / step.script
                )

                if not script_path.exists():
                    message = (
                        "脚本不存在："
                        f"{script_path}"
                    )

                    print(
                        message
                    )

                    results.append(
                        {
                            "name": step.name,
                            "status": "failed",
                            "message": message,
                        }
                    )

                    break

                exit_code = run_script(
                    script_path,
                    run_date,
                    log_file,
                )

                if exit_code != 0:
                    message = (
                        "脚本退出码："
                        f"{exit_code}"
                    )

                    print(
                        f"步骤失败：{message}"
                    )

                    results.append(
                        {
                            "name": step.name,
                            "status": "failed",
                            "message": message,
                        }
                    )

                    break

            valid, message = (
                step.validator(
                    run_date
                )
            )

            elapsed_seconds = round(
                time.perf_counter()
                - step_started_at,
                3,
            )

            if not valid:
                print(
                    "结果检查失败："
                    f"{message}"
                )

                results.append(
                    {
                        "name": step.name,
                        "status": "failed",
                        "message": message,
                        "elapsed_seconds": (
                            elapsed_seconds
                        ),
                    }
                )

                break

            print(
                f"完成：{message}"
            )

            results.append(
                {
                    "name": step.name,
                    "status": "success",
                    "message": message,
                    "elapsed_seconds": (
                        elapsed_seconds
                    ),
                }
            )

    finished_at = beijing_now()

    success = (
        len(
            results
        )
        == len(
            enabled_steps
        )
        and all(
            result[
                "status"
            ]
            == "success"
            for result in results
        )
    )

    summary = {
        "status": (
            "success"
            if success
            else "failed"
        ),

        "run_date": run_date,

        "started_at": (
            started_at.isoformat()
        ),

        "finished_at": (
            finished_at.isoformat()
        ),

        "elapsed_seconds": round(
            (
                finished_at
                - started_at
            ).total_seconds(),
            3,
        ),

        "project_root": str(
            PROJECT_ROOT
        ),

        "python": (
            sys.executable
        ),

        "log_path": str(
            log_path
        ),

        "steps": results,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 80
    )

    print(
        "执行成功"
        if success
        else "执行未完整完成"
    )

    print(
        f"日期：{run_date}"
    )

    print(
        f"日志：{log_path}"
    )

    print(
        f"摘要：{summary_path}"
    )

    print(
        "当天 RAG 文档："
        f"{RAG_ROOT / run_date}"
    )

    print(
        f"索引清单：{INDEX_MANIFEST}"
    )

    print(
        "=" * 80
    )

    return (
        0
        if success
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )