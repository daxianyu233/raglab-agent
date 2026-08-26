from __future__ import annotations

from datetime import datetime, timedelta

from raglab.agent.tools import (
    create_github_intelligence_search_tool,
    resolve_snapshot_date_filter,
)


def require_text(
    output: str,
    expected: str,
    label: str,
) -> None:
    if expected not in output:
        raise AssertionError(
            f"[{label}] 未找到预期文本：{expected!r}\n\n"
            f"实际输出：\n{output}"
        )


def main() -> None:
    tool = create_github_intelligence_search_tool()

    local_today = (
        datetime.now()
        .astimezone()
        .date()
    )
    local_yesterday = (
        local_today
        - timedelta(days=1)
    )

    resolved_today = resolve_snapshot_date_filter(
        "today"
    )
    resolved_yesterday = resolve_snapshot_date_filter(
        "yesterday"
    )

    assert resolved_today == local_today.isoformat()
    assert resolved_yesterday == local_yesterday.isoformat()

    print("=" * 80)
    print("1. 今日日报 metadata pre-filter")
    print("=" * 80)

    today_output = tool.invoke(
        {
            "query": "GitHub 技术情报日报",
            "snapshot_date": "today",
            "doc_types": [
                "daily_brief"
            ],
            "top_k": 5,
        }
    )

    print(today_output)

    require_text(
        today_output,
        f"'snapshot_date': '{local_today.isoformat()}'",
        "today filter",
    )
    require_text(
        today_output,
        "'doc_types': ['daily_brief']",
        "daily_brief filter",
    )
    require_text(
        today_output,
        f"日期：{local_today.isoformat()}",
        "today result date",
    )
    require_text(
        today_output,
        "类型：daily_brief",
        "today result type",
    )

    print()
    print("[PASS] 今日日报只返回今日 daily_brief。")

    print()
    print("=" * 80)
    print("2. 昨日日报 metadata pre-filter")
    print("=" * 80)

    yesterday_output = tool.invoke(
        {
            "query": "GitHub 技术情报日报",
            "snapshot_date": "yesterday",
            "doc_types": [
                "daily_brief"
            ],
            "top_k": 5,
        }
    )

    print(yesterday_output)

    require_text(
        yesterday_output,
        f"'snapshot_date': '{local_yesterday.isoformat()}'",
        "yesterday filter",
    )
    require_text(
        yesterday_output,
        "'doc_types': ['daily_brief']",
        "yesterday daily_brief filter",
    )
    require_text(
        yesterday_output,
        f"日期：{local_yesterday.isoformat()}",
        "yesterday result date",
    )

    print()
    print("[PASS] 昨日日报只返回昨日 daily_brief。")

    print()
    print("=" * 80)
    print("3. 今日热点 metadata pre-filter")
    print("=" * 80)

    hotspot_output = tool.invoke(
        {
            "query": "GitHub 技术热点",
            "snapshot_date": "today",
            "doc_types": [
                "daily_hotspot"
            ],
            "top_k": 5,
        }
    )

    print(hotspot_output)

    require_text(
        hotspot_output,
        f"日期：{local_today.isoformat()}",
        "hotspot date",
    )
    require_text(
        hotspot_output,
        "类型：daily_hotspot",
        "hotspot type",
    )

    print()
    print("[PASS] 今日热点只在今日 daily_hotspot 候选集内排序。")

    print()
    print("=" * 80)
    print("Metadata-aware Retrieval 回归测试通过")
    print("=" * 80)


if __name__ == "__main__":
    main()