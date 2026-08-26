"""
GitHub 与 ArXiv 技术热点情报模块。

当前第一阶段负责：
1. 获取 GitHub Trending 热点仓库；
2. 调用 GitHub Repository Search API；
3. 补全仓库详情；
4. 保存原始 JSON 和 SQLite 快照。

后续阶段会继续加入：
1. GitHub 热点主题提取；
2. 动态搜索关键词生成；
3. ArXiv 论文采集；
4. 热点趋势查询工具；
5. LangGraph 情报分析 Agent。
"""

from __future__ import annotations

__all__ = [
    "collect_github_intelligence",
]


def collect_github_intelligence(*args, **kwargs):
    """
    延迟导入 GitHub 情报采集函数。

    使用延迟导入可以避免仅导入 raglab.intelligence 时，
    就立即加载 requests、BeautifulSoup、SQLite 等依赖。
    """
    from .collector import collect_github_intelligence as _collect

    return _collect(*args, **kwargs)