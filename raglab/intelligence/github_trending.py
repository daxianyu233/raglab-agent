from __future__ import annotations

import re
from collections.abc import Iterable

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

from .models import TrendingRepository


# 用于解析 GitHub 页面中的数字。
# 支持：
# 1,234
# 12.5k
# 3.2m
_COUNT_PATTERN = re.compile(
    r"([\d,.]+)\s*([kKmM]?)"
)


def parse_count(text: str | None) -> int:
    """
    将 GitHub 页面中的数字文本转换成整数。

    示例：
    1,234  -> 1234
    12.5k  -> 12500
    3.2m   -> 3200000

    无法解析时返回 0。
    """
    if not text:
        return 0

    cleaned_text = text.strip().replace(",", "")
    match = _COUNT_PATTERN.search(cleaned_text)

    if match is None:
        return 0

    number_text = match.group(1)
    suffix = match.group(2).lower()

    try:
        value = float(number_text)
    except ValueError:
        return 0

    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000

    return int(value)


def _get_text(node: Tag | None) -> str:
    """
    获取 HTML 节点中的纯文本。

    节点不存在时返回空字符串。
    """
    if node is None:
        return ""

    return node.get_text(
        separator=" ",
        strip=True,
    )


def _extract_repository_full_name(
    article: Tag,
) -> str | None:
    """
    从一个 Trending 仓库节点中提取 owner/repository。

    GitHub Trending 页面中的仓库链接通常位于：

    h2 a
    """
    link = article.select_one("h2 a")

    if not isinstance(link, Tag):
        return None

    # 优先从 href 中提取。
    # href 通常类似：
    # /owner/repository
    href = link.get("href")

    if isinstance(href, str):
        full_name = href.strip("/")

        if full_name.count("/") == 1:
            return full_name

    # 如果 href 格式异常，则退化为读取链接文字。
    text = _get_text(link)
    full_name = re.sub(
        r"\s+",
        "",
        text,
    ).strip("/")

    if full_name.count("/") != 1:
        return None

    return full_name


def _extract_link_count(
    article: Tag,
    path_suffix: str,
) -> int:
    """
    从仓库节点中的链接提取 Star 或 Fork 数量。

    path_suffix 示例：
    /stargazers
    /forks
    """
    links = article.select("a.Link--muted")

    for link in links:
        if not isinstance(link, Tag):
            continue

        href = link.get("href")

        if not isinstance(href, str):
            continue

        normalized_href = href.rstrip("/")

        if normalized_href.endswith(path_suffix):
            return parse_count(_get_text(link))

    return 0


def _extract_period_stars(
    article: Tag,
) -> int:
    """
    提取当前 Trending 周期内新增的 Star 数。

    当 since=daily 时，页面通常显示：
    123 stars today

    当 since=weekly 时，页面通常显示：
    1,234 stars this week

    当 since=monthly 时，页面通常显示：
    3,456 stars this month
    """
    # GitHub 当前页面通常将周期新增 Star
    # 放在右侧的 span 节点中。
    possible_nodes = article.select(
        "span.d-inline-block.float-sm-right"
    )

    for node in possible_nodes:
        if not isinstance(node, Tag):
            continue

        text = _get_text(node)

        if "star" in text.lower():
            return parse_count(text)

    # 如果 GitHub 修改了部分 CSS 类名，
    # 则退化为在整个仓库节点文字中查找。
    article_text = _get_text(article)

    match = re.search(
        pattern=(
            r"([\d,.]+\s*[kKmM]?)"
            r"\s+stars?"
            r"\s+(?:today|this\s+week|this\s+month)"
        ),
        string=article_text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return 0

    return parse_count(match.group(1))


def _extract_description(
    article: Tag,
) -> str:
    """
    提取仓库描述。

    GitHub 当前页面通常将仓库描述放在：
    p.col-9
    """
    description_node = article.select_one("p.col-9")

    if not isinstance(description_node, Tag):
        return ""

    return _get_text(description_node)


def _extract_language(
    article: Tag,
) -> str | None:
    """
    提取仓库主要编程语言。
    """
    language_node = article.select_one(
        '[itemprop="programmingLanguage"]'
    )

    if not isinstance(language_node, Tag):
        return None

    language = _get_text(language_node)

    return language or None


def parse_github_trending_html(
    html: str,
) -> list[TrendingRepository]:
    """
    解析 GitHub Trending 网页 HTML。

    返回按页面顺序排列的热门仓库列表。
    """
    if not html.strip():
        raise ValueError(
            "GitHub Trending HTML 内容为空。"
        )

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # GitHub Trending 页面中，
    # 每个仓库通常对应一个 article.Box-row 节点。
    articles: Iterable[Tag] = soup.select(
        "article.Box-row"
    )

    repositories: list[TrendingRepository] = []

    for article in articles:
        if not isinstance(article, Tag):
            continue

        full_name = _extract_repository_full_name(
            article
        )

        if full_name is None:
            continue

        repository = TrendingRepository(
            full_name=full_name,
            description=_extract_description(article),
            language=_extract_language(article),
            period_stars=_extract_period_stars(article),
            total_stars=_extract_link_count(
                article,
                "/stargazers",
            ),
            forks=_extract_link_count(
                article,
                "/forks",
            ),
            html_url=f"https://github.com/{full_name}",
            rank=len(repositories) + 1,
        )

        repositories.append(repository)

    if not repositories:
        raise RuntimeError(
            "GitHub Trending 页面请求成功，"
            "但没有解析出任何仓库。"
            "GitHub 可能修改了页面 HTML 结构。"
        )

    return repositories


def fetch_github_trending(
    *,
    since: str = "daily",
    spoken_language_code: str = "",
    timeout_seconds: int = 20,
) -> tuple[list[TrendingRepository], str]:
    """
    请求并解析 GitHub Trending 页面。

    参数：
    since：
        daily、weekly 或 monthly。

    spoken_language_code：
        GitHub Trending 的自然语言筛选参数。
        空字符串表示不限制。

    timeout_seconds：
        HTTP 请求超时时间。

    返回：
    1. 解析后的 TrendingRepository 列表；
    2. GitHub 返回的原始 HTML。
    """
    allowed_periods = {
        "daily",
        "weekly",
        "monthly",
    }

    if since not in allowed_periods:
        raise ValueError(
            "since 参数只能是 daily、weekly 或 monthly，"
            f"当前值为：{since}"
        )

    url = "https://github.com/trending"

    params = {
        "since": since,
    }

    if spoken_language_code:
        params["spoken_language_code"] = (
            spoken_language_code
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/150.0.0.0 "
            "Safari/537.36"
        ),
        "Accept": (
            "text/html,"
            "application/xhtml+xml,"
            "application/xml;q=0.9,"
            "*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }

    try:
        response = requests.get(
            url=url,
            params=params,
            headers=headers,
            timeout=timeout_seconds,
        )
    except requests.Timeout as exc:
        raise RuntimeError(
            "请求 GitHub Trending 超时。"
        ) from exc
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "无法连接 GitHub Trending，"
            "请检查网络或代理设置。"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            f"请求 GitHub Trending 失败：{exc}"
        ) from exc

    if not response.ok:
        raise RuntimeError(
            "GitHub Trending 请求失败，"
            f"HTTP 状态码：{response.status_code}，"
            f"响应地址：{response.url}"
        )

    html = response.text

    repositories = parse_github_trending_html(
        html
    )

    return repositories, html