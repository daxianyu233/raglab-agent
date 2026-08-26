from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import requests


class GitHubApiError(RuntimeError):
    """
    GitHub API 请求失败。
    """


class GitHubRateLimitError(GitHubApiError):
    """
    GitHub API 达到主限额或次级限额。
    """


@dataclass(slots=True)
class RateLimitSnapshot:
    """
    一次 GitHub API 响应中的限额信息。
    """

    resource: str
    limit: int | None = None
    remaining: int | None = None
    used: int | None = None
    reset_epoch: int | None = None

    @property
    def reset_time_utc(self) -> str | None:
        """
        将限额重置时间转换成 UTC ISO 时间。
        """
        if self.reset_epoch is None:
            return None

        return datetime.fromtimestamp(
            self.reset_epoch,
            tz=timezone.utc,
        ).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """
        转换为可以保存成 JSON 的字典。
        """
        return {
            "resource": self.resource,
            "limit": self.limit,
            "remaining": self.remaining,
            "used": self.used,
            "reset_epoch": self.reset_epoch,
            "reset_time_utc": self.reset_time_utc,
        }


class GitHubClient:
    """
    GitHub REST API 客户端。

    当前负责：

    1. 搜索仓库；
    2. 获取仓库详情；
    3. 获取 README；
    4. 获取近期 Release；
    5. 获取近期 Issue；
    6. 获取 API 限额状态。
    """

    def __init__(
        self,
        token: str,
        api_base_url: str = "https://api.github.com",
        api_version: str = "2022-11-28",
        timeout_seconds: int = 20,
        max_retries: int = 3,
        retry_base_seconds: float = 2.0,
    ) -> None:
        """
        初始化 GitHub API 客户端。
        """
        cleaned_token = token.strip()

        if not cleaned_token:
            raise ValueError(
                "GITHUB_TOKEN 为空。"
                "请先在当前终端中设置环境变量。"
            )

        self.api_base_url = api_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

        self.session = requests.Session()

        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "Authorization": (
                    f"Bearer {cleaned_token}"
                ),
                "X-GitHub-Api-Version": api_version,
                "User-Agent": (
                    "raglab-github-intelligence/0.2"
                ),
            }
        )

        self.rate_limits: dict[
            str,
            RateLimitSnapshot,
        ] = {}

    @staticmethod
    def _parse_optional_int(
        value: str | None,
    ) -> int | None:
        """
        将响应头字段安全转换为整数。
        """
        if value is None:
            return None

        try:
            return int(value)
        except ValueError:
            return None

    def _capture_rate_limit(
        self,
        response: requests.Response,
    ) -> None:
        """
        记录 GitHub 响应头中的 API 限额。
        """
        resource = response.headers.get(
            "x-ratelimit-resource"
        )

        if not resource:
            return

        self.rate_limits[
            resource
        ] = RateLimitSnapshot(
            resource=resource,
            limit=self._parse_optional_int(
                response.headers.get(
                    "x-ratelimit-limit"
                )
            ),
            remaining=self._parse_optional_int(
                response.headers.get(
                    "x-ratelimit-remaining"
                )
            ),
            used=self._parse_optional_int(
                response.headers.get(
                    "x-ratelimit-used"
                )
            ),
            reset_epoch=self._parse_optional_int(
                response.headers.get(
                    "x-ratelimit-reset"
                )
            ),
        )

    @staticmethod
    def _response_is_rate_limited(
        response: requests.Response,
    ) -> bool:
        """
        判断 403 或 429 是否由 API 限流造成。
        """
        if response.status_code == 429:
            return True

        if response.headers.get(
            "retry-after"
        ):
            return True

        if (
            response.headers.get(
                "x-ratelimit-remaining"
            )
            == "0"
        ):
            return True

        response_text = (
            response.text.lower()
        )

        phrases = (
            "rate limit",
            "secondary rate limit",
            "abuse detection",
        )

        return any(
            phrase in response_text
            for phrase in phrases
        )

    @staticmethod
    def _format_reset_time(
        response: requests.Response,
    ) -> str:
        """
        格式化 API 限额重置时间。
        """
        reset_value = response.headers.get(
            "x-ratelimit-reset"
        )

        if (
            not reset_value
            or not reset_value.isdigit()
        ):
            return "unknown"

        return datetime.fromtimestamp(
            int(reset_value),
            tz=timezone.utc,
        ).isoformat()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        """
        发送一个 GitHub API 请求。

        网络错误和服务器临时错误会按指数退避重试。
        """
        if path.startswith("http"):
            url = path
        else:
            url = (
                f"{self.api_base_url}{path}"
            )

        last_error: Exception | None = None

        for attempt in range(
            self.max_retries + 1
        ):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )

                self._capture_rate_limit(
                    response
                )

                if (
                    response.status_code
                    in (403, 429)
                    and self._response_is_rate_limited(
                        response
                    )
                ):
                    retry_after = (
                        response.headers.get(
                            "retry-after"
                        )
                    )

                    if (
                        retry_after
                        and attempt
                        < self.max_retries
                    ):
                        try:
                            wait_seconds = float(
                                retry_after
                            )
                        except ValueError:
                            wait_seconds = (
                                self.retry_base_seconds
                                * (2**attempt)
                            )

                        wait_seconds = min(
                            max(
                                wait_seconds,
                                1.0,
                            ),
                            60.0,
                        )

                        time.sleep(
                            wait_seconds
                        )

                        continue

                    resource = (
                        response.headers.get(
                            "x-ratelimit-resource",
                            "unknown",
                        )
                    )

                    remaining = (
                        response.headers.get(
                            "x-ratelimit-remaining",
                            "unknown",
                        )
                    )

                    reset_time = (
                        self._format_reset_time(
                            response
                        )
                    )

                    raise GitHubRateLimitError(
                        "GitHub API 达到限额。"
                        f"resource={resource}，"
                        f"remaining={remaining}，"
                        f"reset_utc={reset_time}，"
                        f"response="
                        f"{response.text[:500]}"
                    )

                if (
                    500
                    <= response.status_code
                    < 600
                    and attempt
                    < self.max_retries
                ):
                    wait_seconds = (
                        self.retry_base_seconds
                        * (2**attempt)
                    )

                    time.sleep(
                        wait_seconds
                    )

                    continue

                if not response.ok:
                    raise GitHubApiError(
                        "GitHub API 请求失败。"
                        f"status="
                        f"{response.status_code}，"
                        f"url={response.url}，"
                        f"response="
                        f"{response.text[:1000]}"
                    )

                return response

            except (
                requests.Timeout,
                requests.ConnectionError,
            ) as exc:
                last_error = exc

                if (
                    attempt
                    >= self.max_retries
                ):
                    break

                wait_seconds = (
                    self.retry_base_seconds
                    * (2**attempt)
                )

                time.sleep(
                    wait_seconds
                )

        raise GitHubApiError(
            "GitHub API 网络请求失败："
            f"{url}。"
            f"最后一次错误：{last_error}"
        )

    @staticmethod
    def _split_full_name(
        full_name: str,
    ) -> tuple[str, str]:
        """
        将 owner/repository 拆分为两个字段。
        """
        cleaned_full_name = (
            full_name.strip()
        )

        owner, separator, repository = (
            cleaned_full_name.partition("/")
        )

        if (
            not separator
            or not owner
            or not repository
        ):
            raise ValueError(
                "仓库名称格式错误："
                f"{full_name}。"
                "应使用 owner/repository。"
            )

        return (
            quote(owner, safe=""),
            quote(repository, safe=""),
        )

    def search_repositories(
        self,
        query: str,
        *,
        per_page: int = 15,
        sort: str = "stars",
        order: str = "desc",
    ) -> dict[str, Any]:
        """
        使用 GitHub Repository Search API 搜索仓库。
        """
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "仓库搜索条件不能为空。"
            )

        normalized_per_page = max(
            1,
            min(
                int(per_page),
                100,
            ),
        )

        response = self._request(
            "GET",
            "/search/repositories",
            params={
                "q": cleaned_query,
                "per_page": (
                    normalized_per_page
                ),
                "sort": sort,
                "order": order,
            },
        )

        data = response.json()

        if not isinstance(
            data,
            dict,
        ):
            raise GitHubApiError(
                "Repository Search "
                "没有返回有效的 JSON 对象。"
            )

        items = data.get(
            "items"
        )

        if (
            items is not None
            and not isinstance(
                items,
                list,
            )
        ):
            raise GitHubApiError(
                "Repository Search 的 "
                "items 字段格式错误。"
            )

        return data

    def get_repository(
        self,
        full_name: str,
    ) -> dict[str, Any]:
        """
        获取一个仓库的完整基础信息。
        """
        owner, repository = (
            self._split_full_name(
                full_name
            )
        )

        response = self._request(
            "GET",
            (
                f"/repos/{owner}/"
                f"{repository}"
            ),
        )

        data = response.json()

        if not isinstance(
            data,
            dict,
        ):
            raise GitHubApiError(
                "仓库详情没有返回有效的 "
                f"JSON 对象：{full_name}"
            )

        return data

    def get_repository_readme(
        self,
        full_name: str,
    ) -> dict[str, Any] | None:
        """
        获取并解码仓库默认分支中的 README。

        返回字段：

        name
        path
        sha
        size
        html_url
        download_url
        content
        encoding

        仓库没有 README 时返回 None。
        """
        owner, repository = (
            self._split_full_name(
                full_name
            )
        )

        try:
            response = self._request(
                "GET",
                (
                    f"/repos/{owner}/"
                    f"{repository}/readme"
                ),
            )
        except GitHubApiError as exc:
            # 仓库没有 README 时，
            # GitHub 通常返回 404。
            if "status=404" in str(exc):
                return None

            raise

        data = response.json()

        if not isinstance(
            data,
            dict,
        ):
            raise GitHubApiError(
                "README API 没有返回有效的 "
                f"JSON 对象：{full_name}"
            )

        encoded_content = data.get(
            "content"
        )

        encoding = str(
            data.get("encoding")
            or ""
        ).lower()

        decoded_content = ""

        if (
            isinstance(
                encoded_content,
                str,
            )
            and encoding == "base64"
        ):
            try:
                decoded_bytes = (
                    base64.b64decode(
                        encoded_content,
                        validate=False,
                    )
                )

                decoded_content = (
                    decoded_bytes.decode(
                        "utf-8",
                        errors="replace",
                    )
                )

            except (
                ValueError,
                TypeError,
            ) as exc:
                raise GitHubApiError(
                    "README Base64 解码失败："
                    f"{full_name}"
                ) from exc

        return {
            "name": data.get(
                "name"
            ),
            "path": data.get(
                "path"
            ),
            "sha": data.get(
                "sha"
            ),
            "size": data.get(
                "size"
            ),
            "html_url": data.get(
                "html_url"
            ),
            "download_url": data.get(
                "download_url"
            ),
            "encoding": encoding,
            "content": decoded_content,
        }

    def list_repository_releases(
        self,
        full_name: str,
        *,
        per_page: int = 3,
    ) -> list[dict[str, Any]]:
        """
        获取仓库最近发布的 Release。

        第一阶段默认最多获取 3 条，
        避免读取大量历史版本。
        """
        owner, repository = (
            self._split_full_name(
                full_name
            )
        )

        normalized_per_page = max(
            1,
            min(
                int(per_page),
                20,
            ),
        )

        response = self._request(
            "GET",
            (
                f"/repos/{owner}/"
                f"{repository}/releases"
            ),
            params={
                "per_page": (
                    normalized_per_page
                ),
                "page": 1,
            },
        )

        data = response.json()

        if not isinstance(
            data,
            list,
        ):
            raise GitHubApiError(
                "Release API 没有返回有效的列表："
                f"{full_name}"
            )

        releases: list[
            dict[str, Any]
        ] = []

        for item in data:
            if not isinstance(
                item,
                dict,
            ):
                continue

            releases.append(
                {
                    "id": item.get(
                        "id"
                    ),
                    "tag_name": item.get(
                        "tag_name"
                    ),
                    "name": item.get(
                        "name"
                    ),
                    "body": item.get(
                        "body"
                    ),
                    "draft": bool(
                        item.get(
                            "draft",
                            False,
                        )
                    ),
                    "prerelease": bool(
                        item.get(
                            "prerelease",
                            False,
                        )
                    ),
                    "created_at": item.get(
                        "created_at"
                    ),
                    "published_at": item.get(
                        "published_at"
                    ),
                    "html_url": item.get(
                        "html_url"
                    ),
                }
            )

        return releases

    def list_repository_issues(
        self,
        full_name: str,
        *,
        per_page: int = 5,
        state: str = "all",
        sort: str = "comments",
        direction: str = "desc",
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        获取仓库近期或高评论 Issue。

        GitHub Issues API 会混入 Pull Request，
        因此这里会过滤带有 pull_request 字段的项目。
        """
        owner, repository = (
            self._split_full_name(
                full_name
            )
        )

        normalized_per_page = max(
            1,
            min(
                int(per_page),
                30,
            ),
        )

        params: dict[str, Any] = {
            "state": state,
            "sort": sort,
            "direction": direction,
            "per_page": (
                normalized_per_page
                * 2
            ),
            "page": 1,
        }

        if since:
            params["since"] = since

        response = self._request(
            "GET",
            (
                f"/repos/{owner}/"
                f"{repository}/issues"
            ),
            params=params,
        )

        data = response.json()

        if not isinstance(
            data,
            list,
        ):
            raise GitHubApiError(
                "Issues API 没有返回有效的列表："
                f"{full_name}"
            )

        issues: list[
            dict[str, Any]
        ] = []

        for item in data:
            if not isinstance(
                item,
                dict,
            ):
                continue

            # GitHub 会将 PR 混入 Issues API。
            if "pull_request" in item:
                continue

            labels_data = item.get(
                "labels"
            )

            labels: list[str] = []

            if isinstance(
                labels_data,
                list,
            ):
                for label in labels_data:
                    if isinstance(
                        label,
                        dict,
                    ):
                        name = label.get(
                            "name"
                        )

                        if name:
                            labels.append(
                                str(name)
                            )

            reactions_data = item.get(
                "reactions"
            )

            total_reactions = 0

            if isinstance(
                reactions_data,
                dict,
            ):
                total_reactions = int(
                    reactions_data.get(
                        "total_count"
                    )
                    or 0
                )

            issues.append(
                {
                    "number": item.get(
                        "number"
                    ),
                    "title": item.get(
                        "title"
                    ),
                    "body": item.get(
                        "body"
                    ),
                    "state": item.get(
                        "state"
                    ),
                    "labels": labels,
                    "comments": int(
                        item.get(
                            "comments"
                        )
                        or 0
                    ),
                    "reactions": (
                        total_reactions
                    ),
                    "created_at": item.get(
                        "created_at"
                    ),
                    "updated_at": item.get(
                        "updated_at"
                    ),
                    "closed_at": item.get(
                        "closed_at"
                    ),
                    "html_url": item.get(
                        "html_url"
                    ),
                }
            )

            if (
                len(issues)
                >= normalized_per_page
            ):
                break

        return issues

    def get_rate_limit_status(
        self,
    ) -> dict[str, Any]:
        """
        获取 GitHub API 当前限额状态。
        """
        response = self._request(
            "GET",
            "/rate_limit",
        )

        data = response.json()

        if not isinstance(
            data,
            dict,
        ):
            raise GitHubApiError(
                "Rate Limit API 没有返回"
                "有效的 JSON 对象。"
            )

        return data

    def get_captured_rate_limits(
        self,
    ) -> dict[str, dict[str, Any]]:
        """
        返回客户端请求过程中记录的限额响应头。
        """
        return {
            resource: snapshot.to_dict()
            for resource, snapshot
            in self.rate_limits.items()
        }

    def close(self) -> None:
        """
        关闭底层 HTTP Session。
        """
        self.session.close()

    def __enter__(
        self,
    ) -> "GitHubClient":
        """
        支持使用 with 语句。
        """
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:
        """
        离开 with 语句时关闭客户端。
        """
        self.close()