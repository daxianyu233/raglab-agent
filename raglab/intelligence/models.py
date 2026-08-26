from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TrendingRepository:
    """
    Repository information parsed from the GitHub Trending HTML page.

    This object records the repository's position and star increase
    during the selected Trending period.
    """

    full_name: str
    description: str
    language: str | None
    period_stars: int
    total_stars: int
    forks: int
    html_url: str
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass object into a serializable dictionary."""
        return asdict(self)


@dataclass(slots=True)
class SearchResult:
    """
    A repository result returned by one GitHub Search API query.

    query_name records which configured search rule found the repository.
    query records the complete GitHub query sent during this collection.
    """

    query_name: str
    query: str
    rank: int
    item: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass object into a serializable dictionary."""
        return asdict(self)


@dataclass(slots=True)
class NormalizedRepository:
    """
    Unified repository structure after combining Trending, Search,
    and repository-detail API results.
    """

    full_name: str
    owner: str
    name: str
    html_url: str

    description: str | None = None
    github_id: int | None = None
    language: str | None = None
    topics: list[str] = field(default_factory=list)
    license_spdx: str | None = None

    archived: bool = False
    is_fork: bool = False
    default_branch: str | None = None

    created_at: str | None = None
    updated_at: str | None = None
    pushed_at: str | None = None

    stars: int = 0
    forks: int = 0
    open_issues: int = 0
    subscribers: int = 0

    sources: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)

    trending_rank: int | None = None
    trending_period: str | None = None
    period_stars: int | None = None

    collected_at: str = ""
    snapshot_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass object into a serializable dictionary."""
        return asdict(self)
