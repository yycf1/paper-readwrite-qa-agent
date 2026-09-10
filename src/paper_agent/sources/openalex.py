"""OpenAlex client：主力学术索引（免 Key，mailto 礼貌池）。

API 文档：https://docs.openalex.org
- search 支持自然语言关键词；摘要以倒排索引返回，需重建；
- PDF 候选地址优先级：best_oa_location.pdf_url → open_access.oa_url → primary_location.pdf_url。
"""

from __future__ import annotations

import time

from paper_agent.models import Paper
from paper_agent.sources import SourceUnavailable
from paper_agent.sources._http import http_get

_API = "https://api.openalex.org/works"

_SELECT = (
    "id,doi,display_name,authorships,publication_year,cited_by_count,"
    "abstract_inverted_index,open_access,best_oa_location,primary_location"
)


def rebuild_abstract(inverted: dict[str, list[int]]) -> str:
    """OpenAlex 摘要是「词 → 位置列表」的倒排索引，按位置重建原文。"""
    if not inverted:
        return ""
    positions = [(i, word) for word, idxs in inverted.items() for i in idxs]
    positions.sort()
    return " ".join(word for _, word in positions)


def parse_work(work: dict) -> Paper:
    oa = work.get("open_access") or {}
    best = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}
    pdf_urls: list[str] = []
    for url in (best.get("pdf_url"), oa.get("oa_url"), primary.get("pdf_url")):
        if url and url not in pdf_urls:
            pdf_urls.append(url)
    authors = [
        a.get("author", {}).get("display_name", "") for a in (work.get("authorships") or [])
    ]
    raw_id = (work.get("id") or "").rsplit("/", 1)[-1]
    return Paper(
        source="openalex",
        source_id=f"openalex:{raw_id}",
        title=(work.get("display_name") or "").strip(),
        authors=[a for a in authors if a],
        abstract=rebuild_abstract(work.get("abstract_inverted_index") or {}),
        year=work.get("publication_year"),
        doi=(work.get("doi") or "").replace("https://doi.org/", "") or None,
        citations=work.get("cited_by_count"),
        pdf_urls=pdf_urls,
        abstract_only=not pdf_urls,
        landing_page=best.get("landing_page_url") or primary.get("landing_page_url") or None,
    )


def parse_response(data: dict) -> list[Paper]:
    return [p for p in (parse_work(w) for w in data.get("results", [])) if p.title]


def search(
    query: str,
    max_results: int = 10,
    year_from: int | None = None,
    *,
    proxy: str = "",
    email: str = "",
) -> list[Paper]:
    params: dict = {"search": query, "per-page": max_results, "select": _SELECT}
    filters = []
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if filters:
        params["filter"] = ",".join(filters)
    if email:
        params["mailto"] = email
    started = time.perf_counter()
    resp = http_get(_API, params=params, proxy=proxy, timeout=(10, 60))
    if resp.status_code != 200:
        raise SourceUnavailable(f"OpenAlex HTTP {resp.status_code}: {resp.text[:120]}")
    data = resp.json()
    papers = parse_response(data)
    _ = started  # 计时仅用于调试，probe 里另有计时
    return papers


def probe(proxy: str = "", email: str = "") -> str:
    """连通性探测（pa doctor 用）。成功返回描述，失败抛 SourceUnavailable。"""
    params = {"search": "electron", "per-page": 1, "select": "id"}
    if email:
        params["mailto"] = email
    started = time.perf_counter()
    resp = http_get(_API, params=params, proxy=proxy, timeout=(8, 20))
    cost = time.perf_counter() - started
    if resp.status_code != 200:
        raise SourceUnavailable(f"HTTP {resp.status_code}")
    return f"HTTP 200，检索通道正常（{cost:.1f}s）"
