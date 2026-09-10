"""arXiv client：搜索与 PDF 直下。

注意：
- 本机实测 export.arxiv.org 存在连接级屏蔽，需在 config.yaml 配 proxy 才可用；
  无代理时 http_get 抛 SourceUnavailable，上层降级跳过；
- 官方限速 1 请求/3 秒，模块内写死节流，不依赖调用方自觉。
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET

from paper_agent.models import Paper
from paper_agent.sources import SourceUnavailable
from paper_agent.sources._http import http_get

_API = "https://export.arxiv.org/api/query"
_MIN_INTERVAL = 3.0  # 官方限速：1 req/3s

_NS = {"a": "http://www.w3.org/2005/Atom"}
_WS_RE = re.compile(r"\s+")
_VERSION_RE = re.compile(r"v\d+$")

_last_request = 0.0


def _throttle() -> None:
    global _last_request
    wait = _MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def parse_atom(text: str) -> list[Paper]:
    """解析 Atom XML 为 Paper 列表（纯函数，离线可测）。"""
    root = ET.fromstring(text)
    papers: list[Paper] = []
    for entry in root.findall("a:entry", _NS):
        abs_url = (entry.findtext("a:id", default="", namespaces=_NS) or "").strip()
        rid = abs_url.rsplit("/", 1)[-1]
        bare = _VERSION_RE.sub("", rid)
        if not bare:
            continue
        title = _WS_RE.sub(" ", entry.findtext("a:title", default="", namespaces=_NS) or "").strip()
        summary = _WS_RE.sub(" ", entry.findtext("a:summary", default="", namespaces=_NS) or "").strip()
        authors = [
            (a.findtext("a:name", default="", namespaces=_NS) or "").strip()
            for a in entry.findall("a:author", _NS)
        ]
        published = (entry.findtext("a:published", default="", namespaces=_NS) or "").strip()
        year = int(published[:4]) if published[:4].isdigit() else None
        pdf_link = next(
            (
                link.get("href")
                for link in entry.findall("a:link", _NS)
                if link.get("type") == "application/pdf"
            ),
            None,
        )
        pdf_urls = [pdf_link] if pdf_link else [f"https://arxiv.org/pdf/{bare}"]
        papers.append(
            Paper(
                source="arxiv",
                source_id=f"arxiv:{bare}",
                title=title,
                authors=[a for a in authors if a],
                abstract=summary,
                year=year,
                pdf_urls=pdf_urls,
                landing_page=abs_url or None,
            )
        )
    return papers


def search(
    query: str,
    max_results: int = 10,
    year_from: int | None = None,
    *,
    proxy: str = "",
) -> list[Paper]:
    search_query = f'all:"{query}"' if " " in query else f"all:{query}"
    if year_from:
        search_query += f" AND submittedDate:[{year_from}01010000 TO 209912312359]"
    params = {"search_query": search_query, "max_results": max_results, "sortBy": "relevance"}
    _throttle()
    resp = http_get(_API, params=params, proxy=proxy, timeout=(10, 60))
    if resp.status_code != 200:
        raise SourceUnavailable(f"arXiv HTTP {resp.status_code}")
    return parse_atom(resp.text)


def probe(proxy: str = "") -> str:
    _throttle()
    started = time.perf_counter()
    resp = http_get(
        _API,
        params={"search_query": "all:electron", "max_results": 1},
        proxy=proxy,
        timeout=(8, 15),
    )
    cost = time.perf_counter() - started
    if resp.status_code != 200:
        raise SourceUnavailable(f"HTTP {resp.status_code}")
    return f"HTTP 200（{cost:.1f}s）"
