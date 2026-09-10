"""Europe PMC client：生物医学文献（MEDLINE/PubMed 全量镜像 + OA 全文，免 Key）。

API 文档：https://europepmc.org/RestfulWebService
- resultType=core 返回摘要与全文链接；
- PDF 取 fullTextUrlList 中 documentStyle=pdf 的条目。
"""

from __future__ import annotations

import re
import time

from paper_agent.models import Paper
from paper_agent.sources import SourceUnavailable
from paper_agent.sources._http import http_get

_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

_TAG_RE = re.compile(r"<[^>]+>")


def parse_hit(hit: dict) -> Paper:
    title = _TAG_RE.sub("", hit.get("title") or "").strip()
    pmcid = hit.get("pmcid") or ""
    if pmcid:
        source_id = f"europepmc:{pmcid}"
    else:
        source_id = f"europepmc:{hit.get('source', 'MED')}:{hit.get('id', '')}"
    urls = (hit.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    pdf_urls = [u["url"] for u in urls if u.get("documentStyle") == "pdf" and u.get("url")]
    landing = next(
        (u["url"] for u in urls if u.get("documentStyle") == "html" and u.get("url")), None
    )
    pub_year = str(hit.get("pubYear") or "")
    authors = [a.strip() for a in (hit.get("authorString") or "").split(",") if a.strip()]
    return Paper(
        source="europepmc",
        source_id=source_id,
        title=title,
        authors=authors[:20],
        abstract=hit.get("abstractText") or "",
        year=int(pub_year) if pub_year.isdigit() else None,
        doi=hit.get("doi") or None,
        citations=hit.get("citedByCount"),
        pdf_urls=pdf_urls,
        abstract_only=not pdf_urls,
        landing_page=landing,
    )


def parse_response(data: dict) -> list[Paper]:
    hits = (data.get("resultList") or {}).get("result") or []
    return [p for p in (parse_hit(h) for h in hits) if p.title]


def search(
    query: str,
    max_results: int = 10,
    year_from: int | None = None,
    *,
    proxy: str = "",
) -> list[Paper]:
    # 清理 Europe PMC 查询语法的保留字符，防止用户输入改变语义
    safe_query = re.sub(r"""["^\[\]{}()~*?:\\]""", " ", query).strip()
    if year_from:
        safe_query = f"({safe_query}) AND (PUB_YEAR:[{year_from} TO 2099])"
    started = time.perf_counter()
    resp = http_get(
        _API,
        params={"query": safe_query, "format": "json", "pageSize": max_results,
                "resultType": "core"},
        proxy=proxy,
        timeout=(10, 60),
    )
    _ = started
    if resp.status_code != 200:
        raise SourceUnavailable(f"Europe PMC HTTP {resp.status_code}: {resp.text[:120]}")
    return parse_response(resp.json())


def probe(proxy: str = "") -> str:
    started = time.perf_counter()
    resp = http_get(
        _API,
        params={"query": "electron", "format": "json", "pageSize": 1},
        proxy=proxy,
        timeout=(8, 20),
    )
    cost = time.perf_counter() - started
    if resp.status_code != 200:
        raise SourceUnavailable(f"HTTP {resp.status_code}")
    return f"HTTP 200，检索通道正常（{cost:.1f}s）"
