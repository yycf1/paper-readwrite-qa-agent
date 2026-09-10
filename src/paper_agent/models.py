"""数据模型：Paper 与标题归一化/跨源合并。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 生命周期主链路：discovered → downloaded → parsed → analyzed → indexed
STATUSES = ["discovered", "downloaded", "parsed", "analyzed", "indexed"]
# 失败标记（不在主链路上，可通过重试回到 discovered）
FAILED_STATUSES = ["download_failed"]

_PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)


@dataclass
class Paper:
    """一篇论文的跨源统一元数据。source_id 为全局唯一键（如 openalex:W2741809807）。"""

    source: str  # openalex / europepmc / arxiv
    source_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    doi: str | None = None
    citations: int | None = None
    # 候选 PDF 下载地址，按优先级排序；下载器逐个尝试并以 %PDF 魔数校验
    pdf_urls: list[str] = field(default_factory=list)
    landing_page: str | None = None
    abstract_only: bool = False  # True：无 OA 全文，仅摘要可入知识库


def normalize_title(title: str) -> str:
    """标题归一化（跨源去重用）：小写、去标点、压缩空白。"""
    return _PUNCT_RE.sub(" ", title.lower()).strip()


def merge_papers(a: Paper, b: Paper) -> Paper:
    """合并两源的同篇论文：以 a 为底，用 b 补空缺字段，pdf_urls 取并集（保持顺序）。"""
    urls = list(a.pdf_urls) + [u for u in b.pdf_urls if u not in a.pdf_urls]
    return Paper(
        source=a.source,
        source_id=a.source_id,
        title=a.title or b.title,
        authors=a.authors or b.authors,
        abstract=a.abstract or b.abstract,
        year=a.year or b.year,
        doi=a.doi or b.doi,
        citations=a.citations if a.citations is not None else b.citations,
        pdf_urls=urls,
        landing_page=a.landing_page or b.landing_page,
        abstract_only=not urls,
    )
