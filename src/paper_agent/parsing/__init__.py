"""文档解析（PDF/DOCX → 分节 Markdown）。"""

from __future__ import annotations

from pathlib import Path

from paper_agent.parsing import docx as _docx
from paper_agent.parsing import pdf as _pdf

SUPPORTED_SUFFIXES = {".pdf", ".docx"}


def extract_markdown(path: str | Path, *, strip_references: bool = True) -> str:
    """按扩展名分发到对应解析器；不支持的格式抛 ValueError。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf.extract_markdown(path, strip_references=strip_references)
    if suffix == ".docx":
        return _docx.extract_markdown(path)
    raise ValueError(f"不支持的文件格式：{suffix}（支持 {sorted(SUPPORTED_SUFFIXES)}）")
