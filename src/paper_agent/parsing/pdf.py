"""PDF → 分节 Markdown（PyMuPDF 基线）。

策略：逐页取文本（sort=True 近似阅读顺序），行级启发式识别章节标题，
遇到 References/Bibliography 即截断（剥离参考文献）。双栏论文与公式的
精细还原不在基线范围内——效果不够时按 PLAN.md 引入 MinerU 升级。
"""

from __future__ import annotations

import re

import pymupdf

MAX_PAGES = 60

# 强信号：整行命中已知章节名（允许尾随句点/冒号）
_KW_HEADING_RE = re.compile(
    r"^\s*(abstract|summary|introduction|motivation|related works?|background|"
    r"preliminar(y|ies)|materials? and methods?|methods?|methodology|model|"
    r"experiments?|experimental setup|experimental results?|setup|"
    r"results?( and discussion)?|discussion|conclusions?|limitations?|"
    r"references|bibliography|acknowledg(e)?ments?|appendix|supplementary (material|information))\s*[.:：]?\s*$",
    re.IGNORECASE,
)
# 弱信号：编号标题（"2. Methods"、"3.1 Dataset"），限长限词数压低正文误报
_NUMBERED_RE = re.compile(r"^\s*(\d+(\.\d+)*\.?)\s+(\S.*)$")
_REFS_START_RE = re.compile(r"^\s*(references|bibliography)\s*[.:：]?\s*$", re.IGNORECASE)


def is_heading(line: str) -> bool:
    line = line.rstrip()
    if _KW_HEADING_RE.match(line):
        return True
    m = _NUMBERED_RE.match(line)
    if m:
        rest = m.group(3).strip()
        words = rest.split()
        # 编号标题通常 ≤6 词且不以句号结尾；正文首句 "1. We propose..." 淘汰。
        # 收紧到 6 词：双栏 PDF 里标题行常与正文首行粘连成一行（"2 Related Workoverall…"），
        # ≤10 词时这类粘连行会误判为标题，章节名被污染。
        return len(words) <= 6 and not rest.endswith((".", "。"))
    return False


def is_references_start(line: str) -> bool:
    return bool(_REFS_START_RE.match(line))


def extract_markdown(pdf_path, *, strip_references: bool = True) -> str:
    """PDF 转分节 Markdown：`## 章节` 结构 + 每页 `<!-- p.N -->` 锚点（供后续引用定位）。"""
    doc = pymupdf.open(pdf_path)
    try:
        out: list[str] = []
        refs_started = False
        for page_no, page in enumerate(doc, start=1):
            if page_no > MAX_PAGES:
                out.append(f"<!-- 第 {page_no} 页起超出解析上限，已截断 -->")
                break
            out.append(f"<!-- p.{page_no} -->")
            for raw in page.get_text("text", sort=True).splitlines():
                line = raw.rstrip()
                if not line.strip():
                    out.append("")
                    continue
                if strip_references and is_references_start(line):
                    refs_started = True
                    out.append(f"## {line.strip(' ..:：')}")
                    out.append("<!-- 参考文献已剥离 -->")
                    break
                if refs_started:
                    continue
                if is_heading(line):
                    out.append(f"## {line.strip()}")
                else:
                    out.append(line)
            if refs_started:
                break
            out.append("")
        return "\n".join(out).strip() + "\n"
    finally:
        doc.close()
