"""结构感知分块：分节 Markdown → 带 paper_id/章节/页码元数据的 RAG 块。

按 `## 章节` 切分，章节内按段落聚合到目标 token 数（带重叠）；页码取自
解析器写入的 `<!-- p.N -->` 锚点。纯逻辑模块，无 IO，便于离线测试。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TARGET_TOKENS = 600
MAX_TOKENS = 800
OVERLAP_TOKENS = 100

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")
_PAGE_RE = re.compile(r"^<!--\s*p\.(\d+)\s*-->")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_CJK_RANGES = (
    (0x3000, 0x303F),  # CJK 标点
    (0x3040, 0x30FF),  # 假名
    (0x4E00, 0x9FFF),  # 汉字
    (0xFF00, 0xFFEF),  # 全角字符
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。．.!?！？])\s*")


@dataclass
class Chunk:
    """一个可索引的知识块；index 为该论文内的序号。"""

    paper_id: str
    title: str
    section: str
    text: str
    index: int
    page: int = 0  # 0 表示页码未知（如摘要块）

    @property
    def metadata(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "section": self.section,
            "page": self.page,
            "chunk_index": self.index,
        }


def estimate_tokens(text: str) -> int:
    """粗估 token 数：CJK 字符按 1 计，其余按 4 字符 1 token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES))
    return cjk + -(-(len(text) - cjk) // 4)  # ceil


def chunk_abstract(paper_id: str, title: str, abstract: str) -> list[Chunk]:
    """仅有摘要的论文：摘要整体作为单个块入知识库（PLAN §10 降级策略）。"""
    text = abstract.strip()
    if not text:
        return []
    return [Chunk(paper_id=paper_id, title=title, section="摘要", text=text, index=0)]


def split_markdown(
    markdown: str,
    paper_id: str,
    title: str,
    *,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Chunk]:
    """分节 Markdown → 知识块列表。章节名/页码随解析文本走，块文本带章节头。

    token 预算包含章节头（块文本整体入库，上限约束的是最终文本）。
    """
    chunks: list[Chunk] = []
    for name, page, body in _parse_sections(markdown):
        header = f"《{title}》 {name}\n\n"
        budget = max(32, max_tokens - estimate_tokens(header))
        pieces = _pack_section(body, target_tokens, budget, overlap_tokens)
        for piece in pieces:
            chunks.append(
                Chunk(
                    paper_id=paper_id,
                    title=title,
                    section=name,
                    text=header + piece,
                    index=len(chunks),
                    page=page,
                )
            )
    return chunks


# ---- 内部 ----


def _clean_section_name(raw: str) -> str:
    """章节名清理：PDF 双栏粘连会把正文拼进标题行，压空格并限长兜底。"""
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw[:48].strip()


def _parse_sections(markdown: str) -> list[tuple[str, int, str]]:
    """切成 (章节名, 起始页, 正文) 列表；首个 `##` 之前的内容归入「开篇」。"""
    sections: list[tuple[str, int, str]] = []
    name, page, buf = "开篇", 0, []
    for line in markdown.splitlines():
        m = _PAGE_RE.match(line)
        if m:
            page = int(m.group(1))
            continue
        h = _SECTION_RE.match(line)
        if h:
            if "".join(buf).strip():
                sections.append((name, page, "".join(buf)))
            name, buf = _clean_section_name(h.group(1)), []
            continue
        buf.append(line + "\n")
    if "".join(buf).strip():
        sections.append((name, page, "".join(buf)))
    return [(n, p, _COMMENT_RE.sub("", b).strip()) for n, p, b in sections if b.strip()]


def _pack_section(
    body: str, target_tokens: int, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """章节正文 → 有限长度的文本块：段落聚合 + 超长段落硬切 + 相邻块重叠。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for para in paragraphs:
        for piece in _split_long_paragraph(para, max_tokens, overlap_tokens):
            piece_tokens = estimate_tokens(piece)
            if cur and cur_tokens + piece_tokens > max_tokens:
                chunks.append("\n\n".join(cur))
                cur, cur_tokens = _carry_overlap(cur, overlap_tokens), 0
                cur_tokens = estimate_tokens(cur[0]) if cur else 0
                if cur_tokens + piece_tokens > max_tokens:
                    # 重叠尾过大（缺句读的长段）时放弃重叠，保证上限约束
                    cur, cur_tokens = [], 0
            if piece_tokens > max_tokens:  # 理论上 _split_long_paragraph 已保证，防御性兜底
                piece = piece[: max_tokens * 4]
                piece_tokens = estimate_tokens(piece)
            cur.append(piece)
            cur_tokens += piece_tokens
    if cur:
        chunks.append("\n\n".join(cur))
    return _merge_short_chunks(chunks, target_tokens, max_tokens=max_tokens)


def _split_long_paragraph(para: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """超长段落按句子硬切（带重叠）；正常段落原样返回。"""
    if estimate_tokens(para) <= max_tokens:
        return [para]
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(para) if s.strip()]
    pieces: list[str] = []
    cur = ""
    cur_tokens = 0
    for sent in sentences:
        sent_tokens = estimate_tokens(sent)
        if cur and cur_tokens + sent_tokens > max_tokens:
            pieces.append(cur.strip())
            cur, cur_tokens = _tail_text(cur, overlap_tokens), 0
            cur_tokens = estimate_tokens(cur)
        cur += ("" if cur.endswith((" ", "\n")) else " ") + sent
        cur_tokens += sent_tokens
    if cur.strip():
        pieces.append(cur.strip())
    return pieces


def _carry_overlap(chunk_texts: list[str], overlap_tokens: int) -> list[str]:
    """上一块结尾的少量文本作为下一块的开头（保持块间上下文连续）。"""
    tail = _tail_text(chunk_texts[-1], overlap_tokens)
    return [tail] if tail else []


def _tail_text(text: str, overlap_tokens: int) -> str:
    """取文本结尾约 overlap_tokens 个 token，尽量在句子/段落边界截断。"""
    if overlap_tokens <= 0 or not text:
        return ""
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    tail: list[str] = []
    total = 0
    for sent in reversed(sentences):
        n = estimate_tokens(sent)
        if tail and total + n > overlap_tokens:
            break
        tail.append(sent)
        total += n
    return "".join(reversed(tail)).strip()


def _merge_short_chunks(chunks: list[str], target_tokens: int, *, max_tokens: int) -> list[str]:
    """把明显小于目标的尾块并回前块，避免碎块稀释检索；合并不得突破上限。"""
    merged: list[str] = []
    for text in chunks:
        if (
            merged
            and estimate_tokens(merged[-1]) < target_tokens // 3
            and estimate_tokens(merged[-1]) + estimate_tokens(text) <= max_tokens
        ):
            merged[-1] = merged[-1] + "\n\n" + text
        else:
            merged.append(text)
    return merged
