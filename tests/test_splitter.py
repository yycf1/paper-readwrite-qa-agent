"""结构感知分块（rag/splitter.py）的纯逻辑测试。"""

from __future__ import annotations

from paper_agent.rag.splitter import (
    chunk_abstract,
    estimate_tokens,
    split_markdown,
)


def test_estimate_tokens_mixed() -> None:
    assert estimate_tokens("") == 0
    # 纯英文：4 字符 ≈ 1 token
    assert estimate_tokens("abcd" * 10) == 10
    # 中文按 1 字 1 token
    assert estimate_tokens("机器学习") == 4
    # 混合：8 个 ASCII（2 token）+ 4 个汉字（4 token）
    assert estimate_tokens("abcdefgh机器学习") == 6


def test_sections_and_pages_tracked() -> None:
    md = """<!-- p.1 -->
题名与作者区
<!-- p.2 -->

## Introduction

Graph neural networks are powerful. 首段在第二页。

## Experiments

<!-- p.3 -->

We evaluate on three datasets with strong baselines.
"""
    chunks = split_markdown(md, "arxiv:2401.12345", "Test Paper")
    sections = [c.section for c in chunks]
    assert "开篇" in sections and "Introduction" in sections and "Experiments" in sections
    intro = next(c for c in chunks if c.section == "Introduction")
    exp = next(c for c in chunks if c.section == "Experiments")
    assert intro.page == 2
    assert exp.page == 3
    assert all(c.paper_id == "arxiv:2401.12345" for c in chunks)
    assert all(c.metadata["chunk_index"] == i for i, c in enumerate(chunks))


def test_chunk_text_carries_section_header_and_strips_comments() -> None:
    md = """<!-- p.1 -->
<!-- 解析注释 -->

## Methods

The model uses attention. <!-- 内嵌注释 -->
"""
    chunks = split_markdown(md, "x:1", "T")
    assert len(chunks) == 1
    assert chunks[0].text.startswith("《T》 Methods")
    assert "<!--" not in chunks[0].text


def test_long_section_respects_max_tokens_with_overlap() -> None:
    para = ("This sentence repeats for testing purposes. " * 4).strip()  # ~11 tokens
    md = "## Results\n\n" + "\n\n".join(para for _ in range(120))
    chunks = split_markdown(md, "x:1", "T", target_tokens=100, max_tokens=160, overlap_tokens=20)
    assert len(chunks) >= 3
    body_tokens = [estimate_tokens(c.text) for c in chunks]
    assert max(body_tokens) <= 200  # 上限 + 章节头余量


def test_overlap_between_adjacent_chunks() -> None:
    sent = "相邻块重叠验证句子，用于检查上下文连续性。"
    md = "## 讨论\n\n" + "\n\n".join(sent for _ in range(60))
    chunks = split_markdown(md, "x:1", "T", target_tokens=80, max_tokens=120, overlap_tokens=30)
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:]):
        # 后块去掉章节头后的首段应来自前块结尾的重叠文本
        nxt_head = nxt.text.split("\n\n", 1)[1].split("\n\n", 1)[0]
        assert nxt_head in prev.text


def test_hard_split_of_single_huge_paragraph() -> None:
    para = "这是一个超长段落。" * 300  # 单段 1800 字
    chunks = split_markdown(f"## 一节\n\n{para}", "x:1", "T", max_tokens=400, overlap_tokens=50)
    assert len(chunks) >= 3
    assert all(len(c.text) > 0 for c in chunks)


def test_section_name_sanitized() -> None:
    """双栏 PDF 粘连产生的超长/多空格章节名要被压成可用的短名。"""
    md = "## 2  Related Workoverall preferences of users in different domains\n\n正文。\n"
    chunks = split_markdown(md, "x:1", "T")
    assert len(chunks) == 1
    assert len(chunks[0].section) <= 48
    assert "  " not in chunks[0].section


def test_max_tokens_includes_section_header() -> None:
    """章节头计入预算：最终块文本整体不超过 max_tokens。"""
    para = "This sentence repeats for testing purposes. " * 40
    md = "## A Fairly Long Section Name For Budget Checking\n\n" + (para + "\n\n") * 3
    chunks = split_markdown(md, "x:1", "Some Paper Title", max_tokens=160, overlap_tokens=20)
    assert all(estimate_tokens(c.text) <= 170 for c in chunks)


def test_tiny_markdown_yields_one_chunk() -> None:
    chunks = split_markdown("只有一句话。", "x:1", "T")
    assert len(chunks) == 1
    assert chunks[0].section == "开篇"


def test_chunk_abstract() -> None:
    assert chunk_abstract("x:1", "T", "  ") == []
    c = chunk_abstract("x:1", "T", "摘要内容。")[0]
    assert c.section == "摘要" and c.page == 0 and c.index == 0
