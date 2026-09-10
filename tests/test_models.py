"""models 测试：标题归一化与跨源合并。"""

from paper_agent.models import Paper, merge_papers, normalize_title


def test_normalize_title_strips_case_and_punct():
    assert normalize_title("Attention Is All You Need!") == "attention is all you need"
    assert (
        normalize_title("Attention  IS  All You Need（第二次投稿）")
        == normalize_title("attention is all you need 第二次投稿")
    )


def _paper(source: str, sid: str, **kw) -> Paper:
    defaults = dict(source=source, source_id=sid, title="Same Paper")
    return Paper(**{**defaults, **kw})


def test_merge_papers_fills_blanks_and_unions_urls():
    a = _paper("openalex", "openalex:W1", title="A", doi=None, pdf_urls=["http://x/a.pdf"])
    b = _paper("arxiv", "arxiv:1", title="A", doi="10.1/x", pdf_urls=["http://x/a.pdf", "http://x/b.pdf"])
    m = merge_papers(a, b)
    assert m.doi == "10.1/x"
    assert m.pdf_urls == ["http://x/a.pdf", "http://x/b.pdf"]
    assert not m.abstract_only

    # 反向合并：以 arxiv 条目为底，补 openalex 的 pdf 链接
    m2 = merge_papers(b, a)
    assert m2.pdf_urls == ["http://x/a.pdf", "http://x/b.pdf"]
