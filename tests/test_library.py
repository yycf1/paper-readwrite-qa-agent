"""library.db 账本测试：状态机、两级去重、元数据合并不改状态。"""

from pathlib import Path

from paper_agent.library import Library
from paper_agent.models import Paper


def make_paper(sid: str, title: str, **kw) -> Paper:
    defaults = dict(source="openalex", title=title)
    return Paper(source_id=sid, **{**defaults, **kw})


def make_lib(tmp_path: Path) -> Library:
    return Library(tmp_path / "test.db")


def test_upsert_new_and_status_flow(tmp_path):
    lib = make_lib(tmp_path)
    p = make_paper("openalex:W1", "Deep Learning")
    assert lib.upsert_paper(p) is True
    assert lib.get("openalex:W1").status == "discovered"

    lib.set_status("openalex:W1", "downloaded", pdf_path="data/papers/x/paper.pdf")
    rec = lib.get("openalex:W1")
    assert rec.status == "downloaded"
    assert rec.pdf_path.endswith("paper.pdf")

    # 重试成功清除 error
    lib.set_status("openalex:W1", "download_failed", error="HTTP 403")
    assert lib.get("openalex:W1").error == "HTTP 403"
    lib.set_status("openalex:W1", "downloaded", error=None, pdf_path="p.pdf")
    assert lib.get("openalex:W1").error is None


def test_find_duplicate_by_doi_and_fuzzy_title(tmp_path):
    lib = make_lib(tmp_path)
    lib.upsert_paper(make_paper("openalex:W1", "Attention Is All You Need!", doi="10.1/abc"))

    # DOI 命中（即使标题写法不同）
    dup = lib.find_duplicate(make_paper("arxiv:1", "完全不同的标题", doi="10.1/abc"))
    assert dup == "openalex:W1"

    # 模糊标题命中（大小写/标点变体，无 DOI）
    dup = lib.find_duplicate(make_paper("europepmc:PMC1", "attention   is all you need"))
    assert dup == "openalex:W1"

    # 无重复
    assert lib.find_duplicate(make_paper("arxiv:2", "Unrelated Work")) is None


def test_upsert_existing_merges_metadata_but_keeps_status(tmp_path):
    lib = make_lib(tmp_path)
    lib.upsert_paper(make_paper("openalex:W1", "Paper A", doi="10.1/a"))
    lib.set_status("openalex:W1", "downloaded", pdf_path="p.pdf")

    # 另一源补充了 pdf 链接与摘要（不同 source_id，标题相同 → 合并到新条目）
    # 同一 source_id 再次 upsert：补空字段但状态/路径保留
    lib.upsert_paper(
        make_paper("openalex:W1", "Paper A", doi="10.1/a", abstract="新增摘要",
                   pdf_urls=["http://x/1.pdf"])
    )
    rec = lib.get("openalex:W1")
    assert rec.status == "downloaded"
    assert rec.pdf_path == "p.pdf"
    assert rec.paper.abstract == "新增摘要"
    assert rec.paper.pdf_urls == ["http://x/1.pdf"]
    assert not rec.paper.abstract_only


def test_list_filter_and_counts(tmp_path):
    lib = make_lib(tmp_path)
    lib.upsert_paper(make_paper("openalex:W1", "A"))
    lib.upsert_paper(make_paper("openalex:W2", "B"))
    lib.upsert_paper(make_paper("openalex:W3", "C"))
    lib.set_status("openalex:W2", "downloaded")
    assert lib.counts() == {"discovered": 2, "downloaded": 1}
    assert len(lib.list(status="discovered")) == 2
    assert len(lib.list()) == 3


def test_find_fragment(tmp_path):
    lib = make_lib(tmp_path)
    lib.upsert_paper(make_paper("openalex:W2741809807", "A"))
    assert lib.find("W2741809807") == ["openalex:W2741809807"]
    assert lib.find("不存在") == []
