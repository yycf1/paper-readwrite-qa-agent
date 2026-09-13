"""流水线（graph/pipeline.py）测试：monkeypatch IO，走真实图，全离线。"""

from __future__ import annotations

import pytest

from paper_agent.graph import pipeline as pl
from paper_agent.library import Library
from paper_agent.models import Paper


def _paper(sid: str, title: str, *, citations=None, year=None, pdf_urls=(), doi=None):
    return Paper(
        source=sid.split(":", 1)[0],
        source_id=sid,
        title=title,
        citations=citations,
        year=year,
        pdf_urls=list(pdf_urls),
        doi=doi,
    )


def _cfg(tmp_path):
    return {
        "data_dir": str(tmp_path / "data"),
        "proxy": "",
        "sources": {"openalex": {"enabled": True}, "europepmc": {"enabled": True}, "arxiv": {"enabled": False}},
        "pipeline": {"top_n": 2, "year_from": 2020, "download_budget": 1},
    }


def _make_fake_store(state):
    """带跨阶段持久化的 fake 向量库：模拟 Chroma 的已有内容查询。"""

    class _FakeStore:
        def __init__(self, *_a, **_k):
            pass

        def paper_modes(self):
            return dict(state["modes"])

        def upsert_paper_chunks(self, paper_id, texts, metas, embeddings):
            self.remove_paper(paper_id)
            if texts:
                state["chunks"][paper_id] = list(texts)
                state["modes"][paper_id] = metas[0].get("mode", "")
            return len(texts)

        def remove_paper(self, paper_id):
            state["chunks"].pop(paper_id, None)
            state["modes"].pop(paper_id, None)

    return _FakeStore


@pytest.fixture()
def fake_io(monkeypatch, tmp_path):
    """统一替换源检索 / 下载 / 解析 / LLM / 向量库 / 配置。"""
    cfg = _cfg(tmp_path)
    store_state = {"modes": {}, "chunks": {}}
    papers = {
        "oa": [
            _paper("openalex:W1", "Paper One", citations=100, year=2022, pdf_urls=["http://x/1.pdf"], doi="10.1/1"),
            _paper("openalex:W2", "Paper Two", citations=5, year=2019, pdf_urls=["http://x/2.pdf"]),  # 年份被过滤
            _paper("openalex:W3", "Paper Three", citations=50, year=2023, pdf_urls=["http://x/3.pdf"]),
            _paper("openalex:W4", "Abstract Only", citations=30, year=2021),
        ],
        "epmc": [_paper("europepmc:W1dup", "Paper One", citations=1, year=2022, doi="10.1/1")],
    }

    class _Src:
        @staticmethod
        def search(query, max_n, year_from, proxy="", email=""):
            out = [p for p in papers["oa"] if year_from is None or (p.year or 0) >= year_from]
            return out[:max_n]

    class _Epmc:
        @staticmethod
        def search(query, max_n, year_from, proxy=""):
            return papers["epmc"][:max_n]

    monkeypatch.setattr(pl, "SOURCE_MODULES", {"openalex": _Src, "europepmc": _Epmc})

    def _fake_download(paper, papers_dir, *, proxy=""):
        dest = papers_dir / paper.source_id.replace(":", "_") / "paper.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-fake")
        return dest

    monkeypatch.setattr(pl, "download_pdf", _fake_download)
    monkeypatch.setattr(pl, "extract_markdown", lambda path: "段落内容。" * 60)  # ~300 token
    monkeypatch.setattr(pl, "chat", lambda messages, **k: ("ok", {"prompt": 1, "completion": 1}))
    monkeypatch.setattr(
        pl, "extract_experiment",
        lambda markdown, chat_fn=None: {"token_usage": {"prompt": 100, "completion": 50}},
    )
    monkeypatch.setattr(
        pl, "generate_report",
        lambda markdown, chat_fn=None: ("report", {"prompt": 10, "completion": 5}),
    )
    monkeypatch.setattr(pl, "embed", lambda texts: [[0.0] * 4 for _ in texts])
    monkeypatch.setattr(pl, "VectorStore", _make_fake_store(store_state))
    monkeypatch.setattr(pl, "load_config", lambda: cfg)
    return cfg, papers, store_state


def _invoke(query, params):
    return pl.run_pipeline(query, params)


def test_full_pipeline_end_to_end(fake_io, tmp_path):
    cfg, _papers, store_state = fake_io
    state = _invoke("test topic", {"max_per_source": 10})
    s = state["stats"]
    # 检索：oa 3 条（年份过滤掉 W2）+ epmc 1 条（与 W1 同 DOI 被去重）
    assert s["searched"] == 4 and s["new"] == 3
    # 选文 top_n=2：W1(100 引) > W3(50 引)；W4(30 引)落选
    assert state["selected"] == ["openalex:W1", "openalex:W3"]
    # 下载预算 1：只下载第一篇
    assert s["downloaded"] == 1 and s["download_failed"] == 0
    assert s["parsed"] == 1
    assert s["analyzed"] == 1 and s["analyze_tokens"] == 165
    assert s["indexed"] == 1 and s["index_chunks"] >= 1
    lib = Library(tmp_path / "data" / "library.db")
    try:
        assert lib.get("openalex:W1").status == "indexed"
        assert lib.get("openalex:W3").status == "discovered"  # 预算耗尽，留待下轮
        assert lib.get("openalex:W4").status == "discovered"  # 落选
    finally:
        lib.close()
    assert list(store_state["chunks"]) == ["openalex:W1"]


def test_pipeline_resume_skips_done(fake_io, tmp_path):
    fake_io
    _invoke("test topic", {"max_per_source": 10})
    state2 = _invoke("test topic", {"max_per_source": 10})
    s2 = state2["stats"]
    # 去重：不重复入库；上轮预算耗尽未下载的 W3 本轮继续（top2 = W3, W4）
    assert s2["new"] == 0
    assert state2["selected"] == ["openalex:W3", "openalex:W4"]
    assert s2["downloaded"] == 1 and s2["parsed"] == 1 and s2["analyzed"] == 1
    # W1 已 indexed（fake 库持久化），只处理 W3
    assert s2["indexed"] == 1
    assert s2["downloaded"] == 1 and s2["parsed"] == 1 and s2["analyzed"] == 1


def test_download_failure_marks_state(fake_io, monkeypatch, tmp_path):
    cfg, _papers, _store = fake_io

    def _boom(paper, papers_dir, *, proxy=""):
        raise pl.DownloadError("HTTP 403")

    monkeypatch.setattr(pl, "download_pdf", _boom)
    params = {"max_per_source": 10, "top_n": 1, "year_from": 2020, "download_budget": 1}
    state = _invoke("test topic", params)
    assert state["stats"]["download_failed"] == 1
    assert state["stats"]["parsed"] == 0  # 失败不阻塞后续阶段，但也没有可解析的
    lib = Library(tmp_path / "data" / "library.db")
    try:
        assert lib.get("openalex:W1").status == "download_failed"
    finally:
        lib.close()


def test_parse_empty_result_guard(fake_io, monkeypatch, tmp_path):
    fake_io
    monkeypatch.setattr(pl, "extract_markdown", lambda path: "太短")
    params = {"max_per_source": 10, "top_n": 1, "year_from": 2020, "download_budget": 1}
    state = _invoke("test topic", params)
    assert state["stats"]["parse_failed"] == 1
    assert state["stats"]["analyzed"] == 0


def test_analyze_skipped_without_llm_key(fake_io, monkeypatch, tmp_path):
    fake_io

    def _no_key(messages, **k):
        raise RuntimeError("未配置 LLM API Key")

    monkeypatch.setattr(pl, "chat", _no_key)
    params = {"max_per_source": 10, "top_n": 1, "year_from": 2020, "download_budget": 1}
    state = _invoke("test topic", params)
    assert state["stats"]["analyzed"] == 0
    assert any("知识提取跳过" in n for n in state["notes"])
    # 解析/下载不受影响
    assert state["stats"]["parsed"] == 1
    # 盲区修复：analyze 被跳过的论文入索引后不得误标 indexed（否则 pa analyze 永远扫不到）
    lib = Library(tmp_path / "data" / "library.db")
    try:
        assert lib.get("openalex:W1").status == "parsed"
    finally:
        lib.close()


def test_index_keeps_failed_analyze_retriable(fake_io, monkeypatch, tmp_path):
    """分析失败的论文：全文仍入索引（RAG 可检索），但状态保持 parsed 且 error 保留。"""
    fake_io

    def _bad_extract(markdown, chat_fn=None):
        raise ValueError("模型返回坏 JSON，重问后仍失败")

    monkeypatch.setattr(pl, "extract_experiment", _bad_extract)
    params = {"max_per_source": 10, "top_n": 1, "year_from": 2020, "download_budget": 1}
    state = _invoke("test topic", params)
    assert state["stats"]["analyze_failed"] == 1
    assert state["stats"]["indexed"] == 1  # 全文照样入索引
    lib = Library(tmp_path / "data" / "library.db")
    try:
        rec = lib.get("openalex:W1")
        assert rec.status == "parsed"  # 不再误标 indexed
        assert rec.error and "分析失败" in rec.error  # 错误痕迹保留
        # pa analyze 按 parsed 状态扫描仍能捞到它重试
        assert [r.paper.source_id for r in lib.list(status="parsed")] == ["openalex:W1"]
    finally:
        lib.close()


def test_index_advances_already_indexed_analyzed(fake_io, tmp_path):
    """先索引后分析的时序：向量库已有全文的 analyzed 论文，run_index 补推进到 indexed。"""
    cfg, _papers, store_state = fake_io
    lib = Library(tmp_path / "data" / "library.db")
    try:
        lib.upsert_paper(_paper("openalex:W1", "Paper One", citations=100, year=2022))
        lib.set_status("openalex:W1", "parsed")
        store_state["modes"]["openalex:W1"] = "fulltext"  # 之前 pa index 已入库
        lib.set_status("openalex:W1", "analyzed")  # 用户随后单独跑了 pa analyze
        counts, _notes = pl.run_index(lib, cfg, selected=[])
        assert counts["indexed"] == 0  # 不重复索引
        assert lib.get("openalex:W1").status == "indexed"
    finally:
        lib.close()


def test_download_backfills_budget_from_remaining_discovered(fake_io, tmp_path):
    """选中论文全是仅摘要时，预算结余应回填给其余有 OA 链接的 discovered 论文。"""
    cfg, _papers, store_state = fake_io
    lib = Library(tmp_path / "data" / "library.db")
    try:
        # 检索会带入 W1(100引)/W3(50引)/W4(30引)；W9 引用最高必被选中，但无 OA 链接
        lib.upsert_paper(_paper("openalex:W9", "Only Abstract Selected", citations=999, year=2023))
        params = {"max_per_source": 1, "top_n": 1, "year_from": 2020, "download_budget": 1}
        state = _invoke("topic", params)
        assert state["stats"]["abstract_only"] == 1  # 选中项无 OA
        assert state["stats"]["downloaded"] == 1  # 结余回填 1 篇
        # 回填取剩余中引用最高的 W1，并被流水线一路处理到 indexed
        assert lib.get("openalex:W1").status == "indexed"
        assert any("回填" in n for n in state["notes"])
    finally:
        lib.close()


def test_select_papers_orders_by_citations(tmp_path):
    lib = Library(tmp_path / "lib.db")
    try:
        for sid, cites, year in [("a:A", 10, 2021), ("a:B", 30, 2020), ("a:C", 30, 2024), ("a:D", None, 2024)]:
            lib.upsert_paper(_paper(sid, sid, citations=cites, year=year))
        params = {"top_n": 3, "year_from": None}
        # 引用数优先，年份仅作并列 tiebreak；无引用按 0 计
        selected, reason = pl.select_papers(lib, params)
        assert selected == ["a:C", "a:B", "a:A"]
        assert "按引用数" in reason  # 无 query 时不做预筛，直接按引用数选
    finally:
        lib.close()


def test_select_papers_relevance_filter(tmp_path):
    """LLM 预筛：不相关论文被过滤，理由可解释；打不上分的候选保守保留。"""
    lib = Library(tmp_path / "lib.db")
    try:
        papers = [
            ("openalex:W1", "Deep learning for GNN recommendation", 100),
            ("openalex:W2", "Cooking recipes with neural sparks", 999),  # 引用高但不相关
            ("openalex:W3", "Graph neural networks for recsys survey", 80),
            ("openalex:W4", "Another unrelated paper about birds", 60),
        ]
        for sid, title, cites in papers:
            lib.upsert_paper(_paper(sid, title, citations=cites, year=2023))
        params = {"top_n": 2, "year_from": None}

        def fake_chat(messages, **kw):
            reply = (
                '{"scores": ['
                '{"id": "openalex:W1", "score": 9, "reason": "GNN推荐方法"}, '
                '{"id": "openalex:W2", "score": 1, "reason": "烹饪无关"}, '
                '{"id": "openalex:W3", "score": 8, "reason": "GNN推荐综述"}, '
                '{"id": "openalex:W4", "score": 2, "reason": "鸟类无关"}]}'
            )
            return reply, {"prompt": 1, "completion": 1}

        selected, reason = pl.select_papers(
            lib, params, query="graph neural network recommendation", chat_fn=fake_chat,
        )
        assert selected == ["openalex:W1", "openalex:W3"]  # 高引但不相关的 W2 被预筛掉
        assert "4→2" in reason and "GNN" in reason

    finally:
        lib.close()


def test_select_papers_llm_failure_falls_back(tmp_path):
    """LLM 失败 → 退化为纯引用数排序，理由注明预筛不可用。"""
    lib = Library(tmp_path / "lib.db")
    try:
        for i in range(5):
            lib.upsert_paper(_paper(f"openalex:W{i}", f"T{i}", citations=100 - i, year=2023))
        params = {"top_n": 2, "year_from": None}

        def broken(messages, **kw):
            raise RuntimeError("no key")

        selected, reason = pl.select_papers(
            lib, params, query="any topic", chat_fn=broken,
        )
        assert selected == ["openalex:W0", "openalex:W1"]
        assert "预筛不可用" in reason

    finally:
        lib.close()
