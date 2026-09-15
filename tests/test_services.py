"""服务层测试：search_and_ingest（检索编排）与 handle_message（对话助手），全离线。"""

from __future__ import annotations

from paper_agent.library import Library
from paper_agent.models import Paper
from paper_agent.services import assistant as asst
from paper_agent.services import search as svc
from paper_agent.tools import ToolResult


def _paper(sid: str, title: str, doi: str | None = None) -> Paper:
    return Paper(source=sid.split(":", 1)[0], source_id=sid, title=title, doi=doi)


def _cfg() -> dict:
    return {
        "proxy": "",
        "sources": {
            "openalex": {"enabled": True},
            "europepmc": {"enabled": True},
            "arxiv": {"enabled": False},
        },
    }


def _ok(data) -> ToolResult:
    return ToolResult(tool="search_papers", status="ok", data=data)


def test_search_and_ingest_fallback_to_raw_query(tmp_path, monkeypatch):
    """优化词全军覆没 → 回退原始词重搜；结果校验后入库。"""
    lib = Library(tmp_path / "lib.db")
    try:
        monkeypatch.setattr(
            svc, "optimize_query",
            lambda q: {"topics": [f"{q} survey"], "year_from": None, "max_results": None, "optimized": True},
        )
        calls: list[str] = []

        def fake_tool(name, *, query, max_results, year_from=None, proxy="", email=""):
            calls.append(query)
            if query.endswith("survey"):
                return _ok([])  # 优化词无结果
            return _ok([_paper("openalex:W1", "Paper One", doi="10.1/1")])

        monkeypatch.setattr(svc, "search_source_tool", fake_tool)
        outcome = svc.search_and_ingest("gnn", lib=lib, cfg=_cfg(), sources="openalex", max_results=5)
        assert calls == ["gnn survey", "gnn"]  # 触发了回退
        assert outcome.total_hits == 1
        assert [r.paper.source_id for r in outcome.new_papers] == ["openalex:W1"]
        assert any("回退" in n for n in outcome.notes)
    finally:
        lib.close()


def test_search_and_ingest_cross_command_dedup(tmp_path, monkeypatch):
    """第二次检索同样内容：候选计数仍在，但不重复入库。"""
    lib = Library(tmp_path / "lib.db")
    try:
        monkeypatch.setattr(
            svc, "optimize_query",
            lambda q: {"topics": [q], "year_from": None, "max_results": None, "optimized": False},
        )
        monkeypatch.setattr(
            svc, "search_source_tool",
            lambda *a, **k: _ok([_paper("openalex:W1", "Paper One", doi="10.1/1")]),
        )
        cfg = _cfg()
        first = svc.search_and_ingest("gnn", lib=lib, cfg=cfg, sources="openalex", max_results=5)
        second = svc.search_and_ingest("gnn", lib=lib, cfg=cfg, sources="openalex", max_results=5)
        assert len(first.new_papers) == 1
        assert second.total_hits == 1
        assert second.new_papers == []  # 幂等：不再新增
    finally:
        lib.close()


def test_search_and_ingest_unknown_source(tmp_path):
    lib = Library(tmp_path / "lib.db")
    try:
        try:
            svc.search_and_ingest("q", lib=lib, cfg=_cfg(), sources="openalex,foo")
            raise AssertionError("应当抛 ValueError")
        except ValueError as exc:
            assert "foo" in str(exc)
    finally:
        lib.close()


def test_search_and_ingest_degraded_source_noted(tmp_path, monkeypatch):
    """单源降级不阻塞其它源，降级原因进 notes。"""
    lib = Library(tmp_path / "lib.db")
    try:
        monkeypatch.setattr(
            svc, "optimize_query",
            lambda q: {"topics": [q], "year_from": None, "max_results": None, "optimized": False},
        )

        def fake_tool(name, *, query, max_results, year_from=None, proxy="", email=""):
            if name == "openalex":
                return ToolResult(tool="search_papers", status="degraded", error="连接被重置")
            return _ok([_paper("europepmc:P1", "Bio Paper")])

        monkeypatch.setattr(svc, "search_source_tool", fake_tool)
        outcome = svc.search_and_ingest("gnn", lib=lib, cfg=_cfg(), sources="openalex,europepmc", max_results=5)
        assert [r.paper.source_id for r in outcome.new_papers] == ["europepmc:P1"]
        assert outcome.source_hits.get("europepmc") == 1
        assert any("openalex" in n and "不可用" in n for n in outcome.notes)
    finally:
        lib.close()


# ---------------------------------------------------------------------------
# handle_message：意图分发
# ---------------------------------------------------------------------------


class _FakeStore:
    def count(self) -> int:
        return 1


class _FakeRetriever:
    store = _FakeStore()


def _route(intent: str, **extra) -> dict:
    return {"intent": intent, "argument": "", "via": "test", **extra}


def test_handle_message_status(tmp_path, monkeypatch):
    lib = Library(tmp_path / "lib.db")
    try:
        lib.upsert_paper(_paper("openalex:W1", "Paper One"))
        monkeypatch.setattr(asst, "classify_intent", lambda text, **k: _route("status"))
        reply = asst.handle_message("看看状态", lib=lib, retriever=_FakeRetriever(), params={})
        assert reply.intent == "status"
        assert "共 1 篇" in reply.reply and "discovered: 1" in reply.reply
    finally:
        lib.close()


def test_handle_message_search_returns_new_papers(tmp_path, monkeypatch):
    lib = Library(tmp_path / "lib.db")
    try:
        monkeypatch.setattr(
            asst, "classify_intent",
            lambda text, **k: _route("search", search={"topics": ["gnn survey"], "year_from": 2020, "max_results": 5}),
        )
        rec = lib.upsert_paper(_paper("openalex:W1", "Paper One"))
        from paper_agent.library import PaperRecord

        record = lib.get("openalex:W1")

        def fake_search(query, **kwargs):
            assert query == "gnn survey"
            assert kwargs["year_from"] == 2020
            return svc.SearchOutcome(new_papers=[record], total_hits=3)

        monkeypatch.setattr(asst, "search_and_ingest", fake_search)
        reply = asst.handle_message(
            "帮我找 gnn 综述", lib=lib, retriever=_FakeRetriever(),
            params={"max_per_source": 10, "year_from": None},
        )
        assert reply.intent == "search"
        assert "新入库 1 篇" in reply.reply
        assert "3 条候选" in reply.reply
        assert len(reply.new_papers) == 1
    finally:
        lib.close()


def test_handle_message_qa_empty_library(tmp_path, monkeypatch):
    """qa 意图 + 空向量库：提示建索引而不是报错。"""
    lib = Library(tmp_path / "lib.db")
    try:
        monkeypatch.setattr(asst, "classify_intent", lambda text, **k: _route("qa", argument="什么是 GNN？"))

        class _Empty:
            def count(self):
                return 0

        class _R:
            store = _Empty()

        reply = asst.handle_message("什么是 GNN？", lib=lib, retriever=_R(), params={})
        assert reply.intent == "qa"
        assert "向量库为空" in reply.reply
    finally:
        lib.close()


def test_handle_message_never_raises(tmp_path, monkeypatch):
    """检索环节抛任何异常都转成回复文本，对话不中断。"""
    lib = Library(tmp_path / "lib.db")
    try:
        monkeypatch.setattr(
            asst, "classify_intent",
            lambda text, **k: _route("search", search={"topics": ["boom"]}),
        )

        def boom(*a, **k):
            raise RuntimeError("网络炸了")

        monkeypatch.setattr(asst, "search_and_ingest", boom)
        reply = asst.handle_message("随便搜点", lib=lib, retriever=_FakeRetriever(), params={})
        assert reply.intent == "search"
        assert any("失败" in n for n in reply.notes)
    finally:
        lib.close()
