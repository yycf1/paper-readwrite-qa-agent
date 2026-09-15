"""Web API 测试：TestClient 全离线（monkeypatch 外部 IO 与 LLM）。"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from paper_agent.library import Library
from paper_agent.models import Paper
from paper_agent.server.app import create_app
from paper_agent.server import routes as rt


def _paper(sid: str, title: str, *, doi=None) -> Paper:
    return Paper(
        source=sid.split(":", 1)[0], source_id=sid, title=title,
        doi=doi, pdf_urls=["http://x/a.pdf"], abstract="摘要内容",
    )


class _FakeStore:
    def __init__(self, path):
        pass

    def count(self):
        return 7


def _seed(tmp_path) -> Library:
    lib = Library(tmp_path / "data" / "library.db")
    lib.upsert_paper(_paper("openalex:W1", "Graph Neural Survey", doi="10.1/1"))
    lib.upsert_paper(_paper("openalex:W2", "Cooking With Neural Sparks"))
    lib.set_status("openalex:W1", "parsed")
    return lib


def _client(monkeypatch, tmp_path) -> TestClient:
    cfg = {"data_dir": str(tmp_path / "data"), "proxy": "", "sources": {}}
    monkeypatch.setattr(rt, "load_config", lambda: cfg)
    monkeypatch.setattr(rt, "VectorStore", _FakeStore)
    return TestClient(create_app())


def test_status_endpoint(monkeypatch, tmp_path):
    _seed(tmp_path)
    client = _client(monkeypatch, tmp_path)
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert data["counts"] == {"parsed": 1, "discovered": 1}
    assert data["vector_chunks"] == 7
    assert client.get("/api/health").json()["ok"] is True


def test_papers_list_filter(monkeypatch, tmp_path):
    _seed(tmp_path)
    client = _client(monkeypatch, tmp_path)
    r = client.get("/api/papers", params={"status": "parsed"})
    assert [p["source_id"] for p in r.json()["papers"]] == ["openalex:W1"]
    r = client.get("/api/papers", params={"q": "cooking"})
    assert [p["source_id"] for p in r.json()["papers"]] == ["openalex:W2"]
    assert r.json()["papers"][0]["abstract"] == ""  # 列表不带摘要内容（省流量）


def test_paper_detail_404_and_full(monkeypatch, tmp_path):
    lib = _seed(tmp_path)
    lib.close()
    client = _client(monkeypatch, tmp_path)
    assert client.get("/api/papers/openalex:NOPE").status_code == 404
    r = client.get("/api/papers/openalex:W1")
    assert r.status_code == 200
    data = r.json()
    assert data["title"] == "Graph Neural Survey"
    assert data["abstract"] == "摘要内容"
    assert data["experiment"] is None  # 尚未分析


def test_search_preview_and_ingest(monkeypatch, tmp_path):
    """检索预览不写库；选中入库生效，重复候选带标记。"""
    lib = _seed(tmp_path)
    client = _client(monkeypatch, tmp_path)

    from paper_agent.services.search import Candidate, PreviewResult

    dup_paper = _paper("openalex:W1", "Graph Neural Survey", doi="10.1/1")  # 库内已有
    fresh = _paper("openalex:W9", "Fresh GNN Paper")

    def fake_sources(query, *, lib, cfg, **kwargs):
        assert query == "gnn"
        return PreviewResult(
            candidates=[Candidate(paper=dup_paper, duplicate_of="openalex:W1"),
                        Candidate(paper=fresh, duplicate_of=None)],
            total_hits=2,
            notes=["note1"],
            plan={"topics": ["gnn"], "optimized": True},
            source_hits={"openalex": 2},
        )

    monkeypatch.setattr(rt, "search_sources", fake_sources)
    r = client.post("/api/search", json={"query": "gnn"})
    assert r.status_code == 200
    data = r.json()
    assert data["total_hits"] == 2 and data["optimized"] is True
    assert data["candidates"][0]["duplicate_of"] == "openalex:W1"
    assert data["candidates"][1]["duplicate_of"] is None
    # 预览不写库：库里没有 W9
    assert lib.get("openalex:W9") is None

    # 入库所选（含一个重复项：幂等合并）
    r2 = client.post("/api/papers/ingest", json={
        "tag": "gnn 调研",
        "papers": [
            {"source": "openalex", "source_id": "openalex:W9", "title": "Fresh GNN Paper"},
            {"source": "openalex", "source_id": "openalex:W1", "title": "Graph Neural Survey", "doi": "10.1/1"},
        ],
    })
    assert r2.status_code == 200
    out = r2.json()
    assert out["ingested"] == 1 and out["duplicates"] == 1
    rec = lib.get("openalex:W9")
    assert rec is not None and rec.tag == "gnn 调研"  # 入库时打分组标签
    # 非法源名 → 400
    def bad_sources(query, **k):
        raise ValueError("未知数据源：foo")

    monkeypatch.setattr(rt, "search_sources", bad_sources)
    assert client.post("/api/search", json={"query": "x"}).status_code == 400
    lib.close()


def test_tags_facet(monkeypatch, tmp_path):
    lib = _seed(tmp_path)
    lib.upsert_paper(_paper("local:U1", "Uploaded Doc"), tag="本地上传")
    lib.close()
    client = _client(monkeypatch, tmp_path)
    data = client.get("/api/tags").json()
    assert data["tags"] == {"本地上传": 1}
    assert data["sources"]["openalex"] == 2
    # tag 过滤
    r = client.get("/api/papers", params={"tag": "本地上传"})
    assert [p["source_id"] for p in r.json()["papers"]] == ["local:U1"]


def test_delete_paper(monkeypatch, tmp_path):
    _seed(tmp_path)
    client = _client(monkeypatch, tmp_path)

    def fake_delete(sid, *, cfg, lib):
        return lib.delete(sid)

    monkeypatch.setattr(rt, "delete_paper", fake_delete)
    assert client.delete("/api/papers/openalex:W2").json()["ok"] is True
    lib = Library(tmp_path / "data" / "library.db")
    try:
        assert lib.get("openalex:W2") is None
        assert lib.get("openalex:W1") is not None  # 其他论文不受影响
    finally:
        lib.close()
    assert client.delete("/api/papers/openalex:NOPE").status_code == 404


def test_upload_paper(monkeypatch, tmp_path):
    lib = _seed(tmp_path)
    lib.close()
    client = _client(monkeypatch, tmp_path)

    def fake_import(path, *, cfg, lib, tag="本地上传"):
        from paper_agent.models import Paper

        paper = Paper(source="local", source_id="local:test-aabbccdd", title="My Local Paper")
        lib.upsert_paper(paper, tag=tag)
        lib.set_status("local:test-aabbccdd", "downloaded", pdf_path=str(path))
        return lib.get("local:test-aabbccdd"), True

    monkeypatch.setattr(rt, "import_local_file", fake_import)
    r = client.post("/api/papers/upload", files={"file": ("my paper.pdf", b"%PDF-fake", "application/pdf")})
    assert r.status_code == 201
    data = r.json()
    assert data["is_new"] is True
    assert data["paper"]["source_id"] == "local:test-aabbccdd"
    assert data["paper"]["status"] == "downloaded"


def test_ask_endpoint(monkeypatch, tmp_path):
    _seed(tmp_path).close()
    client = _client(monkeypatch, tmp_path)

    class _Hit:
        def __init__(self, pid, section):
            self.paper_id, self.section, self.title, self.page = pid, section, "T", 3
            self.text, self.distance = "chunk", 0.1

    def fake_answer(question, *, retriever, k=8, paper_id=None, **kw):
        return "答案 [1]", [_Hit("openalex:W1", "methods"), _Hit("openalex:W1", "methods")]

    monkeypatch.setattr(rt, "answer_question", fake_answer)
    r = client.post("/api/ask", json={"question": "用了什么方法？"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"].startswith("答案")
    assert len(data["sources"]) == 1  # 同论文同章节收敛
    assert data["sources"][0]["paper_id"] == "openalex:W1"


def test_chat_endpoint(monkeypatch, tmp_path):
    lib = _seed(tmp_path)
    lib.close()
    client = _client(monkeypatch, tmp_path)

    class _Reply:
        intent, reply, understanding = "status", "库里有 2 篇", "库状态（LLM）"
        notes, new_papers = [], []

    monkeypatch.setattr(rt, "handle_message", lambda msg, **k: _Reply())
    r = client.post("/api/chat", json={"message": "看看状态"})
    assert r.status_code == 200
    assert r.json()["intent"] == "status"
    # 校验失败：空消息 → 422
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_chat_endpoint_persists_session(monkeypatch, tmp_path):
    lib = _seed(tmp_path)
    lib.close()
    client = _client(monkeypatch, tmp_path)

    class _Reply:
        intent, reply, understanding = "status", "库里有 2 篇", "库状态（LLM）"
        notes, new_papers = [], []

    monkeypatch.setattr(rt, "handle_message", lambda msg, **k: _Reply())
    r = client.post("/api/chat", json={"message": "看看状态"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "status"
    sid = data["session_id"]  # 未带 session_id 时自动创建会话
    # 校验失败：空消息 → 422
    assert client.post("/api/chat", json={"message": ""}).status_code == 422

    # 会话出现在列表，消息已持久化
    sessions = client.get("/api/sessions").json()["sessions"]
    assert [s["id"] for s in sessions] == [sid]
    assert sessions[0]["message_count"] == 2
    msgs = client.get(f"/api/sessions/{sid}/messages").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["intent"] == "status"

    # 继续在同一会话发言；history 作为语境传给 handle_message
    seen = {}

    def _capture(msg, **kwargs):
        seen["history"] = kwargs.get("history")
        return _Reply()

    monkeypatch.setattr(rt, "handle_message", _capture)
    r2 = client.post("/api/chat", json={"message": "详细说说", "session_id": sid})
    assert r2.json()["session_id"] == sid
    assert seen["history"] and seen["history"][0] == ("看看状态", "库里有 2 篇")

    # 不存在的会话 → 404；删除后历史消失
    assert client.post("/api/chat", json={"message": "x", "session_id": "nope"}).status_code == 404
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}/messages").status_code == 404
    assert client.get("/api/sessions").json()["sessions"] == []


def test_run_job_lifecycle(monkeypatch, tmp_path):
    _seed(tmp_path).close()
    client = _client(monkeypatch, tmp_path)

    def fake_pipeline(query, params):
        return {
            "stats": {"new": 2, "downloaded": 1, "selected_n": 2},
            "notes": ["n1"],
            "selected": ["openalex:W1", "openalex:W2"],
        }

    monkeypatch.setattr(rt, "run_pipeline", fake_pipeline)
    r = client.post("/api/jobs/run", json={"query": "gnn"})
    assert r.status_code == 202
    job = r.json()
    assert job["status"] in ("running", "done")  # fake 立即完成也算合法时序

    for _ in range(50):  # 轮询等后台线程完成
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done"
    assert job["result"]["stats"]["new"] == 2
    assert client.get("/api/jobs/no-such-id").status_code == 404


def test_process_paper_job(monkeypatch, tmp_path):
    _seed(tmp_path).close()
    client = _client(monkeypatch, tmp_path)

    def fake_process(sid, cfg, lib):
        return {"source_id": sid, "final_status": "indexed",
                "steps": [{"stage": "download", "ok": True, "detail": "ok"}]}

    monkeypatch.setattr(rt, "process_paper", fake_process)
    r = client.post("/api/papers/openalex:W1/process")
    assert r.status_code == 202
    job = r.json()
    for _ in range(50):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done"
    assert job["result"]["final_status"] == "indexed"
    assert client.post("/api/papers/openalex:NOPE/process").status_code == 404
