"""REST API 路由：/api/*。只做参数解析、状态码映射与 JSON 序列化。"""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from paper_agent import __version__, paths
from paper_agent.chatstore import ChatStore
from paper_agent.config import load_config
from paper_agent.download import safe_dirname
from paper_agent.graph.pipeline import run_pipeline
from paper_agent.library import Library, PaperRecord
from paper_agent.rag.qa import Retriever, answer_question
from paper_agent.rag.vectorstore import VectorStore
from paper_agent.services.assistant import handle_message, suggest_directions
from paper_agent.services.process import process_paper
from paper_agent.services.search import search_and_ingest
from paper_agent.server import jobs

router = APIRouter()


# ---------------------------------------------------------------------------
# 依赖：每请求独立打开账本连接
# ---------------------------------------------------------------------------


def open_lib() -> Library:
    lib = paths.library(load_config())
    try:
        yield lib
    finally:
        lib.close()


def _retriever() -> Retriever:
    return Retriever(VectorStore(paths.vector_dir(load_config())))


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------


def record_to_dict(rec: PaperRecord, *, with_abstract: bool = False) -> dict:
    p = rec.paper
    out = {
        "source_id": p.source_id,
        "source": p.source,
        "title": p.title,
        "authors": p.authors,
        "year": p.year,
        "doi": p.doi,
        "citations": p.citations,
        "has_pdf": bool(p.pdf_urls),
        "landing_page": p.landing_page,
        "status": rec.status,
        "error": rec.error,
        "added_at": rec.added_at,
    }
    if with_abstract:
        out["abstract"] = p.abstract
    return out


# ---------------------------------------------------------------------------
# 库浏览
# ---------------------------------------------------------------------------


@router.get("/status")
def api_status() -> dict:
    cfg = load_config()
    lib = paths.library(cfg)
    try:
        counts = lib.counts()
    finally:
        lib.close()
    chunks = 0
    try:
        chunks = VectorStore(paths.vector_dir(cfg)).count()
    except Exception:
        pass  # 向量库未初始化（首次使用）不视为错误
    return {
        "total": sum(counts.values()),
        "counts": counts,
        "vector_chunks": chunks,
        "version": __version__,
    }


@router.get("/papers")
def api_papers(
    status: str | None = Query(None, description="按状态过滤"),
    q: str | None = Query(None, description="标题/作者/DOI 模糊匹配"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    lib: Library = Depends(open_lib),
) -> dict:
    records = lib.list(status=status)
    if q:
        needle = q.lower()
        records = [
            r for r in records
            if needle in r.paper.title.lower()
            or needle in (r.paper.doi or "").lower()
            or any(needle in a.lower() for a in r.paper.authors)
        ]
    return {
        "total": len(records),
        "papers": [record_to_dict(r) for r in records[offset: offset + limit]],
    }


@router.get("/papers/{sid}")
def api_paper_detail(sid: str, lib: Library = Depends(open_lib)) -> dict:
    rec = lib.get(sid)
    if rec is None:  # 允许片段匹配（与 CLI 一致的容错），歧义时报 400
        matches = lib.find(sid)
        if len(matches) == 1:
            rec = lib.get(matches[0])
        elif len(matches) > 1:
            raise HTTPException(400, f"「{sid}」匹配多条：{', '.join(matches)}")
    if rec is None:
        raise HTTPException(404, f"未找到论文：{sid}")

    cfg = load_config()
    out = record_to_dict(rec, with_abstract=True)

    knowledge = paths.knowledge_dir(cfg) / safe_dirname(rec.paper.source_id)
    experiment_path = knowledge / "experiment.json"
    report_path = knowledge / "report.md"
    fulltext_path = paths.parsed_dir(cfg) / safe_dirname(rec.paper.source_id) / "full_text.md"
    out["experiment"] = (
        json.loads(experiment_path.read_text(encoding="utf-8"))
        if experiment_path.exists() else None
    )
    out["report"] = (
        report_path.read_text(encoding="utf-8") if report_path.exists() else None
    )
    out["full_text"] = (
        fulltext_path.read_text(encoding="utf-8") if fulltext_path.exists() else None
    )
    return out


# ---------------------------------------------------------------------------
# 检索 / 问答 / 对话
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    sources: str = "openalex,europepmc,arxiv"
    max_results: int = Field(10, ge=1, le=50)
    year_from: int | None = Field(None, ge=1900, le=2100)
    raw: bool = False


@router.post("/search")
def api_search(req: SearchRequest, lib: Library = Depends(open_lib)) -> dict:
    cfg = load_config()
    try:
        outcome = search_and_ingest(
            req.query, lib=lib, cfg=cfg, sources=req.sources,
            max_results=req.max_results, year_from=req.year_from, raw=req.raw,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:  # 缺 Key 等环境问题
        raise HTTPException(503, str(exc)) from exc
    return {
        "new_papers": [record_to_dict(r, with_abstract=True) for r in outcome.new_papers],
        "total_hits": outcome.total_hits,
        "notes": outcome.notes,
        "source_hits": outcome.source_hits,
        "topics": outcome.plan.get("topics", [req.query]),
        "optimized": outcome.plan.get("optimized", False),
    }


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    paper_id: str | None = None
    k: int = Field(8, ge=1, le=20)


@router.post("/ask")
def api_ask(req: AskRequest) -> dict:
    retriever = _retriever()
    if retriever.store.count() == 0:
        raise HTTPException(400, "向量库为空：先检索并处理论文建立索引")
    try:
        answer, hits = answer_question(
            req.question, retriever=retriever, k=req.k, paper_id=req.paper_id
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    seen: set[str] = set()
    sources = []
    for h in hits:  # 同论文多块收敛为一个来源条目（前端展示用）
        key = f"{h.paper_id}:{h.section}"
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "paper_id": h.paper_id, "title": h.title,
            "section": h.section, "page": h.page,
        })
    return {"answer": answer, "sources": sources}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


@router.post("/chat")
def api_chat(req: ChatRequest, lib: Library = Depends(open_lib)) -> dict:
    cfg = load_config()
    p = cfg.get("pipeline", {}) or {}
    params = {
        "max_per_source": 10,
        "year_from": p.get("year_from"),
        "top_n": p.get("top_n", 5),
        "download_budget": p.get("download_budget", 10),
    }
    store = ChatStore(paths.data_dir(cfg) / "library.db")
    try:
        sid = req.session_id
        if sid is None:
            sid = store.create_session(title=req.message)
        elif not any(s["id"] == sid for s in store.list_sessions()):
            raise HTTPException(404, f"未找到会话：{sid}（可能已删除）")
        prior = store.get_messages(sid)
        store.append_message(sid, role="user", content=req.message)
        # 近几轮问答作为语境传给问答链（指代消解），取相邻的 (user, assistant) 对
        history = [
            (prior[i]["content"], prior[i + 1]["content"])
            for i in range(len(prior) - 1)
            if prior[i]["role"] == "user" and prior[i + 1]["role"] == "assistant"
        ][-3:]

        reply = handle_message(
            req.message, lib=lib, retriever=_retriever(), params=params, cfg=cfg,
            history=history,
        )
        store.append_message(
            sid, role="assistant", content=reply.reply, intent=reply.intent,
            understanding=reply.understanding, notes=reply.notes,
        )
        return {
            "session_id": sid,
            "intent": reply.intent,
            "reply": reply.reply,
            "understanding": reply.understanding,
            "notes": reply.notes,
            "new_papers": [record_to_dict(r) for r in reply.new_papers],
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 会话管理
# ---------------------------------------------------------------------------


@router.get("/sessions")
def api_sessions() -> dict:
    store = _chat_store()
    try:
        return {"sessions": store.list_sessions()}
    finally:
        store.close()


class SessionCreate(BaseModel):
    title: str = Field("新对话", max_length=60)


@router.post("/sessions", status_code=201)
def api_create_session(req: SessionCreate) -> dict:
    store = _chat_store()
    try:
        sid = store.create_session(title=req.title)
        return {"id": sid, "title": req.title[:30], "message_count": 0}
    finally:
        store.close()


@router.get("/sessions/{sid}/messages")
def api_session_messages(sid: str) -> dict:
    store = _chat_store()
    try:
        if not any(s["id"] == sid for s in store.list_sessions()):
            raise HTTPException(404, f"未找到会话：{sid}")
        return {"session_id": sid, "messages": store.get_messages(sid)}
    finally:
        store.close()


@router.delete("/sessions/{sid}")
def api_delete_session(sid: str) -> dict:
    store = _chat_store()
    try:
        if not store.delete_session(sid):
            raise HTTPException(404, f"未找到会话：{sid}")
        return {"ok": True}
    finally:
        store.close()


def _chat_store() -> ChatStore:
    return ChatStore(paths.data_dir(load_config()) / "library.db")


@router.get("/explore")
def api_explore(lib: Library = Depends(open_lib)) -> dict:
    return {"directions": suggest_directions(lib)}


# ---------------------------------------------------------------------------
# 长任务：流水线 / 单篇处理
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_n: int | None = Field(None, ge=1, le=50)
    year_from: int | None = Field(None, ge=1900, le=2100)
    budget: int | None = Field(None, ge=1, le=100)
    max_per_source: int = Field(10, ge=1, le=50)


@router.post("/jobs/run", status_code=202)
def api_run_job(req: RunRequest) -> dict:
    cfg = load_config()
    p = cfg.get("pipeline", {}) or {}
    params = {
        "max_per_source": req.max_per_source,
        "top_n": req.top_n if req.top_n is not None else p.get("top_n", 5),
        "year_from": req.year_from if req.year_from is not None else p.get("year_from"),
        "download_budget": req.budget if req.budget is not None else p.get("download_budget", 10),
    }

    def _run(_result: dict) -> dict:
        state = run_pipeline(req.query, params)
        return {"stats": state["stats"], "notes": state["notes"], "selected": state["selected"]}

    job = jobs.submit("run_pipeline", _run, result={"query": req.query, "params": params})
    return job.to_dict()


@router.post("/papers/{sid}/process", status_code=202)
def api_process_paper(sid: str, lib: Library = Depends(open_lib)) -> dict:
    if lib.get(sid) is None:
        raise HTTPException(404, f"未找到论文：{sid}")

    def _run(_result: dict) -> dict:
        cfg = load_config()
        job_lib = paths.library(cfg)  # 任务线程用自己的连接
        try:
            return process_paper(sid, cfg, job_lib)
        finally:
            job_lib.close()

    job = jobs.submit("process_paper", _run, result={"source_id": sid})
    return job.to_dict()


@router.get("/jobs/{job_id}")
def api_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"未找到任务：{job_id}（或服务已重启）")
    return job.to_dict()


@router.get("/jobs")
def api_jobs() -> dict:
    return {"jobs": [j.to_dict() for j in jobs.list_jobs()]}
