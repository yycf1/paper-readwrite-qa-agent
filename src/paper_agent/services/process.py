"""单篇全链处理服务：download → parse → analyze → index，按状态断点续跑。

与 pipeline 的批量扫描不同，这里指定一篇论文从当前状态推进到 indexed；
每一步失败即停并保留 error（前端可重试），语义与 CLI 各命令一致。
"""

from __future__ import annotations

import json
from pathlib import Path

from paper_agent import paths
from paper_agent.download import download_pdf, safe_dirname
from paper_agent.extraction.experiment_flow import extract_experiment
from paper_agent.extraction.report import generate_report
from paper_agent.library import Library
from paper_agent.llm import chat
from paper_agent.parsing import extract_markdown
from paper_agent.rag.embedder import embed
from paper_agent.rag.splitter import split_markdown
from paper_agent.rag.vectorstore import VectorStore


def process_paper(sid: str, cfg: dict, lib: Library) -> dict:
    """把一篇论文从当前状态推进到 indexed，返回各阶段结果摘要。"""
    rec = lib.get(sid)
    if rec is None:
        raise ValueError(f"未找到论文：{sid}")
    out: dict = {"source_id": sid, "title": rec.paper.title, "steps": []}

    def _step(name: str, ok: bool, detail: str = "") -> None:
        out["steps"].append({"stage": name, "ok": ok, "detail": detail})

    # 1. 下载
    rec = lib.get(sid)
    if rec is not None and rec.status == "discovered":
        if not rec.paper.pdf_urls:
            _step("download", False, "无 OA 全文链接（仅摘要），无法处理")
            out["final_status"] = rec.status
            return out
        try:
            path = download_pdf(
                rec.paper, paths.papers_dir(cfg), proxy=cfg.get("proxy", "") or ""
            )
            lib.set_status(sid, "downloaded", error=None, pdf_path=str(path))
            _step("download", True, f"{path.stat().st_size / 1024:.0f} KB")
        except Exception as exc:
            lib.set_status(sid, "download_failed", error=str(exc)[:500])
            _step("download", False, str(exc)[:200])
            out["final_status"] = "download_failed"
            return out
    else:
        _step("download", True, "已有原文或无需下载")

    # 2. 解析
    rec = lib.get(sid)
    if rec is not None and rec.status == "downloaded":
        min_chars = 30 if (rec.pdf_path or "").lower().endswith(".docx") else 200
        try:
            md = extract_markdown(Path(rec.pdf_path))
        except Exception as exc:
            lib.set_status(sid, "downloaded", error=f"解析失败：{exc}"[:500])
            _step("parse", False, str(exc)[:200])
            out["final_status"] = "downloaded"
            return out
        if len(md.strip()) < min_chars:
            lib.set_status(sid, "downloaded", error="解析结果近乎为空（可能是扫描版 PDF）")
            _step("parse", False, "解析结果近乎为空")
            out["final_status"] = "downloaded"
            return out
        out_md = paths.parsed_dir(cfg) / safe_dirname(sid) / "full_text.md"
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
        lib.set_status(sid, "parsed", error=None)
        _step("parse", True, f"{len(md)} 字符")
    else:
        _step("parse", True, "已解析")

    # 3. 知识提取
    rec = lib.get(sid)
    if rec is not None and rec.status == "parsed":
        md_path = paths.parsed_dir(cfg) / safe_dirname(sid) / "full_text.md"
        markdown = md_path.read_text(encoding="utf-8")
        try:
            experiment = extract_experiment(markdown, chat_fn=chat)
            report, _report_usage = generate_report(markdown, chat_fn=chat)
        except Exception as exc:
            lib.set_status(sid, "parsed", error=f"分析失败：{exc}"[:500])
            _step("analyze", False, str(exc)[:200])
            out["final_status"] = "parsed"
            return out
        out_dir = paths.knowledge_dir(cfg) / safe_dirname(sid)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "experiment.json").write_text(
            json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "report.md").write_text(report, encoding="utf-8")
        lib.set_status(sid, "analyzed", error=None)
        _step("analyze", True, "experiment.json + report.md")
    else:
        _step("analyze", True, "已分析")

    # 4. 索引
    rec = lib.get(sid)
    md_path = paths.parsed_dir(cfg) / safe_dirname(sid) / "full_text.md"
    store = VectorStore(paths.vector_dir(cfg))
    existing = store.paper_modes().get(sid)
    if rec is not None and md_path.exists() and existing != "fulltext":
        chunks = split_markdown(md_path.read_text(encoding="utf-8"), sid, rec.paper.title)
        if chunks:
            vectors = embed([c.text for c in chunks])
            metas = [{**c.metadata, "mode": "fulltext"} for c in chunks]
            store.upsert_paper_chunks(sid, [c.text for c in chunks], metas, vectors)
            _step("index", True, f"{len(chunks)} 块")
    else:
        _step("index", True, "已索引")
    # indexed 严格表示「分析+索引都完成」（M6 语义）
    rec = lib.get(sid)
    if rec is not None and rec.status == "analyzed":
        lib.set_status(sid, "indexed", error=None)
    out["final_status"] = lib.get(sid).status
    return out
