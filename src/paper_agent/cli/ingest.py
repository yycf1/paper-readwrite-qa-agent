"""文献生命周期命令：search / download / add / parse / analyze / index。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import typer

from paper_agent.cli.core import (
    FAIL,
    OK,
    WARN,
    _data_dir,
    _knowledge_dir,
    _library,
    _parsed_dir,
    _papers_dir,
    _print_papers,
    _resolve,
    app,
    console,
)
from paper_agent.config import load_config
from paper_agent.download import DownloadError, download_pdf, safe_dirname
from paper_agent.extraction.experiment_flow import extract_experiment
from paper_agent.extraction.report import generate_report
from paper_agent.library import PaperRecord
from paper_agent.llm import chat
from paper_agent.models import normalize_title  # noqa: F401  re-export 供旧脚本
from paper_agent.parsing import SUPPORTED_SUFFIXES, extract_markdown
from paper_agent.rag.embedder import embed
from paper_agent.rag.splitter import chunk_abstract, split_markdown
from paper_agent.rag.vectorstore import VectorStore
from paper_agent.services.search import search_and_ingest


@app.command()
def search(
    query: str = typer.Argument(..., help="检索关键词（支持中英文，自动优化为英文检索词）"),
    source: str = typer.Option(
        "openalex,europepmc,arxiv", "--source", "-s", help="逗号分隔的数据源"
    ),
    max: int = typer.Option(10, "--max", "-m", min=1, help="每源返回条数上限"),
    year_from: int | None = typer.Option(None, "--year-from", help="只保留该年份及以后"),
    raw: bool = typer.Option(False, "--raw", help="跳过查询优化，原样直搜"),
) -> None:
    """检索论文并入库（状态 discovered），表格展示候选；跨源自动去重。

    默认先用 LLM 优化查询（中译英、缩写展开、复合需求拆子查询），失败自动降级原样直搜。
    """
    cfg = load_config()
    lib = _library(cfg)
    try:
        try:
            outcome = search_and_ingest(
                query, lib=lib, cfg=cfg, sources=source,
                max_results=max, year_from=year_from, raw=raw,
            )
        except ValueError as exc:
            console.print(f"{FAIL} {exc}")
            raise typer.Exit(code=1) from exc

        plan = outcome.plan
        if plan.get("optimized"):
            console.print(
                f"[dim]查询优化：{' / '.join(plan['topics'])}"
                + (f"（{plan['year_from']} 年起）" if plan.get("year_from") else "")
                + "[/dim]"
            )
        for note in outcome.notes:
            console.print(f"{WARN} {note}")
        for name, hits in outcome.source_hits.items():
            console.print(f"[cyan]{name}[/cyan]：{hits} 条候选")

        label = " / ".join(plan["topics"]) if plan.get("optimized") else query
        _print_papers(
            outcome.new_papers,
            title=f"检索「{label}」：{len(outcome.new_papers)} 篇新入库（{outcome.total_hits} 条候选）",
        )
        if outcome.new_papers:
            console.print(
                "\n下一步：[cyan]pa download <id...>[/cyan] 下载 PDF，"
                "或 [cyan]pa download --all[/cyan]"
            )
        elif outcome.total_hits == 0:
            console.print(
                "\n没有检索到结果。可以尝试：放宽或去掉 [cyan]--year-from[/cyan]、"
                "换更通用的关键词、或加 [cyan]--raw[/cyan] 原样直搜。"
            )
    finally:
        lib.close()


@app.command()
def download(
    ids: list[str] = typer.Argument(None, help="论文 id（可用 source_id 片段，如 W274... 或 2401.12345）"),
    all: bool = typer.Option(False, "--all", help="下载所有 discovered 状态的论文"),
    limit: int = typer.Option(10, "--limit", "-l", min=1, help="--all 模式的下载上限"),
) -> None:
    """下载论文 PDF（开放获取）；失败标记 download_failed，不中断其余下载。"""
    cfg = load_config()
    proxy = cfg.get("proxy", "") or ""
    lib = _library(cfg)
    try:
        if all:
            targets = [r for r in lib.list(status="discovered") if r.paper.pdf_urls]
            skipped_no_pdf = sum(1 for r in lib.list(status="discovered") if not r.paper.pdf_urls)
            targets = targets[:limit]
            if skipped_no_pdf:
                console.print(f"{WARN} {skipped_no_pdf} 篇无 OA 全文（仅摘要），自动跳过")
        else:
            targets = _resolve(lib, ids or [])
        if not targets:
            console.print("没有待下载的论文。先用 [cyan]pa search[/cyan] 检索入库。")
            return

        ok = fail = 0
        for rec in targets:
            p = rec.paper
            try:
                path = download_pdf(p, _papers_dir(cfg), proxy=proxy)
                lib.set_status(p.source_id, "downloaded", error=None, pdf_path=str(path))
                size_kb = path.stat().st_size / 1024
                console.print(f"{OK} [dim]{p.source_id}[/dim] {p.title[:40]}…（{size_kb:.0f} KB）")
                ok += 1
            except DownloadError as exc:
                lib.set_status(p.source_id, "download_failed", error=str(exc)[:500])
                console.print(f"{FAIL} [dim]{p.source_id}[/dim] {p.title[:40]}…：{str(exc)[:120]}")
                fail += 1
        console.print(f"\n下载完成：{ok} 成功，{fail} 失败（失败项可用同样命令重试）")
    finally:
        lib.close()


@app.command()
def add(
    paths: list[Path] = typer.Argument(..., exists=True, readable=True, help="本地 PDF/DOCX 文件路径"),
) -> None:
    """导入本地论文（中文文献、已有 PDF 的入口），状态直达 downloaded。

    同一文件重复导入按内容哈希判重，不会产生重复条目。
    """
    cfg = load_config()
    lib = _library(cfg)
    try:
        for path in paths:
            path = Path(path)
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                console.print(f"{FAIL} {path.name}：不支持的格式（{suffix}），仅支持 PDF/DOCX")
                continue
            data = path.read_bytes()
            digest = hashlib.md5(data).hexdigest()[:8]
            stem = re.sub(r"[\W_]+", "-", path.stem, flags=re.UNICODE).strip("-")[:40] or "document"
            sid = f"local:{stem}-{digest}"
            dest_dir = _papers_dir(cfg) / safe_dirname(sid)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"paper{suffix}"
            if not dest.exists() or dest.read_bytes() != data:
                dest.write_bytes(data)

            title = _title_from_document(path, fallback=stem.replace("-", " "))
            paper = _paper_from_local(sid, title)
            is_new = lib.upsert_paper(paper)
            rec = lib.get(sid)
            if rec is not None and rec.status == "discovered":
                lib.set_status(sid, "downloaded", error=None, pdf_path=str(dest))
            mark = "新导入" if is_new else "已存在（幂等合并）"
            console.print(f"{OK} [dim]{sid}[/dim] {title}——{mark}，状态 {lib.get(sid).status}")
    finally:
        lib.close()


@app.command()
def parse(
    ids: list[str] = typer.Argument(None, help="论文 id 片段；缺省配合 --all"),
    all: bool = typer.Option(False, "--all", help="解析所有 downloaded 状态论文"),
) -> None:
    """PDF/DOCX → 分节 Markdown（缓存于 data/parsed/{id}/full_text.md）。"""
    cfg = load_config()
    lib = _library(cfg)
    try:
        targets = _resolve(lib, ids or []) if ids else [r for r in lib.list(status="downloaded")]
        if not all and not ids:
            targets = [r for r in lib.list(status="downloaded")]
        if not targets:
            console.print("没有待解析的论文（downloaded 状态）。")
            return
        ok = fail = 0
        for rec in targets:
            sid, p = rec.paper.source_id, rec.paper
            if not rec.pdf_path or not Path(rec.pdf_path).exists():
                console.print(f"{FAIL} [dim]{sid}[/dim] {p.title[:36]}…：原文文件缺失")
                fail += 1
                continue
            try:
                md = extract_markdown(rec.pdf_path)
            except Exception as exc:
                lib.set_status(sid, "downloaded", error=f"解析失败：{exc}"[:500])
                console.print(f"{FAIL} [dim]{sid}[/dim] {p.title[:36]}…：解析失败 {exc}")
                fail += 1
                continue
            # 空结果守卫仅针对 PDF（扫描版常见）；DOCX 无 OCR 问题，短文档也放行
            min_chars = 30 if rec.pdf_path.lower().endswith(".docx") else 200
            if len(md.strip()) < min_chars:
                lib.set_status(sid, "downloaded", error="解析结果近乎为空（可能是扫描版 PDF）")
                console.print(f"{FAIL} [dim]{sid}[/dim] {p.title[:36]}…：解析结果近乎为空")
                fail += 1
                continue
            out = _parsed_dir(cfg) / safe_dirname(sid) / "full_text.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(md, encoding="utf-8")
            sections = md.count("\n## ")
            lib.set_status(sid, "parsed", error=None)
            console.print(f"{OK} [dim]{sid}[/dim] {p.title[:36]}…（{len(md)} 字符，{sections} 节）")
            ok += 1
        console.print(f"\n解析完成：{ok} 成功，{fail} 失败")
    finally:
        lib.close()


@app.command()
def analyze(
    ids: list[str] = typer.Argument(None, help="论文 id 片段；缺省配合 --all"),
    all: bool = typer.Option(False, "--all", help="分析所有 parsed 状态论文"),
) -> None:
    """LLM 知识提取：生成 experiment.json + report.md（状态 analyzed）。

    需要 LLM API Key（.env）。仅 downloaded 未解析的论文会自动先解析。
    """
    cfg = load_config()
    lib = _library(cfg)
    try:
        if ids:
            targets = _resolve(lib, ids)
        elif all:
            targets = lib.list(status="parsed")
        else:
            targets = []
        if not targets:
            console.print("没有待分析的论文（parsed 状态）。")
            return
        ok = fail = 0
        total_tokens = 0
        for rec in targets:
            sid, p = rec.paper.source_id, rec.paper
            parsed_path = _parsed_dir(cfg) / safe_dirname(sid) / "full_text.md"
            if not parsed_path.exists():
                console.print(f"{WARN} [dim]{sid}[/dim] 未解析，自动补跑 parse…")
                if rec.status != "downloaded" or not rec.pdf_path:
                    console.print(f"{FAIL} [dim]{sid}[/dim] 无法解析（无原文），跳过")
                    fail += 1
                    continue
                try:
                    md = extract_markdown(rec.pdf_path)
                    parsed_path.parent.mkdir(parents=True, exist_ok=True)
                    parsed_path.write_text(md, encoding="utf-8")
                    lib.set_status(sid, "parsed")
                except Exception as exc:
                    console.print(f"{FAIL} [dim]{sid}[/dim] 自动解析失败：{exc}")
                    fail += 1
                    continue
            try:
                markdown = parsed_path.read_text(encoding="utf-8")
                console.print(f"⏳ [dim]{sid}[/dim] {p.title[:36]}… 提取实验知识…")
                experiment = extract_experiment(markdown, chat_fn=chat)
                console.print(f"⏳ [dim]{sid}[/dim] {p.title[:36]}… 生成精读报告…")
                report, report_usage = generate_report(markdown, chat_fn=chat)
                tokens = (
                    experiment["token_usage"]["prompt"] + experiment["token_usage"]["completion"]
                    + report_usage["prompt"] + report_usage["completion"]
                )
                total_tokens += tokens
            except RuntimeError as exc:  # 缺 Key 等配置问题，直接中断
                console.print(f"{FAIL} {exc}")
                raise typer.Exit(code=1) from exc
            except Exception as exc:
                lib.set_status(sid, "parsed", error=f"分析失败：{exc}"[:500])
                console.print(f"{FAIL} [dim]{sid}[/dim] {p.title[:36]}…：{str(exc)[:160]}")
                fail += 1
                continue
            out_dir = _knowledge_dir(cfg) / safe_dirname(sid)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "experiment.json").write_text(
                json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (out_dir / "report.md").write_text(report, encoding="utf-8")
            lib.set_status(sid, "analyzed", error=None)
            low_conf = sum(1 for v in experiment["confidence"].values() if v == "low")
            console.print(
                f"{OK} [dim]{sid}[/dim] {p.title[:36]}…（~{tokens} tokens"
                + (f"，{low_conf} 个低置信字段" if low_conf else "")
                + "）"
            )
            ok += 1
        console.print(
            f"\n分析完成：{ok} 成功，{fail} 失败，共用约 {total_tokens} tokens"
            f"（产物在 data/knowledge/ 下）"
        )
    finally:
        lib.close()


@app.command()
def index(
    ids: list[str] = typer.Argument(None, help="论文 id 片段；缺省索引 parsed/analyzed 状态论文"),
    all: bool = typer.Option(False, "--all", help="索引全部可索引内容（含仅摘要论文）"),
    force: bool = typer.Option(False, "--force", help="已在向量库的论文也重建索引"),
) -> None:
    """结构感知分块 → 向量化 → Chroma 增量索引。

    已解析全文的论文按全文分块；无全文但有摘要的论文把摘要作为单块入库
    （不改变生命周期状态，后续下载成功可再升级为全文索引）。
    indexed 严格表示「分析+索引都完成」：parsed（分析失败/未跑）保持原状，
    `pa analyze` 重试仍能扫到。
    """
    cfg = load_config()
    lib = _library(cfg)
    try:
        if ids:
            targets = _resolve(lib, ids)
        elif all:
            targets = lib.list()
        else:
            targets = lib.list(status="parsed") + lib.list(status="analyzed")
        if not targets:
            console.print("没有可索引的论文（需先 parse，或论文带摘要）。")
            return
        store = VectorStore(_data_dir(cfg) / "db")
        existing = store.paper_modes()
        ok = skip = 0
        for rec in targets:
            sid, p = rec.paper.source_id, rec.paper
            parsed_path = _parsed_dir(cfg) / safe_dirname(sid) / "full_text.md"
            if parsed_path.exists():
                mode = "fulltext"
            elif p.abstract.strip():
                mode = "abstract"
            else:
                console.print(f"{WARN} [dim]{sid}[/dim] {p.title[:36]}…：无全文也无摘要，跳过")
                continue
            if existing.get(sid) == mode and not force:
                # 先索引后分析的时序：向量库已有全文的 analyzed 论文补推进状态
                if mode == "fulltext" and rec.status == "analyzed":
                    lib.set_status(sid, "indexed", error=None)
                skip += 1
                continue
            if mode == "fulltext":
                markdown = parsed_path.read_text(encoding="utf-8")
                chunks = split_markdown(markdown, sid, p.title)
            else:
                chunks = chunk_abstract(sid, p.title, p.abstract)
            if not chunks:
                console.print(f"{FAIL} [dim]{sid}[/dim] {p.title[:36]}…：分块结果为空")
                continue
            try:
                vectors = embed([c.text for c in chunks])
            except RuntimeError as exc:
                console.print(f"{FAIL} {exc}")
                raise typer.Exit(code=1) from exc
            metas = [{**c.metadata, "mode": mode} for c in chunks]
            store.upsert_paper_chunks(sid, [c.text for c in chunks], metas, vectors)
            # 仅对 analyzed 推进状态；parsed（分析失败/未跑）保持原状
            if mode == "fulltext" and rec.status == "analyzed":
                lib.set_status(sid, "indexed", error=None)
            console.print(
                f"{OK} [dim]{sid}[/dim] {p.title[:36]}…（{len(chunks)} 块，{'全文' if mode == 'fulltext' else '仅摘要'}）"
            )
            ok += 1
        extra = f"，{skip} 篇已索引自动跳过（--force 重建）" if skip else ""
        console.print(f"\n索引完成：{ok} 篇入库，向量库共 {store.count()} 块{extra}")
    finally:
        lib.close()


def _title_from_document(path: Path, *, fallback: str) -> str:
    """取文档内建标题元数据；取不到（或像文件名/路径）则退回清洗后的文件名。"""
    if path.suffix.lower() == ".pdf":
        try:
            import pymupdf

            with pymupdf.open(path) as doc:
                meta_title = (doc.metadata or {}).get("title") or ""
            if meta_title.strip() and "untitled" not in meta_title.lower():
                return meta_title.strip()[:200]
        except Exception:
            pass
    elif path.suffix.lower() == ".docx":
        try:
            from docx import Document

            doc = Document(str(path))
            for para in doc.paragraphs[:8]:
                style = (para.style.name or "").lower() if para.style is not None else ""
                text = para.text.strip()
                if text and style in ("title", "heading 1"):
                    return text[:200]
        except Exception:
            pass
    return fallback


def _paper_from_local(sid: str, title: str):
    from paper_agent.models import Paper

    return Paper(source="local", source_id=sid, title=title)
