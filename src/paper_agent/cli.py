"""paper-agent 命令行入口（Typer）。"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import typer
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from paper_agent import __version__
from paper_agent.config import PROJECT_ROOT, EndpointConfig, load_config, resolve_endpoint
from paper_agent.download import DownloadError, download_pdf, safe_dirname
from paper_agent.graph.hello import run_hello
from paper_agent.library import Library, PaperRecord
from paper_agent.models import normalize_title
from paper_agent.parsing import SUPPORTED_SUFFIXES, extract_markdown
from paper_agent.sources import SourceUnavailable
from paper_agent.sources import arxiv as arxiv_src
from paper_agent.sources import europepmc as epmc_src
from paper_agent.sources import openalex as oa_src

app = typer.Typer(
    help="paper-agent：文献阅读复现 Agent（检索→下载→解析→实验知识提取→RAG问答）",
    no_args_is_help=True,
)
console = Console()

OK = "[green]✅[/green]"
FAIL = "[red]❌[/red]"
SKIP = "[yellow]⏭ 未检查[/yellow]"
WARN = "[yellow]⚠[/yellow]"

SOURCE_MODULES = {"openalex": oa_src, "europepmc": epmc_src, "arxiv": arxiv_src}


@app.command()
def version() -> None:
    """显示版本。"""
    console.print(f"paper-agent {__version__}")


# ---------------------------------------------------------------------------
# M1：检索 / 下载 / 库状态
# ---------------------------------------------------------------------------


@app.command()
def search(
    query: str = typer.Argument(..., help="检索关键词（支持中英文）"),
    source: str = typer.Option(
        "openalex,europepmc,arxiv", "--source", "-s", help="逗号分隔的数据源"
    ),
    max: int = typer.Option(10, "--max", "-m", min=1, help="每源返回条数上限"),
    year_from: int | None = typer.Option(None, "--year-from", help="只保留该年份及以后"),
) -> None:
    """检索论文并入库（状态 discovered），表格展示候选；跨源自动去重。"""
    cfg = load_config()
    proxy = cfg.get("proxy", "") or ""
    source_cfg = cfg.get("sources", {}) or {}
    names = [s.strip().lower() for s in source.split(",") if s.strip()]
    unknown = [n for n in names if n not in SOURCE_MODULES]
    if unknown:
        console.print(f"{FAIL} 未知数据源：{', '.join(unknown)}（可用：{', '.join(SOURCE_MODULES)}）")
        raise typer.Exit(code=1)

    lib = _library(cfg)
    try:
        seen: dict[str, str] = {}  # 去重键（doi 或 归一化标题）→ source_id
        merged: list[PaperRecord] = []
        for name in names:
            if not (source_cfg.get(name) or {}).get("enabled", True):
                console.print(f"{SKIP} {name}：已在 config.yaml 停用")
                continue
            module = SOURCE_MODULES[name]
            try:
                if name == "openalex":
                    email = (source_cfg.get("openalex") or {}).get("email", "") or ""
                    papers = module.search(query, max, year_from, proxy=proxy, email=email)
                else:
                    papers = module.search(query, max, year_from, proxy=proxy)
            except SourceUnavailable as exc:
                hint = "可在 config.yaml 配置 proxy" if name == "arxiv" else "请检查网络"
                console.print(f"{WARN} {name} 不可用（{exc}），已跳过——{hint}")
                continue
            added = 0
            for p in papers:
                dup_id = _duplicate_of(lib, seen, p)
                if dup_id:
                    existing = lib.get(dup_id)
                    if existing:
                        lib.upsert_paper(p)  # 合并补充元数据（状态不变）
                    continue
                lib.upsert_paper(p)
                rec = lib.get(p.source_id)
                if rec:
                    merged.append(rec)
                    added += 1
                if p.doi:
                    seen[f"doi:{p.doi}"] = p.source_id
                seen[f"t:{normalize_title(p.title)}"] = p.source_id
            console.print(f"[cyan]{name}[/cyan]：{len(papers)} 条结果")
        _print_papers(merged, title=f"检索「{query}」：{len(merged)} 篇新入库")
        if merged:
            console.print(
                "\n下一步：[cyan]pa download <id...>[/cyan] 下载 PDF，"
                "或 [cyan]pa download --all[/cyan]"
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
                console.print(f"{SKIP} {skipped_no_pdf} 篇无 OA 全文（仅摘要），自动跳过")
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
            paper = _paper_from_local(sid, title, dest)
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
def status(
    show: int = typer.Option(20, "--show", min=0, help="列表展示条数，0 只看统计"),
) -> None:
    """文献库总览：状态统计与论文列表。"""
    cfg = load_config()
    lib = _library(cfg)
    try:
        counts = lib.counts()
        total = sum(counts.values())
        stat_line = "  ".join(f"{s}: [bold]{n}[/bold]" for s, n in sorted(counts.items()))
        console.print(f"文献库（{lib.db_path}）：共 {total} 篇\n{stat_line}")
        if show:
            _print_papers(lib.list()[:show], title=f"最近 {min(show, total)} 篇")
    finally:
        lib.close()


def _duplicate_of(lib: Library, seen: dict[str, str], p) -> str | None:
    """先用本次会话的内存索引去重，再查账本（跨命令重复检索也不重复入库）。"""
    if p.doi and seen.get(f"doi:{p.doi}"):
        return seen[f"doi:{p.doi}"]
    tkey = f"t:{normalize_title(p.title)}"
    if seen.get(tkey):
        return seen[tkey]
    return lib.find_duplicate(p)


def _print_papers(records: list[PaperRecord], *, title: str) -> None:
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("源", no_wrap=True)
    table.add_column("年份", justify="right")
    table.add_column("引用", justify="right")
    table.add_column("PDF", justify="center")
    table.add_column("状态", no_wrap=True)
    table.add_column("标题", overflow="fold", max_width=52)
    for rec in records:
        p = rec.paper
        pdf_mark = "[green]✓[/green]" if p.pdf_urls else "[yellow]仅摘要[/yellow]"
        status_mark = rec.status
        if rec.status == "download_failed":
            status_mark = f"[red]{rec.status}[/red]"
        table.add_row(
            p.source_id.split(":", 1)[1],
            p.source,
            str(p.year or "-"),
            str(p.citations if p.citations is not None else "-"),
            pdf_mark,
            status_mark,
            p.title,
        )
    console.print(table)


def _resolve(lib: Library, ids: list[str]) -> list[PaperRecord]:
    """把用户输入的 id 片段解析为账本记录；歧义/未找到给出提示。"""
    records: list[PaperRecord] = []
    for fragment in ids:
        fragment = fragment.strip()
        if not fragment:
            continue
        matches = lib.find(fragment)
        exact = [m for m in matches if m == fragment]
        if exact:
            matches = exact
        if not matches:
            console.print(f"{WARN} 未找到「{fragment}」，跳过（用 pa status 查看 id）")
            continue
        if len(matches) > 1:
            console.print(f"{WARN} 「{fragment}」匹配多条（{', '.join(matches)}），请用更完整的 id")
            continue
        rec = lib.get(matches[0])
        if rec:
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# 环境自检（M0/M1）
# ---------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """环境自检：配置 → API Key → 平台连通 → 数据源连通 → LangGraph 图执行。"""
    console.print("[bold]paper-agent 环境自检[/bold]\n")
    table = Table(show_header=True, header_style="bold")
    table.add_column("检查项", style="cyan")
    table.add_column("状态")
    table.add_column("详情", overflow="fold")

    try:
        cfg = load_config()
        table.add_row("配置文件 config.yaml", OK, "加载成功")
    except Exception as exc:
        table.add_row("配置文件 config.yaml", FAIL, str(exc))
        console.print(table)
        raise typer.Exit(code=1) from exc

    platform_ok = _check_endpoint(table, "Embedding 平台", resolve_endpoint(cfg["embedding"]), is_embedding=True)
    platform_ok &= _check_endpoint(table, "LLM 平台", resolve_endpoint(cfg["llm"]), is_embedding=False)
    sources_ok = _check_sources(table, cfg)

    try:
        result = run_hello()
        table.add_row("LangGraph 图执行", OK, result["message"])
        graph_ok = True
    except Exception as exc:
        table.add_row("LangGraph 图执行", FAIL, str(exc))
        graph_ok = False

    console.print(table)
    core_ok = platform_ok and graph_ok
    if core_ok and sources_ok:
        console.print("\n[bold green]结论：全部检查通过。[/bold green]")
    elif core_ok:
        console.print(
            "\n[bold yellow]LLM/图执行正常，但部分数据源不可达[/bold yellow]："
            "检索会自动降级到可用源；如需 arXiv，请在 config.yaml 配置 proxy。"
        )
    else:
        console.print(
            "\n[bold yellow]图执行正常，但平台尚未连通，还差 API Key：[/bold yellow]\n"
            "1) 复制 .env.example 为 .env；\n"
            "2) 填入 Key（硅基流动 https://cloud.siliconflow.cn ｜ 智谱 https://open.bigmodel.cn）；\n"
            "3) 重新运行 [cyan]pa doctor[/cyan]。"
        )
        raise typer.Exit(code=1)


def _check_sources(table: Table, cfg: dict) -> bool:
    """逐源连通探测；返回是否全部可用。"""
    proxy = cfg.get("proxy", "") or ""
    source_cfg = cfg.get("sources", {}) or {}
    all_ok = True
    for name, module in SOURCE_MODULES.items():
        label = f"数据源 · {name}"
        if not (source_cfg.get(name) or {}).get("enabled", True):
            table.add_row(label, SKIP, "已在 config.yaml 停用")
            continue
        try:
            if name == "openalex":
                email = (source_cfg.get("openalex") or {}).get("email", "") or ""
                detail = module.probe(proxy=proxy, email=email)
            else:
                detail = module.probe(proxy=proxy)
            table.add_row(label, OK, detail)
        except SourceUnavailable as exc:
            hint = "可在 config.yaml 配置 proxy" if name == "arxiv" else "检查网络后重试"
            table.add_row(label, FAIL, f"{exc}——{hint}")
            all_ok = False
    return all_ok


def _check_endpoint(table: Table, label: str, ep: EndpointConfig, *, is_embedding: bool) -> bool:
    if not ep.base_url or not ep.model:
        table.add_row(label, SKIP, "config.yaml 未配置 base_url/model")
        return False
    if not ep.api_key:
        table.add_row(label, SKIP, ".env 中未找到 API Key")
        return False
    client = OpenAI(base_url=ep.base_url, api_key=ep.api_key, timeout=30, max_retries=1)
    start = time.perf_counter()
    try:
        if is_embedding:
            client.embeddings.create(model=ep.model, input=["连通性测试"])
        else:
            client.chat.completions.create(
                model=ep.model,
                messages=[{"role": "user", "content": "请只回复两个字符：OK"}],
                max_tokens=8,
                temperature=0,
            )
        cost = time.perf_counter() - start
        table.add_row(label, OK, f"{ep.model} @ {ep.base_url}（耗时 {cost:.1f}s）")
        return True
    except Exception as exc:
        cost = time.perf_counter() - start
        table.add_row(label, FAIL, f"{ep.model} @ {ep.base_url}（耗时 {cost:.1f}s）：{exc}")
        return False


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _data_dir(cfg: dict) -> Path:
    return PROJECT_ROOT / cfg.get("data_dir", "data")


def _papers_dir(cfg: dict) -> Path:
    return _data_dir(cfg) / "papers"


def _parsed_dir(cfg: dict) -> Path:
    return _data_dir(cfg) / "parsed"


def _library(cfg: dict) -> Library:
    return Library(_data_dir(cfg) / "library.db")


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


def _paper_from_local(sid: str, title: str, dest: Path):
    from paper_agent.models import Paper

    return Paper(source="local", source_id=sid, title=title)


if __name__ == "__main__":
    app()
