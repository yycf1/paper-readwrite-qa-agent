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

from paper_agent import __version__, paths
from paper_agent.config import PROJECT_ROOT, EndpointConfig, load_config, resolve_endpoint
from paper_agent.download import DownloadError, download_pdf, safe_dirname
from paper_agent.graph.hello import run_hello
from paper_agent.graph.pipeline import run_pipeline, run_search as pipeline_search
from paper_agent.graph.route import classify_intent, optimize_query, summarize_query_understanding
from paper_agent.library import Library, PaperRecord
from paper_agent.extraction.experiment_flow import extract_experiment
from paper_agent.extraction.report import generate_report
from paper_agent.llm import chat
from paper_agent.models import normalize_title
from paper_agent.parsing import SUPPORTED_SUFFIXES, extract_markdown
from paper_agent.rag.embedder import embed
from paper_agent.rag.eval import judge_answer, judge_retrieval, load_evalset as evalset_load
from paper_agent.rag.qa import Retriever, answer_question
from paper_agent.rag.splitter import chunk_abstract, split_markdown
from paper_agent.rag.vectorstore import VectorStore
from paper_agent.sources import SourceUnavailable
from paper_agent.tools import filter_valid_papers, search_source_tool, validate_papers
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
    proxy = cfg.get("proxy", "") or ""
    source_cfg = cfg.get("sources", {}) or {}
    names = [s.strip().lower() for s in source.split(",") if s.strip()]
    unknown = [n for n in names if n not in SOURCE_MODULES]
    if unknown:
        console.print(f"{FAIL} 未知数据源：{', '.join(unknown)}（可用：{', '.join(SOURCE_MODULES)}）")
        raise typer.Exit(code=1)

    # 查询优化（失败降级原样）
    if raw:
        plan = {"topics": [query], "year_from": year_from, "max_results": None, "optimized": False}
    else:
        plan = optimize_query(query)
        if plan["optimized"]:
            console.print(
                f"[dim]查询优化：{' / '.join(plan['topics'])}"
                + (f"（{plan['year_from']} 年起）" if plan["year_from"] else "")
                + "[/dim]"
            )
    eff_year = year_from or plan.get("year_from")
    eff_max = plan.get("max_results") or max

    lib = _library(cfg)
    try:
        seen: dict[str, str] = {}  # 去重键（doi 或 归一化标题）→ source_id
        merged: list[PaperRecord] = []
        total_hits = 0

        def _search_one(keyword: str) -> int:
            """对单个查询词跑全部源并去重入库，返回本次各源返回总数。"""
            hits = 0
            for name in names:
                if not (source_cfg.get(name) or {}).get("enabled", True):
                    console.print(f"{SKIP} {name}：已在 config.yaml 停用")
                    continue
                email = (source_cfg.get(name) or {}).get("email", "") or ""
                result = search_source_tool(
                    name, query=keyword, max_results=eff_max, year_from=eff_year,
                    proxy=proxy, email=email,
                )
                if result.status == "fatal":
                    console.print(f"{FAIL} {name}：{result.error}")
                    continue
                if result.status != "ok":
                    hint = "可在 config.yaml 配置 proxy" if name == "arxiv" else "请检查网络"
                    console.print(f"{WARN} {name} 不可用（{result.error}），已跳过——{hint}")
                    continue
                papers, issues = filter_valid_papers(result.data)
                for issue in issues:
                    console.print(f"{WARN} 结果校验拦截：{issue}")
                hits += len(papers)
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
                    if p.doi:
                        seen[f"doi:{p.doi}"] = p.source_id
                    seen[f"t:{normalize_title(p.title)}"] = p.source_id
                console.print(f"[cyan]{name}[/cyan]：「{keyword}」{len(papers)} 条结果")
            return hits

        for topic in plan["topics"]:
            total_hits += _search_one(topic)

        # 优化后的检索词全军覆没 → 回退原始词重搜一轮
        if total_hits == 0 and plan.get("optimized"):
            console.print("[dim]优化后的检索词没有结果，回退原始词重搜…[/dim]")
            total_hits = _search_one(query)

        label = " / ".join(plan["topics"]) if plan["optimized"] else query
        _print_papers(merged, title=f"检索「{label}」：{len(merged)} 篇新入库（{total_hits} 条候选）")
        if merged:
            console.print(
                "\n下一步：[cyan]pa download <id...>[/cyan] 下载 PDF，"
                "或 [cyan]pa download --all[/cyan]"
            )
        elif total_hits == 0:
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
            import json as _json

            (out_dir / "experiment.json").write_text(
                _json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8"
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
    """结构感知分块 → 向量化 → Chroma 增量索引（状态推进到 indexed）。

    已解析全文的论文按全文分块；无全文但有摘要的论文把摘要作为单块入库
    （不改变生命周期状态，后续下载成功可再升级为全文索引）。
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
            if mode == "fulltext":
                lib.set_status(sid, "indexed", error=None)
            console.print(
                f"{OK} [dim]{sid}[/dim] {p.title[:36]}…（{len(chunks)} 块，{'全文' if mode == 'fulltext' else '仅摘要'}）"
            )
            ok += 1
        extra = f"，{skip} 篇已索引自动跳过（--force 重建）" if skip else ""
        console.print(f"\n索引完成：{ok} 篇入库，向量库共 {store.count()} 块{extra}")
    finally:
        lib.close()


@app.command()
def ask(
    question: str = typer.Argument(None, help="问题；缺省进入多轮问答 REPL"),
    paper: str = typer.Option("", "--paper", "-p", help="限定在某篇论文内回答（id 片段）"),
    k: int = typer.Option(8, "--k", min=1, help="检索块数"),
) -> None:
    """文献库问答：检索 top-k → 带出处回答；范围外明确回答「未提及」。"""
    cfg = load_config()
    paper_id = None
    if paper:
        lib = _library(cfg)
        try:
            matches = lib.find(paper)
            if not matches:
                console.print(f"{FAIL} 未找到论文「{paper}」")
                raise typer.Exit(code=1)
            if len(matches) > 1:
                console.print(f"{FAIL} 「{paper}」匹配多条（{', '.join(matches)}），请用更完整的 id")
                raise typer.Exit(code=1)
            paper_id = matches[0]
        finally:
            lib.close()

    store = VectorStore(_data_dir(cfg) / "db")
    if store.count() == 0:
        console.print("向量库为空：先运行 [cyan]pa index --all[/cyan] 建立索引。")
        return
    retriever = Retriever(store)

    if question:
        _answer_once(question, retriever, k=k, paper_id=paper_id)
        return

    scope = f"（限定：{paper_id}）" if paper_id else "（全库）"
    console.print(f"[bold]多轮问答模式{scope}[/bold]，输入问题回车作答，[cyan]q[/cyan] 退出。")
    history: list[tuple[str, str]] = []
    while True:
        try:
            q = console.input("[bold cyan]问题> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        except UnicodeDecodeError:
            console.print(f"{WARN} 输入编码无法解码，请重试")
            continue
        if not q or q.lower() in ("q", "quit", "exit"):
            break
        answer = _answer_once(q, retriever, k=k, paper_id=paper_id, history=history)
        history.append((q, answer))


def _answer_once(
    question: str,
    retriever: Retriever,
    *,
    k: int,
    paper_id: str | None,
    history: list[tuple[str, str]] | None = None,
) -> str:
    try:
        answer, _hits = answer_question(
            question, retriever=retriever, history=history, k=k, paper_id=paper_id
        )
    except RuntimeError as exc:  # 缺 Key 等配置问题
        console.print(f"{FAIL} {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold]答[/bold] {answer}\n")
    return answer


@app.command("eval")
def eval_cmd(
    k: int = typer.Option(8, "--k", min=1, help="每条 case 的检索块数"),
    with_llm: bool = typer.Option(
        False, "--with-llm", help="同时生成答案并按关键词判定（消耗 LLM 调用）"
    ),
    evalset: Path = typer.Option(
        PROJECT_ROOT / "eval" / "evalset.json", "--evalset", help="评测集 JSON 路径"
    ),
) -> None:
    """RAG 回归评测：检索命中率；--with-llm 附加答案关键词命中率。"""
    cfg = load_config()
    try:
        data = evalset_load(evalset)
    except Exception as exc:
        console.print(f"{FAIL} 评测集加载失败：{exc}")
        raise typer.Exit(code=1) from exc
    cases = data["cases"]
    console.print(f"[bold]RAG 评测[/bold]：{data.get('name', evalset.name)}，{len(cases)} 条")

    store = VectorStore(_data_dir(cfg) / "db")
    if store.count() == 0:
        console.print("向量库为空：先运行 [cyan]pa index --all[/cyan]。")
        return
    retriever = Retriever(store)

    table = Table(show_header=True, header_style="bold")
    table.add_column("case", style="cyan", no_wrap=True)
    table.add_column("问题", overflow="fold", max_width=44)
    table.add_column("检索", justify="center")
    table.add_column("关键词", justify="center")
    rows = []
    llm_error = False
    for case in cases:
        hits = retriever.retrieve(case["question"], k=k, paper_id=case.get("paper_id"))
        retrieval_ok = judge_retrieval(case, hits)
        answer_ok = None
        if with_llm:
            try:
                answer, _ = answer_question(
                    case["question"], retriever=retriever, k=k,
                    paper_id=case.get("paper_id"), hits=hits,
                )
                answer_ok = judge_answer(case, answer)
            except RuntimeError as exc:
                llm_error = True
                console.print(f"{WARN} LLM 不可用（{str(exc)[:80]}），答案判定跳过")
                with_llm = False
        r_mark = {True: "[green]✓[/green]", False: "[red]✗[/red]", None: "–"}[retrieval_ok]
        a_mark = {True: "[green]✓[/green]", False: "[red]✗[/red]", None: "–"}[answer_ok]
        table.add_row(case["id"], case["question"][:44], r_mark, a_mark)
        rows.append((retrieval_ok, answer_ok))
    console.print(table)

    r_judged = [r for r, _ in rows if r is not None]
    r_pass = sum(1 for r in r_judged if r)
    if r_judged:
        console.print(
            f"检索命中率：[bold]{r_pass}/{len(r_judged)}（{r_pass / len(r_judged):.0%}）[/bold]"
        )
    if with_llm:
        a_judged = [a for _, a in rows if a is not None]
        a_pass = sum(1 for a in a_judged if a)
        if a_judged:
            console.print(
                f"答案关键词命中率：[bold]{a_pass}/{len(a_judged)}（{a_pass / len(a_judged):.0%}）[/bold]"
            )
    if llm_error:
        console.print(f"{WARN} 答案判定因 LLM 配置缺失被跳过（补齐 .env Key 后用 --with-llm 重跑）")


# ---------------------------------------------------------------------------
# M4：一键流水线与意图路由
# ---------------------------------------------------------------------------


@app.command()
def run(
    query: str = typer.Argument(..., help="检索主题（支持中英文）"),
    max_per_source: int = typer.Option(10, "--max", "-m", min=1, help="每源检索条数"),
    top_n: int = typer.Option(None, "--top-n", min=1, help="无人值守选文数量（默认读 config）"),
    year_from: int = typer.Option(None, "--year-from", help="年份过滤（默认读 config）"),
    budget: int = typer.Option(None, "--budget", min=1, help="下载预算（默认读 config）"),
) -> None:
    """一键流水线：检索 → 选文 → 下载 → 解析 → 知识提取 → 索引。

    基于 library.db 状态断点续跑：已完成阶段的论文自动跳过，
    中断后重跑同一主题即可从断点继续。
    """
    cfg = load_config()
    p = cfg.get("pipeline", {}) or {}
    params = {
        "max_per_source": max_per_source,
        "top_n": top_n if top_n is not None else p.get("top_n", 5),
        "year_from": year_from if year_from is not None else p.get("year_from"),
        "download_budget": budget if budget is not None else p.get("download_budget", 10),
    }
    console.print(
        f"[bold]一键流水线[/bold]：「{query}」"
        f"（top_n={params['top_n']}, year_from={params['year_from']}, 预算={params['download_budget']}）\n"
    )
    state = run_pipeline(query, params)
    stats = state["stats"]

    table = Table(title="流水线执行结果", show_header=True, header_style="bold")
    table.add_column("阶段", style="cyan")
    table.add_column("计数", overflow="fold")
    table.add_column("说明", overflow="fold")
    rows = [
        ("search", f"检索 {stats.get('searched', 0)}，新入库 {stats.get('new', 0)}",
         f"选中 top {stats.get('selected_n', 0)}（按引用数）"),
        ("download", f"成功 {stats.get('downloaded', 0)}，失败 {stats.get('download_failed', 0)}",
         f"{stats.get('abstract_only', 0)} 篇仅摘要"),
        ("parse", f"成功 {stats.get('parsed', 0)}，失败 {stats.get('parse_failed', 0)}", ""),
        ("analyze", f"成功 {stats.get('analyzed', 0)}，失败 {stats.get('analyze_failed', 0)}",
         f"~{stats.get('analyze_tokens', 0)} tokens" if stats.get("analyze_tokens") else ""),
        ("index", f"全文 {stats.get('indexed', 0)} 篇 / {stats.get('index_chunks', 0)} 块，摘要 {stats.get('abstract_indexed', 0)} 篇", ""),
    ]
    for stage, count, remark in rows:
        table.add_row(stage, count, remark)
    console.print(table)
    for note in state["notes"]:
        console.print(f"{WARN} {note}")
    console.print("\n下一步：[cyan]pa ask \"问题\"[/cyan] 开始问答，或 [cyan]pa status[/cyan] 查看总览。")


@app.command("chat")
def chat_cmd() -> None:
    """交互式助手：自然语言说需求，自动路由到检索 / 问答 / 状态 / 方向引导。"""
    cfg = load_config()
    lib = _library(cfg)
    store = VectorStore(paths.vector_dir(cfg))
    retriever = Retriever(store)
    params = {
        "max_per_source": 10,
        "year_from": (cfg.get("pipeline", {}) or {}).get("year_from"),
        "top_n": (cfg.get("pipeline", {}) or {}).get("top_n", 5),
        "download_budget": (cfg.get("pipeline", {}) or {}).get("download_budget", 10),
    }
    console.print(
        "[bold]paper-agent 助手[/bold]（自然语言，q 退出）\n"
        "我能帮你：\n"
        "  ① [cyan]检索新论文[/cyan]——「帮我找 2023 年以后的 transformer 综述，前 20 篇」（支持中文，自动译成英文检索词）\n"
        "  ② [cyan]问答已入库文献[/cyan]——「S-MBRec 的损失函数是什么」（回答带出处）\n"
        "  ③ [cyan]查看库状态[/cyan]——「看看库里进度」\n"
        "  ④ [cyan]推荐研究方向[/cyan]——「不知道该看什么论文」（基于库内已有论文）\n"
        "[dim]只聊科研和论文哦，别的忙帮不上～[/dim]\n"
    )
    try:
        while True:
            try:
                text = console.input("[bold cyan]pa> [/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            except UnicodeDecodeError:
                console.print(f"{WARN} 输入编码无法解码，请重试")
                continue
            if not text or text.lower() in ("q", "quit", "exit"):
                break
            route = classify_intent(text)
            console.print(f"[dim]（{summarize_query_understanding(route)}）[/dim]")
            if route["intent"] == "exit":
                break
            if route["intent"] == "help":
                console.print(
                    "我支持：①检索新论文（「帮我找…主题…的论文」，可带年份/数量）"
                    "②问答已入库文献（直接提问）③查看库状态（「看看状态」）"
                    "④方向推荐（「不知道看什么论文」）。也可以直接用 pa 命令。"
                )
                continue
            if route["intent"] == "off_topic":
                console.print(
                    "这个忙我帮不上——我是论文研究助手，只擅长检索论文、答疑库内文献和梳理研究方向。\n"
                    "不如告诉我你的研究领域，我帮你找几篇论文？"
                )
                continue
            if route["intent"] == "clarify":
                console.print(route.get("clarify_question") or "你是想检索新论文，还是想问库里已有的内容？")
                continue
            if route["intent"] == "explore":
                console.print(_suggest_directions(lib))
                continue
            if route["intent"] == "status":
                counts = lib.counts()
                total = sum(counts.values())
                stat_line = "  ".join(f"{s}: {n}" for s, n in sorted(counts.items()))
                console.print(f"文献库共 {total} 篇：{stat_line}")
                continue
            if route["intent"] == "search":
                s = route.get("search") or {}
                topics = s.get("topics") or [route["argument"] or text]
                run_params = dict(params)
                if s.get("year_from"):
                    run_params["year_from"] = s["year_from"]
                if s.get("max_results"):
                    run_params["max_per_source"] = s["max_results"]
                new_ids: list[str] = []
                notes: list[str] = []
                for topic in topics:
                    console.print(f"[dim]检索「{topic}」…[/dim]")
                    try:
                        _counts, n, ids = pipeline_search(lib, cfg, run_params, query=topic)
                    except Exception as exc:
                        console.print(f"{FAIL} 检索失败：{str(exc)[:120]}")
                        continue
                    notes.extend(n)
                    new_ids.extend(i for i in ids if i not in new_ids)
                for note in notes:
                    console.print(f"{WARN} {note}")
                if new_ids:
                    recent = [r for r in lib.list() if r.paper.source_id in new_ids]
                    _print_papers(recent, title=f"新入库 {len(new_ids)} 篇")
                    console.print("下一步：[cyan]pa run[/cyan] 可一键走完整流水线，或 [cyan]pa download <id>[/cyan] 下载。")
                else:
                    console.print("没有新入库的论文（可能都已存在，或各源不可用）。")
                continue
            # qa
            question = route["argument"] or text
            if store.count() == 0:
                console.print("向量库为空：先运行 [cyan]pa index --all[/cyan]。")
                continue
            try:
                answer, _hits = answer_question(question, retriever=retriever)
            except RuntimeError as exc:
                console.print(f"{FAIL} {exc}")
                continue
            console.print(f"[bold]答[/bold] {answer}\n")
    finally:
        lib.close()


def _suggest_directions(lib: Library, *, chat_fn=chat) -> str:
    """explore 引导：基于库内论文主题分布，让 LLM 归纳 2~3 个可探索方向。"""
    titles = [r.paper.title for r in lib.list() if r.paper.title]
    if len(titles) < 3:
        return (
            "文献库还比较空（少于 3 篇），先告诉我你的研究领域或课题关键词，"
            "我帮你检索一批论文进来（「帮我找 XX 的论文」），再基于它们给你梳理方向。"
        )
    sample = "\n".join(f"- {t}" for t in titles[:60])
    try:
        reply, _ = chat_fn(
            [{"role": "user", "content": (
                "以下是一个研究者文献库中的论文标题。请归纳出 2~3 个值得深入的研究方向，"
                "每个方向给一句话说明（为什么值得看、库内已有哪些相关论文）。只依据这些标题，"
                "用中文输出 Markdown 列表，不要其它内容：\n" + sample
            )}],
            max_tokens=800,
            temperature=0.3,
            thinking=False,  # 简单归纳任务禁用思考，防思考 tokens 吃光 max_tokens
        )
        text = reply.strip()
        return text if text else (
            f"库内已有 {len(titles)} 篇论文。直接告诉我你感兴趣的关键词，"
            "我帮你检索深入（「帮我找 XX 的论文」）。"
        )
    except Exception:
        # LLM 不可用时退化为纯统计：高频词方向提示
        from collections import Counter

        words = Counter()
        for t in titles:
            for w in re.findall(r"[A-Za-z]{4,}|[\u4e00-\u9fff]{2,6}", t):
                words[w.lower()] += 1
        top = "、".join(w for w, _ in words.most_common(8))
        return f"库内 {len(titles)} 篇论文的高频主题词：{top}。可以从这些方向深入，或直接告诉我感兴趣的关键词。"



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
    return paths.data_dir(cfg)


def _papers_dir(cfg: dict) -> Path:
    return paths.papers_dir(cfg)


def _parsed_dir(cfg: dict) -> Path:
    return paths.parsed_dir(cfg)


def _knowledge_dir(cfg: dict) -> Path:
    return paths.knowledge_dir(cfg)


def _library(cfg: dict) -> Library:
    return paths.library(cfg)


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
