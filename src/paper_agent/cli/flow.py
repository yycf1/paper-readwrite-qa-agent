"""流水线与交互命令：run / chat / status。"""

from __future__ import annotations

import typer
from rich.table import Table

from paper_agent import paths
from paper_agent.cli.core import WARN, _library, _print_papers, app, console
from paper_agent.config import load_config
from paper_agent.graph.pipeline import run_pipeline, run_search as pipeline_search
from paper_agent.rag.qa import Retriever
from paper_agent.rag.vectorstore import VectorStore
from paper_agent.services.assistant import handle_message


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
         f"选中 top {stats.get('selected_n', 0)}"),
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
            reply = handle_message(text, lib=lib, retriever=retriever, params=params, cfg=cfg)
            if reply.understanding:
                console.print(f"[dim]（{reply.understanding}）[/dim]")
            if reply.intent == "exit":
                break
            for note in reply.notes:
                console.print(f"{WARN} {note}")
            console.print(reply.reply)
            if reply.intent == "search" and reply.new_papers:
                _print_papers(reply.new_papers, title=f"新入库 {len(reply.new_papers)} 篇")
                console.print("下一步：[cyan]pa run[/cyan] 可一键走完整流水线，或 [cyan]pa download <id>[/cyan] 下载。")
            if reply.intent == "qa":
                console.print()
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
