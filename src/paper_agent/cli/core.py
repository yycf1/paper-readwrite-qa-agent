"""CLI 共享壳：Typer 应用、Rich 控制台与渲染辅助。

命令按领域拆在 ingest / qa / flow / admin 子模块，导入时向这里的 app 注册；
业务编排优先下沉 services/，本层只做「解析参数 → 调服务/模块 → 渲染」。
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from paper_agent import paths
from paper_agent.library import Library, PaperRecord

app = typer.Typer(
    help="paper-agent：文献阅读复现 Agent（检索→下载→解析→实验知识提取→RAG问答）",
    no_args_is_help=True,
)
console = Console()

OK = "[green]✅[/green]"
FAIL = "[red]❌[/red]"
SKIP = "[yellow]⏭ 未检查[/yellow]"
WARN = "[yellow]⚠[/yellow]"


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
