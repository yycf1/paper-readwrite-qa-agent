"""paper-agent 命令行入口（Typer）。"""

from __future__ import annotations

import time

import typer
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from paper_agent import __version__
from paper_agent.config import EndpointConfig, load_config, resolve_endpoint
from paper_agent.graph.hello import run_hello

app = typer.Typer(
    help="paper-agent：文献阅读复现 Agent（检索→下载→解析→实验知识提取→RAG问答）",
    no_args_is_help=True,
)
console = Console()

OK = "[green]✅[/green]"
FAIL = "[red]❌[/red]"
SKIP = "[yellow]⏭ 未检查[/yellow]"


@app.command()
def version() -> None:
    """显示版本。"""
    console.print(f"paper-agent {__version__}")


@app.command()
def doctor() -> None:
    """M0 验收自检：配置 → API Key → 平台连通（embedding + LLM）→ LangGraph 图执行。"""
    console.print("[bold]paper-agent 环境自检[/bold]\n")
    table = Table(show_header=True, header_style="bold")
    table.add_column("检查项", style="cyan")
    table.add_column("状态")
    table.add_column("详情", overflow="fold")

    # 1. 配置文件
    try:
        cfg = load_config()
        table.add_row("配置文件 config.yaml", OK, "加载成功")
    except Exception as exc:
        table.add_row("配置文件 config.yaml", FAIL, str(exc))
        console.print(table)
        raise typer.Exit(code=1) from exc

    # 2/3. embedding 与 LLM 连通性（无 Key 时跳过并给出指引）
    platform_ok = _check_endpoint(table, "Embedding 平台", resolve_endpoint(cfg["embedding"]), is_embedding=True)
    platform_ok &= _check_endpoint(table, "LLM 平台", resolve_endpoint(cfg["llm"]), is_embedding=False)

    # 4. LangGraph hello world（不依赖外部服务）
    try:
        result = run_hello()
        table.add_row("LangGraph 图执行", OK, result["message"])
        graph_ok = True
    except Exception as exc:
        table.add_row("LangGraph 图执行", FAIL, str(exc))
        graph_ok = False

    console.print(table)
    if platform_ok and graph_ok:
        console.print("\n[bold green]结论：平台连通 + 图执行成功，M0 验收通过。[/bold green]")
    elif graph_ok:
        console.print(
            "\n[bold yellow]图执行正常，但平台尚未连通，还差 API Key：[/bold yellow]\n"
            "1) 复制 .env.example 为 .env；\n"
            "2) 填入 Key（硅基流动 https://cloud.siliconflow.cn ｜ 智谱 https://open.bigmodel.cn）；\n"
            "3) 重新运行 [cyan]pa doctor[/cyan]。"
        )
        raise typer.Exit(code=1)
    else:
        console.print("\n[bold red]图执行失败，请把上方错误信息反馈给开发流程排查。[/bold red]")
        raise typer.Exit(code=1)


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


if __name__ == "__main__":
    app()
