"""问答与评测命令：ask / eval。"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from paper_agent.cli.core import FAIL, WARN, _data_dir, _library, app, console
from paper_agent.config import PROJECT_ROOT, load_config
from paper_agent.rag.eval import judge_answer, judge_retrieval, load_evalset as evalset_load
from paper_agent.rag.qa import Retriever, answer_question
from paper_agent.rag.vectorstore import VectorStore


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
