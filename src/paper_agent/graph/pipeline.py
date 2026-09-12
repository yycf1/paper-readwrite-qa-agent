"""LangGraph 一键流水线：search → select → download → parse → analyze → index。

设计对应 PLAN §9 M4：
- 断点续跑：不引入重型 checkpoint——每个阶段围绕 library.db 的状态决定
  「跳过还是执行」，各阶段独立提交，任何一步中断后重跑即从断点继续；
- 无人值守选文：年份过滤 + 按引用数取 top-N + 下载预算上限（config.yaml
  的 pipeline 节，CLI 可覆盖）；
- 阶段逻辑写成普通函数（可脱离图直接测试/复用），图节点只做薄封装。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from paper_agent import paths
from paper_agent.config import load_config
from paper_agent.download import DownloadError, download_pdf, safe_dirname
from paper_agent.extraction.experiment_flow import extract_experiment
from paper_agent.extraction.report import generate_report
from paper_agent.library import Library
from paper_agent.llm import chat, parse_json_reply
from paper_agent.parsing import extract_markdown
from paper_agent.rag.embedder import embed
from paper_agent.rag.splitter import chunk_abstract, split_markdown
from paper_agent.rag.vectorstore import VectorStore
from paper_agent.sources import SourceUnavailable
from paper_agent.sources import arxiv as arxiv_src
from paper_agent.sources import europepmc as epmc_src
from paper_agent.sources import openalex as oa_src

SOURCE_MODULES = {"openalex": oa_src, "europepmc": epmc_src, "arxiv": arxiv_src}


class PipelineState(TypedDict):
    query: str
    params: dict  # max_per_source / top_n / year_from / download_budget
    stats: dict   # 各阶段计数
    notes: list[str]  # 降级与警告
    selected: list[str]  # 本轮选中的 source_id


def _default_params(cfg: dict) -> dict:
    p = cfg.get("pipeline", {}) or {}
    return {
        "max_per_source": 10,
        "top_n": p.get("top_n", 5),
        "year_from": p.get("year_from"),
        "download_budget": p.get("download_budget", 10),
    }


# ---------------------------------------------------------------------------
# 阶段 1：检索 + 无人值守选文
# ---------------------------------------------------------------------------


def run_search(
    lib: Library, cfg: dict, params: dict, *, query: str
) -> tuple[dict, list[str], list[str]]:
    """三源检索合并去重入库，返回 (计数, 警告, 本轮新入库 id 列表)。"""
    proxy = cfg.get("proxy", "") or ""
    source_cfg = cfg.get("sources", {}) or {}
    counts = {"searched": 0, "new": 0}
    notes: list[str] = []
    new_ids: list[str] = []
    for name, module in SOURCE_MODULES.items():
        if not (source_cfg.get(name) or {}).get("enabled", True):
            continue
        try:
            if name == "openalex":
                email = (source_cfg.get("openalex") or {}).get("email", "") or ""
                papers = module.search(query, params["max_per_source"], params.get("year_from"), proxy=proxy, email=email)
            else:
                papers = module.search(query, params["max_per_source"], params.get("year_from"), proxy=proxy)
        except SourceUnavailable as exc:
            notes.append(f"{name} 不可用，已降级跳过（{str(exc)[:60]}）")
            continue
        counts["searched"] += len(papers)
        for p in papers:
            if lib.find_duplicate(p):
                continue
            lib.upsert_paper(p)
            new_ids.append(p.source_id)
            counts["new"] += 1
    return counts, notes, new_ids


_SELECT_PROMPT = """你是论文选文助手。研究主题：{query}

候选论文（id | 标题 | 摘要截断）：
{papers}

为每篇论文打相关分 0-10（≥6 = 与主题相关）。只依据标题和摘要判断，不要编造。
只输出一个 JSON 对象：{{"scores": [{{"id": "论文id", "score": 7, "reason": "一句话理由"}}]}}
每篇论文都要有 reason。"""


def _relevance_scores(query: str, candidates: list, *, chat_fn) -> dict[str, tuple[int, str]]:
    """LLM 对候选论文打相关分，返回 {source_id: (score, reason)}；任何失败抛异常由上层降级。"""
    lines = []
    for r in candidates[:30]:
        abstract = (r.paper.abstract or "")[:200]
        lines.append(f"[{r.paper.source_id}] {r.paper.title} | {abstract}")
    reply, _usage = chat_fn(
        [{"role": "user", "content": _SELECT_PROMPT.format(query=query, papers="\n".join(lines))}],
        max_tokens=2000,
        temperature=0.0,
        thinking=False,  # 打分是简单任务，禁用思考防 tokens 被思考吃光
    )
    data = parse_json_reply(reply)
    known = {r.paper.source_id for r in candidates}
    scores: dict[str, tuple[int, str]] = {}
    for item in data.get("scores", []):
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or "")
        if sid not in known:
            continue
        score = item.get("score")
        if not isinstance(score, (int, float)):
            continue
        score = max(0, min(10, int(score)))
        reason = str(item.get("reason") or "").strip()
        scores[sid] = (score, reason)
    return scores


def select_papers(lib: Library, params: dict, *, query: str = "", chat_fn=chat) -> tuple[list[str], str]:
    """无人值守选文：相关性预筛（LLM 可用时）→ 按（引用数↓, 年份↓）取 top_n。

    返回 (选中的 source_id 列表, 一句话选文理由)。预筛失败/候选不足时退化为
    纯引用数排序——打不上分的候选保守保留，宁可多读不可漏读。
    不改变落选论文的状态（留在库里可手动处理）。
    """
    year_from = params.get("year_from")
    top_n = params["top_n"]
    candidates = [
        r for r in lib.list(status="discovered")
        if year_from is None or (r.paper.year or 0) >= year_from
    ]
    candidates.sort(key=lambda r: (r.paper.citations or 0, r.paper.year or 0), reverse=True)

    if len(candidates) <= top_n:
        return (
            [r.paper.source_id for r in candidates],
            f"候选 {len(candidates)} 篇未超过 top_n={top_n}，全部保留",
        )

    filtered_note = ""
    if query:
        try:
            scores = _relevance_scores(query, candidates, chat_fn=chat_fn)
            relevant = [r for r in candidates if scores.get(r.paper.source_id, (10, ""))[0] >= 6]
            if relevant:
                reasons = [
                    reason for r in relevant
                    if (reason := scores.get(r.paper.source_id, (0, ""))[1])
                ]
                example = "；".join(reasons[:2])
                filtered_note = (
                    f"相关性预筛 {len(candidates)}→{len(relevant)} 篇"
                    + (f"（如：{example}）" if example else "")
                )
                candidates = relevant
        except Exception:
            filtered_note = "相关性预筛不可用（LLM 失败），按引用数直接选"

    selected = [r.paper.source_id for r in candidates[:top_n]]
    reason = filtered_note or "按引用数直接选"
    reason += f"，按引用数取前 {top_n} 篇"
    return selected, reason


# ---------------------------------------------------------------------------
# 阶段 2~4：下载 / 解析 / 知识提取
# ---------------------------------------------------------------------------


def run_download(lib: Library, cfg: dict, params: dict, *, selected: list[str]) -> tuple[dict, list[str]]:
    """下载 selected 中 discovered 且有 OA 链接的论文（受预算限制）。

    预算结余回填：selected 全是「仅摘要」时预算会白白浪费——按引用数从
    库内其余 discovered 且有 OA 链接的论文继续下载，直到预算用完。
    """
    proxy = cfg.get("proxy", "") or ""
    notes: list[str] = []
    counts = {"downloaded": 0, "download_failed": 0, "abstract_only": 0}
    budget = params.get("download_budget") or 0

    def _try(rec) -> bool:
        """下载一篇；返回是否消耗了预算（无 OA 链接不消耗）。"""
        nonlocal budget
        if not rec.paper.pdf_urls:
            return False
        if budget <= 0:
            return False
        budget -= 1
        try:
            path = download_pdf(rec.paper, paths.papers_dir(cfg), proxy=proxy)
        except DownloadError as exc:
            lib.set_status(rec.paper.source_id, "download_failed", error=str(exc)[:500])
            counts["download_failed"] += 1
            return True
        lib.set_status(rec.paper.source_id, "downloaded", error=None, pdf_path=str(path))
        counts["downloaded"] += 1
        return True

    budget_exhausted = False
    for sid in selected:
        rec = lib.get(sid)
        if rec is None or rec.status != "discovered":
            continue  # 已下载/已处理：断点续跑跳过
        if not rec.paper.pdf_urls:
            counts["abstract_only"] += 1
            continue
        if not _try(rec):
            budget_exhausted = True
            break
    # 结余回填：其余 discovered 且有 OA 链接的，按引用数继续
    if budget > 0:
        remaining = [r for r in lib.list(status="discovered") if r.paper.pdf_urls and r.paper.source_id not in selected]
        remaining.sort(key=lambda r: (r.paper.citations or 0, r.paper.year or 0), reverse=True)
        filled = 0
        for rec in remaining:
            if budget <= 0:
                break
            if _try(rec):
                filled += 1
        if filled:
            notes.append(f"选中论文多为仅摘要，预算结余回填下载了 {filled} 篇其余高引论文")
    if budget_exhausted:
        notes.append("下载预算耗尽，剩余论文未尝试（可调大 pipeline.download_budget）")
    return counts, notes


def run_parse(lib: Library, cfg: dict) -> tuple[dict, list[str]]:
    """解析所有 downloaded 状态论文为分节 Markdown；失败回退 downloaded 并记 error。"""
    counts = {"parsed": 0, "parse_failed": 0}
    notes: list[str] = []
    for rec in lib.list(status="downloaded"):
        sid, p = rec.paper.source_id, rec.paper
        if not rec.pdf_path:
            notes.append(f"{sid} 缺少原文路径，跳过解析")
            counts["parse_failed"] += 1
            continue
        pdf_path = Path(rec.pdf_path)
        try:
            md = extract_markdown(pdf_path)
        except Exception as exc:
            lib.set_status(sid, "downloaded", error=f"解析失败：{exc}"[:500])
            counts["parse_failed"] += 1
            notes.append(f"{p.title[:36]}… 解析失败：{str(exc)[:60]}")
            continue
        min_chars = 30 if str(pdf_path).lower().endswith(".docx") else 200
        if len(md.strip()) < min_chars:
            lib.set_status(sid, "downloaded", error="解析结果近乎为空（可能是扫描版 PDF）")
            counts["parse_failed"] += 1
            continue
        out = paths.parsed_dir(cfg) / safe_dirname(sid) / "full_text.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        lib.set_status(sid, "parsed", error=None)
        counts["parsed"] += 1
    return counts, notes


def run_analyze(lib: Library, cfg: dict) -> tuple[dict, list[str]]:
    """知识提取：parsed → analyzed（experiment.json + report.md）。

    LLM 配置缺失（RuntimeError）属环境问题，直接放弃本阶段而不是逐篇撞墙。
    """
    counts = {"analyzed": 0, "analyze_failed": 0, "analyze_tokens": 0}
    notes: list[str] = []
    targets = lib.list(status="parsed")
    if not targets:
        return counts, notes
    try:
        chat([{"role": "user", "content": "OK"}], max_tokens=8)
    except RuntimeError as exc:
        return counts, [f"知识提取跳过：{exc}"]
    except Exception as exc:
        return counts, [f"知识提取跳过：LLM 平台不可达（{str(exc)[:80]}）"]
    for rec in targets:
        sid, p = rec.paper.source_id, rec.paper
        md_path = paths.parsed_dir(cfg) / safe_dirname(sid) / "full_text.md"
        if not md_path.exists():
            counts["analyze_failed"] += 1
            notes.append(f"{sid} 缺少解析文本")
            continue
        try:
            markdown = md_path.read_text(encoding="utf-8")
            experiment = extract_experiment(markdown, chat_fn=chat)
            report, report_usage = generate_report(markdown, chat_fn=chat)
        except RuntimeError as exc:
            return counts, notes + [f"知识提取中止：{exc}"]
        except Exception as exc:
            lib.set_status(sid, "parsed", error=f"分析失败：{exc}"[:500])
            counts["analyze_failed"] += 1
            notes.append(f"{p.title[:36]}… 分析失败：{str(exc)[:60]}")
            continue
        out_dir = paths.knowledge_dir(cfg) / sid.replace(":", "_").replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "experiment.json").write_text(
            json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "report.md").write_text(report, encoding="utf-8")
        tokens = (
            experiment["token_usage"]["prompt"] + experiment["token_usage"]["completion"]
            + report_usage["prompt"] + report_usage["completion"]
        )
        counts["analyzed"] += 1
        counts["analyze_tokens"] += tokens
        lib.set_status(sid, "analyzed", error=None)
    return counts, notes


# ---------------------------------------------------------------------------
# 阶段 5：向量索引
# ---------------------------------------------------------------------------


def run_index(lib: Library, cfg: dict, *, selected: list[str]) -> tuple[dict, list[str]]:
    """全文索引 parsed/analyzed 状态论文；本轮选中但无全文的论文补摘要块。"""
    counts = {"indexed": 0, "index_chunks": 0, "abstract_indexed": 0}
    notes: list[str] = []
    store = VectorStore(paths.vector_dir(cfg))
    existing = store.paper_modes()
    for rec in lib.list(status="parsed") + lib.list(status="analyzed"):
        sid, p = rec.paper.source_id, rec.paper
        if existing.get(sid) == "fulltext":
            continue
        md_path = paths.parsed_dir(cfg) / safe_dirname(sid) / "full_text.md"
        if not md_path.exists():
            continue
        chunks = split_markdown(md_path.read_text(encoding="utf-8"), sid, p.title)
        if not chunks:
            notes.append(f"{sid} 分块为空，跳过")
            continue
        vectors = embed([c.text for c in chunks])
        metas = [{**c.metadata, "mode": "fulltext"} for c in chunks]
        store.upsert_paper_chunks(sid, [c.text for c in chunks], metas, vectors)
        lib.set_status(sid, "indexed", error=None)
        counts["indexed"] += 1
        counts["index_chunks"] += len(chunks)
    # 本轮选中但无全文的：摘要单块入库，不改状态机
    for sid in selected:
        rec = lib.get(sid)
        if rec is None or rec.status != "discovered" or existing.get(sid) == "abstract":
            continue
        if not rec.paper.abstract.strip():
            continue
        chunks = chunk_abstract(sid, rec.paper.title, rec.paper.abstract)
        vectors = embed([c.text for c in chunks])
        metas = [{**c.metadata, "mode": "abstract"} for c in chunks]
        store.upsert_paper_chunks(sid, [c.text for c in chunks], metas, vectors)
        counts["abstract_indexed"] += 1
    return counts, notes


# ---------------------------------------------------------------------------
# 图装配：节点薄封装（各自开关 Library，阶段间以 library.db 状态衔接）
# ---------------------------------------------------------------------------


def build_pipeline_graph():
    graph = StateGraph(PipelineState)

    def _node(stage):
        def _fn(state: PipelineState) -> dict:
            cfg = load_config()
            lib = paths.library(cfg)
            try:
                if stage == "search":
                    counts, notes, _new_ids = run_search(lib, cfg, state["params"], query=state["query"])
                    selected, reason = select_papers(
                        lib, {**_default_params(cfg), **state["params"]}, query=state["query"],
                    )
                    return {
                        "stats": {**state["stats"], **counts, "selected_n": len(selected)},
                        "notes": state["notes"] + notes + [f"选文：{reason}"],
                        "selected": selected,
                    }
                fn = {
                    "download": lambda: run_download(lib, cfg, state["params"], selected=state["selected"]),
                    "parse": lambda: run_parse(lib, cfg),
                    "analyze": lambda: run_analyze(lib, cfg),
                    "index": lambda: run_index(lib, cfg, selected=state["selected"]),
                }[stage]
                counts, notes = fn()
                return {"stats": {**state["stats"], **counts}, "notes": state["notes"] + notes}
            finally:
                lib.close()

        return _fn

    for stage in ("search", "download", "parse", "analyze", "index"):
        graph.add_node(stage, _node(stage))
    graph.add_edge(START, "search")
    for a, b in zip(("search", "download", "parse", "analyze"), ("download", "parse", "analyze", "index")):
        graph.add_edge(a, b)
    graph.add_edge("index", END)
    return graph.compile()


def run_pipeline(query: str, params: dict | None = None) -> PipelineState:
    """执行全链路；params 覆盖 config.yaml 的 pipeline 节。"""
    cfg = load_config()
    merged = {**_default_params(cfg), **(params or {})}
    return build_pipeline_graph().invoke(
        {"query": query, "params": merged, "stats": {}, "notes": [], "selected": []}
    )
