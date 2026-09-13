"""检索编排服务：查询优化 → 多源检索 → 结果校验 → 跨源去重 → 入库。

对应 CLI `pa search` 与 Web `POST /api/search`。过程事件（源降级、校验
拦截、回退重搜）统一进 outcome.notes，由各入口自行渲染。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from paper_agent.graph.route import optimize_query
from paper_agent.library import Library, PaperRecord
from paper_agent.models import normalize_title
from paper_agent.tools import SOURCE_MODULES, filter_valid_papers, search_source_tool


@dataclass
class SearchOutcome:
    """一次检索入库的结果。new_papers 只含本轮新入库条目（去重合并的不算）。"""

    new_papers: list[PaperRecord] = field(default_factory=list)
    total_hits: int = 0
    notes: list[str] = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    source_hits: dict[str, int] = field(default_factory=dict)  # 源名 → 候选数（跨子查询累计）


def search_and_ingest(
    query: str,
    *,
    lib: Library,
    cfg: dict,
    sources: str | list[str] = "openalex,europepmc,arxiv",
    max_results: int = 10,
    year_from: int | None = None,
    raw: bool = False,
) -> SearchOutcome:
    """执行完整检索入库链，返回 SearchOutcome。未知数据源抛 ValueError。"""
    if isinstance(sources, str):
        names = [s.strip().lower() for s in sources.split(",")]
    else:
        names = [s.strip().lower() for s in sources]
    names = [n for n in names if n]
    unknown = [n for n in names if n not in SOURCE_MODULES]
    if unknown:
        raise ValueError(f"未知数据源：{', '.join(unknown)}（可用：{', '.join(SOURCE_MODULES)}）")

    proxy = cfg.get("proxy", "") or ""
    source_cfg = cfg.get("sources", {}) or {}

    if raw:
        plan = {"topics": [query], "year_from": year_from, "max_results": None, "optimized": False}
    else:
        plan = optimize_query(query)
    eff_year = year_from or plan.get("year_from")
    eff_max = plan.get("max_results") or max_results

    outcome = SearchOutcome(plan=plan)
    seen: dict[str, str] = {}  # 去重键（doi 或归一化标题）→ source_id

    def _search_one(keyword: str) -> int:
        """对单个查询词跑全部源并去重入库，返回各源返回的候选总数。"""
        hits = 0
        for name in names:
            if not (source_cfg.get(name) or {}).get("enabled", True):
                outcome.notes.append(f"{name}：已在 config.yaml 停用，跳过")
                continue
            email = (source_cfg.get(name) or {}).get("email", "") or ""
            result = search_source_tool(
                name, query=keyword, max_results=eff_max, year_from=eff_year,
                proxy=proxy, email=email,
            )
            if result.status == "fatal":
                outcome.notes.append(f"{name}：{result.error}")
                continue
            if result.status != "ok":
                hint = "可在 config.yaml 配置 proxy" if name == "arxiv" else "请检查网络"
                outcome.notes.append(f"{name} 不可用（{result.error}），已跳过——{hint}")
                continue
            papers, issues = filter_valid_papers(result.data)
            outcome.notes.extend(f"结果校验拦截：{issue}" for issue in issues)
            hits += len(papers)
            outcome.source_hits[name] = outcome.source_hits.get(name, 0) + len(papers)
            for p in papers:
                dup_id = _duplicate_of(lib, seen, p)
                if dup_id:
                    lib.upsert_paper(p)  # 合并补充元数据（状态不变）
                    continue
                lib.upsert_paper(p)
                rec = lib.get(p.source_id)
                if rec:
                    outcome.new_papers.append(rec)
                if p.doi:
                    seen[f"doi:{p.doi}"] = p.source_id
                seen[f"t:{normalize_title(p.title)}"] = p.source_id
        return hits

    for topic in plan["topics"]:
        outcome.total_hits += _search_one(topic)

    # 优化后的检索词全军覆没 → 回退原始词重搜一轮
    if outcome.total_hits == 0 and plan.get("optimized"):
        outcome.notes.append("优化后的检索词没有结果，回退原始词重搜")
        outcome.total_hits = _search_one(query)
    return outcome


def _duplicate_of(lib: Library, seen: dict[str, str], p) -> str | None:
    """先用本次会话的内存索引去重，再查账本（跨命令重复检索也不重复入库）。"""
    if p.doi and seen.get(f"doi:{p.doi}"):
        return seen[f"doi:{p.doi}"]
    tkey = f"t:{normalize_title(p.title)}"
    if seen.get(tkey):
        return seen[tkey]
    return lib.find_duplicate(p)
