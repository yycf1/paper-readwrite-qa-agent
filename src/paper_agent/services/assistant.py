"""对话助手服务：意图路由 → 分发执行，返回结构化回复。

对应 CLI `pa chat` REPL 与 Web `POST /api/chat`。单轮处理独立成函数后，
CLI 负责循环与渲染，Web 负责会话与传输，两者行为保持一致。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from paper_agent.config import load_config
from paper_agent.graph.route import classify_intent, summarize_query_understanding
from paper_agent.library import Library
from paper_agent.llm import chat
from paper_agent.rag.qa import Retriever, answer_question
from paper_agent.services.search import search_and_ingest

HELP_TEXT = (
    "我支持：①检索新论文（「帮我找…主题…的论文」，可带年份/数量）"
    "②问答已入库文献（直接提问）③查看库状态（「看看状态」）"
    "④方向推荐（「不知道看什么论文」）。也可以直接用 pa 命令。"
)

OFF_TOPIC_TEXT = (
    "这个忙我帮不上——我是论文研究助手，只擅长检索论文、答疑库内文献和梳理研究方向。\n"
    "不如告诉我你的研究领域，我帮你找几篇论文？"
)

CLARIFY_FALLBACK = "你是想检索新论文，还是想问库里已有的内容？"


@dataclass
class AssistantReply:
    """一轮对话的结果：intent 供入口做后续动作（如 CLI 渲染表格、Web 联动刷新）。"""

    intent: str
    reply: str
    understanding: str = ""
    notes: list[str] = field(default_factory=list)
    new_papers: list = field(default_factory=list)  # search 意图的新入库论文


def handle_message(
    text: str,
    *,
    lib: Library,
    retriever: Retriever,
    params: dict,
    cfg: dict | None = None,
    chat_fn=chat,
    history: list[tuple[str, str]] | None = None,
) -> AssistantReply:
    """处理一条用户消息：分类意图并执行，永不抛异常（错误转成 reply）。

    history 为本会话近几轮 (问题, 回答)，供问答链做指代消解（「它用了什么数据集」）。
    """
    cfg = cfg or load_config()
    route = classify_intent(text, chat_fn=chat_fn)
    understanding = summarize_query_understanding(route)
    intent = route["intent"]

    if intent == "exit":
        return AssistantReply("exit", "再见，祝科研顺利～", understanding=understanding)
    if intent == "help":
        return AssistantReply("help", HELP_TEXT, understanding=understanding)
    if intent == "off_topic":
        return AssistantReply("off_topic", OFF_TOPIC_TEXT, understanding=understanding)
    if intent == "clarify":
        return AssistantReply(
            "clarify", route.get("clarify_question") or CLARIFY_FALLBACK, understanding=understanding
        )
    if intent == "explore":
        return AssistantReply("explore", suggest_directions(lib, chat_fn=chat_fn), understanding=understanding)
    if intent == "status":
        counts = lib.counts()
        total = sum(counts.values())
        stat_line = "  ".join(f"{s}: {n}" for s, n in sorted(counts.items()))
        return AssistantReply("status", f"文献库共 {total} 篇：{stat_line}", understanding=understanding)
    if intent == "search":
        return _do_search(text, route, lib=lib, cfg=cfg, params=params, understanding=understanding)

    # qa：意图分类兜底到这里
    question = route["argument"] or text
    if retriever.store.count() == 0:
        return AssistantReply("qa", "向量库为空：先运行 `pa index --all` 建立索引。", understanding=understanding)
    try:
        answer, _hits = answer_question(question, retriever=retriever, history=history)
    except RuntimeError as exc:  # 缺 Key 等配置问题
        return AssistantReply("qa", f"问答失败：{exc}", understanding=understanding)
    return AssistantReply("qa", answer, understanding=understanding)


def _do_search(
    text: str,
    route: dict,
    *,
    lib: Library,
    cfg: dict,
    params: dict,
    understanding: str,
) -> AssistantReply:
    s = route.get("search") or {}
    classifier_topics = [t for t in (s.get("topics") or []) if isinstance(t, str) and t.strip()]
    topics = classifier_topics or [route["argument"] or text]
    run_params = dict(params)
    if s.get("year_from"):
        run_params["year_from"] = s["year_from"]
    if s.get("max_results"):
        run_params["max_per_source"] = s["max_results"]

    new_papers, notes = [], []
    seen_ids: set[str] = set()
    total_hits = 0
    for topic in topics:
        try:
            outcome = search_and_ingest(
                topic,
                lib=lib,
                cfg=cfg,
                max_results=run_params.get("max_per_source", 10),
                year_from=run_params.get("year_from"),
                # 意图分类已产出英文子查询时跳过检索链的重复优化（省一次 LLM 调用）
                raw=bool(classifier_topics),
            )
        except Exception as exc:
            notes.append(f"检索「{topic}」失败：{str(exc)[:120]}")
            continue
        notes.extend(outcome.notes)
        total_hits += outcome.total_hits
        for rec in outcome.new_papers:
            if rec.paper.source_id not in seen_ids:
                seen_ids.add(rec.paper.source_id)
                new_papers.append(rec)

    if new_papers:
        reply = f"本轮检索到 {total_hits} 条候选，新入库 {len(new_papers)} 篇（其余与库内已有论文重复，未重复添加）。"
    elif total_hits:
        reply = f"本轮检索到 {total_hits} 条候选，但都已存在于文献库中，没有新入库。"
    else:
        reply = "没有检索到结果：可以换个说法、放宽条件，或稍后重试（数据源偶尔限流）。"
    return AssistantReply("search", reply, understanding=understanding, notes=notes, new_papers=new_papers)


def suggest_directions(lib: Library, *, chat_fn=chat) -> str:
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
        words = Counter()
        for t in titles:
            for w in re.findall(r"[A-Za-z]{4,}|[\u4e00-\u9fff]{2,6}", t):
                words[w.lower()] += 1
        top = "、".join(w for w, _ in words.most_common(8))
        return f"库内 {len(titles)} 篇论文的高频主题词：{top}。可以从这些方向深入，或直接告诉我感兴趣的关键词。"
