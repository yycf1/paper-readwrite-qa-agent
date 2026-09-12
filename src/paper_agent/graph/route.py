"""chat 意图理解：一次 LLM 调用完成 意图判定 + 搜索参数抽取 + 查询优化。

意图集合与边界（PLAN.md M5）：
- search/qa/status/exit/help：原有能力；
- explore：用户不清楚看什么，基于库内主题分布引导；
- off_topic：与科研/论文完全无关的请求，拒绝并把话题引回；
- clarify：需求模糊或意图不明，向用户反问而非瞎猜。

边界标准：与科研/论文/文献/技术概念相关的一律接待（库外问题由 qa 链诚实兜底），
仅完全无关才判 off_topic；启发式兜底路径保守接待（不判 off_topic），宁可错接不可误拒。
LLM 分类失败（缺 Key / 网络不可达）时退化为关键词启发式，保证 REPL 永远可用。
"""

from __future__ import annotations

import json

from paper_agent.llm import chat, parse_json_reply

_INTENTS = ("search", "qa", "status", "explore", "off_topic", "clarify", "exit", "help")

_PROMPT = """你是论文研究助手 paper-agent 的意图理解器。分析用户输入，只输出一个 JSON 对象：
{{
  "intent": "search|qa|status|explore|off_topic|clarify|exit|help",
  "argument": "主题词或问题原文",
  "topics": ["英文学术子查询", "..."],
  "year_from": 2023,
  "max_results": 20,
  "clarify_question": "向用户确认的问题"
}}

意图定义：
- "search"：想检索/查找/下载新论文。argument=核心主题；topics=适合英文学术库的子查询
  （中文翻译成英文、缩写展开成全称、口语换成术语；复合需求拆成最多 3 个子查询）；
  year_from/max_results 只在用户明确提出时填写（整数），否则填 null；
- "qa"：就已入库论文内容提问（论文相关的概念解释也算）。argument=问题原文；
- "status"：查看文献库状态、进度；
- "explore"：用户不清楚看什么/求推荐研究方向（「不知道看什么论文」「推荐个方向」）；
- "off_topic"：与科研/论文/文献完全无关的请求（闲聊、写周报、写代码、生活问题）；
- "clarify"：无法判断用户是想搜新论文还是问库里内容，或需求太模糊。clarify_question=向用户确认的问题；
- "exit"：告别、退出；"help"：问你能做什么、怎么用。

边界标准：与科研/论文/文献/技术概念相关的一律接待，不要判 off_topic。

用户输入：{text}"""


def classify_intent(text: str, *, chat_fn=chat) -> dict:
    """意图理解；LLM 失败时启发式兜底。

    返回 {"intent", "argument", "via"}，search 时额外带
    "search": {"topics": [...], "year_from": int|None, "max_results": int|None}，
    clarify 时额外带 "clarify_question"。
    """
    stripped = text.strip()
    if not stripped:
        return {"intent": "help", "argument": "", "via": "heuristic"}
    try:
        reply, _usage = chat_fn(
            [{"role": "user", "content": _PROMPT.format(text=stripped)}],
            max_tokens=400,
            temperature=0.0,
        )
        data = parse_json_reply(reply)
        intent = data.get("intent")
        if intent in _INTENTS:
            route: dict = {
                "intent": intent,
                "argument": str(data.get("argument") or "").strip() or stripped,
                "via": "llm",
            }
            if intent == "search":
                route["search"] = _extract_search_params(data)
            if intent == "clarify":
                route["clarify_question"] = str(data.get("clarify_question") or "").strip()
            return route
    except Exception:
        pass
    return {"intent": _heuristic_intent(stripped), "argument": stripped, "via": "heuristic"}


def _extract_search_params(data: dict) -> dict:
    """从 LLM 输出中抽取结构化搜索参数，逐项容错（宁缺勿错）。"""
    topics = data.get("topics")
    if isinstance(topics, list):
        # 只收非空字符串：数字/嵌套对象 str() 后是无意义的搜索词
        clean = [t.strip() for t in topics if isinstance(t, str) and t.strip()]
    else:
        clean = []
    year = data.get("year_from")
    max_r = data.get("max_results")
    return {
        "topics": clean,
        "year_from": year if isinstance(year, int) and 1900 <= year <= 2100 else None,
        "max_results": max_r if isinstance(max_r, int) and 1 <= max_r <= 50 else None,
    }


def _heuristic_intent(text: str) -> str:
    """无 LLM 时的降级路由：问号/疑问词 → qa；状态词 → status；退出词 → exit。

    保守原则：兜底路径不判 off_topic（宁可错接，不可误拒）。
    """
    low = text.lower()
    if any(w in low for w in ("退出", "再见", "bye", "exit", "quit", "q")):
        return "exit"
    if any(w in low for w in ("状态", "进度", "status", "库里有")):
        return "status"
    if any(w in low for w in ("不知道看什么", "推荐方向", "看什么论文", "研究方向")):
        return "explore"
    if any(w in low for w in ("检索", "搜论文", "找论文", "下载", "search", "帮我找")):
        return "search"
    if any(w in low for w in (" help", "帮助", "你能做什么", "怎么用")) or low.startswith("help"):
        return "help"
    return "qa"  # 缺省当提问


def summarize_query_understanding(route: dict) -> str:
    """把意图理解结果渲染成一行用户可读的说明（chat 界面回显用）。"""
    intent = route["intent"]
    via = "LLM" if route["via"] == "llm" else "启发式"
    if intent == "search" and route.get("search"):
        s = route["search"]
        parts = [f"主题：{' / '.join(s['topics']) or route['argument']}"]
        if s["year_from"]:
            parts.append(f"{s['year_from']} 年起")
        if s["max_results"]:
            parts.append(f"每源≤{s['max_results']} 篇")
        return f"检索（{via}）—" + "，".join(parts)
    labels = {
        "qa": "问答", "status": "库状态", "explore": "方向引导",
        "off_topic": "域外请求", "clarify": "需要澄清",
        "exit": "退出", "help": "帮助", "search": "检索",
    }
    return f"{labels.get(intent, intent)}（{via}）"


# ---------------------------------------------------------------------------
# 查询优化（pa search 显式命令用）：把用户输入规范化为学术库搜索词
# ---------------------------------------------------------------------------

_OPTIMIZE_PROMPT = """把用户的论文检索需求规范化为适合英文学术数据库（OpenAlex 等）的搜索词。
要求：中文翻译成英文学术表述；缩写展开成全称（如 GNN→graph neural network）；
口语换成术语；复合需求拆成最多 3 个子查询。
只输出一个 JSON 对象（不要其它内容）：
{{"topics": ["子查询1", "子查询2"], "year_from": null, "max_results": null}}
year_from / max_results 只在用户明确提出时填整数，否则填 null。

用户输入：{text}"""


def optimize_query(text: str, *, chat_fn=chat) -> dict:
    """查询优化；任何失败都降级为「原样直搜」，绝不阻塞检索。"""
    try:
        reply, _usage = chat_fn(
            [{"role": "user", "content": _OPTIMIZE_PROMPT.format(text=text.strip())}],
            max_tokens=200,
            temperature=0.0,
        )
        data = parse_json_reply(reply)
        params = _extract_search_params(data)
        if params["topics"]:
            params["optimized"] = True
            return params
    except Exception:
        pass
    return {"topics": [text.strip()], "year_from": None, "max_results": None, "optimized": False}
