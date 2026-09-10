"""chat 意图路由：把用户一句自然语言分派到 检索 / 问答 / 状态 / 退出。

LLM 分类失败（缺 Key / 网络不可达）时退化为关键词启发式，保证 REPL 永远可用。
"""

from __future__ import annotations

import json

from paper_agent.llm import chat, parse_json_reply

_INTENTS = ("search", "qa", "status", "exit", "help")

_PROMPT = """你是论文助手的意图分类器。把用户输入分类为以下意图之一，输出 JSON 对象（不要输出其它内容）：
{{"intent": "意图", "argument": "参数"}}

意图定义：
- "search"：用户想检索/查找/下载某主题的新论文。argument = 检索主题词；
- "qa"：用户在提问，想了解已入库论文的内容。argument = 原问题；
- "status"：查看文献库状态、进度；
- "exit"：告别、退出；
- "help"：问你能做什么、怎么用。

用户输入：{text}"""


def classify_intent(text: str, *, chat_fn=chat) -> dict:
    """LLM 意图分类；失败时启发式兜底。返回 {"intent", "argument", "via"}。"""
    stripped = text.strip()
    if not stripped:
        return {"intent": "help", "argument": "", "via": "heuristic"}
    try:
        reply, _usage = chat_fn(
            [{"role": "user", "content": _PROMPT.format(text=stripped)}],
            max_tokens=120,
            temperature=0.0,
        )
        data = parse_json_reply(reply)
        intent = data.get("intent")
        if intent in _INTENTS:
            return {
                "intent": intent,
                "argument": str(data.get("argument") or "").strip(),
                "via": "llm",
            }
    except Exception:
        pass
    return {"intent": _heuristic_intent(stripped), "argument": stripped, "via": "heuristic"}


def _heuristic_intent(text: str) -> str:
    """无 LLM 时的降级路由：问号/疑问词 → qa；状态词 → status；退出词 → exit。"""
    low = text.lower()
    if any(w in low for w in ("退出", "再见", "bye", "exit", "quit", "q")):
        return "exit"
    if any(w in low for w in ("状态", "进度", "status", "库里有")):
        return "status"
    if any(w in low for w in ("检索", "搜论文", "找论文", "下载", "search", "帮我找")):
        return "search"
    if any(w in low for w in (" help", "帮助", "你能做什么", "怎么用")) or low.startswith("help"):
        return "help"
    return "qa"  # 缺省当提问
