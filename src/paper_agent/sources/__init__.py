"""数据源 client（OpenAlex / Europe PMC / arXiv）。

约定：
- 每个源模块提供 `search(query, max_results, year_from, proxy) -> list[Paper]` 与 `probe(proxy) -> str`；
- 源不可达（网络屏蔽/限流/HTTP 非 200）统一抛 `SourceUnavailable`，由上层捕获降级，不中断其他源；
- 解析逻辑抽成纯函数（parse_*），便于离线单测。
"""

from __future__ import annotations


class SourceUnavailable(Exception):
    """数据源不可达（网络屏蔽/限流/服务异常）。上层捕获后降级提示。"""
