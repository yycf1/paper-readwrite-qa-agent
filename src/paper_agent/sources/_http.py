"""数据源共享 HTTP helper：统一 UA、代理、超时、自动重试；网络异常统一包装为 SourceUnavailable。

本机网络对国际站点时好时坏（实测 OpenAlex 偶发读超时），因此这里对
连接类失败自动重试一次；限流（429）是正常响应，不受影响。
"""

from __future__ import annotations

import time

import requests

from paper_agent.sources import SourceUnavailable

UA = "paper-agent/0.1 (personal research reading tool)"


def http_get(
    url: str,
    *,
    params: dict | None = None,
    proxy: str = "",
    timeout: tuple[float, float] = (10, 60),
    retries: int = 1,
) -> requests.Response:
    proxies = {"http": proxy, "https": proxy} if proxy else None
    last_exc: requests.RequestException | None = None
    for attempt in range(retries + 1):
        try:
            return requests.get(
                url,
                params=params,
                timeout=timeout,
                proxies=proxies,
                headers={"User-Agent": UA},
            )
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(2)
    raise SourceUnavailable(f"请求失败（含 {retries} 次重试）：{last_exc}") from last_exc
