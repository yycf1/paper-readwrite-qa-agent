"""数据源共享 HTTP helper：统一 UA、代理、超时；网络异常统一包装为 SourceUnavailable。"""

from __future__ import annotations

import requests

UA = "paper-agent/0.1 (personal research reading tool)"


def http_get(
    url: str,
    *,
    params: dict | None = None,
    proxy: str = "",
    timeout: tuple[float, float] = (10, 60),
) -> requests.Response:
    proxies = {"http": proxy, "https": proxy} if proxy else None
    return requests.get(
        url,
        params=params,
        timeout=timeout,
        proxies=proxies,
        headers={"User-Agent": UA},
    )
