"""OA PDF 下载：多候选地址依次尝试，%PDF 魔数校验，落盘 data/papers/{id}/paper.pdf。"""

from __future__ import annotations

from pathlib import Path

import requests

from paper_agent.models import Paper

_UA = "paper-agent/0.1 (personal research reading tool)"


class DownloadError(Exception):
    """所有候选地址均下载失败。"""


def safe_dirname(source_id: str) -> str:
    """source_id 转目录名：Windows 不允许 ':'，统一替换为 '_'。"""
    return source_id.replace(":", "_").replace("/", "_")


def download_pdf(paper: Paper, papers_dir: Path, *, proxy: str = "") -> Path:
    """依次尝试 paper.pdf_urls；内容校验通过才落盘。全部失败抛 DownloadError。

    校验规则：前 64KB 含 %PDF 魔数，或 Content-Type 声明为 pdf——
    防止把 HTML 落地页/验证页误存成 .pdf。
    """
    if not paper.pdf_urls:
        raise DownloadError("无可用的 PDF 链接（仅摘要论文）")
    dest_dir = papers_dir / safe_dirname(paper.source_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "paper.pdf"
    tmp = dest_dir / "paper.pdf.part"
    proxies = {"http": proxy, "https": proxy} if proxy else None
    errors: list[str] = []

    for url in paper.pdf_urls:
        try:
            with requests.get(
                url,
                stream=True,
                timeout=(10, 120),
                proxies=proxies,
                headers={"User-Agent": _UA},
            ) as resp:
                if resp.status_code != 200:
                    errors.append(f"{url} -> HTTP {resp.status_code}")
                    continue
                chunks = resp.iter_content(chunk_size=65536)
                head = next(chunks, b"")
                content_type = resp.headers.get("content-type", "").lower()
                if b"%PDF" not in head and "pdf" not in content_type:
                    errors.append(f"{url} -> 非 PDF 内容（{content_type or '未知类型'}）")
                    continue
                with tmp.open("wb") as f:
                    f.write(head)
                    for chunk in chunks:
                        f.write(chunk)
                tmp.replace(out)
                return out
        except requests.RequestException as exc:
            errors.append(f"{url} -> {exc}")
            continue
    raise DownloadError("; ".join(errors))
