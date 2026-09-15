"""文献库管理操作：本地文件导入与论文删除（CLI 与 Web 共用）。

删除是「全链清理」：账本记录、向量库块、解析/知识/原文产物一并移除——
留一半状态会让下次检索/问答出现幽灵数据。
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

from paper_agent import paths
from paper_agent.download import safe_dirname
from paper_agent.library import Library, PaperRecord
from paper_agent.parsing import SUPPORTED_SUFFIXES
from paper_agent.rag.vectorstore import VectorStore


class UnsupportedFormat(ValueError):
    pass


def import_local_file(path: Path, *, cfg: dict, lib: Library, tag: str = "本地上传") -> tuple[PaperRecord, bool]:
    """导入本地 PDF/DOCX：内容哈希幂等，状态直达 downloaded。

    返回 (账本记录, 是否为新导入)；重复导入幂等合并，状态不变。
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFormat(f"不支持的格式（{suffix}），仅支持 PDF/DOCX")
    data = path.read_bytes()
    digest = hashlib.md5(data).hexdigest()[:8]
    stem = re.sub(r"[\W_]+", "-", path.stem, flags=re.UNICODE).strip("-")[:40] or "document"
    sid = f"local:{stem}-{digest}"

    dest_dir = paths.papers_dir(cfg) / safe_dirname(sid)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"paper{suffix}"
    if not dest.exists() or dest.read_bytes() != data:
        dest.write_bytes(data)

    title = _title_from_document(path, fallback=stem.replace("-", " "))
    from paper_agent.models import Paper

    is_new = lib.upsert_paper(Paper(source="local", source_id=sid, title=title), tag=tag)
    if lib.get(sid) is not None and lib.get(sid).status == "discovered":
        lib.set_status(sid, "downloaded", error=None, pdf_path=str(dest))
    rec = lib.get(sid)
    assert rec is not None
    return rec, is_new


def delete_paper(source_id: str, *, cfg: dict, lib: Library) -> bool:
    """全链删除一篇论文：账本 + 向量块 + 原文/解析/知识产物目录。

    返回是否存在（不存在视为删除失败，由调用方决定 404）。
    """
    existed = lib.delete(source_id)
    if not existed:
        return False
    try:
        VectorStore(paths.vector_dir(cfg)).remove_paper(source_id)
    except Exception:
        pass  # 向量库不可用时仍完成账本删除，不阻塞
    for directory in (
        paths.parsed_dir(cfg) / safe_dirname(source_id),
        paths.knowledge_dir(cfg) / safe_dirname(source_id),
        paths.papers_dir(cfg) / safe_dirname(source_id),
    ):
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
    return True


def _title_from_document(path: Path, *, fallback: str) -> str:
    """取文档内建标题元数据；取不到则退回清洗后的文件名。"""
    if path.suffix.lower() == ".pdf":
        try:
            import pymupdf

            with pymupdf.open(path) as doc:
                meta_title = (doc.metadata or {}).get("title") or ""
            if meta_title.strip() and "untitled" not in meta_title.lower():
                return meta_title.strip()[:200]
        except Exception:
            pass
    elif path.suffix.lower() == ".docx":
        try:
            from docx import Document

            doc = Document(str(path))
            for para in doc.paragraphs[:8]:
                style = (para.style.name or "").lower() if para.style is not None else ""
                text = para.text.strip()
                if text and style in ("title", "heading 1"):
                    return text[:200]
        except Exception:
            pass
    return fallback
