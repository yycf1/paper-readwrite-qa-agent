"""library.db：文献库账本（SQLite）。

全流程状态记录与跨源去重的唯一事实来源：
- 五态生命周期 discovered → downloaded → parsed → analyzed → indexed；
- DOI / 归一化标题两级去重；
- 所有命令与 LangGraph 节点围绕账本决定「跳过还是执行」（断点续跑、增量处理的基础）。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from paper_agent.models import Paper, merge_papers

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    source_id     TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    norm_title    TEXT NOT NULL,
    authors       TEXT NOT NULL DEFAULT '[]',
    abstract      TEXT NOT NULL DEFAULT '',
    year          INTEGER,
    doi           TEXT,
    citations     INTEGER,
    pdf_urls      TEXT NOT NULL DEFAULT '[]',
    landing_page  TEXT,
    source        TEXT NOT NULL,
    abstract_only INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'discovered',
    pdf_path      TEXT,
    error         TEXT,
    tag           TEXT NOT NULL DEFAULT '',
    added_at      TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_papers_norm_title ON papers(norm_title);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
"""


@dataclass
class PaperRecord:
    """账本里的一行：Paper 元数据 + 流程状态 + 分组标签。"""

    paper: Paper
    status: str
    pdf_path: str | None
    error: str | None
    added_at: str
    tag: str = ""


class Library:
    """papers 表的读写入口。命令完成后 commit，保证跨命令可见。"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：Web 服务下依赖的创建与关闭可能落在不同的
        # 线程池工作线程（anyio 不保证线程亲和）；连接本身不跨线程共享
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """老库原地升级：缺 tag 列时补齐（M10 分组功能）。"""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(papers)")}
        if "tag" not in cols:
            self.conn.execute("ALTER TABLE papers ADD COLUMN tag TEXT NOT NULL DEFAULT ''")
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- 写入 ----

    def upsert_paper(self, paper: Paper, *, tag: str | None = None) -> bool:
        """入库新论文（status=discovered）；已存在则补充空缺元数据但绝不改变其状态。

        tag 只在首次入库时写入（检索词/上传标记，M10 分组用），已有条目不覆盖。
        返回是否为新论文。补充元数据是幂等的：arXiv/EuropePMC 后到的新链接、
        新摘要会并入已有条目，而已 downloaded/parsed 的状态不受影响。
        """
        now = datetime.now(UTC).isoformat(timespec="seconds")
        existing = self.conn.execute(
            "SELECT source_id FROM papers WHERE source_id = ?", (paper.source_id,)
        ).fetchone()
        if existing:
            old = self.get(paper.source_id)
            assert old is not None
            paper = merge_papers(old.paper, paper)
            status = old.status
            is_new = False
        else:
            status = "discovered"
            is_new = True
        self.conn.execute(
            "INSERT OR REPLACE INTO papers (source_id, title, norm_title, authors, abstract,"
            " year, doi, citations, pdf_urls, landing_page, source, abstract_only,"
            " status, pdf_path, error, tag, added_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                paper.source_id,
                paper.title,
                _norm_title(paper.title),
                json.dumps(paper.authors, ensure_ascii=False),
                paper.abstract,
                paper.year,
                paper.doi,
                paper.citations,
                json.dumps(paper.pdf_urls),
                paper.landing_page,
                paper.source,
                int(paper.abstract_only),
                status,
                self._pdf_path_of(paper.source_id) if not is_new else None,
                None,
                (tag or "") if is_new else self._tag_of(paper.source_id),
                now if is_new else self._added_at_of(paper.source_id, now),
                now,
            ),
        )
        self.conn.commit()
        return is_new

    def set_status(
        self,
        source_id: str,
        status: str,
        *,
        error: str | None = None,
        pdf_path: str | None = None,
    ) -> None:
        """推进/标记状态；error 传 None 表示清除旧错误（重试成功时）。"""
        self.conn.execute(
            "UPDATE papers SET status = ?, error = ?,"
            " pdf_path = COALESCE(?, pdf_path), updated_at = ? WHERE source_id = ?",
            (status, error, pdf_path, datetime.now(UTC).isoformat(timespec="seconds"), source_id),
        )
        self.conn.commit()

    # ---- 查询 ----

    def get(self, source_id: str) -> PaperRecord | None:
        row = self.conn.execute(
            "SELECT * FROM papers WHERE source_id = ?", (source_id,)
        ).fetchone()
        return _to_record(row) if row else None

    def list(self, status: str | None = None, *, tag: str | None = None) -> list[PaperRecord]:
        sql = "SELECT * FROM papers"
        conditions, params = [], []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if tag:
            conditions.append("tag = ?")
            params.append(tag)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY added_at DESC"
        return [_to_record(r) for r in self.conn.execute(sql, params).fetchall()]

    def delete(self, source_id: str) -> bool:
        """删除一条账本记录（向量库与产物文件的清理由调用方负责）。"""
        cur = self.conn.execute("DELETE FROM papers WHERE source_id = ?", (source_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def tags(self) -> dict[str, int]:
        """分组标签 → 数量（不含空标签），供前端分组导航。"""
        rows = self.conn.execute(
            "SELECT tag, COUNT(*) AS n FROM papers WHERE tag != '' GROUP BY tag ORDER BY n DESC"
        ).fetchall()
        return {r["tag"]: r["n"] for r in rows}

    def find(self, fragment: str) -> list[str]:
        """按 source_id 片段模糊查找（CLI 允许用户只输入 W274... 或 2401.12345）。"""
        rows = self.conn.execute(
            "SELECT source_id FROM papers WHERE source_id LIKE ? ORDER BY source_id",
            (f"%{fragment}%",),
        ).fetchall()
        return [r["source_id"] for r in rows]

    def find_duplicate(self, paper: Paper) -> str | None:
        """两级去重：先 DOI（非空精确匹配），再归一化标题。返回已有条目的 source_id。"""
        if paper.doi:
            row = self.conn.execute(
                "SELECT source_id FROM papers WHERE doi = ?", (paper.doi,)
            ).fetchone()
            if row:
                return row["source_id"]
        row = self.conn.execute(
            "SELECT source_id FROM papers WHERE norm_title = ?", (_norm_title(paper.title),)
        ).fetchone()
        return row["source_id"] if row else None

    def counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM papers GROUP BY status"
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    # ---- 内部 ----

    def _pdf_path_of(self, source_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT pdf_path FROM papers WHERE source_id = ?", (source_id,)
        ).fetchone()
        return row["pdf_path"] if row else None

    def _tag_of(self, source_id: str) -> str:
        row = self.conn.execute(
            "SELECT tag FROM papers WHERE source_id = ?", (source_id,)
        ).fetchone()
        return (row["tag"] if row else "") or ""

    def _added_at_of(self, source_id: str, fallback: str) -> str:
        row = self.conn.execute(
            "SELECT added_at FROM papers WHERE source_id = ?", (source_id,)
        ).fetchone()
        return (row["added_at"] if row else None) or fallback


def _norm_title(title: str) -> str:
    from paper_agent.models import normalize_title

    return normalize_title(title)


def _to_record(row: sqlite3.Row) -> PaperRecord:
    pdf_urls = json.loads(row["pdf_urls"])
    paper = Paper(
        source=row["source"],
        source_id=row["source_id"],
        title=row["title"],
        authors=json.loads(row["authors"]),
        abstract=row["abstract"],
        year=row["year"],
        doi=row["doi"],
        citations=row["citations"],
        pdf_urls=pdf_urls,
        landing_page=row["landing_page"],
        abstract_only=bool(row["abstract_only"]),
    )
    return PaperRecord(
        paper=paper,
        status=row["status"],
        pdf_path=row["pdf_path"],
        error=row["error"],
        added_at=row["added_at"],
        tag=row["tag"] if "tag" in row.keys() else "",
    )
