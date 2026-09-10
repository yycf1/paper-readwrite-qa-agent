"""运行时目录与账本路径：CLI 与 LangGraph 流水线共用，避免逻辑漂移。"""

from __future__ import annotations

from pathlib import Path

from paper_agent.config import PROJECT_ROOT, load_config
from paper_agent.library import Library


def data_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return PROJECT_ROOT / cfg.get("data_dir", "data")


def papers_dir(cfg: dict | None = None) -> Path:
    return data_dir(cfg) / "papers"


def parsed_dir(cfg: dict | None = None) -> Path:
    return data_dir(cfg) / "parsed"


def knowledge_dir(cfg: dict | None = None) -> Path:
    return data_dir(cfg) / "knowledge"


def vector_dir(cfg: dict | None = None) -> Path:
    return data_dir(cfg) / "db"


def library(cfg: dict | None = None) -> Library:
    return Library(data_dir(cfg) / "library.db")
