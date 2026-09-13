"""paper-agent 命令行入口（Typer 包）。

命令实现在子模块中，导入时向 core.app 注册；这里负责汇总暴露 `app`
（pyproject 的 `pa` 入口指向 `paper_agent.cli:app`）。
"""

from paper_agent.cli.core import app
from paper_agent.cli import admin, flow, ingest, qa  # noqa: F401  导入即注册命令

__all__ = ["app"]
