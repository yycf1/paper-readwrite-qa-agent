"""M0 冒烟测试：配置可加载、LangGraph 图可执行。"""

from paper_agent.config import load_config
from paper_agent.graph.hello import run_hello


def test_config_loads():
    cfg = load_config()
    assert "llm" in cfg and "embedding" in cfg
    assert cfg["llm"]["base_url"].startswith("http")


def test_hello_graph_runs():
    result = run_hello()
    assert "langgraph" in result["message"]
    assert result["steps"] == ["prepare", "finish"]
