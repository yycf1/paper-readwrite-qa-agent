"""统一工具契约（tools.py）测试：四态映射、参数校验、结果合理性校验、注册表。"""

from paper_agent.models import Paper
from paper_agent.sources import SourceUnavailable
from paper_agent.tools import (
    TOOL_REGISTRY,
    ToolResult,
    run_tool,
    search_source_tool,
    validate_papers,
)


def _ok_fn():
    return [Paper(source="openalex", source_id="openalex:W1", title="T")]


def test_run_tool_ok() -> None:
    r = run_tool("t", _ok_fn)
    assert r.ok and r.status == "ok"
    assert len(r.data) == 1


def test_run_tool_fatal_on_value_error() -> None:
    def _bad():
        raise ValueError("参数非法")

    r = run_tool("t", _bad)
    assert r.status == "fatal"
    assert "参数" in r.error


def test_run_tool_degraded_on_source_unavailable() -> None:
    def _down():
        raise SourceUnavailable("连接被重置")

    r = run_tool("t", _down)
    assert r.status == "degraded"
    assert "重置" in r.error


def test_run_tool_unexpected_exception_marked() -> None:
    def _boom():
        raise KeyError("x")

    r = run_tool("t", _boom)
    assert r.status == "degraded"
    assert r.meta.get("unexpected") is True
    assert "KeyError" in r.error


def test_search_source_tool_fatal_on_unknown_source() -> None:
    r = search_source_tool("not_a_source", query="x", max_results=5)
    assert r.status == "fatal"


def test_search_source_tool_fatal_on_bad_range() -> None:
    r = search_source_tool("openalex", query="x", max_results=0)
    assert r.status == "fatal"


def test_validate_papers() -> None:
    good = [Paper(source="openalex", source_id="openalex:W1", title="T", year=2024)]
    assert validate_papers(good) == []

    bad = [
        Paper(source="openalex", source_id="openalex:W2", title="  ", year=1800),
        Paper(source="x", source_id="no-colon", title="T"),
    ]
    issues = validate_papers(bad)
    assert len(issues) == 3  # 空标题、年份越界、source_id 非法各 1 条
    assert any("标题为空" in i for i in issues)
    assert any("年份越界" in i for i in issues)
    assert any("source_id 非法" in i for i in issues)


def test_tool_registry_mcp_shape() -> None:
    """注册表按 MCP 规范：description + inputSchema，required 字段必须在 properties 中。"""
    for name, spec in TOOL_REGISTRY.items():
        assert spec.get("description"), f"{name} 缺 description"
        schema = spec["inputSchema"]
        assert schema["type"] == "object"
        for req in schema.get("required", []):
            assert req in schema.get("properties", {}), f"{name}: required {req} 不在 properties"


def test_tool_result_ok_property() -> None:
    assert ToolResult(tool="t", status="ok").ok
    assert not ToolResult(tool="t", status="degraded", error="x").ok
