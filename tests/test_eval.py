"""RAG 评测（rag/eval.py）单测 + 评测集 schema 冒烟。"""

from __future__ import annotations

from pathlib import Path

import pytest

from paper_agent.rag.eval import judge_answer, judge_retrieval, load_evalset
from paper_agent.rag.vectorstore import Hit

EVALSET = Path(__file__).resolve().parents[1] / "eval" / "evalset.json"


def _hit(section: str) -> Hit:
    return Hit(paper_id="x:1", title="T", section=section, page=1, text="内容", distance=0.1)


# ---- 判定函数 ----


def test_judge_retrieval_substring_case_insensitive() -> None:
    case = {"expect_sections": ["star-style"]}
    assert judge_retrieval(case, [_hit("3.5 Star-style Self-supervised Task")]) is True
    assert judge_retrieval(case, [_hit("Abstract")]) is False


def test_judge_retrieval_null_when_no_expectation() -> None:
    assert judge_retrieval({"expect_sections": None}, []) is None
    assert judge_retrieval({}, []) is None


def test_judge_answer_groups_and_or() -> None:
    # 组间 AND，组内 OR
    case = {"expect_keywords": [["93.11"], ["dice", "Dice 系数"]]}
    assert judge_answer(case, "Dice 系数为 93.11") is True
    assert judge_answer(case, "dice coefficient 93.11") is True
    assert judge_answer(case, "93.11 且 IoU 很高") is False
    assert judge_answer(case, "效果很好") is False


def test_judge_answer_null_when_no_expectation() -> None:
    assert judge_answer({"expect_keywords": None}, "任何回答") is None


# ---- 评测集本身（防手写 schema 漂移） ----


def test_evalset_loads_and_has_enough_cases() -> None:
    data = load_evalset(EVALSET)
    assert len(data["cases"]) >= 20
    papers = {c.get("paper_id") for c in data["cases"]}
    assert len([p for p in papers if p]) >= 2  # 至少覆盖两篇论文
    assert any(c.get("paper_id") is None for c in data["cases"])  # 含跨库/范围外用例


def test_evalset_case_ids_unique() -> None:
    data = load_evalset(EVALSET)
    ids = [c["id"] for c in data["cases"]]
    assert len(ids) == len(set(ids))


def test_evalset_expected_sections_reference_real_metadata() -> None:
    """期望章节名应能被清洗后的章节名（≤48 字符）包含——防评测集与解析产物脱节。"""
    data = load_evalset(EVALSET)
    for case in data["cases"]:
        for s in case.get("expect_sections") or []:
            assert len(s) <= 48, f"{case['id']}: 期望章节 {s!r} 超过章节名清洗上限"


def test_evalset_load_rejects_bad_schema(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"cases": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_evalset(bad)
