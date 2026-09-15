"""后台任务管理：流水线等长任务的线程执行与状态轮询。

SQLite 账本只存论文状态，不存「一次运行」的过程数据——任务状态放进程
内存（重启丢失可接受：library.db 是断点续跑的事实来源，重跑同主题即恢复）。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable

# 全局任务表：job_id → Job。锁只保护表本身的读写，任务体在各自线程里跑。
_JOBS: dict[str, "Job"] = {}
_LOCK = threading.Lock()


@dataclass
class Job:
    """一次后台任务。status: running → done | error。"""

    id: str
    kind: str
    status: str = "running"
    result: dict = field(default_factory=dict)
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }


def submit(kind: str, fn: Callable[[dict], dict], *, result: dict | None = None) -> Job:
    """提交后台任务：fn(job.result) 在新线程执行，返回值并入 result。"""
    job = Job(id=uuid.uuid4().hex[:12], kind=kind)
    if result:
        job.result.update(result)

    def _run():
        try:
            extra = fn(job.result) or {}
            job.result.update(extra)
            job.status = "done"
        except Exception as exc:  # 任务体自带降级，到这里的是意外错误
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"[:500]

    with _LOCK:
        _JOBS[job.id] = job
    threading.Thread(target=_run, daemon=True).start()
    return job


def get(job_id: str) -> Job | None:
    with _LOCK:
        return _JOBS.get(job_id)


def list_jobs() -> list[Job]:
    with _LOCK:
        return list(_JOBS.values())
