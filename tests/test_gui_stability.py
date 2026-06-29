"""Regression tests for desktop log flooding and cooperative cancellation."""
from __future__ import annotations

import time
from queue import Empty

from lvd_surv.app.desktop import QueueWriter
from lvd_surv.app.tasks import TaskRunner
from lvd_surv.runtime.cancellation import check_cancelled


def _drain(runner: TaskRunner):
    items = []
    while True:
        try:
            items.append(runner.events.get_nowait())
        except Empty:
            return items


def test_queue_writer_coalesces_partial_lines() -> None:
    runner = TaskRunner()
    writer = QueueWriter(runner)
    writer.write("hello")
    writer.write(" world")
    assert _drain(runner) == []
    writer.write("\n")
    assert _drain(runner) == [("output", "hello world\n")]


def test_queue_writer_rate_limits_carriage_return_progress() -> None:
    runner = TaskRunner()
    writer = QueueWriter(runner, progress_interval=60.0)
    for index in range(1000):
        writer.write(f"\rprogress {index}")
    events = _drain(runner)
    assert len(events) <= 1


def test_task_runner_honours_cooperative_stop() -> None:
    runner = TaskRunner()

    def work():
        while True:
            check_cancelled("测试任务")
            time.sleep(0.001)

    runner.start("cancel-test", work)
    time.sleep(0.02)
    runner.request_stop()
    assert runner.thread is not None
    runner.thread.join(timeout=2)
    assert not runner.running
    events = _drain(runner)
    assert any(kind == "warning" and "取消" in text for kind, text in events)
