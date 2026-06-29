"""GUI background task execution and cooperative cancellation."""
from __future__ import annotations

import threading
import traceback
from queue import Queue
from typing import Callable, Optional

from lvd_surv.runtime.cancellation import TaskCancelledError, bind_cancel_event


class TaskRunner:
    """Run one heavyweight task in a worker thread and emit UI-safe events."""

    def __init__(self) -> None:
        self.events: Queue[tuple[str, str]] = Queue()
        self.thread: Optional[threading.Thread] = None
        self.cancel_requested = threading.Event()

    @property
    def running(self) -> bool:
        """Return whether a worker thread is currently alive."""
        return bool(self.thread and self.thread.is_alive())

    def start(self, name: str, target: Callable[[], object]) -> None:
        """Start ``target``; only one heavyweight task may run at a time."""
        if self.running:
            raise RuntimeError("已有任务正在运行。")
        self.cancel_requested.clear()

        def worker() -> None:
            self.events.put(("status", f"running:{name}"))
            self.events.put(("output", f"[任务] 正在启动 {name}……\n"))
            try:
                with bind_cancel_event(self.cancel_requested):
                    result = target()
                self.events.put(("result", str(result) if result is not None else "完成"))
            except TaskCancelledError as exc:
                self.events.put(("warning", str(exc)))
            except Exception as exc:  # Keep the desktop process alive after failures.
                self.events.put(("error", f"{exc}\n{traceback.format_exc()}"))
            finally:
                self.events.put(("status", "idle"))

        self.thread = threading.Thread(target=worker, name=f"lvd-{name}", daemon=True)
        self.thread.start()

    def request_stop(self) -> None:
        """Request cancellation; algorithms stop at their next safe checkpoint."""
        if self.running:
            self.cancel_requested.set()
            self.events.put(("warning", "已请求停止；任务将在下一个安全检查点终止。"))
        else:
            self.events.put(("warning", "当前没有正在运行的任务。"))
