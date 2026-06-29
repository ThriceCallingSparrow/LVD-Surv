"""Cooperative cancellation shared by GUI tasks and long-running workflows.

The desktop application never force-kills a Python thread.  Instead, the task
runner binds a ``threading.Event`` to the worker thread and algorithms check it
at safe boundaries (device, fold, feature, batch, and epoch boundaries).
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator, Optional


class TaskCancelledError(RuntimeError):
    """Raised when the user requests cancellation at a safe checkpoint."""


_LOCAL = threading.local()


@contextmanager
def bind_cancel_event(event: threading.Event) -> Iterator[None]:
    """Bind ``event`` to the current worker thread for the duration of a task."""
    previous: Optional[threading.Event] = getattr(_LOCAL, "event", None)
    _LOCAL.event = event
    try:
        yield
    finally:
        if previous is None:
            try:
                delattr(_LOCAL, "event")
            except AttributeError:
                pass
        else:
            _LOCAL.event = previous


def cancellation_requested() -> bool:
    """Return whether the current worker task has received a stop request."""
    event: Optional[threading.Event] = getattr(_LOCAL, "event", None)
    return bool(event and event.is_set())


def check_cancelled(stage: str = "任务") -> None:
    """Raise :class:`TaskCancelledError` when cancellation was requested."""
    if cancellation_requested():
        raise TaskCancelledError(f"{stage}已由用户取消。")
