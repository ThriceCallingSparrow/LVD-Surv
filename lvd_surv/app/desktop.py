"""LVD-Surv desktop console with bounded, batched log rendering."""
from __future__ import annotations

import contextlib
import io
import os
import queue
import threading
import time
from datetime import datetime, timezone
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext

# Worker-side plotting must never attach to the Tk/Cocoa GUI event loop.
os.environ.setdefault("MPLBACKEND", "Agg")

from lvd_surv.app.commands import CommandEngine, HELP_TEXT
from lvd_surv.app.session import SessionContext
from lvd_surv.app.tasks import TaskRunner


class QueueWriter(io.TextIOBase):
    """Buffer stdout/stderr and emit coarse-grained queue messages.

    ``tqdm`` writes many carriage-return fragments.  Treating every fragment as
    a separate Tk update can starve the GUI event loop.  This writer coalesces
    normal lines and rate-limits carriage-return progress updates.
    """

    def __init__(self, runner: TaskRunner, kind: str = "output", *, progress_interval: float = 0.25) -> None:
        self.runner = runner
        self.kind = kind
        self.progress_interval = progress_interval
        self._buffer = ""
        self._last_progress = 0.0
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            # Dynamic progress bars use CR to overwrite one terminal line.  In
            # the GUI we emit at most four snapshots per second.
            if "\r" in text and "\n" not in text:
                latest = text.split("\r")[-1].strip()
                now = time.monotonic()
                if latest and now - self._last_progress >= self.progress_interval:
                    self.runner.events.put((self.kind, f"[进度] {latest}\n"))
                    self._last_progress = now
                return len(text)

            self._buffer += text.replace("\r", "")
            if "\n" in self._buffer or len(self._buffer) >= 4096:
                cut = self._buffer.rfind("\n") + 1 if "\n" in self._buffer else len(self._buffer)
                chunk, self._buffer = self._buffer[:cut], self._buffer[cut:]
                if chunk:
                    self.runner.events.put((self.kind, chunk))
        return len(text)

    def flush(self) -> None:
        with self._lock:
            if self._buffer:
                self.runner.events.put((self.kind, self._buffer))
                self._buffer = ""


class DesktopApp:
    """Stateful desktop workbench with a responsive, bounded output panel."""

    MAX_EVENTS_PER_TICK = 160
    MAX_POLL_TIME_SECONDS = 0.018
    MAX_OUTPUT_LINES = 20_000

    def __init__(self, root: tk.Tk, default_config: str | None = None) -> None:
        self.root = root
        self.root.title("LVD-Surv 可靠性预测工作台")
        self.root.geometry("1050x720")
        self.session = SessionContext()
        # Internal flag: algorithms suppress terminal-style progress bars but
        # retain every mathematical operation and every saved artifact.
        self.session.settings["gui_mode"] = True
        self.runner = TaskRunner()
        self.history_index = 0
        self._log_filename = f"desktop_session_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"

        self.output = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 10))
        self.output.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        input_frame = tk.Frame(root)
        input_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(input_frame, text="lvd>").pack(side=tk.LEFT)
        self.entry = tk.Entry(input_frame, font=("Consolas", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.entry.bind("<Return>", self._on_enter)
        self.entry.bind("<Up>", self._history_up)
        self.entry.bind("<Down>", self._history_down)
        self.entry.focus_set()

        button_frame = tk.Frame(root)
        button_frame.pack(fill=tk.X, padx=8, pady=2)
        for label, command in (
            ("停止任务", self.runner.request_stop),
            ("清空输出", self.clear_output),
            ("帮助", lambda: self.append_output(HELP_TEXT + "\n")),
            ("打开结果目录", lambda: self._execute_line("open output")),
        ):
            tk.Button(button_frame, text=label, command=command).pack(side=tk.LEFT, padx=3)

        self.status_var = tk.StringVar(value="配置: - | 模型: - | 状态: idle")
        tk.Label(root, textvariable=self.status_var, anchor="w", relief=tk.SUNKEN).pack(fill=tk.X, side=tk.BOTTOM)

        self.engine = CommandEngine(
            self.session,
            choose_config=self._choose_config,
            choose_model=self._choose_model,
            clear_output=self.clear_output,
            request_stop=self.runner.request_stop,
            close_app=self.root.destroy,
        )
        self.root.after(40, self._poll_events)
        self.append_output("LVD-Surv 工作台已启动。输入 help 查看命令。\n")
        if default_config and Path(default_config).is_file():
            try:
                self.append_output(self.engine.execute(f'load "{default_config}"') + "\n")
            except Exception as exc:
                self.append_output(f"[错误] 默认配置加载失败: {exc}\n")
        self._update_status()

    def _choose_config(self) -> str:
        return filedialog.askopenfilename(title="选择配置文件", filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")])

    def _choose_model(self) -> str:
        return filedialog.askopenfilename(title="选择 checkpoint", filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")])

    def _on_enter(self, _event=None) -> None:
        line = self.entry.get().strip()
        self.entry.delete(0, tk.END)
        if line:
            self.append_output(f"lvd> {line}\n")
            self._execute_line(line)

    def _execute_line(self, line: str) -> None:
        if self.runner.running and CommandEngine.is_heavy(line):
            self.append_output("[提示] 当前已有重型任务运行。输入 stop 请求停止。\n")
            return
        if CommandEngine.is_heavy(line):
            self.session.task_status = line.split()[0]

            def target():
                out_writer = QueueWriter(self.runner)
                err_writer = QueueWriter(self.runner, "error_output")
                with contextlib.redirect_stdout(out_writer), contextlib.redirect_stderr(err_writer):
                    try:
                        return self.engine.execute(line)
                    finally:
                        out_writer.flush()
                        err_writer.flush()

            try:
                self.runner.start(line.split()[0], target)
            except Exception as exc:
                self.append_output(f"[错误] {exc}\n")
        else:
            try:
                result = self.engine.execute(line)
                if result:
                    self.append_output(result + "\n")
            except Exception as exc:
                self.append_output(f"[错误] {exc}\n")
        self._update_status()

    def _poll_events(self) -> None:
        """Consume a bounded event batch so Tk always regains control."""
        started = time.monotonic()
        processed = 0
        output_chunks: list[str] = []
        while processed < self.MAX_EVENTS_PER_TICK and time.monotonic() - started < self.MAX_POLL_TIME_SECONDS:
            try:
                kind, text = self.runner.events.get_nowait()
            except queue.Empty:
                break
            processed += 1
            if kind == "status":
                self.session.task_status = text
            elif kind == "error":
                display = text if self.session.settings.get("log") == "debug" else text.splitlines()[0]
                output_chunks.append(f"[错误] {display}\n")
            elif kind == "warning":
                output_chunks.append(f"[警告] {text}\n")
            elif kind in {"output", "error_output"}:
                output_chunks.append(text)
            elif kind == "result":
                output_chunks.append(text + "\n")

        if output_chunks:
            self.append_output("".join(output_chunks))
        self._update_status()
        # Drain a backlog quickly, but still yield to native window events.
        delay = 20 if not self.runner.events.empty() else 70
        self.root.after(delay, self._poll_events)

    def append_output(self, text: str) -> None:
        """Append a bounded UI batch and mirror it to the current session log."""
        if not text:
            return
        self._append_session_log(text)
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, text)
        # Keep the widget bounded.  Full algorithm reports remain on disk.
        line_count = int(self.output.index("end-1c").split(".")[0])
        if line_count > self.MAX_OUTPUT_LINES:
            remove = line_count - self.MAX_OUTPUT_LINES
            self.output.delete("1.0", f"{remove + 1}.0")
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)


    def _append_session_log(self, text: str) -> None:
        """Persist all desktop output after a configuration has been loaded."""
        if not self.session.base_config:
            return
        try:
            path = self.session.paths.logs / self._log_filename
            with path.open("a", encoding="utf-8") as handle:
                handle.write(text)
        except Exception:
            # Logging must never make the GUI unusable; workflow records still
            # capture success/failure independently.
            return

    def clear_output(self) -> None:
        self.output.configure(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.output.configure(state=tk.DISABLED)

    def _update_status(self) -> None:
        dataset = self.session.base_config.get("data", {}).get("dataset", "-") if self.session.base_config else "-"
        model = self.session.checkpoint.name if self.session.checkpoint else "-"
        self.status_var.set(f"配置: {self.session.config_path or '-'} | 数据集: {dataset} | 模型: {model} | 状态: {self.session.task_status}")

    def _history_up(self, _event=None):
        if self.engine.history:
            self.history_index = max(0, self.history_index - 1)
            self.entry.delete(0, tk.END)
            self.entry.insert(0, self.engine.history[self.history_index])
        return "break"

    def _history_down(self, _event=None):
        if self.engine.history:
            self.history_index = min(len(self.engine.history), self.history_index + 1)
            self.entry.delete(0, tk.END)
            if self.history_index < len(self.engine.history):
                self.entry.insert(0, self.engine.history[self.history_index])
        return "break"


def launch(default_config: str | None = None) -> None:
    """Create and run the desktop window."""
    root = tk.Tk()
    DesktopApp(root, default_config=default_config)
    root.mainloop()
