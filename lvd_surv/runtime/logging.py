"""logging 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

"""轻量日志门面，避免核心算法依赖具体日志框架。"""

from dataclasses import dataclass

_LEVEL = {"quiet": 0, "normal": 1, "verbose": 2, "debug": 3}


@dataclass
class RuntimeLogger:
    """根据 verbosity 输出阶段信息。错误由异常系统负责。"""

    verbosity: str = "normal"

    def info(self, message: str) -> None:
        """输出普通进度。"""
        if _LEVEL.get(self.verbosity, 1) >= 1:
            print(message)

    def detail(self, message: str) -> None:
        """仅在 verbose/debug 下输出分析细节。"""
        if _LEVEL.get(self.verbosity, 1) >= 2:
            print(message)

    def debug(self, message: str) -> None:
        """仅在 debug 下输出开发信息。"""
        if _LEVEL.get(self.verbosity, 1) >= 3:
            print(message)
