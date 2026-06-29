"""errors 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

"""项目统一异常类型。

所有面向用户的流程错误都通过 :class:`LVDWorkflowError` 描述。异常包含模块、阶段、
原因和修复建议，命令行入口会将其渲染为简洁信息，避免普通用户直接面对冗长 traceback。
"""

from dataclasses import dataclass


@dataclass
class LVDWorkflowError(RuntimeError):
    """表示可定位、可修复的工作流错误。"""

    module: str
    stage: str
    reason: str
    suggestion: str

    def __str__(self) -> str:
        """生成统一的多行错误说明。"""
        return (
            f"[{self.module}] {self.stage} 失败\n"
            f"原因: {self.reason}\n"
            f"建议: {self.suggestion}"
        )
