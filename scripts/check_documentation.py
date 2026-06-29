#!/usr/bin/env python
"""统计 Python 公共定义的文档字符串覆盖率。

项目把“注释率”定义为模块、类及函数的 docstring 覆盖率；行内注释用于解释局部算法，
但不以堆砌无意义注释的方式提高数字。脚本低于 80% 时退出失败。
"""
from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "lvd_surv"
covered = total = 0
for path in ROOT.rglob("*.py"):
    if "__pycache__" in path.parts:
        continue
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not getattr(n, "name", "").startswith("_")]
    for node in nodes:
        total += 1
        covered += int(bool(ast.get_docstring(node)))
rate = covered / total if total else 1.0
print(f"Documentation coverage: {covered}/{total} = {rate:.1%}")
raise SystemExit(0 if rate >= 0.80 else 1)
