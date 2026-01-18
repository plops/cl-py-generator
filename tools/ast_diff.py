#!/usr/bin/env python3
# GPT agent suggested this script. it's a good idea. i was hoping it would help me convert changes to a python script back into the lisp based generator. but i didn't have much success, yet 
"""Compare two Python files by AST and report differences."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def dump_ast(path: Path) -> str:
    return ast.dump(ast.parse(path.read_text()), include_attributes=False)


def find_first_diff(left, right, path="root"):
    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, ast.AST):
        for field in left._fields:
            lval = getattr(left, field)
            rval = getattr(right, field)
            diff = find_first_diff(lval, rval, f"{path}.{field}")
            if diff:
                return diff
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return path + ".len", len(left), len(right)
        for idx, (lval, rval) in enumerate(zip(left, right)):
            diff = find_first_diff(lval, rval, f"{path}[{idx}]")
            if diff:
                return diff
        return None
    if left != right:
        return path, left, right
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff two Python files by AST.")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()

    left_tree = ast.parse(args.left.read_text())
    right_tree = ast.parse(args.right.read_text())
    diff = find_first_diff(left_tree, right_tree)
    if not diff:
        print("AST match: True")
        return 0

    path, lval, rval = diff
    print("AST match: False")
    print(f"First diff at {path}")
    print(f"Left: {lval!r}")
    print(f"Right: {rval!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
