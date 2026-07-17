"""Rebuild docs/_static/source_index.json from the combra package AST.

Walks every combra/**/*.py, emitting one record per top-level function/class and
per class method: {leaf, qual, subpkg, file, start, end} (1-based line range).
linkcode_resolve in conf.py matches on `leaf` and disambiguates via `subpkg`/`qual`.
"""

import ast
import json
import sys
from pathlib import Path

# combra submodule root (defaults to the ``combra/`` submodule beside ``docs/``).
default_root = Path(__file__).resolve().parents[2] / "combra"
repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else default_root
pkg = repo_root / "combra"

records = []


def subpkg_of(rel: Path) -> str:
    # rel like combra/angles/display.py -> "angles"; combra/__init__.py -> ""
    parts = rel.parts
    return parts[1] if len(parts) > 2 else ""


def add(node, qual, subpkg, relfile):
    records.append(
        {
            "leaf": qual.split(".")[-1],
            "qual": qual,
            "subpkg": subpkg,
            "file": relfile,
            "start": node.lineno,
            "end": node.end_lineno,
        }
    )


for path in sorted(pkg.rglob("*.py")):
    rel = path.relative_to(repo_root)
    relfile = str(rel)
    subpkg = subpkg_of(rel)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, node.name, subpkg, relfile)
        elif isinstance(node, ast.ClassDef):
            add(node, node.name, subpkg, relfile)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(sub, f"{node.name}.{sub.name}", subpkg, relfile)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    add(node, tgt.id, subpkg, relfile)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            add(node, node.target.id, subpkg, relfile)

out = repo_root.parent / "docs" / "_static" / "source_index.json"
out.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
print(f"wrote {len(records)} records to {out}")
