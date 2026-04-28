"""Headless executor for the docs/*.ipynb notebooks.

Run from the repo root with PARALLEL_API_KEY set:

    poetry run python scripts/run_notebooks.py

Skips cells that need user interaction (`%pip install`, `getpass.getpass`)
so the rest can run end-to-end against the real Parallel API. Useful as a
pre-release smoke test that the published examples still work.

Pass paths to limit which notebooks run:

    poetry run python scripts/run_notebooks.py docs/chat.ipynb
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
DEFAULT_NOTEBOOKS = [
    DOCS / "chat.ipynb",
    DOCS / "search_tool.ipynb",
    DOCS / "extract_tool.ipynb",
]


def _is_interactive_cell(source: str) -> bool:
    """Skip cells that block on user input or shell-out to install."""
    stripped = source.lstrip()
    return stripped.startswith(("%pip", "!pip")) or "getpass.getpass" in source


def run_notebook(path: Path, *, timeout: int = 180) -> bool:
    """Execute a notebook in-place and report whether it succeeded."""
    nb = nbformat.read(path, as_version=4)

    keep = []
    for cell in nb.cells:
        if cell.cell_type == "code" and _is_interactive_cell(
            "".join(cell.get("source", [])),
        ):
            continue
        keep.append(cell)
    nb.cells = keep

    client = NotebookClient(nb, timeout=timeout, kernel_name="python3")
    try:
        client.execute()
    except CellExecutionError as e:
        print(f"FAIL: {path.name}")
        # Tail of the traceback is what matters; full message is huge.
        print(str(e)[-2000:])
        return False
    print(f"OK:   {path.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="Notebook paths (defaults to docs/*.ipynb)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-cell timeout in seconds (default 180)",
    )
    args = parser.parse_args()

    if not os.environ.get("PARALLEL_API_KEY"):
        print(
            "PARALLEL_API_KEY is not set; notebooks that hit the API will fail.",
            file=sys.stderr,
        )

    notebooks = args.notebooks or DEFAULT_NOTEBOOKS
    ok = True
    for nb_path in notebooks:
        ok &= run_notebook(nb_path.resolve(), timeout=args.timeout)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
