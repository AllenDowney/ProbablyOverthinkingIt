"""Report Jupyter Book cell tags in chapter notebooks and flag mismatches.

Convention (see PROJECT_BOARD.md, Task 6):
    figure output           -> hide-input
    table output            -> hide-input
    other text output       -> hide-cell
    no output (setup)       -> hide-cell
    plumbing                -> remove-cell

By default this script only reports. With --apply, it writes the expected tag
on each code cell of the jupytext .md version (other tags are kept); review the
.md, then convert it to .ipynb.

Usage:
    python jb/check_tags.py                 # summary for all chapters
    python jb/check_tags.py preston -v      # list mismatched cells in one chapter
    python jb/check_tags.py preston --apply # tag notebooks/preston.md
    python jb/check_tags.py --examples      # the examples in jb/_toc.yml
    jupytext --update --to ipynb notebooks/preston.md
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXAMPLES = [
    "ansur_pca",
    "chile",
    "cricket",
    "ideology_gap",
    "midlife",
    "ctokens",
    "florida",
    "sobriety",
    "firm_size",
    "firm_size_france",
]

CHAPTERS = [
    "gaussian",
    "inspection",
    "preston",
    "lognormal",
    "nbue",
    "berkson",
    "birthweight",
    "longtail",
    "base_rate",
    "simpson",
    "progress",
    "overton",
]

JB_TAGS = ["remove-cell", "remove-input", "remove-output", "hide-cell", "hide-input", "hide-output"]

PLUMBING = re.compile(
    r"^\s*[!%]pip|IN_COLAB|google\.colab|\bdownload\(|savefig\(",
    re.MULTILINE,
)

IMAGE_TYPES = ("image/png", "image/svg+xml", "image/jpeg")


def find_notebook(name):
    """Return the path of notebooks/<name>.ipynb or examples/<name>.ipynb."""
    for folder in ["notebooks", "examples"]:
        path = ROOT / folder / f"{name}.ipynb"
        if path.exists():
            return path
    raise FileNotFoundError(f"{name}.ipynb not in notebooks/ or examples/")


def output_kind(cell):
    """Classify a code cell as 'figure', 'table', 'text', or 'none' by its saved outputs."""
    kind = "none"
    for out in cell.get("outputs", []):
        if out["output_type"] == "error":
            continue
        data = out.get("data", {})
        if any(t in data for t in IMAGE_TYPES):
            return "figure"
        if "text/html" in data:
            # DataFrames display as HTML tables
            kind = "table"
        elif (out["output_type"] == "stream" or data) and kind == "none":
            kind = "text"
    return kind


def classify(cell):
    """Return (kind, expected tag) for a code cell."""
    source = "".join(cell["source"])
    kind = output_kind(cell)
    if kind not in ("figure", "table") and PLUMBING.search(source):
        return "plumbing", "remove-cell"
    if kind in ("figure", "table"):
        return kind, "hide-input"
    return kind, "hide-cell"


def current_tag(cell):
    tags = [t for t in cell.get("metadata", {}).get("tags", []) if t in JB_TAGS]
    return ",".join(tags) if tags else "(none)"


def check(name, verbose):
    path = find_notebook(name)
    nb = json.loads(path.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]

    kinds = Counter()
    mismatches = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        kind, expected = classify(cell)
        kinds[kind] += 1
        tag = current_tag(cell)
        if tag != expected:
            mismatches.append((i, kind, tag, expected, "".join(cell["source"])))

    unexecuted = sum(1 for c in code if c.get("execution_count") is None)
    summary = " ".join(f"{k}={kinds[k]}" for k in ["figure", "table", "text", "none", "plumbing"])
    print(
        f"{name:12s} code={len(code):3d}  {summary:48s} "
        f"mismatch={len(mismatches):3d}  unexecuted={unexecuted}"
    )

    if verbose:
        changes = Counter((tag, expected) for _, _, tag, expected, _ in mismatches)
        for (tag, expected), n in changes.most_common():
            print(f"    {tag:>14s} -> {expected:12s} {n:3d}")
        for i, kind, tag, expected, source in mismatches:
            first = source.strip().splitlines()[0][:60] if source.strip() else ""
            print(f"    [{i:3d}] {kind:8s} {tag:>14s} -> {expected:12s} {first}")
        print()


def apply(name):
    """Write the expected Jupyter Book tag on each code cell of the .md version.

    Cells are classified from the .ipynb (which has the outputs); the tags are
    written to <name>.md next to the .ipynb, which must be a jupytext version of the same
    notebook. Review the .md, then convert with
    `jupytext --update --to ipynb <folder>/<name>.md`.
    """
    import jupytext

    ipynb_path = find_notebook(name)
    nb = json.loads(ipynb_path.read_text())
    expected = [classify(c)[1] for c in nb["cells"] if c["cell_type"] == "code"]

    md_path = ipynb_path.with_suffix(".md")
    md = jupytext.read(md_path)
    code = [c for c in md.cells if c.cell_type == "code"]
    if len(code) != len(expected):
        raise ValueError(f"{name}: .md has {len(code)} code cells, .ipynb has {len(expected)}")

    changed = 0
    for cell, tag in zip(code, expected):
        tags = cell.metadata.get("tags", [])
        new_tags = [t for t in tags if t not in JB_TAGS] + [tag]
        if new_tags != tags:
            cell.metadata["tags"] = new_tags
            changed += 1
    jupytext.write(md, md_path, fmt="md")
    print(f"{name}: retagged {changed} cells in {md_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="notebook names; default: all chapters")
    parser.add_argument("--examples", action="store_true", help="the examples in jb/_toc.yml")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--apply", action="store_true", help="retag the notebooks")
    args = parser.parse_args()
    names = args.names or (EXAMPLES if args.examples else CHAPTERS)
    for name in names:
        if args.apply:
            apply(name)
        check(name, args.verbose)


if __name__ == "__main__":
    main()
