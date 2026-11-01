"""Candidate questions for the Overton update, from the full GSS file.

Pass 1 reads the GSS 1972-2024 cumulative file (R3a) in chunks and, for
each variable, counts respondents per year who answered it and also
answered polviews, and collects its distinct valid values. Easy gates:

- at least MIN_WAVES years with at least MIN_PER_WAVE such respondents,
- those years span at least MIN_SPAN years,
- between 2 and MAX_VALUES distinct valid values (excludes continuous
  variables and long lists like occupation codes),
- not an administrative or core variable (id, year, weights, ...).

Pass 2 reads only the variables that pass, writes an extract, and writes a
table of candidates with metadata from the CultureWar repo: the codebook
(gssrdoc: description, question, value labels, subjects, modules) and the
CultureWar variable catalog (selected values, where available).

No trends are computed here.

Writes:
- ../data/gss_overton_candidates_r3a.hdf (key "gss")
- overton_figs/candidates_pass1.csv

Usage: python overton_candidates.py
"""

import gzip
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from overton_pipeline import FIG_DIR, ORDER
from overton_selection import BOOK_CANDIDATES

CULTUREWAR = Path.home() / "CultureWar"
RAW = CULTUREWAR / "data/raw/gss7224_r3a.dta.gz"
GSSRDOC = CULTUREWAR / "codebook/extracted/gssrdoc_dict_flat.json"
CATALOG = CULTUREWAR / "notebooks/variable_metadata/variables.yaml"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXTRACT = DATA_DIR / "gss_overton_candidates_r3a.hdf"

MIN_WAVES = 8
MIN_SPAN = 15
MIN_PER_WAVE = 100
MAX_VALUES = 7
CHUNKSIZE = 5000

CORE = ["year", "id", "polviews", "cohort", "age", "ballot", "wtssall", "wtssps", "wtssnrps", "mode"]
ADMIN_PREFIXES = ("wt", "samp", "vstrat", "vpsu", "form", "version", "issp", "id")


def decompress(path, dest_dir):
    """Decompress the raw file to a temporary .dta (the reader needs to seek)."""
    dest = Path(dest_dir) / "gss7224_r3a.dta"
    with gzip.open(path, "rb") as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return dest


def first_pass(path):
    """Counts per (variable, year) among respondents with polviews; distinct values."""
    counts = []
    values = {}
    reader = pd.read_stata(path, convert_categoricals=False, iterator=True, chunksize=CHUNKSIZE)
    for chunk in reader:
        chunk = chunk[chunk["polviews"].notna()]
        numeric = chunk.select_dtypes("number")
        counts.append(numeric.notna().groupby(chunk["year"]).sum())
        for col in numeric.columns:
            seen = values.setdefault(col, set())
            if len(seen) <= MAX_VALUES:
                seen.update(numeric[col].dropna().unique().tolist())
    by_year = pd.concat(counts).groupby(level=0).sum()
    return by_year, values


def apply_gates(by_year, values):
    """Summarize each variable and apply the easy gates."""
    rows = []
    for col in by_year.columns:
        per_year = by_year[col]
        waves = per_year[per_year >= MIN_PER_WAVE]
        n_values = len(values.get(col, ()))
        rows.append(
            dict(
                variable=col,
                waves=len(waves),
                first=waves.index.min() if len(waves) else np.nan,
                last=waves.index.max() if len(waves) else np.nan,
                n=int(per_year.sum()),
                n_values=n_values,
                values=" ".join(f"{v:g}" for v in sorted(values[col])) if n_values <= MAX_VALUES else "",
            )
        )
    table = pd.DataFrame(rows).set_index("variable")
    table["span"] = table["last"] - table["first"]
    table["admin"] = [v in CORE or v.startswith(ADMIN_PREFIXES) for v in table.index]
    table["passes"] = (
        (table["waves"] >= MIN_WAVES)
        & (table["span"] >= MIN_SPAN)
        & table["n_values"].between(2, MAX_VALUES)
        & ~table["admin"]
    )
    return table


def add_metadata(table):
    """Join codebook and CultureWar catalog metadata."""
    gssrdoc = json.load(open(GSSRDOC))
    for field in ["description", "question", "value_labels", "subjects", "modules"]:
        table[field] = [gssrdoc.get(v, {}).get(field) for v in table.index]

    catalog = yaml.safe_load(open(CATALOG))
    gss_catalog = {k: v for k, v in catalog.items() if v.get("source") == "gss"}
    table["in_culturewar"] = [v in gss_catalog for v in table.index]
    table["cw_selected_values"] = [gss_catalog.get(v, {}).get("selected_values") for v in table.index]
    table["in_book_candidates"] = [v in BOOK_CANDIDATES for v in table.index]
    table["in_book_15"] = [v in ORDER for v in table.index]
    return table


def second_pass(path, columns):
    """Read only the passing variables plus the core columns."""
    reader = pd.read_stata(
        path, columns=columns, convert_categoricals=False, iterator=True, chunksize=CHUNKSIZE
    )
    return pd.concat(reader, ignore_index=True)


def main():
    with tempfile.TemporaryDirectory(dir=DATA_DIR) as tmp:
        path = decompress(RAW, tmp)
        by_year, values = first_pass(path)
        table = apply_gates(by_year, values)
        passing = table.index[table["passes"]].tolist()
        columns = [c for c in CORE if c in table.index or c == "mode"] + passing
        gss = second_pass(path, columns)

    gss.to_hdf(EXTRACT, key="gss", mode="w", complevel=6)
    table = add_metadata(table)
    table.to_csv(FIG_DIR / "candidates_pass1.csv")

    print(f"{len(table)} variables; {len(passing)} pass the gates")
    print(f"extract {gss.shape} -> {EXTRACT}")
    print("book candidates passing:", table.loc[table["in_book_candidates"], "passes"].sum(), "of", len(BOOK_CANDIDATES))
    print("in CultureWar catalog and passing:", (table["passes"] & table["in_culturewar"]).sum())


if __name__ == "__main__":
    main()
