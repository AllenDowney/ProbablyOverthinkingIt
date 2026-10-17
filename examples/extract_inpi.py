"""Extract firm-size fields from the INPI RNE comptes annuels bulk stock.

Source: INPI, Registre national des entreprises (RNE), comptes annuels,
stock file stock_RNE_comptes_annuels_NIVEAU1_20260320_1400.zip,
downloaded 2026-10-15 from the INPI FTP server (www.inpi.net, FTP over TLS)
under the RNE reuse licence. See PROJECT_BOARD.md, Task 5, and
data/inpi_france/TERMS.md.

Writes one row per filing (no deduplication) with these columns:

    siren          firm identifier
    date_cloture   fiscal year end
    duree_mois     length of fiscal year in months (0 if not reported)
    type_bilan     C complete, S simplified, K consolidated
    confidentiality
    naf            activity code
    ca, ca_prev    net revenue, current and prior year (EUR)
    total_bilan    total assets, net (EUR)
    effectif       average headcount
    date_depot, updated_at

Firm names are omitted on purpose (sole proprietorships carry personal names).
Banks and insurers (types B, A, AC, AS) are dropped.

Usage:
    python examples/extract_inpi.py data/inpi_france/stock_RNE_comptes_annuels_NIVEAU1_20260320_1400.zip \
        data/inpi_france/comptes_annuels.parquet data/inpi_comptes_annuels.parquet

The second output (local, gitignored) has one row per filing with siren.
The optional third output is the redistributable extract committed to the repo:
deduplicated, without siren or filing dates (see make_public).
"""

import json
import sys
import zipfile
from multiprocessing import Pool

import pandas as pd

DEC = json.JSONDecoder()
WS = " ,\n\r\t"
TYPES = {"C", "S", "K"}


def records(text):
    """Yield records from a JSON array, tolerating a truncated tail.

    Returns (via StopIteration value) whether the array was closed properly.
    """
    i = text.index("[") + 1
    n = len(text)
    while i < n:
        while i < n and text[i] in WS:
            i += 1
        if i >= n:
            return False
        if text[i] == "]":
            return True
        try:
            obj, i = DEC.raw_decode(text, i)
        except json.JSONDecodeError:
            return False
        yield obj
    return False


def to_int(s):
    """Convert a zero-padded field to int; None if blank or malformed."""
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def get(rows, code, col):
    return to_int(rows.get(code, {}).get(col))


def sum_codes(rows, codes, col):
    values = [get(rows, code, col) for code in codes]
    values = [v for v in values if v is not None]
    return sum(values) if values else None


def extract_record(r):
    t = r.get("typeBilan")
    if t not in TYPES or r.get("deleted"):
        return None
    bilan = r["bilanSaisi"]["bilan"]
    identite = bilan["identite"]
    rows = {
        line["code"]: line
        for page in bilan["detail"]["pages"]
        for line in page["liasses"]
    }

    if t == "S":
        ca = sum_codes(rows, ("210", "214", "218"), "m1")
        ca_prev = sum_codes(rows, ("210", "214", "218"), "m2")
        total_bilan = get(rows, "110", "m3")
        effectif = get(rows, "376", "m1")
    else:
        ca = get(rows, "FJ", "m3")
        ca_prev = get(rows, "FJ", "m4")
        total_bilan = get(rows, "CO", "m3")
        effectif = get(rows, "YP", "m1")

    return dict(
        siren=r.get("siren"),
        date_cloture=r.get("dateCloture"),
        duree_mois=to_int(identite.get("dureeExerciceN")),
        type_bilan=t,
        confidentiality=r.get("confidentiality"),
        naf=identite.get("codeActivite"),
        ca=ca,
        ca_prev=ca_prev,
        total_bilan=total_bilan,
        effectif=effectif,
        date_depot=r.get("dateDepot"),
        updated_at=r.get("updatedAt"),
    )


def extract_file(args):
    """Extract one JSON file; return (name, n_records, complete, DataFrame)."""
    zip_path, name = args
    with zipfile.ZipFile(zip_path) as z:
        text = z.read(name).decode()

    gen = records(text)
    rows, n = [], 0
    while True:
        try:
            r = next(gen)
        except StopIteration as stop:
            complete = stop.value
            break
        n += 1
        row = extract_record(r)
        if row is not None:
            rows.append(row)
    return name, n, complete, pd.DataFrame(rows)


def main(zip_path, out_path, processes=8):
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".json"))

    frames, truncated, total = [], [], 0
    with Pool(processes) as pool:
        tasks = [(zip_path, name) for name in names]
        for i, (name, n, complete, df) in enumerate(
            pool.imap_unordered(extract_file, tasks), 1
        ):
            total += n
            frames.append(df)
            if not complete:
                truncated.append(name)
            if i % 100 == 0:
                print(f"{i}/{len(names)} files, {total:,} records", flush=True)

    df = pd.concat(frames, ignore_index=True)
    for col in ["date_cloture", "date_depot"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce", utc=True)
    for col in ["duree_mois", "ca", "ca_prev", "total_bilan", "effectif"]:
        df[col] = df[col].astype("Int64")
    for col in ["type_bilan", "confidentiality", "naf"]:
        df[col] = df[col].astype("category")
    df = df.sort_values(["siren", "date_cloture", "updated_at"]).reset_index(drop=True)
    df.to_parquet(out_path, index=False)

    print(f"records read: {total:,}; rows written: {len(df):,}")
    print(f"truncated files ({len(truncated)}): {sorted(truncated)}")
    return df


def make_public(df):
    """Make the redistributable extract: deduplicated, with no firm identifiers.

    Keeps the latest refiling of each (siren, date_cloture, type_bilan), then
    drops siren and the filing timestamps. Published results should stay
    aggregate (see data/inpi_france/TERMS.md).
    """
    df = df.sort_values("updated_at").drop_duplicates(
        ["siren", "date_cloture", "type_bilan"], keep="last"
    )
    pub = pd.DataFrame(
        dict(
            year=df["date_cloture"].dt.year.astype("Int16"),
            duree_mois=df["duree_mois"].astype("Int16"),
            type_bilan=df["type_bilan"],
            confidentiality=df["confidentiality"],
            naf=df["naf"],
            ca=df["ca"],
            total_bilan=df["total_bilan"],
            effectif=df["effectif"].astype("Int32"),
        )
    )
    return pub.sort_values(["year", "type_bilan"]).reset_index(drop=True)


if __name__ == "__main__":
    # optional third argument: path for the public, de-identified extract
    df = main(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        pub = make_public(df)
        pub.to_parquet(sys.argv[3], index=False, compression="zstd")
        print(f"public extract: {len(pub):,} rows -> {sys.argv[3]}")
