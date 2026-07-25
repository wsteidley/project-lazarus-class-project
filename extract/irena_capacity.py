"""Read cumulative onshore wind capacity out of the IRENA Stats Tool workbook.

Python's half of the irena-capacity extractor. It exists only because the source is a
.xlsb (binary Excel), which nothing on the TypeScript side reads. Its job is deliberately
small: get the numbers out, apply the two filters that are easy to get wrong, and write a
flat year,capacity_mw CSV. Everything about the derived schema -- units, basis, source
stamps, rounding, sort order -- belongs to src/lib/extractors/irena-capacity.ts.

Exit-code contract: 0 only when a plausible series was written; non-zero on any
structural surprise. A silently empty or half-empty series would be committed as data.

Usage:
    uv run --project extract extract/irena_capacity.py \
        --xlsb data/sources/irena/IRENA_Stats_Tool_v2.xlsb --out /tmp/irena.csv
"""

import argparse
import sys

import pandas as pd

SHEET = "Data"
# The sheet's real header is on row 6 (1-based); rows above it are the workbook's own
# navigation chrome, and the two rows below it repeat the indicator names as a sub-header.
HEADER_ROW = 5
JUNK_REGION = "Indicator"

CAPACITY_COLUMN = "Electricity Installed Capacity (MW)"
TECHNOLOGY = "Onshore wind energy"

# THE GOTCHA. Capacity is reported per producer type. The "All types" rows carry only the
# finance indicators and have a NULL capacity -- filtering to them, or failing to sum the
# grid types, reads global wind capacity as zero, which looks like a plausible number
# rather than an error. Sum on-grid + off-grid; never include "All types".
PRODUCER_TYPES = ("On-grid electricity", "Off-grid electricity")

# Sanity floor. Global onshore wind passed 1 TW (1,000,000 MW) in 2024, so a correct
# extraction is orders of magnitude above this. The check is here to catch the failure
# mode above -- a filter that matches nothing or nearly nothing -- not to validate the
# data itself.
MIN_PLAUSIBLE_LATEST_MW = 100_000


def extract(xlsb_path: str) -> pd.DataFrame:
    frame = pd.read_excel(xlsb_path, sheet_name=SHEET, header=HEADER_ROW, engine="pyxlsb")

    missing = [
        column
        for column in ("Region", "Technology", "Producer Type", "Year", CAPACITY_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise SystemExit(
            f"{xlsb_path}: sheet '{SHEET}' is missing column(s) {missing} -- "
            f"the workbook's layout changed; re-check HEADER_ROW and the column names"
        )

    frame = frame[frame["Region"].notna() & (frame["Region"] != JUNK_REGION)]
    onshore = frame[frame["Technology"] == TECHNOLOGY]
    if onshore.empty:
        raise SystemExit(
            f"no rows with Technology == '{TECHNOLOGY}' -- IRENA renamed the technology"
        )

    # Assert the gotcha still holds rather than assuming it. If a future vintage starts
    # populating capacity on "All types", summing the grid types would double-count and
    # this run must stop rather than quietly halve or double the series.
    all_types_capacity = onshore[onshore["Producer Type"] == "All types"][CAPACITY_COLUMN]
    if all_types_capacity.notna().any():
        raise SystemExit(
            "'All types' rows now carry capacity values -- they were finance-only. "
            "Re-derive the producer-type filter before trusting this series."
        )

    grid = onshore[onshore["Producer Type"].isin(PRODUCER_TYPES)]
    series = grid.groupby("Year")[CAPACITY_COLUMN].sum().sort_index()
    if series.empty:
        raise SystemExit(f"no rows with Producer Type in {PRODUCER_TYPES}")

    latest = float(series.iloc[-1])
    if latest < MIN_PLAUSIBLE_LATEST_MW:
        raise SystemExit(
            f"latest onshore wind capacity is {latest:,.0f} MW, below the "
            f"{MIN_PLAUSIBLE_LATEST_MW:,} MW sanity floor -- a filter matched the wrong rows"
        )

    return pd.DataFrame(
        {"year": series.index.astype(int), "capacity_mw": series.to_numpy()}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsb", required=True, help="path to IRENA_Stats_Tool_v2.xlsb")
    parser.add_argument("--out", required=True, help="path to write year,capacity_mw CSV")
    args = parser.parse_args()

    table = extract(args.xlsb)
    table.to_csv(args.out, index=False)
    print(
        f"irena_capacity: {len(table)} years "
        f"({table['year'].min()}-{table['year'].max()}) -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
