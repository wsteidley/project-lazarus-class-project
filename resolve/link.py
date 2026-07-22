"""Splink fuzzy company de-duplication for Project Lazarus.

Reads a scratch SQLite database (materialized by ``src/steps/resolve-fuzzy.ts`` from the
run directory's CSVs) that holds one precomputed ``companies`` table, runs unsupervised
probabilistic linkage, and writes scored duplicate pairs into a ``merge_candidates``
table in the *same* database. It never touches ``companies`` and never merges anything —
TypeScript's ``resolve:apply`` reads ``merge_candidates`` and applies the merges.

The SQLite file is the only thing that crosses the language boundary. All join keys
(``normalized_name``, ``name_prefix``, ``domain``) are precomputed on the TypeScript side
with the resolver's own normalizers, so this script never re-derives them.
"""

from __future__ import annotations

import argparse
import datetime
import sqlite3

import pandas as pd
import splink.comparison_library as cl
from splink import Linker, SettingsCreator, SQLiteAPI, block_on

# Splink's match_probability floor for a pair to be written at all. Deliberately low: the
# real cutoff is the TypeScript-side FUZZY_MERGE_THRESHOLD, so tuning it never re-runs Python.
DEFAULT_FLOOR = 0.5

MERGE_CANDIDATES_DDL = """
CREATE TABLE IF NOT EXISTS merge_candidates (
  id           INTEGER PRIMARY KEY,
  company_a    TEXT NOT NULL,
  company_b    TEXT NOT NULL,
  match_score  REAL NOT NULL,
  run_at       TEXT NOT NULL,
  applied      INTEGER DEFAULT 0
)
"""


def build_settings() -> SettingsCreator:
    """Dedupe settings: block on a name prefix or founding year to keep pair counts sane,
    then compare on name (Jaro-Winkler), year, domain, and location."""
    return SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name="uuid",
        blocking_rules_to_generate_predictions=[
            block_on("name_prefix"),
            block_on("year_founded"),
        ],
        comparisons=[
            cl.JaroWinklerAtThresholds("normalized_name", [0.9, 0.7]),
            cl.ExactMatch("year_founded"),
            cl.ExactMatch("domain"),
            cl.ExactMatch("location"),
        ],
        retain_intermediate_calculation_columns=False,
    )


def estimate(linker: Linker) -> None:
    """Unsupervised parameter estimation — no labels required."""
    linker.training.estimate_probability_two_random_records_match(
        [block_on("normalized_name")], recall=0.7
    )
    linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000)
    for rule in (block_on("name_prefix"), block_on("year_founded")):
        linker.training.estimate_parameters_using_expectation_maximisation(rule)


def write_candidates(con: sqlite3.Connection, predictions: pd.DataFrame, run_at: str) -> int:
    """Replace the prior unapplied candidates with this run's scored pairs. Rows already
    marked applied=1 are left untouched so the audit trail is not rewritten."""
    con.execute(MERGE_CANDIDATES_DDL)
    con.execute("DELETE FROM merge_candidates WHERE applied = 0")
    rows = [
        (str(row["uuid_l"]), str(row["uuid_r"]), float(row["match_probability"]), run_at)
        for _, row in predictions.iterrows()
    ]
    con.executemany(
        "INSERT INTO merge_candidates (company_a, company_b, match_score, run_at, applied) "
        "VALUES (?, ?, ?, ?, 0)",
        rows,
    )
    con.commit()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Splink fuzzy company matching")
    parser.add_argument("--db", required=True, help="Path to the scratch SQLite database")
    parser.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    try:
        companies = pd.read_sql("SELECT * FROM companies", con)
        run_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

        if len(companies) < 2:
            # Nothing to compare — still (re)create an empty table so the export step
            # always finds merge_candidates and never mistakes absence for a crash.
            con.execute(MERGE_CANDIDATES_DDL)
            con.execute("DELETE FROM merge_candidates WHERE applied = 0")
            con.commit()
            print("resolve/link.py: fewer than 2 companies — no candidates written")
            return

        db_api = SQLiteAPI(con)
        linker = Linker(companies, build_settings(), db_api=db_api)
        estimate(linker)
        predictions = linker.inference.predict(
            threshold_match_probability=args.floor
        ).as_pandas_dataframe()

        written = write_candidates(con, predictions, run_at)
        print(f"resolve/link.py: wrote {written} merge candidates (floor {args.floor})")
    finally:
        con.close()


if __name__ == "__main__":
    main()
