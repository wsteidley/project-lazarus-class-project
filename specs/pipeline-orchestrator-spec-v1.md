# Project Lazarus — Pipeline Orchestrator Spec (v1)

One entry point that encodes the stage order in code, so the correctness rules the
architecture depends on are enforced by the machine rather than by whoever is typing.

## Why

There are ~14 npm scripts whose **order is load-bearing**. The ordering rules *are* the
correctness guarantees — merge before enrichment, enrich before search, evidence before
heuristic — and running them out of order does not crash. It silently produces wrong
precedence: enrichment applied per-duplicate, a heuristic overwriting evidence, an
`outcome_type` derived before funding existed. Those are invisible in the output and
expensive to detect later.

## The canonical sequence

Each edge exists for a stated reason. The orchestrator refuses to violate them.

| # | Stage | Script | Invariant it upholds |
| --- | --- | --- | --- |
| 1 | scrape | `step0` | — corpus entry |
| 2 | extract company | `step1` | needs scraped articles |
| 3 | extract dependencies | `step1b` | needs company rows |
| 4 | resolve dependencies | `step1c` | canonical dependency list before any assessment |
| 5 | resolve companies | `resolve` | **merge before enrichment** |
| 5a | fuzzy propose | `resolve:fuzzy` | optional; **must sit at the resolve boundary** |
| 5b | fuzzy apply | `resolve:apply` | optional; merges via the existing merge path |
| 6 | enrich (static) | `enrich` | **enrich before search** — deterministic, no API cost |
| 7 | funding | `step2` | after canonical uuids exist |
| 8 | funding cleanup | `step3` | deterministic dedupe/standardize |
| 9 | outcome pass | `outcome-pass` | after enrich, so it fills gaps not duplicates |
| 10 | derive outcome_type | `derive` | **after 7–9**: needs `total_raised` + evidenced outcome |
| 11 | reassess | `reassess` | needs canonical dependencies (4) |
| 12 | build DB | `build` | terminal; consumes everything |

`lint:py`, `clean`, `build-sources`, `test` are **not** pipeline stages — they stay
standalone.

## Interface

```
npm run pipeline                  # full run, fresh run dir
npm run pipeline -- --from enrich # resume from a stage in the latest run dir
npm run pipeline -- --only outcome-pass
npm run pipeline -- --through step3
npm run pipeline -- --dry-run     # print the plan, touch nothing
```

- `--from` / `--only` / `--through` operate on the **latest run dir** unless
  `--run-dir <path>` is given.
- `--dry-run` prints the resolved stage list, the run dir, and each stage's declared
  inputs/outputs. No side effects, no API calls.

## Run manifest (the ordering mechanism)

The orchestrator writes `<runDir>/manifest.json`, updated after each stage:

```json
{
  "run_id": "2026-07-21T14-03-11Z",
  "stages": {
    "step1":   { "status": "ok", "started_at": "…", "finished_at": "…", "rows_out": 812 },
    "resolve": { "status": "ok", "…": "…", "merges": 37, "near_misses": 4 },
    "enrich":  { "status": "skipped", "reason": "…" }
  },
  "config": { "PROCESSING_LIMIT": 50, "CONFIDENCE_SCOPE": "high_value" }
}
```

Rules:
- A stage **refuses to run** if any prerequisite is not `ok` in this run dir's manifest.
  That is the enforcement — not file-existence guessing.
- Re-running a completed stage **invalidates every downstream stage** (marked `stale`),
  so a later `--from` can't silently mix a fresh upstream with stale downstream output.
- The manifest records the **config each stage ran under**, so a run is reproducible and
  a mixed-config run dir is detectable.

## Failure and degradation

- **Fail fast by default.** A stage returning non-zero halts the pipeline; the manifest
  records `failed` with the stage's stderr tail. Resume with `--from <that stage>`.
- **Optional stages degrade, they don't fail.** `resolve:fuzzy` / `resolve:apply` warn
  and record `skipped` when `uv` is absent (exit 0). A *present-but-failed* `link.py` is
  a real failure — see the fuzzy spec's exit-code contract.
- **`--continue-on-error`** is available for exploratory runs but marks the manifest
  `partial`, and `build` refuses to run from a `partial` manifest without `--force`.
- Per-stage retry for network-bound stages (`outcome-pass`, `reassess`, `step2`) is a
  config knob, not a default.

## Cost control

The LLM/search stages (`step1`, `step1b`, `outcome-pass`, `reassess`, `step2`) dominate
runtime and spend, so:
- `PROCESSING_LIMIT` and `CONFIDENCE_SCOPE` are surfaced as pipeline flags and echoed in
  the plan output.
- `--dry-run` reports **which stages will make API calls** and the row counts they'd
  process, so a full run is never a surprise.
- A `--smoke` preset (small `PROCESSING_LIMIT`, `CONFIDENCE_SAMPLES=0`) for verifying
  wiring end-to-end cheaply.

## Logging

One summary line per stage — name, duration, rows in/out, and its own headline metric
(`resolve`: merges + near-misses; `enrich`: domain vs. name-join match rate;
`outcome-pass`: evidenced vs. fallback counts; `reassess`: assessments written). These
are the numbers that decide the outstanding builds, so the pipeline should surface them
without a separate query.

## Open decisions

- Whether `build` auto-runs at the end of a full pipeline run or stays explicit.
- Whether a fresh `npm run pipeline` always creates a new run dir, or resumes an
  incomplete latest one by default.
- Whether stale-downstream invalidation blocks (`--force` to override) or only warns.
