# Project Lazarus — Fuzzy Matching Step (Splink) — Spec v1

Extends v5's optional fuzzy tier into a buildable step. **Splink proposes; TypeScript
merges.** Python never mutates `companies`.

## Layout

```
resolve/
  pyproject.toml        # uv-managed
  uv.lock
  .python-version       # pinned
  link.py               # Splink -> merge_candidates
```

`uv add splink rapidfuzz` (rapidfuzz supplies the fuzzy UDFs SQLite lacks; Splink
registers them automatically via `SQLiteAPI()`). Lint/format with **ruff**, config under
`resolve/`, exposed as a `lint:py` npm script mirroring `lint: biome check src`
(decision C1).

**Correction to the original draft:** it said "mirror ruff into CI/pre-commit alongside
Biome." There is no CI, no pre-commit, and no Python linter in the repo today — Biome
runs only via `npm run lint`, so there is nothing to mirror into. `lint:py` matches
current reality. Scaffolding CI/pre-commit is **out of scope for this step** and should
be its own decision.

## Invocation

```json
"scripts": {
  "resolve": "tsx src/steps/resolve-companies.ts",
  "resolve:fuzzy": "tsx src/steps/resolve-fuzzy.ts",
  "resolve:apply": "tsx src/steps/apply-merge-candidates.ts",
  "lint:py": "uv run --project resolve ruff check resolve"
}
```

**`resolve:fuzzy` is a TS wrapper, not a raw `uv run` (decision B1).** It:

1. Materializes a **scratch SQLite** from the latest run dir — `companies.csv` +
   `company_urls.csv` only (decision A1: `<runDir>/resolve.sqlite`, path resolved at
   runtime, already gitignored under `data/output/`, consistent with how `lazarus.db`
   lives in the run dir). The pipeline is CSV-only at the resolve boundary; the real DB
   isn't built until the final `build` step, so this scratch file is a **transport
   format, not a store**.
2. Probes for `uv` (`which uv` / catch ENOENT).
3. Spawns `uv run --project resolve resolve/link.py --db <runDir>/resolve.sqlite`.
4. Exports `merge_candidates` back out to **`<runDir>/merge_candidates.csv`** — the
   scratch DB is ephemeral, so the audit trail must persist in the run dir in the same
   format as every other output.

**Exit-code contract (do not collapse these two cases):**

| Condition | Behaviour |
| --- | --- |
| `uv` not installed | warn, **exit 0** — graceful skip, same posture as a missing `TAVILY_API_KEY` (costs merge recall, not a run) |
| `uv` present, `link.py` fails | **non-zero exit**, real error |

A crashed linker must never look like a graceful skip.

`uv run --project` provisions the venv and deps on first call — no separate setup.
The **SQLite file is the only thing crossing the language boundary.**

## Contract: `merge_candidates`

```sql
CREATE TABLE merge_candidates (
  id           INTEGER PRIMARY KEY,
  company_a    TEXT NOT NULL,   -- canonical uuid
  company_b    TEXT NOT NULL,   -- canonical uuid
  match_score  REAL NOT NULL,   -- Splink probability, 0-1
  run_at       TEXT NOT NULL,
  applied      INTEGER DEFAULT 0
);
```

- **Python writes rows and stops.** No merging, no writes to `companies`.
- **TS (`resolve:apply`) reads them**, applies the score threshold, and merges through
  the *same code path* `resolve` already uses — so `merged_from` auditing, canonical
  uuid selection, and every merge invariant stay in one implementation. Two things that
  can merge companies will drift; this prevents that.
- Threshold is a **TS-side config knob** (`FUZZY_MERGE_THRESHOLD`), retunable without
  re-running Python.

### Two implementation notes (surfaced by repo exploration)

- **There is no force-merge API, and none should be added.** `resolveCompanies` groups
  purely by `canonicalKey`, and two companies Splink flags as duplicates have *different*
  keys by construction — that is precisely why ID-first left them apart. So
  `resolve:apply` **clusters the scored uuid pairs (union-find)** and re-merges each
  cluster through the existing `mergeGroup` + `mergeCompanyUrls` + `remapCompanyUuids`.
  This is the propose/apply contract working as intended: no parallel merge path.
- **`merged_from` must accumulate — this is a latent bug in the existing resolve step.**
  `mergeGroup` currently *overwrites* `merged_from` with only the uuids absorbed in that
  call. Applying fuzzy merges to rows that are already canonical would silently drop the
  earlier provenance. Fix: union prior `merged_from` with newly absorbed uuids. Small and
  test-covered, and **worth doing regardless of whether Splink ever ships**, since it
  would bite any second merge pass.

## Python side (`link.py`)

- `SQLiteAPI()` linker over `companies` (+ `company_urls` for extra comparisons).
- Blocking on normalized name prefix / founding year to keep pair counts sane.
- Comparisons: normalized name (Jaro-Winkler via rapidfuzz), founding year, domain,
  location.
- Unsupervised estimation (no labels needed); writes scored pairs above a low floor
  (e.g. 0.5) so the TS threshold does the real filtering.
- Idempotent: clear prior unapplied rows for the run, or key on `run_at`.

## Sequencing constraint (important)

**Optional to build, not free to run late.** If run at all, it executes at the
**resolve boundary — after `resolve`, before enrichment (Phase 2a) and step2.**
Merging after enrichment violates merge-before-enrichment and leaves per-duplicate
enrichment to reconcile. Introducing it later means **re-running downstream phases for
affected companies**.

## Graceful degradation

The orchestrator probes for `uv` (`which uv` / catch ENOENT) and **skips the step with
a warning** if absent. Same posture as a missing `TAVILY_API_KEY`: absence costs merge
recall, not a broken run. The pipeline and DB build never depend on this step.

## Gating (unchanged from v5)

Build/run only if the measurements justify it, **read in order**: `url_type` coverage
first (sparse keys → fix key coverage upstream, near-miss count is uninterpretable),
then the near-miss count from `npm run resolve`.

## Still open

- `FUZZY_MERGE_THRESHOLD` value, and whether a mid band (e.g. 0.7–0.9) **queues for
  review** instead of auto-merging — `merge_candidates.applied` already supports this.
- Blocking rules, tuned to real near-miss shapes.
- Whether `company_urls` comparisons add enough signal to be worth the join.
