# Project Lazarus — Sourcing & Data Spec (v5): Enrichment-First Sequencing

Supersedes v4's ordering. Same locked decisions; this folds in the enrichment-data
layer (item 8), splits the outcome axis into a deterministic-then-search sequence, and
demotes the fuzzy-matching toolchain to an **optional final step** the rest of the
build never depends on.

## What changed from v4

- **Item 8 (local enrichment data) is Phase 2's deterministic first half.** Local
  failure/enrichment data is consulted **before** search; the search-based outcome
  pass becomes the second half that fills gaps and freshens volatile fields.
- **Precedence splits by field volatility**, not one uniform rule.
- **Name-only enrichment matches are seeds, never evidenced.**
- **Phase 1 resolution is ID-first (built, TypeScript).** The probabilistic/Splink
  tier — and the Python toolchain it needs — moves to an **optional final step**,
  gated on real near-miss counts. The pipeline and DB build run without it.
- **The funding coupling is closed structurally** (`resolve` before step2); the old
  "route the key through canonical id" caveat is removed — no raw uuid exists to
  mis-key on.
- **The two coverage measurements are one query** (shared `normalizeDomain`).
- **Item 4 validation is gated on item 7 having *run*.**

---

## The sequence at a glance

```
Phase 0   Search backbone           Tavily primary + DuckDuckGo fallback       BUILT
Phase 1   Entity resolution         ID-first over company_urls                 BUILT
Phase 2a  Enrichment (deterministic) local failure/enrichment data, no API     data+loader BUILT, step unwired
Phase 2b  Outcome pass (search)     fills gaps + freshens; + #4 confidence      not built  ← binding constraint
Phase 3   Coverage + early typology Tier 2 honest proxy, #3 flared_out          not built
Phase 4   Thresholds + full typology Tier 3, #2                                 columns built, values empty
Phase 5   Exa targeted discovery    earns a true white_space                    not built
Side      Research-gated, parallel  #4 validation (gated on 2b output), #6      not built
Phase 1b  Fuzzy matching (optional)  uv + ruff + Splink, at the resolve boundary  NEXT ITERATION — separate spec
```

Rules threaded throughout: **merge before enrichment · enrich before search ·
evidence-vs-static by field volatility · degrade to a value, never a prompt.**

---

## Phase 0 — Search backbone — BUILT

Tiered search in `gatherSearchContext`: **Tavily primary, DuckDuckGo fallback** on
error/empty/rate-limit. Adds `TAVILY_API_KEY` as an operational dependency — absent, it
degrades silently to DuckDuckGo (costs recall, never breaks a run). Exa is **not** in
this chain (Phase 5). Division of labor: Tavily/DDG for recent-outcome lookups, Exa for
similarity/discovery.

## Phase 1 — Company entity resolution (ID-first) — BUILT

`resolve` merges over the typed `company_urls` table: `website` domain → `crunchbase`
→ `wikipedia` → normalized `name_year`. Identity as typed rows (not per-source columns),
so LinkedIn / Wayback / dead-site links are new rows, not migrations — and `archive`
URLs are already present for Phase 2b's death signal.

- **Funding coupling closed structurally:** `resolve` runs before step2, so funding is
  only ever keyed on canonical uuids. No caveat remains.
- **Fuzzy tier deferred** to the optional final step; every run prints merges-by-tier
  plus a **near-miss report** (distinct canonical companies sharing a normalized name
  that ID-first refused to merge) — the evidence that decides whether Splink is worth it.

## Phase 2a — Enrichment from local data (deterministic, no API) — data BUILT, step unwired

Wire the enrich step (item 8) **right after `resolve`, before any search.** For each
canonical company, look it up in the normalized local sources by domain → external id →
name; fill `living_status`, `exit_type`, total funding, round count, founding/defunct
years, and (from the failure set) a failure reason already mapped to the `CHALLENGE`
vocab. Mark every filled field with its source and join type.

- **This is the cheap first half of the outcome axis** — a zero-API deterministic
  partial answer for matched companies. It makes Phase 2b more specific and cheaper.
- **Join strength governs trust:** the **domain/external-id strong-join** is
  deterministic (same `normalizeDomain` the resolver uses, so it lines up with
  `company_urls`). The **name-only join** (the failure set) is lower confidence.
- **Name-only matches are seeds, never evidenced.** A name-only `challenges` row is a
  strong *lead* to be corroborated by Phase 2b — the same seed-not-truth rule the
  compilations get. It lands at lower `confidence` and feeds the #4 cross-source
  machinery; it is never written as ground truth.

## Phase 2b — Retrospective outcome pass (search) — not built · **binding constraint**

Runs after 2a, **gated to fill and freshen, not to redo.** Where 2a already supplied a
field, search corroborates or updates per the precedence rule below; where 2a left a
gap, search fills it. Evidence every field from storable sources — defunct site, Wayback
captures, news fetched into `raw_documents` (reserved `outcome` source_type + TTL) —
corroborated by cheap death signals: dead domain, **Wayback last-capture date** (the
death-year estimate the overdue-ness calc needs), funding staleness.

**Precedence splits by field volatility** (this replaces v4's uniform "evidence wins"):

- **Volatile fields — evidence wins.** `living_status`, exit/acquisition state: a static
  snapshot goes stale, so a fresh evidenced outcome overrides it.
- **Immutable facts — strong-join static wins.** Founding year, defunct year from a
  strong-join structured source are more reliable than an LLM re-derivation from a
  fetched article; search must not "correct" a solid year with a hallucinated one.
- **`derive-outcome.ts` heuristic stays the last fallback** — fills only where both
  enrichment and search returned nothing, never overrides either, and records itself as
  low `confidence` with a note.

### #4 confidence machinery lands here

Computed `confidence_score` is authoritative; the four-level label is derived from it
(≥0.8 high, 0.5–0.8 medium, 0.2–0.5 low, else unknown). Self-reported label stays a
**separate column**, never blended. Signal tiers cheapest-first: **cross-source
agreement** (primary, near-free; also sets `contested`) → **self-consistency** (enum
fallback) → **semantic/numeric spread** (free-text/metrics). Sampling breadth is a knob
(`CONFIDENCE_SAMPLES`, `CONFIDENCE_SCOPE=high_value|all`, default high-value). **`unknown`
is the defined terminal state** with a reason code — queryable, never blank, never a stop.

## Phase 3 — Coverage + early gap typology (Tier 2) — not built

Honest proxy: **observed density** (company/article count per idea_space × era) as the
confidence signal; fixed eras (≤2013 / 2014–2020 / 2021+, first boundary on the
cleantech 1.0 bust); **every empty space labeled `unsampled`, never `white_space`.**
Early typology halves: `unsampled`, `crowded_solved`, `flared_out`.

**#3 `flared_out`** lives in the **gap-typology view only** (a pattern over rounds +
death, not a terminal label). Definition: reached a late/large round then `year_defunct`
within `N` years of the last round, **gated on `confidence`**. Open: `N`, and "late/large".

## Phase 4 — Thresholds + `became_viable` + full typology (Tier 3) — columns built

`threshold_kind` split: `quantitative_with_threshold` / `quantitative_tbd` /
`qualitative`; only the first yields `became_viable_date`. ~10 hero dependencies get
hand-curated cited thresholds pulling current values from a **data feed** (Our World in
Data / IRENA / BNEF); the tail rides LLM+search. `became_viable` v1 = earliest *observed*
crossing of the **fixed viability bar**; Wright's-Law projection is v2 (flagged distinct,
carries confidence). Completes the typology: `lazarus_candidate`, `tested_dead_end`,
ranked by overdue-ness. Open: the hero list + cited values.

## Phase 5 — Exa targeted discovery — not built

Query Exa per idea_space from its description; write a real `coverage` row and promote
searched-and-empty spaces to a true `white_space`. Output feeds **seeds only** — every
find still passes through resolution (Phase 1), enrichment (2a), and the outcome pass
(2b). Exa optimizes similarity, not freshness — discovery only.

## Side quests (research-gated, parallel, non-blocking)

- **#4 validation — self-supervised, and gated on Phase 2b having *run*.** Its ground
  truth is 2b's Wayback/dead-domain death signals and seed-corpus facts, so the test
  can't produce a verdict until the outcome pass has populated evidenced outcomes to
  score against. Measure **discrimination (AUROC), not calibration**; human review only
  on the unresolvable residue. This promotes or discards the self-reported label.
- **#6 incremental scraping.** `SCRAPE_APPEND=1`, dedupe-on-concatenate by `url` via
  shared `normalizeUrl`, `_data_index.csv` regenerated. Open: how far back current TC
  selectors parse (probe a ~2009 page). Rises with the cleantech 1.0 commitment.

## Phase 1b — fuzzy company matching (uv + ruff + Splink)

**Deferred to the next iteration; a separate spec follows.** Do not build it from this
document — it is recorded here only so the sequence is complete and the placement is
unambiguous.

**Optional and non-blocking: the DB build and the entire pipeline run to completion
whether or not this step is ever executed.** It only collapses *additional* near-miss
duplicates that ID-first left separate; skipping it costs some merge recall, nothing
else.

**"Optional" means never required to build or run — it does not mean "runs last."**
If executed at all, it runs at the **resolve boundary: after Phase 1, before Phase 2a
enrichment and step2.** Merging after enrichment violates merge-before-enrichment and
leaves per-duplicate enrichment to reconcile; bolting it onto the end of the pipeline
later means re-running downstream phases for affected companies.

- **Gated on the near-miss count** from Phase 1. Near-zero → ID-first was sufficient,
  don't build it, don't take on Python. Large → the fuzzy tier earns its keep.
- **Splink on SQLite:** `SQLiteAPI()` (packaged, no extra install) + `rapidfuzz` for the
  fuzzy UDFs; Fellegi-Sunter probabilistic linkage with calibrated match scores and
  clustering. Reads/writes the shared SQLite file at a separate boundary.
- **Toolchain (introduced only by this step):** a **uv** project (own `pyproject.toml`
  + `uv.lock`, pinned `.python-version`), `uv add splink rapidfuzz`, **ruff** for
  lint+format mirrored into CI/pre-commit the way Biome is for TS. `uv` is not currently
  installed — installing it is part of *this* step, not a prerequisite for anything
  before it.
- **Consequence to accept only if built:** the repo becomes two-language and deploys
  need a Python runtime. Because the step is optional, that tax is paid only if the
  near-miss numbers justify it.

---

## Measurements to take on the first real run (read in order)

One query answers three build decisions — **read `url_type` coverage first, then the
near-miss count; never in parallel:**

1. `SELECT url_type, COUNT(*) FROM company_urls GROUP BY url_type` — key-population
   health. If identity URLs are sparse, resolution is running on the weak `name_year`
   tier **and** enrichment's strong-join reach is low (same key). A low near-miss count
   then means "ID-first barely ran," not "ID-first was sufficient" — fix key coverage
   (more Crunchbase enrichment / better website extraction) *before* asking the Splink
   question.
2. Near-miss count from `npm run resolve` — only meaningful once (1) shows healthy
   identity-URL coverage. Healthy keys + high near-misses = the real Splink signal.

## Still genuinely open (decide on reaching them)

- Splink auto-merge threshold and what queues for human review (nothing queues today).
- `N` and "late/large" for `flared_out` (Phase 3).
- Hero-dependency list + cited threshold values (Phase 4).
- Licensed scale path (Crunchbase/CB Insights) if item 7 becomes central.
- The unconverted **Excel failure set**: low priority if it overlaps the normalized
  failure source, rises with item 7 if it covers *different* companies (needs an `.xlsx`
  reader). The anonymized-name prediction set stays set aside — no keys, unjoinable.
