# Project Lazarus — Sourcing & Data Spec (v4): Post-Tier-0/1 Sequencing

Consolidates the seven open questions after Tier 0/1 into a single build order, with
each decision **locked** (not proposed). Supersedes the ordering in v3; v3's tier
definitions still hold, this re-sequences them and folds in the resolved choices.

Guiding change from v3: **the outcome axis is now the binding constraint.** Tier 0/1
raised the dependency axis; outcomes are still inferred from launch articles — v1's
original problem. So company resolution + the retrospective outcome pass jump ahead
of the remaining Tier 2/3 gap work.

---

## The sequence at a glance

```
Phase 0  Search backbone            Tavily primary + DuckDuckGo fallback   (cheap, unblocks all)
Phase 1  Company entity resolution  #5  (prerequisite for the outcome pass)
Phase 2  Retrospective outcome pass #7  (+ #4 confidence machinery lands here)
Phase 3  Coverage + early typology  Tier 2, honest proxy, #3 flared_out
Phase 4  Thresholds + full typology Tier 3, #2
Phase 5  Exa targeted discovery     #1 part 2 (earns a true white_space)
Side     Research-gated, parallel   #4 validation, #6 incremental scraping
```

Rule threaded throughout: **merge before enrichment, evidence before heuristic,
degrade to a value not a prompt.**

---

## Phase 0 — Search backbone (do first; independent, helps everything)

Replace the rate-limited DuckDuckGo primary with a **tiered search** in
`search.ts`: **Tavily primary, DuckDuckGo fallback**. Tavily is the LangChain-native
tool, returns deduped LLM-ready content in one call, and is a near drop-in for the
existing tool shape. On Tavily error / empty / rate-limit, fall through to DuckDuckGo
so the path has no single point of failure.

- This improves the outcome pass (#7), the reassess tail (#2), and confidence
  (#4 leans on retrieval breadth) — not just discovery.
- Exa is **not** in this fallback chain; it plays a different role (Phase 5).
  Division of labor: **Tavily/DDG for recent-outcome lookups; Exa for
  similarity/discovery.** Don't ask either to do the other's job.

## Phase 1 — Company entity resolution (#5)

Canonicalize companies (Tier 0 did dependencies; companies are still
one-article-one-row). Runs **before** the outcome pass so one retrospective search
runs per canonical company, not per duplicate.

- **ID-first.** Merge on a stable external identifier where one exists: website
  **domain** as the pragmatic canonical key for startups; Wikidata QID or Crunchbase
  permalink for notable ones. Fuzzy matching only as fallback.
- **Fuzzy fallback = Splink on SQLite.** Probabilistic Fellegi-Sunter linkage;
  `SQLiteAPI()` backend (packaged, no extra install) with `rapidfuzz` for the fuzzy
  UDFs. Emits calibrated match scores + clustering. Auto-merge above a score
  threshold; queue ambiguous matches. Backend follows the query layer (SQLite here).
- **Packaging.** Splink is Python, the pipeline is TS, so this is a **separate
  Python step** reading/writing the shared SQLite file at the resolve boundary.
  Stand it up with **uv** (own `pyproject.toml` + `uv.lock`, e.g. `resolve/`),
  `uv add splink rapidfuzz`, pinned `.python-version`. Lint/format with **ruff**
  (covers both), mirrored into CI/pre-commit the way Biome is for TS. TS shells out
  via `uv run`.
- **Live coupling to fix here, not later.** `dedupeFundingRows` keys on
  `(company_uuid, round_name)` in `funding.ts`. Route that key through the
  resolution output so it uses the **canonical** id; ideally the canonical id is the
  only company identifier funding ever sees, so no raw uuid remains at that site to
  key on. Otherwise funding dedupe silently under-merges once companies merge.
- **Environment note (a decision, not a surprise):** the repo becomes genuinely
  two-language; anywhere the pipeline is packaged/deployed now needs a Python
  runtime, not just Node. uv makes provisioning painless — state it explicitly.

## Phase 2 — Retrospective outcome pass (#7)

Establish *what became of each company* from evidence, replacing launch-article
inference for `living_status`, `challenges.outcome`, and the exit fields.

- **Seed, don't rehost.** Ingest a seed list of failed-company **names + minimal
  lifecycle facts** from open compilations (the Kaggle CB Insights post-mortem set,
  Crunchbase-derived sets), provenance recorded, into a `seed_companies` input.
  Facts (who failed, when, rough reason, rough raise) are usable; curated prose and
  the compilation-as-product are not. Treat the seed as a **to-research list**, never
  as finished data. Failory is a **discovery surface for names only** — not a scrape
  target; if their write-ups are ever wanted, that's a permission request.
- **Evidence every field from storable sources:** the company's own defunct site,
  **Wayback captures**, and news articles fetched into `raw_documents` (the reserved
  `outcome` source_type + TTL). Corroborate with cheap deterministic death signals:
  dead domain, **Wayback last-successful-capture date** (a death-date estimate — the
  year the overdue-ness calc needs), funding staleness.
- **Precedence, directional:** evidenced outcome **wins**; the `derive-outcome.ts`
  age/funding heuristic drops to **fallback** — it runs only where the outcome pass
  returned nothing and **never overrides evidence, only fills its absence**. When the
  heuristic fires, record it as **low `confidence` with a note**, so "guessed from
  funding staleness" is queryable and distinct from "a 2021 obituary said so."
- **Scale path (noted, not now):** Crunchbase licensed API / CB Insights licensing
  for clean bulk status data if this becomes central.

### #4 confidence machinery lands with this phase

The outcome and reassess passes are where confidence gets scored.

- **Ship the self-reported label now**, marked self-reported; nothing downstream
  depends on it until validated (expect it to fail — verbalized LLM confidence is
  systematically overconfident).
- **`confidence_score`, computed, is authoritative; the four-level label is derived
  from it** by fixed cutoffs (≥0.8 high, 0.5–0.8 medium, 0.2–0.5 low, else unknown).
  Keep the self-reported label in a **separate column** — never blend an unvalidated
  self-report into the computed score.
- **Signal tiers, cheapest-first:** (1) **cross-source agreement** — primary,
  near-free from the multi-source retrieval already happening; count independent
  corroborating sources, weight by authority, penalize conflict; this also sets the
  **`contested`** flag. (2) **self-consistency** — fallback for uncorroborated *enum*
  judgments: sample N times, mode-share = confidence. (3) **semantic/numeric spread**
  — only for free-text/numeric fields (`outcome_summary`, metric values).
- **As automated as possible:** sampling breadth is a config knob
  (`CONFIDENCE_SAMPLES`, `CONFIDENCE_SCOPE=high_value|all`), default the high-value
  subset (outcome_type, blocking-dependency assessments). **`unknown` is the defined
  terminal state** for every case that can't be scored — emitted with a reason code,
  queryable, never a blank, never a stop. No path waits for a human.

## Phase 3 — Coverage + early gap typology (Tier 2)

- **Coverage via the honest proxy for now.** Discovery is still a TechCrunch scrape
  with post-hoc idea_space assignment, so `searched=1` can't be written truthfully
  yet. Use **observed density** (company/article count per idea_space × era) as the
  confidence proxy, and **label every empty space `unsampled`, never `white_space`** —
  absence resolves to "unknown," not "empty." Fixed era cutoffs (≤2013 / 2014–2020 /
  2021+), first boundary aligned to the cleantech 1.0 bust.
- **Early typology halves** (available without the assessment trajectory):
  `unsampled`, `crowded_solved`, and `flared_out`.
- **#3 `flared_out`** is **trajectory-derived and lives in the gap-typology view
  only** — not a column beside `outcome_type` (it's a pattern over rounds + death, not
  a terminal label; duplicating invites drift). Definition: reached a late/large round
  then `year_defunct` within N years of the last round. **Gate on `confidence`** —
  uneven `funding_rounds` coverage means a call on one known round is a guess.

## Phase 4 — Thresholds + `became_viable` + full typology (Tier 3)

- **`threshold_kind` three-way split** so a blank is never ambiguous:
  `quantitative_with_threshold` (metric/value/unit + source), `quantitative_tbd`
  (measurable, number not set), `qualitative` (no scalar — status-only). Only the
  first yields a `became_viable_date`; qualitative gets improved/unchanged/worsened.
- **Hero dependencies (~10) get hand-curated, cited thresholds** (`source_url` each):
  battery pack $/kWh, PV $/W or LCOE, wind LCOE, green-H₂ $/kg, carbon price, DAC
  $/ton, LDES, etc. TBD/qualitative rows stay **explicitly empty** — any LLM-proposed
  number is stored flagged low-confidence/unverified, never as curated.
- **Forked reassess pass:** hero dependencies pull the *current metric value* from a
  **data feed** (Our World in Data, aggregating IRENA/BNEF) on a scheduled fetch — no
  hallucination surface, fully automatable; the long tail rides the automated
  LLM+search path with the Phase 2 confidence machinery.
- **`became_viable` v1 = earliest observed metric value crossing the fixed
  threshold.** The threshold is the **viability bar** (physics/economics of whether
  the idea works — fixed), the metric moves. Label it as the viability bar, not a
  historical breakeven. **Wright's-Law projection is deferred to v2** (projects a
  crossing year from the learning curve for sparse years / not-yet-viable deps),
  flagged distinctly from observed crossings and carrying a confidence — because
  learning curves aren't deterministic (2022 battery-price plateau).
- **Completes the typology:** adds `lazarus_candidate` and `tested_dead_end` (the
  trajectory-dependent types), and ranks candidates by overdue-ness
  (`became_viable_date` vs. last-attempt-died).

## Phase 5 — Exa targeted discovery (#1 part 2)

The one thing that earns a **true `white_space`**. Query Exa by an idea_space's
description; its neural index returns semantically matched companies, so "searched
this space, found nothing" becomes a real record. Deferred to here because it adds a
second discovery mode and a real `coverage`-writing path.

- Run **per idea_space, on demand**, writing a `coverage` row each time (searched,
  source_count, when). Output feeds **seeds only** — every discovered company still
  goes through resolution (Phase 1) and the evidenced outcome pass (Phase 2).
- Promotes a searched-and-empty space from `unsampled` to `white_space`. Until this
  ships, the Phase 3 honest proxy holds the line (nothing is ever a false gap).
- Exa large free tier (~20k/mo). Caveat: Exa optimizes topical similarity, not
  freshness — keep it to discovery, keep Tavily/DDG for recent-outcome lookups.

## Side quests (research-gated, parallel, non-blocking)

- **#4 validation — self-supervised, automated.** Score computed confidence against
  ground truth the pipeline already trusts: #7's Wayback/dead-domain death signals and
  facts stated by the seed corpora. **Measure discrimination (AUROC / does higher
  score rank more-correct items higher), not calibration.** Human review only on the
  residue with no automatic check. This is what promotes (or discards) the
  self-reported label.
- **#6 incremental scraping.** `SCRAPE_APPEND=1` (matches `SCRAPE_URL` style) copies
  the previous latest scrape forward under a new timestamp and concatenates new rows;
  **dedupe-on-concatenate by `url` via the shared `normalizeUrl`**. `_data_index.csv`
  can be regenerated rather than merged. Year range becomes a batching parameter, not
  a filter. One empirical unknown: **how far back the current TC selectors parse** —
  probe a ~2009 tag page. Rises the moment you commit to the cleantech 1.0 window.

---

## Still genuinely open (decide as you reach them)

- Splink auto-merge score threshold, and what queues for human review (Phase 1).
- `N` in the `flared_out` window (Phase 3).
- The exact hero-dependency list and each cited threshold value (Phase 4).
- Whether to take the Crunchbase/CB Insights licensed scale path (Phase 2).
- TechCrunch archive parse depth (Side / #6).
