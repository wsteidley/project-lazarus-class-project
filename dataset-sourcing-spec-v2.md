# Project Lazarus — Sourcing & Data Spec (v2)

Iterates sourcing v1 and extends schema v4. Purpose of this round: make **gaps
visible and typed**, and make absence trustworthy. Everything here is dataset-side
— the app/exploration layer is a separate spec, deferred.

Guiding idea: a gap is only real if the data records what was *searched for and not
found*. Coverage is therefore a first-class citizen, not a footnote.

---

## Part A — Sourcing pipeline refinements

Carries forward the three from the v1 review, tightened:

### A1. Record linkage at enrichment (external entities)
The pre-pass dedups TechCrunch rows against each other using data already
extracted. But the moment the outcome/funding pass hits Crunchbase or Wikipedia,
you must match your canonical company to *their* entity — external record linkage,
which by necessity happens at enrichment time. This does not violate
"merge before enrichment" (no new rows are created to merge later), but it's a
distinct step with its own failure mode: **linking to the wrong external entity
silently poisons the exit fields**.
- Store the resolved external IDs on the company (`crunchbase_id`, `wikipedia_url`,
  etc.) so links are auditable and re-runnable.
- Handle the rename/rebrand-on-acquisition case: a company may be known to the
  external source under a different name than your TC-derived key. This is the one
  place a targeted lookup happens *during* resolution.

### A2. Two-tier cache policy
Discovery text and outcome results have opposite volatility.
- **Discovery text (launch articles): immutable** — cache indefinitely.
- **Outcome / reassessment results: perishable** — a company alive today may fold
  next year, so these get a TTL or an explicit refresh flag.
- `raw_documents` already has `fetched_at`; add `source_type` and a freshness
  policy that decides which rows serve from cache forever and which expire.

### A3. Coverage tracking (promoted from open-decision to requirement)
Absence is only interpretable if you looked. Record, per `idea_space` × era, that a
search happened and how much — so "genuine gap" and "unsampled" are distinguishable.
Discovery must **deliberately sample the present era** in each idea_space, not only
cleantech 1.0, or the default failure is a *false gap* (see Part B / gap typology).

---

## Part B — Gap-analytics data layer (schema additions, extends v4 → v5)

New structure and derived views that let gaps be seen and their *type* known. Most
of this is derivation over data v4 already produces; only a few small tables/fields
are new.

### B1. Viability threshold on dependencies → a "became viable" date
Give each dependency the value at which it stops being blocking, so improvement gets
a *date* instead of a vibe.
- Add to `idea_dependencies`: `threshold_metric`, `threshold_value`,
  `threshold_unit` (e.g. battery pack price, 100, USD/kWh).
- Derive `became_viable_date`: the earliest `assessed_on` in
  `dependency_assessments` whose `metric_value` crosses the threshold.
- Payoff: rank Lazarus candidates by *how overdue* they are —
  `became_viable_date` vs. the year the last attempt died.

### B2. Coverage as a first-class table
```sql
CREATE TABLE coverage (
  id            INTEGER PRIMARY KEY,
  idea_space_id INTEGER REFERENCES idea_spaces(id),
  era           TEXT,        -- e.g. '2006-2013', '2014-2020', '2021-present'
  searched      INTEGER,     -- 0/1
  source_count  INTEGER,     -- how much we actually looked
  last_searched TEXT,        -- ISO
  confidence    TEXT         -- 'high' | 'medium' | 'low'
);
```
This is the guardrail that keeps a real dead-end distinct from an unsampled hole.

### B3. Contested / disagreement state
When sources conflict or the model is unsure, store that rather than forcing a
verdict.
- Add `confidence` and `contested` (0/1) + `contested_note` to the outcome-bearing
  rows: `challenges`, `dependency_assessments`, and alongside `outcome_type`
  (which already carries a rationale).
- Makes "show me contested outcomes" and "show me low-confidence assessments"
  queryable — epistemic gaps surfaced next to substantive ones. Quietly banks part
  of the verification workstream.

### B4. Macro-event / shock entity
Separate macro-timing failures from idiosyncratic ones.
```sql
CREATE TABLE events (
  id          INTEGER PRIMARY KEY,
  name        TEXT,        -- '2011 solar import tariff', '2008 financial crisis'
  type        TEXT,        -- 'policy' | 'market' | 'macro' | 'commodity' | 'other'
  start_date  TEXT,
  end_date    TEXT,
  description TEXT
);
CREATE TABLE company_events (
  company_id INTEGER REFERENCES companies(id),
  event_id   INTEGER REFERENCES events(id),
  role       TEXT,        -- 'killed_by' | 'affected' | 'enabled'
  PRIMARY KEY (company_id, event_id)
);
```
Turns "died on timing" into "died into the 2011 tariff" — a queryable cluster, and
a way to spot gaps a shock created that nobody revisited once conditions changed.

### B5. Gap typology view (the payoff — pure derivation, no new extraction)
A materialized view classifying each `idea_space` from two axes it already has —
*attempt density* (how many tried, when) × *dependency trajectory* (did the blocker
improve) — gated by coverage confidence:

```
white_space        dependency improved, no attempts on record, coverage=high
lazarus_candidate  attempted, failed on a blocker, blocker now past threshold,
                   no active retry
tested_dead_end    repeated attempts, blocker never crossed threshold  (a "not yet")
flared_out         brief success then collapse (outcome_type flags this)
crowded_solved     active successes present
unsampled          coverage low/absent  -> NOT a gap, a data hole
```
Every classification is computable at analysis time. `unsampled` is the critical
one: it prevents a false gap from masquerading as white space.

---

## Part C — New/changed DDL summary

New tables: `coverage`, `events`, `company_events`.
New columns: `idea_dependencies.threshold_*`, `dependency_assessments` /
`challenges` `.confidence` + `.contested` + `.contested_note`,
`companies.crunchbase_id` / `.wikipedia_url`, `raw_documents.source_type`.
New views: `gap_typology`, `became_viable` (per dependency/idea_space).
No change to existing v4 tables' semantics.

## Part D — Open decisions

- **Threshold values (B1):** where do they come from — a curated table per
  dependency category, or LLM-proposed then reviewed? These are the input that
  makes "became viable" trustworthy, so likely curated for the high-value ones
  (battery, solar, compute) and left null elsewhere.
- **Era boundaries (B2/B5):** fixed cutoffs (2013 / 2020) or per-idea_space? Fixed
  is simpler and probably enough.
- **Coverage granularity:** per idea_space × era is the proposed grain; per sector
  may be enough to start.
- **Embedding index:** an optional build artifact consumed by the app phase.
  Deferred to the exploration spec — noted here only so the pipeline leaves room
  for it.
