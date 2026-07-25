# Project Lazarus — Sourcing & Data Spec (v3)

Revises v2 in response to build feedback. No new capabilities — this round fixes
sequencing, defines `confidence` properly, and corrects create-vs-add wording so the
doc is buildable in order.

## What's new in v3 (diff from v2)

1. **`confidence` is now a defined scale**, not a loose "flag" (see below).
2. **Part B is resequenced into build tiers** with explicit dependencies. The
   B1–B5 numbering in v2 read like a build order and wasn't one.
3. **The `dependency_assessments` pass is named as an explicit prerequisite.** v1
   proposed it; it exists in neither the schema nor the pipeline. B1 and the
   trajectory half of B5 are blocked on it.
4. **B5 is split** into an early half (ships without assessments) and a blocked
   half (needs them).
5. **`raw_documents` is specified as create-with-`source_type`**, not an addition —
   the table is net-new, so it's a column in the initial `CREATE TABLE`.

---

## Confidence — the scale

Stored, queryable field is a 4-level ordinal. Optional numeric score for when
confidence is computed from evidence rather than self-reported.

```sql
-- on challenges, dependency_assessments, and coverage
confidence        TEXT CHECK (confidence IN ('unknown','low','medium','high')),
confidence_score  REAL,   -- nullable, 0.0-1.0; present only when computed from signals
```

- **Label is what you query.** `unknown | low | medium | high` — four levels
  because that's the resolution an LLM can actually produce reliably. A 6-point
  hand-scale would be false precision.
- **Score when earned.** When confidence is derived from concrete signals
  (corroborating source count, source authority, cross-source agreement), store the
  0–1 `confidence_score` and derive the label from it by fixed cutoffs. Number when
  grounded, label when judged.
- **Orthogonal to `contested`.** Confidence = evidence strength; contested =
  sources disagree. High-confidence-and-contested is a real, useful state.
- **Coverage reuses the scale, different meaning:** there it measures search
  thoroughness, not certainty of a fact.

---

## Build tiers (replaces v2's B1–B5 ordering)

### Tier 0 — prerequisite (build first; unlocks Tier 3)
- **`dependency_assessments` table + the reassess pass.** Introduced as a proposal
  in sourcing v1, never built. It writes one row per (dependency × assessment date)
  with `status`, `metric_name/value/unit`, evidence (`source_url`, snippet),
  `confidence`(+`contested`), `assessed_on`. Everything trajectory-based is
  downstream of this. Treat it as its own buildable unit.

### Tier 1 — ship now (no upstream dependencies)
- **A2 — `raw_documents` created with `source_type`** + two-tier cache policy
  (discovery text immutable; outcome/reassessment results TTL'd/refreshable).
  ```sql
  CREATE TABLE raw_documents (
    id          INTEGER PRIMARY KEY,
    url         TEXT,
    url_hash    TEXT,
    fetched_at  TEXT,
    text        TEXT,
    source_type TEXT   -- 'discovery' | 'outcome' | 'reassessment' | ...
  );                    -- drives the cache freshness policy
  ```
- **B3 — `contested` + `confidence`** on rows already being written
  (`challenges`, and later `dependency_assessments`). Best cost/benefit item in the
  set, depends on nothing new. Ship it first.

### Tier 2 — mid-tier (new tables leaning on the sourcing/outcome pass, not on assessments)
- **B2 — `coverage` table** (per idea_space × era: searched, source_count,
  last_searched, confidence).
- **B4 — `events` + `company_events`** (macro-shock linkage).
- **B5a — early gap typology.** The attempt-density axis is available from v4 today,
  so this half ships without assessments:
  - `white_space` (no attempts + coverage high), `unsampled` (coverage low —
    the guardrail), `crowded_solved` (active successes present), `flared_out`
    (from `outcome_type` — brief success then collapse).

### Tier 3 — blocked on Tier 0
- **B1 — viability thresholds + `became_viable_date`.** Needs the assessment metric
  series to compute a threshold crossing.
- **B5b — full gap typology.** Adds the trajectory-dependent types that need
  "did the blocker improve": `lazarus_candidate` and `tested_dead_end`.

---

## Sequencing note

Part B as a whole is downstream of v1's reassess pass. Concretely: Tier 1 and Tier 2
are independent and can land against the current dataset; Tier 3 cannot begin until
Tier 0 (`dependency_assessments`) exists. The gap view is therefore *partially* live
after Tier 2 (white_space / unsampled / crowded_solved / flared_out) and only
*complete* after Tier 3.

## Open decisions

- **`confidence_score`:** compute from signals now, or ship label-only first and add
  the score once the reassess pass grounds it in real evidence? Label-only first is
  the low-risk start.
- Threshold value sourcing (B1), era boundaries, coverage granularity — unchanged
  from v2.
