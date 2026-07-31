# Project Lazarus — Thresholds & Progress Spec (v3.2)

Resolves the five open questions raised implementing v3.1, plus two blockers found in
the code. Supersedes v3.1's ambiguous points. Everything not mentioned carries forward
from v3 / v3.1 unchanged.

Context: several of these are v3.1 contradicting itself (the ETS-vs-historical capacity
mixup), not open design — this spec makes the resolved version authoritative.

---

## R1 — Capacity is historical-only; the forward path is extrapolated (was Q1)

v3.1's "Add 1" contradicted itself: the top callout describes the **51-row historical**
`capacity_series.csv` (real, `scenario=historical`), the body still described the
**abandoned ETS-scenario** file (solar 6.9 TW @2035 etc.). The ETS data is not in the
repo. Resolution:

- **Delete the ETS framing from v3.1.** `capacity_series.csv` is the 51-row historical
  series (25 OWID solar AC + 25 IRENA onshore wind + 1 battery anchor), full stop.
- **Wright needs both a fitted rate and a forward path.** Get both from history:
  - fit the learning rate on the historical (cost, cumulative-capacity) pairs;
  - derive the forward capacity path by fitting **log-capacity vs. time** on the same
    historical series and extrapolating — that is the doubling schedule.
- No new data, nothing uncited, reproducible from committed sources.
- **Known limitation (recorded, not fixed):** naive extrapolation predicts capacity
  doubling indefinitely and ignores saturation/policy. A better forward path (BNEF ETS,
  or NZS) is a *nice-to-have*, tracked in the development-ideas doc — not a dependency,
  and explicitly **not** worth relaxing the cited-anchor validator (a sourceless
  scenario number is unfalsifiable).

## R2 — `conditional` is a flag, no second bar (was Q2)

Reusing `threshold_alt` for the policy bar collides on **Direct air capture**, which
already uses `threshold_alt_value=300` for its *contested* bar (progress_alt computes
from it). One column can't mean both on the flagship row.

- Add **`policy_dependent`** (0/1) to `dependency_thresholds`. No second bar column.
- `conditional` trajectory state resolves from **`policy_dependent=1` AND
  economics-progress < 1**.
- `progress_alt` keeps meaning exactly one thing (contested).
- The second policy-scenario crossing (computing "crossed under policy") is **deferred**
  — it needs curated policy-bar values nobody has supplied. Revisit if it earns its keep.

## R3 — `conditional` trajectory state lands; typology cell deferred (was Q3)

The gap typology (`tested_dead_end`, `lazarus_candidate`) **does not exist in `src/`
yet**, so the `conditional` cell has no home.

- **Land now:** the `conditional` trajectory state (a CASE branch in the trajectory
  view).
- **Defer:** the typology cell. Leave a marker
  (`// conditional: typology cell pending gap-typology build`) so the intent is recorded.
- Do **not** build the gap typology as part of this change.

## R4 — Capital pools: name the two VC rows; FOAK is sentiment (was Q4)

Corrects v3.1's ruling. Two of the three capital rows measure capital supply; FOAK does
not.

- **Growth-stage** ($14.2bn annual growth investment) → pool = **growth-stage VC/PE**.
- **Early-stage** (8% seed+A share) → pool = **early-stage VC (seed+A)**.
- **FOAK** ("51% of investors named FOAK hardest to fund", CTVC pulse) → **`investor
  sentiment`**, and **not crossing-test eligible.** It's a survey reading, not a
  supply/cost measurement — forcing a pool name on it would imply it measures
  availability, which it doesn't. Effectively `qualitative` (no scalar bar, status-only).
  - If a real FOAK capital-volume or cost-of-capital figure appears later, *that* gets
    the `project finance` pool and becomes crossing-eligible.
- Edit lands in **`data/curated/cited_anchors.csv`** (not `metric_observations_full.csv`
  — that's generated now). Safe: none of the three capital dependencies has a
  `dependency_thresholds` row, so renaming the metric breaks no join.

## R5 — Full DB switch (was Q5)

`build-db` still reads the thin `data/curated/metric_observations.csv` (25 rows, no
`basis`). All three v3.1 items depend on data that only exists in the derived full file
(the capital rows; the 50 solar price points Wright fits against). Half-measures defeat
the point.

- **Point `build-db` at `data/derived/metric_observations_full.csv`** (76 rows).
- **Add `basis`** to the `metric_observations` table (v3 Fix 2) + load validation.
- **Load `data/derived/capacity_series.csv`** into a new `capacity_series` table.
- Retire the thin 25-row file.
- **Expected visible result: solar flips to `not crossed`** (real-2024 basis, 2024 =
  $0.258/W vs the $0.20 bar). This is the correctness win, not a regression — flag it to
  stakeholders as it lands.

---

## Two blockers to clear (from the code)

- **`PROJECTION_METHOD` enum:** currently `['linear_progress_fit']` with a CHECK on
  `metric_projections.method`. **Add `'wright'`** before the Wright fit writes rows, or
  the insert fails the constraint.
- **Dependency dedupe:** `dependencies.csv` carries both **"FOAK financing"** and
  **"First-of-a-kind plant financing"** — near-duplicates that will double-match on name
  resolution. **Merge to one canonical row now** (same entity-dedup discipline Tier 0
  applied to companies).

## Build order

1. **R5 full DB switch** (basis column, capacity table, derived observations) — the
   enabling change all three items sit on.
2. **R2/R3** (`policy_dependent` flag + `conditional` state) — trajectory-view change.
3. **R4** capital-pool naming + FOAK-as-sentiment — data edit in `cited_anchors.csv`.
4. Clear both blockers (`wright` enum, dependency dedupe).
5. **Wright's law** (R1: historical fit + extrapolated forward path) — last, needs R5
   and the enum.

## Still open

- Whether the extrapolated forward path needs a saturation term (logistic vs log-linear)
  once far-future crossings look unrealistic — deferred to the projection-quality work.
- The alternative-forward-path nice-to-have (ETS/NZS) — see development-ideas doc.
- Carried: `window_years` default, baseline choice, `metric_series_status` gating.
