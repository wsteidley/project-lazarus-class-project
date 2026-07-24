# Project Lazarus — Thresholds & Progress Spec (v2.1)

Patch to v2, prompted by what the real seed exposed at build time. **Correctness fix,
not new capability.** Everything not mentioned here is unchanged from v2.

Wright's-law projection stays deferred to v3 — it is a better projection *method*, not
a correctness issue, and is cleanly separable from all of this.

---

## What the seed exposed

| Symptom | Root cause |
| --- | --- |
| Interconnection `progress` is null | baseline (2yr) == threshold (2yr) → `(B−T)` divides by zero |
| Battery reads `crossed` at 108 vs. a 100 bar | baseline (97) already past the bar → formula inverts |
| Most series `unknown`, 0 projections | single observation per series; slope needs ≥2 points |

The first two are **not** logic bugs in the classifier (`derivation.test.ts` proves
`receded`/`plateaued`/crossing all classify correctly given proper multi-point
baselines). They are consequences of baseline = earliest-observation, which v2 flagged
as open. The third is a data gap, not a code gap.

---

## Fix 1 — Decouple `currently_crossed` from `progress` (do this first)

**`currently_crossed` must not be derived from `progress ≥ 1`.** Crossing is a direct
scalar test against the bar, and involves no baseline at all:

```
below_is_better:  crossed = (latest_value <= threshold_value)
above_is_better:  crossed = (latest_value >= threshold_value)
```

`108 > 100, below_is_better → not crossed.` Full stop. Deriving it from progress is
precisely what let a bad baseline invert the answer — and it is the answer the whole
Lazarus ranking keys on, so a confidently-wrong value here is worse than a null.

Same change for **`became_viable_date`**: the `as_of` of the first observation that
passes the *direct* test, not the first with `progress ≥ 1`.

**Resulting division of labour — keep these separate:**

| Field | Purpose | Depends on baseline? |
| --- | --- | --- |
| `progress` | plotting, cross-dependency comparison on one 0→1 axis | yes |
| `currently_crossed` | is it viable *now* | **no** |
| `became_viable_date` | when it first became viable | **no** |
| trajectory state (`improving`/`plateaued`/`receded`) | direction of travel | slope only, not baseline level |

This makes the viability answers robust to every baseline problem — including ones not
yet encountered.

## Fix 2 — Declared baseline on `dependency_thresholds`

Add, on the same `(dependency, metric, scope)` key (Norway's carbon baseline is not
global's):

```sql
ALTER TABLE dependency_thresholds ADD COLUMN baseline_value REAL;
ALTER TABLE dependency_thresholds ADD COLUMN baseline_as_of TEXT;
ALTER TABLE dependency_thresholds ADD COLUMN baseline_note  TEXT;
```

**Semantics: the attempt-era value — where the metric stood when the companies were
dying.** That is what makes `progress` read as "how far has the world moved since the
failures," which is the thesis. It is *not* "the oldest number we happen to have."

`dependency_thresholds.csv` gains the three columns.

### Fallback and guards (explicit, not arithmetic)

- `baseline_value` null → fall back to earliest observation (current v2 behaviour).
- **`B == T`** → `progress` is **null with a reason code**, not a divide-by-zero.
  (Interconnection today.)
- **`B` already past `T`** (baseline satisfies the bar) → **report at build time as a
  baseline error.** It means the declared baseline postdates viability, so the series
  cannot express 0→1. (Battery today.) Report-not-drop, consistent with the rest of the
  loaders.
- Both guards surface in the build summary line alongside the existing
  unmatched/sourceless counts.

## Fix 3 — Backfill historical observations (curation, not code)

Single-point series are why trajectory is mostly `unknown` and projections are empty.
The data exists and is already sourced in v1/v2 — it just hasn't been seeded. BNEF,
IRENA, LBNL, and Our World in Data all publish full time series.

**Target: ≥3 points per hero — attempt-era, midpoint, current.** Minimum viable is 2
(slope needs two); 3 lets `plateaued` distinguish itself from `improving`.

Priority order:
1. **Grid interconnection** — needs a **pre-2007** point specifically; without one it
   has no room below the bar to show `receded`, which is the state it should be in.
2. **Battery** — a genuine attempt-era value (2010-era ~$1,100/kWh) fixes the inverted
   baseline and gives the cleanest full 0→1→>1 curve in the set.
3. Solar, carbon price, hydrogen — enough points to separate `plateaued` from
   `improving` (solar's 2025 flat year is the test case).

---

## Build order

1. **Fix 1** (decouple crossing) — small, no schema change, corrects battery today.
2. **Fix 2** (baseline columns + guards) — schema + loader + validation.
3. **Fix 3** (backfill) — curation pass, unblocks trajectory and projection for real.

## Still open (unchanged from v2)

- `trajectory_config` values: window (3) and plateau slope threshold (0.03) — retune
  once multi-point series exist.
- Whether trajectory classification needs to move from SQL view to a TS step
  (variance/R² on the fit) — **re-check after Fix 3**, since that is the first point
  real multi-point series exist to judge it on.
- Wright's-law projection for the cost-curve heroes → **v3**.
- `metric_sources.csv` refresh-cadence registry — still optional, still skipped.
