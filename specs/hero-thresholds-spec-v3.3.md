# Project Lazarus — Thresholds & Progress Spec (v3.3)

**Governed by `../foundational-principles.md`.** Everything this spec produces —
viability, crossing dates, progress — is a **tagged, inspectable overlay** (P1/P4). It
never gates a company's or a dependency's inclusion in the dataset.

**Build-state-of-record**, not new design. v3.2 was implemented; two runs and the
extended IRENA extraction since then changed enough that v3.2 no longer describes
reality. This reconciles it into one current narrative.

**Sources of truth (this doc does NOT restate them):**
- **Schema & derivation mechanics** → `metric-data-structure-spec-v1.md` (tables,
  lookups, `scale`/`window` dimensions, log guard, `curve_type` derivation).
- **Cross-cutting decisions & their rulings** → `open-questions-tracker.md`.
- **Where each number comes from** → `data-derivation-map.md` / `DATA_SOURCES.md`.

v3.3 owns only: what's built, what changed, and what's next.

---

## What's built (v3.2 as implemented + corrections since)

- **R5 full DB switch — DONE.** `build-db` reads `data/derived/metric_observations_full.csv`
  (76 rows) + `capacity_series.csv` (51 rows → `capacity_series` table); `basis` column
  added; thin 25-row file deleted.
- **R2/R3 conditional — DONE.** `policy_dependent` on `dependency_thresholds`, carried
  through `progress` into a `trajectory` CASE arm placed *ahead* of the economics arms
  (for a policy-dependent dep below its bar, "improving" answers the wrong question).
  Flags hydrogen, DAC, electrolyzer. No second bar; `progress_alt` still = contested only.
  Typology cell still deferred (typology unbuilt).
- **R4 capital pools — DONE.** Growth/early-stage = VC named; FOAK = `investor sentiment`,
  dependency flipped to `qualitative`.
- **R1 Wright's law — DONE.** `src/lib/wright.ts`: learning rate from ln(cost) vs
  ln(capacity), forward path from ln(capacity) vs time, historical-only, projected onto
  the same 0→1 axis. Solar validated: **28%/doubling, R²=0.95** (endpoint check ~26%).
- **Blockers cleared:** `'wright'` in `PROJECTION_METHOD`; dependency dedupe (26→25).
- **Solar flip confirmed:** `currently_crossed=0`, 2024 = 0.26 real_2024 vs the 0.20 bar.
  Working as designed.

## What changed since v3.2 was written (the reconciliation)

- **Baseline → dual linear + log progress (supersedes v3.2's single-axis).** The run
  exposed that a 1975 solar baseline saturates `progress_linear` to ~0.9995 for every
  modern value. Resolution (tracker R-A): keep earliest-observation baseline, but
  precompute **both** `progress_linear` and `progress_log`. Log is legible across orders
  of magnitude and the preferred trajectory basis for cost curves. Mechanics in the
  structure spec.
- **Precompute grid is now `5 windows × 2 scales`** (`full/15/10/5/3` × `linear/log`);
  `window` and `scale` are both in the trajectory/projection key.
- **`curve_type` is DERIVED, not curated** (tracker R-B). Wright gates on the derived
  value — a rising series can't be mislabeled `learning`, so it can't get a garbage fit.
- **Battery is now a REAL Wright fit — the published-rate shortcut is retired.** The
  extended IRENA extraction supplies battery installed cost (16 pts) + BESS additions
  (→ cumulative GWh). Battery joins solar and wind as a genuine fit. BNEF pack price
  stays a *separate* retail metric — not merged.
- **Five new metrics available** (installed cost $/kW, capacity factor, market LCOE,
  battery cost, BESS additions), and extractors may now emit **multiple metrics per
  source file** (see structure spec + derivation map).

## Known caveats (not bugs)

- **"0 linear projections" is correct.** Wright covers solar; every other series is
  single-point / receding / already-crossed / null — nothing for the linear pass. Will
  change once battery + wind fits load from the extended extraction.
- **Solar crossing projects to ~2025-09 (already past).** Legitimate projection from a
  stale last point (2024-12) — the OWID series needs a refresh, the fit is fine. Tracked
  as a data-freshness item.

## Next work (sequence)

1. **Load the extended IRENA metrics** — commit the raw `.xlsx` to `data/sources/irena/`;
   build the multi-metric extractor; seed the new `technology_metrics` lookup rows.
   **Wind first** (16 LCOE points replace 2, capacity already loaded, not blocked on the
   split) — this alone removes the "0 non-solar projections" state.
2. **Battery reconcile** — pending your review of whether IRENA `battery_installed_cost`
   (140.45) is the same quantity as the existing `4h system cost` (140, 150 bar). Confirm
   basis/scope/duration before merging; keep distinct from BNEF pack price.
3. **Technology / dependency split** — metric data re-keys to `technologies`; dependencies
   join via generic `reference_entities` + `dependency_links` (see
   `technology-dependency-split-spec-v1.md`). Unblocks the 5 orphan technologies (CSP,
   hydro, geothermal, bioenergy, offshore wind) the loader currently drops. Do after wind.
4. **Dual-scale progress + `scale` dimension** — implement `progress_log` and the
   `window × scale` precompute grid per the structure spec.
5. **Trajectory: view vs TS step** — with dense solar / real wind / receded geothermal
   now loaded, run the empirical check: does relative-slope-on-log classify cleanly, or
   promote trajectory to a TS fit with significance/R²? (tracker, active.)
6. **OWID solar refresh** — re-pull before quoting any crossing date.
7. **Deferred, unchanged:** gap typology (+ the `conditional` cell), saturation term for
   forward extrapolation, `metric_series_status` gating.

## Still open (pointers, not restated)
See `open-questions-tracker.md` — active items: trajectory view-vs-TS, region/scope
lookup, metric_series_status gating, IRENA URL verification. Nothing new opened by v3.3.
