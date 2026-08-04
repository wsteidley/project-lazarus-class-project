# Project Lazarus — Open Questions Tracker

Single home for cross-spec open questions, so they stop living scattered across 20+
spec files. Two sections: **recently resolved** (with the ruling, so they're not
re-litigated) and **still open** (filtered to what's actually live — stale/superseded
questions removed). Update this instead of re-opening old specs.

Last audited: 2026-08-02 (across all specs in this folder).

---

## Recently resolved

### The "neutral data, analysis at query time" principle
A through-line now settled three times: **the stored/derived data commits to no
analytical choice; anything that is really an analysis decision moves to query time
(or a precomputed set) where the viewer owns it.** Applied to baseline, trajectory
window, and (via derivation) curve classification below.

### R-A — Progress baseline `B` = earliest observation; dual linear + log axis
- `B` = the earliest observation in each `(technology, metric, basis, segment, scope)`
  series. Automatic, per-series, **no curated attempt-era baselines.**
- Rationale: baseline is a *property of the series* (objective); "progress since era X"
  is an *analytical lens* the viewer applies via query-time re-baselining. Don't freeze
  a lens into the data.
- **Both scales precomputed (resolves the saturation problem the v3.2 run exposed):**
  a 1975 solar baseline makes `progress_linear` saturate to ~1.0 for every modern value
  (uninformative for a 3-orders-of-magnitude learning curve). So store **both**
  `progress_linear` = `(B−v)/(B−T)` and `progress_log` = `(lnB−lnv)/(lnB−lnT)`. Log stays
  legible across orders of magnitude and is the more natural axis for multiplicative cost
  curves; linear stays the intuitive "distance to bar." Baseline stays earliest-obs for
  both — this fixes the *arithmetic*, not the baseline principle.
- **`scale` (`linear`/`log`) becomes a precompute dimension** alongside `window`:
  each `(series × window × scale)` cell carries a progress/trajectory/projection view.
- **Log guard:** `progress_log` only where `B`, `v`, `T` are all strictly positive;
  zero/negative/signed metrics → `progress_log` null-with-reason (not an error). Cost and
  capacity qualify; bounded/signed metrics get linear only.
- **Log is the preferred trajectory basis for cost curves** — constant % decline is a
  straight log line, so `improving`/`plateaued`/`receded` classifies more cleanly on log
  for `learning`/`receded` series.
- **Keep the v2.1 guards:** `B == T` → null progress + reason; `B` past `T` →
  build-time error.
- **Document:** backfilling an earlier point shifts `B` and re-scales that series'
  progress — expected (progress is a derived view), not a bug.

### R-B — `curve_type` is DERIVED, not curated
- Derive from the fitted trend sign: slope clearly negative → `learning`; clearly
  positive → `receded`; near-zero → `flat`. `policy` / `not_applicable` come from
  separate signals (`policy_dependent` flag; no capacity axis), not the trend.
- A derived column/view, recomputed on build — same tier as `progress`/`trajectory`.
- **Wright fit gates on derived `curve_type = 'learning'`.** A rising series (geothermal,
  hydro) can never derive as `learning`, so it structurally cannot get a learning-curve
  fit — the derivation *is* the sanity check (no human tag to mislabel).
- Sequencing (not circular): cheap trend-sign pass classifies first → Wright projection
  runs only on `learning`.

### R-C — Trajectory & projections over 5 precomputed windows
- No fixed `window_years`. Compute over a **standard window set: `full`, `last_15`,
  `last_10`, `last_5`, `last_3`.**
- Per `(series × window)`: trajectory state *and* (for `learning` series) the Wright fit
  + projected crossing. **Models share the window set with trajectory**, so a projection
  and its trend are always over identical spans → directly comparable and plottable.
- Payoff: the same idea projected five ways; divergence between full-history and short-
  window projections is itself signal (uncertain future vs. confident call).
- `window` becomes part of the trajectory/projection table key (one row per series ×
  window).
- **Per-window sparsity guard:** too few points *inside* a window → no state / no fit,
  flagged (extends `metric_series_status` per window), never a garbage 2-point line.
- Pipeline's internal trend call (for `curve_type`, R-B) defaults to **full** history.
- Custom / user-supplied windows: **deferred** (app-layer concern, later).

### R-D — Others settled during build (recorded so they stay closed)
- `outcome_type` → **derived** (living_status + funding + exit + narrative), v4.
- `confidence_score` → **label-first, computed later**, v3.
- Wright capacity source → **solved**: IRENA/OWID capacity series loaded. (Wright fits
  *hardware/installed cost* against capacity — solar module_price, battery installed
  cost. LCOE is NOT a Wright input; it's viability-side. Wind's rate is still blocked on
  its installed-cost series — see Still open.)
- Wind LCOE / DAC / electrolyzer promotion → **solved**: IRENA Fig S.2 (7 techs, 2010–25).
- The two soft values (wind 2010 $89, pvXchange spot) → **superseded** by 16-pt IRENA wind.
- `conditional` = flag not second bar (v3.2 R2); typology cell deferred (v3.2 R3).
- Capital pools: growth/early-stage = VC named; FOAK = `investor sentiment`, not
  crossing-eligible (v3.2 R4).
- cited_anchors schema, xlsb extractor location → resolved in the data-org build.
- Data structure → **star schema + lookups** (metric-data-structure-spec-v1).
- **`basis` split into three orthogonal columns** — *decided* (2026-07-26): `basis`
  (currency), `energy_basis` (nameplate/usable, AC/DC), `duration` (4h/2h/blended).
  Exposed by the battery reconciliation. **⚠ SPECCED, NOT IMPLEMENTED** — see Still open
  "implement basis split + enforce threshold match". Listed here for the *decision*; the
  build is still pending.
- **Battery cost reconciliation** → four non-interchangeable metrics, never merged; the
  old "4h system cost 140" was mislabeled IRENA TIC, merged-with-relabel. See
  `data-derivation-map.md`.
- **Wind learning rate** (2026-08-02) → **RESOLVED, wind fits at 25.0%/doubling, R²=0.85.**
  `total_installed_cost` (16 pts, 2010–2025) extracted from Fig 2.3 of the committed
  `.xlsx`. Wind LCOE stays viability-only and is still refused by the Wright gate — the
  43%-vs-25% contrast between the two series is the evidence for that rule, not an
  inconvenience to it. See `wind-installed-cost-spec-v1.md`.
- **A Wright fit does not require a threshold** (2026-08-02) — exposed by the above. The
  Wright input was read from the `progress` view, which inner-joins
  `dependency_thresholds`, so a series with no curated bar yielded **no fit and no error**.
  Wind TIC has no bar. The fit now reads `metric_observations` with the bar LEFT JOINed;
  only the crossing projection requires one. Same fit-≠-forecast separation as
  `wright_fits`. Consequence: with no bar there is no curated `direction` either, so a
  rising series is now rejected by the negative-slope guard rather than by its label.

---

## Still open (live)

### Active work — metric data structure & thresholds
- **Implement the `basis` three-way split** (SPECCED, not built). Code still has one
  `basis` column; `usable kWh, blended duration` sits in a `note` string as interim.
  Touches `types.ts`, the DDL, every extractor, and the curated CSVs. Not purely
  additive — destructure the conflated `capacity_series` column (`AC`→`energy_basis`,
  `onshore`→`segment`, `energy`→`unit`).
- **Enforce the threshold basis-match rule** (baseline rule #1 — never actually been
  true). `dependency_thresholds` has *no* basis columns; the `progress` join is on
  `(dependency_id, metric, scope)` only — so an observation can merge onto a bar with no
  declared basis (the battery relabel did exactly this). Must land WITH the split: add
  `basis`/`energy_basis`/`duration` to `dependency_thresholds` AND those predicates to
  the progress join. Only then is "mismatch is a build-time error" real.
- **Technology / dependency split** (`technology-dependency-split-spec-v1.md`) — two
  sub-questions still open: (a) do metric tables key directly on `technology` or on
  `reference_entities` filtered to `kind='technology'` (same result, pick per
  implementation simplicity); (b) is `era` on `dependency_links` a year, a range, or a
  tie to an idea-space failure cohort (the deeper "baseline from failure era" lens).
  The split itself is resolved; these are implementation shape.
- **OWID solar refresh (data freshness).** The v3.2 run projected solar's crossing at
  ~2025-09 — already past, because the OWID series ends 2024-12. The fit is fine; the
  input is a year stale. Re-pull `solar-pv-prices` (and capacity) before anyone quotes a
  crossing date. Applies to any series whose last point predates the current year.
- **IRENA source URL** in `irena_lcoe_series.csv` is a placeholder — verify it points at
  the exact RPGC 2025 data file before commit. *(quick, do before merge)*
- **`region`/`scope` lookup** — add a third enforced lookup, or leave `scope` free-text?
  Lean: free-text until a second scope actually appears.
- **`metric_series_status` gating** — does thin/short-window data *suppress* a projection
  or just *label* it? Lean: label, per the absence-is-signal philosophy. *(decide when
  wiring projections)*
- **Saturation term** — log-linear vs logistic forward extrapolation, once far-future
  crossings look unrealistic. Deferred to projection-quality work.
- **Trajectory as SQL view vs TS step** — if relative-slope classification is too noisy
  on real series, promote to a TS fit with significance/R². *Empirical check to run now
  that dense solar + real wind are loaded. **Note:** geothermal/hydro (the other
  `receded` series) are held pending the technology/dependency split, so
  **interconnection is currently the only loaded `receded` series** — the check is
  weaker until the split lands, and may be worth reordering after it.*

### Gated on measurement (don't decide yet)
- **`FUZZY_MERGE_THRESHOLD`** + whether a mid-band queues for review — needs the
  near-miss numbers from a real resolve run first.
- **Blocking rules** for Splink — tune to real near-miss shapes.

### Orchestrator ergonomics (low stakes)
- Does `build` auto-run at pipeline end, or stay explicit?
- Fresh `npm run pipeline` → new run dir, or resume incomplete latest?
- Stale-downstream invalidation → block (`--force`) or warn?

### Parked until data is solid (app layer)
- SQLite-wasm vs backend DuckDB; embedding model + where it runs; annotation storage
  (in-DB vs separate); single- vs multi-user. All correctly deferred.

### Analysis choices deferred to query-time / app (by the R-A…R-C principle)
- Attempt-era re-baselining, custom windows, idea-space-specific baselines — these are
  now explicitly *viewer* concerns, not stored-data decisions. Listed here only so it's
  clear they were considered and intentionally pushed to the app.

---

## How to use this doc
- Resolve an item → move it from **Still open** to **Recently resolved** with the ruling.
- New cross-spec question → add to **Still open** here, not buried in a versioned spec.
- Spec-local, single-spec details can stay in that spec; this tracks the ones that span
  specs or block the active work.
