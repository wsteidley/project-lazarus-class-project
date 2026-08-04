# Project Lazarus — Metric Data Structure Spec (v1)

**Governed by `../foundational-principles.md`.** This schema must satisfy P2 (metric
coverage never gates inclusion), P3 (each absence is a labeled state, never a bare null),
and P4 (a viability call always carries its basis) — check changes here against those.

Restructures metric data into a **star schema** with enforced valid-value lookups, and
adds a `segment` dimension so slices (on/off-grid, on/offshore) are filterable or
rolled-up. Supersedes the flat `metric_observations` shape.

Design principle: **observations are uniform (one long fact table); the per-technology
variation lives in dimension/lookup tables, not in extra columns.**

---

## Tables

### `metric_observations` (fact — the long table)

One row per `(technology, metric, basis, energy_basis, duration, segment, scope, as_of)`.
Every column is populated for every row — no per-technology empty columns.

```sql
CREATE TABLE metric_observations (
  id            INTEGER PRIMARY KEY,
  technology    TEXT NOT NULL,
  metric        TEXT NOT NULL,        -- module_price | LCOE | cumulative_capacity | ...
  value         REAL NOT NULL,
  unit          TEXT NOT NULL,        -- USD/W | USD/MWh | GW | USD/kWh | ...
  basis         TEXT NOT NULL,        -- CURRENCY only: real_2024_usd | real_2025_usd | nominal_usd | na
  energy_basis  TEXT NOT NULL,        -- how the denominator is defined: nameplate | usable | AC | DC | na
  duration      TEXT NOT NULL,        -- storage duration: 4h | 2h | blended | na
  segment       TEXT NOT NULL,        -- all | on_grid | off_grid | onshore | offshore
  scope         TEXT NOT NULL,        -- global | US | EU | ...
  as_of         TEXT NOT NULL,        -- data year (YYYY-12)
  method        TEXT NOT NULL,        -- curated | feed | llm
  source_url    TEXT,
  source_name   TEXT,
  note          TEXT,
  FOREIGN KEY (technology)            REFERENCES technologies(name),
  FOREIGN KEY (technology, metric)    REFERENCES technology_metrics(technology, metric),
  FOREIGN KEY (technology, segment)   REFERENCES technology_segments(technology, segment)
);
```

**`basis` decomposes into three orthogonal columns** (amendment — the IRENA battery
reconciliation exposed that one `basis` column was carrying three unrelated things):
- **`basis`** — *currency* only (`real_2024_usd`, `nominal_usd`, …). What it always
  should have meant.
- **`energy_basis`** — how the per-energy denominator is defined: `nameplate` vs
  `usable` (batteries), `AC` vs `DC` (solar). **DC/AC moves here** from `basis` — it's
  the same *kind* of thing (denominator definition), and this makes solar and battery
  consistent.
- **`duration`** — storage duration (`4h`/`2h`/`blended`/`na`). Battery-specific mostly,
  but real: IRENA TIC is `blended`, BNEF turnkey is duration-specific.

All three are `na` where not applicable. The one-basis-per-series and mismatch-is-a-
build-error rules now apply per column: an observation must match its threshold on
currency **and** energy_basis **and** duration.

**Enforcement must land WITH the split (the rule has never actually been enforced).**
Splitting the observation columns alone doesn't close the hole — `dependency_thresholds`
has no basis columns and the `progress` join is on `(dependency_id, metric, scope)` only,
so today a `real_2025_usd`/usable/blended series can be merged onto a bar with *no
declared basis of any kind* (this is exactly what the battery relabel did). So the same
change must: (a) add `basis` / `energy_basis` / `duration` to `dependency_thresholds`,
and (b) add those three predicates to the progress join. Only then is baseline-spec
rule #1 ("mismatch is a build-time error, not a warning") real rather than aspirational.

**Destructuring the old conflated column (no fourth concept needed).** The current
`capacity_series` `basis` holds three different things — destructure, don't invent:
- `basis=AC` (solar) → **`energy_basis=AC`**.
- `basis=onshore` (wind) → **`segment=onshore`** (it was never a basis).
- `basis=energy` (battery GWh) → **it's the `unit` (`GWh`)**; basis/energy_basis/duration
  all `na`. The old `energy`/`power` "basis" was a mis-filed unit distinction.

### `technologies` (dimension — one row per technology)

Holds the per-technology specifics that would otherwise be empty columns on most rows.

```sql
CREATE TABLE technologies (
  name              TEXT PRIMARY KEY,
  wright_capacity   TEXT,        -- which capacity metric Wright fits (GW vs cumulative-GWh) or NULL
  policy_dependent  INTEGER,     -- 0/1  (curated: drives the `conditional` state)
  note              TEXT
);
-- NOTE: curve_type (learning | receded | flat) is NOT stored here — it is DERIVED from
-- the fitted trend sign over full history (see "Derived: curve_type & windows" below),
-- recomputed on build. `policy` / `not_applicable` come from policy_dependent / absence
-- of a capacity axis. Deriving it means a rising series can never be mislabeled
-- `learning`, so it structurally cannot get a learning-curve fit.
```

### `technology_metrics` + `technology_segments` (lookups — enforce valid pairs)

The valid-value lists. FKs above mean an observation can only exist if its pair is
listed here — an illegal pair (Solar PV / offshore, Carbon price / cumulative_capacity)
is **rejected at insert**, not just discouraged.

> **Metric data keys on `technology`, not `dependency`** (see
> `technology-dependency-split-spec-v1.md`). A technology with zero dependencies is
> still valid data. Dependencies **join** to technologies via a generic
> `reference_entities` (kind=`technology` now) + `dependency_links` table carrying the
> `(era, threshold)` slice — they don't share a key. This spec's tables are the
> technology side; the split spec owns the join.

**Metric vocabulary** (seed `technology_metrics` with the tech×metric pairs that exist):
`module_price` (USD/W, solar), `LCOE` (USD/MWh, 7 techs), `total_installed_cost`
(USD/kW, per tech), `battery_installed_cost` (USD/kWh), `bess_additions` (GWh/yr →
cumulate), `cumulative_capacity` (GW), `capacity_factor` (fraction), `lcoe_by_market`
(USD/MWh, scope-varying), plus cited-anchor metrics (`battery pack price`,
`interconnection queue wait`, `carbon price`, …). See `data-derivation-map.md` for
which source file feeds each.

**One extractor may emit several metrics.** The IRENA RPGC extractor is multi-metric
(LCOE + installed cost + battery cost + BESS additions + capacity factor + market
LCOE from one file). The extractor contract returns a row list, so this is already
supported — just don't assume one-metric-per-extractor, and seed every new metric's
lookup row before loading or the FK rejects it.

```sql
CREATE TABLE technology_metrics  (technology TEXT, metric  TEXT, PRIMARY KEY (technology, metric));
CREATE TABLE technology_segments (technology TEXT, segment TEXT, PRIMARY KEY (technology, segment));
```

Seeded as CSVs in `data/curated/` (data, not code — adding a pair is a PR row, not a
migration). These double as documentation: "what does solar track?" is
`SELECT metric FROM technology_metrics WHERE technology='Solar PV'`.

---

## The `segment` rule (prevents double-counting)

`segment` is **`NOT NULL DEFAULT 'all'`** — it's an equality key in the trajectory
self-join, and `NULL = NULL` is never true in SQL, so a NULL-segment row would silently
**vanish** from trajectory rather than error. Same applies to `energy_basis`, `duration`,
`basis` where used as join keys: default to `na`/`all`, never NULL.

- `segment='all'` is the rolled-up/total value; `on_grid`/`off_grid`/`onshore`/`offshore`
  are the parts.
- **Never sum `all` together with its parts.** Read *either* the `all` row *or* the sum
  of parts — never both.
- If a source gives only the total → store `all`. If it gives the split → store the
  parts and derive `all` by summing (do not also store a separate `all` row for the same
  metric+year). Modeled as an explicit column with an `all` sentinel, the double-count is
  structurally hard to write by accident.
- Generalizes the existing splits: onshore/offshore (wind), on/off-grid (any). DC/AC
  moves to **`energy_basis`** (not `segment`, not `basis`). The capacity totals already
  loaded are effectively `segment=all`
  (they sum on+off-grid) — breaking off-grid out means adding *part* rows, not stacking a
  second total on top.

## Both cost metrics, all technologies

Each technology carries parallel series, never merged:
- `module_price` (USD/W) — the **hardware** cost curve; what **Wright's law fits**
  (manufacturing learning). Currently solar; add others as sources appear.
- `LCOE` (USD/MWh) — the **delivered-energy** cost; the **viability/threshold** side.
  Now loaded for 7 technologies (IRENA Fig S.2, `irena_lcoe_series.csv`).
- `cumulative_capacity` (GW) — the Wright capacity axis.

They answer different questions (hardware-got-cheaper vs. project-pencils-out) and can
move oppositely (2022–23: $/W fell, LCOE rose on financing). Wright uses `module_price`;
thresholds/viability use `LCOE`.

## New data delivered

`irena_lcoe_series.csv` — 112 rows, IRENA RPGC 2025 Fig S.2, `basis=real_2025_usd`,
`metric=LCOE`, 2010–2025 annual:
- Solar PV, Onshore wind, Offshore wind, CSP, Geothermal, Hydropower, Bioenergy.
- **Onshore wind LCOE now 16 points** (was 2) — but **LCOE is NOT Wright-fittable**
  (it's viability-side). Wright needs a *hardware/installed-cost* series. Wind LCOE feeds
  viability/crossing only; the fit does not run on it. (Enforced in `wright.ts`: fitting
  LCOE gave 43%/doubling — ~2× any published onshore figure, because LCOE also captures
  capacity-factor and financing gains that aren't manufacturing learning. Refused rather
  than shipped.) **Wind's learning rate is no longer blocked:** it comes from
  `total_installed_cost` (16 pts, Fig 2.3 of the RPGC `.xlsx`) and fits at
  **25.0%/doubling, R²=0.85** — see `wind-installed-cost-spec-v1.md`.
- Geothermal & hydro **rose** over the period → real `receded` curves.
- Goes in `data/sources/irena/`; extractor emits `metric_observations` rows.

## Build order (dimension before fact — forced by FKs)

1. Seed `technologies`, `technology_metrics`, `technology_segments` (curated CSVs).
2. Load `metric_observations` (extractors), FKs validated against the lookups.
3. Derived views (`progress`, `trajectory`) unchanged in shape — now filter on `segment`.

## Migration note

Existing `metric_observations` rows gain `segment` (default `all`) and keep `basis`.
The v3.2 DB switch (basis column, capacity table) and this restructure should land
together, since both rewrite the observations table — sequence this spec's schema with
v3.2 R5 rather than as two separate migrations.

## Derived: `curve_type` & windows (resolved — see open-questions-tracker R-B/R-C)

- **`curve_type` is derived** from the fitted trend sign (negative → `learning`,
  positive → `receded`, near-zero → `flat`), full-history by default, recomputed on
  build. Wright fit gates on derived `curve_type = 'learning'`.
- **Baseline `B` = earliest observation** per series (R-A). No curated baselines;
  re-baselining is a query-time lens. Keep the `B==T` / `B past T` guards.
- **Progress is computed on TWO scales, both precomputed** (R-A): `progress_linear`
  = `(B−v)/(B−T)` and `progress_log` = `(lnB−lnv)/(lnB−lnT)`. Linear is the intuitive
  distance-to-bar; log stays legible across orders of magnitude (a 1975 solar baseline
  saturates the linear axis to ~1.0 for every modern value). **Log guard:**
  `progress_log` only where `B`, `v`, `T` all strictly positive; else null-with-reason.
  Log is the preferred trajectory basis for cost curves (constant % decline = straight
  log line).
- **Trajectory + Wright projection are precomputed over 5 windows × 2 scales** —
  windows `full`, `last_15`, `last_10`, `last_5`, `last_3`; scales `linear`, `log`.
  `window` **and** `scale` are part of the trajectory/projection table key (one row per
  `series × window × scale`). Models share the window/scale set with trajectory so
  they're comparable. Per-window sparsity guard: too few points → no state/fit, flagged.
  Custom windows deferred to the app.

## Still open

- Whether to add a `region`/`scope` lookup too, or leave scope free-text for now.
- Verify the IRENA source URL (placeholder in the extract) points at the exact RPGC 2025
  data file before committing.
- Whether trajectory stays a SQL view or moves to a TS step (significance/R²) — empirical
  check now runnable on the loaded dense/real/receded series.
