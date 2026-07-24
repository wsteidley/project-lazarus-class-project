# Project Lazarus — Thresholds, Progress & Crossing Derivation Spec (v2)

Extends the hero-thresholds v1. v1 landed the data and columns; v2 specifies the
**derivation** that turns observations into something plottable and comparable across
dependencies, and adds the states/links the second research sweep surfaced.

Core idea: raw metric values don't share an axis (battery $/kWh vs. interconnection
years vs. carbon $/ton). A **normalized, direction-aware progress metric** puts every
dependency on one 0→1 scale, and the categorical states then *fall out of it* rather
than being hand-defined.

---

## Layer model (what lives where)

| Layer | Content | Storage |
| --- | --- | --- |
| Observations | raw dated metric values | **append-only rows** (`metric_observations`, exists) |
| Progress | normalized 0→1 per observation | **pure view** — scalar fn of (observation, threshold) |
| Trajectory | one summary per series | **view + `trajectory_config`** — window slope + classification, tunables as data |
| Projection | computed future crossing points | **TS-computed table** — not observations, flagged |

Progress is a genuinely pure view: a per-row expression over one observation and its
threshold. Trajectory is **not** pure — it needs an ordered window per series (slope
over the trailing N observations) and a classification with tunable cutoffs, so its
knobs live in a one-row `trajectory_config`, not baked into the SQL. Both still
recompute on every build with no pipeline stage to sequence. Projection is the one
layer that is real computation (extrapolation + confidence) whose outputs are not
observations — it is TS-computed into a table.

## Crossing key (mandatory)

Everything is keyed on **`(dependency, metric, scope)`**, never dependency alone.
Battery cost is $108/kWh (BNEF pack) *and* $140/kWh (IRENA 4-hour system)
simultaneously — both correct, different metrics. Comparing a pack value to a system
threshold is the bug this key prevents. Per-scope is why US-interconnection and
global-carbon are separate series (see per-scope decision below).

## Threshold storage — a table, not columns (changed from v1)

v1 put the threshold definition as **columns on `dependencies`** (one bar per
dependency). The per-scope decision below breaks that: carbon price needs a global bar
*and* a Norway bar, which is two thresholds for one dependency — impossible with
one-row-per-dependency columns. So the threshold definition **moves off `dependencies`
into its own `dependency_thresholds` table**, keyed on `(dependency, metric, scope)` —
the same key the crossing derivation and observations already use.

```sql
-- identity only; the bar no longer lives here
CREATE TABLE dependencies (
  name            TEXT PRIMARY KEY,
  category        TEXT,
  description     TEXT,
  threshold_kind  TEXT   -- quantitative_with_threshold | quantitative_tbd | qualitative
);

-- one row per (dependency, metric, scope): the bar itself
CREATE TABLE dependency_thresholds (
  dependency_name        TEXT NOT NULL REFERENCES dependencies(name),
  metric                 TEXT NOT NULL,
  scope                  TEXT NOT NULL,         -- global | US | EU | NO | …
  threshold_value        REAL,
  threshold_unit         TEXT,
  threshold_direction    TEXT,                  -- below_is_better | above_is_better
  threshold_source_url   TEXT,
  threshold_as_of        TEXT,
  threshold_note         TEXT,
  threshold_contested    INTEGER DEFAULT 0,
  threshold_alt_value    REAL,
  threshold_alt_source_url TEXT,
  threshold_contested_note TEXT,
  PRIMARY KEY (dependency_name, metric, scope)
);
```

- `threshold_kind` stays on `dependencies` — it's a property of the dependency, not of
  a per-scope bar.
- **Input CSVs follow the tables:** `dependencies.csv` keeps identity + kind;
  a new `dependency_thresholds.csv` (append/edit-in-place, one row per
  dependency×metric×scope) holds the bars. `metric_observations.csv` is unchanged.
- **Observations join to a threshold row** on `(dependency_name, metric, scope)` —
  an explicit key match, not a scope column compared against a single-bar row. This is
  what makes the progress view a clean join instead of a lookup.
- A dependency with `threshold_kind = qualitative` simply has **no**
  `dependency_thresholds` rows; `quantitative_tbd` may have a row with a null
  `threshold_value`.

## Normalized progress metric

Per `(dependency, metric, scope)` with baseline `B` (earliest observation, or a
declared attempt-era value) and threshold `T`:

```
below_is_better:  progress = (B - v) / (B - T)
above_is_better:  progress = (v - B) / (T - B)
```

Reads as: **0 at baseline, 1 at viability, >1 past the bar, negative if it receded
below where it started.** Dimensionless, so every dependency shares an axis.

Also expose, per observation:
- `value_raw` + `unit` — for a true-units plot.
- `progress` — the normalized 0→1.
- `log_distance = log(T / v)` (cost-curve/`below_is_better` metrics only) — these
  decline multiplicatively and read correctly on a log axis.

## States derived from progress (not hand-defined)

- **`currently_crossed`** = latest observation's `progress ≥ 1`.
- **`became_viable_date`** = `as_of` of the first observation reaching `progress ≥ 1`.
- **Trajectory state** from the sign + magnitude of recent slope
  (Δprogress over a trailing window, default ~3 observations / 3 years):
  - `improving` — slope clearly positive, not yet plateaued
  - `plateaued` — near-zero slope after prior gains (solar: fell 89%, then flat 2025)
  - `receded` — slope negative, moving away from `T` (hydrogen, interconnection)
  - `unknown` — too few observations to fit a slope
- `receded` and `plateaued` are **stored** on the trajectory view (resolves v1's open
  question: they are both derivable states and queryable, not view-only labels).

### How trajectory is computed (view + config, not pure view)

Slope is expressible in SQLite via window functions
(`OVER (PARTITION BY dependency, metric, scope ORDER BY as_of)`), so trajectory stays a
**view** — but the window size and plateau cutoff are the tunable knobs this spec flags
as open, so they must not be hardcoded into the SQL. Put them in a one-row config the
view cross-joins against:

```sql
CREATE TABLE trajectory_config (
  window_n                INTEGER,   -- trailing observations for the slope (default 3)
  plateau_slope_threshold REAL       -- |slope| below this => plateaued (default TBD)
);
```

Retuning is a one-row update, not a query edit; the view remains recompute-on-build
with no stage to sequence.

**Escape hatch to TS (Option B), only if needed.** If classifying a genuine plateau
from noise turns out to need more than slope sign + magnitude — variance, an R²/fit
quality on the trend — that is past comfortable SQL and should move to a
`computeTrajectory` TS step writing a `trajectory` table. Check this once batteries and
solar (the longest series) have enough points: if slope alone classifies them cleanly,
stay with the view; if you're fighting noise, materialize in TS. Default is the view +
config until that check says otherwise.

## Projection (the "equation makes new points" layer)

For series not yet crossed, extrapolate a **projected crossing date** and a short
series of computed future points:
- v2 method: linear fit on recent `progress` slope. (Wright's-law fit over cumulative
  capacity is a v3 upgrade for the cost-curve heroes.)
- Stored in a `metric_projections` table, every point flagged `projected=1`, with the
  method and fit window recorded, so a forecast never masquerades as an observation.
- Carries a confidence — learning curves aren't deterministic (2022 battery plateau,
  hydrogen reversal), so a projected crossing is an estimate, not a fact.

## Per-scope thresholds — DECIDED: first-class rows

One dependency can have several bars by slice. Carbon price is $21 global vs. $170
Norway; capital splits growth-up / early-stage-down. **Each scope is its own row in
`dependency_thresholds`** (this is exactly why the bar moved to its own table above),
so `(dependency, metric, scope)` yields a distinct plottable series and
"is this viable *here*" is a direct query. Leaner scope-tagged-observations was
considered and rejected — it pushes slicing into every consumer.

## Contested thresholds

A threshold can be disputed. DAC's $100/ton is industry shorthand, but the literature
actively argues it's the wrong bar (Climeworks ~$300 by 2030). The
`dependency_thresholds` columns carry this: `threshold_contested`,
`threshold_alt_value`, `threshold_alt_source_url`, `threshold_contested_note`. When
contested, the derivation computes progress against **both** bars and flags the
crossing as contested, so `became_viable_date` doesn't silently pick a side.

## `related_dependency` links

Dependencies cause each other. IRENA attributes US/German solar LCOE at ~2× China's to
permitting, interconnection, and balance-of-system — so interconnection partly *drives*
solar economics in those geographies. Add a `dependency_links(from_dependency,
to_dependency, relation, note)` table (`relation`: `drives` / `enables` / `blocks`), so
those chains are traceable rather than hidden.

---

## New seed rows (from the second research sweep, July 2026)

`dependencies.csv` — set `threshold_kind = quantitative_with_threshold` for the four
below (identity only; the bars go in the threshold table).

`dependency_thresholds.csv` — new rows, one per `(dependency, metric, scope)`:

| dependency | metric | scope | value | unit | dir | contested |
| --- | --- | --- | --- | --- | --- | --- |
| Onshore wind LCOE | LCOE | global | 50 | USD/MWh | below | — |
| Utility-scale battery system | 4h system cost | global | 150 | USD/kWh | below | — |
| Direct air capture | capture cost | global | 100 | USD/tCO2 | below | yes (alt 300) |
| Electrolyzer capex | system capex | global | 500 | USD/kW | below | — |

`metric_observations.csv` — append:

```csv
Onshore wind LCOE,LCOE,33,USD/MWh,2025-12,global,curated,https://www.irena.org/Publications,IRENA RPGC 2025,global weighted average
Utility-scale battery system,4h system cost,140,USD/kWh,2025-12,global,curated,https://www.irena.org/Publications,IRENA RPGC 2025,~30% drop in one year
Direct air capture,capture cost,500,USD/tCO2,2026-01,global,curated,https://energy-solutions.co/articles/sub/carbon-capture-direct-air-dac-cost-analysis,Energy Solutions Intelligence 2026,operating plants 400-600
Electrolyzer capex,system capex,1900,USD/kW,2026-01,global,curated,https://www.greenfueljournal.com/post/green-hydrogen-cost-economics-2026-the-real-path-to-price-parity,Green Fuel Journal 2026,Western installed; CN alkaline 300-700 advertised
Growth-stage climate capital,annual growth investment,14.2,USD_bn,2025-12,global,curated,https://www.ctvc.co/40-5bn-and-8-upturn-as-power-demand-drives-25-investment/,Sightline/CTVC 2025,rebounded from 8.0 in 2024
Early-stage climate capital,seed+A share of climate capital,8,percent,2025-12,global,curated,https://heatmap.news/climate-tech/early-stage-investing,Sightline via Heatmap,down from ~20% in 2021
FOAK financing,share naming FOAK hardest to fund,51,percent,2025-12,global,curated,https://www.ctvc.co/2025-climate-tech-investor-pulse-check/,CTVC investor pulse,unchanged; the cliff
```

Capital rows stay **three separate series that never roll up** — growth improved while
early-stage and FOAK worsened, opposite directions inside the old "capital" category.

---

## Build / derivation integration

- `build-db` gains the `progress` and `trajectory` views, seeds `trajectory_config`,
  and loads `metric_projections` + `dependency_links`.
- `progress` is a **pure view**; `trajectory` is a **view over `metric_observations` +
  `dependency_thresholds` + `trajectory_config`** (window slope, tunables as data).
  Both recompute on every build — no pipeline stage to sequence. Observations join
  their bar on `(dependency_name, metric, scope)`.
- `metric_projections` is **TS-computed** (extrapolation + confidence), the one piece
  that is not a view.
- Replaces the `// TODO(receded)` marker in `build-db.ts` with the real derivation.
- The signature cross-table join this enables (dependency progress curve × death-year
  of every company blocked on it, out through `company_dependencies` → `companies` →
  `idea_spaces`) is left to the app phase, but the views are shaped to feed it directly.

## Still open

- `trajectory_config` values: trailing-window size (default 3) and the `plateaued`
  slope threshold — plus the check on whether slope alone classifies cleanly or
  trajectory needs to move to a TS step (variance/R²).
- Baseline `B`: earliest observation vs. a declared attempt-era value per dependency.
- Wright's-law projection for cost-curve heroes (v3).
- `metric_sources.csv` refresh-cadence registry (still optional, still skipped).
- Sanity-check the v1 best-effort threshold `source_url`s in review.
