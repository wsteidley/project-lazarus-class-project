# Project Lazarus — Full-Series, Basis & Projection Spec (v3)

> **Implementation status (updated post data-reorg).** Fix 6's ingestion path is
> **built** as the `build-metric-data` step: provider extractors read `data/sources/`
> + `data/curated/` and generate `data/derived/metric_observations_full.csv` and
> `data/derived/capacity_series.csv`. So these files are now **derived outputs**, not
> hand-staged inputs. The `metric_observations_full.csv` is 76 rows; `capacity_series.csv`
> is the full **51-row** historical series (25 OWID solar AC + 25 IRENA onshore wind +
> 1 battery anchor). Fixes 1–4, basis, and the `conditional` state remain the DB-side
> work still to do (v3.1 owns sequencing). Wright's-law (Fix 5) is unblocked for solar
> and onshore wind now that real cumulative-capacity history is loaded; battery uses the
> published learning rate.

Ships alongside `metric_observations_full.csv` (76 rows). v3 is what implementing a
real, mixed-density series set requires — the sparse seed hid four problems that dense
real data exposes. Wright's-law projection (deferred here since v2) is now buildable
because solar finally has a real learning curve.

---

## The core problem the real data exposed

The seed had ~1 point per metric, so every series looked alike. The real data does not:

| Metric | Points | Shape |
| --- | --- | --- |
| Solar module cost | **50** (1975–2024, annual) | dense, full learning curve |
| Battery all-segment | 7 anchors | sparse, gaps 2011–17 & 2019–20 (BNEF paywalled) |
| Interconnection | 4 period-medians | era buckets, not annual |
| Wind LCOE | 2 anchors | endpoints only |
| Carbon / hydrogen / DAC / … | 1 | current only |

A **count-based trajectory window breaks across this.** v2's "trailing 3 observations"
is 3 *years* for solar and spans **2018→2022** for battery — incomparable slopes from
the same config. This is the headline fix.

---

## Fix 1 — Time-based trajectory window (replaces count-based)

Trajectory slope must be measured over a **fixed time span, not a fixed observation
count**, or dense and sparse series produce non-comparable rates.

- `trajectory_config` gains `window_years` (default 5); `window_n` is retired.
- Slope = Δprogress over the observations falling inside `[latest_as_of −
  window_years, latest_as_of]`.
- A series with too few points *in the window* → `unknown` (not a forced slope).
- `plateaued` / `improving` / `receded` thresholds are now per-year rates, which are
  physically meaningful and comparable across metrics.

## Fix 2 — `basis` column (unit is not enough)

Solar is **constant 2024 USD/W**; battery BNEF is real dollars in **each survey's own
base year**; carbon is **nominal**. Same unit (`USD/…`), different money — mixing them
silently corrupts both crossings and slopes.

- Add `basis` to `metric_observations` and `dependency_thresholds`:
  `real_2024_usd` | `real_usd` | `nominal_usd` | `na` (for years/percent).
- **Rule: one basis per `(dependency, metric, scope)` series, and the threshold must
  share it.** Validate at load; a basis mismatch between an observation and its bar is
  a build-time error (report-not-drop).
- The CSV already carries this column, populated.

## Fix 3 — Solar basis decision: OWID spine, pvXchange is a different metric

On the OWID constant-2024-USD series, 2024 = **$0.258/W** — *not* across a $0.20 bar.
Nominal pvXchange spot is below $0.20 today. They disagree **across the threshold**, so
"has solar crossed?" depends entirely on basis.

- Use the **OWID 2024-real series as the canonical spine** (consistent, cited, full
  history). Set the solar threshold in `real_2024_usd`.
- If pvXchange is kept, it is a **separate metric** (`module price (nominal spot)`) with
  its own threshold and basis — never a point appended to the OWID line.

## Fix 4 — Multi-metric per dependency is now real (battery)

Battery carries three legitimate, differently-crossing series against a $100 bar:
all-segment $108, **BEV $99 (crossed)**, stationary $70 (crossed). The CSV emits all
three as distinct `metric` values under one dependency.

- Each needs its **own `dependency_thresholds` row** (same `(dependency, metric, scope)`
  key), so the crossing test runs per metric.
- **Selection is a query-time choice:** an EV-era failure maps to the BEV line, a
  grid-storage failure to the stationary line. The derivation does not pick; it exposes
  all three and lets the consumer join the right one. Document this — it is the
  `(dependency, metric, scope)` design finally load-bearing.

## Fix 5 — Wright's-law projection for cost-curve heroes (the deferred v3 item)

Solar's dense series is a real learning curve, so projection can graduate from v2's
linear-progress-slope fit to the physically-motivated model.

- For `curve`-flagged metrics (solar, battery, wind, DAC, electrolyzer — the ones that
  follow experience curves), fit **log-linear cost vs. cumulative capacity** (Wright's
  law: ~20% per doubling for solar, ~18% for battery), and project the threshold
  crossing from the fitted rate.
- Requires a cumulative-capacity series per curve metric — a new small input
  (`capacity_series.csv`); where absent, fall back to the v2 linear-in-time fit.
- Non-curve metrics (carbon price, interconnection years, capital) **keep the linear
  fit** — they don't follow learning curves and a Wright fit would be nonsense.
- `metric_projections` records `method` (`wright` | `linear`) and the fit R² as
  confidence, every point still flagged `projected=1`.

## Fix 6 — Full-series ingestion path (BUILT as `build-metric-data`)

Downloading the OWID CSV worked; this is now formalized so "full series" isn't a manual
transcription each time.

- **Built:** one provider extractor per source (OWID solar cost, OWID solar capacity,
  IRENA `.xlsb` onshore wind capacity, cited-anchor pass-through) reads
  `data/sources/<provider>/` + `data/curated/` and emits rows with basis + citation
  pre-filled, report-not-drop, into `data/derived/`. The IRENA extractor runs in a
  Python/uv project (invoked like the Splink step); **uv-absent → warn + keep the
  existing committed file if present, fail if it's missing** (a from-scratch rebuild
  can't produce a partial file). Output is deterministic (byte-identical across runs).
- **Backfill status:**
  1. Wind capacity — **DONE** (IRENA onshore-only series, 2000–2024, replaces GWEC
     anchors; fixes offshore contamination).
  2. Wind LCOE (cost) — still 2 endpoints; IRENA annual LCOE is the next drop.
  3. Carbon price — still current-only; EU-ETS annual is the next drop.
  4. Battery 2011–17 & 2019–20 cost gaps — only if a non-paywalled compilation is found.
- Hydrogen/DAC/electrolyzer stay current-only for now (young metrics, thin history) —
  and that's honest, not a defect.

---

## Data completeness — make it visible, not assumed

A `metric_series_status` view (or column) classifying each `(dependency, metric,
scope)` as `full` / `anchor` / `current_only`, so a projection or trajectory built on
one point is *labelled* thin rather than silently trusted. This is the same
"absence-is-signal, coverage must be recorded" discipline from the gap-typology work,
applied to metric depth: a `became_viable_date` from a 2-point series should not read
like one from solar's 50.

## Build order

1. **Fix 2** (basis column) — schema + load validation; blocks correct comparison.
2. **Fix 1** (time-based window) — the trajectory-correctness fix.
3. **Fix 3 + 4** (solar spine, battery multi-metric) — threshold rows + basis set.
4. **Fix 6** (ingestion path) — then backfill carbon + wind to `full`.
5. **Fix 5** (Wright's law) — last; needs capacity series and benefits from denser data.

## Still open

- `window_years` default (5) and the per-year plateau/recede slope thresholds — retune
  once solar (dense) and battery (sparse-but-real) are both loaded.
- Cumulative-capacity source for the Wright fit (IRENA/BNEF capacity series).
- Whether `metric_series_status` gates projection (suppress on `current_only`) or only
  labels it.
- The two soft values still flagged in-file: wind 2010 ($89, derived from −71%) and the
  pvXchange current spot — tighten from IRENA/pvXchange directly.
