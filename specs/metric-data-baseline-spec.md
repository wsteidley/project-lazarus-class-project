# Project Lazarus — Metric Data Baseline Spec (LIVING DOCUMENT)

**Status: living.** Update this whenever a new metric series is sourced or a new
methodology gotcha is found. It is the standing reference that informs every future
spec and every model/agent touching cost, capacity, or threshold data. If a data
question isn't answered here, answer it *and add it here*.

Purpose: cost curves, capacity series, and thresholds are only comparable if they
share a **basis** and a consistent **methodology**. Two production bugs have already
come from ignoring this (nominal-vs-real solar cost; DC-vs-AC solar capacity). This
doc exists so the third doesn't happen.

---

## The three rules that override everything

1. **One basis per series, and the threshold must share it.** A series is keyed
   `(dependency, metric, scope, basis)`. Never mix bases in one series; never compare
   an observation to a threshold of a different basis. Validate at load; a mismatch is
   a build-time error, not a warning.
2. **Prefer one source end-to-end over stitched anchors.** A learning-curve fit reads
   the *slope across doublings*; a series that switches methodology partway (e.g.
   conservative early, generous late) bends the curve and biases the rate. A single
   consistent source with more points beats a "better" latest number from a different
   methodology.
3. **`as_of` is the year the number describes, not the publication year.** Reports are
   published the year *after* the data year (BNEF 2025 survey → 2024 data; IRENA 2025
   report → 2024 costs). Getting this wrong shifts crossing dates.

## Basis types (extend as needed)

| Domain | Basis values | Notes |
| --- | --- | --- |
| Money | `real_2024_usd`, `real_usd`, `nominal_usd` | real = inflation-adjusted to a base year; state the base |
| Solar/PV capacity | `AC`, `DC` | DC (panel) runs ~15–25% higher than AC (grid); **never mix** |
| Wind capacity | `total`, `onshore`, `offshore` | LCOE metric is onshore → capacity should be onshore to match |
| Battery volume | `energy` (GWh), `power` (GW) | learning curve fits **cumulative GWh**, not GW or $/kWh |
| Non-scalar | `na` | years, percent, counts |

---

## Per-technology guidance

### Solar PV
- **Cost spine:** OWID `solar-pv-prices` (constant 2024 USD/W). Full annual 1975–2024.
- **Capacity spine:** OWID `installed-solar-pv-capacity` (AC basis, IRENA/Ember).
  Full annual 2000–2024. **Now loaded — replaced the earlier 9-anchor stitch.**
- **GOTCHA — DC vs AC (cost you a 20% error).** REN21/GSR/Statista headline figures
  (2,247 GW for 2024) are **DC/panel** and include latest China; OWID/IRENA
  (1,866 GW for 2024) is **AC/grid**, more conservative and consistent. They agree
  <6% through 2020, then diverge to ~20% by 2024. **Use the AC series wholesale.**
  Same-source-as-cost (both OWID) is the deciding factor.
- **GOTCHA — cost basis.** OWID cost is real-2024; pvXchange spot is nominal. On the
  real spine, 2024 ≈ $0.258/W → **not** across a $0.20 bar; nominal spot is below it.
  The crossing verdict flips on basis. Real spine is canonical; pvXchange is a
  separate metric if kept.
- Learning rate check: fitted rate should land ~20–25%/doubling.

### Wind (onshore)
- **Cost:** IRENA RPGC (LCOE). Currently only 2 points (2010 ~$89, 2025 $33) —
  **thin; fit is a 2-point line, low-confidence** until backfilled from IRENA annual.
- **Capacity:** currently GWEC anchors, `basis=total` — **still to replace** with the
  IRENA/GWEC full annual onshore series.
- **GOTCHA — offshore contamination.** GWEC's 1,136 GW (2024) **includes ~83 GW
  offshore**. The LCOE metric is onshore, so subtract offshore for an onshore-only
  capacity match, or the fit pairs onshore cost with total capacity.

### Lithium-ion battery
- **Cost:** BNEF price survey anchors (real USD, mixed base years — flag). Sub-metrics
  matter: all-segment $108, BEV $99, stationary $70 (2025) — **three metrics, three
  thresholds** against a $100 bar, differently crossed.
- **Capacity:** **the hard case.** The learning curve fits **cumulative GWh
  produced/deployed**, NOT installed GW and NOT $/kWh. Only a single anchor is loaded
  (~3.5 TWh at cost-parity).
- **GOTCHA — manufacturing capacity ≠ cumulative production.** IEA's "4 TWh by 2025"
  is *annual nameplate manufacturing capacity* — a different quantity. Do not put it
  in a cumulative-GWh series.
- **Defensible shortcut:** battery's learning rate is well-measured — Ziegler &
  Trancik (arXiv 2007.13920): **20%/doubling all-cell, 24% cylindrical, 1992–2016.**
  Using that *published, empirical* rate for battery is acceptable where a full fit
  isn't possible — it's a measured value, not a guess. (Contrast: assuming solar's
  rate would have been a guess, which is why solar needed the real fit.)

### Carbon price
- Current only (World Bank State & Trends). **Regional, not global scalar:** ~$21
  global vs ~$170 Norway — separate `scope` rows. `above_is_better`. Historical EU-ETS
  annual is the backfill target.

### Green hydrogen / DAC / electrolyzer
- Current-only, young metrics, thin history — and **policy-dependent** (cross only
  under a support regime → `conditional` trajectory state, not `tested_dead_end`).
- DAC threshold is **contested** ($100 shorthand vs ~$300 literature) — carry both.

### Capital (growth / early-stage / FOAK)
- **Name the pool in the metric** — VC / public equity / total equity / debt /
  project finance. "Climate equity +53%" and "VC down 3 years" are simultaneously
  true; a failed startup faced a *specific* pool. Rulings: growth → growth VC/PE;
  early-stage → seed+A VC; FOAK → project finance/debt (not equity).
- These three **never roll up** — they moved in opposite directions.

---

## Sourcing playbook (the OWID-download pattern that works)

1. Prefer a **downloadable CSV** with a full annual series (OWID graphers, IRENA
   Renewable Capacity Statistics, LBNL data files) over headline figures in reports.
2. OWID is the pragmatic spine for solar/wind cost and capacity — it aggregates
   IRENA/Ember/Nemet and ships a CSV + readme with citation.
3. Point each row's `source_url` at the **specific series/report**, not a homepage.
4. Paywalled series (BNEF price survey, Ziegler-Trancik supplement) → cite by name,
   blank URL, mark as anchor.
5. Record every value's `as_of` as the **data year**, and note the basis.

## Known-thin series (backfill targets, in priority order)

1. Wind LCOE (cost) — 2 points → IRENA annual.
2. Wind capacity onshore-only — replace GWEC total-basis anchors.
3. Carbon price history — EU-ETS annual.
4. Battery cumulative-GWh — Ziegler-Trancik supplement (or use published rate).
5. Solar middle-year capacity — **RESOLVED**, full OWID AC series loaded.

## Changelog (append every update)

- **2026-07-25** — Created. Solar capacity switched from 9 stitched anchors to full
  OWID AC series (2000–2024); logged the DC-vs-AC gotcha (~20% divergence post-2021).
  Added `basis` guidance per technology. Solar cost real-vs-nominal and battery
  cumulative-GWh gotchas carried in from v3/v3.1 specs.
- **2026-07-25 (later)** — Wind capacity RESOLVED: replaced GWEC total-basis anchors
  with full IRENA onshore-only series (2000–2024, 16.9→1049.8 GW) from the IRENA Stats
  Tool `.xlsb`; fixes offshore contamination. Solar OWID vs IRENA cross-validated to
  <0.4%. Logged IRENA extraction gotcha: capacity lives on `On-grid`/`Off-grid
  electricity` producer-type rows; `All types` rows carry only finance and have null
  capacity — sum on/off-grid, exclude `All types`.
- **2026-07-25 (data-reorg)** — Metric data made reproducible: `data/{sources,curated,
  derived}` split; `build-metric-data` step generates the derived CSVs from raw
  sources + cited anchors; audit = `rm -f data/derived/*.csv && build && git diff`.
  **uv-absent contract:** fail loud and non-zero by default (never skip implicitly);
  `--allow-stale-metric-data` is the opt-in override that proceeds on the existing
  committed file, and even then a missing file fails rather than writing a partial. The
  audit loop requires uv.
