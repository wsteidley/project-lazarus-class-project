# Project Lazarus — Development Ideas & Nice-to-Haves

Running backlog of enhancements that are **not** on the critical path but would add
real value. Not specs — seeds for future specs. Roughly grouped; not ordered by
priority. Add freely; promote to a real spec when one gets picked up.

---

## Projections & metric data

- **Parametric LCOS model (Wright capex × market-condition grid) — Ember handed us the
  spec.** Ember's "How Cheap is Battery Storage?" (Dec 2025) fully specifies the model:
  six input assumptions translate $125/kWh capex → $65/MWh LCOS — **20-yr lifetime,
  7% discount rate, 90% round-trip efficiency, 80% utilisation, 2%/yr degradation,
  2% opex**. Their sensitivity test shows even Lazard-aligned inputs (11% discount, 92%
  eff, 96% util) still yield ~$65/MWh on the same capex. This is the concrete input set
  for the parametric model: Wright-project the *capex*, hold these six as scenario
  variables, output an LCOS surface. Ember also ships a **live LCOS calculator** (capex,
  opex, lifetime, degradation, utilisation, efficiency, discount rate) — a working
  reference implementation. Record the six assumptions + ranges as curated reference;
  build the model when the capex Wright fit is solid. (Battery cost decomposition
  $75 core equipment + $50 EPC/grid is also a useful capex breakdown input.)
- **Ember generation & capacity data — dual use, new subsystem.** Ember's open CSVs
  (monthly/yearly generation global + lower-income + US-subnational; monthly wind/solar
  capacity) serve two purposes: (a) an **alternative/fresher capacity source** (monthly
  GW vs the annual IRENA/OWID series — could strengthen or cross-check the Wright
  capacity axis), and (b) a genuinely **new adoption dimension** — actual deployment
  (TWh generated, GW installed) by country and month over time, which is about *where
  and how fast* technologies scaled, not their cost curves. (b) is closer to
  idea-space/market context than to the hero-metric machinery. Ember is open-data
  (attribution) so both are cleanly usable. Scope which use first before loading —
  they land in different places.
- **Multiple forward-projection paths (ETS + NZS + extrapolation).** Today the Wright
  crossing uses a single forward capacity path extrapolated from historical trend
  (v3.2 R1), which naively doubles forever and ignores saturation/policy. Ingest
  BloombergNEF's **Economic Transition Scenario (ETS, base case)** and **Net Zero
  Scenario (NZS, Paris-aligned)** capacity trajectories as *alternative* forward paths,
  and surface **a range of crossing dates** — "viable ~2029 on trend, ~2027 under NZS
  buildout, ~2031 under ETS." A fan of projections is far more honest than a single
  line, and it directly serves the Lazarus question ("how overdue, and under what
  future?"). Requires a citable source for the scenario series (NEO data viewer URL, or
  a licensed drop) so it doesn't break the cited-anchor invariant. **This is the
  headline nice-to-have.**
- **Saturation-aware extrapolation.** Replace log-linear forward extrapolation with a
  logistic/S-curve fit for technologies approaching market saturation, so far-future
  crossings stop being absurd.
- **Backfill the thin cost series.** Wind LCOE (2 points → IRENA annual), carbon price
  (current-only → EU-ETS annual), battery cost gaps 2011–17/2019–20. Each is a source
  drop + existing extractor.
- **Battery cumulative-GWh series.** Pull the Ziegler-Trancik supplement so battery gets
  a real Wright fit instead of the published-rate shortcut.
- **Confidence bands on projections.** Carry the fit R² and capacity-path uncertainty
  into a projected-crossing *range*, not a point.
- **Auto-refresh of live metrics.** Scheduled pulls for the data-feed heroes (OWID,
  IRENA, BNEF where licensed) so "current" values don't silently go stale.

## Gap typology & analysis (the actual product)

- **Build the gap typology** (`white_space` / `unsampled` / `lazarus_candidate` /
  `tested_dead_end` / `flared_out` / `receded` / `conditional` / `crowded_solved`).
  Much of the upstream data now exists; the typology view itself is still unbuilt.
- **Crossed-then-receded state.** The battery case (crossed $100 on some metrics, ticked
  back up 2024→25) isn't cleanly `receded` (which means "never crossed"). A distinct
  "backslid after crossing" state is more accurate.
- **`related_dependency` chains in analysis.** Use the dependency links
  (interconnection → solar cost) to show *compound* revivability — an idea blocked on
  two dependencies that both improved is a stronger candidate than one.
- **Overdue-ranking view.** `became_viable_date` vs. last-attempt-died, ranked — the
  flagship "which dead ideas are most overdue for a retry" query as a first-class output.

## Exploration & application layer (parked until data is solid)

- The whole `exploration-app-spec-v1.md` — negative-space-first UI, dependency-centric
  discovery, semantic search, NL-to-SQL, hypothesis/annotation loop.
- **The signature plot:** a dependency's progress-to-viability curve over time with the
  death-year of every company blocked on it overlaid. Turns the thesis into a picture.
- **Idea-space head-to-head view:** early failure vs. later success in the same space.
- **Exa-powered targeted discovery** (earns a true `white_space` vs `unsampled`).

## Sourcing & outcomes

- **Retrospective outcome pass at scale** + the Crunchbase/CB Insights licensed path if
  it becomes central.
- **More failure corpora** as seed lists (Failory by permission, sector-specific
  post-mortems).
- **Additional IRENA technologies as idea-space anchors** — the IRENA tool also carries
  geothermal, marine, CSP, biofuels, hydro. Geothermal and marine especially have a rich
  "too early" failure history worth mapping.

## Data infrastructure & rigor

- **Per-value confidence validation (item 4 AUROC)** once the outcome pass has run.
- **`metric_series_status` gating** — decide whether thin series suppress projections or
  just label them.
- **Provenance UI** — surface `basis` + `source_url` next to every crossing verdict, so
  nobody re-derives a wrong answer (the DC-vs-AC / nominal-vs-real class of bug, made
  visible at the point of use).
- **Fix the pre-existing `scrape.test.ts`** (fixture removed in bfb9014) — unrelated
  red test that could hide a real regression.

## Reasoning / methodology

- **Learning-rate cross-check.** Where a fitted rate exists (solar, wind), compare it to
  the published literature rate as an automated sanity check — a fit landing far from
  ~20% for solar is a data-quality alarm.
- **Time-varying viability bars.** Current design fixes the viability threshold; some
  bars (the historical breakeven a specific-era company needed) genuinely moved. A
  second, era-relative bar could distinguish "viable now" from "would have saved them
  then."
