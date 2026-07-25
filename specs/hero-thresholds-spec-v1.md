# Project Lazarus — Hero Thresholds & Metric Observations Spec (v1)

Curated, cited threshold data as **input CSVs** — same pattern as `dependencies.csv`
and `idea_spaces.csv`: git-tracked, diffable, reviewable, editable by hand. This is
the input that turns "the dependency improved" into "it became viable in 2024."

## Two files, two grains

| File | Grain | Write mode |
| --- | --- | --- |
| `data/input/dependencies.csv` (extended) | one row per dependency | **edit in place**, rare |
| `data/input/metric_observations.csv` (new) | one row per (dependency × date × scope) | **append-only** |

Never merge them. A threshold is a standing judgment; an observation is a dated fact.
Overwriting observations would destroy the trajectory the whole project depends on.

---

## 1. `dependencies.csv` — new columns

Existing: `name, category, description, threshold_metric, threshold_value, threshold_unit`

Add:

| Column | Values | Why |
| --- | --- | --- |
| `threshold_kind` | `quantitative_with_threshold` \| `quantitative_tbd` \| `qualitative` | a blank must never be ambiguous |
| `threshold_direction` | `below_is_better` \| `above_is_better` | **required** — see finding 1 |
| `threshold_scope` | `global` \| `US` \| `EU` \| … | carbon price and interconnection are not global scalars |
| `threshold_source_url` | URL | a number without provenance silently decides what looks viable |
| `threshold_as_of` | ISO date | when the bar was last reviewed |
| `threshold_note` | free text | e.g. "viability bar, not historical breakeven" |

**`threshold_direction` is not optional.** Every current hero metric is
"lower is better," but interconnection is measured in *years and rose*. Without
direction, a crossing test cannot distinguish "battery fell past $100/kWh" (good) from
"queue wait rose past 2 years" (bad) — identical arithmetic, opposite meaning.

## 2. `metric_observations.csv` — new, append-only

```csv
dependency_name,metric,value,unit,as_of,scope,method,source_url,source_name,note
```

- `as_of` — ISO date or year the value describes (not when you recorded it).
- `scope` — `global` / `US` / `EU` / `CN` …; lets one dependency carry several regional
  series without them colliding.
- `method` — `curated` (hand-entered from a cited report), `feed` (pulled
  programmatically), `llm` (from the reassess pass). Keeps grounded numbers separable
  from generated ones.
- **Append, never edit.** A correction is a new row with a later `as_of`, not an
  overwrite. Same rule as `dependency_assessments`.

### Seed rows from research (July 2026)

```csv
Lithium-ion battery cost,battery pack price,108,USD/kWh,2025-12,global,curated,https://about.bnef.com/insights/clean-transport/lithium-ion-battery-pack-prices-fall-to-108-per-kilowatt-hour-despite-rising-metal-prices-bloombergnef/,BNEF 2025 Battery Price Survey,all-segment average; stationary 70; BEV 99
Lithium-ion battery cost,battery pack price,97,USD/kWh,2024-12,global,curated,https://about.bnef.com/insights/clean-transport/lithium-ion-battery-pack-prices-fall-to-108-per-kilowatt-hour-despite-rising-metal-prices-bloombergnef/,BNEF 2024 Battery Price Survey,BEV packs first year below 100
Solar module cost,module price,0.115,EUR/W,2026-05,EU,curated,https://www.pvxchange.com/Price-Index,pvXchange Price Index,mainstream modules; rose from 2025 all-time low
Green hydrogen production cost,hydrogen price,4.50,USD/kg,2026-06,global,curated,https://www.greenfueljournal.com/post/green-hydrogen-cost-economics-2026-the-real-path-to-price-parity,Green Fuel Journal 2026,range 4.50-8.00; blue 2.00-3.50
Carbon price,carbon price,21,USD/tCO2e,2026-05,global,curated,https://www.worldbank.org/en/publication/state-and-trends-of-carbon-pricing,World Bank State and Trends 2026,average direct price; ETS ~22, taxes ~19.50
Carbon price,carbon price,169.71,USD/tCO2e,2026-04,NO,curated,https://taxfoundation.org/data/all/eu/carbon-taxes-europe/,Tax Foundation Europe 2026,highest national carbon tax
Grid interconnection capacity,interconnection queue wait,5,years,2025-12,US,curated,https://emp.lbl.gov/queues,LBNL Queued Up 2025,median IR-to-COD for projects built 2025
Grid interconnection capacity,interconnection queue wait,2,years,2007-12,US,curated,https://emp.lbl.gov/queues,LBNL Queued Up,under 2 years for projects built 2000-2007
```

### Threshold status implied by the above

| Dependency | Threshold | Status |
| --- | --- | --- |
| Lithium-ion battery cost | 100 USD/kWh, below_is_better | **crossed ~2024** |
| Solar module cost | 0.20 USD/W, below_is_better | **crossed** (well before 2020) |
| Green hydrogen | 2 USD/kg, below_is_better | not crossed; **moving away** |
| Carbon price | 50 USD/tCO2e, above_is_better | not crossed globally; crossed in NO/SE/CH/EU |
| Grid interconnection | 2 years, below_is_better | **crossed in the wrong direction** |

## 3. Source registry (`data/input/metric_sources.csv`, optional)

Refresh cadence per source, so updating is a scheduled chore rather than a memory test:

| Source | Covers | Cadence |
| --- | --- | --- |
| BNEF Battery Price Survey | battery $/kWh | annual, December |
| pvXchange Price Index | module €/W | monthly |
| LBNL "Queued Up" | interconnection years | annual (Excel data file published) |
| World Bank State and Trends | carbon price | annual, May |
| Our World in Data / IRENA | solar, wind LCOE long series | annual |

## 4. Derivation rules (consuming the data)

- **`became_viable_date`** = earliest `as_of` where the observation crosses the
  threshold *in the direction `threshold_direction` specifies*.
- **`currently_crossed`** = does the **latest** observation still satisfy it? Crossings
  can un-cross — solar module prices rose in early 2026 after a 2025 low, and green
  hydrogen's cost curve reversed outright. `became_viable_date` alone would report an
  idea viable when it no longer is.
- **`receded`** — a new trajectory state for dependencies where the metric moved
  *away* from the threshold. Interconnection is neither `lazarus_candidate` (blocker
  improved) nor `tested_dead_end` (blocker unchanged): a company that died on
  interconnection in 2010 faces a *harder* problem today. Deserves its own cell in the
  gap typology — the signal is "still moving away," not "try again."
- **Scope matching:** never compare an observation to a threshold of a different
  `scope`. A global carbon average must not be tested against a bar meant for the EU.

## 5. Loading and validation

- Validate `dependency_name` against the canonical `dependencies.csv` the same way
  `step1c` validates the model's dependency picks — unmatched rows are **reported, not
  dropped**.
- Enforce: `threshold_kind = quantitative_with_threshold` ⇒ value, unit, direction,
  and source_url all present.
- Enforce: `method = curated` ⇒ `source_url` non-empty.
- `feed`-method rows may be written programmatically; `curated` rows are hand-edited
  and reviewed in the PR.

## Open decisions

- Do observations land straight in `dependency_assessments` at build time, or in their
  own `metric_observations` table joined to it? (Own table keeps grounded facts
  separable from LLM verdicts — recommended.)
- Which of the ~10 remaining dependencies get promoted from `quantitative_tbd` — wind
  LCOE, electrolyzer capex $/kW, and DAC $/ton are the obvious next three.
- Whether `receded` is a typology cell only, or also a stored trajectory status.
