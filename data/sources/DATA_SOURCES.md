# Project Lazarus — Data Sources Manifest

Provenance register for every raw data input. One row per source file. If a number in
a derived file or a projection needs validating, trace it here → to the raw file in
`data/sources/<provider>/` → to the extractor that produced it.

**Rule:** raw inputs are immutable (never edited in place); derived files are
reproducible from them via `build-metric-data`. This manifest documents the raw side.

**Three input tiers** (the split is *who produced it*, not which company):
- `data/sources/<provider>/` — files a **provider shipped** (OWID CSVs, IRENA `.xlsb`).
- `data/curated/` — numbers/rows a **human authored**, edited in PRs. Includes
  `cited_anchors.csv` (see below) alongside `dependencies.csv`, `dependency_thresholds.csv`, etc.
- `data/derived/` — **generated** by `build-metric-data`, committed for audit, never hand-edited.

So a BNEF *number typed by a human* lives in `curated/cited_anchors.csv`; a BNEF *file*,
if one existed, would live in `sources/bnef/`. The paywalled BNEF/LBNL/World Bank values
below have no shippable file, so they are curated cited anchors, not `sources/` entries.

---

## Metric-data sources (feed cost/capacity/threshold projections)

| file | provider | series | basis | vintage (data years) | retrieved | access | feeds |
|---|---|---|---|---|---|---|---|
| `owid/solar-pv-prices.csv` | OWID (Nemet 2009 / Farmer & Lafond 2016 / IRENA) | solar module price | `real_2024_usd` | 1975–2024 | 2026-07-25 | ourworldindata.org/grapher/solar-pv-prices | solar cost observations |
| `owid/installed-solar-pv-capacity.csv` | OWID (Ember / IRENA) | cumulative solar PV capacity | `AC` | 2000–2024 | 2026-07-25 | ourworldindata.org/grapher/installed-solar-pv-capacity | solar capacity (Wright fit) |
| `irena/IRENA_Stats_Tool_v2.xlsb` | IRENA Renewable Capacity Statistics | onshore wind + PV capacity, generation | `onshore` (wind) | 2000–2024 | 2026-07-25 | irena.org/Data | onshore wind capacity (Wright fit); solar cross-check |
| *(paywalled — no file)* | BloombergNEF Battery Price Survey | battery pack price (all-segment / BEV / stationary) | `real_usd` (mixed base) | 2010–2025 | 2026-07-25 | about.bnef.com (annual survey, Dec) | battery cost anchors |
| *(paywalled — no file)* | Ziegler & Trancik 2021 | battery cumulative-GWh + learning rate (20%) | `energy` (GWh) | 1992–2016 | 2026-07-25 | arxiv.org/abs/2007.13920 | battery learning rate (published) |
| *(cited — no file)* | LBNL "Queued Up" | interconnection queue wait | `na` (years) | 2000–2025 | 2026-07-25 | emp.lbl.gov/queues | interconnection observations |
| *(cited — no file)* | World Bank State & Trends of Carbon Pricing | carbon price (global / regional) | `nominal_usd` | 2026 (current only) | 2026-07-25 | worldbank.org/en/publication/state-and-trends-of-carbon-pricing | carbon price observations |
| *(cited — no file)* | IRENA RPGC 2025 | onshore wind LCOE | `real_usd` | 2010, 2025 (thin) | 2026-07-25 | irena.org/Publications | wind LCOE observations |
| *(cited — no file)* | BNEF NEO 2025 / ETIT 2026 | capital, investment, ETS trajectories | mixed | 2024–2025 | 2026-07-25 | about.bnef.com | capital dependencies; conditional-state context |

## Company / outcome sources (feed the company + outcome pipeline — different subsystem)

| dir | provider | content | access | feeds |
|---|---|---|---|---|
| `crunchbase/` | Crunchbase (derived, via Kaggle) | company status/funding/acquisition | terms-restricted | entity enrichment, outcome axis |
| `startup-failures/` | CB Insights post-mortems (via Kaggle) | failed-company names + lifecycle facts + failure reason | terms-restricted (seed list only) | seed list for outcome pass |

## Curated anchor files (hand-entered; see the no-file note below)

| file | rows | feeds |
|---|---|---|
| `data/curated/cited_anchors.csv` | BNEF battery prices, LBNL queue waits, World Bank carbon prices, IRENA RPGC LCOE, DAC / hydrogen / electrolyzer / climate-capital single points | `metric_observations_full.csv` |
| `data/curated/capacity_anchors.csv` | Li-ion cumulative-GWh anchor | `capacity_series.csv` |

Two files rather than one because the observation and capacity schemas differ (`technology`
+ `scenario` vs `dependency_name`). The build rejects any anchor row missing `source_url` +
`source_name` — a cited-only number with no citation is unfalsifiable.

---

## Notes per source

- **OWID files** ship with a `readme.md` + `.metadata.json` in the download — keep both
  alongside the CSV in `data/sources/owid/` as the provider-level provenance.
- **IRENA `.xlsb` extraction gotcha:** capacity lives on `Producer Type = "On-grid
  electricity"` / `"Off-grid electricity"` rows; the `"All types"` rows carry **only**
  the finance indicator and have **null capacity**. Summing must include on-grid +
  off-grid and exclude `"All types"`, or capacity reads as zero. (Encoded in the
  extractor.)
- **Solar DC vs AC:** OWID/IRENA is `AC`; REN21/GSR/Statista headline figures are `DC`
  and run ~20% higher post-2021. Use `AC` (matches the cost source). See baseline spec.
- **Paywalled / no-file sources** (BNEF battery, LBNL, World Bank, IRENA RPGC, DAC,
  hydrogen, electrolyzer, capital, Ziegler-Trancik rate) have no shippable file. Their
  hand-entered numbers live in **`data/curated/cited_anchors.csv`** (one row per value,
  keyed by provider via `source_name`), passed through by a thin extractor. Editing a
  datum is a CSV/PR change, never a code change. This manifest is the register; the CSV
  is the data.

## Derived outputs (regenerated, not raw)

| file | built from | by |
|---|---|---|
| `data/derived/metric_observations_full.csv` | `sources/owid` (`owid-solar-cost`), `curated/cited_anchors.csv` (`cited-anchors`) | `build-metric-data` |
| `data/derived/capacity_series.csv` | `sources/owid` (`owid-solar-capacity`), `sources/irena` (`irena-capacity`), `curated/capacity_anchors.csv` (`capacity-anchors`) | `build-metric-data` |

Extractors live in `src/lib/extractors/`, one per source, each stamping the `basis`,
`source_url` and `source_name` recorded above. The IRENA `.xlsb` needs a binary-workbook
reader, so its extractor has a Python half (`extract/irena_capacity.py`, run via uv); the
other three are pure TypeScript. Audit the whole set with:

```
rm -f data/derived/*.csv && npm run build-metric-data && git diff
```

`build-metric-data` **stops at writing `data/derived/`.** It does not modify `build-db`.
The DB-side switch — adding `basis` to the `metric_observations` table, the
`capacity_series` table, the Wright fit — is a separate change owned by
`hero-thresholds-spec-v3.1.md`, sequenced after this. Today nothing in `src/` reads the
derived files, which is why producing them first is safe.

## Adding a source (checklist)

1. Drop the raw file in `data/sources/<provider>/` (unzip there; keep any provider readme).
2. Add a row to this manifest with basis + vintage + retrieved date + `feeds`.
3. Add/extend the provider extractor in `build-metric-data`.
4. Re-run `build-metric-data`; review the diff in `data/derived/`.
5. If a new methodology gotcha appears, log it in the baseline spec's changelog.
