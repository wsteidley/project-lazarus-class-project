# Project Lazarus — Data Derivation Map

Answers "where does this number come from?" at the file→metric level. One raw source
file can feed **several** derived metrics; this map keeps that explicit so nothing goes
unwired and nobody re-derives the lineage from scratch. Pairs with `DATA_SOURCES.md`
(provenance register) — this is the extraction wiring.

**Chain:** `data/sources/<provider>/<file>` → extractor → rows in `data/derived/`
(`metric_observations`, `capacity_series`) → views (`progress`, `trajectory`,
`projection`).

---

## Source file → derived metrics

### `sources/owid/solar-pv-prices.csv`
| metric | unit | basis | energy_basis | duration | segment | years | → |
|---|---|---|---|---|---|---|---|
| `module_price` | USD/W | real_2024_usd | na | na | all | 1975–2024 | solar hardware curve; **Wright fit** |

### `sources/owid/installed-solar-pv-capacity.csv`
| metric | unit | basis | energy_basis | duration | segment | years | → |
|---|---|---|---|---|---|---|---|
| `cumulative_capacity` | GW | na | AC | na | all | 2000–2024 | solar Wright capacity axis |

### `sources/irena/IRENA_Stats_Tool_v2.xlsb`  (capacity/generation product — NO cost)
| metric | unit | basis | energy_basis | duration | segment | years | → |
|---|---|---|---|---|---|---|---|
| `cumulative_capacity` (onshore wind) | GW | na | na | na | onshore | 2000–2024 | wind Wright capacity axis |

*(solar capacity available too — OWID is the chosen spine; IRENA = cross-check)*

### `sources/irena/IRENA_TEC_RPGC_in_2025_data_file_2026.xlsx`  (cost report — MULTI-METRIC)
One file, many metrics — extractor emits several `metric` values:
| metric | unit | basis | energy_basis | duration | segment | years | sheet | → |
|---|---|---|---|---|---|---|---|---|
| `LCOE` (7 techs) | USD/MWh | real_2025_usd | na | na | all/onshore/offshore | 2010–2025 | Fig S.2 | viability/threshold side |
| `total_installed_cost` (onshore wind) | USD/kW | real_2025_usd | na | na | onshore | 2010–2025 | **Fig 2.3** | capex curve; **Wright-fitted, 25.0%/doubling** |
| `total_installed_cost` (other 6 techs) | USD/kW | real_2025_usd | na | na | all | 2025 only | ch. tables | snapshot; held until the tech/dependency split |
| `battery_installed_cost` | USD/kWh | real_2025_usd | usable | blended | all | 2010–2025 | Fig 9.2 | **battery cost series** |
| `bess_additions` | GWh/yr | na | na | na | all | 2015–2025 | Fig 9.1 | → cumulate → **battery Wright capacity axis** |
| `capacity_factor` (per tech) | fraction | na | na | na | all | 2025 only | Fig S.1 | snapshot; feeds parametric-LCOE (future) |
| `lcoe_by_market` (solar, wind) | USD/MWh | real_2025_usd | na | na | all | 2025 only | Fig S.3 | populates real `scope` (Global/BR/CN/DE/IN/US) |

> **Extraction status — two paths, one file.**
> - **Read from the `.xlsx` directly (reproducible):** onshore-wind `total_installed_cost`,
>   via `extract/irena_tic.py` (openpyxl) → `src/lib/extractors/irena-tic.ts`. This is the
>   pattern the rest should follow; the chain from committed raw file to derived row is
>   closed and the rebuild-and-diff audit actually verifies it.
> - **Still hand-staged:** LCOE (7 techs, 2010–25) in `irena_lcoe_series.csv` (112 rows),
>   and `irena_rpgc_extended.csv` (52 rows: 16 battery cost + 11 BESS additions + 6
>   installed cost + 7 capacity factor + 12 market LCOE). The extractor reads these CSVs,
>   not the workbook, so their numbers cannot be re-derived — the audit passes them through
>   rather than verifying them.
>
> **⚠ Figure numbers move between editions.** The wind TIC series is **Fig 2.3** in the 2025
> file and **Fig 2.1** in the 2023 one. `irena_tic.py` therefore binds to the sheet *title*
> ("TICs of onshore wind projects"), locates the year header and the `Weighted average` row
> by **label** rather than by index, and refuses any workbook not denominated in 2025 USD.
> This is not defensive over-engineering: a hand-rolled reader of this exact sheet shifted
> every year by one column and produced a confident −87%/doubling that looked like data.
>
> **TODO:** move the remaining metrics onto the `.xlsx` path too.

### `sources/irena/IRENA_statistics_extract_2026H1.xlsx`  (newer capacity extract)
| `cumulative_capacity` | GW | (verify) | (verify) | →2026H1 | — | potential capacity refresh — verify vs Stats Tool before use |

### `curated/cited_anchors.csv`  (hand-entered, paywalled/no-file)
| metric | unit | basis | energy_basis | duration | segment | years | src | → |
|---|---|---|---|---|---|---|---|---|
| `battery pack price` | USD/kWh | real_usd | nameplate | na | all / bev / stationary | 2010–2025 | BNEF | retail-pack cost (distinct from IRENA install cost); **one metric, three segments, one bar** |
| `battery turnkey` | USD/kWh | real_usd | usable | 4h / 2h | all | 2021–2025 | BNEF | turnkey system, duration-specific |
| `interconnection queue wait` | years | na | na | na | US | 2007–2025 | LBNL | interconnection |
| `carbon price` | USD/tCO2e | nominal | na | na | all | 2026 | World Bank | carbon threshold; scope=global/EU/NO |
| capital (growth/early/FOAK), H₂, DAC, electrolyzer | mixed | | | | | current | various | thin/current-only |

---

## What this changes (from the full IRENA extraction)

- **Battery is now fully fittable.** IRENA Fig 9.2 (cost, 16 yrs) + Fig 9.1 (GWh
  additions → cumulative) supply both axes from one open source. **The "use the
  published 20% rate" shortcut is no longer required** — battery gets a real fit like
  solar and wind. (BNEF pack price stays as a *separate* retail metric; IRENA is
  system/project installed cost — different quantities, both kept.)
- **`total_installed_cost` ($/kW)** is a new capex curve per technology — the
  utility-scale analog to solar's module $/W, Wright-fittable.
- **`capacity_factor`** is a new (currently snapshot-only) dimension — the input the
  future parametric-LCOE model needs.
- **`lcoe_by_market`** gives real per-country values → the `scope` dimension gets
  actual data instead of just global.

## Extractor implications (feeds metric-data-structure + data-org specs)

- **An extractor may emit multiple `metric` values from one source file** — the IRENA
  RPGC extractor is the first multi-metric one. The `build-metric-data` extractor
  contract must allow one file → many metric rows (it already returns a row list; just
  don't assume one-metric-per-extractor).
- **New metrics need `technology_metrics` lookup rows** (`total_installed_cost`,
  `battery_installed_cost`, `bess_additions`, `capacity_factor`, `lcoe_by_market`) or
  the FK rejects them. Seed the lookup before loading.
- **FOUR battery cost metrics, none merged (reconciliation RESOLVED via IRENA RPGC ch.9
  + Ember report):** battery cost has four non-interchangeable definitions, ranging
  $70–140/kWh for the *same year* — duration, usable-vs-nameplate, and scope must all
  be explicit or curves get spliced:
  1. **BNEF pack price** — retail Li-ion pack, cited anchor: $108 all-seg / $99 BEV /
     $70 stationary (2025).
  2. **BNEF turnkey system** — cited anchor, duration-specific: $117 avg / **$110 4h** /
     $124 2h (2025). This is the *real* 4-hour number.
  3. **IRENA TIC** — `battery_installed_cost`, `basis=real_2025_usd`,
     `energy_basis=usable`, `duration=blended` (projects span 1.4–4.3h per Fig 9.6),
     16-pt series 2634→140 (2010–2025).
  4. **Ember all-in capex** — point-in-time (Oct 2025), auction-derived, ex-China/US,
     4h+: $125/kWh (= $75 core + $50 EPC/grid). Reference snapshot, not a series.
- **The old curated point was mislabeled — merge WITH relabel.** The existing
  `Utility-scale battery system / 4h system cost` = 140 @ 2025 (bar 150) is **IRENA TIC**
  (matches 140.45), NOT BNEF 4h turnkey (~110). Its `4h` label was the error — IRENA is
  blended duration. Resolution: **merge the IRENA 16-pt series into this metric, relabel
  to `battery_installed_cost` with `basis=real_2025_usd` / `energy_basis=usable` /
  `duration=blended`, keep the 150 bar.** Already below bar → learning rate + `curve_type`,
  no crossing date. (Fitted 23.5%/doubling, R²=0.90 — reads slightly high because the
  BESS capacity axis cumulates from 2015 only, understating early deployment; recorded on
  the data rows.)
  - **Flag for review:** the 150 bar was set for a "4h system" reading; confirm it's the
    right viability bar for blended-duration usable-kWh TIC (likely fine — 140<150 → not
    yet a `receded`/`crossed` edge case, but the bar's provenance should be re-noted).

## Still open

- Verify `IRENA_statistics_extract_2026H1.xlsx` — does it extend capacity beyond the
  Stats Tool, and on what basis? Use only if it's a clean refresh.
- `capacity_factor` and `lcoe_by_market` are 2025 snapshots here; a time series /
  full market set would need the per-chapter tables (Table 2.x etc.) — pull later if
  the parametric-LCOE or scope work needs them.
- Confirm exact IRENA RPGC data-file URL for citations before commit.
