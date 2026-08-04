import type { CsvRow } from '../csv.js'
import { type ExtractResult, fixed, type ObservationRow, yearAsOf } from './types.js'

// IRENA RPGC 2025 -> the onshore-wind total-installed-cost series ($/kW), the curve the
// wind learning rate is fitted against.
// Source: data/sources/irena/IRENA_TEC_RPGC_in_2025_data_file_2026.xlsx, sheet "Fig 2.3"
// ("TICs of onshore wind projects and global weighted-average, 2010-2025"), read by
// extract/irena_tic.py (an .xlsx needs a real reader; see src/steps/build-metric-data.ts).
//
// This module takes the flat {year, cost_usd_per_kw} scratch CSV that script produces.
// The sheet-location and currency gotchas are handled on the Python side, where the raw
// cells are.
//
// WHY THIS SERIES AND NOT LCOE. Wind has two cost series with two different jobs:
//   total_installed_cost ($/kW) -- hardware capex. Wright-fittable: it is the thing that
//                                  gets cheaper as more units are built.
//   LCOE (USD/MWh)              -- delivered energy. Viability/threshold only.
// LCOE also falls through capacity-factor and financing gains, which are not manufacturing
// learning; fitting it gave 43%/doubling, roughly double any published onshore figure.
// This series fits at ~25%/doubling. Same technology, right quantity, sane answer.

// Loads against the wind dependency rather than a "wind installed cost" one because
// capacity_series is keyed by dependency, and the onshore-wind capacity axis this fit needs
// already hangs off this row. A separate dependency would be orphaned from its own x-axis.
// The naming mismatch (an LCOE-named dependency holding a capex metric) dissolves when the
// technology/dependency split lands.
export const DEPENDENCY_NAME = 'Onshore wind LCOE'
export const METRIC = 'total_installed_cost'
export const UNIT = 'USD/kW'

// Currency basis, matching the LCOE and battery series from the same workbook. The Python
// side refuses a workbook that isn't denominated in 2025 dollars, so this stamp cannot
// silently disagree with the source.
export const BASIS = 'real_2025_usd'

// Onshore only -- offshore has its own cost curve and its own learning rate, and the
// capacity axis this pairs with is onshore-only for the same reason.
export const SEGMENT = 'onshore'

export const SOURCE_URL =
  'https://www.irena.org/Publications/2026/... (RPGC 2025 data file — VERIFY)'
export const SOURCE_NAME =
  'IRENA Renewable Power Generation Costs 2025 (data file 2026), Fig 2.3 (weighted average)'

// Dollars per kW; two decimals keeps the published precision without inviting float noise
// into the rebuild-and-diff audit.
const PRECISION = 2

export const extractIrenaTic = (rows: CsvRow[]): ExtractResult<ObservationRow> => {
  const extractor = 'irena-tic'
  const unparsed: ExtractResult<ObservationRow>['unparsed'] = []
  const points: { year: number; cost: number }[] = []

  for (const row of rows) {
    const year = Number((row.year ?? '').trim())
    const raw = (row.cost_usd_per_kw ?? '').trim()
    const cost = Number(raw)
    if (!Number.isInteger(year) || year <= 0) {
      unparsed.push({ extractor, raw: JSON.stringify(row), reason: 'unparseable year' })
      continue
    }
    if (raw === '' || !Number.isFinite(cost)) {
      unparsed.push({ extractor, raw: JSON.stringify(row), reason: 'unparseable cost_usd_per_kw' })
      continue
    }
    points.push({ year, cost })
  }

  if (points.length === 0) {
    throw new Error(
      `${extractor}: no usable rows in the IRENA TIC scratch CSV — the workbook's layout changed`,
    )
  }

  points.sort((a, b) => a.year - b.year)
  const firstYear = points[0]?.year
  const latestYear = points[points.length - 1]?.year

  const observations = points.map<ObservationRow>((point) => ({
    dependency_name: DEPENDENCY_NAME,
    metric: METRIC,
    value: fixed(point.cost, PRECISION),
    unit: UNIT,
    basis: BASIS,
    segment: SEGMENT,
    as_of: yearAsOf(point.year),
    scope: 'global',
    method: 'curated',
    source_url: SOURCE_URL,
    source_name: SOURCE_NAME,
    note:
      point.year === firstYear
        ? 'series start; hardware capex curve (Wright-fittable, unlike wind LCOE)'
        : point.year === latestYear
          ? 'latest'
          : '',
  }))

  return { rows: observations, unparsed, excluded: 0 }
}
