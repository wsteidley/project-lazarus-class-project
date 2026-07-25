import type { CsvRow } from '../csv.js'
import { type CapacityRow, type ExtractResult, fixed, yearAsOf } from './types.js'

// IRENA Renewable Capacity Statistics -> the cumulative onshore-wind deployment series
// the wind learning-rate fit is run against.
// Source: data/sources/irena/IRENA_Stats_Tool_v2.xlsb, read by extract/irena_capacity.py
// (the .xlsb needs a binary-workbook reader; see src/steps/build-metric-data.ts).
//
// This module takes the flat {year, capacity_mw} scratch CSV that script produces. The
// "All types" gotcha is handled on the Python side, where the raw rows are — see the
// note in irena_capacity.py and in DATA_SOURCES.md.

export const TECHNOLOGY = 'Onshore wind LCOE'
export const METRIC = 'cumulative onshore wind capacity'
export const SOURCE_URL = 'https://www.irena.org/Data'
export const SOURCE_NAME = 'IRENA Renewable Capacity Statistics 2025 (IRENA_Stats_Tool_v2)'
export const UNIT = 'GW'

// Onshore only. IRENA's headline "Wind" total folds in offshore, which has its own cost
// curve and its own learning rate; fitting the two together contaminates both.
export const BASIS = 'onshore'

// IRENA reports MW; the series is published in GW. Dividing by 1000 makes three decimals
// the natural precision — it is the MW figure, not a rounding choice.
const MW_PER_GW = 1000
const PRECISION = 3

export const extractIrenaCapacity = (rows: CsvRow[]): ExtractResult<CapacityRow> => {
  const extractor = 'irena-capacity'
  const unparsed: ExtractResult<CapacityRow>['unparsed'] = []
  const points: { year: number; megawatts: number }[] = []

  for (const row of rows) {
    const year = Number((row.year ?? '').trim())
    const raw = (row.capacity_mw ?? '').trim()
    const megawatts = Number(raw)
    if (!Number.isInteger(year) || year <= 0) {
      unparsed.push({ extractor, raw: JSON.stringify(row), reason: 'unparseable year' })
      continue
    }
    if (raw === '' || !Number.isFinite(megawatts)) {
      unparsed.push({ extractor, raw: JSON.stringify(row), reason: 'unparseable capacity_mw' })
      continue
    }
    points.push({ year, megawatts })
  }

  if (points.length === 0) {
    throw new Error(
      `${extractor}: no usable rows in the IRENA scratch CSV — the workbook's layout changed`,
    )
  }

  points.sort((a, b) => a.year - b.year)
  const years = points.map((point) => point.year)
  const firstYear = Math.min(...years)
  const latestYear = Math.max(...years)

  const capacity = points.map<CapacityRow>((point) => ({
    technology: TECHNOLOGY,
    metric: METRIC,
    value: fixed(point.megawatts / MW_PER_GW, PRECISION),
    unit: UNIT,
    basis: BASIS,
    as_of: yearAsOf(point.year),
    scope: 'global',
    scenario: 'historical',
    method: 'curated',
    source_url: SOURCE_URL,
    source_name: SOURCE_NAME,
    note:
      point.year === firstYear
        ? 'IRENA onshore-only (fixes offshore contamination)'
        : point.year === latestYear
          ? 'latest; onshore only'
          : '',
  }))

  return { rows: capacity, unparsed, excluded: 0 }
}
