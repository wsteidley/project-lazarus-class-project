import type { CsvRow } from '../csv.js'
import { readOwidWorldSeries } from './owid.js'
import { type CapacityRow, type ExtractResult, fixed, yearAsOf } from './types.js'

// OWID "Installed solar PV capacity" -> the cumulative-deployment series the solar
// learning-rate (Wright) fit is run against.
// Source: data/sources/owid/installed-solar-pv-capacity/installed-solar-pv-capacity.csv

export const ENTITY_NAME = 'Solar PV'
export const METRIC = 'cumulative solar PV capacity'
export const VALUE_COLUMN = 'Solar'
export const SOURCE_URL = 'https://ourworldindata.org/grapher/installed-solar-pv-capacity'
export const SOURCE_NAME = 'OWID (Ember/IRENA)'
export const UNIT = 'GW'

// AC, not DC. REN21/GSR/Statista headline capacity figures are DC and run ~20% higher
// post-2021; pairing a DC capacity series with the AC-consistent cost series would bend
// the learning rate. The cost source is AC, so this is too. See DATA_SOURCES.md.
//
// This rode in `basis` until the three-way split, where it was the clearest case of the
// conflation: AC-vs-DC is an ENERGY denominator, never a currency.
export const ENERGY_BASIS = 'AC'
// A capacity figure has no currency vintage.
export const BASIS = 'na'
export const DURATION = 'na'
export const SEGMENT = 'all'

// OWID's series starts in 2000. Earlier years exist in other compilations but not on a
// consistent basis, so the fit starts where this file does.
const PRECISION = 2

export const extractOwidSolarCapacity = (rows: CsvRow[]): ExtractResult<CapacityRow> => {
  const extractor = 'owid-solar-capacity'
  const { points, unparsed, excluded } = readOwidWorldSeries(rows, VALUE_COLUMN, extractor)
  // Non-empty by construction: readOwidWorldSeries throws on an empty series.
  const years = points.map((point) => point.year)
  const firstYear = Math.min(...years)
  const latestYear = Math.max(...years)

  const capacity = points.map<CapacityRow>((point) => ({
    entity_name: ENTITY_NAME,
    metric: METRIC,
    value: fixed(point.value, PRECISION),
    unit: UNIT,
    basis: BASIS,
    energy_basis: ENERGY_BASIS,
    duration: DURATION,
    segment: SEGMENT,
    as_of: yearAsOf(point.year),
    scope: 'global',
    scenario: 'historical',
    method: 'curated',
    source_url: SOURCE_URL,
    source_name: SOURCE_NAME,
    note:
      point.year === firstYear
        ? 'OWID/IRENA World series start'
        : point.year === latestYear
          ? 'latest'
          : '',
  }))

  return { rows: capacity, unparsed, excluded }
}
