import type { CsvRow } from '../csv.js'
import { readOwidWorldSeries } from './owid.js'
import { type ExtractResult, fixed, type ObservationRow, yearAsOf } from './types.js'

// OWID "Solar photovoltaic panel prices" -> solar module cost observations.
// Source: data/sources/owid/solar-pv-prices/solar-pv-prices.csv

export const ENTITY_NAME = 'Solar PV'
export const METRIC = 'module price'
export const VALUE_COLUMN = 'Solar PV module cost'
export const SOURCE_URL = 'https://ourworldindata.org/grapher/solar-pv-prices'

// Declared basis. OWID states "constant 2024 US$ per watt" in the shipped metadata; a
// mixed-basis series silently compared against a nominal threshold is the failure this
// stamp exists to prevent.
export const BASIS = 'real_2024_usd'
// A per-watt module price has no energy denominator and no storage duration -- both axes
// genuinely do not apply, which is what 'na' says. It matches only another 'na'.
export const ENERGY_BASIS = 'na'
export const DURATION = 'na'
export const UNIT = 'USD/W'

// OWID stitches three producers into one series. Attributing every row to "OWID" would
// lose that, so each row names the producer whose data it actually is. The boundaries are
// not our judgment — they are stated in the shipped metadata:
//   "Prices are compiled from three sources: Nemet (2009) for 1975-2003,
//    Farmer & Lafond (2016) for 2004-2009, and IRENA for 2010 onward."
// (solar-pv-prices.metadata.json, descriptionKey). If a future download restitches the
// series, these bands must be re-read from that file.
const producerFor = (year: number): string => {
  if (year <= 2003) {
    return 'OWID (Nemet 2009)'
  }
  if (year <= 2009) {
    return 'OWID (Farmer & Lafond 2016)'
  }
  return 'OWID (IRENA)'
}

// The year the project treats as the attempt era for solar — the point a 2000s-era
// company was making its bet against. Curated judgment, not data; it rides on the row so
// the baseline logic downstream can find it. See the hero-thresholds spec.
const ATTEMPT_ERA_YEAR = 1976

// Two decimals. Matches the precision OWID's own chart reports and keeps sub-cent float
// noise out of the committed file.
const PRECISION = 2

export const extractOwidSolarCost = (rows: CsvRow[]): ExtractResult<ObservationRow> => {
  const extractor = 'owid-solar-cost'
  const { points, unparsed, excluded } = readOwidWorldSeries(rows, VALUE_COLUMN, extractor)
  // Non-empty by construction: readOwidWorldSeries throws on an empty series.
  const latestYear = Math.max(...points.map((point) => point.year))

  const observations = points.map<ObservationRow>((point) => ({
    entity_name: ENTITY_NAME,
    metric: METRIC,
    value: fixed(point.value, PRECISION),
    unit: UNIT,
    basis: BASIS,
    energy_basis: ENERGY_BASIS,
    duration: DURATION,
    // OWID publishes one global module price with no on/off-grid or utility/rooftop split,
    // so the series is the rolled-up total.
    segment: 'all',
    as_of: yearAsOf(point.year),
    scope: 'global',
    method: 'curated',
    source_url: SOURCE_URL,
    source_name: producerFor(point.year),
    note:
      point.year === ATTEMPT_ERA_YEAR
        ? 'attempt-era anchor'
        : point.year === latestYear
          ? 'latest OWID point'
          : '',
  }))

  return { rows: observations, unparsed, excluded }
}
