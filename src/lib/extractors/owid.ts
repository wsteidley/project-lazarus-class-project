import type { CsvRow } from '../csv.js'
import type { Unparsed } from './types.js'

// Shared shape of every OWID grapher download: Entity, Code, Year, <one value column>.
// Both OWID extractors need the same World filter and the same year/value parsing, and
// getting the filter wrong is the expensive mistake here — a per-country file summed or
// averaged by accident looks plausible and is wrong by an order of magnitude.

export type OwidPoint = { year: number; value: number }

// The World aggregate row. OWID ships per-country rows in the same file; we want the
// global series.
const WORLD_ENTITY = 'World'

export type OwidSeries = {
  points: OwidPoint[]
  // Rows that should have parsed and didn't — real anomalies, surfaced individually.
  unparsed: Unparsed[]
  // Rows dropped by the declared World filter. An intentional exclusion, counted rather
  // than listed, and kept separate from `unparsed` so the anomaly count stays meaningful.
  excluded: number
}

export const readOwidWorldSeries = (
  rows: CsvRow[],
  valueColumn: string,
  extractor: string,
): OwidSeries => {
  const points: OwidPoint[] = []
  const unparsed: Unparsed[] = []
  let excluded = 0

  for (const row of rows) {
    if ((row.Entity ?? '').trim() !== WORLD_ENTITY) {
      excluded += 1
      continue
    }
    const year = Number((row.Year ?? '').trim())
    const raw = (row[valueColumn] ?? '').trim()
    const value = Number(raw)
    if (!Number.isInteger(year) || year <= 0) {
      unparsed.push({ extractor, raw: JSON.stringify(row), reason: 'unparseable Year' })
      continue
    }
    if (raw === '' || !Number.isFinite(value)) {
      unparsed.push({
        extractor,
        raw: JSON.stringify(row),
        reason: `unparseable "${valueColumn}"`,
      })
      continue
    }
    points.push({ year, value })
  }

  // An empty series is never a legitimate outcome here — it means the download's shape
  // changed (renamed aggregate, renamed value column) and the filter silently matched
  // nothing. Fail loudly rather than committing an empty derived file.
  if (points.length === 0) {
    throw new Error(
      `${extractor}: no "${WORLD_ENTITY}" rows with a numeric "${valueColumn}" in ${rows.length} input rows — ` +
        'the OWID download’s columns or entity naming changed',
    )
  }

  points.sort((a, b) => a.year - b.year)
  return { points, unparsed, excluded }
}
