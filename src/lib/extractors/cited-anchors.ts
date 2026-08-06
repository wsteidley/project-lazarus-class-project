import type { CsvRow } from '../csv.js'
import {
  CAPACITY_COLUMNS,
  type CapacityRow,
  type ExtractResult,
  OBSERVATION_COLUMNS,
  type ObservationRow,
  type Unparsed,
} from './types.js'

// Pass-through extractor for sources with no committed file: paywalled surveys (BNEF),
// figures cited from a report PDF (LBNL, World Bank, IRENA RPGC), and single-point
// anchors from press coverage. Those numbers are hand-entered into
// data/curated/cited_anchors.csv and data/curated/capacity_anchors.csv.
//
// They live in `curated/` rather than in this file because a data fix must not be a code
// change: correcting a battery price is a one-line CSV edit reviewable by someone who
// doesn't read TypeScript. What this module adds is validation — a hand-typed row that
// can't be used is reported, not carried silently into the derived output.

const AS_OF = /^\d{4}-\d{2}$/

const validate = (row: CsvRow, columns: string[], keyColumn: string): string | null => {
  const missing = columns.filter((column) => !(column in row))
  if (missing.length > 0) {
    return `missing column(s): ${missing.join(', ')}`
  }
  if (!(row[keyColumn] ?? '').trim()) {
    return `empty ${keyColumn}`
  }
  if (!(row.metric ?? '').trim()) {
    return 'empty metric'
  }
  const value = Number((row.value ?? '').trim())
  if ((row.value ?? '').trim() === '' || !Number.isFinite(value)) {
    return `non-numeric value "${row.value ?? ''}"`
  }
  if (!AS_OF.test((row.as_of ?? '').trim())) {
    return `as_of "${row.as_of ?? ''}" is not YYYY-MM`
  }
  // A cited-only source with no citation is the one thing this file cannot be allowed to
  // contain — it would be an unfalsifiable number in an auditable dataset.
  if (!(row.source_url ?? '').trim() || !(row.source_name ?? '').trim()) {
    return 'cited anchor without source_url + source_name'
  }
  return null
}

// Reorders each row into the canonical column order and drops any extra columns, so the
// derived file's shape is set by the schema rather than by whatever the CSV happened to
// carry.
const project = <Row>(row: CsvRow, columns: string[]): Row =>
  Object.fromEntries(columns.map((column) => [column, row[column] ?? ''])) as Row

export const extractCitedAnchors = (rows: CsvRow[]): ExtractResult<ObservationRow> => {
  const extractor = 'cited-anchors'
  const unparsed: Unparsed[] = []
  const kept: ObservationRow[] = []

  for (const row of rows) {
    const problem = validate(row, OBSERVATION_COLUMNS, 'entity_name')
    if (problem) {
      unparsed.push({ extractor, raw: JSON.stringify(row), reason: problem })
      continue
    }
    kept.push(project<ObservationRow>(row, OBSERVATION_COLUMNS))
  }

  return { rows: kept, unparsed, excluded: 0 }
}

export const extractCapacityAnchors = (rows: CsvRow[]): ExtractResult<CapacityRow> => {
  const extractor = 'capacity-anchors'
  const unparsed: Unparsed[] = []
  const kept: CapacityRow[] = []

  for (const row of rows) {
    const problem = validate(row, CAPACITY_COLUMNS, 'entity_name')
    if (problem) {
      unparsed.push({ extractor, raw: JSON.stringify(row), reason: problem })
      continue
    }
    kept.push(project<CapacityRow>(row, CAPACITY_COLUMNS))
  }

  return { rows: kept, unparsed, excluded: 0 }
}
