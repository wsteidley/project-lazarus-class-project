// The shared contract every provider extractor implements. Each one turns exactly one
// raw source into rows that are already stamped with their own provenance — an extractor
// knows its basis, its source_url and its source_name, so no downstream assembly step has
// to guess where a number came from. The manifest (data/sources/DATA_SOURCES.md) and these
// stamps must agree; that agreement is what makes a derived number traceable.
//
// Nothing here does I/O. The step shell (src/steps/build-metric-data.ts) reads files and
// writes them; these modules are pure so they can be tested against literal rows.

// One dated fact about where a metric stood. Column order is the file's column order.
export type ObservationRow = {
  dependency_name: string
  metric: string
  value: string
  unit: string
  basis: string
  as_of: string
  scope: string
  method: string
  source_url: string
  source_name: string
  note: string
}

// One point on a cumulative-deployment curve, for the Wright/learning-rate fit. Keyed by
// technology rather than dependency because a capacity curve belongs to the technology,
// and `scenario` distinguishes observed history from any projected extension.
export type CapacityRow = {
  technology: string
  metric: string
  value: string
  unit: string
  basis: string
  as_of: string
  scope: string
  scenario: string
  method: string
  source_url: string
  source_name: string
  note: string
}

export const OBSERVATION_COLUMNS: (keyof ObservationRow)[] = [
  'dependency_name',
  'metric',
  'value',
  'unit',
  'basis',
  'as_of',
  'scope',
  'method',
  'source_url',
  'source_name',
  'note',
]

export const CAPACITY_COLUMNS: (keyof CapacityRow)[] = [
  'technology',
  'metric',
  'value',
  'unit',
  'basis',
  'as_of',
  'scope',
  'scenario',
  'method',
  'source_url',
  'source_name',
  'note',
]

// Report-not-drop: a row an extractor could not turn into output is described here and
// surfaced in the build summary, never silently discarded. `raw` is the offending input
// rendered for a human; `reason` says what was wrong with it.
export type Unparsed = { extractor: string; raw: string; reason: string }

// `excluded` counts rows dropped by a declared, intentional filter (OWID's per-country
// rows, IRENA's "All types" finance rows). Kept apart from `unparsed` so the anomaly
// count stays a signal: unparsed > 0 wants a human, excluded > 0 is business as usual.
export type ExtractResult<Row> = { rows: Row[]; unparsed: Unparsed[]; excluded: number }

// Fixed-precision rendering, so the same input always produces byte-identical output and
// `git diff` after a rebuild is a real audit rather than float noise. Trailing zeros are
// dropped (2.00 -> "2"), which is lossless for a CSV that is re-parsed as a number.
export const fixed = (value: number, digits: number): string =>
  String(Number(value.toFixed(digits)))

// OWID reports calendar years; the pipeline dates observations to the year's close.
export const yearAsOf = (year: number): string => `${year}-12`

// Deterministic ordering. Plain code-unit comparison, not localeCompare, because a
// locale-sensitive sort would make the output depend on the machine that built it.
const compare = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0)

const by =
  <Row>(keys: (keyof Row)[]) =>
  (a: Row, b: Row): number => {
    for (const key of keys) {
      const result = compare(String(a[key]), String(b[key]))
      if (result !== 0) {
        return result
      }
    }
    return 0
  }

export const sortObservations = (rows: ObservationRow[]): ObservationRow[] =>
  [...rows].sort(by<ObservationRow>(['dependency_name', 'metric', 'scope', 'as_of']))

export const sortCapacity = (rows: CapacityRow[]): CapacityRow[] =>
  [...rows].sort(by<CapacityRow>(['technology', 'metric', 'scope', 'as_of']))
