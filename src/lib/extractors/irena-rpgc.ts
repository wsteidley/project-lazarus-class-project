import type { CsvRow } from '../csv.js'
import {
  type CapacityRow,
  type ExtractResult,
  fixed,
  OBSERVATION_COLUMNS,
  type ObservationRow,
  type Unparsed,
} from './types.js'

// IRENA Renewable Power Generation Costs 2025 (data file 2026) -> observations + the battery
// capacity axis. Two source extracts, one extractor, several metrics: this is the first
// multi-metric extractor (see data-derivation-map.md).
//
//   sources/irena/irena_lcoe_series.csv     LCOE, 7 technologies, 2010-2025
//   sources/irena/irena_rpgc_extended.csv   battery cost, BESS additions, installed cost,
//                                           capacity factor, market LCOE
//
// PROVENANCE CAVEAT: both files are hand-staged extracts, already in output column order, so
// this is a validating pass-through rather than a real workbook reader. The raw
// IRENA_TEC_RPGC_in_2025_data_file_2026.xlsx is now committed alongside them, so these CSVs
// can and should be replaced by a genuine .xlsx extractor -- until then the audit contract
// (rebuild -> git diff) cannot actually verify these rows against the publisher's file.
//
// The three transforms below are the only things this module does beyond validation, and each
// one exists because skipping it corrupts a series silently rather than loudly.

const AS_OF = /^\d{4}-\d{2}$/

// DEPENDENCY_ALIAS used to sit here: a three-entry (technology, metric) -> dependency map that
// bridged IRENA's per-technology publishing onto a loader that resolved through
// dependencies.name. Every technology it did not name -- CSP, hydro, geothermal, bioenergy,
// offshore wind -- had all 120 of its rows held back, because metric_observations.dependency_id
// was a NOT NULL foreign key and there was no dependency to point at.
//
// The technology/dependency split retired it, exactly as its own comment promised (deleted,
// not extended). Rows now carry the published technology name straight through as entity_name,
// and whether any dependency hangs off that entity is a separate question answered by
// dependency_links.
// Published scope label -> the canonical scope value. Two separate problems:
//   1. casing -- the extended file writes 'global' in most rows but 'Global' in the market-LCOE
//      rows. scope is part of the series key in progress, trajectory, the Wright input query and
//      the dependency_thresholds PK, so 'Global' and 'global' would become two series, one of
//      which quietly matches no threshold and produces nothing.
//   2. country names -- the rest of the dataset uses short codes (US, EU, NO), so full names
//      have to be folded onto the same vocabulary or the same country arrives under two labels.
const SCOPE = new Map<string, string>([
  ['global', 'global'],
  ['brazil', 'BR'],
  ['china', 'CN'],
  ['germany', 'DE'],
  ['india', 'IN'],
  ['united states', 'US'],
])

// Metrics carrying a slice that must not be read as a total. Everything else IRENA publishes
// here is already the rolled-up figure.
const BESS_METRIC = 'bess_additions'
const BATTERY_COST_METRIC = 'battery_installed_cost'

// The cumulative-GWh axis the battery Wright fit runs against, built from annual additions.
const BESS_CAPACITY_METRIC = 'cumulative BESS deployed'
const BESS_PRECISION = 2

const validate = (row: CsvRow): string | null => {
  const missing = ['technology', 'metric', 'value', 'as_of', 'scope'].filter(
    (column) => !(column in row),
  )
  if (missing.length > 0) {
    return `missing column(s): ${missing.join(', ')}`
  }
  if (!(row.technology ?? '').trim()) {
    return 'empty technology'
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
  if (!(row.source_url ?? '').trim() || !(row.source_name ?? '').trim()) {
    return 'IRENA row without source_url + source_name'
  }
  // An unrecognised scope is reported rather than passed through: silently admitting a new
  // label is how a series ends up keyed to something no threshold will ever join.
  if (!SCOPE.has((row.scope ?? '').trim().toLowerCase())) {
    return `unrecognised scope "${row.scope ?? ''}"`
  }
  return null
}

export type IrenaResult = {
  observations: ExtractResult<ObservationRow>
  capacity: ExtractResult<CapacityRow>
}

export const extractIrenaRpgc = (rows: CsvRow[]): IrenaResult => {
  const extractor = 'irena-rpgc'
  const unparsed: Unparsed[] = []
  const observations: ObservationRow[] = []
  const bess: { as_of: string; value: number; row: CsvRow }[] = []
  const excluded = 0

  for (const row of rows) {
    const problem = validate(row)
    if (problem) {
      unparsed.push({ extractor, raw: JSON.stringify(row), reason: problem })
      continue
    }

    const technology = (row.technology ?? '').trim()
    const metric = (row.metric ?? '').trim()

    const scope = SCOPE.get((row.scope ?? '').trim().toLowerCase()) as string
    const as_of = (row.as_of ?? '').trim()

    // BESS additions are an annual FLOW, not a level. They never become an observation; they
    // are cumulated into a capacity series below. Storing the flow as if it were a level would
    // put a 306 GWh point on an axis whose 2025 value is ~700 GWh.
    if (metric === BESS_METRIC) {
      bess.push({ as_of, value: Number((row.value ?? '').trim()), row })
      continue
    }

    observations.push({
      // The published technology name, straight through. Nothing is held back any more:
      // whether a dependency hangs off this entity is dependency_links' business, not a
      // precondition for storing the observation.
      entity_name: technology,
      metric,
      value: (row.value ?? '').trim(),
      unit: (row.unit ?? '').trim(),
      basis: (row.basis ?? '').trim() || 'na',
      ...basisFor(metric),
      segment: (row.segment ?? '').trim() || 'all',
      as_of,
      scope,
      method: (row.method ?? '').trim() || 'curated',
      source_url: (row.source_url ?? '').trim(),
      source_name: (row.source_name ?? '').trim(),
      note: noteFor(metric, (row.note ?? '').trim()),
    })
  }

  return {
    observations: { rows: observations, unparsed, excluded },
    capacity: { rows: cumulateBess(bess), unparsed: [], excluded: 0 },
  }
}

// IRENA's battery figure is not interchangeable with the other three battery cost definitions
// in this project (BNEF pack, BNEF turnkey, Ember all-in), which span $70-140/kWh for the SAME
// year. Recording what this one measures on every row is what stops a later reader splicing
// them into one curve.
//
// The measurement qualifiers used to ride in this note string, because `basis` already meant
// the currency and one column could not carry both. The three-way split gave them real
// columns, so they now live in energy_basis/duration where the progress join can ENFORCE them
// (see basisFor below) rather than merely record them. What stays in the note is the part
// that is genuine provenance rather than a misfiled column: the interchangeability warning.
const IRENA_TIC_NOTE =
  'IRENA total installed cost; not interchangeable with BNEF pack, BNEF turnkey or Ember all-in'

const noteFor = (metric: string, existing: string): string => {
  if (metric !== BATTERY_COST_METRIC) {
    return existing
  }
  return existing ? `${IRENA_TIC_NOTE}; ${existing}` : IRENA_TIC_NOTE
}

// IRENA's battery installed cost is quoted per USABLE kWh over a mixed-duration fleet
// (1.4-4.3h). Both facts are now equality-join keys: a bar declared per nameplate kWh, or on a
// straight 4h system, no longer silently joins these observations. Every other metric in this
// source is a per-kW or per-MWh figure with neither axis, hence 'na'.
const basisFor = (metric: string): { energy_basis: string; duration: string } =>
  metric === BATTERY_COST_METRIC
    ? { energy_basis: 'usable', duration: 'blended' }
    : { energy_basis: 'na', duration: 'na' }

// Annual additions -> a running total. Two things a reader has to be told, both recorded on the
// rows themselves rather than in a comment nobody reads at query time:
//
//   1. TRUNCATION. The additions series starts in 2015, so the total omits whatever grid BESS
//      existed before then. That understates cumulative capacity, which BIASES THE LEARNING RATE
//      UPWARD -- a log-log fit reads the same cost fall against too little deployment. Pre-2015
//      grid-scale BESS was small (<5 GWh) so the distortion is modest, but an unstated
//      truncation inside a fitted curve is precisely what the audit contract exists to prevent.
//   2. SCOPE. This is grid BESS only. It is NOT the same quantity as the existing
//      capacity_anchors.csv Li-ion anchor (3500 GWh @ 2024), which counts all Li-ion including
//      EV packs. They stay separate series on separate dependencies; merging them would pair a
//      grid-battery cost curve with an EV-dominated deployment axis.
const cumulateBess = (points: { as_of: string; value: number; row: CsvRow }[]): CapacityRow[] => {
  const sorted = [...points].sort((a, b) => (a.as_of < b.as_of ? -1 : a.as_of > b.as_of ? 1 : 0))
  let total = 0
  return sorted.map((point, index) => {
    total += point.value
    return {
      // The publisher's own string, verbatim. Mapping it to a tidier name would be a new
      // alias map, which is precisely what the split just deleted.
      entity_name: 'Lithium-ion battery cost',
      metric: BESS_CAPACITY_METRIC,
      value: fixed(total, BESS_PRECISION),
      unit: 'GWh',
      // basis was 'energy' here, which was just restating the unit. A cumulative GWh figure
      // has no currency, no nameplate/usable distinction and no duration.
      basis: 'na',
      energy_basis: 'na',
      duration: 'na',
      segment: 'all',
      as_of: point.as_of,
      scope: 'global',
      scenario: 'historical',
      method: 'curated',
      source_url: (point.row.source_url ?? '').trim(),
      source_name: (point.row.source_name ?? '').trim(),
      note:
        index === 0
          ? `cumulated from annual BESS additions; series starts ${point.as_of} so pre-${point.as_of.slice(0, 4)} stock is EXCLUDED (understates cumulative, biases learning rate upward); grid BESS only, not all Li-ion`
          : 'cumulated from annual BESS additions; grid BESS only',
    }
  })
}

// Re-exported for the test suite, which asserts the derived file's column order is the schema's
// rather than whatever the staged CSV happened to carry.
export { OBSERVATION_COLUMNS }
