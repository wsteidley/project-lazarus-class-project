import type { CsvRow } from './csv.js'
import { matchCanonical } from './dependency-resolution.js'

// Validation for the threshold data (dependencies.csv threshold columns) and for the
// derived metric series (data/derived/metric_observations_full.csv, capacity_series.csv —
// the capacity rows reuse this by aliasing `technology` to `dependency_name`, since both
// resolve against the same canonical dependency list). All follow the standing rule:
// unmatched or
// non-conforming rows are *reported*, never silently dropped — a number without provenance
// silently deciding what looks viable is exactly the failure this data exists to prevent.

// A metric observation resolved against the canonical reference-entity list. `entity_name`
// carries the canonical spelling when matched, or '' when nothing fit (kept, not dropped).
// It resolved against DEPENDENCIES until the technology/dependency split; metric data is
// published per technology, and pretending otherwise is what held 120 rows back.
export type ResolvedObservation = CsvRow & { entity_name: string }

export type ObservationValidation = {
  resolved: ResolvedObservation[]
  // Rows whose entity_name matched no canonical reference entity.
  unmatched: CsvRow[]
  // curated rows missing a source_url — a curated number must be citable.
  missingSource: CsvRow[]
}

// Maps each observation onto the canonical entity list (reusing matchCanonical, the
// same case/whitespace-tolerant match step1c uses) and flags the two integrity problems
// the spec calls out: an unresolvable entity_name, and a curated row without a source.
export const validateObservationRows = (
  rows: CsvRow[],
  canonicalNames: readonly string[],
): ObservationValidation => {
  const resolved: ResolvedObservation[] = []
  const unmatched: CsvRow[] = []
  const missingSource: CsvRow[] = []

  for (const row of rows) {
    const canonical = matchCanonical(row.entity_name, canonicalNames)
    if (!canonical) {
      unmatched.push(row)
    }
    if (row.method === 'curated' && !(row.source_url ?? '').trim()) {
      missingSource.push(row)
    }
    resolved.push({ ...row, entity_name: canonical ?? '' })
  }

  return { resolved, unmatched, missingSource }
}

// A threshold-bar row resolved against the canonical dependency list.
export type ResolvedThreshold = CsvRow & { dependency_name: string }

// A dependency declared quantitative_with_threshold but lacking a complete bar (value, unit,
// direction, source_url all present on at least one of its threshold rows).
export type IncompleteThreshold = { name: string; missing: string[] }

export type ThresholdValidation = {
  resolved: ResolvedThreshold[]
  // Threshold rows whose dependency_name matched no canonical dependency.
  unmatched: CsvRow[]
  // Dependencies whose declared kind promises a bar they don't fully carry.
  incomplete: IncompleteThreshold[]
}

// Validates dependency_thresholds.csv against the dependency seed. Threshold rows resolve to
// canonical names (reusing matchCanonical); unmatched rows are reported, not dropped. Then
// the v1 rule is enforced per dependency across its bars: threshold_kind =
// quantitative_with_threshold ⇒ at least one bar with value, unit, direction, and source_url
// all present. Reports offenders; never throws or mutates.
export const validateThresholds = (
  dependencyRows: CsvRow[],
  thresholdRows: CsvRow[],
  canonicalNames: readonly string[],
): ThresholdValidation => {
  const resolved: ResolvedThreshold[] = []
  const unmatched: CsvRow[] = []
  // canonical name -> its complete-enough bars
  const completeByName = new Map<string, number>()

  const required = [
    'threshold_value',
    'threshold_unit',
    'threshold_direction',
    'threshold_source_url',
  ]
  for (const row of thresholdRows) {
    const canonical = matchCanonical(row.dependency_name, canonicalNames)
    if (!canonical) {
      unmatched.push(row)
      resolved.push({ ...row, dependency_name: '' })
      continue
    }
    const isComplete = required.every((column) => (row[column] ?? '').trim())
    if (isComplete) {
      completeByName.set(canonical, (completeByName.get(canonical) ?? 0) + 1)
    }
    resolved.push({ ...row, dependency_name: canonical })
  }

  const incomplete: IncompleteThreshold[] = []
  for (const dependency of dependencyRows) {
    if (dependency.threshold_kind !== 'quantitative_with_threshold') {
      continue
    }
    const name = dependency.name ?? ''
    if (!completeByName.get(name)) {
      incomplete.push({ name, missing: ['a complete bar (value, unit, direction, source_url)'] })
    }
  }

  return { resolved, unmatched, incomplete }
}

// Validates dependency_edges.csv: both endpoints must resolve to canonical dependencies.
// Returns the resolved rows (canonical spellings) and the rows with an unresolvable endpoint.
export type ResolvedEdge = {
  from_dependency: string
  to_dependency: string
  relation: string
  note: string
}
export const validateDependencyEdges = (
  rows: CsvRow[],
  canonicalNames: readonly string[],
): { resolved: ResolvedEdge[]; unmatched: CsvRow[] } => {
  const resolved: ResolvedEdge[] = []
  const unmatched: CsvRow[] = []
  for (const row of rows) {
    const from = matchCanonical(row.from_dependency, canonicalNames)
    const to = matchCanonical(row.to_dependency, canonicalNames)
    if (!from || !to) {
      unmatched.push(row)
      continue
    }
    resolved.push({
      from_dependency: from,
      to_dependency: to,
      relation: row.relation ?? '',
      note: row.note ?? '',
    })
  }
  return { resolved, unmatched }
}
