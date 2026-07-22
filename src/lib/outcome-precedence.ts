import type { OutcomeAssessment } from '../schemas.js'
import type { CsvRow } from './csv.js'

// A field's provenance value that came from a strong join (website/crunchbase-slug
// key), as opposed to a name-only seed. Used by the confidence machinery to judge
// whether an independent source (search) agrees with a trustworthy static value.
export const isStrongJoinSource = (source: string | undefined): boolean =>
  /^enrichment:[^:]+:(website|crunchbase)$/.test(source ?? '')

export type SearchOutcome = OutcomeAssessment | null

const isEmpty = (value: string | undefined): boolean => value === undefined || value.trim() === ''

// Applies the v5 sourcing spec's volatility-split precedence rule for Phase 2b:
//  - Volatile fields (living_status, exit state) — fresh search evidence always wins
//    over Phase 2a's static snapshot, which can go stale.
//  - Immutable facts (year_founded/year_defunct) — never touched by search here.
//    year_founded was already settled by Phase 2a (or extraction); year_defunct only
//    ever gets filled from the Wayback death signal when it's still blank, which is a
//    gap-fill, not a correction.
export const applyOutcomePrecedence = (
  company: CsvRow,
  search: SearchOutcome,
  waybackLastCaptureYear: number | null,
): CsvRow => {
  const updated = { ...company }

  if (search) {
    updated.living_status = search.living_status
    updated.living_status_source = 'search'
    if (search.exit_type) {
      updated.exit_type = search.exit_type
    }
    if (search.exit_amount !== null) {
      updated.exit_amount = String(search.exit_amount)
    }
    if (search.exit_date) {
      updated.exit_date = search.exit_date
    }
    if (search.exit_notes) {
      updated.exit_notes = search.exit_notes
    }
    if (search.outcome_summary) {
      updated.outcome_summary = search.outcome_summary
    }
    if (search.source_url) {
      updated.outcome_source_url = search.source_url
    }
  }

  if (isEmpty(updated.year_defunct) && waybackLastCaptureYear !== null) {
    updated.year_defunct = String(waybackLastCaptureYear)
  }

  return updated
}
