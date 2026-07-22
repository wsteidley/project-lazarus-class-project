import type { CsvRow } from './csv.js'
import { normalizeName, normalizeUrlValue } from './entity-resolution.js'
import type { EnrichmentIndex, EnrichmentRow } from './sources.js'

// Sources tried in trust order: crunchbase carries strong website/permalink keys for
// most rows; startup-failures is name-only and lower confidence (see data/sources/README.md).
export const SOURCE_NAMES = ['crunchbase', 'startup-failures'] as const

export type EnrichTier = 'website' | 'crunchbase' | 'name'

export type Match = { row: EnrichmentRow; source: string; tier: EnrichTier }

// Tries website -> crunchbase permalink -> name, in that order, within one source's
// index — the same strong-first precedence entity resolution uses.
export const lookupInSource = (
  index: EnrichmentIndex,
  source: string,
  websiteKey: string,
  crunchbaseKey: string,
  nameKey: string,
): Match | null => {
  const byWebsite = index.byWebsite.get(websiteKey)?.[0]
  if (byWebsite) {
    return { row: byWebsite, source, tier: 'website' }
  }
  const byCrunchbase = index.byCrunchbase.get(crunchbaseKey)?.[0]
  if (byCrunchbase) {
    return { row: byCrunchbase, source, tier: 'crunchbase' }
  }
  const byName = index.byName.get(nameKey)?.[0]
  if (byName) {
    return { row: byName, source, tier: 'name' }
  }
  return null
}

// Tries every source in trust order and returns the first hit — sources aren't
// blended, so a company's enrichment always traces back to one row and one tier.
export const findMatch = (
  indexBySource: Map<string, EnrichmentIndex>,
  websiteKey: string,
  crunchbaseKey: string,
  nameKey: string,
): Match | null => {
  for (const source of SOURCE_NAMES) {
    const index = indexBySource.get(source)
    if (!index) {
      continue
    }
    const match = lookupInSource(index, source, websiteKey, crunchbaseKey, nameKey)
    if (match) {
      return match
    }
  }
  return null
}

export const isEmpty = (value: string | undefined): boolean =>
  value === undefined || value.trim() === ''

export const provenance = (match: Match): string => `enrichment:${match.source}:${match.tier}`

// Fills only the fields the company is missing, and records provenance for the two
// fields Phase 2b's precedence rule needs to distinguish (living_status/exit_type are
// volatile; year_founded/year_defunct are treated as immutable once strong-joined).
export const applyEnrichment = (company: CsvRow, match: Match): CsvRow => {
  const updated = { ...company }
  if (isEmpty(updated.living_status) && !isEmpty(match.row.living_status)) {
    updated.living_status = match.row.living_status
    updated.living_status_source = provenance(match)
  }
  if (isEmpty(updated.exit_type) && !isEmpty(match.row.exit_type)) {
    updated.exit_type = match.row.exit_type
  }
  if (isEmpty(updated.year_founded) && !isEmpty(match.row.year_founded)) {
    updated.year_founded = match.row.year_founded
    updated.year_founded_source = provenance(match)
  }
  if (isEmpty(updated.year_defunct) && !isEmpty(match.row.year_defunct)) {
    updated.year_defunct = match.row.year_defunct
  }
  return updated
}

// One synthetic funding_rounds row standing in for an aggregate total from a static
// source. round_year is left blank so it never reads as a "recent raise" to
// derive-outcome's recency logic — it's an aggregate, not a dated event.
export const syntheticFundingRow = (
  companyUuid: string,
  match: Match,
): Record<string, unknown> | null => {
  const amount = match.row.funding_total_usd
  if (isEmpty(amount)) {
    return null
  }
  return {
    company_uuid: companyUuid,
    round_name: '',
    amount,
    currency: 'USD',
    round_date: '',
    round_year: '',
    source_url: provenance(match),
  }
}

// startup-failures rows carry a failure reason but no strong join — write it as a
// challenges seed (low confidence, marked uncorroborated) rather than a bare fact on
// the company. Crunchbase rows have no failure_reason, so this only ever fires for
// the startup-failures source.
export const seedChallenges = (companyUuid: string, match: Match): Record<string, unknown>[] => {
  if (match.source !== 'startup-failures' || isEmpty(match.row.challenge_categories)) {
    return []
  }
  const categories = match.row.challenge_categories
    .split(';')
    .map((category) => category.trim())
    .filter(Boolean)
  return categories.map((category) => ({
    company_uuid: companyUuid,
    category,
    outcome: 'fatal',
    detail: match.row.failure_reason,
    confidence: 'low',
    confidence_score: '',
    contested: 0,
    contested_note: `enrichment seed (${provenance(match)}) — name-only match, not yet corroborated`,
    // Provenance tag doubles as the discriminator that keeps re-runs idempotent —
    // enrich drops its own prior seed rows before re-appending. It is not an evidence
    // URL; the "seed, not evidence" status lives in contested_note.
    source_url: provenance(match),
  }))
}

// A challenges row this step wrote as a name-only seed, identified by its provenance
// source_url. Used to drop stale seeds before re-appending on a re-run.
export const isEnrichmentChallengeRow = (row: { source_url?: string }): boolean =>
  (row.source_url ?? '').startsWith('enrichment:')

// The three lookup keys for a company, derived the same way entity resolution derives
// them, so an enrichment lookup lines up with the merge keys already on company_urls.
export const enrichmentKeysFor = (
  companyName: string,
  urls: CsvRow[],
): { websiteKey: string; crunchbaseKey: string; nameKey: string } => {
  const websiteUrl = urls.find((url) => url.url_type === 'website')
  const crunchbaseUrl = urls.find((url) => url.url_type === 'crunchbase')
  return {
    websiteKey: websiteUrl?.normalized_value || normalizeUrlValue('website', websiteUrl?.url ?? ''),
    crunchbaseKey:
      crunchbaseUrl?.normalized_value || normalizeUrlValue('crunchbase', crunchbaseUrl?.url ?? ''),
    nameKey: normalizeName(companyName),
  }
}
