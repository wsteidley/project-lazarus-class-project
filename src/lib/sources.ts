import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { sourcesDir } from '../config.js'
import { readCsv } from './csv.js'
import { normalizeDomain, normalizeName } from './entity-resolution.js'

// One enrichment record, the shared shape every source normalizes to. A record can
// carry several join keys at once (Crunchbase has both a website and a slug); the
// loader indexes each non-empty key, so a lookup can try the strong keys first and
// fall back to name — the same precedence entity resolution uses.
export type EnrichmentRow = {
  source: string
  company_name: string
  // Normalized join keys; any subset may be present.
  key_website: string
  key_crunchbase: string
  key_name: string
  // Enrichment payload; blank where a source doesn't supply it.
  living_status: string
  exit_type: string
  funding_total_usd: string
  funding_rounds: string
  year_founded: string
  year_defunct: string
  idea_summary: string
  failure_reason: string
  challenge_categories: string
  sector_raw: string
  country: string
  // Provenance: the source's own identifier for the row (permalink, or the name).
  source_ref: string
}

export const ENRICHMENT_COLUMNS: (keyof EnrichmentRow)[] = [
  'source',
  'company_name',
  'key_website',
  'key_crunchbase',
  'key_name',
  'living_status',
  'exit_type',
  'funding_total_usd',
  'funding_rounds',
  'year_founded',
  'year_defunct',
  'idea_summary',
  'failure_reason',
  'challenge_categories',
  'sector_raw',
  'country',
  'source_ref',
]

// An empty record, so a normalizer sets only the fields its source actually has.
export const emptyEnrichmentRow = (source: string): EnrichmentRow => ({
  source,
  company_name: '',
  key_website: '',
  key_crunchbase: '',
  key_name: '',
  living_status: '',
  exit_type: '',
  funding_total_usd: '',
  funding_rounds: '',
  year_founded: '',
  year_defunct: '',
  idea_summary: '',
  failure_reason: '',
  challenge_categories: '',
  sector_raw: '',
  country: '',
  source_ref: '',
})

// --- join-key builders (reuse the resolver's normalization so keys line up) ---

export const websiteKey = (homepageUrl: string | null | undefined): string =>
  normalizeDomain(homepageUrl)

// Crunchbase permalinks look like "/organization/helios-energy"; reduce to the slug.
export const crunchbaseKey = (permalink: string | null | undefined): string =>
  (permalink ?? '')
    .trim()
    .replace(/^\/?organization\//i, '')
    .replace(/^https?:\/\/(www\.)?crunchbase\.com\/organization\//i, '')
    .replace(/\/+$/, '')
    .toLowerCase()

export const nameKey = (name: string | null | undefined): string => normalizeName(name)

// --- value parsers ---

// "$12M" -> 12000000, "$1.5B" -> 1500000000, "$500K" -> 500000. Returns '' for
// undisclosed/unknown/empty so a missing raise stays distinct from a zero raise.
export const parseRaisedAmount = (raw: string | null | undefined): string => {
  const text = (raw ?? '').trim()
  if (!text) {
    return ''
  }
  const match = text.replace(/,/g, '').match(/\$?\s*([\d.]+)\s*([kmb])?/i)
  if (!match?.[1]) {
    return ''
  }
  const value = Number(match[1])
  if (!Number.isFinite(value)) {
    return ''
  }
  const scale = { k: 1e3, m: 1e6, b: 1e9 }[(match[2] ?? '').toLowerCase()] ?? 1
  return String(Math.round(value * scale))
}

// "2015-2019" -> founded 2015, defunct 2019. Handles a single year, an open end
// ("2015-Present"), and the index file's "3 (2010-2013)" shape defensively.
export const parseYearsRange = (
  raw: string | null | undefined,
): { year_founded: string; year_defunct: string } => {
  const text = (raw ?? '').trim()
  const years = text.match(/\d{4}/g) ?? []
  return {
    year_founded: years[0] ?? '',
    year_defunct: years[1] ?? '',
  }
}

// Crunchbase status -> our living_status vocab, plus any exit it implies. ipo/acquired
// companies are still "operating" as going concerns; the exit is recorded separately.
export const crunchbaseStatus = (
  status: string | null | undefined,
): { living_status: string; exit_type: string } => {
  switch ((status ?? '').trim().toLowerCase()) {
    case 'operating':
      return { living_status: 'Operating', exit_type: '' }
    case 'acquired':
      return { living_status: 'Acquired', exit_type: 'acquisition' }
    case 'ipo':
      return { living_status: 'Operating', exit_type: 'ipo' }
    case 'closed':
      return { living_status: 'Defunct', exit_type: 'shutdown' }
    default:
      return { living_status: '', exit_type: '' }
  }
}

// The startup-failures 0/1 reason flags -> our CHALLENGE vocabulary. Only flags with a
// defensible mapping are included; the rest are left to the free-text failure_reason so
// nothing is force-fit onto a category it doesn't mean.
const FLAG_TO_CHALLENGE: Record<string, string> = {
  Giants: 'Outcompeted',
  Competition: 'Outcompeted',
  'No Budget': 'Ran Out of Capital',
  'Poor Market Fit': 'Poor Product-Market Fit',
  'Niche Limits': 'No Market Need',
  'Monetization Failure': 'Flawed Business Model',
  'High Operational Costs': 'Pricing / Unit Economics',
  'Execution Flaws': 'Poor Product / Execution',
  'Trend Shifts': 'Bad Timing (Ahead of Market)',
  'Regulatory Pressure': 'Legal / Regulatory',
}

// Reads the set flags on a raw failure row and returns the mapped CHALLENGE categories,
// de-duplicated (several flags can map to one category, e.g. Giants + Competition).
export const failureFlagsToChallenges = (row: Record<string, string>): string[] => {
  const categories = new Set<string>()
  for (const [flag, category] of Object.entries(FLAG_TO_CHALLENGE)) {
    if ((row[flag] ?? '').trim() === '1') {
      categories.add(category)
    }
  }
  return [...categories]
}

// --- loader ---

export type EnrichmentIndex = {
  byWebsite: Map<string, EnrichmentRow[]>
  byCrunchbase: Map<string, EnrichmentRow[]>
  byName: Map<string, EnrichmentRow[]>
}

const addTo = (map: Map<string, EnrichmentRow[]>, key: string, row: EnrichmentRow): void => {
  if (key) {
    map.set(key, [...(map.get(key) ?? []), row])
  }
}

// Loads a normalized source and indexes it by each join key. A future enrich step
// looks up a company by website, then crunchbase slug, then name — strong keys first.
export const loadSource = async (name: string): Promise<EnrichmentIndex> => {
  const path = join(sourcesDir, name, 'companies.csv')
  const index: EnrichmentIndex = {
    byWebsite: new Map(),
    byCrunchbase: new Map(),
    byName: new Map(),
  }
  if (!existsSync(path)) {
    return index
  }
  const rows = (await readCsv(path)) as unknown as EnrichmentRow[]
  for (const row of rows) {
    addTo(index.byWebsite, row.key_website, row)
    addTo(index.byCrunchbase, row.key_crunchbase, row)
    addTo(index.byName, row.key_name, row)
  }
  return index
}
