import type { CsvRow } from './csv.js'
import { normalizeUrl } from './raw-documents.js'

// Which identity key merged a group. Recorded because it is exactly the evidence
// needed to judge whether probabilistic linkage (Splink) is worth adding: merges on
// `website` or an external ID are trustworthy, merges on `name_year` are the guess.
export const KEY_TIER = ['website', 'crunchbase', 'wikipedia', 'name_year'] as const
export type KeyTier = (typeof KEY_TIER)[number]

// The url_types that can identify a company, in precedence order. Anything else
// (linkedin, archive, article) is stored but never used as a merge key — a shared
// press article says nothing about two companies being the same.
const IDENTITY_TYPES: readonly KeyTier[] = ['website', 'crunchbase', 'wikipedia']

export type CanonicalKey = { key: string; tier: KeyTier }

export type ResolutionResult = {
  companies: CsvRow[]
  urls: CsvRow[]
  /** Every original uuid mapped to the uuid that survived, including identity rows. */
  uuidMap: Map<string, string>
  /** How many groups merged, by the tier that matched them. */
  mergesByTier: Record<KeyTier, number>
}

// Reuses the cache's URL normalization, then reduces to a bare hostname: no scheme,
// no `www.`, no path. One shared notion of "same URL" across the codebase.
export const normalizeDomain = (value: string | null | undefined): string => {
  const raw = value?.trim()
  if (!raw) {
    return ''
  }
  const withScheme = /^https?:\/\//i.test(raw) ? raw : `https://${raw}`
  let host: string
  try {
    host = new URL(normalizeUrl(withScheme)).hostname
  } catch {
    return ''
  }
  return host.replace(/^www\./, '').toLowerCase()
}

// The comparable form of a URL, by type. Websites reduce to a bare domain (the thing
// that actually identifies a startup); everything else is normalized as a URL. Stored
// on the row so the merge key is visible in the DB rather than implicit in code.
export const normalizeUrlValue = (urlType: string, url: string): string => {
  const raw = url?.trim() ?? ''
  if (!raw) {
    return ''
  }
  if (urlType === 'website') {
    return normalizeDomain(raw)
  }
  if (urlType === 'crunchbase') {
    // Accept either a bare permalink slug or a full Crunchbase URL.
    const slug = raw.replace(/^https?:\/\/(www\.)?crunchbase\.com\/organization\//i, '')
    return slug.replace(/\/+$/, '').toLowerCase()
  }
  return normalizeUrl(raw).toLowerCase()
}

// Groups URL rows by the company they belong to.
export const urlsByCompany = (urls: CsvRow[]): Map<string, CsvRow[]> => {
  const grouped = new Map<string, CsvRow[]>()
  for (const url of urls) {
    const uuid = url.company_uuid ?? ''
    grouped.set(uuid, [...(grouped.get(uuid) ?? []), url])
  }
  return grouped
}

// Collapses a company name for comparison: case, punctuation, and the corporate
// suffixes that vary between write-ups of the same company. Exported so external
// sources compute the same name-join key that resolution uses.
export const normalizeName = (name: string | null | undefined): string =>
  (name ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\b(inc|llc|ltd|corp|corporation|co|company|technologies|technology)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

// ID-first precedence over the company's URLs. Website domain is the pragmatic
// canonical key for startups; the external IDs cover notable ones; name+year is the
// last resort and the only fuzzy-ish tier, which is why its tier is reported apart.
export const canonicalKey = (company: CsvRow, urls: CsvRow[] = []): CanonicalKey => {
  for (const tier of IDENTITY_TYPES) {
    const match = urls
      .filter((url) => url.url_type === tier)
      .map((url) => url.normalized_value || normalizeUrlValue(tier, url.url ?? ''))
      .find(Boolean)
    if (match) {
      return { key: `${tier}:${match}`, tier }
    }
  }
  const name = normalizeName(company.company_name)
  const year = company.year_founded?.trim() ?? ''
  return { key: `name:${name}|${year}`, tier: 'name_year' }
}

const isEmpty = (value: string | undefined): boolean => value === undefined || value.trim() === ''

// Merges a group of duplicate rows into one canonical row.
//
// Field rules, kept explicit so they are reviewable rather than emergent:
//  - the earliest-created row supplies the surviving uuid and is the base
//  - any field empty on the base is filled from the next row that has it
//  - year_founded takes the earliest non-empty value (the launch article is often
//    later than the true founding)
//  - idea_summary takes the longest (the most informative write-up)
//  - merged_from lists every absorbed uuid, so a merge is always auditable
const mergeGroup = (group: CsvRow[]): CsvRow => {
  const ordered = [...group].sort((a, b) => (a.created_at ?? '').localeCompare(b.created_at ?? ''))
  const [base, ...rest] = ordered
  if (!base) {
    throw new Error('mergeGroup called with an empty group')
  }
  const merged: CsvRow = { ...base }

  for (const row of rest) {
    for (const [field, value] of Object.entries(row)) {
      if (isEmpty(merged[field]) && !isEmpty(value)) {
        merged[field] = value
      }
    }
    const currentYear = merged.year_founded?.trim()
    const candidateYear = row.year_founded?.trim()
    if (candidateYear && (!currentYear || Number(candidateYear) < Number(currentYear))) {
      merged.year_founded = candidateYear
    }
    if ((row.idea_summary ?? '').length > (merged.idea_summary ?? '').length) {
      merged.idea_summary = row.idea_summary ?? ''
    }
  }

  merged.canonical_uuid = base.uuid ?? ''
  merged.merged_from = rest
    .map((row) => row.uuid ?? '')
    .filter(Boolean)
    .join(';')
  return merged
}

// Re-points URL rows at the surviving company and drops duplicates. Two rows are the
// same URL when their type and *normalized* value agree, so `helios.com` absorbed from
// one article and `https://www.helios.com/` from another collapse into one row.
export const mergeCompanyUrls = (urls: CsvRow[], uuidMap: Map<string, string>): CsvRow[] => {
  const seen = new Set<string>()
  const merged: CsvRow[] = []

  for (const url of urls) {
    const urlType = url.url_type ?? ''
    const normalized = normalizeUrlValue(urlType, url.url ?? '')
    const companyUuid = uuidMap.get(url.company_uuid ?? '') ?? url.company_uuid ?? ''
    const key = `${companyUuid}::${urlType}::${normalized || url.url}`
    if (seen.has(key)) {
      continue
    }
    seen.add(key)
    merged.push({ ...url, company_uuid: companyUuid, normalized_value: normalized })
  }
  return merged
}

// Groups companies by canonical key and merges each group. Rows that match nothing
// pass through unchanged apart from canonical_uuid, so the output is always the
// complete company set — resolution never drops a company.
export const resolveCompanies = (rows: CsvRow[], urls: CsvRow[] = []): ResolutionResult => {
  const byCompany = urlsByCompany(urls)
  const groups = new Map<string, { tier: KeyTier; rows: CsvRow[] }>()

  for (const row of rows) {
    const { key, tier } = canonicalKey(row, byCompany.get(row.uuid ?? '') ?? [])
    const existing = groups.get(key)
    if (existing) {
      existing.rows.push(row)
    } else {
      groups.set(key, { tier, rows: [row] })
    }
  }

  const companies: CsvRow[] = []
  const uuidMap = new Map<string, string>()
  const mergesByTier: Record<KeyTier, number> = {
    website: 0,
    crunchbase: 0,
    wikipedia: 0,
    name_year: 0,
  }

  for (const { tier, rows: group } of groups.values()) {
    const merged = mergeGroup(group)
    companies.push(merged)
    for (const row of group) {
      if (row.uuid) {
        uuidMap.set(row.uuid, merged.uuid ?? '')
      }
    }
    if (group.length > 1) {
      mergesByTier[tier] += 1
    }
  }

  return { companies, urls: mergeCompanyUrls(urls, uuidMap), uuidMap, mergesByTier }
}

// Rewrites a child table's company_uuid onto the surviving canonical uuid. Rows whose
// uuid is unknown are left untouched so build-db reports them as unresolved FKs rather
// than silently dropping them here.
export const remapCompanyUuids = (rows: CsvRow[], uuidMap: Map<string, string>): CsvRow[] =>
  rows.map((row) => {
    const canonical = uuidMap.get(row.company_uuid ?? '')
    return canonical ? { ...row, company_uuid: canonical } : row
  })

export type NearMiss = { left: string; right: string; reason: string }

// Pairs of *distinct* canonical companies whose normalized names match, meaning
// ID-first deliberately did not merge them. This is the evidence for whether
// probabilistic linkage is worth adding: an empty report means ID-first was enough.
export const nearMissReport = (companies: CsvRow[], urls: CsvRow[] = []): NearMiss[] => {
  const byCompany = urlsByCompany(urls)
  const websiteOf = (company: CsvRow): string =>
    (byCompany.get(company.uuid ?? '') ?? [])
      .filter((url) => url.url_type === 'website')
      .map((url) => url.normalized_value || normalizeUrlValue('website', url.url ?? ''))
      .find(Boolean) ?? ''

  const byName = new Map<string, CsvRow[]>()
  for (const company of companies) {
    const name = normalizeName(company.company_name)
    if (!name) {
      continue
    }
    byName.set(name, [...(byName.get(name) ?? []), company])
  }

  const misses: NearMiss[] = []
  for (const [name, group] of byName) {
    if (group.length < 2) {
      continue
    }
    for (let i = 0; i < group.length - 1; i += 1) {
      for (let j = i + 1; j < group.length; j += 1) {
        const left = group[i]
        const right = group[j]
        if (!left || !right) {
          continue
        }
        const leftSite = websiteOf(left)
        const rightSite = websiteOf(right)
        const reason =
          leftSite && rightSite
            ? `same name "${name}", different websites (${leftSite} vs ${rightSite})`
            : `same name "${name}", no shared identity URL`
        misses.push({ left: left.uuid ?? '', right: right.uuid ?? '', reason })
      }
    }
  }
  return misses
}
