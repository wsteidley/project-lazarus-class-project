import { describe, expect, it } from 'vitest'
import type { CsvRow } from './csv.js'
import {
  applyEnrichment,
  enrichmentKeysFor,
  findMatch,
  isEnrichmentChallengeRow,
  type Match,
  provenance,
  seedChallenges,
  syntheticFundingRow,
} from './enrich.js'
import { type EnrichmentIndex, type EnrichmentRow, emptyEnrichmentRow } from './sources.js'

const enrichmentRow = (overrides: Partial<EnrichmentRow>): EnrichmentRow => ({
  ...emptyEnrichmentRow(overrides.source ?? 'crunchbase'),
  ...overrides,
})

const indexWith = (rows: {
  byWebsite?: Record<string, EnrichmentRow>
  byCrunchbase?: Record<string, EnrichmentRow>
  byName?: Record<string, EnrichmentRow>
}): EnrichmentIndex => ({
  byWebsite: new Map(Object.entries(rows.byWebsite ?? {}).map(([k, v]) => [k, [v]])),
  byCrunchbase: new Map(Object.entries(rows.byCrunchbase ?? {}).map(([k, v]) => [k, [v]])),
  byName: new Map(Object.entries(rows.byName ?? {}).map(([k, v]) => [k, [v]])),
})

describe('findMatch', () => {
  it('prefers crunchbase over startup-failures at the same tier', () => {
    const cbRow = enrichmentRow({ source: 'crunchbase', living_status: 'Operating' })
    const sfRow = enrichmentRow({ source: 'startup-failures', living_status: 'Defunct' })
    const indexBySource = new Map([
      ['crunchbase', indexWith({ byName: { acme: cbRow } })],
      ['startup-failures', indexWith({ byName: { acme: sfRow } })],
    ])
    const match = findMatch(indexBySource, '', '', 'acme')
    expect(match?.source).toBe('crunchbase')
    expect(match?.tier).toBe('name')
  })

  it('prefers website over name within a source', () => {
    const byWebsiteRow = enrichmentRow({ living_status: 'Operating' })
    const byNameRow = enrichmentRow({ living_status: 'Defunct' })
    const indexBySource = new Map([
      [
        'crunchbase',
        indexWith({ byWebsite: { 'acme.com': byWebsiteRow }, byName: { acme: byNameRow } }),
      ],
    ])
    const match = findMatch(indexBySource, 'acme.com', '', 'acme')
    expect(match?.tier).toBe('website')
    expect(match?.row.living_status).toBe('Operating')
  })

  it('returns null when nothing matches any source', () => {
    const indexBySource = new Map([['crunchbase', indexWith({})]])
    expect(findMatch(indexBySource, 'x.com', 'x', 'x')).toBeNull()
  })
})

describe('applyEnrichment', () => {
  const match: Match = {
    row: enrichmentRow({
      source: 'crunchbase',
      living_status: 'Defunct',
      exit_type: 'shutdown',
      year_founded: '2012',
      year_defunct: '2018',
    }),
    source: 'crunchbase',
    tier: 'website',
  }

  it('fills empty fields and records provenance', () => {
    const company: CsvRow = { company_name: 'Acme', living_status: '', year_founded: '' }
    const updated = applyEnrichment(company, match)
    expect(updated.living_status).toBe('Defunct')
    expect(updated.living_status_source).toBe('enrichment:crunchbase:website')
    expect(updated.year_founded).toBe('2012')
    expect(updated.year_founded_source).toBe('enrichment:crunchbase:website')
    expect(updated.exit_type).toBe('shutdown')
    expect(updated.year_defunct).toBe('2018')
  })

  it('never overwrites a field the company already has', () => {
    const company: CsvRow = {
      company_name: 'Acme',
      living_status: 'Operating',
      year_founded: '2005',
    }
    const updated = applyEnrichment(company, match)
    expect(updated.living_status).toBe('Operating')
    expect(updated.living_status_source).toBeUndefined()
    expect(updated.year_founded).toBe('2005')
  })
})

describe('provenance', () => {
  it('renders source and tier', () => {
    expect(provenance({ row: enrichmentRow({}), source: 'startup-failures', tier: 'name' })).toBe(
      'enrichment:startup-failures:name',
    )
  })
})

describe('syntheticFundingRow', () => {
  it('produces one aggregate row with no round_year', () => {
    const match: Match = {
      row: enrichmentRow({ funding_total_usd: '5000000' }),
      source: 'crunchbase',
      tier: 'website',
    }
    const row = syntheticFundingRow('u1', match)
    expect(row).toMatchObject({ company_uuid: 'u1', amount: '5000000', round_year: '' })
  })

  it('is null when the source has no funding figure', () => {
    const match: Match = { row: enrichmentRow({}), source: 'crunchbase', tier: 'name' }
    expect(syntheticFundingRow('u1', match)).toBeNull()
  })
})

describe('seedChallenges', () => {
  it('maps startup-failures challenge_categories to low-confidence seed rows', () => {
    const match: Match = {
      row: enrichmentRow({
        source: 'startup-failures',
        challenge_categories: 'Ran Out of Capital;Outcompeted',
        failure_reason: 'ran out of money',
      }),
      source: 'startup-failures',
      tier: 'name',
    }
    const rows = seedChallenges('u1', match)
    expect(rows).toHaveLength(2)
    expect(rows[0]).toMatchObject({
      company_uuid: 'u1',
      category: 'Ran Out of Capital',
      confidence: 'low',
      detail: 'ran out of money',
      // Provenance tag on source_url is the idempotency discriminator for re-runs.
      source_url: 'enrichment:startup-failures:name',
    })
    expect(rows.every((row) => isEnrichmentChallengeRow(row))).toBe(true)
    expect(String(rows[0]?.contested_note)).toMatch(/not yet corroborated/)
  })

  it('never fires for crunchbase, which has no failure_reason field', () => {
    const match: Match = {
      row: enrichmentRow({ source: 'crunchbase', challenge_categories: 'Outcompeted' }),
      source: 'crunchbase',
      tier: 'website',
    }
    expect(seedChallenges('u1', match)).toEqual([])
  })

  it('is empty when the source has no mapped categories', () => {
    const match: Match = {
      row: enrichmentRow({ source: 'startup-failures' }),
      source: 'startup-failures',
      tier: 'name',
    }
    expect(seedChallenges('u1', match)).toEqual([])
  })
})

describe('enrichmentKeysFor', () => {
  it('reads website/crunchbase from normalized_value, falling back to raw url', () => {
    const urls: CsvRow[] = [
      { url_type: 'website', url: 'https://Acme.com', normalized_value: 'acme.com' },
      {
        url_type: 'crunchbase',
        url: 'https://www.crunchbase.com/organization/acme',
        normalized_value: '',
      },
    ]
    const keys = enrichmentKeysFor('Acme Inc', urls)
    expect(keys.websiteKey).toBe('acme.com')
    expect(keys.crunchbaseKey).toBe('acme')
    expect(keys.nameKey).toBe('acme')
  })
})
