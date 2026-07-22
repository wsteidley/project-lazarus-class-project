import { describe, expect, it } from 'vitest'
import type { CsvRow } from './csv.js'
import {
  applyOutcomePrecedence,
  isStrongJoinSource,
  type SearchOutcome,
} from './outcome-precedence.js'

const outcome = (overrides: Partial<NonNullable<SearchOutcome>> = {}): SearchOutcome => ({
  living_status: 'Defunct',
  exit_type: null,
  exit_amount: null,
  exit_date: null,
  exit_notes: null,
  outcome_summary: null,
  source_url: null,
  snippet: null,
  confidence: 'medium',
  contested: false,
  contested_note: null,
  ...overrides,
})

describe('isStrongJoinSource', () => {
  it('is true for website/crunchbase tiers', () => {
    expect(isStrongJoinSource('enrichment:crunchbase:website')).toBe(true)
    expect(isStrongJoinSource('enrichment:crunchbase:crunchbase')).toBe(true)
  })

  it('is false for a name-only tier, search, or blank', () => {
    expect(isStrongJoinSource('enrichment:startup-failures:name')).toBe(false)
    expect(isStrongJoinSource('search')).toBe(false)
    expect(isStrongJoinSource(undefined)).toBe(false)
    expect(isStrongJoinSource('')).toBe(false)
  })
})

describe('applyOutcomePrecedence', () => {
  it('lets fresh search evidence override a static living_status', () => {
    const company: CsvRow = {
      living_status: 'Operating',
      living_status_source: 'enrichment:crunchbase:website',
    }
    const updated = applyOutcomePrecedence(company, outcome({ living_status: 'Defunct' }), null)
    expect(updated.living_status).toBe('Defunct')
    expect(updated.living_status_source).toBe('search')
  })

  it('leaves living_status untouched when search produced nothing', () => {
    const company: CsvRow = { living_status: 'Operating', living_status_source: 'search' }
    const updated = applyOutcomePrecedence(company, null, null)
    expect(updated.living_status).toBe('Operating')
    expect(updated.living_status_source).toBe('search')
  })

  it('fills exit fields only when search actually supplies them', () => {
    const company: CsvRow = { living_status: 'Operating', exit_type: '' }
    const updated = applyOutcomePrecedence(
      company,
      outcome({ living_status: 'Acquired', exit_type: 'acquisition', exit_amount: 5_000_000 }),
      null,
    )
    expect(updated.exit_type).toBe('acquisition')
    expect(updated.exit_amount).toBe('5000000')
  })

  it('never overwrites year_founded (search does not carry it)', () => {
    const company: CsvRow = {
      year_founded: '2010',
      year_founded_source: 'enrichment:crunchbase:website',
    }
    const updated = applyOutcomePrecedence(company, outcome(), null)
    expect(updated.year_founded).toBe('2010')
  })

  it('fills a blank year_defunct from the Wayback death signal', () => {
    const company: CsvRow = { year_defunct: '' }
    const updated = applyOutcomePrecedence(company, null, 2019)
    expect(updated.year_defunct).toBe('2019')
  })

  it('never overrides an existing year_defunct with the death signal', () => {
    const company: CsvRow = { year_defunct: '2015' }
    const updated = applyOutcomePrecedence(company, null, 2019)
    expect(updated.year_defunct).toBe('2015')
  })
})
