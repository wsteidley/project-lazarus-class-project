import { describe, expect, it } from 'vitest'
import { companyExtractionSchema, fundingRoundsSchema } from './schemas.js'

const validCompany = {
  company_name: 'Acme Solar',
  founders: null,
  is_climate: true,
  sectors: [{ name: 'Energy', is_primary: true }],
  idea_space_name: 'Long-duration grid storage',
  location: 'North America',
  country: null,
  living_status: 'Defunct',
  has_pivoted: false,
  year_founded: 2010,
  year_defunct: 2018,
  idea_summary: null,
  original_trl: 5,
  exit_type: 'shutdown',
  exit_amount: null,
  exit_date: null,
  exit_notes: null,
  outcome_summary: 'Wound down after failing to raise a Series C',
  outcome_source_url: null,
  challenges: [
    { category: 'Ran Out of Capital', outcome: 'fatal', detail: 'burned through Series B' },
  ],
}

describe('companyExtractionSchema', () => {
  it('accepts an in-vocab company', () => {
    expect(companyExtractionSchema.safeParse(validCompany).success).toBe(true)
  })

  it('rejects an out-of-vocab sector', () => {
    const result = companyExtractionSchema.safeParse({
      ...validCompany,
      sectors: [{ name: 'Fusion Widgets', is_primary: true }],
    })
    expect(result.success).toBe(false)
  })

  it('rejects an out-of-vocab challenge category', () => {
    const result = companyExtractionSchema.safeParse({
      ...validCompany,
      challenges: [{ category: 'Bad Vibes', outcome: 'fatal', detail: null }],
    })
    expect(result.success).toBe(false)
  })

  it('rejects an out-of-vocab challenge outcome', () => {
    const result = companyExtractionSchema.safeParse({
      ...validCompany,
      challenges: [{ category: 'Team', outcome: 'vanished', detail: null }],
    })
    expect(result.success).toBe(false)
  })

  it('rejects an out-of-vocab exit_type', () => {
    const result = companyExtractionSchema.safeParse({ ...validCompany, exit_type: 'merger' })
    expect(result.success).toBe(false)
  })
})

describe('fundingRoundsSchema', () => {
  it('rejects an out-of-vocab round_name', () => {
    const result = fundingRoundsSchema.safeParse({
      funding_rounds: [{ round_name: 'Series Z', amount: 1, currency: 'USD', date: '2020-01' }],
    })
    expect(result.success).toBe(false)
  })
})
