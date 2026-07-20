import { describe, expect, it } from 'vitest'
import { companyExtractionSchema, fundingRoundsSchema } from './schemas.js'

const validCompany = {
  company_name: 'Acme Solar',
  founders: null,
  is_climate: true,
  sector: 'Energy',
  subsector: null,
  location: 'North America',
  country: null,
  living_status: 'Defunct',
  has_pivoted: false,
  year_founded: 2010,
  year_defunct: 2018,
  idea_summary: null,
  original_trl: 5,
  reason_for_demise: null,
  failure_reasons: [{ category: 'Ran Out of Capital', detail: 'burned through Series B' }],
}

describe('companyExtractionSchema', () => {
  it('accepts an in-vocab company', () => {
    expect(companyExtractionSchema.safeParse(validCompany).success).toBe(true)
  })

  it('rejects an out-of-vocab sector', () => {
    const result = companyExtractionSchema.safeParse({ ...validCompany, sector: 'Fusion Widgets' })
    expect(result.success).toBe(false)
  })

  it('rejects an out-of-vocab failure category', () => {
    const result = companyExtractionSchema.safeParse({
      ...validCompany,
      failure_reasons: [{ category: 'Bad Vibes', detail: null }],
    })
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
