import { describe, expect, it } from 'vitest'
import {
  type CompanyDependencyRow,
  dedupeCompanyDependencies,
  matchCanonical,
} from './dependency-resolution.js'

const canonical = ['Lithium-ion battery cost', 'Solar module cost', 'Carbon price']

const row = (overrides: Partial<CompanyDependencyRow> = {}): CompanyDependencyRow => ({
  company_uuid: 'company-1',
  dependency_name: 'Lithium-ion battery cost',
  criticality: 'contributing',
  detail: 'needed cheap cells',
  source_url: 'https://example.com/a',
  ...overrides,
})

describe('matchCanonical', () => {
  it('matches exactly', () => {
    expect(matchCanonical('Solar module cost', canonical)).toBe('Solar module cost')
  })

  it('tolerates case and whitespace drift, returning the canonical spelling', () => {
    expect(matchCanonical('  solar   MODULE cost ', canonical)).toBe('Solar module cost')
  })

  it('returns null rather than guessing when nothing fits', () => {
    expect(matchCanonical('Wind turbine blade length', canonical)).toBeNull()
  })

  it('returns null for null/empty proposals', () => {
    expect(matchCanonical(null, canonical)).toBeNull()
    expect(matchCanonical('   ', canonical)).toBeNull()
  })
})

describe('dedupeCompanyDependencies', () => {
  it('merges two free-text rows landing on the same canonical dependency', () => {
    const merged = dedupeCompanyDependencies([
      row({ detail: 'needed cheap cells' }),
      row({ detail: 'pack price too high' }),
    ])
    expect(merged).toHaveLength(1)
    expect(merged[0]?.detail).toBe('needed cheap cells; pack price too high')
  })

  it('lets was_blocking win over contributing', () => {
    const merged = dedupeCompanyDependencies([
      row({ criticality: 'contributing' }),
      row({ criticality: 'was_blocking', detail: 'blocked launch' }),
    ])
    expect(merged[0]?.criticality).toBe('was_blocking')
  })

  it('keeps rows for different companies separate', () => {
    const merged = dedupeCompanyDependencies([
      row({ company_uuid: 'company-1' }),
      row({ company_uuid: 'company-2' }),
    ])
    expect(merged).toHaveLength(2)
  })

  it('keeps every unresolved row instead of collapsing them together', () => {
    const merged = dedupeCompanyDependencies([
      row({ dependency_name: '', detail: 'something odd' }),
      row({ dependency_name: '', detail: 'something else odd' }),
    ])
    expect(merged).toHaveLength(2)
    expect(merged.every((entry) => entry.dependency_name === '')).toBe(true)
  })

  it('does not drop duplicate details', () => {
    const merged = dedupeCompanyDependencies([row({ detail: 'same' }), row({ detail: 'same' })])
    expect(merged[0]?.detail).toBe('same')
  })
})
