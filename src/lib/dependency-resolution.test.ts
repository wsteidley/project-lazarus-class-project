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
  dependency_name_raw: 'battery pack price',
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

  it('keeps unresolved rows naming different blockers separate', () => {
    const merged = dedupeCompanyDependencies([
      row({ dependency_name: '', dependency_name_raw: 'public trust in AVs' }),
      row({ dependency_name: '', dependency_name_raw: 'municipal permitting appetite' }),
    ])
    expect(merged).toHaveLength(2)
    expect(merged.every((entry) => entry.dependency_name === '')).toBe(true)
  })

  // Unresolved rows used to pass straight through un-merged, because they had no canonical
  // identity. Now that build-db retains them instead of dropping them, two extractions of
  // the same unmatched blocker are the same blocker — and would otherwise duplicate.
  it('merges unresolved rows naming the same blocker, tolerating case drift', () => {
    const merged = dedupeCompanyDependencies([
      row({ dependency_name: '', dependency_name_raw: 'public trust in AVs', detail: 'riders' }),
      row({
        dependency_name: '',
        dependency_name_raw: 'Public Trust in AVs',
        detail: 'regulators',
      }),
    ])
    expect(merged).toHaveLength(1)
    expect(merged[0]?.detail).toBe('riders; regulators')
  })

  it('does not drop duplicate details', () => {
    const merged = dedupeCompanyDependencies([row({ detail: 'same' }), row({ detail: 'same' })])
    expect(merged[0]?.detail).toBe('same')
  })
})
