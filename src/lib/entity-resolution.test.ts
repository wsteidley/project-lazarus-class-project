import { describe, expect, it } from 'vitest'
import type { CsvRow } from './csv.js'
import {
  canonicalKey,
  mergeCompanyUrls,
  nearMissReport,
  normalizeDomain,
  normalizeUrlValue,
  remapCompanyUuids,
  resolveCompanies,
} from './entity-resolution.js'

const company = (overrides: Partial<CsvRow> = {}): CsvRow => ({
  uuid: 'u1',
  company_name: 'Helios Energy',
  year_founded: '2010',
  idea_summary: 'Solar',
  created_at: '2026-01-01T00:00:00Z',
  ...overrides,
})

const url = (companyUuid: string, urlType: string, value: string): CsvRow => ({
  company_uuid: companyUuid,
  url_type: urlType,
  url: value,
  normalized_value: '',
  source_url: 'https://example.com/a',
})

describe('normalizeDomain', () => {
  it('reduces a full URL to a bare hostname', () => {
    expect(normalizeDomain('https://WWW.Foo.com/path?x=1#top')).toBe('foo.com')
  })

  it('accepts a bare domain without a scheme', () => {
    expect(normalizeDomain('Foo.com')).toBe('foo.com')
  })

  it('collapses www and scheme variants together', () => {
    expect(normalizeDomain('http://foo.com')).toBe(normalizeDomain('https://www.foo.com/'))
  })

  it('returns empty for missing or unusable values', () => {
    expect(normalizeDomain(null)).toBe('')
    expect(normalizeDomain('   ')).toBe('')
  })
})

describe('normalizeUrlValue', () => {
  it('reduces a website to a bare domain', () => {
    expect(normalizeUrlValue('website', 'https://www.helios.com/about')).toBe('helios.com')
  })

  it('accepts a Crunchbase permalink either bare or as a full URL', () => {
    expect(normalizeUrlValue('crunchbase', 'https://www.crunchbase.com/organization/Helios/')).toBe(
      'helios',
    )
    expect(normalizeUrlValue('crunchbase', 'helios')).toBe('helios')
  })

  it('normalizes other types as plain URLs', () => {
    expect(normalizeUrlValue('wikipedia', 'https://en.wikipedia.org/wiki/Helios?x=1')).toBe(
      'https://en.wikipedia.org/wiki/helios',
    )
  })
})

describe('canonicalKey', () => {
  it('prefers website over every other URL type', () => {
    const key = canonicalKey(company(), [
      url('u1', 'crunchbase', 'helios'),
      url('u1', 'website', 'helios.com'),
    ])
    expect(key.tier).toBe('website')
  })

  it('falls back through crunchbase, then wikipedia, then name+year', () => {
    expect(canonicalKey(company(), [url('u1', 'crunchbase', 'helios')]).tier).toBe('crunchbase')
    expect(canonicalKey(company(), [url('u1', 'wikipedia', 'https://w/x')]).tier).toBe('wikipedia')
    expect(canonicalKey(company(), []).tier).toBe('name_year')
  })

  it('never keys on a non-identity URL type', () => {
    // A shared press article says nothing about two companies being the same.
    const key = canonicalKey(company(), [url('u1', 'article', 'https://news/x')])
    expect(key.tier).toBe('name_year')
  })

  it('ignores corporate suffixes and punctuation in the name tier', () => {
    expect(canonicalKey(company({ company_name: 'Helios Energy, Inc.' })).key).toBe(
      canonicalKey(company({ company_name: 'helios energy' })).key,
    )
  })

  it('separates same-named companies founded in different years', () => {
    expect(canonicalKey(company({ year_founded: '2010' })).key).not.toBe(
      canonicalKey(company({ year_founded: '2018' })).key,
    )
  })
})

describe('resolveCompanies', () => {
  it('merges rows sharing a website despite different name spellings', () => {
    const { companies, mergesByTier } = resolveCompanies(
      [
        company({ uuid: 'u1', company_name: 'Helios Energy' }),
        company({ uuid: 'u2', company_name: 'Helios Energy Inc' }),
      ],
      [url('u1', 'website', 'helios.com'), url('u2', 'website', 'https://www.helios.com/')],
    )
    expect(companies).toHaveLength(1)
    expect(mergesByTier.website).toBe(1)
  })

  it('keeps the earliest-created row as the surviving identity', () => {
    const { companies } = resolveCompanies(
      [
        company({ uuid: 'late', created_at: '2026-06-01T00:00:00Z' }),
        company({ uuid: 'early', created_at: '2026-01-01T00:00:00Z' }),
      ],
      [url('late', 'website', 'a.com'), url('early', 'website', 'a.com')],
    )
    expect(companies[0]?.uuid).toBe('early')
    expect(companies[0]?.merged_from).toBe('late')
  })

  it('fills empty fields, earliest year, and longest summary from the absorbed row', () => {
    const { companies } = resolveCompanies(
      [
        company({
          uuid: 'u1',
          country: '',
          year_founded: '2012',
          idea_summary: 'Solar',
          created_at: '2026-01-01',
        }),
        company({
          uuid: 'u2',
          country: 'USA',
          year_founded: '2009',
          idea_summary: 'Cylindrical thin-film solar for commercial rooftops',
          created_at: '2026-02-01',
        }),
      ],
      [url('u1', 'website', 'a.com'), url('u2', 'website', 'a.com')],
    )
    expect(companies[0]?.country).toBe('USA')
    expect(companies[0]?.year_founded).toBe('2009')
    expect(companies[0]?.idea_summary).toBe('Cylindrical thin-film solar for commercial rooftops')
  })

  it('gives the merged company the union of its duplicates URLs, de-duplicated', () => {
    const { urls } = resolveCompanies(
      [
        company({ uuid: 'u1', created_at: '2026-01-01' }),
        company({ uuid: 'u2', created_at: '2026-02-01' }),
      ],
      [
        url('u1', 'website', 'helios.com'),
        url('u2', 'website', 'https://www.helios.com/'),
        url('u2', 'crunchbase', 'helios'),
      ],
    )
    expect(urls).toHaveLength(2)
    expect(urls.every((row) => row.company_uuid === 'u1')).toBe(true)
    expect(urls.map((row) => row.url_type).sort()).toEqual(['crunchbase', 'website'])
  })

  it('does NOT merge same-named companies with different websites', () => {
    const { companies } = resolveCompanies(
      [company({ uuid: 'u1' }), company({ uuid: 'u2' })],
      [url('u1', 'website', 'helios-eu.com'), url('u2', 'website', 'helios-us.com')],
    )
    expect(companies).toHaveLength(2)
  })

  it('maps every original uuid, including unmerged ones', () => {
    const { uuidMap } = resolveCompanies(
      [
        company({ uuid: 'u1', created_at: '2026-01-01' }),
        company({ uuid: 'u2', created_at: '2026-02-01' }),
        company({ uuid: 'u3', company_name: 'Other' }),
      ],
      [url('u1', 'website', 'a.com'), url('u2', 'website', 'a.com')],
    )
    expect(uuidMap.get('u1')).toBe('u1')
    expect(uuidMap.get('u2')).toBe('u1')
    expect(uuidMap.get('u3')).toBe('u3')
  })

  it('never drops a company', () => {
    const { companies } = resolveCompanies([
      company({ uuid: 'u1' }),
      company({ uuid: 'u2', company_name: 'Other' }),
      company({ uuid: 'u3', company_name: 'Third' }),
    ])
    expect(companies).toHaveLength(3)
  })
})

describe('mergeCompanyUrls', () => {
  it('fills normalized_value so the merge key is visible in the data', () => {
    const merged = mergeCompanyUrls([url('u1', 'website', 'https://WWW.Helios.com/')], new Map())
    expect(merged[0]?.normalized_value).toBe('helios.com')
  })

  it('keeps distinct URLs of the same type', () => {
    const merged = mergeCompanyUrls(
      [
        url('u1', 'archive', 'https://web.archive.org/a'),
        url('u1', 'archive', 'https://web.archive.org/b'),
      ],
      new Map(),
    )
    expect(merged).toHaveLength(2)
  })
})

describe('remapCompanyUuids', () => {
  it('re-points child rows at the surviving uuid', () => {
    const rows = remapCompanyUuids(
      [{ company_uuid: 'u2', category: 'Team' }],
      new Map([['u2', 'u1']]),
    )
    expect(rows[0]?.company_uuid).toBe('u1')
  })

  it('leaves unknown uuids alone so build-db reports them rather than hiding them', () => {
    expect(remapCompanyUuids([{ company_uuid: 'ghost' }], new Map())[0]?.company_uuid).toBe('ghost')
  })
})

describe('nearMissReport', () => {
  it('flags distinct companies sharing a name', () => {
    const misses = nearMissReport(
      [company({ uuid: 'u1' }), company({ uuid: 'u2' })],
      [url('u1', 'website', 'helios-eu.com'), url('u2', 'website', 'helios-us.com')],
    )
    expect(misses).toHaveLength(1)
    expect(misses[0]?.reason).toContain('different websites')
  })

  it('is empty when names are genuinely distinct', () => {
    expect(
      nearMissReport([
        company({ uuid: 'u1', company_name: 'Helios' }),
        company({ uuid: 'u2', company_name: 'Selene' }),
      ]),
    ).toHaveLength(0)
  })
})
