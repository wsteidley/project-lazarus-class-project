import { describe, expect, it } from 'vitest'
import { isFresh, normalizeUrl, urlHash } from './raw-documents.js'

describe('normalizeUrl', () => {
  it('strips query and fragment', () => {
    expect(normalizeUrl('https://techcrunch.com/a/b?utm_source=x#top')).toBe(
      'https://techcrunch.com/a/b',
    )
  })

  it('lowercases the host but not the path', () => {
    expect(normalizeUrl('https://TechCrunch.com/Foo')).toBe('https://techcrunch.com/Foo')
  })

  it('drops a trailing slash', () => {
    expect(normalizeUrl('https://techcrunch.com/a/')).toBe('https://techcrunch.com/a')
  })

  it('falls back to the trimmed input when unparseable', () => {
    expect(normalizeUrl('  not a url  ')).toBe('not a url')
  })
})

describe('urlHash', () => {
  it('collapses equivalent URLs onto one key', () => {
    const variants = [
      'https://techcrunch.com/a/b',
      'https://TechCrunch.com/a/b/',
      'https://techcrunch.com/a/b?utm_campaign=q',
      'https://techcrunch.com/a/b#section',
    ]
    const hashes = new Set(variants.map(urlHash))
    expect(hashes.size).toBe(1)
  })

  it('separates genuinely different URLs', () => {
    expect(urlHash('https://techcrunch.com/a')).not.toBe(urlHash('https://techcrunch.com/b'))
  })
})

describe('isFresh', () => {
  const now = new Date('2026-07-21T00:00:00Z')
  const longAgo = '2020-01-01T00:00:00Z'

  it('treats discovery text as immutable however old', () => {
    expect(isFresh({ fetched_at: longAgo, source_type: 'discovery' }, now, 30)).toBe(true)
  })

  it('expires reassessment results past the TTL', () => {
    expect(isFresh({ fetched_at: longAgo, source_type: 'reassessment' }, now, 30)).toBe(false)
  })

  it('keeps reassessment results inside the TTL', () => {
    expect(
      isFresh({ fetched_at: '2026-07-20T00:00:00Z', source_type: 'reassessment' }, now, 30),
    ).toBe(true)
  })

  it('treats an unparseable timestamp as stale', () => {
    expect(isFresh({ fetched_at: 'garbage', source_type: 'outcome' }, now, 30)).toBe(false)
  })
})
