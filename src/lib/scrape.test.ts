import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { isLikelyArticleUrl, parseListingsHtml } from './scrape.js'

// The saved TechCrunch listing page checked into the legacy scraper folder.
const fixturePath = fileURLToPath(
  new URL('../../scraping/techcrunch/articles_page.txt', import.meta.url),
)
const fixtureHtml = readFileSync(fixturePath, 'utf-8')

describe('parseListingsHtml', () => {
  const entries = parseListingsHtml(fixtureHtml, '2026-07-20T00:00:00')

  it('extracts every article entry from the fixture', () => {
    expect(entries).toHaveLength(37)
  })

  it('pulls title, url, author and date for the first entry', () => {
    expect(entries[0]).toEqual({
      title: 'The brightest bling of TechCrunch Disrupt 2024',
      url: 'https://techcrunch.com/2024/11/04/the-brightest-bling-of-techcrunch-disrupt-2024/',
      author: 'Amanda Silberling',
      publication_date: '2024-11-04T09:53:48-08:00',
      timestamp: '2026-07-20T00:00:00',
    })
  })

  it('carries the provided timestamp onto every entry', () => {
    expect(entries.every((entry) => entry.timestamp === '2026-07-20T00:00:00')).toBe(true)
  })
})

describe('isLikelyArticleUrl', () => {
  it('keeps standard article URLs', () => {
    expect(isLikelyArticleUrl('https://techcrunch.com/2026/07/17/some-article/')).toBe(true)
  })

  it('drops non-article templates (podcast, video, events, listings)', () => {
    expect(
      isLikelyArticleUrl('https://techcrunch.com/podcast/impact-investing-climate-tech/'),
    ).toBe(false)
    expect(isLikelyArticleUrl('https://techcrunch.com/video/some-clip/')).toBe(false)
    expect(isLikelyArticleUrl('https://techcrunch.com/tag/climate/')).toBe(false)
    expect(isLikelyArticleUrl('https://techcrunch.com/category/startups/')).toBe(false)
  })

  it('drops non-https URLs', () => {
    expect(isLikelyArticleUrl('http://techcrunch.com/2026/07/17/some-article/')).toBe(false)
    expect(isLikelyArticleUrl('')).toBe(false)
  })
})
