import { describe, expect, it } from 'vitest'
import { formatTavilyResponse, shouldFallBack } from './search.js'

describe('shouldFallBack', () => {
  it('falls back on an error', () => {
    expect(shouldFallBack('some text', new Error('rate limited'))).toBe(true)
  })

  it('falls back on empty or whitespace-only results', () => {
    expect(shouldFallBack('')).toBe(true)
    expect(shouldFallBack('   \n ')).toBe(true)
    expect(shouldFallBack(null)).toBe(true)
    expect(shouldFallBack(undefined)).toBe(true)
  })

  it('does not fall back on a usable result', () => {
    expect(shouldFallBack('Helios raised $30M in 2011')).toBe(false)
  })
})

describe('formatTavilyResponse', () => {
  it('keeps each result URL beside its content so facts can carry a source', () => {
    const text = formatTavilyResponse({
      results: [{ title: 'Helios shuts down', url: 'https://news/x', content: 'It closed.' }],
    })
    expect(text).toContain('https://news/x')
    expect(text).toContain('It closed.')
  })

  it('includes the summary answer when present', () => {
    expect(formatTavilyResponse({ answer: 'Helios closed in 2012', results: [] })).toContain(
      'Helios closed in 2012',
    )
  })

  it('returns empty for a response with nothing usable', () => {
    expect(formatTavilyResponse({ results: [] })).toBe('')
    expect(formatTavilyResponse({})).toBe('')
  })

  it('passes a plain string through, so a string provider still works', () => {
    expect(formatTavilyResponse('  raw text  ')).toBe('raw text')
  })
})
