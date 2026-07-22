import { afterEach, describe, expect, it, vi } from 'vitest'
import { checkDomainAlive, lastWaybackCaptureYear, yearFromWaybackTimestamp } from './wayback.js'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('checkDomainAlive', () => {
  it('is alive on a 2xx response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }))
    expect(await checkDomainAlive('acme.com')).toBe('alive')
  })

  it('is dead on a non-ok response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false }))
    expect(await checkDomainAlive('acme.com')).toBe('dead')
  })

  it('is dead when the fetch throws (DNS failure, timeout, ...)', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('ENOTFOUND')))
    expect(await checkDomainAlive('acme.com')).toBe('dead')
  })

  it('is unknown for an empty domain', async () => {
    expect(await checkDomainAlive('')).toBe('unknown')
  })
})

describe('yearFromWaybackTimestamp', () => {
  it('reads the 4-digit year prefix', () => {
    expect(yearFromWaybackTimestamp('20180615120000')).toBe(2018)
  })

  it('is null for an unparseable timestamp', () => {
    expect(yearFromWaybackTimestamp('')).toBeNull()
  })
})

describe('lastWaybackCaptureYear', () => {
  it('reads the closest snapshot year from the availability response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          archived_snapshots: { closest: { available: true, timestamp: '20190301000000' } },
        }),
      }),
    )
    expect(await lastWaybackCaptureYear('https://acme.com')).toBe(2019)
  })

  it('is null when there is no archived snapshot', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }))
    expect(await lastWaybackCaptureYear('https://acme.com')).toBeNull()
  })

  it('is null when the request fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('timeout')))
    expect(await lastWaybackCaptureYear('https://acme.com')).toBeNull()
  })

  it('is null for an empty url', async () => {
    expect(await lastWaybackCaptureYear('')).toBeNull()
  })
})
