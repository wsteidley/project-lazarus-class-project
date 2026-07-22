// Cheap, API-key-free death signals for the outcome pass. Both degrade to a null/
// 'unknown' value rather than throwing — a network hiccup should cost a signal, not
// break the run, the same posture as gatherSearchContext's per-provider fallback.

const FETCH_TIMEOUT_MS = 8_000

export type DomainStatus = 'alive' | 'dead' | 'unknown'

// A dead site is a signal, not proof: DNS failures, timeouts, and non-2xx/3xx
// responses all read as 'dead'; anything unexpected (e.g. no scheme) is 'unknown'.
export const checkDomainAlive = async (domain: string): Promise<DomainStatus> => {
  const trimmed = domain.trim()
  if (!trimmed) {
    return 'unknown'
  }
  const url = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`
  try {
    const response = await fetch(url, {
      method: 'HEAD',
      redirect: 'follow',
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    })
    return response.ok ? 'alive' : 'dead'
  } catch {
    return 'dead'
  }
}

type WaybackAvailability = {
  archived_snapshots?: {
    closest?: {
      available?: boolean
      url?: string
      timestamp?: string // yyyyMMddhhmmss
    }
  }
}

// Wayback capture timestamps are yyyyMMddhhmmss; the year is the first 4 digits.
export const yearFromWaybackTimestamp = (timestamp: string): number | null => {
  const match = timestamp.match(/^(\d{4})/)
  return match?.[1] ? Number(match[1]) : null
}

// The year of the last known Wayback capture for a URL, via the (unauthenticated)
// Availability API — the death-year estimate the overdue-ness calc needs. Null when
// there's no capture on record or the lookup fails.
export const lastWaybackCaptureYear = async (url: string): Promise<number | null> => {
  const trimmed = url.trim()
  if (!trimmed) {
    return null
  }
  try {
    const response = await fetch(
      `https://archive.org/wayback/available?url=${encodeURIComponent(trimmed)}`,
      { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) },
    )
    if (!response.ok) {
      return null
    }
    const data = (await response.json()) as WaybackAvailability
    const timestamp = data.archived_snapshots?.closest?.timestamp
    return timestamp ? yearFromWaybackTimestamp(timestamp) : null
  } catch {
    return null
  }
}
