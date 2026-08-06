// Projection: the one derived layer that is genuine computation, not a view. For a series
// not yet crossed, fit a line to its recent normalized-progress slope and extrapolate a
// projected crossing date plus a few future points. Every output is flagged projected=1 and
// carries a confidence (the fit R^2), because learning curves aren't deterministic — a
// projected crossing is an estimate, not a fact. Pure and side-effect free; build-db feeds
// it the `progress` view and inserts what it returns.

import { PROJECTION_METHOD } from '../schemas.js'

// One row of the `progress` view the projection consumes.
export type ProgressPoint = {
  entity_id: number
  // Carried through so a projected crossing can name the bar it is a crossing OF.
  dependency_id?: number | null
  metric: string
  scope: string
  segment: string
  as_of: string
  progress: number | null
}

// One projected point, shaped for insertion into metric_projections.
export type ProjectionRow = {
  entity_id: number
  // Whose bar the crossing is against. Null when the fit ran with no bar at all.
  dependency_id?: number | null
  metric: string
  scope: string
  segment: string
  as_of: string
  progress: number
  method: (typeof PROJECTION_METHOD)[number]
  fit_window_n: number
  confidence: number
  is_crossing: number
  note: string | null
}

export type ProjectionConfig = {
  // Trailing observations to fit the slope on (shared with trajectory).
  windowN: number
  // Don't project a crossing further out than this many years past the latest observation.
  maxHorizonYears?: number
}

// 'YYYY' or 'YYYY-MM' -> decimal year (month contributes a twelfth each).
export const asOfToYear = (asOf: string): number => {
  const [year, month] = asOf.split('-')
  return Number(year) + (month ? (Number(month) - 1) / 12 : 0)
}

// Decimal year -> 'YYYY-MM', rounding to the nearest month.
export const yearToAsOf = (year: number): string => {
  const whole = Math.floor(year)
  const month = Math.min(12, Math.max(1, Math.round((year - whole) * 12) + 1))
  return `${whole}-${String(month).padStart(2, '0')}`
}

// Ordinary least squares of y on x, plus R^2. Returns null when x has no spread.
export const linearFit = (
  pairs: [number, number][],
): { slope: number; intercept: number; r2: number } | null => {
  const n = pairs.length
  const meanX = pairs.reduce((sum, [x]) => sum + x, 0) / n
  const meanY = pairs.reduce((sum, [, y]) => sum + y, 0) / n
  let sxx = 0
  let sxy = 0
  let syy = 0
  for (const [x, y] of pairs) {
    const dx = x - meanX
    const dy = y - meanY
    sxx += dx * dx
    sxy += dx * dy
    syy += dy * dy
  }
  if (sxx === 0) {
    return null
  }
  const slope = sxy / sxx
  const intercept = meanY - slope * meanX
  // Perfect horizontal data (syy === 0) fits with R^2 = 1.
  const r2 = syy === 0 ? 1 : Math.max(0, Math.min(1, (sxy * sxy) / (sxx * syy)))
  return { slope, intercept, r2 }
}

const seriesKey = (point: ProgressPoint): string =>
  `${point.entity_id}::${point.dependency_id ?? ''}::${point.metric}::${point.segment}::${point.scope}`

// Groups progress rows into series and, for each series not yet crossed, projects future
// points up to (and including) the crossing. Series that are already crossed, too short, or
// not approaching the bar (flat/receding slope) yield nothing — a forecast that never
// reaches viability isn't emitted.
export const computeProjections = (
  points: ProgressPoint[],
  config: ProjectionConfig,
): ProjectionRow[] => {
  const maxHorizon = config.maxHorizonYears ?? 30
  const bySeries = new Map<string, ProgressPoint[]>()
  for (const point of points) {
    if (point.progress === null) {
      continue
    }
    const key = seriesKey(point)
    const list = bySeries.get(key) ?? []
    list.push(point)
    bySeries.set(key, list)
  }

  const rows: ProjectionRow[] = []
  for (const series of bySeries.values()) {
    const sorted = [...series].sort((a, b) => a.as_of.localeCompare(b.as_of))
    const latest = sorted.at(-1)
    if (!latest) {
      continue
    }
    // Already crossed — nothing to project.
    if ((latest.progress ?? 0) >= 1) {
      continue
    }
    const window = sorted.slice(-config.windowN)
    if (window.length < 2) {
      continue
    }
    const fit = linearFit(
      window.map((point) => [asOfToYear(point.as_of), point.progress as number]),
    )
    // No spread, or flat/receding — can't project a crossing.
    if (!fit || fit.slope <= 0) {
      continue
    }
    const latestYear = asOfToYear(latest.as_of)
    const crossingYear = (1 - fit.intercept) / fit.slope
    if (!Number.isFinite(crossingYear) || crossingYear <= latestYear) {
      continue
    }
    if (crossingYear - latestYear > maxHorizon) {
      continue
    }

    const base = {
      entity_id: latest.entity_id,
      dependency_id: latest.dependency_id ?? null,
      metric: latest.metric,
      scope: latest.scope,
      segment: latest.segment,
      method: PROJECTION_METHOD[0],
      fit_window_n: window.length,
      confidence: fit.r2,
    }
    // Yearly future points up to the crossing, then the crossing point itself at progress 1.
    for (let year = Math.floor(latestYear) + 1; year < crossingYear; year++) {
      rows.push({
        ...base,
        as_of: yearToAsOf(year),
        progress: fit.slope * year + fit.intercept,
        is_crossing: 0,
        note: null,
      })
    }
    rows.push({
      ...base,
      as_of: yearToAsOf(crossingYear),
      progress: 1,
      is_crossing: 1,
      note: `projected crossing ~${yearToAsOf(crossingYear)} (linear fit, R2=${fit.r2.toFixed(2)})`,
    })
  }
  return rows
}
