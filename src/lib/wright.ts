// Wright's law: cost falls a fixed fraction per doubling of cumulative production. For the
// cost-curve heroes (solar above all) that is the physically-motivated model, and it beats
// the linear-in-time fit for the reason that matters here — a technology's cost tracks how
// much of it has been built, not how many years have passed. A decade of no deployment
// moves a learning curve not at all; the linear fit would happily extrapolate through it.
//
// Two fits, both on observed history only:
//   1. the learning rate — ln(cost) on ln(cumulative capacity), slope -b
//   2. the forward capacity path — ln(capacity) on time, extrapolated
// Composing them turns "20% cheaper per doubling" into a date, which is what a crossing
// projection needs. Neither fit uses a scenario or forecast input; see R1 in
// specs/hero-thresholds-spec-v3.2.md for why the ETS capacity path was rejected.
//
// Pure and side-effect free. build-db feeds it observations + capacity and inserts what it
// returns; anything that fails a guard falls back to the linear fit in projection.ts.

import type { PROJECTION_METHOD } from '../schemas.js'
import { asOfToYear, linearFit, type ProjectionRow, yearToAsOf } from './projection.js'

// A dated scalar — a cost observation or a cumulative-capacity point.
export type SeriesPoint = { as_of: string; value: number }

// Everything the fit needs for one (dependency, metric, scope) series. `baseline` and
// `threshold` come from the same place the progress view gets them, so Wright's output
// lands on the identical 0->1 axis as the linear projection's.
export type WrightSeries = {
  dependency_id: number
  metric: string
  scope: string
  direction: string
  baseline: number
  threshold: number
  costPoints: SeriesPoint[]
  capacityPoints: SeriesPoint[]
}

export type WrightConfig = {
  // Don't project a crossing further out than this many years past the latest observation.
  maxHorizonYears?: number
  // Minimum (cost, capacity) pairs before a learning rate is worth believing.
  minPairs?: number
}

// Why a series could not be fitted, so the caller can report the split rather than silently
// falling back. Every reason here is a real property of today's data, not a hypothetical:
// wind LCOE has 2 cost points, battery has 1 capacity point.
export type WrightSkip = {
  dependency_id: number
  metric: string
  scope: string
  reason: string
}

export type WrightResult = { rows: ProjectionRow[]; skipped: WrightSkip[] }

const DEFAULT_MIN_PAIRS = 4
const DEFAULT_MAX_HORIZON = 30

// Learning rate from the log-log slope: a doubling multiplies cost by 2^(-b), so the
// fractional drop per doubling is 1 - 2^(-b). Reported in the note because it is the number
// a human can sanity-check (~20% for solar, ~18% for batteries).
export const learningRate = (b: number): number => 1 - 2 ** -b

// Pairs cost with capacity on the calendar year they share. Capacity is annual and cost may
// be denser or sparser; the year is the only key both series agree on.
const pairByYear = (cost: SeriesPoint[], capacity: SeriesPoint[]): [number, number][] => {
  const capacityByYear = new Map<string, number>()
  for (const point of capacity) {
    const year = point.as_of.slice(0, 4)
    if (point.value > 0) {
      capacityByYear.set(year, point.value)
    }
  }
  const pairs: [number, number][] = []
  for (const point of cost) {
    const deployed = capacityByYear.get(point.as_of.slice(0, 4))
    if (deployed !== undefined && point.value > 0) {
      // [ln(capacity), ln(cost)] — regressing cost on deployment, not on time.
      pairs.push([Math.log(deployed), Math.log(point.value)])
    }
  }
  return pairs
}

export const computeWrightProjections = (
  series: WrightSeries[],
  config: WrightConfig = {},
): WrightResult => {
  const maxHorizon = config.maxHorizonYears ?? DEFAULT_MAX_HORIZON
  const minPairs = config.minPairs ?? DEFAULT_MIN_PAIRS
  const rows: ProjectionRow[] = []
  const skipped: WrightSkip[] = []

  for (const entry of series) {
    const key = {
      dependency_id: entry.dependency_id,
      metric: entry.metric,
      scope: entry.scope,
    }
    const skip = (reason: string): void => {
      skipped.push({ ...key, reason })
    }

    // A learning curve is a falling-cost model. An above_is_better metric (carbon price)
    // isn't one, and fitting it would produce confident nonsense.
    if (entry.direction !== 'below_is_better') {
      skip('not a cost curve (direction is not below_is_better)')
      continue
    }
    if (entry.threshold <= 0) {
      skip('threshold is not positive — log-space fit undefined')
      continue
    }

    const pairs = pairByYear(entry.costPoints, entry.capacityPoints)
    if (pairs.length < minPairs) {
      skip(`only ${pairs.length} cost/capacity pairs (need ${minPairs})`)
      continue
    }

    // Fit 1: the learning rate. ln(cost) = lnA - b*ln(capacity), so slope = -b.
    const costFit = linearFit(pairs)
    if (!costFit || costFit.slope >= 0) {
      skip('no falling cost-vs-capacity relationship to fit')
      continue
    }
    const b = -costFit.slope
    const lnA = costFit.intercept

    // Fit 2: the forward capacity path. ln(capacity) = c + d*t.
    //
    // KNOWN LIMITATION (recorded, not fixed — see R1): this implies capacity doubles
    // indefinitely at the historical rate. It has no saturation term and no policy input, so
    // a far-future crossing from this path should be read as "if deployment keeps
    // compounding as it has", not as a forecast. A better forward path is tracked as a
    // nice-to-have; it is explicitly not worth relaxing the cited-anchor rule for.
    const capacityPairs: [number, number][] = entry.capacityPoints
      .filter((point) => point.value > 0)
      .map((point) => [asOfToYear(point.as_of), Math.log(point.value)])
    if (capacityPairs.length < minPairs) {
      skip(`only ${capacityPairs.length} capacity points for the forward path`)
      continue
    }
    const capacityFit = linearFit(capacityPairs)
    if (!capacityFit || capacityFit.slope <= 0) {
      skip('cumulative capacity is not growing — no forward doubling schedule')
      continue
    }
    const { slope: d, intercept: c } = capacityFit

    // Crossing: ln(threshold) = lnA - b*(c + d*t)  =>  t = (lnA - ln(T) - b*c) / (b*d).
    const crossingYear = (lnA - Math.log(entry.threshold) - b * c) / (b * d)
    const latestYear = Math.max(...entry.costPoints.map((point) => asOfToYear(point.as_of)))
    if (!Number.isFinite(crossingYear) || crossingYear <= latestYear) {
      skip('fitted curve is already past the bar or yields no future crossing')
      continue
    }
    if (crossingYear - latestYear > maxHorizon) {
      skip(
        `projected crossing is ${Math.round(crossingYear - latestYear)}y out (cap ${maxHorizon}y)`,
      )
      continue
    }

    // Cost at time t, then onto the same 0->1 progress axis the progress view defines, so a
    // wright row and a linear row mean the same thing on a chart.
    const costAt = (year: number): number => Math.exp(lnA - b * (c + d * year))
    const progressAt = (year: number): number =>
      (entry.baseline - costAt(year)) / (entry.baseline - entry.threshold)

    const rate = learningRate(b)
    const base = {
      dependency_id: entry.dependency_id,
      metric: entry.metric,
      scope: entry.scope,
      method: 'wright' as (typeof PROJECTION_METHOD)[number],
      fit_window_n: pairs.length,
      confidence: costFit.r2,
    }
    for (let year = Math.floor(latestYear) + 1; year < crossingYear; year++) {
      rows.push({
        ...base,
        as_of: yearToAsOf(year),
        progress: progressAt(year),
        is_crossing: 0,
        note: null,
      })
    }
    rows.push({
      ...base,
      as_of: yearToAsOf(crossingYear),
      progress: 1,
      is_crossing: 1,
      note:
        `projected crossing ~${yearToAsOf(crossingYear)} (Wright fit: ` +
        `${(rate * 100).toFixed(0)}% per doubling, R2=${costFit.r2.toFixed(2)}, ` +
        `${pairs.length} pairs; capacity path R2=${capacityFit.r2.toFixed(2)})`,
    })
  }

  return { rows, skipped }
}
