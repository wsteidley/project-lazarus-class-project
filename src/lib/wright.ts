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

// Everything the fit needs for one (dependency, metric, segment, scope) series. When present,
// `baseline` and `threshold` follow the same rule the progress view uses, so a Wright
// projection lands on the identical 0->1 axis as a linear one.
//
// `direction`, `baseline` and `threshold` are NULLABLE, and that is the point: they come from
// a curated bar, and a learning rate does not need one. The log-log slope is a property of
// cost against cumulative capacity alone; the bar only enters when converting that slope into
// a crossing DATE and onto the 0->1 progress axis. Wind's total_installed_cost is the case
// that forced this — it is the Wright-fittable wind series and it has no viability bar, so
// requiring one would have meant either inventing a threshold or silently getting no fit.
export type WrightSeries = {
  entity_id: number
  // The bar's owner, when the series was assembled with one. A fit needs no bar; a projected
  // crossing does, and it must be able to say whose.
  dependency_id?: number | null
  metric: string
  scope: string
  segment: string
  direction: string | null
  baseline: number | null
  threshold: number | null
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
  entity_id: number
  metric: string
  scope: string
  segment: string
  reason: string
}

// A learning-rate fit that succeeded, kept SEPARATELY from the projection rows because the two
// are not the same finding. A series can have an excellent, well-evidenced learning curve and
// still yield no projection -- that is exactly what happens once a series is already past its
// bar (onshore wind crossed in 2019, battery in 2025). Returning only projections would throw
// away the learning rate for every technology that has already succeeded, which is most of the
// interesting ones.
export type WrightFit = {
  entity_id: number
  metric: string
  scope: string
  segment: string
  learning_rate: number
  b: number
  r2: number
  n_pairs: number
  first_as_of: string
  last_as_of: string
}

export type WrightResult = { rows: ProjectionRow[]; skipped: WrightSkip[]; fits: WrightFit[] }

const DEFAULT_MIN_PAIRS = 4
const DEFAULT_MAX_HORIZON = 30

// Metrics that measure DELIVERED-ENERGY cost rather than hardware cost. Wright's law is a
// manufacturing-learning model: cost per unit built falls with cumulative units built. LCOE is
// not that quantity -- it also moves with capacity factor, financing terms and siting, none of
// which are learning-by-doing. Fitting it against cumulative GW credits taller turbines and
// cheaper debt to the factory floor.
//
// This is not a theoretical worry. Onshore wind LCOE fits at 43%/doubling against real IRENA
// data -- roughly double any published onshore-wind learning rate -- precisely because a large
// share of that LCOE fall came from capacity-factor gains rather than cheaper turbines. The
// structure spec already draws this line ("Wright uses module_price; thresholds/viability use
// LCOE"); this enforces it instead of trusting each caller to remember.
//
// The fix is not to relax the gate: it is to load a hardware series (IRENA publishes
// total_installed_cost in USD/kW per technology) and fit that.
export const DELIVERED_ENERGY_METRICS = new Set(['LCOE', 'lcoe_by_market'])

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
  const fits: WrightFit[] = []

  for (const entry of series) {
    const key = {
      entity_id: entry.entity_id,
      dependency_id: entry.dependency_id ?? null,
      metric: entry.metric,
      scope: entry.scope,
      segment: entry.segment,
    }
    const skip = (reason: string): void => {
      skipped.push({ ...key, reason })
    }

    // A learning curve is a falling-cost model. An above_is_better metric (carbon price)
    // isn't one, and fitting it would produce confident nonsense.
    //
    // Only checked when a direction was declared. A series with no bar has none — and that is
    // a real, if acceptable, weakening: such a series can no longer be refused BEFORE fitting
    // on the strength of a curated direction. The backstop is the negative-slope guard below,
    // which a rising series cannot pass, so a receding curve still cannot produce a learning
    // rate; it just gets rejected by its own data rather than by its label.
    if (entry.direction !== null && entry.direction !== 'below_is_better') {
      skip('not a cost curve (direction is not below_is_better)')
      continue
    }
    if (DELIVERED_ENERGY_METRICS.has(entry.metric)) {
      skip(
        `${entry.metric} is a delivered-energy cost, not a hardware cost — Wright needs a ` +
          'manufacturing series (module price / total installed cost)',
      )
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

    // Recorded HERE, before the forward-path and crossing guards, because the learning rate is
    // already established at this point and does not depend on any of them. Recording it later
    // would silently discard the fit for every already-crossed series -- i.e. for the
    // technologies that actually won.
    const costAsOf = entry.costPoints.map((point) => point.as_of).sort()
    fits.push({
      ...key,
      learning_rate: learningRate(b),
      b,
      r2: costFit.r2,
      n_pairs: pairs.length,
      first_as_of: costAsOf[0] as string,
      last_as_of: costAsOf[costAsOf.length - 1] as string,
    })

    // Everything from here on produces the PROJECTION, which is where the bar finally
    // matters: a crossing date is "when does this curve reach T", and the 0->1 axis is
    // measured between B and T. No bar means no crossing to compute — the fit above still
    // stands and has already been recorded.
    const { baseline, threshold } = entry
    if (threshold === null || baseline === null) {
      skip('no threshold declared — fit only, no crossing to project')
      continue
    }
    if (threshold <= 0) {
      skip('threshold is not positive — log-space crossing undefined')
      continue
    }

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
    const crossingYear = (lnA - Math.log(threshold) - b * c) / (b * d)
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
    const progressAt = (year: number): number => (baseline - costAt(year)) / (baseline - threshold)

    const rate = learningRate(b)
    const base = {
      entity_id: entry.entity_id,
      dependency_id: entry.dependency_id ?? null,
      metric: entry.metric,
      scope: entry.scope,
      segment: entry.segment,
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

  return { rows, skipped, fits }
}
