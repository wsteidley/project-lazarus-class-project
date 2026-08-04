import { describe, expect, it } from 'vitest'
import { computeWrightProjections, learningRate, type WrightSeries } from './wright.js'

// A synthetic exact power law: cost = A * capacity^(-b), with capacity doubling on a fixed
// schedule. Because the data is generated from the model the fit is meant to recover, the
// test can assert the recovered learning rate and crossing year rather than just "it ran".
const syntheticSeries = (options: {
  learningRateFraction: number
  startCost: number
  startCapacity: number
  capacityGrowthPerYear: number
  years: number
  threshold: number
  baseline?: number
}): WrightSeries => {
  const b = -Math.log2(1 - options.learningRateFraction)
  const costPoints: { as_of: string; value: number }[] = []
  const capacityPoints: { as_of: string; value: number }[] = []
  for (let index = 0; index < options.years; index++) {
    const capacity = options.startCapacity * options.capacityGrowthPerYear ** index
    const cost = options.startCost * (capacity / options.startCapacity) ** -b
    const year = 2000 + index
    capacityPoints.push({ as_of: `${year}-12`, value: capacity })
    costPoints.push({ as_of: `${year}-12`, value: cost })
  }
  return {
    dependency_id: 1,
    metric: 'module price',
    scope: 'global',
    segment: 'all',
    direction: 'below_is_better',
    baseline: options.baseline ?? options.startCost,
    threshold: options.threshold,
    costPoints,
    capacityPoints,
  }
}

describe('learningRate', () => {
  // The number a human sanity-checks the fit against: ~20% per doubling for solar.
  it('converts a log-log slope into a fraction per doubling', () => {
    expect(learningRate(-Math.log2(0.8))).toBeCloseTo(0.2, 10)
    expect(learningRate(-Math.log2(0.82))).toBeCloseTo(0.18, 10)
    expect(learningRate(0)).toBe(0)
  })
})

describe('computeWrightProjections', () => {
  it('recovers a known learning rate from an exact power law', () => {
    const series = syntheticSeries({
      learningRateFraction: 0.2,
      startCost: 10,
      startCapacity: 1,
      capacityGrowthPerYear: 1.3,
      years: 20,
      threshold: 0.5,
    })
    const { rows, skipped } = computeWrightProjections([series])
    expect(skipped).toEqual([])
    const crossing = rows.find((row) => row.is_crossing === 1)
    expect(crossing).toBeDefined()
    // Exact data -> a perfect fit, and the note reports the rate it recovered.
    expect(crossing?.confidence).toBeCloseTo(1, 6)
    expect(crossing?.note).toContain('20% per doubling')
    expect(crossing?.method).toBe('wright')
    expect(crossing?.progress).toBe(1)
  })

  it('places the crossing where the power law actually reaches the bar', () => {
    // cost = 10 * capacity^-b, capacity = 1.3^t. With a 20% learning rate the bar of 2.5 is
    // reached when capacity^b = 4, which is a checkable closed form. The +11/12 is not a
    // fudge: the points are dated December, so the series' own time axis starts at 2000.917.
    const series = syntheticSeries({
      learningRateFraction: 0.2,
      startCost: 10,
      startCapacity: 1,
      capacityGrowthPerYear: 1.3,
      years: 10,
      threshold: 2.5,
    })
    const b = -Math.log2(0.8)
    const expectedYear = 2000 + 11 / 12 + Math.log(4 ** (1 / b)) / Math.log(1.3)
    const crossing = computeWrightProjections([series]).rows.find((row) => row.is_crossing === 1)
    expect(Number(crossing?.as_of.slice(0, 4))).toBe(Math.floor(expectedYear))
  })

  it('emits yearly points before the crossing, all flagged wright', () => {
    const series = syntheticSeries({
      learningRateFraction: 0.2,
      startCost: 10,
      startCapacity: 1,
      capacityGrowthPerYear: 1.3,
      years: 15,
      threshold: 0.5,
    })
    const { rows } = computeWrightProjections([series])
    expect(rows.length).toBeGreaterThan(1)
    expect(rows.every((row) => row.method === 'wright')).toBe(true)
    expect(rows.filter((row) => row.is_crossing === 1)).toHaveLength(1)
    // Monotonically approaching the bar, and dated in order.
    const asOfs = rows.map((row) => row.as_of)
    expect([...asOfs].sort()).toEqual(asOfs)
  })
})

describe('computeWrightProjections guards (these are today’s real data, not hypotheticals)', () => {
  const base = syntheticSeries({
    learningRateFraction: 0.2,
    startCost: 10,
    startCapacity: 1,
    capacityGrowthPerYear: 1.3,
    years: 20,
    threshold: 0.5,
  })

  // Wind LCOE has exactly 2 cost points; battery has exactly 1 capacity point. Neither can
  // support a learning curve, and both must fall back rather than fit noise.
  it('skips a series with too few cost/capacity pairs', () => {
    const thin: WrightSeries = { ...base, costPoints: base.costPoints.slice(0, 2) }
    const { rows, skipped } = computeWrightProjections([thin])
    expect(rows).toEqual([])
    expect(skipped[0]?.reason).toContain('pairs')
  })

  it('skips a series with a single capacity point', () => {
    const thin: WrightSeries = { ...base, capacityPoints: base.capacityPoints.slice(0, 1) }
    const { rows, skipped } = computeWrightProjections([thin])
    expect(rows).toEqual([])
    expect(skipped).toHaveLength(1)
  })

  // A learning curve is a falling-cost model; carbon price is not one.
  it('refuses an above_is_better metric', () => {
    const rising: WrightSeries = { ...base, direction: 'above_is_better' }
    const { rows, skipped } = computeWrightProjections([rising])
    expect(rows).toEqual([])
    expect(skipped[0]?.reason).toContain('not a cost curve')
  })

  it('refuses a non-positive threshold (log space is undefined there)', () => {
    const { rows, skipped } = computeWrightProjections([{ ...base, threshold: 0 }])
    expect(rows).toEqual([])
    expect(skipped[0]?.reason).toContain('threshold is not positive')
  })

  it('skips when cost rises with deployment — there is no learning curve to fit', () => {
    const rising: WrightSeries = {
      ...base,
      costPoints: base.costPoints.map((point, index) => ({ ...point, value: 1 + index })),
    }
    const { rows, skipped } = computeWrightProjections([rising])
    expect(rows).toEqual([])
    expect(skipped[0]?.reason).toContain('no falling cost-vs-capacity relationship')
  })

  it('skips a crossing beyond the horizon cap rather than projecting decades out', () => {
    const series = syntheticSeries({
      learningRateFraction: 0.02,
      startCost: 1000,
      startCapacity: 1,
      capacityGrowthPerYear: 1.01,
      years: 20,
      threshold: 0.001,
    })
    const { rows, skipped } = computeWrightProjections([series], { maxHorizonYears: 30 })
    expect(rows).toEqual([])
    expect(skipped[0]?.reason).toMatch(/y out \(cap 30y\)|already past/)
  })

  it('skips a series already below its bar (nothing to project)', () => {
    const { rows, skipped } = computeWrightProjections([{ ...base, threshold: 1e9 }])
    expect(rows).toEqual([])
    expect(skipped[0]?.reason).toContain('already past the bar')
  })

  it('reports every skip with its series key, so the caller can log the fallback', () => {
    const thin: WrightSeries = { ...base, costPoints: base.costPoints.slice(0, 2) }
    const { skipped } = computeWrightProjections([thin])
    expect(skipped[0]).toMatchObject({ dependency_id: 1, metric: 'module price', scope: 'global' })
  })

  // The case onshore wind and battery actually hit: a real, well-evidenced learning curve on a
  // series that has already crossed its bar. There is nothing to forecast, but the learning rate
  // is the finding — returning only projections would silently discard it for every technology
  // that already succeeded.
  it('still reports the fit for an already-crossed series that yields no projection', () => {
    const { rows, fits } = computeWrightProjections([{ ...base, threshold: 1e9 }])
    expect(rows).toEqual([])
    expect(fits).toHaveLength(1)
    expect(fits[0]?.learning_rate).toBeCloseTo(0.2, 6)
    expect(fits[0]?.r2).toBeCloseTo(1, 6)
    expect(fits[0]?.first_as_of).toBe('2000-12')
  })

  it('reports no fit when the learning-rate regression itself could not run', () => {
    const thin: WrightSeries = { ...base, costPoints: base.costPoints.slice(0, 2) }
    expect(computeWrightProjections([thin]).fits).toEqual([])
  })

  // Guards a number that looked plausible and was wrong: onshore wind LCOE fits at 43%/doubling
  // on real IRENA data, about double any published onshore-wind learning rate, because LCOE
  // improvements include capacity-factor gains that are not manufacturing learning.
  it('refuses to fit a delivered-energy cost like LCOE', () => {
    const { rows, fits, skipped } = computeWrightProjections([{ ...base, metric: 'LCOE' }])
    expect(rows).toEqual([])
    expect(fits).toEqual([])
    expect(skipped[0]?.reason).toContain('delivered-energy cost')
  })

  // A hardware series with no viability bar — wind's total_installed_cost. The learning rate
  // is a property of cost against capacity and needs no threshold; only the crossing date
  // does. Requiring a bar here would have meant inventing one or silently getting no fit.
  it('fits a series with no threshold, and projects nothing for it', () => {
    const { rows, fits, skipped } = computeWrightProjections([
      { ...base, threshold: null, baseline: null, direction: null },
    ])
    expect(fits).toHaveLength(1)
    expect(fits[0]?.learning_rate).toBeCloseTo(0.2, 6)
    expect(rows).toEqual([])
    expect(skipped[0]?.reason).toContain('no threshold declared')
  })

  // The guard that replaces the curated `direction` check when no bar exists: a rising series
  // cannot produce a negative log-log slope, so it still cannot earn a learning rate.
  it('refuses a rising series even with no direction declared', () => {
    const rising: WrightSeries = {
      ...base,
      threshold: null,
      baseline: null,
      direction: null,
      costPoints: base.costPoints.map((point, index) => ({ ...point, value: 100 + index * 10 })),
    }
    const { fits, skipped } = computeWrightProjections([rising])
    expect(fits).toEqual([])
    expect(skipped[0]?.reason).toContain('no falling cost-vs-capacity relationship')
  })

  it('keeps segment in the fit key so two slices of one metric stay separate', () => {
    const { fits } = computeWrightProjections([
      { ...base, segment: 'onshore', threshold: 1e9 },
      { ...base, segment: 'offshore', threshold: 1e9 },
    ])
    expect(fits.map((fit) => fit.segment)).toEqual(['onshore', 'offshore'])
  })
})
