import { describe, expect, it } from 'vitest'
import { computeProjections, type ProgressPoint } from './projection.js'

const point = (as_of: string, progress: number | null): ProgressPoint => ({
  dependency_id: 1,
  metric: 'm',
  scope: 'global',
  as_of,
  progress,
})

describe('computeProjections', () => {
  it('projects a crossing for a not-yet-crossed series approaching the bar', () => {
    const rows = computeProjections(
      [point('2022-01', 0.4), point('2023-01', 0.6), point('2024-01', 0.8)],
      { windowN: 3 },
    )
    const crossing = rows.find((row) => row.is_crossing === 1)
    expect(crossing).toBeDefined()
    expect(crossing?.progress).toBe(1)
    expect(crossing?.method).toBe('linear_progress_fit')
    expect(crossing?.as_of).toBe('2025-01')
    // A clean line fits with full confidence.
    expect(crossing?.confidence).toBeCloseTo(1, 5)
  })

  it('emits nothing for an already-crossed series', () => {
    expect(
      computeProjections([point('2023-01', 0.9), point('2024-01', 1.1)], { windowN: 3 }),
    ).toEqual([])
  })

  it('emits nothing for a flat or receding series (no positive slope)', () => {
    expect(
      computeProjections([point('2022-01', 0.8), point('2024-01', 0.4)], { windowN: 3 }),
    ).toEqual([])
  })

  it('emits nothing for a single-point series', () => {
    expect(computeProjections([point('2024-01', 0.5)], { windowN: 3 })).toEqual([])
  })

  it('skips null-progress rows and a crossing beyond the horizon', () => {
    // Barely-rising slope pushes the crossing decades out — past maxHorizonYears.
    const rows = computeProjections(
      [point('2022-01', 0.1), point('2023-01', 0.11), point('2024-01', 0.12)],
      { windowN: 3, maxHorizonYears: 5 },
    )
    expect(rows).toEqual([])
  })
})
