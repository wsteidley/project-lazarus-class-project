import { DatabaseSync } from 'node:sqlite'
import { beforeEach, describe, expect, it } from 'vitest'
import { createTablesSql } from './db-schema.js'

// Exercises the progress + trajectory SQL views end to end: seed observations, thresholds,
// and the tunable config, then assert the derived states fall out correctly.

type TrajectoryRow = {
  metric: string
  state: string
  currently_crossed: number
  became_viable_date: string | null
  latest_progress: number
}

let db: DatabaseSync

const addDependency = (name: string): number =>
  Number(db.prepare('INSERT INTO dependencies (name) VALUES (?)').run(name).lastInsertRowid)

const addThreshold = (
  depId: number,
  metric: string,
  value: number,
  direction: string,
  extra: { contested?: number; alt?: number } = {},
): void => {
  db.prepare(
    `INSERT INTO dependency_thresholds
      (dependency_id, metric, scope, threshold_value, threshold_direction, threshold_contested, threshold_alt_value)
     VALUES (?, ?, 'global', ?, ?, ?, ?)`,
  ).run(depId, metric, value, direction, extra.contested ?? 0, extra.alt ?? null)
}

const addObs = (depId: number, metric: string, value: number, asOf: string): void => {
  db.prepare(
    `INSERT INTO metric_observations (dependency_id, metric, value, unit, as_of, scope, method)
     VALUES (?, ?, ?, 'u', ?, 'global', 'curated')`,
  ).run(depId, metric, value, asOf)
}

const trajectoryFor = (depId: number): TrajectoryRow =>
  db
    .prepare(
      `SELECT metric, state, currently_crossed, became_viable_date, latest_progress
       FROM trajectory WHERE dependency_id = ?`,
    )
    .get(depId) as TrajectoryRow

const latestProgressFor = (depId: number): { progress: number | null; progress_status: string } =>
  db
    .prepare(
      `SELECT progress, progress_status FROM progress
       WHERE dependency_id = ? ORDER BY as_of DESC LIMIT 1`,
    )
    .get(depId) as { progress: number | null; progress_status: string }

beforeEach(() => {
  db = new DatabaseSync(':memory:')
  db.exec('PRAGMA foreign_keys = ON')
  db.exec(createTablesSql())
  db.prepare(
    'INSERT INTO trajectory_config (window_n, plateau_slope_threshold) VALUES (3, 0.03)',
  ).run()
})

describe('progress + trajectory views', () => {
  it('marks a declining cost series that passed the bar as crossed and improving', () => {
    const id = addDependency('Cross')
    addThreshold(id, 'm', 100, 'below_is_better')
    for (const [value, asOf] of [
      [200, '2021-12'],
      [150, '2022-12'],
      [100, '2023-12'],
      [80, '2024-12'],
    ] as const) {
      addObs(id, 'm', value, asOf)
    }
    const row = trajectoryFor(id)
    expect(row.currently_crossed).toBe(1)
    expect(row.became_viable_date).toBe('2023-12')
    expect(row.state).toBe('improving')
    expect(row.latest_progress).toBeGreaterThanOrEqual(1)
  })

  it('classifies a metric moving away from the bar as receded', () => {
    const id = addDependency('Recede')
    addThreshold(id, 'm', 2, 'below_is_better')
    for (const [value, asOf] of [
      [3, '2007-12'],
      [4, '2024-12'],
      [5, '2025-12'],
    ] as const) {
      addObs(id, 'm', value, asOf)
    }
    const row = trajectoryFor(id)
    expect(row.state).toBe('receded')
    expect(row.currently_crossed).toBe(0)
  })

  it('classifies a series flat after prior gains as plateaued', () => {
    const id = addDependency('Plateau')
    addThreshold(id, 'm', 50, 'below_is_better')
    for (const [value, asOf] of [
      [100, '2020-12'],
      [55, '2021-12'],
      [55, '2022-12'],
      [55, '2023-12'],
    ] as const) {
      addObs(id, 'm', value, asOf)
    }
    expect(trajectoryFor(id).state).toBe('plateaued')
  })

  it('reports unknown for a single-observation series', () => {
    const id = addDependency('Solo')
    addThreshold(id, 'm', 10, 'below_is_better')
    addObs(id, 'm', 8, '2024-12')
    expect(trajectoryFor(id).state).toBe('unknown')
  })

  it('computes progress against both bars for a contested threshold', () => {
    const id = addDependency('Contested')
    addThreshold(id, 'm', 100, 'below_is_better', { contested: 1, alt: 300 })
    addObs(id, 'm', 400, '2024-12')
    addObs(id, 'm', 350, '2025-12')
    const row = db
      .prepare(
        `SELECT progress, progress_alt FROM progress
         WHERE dependency_id = ? ORDER BY as_of DESC LIMIT 1`,
      )
      .get(id) as { progress: number; progress_alt: number }
    expect(row.progress).not.toBeNull()
    expect(row.progress_alt).not.toBeNull()
    // The looser alt bar (300) reads as more progress than the strict bar (100).
    expect(row.progress_alt).toBeGreaterThan(row.progress)
  })

  it('answers crossing directly when the baseline already sits past the bar (battery today)', () => {
    // Earliest observation (97) is already below the 100 bar — the inverted-baseline case.
    const id = addDependency('Battery')
    addThreshold(id, 'm', 100, 'below_is_better')
    addObs(id, 'm', 97, '2024-12')
    addObs(id, 'm', 108, '2025-12')
    // Crossing is a direct test: 108 > 100 => not crossed, regardless of the bad baseline.
    expect(trajectoryFor(id).currently_crossed).toBe(0)
    // Progress is null with a reason rather than an inverted number.
    const progress = latestProgressFor(id)
    expect(progress.progress).toBeNull()
    expect(progress.progress_status).toBe('baseline_past_threshold')
  })

  it('nulls progress (not divide-by-zero) when baseline equals the bar (interconnection today)', () => {
    const id = addDependency('Interconnect')
    addThreshold(id, 'm', 2, 'below_is_better')
    addObs(id, 'm', 2, '2007-12')
    addObs(id, 'm', 5, '2025-12')
    const progress = latestProgressFor(id)
    expect(progress.progress).toBeNull()
    expect(progress.progress_status).toBe('baseline_equals_threshold')
    const row = trajectoryFor(id)
    // Direct crossing still answers: it was at the bar in 2007, no longer is.
    expect(row.currently_crossed).toBe(0)
    expect(row.became_viable_date).toBe('2007-12')
  })

  it('crosses an above_is_better series when the value rises past the bar', () => {
    const id = addDependency('CarbonLike')
    addThreshold(id, 'm', 50, 'above_is_better')
    addObs(id, 'm', 40, '2024-12')
    addObs(id, 'm', 60, '2025-12')
    const row = trajectoryFor(id)
    expect(row.currently_crossed).toBe(1)
    expect(row.became_viable_date).toBe('2025-12')
  })
})
