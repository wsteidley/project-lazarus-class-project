import { DatabaseSync } from 'node:sqlite'
import { beforeEach, describe, expect, it } from 'vitest'
import { createTablesSql } from './db-schema.js'

// P3: absence is a first-class, queryable state — never a bare null. These tests pin the
// five states apart from each other, and pin the two traps that would collapse them:
//   - a series with no bar must never read as "not viable" (the trajectory ELSE-0 trap)
//   - a null *progress* must never read as "undetermined" (two different null vocabularies)
// See foundational-principles.md P3 and structural-fixes-spec-v1.md Fix B.

let db: DatabaseSync

const addDependency = (name: string, thresholdKind?: string): number =>
  Number(
    db
      .prepare('INSERT INTO dependencies (name, threshold_kind) VALUES (?, ?)')
      .run(name, thresholdKind ?? null).lastInsertRowid,
  )

const addThreshold = (
  depId: number,
  metric: string,
  value: number | null,
  direction: string | null,
): void => {
  db.prepare(
    `INSERT INTO dependency_thresholds
      (dependency_id, metric, scope, threshold_value, threshold_direction)
     VALUES (?, ?, 'global', ?, ?)`,
  ).run(depId, metric, value, direction)
}

// Post-split, observations hang off an ENTITY. These tests are about states, not about the
// split, so each dependency gets its own entity and link — the shape that existed implicitly
// before, made explicit.
const entityFor = new Map<number, number>()
const entityOf = (depId: number): number => {
  const existing = entityFor.get(depId)
  if (existing !== undefined) {
    return existing
  }
  const name = (
    db.prepare('SELECT name FROM dependencies WHERE id = ?').get(depId) as { name: string }
  ).name
  const entityId = Number(
    db
      .prepare('INSERT INTO reference_entities (name, kind) VALUES (?, ?)')
      .run(`${name} (entity)`, 'technology').lastInsertRowid,
  )
  db.prepare('INSERT INTO dependency_links (dependency_name, entity_id) VALUES (?, ?)').run(
    name,
    entityId,
  )
  entityFor.set(depId, entityId)
  return entityId
}

const addObs = (depId: number, metric: string, value: number, asOf: string): void => {
  db.prepare(
    `INSERT INTO metric_observations (entity_id, metric, value, unit, as_of, scope, method)
     VALUES (?, ?, ?, 'u', ?, 'global', 'curated')`,
  ).run(entityOf(depId), metric, value, asOf)
}

const addCompanyWithBlocker = (
  companyName: string,
  depId: number | null,
  raw: string,
): { companyId: number } => {
  const companyId = Number(
    db.prepare('INSERT INTO companies (company_name) VALUES (?)').run(companyName).lastInsertRowid,
  )
  db.prepare(
    `INSERT INTO company_dependencies
      (company_id, dependency_id, dependency_name_raw, resolution_status)
     VALUES (?, ?, ?, ?)`,
  ).run(companyId, depId, raw, depId === null ? 'unresolved_name' : 'resolved')
  return { companyId }
}

const seriesStatus = (depId: number): string =>
  (
    db.prepare('SELECT status FROM series_status WHERE dependency_id = ?').get(depId) as {
      status: string
    }
  ).status

const dependencyStatus = (
  depId: number,
): { status: string; n_series: number; n_no_threshold: number; n_assessed_viable: number } =>
  db
    .prepare(
      `SELECT status, n_series, n_no_threshold, n_assessed_viable
       FROM dependency_status WHERE dependency_id = ?`,
    )
    .get(depId) as {
    status: string
    n_series: number
    n_no_threshold: number
    n_assessed_viable: number
  }

beforeEach(() => {
  db = new DatabaseSync(':memory:')
  db.exec('PRAGMA foreign_keys = ON')
  db.exec(createTablesSql())
  entityFor.clear()
  db.prepare(
    'INSERT INTO trajectory_config (window_n, plateau_slope_threshold) VALUES (3, 0.03)',
  ).run()
})

describe('series_status — the five states, kept apart', () => {
  it('names a series with no bar no_threshold, never not-viable', () => {
    const id = addDependency('Wind installed cost')
    addObs(id, 'total_installed_cost', 1800, '2015-12')
    addObs(id, 'total_installed_cost', 1200, '2024-12')

    expect(seriesStatus(id)).toBe('no_threshold')
    // The trap: trajectory's currently_crossed is CASE ... ELSE 0, so a LEFT-JOINed
    // threshold would have published this unassessed series as "assessed, not viable".
    expect(db.prepare('SELECT COUNT(*) AS n FROM trajectory').get()).toEqual({ n: 0 })
  })

  it('names a bar with no direction undetermined', () => {
    const id = addDependency('Half-specified')
    addThreshold(id, 'm', 100, null)
    addObs(id, 'm', 120, '2024-12')
    expect(seriesStatus(id)).toBe('undetermined')
  })

  it('names a quantitative_tbd bar with no value no_threshold, not undetermined', () => {
    const id = addDependency('Measurable, unjudged', 'quantitative_tbd')
    addThreshold(id, 'm', null, 'below_is_better')
    addObs(id, 'm', 120, '2024-12')
    // A bar row exists but carries no number: still "no bar has been set".
    expect(seriesStatus(id)).toBe('no_threshold')
  })

  it('assesses a crossed and an uncrossed series in both directions', () => {
    const below = addDependency('Battery pack price')
    addThreshold(below, 'm', 100, 'below_is_better')
    addObs(below, 'm', 90, '2024-12')
    expect(seriesStatus(below)).toBe('assessed_viable')

    const above = addDependency('Carbon price')
    addThreshold(above, 'm', 50, 'above_is_better')
    addObs(above, 'm', 30, '2024-12')
    expect(seriesStatus(above)).toBe('assessed_not_viable')
  })

  it('judges on the latest observation, not the first', () => {
    const id = addDependency('Recently crossed')
    addThreshold(id, 'm', 100, 'below_is_better')
    addObs(id, 'm', 200, '2015-12')
    addObs(id, 'm', 90, '2024-12')
    expect(seriesStatus(id)).toBe('assessed_viable')
  })

  // The second trap. These are two different null vocabularies and conflating them would
  // report a real verdict as a gap.
  it('does not call a null progress undetermined', () => {
    const id = addDependency('Grid interconnection')
    addThreshold(id, 'm', 100, 'above_is_better')
    // baseline == threshold: progress is null with a stated reason, but the crossing test
    // is a direct scalar comparison and answers perfectly well.
    addObs(id, 'm', 100, '2015-12')
    addObs(id, 'm', 80, '2024-12')

    const progressRow = db
      .prepare('SELECT progress, progress_status FROM progress ORDER BY as_of DESC LIMIT 1')
      .get() as { progress: number | null; progress_status: string }
    expect(progressRow.progress).toBeNull()
    expect(progressRow.progress_status).toBe('baseline_equals_threshold')

    expect(seriesStatus(id)).toBe('assessed_not_viable')
  })

  it('keeps segments as separate series with their own answers', () => {
    const id = addDependency('Battery pack price')
    addThreshold(id, 'm', 100, 'below_is_better')
    db.prepare(
      `INSERT INTO metric_observations (entity_id, metric, value, unit, segment, as_of, scope, method)
       VALUES (?, 'm', 90, 'u', 'bev', '2024-12', 'global', 'curated')`,
    ).run(entityOf(id))
    db.prepare(
      `INSERT INTO metric_observations (entity_id, metric, value, unit, segment, as_of, scope, method)
       VALUES (?, 'm', 150, 'u', 'stationary', '2024-12', 'global', 'curated')`,
    ).run(entityOf(id))

    const rows = db.prepare('SELECT segment, status FROM series_status ORDER BY segment').all() as {
      segment: string
      status: string
    }[]
    expect(rows).toEqual([
      { segment: 'bev', status: 'assessed_viable' },
      { segment: 'stationary', status: 'assessed_not_viable' },
    ])
  })
})

// The load-bearing invariant. This is what stops a future edit from LEFT-JOINing the
// threshold into `progress` "to be helpful" — which would push no-bar series into
// trajectory, where currently_crossed's CASE ... ELSE 0 would publish them as not-crossed.
//
// The relation is a subset, not an equality. progress admits any series with a non-null
// threshold_value, so a bar carrying a value but no direction lands there with a NULL
// progress column while series_status correctly calls it `undetermined` — the one state
// that legitimately straddles the two. What must never straddle is `no_threshold`.
describe('progress vs series_status', () => {
  beforeEach(() => {
    const barred = addDependency('Barred')
    addThreshold(barred, 'm', 100, 'below_is_better')
    addObs(barred, 'm', 90, '2024-12')

    const unbarred = addDependency('Unbarred')
    addObs(unbarred, 'm', 90, '2024-12')

    const noDirection = addDependency('No direction')
    addThreshold(noDirection, 'm', 100, null)
    addObs(noDirection, 'm', 90, '2024-12')
  })

  it('never lets a no_threshold series reach progress', () => {
    const leaked = db
      .prepare(
        `SELECT COUNT(*) AS n
         FROM (SELECT DISTINCT dependency_id, metric, segment, scope FROM progress) p
         JOIN series_status s
           ON s.dependency_id = p.dependency_id AND s.metric = p.metric
          AND s.segment = p.segment AND s.scope = p.scope
         WHERE s.status = 'no_threshold'`,
      )
      .get() as { n: number }
    expect(leaked.n).toBe(0)
  })

  it('keeps every progress series inside the three judged-or-attempted states', () => {
    const orphan = db
      .prepare(
        `SELECT COUNT(*) AS n
         FROM (SELECT DISTINCT dependency_id, metric, segment, scope FROM progress) p
         WHERE NOT EXISTS (
           SELECT 1 FROM series_status s
           WHERE s.dependency_id = p.dependency_id AND s.metric = p.metric
             AND s.segment = p.segment AND s.scope = p.scope
             AND s.status IN ('assessed_viable', 'assessed_not_viable', 'undetermined'))`,
      )
      .get() as { n: number }
    expect(orphan.n).toBe(0)
  })

  it('gives every assessed series progress rows to plot', () => {
    const missing = db
      .prepare(
        `SELECT COUNT(*) AS n FROM series_status s
         WHERE s.status IN ('assessed_viable', 'assessed_not_viable')
           AND NOT EXISTS (
             SELECT 1 FROM progress p
             WHERE p.dependency_id = s.dependency_id AND p.metric = s.metric
               AND p.segment = s.segment AND p.scope = s.scope)`,
      )
      .get() as { n: number }
    expect(missing.n).toBe(0)
  })
})

describe('dependency_status — the roll-up never hides the parts', () => {
  it('names a dependency with no series at all no_blocker_data, not a null', () => {
    const id = addDependency('Permitting timelines')
    addThreshold(id, 'm', 100, 'below_is_better') // a bar, but nothing measured against it
    expect(dependencyStatus(id)).toMatchObject({ status: 'no_blocker_data', n_series: 0 })
  })

  it('names a non-measurable dependency qualitative_blocker rather than a gap', () => {
    const id = addDependency('Public trust in AVs', 'qualitative')
    expect(dependencyStatus(id).status).toBe('qualitative_blocker')
  })

  it('counts every state, not just the winning one', () => {
    const id = addDependency('Mixed')
    addThreshold(id, 'judged', 100, 'below_is_better')
    addObs(id, 'judged', 90, '2024-12')
    addObs(id, 'unjudged', 5, '2024-12') // second metric, no bar

    const row = dependencyStatus(id)
    expect(row.status).toBe('assessed_viable')
    // The no_threshold series is still visible underneath the headline.
    expect(row).toMatchObject({ n_series: 2, n_assessed_viable: 1, n_no_threshold: 1 })
  })
})

describe('company_dependency_status — the P3 query surface', () => {
  it('keeps an unresolved blocker queryable as no_blocker_data', () => {
    addCompanyWithBlocker('Autonomy Inc', null, 'public trust in AVs')

    const rows = db
      .prepare(
        `SELECT company_name, dependency_label, resolution_status, status
         FROM company_dependency_status WHERE status = 'no_blocker_data'`,
      )
      .all()

    expect(rows).toEqual([
      {
        company_name: 'Autonomy Inc',
        dependency_label: 'public trust in AVs',
        resolution_status: 'unresolved_name',
        status: 'no_blocker_data',
      },
    ])
  })

  it('gives a resolved blocker its dependency-level state', () => {
    const dep = addDependency('Battery pack price')
    addThreshold(dep, 'm', 100, 'below_is_better')
    addObs(dep, 'm', 90, '2024-12')
    addCompanyWithBlocker('Storage Co', dep, 'cheap cells')

    const row = db
      .prepare('SELECT dependency_label, status FROM company_dependency_status')
      .get() as { dependency_label: string; status: string }
    expect(row).toEqual({ dependency_label: 'Battery pack price', status: 'assessed_viable' })
  })

  it('surfaces both blockers for a company that is judged on one and blank on the other', () => {
    const dep = addDependency('Battery pack price')
    addThreshold(dep, 'm', 100, 'below_is_better')
    addObs(dep, 'm', 90, '2024-12')
    const { companyId } = addCompanyWithBlocker('Two Blockers Co', dep, 'cheap cells')
    db.prepare(
      `INSERT INTO company_dependencies
        (company_id, dependency_id, dependency_name_raw, resolution_status)
       VALUES (?, NULL, 'grid operator willingness', 'unresolved_name')`,
    ).run(companyId)

    const rows = db.prepare('SELECT status FROM company_dependency_status ORDER BY status').all()
    expect(rows).toEqual([{ status: 'assessed_viable' }, { status: 'no_blocker_data' }])
  })
})
