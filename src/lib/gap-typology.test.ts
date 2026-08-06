import { DatabaseSync } from 'node:sqlite'
import { beforeEach, describe, expect, it } from 'vitest'
import { createTablesSql } from './db-schema.js'

// The gap typology is a MAP, not a scoring engine. These tests pin the three things that make
// it one rather than the other:
//   - it never removes a company (P1: not a gate, not a filter)
//   - it never labels an unsearched region as searched-and-empty (the cardinal rule)
//   - every classified cell carries the reasoning it was produced from (P4)
// plus one case per cell, because most of them are unreachable from the committed data yet.

let db: DatabaseSync

const addCompany = (name: string, yearDefunct: number | null, ideaSpace?: string): number =>
  Number(
    db
      .prepare(
        'INSERT INTO companies (company_name, year_defunct, idea_space_id) VALUES (?, ?, (SELECT id FROM idea_spaces WHERE name = ?))',
      )
      .run(name, yearDefunct, ideaSpace ?? null).lastInsertRowid,
  )

const addDependency = (name: string, thresholdKind?: string): number =>
  Number(
    db
      .prepare('INSERT INTO dependencies (name, threshold_kind) VALUES (?, ?)')
      .run(name, thresholdKind ?? null).lastInsertRowid,
  )

const blockedOn = (companyId: number, dependencyId: number | null, raw = 'blocker'): void => {
  db.prepare(
    `INSERT INTO company_dependencies
      (company_id, dependency_id, dependency_name_raw, resolution_status, criticality)
     VALUES (?, ?, ?, ?, 'was_blocking')`,
  ).run(companyId, dependencyId, raw, dependencyId === null ? 'unresolved_name' : 'resolved')
}

// A dependency with a curve: entity + link + bar + points.
const withCurve = (
  dependencyName: string,
  options: {
    bar: number
    direction?: string
    points: [number, string][]
    policyDependent?: number
  },
): void => {
  const entityId = Number(
    db
      .prepare('INSERT INTO reference_entities (name, kind) VALUES (?, ?)')
      .run(`${dependencyName} (entity)`, 'technology').lastInsertRowid,
  )
  db.prepare('INSERT INTO dependency_links (dependency_name, entity_id, era) VALUES (?, ?, ?)').run(
    dependencyName,
    entityId,
    '2010',
  )
  db.prepare(
    `INSERT INTO dependency_thresholds
      (dependency_id, metric, scope, threshold_value, threshold_direction, threshold_source_url,
       threshold_as_of, policy_dependent)
     VALUES ((SELECT id FROM dependencies WHERE name = ?), 'm', 'global', ?, ?, ?, '2020-01', ?)`,
  ).run(
    dependencyName,
    options.bar,
    options.direction ?? 'below_is_better',
    'https://example.com/bar',
    options.policyDependent ?? 0,
  )
  const insertObs = db.prepare(
    `INSERT INTO metric_observations (entity_id, metric, value, unit, as_of, scope, method)
     VALUES (?, 'm', ?, 'u', ?, 'global', 'curated')`,
  )
  for (const [value, asOf] of options.points) {
    insertObs.run(entityId, value, asOf)
  }
}

const cellFor = (companyName: string): string =>
  (
    db.prepare('SELECT cell FROM gap_cells WHERE company_name = ?').get(companyName) as {
      cell: string
    }
  ).cell

beforeEach(() => {
  db = new DatabaseSync(':memory:')
  db.exec('PRAGMA foreign_keys = ON')
  db.exec(createTablesSql())
  db.prepare(
    'INSERT INTO trajectory_config (window_n, plateau_slope_threshold) VALUES (3, 0.03)',
  ).run()
})

describe('classified cells', () => {
  // The finding the whole dataset exists to surface: the company died in 2012, the thing it
  // needed became affordable in 2018, and nobody has revisited the idea since.
  it('classifies a company that died before its blocker crossed as lazarus_candidate', () => {
    const dep = addDependency('Cheap modules')
    withCurve('Cheap modules', {
      bar: 50,
      points: [
        [100, '2010-12'],
        [80, '2014-12'],
        [40, '2018-12'],
        [30, '2024-12'],
      ],
    })
    blockedOn(addCompany('Died Early', 2012), dep)
    expect(cellFor('Died Early')).toBe('lazarus_candidate')
  })

  it('does not call it lazarus when the blocker crossed before the company died', () => {
    const dep = addDependency('Cheap modules')
    withCurve('Cheap modules', {
      bar: 50,
      points: [
        [100, '2010-12'],
        [40, '2014-12'],
        [30, '2024-12'],
      ],
    })
    // It had what it needed and still failed — a different story entirely.
    blockedOn(addCompany('Failed Anyway', 2020), dep)
    expect(cellFor('Failed Anyway')).toBe('still_viable')
  })

  it('classifies a live company on a crossed blocker as still_viable', () => {
    const dep = addDependency('Cheap modules')
    withCurve('Cheap modules', {
      bar: 50,
      points: [
        [100, '2010-12'],
        [30, '2024-12'],
      ],
    })
    blockedOn(addCompany('Still Going', null), dep)
    expect(cellFor('Still Going')).toBe('still_viable')
  })

  it('classifies a blocker that never crossed as tested_dead_end', () => {
    const dep = addDependency('Fusion cost')
    withCurve('Fusion cost', {
      bar: 10,
      points: [
        [100, '2010-12'],
        [90, '2018-12'],
        [85, '2024-12'],
      ],
    })
    blockedOn(addCompany('Too Hard', 2015), dep)
    expect(cellFor('Too Hard')).toBe('tested_dead_end')
  })

  // Crossed, then backslid: the window opened AND closed. Reporting this as
  // lazarus_candidate would tell the user to revive something that is no longer viable.
  it('classifies a crossed-then-backslid blocker as crossed_and_receded', () => {
    const dep = addDependency('Shipping cost')
    withCurve('Shipping cost', {
      bar: 50,
      points: [
        [100, '2010-12'],
        [40, '2016-12'],
        [45, '2020-12'],
        [120, '2024-12'],
      ],
    })
    blockedOn(addCompany('Window Closed', 2012), dep)
    expect(cellFor('Window Closed')).toBe('crossed_and_receded')
  })

  // Checked before the economics-only arms: for a policy-dependent blocker below its bar,
  // "improving" answers the wrong question. The regime decides, not the trend.
  it('classifies a policy_dependent blocker short of its bar as conditional', () => {
    const dep = addDependency('Hydrogen cost')
    withCurve('Hydrogen cost', {
      bar: 2,
      policyDependent: 1,
      points: [
        [8, '2018-12'],
        [6, '2022-12'],
        [4.5, '2024-12'],
      ],
    })
    blockedOn(addCompany('Needs Subsidy', 2019), dep)
    expect(cellFor('Needs Subsidy')).toBe('conditional')
  })
})

describe('the blank cells, which are the point', () => {
  it('classifies a blocker with no reference_entity link as qualitative_blocker', () => {
    const dep = addDependency('Public trust in AVs', 'qualitative')
    blockedOn(addCompany('Robotaxi Co', 2018), dep)
    expect(cellFor('Robotaxi Co')).toBe('qualitative_blocker')
  })

  it('classifies a linked blocker with an empty curve as no_blocker_data', () => {
    const dep = addDependency('Permitting timelines', 'quantitative_with_threshold')
    // A bar, a link, and no observations at all.
    const entityId = Number(
      db
        .prepare('INSERT INTO reference_entities (name, kind) VALUES (?, ?)')
        .run('Permitting', 'policy').lastInsertRowid,
    )
    db.prepare('INSERT INTO dependency_links (dependency_name, entity_id) VALUES (?, ?)').run(
      'Permitting timelines',
      entityId,
    )
    blockedOn(addCompany('Slow Permits Co', 2016), dep)
    expect(cellFor('Slow Permits Co')).toBe('no_blocker_data')
  })

  // A curation gap, not a dead end — and emphatically not "not viable".
  it('classifies a series with no bar as unassessed, never tested_dead_end', () => {
    const dep = addDependency('Wind installed cost')
    const entityId = Number(
      db
        .prepare('INSERT INTO reference_entities (name, kind) VALUES (?, ?)')
        .run('Onshore wind', 'technology').lastInsertRowid,
    )
    db.prepare('INSERT INTO dependency_links (dependency_name, entity_id) VALUES (?, ?)').run(
      'Wind installed cost',
      entityId,
    )
    db.prepare(
      `INSERT INTO metric_observations (entity_id, metric, value, unit, as_of, scope, method)
       VALUES (?, 'm', 1200, 'u', '2024-12', 'global', 'curated')`,
    ).run(entityId)
    blockedOn(addCompany('Unjudged Co', 2016), dep)
    expect(cellFor('Unjudged Co')).toBe('unassessed')
  })

  it('keeps an unresolved blocker on the map as no_blocker_data', () => {
    blockedOn(addCompany('Odd Blocker Co', 2016), null, 'municipal appetite for pilots')
    expect(cellFor('Odd Blocker Co')).toBe('no_blocker_data')
  })

  // The P2 regression that hides inside the payoff view: built from the pairings rather than
  // from companies, a company with no recorded blockers would vanish from the map entirely.
  it('keeps a company with no recorded blockers at all on the map', () => {
    addCompany('No Blockers Recorded', 2014)
    expect(cellFor('No Blockers Recorded')).toBe('no_blocker_data')
  })
})

describe('unsampled vs white_space — the cardinal rule', () => {
  beforeEach(() => {
    db.prepare("INSERT INTO sectors (name) VALUES ('Energy')").run()
    db.prepare(
      "INSERT INTO idea_spaces (name, sector_id) VALUES ('Tidal power', (SELECT id FROM sectors WHERE name = 'Energy'))",
    ).run()
    addDependency('Turbine durability')
  })

  it('labels an empty region unsampled by default', () => {
    const rows = db.prepare('SELECT cell FROM space_cells').all()
    expect(rows).toEqual([{ cell: 'unsampled' }])
  })

  it('labels it white_space only once search_coverage says someone looked', () => {
    db.prepare(
      `INSERT INTO search_coverage (idea_space_name, dependency_name, searched_on, source, method)
       VALUES ('Tidal power', 'Turbine durability', '2026-01', 'techcrunch-2015-2024', 'curated')`,
    ).run()
    expect(db.prepare('SELECT cell FROM space_cells').all()).toEqual([{ cell: 'white_space' }])
  })

  it('accepts a whole-idea-space sweep as covering every blocker in it', () => {
    addDependency('Grid connection')
    db.prepare(
      `INSERT INTO search_coverage (idea_space_name, dependency_name, searched_on, source, method)
       VALUES ('Tidal power', NULL, '2026-01', 'full sweep', 'curated')`,
    ).run()
    const rows = db.prepare('SELECT cell FROM space_cells').all()
    expect(rows).toEqual([{ cell: 'white_space' }, { cell: 'white_space' }])
  })

  it('never labels a region white_space on the strength of a different idea space', () => {
    db.prepare(
      "INSERT INTO idea_spaces (name, sector_id) VALUES ('Wave power', (SELECT id FROM sectors WHERE name = 'Energy'))",
    ).run()
    db.prepare(
      `INSERT INTO search_coverage (idea_space_name, dependency_name, searched_on, source, method)
       VALUES ('Tidal power', NULL, '2026-01', 'full sweep', 'curated')`,
    ).run()
    const rows = db
      .prepare('SELECT idea_space_name, cell FROM space_cells ORDER BY idea_space_name')
      .all()
    expect(rows).toEqual([
      { idea_space_name: 'Tidal power', cell: 'white_space' },
      { idea_space_name: 'Wave power', cell: 'unsampled' },
    ])
  })
})

describe('the map is not a gate', () => {
  it('never drops a company from gap_map', () => {
    addCompany('A', 2010)
    const withBlocker = addCompany('B', 2012)
    blockedOn(withBlocker, addDependency('Something'))
    addCompany('C', null)

    const companies = db.prepare('SELECT COUNT(*) AS n FROM companies').get() as { n: number }
    const mapped = db
      .prepare('SELECT COUNT(DISTINCT company_id) AS n FROM gap_map WHERE company_id IS NOT NULL')
      .get() as { n: number }
    expect(mapped.n).toBe(companies.n)
  })

  it('never leaves a cell null', () => {
    addCompany('A', 2010)
    expect(db.prepare('SELECT COUNT(*) AS n FROM gap_map WHERE cell IS NULL').get()).toEqual({
      n: 0,
    })
  })

  // The roll-up must not hide the parts: a company that is lazarus on one blocker and blank on
  // another is BOTH, and a headline that erased the second would be the narrowing P2 forbids.
  it('surfaces both cells for a company that is lazarus on one blocker and blank on another', () => {
    const good = addDependency('Cheap modules')
    withCurve('Cheap modules', {
      bar: 50,
      points: [
        [100, '2010-12'],
        [40, '2018-12'],
      ],
    })
    const company = addCompany('Two Blockers', 2012)
    blockedOn(company, good)
    blockedOn(company, null, 'public trust')

    const summary = db
      .prepare(
        `SELECT headline_cell, n_lazarus_candidate, n_no_blocker_data, n_cells
         FROM company_gap_summary WHERE company_name = 'Two Blockers'`,
      )
      .get()
    expect(summary).toEqual({
      headline_cell: 'lazarus_candidate',
      n_lazarus_candidate: 1,
      n_no_blocker_data: 1,
      n_cells: 2,
    })
  })

  // The spec's "a cell without its reasoning attached is a bug", made executable.
  it('attaches the full basis and bar to every classified cell', () => {
    const dep = addDependency('Cheap modules')
    withCurve('Cheap modules', {
      bar: 50,
      points: [
        [100, '2010-12'],
        [40, '2018-12'],
      ],
    })
    blockedOn(addCompany('Died Early', 2012), dep)

    const bare = db
      .prepare(
        `SELECT COUNT(*) AS n FROM gap_cells
         WHERE cell IN ('lazarus_candidate', 'tested_dead_end', 'receded', 'conditional',
                        'crossed_and_receded', 'still_viable')
           AND (basis IS NULL OR threshold_value IS NULL OR threshold_source_url IS NULL
                OR metric IS NULL OR entity_name IS NULL)`,
      )
      .get()
    expect(bare).toEqual({ n: 0 })

    // And the reasoning is actually readable, not just non-null.
    const row = db
      .prepare(
        `SELECT entity_name, metric, basis, threshold_value, threshold_source_url, era
         FROM gap_cells WHERE company_name = 'Died Early'`,
      )
      .get()
    expect(row).toEqual({
      entity_name: 'Cheap modules (entity)',
      metric: 'm',
      basis: 'na',
      threshold_value: 50,
      threshold_source_url: 'https://example.com/bar',
      era: '2010',
    })
  })
})
