import { DatabaseSync } from 'node:sqlite'
import { describe, expect, it } from 'vitest'
import { createTablesSql } from './db-schema.js'

const freshDb = (): DatabaseSync => {
  const db = new DatabaseSync(':memory:')
  db.exec('PRAGMA foreign_keys = ON')
  db.exec(createTablesSql())
  return db
}

// Post-split, metric data hangs off a reference_entity and reaches a dependency through
// dependency_links. Most derivation tests only care that the two are connected, so this makes
// the pair in one call and returns the entity id the observations key on.
const linkedEntity = (db: DatabaseSync, dependencyName: string, entityName = 'E'): number => {
  const entityId = Number(
    db
      .prepare('INSERT INTO reference_entities (name, kind) VALUES (?, ?)')
      .run(entityName, 'technology').lastInsertRowid,
  )
  db.prepare('INSERT INTO dependency_links (dependency_name, entity_id) VALUES (?, ?)').run(
    dependencyName,
    entityId,
  )
  return entityId
}

describe('createTablesSql', () => {
  it('supports the sector -> idea_space -> company hierarchy and joins', () => {
    const db = freshDb()

    const sectorId = Number(
      db.prepare('INSERT INTO sectors (name) VALUES (?)').run('Energy').lastInsertRowid,
    )
    const ideaSpaceId = Number(
      db
        .prepare('INSERT INTO idea_spaces (name, sector_id) VALUES (?, ?)')
        .run('Long-duration grid storage', sectorId).lastInsertRowid,
    )
    const companyId = Number(
      db
        .prepare('INSERT INTO companies (company_name, idea_space_id) VALUES (?, ?)')
        .run('Acme Storage', ideaSpaceId).lastInsertRowid,
    )
    db.prepare(
      'INSERT INTO company_sectors (company_id, sector_id, is_primary) VALUES (?, ?, ?)',
    ).run(companyId, sectorId, 1)

    const joined = db
      .prepare(
        `SELECT c.company_name, s.name AS sector, i.name AS idea_space
         FROM companies c
         JOIN company_sectors cs ON cs.company_id = c.id
         JOIN sectors s          ON s.id = cs.sector_id
         JOIN idea_spaces i      ON i.id = c.idea_space_id`,
      )
      .all()

    expect(joined).toEqual([
      { company_name: 'Acme Storage', sector: 'Energy', idea_space: 'Long-duration grid storage' },
    ])
    db.close()
  })

  it('enforces the challenge outcome CHECK constraint', () => {
    const db = freshDb()
    const companyId = Number(
      db.prepare('INSERT INTO companies (company_name) VALUES (?)').run('Co').lastInsertRowid,
    )

    expect(() =>
      db
        .prepare('INSERT INTO challenges (company_id, category, outcome) VALUES (?, ?, ?)')
        .run(companyId, 'Team', 'vanished'),
    ).toThrow()

    // A valid outcome inserts fine.
    expect(() =>
      db
        .prepare('INSERT INTO challenges (company_id, category, outcome) VALUES (?, ?, ?)')
        .run(companyId, 'Team', 'overcome'),
    ).not.toThrow()

    db.close()
  })

  it('stores enrichment provenance alongside the fields it explains', () => {
    const db = freshDb()
    const companyId = Number(
      db
        .prepare(
          `INSERT INTO companies (company_name, living_status, living_status_source, year_founded, year_founded_source)
           VALUES (?, ?, ?, ?, ?)`,
        )
        .run(
          'Acme Storage',
          'Defunct',
          'enrichment:crunchbase:website',
          2015,
          'enrichment:startup-failures:name',
        ).lastInsertRowid,
    )
    const row = db
      .prepare('SELECT living_status_source, year_founded_source FROM companies WHERE id = ?')
      .get(companyId)
    expect(row).toEqual({
      living_status_source: 'enrichment:crunchbase:website',
      year_founded_source: 'enrichment:startup-failures:name',
    })
    db.close()
  })

  it('enforces the foreign key from company_sectors to companies', () => {
    const db = freshDb()
    expect(() =>
      db.prepare('INSERT INTO company_sectors (company_id, sector_id) VALUES (?, ?)').run(999, 999),
    ).toThrow()
    db.close()
  })

  it('keeps threshold_kind on dependencies and enforces its CHECK', () => {
    const db = freshDb()
    const insert = db.prepare('INSERT INTO dependencies (name, threshold_kind) VALUES (?, ?)')
    expect(() => insert.run('Battery cost', 'quantitative_with_threshold')).not.toThrow()
    expect(() => insert.run('Bad', 'guess')).toThrow()
    db.close()
  })

  it('stores per-scope bars in dependency_thresholds with a direction CHECK and (dep,metric,scope) PK', () => {
    const db = freshDb()
    const depId = Number(
      db.prepare('INSERT INTO dependencies (name) VALUES (?)').run('Carbon price').lastInsertRowid,
    )
    const insert = db.prepare(
      `INSERT INTO dependency_thresholds (dependency_id, metric, scope, threshold_value, threshold_direction)
       VALUES (?, ?, ?, ?, ?)`,
    )
    // One dependency, two scopes — the whole point of the v2 table.
    insert.run(depId, 'carbon price', 'global', 50, 'above_is_better')
    insert.run(depId, 'carbon price', 'NO', 170, 'above_is_better')
    const count = db
      .prepare('SELECT COUNT(*) AS n FROM dependency_thresholds WHERE dependency_id = ?')
      .get(depId) as { n: number }
    expect(count.n).toBe(2)
    // Same (dep, metric, scope) collides on the PK; a bad direction is rejected.
    expect(() => insert.run(depId, 'carbon price', 'global', 60, 'above_is_better')).toThrow()
    expect(() => insert.run(depId, 'carbon price', 'EU', 40, 'sideways')).toThrow()
    db.close()
  })

  it('enforces the dependency_edges relation CHECK and its FKs', () => {
    const db = freshDb()
    const a = Number(
      db.prepare('INSERT INTO dependencies (name) VALUES (?)').run('A').lastInsertRowid,
    )
    const b = Number(
      db.prepare('INSERT INTO dependencies (name) VALUES (?)').run('B').lastInsertRowid,
    )
    const insert = db.prepare(
      'INSERT INTO dependency_edges (from_dependency_id, to_dependency_id, relation) VALUES (?, ?, ?)',
    )
    expect(() => insert.run(a, b, 'drives')).not.toThrow()
    expect(() => insert.run(a, b, 'causes')).toThrow()
    expect(() => insert.run(a, 999, 'drives')).toThrow()
    db.close()
  })

  // A blocker the resolver could not fold onto the canonical list is retained with a null
  // dependency_id (P2), so the schema has to make the two states impossible to confuse.
  it('accepts an unresolved company_dependency and rejects an inconsistent one', () => {
    const db = freshDb()
    const companyId = Number(
      db.prepare('INSERT INTO companies (company_name) VALUES (?)').run('Co').lastInsertRowid,
    )
    const depId = Number(
      db.prepare('INSERT INTO dependencies (name) VALUES (?)').run('Solar module cost')
        .lastInsertRowid,
    )
    const insert = db.prepare(
      `INSERT INTO company_dependencies
        (company_id, dependency_id, dependency_name_raw, resolution_status, criticality)
       VALUES (?, ?, ?, ?, ?)`,
    )

    expect(() =>
      insert.run(companyId, null, 'public trust in AVs', 'unresolved_name', 'was_blocking'),
    ).not.toThrow()
    expect(() =>
      insert.run(companyId, depId, 'cheap PV modules', 'resolved', 'contributing'),
    ).not.toThrow()

    // The status and the id cannot disagree in either direction.
    expect(() => insert.run(companyId, null, 'x', 'resolved', 'contributing')).toThrow()
    expect(() => insert.run(companyId, depId, 'y', 'unresolved_name', 'contributing')).toThrow()
    // And the status vocab itself is closed.
    expect(() => insert.run(companyId, null, 'z', 'maybe', 'contributing')).toThrow()

    db.close()
  })

  // SQLite treats NULLs in a UNIQUE index as distinct from each other, so a plain
  // UNIQUE(company_id, dependency_id, dependency_name_raw) would let identical unresolved
  // rows both insert and would quietly stop INSERT OR IGNORE deduplicating them.
  it('deduplicates identical unresolved company_dependencies despite the null id', () => {
    const db = freshDb()
    const companyId = Number(
      db.prepare('INSERT INTO companies (company_name) VALUES (?)').run('Co').lastInsertRowid,
    )
    const insert = db.prepare(
      `INSERT OR IGNORE INTO company_dependencies
        (company_id, dependency_id, dependency_name_raw, resolution_status)
       VALUES (?, ?, ?, ?)`,
    )
    insert.run(companyId, null, 'public trust in AVs', 'unresolved_name')
    insert.run(companyId, null, 'public trust in AVs', 'unresolved_name')
    insert.run(companyId, null, 'municipal permitting', 'unresolved_name')

    const { n } = db.prepare('SELECT COUNT(*) AS n FROM company_dependencies').get() as {
      n: number
    }
    expect(n).toBe(2)
    db.close()
  })

  // Baseline rule #1 in the join itself. Two observations identical but for their currency,
  // one bar: exactly one of them is comparable to it. Before the split the join matched on
  // (dependency, metric, scope) alone and BOTH would have been judged against the same bar.
  it('refuses to join an observation to a bar of a different basis', () => {
    const db = freshDb()
    const depId = Number(
      db.prepare('INSERT INTO dependencies (name) VALUES (?)').run('Solar module cost')
        .lastInsertRowid,
    )
    db.prepare(
      `INSERT INTO dependency_thresholds
        (dependency_id, metric, scope, threshold_value, threshold_direction, basis)
       VALUES (?, 'module price', 'global', 0.2, 'below_is_better', 'real_2024_usd')`,
    ).run(depId)
    const entityId = linkedEntity(db, 'Solar module cost', 'Solar PV')
    const insertObs = db.prepare(
      `INSERT INTO metric_observations (entity_id, metric, value, unit, basis, as_of, scope, method)
       VALUES (?, 'module price', ?, 'USD/W', ?, '2024-12', 'global', 'curated')`,
    )
    insertObs.run(entityId, 0.3, 'real_2024_usd')
    insertObs.run(entityId, 0.3, 'nominal_usd')

    const rows = db.prepare('SELECT basis FROM progress').all()
    expect(rows).toEqual([{ basis: 'real_2024_usd' }])
    db.close()
  })

  // energy_basis and duration are separate predicates, so each one alone is enough to stop a
  // comparison. A cost per usable kWh judged against a nameplate-kWh bar is wrong by the
  // depth-of-discharge ratio, silently.
  it('refuses to join across an energy_basis or duration difference alone', () => {
    const db = freshDb()
    const depId = Number(
      db.prepare('INSERT INTO dependencies (name) VALUES (?)').run('Utility-scale battery system')
        .lastInsertRowid,
    )
    db.prepare(
      `INSERT INTO dependency_thresholds
        (dependency_id, metric, scope, threshold_value, threshold_direction, basis, energy_basis, duration)
       VALUES (?, 'c', 'global', 150, 'below_is_better', 'real_2025_usd', 'usable', 'blended')`,
    ).run(depId)
    const entityId = linkedEntity(db, 'Utility-scale battery system', 'Li-ion BESS')
    const insertObs = db.prepare(
      `INSERT INTO metric_observations
        (entity_id, metric, value, unit, basis, energy_basis, duration, as_of, scope, method)
       VALUES (?, 'c', 140, 'USD/kWh', 'real_2025_usd', ?, ?, '2024-12', 'global', 'curated')`,
    )
    insertObs.run(entityId, 'nameplate', 'blended') // energy_basis differs
    insertObs.run(entityId, 'usable', '4h') // duration differs
    expect(db.prepare('SELECT COUNT(*) AS n FROM progress').get()).toEqual({ n: 0 })

    insertObs.run(entityId, 'usable', 'blended') // all three agree
    expect(db.prepare('SELECT COUNT(*) AS n FROM progress').get()).toEqual({ n: 1 })
    db.close()
  })

  // THE hazard of the technology/dependency split, and the reason both id columns are in the
  // trajectory partition keys. One entity's curve now serves N dependencies, so if the series
  // key loses dependency_id those N pour into a single series: one row with 3x the points, one
  // slope fitted across three different bars, one wrong crossing date, and nothing errors.
  it('keeps three dependencies on one entity as three separate series', () => {
    const db = freshDb()
    const entityId = Number(
      db
        .prepare(`INSERT INTO reference_entities (name, kind) VALUES ('Solar PV', 'technology')`)
        .run().lastInsertRowid,
    )
    // One curve: five points falling 1.00 -> 0.20.
    const insertObs = db.prepare(
      `INSERT INTO metric_observations (entity_id, metric, value, unit, as_of, scope, method)
       VALUES (?, 'module price', ?, 'USD/W', ?, 'global', 'curated')`,
    )
    const curve: [number, string][] = [
      [1.0, '2008-12'],
      [0.8, '2010-12'],
      [0.6, '2012-12'],
      [0.4, '2014-12'],
      [0.2, '2016-12'],
    ]
    for (const [value, asOf] of curve) {
      insertObs.run(entityId, value, asOf)
    }

    // Three failure cohorts, three eras, three different bars, one shared curve.
    const cohorts: [string, number][] = [
      ['Cheap modules 2008', 0.9],
      ['Cheap modules 2012', 0.5],
      ['Cheap modules 2016', 0.1],
    ]
    for (const [name, bar] of cohorts) {
      db.prepare('INSERT INTO dependencies (name) VALUES (?)').run(name)
      db.prepare(
        `INSERT INTO dependency_links (dependency_name, entity_id, era) VALUES (?, ?, ?)`,
      ).run(name, entityId, name.slice(-4))
      db.prepare(
        `INSERT INTO dependency_thresholds
          (dependency_id, metric, scope, threshold_value, threshold_direction)
         VALUES ((SELECT id FROM dependencies WHERE name = ?), 'module price', 'global', ?, 'below_is_better')`,
      ).run(name, bar)
    }
    db.prepare(
      'INSERT INTO trajectory_config (window_n, plateau_slope_threshold) VALUES (3, 0.03)',
    ).run()

    const rows = db
      .prepare(
        `SELECT d.name, t.n_obs, t.currently_crossed, t.became_viable_date
         FROM trajectory t JOIN dependencies d ON d.id = t.dependency_id
         ORDER BY d.name`,
      )
      .all()

    // Three rows of five points each — NOT one row of fifteen.
    expect(rows).toEqual([
      // Bar 0.9: crossed as soon as the curve started, at the very first point.
      { name: 'Cheap modules 2008', n_obs: 5, currently_crossed: 1, became_viable_date: '2010-12' },
      { name: 'Cheap modules 2012', n_obs: 5, currently_crossed: 1, became_viable_date: '2014-12' },
      // Bar 0.1: the curve never got there.
      { name: 'Cheap modules 2016', n_obs: 5, currently_crossed: 0, became_viable_date: null },
    ])
    db.close()
  })

  // The core claim of the split: a technology is valid on its own. Before it, metric data
  // resolved through dependencies.name, so a technology with no dependency row dropped every
  // observation it had — 120 rows across five technologies.
  it('loads a technology with zero dependencies and still keeps its data queryable', () => {
    const db = freshDb()
    const entityId = Number(
      db
        .prepare(`INSERT INTO reference_entities (name, kind) VALUES ('Geothermal', 'technology')`)
        .run().lastInsertRowid,
    )
    db.prepare(
      `INSERT INTO metric_observations (entity_id, metric, value, unit, as_of, scope, method)
       VALUES (?, 'LCOE', 80, 'USD/MWh', '2024-12', 'global', 'curated')`,
    ).run(entityId)
    db.prepare(
      'INSERT INTO trajectory_config (window_n, plateau_slope_threshold) VALUES (3, 0.03)',
    ).run()

    // No dependency, so no bar and nothing to derive against.
    expect(db.prepare('SELECT COUNT(*) AS n FROM progress').get()).toEqual({ n: 0 })
    expect(db.prepare('SELECT COUNT(*) AS n FROM trajectory').get()).toEqual({ n: 0 })
    // But the observation is loaded and queryable, which is the entire point.
    const row = db
      .prepare(
        `SELECT e.name, o.value FROM metric_observations o
         JOIN reference_entities e ON e.id = o.entity_id`,
      )
      .get()
    expect(row).toEqual({ name: 'Geothermal', value: 80 })
    db.close()
  })

  it('rejects a duplicate dependency name now that the link FKs to it', () => {
    const db = freshDb()
    db.prepare('INSERT INTO dependencies (name) VALUES (?)').run('Solar module cost')
    expect(() =>
      db.prepare('INSERT INTO dependencies (name) VALUES (?)').run('Solar module cost'),
    ).toThrow()
    db.close()
  })

  it('enforces the reference_entities kind vocab and the link FKs', () => {
    const db = freshDb()
    expect(() =>
      db.prepare('INSERT INTO reference_entities (name, kind) VALUES (?, ?)').run('X', 'gadget'),
    ).toThrow()
    // All four kinds are in the vocab now so a non-technology entity is a data change later,
    // not a migration.
    for (const kind of ['technology', 'infrastructure', 'market', 'policy']) {
      expect(() =>
        db.prepare('INSERT INTO reference_entities (name, kind) VALUES (?, ?)').run(kind, kind),
      ).not.toThrow()
    }
    db.prepare('INSERT INTO dependencies (name) VALUES (?)').run('Dep')
    expect(() =>
      db
        .prepare('INSERT INTO dependency_links (dependency_name, entity_id) VALUES (?, ?)')
        .run('Dep', 9999),
    ).toThrow()
    expect(() =>
      db
        .prepare('INSERT INTO dependency_links (dependency_name, entity_id) VALUES (?, ?)')
        .run('Nonexistent dependency', 1),
    ).toThrow()
    db.close()
  })

  it('appends metric_observations under an entity FK with a method CHECK', () => {
    const db = freshDb()
    const entityId = Number(
      db
        .prepare('INSERT INTO reference_entities (name, kind) VALUES (?, ?)')
        .run('Lithium-ion battery', 'technology').lastInsertRowid,
    )
    const insert = db.prepare(
      `INSERT INTO metric_observations (entity_id, metric, value, unit, as_of, scope, method)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    )
    // Two dated rows for the same entity coexist — append-only, no UNIQUE.
    insert.run(entityId, 'battery pack price', 108, 'USD/kWh', '2025-12', 'global', 'curated')
    insert.run(entityId, 'battery pack price', 97, 'USD/kWh', '2024-12', 'global', 'curated')
    const count = db
      .prepare('SELECT COUNT(*) AS n FROM metric_observations WHERE entity_id = ?')
      .get(entityId) as { n: number }
    expect(count.n).toBe(2)
    // A bad method is rejected, and a dangling entity_id is rejected.
    expect(() => insert.run(entityId, 'x', 1, 'u', '2025-01', 'global', 'telepathy')).toThrow()
    expect(() => insert.run(9999, 'x', 1, 'u', '2025-01', 'global', 'curated')).toThrow()
    db.close()
  })

  // basis distinguishes two series that share a unit — USD/W nominal vs USD/W constant-2024
  // are different measurements, and comparing one against the other's bar answers wrong.
  it('carries basis on an observation so unit alone does not identify a series', () => {
    const db = freshDb()
    const entityId = Number(
      db
        .prepare('INSERT INTO reference_entities (name, kind) VALUES (?, ?)')
        .run('Solar PV', 'technology').lastInsertRowid,
    )
    db.prepare(
      `INSERT INTO metric_observations (entity_id, metric, value, unit, basis, as_of, scope, method)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(entityId, 'module price', 0.26, 'USD/W', 'real_2024_usd', '2024-12', 'global', 'curated')
    const row = db
      .prepare('SELECT unit, basis FROM metric_observations WHERE entity_id = ?')
      .get(entityId) as { unit: string; basis: string }
    expect(row).toEqual({ unit: 'USD/W', basis: 'real_2024_usd' })
    db.close()
  })

  it('stores a capacity_series point under an entity FK, with its scenario', () => {
    const db = freshDb()
    const depId = Number(
      db
        .prepare('INSERT INTO reference_entities (name, kind) VALUES (?, ?)')
        .run('Solar PV', 'technology').lastInsertRowid,
    )
    // 'AC' is an energy_basis here, not a currency — the CHECK now says so. Passing it
    // positionally as `basis` (which is what this test used to do) is rejected outright,
    // which is the point of splitting the column.
    const insert = db.prepare(
      `INSERT INTO capacity_series
        (entity_id, metric, value, unit, basis, energy_basis, as_of, scope, scenario, method)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    insert.run(
      depId,
      'cumulative solar PV capacity',
      1866.31,
      'GW',
      'na',
      'AC',
      '2024-12',
      'global',
      'historical',
      'curated',
    )
    const row = db
      .prepare(
        'SELECT value, scenario, basis, energy_basis FROM capacity_series WHERE entity_id = ?',
      )
      .get(depId) as { value: number; scenario: string; basis: string; energy_basis: string }
    expect(row).toEqual({
      value: 1866.31,
      scenario: 'historical',
      basis: 'na',
      energy_basis: 'AC',
    })
    expect(() =>
      insert.run(9999, 'm', 1, 'GW', 'na', 'AC', '2024-12', 'global', 'historical', 'curated'),
    ).toThrow()
    // A currency in the energy slot, or an energy denominator in the currency slot, is now a
    // constraint violation rather than a silently wrong comparison later.
    expect(() =>
      insert.run(depId, 'm', 1, 'GW', 'AC', 'na', '2024-12', 'global', 'historical', 'curated'),
    ).toThrow()
    db.close()
  })
})

// The `conditional` state is the ETS-vs-NZS signal: a dependency that does not cross on
// economics alone but does under a support regime. For a failed company that means
// "revivable, but only if the policy regime it needed now exists" — a different answer from
// "the cost curve fixed it", which is why it outranks the economics-only arms.
describe('trajectory view — conditional state', () => {
  const seed = (policyDependent: number): DatabaseSync => {
    const db = freshDb()
    db.prepare(
      'INSERT INTO trajectory_config (window_n, plateau_slope_threshold) VALUES (?, ?)',
    ).run(3, 0.03)
    const depId = Number(
      db.prepare('INSERT INTO dependencies (name) VALUES (?)').run('Green hydrogen production cost')
        .lastInsertRowid,
    )
    db.prepare(
      `INSERT INTO dependency_thresholds
        (dependency_id, metric, scope, threshold_value, threshold_direction, policy_dependent,
         baseline_value)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    ).run(depId, 'hydrogen price', 'global', 2, 'below_is_better', policyDependent, 10)
    const entityId = linkedEntity(db, 'Green hydrogen production cost', 'Green hydrogen')
    const insert = db.prepare(
      `INSERT INTO metric_observations (entity_id, metric, value, unit, as_of, scope, method)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    )
    // Improving on economics, but still well short of the bar.
    insert.run(entityId, 'hydrogen price', 8, 'USD/kg', '2022-12', 'global', 'curated')
    insert.run(entityId, 'hydrogen price', 6, 'USD/kg', '2024-12', 'global', 'curated')
    insert.run(entityId, 'hydrogen price', 4.5, 'USD/kg', '2026-06', 'global', 'curated')
    return db
  }

  const stateOf = (db: DatabaseSync): string =>
    (db.prepare('SELECT state FROM trajectory').get() as { state: string }).state

  it('resolves to conditional when policy_dependent and still short of the bar', () => {
    const db = seed(1)
    expect(stateOf(db)).toBe('conditional')
    db.close()
  })

  // Same numbers, same slope — only the flag differs. That isolates the new branch.
  it('falls through to the economics-only states when the flag is off', () => {
    const db = seed(0)
    expect(stateOf(db)).toBe('improving')
    db.close()
  })

  it('does not claim conditional once the metric is past its bar', () => {
    const db = seed(1)
    const entityId = (db.prepare('SELECT id FROM reference_entities').get() as { id: number }).id
    db.prepare(
      `INSERT INTO metric_observations (entity_id, metric, value, unit, as_of, scope, method)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    ).run(entityId, 'hydrogen price', 1.5, 'USD/kg', '2027-12', 'global', 'curated')
    expect(stateOf(db)).not.toBe('conditional')
    db.close()
  })
})
