import { DatabaseSync } from 'node:sqlite'
import { describe, expect, it } from 'vitest'
import { createTablesSql } from './db-schema.js'

const freshDb = (): DatabaseSync => {
  const db = new DatabaseSync(':memory:')
  db.exec('PRAGMA foreign_keys = ON')
  db.exec(createTablesSql())
  return db
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

  it('enforces the dependency_links relation CHECK and its FKs', () => {
    const db = freshDb()
    const a = Number(
      db.prepare('INSERT INTO dependencies (name) VALUES (?)').run('A').lastInsertRowid,
    )
    const b = Number(
      db.prepare('INSERT INTO dependencies (name) VALUES (?)').run('B').lastInsertRowid,
    )
    const insert = db.prepare(
      'INSERT INTO dependency_links (from_dependency_id, to_dependency_id, relation) VALUES (?, ?, ?)',
    )
    expect(() => insert.run(a, b, 'drives')).not.toThrow()
    expect(() => insert.run(a, b, 'causes')).toThrow()
    expect(() => insert.run(a, 999, 'drives')).toThrow()
    db.close()
  })

  it('appends metric_observations under a dependency FK with a method CHECK', () => {
    const db = freshDb()
    const depId = Number(
      db.prepare('INSERT INTO dependencies (name) VALUES (?)').run('Lithium-ion battery cost')
        .lastInsertRowid,
    )
    const insert = db.prepare(
      `INSERT INTO metric_observations (dependency_id, metric, value, unit, as_of, scope, method)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    )
    // Two dated rows for the same dependency coexist — append-only, no UNIQUE.
    insert.run(depId, 'battery pack price', 108, 'USD/kWh', '2025-12', 'global', 'curated')
    insert.run(depId, 'battery pack price', 97, 'USD/kWh', '2024-12', 'global', 'curated')
    const count = db
      .prepare('SELECT COUNT(*) AS n FROM metric_observations WHERE dependency_id = ?')
      .get(depId) as { n: number }
    expect(count.n).toBe(2)
    // A bad method is rejected, and a dangling dependency_id is rejected.
    expect(() => insert.run(depId, 'x', 1, 'u', '2025-01', 'global', 'telepathy')).toThrow()
    expect(() => insert.run(9999, 'x', 1, 'u', '2025-01', 'global', 'curated')).toThrow()
    db.close()
  })
})
