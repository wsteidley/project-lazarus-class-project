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

  it('enforces the foreign key from company_sectors to companies', () => {
    const db = freshDb()
    expect(() =>
      db.prepare('INSERT INTO company_sectors (company_id, sector_id) VALUES (?, ?)').run(999, 999),
    ).toThrow()
    db.close()
  })
})
