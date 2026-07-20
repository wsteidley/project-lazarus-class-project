import { DatabaseSync } from 'node:sqlite'
import { describe, expect, it } from 'vitest'
import { createTablesSql } from './db-schema.js'

describe('createTablesSql', () => {
  it('creates tables and supports a company -> funding join', () => {
    const db = new DatabaseSync(':memory:')
    db.exec(createTablesSql())

    const company = db
      .prepare('INSERT INTO companies (uuid, company_name, sector) VALUES (?, ?, ?)')
      .run('u1', 'Acme Solar', 'Energy')
    const companyId = Number(company.lastInsertRowid)

    db.prepare(
      'INSERT INTO funding_rounds (company_id, round_name, amount, round_year) VALUES (?, ?, ?, ?)',
    ).run(companyId, 'Seed', 1_500_000, 2019)

    const joined = db
      .prepare(
        `SELECT c.company_name, f.round_name, f.amount
         FROM companies c JOIN funding_rounds f ON f.company_id = c.id`,
      )
      .all()

    expect(joined).toEqual([{ company_name: 'Acme Solar', round_name: 'Seed', amount: 1_500_000 }])
    db.close()
  })

  it('enforces the sector CHECK constraint', () => {
    const db = new DatabaseSync(':memory:')
    db.exec(createTablesSql())

    expect(() =>
      db
        .prepare('INSERT INTO companies (company_name, sector) VALUES (?, ?)')
        .run('Bad Co', 'Fusion Widgets'),
    ).toThrow()

    db.close()
  })
})
