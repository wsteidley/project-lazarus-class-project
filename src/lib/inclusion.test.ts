import { DatabaseSync } from 'node:sqlite'
import { describe, expect, it } from 'vitest'
import type { CsvRow } from './csv.js'
import { createTablesSql } from './db-schema.js'
import { type LoadInputs, loadDatabase } from './load-db.js'

// The P2 guarantee, made executable: a company enters the dataset on factual footprint
// alone. Not "usually," not "unless the metric side is empty" — structurally. These tests
// exist because the failure mode is silent: rich learning-curve data for three technologies
// quietly narrowing a *company* dataset into a technologies-we-have-curves-for dataset would
// delete exactly the Lazarus candidates the project exists to surface, and nothing would
// error. See foundational-principles.md P2 and structural-fixes-spec-v1.md Fix A.

const emptyInputs = (): LoadInputs => ({
  ideaSpaces: [],
  companies: [],
  companyUrls: [],
  companySectors: [],
  fundingRounds: [],
  challenges: [],
  dependencies: [],
  thresholds: [],
  companyDependencies: [],
  assessments: [],
  metricObservations: [],
  capacitySeries: [],
  dependencyLinks: [],
  rawDocuments: [],
})

const freshDb = (): DatabaseSync => {
  const db = new DatabaseSync(':memory:')
  db.exec('PRAGMA foreign_keys = ON')
  db.exec(createTablesSql())
  return db
}

// A complete factual footprint and nothing else: what it did, roughly when, its sector and
// idea space. No curve, no bar, no viability call anywhere in the inputs.
const company = (overrides: CsvRow = {}): CsvRow => ({
  uuid: 'company-1',
  company_name: 'Solyndra Redux',
  idea_space_name: 'Thin-film solar',
  year_founded: '2008',
  year_defunct: '2013',
  living_status: 'Defunct',
  idea_summary: 'Cylindrical CIGS modules for commercial rooftops',
  source_url: 'https://example.com/solyndra-redux',
  ...overrides,
})

const ideaSpace: CsvRow = { name: 'Thin-film solar', sector_name: 'Energy' }

describe('P2 — inclusion is never gated on viability data', () => {
  it('loads a company with a full factual footprint and zero curve/threshold data', () => {
    const db = freshDb()

    const report = loadDatabase(db, {
      ...emptyInputs(),
      ideaSpaces: [ideaSpace],
      companies: [company()],
      companySectors: [{ company_uuid: 'company-1', sector_name: 'Energy', is_primary: '1' }],
    })

    // It is present, and it is queryable through the joins a user would actually reach for.
    const found = db
      .prepare(
        `SELECT c.company_name, c.year_defunct, i.name AS idea_space, s.name AS sector
         FROM companies c
         LEFT JOIN idea_spaces i     ON i.id = c.idea_space_id
         LEFT JOIN company_sectors cs ON cs.company_id = c.id
         LEFT JOIN sectors s          ON s.id = cs.sector_id`,
      )
      .all()

    expect(found).toEqual([
      {
        company_name: 'Solyndra Redux',
        year_defunct: 2013,
        idea_space: 'Thin-film solar',
        sector: 'Energy',
      },
    ])
    // Nothing was quietly discarded on the way in.
    expect(report.skipped.size).toBe(0)
    db.close()
  })

  it('never reduces the company count for any absence of metric data', () => {
    const companies = [
      company({ uuid: 'a', company_name: 'A' }),
      company({ uuid: 'b', company_name: 'B' }),
      company({ uuid: 'c', company_name: 'C' }),
    ]
    // Each case removes a different kind of data the metric side might have supplied. None of
    // them is allowed to change the answer.
    const cases: Record<string, Partial<LoadInputs>> = {
      'no observations': { thresholds: [], metricObservations: [] },
      'no dependencies at all': { dependencies: [], companyDependencies: [] },
      'no capacity curves': { capacitySeries: [] },
    }

    for (const [label, overrides] of Object.entries(cases)) {
      const db = freshDb()
      loadDatabase(db, { ...emptyInputs(), ideaSpaces: [ideaSpace], companies, ...overrides })
      const { n } = db.prepare('SELECT COUNT(*) AS n FROM companies').get() as { n: number }
      expect(n, `company count changed with ${label}`).toBe(3)
      db.close()
    }
  })
})

describe('P2/P3 — an unresolvable blocker is retained, not dropped', () => {
  it('keeps a company_dependency whose name matched nothing canonical', () => {
    const db = freshDb()

    const report = loadDatabase(db, {
      ...emptyInputs(),
      ideaSpaces: [ideaSpace],
      companies: [company()],
      dependencies: [{ name: 'Solar module cost', threshold_kind: 'quantitative_with_threshold' }],
      companyDependencies: [
        {
          company_uuid: 'company-1',
          dependency_name: 'Solar module cost',
          dependency_name_raw: 'cheap PV modules',
          criticality: 'was_blocking',
          detail: 'needed modules under $1/W',
        },
        // The case that used to vanish: a real, named blocker with no canonical home.
        {
          company_uuid: 'company-1',
          dependency_name: '',
          dependency_name_raw: 'public trust in rooftop installers',
          criticality: 'contributing',
          detail: 'homeowners would not sign',
        },
      ],
    })

    expect(report.companyDependenciesRetained).toBe(2)
    expect(report.companyDependenciesUnresolved).toBe(1)
    expect(report.skipped.size).toBe(0)

    const rows = db
      .prepare(
        `SELECT resolution_status, dependency_name_raw, dependency_id IS NULL AS unlinked
         FROM company_dependencies ORDER BY resolution_status`,
      )
      .all()

    expect(rows).toEqual([
      {
        resolution_status: 'resolved',
        dependency_name_raw: 'cheap PV modules',
        unlinked: 0,
      },
      {
        resolution_status: 'unresolved_name',
        dependency_name_raw: 'public trust in rooftop installers',
        unlinked: 1,
      },
    ])
    db.close()
  })

  // The raw name is the only handle a curator has for folding the row onto the canonical
  // list later, so a retained row must never be anonymous.
  it('falls back to the extracted detail when the resolver proposed no name at all', () => {
    const db = freshDb()

    loadDatabase(db, {
      ...emptyInputs(),
      companies: [company()],
      companyDependencies: [
        {
          company_uuid: 'company-1',
          dependency_name: '',
          dependency_name_raw: '',
          detail: 'utility interconnection queue over four years',
        },
      ],
    })

    const { dependency_name_raw } = db
      .prepare('SELECT dependency_name_raw FROM company_dependencies')
      .get() as { dependency_name_raw: string }
    expect(dependency_name_raw).toBe('utility interconnection queue over four years')
    db.close()
  })

  it('reports an unknown company_uuid by name rather than as an anonymous count', () => {
    const db = freshDb()

    const report = loadDatabase(db, {
      ...emptyInputs(),
      companies: [company()],
      companyDependencies: [{ company_uuid: 'nobody', dependency_name: '' }],
    })

    // Still skipped — an orphan FK is a genuine load failure, not an absence to record. But
    // it is named, which the old single "N child rows skipped" counter never was.
    expect([...report.skipped]).toEqual([['company_dependencies: unknown company_uuid', 1]])
    db.close()
  })
})
