import { existsSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { config } from '../config.js'
import { type CsvRow, readCsv } from '../lib/csv.js'
import { createTablesSql } from '../lib/db-schema.js'

const dbFile = process.env.DB_FILE ?? 'lazarus.db'

// Coerce CSV string cells to the types SQLite expects: '' -> null.
const toText = (value: string | undefined): string | null =>
  value === undefined || value === '' ? null : value
const toInt = (value: string | undefined): number | null => {
  if (value === undefined || value === '') {
    return null
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? Math.trunc(parsed) : null
}

// Reads a table CSV from DATA_DIR, or returns [] (with a note) if it doesn't exist
// so a partial pipeline run still builds what's available.
const readTableIfExists = async (tableFilename: string): Promise<CsvRow[]> => {
  const fullPath = join(config.dataDir, tableFilename)
  if (!existsSync(fullPath)) {
    console.log(`(skipping ${tableFilename} — not found)`)
    return []
  }
  return readCsv(fullPath)
}

const main = async (): Promise<void> => {
  const companies = await readTableIfExists('companies.csv')
  if (companies.length === 0) {
    throw new Error(`No companies.csv in ${config.dataDir}; run step1 first`)
  }
  const fundingRounds = await readTableIfExists('funding_rounds.csv')
  const failureReasons = await readTableIfExists('failure_reasons.csv')
  const ideaDependencies = await readTableIfExists('idea_dependencies.csv')

  // Fresh build every time — drop the old file so the run is idempotent.
  if (existsSync(dbFile)) {
    rmSync(dbFile)
  }
  const db = new DatabaseSync(dbFile)
  db.exec(createTablesSql())

  // companies first, capturing the assigned integer id per uuid.
  const insertCompany = db.prepare(
    `INSERT INTO companies
      (uuid, company_name, founders, sector, subsector, location, country,
       year_founded, year_defunct, living_status, has_pivoted, idea_summary,
       original_trl, is_climate, source_url, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  const idByUuid = new Map<string, number>()
  for (const row of companies) {
    const info = insertCompany.run(
      toText(row.uuid),
      toText(row.company_name) ?? '',
      toText(row.founders),
      toText(row.sector),
      toText(row.subsector),
      toText(row.location),
      toText(row.country),
      toInt(row.year_founded),
      toInt(row.year_defunct),
      toText(row.living_status),
      toInt(row.has_pivoted),
      toText(row.idea_summary),
      toInt(row.original_trl),
      toInt(row.is_climate),
      toText(row.source_url),
      toText(row.created_at),
    )
    if (row.uuid) {
      idByUuid.set(row.uuid, Number(info.lastInsertRowid))
    }
  }

  // Resolve a child row's company_uuid -> integer company_id; null if unknown.
  const companyIdFor = (row: CsvRow): number | undefined => idByUuid.get(row.company_uuid ?? '')

  let skipped = 0
  const insertFunding = db.prepare(
    `INSERT INTO funding_rounds
      (company_id, round_name, amount, currency, round_date, round_year, source_url)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  )
  for (const row of fundingRounds) {
    const companyId = companyIdFor(row)
    if (companyId === undefined) {
      skipped += 1
      continue
    }
    insertFunding.run(
      companyId,
      toText(row.round_name),
      toInt(row.amount),
      toText(row.currency),
      toText(row.round_date),
      toInt(row.round_year),
      toText(row.source_url),
    )
  }

  const insertFailure = db.prepare(
    `INSERT INTO failure_reasons (company_id, category, detail, source_url) VALUES (?, ?, ?, ?)`,
  )
  for (const row of failureReasons) {
    const companyId = companyIdFor(row)
    if (companyId === undefined) {
      skipped += 1
      continue
    }
    insertFailure.run(companyId, toText(row.category), toText(row.detail), toText(row.source_url))
  }

  const insertDependency = db.prepare(
    `INSERT INTO idea_dependencies
      (uuid, company_id, category, detail, criticality, source_url)
     VALUES (?, ?, ?, ?, ?, ?)`,
  )
  for (const row of ideaDependencies) {
    const companyId = companyIdFor(row)
    if (companyId === undefined) {
      skipped += 1
      continue
    }
    insertDependency.run(
      toText(row.uuid),
      companyId,
      toText(row.category),
      toText(row.detail),
      toText(row.criticality),
      toText(row.source_url),
    )
  }

  db.close()

  console.log(
    `Built ${dbFile}: ${companies.length} companies, ${fundingRounds.length} funding_rounds, ` +
      `${failureReasons.length} failure_reasons, ${ideaDependencies.length} idea_dependencies` +
      (skipped ? ` (${skipped} child rows skipped — unknown company_uuid)` : ''),
  )
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
