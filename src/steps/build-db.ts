import { existsSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { type CsvRow, readCsv } from '../lib/csv.js'
import { createTablesSql } from '../lib/db-schema.js'
import { inputFile, latestRunDir } from '../lib/paths.js'
import { SECTOR } from '../schemas.js'

const toText = (value: string | undefined): string | null =>
  value === undefined || value === '' ? null : value
const toInt = (value: string | undefined): number | null => {
  if (value === undefined || value === '') {
    return null
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? Math.trunc(parsed) : null
}

const readCsvIfExists = async (fullPath: string): Promise<CsvRow[]> => {
  if (!existsSync(fullPath)) {
    console.log(`(skipping ${fullPath} — not found)`)
    return []
  }
  return readCsv(fullPath)
}

const main = async (): Promise<void> => {
  const runDir = latestRunDir()
  // DB_FILE overrides; otherwise the DB lives inside the run folder.
  const dbFile = process.env.DB_FILE || join(runDir, 'lazarus.db')

  const companies = await readCsvIfExists(join(runDir, 'companies.csv'))
  if (companies.length === 0) {
    throw new Error(`No companies.csv in ${runDir}; run step1 first`)
  }
  const ideaSpaces = await readCsvIfExists(inputFile('idea_spaces.csv'))
  const companySectors = await readCsvIfExists(join(runDir, 'company_sectors.csv'))
  const fundingRounds = await readCsvIfExists(join(runDir, 'funding_rounds.csv'))
  const challenges = await readCsvIfExists(join(runDir, 'challenges.csv'))
  const ideaDependencies = await readCsvIfExists(join(runDir, 'idea_dependencies.csv'))

  if (existsSync(dbFile)) {
    rmSync(dbFile)
  }
  const db = new DatabaseSync(dbFile)
  db.exec('PRAGMA foreign_keys = ON')
  db.exec(createTablesSql())

  // sectors: reference rows seeded from the vocab. name -> id.
  const insertSector = db.prepare('INSERT INTO sectors (name) VALUES (?)')
  const sectorIdByName = new Map<string, number>()
  for (const name of SECTOR) {
    const info = insertSector.run(name)
    sectorIdByName.set(name, Number(info.lastInsertRowid))
  }

  // idea_spaces: curated seed. Resolve home sector by name. name -> id.
  const insertIdeaSpace = db.prepare(
    'INSERT INTO idea_spaces (name, sector_id, description) VALUES (?, ?, ?)',
  )
  const ideaSpaceIdByName = new Map<string, number>()
  for (const row of ideaSpaces) {
    const info = insertIdeaSpace.run(
      toText(row.name) ?? '',
      sectorIdByName.get(row.sector_name ?? '') ?? null,
      toText(row.description),
    )
    if (row.name) {
      ideaSpaceIdByName.set(row.name, Number(info.lastInsertRowid))
    }
  }

  // companies: resolve idea_space_name -> id; capture uuid -> id.
  const insertCompany = db.prepare(
    `INSERT INTO companies
      (uuid, company_name, idea_space_id, founders, location, country, year_founded,
       year_defunct, living_status, has_pivoted, idea_summary, exit_type, exit_amount,
       exit_date, exit_notes, outcome_summary, outcome_source_url, outcome_type,
       outcome_rationale, original_trl, is_climate, source_url, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  const idByUuid = new Map<string, number>()
  for (const row of companies) {
    const info = insertCompany.run(
      toText(row.uuid),
      toText(row.company_name) ?? '',
      ideaSpaceIdByName.get(row.idea_space_name ?? '') ?? null,
      toText(row.founders),
      toText(row.location),
      toText(row.country),
      toInt(row.year_founded),
      toInt(row.year_defunct),
      toText(row.living_status),
      toInt(row.has_pivoted),
      toText(row.idea_summary),
      toText(row.exit_type),
      toInt(row.exit_amount),
      toText(row.exit_date),
      toText(row.exit_notes),
      toText(row.outcome_summary),
      toText(row.outcome_source_url),
      toText(row.outcome_type),
      toText(row.outcome_rationale),
      toInt(row.original_trl),
      toInt(row.is_climate),
      toText(row.source_url),
      toText(row.created_at),
    )
    if (row.uuid) {
      idByUuid.set(row.uuid, Number(info.lastInsertRowid))
    }
  }

  const companyIdFor = (row: CsvRow): number | undefined => idByUuid.get(row.company_uuid ?? '')
  let skipped = 0

  // company_sectors: resolve company_uuid + sector_name.
  const insertCompanySector = db.prepare(
    'INSERT OR IGNORE INTO company_sectors (company_id, sector_id, is_primary) VALUES (?, ?, ?)',
  )
  for (const row of companySectors) {
    const companyId = companyIdFor(row)
    const sectorId = sectorIdByName.get(row.sector_name ?? '')
    if (companyId === undefined || sectorId === undefined) {
      skipped += 1
      continue
    }
    insertCompanySector.run(companyId, sectorId, toInt(row.is_primary))
  }

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

  const insertChallenge = db.prepare(
    'INSERT INTO challenges (company_id, category, outcome, detail, source_url) VALUES (?, ?, ?, ?, ?)',
  )
  for (const row of challenges) {
    const companyId = companyIdFor(row)
    if (companyId === undefined) {
      skipped += 1
      continue
    }
    insertChallenge.run(
      companyId,
      toText(row.category),
      toText(row.outcome),
      toText(row.detail),
      toText(row.source_url),
    )
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
    `Built ${dbFile}: ${SECTOR.length} sectors, ${ideaSpaces.length} idea_spaces, ` +
      `${companies.length} companies, ${companySectors.length} company_sectors, ` +
      `${fundingRounds.length} funding_rounds, ${challenges.length} challenges, ` +
      `${ideaDependencies.length} idea_dependencies` +
      (skipped ? ` (${skipped} child rows skipped — unresolved FK)` : ''),
  )
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
