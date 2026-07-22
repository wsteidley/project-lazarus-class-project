import { existsSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { type CsvRow, readCsv } from '../lib/csv.js'
import { createTablesSql } from '../lib/db-schema.js'
import { inputFile, latestRunDir } from '../lib/paths.js'
import { readAllCachedDocuments } from '../lib/raw-documents.js'
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
// Like toInt but keeps the fractional part, for metric/threshold/score columns.
const toReal = (value: string | undefined): number | null => {
  if (value === undefined || value === '') {
    return null
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
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
  const companyUrls = await readCsvIfExists(join(runDir, 'company_urls.csv'))
  const companySectors = await readCsvIfExists(join(runDir, 'company_sectors.csv'))
  const fundingRounds = await readCsvIfExists(join(runDir, 'funding_rounds.csv'))
  const challenges = await readCsvIfExists(join(runDir, 'challenges.csv'))
  // The canonical dependency dimension is a curated input, not a run output.
  const dependencies = await readCsvIfExists(inputFile('dependencies.csv'))
  const companyDependencies = await readCsvIfExists(join(runDir, 'company_dependencies.csv'))
  const assessments = await readCsvIfExists(join(runDir, 'dependency_assessments.csv'))
  const rawDocuments = await readAllCachedDocuments()

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
      (uuid, company_name, idea_space_id, founders, canonical_uuid, merged_from,
       location, country, year_founded, year_defunct, living_status_source,
       year_founded_source, living_status, has_pivoted,
       idea_summary, exit_type, exit_amount, exit_date, exit_notes, outcome_summary,
       outcome_source_url, outcome_confidence, outcome_confidence_score,
       outcome_confidence_self_reported, outcome_contested,
       outcome_contested_note, outcome_type, outcome_rationale, original_trl, is_climate,
       source_url, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  const idByUuid = new Map<string, number>()
  for (const row of companies) {
    const info = insertCompany.run(
      toText(row.uuid),
      toText(row.company_name) ?? '',
      ideaSpaceIdByName.get(row.idea_space_name ?? '') ?? null,
      toText(row.founders),
      toText(row.canonical_uuid),
      toText(row.merged_from),
      toText(row.location),
      toText(row.country),
      toInt(row.year_founded),
      toInt(row.year_defunct),
      toText(row.living_status_source),
      toText(row.year_founded_source),
      toText(row.living_status),
      toInt(row.has_pivoted),
      toText(row.idea_summary),
      toText(row.exit_type),
      toInt(row.exit_amount),
      toText(row.exit_date),
      toText(row.exit_notes),
      toText(row.outcome_summary),
      toText(row.outcome_source_url),
      toText(row.outcome_confidence),
      toReal(row.outcome_confidence_score),
      toText(row.outcome_confidence_self_reported),
      toInt(row.outcome_contested),
      toText(row.outcome_contested_note),
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

  // company_urls: typed identity/reference URLs. UNIQUE(company_id, url_type, url)
  // absorbs any repeats the resolve step did not already collapse.
  const insertCompanyUrl = db.prepare(
    `INSERT OR IGNORE INTO company_urls (company_id, url_type, url, normalized_value, source_url)
     VALUES (?, ?, ?, ?, ?)`,
  )
  for (const row of companyUrls) {
    const companyId = companyIdFor(row)
    if (companyId === undefined) {
      skipped += 1
      continue
    }
    insertCompanyUrl.run(
      companyId,
      toText(row.url_type),
      toText(row.url) ?? '',
      toText(row.normalized_value),
      toText(row.source_url),
    )
  }

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
    `INSERT INTO challenges
      (company_id, category, outcome, detail, confidence, confidence_score, contested,
       contested_note, source_url)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
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
      toText(row.confidence),
      toReal(row.confidence_score),
      toInt(row.contested),
      toText(row.contested_note),
      toText(row.source_url),
    )
  }

  // dependencies: curated canonical dimension. name -> id, for the two tables below.
  const insertDependency = db.prepare(
    `INSERT INTO dependencies
      (uuid, name, category, description, threshold_metric, threshold_value, threshold_unit)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  )
  const dependencyIdByName = new Map<string, number>()
  for (const row of dependencies) {
    const info = insertDependency.run(
      toText(row.uuid),
      toText(row.name) ?? '',
      toText(row.category),
      toText(row.description),
      toText(row.threshold_metric),
      toReal(row.threshold_value),
      toText(row.threshold_unit),
    )
    if (row.name) {
      dependencyIdByName.set(row.name, Number(info.lastInsertRowid))
    }
  }

  // company_dependencies: resolve company_uuid + dependency_name. Rows step1c left
  // unresolved have no canonical dependency and are counted as skipped.
  const insertCompanyDependency = db.prepare(
    `INSERT OR IGNORE INTO company_dependencies
      (company_id, dependency_id, criticality, detail, source_url)
     VALUES (?, ?, ?, ?, ?)`,
  )
  for (const row of companyDependencies) {
    const companyId = companyIdFor(row)
    const dependencyId = dependencyIdByName.get(row.dependency_name ?? '')
    if (companyId === undefined || dependencyId === undefined) {
      skipped += 1
      continue
    }
    insertCompanyDependency.run(
      companyId,
      dependencyId,
      toText(row.criticality),
      toText(row.detail),
      toText(row.source_url),
    )
  }

  const insertAssessment = db.prepare(
    `INSERT INTO dependency_assessments
      (uuid, dependency_id, status, detail, metric_name, metric_value, metric_unit,
       assessed_on, source_url, snippet, confidence, confidence_score, contested, contested_note)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  for (const row of assessments) {
    const dependencyId = dependencyIdByName.get(row.dependency_name ?? '')
    if (dependencyId === undefined) {
      skipped += 1
      continue
    }
    insertAssessment.run(
      toText(row.uuid),
      dependencyId,
      toText(row.status),
      toText(row.detail),
      toText(row.metric_name),
      toReal(row.metric_value),
      toText(row.metric_unit),
      toText(row.assessed_on),
      toText(row.source_url),
      toText(row.snippet),
      toText(row.confidence),
      toReal(row.confidence_score),
      toInt(row.contested),
      toText(row.contested_note),
    )
  }

  // raw_documents: the on-disk fetch cache, mirrored into the DB for querying.
  const insertRawDocument = db.prepare(
    `INSERT OR IGNORE INTO raw_documents (url, url_hash, fetched_at, text, source_type)
     VALUES (?, ?, ?, ?, ?)`,
  )
  for (const document of rawDocuments) {
    insertRawDocument.run(
      document.url,
      document.url_hash,
      document.fetched_at,
      document.text,
      document.source_type,
    )
  }

  db.close()

  console.log(
    `Built ${dbFile}: ${SECTOR.length} sectors, ${ideaSpaces.length} idea_spaces, ` +
      `${companies.length} companies, ${companyUrls.length} company_urls, ` +
      `${companySectors.length} company_sectors, ` +
      `${fundingRounds.length} funding_rounds, ${challenges.length} challenges, ` +
      `${dependencies.length} dependencies, ${companyDependencies.length} company_dependencies, ` +
      `${assessments.length} dependency_assessments, ${rawDocuments.length} raw_documents` +
      (skipped ? ` (${skipped} child rows skipped — unresolved FK)` : ''),
  )
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
