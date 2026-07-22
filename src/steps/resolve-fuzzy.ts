import { spawnSync } from 'node:child_process'
import { existsSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { type CsvRow, readTableCsv, writeTableCsv } from '../lib/csv.js'
import { normalizeName, normalizeUrlValue, urlsByCompany } from '../lib/entity-resolution.js'
import { latestRunDir } from '../lib/paths.js'
import { utcTimestamp } from '../lib/timestamp.js'

// resolve:fuzzy — the optional Splink tier's TypeScript half. It materializes a scratch
// SQLite from the run dir's CSVs (transport, not a store), hands it to link.py, and
// exports the proposed merge_candidates back to a CSV in the run dir. Python never sees
// the CSVs and never touches companies; TypeScript owns normalization and the merge apply.
//
// Exit-code contract (kept strictly distinct):
//   - uv absent            -> warn, exit 0   (graceful skip, like a missing TAVILY_API_KEY)
//   - uv present, link.py fails -> non-zero exit (a real error must never look like a skip)

// The bare hostname for a company, from its website URL row (same key the resolver uses).
const domainFor = (urls: CsvRow[]): string => {
  const website = urls.find((url) => url.url_type === 'website')
  return website?.normalized_value || normalizeUrlValue('website', website?.url ?? '')
}

const toYear = (value: string | undefined): number | null => {
  const year = Number((value ?? '').trim())
  return Number.isInteger(year) && year > 0 ? year : null
}

// Builds the single self-contained `companies` table link.py reads. All join keys are
// precomputed here with the resolver's own normalizers so Python never re-derives them.
// Missing values are stored NULL (not '') so Splink's comparisons treat them as absent
// rather than matching every other blank together.
const materializeScratchDb = (
  dbPath: string,
  companies: CsvRow[],
  companyUrls: CsvRow[],
): number => {
  if (existsSync(dbPath)) {
    rmSync(dbPath)
  }
  const db = new DatabaseSync(dbPath)
  db.exec(`CREATE TABLE companies (
    uuid            TEXT,
    company_name    TEXT,
    normalized_name TEXT,
    name_prefix     TEXT,
    year_founded    INTEGER,
    location        TEXT,
    country         TEXT,
    domain          TEXT
  )`)

  const urlsByUuid = urlsByCompany(companyUrls)
  const insert = db.prepare(
    `INSERT INTO companies
      (uuid, company_name, normalized_name, name_prefix, year_founded, location, country, domain)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  let inserted = 0
  for (const company of companies) {
    const normalized = normalizeName(company.company_name)
    const domain = domainFor(urlsByUuid.get(company.uuid ?? '') ?? [])
    insert.run(
      company.uuid ?? '',
      company.company_name ?? '',
      normalized,
      normalized.slice(0, 4) || null,
      toYear(company.year_founded),
      company.location?.trim() || null,
      company.country?.trim() || null,
      domain || null,
    )
    inserted += 1
  }
  db.close()
  return inserted
}

// Reads the merge_candidates table link.py wrote and returns it as plain rows.
const readMergeCandidates = (dbPath: string): CsvRow[] => {
  const db = new DatabaseSync(dbPath)
  try {
    const rows = db
      .prepare(
        'SELECT company_a, company_b, match_score, run_at, applied FROM merge_candidates ORDER BY match_score DESC',
      )
      .all() as Record<string, unknown>[]
    return rows.map((row) => ({
      company_a: String(row.company_a ?? ''),
      company_b: String(row.company_b ?? ''),
      match_score: String(row.match_score ?? ''),
      run_at: String(row.run_at ?? ''),
      applied: String(row.applied ?? '0'),
    }))
  } finally {
    db.close()
  }
}

const main = async (): Promise<void> => {
  console.log(utcTimestamp())

  const runDir = latestRunDir()
  const companies = await readTableCsv(runDir, 'companies.csv')
  const companyUrls = existsSync(join(runDir, 'company_urls.csv'))
    ? await readTableCsv(runDir, 'company_urls.csv')
    : []

  const dbPath = join(runDir, 'resolve.sqlite')
  const count = materializeScratchDb(dbPath, companies, companyUrls)
  console.log(`Materialized ${count} companies into ${dbPath}`)

  // Probe for uv. ENOENT (not installed) is a graceful skip; anything else is a real error.
  const probe = spawnSync('uv', ['--version'], { stdio: 'ignore' })
  if (probe.error && (probe.error as NodeJS.ErrnoException).code === 'ENOENT') {
    console.warn(
      'uv is not installed — skipping the Splink fuzzy tier (costs merge recall, not a run). ' +
        'Install uv to enable it. DONE\n',
    )
    return
  }
  if (probe.error) {
    throw probe.error
  }

  const link = spawnSync('uv', ['run', '--project', 'resolve', 'resolve/link.py', '--db', dbPath], {
    stdio: 'inherit',
  })
  if (link.status !== 0) {
    // A crashed linker is a real error, never a graceful skip.
    throw new Error(`resolve/link.py exited with status ${link.status ?? 'null (signal)'}`)
  }

  const candidates = readMergeCandidates(dbPath)
  await writeTableCsv(candidates, runDir, 'merge_candidates.csv')
  console.log(`Wrote ${candidates.length} merge candidates to merge_candidates.csv`)
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
