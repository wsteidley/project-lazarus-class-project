import { copyFileSync, mkdirSync, readdirSync } from 'node:fs'
import { basename, join } from 'node:path'
import { config, sourcesDir } from '../config.js'
import { readCsv, writeCsv } from '../lib/csv.js'
import {
  crunchbaseKey,
  crunchbaseStatus,
  type EnrichmentRow,
  emptyEnrichmentRow,
  failureFlagsToChallenges,
  nameKey,
  parseRaisedAmount,
  parseYearsRange,
  websiteKey,
} from '../lib/sources.js'

// Where each source's untouched originals live. Copied into sources/<name>/raw/ on
// first build so the pipeline never reads from data/kaggle and the originals stay put.
const kaggleDir = join(config.dataDir, 'kaggle')
const CRUNCHBASE_RAW = join(
  kaggleDir,
  'startup-success-fail-archive',
  'big_startup_secsees_dataset.csv',
)
const FAILURES_RAW_DIR = join(kaggleDir, 'startup-failures-archive')

const rawDir = (source: string): string => join(sourcesDir, source, 'raw')

// Copies originals into the source folder and returns the raw paths now under raw/.
const stageRaw = (source: string, originals: string[]): string[] => {
  const dir = rawDir(source)
  mkdirSync(dir, { recursive: true })
  return originals.map((original) => {
    const dest = join(dir, basename(original))
    copyFileSync(original, dest)
    return dest
  })
}

// --- crunchbase: joins by website domain and permalink slug ---
const buildCrunchbase = async (): Promise<void> => {
  const [raw] = stageRaw('crunchbase', [CRUNCHBASE_RAW])
  if (!raw) {
    return
  }
  const rows = await readCsv(raw)
  const normalized: EnrichmentRow[] = rows.map((row) => {
    const { living_status, exit_type } = crunchbaseStatus(row.status)
    const { year_founded } = parseYearsRange(row.founded_at)
    return {
      ...emptyEnrichmentRow('crunchbase'),
      company_name: row.name ?? '',
      key_website: websiteKey(row.homepage_url),
      key_crunchbase: crunchbaseKey(row.permalink),
      key_name: nameKey(row.name),
      living_status,
      exit_type,
      funding_total_usd: /^\d+$/.test(row.funding_total_usd ?? '')
        ? (row.funding_total_usd ?? '')
        : '',
      funding_rounds: row.funding_rounds ?? '',
      year_founded,
      sector_raw: row.category_list ?? '',
      country: row.country_code ?? '',
      source_ref: row.permalink ?? '',
    }
  })
  await writeCsv(normalized, join(sourcesDir, 'crunchbase', 'companies.csv'))
  console.log(`crunchbase: ${normalized.length} rows`)
}

// --- startup-failures: a base roster (Startup Failures.csv) plus supporting
// sector-split tables that join back by name for the rich post-mortem detail. The
// normalized set is the *union* of names, so a roster company with no sector row is
// still kept (name/sector/years only), never dropped. ---
const buildStartupFailures = async (): Promise<void> => {
  const originals = readdirSync(FAILURES_RAW_DIR)
    .filter((file) => file.toLowerCase().endsWith('.csv'))
    .map((file) => join(FAILURES_RAW_DIR, file))
  const staged = stageRaw('startup-failures', originals)

  // Merge every file into one record per company. The base roster and the sector
  // tables are told apart by whether a file carries the 'Why They Failed' column.
  type Merged = { name: string; base?: Record<string, string>; rich?: Record<string, string> }
  const byName = new Map<string, Merged>()

  const absorb = (row: Record<string, string>, kind: 'base' | 'rich'): void => {
    const key = nameKey(row.Name)
    if (!key) {
      return
    }
    const entry = byName.get(key) ?? { name: row.Name ?? '' }
    entry[kind] = row
    entry.name ||= row.Name ?? ''
    byName.set(key, entry)
  }

  for (const path of staged) {
    const rows = await readCsv(path, { relaxColumnCount: true, relaxQuotes: true })
    const kind = rows[0] && 'Why They Failed' in rows[0] ? 'rich' : 'base'
    for (const row of rows) {
      if (row.Name?.trim()) {
        absorb(row, kind)
      }
    }
  }

  const normalized: EnrichmentRow[] = [...byName.values()].map(({ name, base, rich }) => {
    // Sector-file years ("2015-2019") are cleaner than the roster's "3 (2010-2013)".
    const { year_founded, year_defunct } = parseYearsRange(
      rich?.['Years of Operation'] || base?.['Years of Operation'],
    )
    return {
      ...emptyEnrichmentRow('startup-failures'),
      company_name: name,
      key_name: nameKey(name),
      living_status: 'Defunct', // failures by construction
      funding_total_usd: rich ? parseRaisedAmount(rich['How Much They Raised']) : '',
      year_founded,
      year_defunct,
      idea_summary: rich?.['What They Did'] ?? '',
      failure_reason: rich?.['Why They Failed'] ?? '',
      challenge_categories: rich ? failureFlagsToChallenges(rich).join(';') : '',
      sector_raw: rich?.Sector || base?.Sector || '',
      source_ref: name,
    }
  })

  await writeCsv(normalized, join(sourcesDir, 'startup-failures', 'companies.csv'))
  const withDetail = normalized.filter((row) => row.failure_reason).length
  console.log(`startup-failures: ${normalized.length} rows (${withDetail} with sector detail)`)
}

const main = async (): Promise<void> => {
  await buildCrunchbase()
  await buildStartupFailures()
  console.log('\nDONE — startup-india (.xlsx) not yet converted; see data/sources/README.md\n')
}

main().catch((error) => {
  console.error(`Error building sources: ${String(error)}`)
  process.exitCode = 1
})
