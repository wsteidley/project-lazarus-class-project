import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { type CsvRow, readTableCsv, writeTableCsv } from '../lib/csv.js'
import {
  KEY_TIER,
  nearMissReport,
  remapCompanyUuids,
  resolveCompanies,
} from '../lib/entity-resolution.js'
import { latestRunDir } from '../lib/paths.js'
import { utcTimestamp } from '../lib/timestamp.js'

// Child tables keyed on company_uuid, rewritten onto the surviving canonical uuid.
// company_urls is handled separately: it is both a merge *input* and needs de-duping.
const CHILD_TABLES = [
  'company_sectors.csv',
  'challenges.csv',
  'idea_dependencies.csv',
  'company_dependencies.csv',
] as const

// resolve: merges duplicate companies into canonical rows before any enrichment runs.
//
// Ordering matters more than the merge itself. Running here — after extraction, before
// step2 — means funding is generated against canonical uuids only, so dedupeFundingRows
// never sees a pre-merge uuid to key on. The coupling is closed structurally rather
// than patched downstream.
const main = async (): Promise<void> => {
  console.log(utcTimestamp())

  const runDir = latestRunDir()
  const companies = await readTableCsv(runDir, 'companies.csv')
  const companyUrls = existsSync(join(runDir, 'company_urls.csv'))
    ? await readTableCsv(runDir, 'company_urls.csv')
    : []
  const {
    companies: canonical,
    urls: canonicalUrls,
    uuidMap,
    mergesByTier,
  } = resolveCompanies(companies, companyUrls)

  await writeTableCsv(canonical, runDir, 'companies.csv')
  // URLs are re-pointed and de-duplicated in the same pass, so the merged company
  // ends up with the union of its duplicates' URLs rather than repeats of one.
  await writeTableCsv(canonicalUrls, runDir, 'company_urls.csv')

  // Provenance: every original uuid and what it became.
  await writeTableCsv(
    [...uuidMap.entries()].map(([uuid, canonicalUuid]) => ({
      uuid,
      canonical_uuid: canonicalUuid,
      merged: uuid === canonicalUuid ? 0 : 1,
    })),
    runDir,
    'company_uuid_map.csv',
  )

  for (const table of CHILD_TABLES) {
    if (!existsSync(join(runDir, table))) {
      console.log(`(skipping ${table} — not found)`)
      continue
    }
    const rows: CsvRow[] = await readTableCsv(runDir, table)
    await writeTableCsv(remapCompanyUuids(rows, uuidMap), runDir, table)
  }

  const absorbed = companies.length - canonical.length
  console.log(`\nResolved ${companies.length} companies into ${canonical.length} canonical rows`)
  if (absorbed > 0) {
    const byTier = KEY_TIER.filter((tier) => mergesByTier[tier] > 0)
      .map((tier) => `${tier}: ${mergesByTier[tier]}`)
      .join(', ')
    console.log(`  ${absorbed} rows absorbed; merge groups by key tier — ${byTier}`)
    if (mergesByTier.name_year > 0) {
      console.log(
        '  note: name_year is the weakest tier (no shared identity key) — review those merges',
      )
    }
  }

  // Distinct canonical companies sharing a name are what ID-first refused to merge.
  // An empty report is the evidence that probabilistic linkage is not yet needed.
  const nearMisses = nearMissReport(canonical, canonicalUrls)
  if (nearMisses.length > 0) {
    console.warn(`\n${nearMisses.length} near-miss pairs left unmerged:`)
    for (const miss of nearMisses) {
      console.warn(`  - ${miss.left} / ${miss.right}: ${miss.reason}`)
    }
    console.warn('  These are the candidates probabilistic linkage (Splink) would catch.\n')
  } else {
    console.log('  no near-miss pairs — ID-first resolution was sufficient for this run')
  }

  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
