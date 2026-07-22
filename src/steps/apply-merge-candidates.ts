import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { config } from '../config.js'
import { type CsvRow, readTableCsv, writeTableCsv } from '../lib/csv.js'
import { applyMergeCandidates, remapCompanyUuids } from '../lib/entity-resolution.js'
import { latestRunDir } from '../lib/paths.js'
import { utcTimestamp } from '../lib/timestamp.js'

// Child tables keyed on company_uuid, re-pointed onto the surviving canonical uuid —
// the same set resolve remaps, so fuzzy merges reach every downstream table.
const CHILD_TABLES = [
  'company_sectors.csv',
  'challenges.csv',
  'idea_dependencies.csv',
  'company_dependencies.csv',
] as const

// resolve:apply — the TypeScript half of the propose/apply contract. Reads the scored
// merge_candidates Splink proposed, applies those at/above FUZZY_MERGE_THRESHOLD through
// the same merge code path resolve uses (applyMergeCandidates → mergeGroup), and re-points
// URLs and child tables onto the survivors. Candidates below the threshold stay applied=0
// as a review queue. Idempotent: applied rows are skipped on a second run.
const main = async (): Promise<void> => {
  console.log(utcTimestamp())

  const runDir = latestRunDir()
  if (!existsSync(join(runDir, 'merge_candidates.csv'))) {
    console.warn(
      'No merge_candidates.csv — run resolve:fuzzy first (or it was skipped because uv is ' +
        'absent). Nothing to apply. DONE\n',
    )
    return
  }

  const companies = await readTableCsv(runDir, 'companies.csv')
  const companyUrls = existsSync(join(runDir, 'company_urls.csv'))
    ? await readTableCsv(runDir, 'company_urls.csv')
    : []
  const candidates = await readTableCsv(runDir, 'merge_candidates.csv')

  const threshold = config.fuzzyMergeThreshold
  const toMerge = candidates.filter(
    (row) => row.applied !== '1' && Number(row.match_score) >= threshold,
  )
  const reviewBand = candidates.filter(
    (row) => row.applied !== '1' && Number(row.match_score) < threshold,
  ).length

  if (toMerge.length === 0) {
    console.log(
      `No candidates at/above FUZZY_MERGE_THRESHOLD=${threshold} to apply ` +
        `(${reviewBand} below it, left for review). DONE\n`,
    )
    return
  }

  const pairs = toMerge.map((row) => ({ a: row.company_a ?? '', b: row.company_b ?? '' }))
  const {
    companies: canonical,
    urls,
    uuidMap,
    mergeCount,
  } = applyMergeCandidates(companies, companyUrls, pairs)

  await writeTableCsv(canonical, runDir, 'companies.csv')
  await writeTableCsv(urls, runDir, 'company_urls.csv')

  // Provenance map: fold this pass's survivors through the existing original→canonical map
  // so a uuid remapped twice (resolve then fuzzy) points at the final survivor.
  if (existsSync(join(runDir, 'company_uuid_map.csv'))) {
    const priorMap = await readTableCsv(runDir, 'company_uuid_map.csv')
    const updated = priorMap.map((row) => {
      const canonicalUuid = uuidMap.get(row.canonical_uuid ?? '') ?? row.canonical_uuid ?? ''
      return {
        uuid: row.uuid ?? '',
        canonical_uuid: canonicalUuid,
        merged: row.uuid === canonicalUuid ? 0 : 1,
      }
    })
    await writeTableCsv(updated, runDir, 'company_uuid_map.csv')
  }

  for (const table of CHILD_TABLES) {
    if (!existsSync(join(runDir, table))) {
      console.log(`(skipping ${table} — not found)`)
      continue
    }
    const rows: CsvRow[] = await readTableCsv(runDir, table)
    await writeTableCsv(remapCompanyUuids(rows, uuidMap), runDir, table)
  }

  // Mark the applied candidates so a re-run is a no-op; leave the review band untouched.
  const appliedKeys = new Set(toMerge.map((row) => `${row.company_a}::${row.company_b}`))
  const stamped = candidates.map((row) =>
    appliedKeys.has(`${row.company_a}::${row.company_b}`) ? { ...row, applied: '1' } : row,
  )
  await writeTableCsv(stamped, runDir, 'merge_candidates.csv')

  console.log(
    `Applied ${toMerge.length} candidates → ${mergeCount} merges ` +
      `(${companies.length} → ${canonical.length} companies); ${reviewBand} left for review`,
  )
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
