import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { readTableCsv, writeTableCsv } from '../lib/csv.js'
import {
  applyEnrichment,
  type EnrichTier,
  enrichmentKeysFor,
  findMatch,
  isEnrichmentChallengeRow,
  SOURCE_NAMES,
  seedChallenges,
  syntheticFundingRow,
} from '../lib/enrich.js'
import { urlsByCompany } from '../lib/entity-resolution.js'
import { isEnrichmentFundingRow } from '../lib/funding.js'
import { latestRunDir } from '../lib/paths.js'
import { type EnrichmentIndex, loadSource } from '../lib/sources.js'
import { utcTimestamp } from '../lib/timestamp.js'

// enrich: Phase 2a — the cheap, deterministic first half of the outcome axis. Fills
// living_status/exit_type/years/funding/failure-reason from local reference data
// (data/sources/) before any search runs. Runs after resolve, before step2, so
// funding rows this step adds are visible to step2/derive like any other round.
const main = async (): Promise<void> => {
  console.log(utcTimestamp())

  const runDir = latestRunDir()
  const companies = await readTableCsv(runDir, 'companies.csv')
  const companyUrls = existsSync(join(runDir, 'company_urls.csv'))
    ? await readTableCsv(runDir, 'company_urls.csv')
    : []
  const existingChallenges = existsSync(join(runDir, 'challenges.csv'))
    ? await readTableCsv(runDir, 'challenges.csv')
    : []
  const existingFunding = existsSync(join(runDir, 'funding_rounds.csv'))
    ? await readTableCsv(runDir, 'funding_rounds.csv')
    : []

  const indexBySource = new Map<string, EnrichmentIndex>()
  for (const source of SOURCE_NAMES) {
    indexBySource.set(source, await loadSource(source))
  }

  const urlsByCompanyUuid = urlsByCompany(companyUrls)
  const matchesByTier: Record<EnrichTier, number> = { website: 0, crunchbase: 0, name: 0 }
  let unmatched = 0
  const newFundingRows: Record<string, unknown>[] = []
  const newChallengeRows: Record<string, unknown>[] = []

  const enriched = companies.map((company) => {
    const urls = urlsByCompanyUuid.get(company.uuid ?? '') ?? []
    const { websiteKey, crunchbaseKey, nameKey } = enrichmentKeysFor(
      company.company_name ?? '',
      urls,
    )

    const match = findMatch(indexBySource, websiteKey, crunchbaseKey, nameKey)
    if (!match) {
      unmatched += 1
      return company
    }
    matchesByTier[match.tier] += 1

    const fundingRow = syntheticFundingRow(company.uuid ?? '', match)
    if (fundingRow) {
      newFundingRows.push(fundingRow)
    }
    newChallengeRows.push(...seedChallenges(company.uuid ?? '', match))

    return applyEnrichment(company, match)
  })

  await writeTableCsv(enriched, runDir, 'companies.csv')

  // Re-runnable: drop this step's own prior rows (identified by their enrichment
  // provenance) before re-appending, so a second run replaces rather than duplicates.
  const priorFunding = existingFunding.filter((row) => !isEnrichmentFundingRow(row))
  if (newFundingRows.length > 0 || priorFunding.length < existingFunding.length) {
    await writeTableCsv([...priorFunding, ...newFundingRows], runDir, 'funding_rounds.csv')
  }
  const priorChallenges = existingChallenges.filter((row) => !isEnrichmentChallengeRow(row))
  if (newChallengeRows.length > 0 || priorChallenges.length < existingChallenges.length) {
    await writeTableCsv([...priorChallenges, ...newChallengeRows], runDir, 'challenges.csv')
  }

  const matched = matchesByTier.website + matchesByTier.crunchbase + matchesByTier.name
  console.log(
    `Enriched ${matched}/${companies.length} companies — website: ${matchesByTier.website}, ` +
      `crunchbase: ${matchesByTier.crunchbase}, name-only: ${matchesByTier.name}, ` +
      `unmatched: ${unmatched}`,
  )
  if (matchesByTier.name > 0) {
    console.log(
      `  note: ${matchesByTier.name} name-only matches — any failure-reason challenges from ` +
        'these are seeds, not evidence (see contested_note)',
    )
  }
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
