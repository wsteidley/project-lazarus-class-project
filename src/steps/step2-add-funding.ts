import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { config } from '../config.js'
import { processInBatches } from '../lib/batch.js'
import { type CsvRow, readTableCsv, writeTableCsv } from '../lib/csv.js'
import { isEnrichmentFundingRow } from '../lib/funding.js'
import { latestRunDir } from '../lib/paths.js'
import { utcTimestamp } from '../lib/timestamp.js'
import { buildChatModel } from '../llm.js'
import { type FundingRounds, fundingRoundsSchema, ROUND } from '../schemas.js'
import { gatherSearchContext } from '../tools/search.js'

// Derives the 4-digit year from a YYYY-MM date; '' when undated. Keeping the full
// year here (rather than a 2-digit MM/YY) is what avoids the century bug downstream.
const yearFromDate = (date: string | null): number | '' => {
  if (!date) {
    return ''
  }
  const match = date.match(/^(\d{4})/)
  return match?.[1] ? Number(match[1]) : ''
}

const extractFundingForCompany = async (company: CsvRow): Promise<Record<string, unknown>[]> => {
  const companyName = company.company_name ?? ''
  const searchContext = await gatherSearchContext(`${companyName} funding rounds`, companyName)

  const model = buildChatModel()
  const extractor = model.withStructuredOutput(fundingRoundsSchema)

  const prompt = `You are a helpful data aggregator and extractor. Using the company information and the search results below, find the company's funding rounds. round_name must be one of ${ROUND.join(', ')}. Give amount as a number with a separate ISO currency code, and date as YYYY-MM. Leave fields null if unavailable.

Company: ${companyName}
Idea space: ${company.idea_space_name ?? ''}

${searchContext}`

  const result = (await extractor.invoke(prompt)) as FundingRounds
  const sourceUrl = company.source_url ?? ''

  return result.funding_rounds.map((round) => ({
    company_uuid: company.uuid ?? '',
    round_name: round.round_name,
    amount: round.amount,
    currency: round.currency,
    round_date: round.date,
    round_year: yearFromDate(round.date),
    source_url: sourceUrl,
  }))
}

const main = async (): Promise<void> => {
  console.log(utcTimestamp())

  const runDir = latestRunDir()
  const companies = await readTableCsv(runDir, 'companies.csv')
  const limitedCompanies = companies.slice(0, config.processingLimit)

  const fundingRows = await processInBatches(
    limitedCompanies,
    extractFundingForCompany,
    config.batchSize,
  )
  const flattened = fundingRows.flat()

  // This step overwrites funding_rounds.csv, but Phase 2a (enrich) may have written
  // aggregate funding rows there first. Carry those across — except for companies
  // this step found real per-round data for, whose granular rounds supersede the
  // lump-sum aggregate (keeping both would double-count in summarizeFunding).
  const existingFunding = existsSync(join(runDir, 'funding_rounds.csv'))
    ? await readTableCsv(runDir, 'funding_rounds.csv')
    : []
  const companiesWithSearchedFunding = new Set(flattened.map((row) => row.company_uuid))
  const enrichmentFunding = existingFunding
    .filter(isEnrichmentFundingRow)
    .filter((row) => !companiesWithSearchedFunding.has(row.company_uuid))

  const combined = [...enrichmentFunding, ...flattened]
  if (combined.length === 0) {
    throw new Error('No funding data (enrichment or search) was produced')
  }

  await writeTableCsv(combined, runDir, 'funding_rounds.csv')
  console.log(
    `Wrote ${combined.length} funding rows (${flattened.length} searched, ` +
      `${enrichmentFunding.length} carried from enrichment)`,
  )
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
