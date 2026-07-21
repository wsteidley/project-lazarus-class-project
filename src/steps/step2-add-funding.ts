import { config } from '../config.js'
import { processInBatches } from '../lib/batch.js'
import { type CsvRow, readTableCsv, writeTableCsv } from '../lib/csv.js'
import { utcTimestamp } from '../lib/timestamp.js'
import { buildChatModel } from '../llm.js'
import { type FundingRounds, fundingRoundsSchema, ROUND } from '../schemas.js'
import { duckDuckGoSearch, wikipediaSearch } from '../tools/search.js'

// Derives the 4-digit year from a YYYY-MM date; '' when undated. Keeping the full
// year here (rather than a 2-digit MM/YY) is what avoids the century bug downstream.
const yearFromDate = (date: string | null): number | '' => {
  if (!date) {
    return ''
  }
  const match = date.match(/^(\d{4})/)
  return match?.[1] ? Number(match[1]) : ''
}

// Best-effort web search; returns empty context rather than failing the row.
const gatherSearchContext = async (companyName: string): Promise<string> => {
  const [ddgResult, wikiResult] = await Promise.all([
    duckDuckGoSearch.invoke(`${companyName} funding rounds`).catch(() => ''),
    wikipediaSearch.invoke(companyName).catch(() => ''),
  ])
  return `DuckDuckGo results:\n${ddgResult}\n\nWikipedia results:\n${wikiResult}`
}

const extractFundingForCompany = async (company: CsvRow): Promise<Record<string, unknown>[]> => {
  const companyName = company.company_name ?? ''
  const searchContext = await gatherSearchContext(companyName)

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

  const companies = await readTableCsv('companies.csv')
  const limitedCompanies = companies.slice(0, config.processingLimit)

  const fundingRows = await processInBatches(
    limitedCompanies,
    extractFundingForCompany,
    config.batchSize,
  )
  const flattened = fundingRows.flat()

  if (flattened.length === 0) {
    throw new Error('No funding data was successfully processed')
  }

  await writeTableCsv(flattened, 'funding_rounds.csv')
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
