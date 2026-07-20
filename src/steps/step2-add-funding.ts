import { config } from '../config.js'
import { processInBatches } from '../lib/batch.js'
import { type CsvRow, readCsv, writeCsv } from '../lib/csv.js'
import { utcTimestamp } from '../lib/timestamp.js'
import { buildChatModel } from '../llm.js'
import { type FundingRounds, fundingRoundNames, fundingRoundsSchema } from '../schemas.js'
import { duckDuckGoSearch, wikipediaSearch } from '../tools/search.js'

type FundingResult = {
  uuid: string
  fundingRounds: FundingRounds['funding_rounds']
}

// Best-effort web search; returns empty context rather than failing the row.
const gatherSearchContext = async (companyName: string): Promise<string> => {
  const [ddgResult, wikiResult] = await Promise.all([
    duckDuckGoSearch.invoke(`${companyName} funding rounds`).catch(() => ''),
    wikipediaSearch.invoke(companyName).catch(() => ''),
  ])
  return `DuckDuckGo results:\n${ddgResult}\n\nWikipedia results:\n${wikiResult}`
}

const extractFundingForCompany = async (company: CsvRow): Promise<FundingResult> => {
  const companyName = company.company_name ?? ''
  const searchContext = await gatherSearchContext(companyName)

  const model = buildChatModel()
  const extractor = model.withStructuredOutput(fundingRoundsSchema)

  const prompt = `You are a helpful data aggregator and extractor. Using the company information and the search results below, find the company's funding rounds. Each round_name should be one of ${fundingRoundNames.join(', ')}. It's okay to leave fields null if the data is unavailable.

Company information: ${JSON.stringify(company)}

${searchContext}`

  const result = await extractor.invoke(prompt)
  return { uuid: company.uuid ?? '', fundingRounds: result.funding_rounds }
}

const main = async (): Promise<void> => {
  const filename = process.env.INPUT_FILE ?? ''
  if (!filename) {
    throw new Error('Set INPUT_FILE to the output CSV from step1')
  }

  const timestamp = utcTimestamp()
  console.log(timestamp)

  const companies = await readCsv(filename)
  const limitedCompanies = companies.slice(0, config.processingLimit)

  const fundingResults = await processInBatches(
    limitedCompanies,
    extractFundingForCompany,
    config.batchSize,
  )

  if (fundingResults.length === 0) {
    throw new Error('No data was successfully processed')
  }

  // Merge funding_rounds back onto the full input set, matched by uuid.
  const fundingByUuid = new Map(
    fundingResults.map((result) => [result.uuid, JSON.stringify(result.fundingRounds)]),
  )
  const enrichedRows = companies.map((row) => ({
    ...row,
    funding_rounds: fundingByUuid.get(row.uuid ?? '') ?? '',
  }))

  await writeCsv(enrichedRows, `added_funding_data_${timestamp}.csv`)
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
