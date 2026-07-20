import { randomUUID } from 'node:crypto'
import { config } from '../config.js'
import { processInBatches } from '../lib/batch.js'
import { type CsvRow, readCsv, writeTableCsv } from '../lib/csv.js'
import { utcTimestamp } from '../lib/timestamp.js'
import { buildChatModel } from '../llm.js'
import { type CompanyExtraction, companyExtractionSchema } from '../schemas.js'
import { crunchbaseCompanySearch } from '../tools/crunchbase.js'

// One company row (companies.csv) plus its failure reasons, linked by uuid.
type CompanyResult = {
  company: Record<string, unknown>
  failureReasons: Record<string, unknown>[]
}

const bit = (value: boolean | null): 0 | 1 | '' => (value === null ? '' : value ? 1 : 0)

const buildPrompt = (
  article: CsvRow,
  crunchbaseContext: string,
): string => `You are a helpful data processor and extractor. Extract structured information about the company described in the article and output it in the required schema. Leave a field null if the article does not support a value. If the company failed, populate failure_reasons with one entry per distinct cause using the controlled categories; use an empty array if it did not fail.
${crunchbaseContext}
Title: ${article.title}
Author: ${article.author}
Publication Date: ${article.publication_date}
Content: ${article.content}`

const extractCompanyFromArticle = async (article: CsvRow): Promise<CompanyResult> => {
  const model = buildChatModel()
  const extractor = model.withStructuredOutput(companyExtractionSchema)

  let extracted = (await extractor.invoke(buildPrompt(article, ''))) as CompanyExtraction

  // Optional enrichment: if a Crunchbase key is configured and we found a name,
  // fetch Crunchbase data and run one refinement pass with it as extra context.
  if (process.env.CRUNCHBASE_API_KEY && extracted.company_name) {
    try {
      const crunchbaseResult = await crunchbaseCompanySearch.invoke({
        companyName: extracted.company_name,
      })
      const crunchbaseContext = `Additional Crunchbase data for reference:\n${crunchbaseResult}\n`
      extracted = (await extractor.invoke(
        buildPrompt(article, crunchbaseContext),
      )) as CompanyExtraction
    } catch (error) {
      console.error(`Crunchbase enrichment failed: ${String(error)}`)
    }
  }

  const uuid = randomUUID()
  const sourceUrl = article.url ?? ''
  const createdAt = utcTimestamp()

  const company = {
    uuid,
    company_name: extracted.company_name,
    founders: extracted.founders,
    sector: extracted.sector,
    subsector: extracted.subsector,
    location: extracted.location,
    country: extracted.country,
    year_founded: extracted.year_founded,
    year_defunct: extracted.year_defunct,
    living_status: extracted.living_status,
    has_pivoted: bit(extracted.has_pivoted),
    idea_summary: extracted.idea_summary,
    original_trl: extracted.original_trl,
    is_climate: bit(extracted.is_climate),
    source_url: sourceUrl,
    created_at: createdAt,
  }

  const failureReasons = extracted.failure_reasons.map((reason) => ({
    company_uuid: uuid,
    category: reason.category,
    detail: reason.detail,
    source_url: sourceUrl,
  }))

  return { company, failureReasons }
}

const main = async (): Promise<void> => {
  const filename = process.env.INPUT_FILE ?? ''
  if (!filename) {
    throw new Error('Set INPUT_FILE to the article CSV from step0')
  }

  console.log(utcTimestamp())

  const articles = await readCsv(filename)
  const limitedArticles = articles.slice(0, config.processingLimit)

  const results = await processInBatches(
    limitedArticles,
    extractCompanyFromArticle,
    config.batchSize,
  )

  if (results.length === 0) {
    throw new Error('No data was successfully processed')
  }

  const companies = results.map((result) => result.company)
  const failureReasons = results.flatMap((result) => result.failureReasons)

  await writeTableCsv(companies, 'companies.csv')
  await writeTableCsv(failureReasons, 'failure_reasons.csv')
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
