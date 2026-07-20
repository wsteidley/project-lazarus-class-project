import { randomUUID } from 'node:crypto'
import { config } from '../config.js'
import { processInBatches } from '../lib/batch.js'
import { type CsvRow, readCsv, writeCsv } from '../lib/csv.js'
import { utcTimestamp } from '../lib/timestamp.js'
import { buildChatModel } from '../llm.js'
import { type CompanyInfo, companyInfoSchema } from '../schemas.js'
import { crunchbaseCompanySearch } from '../tools/crunchbase.js'

// A step1 output row: extracted company info plus provenance metadata.
type ExtractedCompany = CompanyInfo & {
  url: string
  timestamp: string
  uuid: string
}

const buildPrompt = (
  article: CsvRow,
  crunchbaseContext: string,
): string => `You are a helpful data processor and extractor. Pull the proper information for the company described in the article and output it in the required schema. It's okay to leave a field null if the article does not support a value, as long as the required fields are present.
${crunchbaseContext}
Title: ${article.title}
Author: ${article.author}
Publication Date: ${article.publication_date}
Content: ${article.content}`

const extractCompanyFromArticle = async (article: CsvRow): Promise<ExtractedCompany> => {
  const model = buildChatModel()
  const extractor = model.withStructuredOutput(companyInfoSchema)

  // Structured extraction from the article alone.
  let companyInfo = (await extractor.invoke(buildPrompt(article, ''))) as CompanyInfo

  // Optional enrichment: if a Crunchbase key is configured and we found a name,
  // fetch Crunchbase data and run one refinement pass with it as extra context.
  if (process.env.CRUNCHBASE_API_KEY && companyInfo.company_name) {
    try {
      const crunchbaseResult = await crunchbaseCompanySearch.invoke({
        companyName: companyInfo.company_name,
      })
      const crunchbaseContext = `Additional Crunchbase data for reference:\n${crunchbaseResult}\n`
      companyInfo = (await extractor.invoke(buildPrompt(article, crunchbaseContext))) as CompanyInfo
    } catch (error) {
      console.error(`Crunchbase enrichment failed: ${String(error)}`)
    }
  }

  return {
    ...companyInfo,
    url: article.url ?? '',
    timestamp: utcTimestamp(),
    uuid: randomUUID(),
  }
}

const main = async (): Promise<void> => {
  const filename = process.env.INPUT_FILE ?? ''
  if (!filename) {
    throw new Error('Set INPUT_FILE to the output CSV from step0')
  }

  const timestamp = utcTimestamp()
  console.log(timestamp)

  const articles = await readCsv(filename)
  const limitedArticles = articles.slice(0, config.processingLimit)

  const extracted = await processInBatches(
    limitedArticles,
    extractCompanyFromArticle,
    config.batchSize,
  )

  if (extracted.length === 0) {
    throw new Error('No data was successfully processed')
  }

  await writeCsv(extracted, `parsed_data_${timestamp}.csv`)
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
