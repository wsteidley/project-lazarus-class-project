import { randomUUID } from 'node:crypto'
import { config } from '../config.js'
import { processInBatches } from '../lib/batch.js'
import { type CsvRow, readCsv, readTableCsv, writeTableCsv } from '../lib/csv.js'
import { latestRunDir, latestScrapedDataFile } from '../lib/paths.js'
import { utcTimestamp } from '../lib/timestamp.js'
import { buildChatModel } from '../llm.js'
import { type IdeaDependencies, ideaDependenciesSchema } from '../schemas.js'

// A company joined with its source article content (when available), so the
// dependency prompt has more than just the one-sentence idea_summary to work with.
type CompanyWithArticle = {
  company: CsvRow
  content: string
}

const buildPrompt = (
  company: CsvRow,
  content: string,
): string => `You are an analyst decomposing a startup idea into the external things it depended on to succeed. List each dependency using the controlled categories. Mark criticality 'was_blocking' if the dependency plausibly blocked the company's success, otherwise 'contributing'. Leave the list empty if nothing is evident.

Company: ${company.company_name}
Idea space: ${company.idea_space_name ?? ''}
Idea summary: ${company.idea_summary ?? ''}

Article content:
${content}`

const extractDependencies = async (
  input: CompanyWithArticle,
): Promise<Record<string, unknown>[]> => {
  const { company, content } = input
  const model = buildChatModel()
  const extractor = model.withStructuredOutput(ideaDependenciesSchema)

  const result = (await extractor.invoke(buildPrompt(company, content))) as IdeaDependencies

  return result.dependencies.map((dependency) => ({
    uuid: randomUUID(),
    company_uuid: company.uuid ?? '',
    category: dependency.category,
    detail: dependency.detail,
    criticality: dependency.criticality,
    source_url: company.source_url ?? '',
  }))
}

const main = async (): Promise<void> => {
  console.log(utcTimestamp())

  const runDir = latestRunDir()
  // The scraped article CSV supplies full content; join to companies by URL.
  const articleFile = process.env.INPUT_FILE || latestScrapedDataFile()

  const companies = await readTableCsv(runDir, 'companies.csv')
  const articles = await readCsv(articleFile)
  const contentByUrl = new Map(articles.map((article) => [article.url, article.content ?? '']))

  const joined: CompanyWithArticle[] = companies.map((company) => ({
    company,
    content: contentByUrl.get(company.source_url ?? '') ?? company.idea_summary ?? '',
  }))

  const dependencyRows = await processInBatches(joined, extractDependencies, config.batchSize)
  const flattened = dependencyRows.flat()

  await writeTableCsv(flattened, runDir, 'idea_dependencies.csv')
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
