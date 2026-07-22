import { randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import { config } from '../config.js'
import { processInBatches } from '../lib/batch.js'
import { type CsvRow, readCsv, writeTableCsv } from '../lib/csv.js'
import { inputFile, latestScrapedDataFile, newRunDir } from '../lib/paths.js'
import { utcTimestamp } from '../lib/timestamp.js'
import { buildChatModel } from '../llm.js'
import { type CompanyExtraction, companyExtractionSchema } from '../schemas.js'
import { crunchbaseCompanySearch } from '../tools/crunchbase.js'

// The table CSVs step1 produces, linked to the company by uuid.
type CompanyResult = {
  company: Record<string, unknown>
  sectors: Record<string, unknown>[]
  challenges: Record<string, unknown>[]
  urls: Record<string, unknown>[]
}

const bit = (value: boolean | null): 0 | 1 | '' => (value === null ? '' : value ? 1 : 0)

// Loads the curated idea_spaces seed (name + description) to inject into the
// prompt and to validate the model's choice against. Missing seed => no mapping.
const loadIdeaSpaces = async (): Promise<{ names: Set<string>; promptList: string }> => {
  const seedPath = inputFile('idea_spaces.csv')
  if (!existsSync(seedPath)) {
    console.warn(`No curated ${seedPath} — idea_space_name will be left unmapped`)
    return { names: new Set(), promptList: '(none provided)' }
  }
  const rows = await readCsv(seedPath)
  const names = new Set(rows.map((row) => row.name).filter((name): name is string => Boolean(name)))
  const promptList = rows
    .map((row) => `- ${row.name}${row.description ? `: ${row.description}` : ''}`)
    .join('\n')
  return { names, promptList }
}

const buildPrompt = (
  article: CsvRow,
  ideaSpaceList: string,
  crunchbaseContext: string,
): string => `You are a helpful data processor and extractor. Extract structured information about the company described in the article, using the required schema. Leave a field null if the article does not support a value. Record every notable challenge the company faced with its outcome (fatal / overcome / pivoted_from / ongoing) — this applies to survivors as well as failures. For each challenge also rate your confidence in it (unknown / low / medium / high) based on how directly the article supports it, and set contested=true only if the article itself reports conflicting accounts. Capture any exit event (acquisition/ipo/shutdown) in the exit_* fields, separate from funding. In urls, record every URL the sources actually state for the company (its own website, Crunchbase, Wikipedia, LinkedIn, etc.), each tagged with its type. Only include URLs a source really gives — these are used to merge duplicate companies, so an invented URL wrongly fuses two different companies; return an empty array when none are stated. Assign one or more sectors, marking exactly one is_primary.

Choose idea_space_name from this curated list (or null if none genuinely fits):
${ideaSpaceList}
${crunchbaseContext}
Title: ${article.title}
Author: ${article.author}
Publication Date: ${article.publication_date}
Content: ${article.content}`

const extractCompany = (
  ideaSpaceNames: Set<string>,
  ideaSpaceList: string,
): ((article: CsvRow) => Promise<CompanyResult>) => {
  return async (article: CsvRow): Promise<CompanyResult> => {
    const model = buildChatModel()
    const extractor = model.withStructuredOutput(companyExtractionSchema)

    let extracted = (await extractor.invoke(
      buildPrompt(article, ideaSpaceList, ''),
    )) as CompanyExtraction

    if (process.env.CRUNCHBASE_API_KEY && extracted.company_name) {
      try {
        const crunchbaseResult = await crunchbaseCompanySearch.invoke({
          companyName: extracted.company_name,
        })
        const crunchbaseContext = `Additional Crunchbase data for reference:\n${crunchbaseResult}\n`
        extracted = (await extractor.invoke(
          buildPrompt(article, ideaSpaceList, crunchbaseContext),
        )) as CompanyExtraction
      } catch (error) {
        console.error(`Crunchbase enrichment failed: ${String(error)}`)
      }
    }

    const uuid = randomUUID()
    const sourceUrl = article.url ?? ''

    // Only keep an idea_space_name that matches the curated seed; else flag it.
    let ideaSpaceName = extracted.idea_space_name ?? ''
    if (ideaSpaceName && !ideaSpaceNames.has(ideaSpaceName)) {
      console.warn(
        `Unmatched idea_space "${ideaSpaceName}" for ${extracted.company_name} — cleared`,
      )
      ideaSpaceName = ''
    }

    const company = {
      uuid,
      company_name: extracted.company_name,
      idea_space_name: ideaSpaceName,
      founders: extracted.founders,
      // Filled by the resolve step; a company is its own canonical row until merged.
      canonical_uuid: '',
      merged_from: '',
      location: extracted.location,
      country: extracted.country,
      year_founded: extracted.year_founded,
      year_defunct: extracted.year_defunct,
      // Provenance for these two fields; filled in by enrich/outcome-pass steps.
      living_status_source: '',
      year_founded_source: '',
      living_status: extracted.living_status,
      has_pivoted: bit(extracted.has_pivoted),
      idea_summary: extracted.idea_summary,
      exit_type: extracted.exit_type,
      exit_amount: extracted.exit_amount,
      exit_date: extracted.exit_date,
      exit_notes: extracted.exit_notes,
      outcome_summary: extracted.outcome_summary,
      outcome_source_url: extracted.outcome_source_url,
      // Set by the outcome pass (Phase 2b); left empty here.
      outcome_confidence: '',
      outcome_confidence_score: '',
      outcome_confidence_self_reported: '',
      outcome_contested: '',
      outcome_contested_note: '',
      // Derived in the derivation step; left empty here.
      outcome_type: '',
      outcome_rationale: '',
      original_trl: extracted.original_trl,
      is_climate: bit(extracted.is_climate),
      source_url: sourceUrl,
      created_at: utcTimestamp(),
    }

    const sectors = extracted.sectors.map((sector) => ({
      company_uuid: uuid,
      sector_name: sector.name,
      is_primary: sector.is_primary ? 1 : 0,
    }))

    const challenges = extracted.challenges.map((challenge) => ({
      company_uuid: uuid,
      category: challenge.category,
      outcome: challenge.outcome,
      detail: challenge.detail,
      confidence: challenge.confidence,
      confidence_score: '', // only populated when computed from signals, not self-reported
      contested: bit(challenge.contested),
      contested_note: challenge.contested_note,
      source_url: sourceUrl,
    }))

    const urls = extracted.urls
      .filter((entry) => entry.url?.trim())
      .map((entry) => ({
        company_uuid: uuid,
        url_type: entry.url_type,
        url: entry.url.trim(),
        // Comparable form, filled by the resolve step (bare domain for websites).
        normalized_value: '',
        source_url: sourceUrl,
      }))

    return { company, sectors, challenges, urls }
  }
}

const main = async (): Promise<void> => {
  // INPUT_FILE overrides; otherwise use the latest scrape from data/scraped.
  const filename = process.env.INPUT_FILE || latestScrapedDataFile()
  console.log(utcTimestamp())
  console.log(`Reading articles from ${filename}`)

  const { names: ideaSpaceNames, promptList: ideaSpaceList } = await loadIdeaSpaces()
  const articles = await readCsv(filename)
  const limitedArticles = articles.slice(0, config.processingLimit)

  const results = await processInBatches(
    limitedArticles,
    extractCompany(ideaSpaceNames, ideaSpaceList),
    config.batchSize,
  )

  if (results.length === 0) {
    throw new Error('No data was successfully processed')
  }

  // Start a fresh run folder for this extraction; later steps flow into it.
  const runDir = await newRunDir()
  await writeTableCsv(
    results.map((result) => result.company),
    runDir,
    'companies.csv',
  )
  await writeTableCsv(
    results.flatMap((result) => result.sectors),
    runDir,
    'company_sectors.csv',
  )
  await writeTableCsv(
    results.flatMap((result) => result.challenges),
    runDir,
    'challenges.csv',
  )
  await writeTableCsv(
    results.flatMap((result) => result.urls),
    runDir,
    'company_urls.csv',
  )
  console.log(`Run folder: ${runDir}`)
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
