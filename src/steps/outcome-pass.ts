import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { config } from '../config.js'
import { processInBatches } from '../lib/batch.js'
import { computeConfidence, deriveConfidenceLabel } from '../lib/confidence.js'
import { type CsvRow, readTableCsv, writeTableCsv } from '../lib/csv.js'
import { urlsByCompany } from '../lib/entity-resolution.js'
import { applyOutcomePrecedence, isStrongJoinSource } from '../lib/outcome-precedence.js'
import { latestRunDir } from '../lib/paths.js'
import { fetchWithCache } from '../lib/raw-documents.js'
import { utcTimestamp } from '../lib/timestamp.js'
import { checkDomainAlive, lastWaybackCaptureYear } from '../lib/wayback.js'
import { buildChatModel } from '../llm.js'
import {
  CONFIDENCE,
  EXIT_TYPE,
  LIVING_STATUS,
  type OutcomeAssessment,
  outcomeAssessmentSchema,
} from '../schemas.js'
import { gatherSearchContext } from '../tools/search.js'

const isEmpty = (value: string | undefined): boolean => value === undefined || value.trim() === ''

// Placeholder sampling proxy for CONFIDENCE_SCOPE=high_value: a company we already
// have some narrative on is worth the corroboration spend. Refine once real coverage
// data exists to judge "high value" against (see spec's open items).
const isHighValue = (company: CsvRow): boolean => !isEmpty(company.idea_summary)

const buildPrompt = (company: CsvRow, searchContext: string): string =>
  `You are determining what became of a company, using the search results below. living_status must be one of ${LIVING_STATUS.join(', ')}. exit_type must be one of ${EXIT_TYPE.join(', ')} or null if there was no exit. Give a source_url and a short verbatim snippet supporting your verdict. Set confidence (${CONFIDENCE.join(', ')}) by how directly the sources support the verdict, and contested=true if sources disagree.

Company: ${company.company_name ?? ''}
Idea: ${company.idea_summary ?? ''}
Known so far: living_status=${company.living_status || '(unknown)'}, exit_type=${company.exit_type || '(none recorded)'}

${searchContext}`

const assessOutcome = async (
  company: CsvRow,
  searchContext: string,
): Promise<OutcomeAssessment | null> => {
  const model = buildChatModel()
  const extractor = model.withStructuredOutput(outcomeAssessmentSchema)
  try {
    return (await extractor.invoke(buildPrompt(company, searchContext))) as OutcomeAssessment
  } catch (error) {
    console.warn(`Outcome assessment failed for "${company.company_name ?? ''}": ${String(error)}`)
    return null
  }
}

// outcome-pass: Phase 2b — fills gaps Phase 2a left and freshens volatile fields with
// search evidence. Runs after step3 so the funding picture (staleness) is available,
// and after enrich so it knows what's already a trustworthy static value versus a
// gap. Never redoes what 2a already strong-joined; see outcome-precedence.ts.
const main = async (): Promise<void> => {
  console.log(utcTimestamp())

  const runDir = latestRunDir()
  const companies = await readTableCsv(runDir, 'companies.csv')
  const companyUrls = existsSync(join(runDir, 'company_urls.csv'))
    ? await readTableCsv(runDir, 'company_urls.csv')
    : []
  const urlsByCompanyUuid = urlsByCompany(companyUrls)

  let searched = 0
  let skippedStrongJoin = 0
  let contested = 0

  // One row out per row in — a dropped company here would corrupt companies.csv, so
  // every failure path inside the handler resolves to a row rather than rejecting.
  const assessed = await processInBatches(
    companies,
    async (company): Promise<CsvRow> => {
      try {
        const urls = urlsByCompanyUuid.get(company.uuid ?? '') ?? []
        const websiteUrl = urls.find((url) => url.url_type === 'website')?.url ?? ''

        const [domainStatus, waybackYear] = await Promise.all([
          websiteUrl ? checkDomainAlive(websiteUrl) : Promise.resolve<'unknown'>('unknown'),
          websiteUrl ? lastWaybackCaptureYear(websiteUrl) : Promise.resolve(null),
        ])

        const hasStrongEnrichment =
          isStrongJoinSource(company.living_status_source) && !isEmpty(company.living_status)
        const inScope = config.confidenceScope === 'all' || isHighValue(company)
        const underSampleCap = config.confidenceSamples === 0 || searched < config.confidenceSamples

        let search: OutcomeAssessment | null = null
        if (hasStrongEnrichment) {
          skippedStrongJoin += 1
        } else if (inScope && underSampleCap) {
          searched += 1
          const cacheKey = `outcome-search://${company.uuid || company.company_name}`
          const searchContext = await fetchWithCache(cacheKey, 'outcome', () =>
            gatherSearchContext(
              `${company.company_name} shut down OR acquired OR still operating`,
              company.company_name ?? '',
            ),
          )
          search = await assessOutcome(company, searchContext)
        }

        const confidence = computeConfidence({
          enrichmentLivingStatus: company.living_status ?? '',
          searchLivingStatus: search?.living_status ?? '',
          domainStatus,
          selfReportedConfidence: search?.confidence ?? 'unknown',
          strongJoinEnrichment: hasStrongEnrichment,
        })
        if (confidence.contested) {
          contested += 1
        }

        const updated = applyOutcomePrecedence(company, search, waybackYear)
        return {
          ...updated,
          // Label is derived from the authoritative score (spec #4); the LLM's own
          // self-reported label is kept in a separate column, never blended in.
          outcome_confidence: deriveConfidenceLabel(confidence.confidence_score),
          outcome_confidence_score: String(confidence.confidence_score),
          outcome_confidence_self_reported: search?.confidence ?? '',
          outcome_contested: confidence.contested ? '1' : '0',
          outcome_contested_note: confidence.contested_note ?? search?.contested_note ?? '',
        }
      } catch (error) {
        console.warn(`Outcome pass failed for "${company.company_name ?? ''}": ${String(error)}`)
        return company
      }
    },
    config.batchSize,
  )

  await writeTableCsv(assessed, runDir, 'companies.csv')
  console.log(
    `Outcome pass: ${searched} searched, ${skippedStrongJoin} skipped (strong-join enrichment ` +
      `already trusted), ${contested} contested`,
  )
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
