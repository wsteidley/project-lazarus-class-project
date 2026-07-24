import { randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import { config } from '../config.js'
import { processInBatches } from '../lib/batch.js'
import { type CsvRow, readCsv, writeTableCsv } from '../lib/csv.js'
import { inputFile, latestRunDir } from '../lib/paths.js'
import { utcTimestamp } from '../lib/timestamp.js'
import { buildChatModel } from '../llm.js'
import {
  ASSESSMENT_STATUS,
  type DependencyAssessment,
  dependencyAssessmentSchema,
} from '../schemas.js'
import { gatherSearchContext } from '../tools/search.js'

// Search dated to now, so the verdict reflects where the dependency stands today
// rather than when companies first depended on it.
const searchContextFor = (dependency: CsvRow): Promise<string> => {
  const name = dependency.name ?? ''
  const metric = dependency.threshold_metric ?? ''
  const query = metric ? `${name} ${metric} current cost trend` : `${name} current status trend`
  return gatherSearchContext(query, name)
}

const buildPrompt = (dependency: CsvRow, searchContext: string, today: string): string =>
  `You are assessing where one dependency stands TODAY (${today}), compared with when climate-tech companies first depended on it. Use the search results below. status must be one of ${ASSESSMENT_STATUS.join(', ')}.

Report a metric only if the sources actually give one — never estimate a number. Quote a short verbatim snippet from the source that supports your verdict, and give the source_url it came from. Set confidence by how directly the sources support the verdict, and contested=true if they disagree with each other.

Dependency: ${dependency.name ?? ''}
Category: ${dependency.category ?? ''}
Description: ${dependency.description ?? ''}
Threshold of interest: ${dependency.threshold_metric ?? '(none defined)'} ${dependency.threshold_value ?? ''} ${dependency.threshold_unit ?? ''}

${searchContext}`

const assessDependency = (
  assessedOn: string,
): ((dependency: CsvRow) => Promise<Record<string, unknown> | null>) => {
  return async (dependency: CsvRow): Promise<Record<string, unknown> | null> => {
    const searchContext = await searchContextFor(dependency)

    const model = buildChatModel()
    const extractor = model.withStructuredOutput(dependencyAssessmentSchema)

    let assessment: DependencyAssessment
    try {
      assessment = (await extractor.invoke(
        buildPrompt(dependency, searchContext, assessedOn),
      )) as DependencyAssessment
    } catch (error) {
      // One bad dependency shouldn't sink the pass; skip the row and keep going.
      console.warn(`Assessment failed for "${dependency.name ?? ''}": ${String(error)}`)
      return null
    }

    return {
      uuid: randomUUID(),
      dependency_name: dependency.name ?? '',
      status: assessment.status,
      detail: assessment.detail,
      metric_name: assessment.metric_name,
      metric_value: assessment.metric_value,
      metric_unit: assessment.metric_unit,
      assessed_on: assessedOn,
      source_url: assessment.source_url,
      snippet: assessment.snippet,
      confidence: assessment.confidence,
      // Null until confidence is computed from signals rather than self-reported.
      confidence_score: '',
      contested: assessment.contested ? 1 : 0,
      contested_note: assessment.contested_note,
    }
  }
}

const main = async (): Promise<void> => {
  const assessedOn = utcTimestamp()
  console.log(assessedOn)

  const seedPath = inputFile('dependencies.csv')
  if (!existsSync(seedPath)) {
    throw new Error(`No ${seedPath} — the canonical dependency seed is required for reassess`)
  }
  // Assessments attach to canonical dependencies, not companies: the "now" verdict
  // for a shared dependency is established once, not re-derived per company.
  const dependencies = await readCsv(seedPath)
  // v2 moved the bar out of dependencies.csv into dependency_thresholds.csv. Attach each
  // dependency's primary bar so the search query + prompt keep their threshold context.
  const thresholdPath = inputFile('dependency_thresholds.csv')
  const primaryBar = new Map<string, CsvRow>()
  if (existsSync(thresholdPath)) {
    for (const bar of await readCsv(thresholdPath)) {
      if (bar.dependency_name && !primaryBar.has(bar.dependency_name)) {
        primaryBar.set(bar.dependency_name, bar)
      }
    }
  }
  for (const dependency of dependencies) {
    const bar = primaryBar.get(dependency.name ?? '')
    dependency.threshold_metric = bar?.metric ?? ''
    dependency.threshold_value = bar?.threshold_value ?? ''
    dependency.threshold_unit = bar?.threshold_unit ?? ''
  }
  const runDir = latestRunDir()

  const assessed = await processInBatches(
    dependencies,
    assessDependency(assessedOn),
    config.batchSize,
  )
  const rows = assessed.filter((row): row is Record<string, unknown> => row !== null)

  if (rows.length === 0) {
    throw new Error('No dependency assessments were successfully produced')
  }

  await writeTableCsv(rows, runDir, 'dependency_assessments.csv')
  console.log(`Assessed ${rows.length} of ${dependencies.length} dependencies`)
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
