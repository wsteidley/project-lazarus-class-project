import { existsSync } from 'node:fs'
import { config } from '../config.js'
import { processInBatches } from '../lib/batch.js'
import { type CsvRow, readCsv, readTableCsv, writeTableCsv } from '../lib/csv.js'
import {
  type CompanyDependencyRow,
  dedupeCompanyDependencies,
  matchCanonical,
} from '../lib/dependency-resolution.js'
import { curatedFile, latestRunDir } from '../lib/paths.js'
import { utcTimestamp } from '../lib/timestamp.js'
import { buildChatModel } from '../llm.js'
import { type DependencyResolution, dependencyResolutionSchema } from '../schemas.js'

// Loads the curated canonical dependency seed to inject into the prompt and to
// validate the model's choice against. Mirrors loadIdeaSpaces in step1.
const loadCanonicalDependencies = async (): Promise<{
  names: string[]
  promptList: string
}> => {
  const seedPath = curatedFile('dependencies.csv')
  if (!existsSync(seedPath)) {
    throw new Error(`No ${seedPath} — the canonical dependency seed is required for step1c`)
  }
  const rows = await readCsv(seedPath)
  const names = rows.map((row) => row.name).filter((name): name is string => Boolean(name))
  const promptList = rows
    .map((row) => `- ${row.name} [${row.category}]${row.description ? `: ${row.description}` : ''}`)
    .join('\n')
  return { names, promptList }
}

const buildPrompt = (dependency: CsvRow, canonicalList: string): string =>
  `You are matching one company's free-text dependency onto a curated list of canonical dependencies. Choose the single canonical entry that names the same real-world thing. Return null if none genuinely fits — a wrong match is worse than no match, because assessments are shared across every company mapped to the same canonical dependency.

Canonical list:
${canonicalList}

Free-text dependency:
Category: ${dependency.category ?? ''}
Detail: ${dependency.detail ?? ''}`

const resolveDependency = (
  canonicalNames: string[],
  canonicalList: string,
): ((dependency: CsvRow) => Promise<CompanyDependencyRow>) => {
  return async (dependency: CsvRow): Promise<CompanyDependencyRow> => {
    const model = buildChatModel()
    const extractor = model.withStructuredOutput(dependencyResolutionSchema)

    let proposed: string | null = null
    try {
      const result = (await extractor.invoke(
        buildPrompt(dependency, canonicalList),
      )) as DependencyResolution
      proposed = result.canonical_name
    } catch (error) {
      // A failed call must leave the row unresolved, not drop it.
      console.warn(`Resolution failed for "${dependency.detail ?? ''}": ${String(error)}`)
    }

    return {
      company_uuid: dependency.company_uuid ?? '',
      dependency_name: matchCanonical(proposed, canonicalNames) ?? '',
      // The model's own answer, kept whether or not it matched. When it didn't, this is the
      // only name the blocker has, and build-db now retains the row on the strength of it.
      // Falls back to the extracted detail so an unresolved row is never anonymous.
      dependency_name_raw: proposed ?? dependency.detail ?? '',
      criticality: dependency.criticality ?? 'contributing',
      detail: dependency.detail ?? '',
      source_url: dependency.source_url ?? '',
    }
  }
}

const main = async (): Promise<void> => {
  console.log(utcTimestamp())

  const runDir = latestRunDir()
  const { names, promptList } = await loadCanonicalDependencies()
  const rawDependencies = await readTableCsv(runDir, 'idea_dependencies.csv')

  const resolved = await processInBatches(
    rawDependencies,
    resolveDependency(names, promptList),
    config.batchSize,
  )
  const merged = dedupeCompanyDependencies(resolved)

  const unresolved = merged.filter((row) => !row.dependency_name)
  if (unresolved.length > 0) {
    console.warn(
      `\n${unresolved.length} of ${rawDependencies.length} dependencies matched nothing canonical:`,
    )
    for (const row of unresolved) {
      console.warn(`  - ${row.detail}`)
    }
    console.warn('Add them to data/curated/dependencies.csv and re-run to pick them up.\n')
  }

  await writeTableCsv(merged, runDir, 'company_dependencies.csv')
  console.log(
    `Resolved ${rawDependencies.length} raw dependencies into ${merged.length} rows ` +
      `(${merged.length - unresolved.length} canonical, ${unresolved.length} unresolved)`,
  )
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
