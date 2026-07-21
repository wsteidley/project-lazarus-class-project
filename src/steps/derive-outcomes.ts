import { type CsvRow, readTableCsv, writeTableCsv } from '../lib/csv.js'
import { deriveOutcome } from '../lib/derive-outcome.js'
import { latestRunDir } from '../lib/paths.js'
import { utcTimestamp } from '../lib/timestamp.js'

// Per-company funding summary used as derivation input.
type FundingSummary = {
  totalRaised: number
  lastRoundYear: number | null
  lastRoundName: string | null
}

const toNumberOrNull = (value: string | undefined): number | null => {
  if (value === undefined || value === '') {
    return null
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

// Aggregates funding_rounds by company_uuid: total raised + the most recent round.
const summarizeFunding = (rows: CsvRow[]): Map<string, FundingSummary> => {
  const byCompany = new Map<string, FundingSummary>()
  for (const row of rows) {
    const uuid = row.company_uuid ?? ''
    const amount = toNumberOrNull(row.amount) ?? 0
    const roundYear = toNumberOrNull(row.round_year)
    const summary = byCompany.get(uuid) ?? {
      totalRaised: 0,
      lastRoundYear: null,
      lastRoundName: null,
    }
    summary.totalRaised += amount
    if (
      roundYear !== null &&
      (summary.lastRoundYear === null || roundYear >= summary.lastRoundYear)
    ) {
      summary.lastRoundYear = roundYear
      summary.lastRoundName = row.round_name ?? null
    }
    byCompany.set(uuid, summary)
  }
  return byCompany
}

// derivation step: computes outcome_type + outcome_rationale from living_status +
// total_raised + exit_* + company age, writing them back onto companies.csv. Runs
// after step3 (needs funding) and before build. Deterministic — re-runnable.
const main = async (): Promise<void> => {
  console.log(utcTimestamp())

  const buildYear = process.env.BUILD_DATE
    ? new Date(process.env.BUILD_DATE).getFullYear()
    : new Date().getFullYear()

  const runDir = latestRunDir()
  const companies = await readTableCsv(runDir, 'companies.csv')
  const fundingRows = await readTableCsv(runDir, 'funding_rounds.csv').catch(() => [])
  const fundingByCompany = summarizeFunding(fundingRows)

  const updated = companies.map((company) => {
    const summary = fundingByCompany.get(company.uuid ?? '') ?? {
      totalRaised: 0,
      lastRoundYear: null,
      lastRoundName: null,
    }
    const { outcome_type, outcome_rationale } = deriveOutcome({
      living_status: company.living_status ?? '',
      exit_type: company.exit_type || null,
      exit_amount: toNumberOrNull(company.exit_amount),
      total_raised: summary.totalRaised,
      year_founded: toNumberOrNull(company.year_founded),
      last_round_year: summary.lastRoundYear,
      last_round_name: summary.lastRoundName,
      build_year: buildYear,
    })
    return { ...company, outcome_type, outcome_rationale }
  })

  await writeTableCsv(updated, runDir, 'companies.csv')
  console.log(`Derived outcome_type for ${updated.length} companies`)
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
