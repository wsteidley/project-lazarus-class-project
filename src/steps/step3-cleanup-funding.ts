import { writeFile } from 'node:fs/promises'
import { type CsvRow, readCsv, writeCsv } from '../lib/csv.js'
import {
  type StandardizedFundingRound,
  aggregateAmountByYear,
  standardizeFundingData,
} from '../lib/funding.js'
import { utcTimestamp } from '../lib/timestamp.js'

const groupByCompany = (rows: CsvRow[]): Map<string, CsvRow[]> => {
  const groups = new Map<string, CsvRow[]>()
  for (const row of rows) {
    const companyName = row.company_name ?? ''
    const group = groups.get(companyName) ?? []
    group.push(row)
    groups.set(companyName, group)
  }
  return groups
}

// step3: deterministic cleanup. Groups rows by company, standardizes each
// company's funding rounds, and writes one reconciled row per company plus a
// year -> total-amount JSON sidecar (replacing the old matplotlib/plotly plot).
const main = async (): Promise<void> => {
  const filename = process.env.INPUT_FILE ?? ''
  if (!filename) {
    throw new Error('Set INPUT_FILE to the output CSV from step2')
  }

  const timestamp = utcTimestamp()
  console.log(timestamp)

  const rows = await readCsv(filename)
  const rowsWithFunding = rows.filter((row) => row.funding_rounds)

  const reconciledRows: CsvRow[] = []
  const allStandardizedRounds: StandardizedFundingRound[] = []

  for (const [, group] of groupByCompany(rowsWithFunding)) {
    const firstRow = group[0]
    if (!firstRow) {
      continue
    }
    const fundingStrings = group.map((row) => row.funding_rounds ?? '')
    const standardized = standardizeFundingData(fundingStrings)
    allStandardizedRounds.push(...standardized)

    // Keep the first row of the group, replacing its funding_rounds with the
    // reconciled set. Drop the schema-artifact columns if present.
    const { json_schema: _jsonSchema, type: _type, ...keptColumns } = firstRow
    reconciledRows.push({ ...keptColumns, funding_rounds: JSON.stringify(standardized) })
  }

  await writeCsv(reconciledRows, `standardized_funding_${timestamp}.csv`)

  const amountByYear = aggregateAmountByYear(allStandardizedRounds)
  const sidecarName = `standardized_funding_${timestamp}_amount_by_year.json`
  await writeFile(sidecarName, JSON.stringify(amountByYear, null, 2), 'utf-8')
  console.log(`Amount-by-year series saved to ${sidecarName}`)

  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
