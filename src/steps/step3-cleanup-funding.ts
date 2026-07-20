import { readTableCsv, writeTableCsv } from '../lib/csv.js'
import { dedupeFundingRows, type FundingCsvRow } from '../lib/funding.js'
import { utcTimestamp } from '../lib/timestamp.js'

// step3: deterministic cleanup of funding_rounds.csv. Standardizes dates to
// YYYY-MM, derives round_year, and de-duplicates rounds per company + round_name,
// writing the table back in place. No LLM, no century hardcode.
const main = async (): Promise<void> => {
  console.log(utcTimestamp())

  const rows = (await readTableCsv('funding_rounds.csv')) as unknown as FundingCsvRow[]
  const cleaned = dedupeFundingRows(rows)

  await writeTableCsv(cleaned, 'funding_rounds.csv')
  console.log(`Standardized ${rows.length} rows into ${cleaned.length} deduped rounds`)
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
