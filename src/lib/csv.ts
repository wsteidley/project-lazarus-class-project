import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { parse } from 'csv-parse/sync'
import { stringify } from 'csv-stringify/sync'

// A CSV row is a flat map of column -> string value, matching how pandas
// round-trips these files. Callers parse/serialize richer fields themselves.
export type CsvRow = Record<string, string>

// Extra csv-parse options, for the occasional messy external file. `relaxColumnCount`
// tolerates ragged rows (hand-compiled source CSVs vary per row); `relaxQuotes`
// tolerates stray quotes mid-field. Both throw by default, which is what we want for
// the pipeline's own CSVs.
export type ReadCsvOptions = { relaxColumnCount?: boolean; relaxQuotes?: boolean }

export const readCsv = async (
  filename: string,
  options: ReadCsvOptions = {},
): Promise<CsvRow[]> => {
  const fileContents = await readFile(filename, 'utf-8')
  return parse(fileContents, {
    columns: true,
    skip_empty_lines: true,
    relax_column_count: options.relaxColumnCount ?? false,
    relax_quotes: options.relaxQuotes ?? false,
  }) as CsvRow[]
}

export const writeCsv = async (
  rows: Record<string, unknown>[],
  filename: string,
): Promise<void> => {
  if (!filename) {
    throw new Error('Need to provide a filename to save')
  }
  const csvText = stringify(rows, { header: true })
  await writeFile(filename, csvText, 'utf-8')
  console.log(`Data successfully saved to ${filename}`)
}

// Writes a relational table CSV (e.g. "companies.csv") into a run folder, creating
// it if needed. Returns the full path written.
export const writeTableCsv = async (
  rows: Record<string, unknown>[],
  runDir: string,
  tableFilename: string,
): Promise<string> => {
  await mkdir(runDir, { recursive: true })
  const fullPath = join(runDir, tableFilename)
  await writeCsv(rows, fullPath)
  return fullPath
}

// Reads a relational table CSV from a run folder.
export const readTableCsv = async (runDir: string, tableFilename: string): Promise<CsvRow[]> =>
  readCsv(join(runDir, tableFilename))
