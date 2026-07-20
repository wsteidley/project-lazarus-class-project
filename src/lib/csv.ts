import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { parse } from 'csv-parse/sync'
import { stringify } from 'csv-stringify/sync'
import { config } from '../config.js'

// A CSV row is a flat map of column -> string value, matching how pandas
// round-trips these files. Callers parse/serialize richer fields themselves.
export type CsvRow = Record<string, string>

export const readCsv = async (filename: string): Promise<CsvRow[]> => {
  const fileContents = await readFile(filename, 'utf-8')
  return parse(fileContents, {
    columns: true,
    skip_empty_lines: true,
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

// Writes a relational table CSV (e.g. "companies.csv") into DATA_DIR, creating the
// directory if needed. Returns the full path written.
export const writeTableCsv = async (
  rows: Record<string, unknown>[],
  tableFilename: string,
): Promise<string> => {
  await mkdir(config.dataDir, { recursive: true })
  const fullPath = join(config.dataDir, tableFilename)
  await writeCsv(rows, fullPath)
  return fullPath
}

// Reads a relational table CSV from DATA_DIR.
export const readTableCsv = async (tableFilename: string): Promise<CsvRow[]> =>
  readCsv(join(config.dataDir, tableFilename))
