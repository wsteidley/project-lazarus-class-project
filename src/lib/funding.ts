import { ROUND } from '../schemas.js'

// A funding_rounds.csv row, as read back (all cells are strings).
export type FundingCsvRow = {
  company_uuid: string
  round_name: string
  amount: string
  currency: string
  round_date: string
  round_year: string
  source_url: string
}

// A deduped, standardized funding round ready to write back.
export type StandardizedFundingRow = {
  company_uuid: string
  round_name: string
  amount: number
  currency: string
  round_date: string
  round_year: number | ''
  source_url: string
}

// A funding row that came from Phase 2a enrichment (an aggregate total from a static
// source) rather than the step2 LLM search. Tagged by its source_url provenance
// (`enrichment:<source>:<tier>`), so step2 can carry it across its overwrite and
// enrich can drop its own prior rows before re-appending (idempotent re-runs).
export const isEnrichmentFundingRow = (row: { source_url?: string }): boolean =>
  (row.source_url ?? '').startsWith('enrichment:')

// Maps a loosely-worded round name onto the controlled ROUND vocabulary, or "Unknown".
export const fixRoundName = (currentName: string | null | undefined): string => {
  for (const name of ROUND) {
    if (currentName?.toLowerCase().includes(name.toLowerCase())) {
      return name
    }
  }
  return 'Unknown'
}

// Parses many date shapes into YYYY-MM, preserving the 4-digit year (the fix for
// the old MM/YY century bug). Returns null if nothing matches. A 2-digit input
// year is assumed to be 20xx as a last resort; step2 asks for 4-digit years.
export const standardizeDate = (input: string | null | undefined): string | null => {
  if (input === null || input === undefined) {
    return null
  }
  const dateString = String(input).trim()
  if (!dateString) {
    return null
  }
  const pad = (value: string): string => value.padStart(2, '0')
  const fourDigitYear = (year: string): string => (year.length === 2 ? `20${year}` : year)
  const cap = (match: RegExpMatchArray, index: number): string => match[index] ?? ''

  const attempts: [RegExp, (match: RegExpMatchArray) => string][] = [
    // YYYY-MM-DD / YYYY/MM/DD
    [/^(\d{4})[/.-](\d{1,2})[/.-]\d{1,2}$/, (m) => `${cap(m, 1)}-${pad(cap(m, 2))}`],
    // YYYY-MM / YYYY/MM
    [/^(\d{4})[/.-](\d{1,2})$/, (m) => `${cap(m, 1)}-${pad(cap(m, 2))}`],
    // YYYY only
    [/^(\d{4})$/, (m) => `${cap(m, 1)}-01`],
    // MM/DD/YYYY
    [/^(\d{1,2})[/.-]\d{1,2}[/.-](\d{4})$/, (m) => `${cap(m, 2)}-${pad(cap(m, 1))}`],
    // MM/YYYY
    [/^(\d{1,2})[/.-](\d{4})$/, (m) => `${cap(m, 2)}-${pad(cap(m, 1))}`],
    // MM/YY (best-effort century)
    [/^(\d{1,2})[/.-](\d{2})$/, (m) => `${fourDigitYear(cap(m, 2))}-${pad(cap(m, 1))}`],
  ]

  for (const [pattern, formatter] of attempts) {
    const match = dateString.match(pattern)
    if (match) {
      return formatter(match)
    }
  }

  // Textual dates like "March 3, 2020" or "March 2020".
  const textual = new Date(dateString)
  if (!Number.isNaN(textual.getTime()) && /\d{4}/.test(dateString)) {
    return `${textual.getFullYear()}-${pad(String(textual.getMonth() + 1))}`
  }

  return null
}

export const yearFromStandardizedDate = (date: string | null): number | '' => {
  if (!date) {
    return ''
  }
  const match = date.match(/^(\d{4})/)
  return match?.[1] ? Number(match[1]) : ''
}

const parseAmount = (raw: string): number => {
  const parsed = Number(raw)
  return Number.isFinite(parsed) ? Math.trunc(parsed) : 0
}

// De-duplicates funding rows per (company_uuid, round_name): keeps the largest
// amount, a standardized date, currency, and derived 4-digit round_year. Replaces
// the old JSON-blob reconciliation.
export const dedupeFundingRows = (rows: FundingCsvRow[]): StandardizedFundingRow[] => {
  const byKey = new Map<string, StandardizedFundingRow>()

  for (const row of rows) {
    const roundName =
      row.round_name && ROUND.includes(row.round_name as (typeof ROUND)[number])
        ? row.round_name
        : fixRoundName(row.round_name)
    const key = `${row.company_uuid}::${roundName}`
    const standardizedDate = standardizeDate(row.round_date)

    const existing = byKey.get(key) ?? {
      company_uuid: row.company_uuid,
      round_name: roundName,
      amount: 0,
      currency: '',
      round_date: '',
      round_year: '' as number | '',
      source_url: row.source_url,
    }

    existing.amount = Math.max(existing.amount, parseAmount(row.amount))
    if (!existing.currency && row.currency) {
      existing.currency = row.currency
    }
    if (!existing.round_date && standardizedDate) {
      existing.round_date = standardizedDate
      existing.round_year = yearFromStandardizedDate(standardizedDate)
    }
    byKey.set(key, existing)
  }

  return [...byKey.values()]
}
