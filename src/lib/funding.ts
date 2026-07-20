import { fundingRoundNames } from '../schemas.js'

export type RawFundingRound = {
  currency_symbol?: string | null
  round_name?: string | null
  amount?: number | null
  date?: string | null
  // The model sometimes emits schema-shaped junk objects; these mark them.
  type?: unknown
  properties?: unknown
}

export type StandardizedFundingRound = {
  round_name: string
  currency_symbol?: string | null
  amount: number
  date: string | null
}

// Maps a loosely-worded round name onto the controlled vocabulary, or "Unknown".
export const fixRoundName = (currentName: string | null | undefined): string => {
  for (const name of fundingRoundNames) {
    if (currentName?.toLowerCase().includes(name.toLowerCase())) {
      return name
    }
  }
  return 'Unknown'
}

// Parses many date shapes into MM/YY, or null if nothing matches. Mirrors the
// regex ladder in standardize_date from step3_cleanup_funding_data.py.
export const standardizeDate = (input: string | null | undefined): string | null => {
  if (input === null || input === undefined) {
    return null
  }
  const dateString = String(input).trim()
  const pad = (value: string): string => value.padStart(2, '0')
  // Captured groups are guaranteed present once a pattern matches, but
  // noUncheckedIndexedAccess types them as possibly-undefined, so read safely.
  const cap = (match: RegExpMatchArray, index: number): string => match[index] ?? ''

  const attempts: [RegExp, (match: RegExpMatchArray) => string][] = [
    // MM/YY or MM/YYYY
    [/^(\d{1,2})[/.-](\d{2}(?:\d{2})?)$/, (m) => `${pad(cap(m, 1))}/${cap(m, 2).slice(-2)}`],
    // YYYY only
    [/^(\d{4})$/, (m) => `01/${cap(m, 1).slice(-2)}`],
    // YYYY-MM
    [/^(\d{4})[/.-](\d{1,2})$/, (m) => `${pad(cap(m, 2))}/${cap(m, 1).slice(-2)}`],
    // YY/MM or YY-MM
    [/^(\d{2})[/.-](\d{1,2})$/, (m) => `${pad(cap(m, 2))}/${cap(m, 1)}`],
    // MM/DD/YYYY or MM-DD-YYYY
    [/^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$/, (m) => `${pad(cap(m, 1))}/${cap(m, 3).slice(-2)}`],
    // YYYY-MM-DD
    [/^(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})$/, (m) => `${pad(cap(m, 2))}/${cap(m, 1).slice(-2)}`],
  ]

  for (const [pattern, formatter] of attempts) {
    const match = dateString.match(pattern)
    if (match) {
      return formatter(match)
    }
  }

  // Textual dates like "March 3, 2020" or "March 3 2020".
  const textualMatch = dateString.match(/^(\w+)\s+(\d{1,2}),?\s+(\d{4})$/)
  if (textualMatch) {
    const parsed = new Date(`${textualMatch[1]} ${textualMatch[2]}, ${textualMatch[3]}`)
    if (!Number.isNaN(parsed.getTime())) {
      return `${pad(String(parsed.getMonth() + 1))}/${String(parsed.getFullYear()).slice(-2)}`
    }
  }

  // Last resort: ISO / Date-parseable strings.
  const isoParsed = new Date(dateString)
  if (!Number.isNaN(isoParsed.getTime())) {
    return `${pad(String(isoParsed.getMonth() + 1))}/${String(isoParsed.getFullYear()).slice(-2)}`
  }

  return null
}

// Normalizes round names and drops schema-junk objects. Mirrors check_round_names.
const normalizeRoundNames = (fundingStrings: string[]): RawFundingRound[][] =>
  fundingStrings.map((fundingString) => {
    const rounds = JSON.parse(fundingString) as RawFundingRound[]
    const cleaned: RawFundingRound[] = []
    for (const round of rounds) {
      if (round.type && round.properties) {
        continue
      }
      let roundName = round.round_name
      if (!roundName || !fundingRoundNames.includes(roundName)) {
        roundName = fixRoundName(roundName)
      }
      cleaned.push({ ...round, round_name: roundName })
    }
    return cleaned
  })

// De-duplicates rounds per round_name, keeping the largest amount and a
// standardized date. Mirrors standardize_funding_data.
export const standardizeFundingData = (fundingStrings: string[]): StandardizedFundingRound[] => {
  const normalized = normalizeRoundNames(fundingStrings)
  const byRoundName = new Map<string, StandardizedFundingRound>()

  for (const roundList of normalized) {
    for (const round of roundList) {
      const roundName = round.round_name as string
      const existing = byRoundName.get(roundName) ?? {
        round_name: roundName,
        amount: 0,
        date: null,
      }

      if (existing.currency_symbol == null && round.currency_symbol != null) {
        existing.currency_symbol = round.currency_symbol
      }
      if (existing.date == null && round.date != null) {
        existing.date = standardizeDate(round.date)
      }
      const incomingAmount = typeof round.amount === 'number' ? round.amount : 0
      existing.amount = Math.trunc(Math.max(incomingAmount, existing.amount))
      existing.round_name = roundName

      if (existing.date != null) {
        byRoundName.set(roundName, existing)
      }
    }
  }

  return [...byRoundName.values()]
}

// Aggregates standardized rounds into total amount per four-digit year, for the
// JSON sidecar that replaces the Python matplotlib/plotly plot.
export const aggregateAmountByYear = (
  rounds: StandardizedFundingRound[],
): Record<string, number> => {
  const byYear: Record<string, number> = {}
  for (const round of rounds) {
    if (!round.date) {
      continue
    }
    const yearSuffix = round.date.split('/')[1]
    if (!yearSuffix) {
      continue
    }
    const fullYear = `20${yearSuffix}`
    byYear[fullYear] = (byYear[fullYear] ?? 0) + round.amount
  }
  return byYear
}
