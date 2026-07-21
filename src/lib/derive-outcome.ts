// Deterministic derivation of outcome_type from the v4 signal set. Clear cases
// resolve by rule; genuinely ambiguous cases take the conservative label with a
// recorded rationale (LLM adjudication is a deferred follow-up). All thresholds
// are named here so a derived label stays reproducible — tune in one place.
export const THRESHOLDS = {
  breakoutMinRaised: 100_000_000, // "large" total raise that reads as a breakout
  solidMinRaised: 10_000_000, // above this, an operating company isn't "little signal"
  recentRaiseYears: 2, // a raise this recent counts as current momentum
  staleRaiseYears: 3, // no raise in this long reads as stalled
  tooEarlyAgeYears: 4, // younger than this + little signal = too early to judge
}

// Growth/late-stage rounds that, when recent, signal a breakout trajectory.
const GROWTH_ROUNDS = new Set(['Series C', 'Series D+', 'Growth / Late Stage', 'IPO / Public'])

export type DerivationInput = {
  living_status: string
  exit_type: string | null
  exit_amount: number | null
  total_raised: number
  year_founded: number | null
  last_round_year: number | null
  last_round_name: string | null
  build_year: number
}

export type DerivationResult = {
  outcome_type: string
  outcome_rationale: string
}

const result = (outcome_type: string, outcome_rationale: string): DerivationResult => ({
  outcome_type,
  outcome_rationale,
})

const deriveOperating = (input: DerivationInput): DerivationResult => {
  const { total_raised, last_round_year, last_round_name, year_founded, build_year } = input
  const age = year_founded === null ? null : build_year - year_founded
  const yearsSinceRaise = last_round_year === null ? null : build_year - last_round_year
  const raisedRecently = yearsSinceRaise !== null && yearsSinceRaise <= THRESHOLDS.recentRaiseYears
  const growthStage = last_round_name !== null && GROWTH_ROUNDS.has(last_round_name)

  if (raisedRecently && (growthStage || total_raised >= THRESHOLDS.breakoutMinRaised)) {
    return result(
      'Breakout Success',
      `Operating with a recent ${growthStage ? 'growth-stage' : 'large'} raise ($${total_raised.toLocaleString()})`,
    )
  }
  if (
    age !== null &&
    age < THRESHOLDS.tooEarlyAgeYears &&
    total_raised < THRESHOLDS.solidMinRaised
  ) {
    return result('Too Early to Tell', `Only ${age}y old with limited funding signal`)
  }
  if (yearsSinceRaise !== null && yearsSinceRaise >= THRESHOLDS.staleRaiseYears) {
    return result('Struggling / Zombie', `No raise in ${yearsSinceRaise}y`)
  }
  if (total_raised === 0 && (age === null || age >= THRESHOLDS.tooEarlyAgeYears)) {
    return result('Struggling / Zombie', 'Operating with no recorded funding')
  }
  return result(
    'Solid Success',
    `Operating and funded ($${total_raised.toLocaleString()}) but not a breakout`,
  )
}

export const deriveOutcome = (input: DerivationInput): DerivationResult => {
  const { living_status, exit_type, exit_amount, total_raised } = input
  const status = (living_status ?? '').trim()

  if (status === 'Defunct') {
    if (exit_type === 'acquisition') {
      if (exit_amount !== null && exit_amount >= total_raised) {
        return result('Successful Exit', 'Defunct via acquisition at or above total raised')
      }
      if (exit_amount !== null) {
        return result('Soft Landing', 'Acquired below total raised')
      }
      return result('Soft Landing', 'Acquired for an undisclosed amount (conservative)')
    }
    return result('Failed', 'Defunct with no acquisition exit')
  }

  if (status === 'Acquired') {
    if (exit_amount !== null && exit_amount >= total_raised) {
      return result('Successful Exit', 'Acquired at or above total raised')
    }
    return result(
      'Soft Landing',
      exit_amount === null
        ? 'Acquired for an undisclosed amount (conservative)'
        : 'Acquired below total raised',
    )
  }

  if (status === 'Operating') {
    return deriveOperating(input)
  }

  if (status === 'Zombie') {
    return result('Struggling / Zombie', 'Living status is Zombie')
  }

  return result('Unknown', `Unrecognized or missing living_status ("${status}")`)
}
