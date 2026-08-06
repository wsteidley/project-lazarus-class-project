import type { CsvRow } from './csv.js'

// Baseline rule #1 enforcement: an observation is only comparable to a bar declared on the
// same currency, the same energy denominator and the same duration.
//
// The progress view's equality predicates already refuse to join a mismatched pair — but on
// their own that refusal is INVISIBLE. A bar in real_2025_usd against a series in real_usd
// simply yields zero progress rows, which downstream reads as "no data", and P3 says a gap and
// a mismatch must never look alike. So the join gets an explicit companion check that fails
// the build loudly.
//
// This is the only place in the pipeline that hard-fails on data. Everywhere else the standing
// rule is report-not-drop, and deliberately so. The difference is that a report-not-drop basis
// mismatch is a silently WRONG viability verdict rather than a missing one, and P4 forbids
// publishing a call whose basis cannot be trusted.

export type BasisTriple = { basis: string; energy_basis: string; duration: string }

// A bar, plus the entity whose series it is meant to judge. Post-split the two are reached
// through dependency_links: the bar is declared on a dependency, the observations hang off an
// entity, and the link is what says they are about the same thing. Matching on dependency name
// alone would compare a bar against nothing at all.
export type BarKey = BasisTriple & {
  dependency_name: string
  entity_name: string
  metric: string
  scope: string
}

export type BasisMismatch = {
  dependency_name: string
  metric: string
  scope: string
  bar: BasisTriple
  series: BasisTriple
  n_obs: number
}

const tripleOf = (row: {
  basis?: string
  energy_basis?: string
  duration?: string
}): BasisTriple => ({
  basis: row.basis || 'na',
  energy_basis: row.energy_basis || 'na',
  duration: row.duration || 'na',
})

const sameTriple = (a: BasisTriple, b: BasisTriple): boolean =>
  a.basis === b.basis && a.energy_basis === b.energy_basis && a.duration === b.duration

export const formatTriple = (triple: BasisTriple): string =>
  `${triple.basis}/${triple.energy_basis}/${triple.duration}`

// Pure. For each bar, group the observations sharing its (dependency, metric, scope) by their
// distinct basis triple and return every group that disagrees with the bar.
//
// A bar that joins NO observations at all is not a mismatch — that is `no_blocker_data`, a
// legitimate and already-labelled state (two of the eleven curated bars are in it today).
// Failing the build on those would be punishing the dataset for being honest about a gap.
export const findBasisMismatches = (bars: BarKey[], observations: CsvRow[]): BasisMismatch[] => {
  const byKey = new Map<string, Map<string, { triple: BasisTriple; count: number }>>()
  for (const row of observations) {
    const key = `${row.entity_name ?? ''}::${row.metric ?? ''}::${row.scope ?? ''}`
    const triple = tripleOf(row)
    const groups = byKey.get(key) ?? new Map()
    const existing = groups.get(formatTriple(triple))
    if (existing) {
      existing.count += 1
    } else {
      groups.set(formatTriple(triple), { triple, count: 1 })
    }
    byKey.set(key, groups)
  }

  const mismatches: BasisMismatch[] = []
  for (const bar of bars) {
    const groups = byKey.get(`${bar.entity_name}::${bar.metric}::${bar.scope}`)
    if (groups === undefined) {
      continue
    }
    for (const { triple, count } of groups.values()) {
      if (!sameTriple(bar, triple)) {
        mismatches.push({
          dependency_name: bar.dependency_name,
          metric: bar.metric,
          scope: bar.scope,
          bar: tripleOf(bar),
          series: triple,
          n_obs: count,
        })
      }
    }
  }
  return mismatches
}

// Throws with every offending pair named. Called from the loader after observations and
// thresholds are in but before any view is read.
export const assertNoBasisMismatches = (bars: BarKey[], observations: CsvRow[]): void => {
  const mismatches = findBasisMismatches(bars, observations)
  if (mismatches.length === 0) {
    return
  }
  for (const mismatch of mismatches) {
    console.error(
      `  BASIS MISMATCH "${mismatch.dependency_name}" ${mismatch.metric}/${mismatch.scope}: ` +
        `bar ${formatTriple(mismatch.bar)} vs ${mismatch.n_obs} observation(s) ` +
        `${formatTriple(mismatch.series)}`,
    )
  }
  throw new Error(
    `${mismatches.length} threshold/observation basis mismatch(es) — see above. A viability ` +
      `call across mismatched bases is silently wrong, so this fails the build rather than ` +
      `producing one. Fix the bar or the series.`,
  )
}
