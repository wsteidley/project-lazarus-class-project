import { DatabaseSync } from 'node:sqlite'
import { describe, expect, it } from 'vitest'
import { assertNoBasisMismatches, type BarKey, findBasisMismatches } from './basis.js'
import type { CsvRow } from './csv.js'
import { createTablesSql } from './db-schema.js'
import { type LoadInputs, loadDatabase } from './load-db.js'

// Baseline rule #1. The progress join's equality predicates already refuse a mismatched pair,
// but that refusal is invisible — zero rows reads as "no data". These tests pin the companion
// check that makes it loud, and pin the one case that must NOT be loud (a bar with no series
// at all is a labelled gap, not an error).

const bar = (overrides: Partial<BarKey> = {}): BarKey => ({
  dependency_name: 'Utility-scale battery system',
  entity_name: 'Lithium-ion battery cost',
  metric: 'battery_installed_cost',
  scope: 'global',
  basis: 'real_2025_usd',
  energy_basis: 'usable',
  duration: 'blended',
  ...overrides,
})

const obs = (overrides: CsvRow = {}): CsvRow => ({
  entity_name: 'Lithium-ion battery cost',
  metric: 'battery_installed_cost',
  scope: 'global',
  basis: 'real_2025_usd',
  energy_basis: 'usable',
  duration: 'blended',
  ...overrides,
})

describe('findBasisMismatches', () => {
  it('accepts a bar and series that agree on all three axes', () => {
    expect(findBasisMismatches([bar()], [obs(), obs()])).toEqual([])
  })

  // The currency case: a 2025-dollar bar against an undated-real series is not a comparison
  // anyone can defend, and `real_usd` is deliberately not promoted to a vintage we'd invent.
  it('flags a currency-vintage disagreement', () => {
    const [mismatch] = findBasisMismatches([bar()], [obs({ basis: 'real_usd' })])
    expect(mismatch).toMatchObject({
      dependency_name: 'Utility-scale battery system',
      bar: { basis: 'real_2025_usd' },
      series: { basis: 'real_usd' },
      n_obs: 1,
    })
  })

  it('flags an energy_basis-only disagreement', () => {
    const found = findBasisMismatches([bar({ energy_basis: 'nameplate' })], [obs()])
    expect(found).toHaveLength(1)
    expect(found[0]?.series.energy_basis).toBe('usable')
  })

  it('flags a duration-only disagreement', () => {
    const found = findBasisMismatches([bar({ duration: '4h' })], [obs()])
    expect(found).toHaveLength(1)
    expect(found[0]?.series.duration).toBe('blended')
  })

  it('treats na as matching only na, never as a wildcard', () => {
    // The whole reason the columns are NOT NULL DEFAULT 'na': an unstamped bar joins an
    // unstamped series and nothing else.
    expect(
      findBasisMismatches(
        [bar({ basis: 'na', energy_basis: 'na', duration: 'na' })],
        [obs({ basis: 'na', energy_basis: 'na', duration: 'na' })],
      ),
    ).toEqual([])
    expect(
      findBasisMismatches(
        [bar({ basis: 'na', energy_basis: 'na', duration: 'na' })],
        [obs({ basis: 'real_2025_usd', energy_basis: 'na', duration: 'na' })],
      ),
    ).toHaveLength(1)
  })

  // no_blocker_data, not an error. Two of the eleven curated bars are in exactly this state,
  // and failing the build on them would punish the dataset for being honest about a gap.
  it('does not flag a bar whose dependency has no observations at all', () => {
    expect(findBasisMismatches([bar({ entity_name: 'Permitting timelines' })], [])).toEqual([])
  })

  it('ignores observations on a different metric or scope', () => {
    expect(
      findBasisMismatches([bar()], [obs({ metric: 'something else', basis: 'nominal_usd' })]),
    ).toEqual([])
    expect(findBasisMismatches([bar()], [obs({ scope: 'US', basis: 'nominal_usd' })])).toEqual([])
  })

  it('reports one entry per distinct offending triple, with its observation count', () => {
    const found = findBasisMismatches(
      [bar()],
      [obs({ basis: 'real_usd' }), obs({ basis: 'real_usd' }), obs({ duration: '4h' })],
    )
    expect(found).toHaveLength(2)
    expect(found.map((entry) => entry.n_obs).sort()).toEqual([1, 2])
  })
})

describe('assertNoBasisMismatches', () => {
  it('passes silently when everything agrees', () => {
    expect(() => assertNoBasisMismatches([bar()], [obs()])).not.toThrow()
  })

  // The one place the pipeline hard-fails on data. Everywhere else the rule is
  // report-not-drop; here a report would mean publishing a silently wrong verdict.
  it('throws, naming both triples, when they do not', () => {
    expect(() => assertNoBasisMismatches([bar()], [obs({ energy_basis: 'nameplate' })])).toThrow(
      /basis mismatch/i,
    )
  })
})

// End-to-end through the loader, because the check being CORRECT and the check being WIRED
// are different claims. Flipping one bar's currency is the manual verification recipe for
// this stage, run here so it cannot rot.
describe('the loader refuses to build across a basis mismatch', () => {
  const inputs = (thresholdBasis: string): LoadInputs => ({
    ideaSpaces: [],
    companies: [],
    companyUrls: [],
    companySectors: [],
    fundingRounds: [],
    challenges: [],
    dependencies: [{ name: 'Solar module cost', threshold_kind: 'quantitative_with_threshold' }],
    referenceEntities: [{ name: 'Solar PV', kind: 'technology' }],
    dependencyLinks: [{ dependency_name: 'Solar module cost', entity_name: 'Solar PV' }],
    searchCoverage: [],
    thresholds: [
      {
        dependency_name: 'Solar module cost',
        metric: 'module price',
        scope: 'global',
        threshold_value: '0.2',
        threshold_unit: 'USD/W',
        threshold_direction: 'below_is_better',
        threshold_source_url: 'https://example.com/bar',
        basis: thresholdBasis,
        energy_basis: 'na',
        duration: 'na',
      },
    ],
    companyDependencies: [],
    assessments: [],
    metricObservations: [
      {
        entity_name: 'Solar PV',
        metric: 'module price',
        scope: 'global',
        value: '0.26',
        unit: 'USD/W',
        basis: 'real_2024_usd',
        energy_basis: 'na',
        duration: 'na',
        as_of: '2024-12',
        method: 'curated',
        source_url: 'https://example.com/series',
      },
    ],
    capacitySeries: [],
    dependencyEdges: [],
    rawDocuments: [],
  })

  const freshDb = (): DatabaseSync => {
    const db = new DatabaseSync(':memory:')
    db.exec('PRAGMA foreign_keys = ON')
    db.exec(createTablesSql())
    return db
  }

  it('loads when the bar and the series agree', () => {
    const db = freshDb()
    expect(() => loadDatabase(db, inputs('real_2024_usd'))).not.toThrow()
    expect(db.prepare('SELECT COUNT(*) AS n FROM progress').get()).toEqual({ n: 1 })
    db.close()
  })

  it('fails the build when they do not, rather than yielding an empty progress view', () => {
    const db = freshDb()
    // The failure mode this replaces: no throw, zero progress rows, and a dependency that
    // reads as no_blocker_data despite having both a bar and a series.
    expect(() => loadDatabase(db, inputs('nominal_usd'))).toThrow(/basis mismatch/i)
    db.close()
  })
})
