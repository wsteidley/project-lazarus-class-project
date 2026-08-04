import { DatabaseSync } from 'node:sqlite'
import { beforeAll, describe, expect, it } from 'vitest'
import { readCsv } from './csv.js'
import { createTablesSql } from './db-schema.js'
import { curatedFile, derivedFile } from './paths.js'
import { trajectoryConfig } from './trajectory-config.js'
import { computeWrightProjections, type SeriesPoint, type WrightSeries } from './wright.js'

// Loads the COMMITTED derived + curated data into the COMMITTED schema and asserts what falls
// out. The other derivation tests use synthetic rows to prove the SQL is correct in the
// abstract; this one asks a different question -- does the data we actually ship produce the
// crossings and learning rates we claim it does?
//
// It exists because `npm run build` cannot run in a checkout: build-db needs a pipeline run
// directory with companies.csv, which is LLM output and is not committed. Without this, the
// metric half of the build has no end-to-end coverage at all.

type TrajectoryRow = {
  metric: string
  segment: string
  state: string
  currently_crossed: number
  became_viable_date: string | null
  n_obs: number
}

let db: DatabaseSync
let fits: Awaited<ReturnType<typeof loadFixture>>['fits']

const loadFixture = async () => {
  const database = new DatabaseSync(':memory:')
  database.exec(createTablesSql())

  const dependencyId = new Map<string, number>()
  for (const row of await readCsv(curatedFile('dependencies.csv'))) {
    const info = database.prepare('INSERT INTO dependencies (name) VALUES (?)').run(row.name ?? '')
    dependencyId.set(row.name ?? '', Number(info.lastInsertRowid))
  }

  const insertThreshold = database.prepare(
    `INSERT OR IGNORE INTO dependency_thresholds
       (dependency_id, metric, scope, threshold_value, threshold_direction, policy_dependent,
        baseline_value)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  )
  for (const row of await readCsv(curatedFile('dependency_thresholds.csv'))) {
    const id = dependencyId.get(row.dependency_name ?? '')
    if (id === undefined || !row.threshold_value) {
      continue
    }
    insertThreshold.run(
      id,
      row.metric ?? '',
      row.scope ?? '',
      Number(row.threshold_value),
      row.threshold_direction ?? '',
      Number(row.policy_dependent ?? 0),
      row.baseline_value ? Number(row.baseline_value) : null,
    )
  }

  const insertObservation = database.prepare(
    `INSERT INTO metric_observations
       (dependency_id, metric, value, unit, basis, segment, as_of, scope, method)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  for (const row of await readCsv(derivedFile('metric_observations_full.csv'))) {
    const id = dependencyId.get(row.dependency_name ?? '')
    if (id === undefined) {
      continue
    }
    insertObservation.run(
      id,
      row.metric ?? '',
      Number(row.value),
      row.unit ?? '',
      row.basis ?? '',
      row.segment || 'all',
      row.as_of ?? '',
      row.scope ?? '',
      row.method ?? 'curated',
    )
  }

  const insertCapacity = database.prepare(
    `INSERT INTO capacity_series
       (dependency_id, metric, value, unit, basis, as_of, scope, scenario, method)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  for (const row of await readCsv(derivedFile('capacity_series.csv'))) {
    const id = dependencyId.get(row.technology ?? '')
    if (id === undefined) {
      continue
    }
    insertCapacity.run(
      id,
      row.metric ?? '',
      Number(row.value),
      row.unit ?? '',
      row.basis ?? '',
      row.as_of ?? '',
      row.scope ?? '',
      row.scenario ?? 'historical',
      row.method ?? 'curated',
    )
  }

  database
    .prepare('INSERT INTO trajectory_config (window_n, plateau_slope_threshold) VALUES (?, ?)')
    .run(trajectoryConfig.windowN, trajectoryConfig.plateauSlopeThreshold)

  // Same assembly build-db does: group the progress rows into series, attach each one's
  // capacity axis, fit.
  const capacityByDependency = new Map<number, SeriesPoint[]>()
  for (const row of database
    .prepare(
      `SELECT dependency_id, as_of, value FROM capacity_series
       WHERE scenario = 'historical' AND value IS NOT NULL`,
    )
    .all() as { dependency_id: number; as_of: string; value: number }[]) {
    const list = capacityByDependency.get(row.dependency_id) ?? []
    list.push({ as_of: row.as_of, value: row.value })
    capacityByDependency.set(row.dependency_id, list)
  }
  const series = new Map<string, WrightSeries>()
  for (const row of database
    .prepare(
      `SELECT o.dependency_id, o.metric, o.scope, o.segment, o.as_of, o.value AS value_raw,
              t.threshold_direction AS direction,
              t.threshold_value AS threshold,
              COALESCE(t.baseline_value, FIRST_VALUE(o.value) OVER (
                PARTITION BY o.dependency_id, o.metric, o.segment, o.scope ORDER BY o.as_of
              )) AS baseline
       FROM metric_observations o
       LEFT JOIN dependency_thresholds t
         ON t.dependency_id = o.dependency_id AND t.metric = o.metric AND t.scope = o.scope
        AND t.threshold_value IS NOT NULL
       WHERE o.value IS NOT NULL
         AND o.dependency_id IN (SELECT DISTINCT dependency_id FROM capacity_series)`,
    )
    .all() as {
    dependency_id: number
    metric: string
    scope: string
    segment: string
    direction: string | null
    baseline: number | null
    threshold: number | null
    as_of: string
    value_raw: number
  }[]) {
    const key = `${row.dependency_id}::${row.metric}::${row.segment}::${row.scope}`
    const existing = series.get(key)
    if (existing) {
      existing.costPoints.push({ as_of: row.as_of, value: row.value_raw })
      continue
    }
    series.set(key, {
      dependency_id: row.dependency_id,
      metric: row.metric,
      scope: row.scope,
      segment: row.segment,
      direction: row.direction,
      baseline: row.baseline,
      threshold: row.threshold,
      costPoints: [{ as_of: row.as_of, value: row.value_raw }],
      capacityPoints: capacityByDependency.get(row.dependency_id) ?? [],
    })
  }

  return { database, fits: computeWrightProjections([...series.values()]).fits }
}

const trajectoryFor = (metric: string): TrajectoryRow[] =>
  db
    .prepare(
      `SELECT metric, segment, state, currently_crossed, became_viable_date, n_obs
       FROM trajectory WHERE metric = ? ORDER BY segment`,
    )
    .all(metric) as TrajectoryRow[]

beforeAll(async () => {
  const loaded = await loadFixture()
  db = loaded.database
  fits = loaded.fits
})

describe('committed metric data', () => {
  it('loads the full IRENA onshore wind LCOE series, not the two hand-entered anchors', () => {
    const [wind] = trajectoryFor('LCOE')
    expect(wind?.n_obs).toBe(16)
    expect(wind?.segment).toBe('onshore')
  })

  // The finding that reframes this whole load: wind did not need a forecast, it needed a
  // history. It crossed 50 USD/MWh in 2019 and has stayed under it since.
  it('dates the onshore wind crossing to 2019 from real data', () => {
    const [wind] = trajectoryFor('LCOE')
    expect(wind?.currently_crossed).toBe(1)
    expect(wind?.became_viable_date).toBe('2019-12')
  })

  it('dates the battery crossing to 2025 against the 150 USD/kWh bar', () => {
    const [battery] = trajectoryFor('battery_installed_cost')
    expect(battery?.n_obs).toBe(16)
    expect(battery?.currently_crossed).toBe(1)
    expect(battery?.became_viable_date).toBe('2025-12')
  })

  // Regression for a bar that silently joined nothing: the threshold is declared on
  // 'battery pack price', but every observation used to say 'battery pack price (BEV)' etc,
  // so this dependency had a bar, ten observations, and zero progress rows.
  it('joins the battery pack bar to its observations now the slice lives in segment', () => {
    const packs = trajectoryFor('battery pack price')
    expect(packs.map((row) => row.segment)).toEqual(['all', 'bev', 'stationary'])
    // The all-segment average is still above $100; the BEV and stationary slices are below it.
    expect(packs.find((row) => row.segment === 'all')?.currently_crossed).toBe(0)
    expect(packs.find((row) => row.segment === 'bev')?.currently_crossed).toBe(1)
  })

  it('keeps each pack slice on its own trajectory rather than merging them', () => {
    const packs = trajectoryFor('battery pack price')
    // 7 all-segment points, 2 BEV, 1 stationary — if segment were not in the partition key
    // these would collapse into one 10-point series.
    expect(packs.map((row) => row.n_obs)).toEqual([7, 2, 1])
  })

  it('recovers solar’s known learning rate, the regression anchor for the fit', () => {
    const solar = fits.find((fit) => fit.metric === 'module price')
    expect(solar?.learning_rate).toBeGreaterThan(0.25)
    expect(solar?.learning_rate).toBeLessThan(0.31)
    expect(solar?.r2).toBeGreaterThan(0.9)
  })

  // The reason wright_fits exists: battery is already past its bar, so it yields no projection
  // at all. Without somewhere to store the fit, its learning rate would be discarded with the
  // forecast that never happened.
  it('keeps the battery learning rate even though it cannot be projected', () => {
    const battery = fits.find((fit) => fit.metric === 'battery_installed_cost')
    expect(battery).toBeDefined()
    expect(battery?.n_pairs).toBeGreaterThanOrEqual(4)
    // Published Li-ion learning rates cluster near 20%. Above that band here is expected and
    // explained: the BESS axis is cumulated from 2015 additions only, so cumulative deployment
    // is understated and the rate reads high. If this ever drops far below 20%, the axis has
    // changed meaning.
    expect(battery?.learning_rate).toBeGreaterThan(0.15)
    expect(battery?.learning_rate).toBeLessThan(0.3)
  })

  // Wind LCOE fits at 43%/doubling — roughly double any published onshore-wind learning rate,
  // because LCOE falls partly through capacity-factor gains that are not manufacturing
  // learning. The learning rate comes from total_installed_cost instead (below).
  it('produces no learning rate from wind LCOE, because it is not a hardware cost', () => {
    expect(fits.find((fit) => fit.metric === 'LCOE')).toBeUndefined()
  })

  it('loads the wind installed-cost series from Fig 2.3, 2010–2025', () => {
    const rows = db
      .prepare(
        `SELECT COUNT(*) AS n, MIN(as_of) AS first, MAX(as_of) AS last, MIN(segment) AS segment
         FROM metric_observations WHERE metric = 'total_installed_cost'`,
      )
      .get() as { n: number; first: string; last: string; segment: string }
    expect(rows).toMatchObject({ n: 16, first: '2010-12', last: '2025-12', segment: 'onshore' })
  })

  // The payoff: wind gets a learning rate from the hardware series. 25% sits in the plausible
  // band (published onshore wind ~15–25%) against the physically-impossible 43% from LCOE —
  // same technology, same capacity axis, correct cost series.
  it('fits wind at ~25%/doubling from installed cost', () => {
    const wind = fits.find((fit) => fit.metric === 'total_installed_cost')
    expect(wind).toBeDefined()
    expect(wind?.learning_rate).toBeGreaterThan(0.24)
    expect(wind?.learning_rate).toBeLessThan(0.26)
    expect(wind?.r2).toBeGreaterThan(0.8)
    // 2025 cost has no capacity point to pair with — the capacity series ends 2024.
    expect(wind?.n_pairs).toBe(15)
  })

  // Regression for the reason this series needed the Wright input decoupled from `progress`:
  // it has no viability bar, so a threshold-joined query found nothing and reported no fit.
  it('fits wind installed cost despite it having no threshold', () => {
    const bar = db
      .prepare(`SELECT COUNT(*) AS n FROM dependency_thresholds WHERE metric = ?`)
      .get('total_installed_cost') as { n: number }
    expect(bar.n).toBe(0)
    expect(fits.find((fit) => fit.metric === 'total_installed_cost')).toBeDefined()
    // No bar means no crossing and no trajectory row — correct, not a gap.
    expect(trajectoryFor('total_installed_cost')).toEqual([])
  })
})
