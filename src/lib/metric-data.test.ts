import { DatabaseSync } from 'node:sqlite'
import { beforeAll, describe, expect, it } from 'vitest'
import { findBasisMismatches } from './basis.js'
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

  // Post-split: metric data hangs off entities, and reaches a dependency's bar through the link.
  const entityId = new Map<string, number>()
  for (const row of await readCsv(curatedFile('reference_entities.csv'))) {
    const info = database
      .prepare('INSERT INTO reference_entities (name, kind) VALUES (?, ?)')
      .run(row.name ?? '', row.kind ?? 'technology')
    entityId.set(row.name ?? '', Number(info.lastInsertRowid))
  }
  for (const row of await readCsv(curatedFile('dependency_links.csv'))) {
    const id = entityId.get(row.entity_name ?? '')
    if (id === undefined || !dependencyId.has(row.dependency_name ?? '')) {
      continue
    }
    database
      .prepare('INSERT INTO dependency_links (dependency_name, entity_id, era) VALUES (?, ?, ?)')
      .run(row.dependency_name ?? '', id, row.era || null)
  }

  const insertThreshold = database.prepare(
    `INSERT OR IGNORE INTO dependency_thresholds
       (dependency_id, metric, scope, threshold_value, threshold_direction, policy_dependent,
        baseline_value, basis, energy_basis, duration)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
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
      // The bar's basis triple is now part of the progress join, so the fixture has to carry
      // it or every bar defaults to na/na/na, joins nothing, and every assertion below goes
      // silently empty.
      row.basis || 'na',
      row.energy_basis || 'na',
      row.duration || 'na',
    )
  }

  const insertObservation = database.prepare(
    `INSERT INTO metric_observations
       (entity_id, metric, value, unit, basis, energy_basis, duration, segment, as_of,
        scope, method)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  for (const row of await readCsv(derivedFile('metric_observations_full.csv'))) {
    const id = entityId.get(row.entity_name ?? '')
    if (id === undefined) {
      continue
    }
    insertObservation.run(
      id,
      row.metric ?? '',
      Number(row.value),
      row.unit ?? '',
      row.basis || 'na',
      row.energy_basis || 'na',
      row.duration || 'na',
      row.segment || 'all',
      row.as_of ?? '',
      row.scope ?? '',
      row.method ?? 'curated',
    )
  }

  const insertCapacity = database.prepare(
    `INSERT INTO capacity_series
       (entity_id, metric, value, unit, basis, energy_basis, duration, segment, as_of,
        scope, scenario, method)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  for (const row of await readCsv(derivedFile('capacity_series.csv'))) {
    const id = entityId.get(row.entity_name ?? '')
    if (id === undefined) {
      continue
    }
    insertCapacity.run(
      id,
      row.metric ?? '',
      Number(row.value),
      row.unit ?? '',
      row.basis || 'na',
      row.energy_basis || 'na',
      row.duration || 'na',
      row.segment || 'all',
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
  const capacityByEntity = new Map<number, SeriesPoint[]>()
  for (const row of database
    .prepare(
      `SELECT entity_id, as_of, value FROM capacity_series
       WHERE scenario = 'historical' AND value IS NOT NULL`,
    )
    .all() as { entity_id: number; as_of: string; value: number }[]) {
    const list = capacityByEntity.get(row.entity_id) ?? []
    list.push({ as_of: row.as_of, value: row.value })
    capacityByEntity.set(row.entity_id, list)
  }
  const series = new Map<string, WrightSeries>()
  for (const row of database
    .prepare(
      `SELECT o.entity_id, o.metric, o.scope, o.segment, o.as_of, o.value AS value_raw,
              t.threshold_direction AS direction,
              t.threshold_value AS threshold,
              COALESCE(t.baseline_value, FIRST_VALUE(o.value) OVER (
                PARTITION BY o.entity_id, o.metric, o.segment, o.scope ORDER BY o.as_of
              )) AS baseline
       FROM metric_observations o
       LEFT JOIN dependency_links dl ON dl.entity_id = o.entity_id
       LEFT JOIN dependencies d      ON d.name = dl.dependency_name
       LEFT JOIN dependency_thresholds t
         ON t.dependency_id = d.id AND t.metric = o.metric AND t.scope = o.scope
        AND t.threshold_value IS NOT NULL
        AND t.basis = o.basis AND t.energy_basis = o.energy_basis AND t.duration = o.duration
       WHERE o.value IS NOT NULL
         AND o.entity_id IN (SELECT DISTINCT entity_id FROM capacity_series)`,
    )
    .all() as {
    entity_id: number
    metric: string
    scope: string
    segment: string
    direction: string | null
    baseline: number | null
    threshold: number | null
    as_of: string
    value_raw: number
  }[]) {
    const key = `${row.entity_id}::${row.metric}::${row.segment}::${row.scope}`
    const existing = series.get(key)
    if (existing) {
      existing.costPoints.push({ as_of: row.as_of, value: row.value_raw })
      continue
    }
    series.set(key, {
      entity_id: row.entity_id,
      metric: row.metric,
      scope: row.scope,
      segment: row.segment,
      direction: row.direction,
      baseline: row.baseline,
      threshold: row.threshold,
      costPoints: [{ as_of: row.as_of, value: row.value_raw }],
      capacityPoints: capacityByEntity.get(row.entity_id) ?? [],
    })
  }

  return { database, fits: computeWrightProjections([...series.values()]).fits }
}

// Takes the ENTITY as well as the metric. Post-split 'LCOE' spans seven technologies, so the
// old metric-only lookup would have returned whichever sorted first and quietly asserted the
// wrong technology's numbers.
const trajectoryFor = (entityName: string, metric: string): TrajectoryRow[] =>
  db
    .prepare(
      `SELECT t.metric, t.segment, t.state, t.currently_crossed, t.became_viable_date, t.n_obs
       FROM trajectory t JOIN reference_entities e ON e.id = t.entity_id
       WHERE e.name = ? AND t.metric = ? ORDER BY t.segment`,
    )
    .all(entityName, metric) as TrajectoryRow[]

beforeAll(async () => {
  const loaded = await loadFixture()
  db = loaded.database
  fits = loaded.fits
})

describe('committed metric data', () => {
  it('loads the full IRENA onshore wind LCOE series, not the two hand-entered anchors', () => {
    const [wind] = trajectoryFor('Onshore wind', 'LCOE')
    expect(wind?.n_obs).toBe(16)
    expect(wind?.segment).toBe('onshore')
  })

  // The finding that reframes this whole load: wind did not need a forecast, it needed a
  // history. It crossed 50 USD/MWh in 2019 and has stayed under it since.
  it('dates the onshore wind crossing to 2019 from real data', () => {
    const [wind] = trajectoryFor('Onshore wind', 'LCOE')
    expect(wind?.currently_crossed).toBe(1)
    expect(wind?.became_viable_date).toBe('2019-12')
  })

  it('dates the battery crossing to 2025 against the 150 USD/kWh bar', () => {
    const [battery] = trajectoryFor('Lithium-ion battery cost', 'battery_installed_cost')
    expect(battery?.n_obs).toBe(16)
    expect(battery?.currently_crossed).toBe(1)
    expect(battery?.became_viable_date).toBe('2025-12')
  })

  // Regression for a bar that silently joined nothing: the threshold is declared on
  // 'battery pack price', but every observation used to say 'battery pack price (BEV)' etc,
  // so this dependency had a bar, ten observations, and zero progress rows.
  it('joins the battery pack bar to its observations now the slice lives in segment', () => {
    const packs = trajectoryFor('Lithium-ion battery cost', 'battery pack price')
    expect(packs.map((row) => row.segment)).toEqual(['all', 'bev', 'stationary'])
    // The all-segment average is still above $100; the BEV and stationary slices are below it.
    expect(packs.find((row) => row.segment === 'all')?.currently_crossed).toBe(0)
    expect(packs.find((row) => row.segment === 'bev')?.currently_crossed).toBe(1)
  })

  it('keeps each pack slice on its own trajectory rather than merging them', () => {
    const packs = trajectoryFor('Lithium-ion battery cost', 'battery pack price')
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
        `SELECT COUNT(*) AS n, MIN(o.as_of) AS first, MAX(o.as_of) AS last, MIN(o.segment) AS segment
         FROM metric_observations o JOIN reference_entities e ON e.id = o.entity_id
         WHERE o.metric = 'total_installed_cost' AND e.name = 'Onshore wind'`,
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
    expect(trajectoryFor('Onshore wind', 'total_installed_cost')).toEqual([])
  })
})

// P3 over the real curated set, not synthetic fixtures. The committed data already contains
// live instances of three of the five states, so these assertions are about what the project
// actually knows and doesn't.
describe('absence states over the committed data', () => {
  it('labels every curated dependency, never leaving a bare null', () => {
    const nulls = db
      .prepare('SELECT COUNT(*) AS n FROM dependency_status WHERE status IS NULL')
      .get() as {
      n: number
    }
    expect(nulls.n).toBe(0)
    // And every dependency in the seed gets a row — none falls out of the roll-up.
    const [deps, statuses] = [
      db.prepare('SELECT COUNT(*) AS n FROM dependencies').get() as { n: number },
      db.prepare('SELECT COUNT(*) AS n FROM dependency_status').get() as { n: number },
    ]
    expect(statuses.n).toBe(deps.n)
  })

  it('names the bars that have no series to judge no_blocker_data', () => {
    const statusOf = (name: string): string =>
      (
        db.prepare('SELECT status FROM dependency_status WHERE dependency_name = ?').get(name) as {
          status: string
        }
      )?.status

    // Both carry a curated bar and zero observations — a curation gap that is now queryable
    // rather than an empty join nobody could see.
    expect(statusOf('Permitting timelines')).toBe('no_blocker_data')
    expect(statusOf('Consumer willingness to pay green premium')).toBe('no_blocker_data')
  })

  // P4: a viability call that cannot say what it was judged against is a bug. Every progress
  // row carries its full basis triple, so a user can look at a "not viable" and disagree with
  // the measurement rather than only with the verdict.
  it('carries the full basis triple on every progress row', () => {
    const bare = db
      .prepare(
        `SELECT COUNT(*) AS n FROM progress
         WHERE basis IS NULL OR energy_basis IS NULL OR duration IS NULL`,
      )
      .get() as { n: number }
    expect(bare.n).toBe(0)
  })

  // The committed bars and their series agree on all three axes — which is what lets the
  // build-time checker stay silent. If a future curation breaks that, the build fails rather
  // than quietly producing zero progress rows that read as "no data".
  it('has no basis mismatch between any committed bar and its series', async () => {
    // Bars reach their series through the link, same as the loader does.
    const entityFor = new Map(
      (await readCsv(curatedFile('dependency_links.csv'))).map((row) => [
        row.dependency_name ?? '',
        row.entity_name ?? '',
      ]),
    )
    const bars = (await readCsv(curatedFile('dependency_thresholds.csv')))
      .filter((row) => entityFor.has(row.dependency_name ?? ''))
      .map((row) => ({
        dependency_name: row.dependency_name ?? '',
        entity_name: entityFor.get(row.dependency_name ?? '') ?? '',
        metric: row.metric ?? '',
        scope: row.scope ?? '',
        basis: row.basis || 'na',
        energy_basis: row.energy_basis || 'na',
        duration: row.duration || 'na',
      }))
    const observations = await readCsv(derivedFile('metric_observations_full.csv'))
    expect(findBasisMismatches(bars, observations)).toEqual([])
  })

  // The battery bar is the one the enforcement was written for: IRENA quotes per USABLE kWh
  // over a blended-duration fleet, and those qualifiers used to live in a note string where
  // nothing could act on them.
  it('declares the battery bar on usable kWh and blended duration', () => {
    const row = db
      .prepare(
        `SELECT basis, energy_basis, duration FROM progress
         WHERE metric = 'battery_installed_cost' LIMIT 1`,
      )
      .get() as { basis: string; energy_basis: string; duration: string }
    expect(row).toEqual({ basis: 'real_2025_usd', energy_basis: 'usable', duration: 'blended' })
  })

  it('separates the wind series with a bar from the one without', () => {
    const rows = db
      .prepare(
        `SELECT metric, status FROM series_status
         WHERE entity_name = 'Onshore wind' AND metric IN ('LCOE', 'total_installed_cost')
         ORDER BY metric`,
      )
      .all()
    // LCOE crossed its 50 USD/MWh bar in 2019; installed cost has no bar at all. Two very
    // different things to know, and the schema now says which is which.
    expect(rows).toEqual([
      { metric: 'LCOE', status: 'assessed_viable' },
      { metric: 'total_installed_cost', status: 'no_threshold' },
    ])
  })

  // The split's payoff, stated as data rather than as a row count: six of the seven published
  // LCOE curves are now loaded and describable, and every one of them is honestly labelled as
  // having no viability bar rather than being absent from the dataset entirely.
  it('loads all seven IRENA LCOE technologies, six of them without a bar', () => {
    const rows = db
      .prepare(
        `SELECT entity_name, status FROM series_status WHERE metric = 'LCOE'
         ORDER BY entity_name`,
      )
      .all() as { entity_name: string; status: string }[]
    expect(rows.map((row) => row.entity_name)).toEqual([
      'Bioenergy',
      'Concentrated solar power',
      'Geothermal',
      'Hydropower',
      'Offshore wind',
      'Onshore wind',
      'Solar PV',
    ])
    // Only onshore wind has a curated bar; the rest are a curation gap, not a dead end.
    expect(rows.filter((row) => row.status === 'no_threshold')).toHaveLength(6)
    expect(rows.find((row) => row.entity_name === 'Onshore wind')?.status).toBe('assessed_viable')
  })

  // The series the trajectory empirical check has been blocked on: geothermal and hydro are
  // the honest 'receded' cases (LCOE rising), and until the split they could not load at all.
  it('gives geothermal and hydropower real loaded curves', () => {
    const rows = db
      .prepare(
        `SELECT entity_name, n_obs FROM series_status
         WHERE metric = 'LCOE' AND entity_name IN ('Geothermal', 'Hydropower')
         ORDER BY entity_name`,
      )
      .all()
    // Geothermal is 15 points, not 16 — 2011 is missing from the published series.
    expect(rows).toEqual([
      { entity_name: 'Geothermal', n_obs: 15 },
      { entity_name: 'Hydropower', n_obs: 16 },
    ])
  })
})
