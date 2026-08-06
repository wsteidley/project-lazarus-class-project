import { existsSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { classifyBaseline } from '../lib/baseline.js'
import { type CsvRow, readCsv } from '../lib/csv.js'
import { createTablesSql } from '../lib/db-schema.js'
import { loadDatabase } from '../lib/load-db.js'
import { curatedFile, derivedFile, latestRunDir } from '../lib/paths.js'
import { computeProjections, type ProgressPoint } from '../lib/projection.js'
import { readAllCachedDocuments } from '../lib/raw-documents.js'
import { trajectoryConfig } from '../lib/trajectory-config.js'
import { computeWrightProjections, type SeriesPoint, type WrightSeries } from '../lib/wright.js'
import { SECTOR } from '../schemas.js'

const readCsvIfExists = async (fullPath: string): Promise<CsvRow[]> => {
  if (!existsSync(fullPath)) {
    console.log(`(skipping ${fullPath} — not found)`)
    return []
  }
  return readCsv(fullPath)
}

const main = async (): Promise<void> => {
  const runDir = latestRunDir()
  // DB_FILE overrides; otherwise the DB lives inside the run folder.
  const dbFile = process.env.DB_FILE || join(runDir, 'lazarus.db')

  const companies = await readCsvIfExists(join(runDir, 'companies.csv'))
  if (companies.length === 0) {
    throw new Error(`No companies.csv in ${runDir}; run step1 first`)
  }
  const ideaSpaces = await readCsvIfExists(curatedFile('idea_spaces.csv'))
  const companyUrls = await readCsvIfExists(join(runDir, 'company_urls.csv'))
  const companySectors = await readCsvIfExists(join(runDir, 'company_sectors.csv'))
  const fundingRounds = await readCsvIfExists(join(runDir, 'funding_rounds.csv'))
  const challenges = await readCsvIfExists(join(runDir, 'challenges.csv'))
  // The canonical dependency dimension is a curated input, not a run output.
  const dependencies = await readCsvIfExists(curatedFile('dependencies.csv'))
  const companyDependencies = await readCsvIfExists(join(runDir, 'company_dependencies.csv'))
  const assessments = await readCsvIfExists(join(runDir, 'dependency_assessments.csv'))
  // Dated, cited metric facts, and the cumulative-deployment curves the Wright fit runs
  // against. Both are *derived* files: build-metric-data regenerates them from the raw
  // provider files in data/sources plus the hand-entered anchors in data/curated, so every
  // number here traces to a committed source. Kept separate from the LLM-generated
  // assessments above.
  const metricObservations = await readCsvIfExists(derivedFile('metric_observations_full.csv'))
  const capacitySeries = await readCsvIfExists(derivedFile('capacity_series.csv'))
  // v2 threshold layer: bars keyed (dependency, metric, scope), and causal links between
  // dependencies. Curated inputs.
  const thresholds = await readCsvIfExists(curatedFile('dependency_thresholds.csv'))
  const dependencyEdges = await readCsvIfExists(curatedFile('dependency_edges.csv'))
  // The technology/dependency split's two seeds: the data-bearing subjects, and which
  // dependency hangs off which of them.
  const referenceEntities = await readCsvIfExists(curatedFile('reference_entities.csv'))
  const dependencyLinks = await readCsvIfExists(curatedFile('dependency_links.csv'))
  // What was actually searched. Its absence is what keeps an empty region 'unsampled'.
  const searchCoverage = await readCsvIfExists(curatedFile('search_coverage.csv'))
  const rawDocuments = await readAllCachedDocuments()

  if (existsSync(dbFile)) {
    rmSync(dbFile)
  }
  const db = new DatabaseSync(dbFile)
  db.exec('PRAGMA foreign_keys = ON')
  db.exec(createTablesSql())

  // The load half: every CSV -> its table, in FK order. Extracted to lib/load-db.ts so the
  // P2 inclusion guarantee is testable without a run directory. Everything below this call is
  // the derivation half, which reads the loaded tables back.
  const report = loadDatabase(db, {
    ideaSpaces,
    companies,
    companyUrls,
    companySectors,
    fundingRounds,
    challenges,
    dependencies,
    referenceEntities,
    dependencyLinks,
    searchCoverage,
    thresholds,
    companyDependencies,
    assessments,
    metricObservations,
    capacitySeries,
    dependencyEdges,
    rawDocuments,
  })

  // Baseline guards (report-not-drop): a bar whose baseline (declared, or earliest observation)
  // can't express a 0->1 range yields null progress, so surface why. Crossing is unaffected —
  // the trajectory view tests the value against the bar directly.
  const baselineRows = db
    .prepare(
      `SELECT t.dependency_id, t.metric, t.scope, t.threshold_direction AS direction,
              t.threshold_value AS threshold,
              COALESCE(t.baseline_value, (
                SELECT o.value FROM metric_observations o
                WHERE o.dependency_id = t.dependency_id AND o.metric = t.metric AND o.scope = t.scope
                ORDER BY o.as_of LIMIT 1)) AS baseline
       FROM dependency_thresholds t
       WHERE t.threshold_value IS NOT NULL`,
    )
    .all() as { direction: string; threshold: number; baseline: number | null; metric: string }[]
  let baselineEquals = 0
  let baselinePast = 0
  for (const row of baselineRows) {
    if (row.baseline === null) {
      continue
    }
    const status = classifyBaseline({
      direction: row.direction,
      threshold: row.threshold,
      baseline: row.baseline,
    })
    if (status === 'baseline_equals_threshold') {
      baselineEquals += 1
      console.warn(`  baseline == threshold (progress null) for "${row.metric}"`)
    } else if (status === 'baseline_past_threshold') {
      baselinePast += 1
      console.warn(`  baseline already past threshold (progress null) for "${row.metric}"`)
    }
  }

  // metric_projections: the one derived layer that is real computation, not a view. Read the
  // progress view back, fit each not-yet-crossed series, and store the projected points.
  //
  // Two models, and which one ran is recorded per row. Wright (cost vs cumulative capacity)
  // is the physically-motivated fit and wins wherever it can run; the linear-in-time fit
  // covers everything else. Today that means Wright fits solar and linear fits the rest —
  // wind LCOE has 2 cost points and battery has 1 capacity point, so neither can support a
  // learning curve yet. The fallback is the common path, not the exception.
  //
  // Read from metric_observations, NOT from the progress view. progress inner-joins
  // dependency_thresholds, so sourcing from it silently limited Wright to series that have a
  // curated bar — and a learning rate does not need one. Wind's total_installed_cost is the
  // series that exposed this: it is the Wright-fittable wind curve and it has no viability
  // bar, so the old query would have found nothing and reported no fit rather than an error.
  // The bar is LEFT JOINed because the projection still needs it; the fit does not.
  //
  // baseline mirrors the progress view's rule (declared baseline, else earliest observation)
  // so a wright projection lands on exactly the same 0->1 axis as a linear one.
  const wrightInput = db
    .prepare(
      `SELECT o.entity_id, dl.dependency_name, d.id AS dependency_id,
              o.metric, o.scope, o.segment, o.as_of, o.value AS value_raw,
              t.threshold_direction AS direction,
              t.threshold_value AS threshold,
              COALESCE(t.baseline_value, FIRST_VALUE(o.value) OVER (
                PARTITION BY o.entity_id, o.metric, o.segment, o.scope ORDER BY o.as_of
              )) AS baseline
       FROM metric_observations o
       -- The bar hop is LEFT all the way down: a fit needs a curve, not a bar, and not even a
       -- dependency. Wind's total_installed_cost is the case that proved it -- Wright-fittable,
       -- no viability bar -- and post-split five more technologies have curves with no
       -- dependency at all. They still get a learning rate.
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
    dependency_id: number | null
    metric: string
    scope: string
    segment: string
    direction: string | null
    baseline: number | null
    threshold: number | null
    as_of: string
    value_raw: number
  }[]
  const capacityByEntity = new Map<number, SeriesPoint[]>()
  for (const row of db
    .prepare(
      // Historical only: a scenario row is a forecast, and fitting a forecast would launder
      // an assumption into evidence.
      `SELECT entity_id, as_of, value FROM capacity_series
       WHERE scenario = 'historical' AND value IS NOT NULL`,
    )
    .all() as { entity_id: number; as_of: string; value: number }[]) {
    const list = capacityByEntity.get(row.entity_id) ?? []
    list.push({ as_of: row.as_of, value: row.value })
    capacityByEntity.set(row.entity_id, list)
  }
  const wrightSeries = new Map<string, WrightSeries>()
  for (const row of wrightInput) {
    // segment is part of the key: without it an onshore and an offshore series, or a BEV and an
    // all-segment pack price, would pour their points into one costPoints array and be fitted
    // as a single curve.
    // Keyed on the ENTITY: a learning rate is a property of the curve, so three dependencies
    // sharing one curve must not fit it three times (and the LEFT JOIN above emits one row per
    // link, so without this they would).
    const key = `${row.entity_id}::${row.metric}::${row.segment}::${row.scope}`
    const existing = wrightSeries.get(key)
    if (existing) {
      existing.costPoints.push({ as_of: row.as_of, value: row.value_raw })
      continue
    }
    wrightSeries.set(key, {
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
  const wright = computeWrightProjections([...wrightSeries.values()])
  for (const entry of wright.skipped) {
    console.warn(`  wright projection skipped for "${entry.metric}": ${entry.reason}`)
  }

  // wright_fits: the learning rates, stored whether or not a projection followed. An
  // already-crossed series (onshore wind 2019, battery 2025) yields a fit and no projection —
  // recording it here is what keeps that finding rather than discarding it with the forecast.
  const insertWrightFit = db.prepare(
    `INSERT OR REPLACE INTO wright_fits
      (entity_id, metric, scope, segment, learning_rate, b, r2, n_pairs, first_as_of, last_as_of)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  for (const fit of wright.fits) {
    insertWrightFit.run(
      fit.entity_id,
      fit.metric,
      fit.scope,
      fit.segment,
      fit.learning_rate,
      fit.b,
      fit.r2,
      fit.n_pairs,
      fit.first_as_of,
      fit.last_as_of,
    )
    console.log(
      `  wright fit "${fit.metric}" (${fit.segment}): ${(fit.learning_rate * 100).toFixed(0)}%/doubling, ` +
        `R2=${fit.r2.toFixed(2)}, ${fit.n_pairs} pairs (${fit.first_as_of}..${fit.last_as_of})`,
    )
  }

  // Series Wright already covered are excluded from the linear pass, so a series never
  // carries two competing projections.
  const wrightCovered = new Set(
    wright.rows.map((row) => `${row.entity_id}::${row.metric}::${row.segment}::${row.scope}`),
  )
  const progressRows = (
    db
      .prepare(
        'SELECT entity_id, dependency_id, metric, scope, segment, as_of, progress FROM progress',
      )
      .all() as ProgressPoint[]
  ).filter(
    (row) => !wrightCovered.has(`${row.entity_id}::${row.metric}::${row.segment}::${row.scope}`),
  )
  const projections = [
    ...wright.rows,
    ...computeProjections(progressRows, { windowN: trajectoryConfig.windowN }),
  ]
  const insertProjection = db.prepare(
    `INSERT INTO metric_projections
      (entity_id, dependency_id, metric, scope, segment, as_of, progress, projected, method,
       fit_window_n, confidence, is_crossing, note)
     VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)`,
  )
  for (const row of projections) {
    insertProjection.run(
      row.entity_id,
      row.dependency_id ?? null,
      row.metric,
      row.scope,
      row.segment,
      row.as_of,
      row.progress,
      row.method,
      row.fit_window_n,
      row.confidence,
      row.is_crossing,
      row.note,
    )
  }

  // P3 state histograms. Printed every build so a regression is loud: if the metric side
  // silently stops joining, everything slides to no_blocker_data here rather than quietly
  // becoming an empty join nobody sees.
  const absenceHistogram = (view: string): string => {
    const rows = db
      .prepare(`SELECT status, COUNT(*) AS n FROM ${view} GROUP BY status ORDER BY n DESC`)
      .all() as { status: string; n: number }[]
    return rows.map((row) => `${row.status}=${row.n}`).join(', ') || '(none)'
  }
  console.log(`  dependency states: ${absenceHistogram('dependency_status')}`)
  console.log(`  company x blocker states: ${absenceHistogram('company_dependency_status')}`)
  const gapCells = db
    .prepare('SELECT cell AS status, COUNT(*) AS n FROM gap_map GROUP BY cell ORDER BY n DESC')
    .all() as { status: string; n: number }[]
  console.log(
    `  gap map: ${gapCells.map((row) => `${row.status}=${row.n}`).join(', ') || '(none)'}`,
  )

  db.close()

  console.log(
    `Built ${dbFile}: ${SECTOR.length} sectors, ${ideaSpaces.length} idea_spaces, ` +
      `${companies.length} companies, ${companyUrls.length} company_urls, ` +
      `${companySectors.length} company_sectors, ` +
      `${fundingRounds.length} funding_rounds, ${challenges.length} challenges, ` +
      `${dependencies.length} dependencies, ${report.thresholdsInserted} dependency_thresholds, ` +
      `${report.companyDependenciesRetained} company_dependencies ` +
      `(${report.companyDependenciesUnresolved} unresolved, retained), ` +
      `${assessments.length} dependency_assessments, ${report.observationsInserted} metric_observations, ` +
      `${report.capacityInserted} capacity_series, ` +
      `${report.entitiesInserted} reference_entities, ${report.linksInserted} dependency_links, ` +
      `${report.edgesInserted} dependency_edges, ${projections.length} metric_projections ` +
      `(${wright.rows.length} wright, ${projections.length - wright.rows.length} linear), ` +
      `${rawDocuments.length} raw_documents` +
      (baselineEquals || baselinePast
        ? ` (${baselineEquals} baseline==threshold, ${baselinePast} baseline-past-threshold — progress null)`
        : ''),
  )
  // Named, per-table and per-reason. The old single "N child rows skipped" counter told you
  // something was lost and never what — the same unqueryable blank P3 forbids in the data.
  if (report.skipped.size > 0) {
    console.warn('\nRows not loaded:')
    for (const [reason, count] of [...report.skipped].sort()) {
      console.warn(`  ${reason} (${count})`)
    }
  }
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
