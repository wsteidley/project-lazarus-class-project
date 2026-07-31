import { existsSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { classifyBaseline } from '../lib/baseline.js'
import { type CsvRow, readCsv } from '../lib/csv.js'
import { createTablesSql } from '../lib/db-schema.js'
import { curatedFile, derivedFile, latestRunDir } from '../lib/paths.js'
import { computeProjections, type ProgressPoint } from '../lib/projection.js'
import { readAllCachedDocuments } from '../lib/raw-documents.js'
import {
  validateDependencyLinks,
  validateObservationRows,
  validateThresholds,
} from '../lib/thresholds.js'
import { trajectoryConfig } from '../lib/trajectory-config.js'
import { computeWrightProjections, type SeriesPoint, type WrightSeries } from '../lib/wright.js'
import { SECTOR } from '../schemas.js'

const toText = (value: string | undefined): string | null =>
  value === undefined || value === '' ? null : value
const toInt = (value: string | undefined): number | null => {
  if (value === undefined || value === '') {
    return null
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? Math.trunc(parsed) : null
}
// Like toInt but keeps the fractional part, for metric/threshold/score columns.
const toReal = (value: string | undefined): number | null => {
  if (value === undefined || value === '') {
    return null
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

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
  const dependencyLinks = await readCsvIfExists(curatedFile('dependency_links.csv'))
  const rawDocuments = await readAllCachedDocuments()

  if (existsSync(dbFile)) {
    rmSync(dbFile)
  }
  const db = new DatabaseSync(dbFile)
  db.exec('PRAGMA foreign_keys = ON')
  db.exec(createTablesSql())

  // sectors: reference rows seeded from the vocab. name -> id.
  const insertSector = db.prepare('INSERT INTO sectors (name) VALUES (?)')
  const sectorIdByName = new Map<string, number>()
  for (const name of SECTOR) {
    const info = insertSector.run(name)
    sectorIdByName.set(name, Number(info.lastInsertRowid))
  }

  // idea_spaces: curated seed. Resolve home sector by name. name -> id.
  const insertIdeaSpace = db.prepare(
    'INSERT INTO idea_spaces (name, sector_id, description) VALUES (?, ?, ?)',
  )
  const ideaSpaceIdByName = new Map<string, number>()
  for (const row of ideaSpaces) {
    const info = insertIdeaSpace.run(
      toText(row.name) ?? '',
      sectorIdByName.get(row.sector_name ?? '') ?? null,
      toText(row.description),
    )
    if (row.name) {
      ideaSpaceIdByName.set(row.name, Number(info.lastInsertRowid))
    }
  }

  // companies: resolve idea_space_name -> id; capture uuid -> id.
  const insertCompany = db.prepare(
    `INSERT INTO companies
      (uuid, company_name, idea_space_id, founders, canonical_uuid, merged_from,
       location, country, year_founded, year_defunct, living_status_source,
       year_founded_source, living_status, has_pivoted,
       idea_summary, exit_type, exit_amount, exit_date, exit_notes, outcome_summary,
       outcome_source_url, outcome_confidence, outcome_confidence_score,
       outcome_confidence_self_reported, outcome_contested,
       outcome_contested_note, outcome_type, outcome_rationale, original_trl, is_climate,
       source_url, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  const idByUuid = new Map<string, number>()
  for (const row of companies) {
    const info = insertCompany.run(
      toText(row.uuid),
      toText(row.company_name) ?? '',
      ideaSpaceIdByName.get(row.idea_space_name ?? '') ?? null,
      toText(row.founders),
      toText(row.canonical_uuid),
      toText(row.merged_from),
      toText(row.location),
      toText(row.country),
      toInt(row.year_founded),
      toInt(row.year_defunct),
      toText(row.living_status_source),
      toText(row.year_founded_source),
      toText(row.living_status),
      toInt(row.has_pivoted),
      toText(row.idea_summary),
      toText(row.exit_type),
      toInt(row.exit_amount),
      toText(row.exit_date),
      toText(row.exit_notes),
      toText(row.outcome_summary),
      toText(row.outcome_source_url),
      toText(row.outcome_confidence),
      toReal(row.outcome_confidence_score),
      toText(row.outcome_confidence_self_reported),
      toInt(row.outcome_contested),
      toText(row.outcome_contested_note),
      toText(row.outcome_type),
      toText(row.outcome_rationale),
      toInt(row.original_trl),
      toInt(row.is_climate),
      toText(row.source_url),
      toText(row.created_at),
    )
    if (row.uuid) {
      idByUuid.set(row.uuid, Number(info.lastInsertRowid))
    }
  }

  const companyIdFor = (row: CsvRow): number | undefined => idByUuid.get(row.company_uuid ?? '')
  let skipped = 0

  // company_urls: typed identity/reference URLs. UNIQUE(company_id, url_type, url)
  // absorbs any repeats the resolve step did not already collapse.
  const insertCompanyUrl = db.prepare(
    `INSERT OR IGNORE INTO company_urls (company_id, url_type, url, normalized_value, source_url)
     VALUES (?, ?, ?, ?, ?)`,
  )
  for (const row of companyUrls) {
    const companyId = companyIdFor(row)
    if (companyId === undefined) {
      skipped += 1
      continue
    }
    insertCompanyUrl.run(
      companyId,
      toText(row.url_type),
      toText(row.url) ?? '',
      toText(row.normalized_value),
      toText(row.source_url),
    )
  }

  // company_sectors: resolve company_uuid + sector_name.
  const insertCompanySector = db.prepare(
    'INSERT OR IGNORE INTO company_sectors (company_id, sector_id, is_primary) VALUES (?, ?, ?)',
  )
  for (const row of companySectors) {
    const companyId = companyIdFor(row)
    const sectorId = sectorIdByName.get(row.sector_name ?? '')
    if (companyId === undefined || sectorId === undefined) {
      skipped += 1
      continue
    }
    insertCompanySector.run(companyId, sectorId, toInt(row.is_primary))
  }

  const insertFunding = db.prepare(
    `INSERT INTO funding_rounds
      (company_id, round_name, amount, currency, round_date, round_year, source_url)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  )
  for (const row of fundingRounds) {
    const companyId = companyIdFor(row)
    if (companyId === undefined) {
      skipped += 1
      continue
    }
    insertFunding.run(
      companyId,
      toText(row.round_name),
      toInt(row.amount),
      toText(row.currency),
      toText(row.round_date),
      toInt(row.round_year),
      toText(row.source_url),
    )
  }

  const insertChallenge = db.prepare(
    `INSERT INTO challenges
      (company_id, category, outcome, detail, confidence, confidence_score, contested,
       contested_note, source_url)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  for (const row of challenges) {
    const companyId = companyIdFor(row)
    if (companyId === undefined) {
      skipped += 1
      continue
    }
    insertChallenge.run(
      companyId,
      toText(row.category),
      toText(row.outcome),
      toText(row.detail),
      toText(row.confidence),
      toReal(row.confidence_score),
      toInt(row.contested),
      toText(row.contested_note),
      toText(row.source_url),
    )
  }

  // dependencies: curated canonical dimension (identity + kind only; bars live in
  // dependency_thresholds). name -> id, for the tables below.
  const insertDependency = db.prepare(
    `INSERT INTO dependencies (uuid, name, category, description, threshold_kind)
     VALUES (?, ?, ?, ?, ?)`,
  )
  const dependencyIdByName = new Map<string, number>()
  for (const row of dependencies) {
    const info = insertDependency.run(
      toText(row.uuid),
      toText(row.name) ?? '',
      toText(row.category),
      toText(row.description),
      toText(row.threshold_kind),
    )
    if (row.name) {
      dependencyIdByName.set(row.name, Number(info.lastInsertRowid))
    }
  }

  // dependency_thresholds: one bar per (dependency, metric, scope). Validate names against
  // the seed (report, don't drop) and enforce that a quantitative_with_threshold dependency
  // actually carries a complete bar.
  const canonicalNames = [...dependencyIdByName.keys()]
  const {
    resolved: resolvedThresholds,
    unmatched: unmatchedThresholds,
    incomplete: incompleteThresholds,
  } = validateThresholds(dependencies, thresholds, canonicalNames)
  for (const row of unmatchedThresholds) {
    console.warn(
      `  dependency_threshold dependency_name matched nothing canonical: "${row.dependency_name ?? ''}"`,
    )
  }
  for (const { name, missing } of incompleteThresholds) {
    console.warn(`  threshold incomplete for "${name}": missing ${missing.join(', ')}`)
  }
  const insertThreshold = db.prepare(
    `INSERT OR IGNORE INTO dependency_thresholds
      (dependency_id, metric, scope, threshold_value, threshold_unit, threshold_direction,
       threshold_source_url, threshold_as_of, threshold_note, threshold_contested,
       threshold_alt_value, threshold_alt_source_url, threshold_contested_note,
       policy_dependent, baseline_value, baseline_as_of, baseline_note)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  let thresholdsInserted = 0
  for (const row of resolvedThresholds) {
    const dependencyId = dependencyIdByName.get(row.dependency_name)
    if (dependencyId === undefined) {
      skipped += 1
      continue
    }
    insertThreshold.run(
      dependencyId,
      toText(row.metric),
      toText(row.scope),
      toReal(row.threshold_value),
      toText(row.threshold_unit),
      toText(row.threshold_direction),
      toText(row.threshold_source_url),
      toText(row.threshold_as_of),
      toText(row.threshold_note),
      toInt(row.threshold_contested) ?? 0,
      toReal(row.threshold_alt_value),
      toText(row.threshold_alt_source_url),
      toText(row.threshold_contested_note),
      toInt(row.policy_dependent) ?? 0,
      toReal(row.baseline_value),
      toText(row.baseline_as_of),
      toText(row.baseline_note),
    )
    thresholdsInserted += 1
  }

  // company_dependencies: resolve company_uuid + dependency_name. Rows step1c left
  // unresolved have no canonical dependency and are counted as skipped.
  const insertCompanyDependency = db.prepare(
    `INSERT OR IGNORE INTO company_dependencies
      (company_id, dependency_id, criticality, detail, source_url)
     VALUES (?, ?, ?, ?, ?)`,
  )
  for (const row of companyDependencies) {
    const companyId = companyIdFor(row)
    const dependencyId = dependencyIdByName.get(row.dependency_name ?? '')
    if (companyId === undefined || dependencyId === undefined) {
      skipped += 1
      continue
    }
    insertCompanyDependency.run(
      companyId,
      dependencyId,
      toText(row.criticality),
      toText(row.detail),
      toText(row.source_url),
    )
  }

  const insertAssessment = db.prepare(
    `INSERT INTO dependency_assessments
      (uuid, dependency_id, status, detail, metric_name, metric_value, metric_unit,
       assessed_on, source_url, snippet, confidence, confidence_score, contested, contested_note)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  for (const row of assessments) {
    const dependencyId = dependencyIdByName.get(row.dependency_name ?? '')
    if (dependencyId === undefined) {
      skipped += 1
      continue
    }
    insertAssessment.run(
      toText(row.uuid),
      dependencyId,
      toText(row.status),
      toText(row.detail),
      toText(row.metric_name),
      toReal(row.metric_value),
      toText(row.metric_unit),
      toText(row.assessed_on),
      toText(row.source_url),
      toText(row.snippet),
      toText(row.confidence),
      toReal(row.confidence_score),
      toInt(row.contested),
      toText(row.contested_note),
    )
  }

  // metric_observations: curated dated facts. Resolve dependency_name -> id via the same
  // map, and reuse matchCanonical (via validateObservationRows) so an unresolvable name or a
  // curated row without a source is reported, not silently dropped. Append-only table.
  const {
    resolved: resolvedObservations,
    unmatched: unmatchedObservations,
    missingSource,
  } = validateObservationRows(metricObservations, [...dependencyIdByName.keys()])
  for (const row of unmatchedObservations) {
    console.warn(
      `  metric_observation dependency_name matched nothing canonical: "${row.dependency_name ?? ''}"`,
    )
  }
  for (const row of missingSource) {
    console.warn(
      `  curated metric_observation missing source_url: "${row.dependency_name ?? ''}" ${row.metric ?? ''} ${row.as_of ?? ''}`,
    )
  }
  const insertObservation = db.prepare(
    `INSERT INTO metric_observations
      (dependency_id, metric, value, unit, basis, as_of, scope, method, source_url,
       source_name, note)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  let observationsInserted = 0
  for (const row of resolvedObservations) {
    const dependencyId = dependencyIdByName.get(row.dependency_name)
    if (dependencyId === undefined) {
      skipped += 1
      continue
    }
    insertObservation.run(
      dependencyId,
      toText(row.metric),
      toReal(row.value),
      toText(row.unit),
      toText(row.basis),
      toText(row.as_of),
      toText(row.scope),
      toText(row.method),
      toText(row.source_url),
      toText(row.source_name),
      toText(row.note),
    )
    observationsInserted += 1
  }

  // capacity_series: cumulative deployment per technology, the x-axis of the Wright fit.
  // `technology` names a canonical dependency, so it resolves through the same map and the
  // same report-not-drop validator — validateObservationRows keys on `dependency_name`, so
  // the column is aliased rather than the validator duplicated.
  const capacityRows = capacitySeries.map((row) => ({
    ...row,
    dependency_name: row.technology ?? '',
  }))
  const { resolved: resolvedCapacity, unmatched: unmatchedCapacity } = validateObservationRows(
    capacityRows,
    [...dependencyIdByName.keys()],
  )
  for (const row of unmatchedCapacity) {
    console.warn(
      `  capacity_series technology matched nothing canonical: "${row.technology ?? ''}"`,
    )
  }
  const insertCapacity = db.prepare(
    `INSERT INTO capacity_series
      (dependency_id, metric, value, unit, basis, as_of, scope, scenario, method, source_url,
       source_name, note)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  let capacityInserted = 0
  for (const row of resolvedCapacity) {
    const dependencyId = dependencyIdByName.get(row.dependency_name)
    if (dependencyId === undefined) {
      skipped += 1
      continue
    }
    insertCapacity.run(
      dependencyId,
      toText(row.metric),
      toReal(row.value),
      toText(row.unit),
      toText(row.basis),
      toText(row.as_of),
      toText(row.scope),
      toText(row.scenario),
      toText(row.method),
      toText(row.source_url),
      toText(row.source_name),
      toText(row.note),
    )
    capacityInserted += 1
  }

  // dependency_links: causal edges between dependencies. Both endpoints must resolve.
  const { resolved: resolvedLinks, unmatched: unmatchedLinks } = validateDependencyLinks(
    dependencyLinks,
    canonicalNames,
  )
  for (const row of unmatchedLinks) {
    console.warn(
      `  dependency_link endpoint matched nothing canonical: "${row.from_dependency ?? ''}" -> "${row.to_dependency ?? ''}"`,
    )
  }
  const insertLink = db.prepare(
    `INSERT INTO dependency_links (from_dependency_id, to_dependency_id, relation, note)
     VALUES (?, ?, ?, ?)`,
  )
  let linksInserted = 0
  for (const link of resolvedLinks) {
    const fromId = dependencyIdByName.get(link.from_dependency)
    const toId = dependencyIdByName.get(link.to_dependency)
    if (fromId === undefined || toId === undefined) {
      skipped += 1
      continue
    }
    insertLink.run(fromId, toId, link.relation, toText(link.note))
    linksInserted += 1
  }

  // trajectory_config: seed the one tunable row the trajectory view cross-joins against.
  db.prepare('INSERT INTO trajectory_config (window_n, plateau_slope_threshold) VALUES (?, ?)').run(
    trajectoryConfig.windowN,
    trajectoryConfig.plateauSlopeThreshold,
  )

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
  const wrightInput = db
    .prepare(
      `SELECT p.dependency_id, p.metric, p.scope, p.direction, p.baseline, p.threshold,
              p.as_of, p.value_raw
       FROM progress p
       WHERE p.dependency_id IN (SELECT DISTINCT dependency_id FROM capacity_series)`,
    )
    .all() as {
    dependency_id: number
    metric: string
    scope: string
    direction: string
    baseline: number
    threshold: number
    as_of: string
    value_raw: number
  }[]
  const capacityByDependency = new Map<number, SeriesPoint[]>()
  for (const row of db
    .prepare(
      // Historical only: a scenario row is a forecast, and fitting a forecast would launder
      // an assumption into evidence.
      `SELECT dependency_id, as_of, value FROM capacity_series
       WHERE scenario = 'historical' AND value IS NOT NULL`,
    )
    .all() as { dependency_id: number; as_of: string; value: number }[]) {
    const list = capacityByDependency.get(row.dependency_id) ?? []
    list.push({ as_of: row.as_of, value: row.value })
    capacityByDependency.set(row.dependency_id, list)
  }
  const wrightSeries = new Map<string, WrightSeries>()
  for (const row of wrightInput) {
    const key = `${row.dependency_id}::${row.metric}::${row.scope}`
    const existing = wrightSeries.get(key)
    if (existing) {
      existing.costPoints.push({ as_of: row.as_of, value: row.value_raw })
      continue
    }
    wrightSeries.set(key, {
      dependency_id: row.dependency_id,
      metric: row.metric,
      scope: row.scope,
      direction: row.direction,
      baseline: row.baseline,
      threshold: row.threshold,
      costPoints: [{ as_of: row.as_of, value: row.value_raw }],
      capacityPoints: capacityByDependency.get(row.dependency_id) ?? [],
    })
  }
  const wright = computeWrightProjections([...wrightSeries.values()])
  for (const entry of wright.skipped) {
    console.warn(`  wright fit skipped for "${entry.metric}": ${entry.reason} — using linear fit`)
  }
  // Series Wright already covered are excluded from the linear pass, so a series never
  // carries two competing projections.
  const wrightCovered = new Set(
    wright.rows.map((row) => `${row.dependency_id}::${row.metric}::${row.scope}`),
  )
  const progressRows = (
    db
      .prepare('SELECT dependency_id, metric, scope, as_of, progress FROM progress')
      .all() as ProgressPoint[]
  ).filter((row) => !wrightCovered.has(`${row.dependency_id}::${row.metric}::${row.scope}`))
  const projections = [
    ...wright.rows,
    ...computeProjections(progressRows, { windowN: trajectoryConfig.windowN }),
  ]
  const insertProjection = db.prepare(
    `INSERT INTO metric_projections
      (dependency_id, metric, scope, as_of, progress, projected, method, fit_window_n,
       confidence, is_crossing, note)
     VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)`,
  )
  for (const row of projections) {
    insertProjection.run(
      row.dependency_id,
      row.metric,
      row.scope,
      row.as_of,
      row.progress,
      row.method,
      row.fit_window_n,
      row.confidence,
      row.is_crossing,
      row.note,
    )
  }

  // raw_documents: the on-disk fetch cache, mirrored into the DB for querying.
  const insertRawDocument = db.prepare(
    `INSERT OR IGNORE INTO raw_documents (url, url_hash, fetched_at, text, source_type)
     VALUES (?, ?, ?, ?, ?)`,
  )
  for (const document of rawDocuments) {
    insertRawDocument.run(
      document.url,
      document.url_hash,
      document.fetched_at,
      document.text,
      document.source_type,
    )
  }

  db.close()

  console.log(
    `Built ${dbFile}: ${SECTOR.length} sectors, ${ideaSpaces.length} idea_spaces, ` +
      `${companies.length} companies, ${companyUrls.length} company_urls, ` +
      `${companySectors.length} company_sectors, ` +
      `${fundingRounds.length} funding_rounds, ${challenges.length} challenges, ` +
      `${dependencies.length} dependencies, ${thresholdsInserted} dependency_thresholds, ` +
      `${companyDependencies.length} company_dependencies, ` +
      `${assessments.length} dependency_assessments, ${observationsInserted} metric_observations, ` +
      `${capacityInserted} capacity_series, ` +
      `${linksInserted} dependency_links, ${projections.length} metric_projections ` +
      `(${wright.rows.length} wright, ${projections.length - wright.rows.length} linear), ` +
      `${rawDocuments.length} raw_documents` +
      (baselineEquals || baselinePast
        ? ` (${baselineEquals} baseline==threshold, ${baselinePast} baseline-past-threshold — progress null)`
        : '') +
      (skipped ? ` (${skipped} child rows skipped — unresolved FK)` : ''),
  )
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
