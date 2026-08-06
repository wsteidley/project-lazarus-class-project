import type { DatabaseSync } from 'node:sqlite'
import { SECTOR } from '../schemas.js'
import { assertNoBasisMismatches } from './basis.js'
import type { CsvRow } from './csv.js'
import type { RawDocument } from './raw-documents.js'
import {
  validateDependencyEdges,
  validateObservationRows,
  validateThresholds,
} from './thresholds.js'
import { trajectoryConfig } from './trajectory-config.js'

// The load half of the build, split out of build-db.ts so it can run against in-memory
// inputs in a test. build-db.ts reads CSVs off disk and owns the derivation half (baseline
// guards, Wright fits, projections) that runs after this; nothing here reads a view.
//
// The split exists for one specific reason: P2's guarantee — "a company with a factual
// footprint and zero metric data still loads" — is a claim about this code, and it was
// untestable while the only entry point required a run directory full of uncommitted LLM
// output.

export type LoadInputs = {
  ideaSpaces: CsvRow[]
  companies: CsvRow[]
  companyUrls: CsvRow[]
  companySectors: CsvRow[]
  fundingRounds: CsvRow[]
  challenges: CsvRow[]
  dependencies: CsvRow[]
  referenceEntities: CsvRow[]
  dependencyLinks: CsvRow[]
  searchCoverage: CsvRow[]
  thresholds: CsvRow[]
  companyDependencies: CsvRow[]
  assessments: CsvRow[]
  metricObservations: CsvRow[]
  capacitySeries: CsvRow[]
  dependencyEdges: CsvRow[]
  rawDocuments: RawDocument[]
}

export type LoadReport = {
  thresholdsInserted: number
  observationsInserted: number
  capacityInserted: number
  edgesInserted: number
  entitiesInserted: number
  linksInserted: number
  coverageInserted: number
  companyDependenciesRetained: number
  companyDependenciesUnresolved: number
  // Per-table tally of rows that could not be inserted, keyed `table: reason`. Deliberately
  // not one scalar: "12 child rows skipped" told you something was lost but never what, which
  // is the same absence-as-a-bare-null failure P3 forbids in the data.
  skipped: Map<string, number>
}

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

export const loadDatabase = (db: DatabaseSync, inputs: LoadInputs): LoadReport => {
  const skipped = new Map<string, number>()
  const skip = (table: string, reason: string): void => {
    const key = `${table}: ${reason}`
    skipped.set(key, (skipped.get(key) ?? 0) + 1)
  }

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
  for (const row of inputs.ideaSpaces) {
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
  //
  // Unconditional by design (P2): every row of companies.csv becomes a company. An unknown
  // idea space or sector degrades to NULL rather than filtering the company out, and no
  // absence of metric, threshold or viability data is consulted here at all — the inclusion
  // gate is a factual footprint and nothing else.
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
  for (const row of inputs.companies) {
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

  // company_urls: typed identity/reference URLs. UNIQUE(company_id, url_type, url)
  // absorbs any repeats the resolve step did not already collapse.
  const insertCompanyUrl = db.prepare(
    `INSERT OR IGNORE INTO company_urls (company_id, url_type, url, normalized_value, source_url)
     VALUES (?, ?, ?, ?, ?)`,
  )
  for (const row of inputs.companyUrls) {
    const companyId = companyIdFor(row)
    if (companyId === undefined) {
      skip('company_urls', 'unknown company_uuid')
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
  for (const row of inputs.companySectors) {
    const companyId = companyIdFor(row)
    const sectorId = sectorIdByName.get(row.sector_name ?? '')
    if (companyId === undefined) {
      skip('company_sectors', 'unknown company_uuid')
      continue
    }
    if (sectorId === undefined) {
      skip('company_sectors', `sector_name not in vocab: "${row.sector_name ?? ''}"`)
      continue
    }
    insertCompanySector.run(companyId, sectorId, toInt(row.is_primary))
  }

  const insertFunding = db.prepare(
    `INSERT INTO funding_rounds
      (company_id, round_name, amount, currency, round_date, round_year, source_url)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  )
  for (const row of inputs.fundingRounds) {
    const companyId = companyIdFor(row)
    if (companyId === undefined) {
      skip('funding_rounds', 'unknown company_uuid')
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
  for (const row of inputs.challenges) {
    const companyId = companyIdFor(row)
    if (companyId === undefined) {
      skip('challenges', 'unknown company_uuid')
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
  for (const row of inputs.dependencies) {
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

  // reference_entities: the data-bearing subjects. Curated seed, loaded before anything that
  // keys on them. name -> id.
  const insertEntity = db.prepare(
    'INSERT INTO reference_entities (name, kind, note) VALUES (?, ?, ?)',
  )
  const entityIdByName = new Map<string, number>()
  for (const row of inputs.referenceEntities) {
    const name = toText(row.name)
    if (name === null) {
      skip('reference_entities', 'empty name')
      continue
    }
    const info = insertEntity.run(name, toText(row.kind) ?? 'technology', toText(row.note))
    entityIdByName.set(name, Number(info.lastInsertRowid))
  }
  let entitiesInserted = entityIdByName.size

  // dependency_links: dependency -> entity, carrying the era slice. A dependency with no row
  // here is standalone/qualitative and that is a first-class state, not a load failure —
  // which is the whole point of the split (P2).
  const insertDependencyLink = db.prepare(
    'INSERT OR IGNORE INTO dependency_links (dependency_name, entity_id, era, note) VALUES (?, ?, ?, ?)',
  )
  let linksInserted = 0
  for (const row of inputs.dependencyLinks) {
    const dependencyName = toText(row.dependency_name) ?? ''
    const entityId = entityIdByName.get(toText(row.entity_name) ?? '')
    if (!dependencyIdByName.has(dependencyName)) {
      skip('dependency_links', `unknown dependency "${dependencyName}"`)
      continue
    }
    if (entityId === undefined) {
      skip('dependency_links', `unknown entity "${row.entity_name ?? ''}"`)
      continue
    }
    insertDependencyLink.run(dependencyName, entityId, toText(row.era), toText(row.note))
    linksInserted += 1
  }

  // dependency_thresholds: one bar per (dependency, metric, scope). Validate names against
  // the seed (report, don't drop) and enforce that a quantitative_with_threshold dependency
  // actually carries a complete bar.
  const canonicalNames = [...dependencyIdByName.keys()]
  const {
    resolved: resolvedThresholds,
    unmatched: unmatchedThresholds,
    incomplete: incompleteThresholds,
  } = validateThresholds(inputs.dependencies, inputs.thresholds, canonicalNames)
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
       policy_dependent, baseline_value, baseline_as_of, baseline_note,
       basis, energy_basis, duration)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  let thresholdsInserted = 0
  for (const row of resolvedThresholds) {
    const dependencyId = dependencyIdByName.get(row.dependency_name)
    if (dependencyId === undefined) {
      skip('dependency_thresholds', `unknown dependency "${row.dependency_name}"`)
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
      // 'na' rather than null: these are equality-join keys and NULL never equals NULL, so an
      // unstamped bar must fall back to the value that matches an unstamped series.
      toText(row.basis) ?? 'na',
      toText(row.energy_basis) ?? 'na',
      toText(row.duration) ?? 'na',
    )
    thresholdsInserted += 1
  }

  // company_dependencies: resolve company_uuid + dependency_name.
  //
  // A row whose dependency_name matched nothing canonical is RETAINED, with a null
  // dependency_id and its raw text (P2). It used to be dropped: step1c writes those rows
  // deliberately so they can be curated, and the loader threw them away, so the company
  // survived while its recorded reason for dying did not. An unresolved blocker is a
  // *named absence*, not a missing row — which is what makes it queryable (P3).
  const insertCompanyDependency = db.prepare(
    `INSERT OR IGNORE INTO company_dependencies
      (company_id, dependency_id, dependency_name_raw, resolution_status, criticality, detail,
       source_url)
     VALUES (?, ?, ?, ?, ?, ?, ?)`,
  )
  let companyDependenciesRetained = 0
  let companyDependenciesUnresolved = 0
  for (const row of inputs.companyDependencies) {
    const companyId = companyIdFor(row)
    if (companyId === undefined) {
      skip('company_dependencies', 'unknown company_uuid')
      continue
    }
    const dependencyId = dependencyIdByName.get(row.dependency_name ?? '')
    // Fall back to the extracted detail so a retained row is never anonymous — the raw name
    // is the only handle a curator has for folding it onto the canonical list later.
    const raw = toText(row.dependency_name_raw) ?? toText(row.detail) ?? ''
    if (dependencyId === undefined) {
      companyDependenciesUnresolved += 1
      console.warn(
        `  company_dependency retained unresolved: "${raw}" (company ${row.company_uuid ?? ''})`,
      )
    }
    insertCompanyDependency.run(
      companyId,
      dependencyId ?? null,
      raw,
      dependencyId === undefined ? 'unresolved_name' : 'resolved',
      toText(row.criticality),
      toText(row.detail),
      toText(row.source_url),
    )
    companyDependenciesRetained += 1
  }

  const insertAssessment = db.prepare(
    `INSERT INTO dependency_assessments
      (uuid, dependency_id, status, detail, metric_name, metric_value, metric_unit,
       assessed_on, source_url, snippet, confidence, confidence_score, contested, contested_note)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  for (const row of inputs.assessments) {
    const dependencyId = dependencyIdByName.get(row.dependency_name ?? '')
    if (dependencyId === undefined) {
      skip('dependency_assessments', `unknown dependency "${row.dependency_name ?? ''}"`)
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
  const entityNames = [...entityIdByName.keys()]
  const {
    resolved: resolvedObservations,
    unmatched: unmatchedObservations,
    missingSource,
  } = validateObservationRows(inputs.metricObservations, entityNames)
  for (const row of unmatchedObservations) {
    console.warn(
      `  metric_observation entity_name matched nothing canonical: "${row.entity_name ?? ''}"`,
    )
  }
  for (const row of missingSource) {
    console.warn(
      `  curated metric_observation missing source_url: "${row.entity_name ?? ''}" ${row.metric ?? ''} ${row.as_of ?? ''}`,
    )
  }
  const insertObservation = db.prepare(
    `INSERT INTO metric_observations
      (entity_id, metric, value, unit, basis, energy_basis, duration, segment, as_of, scope,
       method, source_url, source_name, note)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  let observationsInserted = 0
  for (const row of resolvedObservations) {
    const entityId = entityIdByName.get(row.entity_name)
    if (entityId === undefined) {
      skip('metric_observations', `unknown entity "${row.entity_name}"`)
      continue
    }
    insertObservation.run(
      entityId,
      toText(row.metric),
      toReal(row.value),
      toText(row.unit),
      toText(row.basis) ?? 'na',
      toText(row.energy_basis) ?? 'na',
      toText(row.duration) ?? 'na',
      // Rows predating the segment column are the rolled-up total by definition.
      toText(row.segment) ?? 'all',
      toText(row.as_of),
      toText(row.scope),
      toText(row.method),
      toText(row.source_url),
      toText(row.source_name),
      toText(row.note),
    )
    observationsInserted += 1
  }

  // capacity_series: cumulative deployment per entity, the x-axis of the Wright fit. Both row
  // types genuinely share `entity_name` since the split, so the old alias hack (copying
  // `technology` into `dependency_name` to reuse the validator) is gone.
  const { resolved: resolvedCapacity, unmatched: unmatchedCapacity } = validateObservationRows(
    inputs.capacitySeries,
    entityNames,
  )
  for (const row of unmatchedCapacity) {
    console.warn(
      `  capacity_series entity_name matched nothing canonical: "${row.entity_name ?? ''}"`,
    )
  }
  const insertCapacity = db.prepare(
    `INSERT INTO capacity_series
      (entity_id, metric, value, unit, basis, energy_basis, duration, segment, as_of, scope,
       scenario, method, source_url, source_name, note)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  let capacityInserted = 0
  for (const row of resolvedCapacity) {
    const entityId = entityIdByName.get(row.entity_name)
    if (entityId === undefined) {
      skip('capacity_series', `unknown entity "${row.entity_name}"`)
      continue
    }
    insertCapacity.run(
      entityId,
      toText(row.metric),
      toReal(row.value),
      toText(row.unit),
      toText(row.basis) ?? 'na',
      toText(row.energy_basis) ?? 'na',
      toText(row.duration) ?? 'na',
      toText(row.segment) ?? 'all',
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

  // Baseline rule #1, enforced. Runs here because both sides are now loaded and nothing has
  // read a view yet — the progress join would otherwise silently return zero rows for a
  // mismatched pair, which is indistinguishable from having no data at all. See lib/basis.ts
  // for why this is the one place the pipeline hard-fails rather than reporting.
  //
  // A bar reaches its series through dependency_links, so an unlinked dependency contributes
  // no pair to check — correct, since there is no series it could disagree with.
  const entityNameForDependency = new Map<string, string>()
  for (const row of inputs.dependencyLinks) {
    const dependencyName = toText(row.dependency_name)
    const entityName = toText(row.entity_name)
    if (dependencyName !== null && entityName !== null && entityIdByName.has(entityName)) {
      entityNameForDependency.set(dependencyName, entityName)
    }
  }
  assertNoBasisMismatches(
    resolvedThresholds.flatMap((row) => {
      const entityName = entityNameForDependency.get(row.dependency_name)
      return entityName === undefined
        ? []
        : [
            {
              dependency_name: row.dependency_name,
              entity_name: entityName,
              metric: row.metric ?? '',
              scope: row.scope ?? '',
              basis: row.basis ?? 'na',
              energy_basis: row.energy_basis ?? 'na',
              duration: row.duration ?? 'na',
            },
          ]
    }),
    resolvedObservations,
  )

  // dependency_edges: causal edges between dependencies. Both endpoints must resolve.
  const { resolved: resolvedEdges, unmatched: unmatchedEdges } = validateDependencyEdges(
    inputs.dependencyEdges,
    canonicalNames,
  )
  for (const row of unmatchedEdges) {
    console.warn(
      `  dependency_edge endpoint matched nothing canonical: "${row.from_dependency ?? ''}" -> "${row.to_dependency ?? ''}"`,
    )
  }
  const insertEdge = db.prepare(
    `INSERT INTO dependency_edges (from_dependency_id, to_dependency_id, relation, note)
     VALUES (?, ?, ?, ?)`,
  )
  let edgesInserted = 0
  for (const link of resolvedEdges) {
    const fromId = dependencyIdByName.get(link.from_dependency)
    const toId = dependencyIdByName.get(link.to_dependency)
    if (fromId === undefined || toId === undefined) {
      skip('dependency_edges', 'unresolved endpoint')
      continue
    }
    insertEdge.run(fromId, toId, link.relation, toText(link.note))
    edgesInserted += 1
  }

  // search_coverage: the record of what was actually searched. Deliberately loaded even when
  // empty — an empty table is a meaningful state (nothing is confirmed swept, so every empty
  // region reads 'unsampled'), not a missing input.
  const insertCoverage = db.prepare(
    `INSERT INTO search_coverage
      (idea_space_name, dependency_name, searched_on, source, method, note)
     VALUES (?, ?, ?, ?, ?, ?)`,
  )
  let coverageInserted = 0
  for (const row of inputs.searchCoverage) {
    const ideaSpace = toText(row.idea_space_name)
    if (ideaSpace === null || !ideaSpaceIdByName.has(ideaSpace)) {
      skip('search_coverage', `unknown idea_space "${row.idea_space_name ?? ''}"`)
      continue
    }
    const dependencyName = toText(row.dependency_name)
    if (dependencyName !== null && !dependencyIdByName.has(dependencyName)) {
      skip('search_coverage', `unknown dependency "${dependencyName}"`)
      continue
    }
    insertCoverage.run(
      ideaSpace,
      dependencyName,
      toText(row.searched_on) ?? '',
      toText(row.source) ?? '',
      toText(row.method) ?? 'curated',
      toText(row.note),
    )
    coverageInserted += 1
  }

  // trajectory_config: seed the one tunable row the trajectory view cross-joins against.
  db.prepare('INSERT INTO trajectory_config (window_n, plateau_slope_threshold) VALUES (?, ?)').run(
    trajectoryConfig.windowN,
    trajectoryConfig.plateauSlopeThreshold,
  )

  // raw_documents: the on-disk fetch cache, mirrored into the DB for querying.
  const insertRawDocument = db.prepare(
    `INSERT OR IGNORE INTO raw_documents (url, url_hash, fetched_at, text, source_type)
     VALUES (?, ?, ?, ?, ?)`,
  )
  for (const document of inputs.rawDocuments) {
    insertRawDocument.run(
      document.url,
      document.url_hash,
      document.fetched_at,
      document.text,
      document.source_type,
    )
  }

  return {
    thresholdsInserted,
    observationsInserted,
    capacityInserted,
    edgesInserted,
    entitiesInserted,
    linksInserted,
    coverageInserted,
    companyDependenciesRetained,
    companyDependenciesUnresolved,
    skipped,
  }
}
