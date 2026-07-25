// The canonical stage graph, as data. The ordering rules ARE the correctness guarantees
// (merge before enrichment, enrich before search, evidence before heuristic, outcome_type
// after funding), and running stages out of order does not crash — it silently produces
// wrong precedence. Encoding the order here lets the orchestrator enforce it by machine
// rather than by whoever is typing. Pure functions only; no spawning, no I/O.

export type StageDef = {
  // Stable id and the npm script the orchestrator shells out to (usually the same string).
  id: string
  script: string
  // Hard prerequisites: this stage refuses to run unless each is `ok` in the manifest.
  prereqs: string[]
  // Optional stages degrade (skip) rather than fail — e.g. the Splink fuzzy tier when uv
  // is absent. A present-but-failed optional stage is still a real failure.
  optional?: boolean
  // Whether the stage spends on LLM/search APIs. Surfaced by --dry-run so a full run is
  // never a cost surprise.
  makesApiCalls?: boolean
  // Declared data flow, shown by --dry-run. Names are run-dir CSVs unless noted.
  inputs: string[]
  outputs: string[]
  // The run-dir file whose row count is the stage's headline rows_out (for the summary
  // line and manifest). Omitted for stages whose output isn't a single countable CSV.
  primaryOutput?: string
  // One-line statement of the invariant the stage's position upholds.
  invariant: string
}

// Declared in execution order. The index in this array is the canonical sequence position.
export const STAGES: StageDef[] = [
  {
    id: 'step0',
    script: 'step0',
    prereqs: [],
    inputs: ['(web: TechCrunch)'],
    outputs: ['scraped/*_data.csv'],
    invariant: 'corpus entry',
  },
  {
    id: 'step1',
    script: 'step1',
    prereqs: ['step0'],
    makesApiCalls: true,
    inputs: ['scraped/*_data.csv'],
    outputs: ['companies.csv', 'company_sectors.csv', 'challenges.csv', 'company_urls.csv'],
    primaryOutput: 'companies.csv',
    invariant: 'needs scraped articles',
  },
  {
    id: 'step1b',
    script: 'step1b',
    prereqs: ['step1'],
    makesApiCalls: true,
    inputs: ['companies.csv'],
    outputs: ['idea_dependencies.csv'],
    primaryOutput: 'idea_dependencies.csv',
    invariant: 'needs company rows',
  },
  {
    id: 'step1c',
    script: 'step1c',
    prereqs: ['step1b'],
    makesApiCalls: true,
    inputs: ['idea_dependencies.csv', 'curated/dependencies.csv'],
    outputs: ['company_dependencies.csv'],
    primaryOutput: 'company_dependencies.csv',
    invariant: 'canonical dependency list before any assessment',
  },
  {
    id: 'resolve',
    script: 'resolve',
    prereqs: ['step1c'],
    inputs: ['companies.csv', 'company_urls.csv'],
    outputs: ['companies.csv', 'company_urls.csv', 'company_uuid_map.csv'],
    primaryOutput: 'companies.csv',
    invariant: 'merge before enrichment',
  },
  {
    id: 'resolve:fuzzy',
    script: 'resolve:fuzzy',
    prereqs: ['resolve'],
    optional: true,
    inputs: ['companies.csv', 'company_urls.csv'],
    outputs: ['merge_candidates.csv'],
    primaryOutput: 'merge_candidates.csv',
    invariant: 'must sit at the resolve boundary',
  },
  {
    id: 'resolve:apply',
    script: 'resolve:apply',
    prereqs: ['resolve:fuzzy'],
    optional: true,
    inputs: ['merge_candidates.csv', 'company_uuid_map.csv'],
    outputs: ['companies.csv'],
    primaryOutput: 'companies.csv',
    invariant: 'merges via the existing merge path',
  },
  {
    id: 'enrich',
    script: 'enrich',
    prereqs: ['resolve'],
    inputs: ['companies.csv', 'company_urls.csv'],
    outputs: ['companies.csv', 'funding_rounds.csv', 'challenges.csv'],
    primaryOutput: 'companies.csv',
    invariant: 'enrich before search — deterministic, no API cost',
  },
  {
    id: 'step2',
    script: 'step2',
    prereqs: ['enrich'],
    makesApiCalls: true,
    inputs: ['companies.csv'],
    outputs: ['funding_rounds.csv'],
    primaryOutput: 'funding_rounds.csv',
    invariant: 'after canonical uuids exist',
  },
  {
    id: 'step3',
    script: 'step3',
    prereqs: ['step2'],
    inputs: ['funding_rounds.csv'],
    outputs: ['funding_rounds.csv'],
    primaryOutput: 'funding_rounds.csv',
    invariant: 'deterministic dedupe/standardize',
  },
  {
    id: 'outcome-pass',
    script: 'outcome-pass',
    prereqs: ['enrich'],
    makesApiCalls: true,
    inputs: ['companies.csv'],
    outputs: ['companies.csv'],
    primaryOutput: 'companies.csv',
    invariant: 'after enrich, so it fills gaps not duplicates',
  },
  {
    id: 'derive',
    // total_raised comes from *cleaned* funding (step3), not raw step2 output, and the
    // evidenced outcome from outcome-pass — so both are hard prereqs. Depending on step3
    // (which itself needs step2) also makes step3 a real upstream, so re-running the
    // funding cleanup correctly stales derive and build.
    script: 'derive',
    prereqs: ['outcome-pass', 'step3'],
    inputs: ['companies.csv', 'funding_rounds.csv'],
    outputs: ['companies.csv'],
    primaryOutput: 'companies.csv',
    invariant: 'after 7-9: needs total_raised + evidenced outcome',
  },
  {
    id: 'reassess',
    script: 'reassess',
    prereqs: ['step1c'],
    makesApiCalls: true,
    inputs: ['curated/dependencies.csv'],
    outputs: ['dependency_assessments.csv', 'company_dependencies.csv'],
    primaryOutput: 'dependency_assessments.csv',
    invariant: 'needs canonical dependencies',
  },
  {
    // Reads only data/sources + data/curated — nothing from the run dir — so it has no
    // prereqs and can run at any point before `build`. Its position here (rather than at
    // the front) is just where the data it produces is needed.
    id: 'build-metric-data',
    script: 'build-metric-data',
    prereqs: [],
    inputs: ['sources/owid/*.csv', 'sources/irena/*.xlsb', 'curated/*_anchors.csv'],
    outputs: ['derived/metric_observations_full.csv', 'derived/capacity_series.csv'],
    invariant: 'derived files reproduce from committed raw sources; no LLM, no network',
  },
  {
    id: 'build',
    script: 'build',
    prereqs: ['derive', 'reassess', 'build-metric-data'],
    inputs: ['all run CSVs', 'curated/*.csv', 'derived/*.csv'],
    outputs: ['lazarus.db'],
    invariant: 'terminal; consumes everything',
  },
]

const byId = new Map(STAGES.map((stage) => [stage.id, stage]))

export const stageById = (id: string): StageDef | undefined => byId.get(id)

const orderIndex = (id: string): number => STAGES.findIndex((stage) => stage.id === id)

const requireStage = (id: string, flag: string): StageDef => {
  const stage = byId.get(id)
  if (!stage) {
    throw new Error(
      `Unknown stage "${id}" for ${flag}. Known: ${STAGES.map((s) => s.id).join(', ')}`,
    )
  }
  return stage
}

export type PlanSelectors = { from?: string; only?: string; through?: string }

// Resolves the flag selectors into an ordered list of stages. `--only` wins (a single
// stage); otherwise `--from`/`--through` bound a contiguous slice of the canonical
// sequence; with neither, the whole sequence. Optional stages stay in the plan and
// degrade at run time rather than being pre-filtered.
export const resolvePlan = (selectors: PlanSelectors): StageDef[] => {
  const { from, only, through } = selectors
  if (only) {
    if (from || through) {
      throw new Error('--only cannot be combined with --from/--through')
    }
    return [requireStage(only, '--only')]
  }
  const start = from ? orderIndex(requireStage(from, '--from').id) : 0
  const end = through ? orderIndex(requireStage(through, '--through').id) : STAGES.length - 1
  if (start > end) {
    throw new Error(`--from ${from} comes after --through ${through}`)
  }
  return STAGES.slice(start, end + 1)
}

// Every stage that transitively depends on `id` via prereqs — the stages a re-run of `id`
// invalidates. Order-independent: a stage that doesn't depend on `id` (e.g. reassess when
// enrich re-runs) is deliberately left untouched.
export const transitiveDependents = (id: string): string[] => {
  const dependents = new Set<string>()
  let changed = true
  while (changed) {
    changed = false
    for (const stage of STAGES) {
      if (dependents.has(stage.id)) {
        continue
      }
      if (stage.prereqs.some((prereq) => prereq === id || dependents.has(prereq))) {
        dependents.add(stage.id)
        changed = true
      }
    }
  }
  // Emit in canonical order so callers marking them stale read top-down.
  return STAGES.filter((stage) => dependents.has(stage.id)).map((stage) => stage.id)
}

// The stages in a plan that will spend on APIs — surfaced by --dry-run.
export const apiStages = (plan: StageDef[]): StageDef[] =>
  plan.filter((stage) => stage.makesApiCalls)

// --smoke preset: a tiny processing limit and no confidence sampling, to verify wiring
// end-to-end cheaply. Applied as env overrides on the child stages.
export const SMOKE_ENV = {
  PROCESSING_LIMIT: '2',
  CONFIDENCE_SAMPLES: '0',
} as const
