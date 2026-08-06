import { z } from 'zod'

// Controlled vocabularies, aligned to the language the climate-tech community and
// investors use. These are the real gate: z.enum forces the LLM output onto the
// vocab, and the DB build mirrors them as CHECK constraints. See
// dataset-schema-spec.md for provenance of each list.

// High-level verticals (Sightline Climate / CTVC, PwC State of Climate Tech).
export const SECTOR = [
  'Energy',
  'Mobility & Transportation',
  'Industry & Manufacturing',
  'Built Environment',
  'Food, Agriculture & Land Use',
  'Carbon & GHG Management',
  'Climate Intelligence & Finance',
  'Water & Oceans',
  'Circular Economy & Waste',
  'Adaptation & Resilience',
] as const

export const REGION = [
  'North America',
  'South America',
  'Europe',
  'Russia',
  'Asia',
  'Middle East',
  'Other',
] as const

export const LIVING_STATUS = ['Operating', 'Acquired', 'Zombie', 'Defunct', 'Unknown'] as const

// Standard VC ladder + climate-relevant non-dilutive / project finance.
export const ROUND = [
  'Grant',
  'Pre-Seed',
  'Seed',
  'Series A',
  'Series B',
  'Series C',
  'Series D+',
  'Growth / Late Stage',
  'Debt / Project Finance',
  'IPO / Public',
  'Acquisition',
  'Unknown',
] as const

// CB Insights post-mortem taxonomy + capital-intensive climate-hardtech causes.
// v2 reframe: a "challenge" is as much what a survivor overcame as what killed a
// failure — the category list is unchanged; the outcome field carries the verdict.
export const CHALLENGE = [
  'No Market Need',
  'Poor Product-Market Fit',
  'Bad Timing (Ahead of Market)',
  'Outcompeted',
  'Flawed Business Model',
  'Pricing / Unit Economics',
  'Poor Product / Execution',
  'Team',
  'Legal / Regulatory',
  'Ran Out of Capital',
  'Technical Feasibility',
  'Scale-Up / Manufacturing',
  'Capital Intensity',
  'Policy / Subsidy Dependence',
  'Input / Commodity Cost',
  'Infrastructure / Supply Chain',
  'Other',
] as const

// Whether a challenge was fatal or cleared.
export const CHALLENGE_OUTCOME = ['fatal', 'overcome', 'pivoted_from', 'ongoing'] as const

// The success/failure judgment (derived — see derive-outcome.ts), distinct from
// the raw living_status.
export const OUTCOME_TYPE = [
  'Breakout Success',
  'Solid Success',
  'Successful Exit',
  'Soft Landing',
  'Struggling / Zombie',
  'Failed',
  'Too Early to Tell',
  'Unknown',
] as const

// Terminal liquidity event, captured separately from funding so "total raised"
// stays clean.
export const EXIT_TYPE = ['acquisition', 'ipo', 'shutdown', 'none', 'unknown'] as const

// What the idea leaned on — bottlenecks the community tracks.
export const DEPENDENCY = [
  'Input / Commodity Cost',
  'Enabling Technology',
  'Infrastructure',
  'Policy & Incentives',
  'Market Demand / Offtake',
  'Manufacturing Scale-Up',
  'Capital Availability',
  'Supply Chain / Critical Materials',
  'Talent / Expertise',
  'Other',
] as const

export const CRITICALITY = ['was_blocking', 'contributing'] as const

// Whether a company's blocker folded onto the canonical dependency list. `unresolved_name`
// is not missing data — it is a company whose blocker we HAVE, named in the extractor's own
// words, that nothing canonical yet covers. Dropping those rows (the pre-P2 behaviour)
// deleted exactly the qualitative and novel blockers the dataset exists to surface: the
// company survived, its reason for dying did not.
export const RESOLUTION_STATUS = ['resolved', 'unresolved_name'] as const

// Evidence strength for any judged row. Four levels because that's the resolution an
// LLM can produce reliably — a finer hand-scale would be false precision. Orthogonal
// to `contested` (which records that sources disagree): high-and-contested is real.
export const CONFIDENCE = ['unknown', 'low', 'medium', 'high'] as const

// Where a dependency stands *now*, relative to when the company needed it. This is
// the "now" axis the reassessment pass exists to establish.
export const ASSESSMENT_STATUS = [
  'resolved',
  'improving',
  'unchanged',
  'worsening',
  'unknown',
] as const

// How to read a dependency's threshold row. A blank must never be ambiguous:
// `quantitative_with_threshold` carries a real crossing bar (value+unit+direction+source
// all required); `quantitative_tbd` is measurable but the bar isn't set yet; `qualitative`
// has no number by nature.
export const THRESHOLD_KIND = [
  'quantitative_with_threshold',
  'quantitative_tbd',
  'qualitative',
] as const

// Which way "better" runs for a threshold. Required, because the crossing arithmetic is
// identical but the meaning inverts: a battery price falling past $100/kWh (below_is_better)
// is good; an interconnection queue rising past 2 years (above_is_better bar, moving away)
// is bad. Without this a crossing test cannot tell viability from regression.
export const THRESHOLD_DIRECTION = ['below_is_better', 'above_is_better'] as const

// The three orthogonal things "basis" used to mean at once. A series is only comparable to a
// bar when all three agree, so they are separate equality-joined columns rather than one
// overloaded string. 'na' means "this axis does not apply", and it matches only another 'na' —
// never "matches anything", which would reintroduce the silent mismatch.
//
// Currency vintage. `real_usd` is deliberately its own member and is NOT quietly promoted to
// real_2024_usd: BNEF does not state the vintage of its pack-price series, and inventing one
// would launder an assumption into a viability verdict. It therefore compares equal only to
// another `real_usd` series — reject, not reconcile.
export const BASIS_CURRENCY = [
  'real_2024_usd',
  'real_2025_usd',
  'real_usd',
  'nominal_usd',
  'na',
] as const

// Which energy the per-kWh (or per-W) denominator counts. Nameplate and usable capacity differ
// by the depth-of-discharge and round-trip losses of the system; AC and DC differ by the
// inverter. Comparing a cost per usable kWh against a bar declared per nameplate kWh answers
// the viability question wrong by exactly that ratio, silently.
export const ENERGY_BASIS = ['nameplate', 'usable', 'AC', 'DC', 'na'] as const

// Storage duration the per-kWh price refers to. A 4h system and a 2h system have materially
// different $/kWh because the power electronics amortise over more energy; `blended` is a
// mixed-duration fleet average and is comparable to neither on its own.
export const DURATION = ['4h', '2h', 'blended', 'na'] as const

// How a metric observation was produced. Keeps grounded numbers separable from generated
// ones: `curated` = hand-entered from a cited report, `feed` = pulled programmatically,
// `llm` = emitted by the reassess pass.
export const OBSERVATION_METHOD = ['curated', 'feed', 'llm'] as const

// What kind of thing a reference entity is. All four are in the vocabulary now but only
// 'technology' is seeded, so admitting an "EV charging network" later is a data change rather
// than a migration -- which is the point of modelling the link for the role instead of the
// type.
export const ENTITY_KIND = ['technology', 'infrastructure', 'market', 'policy'] as const

// How one dependency causes another. IRENA attributes US/German solar LCOE at ~2x China's
// to permitting/interconnection/balance-of-system — so interconnection partly `drives` solar
// economics. Storing the relation makes those chains traceable rather than hidden.
export const DEPENDENCY_RELATION = ['drives', 'enables', 'blocks'] as const

// How a projected point was extrapolated. `linear_progress_fit` fits a line to the recent
// normalized-progress slope — the general case, and the fallback. `wright` is the
// physically-motivated model for cost-curve heroes: cost falls a fixed fraction per doubling
// of cumulative capacity, so the fit is over deployment rather than time. Which one produced
// a row is recorded per point, because they answer with different kinds of confidence.
export const PROJECTION_METHOD = ['linear_progress_fit', 'wright'] as const

// Whether a series' baseline can express a 0->1 progress range. `ok` means progress is
// computable; the other two mean it is NULL with a stated reason: `baseline_equals_threshold`
// (B == T, would divide by zero — interconnection), `baseline_past_threshold` (baseline already
// satisfies the bar, would invert — a declared baseline that postdates viability, e.g. battery).
export const BASELINE_STATUS = [
  'ok',
  'baseline_equals_threshold',
  'baseline_past_threshold',
] as const

// What we can and cannot say about a blocker (P3). These are five DIFFERENT answers and
// must never collapse into one blank: a null that could mean either "we have no data" or
// "we judged it and it hasn't crossed" leaves the user unable to tell a gap from a verdict.
// `no_blocker_data` is the most valuable of them — it is where the user's own knowledge
// does the work the data can't.
//
// Emitted by the series_status / dependency_status / company_dependency_status views, not
// stored: a stored label would be a second source of truth. Listed here as the vocabulary
// of record, so downstream code can name the states without re-deriving them from SQL.
export const ABSENCE_STATE = [
  'no_blocker_data',
  'no_threshold',
  'undetermined',
  'assessed_not_viable',
  'assessed_viable',
  // Structural, not a gap: the blocker is non-measurable ("public trust") and will never
  // have a curve. Fully expressible only after the technology/dependency split (B2).
  'qualitative_blocker',
] as const

// step1: structured company info + challenges extracted from a single article.
// A challenge applies to survivors and failures alike; the outcome carries which.
export const challengeSchema = z.object({
  category: z.enum(CHALLENGE).describe('Controlled challenge category'),
  outcome: z
    .enum(CHALLENGE_OUTCOME)
    .describe("Whether it was 'fatal', 'overcome', 'pivoted_from', or still 'ongoing'"),
  detail: z.string().nullable().describe('The specific, free-text story for this challenge'),
  confidence: z
    .enum(CONFIDENCE)
    .describe("How well the source supports this challenge; 'unknown' if it's a guess"),
  contested: z.boolean().describe('True if sources disagree about this challenge or its outcome'),
  contested_note: z
    .string()
    .nullable()
    .describe('What the disagreement is, when contested; else null'),
})

// Where a URL points. `website` is the identity anchor for entity resolution;
// `crunchbase`/`wikipedia` are the external IDs that cover notable companies;
// `archive` holds Wayback captures, which double as a death signal for the outcome
// pass (a site that stopped capturing is evidence the company stopped).
export const URL_TYPE = [
  'website',
  'crunchbase',
  'wikipedia',
  'linkedin',
  'twitter',
  'archive',
  'article',
  'other',
] as const

// One URL attached to a company. A typed list rather than one column per source, so a
// new source is a new row, not a schema change — and a company can hold several of a
// kind (multiple archive captures, several press links).
export const companyUrlSchema = z.object({
  url_type: z.enum(URL_TYPE).describe('What this URL points at'),
  url: z.string().describe('The URL or identifier as stated by the source'),
})

// One or more sectors a company sits in; exactly one should be is_primary.
export const companySectorSchema = z.object({
  name: z.enum(SECTOR).describe('A broad climate vertical the company sits in'),
  is_primary: z.boolean().describe('True for the single main vertical'),
})

export const companyExtractionSchema = z.object({
  company_name: z.string().describe('Name of the company'),
  founders: z.string().nullable().describe('CSV string of founder names'),
  is_climate: z.boolean().describe('Whether the company is climate-related'),
  sectors: z
    .array(companySectorSchema)
    .describe('One or more broad verticals; mark exactly one is_primary'),
  idea_space_name: z
    .string()
    .nullable()
    .describe(
      'The specific idea space this company pursued — pick the best match from the provided list, or null if none fits',
    ),
  urls: z
    .array(companyUrlSchema)
    .describe('Every URL/handle a source actually states for this company; empty array if none'),
  location: z.enum(REGION).nullable().describe('High-level region the company is based in'),
  country: z.string().nullable().describe('Optional finer country location'),
  living_status: z.enum(LIVING_STATUS).describe('Current raw status of the company'),
  has_pivoted: z
    .boolean()
    .nullable()
    .describe('Whether the company has pivoted from its original idea'),
  year_founded: z.number().int().nullable().describe('Year the company was founded (4-digit)'),
  year_defunct: z
    .number()
    .int()
    .nullable()
    .describe('Year the company shut down (4-digit); null if still operating'),
  idea_summary: z
    .string()
    .nullable()
    .describe('One-sentence summary of what the company did (the idea/technology)'),
  original_trl: z
    .number()
    .int()
    .min(1)
    .max(9)
    .nullable()
    .describe('Technology Readiness Level (1-9) at the time, if evident; else null'),
  exit_type: z
    .enum(EXIT_TYPE)
    .nullable()
    .describe('Terminal liquidity event, if any (kept separate from funding rounds)'),
  exit_amount: z.number().nullable().describe('Value of the exit event, if known (not funding)'),
  exit_date: z.string().nullable().describe('Exit date as an ISO string (YYYY-MM or YYYY-MM-DD)'),
  exit_notes: z.string().nullable().describe('Acquirer, terms, or context for the exit'),
  outcome_summary: z
    .string()
    .nullable()
    .describe('What became of the company, if known — distinct from idea_summary'),
  outcome_source_url: z.string().nullable().describe('Source URL for the outcome narrative'),
  challenges: z
    .array(challengeSchema)
    .describe('Challenges faced, each with an outcome; empty array if none evident'),
})

export type CompanyExtraction = z.infer<typeof companyExtractionSchema>
export type CompanyUrl = z.infer<typeof companyUrlSchema>

// step1b: idea dependencies decomposed from the article in a separate pass.
export const ideaDependencySchema = z.object({
  category: z.enum(DEPENDENCY).describe('Controlled dependency category'),
  detail: z.string().nullable().describe('The specific thing the idea needed'),
  criticality: z
    .enum(CRITICALITY)
    .describe("'was_blocking' if it blocked success, else 'contributing'"),
})

export const ideaDependenciesSchema = z.object({
  dependencies: z.array(ideaDependencySchema),
})

export type IdeaDependency = z.infer<typeof ideaDependencySchema>
export type IdeaDependencies = z.infer<typeof ideaDependenciesSchema>

// step2: funding rounds discovered for a company.
export const fundingRoundSchema = z.object({
  round_name: z.enum(ROUND).nullable().describe('Controlled funding round name'),
  amount: z.number().nullable().describe('Amount raised as a number (no currency symbol)'),
  currency: z.string().nullable().describe('ISO currency code preferred, e.g. USD, EUR'),
  date: z.string().nullable().describe('Funding date as YYYY-MM (keep the full 4-digit year)'),
})

export const fundingRoundsSchema = z.object({
  funding_rounds: z.array(fundingRoundSchema),
})

export type FundingRound = z.infer<typeof fundingRoundSchema>
export type FundingRounds = z.infer<typeof fundingRoundsSchema>

// step1c: maps one company's free-text dependency onto the curated canonical list.
// Null means "none of them fit" — the row is kept unresolved rather than forced, so
// a bad match never silently poisons the shared dependency's assessments.
export const dependencyResolutionSchema = z.object({
  canonical_name: z
    .string()
    .nullable()
    .describe('Exact name from the provided canonical list, or null if none genuinely fits'),
})

// reassess: the "now" verdict for one canonical dependency, with its own evidence so
// the automated call is auditable and can later be swapped for a real data feed.
export const dependencyAssessmentSchema = z.object({
  status: z.enum(ASSESSMENT_STATUS).describe('Where this dependency stands today'),
  detail: z.string().nullable().describe('Free-text justification for the status'),
  metric_name: z
    .string()
    .nullable()
    .describe('The quantity that settles this, e.g. "battery pack price"'),
  metric_value: z.number().nullable().describe('Current value of that metric, as a number'),
  metric_unit: z.string().nullable().describe('Unit for the metric, e.g. "USD/kWh"'),
  source_url: z.string().nullable().describe('URL the value/verdict came from'),
  snippet: z
    .string()
    .nullable()
    .describe('Short verbatim quote from the source supporting the verdict'),
  confidence: z.enum(CONFIDENCE).describe('Evidence strength behind this assessment'),
  contested: z.boolean().describe('True if sources disagree on the current state'),
  contested_note: z.string().nullable().describe('What the disagreement is; else null'),
})

// A single dated metric observation — one row per (dependency x date x scope), the
// grounded fact that a crossing test consumes. Curated rows are hand-entered from a cited
// report and reviewed in the PR; `feed` rows may be written programmatically. Append-only:
// a correction is a new row with a later as_of, never an overwrite.
export const metricObservationSchema = z.object({
  dependency_name: z
    .string()
    .describe('Canonical dependencies.csv name this observation attaches to'),
  metric: z.string().describe('The quantity measured, e.g. "battery pack price"'),
  value: z.number().describe('The observed value, as a number'),
  unit: z.string().describe('Unit for the value, e.g. "USD/kWh"'),
  as_of: z.string().describe('ISO date or year the value describes (not when it was recorded)'),
  scope: z.string().describe('Region the value covers: global / US / EU / CN / NO …'),
  method: z.enum(OBSERVATION_METHOD).describe('How the observation was produced'),
  source_url: z.string().describe('URL the value came from; required when method=curated'),
  source_name: z
    .string()
    .describe('Human-readable source name, e.g. "BNEF 2025 Battery Price Survey"'),
  note: z.string().nullable().describe('Free-text caveat, e.g. segment breakdown; else null'),
})

// One crossing bar, keyed on (dependency, metric, scope) so a dependency can carry several
// bars by slice (carbon $50 global vs. a Norway bar) without them colliding. A `qualitative`
// dependency has no row here; a `quantitative_tbd` one may have a row with a null value. When
// contested, the alt bar lets the derivation compute progress against both and flag the
// crossing rather than silently pick a side.
export const dependencyThresholdSchema = z.object({
  dependency_name: z.string().describe('Canonical dependencies.csv name this bar attaches to'),
  metric: z.string().describe('The quantity the bar is set on, e.g. "battery pack price"'),
  scope: z.string().describe('Region the bar applies to: global / US / EU / NO …'),
  threshold_value: z
    .number()
    .nullable()
    .describe('The value at which the dependency stops blocking'),
  threshold_unit: z.string().describe('Unit for the value, e.g. "USD/kWh"'),
  threshold_direction: z.enum(THRESHOLD_DIRECTION).describe('Which side of the bar is viable'),
  threshold_source_url: z.string().describe('URL the bar came from'),
  threshold_as_of: z.string().describe('When the bar was last reviewed'),
  threshold_note: z.string().nullable().describe('Free-text caveat; else null'),
  threshold_contested: z.boolean().describe('True if the literature disputes this bar'),
  threshold_alt_value: z
    .number()
    .nullable()
    .describe('The disputed alternative bar, when contested'),
  threshold_alt_source_url: z.string().nullable().describe('URL for the alternative bar'),
  threshold_contested_note: z.string().nullable().describe('What the dispute is; else null'),
  baseline_value: z
    .number()
    .nullable()
    .describe(
      'Attempt-era value (where the metric stood when companies died); null falls back to earliest observation',
    ),
  baseline_as_of: z.string().nullable().describe('When the declared baseline was measured'),
  baseline_note: z.string().nullable().describe('Why this baseline; else null'),
})

// A causal edge between two dependencies, so chains like interconnection -> solar economics
// are traceable rather than hidden.
export const dependencyLinkSchema = z.object({
  from_dependency: z.string().describe('Canonical name of the driving dependency'),
  to_dependency: z.string().describe('Canonical name of the affected dependency'),
  relation: z.enum(DEPENDENCY_RELATION).describe('How from acts on to: drives / enables / blocks'),
  note: z.string().nullable().describe('Free-text explanation of the link; else null'),
})

export type DependencyResolution = z.infer<typeof dependencyResolutionSchema>
export type DependencyAssessment = z.infer<typeof dependencyAssessmentSchema>
export type MetricObservation = z.infer<typeof metricObservationSchema>
export type DependencyThreshold = z.infer<typeof dependencyThresholdSchema>
export type DependencyLink = z.infer<typeof dependencyLinkSchema>

// Phase 2b: the retrospective outcome pass. Same evidence shape as
// dependencyAssessmentSchema (confidence/contested/snippet), but for what became of a
// company — living_status and, if it exited, the terminal event.
export const outcomeAssessmentSchema = z.object({
  living_status: z.enum(LIVING_STATUS).describe('Current status, per the search evidence'),
  exit_type: z
    .enum(EXIT_TYPE)
    .nullable()
    .describe('Terminal liquidity event, if any; null if none evident'),
  exit_amount: z.number().nullable().describe('Value of the exit event, if known'),
  exit_date: z.string().nullable().describe('Exit date as YYYY-MM or YYYY-MM-DD, if known'),
  exit_notes: z.string().nullable().describe('Acquirer, terms, or context for the exit'),
  outcome_summary: z
    .string()
    .nullable()
    .describe('What became of the company, per the search evidence'),
  source_url: z.string().nullable().describe('URL the outcome verdict came from'),
  snippet: z
    .string()
    .nullable()
    .describe('Short verbatim quote from the source supporting the verdict'),
  confidence: z.enum(CONFIDENCE).describe('Evidence strength behind this verdict'),
  contested: z.boolean().describe('True if sources disagree about the outcome'),
  contested_note: z.string().nullable().describe('What the disagreement is; else null'),
})

export type OutcomeAssessment = z.infer<typeof outcomeAssessmentSchema>
