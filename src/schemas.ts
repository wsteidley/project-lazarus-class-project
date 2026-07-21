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

export type DependencyResolution = z.infer<typeof dependencyResolutionSchema>
export type DependencyAssessment = z.infer<typeof dependencyAssessmentSchema>
