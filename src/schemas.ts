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

// step1: structured company info + challenges extracted from a single article.
// A challenge applies to survivors and failures alike; the outcome carries which.
export const challengeSchema = z.object({
  category: z.enum(CHALLENGE).describe('Controlled challenge category'),
  outcome: z
    .enum(CHALLENGE_OUTCOME)
    .describe("Whether it was 'fatal', 'overcome', 'pivoted_from', or still 'ongoing'"),
  detail: z.string().nullable().describe('The specific, free-text story for this challenge'),
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
