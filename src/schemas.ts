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
export const FAILURE = [
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

// step1: structured company info + failure reasons extracted from a single article.
export const failureReasonSchema = z.object({
  category: z.enum(FAILURE).describe('Controlled failure category'),
  detail: z.string().nullable().describe('The specific, free-text story for this failure'),
})

export const companyExtractionSchema = z.object({
  company_name: z.string().describe('Name of the company'),
  founders: z.string().nullable().describe('CSV string of founder names'),
  is_climate: z.boolean().describe('Whether the company is climate-related'),
  sector: z
    .enum(SECTOR)
    .nullable()
    .describe('High-level climate vertical; null if not climate-related or unclear'),
  subsector: z.string().nullable().describe('Optional finer free-text label within the sector'),
  location: z.enum(REGION).nullable().describe('High-level region the company is based in'),
  country: z.string().nullable().describe('Optional finer country location'),
  living_status: z.enum(LIVING_STATUS).describe('Current status of the company'),
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
    .describe('One-sentence summary of the startup idea and technology'),
  original_trl: z
    .number()
    .int()
    .min(1)
    .max(9)
    .nullable()
    .describe('Technology Readiness Level (1-9) at the time, if evident; else null'),
  reason_for_demise: z
    .string()
    .nullable()
    .describe('Optional overall prose summary of why it failed'),
  failure_reasons: z
    .array(failureReasonSchema)
    .describe('Structured failure reasons; empty array if the company did not fail'),
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
