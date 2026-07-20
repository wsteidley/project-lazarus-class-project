import { z } from 'zod'

// Controlled vocabularies referenced inside the field descriptions so the model
// is nudged toward consistent values. Ported from response_format.py.
export const climateSectors = [
  'Energy',
  'Transportation',
  'Agriculture',
  'Forestry',
  'Water and Oceans',
  'Built Environment',
  'Waste Management',
  'Land Use and Ecosystem Services',
  'Carbon Markets and Climate Finance',
  'Climate Adaptation and Resilience',
  'Circular Economy',
  'Environmental Technology',
  'Carbon Removal',
  'Climate Advocacy and Policy',
]

export const locations = [
  'North America',
  'South America',
  'Europe',
  'Russia',
  'Asia',
  'Middle East',
  'Other',
]

export const fundingRoundNames = [
  'Pre-Seed',
  'Seed',
  'Series A',
  'Series B',
  'Series C',
  'Series D',
  'Series E',
  'Series F',
  'Other',
]

// step1: structured company info extracted from a single article.
export const companyInfoSchema = z.object({
  company_name: z.string().describe('Name of the company'),
  founders: z.string().nullable().describe('CSV string of founder names'),
  is_climate_related: z.boolean().describe('Indicates if the company is climate-related'),
  climate_sectors: z
    .string()
    .nullable()
    .describe(
      `If is_climate_related is true, a list of the climate sectors from ${climateSectors.join(', ')} related to this company`,
    ),
  location: z
    .string()
    .nullable()
    .describe(`Location of the company from this list: ${locations.join(', ')}`),
  living_status: z.string().describe('Indicates if the company is dead, living, or unknown'),
  has_pivoted: z
    .boolean()
    .nullable()
    .describe('Indicates if the company has pivoted from its original idea'),
  year_founded: z.string().nullable().describe('Year the company was founded'),
  year_died: z.string().nullable().describe('Year the company closed, if applicable'),
  idea_summary: z
    .string()
    .nullable()
    .describe('One-sentence summary of the startup idea and technology'),
  reason_for_demise: z
    .string()
    .nullable()
    .describe("If applicable, describes reasons for the company's demise"),
})

export type CompanyInfo = z.infer<typeof companyInfoSchema>

// step2: funding rounds discovered for a company.
export const fundingRoundSchema = z.object({
  currency_symbol: z.string().nullable().describe('Currency symbol for the money'),
  round_name: z
    .string()
    .nullable()
    .describe(`Type of funding round, should be one of ${fundingRoundNames.join(', ')}`),
  amount: z.number().nullable().describe('Amount of the funding round as a number'),
  date: z.string().nullable().describe('String representing MM/YY date of the funding'),
})

export const fundingRoundsSchema = z.object({
  funding_rounds: z.array(fundingRoundSchema),
})

export type FundingRound = z.infer<typeof fundingRoundSchema>
export type FundingRounds = z.infer<typeof fundingRoundsSchema>
