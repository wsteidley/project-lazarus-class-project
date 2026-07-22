import 'dotenv/config'
import { join } from 'node:path'

export type Provider = 'openai' | 'ollama'

const readProvider = (): Provider => {
  const rawProvider = (process.env.PROVIDER ?? 'openai').toLowerCase()
  if (rawProvider !== 'openai' && rawProvider !== 'ollama') {
    throw new Error(`PROVIDER must be "openai" or "ollama", received "${rawProvider}"`)
  }
  return rawProvider
}

const requireEnv = (name: string): string => {
  const value = process.env[name]
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`)
  }
  return value
}

export const config = {
  provider: readProvider(),
  openAiModel: process.env.OPENAI_MODEL ?? 'gpt-4o-mini',
  ollamaModel: process.env.OLLAMA_MODEL ?? 'llama3.2',
  // How many articles/rows each step processes; mirrors the original first-10 cap.
  processingLimit: Number(process.env.PROCESSING_LIMIT ?? '10'),
  batchSize: Number(process.env.BATCH_SIZE ?? '5'),
  // Root of all pipeline data. The scoped areas below hang off it.
  dataDir: process.env.DATA_DIR ?? './data',
  // How long a perishable cached document stays usable. Discovery text ignores this
  // (launch articles are immutable); outcome/reassessment results expire, because a
  // company alive today may fold next year.
  cacheTtlDays: Number(process.env.CACHE_TTL_DAYS ?? '30'),
  // Primary search provider. Optional on purpose: with no key the search backbone
  // degrades to DuckDuckGo rather than failing, so a missing secret costs recall,
  // not a broken run.
  tavilyApiKey: process.env.TAVILY_API_KEY ?? '',
  // #4 confidence machinery knobs (sampling breadth, not correctness): how many
  // companies to run the fuller cross-source confidence check on, and whether to
  // restrict that to high-value rows or run it over everyone.
  confidenceSamples: Number(process.env.CONFIDENCE_SAMPLES ?? '0'), // 0 = no cap
  confidenceScope: (process.env.CONFIDENCE_SCOPE ?? 'high_value') as 'high_value' | 'all',
  // Optional Splink fuzzy tier: the score at/above which a merge candidate is auto-applied
  // by resolve:apply. Candidates below it (down to link.py's 0.5 floor) stay applied=0 in
  // merge_candidates.csv as a review queue. Retunable without re-running Python.
  fuzzyMergeThreshold: Number(process.env.FUZZY_MERGE_THRESHOLD ?? '0.9'),
}

// Scoped data areas: curated input (tracked), raw scrapes, and timestamped run
// output. Derived from dataDir so DATA_DIR relocates the whole tree.
export const inputDir = join(config.dataDir, 'input')
export const scrapedDir = join(config.dataDir, 'scraped')
export const outputBaseDir = join(config.dataDir, 'output')
// Content-addressed raw-document cache. Deliberately outside the run folders so it
// survives runs: re-extraction under an evolving schema must never re-scrape.
export const cacheDir = join(config.dataDir, 'cache')
// Static external reference data (Crunchbase, startup-failure compilations, …), used
// to enrich companies the pipeline finds. Each `sources/<name>/` holds the untouched
// raw copy plus a normalized `companies.csv` in the shared enrichment schema.
export const sourcesDir = join(config.dataDir, 'sources')

// Provider-specific secrets are read lazily so a step only needs the vars it uses.
export const requireOpenAiKey = (): string => requireEnv('OPENAI_API_KEY')
export const requireLlamaBaseUrl = (): string => requireEnv('LLAMA_BASE_URL')
export const requireCrunchbaseKey = (): string => requireEnv('CRUNCHBASE_API_KEY')
