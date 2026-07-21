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
  // Root of all pipeline data. The three scoped areas below hang off it.
  dataDir: process.env.DATA_DIR ?? './data',
}

// Scoped data areas: curated input (tracked), raw scrapes, and timestamped run
// output. Derived from dataDir so DATA_DIR relocates the whole tree.
export const inputDir = join(config.dataDir, 'input')
export const scrapedDir = join(config.dataDir, 'scraped')
export const outputBaseDir = join(config.dataDir, 'output')

// Provider-specific secrets are read lazily so a step only needs the vars it uses.
export const requireOpenAiKey = (): string => requireEnv('OPENAI_API_KEY')
export const requireLlamaBaseUrl = (): string => requireEnv('LLAMA_BASE_URL')
export const requireCrunchbaseKey = (): string => requireEnv('CRUNCHBASE_API_KEY')
