import { existsSync, readdirSync, statSync } from 'node:fs'
import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { inputDir, outputBaseDir, scrapedDir } from '../config.js'
import { runStamp } from './timestamp.js'

// Path to a curated input file, e.g. inputFile('idea_spaces.csv').
export const inputFile = (name: string): string => join(inputDir, name)

// Newest scraped article data file (excludes the *_data_index.csv companion).
// step1/step1b use this so you never have to name the scrape.
export const latestScrapedDataFile = (): string => {
  if (!existsSync(scrapedDir)) {
    throw new Error(`No ${scrapedDir} yet — run step0 to scrape articles first`)
  }
  const candidates = readdirSync(scrapedDir)
    .filter((name) => name.endsWith('_data.csv') && !name.endsWith('_data_index.csv'))
    .sort()
  const newest = candidates.at(-1)
  if (!newest) {
    throw new Error(`No *_data.csv in ${scrapedDir} — run step0 first`)
  }
  return join(scrapedDir, newest)
}

const runDirs = (): string[] => {
  if (!existsSync(outputBaseDir)) {
    return []
  }
  return readdirSync(outputBaseDir)
    .map((name) => join(outputBaseDir, name))
    .filter((path) => statSync(path).isDirectory())
    .sort()
}

// Creates a fresh timestamped run folder and returns its path. Called by step1.
export const newRunDir = async (): Promise<string> => {
  const dir = join(outputBaseDir, runStamp())
  await mkdir(dir, { recursive: true })
  return dir
}

// Newest existing run folder. Steps after step1 flow into this.
export const latestRunDir = (): string => {
  const dir = runDirs().at(-1)
  if (!dir) {
    throw new Error(`No run folders in ${outputBaseDir} — run step1 to start a run`)
  }
  return dir
}
