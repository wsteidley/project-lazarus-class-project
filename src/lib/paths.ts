import { existsSync, readdirSync, statSync } from 'node:fs'
import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { curatedDir, derivedDir, outputBaseDir, scrapedDir } from '../config.js'
import { runStamp } from './timestamp.js'

// Path to a human-authored input file, e.g. curatedFile('idea_spaces.csv').
export const curatedFile = (name: string): string => join(curatedDir, name)

// Path to a build-metric-data output, e.g. derivedFile('capacity_series.csv'). Generated;
// read it freely, but only build-metric-data writes here.
export const derivedFile = (name: string): string => join(derivedDir, name)

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

// When the pipeline orchestrator owns the run, it exports RUN_DIR so every stage it
// shells out to lands in one folder — pinning a fresh full run and honoring --run-dir
// without any stage having to thread the path through. Read at call time (not via the
// import-time config snapshot) so a per-process override always takes effect.
const runDirOverride = (): string | undefined => process.env.RUN_DIR || undefined

// Creates a fresh timestamped run folder and returns its path. Called by step1. Under a
// RUN_DIR override, returns (and creates) exactly that folder instead of a new stamp, so
// a resumed or orchestrator-pinned run reuses the same directory.
export const newRunDir = async (): Promise<string> => {
  const dir = runDirOverride() ?? join(outputBaseDir, runStamp())
  await mkdir(dir, { recursive: true })
  return dir
}

// Newest existing run folder. Steps after step1 flow into this. A RUN_DIR override wins,
// so the orchestrator can direct every stage at a specific run dir.
export const latestRunDir = (): string => {
  const override = runDirOverride()
  if (override) {
    return override
  }
  const dir = runDirs().at(-1)
  if (!dir) {
    throw new Error(`No run folders in ${outputBaseDir} — run step1 to start a run`)
  }
  return dir
}
