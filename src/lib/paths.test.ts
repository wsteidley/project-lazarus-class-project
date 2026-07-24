import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// paths.ts derives its dirs from config at import time, so point DATA_DIR at a
// temp dir before importing it.
let tempDir: string

beforeEach(() => {
  tempDir = mkdtempSync(join(tmpdir(), 'lazarus-paths-'))
  vi.resetModules()
  process.env.DATA_DIR = tempDir
})

afterEach(() => {
  rmSync(tempDir, { recursive: true, force: true })
  delete process.env.DATA_DIR
})

describe('latestScrapedDataFile', () => {
  it('picks the newest *_data.csv and ignores the _data_index companion', async () => {
    const scraped = join(tempDir, 'scraped')
    mkdirSync(scraped, { recursive: true })
    writeFileSync(join(scraped, 'techcrunch_article_2026-07-19T10-00-00Z_data.csv'), 'x')
    writeFileSync(join(scraped, 'techcrunch_article_2026-07-20T10-00-00Z_data.csv'), 'x')
    writeFileSync(join(scraped, 'techcrunch_article_2026-07-21T10-00-00Z_data_index.csv'), 'x')

    const { latestScrapedDataFile } = await import('./paths.js')
    expect(basename(latestScrapedDataFile())).toBe(
      'techcrunch_article_2026-07-20T10-00-00Z_data.csv',
    )
  })

  it('throws a helpful error when there is no scrape', async () => {
    const { latestScrapedDataFile } = await import('./paths.js')
    expect(() => latestScrapedDataFile()).toThrow(/step0/)
  })
})

describe('run folders', () => {
  it('newRunDir creates a fresh folder and latestRunDir returns the newest', async () => {
    const { newRunDir, latestRunDir } = await import('./paths.js')

    const first = await newRunDir()
    // Predates `first` so sort order is deterministic regardless of timing.
    const older = join(tempDir, 'output', '2000-01-01T00-00-00Z')
    mkdirSync(older, { recursive: true })

    expect(latestRunDir()).toBe(first)
  })

  it('latestRunDir throws when no run exists', async () => {
    const { latestRunDir } = await import('./paths.js')
    expect(() => latestRunDir()).toThrow(/step1/)
  })

  it('RUN_DIR override pins both newRunDir and latestRunDir', async () => {
    const pinned = join(tempDir, 'output', 'pinned-run')
    process.env.RUN_DIR = pinned
    try {
      const { newRunDir, latestRunDir } = await import('./paths.js')
      // newRunDir returns and creates exactly the override, not a fresh stamp.
      expect(await newRunDir()).toBe(pinned)
      // latestRunDir returns the override even with older stamped folders present.
      mkdirSync(join(tempDir, 'output', '2000-01-01T00-00-00Z'), { recursive: true })
      expect(latestRunDir()).toBe(pinned)
    } finally {
      delete process.env.RUN_DIR
    }
  })
})
