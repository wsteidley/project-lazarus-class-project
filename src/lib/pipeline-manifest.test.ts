import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { stageById } from './pipeline.js'
import {
  canRun,
  emptyManifest,
  invalidateDownstream,
  loadManifest,
  markFailed,
  markOk,
  markSkipped,
  markStarted,
  saveManifest,
  stageStatus,
} from './pipeline-manifest.js'

const stage = (id: string) => {
  const s = stageById(id)
  if (!s) throw new Error(`no stage ${id}`)
  return s
}

describe('manifest round-trip', () => {
  let dir: string
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'lazarus-manifest-'))
  })
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true })
  })

  it('persists and reloads stage records and config', () => {
    const manifest = emptyManifest('2026-07-22T00-00-00Z', { PROCESSING_LIMIT: 2 })
    markStarted(manifest, 'step1', 'T1')
    markOk(manifest, 'step1', 'T2', 42)
    saveManifest(dir, manifest)

    const reloaded = loadManifest(dir)
    expect(reloaded?.run_id).toBe('2026-07-22T00-00-00Z')
    expect(reloaded?.stages.step1).toMatchObject({ status: 'ok', rows_out: 42 })
    expect(reloaded?.config).toEqual({ PROCESSING_LIMIT: 2 })
  })

  it('loadManifest returns null when none exists', () => {
    expect(loadManifest(dir)).toBeNull()
  })
})

describe('canRun (prerequisite enforcement)', () => {
  it('refuses a stage whose prereq is not ok, naming the missing prereqs', () => {
    const manifest = emptyManifest('r', {})
    // derive needs outcome-pass AND step3; neither has run.
    const gate = canRun(manifest, stage('derive'))
    expect(gate.ok).toBe(false)
    expect(gate.missing).toEqual(['outcome-pass', 'step3'])
  })

  it('allows a stage once every prereq is ok', () => {
    const manifest = emptyManifest('r', {})
    markOk(manifest, 'outcome-pass', 'T', 1)
    markOk(manifest, 'step3', 'T', 1)
    expect(canRun(manifest, stage('derive')).ok).toBe(true)
  })

  it('treats a stale prereq as not runnable', () => {
    const manifest = emptyManifest('r', {})
    markOk(manifest, 'outcome-pass', 'T', 1)
    markOk(manifest, 'step3', 'T', 1)
    markOk(manifest, 'derive', 'T', 1)
    // Re-running step3 stales derive; build (needs derive) can no longer run.
    invalidateDownstream(manifest, 'step3')
    expect(stageStatus(manifest, 'derive')).toBe('stale')
    expect(canRun(manifest, stage('build')).ok).toBe(false)
  })
})

describe('invalidateDownstream', () => {
  it('marks only ok downstream stages stale, and reports them', () => {
    const manifest = emptyManifest('r', {})
    for (const id of ['step0', 'step1', 'step1b', 'step1c', 'resolve', 'enrich']) {
      markOk(manifest, id, 'T', 1)
    }
    markSkipped(manifest, 'resolve:fuzzy', 'uv absent')
    const staled = invalidateDownstream(manifest, 'resolve')
    expect(staled).toContain('enrich')
    // A skipped optional stage is not flipped to stale.
    expect(stageStatus(manifest, 'resolve:fuzzy')).toBe('skipped')
    // An upstream stage is untouched.
    expect(stageStatus(manifest, 'step1')).toBe('ok')
  })
})

describe('markFailed', () => {
  it('records the status and stderr tail', () => {
    const manifest = emptyManifest('r', {})
    markFailed(manifest, 'step2', 'T', 'boom\ntraceback')
    expect(manifest.stages.step2).toMatchObject({
      status: 'failed',
      stderr_tail: 'boom\ntraceback',
    })
  })
})
