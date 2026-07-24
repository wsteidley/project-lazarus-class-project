import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { type StageDef, transitiveDependents } from './pipeline.js'

// The run manifest is the ordering mechanism. It records, per run dir, which stages have
// completed under which config, so a stage can refuse to run when a prerequisite isn't
// `ok` — enforcement by recorded state, not by file-existence guessing. Re-running a
// completed stage invalidates its downstream (marked `stale`) so a later resume can't mix
// fresh upstream with stale downstream output.

export type StageStatus = 'running' | 'ok' | 'failed' | 'skipped' | 'stale'

export type StageRecord = {
  status: StageStatus
  started_at?: string
  finished_at?: string
  rows_out?: number
  reason?: string
  stderr_tail?: string
}

export type Manifest = {
  run_id: string
  stages: Record<string, StageRecord>
  config: Record<string, string | number>
  // Set when a run proceeded past a failure under --continue-on-error. `build` refuses to
  // run from a partial manifest without --force.
  partial?: boolean
}

export const manifestPath = (runDir: string): string => join(runDir, 'manifest.json')

export const emptyManifest = (
  runId: string,
  config: Record<string, string | number>,
): Manifest => ({ run_id: runId, stages: {}, config })

export const loadManifest = (runDir: string): Manifest | null => {
  const path = manifestPath(runDir)
  if (!existsSync(path)) {
    return null
  }
  return JSON.parse(readFileSync(path, 'utf-8')) as Manifest
}

export const saveManifest = (runDir: string, manifest: Manifest): void => {
  writeFileSync(manifestPath(runDir), `${JSON.stringify(manifest, null, 2)}\n`, 'utf-8')
}

export const stageStatus = (manifest: Manifest, id: string): StageStatus | undefined =>
  manifest.stages[id]?.status

// Whether every prerequisite of `stage` is `ok`. Returns the prereqs that are not, so the
// caller can report precisely why a stage is refused. A `stale` or missing prereq blocks —
// that stale check is exactly what stops a fresh upstream re-run from being consumed by
// stale downstream output without an explicit --force.
export const canRun = (manifest: Manifest, stage: StageDef): { ok: boolean; missing: string[] } => {
  const missing = stage.prereqs.filter((prereq) => manifest.stages[prereq]?.status !== 'ok')
  return { ok: missing.length === 0, missing }
}

export const markStarted = (manifest: Manifest, id: string, startedAt: string): void => {
  manifest.stages[id] = { status: 'running', started_at: startedAt }
}

export const markOk = (
  manifest: Manifest,
  id: string,
  finishedAt: string,
  rowsOut?: number,
): void => {
  manifest.stages[id] = {
    ...manifest.stages[id],
    status: 'ok',
    finished_at: finishedAt,
    rows_out: rowsOut,
  }
}

export const markSkipped = (manifest: Manifest, id: string, reason: string): void => {
  manifest.stages[id] = { ...manifest.stages[id], status: 'skipped', reason }
}

export const markFailed = (
  manifest: Manifest,
  id: string,
  finishedAt: string,
  stderrTail: string,
): void => {
  manifest.stages[id] = {
    ...manifest.stages[id],
    status: 'failed',
    finished_at: finishedAt,
    stderr_tail: stderrTail,
  }
}

// Marks every stage that transitively depends on `id` and is currently `ok` as `stale`.
// Called when an already-completed stage is re-run, so downstream output can't be silently
// trusted. Returns the ids marked, for logging.
export const invalidateDownstream = (manifest: Manifest, id: string): string[] => {
  const staled: string[] = []
  for (const dependent of transitiveDependents(id)) {
    if (manifest.stages[dependent]?.status === 'ok') {
      manifest.stages[dependent] = { ...manifest.stages[dependent], status: 'stale' }
      staled.push(dependent)
    }
  }
  return staled
}
