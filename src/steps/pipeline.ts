import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync } from 'node:fs'
import { basename } from 'node:path'
import { parseArgs } from 'node:util'
import { config } from '../config.js'
import { latestRunDir, newRunDir } from '../lib/paths.js'
import { apiStages, resolvePlan, SMOKE_ENV, type StageDef } from '../lib/pipeline.js'
import {
  canRun,
  emptyManifest,
  invalidateDownstream,
  loadManifest,
  type Manifest,
  markFailed,
  markOk,
  markSkipped,
  markStarted,
  saveManifest,
} from '../lib/pipeline-manifest.js'
import { runStamp, utcTimestamp } from '../lib/timestamp.js'

// One entry point that encodes the stage order in code, so the correctness rules the
// architecture depends on are enforced by the machine rather than by whoever is typing.
// Each stage runs as a separate `npm run <script>` process — config is captured at module
// load, so a child process is the only way per-stage env overrides (RUN_DIR, PROCESSING_LIMIT)
// take effect.

const parsed = parseArgs({
  options: {
    from: { type: 'string' },
    only: { type: 'string' },
    through: { type: 'string' },
    'run-dir': { type: 'string' },
    'dry-run': { type: 'boolean', default: false },
    smoke: { type: 'boolean', default: false },
    'continue-on-error': { type: 'boolean', default: false },
    force: { type: 'boolean', default: false },
    limit: { type: 'string' },
    scope: { type: 'string' },
  },
  allowPositionals: false,
})
const flags = parsed.values

// Config the run advertises and passes to every child. --smoke wins, then explicit flags,
// then the process defaults captured in config. All-string so it drops straight into a
// child process env.
const configSnapshot: Record<string, string> = {
  PROCESSING_LIMIT: flags.smoke
    ? SMOKE_ENV.PROCESSING_LIMIT
    : (flags.limit ?? String(config.processingLimit)),
  CONFIDENCE_SCOPE: flags.scope ?? config.confidenceScope,
  ...(flags.smoke ? { CONFIDENCE_SAMPLES: SMOKE_ENV.CONFIDENCE_SAMPLES } : {}),
}

// Counts data rows (lines minus header) in a run-dir CSV, for the summary line / manifest.
const countRows = (runDir: string, filename?: string): number | undefined => {
  if (!filename) {
    return undefined
  }
  const path = `${runDir}/${filename}`
  if (!existsSync(path)) {
    return undefined
  }
  const lines = readFileSync(path, 'utf-8')
    .split('\n')
    .filter((line) => line.trim().length > 0)
  return Math.max(0, lines.length - 1)
}

const stderrTail = (stderr: string, lines = 20): string =>
  stderr
    .split('\n')
    .filter((line) => line.length > 0)
    .slice(-lines)
    .join('\n')

const printPlan = (plan: StageDef[], runDir: string): void => {
  console.log(`\nRun dir: ${runDir}`)
  console.log(`Config:  ${JSON.stringify(configSnapshot)}`)
  console.log('\nStages:')
  for (const stage of plan) {
    const cost = stage.makesApiCalls ? ' [API $]' : ''
    const opt = stage.optional ? ' (optional)' : ''
    console.log(`  ${stage.id}${opt}${cost} — ${stage.invariant}`)
    console.log(`      in:  ${stage.inputs.join(', ')}`)
    console.log(`      out: ${stage.outputs.join(', ')}`)
  }
  const api = apiStages(plan)
  if (api.length > 0) {
    console.log(
      `\nAPI-spending stages (PROCESSING_LIMIT=${configSnapshot.PROCESSING_LIMIT}): ` +
        api.map((s) => s.id).join(', '),
    )
  } else {
    console.log('\nNo API-spending stages in this plan.')
  }
}

// Resolves the run dir from the flags: an explicit --run-dir wins; any resume selector
// (--from/--only/--through) operates on the latest run dir; a bare run starts fresh.
const resolveRunDir = async (): Promise<string> => {
  if (flags['run-dir']) {
    mkdirSync(flags['run-dir'], { recursive: true })
    return flags['run-dir']
  }
  const resuming = Boolean(flags.from || flags.only || flags.through)
  return resuming ? latestRunDir() : await newRunDir()
}

const main = async (): Promise<void> => {
  const plan = resolvePlan({ from: flags.from, only: flags.only, through: flags.through })
  const runDir = await resolveRunDir()
  // Every child stage lands in this run dir and reads this config.
  process.env.RUN_DIR = runDir

  if (flags['dry-run']) {
    console.log('\n=== DRY RUN — no stages executed, no side effects ===')
    printPlan(plan, runDir)
    console.log('')
    return
  }

  const manifest: Manifest = loadManifest(runDir) ?? emptyManifest(runStamp(), configSnapshot)
  manifest.config = configSnapshot
  printPlan(plan, runDir)
  console.log('')

  for (const stage of plan) {
    // Re-running a completed stage invalidates its downstream so a later resume can't mix
    // fresh upstream with stale output.
    if (manifest.stages[stage.id]?.status === 'ok') {
      const staled = invalidateDownstream(manifest, stage.id)
      if (staled.length > 0) {
        console.log(`  (re-running ${stage.id} → marked stale: ${staled.join(', ')})`)
      }
    }

    // Enforcement: prerequisites (and non-stale-ness) must hold, unless --force.
    const gate = canRun(manifest, stage)
    if (!gate.ok && !flags.force) {
      console.error(
        `\n✗ ${stage.id} refuses to run — prerequisites not ok: ${gate.missing.join(', ')}.\n` +
          '  Run them first, or pass --force to override.',
      )
      saveManifest(runDir, manifest)
      process.exitCode = 1
      return
    }

    // build refuses to run from a partial manifest (a run that continued past a failure)
    // without --force.
    if (stage.id === 'build' && manifest.partial && !flags.force) {
      console.error('\n✗ build refuses to run from a partial manifest — pass --force to override.')
      saveManifest(runDir, manifest)
      process.exitCode = 1
      return
    }

    const startedAt = utcTimestamp()
    markStarted(manifest, stage.id, startedAt)
    saveManifest(runDir, manifest)
    const startMs = Date.now()

    // stdout inherited (live progress); stderr piped so we can capture the tail on failure
    // and detect an optional stage's ##SKIP## marker.
    const result = spawnSync('npm', ['run', stage.script], {
      stdio: ['ignore', 'inherit', 'pipe'],
      encoding: 'utf-8',
      maxBuffer: 64 * 1024 * 1024,
      env: { ...process.env, RUN_DIR: runDir, ...configSnapshot },
    })
    const seconds = ((Date.now() - startMs) / 1000).toFixed(1)
    const stderr = result.stderr ?? ''

    if (result.status === 0) {
      const skip = stage.optional && /##SKIP##/.test(stderr)
      if (skip) {
        const reason = (stderr.match(/##SKIP##\s*(.*)/)?.[1] ?? 'optional stage skipped').trim()
        markSkipped(manifest, stage.id, reason)
        console.log(`○ ${stage.id}  ${seconds}s  skipped — ${reason}`)
      } else {
        const rowsOut = countRows(runDir, stage.primaryOutput)
        markOk(manifest, stage.id, utcTimestamp(), rowsOut)
        console.log(
          `✓ ${stage.id}  ${seconds}s${rowsOut === undefined ? '' : `  rows_out=${rowsOut}`}`,
        )
      }
      saveManifest(runDir, manifest)
      continue
    }

    // Non-zero: a real failure (an optional stage that is present-but-failed included).
    const tail = stderrTail(stderr)
    if (tail) {
      console.error(tail)
    }
    markFailed(manifest, stage.id, utcTimestamp(), tail)
    console.error(`✗ ${stage.id}  ${seconds}s  failed (exit ${result.status ?? 'signal'})`)

    if (flags['continue-on-error']) {
      manifest.partial = true
      saveManifest(runDir, manifest)
      continue
    }
    saveManifest(runDir, manifest)
    console.error(`\nHalted. Resume with:  npm run pipeline -- --from ${stage.id}`)
    process.exitCode = 1
    return
  }

  saveManifest(runDir, manifest)
  console.log(`\nDONE — manifest at ${basename(runDir)}/manifest.json\n`)
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
