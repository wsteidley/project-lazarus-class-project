import { spawnSync } from 'node:child_process'
import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { parseArgs } from 'node:util'
import { derivedDir, sourcesDir } from '../config.js'
import { type CsvRow, readCsv, writeCsv } from '../lib/csv.js'
import { extractCapacityAnchors, extractCitedAnchors } from '../lib/extractors/cited-anchors.js'
import { extractIrenaCapacity } from '../lib/extractors/irena-capacity.js'
import { extractIrenaRpgc } from '../lib/extractors/irena-rpgc.js'
import { extractIrenaTic } from '../lib/extractors/irena-tic.js'
import { extractOwidSolarCapacity } from '../lib/extractors/owid-solar-capacity.js'
import { extractOwidSolarCost } from '../lib/extractors/owid-solar-cost.js'
import {
  type CapacityRow,
  type ExtractResult,
  type ObservationRow,
  sortCapacity,
  sortObservations,
  type Unparsed,
} from '../lib/extractors/types.js'
import { curatedFile, derivedFile } from '../lib/paths.js'

// build-metric-data — regenerates data/derived/** from data/sources/** (raw, immutable
// provider files) plus data/curated/** (human-authored inputs). No LLM, no network, so it
// is always cheap and always safe to re-run.
//
// The point of the step is auditability: the committed derived files must be exactly what
// these extractors produce from the committed raw files, so
//   rm -f data/derived/*.csv && npm run build-metric-data && git diff
// answers "does this number come from where it claims to?" for the whole dataset at once.
// That only holds if the output is deterministic — hence fixed rounding, a stable sort,
// and no timestamps anywhere in the output.
//
// uv contract: two IRENA extractors need Python to read binary/zipped workbooks — the
// .xlsb capacity series and the .xlsx wind installed-cost series — and a missing uv is an
// ERROR you must explicitly override, never an implicit skip. This is unlike
// resolve:fuzzy, where skipping the optional Splink tier costs nothing. Here a silent skip
// would corrupt the audit itself: the loop above deletes the derived CSVs first, so a
// run that declines to regenerate them and still exits 0 leaves a phantom deletion sitting
// in the very diff the audit is being read as. The audit runs with no flags, so it always
// gets the strict path.
//
//   uv present                              -> run normally (flag irrelevant)
//   uv absent, no flag                      -> hard fail, nothing written
//   uv absent, --allow-stale-metric-data,
//     both committed derived CSVs present    -> warn, reuse them, skip BOTH extractors
//   uv absent, flag, either file missing     -> hard fail; nothing to fall back to
//
// BOTH derived files are covered, not just capacity: wind installed cost feeds
// metric_observations, so a uv-less run that still rewrote the observations file would
// drop 16 rows from it silently — the partial-write corruption this flag exists to
// prevent, just in the other file.
//
// The flag is named for what you accept (stale data), not for what it disables, so it
// isn't reached for reflexively. A crashed extractor — as distinct from an absent uv — is
// a hard failure in every case, flag or not.

const { values: options } = parseArgs({
  options: { 'allow-stale-metric-data': { type: 'boolean', default: false } },
})

const OWID_SOLAR_PRICES = join(sourcesDir, 'owid/solar-pv-prices/solar-pv-prices.csv')
const OWID_SOLAR_CAPACITY = join(
  sourcesDir,
  'owid/installed-solar-pv-capacity/installed-solar-pv-capacity.csv',
)
const IRENA_XLSB = join(sourcesDir, 'irena/IRENA_Stats_Tool_v2.xlsb')
// The RPGC cost workbook. The wind installed-cost series is read from it directly (via
// Python); the other RPGC metrics are still hand-staged CSVs alongside it — see the
// derivation map's standing TODO to move those onto this file too.
const IRENA_XLSX = join(sourcesDir, 'irena/IRENA_TEC_RPGC_in_2025_data_file_2026.xlsx')
// Hand-staged extracts of the workbook above. Plain CSV reads, so these need no uv.
const IRENA_LCOE = join(sourcesDir, 'irena/irena_lcoe_series.csv')
const IRENA_EXTENDED = join(sourcesDir, 'irena/irena_rpgc_extended.csv')
const CAPACITY_SERIES = 'capacity_series.csv'
const OBSERVATIONS = 'metric_observations_full.csv'

// Decides the uv question once, up front, before any extractor runs — so a refusal costs
// nothing and can never leave a half-updated data/derived behind. `refuse` follows the
// orchestrator's convention for a declined run (message + non-zero exit, not a throw), so
// the reason reads as a decision rather than a stack trace.
type IrenaPlan = 'run' | 'reuse-committed' | 'refuse'

const resolveIrenaPlan = (): IrenaPlan => {
  const probe = spawnSync('uv', ['--version'], { stdio: 'ignore' })
  if (!probe.error) {
    return 'run'
  }
  if ((probe.error as NodeJS.ErrnoException).code !== 'ENOENT') {
    throw probe.error
  }

  if (!options['allow-stale-metric-data']) {
    console.error(
      `\n✗ uv not found — cannot regenerate ${CAPACITY_SERIES} or ${OBSERVATIONS}\n` +
        '  (the IRENA capacity .xlsb and wind installed-cost .xlsx extractors both require uv).\n' +
        '  Install uv (https://docs.astral.sh/uv/), or re-run with --allow-stale-metric-data\n' +
        '  to use the existing committed files.',
    )
    return 'refuse'
  }
  // Both files, not just capacity: wind installed cost feeds the observations file, so
  // falling back on one while rewriting the other would produce exactly the half-updated
  // data/derived this gate exists to prevent.
  const missing = [CAPACITY_SERIES, OBSERVATIONS].filter((file) => !existsSync(derivedFile(file)))
  if (missing.length > 0) {
    console.error(
      `\n✗ uv not found, and --allow-stale-metric-data has nothing to fall back to:\n` +
        missing.map((file) => `  ${derivedFile(file)} does not exist.\n`).join('') +
        '  Install uv (https://docs.astral.sh/uv/) — there is no committed file to reuse.',
    )
    return 'refuse'
  }

  console.warn(
    `\n⚠ uv not found — reusing the committed ${CAPACITY_SERIES} + ${OBSERVATIONS} (STALE).`,
  )
  console.warn('  Onshore wind capacity and wind installed cost were NOT regenerated from the')
  console.warn('  IRENA workbooks; every other extractor was skipped too, because a partial')
  console.warn('  rewrite of these files is worse than leaving them alone.')
  console.warn(
    '  Install uv and re-run without --allow-stale-metric-data before trusting a diff.\n',
  )
  return 'reuse-committed'
}

// Runs the Python half of an IRENA extractor and hands back its flat scratch CSV. Only
// called once resolveIrenaPlan has confirmed uv is present. The scratch file is transport,
// not a store: it lives in a temp dir and is removed here, so this step writes nothing
// under data/ but its own output.
const readPythonScratch = async (
  script: string,
  sourceFlag: string,
  sourcePath: string,
): Promise<CsvRow[]> => {
  const scratchDir = mkdtempSync(join(tmpdir(), 'lazarus-irena-'))
  try {
    const scratch = join(scratchDir, 'scratch.csv')
    const run = spawnSync(
      'uv',
      ['run', '--project', 'extract', script, sourceFlag, sourcePath, '--out', scratch],
      { stdio: 'inherit' },
    )
    if (run.status !== 0) {
      // A crashed extractor is a real error, never a graceful skip.
      throw new Error(`${script} exited with status ${run.status ?? 'null (signal)'}`)
    }
    return await readCsv(scratch)
  } finally {
    rmSync(scratchDir, { recursive: true, force: true })
  }
}

const report = (label: string, result: ExtractResult<unknown>): void => {
  const parts = [`${result.rows.length} rows`]
  if (result.excluded > 0) {
    parts.push(`${result.excluded} excluded by filter`)
  }
  if (result.unparsed.length > 0) {
    parts.push(`${result.unparsed.length} UNPARSED`)
  }
  console.log(`  ${label}: ${parts.join(', ')}`)
}

const main = async (): Promise<void> => {
  const observations: ObservationRow[] = []
  const capacity: CapacityRow[] = []
  const unparsed: Unparsed[] = []

  // Settle the uv question before doing any work, so a refusal costs nothing and can never
  // leave a half-updated data/derived behind.
  const irena = resolveIrenaPlan()
  if (irena === 'refuse') {
    process.exitCode = 1
    return
  }
  // Both derived files ride on the same decision now: the .xlsb feeds capacity, the .xlsx
  // feeds observations, and rewriting one while the other goes stale is the half-updated
  // state the gate refuses to produce.
  const regenerate = irena === 'run'

  console.log('Extracting metric data from data/sources + data/curated ...')

  const solarCost = extractOwidSolarCost(await readCsv(OWID_SOLAR_PRICES))
  report('owid-solar-cost', solarCost)
  observations.push(...solarCost.rows)
  unparsed.push(...solarCost.unparsed)

  const citedAnchors = extractCitedAnchors(await readCsv(curatedFile('cited_anchors.csv')))
  report('cited-anchors', citedAnchors)
  observations.push(...citedAnchors.rows)
  unparsed.push(...citedAnchors.unparsed)

  // IRENA RPGC: multi-metric, and the first extractor that emits BOTH observations and capacity
  // from the same input (battery cost + the BESS additions cumulated into its Wright axis).
  const irenaRpgc = extractIrenaRpgc([
    ...(await readCsv(IRENA_LCOE)),
    ...(await readCsv(IRENA_EXTENDED)),
  ])
  report('irena-rpgc', irenaRpgc.observations)
  observations.push(...irenaRpgc.observations.rows)
  unparsed.push(...irenaRpgc.observations.unparsed)
  // Named, not just counted: these rows are valid data with nowhere to live until the
  // technology/dependency split lands, and a bare "excluded: 100" would read as a filter
  // working rather than as a queue of held findings.
  for (const [subject, count] of [...irenaRpgc.held].sort()) {
    console.log(`    held for technology/dependency split: ${subject} (${count} rows)`)
  }

  // Wind's Wright-fittable cost curve, read straight from the committed .xlsx. Wind LCOE
  // above is the viability series and is deliberately never fitted; this is the hardware
  // capex series that is.
  if (regenerate) {
    const windCost = extractIrenaTic(
      await readPythonScratch('extract/irena_tic.py', '--xlsx', IRENA_XLSX),
    )
    report('irena-tic', windCost)
    observations.push(...windCost.rows)
    unparsed.push(...windCost.unparsed)
  } else {
    console.log(`  irena-tic: SKIPPED — reusing the committed ${OBSERVATIONS}`)
  }

  const solarCapacity = extractOwidSolarCapacity(await readCsv(OWID_SOLAR_CAPACITY))
  report('owid-solar-capacity', solarCapacity)
  capacity.push(...solarCapacity.rows)
  unparsed.push(...solarCapacity.unparsed)

  if (regenerate) {
    const wind = extractIrenaCapacity(
      await readPythonScratch('extract/irena_capacity.py', '--xlsb', IRENA_XLSB),
    )
    report('irena-capacity', wind)
    capacity.push(...wind.rows)
    unparsed.push(...wind.unparsed)
  } else {
    console.log(`  irena-capacity: SKIPPED — reusing the committed ${CAPACITY_SERIES}`)
  }

  report('irena-rpgc-capacity', irenaRpgc.capacity)
  capacity.push(...irenaRpgc.capacity.rows)

  const capacityAnchors = extractCapacityAnchors(await readCsv(curatedFile('capacity_anchors.csv')))
  report('capacity-anchors', capacityAnchors)
  capacity.push(...capacityAnchors.rows)
  unparsed.push(...capacityAnchors.unparsed)

  // Report-not-drop: every row an extractor could not use is printed, so a build that
  // quietly lost data is impossible to miss in the log.
  for (const entry of unparsed) {
    console.warn(`  UNPARSED [${entry.extractor}] ${entry.reason}: ${entry.raw}`)
  }

  // The audit deletes the derived files before rebuilding, so the step has to be able to
  // recreate the directory itself — otherwise the documented audit command fails.
  await mkdir(derivedDir, { recursive: true })

  if (regenerate) {
    await writeCsv(sortObservations(observations), derivedFile(OBSERVATIONS))
    await writeCsv(sortCapacity(capacity), derivedFile(CAPACITY_SERIES))
  } else {
    // Reached only under --allow-stale-metric-data. Writing what we have would replace the
    // committed files with ones missing the entire wind capacity and wind installed-cost
    // series — the corruption the flag is explicitly not allowed to cause.
    console.warn(
      `\nNOT writing ${OBSERVATIONS} or ${CAPACITY_SERIES} — the committed files are left untouched.`,
    )
    console.warn('  They are stale with respect to the IRENA workbooks until uv is available.')
  }

  console.log(
    regenerate
      ? `\n${observations.length} metric observations, ${capacity.length} capacity points` +
          (unparsed.length > 0 ? ` (${unparsed.length} unparsed — see warnings above)` : '')
      : '\nderived files left stale (uv unavailable)',
  )
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
