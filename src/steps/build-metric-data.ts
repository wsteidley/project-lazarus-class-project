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
// uv contract: the IRENA extractor needs Python to read a .xlsb, and a missing uv is an
// ERROR you must explicitly override — never an implicit skip. This is unlike
// resolve:fuzzy, where skipping the optional Splink tier costs nothing. Here a silent skip
// would corrupt the audit itself: the loop above deletes capacity_series.csv first, so a
// run that declines to regenerate it and still exits 0 leaves a phantom deletion sitting
// in the very diff the audit is being read as. The audit runs with no flags, so it always
// gets the strict path.
//
//   uv present                              -> run normally (flag irrelevant)
//   uv absent, no flag                      -> hard fail, nothing written
//   uv absent, --allow-stale-metric-data,
//     committed capacity_series.csv present  -> warn, reuse it, skip the extractor
//   uv absent, flag, that file missing       -> hard fail; nothing to fall back to
//
// The flag is named for what you accept (stale data), not for what it disables, so it
// isn't reached for reflexively. A crashed irena_capacity.py — as distinct from an absent
// uv — is a hard failure in every case, flag or not.

const { values: options } = parseArgs({
  options: { 'allow-stale-metric-data': { type: 'boolean', default: false } },
})

const OWID_SOLAR_PRICES = join(sourcesDir, 'owid/solar-pv-prices/solar-pv-prices.csv')
const OWID_SOLAR_CAPACITY = join(
  sourcesDir,
  'owid/installed-solar-pv-capacity/installed-solar-pv-capacity.csv',
)
const IRENA_XLSB = join(sourcesDir, 'irena/IRENA_Stats_Tool_v2.xlsb')
const CAPACITY_SERIES = 'capacity_series.csv'

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
      `\n✗ uv not found — cannot regenerate ${CAPACITY_SERIES} (IRENA wind extractor requires uv).\n` +
        '  Install uv (https://docs.astral.sh/uv/), or re-run with --allow-stale-metric-data\n' +
        '  to use the existing committed file.',
    )
    return 'refuse'
  }
  if (!existsSync(derivedFile(CAPACITY_SERIES))) {
    console.error(
      `\n✗ uv not found, and --allow-stale-metric-data has nothing to fall back to:\n` +
        `  ${derivedFile(CAPACITY_SERIES)} does not exist.\n` +
        '  Install uv (https://docs.astral.sh/uv/) — there is no committed file to reuse.',
    )
    return 'refuse'
  }

  console.warn(`\n⚠ uv not found — reusing the committed ${CAPACITY_SERIES} (STALE).`)
  console.warn('  Onshore wind capacity was NOT regenerated from the IRENA workbook.')
  console.warn(
    '  Install uv and re-run without --allow-stale-metric-data before trusting a diff.\n',
  )
  return 'reuse-committed'
}

// Runs the Python half of the IRENA extractor and hands back its flat scratch CSV. Only
// called once resolveIrenaPlan has confirmed uv is present. The scratch file is transport,
// not a store: it lives in a temp dir and is removed here, so this step writes nothing
// under data/ but its own output.
const readIrenaScratch = async (): Promise<CsvRow[]> => {
  const scratchDir = mkdtempSync(join(tmpdir(), 'lazarus-irena-'))
  try {
    const scratch = join(scratchDir, 'irena_onshore_wind.csv')
    const run = spawnSync(
      'uv',
      [
        'run',
        '--project',
        'extract',
        'extract/irena_capacity.py',
        '--xlsb',
        IRENA_XLSB,
        '--out',
        scratch,
      ],
      { stdio: 'inherit' },
    )
    if (run.status !== 0) {
      // A crashed extractor is a real error, never a graceful skip.
      throw new Error(
        `extract/irena_capacity.py exited with status ${run.status ?? 'null (signal)'}`,
      )
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
  const regenerateCapacity = irena === 'run'

  console.log('Extracting metric data from data/sources + data/curated ...')

  const solarCost = extractOwidSolarCost(await readCsv(OWID_SOLAR_PRICES))
  report('owid-solar-cost', solarCost)
  observations.push(...solarCost.rows)
  unparsed.push(...solarCost.unparsed)

  const citedAnchors = extractCitedAnchors(await readCsv(curatedFile('cited_anchors.csv')))
  report('cited-anchors', citedAnchors)
  observations.push(...citedAnchors.rows)
  unparsed.push(...citedAnchors.unparsed)

  const solarCapacity = extractOwidSolarCapacity(await readCsv(OWID_SOLAR_CAPACITY))
  report('owid-solar-capacity', solarCapacity)
  capacity.push(...solarCapacity.rows)
  unparsed.push(...solarCapacity.unparsed)

  if (regenerateCapacity) {
    const wind = extractIrenaCapacity(await readIrenaScratch())
    report('irena-capacity', wind)
    capacity.push(...wind.rows)
    unparsed.push(...wind.unparsed)
  } else {
    console.log(`  irena-capacity: SKIPPED — reusing the committed ${CAPACITY_SERIES}`)
  }

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
  await writeCsv(sortObservations(observations), derivedFile('metric_observations_full.csv'))

  if (regenerateCapacity) {
    await writeCsv(sortCapacity(capacity), derivedFile(CAPACITY_SERIES))
  } else {
    // Reached only under --allow-stale-metric-data. Writing what we have would replace the
    // committed file with one missing its entire wind series — the corruption the flag is
    // explicitly not allowed to cause.
    console.warn(`\nNOT writing ${CAPACITY_SERIES} — the committed file is left untouched.`)
    console.warn('  It is stale with respect to the IRENA workbook until uv is available.')
  }

  console.log(
    `\n${observations.length} metric observations` +
      (regenerateCapacity
        ? `, ${capacity.length} capacity points`
        : ' (capacity series left stale)') +
      (unparsed.length > 0 ? ` (${unparsed.length} unparsed — see warnings above)` : ''),
  )
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
