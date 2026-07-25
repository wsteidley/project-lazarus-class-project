import { spawnSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
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
// uv contract, matching resolve:fuzzy: the IRENA extractor needs Python to read a .xlsb.
// Without uv it is skipped and the other three extractors still run — but a derived file
// missing one of its extractors is NOT written, because overwriting the committed file
// with a truncated one is precisely the corruption the audit exists to catch. A crashed
// irena_capacity.py, as distinct from an absent uv, is always a hard failure.

const OWID_SOLAR_PRICES = join(sourcesDir, 'owid/solar-pv-prices/solar-pv-prices.csv')
const OWID_SOLAR_CAPACITY = join(
  sourcesDir,
  'owid/installed-solar-pv-capacity/installed-solar-pv-capacity.csv',
)
const IRENA_XLSB = join(sourcesDir, 'irena/IRENA_Stats_Tool_v2.xlsb')

// Runs the Python half of the IRENA extractor and hands back its flat scratch CSV, or
// null when uv is absent. The scratch file is transport, not a store: it lives in a temp
// dir and is removed here, so this step writes nothing under data/ but its own output.
const readIrenaScratch = async (): Promise<CsvRow[] | null> => {
  const probe = spawnSync('uv', ['--version'], { stdio: 'ignore' })
  if (probe.error && (probe.error as NodeJS.ErrnoException).code === 'ENOENT') {
    console.warn('uv is not installed — skipping the IRENA .xlsb extractor (onshore wind).')
    console.warn('  Install uv to include the wind capacity series: https://docs.astral.sh/uv/')
    return null
  }
  if (probe.error) {
    throw probe.error
  }

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

  const irenaScratch = await readIrenaScratch()
  if (irenaScratch) {
    const wind = extractIrenaCapacity(irenaScratch)
    report('irena-capacity', wind)
    capacity.push(...wind.rows)
    unparsed.push(...wind.unparsed)
  } else {
    console.log('  irena-capacity: SKIPPED (uv not installed)')
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

  if (irenaScratch) {
    await writeCsv(sortCapacity(capacity), derivedFile('capacity_series.csv'))
  } else {
    console.warn(
      '\nNOT writing capacity_series.csv — the onshore wind series is missing (uv absent).',
    )
    console.warn(
      '  The committed file is left as it is rather than overwritten with a partial one.',
    )
  }

  console.log(
    `\n${observations.length} metric observations, ${capacity.length} capacity points` +
      (unparsed.length > 0 ? ` (${unparsed.length} unparsed — see warnings above)` : ''),
  )
  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
