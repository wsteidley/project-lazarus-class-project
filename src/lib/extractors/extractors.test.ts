import { describe, expect, it } from 'vitest'
import type { CsvRow } from '../csv.js'
import { extractCapacityAnchors, extractCitedAnchors } from './cited-anchors.js'
import { extractIrenaCapacity } from './irena-capacity.js'
import { extractIrenaRpgc } from './irena-rpgc.js'
import { extractIrenaTic } from './irena-tic.js'
import { readOwidWorldSeries } from './owid.js'
import { extractOwidSolarCapacity } from './owid-solar-capacity.js'
import { extractOwidSolarCost } from './owid-solar-cost.js'
import { fixed, sortCapacity, sortObservations } from './types.js'

const owidPrice = (entity: string, year: string, cost: string): CsvRow => ({
  Entity: entity,
  Code: entity === 'World' ? 'OWID_WRL' : 'XXX',
  Year: year,
  'Solar PV module cost': cost,
})

const owidCapacity = (entity: string, year: string, solar: string): CsvRow => ({
  Entity: entity,
  Code: entity === 'World' ? 'OWID_WRL' : 'XXX',
  Year: year,
  Solar: solar,
})

describe('fixed', () => {
  it('rounds to the requested precision and drops trailing zeros', () => {
    expect(fixed(128.26924, 2)).toBe('128.27')
    expect(fixed(1.220137, 2)).toBe('1.22')
    expect(fixed(2.0004, 2)).toBe('2')
    expect(fixed(16894.0 / 1000, 3)).toBe('16.894')
  })
})

describe('readOwidWorldSeries', () => {
  it('keeps World rows, counts the rest as excluded rather than unparsed', () => {
    const { points, unparsed, excluded } = readOwidWorldSeries(
      [
        owidPrice('World', '1975', '128.26924'),
        owidPrice('Germany', '1975', '99.0'),
        owidPrice('World', '1976', '96.507935'),
      ],
      'Solar PV module cost',
      'test',
    )
    expect(points).toEqual([
      { year: 1975, value: 128.26924 },
      { year: 1976, value: 96.507935 },
    ])
    expect(excluded).toBe(1)
    expect(unparsed).toEqual([])
  })

  it('reports an unparseable value instead of dropping it', () => {
    const { points, unparsed } = readOwidWorldSeries(
      [owidPrice('World', '1975', '128.0'), owidPrice('World', '1976', 'n/a')],
      'Solar PV module cost',
      'test',
    )
    expect(points).toHaveLength(1)
    expect(unparsed).toHaveLength(1)
    expect(unparsed[0]?.reason).toContain('unparseable')
  })

  // An empty series means the download's shape changed. Committing an empty derived file
  // would look like "the data went away" rather than "the extractor broke".
  it('throws when no World rows survive', () => {
    expect(() =>
      readOwidWorldSeries([owidPrice('Germany', '1975', '99.0')], 'Solar PV module cost', 'test'),
    ).toThrow(/no "World" rows/)
  })

  it('throws when the value column is missing entirely', () => {
    expect(() => readOwidWorldSeries([owidPrice('World', '1975', '1')], 'Renamed', 'test')).toThrow(
      /Renamed/,
    )
  })
})

describe('extractOwidSolarCost', () => {
  const rows = [
    owidPrice('World', '1975', '128.26924'),
    owidPrice('World', '1976', '96.507935'),
    owidPrice('World', '2004', '4.5601'),
    owidPrice('World', '2010', '2.4399'),
    owidPrice('World', '2024', '0.2611'),
  ]

  // OWID stitches three producers into one series; attributing every row to "OWID" would
  // lose which one a given number actually came from.
  it('attributes each year to the producer OWID stitched it from', () => {
    const byYear = Object.fromEntries(
      extractOwidSolarCost(rows).rows.map((row) => [row.as_of, row.source_name]),
    )
    expect(byYear['1975-12']).toBe('OWID (Nemet 2009)')
    expect(byYear['2004-12']).toBe('OWID (Farmer & Lafond 2016)')
    expect(byYear['2010-12']).toBe('OWID (IRENA)')
  })

  it('stamps basis, unit and source_url on every row', () => {
    for (const row of extractOwidSolarCost(rows).rows) {
      expect(row.basis).toBe('real_2024_usd')
      expect(row.unit).toBe('USD/W')
      expect(row.source_url).toBe('https://ourworldindata.org/grapher/solar-pv-prices')
    }
  })

  it('marks the attempt-era anchor and the latest point', () => {
    const notes = Object.fromEntries(
      extractOwidSolarCost(rows).rows.map((row) => [row.as_of, row.note]),
    )
    expect(notes['1976-12']).toBe('attempt-era anchor')
    expect(notes['2024-12']).toBe('latest OWID point')
    expect(notes['2004-12']).toBe('')
  })

  it('dates observations to the close of the calendar year', () => {
    expect(extractOwidSolarCost(rows).rows.map((row) => row.as_of)).toEqual([
      '1975-12',
      '1976-12',
      '2004-12',
      '2010-12',
      '2024-12',
    ])
  })
})

describe('extractOwidSolarCapacity', () => {
  const rows = [
    owidCapacity('World', '2000', '1.220137'),
    owidCapacity('Germany', '2000', '0.114'),
    owidCapacity('World', '2024', '1866.3149'),
  ]

  // AC vs DC is the methodology trap here: DC headline figures run ~20% higher post-2021,
  // and pairing them with the AC-consistent cost series would bend the learning rate.
  it('declares the AC basis and rounds to two decimals', () => {
    const result = extractOwidSolarCapacity(rows)
    expect(result.rows.map((row) => [row.as_of, row.value, row.basis])).toEqual([
      ['2000-12', '1.22', 'AC'],
      ['2024-12', '1866.31', 'AC'],
    ])
    expect(result.excluded).toBe(1)
  })

  it('marks the series start and the latest point', () => {
    const notes = extractOwidSolarCapacity(rows).rows.map((row) => row.note)
    expect(notes).toEqual(['OWID/IRENA World series start', 'latest'])
  })
})

describe('extractIrenaCapacity', () => {
  const rows: CsvRow[] = [
    { year: '2000', capacity_mw: '16894.0' },
    { year: '2024', capacity_mw: '1049778.0' },
  ]

  it('converts MW to GW and declares the onshore-only basis', () => {
    expect(extractIrenaCapacity(rows).rows.map((row) => [row.as_of, row.value, row.basis])).toEqual(
      [
        ['2000-12', '16.894', 'onshore'],
        ['2024-12', '1049.778', 'onshore'],
      ],
    )
  })

  it('reports an unusable row rather than dropping it', () => {
    const result = extractIrenaCapacity([...rows, { year: '2025', capacity_mw: '' }])
    expect(result.rows).toHaveLength(2)
    expect(result.unparsed).toHaveLength(1)
  })

  it('throws when nothing at all parsed', () => {
    expect(() => extractIrenaCapacity([{ year: 'x', capacity_mw: 'y' }])).toThrow(/no usable rows/)
  })
})

describe('cited anchors', () => {
  const anchor = (overrides: Partial<CsvRow> = {}): CsvRow => ({
    dependency_name: 'Lithium-ion battery cost',
    metric: 'battery pack price',
    value: '108',
    unit: 'USD/kWh',
    basis: 'real_usd',
    segment: 'all',
    as_of: '2025-12',
    scope: 'global',
    method: 'curated',
    source_url: 'https://about.bnef.com/',
    source_name: 'BNEF 2025 Survey',
    note: '',
    ...overrides,
  })

  it('passes a well-formed anchor through unchanged', () => {
    const result = extractCitedAnchors([anchor()])
    expect(result.rows).toHaveLength(1)
    expect(result.unparsed).toEqual([])
    expect(result.rows[0]?.value).toBe('108')
  })

  // A cited-only number with no citation is unfalsifiable — the one thing this dataset
  // cannot carry, since there is no raw file to trace it back to.
  it('rejects an anchor with no citation', () => {
    const result = extractCitedAnchors([anchor({ source_url: '' })])
    expect(result.rows).toEqual([])
    expect(result.unparsed[0]?.reason).toContain('source_url')
  })

  it('rejects a non-numeric value and a malformed as_of', () => {
    expect(extractCitedAnchors([anchor({ value: 'about 100' })]).unparsed[0]?.reason).toContain(
      'non-numeric',
    )
    expect(extractCitedAnchors([anchor({ as_of: '2025' })]).unparsed[0]?.reason).toContain(
      'YYYY-MM',
    )
  })

  it('drops extra columns and fills missing optional ones', () => {
    const result = extractCitedAnchors([{ ...anchor(), stray_column: 'ignore me' }])
    expect(Object.keys(result.rows[0] ?? {})).not.toContain('stray_column')
  })

  it('validates capacity anchors against the capacity schema', () => {
    const result = extractCapacityAnchors([
      {
        technology: 'Lithium-ion battery cost',
        metric: 'cumulative Li-ion deployed',
        value: '3500',
        unit: 'GWh',
        basis: 'energy',
        as_of: '2024-12',
        scope: 'global',
        scenario: 'historical',
        method: 'curated',
        source_url: 'https://www.sciencedirect.com/',
        source_name: 'Sci.Direct bottom-up model',
        note: 'ANCHOR only',
      },
    ])
    expect(result.rows).toHaveLength(1)
    expect(result.unparsed).toEqual([])
  })
})

describe('deterministic ordering', () => {
  // The rebuild-and-diff audit only works if row order is a function of the data, never
  // of the order extractors happened to run in.
  it('sorts observations by dependency, metric, scope, then date', () => {
    const rows = sortObservations([
      { dependency_name: 'B', metric: 'm', scope: 'global', as_of: '2020-12' },
      { dependency_name: 'A', metric: 'm', scope: 'US', as_of: '2019-12' },
      { dependency_name: 'A', metric: 'm', scope: 'US', as_of: '2018-12' },
      // biome-ignore lint/suspicious/noExplicitAny: partial rows are enough to test ordering
    ] as any)
    expect(rows.map((row) => [row.dependency_name, row.as_of])).toEqual([
      ['A', '2018-12'],
      ['A', '2019-12'],
      ['B', '2020-12'],
    ])
  })

  it('sorts capacity by technology, metric, scope, then date', () => {
    const rows = sortCapacity([
      { technology: 'Solar', metric: 'm', scope: 'global', as_of: '2024-12' },
      { technology: 'Onshore', metric: 'm', scope: 'global', as_of: '2000-12' },
      // biome-ignore lint/suspicious/noExplicitAny: partial rows are enough to test ordering
    ] as any)
    expect(rows.map((row) => row.technology)).toEqual(['Onshore', 'Solar'])
  })
})

const irenaRow = (overrides: Partial<CsvRow>): CsvRow => ({
  technology: 'Onshore wind',
  metric: 'LCOE',
  value: '32.95',
  unit: 'USD/MWh',
  basis: 'real_2025_usd',
  segment: 'onshore',
  as_of: '2025-12',
  scope: 'global',
  method: 'curated',
  source_url: 'https://www.irena.org/x',
  source_name: 'IRENA RPGC 2025',
  note: '',
  ...overrides,
})

describe('extractIrenaRpgc', () => {
  it('aliases a technology onto the dependency the loader resolves through', () => {
    const { observations } = extractIrenaRpgc([irenaRow({})])
    expect(observations.rows[0]).toMatchObject({
      dependency_name: 'Onshore wind LCOE',
      metric: 'LCOE',
      segment: 'onshore',
      value: '32.95',
    })
  })

  // The five orphan technologies. They are valid data with no dependency to hang off yet, and
  // metric_observations.dependency_id is NOT NULL — so they must be held and NAMED, never
  // dropped quietly and never forced in by inventing a dependency.
  it('holds a technology with no dependency, counted by subject rather than dropped', () => {
    const { observations, held } = extractIrenaRpgc([
      irenaRow({ technology: 'Geothermal', segment: 'all' }),
      irenaRow({ technology: 'Geothermal', segment: 'all', as_of: '2024-12' }),
      irenaRow({ technology: 'Hydropower', segment: 'all' }),
    ])
    expect(observations.rows).toEqual([])
    expect(observations.excluded).toBe(3)
    expect(held.get('Geothermal / LCOE')).toBe(2)
    expect(held.get('Hydropower / LCOE')).toBe(1)
  })

  // 'Global' and 'global' would otherwise become two series, and the one that no threshold
  // joins would silently produce nothing at all.
  it('folds scope casing and country names onto one vocabulary', () => {
    const rows = [
      irenaRow({ scope: 'Global' }),
      irenaRow({ scope: 'United States' }),
      irenaRow({ scope: 'Germany' }),
    ]
    const { observations } = extractIrenaRpgc(rows)
    expect(observations.rows.map((row) => row.scope)).toEqual(['global', 'US', 'DE'])
  })

  it('reports an unrecognised scope instead of passing it through', () => {
    const { observations } = extractIrenaRpgc([irenaRow({ scope: 'Atlantis' })])
    expect(observations.rows).toEqual([])
    expect(observations.unparsed[0]?.reason).toContain('unrecognised scope')
  })

  it('cumulates BESS additions into a capacity series and never stores the raw flow', () => {
    const additions = ['2.02', '2.24', '2.99'].map((value, index) =>
      irenaRow({
        technology: 'Lithium-ion battery cost',
        metric: 'bess_additions',
        value,
        unit: 'GWh',
        basis: 'na',
        segment: 'all',
        as_of: `${2015 + index}-12`,
      }),
    )
    const { observations, capacity } = extractIrenaRpgc(additions)
    // An annual flow is not an observation of a level — storing it as one would put a 2.99 point
    // on an axis whose real value that year is 7.25.
    expect(observations.rows).toEqual([])
    expect(capacity.rows.map((row) => [row.as_of, row.value])).toEqual([
      ['2015-12', '2.02'],
      ['2016-12', '4.26'],
      ['2017-12', '7.25'],
    ])
    expect(capacity.rows[0]?.technology).toBe('Utility-scale battery system')
    // The truncation biases the learning rate, so it has to travel with the data.
    expect(capacity.rows[0]?.note).toContain('EXCLUDED')
  })

  it('stamps the battery series with what IRENA actually measured', () => {
    const { observations } = extractIrenaRpgc([
      irenaRow({
        technology: 'Lithium-ion battery cost',
        metric: 'battery_installed_cost',
        value: '140.45',
        unit: 'USD/kWh',
        segment: 'all',
      }),
    ])
    // Four battery cost definitions span $70-140/kWh for the same year; an unlabelled row is
    // how they get spliced into one curve.
    expect(observations.rows[0]?.note).toContain('usable kWh')
    expect(observations.rows[0]?.note).toContain('blended duration')
  })

  it('rejects a row with no citation', () => {
    const { observations } = extractIrenaRpgc([irenaRow({ source_url: '' })])
    expect(observations.rows).toEqual([])
    expect(observations.unparsed[0]?.reason).toContain('without source_url')
  })
})

describe('extractIrenaTic', () => {
  const scratch = (year: string, cost: string): CsvRow => ({ year, cost_usd_per_kw: cost })

  it('stamps the wind installed-cost series with its own basis and segment', () => {
    const { rows } = extractIrenaTic([scratch('2010', '2380.8642270915807')])
    expect(rows[0]).toMatchObject({
      dependency_name: 'Onshore wind LCOE',
      metric: 'total_installed_cost',
      unit: 'USD/kW',
      basis: 'real_2025_usd',
      segment: 'onshore',
      as_of: '2010-12',
      // Two decimals: the audit diff must not carry float noise.
      value: '2380.86',
    })
  })

  it('sorts by year regardless of input order', () => {
    const { rows } = extractIrenaTic([
      scratch('2025', '976.01'),
      scratch('2010', '2380.86'),
      scratch('2018', '1910.28'),
    ])
    expect(rows.map((row) => row.as_of)).toEqual(['2010-12', '2018-12', '2025-12'])
  })

  it('reports an unusable row rather than dropping it', () => {
    const { rows, unparsed } = extractIrenaTic([scratch('2010', 'n/a'), scratch('2011', '2333.88')])
    expect(rows).toHaveLength(1)
    expect(unparsed[0]?.reason).toContain('unparseable cost_usd_per_kw')
  })

  // An empty parse means the workbook layout moved. Returning zero rows would delete the
  // series from the derived file and read as a legitimate diff.
  it('throws when nothing parsed at all', () => {
    expect(() => extractIrenaTic([])).toThrow(/layout changed/)
  })
})
