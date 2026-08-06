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
  //
  // AC is an ENERGY basis, not a currency. It sat in `basis` until the three-way split, which
  // is the clearest instance of the conflation that split exists to undo -- a capacity figure
  // has no currency vintage at all, so `basis` is correctly 'na' here.
  it('declares AC as the energy basis, not the currency, and rounds to two decimals', () => {
    const result = extractOwidSolarCapacity(rows)
    expect(
      result.rows.map((row) => [row.as_of, row.value, row.basis, row.energy_basis, row.segment]),
    ).toEqual([
      ['2000-12', '1.22', 'na', 'AC', 'all'],
      ['2024-12', '1866.31', 'na', 'AC', 'all'],
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

  // 'onshore' names WHICH SUBSET was measured, which is a segment by definition. It lived in
  // `basis` only because capacity_series had no segment column to put it in -- it was never a
  // basis of any kind.
  it('converts MW to GW and declares onshore as a segment, not a basis', () => {
    expect(
      extractIrenaCapacity(rows).rows.map((row) => [row.as_of, row.value, row.basis, row.segment]),
    ).toEqual([
      ['2000-12', '16.894', 'na', 'onshore'],
      ['2024-12', '1049.778', 'na', 'onshore'],
    ])
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
    entity_name: 'Lithium-ion battery',
    metric: 'battery pack price',
    value: '108',
    unit: 'USD/kWh',
    // BNEF does not state the vintage of its pack-price series, so real_usd stays its own
    // basis rather than being promoted to a year we would be inventing.
    basis: 'real_usd',
    energy_basis: 'na',
    duration: 'na',
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
        entity_name: 'Lithium-ion battery',
        metric: 'cumulative Li-ion deployed',
        value: '3500',
        unit: 'GWh',
        // 'energy' here was just restating the unit; the split retires it.
        basis: 'na',
        energy_basis: 'na',
        duration: 'na',
        segment: 'all',
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
  it('sorts observations by entity, metric, scope, then date', () => {
    const rows = sortObservations([
      { entity_name: 'B', metric: 'm', scope: 'global', as_of: '2020-12' },
      { entity_name: 'A', metric: 'm', scope: 'US', as_of: '2019-12' },
      { entity_name: 'A', metric: 'm', scope: 'US', as_of: '2018-12' },
      // biome-ignore lint/suspicious/noExplicitAny: partial rows are enough to test ordering
    ] as any)
    expect(rows.map((row) => [row.entity_name, row.as_of])).toEqual([
      ['A', '2018-12'],
      ['A', '2019-12'],
      ['B', '2020-12'],
    ])
  })

  it('sorts capacity by entity, metric, scope, then date', () => {
    const rows = sortCapacity([
      { entity_name: 'Solar', metric: 'm', scope: 'global', as_of: '2024-12' },
      { entity_name: 'Onshore', metric: 'm', scope: 'global', as_of: '2000-12' },
      // biome-ignore lint/suspicious/noExplicitAny: partial rows are enough to test ordering
    ] as any)
    expect(rows.map((row) => row.entity_name)).toEqual(['Onshore', 'Solar'])
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
  it('carries the published technology through as the entity name', () => {
    const { observations } = extractIrenaRpgc([irenaRow({})])
    expect(observations.rows[0]).toMatchObject({
      entity_name: 'Onshore wind',
      metric: 'LCOE',
      segment: 'onshore',
      value: '32.95',
    })
  })

  // The five orphan technologies. These used to be HELD — 120 valid rows with nowhere to live,
  // because metric_observations.dependency_id was NOT NULL and no dependency named them. The
  // split gave them entities of their own, so they now load like any other technology and
  // whether a dependency hangs off them is a separate question.
  it('loads a technology that no dependency names, instead of holding it back', () => {
    const { observations } = extractIrenaRpgc([
      irenaRow({ technology: 'Geothermal', segment: 'all' }),
      irenaRow({ technology: 'Geothermal', segment: 'all', as_of: '2024-12' }),
      irenaRow({ technology: 'Hydropower', segment: 'all' }),
    ])
    expect(observations.rows.map((row) => row.entity_name)).toEqual([
      'Geothermal',
      'Geothermal',
      'Hydropower',
    ])
    expect(observations.excluded).toBe(0)
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
        entity_name: 'Lithium-ion battery',
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
    expect(capacity.rows[0]?.entity_name).toBe('Lithium-ion battery cost')
    // The truncation biases the learning rate, so it has to travel with the data.
    expect(capacity.rows[0]?.note).toContain('EXCLUDED')
  })

  it('stamps the battery series with what IRENA actually measured', () => {
    const { observations } = extractIrenaRpgc([
      irenaRow({
        entity_name: 'Lithium-ion battery',
        metric: 'battery_installed_cost',
        value: '140.45',
        unit: 'USD/kWh',
        segment: 'all',
      }),
    ])
    // Four battery cost definitions span $70-140/kWh for the same year; an unlabelled row is
    // how they get spliced into one curve. The two measurement qualifiers used to ride in the
    // note because there were no columns for them -- now they are join keys, so a bar declared
    // per nameplate kWh or on a straight 4h system cannot silently match these rows.
    expect(observations.rows[0]).toMatchObject({
      energy_basis: 'usable',
      duration: 'blended',
    })
    // What stays in the note is provenance rather than a misfiled column.
    expect(observations.rows[0]?.note).toContain('not interchangeable')
  })

  it('leaves the energy axes na on the metrics that have neither', () => {
    const { observations } = extractIrenaRpgc([
      irenaRow({ technology: 'Onshore wind', metric: 'LCOE', value: '33', unit: 'USD/MWh' }),
    ])
    expect(observations.rows[0]).toMatchObject({ energy_basis: 'na', duration: 'na' })
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
      entity_name: 'Onshore wind',
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
