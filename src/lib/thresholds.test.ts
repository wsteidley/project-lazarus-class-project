import { describe, expect, it } from 'vitest'
import type { CsvRow } from './csv.js'
import {
  validateDependencyLinks,
  validateObservationRows,
  validateThresholds,
} from './thresholds.js'

const CANONICAL = ['Lithium-ion battery cost', 'Carbon price', 'Grid interconnection capacity']

describe('validateObservationRows', () => {
  it('resolves dependency_name to the canonical spelling, tolerating case/whitespace', () => {
    const rows: CsvRow[] = [
      { dependency_name: '  lithium-ion  battery cost ', metric: 'x', method: 'feed' },
    ]
    const { resolved, unmatched } = validateObservationRows(rows, CANONICAL)
    expect(resolved[0]?.dependency_name).toBe('Lithium-ion battery cost')
    expect(unmatched).toHaveLength(0)
  })

  it('reports unmatched dependency_name rather than dropping it', () => {
    const rows: CsvRow[] = [{ dependency_name: 'Fusion power cost', metric: 'x', method: 'feed' }]
    const { resolved, unmatched } = validateObservationRows(rows, CANONICAL)
    expect(resolved).toHaveLength(1)
    expect(resolved[0]?.dependency_name).toBe('')
    expect(unmatched).toHaveLength(1)
  })

  it('flags a curated row missing a source_url', () => {
    const rows: CsvRow[] = [
      { dependency_name: 'Carbon price', method: 'curated', source_url: '' },
      { dependency_name: 'Carbon price', method: 'curated', source_url: 'https://example.org' },
      { dependency_name: 'Carbon price', method: 'feed', source_url: '' },
    ]
    const { missingSource } = validateObservationRows(rows, CANONICAL)
    expect(missingSource).toHaveLength(1)
  })
})

describe('validateThresholds', () => {
  const completeBar = (over: Partial<CsvRow> = {}): CsvRow => ({
    dependency_name: 'Lithium-ion battery cost',
    metric: 'battery pack price',
    scope: 'global',
    threshold_value: '100',
    threshold_unit: 'USD/kWh',
    threshold_direction: 'below_is_better',
    threshold_source_url: 'https://bnef',
    ...over,
  })

  it('resolves threshold rows to canonical names and reports unmatched', () => {
    const deps: CsvRow[] = [
      { name: 'Lithium-ion battery cost', threshold_kind: 'quantitative_with_threshold' },
    ]
    const rows: CsvRow[] = [
      completeBar({ dependency_name: '  lithium-ion battery cost ' }),
      completeBar({ dependency_name: 'Fusion power cost' }),
    ]
    const { resolved, unmatched } = validateThresholds(deps, rows, CANONICAL)
    expect(resolved[0]?.dependency_name).toBe('Lithium-ion battery cost')
    expect(unmatched).toHaveLength(1)
  })

  it('reports a quantitative_with_threshold dependency with no complete bar', () => {
    const deps: CsvRow[] = [
      { name: 'Lithium-ion battery cost', threshold_kind: 'quantitative_with_threshold' },
      { name: 'Carbon price', threshold_kind: 'quantitative_with_threshold' },
    ]
    // Battery has a complete bar; carbon's only bar is missing a direction.
    const rows: CsvRow[] = [
      completeBar(),
      completeBar({
        dependency_name: 'Carbon price',
        metric: 'carbon price',
        threshold_direction: '',
      }),
    ]
    const { incomplete } = validateThresholds(deps, rows, CANONICAL)
    expect(incomplete.map((entry) => entry.name)).toEqual(['Carbon price'])
  })

  it('does not require bars for qualitative or tbd dependencies', () => {
    const deps: CsvRow[] = [
      { name: 'Carbon price', threshold_kind: 'qualitative' },
      { name: 'Grid interconnection capacity', threshold_kind: 'quantitative_tbd' },
    ]
    const { incomplete } = validateThresholds(deps, [], CANONICAL)
    expect(incomplete).toHaveLength(0)
  })
})

describe('validateDependencyLinks', () => {
  it('resolves both endpoints and reports links with an unresolvable end', () => {
    const rows: CsvRow[] = [
      {
        from_dependency: 'Grid interconnection capacity',
        to_dependency: 'Carbon price',
        relation: 'drives',
        note: '',
      },
      {
        from_dependency: 'Carbon price',
        to_dependency: 'Fusion power cost',
        relation: 'blocks',
        note: '',
      },
    ]
    const { resolved, unmatched } = validateDependencyLinks(rows, CANONICAL)
    expect(resolved).toHaveLength(1)
    expect(resolved[0]?.from_dependency).toBe('Grid interconnection capacity')
    expect(unmatched).toHaveLength(1)
  })
})
