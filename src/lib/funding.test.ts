import { describe, expect, it } from 'vitest'
import {
  aggregateAmountByYear,
  fixRoundName,
  standardizeDate,
  standardizeFundingData,
} from './funding.js'

describe('standardizeDate', () => {
  const cases: [string, string | null][] = [
    ['3/2020', '03/20'],
    ['11/19', '11/19'],
    ['2021', '01/21'],
    ['2020-07', '07/20'],
    ['20-7', '07/20'],
    ['03/15/2019', '03/19'],
    ['2018-05-09', '05/18'],
    ['March 3, 2020', '03/20'],
    ['June 12 2022', '06/22'],
    ['not a date', null],
  ]

  it.each(cases)('parses %s -> %s', (input, expected) => {
    expect(standardizeDate(input)).toBe(expected)
  })

  it('returns null for null/undefined', () => {
    expect(standardizeDate(null)).toBeNull()
    expect(standardizeDate(undefined)).toBeNull()
  })
})

describe('fixRoundName', () => {
  it('maps loose names onto the controlled vocabulary', () => {
    expect(fixRoundName('a seed round')).toBe('Seed')
    expect(fixRoundName('Series A extension')).toBe('Series A')
  })

  it('falls back to Unknown', () => {
    expect(fixRoundName('mystery money')).toBe('Unknown')
    expect(fixRoundName(null)).toBe('Unknown')
  })
})

describe('standardizeFundingData', () => {
  it('de-dupes rounds per name keeping the larger amount, standardizes dates, drops schema junk', () => {
    const input = [
      JSON.stringify([
        { round_name: 'Seed', amount: 1_000_000, date: '2019-05', currency_symbol: '$' },
        { round_name: 'seed round', amount: 1_500_000, date: '06/2019' },
        { round_name: 'Series A', amount: 5_000_000, date: 'March 3, 2021' },
        { type: 'object', properties: {} },
      ]),
    ]

    expect(standardizeFundingData(input)).toEqual([
      { round_name: 'Seed', amount: 1_500_000, date: '05/19', currency_symbol: '$' },
      { round_name: 'Series A', amount: 5_000_000, date: '03/21' },
    ])
  })
})

describe('aggregateAmountByYear', () => {
  it('rolls up amounts per four-digit year', () => {
    const rounds = [
      { round_name: 'Seed', amount: 1_500_000, date: '05/19' },
      { round_name: 'Series A', amount: 5_000_000, date: '03/21' },
      { round_name: 'Series B', amount: 500_000, date: '11/21' },
    ]
    expect(aggregateAmountByYear(rounds)).toEqual({
      '2019': 1_500_000,
      '2021': 5_500_000,
    })
  })
})
