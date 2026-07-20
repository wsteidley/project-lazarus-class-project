import { describe, expect, it } from 'vitest'
import {
  dedupeFundingRows,
  type FundingCsvRow,
  fixRoundName,
  standardizeDate,
  yearFromStandardizedDate,
} from './funding.js'

describe('standardizeDate', () => {
  const cases: [string, string | null][] = [
    ['2020-03', '2020-03'],
    ['2020/3', '2020-03'],
    ['2018-05-09', '2018-05'],
    ['3/2020', '2020-03'],
    ['03/15/2019', '2019-03'],
    ['2021', '2021-01'],
    ['11/19', '2019-11'],
    ['March 3, 2020', '2020-03'],
    ['not a date', null],
    ['', null],
  ]

  it.each(cases)('parses %s -> %s', (input, expected) => {
    expect(standardizeDate(input)).toBe(expected)
  })

  it('preserves the 4-digit year (no century collapse)', () => {
    expect(standardizeDate('1998-07')).toBe('1998-07')
    expect(standardizeDate(null)).toBeNull()
  })
})

describe('yearFromStandardizedDate', () => {
  it('derives a 4-digit integer year', () => {
    expect(yearFromStandardizedDate('2020-03')).toBe(2020)
    expect(yearFromStandardizedDate(null)).toBe('')
  })
})

describe('fixRoundName', () => {
  it('maps loose names onto the ROUND vocabulary', () => {
    expect(fixRoundName('a seed round')).toBe('Seed')
    expect(fixRoundName('Series A extension')).toBe('Series A')
    expect(fixRoundName('government grant award')).toBe('Grant')
  })

  it('falls back to Unknown', () => {
    expect(fixRoundName('mystery money')).toBe('Unknown')
    expect(fixRoundName(null)).toBe('Unknown')
  })
})

describe('dedupeFundingRows', () => {
  it('de-dupes per company + round, keeps max amount and a standardized date', () => {
    const rows: FundingCsvRow[] = [
      {
        company_uuid: 'c1',
        round_name: 'Seed',
        amount: '1000000',
        currency: 'USD',
        round_date: '2019-05',
        round_year: '2019',
        source_url: 'http://x/1',
      },
      {
        company_uuid: 'c1',
        round_name: 'Seed',
        amount: '1500000',
        currency: '',
        round_date: '06/2019',
        round_year: '',
        source_url: 'http://x/1',
      },
      {
        company_uuid: 'c1',
        round_name: 'Series A',
        amount: '5000000',
        currency: 'USD',
        round_date: 'March 3, 2021',
        round_year: '',
        source_url: 'http://x/1',
      },
    ]

    expect(dedupeFundingRows(rows)).toEqual([
      {
        company_uuid: 'c1',
        round_name: 'Seed',
        amount: 1_500_000,
        currency: 'USD',
        round_date: '2019-05',
        round_year: 2019,
        source_url: 'http://x/1',
      },
      {
        company_uuid: 'c1',
        round_name: 'Series A',
        amount: 5_000_000,
        currency: 'USD',
        round_date: '2021-03',
        round_year: 2021,
        source_url: 'http://x/1',
      },
    ])
  })
})
