import { describe, expect, it } from 'vitest'
import {
  crunchbaseKey,
  crunchbaseStatus,
  failureFlagsToChallenges,
  parseRaisedAmount,
  parseYearsRange,
} from './sources.js'

describe('crunchbaseKey', () => {
  it('reduces a permalink to its slug', () => {
    expect(crunchbaseKey('/organization/helios-energy')).toBe('helios-energy')
  })

  it('accepts a full Crunchbase URL', () => {
    expect(crunchbaseKey('https://www.crunchbase.com/organization/Helios-Energy/')).toBe(
      'helios-energy',
    )
  })

  it('is empty for a missing permalink', () => {
    expect(crunchbaseKey('')).toBe('')
  })
})

describe('parseRaisedAmount', () => {
  it('scales K / M / B suffixes', () => {
    expect(parseRaisedAmount('$12M')).toBe('12000000')
    expect(parseRaisedAmount('$1.5B')).toBe('1500000000')
    expect(parseRaisedAmount('$500K')).toBe('500000')
  })

  it('handles a plain dollar figure with commas', () => {
    expect(parseRaisedAmount('$2,000,000')).toBe('2000000')
  })

  it('returns empty for undisclosed/unknown so it stays distinct from zero', () => {
    expect(parseRaisedAmount('Unknown')).toBe('')
    expect(parseRaisedAmount('')).toBe('')
    expect(parseRaisedAmount(null)).toBe('')
  })
})

describe('parseYearsRange', () => {
  it('splits a two-year range', () => {
    expect(parseYearsRange('2015-2019')).toEqual({ year_founded: '2015', year_defunct: '2019' })
  })

  it('reads the roster index shape "3 (2010-2013)"', () => {
    expect(parseYearsRange('3 (2010-2013)')).toEqual({ year_founded: '2010', year_defunct: '2013' })
  })

  it('leaves defunct empty for an open end', () => {
    expect(parseYearsRange('2015-Present')).toEqual({ year_founded: '2015', year_defunct: '' })
  })

  it('is empty for no years', () => {
    expect(parseYearsRange('')).toEqual({ year_founded: '', year_defunct: '' })
  })
})

describe('crunchbaseStatus', () => {
  it('maps operating', () => {
    expect(crunchbaseStatus('operating')).toEqual({ living_status: 'Operating', exit_type: '' })
  })

  it('records acquisition and ipo as exits while staying operating for ipo', () => {
    expect(crunchbaseStatus('acquired')).toEqual({
      living_status: 'Acquired',
      exit_type: 'acquisition',
    })
    expect(crunchbaseStatus('ipo')).toEqual({ living_status: 'Operating', exit_type: 'ipo' })
  })

  it('maps closed to Defunct + shutdown', () => {
    expect(crunchbaseStatus('closed')).toEqual({ living_status: 'Defunct', exit_type: 'shutdown' })
  })

  it('is empty for an unknown status', () => {
    expect(crunchbaseStatus('-')).toEqual({ living_status: '', exit_type: '' })
  })
})

describe('failureFlagsToChallenges', () => {
  it('maps set flags onto the CHALLENGE vocabulary', () => {
    const categories = failureFlagsToChallenges({
      'No Budget': '1',
      'Poor Market Fit': '1',
      Competition: '1',
      Giants: '0',
    })
    expect(categories).toContain('Ran Out of Capital')
    expect(categories).toContain('Poor Product-Market Fit')
    expect(categories).toContain('Outcompeted')
  })

  it('de-duplicates when several flags map to one category', () => {
    // Giants and Competition both mean Outcompeted.
    expect(failureFlagsToChallenges({ Giants: '1', Competition: '1' })).toEqual(['Outcompeted'])
  })

  it('ignores unset (0/blank) flags', () => {
    expect(failureFlagsToChallenges({ 'No Budget': '0', Competition: '' })).toEqual([])
  })
})
