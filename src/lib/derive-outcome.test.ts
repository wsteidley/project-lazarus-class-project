import { describe, expect, it } from 'vitest'
import { type DerivationInput, deriveOutcome } from './derive-outcome.js'

const base: DerivationInput = {
  living_status: 'Operating',
  exit_type: null,
  exit_amount: null,
  total_raised: 0,
  year_founded: 2015,
  last_round_year: null,
  last_round_name: null,
  build_year: 2026,
}

describe('deriveOutcome', () => {
  it('Defunct with no exit -> Failed', () => {
    expect(deriveOutcome({ ...base, living_status: 'Defunct' }).outcome_type).toBe('Failed')
  })

  it('Defunct acquired at/above raised -> Successful Exit', () => {
    const out = deriveOutcome({
      ...base,
      living_status: 'Defunct',
      exit_type: 'acquisition',
      exit_amount: 60_000_000,
      total_raised: 50_000_000,
    })
    expect(out.outcome_type).toBe('Successful Exit')
  })

  it('Defunct acquired below raised -> Soft Landing', () => {
    const out = deriveOutcome({
      ...base,
      living_status: 'Defunct',
      exit_type: 'acquisition',
      exit_amount: 5_000_000,
      total_raised: 50_000_000,
    })
    expect(out.outcome_type).toBe('Soft Landing')
  })

  it('Acquired for undisclosed amount -> Soft Landing (conservative)', () => {
    const out = deriveOutcome({ ...base, living_status: 'Acquired', total_raised: 20_000_000 })
    expect(out.outcome_type).toBe('Soft Landing')
    expect(out.outcome_rationale).toMatch(/conservative/)
  })

  it('Operating with a recent growth-stage raise -> Breakout Success', () => {
    const out = deriveOutcome({
      ...base,
      total_raised: 450_000_000,
      last_round_year: 2025,
      last_round_name: 'Series D+',
    })
    expect(out.outcome_type).toBe('Breakout Success')
  })

  it('Operating, young, little signal -> Too Early to Tell', () => {
    const out = deriveOutcome({ ...base, year_founded: 2024, total_raised: 500_000 })
    expect(out.outcome_type).toBe('Too Early to Tell')
  })

  it('Operating but no raise in 3+ years -> Struggling / Zombie', () => {
    const out = deriveOutcome({
      ...base,
      total_raised: 8_000_000,
      last_round_year: 2020,
      last_round_name: 'Series A',
    })
    expect(out.outcome_type).toBe('Struggling / Zombie')
  })

  it('Operating, funded, recent, not breakout -> Solid Success', () => {
    const out = deriveOutcome({
      ...base,
      total_raised: 30_000_000,
      last_round_year: 2025,
      last_round_name: 'Series B',
    })
    expect(out.outcome_type).toBe('Solid Success')
  })

  it('Zombie -> Struggling / Zombie; unknown status -> Unknown', () => {
    expect(deriveOutcome({ ...base, living_status: 'Zombie' }).outcome_type).toBe(
      'Struggling / Zombie',
    )
    expect(deriveOutcome({ ...base, living_status: '' }).outcome_type).toBe('Unknown')
  })
})
