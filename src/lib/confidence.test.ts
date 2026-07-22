import { describe, expect, it } from 'vitest'
import { type ConfidenceInputs, computeConfidence, deriveConfidenceLabel } from './confidence.js'

const inputs = (overrides: Partial<ConfidenceInputs> = {}): ConfidenceInputs => ({
  enrichmentLivingStatus: '',
  searchLivingStatus: '',
  domainStatus: 'unknown',
  selfReportedConfidence: 'unknown',
  strongJoinEnrichment: false,
  ...overrides,
})

describe('deriveConfidenceLabel', () => {
  it('maps the score onto the four-level label at the spec boundaries', () => {
    expect(deriveConfidenceLabel(0.9)).toBe('high')
    expect(deriveConfidenceLabel(0.8)).toBe('high')
    expect(deriveConfidenceLabel(0.6)).toBe('medium')
    expect(deriveConfidenceLabel(0.5)).toBe('medium')
    expect(deriveConfidenceLabel(0.35)).toBe('low')
    expect(deriveConfidenceLabel(0.2)).toBe('low')
    expect(deriveConfidenceLabel(0.1)).toBe('unknown')
    expect(deriveConfidenceLabel(0)).toBe('unknown')
  })
})

describe('computeConfidence', () => {
  it('scores high and uncontested when enrichment and search agree', () => {
    const result = computeConfidence(
      inputs({
        enrichmentLivingStatus: 'Defunct',
        searchLivingStatus: 'Defunct',
        domainStatus: 'dead',
        selfReportedConfidence: 'low',
      }),
    )
    expect(result.confidence_score).toBe(0.9)
    expect(result.contested).toBe(false)
    expect(result.contested_note).toBeNull()
  })

  it('scores low and contested when enrichment and search disagree', () => {
    const result = computeConfidence(
      inputs({
        enrichmentLivingStatus: 'Operating',
        searchLivingStatus: 'Defunct',
        domainStatus: 'unknown',
        selfReportedConfidence: 'high',
      }),
    )
    expect(result.confidence_score).toBe(0.35)
    expect(result.contested).toBe(true)
    expect(result.contested_note).toMatch(/disagree/)
  })

  it('uses the domain check as a cross-source signal when enrichment is absent', () => {
    const agree = computeConfidence(
      inputs({ searchLivingStatus: 'Operating', domainStatus: 'alive' }),
    )
    expect(agree.confidence_score).toBe(0.9)

    const disagree = computeConfidence(
      inputs({ searchLivingStatus: 'Operating', domainStatus: 'dead' }),
    )
    expect(disagree.contested).toBe(true)
  })

  it('cross-checks the domain against a strong-join enrichment status when search is skipped', () => {
    const contradicted = computeConfidence(
      inputs({
        enrichmentLivingStatus: 'Operating',
        domainStatus: 'dead',
        strongJoinEnrichment: true,
      }),
    )
    expect(contradicted.confidence_score).toBe(0.35)
    expect(contradicted.contested).toBe(true)
  })

  it('gives a strong-join enrichment value a trusted baseline when nothing contradicts it', () => {
    const result = computeConfidence(
      inputs({
        enrichmentLivingStatus: 'Acquired',
        domainStatus: 'unknown',
        strongJoinEnrichment: true,
      }),
    )
    expect(result.confidence_score).toBe(0.7)
    expect(deriveConfidenceLabel(result.confidence_score)).toBe('medium')
    expect(result.contested).toBe(false)
  })

  it('has no opinion on Acquired/Zombie domain agreement, so it falls back to self-reported', () => {
    const result = computeConfidence(
      inputs({
        searchLivingStatus: 'Acquired',
        domainStatus: 'dead',
        selfReportedConfidence: 'high',
      }),
    )
    expect(result.confidence_score).toBe(0.85)
    expect(result.contested).toBe(false)
  })

  it('falls back to the self-reported label with no cross-source signal available', () => {
    expect(
      computeConfidence(inputs({ searchLivingStatus: 'Operating', selfReportedConfidence: 'high' }))
        .confidence_score,
    ).toBe(0.85)
  })

  it('is the unknown terminal state, with a reason, when nothing is available at all', () => {
    const result = computeConfidence(inputs())
    expect(result.confidence_score).toBe(0)
    expect(result.contested_note).toMatch(/no enrichment, search, or domain signal/)
  })
})
