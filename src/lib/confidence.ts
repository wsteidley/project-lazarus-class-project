import type { CONFIDENCE } from '../schemas.js'
import type { DomainStatus } from './wayback.js'

// #4 confidence machinery: computed confidence_score is authoritative, cheapest
// signal first. Cross-source agreement (near-free, already-fetched signals) is the
// primary tier; self-reported confidence (the LLM's own label) is the fallback when
// there's nothing independent to cross-check against.
const SELF_REPORTED_SCORE: Record<string, number> = {
  high: 0.85,
  medium: 0.6,
  low: 0.3,
  unknown: 0.1,
}

// A strong-join enrichment status with nothing to contradict it is the reason we skip
// re-searching it — trustworthy on its own, but below the 0.8 "corroborated" bar.
const STRONG_JOIN_BASELINE = 0.7

// The four-level label the spec derives *from the computed score* (never blended with
// the LLM's self-reported label, which is stored separately). Thresholds per spec #4:
// ≥0.8 high, 0.5–0.8 medium, 0.2–0.5 low, else unknown.
export const deriveConfidenceLabel = (score: number): (typeof CONFIDENCE)[number] => {
  if (score >= 0.8) {
    return 'high'
  }
  if (score >= 0.5) {
    return 'medium'
  }
  if (score >= 0.2) {
    return 'low'
  }
  return 'unknown'
}

// Whether a dead/alive domain signal is consistent with a living_status label. Only
// Operating/Defunct have an unambiguous domain expectation — Acquired/Zombie/Unknown
// return null (no opinion) rather than force a guess (a site can survive or vanish
// after an acquisition either way).
const domainAgreesWithStatus = (
  domainStatus: DomainStatus,
  livingStatus: string,
): boolean | null => {
  if (domainStatus === 'unknown' || !livingStatus) {
    return null
  }
  if (livingStatus === 'Operating') {
    return domainStatus === 'alive'
  }
  if (livingStatus === 'Defunct') {
    return domainStatus === 'dead'
  }
  return null
}

export type ConfidenceInputs = {
  // living_status Phase 2a supplied before this pass ran, '' if none.
  enrichmentLivingStatus: string
  // living_status the search pass concluded, '' if search didn't run/produce one.
  searchLivingStatus: string
  domainStatus: DomainStatus
  // The LLM's own confidence label on the search verdict (CONFIDENCE enum).
  selfReportedConfidence: string
  // Whether the enrichment status came from a strong (website/crunchbase) join.
  strongJoinEnrichment: boolean
}

export type ConfidenceResult = {
  confidence_score: number
  contested: boolean
  contested_note: string | null
}

// unknown is a defined terminal state, always returned with a reason — never a blank.
export const computeConfidence = (inputs: ConfidenceInputs): ConfidenceResult => {
  const {
    enrichmentLivingStatus,
    searchLivingStatus,
    domainStatus,
    selfReportedConfidence,
    strongJoinEnrichment,
  } = inputs

  // The status we're actually going with: a fresh search verdict when we have one,
  // otherwise whatever enrichment supplied. Signals are cross-checked against it.
  const effectiveStatus = searchLivingStatus || enrichmentLivingStatus

  const signals: boolean[] = []
  if (enrichmentLivingStatus && searchLivingStatus) {
    signals.push(enrichmentLivingStatus === searchLivingStatus)
  }
  const domainVerdict = domainAgreesWithStatus(domainStatus, effectiveStatus)
  if (domainVerdict !== null) {
    signals.push(domainVerdict)
  }

  if (signals.length > 0) {
    const allAgree = signals.every(Boolean)
    return {
      confidence_score: allAgree ? 0.9 : 0.35,
      contested: !allAgree,
      contested_note: allAgree
        ? null
        : 'cross-source signals disagree on living_status (enrichment/search/domain check)',
    }
  }

  if (!effectiveStatus) {
    return {
      confidence_score: 0,
      contested: false,
      contested_note: 'no enrichment, search, or domain signal available',
    }
  }

  // A strong-join enrichment value we deliberately did not re-search stands on the
  // trust of its join — no independent signal was available to raise or lower it.
  if (strongJoinEnrichment && !searchLivingStatus) {
    return { confidence_score: STRONG_JOIN_BASELINE, contested: false, contested_note: null }
  }

  // No independent signal — fall back to the search's self-reported label.
  return {
    confidence_score: SELF_REPORTED_SCORE[selfReportedConfidence] ?? 0.1,
    contested: false,
    contested_note: null,
  }
}
