import { describe, expect, it } from 'vitest'
import { classifyBaseline } from './baseline.js'

describe('classifyBaseline', () => {
  it('is ok when the baseline sits on the not-yet-viable side of the bar', () => {
    // below_is_better: baseline above the bar (worse), room to fall to it.
    expect(classifyBaseline({ direction: 'below_is_better', threshold: 100, baseline: 1100 })).toBe(
      'ok',
    )
    // above_is_better: baseline below the bar (worse), room to rise to it.
    expect(classifyBaseline({ direction: 'above_is_better', threshold: 50, baseline: 8 })).toBe(
      'ok',
    )
  })

  it('flags baseline == threshold (would divide by zero)', () => {
    expect(classifyBaseline({ direction: 'below_is_better', threshold: 2, baseline: 2 })).toBe(
      'baseline_equals_threshold',
    )
  })

  it('flags a baseline already past the bar (would invert)', () => {
    // 97 already below the 100 bar — battery's real problem.
    expect(classifyBaseline({ direction: 'below_is_better', threshold: 100, baseline: 97 })).toBe(
      'baseline_past_threshold',
    )
    // above_is_better mirror: baseline already above the bar.
    expect(classifyBaseline({ direction: 'above_is_better', threshold: 50, baseline: 60 })).toBe(
      'baseline_past_threshold',
    )
  })
})
