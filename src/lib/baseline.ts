import type { BASELINE_STATUS } from '../schemas.js'

// Classifies whether a series' baseline can express a 0->1 progress range against its bar.
// This is the same logic the `progress` view encodes in SQL, lifted out so build-db can report
// baseline problems at load time (report-not-drop) rather than silently emitting null progress.

export type BaselineStatus = (typeof BASELINE_STATUS)[number]

// One series' inputs: the bar, its direction, and the baseline actually in force (the declared
// attempt-era value, or the earliest observation when none is declared).
export type BaselineInput = {
  direction: string
  threshold: number
  baseline: number
}

// `ok` — progress is computable. `baseline_equals_threshold` — B == T, the formula would
// divide by zero. `baseline_past_threshold` — the baseline already satisfies the bar, so the
// series starts already-viable and can't express 0->1 (the declared baseline postdates
// viability). Mirrors the CASE in the progress view exactly.
export const classifyBaseline = ({
  direction,
  threshold,
  baseline,
}: BaselineInput): BaselineStatus => {
  if (baseline === threshold) {
    return 'baseline_equals_threshold'
  }
  if (
    (direction === 'below_is_better' && baseline < threshold) ||
    (direction === 'above_is_better' && baseline > threshold)
  ) {
    return 'baseline_past_threshold'
  }
  return 'ok'
}
