import { describe, expect, it } from 'vitest'
import { apiStages, resolvePlan, STAGES, transitiveDependents } from './pipeline.js'

const ids = (stages: { id: string }[]): string[] => stages.map((stage) => stage.id)

describe('resolvePlan', () => {
  it('returns the whole canonical sequence with no selectors', () => {
    expect(ids(resolvePlan({}))).toEqual(ids(STAGES))
  })

  it('--only returns exactly that stage', () => {
    expect(ids(resolvePlan({ only: 'outcome-pass' }))).toEqual(['outcome-pass'])
  })

  it('--from returns a contiguous tail through build', () => {
    const plan = ids(resolvePlan({ from: 'enrich' }))
    expect(plan[0]).toBe('enrich')
    expect(plan.at(-1)).toBe('build')
    expect(plan).not.toContain('step1')
  })

  it('--through bounds the head of the sequence', () => {
    const plan = ids(resolvePlan({ through: 'step3' }))
    expect(plan[0]).toBe('step0')
    expect(plan.at(-1)).toBe('step3')
    expect(plan).not.toContain('build')
  })

  it('--from + --through bound a slice', () => {
    expect(ids(resolvePlan({ from: 'step2', through: 'derive' }))).toEqual([
      'step2',
      'step3',
      'outcome-pass',
      'derive',
    ])
  })

  it('rejects an unknown stage id', () => {
    expect(() => resolvePlan({ only: 'nope' })).toThrow(/Unknown stage/)
  })

  it('rejects --only combined with --from', () => {
    expect(() => resolvePlan({ only: 'build', from: 'step1' })).toThrow(/cannot be combined/)
  })

  it('rejects an inverted --from/--through range', () => {
    expect(() => resolvePlan({ from: 'build', through: 'step1' })).toThrow(/comes after/)
  })
})

describe('transitiveDependents', () => {
  it('includes stages that transitively depend on the given stage', () => {
    const dependents = transitiveDependents('resolve')
    expect(dependents).toContain('enrich')
    expect(dependents).toContain('build')
    // resolve itself is not its own dependent.
    expect(dependents).not.toContain('resolve')
  })

  it('does not stale a stage that does not depend on the given one', () => {
    // reassess depends on step1c, not on enrich — re-running enrich must not stale it.
    expect(transitiveDependents('enrich')).not.toContain('reassess')
    // ...but re-running step1c does stale reassess.
    expect(transitiveDependents('step1c')).toContain('reassess')
  })

  it('emits dependents in canonical order', () => {
    const dependents = transitiveDependents('step1')
    const order = STAGES.map((stage) => stage.id)
    const positions = dependents.map((id) => order.indexOf(id))
    expect(positions).toEqual([...positions].sort((a, b) => a - b))
  })
})

describe('apiStages', () => {
  it('flags exactly the LLM/search stages in a plan', () => {
    expect(ids(apiStages(resolvePlan({})))).toEqual([
      'step1',
      'step1b',
      'step1c',
      'step2',
      'outcome-pass',
      'reassess',
    ])
  })
})
