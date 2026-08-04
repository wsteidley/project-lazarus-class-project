// A company_dependencies row, before FK resolution in build-db. An empty
// dependency_name means the free-text dependency matched nothing canonical — kept
// deliberately, so unresolved rows can be counted and curated rather than vanishing.
// dependency_name_raw carries what the model actually proposed either way: for a resolved
// row it is the pre-canonicalisation spelling (so a bad match stays auditable), and for an
// unresolved one it is the only name the blocker has.
export type CompanyDependencyRow = {
  company_uuid: string
  dependency_name: string
  dependency_name_raw: string
  criticality: string
  detail: string
  source_url: string
}

// Normalizes a name for comparison only: case, surrounding space, and internal
// runs of whitespace. Never used as the stored value.
const comparable = (name: string): string => name.trim().toLowerCase().replace(/\s+/g, ' ')

// Maps a model-proposed name onto the curated canonical list, tolerating case and
// whitespace drift. Returns the canonical spelling (not the proposal) so the stored
// value always matches the seed exactly, or null when nothing fits.
export const matchCanonical = (
  proposed: string | null | undefined,
  canonicalNames: readonly string[],
): string | null => {
  if (!proposed) {
    return null
  }
  const target = comparable(proposed)
  if (!target) {
    return null
  }
  return canonicalNames.find((name) => comparable(name) === target) ?? null
}

// company_dependencies is keyed on (company_id, dependency_id), so two free-text
// dependencies collapsing onto the same canonical row must be merged rather than
// inserted twice. 'was_blocking' wins over 'contributing' — if the dependency
// blocked the company under any description, it blocked them — and the distinct
// details are joined so no extracted nuance is lost.
//
// Unresolved rows merge too, on their raw name. They used to be passed through one-by-one
// because they had no canonical identity — but now that they are retained rather than
// dropped, two extractions of the same unmatched blocker are the same blocker and would
// otherwise land as duplicate rows.
export const dedupeCompanyDependencies = (rows: CompanyDependencyRow[]): CompanyDependencyRow[] => {
  const byKey = new Map<string, CompanyDependencyRow>()

  for (const row of rows) {
    const identity = row.dependency_name || `raw:${comparable(row.dependency_name_raw)}`
    const key = `${row.company_uuid}::${identity}`
    const existing = byKey.get(key)
    if (!existing) {
      byKey.set(key, { ...row })
      continue
    }
    if (row.criticality === 'was_blocking') {
      existing.criticality = 'was_blocking'
    }
    const details = new Set([existing.detail, row.detail].filter((detail) => detail?.trim()))
    existing.detail = [...details].join('; ')
  }

  return [...byKey.values()]
}
