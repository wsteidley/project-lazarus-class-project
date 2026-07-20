import {
  CRITICALITY,
  DEPENDENCY,
  FAILURE,
  LIVING_STATUS,
  REGION,
  ROUND,
  SECTOR,
} from '../schemas.js'

// Builds a `col IN ('a','b',...)` CHECK clause from a controlled-vocab array, so
// the DB constraints can't drift from the Zod enums. Nullable columns also allow
// NULL. Single quotes in values are escaped (none currently, but future-proof).
const checkIn = (column: string, values: readonly string[], nullable = true): string => {
  const list = values.map((value) => `'${value.replace(/'/g, "''")}'`).join(', ')
  const nullClause = nullable ? ` OR ${column} IS NULL` : ''
  return `CHECK (${column} IN (${list})${nullClause})`
}

// Full DDL for the relational core. idea_dependencies carries its own uuid so a
// future dependency_assessments table can reference it.
export const createTablesSql = (): string => `
CREATE TABLE companies (
  id             INTEGER PRIMARY KEY,
  uuid           TEXT,
  company_name   TEXT NOT NULL,
  founders       TEXT,
  sector         TEXT ${checkIn('sector', SECTOR)},
  subsector      TEXT,
  location       TEXT ${checkIn('location', REGION)},
  country        TEXT,
  year_founded   INTEGER,
  year_defunct   INTEGER,
  living_status  TEXT ${checkIn('living_status', LIVING_STATUS)},
  has_pivoted    INTEGER,
  idea_summary   TEXT,
  original_trl   INTEGER,
  is_climate     INTEGER,
  source_url     TEXT,
  created_at     TEXT
);

CREATE TABLE funding_rounds (
  id           INTEGER PRIMARY KEY,
  company_id   INTEGER NOT NULL REFERENCES companies(id),
  round_name   TEXT ${checkIn('round_name', ROUND)},
  amount       INTEGER,
  currency     TEXT,
  round_date   TEXT,
  round_year   INTEGER,
  source_url   TEXT
);

CREATE TABLE failure_reasons (
  id           INTEGER PRIMARY KEY,
  company_id   INTEGER NOT NULL REFERENCES companies(id),
  category     TEXT ${checkIn('category', FAILURE)},
  detail       TEXT,
  source_url   TEXT
);

CREATE TABLE idea_dependencies (
  id           INTEGER PRIMARY KEY,
  uuid         TEXT,
  company_id   INTEGER NOT NULL REFERENCES companies(id),
  category     TEXT ${checkIn('category', DEPENDENCY)},
  detail       TEXT,
  criticality  TEXT ${checkIn('criticality', CRITICALITY, false)},
  source_url   TEXT
);
`
