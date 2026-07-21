import {
  CHALLENGE,
  CHALLENGE_OUTCOME,
  CRITICALITY,
  DEPENDENCY,
  EXIT_TYPE,
  LIVING_STATUS,
  OUTCOME_TYPE,
  REGION,
  ROUND,
} from '../schemas.js'

// Builds a `col IN ('a','b',...)` CHECK clause from a controlled-vocab array, so
// the DB constraints can't drift from the Zod enums. Nullable columns also allow
// NULL. Single quotes in values are escaped (none currently, but future-proof).
const checkIn = (column: string, values: readonly string[], nullable = true): string => {
  const list = values.map((value) => `'${value.replace(/'/g, "''")}'`).join(', ')
  const nullClause = nullable ? ` OR ${column} IS NULL` : ''
  return `CHECK (${column} IN (${list})${nullClause})`
}

// Full DDL for the relational schema. sectors are reference rows (seeded from the
// SECTOR vocab); a company sits in one idea_space (its specific pursuit) and 1+
// sectors (broad verticals) via company_sectors. idea_dependencies and challenges
// carry provenance so future assessments can reference them.
export const createTablesSql = (): string => `
CREATE TABLE sectors (
  id    INTEGER PRIMARY KEY,
  name  TEXT NOT NULL UNIQUE
);

CREATE TABLE idea_spaces (
  id          INTEGER PRIMARY KEY,
  name        TEXT NOT NULL,
  sector_id   INTEGER REFERENCES sectors(id),
  description TEXT
);

CREATE TABLE companies (
  id                 INTEGER PRIMARY KEY,
  uuid               TEXT,
  company_name       TEXT NOT NULL,
  idea_space_id      INTEGER REFERENCES idea_spaces(id),
  founders           TEXT,
  location           TEXT ${checkIn('location', REGION)},
  country            TEXT,
  year_founded       INTEGER,
  year_defunct       INTEGER,
  living_status      TEXT ${checkIn('living_status', LIVING_STATUS)},
  has_pivoted        INTEGER,
  idea_summary       TEXT,
  exit_type          TEXT ${checkIn('exit_type', EXIT_TYPE)},
  exit_amount        INTEGER,
  exit_date          TEXT,
  exit_notes         TEXT,
  outcome_summary    TEXT,
  outcome_source_url TEXT,
  outcome_type       TEXT ${checkIn('outcome_type', OUTCOME_TYPE)},
  outcome_rationale  TEXT,
  original_trl       INTEGER,
  is_climate         INTEGER,
  source_url         TEXT,
  created_at         TEXT
);

CREATE TABLE company_sectors (
  company_id  INTEGER NOT NULL REFERENCES companies(id),
  sector_id   INTEGER NOT NULL REFERENCES sectors(id),
  is_primary  INTEGER,
  PRIMARY KEY (company_id, sector_id)
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

CREATE TABLE challenges (
  id           INTEGER PRIMARY KEY,
  company_id   INTEGER NOT NULL REFERENCES companies(id),
  category     TEXT ${checkIn('category', CHALLENGE)},
  outcome      TEXT ${checkIn('outcome', CHALLENGE_OUTCOME)},
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
