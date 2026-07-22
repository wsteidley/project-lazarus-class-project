import {
  ASSESSMENT_STATUS,
  CHALLENGE,
  CHALLENGE_OUTCOME,
  CONFIDENCE,
  CRITICALITY,
  DEPENDENCY,
  EXIT_TYPE,
  LIVING_STATUS,
  OUTCOME_TYPE,
  REGION,
  ROUND,
  URL_TYPE,
} from '../schemas.js'
import { SOURCE_TYPE } from './raw-documents.js'

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
// sectors (broad verticals) via company_sectors. Dependencies are canonical and
// shared (see below); challenges carry provenance and confidence.
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
  -- Identity keys live in company_urls (typed, one row per URL) rather than as
  -- columns here, so a new source is a new row instead of a schema change.
  -- Provenance of a merge: canonical_uuid is this row's surviving identity, and
  -- merged_from lists the uuids it absorbed (empty when nothing merged).
  canonical_uuid     TEXT,
  merged_from        TEXT,
  location           TEXT ${checkIn('location', REGION)},
  country            TEXT,
  year_founded       INTEGER,
  year_defunct       INTEGER,
  -- Provenance for the fields Phase 2a/2b both write, so precedence can tell a
  -- strong-join enrichment value apart from a name-only seed or a fresh search
  -- finding, e.g. "enrichment:crunchbase:website", "enrichment:startup-failures:name",
  -- "search", "heuristic". Blank means neither enrichment nor search ever set it.
  living_status_source TEXT,
  year_founded_source   TEXT,
  living_status      TEXT ${checkIn('living_status', LIVING_STATUS)},
  has_pivoted        INTEGER,
  idea_summary       TEXT,
  exit_type          TEXT ${checkIn('exit_type', EXIT_TYPE)},
  exit_amount        INTEGER,
  exit_date          TEXT,
  exit_notes         TEXT,
  outcome_summary    TEXT,
  outcome_source_url TEXT,
  -- Evidence strength for the outcome pass's living_status/exit verdict. The score is
  -- authoritative; outcome_confidence is the four-level label *derived from it*. The
  -- LLM's own self-reported label is kept separate and never blended into the label.
  outcome_confidence               TEXT ${checkIn('outcome_confidence', CONFIDENCE)},
  outcome_confidence_score         REAL,
  outcome_confidence_self_reported TEXT ${checkIn('outcome_confidence_self_reported', CONFIDENCE)},
  outcome_contested                INTEGER,
  outcome_contested_note           TEXT,
  outcome_type       TEXT ${checkIn('outcome_type', OUTCOME_TYPE)},
  outcome_rationale  TEXT,
  original_trl       INTEGER,
  is_climate         INTEGER,
  source_url         TEXT,
  created_at         TEXT
);

-- Every URL known for a company, tagged by what it points at. Replaces per-source
-- identity columns: adding LinkedIn, a Wayback capture, or a dead-site link is a new
-- row, not a migration. normalized_value holds the comparable form the resolve step
-- keys on (a bare domain for websites), so the merge key is visible and queryable.
CREATE TABLE company_urls (
  id               INTEGER PRIMARY KEY,
  company_id       INTEGER NOT NULL REFERENCES companies(id),
  url_type         TEXT ${checkIn('url_type', URL_TYPE)},
  url              TEXT NOT NULL,
  normalized_value TEXT,
  source_url       TEXT,
  UNIQUE (company_id, url_type, url)
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
  id               INTEGER PRIMARY KEY,
  company_id       INTEGER NOT NULL REFERENCES companies(id),
  category         TEXT ${checkIn('category', CHALLENGE)},
  outcome          TEXT ${checkIn('outcome', CHALLENGE_OUTCOME)},
  detail           TEXT,
  confidence       TEXT ${checkIn('confidence', CONFIDENCE)},
  confidence_score REAL,
  contested        INTEGER,
  contested_note   TEXT,
  source_url       TEXT
);

-- Canonical dependency dimension: one row per real-world thing an idea needed
-- ("battery pack price"), shared across every company that needed it. Companies
-- attach via company_dependencies, and assessments attach here — so the "now"
-- verdict is established once rather than re-derived per company.
CREATE TABLE dependencies (
  id               INTEGER PRIMARY KEY,
  uuid             TEXT,
  name             TEXT NOT NULL,
  category         TEXT ${checkIn('category', DEPENDENCY)},
  description      TEXT,
  -- The value at which this dependency stops blocking. Populated in Tier 3; the
  -- columns exist now so became_viable_date has somewhere to land.
  threshold_metric TEXT,
  threshold_value  REAL,
  threshold_unit   TEXT
);

CREATE TABLE company_dependencies (
  company_id    INTEGER NOT NULL REFERENCES companies(id),
  dependency_id INTEGER NOT NULL REFERENCES dependencies(id),
  criticality   TEXT ${checkIn('criticality', CRITICALITY, false)},
  detail        TEXT,
  source_url    TEXT,
  PRIMARY KEY (company_id, dependency_id)
);

-- The "now" axis: one row per (dependency x assessed_on), each carrying its own
-- evidence (source_url + snippet + confidence) so an automated verdict is auditable
-- and can later be swapped for a grounded data feed without a rewrite.
CREATE TABLE dependency_assessments (
  id               INTEGER PRIMARY KEY,
  uuid             TEXT,
  dependency_id    INTEGER NOT NULL REFERENCES dependencies(id),
  status           TEXT ${checkIn('status', ASSESSMENT_STATUS)},
  detail           TEXT,
  metric_name      TEXT,
  metric_value     REAL,
  metric_unit      TEXT,
  assessed_on      TEXT,
  source_url       TEXT,
  snippet          TEXT,
  confidence       TEXT ${checkIn('confidence', CONFIDENCE)},
  confidence_score REAL,
  contested        INTEGER,
  contested_note   TEXT
);

-- Content-addressed cache of every fetched document, loaded from the on-disk cache
-- at build time. source_type drives the freshness policy: discovery text is
-- immutable, outcome/reassessment results expire.
CREATE TABLE raw_documents (
  id          INTEGER PRIMARY KEY,
  url         TEXT,
  url_hash    TEXT UNIQUE,
  fetched_at  TEXT,
  text        TEXT,
  source_type TEXT ${checkIn('source_type', SOURCE_TYPE)}
);
`
