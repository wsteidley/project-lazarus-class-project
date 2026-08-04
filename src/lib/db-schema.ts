import {
  ASSESSMENT_STATUS,
  CHALLENGE,
  CHALLENGE_OUTCOME,
  CONFIDENCE,
  CRITICALITY,
  DEPENDENCY,
  DEPENDENCY_RELATION,
  EXIT_TYPE,
  LIVING_STATUS,
  OBSERVATION_METHOD,
  OUTCOME_TYPE,
  PROJECTION_METHOD,
  REGION,
  ROUND,
  THRESHOLD_DIRECTION,
  THRESHOLD_KIND,
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
-- Identity only. The crossing bars moved to dependency_thresholds (v2): one dependency now
-- needs several bars by slice (carbon global vs. Norway), which one-row-per-dependency
-- columns can't hold. threshold_kind stays here — it's a property of the dependency, not of
-- a per-scope bar.
CREATE TABLE dependencies (
  id               INTEGER PRIMARY KEY,
  uuid             TEXT,
  name             TEXT NOT NULL,
  category         TEXT ${checkIn('category', DEPENDENCY)},
  description      TEXT,
  threshold_kind   TEXT ${checkIn('threshold_kind', THRESHOLD_KIND)}
);

-- One crossing bar per (dependency, metric, scope) — the same key observations and the
-- progress derivation use. A qualitative dependency has no row here; a quantitative_tbd one
-- may have a row with a null threshold_value. When contested, the alt bar lets the
-- derivation compute progress against both and flag the crossing rather than pick a side.
CREATE TABLE dependency_thresholds (
  dependency_id            INTEGER NOT NULL REFERENCES dependencies(id),
  metric                   TEXT NOT NULL,
  scope                    TEXT NOT NULL,
  threshold_value          REAL,
  threshold_unit           TEXT,
  threshold_direction      TEXT ${checkIn('threshold_direction', THRESHOLD_DIRECTION)},
  threshold_source_url     TEXT,
  threshold_as_of          TEXT,
  threshold_note           TEXT,
  threshold_contested      INTEGER DEFAULT 0,
  threshold_alt_value      REAL,
  threshold_alt_source_url TEXT,
  threshold_contested_note TEXT,
  -- Marks a dependency that does not cross its bar on economics alone but does under a
  -- support regime (subsidy, mandate, carbon price) — hydrogen, DAC, electrolyzers. It
  -- drives the 'conditional' trajectory state, which for a failed company means "revivable,
  -- but only if the policy regime it needed now exists" — a different answer from "the cost
  -- curve fixed it". Deliberately a flag and not a second bar: threshold_alt already means
  -- "contested" on Direct air capture, and one column cannot mean both on the flagship row.
  policy_dependent         INTEGER DEFAULT 0,
  -- Declared baseline: the attempt-era value, where the metric stood when the companies
  -- were dying — so progress reads as "how far the world moved since the failures," not
  -- "the oldest number we happen to have." Null falls back to the earliest observation.
  baseline_value           REAL,
  baseline_as_of           TEXT,
  baseline_note            TEXT,
  PRIMARY KEY (dependency_id, metric, scope)
);

-- Causal edges between dependencies (interconnection drives solar economics, …), so the
-- chains the research surfaced are traceable rather than hidden.
CREATE TABLE dependency_links (
  id                 INTEGER PRIMARY KEY,
  from_dependency_id INTEGER NOT NULL REFERENCES dependencies(id),
  to_dependency_id   INTEGER NOT NULL REFERENCES dependencies(id),
  relation           TEXT ${checkIn('relation', DEPENDENCY_RELATION, false)},
  note               TEXT
);

-- Tunable knobs for the trajectory view, kept as data (one row) so retuning the slope
-- window or the plateau cutoff is an UPDATE, not a query edit.
CREATE TABLE trajectory_config (
  window_n                INTEGER,
  plateau_slope_threshold REAL
);

-- The one derived layer that is genuine computation, not a view: extrapolated future points
-- for series not yet crossed. Every row is flagged projected=1 with its method and fit
-- window, so a forecast never masquerades as an observation; is_crossing marks the projected
-- crossing point.
CREATE TABLE metric_projections (
  id            INTEGER PRIMARY KEY,
  dependency_id INTEGER NOT NULL REFERENCES dependencies(id),
  metric        TEXT,
  scope         TEXT,
  segment       TEXT,
  as_of         TEXT,
  progress      REAL,
  projected     INTEGER DEFAULT 1,
  method        TEXT ${checkIn('method', PROJECTION_METHOD)},
  fit_window_n  INTEGER,
  confidence    REAL,
  is_crossing   INTEGER,
  note          TEXT
);

-- Learning-rate fits, kept apart from metric_projections because a fit and a forecast are
-- different claims. metric_projections answers "when will this cross?"; this answers "how fast
-- does it get cheaper per doubling of deployment?" -- which stays true, and stays interesting,
-- after a series has already crossed. Onshore wind (crossed 2019) and battery (crossed 2025)
-- produce a fit here and no projection at all; storing the rate only inside a projection note
-- would lose it for exactly those series.
--
-- learning_rate is the fraction shaved per doubling (solar ~0.28); b is the raw log-log slope
-- it derives from; r2 and n_pairs are the evidence for believing it.
CREATE TABLE wright_fits (
  dependency_id INTEGER NOT NULL REFERENCES dependencies(id),
  metric        TEXT NOT NULL,
  scope         TEXT NOT NULL,
  segment       TEXT NOT NULL,
  learning_rate REAL,
  b             REAL,
  r2            REAL,
  n_pairs       INTEGER,
  first_as_of   TEXT,
  last_as_of    TEXT,
  PRIMARY KEY (dependency_id, metric, scope, segment)
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

-- The grounded trajectory: one row per (dependency x as_of x scope), each a dated,
-- cited fact about where a metric actually stood. Kept separate from
-- dependency_assessments so hand-curated/feed numbers never blend with LLM verdicts.
-- Append-only (no UNIQUE): a correction is a new row with a later as_of, never an
-- overwrite — overwriting would destroy the trajectory the crossing test depends on.
-- basis is not decoration: unit alone does not identify a series. USD/W in nominal
-- dollars and USD/W in constant 2024 dollars are different measurements, and comparing one
-- against a bar declared on the other silently answers the viability question wrong. One
-- basis per series (v3 Fix 2).
-- segment names the SLICE the value covers: 'all' is the rolled-up total, and the parts
-- (bev/stationary, onshore/offshore, on_grid/off_grid) sit alongside it. Never sum a total
-- together with its own parts. It is distinct from basis: basis says how a value was
-- measured (constant vs nominal dollars, DC vs AC), segment says which subset was measured.
-- Without it a slice has to be smuggled into the metric name -- which is what
-- "battery pack price (BEV)" was doing, silently preventing the bar declared on
-- "battery pack price" from ever joining its own observations.
CREATE TABLE metric_observations (
  id            INTEGER PRIMARY KEY,
  dependency_id INTEGER NOT NULL REFERENCES dependencies(id),
  metric        TEXT,
  value         REAL,
  unit          TEXT,
  basis         TEXT,
  -- NOT NULL DEFAULT 'all' is load-bearing, not tidiness: segment is an equality key in the
  -- trajectory self-join, and NULL = NULL is never true in SQL. A row inserted without a
  -- segment would therefore vanish from trajectory entirely rather than error. The default
  -- makes the safe reading ("this is the rolled-up total") the automatic one.
  segment       TEXT NOT NULL DEFAULT 'all',
  as_of         TEXT,
  scope         TEXT,
  method        TEXT ${checkIn('method', OBSERVATION_METHOD)},
  source_url    TEXT,
  source_name   TEXT,
  note          TEXT
);

-- Cumulative deployment per technology — the x-axis of a learning curve. Kept apart from
-- metric_observations because it answers a different question ("how much has been built?"
-- not "where does the metric stand?") and feeds one consumer: the Wright's-law fit, which
-- needs cost against cumulative capacity rather than cost against time.
--
-- scenario is carried explicitly so a forecast capacity path can never be read as
-- observed history. Everything committed today is 'historical'; the column exists so that
-- if a scenario series is ever added, the fit can refuse to treat it as evidence.
CREATE TABLE capacity_series (
  id            INTEGER PRIMARY KEY,
  dependency_id INTEGER NOT NULL REFERENCES dependencies(id),
  metric        TEXT,
  value         REAL,
  unit          TEXT,
  basis         TEXT,
  as_of         TEXT,
  scope         TEXT,
  scenario      TEXT,
  method        TEXT ${checkIn('method', OBSERVATION_METHOD)},
  source_url    TEXT,
  source_name   TEXT,
  note          TEXT
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

-- Progress: the normalized plotting layer. Per (dependency, metric, scope), baseline B is the
-- declared attempt-era value (or the earliest observation as a fallback) and T the bar;
-- progress reads 0 at baseline, 1 at viability, >1 past the bar, negative if the metric receded
-- below where it started — dimensionless, so every dependency shares one axis. progress is
-- baseline-dependent by nature, so it is NULL (with a progress_status reason) when the baseline
-- can't express a 0->1 range: B == T (divide-by-zero) or B already past the bar (would invert).
-- Crossing is answered directly in the trajectory view, never from progress. progress_alt is
-- the same against a contested bar's alternative; log_distance = ln(T/v) is baseline-free and
-- reads cost-curve (below_is_better) metrics correctly on a log axis.
-- segment is part of the series key everywhere below: a BEV pack and an all-segment pack are
-- different series that happen to share a bar, so they must get their own baseline, their own
-- slope and their own crossing date. The threshold join deliberately does NOT include segment
-- -- a bar is declared per (metric, scope) and applies to every slice of it.
CREATE VIEW progress AS
WITH obs AS (
  SELECT
    o.dependency_id, o.metric, o.segment, o.scope, o.as_of,
    o.value AS value_raw, o.unit,
    FIRST_VALUE(o.value) OVER (
      PARTITION BY o.dependency_id, o.metric, o.segment, o.scope ORDER BY o.as_of
    ) AS earliest_value
  FROM metric_observations o
),
joined AS (
  SELECT
    obs.dependency_id, obs.metric, obs.segment, obs.scope, obs.as_of, obs.value_raw, obs.unit,
    COALESCE(t.baseline_value, obs.earliest_value) AS baseline,
    t.threshold_value AS threshold, t.threshold_direction AS direction,
    t.threshold_contested AS contested, t.threshold_alt_value AS threshold_alt,
    t.policy_dependent AS policy_dependent
  FROM obs
  JOIN dependency_thresholds t
    ON t.dependency_id = obs.dependency_id AND t.metric = obs.metric AND t.scope = obs.scope
  WHERE t.threshold_value IS NOT NULL
),
statused AS (
  SELECT
    joined.*,
    CASE
      WHEN baseline = threshold THEN 'baseline_equals_threshold'
      WHEN (direction = 'below_is_better' AND baseline < threshold)
        OR (direction = 'above_is_better' AND baseline > threshold) THEN 'baseline_past_threshold'
      ELSE 'ok'
    END AS progress_status
  FROM joined
)
SELECT
  s.dependency_id, s.metric, s.segment, s.scope, s.as_of, s.value_raw, s.unit, s.baseline,
  s.threshold, s.direction, s.contested, s.threshold_alt, s.policy_dependent,
  s.progress_status,
  CASE WHEN s.progress_status = 'ok' THEN
    CASE s.direction
      WHEN 'below_is_better' THEN (s.baseline - s.value_raw) / (s.baseline - s.threshold)
      WHEN 'above_is_better' THEN (s.value_raw - s.baseline) / (s.threshold - s.baseline)
    END
  END AS progress,
  CASE WHEN s.progress_status = 'ok' AND s.contested = 1 AND s.threshold_alt IS NOT NULL THEN
    CASE s.direction
      WHEN 'below_is_better' THEN (s.baseline - s.value_raw) / NULLIF(s.baseline - s.threshold_alt, 0)
      WHEN 'above_is_better' THEN (s.value_raw - s.baseline) / NULLIF(s.threshold_alt - s.baseline, 0)
    END
  END AS progress_alt,
  CASE WHEN s.direction = 'below_is_better' AND s.value_raw > 0 AND s.threshold > 0
    THEN ln(s.threshold / s.value_raw) END AS log_distance
FROM statused s;

-- Trajectory: one summary row per series. Not pure — it needs an ordered window and tunable
-- cutoffs, so the window size and plateau threshold come from trajectory_config (data, not
-- baked SQL). slope is Δprogress over the trailing window in progress-per-year; the state
-- (direction of travel) falls out of its sign and magnitude. currently_crossed and
-- became_viable_date are DIRECT scalar tests of the value against the bar — baseline-free, so a
-- bad baseline can never invert the viability answer the ranking keys on.
CREATE VIEW trajectory AS
WITH ranked AS (
  SELECT
    p.dependency_id, p.metric, p.segment, p.scope, p.as_of, p.progress,
    p.value_raw, p.threshold, p.direction, p.policy_dependent,
    ROW_NUMBER() OVER (PARTITION BY p.dependency_id, p.metric, p.segment, p.scope ORDER BY p.as_of DESC) AS rn_desc,
    COUNT(*) OVER (PARTITION BY p.dependency_id, p.metric, p.segment, p.scope) AS n_obs
  FROM progress p
),
paired AS (
  SELECT
    l.dependency_id, l.metric, l.segment, l.scope,
    l.as_of AS latest_as_of, l.progress AS latest_progress, l.n_obs,
    l.value_raw AS latest_value, l.threshold, l.direction, l.policy_dependent,
    s.as_of AS window_start_as_of, s.progress AS window_start_progress,
    (l.progress - s.progress)
      / NULLIF((julianday(l.as_of || '-01') - julianday(s.as_of || '-01')) / 365.25, 0) AS slope
  FROM ranked l
  JOIN ranked s
    ON s.dependency_id = l.dependency_id AND s.metric = l.metric
   AND s.segment = l.segment AND s.scope = l.scope
   AND l.rn_desc = 1
   AND s.rn_desc = MIN(l.n_obs, (SELECT window_n FROM trajectory_config LIMIT 1))
)
SELECT
  paired.dependency_id, paired.metric, paired.segment, paired.scope,
  paired.latest_as_of, paired.latest_progress, paired.n_obs,
  paired.window_start_as_of, paired.window_start_progress, paired.slope,
  CASE
    -- Checked before the economics-only arms, and deliberately so: for a policy-dependent
    -- dependency below its bar, "improving" would answer the wrong question. The direction
    -- of travel is real but it is not what decides revivability — the policy regime is.
    -- conditional: typology cell pending gap-typology build.
    WHEN paired.policy_dependent = 1 AND paired.latest_progress < 1 THEN 'conditional'
    WHEN paired.n_obs < 2 OR paired.slope IS NULL THEN 'unknown'
    WHEN paired.slope > (SELECT plateau_slope_threshold FROM trajectory_config LIMIT 1) THEN 'improving'
    WHEN paired.slope < -(SELECT plateau_slope_threshold FROM trajectory_config LIMIT 1) THEN 'receded'
    ELSE 'plateaued'
  END AS state,
  CASE
    WHEN paired.direction = 'below_is_better' AND paired.latest_value <= paired.threshold THEN 1
    WHEN paired.direction = 'above_is_better' AND paired.latest_value >= paired.threshold THEN 1
    ELSE 0
  END AS currently_crossed,
  (SELECT MIN(p2.as_of) FROM progress p2
   WHERE p2.dependency_id = paired.dependency_id AND p2.metric = paired.metric
     AND p2.segment = paired.segment AND p2.scope = paired.scope
     AND ((p2.direction = 'below_is_better' AND p2.value_raw <= p2.threshold)
       OR (p2.direction = 'above_is_better' AND p2.value_raw >= p2.threshold))) AS became_viable_date
FROM paired;
`
