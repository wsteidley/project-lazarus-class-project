import {
  ASSESSMENT_STATUS,
  BASIS_CURRENCY,
  CHALLENGE,
  CHALLENGE_OUTCOME,
  CONFIDENCE,
  CRITICALITY,
  DEPENDENCY,
  DEPENDENCY_RELATION,
  DURATION,
  ENTITY_KIND,
  ENERGY_BASIS,
  EXIT_TYPE,
  LIVING_STATUS,
  OBSERVATION_METHOD,
  OUTCOME_TYPE,
  PROJECTION_METHOD,
  REGION,
  RESOLUTION_STATUS,
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
  -- UNIQUE for the same reason dependencies.name is: search_coverage references it, and SQLite
  -- rejects a FK whose parent key is not unique.
  name        TEXT NOT NULL UNIQUE,
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
  -- UNIQUE is required, not merely tidy: dependency_links references this column, and SQLite
  -- with PRAGMA foreign_keys = ON raises "foreign key mismatch" at insert time when a FK's
  -- parent key is not unique. Names were already de-facto unique -- the loader's
  -- dependencyIdByName map has always assumed it.
  name             TEXT NOT NULL UNIQUE,
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
  -- The bar's own basis triple, equality-joined against the observation's in the progress
  -- view. Without these the join matched on (dependency, metric, scope) alone, so an
  -- observation could be compared to a bar with NO declared basis of any kind -- which is
  -- exactly what the battery relabel did, and baseline rule #1 ("the threshold must share the
  -- series' basis") stayed aspirational for as long as there was nowhere to declare it.
  --
  -- NOT NULL DEFAULT 'na' on both sides is load-bearing, not tidiness: these are equality
  -- predicates, and NULL = NULL is never true in SQL, so a nullable column would make every
  -- unstamped row silently join nothing rather than join its own kind.
  basis                    TEXT NOT NULL DEFAULT 'na' ${checkIn('basis', BASIS_CURRENCY, false)},
  energy_basis             TEXT NOT NULL DEFAULT 'na' ${checkIn('energy_basis', ENERGY_BASIS, false)},
  duration                 TEXT NOT NULL DEFAULT 'na' ${checkIn('duration', DURATION, false)},
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
--
-- This is NOT dependency_links, which is a different relation entirely: this one runs
-- dependency -> dependency and carries causality; that one runs dependency -> reference_entity
-- and carries the time slice at which a dependency hangs off a curve. It was named
-- dependency_links until the technology/dependency split needed that name for the other
-- meaning; the two must never be conflated again.
CREATE TABLE dependency_edges (
  id                 INTEGER PRIMARY KEY,
  from_dependency_id INTEGER NOT NULL REFERENCES dependencies(id),
  to_dependency_id   INTEGER NOT NULL REFERENCES dependencies(id),
  relation           TEXT ${checkIn('relation', DEPENDENCY_RELATION, false)},
  note               TEXT
);

-- Data-bearing subjects a dependency can hang off: the things that HAVE curves. Solar PV is
-- one whether or not any company ever depended on it, which is exactly the distinction the
-- old conflation lost -- metric data was keyed on dependencies.name, so a technology with no
-- dependency row (CSP, hydro, geothermal, bioenergy, offshore wind) dropped every row it had.
--
-- Modelled for the ROLE, not the type. Today only kind='technology' carries curves, but an
-- "EV charging network" arriving later as kind='infrastructure' slots into the same link and
-- the same progress/trajectory machinery with no schema change. That is the whole reason this
-- is not called technology_id.
CREATE TABLE reference_entities (
  id   INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL ${checkIn('kind', ENTITY_KIND, false)},
  note TEXT
);

-- dependency -> subject, many-to-many, carrying the TIME SLICE. Many dependencies map to one
-- entity: solar-blocked companies from 2008, 2012 and 2016 are three dependencies, three eras
-- and three bars sharing ONE solar curve. That is the Lazarus question itself (the curve
-- moved; when did each cohort die relative to it?), and if dependency WERE technology it
-- would be unrepresentable.
--
-- A dependency with no row here is standalone/qualitative -- "public trust in AVs",
-- "supportive net-metering" -- and that is a first-class state, not missing data. It is what
-- lets a company blocked on something non-measurable into the dataset with its blocker named
-- (P2), which is the whole reason this split is load-bearing rather than a tidy-up.
--
-- Deliberately carries NO threshold_value/threshold_unit, against the letter of
-- technology-dependency-split-spec-v1.md: dependency_thresholds already owns bars, with
-- contested/alt/baseline/policy_dependent and now the basis triple. Two stored homes for "the
-- bar" means a viability call cannot say which one it used, which P4 forbids. The split spec
-- predates dependency_thresholds v2.
CREATE TABLE dependency_links (
  dependency_name TEXT NOT NULL REFERENCES dependencies(name),
  entity_id       INTEGER NOT NULL REFERENCES reference_entities(id),
  era             TEXT,
  note            TEXT,
  PRIMARY KEY (dependency_name, entity_id)
);

-- An explicit record that a region WAS searched for companies. Its ABSENCE is what makes a
-- gap-map cell 'unsampled' rather than 'white_space', and that distinction is the single most
-- dangerous one in the whole typology: inferring "we looked and found nothing" from "we have
-- nothing" would invent opportunities nobody ever checked for. So it is never inferred. A cell
-- is white_space only where a row here says someone actually looked.
--
-- dependency_name NULL means the whole idea space was swept, rather than one (idea x blocker)
-- cell within it.
CREATE TABLE search_coverage (
  id              INTEGER PRIMARY KEY,
  idea_space_name TEXT NOT NULL REFERENCES idea_spaces(name),
  dependency_name TEXT REFERENCES dependencies(name),
  searched_on     TEXT NOT NULL,
  -- What was actually swept ('techcrunch-2015-2024'), so a thin sweep can be told from a
  -- thorough one rather than both reading as a flat "searched".
  source          TEXT NOT NULL,
  method          TEXT NOT NULL ${checkIn('method', OBSERVATION_METHOD, false)},
  note            TEXT
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
  -- The one table that needs BOTH: entity_id says whose CURVE was extrapolated, dependency_id
  -- says whose BAR the crossing is against. Same fit-vs-forecast separation wright_fits makes
  -- above -- and dependency_id is nullable because a fit can run with no bar at all.
  entity_id     INTEGER NOT NULL REFERENCES reference_entities(id),
  dependency_id INTEGER REFERENCES dependencies(id),
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
-- Entity-keyed, with no dependency column at all: a learning rate is a property of the
-- technology's curve, and stays true no matter how many dependencies hang off it (or none).
CREATE TABLE wright_fits (
  entity_id     INTEGER NOT NULL REFERENCES reference_entities(id),
  metric        TEXT NOT NULL,
  scope         TEXT NOT NULL,
  segment       TEXT NOT NULL,
  learning_rate REAL,
  b             REAL,
  r2            REAL,
  n_pairs       INTEGER,
  first_as_of   TEXT,
  last_as_of    TEXT,
  PRIMARY KEY (entity_id, metric, scope, segment)
);

-- A company's blockers. dependency_id is NULLABLE and that is the point (P2): a blocker the
-- resolver could not fold onto the canonical list is RETAINED here, named in the extractor's
-- own words, rather than dropped. step1c deliberately emits those rows so they can be curated;
-- the loader used to discard them into an anonymous skip counter, so the company survived and
-- its reason for dying did not. resolution_status makes the two cases queryable apart, which
-- is P3: "no canonical match" and "no blocker recorded" must never be the same blank.
CREATE TABLE company_dependencies (
  id                  INTEGER PRIMARY KEY,
  company_id          INTEGER NOT NULL REFERENCES companies(id),
  dependency_id       INTEGER REFERENCES dependencies(id),
  -- What the extractor actually said. Always populated: for a resolved row it is the
  -- pre-canonicalisation spelling, which is what makes a bad match auditable after the fact.
  dependency_name_raw TEXT NOT NULL DEFAULT '',
  resolution_status   TEXT NOT NULL DEFAULT 'resolved'
    ${checkIn('resolution_status', RESOLUTION_STATUS, false)},
  criticality         TEXT ${checkIn('criticality', CRITICALITY, false)},
  detail              TEXT,
  source_url          TEXT,
  -- The two columns cannot disagree: a resolved row has an id, an unresolved one does not.
  CHECK ((resolution_status = 'resolved') = (dependency_id IS NOT NULL))
);

-- COALESCE, not a plain UNIQUE(company_id, dependency_id, dependency_name_raw): SQLite treats
-- NULLs in a UNIQUE index as distinct from each other, so two identical unresolved rows would
-- both insert and INSERT OR IGNORE would quietly stop deduplicating.
CREATE UNIQUE INDEX company_dependencies_key
  ON company_dependencies (company_id, COALESCE(dependency_id, -1), dependency_name_raw);

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
-- The basis TRIPLE is not decoration: unit alone does not identify a series. USD/W in nominal
-- dollars and USD/W in constant 2024 dollars are different measurements, and comparing one
-- against a bar declared on the other silently answers the viability question wrong. One
-- basis per series (v3 Fix 2). basis is the currency vintage; energy_basis is which kWh the
-- denominator counts (nameplate vs usable, AC vs DC); duration is which storage duration a
-- $/kWh refers to. They were one column until the split, which is why the IRENA battery
-- qualifiers ("usable kWh, blended duration") had to ride in a note string.
-- segment names the SLICE the value covers: 'all' is the rolled-up total, and the parts
-- (bev/stationary, onshore/offshore, on_grid/off_grid) sit alongside it. Never sum a total
-- together with its own parts. It is distinct from basis: basis says how a value was
-- measured (constant vs nominal dollars, DC vs AC), segment says which subset was measured.
-- Without it a slice has to be smuggled into the metric name -- which is what
-- "battery pack price (BEV)" was doing, silently preventing the bar declared on
-- "battery pack price" from ever joining its own observations.
CREATE TABLE metric_observations (
  id            INTEGER PRIMARY KEY,
  -- Keyed on the ENTITY that has the curve, not on a dependency. Metric data is published per
  -- technology (IRENA/OWID), and keying it on dependencies.name meant a technology with no
  -- dependency row dropped every observation it had -- 120 rows across CSP, hydro, geothermal,
  -- bioenergy and offshore wind. A technology is valid on its own, dependencies or not.
  entity_id     INTEGER NOT NULL REFERENCES reference_entities(id),
  metric        TEXT,
  value         REAL,
  unit          TEXT,
  basis         TEXT NOT NULL DEFAULT 'na' ${checkIn('basis', BASIS_CURRENCY, false)},
  energy_basis  TEXT NOT NULL DEFAULT 'na' ${checkIn('energy_basis', ENERGY_BASIS, false)},
  duration      TEXT NOT NULL DEFAULT 'na' ${checkIn('duration', DURATION, false)},
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
-- The basis triple here is mostly 'na': a cumulative-deployment figure has a unit (GW, GWh)
-- but no currency. The column that used to be basis on this table was carrying three
-- different things at once and none of them was a currency -- 'AC' was an energy_basis,
-- 'onshore' was a segment (which this table had nowhere to put), and 'energy' was just
-- restating the unit. The split destructures all three.
CREATE TABLE capacity_series (
  id            INTEGER PRIMARY KEY,
  entity_id     INTEGER NOT NULL REFERENCES reference_entities(id),
  metric        TEXT,
  value         REAL,
  unit          TEXT,
  basis         TEXT NOT NULL DEFAULT 'na' ${checkIn('basis', BASIS_CURRENCY, false)},
  energy_basis  TEXT NOT NULL DEFAULT 'na' ${checkIn('energy_basis', ENERGY_BASIS, false)},
  duration      TEXT NOT NULL DEFAULT 'na' ${checkIn('duration', DURATION, false)},
  segment       TEXT NOT NULL DEFAULT 'all',
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
--
-- POST-SPLIT SHAPE, and the one hazard worth stating loudly. Observations are entity-keyed and
-- reach a dependency through dependency_links, which is MANY-to-one: one solar curve can serve
-- three dependencies (2008, 2012, 2016 cohorts) with three different bars. So one entity's
-- curve emits N progress rows, and the series key is now
--   (dependency_id, entity_id, metric, segment, scope)
-- Every partition and join key in the trajectory view below carries dependency_id for exactly
-- this reason. Drop it there and three dependencies' points pour into one series, producing a
-- silently wrong slope and crossing date -- the same failure segment was added to prevent.
CREATE VIEW progress AS
WITH obs AS (
  SELECT
    o.entity_id, o.metric, o.segment, o.scope, o.as_of,
    o.basis, o.energy_basis, o.duration,
    o.value AS value_raw, o.unit,
    -- Partitioned on entity, NOT dependency: the earliest point of a curve is a property of
    -- the curve. Three dependencies sharing it share its baseline unless they declare their
    -- own (COALESCE below), which is what the era slice is for.
    FIRST_VALUE(o.value) OVER (
      PARTITION BY o.entity_id, o.metric, o.segment, o.scope ORDER BY o.as_of
    ) AS earliest_value
  FROM metric_observations o
),
joined AS (
  SELECT
    d.id AS dependency_id, obs.entity_id, e.name AS entity_name, dl.era,
    obs.metric, obs.segment, obs.scope, obs.as_of, obs.value_raw, obs.unit,
    obs.basis, obs.energy_basis, obs.duration,
    COALESCE(t.baseline_value, obs.earliest_value) AS baseline,
    t.threshold_value AS threshold, t.threshold_direction AS direction,
    t.threshold_contested AS contested, t.threshold_alt_value AS threshold_alt,
    t.policy_dependent AS policy_dependent,
    t.threshold_source_url AS threshold_source_url, t.threshold_as_of AS threshold_as_of
  FROM obs
  JOIN reference_entities e ON e.id = obs.entity_id
  -- The hop that replaces the old dependency-keyed observation. A technology with no linked
  -- dependency simply produces no progress rows -- correct, not a gap: its observations and
  -- its Wright fit are still there, there is just no failure story to measure against yet.
  JOIN dependency_links dl  ON dl.entity_id = obs.entity_id
  JOIN dependencies d       ON d.name = dl.dependency_name
  JOIN dependency_thresholds t
    ON t.dependency_id = d.id AND t.metric = obs.metric AND t.scope = obs.scope
   -- Baseline rule #1, finally enforceable: a value is only comparable to a bar declared on
   -- the same currency, the same energy denominator and the same duration. Equality on all
   -- three, with 'na' matching only 'na'.
   AND t.basis = obs.basis AND t.energy_basis = obs.energy_basis AND t.duration = obs.duration
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
  s.dependency_id, s.entity_id, s.entity_name, s.era,
  s.metric, s.segment, s.scope, s.as_of, s.value_raw, s.unit, s.baseline,
  -- P4: the call carries what it was judged against, so a user can see the argument and
  -- disagree with it ("that bar is stale, and it is declared on nameplate kWh anyway")
  -- rather than being handed a verdict with no basis attached.
  s.basis, s.energy_basis, s.duration, s.threshold_source_url, s.threshold_as_of,
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
--
-- SERIES KEY, post-split: (dependency_id, entity_id, metric, segment, scope). BOTH id columns
-- are load-bearing and the failure is silent in each direction:
--   drop dependency_id -> three cohorts sharing one solar curve collapse into one series, and
--                         each one's crossing date is computed against the wrong bar;
--   drop entity_id     -> a dependency linked to two entities pours both curves into one
--                         series and fits a line through points that are not comparable.
-- Either way the slope and the crossing date come out wrong with nothing erroring. There is a
-- test for the first case; do not remove either column from these partitions.
CREATE VIEW trajectory AS
WITH ranked AS (
  SELECT
    p.dependency_id, p.entity_id, p.metric, p.segment, p.scope, p.as_of, p.progress,
    p.value_raw, p.threshold, p.direction, p.policy_dependent,
    ROW_NUMBER() OVER (PARTITION BY p.dependency_id, p.entity_id, p.metric, p.segment, p.scope ORDER BY p.as_of DESC) AS rn_desc,
    COUNT(*) OVER (PARTITION BY p.dependency_id, p.entity_id, p.metric, p.segment, p.scope) AS n_obs
  FROM progress p
),
paired AS (
  SELECT
    l.dependency_id, l.entity_id, l.metric, l.segment, l.scope,
    l.as_of AS latest_as_of, l.progress AS latest_progress, l.n_obs,
    l.value_raw AS latest_value, l.threshold, l.direction, l.policy_dependent,
    s.as_of AS window_start_as_of, s.progress AS window_start_progress,
    (l.progress - s.progress)
      / NULLIF((julianday(l.as_of || '-01') - julianday(s.as_of || '-01')) / 365.25, 0) AS slope
  FROM ranked l
  JOIN ranked s
    ON s.dependency_id = l.dependency_id AND s.entity_id = l.entity_id AND s.metric = l.metric
   AND s.segment = l.segment AND s.scope = l.scope
   AND l.rn_desc = 1
   AND s.rn_desc = MIN(l.n_obs, (SELECT window_n FROM trajectory_config LIMIT 1))
)
SELECT
  paired.dependency_id, paired.entity_id, paired.metric, paired.segment, paired.scope,
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
   WHERE p2.dependency_id = paired.dependency_id AND p2.entity_id = paired.entity_id
     AND p2.metric = paired.metric
     AND p2.segment = paired.segment AND p2.scope = paired.scope
     AND ((p2.direction = 'below_is_better' AND p2.value_raw <= p2.threshold)
       OR (p2.direction = 'above_is_better' AND p2.value_raw >= p2.threshold))) AS became_viable_date
FROM paired;

-- ---------------------------------------------------------------------------
-- P3: absence is a NAMED state, never a bare null. Three grains, each built on the one
-- above it. Deliberately views and not stored columns: every input (does a series exist,
-- does a bar exist, does the latest value clear it) is recomputed on each build, so a
-- stored label would be a second source of truth that goes stale the moment
-- data/derived is regenerated.
--
-- On why progress is NOT LEFT JOINed to cover the no-bar case, which is the obvious
-- alternative to these views and is wrong twice over:
--   1. It produces exactly the null P3 forbids. With no bar, threshold/direction/progress
--      all come back NULL, and progress_status already means something else entirely
--      (baseline_equals_threshold / baseline_past_threshold), so no query could tell "no
--      bar" from "bar exists but the baseline is degenerate".
--   2. It turns a gap into a verdict. trajectory.currently_crossed is CASE ... ELSE 0, so
--      a NULL threshold falls through to "not crossed" and an unassessed series gets
--      published as ASSESSED NOT VIABLE. That is the worst outcome available here.
-- So progress stays "the series where a normalized 0->1 axis is meaningful" and remains a
-- strict SUBSET of series_status below. There is a test pinning that relationship.
-- ---------------------------------------------------------------------------

-- Grain 1: one row per metric series, with or without a bar. Entity-keyed like the
-- observations themselves, and carrying dependency_id where a link exists -- a technology with
-- no dependency still gets a row here, which is how its data stays visible (P2).
CREATE VIEW series_status AS
WITH series AS (
  SELECT o.entity_id, o.metric, o.segment, o.scope,
         COUNT(*) AS n_obs, MIN(o.as_of) AS first_as_of, MAX(o.as_of) AS last_as_of
  FROM metric_observations o
  WHERE o.value IS NOT NULL
  GROUP BY o.entity_id, o.metric, o.segment, o.scope
),
latest AS (
  SELECT o.entity_id, o.metric, o.segment, o.scope, o.value, o.as_of, o.unit,
         o.basis, o.energy_basis, o.duration,
         ROW_NUMBER() OVER (
           PARTITION BY o.entity_id, o.metric, o.segment, o.scope ORDER BY o.as_of DESC
         ) AS rn
  FROM metric_observations o
  WHERE o.value IS NOT NULL
),
barred AS (
  SELECT s.entity_id, e.name AS entity_name, d.id AS dependency_id, dl.era,
         s.metric, s.segment, s.scope, s.n_obs, s.first_as_of, s.last_as_of,
         l.value AS latest_value, l.as_of AS latest_as_of, l.unit,
         l.basis, l.energy_basis, l.duration,
         t.threshold_value, t.threshold_unit, t.threshold_direction, t.threshold_as_of,
         t.threshold_source_url, t.policy_dependent
  FROM series s
  JOIN latest l
    ON l.entity_id = s.entity_id AND l.metric = s.metric
   AND l.segment = s.segment AND l.scope = s.scope AND l.rn = 1
  JOIN reference_entities e ON e.id = s.entity_id
  -- LEFT, unlike progress: an unlinked technology's series must still be describable. progress
  -- can inner-join because a 0->1 axis needs a bar; a STATE does not.
  LEFT JOIN dependency_links dl ON dl.entity_id = s.entity_id
  LEFT JOIN dependencies d      ON d.name = dl.dependency_name
  -- Same triple predicate as progress. A bar declared on a different basis does not apply to
  -- this series, so the series reads no_threshold rather than being judged against it. The
  -- build-time checker (lib/basis.ts) is what stops that silently meaning "no data".
  LEFT JOIN dependency_thresholds t
    ON t.dependency_id = d.id AND t.metric = s.metric AND t.scope = s.scope
   AND t.basis = l.basis AND t.energy_basis = l.energy_basis AND t.duration = l.duration
)
SELECT
  b.*,
  CASE
    -- No bar at all, or a quantitative_tbd row whose value was never set. Both mean the
    -- same thing to a user: measurable, not yet judged. A curation gap, not a dead end.
    WHEN b.threshold_value IS NULL THEN 'no_threshold'
    -- Bar and series both present but no verdict is computable. NOT the same as progress
    -- being null: 'Grid interconnection capacity' has baseline == threshold (progress null)
    -- and a perfectly computable crossing, so it is assessed_not_viable, not undetermined.
    -- Conflating the two null-vocabularies here would report a real verdict as a gap.
    WHEN b.threshold_direction IS NULL OR b.latest_value IS NULL THEN 'undetermined'
    -- Kept textually parallel to trajectory.currently_crossed so the two cannot drift.
    WHEN (b.threshold_direction = 'below_is_better' AND b.latest_value <= b.threshold_value)
      OR (b.threshold_direction = 'above_is_better' AND b.latest_value >= b.threshold_value)
      THEN 'assessed_viable'
    ELSE 'assessed_not_viable'
  END AS status
FROM barred b;

-- Grain 2: one row per dependency. Adds no_blocker_data, which cannot exist at the series
-- grain — a dependency with no series has no row there to carry a label. Counts every
-- state alongside the winner so the roll-up never HIDES the others.
CREATE VIEW dependency_status AS
SELECT
  d.id AS dependency_id, d.name AS dependency_name, d.threshold_kind,
  COUNT(s.metric)                                    AS n_series,
  COALESCE(SUM(s.status = 'assessed_viable'), 0)     AS n_assessed_viable,
  COALESCE(SUM(s.status = 'assessed_not_viable'), 0) AS n_assessed_not_viable,
  COALESCE(SUM(s.status = 'undetermined'), 0)        AS n_undetermined,
  COALESCE(SUM(s.status = 'no_threshold'), 0)        AS n_no_threshold,
  EXISTS (SELECT 1 FROM dependency_links dl WHERE dl.dependency_name = d.name) AS has_entity,
  CASE
    -- Link-absence is now the structural fact, checked first, per the gap typology's
    -- first-match-wins order. But link-absence ALONE is not enough: it cannot tell
    -- "non-measurable by nature" from "measurable, nobody has sourced a curve yet". Only a
    -- dependency declared qualitative is the former; the rest are an honest data gap, and
    -- calling them qualitative_blocker would quietly write off work still worth doing.
    WHEN NOT EXISTS (SELECT 1 FROM dependency_links dl WHERE dl.dependency_name = d.name)
     AND d.threshold_kind = 'qualitative' THEN 'qualitative_blocker'
    WHEN COUNT(s.metric) = 0                                      THEN 'no_blocker_data'
    WHEN SUM(s.status = 'assessed_viable')     > 0                THEN 'assessed_viable'
    WHEN SUM(s.status = 'assessed_not_viable') > 0                THEN 'assessed_not_viable'
    WHEN SUM(s.status = 'undetermined')        > 0                THEN 'undetermined'
    ELSE 'no_threshold'
  END AS status
FROM dependencies d
LEFT JOIN series_status s ON s.dependency_id = d.id
GROUP BY d.id, d.name, d.threshold_kind;

-- Grain 3: one row per company x blocker, INCLUDING blockers that resolved to nothing
-- canonical (those carry a null dependency_id and fall to no_blocker_data). This is the P3
-- query surface: "show me companies whose blocker we have no data on" is
--   WHERE status = 'no_blocker_data'
-- which is no harder to write than the assessed_viable case. That symmetry is the point.
CREATE VIEW company_dependency_status AS
SELECT
  cd.company_id, c.company_name, c.year_defunct, c.idea_space_id,
  cd.dependency_id,
  COALESCE(d.name, cd.dependency_name_raw) AS dependency_label,
  cd.dependency_name_raw, cd.resolution_status, cd.criticality, cd.detail,
  COALESCE(ds.status, 'no_blocker_data') AS status,
  ds.n_series, ds.n_assessed_viable, ds.n_assessed_not_viable,
  ds.n_undetermined, ds.n_no_threshold
FROM company_dependencies cd
JOIN companies c               ON c.id = cd.company_id
LEFT JOIN dependencies d       ON d.id = cd.dependency_id
LEFT JOIN dependency_status ds ON ds.dependency_id = cd.dependency_id;


-- ---------------------------------------------------------------------------
-- GAP TYPOLOGY: the view that makes the space navigable. Not a scoring engine that labels
-- winners -- a MAP, including its blank regions, where the unknown and no-data areas are as
-- first-class as the classified ones.
--
-- It never removes a company and it never claims a viability FACT. Every classified cell
-- carries the reasoning it was produced from (which bar, which series, which basis triple,
-- which dates) so a user can inspect the argument and disagree with it (P1/P4). A cell with no
-- reasoning attached is a bug.
-- ---------------------------------------------------------------------------

-- One representative series per dependency, plus the roll-up facts the classification needs.
-- A dependency can have several series (metrics, segments, scopes); the cell grain is
-- company x dependency, so one has to be chosen, and the choice is deliberately the one most
-- favourable to the LAZARUS reading: a crossed series first, then the earliest crossing. If
-- any series says the blocker cleared its bar, that is the finding worth surfacing, and the
-- per-series detail is still there in trajectory for anyone who wants it.
CREATE VIEW dependency_signal AS
WITH ranked AS (
  SELECT
    t.dependency_id, t.entity_id, t.metric, t.segment, t.scope,
    t.state, t.currently_crossed, t.became_viable_date, t.latest_as_of, t.n_obs,
    ROW_NUMBER() OVER (
      PARTITION BY t.dependency_id
      ORDER BY t.currently_crossed DESC, t.became_viable_date ASC, t.metric
    ) AS rn
  FROM trajectory t
),
rolled AS (
  SELECT
    dependency_id,
    COUNT(*) AS n_series,
    MAX(currently_crossed) AS any_crossed,
    MIN(became_viable_date) AS first_viable_date,
    MAX(state = 'conditional') AS any_conditional,
    MAX(state = 'receded') AS any_receded
  FROM trajectory GROUP BY dependency_id
)
SELECT
  r.dependency_id, r.entity_id, r.metric, r.segment, r.scope,
  r.state, r.currently_crossed, r.became_viable_date, r.latest_as_of, r.n_obs,
  rolled.n_series, rolled.any_crossed, rolled.first_viable_date,
  rolled.any_conditional, rolled.any_receded
FROM ranked r JOIN rolled ON rolled.dependency_id = r.dependency_id
WHERE r.rn = 1;

-- Every company x blocker pairing, classified. First match wins, absence cells before verdict
-- cells -- because "we cannot say" outranks a guess made without data.
--
-- Starts FROM companies, LEFT JOINed: a company with NO recorded blockers at all still gets a
-- cell. Starting from the pairings would have silently dropped it from the map, which is a P2
-- regression hiding inside the payoff view.
CREATE VIEW gap_cells AS
SELECT
  c.id AS company_id, c.company_name, c.year_defunct, c.living_status,
  c.idea_space_id, i.name AS idea_space_name,
  cds.dependency_id, cds.dependency_label, cds.resolution_status, cds.criticality,
  -- P4 payload: what the call was judged against. A classified cell without this is a bug.
  sig.entity_id, e.name AS entity_name, dl.era,
  sig.metric, sig.segment, sig.scope,
  ss.basis, ss.energy_basis, ss.duration,
  ss.threshold_value, ss.threshold_unit, ss.threshold_direction,
  ss.threshold_as_of, ss.threshold_source_url,
  sig.latest_as_of, sig.state, sig.currently_crossed, sig.became_viable_date, sig.n_obs,
  COALESCE(cds.status, 'no_blocker_data') AS absence_state,
  CASE
    -- The absence cells, first. These are honest statements about what we cannot say, and
    -- they must never be overwritten by a verdict computed from data that isn't there.
    WHEN cds.company_id IS NULL                     THEN 'no_blocker_data'
    WHEN cds.status = 'qualitative_blocker'         THEN 'qualitative_blocker'
    WHEN cds.status = 'no_blocker_data'             THEN 'no_blocker_data'
    WHEN cds.status = 'no_threshold'                THEN 'unassessed'
    WHEN cds.status = 'undetermined'                THEN 'undetermined'
    WHEN sig.dependency_id IS NULL                  THEN 'undetermined'
    -- Then the classified cells.
    WHEN sig.any_conditional = 1                    THEN 'conditional'
    -- Crossed AFTER the company died: the idea's blocker became viable too late for them, and
    -- may be viable for someone now. The whole point of the dataset.
    WHEN sig.any_crossed = 1 AND sig.first_viable_date IS NOT NULL
         AND c.year_defunct IS NOT NULL
         AND CAST(substr(sig.first_viable_date, 1, 4) AS INTEGER) > c.year_defunct
                                                    THEN 'lazarus_candidate'
    -- Crossed, then backslid. A different answer from lazarus_candidate: the window opened
    -- AND closed, so "it's viable now" would be wrong.
    WHEN sig.any_crossed = 0 AND sig.first_viable_date IS NOT NULL
                                                    THEN 'crossed_and_receded'
    WHEN sig.any_crossed = 1                        THEN 'still_viable'
    WHEN sig.any_receded = 1                        THEN 'receded'
    ELSE 'tested_dead_end'
  END AS cell
FROM companies c
LEFT JOIN idea_spaces i               ON i.id = c.idea_space_id
LEFT JOIN company_dependency_status cds ON cds.company_id = c.id
LEFT JOIN dependency_signal sig       ON sig.dependency_id = cds.dependency_id
LEFT JOIN reference_entities e        ON e.id = sig.entity_id
LEFT JOIN dependencies d              ON d.id = cds.dependency_id
LEFT JOIN dependency_links dl         ON dl.dependency_name = d.name AND dl.entity_id = sig.entity_id
LEFT JOIN series_status ss
  ON ss.dependency_id = sig.dependency_id AND ss.metric = sig.metric
 AND ss.segment = sig.segment AND ss.scope = sig.scope;

-- The blank regions: (idea space x dependency) cells where no company exists. NOT a company
-- row, so it is kept as its own view rather than nullable columns bolted onto gap_cells.
--
-- 'unsampled' is the DEFAULT and that is the cardinal rule of this typology. A cell is
-- white_space -- genuinely searched, genuinely empty -- only where search_coverage confirms
-- someone actually looked. Mislabelling absence-of-search as absence-of-companies would
-- invent opportunities that were never checked.
CREATE VIEW space_cells AS
SELECT
  NULL AS company_id, NULL AS company_name,
  i.id AS idea_space_id, i.name AS idea_space_name,
  d.id AS dependency_id, d.name AS dependency_label,
  CASE WHEN EXISTS (
    SELECT 1 FROM search_coverage sc
    WHERE sc.idea_space_name = i.name
      AND (sc.dependency_name IS NULL OR sc.dependency_name = d.name)
  ) THEN 'white_space' ELSE 'unsampled' END AS cell
FROM idea_spaces i
CROSS JOIN dependencies d
WHERE NOT EXISTS (
  SELECT 1 FROM company_dependencies cd
  JOIN companies c2 ON c2.id = cd.company_id
  WHERE c2.idea_space_id = i.id AND cd.dependency_id = d.id
);

-- The headline surface (P3/P5): "show me no_blocker_data" and "show me lazarus_candidate" are
-- the same query shape. That symmetry is the entire point -- the gaps are not an edge case to
-- be filtered out, they are where the user's own knowledge does the work the data cannot.
CREATE VIEW gap_map AS
  SELECT company_id, company_name, idea_space_id, idea_space_name,
         dependency_id, dependency_label, cell FROM gap_cells
  UNION ALL
  SELECT company_id, company_name, idea_space_id, idea_space_name,
         dependency_id, dependency_label, cell FROM space_cells;

-- Company roll-up: the strongest signal, and the counts that stop it HIDING the rest. A company
-- that is lazarus_candidate on one blocker and no_blocker_data on another is BOTH --
-- headline_cell names the first, the n_ columns prove the second.
--
-- Note the ranking puts no_blocker_data ABOVE the weak classified cells. It is the
-- highest-value gap: a company with one known dead end and one blocker we know nothing about
-- should surface as the unknown, because that is where an outside reader can contribute.
CREATE VIEW company_gap_summary AS
SELECT
  g.company_id, g.company_name, COUNT(*) AS n_cells,
  SUM(g.cell = 'lazarus_candidate')   AS n_lazarus_candidate,
  SUM(g.cell = 'crossed_and_receded') AS n_crossed_and_receded,
  SUM(g.cell = 'conditional')         AS n_conditional,
  SUM(g.cell = 'receded')             AS n_receded,
  SUM(g.cell = 'tested_dead_end')     AS n_tested_dead_end,
  SUM(g.cell = 'still_viable')        AS n_still_viable,
  SUM(g.cell = 'no_blocker_data')     AS n_no_blocker_data,
  SUM(g.cell = 'unassessed')          AS n_unassessed,
  SUM(g.cell = 'undetermined')        AS n_undetermined,
  SUM(g.cell = 'qualitative_blocker') AS n_qualitative_blocker,
  CASE
    WHEN SUM(g.cell = 'lazarus_candidate')   > 0 THEN 'lazarus_candidate'
    WHEN SUM(g.cell = 'crossed_and_receded') > 0 THEN 'crossed_and_receded'
    WHEN SUM(g.cell = 'conditional')         > 0 THEN 'conditional'
    WHEN SUM(g.cell = 'no_blocker_data')     > 0 THEN 'no_blocker_data'
    WHEN SUM(g.cell = 'unassessed')          > 0 THEN 'unassessed'
    WHEN SUM(g.cell = 'undetermined')        > 0 THEN 'undetermined'
    WHEN SUM(g.cell = 'qualitative_blocker') > 0 THEN 'qualitative_blocker'
    WHEN SUM(g.cell = 'receded')             > 0 THEN 'receded'
    WHEN SUM(g.cell = 'still_viable')        > 0 THEN 'still_viable'
    ELSE 'tested_dead_end'
  END AS headline_cell
FROM gap_cells g GROUP BY g.company_id, g.company_name;
`
