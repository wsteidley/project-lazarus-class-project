# Project Lazarus — Dataset Schema Spec

Target for the extraction pipeline's output: a small **relational** dataset that
lets you query past climate-tech failures by cause, funding, and — critically —
whether the thing that killed them has since improved.

## Storage decision

- **Source of truth / artifact:** one CSV per table (portable, diffable, re-importable).
- **Query layer:** a single-file embedded DB built from those CSVs — **DuckDB**
  preferred (analytical workload: filters, joins, aggregations; can query the CSVs
  in place with `SELECT ... FROM 'companies.csv'`). Use **SQLite** instead if the
  consuming app is browser-side or wants the most ubiquitous tooling.
- Do **not** store one-to-many data as JSON inside a cell. That's the current
  `funding_rounds`-as-a-stringified-blob pattern and it's the thing to remove.

## Design rules

- Every "a company can have more than one of X" becomes its own table with a
  `company_id` foreign key. That's `funding_rounds`, `failure_reasons`,
  `idea_dependencies`.
- Reassessments hang off the *dependency*, not the company, so the "now" axis has
  a history you can trend over time (`dependency_assessments`).
- Controlled-vocabulary fields are enforced in the pipeline via **Zod enums**
  (the real gate). The DB `CHECK` constraints below mirror them as documentation;
  relax them if the vocab churns.
- Types are real: years are `INTEGER`, amounts are `INTEGER`/`REAL`, dates are ISO
  `TEXT`. No numbers-as-strings.

## Schema (SQLite / DuckDB compatible DDL)

```sql
CREATE TABLE companies (
  id             INTEGER PRIMARY KEY,
  uuid           TEXT,                 -- carry the pipeline UUID for provenance
  company_name   TEXT NOT NULL,
  founders       TEXT,                 -- CSV of names; promote to a table only if queried
  sector         TEXT,                 -- controlled: see SECTOR vocab
  subsector      TEXT,                 -- optional free-text finer label
  location       TEXT,                 -- controlled: see REGION vocab
  country        TEXT,                 -- optional finer location
  year_founded   INTEGER,
  year_defunct   INTEGER,              -- null if still operating
  living_status  TEXT,                 -- controlled: see LIVING_STATUS vocab
  has_pivoted    INTEGER,              -- 0/1
  idea_summary   TEXT,                 -- one-sentence; the structured detail lives in idea_dependencies
  original_trl   INTEGER,              -- 1-9, TRL at the time (see TRL note)
  is_climate     INTEGER,              -- 0/1; filter non-climate rows out of the climate dataset
  source_url     TEXT,                 -- primary article
  created_at     TEXT
);

CREATE TABLE funding_rounds (
  id           INTEGER PRIMARY KEY,
  company_id   INTEGER NOT NULL REFERENCES companies(id),
  round_name   TEXT,                   -- controlled: see ROUND vocab
  amount       INTEGER,                -- keep amount and currency separate so amounts stay summable
  currency     TEXT,                   -- ISO code preferred (USD, EUR)
  round_date   TEXT,                   -- normalize to YYYY-MM
  round_year   INTEGER,               -- derived 4-digit year; avoids the 2-digit century bug
  source_url   TEXT
);

CREATE TABLE failure_reasons (
  id           INTEGER PRIMARY KEY,
  company_id   INTEGER NOT NULL REFERENCES companies(id),
  category     TEXT,                   -- controlled: see FAILURE vocab
  detail       TEXT,                   -- the specific, free-text story
  source_url   TEXT
);

CREATE TABLE idea_dependencies (
  id           INTEGER PRIMARY KEY,
  company_id   INTEGER NOT NULL REFERENCES companies(id),
  category     TEXT,                   -- controlled: see DEPENDENCY vocab
  detail       TEXT,                   -- the specific thing it needed
  criticality  TEXT,                   -- controlled: 'was_blocking' | 'contributing'
  source_url   TEXT
);

CREATE TABLE dependency_assessments (
  id            INTEGER PRIMARY KEY,
  dependency_id INTEGER NOT NULL REFERENCES idea_dependencies(id),
  assessed_on   TEXT,                  -- ISO date; enables "latest" + trend queries
  status        TEXT,                  -- controlled: 'improved' | 'unchanged' | 'worsened' | 'unknown'
  detail        TEXT,                  -- what changed
  metric_name   TEXT,                  -- e.g. 'battery pack price'
  metric_value  REAL,                  -- the number, so trajectory is computable
  metric_unit   TEXT,                  -- e.g. 'USD/kWh'
  current_trl   INTEGER,               -- 1-9, TRL as of this assessment
  source_url    TEXT
);

-- Optional. Only build this if you want to query BY source. Otherwise the
-- source_url columns above are enough. To adopt: swap source_url -> source_id FK.
CREATE TABLE sources (
  id             INTEGER PRIMARY KEY,
  url            TEXT,
  title          TEXT,
  publisher      TEXT,
  published_date TEXT,
  accessed_date  TEXT,
  type           TEXT                  -- 'article' | 'search' | 'database' | 'manual'
);
```

## Controlled vocabularies

The three vocab fields are aligned to the language the climate-tech community and
investors actually use, so findings are presentable without translation.

### SECTOR — high-level verticals (aligned to Sightline Climate / CTVC and PwC State of Climate Tech)
Consolidated from the original 14-item list into the standard verticals; keep the
finer original labels in `subsector` if wanted.
```
Energy
Mobility & Transportation
Industry & Manufacturing
Built Environment
Food, Agriculture & Land Use
Carbon & GHG Management        -- CDR, CCUS, carbon markets
Climate Intelligence & Finance -- MRV, data, climate fintech
Water & Oceans
Circular Economy & Waste
Adaptation & Resilience
```

### FAILURE — merged CB Insights post-mortem taxonomy + climate-hardtech causes
The first group is the widely-cited CB Insights vocabulary (post-mortem analysis of
why startups fail); the second group covers capital-intensive climate hardware,
which the software-centric list underweights. The project's original four
(technical feasibility, market timing, high cost, scalability) all map in.
```
No Market Need
Poor Product-Market Fit
Bad Timing (Ahead of Market)   -- the core Lazarus signal
Outcompeted
Flawed Business Model
Pricing / Unit Economics
Poor Product / Execution
Team
Legal / Regulatory
Ran Out of Capital
-- climate / hardtech specific --
Technical Feasibility
Scale-Up / Manufacturing       -- first-of-a-kind, the "valley of death"
Capital Intensity              -- couldn't finance the buildout
Policy / Subsidy Dependence    -- incentive changed or never arrived
Input / Commodity Cost
Infrastructure / Supply Chain
Other
```

### DEPENDENCY — what the idea leaned on (bottlenecks the community tracks)
```
Input / Commodity Cost         -- e.g. battery $/kWh, polysilicon, lithium
Enabling Technology            -- a prerequisite tech had to mature first
Infrastructure                 -- grid, charging, pipelines, ports
Policy & Incentives            -- carbon price, subsidy, mandate
Market Demand / Offtake
Manufacturing Scale-Up         -- FOAK cost curve
Capital Availability           -- financing structures, project finance
Supply Chain / Critical Materials
Talent / Expertise
Other
```

### ROUND — standard VC ladder + climate-relevant non-dilutive/project finance
```
Grant            -- non-dilutive; common early in climate hardware
Pre-Seed
Seed
Series A
Series B
Series C
Series D+
Growth / Late Stage
Debt / Project Finance
IPO / Public
Acquisition
Unknown
```

### REGION (unchanged from original)
```
North America | South America | Europe | Russia | Asia | Middle East | Other
```

### LIVING_STATUS (tightened, community terms)
```
Operating | Acquired | Zombie | Defunct | Unknown
```

## TRL note (the feasibility scale)

Use **Technology Readiness Level (1-9)** for `original_trl` and `current_trl` — the
standard scale DOE, ARPA-E, and the EU use (1 = basic principles, 9 = proven at
commercial scale). Storing both lets "is this more feasible now" be an actual
delta (`current_trl - original_trl`) rather than a vibe.

## Queries this unlocks

The whole point — these are one join or filter, not string-parsing:

```sql
-- The Lazarus shortlist: ideas whose BLOCKING dependency has since improved
SELECT c.company_name, c.sector, d.category, d.detail, a.status, a.detail
FROM companies c
JOIN idea_dependencies d      ON d.company_id = c.id
JOIN dependency_assessments a ON a.dependency_id = d.id
WHERE d.criticality = 'was_blocking'
  AND a.status = 'improved'
  AND a.assessed_on = (SELECT MAX(assessed_on)
                       FROM dependency_assessments
                       WHERE dependency_id = d.id);

-- Everything that died on cost or timing, by sector
SELECT c.sector, f.category, COUNT(*) 
FROM companies c JOIN failure_reasons f ON f.company_id = c.id
WHERE f.category IN ('Pricing / Unit Economics','Input / Commodity Cost','Bad Timing (Ahead of Market)')
GROUP BY c.sector, f.category;

-- Total funding raised per year (no century guessing)
SELECT round_year, SUM(amount) FROM funding_rounds GROUP BY round_year ORDER BY round_year;
```

## Pipeline integration

- **step1** writes `companies` (+ `failure_reasons`, and `idea_dependencies` if the
  idea is decomposed at extraction time). Split the current free-text
  `reason_for_demise` into `failure_reasons.category` (enum) + `.detail` (prose).
- **step2** writes `funding_rounds` — one row per round, keep undated rounds
  (the current code silently drops them), set `round_year`.
- **step3** stays deterministic: standardize/dedup `funding_rounds`, no century
  hardcode. It no longer needs to reconcile a JSON blob.
- **new "step4 / reassessment"** writes `dependency_assessments` from current data
  (search + cost sources); append-only, one row per evaluation date.
- **build step** loads the CSVs into the DuckDB/SQLite file.

## Open decisions

- Decompose the idea into `idea_dependencies` at extraction (step1), or as a
  separate pass? Separate pass is cleaner but costs another LLM call per company.
- `sources` table now, or defer until you want to query by source.
- Are `founders` ever queried? If yes, promote to a `founders` table (same pattern).
