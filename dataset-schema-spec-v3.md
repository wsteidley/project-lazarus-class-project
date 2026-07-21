# Project Lazarus — Dataset Schema Spec (v3)

Establishes the **sector → idea_space → company** hierarchy. Sector is the broad,
fixed vertical (10 canonical values, now spelled out as real rows). idea_space is
the specific pursuit within a sector. A company can span several sectors but
belongs to one idea_space.

## What's new in v3 (diff from v2)

1. **Sectors are now reference rows, not a text column.** New `sectors` lookup
   table seeded with the 10 canonical verticals — the vocab is "spelled out" as
   data you can join and query, not just a Zod enum.
2. **A company can have several sectors.** New `company_sectors` join table
   (many-to-many), with an optional `is_primary` flag. Handles genuinely
   cross-cutting cases (e.g. agrivoltaics = Energy + Food/Ag).
3. **idea_space is narrower than sector and sits inside one.** `idea_spaces` gains
   `sector_id` — its home vertical. This is the specific layer that makes
   head-to-heads meaningful.
4. **`companies` loses `sector` and `subsector`.** Broad classification moves to
   `company_sectors`; the specific layer is `idea_space`. `subsector` is retired —
   idea_space replaces it.
5. Everything else (funding_rounds, challenges, idea_dependencies,
   dependency_assessments, sources, TRL, storage) is unchanged from v2.

The hierarchy: `sectors` (10 broad, fixed) → `idea_spaces` (specific, curated,
each in one sector) → `companies` (one idea_space; tagged to 1+ sectors).

## Changed / new DDL

```sql
-- NEW: the fixed climate-tech verticals, spelled out as rows (seed data).
CREATE TABLE sectors (
  id    INTEGER PRIMARY KEY,
  name  TEXT NOT NULL UNIQUE
);
-- seed rows: the 10 SECTOR vocab values (see below)

-- NEW: many-to-many. A company can sit in several sectors.
CREATE TABLE company_sectors (
  company_id  INTEGER NOT NULL REFERENCES companies(id),
  sector_id   INTEGER NOT NULL REFERENCES sectors(id),
  is_primary  INTEGER,            -- 0/1, optional: flags the main vertical
  PRIMARY KEY (company_id, sector_id)
);

-- CHANGED: idea_space now nests under a sector (its home vertical).
CREATE TABLE idea_spaces (
  id          INTEGER PRIMARY KEY,
  name        TEXT NOT NULL,      -- specific: 'Long-duration grid storage'
  sector_id   INTEGER REFERENCES sectors(id),   -- NEW: broad vertical it sits in
  description TEXT
);

-- CHANGED: companies drops sector + subsector (moved to company_sectors / idea_space).
CREATE TABLE companies (
  id             INTEGER PRIMARY KEY,
  uuid           TEXT,
  company_name   TEXT NOT NULL,
  idea_space_id  INTEGER REFERENCES idea_spaces(id),
  founders       TEXT,
  location       TEXT,            -- REGION vocab
  country        TEXT,
  year_founded   INTEGER,
  year_defunct   INTEGER,
  living_status  TEXT,            -- LIVING_STATUS vocab
  outcome_type   TEXT,            -- OUTCOME_TYPE vocab
  has_pivoted    INTEGER,         -- 0/1
  idea_summary   TEXT,
  original_trl   INTEGER,         -- 1-9
  is_climate     INTEGER,         -- 0/1
  source_url     TEXT,
  created_at     TEXT
);
```

Unchanged tables from v2: `funding_rounds`, `challenges`, `idea_dependencies`,
`dependency_assessments`, `sources`.

## SECTOR seed rows (the fixed vocab, spelled out)

Insert these as the `sectors` table rows. Aligned to Sightline Climate / CTVC and
PwC State of Climate Tech verticals:
```
Energy
Mobility & Transportation
Industry & Manufacturing
Built Environment
Food, Agriculture & Land Use
Carbon & GHG Management
Climate Intelligence & Finance
Water & Oceans
Circular Economy & Waste
Adaptation & Resilience
```

## sector vs. idea_space — the distinction

- **sector** = broad, fixed, ~10 values, community-standard. Answers "what vertical."
  A company can have several. Example: `Energy`.
- **idea_space** = specific, curated, grows over time. Answers "what were they
  actually trying to do." A company has one. Example: `Long-duration grid storage`,
  `Residential solar financing`, `Cellular agriculture`.
- idea_space has one home sector (`sector_id`). A company's own `company_sectors`
  tags may be broader than its idea_space's home sector — that's expected and fine;
  the join carries the cross-cutting breadth, the idea_space carries the focus.

## Queries affected

Sector filters now join through `company_sectors`:
```sql
-- Died on cost in the Energy vertical
SELECT c.company_name, ch.category
FROM companies c
JOIN company_sectors cs ON cs.company_id = c.id
JOIN sectors s          ON s.id = cs.sector_id
JOIN challenges ch      ON ch.company_id = c.id
WHERE s.name = 'Energy'
  AND ch.outcome = 'fatal'
  AND ch.category IN ('Pricing / Unit Economics','Input / Commodity Cost');

-- Idea spaces within a sector, with their survival record
SELECT s.name AS sector, i.name AS idea_space,
       SUM(c.outcome_type LIKE '%Success%') AS successes,
       SUM(c.outcome_type = 'Failed')       AS failures
FROM idea_spaces i
JOIN sectors s   ON s.id = i.sector_id
JOIN companies c ON c.idea_space_id = i.id
GROUP BY s.name, i.name;
```

## Pipeline integration (changes)

- **step1** assigns `company_sectors` (one or more, optionally flag `is_primary`)
  and `idea_space_id`. It no longer writes a single `sector` column.
- **sectors** table is seeded once from the vocab above — reference data, not
  extracted.
- **idea_space assignment** unchanged from v2 (curated-then-extend recommended);
  now also set each idea_space's `sector_id` when it's created.
- Everything else unchanged.

## Open decisions

- Can an idea_space belong to more than one sector (agrivoltaics)? v3 keeps a
  single home `sector_id` for clean rollups and pushes cross-cutting breadth to
  `company_sectors`. Revisit only if single-home proves too lossy.
- Is `is_primary` on `company_sectors` worth populating, or is an unordered set of
  sectors enough for your queries?
- Seed the idea_spaces from a curated list up front, or let step1 propose new ones
  and review them? Proposed-then-reviewed keeps the specific layer from fragmenting.
