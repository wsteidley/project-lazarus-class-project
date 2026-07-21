# Project Lazarus — Dataset Schema Spec (v2)

Extends v1 so the dataset holds **successes alongside failures** and can compare
them. Successes are the control group: a hurdle only counts as fatal if you can
see who hit it and survived. Same relational model — mostly reclassification, plus
one new grouping table.

## What's new in v2 (diff from v1)

1. **`failure_reasons` → `challenges`**, with a new `outcome` field
   (`fatal` / `overcome` / `pivoted_from` / `ongoing`). Same category vocabulary —
   a challenge is just as much *what a survivor overcame* as *what killed a failure*.
2. **Dataset no longer filters to defunct companies.** Successes and survivors are
   in-scope. `living_status` already carried the outcome axis; now we use all of it.
3. **`companies.outcome_type`** added — a judgment field so "Acquired" isn't
   auto-counted as a win (fire-sale vs. category winner are different outcomes).
4. **New `idea_spaces` table** — a shared tag so an early failure and a later
   success in the same space (e.g. early EV, residential solar financing) can be
   lined up head-to-head. `companies.idea_space_id` FK.
5. Everything else (funding_rounds, idea_dependencies, dependency_assessments,
   sources, TRL, storage decision) is unchanged from v1.

## Changed / new DDL

```sql
-- NEW: light grouping so failures and later successes in the same space link up.
CREATE TABLE idea_spaces (
  id           INTEGER PRIMARY KEY,
  name         TEXT NOT NULL,         -- e.g. 'Long-duration grid storage', 'Cellular agriculture'
  description  TEXT
);

-- CHANGED: companies gains idea_space_id + outcome_type; is_climate stays as a filter.
CREATE TABLE companies (
  id             INTEGER PRIMARY KEY,
  uuid           TEXT,
  company_name   TEXT NOT NULL,
  idea_space_id  INTEGER REFERENCES idea_spaces(id),   -- NEW
  founders       TEXT,
  sector         TEXT,                 -- SECTOR vocab
  subsector      TEXT,
  location       TEXT,                 -- REGION vocab
  country        TEXT,
  year_founded   INTEGER,
  year_defunct   INTEGER,
  living_status  TEXT,                 -- LIVING_STATUS vocab (raw state)
  outcome_type   TEXT,                 -- NEW: OUTCOME_TYPE vocab (the judgment)
  has_pivoted    INTEGER,              -- 0/1
  idea_summary   TEXT,
  original_trl   INTEGER,              -- 1-9
  is_climate     INTEGER,              -- 0/1
  source_url     TEXT,
  created_at     TEXT
);

-- RENAMED from failure_reasons, + outcome. Applies to survivors AND failures.
CREATE TABLE challenges (
  id           INTEGER PRIMARY KEY,
  company_id   INTEGER NOT NULL REFERENCES companies(id),
  category     TEXT,                   -- CHALLENGE vocab (same list as v1 FAILURE)
  outcome      TEXT,                   -- 'fatal' | 'overcome' | 'pivoted_from' | 'ongoing'
  detail       TEXT,
  source_url   TEXT
);
```

Unchanged tables from v1: `funding_rounds`, `idea_dependencies`,
`dependency_assessments`, `sources`. (See v1 spec for their DDL — nothing moved.)

## Vocabulary changes

### CHALLENGE (was FAILURE in v1 — identical list, reframed)
Same merged CB Insights + climate-hardtech categories. The `outcome` field now
carries whether it was fatal or cleared, so the category list itself is unchanged:
`No Market Need`, `Poor Product-Market Fit`, `Bad Timing (Ahead of Market)`,
`Outcompeted`, `Flawed Business Model`, `Pricing / Unit Economics`,
`Poor Product / Execution`, `Team`, `Legal / Regulatory`, `Ran Out of Capital`,
`Technical Feasibility`, `Scale-Up / Manufacturing`, `Capital Intensity`,
`Policy / Subsidy Dependence`, `Input / Commodity Cost`,
`Infrastructure / Supply Chain`, `Other`.

### CHALLENGE_OUTCOME (new)
```
fatal          -- the challenge killed the company
overcome       -- faced it and got past it
pivoted_from   -- avoided it by changing direction
ongoing        -- still facing it
```

### OUTCOME_TYPE (new — the success/failure judgment, distinct from living_status)
Keeps "success" from being a naive read of `living_status`:
```
Breakout Success       -- scaled, category winner
Solid Success          -- healthy, operating, not a breakout
Successful Exit        -- acquisition / IPO that was a genuine win
Soft Landing           -- acqui-hire, small or fire-sale exit
Struggling / Zombie
Failed                 -- defunct
Too Early to Tell
Unknown
```

Unchanged vocabs: SECTOR, DEPENDENCY, ROUND, REGION, LIVING_STATUS (see v1).

## Queries this newly unlocks

```sql
-- Same challenge, who survived vs. died: the core comparison
SELECT ch.category, c.outcome_type, COUNT(*)
FROM companies c JOIN challenges ch ON ch.company_id = c.id
GROUP BY ch.category, c.outcome_type
ORDER BY ch.category;

-- Head-to-head in one idea space: early failures vs. later successes
SELECT s.name AS idea_space, c.company_name, c.year_founded, c.outcome_type
FROM companies c JOIN idea_spaces s ON s.id = c.idea_space_id
WHERE s.name = 'Long-duration grid storage'
ORDER BY c.year_founded;

-- Challenges that survivors OVERCAME but that killed others — the revival signal
SELECT ch.category,
       SUM(ch.outcome = 'fatal')    AS killed,
       SUM(ch.outcome = 'overcome') AS overcome
FROM challenges ch
GROUP BY ch.category;
```

## Pipeline integration (changes)

- **step1** now writes `companies` for *all* outcomes (drop the defunct-only
  filter), sets `outcome_type`, assigns `idea_space_id`, and writes `challenges`
  with an `outcome` per row.
- **idea_space assignment** needs consistency — either an LLM clustering pass over
  extracted companies, or a curated seed list of spaces the extractor maps into.
  Curated-then-extend is safer than free clustering if you want clean head-to-heads.
- **Sampling / survivorship bias:** the scraper over-indexes launch and funding
  coverage (the optimism moment). To make success-vs-failure fair, deliberately
  pull survivors too, ideally within the same idea_spaces as the failures.
- Everything else (step2/step3 funding, reassessment step4, build step) unchanged.

## Open decisions

- `outcome_type` is a judgment call — decide whether the LLM assigns it or it's a
  derived rule from `living_status` + funding + acquisition signals.
- idea_space granularity: too broad ('Energy') and the head-to-head is meaningless;
  too narrow and nothing shares a space. Aim for the level where a failure and a
  later winner plausibly tried "the same thing."
- Whether to backfill v1-extracted rows or re-run extraction under v2.
