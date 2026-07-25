# Project Lazarus — Dataset Schema Spec (v4)

Makes `outcome_type` a **derived** field instead of a model guess. It's computed
from four inputs: `living_status`, total funding raised, exit/acquisition signal,
and an outcome narrative (what happened, when known). Adds the fields needed to
feed that derivation.

## What's new in v4 (diff from v3)

1. **`outcome_type` is derived, not extracted.** Populated by a dedicated
   derivation step from the signals below — never hand-set at step1. Recomputed
   when its inputs change. Same OUTCOME_TYPE vocab.
2. **Exit signal captured separately from funding.** New exit fields on
   `companies` (`exit_type`, `exit_amount`, `exit_date`, `exit_notes`). Kept out of
   `funding_rounds` on purpose so "total raised" stays clean — a $10M acquisition
   is not $10M raised.
3. **Outcome narrative added.** New `outcome_summary` (nullable — "if it exists")
   plus `outcome_source_url`. This is *what became of them*, distinct from
   `idea_summary` (*what they did*). It's the qualitative input that disambiguates
   the gray-zone cases.
4. **Optional `outcome_rationale`** stores why the derivation landed where it did,
   so a derived label is auditable and reproducible.
5. Everything else (all tables and relationships from v3) is structurally
   unchanged — these are new columns on `companies` only.

## Changed DDL (companies)

```sql
CREATE TABLE companies (
  id                INTEGER PRIMARY KEY,
  uuid              TEXT,
  company_name      TEXT NOT NULL,
  idea_space_id     INTEGER REFERENCES idea_spaces(id),
  founders          TEXT,
  location          TEXT,          -- REGION vocab
  country           TEXT,
  year_founded      INTEGER,
  year_defunct      INTEGER,
  living_status     TEXT,          -- LIVING_STATUS vocab   [derivation input]
  has_pivoted       INTEGER,       -- 0/1
  idea_summary      TEXT,          -- what they did

  -- exit signal (separate from funding_rounds)          [derivation input]
  exit_type         TEXT,          -- 'acquisition' | 'ipo' | 'shutdown' | 'none' | 'unknown'
  exit_amount       INTEGER,       -- value of the exit event, nullable
  exit_date         TEXT,          -- ISO
  exit_notes        TEXT,          -- acquirer, terms, context

  -- outcome narrative                                    [derivation input]
  outcome_summary    TEXT,         -- what became of them, if known
  outcome_source_url TEXT,

  -- derived
  outcome_type      TEXT,          -- OUTCOME_TYPE vocab — DERIVED, do not hand-set
  outcome_rationale TEXT,          -- optional: why the derivation chose it

  original_trl      INTEGER,       -- 1-9
  is_climate        INTEGER,       -- 0/1
  source_url        TEXT,
  created_at        TEXT
);
```

`total_raised` stays a computed sum over `funding_rounds` (not a stored column),
and is the funding input to the derivation.

## Derivation: signals → outcome_type

Deterministic rules cover the clear cases; the LLM adjudicates only the gray zone,
using `outcome_summary`. `total_raised = SUM(funding_rounds.amount)`;
`company_age` measured against the dataset build date.

```
living_status = Defunct
    and exit_type = acquisition and exit_amount >= total_raised   -> Successful Exit
    and exit_type = acquisition and exit_amount <  total_raised   -> Soft Landing
    else                                                          -> Failed

living_status = Acquired
    and exit_amount >= ~1x total_raised (or narrative = strategic win) -> Successful Exit
    else                                                               -> Soft Landing

living_status = Operating
    and strong signal (recent large raise / growth stage / durable)    -> Breakout Success
    and healthy but not breakout                                       -> Solid Success
    and stalled (no raise in ~3+ yrs, early-stage)                     -> Struggling / Zombie
    and company_age < ~3-4 yrs and little signal                       -> Too Early to Tell

living_status = Zombie                                                 -> Struggling / Zombie
missing / conflicting signals                                          -> Unknown
```

Rules:
- The narrative can override a numeric read — "acquired in a fire sale after running
  out of cash" is a Soft Landing even if `exit_amount` is undisclosed.
- Where `exit_amount` is null and the narrative is silent, prefer the more
  conservative label (Soft Landing over Successful Exit) and record it in
  `outcome_rationale`.
- The LLM step is a constrained classifier over the assembled evidence, not a free
  judgment: it must return a value from the OUTCOME_TYPE vocab plus a one-line
  rationale.

## OUTCOME_TYPE vocab (unchanged from v2)
```
Breakout Success | Solid Success | Successful Exit | Soft Landing
Struggling / Zombie | Failed | Too Early to Tell | Unknown
```

## Pipeline integration

- **step1** extracts the raw inputs only: `living_status`, `exit_*`,
  `outcome_summary` (+ `outcome_source_url`). It does **not** set `outcome_type`.
- **new derivation step** (after funding is populated) computes `outcome_type` and
  `outcome_rationale` from living_status + total_raised + exit_* + outcome_summary,
  using the rules above with LLM adjudication for the gray zone. Re-runnable.
- Order matters: `funding_rounds` (step2/step3) must be populated before derivation,
  since `total_raised` is an input.
- Everything else unchanged.

## Open decisions

- Store `outcome_type` as a column (materialized, recomputed) or expose it as a SQL
  view over the inputs? Materialized is simpler to query and lets you keep a
  rationale; a view can't drift but re-derives on every read. Materialized
  recommended given the LLM-adjudicated cases.
- Keep exit as columns (one terminal event per company) or promote to an
  `exit_events` table if you want full liquidity-event history? Columns for now.
- Thresholds above ("~1x raised", "~3-4 yrs") need concrete values — set them once
  and keep them in the derivation step so the label is reproducible.
