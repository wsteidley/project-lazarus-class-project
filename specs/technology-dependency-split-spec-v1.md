# Project Lazarus — Technology / Dependency Split Spec (v1)

**Governed by `../foundational-principles.md`** — and this split is what *implements* P2:
a dependency can exist description-only, with no curve, so a company blocked on a
non-measurable thing ("public trust in AVs") still gets in, with its blocker named.

Corrects an entity conflation the loader exposed: metric data is keyed on
**technology**, but the loader resolves through `dependencies.name`, so any technology
without a dependency row (CSP, hydro, geothermal, bioenergy, offshore wind) drops all
its rows. This separates the two entities and links them generically.

Updates `metric-data-structure-spec-v1.md` (the metric side re-keys); pointer from
`hero-thresholds-spec-v3.3.md`.

---

## The distinction

- **Technology** — a data-bearing subject with cost/capacity curves (Solar PV,
  geothermal). How IRENA/OWID publish. Valid on its own, dependencies or not.
- **Dependency** — a thing a failed company leaned on (affordable storage, net-metering
  policy). Keyed to failure analysis.

They are **not one entity.** A dependency is often "a technology at a slice of time"
(era + threshold on a shared curve) — but only when a measurable curve exists behind
it. Some dependencies have no curve at all.

## Re-keying (the fix)

- **Metric observations, capacity, progress, trajectory, Wright — all key on
  `technology`, not `dependency`.** A technology with zero dependencies still loads and
  gets full derivations. Fixes the 80-dropped-rows bug and gives the trajectory
  empirical check its `receded` series (geothermal, hydro).
- The metric subsystem is **technology-keyed**; the failure subsystem is
  **dependency-keyed**; they **join**, they don't share a key.

## The link (modeled generically now, one kind built)

A dependency **optionally** links to a data-bearing subject. Model the link for the
*role*, not the type, so it generalizes without a future migration.

```sql
-- reference entities: data-bearing subjects a dependency can hang off.
CREATE TABLE reference_entities (
  id    INTEGER PRIMARY KEY,
  name  TEXT NOT NULL,          -- 'Solar PV', 'Onshore wind', (later) 'EV charging network'
  kind  TEXT NOT NULL           -- 'technology' now; 'infrastructure'|'market'|'policy' later
);

-- dependency -> subject, many-to-many, carrying the SLICE.
CREATE TABLE dependency_links (
  dependency_name  TEXT NOT NULL REFERENCES dependencies(name),
  entity_id        INTEGER NOT NULL REFERENCES reference_entities(id),
  era              TEXT,        -- when the company needed it (the time slice)
  threshold_value  REAL,        -- what value would have unblocked it
  threshold_unit   TEXT,
  PRIMARY KEY (dependency_name, entity_id)
);
```

- **Today only `kind='technology'` entities exist** and carry curves. The metric tables
  reference `reference_entities` of that kind (or keep `technology` as the key and treat
  `reference_entities` as the superset — implementer's call, but the *link* must be
  kind-agnostic).
- **Later** "EV charging network" enters as `kind='infrastructure'` with its own series
  and slots into the **same** link and the **same** progress/trajectory machinery — no
  schema change. That's the whole reason the link isn't named `technology_id`.

## Dependency classification

A dependency is one of:
- **Linked (curve-backed)** — points at a reference entity, carries `(era, threshold)`.
  "Affordable storage in 2010" = Solar/battery entity, 2010 era, its bar. Gets progress,
  trajectory, Wright (via the entity's curve).
- **Standalone (qualitative)** — no measurable subject ("favorable net-metering,"
  "public acceptance"). Description only, status-only, no curve, no fit. This is the
  existing `qualitative` bucket.

The `curve_type` / `threshold_kind` machinery already distinguishes these — it now also
decides *whether a dependency links at all.*

## Description

Every dependency has a free-text **`description`** ("what was actually needed"),
independent of linking. The link gives the curve; the description gives the meaning. A
standalone dependency has *only* the description.

## Many-to-one (why the split earns its place)

Many dependencies → one entity. Solar-blocked companies from 2008, 2012, 2016 are three
dependencies — same technology, three eras, three thresholds — sharing one solar curve.
If dependency *were* technology you couldn't represent "same curve, different failure
eras," which is core to the Lazarus thesis (the curve moved; when did each cohort die
relative to it?).

## Sequencing (relative to the three IRENA findings)

1. **Wind first** — 16 LCOE points replace 2, capacity already loaded, wind *has* a
   dependency row, not blocked on this split. Kills "0 non-solar projections" now.
2. **Battery reconcile** — pending user review of whether IRENA `battery_installed_cost`
   (140.45) == existing `4h system cost` (140, 150 bar). Confirm same basis/scope/
   duration before merging; keep distinct from BNEF pack price.
3. **This split** — the larger correction; unblocks the five orphan technologies and
   makes the model right. Do after wind so projections aren't blocked on the refactor.

## Scope now vs later

- **Build now:** the `technology` kind, the generic link, metric re-keying, description,
  standalone/qualitative handling.
- **Not now (door left open):** non-technology reference entities (infrastructure,
  market, policy) with their own plottable series — the "charging network" idea. The
  generic link is the only thing needed now to keep it a pure addition later.

## Still open
- Whether metric tables key directly on `technology` or on `reference_entities` filtered
  to `kind='technology'` — same result; pick per implementation simplicity.
- Whether `era` on the link is a year, a range, or ties to an idea-space failure cohort
  (the deeper "baseline from failure era" idea, still a query-time lens).
