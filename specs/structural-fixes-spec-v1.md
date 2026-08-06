# Project Lazarus — Structural Fixes Spec (v1)

The substrate the gap-typology view sits on. Four linked fixes that make the dataset
**inclusive and honest about gaps** — the build-side expression of
`foundational-principles.md` P2 (inclusion never gated on data) and P3 (absence is a
queryable state), plus the two correctness items already specced (basis enforcement,
tech/dependency split) that these principles now make load-bearing.

Governed by `foundational-principles.md`. Build order: **A → B → C → D** (each unblocks
the next; the gap view needs all four).

---

## Fix A — Inclusion gate: companies enter on factual footprint alone (P2)

**A company's presence in the dataset must never depend on metric, threshold, or
viability data.** The inclusion gate is the company/outcome pipeline, and its bar is
*low*.

- **Inclusion criteria (all a company needs):** a resolved identity + a basic factual
  footprint — what it did, roughly when (founded / died), sector/approach, and its
  dependencies *if known*. Nothing else.
- **Audit + fix:** trace the load path (`build-db` and upstream). Any point that drops or
  filters a company for *lack of* cost data, a missing threshold, a null viability, or an
  unmatched dependency is a **P2 violation** — change it to retain the company and record
  the gap (see Fix B). The 121 currently-held metric rows are a *metric*-side hold, not a
  company-side drop — confirm no company is lost the same way.
- **Test:** a company with a full factual footprint and **zero** curve/threshold data
  loads, is queryable, and appears in the company set. This test is the P2 guarantee.

## Fix B — Absence is a set of distinct, queryable states, never a null (P3)

"We don't know" is information — often the most useful, because it's where the user's own
judgment adds most. Every absence must be a **named, queryable state**, not a bare null a
query can't distinguish from "not yet loaded."

Minimum distinct states (per company × dependency, and per metric series):
| state | meaning |
|---|---|
| `no_blocker_data` | the dependency has no cost/curve series at all |
| `no_threshold` | series exists, but no viability bar is set |
| `undetermined` | data + bar exist but can't produce a crossing verdict (e.g. basis mismatch, too few points) |
| `assessed_not_viable` | assessed, blocker has **not** crossed its bar |
| `assessed_viable` | assessed, blocker **has** crossed |

- These extend the existing `metric_series_status` (full/anchor/current_only) and the
  `unsampled`-≠-`white_space` discipline to the **whole dataset**, not just metric depth.
- **Queryable as a first-class axis:** "show me companies whose blocker we have
  `no_blocker_data` on" must be as easy as "show me `assessed_viable`." Data-availability
  is a filter dimension (P5), not a footnote.
- **Rule:** never collapse two of these into one null. A null that could mean either
  "no data" or "determined not-viable" is a P3/P4 violation — the user can't tell a gap
  from a verdict.

## Fix C — `basis` three-way split + threshold basis-match enforcement

(Already specced in `metric-data-structure-spec-v1.md`; restated here because it's the
mechanism behind P4 — a viability call is only *inspectable/overridable* if its basis is
explicit, and only *trustworthy* if mismatched comparisons can't silently happen.)

- **Split** the one `basis` column into three orthogonal columns: `basis` (currency),
  `energy_basis` (nameplate/usable, AC/DC), `duration` (4h/2h/blended). Destructure the
  conflated `capacity_series` column (`AC`→`energy_basis`, `onshore`→`segment`,
  `energy`→`unit`). Touches `types.ts`, DDL, every extractor, curated CSVs.
- **Enforce (must land WITH the split):** add `basis`/`energy_basis`/`duration` to
  `dependency_thresholds`, AND add those predicates to the progress join. Until both, an
  observation can be compared to a bar of a different (or absent) basis — baseline rule
  #1 stays aspirational. A mismatch becomes a **build-time error**, not a silent wrong
  crossing.
- **P4 link:** a stored viability call carries its full basis, so the user sees *what it
  was judged against* and can disagree ("that 2019 bar is stale, I think it's viable")
  with the data supporting the disagreement.

## Fix D — Technology / dependency split (P2's enabling mechanism)

(Specced in `technology-dependency-split-spec-v1.md`; sequenced here.) This is not just a
data-model tidy — **it's what makes P2 expressible.**

- Metric data keys on `technology`; dependencies **join** to technologies via a generic
  `reference_entities` (kind=`technology` now) + `dependency_links` carrying the
  `(era, threshold)` slice. Unblocks the 121 held rows and the geothermal/hydro `receded`
  series.
- **The P2 payoff:** a dependency can exist as **description-only, no curve** (the
  standalone/qualitative case). A company blocked on "public trust in autonomous vehicles"
  or "supportive net-metering policy" enters the dataset with that dependency *named*,
  even though it will never have a learning curve. Without this split, "include companies
  whose blocker isn't measurable" is not representable — so P2 can't hold.

---

## Why this ordering

- **A first** — inclusion is the foundation; no point classifying a set that's silently
  incomplete.
- **B second** — once everything's included, the gaps must be nameable; B is what makes
  "included but no data" a real, queryable thing rather than a hole.
- **C third** — with data present, its comparisons must be basis-safe, or viability calls
  are untrustworthy (and un-inspectable).
- **D fourth** — unblocks the non-measurable and held dependencies, completing the set A
  guaranteed and B labeled.

After all four: the dataset is complete (A), honest about its gaps (B), trustworthy where
it does judge (C), and able to hold non-measurable blockers (D). **That's the substrate
the gap-typology view classifies** — see `gap-typology-spec-v1.md`.

## Cross-checks against principles
| Fix | Principle it satisfies |
|---|---|
| A | P2 — inclusion never gated on data |
| B | P3 — absence is a first-class queryable state |
| C | P4 — viability calls are tagged/inspectable/trustworthy |
| D | P2 — non-measurable blockers still get their company included |

## Amendment (2026-08-04, on build): the order is A → B1 → C → D → B2

**B has two grains and only one survives D.** The company × dependency grain is keyed
`(company_id, dependency_id)` and is unaffected by the split; the *series* grain is
re-keyed `dependency_id` → `entity_id` by D, and `qualitative_blocker` is *defined* by D
(link-absence). Writing that SQL against the old key and rewriting it a stage later is
wasted work in the most delicate views in the repo.

- **B1** (straight after A) builds the company × dependency grain — the one the user
  actually queries ("show me companies whose blocker we have no data on"), and where A's
  fix first becomes visible.
- **B2** (after D) re-keys the series grain onto `entity_id` and makes
  `qualitative_blocker` structural.

The rationale in "Why this ordering" is untouched: inclusion first, gaps nameable second,
comparisons basis-safe third, non-measurable blockers fourth. Only the *series* half of B
defers, for exactly the reason the spec gives for putting D last.

**Correction: the held rows are 120, not 121.** Geothermal LCOE carries 15 points, not 16
— 2011 is absent from the published IRENA series. Asserted in `metric-data.test.ts`.

## Resolved on build (2026-08-04)
- **P2 audit → one real defect, fixed.** No company was dropped anywhere for lack of
  metric/threshold/viability data. `config.processingLimit` is a volume cap, not a quality
  gate. The one violation was a level down: `build-db` discarded `company_dependencies`
  rows whose name matched nothing canonical, into an anonymous skip counter. They are now
  retained with a null `dependency_id`, their raw text, and `resolution_status`.
- **The five absence states → derived views, not stored columns.** Every input is
  recomputed each build, so a stored label would be a second source of truth that goes
  stale the moment `data/derived` is regenerated. Three views, ascending grain:
  `series_status` → `dependency_status` → `company_dependency_status`.
- **Basis reconciliation → REJECT, not reconcile.** `real_usd` (BNEF pack price, vintage
  unstated by the publisher) stays its own basis and compares equal only to another
  `real_usd`. Promoting it to a year would launder an assumption into a viability verdict.
  Enforced by an equality join plus a build-time throw (`lib/basis.ts`).

## Still open (carried)
- `search_coverage` is built but **seeded empty**, so every companyless region reads
  `unsampled` and `white_space` is currently unreachable outside tests. Filling it in is a
  curation task: what was actually swept, and from which source.
