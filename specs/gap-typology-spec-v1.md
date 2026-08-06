# Project Lazarus — Gap Typology Spec (v1)

The MVP payoff: the view that makes the whole space **navigable**. Not a scoring engine
that labels winners — a **map, including its blank regions**, where the unknown and
no-data areas are as first-class as the classified ones.

Governed by `foundational-principles.md` (P1 substrate-not-verdict, P3 absence-queryable,
P5 playground query layer). Sits on top of the `structural-fixes-spec-v1.md` substrate
(A inclusion / B absence states / C basis / D tech-dependency) — build after those.

---

## What this is (and isn't)

- **Is:** a derived classification over the *already-complete, already-inclusive* company
  set — an annotation layer that tells a user what kind of region each company/dependency
  sits in, so they can navigate, filter, and bring their own judgment.
- **Is NOT:** a gate, a filter, or a verdict. It never removes a company. It never claims
  a viability *fact* — every viability-derived cell is a **tagged guess** (P1/P4) the user
  can inspect and override.

The reframe that matters: this view leads with **"here's what we can and can't say,"** not
with "here are the winners." The blank cells are the point — they're where the user's
outside knowledge does the work the data can't.

## The typology cells

A company-dependency pairing (or a whole company, aggregated) lands in one cell. The cells
are **not** all "verdicts" — several are honest statements of absence.

### Classified (data + bar exist)
| cell | meaning | Lazarus signal |
|---|---|---|
| `lazarus_candidate` | blocker crossed its viability bar *after* the company died | **strong** — idea may be viable now |
| `tested_dead_end` | blocker never crossed; still hasn't | weak — the thing it needed still isn't there |
| `receded` | blocker moved *away* from viability (got worse) | negative — harder now than then |
| `conditional` | crosses only under a policy/support regime, not on economics alone | contingent — viable *if* the regime exists |
| `crossed_and_receded` | crossed, then backslid | mixed — window opened then narrowed |
| `still_viable` (control) | a *success* whose blocker was already viable | baseline / control group |

### Unclassified (the first-class blank regions — P3)
| cell | meaning | why it's first-class |
|---|---|---|
| `no_blocker_data` | company + dependency known, but no cost/curve series exists | **the highest-value gap** — user may know the blocker got commoditized |
| `unassessed` | series exists but no viability bar set yet | a curation gap, not a dead end |
| `undetermined` | data + bar exist but can't produce a verdict (basis mismatch, too few points) | honest "we can't say" |
| `qualitative_blocker` | blocker is non-measurable ("public trust," "regulation") — will never have a curve | belongs in, per P2 |
| `unsampled` | region not yet searched for companies at all | **never** silently shown as `white_space` |

### Space
| cell | meaning |
|---|---|
| `white_space` | genuinely searched, genuinely empty — no company tried this (idea × dependency) combination |

**Critical distinction (carried from earlier work):** `unsampled` (we haven't looked) and
`white_space` (we looked, nothing's there) must never collapse. An empty region is only
`white_space` if coverage confirms it was actually searched — otherwise it's `unsampled`.
Mislabeling absence-of-search as absence-of-companies would invent opportunities that were
never checked.

## Derivation

- A **derived view** over the substrate (company set + `dependency_links` + metric
  overlay + absence states from Fix B), recomputed on build. Same tier as
  `progress`/`trajectory` — not a stored gate.
- Cell assignment order (first match wins), roughly:
  1. non-measurable dependency → `qualitative_blocker`
  2. no series → `no_blocker_data`
  3. no bar → `unassessed`
  4. can't compute → `undetermined`
  5. else compute crossing vs. death year → `lazarus_candidate` / `tested_dead_end` /
     `receded` / `conditional` / `crossed_and_receded` / `still_viable`
- **Every classified cell carries its basis** (which threshold, series, basis-triple,
  date, assumptions) so it's inspectable/overridable (P4). A cell without its reasoning
  attached is a bug.
- **Aggregation:** a company with multiple dependencies has a cell *per dependency*; a
  company-level roll-up takes the strongest signal but must **not** hide the others
  (a company that's `lazarus_candidate` on one blocker and `no_blocker_data` on another is
  both — surface both).

## Query surface (P5 — the playground)

The view exists to be filtered, joined, and combined with the user's own judgment:
- Filter by **cell**, including the blanks — "show me `no_blocker_data`" is a headline
  query, not an edge case.
- Filter by **sector, approach, era, dependency, outcome** — orthogonal to cell.
- **Data-availability as a first-class filter** — "companies I could reassess with my own
  data" = the `no_blocker_data` + `unassessed` + `qualitative_blocker` union.
- Join a company → its dependencies → whatever curve/threshold data exists (or the labeled
  absence). The user layers knowledge onto the gaps; the tool never pretends the gap isn't
  there.

## What this unblocks / needs

- **Needs:** all four structural fixes (A–D). In particular `conditional` needs the
  `policy_dependent` flag (built), `qualitative_blocker` needs the tech/dependency split
  (D), and the absence cells need Fix B's states.
- **Unblocks:** the app/exploration layer (still parked) — the signature "progress curve
  with company death-years overlaid" plot reads directly off this view + the metric
  series.

## Resolved on build (2026-08-04)

- **`crossed_and_receded` stays a distinct cell.** It falls out of the SQL for free
  (`first_viable_date IS NOT NULL AND any_crossed = 0`) and it is a genuinely different
  answer from `lazarus_candidate`: the window opened *and closed*. Collapsing it into a
  flag would make the headline filter lie — a user filtering for revivable ideas would be
  handed one that is no longer viable.
- **`white_space` requires an explicit coverage record.** A `search_coverage` table
  (`idea_space_name`, `dependency_name` nullable for a whole-space sweep, `searched_on`,
  `source`, `method`) is built and loaded. `unsampled` is the DEFAULT; a cell becomes
  `white_space` only where a row confirms someone looked. **Seeded empty today**, so every
  companyless region currently reads `unsampled` — the honest answer, and the one the
  `absence is signal` principle asks for.
- **`era` is carried, not used.** `lazarus_candidate` compares `became_viable_date` to
  `companies.year_defunct` — a fact about the company rather than a curated slice. The
  era rides on the cell for inspection; revisit when real multi-era dependencies exist.
- **Roll-up is strongest-signal + a count per state.** `company_gap_summary` carries both,
  so "must not hide the others" is structural rather than a convention. The ranking puts
  `no_blocker_data` ABOVE the weak classified cells: the spec calls it the highest-value
  gap, so a company with one dead end and one unknown surfaces as the unknown.
- **Cell grain is company × dependency**, with one representative series chosen per
  dependency (`dependency_signal`) — crossed first, then earliest crossing, i.e. the
  reading most favourable to the Lazarus finding. Per-series detail stays in `trajectory`.

## Still open
- Seeding `search_coverage` honestly — what the corpus actually swept, and from where.
  Until then `white_space` is unreachable outside tests.
- Company-level roll-up may want a richer summary once real multi-dependency companies are
  loaded; strongest-signal-plus-counts is the starting point.
