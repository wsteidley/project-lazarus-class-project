# Project Lazarus — Foundational Principles

The doc that sits **above** all the specs. Every other spec gets checked against this.
When a detailed decision conflicts with a principle here, the principle wins — or the
principle gets a deliberate, recorded amendment. These are the commitments that are easy
to violate one small "let's just require X" at a time, so they're written down to be
defended.

---

## The goal, in one line

**A data playground for exploration and discovery** — a searchable, joinable substrate
of climate-tech failures and successes that helps a person surface ideas whose blocker
may have become viable, *while leaving the final judgment to the person.*

The tool **surfaces candidates and shows its work.** It does not hand down verdicts.

---

## Principle 1 — The dataset is a research substrate, not a verdict engine

Viability determinations happen *with* this data, but they are **guesses with a stated
basis**, never facts. The dataset's job is to make the space navigable and to be honest
about what it does and doesn't know — not to declare winners.

- A "viable now" / "not viable" call is always an **overlay**, never the source of truth.
- The most valuable case is often the one the tool *can't* judge — where the user knows
  something the data doesn't (a blocker that quietly got commoditized). The tool must
  make that case easy to find, not delete it.

## Principle 2 — Inclusion is never gated on viability data

A company enters the dataset on a **basic factual footprint alone**: what it did, roughly
when, its sector/approach, and — where known — what it depended on. That's the bar.

**Explicitly does NOT affect inclusion:**
- whether cost/curve data exists for its blocker,
- whether a viability threshold has been set,
- whether the guess is "viable," "not viable," or "undetermined."

The failure mode this prevents: rich learning-curve data for a few technologies
(solar/wind/battery) quietly narrowing a *company* dataset into a
*technologies-we-have-curves-for* dataset. That would delete exactly the Lazarus
candidates the project exists to surface. **Data richness is an overlay on a complete
company set, never a filter that shrinks it.**

## Principle 3 — Absence is a first-class, queryable state

"We have no data on this" is information, and often the *most useful* information — it's
where the user's own judgment adds the most. Missing data is **never a silent null.**

Distinct, queryable states (never collapsed into one blank):
- no cost/curve data for the dependency,
- no threshold set,
- viability undetermined,
- assessed → not viable,
- assessed → viable.

A user must be able to ask *"show me companies whose blocker we have no data on"* as
easily as *"show me the crossed ones."* The map of what we don't know is as queryable as
the map of what we do. (This is the coverage/`metric_series_status`/`unsampled`-≠-
`white_space` discipline, generalized to the whole dataset.)

## Principle 4 — Every viability call is tagged and inspectable

A viability guess must always carry its reasoning: which threshold, which cost series,
which basis (currency / energy_basis / duration), which date, which assumptions. The user
sees *the argument*, not a verdict.

- This is *why* the basis-split + threshold-enforcement work matters beyond correctness:
  it makes each guess **inspectable and overridable.**
- A user must be able to look at a "not viable" call, see it was judged against (say) a
  2019 threshold in nameplate-kWh, disagree ("that bar's outdated"), and have the data
  *support* that disagreement rather than obscure it.

## Principle 5 — Optimize the query layer for join / filter / bring-your-own-judgment

The product's job is to make the substrate **searchable and combinable**, not to deliver
answers:
- filter by sector, approach, era, dependency, and **data-availability**;
- join a company to whatever curve/threshold data exists — or clearly show none does;
- let the user layer their own knowledge onto the gaps.

The gap-typology view is therefore **not** "the thing that labels winners." It's the
thing that makes the whole space navigable — *including the unlabeled regions.*

---

## What each principle implies for the build (cross-checks)

| Principle | Consequence for existing specs |
|---|---|
| 1 — substrate not verdict | Viability flags are derived overlays; never gate storage or inclusion on them. |
| 2 — inclusion ungated | The **company/outcome pipeline** is the inclusion gate, and its bar is *low* (factual footprint). Metric data attaches *if available*. |
| 2 + tech/dependency split | A dependency must be able to exist as **description-only, no curve** (standalone/qualitative) — a company blocked on "public trust in AVs" belongs in, with that dependency named, though it'll never have a Wright fit. The split is what enables this. |
| 3 — absence queryable | Every "missing" is a labeled state, not a null. Extends `metric_series_status`, coverage, and the `unsampled` vs `white_space` distinction to the whole dataset. |
| 4 — tagged guesses | Basis-split + threshold-enforcement is what makes guesses inspectable; a viability call without its basis is a bug. |
| 5 — playground query layer | The app/query layer prioritizes filter/join/BYO-data and data-availability as a first-class filter dimension. |

## How to use this doc

- New spec or feature → check it against these five. If it narrows what's included,
  hides an absence, or hands down an untagged verdict, it's likely violating a principle.
- A principle can be amended, but only **deliberately and in writing** here — never eroded
  by an incidental "let's just require X" in a downstream spec.
