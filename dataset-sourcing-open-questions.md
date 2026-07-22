# Project Lazarus — Build Status and Open Questions

Running record of what is built and what each remaining item still needs. Updated after
v3 Tier 0/1, v4 Phases 0/1, and the static enrichment-data layer.

## Summary

**Built so far**

| Round | What landed |
| --- | --- |
| v3 Tier 0 | Canonical `dependencies` + `company_dependencies` (`step1c`), `dependency_assessments` + the `reassess` pass |
| v3 Tier 1 | `raw_documents` content-addressed cache with two-tier freshness; `confidence` / `contested` on judged rows |
| v4 Phase 0 | Tiered search backbone — Tavily primary, DuckDuckGo fallback, one shared `gatherSearchContext` |
| v4 Phase 1 | Company entity resolution (`resolve`), ID-first over a typed `company_urls` table |
| Enrichment data | Static enrichment sources normalized to a shared lookup schema + loader (`build-sources`); not yet wired into a step |

**Where each open item stands**

| # | Item | Status | Blocking |
| --- | --- | --- | --- |
| 7 | Retrospective outcome pass | Not built — **the binding constraint** | Trustworthy outcomes, and everything derived from them |
| 1 | Coverage + targeted discovery | Approach decided (v4), not built | True `white_space` vs `unsampled` |
| 4 | `confidence_score` | Design decided (v4), column null | Nothing yet; needed before confidence is trusted |
| 2 | Threshold values | Columns built, values mostly empty | `became_viable_date` (Tier 3) |
| 3 | `flared_out` | Decided (v4): trajectory-derived, view-only | Tier 2 typology |
| 5 | Fuzzy company matching | ID-first built; Splink deferred | Nothing — gated on real near-miss counts |
| 6 | Incremental scraping flag | Specced, not built | Multi-batch corpus building |
| 8 | Enrichment step (consume the data) | Data + loader built; step not wired | Cheaper, more specific enrichment before search |

**The one thing to decide next.** Item 7 is the binding constraint and v4 already says
so: the dependency axis is now evidenced, the outcome axis is still inferred from launch
articles. Items 2 and 3 refine a gap typology whose *inputs* remain unreliable until 7
lands, so building them first buys precision on top of weak data.

**The cheapest lever toward 7 is now item 8.** Static enrichment data is already
landed and normalized: for any company it matches, it supplies `living_status`,
funding, founding/defunct years, exit signals, and — from the failure set — a stated
failure reason mapped onto the `CHALLENGE` vocabulary. That is a deterministic,
no-API-cost partial answer to the outcome axis for matched companies, which both seeds
and corroborates the retrospective pass. Wiring the enrich step (item 8) plausibly comes
*before* the full search-based outcome pass, not after.

**Two measurements to take on the first real run**, both cheap and both decide a build:
- `SELECT url_type, COUNT(*) FROM company_urls GROUP BY url_type` — if identity URLs are
  rare, entity resolution is running on the weak `name_year` tier (item 5).
- The near-miss count printed by `npm run resolve` — the trigger for whether Splink is
  worth a two-language repo (item 5).

**Operational dependency added:** `TAVILY_API_KEY`. Search degrades silently to
DuckDuckGo when absent, so a missing secret costs recall rather than breaking a run. Exa
is deliberately not in this chain; v4 keeps it for Phase 5 discovery.

Each item below was **deliberately deferred**, not overlooked. Where v4 has since locked
a decision, the item records it — those need building, not researching.

---

## 1. Coverage and targeted discovery (blocks the whole gap typology)

**Status: dropped from the build.** v3 Tier 2 specifies `coverage(idea_space, era,
searched, source_count, confidence)`, where `searched` records that you looked.

**The problem.** Discovery is a TechCrunch tag listing driven by `SCRAPE_URL`
([step0-scrape.ts](src/steps/step0-scrape.ts)), and `idea_space_name` is assigned
*afterward* by the LLM in step1 from the curated list. You never search *for* an idea
space — you scrape a page and find out post hoc which spaces turned up. So `searched=1`
cannot be written truthfully, and without it `unsampled` collapses into `white_space`:
a hole in the data reads as a discovery.

This matters more than it sounds, because the whole "absence is signal" thesis rests
on telling a genuine gap from an unsampled one.

**v4 decided this — it needs building, not researching.** Two phases:
- **Now (v4 Phase 3):** ship the *honest proxy* — observed density (company/article
  count per idea_space × era) as the confidence signal, fixed era cutoffs
  (≤2013 / 2014–2020 / 2021+, first boundary on the cleantech 1.0 bust), and **label
  every empty space `unsampled`, never `white_space`**. Absence resolves to "unknown".
- **Later (v4 Phase 5):** Exa targeted discovery, queried per idea_space from its
  description, writes a real `coverage` row and promotes searched-and-empty spaces to
  a true `white_space`.

**Still open:** nothing blocking. The proxy holds the line until Exa lands, so no
false gap can appear in the meantime.

## 2. Threshold values (blocks `became_viable_date`, Tier 3)

**Status: columns built, values mostly empty.** `dependencies.threshold_metric/value/unit`
exist and `data/input/dependencies.csv` has plausible starting values for the obvious
ones (battery pack price, module cost, hydrogen, carbon price) and blanks elsewhere.

The derived payoff — ranking Lazarus candidates by *how overdue* they are
(`became_viable_date` vs. the year the last attempt died) — is only as trustworthy as
these numbers.

**v4 decided the structure.** A `threshold_kind` three-way split so a blank is never
ambiguous: `quantitative_with_threshold` (metric + value + unit + citation),
`quantitative_tbd` (measurable, number not yet set), `qualitative` (no scalar —
status-only). Only the first yields a `became_viable_date`. ~10 **hero dependencies**
get hand-curated cited thresholds and pull their current value from a **data feed**
(Our World in Data), removing the hallucination surface; the long tail rides the
LLM+search path. Any LLM-proposed number is stored flagged low-confidence, never as
curated. `became_viable` v1 is the earliest *observed* crossing; Wright's-Law
projection is explicitly v2.

**Still open:**
- The hero list itself, and a cited source per threshold value. A number without
  provenance quietly decides which ideas look "viable now."
- Is the threshold time-varying? v4 treats it as a fixed viability bar (the physics or
  economics of whether the idea works) with the *metric* moving — worth confirming that
  holds for cases where the bar itself has shifted.

## 3. `flared_out` (Tier 2 gap typology)

**Status: not built.** v3 lists it under Tier 2 as coming "from `outcome_type`", but
`OUTCOME_TYPE` ([src/schemas.ts](src/schemas.ts)) has no such value, and peak-then-collapse
is a *trajectory*, not a terminal label. You chose to derive it from trajectory instead.

**v4 decided placement.** It lives in the **gap-typology view only**, not as a column
beside `outcome_type` — it is a pattern over rounds plus death, not a terminal label,
and duplicating it invites drift. Definition: reached a late or large round, then
`year_defunct` within N years of the last round, **gated on `confidence`** because
uneven `funding_rounds` coverage makes a call on one known round a guess.

**Still open:** `N`, and what counts as "late or large".

## 4. `confidence_score` (deferred by design)

**Status: column built, always null.** `confidence` labels are self-reported by the LLM;
the numeric score is reserved for when confidence is *computed* from signals.

**v4 decided the machinery**, landing with the outcome pass (item 7):
- **Computed score is authoritative; the label is derived from it** by fixed cutoffs
  (≥0.8 high, 0.5–0.8 medium, 0.2–0.5 low, else unknown). The self-reported label keeps
  its **own separate column** — an unvalidated self-report is never blended into the
  computed score.
- **Signal tiers, cheapest first:** cross-source agreement (near-free from retrieval
  already happening, and it also sets `contested`); self-consistency sampling for
  uncorroborated enum judgments; semantic/numeric spread for free-text and metrics.
- **`unknown` is a defined terminal state** with a reason code — queryable, never blank,
  never a stop. No path waits for a human.

**Still open:**
- Whether self-reported confidence correlates with anything at all. v4's side-quest
  answers this self-supervised: score against ground truth the pipeline already trusts
  (Wayback/dead-domain death signals, seed-corpus facts) and **measure discrimination
  (AUROC), not calibration**. That test is what promotes or discards the self-reported
  label.

## 5. Company entity resolution — BUILT (ID-first); fuzzy matching still open

**Status: built in TypeScript, ID-first.** `npm run resolve`
([resolve-companies.ts](src/steps/resolve-companies.ts)) merges over the typed
`company_urls` table: `website` domain → `crunchbase` → `wikipedia` → normalized name +
founding year, using [entity-resolution.ts](src/lib/entity-resolution.ts). Identity
lives in a typed URL table rather than per-source columns, so adding LinkedIn, a
Wayback capture, or a dead-site link is a new row rather than a migration — and
`archive` URLs are already there for the outcome pass to use as a death signal.

**Two things this closed:**
- **The funding coupling is closed structurally, not patched.** `resolve` runs *before*
  step2, so funding rows are only ever generated against canonical uuids — there is no
  pre-merge uuid left at the `dedupeFundingRows` site to key on.
- **The Splink decision now has evidence instead of a guess.** Every run prints merges
  by key tier plus a **near-miss report**: distinct canonical companies sharing a
  normalized name that ID-first refused to merge.

**Still open — decide from the near-miss numbers:**
- Is probabilistic linkage (Splink on SQLite, per v4 Phase 1) worth the two-language
  repo? **The trigger is the near-miss count on real data.** If it stays near zero,
  ID-first was sufficient and the Python toolchain is avoidable; if it is large, the
  fuzzy tier is earning its keep. Note `uv` is not currently installed.
- **How often does extraction actually produce a `website` URL?** ID-first is only as
  good as its keys, and Crunchbase's `permalink`/`website_url` only reach the prompt
  when Crunchbase enrichment is on. If most companies have no identity URL, merges fall
  to the weak `name_year` tier — measure this on the first real run, e.g.
  `SELECT url_type, COUNT(*) FROM company_urls GROUP BY url_type`.
- What should queue for human review rather than auto-merge? Nothing queues today;
  every group merges automatically, with `merged_from` making it auditable after
  the fact.

## 6. Incremental scraping supplement flag (specced, not built)

**Status: not built.** Sourcing v1 specifies a step0 flag that copies the previous
latest scrape forward under a new timestamp and concatenates new rows, so year batches
accumulate instead of replacing each other.

`normalizeUrl` in [src/lib/raw-documents.ts](src/lib/raw-documents.ts) already exists
and is the function the dedupe-on-concatenate should use — it was written shared for
exactly this reason.

**v4 settled the shape:** `SCRAPE_APPEND=1` (matching the `SCRAPE_URL` style), dedupe on
concatenate by `url` through the shared `normalizeUrl`, and `_data_index.csv` regenerated
rather than merged.

**Still open — one empirical unknown:** how far back the current TechCrunch selectors
actually parse. Probe a ~2009 tag page. This rises the moment you commit to the
cleantech 1.0 window.

## 7. The retrospective outcome pass (v1's highest-leverage item, still unbuilt)

**Status: not built.** Everything in Tier 0/1 improves the *dependency* axis. The
*outcome* axis is still extracted from launch articles, which is the original
unreliability v1 identified: `living_status`, `challenges.outcome`, and the exit fields
describe a moment before the outcome existed.

The `raw_documents` cache now has an `outcome` `source_type` reserved and a TTL policy
waiting for it, so the plumbing is ready.

**v4 promoted it to the binding constraint and decided the approach:**
- **Seed, don't rehost.** Ingest failed-company *names + minimal lifecycle facts* from
  open compilations as a **to-research list**, never as finished data, with provenance
  recorded. Curated prose stays out.
- **Evidence every field from storable sources** — the company's own defunct site,
  Wayback captures, news fetched into `raw_documents` under the reserved `outcome`
  source_type. Corroborate with cheap deterministic death signals: dead domain, Wayback
  last-successful-capture date (which doubles as the death-date estimate the
  overdue-ness calculation needs), funding staleness.
- **Precedence is directional.** Evidenced outcome **wins**; `derive-outcome.ts` drops
  to fallback, filling only where the outcome pass returned nothing and **never
  overriding evidence**. When the heuristic fires it is recorded as low `confidence`
  with a note, so "guessed from funding staleness" stays queryable and distinct from
  "a 2021 obituary said so."

**Ready for it:** `raw_documents` already reserves the `outcome` source_type with a TTL,
`company_urls` already carries an `archive` type for Wayback captures, and the static
enrichment data (item 8) already supplies status / exit / funding / death-year for
matched companies with no API call — a deterministic head start the search pass then
extends and corroborates.

**Still open:** whether to take a licensed scale path if this becomes central.

## 8. Enrichment data — data + loader BUILT; consuming step not wired

**Status: landed, normalized, unused.** Static enrichment data now sits under
`data/sources/<name>/`, regenerated by `npm run build-sources`. Each source is
normalized to one shared lookup schema and read by `loadSource`
([src/lib/sources.ts](src/lib/sources.ts)). Nothing in the pipeline consults it yet —
that was a deliberate "land the data now, wire the step later" split.

**What the data gives.** For a matched company: `living_status`, `exit_type`
(acquisition / ipo / shutdown), total funding, funding-round count, founding and
defunct years, and — from the failure-oriented source — a free-text failure reason
already mapped onto the `CHALLENGE` vocabulary. Two join strengths:
- a **strong-join** source keyed by normalized **website domain** and an external id —
  deterministic, and keyed with the *same* `normalizeDomain` the resolver uses, so it
  lines up with `company_urls` directly;
- a **name-only** source (the failure set) — lower confidence, joined on the normalized
  name, but its failure-reason content is the closest match to the project thesis.

**Why it matters (see also item 7).** This is a no-API-cost, deterministic partial
answer to the outcome axis for every company it matches. Used *before* search it makes
the outcome/funding passes more specific and cheaper; the risk of using it *instead of*
search is leaving fresher facts on the table, so it should seed and gate search, not
replace it.

**Still open:**
- **Where the enrich step slots in.** Natural place is right after `resolve`: look up
  each canonical company by domain → external id → name, fill what the data has, mark
  what was filled, and let search fill only the gaps. To decide when the step is built.
- **Name-join confidence.** Domain/id matches are trustworthy; name-only matches to the
  failure set are not, and should land as lower `confidence` (and feed the same
  cross-source machinery as item 4), never as ground truth.
- **Precedence vs. the search-based outcome pass (item 7).** When enrichment and an
  evidenced outcome disagree, which wins? Likely: fresh evidence overrides a static
  snapshot, mirroring item 7's "evidence over heuristic" rule.
- **Coverage is unmeasured.** How many pipeline companies actually match the enrichment
  data (by domain vs. by name only) is unknown until the first real run — the same
  `company_urls` key-population question as item 5 decides how much this data can reach.
- **One source is still unconverted** — an Excel-format failure set is not yet
  normalized (no `.xlsx` reader wired), and an anonymized-name prediction set is set
  aside as unjoinable.
