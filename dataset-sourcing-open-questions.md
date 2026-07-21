# Project Lazarus — Open Questions After Tier 0/1

Written after building Tier 0 (canonical dependencies + `dependency_assessments` +
the reassess pass) and Tier 1 (`raw_documents` cache + `confidence`/`contested`) from
`dataset-sourcing-spec-v3.md`.

Each item below is something the build **deliberately deferred**, not something
overlooked. Ordered by how much it blocks.

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

**To research:**
- Is a per-idea_space discovery mode worth building — issuing queries derived from the
  `idea_spaces` seed, so "searched this space, found nothing" becomes a real record?
  What source would it query, given DuckDuckGo's rate limits?
- If not: is *observed density* (article/company count per idea_space × era, derived at
  build time) an honest enough confidence proxy? It can't distinguish "looked and found
  nothing" from "never looked" — is that acceptable, and how should the gap view label
  the difference so nobody over-reads it?
- What era boundaries? Fixed cutoffs (≤2013 / 2014-2020 / 2021+) are simpler than
  per-idea_space ones and probably sufficient — worth confirming against the cleantech
  1.0 boom-bust dating.

## 2. Threshold values (blocks `became_viable_date`, Tier 3)

**Status: columns built, values mostly empty.** `dependencies.threshold_metric/value/unit`
exist and `data/input/dependencies.csv` has plausible starting values for the obvious
ones (battery pack price, module cost, hydrogen, carbon price) and blanks elsewhere.

The derived payoff — ranking Lazarus candidates by *how overdue* they are
(`became_viable_date` vs. the year the last attempt died) — is only as trustworthy as
these numbers.

**To research:**
- Where does each threshold come from, with a citation? A number without provenance
  will quietly decide which ideas look "viable now."
- Several dependencies have no single scalar threshold at all ("Contract manufacturing
  capacity", "Specialist engineering talent"). Do those get a qualitative
  `status`-only assessment, or should the canonical list be split so every row either
  has a metric or is explicitly marked unmeasurable?
- Should thresholds be time-varying? "$100/kWh" is the bar today; the bar a 2008
  company needed to clear may be different from the bar that makes the idea work now.

## 3. `flared_out` (Tier 2 gap typology)

**Status: not built.** v3 lists it under Tier 2 as coming "from `outcome_type`", but
`OUTCOME_TYPE` ([src/schemas.ts](src/schemas.ts)) has no such value, and peak-then-collapse
is a *trajectory*, not a terminal label. You chose to derive it from trajectory instead.

**To research:**
- What's the operational definition? Candidate: raised above some threshold, or reached
  a late round, then `year_defunct` within N years of the last round. Both numbers are
  judgment calls, and `funding_rounds` coverage is uneven.
- Does it belong as a derived column next to `outcome_type` (mirroring
  [derive-outcome.ts](src/lib/derive-outcome.ts)), or only inside the gap-typology view?

## 4. `confidence_score` (deferred by design)

**Status: column built, always null.** `confidence` labels are self-reported by the LLM;
the numeric score is reserved for when confidence is *computed* from signals.

**To research:**
- Which signals — corroborating source count, source authority ranking, cross-source
  agreement? The reassess pass already stores `source_url` + `snippet` per assessment,
  which is the raw material.
- What cutoffs map a 0-1 score onto the four labels, and does a computed score override
  a self-reported label or sit beside it?
- Worth checking empirically whether LLM self-reported confidence correlates with
  anything at all before investing in the computed version.

## 5. Company entity resolution (still unbuilt)

**Status: not built; the principle is now applied to dependencies but not companies.**
Tier 0 canonicalized *dependencies*; companies are still one-article-one-row.

Sourcing v1 established the rule — merge before enrichment, never after — and the
ordering: resolve before the outcome pass, so one retrospective search runs per
canonical company rather than per duplicate.

**To research:**
- Canonical key definition: normalized name + website domain, with founders/founding-year
  as tiebreakers. What match score auto-merges, and what queues for review?
- **Note a live coupling:** `dedupeFundingRows` keys on `(company_uuid, round_name)`
  ([src/lib/funding.ts](src/lib/funding.ts)). Those uuids must become post-merge
  canonical ones, or funding dedupe silently under-merges once companies are merged.

## 6. Incremental scraping supplement flag (specced, not built)

**Status: not built.** Sourcing v1 specifies a step0 flag that copies the previous
latest scrape forward under a new timestamp and concatenates new rows, so year batches
accumulate instead of replacing each other.

`normalizeUrl` in [src/lib/raw-documents.ts](src/lib/raw-documents.ts) already exists
and is the function the dedupe-on-concatenate should use — it was written shared for
exactly this reason.

**To research:**
- Flag shape: an env var (`SCRAPE_APPEND=1`) matches the existing `SCRAPE_URL` style.
- How far back do TechCrunch archives stay parseable with the current selectors? This is
  the one genuinely open scoping question, now that year range is a batching parameter
  rather than a filter.
- Does the `_data_index.csv` companion need the same treatment, or can it be regenerated?

## 7. The retrospective outcome pass (v1's highest-leverage item, still unbuilt)

**Status: not built.** Everything in Tier 0/1 improves the *dependency* axis. The
*outcome* axis is still extracted from launch articles, which is the original
unreliability v1 identified: `living_status`, `challenges.outcome`, and the exit fields
describe a moment before the outcome existed.

The `raw_documents` cache now has an `outcome` `source_type` reserved and a TTL policy
waiting for it, so the plumbing is ready.

**To research:**
- This arguably outranks all of Tier 2/3 for data quality. Worth deciding explicitly
  whether it jumps the queue.
- How does it interact with `derive`? Once outcomes are evidenced rather than inferred,
  the age/funding heuristics in `derive-outcome.ts` become a fallback — which source
  wins on conflict needs stating.
