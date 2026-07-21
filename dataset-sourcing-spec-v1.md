# Project Lazarus — Sourcing Phase Spec (v1)

Workstream #1 (ranked first). Feeds the v4 dataset schema. Reassessment mode chosen:
**fully automated (LLM + web search)**, designed so it can be upgraded to grounded
data feeds later.

## Core principle: separate DISCOVERY from OUTCOME

The current pipeline treats one launch/funding article as a company's whole story.
That's why `living_status`, `challenges`, and the exit fields are unreliable — the
outcome hadn't happened yet when the article was written. Fix: split the pipeline
so the article is used only for *what the company was*, and a separate
**retrospective pass, dated to now**, establishes *what became of it*.

## Target pipeline shape

```
discovery      scrape/seed         -> candidate companies (the launch moment)
extraction     step1               -> idea_summary, idea_dependencies, sectors, idea_space
resolve        NEW (entity res)    -> merge duplicates into canonical companies
outcome pass   NEW (retrospective) -> living_status, exit_*, challenges(+outcome), outcome_summary
funding        step2/step3         -> funding_rounds (raises only; exits live on companies)
derive         v4 derivation       -> outcome_type from the above
reassess       NEW (automated)     -> dependency_assessments (the "now" axis)
build          -> CSVs -> query DB
```

## The retrospective outcome pass (the highest-leverage add)

For each company, search **now** with present-dated queries — "what happened to
{company}", "{company} shut down", "{company} acquired", "{company} raised" — and
extract the outcome from results that postdate the company's active life.

- Writes: `living_status`, `exit_type/amount/date/notes`, `challenges` rows with
  their `outcome` (fatal/overcome/pivoted_from/ongoing), `outcome_summary` +
  `outcome_source_url`.
- Every extracted fact carries its own `source_url`; prefer the most recent,
  most authoritative result over the launch article.
- This single step improves outcome data more than any new feed, because it looks
  back from *after* the outcome.

## Sources

Reuse what's wired, add outcome-capturing ones:

- **Wikipedia** — ALREADY WIRED (`src/tools/search.ts`), but currently only feeds
  step2 funding. **Promote it to the outcome pass**: it's strong for acquisitions,
  shutdowns, founding facts, and status. Keep it in funding too if useful, but its
  primary value is company/outcome enrichment.
- **DuckDuckGo** — wired, but rate-limited and low-precision. Treat as best-effort;
  consider replacing with a proper search API (see open decisions).
- **Crunchbase** — wired (`src/tools/crunchbase.ts`). Structured ground truth for
  status, funding, and acquisition — lean on it for the exit fields.
- **Add, outcome-focused:** startup post-mortem / shutdown collections; the Wayback
  Machine for dead company sites (a defunct site is itself a strong status signal);
  climate-specific databases — Dealroom, Net Zero Insights, Sightline.
- **Widen the time window backward to cleantech 1.0 (~2006-2013).** That boom-bust
  — solar, biofuels, grid-scale storage, smart grid — is the richest vein of
  "failed then, maybe viable now," which is the whole thesis.

## Prerequisites (do these first — otherwise more sources just multiply noise)

- **Entity resolution.** Going multi-source makes the grain problem urgent: the same
  company across four sources must merge into one canonical `companies` row, not
  become four. Define a canonical key (normalized name + website domain, with
  founders/founding-year as tiebreakers) and a match step before merge.
  - **Resolve runs before the outcome pass, not after** (see the pipeline shape
    above). Merging first means one retrospective search per *canonical* company
    instead of one per duplicate row — the expensive pass is paid for once — and it
    avoids the harder problem of reconciling outcomes that disagree because two rows
    for the same company were each researched separately. General rule for this
    pipeline: **merge before enrichment, never after.**
- **Incremental scraping (accumulate, don't replace).** Building the corpus in
  batches — scrape one set of years, check it, scrape another set later to
  supplement — must not mean losing the earlier batch. step0 already writes a
  timestamped `techcrunch_article_<ts>_data.csv` per run and `latestScrapedDataFile()`
  takes the newest, so today a second scrape silently *replaces* the first as step1's
  input. Keep that default (latest file wins, unspecified = newest), and add a step0
  flag meaning **supplement**: copy the previous latest file forward under the new
  timestamp, then concatenate the newly scraped rows into it. The newest file stays
  the whole corpus, so no downstream step changes. Apply the same to the
  `_data_index.csv` companion.
  - **De-duplicate on concatenate**, by article `url` — the same listing page
    re-scraped, or overlapping year ranges, must not double-count. Same principle as
    entity resolution: merge before enrichment, never after, so the LLM passes are
    never paid for twice on the same article.
  - This makes the year range an *operational scoping parameter* — which slice to go
    fetch next — rather than a selection filter on what belongs in the dataset.
- **Raw-text store + caching.** Persist every fetched article/result text keyed by a
  URL hash, with `fetched_at`. Then re-extraction under an evolving schema (this has
  already gone v1->v4) never re-scrapes, and runs are resumable and cheap. A
  `raw_documents(id, url, url_hash, fetched_at, text, source_type)` table or a
  content-addressed file cache both work.

## Reassessment pass (automated mode)

Each `dependency_assessments` row must store its own evidence so the auto-generated
verdict is auditable and later upgradable:

- `status`, `detail`, `metric_name`, `metric_value`, `metric_unit`, `assessed_on`,
  `source_url`, plus a retrieved snippet and a `confidence` flag.
- Storing the snippet + source + confidence is what makes "start automated, upgrade
  the high-value dependencies to real data feeds later" a swap rather than a rewrite.
- This also delivers part of workstream #4 (verification) for free.

## Schema touchpoints (v4)

- outcome pass -> `companies.living_status`, `exit_*`, `outcome_summary`,
  `outcome_source_url`; `challenges` rows.
- entity resolution -> canonical `companies`; carries `uuid` for provenance.
- reassessment -> `dependency_assessments` (evidence fields above).
- No new schema tables required except the optional `raw_documents` cache.

## Open decisions

- Replace DuckDuckGo with a paid/stable search API (better recall + fewer rate
  limits), or keep it as best-effort behind the cache?
- Entity resolution: auto-merge above a match-score threshold, or queue ambiguous
  matches for review?
- ~~How far back to extend discovery — a hard year floor or driven by idea_spaces?~~
  **Decided: neither is a filter.** A company is never excluded for lacking a modern
  counterpart in its `idea_space`. An idea space with a 2008 attempt and nothing
  present-day is either a genuinely tested dead end *or a gap* — something nobody is
  working on that maybe should be — and the gap case is among the most valuable
  things this dataset can surface. Filtering on comparability deletes it before it
  can be seen.
  - "Has a modern counterpart" is a **queryable property derived at analysis time**
    (does the `idea_space` hold companies on both sides of the era boundary?), not a
    collection-time filter. This matches the schema's existing choice to store
    successes alongside failures so `fatal` is a query-time comparison and a hurdle
    with no survivor reads as "none sampled yet" — absence stays visible and
    interpretable rather than silently dropped.
  - What remains open is only **scraper reach**: how far back TechCrunch archives
    stay parseable, and where the Wayback Machine has to fill in. Year ranges are
    then just batches to work through (see incremental scraping above).
- Caching/runtime budget: how aggressively to cache and how often to refresh the
  outcome pass (outcomes change; a company alive today may fold next year).
