# Data sources (enrichment)

Static external reference data used to **enrich companies the pipeline finds** — fill
in status, funding, founding/defunct years, and known failure reasons without (or
before) a web search. This is reference data, not a pipeline run: it lives outside
`output/<run>/` and changes only when regenerated.

Regenerate everything with:

```
npm run build-sources
```

That copies each source's untouched originals from `data/kaggle/` into
`sources/<name>/raw/`, then writes a normalized `sources/<name>/companies.csv` in the
shared enrichment schema below. Originals in `data/kaggle/` are never modified.

> **Not yet wired into the pipeline.** The normalized files and the loader
> (`loadSource` in [`src/lib/sources.ts`](../../src/lib/sources.ts)) exist; the
> enrichment *step* that consults them is a later round. See the "wire later" decision
> in the sourcing specs.

## Layout

```
data/sources/
  crunchbase/
    raw/big_startup_secsees_dataset.csv   copied verbatim
    companies.csv                          normalized, 66,368 rows
  startup-failures/
    raw/*.csv                              copied verbatim (base roster + sector tables)
    companies.csv                          normalized, 890 rows
```

## Shared enrichment schema (`companies.csv`)

One row per company. Join keys are normalized with the **same functions company
resolution uses** (`normalizeDomain`, `normalizeName`), so a lookup lines up with
`company_urls`. A row may carry several keys; a future enrich step tries them strongest
first (website → crunchbase → name).

| Column | Meaning |
| --- | --- |
| `source` | which source produced the row (`crunchbase`, `startup-failures`) |
| `company_name` | original name as stated |
| `key_website` | normalized domain (bare host), when known |
| `key_crunchbase` | Crunchbase permalink slug, when known |
| `key_name` | normalized name — the fallback join key |
| `living_status` | mapped to the `LIVING_STATUS` vocab |
| `exit_type` | `acquisition` / `ipo` / `shutdown` when the source implies one |
| `funding_total_usd` | integer USD; blank (not 0) when undisclosed |
| `funding_rounds` | count, when the source has it |
| `year_founded`, `year_defunct` | 4-digit years, when known |
| `idea_summary` | what the company did, when the source describes it |
| `failure_reason` | free-text cause of failure, when known |
| `challenge_categories` | `;`-joined `CHALLENGE` categories mapped from the source |
| `sector_raw` | the source's own sector/category string (unmapped) |
| `country` | country code, when present |
| `source_ref` | the source's own identifier for the row (permalink or name) |

## Per-source notes

**crunchbase** (Crunchbase-derived, 66,368 rows). The strong-join source: 61,306 rows
carry a website domain and all carry a permalink slug, both matching the resolver's
keys. Supplies `living_status` (from operating/closed/acquired/ipo), `exit_type`,
`funding_total_usd`, `funding_rounds`, `year_founded`, `sector_raw`, `country`.
No description field, so `idea_summary` is blank.

**startup-failures** (name-keyed post-mortems, 890 rows). A **base roster**
(`Startup Failures.csv`, 814 companies: name/sector/years) plus **supporting sector
tables** that join back by name for the rich detail. The normalized set is the *union*
of names: 403 rows get sector detail (`failure_reason`, `idea_summary`,
`challenge_categories`, parsed raise), the other 487 are roster-only (name/sector/years).
All are `Defunct` by construction. Join key is name only — lower confidence than
crunchbase, but the failure-reason content is closest to the project thesis.

## Deferred

**startup-india** — `data/kaggle/startup-india-815-failures-archive/Startup Failures.xlsx`
is an Excel file; it needs an `.xlsx → csv` conversion pass before it can be normalized
into `sources/startup-india/`. Not done yet — no xlsx reader is wired in.

**startup-failure-prediction** — set aside: names are anonymized placeholders
(`Startup_1`, …), so no row can join to a real company.
