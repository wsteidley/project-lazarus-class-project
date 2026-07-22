# project-lazarus

## Project Overview

### Lazarus Project

Reinvigorating climate tech ideas that were before their time

### Problem Statement:

Many climate tech startups have faced significant challenges over recent decades that led to their failure. Despite promising technologies and strong ideas, factors like technical feasibility, market timing, high costs, or scalability issues often got in the way. As the Drawdown project emphasizes, 'we have all the solutions to climate change we need', but the challenge lies in scaling and execution. We aim to understand these causes systematically, so we can identify which ideas were simply ahead of their time and which could succeed today, given the right conditions. This project is about transforming forgotten failures into today’s opportunities in climate tech.

## Technical Approach

In order to gain insights we need to create a dataset which can be queried and analyzed. In order to create the dataset we want to take unstructured data (scraped articles) and pass that through a series of steps using Python and LLMs with AI Agents to create a structured dataset.

### Simplified Flow

Ideally it would be this easy but there are challenges along the way:

```
Scraped, Unstructured Data -> AI Agent Stages -> Structured Data
```

## Overview

The pipeline is a sequence of stages, maintained as the **TypeScript** version
under `src/` (see below). The original Python scripts under `scraping/techcrunch/`
are kept as legacy reference. The output is a small **relational** dataset — one
CSV per table in `DATA_DIR` (default `./data`), loaded into a single-file SQLite
database (`lazarus.db`). The schema holds **successes alongside failures** as a
control group — so a hurdle that killed one company can be told apart from one a
later company overcame, rather than assumed fatal. (Each challenge is labelled
`fatal`/`overcome` per company; whether a hurdle is *genuinely* fatal is a
query-time comparison, and a hurdle with no recorded survivor means "none sampled
yet," not "unbeatable".) It follows the sector → idea_space → company hierarchy in
[dataset-schema-spec-v4.md](dataset-schema-spec-v4.md).

The stages are:

- **step0** — scrape a TechCrunch listing page into an article index, then scrape
  each full article. Outputs `techcrunch_article_<ts>_data.csv` (raw input).
- **step1** — extract structured company info from each article with an LLM
  (optionally Crunchbase-enriched), mapping each company into the curated
  `data/idea_spaces.csv`. Writes `data/companies.csv`, `data/company_sectors.csv`
  (a company can span several sectors), `data/challenges.csv` (each challenge
  tagged fatal / overcome / pivoted_from / ongoing, each with a `confidence` level
  and a `contested` flag for when sources disagree), and `data/company_urls.csv` —
  every URL a source states, tagged by type (`website`, `crunchbase`, `wikipedia`,
  `archive`, …). URLs are a typed table rather than one column per source, so adding
  a source is a new row instead of a schema change.
- **step1b** — a separate LLM pass decomposing each idea into its dependencies.
  Writes `data/idea_dependencies.csv` (free text, one row per company).
- **step1c** — resolves those free-text dependencies onto the curated canonical list
  in `data/input/dependencies.csv`, merging duplicates per company. Writes
  `company_dependencies.csv`. Anything that matches nothing canonical is kept with a
  blank name and reported, never silently dropped — add it to the seed and re-run.
- **resolve** — merges duplicate companies into canonical rows, ID-first over
  `company_urls`: `website` domain → `crunchbase` → `wikipedia` → normalized name +
  founding year. Non-identity URL types (`article`, `linkedin`, …) are never merge
  keys — a shared press link says nothing about two companies being the same. Rewrites
  `companies.csv`, re-points and de-duplicates every child table's `company_uuid` (a
  merged company ends up with the *union* of its duplicates' URLs), and writes
  `company_uuid_map.csv` for provenance. Runs **before** funding on purpose, so rounds
  are only ever attached to canonical companies. Reports merges by key tier plus
  near-miss pairs (same name, no shared key) — the evidence for whether probabilistic
  matching is worth adding later.
- **step2** — find funding-round data per company via tiered web search (Tavily,
  falling back to DuckDuckGo) + Wikipedia + the LLM, one row per round. Writes
  `data/funding_rounds.csv`.
- **step3** — deterministic cleanup: standardize dates to `YYYY-MM` (4-digit year),
  derive `round_year`, de-duplicate rounds. Rewrites `data/funding_rounds.csv`.
- **derive** — deterministic derivation of each company's `outcome_type` +
  `outcome_rationale` from `living_status`, total raised, exit signals, and company
  age. Runs after step3 (needs funding); rewrites `data/companies.csv`. Thresholds
  live in [src/lib/derive-outcome.ts](src/lib/derive-outcome.ts) — tune them there.
- **reassess** — the "now" axis: for each *canonical* dependency, searches the web
  today and records where it stands (`resolved` / `improving` / `unchanged` /
  `worsening`), with a metric where the sources give one, plus its own evidence
  (`source_url`, verbatim `snippet`, `confidence`, `contested`). Writes
  `dependency_assessments.csv`. Assessed per shared dependency rather than per
  company, so one verdict serves every company that depended on it.
- **build** — load the table CSVs into `lazarus.db` (SQLite via Node's built-in
  `node:sqlite`), seeding the `sectors` reference rows, resolving name/uuid foreign
  keys, and enabling FK enforcement. Also mirrors the fetch cache into
  `raw_documents`.

`data/input/idea_spaces.csv` and `data/input/dependencies.csv` are **curated seeds**
you maintain; step1 only maps companies into idea spaces you've defined, and step1c
only resolves dependencies onto canonical rows you've defined. The quality of the
head-to-head comparisons depends on both. Controlled-vocabulary fields are enforced
by Zod enums in [src/schemas.ts](src/schemas.ts) and mirrored as DB `CHECK`
constraints. `outcome_type` gray-zone LLM adjudication, `current_trl`, the
`became_viable_date` derivation over the `threshold_*` columns, and a `sources` table
are specced but not yet implemented.

### Data layout

Everything lives under `data/` (relocatable via `DATA_DIR`):

```
data/
  input/idea_spaces.csv        # curated seeds you maintain (tracked, never cleaned)
  input/dependencies.csv       #   canonical dependencies + their viability thresholds
  scraped/                     # step0 output, timestamped article CSVs
  cache/<url_hash>.json        # content-addressed fetch cache, survives runs
  sources/<name>/              # static enrichment data (Crunchbase, failure sets):
                               #   raw/ copies + a normalized companies.csv
  output/<run>/                # one timestamped folder per pipeline run:
                               #   the table CSVs + lazarus.db
```

`data/sources/` holds external reference data used to enrich companies the pipeline
finds — status, funding, founding/defunct years, known failure reasons. Regenerate with
`npm run build-sources` (copies originals from `data/kaggle/`, writes a normalized
`companies.csv` per source). The normalized files and loader exist; the enrichment step
that consults them is a later round. See [data/sources/README.md](data/sources/README.md).

Steps **auto-resolve the latest input** — you never pass a run id. step1 reads the
newest scrape and opens a fresh `output/<run>/`; the later steps all flow into the
newest run folder (build writes `lazarus.db` there). `npm run clean` wipes `scraped/`
and `output/` but leaves your curated `input/` intact.

`cache/` sits deliberately outside the run folders so it survives them: re-extracting
under an evolving schema never re-scrapes. Freshness is two-tier — launch articles
(`discovery`) are immutable and cached forever, while `outcome`/`reassessment` results
expire after `CACHE_TTL_DAYS` (default 30), because a company alive today may fold
next year.

## TypeScript (src/)

The `src/` project is Node + TypeScript. It replaces `requests`/BeautifulSoup with
`fetch`/cheerio, pandas with `csv-parse`/`csv-stringify`, and the fragile custom
LangChain output parsers with LangChain.js `withStructuredOutput` + Zod schemas so
the LLM returns validated JSON. There is one file per step; the model provider is
a config switch rather than a separate script.

### Setup

Requires Node 20+ (the `build` step uses `node:sqlite`, available in Node 22.5+).

```
$ npm install
$ cp .env.example .env   # then fill in the values you need
```

Environment variables (see `.env.example`):

- `PROVIDER` — `openai` (default) or `ollama`
- `OPENAI_API_KEY` — when `PROVIDER=openai`
- `LLAMA_BASE_URL` — when `PROVIDER=ollama`, e.g. `http://localhost:11434`
- `CRUNCHBASE_API_KEY` — optional, enables Crunchbase enrichment in step1
- `TAVILY_API_KEY` — optional; primary web-search provider. Without it, search
  silently falls back to DuckDuckGo (lower recall, rate-limited, but never fails)
- `SCRAPE_URL` — the TechCrunch listing URL for step0
- `DATA_DIR` — root data dir (default `./data`)
- `INPUT_FILE` — optional override of the auto-selected scrape (step1/step1b)
- `DB_FILE` — optional override of the DB path (default `<run>/lazarus.db`)
- `BUILD_DATE` — reference date for the `derive` step (default: now)
- `CACHE_TTL_DAYS` — how long perishable cached documents stay usable (default 30)
- `PROCESSING_LIMIT` (default 10), `BATCH_SIZE` (default 5) — optional tuning

### Running the pipeline

Populate the curated `data/input/idea_spaces.csv` and `data/input/dependencies.csv`
first (starter files are included). Each step auto-resolves the latest input, so no
paths to pass:

```
$ SCRAPE_URL=https://techcrunch.com/tag/climate  npm run step0   # -> data/scraped/
$ npm run step1      # latest scrape -> new data/output/<run>/
$ npm run step1b
$ npm run step1c     # free-text dependencies -> canonical ones
$ npm run resolve    # merge duplicate companies (before any enrichment)
$ npm run step2
$ npm run step3
$ npm run derive     # compute outcome_type from the signals
$ npm run reassess   # where each dependency stands today
$ npm run build      # -> data/output/<run>/lazarus.db

$ npm run clean      # wipe scraped/ + output/, keep curated input/
```

Switch providers by setting `PROVIDER=ollama` (runs locally, no rate limits). Then
query the DB, e.g. the core comparison — who overcame a challenge that killed
others: `SELECT category, SUM(outcome='fatal') killed, SUM(outcome='overcome') overcame FROM challenges GROUP BY category`.

### Lint / typecheck

Biome is the formatter and linter; `tsc` is the typechecker; Vitest runs the tests.

```
$ npm run typecheck
$ npm run lint     # biome check
$ npm run format   # biome format --write
$ npm test         # vitest run
```

Tests cover the deterministic pieces — funding date standardization + dedup (step3),
the outcome_type derivation rules (every branch + conservative gray-zone fallbacks),
the listing-page parser (against the saved `scraping/techcrunch/articles_page.txt`
fixture), the Zod vocab enums (out-of-vocab values are rejected), and the DB schema
(the sector→idea_space→company join, CHECK constraints, FK enforcement). The LLM
steps are not unit-tested; verify those with a live run.

## Legacy Python

The sections below describe the original Python scripts under
`scraping/techcrunch/`. They are labeled step0–step3 in the same order. There are
multiple step1 and step2 variants (chatgpt vs llama, different agent styles) — you
don't need to run them all. `response_format.py` holds the schema variants;
ChatGPT 4o models worked best.

## Setup

You'll need to set some environement variables depending on what you want to do. For example:

- `LLAMA_BASE_URL`
- `CRUNCHBASE_API_KEY`
- `OPENAI_API_KEY`

#### Activate the Virtual Environment

```
$ source virtualEnv.sh
```

#### Install Requirements

```
(techcrunch_venv) $ pip install -r requirements.txt
```

#### Deactivate Virtual Environment

When you're done you can:

```
(techcrunch_venv) $ deactivate
```

## Scraping

An initial scraping script for Techcrunch has been added. This has a virtual environment setup within that folder. To get going with that scraper do the following:

#### Update URL and run Scraping Script

You'll need to update the `url` to start with for running the scraping.
There is an example url in the comments: `https://techcrunch.com/category/startups/`

or use

**articles_page.txt**

Which is a scraped page that has been saved.

Then run the scraping script as normal with python:

```
(techcrunch_venv) $ python step0_scrape_article_listings.py
```

## Test Data

There is some test data in the repo:

#### Test Input Save From Scrapes

`articles_page.txt`

### Saved Output

Ouptut for each step is saved to a .csv with whatever designated filename

## Troubleshooting

The scripts are generally written to be async which sometimes can swallow exceptions. If you're having trouble running a script try running outside of the asycio/task gathering flow.

## Challenges

#### Langchain

Langchain is the standard but has some issues. For example an older version of the react loop agent seemed to work okay but the latest seems to have issues with outputing json which cannot be parsed.

#### LLM Model Differences

ChatGPT 4o models have worked better during this project and test.

Llama3 (3, 3.1, and 3.2 were tested) work but need different prompting. The benefit of Llama with Ollama is you can run it on your own hardware and don't have rate limits per minute/hour/day.
