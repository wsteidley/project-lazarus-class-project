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
  (a company can span several sectors), and `data/challenges.csv` (each challenge
  tagged fatal / overcome / pivoted_from / ongoing).
- **step1b** — a separate LLM pass decomposing each idea into its dependencies.
  Writes `data/idea_dependencies.csv`.
- **step2** — find funding-round data per company via web search (DuckDuckGo +
  Wikipedia) + the LLM, one row per round. Writes `data/funding_rounds.csv`.
- **step3** — deterministic cleanup: standardize dates to `YYYY-MM` (4-digit year),
  derive `round_year`, de-duplicate rounds. Rewrites `data/funding_rounds.csv`.
- **derive** — deterministic derivation of each company's `outcome_type` +
  `outcome_rationale` from `living_status`, total raised, exit signals, and company
  age. Runs after step3 (needs funding); rewrites `data/companies.csv`. Thresholds
  live in [src/lib/derive-outcome.ts](src/lib/derive-outcome.ts) — tune them there.
- **build** — load the table CSVs into `lazarus.db` (SQLite via Node's built-in
  `node:sqlite`), seeding the `sectors` reference rows, resolving name/uuid foreign
  keys, and enabling FK enforcement.

`data/idea_spaces.csv` is a **curated seed** you maintain (name, home sector,
description); step1 only maps companies into spaces you've defined, and the quality
of the head-to-head comparisons depends on it. Controlled-vocabulary fields are
enforced by Zod enums in [src/schemas.ts](src/schemas.ts) and mirrored as DB
`CHECK` constraints. `outcome_type` gray-zone LLM adjudication, the reassessment
axis (`dependency_assessments`, `current_trl`), and a `sources` table are specced
but not yet implemented.

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
- `SCRAPE_URL` — the TechCrunch listing URL for step0
- `INPUT_FILE` — the article CSV from step0 (used by step1 and step1b)
- `DATA_DIR` — where table CSVs are written (default `./data`)
- `DB_FILE` — the SQLite output path for `build` (default `lazarus.db`)
- `BUILD_DATE` — reference date for the `derive` step (default: now)
- `PROCESSING_LIMIT` (default 10), `BATCH_SIZE` (default 5) — optional tuning

### Running the pipeline

Populate the curated `data/idea_spaces.csv` first (a starter file is included).
step1/step1b/step2/step3/derive read and write the table CSVs in `DATA_DIR`; only
step0, step1, and step1b need `INPUT_FILE` (the raw article CSV).

```
$ SCRAPE_URL=https://techcrunch.com/tag/climate      npm run step0
$ INPUT_FILE=techcrunch_article_<ts>_data.csv        npm run step1
$ INPUT_FILE=techcrunch_article_<ts>_data.csv        npm run step1b
$ npm run step2
$ npm run step3
$ npm run derive     # compute outcome_type from the signals
$ npm run build      # -> lazarus.db
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
