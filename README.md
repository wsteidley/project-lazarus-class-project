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

The pipeline runs in four ordered stages — step0, step1, step2, step3. The
current, maintained implementation is the **TypeScript** version under `src/`
(see below). The original Python scripts under `scraping/techcrunch/` are kept as
legacy reference.

The stages are:

- **step0** — scrape a TechCrunch listing page into an article index, then scrape
  each full article. Outputs `techcrunch_article_<ts>_data.csv`.
- **step1** — extract structured company info from each article with an LLM,
  optionally enriched with Crunchbase. Outputs `parsed_data_<ts>.csv`.
- **step2** — enrich each company with funding-round data using web search
  (DuckDuckGo + Wikipedia) plus the LLM. Outputs `added_funding_data_<ts>.csv`.
- **step3** — deterministic (no LLM) cleanup: standardize round names and dates,
  de-duplicate rounds per company. Outputs `standardized_funding_<ts>.csv` and a
  `..._amount_by_year.json` sidecar (replaces the old matplotlib/plotly plot).

## TypeScript (src/)

The `src/` project is Node + TypeScript. It replaces `requests`/BeautifulSoup with
`fetch`/cheerio, pandas with `csv-parse`/`csv-stringify`, and the fragile custom
LangChain output parsers with LangChain.js `withStructuredOutput` + Zod schemas so
the LLM returns validated JSON. There is one file per step; the model provider is
a config switch rather than a separate script.

### Setup

Requires Node 20+.

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
- `INPUT_FILE` — the input CSV for step1/step2/step3 (output of the prior step)
- `PROCESSING_LIMIT` (default 10), `BATCH_SIZE` (default 5) — optional tuning

### Running the pipeline

```
$ SCRAPE_URL=https://techcrunch.com/tag/climate npm run step0
$ INPUT_FILE=techcrunch_article_<ts>_data.csv       npm run step1
$ INPUT_FILE=parsed_data_<ts>.csv                   npm run step2
$ INPUT_FILE=added_funding_data_<ts>.csv            npm run step3
```

Switch providers by setting `PROVIDER=ollama` (runs locally, no rate limits).

### Lint / typecheck

Biome is the formatter and linter; `tsc` is the typechecker.

```
$ npm run typecheck
$ npm run lint     # biome check
$ npm run format   # biome format --write
```

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
