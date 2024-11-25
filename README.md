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

The scripts are labeled step0, step1, step2, and step3 to make it clear the order. They are not perfect as it depends on the output from the AI agents and sometimes the formatting can be incorrect. `response_format.py` contains various types of formats to try but depending on the need, the ChatGPT 4o models have worked best.

There are multiple step1 and step2 articles. You don't need to run them all they are just different approaches typically using chatgpt vs llama or a different kind of agent.

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
