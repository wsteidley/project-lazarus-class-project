import os

import requests
import httpx
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@tool
async def wiki_search_async(search_string: str):
    """Searches wikipedia for a relevant article with the given search string"""
    try:
        print(f"Searching Wikipedia: {search_string}")
        response = await wikipedia.ainvoke(str(search_string))
    except Exception as e:
        print(f"Wiki search exception response: {str(e)}")
        response = ''
    return response

@tool
async def duck_duck_go_search_async(search_string: str):
    """Fetches search results from DuckDuckGo with the given search string"""
    print(f"Searching DDG: {search_string}")
    response = ''
    try:
        response = await search.ainvoke(str(search_string))
    except Exception as e:
        print(f"DDG search exception response: {str(e)}")
    finally:
        return response


@tool
def company_name_crunchbase_search(company_name):
    """Fetches Crunchbase Basic API data which can be useful to fill in company data. The function
    accepts the name of the company and returns the result of the call to the API."""
    print(f"company_name_crunchbase_search for: {company_name}")
    try:
        api_key = os.environ["CRUNCHBASE_API_KEY"]
        url = "https://api.crunchbase.com/v4/data/searches/organizations"
        payload = { 
            "field_ids": [
                "created_at",
                "entity_def_id",
                "facebook",
                "facet_ids",
                "identifier",
                "image_id",
                "image_url",
                "linkedin",
                "location_identifiers",
                "name",
                "permalink",
                "short_description",
                "stock_exchange_symbol",
                "twitter",
                "updated_at",
                "uuid",
                "website_url"
            ],
            "query": [
                {
                    "operator_id": "includes",
                    "type": "predicate",
                    "field_id": "facet_ids",
                    "values": ["company"]
                },
                {
                    "operator_id": "eq",
                    "type": "predicate",
                    "field_id": "identifier",
                    "values": [f"{company_name}"]
                }
            ]
        }

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "X-cb-user-key": f"{api_key}"
        }

        response = requests.post(url, json=payload, headers=headers)
        return response.text()
            
    except requests.RequestException as e:
        return f"An error occurred while fetching the URL: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

@tool
async def company_name_crunchbase_search_async(company_name):
    """Fetches Crunchbase Basic API data which can be useful to fill in company data. The function
    accepts the name of the company and returns the result of the call to the API."""
    print(f"company_name_crunchbase_search_async for: {company_name}")
    try:
        api_key = os.environ["CRUNCHBASE_API_KEY"]
        url = "https://api.crunchbase.com/v4/data/searches/organizations"
        payload = { 
            "field_ids": [
                "created_at",
                "entity_def_id",
                "facebook",
                "facet_ids",
                "identifier",
                "image_id",
                "image_url",
                "linkedin",
                "location_identifiers",
                "name",
                "permalink",
                "short_description",
                "stock_exchange_symbol",
                "twitter",
                "updated_at",
                "uuid",
                "website_url"
            ],
            "query": [
                {
                    "operator_id": "includes",
                    "type": "predicate",
                    "field_id": "facet_ids",
                    "values": ["company"]
                },
                {
                    "operator_id": "eq",
                    "type": "predicate",
                    "field_id": "identifier",
                    "values": [f"{company_name}"]
                }
            ]
        }

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "X-cb-user-key": f"{api_key}"
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            return response.text

    except requests.RequestException as e:
        return f"An error occurred while fetching the URL: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    print(company_name_crunchbase_search.invoke({"company_name":"Oura"}))
