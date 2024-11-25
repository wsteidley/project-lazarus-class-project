import os

from langchain_openai import ChatOpenAI
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from tools_search import company_name_crunchbase_search_async
import pandas as pd
import json
from datetime import datetime, timezone
from response_format import response_format
import math
from langchain.load.dump import dumps
import asyncio
import uuid
import time
from typing import List, Dict, Any

api_key = os.environ["OPENAI_API_KEY"]

utc_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
print(utc_timestamp)

#-------------------

total_token_tracking = 0
total_requests_tracking = 0
retry_articles = [] # simple retry logic
async def process_article_batch(
    articles: List[Dict[str, Any]], 
    batch_size: int = 5
) -> List[Dict[str, Any]]:
    """Process articles in concurrent batches"""
    global total_token_tracking
    global total_requests_tracking
    results = []
    article_len = len(articles)
    denominator = math.ceil(article_len/batch_size)
    start = time.time()
    
    for i in range(0, article_len, batch_size):
        total_requests_tracking += batch_size
        batch = articles[i:i + batch_size]
        tasks = [
            extract_data_with_conversational_retrieval_agent(
                author=article['author'],
                title=article['title'],
                publication_date=article['publication_date'],
                content=article['content'],
                url=article.get('url')
            )
            for article in batch
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        # Filter out exceptions and add successful results
        for result in batch_results:
            if not isinstance(result, Exception):
                (json_obj, tokens) = result
                total_token_tracking += tokens
                results.append(json_obj)
        numerator = i//batch_size + 1
        end = time.time()
        elapsed_time = end - start
        print(f"Elapsed time: {elapsed_time}")
        print(f"total_token_tracking: {total_token_tracking}")
        avg_time = elapsed_time // numerator
        print(f"Avg time: {avg_time}")
        time_remaining = avg_time * (denominator - numerator)
        print(f"Estimated time remaining: {round(time_remaining / 60)} minutes")
        print(f"Processed batch {numerator}/{denominator}")

    return results

async def extract_data_with_conversational_retrieval_agent(
    author: str,
    title: str,
    publication_date: str,
    content: str,
    url: str
) -> Dict[str, Any]:
    """Extract data from a single article"""
    global retry_articles
    tools = [company_name_crunchbase_search_async]
    tool_names = ", ".join([tool.name for tool in tools])
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model="gpt-4o-mini",
        temperature=0,
        timeout=60
    )
    template = f"""You are a helpful data processor and extractor. Pull the proper information for the fields in {response_format}
    from a given article. Use the title, author, publication date, and content and output in the proper format.
    If you learn the company name you may research using {tools} name {tool_names} as needed.
    The output should only contain one valid response object that follows the schema. It's okay if you don't 
    find data for all the fields if you've found the required fields.
    IMPORTANT: Your final response MUST BE A VALID JSON OBJECT.
    You should return JSON ONLY in the schema format. Do not write code or anything additional.

    SCHEMA: 

    {response_format}

    Begin!
    
    Title, Author, Publication Data, Content: {title}, {author}, {publication_date}, {content}
    """

    agent = create_conversational_retrieval_agent(
        tools=tools,
        llm=llm
    )

    try:
        result = await agent.ainvoke(template)
        # Extract the JSON from the agent's response
        if isinstance(result, dict) and 'output' in result:
            total_tokens_for_result = 0
            try:
                intermediate_steps = result['intermediate_steps']
                if len(intermediate_steps):
                    inner_obj = intermediate_steps[0][0]
                    message_log = json.loads(dumps(inner_obj))['kwargs']['message_log']
                    for log in message_log:
                        total_tokens_for_result += log['kwargs']['usage_metadata']['total_tokens']
            except Exception as e:
                print(f"Error getting tokens: {str(e)}")

            loaded_json = json.loads(result['output'])
            # Add metadata
            loaded_json.update({
                'url': url,
                'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                'uuid': str(uuid.uuid4())
            })
            return (loaded_json, total_tokens_for_result)
        else:
            raise ValueError(f"Unexpected response format from agent: {result}")
    except Exception as e:
        if 'rate_limit_exceeded' in str(e):
            retry_articles.append({
                'author': author,
                'title': title,
                'publication_date': publication_date,
                'content': content,
                'url': url
            })
        print(f"Error processing article '{title}': {str(e)}")
        raise e

async def extract_data_for_file(filename: str, batch_size: int = 5) -> pd.DataFrame:
    """Process entire file with batched async operations"""
    article_data_df = pd.read_csv(filename)
    articles = article_data_df.to_dict('records')
    
    # Using the first 10 items
    limited_articles = articles[:10]
    print(f"Processing {len(limited_articles)} articles...")
    
    json_data_list = await process_article_batch(
        articles=limited_articles,
        batch_size=batch_size
    )

    if len(retry_articles):
        retry_data_list = await process_article_batch(
            articles=retry_articles,
            batch_size=batch_size
        )
        json_data_list.extend(retry_data_list)
    
    if not json_data_list:
        raise ValueError("No data was successfully processed")
        
    return pd.DataFrame(json_data_list)

def save_article_data_to_csv(data_frame: pd.DataFrame, filename: str) -> None:
    """Save processed data to CSV"""
    if not filename:
        raise ValueError('Need to provide a filename to save')
    data_frame.to_csv(filename, index=False, encoding='utf-8')
    print(f"Parsed data successfully saved to {filename}")

async def main():
    filename = ''
    if not filename:
        raise ValueError("Filename should be output file from step0")
    utc_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    
    try:
        extracted_data_df = await extract_data_for_file(filename, batch_size=5)
        save_article_data_to_csv(
            extracted_data_df, 
            f"parsed_data_{utc_timestamp}.csv"
        )
        print("\nDONE\n")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    asyncio.run(main())