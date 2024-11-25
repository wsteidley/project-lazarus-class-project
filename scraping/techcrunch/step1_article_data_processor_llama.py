import os

from langchain_community.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser

import pandas as pd
import json
from datetime import datetime, timezone
from response_format import response_format_llama

import math
from langchain.load.dump import dumps
import asyncio
import uuid
import time
from typing import List, Dict, Any, Union

from langchain_ollama import OllamaLLM
from langchain_core.tools import tool
import re
from langchain.schema import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from tools_search import duck_duck_go_search_async, wiki_search_async, company_name_crunchbase_search, company_name_crunchbase_search_async

# There are other schemas available from response_format file
schema = response_format_llama
utc_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
print(utc_timestamp)
llama_base_url = os.environ["LLAMA_BASE_URL"]

#-------------------
class MyOutputParser(ReActJsonSingleInputOutputParser):
    def parse(self, llm_output: str) -> Union[str, AgentAction, AgentFinish]:
        """
        Parse the output from the language model.
        
        Args:
            llm_output (str): The raw output from the language model
        
        Returns:
            Union[AgentAction, AgentFinish]: Parsed agent output
        """

        # Clean up the output
        llm_output = llm_output.strip()
        print(f"llm_output: \n {llm_output}\n")
        # Check if the agent is finishing
        if "Final Answer" in llm_output:
                
            print("doing Final Answer")
            return_text = llm_output.split("Final Answer\":")[-1][:-1].strip()
            print(f"return_text: {return_text}")
            try:
                json.loads(return_text)
            except Exception as e:
                # print('json.load failed')
                return_text = return_text + '}' 
            return AgentFinish(
                return_values={"output": return_text},
                log=llm_output
            )
        # Parse action and input
        try:
            super_parse = super().parse(llm_output)
            return super_parse
        except Exception as e:
            # print(f'Exception: {e}\n\n in parent parse, attemping to parse myself\n')
            try:
                # Custom parsing logic - adjust based on your specific output format
                action_split = llm_output.split("Action", 1)
                print('--------------------')
                loaded = json.loads(llm_output)
                action_input_split = llm_output.split("Action Input", 1)
                if len(action_split) > 1 and len(action_input_split) > 1:
                    action = loaded['Action']
                    action_input = loaded['Action Input']

                    return AgentAction(
                        tool=action,
                        tool_input=action_input,
                        log=llm_output
                    )
                else:
                    print("\n NOTHING TO DO \n")
            except Exception as e:
                raise OutputParserException(f"Could not parse LLM output: {str(e)}")
        
        # If parsing fails, raise an exception
        raise OutputParserException(f"Could not parse LLM output: {llm_output}")

total_token_tracking = 0
total_requests_tracking = 0
retry_articles = [] # simple retry logic, if it fails it gets added here and retried
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
    print(f"Processing batches. batch size: {batch_size}")
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
    title: str,
    author: str,
    publication_date: str,
    content: str,
    url: str
) -> Dict[str, Any]:
    """Extract data from a single article"""
    global retry_articles
    tools = [company_name_crunchbase_search_async, duck_duck_go_search_async, wiki_search_async]
    tool_names = ", ".join([tool.name for tool in tools])
    
    llm = OllamaLLM(
        model="llama3.2", 
        base_url=llama_base_url, 
        temperature=0, 
        verbose=True,
        format="json"
        # num_ctx="24000" # more context if neede
    )
    article_object = {
                "title": title,
                "author": author,
                "publication_date": publication_ddumpsate,
                "content": content
            }

    template = f"""You are a helpful data processor and extractor. Pull the proper information for the fields in {schema}
    from a given article. Use the title, author, publication date, and content and output in the proper format.
    If you learn the company name you may research using {tools} name {tool_names} as needed.
    The output should only contain one valid response object that follows the schema. It's okay if you don't 
    find data for all the fields if you've found the required fields.
    IMPORTANT: Your final response MUST BE A VALID JSON OBJECT.
    You should return JSON ONLY in the schema format. Do not write code or anything additional.

    SCHEMA: 

    {schema}

    Begin!
    
    Title, Author, Publication Data, Content: {title}, {author}, {publication_date}, {content}
    """

    try:
        agent = create_conversational_retrieval_agent(
            tools=tools,
            llm=llm,
            output_parser=MyOutputParser()
        )

        result = await agent.ainvoke(template)

        if isinstance(result, dict) and 'output' in result:
            total_tokens_for_result = 0
            loaded_json = json.loads(result['output'])["properties"]
            loaded_json = loaded_json.get("properties") if loaded_json.get("properties") else loaded_json

            loaded_json.update({
                'url': url,
                'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                'uuid': str(uuid.uuid4())
            })

            return (loaded_json, total_tokens_for_result)
        else:
            raise ValueError(f"Unexpected response format from agent: {result}")
    except ValueError as e:
        error_string = str(e)
        print(result['output'])
        print(f"Value Error processing: {str(e)}")
        raise e
    except Exception as e:
        print(f"Error processing: {str(e)}")
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

    if not json_data_list:
        raise ValueError("No data was successfully processed")
    json_data_df = pd.DataFrame(json_data_list)
    json_data_df = json_data_df[~json_data_df['company_name'].astype(str).str.contains('type', case=False, na=False)]
    return json_data_df

def save_article_data_to_csv(data_frame: pd.DataFrame, filename: str) -> None:
    """Save processed data to CSV"""
    if not filename:
        raise ValueError('Need to provide a filename to save')
    data_frame.to_csv(filename, index=False, encoding='utf-8')
    print(f"Data enriched with funding rounds successfully saved to {filename}")

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