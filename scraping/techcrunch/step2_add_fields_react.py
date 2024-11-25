import os

from langchain_community.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents import create_react_agent, AgentExecutor, AgentOutputParser
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain_core.prompts import PromptTemplate

import pandas as pd
import json
from datetime import datetime, timezone
from response_format import response_format_funding_rounds, response_format_funding_rounds_llama
import math
from langchain.load.dump import dumps
import asyncio
import uuid
import time
from typing import List, Dict, Any, Union
from tools_search import duck_duck_go_search_async, wiki_search_async
from langchain_ollama import OllamaLLM
from langchain_core.tools import tool
import re
from langchain.schema import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

schema = response_format_funding_rounds_llama

llama_base_url = os.environ["LLAMA_BASE_URL"]

utc_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
print(utc_timestamp)

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
        # print(f"llm_output: \n {llm_output}\n")
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
                # print(action_split)
                print('--------------------')
                loaded = json.loads(llm_output)
                action_input_split = llm_output.split("Action Input", 1)
                if len(action_split) > 1 and len(action_input_split) > 1:
                    # action = action_split[1].split("\n")[0].strip()
                    # action_input = action_input_split[1].strip()
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
    print(f"Processing batches. batch size: {batch_size}")
    for i in range(0, article_len, batch_size):
        total_requests_tracking += batch_size
        batch = articles[i:i + batch_size]
        tasks = [
            extract_data_with_conversational_retrieval_agent(
                company_obj=company_obj
            )
            for company_obj in batch
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
        print(f"Estimated time remaining: {time_remaining}")
        print(f"Processed batch {numerator}/{denominator}")

    return results

async def extract_data_with_conversational_retrieval_agent(
    company_obj: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract data from a single article"""
    global retry_articles
    tools = [duck_duck_go_search_async, wiki_search_async]
    tool_names = ", ".join([tool.name for tool in tools])
    
    llm = OllamaLLM(
        model="llama3", 
        base_url=llama_base_url, 
        temperature=0, 
        verbose=True,
        format="json
        # num_ctx="24000" # more context 
    )

    template = """You are a helpful data aggregator, processor, and extractor. You will be given a JSON object with infromation about a company.
    The output should only contain one valid response object that follows the schema: {schema}.
    It's okay if you don't find data for all fields.

    You have access to these tools:

    {tools}

    Use the following format:

    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!
    
    Company Object: {company_obj}

    {agent_scratchpad}
    """

    try:
        prompt_template = PromptTemplate(
            template=template,
            input_variables=[
                "company_obj", 
                "agent_scratchpad",
                "tools",        # Added these two
                "tool_names"    # Required variables
            ],
            partial_variables={
                "schema": schema,
            }
        )

        agent = create_react_agent(
            tools=tools,
            llm=llm,
            prompt=prompt_template,
            output_parser=MyOutputParser()
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            tool_names=tool_names,
            max_iterations=10,
            max_execution_time=30
            # handle_parsing_errors=True
        )

        result = await agent_executor.ainvoke(
            {
                "company_obj": company_obj,
            }
        )
        if isinstance(result, dict) and 'output' in result:
            total_tokens_for_result = 0
            # This code was written to return chatgpt tokens used
            # It needs to be updated for llama
            # try:
            #     intermediate_steps = result['intermediate_steps']
            #     if len(intermediate_steps):
            #         inner_obj = intermediate_steps[0][0]
            #         message_log = json.loads(dumps(inner_obj))['kwargs']['message_log']
            #         for log in message_log:
            #             total_tokens_for_result += log['kwargs']['usage_metadata']['total_tokens']
            # except Exception as e:
            #     print(f"Error getting tokens: {str(e)}")

            # result['output'].replace("```json", "'")
            loaded_json = json.loads(result['output'])
            # Add metadata
            loaded_json.update({
                'company_name': company_obj['company_name'],
                'uuid': company_obj['uuid'],
            })
            return (loaded_json, total_tokens_for_result)
        else:
            raise ValueError(f"Unexpected response format from agent: {result}")
    except ValueError as e:
        error_string = str(e)
        # print(result)
        print(f"Value Error processing company_obj: {str(e)}")
        raise e
    except Exception as e:
        print(f"Error processing company_obj: {str(e)}")
        raise e

async def extract_data_for_file(filename: str, batch_size: int = 5) -> pd.DataFrame:
    """Process entire file with batched async operations"""
    article_data_df = pd.read_csv(filename)
    articles = article_data_df.to_dict('records')
    
    # Using the first 10 items rather than all
    limited_articles = articles[:10]
    print(f"Processing {len(limited_articles)} articles...")
    
    json_data_list = await process_article_batch(
        articles=limited_articles,
        batch_size=batch_size
    )

    # print(dumps(json_data_list, pretty=True))

    if len(retry_articles):
        retry_data_list = await process_article_batch(
            articles=retry_articles,
            batch_size=batch_size
        )
        json_data_list.extend(retry_data_list)

    if not json_data_list:
        raise ValueError("No data was successfully processed")

    for col in json_data_list[0].keys():
        if col not in article_data_df.columns:
            article_data_df[col] = None

    for data_obj in json_data_list:
        obj_id = data_obj['uuid']
        article_data_df.loc[article_data_df['uuid'] == obj_id, 'funding_rounds'] = json.dumps(data_obj['funding_rounds'])
    return article_data_df

def save_article_data_to_csv(data_frame: pd.DataFrame, filename: str) -> None:
    """Save processed data to CSV"""
    if not filename:
        raise ValueError('Need to provide a filename to save')
    data_frame.to_csv(filename, index=False, encoding='utf-8')
    print(f"Data enriched with funding rounds successfully saved to {filename}")

async def main():
    filename = ''
    if not filename:
        raise ValueError("Filename should be output file from step1")
    utc_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    try:
        funding_data_df = await extract_data_for_file(filename, batch_size=3)
        save_article_data_to_csv(
            funding_data_df, 
            f"added_funding_data_{utc_timestamp}.csv"
        )
        print("\nDONE\n")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    asyncio.run(main())