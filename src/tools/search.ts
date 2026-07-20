import { DuckDuckGoSearch } from '@langchain/community/tools/duckduckgo_search'
import { WikipediaQueryRun } from '@langchain/community/tools/wikipedia_query_run'

// Web-search tools the funding agent can call. LangChain's community tools
// replace the hand-rolled @tool wrappers from tools_search.py.
export const duckDuckGoSearch = new DuckDuckGoSearch({ maxResults: 4 })

export const wikipediaSearch = new WikipediaQueryRun({
  topKResults: 3,
  maxDocContentLength: 4000,
})
