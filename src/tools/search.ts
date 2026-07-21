import { DuckDuckGoSearch } from '@langchain/community/tools/duckduckgo_search'
import { WikipediaQueryRun } from '@langchain/community/tools/wikipedia_query_run'
import { TavilySearch } from '@langchain/tavily'
import { config } from '../config.js'

// Tiered web search: Tavily primary, DuckDuckGo fallback. DuckDuckGo alone is
// rate-limited and low-precision, but it costs nothing and needs no key — so it stays
// as the floor rather than being replaced. No single point of failure in the path.

export const duckDuckGoSearch = new DuckDuckGoSearch({ maxResults: 4 })

export const wikipediaSearch = new WikipediaQueryRun({
  topKResults: 3,
  maxDocContentLength: 4000,
})

// Built only when a key exists; an absent key skips the tier rather than throwing, so
// a missing secret costs recall, not a broken run.
const tavilySearch = config.tavilyApiKey
  ? new TavilySearch({ maxResults: 5, tavilyApiKey: config.tavilyApiKey })
  : null

// One Tavily result. Structured, unlike DuckDuckGo's plain string — the two shapes are
// normalized to text below so callers see a single contract.
type TavilyResult = { title?: string; url?: string; content?: string }
type TavilyResponse = { results?: TavilyResult[]; answer?: string }

// Renders Tavily's structured response as prompt-ready text, keeping each result's URL
// beside its content so extracted facts can carry their own source_url.
export const formatTavilyResponse = (response: TavilyResponse | string): string => {
  if (typeof response === 'string') {
    return response.trim()
  }
  const parts: string[] = []
  if (response.answer?.trim()) {
    parts.push(`Summary: ${response.answer.trim()}`)
  }
  for (const result of response.results ?? []) {
    const line = [result.title?.trim(), result.url?.trim(), result.content?.trim()]
      .filter(Boolean)
      .join('\n')
    if (line) {
      parts.push(line)
    }
  }
  return parts.join('\n\n').trim()
}

// Whether to drop to the next tier. Any error, and any empty/whitespace result, both
// count — a provider returning nothing is as useless as one that throws, and
// rate-limiting surfaces as one or the other.
export const shouldFallBack = (text: string | null | undefined, error?: unknown): boolean =>
  error !== undefined || !text || !text.trim()

// Best-effort tiered search. Returns text from the first tier that produces any, or an
// empty string if none do — callers get a value, never an exception.
export const searchWeb = async (query: string): Promise<string> => {
  if (tavilySearch) {
    let text: string | null = null
    let error: unknown
    try {
      text = formatTavilyResponse((await tavilySearch.invoke({ query })) as TavilyResponse)
    } catch (caught) {
      error = caught
    }
    if (!shouldFallBack(text, error)) {
      return text ?? ''
    }
    console.warn(`Tavily returned nothing for "${query}" — falling back to DuckDuckGo`)
  }

  try {
    const text = await duckDuckGoSearch.invoke(query)
    return typeof text === 'string' ? text : String(text)
  } catch {
    return ''
  }
}

// The standard retrieval context for an enrichment pass: tiered web search plus
// Wikipedia, which is strong for acquisitions, shutdowns, and founding facts. Each
// provider degrades to empty independently, so one dead source never fails a row.
export const gatherSearchContext = async (
  query: string,
  wikipediaTerm: string,
): Promise<string> => {
  const [webResult, wikiResult] = await Promise.all([
    searchWeb(query).catch(() => ''),
    wikipediaSearch.invoke(wikipediaTerm).catch(() => ''),
  ])
  return `Web search results:\n${webResult}\n\nWikipedia results:\n${wikiResult}`
}
