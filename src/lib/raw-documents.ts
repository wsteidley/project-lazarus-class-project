import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { cacheDir, config } from '../config.js'

// Why a document was fetched, which is also what decides how long it stays usable.
// 'discovery' text (launch articles) is immutable; the others are perishable.
export const SOURCE_TYPE = ['discovery', 'outcome', 'reassessment'] as const
export type SourceType = (typeof SOURCE_TYPE)[number]

export type RawDocument = {
  url: string
  url_hash: string
  fetched_at: string // ISO
  text: string
  source_type: SourceType
}

const MS_PER_DAY = 86_400_000

// Collapses URLs that address the same document: lowercased host, no query string,
// no fragment, no trailing slash. Shared on purpose — the cache key and any
// dedupe-on-concatenate must agree on what "the same URL" means, or one of them
// will double-count what the other merged.
export const normalizeUrl = (url: string): string => {
  const trimmed = url.trim()
  let parsed: URL
  try {
    parsed = new URL(trimmed)
  } catch {
    // Not a parseable URL — fall back to the trimmed string so callers still get a
    // stable key rather than an exception.
    return trimmed
  }
  parsed.hash = ''
  parsed.search = ''
  parsed.hostname = parsed.hostname.toLowerCase()
  parsed.protocol = parsed.protocol.toLowerCase()
  const normalized = parsed.toString()
  return normalized.endsWith('/') ? normalized.slice(0, -1) : normalized
}

export const urlHash = (url: string): string =>
  createHash('sha256').update(normalizeUrl(url)).digest('hex')

// Two-tier freshness. Discovery text never expires; perishable types age out after
// ttlDays so the outcome picture can be refreshed without wiping the whole cache.
export const isFresh = (
  document: Pick<RawDocument, 'fetched_at' | 'source_type'>,
  now: Date = new Date(),
  ttlDays: number = config.cacheTtlDays,
): boolean => {
  if (document.source_type === 'discovery') {
    return true
  }
  const fetchedAt = new Date(document.fetched_at).getTime()
  if (Number.isNaN(fetchedAt)) {
    return false
  }
  return now.getTime() - fetchedAt < ttlDays * MS_PER_DAY
}

const documentPath = (hash: string): string => join(cacheDir, `${hash}.json`)

export const readCachedDocument = async (url: string): Promise<RawDocument | null> => {
  const path = documentPath(urlHash(url))
  if (!existsSync(path)) {
    return null
  }
  try {
    return JSON.parse(await readFile(path, 'utf-8')) as RawDocument
  } catch {
    // A truncated or hand-edited cache entry should cost a refetch, not a crash.
    return null
  }
}

export const writeCachedDocument = async (
  url: string,
  text: string,
  sourceType: SourceType,
): Promise<RawDocument> => {
  await mkdir(cacheDir, { recursive: true })
  const document: RawDocument = {
    url,
    url_hash: urlHash(url),
    fetched_at: new Date().toISOString(),
    text,
    source_type: sourceType,
  }
  await writeFile(documentPath(document.url_hash), JSON.stringify(document), 'utf-8')
  return document
}

// Serves `url` from the cache when a fresh entry exists, otherwise calls `fetcher`
// and stores the result. This is the single entry point every fetch should use.
export const fetchWithCache = async (
  url: string,
  sourceType: SourceType,
  fetcher: (url: string) => Promise<string>,
): Promise<string> => {
  const cached = await readCachedDocument(url)
  if (cached && isFresh(cached)) {
    return cached.text
  }
  const text = await fetcher(url)
  await writeCachedDocument(url, text, sourceType)
  return text
}

// Every cached document, for loading the `raw_documents` table at build time.
export const readAllCachedDocuments = async (): Promise<RawDocument[]> => {
  if (!existsSync(cacheDir)) {
    return []
  }
  const files = (await readdir(cacheDir)).filter((name) => name.endsWith('.json'))
  const documents: RawDocument[] = []
  for (const file of files) {
    try {
      documents.push(JSON.parse(await readFile(join(cacheDir, file), 'utf-8')) as RawDocument)
    } catch {
      console.warn(`Skipping unreadable cache entry ${file}`)
    }
  }
  return documents
}
