import * as cheerio from 'cheerio'
import { utcTimestamp } from './timestamp.js'

export type ArticleIndexEntry = {
  title: string
  url: string
  author: string
  publication_date: string
  timestamp: string
}

export type ArticleData = {
  title: string
  author: string
  publication_date: string
  content: string
  timestamp: string
  url: string
}

const fetchHtml = async (url: string): Promise<string> => {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to retrieve ${url}. Status code: ${response.status}`)
  }
  return response.text()
}

// Parses listing HTML into article index entries. Split out from the network
// fetch so it can be unit-tested against a saved fixture. Uses the same CSS
// selectors as scrape_techcrunch_listings in step0_scrape_article_listings.py.
export const parseListingsHtml = (html: string, timestamp: string): ArticleIndexEntry[] => {
  const $ = cheerio.load(html)
  const entries: ArticleIndexEntry[] = []

  $('ul.wp-block-post-template > li.wp-block-post').each((_index, element) => {
    const article = $(element)
    const titleElement = article.find('h3.loop-card__title').first()
    const title = titleElement.text().trim()
    const articleUrl = titleElement.find('a').attr('href') ?? ''
    const author = article.find('a.loop-card__author').first().text().trim()
    const publicationDate = article.find('time').first().attr('datetime') ?? ''

    if (title || articleUrl) {
      entries.push({
        title,
        url: articleUrl,
        author,
        publication_date: publicationDate,
        timestamp,
      })
    }
  })

  return entries
}

// Fetches a TechCrunch listing page and parses it into article index entries.
export const scrapeListings = async (
  url: string,
  timestamp: string,
): Promise<ArticleIndexEntry[]> => {
  const html = await fetchHtml(url)
  return parseListingsHtml(html, timestamp)
}

// Only scrapes a single page for now; loop while page < pageLimit to paginate.
export const scrapeListingsWithPagination = async (
  baseUrl: string,
  searchTerm: string,
  timestamp: string,
  pageLimit = 1,
): Promise<ArticleIndexEntry[]> => {
  const allEntries: ArticleIndexEntry[] = []
  const searchString = searchTerm ? `?s=${searchTerm}` : ''

  for (let page = 0; page < pageLimit; page += 1) {
    const url = `${baseUrl}/page/${page}/${searchString}`
    console.log(`Scraping page ${page + 1} from url: ${url}`)
    const pageEntries = await scrapeListings(url, timestamp)
    if (pageEntries.length === 0) {
      console.log(`No data from page ${page + 1}, stopping`)
      break
    }
    allEntries.push(...pageEntries)
  }

  console.log(`Length of all data: ${allEntries.length}`)
  return allEntries
}

// Parses a single TechCrunch article. Mirrors scrape_techcrunch_article.
export const scrapeArticle = async (url: string, timestamp: string): Promise<ArticleData> => {
  const html = await fetchHtml(url)
  const $ = cheerio.load(html)

  const title = $('h1.wp-block-post-title').first().text().trim()
  const authorBlock = $('div.post-authors-list__authors').length
    ? $('div.post-authors-list__authors')
    : $('div.article-hero__authors')
  const author = authorBlock.find('a').first().text().trim()
  const publicationDate = $('div.wp-block-post-date > time').first().attr('datetime') ?? ''
  const content = $('div.wp-block-post-content').first().text().trim()

  return {
    title,
    author,
    publication_date: publicationDate,
    content,
    timestamp,
    url,
  }
}

// Non-article TechCrunch URL path segments. These pages (podcasts, videos,
// events, tag/category listings) use different templates than standard articles,
// so the article selectors don't match and they yield junk rows downstream.
const nonArticlePathSegments = ['/podcast/', '/video/', '/events/', '/tag/', '/category/']

// TODO: Decide how to handle non-article URLs long-term instead of dropping them.
// Options to consider: (a) route them to template-specific scrapers (podcasts and
// videos have their own useful metadata), (b) keep dropping but record the skipped
// URLs to a sidecar file for auditing, or (c) make the filter configurable so a run
// can opt into including them. For now we simply skip them.
export const isLikelyArticleUrl = (url: string): boolean => {
  if (!url.startsWith('https://')) {
    return false
  }
  const lowerUrl = url.toLowerCase()
  return !nonArticlePathSegments.some((segment) => lowerUrl.includes(segment))
}

// Scrapes every article URL, skipping non-article URLs and ones that error.
// Mirrors scrape_all_article_data.
export const scrapeAllArticles = async (
  articleUrls: string[],
  timestamp: string,
): Promise<ArticleData[]> => {
  const filteredUrls = articleUrls.filter((url) => {
    const keep = isLikelyArticleUrl(url)
    if (!keep) {
      console.log(`Skipping non-article URL: ${url}`)
    }
    return keep
  })

  const totalUrls = filteredUrls.length
  const allArticleData: ArticleData[] = []

  for (const [index, url] of filteredUrls.entries()) {
    try {
      console.log(`${index}/${totalUrls}: ${url}`)
      allArticleData.push(await scrapeArticle(url, timestamp))
    } catch (error) {
      console.error(`Exception for url: ${url}. Exception: ${String(error)}`)
    }
  }

  return allArticleData
}

export { utcTimestamp }
