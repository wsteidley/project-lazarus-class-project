import { writeCsv } from '../lib/csv.js'
import { scrapeAllArticles, scrapeListingsWithPagination } from '../lib/scrape.js'
import { utcTimestamp } from '../lib/timestamp.js'

// step0: scrape a TechCrunch listing into an article index, then scrape each full
// article. Produces two CSVs sharing one timestamp so they can be related.
const main = async (): Promise<void> => {
  // Set a listing URL to scrape, e.g. https://techcrunch.com/tag/climate
  const url = process.env.SCRAPE_URL ?? ''
  if (!url) {
    throw new Error(
      'Set SCRAPE_URL to a TechCrunch listing URL, e.g. https://techcrunch.com/tag/climate',
    )
  }

  const timestamp = utcTimestamp()
  console.log(timestamp)

  const articleIndex = await scrapeListingsWithPagination(url, '', timestamp)
  await writeCsv(articleIndex, `techcrunch_article_${timestamp}_data_index.csv`)

  const articleUrls = articleIndex.map((entry) => entry.url).filter(Boolean)
  const articleData = await scrapeAllArticles(articleUrls, timestamp)
  await writeCsv(articleData, `techcrunch_article_${timestamp}_data.csv`)

  console.log('\nDONE\n')
}

main().catch((error) => {
  console.error(`Error in main execution: ${String(error)}`)
  process.exitCode = 1
})
