import { existsSync, rmSync } from 'node:fs'
import { outputBaseDir, scrapedDir } from '../config.js'

// Removes generated data (scrapes + run outputs) while leaving the curated
// data/input untouched. Cross-platform (no rm -rf).
const main = (): void => {
  for (const dir of [scrapedDir, outputBaseDir]) {
    if (existsSync(dir)) {
      rmSync(dir, { recursive: true, force: true })
      console.log(`Removed ${dir}`)
    } else {
      console.log(`(nothing to remove at ${dir})`)
    }
  }
  console.log('Curated input under data/input was left untouched.')
}

main()
