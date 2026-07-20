// Processes items in concurrent batches, logging elapsed time and a rough ETA.
// Ported from process_article_batch; failed items are dropped (like the Python
// version filtering out exceptions from asyncio.gather).
export const processInBatches = async <Input, Output>(
  items: Input[],
  handler: (item: Input) => Promise<Output>,
  batchSize: number,
): Promise<Output[]> => {
  const results: Output[] = []
  const totalBatches = Math.ceil(items.length / batchSize)
  const startTime = Date.now()

  console.log(`Processing ${items.length} items in batches of ${batchSize}`)

  for (let offset = 0; offset < items.length; offset += batchSize) {
    const batch = items.slice(offset, offset + batchSize)
    const batchResults = await Promise.allSettled(batch.map(handler))

    for (const settled of batchResults) {
      if (settled.status === 'fulfilled') {
        results.push(settled.value)
      } else {
        console.error(`Item failed: ${String(settled.reason)}`)
      }
    }

    const batchNumber = Math.floor(offset / batchSize) + 1
    const elapsedSeconds = (Date.now() - startTime) / 1000
    const averageSeconds = elapsedSeconds / batchNumber
    const remainingSeconds = averageSeconds * (totalBatches - batchNumber)
    console.log(
      `Processed batch ${batchNumber}/${totalBatches} — elapsed ${Math.round(elapsedSeconds)}s, ` +
        `est. ${Math.round(remainingSeconds / 60)} min remaining`,
    )
  }

  return results
}
