// UTC timestamp in the same second-resolution ISO shape the Python scripts used
// (e.g. 2026-07-20T13:45:02) so index files and output files can be related.
export const utcTimestamp = (): string => new Date().toISOString().replace(/\.\d{3}Z$/, '')

// Filename-safe UTC stamp (no colons) for run folders and scraped filenames, so
// paths are valid on every OS. Sorts chronologically. E.g. 2026-07-20T13-45-02Z.
export const runStamp = (): string =>
  new Date()
    .toISOString()
    .replace(/\.\d{3}Z$/, 'Z')
    .replace(/:/g, '-')
