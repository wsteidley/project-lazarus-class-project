// UTC timestamp in the same second-resolution ISO shape the Python scripts used
// (e.g. 2026-07-20T13:45:02) so index files and output files can be related.
export const utcTimestamp = (): string => new Date().toISOString().replace(/\.\d{3}Z$/, '')
