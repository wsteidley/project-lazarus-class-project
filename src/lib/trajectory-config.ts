// Tunables for the trajectory/projection derivation, seeded into the one-row
// `trajectory_config` table at build so retuning is an UPDATE, not a query edit. These are
// the values the v2 spec flags as still open; they ship as defaults until the longest series
// (batteries, solar) have enough points to check whether slope alone classifies cleanly.
export const trajectoryConfig = {
  // Trailing observations used to measure the recent slope.
  windowN: 3,
  // |slope| (progress-per-year) at or below which a series counts as plateaued rather than
  // improving/receding.
  plateauSlopeThreshold: 0.03,
}
