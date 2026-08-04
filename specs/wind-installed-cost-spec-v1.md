# Project Lazarus — Wind Installed-Cost Extraction & Fit Spec (v1)

Unblocks wind's learning rate. The Wright-fittable wind series
(`total_installed_cost`, $/kW) exists in a file **already committed** —
`IRENA_TEC_RPGC_in_2025_data_file_2026.xlsx`, **Fig 2.3** — so this is an extractor
addition, not a new source. Updates `data-derivation-map.md`, `metric-data-baseline-spec.md`,
and the tracker (wind flips blocked → fittable).

---

## Why this exists

Wind LCOE (16 pts) is **viability-side, not Wright-fittable** — fitting it gave a false
43%/doubling because LCOE folds in capacity-factor and financing gains that aren't
manufacturing learning. The fittable series is `total_installed_cost` ($/kW). It was a
single 2025 point; the full 2010–2025 series is in Fig 2.3 of the committed 2025 file.

**Validation (why this is the right series):** fitting the TIC series against cumulative
onshore-wind capacity yields **~25%/doubling** — in the plausible band (published
onshore wind ~15–20%, so slightly high but sane), versus the physically-impossible 43%
from LCOE. Same technology, correct hardware-cost series, sane rate. That contrast is
the proof of the "LCOE-is-not-Wright" rule.

> *(An earlier ~22–23% figure in drafts of this spec was a back-of-envelope sighting
> shot fit against rough hand-typed capacity anchors, only to confirm "sane, not 43%."
> The real pipeline fit against the loaded IRENA capacity series is **25.0%, R²=0.847,
> 15 pairs** — that's the number; 22–23% was never a target.)*

## Source & location

| field | value |
|---|---|
| file | `data/sources/irena/IRENA_TEC_RPGC_in_2025_data_file_2026.xlsx` (already committed) |
| sheet | `Fig 2.3` ("TICs of onshore wind projects and global weighted-average, 2010–2025") |
| row | "Weighted average" (row 7); year header row 5 |
| ⚠ note | figure numbering differs between report editions — in the **2023** file this series is Fig 2.1; in the **2025** file it's **Fig 2.3**. Bind to the sheet *title*, not the number. |

## Emitted rows (the split columns)

| metric | unit | basis | energy_basis | duration | segment | scope | years |
|---|---|---|---|---|---|---|---|
| `total_installed_cost` | USD/kW | `real_2025_usd` | `na` | `na` | `onshore` | `global` | 2010–2025 |

- **Basis is `real_2025_usd`** — consistent with the LCOE and battery series from the
  same file. (The 2023 file's version is `real_2023_usd`; do **not** mix — use the 2025
  file so all IRENA cost series share one currency basis.)
- Weighted-average row only for the series; the 5th/95th-percentile rows are a *range*,
  not part of the fit — capture as a separate `note` or skip for now.
- Seed the `technology_metrics` lookup with `(Onshore wind, total_installed_cost)` or the
  FK rejects the rows.

## Fit behaviour

- `total_installed_cost` is a **hardware/capex** series → **Wright-fittable** (gates on
  derived `curve_type='learning'`).
- Pair with the onshore-wind `cumulative_capacity` (GW) series already loaded, matched on
  year. Measured **25.0%/doubling** (R²=0.847, 15 pairs) against the loaded capacity series.
- Wind LCOE stays **viability/threshold only** — unchanged. Two wind series, two roles:
  TIC → Wright/learning; LCOE → crossing/viability. Never fit LCOE.

## Threshold / basis-match caution

If a wind viability bar is set against `total_installed_cost`, it must carry the same
`basis=real_2025_usd`/`energy_basis=na`/`duration=na` — this is exactly the mismatch the
(pending) threshold-enforcement rule must catch. Set it in 2025 dollars from the start.

## Doc updates this triggers

- **derivation map:** `total_installed_cost` row changes from "single 2025 point" to the
  Fig 2.3 series 2010–2025; note the edition figure-number shift.
- **baseline spec:** wind section — flip "no learning rate until extracted" to "fittable,
  25.0%/doubling from Fig 2.3 TIC"; move wind off the top of the known-thin backfill.
- **tracker:** move "wind learning rate blocked on data" from Still-open Active work to
  Recently-resolved (once the extractor lands).

## Still open

- **RESOLVED:** Fig 2.3 carries 2025 (col R = 976.01) — the series is 16 points,
  2010–2025. The fit uses **15 pairs**: 2025 cost has no capacity point, because the
  onshore capacity series ends 2024.
- Same `total_installed_cost` series exists for the **other techs** in their chapters
  (solar, CSP, geothermal, hydro, offshore) — the capex-curve companion to LCOE. Pull
  per-tech later; this spec covers onshore wind (the current blocker) only.
