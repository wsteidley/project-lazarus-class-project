# Project Lazarus — Metric Data Baseline Spec (LIVING DOCUMENT)

**Status: living.** Update this whenever a new metric series is sourced or a new
methodology gotcha is found. It is the standing reference that informs every future
spec and every model/agent touching cost, capacity, or threshold data. If a data
question isn't answered here, answer it *and add it here*.

Purpose: cost curves, capacity series, and thresholds are only comparable if they
share a **basis** and a consistent **methodology**. Two production bugs have already
come from ignoring this (nominal-vs-real solar cost; DC-vs-AC solar capacity). This
doc exists so the third doesn't happen.

---

## The three rules that override everything

1. **One basis per series, and the threshold must share it.** A series is keyed
   `(technology, metric, scope, basis, energy_basis, duration, segment)`. Never mix
   these within one series; never compare an observation to a threshold that differs on
   currency **or** energy_basis **or** duration. Validate at load; a mismatch should be a
   build-time error, not a warning. **⚠ NOT YET ENFORCED:** `dependency_thresholds` has
   no basis columns and the progress join ignores basis — see the tracker's Still-open
   "enforce threshold basis-match". This rule is the target state, not current behaviour.
2. **Prefer one source end-to-end over stitched anchors.** A learning-curve fit reads
   the *slope across doublings*; a series that switches methodology partway (e.g.
   conservative early, generous late) bends the curve and biases the rate. A single
   consistent source with more points beats a "better" latest number from a different
   methodology.
3. **`as_of` is the year the number describes, not the publication year.** Reports are
   published the year *after* the data year (BNEF 2025 survey → 2024 data; IRENA 2025
   report → 2024 costs). Getting this wrong shifts crossing dates.

## Column model (three orthogonal axes — do not conflate)

`basis` used to carry three unrelated concepts; it's now split. Each observation
declares all three, `na` where not applicable. **DC/AC and usable/nameplate are
`energy_basis`; onshore/offshore and on/off-grid are `segment`; GWh-vs-GW is just the
`unit`. None of those belong in `basis`.**

| Column | Meaning | Values |
| --- | --- | --- |
| `basis` | currency only | `real_2024_usd`, `real_2025_usd`, `nominal_usd`, `na` |
| `energy_basis` | how the per-energy denominator is defined | `nameplate`, `usable` (battery); `AC`, `DC` (solar); `na` |
| `duration` | storage duration | `4h`, `2h`, `blended`, `na` |
| `segment` | sliceable sub-series | `all`, `onshore`, `offshore`, `on_grid`, `off_grid` |
| `unit` | what the value measures | `USD/W`, `USD/MWh`, `USD/kWh`, `GW`, `GWh`, `years`, `percent` |

Gotchas that motivated the split: DC vs AC (solar capacity, ~15–25% gap — `energy_basis`,
never mix); usable vs nameplate (battery — `energy_basis`); `GWh` vs `GW` is a **unit**
difference (the old `energy`/`power` "basis" was mis-filed — it's just `unit`, with
basis/energy_basis/duration all `na`).

---

## Per-technology guidance

### Solar PV
- **Cost spine:** OWID `solar-pv-prices` (constant 2024 USD/W). Full annual 1975–2024.
- **Capacity spine:** OWID `installed-solar-pv-capacity` (AC basis, IRENA/Ember).
  Full annual 2000–2024. **Now loaded — replaced the earlier 9-anchor stitch.**
- **GOTCHA — DC vs AC (cost you a 20% error).** REN21/GSR/Statista headline figures
  (2,247 GW for 2024) are **DC/panel** and include latest China; OWID/IRENA
  (1,866 GW for 2024) is **AC/grid**, more conservative and consistent. They agree
  <6% through 2020, then diverge to ~20% by 2024. **Use the AC series wholesale.**
  Same-source-as-cost (both OWID) is the deciding factor.
- **GOTCHA — cost basis.** OWID cost is real-2024; pvXchange spot is nominal. On the
  real spine, 2024 ≈ $0.258/W → **not** across a $0.20 bar; nominal spot is below it.
  The crossing verdict flips on basis. Real spine is canonical; pvXchange is a
  separate metric if kept.
- **Cross-check (not a target):** published solar learning rates cluster at ~20–25%/
  doubling; the fitted rate on this spine is **27.7% (R²=0.955)**. Slightly above, and
  expected — the OWID series starts in 1975, so the fit spans the steep early decades that
  most published estimates exclude. A rate far *below* ~20% would be the alarm.

### Wind (onshore)
- **Cost — two series, different roles:** LCOE (IRENA RPGC Fig S.2, **16 pts 2010–2025**,
  `real_2025_usd`) is **viability/threshold only — NOT Wright-fittable** (LCOE captures
  capacity-factor + financing gains, not manufacturing learning; fitting it gave a false
  43%/doubling). The Wright-fittable series is `total_installed_cost` ($/kW) — **RESOLVED:
  16 pts 2010–2025 from Fig 2.3** (`segment=onshore`), read straight from the committed
  `.xlsx`. **Wind fits at 25.0%/doubling, R²=0.85 over 15 pairs** (2025 cost has no
  capacity point to pair with — capacity ends 2024).
- **The 43%-vs-25% contrast is the proof of the LCOE rule.** Same technology, same capacity
  axis, two cost series: the delivered-energy one gives a physically implausible rate, the
  hardware one lands in the published band. If a fitted rate ever looks too good, check
  *which cost series* it was fitted on before anything else.
- **GOTCHA — figure numbers move between report editions.** This series is Fig 2.3 in the
  2025 file and Fig 2.1 in the 2023 one. Bind extractors to the sheet *title*, never the
  number — and locate rows by label, not index.
- **Capacity:** IRENA onshore-only annual, 2000–2024 (`segment=onshore`), loaded.
  **RESOLVED** — replaced the earlier GWEC total-basis anchors.
- **GOTCHA — offshore contamination (historical):** GWEC's 1,136 GW (2024) *included*
  ~83 GW offshore; the IRENA onshore series avoids this. Keep wind capacity `onshore`.

### Lithium-ion battery
- **Cost — FOUR non-interchangeable metrics** ($70–140/kWh, same year — never merge):
  BNEF pack ($108 all-seg / $99 BEV / $70 stationary, cited anchor); BNEF turnkey
  ($110 4h / $124 2h, duration-specific); **IRENA TIC** (`battery_installed_cost`,
  `energy_basis=usable`, `duration=blended`, **16-pt series 2010–2025**, loaded); Ember
  all-in ($125, Oct-2025 snapshot). The old "4h system cost 140" was mislabeled IRENA TIC
  → merged/relabeled. Pack sub-metrics are **one metric, three `segment`s, one bar** (not
  three metrics): all-segment above the 100 bar, BEV crossed 2024, stationary 2025.
- **Capacity:** BESS additions (IRENA/BNEF, **11 pts 2015–2025**) cumulated → the Wright
  axis. Battery now fits (**23.5%/doubling, R²=0.90**) — reads slightly high because the
  axis cumulates from 2015 only, understating early deployment. The published-20%-rate
  shortcut is **RETIRED** (a real fit now exists).
- **GOTCHA — manufacturing capacity ≠ cumulative production.** IEA's "4 TWh by 2025"
  is *annual nameplate manufacturing capacity* — a different quantity. Do not put it
  in a cumulative-GWh series.
- **Cross-check (not a shortcut):** the published rate — Ziegler & Trancik
  (arXiv 2007.13920): **20%/doubling all-cell, 24% cylindrical, 1992–2016** — is the
  **validation anchor** for the fitted 23.5%. That the fit lands near-but-above the
  published rate is what tells you 23.5% is *high-but-plausible* (the 2015-truncated
  capacity axis, not a broken fit) rather than wrong. A fit far from ~20% would be a
  data-quality alarm. (Previously proposed as a fallback rate; now that a real fit
  exists it serves as the sanity check on it, not a substitute for it.)

### Carbon price
- Current only (World Bank State & Trends). **Regional, not global scalar:** ~$21
  global vs ~$170 Norway — separate `scope` rows. `above_is_better`. Historical EU-ETS
  annual is the backfill target.

### Green hydrogen / DAC / electrolyzer
- Current-only, young metrics, thin history — and **policy-dependent** (cross only
  under a support regime → `conditional` trajectory state, not `tested_dead_end`).
- DAC threshold is **contested** ($100 shorthand vs ~$300 literature) — carry both.

### Capital (growth / early-stage / FOAK)
- **Name the pool in the metric** — VC / public equity / total equity / debt /
  project finance. "Climate equity +53%" and "VC down 3 years" are simultaneously
  true; a failed startup faced a *specific* pool. Rulings: growth → growth VC/PE;
  early-stage → seed+A VC; FOAK → project finance/debt (not equity).
- These three **never roll up** — they moved in opposite directions.

---

## Sourcing playbook (the OWID-download pattern that works)

1. Prefer a **downloadable CSV** with a full annual series (OWID graphers, IRENA
   Renewable Capacity Statistics, LBNL data files) over headline figures in reports.
2. OWID is the pragmatic spine for solar/wind cost and capacity — it aggregates
   IRENA/Ember/Nemet and ships a CSV + readme with citation.
3. Point each row's `source_url` at the **specific series/report**, not a homepage.
4. Paywalled series (BNEF price survey, Ziegler-Trancik supplement) → cite by name,
   blank URL, mark as anchor.
5. Record every value's `as_of` as the **data year**, and note the basis.

## Known-thin series (backfill targets, in priority order)

1. Carbon price history — EU-ETS annual (still current-only). **Highest.**
2. `total_installed_cost` for the other six technologies (solar, CSP, geothermal, hydro,
   offshore wind, bioenergy) — 2025 snapshots only; the annual series are in their chapter
   tables. Blocked on the technology/dependency split, since those technologies have no
   dependency to load against.
3. Battery cost gaps 2011–17 / 2019–20 — only if a non-paywalled compilation appears.
4. **DONE:** wind LCOE (16 pts), wind capacity onshore (IRENA), **wind installed cost
   (16 pts, Fig 2.3 — wind's learning rate is no longer blocked)**, battery installed cost
   + BESS additions, solar capacity (OWID AC full series).

## Changelog (append every update)

- **2026-07-25** — Created. Solar capacity switched from 9 stitched anchors to full
  OWID AC series (2000–2024); logged the DC-vs-AC gotcha (~20% divergence post-2021).
  Added `basis` guidance per technology. Solar cost real-vs-nominal and battery
  cumulative-GWh gotchas carried in from v3/v3.1 specs.
- **2026-07-25 (later)** — Wind capacity RESOLVED: replaced GWEC total-basis anchors
  with full IRENA onshore-only series (2000–2024, 16.9→1049.8 GW) from the IRENA Stats
  Tool `.xlsb`; fixes offshore contamination. Solar OWID vs IRENA cross-validated to
  <0.4%. Logged IRENA extraction gotcha: capacity lives on `On-grid`/`Off-grid
  electricity` producer-type rows; `All types` rows carry only finance and have null
  capacity — sum on/off-grid, exclude `All types`.
- **2026-07-25 (data-reorg)** — Metric data made reproducible: `data/{sources,curated,
  derived}` split; `build-metric-data` step generates the derived CSVs from raw
  sources + cited anchors; audit = `rm -f data/derived/*.csv && build && git diff`.
  **uv-absent contract:** fail loud and non-zero by default (never skip implicitly);
  `--allow-stale-metric-data` is the opt-in override that proceeds on the existing
  committed file, and even then a missing file fails rather than writing a partial. The
  audit loop requires uv.
- **2026-07-26 (IRENA cost + battery reconcile)** — Pulled full IRENA RPGC 2025 metrics
  (LCOE 7 techs, battery installed cost, BESS additions). Key lessons logged:
  - **`basis` split into three orthogonal columns** — `basis` (currency only),
    `energy_basis` (nameplate/usable, AC/DC), `duration` (4h/2h/blended). DC/AC moved
    from `basis` to `energy_basis`. One column was carrying three unrelated things.
  - **Battery has FOUR non-interchangeable cost metrics** ($70–140/kWh same year): BNEF
    pack, BNEF turnkey (duration-specific), IRENA TIC (usable/blended), Ember all-in.
    Never merge across them.
  - **Label-suspicion:** IRENA battery cost is `usable`-kWh, `blended`-duration — any
    point labeled plain "4h" is suspect. The old curated "4h system cost 140" was
    actually IRENA TIC, mislabeled. Third time a basis/label proved wrong (after DC/AC,
    nominal/real): don't trust a source's implied label — verify the caption.
  - **LCOE is not Wright-fittable** (viability-side). Wind's learning rate is blocked
    pending the annual wind `total_installed_cost` series (in IRENA RPGC chapters,
    unextracted). Fitting wind LCOE gave a false 43%/doubling — refused.
  - **`segment`/join keys are `NOT NULL DEFAULT`** — a NULL equality key silently drops
    the row from the trajectory self-join.
- **2026-07-27 (wind fitted; two hard lessons)** — Wind `total_installed_cost` extracted
  from IRENA RPGC 2025 Fig 2.3 (`real_2025_usd`, 2010–2025); wind now fits at **25.0%/
  doubling** (R²=0.847), vs the false 43% from LCOE. All three heroes (solar 27.7%,
  wind 25.0%, battery 23.5%) now fit from committed open sources. Two standing lessons:
  - **NEVER hand-parse a structured/binary format (xlsx, xlsb, etc.) — use a trusted
    library** (openpyxl). A first regex-over-the-zip read let a shared-string index
    through as a value, shifted every year one column, and returned a confident
    **−87.6%/doubling that looked like real data**. Hand-parsing fails *silently with a
    plausible wrong number*, not with an error — disqualifying in a project whose whole
    discipline is "don't ship confidently-wrong data." Same species as label-suspicion:
    the dangerous failures look valid. Defenses: read by label not index, bind to sheet
    *title* not figure number (editions renumber — Fig 2.1 in 2023 → Fig 2.3 in 2025),
    refuse a workbook not in the expected currency basis.
  - **Wright fit must NOT require a threshold.** A learning rate needs no viability bar;
    only the crossing *projection* does. The progress view inner-joined thresholds, so a
    bar-less series (like wind TIC) silently returned zero rows — no fit, no error. Fix:
    Wright reads `metric_observations` with the bar LEFT JOINed. Consequence: a bar-less
    series has no curated direction, so a rising series is caught by the negative-slope
    guard rather than a label — weaker but sound. This unblocks the whole capex-curve
    family (per-tech `total_installed_cost`), all bar-less at first.
- **2026-08-02 (doc reconciliation)** — Brought the body of this doc in line with the entry
  above; the changelog had been updated while the sections a reader hits *first* still said
  wind was blocked. Changes: Wind section flipped to fitted (25.0%, R²=0.85, 15 pairs, Fig
  2.3) with the 43%-vs-25% contrast and the edition-renumbering gotcha promoted into the
  guidance; wind removed from the known-thin list, replaced at #2 by the other six
  technologies' `total_installed_cost` (2025 snapshots, blocked on the technology/dependency
  split). One standing lesson:
  - **Update the guidance, not just the changelog.** This is the second time the two
    diverged. The per-technology sections are what a reader — human or agent — acts on;
    a changelog entry recording a resolution does not stop stale guidance above it from
    being believed. A resolution isn't logged until the body says so too.
  - **A sanity band must not condemn the shipped number.** The solar check read "should land
    ~20–25%/doubling" while the shipped solar fit is 27.7% — the doc's own guardrail flagged
    the doc's own canonical value. Reframed as a cross-check with the reason for the gap
    (the OWID spine starts in 1975 and includes the steep early decades most published
    estimates omit), matching how the battery entry already uses Ziegler-Trancik.
