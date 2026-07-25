# Project Lazarus — Thresholds & Progress Spec (v3.1)

Patch to v3, from the BNEF NEO 2025 and ETIT 2026 reports. **No change to the
observation values in `metric_observations_full.csv`** — those reports are investment
and scenario documents, not cost-series tables, so nothing supersedes the OWID/BNEF
price anchors. Two additions and one caution.

---

## Add 1 — `capacity_series.csv` (unlocks the deferred Wright fit)

v3 deferred Wright's-law projection pending "a cumulative-capacity series per curve
metric." NEO 2025 supplies it as citeable ETS trajectory points. Delivered as
`capacity_series.csv` (7 seed rows).

- Feeds Fix 5 (Wright's law): cost falls a fixed % per **doubling of cumulative
  capacity**, so the projection needs the capacity doubling schedule, not just time.
- **All rows are PROJECTED (ETS scenario), flagged `scenario=ETS`** — they are a
  forecast capacity path, not history. The Wright fit must treat them as scenario
  input and carry that into the projection's confidence, distinct from the observed
  historical cost points.
- Seeds: solar 6.9 TW and wind 2.6 TW cumulative additions by 2035; US battery
  storage 29 GW (2024) → 175 GW (2035); EV fleet 17.2M (2024) → 42M (2030) → 80M
  (2050) as a proxy for cumulative battery demand.
- Note: these are BNEF-report figures without a public URL; `source_url` is blank and
  `source_name` cites the report. Treat like the paywalled BNEF price anchors —
  cited, not machine-pulled.

## Add 2 — `conditional` trajectory state (the ETS-vs-NZS signal)

v3's states (`improving` / `plateaued` / `receded` / crossed) are all
**economics-only**. NEO draws a sharp, repeated line the typology doesn't capture: a
named set of technologies — **hydrogen, CCS, DAC, sustainable aviation fuel,
low-carbon industrial heat** — do *not* cross on economics alone but *do* under a
policy-support scenario. That is a distinct Lazarus signal.

- Add trajectory state **`conditional`** (a.k.a. `policy_dependent`): the metric has
  not crossed and is not projected to cross its viability bar on economics alone, but
  crosses under a support regime (subsidy, mandate, carbon price).
- For a failed company, `conditional` means: **revivable, but only if the policy
  regime it needed now exists** — different from "the cost curve fixed it"
  (`lazarus_candidate`) and more accurate for hydrogen/CCS-era failures.
- Implementation: a dependency flagged `policy_dependent=1` in
  `dependency_thresholds` whose economics-only progress is `<1` resolves to
  `conditional` rather than `tested_dead_end`. Optionally carry a second bar
  (`threshold_alt` already exists for contested — reuse the pattern for the
  policy-scenario bar) so "crossed under policy" is computable.
- Gap typology gains the matching cell: `conditional` sits between
  `tested_dead_end` (blocker never moved) and `lazarus_candidate` (blocker crossed on
  its own).

## Caution — capital dependencies are basis-fragile (name the pool)

ETIT 2026 shows climate-tech equity at **$77.3B, up 53% in 2025** — but that rebound
came from **public markets and Asian megadeals while venture funding fell for the
third straight year**. "Total equity up" and "VC down" are simultaneously true.

- The three capital dependencies (growth / early-stage / FOAK) must each **name which
  capital pool** in the `metric` field — `VC`, `public equity`, `total equity`, `debt`,
  `project finance` — or a failed startup's "was capital available?" gets answered by a
  BYD secondary offering, which is meaningless to it.
- This is the same one-basis-per-series discipline as Fix 2, applied to capital:
  pool is part of the metric identity, not a note. Update the capital rows in
  `metric_observations_full.csv` to carry the pool explicitly before they feed any
  crossing test.

---

## Build-order impact

- `capacity_series.csv` is a **new input for Fix 5 only** — it doesn't touch the
  progress/trajectory views. Wright's law stays last in the build order; it's now
  unblocked rather than reordered.
- `conditional` state is a **trajectory-classifier change** — lands with the Fix 1
  trajectory work, adds one branch (policy_dependent + economics-progress<1), and one
  gap-typology cell.
- The capital-pool fix is a **data edit** to existing rows — do it before the capital
  dependencies feed any crossing test.

## Still open (carried + new)

- Whether the Wright fit reports two projected crossings (economics-only vs.
  policy-scenario) for `conditional` dependencies, or just flags conditional and
  defers the second curve.
- Sourcing the ETS capacity points to the public NEO data viewer if a URL exists, to
  replace the blank `source_url`.
- Carried from v3: `window_years` default, baseline choice, `metric_series_status`
  gating, the two soft values (wind 2010, pvXchange spot).
