# data/derived — generated. Do not edit.

Every file here is produced by `npm run build-metric-data` from `data/sources/**` (raw,
immutable provider files) plus `data/curated/**` (human-authored inputs). Editing a file
here is a change that the next rebuild silently reverts — and it breaks the audit below.

**To change a number:** change its source. A provider number lives in
`data/sources/<provider>/`; a hand-entered number for a paywalled or cited-only source
lives in `data/curated/cited_anchors.csv` or `data/curated/capacity_anchors.csv`. Then
re-run the build and review the diff.

## Audit

```
rm -f data/derived/*.csv && npm run build-metric-data && git diff
```

An empty diff proves the committed derived files reproduce exactly from the committed raw
sources. That is why these files are tracked in git rather than ignored.

The glob rather than `rm -rf data/derived`, because this README is hand-written, not
generated — deleting it would show up in the diff as a false alarm. (The build does
recreate the directory if you remove the whole thing.)

## Files

| file | built from |
|---|---|
| `metric_observations_full.csv` | OWID solar prices; cited anchors (BNEF, LBNL, World Bank, IRENA RPGC, …) |
| `capacity_series.csv` | OWID solar PV capacity (AC); IRENA onshore wind (`.xlsb`); battery capacity anchor |

Provenance for each source — basis, vintage, retrieval date, access — is in
[`../sources/DATA_SOURCES.md`](../sources/DATA_SOURCES.md).
