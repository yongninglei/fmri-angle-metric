# CLAUDE.md — fmri-angle-metric

Standalone scripts for per-vertex selectivity angle and PCA of condition
response profiles from fMRI PSC maps (BIDS/GIFTI surface data).

Shared neuroimaging script conventions (typer/rich CLI, one-sub-per-call,
parallel option, timing) are in `../CLAUDE.md`.

---

## Scripts

| Script | Input | Output |
|--------|-------|--------|
| `src/fmri_angle_metric/angle_per_vertex.py` | PSC GIFTIs from `l1_surface` | angle + magnitude GIFTIs per condition pair |
| `src/fmri_angle_metric/pca_response_profile.py` | PSC GIFTIs from `l1_surface` (all conditions) | PC score GIFTIs, loadings TSV, variance TSV |
| `src/fmri_angle_metric/pca_cross_session.py` | the PCA outputs above, many sessions | cross-session summary figure (PNG) + similarity TSVs |
| `src/fmri_angle_metric/angle_cross_session.py` | the angle outputs above, many sessions | cross-session summary figure (PNG) + correlation / Δangle TSVs |
| `src/fmri_angle_metric/angle_roi_cross_session.py` | angle outputs + FreeSurfer `.label` ROIs | per-ROI angle across sessions: 2 figures + TSVs |
| `src/fmri_angle_metric/_viz.py` | — | shared palette and plotting helpers for the summary figures |

### ROI labels — use `.label`, not the `.annot`

In `freesurfer-with_t2/sub-XX/label/manual_label_clusters_analysis_27_v3` the
`lh.manual_clusters_5ROIs.annot` name table is **shifted by one** relative to the
vertices it labels: the region the annot calls `pOTS-words` is really `OWA`,
`mOTS-words` is really `pOTS-words`, and so on; `mFus-words` is missing from the
annot entirely. Confirmed by vertex-set overlap (Jaccard 1.00 for each shifted
pairing) for **every subject 01-11**. Always read the `.label` files.

`pca_cross_session.py` is the one script that is *not* one-session-per-call —
it takes `--sessions 01,02,...` and compares them in a single run. It only
reads existing PCA output; it never re-fits.

Installed as console scripts `angle-per-vertex` and `pca-response-profile`
(see `[project.scripts]` in `pyproject.toml`).

Both scripts read from `derivatives/l1_surface/analysis-*/sub-XX/ses-YY/`
and write to `derivatives/angle_calculation/`.

---

## Angle formula

```
angle = atan2(PSC_A, PSC_B) − 45°
```

- `+45°` = pure condA response (condB ≈ 0)
- `−45°` = pure condB response (condA ≈ 0)
- `  0°` = equal response to both
- Range: −135° to +135°

Both-negative vertices (both PSC < 0): controlled by `--discard-both-neg`.
- Default: reflect into Q1 — swap and negate `(a_new = −b_old, b_new = −a_old)`,
  so the less-suppressed condition is treated as relatively preferred.
- `--discard-both-neg`: set angle to NaN.

## Upstream / downstream

**Upstream** (what this repo reads):

| folder (under `<project>/BIDS/derivatives` unless noted) | made by | | space |
|---|---|---|---|
| `l1_surface/analysis-final_v3_basic_MC` (PSC maps) | pyGLM | required | fsnative |
| `freesurfer-with_t2/sub-XX/label/manual_label_clusters_analysis_27_v3/` | FreeSurfer + manual labels | ROI scripts | fsnative |

**Downstream** (what it writes, and who reads it):

| folder | what | space | read by |
|---|---|---|---|
| `angle_calculation/analysis-<name>/` | angle / magnitude maps, PC scores, loadings | fsnative | not read by another repo |

Lab map with every repo: `~/soft/docs/repo_map.dot` → `docs/images/repo_map.png` (soft-conventions). A change of an output folder or its space updates this section, the sections of the repos named here, and the map, together.
