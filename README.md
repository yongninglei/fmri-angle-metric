# fmri-angle-metric

Per-vertex selectivity angle and PCA of condition response profiles from
fMRI percentage signal change (PSC) maps on the cortical surface (GIFTI).

## Scripts

### `angle_per_vertex.py`

Computes the selectivity angle between two conditions for each cortical vertex:

```
angle = atan2(PSC_A, PSC_B) − 45°
```

- `+45°` = pure condA response
- `−45°` = pure condB response
- `  0°` = equal response to both

```bash
angle-per-vertex \
    --glm-dir /path/to/l1_surface/analysis-X_strategy/sub-02/ses-05 \
    --sub 02 --ses 05 --task fLoc --space fsnative \
    --pairs RW:CS,RW:FF,Faces:Limbs \
    --vs-suffix vsNull \
    --out-dir /path/to/derivatives/angle_calculation/analysis-single_ses_testing

# both-negative vertices: discard instead of reflect
angle-per-vertex ... --discard-both-neg
```

Outputs per hemisphere per pair:
- `*_condA-{A}_condB-{B}_stat-angle_statmap.func.gii`
- `*_condA-{A}_condB-{B}_stat-magnitude_statmap.func.gii`

### `pca_response_profile.py`

Fits PCA jointly across both hemispheres on the (n_vertices × n_conditions)
PSC matrix (z-scored per vertex). Finds the main axes of condition selectivity.

```bash
pca-response-profile \
    --glm-dir /path/to/l1_surface/analysis-X_strategy/sub-02/ses-05 \
    --sub 02 --ses 05 --task fLoc --space fsnative \
    --conditions RW,FF,CS,SC,Faces,Limbs \
    --vs-suffix vsNull \
    --out-dir /path/to/derivatives/angle_calculation \
    --n-pcs 5 --normalize zscore
```

Outputs:
- `*_stat-PC{N}score_statmap.func.gii` per hemisphere per PC
- `*_pca_loadings.tsv` — condition weights per PC
- `*_pca_variance.tsv` — variance explained

## Installation

### uv (recommended)

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
# or install the project itself:
uv pip install -e .
```

### pip

```bash
pip install -r requirements.txt
```

### conda / micromamba

```bash
micromamba env create -f environment.yml
micromamba activate fmri-angle-metric
```

## License

MIT — see [LICENSE](LICENSE).
