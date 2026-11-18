"""
pca_response_profile.py
-----------------------
Per-vertex PCA of the condition response profile across N PSC maps.

Each vertex has a response vector  [PSC_RW, PSC_FF, PSC_CS, PSC_SC, PSC_Faces, PSC_Limbs]
— a point in condition space.  PCA finds the main axes of variation across vertices:

    PC1  → dominant axis (usually overall magnitude if raw, or strongest
           selectivity gradient if normalized)
    PC2  → next independent selectivity axis
    ...

Standard neuroimaging approach (Huth et al. 2012 Science; Çukur et al. 2013 Neuron):
    1. Stack vsNull PSC maps → (n_vertices × n_conditions) matrix per hemisphere
    2. Normalize per vertex (z-score removes magnitude; unit-length keeps only direction)
    3. Fit PCA jointly across both hemispheres for unified axes
    4. Save per-vertex PC score GIFTIs + loadings TSV + variance-explained TSV

Outputs (under --out-dir/sub-XX/ses-YY/):
    sub-XX_ses-YY_task-T_hemi-{L,R}_space-S_stat-PC{N}score_statmap.func.gii
    sub-XX_ses-YY_task-T_space-S_pca_loadings.tsv
    sub-XX_ses-YY_task-T_space-S_pca_variance.tsv

Usage
-----
    python pca_response_profile.py \\
        --glm-dir /path/to/analysis-name_strategy \\
        --sub 02 --ses 05 --task fLoc --space fsnative \\
        --conditions RW,FF,CS,SC,Faces,Limbs \\
        --vs-suffix vsNull \\
        --out-dir /path/to/derivatives/angle_calculation \\
        --n-pcs 5 \\
        --normalize zscore
"""
from __future__ import annotations

import datetime
import glob
import os
import os.path as op
import time

import nibabel as nib
import numpy as np
import pandas as pd
import typer
from nilearn.surface import load_surf_data
from rich.console import Console
from sklearn.decomposition import PCA


def save_statmap_to_gifti(data: np.ndarray, outname: str) -> None:
    gii = nib.gifti.gifti.GiftiImage()
    gii.add_gifti_data_array(
        nib.gifti.gifti.GiftiDataArray(
            data=data,
            datatype="NIFTI_TYPE_FLOAT32",
            coordsys=None,
        ),
    )
    nib.save(gii, outname)

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_psc(glm_dir: str, space: str, hemi: str, contrast_id: str) -> str | None:
    pattern = op.join(
        glm_dir,
        f"*_hemi-{hemi}_space-{space}*_contrast-{contrast_id}_stat-PSC_statmap*.func.gii",
    )
    matches = glob.glob(pattern)
    if not matches:
        return None
    if len(matches) > 1:
        console.print(f"  [yellow]WARNING[/yellow] multiple PSC files for {contrast_id} hemi-{hemi}, using first")
    return matches[0]


def _load_psc_matrix(
    glm_dir: str, space: str, hemi: str,
    conditions: list[str], vs_suffix: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Load all condition PSC maps for one hemisphere.

    Returns
    -------
    matrix : (n_vertices, n_conditions)  float32, NaN where file missing
    found  : list of condition names that were successfully loaded
    """
    arrays, found = [], []
    n_verts = None

    for cond in conditions:
        contrast_id = f"{cond}{vs_suffix}"
        path = _find_psc(glm_dir, space, hemi, contrast_id)
        if path is None:
            console.print(f"  [yellow]SKIP[/yellow] hemi-{hemi} {contrast_id}: not found")
            continue
        arr = load_surf_data(path).ravel().astype(np.float32)
        if n_verts is None:
            n_verts = arr.shape[0]
        if arr.shape[0] != n_verts:
            console.print(f"  [red]ERROR[/red] {contrast_id} vertex count mismatch — skipping")
            continue
        arrays.append(arr)
        found.append(cond)
        console.print(f"  hemi-{hemi}  loaded {contrast_id}: {op.basename(path)}")

    if not arrays:
        return np.empty((0, 0), dtype=np.float32), []

    return np.stack(arrays, axis=1), found  # (n_vertices, n_found_conds)


def _normalize(matrix: np.ndarray, method: str) -> np.ndarray:
    """
    Normalize per vertex (row-wise).

    method : 'zscore'  — subtract condition-mean, divide by condition-std
             'unit'    — L2-normalize to unit length
             'none'    — no normalization
    """
    if method == "none":
        return matrix

    out = matrix.copy()

    if method == "zscore":
        mean = np.nanmean(out, axis=1, keepdims=True)
        std  = np.nanstd(out, axis=1, keepdims=True, ddof=1)
        std  = np.where(std < 1e-10, 1.0, std)
        out  = (out - mean) / std

    elif method == "unit":
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        norm = np.where(norm < 1e-10, 1.0, norm)
        out  = out / norm

    return out


# ── PCA core ──────────────────────────────────────────────────────────────────

def run_pca(
    matrix_L: np.ndarray,
    matrix_R: np.ndarray,
    conditions: list[str],
    n_pcs: int,
    normalize: str,
    min_magnitude: float,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Fit PCA jointly on both hemispheres, return per-vertex score arrays.

    Returns
    -------
    scores_L : (n_vertices_L, n_pcs)
    scores_R : (n_vertices_R, n_pcs)
    pca      : fitted sklearn PCA object (contains .components_, .explained_variance_ratio_)
    """
    n_L = matrix_L.shape[0]
    n_R = matrix_R.shape[0]

    # Joint matrix across hemispheres
    joint = np.concatenate([matrix_L, matrix_R], axis=0)  # (n_L+n_R, n_conds)

    # Magnitude mask: exclude vertices where overall PSC is near zero
    magnitude = np.linalg.norm(joint, axis=1)
    valid_mask = (magnitude > min_magnitude) & np.all(np.isfinite(joint), axis=1)

    n_valid = valid_mask.sum()
    console.print(
        f"  PCA fit on {n_valid:,} / {len(valid_mask):,} vertices "
        f"(magnitude > {min_magnitude}, normalize={normalize})"
    )

    # Normalize valid rows
    joint_norm = _normalize(joint, normalize)

    # Fit on valid vertices only
    pca = PCA(n_components=min(n_pcs, n_valid, joint.shape[1]))
    pca.fit(joint_norm[valid_mask])

    # Project ALL vertices (invalid ones get NaN)
    scores_all = np.full((len(joint), pca.n_components_), np.nan, dtype=np.float32)
    scores_all[valid_mask] = pca.transform(joint_norm[valid_mask])

    scores_L = scores_all[:n_L]
    scores_R = scores_all[n_L:]

    return scores_L, scores_R, pca


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def main(
    glm_dir: str = typer.Option(
        ..., "--glm-dir",
        help="Analysis-level or session-level directory containing stat-PSC GIFTIs.",
    ),
    sub: str = typer.Option(..., "--sub"),
    ses: str = typer.Option(..., "--ses"),
    task: str = typer.Option(..., "--task"),
    space: str = typer.Option("fsnative", "--space"),
    conditions: str = typer.Option(
        "RW,FF,CS,SC,Faces,Limbs", "--conditions",
        help="Comma-separated ordered condition list. Order sets the loadings column order.",
    ),
    vs_suffix: str = typer.Option(
        "vsNull", "--vs-suffix",
        help="Suffix appended to each condition name to form the contrast ID.",
    ),
    out_dir: str = typer.Option(
        ..., "--out-dir",
        help="Root output directory. Files go to <out-dir>/sub-XX/ses-YY/.",
    ),
    n_pcs: int = typer.Option(5, "--n-pcs", help="Number of PCs to retain."),
    normalize: str = typer.Option(
        "zscore", "--normalize",
        help="Per-vertex normalization before PCA: 'zscore' | 'unit' | 'none'.\n"
             "  zscore: removes magnitude, focuses on relative condition preference.\n"
             "  unit:   L2-normalize, pure direction in condition space.\n"
             "  none:   raw PSC values (PC1 captures overall responsiveness).",
    ),
    min_magnitude: float = typer.Option(
        0.1, "--min-magnitude",
        help="Vertices with L2-norm of PSC vector <= this are excluded from PCA fit.",
    ),
    hemis: str = typer.Option("L,R", "--hemis"),
) -> None:
    sub = sub.strip().zfill(2)
    ses = ses.strip().zfill(2)

    if normalize not in ("zscore", "unit", "none"):
        console.print(f"[red]ERROR[/red] --normalize must be 'zscore', 'unit', or 'none'")
        raise typer.Exit(1)

    if not op.isdir(glm_dir):
        console.print(f"[red]ERROR[/red] --glm-dir does not exist: {glm_dir}")
        raise typer.Exit(1)

    # Descend into sub-XX/ses-YY if present under glm_dir
    search_dir = glm_dir
    subses_subdir = op.join(glm_dir, f"sub-{sub}", f"ses-{ses}")
    if op.isdir(subses_subdir):
        search_dir = subses_subdir

    cond_list = [c.strip() for c in conditions.split(",")]
    hemi_list = [h.strip() for h in hemis.split(",")]

    # Output directory
    out_subses = op.join(out_dir, f"sub-{sub}", f"ses-{ses}")
    os.makedirs(out_subses, exist_ok=True)

    _t_start = time.time()

    console.rule("[bold]PCA response profile[/bold]")
    console.print(f"  search-dir : {search_dir}")
    console.print(f"  sub-{sub}  ses-{ses}")
    console.print(f"  conditions : {cond_list}")
    console.print(f"  vs-suffix  : {vs_suffix}")
    console.print(f"  normalize  : {normalize}   min-magnitude: {min_magnitude}")
    console.print(f"  n-pcs      : {n_pcs}")
    console.print(f"  out-dir    : {out_subses}")
    console.print()

    # ── Load PSC matrices ─────────────────────────────────────────────────────
    matrices: dict[str, np.ndarray] = {}
    found_conds: list[str] = []

    for hemi in hemi_list:
        console.rule(f"hemi-{hemi}", style="dim")
        mat, found = _load_psc_matrix(search_dir, space, hemi, cond_list, vs_suffix)
        if mat.size == 0:
            console.print(f"  [red]ERROR[/red] no PSC maps found for hemi-{hemi} — skipping")
            continue
        matrices[hemi] = mat
        if not found_conds:
            found_conds = found
        elif found_conds != found:
            console.print(
                f"  [yellow]WARNING[/yellow] condition set differs between hemispheres "
                f"({found} vs {found_conds})"
            )

    if len(matrices) < 2:
        console.print(f"[red]ERROR[/red] need both hemispheres for joint PCA")
        raise typer.Exit(1)

    n_found = len(found_conds)
    actual_n_pcs = min(n_pcs, n_found)
    if actual_n_pcs < n_pcs:
        console.print(
            f"  [yellow]INFO[/yellow] only {n_found} conditions found, "
            f"capping n-pcs at {actual_n_pcs}"
        )

    console.print()
    console.rule("PCA", style="dim")

    # ── Fit PCA jointly ───────────────────────────────────────────────────────
    _t_pca = time.time()
    mat_L = matrices.get("L", np.empty((0, n_found), dtype=np.float32))
    mat_R = matrices.get("R", np.empty((0, n_found), dtype=np.float32))

    scores_L, scores_R, pca = run_pca(
        mat_L, mat_R, found_conds, actual_n_pcs, normalize, min_magnitude
    )
    console.print(f"  PCA done in {time.time() - _t_pca:.1f} s")

    # ── Print explained variance ──────────────────────────────────────────────
    console.print()
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_, cumvar), 1):
        console.print(f"  PC{i}  {var*100:5.1f}%  (cumulative {cum*100:5.1f}%)")

    # ── Save PC score GIFTIs ──────────────────────────────────────────────────
    console.print()
    scores_by_hemi = {"L": scores_L, "R": scores_R}
    n_written = 0

    for hemi, scores in scores_by_hemi.items():
        if hemi not in matrices:
            continue
        for pc_idx in range(pca.n_components_):
            pc_scores = scores[:, pc_idx]
            fname = (
                f"sub-{sub}_ses-{ses}_task-{task}"
                f"_hemi-{hemi}_space-{space}"
                f"_desc-norm{normalize}"
                f"_stat-PC{pc_idx + 1}score_statmap.func.gii"
            )
            out_path = op.join(out_subses, fname)
            save_statmap_to_gifti(pc_scores.astype(np.float32), out_path)
            console.print(f"  [green]saved[/green] {fname}")
            n_written += 1

    # ── Save loadings TSV ─────────────────────────────────────────────────────
    loadings_df = pd.DataFrame(
        pca.components_,
        index=[f"PC{i+1}" for i in range(pca.n_components_)],
        columns=found_conds,
    )
    loadings_path = op.join(
        out_subses,
        f"sub-{sub}_ses-{ses}_task-{task}_space-{space}"
        f"_desc-norm{normalize}_pca_loadings.tsv",
    )
    loadings_df.to_csv(loadings_path, sep="\t", float_format="%.6f")
    console.print(f"  [green]saved[/green] {op.basename(loadings_path)}")

    # ── Save variance explained TSV ───────────────────────────────────────────
    var_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(pca.n_components_)],
        "variance_explained": pca.explained_variance_ratio_,
        "cumulative_variance": cumvar,
    })
    var_path = op.join(
        out_subses,
        f"sub-{sub}_ses-{ses}_task-{task}_space-{space}"
        f"_desc-norm{normalize}_pca_variance.tsv",
    )
    var_df.to_csv(var_path, sep="\t", index=False, float_format="%.6f")
    console.print(f"  [green]saved[/green] {op.basename(var_path)}")

    console.print()
    console.print(f"  Conditions: {found_conds}")
    console.print()
    console.rule(
        f"[dim]Finished {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"— {n_written + 2} files  ({time.time() - _t_start:.1f} s total)[/dim]"
    )


if __name__ == "__main__":
    app()
