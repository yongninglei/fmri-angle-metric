"""
angle_per_vertex.py
-------------------
Compute per-vertex selectivity angle between two PSC maps.

For each (condA, condB) pair supplied via --pairs, reads the
stat-PSC GIFTI files produced by glm_surface_check_model_strategy.py
and writes two output GIFTI maps per hemisphere:

    stat-angle      : selectivity angle in degrees (condA > condB → positive)
    stat-magnitude  : sqrt(PSC_A² + PSC_B²)

The angle is computed as:
    atan2(PSC_A, PSC_B) − 45°

    +45° ≡ pure condA response
    −45° ≡ pure condB response
       0° ≡ equal response to both

Vertices where both PSC values are below min-magnitude are set to NaN.

Usage
-----
    python angle_per_vertex.py \\
        --glm-dir /path/to/analysis-name_strategy \\
        --sub 02 --ses 05 --task fLoc --space fsnative \\
        --pairs RW:CS,RW:FF,Faces:Limbs \\
        --vs-suffix vsNull

The script globs for PSC files matching the BIDS-style tokens so optional
tokens (acq, desc) do not need to be specified explicitly.
"""
from __future__ import annotations

import glob
import os
import os.path as op

import nibabel as nib
import numpy as np
import typer
from nilearn.surface import load_surf_data
from rich.console import Console


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


# ── angle metric ─────────────────────────────────────────────────────────────

def angle_metric(
    response_a: np.ndarray,
    response_b: np.ndarray,
    min_magnitude: float = 0.0,
    discard_both_neg: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-vertex selectivity angle and overall magnitude.

    Parameters
    ----------
    response_a, response_b : ndarray, shape (n_vertices,)
        PSC values for the two conditions.
    min_magnitude : float
        Vertices with sqrt(A²+B²) <= this are set to NaN.
    discard_both_neg : bool
        If True, vertices where both PSC values are negative are set to NaN.
        If False (default), they are reflected into the first quadrant so the
        "less suppressed" condition is treated as relatively preferred.

    Returns
    -------
    angle_deg : ndarray
        Angle in degrees. Positive ↔ A-selective; negative ↔ B-selective.
    magnitude : ndarray
        sqrt(A² + B²).
    """
    a, b = np.broadcast_arrays(
        np.asarray(response_a, dtype=float),
        np.asarray(response_b, dtype=float),
    )
    a = np.atleast_1d(a).copy()
    b = np.atleast_1d(b).copy()

    magnitude = np.hypot(a, b)

    a_orig = a.copy()
    b_orig = b.copy()
    both_neg = (a_orig < 0) & (b_orig < 0)

    if not discard_both_neg:
        # Reflect both-negative into Q1: less suppressed → relatively preferred
        a[both_neg] = -b_orig[both_neg]
        b[both_neg] = -a_orig[both_neg]

    angle_deg = np.degrees(np.arctan2(a, b)) - 45.0

    invalid = (
        ~np.isfinite(a_orig)
        | ~np.isfinite(b_orig)
        | (magnitude <= min_magnitude)
        | (discard_both_neg & both_neg)
    )
    angle_deg[invalid] = np.nan

    return angle_deg.astype(np.float32), magnitude.astype(np.float32)


# ── file discovery ────────────────────────────────────────────────────────────

def _find_psc_file(glm_dir: str, space: str, hemi: str,
                   contrast_id: str) -> str | None:
    """
    Glob for the stat-PSC GIFTI in glm_dir.
    Uses wildcards for sub/ses/task/acq/desc — the directory already
    constrains the right subject and session.
    """
    pattern = op.join(
        glm_dir,
        f"*_hemi-{hemi}_space-{space}*"
        f"_contrast-{contrast_id}_stat-PSC_statmap*.func.gii",
    )
    matches = glob.glob(pattern)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        console.print(
            f"  [yellow]WARNING[/yellow] multiple PSC files for {contrast_id} "
            f"hemi-{hemi}: {[op.basename(m) for m in matches]}\n"
            f"  Using first: {op.basename(matches[0])}"
        )
        return matches[0]
    console.print(f"  [dim]  tried: {pattern}[/dim]")
    return None


def _build_outname(glm_dir: str, sub: str, ses: str, task: str,
                   space: str, hemi: str, cond_a: str, cond_b: str,
                   stat: str) -> str:
    return op.join(
        glm_dir,
        f"sub-{sub}_ses-{ses}_task-{task}"
        f"_hemi-{hemi}_space-{space}"
        f"_condA-{cond_a}_condB-{cond_b}_stat-{stat}_statmap.func.gii",
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def main(
    glm_dir: str = typer.Option(
        ..., "--glm-dir",
        help="Analysis-level or session-level directory containing stat-PSC GIFTIs. "
             "If it contains sub-XX/ses-YY subdirs, the script descends automatically.",
    ),
    sub: str = typer.Option(..., "--sub", help="Subject label (for output filenames and dir lookup)."),
    ses: str = typer.Option(..., "--ses", help="Session label (for output filenames and dir lookup)."),
    task: str = typer.Option(..., "--task", help="Task label, e.g. fLoc"),
    space: str = typer.Option("fsnative", "--space"),
    pairs: str = typer.Option(
        ..., "--pairs",
        help="Comma-separated condA:condB pairs, e.g. 'RW:CS,RW:FF,Faces:Limbs'.",
    ),
    vs_suffix: str = typer.Option(
        "vsNull", "--vs-suffix",
        help="Suffix appended to each condition name to form the contrast ID "
             "(default: vsNull → contrast-RWvsNull).",
    ),
    min_magnitude: float = typer.Option(
        0.1, "--min-magnitude",
        help="Vertices with magnitude <= this PSC value are set to NaN.",
    ),
    discard_both_neg: bool = typer.Option(
        False, "--discard-both-neg",
        help="If set, vertices where both PSC values are negative are set to NaN. "
             "Default: reflect them into Q1 (less suppressed = relatively preferred).",
    ),
    hemis: str = typer.Option(
        "L,R", "--hemis",
        help="Comma-separated hemispheres to process.",
    ),
    out_dir: str = typer.Option(
        "", "--out-dir",
        help="Root output directory. Files go to <out-dir>/sub-XX/ses-YY/. "
             "Defaults to writing alongside the PSC files in --glm-dir.",
    ),
) -> None:
    sub = sub.strip().zfill(2)
    ses = ses.strip().zfill(2)

    if not op.isdir(glm_dir):
        console.print(f"[red]ERROR[/red] --glm-dir does not exist: {glm_dir}")
        raise typer.Exit(1)

    # Descend into sub-XX/ses-YY if they exist under glm_dir
    search_dir = glm_dir
    subses_subdir = op.join(glm_dir, f"sub-{sub}", f"ses-{ses}")
    if op.isdir(subses_subdir):
        search_dir = subses_subdir

    # Output directory
    if out_dir:
        write_dir = op.join(out_dir, f"sub-{sub}", f"ses-{ses}")
        os.makedirs(write_dir, exist_ok=True)
    else:
        write_dir = search_dir

    hemi_list = [h.strip() for h in hemis.split(",")]

    pair_list: list[tuple[str, str]] = []
    for tok in pairs.split(","):
        tok = tok.strip()
        if ":" not in tok:
            console.print(f"[red]ERROR[/red] pair must be 'condA:condB', got: '{tok}'")
            raise typer.Exit(1)
        ca, cb = tok.split(":", 1)
        pair_list.append((ca.strip(), cb.strip()))

    console.rule("[bold]Angle per vertex[/bold]")
    console.print(f"  glm-dir   : {glm_dir}")
    console.print(f"  search-dir: {search_dir}")
    console.print(f"  write-dir : {write_dir}")
    console.print(f"  sub-{sub}  ses-{ses}")
    console.print(f"  space     : {space}  hemis: {hemi_list}")
    console.print(f"  pairs     : {pair_list}")
    console.print(f"  vs-suffix : {vs_suffix}  min-magnitude: {min_magnitude}")
    both_neg_mode = "discard (NaN)" if discard_both_neg else "reflect (less-suppressed=preferred)"
    console.print(f"  both-neg  : {both_neg_mode}")
    console.print()

    n_written = 0

    for cond_a, cond_b in pair_list:
        contrast_a = f"{cond_a}{vs_suffix}"
        contrast_b = f"{cond_b}{vs_suffix}"

        console.rule(f"[bold cyan]{cond_a}  vs  {cond_b}[/bold cyan]", style="dim")

        for hemi in hemi_list:
            path_a = _find_psc_file(search_dir, space, hemi, contrast_a)
            path_b = _find_psc_file(search_dir, space, hemi, contrast_b)

            if path_a is None:
                console.print(
                    f"  [yellow]SKIP[/yellow] hemi-{hemi}: PSC file not found "
                    f"for contrast-{contrast_a}"
                )
                continue
            if path_b is None:
                console.print(
                    f"  [yellow]SKIP[/yellow] hemi-{hemi}: PSC file not found "
                    f"for contrast-{contrast_b}"
                )
                continue

            console.print(f"  hemi-{hemi}  A: {op.basename(path_a)}")
            console.print(f"  hemi-{hemi}  B: {op.basename(path_b)}")

            psc_a = load_surf_data(path_a).ravel().astype(np.float32)
            psc_b = load_surf_data(path_b).ravel().astype(np.float32)

            if psc_a.shape != psc_b.shape:
                console.print(
                    f"  [red]ERROR[/red] shape mismatch: "
                    f"{psc_a.shape} vs {psc_b.shape} — skipping"
                )
                continue

            angle, magnitude = angle_metric(
                psc_a, psc_b,
                min_magnitude=min_magnitude,
                discard_both_neg=discard_both_neg,
            )

            n_valid = np.sum(np.isfinite(angle))
            console.print(
                f"  hemi-{hemi}  {n_valid}/{len(angle)} vertices valid "
                f"(magnitude > {min_magnitude})"
            )

            out_angle = _build_outname(
                write_dir, sub, ses, task, space, hemi, cond_a, cond_b, "angle"
            )
            out_mag = _build_outname(
                write_dir, sub, ses, task, space, hemi, cond_a, cond_b, "magnitude"
            )

            save_statmap_to_gifti(angle, out_angle)
            save_statmap_to_gifti(magnitude, out_mag)

            console.print(f"  [green]saved[/green] {op.basename(out_angle)}")
            console.print(f"  [green]saved[/green] {op.basename(out_mag)}")
            n_written += 2

    console.print()
    console.rule()
    console.print(f"Done — {n_written} files written to {write_dir}")


if __name__ == "__main__":
    app()
