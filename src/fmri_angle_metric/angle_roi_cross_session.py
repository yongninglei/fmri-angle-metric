"""Summarise selectivity angle inside FreeSurfer ROIs, across sessions.

Reads the output of ``angle_per_vertex.py`` together with a FreeSurfer label
directory, and reports how the angle inside each ROI behaves from session to
session: the ROI median angle per session, its spread, and how many vertices
survive in each session.

Note on scope: the shared convention is one subject x one session per call.
This script is inherently cross-session, so its unit of work is one subject
across many sessions.

Why ``.label`` files and not the ``.annot``
-------------------------------------------
For ``manual_label_clusters_analysis_27_v3`` the ``.annot`` name table is
shifted by one relative to the vertices it labels: the region the annot calls
``pOTS-words`` is in fact the ``OWA`` label, and so on down the list, with
``mFus-words`` absent from the annot entirely. This was verified by vertex-set
overlap against the ``.label`` files and holds for every subject 01-11. The
``.label`` files are therefore the default source here. ``--source annot`` is
available but warns, because with that data it silently mislabels every ROI.
"""

from __future__ import annotations

import datetime
import glob
import os.path as op
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import typer
from nilearn.surface import load_surf_data
from rich.console import Console

# Works both when the package is installed and when this file is run directly.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fmri_angle_metric._viz import (
    DIVERGING,
    INK,
    INK_MUTED,
    SEQUENTIAL,
    SERIES,
    SURFACE,
    heatmap,
    series_legend,
    style_axes,
    titles,
)

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

FS_HEMI = {"L": "lh", "R": "rh"}


# ── ROI loading ──────────────────────────────────────────────────────────────

def load_rois_from_labels(label_dir: str, fs_hemi: str,
                          wanted: list[str] | None) -> dict[str, np.ndarray]:
    """Read every ``<hemi>.<name>.label`` as a vertex-index array."""
    rois: dict[str, np.ndarray] = {}
    for path in sorted(glob.glob(op.join(label_dir, f"{fs_hemi}.*.label"))):
        name = op.basename(path)[len(fs_hemi) + 1: -len(".label")]
        if wanted and name not in wanted:
            continue
        rois[name] = nib.freesurfer.read_label(path)
    return rois


def load_rois_from_annot(label_dir: str, fs_hemi: str,
                         wanted: list[str] | None) -> dict[str, np.ndarray]:
    """Read an ``.annot``. See the module docstring for why this is not default."""
    matches = sorted(glob.glob(op.join(label_dir, f"{fs_hemi}.*.annot")))
    if not matches:
        return {}
    labels, _, names = nib.freesurfer.read_annot(matches[0])
    names = [n.decode() if isinstance(n, bytes) else n for n in names]

    rois: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        idx = np.where(labels == i)[0]
        if idx.size == 0:
            continue
        if wanted and name not in wanted:
            continue
        rois[name] = idx
    return rois


# ── angle map loading ────────────────────────────────────────────────────────

def _find_one(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        console.print(f"  [yellow]WARNING[/yellow] {len(matches)} matches for "
                      f"{op.basename(pattern)} — using first")
        return matches[0]
    return None


def load_map(angle_dir: str, sub: str, ses: str, task: str, space: str,
             hemi: str, cond_a: str, cond_b: str, stat: str) -> np.ndarray | None:
    pattern = op.join(
        angle_dir, f"sub-{sub}", f"ses-{ses}",
        f"sub-{sub}_ses-{ses}_task-{task}_hemi-{hemi}_space-{space}"
        f"_condA-{cond_a}_condB-{cond_b}_stat-{stat}_statmap.func.gii",
    )
    path = _find_one(pattern)
    if path is None:
        return None
    return np.asarray(load_surf_data(path), dtype=np.float64).ravel()


# ── plotting ─────────────────────────────────────────────────────────────────

def make_figure(long: pd.DataFrame, sessions: list[str], rois: list[str],
                pair_labels: list[str], sub: str, hemi: str,
                out_png: str) -> None:
    n_cols = len(pair_labels)
    fig, axes = plt.subplots(3, n_cols, figsize=(5.1 * n_cols, 13.5),
                             facecolor=SURFACE, constrained_layout=True,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    colors = {roi: SERIES[i % len(SERIES)] for i, roi in enumerate(rois)}
    x = np.arange(len(sessions))
    heat_axes, im_ref = [], None

    for k, pair in enumerate(pair_labels):
        sub_df = long[long["pair"] == pair]

        # ── Row 0: ROI median angle across sessions ──────────────────────────
        ax = axes[0, k]
        style_axes(ax)
        ax.axhline(0, color="#e3e2df", linewidth=1.5)
        for roi in rois:
            r = sub_df[sub_df["roi"] == roi].set_index("ses").reindex(sessions)
            ax.plot(x, r["median_angle"].values, marker="o", markersize=7,
                    linewidth=2, color=colors[roi])
            ax.fill_between(x, r["q25"].values, r["q75"].values,
                            color=colors[roi], alpha=0.13, linewidth=0)
        ax.set_xticks(x, sessions, fontsize=9, color=INK_MUTED)
        ax.set_xlabel("session", fontsize=9, color=INK_MUTED)
        ax.set_ylabel("angle (degrees)", fontsize=9, color=INK_MUTED)
        titles(ax, pair, "ROI median angle, band = IQR across vertices")

        # ── Row 1: ROI x session heatmap of the median angle ─────────────────
        ax = axes[1, k]
        mat = (sub_df.pivot_table(index="roi", columns="ses",
                                  values="median_angle")
               .reindex(index=rois, columns=sessions))
        im_ref = heatmap(ax, mat, pair, "median angle (degrees)",
                         vmin=-135, vmax=135, cmap=DIVERGING, fmt="{:.0f}")
        heat_axes.append(ax)

        # ── Row 2: vertices surviving per session ────────────────────────────
        ax = axes[2, k]
        style_axes(ax)
        for roi in rois:
            r = sub_df[sub_df["roi"] == roi].set_index("ses").reindex(sessions)
            ax.plot(x, r["n_valid"].values, marker="o", markersize=7,
                    linewidth=2, color=colors[roi])
        ax.set_xticks(x, sessions, fontsize=9, color=INK_MUTED)
        ax.set_xlabel("session", fontsize=9, color=INK_MUTED)
        ax.set_ylabel("vertices with a finite angle", fontsize=9, color=INK_MUTED)
        ax.set_ylim(bottom=0)
        titles(ax, pair, "ROI coverage per session")

    if im_ref is not None:
        cb = fig.colorbar(im_ref, ax=heat_axes, fraction=0.02, pad=0.01)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
        cb.set_label("angle (degrees)", fontsize=8.5, color=INK_MUTED)

    series_legend(fig, rois, colors)
    fig.suptitle(f"sub-{sub} hemi-{hemi} — ROI angle across sessions",
                 fontsize=13.5, color=INK, x=0.006, ha="left", weight="bold")
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def make_stability_figure(summary: pd.DataFrame, rois: list[str],
                          pair_labels: list[str], sub: str, hemi: str,
                          out_png: str) -> None:
    """One compact view: how much each ROI's angle moves across sessions."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), facecolor=SURFACE,
                             constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    sd = summary.pivot_table(index="roi", columns="pair", values="sd_across_ses")
    sd = sd.reindex(index=rois, columns=pair_labels)
    im = heatmap(axes[0], sd, "Session-to-session variability",
                 "SD of the ROI median angle (degrees)",
                 vmin=0, vmax=max(5.0, float(np.nanmax(sd.values)) if np.isfinite(sd.values).any() else 5.0),
                 cmap=SEQUENTIAL, fmt="{:.1f}", xlabel="condition pair")
    axes[0].tick_params(axis="x", labelrotation=30)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    cb.set_label("degrees", fontsize=8.5, color=INK_MUTED)

    rng = summary.pivot_table(index="roi", columns="pair", values="range_across_ses")
    rng = rng.reindex(index=rois, columns=pair_labels)
    im2 = heatmap(axes[1], rng, "Worst-case swing",
                  "max - min of the ROI median angle (degrees)",
                  vmin=0, vmax=max(10.0, float(np.nanmax(rng.values)) if np.isfinite(rng.values).any() else 10.0),
                  cmap=SEQUENTIAL, fmt="{:.0f}", xlabel="condition pair")
    axes[1].tick_params(axis="x", labelrotation=30)
    cb = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    cb.set_label("degrees", fontsize=8.5, color=INK_MUTED)

    fig.suptitle(f"sub-{sub} hemi-{hemi} — ROI angle stability",
                 fontsize=13.5, color=INK, x=0.006, ha="left", weight="bold")
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    angle_dir: str = typer.Option(
        ..., "--angle-dir", help="Root of the angle output tree (sub-XX/ses-YY/)."),
    label_dir: str = typer.Option(
        ..., "--label-dir",
        help="FreeSurfer label directory holding the ROI .label files, e.g. "
             ".../freesurfer-with_t2/sub-02/label/manual_label_clusters_analysis_27_v3"),
    sub: str = typer.Option(..., "--sub", help="Subject label, e.g. 02"),
    sessions: str = typer.Option(..., "--sessions", help="e.g. '01,02,03,04,05,06'"),
    pairs: str = typer.Option(
        ..., "--pairs", help="Comma-separated condA:condB pairs, e.g. 'RW:CS,RW:FF'."),
    rois: str = typer.Option(
        "", "--rois",
        help="Comma-separated ROI names to keep. Default: every .label in the "
             "directory."),
    source: str = typer.Option(
        "label", "--source",
        help="'label' (default, recommended) or 'annot'. See the module "
             "docstring — the annot in this dataset is mislabeled."),
    task: str = typer.Option("fLoc", "--task"),
    space: str = typer.Option("fsnative", "--space"),
    hemi: str = typer.Option(
        "L", "--hemi", help="Single hemisphere: L or R (the manual labels are lh only)."),
    min_magnitude: float = typer.Option(
        0.0, "--min-magnitude",
        help="Extra mask: keep only vertices whose stat-magnitude exceeds this."),
    out_dir: str = typer.Option(..., "--out-dir"),
    n_workers: int = typer.Option(1, "--n-workers"),
) -> None:
    sub = sub.strip().zfill(2)
    ses_list = [s.strip().zfill(2) for s in sessions.split(",") if s.strip()]
    hemi = hemi.strip().upper()
    if hemi not in FS_HEMI:
        console.print(f"[red]ERROR[/red] --hemi must be L or R, got '{hemi}'")
        raise typer.Exit(1)
    if source not in ("label", "annot"):
        console.print(f"[red]ERROR[/red] --source must be 'label' or 'annot'")
        raise typer.Exit(1)

    pair_list: list[tuple[str, str]] = []
    for tok in pairs.split(","):
        tok = tok.strip()
        if ":" not in tok:
            console.print(f"[red]ERROR[/red] pair must be 'condA:condB', got '{tok}'")
            raise typer.Exit(1)
        a, b = tok.split(":", 1)
        pair_list.append((a.strip(), b.strip()))

    wanted = [r.strip() for r in rois.split(",") if r.strip()] or None

    console.rule(f"[bold]angle-roi-cross-session — sub-{sub} hemi-{hemi}[/bold]")
    t0 = time.time()

    for d, what in ((angle_dir, "--angle-dir"), (label_dir, "--label-dir")):
        if not op.isdir(d):
            console.print(f"[red]ERROR[/red] {what} does not exist: {d}")
            raise typer.Exit(1)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── ROIs ─────────────────────────────────────────────────────────────────
    fs_hemi = FS_HEMI[hemi]
    if source == "annot":
        console.print("  [yellow]WARNING[/yellow] --source annot: in this dataset "
                      "the annot name table is shifted by one and omits "
                      "mFus-words. Prefer --source label.")
        roi_map = load_rois_from_annot(label_dir, fs_hemi, wanted)
    else:
        roi_map = load_rois_from_labels(label_dir, fs_hemi, wanted)

    if not roi_map:
        console.print(f"[red]ERROR[/red] no ROIs found in {label_dir} for {fs_hemi}")
        raise typer.Exit(1)

    roi_names = list(roi_map)
    console.print(f"  ROIs ({source}) : " +
                  ", ".join(f"{r} ({roi_map[r].size})" for r in roi_names))
    console.print(f"  sessions   : {', '.join(ses_list)}")
    console.print(f"  pairs      : {', '.join(f'{a}vs{b}' for a, b in pair_list)}")
    console.print()

    rows: list[dict] = []
    pair_labels: list[str] = []
    found_ses: list[str] = []

    for cond_a, cond_b in pair_list:
        label = f"{cond_a}vs{cond_b}"
        console.rule(label, style="dim")

        def _one(ses: str, _a: str = cond_a, _b: str = cond_b):
            ang = load_map(angle_dir, sub, ses, task, space, hemi, _a, _b, "angle")
            mag = None
            if min_magnitude > 0:
                mag = load_map(angle_dir, sub, ses, task, space, hemi, _a, _b,
                               "magnitude")
            return ses, ang, mag

        maps: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_one, s): s for s in ses_list}
                for fut in as_completed(futures):
                    ses, ang, mag = fut.result()
                    if ang is not None:
                        maps[ses] = (ang, mag)
        else:
            for s in ses_list:
                ses, ang, mag = _one(s)
                if ang is not None:
                    maps[ses] = (ang, mag)

        missing = [s for s in ses_list if s not in maps]
        if missing:
            console.print(f"  [yellow]SKIP[/yellow] no angle map for ses-"
                          f"{', ses-'.join(missing)}")
        if not maps:
            console.print(f"  [yellow]SKIP[/yellow] {label}: no sessions")
            continue

        ordered = [s for s in ses_list if s in maps]
        n_vert = maps[ordered[0]][0].shape[0]

        # The labels index into the subject's native surface; a mismatch means
        # the labels and the angle maps are not on the same mesh.
        max_idx = max(int(v.max()) for v in roi_map.values())
        if max_idx >= n_vert:
            console.print(f"  [red]ERROR[/red] ROI vertex index {max_idx} exceeds "
                          f"the {n_vert} vertices in the angle map — labels and "
                          f"maps are on different surfaces")
            raise typer.Exit(1)

        for ses in ordered:
            ang, mag = maps[ses]
            for roi in roi_names:
                idx = roi_map[roi]
                vals = ang[idx]
                if min_magnitude > 0 and mag is not None:
                    vals = np.where(mag[idx] > min_magnitude, vals, np.nan)
                good = vals[np.isfinite(vals)]
                rows.append({
                    "ses": ses, "pair": label, "roi": roi,
                    "n_roi": int(idx.size), "n_valid": int(good.size),
                    "median_angle": float(np.median(good)) if good.size else np.nan,
                    "mean_angle": float(np.mean(good)) if good.size else np.nan,
                    "q25": float(np.percentile(good, 25)) if good.size else np.nan,
                    "q75": float(np.percentile(good, 75)) if good.size else np.nan,
                })

        pair_labels.append(label)
        found_ses = ordered
        med = {r: [d["median_angle"] for d in rows
                   if d["pair"] == label and d["roi"] == r] for r in roi_names}
        for r in roi_names:
            v = np.array(med[r], dtype=float)
            if np.isfinite(v).sum() < 2:
                console.print(f"  {r:<14} [yellow]too few sessions[/yellow]")
                continue
            console.print(f"  {r:<14} median {np.nanmean(v):7.1f}°  "
                          f"SD {np.nanstd(v):5.1f}°  "
                          f"range {np.nanmin(v):7.1f}–{np.nanmax(v):6.1f}°")
        console.print()

    if not rows:
        console.print("[red]ERROR[/red] nothing to summarise")
        raise typer.Exit(1)

    long = pd.DataFrame(rows)

    summary = (long.groupby(["roi", "pair"])["median_angle"]
               .agg(mean_across_ses="mean", sd_across_ses="std",
                    min_across_ses="min", max_across_ses="max",
                    n_sessions="count")
               .reset_index())
    summary["range_across_ses"] = (summary["max_across_ses"]
                                   - summary["min_across_ses"])

    # ── Outputs ──────────────────────────────────────────────────────────────
    console.rule("outputs", style="dim")
    stem = f"sub-{sub}_task-{task}_hemi-{hemi}_space-{space}"

    p = op.join(out_dir, f"{stem}_roi-angle-per-session.tsv")
    long.round(4).to_csv(p, sep="\t", index=False)
    console.print(f"  [green]saved[/green] {op.basename(p)}")

    p = op.join(out_dir, f"{stem}_roi-angle-stability.tsv")
    summary.round(4).to_csv(p, sep="\t", index=False)
    console.print(f"  [green]saved[/green] {op.basename(p)}")

    png = op.join(out_dir, f"{stem}_roi-angle-cross-session.png")
    make_figure(long, found_ses, roi_names, pair_labels, sub, hemi, png)
    console.print(f"  [green]saved[/green] {op.basename(png)}")

    png = op.join(out_dir, f"{stem}_roi-angle-stability.png")
    make_stability_figure(summary, roi_names, pair_labels, sub, hemi, png)
    console.print(f"  [green]saved[/green] {op.basename(png)}")

    console.print()
    console.rule(
        f"Done {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"({time.time() - t0:.0f} s)"
    )


if __name__ == "__main__":
    app()
