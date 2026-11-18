"""Compare per-vertex selectivity angle maps across sessions for one subject.

Reads the output of ``angle_per_vertex.py`` and reports, for each condition
pair, how reproducible the angle map is from session to session.

Note on scope: the shared convention is one subject x one session per call.
This script is inherently cross-session, so its unit of work is one subject
across many sessions.

Two complementary measures, because either alone is misleading
--------------------------------------------------------------
* **Pearson r** — do the two sessions rank vertices the same way? Sensitive to
  spatial pattern, blind to a constant offset.
* **Median |delta angle|** — by how many degrees does a typical vertex actually
  disagree? Sensitive to offset, and in the units you care about. Two sessions
  can correlate at r = 0.9 while sitting 20 degrees apart.

On using Pearson r for angles: ``angle_per_vertex.py`` produces values on
(-135, 135] — a 270 degree arc with no wraparound, because the both-negative
quadrant is either reflected into Q1 or discarded. So the values behave
linearly and Pearson r is interpretable. The *differences* are still computed
wrap-safe, so nothing breaks if that range ever changes.
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
import numpy as np
import pandas as pd
import typer
from nilearn.surface import load_surf_data
from rich.console import Console

# Works both when the package is installed and when this file is run directly
# (python angle_cross_session.py), where src/ is not on sys.path.
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
    session_legend,
    style_axes,
    titles,
)

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


# ── file discovery ───────────────────────────────────────────────────────────

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
             cond_a: str, cond_b: str, stat: str,
             hemis: list[str]) -> np.ndarray | None:
    """Concatenate the L/R maps for one condition pair into one vertex vector."""
    parts = []
    for hemi in hemis:
        pattern = op.join(
            angle_dir, f"sub-{sub}", f"ses-{ses}",
            f"sub-{sub}_ses-{ses}_task-{task}_hemi-{hemi}_space-{space}"
            f"_condA-{cond_a}_condB-{cond_b}_stat-{stat}_statmap.func.gii",
        )
        path = _find_one(pattern)
        if path is None:
            return None
        parts.append(np.asarray(load_surf_data(path), dtype=np.float64).ravel())
    return np.concatenate(parts)


# ── statistics ───────────────────────────────────────────────────────────────

def angular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed difference in degrees, wrapped to (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def correlation_matrix(maps: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Pearson r over vertices valid in both sessions."""
    sessions = list(maps)
    mat = np.full((len(sessions), len(sessions)), np.nan)
    for i, a in enumerate(sessions):
        for j, b in enumerate(sessions):
            va, vb = maps[a], maps[b]
            if va.shape != vb.shape:
                continue
            ok = np.isfinite(va) & np.isfinite(vb)
            if ok.sum() > 2:
                mat[i, j] = float(np.corrcoef(va[ok], vb[ok])[0, 1])
    return pd.DataFrame(mat, index=sessions, columns=sessions)


def delta_matrix(maps: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise median absolute angular difference, in degrees."""
    sessions = list(maps)
    mat = np.full((len(sessions), len(sessions)), np.nan)
    for i, a in enumerate(sessions):
        for j, b in enumerate(sessions):
            va, vb = maps[a], maps[b]
            if va.shape != vb.shape:
                continue
            ok = np.isfinite(va) & np.isfinite(vb)
            if ok.sum() > 0:
                mat[i, j] = float(np.median(np.abs(angular_difference(va[ok], vb[ok]))))
    return pd.DataFrame(mat, index=sessions, columns=sessions)


def _offdiag(df: pd.DataFrame) -> np.ndarray:
    return df.values[np.triu_indices(len(df), k=1)]


# ── plotting ─────────────────────────────────────────────────────────────────

def make_figure(sessions: list[str], pair_labels: list[str],
                corr: dict[str, pd.DataFrame], delta: dict[str, pd.DataFrame],
                angles: dict[str, dict[str, np.ndarray]],
                sub: str, out_png: str) -> None:
    n_pairs = len(pair_labels)
    n_cols = 1 + n_pairs
    fig, axes = plt.subplots(3, n_cols, figsize=(4.9 * n_cols, 13.5),
                             facecolor=SURFACE, constrained_layout=True)
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(SURFACE)

    colors = {ses: SERIES[i % len(SERIES)] for i, ses in enumerate(sessions)}
    corr_axes, delta_axes = [], []
    im_corr = im_delta = None

    # Scale the degree maps to the data, not to an arbitrary ceiling.
    all_delta = np.concatenate([_offdiag(delta[p]) for p in pair_labels])
    dmax = float(np.nanmax(all_delta)) if np.isfinite(all_delta).any() else 90.0
    dmax = max(10.0, np.ceil(dmax / 10.0) * 10.0)

    # ── Row 0 col 0: mean correlation per pair ───────────────────────────────
    ax = axes[0, 0]
    style_axes(ax)
    means = [float(np.nanmean(_offdiag(corr[p]))) for p in pair_labels]
    y = np.arange(n_pairs)
    ax.barh(y, means, color=SERIES[0], height=0.55)
    ax.set_yticks(y, pair_labels, fontsize=9, color=INK_MUTED)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean Pearson r", fontsize=9, color=INK_MUTED)
    ax.grid(True, axis="x", color="#e3e2df", linewidth=0.8)
    ax.grid(False, axis="y")
    for yi, m in zip(y, means):
        ax.text(m + 0.02, yi, f"{m:.2f}", va="center", fontsize=8.5, color=INK)
    titles(ax, "Pattern agreement", "mean of off-diagonal session pairs")

    # ── Row 0 cols 1..: correlation heatmap per pair ─────────────────────────
    for k, pair in enumerate(pair_labels):
        ax = axes[0, 1 + k]
        im_corr = heatmap(ax, corr[pair], pair, "Pearson r between sessions",
                          vmin=-1, vmax=1, cmap=DIVERGING)
        corr_axes.append(ax)

    # ── Row 1 col 0: median |delta| per pair ─────────────────────────────────
    ax = axes[1, 0]
    style_axes(ax)
    dmeans = [float(np.nanmean(_offdiag(delta[p]))) for p in pair_labels]
    ax.barh(y, dmeans, color=SERIES[1], height=0.55)
    ax.set_yticks(y, pair_labels, fontsize=9, color=INK_MUTED)
    ax.invert_yaxis()
    ax.set_xlabel("median |Δ angle| (degrees)", fontsize=9, color=INK_MUTED)
    ax.grid(True, axis="x", color="#e3e2df", linewidth=0.8)
    ax.grid(False, axis="y")
    for yi, m in zip(y, dmeans):
        ax.text(m + dmax * 0.02, yi, f"{m:.1f}°", va="center", fontsize=8.5,
                color=INK)
    titles(ax, "Typical disagreement", "lower is more reproducible")

    # ── Row 1 cols 1..: delta heatmap per pair ───────────────────────────────
    for k, pair in enumerate(pair_labels):
        ax = axes[1, 1 + k]
        im_delta = heatmap(ax, delta[pair], pair, "median |Δ angle| (degrees)",
                           vmin=0, vmax=dmax, cmap=SEQUENTIAL, fmt="{:.0f}")
        delta_axes.append(ax)

    # ── Row 2 col 0: summary text ────────────────────────────────────────────
    ax = axes[2, 0]
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    ax.text(0, 0.98, "Off-diagonal summary", transform=ax.transAxes,
            fontsize=11, color=INK, weight="bold", va="top")
    lines = [f"{'pair':<16}{'mean r':>8}{'min r':>8}{'med Δ':>9}"]
    for p in pair_labels:
        c, d = _offdiag(corr[p]), _offdiag(delta[p])
        lines.append(f"{p:<16}{np.nanmean(c):>8.2f}{np.nanmin(c):>8.2f}"
                     f"{np.nanmean(d):>8.1f}°")
    ax.text(0, 0.88, "\n".join(lines), transform=ax.transAxes, fontsize=9.5,
            color=INK_MUTED, va="top", family="monospace", linespacing=1.9)

    # ── Row 2 cols 1..: angle distribution per session ───────────────────────
    for k, pair in enumerate(pair_labels):
        ax = axes[2, 1 + k]
        style_axes(ax)
        bins = np.linspace(-135, 135, 61)
        centers = (bins[:-1] + bins[1:]) / 2
        for ses in sessions:
            arr = angles[pair].get(ses)
            if arr is None:
                continue
            vals = arr[np.isfinite(arr)]
            if vals.size == 0:
                continue
            dens, _ = np.histogram(vals, bins=bins, density=True)
            ax.plot(centers, dens * 100, linewidth=2, color=colors[ses])
        ax.axvline(0, color="#e3e2df", linewidth=1.5)
        ax.set_xlabel("angle (degrees)", fontsize=9, color=INK_MUTED)
        ax.set_ylabel("% of vertices", fontsize=9, color=INK_MUTED)
        ax.set_xticks([-135, -90, -45, 0, 45, 90, 135])
        titles(ax, pair, "angle distribution per session")

    if im_corr is not None:
        cb = fig.colorbar(im_corr, ax=corr_axes, fraction=0.02, pad=0.01)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
        cb.set_label("Pearson r", fontsize=8.5, color=INK_MUTED)
    if im_delta is not None:
        cb = fig.colorbar(im_delta, ax=delta_axes, fraction=0.02, pad=0.01)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
        cb.set_label("degrees", fontsize=8.5, color=INK_MUTED)

    session_legend(fig, sessions, colors)
    fig.suptitle(f"sub-{sub} — angle-map stability across sessions",
                 fontsize=13.5, color=INK, x=0.006, ha="left", weight="bold")
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    angle_dir: str = typer.Option(
        ..., "--angle-dir",
        help="Root of the angle output tree (contains sub-XX/ses-YY/).",
    ),
    sub: str = typer.Option(..., "--sub", help="Subject label, e.g. 02"),
    sessions: str = typer.Option(
        ..., "--sessions",
        help="Comma-separated session labels, e.g. '01,02,03,04,05,06'.",
    ),
    pairs: str = typer.Option(
        ..., "--pairs",
        help="Comma-separated condA:condB pairs, matching the ones the angle "
             "run produced, e.g. 'RW:CS,RW:FF,Faces:Limbs'.",
    ),
    task: str = typer.Option("fLoc", "--task"),
    space: str = typer.Option("fsnative", "--space"),
    min_magnitude: float = typer.Option(
        0.0, "--min-magnitude",
        help="Extra mask: keep only vertices whose stat-magnitude map exceeds "
             "this in EVERY session. 0 disables (the angle maps already carry "
             "the threshold used at angle time).",
    ),
    hemis: str = typer.Option("L,R", "--hemis"),
    out_dir: str = typer.Option(..., "--out-dir"),
    n_workers: int = typer.Option(1, "--n-workers", help="Threads for loading GIFTIs."),
) -> None:
    sub = sub.strip().zfill(2)
    ses_list = [s.strip().zfill(2) for s in sessions.split(",") if s.strip()]
    hemi_list = [h.strip() for h in hemis.split(",") if h.strip()]

    pair_list: list[tuple[str, str]] = []
    for tok in pairs.split(","):
        tok = tok.strip()
        if ":" not in tok:
            console.print(f"[red]ERROR[/red] pair must be 'condA:condB', got '{tok}'")
            raise typer.Exit(1)
        a, b = tok.split(":", 1)
        pair_list.append((a.strip(), b.strip()))

    console.rule(f"[bold]angle-cross-session — sub-{sub}[/bold]")
    t0 = time.time()

    if not op.isdir(angle_dir):
        console.print(f"[red]ERROR[/red] --angle-dir does not exist: {angle_dir}")
        raise typer.Exit(1)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    console.print(f"  sessions : {', '.join(ses_list)}")
    console.print(f"  pairs    : {', '.join(f'{a}vs{b}' for a, b in pair_list)}")
    if min_magnitude > 0:
        console.print(f"  extra mask: magnitude > {min_magnitude} in every session")
    console.print()

    corr: dict[str, pd.DataFrame] = {}
    delta: dict[str, pd.DataFrame] = {}
    angles: dict[str, dict[str, np.ndarray]] = {}
    pair_labels: list[str] = []

    for cond_a, cond_b in pair_list:
        label = f"{cond_a}vs{cond_b}"
        console.rule(label, style="dim")
        _t = time.time()

        def _one(ses: str, _a: str = cond_a, _b: str = cond_b):
            ang = load_map(angle_dir, sub, ses, task, space, _a, _b, "angle",
                           hemi_list)
            mag = None
            if min_magnitude > 0:
                mag = load_map(angle_dir, sub, ses, task, space, _a, _b,
                               "magnitude", hemi_list)
            return ses, ang, mag

        maps: dict[str, np.ndarray] = {}
        mags: dict[str, np.ndarray] = {}
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_one, s): s for s in ses_list}
                for fut in as_completed(futures):
                    ses, ang, mag = fut.result()
                    if ang is not None:
                        maps[ses] = ang
                    if mag is not None:
                        mags[ses] = mag
        else:
            for s in ses_list:
                ses, ang, mag = _one(s)
                if ang is not None:
                    maps[ses] = ang
                if mag is not None:
                    mags[ses] = mag

        missing = [s for s in ses_list if s not in maps]
        if missing:
            console.print(f"  [yellow]SKIP[/yellow] no angle map for ses-"
                          f"{', ses-'.join(missing)}")
        if len(maps) < 2:
            console.print(f"  [yellow]SKIP[/yellow] {label}: fewer than 2 sessions")
            continue

        shapes = {v.shape[0] for v in maps.values()}
        if len(shapes) > 1:
            console.print(f"  [red]ERROR[/red] vertex counts differ across "
                          f"sessions {sorted(shapes)} — skipping {label}")
            continue

        ordered = {s: maps[s] for s in ses_list if s in maps}

        if min_magnitude > 0 and len(mags) == len(ordered):
            keep = np.ones(next(iter(ordered.values())).shape[0], dtype=bool)
            for s in ordered:
                keep &= np.isfinite(mags[s]) & (mags[s] > min_magnitude)
            console.print(f"  magnitude mask keeps {keep.sum():,} / {keep.size:,} vertices")
            ordered = {s: np.where(keep, v, np.nan) for s, v in ordered.items()}
        elif min_magnitude > 0:
            console.print("  [yellow]WARNING[/yellow] magnitude maps missing — "
                          "mask not applied")

        n_valid = int(np.isfinite(np.vstack(list(ordered.values()))).all(axis=0).sum())
        console.print(f"  {len(ordered)} sessions, {n_valid:,} vertices valid in all")

        corr[label] = correlation_matrix(ordered)
        delta[label] = delta_matrix(ordered)
        angles[label] = ordered
        pair_labels.append(label)

        c, d = _offdiag(corr[label]), _offdiag(delta[label])
        console.print(f"  Pearson r      mean {np.nanmean(c):.3f}  "
                      f"min {np.nanmin(c):.3f}  max {np.nanmax(c):.3f}")
        console.print(f"  median |Δangle| mean {np.nanmean(d):.1f}°  "
                      f"max {np.nanmax(d):.1f}°  [dim]({time.time() - _t:.1f} s)[/dim]")
        console.print()

    if not pair_labels:
        console.print("[red]ERROR[/red] no condition pair had >= 2 usable sessions")
        raise typer.Exit(1)

    # ── Write outputs ────────────────────────────────────────────────────────
    console.rule("outputs", style="dim")
    stem = f"sub-{sub}_task-{task}_space-{space}"
    found_ses = list(angles[pair_labels[0]])

    for label in pair_labels:
        p = op.join(out_dir, f"{stem}_anglecorr-{label}.tsv")
        corr[label].round(6).to_csv(p, sep="\t")
        console.print(f"  [green]saved[/green] {op.basename(p)}")
        p = op.join(out_dir, f"{stem}_angledelta-{label}.tsv")
        delta[label].round(4).to_csv(p, sep="\t")
        console.print(f"  [green]saved[/green] {op.basename(p)}")

    summary = pd.DataFrame({
        "pair": pair_labels,
        "mean_r": [np.nanmean(_offdiag(corr[p])) for p in pair_labels],
        "min_r": [np.nanmin(_offdiag(corr[p])) for p in pair_labels],
        "mean_median_abs_delta_deg": [np.nanmean(_offdiag(delta[p])) for p in pair_labels],
        "max_median_abs_delta_deg": [np.nanmax(_offdiag(delta[p])) for p in pair_labels],
    })
    p = op.join(out_dir, f"{stem}_angle-stability-summary.tsv")
    summary.round(4).to_csv(p, sep="\t", index=False)
    console.print(f"  [green]saved[/green] {op.basename(p)}")

    out_png = op.join(out_dir, f"{stem}_angle-cross-session.png")
    make_figure(found_ses, pair_labels, corr, delta, angles, sub, out_png)
    console.print(f"  [green]saved[/green] {op.basename(out_png)}")

    console.print()
    console.rule(
        f"Done {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"({time.time() - t0:.0f} s)"
    )


if __name__ == "__main__":
    app()
