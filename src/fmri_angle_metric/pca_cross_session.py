"""Compare PCA solutions across sessions for one subject.

Reads the per-session outputs of ``pca_response_profile.py`` and reports how
stable the components are across sessions: variance explained, condition
loadings, and — optionally — the per-vertex score maps.

Note on scope: the shared convention is one subject x one session per call.
This script is inherently cross-session, so its unit of work is one subject
across many sessions. The batch loop over *subjects* still belongs in the
launcher.

Sign handling
-------------
PCA component signs are arbitrary: sklearn may return PC1 as "RW-positive" in
one session and "RW-negative" in the next, describing the same axis. Every
session is therefore sign-aligned to a reference session (default: the first)
by flipping any component whose loading vector points away from the reference.
Loadings and score maps are flipped together, so they stay consistent.
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
# (python pca_cross_session.py), where src/ is not on sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fmri_angle_metric._viz import (
    DIVERGING,
    GRID,
    INK,
    INK_MUTED,
    SERIES,
    SURFACE,
    heatmap,
    session_legend,
    style_axes,
    titles,
)

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


# ── loading per-session PCA outputs ──────────────────────────────────────────

def _find_one(pattern: str) -> str | None:
    """Return the single file matching ``pattern``, or None."""
    matches = sorted(glob.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        console.print(f"  [yellow]WARNING[/yellow] {len(matches)} matches for "
                      f"{op.basename(pattern)} — using first")
        return matches[0]
    return None


def load_session(pca_dir: str, sub: str, ses: str, normalize: str) -> dict | None:
    """Load the loadings and variance tables for one session."""
    ses_dir = op.join(pca_dir, f"sub-{sub}", f"ses-{ses}")
    if not op.isdir(ses_dir):
        console.print(f"  [yellow]SKIP[/yellow] ses-{ses}: no directory {ses_dir}")
        return None

    load_path = _find_one(op.join(ses_dir, f"*desc-norm{normalize}_pca_loadings.tsv"))
    var_path = _find_one(op.join(ses_dir, f"*desc-norm{normalize}_pca_variance.tsv"))

    if load_path is None or var_path is None:
        console.print(f"  [yellow]SKIP[/yellow] ses-{ses}: missing loadings/variance TSV")
        return None

    loadings = pd.read_csv(load_path, sep="\t", index_col=0)
    variance = pd.read_csv(var_path, sep="\t")

    return {"ses": ses, "dir": ses_dir, "loadings": loadings, "variance": variance}


def load_scores(ses_dir: str, sub: str, ses: str, task: str, space: str,
                normalize: str, pc: int, hemis: list[str]) -> np.ndarray | None:
    """Concatenate the L/R score maps for one PC into a single vertex vector."""
    parts = []
    for hemi in hemis:
        pattern = op.join(
            ses_dir,
            f"sub-{sub}_ses-{ses}_task-{task}_hemi-{hemi}_space-{space}"
            f"_desc-norm{normalize}_stat-PC{pc}score_statmap.func.gii",
        )
        path = _find_one(pattern)
        if path is None:
            return None
        parts.append(np.asarray(load_surf_data(path), dtype=np.float64).ravel())
    return np.concatenate(parts)


# ── sign alignment ───────────────────────────────────────────────────────────

def align_signs(loadings: dict[str, pd.DataFrame], ref_ses: str,
                pcs: list[str]) -> dict[str, np.ndarray]:
    """Flip components that point away from the reference session.

    Returns a {session: array of +1/-1 per PC} map, applied to loadings in
    place and reusable for the score maps.
    """
    ref = loadings[ref_ses]
    flips: dict[str, np.ndarray] = {}

    for ses, df in loadings.items():
        sign = np.ones(len(pcs))
        for i, pc in enumerate(pcs):
            if pc not in df.index or pc not in ref.index:
                continue
            if float(np.dot(df.loc[pc].values, ref.loc[pc].values)) < 0:
                sign[i] = -1.0
                df.loc[pc] = -df.loc[pc]
        flips[ses] = sign
        flipped = [pcs[i] for i in range(len(pcs)) if sign[i] < 0]
        if flipped:
            console.print(f"  ses-{ses}: flipped {', '.join(flipped)} to match "
                          f"ses-{ref_ses}")
    return flips


def cosine_matrix(vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise cosine similarity between session loading vectors."""
    sessions = list(vectors)
    mat = np.full((len(sessions), len(sessions)), np.nan)
    for i, a in enumerate(sessions):
        for j, b in enumerate(sessions):
            va, vb = vectors[a], vectors[b]
            denom = np.linalg.norm(va) * np.linalg.norm(vb)
            mat[i, j] = float(np.dot(va, vb) / denom) if denom > 0 else np.nan
    return pd.DataFrame(mat, index=sessions, columns=sessions)


def correlation_matrix(maps: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Pearson r between session score maps, over co-valid vertices."""
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


# ── plotting ─────────────────────────────────────────────────────────────────

def make_figure(sessions: list[str], var_df: pd.DataFrame,
                loadings: dict[str, pd.DataFrame], pcs: list[str],
                sim: dict[str, pd.DataFrame],
                score_corr: dict[str, pd.DataFrame],
                sub: str, normalize: str, ref_ses: str, out_png: str) -> None:
    n_pcs = len(pcs)
    has_scores = bool(score_corr)

    # One column for the variance summary, then one per PC. A third row only
    # exists when score maps were compared, so there are never empty panels.
    n_cols = 1 + n_pcs
    n_rows = 3 if has_scores else 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.9 * n_cols, 4.5 * n_rows),
                             facecolor=SURFACE, constrained_layout=True)
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(SURFACE)

    colors = {ses: SERIES[i % len(SERIES)] for i, ses in enumerate(sessions)}
    heatmap_axes = []
    im_ref = None

    # ── Panel: variance explained per PC, one line per session ───────────────
    ax = axes[0, 0]
    style_axes(ax)
    x = np.arange(len(var_df.index))
    for ses in sessions:
        if ses not in var_df.columns:
            continue
        ax.plot(x, var_df[ses].values * 100, marker="o", markersize=7,
                linewidth=2, color=colors[ses], label=f"ses-{ses}")
    ax.set_xticks(x, var_df.index, fontsize=9, color=INK_MUTED)
    ax.set_ylabel("variance explained (%)", fontsize=9, color=INK_MUTED)
    titles(ax, "Variance explained", "each line is one session")

    # ── Panel: spread of variance explained across sessions ──────────────────
    ax = axes[1, 0]
    style_axes(ax)
    means = var_df.mean(axis=1) * 100
    spread = var_df.std(axis=1) * 100
    ax.bar(x, means, yerr=spread, color=SERIES[0], width=0.55,
           error_kw={"ecolor": INK_MUTED, "elinewidth": 1.5, "capsize": 4})
    for xi, (m, s) in enumerate(zip(means, spread)):
        ax.text(xi, m + s + 1.2, f"{m:.1f}±{s:.1f}", ha="center",
                fontsize=8.5, color=INK)
    ax.set_xticks(x, var_df.index, fontsize=9, color=INK_MUTED)
    ax.set_ylabel("variance explained (%)", fontsize=9, color=INK_MUTED)
    titles(ax, "Mean across sessions", "bars are mean, whiskers ±1 SD")

    # ── Panels: loadings per PC, one line per session ────────────────────────
    conds = list(loadings[sessions[0]].columns)
    cx = np.arange(len(conds))
    for k, pc in enumerate(pcs):
        ax = axes[0, 1 + k]
        style_axes(ax)
        ax.axhline(0, color=GRID, linewidth=1.5)
        for ses in sessions:
            df = loadings[ses]
            if pc not in df.index:
                continue
            ax.plot(cx, df.loc[pc].values, marker="o", markersize=7,
                    linewidth=2, color=colors[ses], label=f"ses-{ses}")
        ax.set_xticks(cx, conds, fontsize=9, color=INK_MUTED, rotation=30,
                      ha="right")
        ax.set_ylabel("loading", fontsize=9, color=INK_MUTED)
        titles(ax, f"{pc} loadings", f"sign-aligned to ses-{ref_ses}")

    # ── Panels: loading similarity matrices ──────────────────────────────────
    for k, pc in enumerate(pcs):
        ax = axes[1, 1 + k]
        im_ref = heatmap(ax, sim[pc], f"{pc} loading similarity",
                         "cosine similarity between sessions",
                         vmin=-1, vmax=1, cmap=DIVERGING)
        heatmap_axes.append(ax)

    # ── Panels: score-map correlation ────────────────────────────────────────
    if has_scores:
        for k, pc in enumerate(pcs):
            ax = axes[2, 1 + k]
            if pc not in score_corr:
                ax.axis("off")
                continue
            im_ref = heatmap(ax, score_corr[pc], f"{pc} score-map correlation",
                             "Pearson r over co-valid vertices",
                             vmin=-1, vmax=1, cmap=DIVERGING)
            heatmap_axes.append(ax)

        # Summary text block instead of an empty axes in the last row.
        ax = axes[2, 0]
        ax.set_facecolor(SURFACE)
        ax.axis("off")
        lines = ["Off-diagonal summary", ""]
        for pc in pcs:
            off = sim[pc].values[np.triu_indices(len(sessions), k=1)]
            lines.append(f"{pc} loadings   mean {np.nanmean(off):.3f}   "
                         f"min {np.nanmin(off):.3f}")
        for pc in pcs:
            if pc not in score_corr:
                continue
            off = score_corr[pc].values[np.triu_indices(len(score_corr[pc]), k=1)]
            lines.append(f"{pc} score maps mean {np.nanmean(off):.3f}   "
                         f"min {np.nanmin(off):.3f}")
        ax.text(0, 0.98, lines[0], transform=ax.transAxes, fontsize=11,
                color=INK, weight="bold", va="top")
        ax.text(0, 0.84, "\n".join(lines[2:]), transform=ax.transAxes,
                fontsize=9.5, color=INK_MUTED, va="top", family="monospace",
                linespacing=1.9)

    # One shared colorbar for every matrix — they all use the same -1..1 scale.
    if im_ref is not None and heatmap_axes:
        cbar = fig.colorbar(im_ref, ax=heatmap_axes, fraction=0.02, pad=0.01)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
        cbar.set_label("similarity / r", fontsize=8.5, color=INK_MUTED)

    session_legend(fig, sessions, colors)

    fig.suptitle(
        f"sub-{sub} — PCA stability across sessions  (normalize={normalize})",
        fontsize=13.5, color=INK, x=0.006, ha="left", weight="bold",
    )
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    pca_dir: str = typer.Option(
        ..., "--pca-dir",
        help="Root of the PCA output tree (contains sub-XX/ses-YY/).",
    ),
    sub: str = typer.Option(..., "--sub", help="Subject label, e.g. 02"),
    sessions: str = typer.Option(
        ..., "--sessions",
        help="Comma-separated session labels, e.g. '01,02,03,04,05,06'.",
    ),
    task: str = typer.Option("fLoc", "--task"),
    space: str = typer.Option("fsnative", "--space"),
    normalize: str = typer.Option(
        "zscore", "--normalize",
        help="Must match the --normalize used by pca_response_profile.py "
             "(it appears in the filenames as desc-norm<value>).",
    ),
    n_pcs: int = typer.Option(2, "--n-pcs", help="How many PCs to compare."),
    ref_ses: str = typer.Option(
        "", "--ref-ses",
        help="Session used as the sign reference. Defaults to the first found.",
    ),
    compare_scores: bool = typer.Option(
        True, "--compare-scores/--no-compare-scores",
        help="Also correlate the per-vertex PC score maps across sessions. "
             "Requires the score GIFTIs and identical vertex counts.",
    ),
    hemis: str = typer.Option("L,R", "--hemis"),
    out_dir: str = typer.Option(..., "--out-dir", help="Where to write the figure and TSVs."),
    n_workers: int = typer.Option(1, "--n-workers", help="Threads for loading score GIFTIs."),
) -> None:
    sub = sub.strip().zfill(2)
    ses_list = [s.strip().zfill(2) for s in sessions.split(",") if s.strip()]
    hemi_list = [h.strip() for h in hemis.split(",") if h.strip()]

    console.rule(f"[bold]pca-cross-session — sub-{sub}[/bold]")
    t0 = time.time()

    if not op.isdir(pca_dir):
        console.print(f"[red]ERROR[/red] --pca-dir does not exist: {pca_dir}")
        raise typer.Exit(1)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── Load every session ───────────────────────────────────────────────────
    console.print(f"  sessions requested : {', '.join(ses_list)}")
    loaded = [load_session(pca_dir, sub, s, normalize) for s in ses_list]
    loaded = [d for d in loaded if d is not None]

    if len(loaded) < 2:
        console.print(f"[red]ERROR[/red] need >= 2 sessions with PCA output, "
                      f"found {len(loaded)}")
        raise typer.Exit(1)

    found_ses = [d["ses"] for d in loaded]
    console.print(f"  sessions loaded    : {', '.join(found_ses)}")

    # Condition set must match across sessions or the loadings aren't comparable.
    cond_sets = {d["ses"]: tuple(d["loadings"].columns) for d in loaded}
    ref_conds = cond_sets[found_ses[0]]
    mismatched = [s for s, c in cond_sets.items() if c != ref_conds]
    if mismatched:
        console.print(f"[red]ERROR[/red] condition set differs in ses-"
                      f"{', ses-'.join(mismatched)}; cannot compare loadings")
        for s, c in cond_sets.items():
            console.print(f"    ses-{s}: {list(c)}")
        raise typer.Exit(1)
    console.print(f"  conditions         : {list(ref_conds)}")

    available_pcs = min(len(d["loadings"].index) for d in loaded)
    use_n = min(n_pcs, available_pcs)
    if use_n < n_pcs:
        console.print(f"  [yellow]INFO[/yellow] only {available_pcs} PCs available, "
                      f"comparing {use_n}")
    pcs = [f"PC{i + 1}" for i in range(use_n)]

    ref = ref_ses.strip().zfill(2) if ref_ses.strip() else found_ses[0]
    if ref not in found_ses:
        console.print(f"[red]ERROR[/red] --ref-ses {ref} not among loaded sessions")
        raise typer.Exit(1)

    # ── Sign-align ───────────────────────────────────────────────────────────
    console.print()
    console.rule("sign alignment", style="dim")
    loadings = {d["ses"]: d["loadings"].copy() for d in loaded}
    flips = align_signs(loadings, ref, pcs)
    if not any((f < 0).any() for f in flips.values()):
        console.print(f"  all sessions already matched ses-{ref}")

    # ── Variance table ───────────────────────────────────────────────────────
    var_df = pd.DataFrame(
        {d["ses"]: d["variance"].set_index("PC")["variance_explained"] for d in loaded}
    ).loc[pcs]

    console.print()
    console.rule("variance explained", style="dim")
    for pc in pcs:
        row = var_df.loc[pc] * 100
        console.print(f"  {pc}  mean {row.mean():5.1f}%  SD {row.std():4.1f}%  "
                      f"range {row.min():5.1f}–{row.max():5.1f}%")

    # ── Loading similarity ───────────────────────────────────────────────────
    sim = {pc: cosine_matrix({s: loadings[s].loc[pc].values for s in found_ses})
           for pc in pcs}

    console.print()
    console.rule("loading similarity (cosine)", style="dim")
    for pc in pcs:
        off = sim[pc].values[np.triu_indices(len(found_ses), k=1)]
        console.print(f"  {pc}  mean {np.nanmean(off):.3f}  "
                      f"min {np.nanmin(off):.3f}  max {np.nanmax(off):.3f}")

    # ── Score-map correlation ────────────────────────────────────────────────
    score_corr: dict[str, pd.DataFrame] = {}
    if compare_scores:
        console.print()
        console.rule("score-map correlation", style="dim")
        dirs = {d["ses"]: d["dir"] for d in loaded}

        for pc_idx, pc in enumerate(pcs, start=1):
            maps: dict[str, np.ndarray] = {}

            def _one(ses: str, _pc: int = pc_idx) -> tuple[str, np.ndarray | None]:
                return ses, load_scores(dirs[ses], sub, ses, task, space,
                                        normalize, _pc, hemi_list)

            _t = time.time()
            if n_workers > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(_one, s): s for s in found_ses}
                    for fut in as_completed(futures):
                        ses, arr = fut.result()
                        if arr is not None:
                            maps[ses] = arr
            else:
                for s in found_ses:
                    ses, arr = _one(s)
                    if arr is not None:
                        maps[ses] = arr

            missing = [s for s in found_ses if s not in maps]
            if missing:
                console.print(f"  [yellow]SKIP[/yellow] {pc}: no score GIFTI for ses-"
                              f"{', ses-'.join(missing)}")
            if len(maps) < 2:
                continue

            shapes = {arr.shape[0] for arr in maps.values()}
            if len(shapes) > 1:
                console.print(f"  [yellow]SKIP[/yellow] {pc}: vertex counts differ "
                              f"across sessions {sorted(shapes)}")
                continue

            # Apply the same sign flips used for the loadings.
            ordered = {s: maps[s] * flips[s][pc_idx - 1] for s in found_ses if s in maps}
            score_corr[pc] = correlation_matrix(ordered)

            off = score_corr[pc].values[np.triu_indices(len(ordered), k=1)]
            console.print(f"  {pc}  mean r {np.nanmean(off):.3f}  "
                          f"min {np.nanmin(off):.3f}  max {np.nanmax(off):.3f}  "
                          f"[dim]({time.time() - _t:.1f} s)[/dim]")

    # ── Write outputs ────────────────────────────────────────────────────────
    console.print()
    console.rule("outputs", style="dim")
    stem = f"sub-{sub}_task-{task}_space-{space}_desc-norm{normalize}"

    var_out = op.join(out_dir, f"{stem}_variance-across-sessions.tsv")
    (var_df * 100).round(3).to_csv(var_out, sep="\t")
    console.print(f"  [green]saved[/green] {op.basename(var_out)}")

    load_out = op.join(out_dir, f"{stem}_loadings-aligned.tsv")
    long = pd.concat({s: loadings[s].loc[pcs] for s in found_ses}, names=["ses", "PC"])
    long.round(6).to_csv(load_out, sep="\t")
    console.print(f"  [green]saved[/green] {op.basename(load_out)}")

    for pc in pcs:
        p = op.join(out_dir, f"{stem}_similarity-{pc}.tsv")
        sim[pc].round(6).to_csv(p, sep="\t")
        console.print(f"  [green]saved[/green] {op.basename(p)}")

    for pc, df in score_corr.items():
        p = op.join(out_dir, f"{stem}_scorecorr-{pc}.tsv")
        df.round(6).to_csv(p, sep="\t")
        console.print(f"  [green]saved[/green] {op.basename(p)}")

    out_png = op.join(out_dir, f"{stem}_pca-cross-session.png")
    make_figure(found_ses, var_df, loadings, pcs, sim, score_corr, sub,
                normalize, ref, out_png)
    console.print(f"  [green]saved[/green] {op.basename(out_png)}")

    console.print()
    console.rule(
        f"Done {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"({time.time() - t0:.0f} s)"
    )


if __name__ == "__main__":
    app()
