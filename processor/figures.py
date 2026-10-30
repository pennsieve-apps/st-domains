"""Figures (PNG, 150 dpi). Every domain keeps one colour across all figures."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger("st-domains")
DPI = 150
INK, MUTED = "#1f2328", "#59636e"


def domain_colours(order: list[str]) -> dict[str, str]:
    """scanpy's categorical palettes (20 / 28 / 102 colours), assigned in domain order, never cycled."""
    from scanpy.plotting import palettes

    pal = palettes.default_20 if len(order) <= 20 else palettes.default_28 if len(order) <= 28 else palettes.default_102
    if len(order) > len(pal):
        pal = list(pal) + ["#9e9e9e"] * (len(order) - len(pal))
    return {d: pal[i] for i, d in enumerate(order)}


def _legend(ax, order: list[str], colours: dict[str, str], counts: dict[str, int] | None = None, ncol: int = 1):
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], marker="o", ls="", markersize=7, markerfacecolor=colours[d], markeredgecolor="none",
                      label=f"{d} ({counts[d]:,})" if counts else d) for d in order]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8, ncol=ncol,
              labelcolor=INK)


def domain_map(path: Path, sample_id: str, xy: np.ndarray, domain: np.ndarray, order: list[str],
               colours: dict[str, str], image: np.ndarray | None, scalef: float | None, spot_diameter: float | None) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    if image is not None and scalef:
        pts = xy * scalef
        ax.imshow(image, alpha=0.5)
        diam = spot_diameter * scalef if spot_diameter else None
    else:
        pts = xy
        diam = spot_diameter
    present = [d for d in order if (domain == d).any()]
    ax.set_xlim(pts[:, 0].min() - (diam or 0) * 2, pts[:, 0].max() + (diam or 0) * 2)
    ax.set_ylim(pts[:, 1].max() + (diam or 0) * 2, pts[:, 1].min() - (diam or 0) * 2)    # image convention: y down
    ax.set_aspect("equal")
    fig.canvas.draw()
    if diam:
        bbox = ax.get_window_extent()
        pt_per_unit = bbox.width / abs(ax.get_xlim()[1] - ax.get_xlim()[0]) * 72.0 / fig.dpi
        s = max(0.3, (diam * pt_per_unit) ** 2)
    else:
        s = max(0.3, min(20.0, 40000.0 / max(len(xy), 1)))
    for d in present:
        m = domain == d
        ax.scatter(pts[m, 0], pts[m, 1], s=s, c=colours[d], linewidths=0, label=d)
    counts = {d: int((domain == d).sum()) for d in present}
    _legend(ax, present, colours, counts, ncol=1 if len(present) <= 20 else 2)
    ax.set_title(f"{sample_id}: BANKSY domains ({len(xy):,} observations)", color=INK, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def umap(path: Path, emb: np.ndarray, panels: list[tuple[str, np.ndarray, list[str], dict[str, str]]], n_total: int) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.6), squeeze=False)
    s = max(0.5, min(12.0, 30000.0 / max(len(emb), 1)))
    for ax, (title, lab, order, cols) in zip(axes[0], panels):
        for c in order:
            m = lab == c
            if m.any():
                ax.scatter(emb[m, 0], emb[m, 1], s=s, c=cols[c], linewidths=0, rasterized=True)
        counts = {c: int((lab == c).sum()) for c in order if (lab == c).any()}
        _legend(ax, [c for c in order if c in counts], cols, None, ncol=1 if len(counts) <= 20 else 2)
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1", color=MUTED)
        ax.set_ylabel("UMAP 2", color=MUTED)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    sub = "" if len(emb) == n_total else f" (seeded subsample of {len(emb):,} of {n_total:,})"
    fig.suptitle(f"UMAP of the BANKSY embedding{sub}", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def markers_dotplot(path: Path, adata, markers: pd.DataFrame, order: list[str], per_domain: int | None = None) -> None:
    import scanpy as sc

    per_domain = per_domain or (3 if len(order) <= 12 else 2)

    genes: dict[str, list[str]] = {}
    seen: set[str] = set()
    for d in order:
        g = [x for x in markers.loc[markers["domain"] == d].sort_values("rank")["gene"] if x not in seen][:per_domain]
        if g:
            genes[d] = g
            seen.update(g)
    if not genes:
        return
    dp = sc.pl.dotplot(adata, genes, groupby="domain", categories_order=order, standard_scale="var",
                       return_fig=True, show=False, title="Top markers per domain (Wilcoxon, scaled mean expression)")
    dp.savefig(str(path), dpi=DPI, bbox_inches="tight")
    plt.close("all")


def confusion(path: Path, conf: pd.DataFrame, order: list[str]) -> None:
    """Annotation x domain, coloured by the fraction of each annotation class (rows sum to 1), counts written."""
    ct = conf.pivot_table(index="annotation", columns="domain", values="n_obs", aggfunc="sum", fill_value=0)
    ct = ct.reindex(columns=[d for d in order if d in ct.columns])
    frac = ct.div(ct.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(1.2 + 0.55 * ct.shape[1] + 1.5, 1.2 + 0.45 * ct.shape[0] + 0.8))
    im = ax.imshow(frac.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = int(ct.iat[i, j])
            if v:
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=7,
                        color="white" if frac.iat[i, j] > 0.55 else INK)
    ax.set_xticks(range(ct.shape[1]), ct.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(ct.shape[0]), [f"{a} ({int(ct.loc[a].sum()):,})" for a in ct.index], fontsize=8)
    ax.set_xlabel("domain", color=MUTED)
    ax.set_ylabel("annotation", color=MUTED)
    ax.set_title("Annotation vs domain, all samples (colour: fraction of the class)", color=INK, fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("fraction of annotation class", color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def composition(path: Path, summary: pd.DataFrame, samples: list[str], order: list[str], colours: dict[str, str]) -> None:
    """Stacked horizontal bars: each sample's domain fractions (2 px white gaps between segments)."""
    fig, ax = plt.subplots(figsize=(9, 0.9 + 0.42 * len(samples)))
    left = np.zeros(len(samples))
    piv = summary.pivot_table(index="sample_id", columns="domain", values="fraction", fill_value=0).reindex(samples)
    for d in order:
        v = piv[d].to_numpy() if d in piv.columns else np.zeros(len(samples))
        ax.barh(range(len(samples)), v, left=left, color=colours[d], edgecolor="white", linewidth=1.0, height=0.7,
                label=d)
        left += v
    ax.set_yticks(range(len(samples)), samples, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of observations", color=MUTED)
    ax.set_title("Domain composition per sample", color=INK, fontsize=11)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    _legend(ax, order, colours, None, ncol=1 if len(order) <= 20 else 2)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
