"""
The st-domains method, after the BANKSY features (processor/features.py):

  load        each sample's X (log-normalized, CSR), coordinates, annotation and lowres image;
              genes = those present in every sample
  genes       the genes BANKSY uses: all of them for panels of <= 1,000 genes, else st-qc's
              highly variable genes pooled over samples (scanpy's batch_key rule: genes flagged
              in the most samples first, then by mean HVG rank), as many as the median sample flags
  cluster     BANKSY PCA -> Harmony by sample (INTEGRATION=harmony, > 1 sample) -> kNN graph ->
              Leiden (igraph, 2 iterations) at RESOLUTION, or a resolution bisection for N_DOMAINS
  name        domains D01, D02, ... by size (largest first), shared across samples
  markers     Wilcoxon rank-sum, each domain against the rest, on every shared gene
  agreement   ARI / NMI with obs["annotation"], confusion counts, best-matching domain per class
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger("st-domains")

SMALL_PANEL = 1000                  # panels with at most this many shared genes use every gene
DEFAULT_N_HVG = 2000                # when no sample carries var["highly_variable"]
MARKER_OBS_PER_DOMAIN = 10000       # seeded cap per domain for the Wilcoxon test (memory / time)
RESOLUTION_RANGE = (0.05, 3.0)
MAX_RESOLUTION_RUNS = 12
MIN_OBS_PRESENT = 20                # a domain "is present" in a sample with at least this many observations
N_NEIGHBORS = 15                    # expression-space kNN graph for Leiden (scanpy default)


@dataclass
class SampleData:
    sample_id: str
    obs_names: np.ndarray
    x: sp.csr_matrix                # log-normalized, shared genes only (set by restrict())
    var_names: np.ndarray
    xy: np.ndarray
    annotation: np.ndarray | None
    hvg: pd.DataFrame               # highly_variable, highly_variable_rank (may be NaN)
    image: np.ndarray | None = None
    lowres_scalef: float | None = None
    spot_diameter: float | None = None   # in coordinate units
    platform: str = ""
    kind: str = ""
    coordinate_unit: str = ""
    normalized_here: bool = False
    info: dict = field(default_factory=dict)


def _read(group):
    from anndata.io import read_elem

    return read_elem(group)


def load_sample(sample_id: str, path: Path) -> SampleData:
    """Read only what st-domains needs (X, obs, var, coordinates, the lowres image), not layers."""
    import h5py

    with h5py.File(path, "r") as f:
        x = _read(f["X"])
        obs = _read(f["obs"])
        var = _read(f["var"])
        if "obsm" not in f or "spatial" not in f["obsm"]:
            raise ValueError("no obsm['spatial']")
        xy = np.asarray(_read(f["obsm"]["spatial"]), dtype=np.float64)[:, :2]
        uns = f["uns"] if "uns" in f else None
        pen = _read(uns["pennsieve"]) if uns is not None and "pennsieve" in uns else {}
        image, scalef, spot = None, None, None
        if uns is not None and "spatial" in uns and len(uns["spatial"].keys()):
            key = sample_id if sample_id in uns["spatial"] else list(uns["spatial"].keys())[0]
            g = uns["spatial"][key]
            if "images" in g and "lowres" in g["images"]:
                image = np.asarray(g["images"]["lowres"][()])
                sf = _read(g["scalefactors"]) if "scalefactors" in g else {}
                scalef = float(sf.get("tissue_lowres_scalef")) if "tissue_lowres_scalef" in sf else None
                if scalef is None:
                    image = None
                spot = sf.get("spot_diameter_fullres")
    if xy.shape[0] != len(obs):
        raise ValueError("obsm['spatial'] and obs differ in length")
    if not np.isfinite(xy).all():
        raise ValueError("non-finite coordinates")
    x = sp.csr_matrix(x, dtype=np.float32)
    normalized_here = False
    if not bool(pen.get("normalized", False)):
        # st-prep output (raw counts) linked directly: normalize as st-qc would (median target, log1p).
        if x.nnz and float(np.max(x.data)) > 50 and np.allclose(x.data[:1000], np.round(x.data[:1000])):
            totals = np.asarray(x.sum(axis=1)).ravel()
            target = float(np.median(totals[totals > 0])) if (totals > 0).any() else 1.0
            x = sp.csr_matrix(sp.diags((target / np.maximum(totals, 1e-12)).astype(np.float32)) @ x)
            x.data = np.log1p(x.data).astype(np.float32)
            normalized_here = True
    hv = pd.DataFrame(index=var.index.astype(str))
    hv["highly_variable"] = var["highly_variable"].astype(bool).to_numpy() if "highly_variable" in var else False
    hv["highly_variable_rank"] = (pd.to_numeric(var["highly_variable_rank"], errors="coerce").to_numpy()
                                  if "highly_variable_rank" in var else np.nan)
    ann = None
    if "annotation" in obs.columns:
        a = obs["annotation"].astype(object).to_numpy()
        ann = np.array(["" if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v) for v in a], dtype=object)
    spot_d = pen.get("spot_diameter", spot)
    return SampleData(
        sample_id=sample_id, obs_names=np.asarray(obs.index.astype(str)), x=x,
        var_names=np.asarray(var.index.astype(str)), xy=xy, annotation=ann, hvg=hv, image=image,
        lowres_scalef=scalef, spot_diameter=float(spot_d) if spot_d not in (None, "") else None,
        platform=str(pen.get("platform", "")), kind=str(pen.get("kind", "")),
        coordinate_unit=str(pen.get("coordinate_unit", "")), normalized_here=normalized_here)


def shared_genes(samples: list[SampleData]) -> list[str]:
    common = set(samples[0].var_names)
    for s in samples[1:]:
        common &= set(s.var_names)
    return [g for g in samples[0].var_names if g in common]


def restrict(s: SampleData, genes: list[str]) -> None:
    idx = pd.Index(s.var_names).get_indexer(genes)
    s.x = sp.csr_matrix(s.x[:, idx])
    s.hvg = s.hvg.iloc[idx]
    s.var_names = np.asarray(genes)


def choose_genes(samples: list[SampleData], genes: list[str]) -> tuple[list[str], str]:
    """The genes BANKSY uses, and how they were chosen."""
    if len(genes) <= SMALL_PANEL:
        return list(genes), f"all {len(genes)} shared genes (panel <= {SMALL_PANEL})"
    flagged = [s for s in samples if s.hvg["highly_variable"].any() and not s.hvg["highly_variable"].all()]
    if not flagged:
        return [], "none flagged"
    n_flag = np.zeros(len(genes))
    rank_sum = np.zeros(len(genes))
    for s in flagged:
        hv = s.hvg["highly_variable"].to_numpy()
        n_flag += hv
        r = s.hvg["highly_variable_rank"].to_numpy(dtype=float)
        r = np.where(hv & np.isfinite(r), r, np.nan)
        rank_sum += np.where(np.isfinite(r), r, 0.0)
    mean_rank = np.where(n_flag > 0, rank_sum / np.maximum(n_flag, 1), np.inf)
    n_top = int(np.median([int(s.hvg["highly_variable"].sum()) for s in flagged]))
    order = np.lexsort((np.arange(len(genes)), mean_rank, -n_flag))
    chosen = [genes[i] for i in order[:n_top] if n_flag[i] > 0]
    return chosen, (f"{len(chosen)} highly variable genes from st-qc (var['highly_variable'] of {len(flagged)} "
                    f"sample(s); flagged in most samples first, then by mean rank)")


def compute_hvg_fallback(x: sp.csr_matrix, batch: np.ndarray, genes: list[str], n_top: int) -> list[str]:
    """No HVG flags upstream: scanpy's seurat-flavour HVGs on the log data, by sample."""
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(X=x, obs=pd.DataFrame({"sample_id": pd.Categorical(batch)}), var=pd.DataFrame(index=genes))
    sc.pp.highly_variable_genes(a, n_top_genes=min(n_top, len(genes)), flavor="seurat",
                                batch_key="sample_id" if len(set(batch)) > 1 else None)
    return [g for g, h in zip(genes, a.var["highly_variable"]) if h]


def harmony(emb: np.ndarray, batch: np.ndarray, seed: int) -> np.ndarray:
    import harmonypy

    meta = pd.DataFrame({"sample_id": pd.Categorical(batch)})
    ho = harmonypy.run_harmony(emb.astype(np.float64), meta, "sample_id", random_state=seed, verbose=False)
    z = np.asarray(ho.Z_corr)
    if z.shape[0] != emb.shape[0]:
        z = z.T
    return z.astype(np.float32)


def leiden(adata, resolution: float, seed: int) -> np.ndarray:
    import scanpy as sc

    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2, random_state=seed,
                 directed=False, key_added="_leiden")
    return adata.obs["_leiden"].astype(str).to_numpy()


def cluster(adata, resolution: float, n_domains: int, seed: int) -> tuple[np.ndarray, float, list[dict]]:
    """Leiden at RESOLUTION, or bisect the resolution for N_DOMAINS (closest count wins, lower resolution on ties)."""
    runs: list[dict] = []

    def run(r: float) -> np.ndarray:
        lab = leiden(adata, r, seed)
        k = len(set(lab))
        runs.append({"resolution": round(r, 6), "n_domains": k})
        log.info("leiden resolution %.4f: %d domains", r, k)
        return lab

    if n_domains <= 0:
        return run(resolution), resolution, runs
    lo, hi = RESOLUTION_RANGE
    best: tuple[int, float, np.ndarray] | None = None
    for _ in range(MAX_RESOLUTION_RUNS):
        mid = (lo + hi) / 2.0
        lab = run(mid)
        k = len(set(lab))
        cand = (abs(k - n_domains), mid, lab)
        if best is None or (cand[0], cand[1]) < (best[0], best[1]):
            best = cand
        if k == n_domains:
            break
        if k < n_domains:
            lo = mid
        else:
            hi = mid
    assert best is not None
    if best[0] != 0:
        log.warning("N_DOMAINS=%d not reached in %d runs; closest: %d domains at resolution %.4f",
                    n_domains, len(runs), len(set(best[2])), best[1])
    return best[2], best[1], runs


def name_domains(raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """D01, D02, ... by size, largest first (ties by the Leiden label)."""
    vc = pd.Series(raw).value_counts()
    keys = sorted(vc.index, key=lambda k: (-vc[k], int(k) if str(k).isdigit() else str(k)))
    width = max(2, len(str(len(keys))))
    names = {k: f"D{i + 1:0{width}d}" for i, k in enumerate(keys)}
    out = np.array([names[k] for k in raw], dtype=object)
    return out, [names[k] for k in keys]


def markers(x: sp.csr_matrix, genes: list[str], domain: np.ndarray, order: list[str], n_markers: int,
            seed: int):
    """Wilcoxon rank-sum per domain vs rest (scanpy), top n_markers by score; pct_in / pct_out on every observation."""
    import anndata as ad
    import scanpy as sc

    cols = ["domain", "gene", "logfoldchange", "pval_adj", "pct_in", "pct_out", "rank"]
    if len(order) < 2 or n_markers <= 0:
        return pd.DataFrame(columns=cols), None
    rng = np.random.default_rng(seed)
    keep = []
    for d in order:
        idx = np.flatnonzero(domain == d)
        if len(idx) > MARKER_OBS_PER_DOMAIN:
            idx = np.sort(rng.choice(idx, MARKER_OBS_PER_DOMAIN, replace=False))
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    a = ad.AnnData(X=x[keep], obs=pd.DataFrame({"domain": pd.Categorical(domain[keep], categories=order)}),
                   var=pd.DataFrame(index=genes))
    a.uns["log1p"] = {"base": None}
    sc.tl.rank_genes_groups(a, "domain", method="wilcoxon", use_raw=False, n_genes=min(n_markers, len(genes)),
                            tie_correct=False, pts=False)
    rg = a.uns["rank_genes_groups"]
    # detection rates on every observation
    present = (x > 0).astype(np.float32)
    gene_idx = {g: i for i, g in enumerate(genes)}
    total = np.asarray(present.sum(axis=0)).ravel()
    rows = []
    for d in order:
        m = domain == d
        n_in, n_out = int(m.sum()), int((~m).sum())
        det_in = np.asarray(present[np.flatnonzero(m)].sum(axis=0)).ravel()
        for r, (g, lfc, padj) in enumerate(zip(rg["names"][d], rg["logfoldchanges"][d], rg["pvals_adj"][d])):
            i = gene_idx[g]
            rows.append({"domain": d, "gene": g, "logfoldchange": float(lfc), "pval_adj": float(padj),
                         "pct_in": det_in[i] / max(n_in, 1), "pct_out": (total[i] - det_in[i]) / max(n_out, 1),
                         "rank": r + 1})
    return pd.DataFrame(rows, columns=cols), a


def agreement(sample_ids: np.ndarray, annotation: np.ndarray, domain: np.ndarray, order_samples: list[str],
              ignore: set[str]) -> dict:
    """ARI / NMI per sample and pooled, confusion counts and the best-matching domain per class."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ok = np.array([bool(a) and a not in ignore for a in annotation])
    rows = []
    for sid in order_samples + ["all"]:
        m = ok & ((sample_ids == sid) if sid != "all" else True)
        n = int(m.sum())
        if n < 2:
            rows.append({"sample_id": sid, "ari": float("nan"), "nmi": float("nan"), "n_obs": n,
                         "n_classes": len(set(annotation[m])), "n_domains": len(set(domain[m]))})
            continue
        rows.append({"sample_id": sid, "ari": float(adjusted_rand_score(annotation[m], domain[m])),
                     "nmi": float(normalized_mutual_info_score(annotation[m], domain[m])), "n_obs": n,
                     "n_classes": len(set(annotation[m])), "n_domains": len(set(domain[m]))})
    conf = []
    for sid in order_samples + ["all"]:
        m = ok & ((sample_ids == sid) if sid != "all" else True)
        if not m.any():
            continue
        ct = pd.crosstab(pd.Series(annotation[m], name="annotation"), pd.Series(domain[m], name="domain"))
        for a_ in ct.index:
            for d in ct.columns:
                conf.append({"sample_id": sid, "annotation": a_, "domain": d, "n_obs": int(ct.loc[a_, d])})
    matches = []
    if ok.any():
        ct = pd.crosstab(annotation[ok], domain[ok])
        n_dom = ct.sum(axis=0)
        for a_ in ct.index:
            n_cls = int(ct.loc[a_].sum())
            best = None
            for d in ct.columns:
                ov = int(ct.loc[a_, d])
                prec, rec = ov / n_dom[d], ov / n_cls
                f1 = 0.0 if ov == 0 else 2 * prec * rec / (prec + rec)
                if best is None or f1 > best["f1"]:
                    best = {"annotation": a_, "domain": d, "n_annotation": n_cls, "n_domain": int(n_dom[d]),
                            "n_overlap": ov, "precision": prec, "recall": rec, "f1": f1}
            matches.append(best)
    return {"agreement": rows, "confusion": conf, "matches": matches, "n_used": int(ok.sum()),
            "n_ignored": int((~ok).sum())}
