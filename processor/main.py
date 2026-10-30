"""
st-domains: spatial domains with BANKSY (Singhal et al. 2024), the step after
st-qc in the Spatial Domain Mapping & Condition Comparison workflow.

Reads st-qc's <sample_id>.h5ad files (port `spatial`) and st-prep's samples.csv
(port `metadata`) and writes, at the root of OUTPUT_DIR (one port, `labels`):

  labels.csv            sample_id, obs_id, domain (D01, D02, ... shared across samples)
  domain_markers.csv    domain, gene, logfoldchange, pval_adj, pct_in, pct_out, rank
  domain_summary.csv    sample_id, domain, n_obs, fraction
  agreement.csv         sample_id, ari, nmi, n_obs, n_classes, n_domains (when obs["annotation"] exists)
  confusion.csv         sample_id, annotation, domain, n_obs (per sample and "all")
  annotation_match.csv  annotation, domain, n_annotation, n_domain, n_overlap, precision, recall, f1
  figures/              <sample_id>_domains.png, umap.png, markers_dotplot.png, confusion.png,
                        domain_composition.png
  summary.json, provenance.json

See processor/features.py (BANKSY features + PCA) and processor/domains.py (the rest).
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("st-domains")
for noisy in ("matplotlib", "PIL", "fontTools", "numba", "h5py"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "st-domains"
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "LAMBDA": "0.8",
    "K_GEOM": "15",
    "N_PCS": "20",
    "RESOLUTION": "0.2",
    "N_DOMAINS": "0",
    "INTEGRATION": "harmony",
    "N_MARKERS": "25",
    "IGNORE_ANNOTATIONS": "Unassigned",
    "MAX_OBS_FOR_UMAP": "50000",
    "SEED": "42",
    "THREADS": "0",
}
PARAMETERS = tuple(DEFAULTS)
INTEGRATIONS = ("harmony", "none")


class ParameterError(ValueError):
    pass


def read_parameters(env=os.environ) -> dict:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p: dict = {}
    p["INTEGRATION"] = raw["INTEGRATION"].strip().lower()
    if p["INTEGRATION"] not in INTEGRATIONS:
        raise ParameterError(f"INTEGRATION must be one of {', '.join(INTEGRATIONS)}, got {raw['INTEGRATION']!r}")
    p["IGNORE_ANNOTATIONS"] = [v.strip() for v in raw["IGNORE_ANNOTATIONS"].split(",") if v.strip()]
    try:
        for k in ("LAMBDA", "RESOLUTION"):
            p[k] = float(raw[k])
        for k in ("K_GEOM", "N_PCS", "N_DOMAINS", "N_MARKERS", "MAX_OBS_FOR_UMAP", "SEED", "THREADS"):
            f = float(raw[k])
            if f != int(f):
                raise ValueError(f"{k} must be an integer, got {raw[k]!r}")
            p[k] = int(f)
    except ValueError as exc:
        raise ParameterError(f"bad numeric parameter: {exc}") from exc
    if not 0.0 <= p["LAMBDA"] <= 1.0:
        raise ParameterError("LAMBDA must be in [0, 1]")
    if p["K_GEOM"] < 1:
        raise ParameterError("K_GEOM must be >= 1")
    if p["N_PCS"] < 2:
        raise ParameterError("N_PCS must be >= 2")
    if not p["RESOLUTION"] > 0:
        raise ParameterError("RESOLUTION must be > 0")
    if min(p["N_DOMAINS"], p["N_MARKERS"], p["MAX_OBS_FOR_UMAP"], p["THREADS"]) < 0:
        raise ParameterError("N_DOMAINS, N_MARKERS, MAX_OBS_FOR_UMAP and THREADS must be >= 0")
    return p


def _rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    return v


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for r in rows:
            w.writerow([_fmt(r.get(c, "")) for c in columns])


def _clean_json(obj):
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_clean_json(v) for v in obj]
    return obj


def write_json(path: Path, data) -> None:
    with open(path, "w") as fh:
        json.dump(_clean_json(data), fh, indent=2)
        fh.write("\n")


def tool_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for name in ("pybanksy", "scanpy", "anndata", "harmonypy", "leidenalg", "igraph", "umap-learn", "pynndescent",
                 "numpy", "scipy", "pandas", "scikit-learn", "h5py", "matplotlib"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            pass
    return out


def peak_rss_mb() -> float | None:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except (ImportError, OSError):
        return None


def cgroup_peak_mb() -> float | None:
    for p in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return round(int(open(p).read().strip()) / 1024 ** 2, 1)
        except (OSError, ValueError):
            continue
    return None


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


METHOD_NOTES = [
    "BANKSY (Singhal et al. 2024): per sample, each observation's log-normalized expression is augmented with "
    "the Gaussian-weighted mean expression of its K_GEOM nearest spatial neighbours and the magnitude of the "
    "first-order azimuthal Gabor filter (AGF) over its 2 x K_GEOM nearest neighbours; each block is z-scored "
    "per gene within the sample and weighted by sqrt(1 - LAMBDA), sqrt(2 LAMBDA / 3) and sqrt(LAMBDA / 3).",
    "Spatial weight graphs from pybanksy (generate_spatial_weights_fixed_nbrs, scaled_gaussian decay); the "
    "features are computed in row chunks in float32 and match pybanksy's create_nbr_matrix(center=True) + "
    "concatenate_all. pybanksy 1.3.5's generate_banksy_matrix passes variance_balance into the 'center' "
    "argument, so it computes the AGF uncentered; st-domains centres it as the R package and pybanksy's own "
    "code comment intend.",
    "PCA of the samples' BANKSY matrices concatenated, from the exact covariance matrix (no randomness).",
    "Harmony (harmonypy) on the PCs by sample_id when INTEGRATION=harmony and there is more than one sample.",
    "scanpy.pp.neighbors (15 neighbours) on the (integrated) PCs, scanpy.tl.leiden flavor=igraph, n_iterations=2.",
    "Domains named D01, D02, ... by size, largest first. Markers: scanpy rank_genes_groups Wilcoxon, each domain "
    "vs the rest, on every gene shared by all samples (at most 10,000 seeded observations per domain); pct_in / "
    "pct_out on every observation.",
]


def main() -> int:
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 2
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
    try:
        params = read_parameters()
    except ParameterError as exc:
        log.error("%s", exc)
        return 2
    # processor.entry set the thread variables from THREADS before numpy was imported
    n_threads = int(os.environ.get("OMP_NUM_THREADS", "0") or 0) or None

    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    from processor import domains as D
    from processor import features as FE
    from processor.inputs import discover

    timings: dict[str, float] = {}
    rss_after: dict[str, float | None] = {}      # peak RSS (MB) reached by the end of each stage

    def lap(name: str, since: float) -> float:
        now = time.monotonic()
        timings[name] = round(now - since, 2)
        rss_after[name] = peak_rss_mb()
        return now

    try:
        found = discover(input_dir)
    except (OSError, ValueError) as exc:
        log.error("cannot read the inputs: %s", exc)
        return 1
    for name, reason in found.skipped.items():
        log.warning("%s: skipped, %s", name, reason)
    if not found.samples:
        log.error("no usable <sample_id>.h5ad under %s", input_dir)
        return 1
    log.info("st-domains %s: %d sample(s), LAMBDA=%s K_GEOM=%d N_PCS=%d %s, %s thread(s)", app_version(),
             len(found.samples), params["LAMBDA"], params["K_GEOM"], params["N_PCS"],
             f"N_DOMAINS={params['N_DOMAINS']}" if params["N_DOMAINS"] else f"RESOLUTION={params['RESOLUTION']}",
             n_threads)

    status: dict[str, dict] = {}
    loaded: list[D.SampleData] = []
    t = time.monotonic()
    for s in found.samples:
        ts = time.monotonic()
        try:
            sd = D.load_sample(s.sample_id, s.path)
            if sd.x.shape[0] < 3:
                raise ValueError(f"only {sd.x.shape[0]} observations")
            loaded.append(sd)
            status[s.sample_id] = {"status": "loaded", "n_obs": int(sd.x.shape[0]), "n_genes": int(sd.x.shape[1]),
                                   "platform": sd.platform, "kind": sd.kind, "image": sd.image is not None,
                                   "normalized_here": sd.normalized_here}
            log.info("%s: %d observations x %d genes, %s %s%s, load %.1fs", s.sample_id, sd.x.shape[0], sd.x.shape[1],
                     sd.platform or "?", sd.kind or "?", " (normalized here: X held raw counts)" if sd.normalized_here else "",
                     time.monotonic() - ts)
        except Exception as exc:  # noqa: BLE001  a bad sample is skipped, not fatal
            log.error("%s: failed to load: %s", s.sample_id, exc)
            status[s.sample_id] = {"status": "failed", "error": str(exc)}
    t = lap("load", t)
    if not loaded:
        log.error("every sample failed to load")
        return 1

    genes = D.shared_genes(loaded)
    if not genes:
        log.error("no gene is shared by every sample")
        return 1
    for sd in loaded:
        D.restrict(sd, genes)
    use, how = D.choose_genes(loaded, genes)
    if not use:
        batch = np.concatenate([np.repeat(sd.sample_id, sd.x.shape[0]) for sd in loaded])
        use = D.compute_hvg_fallback(sp.vstack([sd.x for sd in loaded]).tocsr(), batch, genes, D.DEFAULT_N_HVG)
        how = f"{len(use)} highly variable genes computed here (no var['highly_variable'] upstream; seurat flavour by sample)"
    log.info("%d genes shared by all samples; BANKSY uses %s", len(genes), how)
    gidx = pd.Index(genes).get_indexer(use)

    graphs = []
    kept: list[D.SampleData] = []
    for sd in loaded:
        ts = time.monotonic()
        try:
            g = FE.build_graph(sd.xy, params["K_GEOM"])
            graphs.append((sp.csr_matrix(sd.x[:, gidx]), g))
            kept.append(sd)
            status[sd.sample_id].update({"k_geom": g.k_geom})
            log.info("%s: BANKSY graphs (k=%d / %d) in %.1fs", sd.sample_id, g.k_geom, 2 * g.k_geom, time.monotonic() - ts)
        except Exception as exc:  # noqa: BLE001
            log.error("%s: failed to build the spatial graph: %s", sd.sample_id, exc)
            status[sd.sample_id] = {"status": "failed", "error": str(exc)}
    if not kept:
        log.error("every sample failed")
        return 1
    t = lap("graphs", t)
    pca = FE.banksy_pca(graphs, params["LAMBDA"], params["N_PCS"])
    del graphs
    t = lap("banksy_pca", t)
    log.info("BANKSY matrix: %d observations x %d features; %d PCs explain %.1f%% of the variance",
             pca.embedding.shape[0], pca.n_features, pca.embedding.shape[1], 100 * float(pca.variance_ratio.sum()))

    import anndata as ad
    import scanpy as sc

    sample_ids = np.concatenate([np.repeat(sd.sample_id, sd.x.shape[0]) for sd in kept]).astype(object)
    obs_ids = np.concatenate([sd.obs_names for sd in kept]).astype(object)
    order_samples = [sd.sample_id for sd in kept]
    a = ad.AnnData(X=np.zeros((len(obs_ids), 0), dtype=np.float32),
                   obs=pd.DataFrame({"sample_id": pd.Categorical(sample_ids, categories=order_samples)},
                                    index=pd.Index(obs_ids).astype(str)))
    a.obsm["X_banksy_pca"] = pca.embedding
    rep = "X_banksy_pca"
    integrated = False
    if params["INTEGRATION"] == "harmony" and len(kept) > 1:
        a.obsm["X_banksy_harmony"] = D.harmony(pca.embedding, sample_ids, params["SEED"])
        rep, integrated = "X_banksy_harmony", True
        t = lap("harmony", t)
    sc.pp.neighbors(a, n_neighbors=min(D.N_NEIGHBORS, a.n_obs - 1), use_rep=rep, random_state=params["SEED"])
    t = lap("neighbors", t)
    raw, resolution, runs = D.cluster(a, params["RESOLUTION"], params["N_DOMAINS"], params["SEED"])
    domain, order = D.name_domains(raw)
    a.obs["domain"] = pd.Categorical(domain, categories=order)
    t = lap("leiden", t)
    log.info("%d domains at resolution %.4f (%d Leiden run(s))", len(order), resolution, len(runs))

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    write_csv(output_dir / "labels.csv",
              [{"sample_id": s, "obs_id": o, "domain": d} for s, o, d in zip(sample_ids, obs_ids, domain)],
              ["sample_id", "obs_id", "domain"])
    summ_rows = []
    for sd in kept:
        m = sample_ids == sd.sample_id
        n = int(m.sum())
        vc = pd.Series(domain[m]).value_counts()
        for d in order:
            summ_rows.append({"sample_id": sd.sample_id, "domain": d, "n_obs": int(vc.get(d, 0)),
                              "fraction": float(vc.get(d, 0)) / n})
        status[sd.sample_id].update({"status": "ok", "n_domains": int((vc > 0).sum())})
        log.info("%s: %d observations in %d domains (largest %s %.0f%%)", sd.sample_id, n, int((vc > 0).sum()),
                 vc.index[0], 100.0 * vc.iloc[0] / n)
    # sample coverage: in how many samples each domain holds at least MIN_OBS_PRESENT observations
    coverage = {d: sum(1 for r in summ_rows if r["domain"] == d and r["n_obs"] >= D.MIN_OBS_PRESENT) for d in order}
    for r in summ_rows:
        r["domain_n_samples"] = coverage[r["domain"]]
    n_shared = sum(1 for v in coverage.values() if v >= 2)
    if len(kept) > 1:
        log.info("%d of %d domains have >= %d observations in at least 2 samples; in 1 sample only: %s", n_shared,
                 len(order), D.MIN_OBS_PRESENT, ", ".join(d for d, v in coverage.items() if v <= 1) or "none")
    write_csv(output_dir / "domain_summary.csv", summ_rows,
              ["sample_id", "domain", "n_obs", "fraction", "domain_n_samples"])
    summary_df = pd.DataFrame(summ_rows)

    x_all = sp.vstack([sd.x for sd in kept]).tocsr()
    mk, mk_adata = D.markers(x_all, genes, domain, order, params["N_MARKERS"], params["SEED"])
    mk.to_csv(output_dir / "domain_markers.csv", index=False, float_format="%.6g")
    t = lap("markers", t)

    agree = None
    has_ann = any(sd.annotation is not None for sd in kept)
    if has_ann:
        ann = np.concatenate([sd.annotation if sd.annotation is not None else np.repeat("", sd.x.shape[0])
                              for sd in kept]).astype(object)
        agree = D.agreement(sample_ids, ann, domain, order_samples, set(params["IGNORE_ANNOTATIONS"]))
        write_csv(output_dir / "agreement.csv", agree["agreement"],
                  ["sample_id", "ari", "nmi", "n_obs", "n_classes", "n_domains"])
        write_csv(output_dir / "confusion.csv", agree["confusion"], ["sample_id", "annotation", "domain", "n_obs"])
        write_csv(output_dir / "annotation_match.csv", agree["matches"],
                  ["annotation", "domain", "n_annotation", "n_domain", "n_overlap", "precision", "recall", "f1"])
        pooled = agree["agreement"][-1]
        log.info("agreement with obs['annotation'] (%d observations, %d ignored): ARI %.3f, NMI %.3f",
                 agree["n_used"], agree["n_ignored"], pooled["ari"], pooled["nmi"])
    else:
        log.info("no obs['annotation']: agreement not computed")
    t = lap("agreement", t)

    # figures: a failure here is logged, not fatal
    from processor import figures as FG

    colours = FG.domain_colours(order)
    figs: list[str] = []

    def fig(name: str, fn, *args):
        try:
            fn(fig_dir / name, *args)
            if (fig_dir / name).exists():
                figs.append(f"figures/{name}")
        except Exception as exc:  # noqa: BLE001
            log.warning("figure %s failed: %s", name, exc)

    for sd in kept:
        m = sample_ids == sd.sample_id
        fig(f"{sd.sample_id}_domains.png", FG.domain_map, sd.sample_id, sd.xy, domain[m], order, colours, sd.image,
            sd.lowres_scalef, sd.spot_diameter)
    if params["MAX_OBS_FOR_UMAP"] > 0:
        try:
            if a.n_obs <= params["MAX_OBS_FOR_UMAP"]:
                u = a
            else:
                rng = np.random.default_rng(params["SEED"])
                pick = np.sort(rng.choice(a.n_obs, params["MAX_OBS_FOR_UMAP"], replace=False))
                u = ad.AnnData(X=np.zeros((len(pick), 0), dtype=np.float32), obs=a.obs.iloc[pick].copy())
                u.obsm[rep] = a.obsm[rep][pick]
                sc.pp.neighbors(u, n_neighbors=D.N_NEIGHBORS, use_rep=rep, random_state=params["SEED"])
            sc.tl.umap(u, random_state=params["SEED"])
            panels = [("domain", u.obs["domain"].astype(str).to_numpy(), order, colours)]
            if len(kept) > 1:
                scol = dict(zip(order_samples, FG.domain_colours(order_samples).values()))
                panels.append(("sample", u.obs["sample_id"].astype(str).to_numpy(), order_samples, scol))
            if has_ann:
                pos = a.obs_names.get_indexer(u.obs_names)
                ignored = set(params["IGNORE_ANNOTATIONS"])
                av = np.array([v if v and v not in ignored else "(ignored / none)" for v in ann[pos]], dtype=object)
                classes = sorted(set(av) - {"(ignored / none)"})
                acol = FG.domain_colours(classes)
                if "(ignored / none)" in set(av):
                    classes = ["(ignored / none)"] + classes          # drawn first, underneath, in grey
                    acol["(ignored / none)"] = "#d0d4d9"
                panels.append(("annotation", av, classes, acol))
            fig("umap.png", FG.umap, u.obsm["X_umap"], panels, a.n_obs)
        except Exception as exc:  # noqa: BLE001
            log.warning("UMAP failed: %s", exc)
        t = lap("umap", t)
    if mk_adata is not None and len(mk):
        fig("markers_dotplot.png", FG.markers_dotplot, mk_adata, mk, order)
    if agree is not None and agree["confusion"]:
        conf = pd.DataFrame(agree["confusion"])
        fig("confusion.png", FG.confusion, conf[conf["sample_id"] == "all"], order)
    fig("domain_composition.png", FG.composition, summary_df, order_samples, order, colours)
    t = lap("figures", t)

    pooled = agree["agreement"][-1] if agree else None
    summary = {
        "app": APP_NAME, "version": app_version(),
        "n_samples": len(kept), "n_samples_failed": sum(1 for v in status.values() if v.get("status") == "failed"),
        "n_obs": int(a.n_obs), "n_domains": len(order), "domains": order,
        "resolution": resolution, "resolution_runs": runs, "n_domains_requested": params["N_DOMAINS"],
        "lambda": params["LAMBDA"], "k_geom": params["K_GEOM"], "n_pcs": int(pca.embedding.shape[1]),
        "pca_variance_explained": float(pca.variance_ratio.sum()),
        "n_genes_shared": len(genes), "n_genes_banksy": len(use), "gene_selection": how,
        "integration": "harmony" if integrated else "none",
        "domain_sizes": {d: int((domain == d).sum()) for d in order},
        "domain_sample_coverage": coverage, "min_obs_present": D.MIN_OBS_PRESENT,
        "n_domains_in_2plus_samples": n_shared,
        "per_sample": {sd.sample_id: {"n_obs": int((sample_ids == sd.sample_id).sum()),
                                      "n_domains": status[sd.sample_id].get("n_domains")} for sd in kept},
        "annotation": None if agree is None else {
            "ignored": params["IGNORE_ANNOTATIONS"], "n_obs_used": agree["n_used"], "n_obs_ignored": agree["n_ignored"],
            "ari": pooled["ari"], "nmi": pooled["nmi"],
            "per_sample": {r["sample_id"]: {"ari": r["ari"], "nmi": r["nmi"], "n_obs": r["n_obs"]}
                           for r in agree["agreement"][:-1]},
            "best_match": agree["matches"]},
        "top_markers": {d: mk.loc[mk["domain"] == d].sort_values("rank")["gene"].head(5).tolist() for d in order},
        "figures": figs,
    }
    write_json(output_dir / "summary.json", summary)
    timings["total"] = round(time.monotonic() - t0, 2)
    prov = {
        "app": APP_NAME, "version": app_version(), "started": started.isoformat(),
        "finished": datetime.now(timezone.utc).isoformat(),
        "tool_versions": tool_versions(), "parameters": params, "threads": n_threads,
        "method": METHOD_NOTES,
        "inputs": {"spatial": [_rel(p, input_dir) for p in found.files_seen],
                   "samples_csv": _rel(found.samples_csv, input_dir), "skipped": found.skipped},
        "genes": {"shared": len(genes), "banksy": len(use), "selection": how},
        "timings_seconds": timings, "samples": status,
        "peak_rss_mb": peak_rss_mb(), "peak_rss_mb_by_stage": rss_after, "cgroup_peak_mb": cgroup_peak_mb(),
        "outputs": sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()) + ["provenance.json"],
    }
    write_json(output_dir / "provenance.json", prov)
    log.info("done in %.1fs; %d domains over %d observations; peak RSS %s MB", timings["total"], len(order), a.n_obs,
             prov["peak_rss_mb"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
