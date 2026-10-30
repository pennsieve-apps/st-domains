"""
A synthetic st-qc-shaped fixture that follows the workflow contract
(docs/conventions.md of workflow-spatial-transcriptomics).

  python -m tests.make_fixture data/fixture [--layout linked|flat|both] [--spots 30] [--genes 300]

Two Visium-like samples (hex grid, `uns["spatial"]` H&E stand-ins, full-resolution pixel
coordinates) with four planted tissue domains per sample, each with its own gene program,
so a domain method should recover them (ARI well above 0.8). `obs["annotation"]` holds the
planted domain for most spots, `Unassigned` for a random 15 %, and in the second sample a
small extra class (`Extra`) inside domain 3, which exercises the confusion table.

Layouts: `linked` = INPUT_DIR/spatial/<sample_id>.h5ad + INPUT_DIR/metadata/{samples.csv,
prep.json} (placement: subfolder); `flat` = everything side by side at the root.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

SAMPLES = ["S1", "S2"]
DOMAINS = ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]
SPOT_SPACING = 300.0            # full-resolution pixels between neighbouring spots (~100 um at ~3 px/um)
SPOT_DIAMETER = 165.0           # full-resolution pixels (~55 um)


def hex_grid(n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Visium-style array indices (odd rows hold odd columns) and full-res pixel centres."""
    rows, cols = [], []
    for r in range(n_rows):
        for c in range(r % 2, 2 * n_cols, 2):
            rows.append(r)
            cols.append(c)
    rows, cols = np.array(rows), np.array(cols)
    x = cols * (SPOT_SPACING / 2) + 1000.0
    y = rows * (SPOT_SPACING * np.sqrt(3) / 2) + 1000.0
    return rows, cols, np.column_stack([x, y])


def planted_domains(xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Four contiguous regions: the nearest of four centres, with a smooth wobble on the distances."""
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    centres = lo + rng.uniform(0.15, 0.85, size=(len(DOMAINS), 2)) * (hi - lo)
    d = np.linalg.norm(xy[:, None, :] - centres[None, :, :], axis=2)
    wobble = 0.15 * (hi - lo).mean() * np.sin(xy[:, :1] / 900.0 + rng.uniform(0, 6)) * np.cos(xy[:, 1:] / 1100.0)
    return np.argmin(d + wobble, axis=1)


def expression(domain: np.ndarray, n_genes: int, rng: np.random.Generator) -> np.ndarray:
    """Raw counts: gene-specific base rates, a 30-gene program per domain (x4), spot library sizes."""
    n = len(domain)
    base = rng.lognormal(mean=-1.0, sigma=1.0, size=n_genes)
    programme = np.ones((len(DOMAINS), n_genes))
    per = max(5, n_genes // 10)
    for d in range(len(DOMAINS)):
        programme[d, d * per:(d + 1) * per] = 4.0
    libsize = rng.lognormal(mean=0.0, sigma=0.35, size=n)
    rate = base[None, :] * programme[domain] * libsize[:, None] * 2.0
    return rng.poisson(rate).astype(np.float32)


def make_sample(sample_id: str, n_rows: int, n_cols: int, n_genes: int, rng: np.random.Generator, extra_class: bool):
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp

    rows, cols, xy = hex_grid(n_rows, n_cols)
    domain = planted_domains(xy, rng)
    counts = expression(domain, n_genes, rng)
    genes = [f"GENE{i + 1:04d}" for i in range(n_genes)]
    genes[-3:] = ["MT-CO1", "MT-ND1", "MT-CYB"]
    barcodes = [f"{''.join(rng.choice(list('ACGT'), 16))}-1" for _ in range(len(rows))]
    obs = pd.DataFrame(index=[f"{sample_id}:{b}" for b in barcodes])
    obs["sample_id"] = pd.Categorical([sample_id] * len(rows))
    obs["x"], obs["y"] = xy[:, 0], xy[:, 1]
    obs["in_tissue"] = 1
    obs["array_row"], obs["array_col"] = rows, cols
    annotation = np.array(DOMAINS, dtype=object)[domain]
    annotation[rng.random(len(rows)) < 0.15] = "Unassigned"
    if extra_class:
        c3 = xy[domain == 2].mean(axis=0)
        near = (np.linalg.norm(xy - c3, axis=1) < 700) & (domain == 2)
        annotation[near] = "Extra"
    obs["annotation"] = annotation
    obs["condition"] = "ctrl" if sample_id == "S1" else "treated"
    var = pd.DataFrame(index=genes)
    var["gene_ids"] = [f"ENSG{i:011d}" for i in range(n_genes)]
    var["feature_types"] = "Gene Expression"
    var["mt"] = var.index.str.startswith("MT-")
    adata = ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
    adata.obs["n_genes_by_counts"] = adata.obs["n_genes_by_counts"].astype(np.int32)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(200, n_genes - 10), flavor="seurat")
    # st-qc uses seurat_v3 on the counts, which also writes a rank; mimic that column.
    rank = adata.var["dispersions_norm"].rank(ascending=False, method="first")
    adata.var["highly_variable_rank"] = np.where(adata.var["highly_variable"], rank - 1, np.nan)
    sc.pp.pca(adata, n_comps=20, mask_var="highly_variable", random_state=0)
    adata.obsm["spatial"] = xy.astype(np.float64)
    # A lowres / hires "H&E" stand-in: a pale pink background with slightly darker tissue.
    full_w, full_h = float(xy[:, 0].max() + 1000), float(xy[:, 1].max() + 1000)
    lowres_scalef = 600.0 / max(full_w, full_h)
    hires_scalef = 1000.0 / max(full_w, full_h)

    def image(scalef: float) -> np.ndarray:
        w, h = int(round(full_w * scalef)), int(round(full_h * scalef))
        img = np.full((h, w, 3), [238, 214, 220], dtype=np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        inside = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) < 0.2
        img[inside] = [224, 180, 196]
        return img

    adata.uns["spatial"] = {sample_id: {
        "images": {"hires": image(hires_scalef), "lowres": image(lowres_scalef)},
        "scalefactors": {"tissue_hires_scalef": hires_scalef, "tissue_lowres_scalef": lowres_scalef,
                         "spot_diameter_fullres": SPOT_DIAMETER, "fiducial_diameter_fullres": SPOT_DIAMETER * 1.6},
    }}
    adata.uns["pennsieve"] = {"platform": "visium", "kind": "spot", "coordinate_unit": "pixel",
                              "spot_diameter": SPOT_DIAMETER, "grid": "hex", "source": f"{sample_id}/",
                              "prep_version": "0.1.0", "normalized": True, "target_sum": 10000.0}
    return adata


def make_fixture(root: Path, layout: str = "both", spots: int = 30, genes: int = 300, seed: int = 11) -> dict[str, Path]:
    root = Path(root)
    rng = np.random.default_rng(seed)
    adatas = [make_sample(sid, spots, spots, genes, rng, extra_class=(i == 1)) for i, sid in enumerate(SAMPLES)]
    rows = []
    prep = {"app": "st-prep", "samples": {}}
    for a in adatas:
        sid = str(a.obs["sample_id"].iloc[0])
        rows.append({"sample_id": sid, "path": f"{sid}/", "platform": "visium", "kind": "spot",
                     "n_obs": a.n_obs, "n_genes": a.n_vars, "n_obs_in_tissue": a.n_obs,
                     "condition": str(a.obs["condition"].iloc[0])})
        prep["samples"][sid] = {"platform": "visium", "files": ["filtered_feature_bc_matrix.h5", "spatial/"],
                                "coordinate_unit": "pixel", "image": True, "warnings": []}
    out: dict[str, Path] = {}
    layouts = ["linked", "flat"] if layout == "both" else [layout]
    for lay in layouts:
        base = root / lay
        if base.exists():
            shutil.rmtree(base)
        spatial = base / "spatial" if lay == "linked" else base
        meta = base / "metadata" if lay == "linked" else base
        spatial.mkdir(parents=True, exist_ok=True)
        meta.mkdir(parents=True, exist_ok=True)
        for a in adatas:
            a.write_h5ad(spatial / f"{a.obs['sample_id'].iloc[0]}.h5ad")
        import pandas as pd

        pd.DataFrame(rows).to_csv(meta / "samples.csv", index=False)
        with open(meta / "prep.json", "w") as fh:
            json.dump(prep, fh, indent=2)
        out[lay] = base
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--layout", choices=["linked", "flat", "both"], default="both")
    ap.add_argument("--spots", type=int, default=30, help="hex grid rows and columns per sample (30 -> 900 spots)")
    ap.add_argument("--genes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    paths = make_fixture(args.root, args.layout, args.spots, args.genes, args.seed)
    for lay, p in paths.items():
        print(f"{lay}: {p}")


if __name__ == "__main__":
    main()
