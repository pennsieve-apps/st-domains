# st-domains

Spatial domains with **BANKSY** (Singhal et al. 2024), the clustering step of the Spatial Domain
Mapping & Condition Comparison workflow (`st-prep → st-qc → st-domains → {st-svg, st-compare} →
st-report`). It works on any platform the workflow reads: Visium spots, Visium HD bins and
Xenium / MERSCOPE / CosMx cells.

**For a biologist.** A spatial domain is a region of the tissue whose spots (or cells) share an
expression profile *and* a neighbourhood: tumour nests, stroma, a lymphoid aggregate, a cortical
layer. Ordinary clustering looks at each spot on its own, so neighbouring spots of the same region
can scatter over several clusters and a region's edges are noisy. BANKSY adds to each spot the
average expression of its spatial neighbours and a measure of how expression changes across the
neighbourhood (the azimuthal Gabor filter, a local gradient), then clusters on both. With a high
neighbourhood weight (`LAMBDA = 0.8`, the default) the clusters are contiguous tissue domains;
with a low one (`LAMBDA = 0.2`) they are closer to cell types or states that are spatially
coherent. All samples are clustered together, so `D03` means the same kind of region in every
section, and the app reports for each domain its marker genes, how much of each sample it covers
and, when the samples carry a pathologist's annotation, how well the domains agree with it.

**For a computational analyst.** Per sample: log-normalized expression of the genes used (st-qc's
highly variable genes pooled over samples; every gene for panels of ≤ 1,000 genes); BANKSY's
spatial kNN graphs (pybanksy, scaled-Gaussian weights, `K_GEOM` neighbours for the mean, 2 ×
`K_GEOM` with phases `exp(iθ)` for the first-order AGF); the three blocks (own, neighbour mean,
|AGF|) z-scored per gene within the sample and weighted by √(1 − λ), √(2λ/3), √(λ/3). The samples'
matrices are stacked and reduced by PCA (`N_PCS`), integrated with Harmony by `sample_id`
(`INTEGRATION=harmony`, more than one sample), and clustered with scanpy's kNN graph (15
neighbours) and Leiden (`flavor="igraph"`, `n_iterations=2`) at `RESOLUTION`, or with a bisection
of the resolution (0.05–3, ≤ 12 runs) for `N_DOMAINS`. Domains are named `D01`, `D02`, … by size.
Markers: Wilcoxon rank-sum, each domain against the rest, on every gene shared by all samples.

### How the BANKSY matrix is computed

pybanksy (1.3.5) builds the spatial weight graphs. Its `generate_banksy_matrix` would then
materialize the whole matrix densely in float64 (own + neighbour + AGF, then z-scored copies):
about 24 GB for a 500 k-bin Visium HD sample with 2,000 genes, before PCA. st-domains computes
the same numbers in row chunks in float32 (`processor/features.py`) and gets the PCA from the exact
covariance matrix of the stacked, scaled features (3 × genes columns, at most a few thousand), in
three streaming passes (block means / SDs, covariance, projection). The PCA is exact and has no
random component. `tests/test_processor.py` checks the features against pybanksy's own functions
(max |difference| < 0.01, correlation > 0.99999 per block, λ = 0.2 and 0.8) and the PCA against
scikit-learn's.

One difference from calling pybanksy's top-level function: in pybanksy 1.3.5,
`generate_banksy_matrix` calls `create_nbr_matrix(adata, banksy_dict, decay, max_m,
variance_balance)` positionally, so `variance_balance` (False) lands in the `center` argument and
the AGF is computed **without** subtracting the neighbourhood mean. The paper, the R package and
pybanksy's own comment ("Update Sep 2024 to match Banksy R") centre it; st-domains centres it
(it matches `create_nbr_matrix(..., center=True)`).

## Ports

| Port | Kind | Files | Notes |
|---|---|---|---|
| `spatial` | input (subfolder) | `<sample_id>.h5ad` | st-qc's output: log-normalized `X` (CSR), `var["highly_variable"]` (+ `highly_variable_rank`), `obsm["spatial"]`, `uns["pennsieve"]`, `uns["spatial"]` lowres image (Visium), optional `obs["annotation"]`. Looked for in `INPUT_DIR/spatial/`, else anywhere under `INPUT_DIR` (symlinks followed). Only `X`, `obs`, `var`, `obsm["spatial"]` and the lowres image are read (not `layers`). st-prep's raw-count files also work: the sample is then normalized here (median total counts, `log1p`) and HVGs are computed here (scanpy `seurat` flavour, 2,000, by sample); the log and `provenance.json` say so. |
| `metadata` | input (subfolder, optional) | `samples.csv` | st-prep's sample table: which `.h5ad` files are samples, in which order. Without it every `.h5ad` is a sample (id = file stem). |
| `labels` | output | see below | Everything at the root of `OUTPUT_DIR`. Input of st-svg, st-compare and st-report (`LABEL_COLUMN=domain`). |

## Parameters

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `LAMBDA` | number | `0.8` | Weight of the neighbourhood features (0–1). 0.8: tissue domains (the BANKSY authors' recommendation); 0.2: spatially informed cell types / states; 0: ordinary expression clustering. |
| `K_GEOM` | integer | `15` | Spatial nearest neighbours for the neighbourhood mean (the AGF uses 2 × `K_GEOM`). 15 ≈ two rings of Visium spots; for cells or 8 µm bins it is a smaller physical radius, which is usually what you want. Capped at (n − 1) / 2 for tiny samples. |
| `N_PCS` | integer | `20` | Principal components of the BANKSY matrix. |
| `RESOLUTION` | number | `0.2` | Leiden resolution; higher = more domains. Ignored when `N_DOMAINS > 0`. On 8 TLS Visium sections 0.2 gave 14 domains, 9 shared across sections; 0.5 gave 29, most in a single section (see Results). |
| `N_DOMAINS` | integer | `0` | When > 0: bisect the resolution in 0.05–3 (≤ 12 Leiden runs) for exactly this many domains; the closest count found is used otherwise (lower resolution on ties), with a warning. Every run is listed in `summary.json`. |
| `INTEGRATION` | string | `harmony` | `harmony`: Harmony (harmonypy) on the PCs by `sample_id` before the kNN graph, when there is more than one sample. `none`: no correction. |
| `N_MARKERS` | integer | `25` | Marker genes per domain in `domain_markers.csv`. |
| `IGNORE_ANNOTATIONS` | string | `Unassigned` | Comma-separated `obs["annotation"]` values left out of the agreement statistics (empty values always are). |
| `MAX_OBS_FOR_UMAP` | integer | `50000` | The UMAP figure uses a seeded subsample of at most this many observations (with its own kNN graph when subsampled); `0` = no UMAP. |
| `SEED` | integer | `42` | Seeds Harmony, the kNN graph, Leiden, UMAP and the marker / UMAP subsamples. Same input + same seed = same domains (tested). |
| `THREADS` | integer | `0` | BLAS / OpenMP / numba threads; 0 = from the container's CPU quota (2 if there is none). |

## Outputs

| File | Columns / content |
|---|---|
| `labels.csv` | `sample_id, obs_id, domain`: every observation of every sample, in input order (`obs_id` = the `obs_names` value). |
| `domain_markers.csv` | `domain, gene, logfoldchange, pval_adj, pct_in, pct_out, rank`: the top `N_MARKERS` genes per domain by Wilcoxon score (scanpy `rank_genes_groups`, domain vs rest, Benjamini–Hochberg). `pct_in` / `pct_out`: fraction of observations in / outside the domain where the gene is detected, over every observation. The test uses at most 10,000 seeded observations per domain. |
| `domain_summary.csv` | `sample_id, domain, n_obs, fraction, domain_n_samples`: one row per sample × domain, zeros included; `domain_n_samples` = in how many samples the domain has ≥ 20 observations (sample coverage; repeated on each of the domain's rows). |
| `agreement.csv` | `sample_id, ari, nmi, n_obs, n_classes, n_domains`: adjusted Rand index and normalized mutual information (arithmetic) between `domain` and `obs["annotation"]` per sample and pooled (`all`), over annotated observations not in `IGNORE_ANNOTATIONS`. Only when some sample has `obs["annotation"]`. |
| `confusion.csv` | `sample_id, annotation, domain, n_obs`: the contingency table, per sample and `all`, zeros included. |
| `annotation_match.csv` | `annotation, domain, n_annotation, n_domain, n_overlap, precision, recall, f1`: for each annotation class (pooled), the domain with the best F1; precision = overlap / domain size, recall = overlap / class size. |
| `figures/<sample_id>_domains.png` | The sample's domains on the lowres H&E (Visium), spots drawn at their real diameter; plain scatter for platforms without an image. y down (image convention). |
| `figures/umap.png` | UMAP of the (integrated) BANKSY embedding, coloured by domain, sample and annotation (ignored / missing annotations in grey). |
| `figures/markers_dotplot.png` | Top markers per domain (3, or 2 when there are more than 12 domains), mean expression scaled per gene. |
| `figures/confusion.png` | Annotation × domain, coloured by the fraction of each class, counts written. |
| `figures/domain_composition.png` | Stacked bars of each sample's domain fractions. |
| `summary.json` | Headline numbers: samples, observations, domains and sizes, resolution and every Leiden run, `domain_sample_coverage` and `n_domains_in_2plus_samples`, λ, K_GEOM, genes used and how, integration, PCA variance explained, pooled and per-sample ARI / NMI, best match per class, top 5 markers per domain, figures. |
| `provenance.json` | App and tool versions, parameters, method notes, inputs seen, timings and peak RSS per stage, per-sample status (observations, genes, effective K_GEOM, normalized here or not). |

A sample that cannot be read (or has fewer than 3 observations) is logged and skipped; the run
fails (exit 1) only when no sample is usable, and with exit 2 on a bad parameter.

## Results on the demo data

**TLS Visium** (Dawo, Nonchev & Silina 2025; 8 FFPE sections, 3 kidney + 5 lung cancer), after
st-prep and st-qc: 27,497 spots, 15,392 genes shared by all eight sections (st-qc filters genes per
sample, so each section keeps a different set; st-domains uses the intersection), 1,803 highly
variable genes pooled from st-qc's per-sample flags → 5,409 BANKSY features. Pathologist annotations
on 10,971 spots (Normal 5,118, Tumor 4,442, TLS 885, Immune 493, Lymph_node 33); 16,526 `Unassigned`
spots are left out of the agreement.

With the defaults (`LAMBDA=0.8`, `RESOLUTION=0.2`, Harmony) the image found **14 domains, 9 of them
present (≥ 20 spots) in at least 2 sections** (6 in all eight, 2 in seven); the other 5 are small
(468 spots or fewer) and reach 20 spots in at most one section. Pooled **ARI 0.18, NMI 0.29**; per section ARI 0.04 (KC2) to
0.65 (LC2), median ≈ 0.29. Best matches: TLS → D03 (recall 0.76, precision 0.41; markers LTB, CD37,
CORO1A, IGHG2: a lymphoid domain that also holds lymphocyte-rich spots the pathologist did not call
TLS), Tumor → D04 (precision 0.90, recall 0.41), Normal → D01 (precision 0.91, recall 0.32; SFTPB,
SFTPC: lung alveolar parenchyma), Immune → no domain (best F1 0.15). D07 is smooth-muscle / vessel
wall (ACTA2, MYL9, TAGLN), D06 a squamous / tumour-associated programme (S100A2, IFI6).

The agreement is modest, and it should be: the annotation has 4–5 coarse classes, while domains split
tumour and normal tissue by organ (kidney vs lung) and by composition, and 60 % of the spots are
unannotated. A sweep (native arm64 runs; the amd64 image gives slightly different counts, e.g. 14
instead of 13 domains at resolution 0.2, from floating-point differences in the kNN search):

| LAMBDA | Domains (how) | Domains in ≥ 2 sections | ARI | NMI | TLS → best domain (precision / recall) |
|---|---|---|---|---|---|
| 0.8 | 29 (resolution 0.5, the original default) | 17 | 0.11 | 0.28 | 0.48 / 0.61 |
| 0.8 | 13 (resolution 0.2) | 9 | 0.18 | 0.30 | 0.43 / 0.73 |
| 0.8 | 12 (`N_DOMAINS`, res. 0.159) | 7 | 0.17 | 0.28 | 0.37 / 0.73 |
| 0.8 | 10 (`N_DOMAINS`, res. 0.142) | 6 | 0.24 | 0.29 | 0.40 / 0.77 |
| 0.8 | 8 (`N_DOMAINS`, res. 0.079) | 5 | 0.23 | 0.29 | 0.18 / 0.88 |
| 0.8 | 5 (`N_DOMAINS`, res. 0.062) | 3 | 0.41 | 0.37 | 0.13 / 0.98 |
| 0.8 | 8 (`N_DOMAINS`), `INTEGRATION=none` | 1 | 0.05 | 0.09 | 0.16 / 0.38 |
| 0.2 | 7 (resolution 0.2) | 6 | 0.24 | 0.29 | 0.29 / 0.71 |
| 0.2 | 8 (`N_DOMAINS`) | 7 | 0.25 | 0.31 | 0.53 / 0.69 |
| 0.2 | 12 (`N_DOMAINS`) | 8 | 0.22 | 0.31 | 0.53 / 0.72 |
| 0.0 | 8 (`N_DOMAINS`, no spatial term) | 8 | 0.20 | 0.25 | 0.49 / 0.64 |

What this says: (1) **Harmony matters**: without it every domain but one is a single section.
(2) At `LAMBDA=0.8` the neighbourhood features carry much of each section's own signature, so
higher resolutions split domains by section; the default was lowered from 0.5 (29 domains, 12 in a
single section) to 0.2 for that reason. (3) The highest ARI (0.41, 5 domains) comes from a coarse
tumour / non-tumour split in which TLS disappear into a large domain (precision 0.13): ARI rewards
getting the big classes right, not the small structures. (4) `LAMBDA=0.2` keeps TLS as their own,
more precise domain and shares domains across sections better; for TLS-like small structures it is
the better choice, at the cost of smoother tissue domains. (5) BANKSY's spatial term does not raise
agreement with this annotation much over plain expression clustering (`LAMBDA=0`, ARI 0.20); its
domains are spatially contiguous (see `figures/<sample>_domains.png`), which the annotation-based
scores do not measure. Each run's `summary.json` lists `domain_sample_coverage`.

**Synthetic fixture** (2 × 900 hex spots, four planted zones): `N_DOMAINS=4` recovers them with
ARI 0.80, NMI 0.79 (best-match F1 0.89–0.94 per zone); the misses are boundary spots.

**Lymph node** (10x, 3,928 spots after QC, one sample, no annotation): at resolution 0.5, 16
domains, 30 s natively.

## Runtime

`standard` compute, **4 vCPU, 12 GB**, timeout 4 h. BANKSY is joint across samples, so every
sample's sparse `X` (shared genes) is held at once; the BANKSY matrix itself is never materialized.
Measured peak memory and wall time (`provenance.json`: `peak_rss_mb_by_stage`, `timings_seconds`):

| Data | Where | Peak memory | Wall time |
|---|---|---|---|
| TLS Visium, 8 sections, 27,497 spots, 1,803 genes | linux/amd64 image, **emulated on an M-series Mac**, `--cpus 2 --memory 4g` | 2.9 GB (cgroup peak) | 3.1 min (kNN graph 101 s, mostly numba compilation under emulation; BANKSY PCA 33 s; markers 26 s) |
| Synthetic Visium HD-like, 200,000 bins × 18,000 genes, 2,000 HVGs, `N_DOMAINS=9` | same image, emulated, `--cpus 2 --memory 4g` | 3.8 GB | 6.3 min (BANKSY PCA 169 s, kNN 123 s) |
| Synthetic, 100,000 bins | native arm64, 2 threads | 2.2 GB | 2.0 min |
| Synthetic, **500,000 bins** × 18,000 genes, `N_DOMAINS=9` (12 Leiden runs) | native arm64, 2 threads | **5.8 GB** (Leiden on the 500 k graph) | 6.4 min |

Memory grows roughly linearly with observations (≈ 1 GB per 100 k bins beyond a ~1.5 GB base at
2,000 genes), plus a genes² term (~0.3 GB covariance at 2,000 genes). Extrapolated to a 500 k-bin
Visium HD sample in the image: ~6–7 GB and 20–30 min under emulation on 2 CPUs (less on real amd64
with 4 vCPUs), so 12 GB leaves about 2× headroom. The image is 1.04 GB.

## Running it locally

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python      # 14 tests, ~20 s, no Docker
make run PYTHON=.venv/bin/python       # Docker over data/fixture/linked/ -> data/output/
```

`tests/make_fixture.py` writes two synthetic Visium-like samples (hex grids, four planted zones,
an `obs["annotation"]`); see `sample-input/README.md` for real data.

## Limitations

- **Domains are clusters, not anatomy.** BANKSY makes clusters spatially smooth; it does not
  know what a TLS or a tumour boundary is. A domain can mix tissue types that share a neighbourhood
  profile, and one region can split over several domains. The number of domains is set by
  `RESOLUTION` / `N_DOMAINS`, not discovered; look at the maps and markers before naming them.
- **Boundaries get their own domains.** With `LAMBDA = 0.8`, spots on the edge between two regions
  have a mixed neighbourhood, and at higher resolutions they form thin "interface" domains. That is
  often real (invasive margins), but it lowers ARI against annotations that draw a hard line.
- **Small structures are smoothed away.** A structure a few spots across (a single TLS, a small
  vessel) is averaged with its surroundings at `K_GEOM = 15`; lower `LAMBDA` or `K_GEOM` to keep it.
- **Harmony removes sample differences, including real ones.** If a region exists in only some
  sections, or a condition changes a region's expression, Harmony may pull it towards other
  domains. Compare with `INTEGRATION=none`; condition effects belong in st-compare.
- **Visium spots are mixtures** of several cells; domains describe spot-level composition. For
  cell types per spot use cell2location.
- **Neighbourhoods are kNN, not radius-capped.** BANKSY uses its own kNN graph (as the method
  defines), not the workflow's squidpy graph: an isolated spot or a tissue edge borrows neighbours
  from further away. `K_GEOM` neighbours of a Visium spot span ~2 rings (~200 µm); of Xenium cells
  ~30–50 µm.
- **Scale.** Tested up to 500,000 observations in one run (synthetic); 2 µm Visium HD bins
  (millions of bins) are out of scope. Memory grows with observations (sparse `X` of the shared
  genes, the graphs) and with genes² (the covariance, ~300 MB at 2,000 genes).
- **Markers are one-vs-rest** over all samples pooled; they describe the domain against the rest
  of the tissue, not against a specific neighbour, and spots are not independent (p-values are
  optimistic).

## Licences

Our code is Apache-2.0 (`LICENSE`). The image runs it in-process with pybanksy (GPL-3.0), harmonypy
(GPL-3.0-or-later), leidenalg (GPL-3.0-or-later) and igraph (GPL-2.0-or-later), so **the image as a
whole is distributed under the GPL-3.0**. Details and every bundled component in `THIRD_PARTY.md`.
The TLS demo data are CC BY 4.0.

## Citations

Verified on Crossref (DataCite for the Zenodo record) and PubMed (title, full author list, year).
Full author lists in `CITATION.cff`.

- **BANKSY**: Singhal V, Chou N, Lee J, Yue Y, Liu J, Chock WK, Lin L, Chang Y-C, Teo EML, Aow J,
  Lee HK, Chen KH, Prabhakar S. BANKSY unifies cell typing and tissue domain segmentation for
  scalable spatial omics data analysis. *Nat Genet* 2024;56(3):431–441.
  doi:10.1038/s41588-024-01664-3. PMID 38413725.
- **pybanksy** 1.3.5 (Chou N, Yue Y, Singhal V): https://github.com/prabhakarlab/Banksy_py (GPL-3.0).
- **Scanpy**: Wolf FA, Angerer P, Theis FJ. SCANPY: large-scale single-cell gene expression data
  analysis. *Genome Biol* 2018;19:15. doi:10.1186/s13059-017-1382-0. PMID 29409532.
- **anndata**: Virshup I, Rybakov S, Theis FJ, Angerer P, Wolf FA. anndata: Access and store
  annotated data matrices. *JOSS* 2024;9(101):4371. doi:10.21105/joss.04371. (No PMID.)
- **Harmony**: Korsunsky I, Millard N, Fan J, Slowikowski K, Zhang F, Wei K, Baglaenko Y, Brenner M,
  Loh P-R, Raychaudhuri S. Fast, sensitive and accurate integration of single-cell data with
  Harmony. *Nat Methods* 2019;16(12):1289–1296. doi:10.1038/s41592-019-0619-0. PMID 31740819.
  harmonypy 2.1.0: https://github.com/slowkow/harmonypy.
- **Leiden**: Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected
  communities. *Sci Rep* 2019;9:5233. doi:10.1038/s41598-019-41695-z. PMID 30914743.
- **UMAP**: McInnes L, Healy J, Saul N, Großberger L. UMAP: Uniform Manifold Approximation and
  Projection. *JOSS* 2018;3(29):861. doi:10.21105/joss.00861. (No PMID.)
- **Spatial clustering benchmark** (SDMBench, for choosing and judging domain methods): Yuan Z, Zhao F,
  Lin S, Zhao Y, Yao J, Cui Y, Zhang X-Y, Zhao Y. Benchmarking spatial clustering methods with
  spatially resolved transcriptomics data. *Nat Methods* 2024;21(4):712–722.
  doi:10.1038/s41592-024-02215-8. PMID 38491270.
- **TLS Visium demo data**: Dawo S, Nonchev K, Silina K. 10x Visium Spatial Transcriptomics Dataset:
  Kidney (3) and Lung (5) Cancer with Tertiary Lymphoid Structures. Zenodo 2025.
  doi:10.5281/zenodo.14620362 (CC BY 4.0).
