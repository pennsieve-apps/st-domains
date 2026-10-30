# sample-input

This app reads the output of `st-qc`: `spatial/<sample_id>.h5ad` (log-normalized `X`, raw counts
in `layers["counts"]`, `var["highly_variable"]`, `obsm["spatial"]`, `uns["pennsieve"]`,
optionally `obs["annotation"]`) and st-prep's `metadata/samples.csv`. The contract is in
`workflow-spatial-transcriptomics/docs/conventions.md`. st-prep's output (raw counts in `X`) also
works: the app then normalizes each sample itself (median total counts, `log1p`) and computes
highly variable genes, and says so in the log and `provenance.json`.

## Synthetic fixture (no download)

```sh
make fixture PYTHON=.venv/bin/python   # data/fixture/{linked,flat}/
make run PYTHON=.venv/bin/python       # Docker over data/fixture/linked/ -> data/output/
```

Two Visium-like samples, `S1` and `S2`, each a 30 × 30 hex grid (900 spots, full-resolution
pixel coordinates, a lowres "H&E" stand-in) with 300 genes and four planted, contiguous tissue
zones (`Zone_A` … `Zone_D`), each with its own 30-gene programme at 4× the base rate.
`obs["annotation"]` holds the zone for 85 % of the spots, `Unassigned` for the rest, and in
`S2` a small `Extra` class inside `Zone_C` (to exercise the confusion table). `dev.env` asks for
`N_DOMAINS=4`; BANKSY recovers the zones with ARI ≈ 0.8 (most of the disagreement is the
boundary spots, which BANKSY at `LAMBDA=0.8` tends to give their own small domains at higher
resolutions).

## Real data

Run `st-prep` and `st-qc` on a demo dataset of the workflow repository
(`scripts/fetch-sample-data.sh`):

- Dawo S, Nonchev K, Silina K. *10x Visium Spatial Transcriptomics Dataset: Kidney (3) and Lung
  (5) Cancer with Tertiary Lymphoid Structures.* Zenodo 2025, https://doi.org/10.5281/zenodo.14620362
  (CC BY 4.0): 8 FFPE sections with pathologist spot annotations (TLS, Immune, Tumor, Normal,
  Lymph_node, Unassigned), which st-prep joins onto `obs["annotation"]`; the agreement tables
  compare the domains with them.
- 10x Genomics *Human Lymph Node* (Visium, Space Ranger 1.1.0), 4,035 spots, no annotation.
- 10x Genomics *FFPE Human Breast Cancer* Xenium replicates 1 and 2 (Janesick et al. 2023,
  https://doi.org/10.1038/s41467-023-43458-x; CC BY 4.0): 313-gene panel, so every gene is used.

Then point this app at st-qc's `spatial` output and st-prep's `metadata` output:

```sh
mkdir -p input && ln -s /path/to/st-qc/output/spatial input/spatial \
  && ln -s /path/to/st-prep/output/metadata input/metadata
docker run --rm --platform linux/amd64 --network none --memory 4g \
  -e INPUT_DIR=/data/input -e OUTPUT_DIR=/data/output \
  -v "$PWD/input:/data/input:ro" -v "/path/to/st-qc/output:/path/to/st-qc/output:ro" \
  -v "/path/to/st-prep/output:/path/to/st-prep/output:ro" \
  -v "$PWD/data/output:/data/output" st-domains
```

(Or copy the two folders into `input/` instead of linking them.)
