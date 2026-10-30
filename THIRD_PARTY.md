# Third-party software

Our code (`processor/`, `tests/`) is Apache-2.0 (see `LICENSE`). The image runs it with Python
packages loaded into the same process, four of which are GPL: **pybanksy** (GPL-3.0), **harmonypy**
(GPL-3.0-or-later since 2.0), **leidenalg** (GPL-3.0-or-later) and **igraph** (GPL-2.0-or-later). So
**the image as a whole is distributed under the GPL-3.0**; each package keeps its own licence, and
their sources are on PyPI at the versions below. Nothing here is modified, and no GPL code is copied
into `processor/`: pybanksy is called as a library (`banksy.main.generate_spatial_weights_fixed_nbrs`
builds the spatial weight graphs); the feature arithmetic in `processor/features.py` is our own
implementation of the published method, checked against pybanksy's output in
`tests/test_processor.py`.

| Component | Version | What | Licence | Source |
|---|---|---|---|---|
| pybanksy | 1.3.5 | BANKSY spatial weight graphs (scaled-Gaussian kNN weights, azimuthal phases) | GPL-3.0 | https://github.com/prabhakarlab/Banksy_py |
| harmonypy | 2.1.0 | Harmony integration of the PCs by sample | GPL-3.0-or-later | https://github.com/slowkow/harmonypy |
| leidenalg | 0.12.0 | Leiden clustering (via scanpy, `flavor="igraph"` uses igraph's Leiden; pybanksy imports leidenalg) | GPL-3.0-or-later | https://github.com/vtraag/leidenalg |
| igraph (python-igraph) | 1.0.0 | Graph library, Leiden | GPL-2.0-or-later | https://igraph.org/python |
| scanpy | 1.12.5 | kNN graph, Leiden, UMAP, Wilcoxon markers, dot plot | BSD-3-Clause | https://github.com/scverse/scanpy |
| anndata | 0.13.4 | `.h5ad` reading | BSD-3-Clause | https://github.com/scverse/anndata |
| umap-learn / pynndescent | 0.5.12 / 0.7.0 | UMAP, approximate kNN | BSD-3-Clause / BSD-2-Clause | https://github.com/lmcinnes/umap |
| scikit-learn | 1.9.1 | Spatial kNN (ball tree), ARI / NMI | BSD-3-Clause | https://scikit-learn.org |
| numpy, scipy, pandas, h5py | 2.5.4, 1.18.1, 3.1.0, 3.16.0 | Arrays, sparse matrices, eigendecomposition, tables, HDF5 | BSD-3-Clause (numpy also 0BSD / MIT / Zlib / CC0 parts) | PyPI |
| numba / llvmlite | 0.68.1 / 0.50.0 | JIT for scanpy / UMAP | BSD-2-Clause; llvmlite also Apache-2.0 WITH LLVM-exception | PyPI |
| matplotlib, seaborn | 3.11.2, 0.13.2 | Figures (seaborn is imported by pybanksy) | PSF-based matplotlib licence, BSD-3-Clause | PyPI |
| Python | 3.12.16 | Interpreter (`python:3.12.16-slim-bookworm`, Debian 12) | PSF-2.0; Debian packages under their own free licences | https://hub.docker.com/_/python |

All other Python packages in the image (pinned in `requirements.txt`) are under permissive
licences: MIT (annotated-types, array-api-compat, donfig, fonttools, formulaic, interface_meta,
narwhals, natsort, numcodecs, pydantic, pydantic-core, pydantic-settings, pyparsing, PyYAML, six,
texttable, typing-inspection, zarr), BSD (cloudpickle, contourpy, cycler, joblib, kiwisolver,
msgspec, networkx, patsy, python-dotenv, scverse-misc, statsmodels, threadpoolctl, wrapt),
MPL-2.0 (certifi, fast-array-utils, legacy-api-wrap, session-info2, tqdm with MIT parts),
Apache-2.0 (google-crc32c, tzdata), Apache-2.0 or BSD-2-Clause (packaging), BSD / Apache-2.0
(python-dateutil), MIT-CMU (pillow), PSF-2.0 (typing_extensions).

`tests/` (not in the image) also uses pytest (MIT).

## Citations

See README.md, Citations.
