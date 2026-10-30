"""
BANKSY features (Singhal et al. 2024), computed in row chunks.

pybanksy (GPL-3.0, called as a library) builds the per-sample spatial weight graphs:
`banksy.main.generate_spatial_weights_fixed_nbrs` with the scaled-Gaussian decay,
K_GEOM neighbours for the neighbour mean (m = 0) and 2 x K_GEOM neighbours with the
azimuthal phase exp(i * theta) for the first-order azimuthal Gabor filter (AGF, m = 1).

The feature matrix is the one pybanksy's `generate_banksy_matrix` / `concatenate_all`
produce (max_m = 1), per sample:

    own  = X                                   (log-normalized expression of the genes used)
    nbr  = W0 @ X                              (Gaussian-weighted mean of the K_GEOM neighbours)
    agf  = | W1 @ X - rowsum(W1) * mean_nbr |  (AGF magnitude; mean_nbr = unweighted mean of
                                                the 2 x K_GEOM neighbours, as in pybanksy >= 1.1)

each block z-scored per gene over the sample's observations (zero-variance genes -> 0) and
scaled by sqrt(1 - LAMBDA), sqrt(2 LAMBDA / 3), sqrt(LAMBDA / 3). pybanksy materialises all
of this densely in float64 (about 24 GB for 500 k bins x 2,000 genes), so we compute the same
numbers in row chunks, in float32, and get the PCA of the concatenated, scaled matrix
(pooled over samples) from its exact covariance (genes x 3 is at most a few thousand
columns) without ever holding the matrix: pass 1 collects per-sample block means / SDs,
pass 2 the covariance, pass 3 projects. tests/test_features.py checks the numbers against
pybanksy's own generate_banksy_matrix.
"""
from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

log = logging.getLogger("st-domains")

CHUNK_ELEMENTS = 8_000_000      # rows per chunk x genes; ~32 MB per float32 block


@dataclass
class SampleGraph:
    w0: sp.csr_matrix           # (n, n) real, rows sum to 1: neighbour mean (K_GEOM neighbours)
    w1: sp.csr_matrix           # (n, n) complex: AGF m = 1 weights (2 x K_GEOM neighbours)
    adj1: sp.csr_matrix         # (n, n) binary pattern of w1, rows scaled to 1 / degree
    w1_rowsum: np.ndarray       # (n,) complex
    k_geom: int


def scale_factors(lam: float) -> np.ndarray:
    """sqrt of the block weights for (own, nbr, agf): 1 - lambda, then lambda split 2:1 (pybanksy, max_m = 1)."""
    return np.sqrt(np.array([1.0 - lam, lam * 2.0 / 3.0, lam / 3.0]))


def build_graph(xy: np.ndarray, k_geom: int) -> SampleGraph:
    """BANKSY's spatial weight graphs for one sample, from pybanksy (its own kNN; no radius cap)."""
    from banksy.main import generate_spatial_weights_fixed_nbrs
    from sklearn.neighbors import NearestNeighbors

    n = xy.shape[0]
    k = int(min(k_geom, (n - 1) // 2))
    if k < 1:
        raise ValueError(f"{n} observations: too few for a spatial neighbourhood")
    if k < k_geom:
        log.warning("K_GEOM %d capped to %d (only %d observations)", k_geom, k, n)
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    nbrs = NearestNeighbors(algorithm="ball_tree").fit(xy)
    with contextlib.redirect_stdout(io.StringIO()):     # pybanksy's @timer prints every call
        w0, _, _ = generate_spatial_weights_fixed_nbrs(xy, m=0, num_neighbours=k, decay_type="scaled_gaussian",
                                                       nbr_object=nbrs, verbose=False, max_radius=None)
        w1, _, _ = generate_spatial_weights_fixed_nbrs(xy, m=1, num_neighbours=k, decay_type="scaled_gaussian",
                                                       nbr_object=nbrs, verbose=False, max_radius=None)
    w0 = sp.csr_matrix(w0, dtype=np.float64)
    w1 = sp.csr_matrix(w1, dtype=np.complex128)
    adj1 = sp.csr_matrix((np.ones(w1.nnz), w1.indices, w1.indptr), shape=w1.shape)
    deg = np.asarray(adj1.sum(axis=1)).ravel()
    adj1 = sp.csr_matrix(sp.diags(1.0 / np.maximum(deg, 1.0)) @ adj1)
    rowsum = np.asarray(w1.sum(axis=1)).ravel()
    return SampleGraph(w0=w0, w1=w1, adj1=adj1, w1_rowsum=rowsum, k_geom=k)


def _chunks(n: int, n_cols: int):
    step = max(256, CHUNK_ELEMENTS // max(1, n_cols))
    for r0 in range(0, n, step):
        yield r0, min(n, r0 + step)


def raw_blocks(x: sp.csr_matrix, g: SampleGraph, r0: int, r1: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unscaled (own, nbr, agf) features of rows r0:r1, float32 dense."""
    own = x[r0:r1].toarray().astype(np.float32, copy=False)
    nbr = np.asarray((g.w0[r0:r1] @ x).toarray(), dtype=np.float32)
    weighted = (g.w1[r0:r1] @ x).toarray()                       # complex128
    centre = (g.adj1[r0:r1] @ x).toarray()                        # unweighted mean of the AGF neighbours
    agf = np.abs(weighted - g.w1_rowsum[r0:r1, None] * centre).astype(np.float32)
    return own, nbr, agf


@dataclass
class BlockStats:
    mean: np.ndarray            # (3 * genes,)
    inv: np.ndarray             # scale factor / SD, 0 where SD == 0


def block_stats(x: sp.csr_matrix, g: SampleGraph, lam: float) -> BlockStats:
    n, n_genes = x.shape
    s1 = np.zeros(3 * n_genes)
    s2 = np.zeros(3 * n_genes)
    for r0, r1 in _chunks(n, 3 * n_genes):
        f = np.hstack(raw_blocks(x, g, r0, r1)).astype(np.float64)
        s1 += f.sum(axis=0)
        s2 += np.square(f).sum(axis=0)
    mean = s1 / n
    var = np.maximum(s2 / n - np.square(mean), 0.0)              # population variance, as pybanksy's zscore
    sd = np.sqrt(var)
    scale = np.repeat(scale_factors(lam), n_genes)
    tiny = sd <= 1e-12 * np.maximum(1.0, np.abs(mean))
    inv = np.where(tiny, 0.0, scale / np.where(tiny, 1.0, sd))
    return BlockStats(mean=mean, inv=inv)


def scaled_chunks(x: sp.csr_matrix, g: SampleGraph, st: BlockStats):
    """Yield (r0, r1, Z) with Z the z-scored, lambda-scaled BANKSY features of rows r0:r1 (float32)."""
    n, n_genes = x.shape
    mean, inv = st.mean.astype(np.float32), st.inv.astype(np.float32)
    for r0, r1 in _chunks(n, 3 * n_genes):
        f = np.hstack(raw_blocks(x, g, r0, r1))
        f -= mean
        f *= inv
        yield r0, r1, f


def banksy_matrix(x: sp.csr_matrix, g: SampleGraph, lam: float) -> np.ndarray:
    """The whole scaled BANKSY matrix of one sample (tests and small data only)."""
    st = block_stats(x, g, lam)
    return np.vstack([z for _, _, z in scaled_chunks(x, g, st)])


@dataclass
class PCAResult:
    embedding: np.ndarray       # (n_total, n_pcs) float32, samples in the order given
    variance_ratio: np.ndarray  # (n_pcs,)
    n_features: int


def banksy_pca(samples: list[tuple[sp.csr_matrix, SampleGraph]], lam: float, n_pcs: int) -> PCAResult:
    """PCA of the row-concatenated BANKSY matrices (each sample z-scored on its own), from the covariance."""
    n_genes = samples[0][0].shape[1]
    d = 3 * n_genes
    stats = [block_stats(x, g, lam) for x, g in samples]
    cov = np.zeros((d, d))
    n_total = 0
    for (x, g), st in zip(samples, stats):
        for _, _, z in scaled_chunks(x, g, st):
            cov += z.T @ z
        n_total += x.shape[0]
    cov /= max(1, n_total - 1)                                    # every sample has mean 0 per column
    total = float(np.trace(cov))
    k = int(min(n_pcs, d, max(1, n_total - 1)))
    from scipy.linalg import eigh

    evals, evecs = eigh(cov, subset_by_index=[d - k, d - 1], driver="evr", overwrite_a=True, check_finite=False)
    del cov
    evals, evecs = np.maximum(evals[::-1], 0.0), evecs[:, ::-1]
    v = evecs[:, :k]
    flip = np.sign(v[np.argmax(np.abs(v), axis=0), np.arange(k)])  # deterministic signs
    v = (v * np.where(flip == 0, 1.0, flip)).astype(np.float32)
    emb = np.zeros((n_total, k), dtype=np.float32)
    off = 0
    for (x, g), st in zip(samples, stats):
        for r0, r1, z in scaled_chunks(x, g, st):
            emb[off + r0:off + r1] = z @ v
        off += x.shape[0]
    return PCAResult(embedding=emb, variance_ratio=evals[:k] / total if total > 0 else np.zeros(k), n_features=d)
