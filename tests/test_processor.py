"""st-domains on the synthetic fixture (tests/make_fixture.py), no Docker."""
from __future__ import annotations

import contextlib
import io
import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from tests.make_fixture import make_fixture


@pytest.fixture(scope="session")
def fixture_dirs(tmp_path_factory):
    return make_fixture(tmp_path_factory.mktemp("fixture"), layout="both", spots=24, genes=300, seed=11)


def run(monkeypatch, input_dir, output_dir, **params):
    from processor.main import DEFAULTS, main

    for k in DEFAULTS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("MAX_OBS_FOR_UMAP", "300")
    for k, v in params.items():
        monkeypatch.setenv(k, str(v))
    return main()


# --- BANKSY features -------------------------------------------------------------------------

@pytest.fixture(scope="session")
def one_sample(fixture_dirs):
    import anndata as ad

    a = ad.read_h5ad(fixture_dirs["flat"] / "S1.h5ad")
    a = a[:, a.var["highly_variable"]].copy()
    return a


@pytest.mark.parametrize("lam", [0.2, 0.8])
def test_features_match_pybanksy(one_sample, lam):
    """Our chunked float32 features = pybanksy's create_nbr_matrix(center=True) + concatenate_all."""
    from banksy.embed_banksy import create_mat_list, create_nbr_matrix
    from banksy.initialize_banksy import initialize_banksy
    from banksy.main import concatenate_all

    from processor import features as F

    a = one_sample
    x = sp.csr_matrix(a.X, dtype=np.float32)
    g = F.build_graph(a.obsm["spatial"], 15)
    old = F.CHUNK_ELEMENTS
    F.CHUNK_ELEMENTS = 50 * 3 * x.shape[1]      # several chunks
    try:
        ours = F.banksy_matrix(x, g, lam)
    finally:
        F.CHUNK_ELEMENTS = old
    a.obs["xc"], a.obs["yc"] = a.obsm["spatial"][:, 0], a.obsm["spatial"][:, 1]
    with contextlib.redirect_stdout(io.StringIO()):
        bd = initialize_banksy(a, ("xc", "yc", "spatial"), 15, "scaled_gaussian", 1, False, False, False, False)
        nm = create_nbr_matrix(a, bd, "scaled_gaussian", 1, center=True, variance_balance=False, verbose=False)
        ml, _ = create_mat_list(a, nm, 1)
        ref = np.asarray(concatenate_all(ml, lam, None))
    assert ours.shape == ref.shape == (a.n_obs, 3 * a.n_vars)
    assert np.abs(ours - ref).max() < 1e-2           # pybanksy z-scores the own block in float32
    for b in range(3):
        blk = slice(b * a.n_vars, (b + 1) * a.n_vars)
        assert np.corrcoef(ours[:, blk].ravel(), ref[:, blk].ravel())[0, 1] > 0.99999


def test_pca_matches_sklearn(fixture_dirs):
    import anndata as ad
    from sklearn.decomposition import PCA

    from processor import features as F

    samples = []
    for sid in ("S1", "S2"):
        a = ad.read_h5ad(fixture_dirs["flat"] / f"{sid}.h5ad")
        x = sp.csr_matrix(a.X[:, :120], dtype=np.float32)
        samples.append((x, F.build_graph(a.obsm["spatial"], 10)))
    res = F.banksy_pca(samples, 0.8, 8)
    full = np.vstack([F.banksy_matrix(x, g, 0.8) for x, g in samples])
    ref = PCA(n_components=8, svd_solver="full").fit(full)
    emb = ref.transform(full)
    for j in range(5):
        assert abs(np.corrcoef(res.embedding[:, j], emb[:, j])[0, 1]) > 0.999
    assert np.allclose(res.variance_ratio, ref.explained_variance_ratio_, rtol=1e-3, atol=1e-5)


def test_scale_factors():
    from processor.features import scale_factors

    s = scale_factors(0.8)
    assert np.allclose(s ** 2, [0.2, 0.8 * 2 / 3, 0.8 / 3])
    assert np.isclose((s ** 2).sum(), 1.0)


# --- method pieces ------------------------------------------------------------------------------

def test_name_domains_by_size():
    from processor.domains import name_domains

    out, order = name_domains(np.array(["3", "1", "1", "3", "3", "0"]))
    assert order == ["D01", "D02", "D03"]
    assert list(out) == ["D01", "D02", "D02", "D01", "D01", "D03"]


def test_choose_genes_pools_hvgs():
    from processor.domains import SampleData, choose_genes

    genes = [f"G{i}" for i in range(1200)]

    def sample(flags, ranks):
        hv = pd.DataFrame({"highly_variable": flags, "highly_variable_rank": ranks}, index=genes)
        return SampleData("s", np.array([]), sp.csr_matrix((0, len(genes))), np.array(genes), np.zeros((0, 2)), None, hv)

    f1 = np.zeros(1200, bool); f1[:10] = True
    f2 = np.zeros(1200, bool); f2[5:15] = True
    r = np.where(np.arange(1200) < 15, np.arange(1200), np.nan)
    chosen, _ = choose_genes([sample(f1, r), sample(f2, r)], genes)
    assert len(chosen) == 10
    assert chosen[:5] == ["G5", "G6", "G7", "G8", "G9"]       # flagged in both samples first
    small, how = choose_genes([sample(f1, r)], genes[:500])
    assert len(small) == 500 and "panel" in how


def test_agreement_perfect_and_ignored():
    from processor.domains import agreement

    sid = np.array(["A"] * 6 + ["B"] * 6, dtype=object)
    ann = np.array(["x", "x", "y", "y", "Unassigned", ""] * 2, dtype=object)
    dom = np.array(["D01", "D01", "D02", "D02", "D01", "D02"] * 2, dtype=object)
    res = agreement(sid, ann, dom, ["A", "B"], {"Unassigned"})
    assert res["n_used"] == 8 and res["n_ignored"] == 4
    assert all(r["ari"] == pytest.approx(1.0) and r["nmi"] == pytest.approx(1.0) for r in res["agreement"])
    assert {m["annotation"]: m["domain"] for m in res["matches"]} == {"x": "D01", "y": "D02"}


# --- the whole app -------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["linked", "flat"])
def test_end_to_end(monkeypatch, tmp_path, fixture_dirs, layout):
    out = tmp_path / "out"
    assert run(monkeypatch, fixture_dirs[layout], out, N_DOMAINS=4) == 0
    labels = pd.read_csv(out / "labels.csv")
    assert list(labels.columns) == ["sample_id", "obs_id", "domain"]
    assert len(labels) == 2 * 24 * 24
    import anndata as ad

    s1 = ad.read_h5ad(fixture_dirs["flat"] / "S1.h5ad")
    assert list(labels.loc[labels.sample_id == "S1", "obs_id"]) == list(s1.obs_names)      # input order
    assert sorted(labels["domain"].unique()) == ["D01", "D02", "D03", "D04"]
    sizes = labels["domain"].value_counts()
    assert list(sizes.index) == sorted(sizes.index)                                         # D01 is largest
    mk = pd.read_csv(out / "domain_markers.csv")
    assert list(mk.columns) == ["domain", "gene", "logfoldchange", "pval_adj", "pct_in", "pct_out", "rank"]
    assert mk.groupby("domain").size().max() == 25
    summ = pd.read_csv(out / "domain_summary.csv")
    assert list(summ.columns) == ["sample_id", "domain", "n_obs", "fraction", "domain_n_samples"]
    assert np.allclose(summ.groupby("sample_id")["fraction"].sum(), 1.0)
    agr = pd.read_csv(out / "agreement.csv")
    assert list(agr.columns)[:4] == ["sample_id", "ari", "nmi", "n_obs"]
    pooled = agr.set_index("sample_id").loc["all"]
    assert pooled["ari"] > 0.6 and pooled["nmi"] > 0.6
    conf = pd.read_csv(out / "confusion.csv")
    assert "Unassigned" not in set(conf["annotation"])
    assert "Extra" in set(conf["annotation"])
    match = pd.read_csv(out / "annotation_match.csv")
    assert (match.loc[match.annotation.str.startswith("Zone"), "f1"] > 0.6).all()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_domains"] == 4 and summary["integration"] == "harmony"
    assert summary["n_domains_in_2plus_samples"] == 4                     # the zones exist in both samples
    assert summary["annotation"]["ari"] == pytest.approx(pooled["ari"], abs=1e-6)
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["app"] == "st-domains" and "pybanksy" in prov["tool_versions"]
    assert prov["samples"]["S1"]["status"] == "ok"
    figs = {p.name for p in (out / "figures").iterdir()}
    assert {"S1_domains.png", "S2_domains.png", "umap.png", "markers_dotplot.png", "confusion.png",
            "domain_composition.png"} <= figs


def test_deterministic(monkeypatch, tmp_path, fixture_dirs):
    assert run(monkeypatch, fixture_dirs["linked"], tmp_path / "a", RESOLUTION=0.3, MAX_OBS_FOR_UMAP=0) == 0
    assert run(monkeypatch, fixture_dirs["linked"], tmp_path / "b", RESOLUTION=0.3, MAX_OBS_FOR_UMAP=0) == 0
    a = pd.read_csv(tmp_path / "a" / "labels.csv")
    b = pd.read_csv(tmp_path / "b" / "labels.csv")
    assert a.equals(b)
    assert not (tmp_path / "a" / "figures" / "umap.png").exists()


def test_single_sample_no_annotation(monkeypatch, tmp_path, fixture_dirs):
    import anndata as ad

    inp = tmp_path / "in"
    inp.mkdir()
    a = ad.read_h5ad(fixture_dirs["flat"] / "S1.h5ad")
    del a.obs["annotation"]
    a.write_h5ad(inp / "S1.h5ad")
    out = tmp_path / "out"
    assert run(monkeypatch, inp, out, LAMBDA=0.2) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["integration"] == "none" and summary["annotation"] is None
    assert not (out / "agreement.csv").exists()


def test_raw_counts_input_is_normalized(monkeypatch, tmp_path, fixture_dirs):
    """st-prep output linked directly (X = counts, no 'normalized' flag) still runs."""
    import anndata as ad

    inp = tmp_path / "in"
    inp.mkdir()
    for sid in ("S1", "S2"):
        a = ad.read_h5ad(fixture_dirs["flat"] / f"{sid}.h5ad")
        a.X = a.layers["counts"].copy()
        del a.layers["counts"]
        a.uns["pennsieve"] = {k: v for k, v in a.uns["pennsieve"].items() if k not in ("normalized", "target_sum")}
        a.write_h5ad(inp / f"{sid}.h5ad")
    out = tmp_path / "out"
    assert run(monkeypatch, inp, out, N_DOMAINS=4, MAX_OBS_FOR_UMAP=0) == 0
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["samples"]["S1"]["normalized_here"] is True


def test_bad_sample_is_skipped(monkeypatch, tmp_path, fixture_dirs):
    import shutil

    inp = tmp_path / "in"
    shutil.copytree(fixture_dirs["flat"], inp)
    (inp / "S2.h5ad").write_bytes(b"not an h5ad")
    out = tmp_path / "out"
    assert run(monkeypatch, inp, out, MAX_OBS_FOR_UMAP=0) == 0
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["samples"]["S2"]["status"] == "failed" and prov["samples"]["S1"]["status"] == "ok"


def test_exit_codes(monkeypatch, tmp_path, fixture_dirs):
    assert run(monkeypatch, fixture_dirs["linked"], tmp_path / "o", LAMBDA=1.5) == 2
    assert run(monkeypatch, fixture_dirs["linked"], tmp_path / "o", INTEGRATION="scvi") == 2
    assert run(monkeypatch, fixture_dirs["linked"], tmp_path / "o", K_GEOM="2.5") == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run(monkeypatch, empty, tmp_path / "o") == 1
