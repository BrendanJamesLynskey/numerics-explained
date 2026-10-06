"""
Chapters 6-10 and the tiny model, against independent references: numpy
re-implementations of LLM.int8()'s decomposition, vector-wise INT8,
SmoothQuant's identity, NF4 against INT4 and the dot-product pipelines;
the standard library's NormalDist for the normal CDF and quantiles (and so
NF4's derivation); and a numpy forward pass of the tiny transformer.
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference"))

import numerics as N  # noqa: E402
import tiny as T  # noqa: E402

OL = N.demo_outlier_layer(8, 16, 32, 13)
LAY = N.demo_layer(8, 16, 64, 21)


def np_int8_rows(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric INT8 per row (absmax / 127), ties to even."""
    amax = np.abs(M).max(axis=1)
    s = np.where(amax > 0, amax / 127, 1.0)
    return np.clip(np.rint(M / s[:, None]), -127, 127), s


# --------------------------------------------------------------------------
# Chapter 6
# --------------------------------------------------------------------------


def test_outlier_layer_shape():
    X = np.array(OL["X"])
    amax = np.abs(X).max(axis=1)
    assert sorted(np.nonzero(amax > 6)[0].tolist()) == list(N.OUTLIER_CHANNELS)
    assert amax[[i for i in range(16) if i not in N.OUTLIER_CHANNELS]].max() < 6
    # systematic: an outlier channel has one sign in every token
    for i in N.OUTLIER_CHANNELS:
        assert np.all(X[i] > 0) or np.all(X[i] < 0)


def np_vectorwise(W: np.ndarray, X: np.ndarray, ch: list[int]) -> np.ndarray:
    qw, sw = np_int8_rows(W[:, ch])
    qx, sx = np_int8_rows(X[ch, :].T)
    return (qw @ qx.T) * (sw[:, None] * sx[None, :])


def test_vectorwise_and_decomposition_vs_numpy():
    W = np.array(OL["W"])
    X = np.array(OL["X"])
    allc = list(range(16))
    mine = np.array(N.int8_vectorwise(OL["W"], OL["X"], allc))
    assert np.allclose(mine, np_vectorwise(W, X, allc), rtol=1e-13, atol=1e-15)
    r = N.llm_int8(OL["W"], OL["X"])
    normal = [i for i in allc if i not in r["outliers"]]
    ref = W[:, r["outliers"]] @ X[r["outliers"], :] + np_vectorwise(W, X, normal)
    assert np.allclose(np.array(r["Y"]), ref, rtol=1e-13, atol=1e-15)


def test_outlier_story():
    o = N.outlier_steps(OL["W"], OL["X"])
    assert o["outliers"] == [3, 11]
    # the decomposition removes almost all of the error
    assert o["err_mixed"] * 50 < o["err_vector"] < o["err_tensor"] * 1.1
    assert o["normal_levels"] < 64  # ordinary channels get a few dozen of 255 codes
    assert [s["phase"] for s in o["steps"]][:3] == ["acts", "tensor", "vector"]
    assert o["steps"][-1]["found"] == [3, 11]


def test_smoothquant_is_exact_in_real_arithmetic():
    W = np.array(OL["W"])
    X = np.array(OL["X"])
    for k8 in range(9):
        sq = N.smoothquant(OL["W"], OL["X"], k8)
        assert np.allclose(np.array(sq["Ws"]) @ np.array(sq["Xs"]), W @ X, rtol=1e-12, atol=1e-12)
        # alpha balances the maxima: xmax' / wmax' = (xmax / wmax)^(1 - 2 alpha)... at 0.5 they are equal
        if k8 == 4:
            st = N.smooth_steps(OL["W"], OL["X"])[4]
            assert np.allclose(st["xmax"], st["wmax"], rtol=1e-12)
    errs = [s["err"] for s in N.smooth_steps(OL["W"], OL["X"])]
    assert min(errs) < N.w8a8_error(OL["W"], OL["X"])["err"] / 3


# --------------------------------------------------------------------------
# Chapter 7
# --------------------------------------------------------------------------


def test_gptq_steps_end_where_gptq_ends():
    for bits in (4, 3):
        g = N.gptq_steps(LAY["W"], LAY["X"], bits)
        st = g["steps"]
        assert st[0]["err_gptq"] == 0 and len(st) == 17
        assert st[-1]["err_gptq"] == N.layer_error(LAY["W"], g["Q"], LAY["X"])
        assert st[-1]["err_rtn"] == N.layer_error(LAY["W"], N.rtn(LAY["W"], bits), LAY["X"])
        assert st[-1]["err_gptq"] < st[-1]["err_rtn"]
        # a column's update only moves the columns to its right
        for j, s in enumerate(st[1:]):
            for row in s["delta"]:
                assert all(v == 0 for v in row[: j + 1])
        # the updates add up to the total movement
        tot = np.sum([np.array(s["delta"]) for s in st], axis=0)
        final = np.array(N.gptq(LAY["W"], LAY["X"], bits)["steps"][-1]["w"])
        assert np.allclose(np.array(LAY["W"]) + tot, final, atol=1e-12)


# --------------------------------------------------------------------------
# Chapter 8
# --------------------------------------------------------------------------


def test_normal_cdf_and_ppf_vs_stdlib():
    nd = statistics.NormalDist()
    for x in np.linspace(-3, 3, 121):
        assert abs(N.norm_cdf(float(x)) - nd.cdf(float(x))) < 2e-16 * 8
    for p in np.linspace(0.02, 0.98, 97):
        assert abs(N.norm_ppf(float(p)) - nd.inv_cdf(float(p))) < 1e-14


def test_nf4_build_is_bitsandbytes():
    b = N.nf4_build()
    assert len(b["values"]) == 16 and b["values"][7] == 0.0 and b["values"][-1] == 1.0
    assert b["max_diff"] < 2e-7  # bitsandbytes builds its grid in float32
    assert abs(b["max"] - 1.8481309597573712) < 1e-14  # norm.ppf(0.9677083): "~1.845 standard deviations"
    assert len(b["steps"]) == 19


def test_nf4_vs_int4_vs_numpy():
    xs = N.demo_normal(4096, 17)
    r = N.nf4_vs_int4(xs)
    a = np.array(xs).reshape(-1, 64)
    amax = np.abs(a).max(axis=1, keepdims=True)
    s = amax / 7
    i4 = np.clip(np.rint(a / s), -7, 7) * s
    assert r["mse_int4"] == pytest.approx(float(((a - i4) ** 2).mean()), rel=1e-12)
    nf = np.array(N.NF4)
    t = a / amax
    idx = np.abs(t[..., None] - nf).argmin(axis=-1)
    assert r["mse_nf4"] == pytest.approx(float(((a - nf[idx] * amax) ** 2).mean()), rel=1e-9)
    assert r["mse_nf4"] < r["mse_int4"]


# --------------------------------------------------------------------------
# Chapter 10
# --------------------------------------------------------------------------


@pytest.mark.parametrize("s1,s2", [(5, 6), (7, 8)])
def test_dot_pipelines_vs_numpy(s1: int, s2: int):
    a = N.demo_block("normal", s1)
    b = N.demo_block("normal", s2)
    d = N.dot_steps(a, b)
    s32 = np.float32(0)
    s16 = np.float16(0)
    for x, y, st in zip(a, b, d["steps"]):
        s32 = np.float32(s32 + np.float32(x * y))
        s16 = np.float16(np.float32(s16) + np.float32(np.float16(x * y)))
        assert st["fp32"] == float(s32)
        assert st["fp16"] == float(s16)
    qa, sa = np_int8_rows(np.array([a]))
    qb, sb = np_int8_rows(np.array([b]))
    assert d["steps"][-1]["int8_acc"] == int((qa * qb).sum())
    assert d["steps"][-1]["int8"] == pytest.approx(float((qa * qb).sum() * sa[0] * sb[0]), rel=1e-15)
    # MXFP4: elements are E2M1 values, the shared scales powers of two
    ea = ml_dtypes.float4_e2m1fn
    for st in d["steps"]:
        for v in st["e2m1"]:
            assert float(np.float32(v).astype(ea)) == v
    assert d["steps"][-1]["ref"] == pytest.approx(float(np.dot(a, b)), rel=1e-14)


def test_energy_table():
    assert N.mac_pj("fp32") == 3.7 + 0.9
    assert N.HOROWITZ["dram32"]["pj"] / N.HOROWITZ["add32"]["pj"] == 6400


# --------------------------------------------------------------------------
# The tiny model against a numpy forward pass
# --------------------------------------------------------------------------


def np_forward(ids: list[int], w: dict, kv=None) -> np.ndarray:
    cfg = T.CONFIG
    D, H = cfg["d_model"], cfg["n_heads"]
    dh = D // H
    E = np.array(w["tok_emb"])
    pos = np.array(T.positional_encoding(cfg["seq_len"], D))
    x = E[ids] + pos

    def ln(z):
        mu = z.mean(axis=1, keepdims=True)
        var = ((z - mu) ** 2).mean(axis=1, keepdims=True)
        return (z - mu) / np.sqrt(var + 1e-5)

    S = len(ids)
    mask = np.triu(np.full((S, S), -np.inf), 1)
    for li, b in enumerate(w["blocks"]):
        h1 = ln(x)
        Q, K, V = (h1 @ np.array(b[k]) for k in ("W_q", "W_k", "W_v"))
        if kv is not None:
            Kq, Vq = kv(li, K.tolist(), V.tolist())
            K, V = np.array(Kq), np.array(Vq)
        heads = []
        for hh in range(H):
            sl = slice(hh * dh, (hh + 1) * dh)
            sc = Q[:, sl] @ K[:, sl].T / math.sqrt(dh) + mask
            p = np.exp(sc - sc.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            heads.append(p @ V[:, sl])
        h = x + np.concatenate(heads, axis=1) @ np.array(b["W_o"])
        pre = ln(h) @ np.array(b["W1"])
        act = 0.5 * pre * (1 + np.tanh(math.sqrt(2 / math.pi) * (pre + 0.044715 * pre**3)))
        x = h + act @ np.array(b["W2"])
    return ln(x) @ E.T


def test_tiny_forward_vs_numpy():
    w = T.model_weights()
    for p in T.PROMPTS:
        ids = T.encode_text(p)
        assert np.allclose(np.array(T.forward(ids, T.CONFIG, w)), np_forward(ids, w), rtol=1e-11, atol=1e-12)
    h = T.kv_hook("int4_kivi")
    ids = T.encode_text(T.PROMPTS[1])
    assert np.allclose(np.array(T.forward(ids, T.CONFIG, w, h)), np_forward(ids, w, h), rtol=1e-11, atol=1e-12)


def test_tiny_weight_scales():
    w = T.model_weights()
    E = np.array(w["tok_emb"])
    assert 0.9 < E.std() < 1.1
    for b in w["blocks"]:
        for name in T.LINEARS:
            W = np.array(b[name])
            assert 0.75 < W.std() * math.sqrt(W.shape[0]) < 1.25


def test_kv_quantisers():
    rng = np.random.default_rng(3)
    M = (rng.standard_normal((32, 16)) * 0.7).tolist()
    assert T._round_rows(M, "fp16") == np.array(M).astype(np.float16).astype(float).tolist()
    q, s = np_int8_rows(np.array(M).reshape(64, 8))
    assert np.allclose(np.array(T._int_rows(M, 8, 8)), (q * s[:, None]).reshape(32, 16), rtol=1e-15)
    f4 = T.fp4_e4m3_scaled(M)
    e2m1 = {float(v) for v in N.positive_values("e2m1")}
    for row, src in zip(f4, M):
        amax = max(abs(v) for v in src)
        sc = N.round_to(amax / 6, "e4m3", "rne", True)
        assert N.round_to(sc, "e4m3") == sc
        for v, x in zip(row, src):
            assert abs(v / sc) in e2m1
            # nearest E2M1 value to x / s (saturating at 6)
            best = min(e2m1, key=lambda e: (abs(abs(x / sc) - e), e))
            assert abs(abs(v / sc) - abs(x / sc)) <= abs(best - abs(x / sc)) + 1e-15


def test_tiny_experiment_story():
    e = T.experiment()
    assert e["weights"]["int8_ch"]["agree"] >= e["weights"]["int4_ch"]["agree"] >= e["weights"]["int3_ch"]["agree"]
    assert e["weights"]["nf4"]["drift"] < e["weights"]["int4_b64"]["drift"]  # same blocks: NF4's grid wins
    assert e["kv"]["fp16"]["agree"] == 128
    assert e["kv"]["int8_tok"]["drift"] < e["kv"]["int4_tok"]["drift"] < e["kv"]["int2_tok"]["drift"]
