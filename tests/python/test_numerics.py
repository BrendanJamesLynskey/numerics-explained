"""
The Python reference against independent references:
  - numpy (FP32, FP16) and ml_dtypes (BF16, FP8, FP6, FP4) for every code
    and for many conversions;
  - an exhaustive oracle in exact rational arithmetic (fractions) for every
    rounding mode, saturation mode and format of 16 bits or fewer;
  - the worked values in the OCP FP8 and MX specifications;
  - bitsandbytes' NF4 thresholds;
  - numpy implementations of the quantisers, summations and GPTQ.
"""
from __future__ import annotations

import bisect
import math
import random
import statistics
import sys
from fractions import Fraction
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference"))

import numerics as N  # noqa: E402

ML = {
    "bf16": (ml_dtypes.bfloat16, np.uint16),
    "e5m2": (ml_dtypes.float8_e5m2, np.uint8),
    "e4m3": (ml_dtypes.float8_e4m3fn, np.uint8),
    "e3m2": (ml_dtypes.float6_e3m2fn, np.uint8),
    "e2m3": (ml_dtypes.float6_e2m3fn, np.uint8),
    "e2m1": (ml_dtypes.float4_e2m1fn, np.uint8),
    "fp16": (np.float16, np.uint16),
}
SMALL = ("fp16", "bf16", "e5m2", "e4m3", "e3m2", "e2m3", "e2m1")


def same(a: float, b: float) -> bool:
    """Equal as doubles, NaN equal to NaN, and the sign of zero respected."""
    if a != a or b != b:
        return a != a and b != b
    return a == b and math.copysign(1, a) == math.copysign(1, b)


# --------------------------------------------------------------------------
# Format constants against the specs
# --------------------------------------------------------------------------


def test_spec_tables():
    i = N.INFO
    # OFP8 Tables 1 and 2
    assert (i["e4m3"]["emin"], i["e4m3"]["emax"], i["e4m3"]["max"]) == (-6, 8, 448.0)
    assert (i["e5m2"]["emin"], i["e5m2"]["emax"], i["e5m2"]["max"]) == (-14, 15, 57344.0)
    assert i["e4m3"]["min_sub"] == 2.0**-9 and i["e5m2"]["min_sub"] == 2.0**-16
    assert N.decode(0b0_0000_111, "e4m3") == 0.875 * 2**-6
    assert N.decode(0b0_00000_11, "e5m2") == 0.75 * 2**-14
    assert N.decode(0b0_1111_110, "e4m3") == 448 and math.isnan(N.decode(0b1_1111_111, "e4m3"))
    assert N.decode(0b0_11110_11, "e5m2") == 57344 and N.decode(0b1_11111_00, "e5m2") == -math.inf
    for c in (0b0_11111_01, 0b0_11111_10, 0b0_11111_11):
        assert math.isnan(N.decode(c, "e5m2"))
    # OFP8 "dynamic range": 18 and 32 binades
    assert round(math.log2(i["e4m3"]["max"] / i["e4m3"]["min_sub"])) == 18
    assert round(math.log2(i["e5m2"]["max"] / i["e5m2"]["min_sub"])) == 32
    # MX Tables 4 and 5
    assert (N.decode(0b0_11_111, "e2m3"), N.decode(0b0_01_000, "e2m3"), N.decode(0b0_00_111, "e2m3"), N.decode(0b0_00_001, "e2m3")) == (7.5, 1.0, 0.875, 0.125)
    assert (N.decode(0b0_111_11, "e3m2"), N.decode(0b0_001_00, "e3m2"), N.decode(0b0_000_11, "e3m2"), N.decode(0b0_000_01, "e3m2")) == (28.0, 0.25, 0.1875, 0.0625)
    assert (N.decode(0b0_11_1, "e2m1"), N.decode(0b0_01_0, "e2m1"), N.decode(0b0_00_1, "e2m1")) == (6.0, 1.0, 0.5)
    # IEEE 754 binary16 / binary32 extremes
    assert i["fp16"]["max"] == 65504.0 and i["fp16"]["min_sub"] == 2.0**-24
    assert i["fp32"]["max"] == float(np.finfo(np.float32).max)
    assert i["bf16"]["max"] == float(ml_dtypes.finfo(ml_dtypes.bfloat16).max)
    # E8M0 (MX Table 7) and MXINT8 (Table 6)
    assert math.isnan(N.e8m0_decode(255)) and N.e8m0_decode(127) == 1.0
    assert N.e8m0_decode(0) == 2.0**-127 and N.e8m0_decode(254) == 2.0**127
    assert N.mx_element(10.0, "mxint8")["value"] == 127 / 64
    assert N.mx_element(1 / 64, "mxint8")["value"] == 1 / 64


# --------------------------------------------------------------------------
# decode: every code against numpy / ml_dtypes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fid", SMALL)
def test_decode_every_code(fid):
    dt, ut = ML[fid]
    n = 1 << N.INFO[fid]["bits"]
    codes = np.arange(n, dtype=ut)
    ref = codes.view(dt).astype(np.float64)
    for c in range(n):
        assert same(N.decode(c, fid), float(ref[c])), (fid, c)


def test_decode_fp32_codes():
    rnd = random.Random(1)
    codes = [0, 1, 0x7FFFFF, 0x800000, 0x7F7FFFFF, 0x7F800000, 0x7FC00000, 0x80000000, 0xFF800000, 0x00400000]
    codes += [rnd.getrandbits(32) for _ in range(200_000)]
    ref = np.array(codes, dtype=np.uint32).view(np.float32).astype(np.float64)
    for c, r in zip(codes, ref):
        assert same(N.decode(c, "fp32"), float(r)), hex(c)


# --------------------------------------------------------------------------
# encode (round to nearest even, non-saturating) against numpy / ml_dtypes
# --------------------------------------------------------------------------


def probe_values(fid: str, rnd: random.Random, k: int) -> list[float]:
    """Values around every scale the format covers, ties included."""
    f = N.INFO[fid]
    out = [0.0, -0.0, math.inf, -math.inf, f["max"], f["min_sub"], f["min_normal"]]
    lo = f["emin"] - f["m"] - 3
    hi = f["emax"] + 2
    for _ in range(k):
        e = rnd.randint(lo, hi)
        v = math.ldexp(1 + rnd.random(), e)
        out.append(v if rnd.random() < 0.5 else -v)
    if f["bits"] <= 16:
        vals = N.positive_values(fid)
        for a, b in zip(vals, vals[1:]):
            out += [(a + b) / 2, -(a + b) / 2, a + (b - a) / 3]
        top = vals[-1] + math.ldexp(1.0, f["emax"] - f["m"])
        out += [(vals[-1] + top) / 2, top, top * 2]
    return out


@pytest.mark.parametrize("fid", ("bf16", "e5m2", "e4m3", "e3m2", "e2m3", "e2m1"))
def test_encode_matches_ml_dtypes(fid):
    """From float32 inputs (ml_dtypes rounds a double via float32; see the README)."""
    dt, ut = ML[fid]
    rnd = random.Random(fid)
    xs = np.array(probe_values(fid, rnd, 20_000), dtype=np.float32)
    ref = xs.astype(dt).view(ut)
    sat = N.INFO[fid]["kind"] == "finite"  # ml_dtypes saturates FP6 and FP4
    for x, r in zip(xs.astype(np.float64), ref):
        assert N.encode(float(x), fid, "rne", sat) == int(r), (fid, float(x))


def test_encode_fp16_from_double_matches_numpy():
    rnd = random.Random(16)
    xs = probe_values("fp16", rnd, 50_000)
    xs += [1 + 2**-11 + 2**-40, 1 + 2**-11, 1 + 3 * 2**-11, 65520.0, 65519.99]
    ref = np.array(xs, dtype=np.float64).astype(np.float16).view(np.uint16)
    for x, r in zip(xs, ref):
        assert N.encode(x, "fp16") == int(r), x


def test_encode_fp32_from_double_matches_numpy():
    rnd = random.Random(32)
    xs = probe_values("fp32", rnd, 200_000)
    xs += [1 + 2**-24, 1 + 2**-24 + 2**-50, 1 + 3 * 2**-24, 2.0**-149 * 0.5, 2.0**-149 * 0.75]
    with np.errstate(over="ignore"):
        ref = np.array(xs, dtype=np.float64).astype(np.float32).view(np.uint32)
    for x, r in zip(xs, ref):
        assert N.encode(x, "fp32") == int(r), x


@pytest.mark.parametrize("fid", N.FORMAT_ORDER)
def test_round_trip_every_code(fid):
    f = N.INFO[fid]
    codes = range(1 << f["bits"]) if f["bits"] <= 16 else random.Random(3).sample(range(1 << 32), 50_000)
    for c in codes:
        v = N.decode(c, fid)
        if v != v:
            continue
        for mode in N.MODES:
            assert N.encode(v, fid, mode) == c, (fid, c, mode)


# --------------------------------------------------------------------------
# Every rounding mode against an exact oracle
# --------------------------------------------------------------------------


class Oracle:
    """Rounds by enumerating the format's values and comparing exactly (fractions)."""

    def __init__(self, fid: str) -> None:
        f = N.INFO[fid]
        self.f = f
        self.mags = N.positive_values(fid)  # ascending, code = index
        # the value one ulp past the max, as if the exponent were unbounded
        self.beyond = self.mags[-1] + math.ldexp(1.0, f["emax"] - f["m"])
        self.beyond_even = ((f["max_code"] + 1) & 1) == 0

    def __call__(self, x: float, mode: str, sat: bool, u: int) -> int:
        f = self.f
        neg = math.copysign(1, x) < 0
        sign = f["sign_bit"] if neg else 0
        a = abs(x)
        if a == math.inf:
            if not sat and f["kind"] == "ieee":
                return sign | f["inf_code"]
            if not sat and f["kind"] == "fn":
                return sign | f["nan_code"]
            return sign | f["max_code"]
        mags = self.mags
        k = bisect.bisect_left(mags, a)
        if k < len(mags) and mags[k] == a:
            return sign | k
        if k < len(mags):
            lo_c, lo, hi_c, hi = k - 1, mags[k - 1], k, mags[k]
            hi_even = hi_c % 2 == 0
        else:
            lo_c, lo, hi_c, hi = len(mags) - 1, mags[-1], None, self.beyond
            hi_even = self.beyond_even
        frac = (Fraction(a) - Fraction(lo)) / (Fraction(hi) - Fraction(lo))
        if a > hi:  # far beyond the format
            frac = Fraction(2)
        if mode == "rne":
            up = frac > Fraction(1, 2) or (frac == Fraction(1, 2) and hi_even)
        elif mode == "rna":
            up = frac >= Fraction(1, 2)
        elif mode == "rtz":
            up = False
        elif mode == "rup":
            up = not neg
        elif mode == "rdn":
            up = neg
        else:
            up = Fraction(u, 2**32) < frac
        if not up:
            return sign | lo_c
        if hi_c is not None:
            return sign | hi_c
        # overflow
        if sat or f["kind"] == "finite":
            return sign | f["max_code"]
        if mode in ("rup", "rdn"):
            return sign | (f["inf_code"] if f["kind"] == "ieee" else f["nan_code"])
        return sign | (f["inf_code"] if f["kind"] == "ieee" else f["nan_code"])


@pytest.mark.parametrize("fid", SMALL)
def test_every_mode_against_the_oracle(fid):
    orc = Oracle(fid)
    rnd = random.Random(fid + "modes")
    xs = probe_values(fid, rnd, 3000)
    for x in xs:
        if abs(x) > 4 * orc.beyond and abs(x) != math.inf:
            x = math.copysign(orc.beyond * (1 + rnd.random()), x)
        for mode in N.MODES:
            for sat in (False, True):
                u = rnd.getrandbits(32)
                assert N.encode(x, fid, mode, sat, u) == orc(x, mode, sat, u), (fid, x, mode, sat, u)


def test_ocp_conversion_table():
    """OFP8 Table 3: NaN, Inf and overflow in SAT and NONSAT modes."""
    for fid, nonsat_inf in (("e5m2", 0x7C), ("e4m3", 0x7F)):
        mx = N.INFO[fid]["max_code"]
        assert N.encode(math.inf, fid, sat=True) == mx
        assert N.encode(-math.inf, fid, sat=True) == 0x80 | mx
        assert N.encode(math.inf, fid) == nonsat_inf
        big = N.INFO[fid]["max"] * 4
        assert N.encode(big, fid, sat=True) == mx and N.encode(big, fid) == nonsat_inf
        assert N.encode(N.INFO[fid]["min_sub"] / 4, fid) == 0
        assert N.encode(-N.INFO[fid]["min_sub"] / 4, fid) == 0x80
    assert N.encode(math.nan, "e4m3") == 0x7F and N.encode(math.nan, "e5m2") == 0x7E
    # E4M3: 464 is a tie between 448 (mantissa 110, even) and 480 (the NaN code): 448
    assert N.encode(464, "e4m3") == 0x7E and N.encode(465, "e4m3") == 0x7F
    # E5M2: 61440 is a tie between 57344 (odd) and 65536: overflow to Inf
    assert N.encode(61440, "e5m2") == 0x7C and N.encode(61439, "e5m2") == 0x7B
    # IEEE 754 7.4: towards zero overflows to the largest finite number
    assert N.encode(1e6, "fp16", "rtz") == 0x7BFF and N.encode(-1e6, "fp16", "rup") == 0xFBFF
    assert N.encode(-1e6, "fp16", "rdn") == 0xFC00


def test_stochastic_rounding_is_unbiased():
    """E[SR(x)] = x: the mean of many draws approaches x (law of large numbers)."""
    rng = N.Rng(99)
    x = 1 + 0.3 * 2**-3  # 0.3 ulp above 1 in E4M3
    n = 20000
    s = 0.0
    for _ in range(n):
        s += N.round_to(x, "e4m3", "sr", False, rng.u32())
    assert abs(s / n - x) < 0.005 * 2**-3 * 10


# --------------------------------------------------------------------------
# MX blocks
# --------------------------------------------------------------------------


def test_mx_worked_examples():
    # max|V| = 3: largest power of two <= 3 is 2; FP4's largest power of two is 4: X = 1/2
    r = N.mx_quantise([3.0, 0.3, -1.1] + [0.0] * 29, "mxfp4")
    assert r["shared_exp"] == -1 and r["scale_code"] == 126
    assert r["values"][:3] == [3.0, 0.25, -1.0]
    # MXFP8 E4M3: 1.9 * 2^k scales to 486, beyond 448: clamped (MX 6.3, step 2)
    r = N.mx_quantise([1.9 * 2**5] + [0.0] * 31, "mxfp8_e4m3")
    assert r["shared_exp"] == 5 - 8 and r["values"][0] == 448 * 2.0**-3
    # all zeros: the smallest scale
    assert N.mx_quantise([0.0] * 32, "mxfp6_e3m2")["scale_code"] == 0
    # NaN makes the scale NaN
    assert N.mx_quantise([math.nan] + [1.0] * 31, "mxfp4")["scale_code"] == 255
    # MXINT8: 1 sign bit, 1 integer bit, 6 fraction bits, symmetric
    r = N.mx_quantise([1.0, 0.5, -0.3] + [0.0] * 29, "mxint8")
    assert r["shared_exp"] == 0 and r["values"][:3] == [1.0, 0.5, -19 / 64]


@pytest.mark.parametrize("mid", [m for m in N.MX_ORDER if m != "mxint8"])
def test_mx_elements_match_ml_dtypes(mid):
    elem = N.MX_FORMATS[mid]["elem"]
    dt, ut = ML[elem]
    rnd = random.Random(mid)
    for _ in range(300):
        scale = 2.0 ** rnd.randint(-20, 20)
        block = [float(np.float32(rnd.gauss(0, 1) * scale)) for _ in range(32)]
        r = N.mx_quantise(block, mid)
        X = 2.0 ** r["shared_exp"]
        amax = max(abs(v) for v in block)
        assert X == 2.0 ** math.floor(math.log2(amax)) / 2.0 ** N.MX_FORMATS[mid]["emax_elem"]
        mx = N.INFO[elem]["max"]
        t = np.clip(np.array([v / X for v in block], dtype=np.float32), -mx, mx)
        ref = t.astype(dt).astype(np.float64) * X
        assert [float(v) for v in ref] == r["values"]


# --------------------------------------------------------------------------
# NF4
# --------------------------------------------------------------------------

BNB_THRESHOLDS = [  # csrc/kernels.cu dQuantizeNF4, bitsandbytes 8336490, low to high
    -0.8480964004993439, -0.6106329262256622, -0.4599952697753906, -0.33967943489551544,
    -0.23460740596055984, -0.13791173323988914, -0.045525018125772476, 0.03979014977812767,
    0.1202552504837513, 0.2035212516784668, 0.2920137718319893, 0.3893125355243683,
    0.5016634166240692, 0.6427869200706482, 0.8614784181118011,
]


def test_nf4_thresholds_are_bitsandbytes():
    assert len(N.NF4_MID) == 15
    for mine, theirs in zip(N.NF4_MID, BNB_THRESHOLDS):
        assert np.float32(mine) == np.float32(theirs)
    for i, t in enumerate(N.NF4_MID):
        assert N.nf4_code(t) == i  # ties go to the lower code, as x > threshold does
        assert N.nf4_code(math.nextafter(t, 2)) == i + 1


def test_nf4_table_is_the_normal_map():
    """bitsandbytes' create_normal_map(offset=0.9677083, use_extra_value=True), in doubles."""
    nd = statistics.NormalDist()
    off = 0.9677083
    lin = lambda a, b, n: [a + (b - a) * i / (n - 1) for i in range(n)]  # noqa: E731
    v1 = [nd.inv_cdf(p) for p in lin(off, 0.5, 9)[:-1]]
    v3 = [-nd.inv_cdf(p) for p in lin(off, 0.5, 8)[:-1]]
    v = sorted(v1 + [0.0] + v3)
    mx = max(v)
    v = [x / mx for x in v]
    for a, b in zip(v, N.NF4):
        assert abs(a - b) < 2e-7  # torch builds the grid in float32


def test_nf4_blocks():
    xs = [0.1, -0.2, 0.05, 0.4] * 16 + [2.0, -1.0] * 32
    r = N.nf4_quantise(xs)
    assert r["absmax"] == [0.4, 2.0]
    assert r["codes"][3] == 15 and r["values"][3] == 0.4
    assert r["values"][64] == 2.0 and r["values"][65] == N.NF4[N.nf4_code(-0.5)] * 2.0


# --------------------------------------------------------------------------
# Integer quantisation against numpy
# --------------------------------------------------------------------------


def np_quant(W: np.ndarray, bits: int, scheme: str, gran: str, group: int) -> np.ndarray:
    rows, cols = W.shape
    if gran == "tensor":
        G = W.reshape(1, -1)
    elif gran == "channel":
        G = W
    else:
        G = W.reshape(rows * cols // group, group)
    if scheme == "sym":
        qmax = 2 ** (bits - 1) - 1
        amax = np.abs(G).max(axis=1, keepdims=True)
        s = np.where(amax > 0, amax / qmax, 1.0)
        q = np.clip(np.rint(G / s), -qmax, qmax)
        D = q * s
    else:
        qmax = 2**bits - 1
        mn = np.minimum(G.min(axis=1, keepdims=True), 0)
        mx = np.maximum(G.max(axis=1, keepdims=True), 0)
        s = np.where(mx > mn, (mx - mn) / qmax, 1.0)
        z = np.clip(np.rint(-mn / s), 0, qmax)
        q = np.clip(np.rint(G / s) + z, 0, qmax)
        D = (q - z) * s
    return D.reshape(rows, cols)


@pytest.mark.parametrize("bits", (8, 4, 3))
@pytest.mark.parametrize("scheme", ("sym", "asym"))
@pytest.mark.parametrize("gran,group", N.GRANULARITIES)
def test_quantise_matrix_matches_numpy(bits, scheme, gran, group):
    W = N.demo_weights(16, 64, 5)
    mine = N.quantise_matrix(W, bits, scheme, gran, group)
    ref = np_quant(np.array(W), bits, scheme, gran, group)
    assert np.array_equal(np.array(mine["deq"]), ref)


def test_int_params_closed_forms():
    p = N.int_params([-0.5, 0.25, 1.0], 8, "sym")
    assert p == {"scale": 1.0 / 127, "zero": 0, "lo": -127, "hi": 127}
    p = N.int_params([0.5, 2.0, 3.0], 4, "asym")  # range widened to [0, 3]
    assert p["scale"] == 3.0 / 15 and p["zero"] == 0
    p = N.int_params([-1.0, 3.0], 4, "asym")
    assert p["scale"] == 4.0 / 15 and p["zero"] == 4  # rint(3.75)
    assert N.rne_int(2.5) == 2 and N.rne_int(3.5) == 4 and N.rne_int(-2.5) == -2
    assert N.quantise_matrix([[1.0, -1.0]], 4, "sym", "group", 64)["bits_per_weight"] == 4 + 16 / 64


# --------------------------------------------------------------------------
# Summation against numpy's own float16 / float32 / bfloat16 arithmetic
# --------------------------------------------------------------------------

DT = {"fp16": np.float16, "fp32": np.float32, "bf16": ml_dtypes.bfloat16}


def np_naive(xs, dt):
    s = dt(0)
    for x in xs:
        s = dt(s + dt(x))
    return float(s)


def np_kahan(xs, dt):
    s = dt(0)
    c = dt(0)
    for x in xs:
        y = dt(dt(x) - c)
        t = dt(s + y)
        c = dt(dt(t - s) - y)
        s = t
    return float(s)


def np_pairwise(xs, dt, lo, hi):
    n = hi - lo
    if n == 1:
        return dt(xs[lo])
    p = 1
    while p * 2 < n:
        p *= 2
    return dt(np_pairwise(xs, dt, lo, lo + p) + np_pairwise(xs, dt, lo + p, hi))


@pytest.mark.parametrize("fid", ("fp16", "bf16", "fp32"))
@pytest.mark.parametrize("dist", ("uniform", "ones", "normal"))
def test_sums_match_numpy(fid, dist):
    xs = N.sum_inputs(3000, dist, 11)
    dt = DT[fid]
    assert N.sum_naive(xs, fid) == np_naive(xs, dt)
    assert N.sum_kahan(xs, fid) == np_kahan(xs, dt)
    assert N.sum_pairwise(xs, fid) == float(np_pairwise(xs, dt, 0, len(xs)))


def test_pairwise_stack_is_the_recursive_tree():
    for fid, dist in (("fp16", "uniform"), ("bf16", "normal")):
        r = N.sum_steps(fid, 1000, dist, 3, 37)
        for st in r["steps"]:
            assert st["pairwise"] == N.sum_pairwise(r["inputs"], fid, 0, st["i"])


def test_sum_steps_exact_prefix_and_stagnation():
    r = N.sum_steps("fp16", 8192, "uniform", 7, 128)
    xs = r["inputs"]
    exact = Fraction(0)
    for i, x in enumerate(xs):
        exact += Fraction(x)
        if (i + 1) % 1024 == 0:
            st = [s for s in r["steps"] if s["i"] == i + 1][0]
            assert Fraction(st["exact"]) == exact
    last = r["steps"][-1]
    assert last["naive"] == 2048.0  # FP16 stalls: above 2048 an ulp is 2 and every input is < 1
    assert abs(last["kahan"] - last["exact"]) <= 2.0  # within half an ulp at 4096
    ones = N.sum_steps("fp16", 4096, "ones", 1, 64)["steps"][-1]
    assert ones["naive"] == 2048.0 and ones["kahan"] == 4096.0 and ones["pairwise"] == 4096.0


# --------------------------------------------------------------------------
# Linear algebra and GPTQ
# --------------------------------------------------------------------------


def test_cholesky_and_inverse_match_numpy():
    lay = N.demo_layer(4, 12, 48, 3)
    H = N.hessian(lay["X"], 0.01)
    L = np.array(N.cholesky(H))
    assert np.allclose(L, np.linalg.cholesky(np.array(H)), rtol=1e-12, atol=1e-12)
    Hi = np.array(N.spd_inverse(H))
    assert np.allclose(Hi, np.linalg.inv(np.array(H)), rtol=1e-9, atol=1e-12)


def np_gptq(W, X, bits, damp):
    """GPTQ in its OBS form: after each column, eliminate it from the inverse Hessian."""
    W = np.array(W, dtype=np.float64)
    X = np.array(X, dtype=np.float64)
    d = W.shape[1]
    H = 2 * X @ X.T
    H += damp * np.mean(np.diag(H)) * np.eye(d)
    Hinv = np.linalg.inv(H)
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(W).max(axis=1) / qmax
    Q = np.zeros_like(W)
    for j in range(d):
        q = np.clip(np.rint(W[:, j] / s), -qmax, qmax) * s
        Q[:, j] = q
        e = (W[:, j] - q) / Hinv[j, j]
        W[:, j + 1 :] -= np.outer(e, Hinv[j, j + 1 :])
        Hinv = Hinv - np.outer(Hinv[:, j], Hinv[j, :]) / Hinv[j, j]
    return Q


def test_gptq_matches_the_obs_form_and_beats_rtn():
    lay = N.demo_layer(8, 16, 64, 21)
    W, X = lay["W"], lay["X"]
    r = N.gptq(W, X, 4)
    ref = np_gptq(W, X, 4, 0.01)
    assert np.allclose(np.array(r["Q"]), ref, rtol=0, atol=1e-12)
    e_gptq = N.layer_error(W, r["Q"], X)
    e_rtn = N.layer_error(W, N.rtn(W, 4), X)
    assert e_gptq < e_rtn


def test_pow_dyadic_and_smoothquant():
    for x in (0.001, 0.37, 1.0, 5.5, 1234.5):
        for k in range(9):
            assert math.isclose(N.pow_dyadic(x, k), x ** (k / 8), rel_tol=4e-16)
    lay = N.demo_layer(6, 8, 32, 9)
    sq = N.smoothquant(lay["W"], lay["X"])
    Y = np.array(lay["W"]) @ np.array(lay["X"])
    Ys = np.array(sq["Ws"]) @ np.array(sq["Xs"])
    assert np.allclose(Y, Ys, rtol=1e-12, atol=1e-12)  # X'W' = XW
    # after migration, max|X'_j| = max|W'_j| = sqrt(max|X_j| max|W_j|) at alpha = 0.5
    for j in range(8):
        xm = max(abs(v) for v in sq["Xs"][j])
        wm = max(abs(sq["Ws"][r][j]) for r in range(6))
        assert math.isclose(xm, wm, rel_tol=1e-12)


def test_awq_search_never_worse_than_plain():
    lay = N.demo_layer(8, 16, 64, 4)
    r = N.awq_search(lay["W"], lay["X"], 3, 8)
    errs = [x["err"] for x in r["results"]]
    assert r["results"][r["best"]]["err"] == min(errs) <= errs[0]


# --------------------------------------------------------------------------
# Animation sequences
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fid", N.FORMAT_ORDER)
def test_zoom_steps(fid):
    st = N.zoom_steps(fid)
    assert st[0]["count"] == N.INFO[fid]["max_code"]
    assert len(st) <= 41
    for a, b in zip(st, st[1:]):
        assert b["hi"] < a["hi"] and b["count"] < a["count"]
    if N.INFO[fid]["bits"] <= 16:
        vals = N.positive_values(fid)
        for s in st:
            assert s["count"] == sum(1 for v in vals[1:] if v <= s["hi"])
    assert st[-1]["subnormal"]


def test_rounding_steps():
    st = N.rounding_steps("e4m3", 64, "low", 3)
    assert sum(st[-1]["hist_rne"]) == sum(st[-1]["hist_sr"]) == 64
    for s in st:
        assert s["down"] <= s["x"] <= s["up"] and s["sr"] in (s["down"], s["up"])
        assert -0.5 <= s["err_rne"] <= 0.5
    assert st[-1]["mean_rne"] < -0.1  # "just above a grid point": nearest always rounds down
    ties = N.rounding_steps("e4m3", 64, "ties", 3)
    for s in ties:
        assert N.fields(N.encode(s["rne"], "e4m3"), "e4m3")["M"] % 2 == 0 or s["rne"] == 2.0


def test_stagnation():
    st = N.stagnation_steps("fp16", 1.0, 2.0**-13, 256, 5)
    assert all(s["rne"] == 1.0 for s in st)  # an eighth of an ulp never moves nearest-even
    assert abs(st[-1]["sr"] - st[-1]["exact"]) < 0.01


def test_mx_steps_and_granularity_and_zeropoint():
    blk = N.demo_block("outlier", 5)
    st = N.mx_steps(blk, "mxfp4")
    assert [s["phase"] for s in st[:3]] == ["raw", "amax", "scale"] and len(st) == 35
    assert abs(blk[st[1]["i"]]) == max(abs(v) for v in blk)
    W = N.demo_weights(16, 64, 5)
    g = N.granularity_steps(W, 4, "sym")
    mses = [s["mse"] for s in g]
    assert mses == sorted(mses, reverse=True)  # finer groups, smaller error, on this matrix
    zp = N.zeropoint_steps(N.demo_activations(24, 8), 4)
    assert zp["steps"][-1]["mse_asym"] < zp["steps"][-1]["mse_sym"]
    assert zp["used_asym"] > zp["used_sym"]
