"""
Numerics: the Python reference.

A bit-exact library of the number formats and quantisers that modern ML uses,
written in plain Python (no numpy) so that every operation is an IEEE 754
double operation whose order the TypeScript port, ``src/lib/num/model.ts``,
repeats exactly. ``scripts/make_fixtures.py`` writes the fixtures that port
must reproduce, and ``tests/python`` checks this file against independent
references: numpy and ``ml_dtypes`` for the formats they implement, an
exhaustive exact-arithmetic oracle (``fractions``) for every rounding mode,
and the worked values in the specifications for the rest.

What it covers:
  - minifloat formats: FP32, FP16, BF16 (IEEE 754-2019 and its bfloat16
    variant), FP8 E4M3 and E5M2 (OCP OFP8 v1.0), FP6 E3M2 and E2M3 and FP4
    E2M1 (OCP MX v1.0); encode and decode, every rounding mode IEEE 754
    defines (ties to even, ties away, towards +inf, towards -inf, towards
    zero) plus stochastic rounding, saturating and non-saturating overflow;
  - MX block formats (MXFP8, MXFP6, MXFP4, MXINT8): a shared E8M0 scale per
    block of 32, by the conversion of the MX spec section 6.3;
  - NF4 (QLoRA's NormalFloat), with bitsandbytes' code book;
  - integer quantisation (symmetric absmax and asymmetric zero-point; per
    tensor, per channel and per group);
  - summation in a format (naive, Kahan, pairwise);
  - GPTQ on a small layer, AWQ-style scaling and SmoothQuant migration;
  - the state sequences the site's animations draw (``*_steps``).

Conventions:
  - a value is a Python float (an IEEE double); a format value is held as
    its code (an int) and decoded exactly into a double;
  - every format in this file is narrower than half a double's precision,
    so rounding the exact double result of +, -, x, / or sqrt of two format
    values to the format is the correctly rounded format operation (double
    rounding is innocuous when p' >= 2p + 2; Figueroa, 1995);
  - sums in metrics are plain left-to-right loops (never Python's sum(),
    which compensates floats from 3.12, nor math.fsum).
"""
from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

SOURCES: dict[str, dict[str, str]] = {
    "ieee754": {
        "title": "IEEE Standard for Floating-Point Arithmetic, IEEE Std 754-2019",
        "url": "https://doi.org/10.1109/IEEESTD.2019.8766229",
    },
    "ofp8": {
        "title": "OCP 8-bit Floating Point Specification (OFP8), Revision 1.0 (2023-12-01 corrected biases)",
        "url": "https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1",
    },
    "mx": {
        "title": "OCP Microscaling Formats (MX) Specification, Version 1.0 (September 2023)",
        "url": "https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf",
    },
    "bf16": {
        "title": "Google Cloud, The bfloat16 numerical format",
        "url": "https://cloud.google.com/tpu/docs/bfloat16",
    },
    "fp8paper": {
        "title": "Micikevicius et al., FP8 Formats for Deep Learning (arXiv 2209.05433)",
        "url": "https://arxiv.org/abs/2209.05433",
    },
    "mxpaper": {
        "title": "Rouhani et al., Microscaling Data Formats for Deep Learning (arXiv 2310.10537)",
        "url": "https://arxiv.org/abs/2310.10537",
    },
    "qlora": {
        "title": "Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs (arXiv 2305.14314)",
        "url": "https://arxiv.org/abs/2305.14314",
    },
    "bnb": {
        "title": "bitsandbytes, get_4bit_type('nf4') and dQuantizeNF4 (commit 8336490)",
        "url": "https://github.com/bitsandbytes-foundation/bitsandbytes/blob/833649043474794b8fe7a4136e0c40faf077b2e0/bitsandbytes/functional.py",
    },
    "ml_dtypes": {
        "title": "ml_dtypes: NumPy dtype extensions for FP8, FP6, FP4 and bfloat16",
        "url": "https://github.com/jax-ml/ml_dtypes",
    },
}

# ---------------------------------------------------------------------------
# Minifloat formats
# ---------------------------------------------------------------------------

# kind: "ieee"   - all-ones exponent holds Inf (mantissa 0) and NaN (else);
#       "fn"     - finite + NaN: no Inf, NaN only at S.1111.111 (OCP E4M3);
#       "finite" - every code is a finite number (OCP MX FP6 and FP4).
FORMATS: dict[str, dict[str, Any]] = {
    "fp32": {"id": "fp32", "name": "FP32", "long": "IEEE binary32 (single)", "e": 8, "m": 23, "bias": 127, "kind": "ieee", "src": "ieee754", "ref": "Table 3.5, binary32"},
    "fp16": {"id": "fp16", "name": "FP16", "long": "IEEE binary16 (half)", "e": 5, "m": 10, "bias": 15, "kind": "ieee", "src": "ieee754", "ref": "Table 3.5, binary16"},
    "bf16": {"id": "bf16", "name": "BF16", "long": "bfloat16", "e": 8, "m": 7, "bias": 127, "kind": "ieee", "src": "bf16", "ref": "1 sign, 8 exponent, 7 mantissa bits; FP32's exponent"},
    "e5m2": {"id": "e5m2", "name": "FP8 E5M2", "long": "OCP FP8 E5M2", "e": 5, "m": 2, "bias": 15, "kind": "ieee", "src": "ofp8", "ref": "Tables 1 and 2"},
    "e4m3": {"id": "e4m3", "name": "FP8 E4M3", "long": "OCP FP8 E4M3", "e": 4, "m": 3, "bias": 7, "kind": "fn", "src": "ofp8", "ref": "Tables 1 and 2"},
    "e3m2": {"id": "e3m2", "name": "FP6 E3M2", "long": "OCP MX FP6 E3M2", "e": 3, "m": 2, "bias": 3, "kind": "finite", "src": "mx", "ref": "Table 4"},
    "e2m3": {"id": "e2m3", "name": "FP6 E2M3", "long": "OCP MX FP6 E2M3", "e": 2, "m": 3, "bias": 1, "kind": "finite", "src": "mx", "ref": "Table 4"},
    "e2m1": {"id": "e2m1", "name": "FP4 E2M1", "long": "OCP MX FP4 E2M1", "e": 2, "m": 1, "bias": 1, "kind": "finite", "src": "mx", "ref": "Table 5"},
}

FORMAT_ORDER = ("fp32", "fp16", "bf16", "e5m2", "e4m3", "e3m2", "e2m3", "e2m1")

MODES = ("rne", "rna", "rtz", "rup", "rdn", "sr")
MODE_NAMES = {
    "rne": "roundTiesToEven",
    "rna": "roundTiesToAway",
    "rtz": "roundTowardZero",
    "rup": "roundTowardPositive",
    "rdn": "roundTowardNegative",
    "sr": "stochastic",
}

TWO32 = 4294967296.0


def pow2(k: int) -> float:
    """2**k as a double, exactly."""
    return math.ldexp(1.0, k)


def exponent_of(a: float) -> int:
    """floor(log2(a)) for a finite a > 0, exactly."""
    return math.frexp(a)[1] - 1


def info(fid: str) -> dict[str, Any]:
    """A format's derived constants (all exact)."""
    f = FORMATS[fid]
    e, m, bias, kind = f["e"], f["m"], f["bias"], f["kind"]
    top = (1 << e) - 1
    emin = 1 - bias
    if kind == "ieee":
        emax = top - 1 - bias
        max_mant = (1 << m) - 1
        inf_code = top << m
        nan_code = (top << m) | (1 << (m - 1))
    elif kind == "fn":
        emax = top - bias
        max_mant = (1 << m) - 2
        inf_code = -1
        nan_code = (top << m) | ((1 << m) - 1)
    else:
        emax = top - bias
        max_mant = (1 << m) - 1
        inf_code = -1
        nan_code = -1
    max_code = ((emax + bias) << m) | max_mant
    max_finite = pow2(emax) * (1 + max_mant / (1 << m))
    min_normal = pow2(emin)
    min_sub = pow2(emin - m)
    return {
        "id": fid,
        "bits": 1 + e + m,
        "e": e,
        "m": m,
        "bias": bias,
        "kind": kind,
        "emin": emin,
        "emax": emax,
        "max": max_finite,
        "min_normal": min_normal,
        "min_sub": min_sub,
        "eps": pow2(-m),
        "max_code": max_code,
        "inf_code": inf_code,
        "nan_code": nan_code,
        "sign_bit": 1 << (e + m),
        "binades": emax - (emin - m) + 1,
    }


INFO = {fid: info(fid) for fid in FORMAT_ORDER}


def decode(code: int, fid: str) -> float:
    """The value of a code (NaN for NaN codes; signed zeros and infinities)."""
    f = INFO[fid]
    e, m, bias, kind = f["e"], f["m"], f["bias"], f["kind"]
    neg = (code >> (e + m)) & 1
    E = (code >> m) & ((1 << e) - 1)
    M = code & ((1 << m) - 1)
    top = (1 << e) - 1
    if kind == "ieee" and E == top:
        if M == 0:
            return -math.inf if neg else math.inf
        return math.nan
    if kind == "fn" and E == top and M == (1 << m) - 1:
        return math.nan
    if E == 0:
        v = pow2(f["emin"] - m) * M
    else:
        v = pow2(E - bias - m) * ((1 << m) + M)
    return -v if neg else v


def _round_up(mode: str, frac: float, n: float, neg: bool, u: int) -> bool:
    """Whether the magnitude rounds up, given its fractional part in ulps."""
    if frac == 0:
        return False
    if mode == "rne":
        return frac > 0.5 or (frac == 0.5 and n % 2 == 1)
    if mode == "rna":
        return frac >= 0.5
    if mode == "rtz":
        return False
    if mode == "rup":
        return not neg
    if mode == "rdn":
        return neg
    if mode == "sr":
        return u / TWO32 < frac
    raise ValueError(mode)


def _magnitude_code(v: float, f: dict[str, Any]) -> int:
    """The code of a representable magnitude v > 0."""
    m = f["m"]
    if v < f["min_normal"]:
        return int(v / f["min_sub"])
    E = exponent_of(v)
    return ((E + f["bias"]) << m) | (int(v / pow2(E - m)) - (1 << m))


def _overflow_code(f: dict[str, Any], mode: str, neg: bool, sat: bool) -> int:
    """IEEE 754 section 7.4 overflow, with OCP's saturating mode and NaN for E4M3."""
    if sat or f["kind"] == "finite":
        return f["max_code"]
    if mode == "rtz" or (mode == "rup" and neg) or (mode == "rdn" and not neg):
        return f["max_code"]
    return f["inf_code"] if f["kind"] == "ieee" else f["nan_code"]


def encode(x: float, fid: str, mode: str = "rne", sat: bool = False, u: int = 0) -> int:
    """
    Round the double x to the format and return its code.

    mode: one of MODES; for "sr" (stochastic), ``u`` is a uniform 32-bit
    integer and the magnitude rounds up when u / 2**32 is below the fraction
    of an ulp it lies above the value below it. sat: saturate on overflow
    (OCP's SAT mode); formats without Inf always saturate (MX spec 5.3.2/3).
    """
    f = INFO[fid]
    s = f["sign_bit"]
    neg = math.copysign(1.0, x) < 0
    sign = s if neg else 0
    if x != x:
        if f["kind"] == "finite":
            return 0  # no NaN encoding: implementation-defined (MX 5.3.2); this library gives +0
        return sign | f["nan_code"]
    a = abs(x)
    if a == math.inf:
        if not sat and f["kind"] == "ieee":
            return sign | f["inf_code"]
        if not sat and f["kind"] == "fn":
            return sign | f["nan_code"]
        return sign | f["max_code"]
    if a == 0:
        return sign
    ee = max(exponent_of(a), f["emin"])
    q = pow2(ee - f["m"])
    t = a / q
    n = math.floor(t)
    frac = t - n
    if _round_up(mode, frac, n, neg, u):
        n += 1
    if n == 0:
        return sign
    v = n * q
    if v > f["max"]:
        return sign | _overflow_code(f, mode, neg, sat)
    return sign | _magnitude_code(v, f)


def round_to(x: float, fid: str, mode: str = "rne", sat: bool = False, u: int = 0) -> float:
    """x rounded to the format, as a double."""
    return decode(encode(x, fid, mode, sat, u), fid)


def fields(code: int, fid: str) -> dict[str, int]:
    """Sign, biased exponent and mantissa fields of a code."""
    f = INFO[fid]
    e, m = f["e"], f["m"]
    return {"s": (code >> (e + m)) & 1, "E": (code >> m) & ((1 << e) - 1), "M": code & ((1 << m) - 1)}


def neighbours(x: float, fid: str) -> dict[str, float]:
    """The representable values either side of a finite x in range (down <= x <= up)."""
    lo = round_to(x, fid, "rdn")
    hi = round_to(x, fid, "rup")
    return {"down": lo, "up": hi}


def encode_cases(fid: str, n: int, seed: int) -> list[float]:
    """
    Inputs for the encode parity check, generated by integer arithmetic so
    both languages produce the same list: the specials, every midpoint and
    third-point between neighbouring values (8-bit formats and narrower),
    and n random values of either sign across the format's whole range.
    """
    f = INFO[fid]
    out = [0.0, -0.0, math.inf, -math.inf, f["max"], -f["max"], f["min_sub"], f["min_normal"], f["max"] * 3]
    if f["bits"] <= 8:
        vals = positive_values(fid)
        for i in range(len(vals) - 1):
            a = vals[i]
            b = vals[i + 1]
            out.append((a + b) / 2)
            out.append(-(a + b) / 2)
            out.append(a + (b - a) / 3)
    rng = Rng(seed)
    lo = f["emin"] - f["m"] - 3
    span = f["emax"] + 2 - lo + 1
    for _ in range(n):
        e = lo + int(rng.uniform() * span)
        v = (1 + rng.uniform()) * pow2(e)
        out.append(-v if rng.u32() & 1 else v)
    return out


def positive_values(fid: str) -> list[float]:
    """Every finite value >= 0, in order (formats of 16 bits or fewer)."""
    f = INFO[fid]
    if f["bits"] > 16:
        raise ValueError("too many values")
    return [decode(c, fid) for c in range(0, f["max_code"] + 1)]


# ---------------------------------------------------------------------------
# Pseudo-random numbers (integer only, so every language agrees)
# ---------------------------------------------------------------------------


def xorshift32(x: int) -> int:
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


class Rng:
    """xorshift32 (Marsaglia, 2003): u32() and uniform() in [0, 1)."""

    def __init__(self, seed: int) -> None:
        self.x = (seed & 0xFFFFFFFF) or 1

    def u32(self) -> int:
        self.x = xorshift32(self.x)
        return self.x

    def uniform(self) -> float:
        return self.u32() / TWO32

    def normal(self) -> float:
        """Approximately N(0, 1): the sum of 12 uniforms, minus 6 (Irwin-Hall)."""
        s = 0.0
        for _ in range(12):
            s += self.uniform()
        return s - 6.0


# ---------------------------------------------------------------------------
# Integer quantisation
# ---------------------------------------------------------------------------


def rne_int(t: float) -> int:
    """Round a double to the nearest integer, ties to even (numpy.rint)."""
    n = math.floor(t)
    frac = t - n
    if frac > 0.5 or (frac == 0.5 and n % 2 == 1):
        n += 1
    return int(n)


def int_range(bits: int, signed: bool, restricted: bool = True) -> tuple[int, int]:
    """Code range: signed restricted is symmetric (-127..127 for 8 bits)."""
    if signed:
        hi = (1 << (bits - 1)) - 1
        return (-hi if restricted else -hi - 1, hi)
    return (0, (1 << bits) - 1)


def int_params(xs: list[float], bits: int, scheme: str) -> dict[str, Any]:
    """
    Scale and zero point for a group of values.
    sym (absmax): scale = max|x| / (2^(b-1) - 1), zero = 0, codes in
    -(2^(b-1) - 1) .. 2^(b-1) - 1. asym (zero-point): the range [min, max],
    widened to include 0, maps onto 0 .. 2^b - 1; zero = round(-min / scale).
    A group of zeros gets scale 1.
    """
    if scheme == "sym":
        amax = 0.0
        for x in xs:
            if abs(x) > amax:
                amax = abs(x)
        lo, hi = int_range(bits, True)
        scale = amax / hi if amax > 0 else 1.0
        return {"scale": scale, "zero": 0, "lo": lo, "hi": hi}
    mn = 0.0
    mx = 0.0
    for x in xs:
        if x < mn:
            mn = x
        if x > mx:
            mx = x
    lo, hi = int_range(bits, False)
    scale = (mx - mn) / hi if mx > mn else 1.0
    zero = min(hi, max(lo, rne_int(-mn / scale)))
    return {"scale": scale, "zero": zero, "lo": lo, "hi": hi}


def quant_int(x: float, p: dict[str, Any]) -> int:
    q = rne_int(x / p["scale"]) + p["zero"]
    return min(p["hi"], max(p["lo"], q))


def dequant_int(q: int, p: dict[str, Any]) -> float:
    return (q - p["zero"]) * p["scale"]


def groups_of(rows: int, cols: int, gran: str, group: int = 0) -> list[tuple[int, int, int]]:
    """(row, col_start, col_end) of each group: one, one per row, or g columns."""
    if gran == "tensor":
        return [(-1, 0, cols)]
    if gran == "channel":
        return [(r, 0, cols) for r in range(rows)]
    if gran == "group":
        return [(r, c, min(cols, c + group)) for r in range(rows) for c in range(0, cols, group)]
    raise ValueError(gran)


def quantise_matrix(W: list[list[float]], bits: int, scheme: str, gran: str, group: int = 0) -> dict[str, Any]:
    """Quantise a weight matrix (rows = output channels) at a granularity."""
    rows, cols = len(W), len(W[0])
    gs = groups_of(rows, cols, gran, group)
    deq = [[0.0] * cols for _ in range(rows)]
    codes = [[0] * cols for _ in range(rows)]
    params = []
    for r, c0, c1 in gs:
        rr = range(rows) if r < 0 else [r]
        vals = [W[i][j] for i in rr for j in range(c0, c1)]
        p = int_params(vals, bits, scheme)
        params.append({"row": r, "c0": c0, "c1": c1, "scale": p["scale"], "zero": p["zero"]})
        for i in rr:
            for j in range(c0, c1):
                q = quant_int(W[i][j], p)
                codes[i][j] = q
                deq[i][j] = dequant_int(q, p)
    err = 0.0
    sig = 0.0
    amax_err = 0.0
    for i in range(rows):
        for j in range(cols):
            d = W[i][j] - deq[i][j]
            err += d * d
            sig += W[i][j] * W[i][j]
            if abs(d) > amax_err:
                amax_err = abs(d)
    n = rows * cols
    size = cols if gran == "channel" else (rows * cols if gran == "tensor" else group)
    overhead = 16 + (bits if scheme == "asym" else 0)
    return {
        "params": params,
        "codes": codes,
        "deq": deq,
        "mse": err / n,
        "signal": sig / n,
        "max_err": amax_err,
        "bits_per_weight": bits + overhead / size,
    }


# ---------------------------------------------------------------------------
# MX block formats (OCP MX v1.0)
# ---------------------------------------------------------------------------

MX_FORMATS: dict[str, dict[str, Any]] = {
    "mxfp8_e4m3": {"id": "mxfp8_e4m3", "name": "MXFP8 (E4M3)", "elem": "e4m3", "emax_elem": 8, "bits": 8},
    "mxfp8_e5m2": {"id": "mxfp8_e5m2", "name": "MXFP8 (E5M2)", "elem": "e5m2", "emax_elem": 15, "bits": 8},
    "mxfp6_e3m2": {"id": "mxfp6_e3m2", "name": "MXFP6 (E3M2)", "elem": "e3m2", "emax_elem": 4, "bits": 6},
    "mxfp6_e2m3": {"id": "mxfp6_e2m3", "name": "MXFP6 (E2M3)", "elem": "e2m3", "emax_elem": 2, "bits": 6},
    "mxfp4": {"id": "mxfp4", "name": "MXFP4 (E2M1)", "elem": "e2m1", "emax_elem": 2, "bits": 4},
    "mxint8": {"id": "mxint8", "name": "MXINT8", "elem": "int8", "emax_elem": 0, "bits": 8},
}
MX_ORDER = ("mxfp8_e4m3", "mxfp8_e5m2", "mxfp6_e3m2", "mxfp6_e2m3", "mxfp4", "mxint8")
MX_BLOCK = 32


def e8m0_decode(code: int) -> float:
    """E8M0 scale: 2^(code - 127); 0xFF is NaN (MX Table 7)."""
    return math.nan if code == 255 else pow2(code - 127)


def mx_element(t: float, mid: str, mode: str = "rne", u: int = 0) -> dict[str, Any]:
    """One scaled input t = V/X quantised to the element type (saturating)."""
    f = MX_FORMATS[mid]
    if f["elem"] == "int8":
        # one sign bit, one integer bit, six fraction bits (MX Table 6)
        n = t * 64
        if mode == "sr":
            fl = math.floor(n)
            q = int(fl) + (1 if u / TWO32 < n - fl else 0)
        else:
            q = rne_int(n)
        q = min(127, max(-127, q))
        return {"code": q & 0xFF, "value": q / 64}
    c = encode(t, f["elem"], mode, True, u)
    return {"code": c, "value": decode(c, f["elem"])}


def mx_quantise(block: list[float], mid: str, mode: str = "rne", seed: int = 1) -> dict[str, Any]:
    """
    MX spec section 6.3: X is the largest power of two <= max|V|, divided by
    the largest power of two the element type represents; P_i = V_i / X
    quantised to the element type, normal values beyond its max clamped.
    The shared exponent is clamped to E8M0's -127..127; an all-zero block
    gets X = 2^-127; a non-finite input makes X NaN (all values NaN).
    """
    f = MX_FORMATS[mid]
    amax = 0.0
    finite = True
    for v in block:
        if v != v or abs(v) == math.inf:
            finite = False
        elif abs(v) > amax:
            amax = abs(v)
    if not finite:
        return {"scale_code": 255, "shared_exp": None, "amax": amax, "codes": [0] * len(block), "values": [math.nan] * len(block)}
    if amax == 0:
        se = -127
    else:
        se = max(-127, min(127, exponent_of(amax) - f["emax_elem"]))
    X = pow2(se)
    rng = Rng(seed)
    codes = []
    values = []
    scaled = []
    for v in block:
        t = v / X
        u = rng.u32() if mode == "sr" else 0
        el = mx_element(t, mid, mode, u)
        scaled.append(t)
        codes.append(el["code"])
        values.append(X * el["value"])
    return {"scale_code": se + 127, "shared_exp": se, "amax": amax, "scaled": scaled, "codes": codes, "values": values}


# ---------------------------------------------------------------------------
# NF4 (QLoRA NormalFloat, bitsandbytes' code book)
# ---------------------------------------------------------------------------

NF4 = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
# Decision thresholds: midpoints between neighbouring code values. bitsandbytes'
# dQuantizeNF4 compares x > threshold, so a tie goes to the lower code.
NF4_MID = [(NF4[i] + NF4[i + 1]) / 2 for i in range(15)]


def nf4_code(t: float) -> int:
    """Nearest NF4 code to t in [-1, 1] (ties to the lower code)."""
    c = 0
    for i in range(15):
        if t > NF4_MID[i]:
            c = i + 1
    return c


def nf4_quantise(xs: list[float], block: int = 64) -> dict[str, Any]:
    """Blockwise absmax NF4: t = x / absmax per block of 64, then the nearest code."""
    codes = []
    values = []
    absmax = []
    for b0 in range(0, len(xs), block):
        blk = xs[b0 : b0 + block]
        am = 0.0
        for x in blk:
            if abs(x) > am:
                am = abs(x)
        s = am if am > 0 else 1.0
        absmax.append(am)
        for x in blk:
            c = nf4_code(x / s)
            codes.append(c)
            values.append(NF4[c] * s)
    return {"codes": codes, "values": values, "absmax": absmax}


# ---------------------------------------------------------------------------
# Summation in a format
# ---------------------------------------------------------------------------


def add_in(a: float, b: float, fid: str) -> float:
    """a + b in the format: the exact double sum rounded once (see the header)."""
    return round_to(a + b, fid)


def sum_naive(xs: list[float], fid: str) -> float:
    s = 0.0
    for x in xs:
        s = add_in(s, round_to(x, fid), fid)
    return s


def sum_kahan(xs: list[float], fid: str) -> float:
    """Kahan (1965) compensated summation, every operation in the format."""
    s = 0.0
    c = 0.0
    for x in xs:
        y = round_to(round_to(x, fid) - c, fid)
        t = add_in(s, y, fid)
        c = round_to(round_to(t - s, fid) - y, fid)
        s = t
    return s


def sum_pairwise(xs: list[float], fid: str, lo: int = 0, hi: int = -1) -> float:
    """
    Recursive pairwise summation: the left part is the largest power of two
    below n (so a power-of-two n splits in half, and every subtree over an
    aligned power-of-two block is a perfectly balanced tree).
    """
    if hi < 0:
        hi = len(xs)
    n = hi - lo
    if n == 0:
        return 0.0
    if n == 1:
        return round_to(xs[lo], fid)
    p = 1
    while p * 2 < n:
        p *= 2
    return add_in(sum_pairwise(xs, fid, lo, lo + p), sum_pairwise(xs, fid, lo + p, hi), fid)


class PairwiseStack:
    """
    sum_pairwise of a growing prefix, incrementally: a binary counter of
    complete power-of-two blocks (merged as they fill), folded from the
    right when the total is read. Equal to sum_pairwise(xs[:i]) exactly.
    """

    def __init__(self, fid: str) -> None:
        self.fid = fid
        self.blocks: list[list[float]] = []  # [size, value]

    def push(self, x: float) -> None:
        self.blocks.append([1, round_to(x, self.fid)])
        while len(self.blocks) >= 2 and self.blocks[-1][0] == self.blocks[-2][0]:
            r = self.blocks.pop()
            l = self.blocks.pop()
            self.blocks.append([l[0] * 2, add_in(l[1], r[1], self.fid)])

    def total(self) -> float:
        if not self.blocks:
            return 0.0
        acc = self.blocks[-1][1]
        for i in range(len(self.blocks) - 2, -1, -1):
            acc = add_in(self.blocks[i][1], acc, self.fid)
        return acc


# ---------------------------------------------------------------------------
# Data sets the chapters use (deterministic; pure arithmetic)
# ---------------------------------------------------------------------------


def sum_inputs(n: int, dist: str, seed: int) -> list[float]:
    """
    n values, each an FP16 number: uniform in [0, 1), all ones, or
    approximately normal. Being FP16 values, their double prefix sums are
    exact for n <= 8192 (multiples of 2^-24 below 2^29).
    """
    rng = Rng(seed)
    out = []
    for _ in range(n):
        if dist == "ones":
            v = 1.0
        elif dist == "uniform":
            v = rng.uniform()
        elif dist == "normal":
            v = rng.normal()
        else:
            raise ValueError(dist)
        out.append(round_to(v, "fp16", "rtz"))
    return out


def demo_weights(rows: int, cols: int, seed: int) -> list[list[float]]:
    """
    An illustrative weight matrix: approximately normal values, each row
    (output channel) with its own scale and a few input columns that are
    larger everywhere, the structure per-channel and per-group scales
    exploit. Rounded to FP16 so the data set prints exactly.
    """
    rng = Rng(seed)
    row_scale = [0.02 * (0.5 + 1.5 * rng.uniform()) for _ in range(rows)]
    col_scale = [1.0] * cols
    for _ in range(max(1, cols // 16)):
        j = int(rng.uniform() * cols)
        col_scale[j] = 3.0 + 3.0 * rng.uniform()
    return [[round_to(row_scale[i] * col_scale[j] * rng.normal(), "fp16") for j in range(cols)] for i in range(rows)]


def demo_activations(n: int, seed: int) -> list[float]:
    """Skewed, mostly positive values, like a GELU's outputs (illustrative), as FP16."""
    rng = Rng(seed)
    out = []
    for _ in range(n):
        z = rng.normal()
        v = z * (0.5 + 0.5 * z / 3) if z > -0.75 else -0.17 * rng.uniform()
        out.append(round_to(v, "fp16"))
    return out


def demo_block(kind: str, seed: int, k: int = MX_BLOCK) -> list[float]:
    """A block of 32 values for the MX animation: normal, with an outlier, or tiny."""
    rng = Rng(seed)
    vals = [round_to(rng.normal(), "fp16") for _ in range(k)]
    if kind == "outlier":
        vals[int(rng.uniform() * k)] = 24.0
    elif kind == "tiny":
        vals = [round_to(v * pow2(-12), "fp16") for v in vals]
    return vals


# ---------------------------------------------------------------------------
# Linear algebra for GPTQ (plain loops; the TS port repeats the order)
# ---------------------------------------------------------------------------


def cholesky(A: list[list[float]]) -> list[list[float]]:
    """Lower-triangular L with A = L L^T (A symmetric positive definite)."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j]
            for k in range(j):
                s -= L[i][k] * L[j][k]
            if i == j:
                if s <= 0:
                    raise ValueError("not positive definite")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def spd_inverse(A: list[list[float]]) -> list[list[float]]:
    """A^-1 from its Cholesky factor: solve L Y = I, then L^T X = Y."""
    n = len(A)
    L = cholesky(A)
    X = [[0.0] * n for _ in range(n)]
    for c in range(n):
        y = [0.0] * n
        for i in range(n):
            s = 1.0 if i == c else 0.0
            for k in range(i):
                s -= L[i][k] * y[k]
            y[i] = s / L[i][i]
        for i in range(n - 1, -1, -1):
            s = y[i]
            for k in range(i + 1, n):
                s -= L[k][i] * X[k][c]
            X[i][c] = s / L[i][i]
    return X


def hessian(X: list[list[float]], damp: float) -> list[list[float]]:
    """H = 2 X X^T + damp * mean(diag) * I, for X of shape (d_in, n_samples)."""
    d = len(X)
    n = len(X[0])
    H = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            s = 0.0
            for k in range(n):
                s += X[i][k] * X[j][k]
            H[i][j] = 2 * s
    tr = 0.0
    for i in range(d):
        tr += H[i][i]
    lam = damp * tr / d
    for i in range(d):
        H[i][i] += lam
    return H


def layer_error(W: list[list[float]], Wq: list[list[float]], X: list[list[float]]) -> float:
    """||W X - Wq X||_F^2, the layer-output error GPTQ minimises."""
    rows = len(W)
    d = len(X)
    n = len(X[0])
    err = 0.0
    for r in range(rows):
        for k in range(n):
            a = 0.0
            b = 0.0
            for i in range(d):
                a += W[r][i] * X[i][k]
                b += Wq[r][i] * X[i][k]
            err += (a - b) * (a - b)
    return err


def rtn(W: list[list[float]], bits: int) -> list[list[float]]:
    """Round to nearest, per output channel, symmetric (GPTQ's baseline)."""
    return quantise_matrix(W, bits, "sym", "channel")["deq"]


def gptq(W: list[list[float]], X: list[list[float]], bits: int, damp: float = 0.01) -> dict[str, Any]:
    """
    GPTQ (Frantar et al., 2022, Algorithm 1) on a small layer, one column at a
    time (no lazy batching, which changes the speed, not the result): with
    U the upper Cholesky factor of H^-1, quantise column j, then spread its
    error over the columns not yet quantised: W[:, j+1:] -= e_j U[j, j+1:],
    e_j = (w_j - q_j) / U[j, j]. Scales per output channel, symmetric,
    fixed from the original weights.
    """
    rows = len(W)
    d = len(W[0])
    H = hessian(X, damp)
    Hinv = spd_inverse(H)
    L = cholesky(Hinv)  # Hinv = L L^T, so U = L^T is the upper factor
    U = [[L[j][i] for j in range(d)] for i in range(d)]
    params = []
    for r in range(rows):
        params.append(int_params(W[r], bits, "sym"))
    Wc = [row[:] for row in W]
    Q = [[0.0] * d for _ in range(rows)]
    steps = []
    for j in range(d):
        errs = []
        for r in range(rows):
            w = Wc[r][j]
            q = dequant_int(quant_int(w, params[r]), params[r])
            Q[r][j] = q
            e = (w - q) / U[j][j]
            errs.append(e)
            for k in range(j + 1, d):
                Wc[r][k] -= e * U[j][k]
        steps.append({"col": j, "q": [Q[r][j] for r in range(rows)], "err": errs, "w": [row[:] for row in Wc]})
    return {"Q": Q, "U": U, "steps": steps, "scales": [p["scale"] for p in params]}


def demo_layer(rows: int, d: int, n: int, seed: int) -> dict[str, Any]:
    """A small layer and calibration inputs with correlated, uneven channels (illustrative)."""
    rng = Rng(seed)
    W = [[round_to(0.1 * rng.normal(), "fp16") for _ in range(d)] for _ in range(rows)]
    mix = [[rng.normal() for _ in range(d)] for _ in range(d)]
    gain = [0.25 + 2.0 * rng.uniform() for _ in range(d)]
    X = [[0.0] * n for _ in range(d)]
    for k in range(n):
        z = [rng.normal() for _ in range(d)]
        for i in range(d):
            s = z[i]
            for j in range(d):
                s += 0.3 * mix[i][j] * z[j]
            X[i][k] = round_to(gain[i] * s, "fp16")
    return {"W": W, "X": X}


# ---------------------------------------------------------------------------
# AWQ-style scaling and SmoothQuant migration
# ---------------------------------------------------------------------------


def pow_dyadic(x: float, k: int) -> float:
    """
    x^(k/8) for x > 0 and k in 0..8, as a product of square roots (x^(1/2),
    x^(1/4), x^(1/8)): sqrt is correctly rounded in every language, so the
    result is identical in Python and TypeScript (a general pow is not).
    """
    if k == 0:
        return 1.0
    if k == 8:
        return x
    r2 = math.sqrt(x)
    r4 = math.sqrt(r2)
    r8 = math.sqrt(r4)
    out = 1.0
    if k & 4:
        out *= r2
    if k & 2:
        out *= r4
    if k & 1:
        out *= r8
    return out


def matmul_wx(W: list[list[float]], X: list[list[float]]) -> list[list[float]]:
    rows = len(W)
    d = len(X)
    n = len(X[0])
    Y = [[0.0] * n for _ in range(rows)]
    for r in range(rows):
        for k in range(n):
            s = 0.0
            for i in range(d):
                s += W[r][i] * X[i][k]
            Y[r][k] = s
    return Y


def awq_search(W: list[list[float]], X: list[list[float]], bits: int, group: int) -> dict[str, Any]:
    """
    AWQ-style search (Lin et al., 2023): scale input channel i by
    s_i = mean|x_i|^alpha (normalised by sqrt(max s * min s)), quantise
    W diag(s) per group, divide the activations by s, and keep the alpha
    (here k/8, k = 0..8) with the smallest output error.
    """
    rows = len(W)
    d = len(X)
    n = len(X[0])
    mean_abs = []
    for i in range(d):
        s = 0.0
        for k in range(n):
            s += abs(X[i][k])
        mean_abs.append(s / n)
    Y = matmul_wx(W, X)
    results = []
    for k8 in range(9):
        s = [pow_dyadic(max(m, 1e-8), k8) for m in mean_abs]
        smax = max(s)
        smin = min(s)
        norm = math.sqrt(smax * smin)
        s = [v / norm for v in s]
        Ws = [[W[r][i] * s[i] for i in range(d)] for r in range(rows)]
        Wq = quantise_matrix(Ws, bits, "sym", "group", group)["deq"]
        Wb = [[Wq[r][i] / s[i] for i in range(d)] for r in range(rows)]
        Yq = matmul_wx(Wb, X)
        err = 0.0
        for r in range(rows):
            for kk in range(n):
                dd = Y[r][kk] - Yq[r][kk]
                err += dd * dd
        results.append({"k": k8, "alpha": k8 / 8, "err": err / (rows * n), "scales": s})
    best = results[0]
    for r in results:
        if r["err"] < best["err"]:
            best = r
    return {"results": results, "best": best["k"]}


def smoothquant(W: list[list[float]], X: list[list[float]], k8: int = 4) -> dict[str, Any]:
    """
    SmoothQuant (Xiao et al., 2022): s_j = max|X_j|^alpha / max|W_j|^(1-alpha)
    per input channel j, X' = X / s, W' = W s, so X'W' = XW exactly in real
    arithmetic; alpha = k8/8 (0.5 by default). Then per-tensor symmetric
    INT8 for both.
    """
    rows = len(W)
    d = len(X)
    n = len(X[0])
    xmax = []
    wmax = []
    for j in range(d):
        a = 0.0
        for k in range(n):
            if abs(X[j][k]) > a:
                a = abs(X[j][k])
        b = 0.0
        for r in range(rows):
            if abs(W[r][j]) > b:
                b = abs(W[r][j])
        xmax.append(max(a, 1e-8))
        wmax.append(max(b, 1e-8))
    s = [pow_dyadic(xmax[j], k8) / pow_dyadic(wmax[j], 8 - k8) for j in range(d)]
    Xs = [[X[j][k] / s[j] for k in range(n)] for j in range(d)]
    Ws = [[W[r][j] * s[j] for j in range(d)] for r in range(rows)]
    return {"s": s, "xmax": xmax, "wmax": wmax, "Xs": Xs, "Ws": Ws}


def w8a8_error(W: list[list[float]], X: list[list[float]]) -> dict[str, Any]:
    """Per-tensor symmetric INT8 on weights and activations: the output error."""
    Wq = quantise_matrix(W, 8, "sym", "tensor")["deq"]
    Xq = quantise_matrix(X, 8, "sym", "tensor")["deq"]
    Y = matmul_wx(W, X)
    Yq = matmul_wx(Wq, Xq)
    rows = len(Y)
    n = len(Y[0])
    err = 0.0
    for r in range(rows):
        for k in range(n):
            dd = Y[r][k] - Yq[r][k]
            err += dd * dd
    return {"err": err / (rows * n)}


# ---------------------------------------------------------------------------
# Animation state sequences
# ---------------------------------------------------------------------------


def count_upto(hi: float, fid: str) -> int:
    """How many positive finite values are <= hi (codes are ordered like values)."""
    f = INFO[fid]
    if hi >= f["max"]:
        return f["max_code"]
    return encode(hi, fid, "rdn")


def zoom_steps(fid: str, max_steps: int = 40) -> list[dict[str, Any]]:
    """
    Chapter 1: zoom a linear number line [0, hi] towards zero, halving hi
    (or dividing it by a larger power of two for wide formats) each step,
    from 2^(emax+1) down to the bottom of the subnormals. Every step shows
    the same number of values per binade in half the width: the values
    bunch up towards zero.
    """
    f = INFO[fid]
    top = f["emax"] + 1
    bottom = min(f["emin"], f["emin"] - f["m"] + 2)
    span = top - bottom
    z = max(1, -(-span // max_steps))
    steps = []
    e = top
    while True:
        hi = pow2(e)
        c = count_upto(hi, fid)
        steps.append({
            "exp": e,
            "hi": hi,
            "count": c,
            # the gap between the two largest values in view
            "spacing": decode(c, fid) - decode(c - 1, fid),
            "subnormal": hi <= f["min_normal"],
        })
        if e <= bottom:
            break
        e = max(bottom, e - z)
    return steps


def rounding_inputs(fid: str, n: int, dist: str, seed: int) -> list[float]:
    """
    Inputs between 1 and 2 for the rounding animation: k ulps above 1 plus
    a fraction f of an ulp, with f uniform in [0, 1) ("uniform"), in
    [0, 0.375) ("low": just above a grid point) or exactly 0.5 ("ties").
    """
    f = INFO[fid]
    rng = Rng(seed)
    ulp = pow2(-f["m"])
    per = 1 << f["m"]
    out = []
    for _ in range(n):
        k = int(rng.uniform() * per)
        if dist == "uniform":
            fr = rng.uniform()
        elif dist == "low":
            fr = rng.uniform() * 0.375
        elif dist == "ties":
            fr = 0.5
        else:
            raise ValueError(dist)
        out.append(1.0 + (k + fr) * ulp)
    return out


HIST_BINS = 16


def rounding_steps(fid: str, n: int, dist: str, seed: int) -> list[dict[str, Any]]:
    """
    Chapter 2: each input rounded to nearest-even and stochastically, with
    the errors in ulps, the running mean error of each (the bias) and the
    error histograms (16 bins over -1 .. 1 ulp) as they build up.
    """
    f = INFO[fid]
    xs = rounding_inputs(fid, n, dist, seed)
    rng = Rng(seed ^ 0x5A5A5A5A)
    ulp = pow2(-f["m"])
    h_rne = [0] * HIST_BINS
    h_sr = [0] * HIST_BINS
    tot_rne = 0.0
    tot_sr = 0.0
    out = []
    for i, x in enumerate(xs):
        u = rng.u32()
        nb = neighbours(x, fid)
        rne = round_to(x, fid, "rne")
        sr = round_to(x, fid, "sr", False, u)
        e_rne = (rne - x) / ulp
        e_sr = (sr - x) / ulp
        tot_rne += e_rne
        tot_sr += e_sr
        h_rne[min(HIST_BINS - 1, int((e_rne + 1) * HIST_BINS / 2))] += 1
        h_sr[min(HIST_BINS - 1, int((e_sr + 1) * HIST_BINS / 2))] += 1
        out.append({
            "i": i,
            "x": x,
            "down": nb["down"],
            "up": nb["up"],
            "frac": (x - nb["down"]) / ulp,
            "u": u,
            "rne": rne,
            "sr": sr,
            "err_rne": e_rne,
            "err_sr": e_sr,
            "mean_rne": tot_rne / (i + 1),
            "mean_sr": tot_sr / (i + 1),
            "hist_rne": h_rne[:],
            "hist_sr": h_sr[:],
        })
    return out


def stagnation_steps(fid: str, start: float, delta: float, n: int, seed: int) -> list[dict[str, Any]]:
    """
    Chapter 2: h <- h + delta repeated n times in the format, rounding to
    nearest-even and stochastically. With delta below half an ulp of h,
    nearest-even never moves; stochastic rounding moves by whole ulps with
    the right probability, so on average it follows the exact sum.
    """
    rng = Rng(seed)
    exact = start
    rne = round_to(start, fid)
    sr = rne
    out = [{"t": 0, "exact": exact, "rne": rne, "sr": sr}]
    for t in range(1, n + 1):
        exact += delta
        rne = round_to(rne + delta, fid, "rne")
        sr = round_to(sr + delta, fid, "sr", False, rng.u32())
        out.append({"t": t, "exact": exact, "rne": rne, "sr": sr})
    return out


def sum_steps(fid: str, n: int, dist: str, seed: int, every: int) -> dict[str, Any]:
    """
    Chapter 3: the running totals of n inputs summed four ways: naive in
    the low-precision format, naive in FP32, Kahan in the low-precision
    format, and pairwise in the low-precision format (the pairwise tree of
    the prefix summed so far); the exact prefix sum is a double, exactly.
    One state every ``every`` inputs.
    """
    xs = sum_inputs(n, dist, seed)
    exact = 0.0
    naive = 0.0
    naive32 = 0.0
    ks = 0.0
    kc = 0.0
    pw = PairwiseStack(fid)
    steps = []
    for i, x in enumerate(xs):
        exact += x
        pw.push(x)
        xl = round_to(x, fid)
        naive = add_in(naive, xl, fid)
        naive32 = add_in(naive32, round_to(x, "fp32"), "fp32")
        y = round_to(xl - kc, fid)
        t = add_in(ks, y, fid)
        kc = round_to(round_to(t - ks, fid) - y, fid)
        ks = t
        if (i + 1) % every == 0 or i + 1 == n:
            steps.append({
                "i": i + 1,
                "exact": exact,
                "naive": naive,
                "naive32": naive32,
                "kahan": ks,
                "pairwise": pw.total(),
            })
    return {"inputs": xs, "steps": steps}


def mx_steps(block: list[float], mid: str) -> list[dict[str, Any]]:
    """
    Chapter 4: an MX block quantised step by step: the raw values, the
    absolute maximum found, the shared scale chosen, then one element per
    step (scaled, rounded to the element type, scaled back).
    """
    r = mx_quantise(block, mid)
    k = len(block)
    imax = 0
    for i in range(k):
        if abs(block[i]) > abs(block[imax]):
            imax = i
    steps = [
        {"phase": "raw", "i": -1, "done": 0},
        {"phase": "amax", "i": imax, "done": 0},
        {"phase": "scale", "i": -1, "done": 0},
    ]
    sq = 0.0
    for i in range(k):
        d = block[i] - r["values"][i]
        sq += d * d
        steps.append({"phase": "element", "i": i, "done": i + 1, "sq_err": sq})
    return steps


GRANULARITIES = [("tensor", 0), ("channel", 0), ("group", 32), ("group", 16), ("group", 8)]


def granularity_steps(W: list[list[float]], bits: int, scheme: str) -> list[dict[str, Any]]:
    """Chapter 5: the same matrix quantised per tensor, per channel, then in ever smaller groups."""
    out = []
    for gran, g in GRANULARITIES:
        r = quantise_matrix(W, bits, scheme, gran, g)
        out.append({"gran": gran, "group": g, **r})
    return out


def zeropoint_steps(xs: list[float], bits: int) -> dict[str, Any]:
    """
    Chapter 5: the same values on a symmetric (absmax) grid and on an
    asymmetric (zero-point) grid: the ranges found, the grids laid out,
    then one value snapped to both grids per step.
    """
    ps = int_params(xs, bits, "sym")
    pa = int_params(xs, bits, "asym")
    sq_s = 0.0
    sq_a = 0.0
    steps = [{"phase": "range", "i": -1}, {"phase": "grids", "i": -1}]
    for i, x in enumerate(xs):
        qs = quant_int(x, ps)
        qa = quant_int(x, pa)
        ds = x - dequant_int(qs, ps)
        da = x - dequant_int(qa, pa)
        sq_s += ds * ds
        sq_a += da * da
        steps.append({
            "phase": "snap",
            "i": i,
            "q_sym": qs,
            "q_asym": qa,
            "v_sym": dequant_int(qs, ps),
            "v_asym": dequant_int(qa, pa),
            "mse_sym": sq_s / (i + 1),
            "mse_asym": sq_a / (i + 1),
        })
    levels_s = ps["hi"] - ps["lo"] + 1
    used_s = len({quant_int(x, ps) for x in xs})
    used_a = len({quant_int(x, pa) for x in xs})
    return {"sym": ps, "asym": pa, "steps": steps, "levels_sym": levels_s, "levels_asym": pa["hi"] + 1, "used_sym": used_s, "used_asym": used_a}


PROBE_FORMATS = ("fp32", "bf16", "fp16", "e5m2", "e4m3", "e2m1")


def probe_steps(lo: int = -26, hi: int = 18) -> list[dict[str, Any]]:
    """
    Chapter 4: a probe value x = (4/3) * 2^k swept from k = lo to hi, rounded
    (nearest-even, non-saturating) into each format: the stored value, its
    relative error, and what happened (exact, rounded, subnormal, flushed
    to zero, overflowed to Inf or NaN, or saturated at the largest value).
    """
    out = []
    for k in range(lo, hi + 1):
        x = 4 / 3 * pow2(k)
        row = {}
        for fid in PROBE_FORMATS:
            f = INFO[fid]
            c = encode(x, fid)
            v = decode(c, fid)
            if v != v or v == math.inf:
                status = "overflow"
                rel = -1.0
            elif v == 0:
                status = "underflow"
                rel = 1.0
            else:
                rel = abs(v - x) / x
                if x > f["max"]:
                    status = "saturated"  # beyond the largest value; clamped (FP6, FP4)
                else:
                    status = "subnormal" if v < f["min_normal"] else "normal"
            row[fid] = {"code": c, "value": v if status != "overflow" else -1.0, "rel": rel, "status": status}
        out.append({"k": k, "x": x, "formats": row})
    return out
