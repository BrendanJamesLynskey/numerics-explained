"""
Writes the site's format data and the parity fixtures from the Python
reference (reference/numerics.py):

  src/data/formats.json               the formats, their constants and sources
  tests/fixtures/num_fixtures.json    reference results the TypeScript port must
                                      reproduce exactly (tests/unit/model.test.ts)

Large results are stored as digests: a SHA-256 over the IEEE bit patterns of
every number (see ``digest``), which the TypeScript tests recompute. Every
code of every format of 16 bits or fewer is covered that way, and every
encode case in every rounding mode.

    python3 scripts/make_fixtures.py          # write both
    python3 scripts/make_fixtures.py --check  # fail if either is out of date (CI)
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))

import numerics as N  # noqa: E402
import tiny as T  # noqa: E402


def hexf(x: float) -> str:
    """The IEEE double bit pattern of x, as 16 hex digits (NaNs canonical)."""
    if x != x:
        return "nan"
    return struct.pack(">d", float(x)).hex()


def flat(obj, out: list[str]) -> None:
    if isinstance(obj, bool):
        out.append("t" if obj else "f")
    elif isinstance(obj, (int, float)):
        out.append(hexf(obj))
    elif isinstance(obj, (list, tuple)):
        out.append("[")
        for o in obj:
            flat(o, out)
        out.append("]")
    elif isinstance(obj, dict):
        out.append("{")
        for k in sorted(obj):
            out.append(k)
            flat(obj[k], out)
        out.append("}")
    elif obj is None:
        out.append("null")
    else:
        out.append(str(obj))


def digest(obj) -> str:
    """SHA-256 of a canonical serialisation (src/lib/num/digest.ts repeats it)."""
    parts: list[str] = []
    flat(obj, parts)
    return hashlib.sha256(",".join(parts).encode()).hexdigest()


ENCODE_N = {"fp32": 20000, "fp16": 8000, "bf16": 8000}
ENCODE_SEED = 2024
U_SEED = 77


def encode_fixture(fid: str) -> dict:
    xs = N.encode_cases(fid, ENCODE_N.get(fid, 3000), ENCODE_SEED)
    rng = N.Rng(U_SEED)
    codes = []
    sample = []
    for x in xs:
        for mode in N.MODES:
            for sat in (False, True):
                u = rng.u32()
                c = N.encode(x, fid, mode, sat, u)
                codes.append(c)
                if len(sample) < 60 and math.isfinite(x):
                    sample.append({"x": x, "mode": mode, "sat": sat, "u": u, "code": c})
    return {"cases": len(xs), "digest": digest(codes), "sample": sample}


def decode_fixture(fid: str) -> dict:
    f = N.INFO[fid]
    if f["bits"] <= 16:
        vals = [N.decode(c, fid) for c in range(1 << f["bits"])]
        return {"all": True, "digest": digest(vals)}
    rng = N.Rng(31)
    codes = [rng.u32() for _ in range(5000)]
    return {"all": False, "codes": codes, "digest": digest([N.decode(c, fid) for c in codes])}


def finite(obj):
    """Fixtures hold finite numbers only (JSON has no NaN or Infinity)."""
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError("non-finite number in a fixture")
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite(v) for v in obj]
    return obj


DEMO_W = (16, 64, 5)
DEMO_ACT = (24, 8)
BLOCKS = [(k, s) for k in ("normal", "outlier", "tiny") for s in (5, 6)]
SUMS = [(fid, dist) for fid in ("fp16", "bf16") for dist in ("uniform", "ones", "normal")]
SUM_N = 8192
SUM_EVERY = 128
ROUNDING = [(fid, dist) for fid in ("e4m3", "e2m1", "fp16") for dist in ("uniform", "low", "ties")]
DEMO_OUTLIER = (8, 16, 32, 13)
DEMO_NORMAL = (4096, 17)
DOT_SEEDS = [(5, 6), (7, 8)]
STAGNATION = [("fp16", 2.0**-13), ("bf16", 2.0**-10), ("e4m3", 2.0**-6)]


def site_data() -> dict:
    return {
        "sources": N.SOURCES,
        "formats": [{**N.FORMATS[f], **N.INFO[f]} for f in N.FORMAT_ORDER],
        "formatOrder": list(N.FORMAT_ORDER),
        "modes": list(N.MODES),
        "modeNames": N.MODE_NAMES,
        "mx": [N.MX_FORMATS[m] for m in N.MX_ORDER],
        "nf4": N.NF4,
        "granularities": [list(g) for g in N.GRANULARITIES],
        "probeFormats": list(N.PROBE_FORMATS),
        "outlierChannels": list(N.OUTLIER_CHANNELS),
        "llmInt8Threshold": N.LLM_INT8_THRESHOLD,
        "nf4Offset": N.NF4_OFFSET,
        "horowitz": [{"id": k, **N.HOROWITZ[k]} for k in N.HOROWITZ_ORDER],
        "dotMac": {k: list(v) for k, v in N.DOT_MAC.items()},
    }


def tiny_data() -> dict:
    return {
        "config": T.CONFIG,
        "prompts": T.PROMPTS,
        "calib": T.CALIB,
        "weightConfigs": T.WEIGHT_CONFIGS,
        "weightOrder": list(T.WEIGHT_ORDER),
        "kvConfigs": T.KV_CONFIGS,
        "kvOrder": list(T.KV_ORDER),
    }


def fixtures() -> dict:
    W = N.demo_weights(*DEMO_W)
    out: dict = {
        "info": {f: N.INFO[f] for f in N.FORMAT_ORDER},
        "decode": {f: decode_fixture(f) for f in N.FORMAT_ORDER},
        "encode": {f: encode_fixture(f) for f in N.FORMAT_ORDER},
        "zoom": {f: N.zoom_steps(f) for f in N.FORMAT_ORDER},
        "demo": {
            "weights": {"args": list(DEMO_W), "digest": digest(W), "row0": W[0]},
            "activations": {"args": list(DEMO_ACT), "values": N.demo_activations(*DEMO_ACT)},
            "blocks": [{"kind": k, "seed": s, "values": N.demo_block(k, s)} for k, s in BLOCKS],
        },
    }
    out["quant"] = []
    for bits in (8, 4, 3):
        for scheme in ("sym", "asym"):
            for gran, g in N.GRANULARITIES:
                r = N.quantise_matrix(W, bits, scheme, gran, g)
                out["quant"].append({
                    "bits": bits, "scheme": scheme, "gran": gran, "group": g,
                    "params": r["params"], "mse": r["mse"], "signal": r["signal"],
                    "max_err": r["max_err"], "bits_per_weight": r["bits_per_weight"],
                    "digest": digest([r["codes"], r["deq"]]),
                })
    out["mx"] = []
    for k, s in BLOCKS:
        blk = N.demo_block(k, s)
        for mid in N.MX_ORDER:
            for mode in ("rne", "sr"):
                out["mx"].append({"kind": k, "seed": s, "mx": mid, "mode": mode, "out": N.mx_quantise(blk, mid, mode, 9)})
    out["mxSteps"] = {mid: N.mx_steps(N.demo_block("outlier", 5), mid) for mid in N.MX_ORDER}
    nf_in = [v for row in W[:2] for v in row]
    out["nf4"] = {"in": nf_in, "out": N.nf4_quantise(nf_in)}
    out["sums"] = []
    for fid, dist in SUMS:
        r = N.sum_steps(fid, SUM_N, dist, 7, SUM_EVERY)
        out["sums"].append({"fmt": fid, "dist": dist, "inputs": digest(r["inputs"]), "steps": r["steps"]})
    out["sumTotals"] = []
    for fid in ("fp16", "bf16", "fp32"):
        xs = N.sum_inputs(3000, "normal", 11)
        out["sumTotals"].append({"fmt": fid, "naive": N.sum_naive(xs, fid), "kahan": N.sum_kahan(xs, fid), "pairwise": N.sum_pairwise(xs, fid)})
    out["rounding"] = [{"fmt": f, "dist": d, "steps": N.rounding_steps(f, 64, d, 3)} for f, d in ROUNDING]
    out["stagnation"] = [{"fmt": f, "delta": dl, "steps": N.stagnation_steps(f, 1.0, dl, 128, 5)} for f, dl in STAGNATION]
    out["granularity"] = []
    for bits in (8, 4, 3):
        for scheme in ("sym", "asym"):
            st = N.granularity_steps(W, bits, scheme)
            out["granularity"].append({
                "bits": bits, "scheme": scheme,
                "steps": [{"gran": s["gran"], "group": s["group"], "mse": s["mse"], "max_err": s["max_err"],
                           "bits_per_weight": s["bits_per_weight"], "nparams": len(s["params"]),
                           "digest": digest([s["codes"], s["deq"], s["params"]])} for s in st],
            })
    acts = N.demo_activations(*DEMO_ACT)
    out["zeropoint"] = [{"bits": b, "out": N.zeropoint_steps(acts, b)} for b in (8, 4, 3)]
    lay = N.demo_layer(8, 16, 64, 21)
    g4 = N.gptq(lay["W"], lay["X"], 4)
    g3 = N.gptq(lay["W"], lay["X"], 3)
    out["gptq"] = {
        "layer": digest([lay["W"], lay["X"]]),
        "bits4": {"Q": g4["Q"], "scales": g4["scales"], "U": digest(g4["U"]), "steps": digest(g4["steps"]),
                  "err": N.layer_error(lay["W"], g4["Q"], lay["X"]), "rtn": N.layer_error(lay["W"], N.rtn(lay["W"], 4), lay["X"])},
        "bits3": {"Q": g3["Q"], "err": N.layer_error(lay["W"], g3["Q"], lay["X"]), "rtn": N.layer_error(lay["W"], N.rtn(lay["W"], 3), lay["X"])},
    }
    aw = N.awq_search(lay["W"], lay["X"], 3, 8)
    out["awq"] = {"best": aw["best"], "errs": [r["err"] for r in aw["results"]], "scales": digest([r["scales"] for r in aw["results"]])}
    sq = N.smoothquant(lay["W"], lay["X"])
    out["smoothquant"] = {
        "s": sq["s"], "Xs": digest(sq["Xs"]), "Ws": digest(sq["Ws"]),
        "before": N.w8a8_error(lay["W"], lay["X"])["err"], "after": N.w8a8_error(sq["Ws"], sq["Xs"])["err"],
    }
    out["probe"] = N.probe_steps()
    ol = N.demo_outlier_layer(*DEMO_OUTLIER)
    out["outlier"] = {"layer": digest([ol["W"], ol["X"]]), "W0": ol["W"][0], "X3": ol["X"][3], "run": N.outlier_steps(ol["W"], ol["X"])}
    out["smooth"] = N.smooth_steps(ol["W"], ol["X"])
    gs = N.gptq_steps(lay["W"], lay["X"], 4)
    out["gptqSteps"] = {"bits4": {"digest": digest(gs["steps"]), "frames": [gs["steps"][i] for i in (0, 1, 8, 16)],
                                  "errs": [[s["err_gptq"], s["err_rtn"]] for s in gs["steps"]]}}
    gs3 = N.gptq_steps(lay["W"], lay["X"], 3)
    out["gptqSteps"]["bits3"] = {"digest": digest(gs3["steps"]), "errs": [[s["err_gptq"], s["err_rtn"]] for s in gs3["steps"]]}
    out["awqMeanAbs"] = aw["mean_abs"]
    aw4 = N.awq_search(lay["W"], lay["X"], 4, 8)
    out["awq4"] = {"best": aw4["best"], "errs": [r["err"] for r in aw4["results"]], "scales": digest([r["scales"] for r in aw4["results"]])}
    out["nf4Build"] = N.nf4_build()
    out["nf4VsInt4"] = N.nf4_vs_int4(N.demo_normal(*DEMO_NORMAL))
    out["dot"] = {f"{s1}-{s2}": N.dot_steps(N.demo_block("normal", s1), N.demo_block("normal", s2)) for s1, s2 in DOT_SEEDS}
    out["pow"] = [[x, k, N.pow_dyadic(x, k)] for x in (0.001, 0.37, 1.0, 5.5, 1234.5) for k in range(9)]
    return finite(out)


def dump(obj: dict, indent: int | None = 1) -> str:
    return json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


# (generator, JSON indent): the fixtures are compact, the site data readable.
def tiny_fixtures() -> dict:
    """The tiny model's runs. The TS port matches these to a relative 1e-12 (shared transcendentals)."""
    w = T.model_weights()
    calib = T.calibration(w)
    out: dict = {"emb0": w["tok_emb"][0], "wq0": w["blocks"][0]["W_q"][0], "w2last": w["blocks"][1]["W2"][-1]}
    runs: dict = {"weights": {}, "kv": {}}
    first = None
    for target, order in (("weights", T.WEIGHT_ORDER), ("kv", T.KV_ORDER)):
        for c in order:
            r = T.run(target, c, w, calib)
            first = first or r["refs"]
            runs[target][c] = {"summary": r["summary"], "steps0": r["steps"][0],
                               "agree": [sum(1 for s in st if s["agree"]) for st in r["steps"]]}
    out["runs"] = runs
    out["ref0"] = first[0]
    out["margins"] = T.margins(first)
    out["calibW2"] = calib["W2"][1][:2]
    return finite(out)


TARGETS = {
    ROOT / "src" / "data" / "formats.json": (site_data, 1),
    ROOT / "src" / "data" / "tiny.json": (tiny_data, 1),
    ROOT / "tests" / "fixtures" / "num_fixtures.json": (fixtures, None),
    ROOT / "tests" / "fixtures" / "tiny_fixtures.json": (tiny_fixtures, None),
}


def close(a, b, rel: float = 1e-12) -> bool:
    """Structural equality, floats to a relative tolerance (ints, strings and bools exactly)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return False
        return abs(a - b) <= rel * max(1.0, abs(a), abs(b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k], rel) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, y, rel) for x, y in zip(a, b))
    return a == b


# Results that pass through exp, log, sin, cos or tanh, whose last bit depends
# on the platform's maths library: --check compares them to a relative 1e-12.
TOLERANT = {"tiny_fixtures.json": None, "num_fixtures.json": "nf4Build"}


def up_to_date(path: Path, text: str) -> bool:
    if not path.exists():
        return False
    old = path.read_text()
    if old == text:
        return True
    if path.name not in TOLERANT:
        return False
    a = json.loads(old)
    b = json.loads(text)
    key = TOLERANT[path.name]
    if key is None:
        return close(a, b)
    rest_a = {k: v for k, v in a.items() if k != key}
    rest_b = {k: v for k, v in b.items() if k != key}
    return rest_a == rest_b and close(a.get(key), b.get(key))


def main() -> int:
    check = "--check" in sys.argv
    stale = []
    for path, (fn, indent) in TARGETS.items():
        text = dump(fn(), indent)
        if check:
            if not up_to_date(path, text):
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)} ({len(text):,} bytes)")
    if stale:
        print("out of date (run python3 scripts/make_fixtures.py):", ", ".join(stale))
        return 1
    if check:
        print("fixtures and site data are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
