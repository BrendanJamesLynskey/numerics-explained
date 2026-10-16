"""
The tiny transformer the chapters quantise: a plain-Python port of the
Transformer Decoder Explainer's TypeScript model (vendored into
``src/lib/transformer/`` from transformer-explainer @0049cf3), with hooks
that fake-quantise its weights or its KV cache.

The model: character level (64 tokens), d_model 16, 2 heads, 2 pre-norm
blocks, d_ff 32, sinusoidal positions, a tied output head, GPT-style
N(0, 0.02) weights from a seeded mulberry32 + Box-Muller generator. Its
weights are random (untrained), so its text is gibberish; the arithmetic is
real, and this file repeats the TypeScript's operation order so the two
agree to rounding (they share exp, log, sin, cos and tanh, which each
language's maths library may round differently in the last place).

The explainer draws every matrix from N(0, 0.02), which leaves its blocks
nearly silent next to the residual stream (quantising them then changes
almost nothing). This site rescales the same draws (``model_weights``):
each linear layer to the LeCun standard deviation 1/sqrt(fan-in) and the
embedding to N(0, 1), PyTorch's nn.Embedding default, so the blocks and the
tied output head carry a trained model's share of the signal. Still random.

Since brief 41 the chapters run, by default, the same model after
Training's default run on Alice's Adventures in Wonderland (the Decoder
site's trained weights, vendored as ``public/data/trained-weights.json``; see
``TRAINED``). The trained model learned its LayerNorm gains and shifts and
the FFN biases, so the forward pass below applies them when the weights
carry them (the random model's are exactly 1 and 0 and are omitted), and it
saw 8-character windows, so it runs 16 eight-character prompts cut from the
corpus's validation split (128 predictions, as the random model's four
32-character prompts give). The random model stays as the contrast.

What the chapters measure (``experiment``): the reference logits of every
position of a few prompts, then the logits with the weights or the KV cache
quantised, and from the two: the relative RMS logit drift, the largest
logit change, and how often the top next-token prediction is unchanged.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numerics as N

CONFIG: dict[str, int] = {
    "seq_len": 32,
    "d_model": 16,
    "n_heads": 2,
    "d_ff": 32,
    "n_blocks": 2,
    "vocab_size": 64,
    "seed": 42,
}

ALPHABET = "abcdefghijklmnopqrstuvwxyz" "0123456789" " .,!?;:'\"()-\n" "+-*/=<>[]{}@#$%"
assert len(ALPHABET) == 64

# Four 32-character prompts the measurements run over (teacher forced: the
# model predicts the next character at every position of each).
PROMPTS = [
    "the quick brown fox jumps over t",
    "to be, or not to be: that is the",
    "numbers: 3.14159, 2.71828 and 1.",
    "quantise the weights, then the k",
]
# Four different prompts that GPTQ calibrates on.
CALIB = [
    "a stitch in time saves nine, and",
    "all that glitters is not gold; s",
    "x = (a + b) * c - d / 2; y = x %",
    "she sells sea shells by the sea ",
]


def encode_text(text: str) -> list[int]:
    return [ALPHABET.index(c) if c in ALPHABET else 0 for c in text]


# ---------------------------------------------------------------------------
# Weights (src/lib/transformer/random.ts and init.ts)
# ---------------------------------------------------------------------------

M32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return (a * b) & M32


def mulberry32(seed: int) -> Callable[[], float]:
    s = [seed & M32]

    def nxt() -> float:
        s[0] = (s[0] + 0x6D2B79F5) & M32
        t = s[0]
        t = _imul(t ^ (t >> 15), t | 1)
        t = (t ^ ((t + _imul(t ^ (t >> 7), t | 61)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296

    return nxt


def init_weights(cfg: dict[str, int]) -> dict[str, Any]:
    rng = mulberry32(cfg["seed"])

    def sample() -> float:
        u1 = max(rng(), 1e-12)
        u2 = rng()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    def normal(r: int, c: int) -> list[list[float]]:
        return [[sample() * 0.02 for _ in range(c)] for _ in range(r)]

    D, F = cfg["d_model"], cfg["d_ff"]
    tok = normal(cfg["vocab_size"], D)
    blocks = []
    for _ in range(cfg["n_blocks"]):
        blocks.append({
            "W_q": normal(D, D),
            "W_k": normal(D, D),
            "W_v": normal(D, D),
            "W_o": normal(D, D),
            "W1": normal(D, F),
            "W2": normal(F, D),
        })
    # LayerNorm gammas are 1, betas and FFN biases 0: omitted (exact no-ops)
    return {"tok_emb": tok, "blocks": blocks}


LINEARS = ("W_q", "W_k", "W_v", "W_o", "W1", "W2")
EMB_GAIN = 1 / 0.02


def model_weights(cfg: dict[str, int] = CONFIG) -> dict[str, Any]:
    """The explainer's weights, rescaled: linears to std 1/sqrt(fan-in), the embedding to std 1."""
    w = init_weights(cfg)
    out: dict[str, Any] = {"tok_emb": [[v * EMB_GAIN for v in r] for r in w["tok_emb"]], "blocks": []}
    for b in w["blocks"]:
        nb = {}
        for name in LINEARS:
            W = b[name]
            f = (1 / math.sqrt(len(W))) / 0.02
            nb[name] = [[v * f for v in r] for r in W]
        out["blocks"].append(nb)
    return out

# ---------------------------------------------------------------------------
# The trained model (brief 41): Training's default run, vendored
# ---------------------------------------------------------------------------

TRAINED_PATH = Path(__file__).resolve().parent.parent / "public" / "data" / "trained-weights.json"
TRAINED_CONFIG: dict[str, int] = dict(CONFIG, seq_len=8)

# 16 eight-character windows of the corpus's validation split (its last 10%,
# held out from training), evenly spaced: window k starts at floor(k * span / 16),
# span = len(val) - 9, as Training's val_batch spaces its windows.
TRAINED_PROMPTS = [
    "ather a ", " the lis", "n a deep", ", at the", " in thei", "o alice.", "d read o", '" said t',
    "ke an ho", "ore.\nif ", 'ng, "tha', "e had th", 'ead!" th', "but now ", " white r", "g sheep-",
]
# GPTQ calibrates on 16 windows of the training split, spaced the same way.
TRAINED_CALIB = [
    "chapter ", "f you ho", "if i lik", " the duc", "alice, a", "you? tel", "ssantly ", "the righ",
    " and she", " have so", " said al", "e unfort", "asn't do", "in all m", "ion how ", "n a cry ",
]

_trained: dict[str, Any] | None = None


def trained_weights() -> dict[str, Any]:
    """Training's trained weights in this file's layout, LayerNorm and biases included."""
    global _trained
    if _trained is None:
        import json

        p = json.loads(TRAINED_PATH.read_text(encoding="utf-8"))["params"]
        _trained = {"tok_emb": p["tok_emb"], "lnf_g": p["ln_final.gamma"], "lnf_b": p["ln_final.beta"],
                    "blocks": []}
        for i in range(CONFIG["n_blocks"]):
            q = f"block_{i}."
            blk = {name: p[q + ("attn." if name.startswith("W_") else "ffn.") + name] for name in LINEARS}
            blk.update(ln1_g=p[q + "ln1.gamma"], ln1_b=p[q + "ln1.beta"], ln2_g=p[q + "ln2.gamma"],
                       ln2_b=p[q + "ln2.beta"], b1=p[q + "ffn.b1"], b2=p[q + "ffn.b2"])
            _trained["blocks"].append(blk)
    return _trained


RANDOM: dict[str, Any] = {"id": "random", "config": CONFIG, "prompts": PROMPTS, "calib": CALIB,
                          "weights": lambda: model_weights()}
TRAINED: dict[str, Any] = {"id": "trained", "config": TRAINED_CONFIG, "prompts": TRAINED_PROMPTS,
                           "calib": TRAINED_CALIB, "weights": trained_weights}

# ---------------------------------------------------------------------------
# Forward pass (src/lib/transformer/*.ts, same operation order)
# ---------------------------------------------------------------------------


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """C = A B with the TS loop order (i, k, j) and its skip of a == 0."""
    n = len(B[0]) if B else 0
    out = []
    for Ai in A:
        Ci = [0.0] * n
        for k, a in enumerate(Ai):
            if a == 0:
                continue
            Bk = B[k]
            for j in range(n):
                Ci[j] = Ci[j] + a * Bk[j]
        out.append(Ci)
    return out


def transpose(A: list[list[float]]) -> list[list[float]]:
    return [list(r) for r in zip(*A)]


def layernorm_rows(x: list[list[float]], g: list[float] | None = None,
                   b: list[float] | None = None) -> list[list[float]]:
    """LayerNorm of each row; gamma and beta default to the random model's 1 and 0."""
    out = []
    for row in x:
        n = len(row)
        mean = 0.0
        for v in row:
            mean += v
        mean /= n
        vs = 0.0
        for v in row:
            d = v - mean
            vs += d * d
        inv = 1 / math.sqrt(vs / n + 1e-5)
        if g is None:
            # gamma = 1, beta = 0: (x - mu) * inv * 1 + 0, as the TS computes it
            out.append([(v - mean) * inv * 1 + 0 for v in row])
        else:
            out.append([(v - mean) * inv * g[j] + b[j] for j, v in enumerate(row)])
    return out


def positional_encoding(seq_len: int, d: int) -> list[list[float]]:
    half = d // 2
    factor = -math.log(10000) / d
    div = [math.exp(2 * i * factor) for i in range(half)]
    out = []
    for pos in range(seq_len):
        row = [0.0] * d
        for i in range(half):
            ang = pos * div[i]
            row[2 * i] = math.sin(ang)
            if 2 * i + 1 < d:
                row[2 * i + 1] = math.cos(ang)
        out.append(row)
    return out


SQRT_2_OVER_PI = math.sqrt(2 / math.pi)


def gelu(x: float) -> float:
    x3 = x * x * x
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x3)
    return 0.5 * x * (1 + math.tanh(inner))


def softmax_causal(row: list[float], i: int) -> list[float]:
    """softmax of row with positions j > i masked (the TS adds -inf, then skips them)."""
    n = len(row)
    mx = -math.inf
    for j in range(n):
        v = row[j] + (0 if j <= i else -math.inf)
        if v > mx:
            mx = v
    out = [0.0] * n
    s = 0.0
    for j in range(n):
        if j <= i:
            e = math.exp(row[j] + 0 - mx)
        else:
            e = 0.0
        out[j] = e
        s += e
    return [v / s for v in out]


KvHook = Callable[[int, list[list[float]], list[list[float]]], tuple[list[list[float]], list[list[float]]]]


def attention(x: list[list[float]], w: dict[str, Any], n_heads: int, layer: int, kv: KvHook | None, cap: dict | None) -> list[list[float]]:
    S = len(x)
    D = len(x[0])
    dh = D // n_heads
    Q = matmul(x, w["W_q"])
    K = matmul(x, w["W_k"])
    V = matmul(x, w["W_v"])
    if kv is not None:
        K, V = kv(layer, K, V)
    if cap is not None:
        cap.setdefault("K", []).append(K)
        cap.setdefault("V", []).append(V)
    scale = 1 / math.sqrt(max(1, dh))
    heads = []
    for h in range(n_heads):
        Qh = [r[h * dh:(h + 1) * dh] for r in Q]
        Kh = [r[h * dh:(h + 1) * dh] for r in K]
        Vh = [r[h * dh:(h + 1) * dh] for r in V]
        raw = matmul(Qh, transpose(Kh))
        scores = [[v * scale for v in r] for r in raw]
        weights = [softmax_causal(scores[i], i) for i in range(S)]
        heads.append(matmul(weights, Vh))
    concat = [[v for h in heads for v in h[i]] for i in range(S)]
    if cap is not None:
        cap.setdefault("in_W_o", []).append(concat)
    return matmul(concat, w["W_o"])


def add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[a[i][j] + b[i][j] for j in range(len(a[i]))] for i in range(len(a))]


def forward(ids: list[int], cfg: dict[str, int], w: dict[str, Any], kv: KvHook | None = None, cap: dict | None = None) -> list[list[float]]:
    """Logits [S, V]. ``kv`` fake-quantises K and V per layer; ``cap`` records each linear layer's input."""
    tok = [w["tok_emb"][i][:] for i in ids]
    pos = positional_encoding(cfg["seq_len"], cfg["d_model"])
    x = add(tok, pos)  # the TS adds the full [seq_len, D] table: prompts are seq_len long
    for li, b in enumerate(w["blocks"]):
        ln1 = layernorm_rows(x, b.get("ln1_g"), b.get("ln1_b"))
        if cap is not None:
            cap.setdefault("in_W_qkv", []).append(ln1)
        h = add(x, attention(ln1, b, cfg["n_heads"], li, kv, cap))
        ln2 = layernorm_rows(h, b.get("ln2_g"), b.get("ln2_b"))
        b1 = b.get("b1") or [0] * len(b["W1"][0])
        pre = [[v + b1[j] for j, v in enumerate(r)] for r in matmul(ln2, b["W1"])]
        act = [[gelu(v) for v in r] for r in pre]
        if cap is not None:
            cap.setdefault("in_W1", []).append(ln2)
            cap.setdefault("in_W2", []).append(act)
        b2 = b.get("b2") or [0] * len(b["W2"][0])
        out = [[v + b2[j] for j, v in enumerate(r)] for r in matmul(act, b["W2"])]
        x = add(h, out)
    xf = layernorm_rows(x, w.get("lnf_g"), w.get("lnf_b"))
    return matmul(xf, transpose(w["tok_emb"]))


# ---------------------------------------------------------------------------
# Quantisers for a weight matrix W [d_in, d_out] (an output channel is a column)
# ---------------------------------------------------------------------------


def fq_int(W: list[list[float]], bits: int, gran: str, group: int = 0) -> list[list[float]]:
    """Symmetric INT, per output channel or per group of inputs within a channel."""
    Wt = transpose(W)
    return transpose(N.quantise_matrix(Wt, bits, "sym", gran, group)["deq"])


def _flat_cols(W: list[list[float]]) -> list[float]:
    """Column-major (each output channel's inputs in turn)."""
    return [W[i][j] for j in range(len(W[0])) for i in range(len(W))]


def _unflat_cols(v: list[float], rows: int, cols: int) -> list[list[float]]:
    return [[v[j * rows + i] for j in range(cols)] for i in range(rows)]


def fq_nf4(W: list[list[float]]) -> list[list[float]]:
    """NF4 in blocks of 64 consecutive weights, column-major (QLoRA's block size)."""
    r = N.nf4_quantise(_flat_cols(W), 64)
    return _unflat_cols(r["values"], len(W), len(W[0]))


def fq_int_blocks(W: list[list[float]], bits: int, block: int = 64) -> list[list[float]]:
    """Symmetric INT in blocks of 64 consecutive weights, column-major (NF4's blocks)."""
    v = _flat_cols(W)
    out: list[float] = []
    for b0 in range(0, len(v), block):
        blk = v[b0:b0 + block]
        p = N.int_params(blk, bits, "sym")
        out.extend(N.dequant_int(N.quant_int(x, p), p) for x in blk)
    return _unflat_cols(out, len(W), len(W[0]))


def fq_mx(W: list[list[float]], mid: str) -> list[list[float]]:
    """An MX format in blocks of 32 consecutive weights, column-major."""
    v = _flat_cols(W)
    out: list[float] = []
    for b0 in range(0, len(v), N.MX_BLOCK):
        out.extend(N.mx_quantise(v[b0:b0 + N.MX_BLOCK], mid)["values"])
    return _unflat_cols(out, len(W), len(W[0]))


def fq_fp8(W: list[list[float]]) -> list[list[float]]:
    """FP8 E4M3 with a per-output-channel scale amax / 448 (a double)."""
    Wt = transpose(W)
    out = []
    for row in Wt:
        amax = 0.0
        for v in row:
            if abs(v) > amax:
                amax = abs(v)
        s = amax / 448 if amax > 0 else 1.0
        out.append([N.round_to(v / s, "e4m3", "rne", True) * s for v in row])
    return transpose(out)


WEIGHT_CONFIGS: dict[str, dict[str, Any]] = {
    "int8_ch": {"label": "INT8 per channel", "bits": 8 + 16 / 16},
    "fp8_ch": {"label": "FP8 E4M3 per channel", "bits": 8 + 16 / 16},
    "int4_ch": {"label": "INT4 per channel", "bits": 4 + 16 / 16},
    "int4_g8": {"label": "INT4 groups of 8", "bits": 4 + 16 / 8},
    "int4_b64": {"label": "INT4 blocks of 64", "bits": 4 + 16 / 64},
    "nf4": {"label": "NF4 blocks of 64", "bits": 4 + 16 / 64},
    "mxfp4": {"label": "MXFP4 blocks of 32", "bits": 4 + 8 / 32},
    "int3_ch": {"label": "INT3 per channel", "bits": 3 + 16 / 16},
    "int3_g8": {"label": "INT3 groups of 8", "bits": 3 + 16 / 8},
    "gptq3_ch": {"label": "GPTQ INT3 per channel", "bits": 3 + 16 / 16},
}
WEIGHT_ORDER = tuple(WEIGHT_CONFIGS)

# "bits" above is for a 16-input channel (W2 has 32 inputs and so a lower
# overhead); scales count as FP16 except MX's 8-bit E8M0.


def quantise_weights(w: dict[str, Any], cfg_id: str, calib: dict | None = None) -> dict[str, Any]:
    """A copy of the weights with every linear layer of every block fake-quantised (embeddings kept)."""
    out = {k: v for k, v in w.items() if k != "blocks"}  # embeddings and final LN kept
    out["blocks"] = []
    for li, b in enumerate(w["blocks"]):
        nb = {k: v for k, v in b.items() if k not in LINEARS}  # LN and biases kept
        for name in LINEARS:
            W = b[name]
            if cfg_id == "int8_ch":
                q = fq_int(W, 8, "channel")
            elif cfg_id == "int4_ch":
                q = fq_int(W, 4, "channel")
            elif cfg_id == "int4_g8":
                q = fq_int(W, 4, "group", 8)
            elif cfg_id == "int3_ch":
                q = fq_int(W, 3, "channel")
            elif cfg_id == "int3_g8":
                q = fq_int(W, 3, "group", 8)
            elif cfg_id == "int4_b64":
                q = fq_int_blocks(W, 4)
            elif cfg_id == "nf4":
                q = fq_nf4(W)
            elif cfg_id == "mxfp4":
                q = fq_mx(W, "mxfp4")
            elif cfg_id == "fp8_ch":
                q = fq_fp8(W)
            elif cfg_id == "gptq3_ch":
                assert calib is not None
                X = calib[name][li]  # [d_in, n]
                q = transpose(N.gptq(transpose(W), X, 3)["Q"])
            else:
                raise ValueError(cfg_id)
            nb[name] = q
        out["blocks"].append(nb)
    return out


def calibration(w: dict[str, Any], model: dict | None = None) -> dict[str, list[list[list[float]]]]:
    """Each linear layer's inputs over the calibration prompts, as X [d_in, n] (GPTQ's layout)."""
    m = model or RANDOM
    caps = []
    for p in m["calib"]:
        cap: dict = {}
        forward(encode_text(p), m["config"], w, None, cap)
        caps.append(cap)
    key = {"W_q": "in_W_qkv", "W_k": "in_W_qkv", "W_v": "in_W_qkv", "W_o": "in_W_o", "W1": "in_W1", "W2": "in_W2"}
    out: dict[str, list] = {}
    for name in LINEARS:
        per_layer = []
        for li in range(CONFIG["n_blocks"]):
            rows = [r for cap in caps for r in cap[key[name]][li]]  # [n, d_in]
            per_layer.append(transpose(rows))
        out[name] = per_layer
    return out


# ---------------------------------------------------------------------------
# KV-cache quantisers: K and V are [S, D] per layer (a row is one token)
# ---------------------------------------------------------------------------


def _int_rows(M: list[list[float]], bits: int, width: int) -> list[list[float]]:
    """Per token: symmetric INT over each head's channels of each row."""
    return N.quantise_matrix(M, bits, "sym", "group", width)["deq"]


def _int_cols(M: list[list[float]], bits: int) -> list[list[float]]:
    """Per channel: symmetric INT over each channel's values across the tokens."""
    return transpose(N.quantise_matrix(transpose(M), bits, "sym", "channel")["deq"])


def _round_rows(M: list[list[float]], fid: str) -> list[list[float]]:
    return [[N.round_to(v, fid) for v in r] for r in M]


def fp4_e4m3_scaled(M: list[list[float]], block: int = 16) -> list[list[float]]:
    """
    FP4 E2M1 with one E4M3 scale per 16 channels of a token (the layout of
    DeepSeek-V4.1-Flash's FP4 KV, arXiv 2609.19969 section 2.4.4; the
    rounding of the scale here is this site's reconstruction): s =
    E4M3(amax / 6), element = E2M1(x / s), saturating.
    """
    out = []
    for r in M:
        row = []
        for c0 in range(0, len(r), block):
            blk = r[c0:c0 + block]
            amax = 0.0
            for v in blk:
                if abs(v) > amax:
                    amax = abs(v)
            s = N.round_to(amax / 6, "e4m3", "rne", True)
            if s == 0:
                s = N.INFO["e4m3"]["min_sub"]
            row.extend(N.round_to(v / s, "e2m1", "rne", True) * s for v in blk)
        out.append(row)
    return out


def fp8_tensor(M: list[list[float]]) -> list[list[float]]:
    """FP8 E4M3 with one scale per layer's K (or V): amax / 448."""
    amax = 0.0
    for r in M:
        for v in r:
            if abs(v) > amax:
                amax = abs(v)
    s = amax / 448 if amax > 0 else 1.0
    return [[N.round_to(v / s, "e4m3", "rne", True) * s for v in r] for r in M]


KV_CONFIGS: dict[str, dict[str, Any]] = {
    "fp16": {"label": "FP16", "bits": 16.0},
    "fp8": {"label": "FP8 E4M3, one scale per tensor", "bits": 8.0},
    "int8_tok": {"label": "INT8 per token", "bits": 8 + 16 / 8},
    "fp4_16": {"label": "FP4 E2M1, E4M3 scale per 16", "bits": 4 + 8 / 16},
    "int4_tok": {"label": "INT4 per token", "bits": 4 + 16 / 8},
    "int4_kivi": {"label": "INT4: K per channel, V per token", "bits": 4 + 16 / 8},
    "int2_tok": {"label": "INT2 per token", "bits": 2 + 16 / 8},
    "int2_kivi": {"label": "INT2: K per channel, V per token", "bits": 2 + 16 / 8},
}
KV_ORDER = tuple(KV_CONFIGS)


def kv_hook(cfg_id: str) -> KvHook:
    dh = CONFIG["d_model"] // CONFIG["n_heads"]

    def hook(layer: int, K: list[list[float]], V: list[list[float]]):
        if cfg_id == "fp16":
            return _round_rows(K, "fp16"), _round_rows(V, "fp16")
        if cfg_id == "fp8":
            return fp8_tensor(K), fp8_tensor(V)
        if cfg_id == "int8_tok":
            return _int_rows(K, 8, dh), _int_rows(V, 8, dh)
        if cfg_id == "fp4_16":
            return fp4_e4m3_scaled(K), fp4_e4m3_scaled(V)
        if cfg_id == "int4_tok":
            return _int_rows(K, 4, dh), _int_rows(V, 4, dh)
        if cfg_id == "int4_kivi":
            return _int_cols(K, 4), _int_rows(V, 4, dh)
        if cfg_id == "int2_tok":
            return _int_rows(K, 2, dh), _int_rows(V, 2, dh)
        if cfg_id == "int2_kivi":
            return _int_cols(K, 2), _int_rows(V, 2, dh)
        raise ValueError(cfg_id)

    return hook


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def argmax(row: list[float]) -> int:
    b = 0
    for i in range(1, len(row)):
        if row[i] > row[b]:
            b = i
    return b


def compare(ref: list[list[float]], q: list[list[float]]) -> dict[str, Any]:
    """Per position: the RMS logit change, the top token of each, and whether they agree."""
    pos = []
    for r, s in zip(ref, q):
        sq = 0.0
        for a, b in zip(r, s):
            sq += (a - b) * (a - b)
        ta = argmax(r)
        tb = argmax(s)
        pos.append({"rms": math.sqrt(sq / len(r)), "top_ref": ta, "top_q": tb, "agree": ta == tb})
    return {"pos": pos}


def summarise(refs: list[list[list[float]]], qs: list[list[list[float]]]) -> dict[str, Any]:
    """Over every position of every prompt: relative RMS drift, largest change, top-1 agreement."""
    sq_d = 0.0
    sq_r = 0.0
    mx = 0.0
    agree = 0
    n = 0
    for ref, q in zip(refs, qs):
        for r, s in zip(ref, q):
            for a, b in zip(r, s):
                sq_d += (a - b) * (a - b)
                sq_r += a * a
                if abs(a - b) > mx:
                    mx = abs(a - b)
            agree += 1 if argmax(r) == argmax(s) else 0
            n += 1
    return {"drift": math.sqrt(sq_d / sq_r), "max_change": mx, "agree": agree, "n": n}


def margins(refs: list[list[list[float]]]) -> dict[str, float]:
    """The gap between the top two logits (how far a prediction is from flipping), and the logits' RMS."""
    gaps = []
    sq = 0.0
    n = 0
    for ref in refs:
        for r in ref:
            s = sorted(r, reverse=True)
            gaps.append(s[0] - s[1])
            for v in r:
                sq += v * v
                n += 1
    gaps.sort()
    return {"median_gap": gaps[len(gaps) // 2], "min_gap": gaps[0], "rms": math.sqrt(sq / n)}


TOP = 5


def tiny_steps(refs: list[list[float]], qs: list[list[float]], ids: list[int]) -> list[dict[str, Any]]:
    """
    Chapters 8 and 9: one prompt, one position per step: the reference's
    top five next tokens with both models' logits for them, each model's top
    token, the RMS logit change and the running agreement.
    """
    out = []
    agree = 0
    for t, (r, q) in enumerate(zip(refs, qs)):
        order = sorted(range(len(r)), key=lambda j: (-r[j], j))[:TOP]
        sq = 0.0
        for a, b in zip(r, q):
            sq += (a - b) * (a - b)
        ta = argmax(r)
        tb = argmax(q)
        agree += 1 if ta == tb else 0
        out.append({
            "t": t,
            "token": ids[t],
            "top": order,
            "ref": [r[j] for j in order],
            "q": [q[j] for j in order],
            "top_ref": ta,
            "top_q": tb,
            "q_top_logit": q[tb],
            "rms": math.sqrt(sq / len(r)),
            "agree": ta == tb,
            "agreed": agree,
        })
    return out


def run(target: str, cfg_id: str, w: dict[str, Any] | None = None, calib: dict | None = None,
        model: dict | None = None) -> dict[str, Any]:
    """Every prompt through the reference and the quantised model: logits, summary and per-prompt steps."""
    m = model or RANDOM
    cfg = m["config"]
    if w is None:
        w = m["weights"]()
    ids = [encode_text(p) for p in m["prompts"]]
    refs = [forward(t, cfg, w) for t in ids]
    if target == "weights":
        if cfg_id == "gptq3_ch" and calib is None:
            calib = calibration(w, m)
        wq = quantise_weights(w, cfg_id, calib)
        qs = [forward(t, cfg, wq) for t in ids]
    else:
        h = kv_hook(cfg_id)
        qs = [forward(t, cfg, w, h) for t in ids]
    return {"summary": summarise(refs, qs), "steps": [tiny_steps(r, q, t) for r, q, t in zip(refs, qs, ids)], "refs": refs, "qs": qs}


def experiment(model: dict | None = None) -> dict[str, Any]:
    m = model or RANDOM
    cfg = m["config"]
    w = m["weights"]()
    ids = [encode_text(p) for p in m["prompts"]]
    refs = [forward(t, cfg, w) for t in ids]
    calib = calibration(w, m)
    weights = {}
    for c in WEIGHT_ORDER:
        wq = quantise_weights(w, c, calib)
        qs = [forward(t, cfg, wq) for t in ids]
        weights[c] = summarise(refs, qs)
    kv = {}
    for c in KV_ORDER:
        h = kv_hook(c)
        qs = [forward(t, cfg, w, h) for t in ids]
        kv[c] = summarise(refs, qs)
    return {"weights": weights, "kv": kv, "margins": margins(refs)}


if __name__ == "__main__":
    import json

    print(json.dumps({"random": experiment(), "trained": experiment(TRAINED)}, indent=1))
