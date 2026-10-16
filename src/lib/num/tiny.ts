/**
 * The tiny transformer the chapters quantise: the TypeScript port of
 * reference/tiny.py, built from the Transformer Decoder Explainer's model,
 * vendored byte for byte into src/lib/transformer/ (transformer-explainer
 * @0049cf3). This file adds only the hooks: fake-quantised weights, a
 * fake-quantised KV cache, and the measurements.
 *
 * The explainer's weights come from N(0, 0.02); like the reference, this
 * site rescales the same draws (linears to std 1/sqrt(fan-in), the
 * embedding to std 1) so the blocks matter. The weights are still random:
 * the model's text is gibberish, the arithmetic is real.
 *
 * The forward pass uses exp, log, sin, cos and tanh, which JavaScript and
 * Python may round differently in the last place, so the fixtures are
 * matched to a relative 1e-12 (and exactly for every discrete result: top
 * tokens and agreement counts).
 *
 * Since brief 41 the chapters default to the trained model: the same
 * architecture after Training's default run on Alice's Adventures in
 * Wonderland (the Decoder site's weights, vendored as
 * public/data/trained-weights.json), run on 16 eight-character prompts from
 * the corpus's held-out split. Every function takes a `model` ("trained" or
 * "random"); "random" (the default argument) is the original model and its
 * fixtures are unchanged.
 */
import {
  singleHeadAttention,
  type AttentionWeights,
} from "@/lib/transformer/attention";
import {
  positionalEncoding,
  tokenEmbedding,
} from "@/lib/transformer/embeddings";
import { ffn } from "@/lib/transformer/ffn";
import { initModelWeights } from "@/lib/transformer/init";
import { layernormRows } from "@/lib/transformer/layernorm";
import { matmul, transpose } from "@/lib/transformer/matmul";
import {
  unpackWeights,
  type ModelConfig,
  type ModelWeights,
} from "@/lib/transformer/model";
import { addMat } from "@/lib/transformer/tensor";
import { ALPHABET } from "@/lib/transformer/tokenizer";

import tinyData from "@/data/tiny.json";
import trainedData from "@/data/tiny_trained.json";

import {
  INFO,
  MX_BLOCK,
  dequantInt,
  gptq,
  intParams,
  mxQuantise,
  nf4Quantise,
  quantInt,
  quantiseMatrix,
  roundTo,
  type Gran,
} from "./model";

type Mat = number[][];

/** d_model 16, 2 heads, 2 blocks: the explainer's widget model. */
export const TINY_CONFIG: ModelConfig = tinyData.config;
export const PROMPTS: readonly string[] = tinyData.prompts;
export const CALIB: readonly string[] = tinyData.calib;

/** Which weights the chapters quantise. */
export type TinyModel = "trained" | "random";
export const TINY_MODELS: readonly TinyModel[] = ["trained", "random"];

/** The trained model's config (8-character context), prompts and calibration prompts. */
export const TRAINED_CONFIG: ModelConfig = trainedData.config;
export const TRAINED_PROMPTS: readonly string[] = trainedData.prompts;
export const TRAINED_CALIB: readonly string[] = trainedData.calib;

/** The prompts a model is measured on. */
export function promptsOf(m: TinyModel): readonly string[] {
  return m === "trained" ? TRAINED_PROMPTS : PROMPTS;
}

/** The prompts a model's GPTQ calibrates on. */
export function calibOf(m: TinyModel): readonly string[] {
  return m === "trained" ? TRAINED_CALIB : CALIB;
}
export const LINEARS = ["W_q", "W_k", "W_v", "W_o", "W1", "W2"] as const;
export type LinearName = (typeof LINEARS)[number];
export const EMB_GAIN = 1 / 0.02;
export const TOP = 5;

export type WeightCfg = keyof typeof tinyData.weightConfigs;
export type KvCfg = keyof typeof tinyData.kvConfigs;
export const WEIGHT_ORDER = tinyData.weightOrder as WeightCfg[];
export const KV_ORDER = tinyData.kvOrder as KvCfg[];
export const WEIGHT_CONFIGS = tinyData.weightConfigs as Record<
  WeightCfg,
  { label: string; bits: number }
>;
export const KV_CONFIGS = tinyData.kvConfigs as Record<
  KvCfg,
  { label: string; bits: number }
>;

/** Character ids (unknown characters map to 0, as the explainer's encode does). */
export function encodeText(text: string): number[] {
  return Array.from(text, (c) => Math.max(0, ALPHABET.indexOf(c)));
}

/** One token id as a printable label. */
export function tokenLabel(id: number): string {
  const ch = ALPHABET[id] ?? "?";
  if (ch === " ") return "␣";
  if (ch === "\n") return "↵";
  return ch;
}

let cached: ModelWeights | null = null;
let cachedTrained: ModelWeights | null = null;

/** The trained weights' file (public/data/trained-weights.json). */
export const TRAINED_WEIGHTS_URL = "/data/trained-weights.json";
export type TrainedWeightsFile = {
  params: Record<string, Mat | number[]>;
  steps: number;
};
let trainedFile: TrainedWeightsFile | null = null;

/**
 * Hand the trained weights to the library. They are not bundled (110 kB of
 * doubles would ride along with every widget that imports this file): the
 * server imports them (`trained-data.ts`), the browser fetches them.
 */
export function provideTrainedWeights(file: TrainedWeightsFile): void {
  if (!trainedFile) trainedFile = file;
}

/** Whether the trained weights have been provided. */
export function hasTrainedWeights(): boolean {
  return trainedFile !== null;
}

/** Training's trained weights (reference: trained_weights). */
export function trainedModelWeights(): ModelWeights {
  if (!trainedFile)
    throw new Error("tiny: the trained weights are not loaded yet");
  cachedTrained ??= unpackWeights(trainedFile.params, TRAINED_CONFIG);
  return cachedTrained;
}

/** A model's weights. */
export function weightsOf(m: TinyModel): ModelWeights {
  return m === "trained" ? trainedModelWeights() : modelWeights();
}

/** The explainer's weights, rescaled (reference: model_weights). */
export function modelWeights(): ModelWeights {
  if (cached) return cached;
  const w = initModelWeights(TINY_CONFIG);
  const scale = (W: Mat): Mat => {
    const f = 1 / Math.sqrt(W.length) / 0.02;
    return W.map((r) => r.map((v) => v * f));
  };
  cached = {
    tok_emb: w.tok_emb.map((r) => r.map((v) => v * EMB_GAIN)),
    blocks: w.blocks.map((b) => ({
      ...b,
      attn: {
        W_q: scale(b.attn.W_q),
        W_k: scale(b.attn.W_k),
        W_v: scale(b.attn.W_v),
        W_o: scale(b.attn.W_o),
      },
      ffn: { ...b.ffn, W1: scale(b.ffn.W1), W2: scale(b.ffn.W2) },
    })),
    ln_final: w.ln_final,
  };
  return cached;
}

function withLinears(
  w: ModelWeights,
  f: (W: Mat, li: number, name: LinearName) => Mat,
): ModelWeights {
  return {
    ...w,
    blocks: w.blocks.map((b, li) => ({
      ...b,
      attn: {
        W_q: f(b.attn.W_q, li, "W_q"),
        W_k: f(b.attn.W_k, li, "W_k"),
        W_v: f(b.attn.W_v, li, "W_v"),
        W_o: f(b.attn.W_o, li, "W_o"),
      },
      ffn: { ...b.ffn, W1: f(b.ffn.W1, li, "W1"), W2: f(b.ffn.W2, li, "W2") },
    })),
  };
}

// ---------------------------------------------------------------------------
// Forward pass with hooks (the vendored block, with K and V exposed)
// ---------------------------------------------------------------------------

export type KvHook = (layer: number, K: Mat, V: Mat) => [Mat, Mat];
export type Capture = Record<string, Mat[]>;

function push(cap: Capture | undefined, key: string, m: Mat): void {
  if (cap) (cap[key] ??= []).push(m);
}

function attention(
  x: Mat,
  w: AttentionWeights,
  nHeads: number,
  layer: number,
  kv?: KvHook,
  cap?: Capture,
): Mat {
  const dh = x[0]!.length / nHeads;
  const Q = matmul(x, w.W_q);
  let K = matmul(x, w.W_k);
  let V = matmul(x, w.W_v);
  if (kv) [K, V] = kv(layer, K, V);
  push(cap, "K", K);
  push(cap, "V", V);
  const slice = (m: Mat, h: number) =>
    m.map((r) => r.slice(h * dh, (h + 1) * dh));
  const heads: Mat[] = [];
  for (let h = 0; h < nHeads; h++)
    heads.push(
      singleHeadAttention(slice(Q, h), slice(K, h), slice(V, h)).output,
    );
  const concat = x.map((_, i) => heads.flatMap((hd) => hd[i]!));
  push(cap, "in_W_o", concat);
  return matmul(concat, w.W_o);
}

/** Logits [S, V] (reference: forward). Equals the explainer's forwardTyped with no hooks. */
export function forward(
  ids: number[],
  w: ModelWeights,
  kv?: KvHook,
  cap?: Capture,
): Mat {
  const cfg = TINY_CONFIG;
  // prompts are a model's full context: 32 characters (random), 8 (trained)
  let x = addMat(
    tokenEmbedding(ids, w.tok_emb),
    positionalEncoding(ids.length, cfg.d_model),
  );
  w.blocks.forEach((b, li) => {
    const ln1 = layernormRows(x, b.ln1.gamma, b.ln1.beta);
    push(cap, "in_W_qkv", ln1);
    const h = addMat(x, attention(ln1, b.attn, cfg.n_heads, li, kv, cap));
    const ln2 = layernormRows(h, b.ln2.gamma, b.ln2.beta);
    const tr = cap
      ? { pre: [] as Mat, act: [] as Mat, out: [] as Mat }
      : undefined;
    const out = ffn(ln2, b.ffn, tr);
    if (tr) {
      push(cap, "in_W1", ln2);
      push(cap, "in_W2", tr.act);
    }
    x = addMat(h, out);
  });
  const xf = layernormRows(x, w.ln_final.gamma, w.ln_final.beta);
  return matmul(xf, transpose(w.tok_emb));
}

// ---------------------------------------------------------------------------
// Weight quantisers: W is [d_in, d_out]; an output channel is a column
// ---------------------------------------------------------------------------

function fqInt(W: Mat, bits: number, gran: Gran, group = 0): Mat {
  return transpose(quantiseMatrix(transpose(W), bits, "sym", gran, group).deq);
}

function flatCols(W: Mat): number[] {
  const out: number[] = [];
  for (let j = 0; j < W[0]!.length; j++)
    for (let i = 0; i < W.length; i++) out.push(W[i]![j]!);
  return out;
}

function unflatCols(v: number[], rows: number, cols: number): Mat {
  return Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) => v[j * rows + i]!),
  );
}

function fqIntBlocks(W: Mat, bits: number, block = 64): Mat {
  const v = flatCols(W);
  const out: number[] = [];
  for (let b0 = 0; b0 < v.length; b0 += block) {
    const blk = v.slice(b0, b0 + block);
    const p = intParams(blk, bits, "sym");
    for (const x of blk) out.push(dequantInt(quantInt(x, p), p));
  }
  return unflatCols(out, W.length, W[0]!.length);
}

function fqNf4(W: Mat): Mat {
  return unflatCols(
    nf4Quantise(flatCols(W), 64).values,
    W.length,
    W[0]!.length,
  );
}

function fqMx(W: Mat): Mat {
  const v = flatCols(W);
  const out: number[] = [];
  for (let b0 = 0; b0 < v.length; b0 += MX_BLOCK)
    out.push(...mxQuantise(v.slice(b0, b0 + MX_BLOCK), "mxfp4").values);
  return unflatCols(out, W.length, W[0]!.length);
}

function amaxOf(xs: readonly number[]): number {
  let a = 0;
  for (const v of xs) if (Math.abs(v) > a) a = Math.abs(v);
  return a;
}

function fqFp8(W: Mat): Mat {
  return transpose(
    transpose(W).map((row) => {
      const amax = amaxOf(row);
      const s = amax > 0 ? amax / 448 : 1;
      return row.map((v) => roundTo(v / s, "e4m3", "rne", true) * s);
    }),
  );
}

export type Calibration = Record<LinearName, Mat[]>;

/** Each linear layer's inputs over the calibration prompts, as X [d_in, n] (reference: calibration). */
export function calibration(
  w: ModelWeights,
  m: TinyModel = "random",
): Calibration {
  return calibrationFrom(
    calibOf(m).map((p) => {
      const cap: Capture = {};
      forward(encodeText(p), w, undefined, cap);
      return cap;
    }),
  );
}

/** The calibration matrices from the four prompts' captured inputs. */
export function calibrationFrom(caps: Capture[]): Calibration {
  const key: Record<LinearName, string> = {
    W_q: "in_W_qkv",
    W_k: "in_W_qkv",
    W_v: "in_W_qkv",
    W_o: "in_W_o",
    W1: "in_W1",
    W2: "in_W2",
  };
  const out = {} as Calibration;
  for (const name of LINEARS)
    out[name] = Array.from({ length: TINY_CONFIG.n_blocks }, (_, li) =>
      transpose(caps.flatMap((cap) => cap[key[name]]![li]!)),
    );
  return out;
}

/** The weights with every block's linear layers fake-quantised (reference: quantise_weights). */
export function quantiseWeights(
  w: ModelWeights,
  cfg: WeightCfg,
  calib?: Calibration,
): ModelWeights {
  return withLinears(w, (W, li, name) => {
    switch (cfg) {
      case "int8_ch":
        return fqInt(W, 8, "channel");
      case "int4_ch":
        return fqInt(W, 4, "channel");
      case "int4_g8":
        return fqInt(W, 4, "group", 8);
      case "int3_ch":
        return fqInt(W, 3, "channel");
      case "int3_g8":
        return fqInt(W, 3, "group", 8);
      case "int4_b64":
        return fqIntBlocks(W, 4);
      case "nf4":
        return fqNf4(W);
      case "mxfp4":
        return fqMx(W);
      case "fp8_ch":
        return fqFp8(W);
      case "gptq3_ch":
        return transpose(gptq(transpose(W), calib![name][li]!, 3).Q);
    }
  });
}

// ---------------------------------------------------------------------------
// KV-cache quantisers: K and V are [S, D] per layer (a row is one token)
// ---------------------------------------------------------------------------

const intRows = (M: Mat, bits: number, width: number): Mat =>
  quantiseMatrix(M, bits, "sym", "group", width).deq;

const intCols = (M: Mat, bits: number): Mat =>
  transpose(quantiseMatrix(transpose(M), bits, "sym", "channel").deq);

const roundRows = (M: Mat): Mat =>
  M.map((r) => r.map((v) => roundTo(v, "fp16")));

/** FP4 E2M1 with one E4M3 scale per 16 channels of a token (reference: fp4_e4m3_scaled). */
export function fp4E4m3Scaled(M: Mat, block = 16): Mat {
  return M.map((r) => {
    const row: number[] = [];
    for (let c0 = 0; c0 < r.length; c0 += block) {
      const blk = r.slice(c0, c0 + block);
      let s = roundTo(amaxOf(blk) / 6, "e4m3", "rne", true);
      if (s === 0) s = INFO.e4m3.min_sub;
      for (const v of blk) row.push(roundTo(v / s, "e2m1", "rne", true) * s);
    }
    return row;
  });
}

/** FP8 E4M3 with one scale per tensor (reference: fp8_tensor). */
export function fp8Tensor(M: Mat): Mat {
  let amax = 0;
  for (const r of M)
    for (const v of r) if (Math.abs(v) > amax) amax = Math.abs(v);
  const s = amax > 0 ? amax / 448 : 1;
  return M.map((r) => r.map((v) => roundTo(v / s, "e4m3", "rne", true) * s));
}

export function kvHook(cfg: KvCfg): KvHook {
  const dh = TINY_CONFIG.d_model / TINY_CONFIG.n_heads;
  return (_layer, K, V) => {
    switch (cfg) {
      case "fp16":
        return [roundRows(K), roundRows(V)];
      case "fp8":
        return [fp8Tensor(K), fp8Tensor(V)];
      case "int8_tok":
        return [intRows(K, 8, dh), intRows(V, 8, dh)];
      case "fp4_16":
        return [fp4E4m3Scaled(K), fp4E4m3Scaled(V)];
      case "int4_tok":
        return [intRows(K, 4, dh), intRows(V, 4, dh)];
      case "int4_kivi":
        return [intCols(K, 4), intRows(V, 4, dh)];
      case "int2_tok":
        return [intRows(K, 2, dh), intRows(V, 2, dh)];
      case "int2_kivi":
        return [intCols(K, 2), intRows(V, 2, dh)];
    }
  };
}

// ---------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------

export function argmax(row: readonly number[]): number {
  let b = 0;
  for (let i = 1; i < row.length; i++) if (row[i]! > row[b]!) b = i;
  return b;
}

export type TinySummary = {
  drift: number;
  max_change: number;
  agree: number;
  n: number;
};

/** Relative RMS drift, largest change and top-1 agreement over every position (reference: summarise). */
export function summarise(refs: Mat[], qs: Mat[]): TinySummary {
  let sqD = 0;
  let sqR = 0;
  let mx = 0;
  let agree = 0;
  let n = 0;
  refs.forEach((ref, p) =>
    ref.forEach((r, t) => {
      const s = qs[p]![t]!;
      r.forEach((a, j) => {
        const b = s[j]!;
        sqD += (a - b) * (a - b);
        sqR += a * a;
        if (Math.abs(a - b) > mx) mx = Math.abs(a - b);
      });
      if (argmax(r) === argmax(s)) agree += 1;
      n += 1;
    }),
  );
  return { drift: Math.sqrt(sqD / sqR), max_change: mx, agree, n };
}

export type TinyStep = {
  t: number;
  token: number;
  top: number[];
  ref: number[];
  q: number[];
  top_ref: number;
  top_q: number;
  q_top_logit: number;
  rms: number;
  agree: boolean;
  agreed: number;
};

/** One prompt, one position per step (reference: tiny_steps). */
export function tinySteps(refs: Mat, qs: Mat, ids: number[]): TinyStep[] {
  let agreed = 0;
  return refs.map((r, t) => {
    const q = qs[t]!;
    const order = r
      .map((_, j) => j)
      .sort((a, b) => r[b]! - r[a]! || a - b)
      .slice(0, TOP);
    let sq = 0;
    r.forEach((a, j) => {
      sq += (a - q[j]!) * (a - q[j]!);
    });
    const ta = argmax(r);
    const tb = argmax(q);
    if (ta === tb) agreed += 1;
    return {
      t,
      token: ids[t]!,
      top: order,
      ref: order.map((j) => r[j]!),
      q: order.map((j) => q[j]!),
      top_ref: ta,
      top_q: tb,
      q_top_logit: q[tb]!,
      rms: Math.sqrt(sq / r.length),
      agree: ta === tb,
      agreed,
    };
  });
}

export type TinyTarget = "weights" | "kv";

export type TinyRun = {
  summary: TinySummary;
  steps: TinyStep[][];
  refs: Mat[];
  qs: Mat[];
};

// Caches, so a widget can compute a run in small pieces (one forward pass
// per task) and keep the page responsive: see `pendingTasks`.
const refCache = new Map<string, Mat>();
const capCache = new Map<string, Capture>();
const wqCache = new Map<string, ModelWeights>();
const qCache = new Map<string, Mat>();
const runCache = new Map<string, TinyRun>();

/** Forget every cached result (tests). */
export function clearCaches(): void {
  refCache.clear();
  capCache.clear();
  wqCache.clear();
  qCache.clear();
  runCache.clear();
}

/** The reference logits of prompt p. */
export function refLogits(p: number, m: TinyModel = "random"): Mat {
  const key = `${m}|${p}`;
  let r = refCache.get(key);
  if (!r) {
    r = forward(encodeText(promptsOf(m)[p]!), weightsOf(m));
    refCache.set(key, r);
  }
  return r;
}

/** The linear layers' inputs for calibration prompt i. */
export function calibrationCapture(
  i: number,
  m: TinyModel = "random",
): Capture {
  const key = `${m}|${i}`;
  let c = capCache.get(key);
  if (!c) {
    c = {};
    forward(encodeText(calibOf(m)[i]!), weightsOf(m), undefined, c);
    capCache.set(key, c);
  }
  return c;
}

/** The model's weights quantised to one format (cached). */
export function quantisedWeights(
  cfg: WeightCfg,
  m: TinyModel = "random",
): ModelWeights {
  const key = `${m}|${cfg}`;
  let wq = wqCache.get(key);
  if (!wq) {
    const w = weightsOf(m);
    const calib =
      cfg === "gptq3_ch"
        ? calibrationFrom(calibOf(m).map((_, i) => calibrationCapture(i, m)))
        : undefined;
    wq = quantiseWeights(w, cfg, calib);
    wqCache.set(key, wq);
  }
  return wq;
}

/** The quantised model's logits for prompt p. */
export function quantLogits(
  target: TinyTarget,
  cfg: WeightCfg | KvCfg,
  p: number,
  m: TinyModel = "random",
): Mat {
  const key = `${m}|${target}|${cfg}|${p}`;
  let q = qCache.get(key);
  if (!q) {
    const ids = encodeText(promptsOf(m)[p]!);
    q =
      target === "weights"
        ? forward(ids, quantisedWeights(cfg as WeightCfg, m))
        : forward(ids, weightsOf(m), kvHook(cfg as KvCfg));
    qCache.set(key, q);
  }
  return q;
}

/**
 * The pieces of work a run still needs, each a few milliseconds: the
 * reference passes, the calibration passes (GPTQ), the quantised weights,
 * then the quantised passes. Run one per task, then call `run`.
 */
export function pendingTasks(
  target: TinyTarget,
  cfg: WeightCfg | KvCfg,
  m: TinyModel = "random",
): (() => unknown)[] {
  const out: (() => unknown)[] = [];
  const prompts = promptsOf(m);
  prompts.forEach((_, p) => {
    if (!refCache.has(`${m}|${p}`)) out.push(() => refLogits(p, m));
  });
  if (target === "weights") {
    const c = cfg as WeightCfg;
    if (!wqCache.has(`${m}|${c}`)) {
      if (c === "gptq3_ch")
        calibOf(m).forEach((_, i) => {
          if (!capCache.has(`${m}|${i}`))
            out.push(() => calibrationCapture(i, m));
        });
      out.push(() => quantisedWeights(c, m));
    }
  }
  prompts.forEach((_, p) => {
    if (!qCache.has(`${m}|${target}|${cfg}|${p}`))
      out.push(() => quantLogits(target, cfg, p, m));
  });
  return out;
}

/** Every prompt through the reference and the quantised model (reference: run). */
export function run(
  target: TinyTarget,
  cfg: WeightCfg | KvCfg,
  m: TinyModel = "random",
): TinyRun {
  const key = `${m}|${target}|${cfg}`;
  const hit = runCache.get(key);
  if (hit) return hit;
  const prompts = promptsOf(m);
  const ids = prompts.map(encodeText);
  const refs = prompts.map((_, p) => refLogits(p, m));
  const qs = prompts.map((_, p) => quantLogits(target, cfg, p, m));
  const out = {
    summary: summarise(refs, qs),
    steps: refs.map((r, p) => tinySteps(r, qs[p]!, ids[p]!)),
    refs,
    qs,
  };
  runCache.set(key, out);
  return out;
}

/** The gap between the two largest logits, and the logits' RMS (reference: margins). */
export function margins(refs: Mat[]): {
  median_gap: number;
  min_gap: number;
  rms: number;
} {
  const gaps: number[] = [];
  let sq = 0;
  let n = 0;
  for (const ref of refs)
    for (const r of ref) {
      const s = [...r].sort((a, b) => b - a);
      gaps.push(s[0]! - s[1]!);
      for (const v of r) {
        sq += v * v;
        n += 1;
      }
    }
  gaps.sort((a, b) => a - b);
  return {
    median_gap: gaps[Math.floor(gaps.length / 2)]!,
    min_gap: gaps[0]!,
    rms: Math.sqrt(sq / n),
  };
}
