/**
 * The live caption of every animation step: a pure function of the model's
 * state for that step. The frame tests build each caption twice, once from
 * the TypeScript model's state and once from the Python reference's state
 * (the fixtures), and require the same text.
 */
import {
  exact,
  fmtDb,
  fmtUlp,
  int,
  minus,
  pct,
  pow2Text,
  sci,
  sqnrDb,
  sup,
  trim,
} from "@/lib/format";

import {
  FORMATS,
  INFO,
  MX_FORMATS,
  PROBE_FORMATS,
  TWO32,
  type FormatId,
  type IntParams,
  type MxId,
  type MxResult,
  type MxStep,
  type ProbeStep,
  type RoundStep,
  type StagStep,
  type SumStep,
  type ZoomStep,
  type ZpStep,
  LLM_INT8_THRESHOLD,
  NF4_OFFSET,
  type DotRun,
  type GptqFrame,
  type Nf4Build,
  type OutlierRun,
  type SmoothStep,
} from "./model";
import { tokenLabel, type TinyStep } from "./tiny";

const name = (f: FormatId) => FORMATS[f].name;

/** Chapter 1: one zoom level of the number line. */
export function zoomCaption(
  fid: FormatId,
  s: ZoomStep,
  first: boolean,
): string {
  const f = INFO[fid];
  const head = first
    ? `The whole positive range, 0 to ${pow2Text(s.hi)}`
    : `Zoomed in to 0 … ${pow2Text(s.hi)} (${exact(s.hi)})`;
  const tail = s.subnormal
    ? `Below ${pow2Text(f.min_normal)} every value is subnormal: the exponent stops shrinking, so the spacing stops shrinking too.`
    : `Every binade holds the same ${int(2 ** f.m)} values, so each halving of the view packs them twice as densely towards zero.`;
  return `${head}: ${int(s.count)} positive ${name(fid)} values, ${exact(s.spacing)} apart at the right edge. ${tail}`;
}

/** Chapter 2: one input rounded both ways. */
export function roundCaption(fid: FormatId, s: RoundStep, n: number): string {
  const u = s.u / TWO32;
  const tie = s.frac === 0.5;
  const rneText = tie
    ? `exactly halfway, so nearest-even takes the neighbour whose last mantissa bit is 0: ${exact(s.rne)}`
    : `nearest-even gives ${exact(s.rne)} (${fmtUlp(s.err_rne)})`;
  const dir = s.sr === s.up && s.up !== s.down ? "up" : "down";
  return (
    `Input ${s.i + 1} of ${n}: x = ${exact(s.x)}, ${s.frac.toFixed(2)} ulp above ${exact(s.down)}; ${rneText}. ` +
    `Stochastic draws u = ${u.toFixed(2)} ${u < s.frac ? "<" : "≥"} ${s.frac.toFixed(2)}, so it rounds ${dir} to ${exact(s.sr)} (${fmtUlp(s.err_sr)}). ` +
    `Mean error so far: nearest-even ${fmtUlp(s.mean_rne, 3)}, stochastic ${fmtUlp(s.mean_sr, 3)}.`
  );
}

/** Chapter 2: the running sum h + delta, both ways. */
export function stagCaption(fid: FormatId, s: StagStep, start: number): string {
  if (s.t === 0)
    return `Start at h = ${trim(start, 6)} in ${name(fid)} and add δ, an eighth of an ulp, again and again.`;
  const stuck =
    s.rne === start
      ? " (stuck: h + δ is nearer h than the next value, every time)"
      : "";
  return `After ${s.t} additions: exact ${trim(s.exact, 7)}; nearest-even ${trim(s.rne, 7)}${stuck}; stochastic ${trim(s.sr, 7)}, off by ${trim(Math.abs(s.sr - s.exact), 3)}.`;
}

function off(v: number, exactV: number): string {
  if (v === exactV) return "exact";
  if (exactV === 0) return `off by ${trim(Math.abs(v), 3)}`;
  return `${pctOrSci(Math.abs(v - exactV) / Math.abs(exactV))} off`;
}

/** Chapter 3: the running totals after i inputs. */
export function sumCaption(fid: FormatId, s: SumStep, n: number): string {
  const nm = name(fid);
  return (
    `After ${int(s.i)} of ${int(n)} inputs the exact total is ${trim(s.exact, 7)}. ` +
    `${nm} naive: ${trim(s.naive, 6)} (${off(s.naive, s.exact)}); ${nm} Kahan: ${trim(s.kahan, 6)} (${off(s.kahan, s.exact)}); ` +
    `${nm} pairwise: ${trim(s.pairwise, 6)} (${off(s.pairwise, s.exact)}); FP32 naive: ${trim(s.naive32, 8)} (${off(s.naive32, s.exact)}).`
  );
}

const STATUS: Record<string, (v: string, rel: number) => string> = {
  normal: (v, rel) =>
    rel === 0 ? `${v} exactly` : `${v}, ${pctOrSci(rel)} off`,
  subnormal: (v, rel) => `${v} (subnormal), ${pctOrSci(rel)} off`,
  underflow: () => "flushed to 0",
  overflow: () => "overflows",
  saturated: (v, rel) => `saturates at ${v}, ${pctOrSci(rel)} off`,
};

/** A relative error: a percentage, or scientific notation when tiny. */
export function pctOrSci(r: number): string {
  if (r >= 0.001) return pct(r, r >= 0.1 ? 0 : 1);
  if (r >= 1e-5) return pct(r, 3);
  return sci(r, 1);
}

/** Chapter 4: the probe (4/3)·2^k in each format. */
export function probeCaption(s: ProbeStep): string {
  const parts = PROBE_FORMATS.map((fid) => {
    const r = s.formats[fid]!;
    const v = trim(r.value, 4);
    return `${name(fid)} ${STATUS[r.status]!(v, r.rel)}`;
  });
  return `x = 4/3 × 2${sup(s.k)} = ${trim(s.x, 5)}. ${parts.join("; ")}.`;
}

/** Chapter 4: one step of an MX block's conversion. */
export function mxCaption(
  mid: MxId,
  block: readonly number[],
  r: MxResult,
  s: MxStep,
): string {
  const f = MX_FORMATS[mid];
  const elem = f.elem === "int8" ? "INT8 (6 fraction bits)" : name(f.elem);
  if (s.phase === "raw")
    return `A block of ${block.length} values. MX stores them as ${block.length} ${f.bits}-bit ${elem} elements plus one shared 8-bit power-of-two scale.`;
  if (s.phase === "amax")
    return `Find the largest magnitude: |V| = ${trim(r.amax, 5)}, element ${s.i}.`;
  if (s.phase === "scale") {
    const lg = r.shared_exp + f.emax_elem;
    return `The largest power of two ≤ ${trim(r.amax, 5)} is ${pow2Text(2 ** lg)}; the element type's largest is ${pow2Text(2 ** f.emax_elem)}. So X = ${pow2Text(2 ** r.shared_exp)} (E8M0 code ${r.scale_code}), and every element is stored as V / X.`;
  }
  const v = block[s.i]!;
  const t = r.scaled[s.i]!;
  const q = r.values[s.i]!;
  const p = q / 2 ** r.shared_exp;
  const clamp =
    f.elem !== "int8" && Math.abs(t) > INFO[f.elem].max
      ? ` (beyond ±${exact(INFO[f.elem].max)}: clamped)`
      : "";
  return `Element ${s.i}: V = ${trim(v, 5)}, V / X = ${trim(t, 5)} → ${elem} ${exact(p)}${clamp} → stored value ${trim(q, 5)}, error ${trim(Math.abs(v - q), 3)}.`;
}

export type GranSummary = {
  gran: string;
  group: number;
  mse: number;
  max_err: number;
  bits_per_weight: number;
  nparams: number;
};

/** Chapter 5: the matrix at one granularity. */
export function granCaption(
  s: GranSummary,
  bits: number,
  scheme: string,
  signal: number,
  size: number,
): string {
  const what =
    s.gran === "tensor"
      ? `Per tensor: one scale for all ${int(size)} weights`
      : s.gran === "channel"
        ? `Per channel: one scale per output row, ${s.nparams} in all`
        : `Per group of ${s.group}: one scale per ${s.group} weights of a row, ${s.nparams} in all`;
  const kind = scheme === "sym" ? "absmax" : "zero-point";
  return `${what} (INT${bits}, ${kind}). Mean squared error ${trim(s.mse, 3)}, SQNR ${fmtDb(sqnrDb(signal, s.mse))}, worst error ${trim(s.max_err, 3)}; ${trim(s.bits_per_weight, 4)} bits per weight with FP16 scales.`;
}

/** Chapter 5: one step of the absmax vs zero-point comparison. */
export function zpCaption(
  xs: readonly number[],
  sym: IntParams,
  asym: IntParams,
  s: ZpStep,
  bits: number,
): string {
  if (s.phase === "range") {
    let mn = Infinity;
    let mx = -Infinity;
    for (const x of xs) {
      if (x < mn) mn = x;
      if (x > mx) mx = x;
    }
    return `${xs.length} activations from ${trim(mn, 4)} to ${trim(mx, 4)}: mostly positive, like a GELU's outputs.`;
  }
  if (s.phase === "grids")
    return `INT${bits} absmax: scale ${trim(sym.scale, 4)}, codes ${minus(String(sym.lo))} … ${sym.hi}, symmetric about 0, so the negative half is nearly empty. Zero-point: scale ${trim(asym.scale, 4)}, zero point ${asym.zero}, codes 0 … ${asym.hi} spread over the actual range.`;
  const x = xs[s.i]!;
  return `Value ${s.i + 1} of ${xs.length}: ${trim(x, 4)} → absmax code ${minus(String(s.q_sym))} (${trim(s.v_sym!, 4)}), zero-point code ${s.q_asym} (${trim(s.v_asym!, 4)}). Mean squared error so far: ${trim(s.mse_sym!, 3)} vs ${trim(s.mse_asym!, 3)}.`;
}

/** Chapter 6: one step of the outlier story. */
export function outlierCaption(
  r: OutlierRun,
  i: number,
  tokens: number,
): string {
  const s = r.steps[i]!;
  const d = r.amax.length;
  const big = r.outliers
    .map((c) => `${c} (${trim(r.amax[c]!, 3)})`)
    .join(" and ");
  let small = 0;
  r.amax.forEach((a, c) => {
    if (!r.outliers.includes(c) && a > small) small = a;
  });
  if (s.phase === "acts")
    return `${d} input channels × ${tokens} tokens. The largest values sit in channels ${big}, in every token; no other channel exceeds ${trim(small, 3)}.`;
  if (s.phase === "tensor")
    return `Per-tensor INT8: one scale, ${trim(r.tensor_scale, 4)}, set by the outliers. The ordinary channels, all below ${LLM_INT8_TEXT}, reach only ${r.normal_levels} of the 255 codes. Output error ${trim(r.err_tensor, 3)}.`;
  if (s.phase === "vector")
    return `Vector-wise INT8: a scale per token and per output channel. No help: every token contains the outliers, so every token's scale is set by them. Output error ${trim(r.err_vector, 3)}.`;
  if (s.phase === "scan") {
    const hit = s.found.includes(s.i);
    return `Channel ${s.i}: largest magnitude ${trim(r.amax[s.i]!, 3)} ${hit ? `> ${LLM_INT8_TEXT}: an outlier channel, kept in 16-bit` : `≤ ${LLM_INT8_TEXT}: INT8`}. Found so far: ${s.found.length === 0 ? "none" : s.found.join(", ")}.`;
  }
  return `LLM.int8(): channels ${s.found.join(" and ")} (${s.found.length} of ${d}) multiply in 16-bit, the other ${d - s.found.length} in vector-wise INT8, and the two parts are added. Output error ${trim(r.err_mixed, 3)}, ${trim(r.err_vector / r.err_mixed, 3)}× smaller than vector-wise INT8 alone.`;
}

const LLM_INT8_TEXT = trim(LLM_INT8_THRESHOLD);

/** Chapter 6: SmoothQuant at one alpha. */
export function smoothCaption(s: SmoothStep, base: number): string {
  const xm = Math.max(...s.xmax);
  const wm = Math.max(...s.wmax);
  const what =
    s.k === 0
      ? "all the difficulty stays in the activations"
      : s.k === 8
        ? "all of it moves into the weights"
        : s.k === 4
          ? "the activations' and weights' channel maxima are equal (SmoothQuant's default)"
          : s.k < 4
            ? "most of it stays in the activations"
            : "most of it moves into the weights";
  return `α = ${trim(s.alpha, 3)}: ${what}. Largest activation channel ${trim(xm, 3)}, largest weight column ${trim(wm, 3)}; per-tensor W8A8 output error ${trim(s.err, 3)} (${trim(base / s.err, 3)}× smaller than without smoothing).`;
}

/** Chapter 7: GPTQ after column j. */
export function gptqCaption(f: GptqFrame, bits: number, d: number): string {
  if (f.col < 0)
    return `A ${f.delta.length} × ${d} weight matrix and its calibration inputs. GPTQ quantises it to INT${bits} one column (input) at a time, left to right.`;
  let moved = 0;
  for (const row of f.delta) for (const v of row) if (v !== 0) moved += 1;
  const tail =
    f.col === d - 1
      ? `Done: layer error ${trim(f.err_gptq, 3)} against ${trim(f.err_rtn, 3)} for plain rounding, ${trim(f.err_rtn / f.err_gptq, 3)}× smaller.`
      : `Layer error so far: GPTQ ${trim(f.err_gptq, 3)}, plain rounding ${trim(f.err_rtn, 3)}.`;
  return `Column ${f.col}: rounded to INT${bits}; its error, weighted by the inverse Hessian, nudges ${moved} weights in the ${d - 1 - f.col} columns to its right. ${tail}`;
}

/** Chapter 8: AWQ at one alpha. */
export function awqCaption(
  r: { k: number; alpha: number; err: number },
  best: number,
  base: number,
  bits: number,
): string {
  const tag = r.k === 0 ? " (no scaling: plain group-wise rounding)" : "";
  const b =
    r.k === best
      ? ` The best α on this layer: ${trim(base / r.err, 3)}× less error than none.`
      : "";
  return `α = ${trim(r.alpha, 3)}${tag}: input channel i is scaled by mean|xᵢ|^α before INT${bits} rounding in groups of 8, and the activations by its inverse. Output error ${trim(r.err, 3)}.${b}`;
}

/** Chapter 8: one step of NF4's construction. */
export function nf4Caption(b: Nf4Build, i: number): string {
  const s = b.steps[i]!;
  if (s.phase === "pdf")
    return `The standard normal distribution: QLoRA assumes a block of weights, divided by its absmax, looks like a scaled sample of it.`;
  if (s.phase === "pos") {
    const q = b.pos[s.n - 1]!;
    return `Positive value ${s.n} of 8: the quantile at p = ${trim(q.p, 5)} is z = ${trim(q.z, 5)}. The probabilities are evenly spaced from ${trim(NF4_OFFSET_TEXT, 7)} down to 0.5, so each bin holds the same share of a normal sample.`;
  }
  if (s.phase === "neg") {
    const q = b.neg[s.n - 1]!;
    return `Negative value ${s.n} of 7: z = ${trim(q.z, 5)} (p = ${trim(q.p, 5)}). One fewer on this side: 16 codes = 8 positive + 7 negative + zero.`;
  }
  if (s.phase === "zero")
    return `Zero is added as a value of its own, so zeros (padding, pruned weights) quantise exactly.`;
  if (s.phase === "normalise")
    return `Divide by the largest, ${trim(b.max, 5)}, so the code book spans −${trim(-b.raw[0]! / b.max, 4)} … 1, the range of absmax-scaled weights.`;
  return `The result matches bitsandbytes' NF4 table to ${sci(b.max_diff, 2)} (its table was built in 32-bit floats).`;
}

const NF4_OFFSET_TEXT = NF4_OFFSET;

/** Chapters 8 and 9: one position of the tiny model. */
export function tinyCaption(s: TinyStep, n: number, label: string): string {
  const ref = tokenLabel(s.top_ref);
  const q = tokenLabel(s.top_q);
  const verdict = s.agree
    ? `both predict “${ref}”`
    : `the reference predicts “${ref}”, the quantised model “${q}”`;
  return `Position ${s.t + 1} of ${n}, after “${tokenLabel(s.token)}”: ${verdict}. RMS logit change ${trim(s.rms, 3)}. ${label}: ${s.agreed} of ${s.t + 1} predictions agree so far.`;
}

/** Chapter 10: the dot product after i + 1 products. */
export function dotCaption(r: DotRun, i: number): string {
  const s = r.steps[i]!;
  const n = r.steps.length;
  const rel = (v: number) => pctOrSci(Math.abs(v - s.ref) / Math.abs(s.ref));
  const head = `Product ${i + 1} of ${n}: exact running sum ${trim(s.ref, 6)}.`;
  const body = `FP32 ${trim(s.fp32, 7)} (${rel(s.fp32)} off), FP16 ${trim(s.fp16, 5)} (${rel(s.fp16)}), INT8 ${trim(s.int8, 5)} (${rel(s.int8)}), MXFP4 ${trim(s.mx, 4)} (${rel(s.mx)}).`;
  const energy = `Energy so far: FP32 ${trim((i + 1) * r.pj.fp32, 4)} pJ, FP16 ${trim((i + 1) * r.pj.fp16, 4)} pJ, INT8 ${trim((i + 1) * r.pj.int8, 4)} pJ.`;
  const last =
    i === n - 1
      ? ` The integer sum ${int(s.int8_acc)} is scaled once by the two INT8 scales; the MXFP4 sum ${trim(s.mx_acc, 4)} by 2${sup(r.scales.mx[0] + r.scales.mx[1])}, an exponent add.`
      : "";
  return `${head} ${body} ${energy}${last}`;
}
