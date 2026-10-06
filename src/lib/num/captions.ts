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
} from "./model";

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
