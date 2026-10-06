/**
 * The numerics library: the TypeScript port of reference/numerics.py.
 *
 * Every function repeats the Python reference's double operations in the
 * same order, so the results are bit-identical (tests/unit/model.test.ts
 * checks them against the fixtures scripts/make_fixtures.py writes, with
 * no tolerance). Codes are plain numbers (up to 32 bits), handled with
 * arithmetic rather than JavaScript's signed 32-bit bit operators.
 *
 * Formats: FP32, FP16, BF16 (IEEE 754-2019 and bfloat16), FP8 E4M3 and
 * E5M2 (OCP OFP8 1.0), FP6 E3M2 and E2M3 and FP4 E2M1 (OCP MX 1.0); MX
 * blocks with an E8M0 scale; NF4; integer quantisation; summation in a
 * format; GPTQ, AWQ-style scaling and SmoothQuant; and the state sequences
 * the animations draw.
 */
import data from "@/data/formats.json";

// ---------------------------------------------------------------------------
// Exact powers of two and exponents
// ---------------------------------------------------------------------------

const DV = new DataView(new ArrayBuffer(8));

/** 2**k as a double, exactly (0 below the subnormals, Infinity above). */
export function pow2(k: number): number {
  if (k > 1023) return Infinity;
  if (k < -1074) return 0;
  if (k < -1022) return pow2(k + 52) * pow2(-52);
  DV.setUint32(0, (k + 1023) * 0x100000);
  DV.setUint32(4, 0);
  return DV.getFloat64(0);
}

/** floor(log2(a)) for a finite a > 0, exactly (Python's frexp(a)[1] - 1). */
export function exponentOf(a: number): number {
  DV.setFloat64(0, a);
  const biased = (DV.getUint32(0) >>> 20) & 0x7ff;
  if (biased === 0) return exponentOf(a * pow2(64)) - 64;
  return biased - 1023;
}

/** Python's math.copysign(1, x) < 0: the sign bit (also of -0 and NaNs). */
function signBit(x: number): boolean {
  DV.setFloat64(0, x);
  return DV.getUint32(0) >= 0x80000000;
}

// ---------------------------------------------------------------------------
// Minifloat formats
// ---------------------------------------------------------------------------

export type FormatId =
  | "fp32"
  | "fp16"
  | "bf16"
  | "e5m2"
  | "e4m3"
  | "e3m2"
  | "e2m3"
  | "e2m1";
export type Kind = "ieee" | "fn" | "finite";
export type Mode = "rne" | "rna" | "rtz" | "rup" | "rdn" | "sr";

export const FORMAT_ORDER = data.formatOrder as FormatId[];
export const MODES = data.modes as Mode[];
export const MODE_NAMES = data.modeNames as Record<Mode, string>;

export type FormatDef = {
  id: FormatId;
  name: string;
  long: string;
  e: number;
  m: number;
  bias: number;
  kind: Kind;
  src: string;
  ref: string;
};

export type Info = {
  id: FormatId;
  bits: number;
  e: number;
  m: number;
  bias: number;
  kind: Kind;
  emin: number;
  emax: number;
  max: number;
  min_normal: number;
  min_sub: number;
  eps: number;
  max_code: number;
  inf_code: number;
  nan_code: number;
  sign_bit: number;
  binades: number;
};

export const FORMATS: Record<FormatId, FormatDef> = Object.fromEntries(
  data.formats.map((f) => [
    f.id,
    {
      id: f.id,
      name: f.name,
      long: f.long,
      e: f.e,
      m: f.m,
      bias: f.bias,
      kind: f.kind,
      src: f.src,
      ref: f.ref,
    },
  ]),
) as Record<FormatId, FormatDef>;

export const TWO32 = 4294967296;

/** A format's derived constants (all exact). */
export function info(fid: FormatId): Info {
  const f = FORMATS[fid];
  const { e, m, bias, kind } = f;
  const top = pow2(e) - 1;
  const emin = 1 - bias;
  let emax: number;
  let maxMant: number;
  let infCode: number;
  let nanCode: number;
  if (kind === "ieee") {
    emax = top - 1 - bias;
    maxMant = pow2(m) - 1;
    infCode = top * pow2(m);
    nanCode = top * pow2(m) + pow2(m - 1);
  } else if (kind === "fn") {
    emax = top - bias;
    maxMant = pow2(m) - 2;
    infCode = -1;
    nanCode = top * pow2(m) + (pow2(m) - 1);
  } else {
    emax = top - bias;
    maxMant = pow2(m) - 1;
    infCode = -1;
    nanCode = -1;
  }
  const maxCode = (emax + bias) * pow2(m) + maxMant;
  return {
    id: fid,
    bits: 1 + e + m,
    e,
    m,
    bias,
    kind,
    emin,
    emax,
    max: pow2(emax) * (1 + maxMant / pow2(m)),
    min_normal: pow2(emin),
    min_sub: pow2(emin - m),
    eps: pow2(-m),
    max_code: maxCode,
    inf_code: infCode,
    nan_code: nanCode,
    sign_bit: pow2(e + m),
    binades: emax - (emin - m) + 1,
  };
}

export const INFO: Record<FormatId, Info> = Object.fromEntries(
  FORMAT_ORDER.map((f) => [f, info(f)]),
) as Record<FormatId, Info>;

/** Sign, biased exponent and mantissa fields of a code. */
export function fields(
  code: number,
  fid: FormatId,
): { s: number; E: number; M: number } {
  const f = INFO[fid];
  const s = code >= f.sign_bit ? 1 : 0;
  const rest = code - s * f.sign_bit;
  const mm = pow2(f.m);
  return { s, E: Math.floor(rest / mm), M: rest % mm };
}

/** The value of a code (NaN for NaN codes; signed zeros and infinities). */
export function decode(code: number, fid: FormatId): number {
  const f = INFO[fid];
  const { s, E, M } = fields(code, fid);
  const top = pow2(f.e) - 1;
  if (f.kind === "ieee" && E === top) {
    if (M === 0) return s ? -Infinity : Infinity;
    return NaN;
  }
  if (f.kind === "fn" && E === top && M === pow2(f.m) - 1) return NaN;
  let v: number;
  if (E === 0) v = pow2(f.emin - f.m) * M;
  else v = pow2(E - f.bias - f.m) * (pow2(f.m) + M);
  return s ? -v : v;
}

function roundUp(
  mode: Mode,
  frac: number,
  n: number,
  neg: boolean,
  u: number,
): boolean {
  if (frac === 0) return false;
  switch (mode) {
    case "rne":
      return frac > 0.5 || (frac === 0.5 && n % 2 === 1);
    case "rna":
      return frac >= 0.5;
    case "rtz":
      return false;
    case "rup":
      return !neg;
    case "rdn":
      return neg;
    case "sr":
      return u / TWO32 < frac;
  }
}

function magnitudeCode(v: number, f: Info): number {
  if (v < f.min_normal) return v / f.min_sub;
  const E = exponentOf(v);
  return (E + f.bias) * pow2(f.m) + (v / pow2(E - f.m) - pow2(f.m));
}

function overflowCode(f: Info, mode: Mode, neg: boolean, sat: boolean) {
  if (sat || f.kind === "finite") return f.max_code;
  if (mode === "rtz" || (mode === "rup" && neg) || (mode === "rdn" && !neg))
    return f.max_code;
  return f.kind === "ieee" ? f.inf_code : f.nan_code;
}

/**
 * Round the double x to the format and return its code. For "sr"
 * (stochastic), `u` is a uniform 32-bit integer: the magnitude rounds up
 * when u / 2^32 is below its fraction of an ulp. `sat` saturates on
 * overflow (OCP's SAT mode); formats without Inf always saturate.
 */
export function encode(
  x: number,
  fid: FormatId,
  mode: Mode = "rne",
  sat = false,
  u = 0,
): number {
  const f = INFO[fid];
  const neg = signBit(x);
  const sign = neg ? f.sign_bit : 0;
  if (Number.isNaN(x)) {
    if (f.kind === "finite") return 0;
    return sign + f.nan_code;
  }
  const a = Math.abs(x);
  if (a === Infinity) {
    if (!sat && f.kind === "ieee") return sign + f.inf_code;
    if (!sat && f.kind === "fn") return sign + f.nan_code;
    return sign + f.max_code;
  }
  if (a === 0) return sign;
  const ee = Math.max(exponentOf(a), f.emin);
  const q = pow2(ee - f.m);
  const t = a / q;
  let n = Math.floor(t);
  const frac = t - n;
  if (roundUp(mode, frac, n, neg, u)) n += 1;
  if (n === 0) return sign;
  const v = n * q;
  if (v > f.max) return sign + overflowCode(f, mode, neg, sat);
  return sign + magnitudeCode(v, f);
}

/** x rounded to the format, as a double. */
export function roundTo(
  x: number,
  fid: FormatId,
  mode: Mode = "rne",
  sat = false,
  u = 0,
): number {
  return decode(encode(x, fid, mode, sat, u), fid);
}

/** The representable values either side of a finite x in range. */
export function neighbours(
  x: number,
  fid: FormatId,
): { down: number; up: number } {
  return { down: roundTo(x, fid, "rdn"), up: roundTo(x, fid, "rup") };
}

/** Every finite value >= 0, in order (formats of 16 bits or fewer). */
export function positiveValues(fid: FormatId): number[] {
  const f = INFO[fid];
  if (f.bits > 16) throw new Error("too many values");
  const out: number[] = [];
  for (let c = 0; c <= f.max_code; c++) out.push(decode(c, fid));
  return out;
}

/** Inputs for the encode parity check (reference: encode_cases). */
export function encodeCases(fid: FormatId, n: number, seed: number): number[] {
  const f = INFO[fid];
  const out = [
    0,
    -0,
    Infinity,
    -Infinity,
    f.max,
    -f.max,
    f.min_sub,
    f.min_normal,
    f.max * 3,
  ];
  if (f.bits <= 8) {
    const vals = positiveValues(fid);
    for (let i = 0; i < vals.length - 1; i++) {
      const a = vals[i]!;
      const b = vals[i + 1]!;
      out.push((a + b) / 2);
      out.push(-(a + b) / 2);
      out.push(a + (b - a) / 3);
    }
  }
  const rng = new Rng(seed);
  const lo = f.emin - f.m - 3;
  const span = f.emax + 2 - lo + 1;
  for (let i = 0; i < n; i++) {
    const e = lo + Math.trunc(rng.uniform() * span);
    const v = (1 + rng.uniform()) * pow2(e);
    out.push(rng.u32() & 1 ? -v : v);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Pseudo-random numbers
// ---------------------------------------------------------------------------

export function xorshift32(x0: number): number {
  let x = x0 >>> 0;
  x = (x ^ (x << 13)) >>> 0;
  x = (x ^ (x >>> 17)) >>> 0;
  x = (x ^ (x << 5)) >>> 0;
  return x;
}

/** xorshift32 (Marsaglia, 2003): u32() and uniform() in [0, 1). */
export class Rng {
  x: number;
  constructor(seed: number) {
    this.x = seed >>> 0 || 1;
  }
  u32(): number {
    this.x = xorshift32(this.x);
    return this.x;
  }
  uniform(): number {
    return this.u32() / TWO32;
  }
  /** Approximately N(0, 1): the sum of 12 uniforms, minus 6. */
  normal(): number {
    let s = 0;
    for (let i = 0; i < 12; i++) s += this.uniform();
    return s - 6;
  }
}

// ---------------------------------------------------------------------------
// Integer quantisation
// ---------------------------------------------------------------------------

/** Round to the nearest integer, ties to even (numpy.rint). */
export function rneInt(t: number): number {
  let n = Math.floor(t);
  const frac = t - n;
  if (frac > 0.5 || (frac === 0.5 && n % 2 !== 0)) n += 1;
  return n;
}

export function intRange(
  bits: number,
  signed: boolean,
  restricted = true,
): [number, number] {
  if (signed) {
    const hi = pow2(bits - 1) - 1;
    return [restricted ? -hi : -hi - 1, hi];
  }
  return [0, pow2(bits) - 1];
}

export type Scheme = "sym" | "asym";
export type IntParams = { scale: number; zero: number; lo: number; hi: number };

/** Scale and zero point for a group of values (reference: int_params). */
export function intParams(
  xs: readonly number[],
  bits: number,
  scheme: Scheme,
): IntParams {
  if (scheme === "sym") {
    let amax = 0;
    for (const x of xs) if (Math.abs(x) > amax) amax = Math.abs(x);
    const [lo, hi] = intRange(bits, true);
    const scale = amax > 0 ? amax / hi : 1;
    return { scale, zero: 0, lo, hi };
  }
  let mn = 0;
  let mx = 0;
  for (const x of xs) {
    if (x < mn) mn = x;
    if (x > mx) mx = x;
  }
  const [lo, hi] = intRange(bits, false);
  const scale = mx > mn ? (mx - mn) / hi : 1;
  const zero = Math.min(hi, Math.max(lo, rneInt(-mn / scale)));
  return { scale, zero, lo, hi };
}

export function quantInt(x: number, p: IntParams): number {
  const q = rneInt(x / p.scale) + p.zero;
  return Math.min(p.hi, Math.max(p.lo, q));
}

export function dequantInt(q: number, p: IntParams): number {
  return (q - p.zero) * p.scale;
}

export type Gran = "tensor" | "channel" | "group";

export function groupsOf(
  rows: number,
  cols: number,
  gran: Gran,
  group = 0,
): [number, number, number][] {
  if (gran === "tensor") return [[-1, 0, cols]];
  if (gran === "channel")
    return Array.from({ length: rows }, (_, r) => [r, 0, cols]);
  const out: [number, number, number][] = [];
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c += group)
      out.push([r, c, Math.min(cols, c + group)]);
  return out;
}

export type GroupParams = {
  row: number;
  c0: number;
  c1: number;
  scale: number;
  zero: number;
};
export type QuantMatrix = {
  params: GroupParams[];
  codes: number[][];
  deq: number[][];
  mse: number;
  signal: number;
  max_err: number;
  bits_per_weight: number;
};

/** Quantise a weight matrix (rows = output channels) at a granularity. */
export function quantiseMatrix(
  W: readonly (readonly number[])[],
  bits: number,
  scheme: Scheme,
  gran: Gran,
  group = 0,
): QuantMatrix {
  const rows = W.length;
  const cols = W[0]!.length;
  const gs = groupsOf(rows, cols, gran, group);
  const deq = W.map((r) => r.map(() => 0));
  const codes = W.map((r) => r.map(() => 0));
  const params: GroupParams[] = [];
  for (const [r, c0, c1] of gs) {
    const rr = r < 0 ? Array.from({ length: rows }, (_, i) => i) : [r];
    const vals: number[] = [];
    for (const i of rr) for (let j = c0; j < c1; j++) vals.push(W[i]![j]!);
    const p = intParams(vals, bits, scheme);
    params.push({ row: r, c0, c1, scale: p.scale, zero: p.zero });
    for (const i of rr)
      for (let j = c0; j < c1; j++) {
        const q = quantInt(W[i]![j]!, p);
        codes[i]![j] = q;
        deq[i]![j] = dequantInt(q, p);
      }
  }
  let err = 0;
  let sig = 0;
  let amaxErr = 0;
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) {
      const d = W[i]![j]! - deq[i]![j]!;
      err += d * d;
      sig += W[i]![j]! * W[i]![j]!;
      if (Math.abs(d) > amaxErr) amaxErr = Math.abs(d);
    }
  const n = rows * cols;
  const size =
    gran === "channel" ? cols : gran === "tensor" ? rows * cols : group;
  const overhead = 16 + (scheme === "asym" ? bits : 0);
  return {
    params,
    codes,
    deq,
    mse: err / n,
    signal: sig / n,
    max_err: amaxErr,
    bits_per_weight: bits + overhead / size,
  };
}

// ---------------------------------------------------------------------------
// MX block formats (OCP MX v1.0)
// ---------------------------------------------------------------------------

export type MxId =
  | "mxfp8_e4m3"
  | "mxfp8_e5m2"
  | "mxfp6_e3m2"
  | "mxfp6_e2m3"
  | "mxfp4"
  | "mxint8";
export type MxDef = {
  id: MxId;
  name: string;
  elem: FormatId | "int8";
  emax_elem: number;
  bits: number;
};
export const MX_FORMATS = Object.fromEntries(
  data.mx.map((m) => [m.id, m]),
) as Record<MxId, MxDef>;
export const MX_ORDER = data.mx.map((m) => m.id) as MxId[];
export const MX_BLOCK = 32;

/** E8M0 scale: 2^(code - 127); 0xFF is NaN. */
export function e8m0Decode(code: number): number {
  return code === 255 ? NaN : pow2(code - 127);
}

/** One scaled input t = V/X quantised to the element type (saturating). */
export function mxElement(
  t: number,
  mid: MxId,
  mode: Mode = "rne",
  u = 0,
): { code: number; value: number } {
  const f = MX_FORMATS[mid];
  if (f.elem === "int8") {
    const n = t * 64;
    let q: number;
    if (mode === "sr") {
      const fl = Math.floor(n);
      q = fl + (u / TWO32 < n - fl ? 1 : 0);
    } else q = rneInt(n);
    q = Math.min(127, Math.max(-127, q));
    return { code: q & 0xff, value: q / 64 };
  }
  const c = encode(t, f.elem, mode, true, u);
  return { code: c, value: decode(c, f.elem) };
}

export type MxResult = {
  scale_code: number;
  shared_exp: number;
  amax: number;
  scaled: number[];
  codes: number[];
  values: number[];
};

/** MX spec section 6.3 (reference: mx_quantise), for finite blocks. */
export function mxQuantise(
  block: readonly number[],
  mid: MxId,
  mode: Mode = "rne",
  seed = 1,
): MxResult {
  const f = MX_FORMATS[mid];
  let amax = 0;
  for (const v of block) {
    if (!Number.isFinite(v)) throw new Error("non-finite input");
    if (Math.abs(v) > amax) amax = Math.abs(v);
  }
  const se =
    amax === 0
      ? -127
      : Math.max(-127, Math.min(127, exponentOf(amax) - f.emax_elem));
  const X = pow2(se);
  const rng = new Rng(seed);
  const codes: number[] = [];
  const values: number[] = [];
  const scaled: number[] = [];
  for (const v of block) {
    const t = v / X;
    const u = mode === "sr" ? rng.u32() : 0;
    const el = mxElement(t, mid, mode, u);
    scaled.push(t);
    codes.push(el.code);
    values.push(X * el.value);
  }
  return { scale_code: se + 127, shared_exp: se, amax, scaled, codes, values };
}

// ---------------------------------------------------------------------------
// NF4
// ---------------------------------------------------------------------------

export const NF4: readonly number[] = data.nf4;
export const NF4_MID: readonly number[] = Array.from(
  { length: 15 },
  (_, i) => (NF4[i]! + NF4[i + 1]!) / 2,
);

/** Nearest NF4 code to t in [-1, 1] (ties to the lower code). */
export function nf4Code(t: number): number {
  let c = 0;
  for (let i = 0; i < 15; i++) if (t > NF4_MID[i]!) c = i + 1;
  return c;
}

/** Blockwise absmax NF4 (block 64). */
export function nf4Quantise(
  xs: readonly number[],
  block = 64,
): { codes: number[]; values: number[]; absmax: number[] } {
  const codes: number[] = [];
  const values: number[] = [];
  const absmax: number[] = [];
  for (let b0 = 0; b0 < xs.length; b0 += block) {
    const blk = xs.slice(b0, b0 + block);
    let am = 0;
    for (const x of blk) if (Math.abs(x) > am) am = Math.abs(x);
    const s = am > 0 ? am : 1;
    absmax.push(am);
    for (const x of blk) {
      const c = nf4Code(x / s);
      codes.push(c);
      values.push(NF4[c]! * s);
    }
  }
  return { codes, values, absmax };
}

// ---------------------------------------------------------------------------
// Summation in a format
// ---------------------------------------------------------------------------

/** a + b in the format: the exact double sum rounded once. */
export function addIn(a: number, b: number, fid: FormatId): number {
  return roundTo(a + b, fid);
}

export function sumNaive(xs: readonly number[], fid: FormatId): number {
  let s = 0;
  for (const x of xs) s = addIn(s, roundTo(x, fid), fid);
  return s;
}

/** Kahan (1965) compensated summation, every operation in the format. */
export function sumKahan(xs: readonly number[], fid: FormatId): number {
  let s = 0;
  let c = 0;
  for (const x of xs) {
    const y = roundTo(roundTo(x, fid) - c, fid);
    const t = addIn(s, y, fid);
    c = roundTo(roundTo(t - s, fid) - y, fid);
    s = t;
  }
  return s;
}

/**
 * Recursive pairwise summation: the left part is the largest power of two
 * below n (a power-of-two n splits in half).
 */
export function sumPairwise(
  xs: readonly number[],
  fid: FormatId,
  lo = 0,
  hi = -1,
): number {
  if (hi < 0) hi = xs.length;
  const n = hi - lo;
  if (n === 0) return 0;
  if (n === 1) return roundTo(xs[lo]!, fid);
  let p = 1;
  while (p * 2 < n) p *= 2;
  return addIn(
    sumPairwise(xs, fid, lo, lo + p),
    sumPairwise(xs, fid, lo + p, hi),
    fid,
  );
}

/**
 * sumPairwise of a growing prefix, incrementally: a binary counter of
 * complete power-of-two blocks, folded from the right when read.
 */
export class PairwiseStack {
  blocks: [number, number][] = [];
  constructor(private fid: FormatId) {}
  push(x: number): void {
    this.blocks.push([1, roundTo(x, this.fid)]);
    while (
      this.blocks.length >= 2 &&
      this.blocks[this.blocks.length - 1]![0] ===
        this.blocks[this.blocks.length - 2]![0]
    ) {
      const r = this.blocks.pop()!;
      const l = this.blocks.pop()!;
      this.blocks.push([l[0] * 2, addIn(l[1], r[1], this.fid)]);
    }
  }
  total(): number {
    if (this.blocks.length === 0) return 0;
    let acc = this.blocks[this.blocks.length - 1]![1];
    for (let i = this.blocks.length - 2; i >= 0; i--)
      acc = addIn(this.blocks[i]![1], acc, this.fid);
    return acc;
  }
}

// ---------------------------------------------------------------------------
// Data sets the chapters use
// ---------------------------------------------------------------------------

export type SumDist = "uniform" | "ones" | "normal";

/** n FP16 values: uniform in [0, 1), all ones, or approximately normal. */
export function sumInputs(n: number, dist: SumDist, seed: number): number[] {
  const rng = new Rng(seed);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    let v: number;
    if (dist === "ones") v = 1;
    else if (dist === "uniform") v = rng.uniform();
    else v = rng.normal();
    out.push(roundTo(v, "fp16", "rtz"));
  }
  return out;
}

/** The illustrative weight matrix (reference: demo_weights). */
export function demoWeights(
  rows: number,
  cols: number,
  seed: number,
): number[][] {
  const rng = new Rng(seed);
  const rowScale = Array.from(
    { length: rows },
    () => 0.02 * (0.5 + 1.5 * rng.uniform()),
  );
  const colScale = Array.from({ length: cols }, () => 1);
  for (let k = 0; k < Math.max(1, Math.floor(cols / 16)); k++) {
    const j = Math.trunc(rng.uniform() * cols);
    colScale[j] = 3 + 3 * rng.uniform();
  }
  const W: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++)
      row.push(roundTo(rowScale[i]! * colScale[j]! * rng.normal(), "fp16"));
    W.push(row);
  }
  return W;
}

/** Skewed, mostly positive values like a GELU's outputs (illustrative). */
export function demoActivations(n: number, seed: number): number[] {
  const rng = new Rng(seed);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const z = rng.normal();
    const v = z > -0.75 ? z * (0.5 + (0.5 * z) / 3) : -0.17 * rng.uniform();
    out.push(roundTo(v, "fp16"));
  }
  return out;
}

export type BlockKind = "normal" | "outlier" | "tiny";

/** A block of 32 values for the MX animation. */
export function demoBlock(kind: BlockKind, seed: number, k = MX_BLOCK) {
  const rng = new Rng(seed);
  let vals = Array.from({ length: k }, () => roundTo(rng.normal(), "fp16"));
  if (kind === "outlier") vals[Math.trunc(rng.uniform() * k)] = 24;
  else if (kind === "tiny")
    vals = vals.map((v) => roundTo(v * pow2(-12), "fp16"));
  return vals;
}

// ---------------------------------------------------------------------------
// Linear algebra and GPTQ
// ---------------------------------------------------------------------------

type Mat = number[][];

const zeros = (r: number, c: number): Mat =>
  Array.from({ length: r }, () => new Array<number>(c).fill(0));

/** Lower-triangular L with A = L L^T. */
export function cholesky(A: Mat): Mat {
  const n = A.length;
  const L = zeros(n, n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = A[i]![j]!;
      for (let k = 0; k < j; k++) s -= L[i]![k]! * L[j]![k]!;
      if (i === j) {
        if (s <= 0) throw new Error("not positive definite");
        L[i]![i] = Math.sqrt(s);
      } else L[i]![j] = s / L[j]![j]!;
    }
  return L;
}

/** A^-1 from its Cholesky factor. */
export function spdInverse(A: Mat): Mat {
  const n = A.length;
  const L = cholesky(A);
  const X = zeros(n, n);
  for (let c = 0; c < n; c++) {
    const y = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) {
      let s = i === c ? 1 : 0;
      for (let k = 0; k < i; k++) s -= L[i]![k]! * y[k]!;
      y[i] = s / L[i]![i]!;
    }
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i]!;
      for (let k = i + 1; k < n; k++) s -= L[k]![i]! * X[k]![c]!;
      X[i]![c] = s / L[i]![i]!;
    }
  }
  return X;
}

/** H = 2 X X^T + damp * mean(diag) * I. */
export function hessian(X: Mat, damp: number): Mat {
  const d = X.length;
  const n = X[0]!.length;
  const H = zeros(d, d);
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += X[i]![k]! * X[j]![k]!;
      H[i]![j] = 2 * s;
    }
  let tr = 0;
  for (let i = 0; i < d; i++) tr += H[i]![i]!;
  const lam = (damp * tr) / d;
  for (let i = 0; i < d; i++) H[i]![i]! += lam;
  return H;
}

/** ||W X - Wq X||_F^2. */
export function layerError(W: Mat, Wq: Mat, X: Mat): number {
  const rows = W.length;
  const d = X.length;
  const n = X[0]!.length;
  let err = 0;
  for (let r = 0; r < rows; r++)
    for (let k = 0; k < n; k++) {
      let a = 0;
      let b = 0;
      for (let i = 0; i < d; i++) {
        a += W[r]![i]! * X[i]![k]!;
        b += Wq[r]![i]! * X[i]![k]!;
      }
      err += (a - b) * (a - b);
    }
  return err;
}

/** Round to nearest, per output channel, symmetric. */
export function rtn(W: Mat, bits: number): Mat {
  return quantiseMatrix(W, bits, "sym", "channel").deq;
}

export type GptqStep = { col: number; q: number[]; err: number[]; w: Mat };

/** GPTQ (Frantar et al., 2022, Algorithm 1), one column at a time. */
export function gptq(
  W: Mat,
  X: Mat,
  bits: number,
  damp = 0.01,
): { Q: Mat; U: Mat; steps: GptqStep[]; scales: number[] } {
  const rows = W.length;
  const d = W[0]!.length;
  const H = hessian(X, damp);
  const Hinv = spdInverse(H);
  const L = cholesky(Hinv);
  const U = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) => L[j]![i]!),
  );
  const params = W.map((row) => intParams(row, bits, "sym"));
  const Wc = W.map((r) => r.slice());
  const Q = zeros(rows, d);
  const steps: GptqStep[] = [];
  for (let j = 0; j < d; j++) {
    const errs: number[] = [];
    for (let r = 0; r < rows; r++) {
      const w = Wc[r]![j]!;
      const q = dequantInt(quantInt(w, params[r]!), params[r]!);
      Q[r]![j] = q;
      const e = (w - q) / U[j]![j]!;
      errs.push(e);
      for (let k = j + 1; k < d; k++) Wc[r]![k]! -= e * U[j]![k]!;
    }
    steps.push({
      col: j,
      q: Q.map((row) => row[j]!),
      err: errs,
      w: Wc.map((r) => r.slice()),
    });
  }
  return { Q, U, steps, scales: params.map((p) => p.scale) };
}

/** A small layer and calibration inputs (reference: demo_layer). */
export function demoLayer(
  rows: number,
  d: number,
  n: number,
  seed: number,
): { W: Mat; X: Mat } {
  const rng = new Rng(seed);
  const W = Array.from({ length: rows }, () =>
    Array.from({ length: d }, () => roundTo(0.1 * rng.normal(), "fp16")),
  );
  const mix = Array.from({ length: d }, () =>
    Array.from({ length: d }, () => rng.normal()),
  );
  const gain = Array.from({ length: d }, () => 0.25 + 2 * rng.uniform());
  const X = zeros(d, n);
  for (let k = 0; k < n; k++) {
    const z = Array.from({ length: d }, () => rng.normal());
    for (let i = 0; i < d; i++) {
      let s = z[i]!;
      for (let j = 0; j < d; j++) s += 0.3 * mix[i]![j]! * z[j]!;
      X[i]![k] = roundTo(gain[i]! * s, "fp16");
    }
  }
  return { W, X };
}

// ---------------------------------------------------------------------------
// AWQ-style scaling and SmoothQuant
// ---------------------------------------------------------------------------

/** x^(k/8) as a product of square roots (identical in every language). */
export function powDyadic(x: number, k: number): number {
  if (k === 0) return 1;
  if (k === 8) return x;
  const r2 = Math.sqrt(x);
  const r4 = Math.sqrt(r2);
  const r8 = Math.sqrt(r4);
  let out = 1;
  if (k & 4) out *= r2;
  if (k & 2) out *= r4;
  if (k & 1) out *= r8;
  return out;
}

export function matmulWX(W: Mat, X: Mat): Mat {
  const rows = W.length;
  const d = X.length;
  const n = X[0]!.length;
  const Y = zeros(rows, n);
  for (let r = 0; r < rows; r++)
    for (let k = 0; k < n; k++) {
      let s = 0;
      for (let i = 0; i < d; i++) s += W[r]![i]! * X[i]![k]!;
      Y[r]![k] = s;
    }
  return Y;
}

/** AWQ-style search over alpha = k/8 (reference: awq_search). */
export function awqSearch(
  W: Mat,
  X: Mat,
  bits: number,
  group: number,
): {
  results: { k: number; alpha: number; err: number; scales: number[] }[];
  best: number;
  meanAbs: number[];
} {
  const rows = W.length;
  const d = X.length;
  const n = X[0]!.length;
  const meanAbs: number[] = [];
  for (let i = 0; i < d; i++) {
    let s = 0;
    for (let k = 0; k < n; k++) s += Math.abs(X[i]![k]!);
    meanAbs.push(s / n);
  }
  const Y = matmulWX(W, X);
  const results = [];
  for (let k8 = 0; k8 < 9; k8++) {
    let s = meanAbs.map((m) => powDyadic(Math.max(m, 1e-8), k8));
    const smax = Math.max(...s);
    const smin = Math.min(...s);
    const norm = Math.sqrt(smax * smin);
    s = s.map((v) => v / norm);
    const Ws = W.map((row) => row.map((w, i) => w * s[i]!));
    const Wq = quantiseMatrix(Ws, bits, "sym", "group", group).deq;
    const Wb = Wq.map((row) => row.map((w, i) => w / s[i]!));
    const Yq = matmulWX(Wb, X);
    let err = 0;
    for (let r = 0; r < rows; r++)
      for (let kk = 0; kk < n; kk++) {
        const dd = Y[r]![kk]! - Yq[r]![kk]!;
        err += dd * dd;
      }
    results.push({ k: k8, alpha: k8 / 8, err: err / (rows * n), scales: s });
  }
  let best = results[0]!;
  for (const r of results) if (r.err < best.err) best = r;
  return { results, best: best.k, meanAbs };
}

/** SmoothQuant migration with alpha = k8/8 (reference: smoothquant). */
export function smoothquant(
  W: Mat,
  X: Mat,
  k8 = 4,
): { s: number[]; xmax: number[]; wmax: number[]; Xs: Mat; Ws: Mat } {
  const rows = W.length;
  const d = X.length;
  const n = X[0]!.length;
  const xmax: number[] = [];
  const wmax: number[] = [];
  for (let j = 0; j < d; j++) {
    let a = 0;
    for (let k = 0; k < n; k++)
      if (Math.abs(X[j]![k]!) > a) a = Math.abs(X[j]![k]!);
    let b = 0;
    for (let r = 0; r < rows; r++)
      if (Math.abs(W[r]![j]!) > b) b = Math.abs(W[r]![j]!);
    xmax.push(Math.max(a, 1e-8));
    wmax.push(Math.max(b, 1e-8));
  }
  const s = Array.from(
    { length: d },
    (_, j) => powDyadic(xmax[j]!, k8) / powDyadic(wmax[j]!, 8 - k8),
  );
  const Xs = X.map((row, j) => row.map((x) => x / s[j]!));
  const Ws = W.map((row) => row.map((w, j) => w * s[j]!));
  return { s, xmax, wmax, Xs, Ws };
}

/** Per-tensor symmetric INT8 on weights and activations: the output error. */
export function w8a8Error(W: Mat, X: Mat): { err: number } {
  const Wq = quantiseMatrix(W, 8, "sym", "tensor").deq;
  const Xq = quantiseMatrix(X, 8, "sym", "tensor").deq;
  const Y = matmulWX(W, X);
  const Yq = matmulWX(Wq, Xq);
  let err = 0;
  for (let r = 0; r < Y.length; r++)
    for (let k = 0; k < Y[0]!.length; k++) {
      const dd = Y[r]![k]! - Yq[r]![k]!;
      err += dd * dd;
    }
  return { err: err / (Y.length * Y[0]!.length) };
}

// ---------------------------------------------------------------------------
// Animation state sequences
// ---------------------------------------------------------------------------

/** How many positive finite values are <= hi. */
export function countUpto(hi: number, fid: FormatId): number {
  const f = INFO[fid];
  if (hi >= f.max) return f.max_code;
  return encode(hi, fid, "rdn");
}

export type ZoomStep = {
  exp: number;
  hi: number;
  count: number;
  spacing: number;
  subnormal: boolean;
};

/** Chapter 1: zoom a linear number line [0, hi] towards zero. */
export function zoomSteps(fid: FormatId, maxSteps = 40): ZoomStep[] {
  const f = INFO[fid];
  const top = f.emax + 1;
  const bottom = Math.min(f.emin, f.emin - f.m + 2);
  const span = top - bottom;
  const z = Math.max(1, Math.ceil(span / maxSteps));
  const steps: ZoomStep[] = [];
  let e = top;
  for (;;) {
    const hi = pow2(e);
    const c = countUpto(hi, fid);
    steps.push({
      exp: e,
      hi,
      count: c,
      spacing: decode(c, fid) - decode(c - 1, fid),
      subnormal: hi <= f.min_normal,
    });
    if (e <= bottom) break;
    e = Math.max(bottom, e - z);
  }
  return steps;
}

export type RoundDist = "uniform" | "low" | "ties";

/** Inputs between 1 and 2 for the rounding animation. */
export function roundingInputs(
  fid: FormatId,
  n: number,
  dist: RoundDist,
  seed: number,
): number[] {
  const f = INFO[fid];
  const rng = new Rng(seed);
  const ulp = pow2(-f.m);
  const per = pow2(f.m);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const k = Math.trunc(rng.uniform() * per);
    let fr: number;
    if (dist === "uniform") fr = rng.uniform();
    else if (dist === "low") fr = rng.uniform() * 0.375;
    else fr = 0.5;
    out.push(1 + (k + fr) * ulp);
  }
  return out;
}

export const HIST_BINS = 16;

export type RoundStep = {
  i: number;
  x: number;
  down: number;
  up: number;
  frac: number;
  u: number;
  rne: number;
  sr: number;
  err_rne: number;
  err_sr: number;
  mean_rne: number;
  mean_sr: number;
  hist_rne: number[];
  hist_sr: number[];
};

/** Chapter 2: nearest-even vs stochastic rounding, input by input. */
export function roundingSteps(
  fid: FormatId,
  n: number,
  dist: RoundDist,
  seed: number,
): RoundStep[] {
  const f = INFO[fid];
  const xs = roundingInputs(fid, n, dist, seed);
  const rng = new Rng((seed ^ 0x5a5a5a5a) >>> 0);
  const ulp = pow2(-f.m);
  const hRne = new Array<number>(HIST_BINS).fill(0);
  const hSr = new Array<number>(HIST_BINS).fill(0);
  let totRne = 0;
  let totSr = 0;
  const out: RoundStep[] = [];
  xs.forEach((x, i) => {
    const u = rng.u32();
    const nb = neighbours(x, fid);
    const rne = roundTo(x, fid, "rne");
    const sr = roundTo(x, fid, "sr", false, u);
    const eRne = (rne - x) / ulp;
    const eSr = (sr - x) / ulp;
    totRne += eRne;
    totSr += eSr;
    hRne[Math.min(HIST_BINS - 1, Math.trunc(((eRne + 1) * HIST_BINS) / 2))]! +=
      1;
    hSr[Math.min(HIST_BINS - 1, Math.trunc(((eSr + 1) * HIST_BINS) / 2))]! += 1;
    out.push({
      i,
      x,
      down: nb.down,
      up: nb.up,
      frac: (x - nb.down) / ulp,
      u,
      rne,
      sr,
      err_rne: eRne,
      err_sr: eSr,
      mean_rne: totRne / (i + 1),
      mean_sr: totSr / (i + 1),
      hist_rne: hRne.slice(),
      hist_sr: hSr.slice(),
    });
  });
  return out;
}

export type StagStep = { t: number; exact: number; rne: number; sr: number };

/** Chapter 2: h <- h + delta, n times, nearest-even vs stochastic. */
export function stagnationSteps(
  fid: FormatId,
  start: number,
  delta: number,
  n: number,
  seed: number,
): StagStep[] {
  const rng = new Rng(seed);
  let exact = start;
  let rne = roundTo(start, fid);
  let sr = rne;
  const out: StagStep[] = [{ t: 0, exact, rne, sr }];
  for (let t = 1; t <= n; t++) {
    exact += delta;
    rne = roundTo(rne + delta, fid, "rne");
    sr = roundTo(sr + delta, fid, "sr", false, rng.u32());
    out.push({ t, exact, rne, sr });
  }
  return out;
}

export type SumStep = {
  i: number;
  exact: number;
  naive: number;
  naive32: number;
  kahan: number;
  pairwise: number;
};

/** Chapter 3: running totals summed four ways (reference: sum_steps). */
export function sumSteps(
  fid: FormatId,
  n: number,
  dist: SumDist,
  seed: number,
  every: number,
): { inputs: number[]; steps: SumStep[] } {
  const xs = sumInputs(n, dist, seed);
  let exact = 0;
  let naive = 0;
  let naive32 = 0;
  let ks = 0;
  let kc = 0;
  const pw = new PairwiseStack(fid);
  const steps: SumStep[] = [];
  xs.forEach((x, i) => {
    exact += x;
    pw.push(x);
    const xl = roundTo(x, fid);
    naive = addIn(naive, xl, fid);
    naive32 = addIn(naive32, roundTo(x, "fp32"), "fp32");
    const y = roundTo(xl - kc, fid);
    const t = addIn(ks, y, fid);
    kc = roundTo(roundTo(t - ks, fid) - y, fid);
    ks = t;
    if ((i + 1) % every === 0 || i + 1 === n)
      steps.push({
        i: i + 1,
        exact,
        naive,
        naive32,
        kahan: ks,
        pairwise: pw.total(),
      });
  });
  return { inputs: xs, steps };
}

export type MxStep = {
  phase: "raw" | "amax" | "scale" | "element";
  i: number;
  done: number;
  sq_err?: number;
};

/** Chapter 4: an MX block quantised step by step. */
export function mxSteps(block: readonly number[], mid: MxId): MxStep[] {
  const r = mxQuantise(block, mid);
  const k = block.length;
  let imax = 0;
  for (let i = 0; i < k; i++)
    if (Math.abs(block[i]!) > Math.abs(block[imax]!)) imax = i;
  const steps: MxStep[] = [
    { phase: "raw", i: -1, done: 0 },
    { phase: "amax", i: imax, done: 0 },
    { phase: "scale", i: -1, done: 0 },
  ];
  let sq = 0;
  for (let i = 0; i < k; i++) {
    const d = block[i]! - r.values[i]!;
    sq += d * d;
    steps.push({ phase: "element", i, done: i + 1, sq_err: sq });
  }
  return steps;
}

export const GRANULARITIES = data.granularities as [Gran, number][];

export type GranStep = QuantMatrix & { gran: Gran; group: number };

/** Chapter 5: per tensor, per channel, then smaller and smaller groups. */
export function granularitySteps(
  W: Mat,
  bits: number,
  scheme: Scheme,
): GranStep[] {
  return GRANULARITIES.map(([gran, g]) => ({
    gran,
    group: g,
    ...quantiseMatrix(W, bits, scheme, gran, g),
  }));
}

export type ZpStep = {
  phase: "range" | "grids" | "snap";
  i: number;
  q_sym?: number;
  q_asym?: number;
  v_sym?: number;
  v_asym?: number;
  mse_sym?: number;
  mse_asym?: number;
};

/** Chapter 5: the same values on a symmetric and an asymmetric grid. */
export function zeropointSteps(
  xs: readonly number[],
  bits: number,
): {
  sym: IntParams;
  asym: IntParams;
  steps: ZpStep[];
  levels_sym: number;
  levels_asym: number;
  used_sym: number;
  used_asym: number;
} {
  const ps = intParams(xs, bits, "sym");
  const pa = intParams(xs, bits, "asym");
  let sqS = 0;
  let sqA = 0;
  const steps: ZpStep[] = [
    { phase: "range", i: -1 },
    { phase: "grids", i: -1 },
  ];
  xs.forEach((x, i) => {
    const qs = quantInt(x, ps);
    const qa = quantInt(x, pa);
    const ds = x - dequantInt(qs, ps);
    const da = x - dequantInt(qa, pa);
    sqS += ds * ds;
    sqA += da * da;
    steps.push({
      phase: "snap",
      i,
      q_sym: qs,
      q_asym: qa,
      v_sym: dequantInt(qs, ps),
      v_asym: dequantInt(qa, pa),
      mse_sym: sqS / (i + 1),
      mse_asym: sqA / (i + 1),
    });
  });
  return {
    sym: ps,
    asym: pa,
    steps,
    levels_sym: ps.hi - ps.lo + 1,
    levels_asym: pa.hi + 1,
    used_sym: new Set(xs.map((x) => quantInt(x, ps))).size,
    used_asym: new Set(xs.map((x) => quantInt(x, pa))).size,
  };
}

export const PROBE_FORMATS = data.probeFormats as FormatId[];

export type ProbeStatus =
  | "normal"
  | "subnormal"
  | "underflow"
  | "overflow"
  | "saturated";
export type ProbeStep = {
  k: number;
  x: number;
  formats: Record<
    string,
    { code: number; value: number; rel: number; status: ProbeStatus }
  >;
};

/** Chapter 4: a probe (4/3)·2^k rounded into each format (reference: probe_steps). */
export function probeSteps(lo = -26, hi = 18): ProbeStep[] {
  const out: ProbeStep[] = [];
  for (let k = lo; k <= hi; k++) {
    const x = (4 / 3) * pow2(k);
    const row: ProbeStep["formats"] = {};
    for (const fid of PROBE_FORMATS) {
      const f = INFO[fid];
      const c = encode(x, fid);
      const v = decode(c, fid);
      let status: ProbeStatus;
      let rel: number;
      if (Number.isNaN(v) || v === Infinity) {
        status = "overflow";
        rel = -1;
      } else if (v === 0) {
        status = "underflow";
        rel = 1;
      } else {
        rel = Math.abs(v - x) / x;
        if (x > f.max) status = "saturated";
        else status = v < f.min_normal ? "subnormal" : "normal";
      }
      row[fid] = {
        code: c,
        value: status !== "overflow" ? v : -1,
        rel,
        status,
      };
    }
    out.push({ k, x, formats: row });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Chapter 6: activation outliers, LLM.int8() and SmoothQuant
// ---------------------------------------------------------------------------

export const OUTLIER_CHANNELS: readonly number[] = data.outlierChannels;
export const LLM_INT8_THRESHOLD: number = data.llmInt8Threshold;

/** A layer whose input has systematic outlier channels (reference: demo_outlier_layer). */
export function demoOutlierLayer(
  rows: number,
  d: number,
  n: number,
  seed: number,
  gain = 20,
): { W: Mat; X: Mat } {
  const rng = new Rng(seed);
  const W = Array.from({ length: rows }, () =>
    Array.from({ length: d }, () => roundTo(0.1 * rng.normal(), "fp16")),
  );
  const X = zeros(d, n);
  for (let i = 0; i < d; i++) {
    const g = 0.4 + 0.6 * rng.uniform();
    const sign = rng.u32() % 2 !== 0 ? 1 : -1;
    for (let k = 0; k < n; k++) {
      const z = rng.normal();
      X[i]![k] = OUTLIER_CHANNELS.includes(i)
        ? roundTo(sign * gain * (1 + 0.15 * z), "fp16")
        : roundTo(g * z, "fp16");
    }
  }
  return { W, X };
}

/** Mean squared difference of two outputs. */
export function outError(Y: Mat, Yq: Mat): number {
  const rows = Y.length;
  const n = Y[0]!.length;
  let err = 0;
  for (let r = 0; r < rows; r++)
    for (let k = 0; k < n; k++) {
      const dd = Y[r]![k]! - Yq[r]![k]!;
      err += dd * dd;
    }
  return err / (rows * n);
}

/** W X over some input channels in vector-wise INT8 (reference: int8_vectorwise). */
export function int8Vectorwise(W: Mat, X: Mat, channels: number[]): Mat {
  const rows = W.length;
  const n = X[0]!.length;
  const pw = W.map((row) =>
    intParams(
      channels.map((i) => row[i]!),
      8,
      "sym",
    ),
  );
  const px = Array.from({ length: n }, (_, k) =>
    intParams(
      channels.map((i) => X[i]![k]!),
      8,
      "sym",
    ),
  );
  const qw = W.map((row, r) => channels.map((i) => quantInt(row[i]!, pw[r]!)));
  const qx = Array.from({ length: n }, (_, k) =>
    channels.map((i) => quantInt(X[i]![k]!, px[k]!)),
  );
  const Y = zeros(rows, n);
  for (let r = 0; r < rows; r++)
    for (let k = 0; k < n; k++) {
      let acc = 0;
      for (let c = 0; c < channels.length; c++) acc += qw[r]![c]! * qx[k]![c]!;
      Y[r]![k] = acc * (pw[r]!.scale * px[k]!.scale);
    }
  return Y;
}

/** LLM.int8()'s mixed-precision decomposition (reference: llm_int8). */
export function llmInt8(
  W: Mat,
  X: Mat,
  threshold = LLM_INT8_THRESHOLD,
): { outliers: number[]; Y: Mat } {
  const d = X.length;
  const n = X[0]!.length;
  const rows = W.length;
  const outl: number[] = [];
  for (let i = 0; i < d; i++)
    for (let k = 0; k < n; k++)
      if (Math.abs(X[i]![k]!) > threshold) {
        outl.push(i);
        break;
      }
  const normal = Array.from({ length: d }, (_, i) => i).filter(
    (i) => !outl.includes(i),
  );
  const Y8 = int8Vectorwise(W, X, normal);
  const Y = zeros(rows, n);
  for (let r = 0; r < rows; r++)
    for (let k = 0; k < n; k++) {
      let s = 0;
      for (const i of outl) s += W[r]![i]! * X[i]![k]!;
      Y[r]![k] = s + Y8[r]![k]!;
    }
  return { outliers: outl, Y };
}

/** Per-tensor INT8 of X: its scale and the codes some channels reach. */
export function tensorLevels(
  X: Mat,
  channels: number[],
): { scale: number; top: number; levels: number } {
  const p = quantiseMatrix(X, 8, "sym", "tensor").params[0]!;
  let mx = 0;
  for (const i of channels)
    for (const v of X[i]!) if (Math.abs(v) > mx) mx = Math.abs(v);
  const top = rneInt(mx / p.scale);
  return { scale: p.scale, top, levels: 2 * top + 1 };
}

export type OutlierStep = {
  phase: "acts" | "tensor" | "vector" | "scan" | "mixed";
  i: number;
  found: number[];
};

export type OutlierRun = {
  amax: number[];
  outliers: number[];
  tensor_scale: number;
  normal_levels: number;
  err_tensor: number;
  err_vector: number;
  err_mixed: number;
  signal: number;
  steps: OutlierStep[];
};

/** Chapter 6: per tensor, vector-wise, the outlier scan and the decomposition. */
export function outlierSteps(W: Mat, X: Mat): OutlierRun {
  const d = X.length;
  const Y = matmulWX(W, X);
  const amax = X.map((row) => {
    let a = 0;
    for (const v of row) if (Math.abs(v) > a) a = Math.abs(v);
    return a;
  });
  const r = llmInt8(W, X);
  const all = Array.from({ length: d }, (_, i) => i);
  const normal = all.filter((i) => !r.outliers.includes(i));
  const tl = tensorLevels(X, normal);
  const steps: OutlierStep[] = [
    { phase: "acts", i: -1, found: [] },
    { phase: "tensor", i: -1, found: [] },
    { phase: "vector", i: -1, found: [] },
  ];
  const found: number[] = [];
  for (let i = 0; i < d; i++) {
    if (amax[i]! > LLM_INT8_THRESHOLD) found.push(i);
    steps.push({ phase: "scan", i, found: found.slice() });
  }
  steps.push({ phase: "mixed", i: -1, found: found.slice() });
  return {
    amax,
    outliers: r.outliers,
    tensor_scale: tl.scale,
    normal_levels: tl.levels,
    err_tensor: w8a8Error(W, X).err,
    err_vector: outError(Y, int8Vectorwise(W, X, all)),
    err_mixed: outError(Y, r.Y),
    signal: outError(
      Y,
      Y.map((row) => row.map(() => 0)),
    ),
    steps,
  };
}

export type SmoothStep = {
  k: number;
  alpha: number;
  s: number[];
  xmax: number[];
  wmax: number[];
  err: number;
};

/** Chapter 6: SmoothQuant's alpha swept from 0 to 1 (reference: smooth_steps). */
export function smoothSteps(W: Mat, X: Mat): SmoothStep[] {
  const Y = matmulWX(W, X);
  const out: SmoothStep[] = [];
  for (let k8 = 0; k8 < 9; k8++) {
    const sq = smoothquant(W, X, k8);
    const xm = sq.Xs.map((row) => {
      let a = 0;
      for (const v of row) if (Math.abs(v) > a) a = Math.abs(v);
      return a;
    });
    const wm = sq.s.map((_, j) => {
      let b = 0;
      for (const row of sq.Ws) if (Math.abs(row[j]!) > b) b = Math.abs(row[j]!);
      return b;
    });
    const Wq = quantiseMatrix(sq.Ws, 8, "sym", "tensor").deq;
    const Xq = quantiseMatrix(sq.Xs, 8, "sym", "tensor").deq;
    out.push({
      k: k8,
      alpha: k8 / 8,
      s: sq.s,
      xmax: xm,
      wmax: wm,
      err: outError(Y, matmulWX(Wq, Xq)),
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Chapter 7: GPTQ column by column
// ---------------------------------------------------------------------------

export type GptqFrame = {
  col: number;
  q: number[];
  err: number[];
  delta: Mat;
  err_gptq: number;
  err_rtn: number;
};

/** Chapter 7: each column quantised and its error pushed right (reference: gptq_steps). */
export function gptqSteps(
  W: Mat,
  X: Mat,
  bits: number,
): { Q: Mat; rtn: Mat; scales: number[]; steps: GptqFrame[] } {
  const g = gptq(W, X, bits);
  const Wr = rtn(W, bits);
  const rows = W.length;
  const d = W[0]!.length;
  let prev = W.map((r) => r.slice());
  const steps: GptqFrame[] = [
    {
      col: -1,
      q: [],
      err: [],
      delta: zeros(rows, d),
      err_gptq: 0,
      err_rtn: 0,
    },
  ];
  g.steps.forEach((st, j) => {
    const P = Array.from({ length: rows }, (_, r) =>
      Array.from({ length: d }, (_, c) =>
        c <= j ? g.Q[r]![c]! : st.w[r]![c]!,
      ),
    );
    const R = Array.from({ length: rows }, (_, r) =>
      Array.from({ length: d }, (_, c) => (c <= j ? Wr[r]![c]! : W[r]![c]!)),
    );
    const before = prev;
    const delta = Array.from({ length: rows }, (_, r) =>
      Array.from({ length: d }, (_, c) => st.w[r]![c]! - before[r]![c]!),
    );
    prev = st.w;
    steps.push({
      col: j,
      q: st.q,
      err: st.err,
      delta,
      err_gptq: layerError(W, P, X),
      err_rtn: layerError(W, R, X),
    });
  });
  return { Q: g.Q, rtn: Wr, scales: g.scales, steps };
}

// ---------------------------------------------------------------------------
// Chapter 8: deriving NF4, and NF4 against INT4 on normal data
// ---------------------------------------------------------------------------

export const NF4_OFFSET: number = data.nf4Offset;

/** The standard normal CDF by its power series (reference: norm_cdf). */
export function normCdf(x: number): number {
  if (x < 0) return 1 - normCdf(-x);
  let term = x;
  let s = x;
  let n = 0;
  for (;;) {
    term = (term * x * x) / (2 * n + 3);
    n += 1;
    if (s + term === s) break;
    s += term;
  }
  return 0.5 + (Math.exp((-x * x) / 2) / Math.sqrt(2 * Math.PI)) * s;
}

/** The normal quantile function by bisection (reference: norm_ppf). */
export function normPpf(p: number): number {
  let lo = -8;
  let hi = 8;
  for (;;) {
    const mid = (lo + hi) / 2;
    if (mid === lo || mid === hi) return mid;
    if (normCdf(mid) < p) lo = mid;
    else hi = mid;
  }
}

export type Nf4Step = {
  phase: "pdf" | "pos" | "neg" | "zero" | "normalise" | "compare";
  n: number;
};

export type Nf4Build = {
  pos: { p: number; z: number }[];
  neg: { p: number; z: number }[];
  raw: number[];
  max: number;
  values: number[];
  max_diff: number;
  steps: Nf4Step[];
};

/** bitsandbytes' create_normal_map in doubles, step by step (reference: nf4_build). */
export function nf4Build(): Nf4Build {
  const pos = Array.from({ length: 8 }, (_, i) => {
    const p = NF4_OFFSET + ((0.5 - NF4_OFFSET) * i) / 8;
    return { p, z: normPpf(p) };
  });
  const neg = Array.from({ length: 7 }, (_, i) => {
    const p = NF4_OFFSET + ((0.5 - NF4_OFFSET) * i) / 7;
    return { p: 1 - p, z: -normPpf(p) };
  });
  const raw = [...pos.map((q) => q.z), 0, ...neg.map((q) => q.z)].sort(
    (a, b) => a - b,
  );
  const mx = raw[raw.length - 1]!;
  const values = raw.map((v) => v / mx);
  let diff = 0;
  values.forEach((a, i) => {
    if (Math.abs(a - NF4[i]!) > diff) diff = Math.abs(a - NF4[i]!);
  });
  const steps: Nf4Step[] = [{ phase: "pdf", n: 0 }];
  for (let i = 0; i < 8; i++) steps.push({ phase: "pos", n: i + 1 });
  for (let i = 0; i < 7; i++) steps.push({ phase: "neg", n: i + 1 });
  steps.push({ phase: "zero", n: 16 });
  steps.push({ phase: "normalise", n: 16 });
  steps.push({ phase: "compare", n: 16 });
  return { pos, neg, raw, max: mx, values, max_diff: diff, steps };
}

/** n approximately normal FP16 values (reference: demo_normal). */
export function demoNormal(n: number, seed: number): number[] {
  const rng = new Rng(seed);
  return Array.from({ length: n }, () => roundTo(rng.normal(), "fp16"));
}

/** Blockwise NF4 against blockwise INT4 (reference: nf4_vs_int4). */
export function nf4VsInt4(
  xs: readonly number[],
  block = 64,
): { mse_nf4: number; mse_int4: number; signal: number } {
  const nf = nf4Quantise(xs, block).values;
  const i4: number[] = [];
  for (let b0 = 0; b0 < xs.length; b0 += block) {
    const blk = xs.slice(b0, b0 + block);
    const p = intParams(blk, 4, "sym");
    for (const x of blk) i4.push(dequantInt(quantInt(x, p), p));
  }
  let eNf = 0;
  let eI4 = 0;
  let sig = 0;
  xs.forEach((x, i) => {
    eNf += (x - nf[i]!) * (x - nf[i]!);
    eI4 += (x - i4[i]!) * (x - i4[i]!);
    sig += x * x;
  });
  const n = xs.length;
  return { mse_nf4: eNf / n, mse_int4: eI4 / n, signal: sig / n };
}

// ---------------------------------------------------------------------------
// Chapter 10: one dot product in four number systems, and its energy
// ---------------------------------------------------------------------------

export type HorowitzRow = {
  id: string;
  label: string;
  pj: number;
  um2: number | null;
};

/** Energy and area per operation at 45 nm (Horowitz, ISSCC 2014). */
export const HOROWITZ: readonly HorowitzRow[] = data.horowitz;
const HZ = Object.fromEntries(HOROWITZ.map((h) => [h.id, h]));

export type MacKind = "fp32" | "fp16" | "int8";
export const DOT_MAC = data.dotMac as Record<MacKind, [string, string]>;

/** Energy of one multiply-accumulate (pJ) from the table. */
export function macPj(kind: MacKind): number {
  const [m, a] = DOT_MAC[kind];
  return HZ[m]!.pj + HZ[a]!.pj;
}

export type DotStep = {
  i: number;
  ref: number;
  fp32: number;
  fp16: number;
  int8_acc: number;
  int8: number;
  mx_acc: number;
  mx: number;
  q8: [number, number];
  e2m1: [number, number];
};

export type DotRun = {
  scales: { int8: [number, number]; mx: [number, number] };
  pj: Record<MacKind, number>;
  steps: DotStep[];
};

/** Chapter 10: a dot product in FP32, FP16, INT8 and MXFP4 (reference: dot_steps). */
export function dotSteps(a: readonly number[], b: readonly number[]): DotRun {
  const pa = intParams(a, 8, "sym");
  const pb = intParams(b, 8, "sym");
  const ma = mxQuantise(a, "mxfp4");
  const mb = mxQuantise(b, "mxfp4");
  const ea = pow2(ma.shared_exp);
  const eb = pow2(mb.shared_exp);
  let ref = 0;
  let s32 = 0;
  let s16 = 0;
  let acc8 = 0;
  let accmx = 0;
  const steps: DotStep[] = [];
  for (let i = 0; i < a.length; i++) {
    ref += a[i]! * b[i]!;
    s32 = roundTo(s32 + roundTo(a[i]! * b[i]!, "fp32"), "fp32");
    s16 = roundTo(s16 + roundTo(a[i]! * b[i]!, "fp16"), "fp16");
    const qa = quantInt(a[i]!, pa);
    const qb = quantInt(b[i]!, pb);
    acc8 += qa * qb;
    const xa = ma.values[i]! / ea;
    const xb = mb.values[i]! / eb;
    accmx += xa * xb;
    steps.push({
      i,
      ref,
      fp32: s32,
      fp16: s16,
      int8_acc: acc8,
      int8: acc8 * (pa.scale * pb.scale),
      mx_acc: accmx,
      mx: accmx * (ea * eb),
      q8: [qa, qb],
      e2m1: [xa, xb],
    });
  }
  return {
    scales: {
      int8: [pa.scale, pb.scale],
      mx: [ma.shared_exp, mb.shared_exp],
    },
    pj: { fp32: macPj("fp32"), fp16: macPj("fp16"), int8: macPj("int8") },
    steps,
  };
}
