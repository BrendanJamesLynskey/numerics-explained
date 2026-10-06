/**
 * Every number the chapters quote comes from here: a path into the tested
 * model's results, e.g. "e4m3.max", "sum.fp16.uniform.naive" or
 * "gran.4.sym.2.mse", formatted by kind. The MDX writes <V of="…" fmt="…"/>,
 * so prose cannot drift from the model (tests/unit/values.test.ts checks
 * every path the chapters use resolves, and recomputes the numbers the
 * prose spells out in words).
 */
import {
  exact,
  fmtDb,
  int,
  pct,
  pow2Text,
  sci,
  sqnrDb,
  trim,
} from "@/lib/format";

import {
  FORMAT_ORDER,
  INFO,
  MX_FORMATS,
  MX_ORDER,
  demoActivations,
  demoBlock,
  demoWeights,
  granularitySteps,
  mxQuantise,
  probeSteps,
  roundTo,
  roundingSteps,
  stagnationSteps,
  sumSteps,
  zeropointSteps,
  type BlockKind,
  type FormatId,
  type RoundDist,
  type SumDist,
} from "./model";

export type Fmt =
  | "num"
  | "exact"
  | "pow2"
  | "sci"
  | "int"
  | "pct"
  | "db"
  | "raw";

/** The parameters every chapter's data set uses (also the widgets' defaults). */
export const DEMO = {
  sumN: 8192,
  sumEvery: 128,
  sumSeed: 7,
  roundN: 64,
  roundSeed: 3,
  stagSteps: 128,
  stagSeed: 5,
  weights: [16, 64, 5] as const,
  acts: [24, 8] as const,
  blockSeed: 5,
} as const;

/** delta = an eighth of an ulp at 1, per format, for the stagnation demo. */
export function stagDelta(fid: FormatId): number {
  return INFO[fid].eps / 8;
}

function build(): Record<string, unknown> {
  const t: Record<string, unknown> = { demo: { ...DEMO } };
  for (const f of FORMAT_ORDER) t[f] = { ...INFO[f], digits: INFO[f].m + 1 };
  const sum: Record<string, unknown> = {};
  for (const f of ["fp16", "bf16"] as FormatId[]) {
    const byDist: Record<string, unknown> = {};
    for (const d of ["uniform", "ones", "normal"] as SumDist[]) {
      const st = sumSteps(f, DEMO.sumN, d, DEMO.sumSeed, DEMO.sumEvery).steps;
      byDist[d] = st[st.length - 1];
    }
    sum[f] = byDist;
  }
  t.sum = sum;
  const round: Record<string, unknown> = {};
  for (const f of ["e4m3", "e2m1", "fp16"] as FormatId[]) {
    const byDist: Record<string, unknown> = {};
    for (const d of ["uniform", "low", "ties"] as RoundDist[]) {
      const st = roundingSteps(f, DEMO.roundN, d, DEMO.roundSeed);
      byDist[d] = st[st.length - 1];
    }
    round[f] = byDist;
  }
  t.round = round;
  const stag: Record<string, unknown> = {};
  for (const f of ["fp16", "bf16", "e4m3"] as FormatId[]) {
    const st = stagnationSteps(
      f,
      1,
      stagDelta(f),
      DEMO.stagSteps,
      DEMO.stagSeed,
    );
    stag[f] = { ...st[st.length - 1], delta: stagDelta(f) };
  }
  t.stag = stag;
  const W = demoWeights(...DEMO.weights);
  const gran: Record<string, unknown> = {};
  for (const b of [8, 4, 3]) {
    const bySch: Record<string, unknown> = {};
    for (const s of ["sym", "asym"] as const) {
      const st = granularitySteps(W, b, s);
      bySch[s] = st.map((g) => ({
        mse: g.mse,
        max_err: g.max_err,
        bits_per_weight: g.bits_per_weight,
        scales: g.params.length,
        sqnr: sqnrDb(g.signal, g.mse),
      }));
    }
    gran[String(b)] = bySch;
  }
  t.gran = gran;
  const acts = demoActivations(...DEMO.acts);
  const zp: Record<string, unknown> = {};
  for (const b of [8, 4, 3]) {
    const z = zeropointSteps(acts, b);
    const last = z.steps[z.steps.length - 1]!;
    zp[String(b)] = {
      sym_scale: z.sym.scale,
      asym_scale: z.asym.scale,
      zero: z.asym.zero,
      mse_sym: last.mse_sym,
      mse_asym: last.mse_asym,
      used_sym: z.used_sym,
      used_asym: z.used_asym,
      levels_sym: z.levels_sym,
      levels_asym: z.levels_asym,
    };
  }
  t.zp = zp;
  const mx: Record<string, unknown> = {};
  for (const m of MX_ORDER) {
    const byKind: Record<string, unknown> = {};
    for (const k of ["normal", "outlier", "tiny"] as BlockKind[]) {
      const blk = demoBlock(k, DEMO.blockSeed);
      const r = mxQuantise(blk, m);
      let sq = 0;
      let zeros = 0;
      blk.forEach((v, i) => {
        const d = v - r.values[i]!;
        sq += d * d;
        if (r.values[i] === 0 && v !== 0) zeros += 1;
      });
      // the same values in the element format with no shared scale
      const elem = MX_FORMATS[m].elem;
      let plainZeros = 0;
      if (elem !== "int8")
        for (const v of blk)
          if (roundTo(v, elem, "rne", true) === 0 && v !== 0) plainZeros += 1;
      byKind[k] = {
        plain_zeros: plainZeros,
        bits_per_value: MX_FORMATS[m].bits + 8 / blk.length,
        shared_exp: r.shared_exp,
        scale: 2 ** r.shared_exp,
        amax: r.amax,
        mse: sq / blk.length,
        zeros,
      };
    }
    mx[m] = byKind;
  }
  t.mx = mx;
  const probe: Record<string, unknown> = {};
  for (const p of probeSteps()) probe[String(p.k)] = p.formats;
  t.probe = probe;
  return t;
}

let TREE: Record<string, unknown> | null = null;

export function lookup(path: string): number | string | boolean {
  if (!TREE) TREE = build();
  let cur: unknown = TREE;
  for (const k of path.split(".")) {
    if (cur === null || typeof cur !== "object" || !(k in cur))
      throw new Error(`no value at ${path}`);
    cur = (cur as Record<string, unknown>)[k];
  }
  if (
    typeof cur !== "number" &&
    typeof cur !== "string" &&
    typeof cur !== "boolean"
  )
    throw new Error(`${path} is not a number or string`);
  return cur;
}

export function formatValue(v: number | string | boolean, fmt: Fmt): string {
  if (typeof v !== "number") return String(v);
  switch (fmt) {
    case "exact":
      return exact(v);
    case "pow2":
      return pow2Text(v);
    case "sci":
      return sci(v, 3);
    case "int":
      return int(v);
    case "pct":
      return pct(v, 1);
    case "db":
      return fmtDb(v);
    case "raw":
      return String(v);
    default:
      return trim(v);
  }
}
