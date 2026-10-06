/**
 * Parity: the TypeScript port against the Python reference's fixtures,
 * exactly (toEqual / identical digests, no tolerance). Every code of every
 * format of 16 bits or fewer is decoded, and every encode case is run in
 * every rounding mode and both saturation modes, through the digests.
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/num_fixtures.json";

import { digest } from "./helpers/digest";

import {
  FORMAT_ORDER,
  INFO,
  MODES,
  MX_ORDER,
  NF4,
  NF4_MID,
  Rng,
  awqSearch,
  decode,
  demoActivations,
  demoBlock,
  demoLayer,
  demoWeights,
  e8m0Decode,
  encode,
  encodeCases,
  exponentOf,
  fields,
  gptq,
  granularitySteps,
  layerError,
  mxElement,
  mxQuantise,
  mxSteps,
  nf4Code,
  nf4Quantise,
  pow2,
  powDyadic,
  probeSteps,
  quantiseMatrix,
  rneInt,
  roundTo,
  roundingSteps,
  rtn,
  smoothquant,
  stagnationSteps,
  sumInputs,
  sumKahan,
  sumNaive,
  sumPairwise,
  sumSteps,
  w8a8Error,
  zeropointSteps,
  zoomSteps,
  type BlockKind,
  type FormatId,
  type Gran,
  type Mode,
  type MxId,
  type RoundDist,
  type Scheme,
  type SumDist,
} from "@/lib/num/model";

const ENCODE_N: Record<string, number> = {
  fp32: 20000,
  fp16: 8000,
  bf16: 8000,
};

describe("exact helpers", () => {
  it("pow2 and exponentOf are exact over the whole double range", () => {
    for (let k = -1074; k <= 1023; k++) {
      expect(pow2(k)).toBe(2 ** k);
      expect(exponentOf(pow2(k))).toBe(k);
      if (k < 1023 && k > -1070) expect(exponentOf(pow2(k) * 1.75)).toBe(k);
    }
    expect(pow2(1024)).toBe(Infinity);
    expect(pow2(-1075)).toBe(0);
  });
  it("rneInt rounds ties to even, also below zero", () => {
    expect([2.5, 3.5, -2.5, -3.5, -0.5, 0.49].map(rneInt)).toEqual([
      2, 4, -2, -4, 0, 0,
    ]);
  });
  it("e8m0 and MXINT8", () => {
    expect(e8m0Decode(255)).toBeNaN();
    expect(e8m0Decode(127)).toBe(1);
    expect(mxElement(10, "mxint8").value).toBe(127 / 64);
    expect(mxElement(-0.3, "mxint8")).toEqual({ code: 237, value: -19 / 64 });
    expect(mxElement(0.3, "mxint8", "sr", 0).value).toBe(20 / 64);
  });
});

describe("format constants", () => {
  for (const f of FORMAT_ORDER)
    it(f, () => expect(INFO[f]).toEqual(fx.info[f]));
});

describe("decode: every code", () => {
  for (const f of FORMAT_ORDER) {
    it(f, () => {
      const d = fx.decode[f]!;
      const codes =
        "codes" in d && d.codes
          ? d.codes
          : Array.from({ length: 2 ** INFO[f].bits }, (_, c) => c);
      expect(digest(codes.map((c) => decode(c, f)))).toBe(d.digest);
    });
  }
  it("fields split a code", () => {
    expect(fields(0b1_1111_110, "e4m3")).toEqual({ s: 1, E: 15, M: 6 });
    expect(fields(0xff800000, "fp32")).toEqual({ s: 1, E: 255, M: 0 });
  });
});

describe("encode: every mode, both saturation modes", () => {
  for (const f of FORMAT_ORDER) {
    it(f, () => {
      const e = fx.encode[f]!;
      const xs = encodeCases(f, ENCODE_N[f] ?? 3000, 2024);
      expect(xs.length).toBe(e.cases);
      const rng = new Rng(77);
      const codes: number[] = [];
      for (const x of xs)
        for (const mode of MODES)
          for (const sat of [false, true])
            codes.push(encode(x, f, mode, sat, rng.u32()));
      for (const s of e.sample)
        expect(encode(s.x, f, s.mode as Mode, s.sat, s.u)).toBe(s.code);
      expect(digest(codes)).toBe(e.digest);
    });
  }
});

describe("animation states", () => {
  for (const f of FORMAT_ORDER)
    it(`zoom ${f}`, () => expect(zoomSteps(f)).toEqual(fx.zoom[f]));
  for (const r of fx.rounding)
    it(`rounding ${r.fmt} ${r.dist}`, () =>
      expect(
        roundingSteps(r.fmt as FormatId, 64, r.dist as RoundDist, 3),
      ).toEqual(r.steps));
  for (const s of fx.stagnation)
    it(`stagnation ${s.fmt}`, () =>
      expect(stagnationSteps(s.fmt as FormatId, 1, s.delta, 128, 5)).toEqual(
        s.steps,
      ));
  for (const s of fx.sums)
    it(`sums ${s.fmt} ${s.dist}`, () => {
      const r = sumSteps(s.fmt as FormatId, 8192, s.dist as SumDist, 7, 128);
      expect(digest(r.inputs)).toBe(s.inputs);
      expect(r.steps).toEqual(s.steps);
    });
  for (const [mid, st] of Object.entries(fx.mxSteps))
    it(`mx steps ${mid}`, () =>
      expect(mxSteps(demoBlock("outlier", 5), mid as MxId)).toEqual(st));
  for (const g of fx.granularity)
    it(`granularity ${g.bits} ${g.scheme}`, () => {
      const W = demoWeights(16, 64, 5);
      const st = granularitySteps(W, g.bits, g.scheme as Scheme);
      expect(
        st.map((s) => ({
          gran: s.gran,
          group: s.group,
          mse: s.mse,
          max_err: s.max_err,
          bits_per_weight: s.bits_per_weight,
          nparams: s.params.length,
          digest: digest([s.codes, s.deq, s.params]),
        })),
      ).toEqual(g.steps);
    });
  for (const z of fx.zeropoint)
    it(`zero point ${z.bits}`, () =>
      expect(zeropointSteps(demoActivations(24, 8), z.bits)).toEqual(z.out));
});

describe("data sets", () => {
  it("weights, activations, blocks", () => {
    const W = demoWeights(16, 64, 5);
    expect(digest(W)).toBe(fx.demo.weights.digest);
    expect(W[0]).toEqual(fx.demo.weights.row0);
    expect(demoActivations(24, 8)).toEqual(fx.demo.activations.values);
    for (const b of fx.demo.blocks)
      expect(demoBlock(b.kind as BlockKind, b.seed)).toEqual(b.values);
  });
});

describe("quantisers", () => {
  const W = demoWeights(16, 64, 5);
  for (const q of fx.quant)
    it(`int ${q.bits} ${q.scheme} ${q.gran} ${q.group}`, () => {
      const r = quantiseMatrix(
        W,
        q.bits,
        q.scheme as Scheme,
        q.gran as Gran,
        q.group,
      );
      expect(r.params).toEqual(q.params);
      expect([r.mse, r.signal, r.max_err, r.bits_per_weight]).toEqual([
        q.mse,
        q.signal,
        q.max_err,
        q.bits_per_weight,
      ]);
      expect(digest([r.codes, r.deq])).toBe(q.digest);
    });
  for (const m of fx.mx)
    it(`mx ${m.kind} ${m.seed} ${m.mx} ${m.mode}`, () =>
      expect(
        mxQuantise(
          demoBlock(m.kind as BlockKind, m.seed),
          m.mx as MxId,
          m.mode as Mode,
          9,
        ),
      ).toEqual(m.out));
  it("nf4", () => {
    expect(nf4Quantise(fx.nf4.in)).toEqual(fx.nf4.out);
    NF4_MID.forEach((t, i) => expect(nf4Code(t)).toBe(i));
    expect(NF4.length).toBe(16);
  });
  it("sum totals", () => {
    const xs = sumInputs(3000, "normal", 11);
    for (const s of fx.sumTotals) {
      const f = s.fmt as FormatId;
      expect([sumNaive(xs, f), sumKahan(xs, f), sumPairwise(xs, f)]).toEqual([
        s.naive,
        s.kahan,
        s.pairwise,
      ]);
    }
  });
  it("GPTQ, AWQ, SmoothQuant", () => {
    const lay = demoLayer(8, 16, 64, 21);
    expect(digest([lay.W, lay.X])).toBe(fx.gptq.layer);
    const g4 = gptq(lay.W, lay.X, 4);
    expect(g4.Q).toEqual(fx.gptq.bits4.Q);
    expect(g4.scales).toEqual(fx.gptq.bits4.scales);
    expect(digest(g4.U)).toBe(fx.gptq.bits4.U);
    expect(digest(g4.steps)).toBe(fx.gptq.bits4.steps);
    expect(layerError(lay.W, g4.Q, lay.X)).toBe(fx.gptq.bits4.err);
    expect(layerError(lay.W, rtn(lay.W, 4), lay.X)).toBe(fx.gptq.bits4.rtn);
    const g3 = gptq(lay.W, lay.X, 3);
    expect(g3.Q).toEqual(fx.gptq.bits3.Q);
    const aw = awqSearch(lay.W, lay.X, 3, 8);
    expect(aw.best).toBe(fx.awq.best);
    expect(aw.results.map((r) => r.err)).toEqual(fx.awq.errs);
    expect(digest(aw.results.map((r) => r.scales))).toBe(fx.awq.scales);
    const sq = smoothquant(lay.W, lay.X);
    expect(sq.s).toEqual(fx.smoothquant.s);
    expect(digest(sq.Xs)).toBe(fx.smoothquant.Xs);
    expect(digest(sq.Ws)).toBe(fx.smoothquant.Ws);
    expect(w8a8Error(lay.W, lay.X).err).toBe(fx.smoothquant.before);
    expect(w8a8Error(sq.Ws, sq.Xs).err).toBe(fx.smoothquant.after);
    for (const [x, k, v] of fx.pow) expect(powDyadic(x!, k!)).toBe(v);
  });
  it("roundTo is decode(encode())", () => {
    expect(roundTo(464, "e4m3")).toBe(448);
    expect(roundTo(465, "e4m3")).toBeNaN();
    expect(roundTo(1e6, "e2m1")).toBe(6);
    expect(roundTo(-1e6, "fp16")).toBe(-Infinity);
    expect(roundTo(NaN, "e2m1")).toBe(0);
    expect(MX_ORDER).toHaveLength(6);
  });
});

describe("probe", () => {
  it("matches the reference", () => expect(probeSteps()).toEqual(fx.probe));
});

describe("NaN", () => {
  it("encodes NaN as each format's NaN, and as +0 where there is none", () => {
    expect(encode(NaN, "e4m3")).toBe(0x7f);
    expect(encode(NaN, "fp16")).toBe(0x7e00);
    expect(encode(NaN, "e3m2")).toBe(0);
  });
});
