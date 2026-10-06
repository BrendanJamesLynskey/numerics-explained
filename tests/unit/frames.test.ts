/**
 * Frame tests (visual standard §4): for key frames of every animation, the
 * state the site draws equals the Python reference's state for that frame
 * (from the fixtures), and the caption built from the reference's state is
 * the caption the site shows. tests/e2e/frames.spec.ts then checks the page.
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/num_fixtures.json";

import {
  granCaption,
  mxCaption,
  probeCaption,
  roundCaption,
  stagCaption,
  sumCaption,
  zoomCaption,
  zpCaption,
  type GranSummary,
} from "@/lib/num/captions";
import {
  demoActivations,
  demoBlock,
  demoWeights,
  granularitySteps,
  mxQuantise,
  mxSteps,
  probeSteps,
  roundingSteps,
  stagnationSteps,
  sumSteps,
  zeropointSteps,
  zoomSteps,
  type FormatId,
  type MxId,
  type MxResult,
  type MxStep,
  type ProbeStep,
  type RoundDist,
  type RoundStep,
  type Scheme,
  type StagStep,
  type SumDist,
  type SumStep,
  type ZoomStep,
  type ZpStep,
} from "@/lib/num/model";
import { DEMO, stagDelta } from "@/lib/num/values";

describe("chapter 1: zoom frames", () => {
  for (const fid of ["e4m3", "fp16", "fp32", "e2m1"] as FormatId[]) {
    const ts = zoomSteps(fid);
    const py = fx.zoom[fid] as ZoomStep[];
    for (const i of [0, Math.floor(ts.length / 2), ts.length - 1])
      it(`${fid} step ${i}`, () => {
        expect(ts[i]).toEqual(py[i]);
        expect(zoomCaption(fid, ts[i]!, i === 0)).toBe(
          zoomCaption(fid, py[i]!, i === 0),
        );
      });
  }
  it("says what the reference says", () => {
    const s = zoomSteps("e4m3");
    expect(zoomCaption("e4m3", s[0]!, true)).toBe(
      "The whole positive range, 0 to 2⁹: 126 positive FP8 E4M3 values, 32 apart at the right edge. Every binade holds the same 8 values, so each halving of the view packs them twice as densely towards zero.",
    );
    expect(zoomCaption("e4m3", s[s.length - 1]!, false)).toContain(
      "every value is subnormal",
    );
  });
});

describe("chapter 2: rounding frames", () => {
  for (const [fid, dist] of [
    ["e4m3", "low"],
    ["fp16", "ties"],
    ["e2m1", "uniform"],
  ] as const) {
    const ts = roundingSteps(
      fid,
      DEMO.roundN,
      dist as RoundDist,
      DEMO.roundSeed,
    );
    const py = fx.rounding.find((r) => r.fmt === fid && r.dist === dist)!
      .steps as RoundStep[];
    for (const i of [0, 17, 63])
      it(`${fid} ${dist} input ${i}`, () => {
        expect(ts[i]).toEqual(py[i]);
        expect(roundCaption(fid, ts[i]!, 64)).toBe(
          roundCaption(fid, py[i]!, 64),
        );
      });
  }
  it("describes a tie", () => {
    const s = roundingSteps("fp16", 64, "ties", DEMO.roundSeed)[0]!;
    expect(roundCaption("fp16", s, 64)).toContain(
      "exactly halfway, so nearest-even takes the neighbour whose last mantissa bit is 0",
    );
  });
  for (const fid of ["fp16", "bf16", "e4m3"] as FormatId[]) {
    const ts = stagnationSteps(
      fid,
      1,
      stagDelta(fid),
      DEMO.stagSteps,
      DEMO.stagSeed,
    );
    const py = fx.stagnation.find((s) => s.fmt === fid)!.steps as StagStep[];
    for (const i of [0, 8, 128])
      it(`stagnation ${fid} step ${i}`, () => {
        expect(ts[i]).toEqual(py[i]);
        expect(stagCaption(fid, ts[i]!, 1)).toBe(stagCaption(fid, py[i]!, 1));
      });
  }
  it("says nearest-even is stuck", () => {
    const s = stagnationSteps("fp16", 1, stagDelta("fp16"), 128, DEMO.stagSeed);
    expect(stagCaption("fp16", s[128]!, 1)).toContain("nearest-even 1 (stuck");
  });
});

describe("chapter 3: sum frames", () => {
  for (const [fid, dist] of [
    ["fp16", "uniform"],
    ["bf16", "ones"],
    ["fp16", "normal"],
  ] as const) {
    const ts = sumSteps(
      fid,
      DEMO.sumN,
      dist as SumDist,
      DEMO.sumSeed,
      DEMO.sumEvery,
    ).steps;
    const py = fx.sums.find((s) => s.fmt === fid && s.dist === dist)!
      .steps as SumStep[];
    for (const i of [0, 15, 63])
      it(`${fid} ${dist} checkpoint ${i}`, () => {
        expect(ts[i]).toEqual(py[i]);
        expect(sumCaption(fid, ts[i]!, DEMO.sumN)).toBe(
          sumCaption(fid, py[i]!, DEMO.sumN),
        );
      });
  }
  it("names the stall", () => {
    const st = sumSteps("fp16", 8192, "uniform", DEMO.sumSeed, 128).steps;
    expect(sumCaption("fp16", st[63]!, 8192)).toContain(
      "FP16 naive: 2,048 (50% off)",
    );
  });
});

describe("chapter 4: probe and MX frames", () => {
  const ts = probeSteps();
  const py = fx.probe as ProbeStep[];
  for (const i of [0, 16, 35, 44])
    it(`probe step ${i}`, () => {
      expect(ts[i]).toEqual(py[i]);
      expect(probeCaption(ts[i]!)).toBe(probeCaption(py[i]!));
    });
  it("names underflow, overflow and saturation", () => {
    expect(probeCaption(ts[16]!)).toContain(
      "FP8 E4M3 0.001953 (subnormal), 50% off",
    );
    expect(probeCaption(ts[0]!)).toContain("FP16 flushed to 0");
    expect(probeCaption(ts[35]!)).toContain("FP8 E4M3 overflows");
    expect(probeCaption(ts[44]!)).toContain("FP4 E2M1 saturates at 6");
  });
  const block = demoBlock("outlier", DEMO.blockSeed);
  for (const mid of ["mxfp4", "mxfp8_e4m3", "mxint8"] as MxId[]) {
    const r = mxQuantise(block, mid);
    const pyR = fx.mx.find(
      (m) =>
        m.kind === "outlier" &&
        m.seed === 5 &&
        m.mx === mid &&
        m.mode === "rne",
    )!.out as MxResult;
    const st = mxSteps(block, mid);
    const pySt = (fx.mxSteps as Record<string, MxStep[]>)[mid]!;
    for (const i of [1, 2, 3, 20, 34])
      it(`${mid} step ${i}`, () => {
        expect(st[i]).toEqual(pySt[i]);
        expect(r).toEqual(pyR);
        expect(mxCaption(mid, block, r, st[i]!)).toBe(
          mxCaption(mid, block, pyR, pySt[i]!),
        );
      });
  }
  it("explains the scale", () => {
    const r = mxQuantise(block, "mxfp4");
    expect(mxCaption("mxfp4", block, r, mxSteps(block, "mxfp4")[2]!)).toBe(
      "The largest power of two ≤ 24 is 2⁴; the element type's largest is 2². So X = 2² (E8M0 code 129), and every element is stored as V / X.",
    );
  });
});

describe("chapter 5: granularity and zero-point frames", () => {
  const W = demoWeights(...DEMO.weights);
  for (const [bits, scheme] of [
    [4, "sym"],
    [3, "asym"],
  ] as const) {
    const ts = granularitySteps(W, bits, scheme as Scheme);
    const py = fx.granularity.find(
      (g) => g.bits === bits && g.scheme === scheme,
    )!.steps as (GranSummary & { digest: string })[];
    for (const i of [0, 2, 4])
      it(`INT${bits} ${scheme} step ${i}`, () => {
        const t = ts[i]!;
        const mine: GranSummary = {
          gran: t.gran,
          group: t.group,
          mse: t.mse,
          max_err: t.max_err,
          bits_per_weight: t.bits_per_weight,
          nparams: t.params.length,
        };
        const theirs: GranSummary = { ...py[i]! };
        delete (theirs as Partial<{ digest: string }>).digest;
        expect(mine).toEqual(theirs);
        expect(granCaption(mine, bits, scheme, t.signal, 1024)).toBe(
          granCaption(theirs, bits, scheme, t.signal, 1024),
        );
      });
  }
  const xs = demoActivations(...DEMO.acts);
  for (const bits of [4, 8]) {
    const z = zeropointSteps(xs, bits);
    const py = fx.zeropoint.find((q) => q.bits === bits)!.out;
    for (const i of [0, 1, 2, 25])
      it(`zero point INT${bits} step ${i}`, () => {
        expect(z.steps[i]).toEqual(py.steps[i]);
        expect(zpCaption(xs, z.sym, z.asym, z.steps[i]!, bits)).toBe(
          zpCaption(xs, py.sym, py.asym, py.steps[i] as ZpStep, bits),
        );
      });
  }
});

describe("caption branches", () => {
  it("covers the remaining cases", () => {
    expect(
      stagCaption("fp16", { t: 3, exact: 1.5, rne: 1.5, sr: 1.5 }, 1),
    ).not.toContain("stuck");
    const blk = [1.9 * 2 ** 5, ...new Array<number>(31).fill(0)];
    const r = mxQuantise(blk, "mxfp8_e4m3");
    expect(
      mxCaption("mxfp8_e4m3", blk, r, mxSteps(blk, "mxfp8_e4m3")[3]!),
    ).toContain("(beyond ±448: clamped)");
    expect(mxQuantise(new Array<number>(32).fill(0), "mxfp4").scale_code).toBe(
      0,
    );
    expect(
      granCaption(
        {
          gran: "channel",
          group: 0,
          mse: 1,
          max_err: 1,
          bits_per_weight: 4.25,
          nparams: 16,
        },
        4,
        "sym",
        2,
        1024,
      ),
    ).toContain("Per channel: one scale per output row, 16 in all");
  });
});
