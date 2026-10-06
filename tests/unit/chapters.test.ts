/**
 * Parity for chapters 6-10: the TypeScript port against the Python
 * reference's fixtures, exactly, except where the reference itself uses a
 * transcendental function (the normal CDF's exp, in NF4's derivation),
 * which is matched to a relative 1e-15.
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/num_fixtures.json";

import { digest } from "./helpers/digest";

import {
  HOROWITZ,
  LLM_INT8_THRESHOLD,
  NF4,
  NF4_OFFSET,
  OUTLIER_CHANNELS,
  awqSearch,
  demoBlock,
  demoLayer,
  demoNormal,
  demoOutlierLayer,
  dotSteps,
  gptqSteps,
  int8Vectorwise,
  llmInt8,
  macPj,
  matmulWX,
  nf4Build,
  nf4VsInt4,
  normCdf,
  normPpf,
  outError,
  outlierSteps,
  smoothSteps,
} from "@/lib/num/model";

const close = (a: number, b: number, rel = 1e-15) =>
  expect(Math.abs(a - b)).toBeLessThanOrEqual(rel * Math.max(1, Math.abs(b)));

describe("chapter 6: outliers", () => {
  const ol = demoOutlierLayer(8, 16, 32, 13);
  it("the data set", () => {
    expect(digest([ol.W, ol.X])).toBe(fx.outlier.layer);
    expect(ol.W[0]).toEqual(fx.outlier.W0);
    expect(ol.X[3]).toEqual(fx.outlier.X3);
    expect(OUTLIER_CHANNELS).toEqual([3, 11]);
    expect(LLM_INT8_THRESHOLD).toBe(6);
  });
  it("per tensor, vector-wise, decomposition", () => {
    expect(outlierSteps(ol.W, ol.X)).toEqual(fx.outlier.run);
  });
  it("the decomposition finds exactly the outlier channels", () => {
    expect(llmInt8(ol.W, ol.X).outliers).toEqual([3, 11]);
    // with no channels the INT8 part is zero
    const z = int8Vectorwise(ol.W, ol.X, []);
    expect(z.every((r) => r.every((v) => v === 0))).toBe(true);
    const Y = matmulWX(ol.W, ol.X);
    expect(outError(Y, Y)).toBe(0);
  });
  it("SmoothQuant sweep", () => {
    expect(smoothSteps(ol.W, ol.X)).toEqual(fx.smooth);
  });
});

describe("chapter 7: GPTQ steps", () => {
  const lay = demoLayer(8, 16, 64, 21);
  it("4-bit", () => {
    const g = gptqSteps(lay.W, lay.X, 4);
    expect(digest(g.steps)).toBe(fx.gptqSteps.bits4.digest);
    [0, 1, 8, 16].forEach((i, k) =>
      expect(g.steps[i]).toEqual(fx.gptqSteps.bits4.frames[k]),
    );
    expect(g.steps.map((s) => [s.err_gptq, s.err_rtn])).toEqual(
      fx.gptqSteps.bits4.errs,
    );
  });
  it("3-bit", () => {
    const g = gptqSteps(lay.W, lay.X, 3);
    expect(digest(g.steps)).toBe(fx.gptqSteps.bits3.digest);
  });
  it("AWQ's activation statistic", () => {
    expect(awqSearch(lay.W, lay.X, 3, 8).meanAbs).toEqual(fx.awqMeanAbs);
  });
});

describe("chapter 8: NF4", () => {
  it("the normal CDF and quantiles", () => {
    expect(normCdf(0)).toBe(0.5);
    close(normCdf(-1) + normCdf(1), 1);
    close(normPpf(0.5), 0, 1e-15);
    close(normPpf(NF4_OFFSET), fx.nf4Build.max);
  });
  it("the derivation matches the reference and bitsandbytes' table", () => {
    const b = nf4Build();
    expect(b.steps).toEqual(fx.nf4Build.steps);
    b.values.forEach((v, i) => close(v, fx.nf4Build.values[i]!));
    b.pos.forEach((q, i) => {
      close(q.p, fx.nf4Build.pos[i]!.p);
      close(q.z, fx.nf4Build.pos[i]!.z);
    });
    b.neg.forEach((q, i) => close(q.z, fx.nf4Build.neg[i]!.z));
    close(b.max_diff, fx.nf4Build.max_diff, 1e-6);
    expect(b.max_diff).toBeLessThan(2e-7);
    b.values.forEach((v, i) =>
      expect(Math.abs(v - NF4[i]!)).toBeLessThan(2e-7),
    );
  });
  it("NF4 against INT4 on normal data", () => {
    expect(nf4VsInt4(demoNormal(4096, 17))).toEqual(fx.nf4VsInt4);
  });
});

describe("chapter 10: dot products", () => {
  it("matches the reference", () => {
    for (const [s1, s2] of [
      [5, 6],
      [7, 8],
    ] as const)
      expect(
        dotSteps(demoBlock("normal", s1), demoBlock("normal", s2)),
      ).toEqual(fx.dot[`${s1}-${s2}` as keyof typeof fx.dot]);
  });
  it("energy per multiply-accumulate", () => {
    expect(HOROWITZ).toHaveLength(11);
    expect(macPj("fp32")).toBe(3.7 + 0.9);
    expect(macPj("int8")).toBe(0.2 + 0.1);
  });
});
