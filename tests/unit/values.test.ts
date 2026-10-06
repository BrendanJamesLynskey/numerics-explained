/**
 * The numbers the chapters print: every <V of="…" /> path in the MDX
 * resolves in the library, and every number the prose spells out in words
 * (or in a chapter summary) is recomputed here from the library.
 */
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  asPow2,
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
import { SECTIONS } from "@/lib/mdx/sections";
import {
  INFO,
  MX_FORMATS,
  demoActivations,
  demoWeights,
  neighbours,
  pow2,
  roundTo,
} from "@/lib/num/model";
import { DEMO, formatValue, lookup, type Fmt } from "@/lib/num/values";

const DIR = path.join(__dirname, "../../content/chapters");
const FILES = readdirSync(DIR).filter((f) => f.endsWith(".mdx"));

describe("library values quoted in the chapters", () => {
  const uses: { file: string; of: string; fmt: string }[] = [];
  for (const f of FILES) {
    const src = readFileSync(path.join(DIR, f), "utf8");
    for (const m of src.matchAll(/<V of="([^"]+)"(?: fmt="([^"]+)")? \/>/g))
      uses.push({ file: f, of: m[1]!, fmt: m[2] ?? "num" });
  }
  const home = readFileSync(
    path.join(__dirname, "../../src/app/page.tsx"),
    "utf8",
  );
  for (const m of home.matchAll(/<V of="([^"]+)"(?: fmt="([^"]+)")? \/>/g))
    uses.push({ file: "page.tsx", of: m[1]!, fmt: m[2] ?? "num" });

  it("are used", () => expect(uses.length).toBeGreaterThan(40));
  it("all resolve to a number", () => {
    for (const u of uses) {
      const v = lookup(u.of);
      expect(typeof v, `${u.file}: ${u.of}`).toBe("number");
      expect(formatValue(v, u.fmt as Fmt).length).toBeGreaterThan(0);
    }
  });
  it("rejects bad paths", () => {
    expect(() => lookup("e4m3.nope")).toThrow();
    expect(() => lookup("sum.fp16")).toThrow();
  });
});

describe("numbers written in the prose", () => {
  it("chapter 1", () => {
    // E4M3: 448 = 2^8 x 1.75; an IEEE-style E4M3 would stop at 240
    expect(INFO.e4m3.max).toBe(2 ** 8 * 1.75);
    expect(2 ** 7 * (2 - 2 ** -3)).toBe(240);
    // near 400 the neighbours are 32 apart
    const nb = neighbours(400, "e4m3");
    expect(nb.up - nb.down).toBe(32);
    // BF16 reaches about 3.4 x 10^38
    expect(trim(INFO.bf16.max, 2)).toBe("3.4 × 10³⁸");
  });
  it("chapter 2", () => {
    expect(lookup("demo.roundN")).toBe(64);
    expect(lookup("stag.fp16.delta")).toBe(INFO.fp16.eps / 8);
    expect(lookup("stag.fp16.t")).toBe(DEMO.stagSteps);
  });
  it("chapter 3", () => {
    const naive = lookup("sum.fp16.uniform.naive") as number;
    const ex = lookup("sum.fp16.uniform.exact") as number;
    expect(naive).toBe(2048);
    const nb = neighbours(2048.5, "fp16");
    expect(nb.up - nb.down).toBe(2); // the ulp at 2,048 is 2
    expect(naive / ex).toBeGreaterThan(0.45);
    expect(naive / ex).toBeLessThan(0.55); // half the true answer
    expect(INFO.fp16.m - INFO.bf16.m).toBe(3);
    expect(lookup("sum.bf16.uniform.naive")).toBe(256);
    // Kahan reaches the FP16 value nearest the exact total
    expect(lookup("sum.fp16.uniform.kahan")).toBe(roundTo(ex, "fp16"));
    expect(roundTo(2049, "fp16")).toBe(2048); // 2048 + 1 is a tie, to even
    expect(lookup("sum.fp16.ones.naive")).toBe(2048);
    // the summary: "Summing 8,192 numbers in FP16 stalls at 2,048"
    expect(SECTIONS[2].summary).toContain(
      `Summing ${int(DEMO.sumN)} numbers in FP16 stalls at ${int(naive)}`,
    );
  });
  it("chapter 4", () => {
    expect(2 ** (8 + 1)).toBe(512); // E4M3 scaled maximum below 2^9
    expect(2 ** MX_FORMATS.mxfp4.emax_elem).toBe(4);
    expect(lookup("mx.mxfp8_e4m3.outlier.zeros")).toBe(0);
    expect(lookup("mx.mxfp4.tiny.plain_zeros")).toBe(32); // "all 32"
    expect(lookup("mx.mxfp4.outlier.amax")).toBe(24);
    // the probe crosses: E4M3 overflows at 4/3 x 2^9
    expect(lookup("probe.9.e4m3.status")).toBe("overflow");
    expect(lookup("probe.3.e2m1.status")).toBe("saturated");
  });
  it("chapter 5", () => {
    const W = demoWeights(...DEMO.weights);
    expect([W.length, W[0]!.length]).toEqual([16, 64]);
    // four input columns larger in every row: column RMS relative to each
    // row's RMS stands out (above 1.5; every other column is below 1)
    const rowRms = W.map((r) =>
      Math.sqrt(r.reduce((a, w) => a + w * w, 0) / 64),
    );
    const colRel = W[0]!.map((_, j) =>
      Math.sqrt(W.reduce((a, r, i) => a + (r[j]! / rowRms[i]!) ** 2, 0) / 16),
    );
    expect(colRel.filter((c) => c > 1.5)).toHaveLength(4);
    expect(colRel.filter((c) => c > 1 && c <= 1.5)).toHaveLength(0);
    expect(demoActivations(...DEMO.acts)).toHaveLength(24);
    expect(16 / 128).toBe(1 / 8);
    expect(16 / 64).toBe(1 / 4);
    expect(lookup("zp.4.levels_sym")).toBe(15);
    expect(lookup("zp.4.levels_asym")).toBe(16);
    // about 6 dB per bit: per tensor, INT8 against INT4
    const d =
      ((lookup("gran.8.sym.0.sqnr") as number) -
        (lookup("gran.4.sym.0.sqnr") as number)) /
      4;
    expect(d).toBeGreaterThan(5.5);
    expect(d).toBeLessThan(7);
  });
});

describe("numbers written in the prose, chapters 6-10", () => {
  const n = (k: string) => lookup(k) as number;
  it("chapter 6", () => {
    // "under five bits' worth"
    expect(Math.log2(n("outlier.normal_levels"))).toBeLessThan(5);
    // "with two channels whose values sit near ±20 ... the others are of order 1"
    expect(n("outlier.amax_out")).toBeGreaterThan(20);
    expect(n("outlier.amax_out")).toBeLessThan(30);
    expect(n("outlier.amax_normal")).toBeLessThan(6);
    expect(n("outlier.n_out")).toBe(2);
    // "loses 20 log10 20 ≈ 26 dB, more than four bits at about 6 dB per bit"
    expect(Math.round(20 * Math.log10(20))).toBe(26);
    expect(26 / 6.02).toBeGreaterThan(4);
    // "Too far, and the weights' columns become the outliers": alpha = 1 is worse than the best
    expect(n("smooth.k8")).toBeGreaterThan(n("smooth.best_err"));
  });
  it("chapter 7", () => {
    expect(n("gptq.b4.gptq")).toBeLessThan(n("gptq.b4.rtn"));
    expect(n("gptq.b3.gptq")).toBeLessThan(n("gptq.b3.rtn"));
  });
  it("chapter 8", () => {
    // "about 1.85 standard deviations"
    expect(n("nf4.zmax")).toBeCloseTo(1.85, 2);
    expect(n("nf4.mse_nf4")).toBeLessThan(n("nf4.mse_int4"));
    expect(n("awq.b3.best")).toBeLessThan(n("awq.b3.base"));
    // "each bit removed roughly doubles the drift or more"
    const w = (c: string) => n(`tiny.weights.${c}.drift`);
    expect((w("int4_ch") / w("int8_ch")) ** (1 / 4)).toBeGreaterThan(1.9);
    expect(w("int3_ch") / w("int4_ch")).toBeGreaterThan(2);
    // "with the same blocks of 64, NF4's grid beats INT4's"
    expect(w("nf4")).toBeLessThan(w("int4_b64"));
    // "GPTQ ... recovers part of what plain INT3 rounding loses"
    expect(w("gptq3_ch")).toBeLessThan(w("int3_ch"));
    expect(n("tiny.weights.gptq3_ch.agree")).toBeGreaterThan(
      n("tiny.weights.int3_ch.agree"),
    );
    expect(n("tiny.weights.int8_ch.n")).toBe(128);
  });
  it("chapter 9", () => {
    const k = (c: string, f: string) => n(`tiny.kv.${c}.${f}`);
    // "Eight bits are close to free; four bits cost a few per cent of drift
    // and almost no flipped predictions; two bits break it"
    expect(k("int8_tok", "drift")).toBeLessThan(0.01);
    expect(k("int4_tok", "drift")).toBeLessThan(0.1);
    expect(k("int4_tok", "agree")).toBeGreaterThanOrEqual(124);
    expect(k("int2_tok", "drift")).toBeGreaterThan(0.2);
    expect(k("int2_tok", "agree")).toBeLessThan(110);
    // "the two layouts give about the same drift"
    expect(k("int4_kivi", "drift") / k("int4_tok", "drift")).toBeGreaterThan(
      0.8,
    );
    expect(k("int4_kivi", "drift") / k("int4_tok", "drift")).toBeLessThan(1.25);
    // "4 + 16/8 = 6", "4 + 8/16 = 4.5"; "lands near INT4 per token"
    expect(k("int4_tok", "bits")).toBe(6);
    expect(k("fp4_16", "bits")).toBe(4.5);
    expect(k("fp4_16", "drift") / k("int4_tok", "drift")).toBeLessThan(1.5);
  });
  it("chapter 10", () => {
    // "more than a hundred FP32 multiply-adds"; "15.5 times more"
    expect(n("hz.dram32.pj") / n("dot.pj_fp32")).toBeGreaterThan(100);
    expect(n("hz.mul32.pj") / n("hz.mul8.pj")).toBeCloseTo(15.5, 10);
    expect(n("dot.n")).toBe(32);
  });
});

describe("formatting", () => {
  it("prints exact values, powers of two and scientific notation", () => {
    expect(exact(448)).toBe("448");
    expect(exact(57344)).toBe("57,344");
    expect(exact(0.001953125)).toBe("0.001953125");
    expect(exact(-0)).toBe("−0");
    expect(exact(-1.5)).toBe("−1.5");
    expect(exact(2 ** -40)).toBe("9.09495 × 10⁻¹³");
    expect(exact(Infinity)).toBe("∞");
    expect(pow2Text(pow2(-9))).toBe("2⁻⁹");
    expect(pow2Text(3)).toBe("3");
    expect(asPow2(0)).toBeNull();
    expect(sci(6.103515625e-5, 4)).toBe("6.104 × 10⁻⁵");
    expect(sci(0)).toBe("0");
    expect(trim(0)).toBe("0");
    expect(trim(NaN)).toBe("NaN");
    expect(trim(-Infinity)).toBe("−∞");
    expect(trim(1234.5678)).toBe("1,235");
    expect(trim(-0.25)).toBe("−0.25");
    expect(sup(-12)).toBe("⁻¹²");
    expect(minus("-3")).toBe("−3");
    expect(pct(0.125, 1)).toBe("12.5%");
    expect(fmtDb(sqnrDb(100, 1))).toBe("20.0 dB");
    expect(fmtDb(sqnrDb(1, 0))).toBe("∞ dB");
    expect(fmtUlp(0.25)).toBe("+0.25 ulp");
    expect(fmtUlp(-0.125, 3)).toBe("−0.125 ulp");
    expect(int(8192)).toBe("8,192");
  });
  it("formats every kind", () => {
    const kinds: Fmt[] = [
      "num",
      "exact",
      "pow2",
      "sci",
      "int",
      "pct",
      "db",
      "raw",
    ];
    for (const k of kinds)
      expect(formatValue(0.5, k).length).toBeGreaterThan(0);
    expect(formatValue("overflow", "num")).toBe("overflow");
  });
});
