/**
 * Frame tests for chapters 6-10 (visual standard §4): for key frames of
 * every animation, the state the site draws equals the Python reference's
 * state for that frame (from the fixtures), and the caption built from the
 * reference's state is the caption the site shows. The NF4 derivation and
 * the tiny model pass through exp and friends, so their states are matched
 * to a relative 1e-12 and their captions exactly.
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/num_fixtures.json";
import tfx from "../fixtures/tiny_fixtures.json";

import { digest } from "./helpers/digest";

import {
  awqCaption,
  dotCaption,
  gptqCaption,
  nf4Caption,
  outlierCaption,
  smoothCaption,
  tinyCaption,
} from "@/lib/num/captions";
import {
  awqSearch,
  demoBlock,
  demoLayer,
  demoOutlierLayer,
  dotSteps,
  gptqSteps,
  nf4Build,
  outlierSteps,
  smoothSteps,
  w8a8Error,
  type DotRun,
  type GptqFrame,
  type Nf4Build,
  type OutlierRun,
  type SmoothStep,
} from "@/lib/num/model";
import {
  KV_CONFIGS,
  KV_ORDER,
  WEIGHT_CONFIGS,
  WEIGHT_ORDER,
  run,
  type TinyStep,
} from "@/lib/num/tiny";
import { DEMO } from "@/lib/num/values";

const ol = demoOutlierLayer(...DEMO.outlier);
const lay = demoLayer(...DEMO.layer);

describe("chapter 6: outlier frames", () => {
  const ts = outlierSteps(ol.W, ol.X);
  const py = fx.outlier.run as OutlierRun;
  for (const i of [0, 1, 2, 3 + 3, 3 + 11, ts.steps.length - 1])
    it(`step ${i}`, () => {
      expect(ts.steps[i]).toEqual(py.steps[i]);
      expect(outlierCaption(ts, i, 32)).toBe(outlierCaption(py, i, 32));
    });
});

describe("chapter 6: SmoothQuant frames", () => {
  const ts = smoothSteps(ol.W, ol.X);
  const py = fx.smooth as SmoothStep[];
  const base = w8a8Error(ol.W, ol.X).err;
  for (const k of [0, 2, 4, 6, 8])
    it(`alpha ${k}/8`, () => {
      expect(ts[k]).toEqual(py[k]);
      expect(smoothCaption(ts[k]!, base)).toBe(smoothCaption(py[k]!, base));
    });
});

describe("chapter 7: GPTQ frames", () => {
  const ts = gptqSteps(lay.W, lay.X, 4).steps;
  const frames = fx.gptqSteps.bits4.frames as GptqFrame[];
  [0, 1, 8, 16].forEach((i, k) =>
    it(`column ${i - 1}`, () => {
      expect(ts[i]).toEqual(frames[k]);
      expect(gptqCaption(ts[i]!, 4, 16)).toBe(gptqCaption(frames[k]!, 4, 16));
    }),
  );
  it("3-bit run", () =>
    expect(digest(gptqSteps(lay.W, lay.X, 3).steps)).toBe(
      fx.gptqSteps.bits3.digest,
    ));
});

describe("chapter 8: AWQ frames", () => {
  for (const [bits, py] of [
    [3, fx.awq],
    [4, fx.awq4],
  ] as const) {
    const a = awqSearch(lay.W, lay.X, bits, 8);
    it(`${bits}-bit: every alpha's error and scales`, () => {
      expect(a.best).toBe(py.best);
      expect(a.results.map((r) => r.err)).toEqual(py.errs);
      expect(digest(a.results.map((r) => r.scales))).toBe(py.scales);
    });
    for (const k of [0, py.best, 8])
      it(`${bits}-bit alpha ${k}/8 caption`, () => {
        const r = a.results[k]!;
        const p = { k, alpha: k / 8, err: py.errs[k]! };
        expect(awqCaption(r, a.best, a.results[0]!.err, bits)).toBe(
          awqCaption(p, py.best, py.errs[0]!, bits),
        );
      });
  }
});

describe("chapter 8: NF4 frames", () => {
  const ts = nf4Build();
  const py = fx.nf4Build as Nf4Build;
  for (const i of [0, 3, 12, 16, 17, 18])
    it(`step ${i}`, () => {
      expect(ts.steps[i]).toEqual(py.steps[i]);
      expect(nf4Caption(ts, i)).toBe(nf4Caption(py, i));
    });
});

describe("chapters 8 and 9: tiny-model frames", () => {
  for (const [target, order, cfgs] of [
    ["weights", WEIGHT_ORDER, WEIGHT_CONFIGS],
    ["kv", KV_ORDER, KV_CONFIGS],
  ] as const)
    for (const c of order) {
      const label = (cfgs as Record<string, { label: string }>)[c]!.label;
      const ts = run(target, c).steps[0]!;
      const py = (tfx.runs[target] as Record<string, { steps0: TinyStep[] }>)[
        c
      ]!.steps0;
      it(`${target} ${c}: positions 0, 15, 31`, () => {
        for (const t of [0, 15, 31]) {
          expect([ts[t]!.top, ts[t]!.agree, ts[t]!.agreed]).toEqual([
            py[t]!.top,
            py[t]!.agree,
            py[t]!.agreed,
          ]);
          expect(tinyCaption(ts[t]!, 32, label)).toBe(
            tinyCaption(py[t]!, 32, label),
          );
        }
      });
    }
});

describe("chapter 10: dot-product frames", () => {
  for (const [s1, s2] of [
    [5, 6],
    [7, 8],
  ] as const) {
    const ts = dotSteps(demoBlock("normal", s1), demoBlock("normal", s2));
    const py = fx.dot[`${s1}-${s2}` as keyof typeof fx.dot] as DotRun;
    for (const i of [0, 15, 31])
      it(`pair ${s1}-${s2}, product ${i}`, () => {
        expect(ts.steps[i]).toEqual(py.steps[i]);
        expect(dotCaption(ts, i)).toBe(dotCaption(py, i));
      });
  }
});
