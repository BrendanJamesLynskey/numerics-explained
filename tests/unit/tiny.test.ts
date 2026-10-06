/**
 * The tiny model: the vendored explainer forward pass is reproduced exactly
 * by the hooked one, and the runs match the Python reference (relative
 * 1e-12 for logits and drifts, which pass through exp, log, sin, cos and
 * tanh; exactly for top tokens and agreement counts).
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/tiny_fixtures.json";

import {
  CALIB,
  KV_ORDER,
  PROMPTS,
  WEIGHT_ORDER,
  calibration,
  encodeText,
  forward,
  fp4E4m3Scaled,
  margins,
  modelWeights,
  pendingTasks,
  quantisedWeights,
  clearCaches,
  run,
  tokenLabel,
  type TinyStep,
} from "@/lib/num/tiny";
import { forwardTyped } from "@/lib/transformer/model";
import { TINY_CONFIG } from "@/lib/num/tiny";

const REL = 1e-12;

function closeArr(a: readonly number[], b: readonly number[], rel = REL) {
  expect(a.length).toBe(b.length);
  a.forEach((v, i) =>
    expect(Math.abs(v - b[i]!)).toBeLessThanOrEqual(
      rel * Math.max(1, Math.abs(b[i]!)),
    ),
  );
}

function closeStep(a: TinyStep, b: TinyStep) {
  expect([a.t, a.token, a.top, a.top_ref, a.top_q, a.agree, a.agreed]).toEqual([
    b.t,
    b.token,
    b.top,
    b.top_ref,
    b.top_q,
    b.agree,
    b.agreed,
  ]);
  closeArr(
    [...a.ref, ...a.q, a.q_top_logit, a.rms],
    [...b.ref, ...b.q, b.q_top_logit, b.rms],
  );
}

describe("tiny model", () => {
  const w = modelWeights();
  it("prompts are a full context each", () => {
    for (const p of [...PROMPTS, ...CALIB])
      expect(p.length).toBe(TINY_CONFIG.seq_len);
    expect(encodeText("a?")).toEqual([0, 40]);
    expect(tokenLabel(encodeText(" ")[0]!)).toBe("␣");
    expect(tokenLabel(encodeText("\n")[0]!)).toBe("↵");
    expect(tokenLabel(99)).toBe("?");
  });
  it("weights match the reference", () => {
    closeArr(w.tok_emb[0]!, fx.emb0);
    closeArr(w.blocks[0]!.attn.W_q[0]!, fx.wq0);
    closeArr(w.blocks[1]!.ffn.W2.at(-1)!, fx.w2last);
  });
  it("the hooked forward pass is the explainer's, exactly", () => {
    const ids = encodeText(PROMPTS[0]!);
    expect(forward(ids, w)).toEqual(forwardTyped(ids, TINY_CONFIG, w).logits);
  });
  it("reference logits", () => {
    const ref = forward(encodeText(PROMPTS[0]!), w);
    ref.forEach((r, t) => closeArr(r, fx.ref0[t]!));
    const m = margins(PROMPTS.map((p) => forward(encodeText(p), w)));
    closeArr(
      [m.median_gap, m.min_gap, m.rms],
      [fx.margins.median_gap, fx.margins.min_gap, fx.margins.rms],
    );
  });
  it("calibration inputs", () => {
    const c = calibration(w);
    c.W2[1]!.slice(0, 2).forEach((r, i) => closeArr(r, fx.calibW2[i]!));
  });
  for (const [target, order] of [
    ["weights", WEIGHT_ORDER],
    ["kv", KV_ORDER],
  ] as const)
    for (const cfg of order)
      it(`${target} ${cfg}`, () => {
        const r = run(target, cfg);
        const py = (
          fx.runs[target] as Record<string, (typeof fx.runs.kv)["fp16"]>
        )[cfg]!;
        expect([r.summary.agree, r.summary.n]).toEqual([
          py.summary.agree,
          py.summary.n,
        ]);
        closeArr(
          [r.summary.drift, r.summary.max_change],
          [py.summary.drift, py.summary.max_change],
        );
        expect(r.steps.map((st) => st.filter((s) => s.agree).length)).toEqual(
          py.agree,
        );
        r.steps[0]!.forEach((s, i) => closeStep(s, py.steps0[i] as TinyStep));
      });
  it("runs in pieces, each computed once", () => {
    clearCaches();
    const tasks = pendingTasks("weights", "gptq3_ch");
    // 4 reference passes, 4 calibration passes, the weights, 4 quantised passes
    expect(tasks).toHaveLength(13);
    for (const t of tasks) t();
    expect(pendingTasks("weights", "gptq3_ch")).toEqual([]);
    expect(pendingTasks("kv", "fp16")).toHaveLength(4);
    expect(pendingTasks("weights", "nf4")).toHaveLength(5);
    expect(quantisedWeights("nf4")).toBe(quantisedWeights("nf4"));
  });
  it("FP4 KV: an all-zero block gets the smallest scale", () => {
    expect(fp4E4m3Scaled([[0, 0, 0, 0]], 4)).toEqual([[0, 0, 0, 0]]);
  });
});
