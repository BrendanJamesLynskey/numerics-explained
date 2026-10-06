/**
 * Frame tests on the page (visual standard §4): set key frames of every
 * animation and require the caption on screen to be the caption built from
 * the Python reference's state for that frame (tests/fixtures).
 */
import { expect, test, type Locator } from "@playwright/test";

import fx from "../fixtures/num_fixtures.json";
import tfx from "../fixtures/tiny_fixtures.json";

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
  awqCaption,
  dotCaption,
  gptqCaption,
  nf4Caption,
  outlierCaption,
  smoothCaption,
  tinyCaption,
} from "@/lib/num/captions";
import {
  demoBlock,
  demoWeights,
  quantiseMatrix,
  type MxResult,
  type MxStep,
  type ProbeStep,
  type RoundStep,
  type StagStep,
  type SumStep,
  type ZoomStep,
  type ZpStep,
  type DotRun,
  type GptqFrame,
  type Nf4Build,
  type OutlierRun,
  type SmoothStep,
} from "@/lib/num/model";
import { KV_CONFIGS, WEIGHT_CONFIGS, type TinyStep } from "@/lib/num/tiny";

// No autoplay (reduced motion): the test sets each frame itself.
test.use({ contextOptions: { reducedMotion: "reduce" } });

/**
 * Change a parameter and wait until the animation has restarted for it
 * (its data-key changes in the reset itself), so a following scrub cannot
 * be undone by the reset.
 */
async function change(fig: Locator, act: () => Promise<void>): Promise<void> {
  const before = (await fig.getAttribute("data-key")) ?? "";
  await act();
  await expect(fig).not.toHaveAttribute("data-key", before);
}

async function show(fig: Locator, s: number): Promise<void> {
  if ((await fig.getAttribute("data-playing")) === "true")
    await fig.getByTestId("play").click();
  await fig.getByTestId("scrub").fill(String(s));
  await expect(fig).toHaveAttribute("data-step", String(s));
  await expect(fig).toHaveAttribute("data-playing", "false");
}

const caption = (fig: Locator) => fig.getByTestId("caption");

test("bits: zooming in on E4M3, then FP16", async ({ page }) => {
  await page.goto("/learn/01-bits-to-numbers");
  const fig = page.getByTestId("bits-widget");
  const z = fx.zoom.e4m3 as ZoomStep[];
  for (const s of [0, 8, z.length - 1]) {
    await show(fig, s);
    await expect(caption(fig)).toHaveText(zoomCaption("e4m3", z[s]!, s === 0));
  }
  // flipping a bit decodes the new code
  await fig.getByRole("button", { name: "sign bit 7" }).click();
  await expect(fig.getByTestId("decoding")).toContainText("−13");
  await change(fig, () => fig.getByRole("radio", { name: "FP16" }).click());
  const z16 = fx.zoom.fp16 as ZoomStep[];
  for (const s of [0, 20, z16.length - 1]) {
    await show(fig, s);
    await expect(caption(fig)).toHaveText(
      zoomCaption("fp16", z16[s]!, s === 0),
    );
  }
});

test("rounding: just above a value, then anywhere", async ({ page }) => {
  await page.goto("/learn/02-rounding");
  const fig = page.getByTestId("rounding-widget");
  const low = fx.rounding.find((r) => r.fmt === "e4m3" && r.dist === "low")!
    .steps as RoundStep[];
  for (const s of [0, 17, 63]) {
    await show(fig, s);
    await expect(caption(fig)).toHaveText(roundCaption("e4m3", low[s]!, 64));
  }
  await change(fig, () => fig.getByRole("radio", { name: "anywhere" }).click());
  const uni = fx.rounding.find((r) => r.fmt === "e4m3" && r.dist === "uniform")!
    .steps as RoundStep[];
  for (const s of [0, 40]) {
    await show(fig, s);
    await expect(caption(fig)).toHaveText(roundCaption("e4m3", uni[s]!, 64));
  }
});

test("stagnation: FP16 and BF16", async ({ page }) => {
  await page.goto("/learn/02-rounding");
  const fig = page.getByTestId("stagnation-widget");
  const st = fx.stagnation.find((s) => s.fmt === "fp16")!.steps as StagStep[];
  for (const s of [0, 8, 128]) {
    await show(fig, s);
    await expect(caption(fig)).toHaveText(stagCaption("fp16", st[s]!, 1));
  }
  await change(fig, () => fig.getByRole("radio", { name: "BF16" }).click());
  const b = fx.stagnation.find((s) => s.fmt === "bf16")!.steps as StagStep[];
  await show(fig, 100);
  await expect(caption(fig)).toHaveText(stagCaption("bf16", b[100]!, 1));
});

test("sums: FP16 uniform, then BF16", async ({ page }) => {
  await page.goto("/learn/03-accumulation");
  const fig = page.getByTestId("sum-widget");
  const st = fx.sums.find((s) => s.fmt === "fp16" && s.dist === "uniform")!
    .steps as SumStep[];
  // widget step k shows checkpoint k - 1 (step 0 is the empty sum)
  for (const s of [1, 16, 64]) {
    await show(fig, s);
    await expect(caption(fig)).toHaveText(sumCaption("fp16", st[s - 1]!, 8192));
  }
  await change(fig, () => fig.getByRole("radio", { name: "BF16" }).click());
  const b = fx.sums.find((s) => s.fmt === "bf16" && s.dist === "uniform")!
    .steps as SumStep[];
  await show(fig, 64);
  await expect(caption(fig)).toHaveText(sumCaption("bf16", b[63]!, 8192));
});

test("probe: from underflow to overflow", async ({ page }) => {
  await page.goto("/learn/04-formats-zoo");
  const fig = page.getByTestId("probe-widget");
  const st = fx.probe as ProbeStep[];
  for (const s of [0, 16, 35, 44]) {
    await show(fig, s);
    await expect(caption(fig)).toHaveText(probeCaption(st[s]!));
  }
});

test("MX: MXFP4 with an outlier, then MXFP8", async ({ page }) => {
  await page.goto("/learn/04-formats-zoo");
  const fig = page.getByTestId("mx-widget");
  const block = demoBlock("outlier", 5);
  for (const [mid, label] of [
    ["mxfp4", null],
    ["mxfp8_e4m3", "MXFP8 (E4M3)"],
  ] as const) {
    if (label)
      await change(fig, () => fig.getByRole("radio", { name: label }).click());
    const r = fx.mx.find(
      (m) =>
        m.kind === "outlier" &&
        m.seed === 5 &&
        m.mx === mid &&
        m.mode === "rne",
    )!.out as MxResult;
    const st = (fx.mxSteps as Record<string, MxStep[]>)[mid]!;
    for (const s of [0, 2, 3, 34]) {
      await show(fig, s);
      await expect(caption(fig)).toHaveText(mxCaption(mid, block, r, st[s]!));
    }
  }
});

test("granularity: INT4 absmax", async ({ page }) => {
  await page.goto("/learn/05-quantisation-basics");
  const fig = page.getByTestId("granularity-widget");
  const W = demoWeights(16, 64, 5);
  const signal = quantiseMatrix(W, 4, "sym", "tensor").signal;
  const st = fx.granularity.find((g) => g.bits === 4 && g.scheme === "sym")!
    .steps as GranSummary[];
  for (const s of [0, 2, 4]) {
    await show(fig, s);
    await expect(caption(fig)).toHaveText(
      granCaption(st[s]!, 4, "sym", signal, 1024),
    );
  }
});

test("zero point: INT4", async ({ page }) => {
  await page.goto("/learn/05-quantisation-basics");
  const fig = page.getByTestId("zeropoint-widget");
  const z = fx.zeropoint.find((q) => q.bits === 4)!.out;
  const xs = fx.demo.activations.values;
  for (const s of [0, 1, 2, 25]) {
    await show(fig, s);
    await expect(caption(fig)).toHaveText(
      zpCaption(xs, z.sym, z.asym, z.steps[s] as ZpStep, 4),
    );
  }
});

test("outliers: per tensor, the scan, the decomposition", async ({ page }) => {
  await page.goto("/learn/06-outliers");
  const fig = page.getByTestId("outlier-widget");
  const r = fx.outlier.run as OutlierRun;
  for (const s of [0, 1, 6, 14, r.steps.length - 1]) {
    await show(fig, s);
    await expect(caption(fig)).toHaveText(outlierCaption(r, s, 32));
  }
});

test("SmoothQuant: alpha 0, 1/2 and 1", async ({ page }) => {
  await page.goto("/learn/06-outliers");
  const fig = page.getByTestId("smooth-widget");
  const st = fx.smooth as SmoothStep[];
  const base = fx.outlier.run.err_tensor;
  for (const s of [0, 4, 8]) {
    await show(fig, s);
    await expect(caption(fig)).toHaveText(smoothCaption(st[s]!, base));
  }
});

test("GPTQ: INT4, columns -1, 0, 7 and 15", async ({ page }) => {
  await page.goto("/learn/07-gptq");
  const fig = page.getByTestId("gptq-widget");
  const frames = fx.gptqSteps.bits4.frames as GptqFrame[];
  for (const [s, k] of [
    [0, 0],
    [1, 1],
    [8, 2],
    [16, 3],
  ] as const) {
    await show(fig, s);
    await expect(caption(fig)).toHaveText(gptqCaption(frames[k]!, 4, 16));
  }
});

test("NF4: derivation steps", async ({ page }) => {
  await page.goto("/learn/08-awq-and-nf4");
  const fig = page.getByTestId("nf4-widget");
  const b = fx.nf4Build as Nf4Build;
  for (const s of [0, 3, 12, 18]) {
    await show(fig, s);
    await expect(caption(fig)).toHaveText(nf4Caption(b, s));
  }
});

test("AWQ: INT3 then INT4", async ({ page }) => {
  await page.goto("/learn/08-awq-and-nf4");
  const fig = page.getByTestId("awq-widget");
  for (const [bits, py] of [
    [3, fx.awq],
    [4, fx.awq4],
  ] as const) {
    if (bits === 4)
      await change(fig, () => fig.getByRole("radio", { name: "INT4" }).click());
    for (const k of [0, py.best, 8]) {
      await show(fig, k);
      await expect(caption(fig)).toHaveText(
        awqCaption(
          { k, alpha: k / 8, err: py.errs[k]! },
          py.best,
          py.errs[0]!,
          bits,
        ),
      );
    }
  }
});

test("tiny model: NF4 weights, then INT2 KV cache", async ({ page }) => {
  await page.goto("/learn/08-awq-and-nf4");
  const fw = page.getByTestId("tiny-weights-widget");
  const w = tfx.runs.weights.nf4.steps0 as TinyStep[];
  for (const s of [0, 15, 31]) {
    await show(fw, s);
    await expect(caption(fw)).toHaveText(
      tinyCaption(w[s]!, 32, WEIGHT_CONFIGS.nf4.label),
    );
  }
  await page.goto("/learn/09-kv-cache");
  const fk = page.getByTestId("tiny-kv-widget");
  await change(fk, async () => {
    await fk.getByTestId("format").selectOption("int2_tok");
  });
  const k = tfx.runs.kv.int2_tok.steps0 as TinyStep[];
  for (const s of [0, 15, 31]) {
    await show(fk, s);
    await expect(caption(fk)).toHaveText(
      tinyCaption(k[s]!, 32, KV_CONFIGS.int2_tok.label),
    );
  }
});

test("dot product: pair 1 then pair 2", async ({ page }) => {
  await page.goto("/learn/10-hardware");
  const fig = page.getByTestId("dot-widget");
  for (const [key, label] of [
    ["5-6", null],
    ["7-8", "pair 2"],
  ] as const) {
    if (label)
      await change(fig, () => fig.getByRole("radio", { name: label }).click());
    const r = fx.dot[key] as DotRun;
    for (const s of [0, 15, 31]) {
      await show(fig, s);
      await expect(caption(fig)).toHaveText(dotCaption(r, s));
    }
  }
});
