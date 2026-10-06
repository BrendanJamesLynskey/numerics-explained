/**
 * Frame tests on the page (visual standard §4): set key frames of every
 * animation and require the caption on screen to be the caption built from
 * the Python reference's state for that frame (tests/fixtures).
 */
import { expect, test, type Locator } from "@playwright/test";

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
} from "@/lib/num/model";

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
