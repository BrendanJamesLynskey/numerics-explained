/**
 * Captures the README screenshots. Manual run, output committed.
 *
 *   pnpm build && pnpm start   # in another shell
 *   pnpm screenshots           # headless Chromium writes docs/screenshots/*.png
 *
 * Light theme, fixed viewport, reduced motion (nothing plays by itself), and
 * each animation set to a chosen mid-animation frame by its scrub bar, so
 * the pictures are reproducible.
 */
import { mkdirSync } from "node:fs";
import path from "node:path";

import { chromium } from "@playwright/test";

const OUT = path.join(process.cwd(), "docs", "screenshots");
const BASE = process.env.SCREENSHOT_BASE_URL ?? "http://localhost:3000";

type Shot = { name: string; path: string; widget?: string; step?: number };

const SHOTS: Shot[] = [
  { name: "01-landing", path: "/" },
  {
    name: "02-bits",
    path: "/learn/01-bits-to-numbers",
    widget: "bits-widget",
    step: 8,
  },
  {
    name: "03-rounding",
    path: "/learn/02-rounding",
    widget: "rounding-widget",
    step: 40,
  },
  {
    name: "04-stagnation",
    path: "/learn/02-rounding",
    widget: "stagnation-widget",
    step: 96,
  },
  {
    name: "05-accumulation",
    path: "/learn/03-accumulation",
    widget: "sum-widget",
    step: 48,
  },
  {
    name: "06-formats",
    path: "/learn/04-formats-zoo",
    widget: "probe-widget",
    step: 16,
  },
  {
    name: "07-mx",
    path: "/learn/04-formats-zoo",
    widget: "mx-widget",
    step: 14,
  },
  {
    name: "08-granularity",
    path: "/learn/05-quantisation-basics",
    widget: "granularity-widget",
    step: 2,
  },
  {
    name: "09-zero-point",
    path: "/learn/05-quantisation-basics",
    widget: "zeropoint-widget",
    step: 14,
  },
  { name: "10-formats-table", path: "/formats" },
  {
    name: "11-outliers",
    path: "/learn/06-outliers",
    widget: "outlier-widget",
    step: 19,
  },
  {
    name: "12-smoothquant",
    path: "/learn/06-outliers",
    widget: "smooth-widget",
    step: 4,
  },
  { name: "13-gptq", path: "/learn/07-gptq", widget: "gptq-widget", step: 9 },
  {
    name: "14-nf4",
    path: "/learn/08-awq-and-nf4",
    widget: "nf4-widget",
    step: 18,
  },
  {
    name: "15-awq",
    path: "/learn/08-awq-and-nf4",
    widget: "awq-widget",
    step: 3,
  },
  {
    name: "16-tiny-weights",
    path: "/learn/08-awq-and-nf4",
    widget: "tiny-weights-widget",
    step: 24,
  },
  {
    name: "17-kv-cache",
    path: "/learn/09-kv-cache",
    widget: "tiny-kv-widget",
    step: 24,
  },
  {
    name: "18-dot-product",
    path: "/learn/10-hardware",
    widget: "dot-widget",
    step: 31,
  },
];

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    colorScheme: "light",
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  for (const s of SHOTS) {
    await page.goto(BASE + s.path, { waitUntil: "networkidle" });
    const file = path.join(OUT, `${s.name}.png`);
    if (s.widget) {
      const fig = page.getByTestId(s.widget);
      await fig.waitFor();
      if (s.step !== undefined)
        await fig.getByTestId("scrub").fill(String(s.step));
      await fig.screenshot({ path: file });
    } else {
      await page.screenshot({ path: file });
    }
    console.log(`wrote ${path.relative(process.cwd(), file)}`);
  }
  await browser.close();
}

void main();
