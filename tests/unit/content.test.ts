/**
 * The chapters' MDX: each opens with its animation; every ```ts block is
 * cut from the library, src/lib/num/model.ts or tiny.ts (whitespace-collapsed, because Prettier reformats
 * MDX code blocks); every equation compiles in KaTeX; internal links point
 * at real pages; deck links point at the owner's slide decks with a slide
 * anchor; every arXiv link is one of the ids checked at export.arxiv.org
 * (scripts/check_links.py records the titles).
 */
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";

import katex from "katex";
import { describe, expect, it } from "vitest";

import { SECTIONS } from "@/lib/mdx/sections";

const ROOT = path.join(__dirname, "../..");
const DIR = path.join(ROOT, "content/chapters");
const FILES = readdirSync(DIR).filter((f) => f.endsWith(".mdx"));
// whitespace and trailing commas collapsed: Prettier formats MDX code
// blocks and source files at different widths
const squash = (s: string) =>
  s
    .replace(/\s+/g, " ")
    .replace(/,\s*([)\]}])/g, "$1")
    .replace(/([([{])\s+/g, "$1")
    .replace(/\s+([)\]}])/g, "$1")
    .trim();

/** Checked at export.arxiv.org (title and authors), 2026-10-06. */
export const ARXIV = new Set([
  "1502.02551",
  "1710.03740",
  "1712.05877",
  "1905.12322",
  "2106.08295",
  "2209.05433",
  "2310.10537",
  "2305.14314",
  "2208.07339",
  "2211.10438",
  "2210.17323",
  "2208.11580",
  "2306.00978",
  "2402.02750",
  "2401.18079",
  "2609.19969",
  "2103.13630",
]);

describe("chapter files", () => {
  it("there is one per section, in order", () => {
    expect(FILES.sort()).toEqual(SECTIONS.map((s) => `${s.slug}.mdx`));
  });
});

for (const f of FILES) {
  const src = readFileSync(path.join(DIR, f), "utf8");
  describe(f, () => {
    it("opens with its animation (the hero comes before any prose)", () => {
      expect(src.trimStart()).toMatch(/^<[A-Z][a-zA-Z0-9]+Widget[ >\n]/);
    });

    it("cuts every TypeScript block from the library source", () => {
      const model = squash(
        ["model.ts", "tiny.ts"]
          .map((f) => readFileSync(path.join(ROOT, "src/lib/num", f), "utf8"))
          .join("\n"),
      );
      const blocks = [...src.matchAll(/```ts\n([\s\S]*?)```/g)].map(
        (m) => m[1]!,
      );
      expect(blocks.length).toBeGreaterThan(0);
      for (const b of blocks) expect(model, b).toContain(squash(b));
    });

    it("has no code in a language the tests do not check", () => {
      for (const m of src.matchAll(/```([a-z]*)\n/g))
        expect(["ts", ""]).toContain(m[1]);
    });

    it("compiles every animation equation", () => {
      for (const m of src.matchAll(/tex="([^"]+)"/g)) {
        expect(() =>
          katex.renderToString(m[1]!, {
            displayMode: true,
            throwOnError: true,
            strict: "ignore",
            trust: (ctx) => ctx.command === "\\htmlClass",
          }),
        ).not.toThrow();
      }
    });

    it("links only to real pages, deck slides and checked papers", () => {
      const slugs = new Set<string>(SECTIONS.map((s) => s.slug));
      for (const m of src.matchAll(/\]\((\/[^)]+)\)/g)) {
        const href = m[1]!;
        if (href.startsWith("/learn/"))
          expect(slugs.has(href.slice(7)), href).toBe(true);
        else expect(["/formats", "/about", "/learn"]).toContain(href);
      }
      for (const m of src.matchAll(
        /https:\/\/brendanjameslynskey\.github\.io\/([A-Za-z0-9_]+)\/(#[a-z0-9-]+)?/g,
      )) {
        if (m[1] === "Hardware_Aware_Quantisation") continue;
        expect(m[1], m[0]).toMatch(
          /^(Local_LLM_\d\d_|Google_TPU_\d\d_|Linear_Algebra_AI_\d\d_|NVIDIA_GPU_\d\d_|Arch_\d\d_|AI_MMUL_Unit$)/,
        );
        expect(m[2], m[0]).toMatch(/^#(slide-\d\d|s\d+)$/);
      }
      for (const m of src.matchAll(/arxiv\.org\/abs\/([0-9.]+)/g))
        expect(ARXIV.has(m[1]!), m[1]).toBe(true);
    });
  });
}
