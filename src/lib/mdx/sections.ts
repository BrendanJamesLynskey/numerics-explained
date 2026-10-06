/**
 * Chapter catalogue + filesystem loader for /learn content.
 *
 * MDX sources live under `/content/chapters/`, one per mechanism, each built
 * around an animation. Their slugs and order are defined here (single source
 * of truth); the `[slug]` route validates incoming params against this list
 * before reading from disk. Same shape as the companion sites'
 * `src/lib/mdx/sections.ts`.
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";

export const SECTIONS = [
  {
    slug: "01-bits-to-numbers",
    title: "Bits to numbers",
    summary:
      "Sign, exponent and mantissa: click the bits of a floating-point code and watch its value move, then zoom in and see the values crowd towards zero.",
  },
  {
    slug: "02-rounding",
    title: "Rounding",
    summary:
      "Round to nearest, ties to even, against stochastic rounding: one is the most accurate on each value, the other the only one that is right on average.",
  },
  {
    slug: "03-accumulation",
    title: "Accumulation error",
    summary:
      "Summing 8,192 numbers in FP16 stalls at 2,048. FP32, Kahan's compensated sum and pairwise summation, raced live.",
  },
  {
    slug: "04-formats-zoo",
    title: "The formats zoo",
    summary:
      "FP8 E4M3 against E5M2, BF16 against FP16: range traded for precision. Then MX: 32 small numbers sharing one power-of-two scale.",
  },
  {
    slug: "05-quantisation-basics",
    title: "Quantisation basics",
    summary:
      "Integers and a scale: absmax and zero-point, and why one scale per group of weights beats one per tensor, on a weight heat map.",
  },
] as const;

export type SectionSlug = (typeof SECTIONS)[number]["slug"];

const SLUG_SET = new Set<string>(SECTIONS.map((s) => s.slug));

export function isValidSlug(slug: string): slug is SectionSlug {
  return SLUG_SET.has(slug);
}

export function getSectionMeta(slug: SectionSlug): (typeof SECTIONS)[number] {
  return SECTIONS.find((s) => s.slug === slug) ?? SECTIONS[0];
}

/** Read the raw MDX source for a chapter, or `null` if it doesn't exist. */
export async function readSectionMdx(
  slug: SectionSlug,
): Promise<string | null> {
  const path = join(process.cwd(), "content", "chapters", `${slug}.mdx`);
  try {
    return await readFile(path, "utf-8");
  } catch {
    return null;
  }
}
