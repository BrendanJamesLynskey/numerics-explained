import Link from "next/link";

import { V } from "@/components/mdx/V";
import { ValueStrips } from "@/components/viz/ValueStrips";
import { SECTIONS } from "@/lib/mdx/sections";
import {
  ARCHITECTURES_URL,
  DECODER_URL,
  INFERENCE_URL,
  KERNELS_URL,
  SILICON_URL,
  TRADEOFFS_URL,
} from "@/lib/site";

const A =
  "focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent";

/**
 * Landing page: what the site is, a picture of where a format's values
 * are, and the chapters. Server Component with no client JavaScript of its
 * own.
 */
export default function HomePage(): JSX.Element {
  return (
    <main className="mx-auto max-w-5xl px-6 py-12 sm:py-20">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        Numerics Explained
      </p>
      <h1 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
        How few bits can a number have?
      </h1>
      <div className="mt-8 grid items-center gap-8 md:grid-cols-[1fr_minmax(0,22rem)]">
        <div>
          <p className="max-w-2xl text-lg text-neutral-600 dark:text-neutral-300">
            Models are trained in 16-bit formats and served in 8, 6 or 4. An FP8
            E4M3 number has <V of="e4m3.digits" fmt="int" /> significant bits
            and nothing above <V of="e4m3.max" fmt="exact" />; FP4 has{" "}
            <V of="e2m1.digits" fmt="int" /> and stops at{" "}
            <V of="e2m1.max" fmt="exact" />. This site is about what that costs
            and the tricks that make it work: rounding, accumulating in wider
            formats, scales shared by blocks of numbers, and quantisers that
            choose their integers with care.
          </p>
          <p className="mt-4 max-w-2xl text-neutral-600 dark:text-neutral-300">
            Each chapter is built around an animation, and every frame is
            computed by a small numerics library that reproduces numpy and
            ml_dtypes bit for bit and follows the IEEE 754 and OCP FP8 and MX
            specifications.
          </p>
        </div>
        <figure className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
          <ValueStrips />
          <figcaption className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
            Every value of three small formats, to scale: the gaps double at
            each power of two.
          </figcaption>
        </figure>
      </div>
      <nav
        aria-label="Chapters"
        className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
      >
        {SECTIONS.map((s, i) => (
          <Link
            key={s.slug}
            href={`/learn/${s.slug}`}
            className="focus-ring group rounded-lg border border-neutral-200 p-5 hover:border-accent dark:border-neutral-800 dark:hover:border-indigo-400"
          >
            <p className="font-mono text-xs text-neutral-500 dark:text-neutral-400">
              {String(i + 1).padStart(2, "0")}
            </p>
            <h2 className="mt-1 font-semibold">{s.title}</h2>
            <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-400">
              {s.summary}
            </p>
          </Link>
        ))}
      </nav>
      <p className="mt-6 text-sm text-neutral-600 dark:text-neutral-400">
        Every format, its constants and the specification that defines it:{" "}
        <Link href="/formats" className={A}>
          the formats
        </Link>
        .
      </p>
      <p className="mt-12 text-sm text-neutral-600 dark:text-neutral-400">
        Part of a family of companion sites: the{" "}
        <a href={DECODER_URL} className={A}>
          Transformer Decoder Explainer
        </a>{" "}
        (one forward pass),{" "}
        <a href={INFERENCE_URL} className={A}>
          LLM Inference Explained
        </a>{" "}
        (serving it),{" "}
        <a href={ARCHITECTURES_URL} className={A}>
          LLM Architectures Explained
        </a>{" "}
        (how the models differ),{" "}
        <a href={KERNELS_URL} className={A}>
          GPU Kernels Explained
        </a>{" "}
        (how a GPU runs the maths),{" "}
        <a href={SILICON_URL} className={A}>
          Systolic Arrays Explained
        </a>{" "}
        (the matrix hardware) and{" "}
        <a href={TRADEOFFS_URL} className={A}>
          Inference Trade-offs Explained
        </a>{" "}
        (which serving lever helps which metric). This site is about the numbers
        themselves. How it was built, and how to check it:{" "}
        <Link href="/about" className={A}>
          about
        </Link>
        .
      </p>
    </main>
  );
}
