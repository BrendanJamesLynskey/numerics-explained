/**
 * /about: what the site is, what the numerics library implements and how
 * it is checked, what is illustrative, how the animations are driven, and
 * where the design came from. Server Component, static.
 */
import Link from "next/link";

import {
  ARCHITECTURES_URL,
  DECODER_URL,
  GITHUB_URL,
  INFERENCE_URL,
  KERNELS_URL,
  LOCAL_LLM_HUB,
  TPU_HUB,
  repoFile,
} from "@/lib/site";

export const metadata = {
  title: "About",
  description:
    "What Numerics Explained's library implements, how it is checked bit for bit against numpy, ml_dtypes and the specifications, and what is illustrative.",
};

const A =
  "focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300";

export default function AboutPage(): JSX.Element {
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        /about
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        About this site
      </h1>
      <div className="mdx-content mt-6">
        <p>
          Numerics Explained is about the numbers inside a model: how a format
          spends its bits, how rounding and accumulation go wrong, and the
          scaling and quantisation tricks that let 8-, 6- and 4-bit numbers
          work. Each chapter is built around an animation. It is the fifth of a
          family of companion sites, with the{" "}
          <a href={DECODER_URL} className={A}>
            Transformer Decoder Explainer
          </a>
          ,{" "}
          <a href={INFERENCE_URL} className={A}>
            LLM Inference Explained
          </a>
          ,{" "}
          <a href={ARCHITECTURES_URL} className={A}>
            LLM Architectures Explained
          </a>{" "}
          and{" "}
          <a href={KERNELS_URL} className={A}>
            GPU Kernels Explained
          </a>
          . The chapters link the matching slides of the{" "}
          <a href={LOCAL_LLM_HUB} className={A}>
            Local LLM hosting
          </a>{" "}
          and{" "}
          <a href={TPU_HUB} className={A}>
            Google TPU
          </a>{" "}
          series.
        </p>

        <h2>The numerics library</h2>
        <p>
          <a href={repoFile("reference/numerics.py")} className={A}>
            reference/numerics.py
          </a>{" "}
          is written in plain Python, so that every step is one IEEE double
          operation, and{" "}
          <a href={repoFile("src/lib/num/model.ts")} className={A}>
            src/lib/num/model.ts
          </a>{" "}
          repeats it operation for operation. It implements:
        </p>
        <ul>
          <li>
            encoding and decoding for FP32, FP16, BF16, FP8 E4M3 and E5M2, FP6
            E3M2 and E2M3 and FP4 E2M1, in every rounding mode IEEE 754-2019
            defines plus stochastic rounding, with IEEE overflow or the OCP
            saturating mode (see{" "}
            <Link href="/formats" className={A}>
              the formats
            </Link>
            );
          </li>
          <li>
            the MX block formats (a shared E8M0 power-of-two scale per 32
            elements, by section 6.3 of the MX specification) and NF4;
          </li>
          <li>
            integer quantisation, symmetric (absmax) and asymmetric
            (zero-point), per tensor, per channel and per group;
          </li>
          <li>
            summation in a format (naive, Kahan, pairwise), and GPTQ, AWQ-style
            scaling and SmoothQuant on small layers;
          </li>
          <li>
            LLM.int8()&apos;s mixed-precision decomposition, NF4&apos;s
            derivation from normal quantiles, and one dot product in FP32, FP16,
            INT8 and MXFP4 with the energy of each operation.
          </li>
        </ul>

        <h2>The tiny transformer</h2>
        <p>
          Chapters 8 and 9 quantise a real network: the{" "}
          <a href={DECODER_URL} className={A}>
            Transformer Decoder Explainer
          </a>
          &apos;s tiny model (two blocks, 16 wide, a 64-character vocabulary),
          whose TypeScript is vendored unchanged into this site, with a
          plain-Python port in{" "}
          <a href={repoFile("reference/tiny.py")} className={A}>
            reference/tiny.py
          </a>
          . It runs in your browser twice, as is and with its weights or its KV
          cache quantised, and the pages compare the two over four prompts. Its
          weights are random, rescaled from the explainer&apos;s so that its
          blocks matter; its text is gibberish, its arithmetic real. A test
          proves the hooked forward pass equals the explainer&apos;s exactly,
          and the Python and TypeScript runs agree to a relative 10⁻¹² (they
          share exp, log, sin, cos and tanh, which each language may round
          differently in the last place) and exactly in every top prediction.
        </p>

        <h2>How it is checked</h2>
        <ul>
          <li>
            <strong>Against numpy and ml_dtypes.</strong>{" "}
            <a href={repoFile("tests/python/test_numerics.py")} className={A}>
              tests/python/test_numerics.py
            </a>{" "}
            decodes every code of every format up to 16 bits and compares it
            with numpy (FP16) and ml_dtypes (BF16, FP8, FP6, FP4); encodes tens
            of thousands of values, ties and overflows included, and compares
            the codes; and checks the integer quantisers, the three summations
            and GPTQ against numpy implementations.
          </li>
          <li>
            <strong>Against an exact oracle.</strong> For every rounding mode
            and saturation mode, the encoder is compared with an oracle that
            lists the format&apos;s values and decides with exact rational
            arithmetic.
          </li>
          <li>
            <strong>Against the specifications.</strong> The tables of the OCP
            FP8 and MX specifications (largest, smallest normal and subnormal
            values, NaN and infinity codes, the conversion cases of FP8&apos;s
            Table 3), worked MX conversions, and bitsandbytes&apos; NF4 decision
            thresholds.
          </li>
          <li>
            <strong>Exact parity.</strong>{" "}
            <a href={repoFile("scripts/make_fixtures.py")} className={A}>
              scripts/make_fixtures.py
            </a>{" "}
            writes the reference&apos;s results, and the unit tests require the
            TypeScript port to reproduce all of them exactly: every code of
            every format up to 16 bits, and every encode case in all six modes,
            through SHA-256 digests of the bit patterns. CI fails if the
            fixtures are out of date.
          </li>
          <li>
            <strong>Animations from the model.</strong> Every animation draws a
            sequence of states the library computes; a frame is a pure function
            of one state. The tests set chosen frames of every animation and
            require the caption to match the caption built from the Python
            reference&apos;s state for that frame.
          </li>
          <li>
            <strong>Numbers in the prose</strong> are printed from the library
            when the page is built, not typed.
          </li>
        </ul>

        <h2>What is illustrative</h2>
        <ul>
          <li>
            The data sets (the weight matrix, the activations, the MX blocks,
            the vectors summed) are generated from a seeded integer random
            number generator; the &quot;normal&quot; values are sums of 12
            uniforms. They show the mechanisms, not a particular model&apos;s
            numbers.
          </li>
          <li>
            Bits per weight count FP16 scales (and, for zero-point, a zero point
            of the weight&apos;s width); real formats store their scales in
            several ways.
          </li>
          <li>
            The FP16 and BF16 sums round after every addition, as a scalar loop
            does; GPU kernels usually accumulate in FP32 and in their own order.
          </li>
          <li>
            The layers of chapters 6 to 8 are generated; the tiny
            transformer&apos;s weights are random; the KV-cache layouts are
            simplified (one scale over the whole prompt for per-channel keys);
            the energy figures are for a 45 nm process.
          </li>
        </ul>

        <h2>The animations</h2>
        <p>
          Every animation has play and pause, step back and forward, a scrub
          bar, speeds from 0.25× to 4× and reset; with the animation focused,
          Space plays or pauses and the arrow keys step. Each step has a
          one-line caption, also announced to screen readers. With{" "}
          <em>reduce motion</em> set in your system, nothing plays by itself.
          Animations pause when scrolled out of view. Colours come from Okabe
          and Ito&apos;s colour-blind-safe palette, the same in light and dark
          mode: one colour per bit field (sign purple, exponent sky blue,
          mantissa green, shared scale orange) and one per method (nearest-even
          blue, stochastic orange); an error is a warning colour.
        </p>

        <h2>Source</h2>
        <p>
          The code, the library and the tests are on{" "}
          <a href={GITHUB_URL} className={A}>
            GitHub
          </a>{" "}
          (MIT licence). The design system is copied from the companion sites;
          the README records where each piece came from.
        </p>
      </div>
    </main>
  );
}
