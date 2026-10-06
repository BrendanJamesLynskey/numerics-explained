/**
 * /formats: every format the library implements, with the constants the
 * tested model derives from its definition and the source that defines it;
 * the MX block formats; the NF4 code book; the rounding modes. Server
 * Component, static. Every number here is computed by src/lib/num/model.ts.
 */
import Link from "next/link";

import data from "@/data/formats.json";
import { exact, int, pow2Text } from "@/lib/format";
import {
  FORMATS,
  FORMAT_ORDER,
  INFO,
  MODES,
  MODE_NAMES,
  MX_FORMATS,
  MX_ORDER,
  NF4,
  NF4_MID,
} from "@/lib/num/model";
import { repoFile } from "@/lib/site";

export const metadata = {
  title: "Formats",
  description:
    "Every number format the site implements (FP32, FP16, BF16, FP8 E4M3 and E5M2, FP6, FP4, MX, NF4) with its constants and its defining source.",
};

const SOURCES = data.sources as Record<string, { title: string; url: string }>;

const A =
  "focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300";
const TH =
  "border-b border-neutral-300 px-2 py-1.5 text-left font-semibold dark:border-neutral-700";
const TD =
  "border-b border-neutral-200 px-2 py-1.5 align-top dark:border-neutral-800";

const KIND: Record<string, string> = {
  ieee: "±∞ and NaN (all-ones exponent)",
  fn: "NaN only (S.1111.111); no ∞",
  finite: "none: every code is a number",
};

const MODE_TEXT: Record<string, string> = {
  rne: "to the nearer neighbour; a tie goes to the one whose last mantissa bit is 0 (IEEE 754's default, the only mode OCP requires)",
  rna: "to the nearer neighbour; a tie goes away from zero",
  rtz: "always towards zero (truncation)",
  rup: "always towards +∞",
  rdn: "always towards −∞",
  sr: "up with probability equal to the distance above the lower neighbour, in ulps (not an IEEE mode; MX allows other modes)",
};

function Table({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <div
      role="region"
      aria-label={label}
      tabIndex={0}
      className="focus-ring my-4 max-w-full overflow-x-auto rounded"
    >
      <table className="text-sm">{children}</table>
    </div>
  );
}

export default function FormatsPage(): JSX.Element {
  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        /formats
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        The formats, and where they are defined
      </h1>
      <p className="mt-4 max-w-3xl text-neutral-600 dark:text-neutral-300">
        Each format is four numbers (exponent bits, mantissa bits, bias, and how
        it spends the all-ones exponent). Everything else in these tables is
        derived from them by the{" "}
        <a href={repoFile("src/lib/num/model.ts")} className={A}>
          tested library
        </a>{" "}
        and agrees with its source&apos;s own table. The library decodes every
        code of every format up to 16 bits identically to numpy and{" "}
        <a href={SOURCES.ml_dtypes!.url} className={A}>
          ml_dtypes
        </a>
        ; see{" "}
        <Link href="/about" className={A}>
          about
        </Link>{" "}
        for the checks.
      </p>

      <h2 className="mt-10 text-xl font-semibold tracking-tight">
        Floating-point formats
      </h2>
      <Table label="Floating-point formats">
        <thead>
          <tr>
            {[
              "Format",
              "Bits (s·e·m)",
              "Bias",
              "Largest",
              "Smallest normal",
              "Smallest subnormal",
              "Precision",
              "Binades",
              "Specials",
              "Defined in",
            ].map((h) => (
              <th key={h} className={TH} scope="col">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {FORMAT_ORDER.map((f) => {
            const i = INFO[f];
            const d = FORMATS[f];
            const src = SOURCES[d.src]!;
            return (
              <tr key={f} data-format={f}>
                <th
                  scope="row"
                  className={`${TD} whitespace-nowrap font-semibold`}
                >
                  {d.name}
                </th>
                <td className={`${TD} font-mono`}>
                  1·{i.e}·{i.m}
                </td>
                <td className={`${TD} font-mono`}>{i.bias}</td>
                <td className={`${TD} font-mono`}>{exact(i.max)}</td>
                <td className={`${TD} font-mono`}>{pow2Text(i.min_normal)}</td>
                <td className={`${TD} font-mono`}>{pow2Text(i.min_sub)}</td>
                <td className={`${TD} font-mono`}>
                  {i.m + 1} bits (ulp at 1 = {pow2Text(i.eps)})
                </td>
                <td className={`${TD} font-mono`}>{i.binades}</td>
                <td className={TD}>{KIND[i.kind]}</td>
                <td className={TD}>
                  <a href={src.url} className={A}>
                    {src.title.split(",")[0]}
                  </a>
                  , {d.ref}
                </td>
              </tr>
            );
          })}
        </tbody>
      </Table>
      <p className="text-sm text-neutral-600 dark:text-neutral-400">
        Binades: the powers of two from the smallest subnormal to the largest
        value, inclusive. For E4M3 and E5M2 this gives 18 and 32, as the OCP FP8
        specification&apos;s Table 2 states.
      </p>

      <h2 className="mt-10 text-xl font-semibold tracking-tight">
        MX block formats
      </h2>
      <p className="mt-2 max-w-3xl text-neutral-600 dark:text-neutral-300">
        A block of 32 elements shares one E8M0 scale X = 2<sup>k</sup> (k from
        −127 to 127; code 255 is NaN). The concrete formats of the{" "}
        <a href={SOURCES.mx!.url} className={A}>
          OCP MX specification 1.0
        </a>{" "}
        (Table 1):
      </p>
      <Table label="MX formats">
        <thead>
          <tr>
            {[
              "Format",
              "Element",
              "Element bits",
              "Block",
              "Scale",
              "Bits per value",
              "Largest element power of two",
            ].map((h) => (
              <th key={h} className={TH} scope="col">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {MX_ORDER.map((m) => {
            const f = MX_FORMATS[m];
            return (
              <tr key={m}>
                <th
                  scope="row"
                  className={`${TD} whitespace-nowrap font-semibold`}
                >
                  {f.name}
                </th>
                <td className={TD}>
                  {f.elem === "int8"
                    ? "INT8, 6 fraction bits (±1 63/64)"
                    : FORMATS[f.elem].name}
                </td>
                <td className={`${TD} font-mono`}>{f.bits}</td>
                <td className={`${TD} font-mono`}>32</td>
                <td className={`${TD} font-mono`}>E8M0</td>
                <td className={`${TD} font-mono`}>{f.bits + 8 / 32}</td>
                <td className={`${TD} font-mono`}>
                  2{f.emax_elem ? <sup>{f.emax_elem}</sup> : <sup>0</sup>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </Table>

      <h2 className="mt-10 text-xl font-semibold tracking-tight">
        NF4 (NormalFloat 4)
      </h2>
      <p className="mt-2 max-w-3xl text-neutral-600 dark:text-neutral-300">
        Not a floating-point format: a code book of 16 values derived from
        quantiles of the normal distribution, scaled per block of 64 by the
        block&apos;s absolute maximum (
        <a href={SOURCES.qlora!.url} className={A}>
          QLoRA
        </a>
        ). The values are{" "}
        <a href={SOURCES.bnb!.url} className={A}>
          bitsandbytes&apos;
        </a>{" "}
        table; a value is stored as the nearest code, ties to the lower one.
      </p>
      <Table label="NF4 code book">
        <thead>
          <tr>
            <th className={TH} scope="col">
              Code
            </th>
            <th className={TH} scope="col">
              Value
            </th>
            <th className={TH} scope="col">
              Upper decision threshold
            </th>
          </tr>
        </thead>
        <tbody>
          {NF4.map((v, i) => (
            <tr key={i}>
              <td className={`${TD} font-mono`}>{i}</td>
              <td className={`${TD} font-mono`}>{v}</td>
              <td className={`${TD} font-mono`}>{i < 15 ? NF4_MID[i] : "–"}</td>
            </tr>
          ))}
        </tbody>
      </Table>

      <h2 className="mt-10 text-xl font-semibold tracking-tight">
        Rounding modes
      </h2>
      <p className="mt-2 max-w-3xl text-neutral-600 dark:text-neutral-300">
        The five modes of IEEE 754-2019 (section 4.3) and stochastic rounding.
        Overflow follows section 7.4 (towards-zero modes give the largest finite
        value), or the OCP saturating mode when it is chosen; formats without
        infinity saturate or give NaN, as their specifications say.
      </p>
      <Table label="Rounding modes">
        <thead>
          <tr>
            <th className={TH} scope="col">
              Mode
            </th>
            <th className={TH} scope="col">
              Rounds
            </th>
          </tr>
        </thead>
        <tbody>
          {MODES.map((m) => (
            <tr key={m}>
              <th scope="row" className={`${TD} whitespace-nowrap font-mono`}>
                {MODE_NAMES[m]}
              </th>
              <td className={TD}>{MODE_TEXT[m]}</td>
            </tr>
          ))}
        </tbody>
      </Table>

      <h2 className="mt-10 text-xl font-semibold tracking-tight">Sources</h2>
      <ul className="mt-2 list-disc space-y-1 pl-6 text-sm">
        {Object.entries(SOURCES).map(([k, s]) => (
          <li key={k}>
            <a href={s.url} className={A}>
              {s.title}
            </a>
          </li>
        ))}
      </ul>
      <p className="mt-6 text-sm text-neutral-600 dark:text-neutral-400">
        {int(FORMAT_ORDER.length)} formats, {MX_ORDER.length} MX formats,{" "}
        {MODES.length} rounding modes.
      </p>
    </main>
  );
}
