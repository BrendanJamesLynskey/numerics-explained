"use client";

/**
 * Chapter 3's hero: 8,192 numbers summed four ways, as running totals
 * (above) and as the error against the exact sum on a log scale (below):
 * naive in FP16 or BF16, naive in FP32, Kahan's compensated sum and
 * pairwise summation in the low format. A state every 128 inputs, from
 * `sumSteps`; the exact prefix sums are exact (the inputs are FP16 values).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { int, sup, trim } from "@/lib/format";
import { sumCaption } from "@/lib/num/captions";
import {
  FORMATS,
  sumSteps,
  type FormatId,
  type SumDist,
  type SumStep,
} from "@/lib/num/model";
import { DEMO } from "@/lib/num/values";
import { METHOD_COLOUR } from "@/lib/viz/palette";

const VW = 640;
type Key = "naive" | "naive32" | "kahan" | "pairwise";
const KEYS: Key[] = ["naive", "naive32", "kahan", "pairwise"];
const FLOOR = -7;

export default function SumWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [fid, setFid] = useState<FormatId>("fp16");
  const [dist, setDist] = useState<SumDist>("uniform");
  const [hover, setHover] = useState<string | null>(null);
  const n = DEMO.sumN;
  const r = useMemo(
    () => sumSteps(fid, n, dist, DEMO.sumSeed, DEMO.sumEvery),
    [fid, dist, n],
  );
  const steps: SumStep[] = [
    { i: 0, exact: 0, naive: 0, naive32: 0, kahan: 0, pairwise: 0 },
    ...r.steps,
  ];
  const st = useStepper(steps.length, {
    stepMs: 160,
    resetKey: `${fid}-${dist}`,
  });
  const s = steps[st.step]!;
  const font = useSvgFont(VW);
  const efont = useSvgFont(VW);
  const nm = FORMATS[fid].name;
  const label: Record<Key, string> = {
    naive: `${nm} naive`,
    naive32: "FP32 naive",
    kahan: `${nm} Kahan`,
    pairwise: `${nm} pairwise`,
  };

  // margins sized from the label font, so labels fit on a phone
  const fz = font.fs(12);
  const PAD = { l: fz * 3.4 + 10, r: 14, t: fz + 12, b: fz + 14 };
  const all = steps.flatMap((p) => [p.exact, ...KEYS.map((k) => p[k])]);
  const yMax = Math.max(...all);
  const yMin = Math.min(0, ...all);
  const H1 = 200;
  const X = (i: number) => PAD.l + (i / n) * (VW - PAD.l - PAD.r);
  const Y = (v: number) =>
    H1 - PAD.b - ((v - yMin) / (yMax - yMin || 1)) * (H1 - PAD.t - PAD.b);
  const shown = steps.slice(0, st.step + 1);
  const path = (f: (p: SumStep) => number, Yf: (v: number) => number) =>
    shown
      .map(
        (p, i) => `${i ? "L" : "M"}${X(p.i).toFixed(1)} ${Yf(f(p)).toFixed(1)}`,
      )
      .join("");

  // the error panel: log10 |sum - exact|, an exact result drawn at the floor
  const errs = steps.flatMap((p) =>
    KEYS.map((k) => Math.abs(p[k] - p.exact)).filter((e) => e > 0),
  );
  const top = Math.ceil(Math.log10(Math.max(1e-6, ...errs)));
  const H2 = 170;
  const lg = (e: number) => (e > 0 ? Math.max(FLOOR, Math.log10(e)) : FLOOR);
  const Y2 = (l: number) =>
    H2 - PAD.b - ((l - FLOOR) / (top - FLOOR)) * (H2 - PAD.t - PAD.b);
  const dash: Record<Key, string | undefined> = {
    naive: undefined,
    naive32: "6 3",
    kahan: undefined,
    pairwise: "2 3",
  };

  const visual = (
    <div className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${VW} ${H1}`}
        className="w-full"
        role="img"
        aria-label="Running totals of the four summations against the exact sum"
      >
        {[yMin, (yMin + yMax) / 2, yMax].map((v) => (
          <g key={v}>
            <line
              x1={PAD.l}
              x2={VW - PAD.r}
              y1={Y(v)}
              y2={Y(v)}
              stroke="currentColor"
              strokeOpacity={0.12}
            />
            <text
              x={PAD.l - 6}
              y={Y(v) + 4}
              fontSize={font.fs(12)}
              textAnchor="end"
              fill="currentColor"
            >
              {trim(v, 4)}
            </text>
          </g>
        ))}
        <path
          d={path((p) => p.exact, Y)}
          fill="none"
          stroke={METHOD_COLOUR.exact}
          strokeWidth={5}
          strokeOpacity={0.35}
        />
        {KEYS.map((k) => (
          <path
            key={k}
            d={path((p) => p[k], Y)}
            fill="none"
            stroke={METHOD_COLOUR[k]}
            strokeWidth={2.5}
            strokeDasharray={dash[k]}
          />
        ))}
        <text
          x={VW - PAD.r}
          y={H1 - 6}
          fontSize={font.fs(12)}
          textAnchor="end"
          fill="currentColor"
        >
          inputs summed: {int(s.i)} →
        </text>
        <text x={PAD.l} y={fz + 2} fontSize={fz} fill="currentColor">
          running total (wide grey: exact)
        </text>
      </svg>
      <svg
        ref={efont.ref}
        viewBox={`0 0 ${VW} ${H2}`}
        className="mt-1 w-full"
        role="img"
        aria-label="The error of each summation against the exact sum, on a log scale"
      >
        {Array.from({ length: top - FLOOR + 1 }, (_, i) => FLOOR + i)
          .filter((l) => (l - FLOOR) % (font.narrow ? 3 : 2) === 0)
          .map((l) => (
            <g key={l}>
              <line
                x1={PAD.l}
                x2={VW - PAD.r}
                y1={Y2(l)}
                y2={Y2(l)}
                stroke="currentColor"
                strokeOpacity={0.12}
              />
              <text
                x={PAD.l - 6}
                y={Y2(l) + 4}
                fontSize={efont.fs(12)}
                textAnchor="end"
                fill="currentColor"
              >
                {l === FLOOR ? "exact" : `10${sup(l)}`}
              </text>
            </g>
          ))}
        {KEYS.map((k) => (
          <path
            key={k}
            d={path((p) => lg(Math.abs(p[k] - p.exact)), Y2)}
            fill="none"
            stroke={METHOD_COLOUR[k]}
            strokeWidth={2.5}
            strokeDasharray={dash[k]}
          />
        ))}
        <text x={PAD.l} y={fz + 2} fontSize={efont.fs(12)} fill="currentColor">
          |error| against the exact sum (log scale)
        </text>
      </svg>
      <ul
        className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs"
        aria-label="Legend"
      >
        {KEYS.map((k) => (
          <li key={k} className="flex items-center gap-1.5">
            <svg width="22" height="8" aria-hidden>
              <line
                x1={0}
                x2={22}
                y1={4}
                y2={4}
                stroke={METHOD_COLOUR[k]}
                strokeWidth={3}
                strokeDasharray={dash[k]}
              />
            </svg>
            {label[k]}
          </li>
        ))}
      </ul>
    </div>
  );

  return (
    <AnimationPanel
      testId="sum-widget"
      title="Summing 8,192 numbers four ways"
      summary={`Every addition is rounded to ${nm} (or FP32), as the format's arithmetic would; the exact sum is computed alongside.`}
      stepper={st}
      stepLabel="inputs ×128"
      caption={
        s.i === 0
          ? `${int(n)} inputs, each an FP16 number. Every running total starts at 0.`
          : sumCaption(fid, s, n)
      }
      visual={visual}
      equation={children}
      hl={hover ?? "comp"}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
          <Stat label="Exact" value={trim(s.exact, 7)} />
          {KEYS.map((k) => (
            <Stat key={k} label={label[k]} value={trim(s[k], 7)} />
          ))}
        </div>
      }
      params={
        <>
          <Segmented
            label="Low-precision format"
            value={fid}
            options={(["fp16", "bf16"] as const).map((v) => ({
              value: v,
              label: FORMATS[v].name,
            }))}
            onChange={setFid}
          />
          <Segmented
            label="Inputs"
            value={dist}
            options={[
              { value: "uniform", label: "uniform 0–1" },
              { value: "ones", label: "all ones" },
              { value: "normal", label: "± normal" },
            ]}
            onChange={setDist}
          />
        </>
      }
    />
  );
}
