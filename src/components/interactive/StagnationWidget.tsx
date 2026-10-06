"use client";

/**
 * Chapter 2's second animation: add a small step delta (an eighth of an
 * ulp) to h over and over in a low-precision format. Rounding to nearest
 * returns h every time, so h never moves; stochastic rounding moves by a
 * whole ulp one time in eight, and follows the exact sum on average.
 * Driven by `stagnationSteps`.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { pow2Text, trim } from "@/lib/format";
import { stagCaption } from "@/lib/num/captions";
import { FORMATS, stagnationSteps, type FormatId } from "@/lib/num/model";
import { DEMO, stagDelta } from "@/lib/num/values";
import { METHOD_COLOUR } from "@/lib/viz/palette";

const VW = 640;
const VH = 240;
const PAD = { l: 52, r: 16, t: 14, b: 34 };

export default function StagnationWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [fid, setFid] = useState<FormatId>("fp16");
  const [hover, setHover] = useState<string | null>(null);
  const delta = stagDelta(fid);
  const steps = useMemo(
    () => stagnationSteps(fid, 1, delta, DEMO.stagSteps, DEMO.stagSeed),
    [fid, delta],
  );
  const st = useStepper(steps.length, { stepMs: 90, resetKey: fid });
  const s = steps[st.step]!;
  const font = useSvgFont(VW);

  const yMax = Math.max(...steps.map((p) => Math.max(p.exact, p.sr)));
  const yMin = 1 - (yMax - 1) * 0.06;
  const X = (t: number) => PAD.l + (t / DEMO.stagSteps) * (VW - PAD.l - PAD.r);
  const Y = (v: number) =>
    VH - PAD.b - ((v - yMin) / (yMax - yMin)) * (VH - PAD.t - PAD.b);
  const shown = steps.slice(0, st.step + 1);
  const line = (k: "exact" | "rne" | "sr") =>
    shown
      .map(
        (p, i) => `${i ? "L" : "M"}${X(p.t).toFixed(1)} ${Y(p[k]).toFixed(1)}`,
      )
      .join("");

  const visual = (
    <svg
      ref={font.ref}
      viewBox={`0 0 ${VW} ${VH}`}
      className="w-full"
      role="img"
      aria-label="The running value of h after each addition: exact, nearest-even and stochastic"
    >
      {[1, (1 + yMax) / 2, yMax].map((v) => (
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
      <text
        x={VW - PAD.r}
        y={VH - 6}
        fontSize={font.fs(12)}
        textAnchor="end"
        fill="currentColor"
      >
        additions →
      </text>
      <path
        d={line("exact")}
        fill="none"
        stroke={METHOD_COLOUR.exact}
        strokeWidth={2}
        strokeDasharray="5 4"
      />
      <path
        d={line("rne")}
        fill="none"
        stroke={METHOD_COLOUR.rne}
        strokeWidth={3}
      />
      <path
        d={line("sr")}
        fill="none"
        stroke={METHOD_COLOUR.sr}
        strokeWidth={2.5}
      />
      <g fontSize={font.fs(12)}>
        <text x={PAD.l + 8} y={PAD.t + 12} fill={METHOD_COLOUR.exact}>
          exact (dashed)
        </text>
        <text
          x={PAD.l + 8}
          y={PAD.t + 12 + font.fs(15)}
          fill={METHOD_COLOUR.sr}
          fontWeight={600}
        >
          stochastic
        </text>
        <text
          x={PAD.l + 8}
          y={Y(1) - 6}
          fill={METHOD_COLOUR.rne}
          fontWeight={600}
        >
          nearest-even
        </text>
      </g>
    </svg>
  );

  return (
    <AnimationPanel
      testId="stagnation-widget"
      title="Adding less than half an ulp"
      summary={`h starts at 1 and gains δ = ${pow2Text(delta)} (an eighth of an ulp of 1 in ${FORMATS[fid].name}) ${DEMO.stagSteps} times.`}
      stepper={st}
      stepLabel="addition"
      caption={stagCaption(fid, s, 1)}
      visual={visual}
      equation={children}
      hl={hover ?? "delta"}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-3 gap-2">
          <Stat label="Exact" value={trim(s.exact, 6)} />
          <Stat label="Nearest-even" value={trim(s.rne, 6)} />
          <Stat label="Stochastic" value={trim(s.sr, 6)} />
        </div>
      }
      params={
        <Segmented
          label="Format"
          value={fid}
          options={(["fp16", "bf16", "e4m3"] as const).map((v) => ({
            value: v,
            label: FORMATS[v].name,
          }))}
          onChange={setFid}
        />
      }
    />
  );
}
