"use client";

/**
 * Chapter 6's second animation: SmoothQuant's migration strength alpha
 * swept from 0 to 1 in eighths (`smoothSteps`). Above: each input
 * channel's largest activation (orange) and largest weight (blue) after
 * smoothing, on a log scale: as alpha grows, the activations' outliers
 * shrink and the weights' columns grow. Below: the per-tensor W8A8 output
 * error at each alpha, the current one highlighted.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { trim } from "@/lib/format";
import { smoothCaption } from "@/lib/num/captions";
import {
  OUTLIER_CHANNELS,
  demoOutlierLayer,
  smoothSteps,
  w8a8Error,
} from "@/lib/num/model";
import { DEMO } from "@/lib/num/values";
import { STATE_COLOUR } from "@/lib/viz/palette";

const VW = 640;

export default function SmoothWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [hover, setHover] = useState<string | null>(null);
  const L = useMemo(() => demoOutlierLayer(...DEMO.outlier), []);
  const steps = useMemo(() => smoothSteps(L.W, L.X), [L]);
  const base = useMemo(() => w8a8Error(L.W, L.X).err, [L]);
  const st = useStepper(steps.length, { stepMs: 1300 });
  const s = steps[st.step]!;
  const font = useSvgFont(VW);
  const d = s.xmax.length;
  const fz = font.fs(11);

  // top: per-channel maxima on a log axis shared by every alpha
  const all = steps.flatMap((x) => [...x.xmax, ...x.wmax]);
  const lo = Math.log10(Math.min(...all)) - 0.1;
  const hi = Math.log10(Math.max(...all)) + 0.1;
  const H1 = font.narrow ? 280 : 170;
  const PAD = { l: fz * 3 + 6, r: 8, t: 2 * fz + 12, b: fz + 12 };
  const slot = (VW - PAD.l - PAD.r) / d;
  const Y = (v: number) =>
    H1 - PAD.b - ((Math.log10(v) - lo) / (hi - lo)) * (H1 - PAD.t - PAD.b);
  // 1-2-5 ticks inside the range
  const ticks: number[] = [];
  for (let k = Math.floor(lo); k <= Math.ceil(hi); k++)
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** k;
      if (Math.log10(v) >= lo && Math.log10(v) <= hi) ticks.push(v);
    }

  // bottom: error against alpha
  const H2 = font.narrow ? 200 : 120;
  const top2 = H1 + 16;
  const emax = Math.max(base, ...steps.map((x) => x.err));
  const X2 = (k: number) => PAD.l + (k / 8) * (VW - PAD.l - PAD.r - 20);
  const Y2 = (e: number) => top2 + H2 - fz - 8 - (e / emax) * (H2 - fz - 26);

  const visual = (
    <svg
      ref={font.ref}
      viewBox={`0 0 ${VW} ${top2 + H2}`}
      className="w-full"
      role="img"
      aria-label={`Per-channel maxima of activations and weights after smoothing with alpha ${s.alpha}, and the output error for every alpha`}
    >
      <text x={PAD.l} y={fz} fontSize={fz} fill="currentColor">
        {font.narrow
          ? "max |x| (orange), |w| (blue), log"
          : "largest |x| (orange) and |w| (blue) per input channel, log scale"}
      </text>
      {ticks.map((v) => (
        <g key={v}>
          <line
            x1={PAD.l}
            x2={VW - PAD.r}
            y1={Y(v)}
            y2={Y(v)}
            stroke="currentColor"
            strokeOpacity={0.15}
          />
          <text
            x={PAD.l - 4}
            y={Y(v) + fz / 3}
            fontSize={fz}
            textAnchor="end"
            fill="currentColor"
          >
            {trim(v, 2)}
          </text>
        </g>
      ))}
      {s.xmax.map((xv, j) => {
        const x0 = PAD.l + j * slot;
        const bw = slot * 0.38;
        const out = OUTLIER_CHANNELS.includes(j);
        return (
          <g key={j}>
            <rect
              x={x0 + slot * 0.08}
              y={Y(xv)}
              width={bw}
              height={H1 - PAD.b - Y(xv)}
              fill={STATE_COLOUR.positive}
              stroke={out ? STATE_COLOUR.error : "none"}
              strokeWidth={out ? 2 : 0}
            />
            <rect
              x={x0 + slot * 0.08 + bw}
              y={Y(s.wmax[j]!)}
              width={bw}
              height={H1 - PAD.b - Y(s.wmax[j]!)}
              fill={STATE_COLOUR.negative}
            />
            {(!font.narrow || j % 3 === 0) && (
              <text
                x={x0 + slot / 2}
                y={H1 - PAD.b + fz + 2}
                fontSize={fz}
                textAnchor="middle"
                fill="currentColor"
              >
                {j}
              </text>
            )}
          </g>
        );
      })}
      <text x={PAD.l} y={top2 + fz} fontSize={fz} fill="currentColor">
        {font.narrow
          ? "W8A8 error against α"
          : "W8A8 output error against α (dashed: no smoothing)"}
      </text>
      <line
        x1={X2(0)}
        x2={X2(8)}
        y1={Y2(base)}
        y2={Y2(base)}
        stroke="currentColor"
        strokeOpacity={0.5}
        strokeDasharray="5 4"
      />
      <path
        d={steps
          .map((x, k) => `${k ? "L" : "M"}${X2(k)} ${Y2(x.err)}`)
          .join(" ")}
        fill="none"
        stroke="currentColor"
        strokeOpacity={0.6}
      />
      {steps.map((x, k) => (
        <circle
          key={k}
          cx={X2(k)}
          cy={Y2(x.err)}
          r={k === st.step ? 7 : 4}
          fill={k === st.step ? STATE_COLOUR.active : "currentColor"}
          fillOpacity={k === st.step ? 1 : 0.45}
        />
      ))}
      {[0, 4, 8].map((k) => (
        <text
          key={k}
          x={X2(k)}
          y={top2 + H2 - 2}
          fontSize={fz}
          textAnchor="middle"
          fill="currentColor"
        >
          α = {k / 8}
        </text>
      ))}
    </svg>
  );

  return (
    <AnimationPanel
      testId="smooth-widget"
      title="SmoothQuant: moving the difficulty into the weights"
      summary="The same layer, with each input channel divided by s in the activations and multiplied by s in the weights, then per-tensor INT8 for both, by the tested model. The product is unchanged in exact arithmetic."
      stepper={st}
      stepLabel="α step"
      caption={smoothCaption(s, base)}
      visual={visual}
      equation={children}
      hl={hover ?? (st.step === 0 ? "s" : "alpha")}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="α" value={trim(s.alpha, 3)} plain />
          <Stat label="Largest |x|" value={trim(Math.max(...s.xmax), 3)} />
          <Stat label="Largest |w|" value={trim(Math.max(...s.wmax), 3)} />
          <Stat
            label="Output error"
            value={trim(s.err, 3)}
            hint={`${trim(base / s.err, 3)}× less than none`}
          />
        </div>
      }
    />
  );
}
