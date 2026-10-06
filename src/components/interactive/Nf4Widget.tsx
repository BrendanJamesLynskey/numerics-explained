"use client";

/**
 * Chapter 8's hero: NF4's 16 values derived step by step (`nf4Build`, the
 * reference's port of bitsandbytes' create_normal_map). Above: the standard
 * normal density with each quantile placed as it is computed. Below: the
 * code book on the [-1, 1] line once normalised, and finally bitsandbytes'
 * own table drawn over it.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { sci, trim } from "@/lib/format";
import { nf4Caption } from "@/lib/num/captions";
import { NF4, demoNormal, nf4Build, nf4VsInt4 } from "@/lib/num/model";
import { DEMO } from "@/lib/num/values";
import { METHOD_COLOUR, STATE_COLOUR } from "@/lib/viz/palette";

const VW = 640;
const ZMAX = 2.4;

export default function Nf4Widget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [hover, setHover] = useState<string | null>(null);
  const b = useMemo(() => nf4Build(), []);
  const cmp = useMemo(() => nf4VsInt4(demoNormal(...DEMO.normal)), []);
  const st = useStepper(b.steps.length, { stepMs: 900 });
  const s = b.steps[st.step]!;
  const font = useSvgFont(VW);
  const fz = font.fs(11);
  const L = 20;
  const R = VW - 20;
  const X = (z: number) => L + ((z + ZMAX) / (2 * ZMAX)) * (R - L);
  const H1 = font.narrow ? 250 : 170;
  const base = H1 - fz - 8;
  const pdf = (z: number) => Math.exp((-z * z) / 2) / Math.sqrt(2 * Math.PI);
  const Y = (p: number) => base - (p / pdf(0)) * (base - fz - 14);
  const curve = Array.from(
    { length: 97 },
    (_, i) => -ZMAX + (i * 2 * ZMAX) / 96,
  )
    .map((z, i) => `${i ? "L" : "M"}${X(z).toFixed(1)} ${Y(pdf(z)).toFixed(1)}`)
    .join(" ");

  const nPos = s.phase === "pdf" ? 0 : s.phase === "pos" ? s.n : 8;
  const nNeg =
    s.phase === "pdf" || s.phase === "pos" ? 0 : s.phase === "neg" ? s.n : 7;
  const zeroShown = ["zero", "normalise", "compare"].includes(s.phase);
  const marks = [
    ...b.pos
      .slice(0, nPos)
      .map((q, i) => ({ z: q.z, cur: s.phase === "pos" && i === s.n - 1 })),
    ...b.neg
      .slice(0, nNeg)
      .map((q, i) => ({ z: q.z, cur: s.phase === "neg" && i === s.n - 1 })),
    ...(zeroShown ? [{ z: 0, cur: s.phase === "zero" }] : []),
  ];

  const top2 = H1 + 10;
  const H2 = font.narrow ? 110 : 90;
  const lineY = top2 + 44;
  const V = (v: number) => L + ((v + 1.1) / 2.2) * (R - L);
  const normalised = s.phase === "normalise" || s.phase === "compare";

  const visual = (
    <svg
      ref={font.ref}
      viewBox={`0 0 ${VW} ${top2 + H2}`}
      className="w-full"
      role="img"
      aria-label={`The standard normal density with ${marks.length} NF4 quantiles placed`}
    >
      <text x={L} y={fz} fontSize={fz} fill="currentColor">
        {font.narrow
          ? "N(0, 1) and NF4’s quantiles"
          : "standard normal density; NF4’s quantiles z"}
      </text>
      <path d={curve} fill="none" stroke="currentColor" strokeWidth={1.5} />
      <line
        x1={L}
        x2={R}
        y1={base}
        y2={base}
        stroke="currentColor"
        strokeOpacity={0.4}
      />
      {marks.map((m, i) => (
        <g key={i}>
          <line
            x1={X(m.z)}
            x2={X(m.z)}
            y1={base}
            y2={Y(pdf(m.z))}
            stroke={m.cur ? STATE_COLOUR.active : METHOD_COLOUR.pairwise}
            strokeWidth={m.cur ? 3 : 2}
          />
          <circle
            cx={X(m.z)}
            cy={base}
            r={m.cur ? 5 : 3.5}
            fill={m.cur ? STATE_COLOUR.active : METHOD_COLOUR.pairwise}
          />
        </g>
      ))}
      {[-2, -1, 0, 1, 2].map((z) => (
        <text
          key={z}
          x={X(z)}
          y={base + fz + 4}
          fontSize={fz}
          textAnchor="middle"
          fill="currentColor"
        >
          {z === 0 ? "0" : `${z < 0 ? "−" : ""}${Math.abs(z)}σ`}
        </text>
      ))}
      <g opacity={normalised ? 1 : 0.15}>
        <text x={L} y={top2 + fz + 4} fontSize={fz} fill="currentColor">
          the code book on [−1, 1]
          {s.phase === "compare"
            ? font.narrow
              ? "; ticks: bitsandbytes"
              : " — ticks above: bitsandbytes’ table"
            : ""}
        </text>
        <line
          x1={V(-1)}
          x2={V(1)}
          y1={lineY}
          y2={lineY}
          stroke="currentColor"
          strokeOpacity={0.4}
        />
        {b.values.map((v, i) => (
          <circle
            key={i}
            cx={V(v)}
            cy={lineY}
            r={4}
            fill={METHOD_COLOUR.pairwise}
          />
        ))}
        {s.phase === "compare" &&
          NF4.map((v, i) => (
            <line
              key={i}
              x1={V(v)}
              x2={V(v)}
              y1={lineY - 14}
              y2={lineY - 6}
              stroke={STATE_COLOUR.active}
              strokeWidth={2}
            />
          ))}
        {[-1, -0.5, 0, 0.5, 1].map((v) => (
          <text
            key={v}
            x={V(v)}
            y={lineY + fz + 10}
            fontSize={fz}
            textAnchor="middle"
            fill="currentColor"
          >
            {v < 0 ? `−${Math.abs(v)}` : v}
          </text>
        ))}
      </g>
    </svg>
  );

  const hl =
    s.phase === "pos" || s.phase === "neg"
      ? "p"
      : s.phase === "normalise" || s.phase === "compare"
        ? "norm"
        : "z";

  return (
    <AnimationPanel
      testId="nf4-widget"
      title="Deriving NF4 from the normal distribution"
      summary="bitsandbytes’ construction of the NF4 code book, recomputed in double precision by the tested model: quantiles of N(0, 1) at evenly spaced probabilities, plus zero, scaled to [−1, 1]."
      stepper={st}
      stepLabel="step"
      caption={nf4Caption(b, st.step)}
      visual={visual}
      equation={children}
      hl={hover ?? hl}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Values placed" value={`${marks.length} of 16`} />
          <Stat label="Largest quantile" value={`${trim(b.max, 5)}σ`} />
          <Stat
            label="Gap to bitsandbytes"
            value={s.phase === "compare" ? sci(b.max_diff, 2) : "–"}
          />
          <Stat
            label="MSE, NF4 vs INT4"
            value={`${trim(cmp.mse_nf4, 3)} vs ${trim(cmp.mse_int4, 3)}`}
            hint="4,096 normal values, blocks of 64"
          />
        </div>
      }
    />
  );
}
