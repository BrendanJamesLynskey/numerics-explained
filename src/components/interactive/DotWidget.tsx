"use client";

/**
 * Chapter 10's hero: one 32-element dot product in four number systems
 * (`dotSteps`), one multiply-accumulate per step. Left: the relative error
 * of each running sum against the exact one (log scale). Right: the energy
 * spent so far by FP32, FP16 and INT8 multiply-accumulates, from Horowitz's
 * 45 nm table. MXFP4 has no entry in that table, so it gets no bar.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { sup, trim } from "@/lib/format";
import { dotCaption, pctOrSci } from "@/lib/num/captions";
import { demoBlock, dotSteps, type DotStep } from "@/lib/num/model";
import { METHOD_COLOUR, OKABE_ITO } from "@/lib/viz/palette";

const VW = 640;
type Lane = "fp32" | "fp16" | "int8" | "mx";
const LANES: { key: Lane; label: string; colour: string }[] = [
  { key: "fp32", label: "FP32", colour: METHOD_COLOUR.naive32 },
  { key: "fp16", label: "FP16", colour: METHOD_COLOUR.naive },
  { key: "int8", label: "INT8", colour: METHOD_COLOUR.kahan },
  { key: "mx", label: "MXFP4", colour: OKABE_ITO.purple },
];
const FLOOR = -9;

export default function DotWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [pair, setPair] = useState<"5-6" | "7-8">("5-6");
  const [hover, setHover] = useState<string | null>(null);
  const [s1, s2] = pair.split("-").map(Number) as [number, number];
  const r = useMemo(
    () => dotSteps(demoBlock("normal", s1), demoBlock("normal", s2)),
    [s1, s2],
  );
  const st = useStepper(r.steps.length, { stepMs: 380, resetKey: pair });
  const s = r.steps[st.step]!;
  const font = useSvgFont(VW);
  const fz = font.fs(11);
  const n = r.steps.length;
  const rel = (p: DotStep, k: Lane) => {
    const e = Math.abs(p[k] - p.ref) / Math.abs(p.ref || 1);
    return Math.max(FLOOR, e > 0 ? Math.log10(e) : FLOOR);
  };
  const left = fz * 3 + 8;
  const W1 = font.narrow ? VW - 16 : 380;
  const H = font.narrow ? 300 : 200;
  const X = (i: number) => left + (i / (n - 1)) * (W1 - left);
  const Y = (l: number) =>
    fz + 10 + ((0 - l) / (0 - FLOOR)) * (H - 2 * fz - 20);
  const shown = r.steps.slice(0, st.step + 1);

  // energy bars
  const ex0 = font.narrow ? left : W1 + 30;
  const ey0 = font.narrow ? H + 30 : 0;
  const EW = VW - ex0 - 10;
  const emax = n * r.pj.fp32;
  const kinds = ["fp32", "fp16", "int8"] as const;
  const totalH = font.narrow ? ey0 + 3 * (fz * 2 + 20) + fz + 20 : H;

  const visual = (
    <svg
      ref={font.ref}
      viewBox={`0 0 ${VW} ${totalH}`}
      className="w-full"
      role="img"
      aria-label={`Dot product after ${s.i + 1} of ${n} products: relative error of each format and the energy spent`}
    >
      <text x={left} y={fz} fontSize={fz} fill="currentColor">
        relative error of the running sum (log)
      </text>
      {[0, -3, -6, -9].map((l) => (
        <g key={l}>
          <line
            x1={left}
            x2={W1}
            y1={Y(l)}
            y2={Y(l)}
            stroke="currentColor"
            strokeOpacity={0.12}
          />
          <text
            x={left - 4}
            y={Y(l) + fz / 3}
            fontSize={fz}
            textAnchor="end"
            fill="currentColor"
          >
            {l === 0 ? "1" : `10${sup(l)}`}
          </text>
        </g>
      ))}
      {LANES.map((ln) => (
        <path
          key={ln.key}
          d={shown
            .map((p, i) => `${i ? "L" : "M"}${X(p.i)} ${Y(rel(p, ln.key))}`)
            .join(" ")}
          fill="none"
          stroke={ln.colour}
          strokeWidth={2.5}
          strokeDasharray={ln.key === "mx" ? "6 3" : undefined}
        />
      ))}
      {LANES.map((ln, k) => (
        <text
          key={ln.key}
          x={left + 4 + k * ((W1 - left) / 4)}
          y={H - 4}
          fontSize={fz}
          fill={ln.colour}
          fontWeight={600}
        >
          {ln.label}
        </text>
      ))}
      <text x={ex0} y={ey0 + fz} fontSize={fz} fill="currentColor">
        energy so far (45 nm)
      </text>
      {kinds.map((k, j) => {
        const y = ey0 + fz + 14 + j * (fz * 2 + 20);
        const e = (s.i + 1) * r.pj[k];
        const lane = LANES.find((l) => l.key === k)!;
        return (
          <g key={k}>
            <text x={ex0} y={y + fz} fontSize={fz} fill="currentColor">
              {lane.label}: {trim(e, 3)} pJ
            </text>
            <rect
              x={ex0}
              y={y + fz + 4}
              width={EW}
              height={12}
              fill="currentColor"
              fillOpacity={0.08}
            />
            <rect
              x={ex0}
              y={y + fz + 4}
              width={(e / emax) * EW}
              height={12}
              fill={lane.colour}
            />
          </g>
        );
      })}
    </svg>
  );

  const relText = (k: Lane) =>
    pctOrSci(Math.abs(s[k] - s.ref) / Math.abs(s.ref || 1));
  return (
    <AnimationPanel
      testId="dot-widget"
      title="One dot product, four number systems"
      summary={`Two blocks of ${n} values (illustrative), multiplied and summed by the tested model in FP32, FP16, INT8 (one absmax scale per block) and MXFP4 (one power-of-two scale per block). Energy per operation: Horowitz’s 45 nm figures.`}
      stepper={st}
      stepLabel="product"
      caption={dotCaption(r, st.step)}
      visual={visual}
      equation={children}
      hl={hover ?? (st.step === n - 1 ? "scale" : "acc")}
      onEquationHover={setHover}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {LANES.map((ln) => (
            <Stat
              key={ln.key}
              label={ln.label}
              value={trim(s[ln.key], 5)}
              hint={`${relText(ln.key)} off`}
            />
          ))}
        </div>
      }
      params={
        <Segmented
          label="Blocks"
          value={pair}
          options={[
            { value: "5-6", label: "pair 1" },
            { value: "7-8", label: "pair 2" },
          ]}
          onChange={setPair}
        />
      }
    />
  );
}
