"use client";

/**
 * Chapter 6's hero: a layer input with two systematic outlier channels.
 * Left: the activations as a heat map (rows: input channels, columns:
 * tokens), each row's largest magnitude as a bar against LLM.int8()'s
 * threshold of 6. Right: the output error of per-tensor INT8, vector-wise
 * INT8 and the mixed-precision decomposition, revealed as the steps reach
 * them (`outlierSteps`, one step per channel during the scan).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Stat } from "@/components/ui/Controls";
import { Hatch } from "@/components/interactive/Hatch";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtDb, sci, sqnrDb } from "@/lib/format";
import { outlierCaption } from "@/lib/num/captions";
import {
  LLM_INT8_THRESHOLD,
  demoOutlierLayer,
  outlierSteps,
} from "@/lib/num/model";
import { DEMO } from "@/lib/num/values";
import { METHOD_COLOUR, OKABE_ITO, STATE_COLOUR } from "@/lib/viz/palette";

const VW = 640;
const CELL = 12;

export default function OutlierWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [hover, setHover] = useState<string | null>(null);
  const L = useMemo(() => demoOutlierLayer(...DEMO.outlier), []);
  const r = useMemo(() => outlierSteps(L.W, L.X), [L]);
  const st = useStepper(r.steps.length, { stepMs: 900 });
  const s = r.steps[st.step]!;
  const font = useSvgFont(VW);
  const d = L.X.length;
  const n = L.X[0]!.length;
  const fz = font.fs(11);
  // heat map and bars side by side; the error bars in a row below
  const labW = fz * 1.6 + 4;
  const barW = font.narrow ? 170 : 180;
  const gap = fz * 2 + 10;
  const cw = (VW - labW - gap - barW - 8) / n;
  const ch = font.narrow ? fz * 1.05 : CELL;
  const gridW = n * cw;
  const barX = labW + gridW + gap;
  const amaxAll = Math.max(...r.amax);
  // bars on a square-root scale so the ordinary channels stay visible
  const bx = (v: number) => barX + Math.sqrt(v / amaxAll) * barW;
  const top = fz + 8;
  const H = d * ch;
  const phaseIdx = ["acts", "tensor", "vector", "scan", "mixed"].indexOf(
    s.phase,
  );
  const scanned = s.phase === "scan" ? s.i : -1;
  const isOut = (i: number) => s.found.includes(i);

  const errs = [
    { key: "tensor", label: "per-tensor INT8", v: r.err_tensor, at: 1 },
    { key: "vector", label: "vector-wise INT8", v: r.err_vector, at: 2 },
    { key: "mixed", label: "LLM.int8()", v: r.err_mixed, at: 4 },
  ];
  const lmax = Math.log10(Math.max(...errs.map((e) => e.v)));
  const lmin = Math.log10(Math.min(...errs.map((e) => e.v))) - 0.5;
  const eTop = top + H + fz + 26;
  const eLab = fz * 9;
  const eRow = fz * 1.6 + 6;
  const ex = (v: number) =>
    ((Math.log10(v) - lmin) / (lmax - lmin)) * (VW - eLab - fz * 6 - 8);
  const totalH = eTop + fz + 8 + 3 * eRow;

  const visual = (
    <svg
      ref={font.ref}
      viewBox={`0 0 ${VW} ${totalH}`}
      className="w-full"
      role="img"
      aria-label={`Heat map of ${d} input channels by ${n} tokens, with each channel's largest magnitude against the threshold ${LLM_INT8_THRESHOLD}, and the output error of three INT8 schemes`}
    >
      <defs>
        <Hatch id="ol-hatch" />
      </defs>
      <text x={labW} y={fz} fontSize={fz} fill="currentColor">
        {font.narrow ? "channel × token" : "activations: channel × token"}
      </text>
      <text x={barX} y={fz} fontSize={fz} fill="currentColor">
        {font.narrow ? "max |x|" : "max |x| per channel"}
      </text>
      {L.X.map((row, i) => (
        <g key={i}>
          {row.map((v, k) => (
            <rect
              key={k}
              x={labW + k * cw}
              y={top + i * ch}
              width={cw}
              height={ch}
              fill={v >= 0 ? STATE_COLOUR.positive : STATE_COLOUR.negative}
              fillOpacity={Math.sqrt(Math.abs(v) / amaxAll)}
            />
          ))}
          {(i % 2 === 1 || !font.narrow) && (
            <text
              x={labW - 4}
              y={top + i * ch + ch * 0.85}
              fontSize={font.narrow ? fz : font.fs(9)}
              textAnchor="end"
              fill="currentColor"
            >
              {i}
            </text>
          )}
          <rect
            x={barX}
            y={top + i * ch + 1}
            width={bx(r.amax[i]!) - barX}
            height={ch - 2}
            fill={isOut(i) ? STATE_COLOUR.error : "currentColor"}
            fillOpacity={isOut(i) ? 1 : 0.35}
          />
          {isOut(i) && (
            <rect
              x={barX}
              y={top + i * ch + 1}
              width={bx(r.amax[i]!) - barX}
              height={ch - 2}
              fill="url(#ol-hatch)"
            />
          )}
          {i === scanned && (
            <rect
              x={labW - 1}
              y={top + i * ch - 1}
              width={bx(amaxAll) - labW + 2}
              height={ch + 2}
              fill="none"
              stroke={STATE_COLOUR.active}
              strokeWidth={2}
            />
          )}
          {s.phase === "mixed" && isOut(i) && (
            <text
              x={labW + gridW + 4}
              y={top + i * ch + ch * 0.85}
              fontSize={fz}
              fill="currentColor"
              fontWeight={700}
            >
              16b
            </text>
          )}
        </g>
      ))}
      <line
        x1={bx(LLM_INT8_THRESHOLD)}
        x2={bx(LLM_INT8_THRESHOLD)}
        y1={top - 4}
        y2={top + H + 4}
        stroke={OKABE_ITO.vermillion}
        strokeDasharray="4 3"
      />
      <text
        x={bx(LLM_INT8_THRESHOLD)}
        y={top + H + fz + 4}
        fontSize={fz}
        textAnchor="middle"
        fill="currentColor"
      >
        threshold 6
      </text>
      <text x={0} y={eTop} fontSize={fz} fill="currentColor">
        output error (log scale)
      </text>
      {errs.map((e, k) => {
        const shown = phaseIdx >= e.at;
        const y = eTop + 8 + k * eRow;
        return (
          <g key={e.key} opacity={shown ? 1 : 0.2}>
            <text x={0} y={y + fz} fontSize={fz} fill="currentColor">
              {e.label}
            </text>
            <rect
              x={eLab}
              y={y + 2}
              width={Math.max(2, ex(e.v))}
              height={fz}
              fill={
                e.key === "mixed" ? METHOD_COLOUR.kahan : METHOD_COLOUR.naive
              }
            />
            <text
              x={eLab + Math.max(2, ex(e.v)) + 6}
              y={y + fz}
              fontSize={fz}
              fill="currentColor"
            >
              {shown ? sci(e.v, 2) : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );

  const hl =
    s.phase === "scan"
      ? "outl"
      : s.phase === "mixed"
        ? "outl int8"
        : s.phase === "acts"
          ? "x"
          : "int8";

  return (
    <AnimationPanel
      testId="outlier-widget"
      title="Outlier channels against INT8"
      summary={`A layer input with ${r.outliers.length} systematic outlier channels (illustrative), multiplied by an 8 × ${d} weight matrix: per-tensor INT8, vector-wise INT8, then LLM.int8()'s decomposition, all computed by the tested model.`}
      stepper={st}
      stepLabel="step"
      caption={outlierCaption(r, st.step, n)}
      visual={visual}
      equation={children}
      hl={hover ?? hl}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat
            label="Codes, ordinary channels"
            value={`${r.normal_levels} of 255`}
            hint="per-tensor INT8"
          />
          <Stat
            label="SQNR per tensor"
            value={fmtDb(sqnrDb(r.signal, r.err_tensor))}
          />
          <Stat
            label="SQNR LLM.int8()"
            value={phaseIdx >= 4 ? fmtDb(sqnrDb(r.signal, r.err_mixed)) : "–"}
          />
          <Stat
            label="Channels in 16-bit"
            value={`${s.found.length} of ${d}`}
          />
        </div>
      }
    />
  );
}
