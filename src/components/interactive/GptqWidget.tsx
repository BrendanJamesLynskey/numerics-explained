"use client";

/**
 * Chapter 7's hero: GPTQ on an 8 × 16 layer, one column per step
 * (`gptqSteps`, with the weights after each step from `gptq`). Top: the
 * weights, quantised columns to the left of the cursor, the not-yet-
 * quantised (and already nudged) columns to its right. Middle: how much
 * this step moved each remaining weight. Bottom: the layer's output error
 * so far, GPTQ against plain rounding of the same columns.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { trim } from "@/lib/format";
import { gptqCaption } from "@/lib/num/captions";
import { demoLayer, gptq, gptqSteps } from "@/lib/num/model";
import { DEMO } from "@/lib/num/values";
import { METHOD_COLOUR, STATE_COLOUR } from "@/lib/viz/palette";

const VW = 640;

export default function GptqWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [bits, setBits] = useState<"4" | "3">("4");
  const [hover, setHover] = useState<string | null>(null);
  const b = Number(bits);
  const lay = useMemo(() => demoLayer(...DEMO.layer), []);
  const r = useMemo(() => gptqSteps(lay.W, lay.X, b), [lay, b]);
  const snaps = useMemo(() => gptq(lay.W, lay.X, b).steps, [lay, b]);
  const st = useStepper(r.steps.length, { stepMs: 1100, resetKey: bits });
  const f = r.steps[st.step]!;
  const font = useSvgFont(VW);
  const rows = lay.W.length;
  const d = lay.W[0]!.length;
  const fz = font.fs(11);
  const left = fz * 2 + 6;
  const cell = (VW - left - 8) / d;
  const ch = font.narrow ? cell : Math.min(cell, 22);
  const cur = f.col < 0 ? lay.W : snaps[f.col]!.w;
  const shown = cur.map((row, i) =>
    row.map((w, j) => (j <= f.col ? r.Q[i]![j]! : w)),
  );
  let wmax = 0;
  for (const row of lay.W)
    for (const w of row) if (Math.abs(w) > wmax) wmax = Math.abs(w);
  let dmax = 0;
  for (const s of r.steps)
    for (const row of s.delta)
      for (const v of row) if (Math.abs(v) > dmax) dmax = Math.abs(v);

  const y1 = fz + 8;
  const H1 = rows * ch;
  const y2 = y1 + H1 + fz + 16;
  const y3 = y2 + H1 + fz + 18;
  const H3 = font.narrow ? 200 : 110;
  const emax = Math.max(...r.steps.map((s) => s.err_rtn));
  const X3 = (k: number) => left + ((k + 1) / d) * (VW - left - 8);
  const Y3 = (e: number) => y3 + H3 - (e / emax) * (H3 - 10);
  const line = (key: "err_gptq" | "err_rtn") =>
    r.steps
      .slice(0, st.step + 1)
      .map((s, k) => `${k ? "L" : "M"}${X3(s.col)} ${Y3(s[key])}`)
      .join(" ");

  const grid = (
    y: number,
    val: (i: number, j: number) => number,
    colour: (v: number) => string,
    max: number,
  ) =>
    Array.from({ length: rows }, (_, i) =>
      Array.from({ length: d }, (_, j) => {
        const v = val(i, j);
        return (
          <rect
            key={`${i}-${j}`}
            x={left + j * cell}
            y={y + i * ch}
            width={cell - 1}
            height={ch - 1}
            fill={colour(v)}
            fillOpacity={
              max > 0 ? Math.min(1, Math.sqrt(Math.abs(v) / max)) : 0
            }
          />
        );
      }),
    );

  const visual = (
    <svg
      ref={font.ref}
      viewBox={`0 0 ${VW} ${y3 + H3 + fz + 6}`}
      className="w-full"
      role="img"
      aria-label={`GPTQ at column ${f.col}: the weights, this step's updates and the layer error`}
    >
      <text x={left} y={fz} fontSize={fz} fill="currentColor">
        {font.narrow
          ? "weights: done ← | → to do"
          : "weights: quantised ← | → still to quantise (rows: outputs)"}
      </text>
      {grid(
        y1,
        (i, j) => shown[i]![j]!,
        (v) => (v >= 0 ? STATE_COLOUR.positive : STATE_COLOUR.negative),
        wmax,
      )}
      {f.col >= 0 && (
        <rect
          x={left + f.col * cell - 1}
          y={y1 - 2}
          width={cell + 1}
          height={H1 + 3}
          fill="none"
          stroke={STATE_COLOUR.active}
          strokeWidth={2.5}
        />
      )}
      <text x={left} y={y2 - 6} fontSize={fz} fill="currentColor">
        {font.narrow
          ? "this step’s |Δw|"
          : "this step’s update to each weight, |Δw|"}
      </text>
      {grid(
        y2,
        (i, j) => f.delta[i]![j]!,
        () => STATE_COLOUR.error,
        dmax,
      )}
      <text x={left} y={y3 - 6} fontSize={fz} fill="currentColor">
        layer output error so far
      </text>
      <path
        d={line("err_rtn")}
        fill="none"
        stroke={METHOD_COLOUR.naive}
        strokeWidth={2.5}
        strokeDasharray="6 4"
      />
      <path
        d={line("err_gptq")}
        fill="none"
        stroke={METHOD_COLOUR.kahan}
        strokeWidth={2.5}
      />
      <text
        x={VW - 8}
        y={y3 + 4 + fz}
        fontSize={fz}
        textAnchor="end"
        fill={METHOD_COLOUR.naive}
      >
        rounding (dashed)
      </text>
      <text
        x={VW - 8}
        y={y3 + 8 + 2 * fz}
        fontSize={fz}
        textAnchor="end"
        fill="currentColor"
      >
        GPTQ (solid)
      </text>
      <text x={left} y={y3 + H3 + fz + 2} fontSize={fz} fill="currentColor">
        column 0
      </text>
      <text
        x={VW - 8}
        y={y3 + H3 + fz + 2}
        fontSize={fz}
        textAnchor="end"
        fill="currentColor"
      >
        column {d - 1}
      </text>
    </svg>
  );

  return (
    <AnimationPanel
      testId="gptq-widget"
      title="GPTQ, one column at a time"
      summary={`An 8 × ${d} layer with 64 calibration inputs (illustrative), quantised to INT${b} per output channel by the tested GPTQ, against plain round-to-nearest of the same columns.`}
      stepper={st}
      stepLabel="column"
      caption={gptqCaption(f, b, d)}
      visual={visual}
      equation={children}
      hl={hover ?? (f.col < 0 ? "col" : "upd")}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Columns done" value={`${f.col + 1} of ${d}`} />
          <Stat label="Error, GPTQ" value={trim(f.err_gptq, 3)} />
          <Stat label="Error, rounding" value={trim(f.err_rtn, 3)} />
          <Stat
            label="Ratio"
            value={f.err_gptq > 0 ? `${trim(f.err_rtn / f.err_gptq, 3)}×` : "–"}
          />
        </div>
      }
      params={
        <Segmented
          label="Bits"
          value={bits}
          options={(["4", "3"] as const).map((v) => ({
            value: v,
            label: `INT${v}`,
          }))}
          onChange={setBits}
        />
      }
    />
  );
}
