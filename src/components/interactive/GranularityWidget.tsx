"use client";

/**
 * Chapter 5's hero: one weight matrix (16 output channels × 64 inputs)
 * quantised to INT-b with ever finer scales: per tensor, per channel, then
 * groups of 32, 16 and 8 weights along each row (`granularitySteps`).
 * Above: the weights as a heat map (orange positive, blue negative), with
 * the current groups outlined. Below: the quantisation error of every
 * weight, on one scale for all steps, so it visibly fades as the groups
 * shrink. The scales cost bits: the readout counts them.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtDb, int, sqnrDb, trim } from "@/lib/format";
import { granCaption } from "@/lib/num/captions";
import { demoWeights, granularitySteps, type Scheme } from "@/lib/num/model";
import { DEMO } from "@/lib/num/values";
import { STATE_COLOUR } from "@/lib/viz/palette";

const VW = 640;
const CELL = 10;

export default function GranularityWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [bits, setBits] = useState<"8" | "4" | "3">("4");
  const [scheme, setScheme] = useState<Scheme>("sym");
  const [hover, setHover] = useState<string | null>(null);
  const W = useMemo(() => demoWeights(...DEMO.weights), []);
  const b = Number(bits);
  const steps = useMemo(() => granularitySteps(W, b, scheme), [W, b, scheme]);
  const st = useStepper(steps.length, {
    stepMs: 1800,
    resetKey: `${bits}-${scheme}`,
  });
  const s = steps[st.step]!;
  const font = useSvgFont(VW);
  const efont = useSvgFont(VW);
  const rows = W.length;
  const cols = W[0]!.length;
  const H = rows * CELL;

  let wmax = 0;
  for (const r of W)
    for (const w of r) if (Math.abs(w) > wmax) wmax = Math.abs(w);
  const emax = Math.max(...steps.map((x) => x.max_err));

  const outlines = s.params.map((p, i) => {
    const r0 = p.row < 0 ? 0 : p.row;
    const r1 = p.row < 0 ? rows : p.row + 1;
    return (
      <rect
        key={i}
        x={p.c0 * CELL}
        y={r0 * CELL}
        width={(p.c1 - p.c0) * CELL}
        height={(r1 - r0) * CELL}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.2}
      />
    );
  });

  const visual = (
    <div className="min-w-0">
      <p className="text-[0.7rem] font-medium uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
        Weights (rows: output channels), outlined by the groups that share a
        scale
      </p>
      <svg
        ref={font.ref}
        viewBox={`-2 -2 ${VW + 4} ${H + 4}`}
        className="mt-1 w-full"
        role="img"
        aria-label={`Heat map of the ${rows} by ${cols} weight matrix with ${s.params.length} groups outlined`}
      >
        {W.map((row, i) =>
          row.map((w, j) => (
            <rect
              key={`${i}-${j}`}
              x={j * CELL}
              y={i * CELL}
              width={CELL}
              height={CELL}
              fill={w >= 0 ? STATE_COLOUR.positive : STATE_COLOUR.negative}
              fillOpacity={Math.sqrt(Math.abs(w) / wmax)}
            />
          )),
        )}
        {outlines}
      </svg>
      <p className="mt-3 text-[0.7rem] font-medium uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
        |error| of each weight after INT{b} (same scale at every step)
      </p>
      <svg
        ref={efont.ref}
        viewBox={`-2 -2 ${VW + 4} ${H + 4}`}
        className="mt-1 w-full"
        role="img"
        aria-label="Heat map of the quantisation error of every weight"
        data-testid="error-map"
      >
        {s.deq.map((row, i) =>
          row.map((q, j) => (
            <rect
              key={`${i}-${j}`}
              x={j * CELL}
              y={i * CELL}
              width={CELL}
              height={CELL}
              fill={STATE_COLOUR.error}
              fillOpacity={Math.abs(W[i]![j]! - q) / emax}
            />
          )),
        )}
        {outlines}
      </svg>
    </div>
  );

  return (
    <AnimationPanel
      testId="granularity-widget"
      title="One scale per tensor, per channel, per group"
      summary={`The same matrix quantised to INT${b} (${scheme === "sym" ? "absmax" : "zero-point"}) five times, each with finer scales, by the tested quantiser.`}
      stepper={st}
      stepLabel="granularity"
      caption={granCaption(
        {
          gran: s.gran,
          group: s.group,
          mse: s.mse,
          max_err: s.max_err,
          bits_per_weight: s.bits_per_weight,
          nparams: s.params.length,
        },
        b,
        scheme,
        s.signal,
        rows * cols,
      )}
      visual={visual}
      equation={children}
      hl={hover ?? (st.step === 0 ? "scale" : "group")}
      onEquationHover={setHover}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Scales" value={int(s.params.length)} />
          <Stat label="SQNR" value={fmtDb(sqnrDb(s.signal, s.mse))} />
          <Stat label="Worst error" value={trim(s.max_err, 3)} />
          <Stat
            label="Bits per weight"
            value={trim(s.bits_per_weight, 4)}
            hint="FP16 scales"
          />
        </div>
      }
      params={
        <>
          <Segmented
            label="Bits"
            value={bits}
            options={(["8", "4", "3"] as const).map((v) => ({
              value: v,
              label: `INT${v}`,
            }))}
            onChange={setBits}
          />
          <Segmented
            label="Scheme"
            value={scheme}
            options={[
              { value: "sym", label: "absmax (symmetric)" },
              { value: "asym", label: "zero-point" },
            ]}
            onChange={setScheme}
          />
        </>
      }
    />
  );
}
