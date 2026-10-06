"use client";

/**
 * Chapter 5's second animation: 24 skewed, mostly positive activations
 * put on two integer grids of the same width: absmax (symmetric about
 * zero, so the negative half of its codes go unused) and zero-point
 * (shifted to cover the values' actual range). One value snaps to both
 * grids per step (`zeropointSteps`).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { trim } from "@/lib/format";
import { zpCaption } from "@/lib/num/captions";
import { demoActivations, dequantInt, zeropointSteps } from "@/lib/num/model";
import { DEMO } from "@/lib/num/values";
import { METHOD_COLOUR, STATE_COLOUR } from "@/lib/viz/palette";

const VW = 640;
const ROWS = { vals: 34, sym: 100, asym: 160 };

export default function ZeroPointWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [bits, setBits] = useState<"8" | "4" | "3">("4");
  const [hover, setHover] = useState<string | null>(null);
  const xs = useMemo(() => demoActivations(...DEMO.acts), []);
  const b = Number(bits);
  const z = useMemo(() => zeropointSteps(xs, b), [xs, b]);
  const st = useStepper(z.steps.length, { stepMs: 700, resetKey: bits });
  const s = z.steps[st.step]!;
  const font = useSvgFont(VW);

  const amax = Math.max(...xs.map(Math.abs));
  const X = (v: number) => 24 + ((v + amax) / (2 * amax)) * (VW - 48);
  const symLevels = Array.from({ length: z.sym.hi - z.sym.lo + 1 }, (_, i) =>
    dequantInt(z.sym.lo + i, z.sym),
  );
  const asymLevels = Array.from({ length: z.asym.hi + 1 }, (_, q) =>
    dequantInt(q, z.asym),
  );
  const done = z.steps.slice(2, st.step + 1);
  const usedS = new Set(done.map((d) => d.q_sym));
  const usedA = new Set(done.map((d) => d.q_asym));
  const showGrids = s.phase !== "range";
  const cur = s.phase === "snap" ? s : null;
  let mn = 0;
  let mx = 0;
  for (const x of xs) {
    if (x < mn) mn = x;
    if (x > mx) mx = x;
  }

  const visual = (
    <svg
      ref={font.ref}
      viewBox={`0 0 ${VW} 200`}
      className="w-full"
      role="img"
      aria-label="The activations above two integer grids, absmax and zero-point"
    >
      <text x={4} y={ROWS.vals - 16} fontSize={font.fs(12)} fill="currentColor">
        values
      </text>
      {xs.map((x, i) => (
        <circle
          key={i}
          cx={X(x)}
          cy={ROWS.vals}
          r={i === cur?.i ? 6 : 4}
          fill={i === cur?.i ? STATE_COLOUR.active : "currentColor"}
          fillOpacity={i === cur?.i ? 1 : 0.55}
        />
      ))}
      {s.phase === "range" && (
        <path
          d={`M${X(mn)} ${ROWS.vals + 12} V${ROWS.vals + 18} H${X(mx)} V${ROWS.vals + 12}`}
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
        />
      )}
      {[
        {
          y: ROWS.sym,
          levels: symLevels,
          lo: z.sym.lo,
          used: usedS,
          colour: METHOD_COLOUR.rne,
          name: "absmax",
        },
        {
          y: ROWS.asym,
          levels: asymLevels,
          lo: 0,
          used: usedA,
          colour: METHOD_COLOUR.sr,
          name: "zero-point",
        },
      ].map((g) => (
        <g key={g.name} opacity={showGrids ? 1 : 0.15}>
          <text
            x={4}
            y={g.y - 16}
            fontSize={font.fs(12)}
            fill={g.colour}
            fontWeight={600}
          >
            {g.name}: {g.levels.length} codes
          </text>
          <line
            x1={X(-amax)}
            x2={X(amax)}
            y1={g.y}
            y2={g.y}
            stroke="currentColor"
            strokeOpacity={0.3}
          />
          {g.levels.map((v, i) => (
            <line
              key={i}
              x1={X(v)}
              x2={X(v)}
              y1={g.y - 8}
              y2={g.y + 8}
              stroke={g.colour}
              strokeWidth={g.used.has(g.lo + i) ? 3 : 1}
              strokeOpacity={g.used.has(g.lo + i) ? 1 : 0.45}
            />
          ))}
        </g>
      ))}
      {cur && (
        <g stroke={STATE_COLOUR.active} strokeWidth={2} fill="none">
          <path
            d={`M${X(xs[cur.i]!)} ${ROWS.vals + 6} L${X(cur.v_sym!)} ${ROWS.sym - 9}`}
          />
          <path
            d={`M${X(xs[cur.i]!)} ${ROWS.vals + 6} L${X(cur.v_asym!)} ${ROWS.asym - 9}`}
            strokeDasharray="5 3"
          />
        </g>
      )}
      <line
        x1={X(0)}
        x2={X(0)}
        y1={ROWS.vals - 10}
        y2={190}
        stroke="currentColor"
        strokeOpacity={0.25}
        strokeDasharray="2 3"
      />
      <text
        x={X(0)}
        y={198}
        fontSize={font.fs(11)}
        textAnchor="middle"
        fill="currentColor"
      >
        0
      </text>
      <text
        x={X(-amax)}
        y={198}
        fontSize={font.fs(11)}
        textAnchor="start"
        fill="currentColor"
      >
        −{trim(amax, 3)}
      </text>
      <text
        x={X(amax)}
        y={198}
        fontSize={font.fs(11)}
        textAnchor="end"
        fill="currentColor"
      >
        {trim(amax, 3)}
      </text>
    </svg>
  );

  const hl = s.phase === "grids" ? "zero" : s.phase === "snap" ? "q" : "scale";

  return (
    <AnimationPanel
      testId="zeropoint-widget"
      title="Absmax against zero-point"
      summary={`${xs.length} activations (illustrative, mostly positive), quantised to INT${b} both ways by the tested quantiser. Thick ticks: codes in use.`}
      stepper={st}
      stepLabel="step"
      caption={zpCaption(xs, z.sym, z.asym, s, b)}
      visual={visual}
      equation={children}
      hl={hover ?? hl}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat
            label="Codes used, absmax"
            value={`${usedS.size} of ${z.levels_sym}`}
          />
          <Stat
            label="Codes used, zero-point"
            value={`${usedA.size} of ${z.levels_asym}`}
          />
          <Stat label="MSE, absmax" value={cur ? trim(cur.mse_sym!, 3) : "–"} />
          <Stat
            label="MSE, zero-point"
            value={cur ? trim(cur.mse_asym!, 3) : "–"}
          />
        </div>
      }
      params={
        <Segmented
          label="Bits"
          value={bits}
          options={(["8", "4", "3"] as const).map((v) => ({
            value: v,
            label: `INT${v}`,
          }))}
          onChange={setBits}
        />
      }
    />
  );
}
