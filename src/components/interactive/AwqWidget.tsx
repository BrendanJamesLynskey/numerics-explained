"use client";

/**
 * Chapter 8's AWQ animation: the search over alpha = k/8 (`awqSearch`).
 * Above: each input channel's mean |x| (grey) and the scale AWQ gives it at
 * this alpha (orange): the channels with large activations get large
 * scales, so their weights are rounded on a finer grid. Below: the output
 * error at every alpha, the current one highlighted, the best one ringed.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { trim } from "@/lib/format";
import { awqCaption } from "@/lib/num/captions";
import { awqSearch, demoLayer } from "@/lib/num/model";
import { DEMO } from "@/lib/num/values";
import { STATE_COLOUR } from "@/lib/viz/palette";

const VW = 640;

export default function AwqWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [bits, setBits] = useState<"3" | "4">("3");
  const [hover, setHover] = useState<string | null>(null);
  const b = Number(bits);
  const lay = useMemo(() => demoLayer(...DEMO.layer), []);
  const a = useMemo(() => awqSearch(lay.W, lay.X, b, 8), [lay, b]);
  const st = useStepper(a.results.length, { stepMs: 1300, resetKey: bits });
  const r = a.results[st.step]!;
  const font = useSvgFont(VW);
  const fz = font.fs(11);
  const d = a.meanAbs.length;
  const left = fz * 2 + 8;
  const slot = (VW - left - 8) / d;
  const H1 = font.narrow ? 240 : 150;
  const smax = Math.max(...a.results.flatMap((x) => x.scales));
  const mmax = Math.max(...a.meanAbs);
  const base = H1 - fz - 6;
  const Yb = (v: number, max: number) => base - (v / max) * (base - fz - 10);

  const top2 = H1 + 14;
  const H2 = font.narrow ? 200 : 120;
  const errs = a.results.map((x) => x.err);
  const emax = Math.max(...errs);
  const emin = Math.min(...errs);
  const X2 = (k: number) => left + (k / 8) * (VW - left - 30);
  const Y2 = (e: number) =>
    top2 +
    H2 -
    fz -
    10 -
    ((e - emin * 0.9) / (emax - emin * 0.9)) * (H2 - fz - 34);

  const visual = (
    <svg
      ref={font.ref}
      viewBox={`0 0 ${VW} ${top2 + H2}`}
      className="w-full"
      role="img"
      aria-label={`AWQ at alpha ${r.alpha}: per-channel activation size and scale, and the error for every alpha`}
    >
      <text x={left} y={fz} fontSize={fz} fill="currentColor">
        {font.narrow
          ? "mean |x| (grey), scale s (orange)"
          : "per input channel: mean |x| (grey), AWQ scale s (orange)"}
      </text>
      {a.meanAbs.map((m, i) => {
        const x0 = left + i * slot;
        const bw = slot * 0.4;
        return (
          <g key={i}>
            <rect
              x={x0 + slot * 0.08}
              y={Yb(m, mmax)}
              width={bw}
              height={base - Yb(m, mmax)}
              fill="currentColor"
              fillOpacity={0.35}
            />
            <rect
              x={x0 + slot * 0.08 + bw}
              y={Yb(r.scales[i]!, smax)}
              width={bw}
              height={base - Yb(r.scales[i]!, smax)}
              fill={STATE_COLOUR.positive}
            />
            {(!font.narrow || i % 3 === 0) && (
              <text
                x={x0 + slot / 2}
                y={base + fz + 2}
                fontSize={fz}
                textAnchor="middle"
                fill="currentColor"
              >
                {i}
              </text>
            )}
          </g>
        );
      })}
      <text x={left} y={top2 + fz} fontSize={fz} fill="currentColor">
        output error against α
      </text>
      <path
        d={a.results
          .map((x, k) => `${k ? "L" : "M"}${X2(k)} ${Y2(x.err)}`)
          .join(" ")}
        fill="none"
        stroke="currentColor"
        strokeOpacity={0.6}
      />
      {a.results.map((x, k) => (
        <g key={k}>
          {k === a.best && (
            <circle
              cx={X2(k)}
              cy={Y2(x.err)}
              r={11}
              fill="none"
              stroke="currentColor"
              strokeWidth={1.5}
            />
          )}
          <circle
            cx={X2(k)}
            cy={Y2(x.err)}
            r={k === st.step ? 7 : 4}
            fill={k === st.step ? STATE_COLOUR.active : "currentColor"}
            fillOpacity={k === st.step ? 1 : 0.45}
          />
        </g>
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
      testId="awq-widget"
      title="AWQ: scale the channels that matter"
      summary={`The 8 × ${d} layer of chapter 7, quantised to INT${b} in groups of 8 after scaling each input channel by mean|x|^α, by the tested search (α in eighths; AWQ itself searches a grid of 20).`}
      stepper={st}
      stepLabel="α step"
      caption={awqCaption(r, a.best, a.results[0]!.err, b)}
      visual={visual}
      equation={children}
      hl={hover ?? (st.step === 0 ? "s" : "alpha")}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="α" value={trim(r.alpha, 3)} plain />
          <Stat
            label="Largest / smallest s"
            value={trim(Math.max(...r.scales) / Math.min(...r.scales), 3)}
          />
          <Stat label="Output error" value={trim(r.err, 3)} />
          <Stat
            label="Best α"
            value={trim(a.results[a.best]!.alpha, 3)}
            plain
          />
        </div>
      }
      params={
        <Segmented
          label="Bits"
          value={bits}
          options={(["3", "4"] as const).map((v) => ({
            value: v,
            label: `INT${v}`,
          }))}
          onChange={setBits}
        />
      }
    />
  );
}
