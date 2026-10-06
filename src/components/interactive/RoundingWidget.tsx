"use client";

/**
 * Chapter 2's hero: inputs between two neighbouring values of a format,
 * one per step, each rounded to nearest (ties to even) and stochastically.
 * Above: a close-up of the number line around the input, with the share of
 * the gap that sends stochastic rounding up shaded, and the random draw u.
 * Below: the two error histograms building up, with their means (the
 * bias). Driven by `roundingSteps`.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { exact, fmtUlp, trim } from "@/lib/format";
import { roundCaption } from "@/lib/num/captions";
import {
  FORMATS,
  HIST_BINS,
  INFO,
  TWO32,
  roundingSteps,
  type FormatId,
  type RoundDist,
} from "@/lib/num/model";
import { DEMO } from "@/lib/num/values";
import { METHOD_COLOUR } from "@/lib/viz/palette";

const VW = 640;

function Histogram({
  counts,
  mean,
  colour,
  label,
  font,
  total,
  w = 300,
}: {
  counts: number[];
  mean: number;
  colour: string;
  label: string;
  font: (u: number) => number;
  total: number;
  w?: number;
}): JSX.Element {
  const h = 120;
  const bw = w / HIST_BINS;
  const peak = Math.max(4, total / 3);
  const mx = ((mean + 1) / 2) * w;
  return (
    <g>
      <text x={0} y={font(12)} fontSize={font(12)} fill="currentColor">
        {label}
      </text>
      <g transform={`translate(0 ${font(12) + 8})`}>
        {counts.map((c, i) => {
          const bh = Math.min(h, (c / peak) * h);
          return (
            <rect
              key={i}
              x={i * bw + 1}
              y={h - bh}
              width={bw - 2}
              height={bh}
              fill={colour}
              fillOpacity={0.85}
            />
          );
        })}
        <line
          x1={0}
          x2={w}
          y1={h}
          y2={h}
          stroke="currentColor"
          strokeOpacity={0.4}
        />
        <line
          x1={w / 2}
          x2={w / 2}
          y1={0}
          y2={h}
          stroke="currentColor"
          strokeOpacity={0.3}
          strokeDasharray="3 3"
        />
        <line
          x1={mx}
          x2={mx}
          y1={-4}
          y2={h + 4}
          stroke={colour}
          strokeWidth={3}
        />
        <text
          x={0}
          y={h + 4 + font(12)}
          fontSize={font(12)}
          fill="currentColor"
        >
          −1 ulp
        </text>
        <text
          x={w / 2}
          y={h + 4 + font(12)}
          fontSize={font(12)}
          textAnchor="middle"
          fill="currentColor"
        >
          0
        </text>
        <text
          x={w}
          y={h + 4 + font(12)}
          fontSize={font(12)}
          textAnchor="end"
          fill="currentColor"
        >
          +1
        </text>
      </g>
    </g>
  );
}

export default function RoundingWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [fid, setFid] = useState<FormatId>("e4m3");
  const [dist, setDist] = useState<RoundDist>("low");
  const [hover, setHover] = useState<string | null>(null);
  const n = DEMO.roundN;
  const steps = useMemo(
    () => roundingSteps(fid, n, dist, DEMO.roundSeed),
    [fid, dist, n],
  );
  const st = useStepper(steps.length, {
    stepMs: 650,
    resetKey: `${fid}-${dist}`,
  });
  const s = steps[st.step]!;
  const ulp = INFO[fid].eps;
  const font = useSvgFont(VW);
  const hfont = useSvgFont(VW);

  const absMeans = useMemo(() => {
    let a = 0;
    let b = 0;
    for (const x of steps.slice(0, st.step + 1)) {
      a += Math.abs(x.err_rne);
      b += Math.abs(x.err_sr);
    }
    return [a / (st.step + 1), b / (st.step + 1)];
  }, [steps, st.step]);

  // the close-up: one ulp either side of the gap [down, up]
  const lo = s.down - ulp;
  const span = 3 * ulp;
  const X = (v: number) => 20 + ((v - lo) / span) * (VW - 40);
  const u = s.u / TWO32;
  const tie = s.frac === 0.5;
  const up = s.up !== s.down;
  // rows, sized from the label font so nothing collides on a phone
  const f = font.fs(12);
  const yX = f + 2;
  const yNear = yX + f + 6;
  const yLine = yNear + 26;
  const yTick = yLine + 12 + f;
  const yArrow = yTick + 12;
  const ySr = yArrow + f + 4;
  const H = ySr + 6;
  const anchor = (to: number) =>
    X(to) > VW * 0.7 ? "end" : X(to) < VW * 0.3 ? "start" : "middle";
  const narrow = hfont.narrow;
  const HW = narrow ? VW : 300;
  const HH = 2 * hfont.fs(12) + 8 + 120 + 4 + 8;

  const visual = (
    <div className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${VW} ${H}`}
        className="w-full"
        role="img"
        aria-label={`Input ${exact(s.x)} between ${exact(s.down)} and ${exact(s.up)}, rounded both ways`}
      >
        <line
          x1={10}
          x2={VW - 10}
          y1={yLine}
          y2={yLine}
          stroke="currentColor"
          strokeOpacity={0.4}
        />
        {/* the share of the gap that sends stochastic rounding up */}
        {up && (
          <rect
            x={X(s.down)}
            y={yLine - 8}
            width={X(s.down + s.frac * ulp) - X(s.down)}
            height={16}
            fill={METHOD_COLOUR.sr}
            fillOpacity={0.3}
          />
        )}
        {[s.down - ulp, s.down, s.down + ulp, s.down + 2 * ulp].map((v, i) => (
          <g key={i}>
            <line
              x1={X(v)}
              x2={X(v)}
              y1={yLine - 12}
              y2={yLine + 12}
              stroke="currentColor"
              strokeWidth={i === 1 || i === 2 ? 2 : 1}
            />
            {(i === 1 || i === 2) && (
              <text
                x={X(v)}
                y={yTick}
                fontSize={f}
                textAnchor="middle"
                fill="currentColor"
              >
                {narrow ? trim(v, 6) : exact(v)}
              </text>
            )}
          </g>
        ))}
        {/* the input */}
        <circle cx={X(s.x)} cy={yLine} r={6} fill="currentColor" />
        <text
          x={X(s.x)}
          y={yX}
          fontSize={f}
          textAnchor={anchor(s.x)}
          fill="currentColor"
        >
          x = {narrow ? trim(s.x, 6) : exact(s.x)}
        </text>
        {/* nearest-even: an arrow above the line */}
        <path
          d={`M${X(s.x)} ${yNear + 6} L${X(s.rne)} ${yNear + 6} L${X(s.rne)} ${yLine - 14}`}
          fill="none"
          stroke={METHOD_COLOUR.rne}
          strokeWidth={3}
        />
        <text
          x={X(s.rne) + (X(s.rne) < VW / 2 ? 6 : -6)}
          y={yNear}
          fontSize={f}
          textAnchor={X(s.rne) < VW / 2 ? "start" : "end"}
          fill={METHOD_COLOUR.rne}
          fontWeight={600}
        >
          {tie ? "nearest, tie → even" : "nearest"}
        </text>
        {/* stochastic: an arrow below, with the draw u on the gap */}
        <path
          d={`M${X(s.x)} ${yArrow} L${X(s.sr)} ${yArrow} L${X(s.sr)} ${yLine + 14}`}
          fill="none"
          stroke={METHOD_COLOUR.sr}
          strokeWidth={3}
          strokeDasharray="7 4"
        />
        {up && (
          <line
            x1={X(s.down + u * ulp)}
            x2={X(s.down + u * ulp)}
            y1={yLine - 10}
            y2={yLine + 10}
            stroke={METHOD_COLOUR.sr}
            strokeWidth={2}
          />
        )}
        <text
          x={X(s.sr) + (X(s.sr) < VW / 2 ? 6 : -6)}
          y={ySr}
          fontSize={f}
          textAnchor={X(s.sr) < VW / 2 ? "start" : "end"}
          fill={METHOD_COLOUR.sr}
          fontWeight={600}
        >
          stochastic (u = {u.toFixed(2)})
        </text>
      </svg>
      <svg
        ref={hfont.ref}
        viewBox={`0 0 ${VW} ${narrow ? 2 * HH + 10 : HH}`}
        className="mt-2 w-full"
        role="img"
        aria-label="Histograms of the rounding errors so far, nearest-even and stochastic"
      >
        <Histogram
          counts={s.hist_rne}
          mean={s.mean_rne}
          colour={METHOD_COLOUR.rne}
          label={`Nearest-even errors, mean ${fmtUlp(s.mean_rne, 3)}`}
          font={hfont.fs}
          total={n}
          w={HW}
        />
        <g transform={narrow ? `translate(0 ${HH + 10})` : "translate(340 0)"}>
          <Histogram
            counts={s.hist_sr}
            mean={s.mean_sr}
            colour={METHOD_COLOUR.sr}
            label={`Stochastic errors, mean ${fmtUlp(s.mean_sr, 3)}`}
            font={hfont.fs}
            total={n}
            w={HW}
          />
        </g>
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Shaded orange: the part of the gap where a uniform draw u sends
        stochastic rounding up; its width is the input&apos;s distance above the
        lower neighbour. The thick line in each histogram is the mean error (the
        bias).
      </p>
    </div>
  );

  return (
    <AnimationPanel
      testId="rounding-widget"
      title="Nearest-even against stochastic rounding"
      summary={`${n} inputs between neighbouring ${FORMATS[fid].name} values, each rounded both ways by the format's encoder; errors in ulps.`}
      stepper={st}
      stepLabel="input"
      caption={roundCaption(fid, s, n)}
      visual={visual}
      equation={children}
      hl={hover ?? (tie ? "rne" : "frac")}
      onEquationHover={setHover}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Bias, nearest" value={fmtUlp(s.mean_rne, 3)} />
          <Stat label="Bias, stochastic" value={fmtUlp(s.mean_sr, 3)} />
          <Stat label="Mean |error|, nearest" value={fmtUlp(absMeans[0]!, 3)} />
          <Stat
            label="Mean |error|, stochastic"
            value={fmtUlp(absMeans[1]!, 3)}
          />
        </div>
      }
      params={
        <>
          <Segmented
            label="Format"
            value={fid}
            options={(["e4m3", "e2m1", "fp16"] as const).map((v) => ({
              value: v,
              label: FORMATS[v].name,
            }))}
            onChange={setFid}
          />
          <Segmented
            label="Inputs"
            value={dist}
            options={[
              { value: "low", label: "just above a value" },
              { value: "uniform", label: "anywhere" },
              { value: "ties", label: "exact ties" },
            ]}
            onChange={setDist}
          />
        </>
      }
    />
  );
}
