"use client";

/**
 * Chapter 4's MX animation: one block of 32 values converted by the MX
 * spec's section 6.3, step by step (`mxSteps`, `mxQuantise`). Above: the
 * values as bars, the largest picked out, then the range the shared scale
 * allows, then each element's stored value as a dot. Below: the element
 * grid (the element format's own values) with each scaled input V/X
 * snapping onto it. An outlier stretches the scale, and the small values
 * fall into the grid's gap around zero.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { pow2Text, trim } from "@/lib/format";
import { mxCaption } from "@/lib/num/captions";
import {
  INFO,
  MX_FORMATS,
  MX_ORDER,
  demoBlock,
  mxQuantise,
  mxSteps,
  positiveValues,
  roundTo,
  type BlockKind,
  type MxId,
} from "@/lib/num/model";
import { DEMO } from "@/lib/num/values";
import { FIELD_COLOUR, STATE_COLOUR } from "@/lib/viz/palette";

const VW = 640;

function elementGrid(mid: MxId): number[] {
  const elem = MX_FORMATS[mid].elem;
  if (elem === "int8")
    return Array.from({ length: 255 }, (_, i) => (i - 127) / 64);
  const pos = positiveValues(elem).filter((v) => v <= INFO[elem].max);
  return [
    ...pos
      .slice(1)
      .map((v) => -v)
      .reverse(),
    ...pos,
  ];
}

export default function MxWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [mid, setMid] = useState<MxId>("mxfp4");
  const [kind, setKind] = useState<BlockKind>("outlier");
  const [hover, setHover] = useState<string | null>(null);
  const block = useMemo(() => demoBlock(kind, DEMO.blockSeed), [kind]);
  const r = useMemo(() => mxQuantise(block, mid), [block, mid]);
  const steps = useMemo(() => mxSteps(block, mid), [block, mid]);
  const grid = useMemo(() => elementGrid(mid), [mid]);
  const st = useStepper(steps.length, {
    stepMs: 600,
    resetKey: `${mid}-${kind}`,
  });
  const s = steps[st.step]!;
  const f = MX_FORMATS[mid];
  const font = useSvgFont(VW);
  const gfont = useSvgFont(VW);
  const X = 2 ** r.shared_exp;
  const emax = f.elem === "int8" ? 127 / 64 : INFO[f.elem].max;

  const done = s.done;
  let zeros = 0;
  for (let i = 0; i < done; i++)
    if (r.values[i] === 0 && block[i] !== 0) zeros += 1;
  const mse = done ? (s.sq_err ?? 0) / done : 0;

  // the same values in the element format with no shared scale
  const plain = useMemo(() => {
    if (f.elem === "int8") return null;
    const elem = f.elem;
    let sq = 0;
    let z = 0;
    for (const v of block) {
      const q = roundTo(v, elem, "rne", true);
      sq += (v - q) * (v - q);
      if (q === 0 && v !== 0) z += 1;
    }
    return { mse: sq / block.length, zeros: z };
  }, [block, f.elem]);

  // bars
  const amax = r.amax;
  const H1 = 170;
  const mid1 = H1 / 2;
  const half = H1 / 2 - 14;
  const bw = (VW - 20) / block.length;
  // the bars' scale covers the block and the range the shared scale allows
  const top = Math.max(amax, emax * X);
  const BY = (v: number) => mid1 - (v / top) * half;
  const showScale = s.phase === "scale" || s.phase === "element";

  // element grid
  const H2 = 110;
  const span = emax * 1.08;
  const GX = (t: number) => 10 + ((t + span) / (2 * span)) * (VW - 20);
  const cur = s.phase === "element" ? s.i : -1;

  const visual = (
    <div className="min-w-0">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${VW} ${H1}`}
        className="w-full"
        role="img"
        aria-label="The block's 32 values as bars, with the stored values as dots"
      >
        <line
          x1={10}
          x2={VW - 10}
          y1={mid1}
          y2={mid1}
          stroke="currentColor"
          strokeOpacity={0.4}
        />
        {showScale && (
          <g>
            {[1, -1].map((sg) => (
              <line
                key={sg}
                x1={10}
                x2={VW - 10}
                y1={BY(sg * emax * X)}
                y2={BY(sg * emax * X)}
                stroke={FIELD_COLOUR.scale}
                strokeWidth={2}
                strokeDasharray="6 4"
              />
            ))}
            <text
              x={VW - 12}
              y={BY(emax * X) + font.fs(12) + 2}
              fontSize={font.fs(12)}
              textAnchor="end"
              fill="currentColor"
            >
              {font.narrow
                ? `±${trim(emax * X, 4)}`
                : `±${trim(emax * X, 4)} = largest element × X`}
            </text>
          </g>
        )}
        {block.map((v, i) => {
          const isMax = s.phase !== "raw" && i === steps[1]!.i;
          const isCur = i === cur;
          const y0 = Math.min(BY(v), mid1);
          return (
            <g key={i}>
              <rect
                x={10 + i * bw + 1}
                y={y0}
                width={bw - 2}
                height={Math.max(1, Math.abs(BY(v) - mid1))}
                fill={
                  isCur
                    ? STATE_COLOUR.active
                    : isMax
                      ? FIELD_COLOUR.scale
                      : "#a3a3a3"
                }
                fillOpacity={i < done && !isCur ? 0.45 : 0.9}
              />
              {i < done && (
                <circle
                  cx={10 + i * bw + bw / 2}
                  cy={BY(r.values[i]!)}
                  r={Math.min(4, bw / 2.5)}
                  fill={
                    r.values[i] === 0 && v !== 0
                      ? STATE_COLOUR.error
                      : FIELD_COLOUR.mant
                  }
                  stroke="#000"
                  strokeWidth={0.5}
                />
              )}
            </g>
          );
        })}
      </svg>
      <p className="mt-2 text-[0.7rem] font-medium uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
        The element grid: every value V / X can become
      </p>
      <svg
        ref={gfont.ref}
        viewBox={`0 0 ${VW} ${H2}`}
        className="w-full"
        role="img"
        aria-label="The element format's values on a line, with the scaled inputs snapping to them"
      >
        <line
          x1={10}
          x2={VW - 10}
          y1={50}
          y2={50}
          stroke="currentColor"
          strokeOpacity={0.4}
        />
        {grid.map((g, i) => (
          <line
            key={i}
            x1={GX(g)}
            x2={GX(g)}
            y1={42}
            y2={58}
            stroke={FIELD_COLOUR.mant}
            strokeWidth={1}
          />
        ))}
        {/* elements already stored */}
        {r.values.slice(0, done).map((q, i) => (
          <circle
            key={i}
            cx={GX(q / X)}
            cy={66 + (i % 4) * 7}
            r={3}
            fill={
              q === 0 && block[i] !== 0 ? STATE_COLOUR.error : FIELD_COLOUR.mant
            }
          />
        ))}
        {cur >= 0 && (
          <g>
            <circle
              cx={GX(Math.max(-span, Math.min(span, r.scaled[cur]!)))}
              cy={24}
              r={5}
              fill={STATE_COLOUR.active}
            />
            <path
              d={`M${GX(Math.max(-span, Math.min(span, r.scaled[cur]!)))} 29 L${GX(r.values[cur]! / X)} 46`}
              stroke={STATE_COLOUR.active}
              strokeWidth={2.5}
            />
            <text
              x={Math.min(VW - 60, Math.max(60, GX(r.scaled[cur]!)))}
              y={gfont.fs(12)}
              fontSize={gfont.fs(12)}
              textAnchor="middle"
              fill="currentColor"
            >
              V/X = {trim(r.scaled[cur]!, 4)}
            </text>
          </g>
        )}
        <text x={10} y={H2 - 2} fontSize={gfont.fs(12)} fill="currentColor">
          −{trim(emax, 4)}
        </text>
        <text
          x={GX(0)}
          y={H2 - 2}
          fontSize={gfont.fs(12)}
          textAnchor="middle"
          fill="currentColor"
        >
          0
        </text>
        <text
          x={VW - 10}
          y={H2 - 2}
          fontSize={gfont.fs(12)}
          textAnchor="end"
          fill="currentColor"
        >
          {trim(emax, 4)}
        </text>
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Orange bar: the largest magnitude, which sets the scale. Dots: stored
        values; red dots were flushed to zero.
      </p>
    </div>
  );

  const hl =
    s.phase === "amax"
      ? "amax"
      : s.phase === "scale"
        ? "scale"
        : s.phase === "element"
          ? "elem"
          : "";

  return (
    <AnimationPanel
      testId="mx-widget"
      title="An MX block: 32 elements, one shared scale"
      summary="The conversion of the MX spec, section 6.3, run by the tested model: find the largest magnitude, pick the power-of-two scale X, store each V / X in the element format."
      stepper={st}
      stepLabel="step"
      caption={mxCaption(mid, block, r, s)}
      visual={visual}
      equation={children}
      hl={hover ?? hl}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat
            label="Shared scale X"
            value={pow2Text(X)}
            hint={`E8M0 code ${r.scale_code}`}
          />
          <Stat
            label="MSE so far"
            value={trim(mse, 3)}
            hint={`${zeros} of ${done} flushed to 0`}
          />
          <Stat
            label="Bits per value"
            value={trim(f.bits + 8 / block.length, 4)}
            hint={`${f.bits} + 8 / ${block.length}`}
          />
          <Stat
            label="Without the scale"
            value={plain ? trim(plain.mse, 3) : "n/a"}
            hint={
              plain ? `MSE; ${plain.zeros} flushed to 0` : "INT8 needs a scale"
            }
          />
        </div>
      }
      params={
        <>
          <Segmented
            label="MX format"
            value={mid}
            options={MX_ORDER.map((m) => ({
              value: m,
              label: MX_FORMATS[m].name,
            }))}
            onChange={setMid}
          />
          <Segmented
            label="Block"
            value={kind}
            options={[
              { value: "outlier", label: "with an outlier" },
              { value: "normal", label: "normal" },
              { value: "tiny", label: "tiny values" },
            ]}
            onChange={setKind}
          />
        </>
      }
    />
  );
}
