"use client";

/**
 * Chapter 4's hero: the formats side by side on one logarithmic axis, each
 * a bar from its smallest subnormal (hatched part: subnormals) to its
 * largest value. A probe x = (4/3)·2^k sweeps from tiny to huge, and each
 * format's row says what storing x does: the relative error inside the
 * normal range (set by the mantissa), growing errors among the subnormals,
 * zero below them, overflow above (set by the exponent). Driven by
 * `probeSteps`.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { sup, trim } from "@/lib/format";
import { pctOrSci, probeCaption } from "@/lib/num/captions";
import { FORMATS, INFO, PROBE_FORMATS, probeSteps } from "@/lib/num/model";
import { FIELD_COLOUR, STATE_COLOUR } from "@/lib/viz/palette";

const VW = 640;
const LO = -28;
const HI = 20;
const ROW = 50;
const BX0 = 8;
const BX1 = VW - 8;

export default function ProbeWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [hover, setHover] = useState<string | null>(null);
  const steps = useMemo(() => probeSteps(), []);
  const st = useStepper(steps.length, { stepMs: 600 });
  const s = steps[st.step]!;
  const font = useSvgFont(VW);
  const X = (l2: number) =>
    BX0 + ((Math.min(HI, Math.max(LO, l2)) - LO) / (HI - LO)) * (BX1 - BX0);
  const px = X(Math.log2(s.x));
  // rows sized from the label font (larger in viewBox units on a phone)
  const fl = font.fs(13);
  const barH = Math.max(14, font.fs(11) + 4);
  const rowH = Math.max(ROW, fl + 6 + barH + 12);
  const H = PROBE_FORMATS.length * rowH + font.fs(12) + 16;

  const visual = (
    <svg
      ref={font.ref}
      viewBox={`0 0 ${VW} ${H}`}
      className="w-full"
      role="img"
      aria-label={`Each format's range on a log scale, and the probe ${trim(s.x, 4)}`}
    >
      <defs>
        <pattern
          id="probe-sub"
          width={6}
          height={6}
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(45)"
        >
          <line
            x1={0}
            y1={0}
            x2={0}
            y2={6}
            stroke={FIELD_COLOUR.exp}
            strokeWidth={2}
          />
        </pattern>
      </defs>
      {PROBE_FORMATS.map((fid, r) => {
        const f = INFO[fid];
        const y = r * rowH;
        const by = y + fl + 6;
        const out = s.formats[fid]!;
        const lo = Math.log2(f.min_sub);
        const mn = Math.log2(f.min_normal);
        const mx = Math.log2(f.max);
        const text =
          out.status === "overflow"
            ? "overflows"
            : out.status === "underflow"
              ? "flushed to 0"
              : out.status === "saturated"
                ? `saturated at ${trim(out.value, 3)}`
                : out.rel === 0
                  ? "exact"
                  : `${out.status === "subnormal" ? "subnormal, " : ""}${pctOrSci(out.rel)} off`;
        const bad =
          out.status === "overflow" ||
          out.status === "underflow" ||
          out.status === "saturated";
        return (
          <g key={fid}>
            <text
              x={BX0}
              y={y + fl}
              fontSize={fl}
              fontWeight={600}
              fill="currentColor"
            >
              {FORMATS[fid].name}
            </text>
            <text
              x={BX1}
              y={y + fl}
              fontSize={fl}
              textAnchor="end"
              fill={bad ? STATE_COLOUR.error : "currentColor"}
              fontWeight={bad ? 700 : 400}
            >
              {text}
            </text>
            <rect
              x={X(lo)}
              y={by}
              width={Math.max(1, X(mn) - X(lo))}
              height={barH}
              fill="url(#probe-sub)"
              stroke={FIELD_COLOUR.exp}
            />
            <rect
              x={X(mn)}
              y={by}
              width={Math.max(1, X(mx) - X(mn))}
              height={barH}
              fill={FIELD_COLOUR.exp}
              fillOpacity={0.8}
            />
            {mx > HI && (
              <text
                x={BX1 - 2}
                y={by + barH - 3}
                fontSize={font.fs(11)}
                textAnchor="end"
                fill="#000"
              >
                → 2{sup(Math.round(mx))}
              </text>
            )}
            {lo < LO && (
              <text
                x={BX0 + 2}
                y={by + barH - 3}
                fontSize={font.fs(11)}
                fill="#000"
              >
                2{sup(Math.round(lo))} ←
              </text>
            )}
          </g>
        );
      })}
      <line
        x1={px}
        x2={px}
        y1={14}
        y2={H - font.fs(12) - 8}
        stroke={STATE_COLOUR.active}
        strokeWidth={2.5}
      />
      {[-24, -16, -8, 0, 8, 16].map((l) => (
        <text
          key={l}
          x={X(l)}
          y={H - 4}
          fontSize={font.fs(12)}
          textAnchor="middle"
          fill="currentColor"
        >
          2{sup(l)}
        </text>
      ))}
    </svg>
  );

  return (
    <AnimationPanel
      testId="probe-widget"
      title="Range and precision, side by side"
      summary="A probe value (4/3)·2^k, from 2^-26 to 2^18, stored in each format by its encoder (nearest-even, no saturation). Solid: normal range; hatched: subnormals."
      stepper={st}
      stepLabel="probe"
      caption={probeCaption(s)}
      visual={visual}
      equation={children}
      hl={hover ?? "mant"}
      onEquationHover={setHover}
      countFrom={0}
    />
  );
}
