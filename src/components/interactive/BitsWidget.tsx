"use client";

/**
 * Chapter 1's hero. Above: the code of one floating-point number, a button
 * per bit, coloured by field (sign, exponent, mantissa); click a bit and
 * the value, its decoding and its place on the number line follow. Below:
 * the animation, a linear number line from 0 to hi that zooms towards zero
 * one power of two at a time (`zoomSteps`), drawing every representable
 * value: each binade holds the same number of values in half the width, so
 * they crowd towards zero, until the subnormals space them evenly.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { exact, int, pow2Text } from "@/lib/format";
import { zoomCaption } from "@/lib/num/captions";
import {
  FORMATS,
  INFO,
  decode,
  encode,
  exponentOf,
  fields,
  pow2,
  zoomSteps,
  type FormatId,
} from "@/lib/num/model";
import { FIELD_COLOUR, STATE_COLOUR } from "@/lib/viz/palette";

const CHOICES: FormatId[] = [
  "e2m1",
  "e3m2",
  "e2m3",
  "e4m3",
  "e5m2",
  "bf16",
  "fp16",
  "fp32",
];

type Field = "sign" | "exp" | "mant";

const VW = 640;
const LINE_Y = 58;

/** Tick positions (px) and solid bands where values are under 3 px apart. */
function ticks(
  fid: FormatId,
  hi: number,
): { path: string; bands: [number, number][] } {
  const f = INFO[fid];
  const px = VW / hi;
  let path = "";
  const bands: [number, number][] = [];
  const kTop = Math.min(exponentOf(hi), f.emax);
  for (let k = kTop; k >= f.emin - 1; k--) {
    const lo = k >= f.emin ? pow2(k) : 0;
    const top = k >= f.emin ? pow2(k + 1) : f.min_normal;
    const sp = k >= f.emin ? pow2(k - f.m) : f.min_sub;
    if (top * px < 0.5) {
      bands.push([0, top * px]);
      break;
    }
    if (sp * px >= 3) {
      for (let v = lo; v < top && v <= hi && v <= f.max; v += sp) {
        const x = v * px;
        path += `M${x.toFixed(2)} ${LINE_Y - 9}V${LINE_Y + 9}`;
      }
    } else bands.push([lo * px, Math.min(top, hi, f.max) * px]);
  }
  return { path, bands };
}

function kindOf(v: number, fid: FormatId): string {
  if (Number.isNaN(v)) return "NaN";
  if (!Number.isFinite(v)) return "infinity";
  if (v === 0) return "zero";
  return Math.abs(v) < INFO[fid].min_normal ? "subnormal" : "normal";
}

export default function BitsWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [sel, setSel] = useState<{ fid: FormatId; code: number }>({
    fid: "e4m3",
    code: encode(13, "e4m3"),
  });
  const [typed, setTyped] = useState("13");
  const [hover, setHover] = useState<Field | null>(null);
  const { fid, code } = sel;
  const f = INFO[fid];
  const fd = fields(code, fid);
  const value = decode(code, fid);

  const steps = useMemo(() => zoomSteps(fid), [fid]);
  const st = useStepper(steps.length, {
    stepMs: 900,
    resetKey: fid,
    smooth: true,
  });
  const s = steps[st.step]!;
  const next = steps[st.step + 1];
  // between two model states the view zooms smoothly (geometrically)
  const hi = next && st.frac > 0 ? s.hi * (next.hi / s.hi) ** st.frac : s.hi;
  const tk = useMemo(() => ticks(fid, hi), [fid, hi]);
  const font = useSvgFont(VW);

  const setFormat = (nf: FormatId) => {
    const c = encode(decode(code, fid), nf);
    setSel({ fid: nf, code: c });
    setTyped(exact(decode(c, nf)));
  };
  const setCode = (c: number) => {
    setSel({ fid, code: c });
    setTyped(exact(decode(c, fid)));
  };
  const flip = (b: number) => {
    const bit = Math.floor(code / pow2(b)) % 2;
    setCode(bit ? code - pow2(b) : code + pow2(b));
  };
  const fieldOf = (b: number): Field =>
    b === f.bits - 1 ? "sign" : b >= f.m ? "exp" : "mant";

  const mag = Math.abs(value);
  const inView = Number.isFinite(mag) && mag <= hi;
  const markX = inView ? (mag / hi) * VW : VW;

  const sub = fd.E === 0;
  const decoding = Number.isNaN(value) ? (
    <span>NaN (not a number): this exponent and mantissa are reserved</span>
  ) : !Number.isFinite(value) ? (
    <span>
      {value > 0 ? "+∞" : "−∞"}: the all-ones exponent with mantissa 0
    </span>
  ) : (
    <span>
      (−1)
      <sup style={{ borderBottom: `2px solid ${FIELD_COLOUR.sign}` }}>
        {fd.s}
      </sup>{" "}
      × 2
      <sup style={{ borderBottom: `2px solid ${FIELD_COLOUR.exp}` }}>
        {sub ? `1 − ${f.bias}` : `${fd.E} − ${f.bias}`}
      </sup>{" "}
      × (<span>{sub ? "0" : "1"}</span> +{" "}
      <span style={{ borderBottom: `2px solid ${FIELD_COLOUR.mant}` }}>
        {fd.M}/{int(2 ** f.m)}
      </span>
      ) = <strong>{exact(value)}</strong>
    </span>
  );

  const visual = (
    <div className="min-w-0">
      <p className="text-[0.7rem] font-medium uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
        The {f.bits} bits of one {FORMATS[fid].name} code (click to flip)
      </p>
      <div
        className="mt-1 flex flex-wrap gap-1"
        role="group"
        aria-label={`The ${f.bits} bits, most significant first`}
        data-testid="bits"
      >
        {Array.from({ length: f.bits }, (_, i) => f.bits - 1 - i).map((b) => {
          const fl = fieldOf(b);
          const on = Math.floor(code / pow2(b)) % 2 === 1;
          return (
            <button
              key={b}
              type="button"
              aria-pressed={on}
              aria-label={`${fl === "sign" ? "sign" : fl === "exp" ? "exponent" : "mantissa"} bit ${b}`}
              onClick={() => flip(b)}
              onMouseEnter={() => setHover(fl)}
              onMouseLeave={() => setHover(null)}
              className="focus-ring h-11 w-8 rounded font-mono text-sm font-semibold sm:w-9"
              style={{
                background: on ? FIELD_COLOUR[fl] : "transparent",
                color: on ? "#000" : "inherit",
                border: `2px solid ${FIELD_COLOUR[fl]}`,
                opacity: hover && hover !== fl ? 0.55 : 1,
              }}
            >
              {on ? 1 : 0}
            </button>
          );
        })}
      </div>
      <p className="mt-2 font-mono text-xs text-neutral-600 dark:text-neutral-400">
        code 0x
        {code
          .toString(16)
          .toUpperCase()
          .padStart(Math.ceil(f.bits / 4), "0")}
        : sign{" "}
        <span style={{ borderBottom: `2px solid ${FIELD_COLOUR.sign}` }}>
          {fd.s}
        </span>
        , exponent field{" "}
        <span style={{ borderBottom: `2px solid ${FIELD_COLOUR.exp}` }}>
          {fd.E}
        </span>
        , mantissa field{" "}
        <span style={{ borderBottom: `2px solid ${FIELD_COLOUR.mant}` }}>
          {fd.M}
        </span>{" "}
        · {kindOf(value, fid)}
      </p>
      <p
        className="mt-1 break-words text-sm text-neutral-800 dark:text-neutral-200"
        data-testid="decoding"
      >
        {decoding}
      </p>

      <p className="mt-4 text-[0.7rem] font-medium uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
        Every {FORMATS[fid].name} value from 0 to {pow2Text(s.hi)}, to scale
      </p>
      <svg
        ref={font.ref}
        viewBox={`0 0 ${VW} 96`}
        className="mt-1 w-full"
        role="img"
        aria-label={`Number line from 0 to ${exact(s.hi)} with every representable value drawn as a tick`}
      >
        <line
          x1={0}
          x2={VW}
          y1={LINE_Y}
          y2={LINE_Y}
          stroke="currentColor"
          strokeOpacity={0.4}
        />
        {tk.bands.map(([a, b], i) => (
          <rect
            key={i}
            x={a}
            y={LINE_Y - 9}
            width={Math.max(0.6, b - a)}
            height={18}
            fill={FIELD_COLOUR.exp}
            fillOpacity={0.75}
          />
        ))}
        <path d={tk.path} stroke={FIELD_COLOUR.exp} strokeWidth={1.2} />
        {hi > f.max && (
          <rect
            x={(f.max / hi) * VW + 1}
            y={LINE_Y - 9}
            width={Math.max(0, VW - (f.max / hi) * VW - 1)}
            height={18}
            fill="url(#bits-over)"
          />
        )}
        <defs>
          <pattern
            id="bits-over"
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
              stroke={STATE_COLOUR.error}
              strokeWidth={1.5}
              strokeOpacity={0.6}
            />
          </pattern>
        </defs>
        <g transform={`translate(${markX.toFixed(2)} 0)`}>
          <path
            d={`M0 ${LINE_Y - 12}L-7 ${LINE_Y - 26}H7Z`}
            fill={STATE_COLOUR.active}
          />
        </g>
        <text
          x={Math.min(VW - 4, Math.max(4, markX))}
          y={font.fs(14)}
          fontSize={font.fs(12)}
          textAnchor={
            markX > VW * 0.75 ? "end" : markX < VW * 0.25 ? "start" : "middle"
          }
          fill="currentColor"
        >
          {inView
            ? `${value < 0 || Object.is(value, -0) ? "|v| = " : ""}${exact(mag)}`
            : Number.isNaN(value)
              ? "NaN: not on the line"
              : `${exact(mag)} → off this view`}
        </text>
        <text
          x={2}
          y={LINE_Y + 14 + font.fs(12)}
          fontSize={font.fs(12)}
          fill="currentColor"
        >
          0
        </text>
        <text
          x={VW - 2}
          y={LINE_Y + 14 + font.fs(12)}
          fontSize={font.fs(12)}
          textAnchor="end"
          fill="currentColor"
        >
          {pow2Text(s.hi)}
        </text>
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        A tick per value; a solid bar where they are closer than 3 px. Hatched:
        beyond the largest finite value. The triangle marks |v| for the code
        above.
      </p>
    </div>
  );

  const presets: [string, number][] = [
    ["1", encode(1, fid)],
    ["max", f.max_code],
    ["min normal", encode(f.min_normal, fid)],
    ["min subnormal", 1],
  ];

  return (
    <AnimationPanel
      testId="bits-widget"
      title="Bits to a value, and where the values are"
      summary={`Click the bits to decode one value. The animation zooms the number line towards zero, one or more powers of two per step; the values come from the format's decoder.`}
      stepper={st}
      stepLabel="zoom"
      caption={zoomCaption(fid, s, st.step === 0)}
      visual={visual}
      equation={children}
      hl={hover ?? "exp"}
      onEquationHover={(k) =>
        setHover(k === "sign" || k === "exp" || k === "mant" ? k : null)
      }
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Largest" value={exact(f.max)} />
          <Stat label="Smallest normal" value={pow2Text(f.min_normal)} />
          <Stat label="Smallest subnormal" value={pow2Text(f.min_sub)} />
          <Stat
            label="Precision"
            value={`${f.m + 1} bits`}
            hint={`ulp at 1: ${pow2Text(f.eps)}`}
          />
        </div>
      }
      params={
        <>
          <Segmented
            label="Format"
            value={fid}
            options={CHOICES.map((c) => ({ value: c, label: FORMATS[c].name }))}
            onChange={setFormat}
          />
          <div className="min-w-0 text-sm">
            <label
              htmlFor="bits-typed"
              className="text-xs font-medium uppercase tracking-widest text-neutral-500 dark:text-neutral-400"
            >
              Type a number (rounded to nearest, ties to even)
            </label>
            <input
              id="bits-typed"
              inputMode="decimal"
              value={typed}
              onChange={(e) => {
                setTyped(e.target.value);
                const v = Number(e.target.value);
                if (e.target.value.trim() !== "" && !Number.isNaN(v))
                  setSel({ fid, code: encode(v, fid) });
              }}
              className="focus-ring mt-1 h-11 w-full rounded border border-neutral-300 bg-white px-2 font-mono dark:border-neutral-700 dark:bg-neutral-950"
            />
            <div className="mt-2 flex flex-wrap gap-1">
              {presets.map(([label, c]) => (
                <button
                  key={label}
                  type="button"
                  onClick={() => setCode(c)}
                  className="focus-ring h-11 min-w-11 rounded border border-neutral-300 px-2 text-xs dark:border-neutral-700"
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </>
      }
    />
  );
}
