"use client";

/**
 * The tiny model, live (chapters 8 and 9): the vendored explainer
 * transformer runs in the browser twice, as is and with its weights (or its
 * KV cache) quantised, over a 32-character prompt (`run`, `tinySteps`).
 * One position per step: the prompt with a mark under every position
 * whose top prediction survived, the reference's five most likely next
 * characters with both models' logits, and the RMS logit change so far.
 * Below: every format's bits per number against its logit drift over all
 * four prompts.
 *
 * The model runs in small pieces, one forward pass per task (`pendingTasks`),
 * the chosen format first and then the others, so the page stays
 * responsive while it computes.
 */
import { useEffect, useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Stat } from "@/components/ui/Controls";
import { Hatch } from "@/components/interactive/Hatch";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { pct, sup, trim } from "@/lib/format";
import { tinyCaption } from "@/lib/num/captions";
import {
  KV_CONFIGS,
  KV_ORDER,
  PROMPTS,
  WEIGHT_CONFIGS,
  WEIGHT_ORDER,
  pendingTasks,
  run,
  tokenLabel,
  type KvCfg,
  type TinyTarget,
  type WeightCfg,
} from "@/lib/num/tiny";
import { METHOD_COLOUR, STATE_COLOUR } from "@/lib/viz/palette";

const VW = 640;

const SELECT =
  "focus-ring mt-1 h-11 w-full rounded border border-neutral-300 bg-white px-2 text-sm dark:border-neutral-700 dark:bg-neutral-950";

type Cfg = WeightCfg | KvCfg;

export default function TinyWidget({
  target,
  initial,
  children,
}: {
  target: TinyTarget;
  initial: string;
  children?: ReactNode;
}): JSX.Element {
  const order = (target === "weights" ? WEIGHT_ORDER : KV_ORDER) as Cfg[];
  const [cfg, setCfg] = useState<Cfg>(initial as Cfg);
  const [prompt, setPrompt] = useState(0);
  const [version, setVersion] = useState(0);

  // compute every format's run in small pieces, the chosen one first
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const queue: { cfg: Cfg; task: () => unknown }[] = [cfg, ...order].flatMap(
      (c) => pendingTasks(target, c).map((task) => ({ cfg: c, task })),
    );
    const next = () => {
      if (cancelled) return;
      const item = queue.shift();
      if (!item) return;
      item.task();
      if (pendingTasks(target, item.cfg).length === 0) setVersion((v) => v + 1);
      timer = setTimeout(next, 0);
    };
    timer = setTimeout(next, 0);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [target, cfg, order]);

  const ready = useMemo(
    () => pendingTasks(target, cfg).length === 0,
    // re-evaluated when a format finishes
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [target, cfg, version],
  );
  if (!ready)
    return (
      <p
        data-pending-widget
        className="my-8 min-h-96 text-sm text-neutral-600 dark:text-neutral-400"
      >
        Running the tiny model…
      </p>
    );
  return (
    <TinyPanel
      target={target}
      order={order}
      cfg={cfg}
      setCfg={setCfg}
      prompt={prompt}
      setPrompt={setPrompt}
      version={version}
    >
      {children}
    </TinyPanel>
  );
}

function TinyPanel({
  target,
  order,
  cfg,
  setCfg,
  prompt,
  setPrompt,
  version,
  children,
}: {
  target: TinyTarget;
  order: Cfg[];
  cfg: Cfg;
  setCfg: (c: Cfg) => void;
  prompt: number;
  setPrompt: (p: number) => void;
  version: number;
  children?: ReactNode;
}): JSX.Element {
  const configs = (
    target === "weights" ? WEIGHT_CONFIGS : KV_CONFIGS
  ) as Record<string, { label: string; bits: number }>;
  const [hover, setHover] = useState<string | null>(null);
  const r = useMemo(() => run(target, cfg), [target, cfg]);
  // the formats computed so far (all of them, a moment after the first)
  const all = useMemo(
    () =>
      order
        .filter((c) => pendingTasks(target, c).length === 0)
        .map((c) => ({
          c,
          bits: configs[c]!.bits,
          ...run(target, c).summary,
        })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [order, configs, target, version],
  );
  const steps = r.steps[prompt]!;
  const st = useStepper(steps.length, {
    stepMs: 650,
    resetKey: `${cfg}-${prompt}`,
  });
  const s = steps[st.step]!;
  const font = useSvgFont(VW);
  const fz = font.fs(11);
  const text = PROMPTS[prompt]!;
  const n = text.length;
  // one row of 32 characters, or two of 16 on a phone
  const perRow = font.narrow ? 16 : 32;
  const cw = (VW - 8) / perRow;
  const boxH = fz * 1.5;
  const markH = Math.max(6, fz * 0.5);
  const rowH = boxH + markH + 8;
  const promptH = Math.ceil(n / perRow) * rowH;

  // the five bars
  const y0 = 18 + promptH + fz + 12;
  const H = font.narrow ? 200 : 120;
  const lmax = Math.max(...s.ref.map(Math.abs), ...s.q.map(Math.abs), 1e-9);
  const bh = (v: number) => (Math.abs(v) / lmax) * (H - fz - 14);
  const slot = (VW - 16) / 5;
  const baseY = y0 + H - fz - 4;

  // drift sparkline
  const y1 = y0 + H + 18;
  const H1 = font.narrow ? 110 : 60;
  const rmax = Math.max(...steps.map((x) => x.rms), 1e-12);
  const sx = (t: number) => 8 + (t / (n - 1)) * (VW - 16);
  const sy = (v: number) => y1 + H1 - (v / rmax) * (H1 - fz - 8);

  // all formats: bits against drift (log)
  const y2 = y1 + H1 + 22;
  const H2 = font.narrow ? 280 : 140;
  const drifts = all.map((a) => Math.max(a.drift, 1e-9));
  const dlo = Math.floor(Math.log10(Math.min(...drifts)));
  const dhi = Math.ceil(Math.log10(Math.max(...drifts)));
  const bmax = Math.max(...all.map((a) => a.bits));
  const left = fz * 3 + 8;
  const px = (b: number) => left + (b / bmax) * (VW - left - 16);
  const py = (dv: number) =>
    y2 +
    H2 -
    fz -
    8 -
    ((Math.log10(dv) - dlo) / (dhi - dlo || 1)) * (H2 - 2 * fz - 22);

  const visual = (
    <svg
      ref={font.ref}
      viewBox={`0 0 ${VW} ${y2 + H2}`}
      className="w-full"
      role="img"
      aria-label={`Prompt ${prompt + 1}, position ${s.t + 1}: reference and quantised logits of the top five next characters`}
    >
      <defs>
        <Hatch id="tiny-hatch" />
      </defs>
      <text x={4} y={fz} fontSize={fz} fill="currentColor">
        {font.narrow
          ? "prompt; under it: kept (green), flipped (hatched)"
          : "prompt; below each character: top prediction unchanged (green) or changed (hatched)"}
      </text>
      {Array.from(text).map((_, t) => {
        const done = t <= st.step;
        const step = steps[t]!;
        const x = 4 + (t % perRow) * cw;
        const y = 18 + Math.floor(t / perRow) * rowH;
        return (
          <g key={t}>
            <rect
              x={x}
              y={y}
              width={cw - 1}
              height={boxH}
              fill={t === st.step ? STATE_COLOUR.active : "currentColor"}
              fillOpacity={t === st.step ? 0.9 : 0.08}
            />
            <text
              x={x + cw / 2}
              y={y + boxH * 0.75}
              fontSize={fz}
              textAnchor="middle"
              fill={t === st.step ? "#fff" : "currentColor"}
              fontFamily="monospace"
            >
              {tokenLabel(step.token)}
            </text>
            {done && (
              <rect
                x={x}
                y={y + boxH + 2}
                width={cw - 1}
                height={markH}
                fill={step.agree ? METHOD_COLOUR.kahan : "url(#tiny-hatch)"}
              >
                <title>{step.agree ? "unchanged" : "changed"}</title>
              </rect>
            )}
          </g>
        );
      })}
      <text x={4} y={y0 - 4} fontSize={fz} fill="currentColor">
        {font.narrow
          ? "top-5 logits: reference grey, quantised orange"
          : "next-character logits: reference (grey) and quantised (orange)"}
      </text>
      {s.top.map((tok, k) => {
        const x0 = 8 + k * slot;
        const bw = slot * 0.32;
        const qTop = tok === s.top_q;
        return (
          <g key={k}>
            <rect
              x={x0 + slot * 0.15}
              y={baseY - bh(s.ref[k]!)}
              width={bw}
              height={bh(s.ref[k]!)}
              fill="currentColor"
              fillOpacity={0.4}
            />
            <rect
              x={x0 + slot * 0.15 + bw}
              y={baseY - bh(s.q[k]!)}
              width={bw}
              height={bh(s.q[k]!)}
              fill={STATE_COLOUR.positive}
              stroke={qTop ? STATE_COLOUR.active : "none"}
              strokeWidth={qTop ? 3 : 0}
            />
            <text
              x={x0 + slot * 0.15 + bw}
              y={baseY + fz + 2}
              fontSize={fz}
              textAnchor="middle"
              fill="currentColor"
              fontFamily="monospace"
            >
              “{tokenLabel(tok)}”{font.narrow ? "" : ` ${trim(s.ref[k]!, 3)}`}
            </text>
          </g>
        );
      })}
      <text x={4} y={y1 + fz} fontSize={fz} fill="currentColor">
        RMS logit change at each position (top: {trim(rmax, 2)})
      </text>
      <path
        d={steps
          .slice(0, st.step + 1)
          .map((x, t) => `${t ? "L" : "M"}${sx(t)} ${sy(x.rms)}`)
          .join(" ")}
        fill="none"
        stroke={STATE_COLOUR.error}
        strokeWidth={2}
      />
      <line
        x1={8}
        x2={VW - 8}
        y1={y1 + H1}
        y2={y1 + H1}
        stroke="currentColor"
        strokeOpacity={0.3}
      />
      <text x={left} y={y2 + fz} fontSize={fz} fill="currentColor">
        {font.narrow
          ? "every format: bits → drift (log)"
          : `every ${target === "weights" ? "weight" : "KV-cache"} format: bits per number → logit drift (log)`}
      </text>
      {Array.from({ length: dhi - dlo + 1 }, (_, i) => dlo + i)
        .filter((k) => !font.narrow || (k - dlo) % 2 === 0)
        .map((k) => (
          <g key={k}>
            <line
              x1={left}
              x2={VW - 16}
              y1={py(10 ** k)}
              y2={py(10 ** k)}
              stroke="currentColor"
              strokeOpacity={0.12}
            />
            <text
              x={left - 4}
              y={py(10 ** k) + fz / 3}
              fontSize={fz}
              textAnchor="end"
              fill="currentColor"
            >
              10{sup(k)}
            </text>
          </g>
        ))}
      {all.map((a) => (
        <circle
          key={a.c}
          cx={px(a.bits)}
          cy={py(Math.max(a.drift, 1e-9))}
          r={a.c === cfg ? 8 : 5}
          fill={a.c === cfg ? STATE_COLOUR.active : "currentColor"}
          fillOpacity={a.c === cfg ? 1 : 0.4}
        >
          <title>{`${configs[a.c]!.label}: ${trim(a.bits, 3)} bits, drift ${trim(a.drift, 3)}`}</title>
        </circle>
      ))}
      {(() => {
        const cur = all.find((a) => a.c === cfg)!;
        const x = px(cur.bits);
        const right = x > VW / 2;
        return (
          <text
            x={right ? x - 12 : x + 12}
            y={py(Math.max(cur.drift, 1e-9)) - 10}
            fontSize={fz}
            textAnchor={right ? "end" : "start"}
            fill="currentColor"
            fontWeight={600}
          >
            {configs[cfg]!.label}
          </text>
        );
      })()}
      {[0, Math.round(bmax / 2), Math.round(bmax)].map((b, i) => (
        <text
          key={b}
          x={px(b)}
          y={y2 + H2 - 2}
          fontSize={fz}
          textAnchor={i === 0 ? "start" : i === 2 ? "end" : "middle"}
          fill="currentColor"
        >
          {b} bits
        </text>
      ))}
    </svg>
  );

  const label = configs[cfg]!.label;
  return (
    <AnimationPanel
      testId={`tiny-${target}-widget`}
      title={
        target === "weights"
          ? "The tiny model with quantised weights"
          : "The tiny model with a quantised KV cache"
      }
      summary={`The Transformer Decoder Explainer’s model (2 blocks, d = 16, random weights, rescaled) run in your browser twice: as is, and with ${target === "weights" ? "every block’s weights" : "its keys and values"} in ${label}.`}
      stepper={st}
      stepLabel="position"
      caption={tinyCaption(s, n, label)}
      visual={visual}
      equation={children}
      hl={hover ?? (target === "weights" ? "w" : "kv")}
      onEquationHover={setHover}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat
            label="Agreement, 4 prompts"
            value={`${r.summary.agree} of ${r.summary.n}`}
            hint={pct(r.summary.agree / r.summary.n, 1)}
          />
          <Stat
            label="Logit drift"
            value={trim(r.summary.drift, 3)}
            hint="relative RMS"
          />
          <Stat label="Largest change" value={trim(r.summary.max_change, 3)} />
          <Stat
            label="Bits per number"
            value={trim(configs[cfg]!.bits, 4)}
            hint={
              target === "kv"
                ? `${pct(configs[cfg]!.bits / 16, 0)} of FP16`
                : "16-input channel"
            }
          />
        </div>
      }
      params={
        <>
          <label className="text-xs font-medium uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
            Format
            <select
              className={SELECT}
              value={cfg}
              onChange={(e) => setCfg(e.target.value as Cfg)}
              data-testid="format"
            >
              {order.map((c) => (
                <option key={c} value={c}>
                  {configs[c]!.label}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs font-medium uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
            Prompt
            <select
              className={SELECT}
              value={prompt}
              onChange={(e) => setPrompt(Number(e.target.value))}
              data-testid="prompt"
            >
              {PROMPTS.map((p, i) => (
                <option key={i} value={i}>
                  {i + 1}: “{p}”
                </option>
              ))}
            </select>
          </label>
        </>
      }
    />
  );
}
