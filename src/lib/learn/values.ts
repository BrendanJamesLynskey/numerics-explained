/**
 * The learning layer's numbers (brief 44) that are not already a path in
 * the chapters' value tree (`src/lib/num/values.ts`): each is computed here
 * from the tested model's values, at build time. A self-check cites one as
 * `<Check v="…">`, prose as `<V v="…" />`; tests/unit/learn.test.ts checks
 * every one.
 */
import { lookup } from "@/lib/num/values";

const num = (path: string): number => {
  const v = lookup(path);
  if (typeof v !== "number") throw new Error(`${path} is not a number`);
  return v;
};

const VALUES: Record<string, () => number> = {
  // chapter 5: what one scale per channel buys over one per tensor, in dB
  "ch05.channelGainDb": () =>
    num("gran.4.sym.1.sqnr") - num("gran.4.sym.0.sqnr"),
  // chapter 8: predictions GPTQ wins back at INT3 per channel
  "ch08.gptqWinsBack": () =>
    num("tiny.trained.weights.gptq3_ch.agree") -
    num("tiny.trained.weights.int3_ch.agree"),
};

export const LEARN_VALUE_NAMES = Object.keys(VALUES);

const CACHE = new Map<string, number>();

export function learnValue(name: string): number {
  const cached = CACHE.get(name);
  if (cached !== undefined) return cached;
  const f = VALUES[name];
  if (!f) throw new Error(`no learn value ${name}`);
  const v = f();
  if (!Number.isFinite(v)) throw new Error(`learn value ${name} is ${v}`);
  CACHE.set(name, v);
  return v;
}
