/**
 * Number formatting for the captions, readouts and prose. Exact values of
 * small formats are printed exactly (every FP8/FP6/FP4 value has a short
 * decimal expansion); others get significant figures, switching to
 * scientific notation outside 1e-4 .. 1e6.
 */

const SUP: Record<string, string> = {
  "-": "⁻",
  "0": "⁰",
  "1": "¹",
  "2": "²",
  "3": "³",
  "4": "⁴",
  "5": "⁵",
  "6": "⁶",
  "7": "⁷",
  "8": "⁸",
  "9": "⁹",
};

/** An integer as superscript digits (with a typographic minus). */
export function sup(k: number): string {
  return String(k)
    .split("")
    .map((c) => SUP[c] ?? c)
    .join("");
}

/** A typographic minus for negative numbers. */
export function minus(s: string): string {
  return s.replace(/^-/, "−");
}

/** Up to `digits` significant figures, without trailing zeros. */
export function trim(v: number, digits = 4): string {
  if (v === 0) return "0";
  if (!Number.isFinite(v)) return Number.isNaN(v) ? "NaN" : v > 0 ? "∞" : "−∞";
  const a = Math.abs(v);
  if (a < 1e-4 || a >= 1e7) return sci(v, digits);
  const s = Number(v.toPrecision(digits));
  return minus(s.toLocaleString("en-GB", { maximumFractionDigits: 10 }));
}

/** Scientific notation: 6.104 × 10⁻⁵. */
export function sci(v: number, digits = 4): string {
  if (v === 0) return "0";
  const [m, e] = v.toExponential(digits - 1).split("e");
  const mant = String(Number(m));
  return `${minus(mant)} × 10${sup(Number(e))}`;
}

/** The shortest decimal that reads back as this double (exact for short ones). */
export function exact(v: number): string {
  if (!Number.isFinite(v)) return trim(v);
  if (Object.is(v, -0)) return "−0";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-6 || a >= 1e15)) return sci(v, 6);
  const s = String(v);
  const [i, f] = s.replace("-", "").split(".");
  const grouped = Number(i).toLocaleString("en-GB");
  return (v < 0 ? "−" : "") + grouped + (f ? `.${f}` : "");
}

/** 2^k for an exact power of two, else null. */
export function asPow2(v: number): number | null {
  if (v <= 0 || !Number.isFinite(v)) return null;
  const k = Math.round(Math.log2(v));
  return 2 ** k === v ? k : null;
}

/** "2⁻⁹" for powers of two, else exact(). */
export function pow2Text(v: number): string {
  const k = asPow2(v);
  return k === null ? exact(v) : `2${sup(k)}`;
}

export function pct(f: number, digits = 0): string {
  return `${(f * 100).toFixed(digits)}%`;
}

/** Signal-to-quantisation-noise ratio in decibels. */
export function sqnrDb(signal: number, noise: number): number {
  return noise === 0 ? Infinity : 10 * Math.log10(signal / noise);
}

export function fmtDb(db: number): string {
  return Number.isFinite(db) ? `${db.toFixed(1)} dB` : "∞ dB";
}

/** An error in ulps with a sign. */
export function fmtUlp(e: number, digits = 2): string {
  const s = e.toFixed(digits);
  return `${e > 0 ? "+" : ""}${minus(s)} ulp`;
}

/** A whole number with thousands separators. */
export function int(v: number): string {
  return Math.round(v).toLocaleString("en-GB");
}
