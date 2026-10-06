/**
 * The family's visual language (explained_sites_visual_standard.md §3):
 * Okabe and Ito's colour-blind-safe palette, the same in light and dark
 * mode. The memory-level colours are the family's (GPU Kernels Explained);
 * this site adds one colour per bit field and per method, used in every
 * picture and in the equations (globals.css, `.hl-*`). "Active" is a
 * highlight, "done" is muted, and an error or overflow is the warning hue
 * plus a hatch pattern, never colour alone.
 *
 * Okabe, M. and Ito, K. (2008), "Color Universal Design (CUD): how to make
 * figures and presentations that are friendly to colorblind people",
 * https://jfly.uni-koeln.de/color/
 */

export const OKABE_ITO = {
  black: "#000000",
  orange: "#E69F00",
  sky: "#56B4E9",
  green: "#009E73",
  yellow: "#F0E442",
  blue: "#0072B2",
  vermillion: "#D55E00",
  purple: "#CC79A7",
} as const;

/** One colour per memory level, the same on every site in the family. */
export const LEVEL_COLOUR = {
  reg: OKABE_ITO.orange,
  smem: OKABE_ITO.green,
  l2: OKABE_ITO.sky,
  hbm: OKABE_ITO.purple,
} as const;

/** The fields of a floating-point code, and an MX block's shared scale. */
export const FIELD_COLOUR = {
  sign: OKABE_ITO.purple,
  exp: OKABE_ITO.sky,
  mant: OKABE_ITO.green,
  scale: OKABE_ITO.orange,
} as const;

/** Rounding modes and summation methods. */
export const METHOD_COLOUR = {
  rne: OKABE_ITO.blue,
  sr: OKABE_ITO.orange,
  naive: OKABE_ITO.vermillion,
  naive32: OKABE_ITO.sky,
  kahan: OKABE_ITO.green,
  pairwise: OKABE_ITO.purple,
  exact: "#737373",
} as const;

/** States of an element in an animation. */
export const STATE_COLOUR = {
  active: OKABE_ITO.blue,
  error: OKABE_ITO.vermillion,
  positive: OKABE_ITO.orange,
  negative: OKABE_ITO.blue,
} as const;

/** Muted ("done", "idle") greys: Tailwind neutral-400 and neutral-600. */
export const MUTED = { light: "#a3a3a3", dark: "#525252" } as const;
