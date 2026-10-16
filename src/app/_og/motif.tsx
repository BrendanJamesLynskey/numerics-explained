/**
 * This site's favicon glyph (src/app/icon.svg) and its card's graphic.
 * Glyph: the hexagon of the Inference track, the letter N, #e11d48.
 */
import { INFO, positiveValues, type FormatId } from "@/lib/num/model";
import { FIELD_COLOUR } from "@/lib/viz/palette";

export function Glyph({ size }: { size: number }): JSX.Element {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32">
      <path d="M16 0.5 29.5 8.25v15.5L16 31.5 2.5 23.75V8.25z" fill="#e11d48" />
      <path
        d="M22.21 23H18.44L12.34 12.41H12.26Q12.3 13.41 12.35 14.41Q12.4 15.41 12.44 16.41V23H9.79V9H13.54L19.62 19.49H19.68Q19.66 18.51 19.62 17.55Q19.58 16.59 19.54 15.63V9H22.21Z"
        fill="#fff"
      />
    </svg>
  );
}

/**
 * The landing hero's strips: every non-negative value of FP4 E2M1, FP6 E2M3 and FP8 E4M3, each to
 * scale from 0 to its largest value; the gaps double at each power of two. From the tested decoder.
 */
const ROWS: [FormatId, string][] = [
  ["e2m1", FIELD_COLOUR.mant],
  ["e2m3", FIELD_COLOUR.exp],
  ["e4m3", FIELD_COLOUR.sign],
];

export function Motif(): JSX.Element {
  return (
    <svg width="440" height="420" viewBox="0 0 440 420">
      {ROWS.map(([f, c], r) => {
        const y = 70 + r * 130;
        const max = INFO[f].max;
        return (
          <g key={f}>
            <line
              x1={14}
              y1={y}
              x2={426}
              y2={y}
              stroke="#475569"
              strokeWidth={3}
            />
            {positiveValues(f).map((v) => (
              <line
                key={v}
                x1={14 + (412 * v) / max}
                y1={y - 44}
                x2={14 + (412 * v) / max}
                y2={y + 4}
                stroke={c}
                strokeWidth={r === 2 ? 2 : 5}
                strokeLinecap="round"
              />
            ))}
          </g>
        );
      })}
    </svg>
  );
}
