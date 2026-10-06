/**
 * The landing page's picture: every non-negative value of three small
 * formats, to scale on a line from 0 to the format's largest value. The
 * gaps double at every power of two, so the values crowd towards zero.
 * Server Component (static SVG from the tested decoder).
 */
import { exact } from "@/lib/format";
import { FORMATS, INFO, positiveValues, type FormatId } from "@/lib/num/model";
import { FIELD_COLOUR } from "@/lib/viz/palette";

const ROWS: FormatId[] = ["e2m1", "e2m3", "e4m3"];
const W = 320;

export function ValueStrips(): JSX.Element {
  return (
    <svg
      viewBox={`0 0 ${W} ${ROWS.length * 46}`}
      className="w-full"
      role="img"
      aria-label="Every non-negative value of FP4 E2M1, FP6 E2M3 and FP8 E4M3, to scale"
    >
      {ROWS.map((f, r) => {
        const vals = positiveValues(f);
        const max = INFO[f].max;
        const y = r * 46 + 30;
        return (
          <g key={f}>
            <text x={0} y={y - 14} fontSize={11} fill="currentColor">
              {FORMATS[f].name}: {vals.length} values, 0 to {exact(max)}
            </text>
            <line
              x1={0}
              x2={W}
              y1={y}
              y2={y}
              stroke="currentColor"
              strokeOpacity={0.35}
            />
            {vals.map((v) => (
              <line
                key={v}
                x1={(v / max) * (W - 2) + 1}
                x2={(v / max) * (W - 2) + 1}
                y1={y - 7}
                y2={y + 7}
                stroke={FIELD_COLOUR.exp}
                strokeWidth={1}
              />
            ))}
          </g>
        );
      })}
    </svg>
  );
}
