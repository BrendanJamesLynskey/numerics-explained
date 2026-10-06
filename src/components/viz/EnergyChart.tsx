/**
 * Chapter 10's static chart: energy and area per operation at 45 nm
 * (Horowitz, ISSCC 2014, as tabulated by Gholami et al. 2021, Figure 7),
 * as bars on a log scale. HTML, not SVG, so the labels reflow on a phone.
 */
import { HOROWITZ } from "@/lib/num/model";
import { trim } from "@/lib/format";
import { OKABE_ITO } from "@/lib/viz/palette";

const LO = Math.log10(0.01);
const HI = Math.log10(1000);
const ALO = Math.log10(10);
const AHI = Math.log10(10000);

function width(v: number, lo: number, hi: number): string {
  return `${Math.max(2, ((Math.log10(v) - lo) / (hi - lo)) * 100).toFixed(1)}%`;
}

export function EnergyChart(): JSX.Element {
  return (
    <figure
      className="my-8 rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-800 dark:bg-neutral-900"
      data-testid="energy-chart"
    >
      <figcaption className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
        Energy and area per operation, 45 nm (log scales)
      </figcaption>
      <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
        Horowitz, ISSCC 2014, as tabulated by Gholami et al. (2021), Figure 7.
        Memory reads have no area entry.
      </p>
      <div className="mt-3 grid gap-2 text-xs">
        {HOROWITZ.map((h) => (
          <div
            key={h.id}
            className="grid grid-cols-[minmax(0,9rem)_1fr] gap-2 sm:grid-cols-[12rem_1fr_1fr]"
          >
            <span className="text-neutral-700 dark:text-neutral-300">
              {h.label}
            </span>
            <div className="min-w-0">
              <div
                className="h-3 rounded-sm"
                style={{
                  width: width(h.pj, LO, HI),
                  background: OKABE_ITO.vermillion,
                }}
              />
              <span className="font-mono text-neutral-700 dark:text-neutral-300">
                {trim(h.pj, 3)} pJ
              </span>
            </div>
            <div className="col-start-2 min-w-0 sm:col-start-auto">
              {h.um2 !== null ? (
                <>
                  <div
                    className="h-3 rounded-sm"
                    style={{
                      width: width(h.um2, ALO, AHI),
                      background: OKABE_ITO.blue,
                    }}
                  />
                  <span className="font-mono text-neutral-700 dark:text-neutral-300">
                    {trim(h.um2, 4)} µm²
                  </span>
                </>
              ) : (
                <span className="text-neutral-500 dark:text-neutral-400">
                  area not given
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </figure>
  );
}
