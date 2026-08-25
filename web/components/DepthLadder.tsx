"use client";

/**
 * Panel 4: where the positional rotation lives, by depth. The committed 14-run probe, plotted.
 *
 * Every NoPE model builds the rotation in its first two blocks whatever its weight field, and the
 * embedding input reads about zero, which is the probe's own sanity check. The conditions separate
 * only at depth. Section 9 explains why that difference does not account for the behavior: within the
 * real+NoPE seeds the property anti-correlates with accuracy.
 *
 * Fourteen runs is far past the number of hues that can carry identity, so colour encodes the
 * *condition* (four classes, in a legend) and the run identity lives in the hover layer and in the
 * table below. Direct-labelling fourteen line ends would collide, which is what it did when tried.
 */

import { useEffect, useMemo, useState } from "react";
import { Busy, Card, Field, Select } from "./ui";
import { Frame, Legend, linear, Plot, path, useMeasure } from "./viz/Chart";

type Rot = Record<string, { r2: number; omega_rad_per_step: number; total_sweep_rad: number }>;
type Act = Record<string, Record<string, {
  acc: number; field: string; dim: number; rotating_plane: Rot;
  retrieval: { hit_rate: number | null; mass: number | null; layer: number | null };
}>>;

type Group = "complex" | "diag" | "real" | "rope";

function groupOf(run: string): Group {
  if (run.includes("RoPE")) return "rope";
  if (run.startsWith("C-diag")) return "diag";
  if (run.startsWith("C-NoPE")) return "complex";
  return "real";
}

const GROUP: Record<Group, { label: string; color: string; dash?: string }> = {
  complex: { label: "complex+NoPE (two-stream)", color: "var(--series-complex)" },
  diag: { label: "block-diagonal control", color: "var(--series-alt)", dash: "2 3" },
  real: { label: "real+NoPE", color: "var(--series-real)", dash: "6 3" },
  rope: { label: "RoPE (positive control)", color: "var(--series-rope)" },
};

export function DepthLadder() {
  const [act, setAct] = useState<Act | null>(null);
  const [length, setLength] = useState("24");
  const [focus, setFocus] = useState<string | null>(null);
  const [box, cw] = useMeasure<HTMLDivElement>(620);

  useEffect(() => { fetch("/data/act_probe.json").then((r) => r.json()).then(setAct); }, []);

  const runs = useMemo(() => {
    if (!act) return [];
    return Object.entries(act)
      .filter(([, byLen]) => byLen[length])
      .map(([run, byLen]) => {
        const rot = byLen[length].rotating_plane;
        const ks = Object.keys(rot).sort();
        return {
          run, group: groupOf(run), acc: byLen[length].acc,
          pts: ks.map((k, i) => ({ i, r2: rot[k].r2 })),
          deep: rot[ks[ks.length - 1]].r2,
        };
      });
  }, [act, length]);

  const W = Math.min(cw, 760), H = 340;
  const maxL = Math.max(1, ...runs.map((r) => r.pts.length - 1));
  const x = linear([0, maxL], [46, W - 18]);
  const y = linear([0, 1], [H - 34, 14]);
  const present = [...new Set(runs.map((r) => r.group))];

  return (
    <Card>
      <div className="mb-4 flex flex-wrap items-end justify-between gap-4">
        <div className="w-[150px]">
          <Field label="evaluation length">
            <Select value={length} onChange={(v) => { setLength(v); setFocus(null); }}
              options={["16", "24", "32", "48", "64"].map((L) => ({ value: L, label: L }))} />
          </Field>
        </div>
        <Legend items={present.map((g) => GROUP[g])} />
      </div>

      <div ref={box}>
        {!act ? <Busy>loading the committed probe</Busy> : (
          <Plot width={W} height={H} ariaLabel="Angle-fit R-squared by block depth, one line per probed run">
            {(setHover) => (
              <Frame width={W} height={H} x={x} y={y} xLabel="block input" yLabel="angle fit R²"
                fmtX={(v) => `L${v}`} fmtY={(v) => v.toFixed(1)}
                xTicks={Array.from({ length: maxL + 1 }, (_, i) => i)}
                pad={{ top: 14, right: 18, bottom: 34, left: 46 }}>
                {runs.map(({ run, group, pts, acc }) => {
                  const g = GROUP[group], dim = focus !== null && focus !== run;
                  return (
                    <g key={run} opacity={dim ? 0.14 : 1}>
                      <path d={path(pts.map((p) => [x(p.i), y(p.r2)]))} fill="none"
                        stroke={g.color} strokeWidth={focus === run ? 2.8 : 1.8}
                        strokeDasharray={g.dash} strokeLinejoin="round" />
                      {pts.map((p) => (
                        <circle key={p.i} cx={x(p.i)} cy={y(p.r2)} r={focus === run ? 5 : 3.4}
                          fill={g.color} stroke="var(--card)" strokeWidth={1.5}
                          onPointerEnter={() => {
                            setFocus(run);
                            setHover({
                              x: x(p.i), y: y(p.r2),
                              content: <><b>{run}</b><br />block L{p.i}: R² {p.r2.toFixed(3)}<br />
                                exact-match {acc.toFixed(2)}</>,
                            });
                          }}
                          onPointerLeave={() => setFocus(null)} />
                      ))}
                    </g>
                  );
                })}
              </Frame>
            )}
          </Plot>
        )}
      </div>

      <details className="mt-4">
        <summary className="cursor-pointer text-[12px]" style={{ color: "var(--text-secondary)" }}>
          the same numbers as a table
        </summary>
        <table className="mt-3 w-full text-[12px]">
          <thead>
            <tr style={{ color: "var(--text-muted)" }}>
              <th className="py-1 text-left font-normal">run</th>
              {Array.from({ length: maxL + 1 }, (_, i) => (
                <th key={i} className="py-1 text-right font-normal">L{i}</th>
              ))}
              <th className="py-1 text-right font-normal">exact-match</th>
            </tr>
          </thead>
          <tbody className="tnum">
            {runs.map(({ run, group, pts, acc }) => (
              <tr key={run} className="border-t" style={{ borderColor: "var(--rule)" }}>
                <td className="py-1 text-left" style={{ color: GROUP[group].color, fontFamily: "var(--font-sans)" }}>
                  {run}
                </td>
                {pts.map((p) => <td key={p.i} className="py-1 text-right">{p.r2.toFixed(3)}</td>)}
                <td className="py-1 text-right" style={{ color: "var(--text-secondary)" }}>{acc.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </Card>
  );
}
