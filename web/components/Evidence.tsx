"use client";

/**
 * The committed evidence base, plotted: length generalization on the four tasks, the weight-tying
 * ablation, and the real-width scan. Every number here is a projection of the committed `results` metrics.
 */

import { useEffect, useState } from "react";
import { Busy, Card } from "./ui";
import { Frame, Legend, linear, Plot, path, useMeasure } from "./viz/Chart";

interface Behavior {
  series: { cond: string; dim: number; label: string; color: string; dash: string }[];
  tasks: Record<string, { cond: string; dim: number; points: { length: number; mean: number; std: number; n: number }[] }[]>;
}
interface WidthRow {
  label: string; field: string; dim: number; params: number; x_complex: number; lr: number;
  mean: number; std: number; n: number; delta_vs_complex: number; p_vs_complex: number;
}
interface AlgRow {
  algebra: string; tying: string; algebra_label: string; mean: number; std: number; n: number;
  params: number; p_vs_complex: number | null;
}

const HUE: Record<string, string> = {
  "#2a78d6": "var(--series-complex)", "#eda100": "var(--series-real)",
  "#008300": "var(--series-rope)", "#1baf7a": "var(--series-alt)", "#4a3aa7": "var(--series-violet)",
};
const DASH: Record<string, string | undefined> = { solid: undefined, dashed: "6 3", dotted: "1 3" };
const TASKS = ["reverse", "copy", "sort", "needle"] as const;

export function LengthGeneralization() {
  const [b, setB] = useState<Behavior | null>(null);
  useEffect(() => { fetch("/data/behavior.json").then((r) => r.json()).then(setB); }, []);
  if (!b) return <Card><Busy>loading committed results</Busy></Card>;

  const key = (c: string, d: number) => `${c}|${d}`;
  const meta = Object.fromEntries(b.series.map((s) => [key(s.cond, s.dim), s]));
  const present = new Set(TASKS.flatMap((t) => (b.tasks[t] ?? []).map((r) => key(r.cond, r.dim))));

  return (
    <Card>
      <div className="mb-4">
        <Legend items={b.series.filter((s) => present.has(key(s.cond, s.dim)))
          .map((s) => ({ label: s.label, color: HUE[s.color] ?? s.color, dash: DASH[s.dash] }))} />
      </div>
      <div className="grid gap-5 sm:grid-cols-2">
        {TASKS.map((task) => (
          <TaskPanel key={task} task={task} rows={b.tasks[task] ?? []} meta={meta}
            labelBand={task === "reverse"} />
        ))}
      </div>
      <p className="mt-4 text-[11.5px]" style={{ color: "var(--text-muted)" }}>
        Exact-match sequence accuracy, mean over 5 seeds, band is ±1 std. Models train on lengths 6–16;
        everything to the right of the marked boundary is extrapolation.
      </p>
    </Card>
  );
}

function TaskPanel({ task, rows, meta, labelBand }: {
  task: string; labelBand?: boolean;
  rows: { cond: string; dim: number; points: { length: number; mean: number; std: number }[] }[];
  meta: Record<string, { label: string; color: string; dash: string }>;
}) {
  const [box, cw] = useMeasure<HTMLDivElement>(330);
  const W = cw, H = 215;
  const lengths = [...new Set(rows.flatMap((r) => r.points.map((p) => p.length)))].sort((a, z) => a - z);
  if (!lengths.length) return null;
  // Ordinal spacing, not linear. The sampled lengths are 16, 24, 32, 48, 64, 96, 128: on a linear
  // axis the four lengths where anything happens occupy the leftmost fifth of the panel and the rest
  // is a flat run along zero. Equal spacing per sampled length gives each measurement equal room.
  const xi = new Map(lengths.map((L, i) => [L, i]));
  const xo = linear([0, lengths.length - 1], [42, W - 14]);
  const x = (L: number) => xo(xi.get(L) ?? 0);
  const y = linear([0, 1], [H - 32, 14]);
  const trainedAt = lengths.indexOf(16);
  return (
    <div ref={box}>
      <h4 className="mb-1 text-[13px] font-medium">{task}</h4>
      <Plot width={W} height={H} ariaLabel={`Exact-match accuracy against evaluation length on ${task}`}>
        {(setHover) => (
          <Frame width={W} height={H} x={xo} y={y} xLabel="evaluation length" yLabel="exact-match"
            fmtY={(v) => v.toFixed(1)} fmtX={(i) => String(lengths[i] ?? "")}
            xTicks={lengths.map((_, i) => i)} pad={{ top: 14, right: 14, bottom: 32, left: 42 }}>
            {trainedAt >= 0 && (
              <>
                <rect x={xo(trainedAt + 0.5)} y={14} width={Math.max(0, W - 14 - xo(trainedAt + 0.5))}
                  height={H - 46} fill="var(--grid)" opacity={0.35} />
                {labelBand && (
                  <text x={xo(trainedAt + 0.5) + 5} y={22} fontSize={9.5} fill="var(--text-muted)">
                    extrapolation
                  </text>
                )}
              </>
            )}
            {rows.map((r) => {
              const m = meta[`${r.cond}|${r.dim}`];
              if (!m) return null;
              const col = HUE[m.color] ?? m.color;
              const pts = r.points.filter((p) => lengths.includes(p.length));
              return (
                <g key={`${r.cond}${r.dim}`}>
                  <path d={`${path(pts.map((p) => [x(p.length), y(Math.min(1, p.mean + p.std))]))}L${
                    [...pts].reverse().map((p) => `${x(p.length).toFixed(2)},${y(Math.max(0, p.mean - p.std)).toFixed(2)}`).join("L")}Z`}
                    fill={col} opacity={0.1} />
                  <path d={path(pts.map((p) => [x(p.length), y(p.mean)]))} fill="none" stroke={col}
                    strokeWidth={1.9} strokeDasharray={DASH[m.dash]} strokeLinejoin="round" />
                  {pts.map((p) => (
                    <circle key={p.length} cx={x(p.length)} cy={y(p.mean)} r={3.4} fill={col}
                      stroke="var(--card)" strokeWidth={1.4}
                      onPointerEnter={() => setHover({
                        x: x(p.length), y: y(p.mean),
                        content: <><b>{m.label}</b><br />length {p.length}: {p.mean.toFixed(3)} ± {p.std.toFixed(3)}</>,
                      })} />
                  ))}
                </g>
              );
            })}
          </Frame>
        )}
      </Plot>
    </div>
  );
}

export function WidthScan() {
  const [rows, setRows] = useState<WidthRow[] | null>(null);
  const [box, cw] = useMeasure<HTMLDivElement>(560);
  useEffect(() => { fetch("/data/width_scan.json").then((r) => r.json()).then(setRows); }, []);
  if (!rows) return <Card><Busy>loading</Busy></Card>;

  const W = Math.min(cw, 620), H = 270;
  const xs = rows.map((r) => Math.log10(r.params));
  const x = linear([Math.min(...xs) - 0.06, Math.max(...xs) + 0.16], [46, W - 18]);
  const y = linear([0, 1], [H - 34, 14]);

  return (
    <Card>
      <h3 className="mb-1 text-[15px]" style={{ fontFamily: "var(--font-serif)" }}>
        Real width buys the effect, at about four times the parameters
      </h3>
      <p className="mb-4 text-[12.5px]" style={{ color: "var(--text-secondary)" }}>
        Both arms tuned over the same learning-rate grid, converged seeds only. The d320 row is the
        control that matters: an ordinary real transformer at exactly the same activation width, with
        the tying removed, costing twice as much and still scoring lower.
      </p>
      <div ref={box}>
      <Plot width={W} height={H} ariaLabel="Exact-match accuracy on reverse at length 24 against parameter count">
        {(setHover) => (
          <Frame width={W} height={H} x={x} y={y} xLabel="real parameters (log scale)" yLabel="exact-match @24"
            fmtX={(v) => `${(Math.pow(10, v) / 1e6).toFixed(1)}M`} fmtY={(v) => v.toFixed(1)}>
            {(() => {
              const real = rows.filter((r) => r.field === "real").sort((a, b) => a.params - b.params);
              return (
                <path d={path(real.map((r) => [x(Math.log10(r.params)), y(r.mean)]))} fill="none"
                  stroke="var(--series-real)" strokeWidth={1.9} strokeLinejoin="round" />
              );
            })()}
            {rows.map((r) => {
              const col = r.field === "complex" ? "var(--series-complex)" : "var(--series-real)";
              const px = x(Math.log10(r.params)), py = y(r.mean);
              return (
                <g key={r.label}>
                  <line x1={px} x2={px} y1={y(Math.max(0, r.mean - r.std))} y2={y(Math.min(1, r.mean + r.std))}
                    stroke={col} strokeWidth={1.4} opacity={0.55} />
                  <circle cx={px} cy={py} r={r.field === "complex" ? 6 : 4.6} fill={col}
                    stroke="var(--card)" strokeWidth={1.8}
                    onPointerEnter={() => setHover({
                      x: px, y: py,
                      content: <><b>{r.label}</b><br />{r.mean.toFixed(3)} ± {r.std.toFixed(3)} (n={r.n})<br />
                        {(r.params / 1e6).toFixed(2)}M params, {r.x_complex}× complex<br />
                        lr {r.lr.toExponential(0)}, p = {r.p_vs_complex}</>,
                    })} />
                  <text x={px} y={py - 12} textAnchor="middle" fontSize={10} fill="var(--text-secondary)">
                    {r.field === "complex" ? "complex d160" : `d${r.dim}`}
                  </text>
                </g>
              );
            })}
          </Frame>
        )}
      </Plot>
      </div>
      <div className="mt-3">
        <Legend items={[
          { label: "complex+NoPE d160 (the tying)", color: "var(--series-complex)", mark: "dot" as const },
          { label: "real+NoPE, width scan", color: "var(--series-real)" },
        ]} />
      </div>
    </Card>
  );
}

export function AlgebraTable() {
  const [rows, setRows] = useState<AlgRow[] | null>(null);
  useEffect(() => { fetch("/data/algebra.json").then((r) => r.json()).then(setRows); }, []);
  if (!rows) return <Card><Busy>loading</Busy></Card>;
  const max = Math.max(...rows.map((r) => r.mean + r.std));
  return (
    <Card>
      <h3 className="mb-1 text-[15px]" style={{ fontFamily: "var(--font-serif)" }}>
        The win is not the complex algebra
      </h3>
      <p className="mb-5 text-[12.5px]" style={{ color: "var(--text-secondary)" }}>
        All four tyings have identical parameter count (3,085,920) and identical activation width (320).
        Only the sharing pattern differs. A block-diagonal control with no complex arithmetic and no
        cross-stream mixing at all reproduces the result.
      </p>
      <div className="space-y-3">
        {rows.map((r) => {
          const isRef = r.algebra === "complex";
          const col = isRef ? "var(--series-complex)" : r.algebra === "diag" ? "var(--series-alt)" : "var(--null)";
          return (
            <div key={r.algebra} className="grid grid-cols-[minmax(0,132px)_minmax(0,1fr)_88px] items-center gap-3">
              <div>
                <div className="tnum text-[12px]">{r.tying}</div>
                <div className="text-[10.5px]" style={{ color: "var(--text-muted)" }}>{r.algebra_label}</div>
              </div>
              <div className="relative h-6 rounded" style={{ background: "var(--code-bg)" }}>
                <div className="absolute inset-y-0 left-0 rounded-r" style={{
                  width: `${(r.mean / max) * 100}%`, background: col,
                  borderRight: "2px solid var(--card)",
                }} />
                <div className="absolute inset-y-0" style={{
                  left: `${(Math.max(0, r.mean - r.std) / max) * 100}%`,
                  width: `${((Math.min(max, r.mean + r.std) - Math.max(0, r.mean - r.std)) / max) * 100}%`,
                  borderLeft: "1px solid var(--text-muted)", borderRight: "1px solid var(--text-muted)",
                  opacity: 0.5,
                }} />
              </div>
              <div className="tnum text-right text-[12px]">
                {r.mean.toFixed(3)}
                <span className="ml-1.5 text-[10.5px]" style={{ color: "var(--text-muted)" }}>
                  {isRef ? "ref" : `p ${r.p_vs_complex?.toFixed(2)}`}
                </span>
              </div>
            </div>
          );
        })}
      </div>
      <p className="mt-4 text-[11.5px]" style={{ color: "var(--text-muted)" }}>
        Exact-match @24 on reverse, 5 seeds each, Welch p against the complex arm.
      </p>
    </Card>
  );
}
