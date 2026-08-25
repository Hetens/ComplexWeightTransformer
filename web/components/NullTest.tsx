"use client";

/**
 * Panel 3: the probability test, run by the reader.
 *
 * The paper's null shuffles the position labels. That leaves the point cloud and its principal axes
 * untouched and destroys only the ordering, so it isolates exactly the thing being claimed: that the
 * angle advances *with position*. Press the button and the observed R2 is scored against 200 fresh
 * shuffles of the model you are looking at, computed here, not looked up.
 *
 * Beside it sits the weight-level test the paper reports as null at all three scales: the phases of
 * the trained complex QK weights against the uniform circle (Rayleigh, KS, Kuiper).
 */

import { useEffect, useState } from "react";
import { MODEL_IDS, ModelId, useEngineCtx } from "./EngineProvider";
import { Busy, Card, Field, Select, Stat } from "./ui";
import { Frame, linear, Plot, ticks, useMeasure } from "./viz/Chart";
import { SERIES } from "./viz/theme";
import type { ProbePayload } from "@/lib/useEngine";
import { useInView } from "@/lib/useInView";

interface PhaseEntry {
  n: number; resultant_R: number; rayleigh_p: number; ks_stat: number; ks_p: number;
  kuiper_V: number; kuiper_p: number;
}
type PhaseDoc = Record<string, Record<string, { acc: number | null; all_entries: PhaseEntry;
  top25pct_magnitude: PhaseEntry; eigen_angles: PhaseEntry }>>;

export function NullTest() {
  const engine = useEngineCtx();
  const [modelId, setModelId] = useState<ModelId>("complex-nope-d160-s0");
  const [length, setLength] = useState(24);
  // Keyed by the request it answers, so "is this stale?" is a comparison rather than a flag that has
  // to be set at the top of an effect and cleared in two callbacks.
  const [res, setRes] = useState<{ key: string; probe: ProbePayload | null; err: string | null } | null>(null);
  const [seq, setSeq] = useState(0);
  const [phase, setPhase] = useState<PhaseDoc | null>(null);
  const [scale, setScale] = useState("d160");
  const [ref, seen] = useInView<HTMLDivElement>();

  useEffect(() => { fetch("/data/phase_probe.json").then((r) => r.json()).then(setPhase); }, []);

  const key = `${modelId}:${length}:${seq}`;

  useEffect(() => {
    if (!seen) return;
    let live = true;
    engine.probe(modelId, length, 8, 200)
      .then((r) => { if (live) setRes({ key, probe: r, err: null }); })
      .catch((e: Error) => { if (live) setRes({ key, probe: null, err: e.message }); });
    return () => { live = false; };
  }, [engine, seen, modelId, length, seq, key]);

  const fresh = res?.key === key ? res : null;
  const probe = fresh?.probe ?? null;
  const err = fresh?.err ?? null;
  const busy = seen && !fresh;
  const color = SERIES[modelId].color;

  return (
    <div ref={ref} className="grid gap-5 lg:grid-cols-2">
      <Card>
        <h3 className="mb-1 text-[15px]" style={{ fontFamily: "var(--font-serif)" }}>
          Activation level: is the rotation really about position?
        </h3>
        <p className="mb-4 text-[12.5px]" style={{ color: "var(--text-secondary)" }}>
          200 shuffles of the position labels, on the model and length you pick, computed now.
        </p>
        <div className="mb-4 grid gap-3 sm:grid-cols-[minmax(0,1.4fr)_minmax(0,0.7fr)_auto]">
          <Field label="model">
            <Select value={modelId} onChange={setModelId}
              options={MODEL_IDS.map((id) => ({ value: id, label: SERIES[id].label }))} />
          </Field>
          <Field label="length">
            <Select value={String(length)} onChange={(v) => setLength(Number(v))}
              options={[16, 24, 32, 48].map((L) => ({ value: String(L), label: String(L) }))} />
          </Field>
          <div className="flex items-end">
            <button onClick={() => setSeq((s) => s + 1)} disabled={busy}
              className="rounded-md border px-3 py-1.5 text-[12.5px] transition-colors hover:bg-[var(--code-bg)] disabled:opacity-50"
              style={{ borderColor: "var(--rule)", color: "var(--text-secondary)" }}>
              re-run
            </button>
          </div>
        </div>

        <NullHistogram probe={probe} color={color} error={err} />

        <div className="mt-4 grid gap-5 border-t pt-4 sm:grid-cols-3" style={{ borderColor: "var(--rule)" }}>
          <Stat label="observed R²" value={probe ? probe.deepest.r2.toFixed(3) : "…"} color={color} />
          <Stat label="null 95th pct" value={probe ? probe.null.p95.toFixed(3) : "…"}
            hint={`mean ${probe ? probe.null.mean.toFixed(3) : "…"}`} />
          <Stat label="empirical p" value={probe ? (probe.null.p === 0 ? "< 0.005" : probe.null.p.toFixed(3)) : "…"}
            hint="shuffles at least as good" />
        </div>
      </Card>

      <Card>
        <h3 className="mb-1 text-[15px]" style={{ fontFamily: "var(--font-serif)" }}>
          Weight level: is RoPE&rsquo;s phase structure in the trained weights?
        </h3>
        <p className="mb-4 text-[12.5px]" style={{ color: "var(--text-secondary)" }}>
          The paper&rsquo;s answer is no, at three scales. These are its committed circular tests against a
          uniform phase distribution: a trained model that had learned a phase code would reject.
        </p>
        <div className="mb-3 max-w-[180px]">
          <Field label="scale">
            <Select value={scale} onChange={setScale} options={[
              { value: "d64", label: "d64, 2 layers" },
              { value: "d128", label: "d128, 4 layers" },
              { value: "d160", label: "d160, 5 layers" },
            ]} />
          </Field>
        </div>
        {!phase ? <Busy>loading committed metrics</Busy> : (
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]">
              <thead>
                <tr style={{ color: "var(--text-muted)" }}>
                  <th className="py-1.5 text-left font-normal">run</th>
                  <th className="py-1.5 text-right font-normal">resultant R</th>
                  <th className="py-1.5 text-right font-normal">Rayleigh p</th>
                  <th className="py-1.5 text-right font-normal">Kuiper p</th>
                  <th className="py-1.5 text-right font-normal">verdict</th>
                </tr>
              </thead>
              <tbody className="tnum">
                {Object.entries(phase[scale] ?? {})
                  // real-field runs appear in the same file but have no phase spectrum to test:
                  // their weights are real, so there is no angle to compare against the circle.
                  .filter(([, e]) => e.top25pct_magnitude)
                  .map(([run, e]) => {
                  const t = e.top25pct_magnitude;
                  const reject = t.rayleigh_p < 0.05 || t.kuiper_p < 0.05;
                  return (
                    <tr key={run} className="border-t" style={{ borderColor: "var(--rule)" }}>
                      <td className="py-1.5 text-left" style={{ color: "var(--text-primary)", fontFamily: "var(--font-sans)" }}>
                        {run}
                      </td>
                      <td className="py-1.5 text-right">{t.resultant_R.toFixed(4)}</td>
                      <td className="py-1.5 text-right">{t.rayleigh_p.toFixed(3)}</td>
                      <td className="py-1.5 text-right">{t.kuiper_p.toFixed(3)}</td>
                      <td className="py-1.5 text-right" style={{ color: reject ? "var(--accent)" : "var(--text-muted)" }}>
                        {reject ? "rejects" : "uniform"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="mt-3 text-[11.5px]" style={{ color: "var(--text-muted)" }}>
              Top-quartile-magnitude QK entries, the entries most able to carry a code. Every trained run
              is indistinguishable from the untrained Trabelsi initialisation, which is the first row.
            </p>
          </div>
        )}
      </Card>
    </div>
  );
}

function NullHistogram({ probe, color, error }: {
  probe: ProbePayload | null; color: string; error?: string | null;
}) {
  const [box, cw] = useMeasure<HTMLDivElement>(460);
  const W = cw, H = 200;
  if (!probe) {
    return (
      <div ref={box} className="grid place-items-center rounded-lg border px-4 text-center"
        style={{ aspectRatio: "460/200", borderColor: "var(--rule)", background: "var(--code-bg)" }}>
        {error
          ? <span className="text-[12px]" style={{ color: "var(--accent)" }}>could not run the probe: {error}</span>
          : <Busy>shuffling positions</Busy>}
      </div>
    );
  }
  const bins = 26, counts = new Array(bins).fill(0);
  for (const v of probe.null.nullR2) counts[Math.min(bins - 1, Math.max(0, Math.floor(v * bins)))]++;
  const x = linear([0, 1], [46, W - 18]);
  const y = linear([0, Math.max(...counts) * 1.12], [H - 34, 14]);
  const bw = (x(1) - x(0)) / bins;
  return (
    <div ref={box}>
    <Plot width={W} height={H} ariaLabel="Null distribution of the angle fit R-squared under shuffled position labels">
      {(setHover) => (
        <Frame width={W} height={H} x={x} y={y} xLabel="angle-fit R²  (shuffled position labels)"
          yLabel="shuffles" fmtX={(v) => v.toFixed(1)} fmtY={(v) => String(Math.round(v))}
          xTicks={ticks(0, 1, 5)}>
          {counts.map((c, i) => c > 0 && (
            <rect key={i} x={x(i / bins) + 1} y={y(c)} width={Math.max(1, bw - 2)} height={y(0) - y(c)}
              fill="var(--null)" rx={2}
              onPointerEnter={() => setHover({
                x: x(i / bins) + bw / 2, y: y(c),
                content: <>{c} of 200 shuffles<br />R² {(i / bins).toFixed(2)}–{((i + 1) / bins).toFixed(2)}</>,
              })} />
          ))}
          <line x1={x(probe.deepest.r2)} x2={x(probe.deepest.r2)} y1={14} y2={y(0)}
            stroke={color} strokeWidth={2} />
          <text x={Math.min(x(probe.deepest.r2) - 6, W - 20)} y={22} textAnchor="end" fontSize={11}
            fill={color} className="tnum">
            observed {probe.deepest.r2.toFixed(3)}
          </text>
        </Frame>
      )}
    </Plot>
    </div>
  );
}
