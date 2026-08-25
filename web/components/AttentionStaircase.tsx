"use client";

/**
 * Panel 5: the retrieval head, and the paradox.
 *
 * To emit answer token j on reverse, the model must look back 2j+1 positions. A head that solved the
 * task by direct retrieval would place its attention peak exactly there at every j: the dashed
 * staircase. The parameter-matched real model, which scores 0.41, has that head. The complex model,
 * which scores 0.86, has essentially none. Higher-scoring models here are not the ones attending
 * better, which is what put the mechanism downstream of attention.
 */

import { useEffect, useMemo, useState } from "react";
import { MODEL_IDS, ModelId, useEngineCtx } from "./EngineProvider";
import { Busy, Card, Field, Select, Stat } from "./ui";
import { Frame, Legend, linear, Plot, path, useMeasure } from "./viz/Chart";
import { SERIES } from "./viz/theme";
import type { SingleRun } from "@/lib/engine/runner";
import { useInView } from "@/lib/useInView";

const N_SPECIAL = 3;
const randomSrc = (L: number) => Array.from({ length: L }, () => N_SPECIAL + Math.floor(Math.random() * 16));

export function AttentionStaircase() {
  const engine = useEngineCtx();
  const [modelId, setModelId] = useState<ModelId>("real-nope-d224-s0");
  const [res, setRes] = useState<{ key: string; run: SingleRun } | null>(null);
  const [pick, setPick] = useState<number>(0);
  const [ref, seen] = useInView<HTMLDivElement>();

  useEffect(() => {
    if (!seen) return;
    let live = true;
    engine.single(modelId, randomSrc(16)).then((r) => { if (live) setRes({ key: modelId, run: r }); });
    return () => { live = false; };
  }, [engine, seen, modelId]);

  // Switching model invalidates the head ranking, so reset the selection from the key rather than
  // clearing it inside the effect that starts the request.
  const run = res?.key === modelId ? res.run : null;
  const sel = run ? pick : 0;

  // rank heads the way the paper does: by exact-hit rate on the required source, then by mass
  const ranked = useMemo(() => {
    if (!run) return [];
    return [...run.staircase].sort((a, b) => b.hitRate - a.hitRate || b.mass - a.mass);
  }, [run]);
  const head = ranked[sel];
  const color = SERIES[modelId].color;

  return (
    <Card innerRef={ref}>
      <div className="mb-4 grid gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Field label="model">
          <Select value={modelId} onChange={setModelId}
            options={MODEL_IDS.map((id) => ({ value: id, label: SERIES[id].label }))} />
        </Field>
        <Field label="head" hint="ranked by exact-hit rate on the required source">
          <Select value={String(sel)} onChange={(v) => setPick(Number(v))}
            options={ranked.map((h, i) => ({
              value: String(i),
              label: `L${h.layer} H${h.head} — hit ${h.hitRate.toFixed(2)}, mass ${h.mass.toFixed(2)}`,
            }))} />
        </Field>
      </div>

      {!run || !head ? <Busy>running the network</Busy> : (
        <>
          <div className="mb-2">
            <Legend items={[
              { label: "required source, −(2j+1)", color: "var(--text-muted)", dash: "5 4" },
              { label: `observed peak, L${head.layer} H${head.head}`, color },
            ]} />
          </div>
          <Staircase run={run} head={head} color={color} />
          <div className="mt-4 grid gap-5 border-t pt-4 sm:grid-cols-3" style={{ borderColor: "var(--rule)" }}>
            <Stat label="exact-hit rate" value={head.hitRate.toFixed(2)} color={color}
              hint="answer indices where the peak lands on the required source" />
            <Stat label="mass at required" value={head.mass.toFixed(2)}
              hint="share of attention on that source" />
            <Stat label="this model scores" value={modelId === "complex-nope-d160-s0" ? "0.86" : modelId === "real-nope-d224-s0" ? "0.41" : modelId === "real-nope-d160-s0" ? "0.42" : "0.00"}
              unit="@24" hint="paper, 5 seeds" />
          </div>
        </>
      )}
    </Card>
  );
}

function Staircase({ run, head, color }: {
  run: SingleRun; head: SingleRun["staircase"][number]; color: string;
}) {
  const [box, cw] = useMeasure<HTMLDivElement>(620);
  const W = Math.min(cw, 760), H = 320, L = run.length;
  const all = [...head.peaks, ...run.required];
  const x = linear([0, L - 1], [46, W - 18]);
  const y = linear([Math.min(...all) - 2, 0], [H - 34, 14]);
  return (
    <div ref={box}>
    <Plot width={W} height={H} ariaLabel="Attention peak offset against answer index, with the required retrieval staircase">
      {(setHover) => (
        <Frame width={W} height={H} x={x} y={y} xLabel="answer index j"
          yLabel="attention peak offset (positions back)" fmtY={(v) => String(v)}>
          {/* The required staircase is drawn on top of the observed peaks, not under them: when a
              head is an exact retrieval head the two coincide, and a solid observed line drawn last
              hides the very thing the panel is about. Observed peaks are markers only, so a perfect
              head reads as dots sitting on the dashed line. */}
          <path d={path(head.peaks.map((o, j) => [x(j), y(o)]))} fill="none"
            stroke={color} strokeWidth={1} opacity={0.35} strokeLinejoin="round" />
          {head.peaks.map((o, j) => (
            <circle key={j} cx={x(j)} cy={y(o)} r={4.5}
              fill={o === run.required[j] ? color : "var(--card)"} stroke={color} strokeWidth={2}
              onPointerEnter={() => setHover({
                x: x(j), y: y(o),
                content: <><b>answer index {j}</b><br />peak at {o}, required {run.required[j]}<br />
                  {o === run.required[j] ? "on the required source" : "elsewhere"}</>,
              })} />
          ))}
          <path d={path(run.required.map((o, j) => [x(j), y(o)]))} fill="none"
            stroke="var(--text-muted)" strokeWidth={1.75} strokeDasharray="5 4" />
        </Frame>
      )}
    </Plot>
    </div>
  );
}
