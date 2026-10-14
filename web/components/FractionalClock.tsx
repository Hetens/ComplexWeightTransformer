"use client";

/**
 * Panel 2: Section 9, made draggable.
 *
 * Left, the dial: each dot is one token position's position-conditional mean residual stream at the
 * deepest block, projected on its own top-two principal plane and shaded light to dark with position.
 * Right, the same rotation unwrapped against position, with the fitted line whose slope is omega.
 *
 * Dragging the length slider shows the whole-sequence statistic: on a NoPE model omega shrinks like
 * 1/T and the total sweep stays near 5.4 rad, while on the RoPE control the rate is pinned. Round three
 * (figures/clock) shows that this fit blends a saturating input-side code with a roughly fractional
 * answer-side one; the input side cannot be fractional under the causal mask.
 */

import { useEffect, useMemo, useState } from "react";
import { MODEL_IDS, ModelId, useEngineCtx } from "./EngineProvider";
import { Busy, Card, Field, Select, Slider, Stat } from "./ui";
import { Frame, HoverState, linear, Plot, path, useMeasure } from "./viz/Chart";
import { positionRamp, SERIES } from "./viz/theme";
import type { ProbePayload } from "@/lib/useEngine";
import { useInView } from "@/lib/useInView";

const LENGTHS = [16, 24, 32, 48, 64];

export function FractionalClock() {
  const engine = useEngineCtx();
  const [modelId, setModelId] = useState<ModelId>("complex-nope-d160-s0");
  const [li, setLi] = useState(1);
  const [nSeqsOverride, setNSeqs] = useState<number | null>(null);
  const [byLen, setByLen] = useState<Record<string, ProbePayload>>({});
  const [ref, seen] = useInView<HTMLDivElement>();
  const length = LENGTHS[li];
  const isRope = modelId.includes("rope");
  // The RoPE control needs more sequences than the NoPE models. At about 1 rad per position its plane
  // trajectory wraps many times over the sequence, so residual content noise moves the angle much
  // further: at n=8 it reads omega 0.91 against a committed 0.99, which understates the very control
  // it is here to provide. Derived rather than pushed into state by an effect on modelId.
  const nSeqs = nSeqsOverride ?? (isRope ? 64 : 8);
  const key = `${modelId}:${length}:${nSeqs}`;

  useEffect(() => {
    if (!seen || byLen[key]) return;
    let live = true;
    engine.probe(modelId, length, nSeqs, 200)
      .then((r) => { if (live) setByLen((b) => ({ ...b, [key]: r })); })
      .catch(() => {});
    return () => { live = false; };
  }, [engine, seen, modelId, length, nSeqs, key, byLen]);

  // Derived, not stored: a `busy` flag set at the top of the effect is a second render pass that
  // says exactly what "we do not have this key yet" already says.
  const cur = byLen[key];
  const busy = seen && !cur;
  const color = SERIES[modelId].color;

  // the across-length picture, for whichever lengths have been computed so far
  const swept = useMemo(() =>
    LENGTHS.map((L) => ({ L, p: byLen[`${modelId}:${L}:${nSeqs}`] })).filter((r) => r.p),
    [byLen, modelId, nSeqs]);

  return (
    <Card innerRef={ref}>
      <div className="mb-5 grid gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)_minmax(0,0.7fr)]">
        <Field label="model">
          <Select value={modelId} onChange={setModelId}
            options={MODEL_IDS.map((id) => ({ value: id, label: SERIES[id].label }))} />
        </Field>
        <Field label={`evaluation length  ${length}   (T = ${2 * length} positions)`}>
          <Slider value={li} onChange={setLi} min={0} max={LENGTHS.length - 1}
            marks={LENGTHS.map((L, i) => ({ at: i, label: String(L) }))} />
        </Field>
        <Field label="sequences averaged"
          hint={isRope ? "the RoPE control needs more; the paper uses 512" : "the paper uses 512"}>
          <Select value={String(nSeqs)} onChange={(v) => setNSeqs(Number(v))}
            options={[["8", "8 (fast)"], ["16", "16"], ["32", "32"], ["64", "64 (steady)"]]
              .map(([value, label]) => ({ value, label }))} />
        </Field>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
        <div>
          <Caption>the plane, shaded by position</Caption>
          <Dial probe={cur} />
        </div>
        <div>
          <Caption>the same rotation, unwrapped against position</Caption>
          <Unwrapped probe={cur} color={color} />
        </div>
      </div>

      <div className="mt-6 grid gap-5 border-t pt-5 sm:grid-cols-4" style={{ borderColor: "var(--rule)" }}>
        <Stat label="rate ω" unit="rad / position" value={cur ? cur.deepest.omega.toFixed(4) : "…"}
          color={color} hint={isRope ? "held fixed by construction" : "falls as the sequence grows"} />
        <Stat label="total sweep" unit="rad" value={cur ? cur.deepest.sweep.toFixed(2) : "…"}
          color={color} hint={isRope ? "grows with length" : "invariant across length"} />
        <Stat label="fit R²" value={cur ? cur.deepest.r2.toFixed(3) : "…"}
          hint={`deepest block, ${cur?.T ?? "–"} positions`} />
        <Stat label="computed in" unit="ms" value={cur ? Math.round(cur.ms).toString() : "…"}
          hint={busy ? "running" : `${nSeqs} forward passes, in your browser`} />
      </div>

      {swept.length > 1 && (
        <div className="mt-6 border-t pt-5" style={{ borderColor: "var(--rule)" }}>
          <Caption>
            {isRope
              ? "a fixed-rate code: ω is pinned, so the sweep grows with the sequence"
              : "a fractional code: ω falls like 1/T, so the sweep does not move"}
          </Caption>
          <table className="mt-2 w-full text-[12.5px]">
            <thead>
              <tr style={{ color: "var(--text-muted)" }}>
                <th className="py-1 text-left font-normal">eval length</th>
                <th className="py-1 text-right font-normal">positions T</th>
                <th className="py-1 text-right font-normal">ω (rad/pos)</th>
                <th className="py-1 text-right font-normal">total sweep (rad)</th>
                <th className="py-1 text-right font-normal">R²</th>
              </tr>
            </thead>
            <tbody className="tnum">
              {swept.map(({ L, p }) => (
                <tr key={L} className="border-t" style={{ borderColor: "var(--rule)" }}>
                  <td className="py-1 text-left" style={{ color: L === length ? color : "var(--text-primary)" }}>
                    {L}{L === length ? "  ←" : ""}
                  </td>
                  <td className="py-1 text-right">{p!.T}</td>
                  <td className="py-1 text-right">{p!.deepest.omega.toFixed(4)}</td>
                  <td className="py-1 text-right" style={{ color }}>{p!.deepest.sweep.toFixed(2)}</td>
                  <td className="py-1 text-right" style={{ color: "var(--text-secondary)" }}>{p!.deepest.r2.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-[11.5px]" style={{ color: "var(--text-muted)" }}>
            Drag the slider across every length to fill the table. The sweep column is the one to watch.
          </p>
        </div>
      )}
    </Card>
  );
}

const Caption = ({ children }: { children: React.ReactNode }) => (
  <p className="mb-2 text-[11.5px]" style={{ color: "var(--text-secondary)" }}>{children}</p>
);

function Dial({ probe }: { probe: ProbePayload | undefined }) {
  const [box, cw] = useMeasure<HTMLDivElement>(320);
  const W = Math.min(cw, 340), H = W;   // the plane is isotropic; a stretched dial is a lie about it
  if (!probe) return <div ref={box}><Skeleton w={320} h={320} /></div>;
  const c = probe.deepest.coords;
  let m = 0;
  for (let i = 0; i < c.length; i++) m = Math.max(m, Math.abs(c[i]));
  const x = linear([-m * 1.08, m * 1.08], [12, W - 12]);
  const y = linear([-m * 1.08, m * 1.08], [H - 12, 12]);
  const T = probe.deepest.T;
  return (
    <div ref={box}>
    <Plot width={W} height={H} ariaLabel="Position-conditional residual stream on its top-two principal plane">
      {(setHover) => (
        <g>
          <line x1={x(0)} x2={x(0)} y1={12} y2={H - 12} stroke="var(--grid)" strokeWidth={0.75} />
          <line x1={12} x2={W - 12} y1={y(0)} y2={y(0)} stroke="var(--grid)" strokeWidth={0.75} />
          <path d={path(Array.from({ length: T }, (_, i) => [x(c[i * 2]), y(c[i * 2 + 1])]))}
            fill="none" stroke="var(--text-muted)" strokeWidth={1} opacity={0.35} />
          {Array.from({ length: T }, (_, i) => (
            <circle key={i} cx={x(c[i * 2])} cy={y(c[i * 2 + 1])} r={4.5}
              fill={positionRamp(i / (T - 1))} stroke="var(--card)" strokeWidth={1.5}
              onPointerEnter={() => setHover({
                x: x(c[i * 2]), y: y(c[i * 2 + 1]),
                content: <><b>position {i}</b> of {T - 1}<br />angle {probe.deepest.angles[i].toFixed(2)} rad</>,
              })} />
          ))}
          <text x={16} y={H - 16} fontSize={10.5} fill="var(--text-muted)">PC1</text>
          <text x={x(0) + 6} y={20} fontSize={10.5} fill="var(--text-muted)">PC2</text>
        </g>
      )}
    </Plot>
    </div>
  );
}

function Unwrapped({ probe, color }: { probe: ProbePayload | undefined; color: string }) {
  const [box, cw] = useMeasure<HTMLDivElement>(460);
  const [hover, setHover] = useState<HoverState | null>(null);
  const W = Math.min(cw, 620), H = 320;
  if (!probe) return <div ref={box}><Skeleton w={460} h={300} /></div>;
  const a = probe.deepest.angles, f = probe.deepest.fitted, T = probe.deepest.T;
  const lo = Math.min(...a, ...f), hi = Math.max(...a, ...f);
  const x = linear([0, T - 1], [46, W - 18]);
  const y = linear([lo - 0.3, hi + 0.3], [H - 34, 14]);
  return (
    <div ref={box}>
    <Plot width={W} height={H} hover={hover}
      ariaLabel="Unwrapped rotation angle against token position, with the fitted line"
      onProbe={(px) => {
        if (!Number.isFinite(px)) return setHover(null);
        const i = Math.round(Math.max(0, Math.min(T - 1, ((px - 46) / (W - 18 - 46)) * (T - 1))));
        setHover({ x: x(i), y: y(a[i]), content: <><b>position {i}</b><br />angle {a[i].toFixed(2)} rad<br />fit {f[i].toFixed(2)} rad</> });
      }}>
      {() => {
        return (
          <Frame width={W} height={H} x={x} y={y} xLabel="token position" yLabel="unwrapped angle (rad)"
            fmtY={(v) => v.toFixed(0)}>
            <path d={path(Array.from({ length: T }, (_, i) => [x(i), y(f[i])]))}
              fill="none" stroke="var(--text-muted)" strokeWidth={1.5} strokeDasharray="5 4" />
            <path d={path(Array.from({ length: T }, (_, i) => [x(i), y(a[i])]))}
              fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />
            {hover && <line x1={hover.x} x2={hover.x} y1={14} y2={H - 34} stroke="var(--rule)" strokeWidth={1} />}
            <g>
              <line x1={x(0)} x2={x(0)} y1={y(f[0])} y2={y(f[T - 1])} stroke={color} strokeWidth={1} opacity={0.45} />
              <text x={x(0) + 7} y={(y(f[0]) + y(f[T - 1])) / 2} fontSize={11} fill={color} className="tnum">
                sweep {probe.deepest.sweep.toFixed(2)} rad
              </text>
            </g>
            <text x={W - 20} y={y(f[T - 1]) - 8} fontSize={11} textAnchor="end" fill="var(--text-secondary)"
              className="tnum">fit: ω = {probe.deepest.omega.toFixed(4)} rad/pos</text>
          </Frame>
        );
      }}
    </Plot>
    </div>
  );
}

const Skeleton = ({ w, h }: { w: number; h: number }) => (
  <div className="grid place-items-center rounded-lg border"
    style={{ aspectRatio: `${w}/${h}`, borderColor: "var(--rule)", background: "var(--code-bg)" }}>
    <Busy>running the probe in a worker</Busy>
  </div>
);
