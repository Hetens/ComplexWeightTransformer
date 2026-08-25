"use client";

/**
 * Panel 1: the behavioral result, made touchable.
 *
 * Everything here is a real forward pass through the trained checkpoint. The length slider crosses
 * the training boundary at 16, which is the whole point: past it the parameter-matched real model
 * starts dropping tokens and the complex model does not.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { MODEL_IDS, ModelId, useEngineCtx } from "./EngineProvider";
import { Busy, Card, Field, Select, Slider, Stat } from "./ui";
import { SERIES } from "./viz/theme";
import { useInView } from "@/lib/useInView";
import type { SingleRun } from "@/lib/engine/runner";
import type { AccRow } from "@/lib/useEngine";

const N_SYMBOLS = 16, N_SPECIAL = 3;
/** Sequences behind the live exact-match readout. Small enough to land in about a second, which is
 *  what keeps the length slider feeling like a control rather than a form submission. The paper's
 *  own figure for the same quantity is shown beside it, at 512. */
const ACC_SEQS = 16;
const SYM = "ABCDEFGHIJKLMNOP";
const label = (t: number) => (t === 1 ? "|" : SYM[t - N_SPECIAL] ?? "?");

const randomSrc = (L: number) =>
  Array.from({ length: L }, () => N_SPECIAL + Math.floor(Math.random() * N_SYMBOLS));

/** The server and the first client render must agree, so the initial sequence cannot be random.
 *  It is randomised on mount instead, which is a client-only effect. */
const seedSrc = (L: number) => Array.from({ length: L }, (_, i) => N_SPECIAL + ((i * 7 + 3) % N_SYMBOLS));

export function LiveDemo() {
  const engine = useEngineCtx();
  const [modelId, setModelId] = useState<ModelId>("complex-nope-d160-s0");
  const [length, setLength] = useState(24);
  const [src, setSrc] = useState<number[]>(() => seedSrc(24));
  const [res, setRes] = useState<{ key: string; run: SingleRun } | null>(null);
  const [acc, setAcc] = useState<Record<string, AccRow[]>>({});
  const [ref, seen] = useInView<HTMLDivElement>();

  // The first sequence has to be deterministic so the server and client agree, then it is randomised
  // once on the client. This is the one place a mount effect legitimately sets state.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { setSrc(randomSrc(24)); }, []);
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { setSrc((s) => (s.length === length ? s : randomSrc(length))); }, [length]);

  const key = `${modelId}:${src.join(",")}`;

  useEffect(() => {
    if (!seen || src.length !== length) return;
    let live = true;
    engine.single(modelId, src).then((r) => { if (live) setRes({ key, run: r }); });
    return () => { live = false; };
  }, [engine, seen, modelId, src, length, key]);

  const run = res?.key === key ? res.run : null;

  // the paper's metric over a fresh sample, for the two headline models, computed once per length
  useEffect(() => {
    if (!seen) return;
    let live = true;
    const want: ModelId[] = ["complex-nope-d160-s0", "real-nope-d224-s0"];
    Promise.all(want.map((id) => engine.acc(id, [length], ACC_SEQS).then((r) => [id, r] as const)))
      .then((pairs) => { if (live) setAcc((a) => ({ ...a, ...Object.fromEntries(pairs) })); });
    return () => { live = false; };
  }, [engine, seen, length]);

  const answer = useMemo(() => [...src].reverse(), [src]);
  const nRight = run?.correct.filter(Boolean).length ?? 0;
  const exact = run ? nRight === run.length : false;
  const color = SERIES[modelId].color;

  const setSymbol = useCallback((i: number, v: number) => {
    setSrc((s) => s.map((t, k) => (k === i ? v : t)));
  }, []);

  return (
    <Card innerRef={ref}>
      <div className="mb-5 grid gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)_auto]">
        <Field label="model">
          <Select value={modelId} onChange={setModelId}
            options={MODEL_IDS.map((id) => ({ value: id, label: SERIES[id].label }))} />
        </Field>
        <Field label={`input length  ${length}${length <= 16 ? "  (inside training range)" : `  (${(length / 16).toFixed(2)}x past training)`}`}>
          <Slider value={length} onChange={setLength} min={6} max={48}
            marks={[{ at: 6, label: "6" }, { at: 16, label: "16 trained" }, { at: 32, label: "32" }, { at: 48, label: "48" }]} />
        </Field>
        <div className="flex items-end">
          <button onClick={() => setSrc(randomSrc(length))}
            className="rounded-md border px-3 py-1.5 text-[12.5px] transition-colors hover:bg-[var(--code-bg)]"
            style={{ borderColor: "var(--rule)", color: "var(--text-secondary)" }}>
            new sequence
          </button>
        </div>
      </div>

      <div className="space-y-3 overflow-x-auto">
        <Row title="input" hint="click a cell to change the symbol">
          {src.map((t, i) => (
            <button key={i} onClick={() => setSymbol(i, N_SPECIAL + ((t - N_SPECIAL + 1) % N_SYMBOLS))}
              className="tnum grid h-7 w-7 shrink-0 place-items-center rounded border text-[12.5px] transition-colors hover:border-[var(--accent)]"
              style={{ borderColor: "var(--rule)", background: "var(--code-bg)" }}
              aria-label={`input position ${i}, symbol ${label(t)}`}>
              {label(t)}
            </button>
          ))}
          <span className="mx-1 grid h-7 w-5 shrink-0 place-items-center text-[13px]"
            style={{ color: "var(--text-muted)" }}>|</span>
        </Row>

        <Row title="model output" hint={run ? `${nRight}/${length} tokens` : undefined}>
          {run?.predicted.map((t, i) => (
            <span key={i}
              className="tnum grid h-7 w-7 shrink-0 place-items-center rounded border text-[12.5px]"
              style={{
                borderColor: run.correct[i] ? color : "var(--rule)",
                background: run.correct[i] ? `color-mix(in oklab, ${color} 13%, transparent)` : "transparent",
                color: run.correct[i] ? "var(--text-primary)" : "var(--text-muted)",
                textDecoration: run.correct[i] ? "none" : "line-through",
              }}>
              {label(t)}
            </span>
          ))}
          {!run && <Busy>running the network</Busy>}
        </Row>

        <Row title="required" hint="the reversal of the input">
          {answer.map((t, i) => (
            <span key={i} className="tnum grid h-7 w-7 shrink-0 place-items-center text-[12.5px]"
              style={{ color: "var(--text-muted)" }}>{label(t)}</span>
          ))}
        </Row>
      </div>

      <div className="mt-6 grid gap-5 border-t pt-5 sm:grid-cols-4" style={{ borderColor: "var(--rule)" }}>
        <Stat label="this sequence" value={exact ? "exact" : `${nRight}/${length}`}
          color={exact ? color : "var(--text-primary)"}
          hint={exact ? "every answer token correct" : "not an exact match"} />
        <Stat label="complex+NoPE" unit={`@${length}`}
          value={acc["complex-nope-d160-s0"]?.[0] ? acc["complex-nope-d160-s0"][0].seqAcc.toFixed(2) : "…"}
          color="var(--series-complex)" hint={`exact-match rate, ${ACC_SEQS} fresh sequences`} />
        <Stat label="real+NoPE d224" unit={`@${length}`}
          value={acc["real-nope-d224-s0"]?.[0] ? acc["real-nope-d224-s0"][0].seqAcc.toFixed(2) : "…"}
          color="var(--series-real)" hint="parameter-matched, 3.02M vs 3.09M" />
        <Stat label="paper, 5 seeds" unit="@24"
          value={length === 24 ? "0.86 vs 0.41" : "0.86 vs 0.41"}
          hint="p = 0.0002 (Welch), n = 512 each" />
      </div>
    </Card>
  );
}

function Row({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    // The label sits above the strip on narrow screens: a fixed label column costs a third of a
    // phone's width, and the token strip is the thing worth the space.
    <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:gap-3">
      <div className="shrink-0 sm:w-[104px] sm:text-right">
        <div className="text-[11px] uppercase tracking-[0.07em]" style={{ color: "var(--text-muted)" }}>
          {title}
        </div>
        {hint && <div className="text-[10.5px]" style={{ color: "var(--text-muted)", opacity: 0.75 }}>{hint}</div>}
      </div>
      <div className="flex min-h-[28px] flex-1 items-center gap-1">{children}</div>
    </div>
  );
}
