"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";
import type { Manifest } from "./engine/model";
import type { SingleRun } from "./engine/runner";

export interface ProbePayload {
  length: number; T: number; nSeqs: number; ms: number; deepestIndex: number;
  layers: { label: string; r2: number; omega: number; sweep: number }[];
  deepest: { r2: number; omega: number; sweep: number; T: number; top2VarShare: number;
    radiusCv: number; coords: number[]; angles: number[]; fitted: number[] };
  null: { nullR2: number[]; p: number; p95: number; mean: number };
}
export interface AccRow { length: number; seqAcc: number; tokenAcc: number; n: number }

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void };

interface Lane { worker: Worker; queued: number; models: Set<string> }

/**
 * A small pool of model workers, scheduled with model affinity.
 *
 * One worker is not enough: the panels on this page each want a forward pass as they scroll into
 * view, and a probe pass is a couple of seconds, so a single lane makes the last panel wait for
 * every panel above it. Affinity matters because a lane that has already decoded a model can start
 * immediately, while a cold lane pays for the fetch and the fp16 decode again.
 */
const LANES = 3;

export function useEngine() {
  const lanes = useRef<Lane[]>([]);
  const pending = useRef(new Map<number, Pending>());
  const next = useRef(1);

  // Lanes are built on first use rather than in a mount effect. Workers only exist on the client and
  // every caller is already inside an effect or an event handler, so there is nothing to wait for,
  // and a `ready` flag would only exist to be flipped by a setState during mount.
  const ensure = useCallback((): Lane[] => {
    if (lanes.current.length) return lanes.current;
    const made: Lane[] = [];
    for (let i = 0; i < LANES; i++) {
      const w = new Worker(new URL("./engine/worker.ts", import.meta.url));
      const lane: Lane = { worker: w, queued: 0, models: new Set() };
      w.onmessage = (e: MessageEvent<{ id: number; ok: boolean; result?: unknown; error?: string }>) => {
        lane.queued = Math.max(0, lane.queued - 1);
        const p = pending.current.get(e.data.id);
        if (!p) return;
        pending.current.delete(e.data.id);
        if (e.data.ok) p.resolve(e.data.result);
        else p.reject(new Error(e.data.error ?? "worker error"));
      };
      made.push(lane);
    }
    lanes.current = made;
    return made;
  }, []);

  useEffect(() => () => {
    lanes.current.forEach((l) => l.worker.terminate());
    lanes.current = [];
  }, []);

  const pick = (ls: Lane[], modelId: string): Lane => {
    const idlest = ls.reduce((a, b) => (b.queued < a.queued ? b : a));
    // Affinity is a preference, not a rule. Preferring a warm lane unconditionally piles every
    // request for the headline model onto one lane while the others sit idle, which is slower than
    // paying the decode again: three panels default to the same model.
    const warm = ls.filter((l) => l.models.has(modelId)).sort((a, b) => a.queued - b.queued)[0];
    return warm && warm.queued <= idlest.queued + 1 ? warm : idlest;
  };

  const call = useCallback(<T,>(modelId: string, msg: Record<string, unknown>): Promise<T> => {
    const lane = pick(ensure(), modelId);
    lane.queued++;
    lane.models.add(modelId);
    const id = next.current++;
    return new Promise<T>((resolve, reject) => {
      pending.current.set(id, { resolve: resolve as (v: unknown) => void, reject });
      lane.worker.postMessage({ ...msg, modelId, id });
    });
  }, [ensure]);

  // Stable identity. Panels list the engine in their effect dependencies, so returning a fresh object
  // literal each render restarts every in-flight forward pass on every state update, and a panel that
  // sets state when its own probe resolves never settles.
  return useMemo(() => ({
    load: (modelId: string) => call<Manifest>(modelId, { type: "load" }),
    single: (modelId: string, src: number[]) => call<SingleRun>(modelId, { type: "single", src }),
    probe: (modelId: string, length: number, nSeqs: number, nNull = 200) =>
      call<ProbePayload>(modelId, { type: "probe", length, nSeqs, nNull }),
    acc: (modelId: string, lengths: number[], nSeqs: number) =>
      call<AccRow[]>(modelId, { type: "acc", lengths, nSeqs }),
  }), [call]);
}
