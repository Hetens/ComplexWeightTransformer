/// <reference lib="webworker" />
/**
 * The model runs here, off the main thread. A probe pass is a second or two of dense f32 arithmetic;
 * on the UI thread that would freeze every control on the page while it ran.
 */

import { Model } from "./model";
import { runProbe, runSingle, teacherForcedAcc } from "./runner";

type Req =
  | { id: number; type: "load"; modelId: string }
  | { id: number; type: "single"; modelId: string; src: number[] }
  | { id: number; type: "probe"; modelId: string; length: number; nSeqs: number; nNull: number }
  | { id: number; type: "acc"; modelId: string; lengths: number[]; nSeqs: number };

const cache = new Map<string, Promise<Model>>();
const get = (id: string) => {
  if (!cache.has(id)) cache.set(id, Model.load(id));
  return cache.get(id)!;
};

self.onmessage = async (e: MessageEvent<Req>) => {
  const req = e.data;
  try {
    const model = await get(req.modelId);
    switch (req.type) {
      case "load":
        return void self.postMessage({ id: req.id, ok: true, result: model.manifest });
      case "single":
        return void self.postMessage({ id: req.id, ok: true, result: runSingle(model, Int32Array.from(req.src)) });
      case "probe": {
        const r = runProbe(model, req.length, req.nSeqs, 0, req.nNull);
        // structured-clone the typed arrays as plain numbers the chart layer can map over
        return void self.postMessage({
          id: req.id, ok: true, result: {
            ...r,
            deepest: {
              ...r.deepest,
              coords: Array.from(r.deepest.coords),
              angles: Array.from(r.deepest.angles),
              fitted: Array.from(r.deepest.fitted),
            },
            null: { ...r.null, nullR2: Array.from(r.null.nullR2) },
          },
        });
      }
      case "acc": {
        const rows = req.lengths.map((L) => teacherForcedAcc(model, L, req.nSeqs, 1234));
        return void self.postMessage({ id: req.id, ok: true, result: rows });
      }
    }
  } catch (err) {
    self.postMessage({ id: req.id, ok: false, error: (err as Error).message });
  }
};
