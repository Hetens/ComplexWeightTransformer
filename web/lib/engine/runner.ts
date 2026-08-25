/**
 * The three measurements the site makes on a live model: the paper's teacher-forced accuracy, the
 * rotating-plane probe, and the retrieval-head staircase. All read from the same forward passes.
 */

import { Model } from "./model";
import { planeFit, PlaneFit, shuffleNull, NullResult } from "./probe";
import { mulberry32, randomSource, requiredOffsets, reverseExample } from "./tasks";

export interface ProbeResult {
  length: number; T: number;
  layers: { label: string; r2: number; omega: number; sweep: number }[];
  deepest: PlaneFit;
  deepestIndex: number;
  null: NullResult;
  nSeqs: number;
  ms: number;
}

/**
 * Mean residual stream per position, over `nSeqs` random sequences, then the plane fit per block.
 *
 * Averaging over random content marginalises content out and leaves the position-dependent part.
 * The paper uses 512 sequences; the mean converges far sooner than that (at 8 sequences the deepest
 * block's omega reads 0.1151 against 0.1155 at 512), which is what makes this affordable live.
 */
export function runProbe(
  model: Model, length: number, nSeqs: number, seed = 0, nNull = 200,
): ProbeResult {
  const t0 = performance.now();
  const rng = mulberry32(seed);
  const nSym = model.cfg.n_symbols;
  let acc: Float32Array[] | null = null;
  let T = 0, W = 0;

  for (let s = 0; s < nSeqs; s++) {
    const { x } = reverseExample(randomSource(length, nSym, rng));
    const f = model.forward(x, { resid: true });
    T = f.T; W = f.residWidth;
    if (!acc) acc = f.resid.map((r) => Float32Array.from(r));
    else for (let b = 0; b < acc.length; b++) { const r = f.resid[b]; for (let i = 0; i < r.length; i++) acc[b][i] += r[i]; }
  }
  const means = acc!.map((a) => { const o = new Float32Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] / nSeqs; return o; });

  const fits = means.map((H) => planeFit(H, T, W));
  // The paper reports one entry per block *input* (L0 is the embedding, which carries no position and
  // is the probe's own sanity check). The extra trailing stream we capture is past the last block.
  const layers = fits.slice(0, model.cfg.n_layers).map((f, i) => ({
    label: `L${i}`, r2: f.r2, omega: f.omega, sweep: f.sweep,
  }));
  const deepestIndex = model.cfg.n_layers - 1;
  const deepest = fits[deepestIndex];
  return {
    length, T, layers, deepest, deepestIndex,
    null: shuffleNull(deepest, nNull, seed + 7), nSeqs, ms: performance.now() - t0,
  };
}

export interface AccResult { length: number; seqAcc: number; tokenAcc: number; n: number }

/** The paper's metric: one forward with the true answer in context, argmax at each answer position. */
export function teacherForcedAcc(model: Model, length: number, nSeqs: number, seed = 0): AccResult {
  const rng = mulberry32(seed), nSym = model.cfg.n_symbols, V = model.manifest.vocab_size;
  let seqHits = 0, tokHits = 0, tokTotal = 0;
  for (let s = 0; s < nSeqs; s++) {
    const { x, y } = reverseExample(randomSource(length, nSym, rng));
    const { logits } = model.forward(x);
    let all = true;
    for (let j = x.length - length; j < x.length; j++) {
      let best = 0, bv = -Infinity;
      for (let v = 0; v < V; v++) if (logits[j * V + v] > bv) { bv = logits[j * V + v]; best = v; }
      const hit = best === y[j];
      if (hit) tokHits++; else all = false;
      tokTotal++;
    }
    if (all) seqHits++;
  }
  return { length, seqAcc: seqHits / nSeqs, tokenAcc: tokHits / tokTotal, n: nSeqs };
}

export interface SingleRun {
  src: number[]; input: number[]; predicted: number[]; target: number[];
  correct: boolean[]; length: number;
  attn: { layer: number; head: number; grid: number[][] }[];
  staircase: { layer: number; head: number; hitRate: number; mass: number; peaks: number[] }[];
  required: number[];
}

/** One sequence, teacher-forced, with attention kept: what the interactive panel shows. */
export function runSingle(model: Model, src: Int32Array): SingleRun {
  const L = src.length, { x, y } = reverseExample(src), V = model.manifest.vocab_size;
  const f = model.forward(x, { attn: true });
  const start = x.length - L;
  const predicted: number[] = [], target: number[] = [], correct: boolean[] = [];
  for (let j = start; j < x.length; j++) {
    let best = 0, bv = -Infinity;
    for (let v = 0; v < V; v++) if (f.logits[j * V + v] > bv) { bv = f.logits[j * V + v]; best = v; }
    predicted.push(best); target.push(y[j]); correct.push(best === y[j]);
  }

  const H = model.cfg.n_heads, T = f.T, req = requiredOffsets(L);
  const attn: SingleRun["attn"] = [], staircase: SingleRun["staircase"] = [];
  f.attn!.forEach((a, layer) => {
    for (let h = 0; h < H; h++) {
      const grid: number[][] = [], peaks: number[] = [];
      let hits = 0, mass = 0;
      for (let i = 0; i < T; i++) {
        const row: number[] = [];
        for (let j = 0; j < T; j++) row.push(a[h * T * T + i * T + j]);
        grid.push(row);
      }
      for (let j = 0; j < L; j++) {
        const q = start + j;
        let peak = 0, pv = -1;
        for (let k = 0; k <= q; k++) { const w = a[h * T * T + q * T + k]; if (w > pv) { pv = w; peak = k; } }
        peaks.push(peak - q);
        const need = q + req[j];
        if (peak - q === req[j]) hits++;
        if (need >= 0) mass += a[h * T * T + q * T + need];
      }
      attn.push({ layer, head: h, grid });
      staircase.push({ layer, head: h, hitRate: hits / L, mass: mass / L, peaks });
    }
  });
  return {
    src: Array.from(src), input: Array.from(x), predicted, target, correct, length: L,
    attn, staircase, required: req,
  };
}
