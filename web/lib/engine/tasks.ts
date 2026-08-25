/**
 * The reverse task, in the paper's decoder-only layout: [x_0..x_{L-1}] SEP [x_{L-1}..x_0].
 * Port of experiments/data/tasks.py. Content symbols start at 3; the LM shift drops the last token.
 */

import { N_SPECIAL, SEP } from "./model";

export interface Rng { (): number }

export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const randomSource = (L: number, nSymbols: number, rng: Rng): Int32Array => {
  const x = new Int32Array(L);
  for (let i = 0; i < L; i++) x[i] = N_SPECIAL + Math.floor(rng() * nSymbols);
  return x;
};

/** The full sequence and its LM shift. `x` is the model input, `y` the target, answer = last L of y. */
export function reverseExample(src: Int32Array) {
  const L = src.length, seq = new Int32Array(2 * L + 1);
  seq.set(src, 0);
  seq[L] = SEP;
  for (let j = 0; j < L; j++) seq[L + 1 + j] = src[L - 1 - j];
  return { seq, x: seq.slice(0, 2 * L), y: seq.slice(1), L };
}

/** Where answer token j must be fetched from, as an offset back from the position that predicts it.
 *  reverse is end-anchored: offset(j) = -(2j+1), independent of L. */
export const requiredOffsets = (L: number) =>
  Array.from({ length: L }, (_, j) => -(2 * j + 1));

export const SYMBOLS = "ABCDEFGHIJKLMNOP";
export const tokenLabel = (t: number) =>
  t === 0 ? "·" : t === 1 ? "|" : t === 2 ? "◆" : (SYMBOLS[t - N_SPECIAL] ?? "?");
