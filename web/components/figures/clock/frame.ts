/**
 * Paths for the parts of the clock instrument that move, as a pure function of the state. Imported by
 * the live component (client) and by the preview script, so both draw the same thing.
 */
import * as k from "../../iso/iso-kit";
import { CLOCK_CONST as C, CLOCK_DATA } from "./art.generated";

export type ModelKey = keyof typeof CLOCK_DATA;
export type Ablation = "none" | "clock" | "control";

export interface State {
  model: ModelKey;
  length: number;        // evaluation length L (input cells and answer tokens)
  phase: "input" | "answer";
  pos: number;           // continuous token index within the current phase, 0..L-1
  ablation: Ablation;
  knob: number;          // eased knob angle (rad)
  gauge: number;         // eased accuracy reading 0..1
}

const P = C.P as unknown as k.Projection;
const TAPE_D = C.TAPE_D;
const MIN_R2 = 0.75;      // below this the answer-side fit is not a rotation (same cut as the paper figure)

/** Measured input-side clock angle at a continuous position (curve sampled at integer positions). */
export function inputAngle(model: ModelKey, m: number) {
  const curve = CLOCK_DATA[model].curve as readonly number[];
  const i = Math.max(0, Math.min(curve.length - 1, m));
  const lo = Math.floor(i), hi = Math.min(curve.length - 1, lo + 1);
  return curve[lo] + (curve[hi] - curve[lo]) * (i - lo);
}

export function answerFit(model: ModelKey, length: number) {
  const row = (CLOCK_DATA[model].answer as Record<string, { sweep: number; r2: number }>)[String(length)];
  return { ...row, coherent: row.r2 >= MIN_R2, omega: row.sweep / (length - 1) };
}

export const accuracyOf = (model: ModelKey, a: Ablation) => CLOCK_DATA[model].ablation[a];
export const knobAngle = (a: Ablation) => C.DETENTS[a === "none" ? 0 : a === "clock" ? 1 : 2];

/** Carriage x (world) for a token index on a rail of n tokens. */
export const carriageX = (j: number, n: number) => C.TRAVEL[0] + (C.TRAVEL[1] - C.TRAVEL[0]) * (n > 1 ? j / (n - 1) : 0);
/** Input-tape cells are a fixed pitch: the rail is CELLS long, so position m sits at its own cell. */
export const inputX = (m: number) => carriageX(m, C.CELLS);

function needle(cx: number, cy: number, angle: number, reach: number, z: number) {
  return k.lineOnTop([cx - Math.sin(angle) * 3.2, cy + Math.cos(angle) * 3.2],
    [cx + Math.sin(angle) * reach, cy - Math.cos(angle) * reach], z, P);
}

/** A thin annular sector on a face, from angle a0 to a1 between radii r0 < r1: a closed outline, so
 *  the printed band on a dial has no free ends. */
function bandOnTop(cx: number, cy: number, r0: number, r1: number, a0: number, a1: number, z: number) {
  if (a1 - a0 < 1e-3) return "";
  const n = Math.max(2, Math.ceil((a1 - a0) / 0.06));
  const at = (a: number, r: number): k.Plane => [cx + Math.sin(a) * r, cy - Math.cos(a) * r];
  const outer: k.Plane[] = [], inner: k.Plane[] = [];
  for (let i = 0; i <= n; i++) {
    const a = a0 + ((a1 - a0) * i) / n;
    outer.push(at(a, r1));
    inner.push(at(a, r0));
  }
  return k.onTop([...outer, ...inner.reverse()], z, P, true);
}

/** Cell boundaries on a tape: one tick per cell, rooted on the tape's near edge. */
export function cellsPath(y: number, xs: number[], major: (i: number) => boolean) {
  const minor: string[] = [], majorD: string[] = [];
  xs.forEach((x, i) => {
    const len = major(i) ? 6 : 3.2;
    (major(i) ? majorD : minor).push(k.lineOnTop([x, y + TAPE_D], [x, y + TAPE_D - len], 0.6, P));
  });
  return { minor: minor.join(""), major: majorD.join("") };
}

export function frameOf(s: State) {
  const L = s.length;
  const [ix, iy] = C.LANES.input.dial;
  const [ax, ay] = C.LANES.answer.dial;
  const reach = C.DIAL_R - 3.4;

  const mIn = s.phase === "input" ? s.pos : L - 1;
  const jAns = s.phase === "answer" ? s.pos : 0;
  const inAngle = inputAngle(s.model, mIn);
  const fit = answerFit(s.model, L);
  const ansAngle = fit.coherent ? fit.omega * jAns : 0;

  const xIn = inputX(mIn);
  const xAns = carriageX(jAns, L);
  const wire = (x: number, lane: { rail: number }) => {
    const y = lane.rail + C.RAIL_D / 2;
    return k.segment([x + 4, y, C.WIRE_Z], [C.PULLEY_X, y, C.WIRE_Z], P);
  };

  return {
    inputCarriage: k.translateAlong(P, [xIn, 0, 0]),
    answerCarriage: k.translateAlong(P, [xAns, 0, 0]),
    inputWire: wire(xIn, C.LANES.input),
    answerWire: wire(xAns, C.LANES.answer),
    inputNeedle: needle(ix, iy, inAngle, reach, C.DIAL_TOP + 0.3),
    answerNeedle: needle(ax, ay, ansAngle, reach, C.DIAL_TOP + 0.3),
    // the band on each dial: input, the angle the 16 trained cells cover; answer, the whole sweep at L
    inputBand: bandOnTop(ix, iy, C.DIAL_R - 7, C.DIAL_R - 5.6, 0, inputAngle(s.model, 15), C.DIAL_TOP + 0.1),
    answerBand: fit.coherent ? bandOnTop(ax, ay, C.DIAL_R - 7, C.DIAL_R - 5.6, 0, fit.sweep, C.DIAL_TOP + 0.1) : "",
    knobPointer: k.lineOnTop([C.KNOB[0], C.KNOB[1]],
      [C.KNOB[0] + Math.sin(s.knob) * 5.2, C.KNOB[1] - Math.cos(s.knob) * 5.2], C.CONSOLE_H + 7, P),
    gaugeNeedle: needle(C.GAUGE[0], C.GAUGE[1], C.GAUGE_FROM + C.GAUGE_SPAN * s.gauge, C.GAUGE_R - 3.4, C.CONSOLE_H + 3.3),
    inAngle, ansAngle, fit, mIn, jAns,
  };
}

/** Tape cells for a length: the input tape shows L cells at fixed pitch, the answer rail L ticks over
 *  its fixed length. Trained input cells (0..15) are major. */
export function cellsOf(L: number) {
  const inputXs = Array.from({ length: L }, (_, m) => inputX(m));
  const answerXs = Array.from({ length: L }, (_, j) => carriageX(j, L));
  return {
    input: cellsPath(C.LANES.input.tape, inputXs, (i) => i < 16),
    answer: cellsPath(C.LANES.answer.tape, answerXs, (i) => i === 0 || i === L - 1),
  };
}

export function readoutOf(s: State, f: ReturnType<typeof frameOf>) {
  const acc = accuracyOf(s.model, s.ablation).toFixed(2);
  if (s.phase === "input") {
    const share = Math.round((f.inAngle / inputAngle(s.model, 63)) * 100);
    return `input m ${Math.round(f.mIn)} · ${f.inAngle.toFixed(2)} rad (${share}%) · acc ${acc}`;
  }
  if (!f.fit.coherent) return `answer j ${Math.round(f.jAns)} · no rotation (R² ${f.fit.r2.toFixed(2)}) · acc ${acc}`;
  return `answer j ${Math.round(f.jAns)} · ${f.ansAngle.toFixed(2)} of ${f.fit.sweep.toFixed(2)} rad · acc ${acc}`;
}
