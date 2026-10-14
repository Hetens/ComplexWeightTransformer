/**
 * Builds the static art and the data for the clock instrument figure (components/figures/clock).
 *
 *     node scripts/build-clock-figure.mjs      # -> components/figures/clock/art.generated.ts
 *
 * The instrument: two reading carriages on rails, each pulling a wire that turns a dial gauge.
 *   - The input carriage reads along a tape of input cells. Its dial shows the measured clock angle at
 *     that input position, which saturates past the 16 trained cells: the input side cannot know the
 *     sequence length under the causal mask, so it cannot be fractional.
 *   - The answer carriage runs on a rail of fixed length whose tick pitch shrinks as the evaluation
 *     length grows. Its dial shows omega_L * j, whose total stays near-constant: the fractional side.
 *   - A console on the right holds a three-position selector (nothing / clock plane / control plane
 *     removed) and an accuracy gauge reading the five-seed token accuracy under that ablation.
 *
 * Every number comes from ../results/_figs/clock/clock_metrics_10k.json (experiments/clock_causal.py,
 * five checkpointed retrains per condition), averaged over seeds here.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "../components/iso/iso-kit.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const OUT = join(HERE, "..", "components", "figures", "clock", "art.generated.ts");

// --------------------------------------------------------------------------- data
const metrics = JSON.parse(readFileSync(join(ROOT, "results/_figs/clock/clock_metrics_10k.json"), "utf8"));
const CONDITIONS = {
  complex: (n) => n.includes("complex") && !n.includes("diag"),
  diag: (n) => n.includes("diag"),
  real: (n) => n.includes("d224"),
};
const LENGTHS = [16, 24, 32, 48, 64];
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const r3 = (v) => Math.round(v * 1000) / 1000;

const DATA = {};
for (const [key, match] of Object.entries(CONDITIONS)) {
  const runs = Object.entries(metrics).filter(([n]) => match(n)).map(([n, r]) => ({ n, r }));
  const curve = runs[0].r.segments.input_angle_at_max_len.map((_, m) => r3(mean(runs.map(({ r }) => r.segments.input_angle_at_max_len[m]))));
  const answer = Object.fromEntries(LENGTHS.map((L) => {
    const rows = runs.map(({ r }) => r.segments.per_length[String(L)].answer_fixed);
    return [L, { sweep: r3(mean(rows.map((x) => x.sweep_rad))), r2: r3(mean(rows.map((x) => x.r2))) }];
  }));
  const abl = (k2) => r3(mean(runs.map(({ r }) => r.ablation["16"][k2].token_acc)));
  const seq24 = r3(mean(runs.map(({ n }) => JSON.parse(readFileSync(join(ROOT, "results", n, "result.json"), "utf8")).evals["24"].seq_acc)));
  DATA[key] = { seeds: runs.length, curve, answer, ablation: { none: abl("baseline"), clock: abl("remove_clock"), control: abl("remove_control") }, seq24 };
}

// --------------------------------------------------------------------------- world
const WIDTH = 640;
const HEIGHT = 380;
const PLATE = { x: -156, y: -54, w: 312, d: 126, r: 8 };
const PLATE_H = 12;
const FOOT_H = 5;

const RAIL_X0 = -134;           // both rails and both tapes run from here ...
const RAIL_X1 = 40;             // ... to here
const TRAVEL = [RAIL_X0 + 9, RAIL_X1 - 9];   // carriage centre range
const CELLS = 64;               // the input tape is long enough for the longest evaluation length

const LANES = {
  input: { tape: 46, rail: 30, dial: [82, 30] },
  answer: { tape: -6, rail: -22, dial: [82, -22] },
};
const RAIL_D = 6;
const RAIL_H = 5;
const CAR = { w: 18, d: 16 };
const CAR_Z = RAIL_H;
const CAR_TOP = CAR_Z + 6;
const POST_TOP = 26;           // carriage post top, where the wire leaves
const WIRE_Z = 24;
const DIAL_R = 19;
const DIAL_Z = 42;
const DIAL_H = 8;
const DIAL_TOP = DIAL_Z + DIAL_H;
const PULLEY_X = LANES.input.dial[0] - DIAL_R - 2;

const CONSOLE = { x: 114, y: -42, w: 32, d: 98, r: 4 };
const CONSOLE_H = 22;
const KNOB = [130, 32];
const GAUGE = [130, -14];
const GAUGE_R = 14;

const P = k.fitProjection(
  [...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0), [LANES.input.dial[0], LANES.input.dial[1] + DIAL_R, DIAL_TOP + 2],
    [LANES.answer.dial[0], LANES.answer.dial[1] - DIAL_R, DIAL_TOP + 2]],
  WIDTH, HEIGHT, { pad: 26, azimuth: 52 },
);

const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const byDepth = (pts) => [...pts].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));

// --------------------------------------------------------------------------- parts
function plate() {
  const out = [`<path class="iso-halo" d="${k.haloOf(PLATE, -PLATE_H - FOOT_H, PLATE_H, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(PLATE, 22))) {
    out.push(S(k.cylinder(x, y, 7, -PLATE_H - FOOT_H, 1.6, P, 28), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 5.4, -PLATE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: "mid" }));
  }
  out.push(S(k.slabOf(PLATE, -PLATE_H, PLATE_H, P, 8, 1.6), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(PLATE, 5), 0, P), { tone: "lo" }));
  const screws = k.corners(PLATE, 10);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.2, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
  // label plate, front left, three rule lines standing for printed text
  const label = { x: -138, y: 58, w: 46, d: 8, r: 1.2 };
  out.push(S(k.slabOf(label, 0, 0.8, P, 3), { tone: "lo", crease: "none" }));
  out.push(L(k.planOutline(k.insetPlan(label, 1.4), 0.8, P, 3), { tone: "faint" }));
  [[4, 30], [4, 22], [4, 36]].forEach(([dx, len], i) =>
    out.push(L(k.lineOnTop([label.x + dx, label.y + 2.4 + i * 1.6], [label.x + dx + len, label.y + 2.4 + i * 1.6], 0.8, P), { tone: "lo", free: true })));
  out.push(D([at([label.x + 2, label.y + 4, 0.8]), at([label.x + label.w - 2, label.y + 4, 0.8])], { size: 0.4 }));
  return out.join("");
}

/** A tape strip on the plate: a thin paper slab with cell boundaries hanging from its near edge. */
function tape(y, name) {
  const plan = { x: RAIL_X0, y, w: RAIL_X1 - RAIL_X0, d: 9, r: 0.8 };
  const out = [S(k.slabOf(plan, 0, 0.6, P, 3), { tone: "hi", crease: "none", flat: false })];
  // end rollers, so the strip is held rather than floating
  for (const x of [RAIL_X0 - 3, RAIL_X1 + 3]) {
    out.push(S(k.slabOf({ x: x - 2, y: y - 1.5, w: 4, d: 12, r: 1.6 }, 0, 3, P, 3, 0.3), { tone: "mid" }));
    out.push(D([at([x, y + 1, 3]), at([x, y + 8, 3])], { size: 0.4 }));
  }
  out.push(`<g data-part="${name}-cells"></g>`);
  return out.join("");
}

function rail(y) {
  const out = [];
  out.push(S(k.slabOf({ x: RAIL_X0, y, w: RAIL_X1 - RAIL_X0, d: RAIL_D, r: 1 }, 0, RAIL_H, P, 3, 0.4), { tone: "mid" }));
  out.push(L(k.lineOnTop([RAIL_X0, y + RAIL_D / 2], [RAIL_X1, y + RAIL_D / 2], RAIL_H, P), { tone: "faint" }));
  for (const x of [RAIL_X0 - 6, RAIL_X1]) {
    out.push(S(k.slabOf({ x, y: y - 3, w: 6, d: RAIL_D + 6, r: 1.2 }, 0, RAIL_H + 4, P, 3, 0.5), { tone: "mid" }));
    out.push(D([at([x + 3, y - 0.5, RAIL_H + 4]), at([x + 3, y + RAIL_D + 0.5, RAIL_H + 4])], { size: 0.45 }));
  }
  return out.join("");
}

/** A carriage built at x = 0 on the rail at y; moved with translateAlong. Reads the tape at tapeY. */
function carriage(y, tapeY) {
  const cy = y + RAIL_D / 2;
  const plan = { x: -CAR.w / 2, y: cy - CAR.d / 2, w: CAR.w, d: CAR.d, r: 2 };
  const out = [`<path class="iso-halo" d="${k.haloOf(k.insetPlan(plan, 2), CAR_Z, 0.1, P)}" style="filter:blur(3px)"/>`];
  // bushings under the plate
  for (const dx of [-6, 6]) out.push(S(k.slabOf({ x: dx - 2.5, y: y - 1, w: 5, d: RAIL_D + 2, r: 1 }, CAR_Z - 1, 1.6, P, 2), { tone: "lo" }));
  out.push(S(k.slabOf(plan, CAR_Z, CAR_TOP - CAR_Z, P, 4, 0.6), { tone: "hi", crease: "lo", lit: true }));
  out.push(D([-6, 6].flatMap((dx) => [at([dx, plan.y + 2, CAR_TOP]), at([dx, plan.y + plan.d - 2, CAR_TOP])]), { size: 0.45, tone: "lo" }));
  // reading arm out over the tape, ending in a pointer pad
  const reach = tapeY + 4.5 - (plan.y + plan.d);
  out.push(S(k.slabOf({ x: -2, y: plan.y + plan.d, w: 4, d: reach, r: 0.8 }, CAR_TOP - 2, 2, P, 2), { tone: "hi", crease: "lo", lit: true }));
  out.push(S(k.cylinder(0, tapeY + 4.5, 1.6, 1.4, CAR_TOP - 2 - 1.4, P, 16), { tone: "hi", crease: "none", lit: true }));
  // wire post with a collar; the wire leaves its top toward the dial pulley
  out.push(S(k.cylinder(4, cy, 3, CAR_TOP, 2, P, 20), { tone: "mid" }));
  out.push(S(k.cylinder(4, cy, 1.6, CAR_TOP + 2, POST_TOP - CAR_TOP - 2, P, 16), { tone: "mid", crease: "none" }));
  out.push(S(k.cylinder(4, cy, 2.4, WIRE_Z - 1, 2, P, 16), { tone: "mid" }));
  // knurled thumb knob on the near-left corner
  out.push(S(k.cylinder(-5, cy + 2, 2.6, CAR_TOP, 3, P, 18), { tone: "mid" }));
  out.push(L(k.knurl(-5, cy + 2, 2.6, CAR_TOP, CAR_TOP + 3, 16, P), { tone: "lo" }));
  return out.join("");
}

/** Dial gauge on a post at (x, y), face up; pulley at wire height on the post's left. */
function dialPost(x, y) {
  const out = [];
  out.push(S(k.cylinder(x, y, 6, 0, 3, P, 28), { tone: "lo" }));
  out.push(D(byDepth([[x - 4, y], [x + 4, y], [x, y - 4], [x, y + 4]]).map(([px, py]) => at([px, py, 3])), { size: 0.4 }));
  out.push(S(k.cylinder(x, y, 3, 3, DIAL_Z - 3, P, 20), { tone: "mid", crease: "none" }));
  out.push(L([12, 30].map((z) => k.sideArc(x, y, 3, z, P, 12)).join(""), { tone: "faint" }));
  // pulley bracket: an arm from the post to the pulley axle at wire height
  out.push(S(k.slabOf({ x: PULLEY_X, y: y - 1.5, w: x - PULLEY_X, d: 3, r: 0.8 }, WIRE_Z - 4, 2, P, 2), { tone: "mid" }));
  out.push(S(k.cylinder(PULLEY_X, y + 1.5, 3.4, WIRE_Z - 2, 4, P, 24), { tone: "mid" }));
  out.push(L(k.sideArc(PULLEY_X, y + 1.5, 3.4, WIRE_Z, P, 16), { tone: "lo" }));
  return out.join("");
}

function dialHead(x, y, name) {
  const out = [];
  out.push(S(k.cylinder(x, y, DIAL_R, DIAL_Z, DIAL_H, P, 64, 0.8), { tone: "hi", crease: "lo" }));
  out.push(L(k.knurl(x, y, DIAL_R, DIAL_TOP - 3, DIAL_TOP, 80, P, 0.3), { tone: "lo" }));
  out.push(L(k.sideArc(x, y, DIAL_R, DIAL_TOP - 3, P, 40), { tone: "faint" }));
  out.push(L(k.ring(x, y, DIAL_R - 2, DIAL_TOP, P, 64), { tone: "lo" }));
  // one full turn is 2*pi rad of clock angle: 64 minor ticks, a major every 8 (pi/4)
  const t = k.radialTicks(x, y, DIAL_R - 2.6, 64, 8, DIAL_TOP, [1.1, 2.4], P);
  out.push(L(t.minor, { tone: "lo" }), L(t.major, { tone: "mid" }));
  out.push(`<path data-part="${name}-band" class="iso-line" data-tone="hi" d="" style="stroke-width:.8;fill:var(--anatomy-hi);fill-opacity:.85"/>`);
  // crown and stem on the far side
  out.push(S(k.slabOf({ x: x - 2.2, y: y - DIAL_R - 4, w: 4.4, d: 5, r: 1.2 }, DIAL_Z + 2, 4, P, 3), { tone: "mid" }));
  out.push(S(k.slabOf({ x: x - 3, y: y - DIAL_R - 6.4, w: 6, d: 2.6, r: 1.2 }, DIAL_Z + 1.4, 5.2, P, 3), { tone: "mid" }));
  out.push(`<path data-part="${name}-needle" class="iso-line" data-tone="lit" data-free="start" d="" style="stroke-width:1.1"/>`);
  out.push(L(k.ring(x, y, 1.5, DIAL_TOP + 0.2, P, 16), { tone: "mid" }));
  out.push(D([at([x, y, DIAL_TOP + 0.2])], { size: 0.55, tone: "hi" }));
  return out.join("");
}

function consoleBox() {
  const out = [`<path class="iso-halo" d="${k.haloOf(CONSOLE, 0, CONSOLE_H, P)}" style="opacity:.5"/>`];
  out.push(S(k.slabOf(CONSOLE, 0, CONSOLE_H, P, 6, 1.0), { tone: "mid", inner: L(k.sideSeam(CONSOLE, 6, P, 6), { tone: "faint" }) }));
  out.push(L(k.planOutline(k.insetPlan(CONSOLE, 3), CONSOLE_H, P, 6), { tone: "faint" }));
  out.push(D(k.corners(CONSOLE, 6).map(([x, y]) => at([x, y, CONSOLE_H])), { size: 0.5 }));
  // selector: three detents (none, clock, control) on an arc, knurled knob with a pointer
  const [kx, ky] = KNOB;
  out.push(L(k.ring(kx, ky, 9.5, CONSOLE_H, P, 40), { tone: "faint" }));
  out.push(D(DETENTS.map((a) => at([kx + Math.sin(a) * 11, ky - Math.cos(a) * 11, CONSOLE_H])), { size: 0.7, tone: "mid" }));
  out.push(S(k.cylinder(kx, ky, 7, CONSOLE_H, 2, P, 32), { tone: "lo" }));
  out.push(S(k.cylinder(kx, ky, 5.6, CONSOLE_H + 2, 5, P, 32), { tone: "mid" }));
  out.push(L(k.knurl(kx, ky, 5.6, CONSOLE_H + 2, CONSOLE_H + 7, 32, P), { tone: "lo" }));
  out.push(`<path data-part="knob-pointer" class="iso-line" data-tone="hi" data-free="start" d="" style="stroke-width:1.2"/>`);
  // closed vent slots on the top, back end, clear of the gauge rim
  for (let i = 0; i < 5; i++) {
    const slot = { x: CONSOLE.x + 6 + i * 4.4, y: CONSOLE.y + 3.5, w: 2, d: 5, r: 1 };
    out.push(L(k.planOutline(slot, CONSOLE_H, P, 3), { tone: "lo" }));
  }
  // accuracy gauge set into the console top: a 270-degree scale, 0 to 1
  const [gx, gy] = GAUGE;
  out.push(S(k.cylinder(gx, gy, GAUGE_R, CONSOLE_H, 3, P, 56, 0.6), { tone: "hi", crease: "lo" }));
  out.push(L(k.ring(gx, gy, GAUGE_R - 1.6, CONSOLE_H + 3, P, 56), { tone: "lo" }));
  const t = k.radialTicks(gx, gy, GAUGE_R - 2.2, 20, 5, CONSOLE_H + 3, [1.1, 2.4], P, GAUGE_FROM, GAUGE_SPAN);
  out.push(L(t.minor, { tone: "lo" }), L(t.major, { tone: "mid" }));
  out.push(`<path data-part="gauge-needle" class="iso-line" data-tone="lit" data-free="start" d="" style="stroke-width:1.1"/>`);
  out.push(D([at([gx, gy, CONSOLE_H + 3.2])], { size: 0.55, tone: "hi" }));
  out.push(k.faceTextSvg(k.topMatrix([gx, gy + 7], CONSOLE_H + 3, P, "x"), "acc", { size: 2.2, tone: "lo", anchor: "middle" }));
  // pilot lamps between gauge and knob, each a bezel and a lens
  for (const y of [8, 14]) {
    out.push(S(k.cylinder(CONSOLE.x + 10, y, 2.4, CONSOLE_H, 1.2, P, 20), { tone: "lo" }));
    out.push(S(k.cylinder(CONSOLE.x + 10, y, 1.5, CONSOLE_H + 1.2, 1, P, 16), { tone: "hi", crease: "lo" }));
  }
  // front-face label plate: a thin slab proud of the near (+y) face
  const fy = CONSOLE.y + CONSOLE.d;
  out.push(S(k.slabOf({ x: CONSOLE.x + 6, y: fy, w: 20, d: 0.8, r: 0.3 }, 8, 9, P, 2), { tone: "lo", crease: "none" }));
  return out.join("");
}

const DETENTS = [-0.9, 0, 0.9];             // none, clock, control (rad around the knob)
const GAUGE_FROM = -Math.PI * 0.75;
const GAUGE_SPAN = Math.PI * 1.5;

// --------------------------------------------------------------------------- assemble, back to front
const carX = TRAVEL[0];
const svgBody = [
  plate(),
  tape(LANES.answer.tape, "answer"),
  rail(LANES.answer.rail),
  `<path data-part="answer-wire" class="iso-line" data-tone="mid" d=""/>`,
  `<g data-part="answer-carriage" transform="${k.translateAlong(P, [carX, 0, 0])}">${carriage(LANES.answer.rail, LANES.answer.tape)}</g>`,
  dialPost(...LANES.answer.dial),
  dialHead(...LANES.answer.dial, "answer"),
  tape(LANES.input.tape, "input"),
  rail(LANES.input.rail),
  `<path data-part="input-wire" class="iso-line" data-tone="mid" d=""/>`,
  `<g data-part="input-carriage" transform="${k.translateAlong(P, [carX, 0, 0])}">${carriage(LANES.input.rail, LANES.input.tape)}</g>`,
  dialPost(...LANES.input.dial),
  dialHead(...LANES.input.dial, "input"),
  consoleBox(),
].join("");

const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label:
    "A bench instrument with two rails. The front carriage reads along a tape of input cells and pulls a wire that turns the input dial; the back carriage runs on a fixed-length rail whose ticks get finer for longer sequences and turns the answer dial. A console on the right has a three-position selector and an accuracy gauge.",
  body: svgBody,
});

const CONST = {
  P, WIDTH, HEIGHT, LENGTHS, CELLS, RAIL_X0, RAIL_X1, TRAVEL, LANES, RAIL_D, POST_TOP, WIRE_Z, DIAL_R, DIAL_TOP,
  PULLEY_X, KNOB, GAUGE, GAUGE_R, CONSOLE_H, DETENTS, GAUGE_FROM, GAUGE_SPAN, TAPE_D: 9,
};

writeFileSync(OUT, `// Generated by web/scripts/build-clock-figure.mjs from results/_figs/clock/clock_metrics_10k.json.
// Do not edit by hand: rerun the script.
export const CLOCK_SVG = ${JSON.stringify(svg)};
export const CLOCK_DATA = ${JSON.stringify(DATA)} as const;
export const CLOCK_CONST = ${JSON.stringify(CONST)} as const;
`);
console.log("wrote", OUT, `${(svg.length / 1024).toFixed(0)} KB svg`);
for (const [key, d] of Object.entries(DATA)) console.log(key, d.seeds, "seeds", "seq24", d.seq24, "ablation", JSON.stringify(d.ablation), "curve[15]/[63]", d.curve[15], d.curve[63]);
