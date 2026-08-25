/**
 * Numerical gate on the browser engine: the TS forward pass and probe must reproduce the PyTorch
 * checkpoint and the paper's committed metrics. Run with `npm run verify` in web/.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Model, Manifest } from "../lib/engine/model";
import { runProbe, teacherForcedAcc } from "../lib/engine/runner";

/**
 * `--quick` runs the subset that catches a broken port: teacher-forced accuracy, the full depth
 * profile, and the RoPE control at the shortest length. The lengths it drops (48 and 64) and the
 * end-to-end ratio checks cost minutes, because a probe pass is quadratic in sequence length. The
 * build gate uses --quick; the full run is `npm run verify`.
 */
const QUICK = process.argv.includes("--quick");
const MODELS = join(import.meta.dirname, "..", "public", "models");
const DATA = join(import.meta.dirname, "..", "public", "data");

function load(id: string): Model {
  const manifest = JSON.parse(readFileSync(join(MODELS, id, "manifest.json"), "utf8")) as Manifest;
  const buf = readFileSync(join(MODELS, id, "weights.f16"));
  return new Model(manifest, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer);
}

let failures = 0;
const check = (name: string, got: number, want: number, tol: number) => {
  const ok = Math.abs(got - want) <= tol;
  if (!ok) failures++;
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name.padEnd(46)} got ${got.toFixed(4)}  want ${want.toFixed(4)} ± ${tol}`);
};

// ---- 1. teacher-forced accuracy vs the paper's own evals -------------------
console.log("\nteacher-forced seq_acc vs results/*/result.json (n=64, binomial se ~0.06)");
for (const [id, lengths] of [
  ["complex-nope-d160-s0", [16, 24]],
  ["real-nope-d224-s0", [16, 24]],
  ["real-rope-d160-s0", [16, 24]],
] as [string, number[]][]) {
  const m = load(id);
  for (const L of lengths) {
    const got = teacherForcedAcc(m, L, 64, 1234).seqAcc;
    check(`${id} @${L}`, got, m.manifest.paper_seq_acc[String(L)].seq_acc, 0.15);
  }
}

// ---- 2. the rotating-plane probe vs the committed activation metrics -------
console.log("\nrotating plane vs results/_figs/act/activation_metrics.json (complex+NoPE d160 s0)");
const act = JSON.parse(readFileSync(join(DATA, "act_probe.json"), "utf8"));
const m = load("complex-nope-d160-s0");
for (const L of (QUICK ? [16, 24] : [16, 24, 32, 48, 64])) {
  const ref = act["C-NoPE s0"][String(L)].rotating_plane;
  const got = runProbe(m, L, 16, 0, 50);
  const deep = got.layers[got.deepestIndex];
  check(`L=${L} deepest r2`, deep.r2, ref[`L${got.deepestIndex}`].r2, 0.02);
  check(`L=${L} deepest omega`, Math.abs(deep.omega), Math.abs(ref[`L${got.deepestIndex}`].omega_rad_per_step), 0.004);
  check(`L=${L} deepest sweep`, deep.sweep, ref[`L${got.deepestIndex}`].total_sweep_rad, 0.15);
}

// ---- 3. the embedding sanity check and the depth profile ------------------
console.log("\ndepth profile at L=24 (every block, complex+NoPE d160 s0)");
const p24 = runProbe(m, 24, 16, 0, 50);
p24.layers.forEach((l, i) => {
  check(`L${i} r2`, l.r2, act["C-NoPE s0"]["24"].rotating_plane[`L${i}`].r2, 0.02);
});

// ---- 4. the positive control: RoPE must hold its rate ---------------------
// The RoPE arm needs more sequences than the NoPE arms: at ~1 rad/position its plane trajectory wraps
// many times over the sequence, so residual content noise moves the angle much further. n=16 reads
// omega 0.905 at L=24 against the committed 0.991; n=64 reads 0.990.
console.log("\npositive control: real+RoPE d160 holds omega while the sweep grows");
const rope = load("real-rope-d160-s0");
for (const L of (QUICK ? [16] : [16, 24])) {
  const ref = act["R-RoPE d160 s0 (positive control)"][String(L)].rotating_plane;
  const got = runProbe(rope, L, 64, 0, 50);
  const deep = got.layers[got.deepestIndex];
  check(`L=${L} rope omega`, Math.abs(deep.omega), Math.abs(ref[`L${got.deepestIndex}`].omega_rad_per_step), 0.02);
}

// ---- 5. the claim itself: fractional code vs fixed-rate code ---------------
// This is what Section 9 asserts, stated as a ratio over a 4x span of evaluation length, so it holds
// the paper's finding rather than any one number in its tables.
console.log(QUICK ? "\nSection 9's end-to-end claim: skipped (--quick)" : "\nSection 9's claim, measured end to end (L=16 -> L=64)");
if (!QUICK) {
  const a = runProbe(m, 16, 32, 0, 10).layers[m.cfg.n_layers - 1];
  const b = runProbe(m, 64, 32, 0, 10).layers[m.cfg.n_layers - 1];
  check("complex+NoPE omega ratio (1/T predicts 0.25)", Math.abs(b.omega / a.omega), 0.25, 0.03);
  check("complex+NoPE sweep ratio (fractional predicts 1.0)", b.sweep / a.sweep, 1.0, 0.08);
  const ra = runProbe(rope, 16, 32, 0, 10).layers[rope.cfg.n_layers - 1];
  const rb = runProbe(rope, 64, 32, 0, 10).layers[rope.cfg.n_layers - 1];
  check("real+RoPE omega ratio (fixed rate predicts 1.0)", Math.abs(rb.omega / ra.omega), 1.0, 0.15);
  check("real+RoPE sweep ratio (fixed rate predicts ~4)", rb.sweep / ra.sweep, 4.0, 0.7);
}

console.log(failures === 0 ? "\nall checks passed\n" : `\n${failures} CHECK(S) FAILED\n`);
process.exit(failures === 0 ? 0 : 1);
