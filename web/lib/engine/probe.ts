/**
 * The rotating-plane probe of the paper's Section 9, computed in the browser.
 *
 * Port of experiments/activation_probe.py: project the position-conditional mean residual stream on
 * its own top-two principal plane, read each position's in-plane angle, unwrap, and fit a line
 * against position. The slope is the rate omega (rad/position); the rise across the sequence is the
 * total sweep. A fixed-rate code (RoPE) holds omega as T grows; a fractional code holds the sweep.
 *
 * The null shuffles the position labels. That leaves the point cloud and its principal axes
 * untouched and destroys only the ordering, so it needs no re-decomposition: permuting the already
 * computed coordinates is the whole operation, which is why the null is cheap enough to run live.
 */

export interface PlaneFit {
  r2: number; omega: number; sweep: number; T: number;
  top2VarShare: number; radiusCv: number; pc1RampR2: number;
  coords: Float32Array;   // (T, 2) in-plane coordinates, in position order
  angles: Float32Array;   // (T,) unwrapped angle
  fitted: Float32Array;   // (T,) the fitted line, for drawing
}

export interface NullResult { nullR2: Float32Array; p: number; p95: number; mean: number }

/** Least-squares y ~ m. Returns [r2, slope, intercept]. */
function linFit(y: Float32Array, m: Float32Array): [number, number, number] {
  const n = y.length;
  let sm = 0, sy = 0;
  for (let i = 0; i < n; i++) { sm += m[i]; sy += y[i]; }
  const mm = sm / n, my = sy / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (m[i] - mm) * (y[i] - my); den += (m[i] - mm) * (m[i] - mm); }
  const slope = den > 0 ? num / den : 0, icpt = my - slope * mm;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < n; i++) {
    const r = y[i] - (slope * m[i] + icpt);
    ssRes += r * r; ssTot += (y[i] - my) * (y[i] - my);
  }
  return [1 - ssRes / Math.max(ssTot, 1e-12), slope, icpt];
}

/**
 * Top-two principal coordinates of H (T x W), column-centred.
 *
 * Via the T x T Gram matrix rather than a W x W covariance or a full SVD: T is at most ~130 here
 * while W is up to 896, and the eigenvectors of the Gram matrix are exactly the left singular
 * vectors we need, already scaled by the singular values. Block power iteration on two columns
 * converges to the top-two invariant subspace; a 2x2 Rayleigh-Ritz step then separates PC1 from PC2
 * inside it, which matters because the angle is measured *in* that plane.
 */
function topTwoPlane(H: Float32Array, T: number, W: number): { coords: Float32Array; top2: number } {
  const mean = new Float32Array(W);
  for (let i = 0; i < T; i++) for (let d = 0; d < W; d++) mean[d] += H[i * W + d];
  for (let d = 0; d < W; d++) mean[d] /= T;
  const Hc = new Float32Array(T * W);
  for (let i = 0; i < T; i++) for (let d = 0; d < W; d++) Hc[i * W + d] = H[i * W + d] - mean[d];

  const G = new Float64Array(T * T);
  let trace = 0;
  for (let i = 0; i < T; i++)
    for (let j = i; j < T; j++) {
      let s = 0;
      for (let d = 0; d < W; d++) s += Hc[i * W + d] * Hc[j * W + d];
      G[i * T + j] = s; G[j * T + i] = s;
      if (i === j) trace += s;
    }

  // block power iteration on a 2-column basis
  let v0 = new Float64Array(T), v1 = new Float64Array(T);
  for (let i = 0; i < T; i++) { v0[i] = Math.sin(i * 0.7 + 1); v1[i] = Math.cos(i * 1.3 + 2); }
  const mul = (v: Float64Array) => {
    const o = new Float64Array(T);
    for (let i = 0; i < T; i++) { let s = 0; const r = i * T; for (let j = 0; j < T; j++) s += G[r + j] * v[j]; o[i] = s; }
    return o;
  };
  const orth = () => {
    let n0 = 0; for (let i = 0; i < T; i++) n0 += v0[i] * v0[i];
    n0 = Math.sqrt(n0) || 1; for (let i = 0; i < T; i++) v0[i] /= n0;
    let d = 0; for (let i = 0; i < T; i++) d += v1[i] * v0[i];
    for (let i = 0; i < T; i++) v1[i] -= d * v0[i];
    let n1 = 0; for (let i = 0; i < T; i++) n1 += v1[i] * v1[i];
    n1 = Math.sqrt(n1) || 1; for (let i = 0; i < T; i++) v1[i] /= n1;
  };
  orth();
  for (let it = 0; it < 200; it++) { v0 = mul(v0); v1 = mul(v1); orth(); }

  // Rayleigh-Ritz inside the 2D subspace: diagonalise the 2x2 projection to split PC1 from PC2
  const g0 = mul(v0), g1 = mul(v1);
  let a = 0, b = 0, c = 0;
  for (let i = 0; i < T; i++) { a += v0[i] * g0[i]; b += v0[i] * g1[i]; c += v1[i] * g1[i]; }
  const th = 0.5 * Math.atan2(2 * b, a - c), ct = Math.cos(th), st = Math.sin(th);
  const e0 = new Float64Array(T), e1 = new Float64Array(T);
  for (let i = 0; i < T; i++) { e0[i] = ct * v0[i] + st * v1[i]; e1[i] = -st * v0[i] + ct * v1[i]; }
  const l0 = a * ct * ct + 2 * b * ct * st + c * st * st;
  const l1 = a * st * st - 2 * b * ct * st + c * ct * ct;

  // coordinates are the eigenvectors scaled by the singular value sqrt(lambda)
  const [s0, s1] = [Math.sqrt(Math.max(l0, 0)), Math.sqrt(Math.max(l1, 0))];
  const coords = new Float32Array(T * 2);
  for (let i = 0; i < T; i++) { coords[i * 2] = e0[i] * s0; coords[i * 2 + 1] = e1[i] * s1; }
  return { coords, top2: (l0 + l1) / Math.max(trace, 1e-12) };
}

/** Unwrap in place, matching numpy.unwrap's default period. */
function unwrap(a: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  out[0] = a[0];
  let corr = 0;
  for (let i = 1; i < a.length; i++) {
    const d = a[i] - a[i - 1];
    if (d > Math.PI) corr -= 2 * Math.PI;
    else if (d < -Math.PI) corr += 2 * Math.PI;
    out[i] = a[i] + corr;
  }
  return out;
}

export function planeFit(H: Float32Array, T: number, W: number): PlaneFit {
  const { coords, top2 } = topTwoPlane(H, T, W);
  // The sign of PC2 is arbitrary (an eigenvector is defined up to sign), so the sign of omega is too:
  // the committed metrics carry both signs for the same run at different lengths. Fix it to positive
  // so the UI can compare rates across lengths without the reader parsing a meaningless flip.
  {
    let net = 0;
    for (let i = 1; i < T; i++) {
      let d = Math.atan2(coords[i * 2 + 1], coords[i * 2]) - Math.atan2(coords[(i - 1) * 2 + 1], coords[(i - 1) * 2]);
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      net += d;
    }
    if (net < 0) for (let i = 0; i < T; i++) coords[i * 2 + 1] = -coords[i * 2 + 1];
  }
  const raw = new Float32Array(T), m = new Float32Array(T);
  let rSum = 0, rSq = 0;
  for (let i = 0; i < T; i++) {
    raw[i] = Math.atan2(coords[i * 2 + 1], coords[i * 2]);
    m[i] = i;
    const rad = Math.hypot(coords[i * 2], coords[i * 2 + 1]);
    rSum += rad; rSq += rad * rad;
  }
  const angles = unwrap(raw);
  const [r2, omega, icpt] = linFit(angles, m);
  const pc1 = new Float32Array(T);
  for (let i = 0; i < T; i++) pc1[i] = coords[i * 2];
  const [rampR2] = linFit(pc1, m);
  const rMean = rSum / T, rStd = Math.sqrt(Math.max(rSq / T - rMean * rMean, 0));
  const fitted = new Float32Array(T);
  for (let i = 0; i < T; i++) fitted[i] = omega * i + icpt;
  return {
    r2, omega, sweep: Math.abs(omega) * (T - 1), T, top2VarShare: top2,
    radiusCv: rStd / Math.max(rMean, 1e-12), pc1RampR2: rampR2, coords, angles, fitted,
  };
}

/** The paper's null: shuffle the position labels, keep the cloud. Reuses the fitted coordinates. */
export function shuffleNull(fit: PlaneFit, nNull: number, seed = 1): NullResult {
  const T = fit.T, nullR2 = new Float32Array(nNull);
  let s = seed >>> 0;
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const perm = new Int32Array(T), m = new Float32Array(T), raw = new Float32Array(T);
  for (let i = 0; i < T; i++) m[i] = i;
  for (let n = 0; n < nNull; n++) {
    for (let i = 0; i < T; i++) perm[i] = i;
    for (let i = T - 1; i > 0; i--) { const j = (rand() * (i + 1)) | 0; const t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    for (let i = 0; i < T; i++) {
      const k = perm[i];
      raw[i] = Math.atan2(fit.coords[k * 2 + 1], fit.coords[k * 2]);
    }
    nullR2[n] = linFit(unwrap(raw), m)[0];
  }
  const sorted = Float32Array.from(nullR2).sort();
  let ge = 0, sum = 0;
  for (let i = 0; i < nNull; i++) { if (nullR2[i] >= fit.r2) ge++; sum += nullR2[i]; }
  return {
    nullR2, p: ge / nNull, mean: sum / nNull,
    p95: sorted[Math.min(nNull - 1, Math.floor(0.95 * nNull))],
  };
}
