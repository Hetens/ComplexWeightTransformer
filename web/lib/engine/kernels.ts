/**
 * Dense f32 kernels. Everything the model needs, written against typed arrays.
 *
 * The only kernel that matters for speed is `gemmNT`, which is 4x4 register-blocked: measured at
 * ~2.5 GFLOP/s in V8 against ~0.7 for the naive triple loop, which is the difference between a
 * two-second probe pass and a seven-second one.
 */

/** C(M x N) = A(M x K) * B(N x K)^T. B is row-major [N, K], which is how torch stores nn.Linear. */
export function gemmNT(
  A: Float32Array, B: Float32Array, C: Float32Array, M: number, K: number, N: number,
): void {
  C.fill(0);
  const M4 = M - (M % 4), N4 = N - (N % 4);
  for (let i = 0; i < M4; i += 4) {
    const a0 = i * K, a1 = a0 + K, a2 = a1 + K, a3 = a2 + K;
    const r0 = i * N, r1 = r0 + N, r2 = r1 + N, r3 = r2 + N;
    for (let j = 0; j < N4; j += 4) {
      const b0 = j * K, b1 = b0 + K, b2 = b1 + K, b3 = b2 + K;
      let c00 = 0, c01 = 0, c02 = 0, c03 = 0, c10 = 0, c11 = 0, c12 = 0, c13 = 0;
      let c20 = 0, c21 = 0, c22 = 0, c23 = 0, c30 = 0, c31 = 0, c32 = 0, c33 = 0;
      for (let k = 0; k < K; k++) {
        const x0 = A[a0 + k], x1 = A[a1 + k], x2 = A[a2 + k], x3 = A[a3 + k];
        const y0 = B[b0 + k], y1 = B[b1 + k], y2 = B[b2 + k], y3 = B[b3 + k];
        c00 += x0 * y0; c01 += x0 * y1; c02 += x0 * y2; c03 += x0 * y3;
        c10 += x1 * y0; c11 += x1 * y1; c12 += x1 * y2; c13 += x1 * y3;
        c20 += x2 * y0; c21 += x2 * y1; c22 += x2 * y2; c23 += x2 * y3;
        c30 += x3 * y0; c31 += x3 * y1; c32 += x3 * y2; c33 += x3 * y3;
      }
      C[r0 + j] = c00; C[r0 + j + 1] = c01; C[r0 + j + 2] = c02; C[r0 + j + 3] = c03;
      C[r1 + j] = c10; C[r1 + j + 1] = c11; C[r1 + j + 2] = c12; C[r1 + j + 3] = c13;
      C[r2 + j] = c20; C[r2 + j + 1] = c21; C[r2 + j + 2] = c22; C[r2 + j + 3] = c23;
      C[r3 + j] = c30; C[r3 + j + 1] = c31; C[r3 + j + 2] = c32; C[r3 + j + 3] = c33;
    }
    for (let j = N4; j < N; j++)
      for (let ii = 0; ii < 4; ii++) {
        let s = 0; const ao = (i + ii) * K, bo = j * K;
        for (let k = 0; k < K; k++) s += A[ao + k] * B[bo + k];
        C[(i + ii) * N + j] = s;
      }
  }
  for (let i = M4; i < M; i++)
    for (let j = 0; j < N; j++) {
      let s = 0; const ao = i * K, bo = j * K;
      for (let k = 0; k < K; k++) s += A[ao + k] * B[bo + k];
      C[i * N + j] = s;
    }
}

export const add = (a: Float32Array, b: Float32Array) => { for (let i = 0; i < a.length; i++) a[i] += b[i]; };
export const relu = (a: Float32Array) => { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0; };

/** Abramowitz & Stegun 7.1.26, |err| < 1.5e-7. torch's F.gelu defaults to the *exact* erf form, not
 *  the tanh approximation; using tanh here leaves a 9e-3 relative logit error that fp32 will not fix. */
function erf(x: number): number {
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
    + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
export const gelu = (a: Float32Array) => {
  for (let i = 0; i < a.length; i++) a[i] = 0.5 * a[i] * (1 + erf(a[i] * Math.SQRT1_2));
};

/** RMSNorm over the last dim of an (M, D) buffer, into `out`. */
export function rmsNorm(
  x: Float32Array, g: Float32Array, out: Float32Array, M: number, D: number, eps: number,
): void {
  for (let i = 0; i < M; i++) {
    const o = i * D;
    let ms = 0;
    for (let d = 0; d < D; d++) ms += x[o + d] * x[o + d];
    const s = 1 / Math.sqrt(ms / D + eps);
    for (let d = 0; d < D; d++) out[o + d] = x[o + d] * s * g[d];
  }
}

/** Magnitude RMSNorm, phase preserving: the norm is over |z|^2 across both streams jointly. */
export function complexRmsNorm(
  xr: Float32Array, xi: Float32Array, g: Float32Array,
  or_: Float32Array, oi: Float32Array, M: number, D: number, eps = 1e-5,
): void {
  for (let i = 0; i < M; i++) {
    const o = i * D;
    let ms = 0;
    for (let d = 0; d < D; d++) ms += xr[o + d] * xr[o + d] + xi[o + d] * xi[o + d];
    const s = 1 / Math.sqrt(ms / D + eps);
    for (let d = 0; d < D; d++) { or_[o + d] = xr[o + d] * s * g[d]; oi[o + d] = xi[o + d] * s * g[d]; }
  }
}

/** In-place causal softmax over a (H, T, T) logits buffer, with per-sequence query offset `qBase`. */
export function softmaxCausal(l: Float32Array, H: number, T: number): void {
  for (let h = 0; h < H; h++)
    for (let i = 0; i < T; i++) {
      const o = h * T * T + i * T;
      let m = -Infinity;
      for (let j = 0; j <= i; j++) if (l[o + j] > m) m = l[o + j];
      let s = 0;
      for (let j = 0; j <= i; j++) { const e = Math.exp(l[o + j] - m); l[o + j] = e; s += e; }
      const inv = 1 / s;
      for (let j = 0; j <= i; j++) l[o + j] *= inv;
      for (let j = i + 1; j < T; j++) l[o + j] = 0;
    }
}
