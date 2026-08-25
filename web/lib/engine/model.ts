/**
 * The trained transformer, rebuilt in the browser from a flat fp16 blob.
 *
 * This is a port of complexnn/models.py + complexnn/layers.py + complexnn/attention.py, verified
 * against PyTorch before it was written in TS: fp32 relative logit error 2.4e-6 (complex d160) and
 * 7.1e-6 (real d224); at the fp16 precision actually shipped, ~1e-3 with the argmax of every logit
 * unchanged at eval lengths 16, 24 and 48.
 */

import { add, complexRmsNorm, gelu, gemmNT, relu, rmsNorm, softmaxCausal } from "./kernels";

export const PAD = 0, SEP = 1, MARK = 2, N_SPECIAL = 3;

export interface ModelConfig {
  field: "real" | "complex";
  pe: "none" | "rope" | "learned";
  dim: number; n_layers: number; n_heads: number; mlp_ratio: number;
  learn_imag: boolean; algebra: "complex" | "split" | "dual" | "diag" | null;
  complex_act: string; rope_base: number; n_symbols: number; task: string;
  train_min: number; train_max: number;
}

export interface Manifest {
  id: string; label: string; config: ModelConfig; vocab_size: number;
  real_params: number; n_floats: number;
  paper_seq_acc: Record<string, { length: number; token_acc: number; seq_acc: number }>;
  tensors: Record<string, { offset: number; shape: number[] }>;
}

/** fp16 -> f32. Float16Array is ES2025 and not yet everywhere, so decode by hand when it is absent. */
function decodeF16(buf: ArrayBuffer): Float32Array {
  const u16 = new Uint16Array(buf);
  const F16 = (globalThis as { Float16Array?: new (b: ArrayBuffer) => ArrayLike<number> }).Float16Array;
  if (F16) return Float32Array.from(new F16(buf));
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * Math.pow(2, -14) * (m / 1024)
      : e === 31 ? (m ? NaN : s * Infinity)
        : s * Math.pow(2, e - 15) * (1 + m / 1024);
  }
  return out;
}

/** One forward pass' outputs. `resid[b]` is the input to block b; the last entry is the final stream. */
export interface Forward {
  logits: Float32Array;        // (T, vocab)
  T: number;
  resid: Float32Array[];       // n_layers + 1 buffers of (T, width), width = 2*dim for complex
  residWidth: number;
  attn: Float32Array[] | null; // per layer, (H, T, T) post-softmax weights
}

export class Model {
  readonly manifest: Manifest;
  readonly cfg: ModelConfig;
  private buf: Float32Array;

  constructor(manifest: Manifest, weights: ArrayBuffer) {
    this.manifest = manifest;
    this.cfg = manifest.config;
    this.buf = decodeF16(weights);
  }

  static async load(id: string, base = "/models"): Promise<Model> {
    const [manifest, weights] = await Promise.all([
      fetch(`${base}/${id}/manifest.json`).then((r) => r.json() as Promise<Manifest>),
      fetch(`${base}/${id}/weights.f16`).then((r) => r.arrayBuffer()),
    ]);
    return new Model(manifest, weights);
  }

  private t(name: string): Float32Array {
    const e = this.manifest.tensors[name];
    if (!e) throw new Error(`missing tensor ${name}`);
    const n = e.shape.reduce((a, b) => a * b, 1);
    return this.buf.subarray(e.offset, e.offset + n);
  }

  /** Gather token (and optionally learned-position) embeddings into a fresh (T, dim) buffer. */
  private embed(name: string, idx: Int32Array): Float32Array {
    const D = this.cfg.dim, W = this.t(name), out = new Float32Array(idx.length * D);
    for (let i = 0; i < idx.length; i++) out.set(W.subarray(idx[i] * D, idx[i] * D + D), i * D);
    return out;
  }

  /** y = W z under the configured 2D algebra. See complexnn/layers.py ALGEBRAS. */
  private complexLinear(
    p: string, xr: Float32Array, xi: Float32Array, M: number, K: number, N: number,
  ): [Float32Array, Float32Array] {
    const A = this.t(`${p}.A`), B = this.t(`${p}.B`);
    const alg = this.cfg.algebra ?? "complex";
    const t1 = new Float32Array(M * N), t2 = new Float32Array(M * N);
    const outR = new Float32Array(M * N), outI = new Float32Array(M * N);
    if (alg === "complex") {          // (A + iB)(x + iy)
      gemmNT(xr, A, t1, M, K, N); gemmNT(xi, B, t2, M, K, N);
      for (let i = 0; i < outR.length; i++) outR[i] = t1[i] - t2[i];
      gemmNT(xr, B, t1, M, K, N); gemmNT(xi, A, t2, M, K, N);
      for (let i = 0; i < outI.length; i++) outI[i] = t1[i] + t2[i];
    } else if (alg === "split") {     // j^2 = +1
      gemmNT(xr, A, t1, M, K, N); gemmNT(xi, B, t2, M, K, N);
      for (let i = 0; i < outR.length; i++) outR[i] = t1[i] + t2[i];
      gemmNT(xr, B, t1, M, K, N); gemmNT(xi, A, t2, M, K, N);
      for (let i = 0; i < outI.length; i++) outI[i] = t1[i] + t2[i];
    } else if (alg === "dual") {      // e^2 = 0, no y term in the real stream
      gemmNT(xr, A, outR, M, K, N);
      gemmNT(xr, B, t1, M, K, N); gemmNT(xi, A, t2, M, K, N);
      for (let i = 0; i < outI.length; i++) outI[i] = t1[i] + t2[i];
    } else {                          // diag: two independent streams, no mixing at all
      gemmNT(xr, A, outR, M, K, N);
      gemmNT(xi, B, outI, M, K, N);
    }
    return [outR, outI];
  }

  forward(idx: Int32Array, opts: { attn?: boolean; resid?: boolean } = {}): Forward {
    const c = this.cfg, T = idx.length, H = c.n_heads, D = c.dim, hd = D / H;
    const hidden = Math.round(D * c.mlp_ratio);
    const eps32 = 1.1920929e-7;                      // torch nn.RMSNorm default: finfo(float32).eps
    const resid: Float32Array[] = [], attn: Float32Array[] = [];
    const wantR = opts.resid ?? false, wantA = opts.attn ?? false;

    if (c.field === "complex") {
      const zr = this.embed("tok.weight", idx);
      const zi = c.learn_imag ? this.embed("tok_imag.weight", idx) : new Float32Array(T * D);
      const nr = new Float32Array(T * D), ni = new Float32Array(T * D);

      for (let b = 0; b < c.n_layers; b++) {
        if (wantR) resid.push(cat(zr, zi, T, D));
        const p = `blocks.${b}.`;
        complexRmsNorm(zr, zi, this.t(`${p}n1.weight`), nr, ni, T, D);
        const [qr, qi] = this.complexLinear(`${p}attn.q`, nr, ni, T, D, D);
        const [kr, ki] = this.complexLinear(`${p}attn.k`, nr, ni, T, D, D);
        const [vr, vi] = this.complexLinear(`${p}attn.v`, nr, ni, T, D, D);
        if (c.pe === "rope") { ropeComplex(qr, qi, T, H, hd, c.rope_base); ropeComplex(kr, ki, T, H, hd, c.rope_base); }

        // Re(q k^H) = sum_d (q.re k.re + q.im k.im): real logits from complex q/k
        const l = new Float32Array(H * T * T), scale = 1 / Math.sqrt(2 * hd);
        for (let h = 0; h < H; h++)
          for (let i = 0; i < T; i++)
            for (let j = 0; j <= i; j++) {
              let s = 0; const oi_ = i * D + h * hd, oj = j * D + h * hd;
              for (let d = 0; d < hd; d++) s += qr[oi_ + d] * kr[oj + d] + qi[oi_ + d] * ki[oj + d];
              l[h * T * T + i * T + j] = s * scale;
            }
        softmaxCausal(l, H, T);
        if (wantA) attn.push(l.slice());

        const or_ = new Float32Array(T * D), oi_ = new Float32Array(T * D);
        attendInto(l, vr, or_, H, T, D, hd); attendInto(l, vi, oi_, H, T, D, hd);
        const [pr, pi] = this.complexLinear(`${p}attn.proj`, or_, oi_, T, D, D);
        add(zr, pr); add(zi, pi);

        complexRmsNorm(zr, zi, this.t(`${p}n2.weight`), nr, ni, T, D);
        let [fr, fi] = this.complexLinear(`${p}mlp.fc1`, nr, ni, T, D, hidden);
        relu(fr); relu(fi);                                    // CReLU
        [fr, fi] = this.complexLinear(`${p}mlp.fc2`, fr, fi, T, hidden, D);
        add(zr, fr); add(zi, fi);
      }
      if (wantR) resid.push(cat(zr, zi, T, D));
      complexRmsNorm(zr, zi, this.t("norm_f.weight"), nr, ni, T, D);
      const head = this.t("head.weight"), V = this.manifest.vocab_size;
      const logits = new Float32Array(T * V);
      gemmNT(cat(nr, ni, T, D), head, logits, T, 2 * D, V);     // head reads [Re || Im]
      return { logits, T, resid, residWidth: 2 * D, attn: wantA ? attn : null };
    }

    const x = this.embed("tok.weight", idx);
    if (c.pe === "learned") {
      const P = this.t("pos.weight");
      for (let i = 0; i < T; i++) for (let d = 0; d < D; d++) x[i * D + d] += P[i * D + d];
    }
    const n = new Float32Array(T * D);
    for (let b = 0; b < c.n_layers; b++) {
      if (wantR) resid.push(x.slice());
      const p = `blocks.${b}.`;
      rmsNorm(x, this.t(`${p}n1.weight`), n, T, D, eps32);
      const qkv = new Float32Array(T * 3 * D);
      gemmNT(n, this.t(`${p}attn.qkv.weight`), qkv, T, D, 3 * D);
      const q = slab(qkv, T, 3 * D, 0, D), k = slab(qkv, T, 3 * D, D, D), v = slab(qkv, T, 3 * D, 2 * D, D);
      if (c.pe === "rope") { ropeReal(q, T, H, hd, c.rope_base); ropeReal(k, T, H, hd, c.rope_base); }
      const l = new Float32Array(H * T * T), scale = 1 / Math.sqrt(hd);
      for (let h = 0; h < H; h++)
        for (let i = 0; i < T; i++)
          for (let j = 0; j <= i; j++) {
            let s = 0; const oi_ = i * D + h * hd, oj = j * D + h * hd;
            for (let d = 0; d < hd; d++) s += q[oi_ + d] * k[oj + d];
            l[h * T * T + i * T + j] = s * scale;
          }
      softmaxCausal(l, H, T);
      if (wantA) attn.push(l.slice());
      const o = new Float32Array(T * D);
      attendInto(l, v, o, H, T, D, hd);
      const proj = new Float32Array(T * D);
      gemmNT(o, this.t(`${p}attn.proj.weight`), proj, T, D, D);
      add(x, proj);

      rmsNorm(x, this.t(`${p}n2.weight`), n, T, D, eps32);
      const h1 = new Float32Array(T * hidden);
      gemmNT(n, this.t(`${p}mlp.fc1.weight`), h1, T, D, hidden);
      gelu(h1);
      const h2 = new Float32Array(T * D);
      gemmNT(h1, this.t(`${p}mlp.fc2.weight`), h2, T, hidden, D);
      add(x, h2);
    }
    if (wantR) resid.push(x.slice());
    rmsNorm(x, this.t("norm_f.weight"), n, T, D, eps32);
    const V = this.manifest.vocab_size, logits = new Float32Array(T * V);
    gemmNT(n, this.t("head.weight"), logits, T, D, V);
    return { logits, T, resid, residWidth: D, attn: wantA ? attn : null };
  }
}

// ------------------------------------------------------------------ helpers
const cat = (a: Float32Array, b: Float32Array, T: number, D: number) => {
  const o = new Float32Array(T * 2 * D);
  for (let i = 0; i < T; i++) { o.set(a.subarray(i * D, i * D + D), i * 2 * D); o.set(b.subarray(i * D, i * D + D), i * 2 * D + D); }
  return o;
};

const slab = (src: Float32Array, T: number, stride: number, off: number, D: number) => {
  const o = new Float32Array(T * D);
  for (let i = 0; i < T; i++) o.set(src.subarray(i * stride + off, i * stride + off + D), i * D);
  return o;
};

/** out(T, D) = attn(H, T, T) @ v(T, D), per head. */
function attendInto(a: Float32Array, v: Float32Array, out: Float32Array, H: number, T: number, D: number, hd: number) {
  for (let h = 0; h < H; h++)
    for (let i = 0; i < T; i++) {
      const ao = h * T * T + i * T, oo = i * D + h * hd;
      for (let j = 0; j <= i; j++) {
        const w = a[ao + j]; if (w === 0) continue;
        const vo = j * D + h * hd;
        for (let d = 0; d < hd; d++) out[oo + d] += w * v[vo + d];
      }
    }
}

/** Real RoPE, half-split rotate_half convention (complexnn/attention.build_rope_real). */
function ropeReal(x: Float32Array, T: number, H: number, hd: number, base: number) {
  const half = hd / 2, D = H * hd;
  for (let i = 0; i < T; i++)
    for (let h = 0; h < H; h++) {
      const o = i * D + h * hd;
      for (let d = 0; d < half; d++) {
        const f = i / Math.pow(base, (2 * d) / hd), c = Math.cos(f), s = Math.sin(f);
        const a = x[o + d], b = x[o + d + half];
        x[o + d] = a * c - b * s; x[o + d + half] = b * c + a * s;
      }
    }
}

/** Complex RoPE: one phasor per complex channel (complexnn/attention.build_rope_complex). */
function ropeComplex(xr: Float32Array, xi: Float32Array, T: number, H: number, hd: number, base: number) {
  const D = H * hd;
  for (let i = 0; i < T; i++)
    for (let h = 0; h < H; h++) {
      const o = i * D + h * hd;
      for (let d = 0; d < hd; d++) {
        const f = i / Math.pow(base, d / hd), c = Math.cos(f), s = Math.sin(f);
        const a = xr[o + d], b = xi[o + d];
        xr[o + d] = a * c - b * s; xi[o + d] = a * s + b * c;
      }
    }
}
