/** Fixed entity -> hue/dash assignment. Never cycled, never reassigned by rank. */
export const SERIES: Record<string, { label: string; color: string; dash?: string }> = {
  "complex-nope-d160-s0": { label: "complex+NoPE d160", color: "var(--series-complex)" },
  "real-nope-d224-s0": { label: "real+NoPE d224 (matched)", color: "var(--series-real)", dash: "6 3" },
  "real-nope-d160-s0": { label: "real+NoPE d160", color: "var(--series-real)" },
  "real-rope-d160-s0": { label: "real+RoPE d160", color: "var(--series-rope)" },
};

export const dashFor = (dash: string | undefined) => dash ?? undefined;

export const fmt = (v: number, d = 3) =>
  Number.isFinite(v) ? v.toFixed(d) : "–";

/**
 * Sequential ramp for position on the dial: one hue, interpolated in OKLab between two theme steps.
 *
 * Written as `color-mix` against CSS variables rather than computed in JS so the ramp is selected per
 * surface. A single light-to-dark ramp reused on the dark surface buries the late positions, which
 * are the ones the reader is tracking.
 */
export const positionRamp = (t: number) =>
  `color-mix(in oklab, var(--ramp-hi) ${(t * 100).toFixed(1)}%, var(--ramp-lo))`;
