"use client";

/**
 * A small SVG chart kit: scales, recessive axes, and a shared crosshair-plus-tooltip layer.
 *
 * Hand-rolled rather than pulled from a chart library because three of the four figures on this page
 * (the dial, the unwrapped-angle fit, the attention staircase) are not standard chart types, and a
 * library would be carried for the two that are.
 */

import { ReactNode, useCallback, useEffect, useRef, useState } from "react";

/**
 * The rendered pixel width of a container.
 *
 * Charts are drawn at their true pixel size rather than at a fixed viewBox scaled with `width=100%`:
 * a viewBox stretches the height along with the width, and it also decouples the SVG's coordinate
 * space from the CSS pixels the tooltip is positioned in, so tooltips drift away from their marks at
 * any size but the design size.
 */
export function useMeasure<T extends HTMLElement>(fallback = 620) {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export interface Scale { (v: number): number; domain: [number, number]; range: [number, number] }

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain, [r0, r1] = range;
  const span = d1 - d0 || 1;
  const f = ((v: number) => r0 + ((v - d0) / span) * (r1 - r0)) as Scale;
  f.domain = domain; f.range = range;
  return f;
}

export const invert = (s: Scale, px: number) =>
  s.domain[0] + ((px - s.range[0]) / (s.range[1] - s.range[0] || 1)) * (s.domain[1] - s.domain[0]);

export function ticks(d0: number, d1: number, count = 5): number[] {
  const span = d1 - d0;
  if (span === 0) return [d0];
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const out: number[] = [];
  for (let t = Math.ceil(d0 / step) * step; t <= d1 + 1e-9; t += step) out.push(Math.abs(t) < 1e-9 ? 0 : t);
  return out;
}

export const PAD = { top: 14, right: 18, bottom: 34, left: 46 };

export function Frame({
  width, height, x, y, xLabel, yLabel, xTicks, yTicks, fmtX, fmtY, children, pad = PAD,
}: {
  width: number; height: number; x: Scale; y: Scale;
  xLabel?: string; yLabel?: string; xTicks?: number[]; yTicks?: number[];
  fmtX?: (v: number) => string; fmtY?: (v: number) => string;
  children: ReactNode; pad?: typeof PAD;
}) {
  const xs = xTicks ?? ticks(x.domain[0], x.domain[1], 5);
  const ys = yTicks ?? ticks(y.domain[0], y.domain[1], 4);
  return (
    <g>
      {ys.map((t) => (
        <g key={`y${t}`}>
          <line x1={pad.left} x2={width - pad.right} y1={y(t)} y2={y(t)}
            stroke="var(--grid)" strokeWidth={0.75} />
          <text x={pad.left - 7} y={y(t)} dy="0.32em" textAnchor="end"
            fill="var(--text-muted)" fontSize={10.5} className="tnum">
            {(fmtY ?? ((v: number) => String(v)))(t)}
          </text>
        </g>
      ))}
      {xs.map((t) => (
        <text key={`x${t}`} x={x(t)} y={height - pad.bottom + 15} textAnchor="middle"
          fill="var(--text-muted)" fontSize={10.5} className="tnum">
          {(fmtX ?? ((v: number) => String(v)))(t)}
        </text>
      ))}
      <line x1={pad.left} x2={width - pad.right} y1={height - pad.bottom} y2={height - pad.bottom}
        stroke="var(--rule)" strokeWidth={0.75} />
      {children}
      {xLabel && (
        <text x={(pad.left + width - pad.right) / 2} y={height - 2} textAnchor="middle"
          fill="var(--text-secondary)" fontSize={11}>{xLabel}</text>
      )}
      {yLabel && (
        <text transform={`rotate(-90) translate(${-(pad.top + height - pad.bottom) / 2} 11)`}
          textAnchor="middle" fill="var(--text-secondary)" fontSize={11}>{yLabel}</text>
      )}
    </g>
  );
}

export function Legend({ items }: {
  items: { label: string; color: string; dash?: string; mark?: "line" | "dot" }[];
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
      {items.map((it) => (
        <span key={it.label} className="inline-flex items-center gap-1.5 text-[11.5px]"
          style={{ color: "var(--text-secondary)" }}>
          <svg width="18" height="10" aria-hidden>
            {it.mark === "dot"
              ? <circle cx="9" cy="5" r="4.5" fill={it.color} />
              : <line x1="0" y1="5" x2="18" y2="5" stroke={it.color} strokeWidth="2"
                  strokeDasharray={it.dash} strokeLinecap="round" />}
          </svg>
          {it.label}
        </span>
      ))}
    </div>
  );
}

export interface HoverState { x: number; y: number; content: ReactNode }

/** Tooltip positioned in the chart's own coordinate space, flipped near the right edge. */
export function Tooltip({ hover, width }: { hover: HoverState | null; width: number }) {
  if (!hover) return null;
  const flip = hover.x > width * 0.62;
  return (
    <div
      className="pointer-events-none absolute z-10 rounded-md px-2.5 py-1.5 text-[11.5px] leading-snug shadow-sm"
      style={{
        left: hover.x, top: hover.y, transform: `translate(${flip ? "-100%" : "0"}, -50%)`,
        marginLeft: flip ? -10 : 10,
        background: "var(--card)", color: "var(--text-primary)",
        border: "1px solid var(--rule)", whiteSpace: "nowrap",
      }}
    >
      {hover.content}
    </div>
  );
}

/** Wraps an SVG so children can report hover positions in element coordinates. */
export function Plot({
  width, height, children, onProbe, className, ariaLabel, hover: controlled,
}: {
  width: number; height: number; children: (h: (s: HoverState | null) => void) => ReactNode;
  onProbe?: (px: number, py: number) => void; className?: string; ariaLabel: string;
  /** Charts that derive the hovered point from a pointer position (rather than from a mark under the
   *  cursor) own that state themselves and pass it here. Setting it through the render callback
   *  instead would be a state update during render. */
  hover?: HoverState | null;
}) {
  const [internal, setInternal] = useState<HoverState | null>(null);
  const hover = controlled !== undefined ? controlled : internal;
  const setHover = setInternal;
  const ref = useRef<SVGSVGElement>(null);
  const move = useCallback((e: React.PointerEvent) => {
    if (!onProbe || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    onProbe(e.clientX - r.left, e.clientY - r.top);
  }, [onProbe]);
  return (
    <div className={`relative ${className ?? ""}`}>
      <svg ref={ref} width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img"
        aria-label={ariaLabel}
        onPointerMove={move} onPointerLeave={() => { setInternal(null); onProbe?.(NaN, NaN); }}
        style={{ display: "block", touchAction: "pan-y", maxWidth: "100%" }}>
        {children(setHover)}
      </svg>
      <Tooltip hover={hover} width={width} />
    </div>
  );
}

/** A polyline through (x, y) pairs already in pixel space. */
export const path = (pts: [number, number][]) =>
  pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
