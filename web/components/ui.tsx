"use client";

import { ReactNode } from "react";

export function Section({ id, kicker, title, lede, children }: {
  id: string; kicker: string; title: string; lede?: ReactNode; children: ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-16 border-t py-14" style={{ borderColor: "var(--rule)" }}>
      <p className="mb-1.5 text-[11px] font-medium uppercase tracking-[0.13em]"
        style={{ color: "var(--accent)" }}>{kicker}</p>
      <h2 className="mb-3 text-[26px] leading-tight tracking-[-0.01em]"
        style={{ fontFamily: "var(--font-serif)" }}>{title}</h2>
      {lede && (
        <div className="mb-7 max-w-[68ch] text-[15px] leading-relaxed"
          style={{ color: "var(--text-secondary)" }}>{lede}</div>
      )}
      {children}
    </section>
  );
}

export function Card({ children, className = "", innerRef }: {
  children: ReactNode; className?: string; innerRef?: React.Ref<HTMLDivElement>;
}) {
  return (
    <div ref={innerRef} className={`rounded-xl border p-5 ${className}`}
      style={{ borderColor: "var(--rule)", background: "var(--card)" }}>
      {children}
    </div>
  );
}

export function Stat({ label, value, unit, hint, color }: {
  label: string; value: string; unit?: string; hint?: string; color?: string;
}) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-[0.08em]" style={{ color: "var(--text-muted)" }}>
        {label}
      </div>
      <div className="mt-0.5 flex items-baseline gap-1">
        <span className="tnum text-[25px] leading-none" style={{ color: color ?? "var(--text-primary)" }}>
          {value}
        </span>
        {unit && <span className="text-[12px]" style={{ color: "var(--text-muted)" }}>{unit}</span>}
      </div>
      {hint && <div className="mt-1 text-[11.5px]" style={{ color: "var(--text-muted)" }}>{hint}</div>}
    </div>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] uppercase tracking-[0.08em]"
        style={{ color: "var(--text-muted)" }}>{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[11px]" style={{ color: "var(--text-muted)" }}>{hint}</span>}
    </label>
  );
}

export function Select<T extends string>({ value, onChange, options }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: string }[];
}) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value as T)}
      className="w-full rounded-md border px-2.5 py-1.5 text-[13px]"
      style={{ borderColor: "var(--rule)", background: "var(--card)", color: "var(--text-primary)" }}>
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}

export function Slider({ value, onChange, min, max, step = 1, marks }: {
  value: number; onChange: (v: number) => void; min: number; max: number; step?: number;
  marks?: { at: number; label: string }[];
}) {
  return (
    <div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[var(--accent)]" style={{ height: 22 }} />
      {marks && (
        <div className="relative mt-0.5 h-3.5">
          {marks.map((m) => (
            <span key={m.label} className="absolute -translate-x-1/2 text-[10.5px] tnum"
              style={{ left: `${((m.at - min) / (max - min)) * 100}%`, color: "var(--text-muted)" }}>
              {m.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export function Busy({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[11.5px]" style={{ color: "var(--text-muted)" }}>
      <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full"
        style={{ background: "var(--accent)" }} />
      {children}
    </span>
  );
}
