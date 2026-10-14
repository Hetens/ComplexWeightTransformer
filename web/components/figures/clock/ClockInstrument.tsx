"use client";

/**
 * The two-part positional clock as a bench instrument (static art from scripts/build-clock-figure.mjs).
 *
 * The front carriage reads along the input tape and turns the input dial: the measured clock angle at
 * that input position, which saturates past the 16 trained cells. The back carriage runs on a rail of
 * fixed length whose ticks get finer as the evaluation length grows, and turns the answer dial by
 * omega_L * j. The console selects which plane is removed and reads the five-seed token accuracy.
 *
 * Everything moving is written to the DOM from a requestAnimationFrame loop through frame.ts; React
 * re-renders only when a control changes.
 */

import "../../iso/iso.css";
import "./clock.css";
import { useEffect, useRef, useState } from "react";
import * as k from "../../iso/iso-kit";
import { CLOCK_DATA, CLOCK_SVG } from "./art.generated";
import { accuracyOf, cellsOf, frameOf, knobAngle, readoutOf, type Ablation, type ModelKey, type State } from "./frame";

const MODELS: { key: ModelKey; label: string }[] = [
  { key: "complex", label: "complex+NoPE" },
  { key: "diag", label: "block-diagonal" },
  { key: "real", label: "real+NoPE d224" },
];
const LENGTHS = [16, 24, 32, 48, 64];
const ABLATIONS: { key: Ablation; label: string }[] = [
  { key: "none", label: "nothing removed" },
  { key: "clock", label: "clock plane removed" },
  { key: "control", label: "control plane removed" },
];
const SECONDS_PER_TOKEN = 0.11;
const DWELL = 1.4;

function Choice<T extends string | number>({ label, value, options, onChange }: {
  label: string; value: T; options: { key: T; label: string }[]; onChange: (v: T) => void;
}) {
  return (
    <div className="clock-choice" role="radiogroup" aria-label={label}>
      <span className="clock-choice-label">{label}</span>
      {options.map((o) => (
        <button key={String(o.key)} type="button" role="radio" aria-checked={o.key === value}
          className="clock-chip" data-on={o.key === value ? "" : undefined} onClick={() => onChange(o.key)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function ClockInstrument() {
  const [model, setModel] = useState<ModelKey>("complex");
  const [length, setLength] = useState(32);
  const [ablation, setAblation] = useState<Ablation>("none");
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const stageRef = useRef<HTMLDivElement>(null);
  const readoutRef = useRef<HTMLSpanElement>(null);
  const live = useRef({ model, length, ablation });
  const nudge = useRef<(dir: number) => void>(() => {});

  // controls feed the loop through a ref, so the loop is set up once
  useEffect(() => { live.current = { model, length, ablation }; }, [model, length, ablation]);

  useEffect(() => {
    const dark = matchMedia("(prefers-color-scheme: dark)");
    const set = () => setTheme(document.documentElement.dataset.theme === "light" || !dark.matches ? "light" : "dark");
    set();
    dark.addEventListener("change", set);
    return () => dark.removeEventListener("change", set);
  }, []);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const q = (part: string) => stage.querySelector<SVGElement>(`[data-part="${part}"]`);
    const el = {
      inputCarriage: q("input-carriage"), answerCarriage: q("answer-carriage"),
      inputWire: q("input-wire"), answerWire: q("answer-wire"),
      inputNeedle: q("input-needle"), answerNeedle: q("answer-needle"),
      inputBand: q("input-band"), answerBand: q("answer-band"),
      knobPointer: q("knob-pointer"), gaugeNeedle: q("gauge-needle"),
      inputCells: q("input-cells"), answerCells: q("answer-cells"),
    };
    const still = matchMedia("(prefers-reduced-motion: reduce)");
    const s: State = { model, length, phase: "input", pos: 0, ablation, knob: knobAngle(ablation), gauge: accuracyOf(model, ablation) };
    let target = 0, clock = 0, touring = true, idle = 0, frame = 0, last = 0, visible = false, cellsFor = -1;

    const draw = () => {
      if (cellsFor !== s.length) {
        const c = cellsOf(s.length);
        if (el.inputCells) el.inputCells.innerHTML = k.lineSvg(c.input.minor, { tone: "lo", free: "end" }) + k.lineSvg(c.input.major, { tone: "mid", free: "end" });
        if (el.answerCells) el.answerCells.innerHTML = k.lineSvg(c.answer.minor, { tone: "lo", free: "end" }) + k.lineSvg(c.answer.major, { tone: "mid", free: "end" });
        cellsFor = s.length;
      }
      const f = frameOf(s);
      el.inputCarriage?.setAttribute("transform", f.inputCarriage);
      el.answerCarriage?.setAttribute("transform", f.answerCarriage);
      el.inputWire?.setAttribute("d", f.inputWire);
      el.answerWire?.setAttribute("d", f.answerWire);
      el.inputNeedle?.setAttribute("d", f.inputNeedle);
      el.answerNeedle?.setAttribute("d", f.answerNeedle);
      el.answerNeedle?.setAttribute("data-tone", f.fit.coherent ? "lit" : "lo");
      el.inputBand?.setAttribute("d", f.inputBand);
      el.answerBand?.setAttribute("d", f.answerBand);
      el.knobPointer?.setAttribute("d", f.knobPointer);
      el.gaugeNeedle?.setAttribute("d", f.gaugeNeedle);
      const text = readoutOf(s, f);
      if (readoutRef.current && readoutRef.current.textContent !== text) readoutRef.current.textContent = text;
      stage.setAttribute("aria-valuetext", text);
      stage.setAttribute("aria-valuenow", String(Math.round((s.phase === "input" ? 0 : s.length) + s.pos)));
    };

    // the tour: input carriage across L cells, dwell, answer carriage across its rail, dwell, repeat
    const tourAt = (t: number, L: number) => {
      const run = (L - 1) * SECONDS_PER_TOKEN, cycle = 2 * (run + DWELL);
      const u = t % cycle;
      if (u < run) return { phase: "input" as const, pos: u / SECONDS_PER_TOKEN };
      if (u < run + DWELL) return { phase: "input" as const, pos: L - 1 };
      if (u < 2 * run + DWELL) return { phase: "answer" as const, pos: (u - run - DWELL) / SECONDS_PER_TOKEN };
      return { phase: "answer" as const, pos: L - 1 };
    };

    const tick = (now: number) => {
      const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
      last = now;
      const want = live.current;
      if (want.length !== s.length || want.model !== s.model) { s.length = want.length; s.model = want.model; s.pos = Math.min(s.pos, s.length - 1); target = Math.min(target, s.length - 1); }
      s.ablation = want.ablation;
      const calm = still.matches;
      if (touring && !calm) {
        clock += dt;
        const t = tourAt(clock, s.length);
        s.phase = t.phase; s.pos = t.pos; target = t.pos;
      } else {
        s.pos = calm ? target : s.pos + (target - s.pos) * (1 - Math.exp(-dt / 0.18));
      }
      const ease = calm ? 1 : 1 - Math.exp(-dt / 0.22);
      s.knob += (knobAngle(s.ablation) - s.knob) * ease;
      s.gauge += (accuracyOf(s.model, s.ablation) - s.gauge) * ease;
      draw();
      const settled = (!touring || calm) && Math.abs(target - s.pos) < 1e-3
        && Math.abs(knobAngle(s.ablation) - s.knob) < 1e-3 && Math.abs(accuracyOf(s.model, s.ablation) - s.gauge) < 1e-3;
      frame = visible && !settled ? requestAnimationFrame(tick) : 0;
    };
    const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };
    const hold = () => {
      touring = false;
      clearTimeout(idle);
      idle = window.setTimeout(() => { touring = !still.matches; run(); }, 5000);
      run();
    };
    nudge.current = (dir: number) => {
      hold();
      const L = s.length;
      // step through input positions, then answer positions, as one index 0 .. 2L-1
      let index = (s.phase === "input" ? 0 : L) + Math.round(target) + dir;
      index = Math.max(0, Math.min(2 * L - 1, index));
      const phase = index < L ? "input" : "answer";
      target = index < L ? index : index - L;
      if (phase !== s.phase) { s.phase = phase; s.pos = target; }   // the other carriage takes over
      run();
    };
    if (still.matches) { touring = false; s.phase = "input"; s.pos = target = 24 - 1; }

    const io = new IntersectionObserver((entries) => {
      visible = entries[entries.length - 1].isIntersecting;
      if (visible) run();
    }, { rootMargin: "120px 0px" });
    io.observe(stage);
    const poke = () => run();
    window.addEventListener("clock:changed", poke);
    draw();
    return () => { io.disconnect(); cancelAnimationFrame(frame); clearTimeout(idle); window.removeEventListener("clock:changed", poke); };
    // set up once; controls reach the loop through `live`
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // a control change has to wake a settled loop
  useEffect(() => { window.dispatchEvent(new Event("clock:changed")); }, [model, length, ablation]);

  const d = CLOCK_DATA[model];
  return (
    <figure className="iso clock-figure" data-theme={theme}>
      <div className="iso-plate">
        <div className="iso-plate-corners" aria-hidden="true">
          <span className="iso-plate-corner" data-corner="fig">Fig 3</span>
          <span className="iso-plate-corner" data-corner="title">The positional clock</span>
          <span className="iso-plate-corner" data-corner="hint">Arrow keys step the carriages</span>
          <span ref={readoutRef} className="iso-plate-corner" data-corner="readout" data-readout="">input m 0 · 0.00 rad (0%) · acc 1.00</span>
        </div>
        <div ref={stageRef} className="iso-stage" tabIndex={0} role="slider"
          aria-label="Positional clock instrument: arrow keys move the reading carriages through the input, then the answer"
          aria-valuemin={0} aria-valuemax={2 * length - 1} aria-valuenow={0}
          onKeyDown={(e) => {
            const dir = e.key === "ArrowRight" || e.key === "ArrowUp" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowDown" ? -1 : 0;
            if (!dir) return;
            e.preventDefault();
            nudge.current(dir);
          }}
          dangerouslySetInnerHTML={{ __html: CLOCK_SVG }} />
      </div>
      <div className="clock-controls">
        <Choice label="model" value={model} options={MODELS} onChange={setModel} />
        <Choice label="length" value={length} options={LENGTHS.map((L) => ({ key: L, label: String(L) }))} onChange={setLength} />
        <Choice label="console" value={ablation} options={ABLATIONS} onChange={setAblation} />
      </div>
      <figcaption className="clock-caption">
        Front rail: the input. Each input cell is one position, at a fixed pitch; the dial shows the measured
        clock angle there, and the heavy band on it is the angle the 16 trained cells cover. Past them the
        needle barely moves. Back rail: the answer. The rail is the same length at every evaluation length
        and its ticks get finer, and the dial&rsquo;s full sweep hardly changes: a fractional code. When the
        answer-side rotation is not coherent (fit R² below 0.75) the needle parks. Console: token accuracy at
        the training length with a 2-D plane of the residual stream removed at every block.{" "}
        {d.seeds} seeds; exact match at 1.5× training length for this model: {d.seq24.toFixed(2)}.
      </figcaption>
    </figure>
  );
}
