import { AttentionStaircase } from "@/components/AttentionStaircase";
import { AlgebraTable, LengthGeneralization, WidthScan } from "@/components/Evidence";
import { DepthLadder } from "@/components/DepthLadder";
import { EngineProvider } from "@/components/EngineProvider";
import { FractionalClock } from "@/components/FractionalClock";
import { LiveDemo } from "@/components/LiveDemo";
import { NullTest } from "@/components/NullTest";
import { ClockInstrument } from "@/components/figures/clock/ClockInstrument";
import { Section } from "@/components/ui";

export default function Page() {
  return (
    <EngineProvider>
      <main className="mx-auto max-w-[1080px] px-6 pb-28">
        <header className="pt-16 pb-10">
          <p className="mb-2 text-[11px] font-medium uppercase tracking-[0.13em]"
            style={{ color: "var(--accent)" }}>
            interactive companion
          </p>
          <h1 className="mb-4 text-[40px] leading-[1.08] tracking-[-0.02em]"
            style={{ fontFamily: "var(--font-serif)" }}>
            Phase Without Position
          </h1>
          <p className="max-w-[64ch] text-[16.5px] leading-relaxed"
            style={{ color: "var(--text-secondary)", fontFamily: "var(--font-serif)" }}>
            Can complex-valued transformer weights infer the <code className="tnum">cos + i·sin</code>{" "}
            structure of RoPE, removing the need for an explicit positional module? No. The mechanism
            is falsified at three scales and again in the activations, and the one behavioral effect
            that survived turns out not to be about complex numbers either.
          </p>
          <p className="mt-4 max-w-[64ch] text-[14px] leading-relaxed"
            style={{ color: "var(--text-muted)" }}>
            Every number on this page that is not labelled as coming from the paper is computed in
            your browser, right now, by the trained checkpoints themselves. Nothing is pre-rendered.
            The engine is a from-scratch port of the training code, checked against PyTorch to a
            relative logit error of 2.4e-6 in fp32 and shipped at fp16, where the argmax of every
            logit is unchanged.
          </p>
        </header>

        <Section
          id="demo"
          kicker="the behavioral result"
          title="Train on 16 tokens, ask for 24"
          lede={
            <>
              Both models below reverse a sequence perfectly at the length they were trained on. Drag
              past 16 and they come apart. The real baseline here is parameter-matched, not smaller:
              3.02M against 3.09M. Click any input cell to change a symbol.
            </>
          }
        >
          <LiveDemo />
        </Section>

        <Section
          id="clock"
          kicker="the mechanism, section 9"
          title="The clock that seems to run slower on longer sequences"
          lede={
            <>
              If a NoPE model builds position at all, it must do so in the residual stream at run time.
              It does: average the residual stream over random content at each position, project on the
              top-two principal plane, and the trajectory turns monotonically. A fixed-rate code, which is
              what RoPE supplies, advances a fixed angle per token, so the total sweep grows with the
              sequence. Fit one line to the whole sequence here and the opposite happens: drag the length
              slider and the total sweep holds while the rate falls, which first read as a{" "}
              <em>fractional</em> code. That reading turned out to blend two different codes, one on each
              side of the separator; the next section takes them apart.
            </>
          }
        >
          <FractionalClock />
        </Section>

        <Section
          id="two-clocks"
          kicker="round three: the clock, split and ablated"
          title="Two codes in one clock, and the tied models depend on theirs"
          lede={
            <>
              Under a causal mask the state at input position <em>m</em> sees only tokens 0 to <em>m</em>, so
              it cannot know how long the sequence will be, and it does not: across 20 retrained checkpoints
              the input-side states are identical at every evaluation length, to sampling noise. Fitted
              separately in one fixed plane, the input side is a <em>saturating</em> absolute code that
              crowds every position past the training range together, and the answer side, where the
              input length is known, is roughly fractional, cleanly so only in the two-stream models.
              Removing the clock&rsquo;s plane breaks those models; removing a same-rank control plane
              barely touches them. In real models both removals hurt about equally. Five seeds per
              condition; these analyses were not pre-registered.
            </>
          }
        >
          <ClockInstrument />
        </Section>

        <Section
          id="null"
          kicker="the probability test"
          title="Check it against chance yourself"
          lede={
            <>
              The rotation could be an artifact of projecting anything onto two dimensions. The paper&rsquo;s
              null shuffles the position labels, which leaves the point cloud and its principal axes
              untouched and destroys only the ordering. Press re-run: 200 fresh shuffles, scored against
              the observed fit, computed here.
            </>
          }
        >
          <NullTest />
        </Section>

        <Section
          id="depth"
          kicker="where it lives"
          title="Every NoPE model builds the rotation, whatever its weight field"
          lede={
            <>
              The committed 14-run probe. All of them build the clock in the first two blocks; the
              embedding input correctly reads about zero. The conditions separate only at depth, and
              Section 9 shows that difference cannot be what produces the behavior: among the real+NoPE
              seeds, the one with no deep rotation has the <em>best</em> accuracy and the one with the
              strongest has the worst.
            </>
          }
        >
          <DepthLadder />
        </Section>

        <Section
          id="attention"
          kicker="the pre-registered prediction that failed backwards"
          title="The better retrieval head belongs to the worse model"
          lede={
            <>
              To emit answer token <em>j</em>, the model must attend back exactly 2<em>j</em>+1 positions.
              A head doing the task by direct retrieval would sit on that staircase at every index.
              Switch between the models: the parameter-matched real one, which scores 0.41, has the clean
              head. The complex one, which scores 0.86, has essentially none.
            </>
          }
        >
          <AttentionStaircase />
        </Section>

        <Section
          id="evidence"
          kicker="the committed evidence base"
          title="270 runs, and what survived them"
          lede={
            <>
              One effect survived round one: complex+NoPE roughly doubles exact-match length
              generalization on order reversal. Round two then showed it is not a property of ℂ and not
              out of reach of real width. What remains is a parameter-efficiency result on one task
              out of six.
            </>
          }
        >
          <div className="space-y-5">
            <LengthGeneralization />
            <div className="grid gap-5 lg:grid-cols-2">
              <AlgebraTable />
              <WidthScan />
            </div>
          </div>
        </Section>

        <footer className="border-t pt-8 text-[12.5px]" style={{ borderColor: "var(--rule)", color: "var(--text-muted)" }}>
          <p>
            Companion to <em>Phase Without Position</em>. Every figure is a projection of the committed
            metrics in <code>results/</code>, or a live computation by the checkpoints in{" "}
            <code>web/public/models/</code>. The browser engine is gated by{" "}
            <code>web/scripts/verify.ts</code>, which reproduces the paper&rsquo;s teacher-forced accuracies
            and its rotating-plane statistics before the site is allowed to build.
          </p>
        </footer>
      </main>
    </EngineProvider>
  );
}
