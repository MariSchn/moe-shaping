import json
import sys

FINDINGS = [
    ("none", "<b>Load balancing buys no measurable perplexity here.</b> Aux-loss (8.37), aux-free (8.40), the "
             "sigmoid router without balancing (8.37) and the baseline (8.42) all finish within 0.006 nats of each "
             "other. That is the same size as the seed-to-seed spread (up to 0.012 nats), so treat these four as a tie."),
    ("balance", "<b>But only load balancing prevents expert collapse.</b> With aux-loss or aux-free balancing, "
                "MaxVio<sub>global</sub> is 0.56 / 0.66, 0–2 of 768 experts are dead and ~61 of 64 experts per layer are "
                "effectively used. Without it: MaxVio 4.3–4.7, 102–134 dead experts, ~36 used. About one expert in six "
                "goes unused, and at 1.57B tokens the remaining ones are enough to reach the same perplexity."),
    ("balance", "<b>Aux-loss-free balancing gives the most specialized experts.</b> Domain NMI is 0.097 for aux-free "
                "against 0.058 for aux-loss and 0.086 for the baseline, the same order in both seeds, and aux-free also has "
                "the highest vocabulary specialization (0.514). Aux-loss balances hardest (MaxVio 0.56) but makes experts "
                "the least domain-specific. The auxiliary loss pushes toward uniform use in a way the bias correction does not."),
    ("router", "<b>Bilevel alternation and bandit + gradient cost about 0.04–0.05 nats</b> (ppl 8.76 and 8.81), "
               "clearly outside seed noise. Bilevel roughly halves the collapse (MaxVio 3.15, 50 dead experts) because "
               "experts keep training while the router is frozen, but the router's reduced share of updates hurts perplexity. "
               "Adding the bandit signal to the task gradient does not improve balance at all (MaxVio 4.5, 134 dead)."),
    ("router", "<b>Pure bandit routing is far worse</b> (ppl 16.7 / 18.3). The per-token REINFORCE credit is too "
               "weak to learn a router from, which matches the earlier small-scale study."),
    ("none", "<b>Routing quality only matters a little with top-8 of 64 experts.</b> A frozen random router reaches "
             "8.57, only 0.018 nats behind the trained baseline, and ahead of bilevel and both bandit variants. "
             "Differences between routing methods are therefore small at this scale."),
    ("router", "<b>LOLA opponent shaping did not train stably.</b> All three unbounded runs diverged to NaN "
               "(α = 3e-4 by step ~357, α = 1e-4 at steps 591 and ~495). A variant that caps each token's correction "
               "is described <a href=\"#lola-h\">below</a>."),
]

LOLA = """
<div class="stack">
<p>LOLA is scaled from the toy <span class="mono">apply_router_router_lola_shaping</span> (<span class="mono">lola_learner="all"</span>): every selected router row is corrected to anticipate the naive gradient step of the token's other selected rows. At this scale the correction has a closed form, with no second backward pass. It treats the loss as linear in each MoE layer's output, just as the toy detaches expert outputs, and was checked against a double-backward implementation of the toy formula (maximum error 1e-7). It costs no measurable step time.</p>
<p><b>The problem is scale, not compute.</b> The correction grows as α·‖x‖²·g². The router input satisfies ‖x‖² ≈ hidden size (1,024), compared with about 1 in the toy, so toy-sized α values are roughly 1,000× too strong. In 300-step calibration runs α = 1e-2 diverged by step 22 and 1e-3 by step ~260. α = 3e-4 was stable over 300 steps with the correction at ~10% of the task gradient, but it was already 0.8 nats worse than the baseline at step 200. In full runs α = 1e-4 still diverged in both seeds. Just before the NaN the correction was only ~2–3% of the task gradient, while the global gradient norm had sat 3–5× above the other arms for hundreds of steps and then spiked to ~5·10⁶. The shaped dynamics are unstable as a whole; the failure is not a single oversized correction.</p>
<p id="lola-capped">CAPPED_TEXT</p>
</div>
"""

SETUP = [
    ("Model", "GPT-style MoE, 12 layers, hidden size 1,024, 16 attention heads (4 KV groups), RoPE, SwiGLU. Every layer is MoE with 64 experts of FFN size 512 and top-8 routing. 1.34B parameters in total, ~0.28B active per token."),
    ("Data", "Nemotron-ClimbMix <span class=\"mono\">climbmix_small</span>, GPT-2 BPE, sequences of 4,096 tokens, 1M tokens per step, 1,500 steps (1.57B tokens)."),
    ("Optimization", "Adam (β = 0.9 / 0.95), LR 1e-3 with 150 warmup steps and cosine decay to 1e-4, weight decay 0.1, gradient clipping at 1.0, bf16 with fp32 master weights. Identical for every arm; no arm has its own router LR."),
    ("Arms", "Baseline: no load balancing (aux coefficient 0). Aux-loss: Switch-style loss, coefficient 0.01. Aux-free: DeepSeek-V3 expert bias, update rate 1e-3 (Megatron requires a sigmoid router for this, hence the sigmoid-no-LB control). Bilevel: repeating 50-step cycle of 10 router-only then 40 expert-only steps (non-MoE weights train every step). Bandit: REINFORCE with ε-greedy exploration (0.1→0.01) and a learned critic baseline, with the task gradient to the router blocked (pure) or kept (hybrid); settings reused from the earlier bandit study. Frozen: router LR 0."),
    ("Metrics", "MaxVio<sub>batch</sub> every step and MaxVio<sub>global</sub> on validation (Wang et al., 2024), both from the routing the router actually chose, including expert bias and bandit exploration. OLMoE specialization metrics after training on a 5-domain × 0.5M-token held-out set, with saturation snapshots every 250 steps. Validation perplexity on 10.5M held-out tokens."),
    ("Deviations", "<b>Dropless instead of capacity 1.0:</b> in smoke tests with capacity 1.0, every arm without load balancing dropped 80–86% of expert assignments, so the comparison would have measured token dropping rather than routing. <b>LR 1e-3</b> instead of the earlier studies' 2e-3, which was tuned on a 512-wide model. <b>~3h per run</b> instead of ~2h: 1,500 steps ran at 6–9 s each. <b>fp32 optimizer state</b> instead of the precision-aware optimizer, because resuming the latter from a checkpoint fails in Transformer Engine; an A/B run showed no speed difference."),
    ("Caveats", "Two seeds per arm (one for the frozen control), so differences under ~0.01 nats are not resolved. One training budget and one model size. Most arm hyper-parameters are defaults, not tuned (aux coefficient, bias rate, bilevel ratio, bandit settings), with LOLA's α calibrated only for stability. Several runs were preempted and resumed from checkpoints (resume reproduces the loss bit-for-bit); <span class=\"mono\">cmp-noLB-s43</span> was moved to another node at step 250 because the first node had slowed."),
]


def main(template, data_path, out, capped_text):
    html = open(template).read()
    data = json.load(open(data_path))
    text = {
        "findings": [{"family": f, "html": h} for f, h in FINDINGS],
        "lola": LOLA.replace("CAPPED_TEXT", capped_text),
        "setup": SETUP,
    }
    html = html.replace("/*DATA*/", json.dumps(data).replace("</", "<\\/"))
    html = html.replace("/*TEXT*/", json.dumps(text).replace("</", "<\\/"))
    open(out, "w").write(html)


if __name__ == "__main__":
    main(*sys.argv[1:4], sys.argv[4] if len(sys.argv) > 4 else "The capped runs are still training.")
