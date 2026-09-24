import json
import sys

FINDINGS = [
    ("none", "<b>Load balancing buys no measurable perplexity here.</b> Aux-loss (8.37), aux-free (8.40), the "
             "sigmoid router without balancing (8.37) and the baseline (8.42) all finish within 0.006 nats of each "
             "other. That is the same size as the seed-to-seed spread (up to 0.012 nats), so treat these four as a tie. It still holds "
             "at twice the tokens: after 3,000 steps aux-loss leads the baseline by 0.002 nats (ppl 7.16 vs 7.18). "
             "Beneath the tie, every balanced run is 0.01–0.06 nats worse on arXiv and roughly even elsewhere."),
    ("balance", "<b>But only load balancing prevents expert collapse.</b> With aux-loss or aux-free balancing, "
                "MaxVio<sub>global</sub> is 0.56 / 0.66, 0–2 of 768 experts are dead and ~61 of 64 experts per layer are "
                "effectively used. Without it: MaxVio 4.3–4.7, 102–134 dead experts, ~36 used. About one expert in six "
                "goes unused. Collapse is decided during warmup, in layers 2–3 and the last layer, and heals only partly with "
                "longer training (95 dead experts at 3,000 steps)."),
    ("balance", "<b>Aux-loss-free balancing gives the most specialized experts.</b> Domain NMI is 0.097 for aux-free "
                "against 0.058 for aux-loss and 0.086 for the baseline, the same order in both seeds, and aux-free also has "
                "the highest vocabulary specialization (0.514). Aux-loss balances hardest (MaxVio 0.56) but makes experts "
                "the least domain-specific. Follow-up runs show this comes from the aux loss's mechanism, not from how hard it "
                "balances: at matched balance (MaxVio ≈ 0.6) aux-free keeps NMI 0.100 against aux-loss's 0.058, and a very "
                "weak aux loss (0.001) specializes more than none at all."),
    ("router", "<b>Bilevel alternation and bandit + gradient cost about 0.04–0.05 nats</b> (ppl 8.76 and 8.81), "
               "clearly outside seed noise. My first explanation for bilevel, that the router gets too few updates, is refuted: "
               "giving the router more steps makes it much worse (ppl 9.87 at 25/25, 11.56 at 40/10). The cost tracks the updates "
               "the experts lose while frozen. Why bandit + gradient loses is still open: turning exploration off makes it worse "
               "(8.97), not better."),
    ("router", "<b>Pure bandit routing is far worse</b> (ppl 16.7 / 18.3). The REINFORCE gradient's cosine with the "
               "task gradient is only 0.02–0.03 throughout training, too little signal to learn a router from."),
    ("none", "<b>Routing quality only matters a little with top-8 of 64 experts.</b> A frozen random router reaches "
             "8.57, only 0.018 nats behind the trained baseline, and ahead of bilevel and both bandit variants. "
             "Differences between routing methods are therefore small at this scale."),
    ("router", "<b>LOLA opponent shaping is feasible to compute but harmful.</b> All three unbounded runs diverged "
               "to NaN (α = 3e-4 by step ~357, α = 1e-4 at steps 591 and ~495). With each token's correction capped, "
               "training is stable but ends at ppl 11.9 (+0.34 nats), still collapsed (MaxVio 4.1) and with the "
               "<i>lowest</i> domain specialization of any arm (NMI 0.052). <a href=\"#lola-h\">Details below.</a>"),
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


WHY = [
    {"claim": "Collapse is decided at initialization, in the layers whose routers see near-identical inputs",
     "verdict": "supported",
     "test": "Rebuilt each expert's load at every 250-step snapshot from the saturation probes, per layer (no new runs).",
     "prediction": "Most experts that end up dead are already dead in the first snapshot, and layer 1 (router input = token embeddings, which differ strongly between tokens) does not collapse.",
     "result": "In the baseline, 112–130 experts are dead at step 250 (still in warmup) versus 123–128 at the end, and 68–75% of the finally dead experts were already dead then. Layer 1 has 0 dead experts in every no-LB arm, while layers 2–3 lose 26–32 of their 64. At initialization every token's attention output is close to the same vector, so every router in layers 2+ picks the same few experts, which then get all the gradient."},
    {"claim": "The auxiliary loss lowers specialization by pushing toward uniform use",
     "verdict": "partly",
     "test": "Aux coefficient 0.001, 0.03 and 0.1 (one seed each) alongside the main sweep's 0 and 0.01.",
     "prediction": "Domain NMI falls monotonically as the coefficient rises.",
     "result": "Monotone from 0.001 upward: NMI 0.102 → 0.058 → 0.027 → 0.011, while perplexity worsens from 8.37 to 8.60 at 0.1. But 0.001 specializes <i>more</i> than no aux loss at all (0.102 vs 0.086): a little balancing rescues the collapsed layers, and only pressure beyond that erodes domain specialization. Vocabulary specialization moves the other way (0.494 → 0.525), so strong aux loss makes experts token-specific rather than domain-specific."},
    {"claim": "Aux-free specializes more because its balancing is gentler",
     "verdict": "refuted",
     "test": "Aux-free bias update rate 3e-4 and 3e-3 alongside the main sweep's 1e-3.",
     "prediction": "Faster bias updates balance harder and lower domain NMI.",
     "result": "Balance tightens (MaxVio<sub>global</sub> 0.90 → 0.66 → 0.62) but NMI stays flat (0.097, 0.097, 0.100). At matched balance the two mechanisms differ: at MaxVio ≈ 0.6 aux-loss has NMI 0.058 while aux-free has 0.100, and at MaxVio ≈ 1 both are ≈ 0.10. What costs specialization is how the aux loss balances, not how much: it adds a gradient to every token's router probabilities toward uniform, while the aux-free bias only shifts which experts are selected and leaves the probabilities, and their gradients, alone."},
    {"claim": "Bilevel loses perplexity because the router gets fewer updates, and its expert-only phases reduce collapse",
     "verdict": "refuted",
     "test": "Router/expert steps per 50-step cycle 25/25 and 40/10 alongside the main sweep's 10/40.",
     "prediction": "More router steps: better perplexity and more collapse.",
     "result": "The opposite for perplexity: 8.76 (10/40) → 9.87 (25/25) → 11.56 (40/10). The cost tracks the experts' lost updates (experts are frozen in router phases, so they train on 80%, 50% and 20% of steps), not the router's. Collapse is mixed: MaxVio<sub>global</sub> falls (3.15 → 2.91 → 2.38) while dead experts rise (50 → 55 → 101), so no clean conclusion there. As implemented, with both halves at the same learning rate, alternation mostly just withholds training from the experts."},
    {"claim": "Pure bandit routing fails because the REINFORCE signal barely points the right way",
     "verdict": "supported",
     "test": "Re-ran bandit + task gradient logging the cosine between the REINFORCE gradient and the task gradient on the router logits every 25 steps.",
     "prediction": "The cosine stays near zero.",
     "result": "0.02–0.03 for the whole run: above the ~0.001 expected for unrelated directions of this size, so there is some signal, but almost all of each REINFORCE update is orthogonal to the task gradient. A router trained on that alone (the pure bandit arm) learns little."},
    {"claim": "Bandit + task gradient loses because ε-exploration perturbs the forward pass, not because of the REINFORCE term",
     "verdict": "refuted",
     "test": "Bandit + task gradient with exploration switched off (otherwise identical).",
     "prediction": "Matches the no-LB baseline (ppl ≈ 8.42) instead of 8.79–8.83.",
     "result": "The opposite: without exploration the arm ends at ppl 8.97, worse than both the arm with exploration (8.77–8.83) and the baseline (8.42). Exploration seems to <i>help</i> this arm by sending some tokens to experts that would otherwise die. The critic is ruled out too, since its input is detached from the hidden state. COEFF0_TEXT"},
    {"claim": "Balancing buys no perplexity only because the ~36 live experts per layer suffice at 1.57B tokens",
     "verdict": "refuted",
     "test": "Baseline and aux-loss (0.01) retrained for 3,000 steps (3.1B tokens), one seed each.",
     "prediction": "Aux-loss pulls ahead of the baseline once the token budget grows.",
     "result": "At 3,000 steps aux-loss leads by only 0.002 nats (ppl 7.16 vs 7.18), less than its 0.006 lead at 1,500 steps, while the baseline still has 95 dead experts and MaxVio 4.0. Within a 2× budget, collapse does not start to cost perplexity. The collapsed baseline partly heals instead (124 → 95 dead experts), and aux-loss's arXiv cost persists (+0.021 nats)."},
    {"claim": "Balancing costs arXiv quality because it spreads a rare domain over too many experts",
     "verdict": "refuted",
     "test": "Per-domain routing concentration (effective number of experts serving each domain) on the existing runs.",
     "prediction": "The baseline routes arXiv more concentrated than other domains, relative to the balanced arms.",
     "result": "The baseline routes every domain about twice as concentrated (≈32 vs ≈57 effective experts), with no arXiv-specific difference, and aux-free concentrates arXiv almost as much as the baseline yet also loses on it. The arXiv cost itself is robust across runs: all seven balanced runs across both mechanisms and every strength are 0.01–0.06 nats worse on arXiv than both baseline seeds, while roughly even on web, Wikipedia and books. Its mechanism is open, and it may be specific to this arXiv source, which is lowercased and re-spaced; it should be re-checked on raw arXiv text."},
]

CAPPED = ("<b>A stabilized variant trains but does not help.</b> <span class=\"mono\">--lola-max-ratio 0.5</span> caps "
          "each token's correction at half the norm of that token's own task gradient and zeroes non-finite values. "
          "With α = 1e-4 both seeds completed, but the global gradient norm stayed at ~2, about 10× the other arms, "
          "and the models finished at val ppl 11.95 / 11.77, +0.34 nats against the baseline. Routing still collapsed "
          "(MaxVio<sub>global</sub> 4.1, 85 dead experts, fully collapsed layers at MaxVio 7.0) and was the least "
          "specialized of all arms (domain NMI 0.052, vocabulary specialization 0.352). Router-router LOLA pushes "
          "each selected row to anticipate its co-selected rows' updates. At this scale that disrupts the routing "
          "rather than coordinating it.")


def main(template, data_path, out, capped_text, coeff0_text):
    html = open(template).read()
    data = json.load(open(data_path))
    text_why = [dict(h, result=h["result"].replace("COEFF0_TEXT", coeff0_text)) for h in WHY]
    val = {r["arm"]: r for r in json.load(open("results/val.json"))["runs"] if r["complete"]}
    main_arms = json.load(open("results/compare.json"))["arms"]
    mm = lambda arm, key: main_arms[arm][key]["mean"]
    vv = lambda arm, key: val[arm][key] if key in val[arm] else val[arm]["spec_summary"][key]
    data["sweeps"] = [
        {"title": "Aux-loss coefficient", "xlabel": "coefficient",
         "note": "Hollow points are from the main sweep (2-seed mean), filled points from the follow-up runs (1 seed).",
         "panels": [
             {"metric": "Domain NMI", "digits": 2, "range": [0, 0.12], "points": [["0", mm("noLB", "domain_nmi"), True], ["0.001", vv("aux-0.001", "domain_nmi"), False], ["0.01", mm("aux", "domain_nmi"), True], ["0.03", vv("aux-0.03", "domain_nmi"), False], ["0.1", vv("aux-0.1", "domain_nmi"), False]]},
             {"metric": "MaxVio global (valid)", "digits": 1, "points": [["0", mm("noLB", "maxvio_global_valid"), True], ["0.001", vv("aux-0.001", "final_maxvio_global_valid"), False], ["0.01", mm("aux", "maxvio_global_valid"), True], ["0.03", vv("aux-0.03", "final_maxvio_global_valid"), False], ["0.1", vv("aux-0.1", "final_maxvio_global_valid"), False]]},
             {"metric": "Validation perplexity", "digits": 2, "points": [["0", mm("noLB", "final_valid_ppl"), True], ["0.001", vv("aux-0.001", "final_valid_ppl"), False], ["0.01", mm("aux", "final_valid_ppl"), True], ["0.03", vv("aux-0.03", "final_valid_ppl"), False], ["0.1", vv("aux-0.1", "final_valid_ppl"), False]]},
         ]},
        {"title": "Aux-free bias update rate", "xlabel": "bias rate",
         "note": "Balance tightens with the rate while domain specialization stays flat.",
         "panels": [
             {"metric": "Domain NMI (same scale as the aux-loss panel)", "digits": 2, "range": [0, 0.12], "points": [["3e-4", vv("auxfree-3e-4", "domain_nmi"), False], ["1e-3", mm("auxfree", "domain_nmi"), True], ["3e-3", vv("auxfree-3e-3", "domain_nmi"), False]]},
             {"metric": "MaxVio global (valid)", "digits": 2, "points": [["3e-4", vv("auxfree-3e-4", "final_maxvio_global_valid"), False], ["1e-3", mm("auxfree", "maxvio_global_valid"), True], ["3e-3", vv("auxfree-3e-3", "final_maxvio_global_valid"), False]]},
         ]},
        {"title": "Bilevel router / expert steps", "xlabel": "router/expert",
         "note": "Perplexity worsens as the experts' share of updates shrinks.",
         "panels": [
             {"metric": "Validation perplexity", "digits": 1, "points": [["10/40", mm("bilevel", "final_valid_ppl"), True], ["25/25", vv("bilevel-25x25", "final_valid_ppl"), False], ["40/10", vv("bilevel-40x10", "final_valid_ppl"), False]]},
             {"metric": "Dead experts / 768", "digits": 0, "points": [["10/40", mm("bilevel", "dead_experts_total"), True], ["25/25", vv("bilevel-25x25", "dead_experts_total"), False], ["40/10", vv("bilevel-40x10", "dead_experts_total"), False]]},
         ]},
    ]
    text = {
        "findings": [{"family": f, "html": h} for f, h in FINDINGS],
        "lola": LOLA.replace("CAPPED_TEXT", capped_text),
        "setup": SETUP,
        "why": text_why,
    }
    html = html.replace("/*DATA*/", json.dumps(data).replace("</", "<\\/"))
    html = html.replace("/*TEXT*/", json.dumps(text).replace("</", "<\\/"))
    open(out, "w").write(html)


if __name__ == "__main__":
    main(*sys.argv[1:4], CAPPED, sys.argv[4] if len(sys.argv) > 4 else "")
