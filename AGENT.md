Project
Detecting activation steering from trajectory mismatch.
Can you tell that a language model is being steered, using only the steered forward pass — without knowing the steering vector?

Core claim being tested
Activation steering pushes the residual stream off the manifold of states reachable from real prompts. That displacement should be detectable as an inconsistency in the model's own layer-to-layer dynamics.

Hard constraint: the detector must be computable from the steered forward pass alone. No access to the steering vector, no clean-run counterfactual, no knowledge of which layer was hit. If a design violates this, it is measuring the wrong thing.

Why the obvious version doesn't work
The naive approach — train a linear probe on activations to classify steered vs. unsteered — is circular. A vector was added to the residual stream, so of course a probe can find it. That detects the intervention, not the model's response to it. Rejected.

Prior work (know this before proposing anything)
* Mishra et al., "Steered LLM Activations are Non-Surjective" (ICLR 2026, arXiv 2604.09839). Proves steered states almost surely have no prompt preimage. Empirically tried SipIt, many-shot ICL, prefix tuning, GEPA to find prompts reproducing steered activations — none succeeded. This is the theoretical basis for the project.
* Lindsey (2025), Anthropic. Inject a concept vector, ask the model if it detects an injected thought. Sometimes works, low reliability. Started the line.
* Fonseca Rivera & Africa. Fine-tuned Qwen 2.5 32B to report injections: 95.5% detection, 71.2% concept ID on held-out concepts. Detection did not produce resistance. Their detector generalized to unseen steering methods only when directions had high cosine similarity to training — i.e. it learned geometry, not generic anomaly. This is the specific gap this project targets.
* Macar et al. Localized the mechanism: early-layer "evidence carrier" features suppress later-layer "gate" features. Capability comes from post-training (DPO elicits it, SFT doesn't).
* Skeptic camp. One replication found the effect in small models was largely noise affecting all yes/no questions equally. Lederman & Mahowald argue detection is content-agnostic — models notice an anomaly but guess high-frequency concepts rather than identifying anything. Read these first.

What is actually open (verified against the full paper + appendices, Aug 2026)
Mishra et al.'s setup is smaller than its reputation: 3 models (Llama-3.2-1B, Qwen-2.5-0.5B, Gemma-3-1B), 10 prompts per steering type, refusal + evil-persona vectors, steering applied at all token positions (not last-token-only). SipIt fails at the first token on steered runs; recovers clean prompts exactly. Match rule was "order-of-magnitude L2 gap to the runner-up token," not exact equality.
1. No detector. They prove the gap exists but never build an alarm from it — no threshold, no ROC, no operating point. This is the whole distance between "interesting fact" and "usable tool."
2. No false-positive analysis. [STILL OPEN — now the main gap; see Status.] Nothing on OOD-but-clean inputs (code, base64, other languages). Their Fig 4 does show L2 ≈ 0 for natural activations regardless of how odd the text is, which is the argument that this family of score separates *unreachable* from *merely rare* — but the FPR itself is uncomputed. This is the number reviewers will ask for.
3. Tradeoff curve half-drawn. Fig 4 does sweep steering strength (λ) against L2 distance. What is missing is distance and behavioral effect on the same axis — the regime where steering still works but the gap has sunk into the noise. Unexamined. [ANSWERED — see finding 1. Distance and behavioral effect now sit on one axis across 8 runs; that regime does not exist for additive steering. Detection saturates at 100% before the first behavioral change in every model.]
4. Depth untouched. They measure at the injection layer only. Whether the anomaly relaxes downstream decides whether detection has a deadline, and would mechanically explain unreliable self-report (Lindsey, skeptic camp).
5. Adversarial version absent. No attempt at steering that preserves behavior while staying near-manifold. They explicitly concede the theorem does not cover soft prompts (App. D), so the escape hatch is acknowledged and unexplored. Strongest open ground; matches the framing below.
6. Scale and rigor. 10 prompts, 1B models, order-of-magnitude match rule. Replication at hundreds of prompts with real statistics and held-out steering methods is a contribution on its own, especially given the skeptic replication cited above. [PARTLY DONE — 6 models to 7B across 3 architecture families, exact-equality match rule, norm-matched random control on every run, 8-point strength sweep. Still single steering type (CAA sentiment); held-out steering methods untouched.]

Do not race them on epsilon-closeness. Their stated future work is "analyze potential epsilon-closeness of steered activations to natural prompts" (plus quantized activation spaces). Own detection + adversary instead; they show no interest in either.
Theory footnote: the proof assumes exact real arithmetic. The authors concede quantization breaks analyticity. Measured gaps are orders of magnitude, so this does not bite in practice — but the detector rests on the empirical gap, not the theorem.

Two oracles available under full access (both trivial, both useful as ground truth, neither is the paper)
* Recomputation audit — redo each layer transition from the weights; the mismatch is exactly the steering vector. Catches mid-stream additive tampering, silent on soft prompts. Keep as a plumbing regression test, not an experiment.
* SipIt inversion — tests reachability from tokens. Catches soft prompts too. Where the left-to-right sweep first fails localizes *which positions* were steered; the audit localizes *which layer*.

Stronger framing if the base result holds
Invert it adversarially: given that mismatch is detectable, can you construct steering vectors that stay on-manifold and evade detection? Is there a steering-strength vs. detectability tradeoff curve?
This is the version with a claim someone would argue with, and it bears directly on whether steering-based safety evaluations can be trusted. The non-subjectivity paper was accepted on a reframing plus a negative result, not a new method — that's the bar for a main-track contribution here.

Required baselines
Run these before believing any result:
* Random vector, matched norm
* Steering applied at a wrong/irrelevant layer
* Held-out prompts not used for tuning
* Unsteered but out-of-distribution inputs (the FPR test above)

Results (8 runs: gpt2, Qwen-2.5 0.5B/1.5B/3B/7B, Pythia-1.4B; 6 at fp16, plus fp32 controls on Pythia-1.4B and Qwen-7B)
Setup: CAA sentiment vector, injected last-token at every layer in turn, swept over fraction = ||delta|| / mean state norm in {0.01 .. 2}. Every number below has a norm-matched random-direction control run alongside it. Detection = SipIt residual exceeding rel_tol at the injected layer; efficacy = argmax flip rate and KL from clean on the held-out sentiment probe.

Findings 1-3 are dtype-invariant and are the scientific claims. Findings 4-5 only appear at fp16 and are the operating characteristics under the precision people actually deploy at — the realistic threat model, since an attacker with fp32 budget would not be using naive additive CAA in the first place.

1. There is no evasion window. This answers open item 3. In every model, at both dtypes, detection is already 100% at the smallest fraction that produces any behavioral change at all. First fraction with a nonzero flip rate is 0.02-0.1 depending on model; detection at that fraction is 100% in all 8 runs. Fraction 0.01 has a flip rate of exactly 0% everywhere and KL indistinguishable from the random control, so the imperfect detection at 0.01 is imperfect detection of an intervention that does nothing. The regime "steering still works but the gap has sunk into the noise" does not exist for additive steering. Corollary: do not sweep below 0.01 — there is nothing there to find.
   The upper end of the window is scale-dependent, and the comparison has to be paired by layer (see caveats). Paired, steering beats norm-matched noise across the entire range, and the margin widens with size: at fraction 0.5 Qwen-7B flips 68% steered vs 4% random at the same layer, and even at 2.0 it is 100% vs 24%. Small models do collapse into pure lesioning — gpt2 and Qwen-0.5B both reach 100% steered and 100% random by fraction 1.0. So larger models are simultaneously more robust to random perturbation and no less susceptible to steering, which widens the usable steering window with scale rather than narrowing it.

2. margin_spent falls monotonically with scale, and varies by architecture at matched size. margin_spent = residual/gap, scale-normalized. At fraction 0.1, within Qwen: 0.363 (0.5B) -> 0.309 (1.5B) -> 0.280 (3B) -> 0.213 (7B). Across families at matched size, Pythia-1.4B spends 0.379 vs Qwen-1.5B's 0.309, ~23% more. gpt2 is highest at 0.485. Bigger models absorb the same relative perturbation with proportionally less of their margin. Verified dtype-invariant: agrees within 0.78% at all 8 fractions in both fp16/fp32 pairs, so this is geometry, not numerics.

3. Steering is geometrically indistinguishable from norm-matched random noise — except in gpt2. steer/rand margin_spent ratio at fraction 0.1 is within 1.5% of 1.000x for all five RMSNorm+rotary models. gpt2 is the lone outlier at 1.367x (decaying to 1.097x at fraction 1.0). gpt2 is also the only model here with learned absolute position embeddings, LayerNorm rather than RMSNorm, and documented residual-stream outlier dimensions — an anisotropic space where a real steering direction costs more margin than an isotropic random one. Note this is a claim about hidden-state geometry only; behaviorally steering does beat random by 1.3-5.7x in KL. The detector cannot tell the two apart even though the model's output can.
   Localization is 100% across all 8 runs: the recomputation audit recovers the exact injected layer every time, on every prompt. Deviation persists rather than relaxing downstream — the model does not self-correct.

4. [fp16] The detector's sensitivity floor is set by inference precision, not by geometry. det@0.01 is 42.5-78.6% at fp16 (Pythia 42.5%, gpt2 58.3%, Qwen-0.5B 59.2%, Qwen-3B 55.6%, Qwen-1.5B 61.4%, Qwen-7B 78.6%) and exactly 100% at fp32 in both controlled pairs. Mechanism: fp32 lowers the clean-run noise floor ~250-500x (Pythia 1.32e-03 -> 5.83e-06, Qwen-7B 1.71e-03 -> 3.32e-06) while leaving the steering signal untouched (peak rel deviation 0.1287 -> 0.1285, 0.1111 -> 0.1112). Same signal, quieter ruler. Consequence: the cross-model spread in det@0.01 is fp16 noise accumulation, NOT an architecture effect — do not tell an architecture story with this column. But per finding 1 this costs a real defender nothing, because 0.01 is behaviorally inert anyway.

5. [fp16] SipIt has a deep-layer blind spot at fp16 that vanishes at fp32. Exact-recovery misses at fp16 cluster in the last ~10-15% of depth: Qwen-0.5B 5/500 (onset 92% depth), Qwen-1.5B 7/580 (96%), Qwen-3B 10/740 (86%), Qwen-7B 9/580 (61%), Pythia-1.4B 31/500 (88%), gpt2 4/260 (100%). Both fp32 controls: zero misses at every layer. So the depth-onset "trend" across scale is rounding error crossing rel_tol, not a reachability boundary. Practical form: a defender running SipIt on fp16 inference has a blind spot in the final layers and should audit at an earlier layer or in fp32.

Caveats and known issues
* Only Pythia-1.4B and Qwen-7B have fp32 controls. The other four runs are fp16-only, so their det@0.01 and sipit-miss columns are precision measurements and should not be read as geometry.
* Flip-rate comparisons must be paired by layer. Taking max-over-layers separately for the steered and random arms inflates the random arm, because random's best layer is almost always the final one — where any large perturbation wrecks the output but steering is too late to propagate — while steering's best is mid-to-late. Paired at Qwen-1.5B fraction 0.2: steering wins 18 layers, ties 8, loses 2. Unpaired the same data reads as random winning 44% to 32%. An earlier draft of these notes recorded Qwen-1.5B as behaviorally anomalous on exactly that mistake; it is not anomalous. The margin_spent steer/rand ratios in finding 3 are unaffected — steer and rand there are computed on the same prompt at the same layer by construction.
* gpt2's layer-0 SipIt rows were wrong until the wpe fix (candidate_states omitted the learned position embedding, so every layer-0 candidate was off by exactly ||wpe[pos]||, 0/20 exact). Fixed and re-run; now 20/20. Rotary models were never affected. Any layer-0 number from before that fix is invalid.
* Pythia at fp32 needs AAT_KV_BUDGET lowered (~1.5e9) on a 24 GB machine. It is MHA, not GQA, so its KV cache per token is 3.4x a Qwen-7B's; the default 6 GB budget counts only the cache and ignores weights plus the deepcopy transient, giving a 25 GB peak.

Status
Detection side is done and the tradeoff curve is closed (item 3). Open item 2 is now the gap: 100% detection is quoted with no denominator — there is no false-positive rate on clean-but-OOD input yet, though the clean arm of every audit row and the existing sipit categories (code, base64, weird_clean, natural_en) may already contain enough to compute it without a new run. Item 5, the adversarial/on-manifold construction, is untouched and remains the strongest ground.

Publication plan
Not deadline-driven. A believed first result with baselines run is the milestone; venues recur.
* NeurIPS / ICLR interpretability + safety workshops — right size for 2B-model work with honest negative results
* ACL/EMNLP via ARR — rolling monthly submission, decouples readiness from deadlines
* BlackboxNLP — colocates with EMNLP
* ICLR 2027 main track — abstract Sep 18, full paper Sep 25 2026 AOE (abstract deadline is binding; no authors addable after it)
Interim: LessWrong / Alignment Forum writeup of early results. Cheap, gets feedback from people who know this literature, timestamps the idea. Several papers above started this way.
Field moves fast — roughly six months separated the papers listed above. Pick the version of this that stays interesting if someone scoops the framing.
