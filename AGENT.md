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
2. No false-positive analysis. Nothing on OOD-but-clean inputs (code, base64, other languages). Their Fig 4 does show L2 ≈ 0 for natural activations regardless of how odd the text is, which is the argument that this family of score separates *unreachable* from *merely rare* — but the FPR itself is uncomputed. This is the number reviewers will ask for.
3. Tradeoff curve half-drawn. Fig 4 does sweep steering strength (λ) against L2 distance. What is missing is distance and behavioral effect on the same axis — the regime where steering still works but the gap has sunk into the noise. Unexamined.
4. Depth untouched. They measure at the injection layer only. Whether the anomaly relaxes downstream decides whether detection has a deadline, and would mechanically explain unreliable self-report (Lindsey, skeptic camp).
5. Adversarial version absent. No attempt at steering that preserves behavior while staying near-manifold. They explicitly concede the theorem does not cover soft prompts (App. D), so the escape hatch is acknowledged and unexplored. Strongest open ground; matches the framing below.
6. Scale and rigor. 10 prompts, 1B models, order-of-magnitude match rule. Replication at hundreds of prompts with real statistics and held-out steering methods is a contribution on its own, especially given the skeptic replication cited above.

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

Status
Early. Building the CAA steering loop from scratch first — raw PyTorch forward hooks before reaching for a library, to understand what the library does. No results yet.

Publication plan
Not deadline-driven. A believed first result with baselines run is the milestone; venues recur.
* NeurIPS / ICLR interpretability + safety workshops — right size for 2B-model work with honest negative results
* ACL/EMNLP via ARR — rolling monthly submission, decouples readiness from deadlines
* BlackboxNLP — colocates with EMNLP
* ICLR 2027 main track — abstract Sep 18, full paper Sep 25 2026 AOE (abstract deadline is binding; no authors addable after it)
Interim: LessWrong / Alignment Forum writeup of early results. Cheap, gets feedback from people who know this literature, timestamps the idea. Several papers above started this way.
Field moves fast — roughly six months separated the papers listed above. Pick the version of this that stays interesting if someone scoops the framing.
