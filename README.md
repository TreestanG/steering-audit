# Auditing activation trajectories

**Can you tell that a language model is being steered, from the steered forward pass alone — without knowing the steering vector?**

Activation steering adds a vector to the residual stream. Mishra et al. (ICLR 2026) proved that the resulting state almost surely has no prompt preimage: no sequence of tokens produces it. They stopped at the proof. This repo turns that gap into a **detector** and measures the two numbers a detector actually needs — a false-positive rate on clean-but-weird inputs, and the steering strength at which detection fails.

Results below are over four models end to end — GPT-2 (124M), Qwen2.5-0.5B/1.5B/7B-Instruct — across 5 prompt categories and every hidden-state layer. Qwen2.5-3B has the inversion sweep only.

---

## Headline results

| | GPT-2 | Qwen-0.5B | Qwen-1.5B | Qwen-7B |
|---|---|---|---|---|
| Steering detected (fraction ≥ 0.02) | 100% | 100% | 100% | 100% |
| Injection layer localized | 15/15 | 15/15 | 15/15 | 15/15 |
| Steering vector recovered (median cos) | 0.99999 | 0.99995 | 0.99996 | 1.00000 |
| Exact inversion, **clean** prompts | 90.8% | 99.0% | 98.8% | 98.4% |
| Clean relative residual (median) | 6.3e-4 | 9.3e-4 | 8.7e-4 | 1.1e-3 |
| Steered relative residual (median) | 1.0e-1 | 1.0e-1 | 1.0e-1 | 1.0e-1 |

Steered and unsteered states are separated by **two orders of magnitude** in the same score, at every layer, in every model.

### 1. The detector does not fire on merely-unusual text

The obvious failure mode is a detector that flags anything out-of-distribution. It doesn't. Inverting clean prompts across five deliberately adversarial categories:

| category | GPT-2 | Qwen-0.5B | Qwen-1.5B | Qwen-7B |
|---|---|---|---|---|
| natural English | 48/52 | 100/100 | 116/116 | 111/116 |
| source code | 45/52 | 100/100 | 116/116 | 116/116 |
| base64 | 48/52 | 99/100 | 114/116 | 116/116 |
| non-English | 48/52 | 96/100 | 113/116 | 116/116 |
| weird-but-clean | 47/52 | 100/100 | 114/116 | 112/116 |

Base64 is about as far off the natural-language manifold as clean text gets, and it still inverts exactly. The score separates **unreachable** from **merely rare** — which is the property that makes it a usable alarm rather than an OOD flag.

![exact inversion by layer](results/Qwen_Qwen2.5-7B-Instruct_fp16/sipit/figures/layer_metrics_exact.png)

### 2. There is no window where steering works but hides

The question a red-teamer asks: can I steer weakly enough to evade detection while still changing behavior? Sweeping steering strength against both behavioral effect (KL from the clean distribution) and detectability:

![efficacy vs detectability](results/Qwen_Qwen2.5-7B-Instruct_fp16/figures/gap_vs_steering.png)

Detectability (right panel, blue) is already ~10× above the alarm floor at fraction 0.1 and rises linearly from there. Behavioral efficacy has barely started. The two curves put against each other, on Qwen-7B:

| steering fraction | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|
| **detected** | 79% | **100%** | 100% | 100% | 100% | 100% | 100% |
| sentiment flip rate | 0.0% | 0.3% | 2.0% | 5.4% | 12.4% | 32.1% | 59.6% |
| flip rate, random control | 0.0% | 0.0% | 0.3% | 1.9% | 3.9% | 11.3% | 19.4% |

Detection saturates at fraction 0.02, where steering flips 0.3% of sentiment predictions — i.e. does nothing. By the time steering is doing real work (fraction 0.5, 32% flips against an 11% random floor), the alarm has been at 100% for a 25× range of strengths. **There is no operating point that steers without tripping the detector**, at least for additive steering with a fixed vector — which is the caveat that matters (see limitations).

The knee sits at the same place in every model (GPT-2 58% → 100%, Qwen-0.5B 59% → 100%, Qwen-1.5B 61% → 100% across fraction 0.01 → 0.02), below the strength at which published steering work operates.

### 3. Detection localizes, and inverts

Beyond a binary alarm, the audit recovers *where* and *what*:

- **Which layer** was injected: 75/75 correct across all models, from the takeoff point of the per-layer residual profile.
- **What vector** was added: median cosine similarity ≥ 0.9999 to the true steering vector, norm ratio ≈ 1.000.

![steering audit](results/Qwen_Qwen2.5-7B-Instruct_fp16/steer/figures/audit.png)

The bottom-right panel is the one worth reading: the margin a steer spends is identical to a norm-matched random vector's until the final layer. **Detection cost is set by magnitude, not alignment** — you cannot hide by picking a better direction, only a smaller one.

---

## What this is measuring, precisely

Two independent signals, with different threat models:

**Recomputation audit** (white-box). Redo each layer transition from the weights; the mismatch is exactly the injected vector. This is an oracle — it catches any mid-stream additive tampering, and it also fires on the random control, by construction. It is ground truth for the experiments, not the contribution.

**SipIt inversion** (tokens + activations only). Greedily invert the residual stream back to a token sequence, left to right. Clean states invert exactly; steered states fail at the first token. This is the one that generalizes — it tests reachability, so it catches soft prompts too, and the position where the sweep first fails localizes *which tokens* were steered.

The naive baseline — train a linear probe to classify steered vs. unsteered activations — is circular and deliberately not used. A vector was added to the residual stream, so a probe finds it trivially; that detects the intervention, not the model's response to it.

### Baselines run

- Norm-matched random vector (in every audit, reported alongside)
- Steering applied at every layer, not a chosen one
- Held-out prompts across 5 categories
- Unsteered out-of-distribution inputs — the false-positive test above

## Limitations

- **Additive steering only.** The reachability theorem explicitly does not cover soft prompts (Mishra et al., App. D). A near-manifold adversarial steer that preserves behavior is unexplored, and is the most likely way this breaks.
- **Small models.** 124M–7B. Nothing here says the separation holds at 70B+.
- **The theorem assumes exact real arithmetic**; quantization breaks analyticity. Measured gaps are ~100× so it doesn't bite in practice, but the detector rests on the empirical gap, not the proof.
- **Gaps from a truncated vocabulary scan** are upper bounds, so the reported margins are optimistic. Figures say so on their face.

## Running it

Full pipeline for one model, from nothing to figures:

```bash
scripts/run_model.sh Qwen/Qwen2.5-0.5B-Instruct --full --dtype float16
```

Stages run in order and each is skipped when its output exists (`--force` to redo): activations → vocab table → SipIt layer sweep → sentiment direction → steering audit → strength sweep → recovery/localization → figures. Several models can be passed at once; they run sequentially to share one GPU. `--help` documents every flag.

The vocabulary activation table is the expensive artifact (vocab × layers × hidden — ~13 GB for Qwen-0.5B at fp32) and is never rebuilt unless forced.

```
src/     experiment + plotting scripts, one concern each
scripts/ pipeline drivers (run_model.sh is the entry point)
results/<model-slug>/  jsonl records, figures, and per-stage logs
```

## References

- Mishra et al., *Steered LLM Activations are Non-Surjective*, ICLR 2026 — the reachability result this builds on
- Lindsey (2025), Anthropic — introspective reporting of injected concepts
- Fonseca Rivera & Africa — fine-tuned injection detection; generalizes only to directions geometrically near training
- Macar et al. — localizes the introspection mechanism to early-layer evidence carriers
