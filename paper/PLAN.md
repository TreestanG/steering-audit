# Paper plan

Built from Neel Nanda's "Highly opinionated advice on how to write ML papers" and Nicholas Carlini's "How to win a best paper award", applied to what is actually on disk in this repo as of 8 Sep 2026. Both pieces agree on the spine: one idea, evidence a skeptic accepts, figures that stand alone, limitations stated before a reviewer states them. Everything below is that spine filled in with this repo's material.

---

## 0. The one sentence

Carlini: exactly one core contribution, written down before the paper, and every figure and paragraph serves it. Nanda: one to three claims that fit one theme, each with the evidence a skeptic would need.

**The sentence:**

> Activation steering can be detected from a sparse log of layers, one is enough, without the vector, the prompt, or adjacent layers, by inverting each logged state back to tokens; holding one layer quiet leaves a larger deviation at another; on five models the perturbation that first changes behaviour is far above the alarm, and no behaviour-changing perturbation went undetected under the single-layer read, the multi-layer read, or the full-log recomputation check.

**The three claims that hang off it** (in Nanda's terms: one systematic claim, one hedged negative, one methodological):

1. **Detector (systematic).** A per-position z-score of the SipIt inversion residual at one logged layer, calibrated per (layer, template role) on a clean bank, read at a single late layer with a 5 percent per-layer threshold, catches every PGD attack from budget 0.02 upward on all five models and the legal budget on three, where recomputation from the same log reads its own floor. The signal grows with depth under the constrained attack, and the mechanism is measured: holding the injection layer at the budget leaves a deviation 1.5x (gpt2) to 3.0x (Qwen-7B) larger downstream, 11x on gemma, peaking at or next to the last layer on the larger models; holding every layer at the budget shrinks the injection delta by the same factor. Evidence: RESULTS.md "One logged layer" tables and the constraint-scope table, fig 7 with the single-layer line, fig 3 as the offset curve. The held-out FPR of the single-layer rule is section 2, item 1.
2. **The window is empty (hedged negative).** No attacked prompt that went judge-negative to judge-positive against its own paired clean arm was undetected, and it holds three ways: under the run rule, under the last-layer read (every cell with a gain is fully flagged at the last layer), and under recomputation with its lower floor. That makes it a property of the perturbations, not of a detector. Evidence: transitions table with McNemar, fig 6, fig 8. Stated scope: "using the attacks and settings tested", never "no evasion exists". Fig 8's detector-aware attacker held every layer from the injection onward under the alarm; the attacker who holds only the read layer quiet, and is judged on decoded text, is section 2, item 4, and claim 2 is not stated for a single-layer read until it is run.
3. **Calibration is the result, not a detail (methodological).** The nominal 5 percent FPR is in-sample; the shipped statistic is 8x heavier-tailed than Gaussian; the top-5 read-out misses concentrated attacks entirely while top-1 cannot price spread ones; and a density on the raw state at the same layer, given the position and a matched reference, sits at the clean rate to budget 0.12. Calibration of the *inversion residual* is the useful thing, not calibration. Evidence: fig 4, fig 5, fig 10, held-out table, RESULTS.md "A raw-state density baseline".

**Naming the detector.** The detector is introduced as its own architecture, built around SipIt, not as SipIt. SipIt is the inverter and Mishra et al. supply the theory. What is new here: full inversion of one logged layer at every position; the log-residual z-scored per (layer, role) with a top-1 read-out (fig 4, 5, 10 were all measured at one layer); three-state verdicts, which matter more with one layer because there is nothing to fall back to when it is unscorable; the held-out calibration protocol; and the threat model, no vector, no prompt, no injection layer, no adjacent layers. Not part of the detector: the run rule over consecutive layers (it lets a two-layer tail through, loses to a lone late layer on Qwen-7B and gemma, and needs adjacent layers a sparse log does not have), the stride scan and catch-all sigma, and the word "trajectory". Reading several layers is not dead, the run rule is: the any-layer rule over whatever set is logged (section 2 item 5) is the multi-layer detector, and what several logged layers buy is stated from the constraint-scope table, the attacker must silence each of them and silencing one moves the deviation to another. Localization is what adjacent layers buy on top. Working name: **[NAME]**, a "reachability audit"; the unit of audit is one state and the question asked of it is reachability. The intro carries one delineating sentence: "We build [NAME] around SipIt inversion: SipIt answers whether one state has a prompt preimage; [NAME] turns that answer into a calibrated verdict with a measured false-positive rate, from one logged layer." After that sentence the paper never says "SipIt detects". Do not describe the inversion step as improved; the exact-equality match rule is a cost (A3), not a gain.

If a fourth claim keeps trying to get in (the hop attack, the CAA sweep, the judge inconsistency), it goes to the appendix or the companion post. Carlini's test: if the title is hard to write, the paper is doing too much.

**The "so what"** (Carlini's conclusion test: if the best-case conclusion is "numbers went up", kill it). The so-what here is that steering-based interventions, whether used for safety evaluation, for backdoors, or for silent behavioural modification in a deployment, leave a signature an auditor can read from one layer's activations without knowing what was done. The strength-vs-detectability curve has no middle. That is a claim someone would argue with, and it is the version that stays interesting if someone else publishes a detector first.

---

## 1. The observation model

Carlini: de-risk the riskiest sub-problem first. Nanda: optimise the boring baselines as hard as the method.

The riskiest sub-problem was the recomputation check: given every layer of a single-layer injection, running block j on the logged h_j and comparing with the logged h_{j+1} exposes the injection directly and cheaply. It has been run, with a linear probe beside it, on every ladder (RESULTS.md, "Two baselines on the same rows"). With the full trajectory recompute is the reference: its floor is the fp16 kernel discrepancy, 5e-4 to 9e-4 relative, ten times under the inversion tolerance, and it flags every attacked prompt whose delta clears that, 15/15 on Llama-3.2-1B and Qwen-7B at the legal budget where the detector reads 0/15 and 1/15. gemma's `all`-constrained delta sits under the floor and neither method sees it. The probe is at chance at the legal budget and fires on the random arm at the clean rate: it learns the direction it was shown.

Recompute fires only on the pair of layers that straddles the injection; every other pair is consistent by construction, because the later blocks really ran on the perturbed state. From a log with no layer before the injection it reads its own floor on every attacked prompt (RESULTS.md, "One logged layer: inversion against recomputation"). Inversion read at a single layer, with a 5 percent per-layer threshold checked on the held-out clean arm, catches 0.02 from the last layer on all five models and 0.0085 on three. The signal grows with depth; on Qwen-7B and gemma a lone late layer beats the shipped run rule at matched FPR. Llama at 0.0085 is invisible to inversion from every layer. The cheap competitor at a single layer, a density check on the raw state (RESULTS.md, "A raw-state density baseline"), is at the clean rate to budget 0.12 with a matched leave-one-out reference and flags rare clean text wholesale with the bank.

So the observation model is a ladder of what the auditor holds, with a method at each rung:

| auditor holds | method | fires at | floor |
|---|---|---|---|
| prompt, weights, any layer | clean counterfactual | every layer after the injection | fp16 kernel |
| weights, layers on both sides of the injection | recomputation | the one straddling pair | fp16 kernel |
| weights, any set of layers at or after the injection, not adjacent, not known to the attacker | inversion (this paper) | each logged layer, growing with depth under the constrained attack | inversion tolerance |

The method claim lives on the third rung, which is the realistic monitoring log: a few layers, not adjacent, usually late, and not known to the attacker. One is enough against the attacks run so far. The empty-window claim is method-independent and stronger under recompute's lower floor.

**What several logged layers buy, measured.** The fp32 sentiment ladders were run under two constraint scopes (RESULTS.md, the constraint-scope table). Holding the injection layer at the budget leaves a deviation 1.5x (gpt2), 1.8x (Qwen-0.5B), 2.0x (1.5B), 2.2x (3B), 3.0x (7B) and 11x (gemma) larger downstream, peaking at or next to the last layer on the larger models; that era's tolerance check read 0/25 at the injection layer and 25/25 at any layer on every model. Holding every layer at the budget shrinks the injection delta by the same factor, which is the backoff under which the jailbreak ladders measured no behavioural change at the legal budget. The random arm does not amplify. So an attacker who keeps the injection layer quiet is easier to read late, not harder, and an attacker who keeps one late layer quiet pays the `all` constraint's full tax on the models where the deviation peaks there. The attacker this does not cover is the one who sculpts the profile: quiet at the read layer, large elsewhere, at a behaviour-changing budget. That is section 2 item 4, and it is why the read set is stated as unknown to the attacker until it is run. Two things the paper does not claim: an "untrusted weights" edge, since both methods run the auditor's weights and inversion accumulates weight drift over more blocks than a one-block recompute; and soft prompts as a separate edge, since with layer 0 logged a table lookup catches them, and only the mid-layer-log version is inversion's, which is unmeasured. The objection the ladder invites, that an attacker who knows which layer is logged injects after it, is answered by section 2 item 7, not by assertion.

---

## 2. What is missing, ranked by what breaks if it stays missing

Nanda's test for every element: "what breaks if I cut this?" Turned around, for every gap: "what claim falls if this stays open?"

**Blocks the paper**

1. **The single-layer rule, pre-declared and held out.** RESULTS.md's best-offset column is chosen after the fact. Declare the offset (a fixed distance from the end, not the last layer itself, which is unscorable on Qwen under the deployment cut), then measure its FPR with the held-out protocol per layer. Claim 1's number is this.
2. **Red-team P1 findings** in `audits/redteam-extensive-2026-09-08.md`: generation and detection can be joined across different attacks, and judge parse failures become negative labels. Fix, regenerate RESULTS.md, confirm the byte-match check still passes. A paper cannot rest on a join whose identity is unverified.
3. **Deployment stopping rule, end to end, on one model.** The evaluator/deployment split is disclosed but not benchmarked. Refit the calibration under the `miss` rule on Qwen-0.5B (the n=50 model, whose bank has 0 rows past a miss) and rerun one ladder at the declared offset. This turns the largest limitation into a table.
4. **The sculpting attacker: quiet at the read layer, free elsewhere.** Fig 8's attacker held every layer from the injection onward under the alarm and did nothing. An attacker who knows the read layer has an easier problem, and the constraint-scope table says silencing one layer moves the deviation to another rather than removing it, so the question is whether a profile can be sculpted to be quiet at the read layer while changing behaviour. Run the detector penalty restricted to the read set (`DetectorSurrogate` spans every layer from the injection; it needs a `--penalty_layers` option), at budgets 0.12 and 0.3 on Qwen-0.5B, decoded and judged, because the K/V at the layers between the injection and the read carry the perturbation into generation whether or not the read layer is quiet. Then the same with two and three read layers. Blocks claim 2 for the single-layer read.
5. **The any-layer combination rule**, for logs with several post-injection layers: any layer over its own threshold, sigma tuned to a 5 percent union FPR. A re-read of saved rows. Expected to dominate the run rule; if it does, the run rule goes to the appendix as history.

**Weakens the paper**

6. **n=15 on four of five ladders.** A 15/15 cell has a 78 percent lower bound. Topping up to n=50 on at least Qwen-7B and gemma (the two models where the legal budget is a turn-on point) moves the lower bound to 93 percent and gives the transition tables enough discordant pairs for McNemar to say anything.
7. **A late injection layer** (about 0.9 of depth), decoded and judged, with the aware attacker on it. This answers the objection in section 1: that steering injected after a late logged layer does not change behaviour, so the attacker cannot go where the log cannot see. Measured, not asserted.
8. **`n_rand = 1` on every behaviour run.** The random control has no error bar. Three draws, as the sentiment sweep did.
9. **The density baseline on the four CUDA ladders.** Run on Qwen-0.5B only; the other ladders' deltas are on the CUDA boxes.

**Would be nice, and each is one appendix paragraph if not done**

10. The gemma sentiment "crack" (2/50 argmax flips, CI to 13.7 percent). About 200 prompts. Either close it or state it exactly as RESULTS.md does now.
11. The Qwen-3B pre-registered ladder. The spread rule it tests does not partition the five models, so the prediction is already known not to hold. Report the pre-registration and its failure in one paragraph; it is a credibility asset, not a result.
12. Rescoring the August CAA sweep with the real detector, so the CAA and PGD results are on one scale. Otherwise the CAA sweep stays in the appendix with its own check clearly labelled, as RESULTS.md does.

Carlini's "maximal" test: the reader should not finish and feel something essential is missing. Items 1 to 5 are that something. Items 6 to 9 are what a careful reviewer will ask for. Everything past that is future work.

---

## 3. Evidence map: claim, figure, table, source file

Every number in the paper should trace to one row of this table, which traces to the provenance manifest. `figures/paper/CAPTIONS.md` has a target claim per figure, which is the right habit; the paper's captions are those target claims rewritten as the one-sentence takeaway Carlini asks for.

| Claim | Hero figure | Supporting table | Source |
|---|---|---|---|
| Recompute is blind without a pre-injection layer; one late layer catches 0.02 on all five, 0.0085 on three | fig 3, redrawn: detection against layer offset, five models, recompute as a flat line at its floor | single-layer summary and per-offset tables | `single-layer-2026-09-08.json`, `baselines/*.json` |
| Silencing one layer moves the deviation, 1.5x to 11x, growing with scale | fig 3 panel (b): deviation by layer under `injection` and `all` scope | constraint-scope table | `*_fp32/pgd/pgd_sentiment_b0.0085.jsonl` |
| Detection turns on at 0.02 on five models | fig 7 (five models, one test; single-layer line beside the run rule, recompute's full-log count as a marker) | ladder table, Clopper-Pearson | `tpr_summary_*` per model, `single-layer-2026-09-08.json` |
| Calibration FPR held out | (table only) | held-out FPR, 3-way vs 2-way; per-layer once item 1 is done | `detector_holdout_*` |
| The window is empty | fig 6 (Qwen-0.5B ladder, ASR beside detection, single-layer line added) | transitions with McNemar and gate | `n50-transitions-2026-09-08.json` |
| Detector-aware attacker has no middle | fig 8 (last-layer read on the same rows; replaced by item 4's run) | (inline numbers) | `aware_*` directories |
| Top-5 misses concentrated, top-1 misses spread | fig 4, fig 5 | (inline) | gpt2 L8 cells |
| The statistic is heavy-tailed | fig 10 | (inline) | sweep B clean rows |
| A raw-state density at one layer is at the clean rate to 0.12 | (table only) | density baseline table | `mahalanobis-qwen05b-2026-09-08.json` |
| Judge is measured, and inconsistent | fig 11 | judge inconsistency table | `_gen_judge_repeats.json` |
| Deployment rule changes verdicts, not flagged counts | (table) | evaluator vs deployment, both sections | `deployment-stop-2026-09-08.json`, `single-layer-deploy-2026-09-08.json` |
| Localization is good where detection saturates | (table, appendix) | localization table | `localization-*.json` |
| Score separates unreachable from rare | fig 1 | (appendix) | sweep B, five text categories |

Figures to cut: fig 2 (CAA detection vs efficacy uses the older check, and its story is told by fig 7 with the real detector), fig 9 (metric disagreement is a judge-methodology point; one sentence plus fig 11 covers it). Fig 3's norm panel goes to the appendix with the localization table as "what more layers buy you".

**Figure work** (all re-reads; the per-layer rows behind fig 6, 7 and 8 are saved, including the aware attacker's at layers 17 to 24):

| figure | change |
|---|---|
| fig 3 | (a) detection against layer offset at 0.0085 and 0.02, five models, recomputation as a flat line at its floor (data: `single-layer-2026-09-08.json`); (b) deviation by layer under `injection` and `all` scope, six models (data: `*_fp32/pgd/pgd_sentiment_b0.0085.jsonl`); the hero |
| fig 6, fig 7 | add "single layer at the declared offset" and "injection layer alone" lines beside the run rule; recomputation's full-log count as a marker |
| fig 8 | add the single-layer read on the same rows; replace with item 4's run once done |
| fig 4, 5, 10 | keep; already one layer (GPT-2 layer 8); they are the calibration methodology |
| fig 1, 11 | keep |

Nanda on figures: red-green off, annotate the threshold and the legal budget on every ladder plot, one takeaway per caption. Carlini: if the figure needs the text to be understood, split it.

---

## 4. Structure, with what goes where

Nanda's abstract flow and intro paragraph template, filled in.

### Abstract

Not a template. The test is whether the first two sentences make a reader who audits models or uses steering stop, and whether the last two say what changes for them if the result holds. The draft in paper-ai.md is built as: the threat (steering changes a model with no change to weights or prompts, so no weight or prompt audit sees it); the question (does the forward pass give it away); the result under the one-layer threat model, with what the auditor does not need listed; why it works (hold one layer quiet, another gets louder); the recomputation reference, stated as the sharper check where it can run; the five-model evidence in one sentence; the scope phrase verbatim, turned into the point (the window in which the attacker is invisible is the window in which the attack does nothing); and the two consequences (tampering invisible to weight and prompt audits is auditable with logs that already exist; a steered evaluation is distinguishable from a prompted one after the fact).

Not in the abstract: unrun experiments, the deployment stopping rule, sample sizes, per-cell counts, the name of the statistic.

Be specific with numbers, no hedging words where a number exists. Most readers stop here.

### Introduction

- Para 1: steering is everywhere (Arditi refusal direction, CAA, Lindsey introspection). Why an auditor would want to know it happened.
- Para 2: the observation model from section 1, as the ladder: prompt plus weights (clean counterfactual), layers on both sides of the injection plus weights (recompute one block; the reference, reported), any set of layers at or after the injection plus weights, not adjacent, not known to the attacker (this paper). Then Mishra et al.'s non-surjectivity result and why a proof is not a detector. This paragraph is where the recompute objection is answered, before it is raised.
- Para 3: main claim, and the novelty delineation Nanda insists on: the theory is theirs; the calibrated single-layer detector with measured FPR and TPR, the 2x2 of jailbreak against detection, and the detector-aware attacker are new here. Fonseca Rivera & Africa's learned detector generalises only to nearby directions; this one is direction-agnostic by construction.
- Para 3.5: the single strongest piece of evidence, which is fig 3 as the offset curve plus the "0 undetected gains" line.
- Para 4: what this means for trusting steering-based evaluations and for auditing deployed models.
- Contributions bullets, each pointing at a section and a figure.

Carlini's storytelling note: the reader arrives thinking "of course a probe would find an added vector". Meet them there. AGENT.md has the sentence: a probe detects the intervention, not the model's response to it, and needs the vector's direction at training time. Say that in the intro, then say why inversion is different, and that a density on the raw state fails for the same reason.

### Threat model and detector (methods)

- The auditor's inputs, precisely: the residual state at each logged layer, every prompt position; recovered tokens; a clean bank; the model. Never the clean forward pass of the prompt under test, never adjacent layers, and the logged set is not known to the attacker.
- The ladder from section 1, stated once: which inputs each of the three methods needs, and that the detector's rung is any set of layers at or after the injection, not adjacent, not known to the attacker, one being enough against the attacks run.
- SipIt inversion, one paragraph, cite Mishra et al. Then the [NAME] architecture as its own subsection, with the delineating sentence from section 0.
- The statistic: per-position log-residual at the logged layer, z-scored per (layer, role), top-1, against a per-layer threshold. Three verdicts: flagged, clear, unscorable. State plainly that unscorable is never read as clean, and that with one layer there is no fallback.
- Several logged layers, one subsection: the any-layer rule from section 2 item 5, the constraint-scope table as the reason several layers cost the attacker more than one, and localization as what adjacent layers buy. The run rule with min_run and sigma_any is mentioned as the shipped variant and sent to the appendix.
- Calibration protocol: what is fitted on what, how sigma is set, and that the in-sample rate is a training constraint.
- The attack: PGD at layer round(0.7 x blocks), last position, `all` constraint, arms pgd/random/clean, budget ladder, what "legal" means and that it is a rule of the experiment.

### Results

One subsection per claim, in the order of the abstract. Every table carries n and an interval. Paired transitions replace aggregate ASR everywhere; RESULTS.md already made that rule.

### Attacking the detector

Fig 4, fig 5, fig 8 belong together as "what an attacker can do": concentrate, spread, differentiate through the rule, and, with section 2 item 7, inject after the logged layer. This is the section that makes the paper a security paper rather than a method paper, and it is where Carlini's "reader should want to argue" test is met.

### Limitations

Nanda: respect goes up, not down, for stating these first. The list is already written in RESULTS.md and the red-team closure. In the paper:

- Deployment vs evaluator stopping rule, and what changes (unscorable, not flagged).
- Sample sizes and what the lower bounds are.
- The judge changes 34 of 750 labels on identical text.
- No proof of no evasion; the attacks tested and the ones not (other layers, all-position deltas, profile-sculpting).
- Cost: full inversion of the logged layer at every position; the vocabulary table is 13 GB for all layers of a 0.5B model, one layer's slice in deployment.
- The fp16 tolerance floor that makes Qwen-1.5B unscorable at its last two layers under the deployment rule, which is why the declared offset is not the last layer.
- With the full trajectory recompute is the sharper tool (floor 5e-4 to 9e-4 against a 1e-2 tolerance), and Llama-3.2-1B at the legal budget is invisible to inversion from every layer while recompute has a 2x floor margin.
- The density baseline is on one model.

### Related work

After the results (Nanda), since the motivation is carried by Mishra et al. alone in the intro. Introspection line (Lindsey, Fonseca Rivera & Africa, Macar, the skeptics), inversion (SipIt), jailbreak evaluation (Arditi, JailbreakBench, HarmBench, StrongREJECT), and any activation-anomaly or backdoor-detection work found in a fresh search.

### Appendix and companion post

Nanda's tacit-knowledge section is unusually well stocked here. The corrections log in AGENT.md, the provenance losses, the wrong-units incident, the failed prediction that z-scoring would rescue top-5, the shell-script-edited-while-running incident, and the run rule that lost to a single layer are exactly what he says the field discards. Put a cleaned corrections log in the appendix and the full thing in an Alignment Forum companion post. It also timestamps the idea, which the publication plan already wanted.

---

## 5. Writing rules specific to this paper

Most are already enforced in RESULTS.md. Keep them in the paper.

- Quote measured rates, never nominal ones. "3 sigma" is 1 percent here, not 0.135 percent.
- "No effective evasion was found using the attacks and settings tested." Never shorter.
- A prompt the calibration cannot cover is unscorable, not clean.
- Every rate carries n and an interval; every zero-failure claim carries its cluster count and bound.
- Prompts at five budgets are clusters, not independent draws.
- Evaluator results and deployment-rule results are labelled as such wherever they appear.
- Define "legal budget", "relative budget", "role", "logged layer", "read set", "offset", "constraint scope", "arm", and "gain" once, early, in a notation table. Nanda's illusion-of-transparency warning applies: these words carry months of context nobody else has.
- Name the models the same way everywhere. Pick one set of short names and use them in every figure.
- No architecture-of-the-code in the paper. Script names go in the reproducibility appendix.

Carlini: read the draft aloud or through text-to-speech. Nanda: budget equal effort for abstract, intro, figures, and everything else, because they get equal reader-hours.

---

## 6. Process and calendar

Nanda's seven stages, with dates. Today is 8 Sep 2026. The ICLR 2027 abstract deadline is 18 Sep and the paper deadline is 25 Sep. The publication plan says the project is not deadline-driven, and section 2's blocking items are not ten days of work alongside writing. Two honest paths:

- **Path A, ICLR main track.** Fix the red-team P1s (2 days); declare the offset and measure its held-out FPR, run the any-layer rule, redraw fig 3, 6, 7 (1 to 2 days, all re-reads); add `--penalty_layers` and run the sculpting attacker on Qwen-0.5B, judged (1 to 2 days); write in parallel. Skip the top-ups, the late-injection ladder and the deployment benchmark; state them as limitations. Risk: reviewers reject on n=15, on the unbenchmarked deployment rule, and on the unmeasured "inject after the log" objection. Carlini says award-winning papers are often rejected first and strengthened, so this is not fatal, but it spends a submission.
- **Path B, workshop or ARR plus companion post first.** Do items 1 to 9 properly over three to four weeks, post the Alignment Forum writeup at the two-week mark to timestamp and get feedback, then submit to ARR or the next workshop round with a maximal paper. This is what the publication plan already leans toward.

Recommendation: B, with the Alignment Forum post as the forcing function for the compressed narrative (stage 1 below). If A is chosen anyway, the abstract deadline requires only the abstract and author list, so section 4's abstract can be written and submitted by the 18th without committing to the paper.

Stages either way:

1. **Compress** (1 day). Explain the paper in five minutes to someone. Write the one sentence and three claims from section 0 in your own words. If they differ from section 0, section 0 is wrong.
2. **Bullet narrative** (1 day). One page of bullets. Get one reader who knows the literature to read it. The Alignment Forum post is this stage, published.
3. **Intro outline** (1 day). Section 4's intro paragraphs as bullets with citations.
4. **Full outline** (1 day). Every section, every figure named, each with "what breaks if cut".
5. **Results collection** (the section 2 work). Regenerate RESULTS.md and figures from the manifest. Every paper number comes from `scripts/current_results.py` output or `src/plot_paper.py`, never typed by hand.
6. **First draft** (3 to 4 days). Prose from the outline. Methods and results first, intro second, abstract last.
7. **Iterate** (1 week). Two outside readers. Cut whatever they did not remember. Captions last.

---

## 7. Title candidates

Carlini: accurate over clever, and trouble titling means the paper does too much.

- *Steered states have no prompt: a reachability audit for activation steering from one logged layer*
- *Steered states have no prompt: a reachability audit for activation steering*
- *The evasion window is empty: steering that changes behaviour is detectable without the vector*
- *[NAME]: detecting activation steering from a single logged layer*

The first names the threat model, the method and the theorem in one line, and it is the one to start from. The third is the claim alone and is the fallback if the method half shrinks further. The method's contribution is the one-layer setting; the negative result stands under both methods.
