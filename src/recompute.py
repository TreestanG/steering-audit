"""Two baselines against the PGD ladder rows: a recomputation check and a linear probe.

Both see the trajectory the detector inverted: the fp16 forward pass with the saved
stage-1 delta added at the last prompt position of the injection layer, regenerated here
from <stem>_deltas.pt. What each method observes beyond that trajectory is declared in
OBSERVATION and written into the output.

The recomputation check runs block j on the observed h_j and compares with the observed
h_{j+1}, at every layer and prompt position. Two floors are measured for it. "same-kernel":
the observed states come from the same unpadded single-prompt pass the auditor recomputes
with, so a clean trajectory recomputes exactly and the floor is numerical noise only.
"serving-batch": the observed states come from the left-padded batched pass that produced
the decoded completions (generate.py's path), and the auditor recomputes each prompt on its
own, so the floor is the fp16 discrepancy between the two kernels. The threshold in each
setting is the 95th percentile of the clean chat bank's score, the detector's own 5%
in-sample rule. The probe is a logistic regression on the last-position state at the
injection layer or at every layer, trained on labelled clean and attacked states of other
prompts. Driver: scripts/baselines_recompute_probe.py.
"""

import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

os.environ.setdefault("AAT_DTYPE", "float16")

import provenance  # noqa: E402
import utils  # noqa: E402
from generate import Intervention, position_ids_for  # noqa: E402
from ladders import resolve  # noqa: E402
from utils import DTYPES, get_decoder_layers  # noqa: E402

OBSERVATION = {
    "trajectory": "fp16 forward pass of the prompt with the stage-1 delta added at the last "
                  "prompt position of the injection layer; every block's input and output "
                  "at every prompt position. The same states the detector inverted.",
    "recompute": "the trajectory plus a trusted copy of the weights. Block j is run on the "
                 "observed h_j and compared with the observed h_{j+1} at every layer and "
                 "position; the score is the largest relative residual. No tokens, no clean "
                 "pass, no labels. Threshold: 95th percentile of the score on the clean chat "
                 "bank, in the same setting.",
    "probe": "the last-position state at the injection layer ('inj') or at every layer "
             "concatenated ('all'), plus labelled clean and attacked states of OTHER prompts "
             "(5 folds by prompt, logistic regression on standardised features) and the "
             "injection layer index.",
}
FOLDS = 5
SEED = 0
BANK_CHUNK = 25
DEV = torch.device("cpu")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_model(model_name: str, dtype: torch.dtype, device: str):
    global DEV
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.eval().to(utils.resolve_device(device))
    utils.model, utils.tokenizer = model, tok
    DEV = next(model.parameters()).device
    return model, tok


class Trace:
    """h[0] is block 0's input, h[k] is block k-1's output. `override` replaces block j's
    input with a given tensor, so one pass recomputes every layer transition from the
    observed states at once."""

    def __init__(self, override: dict[int, Tensor] | None = None):
        self.override = override or {}
        self.h: dict[int, Tensor] = {}
        self.handles = []

    def __enter__(self):
        for j, block in enumerate(get_decoder_layers()):
            self.handles.append(block.register_forward_pre_hook(self._pre(j), with_kwargs=True))
            self.handles.append(block.register_forward_hook(self._post(j)))
        return self

    def _pre(self, j: int):
        def fn(module, args, kwargs):
            if j == 0:
                x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                self.h[0] = x.detach()
            if j not in self.override:
                return None
            x = self.override[j]
            if "hidden_states" in kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = x
                return args, kwargs
            return (x,) + tuple(args[1:]), kwargs
        return fn

    def _post(self, j: int):
        def fn(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            self.h[j + 1] = out.detach()
        return fn

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def _hook(layer: int, delta: Tensor | None, mask: Tensor):
    if delta is None:
        return []
    iv = Intervention(layer=layer, delta=delta, positions="index", index=-1)
    return [get_decoder_layers()[layer - 1].register_forward_hook(iv.hook(mask=mask))]


@torch.no_grad()
def observe_single(ids: Tensor, layer: int, delta: Tensor | None) -> dict[int, Tensor]:
    model = utils.model
    x = ids.to(DEV).unsqueeze(0)
    mask = torch.ones_like(x)
    handles = _hook(layer, None if delta is None else delta.unsqueeze(0).to(DEV), mask)
    try:
        with Trace() as tr:
            model(input_ids=x, attention_mask=mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {k: v[0] for k, v in tr.h.items()}


@torch.no_grad()
def observe_batch(ids_list: list[Tensor], layer: int,
                  deltas: Tensor | None) -> list[dict[int, Tensor]]:
    model, tok = utils.model, utils.tokenizer
    width = max(int(t.shape[0]) for t in ids_list)
    x = torch.full((len(ids_list), width), int(tok.pad_token_id), dtype=torch.long)
    mask = torch.zeros((len(ids_list), width), dtype=torch.long)
    for b, t in enumerate(ids_list):
        x[b, width - t.shape[0]:] = t
        mask[b, width - t.shape[0]:] = 1
    x, mask = x.to(DEV), mask.to(DEV)
    handles = _hook(layer, None if deltas is None else deltas.to(DEV), mask)
    try:
        with Trace() as tr:
            model(input_ids=x, attention_mask=mask, position_ids=position_ids_for(mask),
                  use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return [{k: v[b, width - t.shape[0]:].contiguous() for k, v in tr.h.items()}
            for b, t in enumerate(ids_list)]


@torch.no_grad()
def recompute(ids: Tensor, obs: dict[int, Tensor], n_blocks: int) -> Tensor:
    """Relative residual, shape (n_blocks, n_tokens): row j is block j run on the observed
    h_j against the observed h_{j+1}, i.e. layer index j+1."""
    x = ids.to(DEV).unsqueeze(0)
    with Trace({j: obs[j].unsqueeze(0) for j in range(n_blocks)}) as tr:
        utils.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
    rows = []
    for j in range(n_blocks):
        pred, ref = tr.h[j + 1][0].float(), obs[j + 1].float()
        rows.append((pred - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12))
    return torch.stack(rows).cpu()


def residual_summary(res: Tensor, layer: int) -> dict:
    inj = float(res[layer - 1, -1])
    masked = res.clone()
    masked[layer - 1, -1] = -1.0
    return {"score": float(res.max()), "inj": inj, "off": float(masked.max())}


def last_states(obs: dict[int, Tensor], n_blocks: int) -> np.ndarray:
    return torch.stack([obs[k][-1].float() for k in range(1, n_blocks + 1)]).cpu().numpy()


def load_ladder(ladder: Path):
    files = sorted(glob.glob(str(resolve(ladder) / "jbb_refusal_harmful_b*_deltas.pt")))
    if not files:
        raise SystemExit(f"no jbb_refusal_harmful_b*_deltas.pt under {ladder}")
    layer, prompts, cells, h_norm = None, None, {}, {}
    for f in files:
        blob = torch.load(f, weights_only=False)
        if int(blob.get("n_positions", 1)) != 1:
            continue
        b = float(blob["budget"])
        if layer is None:
            layer = int(blob["layer"])
        elif int(blob["layer"]) != layer:
            raise SystemExit(f"{f}: layer {blob['layer']} differs from {layer}")
        rows = {}
        for line in Path(f.replace("_deltas.pt", ".jsonl")).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["arm"], r["prompt_index"])] = r
        texts = {i: r["prompt"] for (arm, i), r in rows.items() if arm == "pgd"}
        texts = [texts[i] for i in range(len(texts))]
        if prompts is None:
            prompts = texts
            h_norm = {i: rows[("pgd", i)]["h_norm"] for i in range(len(texts))}
        elif texts != prompts:
            raise SystemExit(f"{f}: prompt set differs from the other budgets")
        deltas = {}
        for key, vec in blob["deltas"].items():
            c, arm, i = key.split("|")
            if c == "all":
                deltas[(arm, int(i))] = vec.float()
        cells[b] = {"deltas": deltas, "rows": rows, "file": f}
    assert layer is not None and prompts is not None
    return layer, prompts, h_norm, cells


def bank_prompts(banks: list[Path], limit: int = 0) -> list[str]:
    texts = []
    for path in banks:
        texts += [q["text"] for q in json.loads(resolve(path).read_text())["prompts"]]
    return texts[:limit] if limit else texts


def detector_counts(summary: Path | None, layer: int, n: int) -> dict:
    if summary is None or not resolve(summary).exists():
        return {}
    out = {}
    for c in json.loads(resolve(summary).read_text())["cells"]:
        if (c["behavior"] == "jbb_refusal" and c.get("inj_layer") == layer
                and c["n_positions"] in (0, 1) and c["n"] == n):
            out[(c["arm"], float(c["budget"]))] = {"flagged": c["flagged"], "n": c["n"],
                                                  "unscorable": c["unscorable"]}
    return out


def quantile_threshold(scores: list[float]) -> float:
    return float(np.quantile(np.array(scores), 0.95, method="higher"))


def probe(feats: dict[tuple[str, float], np.ndarray], budgets: list[float], layer: int,
          n: int) -> list[dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def view(arr: np.ndarray, kind: str) -> np.ndarray:
        return arr[:, layer - 1, :] if kind == "inj" else arr.reshape(arr.shape[0], -1)

    def fit(x0: np.ndarray, x1: np.ndarray):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        return clf.fit(np.concatenate([x0, x1]), np.r_[np.zeros(len(x0)), np.ones(len(x1))])

    folds = list(KFold(FOLDS, shuffle=True, random_state=SEED).split(np.arange(n)))
    top = max(budgets)
    out = []
    for kind in ("inj", "all"):
        clean = view(feats[("clean", 0.0)], kind)
        for b in budgets:
            pgd, rnd = view(feats[("pgd", b)], kind), view(feats[("random", b)], kind)
            pgd_top = view(feats[("pgd", top)], kind)
            counts = {"tpr_pgd": 0, "fpr_clean": 0, "tpr_random": 0,
                      "transfer_tpr_pgd": 0, "transfer_fpr_clean": 0}
            for tr, te in folds:
                clf = fit(clean[tr], pgd[tr])
                counts["tpr_pgd"] += int(clf.predict(pgd[te]).sum())
                counts["fpr_clean"] += int(clf.predict(clean[te]).sum())
                counts["tpr_random"] += int(clf.predict(rnd[te]).sum())
                clf_top = fit(clean[tr], pgd_top[tr])
                counts["transfer_tpr_pgd"] += int(clf_top.predict(pgd[te]).sum())
                counts["transfer_fpr_clean"] += int(clf_top.predict(clean[te]).sum())
            out.append({"feature": kind, "budget": b, "n": n, "transfer_from": top, **counts})
            log("probe %-3s b=%-6g pgd %2d/%d clean %2d/%d random %2d/%d | from %g: pgd %2d/%d clean %2d/%d"
                % (kind, b, counts["tpr_pgd"], n, counts["fpr_clean"], n, counts["tpr_random"], n,
                   top, counts["transfer_tpr_pgd"], n, counts["transfer_fpr_clean"], n))
    return out


def run(model_name: str, ladder: Path, banks: list[Path], summary: Path | None = None,
        dtype: str = "float16", device: str | None = None, limit: int = 0) -> dict:
    """Both baselines on one ladder; `limit` keeps the first k prompts for a smoke test."""
    t0 = time.time()
    model, tok = load_model(model_name, DTYPES[dtype], device or utils.pick_device())
    n_blocks = int(model.config.num_hidden_layers)
    layer, prompts, h_norm1, cells = load_ladder(ladder)
    budgets = sorted(cells)
    if limit:
        prompts = prompts[:limit]
    n = len(prompts)
    ids = [tok(t, return_tensors="pt")["input_ids"][0] for t in prompts]
    log(f"{model_name} on {DEV} {dtype}: {n_blocks} blocks, injection layer {layer}, "
        f"{n} prompts, budgets {budgets}")

    bank_ids = [tok(t, return_tensors="pt")["input_ids"][0] for t in bank_prompts(banks, limit)]

    bank_single, bank_batch, floor_single, floor_batch = [], [], [], []
    for t in bank_ids:
        res = recompute(t, observe_single(t, layer, None), n_blocks)
        bank_single.append(float(res.max()))
        floor_single.append(res.max(dim=1).values.numpy())
    for s in range(0, len(bank_ids), BANK_CHUNK):
        chunk = bank_ids[s:s + BANK_CHUNK]
        for t, obs in zip(chunk, observe_batch(chunk, layer, None)):
            res = recompute(t, obs, n_blocks)
            bank_batch.append(float(res.max()))
            floor_batch.append(res.max(dim=1).values.numpy())
    thr_single, thr_batch = quantile_threshold(bank_single), quantile_threshold(bank_batch)
    fs, fb = np.stack(floor_single), np.stack(floor_batch)
    bank = {"n": len(bank_ids), "files": [str(b) for b in banks],
            "threshold_single": thr_single, "threshold_batch": thr_batch,
            "fpr_single": int(sum(s > thr_single for s in bank_single)),
            "fpr_batch": int(sum(s > thr_batch for s in bank_batch)),
            "score_single": {"median": float(np.median(bank_single)), "max": float(max(bank_single))},
            "score_batch": {"median": float(np.median(bank_batch)), "max": float(max(bank_batch))},
            "per_layer_median_single": np.median(fs, axis=0).tolist(),
            "per_layer_median_batch": np.median(fb, axis=0).tolist(),
            "per_layer_max_batch": fb.max(axis=0).tolist()}
    log("bank n=%d | same-kernel floor median %.2e max %.2e thr %.2e | serving-batch floor "
        "median %.2e max %.2e thr %.2e" % (len(bank_ids), bank["score_single"]["median"],
                                           bank["score_single"]["max"], thr_single,
                                           bank["score_batch"]["median"], bank["score_batch"]["max"],
                                           thr_batch))

    det = detector_counts(summary, layer, n)
    feats: dict[tuple[str, float], np.ndarray] = {}
    clean_obs = [observe_single(t, layer, None) for t in ids]
    drift_h, drift_rel = 0.0, 0.0
    for i, obs in enumerate(clean_obs):
        got = float(obs[layer][-1].float().norm())
        drift_h = max(drift_h, abs(got - h_norm1[i]) / h_norm1[i])
    if drift_h > 0.02:
        raise SystemExit(f"||h|| at the injection differs from stage 1 by {drift_h:.1%}; "
                         "the prompts or positions do not match the attacked rows")

    plan = [("clean", 0.0)] + [(arm, b) for b in budgets for arm in ("pgd", "random")]
    out_cells = []
    for arm, b in plan:
        if arm == "clean":
            deltas, singles = None, clean_obs
        else:
            deltas = torch.stack([cells[b]["deltas"][(arm, i)] for i in range(n)])
            singles = [observe_single(t, layer, deltas[i]) for i, t in enumerate(ids)]
        feats[(arm, b)] = np.stack([last_states(o, n_blocks) for o in singles])
        single = [residual_summary(recompute(t, o, n_blocks), layer)
                  for t, o in zip(ids, singles)]
        batch = [residual_summary(recompute(t, o, n_blocks), layer)
                 for t, o in zip(ids, observe_batch(ids, layer, deltas))]
        cell = {"arm": arm, "budget": b, "n": n,
                "flagged_single": int(sum(s["score"] > thr_single for s in single)),
                "flagged_batch": int(sum(s["score"] > thr_batch for s in batch)),
                "score_single": [s["score"] for s in single],
                "score_batch": [s["score"] for s in batch],
                "inj_residual_single": [s["inj"] for s in single],
                "inj_residual_batch": [s["inj"] for s in batch],
                "off_injection_max_single": [s["off"] for s in single],
                "off_injection_max_batch": [s["off"] for s in batch],
                "detector": det.get((arm, b))}
        if arm != "clean":
            realised = []
            for i, o in enumerate(singles):
                ref = clean_obs[i][layer][-1].float()
                realised.append(float((o[layer][-1].float() - ref).norm() / ref.norm()))
                drift_rel = max(drift_rel, abs(realised[-1] - cells[b]["rows"][(arm, i)]["rel_delta"]))
            cell["rel_delta_stage1"] = [cells[b]["rows"][(arm, i)]["rel_delta"] for i in range(n)]
            cell["rel_delta_realised_fp16"] = realised
        out_cells.append(cell)
        log("%-6s b=%-6g | same-kernel flagged %2d/%d (inj median %.2e, off %.2e) | serving-batch "
            "flagged %2d/%d (inj median %.2e, off %.2e) | detector %s"
            % (arm, b, cell["flagged_single"], n, np.median(cell["inj_residual_single"]),
               np.median(cell["off_injection_max_single"]), cell["flagged_batch"], n,
               np.median(cell["inj_residual_batch"]), np.median(cell["off_injection_max_batch"]),
               "%d/%d" % (det[(arm, b)]["flagged"], det[(arm, b)]["n"]) if (arm, b) in det else "-"))

    probe_cells = probe(feats, budgets, layer, n)

    return {
        "model": model_name, "dtype": dtype, "device": str(DEV), "layer": layer,
        "n_blocks": n_blocks, "n_prompts": n, "ladder": str(ladder),
        "summary": None if summary is None else str(summary),
        "observation": OBSERVATION,
        "sanity": {"max_h_norm_drift_vs_stage1": drift_h,
                   "max_rel_delta_drift_vs_stage1": drift_rel},
        "recompute": {"bank": bank, "cells": out_cells},
        "probe": {"folds": FOLDS, "seed": SEED, "features": {
            "inj": int(feats[("clean", 0.0)].shape[2]),
            "all": int(np.prod(feats[("clean", 0.0)].shape[1:]))}, "cells": probe_cells},
        "provenance": provenance.stamp(**{Path(c["file"]).name: c["file"] for c in cells.values()},
                                       summary=summary),
        "elapsed_s": time.time() - t0,
    }
