"""A raw-state density baseline for the one-logged-layer setting.

How far the last-position state at one layer sits from the clean bank's states at the
same layer and position, given the injected position outright (last prompt token) and
read at the injection layer and each layer after it. Three scores, each fitted on the
bank with a leave-one-out 95th-percentile threshold (the in-sample rule would under-read
a density fitted on the same points): 'diag' standardised Euclidean distance, 'lw'
Mahalanobis under a Ledoit-Wolf shrinkage covariance, 'knn' relative distance to the
nearest bank state. The ladder's clean arm is held out from the bank, and is also used as
a matched leave-one-out reference of the same kind of text. Driver:
scripts/mahalanobis_baseline.py.
"""

import time
from pathlib import Path

import numpy as np
import torch

import provenance
import recompute
from recompute import log
from utils import DTYPES

SD_FLOOR = 1e-6


class Diag:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), np.maximum(X.std(0, ddof=1), SD_FLOOR)
        return self

    def score(self, X):
        return np.sqrt((((X - self.mu) / self.sd) ** 2).mean(1))


class LW:
    def fit(self, X):
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(X)
        self.mu, self.P = lw.location_, lw.precision_
        self.shrinkage = float(lw.shrinkage_)
        return self

    def score(self, X):
        d = X - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.P, d) / X.shape[1])


class KNN:
    def fit(self, X):
        self.X = X
        return self

    def score(self, X, exclude=None):
        d = np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=-1)
        if exclude is not None:
            d[np.arange(len(X)), exclude] = np.inf
        return d.min(1) / np.linalg.norm(X, axis=1)


ESTIMATORS = {"diag": Diag, "lw": LW, "knn": KNN}


def loo_scores(kind: str, X: np.ndarray) -> np.ndarray:
    n = len(X)
    out = np.empty(n)
    if kind == "knn":
        return KNN().fit(X).score(X, exclude=np.arange(n))
    for i in range(n):
        keep = np.arange(n) != i
        out[i] = ESTIMATORS[kind]().fit(X[keep]).score(X[i:i + 1])[0]
    return out


def q95(x: np.ndarray) -> float:
    return float(np.quantile(x, 0.95, method="higher"))


def run(model_name: str, ladder: Path, banks: list[Path], summary: Path | None = None,
        dtype: str = "float16", device: str | None = None, limit: int = 0) -> dict:
    t0 = time.time()
    model, tok = recompute.load_model(model_name, DTYPES[dtype], device or recompute.utils.pick_device())
    n_blocks = int(model.config.num_hidden_layers)
    layer, prompts, h_norm1, cells = recompute.load_ladder(ladder)
    budgets = sorted(cells)
    if limit:
        prompts = prompts[:limit]
    n = len(prompts)
    ids = [tok(t, return_tensors="pt")["input_ids"][0] for t in prompts]
    bank_ids = [tok(t, return_tensors="pt")["input_ids"][0] for t in recompute.bank_prompts(banks, limit)]
    log(f"{model_name} on {recompute.DEV} {dtype}: {n_blocks} blocks, injection layer {layer}, "
        f"{n} prompts, bank {len(bank_ids)}, budgets {budgets}")

    def last_states(ids_t, delta):
        return recompute.last_states(recompute.observe_single(ids_t, layer, delta), n_blocks)

    bank = np.stack([last_states(t, None) for t in bank_ids])          # (nb, blocks, d)
    clean = np.stack([last_states(t, None) for t in ids])              # (n, blocks, d)
    drift = max(abs(float(np.linalg.norm(clean[i, layer - 1])) - h_norm1[i]) / h_norm1[i]
                for i in range(n))
    if drift > 0.02:
        raise SystemExit(f"||h|| at the injection differs from stage 1 by {drift:.1%}")
    arms = {("clean", 0.0): clean}
    realised = {}
    for b in budgets:
        for arm in ("pgd", "random"):
            deltas = torch.stack([cells[b]["deltas"][(arm, i)] for i in range(n)])
            arms[(arm, b)] = np.stack([last_states(t, deltas[i]) for i, t in enumerate(ids)])
            ref = clean[:, layer - 1]
            realised[(arm, b)] = float(np.median(
                np.linalg.norm(arms[(arm, b)][:, layer - 1] - ref, axis=1) / np.linalg.norm(ref, axis=1)))
    log(f"states captured in {time.time() - t0:.0f}s; drift vs stage 1 {drift:.2%}")

    detector = recompute.detector_counts(summary, layer, n)
    per_layer = []
    for L in range(layer, n_blocks + 1):
        Xb = bank[:, L - 1].astype(np.float64)
        entry = {"layer": L, "offset": L - layer, "estimators": {}}
        for kind in ESTIMATORS:
            loo = loo_scores(kind, Xb)
            thr = q95(loo)
            est = ESTIMATORS[kind]().fit(Xb)
            sd_loo = float(loo.std(ddof=1))
            cellz = {}
            for (arm, b), X in arms.items():
                s = est.score(X[:, L - 1].astype(np.float64))
                cellz[f"{arm}|{b}"] = {"flagged": int((s > thr).sum()), "n": n,
                                       "median_score": float(np.median(s))}
                if arm != "clean":
                    s0 = est.score(clean[:, L - 1].astype(np.float64))
                    cellz[f"{arm}|{b}"]["median_shift_in_bank_sd"] = float(np.median(s - s0) / sd_loo)
            entry["estimators"][kind] = {
                "threshold_loo95": thr, "bank_loo_median": float(np.median(loo)),
                "bank_loo_sd": sd_loo, "bank_loo_flagged": int((loo > thr).sum()),
                "shrinkage": getattr(est, "shrinkage", None), "cells": cellz}
        # matched reference: the ladder's own clean arm, leave-one-out. Prompt i is
        # scored, clean and attacked, against a fit on the other n-1 clean states, so
        # the reference is the same kind of text and the prompt under test is held out.
        Xc = clean[:, L - 1].astype(np.float64)
        entry["matched"] = {}
        for kind in ESTIMATORS:
            loo_c = np.empty(n)
            att = {key: np.empty(n) for key in arms if key[0] != "clean"}
            for i in range(n):
                keep = np.arange(n) != i
                est = ESTIMATORS[kind]().fit(Xc[keep])
                loo_c[i] = est.score(Xc[i:i + 1])[0]
                for key in att:
                    att[key][i] = est.score(arms[key][i:i + 1, L - 1].astype(np.float64))[0]
            thr = q95(loo_c)
            sd_c = float(loo_c.std(ddof=1))
            cellz = {"clean|0.0": {"flagged": int((loo_c > thr).sum()), "n": n,
                                   "median_score": float(np.median(loo_c))}}
            for key, s in att.items():
                cellz[f"{key[0]}|{key[1]}"] = {
                    "flagged": int((s > thr).sum()), "n": n, "median_score": float(np.median(s)),
                    "median_shift_in_clean_sd": float(np.median(s - loo_c) / sd_c)}
            entry["matched"][kind] = {"threshold_loo95": thr, "clean_loo_sd": sd_c, "cells": cellz}
        per_layer.append(entry)
        m = entry["matched"]
        log(f"L{L:>2d} (+{L - layer}) matched-reference " + " ".join(
            f"| {kind}: clean {m[kind]['cells']['clean|0.0']['flagged']:>2d}/{n} " + " ".join(
                f"p{b:g}={m[kind]['cells'][f'pgd|{b}']['flagged']:>2d}" for b in budgets)
            + f" r0.3={m[kind]['cells']['random|0.3']['flagged']:>2d}"
            for kind in ESTIMATORS))
        line = f"L{L:>2d} (+{L - layer}) bank-reference "
        for kind in ESTIMATORS:
            c = entry["estimators"][kind]["cells"]
            line += f"| {kind}: clean {c['clean|0.0']['flagged']:>2d}/{n} " + " ".join(
                f"p{b:g}={c[f'pgd|{b}']['flagged']:>2d}" for b in budgets) + " "
        log(line)

    return {
        "model": model_name, "dtype": dtype, "device": str(recompute.DEV), "layer": layer,
        "n_blocks": n_blocks, "n_prompts": n, "bank_n": len(bank_ids),
        "observation": "last-position residual state at one layer, the injected position "
                       "given; bank states at the same layer and position; nothing else",
        "threshold_rule": "leave-one-out 95th percentile; 'estimators' use the clean bank "
                          "(five text categories) as reference, 'matched' uses the ladder's "
                          "own clean arm leave-one-out",
        "realised_rel_delta_median": {f"{k[0]}|{k[1]}": v for k, v in realised.items()},
        "detector": {f"{k[0]}|{k[1]}": v for k, v in detector.items()},
        "per_layer": per_layer,
        "sanity": {"max_h_norm_drift_vs_stage1": drift},
        "provenance": provenance.stamp(**{Path(c["file"]).name: c["file"] for c in cells.values()},
                                       summary=summary, **{f"bank{i}": b for i, b in enumerate(banks)}),
        "elapsed_s": time.time() - t0,
    }
