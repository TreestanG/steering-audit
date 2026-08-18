import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sipit
import utils
from sentiment_dir import _steer_hidden, build_steering_vectors, load_pairs
from utils import get_decoder_layers, load_model


class Top2:
    """Two smallest distances seen so far, over a chunked scan."""

    def __init__(self) -> None:
        self.best = float("inf")
        self.best_id = -1
        self.runner = float("inf")

    def update(self, d: Tensor, ids: Tensor) -> None:
        # The global top-2 must appear in some chunk's top-2, so merging per-chunk
        # pairs is exact and avoids keeping a |V|-long distance vector around.
        top = torch.topk(d, k=min(2, d.shape[0]), largest=False)
        for dist, idx in zip(top.values.tolist(), top.indices.tolist()):
            if dist < self.best:
                self.runner, self.best, self.best_id = self.best, dist, int(ids[idx])
            elif dist < self.runner:
                self.runner = dist

    @property
    def gap(self) -> float:
        return self.runner - self.best


def steering_delta(direction: Tensor, scale: Tensor, fraction: float) -> Tensor:
    """The vector make_add_vector_hook injects — fraction of the layer's mean state norm."""
    return fraction * scale * direction / direction.norm()


def random_delta(hidden_size: int, scale: Tensor, fraction: float, seed: int) -> Tensor:
    """Norm-matched control direction, so only orientation differs from the steering vector."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(hidden_size, generator=g)
    return fraction * scale * v / v.norm()


def make_delta_hook(delta: Tensor):
    def add(module, input, output):
        return _steer_hidden(output, delta)

    return add


@torch.no_grad()
def layer_state(prompt: str, layer: int, hook_fn=None) -> Tensor:
    """Hidden state at `layer` for the last position, under an optional steering hook.

    Runs the real perturbed forward rather than adding delta to a clean state: at the
    final block the model's own norm is applied after the hook, so the two differ there.
    """
    model, tokenizer = utils.model, utils.tokenizer
    if model is None or tokenizer is None:
        raise RuntimeError("Call load_model(...) first")
    captured: dict[str, Tensor] = {}

    def capture(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured["act"] = out[0, -1].detach().clone()

    block = get_decoder_layers()[layer - 1]
    handles = []
    if hook_fn is not None:  # registered first, so `capture` sees the steered output
        handles.append(block.register_forward_hook(hook_fn))
    handles.append(block.register_forward_hook(capture))

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        model(**{k: v.to(device) for k, v in inputs.items()})
    finally:
        for h in handles:
            h.remove()

    h = captured["act"]
    if layer == model.config.num_hidden_layers:
        h = model.model.norm(h)  # match sipit.candidate_states: last layer is post-norm
    return h.float().cpu()


@torch.no_grad()
def scan_vocab(cache, prefix_len: int, targets: dict[tuple[int, str], Tensor], chunk: int):
    """One full-vocabulary pass; running top-2 distance for every (layer, kind) target.

    Candidate states depend only on the pinned prefix, so a single forward with
    output_hidden_states serves every layer and every target at once — the alternative
    (sipit.candidate_states per layer) recomputes the same states once per depth.
    """
    model = utils.model
    if model is None:
        raise RuntimeError("Call load_model(...) first")
    device = next(model.parameters()).device
    vocab_size = int(model.config.vocab_size)
    layers = sorted({layer for layer, _ in targets})

    # Same KV budget sipit uses: expanding the prefix cache per candidate is what blows up.
    cap = max(256, min(chunk, sipit.KV_BUDGET_BYTES // max(1, sipit.kv_bytes_per_token() * prefix_len)))
    tracked = {key: Top2() for key in targets}

    for start in range(0, vocab_size, cap):
        cands = torch.arange(start, min(start + cap, vocab_size))
        n = cands.shape[0]
        batch_cache = copy.deepcopy(cache)
        batch_cache.batch_repeat_interleave(n)
        out = model.model(
            input_ids=cands.view(-1, 1).to(device),
            attention_mask=torch.ones(n, prefix_len + 1, dtype=torch.long, device=device),
            past_key_values=batch_cache,
            output_hidden_states=True,
        )
        for layer in layers:
            states = out.hidden_states[layer][:, 0].float().cpu()
            for key, target in targets.items():
                if key[0] == layer:
                    tracked[key].update((states - target).norm(dim=1), cands)
        del out

    return tracked


@torch.no_grad()
def audit_prompt(
    prompt: str,
    steering: dict[int, tuple[Tensor, Tensor]],
    *,
    layers: list[int],
    fraction: float,
    rel_tol: float,
    chunk: int,
    seed: int,
) -> list[dict]:
    """Clean/steered/random targets at every layer, scored against one shared vocab scan."""
    model, tokenizer = utils.model, utils.tokenizer
    if model is None or tokenizer is None:
        raise RuntimeError("Call load_model(...) first")

    ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    if len(ids) < 2:
        raise ValueError(f"prompt tokenizes to {len(ids)} token(s); need >= 2")
    prefix, true_id = ids[:-1], ids[-1]

    targets: dict[tuple[int, str], Tensor] = {}
    deltas: dict[int, dict[str, Tensor]] = {}
    for layer in layers:
        direction, scale = steering[layer]
        d_steer = steering_delta(direction, scale, fraction)
        d_rand = random_delta(model.config.hidden_size, scale, fraction, seed)
        deltas[layer] = {"steer": d_steer, "rand": d_rand}
        targets[layer, "clean"] = layer_state(prompt, layer)
        targets[layer, "steer"] = layer_state(prompt, layer, make_delta_hook(d_steer))
        targets[layer, "rand"] = layer_state(prompt, layer, make_delta_hook(d_rand))

    cache, _ = sipit.encode_step(torch.tensor([prefix]), cache=None, attn_len=len(prefix))
    tracked = scan_vocab(cache, len(prefix), targets, chunk)

    rows = []
    for layer in layers:
        clean, steer, rand = (tracked[layer, k] for k in ("clean", "steer", "rand"))
        h_norm = float(targets[layer, "clean"].norm())
        gap = clean.gap
        rows.append(
            {
                "prompt": prompt,
                "layer": layer,
                "h_norm": h_norm,
                "true_token": true_id,
                "rel_gap": gap / h_norm,  # the curve from plot_sipit_layers, on this prompt
                "clean": {
                    "residual": clean.best,
                    "gap": gap,
                    "token": clean.best_id,
                    "recovered": clean.best_id == true_id,
                },
                **{
                    kind: {
                        "delta_norm": float(deltas[layer][kind].norm()),
                        "residual": t.best,
                        "gap": t.gap,
                        "token": t.best_id,
                        "recovered": t.best_id == true_id,
                        "rel_residual": t.best / h_norm,
                        # Thm 3.2 guarantees exact recovery below 0.5.
                        "margin_spent": t.best / gap if gap > 0 else float("inf"),
                        "detected": t.best > rel_tol * h_norm,
                    }
                    for kind, t in (("steer", steer), ("rand", rand))
                },
            }
        )
    return rows


def summarize(rows: list[dict], layers: list[int]) -> None:
    """Per-layer means across prompts — the paired version of the two-curve overlay."""
    print(
        f"\n{'layer':>5} {'||h||':>9} {'gap/||h||':>10} {'res/||h||':>10} {'res/gap':>9} "
        f"{'recov':>6} {'det':>5}   | random: {'res/gap':>9} {'recov':>6} {'det':>5}"
    )
    for layer in layers:
        at = [r for r in rows if r["layer"] == layer]
        if not at:
            continue

        def mean(fn):
            return sum(fn(r) for r in at) / len(at)

        print(
            f"{layer:>5} {mean(lambda r: r['h_norm']):>9.2f} "
            f"{mean(lambda r: r['rel_gap']):>10.4f} "
            f"{mean(lambda r: r['steer']['rel_residual']):>10.4f} "
            f"{mean(lambda r: r['steer']['margin_spent']):>9.3f} "
            f"{mean(lambda r: r['steer']['recovered']) * 100:>5.0f}% "
            f"{mean(lambda r: r['steer']['detected']) * 100:>4.0f}%   | "
            f"        {mean(lambda r: r['rand']['margin_spent']):>9.3f} "
            f"{mean(lambda r: r['rand']['recovered']) * 100:>5.0f}% "
            f"{mean(lambda r: r['rand']['detected']) * 100:>4.0f}%"
        )
    clean_res = max(r["clean"]["residual"] / r["h_norm"] for r in rows)
    clean_ok = sum(r["clean"]["recovered"] for r in rows) / len(rows)
    print(f"\nsanity: clean recovery {clean_ok * 100:.0f}%, worst clean residual/||h|| {clean_res:.2e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_path", type=str, default="data/sentiment_opposites_train.json")
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/sentiment_opposites_test.json",
        help="prompts to audit; the negative half of each pair is used, as in sentiment_dir",
    )
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=5,
        help="a full vocab scan per prompt covers every layer, so this is the runtime knob",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="",
        help="comma-separated blocks to audit (default: every block)",
    )
    parser.add_argument(
        "--rel_tol",
        type=float,
        default=1e-3,
        help="detection threshold: residual > rel_tol * ||h|| counts as caught",
    )
    parser.add_argument("--chunk", type=int, default=2048, help="candidates per forward")
    parser.add_argument("--seed", type=int, default=42, help="random control direction")
    parser.add_argument("--out", type=Path, default=Path("results/steer_audit.jsonl"))
    args = parser.parse_args()

    load_model(args.model_name)
    if utils.model is None:
        raise RuntimeError("load_model(...) did not set utils.model")

    # Built before the device move: get_token_activations feeds the model CPU tensors.
    steering = build_steering_vectors(load_pairs(args.train_path))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    utils.model.to(device)
    print(f"model on {next(utils.model.parameters()).device}")

    layers = (
        [int(s) for s in args.layers.split(",")]
        if args.layers
        else list(range(1, utils.model.config.num_hidden_layers + 1))
    )
    prompts = [neg for _, neg in load_pairs(args.test_path)][: args.n_prompts]
    print(f"{len(prompts)} prompts x {len(layers)} layers, fraction={args.fraction}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with args.out.open("w") as f:
        for i, prompt in enumerate(prompts, start=1):
            start = time.time()
            got = audit_prompt(
                prompt,
                steering,
                layers=layers,
                fraction=args.fraction,
                rel_tol=args.rel_tol,
                chunk=args.chunk,
                seed=args.seed,
            )
            for row in got:
                f.write(json.dumps(row) + "\n")
            f.flush()
            rows.extend(got)
            print(f"[{i}/{len(prompts)}] {prompt!r} in {time.time() - start:.1f}s")

    summarize(rows, layers)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
