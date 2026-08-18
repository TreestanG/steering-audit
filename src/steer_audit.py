import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import Tensor

import sipit
from sentiment_dir import _steer_hidden, build_steering_vectors, load_pairs
from sipit import Top2
from paths import results_dir
from utils import (
    apply_final_norm,
    get_base_model,
    get_decoder_layers,
    load_model,
    pick_device,
    require_model,
)


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
def layer_states(prompt: str, layers: list[int], hook_layer: int = 0, hook_fn=None,
                 post_norm: bool = True) -> dict[int, Tensor]:
    """Last-position hidden state at each of `layers`, from one forward pass.

    Runs the real perturbed forward rather than adding delta to a clean state: at the
    final block the model's own norm is applied after the hook, so the two differ there.

    post_norm=False returns the raw block output at the last layer instead of the
    model's post-norm state — what you want when comparing a block's output to
    itself, since the norm is nonlinear and would break `steered - base == delta`.

    States are captured straight off block L-1 rather than from out.hidden_states,
    because a forward hook that rewrites a block's output is not reflected in that
    block's own hidden_states slot — HF fills the slot before the rewrite propagates,
    which would place an injection one layer late.
    """
    model, tokenizer = require_model()
    blocks = get_decoder_layers()
    captured: dict[int, Tensor] = {}

    def make_capture(layer: int):
        def capture(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            captured[layer] = out[0, -1].detach().clone()

        return capture

    handles = []
    if hook_fn is not None:  # registered first, so `capture` sees the steered output
        handles.append(blocks[hook_layer - 1].register_forward_hook(hook_fn))
    for layer in layers:
        handles.append(blocks[layer - 1].register_forward_hook(make_capture(layer)))

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        model(**{k: v.to(device) for k, v in inputs.items()})
    finally:
        for h in handles:
            h.remove()

    # match sipit.candidate_states: the last layer is post-norm
    finish = apply_final_norm if post_norm else (lambda h, _L: h)
    return {L: finish(captured[L], L).float().cpu() for L in layers}


def layer_state(prompt: str, layer: int, hook_fn=None, post_norm: bool = True) -> Tensor:
    """Single-layer `layer_states`, hooking the same block it captures."""
    return layer_states(prompt, [layer], hook_layer=layer, hook_fn=hook_fn,
                        post_norm=post_norm)[layer]


def prefix_cache(prompt: str):
    """(prefix ids, true last-token id, KV cache for the prefix) — the setup every
    vocab scan needs. The last token is what the scan tries to recover."""
    _, tokenizer = require_model()
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    if len(ids) < 2:
        raise ValueError(f"prompt tokenizes to {len(ids)} token(s); need >= 2")
    prefix, true_id = ids[:-1], ids[-1]
    cache, _ = sipit.encode_step(torch.tensor([prefix]), cache=None, attn_len=len(prefix))
    return prefix, true_id, cache


@torch.no_grad()
def build_targets(prompt: str, steering: dict[int, tuple[Tensor, Tensor]], *,
                  layers: list[int], fraction: float, seed: int):
    """Clean / steered / random targets at every layer, plus the deltas injected.

    The clean states all come from one forward pass; only the steered and random
    ones need a hooked forward per layer, since each layer gets its own delta.
    """
    model, _ = require_model()
    clean = layer_states(prompt, layers)
    targets: dict[tuple[int, str], Tensor] = {}
    deltas: dict[int, dict[str, Tensor]] = {}
    for layer in layers:
        direction, scale = steering[layer]
        d_steer = steering_delta(direction, scale, fraction)
        d_rand = random_delta(model.config.hidden_size, scale, fraction, seed)
        deltas[layer] = {"steer": d_steer, "rand": d_rand}
        targets[layer, "clean"] = clean[layer]
        targets[layer, "steer"] = layer_state(prompt, layer, make_delta_hook(d_steer))
        targets[layer, "rand"] = layer_state(prompt, layer, make_delta_hook(d_rand))
    return targets, deltas


@torch.no_grad()
def scan_vocab(cache, prefix_len: int, targets: dict[tuple[int, str], Tensor], chunk: int):
    """One full-vocabulary pass; running top-2 distance for every (layer, kind) target.

    Candidate states depend only on the pinned prefix, so a single forward with
    output_hidden_states serves every layer and every target at once — the alternative
    (sipit.candidate_states per layer) recomputes the same states once per depth.
    """
    model, _ = require_model()
    device = next(model.parameters()).device
    vocab_size = int(model.config.vocab_size)

    # Grouped once: the layer -> targets mapping is loop-invariant, and rescanning
    # every target per layer per chunk is O(layers x targets) work for nothing.
    by_layer: dict[int, list[tuple[tuple[int, str], Tensor]]] = {}
    for key, target in targets.items():
        by_layer.setdefault(key[0], []).append((key, target))

    # Same KV budget sipit uses: expanding the prefix cache per candidate is what blows up.
    cap = sipit.batch_cap(prefix_len, chunk)
    tracked = {key: Top2() for key in targets}

    for start in range(0, vocab_size, cap):
        cands = torch.arange(start, min(start + cap, vocab_size))
        n = cands.shape[0]
        batch_cache = copy.deepcopy(cache)
        batch_cache.batch_repeat_interleave(n)
        out = get_base_model()(
            input_ids=cands.view(-1, 1).to(device),
            attention_mask=torch.ones(n, prefix_len + 1, dtype=torch.long, device=device),
            past_key_values=batch_cache,
            output_hidden_states=True,
        )
        for layer, entries in by_layer.items():
            states = out.hidden_states[layer][:, 0].float().cpu()
            for key, target in entries:
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
    prefix, true_id, cache = prefix_cache(prompt)
    targets, deltas = build_targets(prompt, steering, layers=layers, fraction=fraction, seed=seed)
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
    parser.add_argument("--device", type=str, default=pick_device())
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
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: results/<model-slug>/steer_audit.jsonl, where plot_steer_audit.py looks",
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = results_dir(args.model_name) / "steer_audit.jsonl"

    model, _ = load_model(args.model_name)

    # Built before the device move: get_token_activations feeds the model CPU tensors.
    steering = build_steering_vectors(load_pairs(args.train_path))

    model.to(args.device)
    print(f"model on {next(model.parameters()).device}")

    layers = (
        [int(s) for s in args.layers.split(",")]
        if args.layers
        else list(range(1, model.config.num_hidden_layers + 1))
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
