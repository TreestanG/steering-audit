"""SipIt — exact prompt recovery from hidden activations.

Nikolaou et al., "Language Models are Injective and Hence Invertible" (arXiv 2510.15511).
Causal masking makes h_t a function of s_1..s_t only, so each position is a
vocabulary scan given the pinned prefix. Injectivity means the first exact
match is the answer.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import Tensor

import utils
from utils import load_model

# Widen on a miss: true tokens usually rank near the top of next-token order.
DEFAULT_SCHEDULE: tuple[int, ...] = (32, 96, 384, 1536, 6144, 24576)

# Cap a candidate batch so expanding the prefix KV cache doesn't OOM.
MAX_BATCH = 4096
KV_BUDGET_BYTES = 6_000_000_000


def kv_bytes_per_token() -> int:
    """Bytes of KV cache one token costs for a single sequence."""
    model = utils.model
    if model is None:
        raise RuntimeError("Call load_model(...) first")
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * head_dim * 4


def batch_sizes(schedule: tuple[int, ...], vocab_size: int, prefix_len: int) -> list[int]:
    """Widening probe sizes, split so no single batch blows the KV memory budget."""
    cap = max(256, min(MAX_BATCH, KV_BUDGET_BYTES // max(1, kv_bytes_per_token() * prefix_len)))
    sizes: list[int] = []
    total = 0
    for size in (*schedule, vocab_size):
        while size > 0 and total < vocab_size:
            take = min(size, cap, vocab_size - total)
            sizes.append(take)
            size -= take
            total += take
        if total >= vocab_size:
            break
    return sizes


def match_tol(target: Tensor, rel_tol: float, abs_tol: float) -> float:
    """Acceptance threshold: abs_tol, or rel_tol * ||target|| (norms vary ~47x by depth)."""
    if abs_tol > 0:
        return abs_tol
    return rel_tol * float(target.norm())


def load_vocab_layer(path: str, layer: int) -> Tensor:
    """[vocab, hidden] solo-token states at one layer, mmap-sliced from the big table."""
    blob = torch.load(path, mmap=True, weights_only=False)
    return blob["activations"][:, layer, :].contiguous()


def dists_to(vec: Tensor, table: Tensor, chunk: int = 8192) -> Tensor:
    """L2 from vec [hidden] to every row of table [vocab, hidden], chunked to avoid a 545 MB broadcast."""
    out = torch.empty(table.shape[0])
    for start in range(0, table.shape[0], chunk):
        out[start : start + chunk] = (table[start : start + chunk] - vec).norm(dim=1)
    return out


class _EarlyExit(Exception):
    """Abort a forward once the inverted layer has run."""

    def __init__(self, value: Tensor):
        self.value = value


@torch.no_grad()
def encode_step(input_ids: Tensor, cache, attn_len: int):
    """Full-depth step extending `cache`. input_ids [1, k] → (new_cache, last hidden [1, hidden])."""
    model = utils.model
    if model is None:
        raise RuntimeError("Call load_model(...) first")
    device = next(model.parameters()).device
    attn = torch.ones(1, attn_len, dtype=torch.long, device=device)
    out = model.model(
        input_ids=input_ids.to(device), attention_mask=attn, past_key_values=cache, use_cache=True
    )
    return out.past_key_values, out.last_hidden_state[:, -1]


@torch.no_grad()
def next_token_logits(hidden_last: Tensor) -> Tensor:
    """lm_head on an already-computed final hidden state — no extra forward pass."""
    model = utils.model
    if model is None:
        raise RuntimeError("Call load_model(...) first")
    return model.lm_head(hidden_last)[0].float().cpu()


@torch.no_grad()
def candidate_states(cache, prefix_len: int, candidates: Tensor, layer: int) -> Tensor:
    """h_layer at the appended position for each candidate. [n_cands, hidden]"""
    model = utils.model
    if model is None:
        raise RuntimeError("Call load_model(...) first")
    device = next(model.parameters()).device
    n = candidates.shape[0]

    if layer == 0:
        return model.model.embed_tokens(candidates.to(device)).float().cpu()

    batch_cache = copy.deepcopy(cache)
    batch_cache.batch_repeat_interleave(n)

    handle = model.model.layers[layer - 1].register_forward_hook(
        lambda module, inp, out: (_ for _ in ()).throw(
            _EarlyExit(out[0] if isinstance(out, tuple) else out)
        )
    )
    try:
        model.model(
            input_ids=candidates.view(-1, 1).to(device),
            attention_mask=torch.ones(n, prefix_len + 1, dtype=torch.long, device=device),
            past_key_values=batch_cache,
        )
        raise RuntimeError(f"layer {layer} hook never fired — is layer <= n_layers?")
    except _EarlyExit as exit:
        h = exit.value[:, 0]
        # Last-layer hidden_states is post-norm; the hook fires before that.
        if layer == model.config.num_hidden_layers:
            h = model.model.norm(h)
        return h.float().cpu()
    finally:
        handle.remove()


@torch.no_grad()
def solve_position(
    cache,
    prefix_len: int,
    logits: Tensor,
    target: Tensor,
    layer: int,
    *,
    rel_tol: float,
    abs_tol: float,
    schedule: tuple[int, ...],
    exhaustive: bool,
) -> dict:
    """Find the token whose h_layer at this position equals target."""
    model = utils.model
    if model is None:
        raise RuntimeError("Call load_model(...) first")
    vocab_size = int(model.config.vocab_size)
    tol = match_tol(target, rel_tol, abs_tol)
    order = torch.argsort(logits[:vocab_size], descending=True)

    best_d, best_id, runner_d, tried = float("inf"), -1, float("inf"), 0
    start = 0
    for size in batch_sizes(schedule, vocab_size, prefix_len):
        if start >= vocab_size:
            break
        end = min(start + size, vocab_size)
        cands = order[start:end]
        d = (candidate_states(cache, prefix_len, cands, layer) - target).norm(dim=1)
        tried += cands.shape[0]

        top2 = torch.topk(d, k=min(2, d.shape[0]), largest=False)
        for dist, idx in zip(top2.values.tolist(), top2.indices.tolist()):
            if dist < best_d:
                runner_d, best_d, best_id = best_d, dist, int(cands[idx])
            elif dist < runner_d:
                runner_d = dist

        start = end
        if best_d <= tol and not exhaustive:
            break

    gap = runner_d - best_d
    return {
        "token": best_id,
        "residual": best_d,
        "runner_up": runner_d,  # true runner-up only when exhaustive
        "gap": gap,
        "ratio": best_d / gap if gap > 0 else float("inf"),  # Thm 3.2: recovery guaranteed if residual < gap/2
        "tol": tol,
        "tried": tried,
        "matched": best_d <= tol,
    }


@torch.no_grad()
def sipit(
    target: Tensor,
    layer: int,
    vocab_layer: Tensor,
    *,
    rel_tol: float = 1e-3,
    abs_tol: float = 0.0,
    schedule: tuple[int, ...] = DEFAULT_SCHEDULE,
    exhaustive: bool = False,
    stop_on_fail: bool = True,
) -> list[dict]:
    """Recover the token sequence behind target [seq, hidden] at one layer."""
    steps: list[dict] = []
    tol0 = match_tol(target[0], rel_tol, abs_tol)
    d0 = dists_to(target[0], vocab_layer)
    top2 = torch.topk(d0, k=2, largest=False)
    best = float(top2.values[0])
    token0 = int(top2.indices[0])
    gap0 = float(top2.values[1] - top2.values[0])
    steps.append(
        {
            "token": token0,
            "residual": best,
            "runner_up": float(top2.values[1]),
            "gap": gap0,
            "ratio": best / gap0 if gap0 > 0 else float("inf"),
            "tol": tol0,
            "tried": vocab_layer.shape[0],
            "matched": best <= tol0,
        }
    )
    if stop_on_fail and not steps[0]["matched"]:
        return steps
    if target.shape[0] == 1:
        return steps

    cache, hidden_last = encode_step(torch.tensor([[token0]]), cache=None, attn_len=1)
    logits = next_token_logits(hidden_last)
    prefix_len = 1

    for t in range(1, target.shape[0]):
        step = solve_position(
            cache, prefix_len, logits, target[t], layer,
            rel_tol=rel_tol, abs_tol=abs_tol, schedule=schedule, exhaustive=exhaustive,
        )
        steps.append(step)
        if stop_on_fail and not step["matched"]:
            break

        cache, hidden_last = encode_step(
            torch.tensor([[step["token"]]]), cache=cache, attn_len=prefix_len + 1
        )
        logits = next_token_logits(hidden_last)
        prefix_len += 1

    return steps


def invert_file(
    act_path: str,
    tokenizer,
    vocab_layer: Tensor,
    gold_by_id: dict[str, str],
    *,
    layer: int,
    rel_tol: float,
    abs_tol: float,
    schedule: tuple[int, ...],
    exhaustive: bool,
    stop_on_fail: bool,
    max_len: int,
    noise: float,
) -> dict:
    """Invert one saved activation file. Prints the table. Returns a JSON-serializable row."""
    blob = torch.load(act_path, weights_only=False)
    target = blob["activations"][layer]
    if max_len:
        target = target[:max_len]
    if noise:
        g = torch.Generator().manual_seed(0)
        v = torch.randn(target.shape[-1], generator=g)
        target = target + noise * v / v.norm()

    start = time.time()
    steps = sipit(
        target,
        layer,
        vocab_layer,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        schedule=schedule,
        exhaustive=exhaustive,
        stop_on_fail=stop_on_fail,
    )
    elapsed = time.time() - start

    print(f"\n{blob['id']}  layer {layer}  {len(steps)}/{target.shape[0]} positions in {elapsed:.1f}s")
    print(f"{'t':>3} {'token':>7} {'residual':>10} {'gap':>10} {'res/gap':>10} {'tried':>7}  text")
    for t, s in enumerate(steps):
        flag = "" if s["matched"] else "  <-- NO MATCH"
        print(
            f"{t:>3} {s['token']:>7} {s['residual']:>10.4g} {s['gap']:>10.4g} "
            f"{s['ratio']:>10.2e} {s['tried']:>7}  {tokenizer.decode(token_ids=[s['token']])!r}{flag}"
        )
    recovered = [s["token"] for s in steps]
    recovered_text = tokenizer.decode(token_ids=recovered)
    print(f"\nrecovered: {recovered_text!r}")
    print(f"max residual: {max(s['residual'] for s in steps):.6g}")

    exact = None
    gold_text = gold_by_id.get(blob["id"])
    if gold_text is not None:
        gold = tokenizer(gold_text, return_tensors="pt")["input_ids"][0].tolist()
        if max_len:
            gold = gold[:max_len]
        exact = recovered == gold
        print(f"exact: {exact}")

    return {
        "id": blob["id"],
        "category": blob.get("category"),
        "layer": layer,
        "n_target": int(target.shape[0]),
        "n_recovered": len(steps),
        "elapsed": elapsed,
        "max_residual": max(s["residual"] for s in steps),
        "max_ratio": max(s["ratio"] for s in steps),
        "recovered_ids": recovered,
        "recovered_text": recovered_text,
        "exact": exact,
        "steps": [
            {
                "token": s["token"],
                "residual": s["residual"],
                "gap": s["gap"],
                "ratio": s["ratio"],
                "tol": s["tol"],
                "tried": s["tried"],
                "matched": s["matched"],
            }
            for s in steps
        ],
    }


def n_layers_in(act_path: str) -> int:
    """Hidden-state count in a saved activation file (embed + each block)."""
    blob = torch.load(act_path, map_location="cpu", weights_only=False)
    return int(blob["activations"].shape[0])


def run_layer(
    act_paths: list[str],
    tokenizer,
    vocab_layer: Tensor,
    gold_by_id: dict[str, str],
    *,
    layer: int,
    out: Path | None,
    rel_tol: float,
    abs_tol: float,
    schedule: tuple[int, ...],
    exhaustive: bool,
    stop_on_fail: bool,
    max_len: int,
    noise: float,
) -> tuple[int, int]:
    """Invert every prompt at one layer. Returns (n_scored, n_exact)."""
    # Stream to a sibling .partial and rename only once every prompt succeeded.
    # Opening the destination directly would truncate a previous run's results
    # before the first forward pass — one bad --act_path then destroys them.
    out_f = None
    out_path = tmp_path = None
    if out is not None:
        out_path = out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
        out_f = tmp_path.open("w")

    ok = 0
    n = 0
    try:
        for i, act_path in enumerate(act_paths, start=1):
            if len(act_paths) > 1:
                print(f"======== [{i}/{len(act_paths)}] {Path(act_path).stem} ========")
            row = invert_file(
                act_path,
                tokenizer,
                vocab_layer,
                gold_by_id,
                layer=layer,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                schedule=schedule,
                exhaustive=exhaustive,
                stop_on_fail=stop_on_fail,
                max_len=max_len,
                noise=noise,
            )
            if out_f is not None:
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
            if row["exact"] is not None:
                n += 1
                ok += int(row["exact"])
    finally:
        if out_f is not None:
            out_f.close()

    if tmp_path is not None and out_path is not None:
        tmp_path.replace(out_path)  # atomic; prior results survive a crash above

    if n:
        print(f"\nexact {ok}/{n} ({ok * 100 // n}%)")
    if out_path is not None:
        print(f"wrote {out_path}")
    return n, ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--act_path",
        type=str,
        nargs="+",
        required=True,
        help="one or more .pt files from save_activations.py",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="data/activations/Qwen_Qwen2.5-0.5B-Instruct/vocab/vocab_table.pt",
    )
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--layer",
        type=int,
        default=None,
        help="hidden-state index (default: 12). incompatible with --all_layers",
    )
    layer_group.add_argument(
        "--all_layers",
        action="store_true",
        help="invert every layer in the activation tensor (embed through final-normed last block)",
    )
    parser.add_argument(
        "--rel_tol",
        type=float,
        default=1e-3,
        help="accept a candidate whose L2 to the target is below rel_tol * ||target||. "
        "Relative because activation norms grow ~47x from layer 1 to 24, so a fixed "
        "absolute threshold is a different standard at every depth.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="absolute L2 threshold, overriding --rel_tol. 0 = use --rel_tol. "
        "Only use this if you want one fixed threshold across layers on purpose.",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SCHEDULE),
        help="comma-separated widening candidate-batch sizes, e.g. '32,96,384,1536'",
    )
    parser.add_argument("--max_len", type=int, default=0, help="0 = whole sequence")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="add a random vector of this L2 norm to the target — a steering stand-in",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="scan all |V| per position — needed for true runner-up gaps",
    )
    parser.add_argument("--no_stop_on_fail", action="store_true")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/trajectory_bank_prompts.json",
        help="prompt bank used to score exact recovery against the original text",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="write one JSON object per prompt (JSONL). flushed after each row",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="with --all_layers: write sipit_layer_XX.jsonl per layer",
    )
    args = parser.parse_args()
    if args.all_layers and args.out:
        parser.error("--all_layers writes per-layer jsonl; use --out_dir, not --out")
    if args.out_dir and not args.all_layers:
        parser.error("--out_dir requires --all_layers")
    schedule = tuple(int(s) for s in args.schedule.split(","))

    _, tokenizer = load_model(args.model_name)
    if utils.model is None:
        raise RuntimeError("load_model(...) did not set utils.model")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required but torch.backends.mps.is_available() is False")
    utils.model.to("mps")
    print(f"model on {next(utils.model.parameters()).device}, dtype {next(utils.model.parameters()).dtype}")

    gold_by_id: dict[str, str] = {}
    bank = Path(args.data_path)
    if bank.exists():
        gold_by_id = {p["id"]: p["text"] for p in json.loads(bank.read_text())["prompts"]}

    run_kw = dict(
        rel_tol=args.rel_tol,
        abs_tol=args.tol,
        schedule=schedule,
        exhaustive=args.exhaustive,
        stop_on_fail=not args.no_stop_on_fail,
        max_len=args.max_len,
        noise=args.noise,
    )
    if args.all_layers:
        n_layers = n_layers_in(args.act_path[0])
        last = n_layers - 1
        out_dir = Path(args.out_dir) if args.out_dir else None
        print(f"sweeping layers 0..{last}" + (f" -> {out_dir}/sipit_layer_XX.jsonl" if out_dir else ""))
        total_ok = total_n = 0
        for layer in range(n_layers):
            out = out_dir / f"sipit_layer_{layer:02d}.jsonl" if out_dir else None
            print(f"======== layer {layer} / {last} -> {out} ========")
            vocab_layer = load_vocab_layer(args.vocab_path, layer)
            n, ok = run_layer(
                args.act_path, tokenizer, vocab_layer, gold_by_id,
                layer=layer, out=out, **run_kw,
            )
            total_n += n
            total_ok += ok
        if total_n:
            print(f"\nexact {total_ok}/{total_n} across {n_layers} layers")
        return

    layer = 12 if args.layer is None else args.layer
    vocab_layer = load_vocab_layer(args.vocab_path, layer)
    run_layer(
        args.act_path,
        tokenizer,
        vocab_layer,
        gold_by_id,
        layer=layer,
        out=Path(args.out) if args.out else None,
        **run_kw,
    )


if __name__ == "__main__":
    main()
