import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import Tensor

from utils import apply_final_norm, get_base_model, get_decoder_layers, load_model, require_model

# Widen on a miss: true tokens usually rank near the top of next-token order.
DEFAULT_SCHEDULE: tuple[int, ...] = (32, 96, 384, 1536, 6144, 24576)

# Cap a candidate batch so expanding the prefix KV cache doesn't OOM.
MAX_BATCH = 4096
KV_BUDGET_BYTES = 6_000_000_000


def kv_bytes_per_token() -> int:
    """Bytes of KV cache one token costs for a single sequence."""
    model, _ = require_model()
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * head_dim * 4


def batch_cap(prefix_len: int, max_batch: int = MAX_BATCH) -> int:
    """Largest candidate batch whose expanded prefix KV cache stays inside the budget."""
    return max(256, min(max_batch, KV_BUDGET_BYTES // max(1, kv_bytes_per_token() * prefix_len)))


def batch_sizes(schedule: tuple[int, ...], vocab_size: int, prefix_len: int) -> list[int]:
    """Widening probe sizes, split so no single batch blows the KV memory budget."""
    cap = batch_cap(prefix_len)
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


def load_vocab_table(path: str) -> Tensor:
    """[vocab, n_layers, hidden] solo-token states, mmapped so pages load on demand."""
    return torch.load(path, mmap=True, weights_only=False)["activations"]


def load_vocab_layer(table: Tensor, layer: int) -> Tensor:
    """[vocab, hidden] slice at one layer. Materializes ~‖V‖x hidden floats."""
    return table[:, layer, :].contiguous()


def dists_to(vec: Tensor, table: Tensor, chunk: int = 8192) -> Tensor:
    """L2 from vec [hidden] to every row of table [vocab, hidden], chunked to avoid a 545 MB broadcast."""
    out = torch.empty(table.shape[0])
    for start in range(0, table.shape[0], chunk):
        out[start : start + chunk] = (table[start : start + chunk] - vec).norm(dim=1)
    return out


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


class _EarlyExit(Exception):
    """Abort a forward once the inverted layer has run."""

    def __init__(self, value: Tensor):
        self.value = value


@torch.no_grad()
def encode_step(input_ids: Tensor, cache, attn_len: int):
    """Full-depth step extending `cache`. input_ids [1, k] → (new_cache, last hidden [1, hidden])."""
    model, _ = require_model()
    device = next(model.parameters()).device
    attn = torch.ones(1, attn_len, dtype=torch.long, device=device)
    out = get_base_model()(
        input_ids=input_ids.to(device), attention_mask=attn, past_key_values=cache, use_cache=True
    )
    return out.past_key_values, out.last_hidden_state[:, -1]


@torch.no_grad()
def next_token_logits(hidden_last: Tensor) -> Tensor:
    """lm_head on an already-computed final hidden state — no extra forward pass."""
    model, _ = require_model()
    head = model.get_output_embeddings()
    if head is None:
        raise AttributeError(f"{type(model).__name__} has no output embedding layer")
    return head(hidden_last)[0].float().cpu()


@torch.no_grad()
def candidate_states(cache, prefix_len: int, candidates: Tensor, layer: int) -> Tensor:
    """h_layer at the appended position for each candidate. [n_cands, hidden]"""
    model, _ = require_model()
    device = next(model.parameters()).device
    n = candidates.shape[0]

    if layer == 0:
        return model.get_input_embeddings()(candidates.to(device)).float().cpu()

    batch_cache = copy.deepcopy(cache)
    batch_cache.batch_repeat_interleave(n)

    handle = get_decoder_layers()[layer - 1].register_forward_hook(
        lambda module, inp, out: (_ for _ in ()).throw(
            _EarlyExit(out[0] if isinstance(out, tuple) else out)
        )
    )
    try:
        get_base_model()(
            input_ids=candidates.view(-1, 1).to(device),
            attention_mask=torch.ones(n, prefix_len + 1, dtype=torch.long, device=device),
            past_key_values=batch_cache,
        )
        raise RuntimeError(f"layer {layer} hook never fired — is layer <= n_layers?")
    except _EarlyExit as exit:
        # Last-layer hidden_states is post-norm; the hook fires before that.
        h = apply_final_norm(exit.value[:, 0], layer)
        return h.float().cpu()
    finally:
        handle.remove()


def _step(top2: Top2, *, tol: float, h_norm: float, tried: int, exhaustive: bool) -> dict:
    """One position's result row, shared by the position-0 lookup and the vocab scan."""
    gap = top2.gap
    return {
        "token": top2.best_id,
        "residual": top2.best,
        "runner_up": top2.runner,
        "gap": gap,
        # Thm 3.2: recovery guaranteed if residual < gap/2.
        "ratio": top2.best / gap if gap > 0 else float("inf"),
        # A partial scan's runner-up can only be farther than the true one, so gap
        # is an over-estimate and ratio an under-estimate unless this is set.
        # Downstream plots need it to know whether the Thm 3.2 margin is real.
        "gap_exhaustive": exhaustive,
        "tol": tol,
        "h_norm": h_norm,  # so ‖h‖ survives an absolute --tol run
        "tried": tried,
        "matched": top2.best <= tol,
    }


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
    model, _ = require_model()
    vocab_size = int(model.config.vocab_size)
    tol = match_tol(target, rel_tol, abs_tol)
    order = torch.argsort(logits[:vocab_size], descending=True)

    top2 = Top2()
    tried = 0
    start = 0
    for size in batch_sizes(schedule, vocab_size, prefix_len):
        if start >= vocab_size:
            break
        end = min(start + size, vocab_size)
        cands = order[start:end]
        d = (candidate_states(cache, prefix_len, cands, layer) - target).norm(dim=1)
        tried += cands.shape[0]
        top2.update(d, cands)

        start = end
        if top2.best <= tol and not exhaustive:
            break

    return _step(top2, tol=tol, h_norm=float(target.norm()), tried=tried,
                 exhaustive=tried >= vocab_size)


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
    n_vocab = vocab_layer.shape[0]
    top2 = Top2()
    top2.update(dists_to(target[0], vocab_layer), torch.arange(n_vocab))
    steps.append(
        _step(
            top2,
            tol=match_tol(target[0], rel_tol, abs_tol),
            h_norm=float(target[0].norm()),
            tried=n_vocab,
            exhaustive=True,  # position 0 is a full table lookup
        )
    )
    token0 = steps[0]["token"]
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
    blob = torch.load(act_path, mmap=True, weights_only=False)
    target = blob["activations"][layer]
    if max_len:
        target = target[:max_len]
    if noise:
        g = torch.Generator().manual_seed(0)
        v = torch.randn(target.shape[-1], generator=g)
        # t=0 is scored against the exact vocab table, so any perturbation there
        # fails to match and stop_on_fail ends the run before a single forward
        # pass. Perturb t>0 only, which is also where a steer would land.
        target = torch.cat([target[:1], target[1:] + noise * v / v.norm()])

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
                "gap_exhaustive": s["gap_exhaustive"],
                "tol": s["tol"],
                "h_norm": s["h_norm"],
                "tried": s["tried"],
                "matched": s["matched"],
            }
            for s in steps
        ],
    }


def n_layers_in(act_path: str) -> int:
    """Hidden-state count in a saved activation file (embed + each block)."""
    blob = torch.load(act_path, mmap=True, weights_only=False)
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
    out_f = tmp_path = None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out.with_suffix(out.suffix + ".partial")
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

    if tmp_path is not None and out is not None:
        tmp_path.replace(out)  # atomic; prior results survive a crash above

    if n:
        print(f"\nexact {ok}/{n} ({ok * 100 // n}%)")
    if out is not None:
        print(f"wrote {out}")
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
        help="add a random vector of this L2 norm to target positions t>0 — a "
        "steering stand-in. t=0 is left clean because it is matched against the "
        "exact vocab table, so perturbing it just trips --stop_on_fail.",
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

    model, tokenizer = load_model(args.model_name)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required but torch.backends.mps.is_available() is False")
    model.to("mps")
    print(f"model on {next(model.parameters()).device}, dtype {next(model.parameters()).dtype}")

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
    # Opened once: re-loading per layer re-mmaps the whole table and re-materializes
    # a fresh [vocab, hidden] slice each time, for no gain over letting the page
    # cache serve the same mapping.
    vocab_table = load_vocab_table(args.vocab_path)

    if args.all_layers:
        n_layers = n_layers_in(args.act_path[0])
        last = n_layers - 1
        out_dir = Path(args.out_dir) if args.out_dir else None
        print(f"sweeping layers 0..{last}" + (f" -> {out_dir}/sipit_layer_XX.jsonl" if out_dir else ""))
        total_ok = total_n = 0
        for layer in range(n_layers):
            out = out_dir / f"sipit_layer_{layer:02d}.jsonl" if out_dir else None
            print(f"======== layer {layer} / {last} -> {out} ========")
            vocab_layer = load_vocab_layer(vocab_table, layer)
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
    vocab_layer = load_vocab_layer(vocab_table, layer)
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
