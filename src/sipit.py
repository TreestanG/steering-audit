import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from log import add_logging_args, get_logger
from log import setup as log_setup
from paths import experiment_dir, logs_dir, vocab_table_path
from utils import (
    DTYPES,
    add_model_args,
    apply_final_norm,
    get_base_model,
    get_decoder_layers,
    load_model,
    model_device,
    rel_tol_for,
    require_model,
)

# Widen on a miss: true tokens usually rank near the top of next-token order.
DEFAULT_SCHEDULE: tuple[int, ...] = (32, 96, 384, 1536, 6144, 24576)

logger = get_logger(__name__)

# Cap a candidate batch so expanding the prefix KV cache doesn't OOM.
# The budget counts the expanded prefix cache only -- model weights, the deepcopy
# made before batch_repeat_interleave, and output_hidden_states all sit on top of
# it, so the real peak is roughly 2x this plus the weights. 6 GB suits a machine
# with headroom; MHA models (Pythia: no GQA, so 3.4x the KV/token of Qwen-7B) at
# fp32 on 24 GB need it lower. AAT_KV_BUDGET overrides it in bytes.
MAX_BATCH = 4096
KV_BUDGET_BYTES = int(os.environ.get("AAT_KV_BUDGET") or 6_000_000_000)


def kv_bytes_per_token() -> int:
    """Bytes of KV cache one token costs for a single sequence."""
    model, _ = require_model()
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    return cfg.num_hidden_layers * 2 * kv_heads * head_dim * 4


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


def rel_tol_at(rel_tol, layer: int) -> float:
    """rel_tol is one float for every layer, or a per-layer dict with a 'base' fallback."""
    if isinstance(rel_tol, dict):
        return rel_tol.get(layer, rel_tol["base"])
    return rel_tol


def load_rel_tol_by_layer(path, base: float) -> dict:
    d = json.loads(Path(path).read_text())
    out = {int(k): float(v) for k, v in d.get("layers", d).items()}
    out["base"] = base
    return out


# --stop_on: 'miss' ends a scan at the first position over tolerance (the deployment
# rule, where gold is unknown); 'wrong' ends it only where the recovered token is known
# to differ from gold, so a correct token over the numerics floor -- Qwen's massive
# first-token state collapsing in the last block -- no longer truncates a clean-bank or
# stage-2 row; 'never' scans every position regardless.
STOP_MODES: dict[str, bool | str] = {"miss": True, "wrong": "wrong", "never": False}


def stop_mode_name(stop_on_fail: bool | str) -> str:
    return next(k for k, v in STOP_MODES.items() if v == stop_on_fail)


def _halts(step: dict, stop_on_fail: bool | str) -> bool:
    if not stop_on_fail:
        return False
    if stop_on_fail == "wrong":
        return step.get("correct") is False
    return not step["matched"]


def load_vocab_table(
    path: str, expect_model: str | None = None, expect_dtype: str | None = None
) -> tuple[Tensor, str]:
    """Solo-token states, mmapped so pages load on demand.

    Returns (table, layout). Tables written before the layout key are vocab-major
    ([vocab, n_layers, hidden]); new ones are layer-major ([n_layers, vocab, hidden]).

    Refuses a table built for a different model: the shapes are often compatible,
    so the mismatch would otherwise surface as silently wrong inversions.
    """
    if not Path(path).exists():
        raise SystemExit(
            f"no vocab table at {path}\n"
            f"Build it: uv run src/vocab_activation_table.py --model_name {expect_model or '<model>'}"
        )
    blob = torch.load(path, mmap=True, weights_only=False)
    # Tables built before --dtype existed are fp32: load_model hardcoded it.
    for field, want, flag, fallback in (("model_name", expect_model, "--model_name", None),
                                        ("dtype", expect_dtype, "--dtype", "float32")):
        built = blob.get(field, fallback)
        if want and built and built != want:
            raise SystemExit(
                f"{path} was built with {field}={built!r}, not {want!r}.\n"
                f"Rebuild it: uv run src/vocab_activation_table.py "
                f"--model_name {expect_model} --dtype {expect_dtype} "
                f"(mismatched {flag} makes every residual meaningless)"
            )
    return blob["activations"], str(blob.get("layout", "vocab_major"))


def load_vocab_layer(table: Tensor, layer: int, layout: str = "vocab_major") -> Tensor:
    """[vocab, hidden] slice at one layer, in fp32.

    Layer-major slices are one contiguous read; vocab-major ones stride the whole
    file, which is what made large tables unusable. The table is stored at the
    model's dtype and distances are computed in fp32, so the cast happens once
    here rather than being left to type promotion inside the chunked scan.
    """
    rows = table[layer] if layout == "layer_major" else table[:, layer, :]
    return rows.float().contiguous()


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


def abs_position_embedding() -> torch.nn.Embedding | None:
    """The learned absolute position embedding, if this architecture uses one.

    Rotary and ALiBi models inject position inside attention, so their layer-0
    state is the token embedding alone. GPT-2 style models instead add wpe[pos]
    to it, and dropping that term leaves every layer-0 candidate off by exactly
    ||wpe[pos]|| -- a token-independent error that shifts all candidates equally,
    so the scan still ranks them sensibly but no candidate ever clears tol.
    """
    base = get_base_model()
    for attr in ("wpe", "position_embeddings"):
        mod = getattr(base, attr, None)
        if isinstance(mod, torch.nn.Embedding):
            return mod
    return None


@torch.no_grad()
def candidate_states(cache, prefix_len: int, candidates: Tensor, layer: int) -> Tensor:
    """h_layer at the appended position for each candidate. [n_cands, hidden]"""
    model, _ = require_model()
    device = next(model.parameters()).device
    n = candidates.shape[0]

    if layer == 0:
        h = model.get_input_embeddings()(candidates.to(device))
        pos = abs_position_embedding()
        if pos is not None:  # candidate sits at index prefix_len, right after the prefix
            h = h + pos.weight[prefix_len].to(h.dtype)
        return h.float().cpu()

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


@torch.no_grad()
def candidate_states_multi(cache, prefix_len: int, candidates: Tensor,
                           layers: list[int]) -> dict[int, Tensor]:
    """candidate_states at several layers from ONE forward, exiting after the deepest.

    The pass through block L is the same computation whether it exits there or is
    read on the way to a deeper block, so each layer's states are the ones its own
    scan would have produced. {layer: [n_cands, hidden]}
    """
    model, _ = require_model()
    device = next(model.parameters()).device
    n = candidates.shape[0]
    out: dict[int, Tensor] = {}
    if 0 in layers:
        h = model.get_input_embeddings()(candidates.to(device))
        pos = abs_position_embedding()
        if pos is not None:
            h = h + pos.weight[prefix_len].to(h.dtype)
        out[0] = h.float().cpu()
    deep = sorted(l for l in layers if l > 0)
    if not deep:
        return out
    deepest = deep[-1]
    batch_cache = copy.deepcopy(cache)
    batch_cache.batch_repeat_interleave(n)
    blocks = get_decoder_layers()
    grabbed: dict[int, Tensor] = {}

    def grab_for(layer: int):
        def grab(module, args, output):
            grabbed[layer] = output[0] if isinstance(output, tuple) else output
            if layer == deepest:
                raise _EarlyExit(grabbed[layer])
        return grab

    handles = [blocks[l - 1].register_forward_hook(grab_for(l)) for l in deep]
    try:
        get_base_model()(
            input_ids=candidates.view(-1, 1).to(device),
            attention_mask=torch.ones(n, prefix_len + 1, dtype=torch.long, device=device),
            past_key_values=batch_cache,
        )
        raise RuntimeError(f"layer {deepest} hook never fired — is layer <= n_layers?")
    except _EarlyExit:
        for l in deep:
            out[l] = apply_final_norm(grabbed[l][:, 0], l).float().cpu()
    finally:
        for h in handles:
            h.remove()
    return out


@torch.no_grad()
def prefix_cache(tokens: list[int]):
    """Cache and next-token logits after encoding tokens one at a time -- the same
    steps the scan takes, so a continuation from here is what a full run produces."""
    cache, hidden_last = None, None
    for q, tok in enumerate(tokens):
        cache, hidden_last = encode_step(torch.tensor([[tok]]), cache=cache, attn_len=q + 1)
    return cache, next_token_logits(hidden_last)


@torch.no_grad()
def solve_position_multi(
    cache,
    prefix_len: int,
    logits: Tensor,
    targets: dict[int, Tensor],
    *,
    rel_tol: float,
    abs_tol: float,
    schedule: tuple[int, ...],
    exhaustive: bool,
    gold: int | None = None,
) -> dict[int, dict]:
    """solve_position at several layers of one position from a single candidate scan.

    The candidate order and batch boundaries depend only on the prefix, so every
    layer sees exactly the batches its own scan would have seen and closes on the
    batch it would have closed on; a layer that never matches goes exhaustive. Each
    batch is one forward through the deepest layer still open.
    """
    model, _ = require_model()
    vocab_size = int(model.config.vocab_size)
    order = torch.argsort(logits[:vocab_size], descending=True)
    tol = {l: match_tol(t, rel_tol_at(rel_tol, l), abs_tol) for l, t in targets.items()}
    top2 = {l: Top2() for l in targets}
    tried = {l: 0 for l in targets}
    open_layers = set(targets)
    start = 0
    for size in batch_sizes(schedule, vocab_size, prefix_len):
        if start >= vocab_size or not open_layers:
            break
        end = min(start + size, vocab_size)
        cands = order[start:end]
        states = candidate_states_multi(cache, prefix_len, cands, sorted(open_layers))
        for l in sorted(open_layers):
            d = (states[l] - targets[l]).norm(dim=1)
            tried[l] += cands.shape[0]
            top2[l].update(d, cands)
            if top2[l].best <= tol[l] and not exhaustive:
                open_layers.discard(l)
        start = end
    return {l: _step(top2[l], tol=tol[l], h_norm=float(targets[l].norm()), tried=tried[l],
                     exhaustive=tried[l] >= vocab_size, gold=gold) for l in targets}


@torch.no_grad()
def sipit_multi(
    targets: dict[int, Tensor],
    vocab_layer_for,
    *,
    rel_tol: float = 1e-3,
    abs_tol: float = 0.0,
    schedule: tuple[int, ...] = DEFAULT_SCHEDULE,
    exhaustive: bool = False,
    stop_on_fail: bool | str = True,
    gold: list[int] | None = None,
) -> tuple[dict[int, list[dict]], set[int]]:
    """sipit() at several layers of ONE sequence from a shared scan.

    Every layer's cache is built from the tokens it recovered, and while the layers
    agree on those tokens the cache, the candidate order and the batch boundaries
    are identical, so each position is solved for all still-open layers by one
    forward per batch (solve_position_multi). A layer stops where its own scan
    would have stopped. A layer that recovers a DIFFERENT token from the others is
    dropped from the shared scan; its steps so far are returned as a valid prefix
    and the caller finishes it with sipit(known_steps=...). Returns (steps per
    layer, the set of layers that diverged).
    """
    layers = sorted(targets)
    seq_len = int(next(iter(targets.values())).shape[0])
    if any(int(t.shape[0]) != seq_len for t in targets.values()):
        raise ValueError("sipit_multi: every layer's target must cover the same positions")

    def gold_at(t: int) -> int | None:
        return gold[t] if gold is not None and t < len(gold) else None

    steps: dict[int, list[dict]] = {}
    for l in layers:
        table = vocab_layer_for(l)
        top2 = Top2()
        top2.update(dists_to(targets[l][0], table), torch.arange(table.shape[0]))
        steps[l] = [_step(top2, tol=match_tol(targets[l][0], rel_tol_at(rel_tol, l), abs_tol),
                          h_norm=float(targets[l][0].norm()), tried=table.shape[0],
                          exhaustive=True, gold=gold_at(0))]
        del table
    open_layers = {l for l in layers if not _halts(steps[l][0], stop_on_fail)}
    diverged: set[int] = set()
    if seq_len == 1 or not open_layers:
        return steps, diverged

    def split_on_token(t: int) -> int | None:
        ref = steps[min(open_layers)][t]["token"]
        for l in sorted(open_layers):
            if steps[l][t]["token"] != ref:
                open_layers.discard(l)
                diverged.add(l)
        return ref

    token = split_on_token(0)
    cache, hidden_last = encode_step(torch.tensor([[token]]), cache=None, attn_len=1)
    logits = next_token_logits(hidden_last)
    prefix_len = 1
    for t in range(1, seq_len):
        if not open_layers:
            break
        res = solve_position_multi(
            cache, prefix_len, logits, {l: targets[l][t] for l in open_layers},
            rel_tol=rel_tol, abs_tol=abs_tol, schedule=schedule, exhaustive=exhaustive,
            gold=gold_at(t),
        )
        for l in sorted(open_layers):
            steps[l].append(res[l])
            if _halts(res[l], stop_on_fail):
                open_layers.discard(l)
        if not open_layers:
            break
        token = split_on_token(t)
        if not open_layers:
            break
        cache, hidden_last = encode_step(
            torch.tensor([[token]]), cache=cache, attn_len=prefix_len + 1
        )
        logits = next_token_logits(hidden_last)
        prefix_len += 1
    return steps, diverged


def _step(top2: Top2, *, tol: float, h_norm: float, tried: int, exhaustive: bool,
          gold: int | None = None) -> dict:
    """One position's result row, shared by the position-0 lookup and the vocab scan."""
    gap = top2.gap
    step = {
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
    # matched is the alarm; correct is the truth. They can disagree.
    if gold is not None:
        step["gold_token"] = gold
        step["correct"] = top2.best_id == gold
    return step


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
    gold: int | None = None,
) -> dict:
    """Find the token whose h_layer at this position equals target."""
    model, _ = require_model()
    vocab_size = int(model.config.vocab_size)
    tol = match_tol(target, rel_tol_at(rel_tol, layer), abs_tol)
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
                 exhaustive=tried >= vocab_size, gold=gold)


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
    stop_on_fail: bool | str = True,
    gold: list[int] | None = None,
    known_steps: list[dict] | None = None,
) -> list[dict]:
    """Recover the token sequence behind target [seq, hidden] at one layer.

    known_steps: rows already solved for positions 0..len-1 of this same sequence,
    e.g. the clean inversion of a prefix that an intervention did not touch. They are
    reused verbatim and the KV cache is rebuilt from their tokens by the same
    one-token steps the scan takes, so the continuation is what a full run produces.
    """
    steps: list[dict] = []
    n_vocab = vocab_layer.shape[0]

    def gold_at(t: int) -> int | None:
        return gold[t] if gold is not None and t < len(gold) else None

    if known_steps:
        steps = [dict(s) for s in known_steps[: target.shape[0]]]
        for q, s in enumerate(steps):
            if _halts(s, stop_on_fail):
                return steps[: q + 1]
        if len(steps) >= target.shape[0]:
            return steps
        cache, logits = prefix_cache([s["token"] for s in steps])
        prefix_len = len(steps)
        return _continue(steps, cache, prefix_len, logits, target, layer, rel_tol=rel_tol,
                         abs_tol=abs_tol, schedule=schedule, exhaustive=exhaustive,
                         stop_on_fail=stop_on_fail, gold_at=gold_at)

    top2 = Top2()
    top2.update(dists_to(target[0], vocab_layer), torch.arange(n_vocab))
    steps.append(
        _step(
            top2,
            tol=match_tol(target[0], rel_tol_at(rel_tol, layer), abs_tol),
            h_norm=float(target[0].norm()),
            tried=n_vocab,
            exhaustive=True,  # position 0 is a full table lookup
            gold=gold_at(0),
        )
    )
    token0 = steps[0]["token"]
    if _halts(steps[0], stop_on_fail):
        return steps
    if target.shape[0] == 1:
        return steps

    cache, hidden_last = encode_step(torch.tensor([[token0]]), cache=None, attn_len=1)
    logits = next_token_logits(hidden_last)
    return _continue(steps, cache, 1, logits, target, layer, rel_tol=rel_tol,
                     abs_tol=abs_tol, schedule=schedule, exhaustive=exhaustive,
                     stop_on_fail=stop_on_fail, gold_at=gold_at)


@torch.no_grad()
def _continue(steps, cache, prefix_len, logits, target, layer, *, rel_tol, abs_tol,
              schedule, exhaustive, stop_on_fail, gold_at) -> list[dict]:
    """Solve positions len(steps).. given the cache and logits of the recovered prefix."""
    for t in range(len(steps), target.shape[0]):
        step = solve_position(
            cache, prefix_len, logits, target[t], layer,
            rel_tol=rel_tol, abs_tol=abs_tol, schedule=schedule, exhaustive=exhaustive,
            gold=gold_at(t),
        )
        steps.append(step)
        if _halts(step, stop_on_fail):
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
    stop_on_fail: bool | str,
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

    gold = None
    gold_text = gold_by_id.get(blob["id"])
    if gold_text is not None:
        gold = tokenizer(gold_text, return_tensors="pt")["input_ids"][0].tolist()
        if max_len:
            gold = gold[:max_len]

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
        gold=gold,
    )
    elapsed = time.time() - start

    logger.debug("%s  layer %d  %d/%d positions in %.1fs",
                 blob["id"], layer, len(steps), target.shape[0], elapsed)
    logger.debug("%3s %7s %10s %10s %10s %7s  text",
                 "t", "token", "residual", "gap", "res/gap", "tried")
    for t, s in enumerate(steps):
        flag = "" if s["matched"] else "  <-- NO MATCH"
        logger.debug(
            "%3d %7d %10.4g %10.4g %10.2e %7d  %r%s",
            t, s["token"], s["residual"], s["gap"], s["ratio"], s["tried"],
            tokenizer.decode(token_ids=[s["token"]]), flag,
        )
    recovered = [s["token"] for s in steps]
    recovered_text = tokenizer.decode(token_ids=recovered)
    max_residual = max(s["residual"] for s in steps)
    logger.debug("max residual: %.6g", max_residual)

    exact = None if gold is None else recovered == gold
    first_fail = next((i for i, s in enumerate(steps) if not s["matched"]), None)
    first_wrong = next((i for i, s in enumerate(steps) if not s.get("correct", True)), None)

    # One line per prompt is the right granularity for a stage that runs for
    # hours; the recovered text is only interesting when it went wrong.
    verdict = "exact" if exact else ("MISS" if exact is False else "unscored")
    logger.info(
        "  %-18s %3d/%-3d pos %7.1fs  max_res %.2e  %s",
        blob["id"], len(steps), target.shape[0], elapsed, max_residual, verdict,
    )
    if exact is False:
        logger.info("      recovered: %r", recovered_text)
    else:
        logger.debug("recovered: %r", recovered_text)

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
        "first_fail": first_fail,
        "first_wrong": first_wrong,
        "stop_on": stop_mode_name(stop_on_fail),
        "silent_corruption": (first_wrong is not None
                              and (first_fail is None or first_wrong < first_fail)),
        "steps": [
            {
                k: s[k] for k in
                ("token", "residual", "gap", "ratio", "gap_exhaustive", "tol",
                 "h_norm", "tried", "matched", "gold_token", "correct")
                if k in s
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
    stop_on_fail: bool | str,
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
                logger.debug("[%d/%d] %s", i, len(act_paths), Path(act_path).stem)
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

    if n or out is not None:
        tally = f"exact {ok}/{n} ({ok * 100 // n}%)" if n else "unscored"
        logger.info("  %s%s", tally, f" -> {out.name}" if out is not None else "")
    return n, ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    add_model_args(parser)
    add_logging_args(parser)
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
        default=None,
        help="default: derived from --model_name",
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
        default=None,
        help="accept a candidate whose L2 to the target is below rel_tol * ||target||. "
        "Relative because activation norms grow ~47x from layer 1 to 24, so a fixed "
        "absolute threshold is a different standard at every depth. "
        "Default follows --dtype: fp32 1e-3, fp16 1e-2, bf16 5e-2 -- below the dtype's "
        "own noise floor every position reports NO MATCH despite recovering the token.",
    )
    parser.add_argument(
        "--rel_tol_by_layer",
        type=str,
        default=None,
        help="JSON with a per-layer relative tolerance ({'layers': {layer: rel_tol}}), e.g. "
        "scripts/deploy_tolerance.py's output; layers it omits use --rel_tol",
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
    parser.add_argument("--stop_on", type=str, default="miss", choices=list(STOP_MODES),
                        help="miss (default): stop at the first position over tolerance. "
                             "wrong: stop only where the recovered token differs from the "
                             "bank's gold text, so a correct token over the fp16 floor "
                             "(Qwen's first-token state collapsing in the last block) does "
                             "not truncate the row. never: scan every position")
    parser.add_argument("--no_stop_on_fail", action="store_true", help="alias for --stop_on never")
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
        help="default: results/<slug>/sipit/sipit.jsonl",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="default: results/<slug>/sipit/layers/",
    )
    args = parser.parse_args()
    log_setup(args, default_log=logs_dir(args.model_name) / "sipit.log")
    if args.all_layers and args.out:
        parser.error("--all_layers writes per-layer jsonl; use --out_dir, not --out")
    if args.out_dir and not args.all_layers:
        parser.error("--out_dir requires --all_layers")
    sipit_dir = experiment_dir(args.model_name, "sipit")
    if args.all_layers:
        args.out_dir = args.out_dir or str(sipit_dir / "layers")
    else:
        args.out = args.out or str(sipit_dir / "sipit.jsonl")
    schedule = tuple(int(s) for s in args.schedule.split(","))

    vocab_path = args.vocab_path or str(vocab_table_path(args.model_name))
    dtype = DTYPES[args.dtype]
    rel_tol = rel_tol_for(dtype) if args.rel_tol is None else float(args.rel_tol)
    if args.rel_tol_by_layer:
        rel_tol = load_rel_tol_by_layer(args.rel_tol_by_layer, rel_tol)
    model, tokenizer = load_model(args.model_name, dtype=dtype, device=args.device)
    stop_on = "never" if args.no_stop_on_fail else args.stop_on
    logger.info("model on %s, %s, rel_tol %s, stop_on %s", model_device(), args.dtype,
                f"per layer from {args.rel_tol_by_layer}" if args.rel_tol_by_layer else rel_tol,
                stop_on)

    gold_by_id: dict[str, str] = {}
    bank = Path(args.data_path)
    if bank.exists():
        gold_by_id = {p["id"]: p["text"] for p in json.loads(bank.read_text())["prompts"]}

    run_kw: dict[str, Any] = dict(
        rel_tol=rel_tol,
        abs_tol=args.tol,
        schedule=schedule,
        exhaustive=args.exhaustive,
        stop_on_fail=STOP_MODES[stop_on],
        max_len=args.max_len,
        noise=args.noise,
    )
    # Opened once: re-loading per layer re-mmaps the whole table and re-materializes
    # a fresh [vocab, hidden] slice each time, for no gain over letting the page
    # cache serve the same mapping.
    vocab_table, vocab_layout = load_vocab_table(vocab_path, expect_model=args.model_name,
                                                 expect_dtype=args.dtype)
    if vocab_layout != "layer_major":
        logger.warning("%s is vocab-major: each layer read strides the whole file. "
                       "Rebuild for large models.", vocab_path)

    if args.all_layers:
        n_layers = n_layers_in(args.act_path[0])
        last = n_layers - 1
        out_dir = Path(args.out_dir) if args.out_dir else None
        logger.info("sweeping layers 0..%d%s", last,
                    f" -> {out_dir}/sipit_layer_XX.jsonl" if out_dir else "")
        total_ok = total_n = 0
        for layer in range(n_layers):
            out = out_dir / f"sipit_layer_{layer:02d}.jsonl" if out_dir else None
            logger.info("layer %02d/%d", layer, last)
            vocab_layer = load_vocab_layer(vocab_table, layer, vocab_layout)
            n, ok = run_layer(
                args.act_path, tokenizer, vocab_layer, gold_by_id,
                layer=layer, out=out, **run_kw,
            )
            total_n += n
            total_ok += ok
        if total_n:
            logger.info("exact %d/%d across %d layers", total_ok, total_n, n_layers)
        return

    layer = 12 if args.layer is None else args.layer
    vocab_layer = load_vocab_layer(vocab_table, layer, vocab_layout)
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
