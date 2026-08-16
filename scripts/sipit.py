"""SipIt — exact prompt recovery from hidden activations.

Nikolaou et al., "Language Models are Injective and Hence Invertible" (arXiv 2510.15511).

Causal masking means h_t is a function of s_1..s_t only, so once s_1..s_{t-1} are
pinned the single free variable in h_t is s_t and you scan the vocabulary for it
directly. Injectivity makes the scan safe: exactly one token reproduces h_t, so
the first exact match is the answer and you never backtrack. O(N|V|) forward
passes instead of a |V|^N search.

The vocabulary table only answers position 0, where the prefix is empty and h_0
IS the token's solo activation. Every later position needs a real forward pass
over prefix + candidate.
"""

import argparse
import time

import torch
from torch import Tensor

import utils
from utils import load_model


def load_vocab_layer(path: str, layer: int) -> Tensor:
    """[vocab, hidden] solo-token states at one layer, mmap-sliced from the big table."""
    blob = torch.load(path, mmap=True, weights_only=False)
    return blob["activations"][:, layer, :].contiguous()


def dists_to(vec: Tensor, table: Tensor, chunk: int = 8192) -> Tensor:
    """L2 from vec [hidden] to every row of table [vocab, hidden].

    Chunked because table - vec would materialise another 545 MB.
    """
    out = torch.empty(table.shape[0])
    for start in range(0, table.shape[0], chunk):
        out[start : start + chunk] = (table[start : start + chunk] - vec).norm(dim=1)
    return out


@torch.no_grad()
def prefix_pass(prefix_ids: Tensor):
    """One forward over the known prefix. Returns (kv cache, next-token logits)."""
    model = utils.model
    if model is None:
        raise RuntimeError("Call load_model(...) first")
    device = next(model.parameters()).device
    out = model(input_ids=prefix_ids.view(1, -1).to(device), use_cache=True)
    return out.past_key_values, out.logits[0, -1].float().cpu()


@torch.no_grad()
def candidate_states(prefix_ids: Tensor, candidates: Tensor, layer: int) -> Tensor:
    """h_layer at the appended position, for each candidate token. [n_cands, hidden]

    The prefix is encoded once into a KV cache and replayed across the batch, so a
    candidate costs one single-token step rather than a full re-read of the prefix.
    """
    model = utils.model
    if model is None:
        raise RuntimeError("Call load_model(...) first")
    device = next(model.parameters()).device
    n = candidates.shape[0]

    cache, _ = prefix_pass(prefix_ids)
    cache.batch_repeat_interleave(n)

    out = model(
        input_ids=candidates.view(-1, 1).to(device),
        attention_mask=torch.ones(n, prefix_ids.shape[0] + 1, dtype=torch.long, device=device),
        past_key_values=cache,
        output_hidden_states=True,
    )
    return out.hidden_states[layer][:, 0].float().cpu()


@torch.no_grad()
def solve_position(
    prefix_ids: Tensor,
    target: Tensor,
    layer: int,
    *,
    tol: float,
    chunk: int,
    exhaustive: bool,
) -> dict:
    """Find the token whose h_layer at this position equals target."""
    model = utils.model
    if model is None:
        raise RuntimeError("Call load_model(...) first")
    vocab_size = int(model.config.vocab_size)

    # Scan in the model's own next-token order. For real text the true token is
    # nearly always in the first chunk, so this exits ~150x early. Ordering cannot
    # change the answer, only how soon the scan reaches it.
    _, logits = prefix_pass(prefix_ids)
    order = torch.argsort(logits[:vocab_size], descending=True)

    best_d, best_id, runner_d, tried = float("inf"), -1, float("inf"), 0

    for start in range(0, vocab_size, chunk):
        cands = order[start : start + chunk]
        d = (candidate_states(prefix_ids, cands, layer) - target).norm(dim=1)
        tried += cands.shape[0]

        top2 = torch.topk(d, k=min(2, d.shape[0]), largest=False)
        for dist, idx in zip(top2.values.tolist(), top2.indices.tolist()):
            if dist < best_d:
                runner_d, best_d, best_id = best_d, dist, int(cands[idx])
            elif dist < runner_d:
                runner_d = dist

        if best_d <= tol and not exhaustive:
            break

    return {
        "token": best_id,
        "residual": best_d,  # || h_observed - h(recovered) || — the detector signal
        "runner_up": runner_d,  # a true runner-up only when exhaustive
        "gap": runner_d - best_d,
        "tried": tried,
        "matched": best_d <= tol,
    }


@torch.no_grad()
def sipit(
    target: Tensor,
    layer: int,
    vocab_layer: Tensor,
    *,
    tol: float,
    chunk: int = 1024,
    exhaustive: bool = False,
    stop_on_fail: bool = True,
) -> list[dict]:
    """Recover the token sequence behind target [seq, hidden] at one layer."""
    steps: list[dict] = []

    # Position 0: empty prefix, so this is a nearest-neighbour lookup.
    d0 = dists_to(target[0], vocab_layer)
    top2 = torch.topk(d0, k=2, largest=False)
    best = float(top2.values[0])
    steps.append(
        {
            "token": int(top2.indices[0]),
            "residual": best,
            "runner_up": float(top2.values[1]),
            "gap": float(top2.values[1] - top2.values[0]),
            "tried": vocab_layer.shape[0],
            "matched": best <= tol,
        }
    )
    if stop_on_fail and not steps[0]["matched"]:
        return steps

    # Positions 1..N-1: prefix fixed, scan the vocabulary for the next token.
    for t in range(1, target.shape[0]):
        prefix = torch.tensor([s["token"] for s in steps], dtype=torch.long)
        step = solve_position(prefix, target[t], layer, tol=tol, chunk=chunk, exhaustive=exhaustive)
        steps.append(step)
        if stop_on_fail and not step["matched"]:
            break

    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--act_path", type=str, required=True, help="a .pt from save_activations.py")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="data/activations/Qwen_Qwen2.5-0.5B-Instruct/vocab/vocab_table.pt",
    )
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="accept a candidate below this L2. Activation norms are ~1e3 and an exact "
        "re-run is bit-identical, so this is loose.",
    )
    parser.add_argument("--chunk", type=int, default=1024)
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
    args = parser.parse_args()

    _, tokenizer = load_model(args.model_name)

    blob = torch.load(args.act_path, weights_only=False)
    target = blob["activations"][args.layer]  # [seq, hidden]
    if args.max_len:
        target = target[: args.max_len]
    if args.noise:
        g = torch.Generator().manual_seed(0)
        v = torch.randn(target.shape[-1], generator=g)
        target = target + args.noise * v / v.norm()

    vocab_layer = load_vocab_layer(args.vocab_path, args.layer)

    start = time.time()
    steps = sipit(
        target,
        args.layer,
        vocab_layer,
        tol=args.tol,
        chunk=args.chunk,
        exhaustive=args.exhaustive,
        stop_on_fail=not args.no_stop_on_fail,
    )
    elapsed = time.time() - start

    print(
        f"\n{blob['id']}  layer {args.layer}  "
        f"{len(steps)}/{target.shape[0]} positions in {elapsed:.1f}s"
    )
    print(f"{'t':>3} {'token':>7} {'residual':>10} {'gap':>10} {'tried':>7}  text")
    for t, s in enumerate(steps):
        flag = "" if s["matched"] else "  <-- NO MATCH"
        print(
            f"{t:>3} {s['token']:>7} {s['residual']:>10.4g} {s['gap']:>10.4g} "
            f"{s['tried']:>7}  {tokenizer.decode(token_ids=[s['token']])!r}{flag}"
        )
    print(f"\nrecovered: {tokenizer.decode(token_ids=[s['token'] for s in steps])!r}")
    print(f"max residual: {max(s['residual'] for s in steps):.6g}")


if __name__ == "__main__":
    main()
