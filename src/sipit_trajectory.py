import torch
from torch import Tensor

import sipit
from generate import Intervention, _hooked
from utils import apply_final_norm, get_decoder_layers, model_device, require_model


@torch.no_grad()
def capture(input_ids: Tensor, layer: int,
            intervention: Intervention | None = None) -> Tensor:
    """The layer-`layer` residual stream over every position, under `intervention`.

    Read off the block with a hook rather than from output_hidden_states: a forward
    hook that rewrites output is NOT visible at its own index there, only from the
    next one on, so hidden_states[layer] would silently be the unsteered state.
    """
    model, _ = require_model()
    ids = input_ids.to(model_device())
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    mask = torch.ones_like(ids)
    grabbed: dict[str, Tensor] = {}

    def grab(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        grabbed["h"] = out[0].detach()

    handles = _hooked(intervention, mask=mask)
    handles.append(get_decoder_layers()[layer - 1].register_forward_hook(grab))
    try:
        model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return apply_final_norm(grabbed["h"].float(), layer).cpu()


def invert(target: Tensor, layer: int, vocab_layer: Tensor, gold_ids: list[int], *,
           rel_tol: float, stop_on_fail: bool = False, exhaustive: bool = False,
           schedule: tuple[int, ...] = sipit.DEFAULT_SCHEDULE) -> dict:
    """Left-to-right inversion of a trajectory, with no clean prefix to lean on.

    The prefix is rebuilt from whatever the sweep recovered, through the unhooked
    model, so a wrong token at t corrupts every position after it.
    """
    steps = sipit.sipit(target, layer, vocab_layer, rel_tol=rel_tol, abs_tol=0.0,
                        schedule=schedule, exhaustive=exhaustive,
                        stop_on_fail=stop_on_fail, gold=gold_ids)
    recovered = [s["token"] for s in steps]
    first_fail = next((i for i, s in enumerate(steps) if not s["matched"]), None)
    first_wrong = next((i for i, s in enumerate(steps) if not s["correct"]), None)
    return {
        "n_target": int(target.shape[0]),
        "n_steps": len(steps),
        "recovered_ids": recovered,
        "exact": recovered == gold_ids[:len(recovered)],
        "token0_correct": bool(steps[0]["correct"]),
        "token0_matched": bool(steps[0]["matched"]),
        "first_fail": first_fail,
        "first_wrong": first_wrong,
        "silent_corruption": (first_wrong is not None
                              and (first_fail is None or first_wrong < first_fail)),
        "n_wrong": sum(1 for s in steps if not s["correct"]),
        "n_unmatched": sum(1 for s in steps if not s["matched"]),
        "steps": steps,
    }
