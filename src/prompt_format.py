"""Rendering a behavior's questions as prompts, plainly or through a chat template.

The behavior datasets ship a plain "{system}\\n\\nUser: ...\\nAssistant:" template,
which is deliberate: gpt2 and Pythia have no chat template at all, and every existing
result in this repo was measured that way.

For a jailbreak benchmark on an instruct model that is the wrong format, and not by a
little. An aligned model's refusal training is keyed to its own chat template; asked
the same harmful question in a bare "User: ..." completion frame, Qwen2.5-0.5B-Instruct
refuses well under half the time, so an ASR measured there is mostly measuring that
the safety behavior was never switched on. Arditi et al. (2406.11717) evaluate every
model through its own template, and matching them requires the same.

So the format is a flag, not a property of the dataset:

  plain   the dataset's own template. Works on base models. The regression path.
  chat    tokenizer.apply_chat_template, with the behavior's system prompt as a system
          message when fitting the direction and no system message at eval time --
          the same held-out setup the plain path uses.
  auto    chat when the tokenizer has a template, plain when it does not.

The contrast pairs and the evaluation prompts must be rendered the same way. A
direction fitted on chat-formatted activations and applied to plain prompts is fitted
on a different distribution than it is used on, and the resulting null result says
nothing about steering.
"""

from behaviors import PROMPT_PAIRS, Behavior, Item
from log import get_logger
from utils import require_model

logger = get_logger(__name__)

FORMATS = ("auto", "plain", "chat")


def has_chat_template() -> bool:
    _, tokenizer = require_model()
    return bool(getattr(tokenizer, "chat_template", None))


def resolve_format(fmt: str, behavior: Behavior) -> str:
    if fmt not in FORMATS:
        raise SystemExit(f"--prompt_format: expected one of {FORMATS}, got {fmt!r}")
    if behavior.kind == PROMPT_PAIRS:
        # sentiment's pairs are raw completions ("The movie was wonderful and I felt"),
        # with no instruction to put in a user turn. Wrapping them in a chat frame
        # would change what is being measured, so this one is always plain.
        if fmt == "chat":
            logger.warning("behavior %s is prompt_pairs; ignoring --prompt_format chat",
                           behavior.name)
        return "plain"
    if fmt != "auto":
        if fmt == "chat" and not has_chat_template():
            raise SystemExit(
                "--prompt_format chat: this tokenizer has no chat_template. Use plain "
                "(or auto), and read any refusal number from it as a base-model number.")
        return fmt
    return "chat" if has_chat_template() else "plain"


def _chat(messages: list[dict]) -> str:
    _, tokenizer = require_model()
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def render_prompt(behavior: Behavior, item: Item, fmt: str) -> str:
    """One evaluation prompt. No system message either way -- the steering vector is
    what has to supply the behavior, and leaving the system prompt in would measure
    instruction following instead."""
    if fmt == "plain":
        return behavior.prompt_for(item)
    return _chat([{"role": "user", "content": item.question}])


def render_prompts(behavior: Behavior, items: list[Item], fmt: str) -> list[str]:
    return [render_prompt(behavior, item, fmt) for item in items]


def render_contrast_pairs(behavior: Behavior, fmt: str) -> list[tuple[str, str]]:
    """(positive, negative) prompt pairs in the same format the evaluation uses."""
    if fmt == "plain":
        return behavior.contrast_pairs()

    def one(system: str, question: str) -> str:
        return _chat([{"role": "system", "content": system},
                      {"role": "user", "content": question}])

    assert behavior.pos_system is not None and behavior.neg_system is not None
    return [(one(behavior.pos_system, q), one(behavior.neg_system, q))
            for q in behavior.train_questions]
