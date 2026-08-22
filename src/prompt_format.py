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
    if fmt == "plain":
        return behavior.prompt_for(item)
    return _chat([{"role": "user", "content": item.question}])


def render_prompts(behavior: Behavior, items: list[Item], fmt: str) -> list[str]:
    return [render_prompt(behavior, item, fmt) for item in items]


def render_contrast_pairs(behavior: Behavior, fmt: str) -> list[tuple[str, str]]:
    if fmt == "plain":
        return behavior.contrast_pairs()

    def one(system: str, question: str) -> str:
        return _chat([{"role": "system", "content": system},
                      {"role": "user", "content": question}])

    assert behavior.pos_system is not None and behavior.neg_system is not None
    return [(one(behavior.pos_system, q), one(behavior.neg_system, q))
            for q in behavior.train_questions]
