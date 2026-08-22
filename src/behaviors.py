"""The steering behaviors under test, loaded from data/behavior_<name>.json.

One schema covers five datasets that are not shaped alike, so the loader normalises
two construction forms rather than making every dataset pretend to be the other:

    system_contrast   (refusal, jbb_refusal, evil_persona, language_fr)
        A pos_system / neg_system pair and a list of train_questions. The contrast
        pair for a question is the same question under each system prompt, which is
        the persona-vector construction: everything except the behavior is held
        fixed, so the difference-in-means isolates it.

    prompt_pairs      (sentiment)
        Explicit (positive, negative) prompt pairs. The behavior is carried by the
        prompt's own wording, so there is no system prompt to vary.

Evaluation prompts come from `test_arms` and are rendered with `test_template`,
which deliberately carries NO system prompt: the direction was fitted with the
system prompts, so at eval time the steering vector has to supply the behavior on
its own. Anything that leaves the system prompt in is measuring instruction
following, not steering.

Deliberately free of torch/transformers imports -- the scorers, the judge and the
dataset stats want this module without paying for the model stack.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_ROOT = Path("data")

# Both construction forms; `kind` defaults to system_contrast because four of the
# five datasets use it and only sentiment has to say so.
SYSTEM_CONTRAST = "system_contrast"
PROMPT_PAIRS = "prompt_pairs"


def behavior_path(name_or_path: str) -> Path:
    """'refusal' -> data/behavior_refusal.json; a path is taken as given."""
    path = Path(name_or_path)
    if path.suffix == ".json" or path.exists():
        return path
    return DATA_ROOT / f"behavior_{name_or_path}.json"


def list_behaviors(root: Path = DATA_ROOT) -> list[str]:
    return sorted(p.stem[len("behavior_"):] for p in root.glob("behavior_*.json"))


@dataclass(frozen=True)
class Item:
    """One evaluation instruction, with whatever the dataset knew about it.

    `target` is the affirmative continuation JailbreakBench ships per behavior
    ("Sure, here is ..."). It is None for the datasets whose targets are global,
    and Behavior.target_for() is what resolves the two cases.
    """

    index: int
    question: str
    target: str | None = None
    category: str | None = None
    behavior: str | None = None
    source: str | None = None
    key: str | None = None


@dataclass(frozen=True)
class Behavior:
    name: str
    kind: str
    test_template: str
    train_questions: tuple[str, ...] = ()
    train_pairs: tuple[tuple[str, str], ...] = ()
    pos_system: str | None = None
    neg_system: str | None = None
    template: str | None = None
    arms: dict[str, tuple[Item, ...]] = field(default_factory=dict)
    default_arm: str = "default"
    targets: dict[str, str | None] = field(default_factory=dict)
    scorer: dict = field(default_factory=dict)
    description: str = ""
    source: str = ""

    # -- fitting the direction ------------------------------------------------

    def contrast_pairs(self) -> list[tuple[str, str]]:
        """(positive, negative) prompt pairs, in the shape build_steering_vectors wants.

        Positive means "exhibits the behavior": refusing, hostile, French, happy.
        The steering vector therefore points toward the behavior, and a negative
        fraction points away from it -- which is how the jailbreak arm is run.
        """
        if self.kind == PROMPT_PAIRS:
            return [tuple(pair) for pair in self.train_pairs]
        assert self.template is not None
        return [
            (self.template.format(system=self.pos_system, question=q),
             self.template.format(system=self.neg_system, question=q))
            for q in self.train_questions
        ]

    # -- evaluating -----------------------------------------------------------

    def arm_names(self) -> list[str]:
        return list(self.arms)

    def items(self, arm: str | None = None, n: int = 0) -> list[Item]:
        arm = arm or self.default_arm
        if arm not in self.arms:
            raise KeyError(f"{self.name}: no test arm {arm!r} (have {self.arm_names()})")
        got = list(self.arms[arm])
        return got[:n] if n else got

    def prompt_for(self, item: Item) -> str:
        return self.test_template.format(question=item.question)

    def prompts(self, arm: str | None = None, n: int = 0) -> list[str]:
        return [self.prompt_for(item) for item in self.items(arm, n)]

    def target_for(self, item: Item, side: str) -> str | None:
        """The teacher-forced continuation for one side, per item.

        A dataset-level target wins; a null one falls back to the item's own, which
        is how JailbreakBench's per-behavior affirmative targets ("Sure, here is a
        defamatory article ...") reach the negative side while the refusal side
        stays a single shared string.
        """
        if side not in ("pos", "neg"):
            raise ValueError(f"side must be 'pos' or 'neg', got {side!r}")
        target = self.targets.get(side)
        if target is not None:
            return target
        return item.target


def _items(raw: list, arm: str) -> tuple[Item, ...]:
    """Accept both shapes a test arm comes in: bare strings, or dicts with metadata."""
    out = []
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            out.append(Item(index=i, question=entry))
            continue
        if "question" not in entry:
            raise ValueError(f"arm {arm!r} item {i} has no 'question' key")
        out.append(Item(
            index=int(entry.get("index", i)),
            question=entry["question"],
            target=entry.get("target"),
            category=entry.get("category"),
            behavior=entry.get("behavior"),
            source=entry.get("source"),
            key=entry.get("key"),
        ))
    return tuple(out)


def load_behavior(name_or_path: str) -> Behavior:
    path = behavior_path(name_or_path)
    if not path.exists():
        raise SystemExit(
            f"no such behavior: {name_or_path!r} ({path} does not exist). "
            f"Have: {', '.join(list_behaviors())}"
        )
    raw = json.loads(path.read_text())
    kind = raw.get("kind", SYSTEM_CONTRAST)
    if kind not in (SYSTEM_CONTRAST, PROMPT_PAIRS):
        raise ValueError(f"{path}: unknown kind {kind!r}")

    arms = {name: _items(items, name) for name, items in raw.get("test_arms", {}).items()}
    if not arms:
        raise ValueError(f"{path}: no test_arms")
    default_arm = raw.get("default_arm") or next(iter(arms))
    if default_arm not in arms:
        raise ValueError(f"{path}: default_arm {default_arm!r} is not one of {list(arms)}")

    behavior = Behavior(
        name=raw.get("name", path.stem.removeprefix("behavior_")),
        kind=kind,
        test_template=raw.get("test_template", "{question}"),
        train_questions=tuple(raw.get("train_questions", ())),
        train_pairs=tuple(tuple(p) for p in raw.get("train_pairs", ())),
        pos_system=raw.get("pos_system"),
        neg_system=raw.get("neg_system"),
        template=raw.get("template"),
        arms=arms,
        default_arm=default_arm,
        targets=raw.get("targets", {}),
        scorer=raw.get("scorer", {}),
        description=raw.get("description", ""),
        source=raw.get("source", ""),
    )

    if kind == SYSTEM_CONTRAST:
        missing = [k for k in ("pos_system", "neg_system", "template")
                   if getattr(behavior, k) is None]
        if missing or not behavior.train_questions:
            raise ValueError(f"{path}: system_contrast needs {missing or 'train_questions'}")
    elif not behavior.train_pairs:
        raise ValueError(f"{path}: prompt_pairs needs train_pairs")
    return behavior
