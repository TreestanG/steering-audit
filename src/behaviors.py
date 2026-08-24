import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_ROOT = Path("data")

SYSTEM_CONTRAST = "system_contrast"
PROMPT_PAIRS = "prompt_pairs"


def behavior_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.suffix == ".json" or path.exists():
        return path
    return DATA_ROOT / f"behavior_{name_or_path}.json"


def list_behaviors(root: Path = DATA_ROOT) -> list[str]:
    return sorted(p.stem[len("behavior_"):] for p in root.glob("behavior_*.json"))


@dataclass(frozen=True)
class Item:
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

    def contrast_pairs(self) -> list[tuple[str, str]]:
        if self.kind == PROMPT_PAIRS:
            return [(pos, neg) for pos, neg in self.train_pairs]
        assert self.template is not None
        return [
            (self.template.format(system=self.pos_system, question=q),
             self.template.format(system=self.neg_system, question=q))
            for q in self.train_questions
        ]

    def arm_names(self) -> list[str]:
        return list(self.arms)

    def items(self, arm: str | None = None, n: int = 0, stratify: bool = True) -> list[Item]:
        arm = arm or self.default_arm
        if arm not in self.arms:
            raise KeyError(f"{self.name}: no test arm {arm!r} (have {self.arm_names()})")
        got = list(self.arms[arm])
        if not n or n >= len(got):
            return got
        return stratified(got, n) if stratify else got[:n]

    def prompt_for(self, item: Item) -> str:
        return self.test_template.format(question=item.question)

    def prompts(self, arm: str | None = None, n: int = 0, stratify: bool = True) -> list[str]:
        return [self.prompt_for(item) for item in self.items(arm, n, stratify)]

    def target_for(self, item: Item, side: str) -> str | None:
        if side not in ("pos", "neg"):
            raise ValueError(f"side must be 'pos' or 'neg', got {side!r}")
        target = self.targets.get(side)
        if target is not None:
            return target
        return item.target


def stratified(items: list[Item], n: int) -> list[Item]:
    """First n in round-robin category order, rather than the first n in file order.

    JailbreakBench ships 100 prompts as 10 contiguous blocks of 10 categories, so a
    plain head takes 2 of 10 categories at n=20 and calls it a sample of the benchmark.

    Round-robin is prefix-consistent -- the n=5 sample is the first 5 of the n=20
    sample -- so a small audit stays a subset of a larger generation run and the two
    still join.
    """
    by_category: dict[str | None, list[Item]] = {}
    for item in items:
        by_category.setdefault(item.category, []).append(item)
    if len(by_category) < 2:
        return items[:n]
    ordered = []
    for tier in range(max(len(group) for group in by_category.values())):
        for group in by_category.values():
            if tier < len(group):
                ordered.append(group[tier])
    return ordered[:n]


def _items(raw: list, arm: str) -> tuple[Item, ...]:
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
