from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
HAND_PATH = ROOT / "data" / "trajectory_bank_handauthored.json"
OUT_PATH = ROOT / "data" / "trajectory_bank_prompts.json"

SEED = 0
N_NATURAL = 200
MAX_CHARS = 400  

TINYSTORIES_REPO = "roneneldan/TinyStories"
TINYSTORIES_CANDIDATES = (
    "TinyStoriesV2-GPT4-valid.txt",
    "TinyStories-valid.txt",
)


def load_handauthored(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    prompts: list[dict] = []
    for category, texts in raw.items():
        for i, text in enumerate(texts, start=1):
            text = text.strip()
            if not text:
                raise ValueError(f"empty prompt in {category}#{i}")
            prompts.append(
                {
                    "id": f"{category}_{i:04d}",
                    "category": category,
                    "text": text,
                }
            )
    return prompts


def download_tinystories_valid() -> Path:
    last_err: Exception | None = None
    for filename in TINYSTORIES_CANDIDATES:
        try:
            path = hf_hub_download(
                repo_id=TINYSTORIES_REPO,
                filename=filename,
                repo_type="dataset",
            )
            return Path(path)
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"could not download TinyStories validation from {TINYSTORIES_REPO}; "
        f"tried {TINYSTORIES_CANDIDATES}"
    ) from last_err


def split_stories(raw: str) -> list[str]:
    chunks = re.split(r"\n\s*\n+", raw)
    stories = []
    for chunk in chunks:
        text = " ".join(line.strip() for line in chunk.splitlines() if line.strip())
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= 40:
            stories.append(text)
    return stories


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0].strip()
    return cut if cut else text[:max_chars].strip()


def sample_natural(rng: random.Random, n: int, max_chars: int) -> list[dict]:
    path = download_tinystories_valid()
    stories = split_stories(path.read_text(encoding="utf-8", errors="replace"))
    if len(stories) < n:
        raise RuntimeError(f"only {len(stories)} stories available, need {n}")
    chosen = rng.sample(stories, n)
    prompts = []
    for i, story in enumerate(chosen, start=1):
        prompts.append(
            {
                "id": f"natural_en_{i:04d}",
                "category": "natural_en",
                "text": truncate(story, max_chars),
            }
        )
    return prompts


def validate(prompts: list[dict]) -> None:
    ids = [p["id"] for p in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prompt ids")
    for p in prompts:
        if not p["text"].strip():
            raise ValueError(f"empty text for {p['id']}")


def build(seed: int, n_natural: int, out_path: Path) -> dict:
    rng = random.Random(seed)
    prompts = load_handauthored(HAND_PATH)
    prompts.extend(sample_natural(rng, n_natural, MAX_CHARS))
    validate(prompts)
    prompts.sort(key=lambda p: (p["category"], p["id"]))
    return {
        "version": 1,
        "seed": seed,
        "source": {
            "handauthored": str(HAND_PATH.relative_to(ROOT)),
            "natural": {
                "repo": TINYSTORIES_REPO,
                "n": n_natural,
                "max_chars": MAX_CHARS,
            },
        },
        "prompts": prompts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-natural", type=int, default=N_NATURAL)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    payload = build(args.seed, args.n_natural, args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    counts = Counter(p["category"] for p in payload["prompts"])
    print(f"wrote {args.out} ({len(payload['prompts'])} prompts)")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")
    print("examples:")
    for cat in sorted(counts):
        example = next(p for p in payload["prompts"] if p["category"] == cat)
        preview = example["text"].replace("\n", "\\n")[:80]
        print(f"  [{cat}] {preview}")


if __name__ == "__main__":
    main()
