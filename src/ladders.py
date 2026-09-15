"""The PGD ladders the current results are read from.

One entry per (model, prompt set): where the attack rows and the detector calibration
that scored them live, the calibration bank's per-layer inversions and prompt files, the
stage-1 deltas the attack saved, the shipped detection summary, and the recomputation
baseline run on the same rows. Paths are relative to the repository root; `resolve` makes
them absolute. Every driver that re-reads a ladder takes labels from here, so a new model
is one entry.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Ladder:
    label: str
    model: str
    slug: str
    root: Path
    calibration: str
    bank_layers: Path
    bank_prompts: tuple[Path, ...]
    deltas: Path
    summary: Path
    baselines: Path
    n_prompts: int
    behavior: str = "jbb_refusal"

    @property
    def calibration_path(self) -> Path:
        return self.root / "sipit" / self.calibration

    @property
    def rows(self) -> Path:
        return self.root / "pgd_sipit" / "pgd_rows.jsonl"


def _cuda(label: str, model: str) -> Ladder:
    slug = model.replace("/", "_")
    fp16 = Path("results_cuda") / f"{slug}_fp16"
    return Ladder(label, model, slug, fp16, "detector_calibration_rolezlog_k1_fpr5_n75.json",
                  fp16 / "sipit/chat_bank_n75/layers",
                  (fp16 / "sipit/chat_bank/trajectory_bank_chat_ext.json",),
                  Path("results_cuda") / f"{slug}_fp32/pgd/ladder",
                  fp16 / "pgd_sipit/tpr_summary_rolezlog_k1_fpr5_n75.json",
                  Path("results/current/baselines") / f"{slug}_fp16.json", 15)


N50 = Ladder("Qwen-0.5B (n=50)", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen_Qwen2.5-0.5B-Instruct",
             Path("results/Qwen_Qwen2.5-0.5B-Instruct_n50"),
             "detector_calibration_rolezlog_k1_fpr5_n100.json",
             Path("results/Qwen_Qwen2.5-0.5B-Instruct_fp16/sipit/chat_bank_n100/layers"),
             (Path("results/Qwen_Qwen2.5-0.5B-Instruct_fp16/sipit/chat_bank/trajectory_bank_chat.json"),
              Path("results/Qwen_Qwen2.5-0.5B-Instruct_fp16/sipit/chat_bank/trajectory_bank_chat_ext.json")),
             Path("results/Qwen_Qwen2.5-0.5B-Instruct_n50/pgd/ladder"),
             Path("results/Qwen_Qwen2.5-0.5B-Instruct_n50/pgd_sipit/tpr_summary_rolezlog_k1_fpr5_n100.json"),
             Path("results/current/baselines/Qwen_Qwen2.5-0.5B-Instruct_n50.json"), 50)

LADDERS: dict[str, Ladder] = {lad.label: lad for lad in (
    N50,
    _cuda("Qwen-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    _cuda("Qwen-7B", "Qwen/Qwen2.5-7B-Instruct"),
    _cuda("gemma-3-1b", "google/gemma-3-1b-it"),
    _cuda("Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"),
)}


@dataclass(frozen=True)
class ReadRun:
    """One read-layer attack tree: the attacker held the surrogate under a ceiling at
    `read_layers` only, or (free) at nothing."""
    label: str
    model: str
    slug: str
    root: Path
    calibration: Path
    bank_layers: Path
    read_layers: tuple[int, ...]
    n_prompts: int

    @property
    def rows(self) -> Path:
        return self.root / "pgd_sipit" / "pgd_rows.jsonl"

    @property
    def ladder_dir(self) -> Path:
        return self.root / "pgd" / "ladder"


def _read(model: str, tag: str, read_layers: tuple[int, ...], n_prompts: int,
          calibration: str, bank: str, root: str = "results_cuda") -> ReadRun:
    slug = model.replace("/", "_")
    fp16 = Path(root) / f"{slug}_fp16"
    return ReadRun(f"{slug.split('_')[-1]} {tag}", model, slug, Path(root) / f"{slug}_{tag}",
                   fp16 / "sipit" / calibration, fp16 / "sipit" / bank, read_layers, n_prompts)


_N75 = ("detector_calibration_rolezlog_k1_fpr5_n75.json", "chat_bank_n75/layers")
_QWEN = _N75

READ_RUNS: dict[str, ReadRun] = {r.label: r for r in (
    _read("meta-llama/Llama-3.2-1B-Instruct", "read15", (15,), 15, *_N75),
    _read("meta-llama/Llama-3.2-1B-Instruct", "read16", (16,), 15, *_N75),
    _read("meta-llama/Llama-3.2-1B-Instruct", "free", (15, 16), 15, *_N75),
    _read("google/gemma-3-1b-it", "read22", (22,), 15, *_N75),
    _read("google/gemma-3-1b-it", "read26", (26,), 15, *_N75),
    _read("google/gemma-3-1b-it", "free", (22, 26), 15, *_N75),
    _read("meta-llama/Llama-3.2-1B-Instruct", "read16n50", (16,), 50, *_N75),
    _read("meta-llama/Llama-3.2-1B-Instruct", "freen50", (16,), 50, *_N75),
    _read("Qwen/Qwen2.5-0.5B-Instruct", "read24", (24,), 25, *_N75),
    _read("Qwen/Qwen2.5-0.5B-Instruct", "free", (24,), 25, *_N75),
)}


def select_read(labels: list[str] | None) -> list[ReadRun]:
    if not labels or labels == ["all"]:
        return list(READ_RUNS.values())
    unknown = [x for x in labels if x not in READ_RUNS]
    if unknown:
        raise SystemExit(f"unknown read run {unknown}; choose from {list(READ_RUNS)}")
    return [READ_RUNS[x] for x in labels]


def select(labels: list[str] | None) -> list[Ladder]:
    """No labels, or `all`, means every ladder in table order."""
    if not labels or labels == ["all"]:
        return list(LADDERS.values())
    unknown = [x for x in labels if x not in LADDERS]
    if unknown:
        raise SystemExit(f"unknown ladder {unknown}; choose from {list(LADDERS)}")
    return [LADDERS[x] for x in labels]
