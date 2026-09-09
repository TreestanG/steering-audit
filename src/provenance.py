"""What produced a result file: the code state and the inputs, by hash."""

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_fingerprint() -> dict:
    """git HEAD, whether the tree is dirty, and a hash over every tracked-or-not source
    file, so an uncommitted change still leaves a mark on what it produced."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    files = sorted(p for d in ("src", "scripts") for p in (ROOT / d).glob("*.py"))
    files += sorted((ROOT / "scripts").glob("*.sh"))
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return {"git_head": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", "src", "scripts")),
            "source_sha256": h.hexdigest()}


def stamp(**inputs: Path | str | None) -> dict:
    """A provenance block: code fingerprint plus sha256 of each named input file."""
    out = {"code": code_fingerprint(), "inputs": {}}
    for name, path in inputs.items():
        if path is not None and Path(path).exists():
            out["inputs"][name] = {"path": str(path), "sha256": file_sha256(path)}
    return out
