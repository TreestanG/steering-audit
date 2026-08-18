"""Logging shared by every entrypoint.

Console goes to stderr and stdout stays empty: nothing in this repo emits
machine-readable stdout (every handoff is jsonl/json/png), and run_model.sh
parses `num_hidden_layers` off a child's stdout, so keeping the two apart makes
that safe by construction.

run_model.sh passes AAT_LOG_FILE / AAT_LOG_STAGE / AAT_LOG_LEVEL rather than
flags, so the wrapper scripts in scripts/ need no argument forwarding.
Precedence is explicit flag > environment > default.
"""

import logging
import os
import sys
from pathlib import Path

# Our loggers live under one prefix so root can stay at WARNING for libraries:
# httpx/urllib3/filelock all log at INFO and would otherwise flood the console.
PREFIX = "aat"

CONSOLE_FORMAT = "%(message)s"  # bare: a level prefix would break table alignment
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"

_configured = False


def add_logging_args(parser) -> None:
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v per-item detail, -vv also un-silences transformers")
    parser.add_argument("-q", "--quiet", action="count", default=0,
                        help="warnings and errors only")
    parser.add_argument("--log_file", type=str, default=None,
                        help="append a full DEBUG log here (default: env AAT_LOG_FILE)")


def get_logger(name: str) -> logging.Logger:
    """A logger named after the script, so %(name)s means something in a shared log."""
    if name == "__main__":
        name = Path(sys.argv[0]).stem
    return logging.getLogger(f"{PREFIX}.{name}")


def _console_level(args) -> int:
    verbose = getattr(args, "verbose", 0)
    quiet = getattr(args, "quiet", 0)
    if not verbose and not quiet:
        env = os.environ.get("AAT_LOG_LEVEL")
        if env:
            return logging.getLevelNamesMapping().get(env.upper(), logging.INFO)
    level = logging.INFO - 10 * verbose + 10 * quiet
    return max(logging.DEBUG, min(logging.CRITICAL, level))


def setup(args, default_log: Path | None = None) -> logging.Logger:
    """Configure root logging. Call once, from main() — never at import."""
    global _configured
    logger = get_logger("__main__")
    if _configured:
        return logger

    level = _console_level(args)
    root = logging.getLogger()
    # Root filters third-party records; ours are exempt via the aat.* logger.
    root.setLevel(logging.DEBUG if getattr(args, "verbose", 0) >= 2 else logging.WARNING)
    logging.getLogger(PREFIX).setLevel(logging.DEBUG)

    stage = os.environ.get("AAT_LOG_STAGE")
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root.addHandler(console)

    path = getattr(args, "log_file", None) or os.environ.get("AAT_LOG_FILE")
    if path is None and default_log is not None:
        path = str(default_log)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Append, never truncate: the fractions stage runs steer_audit 8 times
        # into one file, and truncating would keep only the last.
        handler = logging.FileHandler(path, mode="a")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(FILE_FORMAT))
        root.addHandler(handler)
        banner = (f"==== pid={os.getpid()}{f' stage={stage}' if stage else ''} "
                  f":: {' '.join(sys.argv)}")
        handler.emit(logging.LogRecord(
            f"{PREFIX}.run", logging.INFO, __file__, 0, banner, None, None
        ))

    logging.captureWarnings(True)
    _install_excepthook(logger)
    if getattr(args, "verbose", 0) < 2:
        _silence_libraries()

    _configured = True
    return logger


def _install_excepthook(logger: logging.Logger) -> None:
    """Send tracebacks through logging, so the log file does not just stop mid-stream."""
    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("unhandled %s", exc_type.__name__, exc_info=(exc_type, exc, tb))

    sys.excepthook = hook


def _silence_libraries() -> None:
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:  # transformers not imported by pure-plotting entrypoints
        pass
    # Not in huggingface_hub.utils.__all__, so reach for it dynamically; this is
    # best-effort silencing and the try/except already covers it going away.
    try:
        import huggingface_hub.utils as hub_utils

        getattr(hub_utils, "disable_progress_bars")()
    except Exception:
        pass


def heartbeat(logger: logging.Logger, i: int, total: int, what: str, every: float = 0.1) -> None:
    """INFO on each `every` fraction of the way, DEBUG otherwise.

    Long loops (152k vocab entries, 350 prompts) should neither print per item
    nor go silent for the whole of the most expensive stage in the pipeline.
    """
    logger.debug("%s %d/%d", what, i, total)
    if total <= 0:
        return
    step = max(1, int(total * every))
    if i % step < 1 or i >= total:
        logger.info("%s %d/%d (%d%%)", what, i, total, round(100 * i / total))
