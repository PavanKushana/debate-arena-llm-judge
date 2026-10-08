"""
utils/logging_config.py — Central logging setup for judge_api.

Writes TWO log files, plus the console:

  1. logs/judge_api.log            — human-readable narrative (as before).
     One line per logger.info/warning/error call across every module.

  2. logs/judge_api_telemetry.jsonl — structured, machine-parseable record
     of every external/model call made during scoring: IBM ArgQ-30k nearest
     neighbours, Elasticsearch requests+responses, Ollama LLM calls (argument
     mining, ArgQ fallback, rebuttal quality), and NLI model calls. ONE JSON
     object per line, written via utils/telemetry.py — never mixed into the
     narrative log, since the two serve different readers (a person
     following what happened, vs. a script re-parsing exact inputs/outputs).

Because every module uses logging.getLogger(__name__), configuring the root
logger here captures all of them automatically — no per-module changes needed
for the narrative log. The telemetry log is a separate, non-propagating
logger ("judge_api.telemetry") so it never also prints to the console.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# All logging settings are env-overridable so nothing is hardcoded.
LOG_DIR   = os.getenv("LOG_DIR", "logs")
LOG_FILE  = os.getenv("LOG_FILE", "judge_api.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Rotate after ~5 MB, keep 5 old files (so the log never grows unbounded).
LOG_MAX_BYTES    = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# Telemetry: structured, one-JSON-object-per-line records of every metric
# call. These payloads (LLM prompts, ES hit lists, corpus neighbours) are
# much bigger than a narrative log line, so it gets its own file, its own
# (larger) rotation budget, and its own retention — independent of the
# settings above.
TELEMETRY_LOG_FILE      = os.getenv("TELEMETRY_LOG_FILE", "judge_api_telemetry.jsonl")
TELEMETRY_MAX_BYTES     = int(os.getenv("TELEMETRY_MAX_BYTES", str(20 * 1024 * 1024)))
TELEMETRY_BACKUP_COUNT  = int(os.getenv("TELEMETRY_BACKUP_COUNT", "10"))


def setup_logging() -> str:
    """
    Configure the root logger (console + rotating narrative file) AND the
    dedicated telemetry logger (rotating JSONL file). Safe to call multiple
    times — it won't add duplicate handlers.
    Returns the absolute path of the narrative log file.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.abspath(os.path.join(LOG_DIR, LOG_FILE))

    root = logging.getLogger()           # the root logger
    root.setLevel(LOG_LEVEL)

    # Guard against double-configuration (e.g. uvicorn reload / repeated import)
    if getattr(setup_logging, "_configured", False):
        return log_path

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # Rotating narrative file
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # ── Telemetry: separate logger, separate file, no console duplication ──
    telemetry_path = os.path.abspath(os.path.join(LOG_DIR, TELEMETRY_LOG_FILE))
    telemetry_logger = logging.getLogger("judge_api.telemetry")
    telemetry_logger.setLevel(logging.INFO)
    telemetry_logger.propagate = False   # never bubble up to root/console —
                                          # keeps the console and narrative
                                          # log free of raw JSON payloads

    telemetry_handler = RotatingFileHandler(
        telemetry_path,
        maxBytes=TELEMETRY_MAX_BYTES,
        backupCount=TELEMETRY_BACKUP_COUNT,
        encoding="utf-8",
    )
    # Raw message only — each line IS a complete JSON object (see
    # utils/telemetry.py), so no timestamp/level prefix is added here.
    telemetry_handler.setFormatter(logging.Formatter("%(message)s"))
    telemetry_logger.addHandler(telemetry_handler)

    setup_logging._configured = True
    logging.getLogger("judge_api").info(
        "Logging initialised → narrative=%s  telemetry=%s", log_path, telemetry_path
    )
    return log_path
