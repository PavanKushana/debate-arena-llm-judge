"""
utils/prompts.py — load LLM instruction prompts from judge_api/prompts/.

Every prompt lives as its own text file under prompts/, so anyone reading the
code can find and read the exact wording without digging through source. Each
function fetches its prompt at call time with load_prompt("<file>"); results
are cached, so repeated calls don't re-read the file.

Point PROMPTS_DIR at another location to override (e.g. per-environment prompt
sets) without touching code.

    from .prompts import load_prompt
    system_prompt = load_prompt("argument_miner_system.txt")

Subfolders are fine: load_prompt("stance/pro_con_system.txt").
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

# Default: judge_api/prompts/ (this file lives in judge_api/utils/).
PROMPTS_DIR = Path(
    os.getenv("PROMPTS_DIR", str(Path(__file__).resolve().parent.parent / "prompts"))
)


def prompt_path(name: str) -> Path:
    """Absolute path of a prompt file (whether or not it exists)."""
    return PROMPTS_DIR / name


@functools.lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Read and return the prompt text for `name` (relative to PROMPTS_DIR).

    Cached: the file is read once per process per name. Raises a clear
    FileNotFoundError naming the resolved path if the file is missing.
    """
    path = PROMPTS_DIR / name
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Prompt file not found: {path}. Add it under judge_api/prompts/ "
            f"or set PROMPTS_DIR to the folder that contains it."
        ) from exc
    return text.strip()


def clear_cache() -> None:
    """Drop the in-process prompt cache (e.g. after editing a prompt file
    during development so the next load_prompt re-reads from disk)."""
    load_prompt.cache_clear()
