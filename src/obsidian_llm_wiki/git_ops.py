"""
Git safety net: auto-commit and safe undo via git revert.
Only touches [olw] prefixed commits — never user commits.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_OLW_PREFIX = "[olw]"


def _run(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=check)


def git_commit(
    vault: Path,
    message: str,
    paths: list[str] | None = None,
) -> bool:
    """Stage paths and commit. Returns True if committed.

    paths defaults to wiki/, raw/, vault-schema.md, .olw/ (full snapshot).
    Pass a subset to create targeted commits (e.g. ingest vs approve).
    """
    if paths is None:
        paths = ["wiki/", "raw/", "vault-schema.md", ".olw/"]
    try:
        _run(["git", "add"] + paths, cwd=vault)
        # Check if there's anything staged
        result = _run(["git", "status", "--porcelain"], cwd=vault)
        if not result.stdout.strip():
            log.debug("git_commit: nothing to commit")
            return False
        _run(["git", "commit", "-m", f"{_OLW_PREFIX} {message}"], cwd=vault)
        log.info("git commit: %s %s", _OLW_PREFIX, message)
        return True
    except subprocess.CalledProcessError as e:
        log.warning("git commit failed: %s", e.stderr)
        return False


def git_log_olw(vault: Path, n: int = 10) -> list[dict]:
    """Return last N [olw] commits as list of {hash, message}."""
    try:
        result = _run(
            ["git", "log", f"--max-count={n * 3}", "--oneline", "--format=%H %s"],
            cwd=vault,
        )
        commits = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2 and _OLW_PREFIX in parts[1]:
                commits.append({"hash": parts[0], "message": parts[1]})
                if len(commits) >= n:
                    break
        return commits
    except subprocess.CalledProcessError:
        return []


def git_undo(vault: Path, steps: int = 1) -> list[str]:
    """
    Revert last N [olw] auto-commits using git revert (safe — creates new commits).
    Returns list of reverted commit messages.
    """
    commits = git_log_olw(vault, n=steps)
    if not commits:
        return []
    reverted = []
    for c in commits:
        try:
            _run(
                ["git", "-c", "merge.conflictstyle=merge", "revert", "--no-edit", c["hash"]],
                cwd=vault,
            )
            reverted.append(c["message"])
        except subprocess.CalledProcessError as e:
            log.warning("git revert failed for %s: %s", c["hash"], e.stderr)
            break
    return reverted


def git_init(vault: Path) -> None:
    """Init git repo if not already initialised."""
    if not (vault / ".git").exists():
        _run(["git", "init"], cwd=vault)
        log.info("Initialised git repo at %s", vault)
