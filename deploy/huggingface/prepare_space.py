"""Prepare a folder to push to a Hugging Face Space (works on Windows, macOS, Linux).

    python deploy/huggingface/prepare_space.py ../scamshield-space

Copies the project WITHOUT trained models, reports, caches or your private corrections
(the Docker image rebuilds the data and retrains at build time), and puts the Space's
README.md (with the YAML header Hugging Face needs) at the root. Hugging Face rejects
binary files that aren't stored with Git LFS, so leaving models/ and reports/ out keeps
the push simple.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDE_DIRS = {".git", ".github", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv",
                "models", "reports", "processed"}
EXCLUDE_FILES = {"feedback.csv"}


def ignore(directory: str, names: list[str]) -> set[str]:
    skipped = {n for n in names if n in EXCLUDE_DIRS or n in EXCLUDE_FILES or n.endswith((".zip", ".pyc"))}
    if Path(directory).name == "external":  # private datasets you dropped in
        skipped |= {n for n in names if n.endswith(".csv")}
    return skipped


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python deploy/huggingface/prepare_space.py <target-folder>")
    target = Path(sys.argv[1]).resolve()
    if target.exists() and any(p.name != ".git" for p in target.iterdir()):
        raise SystemExit(f"{target} is not empty. Use the folder you cloned your Space into, or a new one.")
    target.mkdir(parents=True, exist_ok=True)
    for item in ROOT.iterdir():
        if item.name in EXCLUDE_DIRS or item.name == "README.md":
            continue
        if item.is_dir():
            shutil.copytree(item, target / item.name, ignore=ignore)
        elif not ignore(str(ROOT), [item.name]):
            shutil.copy2(item, target / item.name)
    shutil.copy2(ROOT / "deploy" / "huggingface" / "README.md", target / "README.md")
    print(f"Space folder ready: {target}")
    print("Next: cd into it, then  git add .  &&  git commit -m \"Deploy ScamShield\"  &&  git push")


if __name__ == "__main__":
    main()
