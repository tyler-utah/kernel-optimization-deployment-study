"""Reproduce derived tables, figures, reports, and consistency checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from artifact_config import ARTIFACT_ROOT, load_config


def run(*args: str) -> None:
    print(">", " ".join(args), flush=True)
    subprocess.run(args, cwd=ARTIFACT_ROOT, check=True)


def preliminary() -> None:
    # Sampling creates blank coding sheets and belongs to the agent stage.
    # Never regenerate it here: doing so would erase completed labels.
    for command in ("classify", "metrics", "analyze", "figures"):
        run(sys.executable, "scripts/system_evolution.py", command)
    run(sys.executable, "scripts/pr_lifecycle.py", "analyze")
    run(sys.executable, "scripts/verify_consistency.py")


def deep() -> None:
    run(sys.executable, "deep-study/scripts/a2_analysis.py")
    run(sys.executable, "deep-study/scripts/a3_code.py", "merge")
    run(sys.executable, "deep-study/scripts/b_analysis.py", "final")
    run(sys.executable, "deep-study/scripts/b_summary.py")
    run(sys.executable, "deep-study/scripts/c_analysis.py", "final")
    run(sys.executable, "deep-study/scripts/d_analysis.py", "final")
    run(sys.executable, "deep-study/scripts/reports.py", "all")
    run(sys.executable, "deep-study/scripts/consistency.py")
    run(sys.executable, "scripts/talk_numbers.py")


def lineage() -> None:
    run(sys.executable, "kernel-lineage-study/scripts/run_all.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "study",
        nargs="?",
        default="all",
        choices=["preliminary", "deep", "lineage", "all"],
    )
    args = parser.parse_args()
    load_config()

    if args.study in {"preliminary", "all"}:
        preliminary()
    if args.study in {"deep", "all"}:
        deep()
    if args.study in {"lineage", "all"}:
        lineage()


if __name__ == "__main__":
    main()
