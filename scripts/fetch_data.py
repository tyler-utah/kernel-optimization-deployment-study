"""Fetch rebuildable source and GitHub data for the artifact.

Commands:
  python scripts/fetch_data.py repositories
  python scripts/fetch_data.py preliminary
  python scripts/fetch_data.py deep-population
  python scripts/fetch_data.py lineage-git
  python scripts/fetch_data.py all

The collectors are resumable and reuse existing cache pages. They require
authenticated ``gh`` access. Agent-coded records are intentionally separate;
run ``scripts/run_agents.ps1`` after collection.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from artifact_config import ARTIFACT_ROOT, load_config


def run(*args: str, cwd: Path = ARTIFACT_ROOT) -> None:
    print(">", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def ensure_tool(name: str, *version_args: str) -> None:
    try:
        subprocess.run(
            [name, *version_args],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Required tool is unavailable: {name}") from exc


def fetch_repositories(config: dict) -> None:
    repo_dir = ARTIFACT_ROOT / "data" / "repos"
    repo_dir.mkdir(parents=True, exist_ok=True)
    required_heads = {
        repo: {
            study["heads"][repo]
            for study in config["studies"].values()
        }
        for repo in config["repositories"]
    }

    for repo, metadata in config["repositories"].items():
        target = repo_dir / repo
        if not (target / ".git").exists():
            run(
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                metadata["clone_url"],
                str(target),
            )
        run("git", "-C", str(target), "remote", "set-url", "origin", metadata["clone_url"])
        for sha in sorted(required_heads[repo]):
            run("git", "-C", str(target), "fetch", "--filter=blob:none", "origin", sha)
            resolved = subprocess.check_output(
                ["git", "-C", str(target), "rev-parse", sha],
                text=True,
            ).strip()
            if resolved != sha:
                raise RuntimeError(f"Frozen commit mismatch for {repo}: {sha}")

        preliminary = config["studies"]["preliminary"]
        log_path = ARTIFACT_ROOT / "data" / f"{repo}-log.txt"
        command = [
            "git",
            "-C",
            str(target),
            "log",
            "--no-renames",
            "--first-parent",
            '--format=@@@%H|%as|%ae|%s',
            "--name-only",
            preliminary["heads"][repo],
            f"--until={preliminary['cutoff']}",
        ]
        print(">", " ".join(command), ">", log_path, flush=True)
        with log_path.open("w", encoding="utf-8", newline="\n") as handle:
            subprocess.run(command, check=True, stdout=handle)


def collect_preliminary() -> None:
    run(sys.executable, "scripts/pr_lifecycle.py", "collect")


def collect_deep_population() -> None:
    run(sys.executable, "deep-study/scripts/status.py", "init")
    run(sys.executable, "deep-study/scripts/a1_metadata.py", "compact")
    run(sys.executable, "deep-study/scripts/a1_metadata.py", "enrich")


def collect_lineage_git() -> None:
    run(sys.executable, "kernel-lineage-study/scripts/git_extract.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["repositories", "preliminary", "deep-population", "lineage-git", "all"],
    )
    args = parser.parse_args()

    ensure_tool("git", "--version")
    if args.command in {"preliminary", "deep-population", "all"}:
        ensure_tool("gh", "--version")
    config = load_config()

    if args.command in {"repositories", "all"}:
        fetch_repositories(config)
    if args.command in {"preliminary", "all"}:
        collect_preliminary()
    if args.command in {"deep-population", "all"}:
        collect_deep_population()
    if args.command in {"lineage-git", "all"}:
        collect_lineage_git()


if __name__ == "__main__":
    main()
