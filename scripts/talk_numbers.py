"""Reproduce the headline PR numbers used in the talk.

Writes ``results/talk-numbers.json`` and ``results/talk-numbers.md`` from the
checked-in derived CSVs. This script does not call an agent or the network.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from artifact_config import ARTIFACT_ROOT, load_config

DEEP_DATA = ARTIFACT_ROOT / "deep-study" / "data"
RESULTS = ARTIFACT_ROOT / "results"


def read_csv(name: str) -> list[dict]:
    with (DEEP_DATA / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def keyed(rows: list[dict], *fields: str) -> dict[tuple[str, ...], dict]:
    return {tuple(row[field] for field in fields): row for row in rows}


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    config = load_config()
    population = keyed(read_csv("a2-population-summary.csv"), "repo", "group")
    survival = keyed(read_csv("a2-survival.csv"), "repo", "group", "horizon_days")
    open_queue = keyed(read_csv("a1-population-summary.csv"), "repo")
    reverts = keyed(read_csv("a2-revert-rates.csv"), "repo", "group")
    cohort_status = Counter(
        (row["repo"], row["state"])
        for row in read_csv("pr-population-metadata.csv")
        if row["in_window"] == "1"
    )

    repos = ("vllm", "sglang")
    repo_rows = {}
    for repo in repos:
        perf = population[(repo, "performance")]
        other = population[(repo, "non_performance")]
        merged = cohort_status[(repo, "merged")]
        closed = cohort_status[(repo, "closed")]
        open_count = cohort_status[(repo, "open")]
        total = merged + closed + open_count
        repo_rows[repo] = {
            "created": total,
            "merged": merged,
            "closed_without_merge": closed,
            "still_open": open_count,
            "confirmed_performance": int(perf["n"]),
            "performance_merged_30d": float(
                survival[(repo, "performance", "30")]["cif_merged"]
            ),
            "other_merged_30d": float(
                survival[(repo, "non_performance", "30")]["cif_merged"]
            ),
            "performance_median_lines": int(float(perf["median_lines_changed"])),
            "other_median_lines": int(float(other["median_lines_changed"])),
            "performance_revert_rate": float(
                reverts[(repo, "performance")]["revert_rate"]
            ),
            "other_revert_rate": float(
                reverts[(repo, "non_performance")]["revert_rate"]
            ),
            "complete_open_queue": int(open_queue[(repo,)]["open_at_cutoff_all"]),
            "performance_open_queue": int(
                open_queue[(repo,)]["open_confirmed_performance"]
            ),
        }

    combined = {
        field: sum(row[field] for row in repo_rows.values())
        for field in (
            "created",
            "merged",
            "closed_without_merge",
            "still_open",
            "confirmed_performance",
            "complete_open_queue",
            "performance_open_queue",
        )
    }
    result = {
        "configuration": config["studies"]["deep"],
        "repositories": repo_rows,
        "combined": combined,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "talk-numbers.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Reproduced talk numbers",
        "",
        "| Repository | Created | Merged | Closed without merge | Still open |",
        "|---|---:|---:|---:|---:|",
    ]
    for repo in repos:
        row = repo_rows[repo]
        lines.append(
            f"| {repo} | {row['created']:,} | {row['merged']:,} "
            f"({percent(row['merged'] / row['created'])}) | "
            f"{row['closed_without_merge']:,} "
            f"({percent(row['closed_without_merge'] / row['created'])}) | "
            f"{row['still_open']:,} "
            f"({percent(row['still_open'] / row['created'])}) |"
        )
    lines.extend(
        [
            f"| **Combined** | **{combined['created']:,}** | "
            f"**{combined['merged']:,}** "
            f"**({percent(combined['merged'] / combined['created'])})** | "
            f"**{combined['closed_without_merge']:,}** "
            f"**({percent(combined['closed_without_merge'] / combined['created'])})** | "
            f"**{combined['still_open']:,}** "
            f"**({percent(combined['still_open'] / combined['created'])})** |",
            "",
            "## Performance comparison",
            "",
        ]
    )
    for repo in repos:
        row = repo_rows[repo]
        gap = row["performance_merged_30d"] - row["other_merged_30d"]
        lines.append(
            f"- **{repo}:** 30-day merge {percent(row['performance_merged_30d'])} "
            f"vs {percent(row['other_merged_30d'])} other "
            f"({gap * 100:+.1f} points); median size "
            f"{row['performance_median_lines']} vs {row['other_median_lines']} lines; "
            f"reverts {percent(row['performance_revert_rate'])} vs "
            f"{percent(row['other_revert_rate'])}."
        )
    lines.extend(
        [
            "",
            f"Complete open queue: **{combined['complete_open_queue']:,}**, including "
            f"**{combined['performance_open_queue']:,} confirmed performance PRs**.",
            "",
        ]
    )
    (RESULTS / "talk-numbers.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(RESULTS / "talk-numbers.md")


if __name__ == "__main__":
    main()
