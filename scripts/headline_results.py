"""Reproduce every headline result from versioned study tables."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from artifact_config import ARTIFACT_ROOT, load_config

DEEP_DATA = ARTIFACT_ROOT / "deep-study" / "data"
RESULTS = ARTIFACT_ROOT / "results"
REVIEW_SUMMARY = DEEP_DATA / "a3-review-message-summary.csv"
REPOS = ("vllm", "sglang")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_deep(name: str) -> list[dict]:
    return read_csv(DEEP_DATA / name)


def keyed(rows: list[dict], *fields: str) -> dict[tuple[str, ...], dict]:
    return {tuple(row[field] for field in fields): row for row in rows}


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def review_message_summary() -> dict[tuple[str, str], dict]:
    cache = DEEP_DATA / "cache" / "a3"
    metric_paths = {repo: cache / f"metrics-{repo}.jsonl" for repo in REPOS}
    if all(path.exists() for path in metric_paths.values()):
        corpus = {
            (row["repo"], int(row["number"])): row["group"]
            for row in read_deep("a3-detailed-corpus.csv")
        }
        rows = []
        for repo, path in metric_paths.items():
            values: dict[str, list[int]] = {"performance": [], "comparison": []}
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    metric = json.loads(line)
                    count = int(metric["n_substantive_reviewer_msgs"])
                    if count <= 0:
                        continue
                    group = corpus.get((repo, int(metric["number"])))
                    if group is None:
                        continue
                    values[group].append(count)
            for group, counts in values.items():
                rows.append(
                    {
                        "repo": repo,
                        "group": group,
                        "reviewed_prs": len(counts),
                        "median_substantive_reviewer_messages": statistics.median(counts),
                        "definition": (
                            "Median among corpus PRs with at least one substantive "
                            "non-author human review message"
                        ),
                    }
                )
        with REVIEW_SUMMARY.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    elif not REVIEW_SUMMARY.exists():
        raise FileNotFoundError(
            "Review caches and the versioned review-message summary are both missing."
        )
    return keyed(read_csv(REVIEW_SUMMARY), "repo", "group")


def main() -> None:
    config = load_config()
    population = keyed(read_deep("a2-population-summary.csv"), "repo", "group")
    survival = keyed(read_deep("a2-survival.csv"), "repo", "group", "horizon_days")
    open_queue = keyed(read_deep("a1-population-summary.csv"), "repo")
    reverts = keyed(read_deep("a2-revert-rates.csv"), "repo", "group")
    review = review_message_summary()
    evidence = keyed(read_deep("a3-evidence-summary.csv"), "repo", "group", "measure")
    descriptions = keyed(read_deep("a6-pr-description-mentions.csv"), "repo", "measure")
    revert_reasons = keyed(
        read_deep("b1-revert-summary.csv"),
        "repo",
        "measure",
        "category",
    )
    cohort_status = Counter(
        (row["repo"], row["state"])
        for row in read_deep("pr-population-metadata.csv")
        if row["in_window"] == "1"
    )

    repositories = {}
    for repo in REPOS:
        perf = population[(repo, "performance")]
        other = population[(repo, "non_performance")]
        merged = cohort_status[(repo, "merged")]
        closed = cohort_status[(repo, "closed")]
        still_open = cohort_status[(repo, "open")]
        repositories[repo] = {
            "created": merged + closed + still_open,
            "merged": merged,
            "closed_without_merge": closed,
            "still_open": still_open,
            "created_per_day": (merged + closed + still_open) / 365,
            "merged_per_day": merged / 365,
            "performance_prs": int(perf["n"]),
            "performance_merged": int(perf["merged"]),
            "performance_median_merge_days": float(
                perf["median_days_to_merge_among_merged"]
            ),
            "other_median_merge_days": float(
                other["median_days_to_merge_among_merged"]
            ),
            "performance_closed_share": float(perf["closed_share"]),
            "other_closed_share": float(other["closed_share"]),
            "performance_open_share": float(perf["open_share"]),
            "other_open_share": float(other["open_share"]),
            "performance_merged_30d": float(
                survival[(repo, "performance", "30")]["cif_merged"]
            ),
            "other_merged_30d": float(
                survival[(repo, "non_performance", "30")]["cif_merged"]
            ),
            "performance_closed_30d": float(
                survival[(repo, "performance", "30")]["cif_closed"]
            ),
            "other_closed_30d": float(
                survival[(repo, "non_performance", "30")]["cif_closed"]
            ),
            "performance_median_lines": int(float(perf["median_lines_changed"])),
            "other_median_lines": int(float(other["median_lines_changed"])),
            "performance_median_files": int(float(perf["median_changed_files"])),
            "other_median_files": int(float(other["median_changed_files"])),
            "performance_median_commits": int(
                float(evidence[(repo, "performance", "median_n_commits")]["share"])
            ),
            "other_median_commits": int(
                float(evidence[(repo, "comparison", "median_n_commits")]["share"])
            ),
            "performance_median_review_messages": float(
                review[(repo, "performance")][
                    "median_substantive_reviewer_messages"
                ]
            ),
            "other_median_review_messages": float(
                review[(repo, "comparison")][
                    "median_substantive_reviewer_messages"
                ]
            ),
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
        field: sum(row[field] for row in repositories.values())
        for field in (
            "created",
            "merged",
            "closed_without_merge",
            "still_open",
            "performance_prs",
            "performance_merged",
            "complete_open_queue",
            "performance_open_queue",
        )
    }
    combined["closed_without_merge_per_day"] = (
        combined["closed_without_merge"] / 365
    )
    combined["other_open_queue"] = (
        combined["complete_open_queue"] - combined["performance_open_queue"]
    )
    combined["performance_open_queue_share"] = (
        combined["performance_open_queue"] / combined["complete_open_queue"]
    )
    combined["performance_share_of_merged"] = (
        combined["performance_merged"] / combined["merged"]
    )

    language = {
        measure: {
            "count": int(descriptions[("both", measure)]["n"]),
            "denominator": int(descriptions[("both", measure)]["denominator"]),
            "share": float(descriptions[("both", measure)]["share"]),
        }
        for measure in (
            "performance_language",
            "testing_language",
            "correctness_without_testing_language",
            "explicit_maintenance_or_compatibility",
        )
    }
    reason_rows = [
        row
        for row in read_deep("b1-revert-summary.csv")
        if row["repo"] == "both"
        and row["measure"] == "kernel_performance_revert_reason"
    ]
    performance_revert_reasons = {
        "total": int(
            revert_reasons[("both", "reverted_class", "kernel_performance")]["n"]
        ),
        "performance_regression": int(
            revert_reasons[
                (
                    "both",
                    "kernel_performance_revert_reason",
                    "performance_regression",
                )
            ]["n"]
        ),
        "counts": {
            row["category"]: int(row["n"])
            for row in sorted(reason_rows, key=lambda item: int(item["n"]), reverse=True)
        },
    }

    result = {
        "configuration": config["studies"]["deep"],
        "repositories": repositories,
        "combined": combined,
        "description_language": language,
        "performance_revert_reasons": performance_revert_reasons,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "headline-results.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Headline results",
        "",
        "Generated by `python scripts/headline_results.py` from versioned study tables.",
        "",
        "## Complete one-year PR population",
        "",
        "| Repository | Created | Merged | Closed without merge | Still open |",
        "|---|---:|---:|---:|---:|",
    ]
    for repo in REPOS:
        row = repositories[repo]
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
            f"**{combined['merged']:,} ({percent(combined['merged'] / combined['created'])})** | "
            f"**{combined['closed_without_merge']:,} "
            f"({percent(combined['closed_without_merge'] / combined['created'])})** | "
            f"**{combined['still_open']:,} "
            f"({percent(combined['still_open'] / combined['created'])})** |",
            "",
            f"Daily averages: {repositories['vllm']['created_per_day']:.0f} created / "
            f"{repositories['vllm']['merged_per_day']:.0f} merged in vLLM; "
            f"{repositories['sglang']['created_per_day']:.0f} created / "
            f"{repositories['sglang']['merged_per_day']:.0f} merged in SGLang; "
            f"{combined['closed_without_merge_per_day']:.0f} combined closures without merge.",
            "",
            "## Performance delivery",
            "",
            f"- {combined['performance_merged']:,} confirmed performance PRs merged "
            f"({percent(combined['performance_share_of_merged'])} of merged PRs).",
        ]
    )
    for repo in REPOS:
        row = repositories[repo]
        gap = row["performance_merged_30d"] - row["other_merged_30d"]
        lines.append(
            f"- {repo}: median merge time {row['performance_median_merge_days']:.1f} "
            f"vs {row['other_median_merge_days']:.1f} days; 30-day merge "
            f"{percent(row['performance_merged_30d'])} vs "
            f"{percent(row['other_merged_30d'])} ({gap * 100:+.1f} points)."
        )
        lines.append(
            f"  Closed without merge: {percent(row['performance_closed_share'])} "
            f"vs {percent(row['other_closed_share'])}; still open: "
            f"{percent(row['performance_open_share'])} vs "
            f"{percent(row['other_open_share'])}."
        )
    lines.extend(["", "## Evaluation work", ""])
    for repo in REPOS:
        row = repositories[repo]
        lines.append(
            f"- {repo}: {row['performance_median_lines']} vs "
            f"{row['other_median_lines']} median lines; "
            f"{row['performance_median_files']} vs {row['other_median_files']} files; "
            f"{row['performance_median_commits']} vs "
            f"{row['other_median_commits']} commits; "
            f"{row['performance_median_review_messages']:.0f} vs "
            f"{row['other_median_review_messages']:.0f} substantive reviewer messages "
            f"among substantively reviewed PRs."
        )
    lines.extend(
        [
            "",
            "## Open queue and reverts",
            "",
            f"- {combined['complete_open_queue']:,} PRs were open at the cutoff; "
            f"{combined['performance_open_queue']:,} "
            f"({percent(combined['performance_open_queue_share'])}) were confirmed "
            f"performance PRs and {combined['other_open_queue']:,} were other work.",
        ]
    )
    for repo in REPOS:
        row = repositories[repo]
        lines.append(
            f"- {repo}: merged performance PRs were reverted at "
            f"{percent(row['performance_revert_rate'])}, versus "
            f"{percent(row['other_revert_rate'])} for other merged PRs."
        )
    lines.append(
        f"- In the separate two-year census, "
        f"{performance_revert_reasons['performance_regression']} of "
        f"{performance_revert_reasons['total']} performance-change reverts cited "
        "a performance regression."
    )
    reasons = performance_revert_reasons["counts"]
    lines.append(
        "- Stated reasons in that census: "
        + "; ".join(
            f"{name.replace('_', ' ')} {count}"
            for name, count in reasons.items()
        )
        + "."
    )
    lines.extend(["", "## Language in performance PR descriptions", ""])
    for measure, label in (
        ("performance_language", "Performance language"),
        ("testing_language", "Testing language"),
        (
            "correctness_without_testing_language",
            "Correctness language without testing language",
        ),
        (
            "explicit_maintenance_or_compatibility",
            "Explicit maintenance or compatibility language",
        ),
    ):
        row = language[measure]
        lines.append(f"- {label}: {row['count']:,}/{row['denominator']:,} ({percent(row['share'])}).")
    lines.append("")
    (RESULTS / "headline-results.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(RESULTS / "headline-results.md")


if __name__ == "__main__":
    main()
