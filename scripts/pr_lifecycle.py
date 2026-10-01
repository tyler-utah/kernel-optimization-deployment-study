"""Bounded, cached GitHub PR collection and preliminary lifecycle analysis.

Collection is resumable. REST list responses are cached per 100-record page;
detailed review/comment responses are cached in batches of at most 20 PRs.
The persistent manifest is atomically updated after every batch.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import math
import os
import random
import re
import subprocess
import time
from pathlib import Path

import system_evolution as evolution
from artifact_config import ARTIFACT_ROOT, load_config, parse_utc


CONFIG = load_config()
EXP = ARTIFACT_ROOT
RESULTS = EXP / "results"
CACHE = EXP / "data" / "github"
STATUS = RESULTS / "study-status.json"
CUTOFF = parse_utc(CONFIG["studies"]["preliminary"]["cutoff"])
START = parse_utc(CONFIG["studies"]["preliminary"]["start"])
SEED = CONFIG["studies"]["preliminary"]["seed"]
REPOS = {repo: metadata["slug"] for repo, metadata in CONFIG["repositories"].items()}
COMMIT_BODY_CACHE: dict[tuple[str, str], str] = {}


def iso(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_manifest(slug: str, dataset: str, status: str, cursor, records: int, files: list[str]) -> None:
    data = json.loads(STATUS.read_text(encoding="utf-8"))
    entry = data["github_collection"][slug].setdefault(dataset, {
        "status": "pending",
        "cursor": None,
        "records_written": 0,
        "cache_files": [],
    })
    entry.update({
        "status": status,
        "cursor": cursor,
        "records_written": records,
        "cache_files": [str(Path(f).relative_to(ROOT)).replace("\\", "/") for f in files],
    })
    data["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    data["current_phase"] = "05-pr-lifecycle"
    data["resume_next"] = (
        f"Resume {slug} {dataset} at cursor/page {cursor}; cached batches are reused."
        if status != "complete"
        else "Continue the next incomplete Phase 05 GitHub dataset or analyze the completed caches."
    )
    atomic_json(STATUS, data)


def gh_json(args: list[str], attempts: int = 3):
    for attempt in range(attempts):
        proc = subprocess.run(
            ["gh", "api", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=75,
        )
        if proc.returncode == 0:
            return json.loads(proc.stdout)
        if attempt + 1 < attempts:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"gh api failed: {' '.join(args)}\n{proc.stderr[-1000:]}")


def collect_recent(repo: str, slug: str) -> list[dict]:
    owner, name = slug.split("/")
    directory = CACHE / repo / "recent_prs"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    page = 1
    cache_files = []
    while True:
        cache = directory / f"page-{page:04d}.json"
        if cache.exists():
            batch = json.loads(cache.read_text(encoding="utf-8"))
        else:
            print(f"{repo}: recent PR metadata page {page} ({len(records)} records)", flush=True)
            batch = gh_json([
                f"repos/{owner}/{name}/pulls",
                "-f", "state=all",
                "-f", "sort=created",
                "-f", "direction=desc",
                "-f", "per_page=100",
                "-f", f"page={page}",
                "-X", "GET",
            ])
            atomic_json(cache, batch)
        cache_files.append(str(cache))
        if not batch:
            break
        records.extend(batch)
        update_manifest(slug, "recent_prs", "in_progress", page + 1, len(records), cache_files)
        oldest = min(iso(item["created_at"]) for item in batch)
        if oldest < START:
            break
        page += 1
    filtered = [item for item in records if START <= iso(item["created_at"]) <= CUTOFF]
    update_manifest(slug, "recent_prs", "complete", None, len(filtered), cache_files)
    return filtered


def collect_open(repo: str, slug: str) -> list[dict]:
    owner, name = slug.split("/")
    directory = CACHE / repo / "open_prs"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    page = 1
    cache_files = []
    while True:
        cache = directory / f"page-{page:04d}.json"
        if cache.exists():
            batch = json.loads(cache.read_text(encoding="utf-8"))
        else:
            print(f"{repo}: open PR metadata page {page} ({len(records)} records)", flush=True)
            batch = gh_json([
                f"repos/{owner}/{name}/pulls",
                "-f", "state=open",
                "-f", "sort=created",
                "-f", "direction=desc",
                "-f", "per_page=100",
                "-f", f"page={page}",
                "-X", "GET",
            ])
            atomic_json(cache, batch)
        cache_files.append(str(cache))
        if not batch:
            break
        records.extend(batch)
        update_manifest(slug, "open_prs", "in_progress", page + 1, len(records), cache_files)
        if len(batch) < 100:
            break
        page += 1
    records = [item for item in records if iso(item["created_at"]) <= CUTOFF]
    update_manifest(slug, "open_prs", "complete", None, len(records), cache_files)
    return records


def classify_pr(item: dict, commit_map: dict[int, dict]) -> dict:
    number = int(item["number"])
    if number in commit_map:
        row = commit_map[number]
        return {
            "category": row["primary_purpose"],
            "classifier_source": "matched_squash_commit",
            "classifier_confidence": row["classifier_confidence"],
            "kernel_touch": int(row["touches_kernel_implementation"]),
            "matched_sha": row["sha"],
        }
    pseudo = {"subject": item["title"], "files": []}
    measured = evolution.classify(pseudo)
    return {
        "category": measured["primary_purpose"],
        "classifier_source": "pr_title",
        "classifier_confidence": measured["classifier_confidence"],
        "kernel_touch": int(measured["touches_kernel_implementation"]),
        "matched_sha": "",
    }


def state_at_cutoff(item: dict) -> str:
    merged = iso(item.get("merged_at") or item.get("mergedAt"))
    closed = iso(item.get("closed_at") or item.get("closedAt"))
    if merged and merged <= CUTOFF:
        return "merged"
    if closed and closed <= CUTOFF:
        return "closed_unmerged"
    return "open"


def category_group(category: str) -> str:
    return category if category in {
        "kernel_performance", "kernel_correctness", "hardware_backend", "bugfix"
    } else "other"


def commit_pr_map(repo: str) -> dict[int, dict]:
    mapping = {}
    with (RESULTS / f"{repo}-commits.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            match = re.search(r"\(#(\d+)\)\s*$", row["subject"])
            if match:
                mapping[int(match.group(1))] = row
    return mapping


def sample_recent(repo: str, recent: list[dict]) -> list[dict]:
    sample_file = RESULTS / f"pr-lifecycle-sample-{repo}.csv"
    detail_cache = CACHE / repo / "lifecycle_details"
    if sample_file.exists() and detail_cache.exists() and any(detail_cache.glob("batch-*.json")):
        recent_by_number = {int(item["number"]): item for item in recent}
        with sample_file.open(encoding="utf-8") as handle:
            numbers = [int(row["number"]) for row in csv.DictReader(handle)]
        missing = [number for number in numbers if number not in recent_by_number]
        if missing:
            raise ValueError(f"{repo}: cached lifecycle sample has {len(missing)} PRs outside recent metadata")
        return [recent_by_number[number] for number in numbers]

    mapping = commit_pr_map(repo)
    pools = collections.defaultdict(list)
    for item in recent:
        measured = classify_pr(item, mapping)
        state = state_at_cutoff(item)
        key_category = category_group(measured["category"])
        pools[(key_category, state)].append(item)

    rng = random.Random(SEED + sum(ord(x) for x in repo))
    selected = []
    # Preregistered allocation: oversample pipeline-relevant purposes and
    # non-merged states while retaining a large other-purpose comparison.
    target_by_cat = {
        "kernel_performance": 120,
        "kernel_correctness": 70,
        "hardware_backend": 90,
        "bugfix": 100,
        "other": 220,
    }
    state_weights = {"merged": 0.55, "closed_unmerged": 0.25, "open": 0.20}
    for category, total in target_by_cat.items():
        for state, weight in state_weights.items():
            pool = pools[(category, state)]
            target = min(len(pool), round(total * weight))
            selected.extend(rng.sample(pool, target))
    unique = {int(x["number"]): x for x in selected}
    if len(unique) < 600:
        rest = [x for x in recent if int(x["number"]) not in unique]
        for item in rng.sample(rest, min(600 - len(unique), len(rest))):
            unique[int(item["number"])] = item
    sample = list(unique.values())
    rng.shuffle(sample)
    rows = [{
        "repo": repo,
        "number": x["number"],
        "created_at": x["created_at"],
        "state_at_cutoff": state_at_cutoff(x),
        **classify_pr(x, mapping),
    } for x in sample]
    evolution.write_csv(sample_file, rows)
    return sample


def graphql_batch(slug: str, numbers: list[int]) -> dict:
    owner, name = slug.split("/")
    fragments = []
    for i, number in enumerate(numbers):
        fragments.append(f"""
        p{i}: pullRequest(number: {number}) {{
          number title body state createdAt closedAt mergedAt
          author {{ login __typename }}
          labels(first: 30) {{ nodes {{ name }} }}
          additions deletions changedFiles
          reviews(first: 100) {{ nodes {{ author {{ login __typename }} submittedAt state body }} totalCount }}
          comments(first: 100) {{ nodes {{ author {{ login __typename }} createdAt body }} totalCount }}
        }}""")
    query = f'query {{ repository(owner: "{owner}", name: "{name}") {{ {" ".join(fragments)} }} }}'
    return gh_json(["graphql", "-f", f"query={query}"])


def collect_details(repo: str, slug: str, sample: list[dict]) -> dict[int, dict]:
    directory = CACHE / repo / "lifecycle_details"
    directory.mkdir(parents=True, exist_ok=True)
    numbers = sorted(int(x["number"]) for x in sample)
    details = {}
    cache_files = []
    for offset in range(0, len(numbers), 20):
        batch_numbers = numbers[offset:offset + 20]
        cache = directory / f"batch-{offset // 20:04d}.json"
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
        else:
            print(f"{repo}: lifecycle details {offset}/{len(numbers)}", flush=True)
            payload = graphql_batch(slug, batch_numbers)
            atomic_json(cache, payload)
        cache_files.append(str(cache))
        nodes = payload["data"]["repository"]
        for node in nodes.values():
            if node:
                details[int(node["number"])] = node
        update_manifest(slug, "lifecycle_sample_details", "in_progress", offset + len(batch_numbers), len(details), cache_files)
    update_manifest(slug, "lifecycle_sample_details", "complete", None, len(details), cache_files)
    return details


def evidence(text: str) -> dict:
    low = text.lower()
    hardware_terms = set(re.findall(r"\b(?:h100|h200|a100|a10|l40s|b200|gb200|mi250|mi300x|mi325|mi355|cpu|xpu|tpu|rocm|cuda)\b", low))
    return {
        "evidence_microbenchmark": int(bool(re.search(r"microbench|benchmark.{0,80}(?:\d+(?:\.\d+)?\s*(?:ms|us|µs|%|x)|speedup)", low, re.S))),
        "evidence_end_to_end": int(bool(re.search(r"end[- ]to[- ]end|\be2e\b|tokens?[/ ]s|throughput|ttft|tpot|p99|serving latency", low))),
        "evidence_accuracy_eval": int(bool(re.search(r"lm[-_ ]?eval|gsm8k|mmlu|accuracy eval|quality eval|perplexity", low))),
        "evidence_numerical_tests": int(bool(re.search(r"correctness|numerical|assert[_ ]?close|allclose|atol|rtol|tolerance|unit tests? pass", low))),
        "evidence_multi_hardware": int(len(hardware_terms) >= 2),
        "concern_correctness": int(bool(re.search(r"correctness|wrong result|accuracy|numerical|precision|tolerance|nan|race|crash", low))),
        "concern_other_hardware": int(bool(re.search(r"other (?:gpu|hardware|backend)|rocm|amd|xpu|blackwell|h100|a100|compatib", low))),
        "concern_integration": int(bool(re.search(r"integration|cuda graph|torch\.compile|scheduler|serving|end[- ]to[- ]end|distributed", low))),
        "concern_benchmark_methodology": int(bool(re.search(r"benchmark|baseline|warmup|variance|profil|measurement|reproduc", low))),
        "concern_maintainability": int(bool(re.search(r"maintain|complexity|duplicate|refactor|readab|technical debt|abstraction", low))),
        "concern_tests": int(bool(re.search(r"add (?:a )?test|tests? (?:missing|required|fail)|test coverage|unit test", low))),
        "concern_documentation": int(bool(re.search(r"document|docs|comment|release note", low))),
    }


def author_type(author: dict | None) -> str:
    if not author:
        return "unknown"
    typename = author.get("__typename", "")
    login = author.get("login", "")
    return "bot" if typename == "Bot" or login.lower().endswith("[bot]") or login.lower().endswith("-bot") else "human"


def is_bot(author: dict | None) -> bool:
    return author_type(author) == "bot"


def hours_between(start: dt.datetime, end: dt.datetime | None) -> float | None:
    return (end - start).total_seconds() / 3600 if end else None


def commit_body(repo: str, sha: str) -> str:
    key = (repo, sha)
    if key not in COMMIT_BODY_CACHE:
        proc = subprocess.run(
            [
                "git", "-C", str(EXP / "data" / "repos" / repo),
                "show", "-s", "--format=%B", sha,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Unable to read frozen commit {repo}:{sha}: {proc.stderr[-500:]}")
        COMMIT_BODY_CACHE[key] = proc.stdout
    return COMMIT_BODY_CACHE[key]


def build_rows(repo: str, recent: list[dict], open_items: list[dict], sample: list[dict], details: dict[int, dict]) -> list[dict]:
    mapping = commit_pr_map(repo)
    paths_by_sha = {c["sha"]: c["files"] for c in evolution.parse_log(evolution.REPOS[repo]["log"])}
    population_counts = collections.Counter(
        (category_group(classify_pr(item, mapping)["category"]), state_at_cutoff(item))
        for item in recent
    )
    sample_counts = collections.Counter(
        (category_group(classify_pr(item, mapping)["category"]), state_at_cutoff(item))
        for item in sample
    )
    recent_map = {int(x["number"]): x for x in recent}
    rows = []
    for item in sample:
        number = int(item["number"])
        node = details.get(number)
        if not node:
            continue
        author = (node.get("author") or {}).get("login")
        created = iso(node["createdAt"])
        merged = iso(node.get("mergedAt"))
        closed = iso(node.get("closedAt"))
        if merged and merged <= CUTOFF:
            state = "merged"
        elif closed and closed <= CUTOFF:
            state = "closed_unmerged"
        else:
            state = "open"
        interactions = []
        review_nodes = node["reviews"]["nodes"]
        comment_nodes = node["comments"]["nodes"]
        for review in review_nodes:
            if (
                review.get("submittedAt")
                and (review.get("author") or {}).get("login") != author
                and not is_bot(review.get("author"))
            ):
                interactions.append(iso(review["submittedAt"]))
        for comment in comment_nodes:
            if (
                comment.get("createdAt")
                and (comment.get("author") or {}).get("login") != author
                and not is_bot(comment.get("author"))
            ):
                interactions.append(iso(comment["createdAt"]))
        first_review = min(interactions) if interactions else None
        review_days = {
            (review.get("author") or {}).get("login", review.get("submittedAt", "")[:10])
            for review in review_nodes
            if review.get("submittedAt")
            and (review.get("author") or {}).get("login") != author
            and not is_bot(review.get("author"))
        }
        discussion = "\n".join(
            [node.get("body") or ""]
            + [x.get("body") or "" for x in review_nodes if not is_bot(x.get("author"))]
            + [x.get("body") or "" for x in comment_nodes if not is_bot(x.get("author"))]
        )
        measured = classify_pr(recent_map[number], mapping)
        stratum = (category_group(measured["category"]), state)
        sample_weight = population_counts[stratum] / sample_counts[stratum]
        signals = evidence(discussion)
        body = node.get("body") or ""
        checklist = re.findall(r"(?mi)^\s*[-*]\s*\[([ xX])\]", body)
        changed_paths = paths_by_sha.get(measured["matched_sha"], [])
        source_paths = [p for p in changed_paths if not evolution.TEST_DOC_RE.search(p)]
        test_paths = [p for p in changed_paths if evolution.TEST_DOC_RE.search(p) and re.search(r"test|benchmark", p, re.I)]
        signed_off = (
            int(bool(re.search(r"(?mi)^Signed-off-by:\s*.+<[^>]+>\s*$", commit_body(repo, measured["matched_sha"]))))
            if measured["matched_sha"] else ""
        )
        row = {
            "repo": repo,
            "number": number,
            "title": node["title"],
            "author_type": author_type(node.get("author")),
            "created_at": node["createdAt"],
            "first_non_author_review_or_comment_at": first_review.isoformat() if first_review else "",
            "closed_at": node.get("closedAt") or "",
            "merged_at": node.get("mergedAt") or "",
            "state_at_cutoff": state,
            "labels": ";".join(x["name"] for x in node["labels"]["nodes"]),
            "kernel_touch": measured["kernel_touch"],
            "review_count": node["reviews"]["totalCount"],
            "review_rounds": len(review_days),
            "additions": node["additions"],
            "deletions": node["deletions"],
            "files_changed": node["changedFiles"],
            "linked_or_superseding_signal": int(bool(re.search(r"(?:supersed|replace|follow[- ]?up|fix(?:es)?|close[sd]?)\s+#\d+", discussion, re.I))),
            "revert_or_rollback_signal": int(bool(re.search(r"\b(revert|rollback|rolled back)\b", discussion, re.I))),
            "category": measured["category"],
            "classifier_source": measured["classifier_source"],
            "classifier_confidence": measured["classifier_confidence"],
            "sample_weight": sample_weight,
            "title_starts_with_bracket_tag": int(bool(re.match(r"^\s*\[[^\]]+\]", node["title"]))),
            "checklist_items": len(checklist),
            "checklist_items_completed": sum(x.lower() == "x" for x in checklist),
            "checklist_complete": int(bool(checklist) and all(x.lower() == "x" for x in checklist)),
            "source_change_observed": int(bool(source_paths)),
            "test_or_benchmark_change_observed": int(bool(test_paths)),
            "non_author_approval_observed": int(any(
                review.get("state") == "APPROVED"
                and (review.get("author") or {}).get("login") != author
                and not is_bot(review.get("author"))
                for review in review_nodes
            )),
            "signed_off_by_trailer_observed": signed_off,
            "hours_to_first_review": hours_between(created, first_review),
            "hours_to_merge": hours_between(created, merged if merged and merged <= CUTOFF else None),
            "age_or_resolution_hours": hours_between(created, min(x for x in [merged, closed, CUTOFF] if x and x <= CUTOFF)),
            "abandoned": int(state == "closed_unmerged" or (state == "open" and (CUTOFF - created).days > 90)),
            "review_data_truncated": int(node["reviews"]["totalCount"] > 100 or node["comments"]["totalCount"] > 100),
            **signals,
        }
        rows.append(row)

    # Add the complete open queue as title-classified records with lifecycle
    # detail fields intentionally blank; this supports queue composition/age.
    sampled = {int(x["number"]) for x in sample}
    for item in open_items:
        number = int(item["number"])
        if number in sampled:
            continue
        measured = classify_pr(item, mapping)
        created = iso(item["created_at"])
        rows.append({
            "repo": repo,
            "number": number,
            "title": item["title"],
            "author_type": "bot" if item["user"]["type"] == "Bot" else "human",
            "created_at": item["created_at"],
            "first_non_author_review_or_comment_at": "",
            "closed_at": "",
            "merged_at": "",
            "state_at_cutoff": "open",
            "labels": ";".join(x["name"] for x in item.get("labels", [])),
            "kernel_touch": measured["kernel_touch"],
            "review_count": "",
            "review_rounds": "",
            "additions": "",
            "deletions": "",
            "files_changed": "",
            "linked_or_superseding_signal": "",
            "revert_or_rollback_signal": int(bool(re.search(r"\b(revert|rollback)\b", (item.get("body") or "") + " " + item["title"], re.I))),
            "category": measured["category"],
            "classifier_source": "open_queue_pr_title",
            "classifier_confidence": measured["classifier_confidence"],
            "sample_weight": "",
            "title_starts_with_bracket_tag": int(bool(re.match(r"^\s*\[[^\]]+\]", item["title"]))),
            "checklist_items": len(re.findall(r"(?mi)^\s*[-*]\s*\[([ xX])\]", item.get("body") or "")),
            "checklist_items_completed": sum(
                x.lower() == "x"
                for x in re.findall(r"(?mi)^\s*[-*]\s*\[([ xX])\]", item.get("body") or "")
            ),
            "checklist_complete": int(bool(re.findall(r"(?mi)^\s*[-*]\s*\[([ xX])\]", item.get("body") or "")) and all(
                x.lower() == "x"
                for x in re.findall(r"(?mi)^\s*[-*]\s*\[([ xX])\]", item.get("body") or "")
            )),
            "source_change_observed": "",
            "test_or_benchmark_change_observed": "",
            "non_author_approval_observed": "",
            "signed_off_by_trailer_observed": "",
            "hours_to_first_review": "",
            "hours_to_merge": "",
            "age_or_resolution_hours": (CUTOFF - created).total_seconds() / 3600,
            "abandoned": int((CUTOFF - created).days > 90),
            "review_data_truncated": "",
            **evidence(item.get("body") or ""),
        })
    return rows


def summarize(rows_by_repo: dict[str, list[dict]]) -> None:
    def weighted_mean(group: list[dict], column: str) -> float:
        total = sum(float(r["sample_weight"]) for r in group)
        return sum(float(r["sample_weight"]) * float(r[column]) for r in group) / total

    def weighted_median(group: list[dict], column: str) -> float | str:
        valid = sorted(
            (float(r[column]), float(r["sample_weight"]))
            for r in group if r[column] not in ("", None)
        )
        if not valid:
            return ""
        half = sum(weight for _, weight in valid) / 2
        cumulative = 0.0
        for value, weight in valid:
            cumulative += weight
            if cumulative >= half:
                return value
        return valid[-1][0]

    summaries = []
    for repo, rows in rows_by_repo.items():
        detailed = [r for r in rows if r["classifier_source"] != "open_queue_pr_title"]
        for category in ["kernel_performance", "kernel_correctness", "hardware_backend", "bugfix", "other"]:
            group = [r for r in detailed if (r["category"] if r["category"] in {"kernel_performance", "kernel_correctness", "hardware_backend", "bugfix"} else "other") == category]
            if not group:
                continue
            merged_hours = [r for r in group if r["hours_to_merge"] not in ("", None)]
            review_hours = [r for r in group if r["hours_to_first_review"] not in ("", None)]
            row = {
                "repo": repo,
                "category": category,
                "sample_n": len(group),
                "merged_n": len(merged_hours),
                "median_hours_to_merge": weighted_median(merged_hours, "hours_to_merge"),
                "median_hours_to_first_review": weighted_median(review_hours, "hours_to_first_review"),
                "abandoned_share": weighted_mean(group, "abandoned"),
                "weighted_population_n": sum(float(r["sample_weight"]) for r in group),
                "open_queue_performance_count": "",
                "open_queue_share": "",
            }
            for signal in [
                "evidence_microbenchmark", "evidence_end_to_end", "evidence_accuracy_eval",
                "evidence_numerical_tests", "evidence_multi_hardware", "concern_correctness",
                "concern_other_hardware", "concern_integration", "concern_benchmark_methodology",
                "concern_maintainability", "concern_tests", "concern_documentation",
            ]:
                row[f"{signal}_share"] = weighted_mean(group, signal)
            summaries.append(row)
        queue = [r for r in rows if r["state_at_cutoff"] == "open"]
        perf_queue = [r for r in queue if r["category"] == "kernel_performance"]
        summaries.append({
            "repo": repo,
            "category": "open_queue_kernel_performance",
            "sample_n": len(queue),
            "merged_n": "",
            "median_hours_to_merge": "",
            "median_hours_to_first_review": "",
            "abandoned_share": sum(int(r["abandoned"]) for r in perf_queue) / len(perf_queue) if perf_queue else "",
            "weighted_population_n": "",
            "open_queue_performance_count": len(perf_queue),
            "open_queue_share": len(perf_queue) / len(queue) if queue else 0,
        })
    evolution.write_csv(RESULTS / "pr-evidence-summary.csv", summaries)


def compliance(rows_by_repo: dict[str, list[dict]]) -> None:
    output = []
    for repo, rows in rows_by_repo.items():
        detailed = [r for r in rows if r["classifier_source"] != "open_queue_pr_title"]
        checks = [
            (
                "title_bracket_tag",
                detailed,
                lambda r: int(r["title_starts_with_bracket_tag"]),
                "Preregistered lifecycle sample; format signal is a leading [tag], not semantic tag validity.",
            ),
            (
                "checklist_completion",
                [r for r in detailed if int(r["checklist_items"]) > 0],
                lambda r: int(r["checklist_complete"]),
                "Only PR bodies in the sample that contain markdown checklist items.",
            ),
            (
                "benchmark_evidence_for_performance",
                [r for r in detailed if r["category"] == "kernel_performance"],
                lambda r: int(r["evidence_microbenchmark"]) or int(r["evidence_end_to_end"]),
                "Textual benchmark evidence in PR body, reviews, or issue comments.",
            ),
            (
                "tests_with_source_changes",
                [r for r in detailed if int(r["source_change_observed"]) == 1],
                lambda r: int(r["test_or_benchmark_change_observed"]),
                "Matched squash commits only; detects changed test/benchmark paths, not external CI coverage.",
            ),
            (
                "non_author_approval",
                [r for r in detailed if r["state_at_cutoff"] == "merged"],
                lambda r: int(r["non_author_approval_observed"]),
                "Observable GitHub APPROVED review in first 100 reviews; cannot prove CODEOWNERS status.",
            ),
        ]
        for rule, denominator_rows, predicate, notes in checks:
            compliant = sum(bool(predicate(r)) for r in denominator_rows)
            weighted_denominator = sum(float(r["sample_weight"]) for r in denominator_rows)
            weighted_compliant = sum(
                float(r["sample_weight"]) * bool(predicate(r))
                for r in denominator_rows
            )
            output.append({
                "repo": repo,
                "rule_or_signal": rule,
                "compliant": compliant,
                "denominator": len(denominator_rows),
                "weighted_compliant": weighted_compliant,
                "weighted_denominator": weighted_denominator,
                "compliance_rate": weighted_compliant / weighted_denominator if denominator_rows else "",
                "notes": notes + " Rate is post-stratified to the recent-PR population.",
            })
        dco_rows = [r for r in detailed if r["signed_off_by_trailer_observed"] != ""] if repo == "vllm" else []
        dco_compliant = sum(int(r["signed_off_by_trailer_observed"]) for r in dco_rows)
        output.append({
            "repo": repo,
            "rule_or_signal": "dco_signed_off_by",
            "compliant": dco_compliant if dco_rows else "",
            "denominator": len(dco_rows),
            "weighted_compliant": sum(
                float(r["sample_weight"]) * int(r["signed_off_by_trailer_observed"])
                for r in dco_rows
            ) if dco_rows else "",
            "weighted_denominator": sum(float(r["sample_weight"]) for r in dco_rows) if dco_rows else "",
            "compliance_rate": (
                sum(float(r["sample_weight"]) * int(r["signed_off_by_trailer_observed"]) for r in dco_rows)
                / sum(float(r["sample_weight"]) for r in dco_rows)
            ) if dco_rows else "",
            "notes": (
                "Matched squash-commit trailers in the lifecycle sample; this can undercount DCO checks on pre-squash commits."
                if dco_rows else "No repository-wide DCO requirement was identified in the frozen SGLang rules."
            ),
        })
        output.append({
            "repo": repo,
            "rule_or_signal": "required_label_before_full_ci",
            "compliant": "",
            "denominator": 0,
            "weighted_compliant": "",
            "weighted_denominator": "",
            "compliance_rate": "",
            "notes": "Not measured: current labels do not preserve when a label was applied relative to CI.",
        })
    evolution.write_csv(RESULTS / "rule-compliance-observed.csv", output)


def make_figures(rows_by_repo: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (repo, rows) in zip(axes, rows_by_repo.items()):
        detailed = [r for r in rows if r["classifier_source"] != "open_queue_pr_title" and r["hours_to_merge"] not in ("", None)]
        for name, predicate, color in [
            ("kernel performance", lambda r: r["category"] == "kernel_performance", "#d62728"),
            ("other", lambda r: r["category"] != "kernel_performance", "#1f77b4"),
        ]:
            values = sorted(float(r["hours_to_merge"]) / 24 for r in detailed if predicate(r))
            if values:
                y = [(i + 1) / len(values) for i in range(len(values))]
                ax.step(values, y, where="post", label=f"{name} (n={len(values)})", color=color)
        ax.set_xscale("log")
        ax.set_xlabel("Days to merge (log scale)")
        ax.set_title(repo)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Empirical cumulative share")
    fig.suptitle("Time to merge in the preregistered lifecycle sample")
    fig.text(0.01, 0.01, f"PRs created {START.date()}–{CUTOFF.date()}; merged by cutoff. Descriptive ECDF, not a causal comparison.", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(figures / "pr-time-to-merge-ecdf.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (repo, rows) in zip(axes, rows_by_repo.items()):
        ages = [float(r["age_or_resolution_hours"]) / 24 for r in rows if r["state_at_cutoff"] == "open"]
        bins = [0, 7, 30, 90, 180, 365, 730, max(731, math.ceil(max(ages, default=731)))]
        ax.hist(ages, bins=bins, color="#ff7f0e", edgecolor="white")
        ax.set_xscale("log")
        ax.set_xlabel("Open PR age in days (log bins)")
        ax.set_title(f"{repo} (n={len(ages):,})")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Open PRs")
    fig.suptitle("Age distribution of the complete open-PR queue at cutoff")
    fig.text(0.01, 0.01, f"All PRs open in the GitHub REST listing and created by {CUTOFF.date()}; title-only category for unsampled records.", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(figures / "open-pr-age-distribution.png", dpi=170)
    plt.close(fig)

    with (RESULTS / "rule-compliance-observed.csv").open(encoding="utf-8") as handle:
        compliance_rows = [r for r in csv.DictReader(handle) if r["compliance_rate"] != ""]
    labels = sorted({r["rule_or_signal"] for r in compliance_rows})
    fig, ax = plt.subplots(figsize=(12, 5.5))
    width = 0.36
    x = list(range(len(labels)))
    for offset, repo in [(-width / 2, "vllm"), (width / 2, "sglang")]:
        by_rule = {r["rule_or_signal"]: r for r in compliance_rows if r["repo"] == repo}
        values = [float(by_rule[label]["compliance_rate"]) if label in by_rule else float("nan") for label in labels]
        bars = ax.bar([i + offset for i in x], values, width, label=repo)
        for bar, label in zip(bars, labels):
            if label in by_rule:
                row = by_rule[label]
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f"{row['compliant']}/{row['denominator']}", ha="center", fontsize=7, rotation=90)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Observed compliance share")
    ax.set_xticks(x, [label.replace("_", " ") for label in labels], rotation=25, ha="right")
    ax.set_title("Codified-rule and review-signal compliance in the lifecycle sample")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.01, 0.01, "Signals have different denominators; see rule-compliance-observed.csv. Current labels cannot establish label-before-CI timing.", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(figures / "codified-rules-compliance.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["collect", "analyze", "all"])
    args = parser.parse_args()
    try:
        recent_by_repo = {}
        open_by_repo = {}
        if args.command in {"collect", "all"}:
            for repo, slug in REPOS.items():
                recent_by_repo[repo] = collect_recent(repo, slug)
                open_by_repo[repo] = collect_open(repo, slug)
                sample = sample_recent(repo, recent_by_repo[repo])
                collect_details(repo, slug, sample)
        if args.command in {"analyze", "all"}:
            rows_by_repo = {}
            for repo, slug in REPOS.items():
                recent = collect_recent(repo, slug)
                open_items = collect_open(repo, slug)
                sample = sample_recent(repo, recent)
                details = collect_details(repo, slug, sample)
                rows = build_rows(repo, recent, open_items, sample, details)
                evolution.write_csv(RESULTS / f"pr-lifecycle-{repo}.csv", rows)
                rows_by_repo[repo] = rows
            summarize(rows_by_repo)
            compliance(rows_by_repo)
            make_figures(rows_by_repo)
    except Exception as error:
        if STATUS.exists():
            status = json.loads(STATUS.read_text(encoding="utf-8"))
            status["failures"].append({
                "phase": "05-pr-lifecycle",
                "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                "error": f"{type(error).__name__}: {error}",
                "resume": "Rerun python experiments/scripts/pr_lifecycle.py all; completed cache batches are reused.",
            })
            status["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            atomic_json(STATUS, status)
        raise


if __name__ == "__main__":
    main()
