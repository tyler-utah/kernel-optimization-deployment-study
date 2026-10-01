"""A1/A2 input: complete 12-month PR metadata.

compact  - reduce the first study's REST caches to one JSONL per repository
enrich   - GraphQL enrichment (size, files, labels, author type, closing and
           cross-reference events, last comments) in cached 50-PR batches
"""

from __future__ import annotations

import glob
import json
import sys

from common import (A_START, CACHE, CUTOFF, FIRST_CACHE, REPOS, atomic_json, graphql, iso,
                    now_utc, read_json, read_jsonl, update_status, write_jsonl)

BATCH = 50


def meta_path(repo: str):
    return CACHE / repo / "meta" / "prs.jsonl"


def compact(repo: str) -> None:
    out = {}
    for kind in ["recent_prs", "open_prs"]:
        for path in sorted(glob.glob(str(FIRST_CACHE / repo / kind / "page-*.json"))):
            for item in json.load(open(path, encoding="utf-8")) or []:
                n = int(item["number"])
                rec = {
                    "number": n,
                    "title": item["title"],
                    "body": (item.get("body") or "")[:30000],
                    "rest_state": item["state"],
                    "created_at": item["created_at"],
                    "closed_at": item.get("closed_at"),
                    "merged_at": item.get("merged_at"),
                    "user": (item.get("user") or {}).get("login"),
                    "user_type": (item.get("user") or {}).get("type"),
                    "author_association": item.get("author_association"),
                    "labels": [x["name"] for x in item.get("labels", [])],
                    "draft": item.get("draft"),
                    "merge_commit_sha": item.get("merge_commit_sha"),
                    "source": kind,
                }
                prev = out.get(n)
                if prev is None or kind == "recent_prs":
                    if prev:
                        rec["source"] = "recent_prs+open_prs"
                    out[n] = rec
                elif prev and kind == "open_prs":
                    prev["source"] = "recent_prs+open_prs"
    rows = sorted(out.values(), key=lambda r: r["number"])
    created_ok = [r for r in rows if iso(r["created_at"]) <= CUTOFF]
    write_jsonl(meta_path(repo), created_ok)
    in_window = sum(A_START <= iso(r["created_at"]) <= CUTOFF for r in created_ok)
    print(repo, "compact records", len(created_ok), "in 12m window", in_window)


QUERY_PR = """
p{i}: pullRequest(number: {n}) {{
  number state isDraft createdAt closedAt mergedAt additions deletions changedFiles
  author {{ login __typename }} authorAssociation
  labels(first: 25) {{ nodes {{ name }} }}
  files(first: 100) {{ totalCount nodes {{ path additions deletions }} }}
  reviews(first: 1) {{ totalCount }}
  comments(last: 3) {{ totalCount nodes {{ author {{ login __typename }} createdAt body }} }}
  mergeCommit {{ oid }}
  timelineItems(last: 12, itemTypes: [CLOSED_EVENT, CROSS_REFERENCED_EVENT, REOPENED_EVENT]) {{
    nodes {{ __typename
      ... on ClosedEvent {{ createdAt actor {{ login }} closer {{ __typename ... on PullRequest {{ number }} ... on Commit {{ oid }} }} }}
      ... on ReopenedEvent {{ createdAt }}
      ... on CrossReferencedEvent {{ createdAt willCloseTarget isCrossRepository source {{ __typename
          ... on PullRequest {{ number state createdAt mergedAt title repository {{ nameWithOwner }} }}
          ... on Issue {{ number repository {{ nameWithOwner }} }} }} }}
    }}
  }}
}}"""


QUERY_LIGHT = """
p{i}: pullRequest(number: {n}) {{
  number state isDraft createdAt closedAt mergedAt additions deletions changedFiles
  author {{ login __typename }} authorAssociation
  labels(first: 25) {{ nodes {{ name }} }}
  files(first: 100) {{ totalCount nodes {{ path }} }}
  reviews(first: 1) {{ totalCount }}
  comments(first: 1) {{ totalCount }}
  mergeCommit {{ oid }}
}}"""


def enrich(repo: str, shard: int = 0, nshards: int = 1) -> None:
    """v2: merged PRs get a light query; non-merged PRs keep the heavy query
    (closing/cross-reference events and last comments for superseded checks).
    PRs already present in v1 ``batch-*.json`` caches are skipped."""
    slug = REPOS[repo]
    owner, name = slug.split("/")
    rows = read_jsonl(meta_path(repo))
    rows = [r for r in rows if iso(r["created_at"]) >= A_START or "open_prs" in r["source"]]
    directory = CACHE / repo / "enrich"
    directory.mkdir(parents=True, exist_ok=True)
    have = set(load_enriched(repo))
    merged = sorted(r["number"] for r in rows if r.get("merged_at") and r["number"] not in have)
    other = sorted(r["number"] for r in rows if not r.get("merged_at") and r["number"] not in have)
    jobs = [("light", merged[i:i + BATCH], QUERY_LIGHT) for i in range(0, len(merged), BATCH)]
    # 20-PR heavy batches stay under GitHub's ~10 s server-side query timeout
    jobs += [("heavy", other[i:i + 20], QUERY_PR) for i in range(0, len(other), 20)]
    total = len(jobs)
    done = 0
    for idx, (kind, batch, template) in enumerate(jobs):
        if idx % nshards != shard:
            continue
        cache = directory / f"v2-{kind}-{batch[0]}-{batch[-1]}.json"
        if cache.exists():
            continue
        frag = "\n".join(template.format(i=i, n=n) for i, n in enumerate(batch))
        query = f'query {{ repository(owner: "{owner}", name: "{name}") {{ {frag} }} rateLimit {{ cost remaining resetAt }} }}'
        payload = graphql(query)
        payload["_numbers"] = batch
        payload["_kind"] = kind
        payload["_fetched_at_utc"] = now_utc()
        atomic_json(cache, payload, indent=None)
        done += 1
        rl = payload["data"].get("rateLimit", {})
        print(f"{repo} {kind} job {idx}/{total} cost={rl.get('cost')} remaining={rl.get('remaining')}", flush=True)
        if done % 2 == 0:  # checkpoint every <=100 PRs
            _checkpoint(repo, total, idx, rl)
    _checkpoint(repo, total, "complete", {})


def _checkpoint(repo: str, total: int, idx, rl) -> None:
    directory = CACHE / repo / "enrich"
    have = len(list(directory.glob("*.json")))
    value = {"cache_files": have, "v2_jobs_total_at_start": total, "last_job": idx,
             "batch_size": BATCH, "rate_limit": rl, "at_utc": now_utc(),
             "resume": f"python experiments/deep-study/scripts/a1_metadata.py enrich {repo}"}
    update_status(lambda d: d.setdefault("api", {}).setdefault("cursors", {}).__setitem__(f"enrich_{repo}", value))


def load_enriched(repo: str, v1_only: bool = False) -> dict[int, dict]:
    out = {}
    pattern = "batch-*.json" if v1_only else "*.json"
    for path in sorted((CACHE / repo / "enrich").glob(pattern)):
        payload = read_json(path)
        for key, node in ((payload.get("data") or {}).get("repository") or {}).items():
            if node:
                out[int(node["number"])] = node
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    repos = [sys.argv[2]] if len(sys.argv) > 2 else list(REPOS)
    if cmd == "compact":
        for r in repos:
            compact(r)
    elif cmd == "enrich":
        shard = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        nshards = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        for r in repos:
            enrich(r, shard, nshards)
