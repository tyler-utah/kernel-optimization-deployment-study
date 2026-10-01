"""A3: detailed review/comment/timeline corpus.

targets <tier>  - write/extend target list(s) data/cache/a3/targets-<repo>.json
collect [repo]  - fetch full reviews, review comments, issue comments, commits and
                  timeline events for every target, 10 PRs per GraphQL call,
                  cached per batch; checkpoint after every batch
"""

from __future__ import annotations

import sys

from common import CACHE, REPOS, atomic_json, graphql, now_utc, read_json, read_jsonl, update_status

A3 = CACHE / "a3"

DETAIL_Q = """
p{i}: pullRequest(number: {n}) {{
  number title body state isDraft createdAt closedAt mergedAt additions deletions changedFiles
  author {{ login __typename }} authorAssociation
  mergedBy {{ login }}
  labels(first: 30) {{ nodes {{ name }} }}
  files(first: 100) {{ totalCount nodes {{ path additions deletions }} }}
  commits(first: 100) {{ totalCount nodes {{ commit {{ oid committedDate messageHeadline }} }} }}
  reviews(first: 60) {{ totalCount nodes {{ author {{ login __typename }} authorAssociation submittedAt state body
      comments(first: 30) {{ totalCount nodes {{ author {{ login __typename }} createdAt path body }} }} }} }}
  comments(first: 100) {{ totalCount nodes {{ author {{ login __typename }} authorAssociation createdAt body }} }}
  timelineItems(first: 100, itemTypes: [LABELED_EVENT, UNLABELED_EVENT, REVIEW_REQUESTED_EVENT,
      READY_FOR_REVIEW_EVENT, CONVERT_TO_DRAFT_EVENT, CLOSED_EVENT, REOPENED_EVENT, MERGED_EVENT,
      CROSS_REFERENCED_EVENT, HEAD_REF_FORCE_PUSHED_EVENT]) {{ totalCount nodes {{ __typename
      ... on LabeledEvent {{ createdAt label {{ name }} actor {{ login }} }}
      ... on UnlabeledEvent {{ createdAt label {{ name }} }}
      ... on ReviewRequestedEvent {{ createdAt }}
      ... on ReadyForReviewEvent {{ createdAt }}
      ... on ConvertToDraftEvent {{ createdAt }}
      ... on ClosedEvent {{ createdAt closer {{ __typename ... on PullRequest {{ number }} ... on Commit {{ oid }} }} }}
      ... on ReopenedEvent {{ createdAt }}
      ... on MergedEvent {{ createdAt commit {{ oid }} }}
      ... on HeadRefForcePushedEvent {{ createdAt }}
      ... on CrossReferencedEvent {{ createdAt willCloseTarget source {{ __typename
          ... on PullRequest {{ number title state mergedAt createdAt repository {{ nameWithOwner }} }}
          ... on Issue {{ number title repository {{ nameWithOwner }} }} }} }} }} }}
}}"""


def targets_path(repo: str):
    return A3 / f"targets-{repo}.json"


def add_targets(repo: str, numbers: list[int], reason: str) -> None:
    cur = read_json(targets_path(repo), {})
    for n in numbers:
        cur.setdefault(str(n), reason)
    atomic_json(targets_path(repo), cur)
    print(repo, "targets", len(cur))


def cached_numbers(repo: str) -> set[int]:
    have = set()
    for p in (A3 / "detail" / repo).glob("*.json"):
        d = read_json(p)
        have.update(d.get("_numbers", []))
    return have


def load_details(repo: str) -> dict[int, dict]:
    out = {}
    for p in sorted((A3 / "detail" / repo).glob("*.json")):
        for node in ((read_json(p).get("data") or {}).get("repository") or {}).values():
            if node:
                out[int(node["number"])] = node
    return out


def collect(repo: str) -> None:
    owner, name = REPOS[repo].split("/")
    want = sorted(int(n) for n in read_json(targets_path(repo), {}))
    have = cached_numbers(repo)
    todo = [n for n in want if n not in have]
    (A3 / "detail" / repo).mkdir(parents=True, exist_ok=True)
    print(repo, "detail targets", len(want), "cached", len(want) - len(todo), "todo", len(todo), flush=True)
    for i in range(0, len(todo), 10):
        chunk = todo[i:i + 10]
        frag = "\n".join(DETAIL_Q.format(i=j, n=n) for j, n in enumerate(chunk))
        payload = graphql(f'query {{ repository(owner: "{owner}", name: "{name}") {{ {frag} }} rateLimit {{ cost remaining resetAt }} }}')
        payload["_numbers"] = chunk
        payload["_fetched_at_utc"] = now_utc()
        atomic_json(A3 / "detail" / repo / f"{chunk[0]}-{chunk[-1]}.json", payload, indent=None)
        rl = payload["data"].get("rateLimit", {})
        print(repo, "detail", i + len(chunk), "/", len(todo), "remaining", rl.get("remaining"), flush=True)
        value = {"targets": len(want), "cached": len(want) - len(todo) + i + len(chunk), "next_number": todo[i + 10] if i + 10 < len(todo) else None,
                 "rate_limit": rl, "at_utc": now_utc(),
                 "resume": f"python experiments/deep-study/scripts/a3_details.py collect {repo}"}
        update_status(lambda d: d.setdefault("api", {}).setdefault("cursors", {}).__setitem__(f"a3_detail_{repo}", value))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "targets":
        tier = sys.argv[2]
        for repo in REPOS:
            pop = read_jsonl(CACHE / "a1" / f"population-{repo}.jsonl")
            add_targets(repo, [r["number"] for r in pop if r["tier"] == tier and r["in_window"]], tier)
    elif cmd == "collect":
        for repo in (sys.argv[2:] or list(REPOS)):
            collect(repo)
