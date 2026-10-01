"""B2/B3: confirmed kernel-correctness corpus.

frame     - all first-parent commits (24-month window) that touch production kernel
            paths and whose subject has fix/failure language; seeded random order
context   - GraphQL context for sampled PRs (body, closing issues, comments, files)
dossiers  - screening+coding batches (codebook-b2-failures.md)
intro     - fetch the referenced introducing PRs for confirmed cases (stage 2)
"""

from __future__ import annotations

import re
import sys

from a1_population import KERNEL_PATH_RE, TEST_DOC_RE, clean_body
from b1_reverts import is_bot, log_window
from common import (BATCHES, CACHE, REPOS, atomic_json, graphql, read_json, read_jsonl, rng,
                    update_status, now_utc, write_jsonl)

FIXRE = re.compile(
    r"\b(fix\w*|bug\w*|hotfix|wrong|incorrect|nan|inf\b|overflow|underflow|race|deadlock|hang\w*|"
    r"illegal (?:memory|address|instruction)|\bima\b|out[- ]of[- ]bounds?|\boob\b|segfault|crash\w*|"
    r"accuracy|mismatch\w*|precision|garbage|corrupt\w*|misalign\w*|uninitiali[sz]ed|invalid|"
    r"nondetermin\w*|non-determin\w*|regression)\b", re.I)
SAMPLE_PER_REPO = 220
D = CACHE / "b2"


def frame() -> None:
    for repo in REPOS:
        rows = []
        for c in log_window(repo):
            kp = [p for p in c["files"] if KERNEL_PATH_RE.search(p) and not TEST_DOC_RE.search(p)]
            if not kp or not FIXRE.search(c["subject"]):
                continue
            if re.match(r"^\s*(?:\[[^\]]*\]\s*)*revert", c["subject"], re.I):
                continue
            m = re.search(r"\(#(\d+)\)\s*$", c["subject"])
            rows.append({"repo": repo, "sha": c["sha"], "pr": int(m.group(1)) if m else None,
                         "commit_date": c["commit_date"], "subject": c["subject"], "body": c["body"][:3000],
                         "files": c["files"][:120], "kernel_paths": kp[:40],
                         "test_paths": [p for p in c["files"] if re.search(r"(^|/)tests?/|test_", p)][:30]})
        rows.sort(key=lambda r: r["sha"])
        rng(f"b2-frame-{repo}").shuffle(rows)
        for i, r in enumerate(rows):
            r["frame_rank"] = i
        write_jsonl(D / f"frame-{repo}.jsonl", rows)
        print(repo, "frame", len(rows))


CTX_Q = """
p{i}: pullRequest(number: {n}) {{
  number title body createdAt mergedAt author {{ login __typename }}
  labels(first: 20) {{ nodes {{ name }} }}
  files(first: 80) {{ totalCount nodes {{ path additions deletions }} }}
  closingIssuesReferences(first: 5) {{ nodes {{ number title createdAt body author {{ login }} }} }}
  comments(first: 12) {{ nodes {{ author {{ login __typename }} createdAt body }} }}
  reviews(first: 10) {{ nodes {{ author {{ login __typename }} submittedAt state body }} }}
}}"""


def sample(repo: str, upto: int) -> list[dict]:
    return [r for r in read_jsonl(D / f"frame-{repo}.jsonl") if r["frame_rank"] < upto]


def context(upto: int = SAMPLE_PER_REPO) -> None:
    for repo in REPOS:
        owner, name = REPOS[repo].split("/")
        nums = sorted({r["pr"] for r in sample(repo, upto) if r["pr"]})
        (D / "ctx" / repo).mkdir(parents=True, exist_ok=True)
        for i in range(0, len(nums), 20):
            chunk = nums[i:i + 20]
            cache = D / "ctx" / repo / f"{chunk[0]}-{chunk[-1]}.json"
            if cache.exists():
                continue
            frag = "\n".join(CTX_Q.format(i=j, n=n) for j, n in enumerate(chunk))
            payload = graphql(f'query {{ repository(owner: "{owner}", name: "{name}") {{ {frag} }} }}')
            atomic_json(cache, payload, indent=None)
            print(repo, "b2 ctx", i + len(chunk), "/", len(nums), flush=True)
        value = {"repo": repo, "prs": len(nums), "sample_upto_rank": upto, "at_utc": now_utc()}
        update_status(lambda d: d.setdefault("api", {}).setdefault("cursors", {}).__setitem__(f"b2_ctx_{repo}", value))


def load_ctx(repo: str, sub: str = "ctx") -> dict[int, dict]:
    out = {}
    for p in sorted((D / sub / repo).glob("*.json")):
        for node in ((read_json(p).get("data") or {}).get("repository") or {}).values():
            if node:
                out[int(node["number"])] = node
    return out


def clip(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " …[truncated]"


def dossiers(upto: int = SAMPLE_PER_REPO, start_rank: int = 0, tag: str = "") -> None:
    items = []
    for repo in REPOS:
        ctx = load_ctx(repo)
        for r in sample(repo, upto):
            if r["frame_rank"] < start_rank:
                continue
            pr = ctx.get(r["pr"] or -1, {})
            files = [(f["path"], f["additions"], f["deletions"]) for f in ((pr.get("files") or {}).get("nodes") or [])]
            disc = [c for c in ((pr.get("reviews") or {}).get("nodes") or []) + ((pr.get("comments") or {}).get("nodes") or [])
                    if (c.get("body") or "").strip() and not is_bot(c.get("author"))]
            body = clean_body(pr.get("body") or r["body"])
            items.append({
                "id": f"{repo}:{r['sha'][:10]}", "repo": repo, "sha": r["sha"], "pr": r["pr"],
                "url": f"https://github.com/{REPOS[repo]}/pull/{r['pr']}" if r["pr"] else f"https://github.com/{REPOS[repo]}/commit/{r['sha']}",
                "merged": r["commit_date"], "subject": r["subject"], "labels": [x["name"] for x in ((pr.get("labels") or {}).get("nodes") or [])],
                "pr_body": clip(body, 2200),
                "linked_issues": [{"number": x["number"], "created": x["createdAt"], "title": x["title"],
                                   "body": clip(re.sub(r"<!--.*?-->", "", x.get("body") or "", flags=re.S), 700)}
                                  for x in ((pr.get("closingIssuesReferences") or {}).get("nodes") or [])][:3],
                "discussion": [clip(f"{(c.get('author') or {}).get('login', '?')}: {c['body']}", 350) for c in disc[:6]],
                "files_changed": [f"{p} (+{a}/-{d})" for p, a, d in files][:30] or r["files"][:30],
                "kernel_paths": r["kernel_paths"][:15], "test_paths_in_fix": r["test_paths"][:10],
                "referenced_prs_or_issues": sorted({int(x) for x in re.findall(r"#(\d{3,6})", body + " " + r["subject"]) if int(x) != r["pr"]})[:10],
            })
    out = BATCHES / f"b2_failures{tag}"
    out.mkdir(parents=True, exist_ok=True)
    size = 25
    rng("b2-batches" + tag).shuffle(items)
    for i in range(0, len(items), size):
        write_jsonl(out / f"batch-{i // size:03d}-input.jsonl", items[i:i + size])
    print("b2 dossiers", len(items), "batches", (len(items) + size - 1) // size)


INTRO_Q = """
p{i}: pullRequest(number: {n}) {{
  number title body createdAt mergedAt author {{ login __typename }}
  labels(first: 20) {{ nodes {{ name }} }}
  files(first: 60) {{ totalCount nodes {{ path additions deletions }} }}
  comments(first: 15) {{ nodes {{ author {{ login __typename }} body }} }}
  reviews(first: 15) {{ nodes {{ author {{ login __typename }} state body }} }}
}}"""


def intro_refs() -> dict[str, list[int]]:
    from batches import load_outputs
    out = {}
    for cid, o in load_outputs("b2_failures").items():
        if o["case_status"] != "confirmed_kernel_correctness":
            continue
        nums = [int(x) for x in re.findall(r"#(\d{2,6})", o.get("introducing_ref") or "")]
        if nums:
            out[cid] = nums
    return out


def intro() -> None:
    refs = intro_refs()
    for repo in REPOS:
        owner, name = REPOS[repo].split("/")
        nums = sorted({n for cid, ns in refs.items() if cid.startswith(repo + ":") for n in ns})
        (D / "intro" / repo).mkdir(parents=True, exist_ok=True)
        for i in range(0, len(nums), 20):
            chunk = nums[i:i + 20]
            cache = D / "intro" / repo / f"{chunk[0]}-{chunk[-1]}.json"
            if cache.exists():
                continue
            frag = "\n".join(INTRO_Q.format(i=j, n=n) for j, n in enumerate(chunk))
            atomic_json(cache, graphql(f'query {{ repository(owner: "{owner}", name: "{name}") {{ {frag} }} }}'), indent=None)
        print(repo, "introducing PRs", len(nums))
    # dossiers: one per case with an introducing PR
    items = []
    for repo in REPOS:
        ctx = load_ctx(repo, "intro")
        for cid, ns in refs.items():
            if not cid.startswith(repo + ":"):
                continue
            for n in ns[:2]:
                pr = ctx.get(n)
                if not pr:
                    continue
                disc = [c for c in ((pr.get("reviews") or {}).get("nodes") or []) + ((pr.get("comments") or {}).get("nodes") or [])
                        if (c.get("body") or "").strip() and not is_bot(c.get("author"))]
                items.append({
                    "id": f"{cid}|{n}", "case_id": cid, "introducing_pr": n,
                    "url": f"https://github.com/{REPOS[repo]}/pull/{n}", "title": pr["title"],
                    "merged_at": pr.get("mergedAt"), "labels": [x["name"] for x in ((pr.get("labels") or {}).get("nodes") or [])],
                    "files": [f"{f['path']} (+{f['additions']}/-{f['deletions']})" for f in ((pr.get("files") or {}).get("nodes") or [])][:25],
                    "body": clip(clean_body(pr.get("body") or ""), 3000),
                    "discussion": [clip(f"{(c.get('author') or {}).get('login', '?')}: {c['body']}", 350) for c in disc[:8]],
                })
    out = BATCHES / "b2_intro"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(items), 35):
        write_jsonl(out / f"batch-{i // 35:03d}-input.jsonl", items[i:i + 35])
    import json
    (out / "schema.json").write_text(json.dumps({
        "introducing_pr_is_performance": ["yes", "no", "unclear"],
        "reported_correctness_tests": ["yes", "no"], "reported_accuracy_eval": ["yes", "no"],
        "reported_multi_hardware": ["yes", "no"], "evidence_snippet": None, "rationale": None,
        "confidence": ["high", "medium", "low"]}))
    print("b2_intro dossiers", len(items))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "frame":
        frame()
    elif cmd == "context":
        context(int(sys.argv[2]) if len(sys.argv) > 2 else SAMPLE_PER_REPO)
    elif cmd == "dossiers":
        dossiers()
    elif cmd == "intro":
        intro()
