"""B1: 24-month revert census.

candidates - scan first-parent history (committer date in window) for revert /
             rollback language; resolve reverted commit/PR hints locally
context    - GraphQL fetch revert PR bodies/comments and reverted-PR metadata
dossiers   - build adjudication batches for contextual confirmation
merge      - merge adjudications -> data/confirmed-reverts.csv
"""

from __future__ import annotations

import re
import sys

from common import (B_START, BATCHES, CACHE, CUTOFF, DATA, REPOS, atomic_csv, atomic_json, git,
                    graphql, iso, now_utc, read_json, read_jsonl, set_subtask, update_status,
                    write_jsonl)

REVERT_RE = re.compile(r"\brevert|\broll(?:ed|ing)?[- ]?back\b|\bback(?:ed)?[- ]?out\b|\bundo\b|\breapply|\breland", re.I)
SEP = "\x1e"


def log_window(repo: str) -> list[dict]:
    out = git(repo, "log", "--first-parent", f"--since={B_START.isoformat()}",
              f"--until={CUTOFF.isoformat()}", f"--format={SEP}%H%x1f%aI%x1f%cI%x1f%s%x1f%b",
              "--name-only", "--no-renames", "HEAD", timeout=900)
    commits = []
    for chunk in out.split(SEP)[1:]:
        head, _, rest = chunk.partition("\x1f")
        sha = head
        adate, cdate, subject, tail = rest.split("\x1f", 3)
        # body and file list are separated by a blank line after the body
        body, files = tail, []
        lines = tail.rstrip("\n").split("\n")
        # name-only files follow the body; they are the trailing lines without spaces after last blank line
        if "\n\n" in tail:
            body_part, _, file_part = tail.rstrip("\n").rpartition("\n\n")
            cand = [x for x in file_part.split("\n") if x]
            if cand and all(" " not in x and ("/" in x or "." in x) for x in cand):
                body, files = body_part, cand
        commits.append({"sha": sha, "author_date": adate, "commit_date": cdate, "subject": subject,
                        "body": body.strip(), "files": files})
    return commits


def pr_number(subject: str):
    m = re.search(r"\(#(\d+)\)\s*$", subject)
    return int(m.group(1)) if m else None


def candidates(repo: str) -> None:
    commits = log_window(repo)
    by_sha = {c["sha"]: c for c in commits}
    # also map every first-parent commit in full history for reverted-target lookup
    full = git(repo, "log", "--first-parent", "--format=%H%x1f%aI%x1f%s", "HEAD", timeout=900)
    full_map, by_pr = {}, {}
    for line in full.splitlines():
        sha, date, subj = line.split("\x1f", 2)
        full_map[sha] = {"date": date, "subject": subj}
        n = pr_number(subj)
        if n:
            by_pr.setdefault(n, {"sha": sha, "date": date, "subject": subj})
    rows = []
    for c in commits:
        if not REVERT_RE.search(c["subject"]):
            continue
        hints = []
        for sha in re.findall(r"This reverts commit ([0-9a-f]{7,40})", c["body"]):
            tgt = next((v | {"sha": k} for k, v in full_map.items() if k.startswith(sha)), None)
            hints.append({"via": "reverts_commit", "sha": sha, "subject": tgt["subject"] if tgt else "",
                          "pr": pr_number(tgt["subject"]) if tgt else None, "date": tgt["date"] if tgt else ""})
        own = pr_number(c["subject"])
        nums = [int(x) for x in re.findall(r"#(\d+)", c["subject"]) if int(x) != own]
        nums += [int(x) for x in re.findall(r"(?:revert|reverts|reverting|rollback|roll back|undo)[^\n#]{0,60}#(\d+)", c["body"], re.I)]
        nums += [int(x) for x in re.findall(r"github\.com/[\w.-]+/[\w.-]+/pull/(\d+)", c["body"])][:3]
        for n in dict.fromkeys(nums):
            tgt = by_pr.get(n)
            hints.append({"via": "pr_ref", "pr": n, "sha": tgt["sha"] if tgt else "",
                          "subject": tgt["subject"] if tgt else "", "date": tgt["date"] if tgt else ""})
        quoted = re.match(r'^\s*(?:\[[^\]]*\]\s*)*revert\s*"(.+?)"', c["subject"], re.I)
        if quoted and not hints:
            title = quoted.group(1)
            for n, v in by_pr.items():
                if v["subject"].startswith(title[:60]):
                    hints.append({"via": "quoted_title", "pr": n, "sha": v["sha"], "subject": v["subject"], "date": v["date"]})
                    break
        rows.append({"repo": repo, "revert_sha": c["sha"], "revert_pr": own, "author_date": c["author_date"],
                     "commit_date": c["commit_date"], "subject": c["subject"], "body": c["body"][:4000],
                     "files": c["files"][:80], "n_files": len(c["files"]), "target_hints": hints})
    write_jsonl(CACHE / repo / "reverts" / "candidates.jsonl", rows)
    print(repo, "revert-language candidates", len(rows), "of", len(commits), "first-parent commits in window")


CTX_Q = """
p{i}: pullRequest(number: {n}) {{
  number title body createdAt mergedAt closedAt state additions deletions changedFiles
  author {{ login __typename }} labels(first: 20) {{ nodes {{ name }} }}
  files(first: 60) {{ totalCount nodes {{ path }} }}
  comments(first: 40) {{ nodes {{ author {{ login __typename }} createdAt body }} }}
  reviews(first: 20) {{ nodes {{ author {{ login __typename }} submittedAt state body }} }}
  timelineItems(first: 30, itemTypes: [CROSS_REFERENCED_EVENT]) {{ nodes {{ ... on CrossReferencedEvent {{ createdAt source {{ __typename ... on PullRequest {{ number title state mergedAt }} ... on Issue {{ number title state }} }} }} }} }}
}}"""


def context(repo: str) -> None:
    owner, name = REPOS[repo].split("/")
    rows = read_jsonl(CACHE / repo / "reverts" / "candidates.jsonl")
    wanted = set()
    for r in rows:
        if r["revert_pr"]:
            wanted.add(r["revert_pr"])
        for h in r["target_hints"]:
            if h.get("pr"):
                wanted.add(int(h["pr"]))
    wanted = sorted(wanted)
    directory = CACHE / repo / "reverts" / "ctx"
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(wanted), 20):
        cache = directory / f"batch-{i // 20:04d}.json"
        if cache.exists():
            continue
        chunk = wanted[i:i + 20]
        frag = "\n".join(CTX_Q.format(i=j, n=n) for j, n in enumerate(chunk))
        payload = graphql(f'query {{ repository(owner: "{owner}", name: "{name}") {{ {frag} }} }}')
        payload["_numbers"] = chunk
        atomic_json(cache, payload, indent=None)
        print(repo, "revert ctx", i + len(chunk), "/", len(wanted), flush=True)
        if (i // 20) % 5 == 4:
            _cur(repo, i + len(chunk), len(wanted))
    _cur(repo, len(wanted), len(wanted))


def _cur(repo, done, total):
    value = {"prs_fetched": done, "total": total, "at_utc": now_utc(),
             "resume": f"python experiments/deep-study/scripts/b1_reverts.py context {repo}"}
    update_status(lambda d: d.setdefault("api", {}).setdefault("cursors", {}).__setitem__(f"revert_ctx_{repo}", value))


def load_ctx(repo: str) -> dict[int, dict]:
    out = {}
    for path in sorted((CACHE / repo / "reverts" / "ctx").glob("batch-*.json")):
        for node in (read_json(path).get("data", {}).get("repository") or {}).values():
            if node:
                out[int(node["number"])] = node
    return out


def clip(s: str, n: int) -> str:
    s = re.sub(r"<!--.*?-->", "", s or "", flags=re.S)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s if len(s) <= n else s[:n] + " …[truncated]"


def is_bot(a) -> bool:
    a = a or {}
    login = (a.get("login") or "").lower()
    return a.get("__typename") == "Bot" or login.endswith("[bot]") or login in {"mergify", "github-actions"}


def dossiers() -> None:
    items = []
    for repo in REPOS:
        ctx = load_ctx(repo)
        for r in read_jsonl(CACHE / repo / "reverts" / "candidates.jsonl"):
            pr = ctx.get(r["revert_pr"] or -1, {})
            comments = [c for c in (pr.get("comments", {}) or {}).get("nodes", []) if not is_bot(c.get("author"))]
            reviews = [c for c in (pr.get("reviews", {}) or {}).get("nodes", []) if (c.get("body") or "").strip() and not is_bot(c.get("author"))]
            targets = []
            for h in r["target_hints"][:4]:
                t = ctx.get(int(h["pr"])) if h.get("pr") else None
                targets.append({
                    "pr": h.get("pr"), "via": h["via"], "reverted_commit_subject": h.get("subject", ""),
                    "reverted_commit_date": h.get("date", ""),
                    "title": (t or {}).get("title", ""), "merged_at": (t or {}).get("mergedAt"),
                    "body": clip((t or {}).get("body", ""), 900),
                    "files": [f["path"] for f in ((t or {}).get("files") or {}).get("nodes", [])][:40],
                    "labels": [x["name"] for x in ((t or {}).get("labels") or {}).get("nodes", [])],
                })
            items.append({
                "id": f"{repo}:{r['revert_sha'][:10]}", "repo": repo, "revert_sha": r["revert_sha"],
                "revert_pr": r["revert_pr"], "commit_date": r["commit_date"], "subject": r["subject"],
                "commit_body": clip(r["body"], 700), "revert_files": r["files"][:40],
                "revert_pr_body": clip(pr.get("body", ""), 1500),
                "revert_pr_discussion": [clip(f"{(c.get('author') or {}).get('login','?')}: {c.get('body','')}", 500) for c in (reviews + comments)[:8]],
                "targets": targets,
            })
    items.sort(key=lambda x: x["id"])
    out_dir = BATCHES / "b1_reverts"
    out_dir.mkdir(parents=True, exist_ok=True)
    size = 45
    for i in range(0, len(items), size):
        write_jsonl(out_dir / f"batch-{i // size:03d}-input.jsonl", items[i:i + size])
    print("revert dossiers", len(items), "batches", (len(items) + size - 1) // size)


def merge() -> None:
    import datetime as dt
    from batches import load_outputs
    from common import atomic_csv, DATA
    coded = load_outputs("b1_reverts")
    rows = []
    for repo in REPOS:
        ctx = load_ctx(repo)
        full = {}
        for line in git(repo, "log", "--first-parent", "--format=%H%x1f%cI%x1f%s", "HEAD", timeout=900).splitlines():
            sha, date, subj = line.split("\x1f", 2)
            n = pr_number(subj)
            if n:
                full.setdefault(n, date)
        for r in read_jsonl(CACHE / repo / "reverts" / "candidates.jsonl"):
            cid = f"{repo}:{r['revert_sha'][:10]}"
            c = coded[cid]
            prs = [int(x) for x in (c.get("reverted_prs") or []) if str(x).isdigit()]
            merged_dates = []
            for n in prs:
                t = ctx.get(n, {}).get("mergedAt") or full.get(n)
                if t:
                    merged_dates.append(iso(t))
            rev_date = iso(r["commit_date"])
            ttr = (rev_date - min(merged_dates)).total_seconds() / 86400 if merged_dates else None
            rows.append({
                "repo": repo, "revert_sha": r["revert_sha"], "revert_pr": r["revert_pr"] or "",
                "revert_date": rev_date.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                "subject": r["subject"], "revert_status": c["revert_status"],
                "is_revert_or_rollback": int(c["revert_status"] in {"confirmed_revert", "partial_revert", "explicit_rollback"}),
                "reverted_prs": ";".join(str(x) for x in prs), "reverted_title": c.get("reverted_title", ""),
                "reverted_merged_at": min(merged_dates).isoformat().replace("+00:00", "Z") if merged_dates else "",
                "days_to_revert": round(ttr, 3) if ttr is not None and ttr >= 0 else "",
                "reverted_class": c["reverted_class"], "reverted_touches_kernel": c["reverted_touches_kernel"],
                "revert_reason": c["revert_reason"], "hardware_specific": c["hardware_specific"],
                "reason_evidence": c.get("reason_evidence", ""), "confidence": c["confidence"],
                "rationale": c.get("rationale", ""),
                "url": f"https://github.com/{REPOS[repo]}/pull/{r['revert_pr']}" if r["revert_pr"] else f"https://github.com/{REPOS[repo]}/commit/{r['revert_sha']}",
            })
    rows.sort(key=lambda x: (x["repo"], x["revert_date"]))
    atomic_csv(DATA / "confirmed-reverts.csv", rows)
    conf = [x for x in rows if x["is_revert_or_rollback"]]
    set_subtask("B", "B1_reverts", status="complete", completed=len(rows), required=len(rows),
                outputs=["experiments/deep-study/data/confirmed-reverts.csv"],
                notes=f"{len(rows)} revert-language candidates adjudicated; {len(conf)} confirmed reverts/partial reverts/explicit rollbacks.",
                next="done")
    print("confirmed-reverts.csv", len(rows), "confirmed", len(conf))


if __name__ == "__main__":
    cmd = sys.argv[1]
    repos = [sys.argv[2]] if len(sys.argv) > 2 else list(REPOS)
    if cmd == "candidates":
        for r in repos:
            candidates(r)
    elif cmd == "context":
        for r in repos:
            context(r)
    elif cmd == "dossiers":
        dossiers()
    elif cmd == "merge":
        merge()
