"""Cached GitHub access via the authenticated `gh` CLI.

Every response is cached under cache/gh/ keyed by a hash of the request, so
reruns never repeat network calls. Rate-limit state and fetched URLs are
recorded in STATUS.json (query_state.github).
"""
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402

CACHE = os.path.join(S.STUDY, "cache", "gh")
os.makedirs(CACHE, exist_ok=True)

REPO_SLUG = {"sglang": "sgl-project/sglang", "vllm": "vllm-project/vllm"}


def _key(kind, payload):
    h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    return os.path.join(CACHE, kind, h[:2], h + ".json")


def _run(args, input_text=None, retries=6):
    delay = 20
    for attempt in range(retries):
        p = subprocess.run(["gh", *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", input=input_text)
        if p.returncode == 0:
            return p.stdout
        err = (p.stderr or "") + (p.stdout or "")[:500]
        if "Not Found" in err or "HTTP 404" in err:
            return None
        if "HTTP 422" in err:
            return json.dumps({"__error__": err[:500]})
        low = err.lower()
        if "rate limit" in low or "secondary" in low or "abuse" in low or "HTTP 502" in err or "HTTP 503" in err or "timeout" in low or "HTTP 500" in err:
            sys.stderr.write(f"[gh] transient error (attempt {attempt+1}): {err[:200]}\n")
            time.sleep(delay)
            delay = min(delay * 2, 600)
            continue
        raise RuntimeError(f"gh {' '.join(args[:3])} failed: {err[:800]}")
    raise RuntimeError(f"gh {' '.join(args[:3])} failed after retries")


def rest(path, paginate=False, cache=True):
    """GET a REST path like 'repos/o/r/releases'. Returns parsed JSON (list merged if paginate)."""
    fn = _key("rest", {"path": path, "paginate": paginate})
    if cache and os.path.exists(fn):
        with open(fn, encoding="utf-8") as f:
            return json.load(f)["data"]
    args = ["api", "-H", "Accept: application/vnd.github+json", path]
    if paginate:
        args.insert(1, "--paginate")
        args.insert(2, "--slurp")
    out = _run(args)
    data = None if out is None else json.loads(out)
    if paginate and isinstance(data, list) and data and isinstance(data[0], list):
        data = [x for page in data for x in page]
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w", encoding="utf-8") as f:
        json.dump({"path": path, "fetched_at": S.now(), "data": data}, f)
    return data


def graphql(query, variables=None, cache=True, pace=0.6):
    payload = {"query": query, "variables": variables or {}}
    fn = _key("gql", payload)
    if cache and os.path.exists(fn):
        with open(fn, encoding="utf-8") as f:
            return json.load(f)["data"]
    t0 = time.time()
    body = json.dumps(payload)
    out = _run(["api", "graphql", "--input", "-"], input_text=body)
    data = json.loads(out)
    if "errors" in data and not data.get("data"):
        raise RuntimeError(f"GraphQL error: {json.dumps(data['errors'])[:800]}")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": S.now(), "data": data}, f)
    time.sleep(pace * (time.time() - t0))
    return data


def rate_limit():
    out = _run(["api", "rate_limit"])
    d = json.loads(out)["resources"]
    return {k: d[k] for k in ("core", "graphql", "search")}


PR_FIELDS = """
      number title url state createdAt mergedAt closedAt isDraft
      body
      baseRefName headRefName
      mergeCommit { oid }
      additions deletions changedFiles
      labels(first: 20) { nodes { name } }
      files(first: 100) { totalCount nodes { path additions deletions changeType } }
      closingIssuesReferences(first: 10) { nodes { number title url } }
      reviews(first: 30) { totalCount nodes { state submittedAt body authorAssociation comments(first: 20) { nodes { body path } } } }
      comments(first: 60) { totalCount nodes { body createdAt authorAssociation author { login } } }
      timelineItems(first: 60, itemTypes: [CROSS_REFERENCED_EVENT, REFERENCED_EVENT, CONNECTED_EVENT]) {
        nodes {
          __typename
          ... on CrossReferencedEvent { createdAt source { __typename ... on PullRequest { number title url mergedAt state } ... on Issue { number title url state } } }
          ... on ReferencedEvent { createdAt commit { oid messageHeadline } }
        }
      }
"""


def prs(repo, numbers, batch=10):
    """Fetch rich PR records for many PR numbers; cached per PR under cache/gh/pr/<repo>/<n>.json."""
    owner, name = REPO_SLUG[repo].split("/")
    pdir = os.path.join(CACHE, "pr", repo)
    os.makedirs(pdir, exist_ok=True)
    result = {}
    todo = []
    for n in sorted(set(int(x) for x in numbers)):
        fn = os.path.join(pdir, f"{n}.json")
        if os.path.exists(fn):
            with open(fn, encoding="utf-8") as f:
                result[n] = json.load(f)
        else:
            todo.append(n)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        parts = [f"p{n}: pullRequest(number: {n}) {{ {PR_FIELDS} }}" for n in chunk]
        q = f"query {{ repository(owner: \"{owner}\", name: \"{name}\") {{ {' '.join(parts)} }} }}"
        try:
            d = graphql(q, cache=False)
        except RuntimeError as e:
            # fall back to one at a time (some numbers are issues, not PRs)
            d = {"data": {"repository": {}}}
            for n in chunk:
                q1 = f"query {{ repository(owner: \"{owner}\", name: \"{name}\") {{ p{n}: pullRequest(number: {n}) {{ {PR_FIELDS} }} }} }}"
                try:
                    d1 = graphql(q1, cache=False)
                    d["data"]["repository"].update(d1["data"]["repository"])
                except RuntimeError as e1:
                    d["data"]["repository"][f"p{n}"] = {"__error__": str(e1)[:300]}
        repo_d = (d.get("data") or {}).get("repository") or {}
        for n in chunk:
            rec = repo_d.get(f"p{n}")
            if rec is None:
                rec = {"__missing__": True, "number": n}
            rec["__fetched_at__"] = S.now()
            with open(os.path.join(pdir, f"{n}.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f)
            result[n] = rec
        sys.stderr.write(f"[gh] {repo} PRs {i+len(chunk)}/{len(todo)} fetched\n")
    return result


LIGHT_FIELDS = """
      number title url state createdAt mergedAt closedAt body
      additions deletions changedFiles
      mergeCommit { oid }
      labels(first: 10) { nodes { name } }
      files(first: 40) { totalCount nodes { path additions deletions changeType } }
      closingIssuesReferences(first: 5) { nodes { number title url } }
"""


def prs_light(repo, numbers, batch=25, log_every=10):
    """Light PR records for screening; cached under cache/gh/prlight/<repo>/<n>.json."""
    owner, name = REPO_SLUG[repo].split("/")
    pdir = os.path.join(CACHE, "prlight", repo)
    os.makedirs(pdir, exist_ok=True)
    result, todo = {}, []
    for n in sorted(set(int(x) for x in numbers)):
        fn = os.path.join(pdir, f"{n}.json")
        if os.path.exists(fn):
            with open(fn, encoding="utf-8") as f:
                result[n] = json.load(f)
        else:
            todo.append(n)
    for bi, i in enumerate(range(0, len(todo), batch)):
        chunk = todo[i:i + batch]
        q = "query { repository(owner: \"%s\", name: \"%s\") { %s } rateLimit { remaining resetAt cost } }" % (
            owner, name, " ".join(f"p{n}: pullRequest(number: {n}) {{ {LIGHT_FIELDS} }}" for n in chunk))
        try:
            d = graphql(q, cache=False, pace=0.3)
            repo_d = (d.get("data") or {}).get("repository") or {}
            rl = (d.get("data") or {}).get("rateLimit") or {}
        except RuntimeError:
            repo_d, rl = {}, {}
            for n in chunk:
                q1 = "query { repository(owner: \"%s\", name: \"%s\") { p%d: pullRequest(number: %d) { %s } } }" % (
                    owner, name, n, n, LIGHT_FIELDS)
                try:
                    repo_d.update(graphql(q1, cache=False)["data"]["repository"])
                except RuntimeError as e1:
                    repo_d[f"p{n}"] = {"__error__": str(e1)[:300]}
        for n in chunk:
            rec = repo_d.get(f"p{n}") or {"__missing__": True, "number": n}
            rec["__fetched_at__"] = S.now()
            with open(os.path.join(pdir, f"{n}.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f)
            result[n] = rec
        if bi % log_every == 0:
            sys.stderr.write(f"[gh] {repo} light PRs {i + len(chunk)}/{len(todo)} rate={rl}\n")
            if rl and rl.get("remaining", 5000) < 300:
                S.update(lambda st: st["rate_limits"].append({"at": S.now(), "api": "graphql", "remaining": rl.get("remaining"), "resetAt": rl.get("resetAt"), "cursor": f"prs_light {repo} index {i}"}), "graphql rate limit low; sleeping until reset")
                import datetime as _dt
                reset = _dt.datetime.fromisoformat(rl["resetAt"].replace("Z", "+00:00"))
                wait = max(30, (reset - _dt.datetime.now(_dt.timezone.utc)).total_seconds() + 30)
                time.sleep(wait)
    return result


if __name__ == "__main__":
    print(json.dumps(rate_limit(), indent=1))
