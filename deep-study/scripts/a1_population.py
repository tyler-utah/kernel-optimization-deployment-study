"""A1: complete, high-precision performance-PR population.

build       - join compact REST metadata + GraphQL enrichment + local squash paths;
              compute signals and tiers -> data/cache/a1/population-<repo>.jsonl
candidates  - write adjudication batches (every ambiguous candidate, every first-study
              open-queue candidate, and validation samples of both rule tiers)
merge       - merge adjudications -> data/performance-pr-population.csv
"""

from __future__ import annotations

import collections
import re
import sys

from a1_metadata import load_enriched, meta_path
from common import (A_START, BATCHES, CACHE, CUTOFF, DATA, FIRST, EXP, REPOS, atomic_csv, iso,
                    read_csv, read_jsonl, rng, write_jsonl)

TEST_DOC_RE = re.compile(
    r"(^|/)(tests?|testing|benchmarks?|docs?|examples?|scripts?|tools?)(/|$)|"
    r"(^|/)(test_|conftest)|(^|/)\.github/|\.md$|\.rst$|(^|/)\.buildkite/|"
    r"(^|/)(bench|benchmark)_[^/]*\.py$|(^|/)(dockerfile[^/]*|requirements[^/]*\.txt|"
    r"pyproject\.toml|setup\.py|cmakelists\.txt|package\.json|uv\.lock)$", re.I)
KERNEL_PATH_RE = re.compile(
    r"\.(cu|cuh|hip)$|(^|/)csrc/|^sgl-kernel/|/jit_kernel/|fused_moe|/layers/moe/|"
    r"/attention/(ops|backends)/|/v1/attention/|/layers/attention/|(^|/)kernels?/|/ops/|"
    r"triton_ops|triton_kernel|_triton\.py$|/device_communicators/|/compilation/.*fus|"
    r"/quantization/(kernels|utils)/|/layers/(layernorm|activation|rotary_embedding|sampler)|"
    r"_custom_ops\.py$|_aiter_ops\.py$|/sample/ops/|/lora/(ops|triton_ops)/|/mamba/ops/|/fla/ops/",
    re.I)
PERF_TAGS = {"perf", "performance", "perf.", "optimization", "optimize", "optim", "speedup",
             "perf improvement", "performance improvement", "opt", "kernel perf", "perf/kernel"}
FIX_TAGS = {"bugfix", "bug", "fix", "bug fix", "hotfix", "bugfixes", "fixes"}
NONPERF_TAGS = {"doc", "docs", "documentation", "ci", "ci/build", "test", "tests", "build", "docker",
                "release", "chore", "deps", "dependencies", "ci fix", "revert", "wip", "do not merge",
                "benchmark", "benchmarks", "bench", "misc", "refactor", "cleanup", "ux", "security"}
KERNEL_TAGS = {"kernel", "kernels", "sgl-kernel", "jit kernel", "jit-kernel", "cuda", "triton", "cutlass",
               "helion", "cuda kernel", "sgl_kernel", "jit_kernel"}
TITLE_OPT_RE = re.compile(
    r"\b(optimi[sz](?:e|es|ed|ing|ation|ations)|speed(?:s|ed)?[- ]?up|speedups?|faster|accelerat\w+|"
    r"fus(?:e|ed|es|ing|ion)\b|reduc\w* .{0,40}(?:overhead|latency|memory|time|launch|sync|copies|copy|"
    r"allocation|bubble|cpu|kernel|host)|improv\w* .{0,40}(?:performance|perf|throughput|latency|speed|"
    r"efficien|ttft|tpot|itl)|(?:boost|increase)\w* .{0,30}(?:throughput|performance|speed)|"
    r"tun(?:e|ed|ing)\b|autotun\w*|vectori[sz]\w*|\d+(?:\.\d+)?\s*[x×]\s*(?:faster|speed)|"
    r"\d+(?:\.\d+)?\s*%\s*(?:faster|speed|improv|throughput|latency|lower|higher|reduc|e2e|perf)|"
    r"perf(?:ormance)? regression|low(?:er)?[- ]latency|high(?:er)?[- ]throughput|overlap\w*|"
    r"avoid\w* .{0,30}(?:sync|copy|copies|recompil|allocation|overhead|d2h|h2d)|zero[- ]copy|"
    r"persistent kernel|mega[- ]?kernel|efficient|efficiency|cheaper|fast path|hot path)", re.I)
BODY_OPT_RE = re.compile(
    r"speed[- ]?up|faster|throughput|latency|\bttft\b|\btpot\b|\bitl\b|tok(?:en)?s?/s|"
    r"\d+(?:\.\d+)?\s*[x×]\b|\d+(?:\.\d+)?\s*%\s*(?:improv|faster|speed|reduc|lower|higher|gain)|"
    r"optimi[sz]", re.I)
REVERT_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)*revert\b", re.I)


def tags_of(title: str) -> set[str]:
    tags = {t.strip().lower() for t in re.findall(r"\[([^\]]{1,40})\]", title[:160])}
    m = re.match(r"^\s*([a-zA-Z][\w-]{0,15})(?:\(([^)]*)\))?!?:", title)
    if m:
        tags.add(m.group(1).lower())
        if m.group(2):
            tags.add(m.group(2).lower())
    return tags


def clean_body(body: str) -> str:
    b = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)
    b = re.sub(r"<details>.*?(?:PR Checklist|Essential Elements|BEFORE SUBMITTING).*?</details>", "", b, flags=re.S | re.I)
    b = re.split(r"\n#+\s*(?:Checklist|Review Process|CI States)\b", b, flags=re.I)[0]
    b = re.sub(r"(?m)^\s*[-*]\s*\[[ xX]\].*$", "", b)
    b = re.sub(r"\n{3,}", "\n\n", b).strip()
    return b


def local_paths(repo: str) -> dict[int, list[str]]:
    out, cur = {}, None
    with (EXP / "data" / f"{repo}-log.txt").open(encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith("@@@"):
                subj = line[3:].split("|", 3)[3]
                m = re.search(r"\(#(\d+)\)\s*$", subj)
                cur = out.setdefault(int(m.group(1)), []) if m else None
            elif line and cur is not None:
                cur.append(line.replace("\\", "/"))
    return out


def state_at_cutoff(merged, closed) -> str:
    if merged and iso(merged) <= CUTOFF:
        return "merged"
    if closed and iso(closed) <= CUTOFF:
        return "closed"
    return "open"


def first_study_queue(repo: str) -> set[int]:
    rows = read_csv(FIRST / f"pr-lifecycle-{repo}.csv")
    return {int(r["number"]) for r in rows if r["state_at_cutoff"] == "open" and r["category"] == "kernel_performance"}


def build(repo: str) -> list[dict]:
    meta = read_jsonl(meta_path(repo))
    enr = load_enriched(repo)
    lp = local_paths(repo)
    queue = first_study_queue(repo)
    out = []
    for m in meta:
        n = m["number"]
        in_window = iso(m["created_at"]) >= A_START
        if not in_window and n not in queue and "open_prs" not in m["source"]:
            continue
        node = enr.get(n) or {}
        files = [f["path"] for f in ((node.get("files") or {}).get("nodes") or [])]
        total_files = (node.get("files") or {}).get("totalCount")
        if not files and n in lp:
            files = lp[n]
        prod = [p for p in files if not TEST_DOC_RE.search(p)]
        kpaths = [p for p in prod if KERNEL_PATH_RE.search(p)]
        tags = tags_of(m["title"])
        labels = [x["name"] for x in ((node.get("labels") or {}).get("nodes") or [])] or m["labels"]
        body = clean_body(m["body"])
        author = node.get("author") or {}
        login = (author.get("login") or m.get("user") or "")
        is_bot = author.get("__typename") == "Bot" or m.get("user_type") == "Bot" or login.lower().endswith("[bot]")
        merged = node.get("mergedAt", m.get("merged_at"))
        closed = node.get("closedAt", m.get("closed_at"))
        sig = {
            "perf_tag": bool(tags & PERF_TAGS),
            "fix_tag": bool(tags & FIX_TAGS),
            "nonperf_tag": bool(tags & NONPERF_TAGS),
            "kernel_tag": bool(tags & KERNEL_TAGS),
            "perf_label": "performance" in [x.lower() for x in labels],
            "title_opt": bool(TITLE_OPT_RE.search(m["title"].replace("_", " "))),
            "revert_title": bool(REVERT_RE.search(m["title"])),
            "kernel_path": bool(kpaths),
            "body_opt": bool(BODY_OPT_RE.search(body)),
        }
        if sig["perf_tag"] and not (sig["fix_tag"] or sig["nonperf_tag"] or sig["revert_title"]):
            tier = "rule_confirmed"
        elif not sig["revert_title"] and (
            sig["perf_tag"] or sig["title_opt"] or sig["kernel_tag"] or sig["perf_label"]
            or (sig["kernel_path"] and sig["body_opt"] and not sig["fix_tag"])
        ):
            tier = "ambiguous"
        else:
            tier = "rule_excluded"
        out.append({
            "repo": repo, "number": n, "title": m["title"], "created": m["created_at"],
            "merged": merged if merged and iso(merged) <= CUTOFF else "",
            "closed": closed if closed and iso(closed) <= CUTOFF else "",
            "state": state_at_cutoff(merged, closed), "in_window": int(in_window),
            "first_study_open_queue": int(n in queue), "author_type": "bot" if is_bot else "human",
            "labels": labels, "additions": node.get("additions"), "deletions": node.get("deletions"),
            "changed_files": node.get("changedFiles", len(files) if files else None),
            "files_total": total_files, "paths": files[:100], "kernel_paths": kpaths[:30],
            "enriched": int(bool(node)), "tier": tier, **{f"sig_{k}": int(v) for k, v in sig.items()},
            "body_clean": body[:6000],
        })
    write_jsonl(CACHE / "a1" / f"population-{repo}.jsonl", out)
    c = collections.Counter((r["tier"], r["in_window"]) for r in out)
    print(repo, len(out), dict(c), "enriched", sum(r["enriched"] for r in out),
          "queue", sum(r["first_study_open_queue"] for r in out))
    return out


def dossier(r: dict) -> dict:
    paths = r["paths"]
    return {
        "id": f"{r['repo']}#{r['number']}", "repo": r["repo"], "number": r["number"],
        "url": f"https://github.com/{REPOS[r['repo']]}/pull/{r['number']}",
        "title": r["title"], "state_at_cutoff": r["state"], "labels": r["labels"],
        "size": f"+{r['additions']}/-{r['deletions']} in {r['changed_files']} files",
        "kernel_paths": r["kernel_paths"][:8],
        "other_paths": [p for p in paths if p not in r["kernel_paths"]][:6],
        "body": r["body_clean"][:900] + (" …[truncated]" if len(r["body_clean"]) > 900 else ""),
    }


def candidates() -> None:
    items, manifest = [], collections.Counter()
    for repo in REPOS:
        pop = read_jsonl(CACHE / "a1" / f"population-{repo}.jsonl")
        chosen = {}
        for r in pop:
            if r["tier"] == "ambiguous" and r["in_window"]:
                chosen[r["number"]] = "ambiguous"
            if r["first_study_open_queue"]:
                chosen.setdefault(r["number"], "first_study_open_queue")
        conf = [r for r in pop if r["tier"] == "rule_confirmed" and r["in_window"] and r["number"] not in chosen]
        excl = [r for r in pop if r["tier"] == "rule_excluded" and r["in_window"] and r["number"] not in chosen]
        for r in rng(f"a1-conf-{repo}").sample(conf, min(150, len(conf))):
            chosen[r["number"]] = "validation_rule_confirmed"
        for r in rng(f"a1-excl-{repo}").sample(excl, min(200, len(excl))):
            chosen[r["number"]] = "validation_rule_excluded"
        byn = {r["number"]: r for r in pop}
        for n, why in chosen.items():
            d = dossier(byn[n])
            d["_selection"] = why
            items.append(d)
            manifest[(repo, why)] += 1
    rng("a1-shuffle").shuffle(items)
    out = BATCHES / "a1_adjudication"
    out.mkdir(parents=True, exist_ok=True)
    size = 100
    for i in range(0, len(items), size):
        # selection reason is kept out of the coder's view
        write_jsonl(out / f"batch-{i // size:03d}-input.jsonl", [{k: v for k, v in d.items() if k != "_selection"} for d in items[i:i + size]])
    write_jsonl(CACHE / "a1" / "adjudication-selection.jsonl", [{"id": d["id"], "selection": d["_selection"]} for d in items])
    print("adjudication items", len(items), "batches", (len(items) + size - 1) // size)
    for k, v in sorted(manifest.items()):
        print(" ", k, v)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - r) / d, (c + r) / d)


def merge() -> None:
    from batches import load_outputs
    adj = load_outputs("a1_adjudication")
    sel = {r["id"]: r["selection"] for r in read_jsonl(CACHE / "a1" / "adjudication-selection.jsonl")}
    rows, val = [], []
    for repo in REPOS:
        pop = read_jsonl(CACHE / "a1" / f"population-{repo}.jsonl")
        tv = collections.defaultdict(lambda: collections.Counter())
        for r in pop:
            pid = f"{repo}#{r['number']}"
            if pid in adj:
                a = adj[pid]
                cls, ev, rat, conf, ptype = a["classification"], a["evidence"], a["rationale"], a["confidence"], a["perf_type"]
                src = f"adjudicated ({sel.get(pid, '?')})"
                if sel.get(pid, "").startswith("validation"):
                    tv[sel[pid]][cls] += 1
            elif r["tier"] == "rule_confirmed":
                tags = ", ".join(sorted(tags_of(r["title"]) & PERF_TAGS))
                cls, ev, conf, ptype = "confirmed_performance", f"title tag: {tags}", "high", ""
                rat = "Explicit performance title tag without bug-fix/CI/doc/revert tag; tier precision estimated from an adjudicated random sample."
                src = "rule_confirmed"
            elif r["tier"] == "rule_excluded":
                cls, ev, conf, ptype = "not_performance", "no perf tag/label/title claim; no kernel-path+body perf claim", "medium", ""
                rat = "No performance signal in title, tags, labels, or kernel paths with a body performance claim; tier miss rate estimated from an adjudicated random sample."
                src = "rule_excluded"
            else:
                cls, ev, conf, ptype = "uncertain", "ambiguous candidate not adjudicated", "low", ""
                rat = "Ambiguous candidate outside the adjudication set (out-of-window, not in first-study queue)."
                src = "unadjudicated_ambiguous"
            rows.append({"repo": repo, "number": r["number"], "state": r["state"], "title": r["title"], "created": r["created"],
                         "merged": r["merged"], "closed": r["closed"], "classification": cls, "evidence": ev, "rationale": rat,
                         "confidence": conf, "perf_type": ptype, "tier": r["tier"], "classification_source": src,
                         "in_window": r["in_window"], "first_study_open_queue": r["first_study_open_queue"],
                         "url": f"https://github.com/{REPOS[repo]}/pull/{r['number']}"})
        for s, c in tv.items():
            n = sum(c.values())
            k = c["confirmed_performance"]
            lo, hi = wilson(k, n)
            val.append({"repo": repo, "validation_sample": s, "n": n, "confirmed_performance": k,
                        "not_performance": c["not_performance"], "uncertain": c["uncertain"],
                        "share_confirmed": round(k / n, 4), "wilson95_low": round(lo, 4), "wilson95_high": round(hi, 4)})
    fields = ["repo", "number", "state", "title", "created", "merged", "closed", "classification", "evidence", "rationale",
              "confidence", "perf_type", "tier", "classification_source", "in_window", "first_study_open_queue", "url"]
    atomic_csv(DATA / "performance-pr-population.csv", rows, fields)
    atomic_csv(DATA / "a1-tier-validation.csv", val)
    summ = []
    for repo in REPOS:
        rr = [r for r in rows if r["repo"] == repo]
        win = [r for r in rr if r["in_window"]]
        q = [r for r in rr if r["first_study_open_queue"]]
        c = collections.Counter(r["classification"] for r in win)
        cq = collections.Counter(r["classification"] for r in q)
        oq = collections.Counter(r["classification"] for r in rr if r["state"] == "open")
        summ.append({"repo": repo, "window_prs": len(win), "confirmed_performance": c["confirmed_performance"],
                     "not_performance": c["not_performance"], "uncertain": c["uncertain"],
                     "first_study_queue_candidates": len(q), "queue_confirmed": cq["confirmed_performance"],
                     "queue_not_performance": cq["not_performance"], "queue_uncertain": cq["uncertain"],
                     "open_at_cutoff_all": sum(1 for r in rr if r["state"] == "open"),
                     "open_confirmed_performance": oq["confirmed_performance"],
                     "adjudicated_records": sum(1 for r in rr if r["classification_source"].startswith("adjudicated"))})
    atomic_csv(DATA / "a1-population-summary.csv", summ)
    for s in summ:
        print(s)
    for v in val:
        print(v)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        for r in (sys.argv[2:] or list(REPOS)):
            build(r)
    elif cmd == "candidates":
        candidates()
    elif cmd == "merge":
        merge()
