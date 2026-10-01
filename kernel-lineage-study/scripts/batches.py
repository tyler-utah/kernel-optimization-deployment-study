"""Build screening/coding dossiers and batches; validate batch outputs.

  python batches.py screen-build [--size 180]     -> staging/screen/<L>/batch-NNN-input.{jsonl,md}
  python batches.py code-build   [--size 45]      -> staging/code/<L>/batch-NNN-input.{jsonl,md}
  python batches.py check <screen|code|...>       -> validates *-output.jsonl against inputs
  python batches.py merge-screen                  -> staging/screen-results.jsonl
  python batches.py merge-code                    -> staging/code-results.jsonl
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402

csv.field_size_limit(10 ** 9)
ARTIFACT = os.path.dirname(S.STUDY)
DEEP = os.path.join(ARTIFACT, "deep-study", "data")
STAGING = os.path.join(S.STUDY, "staging")
REPO_KEY = {"sgl-project/sglang": "sglang", "vllm-project/vllm": "vllm"}

SCREEN_LABELS = {"verified_lineage_event", "related_context_only", "name_collision", "insufficient_evidence", "out_of_scope"}
EVENT_TYPES = {"introduce", "optimize", "retune", "port", "integrate", "adapt_framework", "repair_correctness",
               "repair_performance", "repair_build_dependency", "revert", "reland", "deprecate", "replace",
               "change_default", "remove", "extend_support"}
CAUSES = {"specification", "framework_integration", "hardware_compiler", "performance", "correctness",
          "build_dependency", "maintenance"}


def load_candidates():
    with open(os.path.join(STAGING, "candidates.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_registries():
    regs = {}
    for L in ("L1", "L2", "L3"):
        fn = os.path.join(STAGING, "registry", f"{L}-registry.json")
        if os.path.exists(fn):
            d = json.load(open(fn, encoding="utf-8"))
            regs[L] = [(a["artifact_id"], [re.compile(x) for x in a.get("path_regexes", [])]) for a in d["artifacts"]]
    return regs


def pr_light(repo, n):
    fn = os.path.join(S.STUDY, "cache", "gh", "prlight", repo, f"{n}.json")
    if os.path.exists(fn):
        with open(fn, encoding="utf-8") as f:
            return json.load(f)
    return None


def clean_body(body):
    body = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)
    body = re.sub(r"(?m)^\s*-\s*\[[ xX]\].*$", "", body)          # checklists
    body = re.sub(r"(?is)<details>.*?</details>", "[details omitted]", body)
    body = re.sub(r"(?m)^(Signed-off-by|Co-authored-by):.*$", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def perf_lines(body, limit=400):
    out = []
    for line in body.splitlines():
        if re.search(r"(?i)(\d+(\.\d+)?\s*x\b|\d+(\.\d+)?\s*%|speed ?up|tok/s|tokens/s|throughput|latency|\bus\b|\bms\b|TFLOPS|GB/s)", line):
            out.append(line.strip()[:160])
    s = " | ".join(out)
    return s[:limit]


def corpus_flags():
    flags = defaultdict(list)  # (repo, pr) -> [str]

    def rd(fn):
        with open(os.path.join(DEEP, fn), encoding="utf-8") as f:
            return list(csv.DictReader(f))

    repo_of = {"vllm": "vllm", "sglang": "sglang", "vllm-project/vllm": "vllm", "sgl-project/sglang": "sglang"}
    for r in rd("confirmed-reverts.csv"):
        repo = repo_of.get(r["repo"])
        if not repo or r["revert_status"] == "not_revert":
            continue
        m = re.search(r"\d+", r["revert_pr"] or "")
        if m:
            flags[(repo, int(m.group()))].append(f"deep-study revert record: {r['revert_status']} of PR(s) {r['reverted_prs']} reason={r['revert_reason']}")
        for t in re.findall(r"\d+", r["reverted_prs"] or ""):
            flags[(repo, int(t))].append(f"deep-study: this PR was reverted by PR {r['revert_pr']} ({r['revert_status']}, reason={r['revert_reason']})")
    for r in rd("kernel-correctness-cases.csv"):
        repo = repo_of.get(r["repo"])
        if not repo or r["case_status"] != "confirmed_kernel_correctness":
            continue
        m = re.search(r"\d+", r["fix_pr"] or "")
        if m:
            flags[(repo, int(m.group()))].append(f"deep-study correctness case {r['case_id']}: class={r['failure_class']}; symptom={r['symptom'][:120]}; introducing={r['introducing_ref']}")
        m2 = re.search(r"#?(\d{3,6})", r["introducing_ref"] or "")
        if m2:
            flags[(repo, int(m2.group(1)))].append(f"deep-study: introduced the defect fixed in case {r['case_id']} (fix PR {r['fix_pr']})")
    for r in rd("performance-pr-population.csv"):
        if r["classification"] == "confirmed_performance":
            repo = repo_of.get(r["repo"])
            if repo:
                flags[(repo, int(r["number"]))].append(f"deep-study performance PR ({r['perf_type']})")
    return flags


def artifact_hints(regs, L, paths):
    hits = []
    for aid, rxs in regs.get(L, []):
        if any(rx.search(p) for rx in rxs for p in paths):
            hits.append(aid)
    return hits


def dossier(c, regs, flags, mode):
    repo = REPO_KEY[c["repo"]]
    pr = pr_light(repo, c["pr_number"]) if c["pr_number"] else None
    files = []
    if pr and pr.get("files"):
        for f in pr["files"]["nodes"]:
            files.append((f["path"], f.get("additions"), f.get("deletions")))
        more = pr["files"]["totalCount"] - len(files)
    else:
        more = 0
    if not files:
        from paths import load  # noqa
        files = [(p, None, None) for p in c["source_paths"]]
    lineage_paths = set(c["source_paths"])
    files.sort(key=lambda t: (t[0] not in lineage_paths, t[0]))
    cap = 10 if mode == "screen" else 40
    fl = [f"{p} (+{a}/-{d})" if a is not None else p for p, a, d in files[:cap]]
    extra = len(files) - cap + more
    body = clean_body((pr or {}).get("body") or "")
    blen = 450 if mode == "screen" else 2600
    d = {
        "id": c["candidate_id"], "lineage": c["lineage"], "date": c["date"][:10], "sha": c["commit_sha"][:12],
        "pr": c["pr_number"], "title": c["title"], "sources": c["discovery_source"],
        "files": fl, "more_files": max(0, extra), "n_files_commit": c["n_files"],
        "artifact_hints": artifact_hints(regs, c["lineage"], [p for p, _, _ in files] + c["source_paths"]),
        "labels": [x["name"] for x in ((pr or {}).get("labels") or {}).get("nodes", [])],
        "linked_issues": [f"#{x['number']} {x['title']}" for x in ((pr or {}).get("closingIssuesReferences") or {}).get("nodes", [])],
        "body": body[:blen] + (" …[truncated]" if len(body) > blen else ""),
        "perf_lines": perf_lines(body) if mode == "code" else "",
        "deep_study": flags.get((repo, int(c["pr_number"])) if c["pr_number"] else None, []),
        "pr_state": (pr or {}).get("state"), "pr_missing": pr is None or bool((pr or {}).get("__missing__")),
    }
    return d


def render_md(recs, mode):
    out = []
    for d in recs:
        out.append(f"### {d['id']}  ({d['lineage']}, {d['date']}, sha {d['sha']}, PR #{d['pr']})")
        out.append(f"TITLE: {d['title']}")
        out.append(f"SOURCES: {', '.join(d['sources'])}")
        if d.get("stage1"):
            out.append(f"STAGE1: {d['stage1']}")
        out.append(f"ARTIFACT_HINTS: {', '.join(d['artifact_hints']) or '-'}")
        out.append("FILES: " + "; ".join(d["files"]) + (f"; (+{d['more_files']} more)" if d["more_files"] else ""))
        if d["labels"]:
            out.append("LABELS: " + ", ".join(d["labels"]))
        if d["linked_issues"]:
            out.append("ISSUES: " + " | ".join(d["linked_issues"]))
        if d["deep_study"]:
            out.append("DEEP_STUDY: " + " || ".join(d["deep_study"]))
        if d.get("perf_lines"):
            out.append("PERF_LINES: " + d["perf_lines"])
        if d["pr_missing"]:
            out.append("PR_RECORD: missing (use git/gh if needed)")
        body = d["body"].replace("\n", " ⏎ ")
        out.append(f"BODY: {body}")
        out.append("")
    return "\n".join(out)


def build(mode, size, only_ids=None, extra=None, lineages=None):
    cands = load_candidates()
    if lineages:
        cands = [c for c in cands if c["lineage"] in lineages]
    if only_ids is not None:
        cands = [c for c in cands if c["candidate_id"] in only_ids]
    regs = load_registries()
    flags = corpus_flags()
    by_L = defaultdict(list)
    for c in cands:
        by_L[c["lineage"]].append(c)
    base = os.path.join(STAGING, mode)
    total = 0
    for L, cs in sorted(by_L.items()):
        cs.sort(key=lambda c: c["date"])
        d = os.path.join(base, L)
        os.makedirs(d, exist_ok=True)
        for bi, i in enumerate(range(0, len(cs), size)):
            chunk = cs[i:i + size]
            recs = [dossier(c, regs, flags, mode) for c in chunk]
            if extra:
                for r in recs:
                    r.update(extra.get(r["id"], {}))
            stem = os.path.join(d, f"batch-{bi:03d}-input")
            with open(stem + ".jsonl", "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(stem + ".md", "w", encoding="utf-8") as f:
                f.write(render_md(recs, mode))
            total += len(recs)
        print(mode, L, len(cs), "candidates ->", (len(cs) + size - 1) // size, "batches")
    return total


def check(mode, validate_record):
    bad = 0
    summary = defaultdict(lambda: [0, 0, 0])
    for inp in sorted(glob.glob(os.path.join(STAGING, mode, "L*", "batch-*-input.jsonl"))):
        out = inp.replace("-input.jsonl", "-output.jsonl")
        L = os.path.basename(os.path.dirname(inp))
        ids = [json.loads(l)["id"] for l in open(inp, encoding="utf-8")]
        summary[L][0] += 1
        if not os.path.exists(out):
            continue
        recs, errs = {}, []
        for ln, line in enumerate(open(out, encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                errs.append(f"line {ln}: bad json {e}")
                continue
            cid = r.get("candidate_id")
            if cid in recs:
                errs.append(f"dup {cid}")
            recs[cid] = r
            errs += [f"{cid}: {e}" for e in validate_record(r)]
        missing = [i for i in ids if i not in recs]
        extra = [i for i in recs if i not in ids]
        if missing:
            errs.append(f"missing {len(missing)}: {missing[:5]}")
        if extra:
            errs.append(f"extra {len(extra)}: {extra[:5]}")
        if errs:
            bad += 1
            print("INVALID", os.path.relpath(out, S.STUDY), errs[:8])
        else:
            summary[L][1] += 1
            summary[L][2] += len(recs)
    for L, (n, ok, recs) in sorted(summary.items()):
        print(f"{mode} {L}: batches {n}, valid outputs {ok}, records {recs}")
    return bad, summary


def v_screen(r):
    e = []
    if r.get("screening_label") not in SCREEN_LABELS:
        e.append(f"label {r.get('screening_label')}")
    if r.get("screening_label") == "verified_lineage_event":
        if r.get("prelim_event_type") not in EVENT_TYPES:
            e.append(f"event_type {r.get('prelim_event_type')}")
        if not r.get("artifact_ids"):
            e.append("no artifact_ids")
    elif not r.get("rejection_reason"):
        e.append("no rejection_reason")
    return e


def v_code(r):
    e = []
    if r.get("final_label", "verified_lineage_event") not in SCREEN_LABELS:
        e.append("final_label")
    if r.get("final_label", "verified_lineage_event") == "verified_lineage_event":
        if r.get("event_type") not in EVENT_TYPES:
            e.append(f"event_type {r.get('event_type')}")
        if r.get("primary_cause") not in CAUSES:
            e.append(f"primary_cause {r.get('primary_cause')}")
        for k in ("artifact_ids", "evidence_excerpt", "confidence"):
            if not r.get(k):
                e.append(f"missing {k}")
    return e


def merge(mode, out_name):
    rows = []
    for out in sorted(glob.glob(os.path.join(STAGING, mode, "L*", "batch-*-output.jsonl"))):
        for line in open(out, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                r["_batch"] = os.path.relpath(out, STAGING)
                rows.append(r)
    with open(os.path.join(STAGING, out_name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("merged", len(rows), "->", out_name)
    return rows


if __name__ == "__main__":
    cmd = sys.argv[1]
    size = int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else None
    lins = sys.argv[sys.argv.index("--lineages") + 1].split(",") if "--lineages" in sys.argv else None
    if cmd == "screen-build":
        build("screen", size or 180, lineages=lins)
    elif cmd == "code-build":
        res = {}
        for out in sorted(glob.glob(os.path.join(STAGING, "screen", "L*", "batch-*-output.jsonl"))):
            for line in open(out, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    res[r["candidate_id"]] = r
        ids = {k for k, r in res.items() if r["screening_label"] == "verified_lineage_event"}
        extra = {k: {"stage1": f"{res[k]['prelim_event_type']}; artifacts={','.join(res[k]['artifact_ids'])}; {res[k]['rationale']}"} for k in ids}
        n = build("code", size or 45, only_ids=ids, extra=extra, lineages=lins)
        print("verified candidates batched:", n)
    elif cmd == "check":
        mode = sys.argv[2]
        check(mode, v_screen if mode.startswith("screen") else v_code)
    elif cmd == "merge-screen":
        merge("screen", "screen-results.jsonl")
    elif cmd == "merge-code":
        merge("code", "code-results.jsonl")
