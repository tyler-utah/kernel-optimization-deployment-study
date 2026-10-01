"""A3 coding: dossiers with keyword-prefiltered snippets, mechanical review metrics,
second-pass selection, merge and agreement.

metrics            - mechanical per-PR metrics from detailed timelines -> data/cache/a3/metrics-<repo>.jsonl
dossiers <task>    - coding batches for every detailed PR not yet coded
pass2              - fresh second-pass batches: 200 random coded PRs per repo
merge              - merge codes + metrics -> data/review-coding.csv ; agreement -> data/review-coding-agreement.csv
"""

from __future__ import annotations

import json
import re
import sys

from a1_population import clean_body
from a3_details import A3, load_details
from common import BATCHES, CUTOFF, DATA, REPOS, atomic_csv, iso, read_json, read_jsonl, rng, write_jsonl

CI_CMD = re.compile(r"^\s*(/[\w-]+|@mergify|@sglang-bot|rerun|re-run|/rerun|run[- ]ci)\b", re.I)
HW = r"amd|rocm|hip\b|mi\d{3}x?|nvidia|hopper|blackwell|ampere|ada\b|sm_?\d{2,3}|h100|h200|h20\b|b200|gb200|b300|a100|a10g?\b|l40s?|l20|4090|5090|xpu|tpu|npu|ascend|cpu|intel|gaudi|hpu|fa2|fa3|fa4|flashinfer|triton|cutlass|trtllm|aiter|backends?|other (?:gpus?|hardware|platforms?)|arch(?:itecture)?s?"
PATTERNS = {
    "ev_microbenchmark": r"micro[- ]?bench|kernel (?:time|latency|perf\w*|bench\w*)|op(?:erator)?[- ]level|op latency|\d\s*(?:us|µs)\b|tflops|tb/s|gb/s|bandwidth|\d+(?:\.\d+)?\s*x\b|speed[- ]?up|bench_\w+|benchmark",
    "ev_end_to_end": r"throughput|tok(?:en)?s?/s|\bttft\b|\btpot\b|\bitl\b|\be2e\b|end[- ]to[- ]end|bench(?:mark)?_serving|bench serve|bench_one_batch|req(?:uest)?s?/s|latency|serving|offline",
    "ev_accuracy_eval": r"gsm8k|mmlu|lm[-_ ]?eval|accuracy|perplexity|\bppl\b|humaneval|gpqa|aime|mgsm|\beval\b|evaluation|score",
    "ev_numerical_tests": r"allclose|assert_close|\batol\b|\brtol\b|tolerance|unit ?tests?|pytest|test_\w+|correctness|numerical|max(?:imum)? ?(?:abs|diff|error)|passed",
    "ev_multi_hardware": HW,
    "ev_memory": r"memory|\bmem\b|\bgi?b\b|\boom\b|kv[- ]?cache (?:size|capacity|usage)|peak",
    "ev_compile_graph": r"cuda[- ]?graphs?|torch\.compile|piecewise|graph capture|dynamo|inductor|compil\w+",
}
REVIEW_PATTERNS = {
    "rq_tests": r"\btests?\b|unit ?test|coverage|\bci\b|pytest",
    "rq_other_backend_hw": HW,
    "rq_accuracy_eval": r"gsm8k|mmlu|lm[-_ ]?eval|accuracy|quality|\beval\w*",
    "rq_perf_evidence": r"bench\w*|perf\w*|speed ?up|throughput|latency|profil\w*|numbers?|results?",
    "cn_integration": r"cuda[- ]?graphs?|torch\.compile|compil\w+|spec(?:ulative)?\b|eagle|mtp\b|\btp\b|\bep\b|\bpp\b|\bdp\b|parallel|backends?|default|config\w*|flags?|env\b|integrat\w*|compat\w*|break\w*|other (?:paths?|features?)|dispatch\w*|fallback|chunked|prefix cach\w*|lora|quantiz\w*",
    "cn_maintenance": r"maintain\w*|complex\w*|duplicat\w*|copy|copied|refactor\w*|abstract\w*|clean\w*|readab\w*|hack\w*|hard to|messy|simplif\w*|unif\w*|reuse|dead code|move (?:this|it)|separate|dependenc\w*|owner\w*|generic|special[- ]case",
    "cn_benchmark_method": r"baseline|compar\w+|warm ?up|variance|noise|repeat\w*|workload|shapes?\b|batch size|seq\w* len\w*|input len|settings?|fair\w*|\be2e\b|end[- ]to[- ]end|real model|representative|reproduc\w*|command|bench\w*",
    "cn_documentation": r"\bdocs?\b|documentation|comments?\b|docstring|readme|explain\w*|\bnote\b|describe",
}
CODES = list(PATTERNS) + list(REVIEW_PATTERNS)


def is_bot(a) -> bool:
    a = a or {}
    login = (a.get("login") or "").lower()
    return a.get("__typename") == "Bot" or login.endswith("[bot]") or login in {"mergify", "github-actions", "copilot"} or login.endswith("-bot")


def events(node: dict) -> list[dict]:
    """Flatten body, reviews, inline comments and issue comments with roles."""
    author = (node.get("author") or {}).get("login")
    out = []
    for r in (node.get("reviews") or {}).get("nodes") or []:
        who = (r.get("author") or {})
        base = {"login": who.get("login"), "bot": is_bot(who), "role": "author" if who.get("login") == author else (r.get("authorAssociation") or "NONE").lower()}
        if (r.get("body") or "").strip():
            out.append(base | {"kind": "review", "at": r.get("submittedAt"), "state": r.get("state"), "text": r["body"]})
        elif r.get("state") in ("APPROVED", "CHANGES_REQUESTED"):
            out.append(base | {"kind": "review", "at": r.get("submittedAt"), "state": r.get("state"), "text": ""})
        for c in (r.get("comments") or {}).get("nodes") or []:
            cw = c.get("author") or {}
            out.append({"login": cw.get("login"), "bot": is_bot(cw), "role": "author" if cw.get("login") == author else base["role"],
                        "kind": "inline", "at": c.get("createdAt"), "path": c.get("path"), "text": c.get("body") or ""})
    for c in (node.get("comments") or {}).get("nodes") or []:
        cw = c.get("author") or {}
        out.append({"login": cw.get("login"), "bot": is_bot(cw), "role": "author" if cw.get("login") == author else (c.get("authorAssociation") or "NONE").lower(),
                    "kind": "comment", "at": c.get("createdAt"), "text": c.get("body") or ""})
    return [e for e in out if e.get("at") and iso(e["at"]) <= CUTOFF]


def human_nonauthor(e) -> bool:
    return not e["bot"] and e["role"] != "author"


def substantive(e) -> bool:
    t = re.sub(r"<!--.*?-->", "", e["text"] or "", flags=re.S).strip()
    if e["kind"] == "review" and e.get("state") == "CHANGES_REQUESTED":
        return True
    if CI_CMD.match(t):
        return False
    return len(t) >= (20 if e["kind"] in ("review", "inline") else 40)


def metrics(repo: str) -> list[dict]:
    rows = []
    for n, node in load_details(repo).items():
        ev = events(node)
        created = iso(node["createdAt"])
        hn = sorted((e for e in ev if human_nonauthor(e) and (substantive(e) or e.get("state") == "APPROVED")), key=lambda e: e["at"])
        sub = [e for e in ev if human_nonauthor(e) and substantive(e)]
        commits = (node.get("commits") or {}).get("nodes") or []
        tl = (node.get("timelineItems") or {}).get("nodes") or []
        rows.append({
            "repo": repo, "number": n,
            "hours_to_first_human_response": round((iso(hn[0]["at"]) - created).total_seconds() / 3600, 3) if hn else None,
            "substantive_review_rounds": len({e["at"][:10] for e in sub}),
            "n_substantive_reviewer_msgs": len(sub),
            "n_human_reviewers": len({e["login"] for e in ev if human_nonauthor(e)}),
            "n_approvals": sum(1 for e in ev if human_nonauthor(e) and e.get("state") == "APPROVED"),
            "n_changes_requested": sum(1 for e in ev if human_nonauthor(e) and e.get("state") == "CHANGES_REQUESTED"),
            "n_commits": (node.get("commits") or {}).get("totalCount"),
            "n_commits_after_first_review": sum(1 for c in commits if hn and c["commit"]["committedDate"] > hn[0]["at"]),
            "n_force_pushes": sum(1 for t in tl if t.get("__typename") == "HeadRefForcePushedEvent"),
            "truncated": int(((node.get("reviews") or {}).get("totalCount") or 0) > 60 or ((node.get("comments") or {}).get("totalCount") or 0) > 100
                             or ((node.get("timelineItems") or {}).get("totalCount") or 0) > 100),
            "bot_messages": sum(1 for e in ev if e["bot"]),
        })
    write_jsonl(A3 / f"metrics-{repo}.jsonl", rows)
    print(repo, "metrics", len(rows))
    return rows


def sentences(text: str) -> list[str]:
    text = re.sub(r"<!--.*?-->", "", text or "", flags=re.S)
    text = re.sub(r"```.*?```", lambda m: m.group(0)[:600], text, flags=re.S)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def snippet_lines(items, pattern, limit=3, maxlen=260):
    rx = re.compile(pattern, re.I)
    out = []
    for src, sents in items:
        for i, s in enumerate(sents):
            if rx.search(s):
                ctx = s if len(s) > 80 or i == 0 else sents[i - 1][-80:] + " " + s
                out.append(f"[{src}] {ctx[:maxlen]}")
                break  # one snippet per message keeps coverage across messages
        if len(out) >= limit:
            break
    return out


def dossier(repo: str, node: dict) -> dict:
    ev = events(node)
    body = clean_body(node.get("body") or "")
    body_s = sentences(body)
    tables = [ln for ln in body.split("\n") if ln.strip().startswith("|") and re.search(r"\d", ln)][:6]
    author_msgs = [(f"author {e['kind']} {e['at'][:10]}", sentences(e["text"])) for e in ev if e["role"] == "author" and not e["bot"] and e["text"]]
    rev_msgs = [(f"{e['kind']} {e['role']} {e['at'][:10]}" + (f" {e.get('path')}" if e.get("path") else ""), sentences(e["text"]))
                for e in ev if human_nonauthor(e) and substantive(e)]
    ev_src = [("body", body_s)] + author_msgs + rev_msgs
    snips = {}
    for code, pat in PATTERNS.items():
        s = snippet_lines(ev_src, pat)
        if code == "ev_multi_hardware":
            hw = sorted({m.lower() for m in re.findall(HW, body + " " + " ".join(e["text"] for e in ev if not e["bot"]), re.I)})
            s = s + ([f"[hardware terms in text] {', '.join(hw[:12])}"] if hw else [])
        if code in ("ev_microbenchmark", "ev_end_to_end") and tables:
            s = s + ["[body table] " + " / ".join(t[:120] for t in tables[:3])]
        if s:
            snips[code] = s
    for code, pat in REVIEW_PATTERNS.items():
        s = snippet_lines(rev_msgs, pat, limit=4)
        if s:
            snips[code] = s
    state = "merged" if node.get("mergedAt") and iso(node["mergedAt"]) <= CUTOFF else ("closed" if node.get("closedAt") and iso(node["closedAt"]) <= CUTOFF else "open")
    last = [f"[{e['role']} {e['at'][:10]}] {e['text'][:220]}" for e in ev if not e["bot"] and e["text"]][-2:] if state == "closed" else []
    files = [f["path"] for f in ((node.get("files") or {}).get("nodes") or [])]
    return {
        "id": f"{repo}#{node['number']}", "repo": repo, "number": node["number"],
        "url": f"https://github.com/{REPOS[repo]}/pull/{node['number']}", "title": node["title"], "state": state,
        "size": f"+{node['additions']}/-{node['deletions']} in {node['changedFiles']} files",
        "labels": [x["name"] for x in ((node.get("labels") or {}).get("nodes") or [])][:10],
        "paths_sample": files[:6], "body_opening": body[:700],
        "n_nonauthor_human_msgs": len(rev_msgs), "candidate_snippets": snips, "last_messages_if_closed": last,
    }


def build_dossiers(task: str, numbers_by_repo: dict[str, list[int]], size: int = 45) -> None:
    items = []
    for repo, nums in numbers_by_repo.items():
        det = load_details(repo)
        for n in nums:
            if n in det:
                items.append(dossier(repo, det[n]))
    items.sort(key=lambda x: x["id"])
    rng("a3-" + task).shuffle(items)
    out = BATCHES / task
    out.mkdir(parents=True, exist_ok=True)
    start = len(list(out.glob("batch-*-input.jsonl")))
    for i in range(0, len(items), size):
        write_jsonl(out / f"batch-{start + i // size:03d}-input.jsonl", items[i:i + size])
    schema = {c: [0, 1] for c in CODES} | {"superseded": ["yes", "no", "unknown", "na"], "evidence": None}
    (out / "schema.json").write_text(json.dumps(schema))
    print(task, "dossiers", len(items), "new batches", (len(items) + size - 1) // size, "starting at", start)


def size_of(r) -> float:
    return float((r.get("additions") or 0) + (r.get("deletions") or 0))


def select() -> None:
    """Performance corpus (all if <=2500/repo else stratified 1000) + matched comparison (>=550/repo)."""
    import collections as C
    from a3_details import add_targets
    from common import CACHE, read_csv
    pop = read_csv(DATA / "performance-pr-population.csv")
    rows = []
    for repo in REPOS:
        meta = {r["number"]: r for r in read_jsonl(CACHE / "a1" / f"population-{repo}.jsonl")}
        perf = [meta[int(p["number"])] for p in pop if p["repo"] == repo and p["classification"] == "confirmed_performance" and p["in_window"] == "1"]
        nonp = [meta[int(p["number"])] for p in pop if p["repo"] == repo and p["classification"] == "not_performance" and p["in_window"] == "1"
                and meta[int(p["number"])]["author_type"] == "human" and meta[int(p["number"])]["additions"] is not None]
        month = lambda r: r["created"][:7]
        if len(perf) <= 2500:
            chosen, design = perf, "all_confirmed_performance"
        else:
            strata = C.defaultdict(list)
            for r in perf:
                strata[(r["state"], month(r))].append(r)
            chosen = []
            for k, v in sorted(strata.items()):
                take = round(1000 * len(v) / len(perf))
                chosen += rng(f"a3-perf-{repo}-{k}").sample(v, min(take, len(v)))
            if len(chosen) < 1000:  # top up rounding shortfall with a seeded draw
                got = {r["number"] for r in chosen}
                rest = sorted((r for r in perf if r["number"] not in got), key=lambda r: r["number"])
                chosen += rng(f"a3-perf-topup-{repo}").sample(rest, 1000 - len(chosen))
            design = "stratified_sample_1000_state_x_month"
        sizes = sorted(size_of(r) for r in chosen)
        cuts = (sizes[len(sizes) // 3], sizes[2 * len(sizes) // 3])
        sbin = lambda r: 0 if size_of(r) <= cuts[0] else (1 if size_of(r) <= cuts[1] else 2)
        target = C.Counter((month(r), sbin(r), r["state"]) for r in chosen)
        pool = C.defaultdict(list)
        for r in nonp:
            pool[(month(r), sbin(r), r["state"])].append(r)
        comp, short = [], 0
        scale = 550 / len(chosen)
        for k, cnt in sorted(target.items()):
            want = max(1, round(cnt * scale))
            cand = pool.get(k, [])
            got = rng(f"a3-comp-{repo}-{k}").sample(cand, min(want, len(cand)))
            comp += got
            short += want - len(got)
        if short > 0:  # fill from same size-bin and state, any month
            used = {r["number"] for r in comp}
            rest = [r for r in nonp if r["number"] not in used]
            by = C.defaultdict(list)
            for r in rest:
                by[(sbin(r), r["state"])].append(r)
            need = C.Counter()
            for k, cnt in target.items():
                need[(k[1], k[2])] += cnt
            tot = sum(need.values())
            for k, cnt in need.items():
                add = round(short * cnt / tot)
                comp += rng(f"a3-fill-{repo}-{k}").sample(by[k], min(add, len(by[k])))
        for r in chosen:
            rows.append({"repo": repo, "number": r["number"], "group": "performance", "design": design, "state": r["state"],
                         "month": month(r), "size_bin": sbin(r), "lines_changed": size_of(r)})
        for r in comp:
            rows.append({"repo": repo, "number": r["number"], "group": "comparison", "design": "matched_month_x_sizebin_x_state",
                         "state": r["state"], "month": month(r), "size_bin": sbin(r), "lines_changed": size_of(r)})
        add_targets(repo, [r["number"] for r in chosen], "performance")
        add_targets(repo, [r["number"] for r in comp], "comparison")
        print(repo, "perf population", len(perf), "corpus", len(chosen), design, "comparison", len(comp), "size cuts", cuts)
    atomic_csv(DATA / "a3-detailed-corpus.csv", rows)


def dossiers_all(task: str = "a3_review") -> None:
    from common import read_csv
    done = set()
    for p in (BATCHES / task).glob("batch-*-input.jsonl"):
        done.update(r["id"] for r in read_jsonl(p))
    want = {}
    for r in read_csv(DATA / "a3-detailed-corpus.csv"):
        if f"{r['repo']}#{r['number']}" not in done:
            want.setdefault(r["repo"], []).append(int(r["number"]))
    build_dossiers(task, want)


def pass2() -> None:
    from batches import load_outputs
    coded = load_outputs("a3_review")
    items = []
    inputs = {}
    for p in sorted((BATCHES / "a3_review").glob("batch-*-input.jsonl")):
        for r in read_jsonl(p):
            inputs[r["id"]] = r
    for repo in REPOS:
        ids = sorted(i for i in coded if i.startswith(repo + "#"))
        items += [inputs[i] for i in rng(f"a3-p2-{repo}").sample(ids, min(200, len(ids)))]
    rng("a3-p2").shuffle(items)
    out = BATCHES / "a3_pass2"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(items), 45):
        write_jsonl(out / f"batch-{i // 45:03d}-input.jsonl", items[i:i + 45])
    (out / "schema.json").write_text((BATCHES / "a3_review" / "schema.json").read_text())
    print("pass2 items", len(items))


def merge() -> None:
    import collections as C
    import numpy as np
    from b_analysis import kappa
    from batches import load_outputs
    from common import read_csv
    corpus = {(r["repo"], int(r["number"])): r for r in read_csv(DATA / "a3-detailed-corpus.csv")}
    codes = load_outputs("a3_review")
    p2 = load_outputs("a3_pass2")
    met = {}
    for repo in REPOS:
        for m in read_jsonl(A3 / f"metrics-{repo}.jsonl"):
            met[(repo, m["number"])] = m
    rows = []
    for (repo, n), c in corpus.items():
        pid = f"{repo}#{n}"
        o = codes.get(pid)
        if not o:
            continue
        m = met.get((repo, n), {})
        ev = o.get("evidence") or {}
        rows.append({"repo": repo, "number": n, "group": c["group"], "design": c["design"], "state": c["state"], "month": c["month"],
                     "size_bin": c["size_bin"], "lines_changed": c["lines_changed"],
                     **{k: int(o[k]) for k in CODES}, "superseded": o["superseded"],
                     **{k: v for k, v in m.items() if k not in ("repo", "number")},
                     "evidence_json": json.dumps(ev, ensure_ascii=False)[:1500],
                     "url": f"https://github.com/{REPOS[repo]}/pull/{n}"})
    atomic_csv(DATA / "review-coding.csv", rows)
    agr = []
    for repo in list(REPOS) + ["both"]:
        ids = [i for i in p2 if i in codes and (repo == "both" or i.startswith(repo + "#"))]
        for k in CODES:
            pairs = [(str(codes[i][k]), str(p2[i][k])) for i in ids]
            if not pairs:
                continue
            kp = kappa(pairs)
            agr.append({"repo": repo, "code": k, "n": len(pairs), "percent_agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4),
                        "cohen_kappa": "" if kp is None else round(kp, 4), "prevalence_pass1": round(sum(a == "1" for a, _ in pairs) / len(pairs), 4),
                        "prevalence_pass2": round(sum(b == "1" for _, b in pairs) / len(pairs), 4),
                        "comparison": "first pass vs fresh second pass on identical dossiers (model-model adjudication consistency; not human IRR)"})
    atomic_csv(DATA / "review-coding-agreement.csv", agr)
    # group proportions with bootstrap CIs
    summ = []
    rs = np.random.default_rng(20260929)
    for repo in REPOS:
        for g in ["performance", "comparison"]:
            sub = [r for r in rows if r["repo"] == repo and r["group"] == g]
            if not sub:
                continue
            for k in CODES + ["reviewed_any"]:
                x = np.array([(r["n_substantive_reviewer_msgs"] or 0) > 0 if k == "reviewed_any" else r[k] for r in sub], float)
                bs = [x[rs.integers(0, len(x), len(x))].mean() for _ in range(1000)]
                summ.append({"repo": repo, "group": g, "measure": k, "n": len(x), "share": round(float(x.mean()), 4),
                             "ci95_low": round(float(np.percentile(bs, 2.5)), 4), "ci95_high": round(float(np.percentile(bs, 97.5)), 4)})
            for k in ["substantive_review_rounds", "hours_to_first_human_response", "n_human_reviewers", "n_commits"]:
                x = np.array([r[k] for r in sub if r.get(k) not in (None, "")], float)
                if len(x):
                    bs = [np.median(x[rs.integers(0, len(x), len(x))]) for _ in range(1000)]
                    summ.append({"repo": repo, "group": g, "measure": f"median_{k}", "n": len(x), "share": round(float(np.median(x)), 3),
                                 "ci95_low": round(float(np.percentile(bs, 2.5)), 3), "ci95_high": round(float(np.percentile(bs, 97.5)), 3)})
    atomic_csv(DATA / "a3-evidence-summary.csv", summ)
    print("review-coding rows", len(rows), "agreement rows", len(agr))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "metrics":
        for r in REPOS:
            metrics(r)
    elif cmd == "select":
        select()
    elif cmd == "dossiers":
        dossiers_all()
    elif cmd == "pass2":
        pass2()
    elif cmd == "merge":
        merge()
