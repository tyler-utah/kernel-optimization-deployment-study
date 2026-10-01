"""A4: codified vs undocumented reviewer requirements.

dossiers  - per performance PR in the detailed corpus: substantive non-author human
            review messages before merge/close/cutoff (task a4_requests)
induce    - taxonomy-induction sample (task a4_taxonomy_input.jsonl)
docs      - list frozen contribution docs/templates for the documentation check
analyze   - request-type prevalence, >=10-PR undocumented requests, per-PR undocumented counts,
            mechanically observable rule compliance -> data/a4-*.csv
"""

from __future__ import annotations

import collections
import json
import re
import sys

import numpy as np

from a1_population import clean_body
from a3_code import events, human_nonauthor, substantive
from a3_details import load_details
from common import BATCHES, CODING, CUTOFF, DATA, DEEP, REPOS, atomic_csv, iso, read_csv, read_jsonl, rng, write_jsonl

REQ_HINT = re.compile(r"\?|please|could you|can you|would you|should|need|must|let's|lets|pls|suggest|consider|why|instead|nit|what about|how about|add |test|bench|accura|doc", re.I)


def pr_end(node):
    for k in ("mergedAt", "closedAt"):
        if node.get(k) and iso(node[k]) <= CUTOFF:
            return node[k]
    return CUTOFF.isoformat().replace("+00:00", "Z")


def dossier(repo, node):
    end = pr_end(node)
    msgs = []
    for e in events(node):
        if human_nonauthor(e) and substantive(e) and e["at"] <= end:
            t = re.sub(r"<!--.*?-->", "", e["text"] or "", flags=re.S).strip()
            t = re.sub(r"\s+", " ", t)
            msgs.append((0 if REQ_HINT.search(t) else 1, e["at"], f"[{e['kind']} {e['role']} {e['at'][:10]}{(' ' + e['path']) if e.get('path') else ''}] {t[:320]}"))
    msgs.sort(key=lambda x: (x[0], x[1]))
    keep = sorted(msgs[:14], key=lambda x: x[1])
    state = "merged" if node.get("mergedAt") and iso(node["mergedAt"]) <= CUTOFF else ("closed" if node.get("closedAt") and iso(node["closedAt"]) <= CUTOFF else "open")
    return {"id": f"{repo}#{node['number']}", "repo": repo, "number": node["number"], "title": node["title"], "state": state,
            "created": node["createdAt"][:10], "ended": end[:10], "n_substantive_msgs": len(msgs),
            "body_opening": clean_body(node.get("body") or "")[:300], "reviewer_messages": [m[2] for m in keep]}


def build() -> None:
    corpus = [r for r in read_csv(DATA / "a3-detailed-corpus.csv") if r["group"] == "performance"]
    items = []
    for repo in REPOS:
        det = load_details(repo)
        for r in corpus:
            if r["repo"] == repo and int(r["number"]) in det:
                d = dossier(repo, det[int(r["number"])])
                if d["reviewer_messages"]:
                    items.append(d)
    items.sort(key=lambda x: x["id"])
    write_jsonl(DATA / "cache" / "a4" / "all-dossiers.jsonl", items)
    tax = rng("a4-tax").sample(items, min(160, len(items)))
    write_jsonl(DATA / "cache" / "a4" / "taxonomy-sample.jsonl", tax)
    rng("a4-batches").shuffle(items)
    out = BATCHES / "a4_requests"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(items), 60):
        write_jsonl(out / f"batch-{i // 60:03d}-input.jsonl", items[i:i + 60])
    (out / "schema.json").write_text(json.dumps({"request_types": None, "evidence": None, "n_requests": None}))
    print("perf PRs with reviewer messages", len(items), "batches", (len(items) + 59) // 60)


def docs() -> None:
    pats = {
        "vllm": ["CONTRIBUTING.md", ".github/PULL_REQUEST_TEMPLATE.md", ".github/CODEOWNERS", "docs/contributing/**/*.md",
                 "docs/governance/**/*.md", "AGENTS.md", "CLAUDE.md", ".agents/skills/**/SKILL.md"],
        "sglang": [".github/pull_request_template.md", ".github/CODEOWNERS", ".github/MAINTAINER.md", "docs/CONTRIBUTING.md",
                   "docs/AGENTS.md", "AGENTS.md", "CLAUDE.md", "docs/docs/developer_guide/*.mdx",
                   "docs/docs/sglang-diffusion/contributing.mdx", "docs/docs/hardware-platforms/ascend-npus/development/contribution_guide.mdx",
                   ".claude/skills/**/SKILL.md", "sgl-kernel/README.md"],
    }
    out = {}
    for repo, ps in pats.items():
        base = DEEP / "data" / "src" / repo
        files = sorted({str(p.relative_to(base)).replace("\\", "/") for pat in ps for p in base.glob(pat) if p.is_file()})
        out[repo] = files
        print(repo, len(files))
    (DATA / "cache" / "a4").mkdir(parents=True, exist_ok=True)
    (DATA / "cache" / "a4" / "doc-files.json").write_text(json.dumps(out, indent=1))


def merge_docs() -> None:
    from b_analysis import kappa
    p1 = {(r["repo"], r["request_type"]): r for r in read_csv(DATA / "a4-request-documentation.csv")}
    p2 = {(r["repo"], r["request_type"]): r for r in read_csv(DATA / "a4-request-documentation-p2.csv")}
    rows = []
    for k in sorted(p1):
        a, b = p1[k]["documented"], p2.get(k, {}).get("documented", "")
        final = "no" if (a == "no" and b == "no") else ("yes" if (a == "yes" and b == "yes") else "partial")
        src = p1[k] if a != "no" else p2.get(k, p1[k])
        rows.append({"repo": k[0], "request_type": k[1], "documented": final, "pass1": a, "pass2": b,
                     "source": src.get("source", ""), "quote": src.get("quote", ""),
                     "notes": "final=no only when both independent passes found no written rule; yes only when both agree"})
    atomic_csv(DATA / "a4-request-documentation-final.csv", rows)
    pairs = [(r["pass1"], r["pass2"]) for r in rows]
    agr = {"n": len(pairs), "percent_agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4), "cohen_kappa": round(kappa(pairs), 4)}
    atomic_csv(DATA / "a4-doc-check-agreement.csv", [agr | {"comparison": "two independent model passes over frozen docs; not human IRR"}])
    print(agr, collections.Counter((r["repo"], r["documented"]) for r in rows))


def label_agreement() -> dict:
    from b_analysis import kappa
    from batches import load_outputs
    tax = json.loads((CODING / "a4-request-taxonomy.json").read_text(encoding="utf-8"))
    a, b = load_outputs("a4_requests"), load_outputs("a4_requests_p2")
    ids = sorted(b)
    out = []
    jac = []
    for i in ids:
        s1, s2 = set(a[i]["request_types"]), set(b[i]["request_types"])
        jac.append(1.0 if not (s1 | s2) else len(s1 & s2) / len(s1 | s2))
    out.append({"request_type": "ALL (mean Jaccard of type sets)", "n": len(ids), "percent_agreement": round(float(np.mean(jac)), 4), "cohen_kappa": "",
                "prevalence_pass1": "", "prevalence_pass2": ""})
    per = {}
    for t in tax["types"]:
        name = t["name"]
        pairs = [(str(name in a[i]["request_types"]), str(name in b[i]["request_types"])) for i in ids]
        k = kappa(pairs)
        per[name] = k
        out.append({"request_type": name, "n": len(ids), "percent_agreement": round(sum(x == y for x, y in pairs) / len(pairs), 4),
                    "cohen_kappa": "" if k is None else round(k, 4), "prevalence_pass1": round(sum(x == "True" for x, _ in pairs) / len(pairs), 4),
                    "prevalence_pass2": round(sum(y == "True" for _, y in pairs) / len(pairs), 4)})
    atomic_csv(DATA / "a4-label-agreement.csv", out)
    print(out[0])
    return per


def analyze() -> None:
    from batches import load_outputs
    import numpy as _np  # noqa: F401
    per_k = label_agreement() if (BATCHES / "a4_requests_p2").exists() and load_outputs("a4_requests_p2") else {}
    p2 = load_outputs("a4_requests_p2") if per_k else {}
    tax = json.loads((CODING / "a4-request-taxonomy.json").read_text(encoding="utf-8"))
    docmap = {(r["repo"], r["request_type"]): r for r in read_csv(DATA / "a4-request-documentation-final.csv")}
    labels = load_outputs("a4_requests")
    items = {r["id"]: r for r in read_jsonl(DATA / "cache" / "a4" / "all-dossiers.jsonl")}
    corpus = {(r["repo"], int(r["number"])): r for r in read_csv(DATA / "a3-detailed-corpus.csv") if r["group"] == "performance"}
    # prevalence per repo and request type
    prev = []
    per_pr = []
    for repo in REPOS:
        n_corpus = sum(1 for k in corpus if k[0] == repo)
        cnt = collections.Counter()
        ex = collections.defaultdict(list)
        exfallback = collections.defaultdict(list)
        for pid, lab in labels.items():
            if not pid.startswith(repo + "#"):
                continue
            types = sorted(set(lab["request_types"]))
            for t in types:
                cnt[t] += 1
                ev = (lab.get("evidence") or {}).get(t, "")
                agreed = pid in p2 and t in p2[pid]["request_types"]
                if ev and (agreed or not p2) and len(ex[t]) < 3:
                    ex[t].append(f"{items[pid]['title'][:60]} — {ev[:150]} (https://github.com/{REPOS[repo]}/pull/{items[pid]['number']})")
                elif ev and len(exfallback[t]) < 3:
                    exfallback[t].append(f"{items[pid]['title'][:60]} — {ev[:150]} (https://github.com/{REPOS[repo]}/pull/{items[pid]['number']}) [single-coder]")
            undoc = [t for t in types if docmap.get((repo, t), {}).get("documented") == "no"]
            part = [t for t in types if docmap.get((repo, t), {}).get("documented") == "partial"]
            per_pr.append({"repo": repo, "id": pid, "state": items[pid]["state"], "n_request_types": len(types),
                           "n_undocumented": len(undoc), "n_partially_documented": len(part), "undocumented_types": ";".join(undoc)})
        for t in tax["types"]:
            name = t["name"]
            d = docmap.get((repo, name), {})
            prev.append({"repo": repo, "request_type": name, "group": t.get("group", ""), "distinct_perf_prs": cnt[name],
                         "denominator_perf_prs_in_corpus": n_corpus, "share": round(cnt[name] / n_corpus, 4) if n_corpus else "",
                         "documented": d.get("documented", ""), "doc_source": d.get("source", ""), "doc_quote": d.get("quote", ""),
                         "recurring_ge10": int(cnt[name] >= 10), "undocumented_recurring": int(cnt[name] >= 10 and d.get("documented") == "no"),
                         "label_kappa": "" if per_k.get(name) is None else round(per_k[name], 4),
                         "examples": " || ".join(ex[name] or exfallback[name])})
    atomic_csv(DATA / "a4-request-types.csv", prev)
    atomic_csv(DATA / "a4-per-pr-undocumented.csv", per_pr)
    summ = []
    for repo in REPOS:
        n_corpus = sum(1 for k in corpus if k[0] == repo)
        merged = [k for k, v in corpus.items() if k[0] == repo and v["state"] == "merged"]
        pp = {r["id"]: r for r in per_pr if r["repo"] == repo}
        m_with = sum(1 for k in merged if pp.get(f"{repo}#{k[1]}", {}).get("n_undocumented", 0) > 0)
        m_with_x = sum(1 for k in merged if [t for t in pp.get(f"{repo}#{k[1]}", {}).get("undocumented_types", "").split(";")
                                              if t and t != "fix_functional_correctness_issue"])
        tot_undoc = sum(r["n_undocumented"] for r in pp.values() if r["state"] == "merged")
        tot_req = sum(r["n_request_types"] for r in pp.values() if r["state"] == "merged")
        summ.append({"repo": repo, "perf_prs_in_corpus": n_corpus, "merged_perf_prs": len(merged),
                     "merged_with_ge1_undocumented_request_before_merge": m_with,
                     "share_merged_with_ge1_undocumented": round(m_with / len(merged), 4) if merged else "",
                     "merged_with_ge1_undocumented_excl_defect_fix": m_with_x,
                     "share_merged_with_ge1_undocumented_excl_defect_fix": round(m_with_x / len(merged), 4) if merged else "",
                     "merged_with_ge1_request_before_merge": sum(1 for k in merged if pp.get(f"{repo}#{k[1]}", {}).get("n_request_types", 0) > 0),
                     "request_instances_before_merge": tot_req, "undocumented_request_instances_before_merge": tot_undoc,
                     "share_request_instances_undocumented": round(tot_undoc / tot_req, 4) if tot_req else "",
                     "undocumented_recurring_types": sum(1 for p in prev if p["repo"] == repo and p["undocumented_recurring"])})
    atomic_csv(DATA / "a4-summary.csv", summ)
    for s in summ:
        print(s)


def section_filled(body: str, heading: str) -> bool:
    b = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)
    m = re.search(rf"(?im)^#+\s*{heading}[^\n]*\n(.*?)(?=^#+\s|\Z)", b, re.S | re.M)
    if not m:
        return False
    txt = re.sub(r"(?m)^\s*[-*]\s*\[[ xX]\].*$", "", m.group(1))
    return len(re.sub(r"\s+", "", txt)) >= 20


def compliance() -> None:
    from batches import load_outputs
    codes = load_outputs("a3_review")
    corpus = read_csv(DATA / "a3-detailed-corpus.csv")
    WRITE = {"member", "owner", "collaborator"}
    rows = []
    for repo in REPOS:
        det = load_details(repo)
        for c in corpus:
            if c["repo"] != repo:
                continue
            n = int(c["number"])
            node = det[n]
            merged_at = node.get("mergedAt") if node.get("mergedAt") and iso(node["mergedAt"]) <= CUTOFF else None
            files = [f["path"] for f in ((node.get("files") or {}).get("nodes") or [])]
            labels_before = {}
            for t in ((node.get("timelineItems") or {}).get("nodes") or []):
                if t.get("__typename") == "LabeledEvent":
                    labels_before.setdefault((t.get("label") or {}).get("name", ""), t["createdAt"])
            ev = events(node)
            appr = [e for e in ev if human_nonauthor(e) and e.get("state") == "APPROVED" and (not merged_at or e["at"] <= merged_at)]
            body = node.get("body") or ""
            code = codes.get(f"{repo}#{n}", {})
            kernel = any(re.search(r"\.(cu|cuh)$|(^|/)csrc/|sgl-kernel/|/kernels?/|jit_kernel|fused_moe|triton", p) for p in files)
            aot = [p for p in files if re.search(r"^sgl-kernel/|python/sglang/kernels/aot/", p)]
            r = {"repo": repo, "number": n, "group": c["group"], "state": c["state"], "merged": int(bool(merged_at)),
                 "title_bracket_tag": int(bool(re.match(r"^\s*\[[^\]]+\]", node["title"]))),
                 "approval_by_write_role_before_merge": int(any(e["role"] in WRITE for e in appr)) if merged_at else "",
                 "any_nonauthor_approval_before_merge": int(bool(appr)) if merged_at else "",
                 "tests_changed": int(any(re.search(r"(^|/)tests?/|(^|/)test_", p) for p in files)),
                 "kernel_paths_changed": int(kernel),
                 "kernel_change_with_numerical_test_evidence": (int(code.get("ev_numerical_tests", 0) == 1) if kernel and code else ""),
                 "perf_evidence_reported": (int(bool(code.get("ev_microbenchmark") or code.get("ev_end_to_end"))) if code else "")}
            if repo == "vllm":
                r["template_purpose_filled"] = int(section_filled(body, "Purpose"))
                r["template_test_plan_filled"] = int(section_filled(body, "Test Plan"))
                r["template_test_result_filled"] = int(section_filled(body, "Test Result"))
                r["ready_label_before_merge"] = int("ready" in labels_before and (labels_before["ready"] <= merged_at)) if merged_at else ""
            else:
                r["template_accuracy_tests_filled"] = int(section_filled(body, "Accuracy Tests"))
                r["template_benchmarking_filled"] = int(section_filled(body, "(?:Speed Tests and Profiling|Benchmarking and Profiling|Benchmarking)"))
                r["run_ci_label_before_merge"] = int("run-ci" in labels_before and labels_before["run-ci"] <= merged_at) if merged_at else ""
                r["aot_kernel_change"] = int(bool(aot))
                r["aot_change_with_tests_and_benchmark"] = (int(any("/tests/" in p or "/test_" in p for p in aot) and any("bench" in p for p in aot)) if aot else "")
            rows.append(r)
    atomic_csv(DATA / "a4-rule-compliance-per-pr.csv", rows)
    measures = {
        "vllm": [("vllm-title-001", "title_bracket_tag", "all"), ("vllm-prdesc-001", "template_purpose_filled", "all"),
                 ("vllm-prdesc-001", "template_test_plan_filled", "all"), ("vllm-prdesc-001", "template_test_result_filled", "all"),
                 ("vllm-review-001 (proxy)", "approval_by_write_role_before_merge", "merged"), ("vllm-ci-002", "ready_label_before_merge", "merged"),
                 ("vllm-tests-001 (proxy)", "tests_changed", "all"), ("vllm-kernel-003 (proxy)", "kernel_change_with_numerical_test_evidence", "kernel")],
        "sglang": [("sglang-accuracy-001 (template)", "template_accuracy_tests_filled", "all"), ("sglang-perf-001 (template)", "template_benchmarking_filled", "all"),
                   ("sglang-perf-001 (coded evidence)", "perf_evidence_reported", "all"), ("sglang-ci-001", "run_ci_label_before_merge", "merged"),
                   ("sglang-review-001 (proxy)", "approval_by_write_role_before_merge", "merged"), ("sglang-tests-001 (proxy)", "tests_changed", "all"),
                   ("sglang-kernel-001", "aot_change_with_tests_and_benchmark", "aot")],
    }
    out = []
    for repo, ms in measures.items():
        for rule, col, scope in ms:
            for g in ["performance", "comparison"]:
                vals = [r[col] for r in rows if r["repo"] == repo and r["group"] == g and r.get(col) not in ("", None)]
                if not vals:
                    continue
                out.append({"repo": repo, "rule_id": rule, "signal": col, "scope": scope, "group": g, "n": len(vals),
                            "compliant": sum(vals), "share": round(sum(vals) / len(vals), 4)})
    atomic_csv(DATA / "a4-rule-compliance.csv", out)
    for o in out:
        print(o)


if __name__ == "__main__":
    {"build": build, "docs": docs, "merge_docs": merge_docs, "analyze": analyze, "compliance": compliance}[sys.argv[1]]()
