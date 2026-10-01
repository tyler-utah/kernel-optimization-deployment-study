"""WP-I3: exact consistency checks -> data/consistency-check.json

  python consistency.py
Exit code 0 iff no fatal failures.
"""
import csv
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
from study_config import CUTOFF_ISO  # noqa: E402

csv.field_size_limit(10 ** 9)
DATA = os.path.join(S.STUDY, "data")
STAGING = os.path.join(S.STUDY, "staging")
CUT = CUTOFF_ISO
CLAIM = re.compile(r"([-+]?\d[\d,]*\.?\d*)%?\s*<!--\s*claim:([^:]+?)::(.*?)::(.*?)::(\w+)\s*-->")


def rcsv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def utc(s):
    from datetime import datetime, timezone
    s = s.strip()
    if len(s) == 10:
        s += "T00:00:00+00:00"
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt(v, f):
    v = float(v)
    return {"int": lambda x: f"{int(round(x)):,}", "pct1": lambda x: f"{100 * x:.1f}", "dec1": lambda x: f"{x:.1f}",
            "dec2": lambda x: f"{x:.2f}", "dec3": lambda x: f"{x:.3f}"}[f](v)


def main():
    res = {"checked_at": S.now(), "checks": [], "warnings": [], "fatal": 0}

    def check(name, ok, detail="", fatal=True, n=None):
        res["checks"].append({"check": name, "passed": bool(ok), "n": n, "detail": detail if not ok else "", "fatal": fatal})
        if not ok and fatal:
            res["fatal"] += 1
        if not ok and not fatal:
            res["warnings"].append(f"{name}: {detail}")

    cands = rcsv(os.path.join(DATA, "candidate-events.csv"))
    events = rcsv(os.path.join(DATA, "lineage-events.csv"))
    edges = rcsv(os.path.join(DATA, "lineage-edges.csv"))
    moves = rcsv(os.path.join(DATA, "optimization-moves.csv")) if os.path.exists(os.path.join(DATA, "optimization-moves.csv")) else []
    rmap = rcsv(os.path.join(DATA, "release-map.csv"))
    arts = {}
    for L in ("L1", "L2", "L3"):
        for a in json.load(open(os.path.join(STAGING, "registry", f"{L}-registry.json"), encoding="utf-8"))["artifacts"]:
            arts[a["artifact_id"]] = a
    # 1 uniqueness and FKs
    for name, rows, key in (("candidate_id", cands, "candidate_id"), ("event_id", events, "event_id"), ("edge_id", edges, "edge_id")):
        c = Counter(r[key] for r in rows)
        d = [k for k, v in c.items() if v > 1]
        check(f"unique {name}", not d, f"duplicates: {d[:5]}", n=len(rows))
    ev = {e["event_id"]: e for e in events}
    bad = [x["edge_id"] for x in edges if x["from_event_id"] not in ev or x["to_event_id"] not in ev]
    check("edge endpoints exist", not bad, f"{len(bad)} bad: {bad[:5]}", n=len(edges))
    badp = [(e["event_id"], p) for e in events for p in e["parent_event_ids"].split(";") if p and p not in ev]
    check("parent_event_ids exist", not badp, f"{len(badp)}: {badp[:5]}", n=len(events))
    bada = sorted({a for e in events for a in e["artifact_ids"].split(";") if a and a not in arts})
    check("event artifact_ids exist in registry", not bada, f"{len(bada)} unknown: {bada[:10]}", n=len(events))
    mids = {m["move_id"] for m in moves}
    badm = sorted({m for e in events for m in e["optimization_move_ids"].split(";") if m and m not in mids})
    check("event optimization_move_ids exist", not badm or not moves, f"{badm[:10]}", n=len(events))
    cand_ids = {c["candidate_id"] for c in cands}
    badc = [e["event_id"] for e in events if e.get("candidate_id") and e["candidate_id"] not in cand_ids]
    check("event candidate_id exists", not badc, f"{badc[:5]}", n=len(events))
    ver = [c for c in cands if c["adjudication_status"] == "verified"]
    ev_c = {e["candidate_id"] for e in events if e.get("candidate_id")}
    miss = [c["candidate_id"] for c in ver if c["candidate_id"] not in ev_c]
    check("every verified candidate has an event row", not miss, f"{len(miss)}: {miss[:5]}", n=len(ver))
    unscreened = [c["candidate_id"] for c in cands if c["adjudication_status"] in ("unscreened", "awaiting_stage2")]
    check("all candidates screened and adjudicated", not unscreened, f"{len(unscreened)}: {unscreened[:5]}", n=len(cands))
    # 2 chronology
    viol = []
    for x in edges:
        a, b = ev.get(x["from_event_id"]), ev.get(x["to_event_id"])
        if a and b and a["date"] > b["date"] and "[retrospective]" not in x["evidence"]:
            viol.append(x["edge_id"])
    check("edges chronological (except retrospective)", not viol, f"{len(viol)}: {viol[:5]}", n=len(edges))
    # 3 URL correspondence
    badu = []
    for e in events:
        if e["pr_number"] and f"https://github.com/{e['repo']}/pull/{e['pr_number']}" not in e["evidence_urls"]:
            badu.append(e["event_id"])
        if e["commit_sha"] and f"https://github.com/{e['repo']}/commit/{e['commit_sha']}" not in e["evidence_urls"]:
            badu.append(e["event_id"])
    check("event URLs match repo/PR/SHA", not badu, f"{len(badu)}: {badu[:5]}", n=len(events))
    badcu = [c["candidate_id"] for c in cands if c["pr_number"] and f"https://github.com/{c['repo']}/pull/{c['pr_number']}" not in c["evidence_urls"]]
    check("candidate URLs match repo/PR", not badcu, f"{len(badcu)}: {badcu[:5]}", n=len(cands))
    noev = [e["event_id"] for e in events if not e["evidence_urls"].strip() or not e["evidence_excerpt"].strip()]
    check("every event has URL and evidence excerpt", not noev, f"{len(noev)}: {noev[:5]}", n=len(events))
    # 4 releases
    tags = {(r["repo"], r["tag"]): r for r in rmap}
    badr = [e["event_id"] for e in events if e["release"].startswith("v") and (
        "sglang" if "sglang" in e["repo"] else "vllm", e["release"]) not in tags]
    check("event releases exist in release-map", not badr, f"{badr[:5]}", n=len(events))
    late = [r["tag"] for r in rmap if r["release_date"] > CUT]
    check("release dates <= cutoff", not late, f"{late}", n=len(rmap))
    from releases import Releases
    R = Releases()
    fp = {}
    for repo in ("sglang", "vllm"):
        with open(os.path.join(S.STUDY, "cache", "git", f"{repo}-fp.jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                fp[(repo, r["sha"])] = [x[-1] for x in r["files"]] + [x[1] for x in r["files"] if len(x) == 3]
    mism = []
    for e in events:
        if not e.get("candidate_id"):
            continue
        repo = "sglang" if "sglang" in e["repo"] else "vllm"
        rr = R.first_release(repo, e["commit_sha"], e["pr_number"] or None, fp.get((repo, e["commit_sha"]), []))
        exp = rr["tag"] or rr["route"]
        if exp != e["release"]:
            mism.append((e["event_id"], e["release"], exp))
    check("first-containing release recomputes", not mism, f"{len(mism)}: {mism[:5]}", n=len(events))
    # 8 cutoff
    after = [e["event_id"] for e in events if utc(e["date"]) > CUT]
    check("no event after cutoff", not after, f"{after[:5]}", n=len(events))
    afterc = [c["candidate_id"] for c in cands if utc(c["date"]) > CUT]
    check("no candidate after cutoff", not afterc, f"{afterc[:5]}", n=len(cands))
    # 6 biographies reference valid events and moves
    bad_bio = []
    nbio = 0
    for fn in glob.glob(os.path.join(STAGING, "moves", "*-moves-final.json")):
        d = json.load(open(fn, encoding="utf-8"))
        for m in d["moves"]:
            if m.get("biography"):
                nbio += 1
                if m["move_id"] not in mids and moves:
                    bad_bio.append(("move", m["move_id"]))
                for x in m["biography"].get("event_ids", []):
                    if x not in ev:
                        bad_bio.append((m["move_id"], x))
    check("biographies reference valid events and moves", not bad_bio, f"{len(bad_bio)}: {bad_bio[:6]}", n=nbio)
    bad_f = []
    nf = 0
    for fn in glob.glob(os.path.join(STAGING, "failures", "*-failures.json")):
        d = json.load(open(fn, encoding="utf-8"))
        for h in d["histories"]:
            nf += 1
            for t in h.get("timeline", []):
                if t.get("event_id") and t["event_id"] not in ev:
                    bad_f.append((h["history_id"], t["event_id"]))
    check("failure histories reference valid events", not bad_f, f"{len(bad_f)}: {bad_f[:6]}", n=nf)
    # 5 report claim tags
    total = passed = 0
    fails = []
    reports = [os.path.join(S.STUDY, x) for x in ("sglang-moe-lineage.md", "sglang-mla-lineage.md", "vllm-attention-lineage.md", "SYNTHESIS.md")]
    for rp in reports:
        if not os.path.exists(rp):
            check(f"report exists: {os.path.basename(rp)}", False, "missing")
            continue
        text = open(rp, encoding="utf-8").read()
        for m in CLAIM.finditer(text):
            total += 1
            shown, fn, flt, col, f = m.groups()
            try:
                rows = rcsv(os.path.join(S.STUDY, fn))
                conds = [c.split("=", 1) for c in flt.split("&") if c]
                hit = [r for r in rows if all(r.get(k) == v for k, v in conds)]
                if len(hit) != 1:
                    fails.append((os.path.basename(rp), m.group(0)[:120], f"{len(hit)} rows"))
                    continue
                exp = fmt(hit[0][col], f)
                if exp.replace(",", "") != shown.replace(",", ""):
                    fails.append((os.path.basename(rp), m.group(0)[:120], f"expected {exp}"))
                else:
                    passed += 1
            except Exception as ex:  # noqa: BLE001
                fails.append((os.path.basename(rp), m.group(0)[:120], str(ex)[:80]))
        # 7 executive summary: every number in the findings sections must carry a claim tag
        for sec in ("Five strongest defensible findings", "Executive summary"):
            i = text.find(sec)
            if i < 0:
                continue
            j = text.find("\n## ", i + 5)
            body = text[i: j if j > 0 else len(text)]
            body_wo = CLAIM.sub("", body)
            body_wo = re.sub(r"\[[^\]]*\]\([^)]*\)", "", body_wo)          # links
            body_wo = re.sub(r"`[^`]*`", "", body_wo)                      # code
            body_wo = re.sub(r"#\d+|\bv\d[\d.]*\w*|\b(19|20)\d\d(-\d\d){0,2}\b|\bL[123]\b|\bsm\d+\b|\bFA\d\b|\bFP\d\b|\bk=\d\b|\b\d+(st|nd|rd|th)\b|\b[1-9]\.\s", "", body_wo)
            stray = re.findall(r"(?<![\w.])\d+(?:\.\d+)?%?(?![\w])", body_wo)
            check(f"{os.path.basename(rp)}: '{sec}' numbers are claim-tagged", not stray, f"untagged: {stray[:12]}",
                  fatal=True, n=len(stray))
    check("report claim tags reproduce from CSVs", not fails and total > 0, f"{len(fails)} failed: {fails[:6]}", n=total)
    res["claims"] = {"total": total, "passed": passed, "failed": len(fails)}
    res["summary"] = {"candidates": len(cands), "events": len(events), "edges": len(edges), "moves": len(moves)}
    res["passed"] = res["fatal"] == 0
    json.dump(res, open(os.path.join(DATA, "consistency-check.json"), "w", encoding="utf-8"), indent=1)
    for c in res["checks"]:
        print(("PASS " if c["passed"] else ("FAIL " if c["fatal"] else "WARN ")) + c["check"], c["n"] if c["n"] is not None else "", c["detail"][:160])
    print("fatal:", res["fatal"], "warnings:", len(res["warnings"]), "claims:", res["claims"])
    S.update(lambda st: st.__setitem__("last_consistency_check", {"at": res["checked_at"], "passed": res["passed"], "fatal": res["fatal"],
                                                                  "warnings": len(res["warnings"]), "claims": res["claims"]}))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
