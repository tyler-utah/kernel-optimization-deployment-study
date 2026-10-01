"""WP-F helpers: first-containing public release for commits.

SGLang kernels under sgl-kernel/** (and later python/sglang/kernels/aot/**) ship in
a separately versioned wheel (sgl-kernel, renamed sglang-kernel). A change in those
paths reaches users only when (1) a wheel version built at/after the change exists
and (2) an SGLang release pins a version >= that wheel version. Everything else
(including JIT kernels in python/sglang/jit_kernel and python/sglang/kernels/jit)
ships with the SGLang Python package itself. vLLM compiles csrc/ into its own wheel.

  python releases.py wheels        -> cache/git/sglang-kernel-wheels.json (version timeline + release pins)
  python releases.py test <repo> <sha> [pr]
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
from study_config import CUTOFF_COMMITS  # noqa: E402

GITDIR = {r: os.path.join(S.STUDY, "cache", "git", f"{r}.git") for r in ("sglang", "vllm")}
CUTOFF = CUTOFF_COMMITS
VERSION_FILES = ["sgl-kernel/python/sgl_kernel/version.py", "sgl-kernel/pyproject.toml", "sgl-kernel/setup.py",
                 "python/sglang/kernels/aot/python/sgl_kernel/version.py", "python/sglang/kernels/aot/pyproject.toml",
                 "sgl-kernel/version.py", "sgl-kernel/src/sgl-kernel/version.py"]
WHEEL_PATH = re.compile(r"^(sgl-kernel/|python/sglang/kernels/aot/)")


def git(repo, *args, ok=False):
    p = subprocess.run(["git", f"--git-dir={GITDIR[repo]}", *args], capture_output=True)
    if p.returncode and not ok:
        raise RuntimeError(p.stderr.decode("utf-8", "replace"))
    return p.stdout.decode("utf-8", "replace").replace("\r\n", "\n")


def vtuple(v):
    parts = re.findall(r"\d+|post\d+|rc\d+", v)
    out = []
    for p in parts:
        if p.startswith("post"):
            out.append(("post", int(p[4:])))
        elif p.startswith("rc"):
            out.append(("rc", int(p[2:])))
        else:
            out.append(("n", int(p)))
    # rc < release < post
    key = []
    for kind, n in out:
        key.append({"rc": (0, n), "n": (1, n), "post": (2, n)}[kind] if kind != "n" else (1, n))
    return tuple(key)


def fp_index(repo):
    with open(os.path.join(S.STUDY, "cache", "git", f"{repo}-fp.jsonl"), encoding="utf-8") as f:
        recs = [json.loads(l) for l in f]
    return recs, {r["sha"]: i for i, r in enumerate(recs)}


def read_version(sha, path):
    t = git("sglang", "show", f"{sha}:{path}", ok=True)
    m = re.search(r"""(?m)^\s*(?:__version__|version)\s*=\s*["']([0-9][^"']*)["']""", t)
    return m.group(1) if m else None


def wheels():
    recs, idx = fp_index("sglang")
    out = git("sglang", "log", "--first-parent", "--format=%H", CUTOFF["sglang"], "--", *VERSION_FILES).split()
    timeline = []
    last = None
    for sha in reversed(out):  # oldest first
        v = None
        for p in VERSION_FILES:
            v = read_version(sha, p) or v
            if v:
                break
        if v and v != last:
            timeline.append({"sha": sha, "fp_index": idx.get(sha), "date": recs[idx[sha]]["cdate"] if sha in idx else "",
                             "version": v, "pr": recs[idx[sha]]["pr"] if sha in idx else None})
            last = v
    rel = json.load(open(os.path.join(S.STUDY, "cache", "git", "sglang-release-index.json"), encoding="utf-8"))
    pins = []
    for r in rel:
        t = git("sglang", "show", f"{r['commit']}:python/pyproject.toml", ok=True)
        m = re.search(r"""["'](?:sgl-kernel|sgl_kernel|sglang-kernel)\s*(==|>=|~=)\s*([0-9][^"',;\s\]]*)""", t)
        pins.append({"tag": r["tag"], "final": r["final"], "release_date": r["release_date"],
                     "pin_op": m.group(1) if m else None, "pin": m.group(2) if m else None})
    d = {"timeline": timeline, "release_pins": pins}
    with open(os.path.join(S.STUDY, "cache", "git", "sglang-kernel-wheels.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    print(len(timeline), "wheel versions;", sum(1 for p in pins if p["pin"]), "releases with pins")
    return d


class Releases:
    def __init__(self):
        self.rel = {r: json.load(open(os.path.join(S.STUDY, "cache", "git", f"{r}-release-index.json"), encoding="utf-8"))
                    for r in ("sglang", "vllm")}
        self.fp = {r: fp_index(r) for r in ("sglang", "vllm")}
        wf = os.path.join(S.STUDY, "cache", "git", "sglang-kernel-wheels.json")
        self.wheels = json.load(open(wf, encoding="utf-8")) if os.path.exists(wf) else None
        self.finals = {r: [x for x in self.rel[r] if x["final"]] for r in self.rel}

    def code_release(self, repo, sha, pr):
        recs, idx = self.fp[repo]
        i = idx.get(sha)
        best = None
        for r in self.finals[repo]:
            route = None
            bp = r["effective_branch_point_fp_index"]
            if i is not None and bp is not None and i >= bp:
                route = "branch_point"
            elif pr and int(pr) in r["cherry_pick_prs"]:
                route = "cherry_pick"
            if route and (best is None or r["release_date"] < best["release_date"]):
                best = {"tag": r["tag"], "route": route + ("_tag_moved" if r["tag_moved_after_publication"] else ""),
                        "release_date": r["release_date"]}
        return best

    def wheel_version(self, sha):
        recs, idx = self.fp["sglang"]
        i = idx.get(sha)
        if i is None or not self.wheels:
            return None
        cands = [w for w in self.wheels["timeline"] if w["fp_index"] is not None and w["fp_index"] <= i]
        if not cands:
            return None
        w = max(cands, key=lambda w: w["fp_index"])  # oldest bump at/after the change
        return w

    def wheel_release(self, sha):
        w = self.wheel_version(sha)
        if not w:
            return None, None
        vt = vtuple(w["version"])
        best = None
        for p in self.wheels["release_pins"]:
            if not p["final"] or not p["pin"]:
                continue
            if vtuple(p["pin"]) >= vt and (best is None or p["release_date"] < best["release_date"]):
                best = {"tag": p["tag"], "route": f"kernel_wheel>={w['version']}", "release_date": p["release_date"]}
        return w, best

    def first_release(self, repo, sha, pr, paths):
        """Return dict(tag, route, release_date, wheel_version, code_tag) or status string."""
        code = self.code_release(repo, sha, pr)
        res = {"code_tag": code["tag"] if code else "", "wheel_version": "", "tag": "", "route": "", "release_date": ""}
        chosen = code
        if repo == "sglang" and any(WHEEL_PATH.search(p) for p in paths):
            w, wr = self.wheel_release(sha)
            res["wheel_version"] = w["version"] if w else "unreleased_wheel"
            if w is None or wr is None:
                chosen = None if w is None else (None if wr is None else wr)
                if chosen is None:
                    res["route"] = "wheel_not_pinned_by_cutoff" if w else "wheel_not_built_by_cutoff"
            elif code is None or wr["release_date"] > code["release_date"]:
                chosen = wr
        if chosen:
            res.update({"tag": chosen["tag"], "route": chosen["route"], "release_date": chosen["release_date"]})
        elif not res["route"]:
            res["route"] = "unreleased_at_cutoff"
        return res


if __name__ == "__main__":
    if sys.argv[1] == "wheels":
        d = wheels()
        for w in d["timeline"][:3] + d["timeline"][-3:]:
            print(w["date"][:10], w["version"], w["pr"])
        print([ (p["tag"], p["pin"]) for p in d["release_pins"] if p["final"]][-8:])
    elif sys.argv[1] == "test":
        R = Releases()
        print(R.first_release(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None, sys.argv[5:] or []))
