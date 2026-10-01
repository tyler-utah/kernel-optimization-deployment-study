"""Extract structured first-parent history (name-status, no renames) and tags
from the blobless clones, bounded by the frozen cutoff commits.

Outputs (cache/git/):
  <repo>-fp.jsonl     one record per first-parent commit:
                      {sha, parents, cdate, adate, subject, body, pr, files:[[status, path(, path2)]]}
  <repo>-tags.json    [{tag, commit, tag_type, tagger_date, commit_date}]

Blobless-safe: --no-renames means no blob contents are needed.
"""
import json
import os
import re
import subprocess
import sys

from study_config import CUTOFF_COMMITS

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ARTIFACT = os.path.dirname(STUDY)
REPOS = {
    "sglang": os.path.join(ARTIFACT, "data", "repos", "sglang"),
    "vllm": os.path.join(ARTIFACT, "data", "repos", "vllm"),
}
CUTOFF_COMMIT = CUTOFF_COMMITS
OUT = os.path.join(STUDY, "cache", "git")

SEP = "\x1e@@@"
FS = "\x1f"
PR_RE = re.compile(r"\(#(\d+)\)\s*$")


def git(repo, *args, text=True):
    return subprocess.run(["git", "-C", REPOS[repo], *args], capture_output=True,
                          check=True, text=text, encoding="utf-8" if text else None,
                          errors="replace" if text else None).stdout


def extract_log(repo):
    fmt = SEP + FS.join(["%H", "%P", "%cI", "%aI", "%s", "%b"]) + FS
    raw = git(repo, "log", "--first-parent", "--no-renames", "--name-status",
              "-z", f"--format={fmt}", CUTOFF_COMMIT[repo])
    recs = []
    for chunk in raw.split(SEP)[1:]:
        parts = chunk.split(FS)
        sha, parents, cdate, adate, subject, body = parts[:6]
        rest = FS.join(parts[6:])
        toks = [t for t in rest.split("\x00")]
        files = []
        i = 0
        toks = [t.strip("\n") if j == 0 else t for j, t in enumerate(toks)]
        while i < len(toks):
            st = toks[i].strip()
            if not st:
                i += 1
                continue
            if st[0] in "RC":
                files.append([st, toks[i + 1], toks[i + 2]])
                i += 3
            else:
                files.append([st, toks[i + 1]])
                i += 2
        m = PR_RE.search(subject)
        recs.append({"sha": sha, "parents": parents.split(), "cdate": cdate, "adate": adate,
                     "subject": subject, "body": body.strip(), "pr": int(m.group(1)) if m else None,
                     "files": files})
    path = os.path.join(OUT, f"{repo}-fp.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(repo, len(recs), "first-parent commits ->", path)


def extract_tags(repo):
    raw = git(repo, "for-each-ref", "refs/tags",
              "--format=%(refname:short)\x1f%(objecttype)\x1f%(*objectname)\x1f%(objectname)\x1f%(taggerdate:iso-strict)\x1f%(*committerdate:iso-strict)\x1f%(committerdate:iso-strict)")
    tags = []
    for line in raw.splitlines():
        name, otype, deref, obj, tdate, dcdate, cdate = line.split("\x1f")
        commit = deref if otype == "tag" else obj
        tags.append({"tag": name, "tag_type": otype, "commit": commit,
                     "tagger_date": tdate or None,
                     "commit_date": dcdate if otype == "tag" else cdate})
    path = os.path.join(OUT, f"{repo}-tags.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tags, f, indent=1)
    print(repo, len(tags), "tags ->", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for repo in (sys.argv[1:] or REPOS):
        extract_log(repo)
        extract_tags(repo)
