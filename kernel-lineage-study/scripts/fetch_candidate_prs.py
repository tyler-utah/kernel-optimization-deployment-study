"""Fetch light PR records for every candidate PR (cached; resumable).

  python fetch_candidate_prs.py
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
import gh  # noqa: E402

REPO_KEY = {"sgl-project/sglang": "sglang", "vllm-project/vllm": "vllm"}

if __name__ == "__main__":
    want = defaultdict(set)
    with open(os.path.join(S.STUDY, "staging", "candidates.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["pr_number"]:
                want[REPO_KEY[c["repo"]]].add(int(c["pr_number"]))
    for repo, nums in want.items():
        S.log(f"network batch start: light PR records {repo} ({len(nums)} PRs)")
        S.update(lambda st: st["query_state"]["github"].__setitem__(f"prs_light_{repo}", {"total": len(nums), "cache": f"cache/gh/prlight/{repo}/<n>.json", "resume": "python scripts/fetch_candidate_prs.py"}))
        res = gh.prs_light(repo, nums)
        missing = sum(1 for v in res.values() if v.get("__missing__") or v.get("__error__"))
        S.log(f"network batch end: light PR records {repo}: {len(res)} cached, {missing} missing/error")
        print(repo, len(res), "cached;", missing, "missing/error")
