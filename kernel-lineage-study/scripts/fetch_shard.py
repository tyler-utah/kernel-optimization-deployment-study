"""Sharded, resumable light-PR fetch for all candidate PRs.

  python fetch_shard.py <shard> <nshards>
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
    shard, n = int(sys.argv[1]), int(sys.argv[2])
    want = defaultdict(set)
    with open(os.path.join(S.STUDY, "staging", "candidates.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["pr_number"] and int(c["pr_number"]) % n == shard:
                want[REPO_KEY[c["repo"]]].add(int(c["pr_number"]))
    for repo, nums in sorted(want.items()):
        todo = [x for x in nums if not os.path.exists(os.path.join(gh.CACHE, "prlight", repo, f"{x}.json"))]
        print(f"shard {shard}/{n} {repo}: {len(nums)} wanted, {len(todo)} to fetch", flush=True)
        if todo:
            gh.prs_light(repo, todo, log_every=20)
    print(f"shard {shard} done", flush=True)
