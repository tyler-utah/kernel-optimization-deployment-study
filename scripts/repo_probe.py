"""Quick scale probe of candidate open-source repos via the GitHub API (requires an authenticated `gh` CLI).

Usage: python repo_probe.py [days]   (default window: 90 days)

Keyword columns are crude title matches. They are for sizing only, not for results; see ../README.md.
"""
import datetime
import json
import re
import subprocess
import sys
import time

REPOS = [
    "pytorch/pytorch", "vllm-project/vllm", "sgl-project/sglang", "NVIDIA/TensorRT-LLM", "ggml-org/llama.cpp",
    "flashinfer-ai/flashinfer", "NVIDIA/cutlass", "triton-lang/triton", "Dao-AILab/flash-attention",
    "deepseek-ai/DeepGEMM", "ROCm/aiter", "linkedin/Liger-Kernel", "tile-ai/tilelang", "microsoft/onnxruntime",
]


def gh(path):
    return subprocess.run(["gh", "api", "-i", path], capture_output=True, text=True, encoding="utf-8").stdout


def body(raw):
    sep = "\r\n\r\n" if "\r\n\r\n" in raw else "\n\n"
    return json.loads(raw.split(sep, 1)[-1])


def search_count(q):
    time.sleep(2.2)  # search API: 30 req/min
    try:
        return body(gh("search/issues?per_page=1&q=" + q.replace(" ", "+")))["total_count"]
    except Exception:
        return "?"


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    since_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    since, d = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), since_dt.strftime("%Y-%m-%d")
    cols = ["repo", "license", "stars", "open_prs", "open_issues", f"commits_{days}d", "open_prs_kernel_title",
            f"merged_prs_{days}d", f"merged_prs_{days}d_kernel_title", f"issues_{days}d_correctness_kw"]
    print(" | ".join(cols))
    for r in REPOS:
        meta = body(gh(f"repos/{r}"))
        raw = gh(f"repos/{r}/commits?per_page=1&since={since}")
        m = re.search(r'page=(\d+)>; rel="last"', raw)
        commits = int(m.group(1)) if m else len(body(raw))
        row = [
            r, (meta.get("license") or {}).get("spdx_id"), meta.get("stargazers_count"),
            search_count(f"repo:{r} is:pr is:open"),
            search_count(f"repo:{r} is:issue is:open"),
            commits,
            search_count(f"repo:{r} is:pr is:open kernel in:title"),
            search_count(f"repo:{r} is:pr is:merged merged:>={d}"),
            search_count(f"repo:{r} is:pr is:merged merged:>={d} kernel in:title"),
            search_count(f"repo:{r} is:issue created:>={d} accuracy OR wrong OR nan OR incorrect in:title"),
        ]
        print(" | ".join(map(str, row)), flush=True)


if __name__ == "__main__":
    main()
