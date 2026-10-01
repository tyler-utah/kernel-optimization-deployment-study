"""Bulk-prefetch blobs for lineage-relevant paths into the study bare clones.

  python blobs.py <repo> [--dry]

Collects every old/new blob OID for files changed under the configured
pathspecs across first-parent history up to the cutoff (git log --raw, which
needs only trees), keeps the OIDs missing locally (GIT_NO_LAZY_FETCH=1), and
fetches them in batches with the same command Git's promisor machinery uses.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
from study_config import CUTOFF_COMMITS  # noqa: E402

GITDIR = {r: os.path.join(S.STUDY, "cache", "git", f"{r}.git") for r in ("sglang", "vllm")}
CUTOFF = CUTOFF_COMMITS

PATHSPECS = {
    "sglang": [
        "python/sglang/srt/layers/moe", "python/sglang/srt/layers/fused_moe.py", "python/sglang/srt/layers/fused_moe",
        "python/sglang/srt/layers/triton_fused_moe", "python/sglang/srt/layers/fused_moe_triton",
        "python/sglang/srt/layers/fused_moe_grok", "python/sglang/srt/layers/fused_moe_patch.py",
        "python/sglang/srt/layers/ep_moe", "python/sglang/srt/layers/attention",
        "python/sglang/srt/layers/radix_attention.py", "python/sglang/srt/models/deepseek_v2.py",
        "python/sglang/srt/models/deepseek_common", "python/sglang/srt/server_args.py",
        "python/sglang/srt/model_executor/model_runner.py", "python/sglang/srt/model_executor/cuda_graph_runner.py",
        "python/pyproject.toml", "python/sglang/jit_kernel", "python/sglang/kernels",
        ":(glob)sgl-kernel/csrc/**", ":(glob)sgl-kernel/python/**", "sgl-kernel/CMakeLists.txt", "sgl-kernel/setup.py",
        "sgl-kernel/pyproject.toml", ":(glob)sgl-kernel/cmake/**", ":(glob)sgl-kernel/tests/**", ":(glob)sgl-kernel/benchmark/**",
        ":(glob)sgl-kernel/src/**", ":(glob)sgl-kernel/include/**", "python/sglang/srt/hardware_backend",
        ":(glob)test/**/*moe*", ":(glob)test/**/*mla*", ":(glob)test/**/*topk*", ":(glob)test/**/*align*",
        ":(glob)benchmark/kernels/**", ":(exclude)**/configs/**",
        # pickaxe integration paths (added for WP-B symbol_pickaxe strategy)
        "python/sglang/srt/models", "python/sglang/srt/layers/quantization", "python/sglang/srt/model_executor",
        "python/sglang/srt/mem_cache/memory_pool.py", "python/sglang/srt/layers/moe",
        "python/sglang/srt/two_batch_overlap.py", "python/sglang/srt/batch_overlap", "python/sglang/srt/eplb",
        ".gitmodules", ":(glob)docker/Dockerfile*", "python/setup.py",
    ],
    "vllm": [
        "csrc/attention", "csrc/rocm", "csrc/cache_kernels.cu", "csrc/cache.h", "csrc/ops.h", "csrc/torch_bindings.cpp",
        "csrc/moe", "csrc/cpu/attention.cpp", "vllm/attention", "vllm/v1/attention",
        "vllm/model_executor/layers/attention", "vllm/model_executor/layers/mla.py", "cmake/external_projects",
        "CMakeLists.txt", "setup.py", ":(glob)requirements*", "requirements", "vllm/platforms", "vllm/vllm_flash_attn",
        "vllm/utils/flashinfer.py", "vllm/_custom_ops.py", "vllm/envs.py",
        "vllm/model_executor/layers/fused_moe/moe_align_block_size.py", "vllm/model_executor/layers/fused_moe/fused_moe.py",
        "vllm/model_executor/models/deepseek_v2.py", "vllm/config/attention.py",
        ":(glob)tests/kernels/attention/**", ":(glob)tests/kernels/test_*attention*", ":(glob)tests/v1/attention/**",
        ":(glob)benchmarks/kernels/*attention*", ":(glob)benchmarks/kernels/*moe_align*",
        # pickaxe integration paths (added for WP-B symbol_pickaxe strategy)
        "vllm/config", "vllm/config.py", "vllm/engine/arg_utils.py", "vllm/worker", "vllm/v1/worker",
        "vllm/model_executor/layers/attention", "vllm/model_executor/models/deepseek_v2.py", ":(glob)cmake/*.cmake",
        ":(glob)docker/Dockerfile*", "Dockerfile", "Dockerfile.rocm", ":(glob)requirements-*.txt",
    ],
}


def git(repo, *args, env=None, input_text=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(["git", f"--git-dir={GITDIR[repo]}", *args], capture_output=True, env=e,
                       input=None if input_text is None else input_text.encode("utf-8"))
    p.stdout = p.stdout.decode("utf-8", "replace").replace("\r\n", "\n")
    p.stderr = p.stderr.decode("utf-8", "replace")
    return p


def collect(repo):
    p = git(repo, "log", "--first-parent", "--raw", "--no-renames", "--no-abbrev", "--format=@@%H",
            CUTOFF[repo], "--", *PATHSPECS[repo])
    if p.returncode:
        raise SystemExit(p.stderr)
    oids = set()
    for line in p.stdout.splitlines():
        if line.startswith(":"):
            parts = line.split()
            for oid in (parts[2], parts[3]):
                if set(oid) != {"0"}:
                    oids.add(oid)
    # also the cutoff-tree blobs under the same paths (for snapshot hashing)
    t = git(repo, "ls-tree", "-r", CUTOFF[repo], "--", *[x for x in PATHSPECS[repo] if not x.startswith(":")])
    for line in t.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "blob":
            oids.add(parts[2])
    return oids


def missing(repo, oids):
    p = git(repo, "cat-file", "--batch-check", env={"GIT_NO_LAZY_FETCH": "1"}, input_text="\n".join(oids) + "\n")
    miss = [line.split()[0] for line in p.stdout.splitlines() if line.endswith(" missing")]
    return miss


def fetch(repo, oids, chunk=4000):
    for i in range(0, len(oids), chunk):
        part = oids[i:i + chunk]
        p = git(repo, "-c", "fetch.negotiationAlgorithm=noop", "fetch", "origin", "--no-tags",
                "--no-write-fetch-head", "--recurse-submodules=no", "--filter=blob:none", "--stdin",
                input_text="\n".join(part) + "\n")
        if p.returncode:
            sys.stderr.write(p.stderr[-2000:])
            raise SystemExit(f"fetch failed at chunk {i}")
        print(f"  fetched {i + len(part)}/{len(oids)}", flush=True)


if __name__ == "__main__":
    repo = sys.argv[1]
    oids = sorted(collect(repo))
    miss = missing(repo, oids)
    print(repo, "blobs referenced:", len(oids), "missing:", len(miss), flush=True)
    if "--dry" not in sys.argv and miss:
        S.log(f"network batch start: blob prefetch {repo} ({len(miss)} blobs)")
        fetch(repo, miss)
        S.log(f"network batch end: blob prefetch {repo}")
        print("still missing:", len(missing(repo, oids)))
