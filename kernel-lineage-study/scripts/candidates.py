"""WP-B: multi-strategy, over-inclusive candidate discovery.

  python candidates.py build     -> staging/candidates.jsonl (+ staging/discovery-log.json)

Strategies (a candidate may carry several; recorded in discovery_source):
  path_core                 commit touches a lineage core path (scope.py)
  path_integration+keyword  touches an integration path AND subject matches lineage keywords
  subject_keyword           subject matches lineage keywords (any path)
  body_keyword              commit body matches lineage keywords
  symbol_pickaxe            diff adds/removes a line matching lineage symbols (git log -G) in
                            integration/kernel pathspecs
  dependency_pin            diff of a dependency/build file touches a lineage dependency
  release_notes             a GitHub release-notes line matching lineage keywords cites the PR
  corpus:<file>             deep-study record (provenance / correctness / revert / perf) about lineage
Later strategies (cross_reference, registry_seed, upstream_repo) are appended by other steps.
"""
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
from scope import SCOPES  # noqa: E402
from paths import load  # noqa: E402
from study_config import CUTOFF_COMMITS  # noqa: E402

csv.field_size_limit(10 ** 9)
ARTIFACT = os.path.dirname(S.STUDY)
DEEP = os.path.join(ARTIFACT, "deep-study", "data")
STAGING = os.path.join(S.STUDY, "staging")
GITDIR = {r: os.path.join(S.STUDY, "cache", "git", f"{r}.git") for r in ("sglang", "vllm")}
CUTOFF = CUTOFF_COMMITS
SLUG = {"sglang": "sgl-project/sglang", "vllm": "vllm-project/vllm"}
NON_IMPL = re.compile(r"(^|/)(docs?|test|tests|benchmark|benchmarks|examples)/|\.md$|\.mdx$")

PICKAXE = {
    "L1": (r"moe_align_block_size|biased_grouped_topk|grouped_topk|moe_fused_gate|topk_softmax|topk_sigmoid|"
           r"routed_scaling_factor|moe_sum_reduce|topk_reduce|fused_experts|select_experts|num_fused_shared_experts|"
           r"deepep_mode|moe_runner_backend|moe_a2a_backend|enable_ep_moe|enable_deepep_moe|cutlass_fused_experts|"
           r"triton_kernels|flashinfer_cutlass_moe|enable_flashinfer_moe|fused_moe\(",
           ["python/sglang/srt/server_args.py", "python/sglang/srt/models", "python/sglang/srt/layers/moe",
            "python/sglang/srt/layers/quantization", "python/sglang/srt/model_executor", "sgl-kernel/python",
            "sgl-kernel/csrc", "python/sglang/jit_kernel", "python/sglang/kernels", "python/sglang/srt/layers/fused_moe_triton",
            "python/sglang/srt/layers/fused_moe", "python/sglang/srt/layers/ep_moe", "python/sglang/srt/layers/fused_moe.py"]),
    "L2": (r"flashinfer_mla|enable_flashinfer_mla|flashmla|flash_mla|cutlass_mla|trtllm_mla|BatchMLAPagedAttention|"
           r"w_kc|w_vc|mla_decode|MLATokenToKVPool|disable_mla|enable_mla|use_mla|absorb|kv_lora_rank.*(attention|backend)|"
           r"attention_backend.*(mla|flashinfer|fa3|triton|flashmla|cutlass|trtllm)",
           ["python/sglang/srt/server_args.py", "python/sglang/srt/models/deepseek_v2.py", "python/sglang/srt/models/deepseek_common",
            "python/sglang/srt/layers/attention", "python/sglang/srt/model_executor", "python/sglang/srt/mem_cache/memory_pool.py",
            "sgl-kernel/python", "sgl-kernel/csrc/attention", "python/sglang/kernels", "python/sglang/jit_kernel"]),
    "L3": (r"paged_attention_v1|paged_attention_v2|flash_attn_varlen_func|flash_attn_with_kvcache|vllm_flash_attn|"
           r"fa_version|get_flash_attn_version|BatchDecodeWithPagedKVCacheWrapper|BatchPrefillWithPagedKVCacheWrapper|"
           r"trtllm_batch_decode|trtllm_batch_context|xqa|unified_attention|chunked_prefill_paged_decode|"
           r"context_attention_fwd|decode_attention_fwd|VLLM_ATTENTION_BACKEND|get_attn_backend|AttentionBackendEnum|"
           r"_Backend\.|flash_mla_with_kvcache|cutlass_mla|merge_attn_states|use_cascade_attention",
           ["vllm/attention", "vllm/v1/attention", "vllm/platforms", "vllm/config", "vllm/config.py", "vllm/engine/arg_utils.py",
            "vllm/worker", "vllm/v1/worker", "vllm/envs.py", "vllm/_custom_ops.py", "csrc/attention", "csrc/rocm",
            "csrc/cache_kernels.cu", "CMakeLists.txt", "cmake/external_projects", "vllm/model_executor/layers/attention",
            "vllm/model_executor/layers/mla.py", "vllm/model_executor/models/deepseek_v2.py", "vllm/utils/flashinfer.py"]),
}

DEPS = {
    "sglang": (["python/pyproject.toml", "python/setup.py", "sgl-kernel/CMakeLists.txt", "sgl-kernel/cmake", ".gitmodules",
                "python/sglang/kernels/aot/CMakeLists.txt", "python/sglang/kernels/aot/cmake", "docker"],
               {"L1": r"flashinfer|sgl[-_]kernel|sglang[-_]kernel|deep[-_]?ep|DeepEP|deep[-_]?gemm|DeepGEMM|triton[-_]kernels|cutlass",
                "L2": r"flashinfer|sgl[-_]kernel|sglang[-_]kernel|flash[-_]?mla|FlashMLA|flash[-_]attn|flash-attention|cutlass"}),
    "vllm": (["requirements", "requirements-cuda.txt", "requirements-rocm.txt", "requirements-common.txt", "setup.py",
              "CMakeLists.txt", "cmake/external_projects", "docker", "Dockerfile", "Dockerfile.rocm"],
             {"L3": r"flashinfer|vllm[-_]flash[-_]attn|flash[-_]attn|flash-attention|FlashMLA|flashmla|xformers|aiter|GIT_TAG"}),
}


def git(repo, *args):
    p = subprocess.run(["git", f"--git-dir={GITDIR[repo]}", *args], capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode("utf-8", "replace"))
    return p.stdout.decode("utf-8", "replace").replace("\r\n", "\n")


def pickaxe(repo, regex, pathspecs):
    out = git(repo, "log", "--first-parent", "-G", regex, "--format=%H", CUTOFF[repo], "--", *pathspecs)
    return set(out.split())


def all_paths(r):
    ps = [f[-1] for f in r["files"]]
    ps += [f[1] for f in r["files"] if len(f) == 3]
    return ps


def build():
    cands = {}  # (L, sha) -> rec
    log = {"strategies": {}, "notes": []}

    def add(L, repo, r, src, source_paths=None):
        key = (L, r["sha"])
        c = cands.get(key)
        if c is None:
            c = cands[key] = {"candidate_id": f"{L}-{r['sha'][:10]}", "lineage": L, "repo": SLUG[repo],
                              "date": r["cdate"], "commit_sha": r["sha"], "pr_number": r["pr"] or "",
                              "title": r["subject"], "source_paths": [], "n_files": len(r["files"]),
                              "discovery_source": []}
        if src not in c["discovery_source"]:
            c["discovery_source"].append(src)
        for p in source_paths or []:
            if p not in c["source_paths"]:
                c["source_paths"].append(p)
        log["strategies"].setdefault(f"{L}:{src}", 0)
        log["strategies"][f"{L}:{src}"] += 1

    recs_by_repo = {repo: load(repo) for repo in ("sglang", "vllm")}
    by_sha = {repo: {r["sha"]: r for r in recs} for repo, recs in recs_by_repo.items()}
    by_pr = {repo: {} for repo in recs_by_repo}
    for repo, recs in recs_by_repo.items():
        for r in recs:
            if r["pr"] is not None:
                by_pr[repo].setdefault(r["pr"], r)

    for L, sc in SCOPES.items():
        repo = sc["repo"]
        core = [re.compile(p) for p in sc["core_paths"]]
        integ = [re.compile(p) for p in sc["integ_paths"]]
        cfg = [re.compile(p) for p in sc["config_paths"]]
        kw = re.compile(sc["keywords"])
        for r in recs_by_repo[repo]:
            paths = all_paths(r)
            impl = [p for p in paths if not NON_IMPL.search(p)]
            core_hits = [p for p in impl if any(x.search(p) for x in core)]
            integ_hits = [p for p in impl if any(x.search(p) for x in integ)]
            is_cfg = bool(paths) and bool(cfg) and all(any(x.search(p) for x in cfg) for p in paths)
            subj = bool(kw.search(r["subject"]))
            body = bool(r["body"]) and bool(kw.search(re.sub(r"(?im)^(signed-off-by|co-authored-by):.*$", "", r["body"])))
            if is_cfg:
                add(L, repo, r, "path_config_only", paths[:5])
                continue
            if core_hits:
                add(L, repo, r, "path_core", core_hits)
            if integ_hits and subj:
                add(L, repo, r, "path_integration+keyword", integ_hits)
            if subj:
                add(L, repo, r, "subject_keyword", core_hits or integ_hits)
            if body:
                add(L, repo, r, "body_keyword", core_hits or integ_hits)
        # symbol pickaxe
        rx, specs = PICKAXE[L]
        hits = pickaxe(repo, rx, specs)
        log["notes"].append(f"{L} pickaxe hits: {len(hits)}")
        for sha in hits:
            r = by_sha[repo].get(sha)
            if r:
                add(L, repo, r, "symbol_pickaxe")

    # dependency pins
    for repo, (files, per_lineage) in DEPS.items():
        for L, rx in per_lineage.items():
            hits = pickaxe(repo, rx, files)
            log["notes"].append(f"{L} dependency_pin hits: {len(hits)}")
            for sha in hits:
                r = by_sha[repo].get(sha)
                if r:
                    dep_paths = [p for p in all_paths(r) if any(p.startswith(f) for f in files)]
                    add(L, repo, r, "dependency_pin", dep_paths)

    # release notes
    import gh
    for repo in ("sglang", "vllm"):
        rels = gh.rest(f"repos/{SLUG[repo]}/releases?per_page=100", paginate=True)
        for L, sc in SCOPES.items():
            if sc["repo"] != repo:
                continue
            kw = re.compile(sc["keywords"])
            n = 0
            for rel in rels:
                if not rel.get("published_at") or rel["published_at"] > S.CUTOFF:
                    continue
                for line in (rel.get("body") or "").splitlines():
                    if not kw.search(line):
                        continue
                    for m in re.findall(r"(?:/pull/|#)(\d{2,6})\b", line):
                        r = by_pr[repo].get(int(m))
                        if r:
                            add(L, repo, r, "release_notes")
                            n += 1
            log["notes"].append(f"{L} release_notes PR citations mapped: {n}")

    # deep-study corpora
    def corpus_rows(fn):
        with open(os.path.join(DEEP, fn), encoding="utf-8") as f:
            return list(csv.DictReader(f))

    repo_of = {"vllm": "vllm", "sglang": "sglang", "vllm-project/vllm": "vllm", "sgl-project/sglang": "sglang"}

    def resolve(repo, ref):
        ref = (ref or "").strip()
        m = re.search(r"(?:#|/pull/)(\d+)", ref) or (re.fullmatch(r"\d+", ref) and re.match(r"(\d+)", ref))
        if m:
            return by_pr[repo].get(int(m.group(1)))
        m = re.search(r"\b([0-9a-f]{7,40})\b", ref)
        if m:
            for sha, r in by_sha[repo].items():
                if sha.startswith(m.group(1)):
                    return r
        return None

    for L, sc in SCOPES.items():
        repo = sc["repo"]
        kw = re.compile(sc["keywords"])
        n = defaultdict(int)
        for row in corpus_rows("production-kernel-provenance.csv"):
            if repo_of.get(row["repo"]) != repo:
                continue
            text = " ".join([row["operator_class"], row["family"], row["implementation_name"], row["source_paths"]])
            if kw.search(text) or (L == "L1" and row["operator_class"] == "fused_moe") or (L == "L3" and row["operator_class"] == "attention" and not re.search(r"(?i)mamba|gdn|linear|kda|fla\b", text)) or (L == "L2" and re.search(r"(?i)mla", text)):
                r = resolve(repo, row["first_intro_ref"])
                if r:
                    add(L, repo, r, "corpus:production-kernel-provenance")
                    n["prov"] += 1
        for row in corpus_rows("kernel-correctness-cases.csv"):
            if repo_of.get(row["repo"]) != repo or row["case_status"] != "confirmed_kernel_correctness":
                continue
            text = " ".join([row["subject"], row["affected_kernel"], row["symptom"]])
            if kw.search(text):
                r = resolve(repo, row["fix_pr"]) or resolve(repo, row["fix_sha"])
                if r:
                    add(L, repo, r, "corpus:kernel-correctness-cases")
                    n["corr"] += 1
                ri = resolve(repo, row["introducing_ref"])
                if ri:
                    add(L, repo, ri, "corpus:kernel-correctness-cases(introducing)")
                    n["corr_intro"] += 1
        for row in corpus_rows("confirmed-reverts.csv"):
            if repo_of.get(row["repo"]) != repo or row["revert_status"] == "not_revert":
                continue
            text = " ".join([row["subject"], row["reverted_title"]])
            if kw.search(text):
                r = resolve(repo, row["revert_pr"]) or resolve(repo, row["revert_sha"])
                if r:
                    add(L, repo, r, "corpus:confirmed-reverts")
                    n["rev"] += 1
                for ref in re.findall(r"\d+", row["reverted_prs"] or ""):
                    rr = by_pr[repo].get(int(ref))
                    if rr:
                        add(L, repo, rr, "corpus:confirmed-reverts(reverted)")
                        n["rev_target"] += 1
        for row in corpus_rows("performance-pr-population.csv"):
            if repo_of.get(row["repo"]) != repo or not row["merged"]:
                continue
            if row["classification"] != "confirmed_performance":
                continue
            if kw.search(row["title"]):
                r = by_pr[repo].get(int(row["number"]))
                if r:
                    add(L, repo, r, "corpus:performance-pr-population")
                    n["perf"] += 1
        log["notes"].append(f"{L} corpus mapped: {dict(n)}")

    # PR-body keywords over the first study's cached REST listings (all PRs created
    # 2025-09-29..2026-09-28; bodies of older PRs are not cached -> documented limitation)
    import glob
    STRONG = {
        "L1": r"(?i)moe_align|align_block_size|grouped_topk|biased_grouped_topk|moe_fused_gate|topk_softmax|topk_sigmoid|routed_scaling|topk_reduce|moe_sum_reduce|fused_moe\b|DeepEP|deep_ep|moe_runner|token[_ ]dispatcher|moe_a2a",
        "L2": r"(?i)\bMLA\b|flashinfer_mla|FlashMLA|flash_mla|cutlass_mla|trtllm_mla|cutedsl_mla|latent attention|weight absorption",
        "L3": r"(?i)PagedAttention|paged_attention_v[12]|vllm[-_]flash[-_]attn|FLASHINFER\b|FLASH_ATTN\b|TRITON_ATTN|ROCM_AITER_(FA|MLA|UNIFIED)|unified_attention|trtllm[-_]gen|\bXQA\b|attention backend|FlashAttention ?[234]|\bFA[234]\b|FlashMLA|CUTLASS_MLA|FLASHINFER_MLA",
    }
    for L, sc in SCOPES.items():
        repo = sc["repo"]
        rx = re.compile(STRONG[L])
        n = 0
        for fn in sorted(glob.glob(os.path.join(ARTIFACT, "data", "github", repo, "recent_prs", "page-*.json"))):
            with open(fn, encoding="utf-8") as f:
                page = json.load(f)
            for pr in page:
                if not pr.get("merged_at") or pr["merged_at"] > S.CUTOFF:
                    continue
                body = pr.get("body") or ""
                body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
                if rx.search(body):
                    r = by_pr[repo].get(pr["number"])
                    if r:
                        add(L, repo, r, "body_keyword")
                        n += 1
        log["notes"].append(f"{L} PR-body keyword hits (12-month REST cache): {n}")

    out = os.path.join(STAGING, "candidates.jsonl")
    rows = sorted(cands.values(), key=lambda c: (c["lineage"], c["date"]))
    for c in rows:
        ds = set(c["discovery_source"])
        c["stratum"] = "C" if ds == {"path_config_only"} else ("A" if "path_core" in ds else "B")
    with open(out, "w", encoding="utf-8") as f:
        for c in rows:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(STAGING, "discovery-log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    by = defaultdict(lambda: defaultdict(int))
    for c in rows:
        by[c["lineage"]][c["stratum"]] += 1
    print({k: dict(v) for k, v in by.items()})
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
