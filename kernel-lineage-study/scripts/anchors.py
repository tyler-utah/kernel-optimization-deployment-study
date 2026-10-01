"""WP-B upstream_repo strategy: hand-verified cross-repository anchor events.

  python anchors.py     (idempotent)

Adds anchor candidates (discovery_source = upstream_repo) to staging/candidates.jsonl, writes
staging/code/<L>/batch-901-{input,output}.jsonl with coded records verified by reading the PR
bodies and diffs (evidence excerpts are verbatim), and appends relations to existing events via
staging/code-overrides.jsonl (merging with the first-pass relation list).
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import status as S  # noqa: E402
from batches import dossier, load_registries, corpus_flags, render_md  # noqa: E402

STAGING = os.path.join(S.STUDY, "staging")

ANCHORS = {
    "L1": [
        {"sha": "5d60def02cb5a43fa5864fcb123909b101df9ec5", "pr": 2453, "repo": "vllm-project/vllm",
         "code": {"event_type": "introduce", "primary_cause": "performance",
                  "artifact_ids": ["L1.upstream.vllm.align_sum"], "spec_change": "none",
                  "integration_change": "vLLM fused MoE layer calls a C++ moe_align_block_size op before the Triton fused_moe kernel",
                  "default_change": "none", "hardware_scope": ["all_cuda"],
                  "performance_claim": "align_block_size moved to C++ for ~10% improvement; deepseek-moe-16b 16587.82 tok/s vs llama2-7b 10978.67 tok/s (PR body)",
                  "correctness_evidence": "adds tests/kernels/test_fused_moe.py",
                  "moves": [{"move_id": "M-L1-align-block-padding", "name": "", "category": "tiling_blocking",
                             "mechanism": "per-expert token counts padded to BLOCK_SIZE multiples and sorted ids emitted for the Triton fused MoE kernel",
                             "source": "diff", "status": "new"},
                            {"move_id": "M-L1-fused-moe-triton-tiled-pipeline", "name": "", "category": "tiling_blocking",
                             "mechanism": "Triton fused MoE GEMM over expert-sorted, block-padded token tiles", "source": "diff", "status": "new"}],
                  "relations": [], "assumptions": [],
                  "evidence_excerpt": "Implement the `align_block_size` function in C++ to achieve a 10% performance improvement.",
                  "evidence_types": ["pr_body", "diff", "upstream_source"], "confidence": "high",
                  "notes": "upstream anchor: origin of the alignment contract later adopted by SGLang"}},
        {"sha": "95460fc51318702a33226b87152afe810187e01e", "pr": 12574, "repo": "vllm-project/vllm",
         "code": {"event_type": "port", "primary_cause": "performance", "artifact_ids": ["L1.upstream.vllm.align_sum"],
                  "spec_change": "none", "integration_change": "adds sgl_moe_align_block_size and a Triton moe_align_block_size variant to vLLM",
                  "default_change": "none", "hardware_scope": ["all_cuda"], "performance_claim": "none",
                  "correctness_evidence": "none",
                  "moves": [{"move_id": "M-L1-align-block-padding", "name": "", "category": "tiling_blocking",
                             "mechanism": "ports SGLang's alignment kernels (CUDA and Triton) that produce block-padded expert-sorted ids",
                             "source": "pr_body", "status": "reused"}],
                  "relations": [{"target": "PR#2735", "repo": "sgl-project/sglang", "relation": "ported_from",
                                 "evidence": "sgl_moe_align_block_size is based on: sglang commit ded9fcd09a43 (#2735)"},
                                {"target": "PR#2712", "repo": "sgl-project/sglang", "relation": "ported_from",
                                 "evidence": "moe_align_block_size is based on: sglang commit ba5112ff691d (#2712)"}],
                  "assumptions": [],
                  "evidence_excerpt": "sgl_moe_align_block_size is based on: https://github.com/sgl-project/sglang/commit/ded9fcd09a43d5e7d5bb31a2bc3e9fc21bf65d2a",
                  "evidence_types": ["pr_body", "diff", "upstream_source"], "confidence": "high",
                  "notes": "downstream anchor: vLLM ports SGLang's alignment kernels back"}},
        {"sha": "ffb2cd6b5441af28da22694a19838a19507a15e1", "pr": 19572, "repo": "vllm-project/vllm",
         "code": {"event_type": "port", "primary_cause": "performance", "artifact_ids": ["L1.upstream.vllm.align_sum"],
                  "spec_change": "none",
                  "integration_change": "replaces vLLM's align kernel with SGLang's implementation; marks moe_align_block_size_triton and sgl_moe_align_block_size for deprecation",
                  "default_change": "sgl_moe_align_block_size / Triton variant -> optimized moe_align_block_size for vLLM fused MoE",
                  "hardware_scope": ["all_cuda"], "performance_claim": "kernel speedup reported in PR benchmark tables (benchmark_moe_align_block_size.py)",
                  "correctness_evidence": "tests/kernels/moe/test_moe_align_block_size.py updated",
                  "moves": [{"move_id": "M-L1-align-two-phase-shared-cumsum", "name": "", "category": "multi_block",
                             "mechanism": "SGLang's two-phase alignment (shared-memory cumsum, then parallel scatter) taken verbatim",
                             "source": "pr_body", "status": "reused"}],
                  "relations": [{"target": "PR#6369", "repo": "sgl-project/sglang", "relation": "ported_from",
                                 "evidence": "The implementation is taken from sgl-project/sglang blob 8b5f83ed3b7d (#6369)"},
                                {"target": "PR#12574", "repo": "vllm-project/vllm", "relation": "replaces",
                                 "evidence": "Mark `moe_align_block_size_triton` and `sgl_moe_align_block_size` for deprecation"}],
                  "assumptions": [],
                  "evidence_excerpt": "The implementation is taken from https://github.com/sgl-project/sglang/blob/8b5f83ed3b7d2a49ad5c5cd5aa61c5d502f47dbc Specially thanks to SGL developers!",
                  "evidence_types": ["pr_body", "diff", "upstream_source"], "confidence": "high",
                  "notes": "downstream anchor: second port of SGLang's alignment kernel into vLLM"}},
    ],
}
# relations to append to existing first-pass events: candidate_id -> relation
APPEND = {
    "L1-09de730dee": {"target": "PR#2453", "repo": "vllm-project/vllm", "relation": "ported_from",
                      "evidence": "SGLang fused_moe.py header 'Adapted from' vLLM fused_moe.py; it called vLLM's moe_align_block_size op introduced in vLLM #2453 (file evolved in between)"},
}


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    cfile = os.path.join(STAGING, "candidates.jsonl")
    cands = jl(cfile)
    ids = {c["candidate_id"]: c for c in cands}
    fp = {}
    for r in jl(os.path.join(S.STUDY, "cache", "git", "vllm-fp.jsonl")):
        fp[r["sha"]] = r
    for r in jl(os.path.join(S.STUDY, "cache", "git", "sglang-fp.jsonl")):
        fp[r["sha"]] = r
    regs, flags = load_registries(), corpus_flags()
    for L, items in ANCHORS.items():
        inputs, outputs = [], []
        for a in items:
            r = fp[a["sha"]]
            cid = f"{L}-{a['sha'][:10]}"
            if cid not in ids:
                c = {"candidate_id": cid, "lineage": L, "repo": a["repo"], "date": r["cdate"], "commit_sha": a["sha"],
                     "pr_number": a["pr"], "title": r["subject"], "source_paths": [f[-1] for f in r["files"]][:12],
                     "n_files": len(r["files"]), "discovery_source": ["upstream_repo"], "stratum": "U"}
                cands.append(c)
                ids[cid] = c
            elif "upstream_repo" not in ids[cid]["discovery_source"]:
                ids[cid]["discovery_source"].append("upstream_repo")
            d = dossier(ids[cid], regs, flags, "code")
            d["stage1"] = "UPSTREAM ANCHOR (hand-verified from PR body and diff)"
            inputs.append(d)
            outputs.append({"candidate_id": cid, "final_label": "verified_lineage_event", "rejection_reason": "", **a["code"]})
        dd = os.path.join(STAGING, "code", L)
        with open(os.path.join(dd, "batch-901-input.jsonl"), "w", encoding="utf-8") as f:
            for x in inputs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        with open(os.path.join(dd, "batch-901-input.md"), "w", encoding="utf-8") as f:
            f.write(render_md(inputs, "code"))
        with open(os.path.join(dd, "batch-901-output.jsonl"), "w", encoding="utf-8") as f:
            for x in outputs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(cfile, "w", encoding="utf-8") as f:
        for c in sorted(cands, key=lambda c: (c["lineage"], c["date"])):
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    # merge appended relations into overrides
    first = {}
    for fn in glob.glob(os.path.join(STAGING, "code", "L*", "batch-*-output.jsonl")):
        for r in jl(fn):
            first[r["candidate_id"]] = r
    ofile = os.path.join(STAGING, "code-overrides.jsonl")
    ov = {o["candidate_id"]: o for o in (jl(ofile) if os.path.exists(ofile) else [])}
    for cid, rel in APPEND.items():
        o = ov.setdefault(cid, {"candidate_id": cid})
        rels = list(o.get("relations") or first[cid].get("relations") or [])
        if not any(x.get("target") == rel["target"] and x.get("relation") == rel["relation"] for x in rels):
            rels.append(rel)
        o["relations"] = rels
    with open(ofile, "w", encoding="utf-8") as f:
        for o in ov.values():
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("anchors written; overrides", len(ov))


if __name__ == "__main__":
    main()
