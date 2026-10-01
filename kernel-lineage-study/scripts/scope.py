"""Lineage scope definitions used by candidate discovery (WP-B).

Each lineage has:
  repo            primary repository
  core_paths      regexes for kernel implementation / dispatch paths. Every
                  first-parent commit touching one is a candidate.
  integ_paths     regexes for integration paths (layers, backends, model files,
                  server args, dependency pins). A commit touching one is a
                  candidate only if its subject/body matches `keywords` OR it
                  also touches a core path.
  keywords        subject/body regex: any commit matching is a candidate even
                  when it touches none of the paths (catches renamed/moved
                  paths, dependency bumps, reverts).
  exclude_paths   regexes for paths that never make a commit a candidate on
                  their own (docs, generic tests) — used only to avoid
                  counting pure-doc/test-only changes as path hits.
  config_paths    tuned-config JSON paths: commits touching ONLY these are
                  candidates flagged `routine_config`.

The scopes are deliberately over-inclusive; screening decides membership.
Version history of this file is recorded in CODEBOOK.md (identity rules).
"""

SCOPES = {
    "L1": {
        "repo": "sglang",
        "title": "SGLang MoE alignment, routing, top-k and fusion",
        "core_paths": [
            r"moe_align", r"align_block_size", r"(^|/)topk\.py$", r"grouped_topk", r"fused_gate",
            r"topk_softmax", r"topk_sigmoid", r"moe_sum", r"topk_reduce", r"routed_scal",
            r"(^|/)fused_moe[^/]*\.py$", r"fused_moe_triton/", r"triton_fused_moe/", r"fused_moe_grok/",
            r"(^|/)fused_moe/", r"layers/fused_moe", r"fused_moe_patch", r"fused_moe_native",
            r"ep_moe/", r"layers/moe/", r"token_dispatcher", r"moe_runner",
            r"csrc/moe/", r"sgl-kernel/.*moe", r"jit_kernel/.*moe", r"kernels/(aot|jit)/csrc/moe/",
            r"kernels/ops/moe/", r"python/sglang/kernels/.*moe", r"sgl_kernel/(fused_moe|moe|top_k|cutlass_moe)",
            r"csrc/cpu/(topk|moe)", r"hardware_backend/.*/moe/", r"(^|/)router\.py$",
            r"deepep", r"deep_ep", r"cutlass_moe", r"cutlass_w4a8_moe", r"flashinfer_.*moe", r"mega_moe",
            r"hash_topk", r"marlin_moe", r"triton_kernels_moe",
        ],
        "integ_paths": [
            r"python/sglang/srt/models/deepseek", r"python/sglang/srt/models/(qwen\d?_moe|mixtral|grok|dbrx|kimi|glm4_moe|gpt_oss|llama4|minimax)",
            r"python/sglang/srt/server_args\.py$", r"python/sglang/srt/model_executor/", r"python/pyproject\.toml$",
            r"python/sglang/srt/layers/quantization/", r"python/sglang/srt/eplb", r"python/sglang/srt/two_batch_overlap",
            r"sgl-kernel/(CMakeLists\.txt|setup\.py|pyproject\.toml)$", r"sgl-kernel/csrc/(common_extension|torch_extension)",
            r"sgl-kernel/python/sgl_kernel/__init__\.py$", r"sgl-kernel/include/sgl_kernel_ops\.h$",
        ],
        "keywords": r"(?i)(moe[_ ]?align|align[_ ]block|multi[- ]?block|grouped[_ ]?top[-_ ]?k|biased[_ ]grouped|top[-_ ]?k[_ ]softmax|topk[_ ]sigmoid|fused[_ ]gate|routed[_ ]scal|topk[_ ]reduce|moe[_ ]sum|sum[_ ]reduce|fused[_ ]?moe|\bmoe\b.*(kernel|triton|cutlass|flashinfer|deepep|deep_ep|deepgemm|marlin|runner|backend|fusion|fuse)|deepep|deep[_ ]ep|\bep[_ ]moe|expert[_ ]parallel|token[_ ]dispatch|moe[_ ]runner|select[_ ]experts|\brouter\b.*(kernel|fuse|topk)|w4a8.*moe|moe.*(fp8|fp4|nvfp4|mxfp4|int4|int8|w8a8|bf16))",
        "config_paths": [r"fused_moe_triton/configs/", r"moe_runner/triton_utils/configs/", r"configs/.*E=\d+"],
    },
    "L2": {
        "repo": "sglang",
        "title": "SGLang MLA, FlashInfer MLA and FlashMLA",
        "core_paths": [
            r"flashinfer_mla", r"flashmla", r"flash_mla", r"cutlass_mla", r"trtllm_mla", r"cutedsl_mla",
            r"tokenspeed_mla", r"hip_flash_mla", r"aiter_mla", r"(^|/)mla[_/]", r"_mla\.py$", r"_mla\.cu",
            r"concat_mla", r"set_mla_kv", r"mla_kv", r"forward_mla", r"triton_ops/decode_attention\.py$",
            r"triton_ops/rocm_mla", r"attention/attention_registry\.py$", r"sgl-kernel/.*flash_?mla",
            r"cmake/flashmla", r"3rdparty/flashmla", r"sgl-kernel/csrc/attention/.*(mla|sm100)",
            r"kernels/(aot|jit)/.*mla", r"kernels/ops/attention/.*mla", r"flash_attn/cute/.*mla",
        ],
        "integ_paths": [
            r"python/sglang/srt/models/deepseek", r"python/sglang/srt/server_args\.py$",
            r"python/sglang/srt/model_executor/(model_runner|cuda_graph_runner|forward_batch_info)",
            r"python/sglang/srt/layers/attention/", r"python/pyproject\.toml$", r"sgl-kernel/(CMakeLists\.txt|setup\.py|pyproject\.toml)$",
            r"python/sglang/srt/mem_cache/memory_pool\.py$", r"python/sglang/srt/layers/radix_attention\.py$",
            r"python/sglang/srt/speculative/", r"python/sglang/srt/compilation/", r"docker/",
            r"attention/(flashattention|flashinfer|triton|aiter|base_attn)_?backend\.py$", r"attention/utils\.py$",
            r"3rdparty/flash", r"cmake/flash_attention", r"sgl-kernel/csrc/attention/", r"kernels/ops/attention/",
        ],
        "keywords": r"(?i)(\bmla\b|flashinfer[_ ]mla|flash[_ ]?mla|cutlass[_ ]mla|trtllm[_ ]mla|cutedsl[_ ]mla|multi[- ]head latent|latent attention|absorb|deepseek.*(attention|attn|decode|prefill)|(fa3|fa4|flash[_ ]?attention).*(mla|deepseek)|kv[_ ]lora|q[_ ]nope|nope|rope.*mla|mla.*(cuda[_ ]graph|graph)|attention[_ ]backend.*(deepseek|mla|default))",
        "config_paths": [],
    },
    "L3": {
        "repo": "vllm",
        "title": "vLLM attention implementation family",
        "core_paths": [
            r"^csrc/attention", r"^csrc/rocm/attention", r"^csrc/cpu/(attention|mla)", r"paged_attn", r"paged_attention",
            r"attention/backends/(?!.*(mamba|gdn|linear_attn|short_conv|mamba2|mamba1))", r"attention/ops/(?!.*(mamba|gdn|linear|ssd|causal_conv))",
            r"v1/attention/(?!.*(mamba|gdn|linear_attn|short_conv))", r"attention/selector\.py$",
            r"attention/layer\.py$", r"attention/layers/", r"vllm_flash_attn", r"flash_attn", r"utils/flashinfer\.py$",
            r"xqa", r"trtllm.*(attn|attention|decode|mla)", r"flashmla", r"cutlass_mla", r"triton_mla",
            r"aiter.*(attn|attention|mla|_fa|paged)", r"prefix_prefill",
            r"triton_(unified|decode|flash|prefill)_attention", r"unified_attention", r"chunked_prefill_paged_decode",
            r"merge_attn_states", r"cmake/external_projects/(vllm_flash_attn|flashmla|flash_attn)",
            r"model_executor/layers/(mla|attention)", r"attention/utils/fa_utils", r"fa_utils\.py$",
            r"^csrc/cache_kernels", r"reshape_and_cache",
        ],
        "integ_paths": [
            r"^CMakeLists\.txt$", r"^setup\.py$", r"^requirements", r"^pyproject\.toml$", r"^docker/",
            r"vllm/platforms/", r"vllm/envs\.py$", r"vllm/config", r"vllm/engine/arg_utils\.py$",
            r"vllm/worker/", r"vllm/v1/worker/", r"vllm/compilation/", r"vllm/model_executor/models/(deepseek|llama\.py|gemma)",
            r"vllm/utils", r"vllm/_custom_ops\.py$", r"csrc/ops\.h$", r"csrc/torch_bindings\.cpp$",
        ],
        "keywords": r"(?i)(paged[_ ]?attention|pagedattention|flash[_ ]?attn|flash[_ ]?attention|\bfa[234]\b|flashinfer(?!.*(moe|sampl|all[-_ ]?reduce|gemm|mm\b|quant|fp4|a2a|all2all|alltoall|comm))|xqa|trtllm[-_ ]gen|trt[-_ ]?llm.*attn|triton.*attn|triton.*attention|unified[_ ]attention|prefix[_ ]prefill|attention[_ ]backend|attn[_ ]backend|\bmla\b|flash[_ ]?mla|cutlass[_ ]mla|aiter.*(attn|attention|mla|fa)|rocm.*(attn|attention)|decode[_ ]attention|split[-_ ]?kv|merge[_ ]attn|cascade|sliding[_ ]window.*attn|attention.*(kernel|backend|cuda[_ ]graph|dispatch|selector)|attn[_ ]selector|attention[_ ]selector|vllm[-_]flash[-_]attn)",
        "config_paths": [],
    },
}
