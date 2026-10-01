# L2 NEW artifact proposals from stage-2 coding (30 names)

## NEW:dsv4_unified_kv_triton  (12 events)
- L2-f2bcdb0508 2026-06-09 PR #27380: [AMD] Add unified kv attention support in dpsk-v4 (#27380) | paths:  | notes: NEW artifact retained from stage-1; registry lacks an exact dedicated artifact id.
- L2-371b96e210 2026-06-12 PR #27972: [AMD] Fix DeepSeek-V4-Flash-FP8 on MI300 (#27972) | paths:  | notes: NEW artifact retained from stage-1; registry lacks an exact dedicated artifact id.
- L2-10d3337048 2026-06-13 PR #27935: [AMD] Support unified_kv_triton for disaggregation (#27935) | paths:  | notes: NEW artifact retained from stage-1; registry lacks an exact dedicated artifact id.
- L2-a362ba9da3 2026-06-16 PR #27928: [AMD] Feat: Add prefill context parallel support for deepseek v4 unified kv attention (#27928) | paths: python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py; python/sglang/srt/models/deepseek_v4.py | notes: NEW artifact retained from stage-1; registry lacks an exact dedicated artifact id.
- L2-f5b041622b 2026-06-17 PR #28520: [AMD] Fix deepseek-v4 mtp accept length issue (#28520) | paths:  | notes: NEW artifact retained from stage-1; registry lacks an exact dedicated artifact id.
- L2-24d15dd92e 2026-06-18 PR #28541: [AMD][DSV4] fix nonetype issue when enabling hicache (#28541) | paths:  | notes: NEW artifact retained from stage-1; registry lacks an exact dedicated artifact id.

## NEW:verify_mla_kernel  (3 events)
- L2-6679d9b60c 2026-08-07 PR #33981: [AMD] Add K3 verified mla kernel for DSpark on triton backend (#33981) | paths: python/sglang/kernels/ops/attention/verify_mla.py; python/sglang/srt/layers/attention/triton_backend.py | notes: Uses NEW:verify_mla_kernel because registry lacks a dedicated artifact for this newly introduced AMD verify MLA kernel.
- L2-0977b22431 2026-08-10 PR #34261: [AMD] Restore K3 MLA verify kernel path blocked by can_handle() guard (#34261) | paths: python/sglang/kernels/ops/attention/verify_mla.py | notes: NEW:verify_mla_kernel is late verify_mla.py kernel family not in registry.
- L2-6cbfa791d6 2026-08-15 PR #34517: [AMD][Spec] Accelerate Qwen3.5 verification with grouped-head shared KV (#34517) | paths: python/sglang/kernels/ops/attention/verify_mla.py | notes: 

## NEW:dcp_lse_merge  (3 events)
- L2-955aab8db1 2026-08-10 PR #34213: [DCP] Reuse partial output in natural-log LSE merge (#34213) | paths:  | notes: NEW:dcp_lse_merge covers DCP MLA LSE reduce code not yet in registry.
- L2-6c6294b7be 2026-08-12 PR #34614: [DCP] Fuse the a2a pack/unpack copies in the MLA LSE reduce (#34614) | paths: python/sglang/kernels/ops/attention/__init__.py; python/sglang/kernels/ops/attention/dcp_kernels.py | notes: NEW:dcp_lse_merge covers DCP MLA LSE reduce kernels outside registry.
- L2-81fe452810 2026-08-13 PR #34651: [DCP] Share one pack kernel between both a2a backends (#34651) | paths:  | notes: NEW:dcp_lse_merge covers DCP MLA LSE reduce kernels outside registry.

## NEW:dsv4_nonpaged_indexer  (2 events)
- L2-a6ee64d237 2026-07-02 PR #29619: [DeepSeek-V4] Add an opt-in non-paged indexer for long-context prefill (#29619) | paths: python/sglang/srt/layers/attention/deepseek_v4_backend.py; python/sglang/srt/layers/attention/dsv4/indexer.py; python/sglang/srt/layers/attention/dsv4/metadata.py | notes: 
- L2-48ad6a83cf 2026-07-07 PR #30140: [DeepSeek-V4] Enable non-paged indexer by default for large prefill chunks (#30140) | paths: python/sglang/srt/layers/attention/dsv4/indexer.py | notes: 

## NEW:dsv4_unified_kv_pool  (2 events)
- L2-514b45fd34 2026-09-05 PR #30315: [AMD][DSV4] Fix unified-KV pool sizing and SWA ring accounting (#30315) | paths:  | notes: 
- L2-e54009240a 2026-09-20 PR #38901: [AMD][DSV4] feat: enable DSpark with fp8 unified_kv on gfx950 (#38901) | paths:  | notes: 

## NEW:dsv4_hip_radix_backend  (2 events)
- L2-21289cfd50 2026-09-12 PR #39116: [AMD] Fix Dspark accept length and reduce host bubble on DSV4 (#39116) | paths:  | notes: 
- L2-a813224e78 2026-09-16 PR #37810: [ROCm][DSV4] Enable breakable CUDA graph prefill (#37810) | paths:  | notes: 

## NEW:dsv4_fp8_unified_kv  (2 events)
- L2-5aa9b8fb3e 2026-09-14 PR #37413: [AMD][DSV4] feat: enable fp8 two-pool unified_kv on gfx950 (#37413) | paths:  | notes: 
- L2-6c73368c32 2026-09-17 PR #37778: [AMD][DSV4] Enable hicache on deepseek-v4 fp8 unified attn (#37778) | paths:  | notes: 

## NEW:intel_xpu_flash_mla_decode  (1 events)
- L2-9dfb1d2ebe 2026-05-07 PR #24372: [Intel GPU] Fix flash_mla_get_workspace_size call in intel_xpu (#24372) | paths: python/sglang/srt/layers/attention/xpu_backend.py | notes: NEW artifact used because no matching L2 registry artifact exists.

## NEW:glm47_flash_mla_model  (1 events)
- L2-7ef06bfc06 2026-05-26 PR #26088: GLM-4.7-Flash: standalone MLA impl and MLA NextN/MTP (#26088) | paths:  | notes: NEW artifact is a model-side MLA implementation not present in L2 registry.

## NEW:dsv3_fused_a_gemm  (1 events)
- L2-e4253b39e2 2026-06-27 PR #27397: Support JIT fused A GEMM (MLA down projection) and support GLM-5 hidden size, SM120 (#27397) | paths: python/sglang/srt/models/deepseek_v2.py | notes: 

## NEW:q8kv8_sparse_mla_prefill_sm90  (1 events)
- L2-bc8b3ab1f5 2026-06-30 PR #25751: [Kernel] Add SM90 Q8KV8 FP8 Sparse MLA Prefill JIT Kernel with Tests and Benchmark (#25751) | paths:  | notes: 

## NEW:dspark_dsv4_attention  (1 events)
- L2-6cc9352dfe 2026-07-12 PR #30261: [Spec] Add DSpark: confidence-scheduled speculative decoding (#30261) | paths:  | notes: 

## NEW:concat_and_cast_mha_k_pad_kernel  (1 events)
- L2-4d0c5a89af 2026-08-15 PR #34837: [AMD] Add concat_and_cast_mha_k_pad_kernel to support 12-head and enable K3 aiter prefill kernel (#3 | paths:  | notes: 

## NEW:verify_mla_triton  (1 events)
- L2-34180a0d35 2026-08-20 PR #35499: [AMD] Improve K3 dspark draft attn kernel perf (#35499) | paths: python/sglang/kernels/ops/attention/verify_mla.py | notes: NEW:verify_mla_triton is separate DSpark draft attention use of verify_mla.py.

## NEW:dsv4_unified_kv_decode  (1 events)
- L2-2a96ebf648 2026-08-28 PR #36094: [AMD][DSV4] perf: retune decode split-K heuristic for MI355X (#36094) | paths:  | notes: NEW:dsv4_unified_kv_decode is late DSV4 unified-KV kernel outside registry.

## NEW:lean_attention_triton  (1 events)
- L2-4944e50e2c 2026-08-28 PR #33576:  [AMD] Add Work-Centric (Lean) Attention: a persistent-CTA decode kernel for long-context serving (# | paths:  | notes: NEW:lean_attention_triton introduced in decode_attention.py; related to triton decode lineage but new artifact.

## NEW:pd_dcp_gather  (1 events)
- L2-3760296be8 2026-08-28 PR #35762: [PD] Pack DCP1→DCP-N PD KV transfers into dest-contiguous RDMA blocks (#35762) | paths:  | notes: 

## NEW:dsv4_unified_attention_sink  (1 events)
- L2-7825e5ffca 2026-09-03 PR #35092: [AMD] Fix DSV4 unified attention sink TP slice (#35092) | paths:  | notes: 

## NEW:triton_mla_prefill  (1 events)
- L2-3c2724c48d 2026-09-04 PR #35770: [AMD] Optimize Kimi-K3 Triton MLA prefill on gfx950 (#35770) | paths: python/sglang/kernels/ops/attention/extend_attention.py; python/sglang/srt/layers/attention/triton_backend.py; python/sglang/srt/models/deepseek_common/attention_backend_handler.py; python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py | notes: 

## NEW:dsv4_unified_kv_cache  (1 events)
- L2-e9e9e37ddc 2026-09-07 PR #32759: [AMD] Restore SWA reprefill-tail on UnifiedRadixCache when HiCache is off (#32759) | paths:  | notes: 

## NEW:intel_xpu_mla_backend  (1 events)
- L2-cf35384fe4 2026-09-08 PR #35866: [Intel GPU] Add MLA support to Intel XPU Attention backend for Prefill (#35866) | paths: python/sglang/srt/layers/attention/xpu_backend.py | notes: 

## NEW:dsv4_trtllm_attention  (1 events)
- L2-880d6fa64d 2026-09-09 PR #30805: [DSv4] Integrate TRT-LLM DSv4 Attention for SM100/103 (#30805) | paths: python/sglang/srt/layers/attention/attention_registry.py | notes: 

## NEW:dsv4_1_standalone_kernels  (1 events)
- L2-91f691c490 2026-09-15 PR #39646: dsv4.1: standalone kernels and Python wrappers (#39646) | paths: python/sglang/kernels/jit/csrc/deepseek_v4/flashmla_sched_meta.cuh; python/sglang/kernels/ops/attention/dsv4/flashmla_sched_meta.py | notes: 

## NEW:dsv4_1_kv_layout  (1 events)
- L2-13d593b6cf 2026-09-16 PR #39652: dsv4.1: compression, KV I/O, and metadata kernels (#39652) | paths:  | notes: 

## NEW:dsv4_compress_ratio_metadata  (1 events)
- L2-1f0c73e9bd 2026-09-17 PR #39921: [DSV4] Generalize attention metadata, sparse prefill, and KV pool over compress ratios (#39921) | paths:  | notes: 

## NEW:hisparse_mla_slot_mapping  (1 events)
- L2-59dd2fc734 2026-09-20 PR #39837: [2/N] [Kernel] Fuse padding-preserving HiSparse slot translation (#39837) | paths: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py | notes: 

## NEW:hicache_tma_mla_transfer  (1 events)
- L2-7ad55e4386 2026-09-21 PR #40278: [HiCache] TMA-staged host<->device KV transfer kernel (sm_90+) (#40278) | paths:  | notes: 

## NEW:dsv4_kv_store  (1 events)
- L2-ec070ec8c8 2026-09-24 PR #41159: [AMD] Fix int32 offset overflow in Triton DSv4 KV store kernels (#41159) | paths:  | notes: 

## NEW:page_unified_mla_loadback  (1 events)
- L2-e581520c67 2026-09-28 PR #39726: [HiCache] Add the page-unified KV load-back JIT kernel  (#39726) | paths:  | notes: 

## NEW:dsv4_amd_sparse_decode  (1 events)
- L2-096b066fb4 2026-09-28 PR #41020: dsv4.1-amd: gfx950 sparse decode attention and sorted top-k (#41020) | paths:  | notes: 
