# L3 NEW artifact proposals from stage-2 coding (114 names)

## NEW:sparse_mla_indexer  (15 events)
- L3-9095cbbfb6 2026-03-10 PR #36519: [Bugfix][Sparse MLA] report indexer CG support properly (#36519) | paths: vllm/v1/attention/backends/mla/indexer.py | notes: Stage-1 NEW sparse_mla_indexer retained; registry lacks this indexer artifact.
- L3-9040cd40af 2026-03-11 PR #36723: [DSV3.2][MTP] Optimize Indexer MTP handling (#36723) | paths: vllm/v1/attention/backends/mla/indexer.py | notes: Registry lacks sparse MLA indexer artifact; stage-1 NEW retained.
- L3-c88ea8338b 2026-03-16 PR #36982: [MTP][Sparse MLA] Take advantage of native MTP support in indexer when possible (#36982) | paths: vllm/v1/attention/backends/mla/indexer.py | notes: NEW artifact: sparse MLA indexer not in registry.
- L3-09c3dc9186 2026-03-25 PR #37968: [Revert] Remove CUDA torch fallbacks for fp8_mqa_logits/fp8_paged_mqa_logits_torch function (#37968) | paths: vllm/v1/attention/backends/mla/indexer.py | notes: NEW artifact: sparse MLA indexer not in registry.
- L3-87f05d6880 2026-03-26 PR #38076: [Revert] Remove DeepGEMM availability check in DeepseekV32IndexerMetadataBuilder (#38076) | paths: vllm/v1/attention/backends/mla/indexer.py | notes: NEW artifact: sparse MLA indexer not in registry.
- L3-bcc6f67447 2026-03-30 PR #35431: [Bugfix] Use null block (0) for padded block table entries (#35431) | paths: vllm/v1/attention/backends/mla/indexer.py; vllm/v1/attention/backends/utils.py | notes: Includes NEW sparse indexer because registry lacks it.

## NEW:turboquant_attention  (9 events)
- L3-f4b42df048 2026-04-14 PR #38479: [Attention Backend] TurboQuant: 2-bit KV cache compression with 4x capacity (#38479) | paths: vllm/model_executor/layers/attention/attention.py; vllm/v1/attention/backends/registry.py; vllm/v1/attention/backends/turboquant_attn.py; vllm/v1/attention/ops/triton_turboquant_decode.py | notes: NEW:turboquant_attention absent from registry.
- L3-1174723eba 2026-04-17 PR #40060: Fix TURBOQUANT backend selection in cuda.py (#40060) | paths:  | notes: NEW:turboquant_attention absent from registry.
- L3-6ef1efd51f 2026-04-17 PR #39953: [ROCm] Fix TurboQuant on ROCm: backend routing, flash-attn compat, int64 overflow (#39953) | paths: vllm/v1/attention/backends/turboquant_attn.py; vllm/v1/attention/ops/triton_turboquant_decode.py; vllm/v1/attention/ops/triton_turboquant_store.py; vllm/platforms/rocm.py | notes: NEW:turboquant_attention absent from registry.
- L3-251c18d1f8 2026-04-17 PR #39957: skip fp8e4b15 on xpu (#39957) | paths: vllm/v1/attention/ops/triton_turboquant_decode.py | notes: NEW:turboquant_attention absent from registry.
- L3-ed0622e3a8 2026-04-18 PR #40194: [Attention] TurboQuant: remove redundant random signs, add prior art attribution (#40194) | paths: vllm/model_executor/layers/attention/attention.py; vllm/v1/attention/backends/turboquant_attn.py | notes: NEW:turboquant_attention absent from registry.
- L3-fe9c3d6c5f 2026-04-23 PR #40092: [TurboQuant] enable FA3/FA4 for prefill paths (#40092) | paths: vllm/v1/attention/backends/flash_attn.py; vllm/v1/attention/backends/turboquant_attn.py | notes: NEW:turboquant_attention absent from registry.

## NEW:deepseek_v4_attention_ops  (7 events)
- L3-4d51588e23 2026-04-26 PR #40860: [Feat] DeepSeek V4 Rebased  (#40860) | paths: cmake/external_projects/flashmla.cmake; vllm/model_executor/layers/attention/mla_attention.py; vllm/v1/attention/backend.py; vllm/v1/attention/backends/mla/compressor_utils.py | notes: NEW artifact covers new DeepSeek V4 attention/fused op family.
- L3-2ae73c758c 2026-04-28 PR #41135: [Bugfix] fix inductor error for dpsk v4 (#41135) | paths: vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py | notes: NEW artifact is DeepSeek V4 fused inv-RoPE FP8 op.
- L3-296741d025 2026-04-29 PR #41015: [DSv4] Use `cvt` PTX for FP32->FP4 conversion (#41015) | paths: vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py; vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py | notes: 
- L3-628c436301 2026-05-05 PR #40871: [New Model][ROCm] Add AMD support for DeepSeek V4 (#40871) | paths: vllm/v1/attention/backends/mla/sparse_swa.py; vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py; vllm/v1/attention/ops/rocm_aiter_mla_sparse.py | notes: 
- L3-530d371302 2026-05-09 PR #41428: [DSv4] Improved fused Indexer Q quant kernel (#41428) | paths: vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py; vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py | notes: 
- L3-724ed2fc35 2026-05-11 PR #42236: [DSv4] Improved dequant gather K cache kernel (#42236) | paths: vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py; vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py; vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py; vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py | notes: 

## NEW:minimax_m3_sparse_attention  (6 events)
- L3-0a1c5034f5 2026-06-16 PR #45381: [Model] Add MiniMax M3 support (#45381) | paths: vllm/model_executor/layers/attention/attention.py; vllm/v1/attention/backends/flashinfer.py; vllm/v1/attention/backends/registry.py; CMakeLists.txt | notes: NEW artifact for model-specific sparse attention not present in L3 registry.
- L3-4c62663315 2026-06-16 PR #45744: [M3] Enable FP8 sparse GQA (#45744) | paths:  | notes: 
- L3-efd15e192a 2026-06-17 PR #45720: [Bugfix][ROCm] Fix MiniMax-M3 FP8 KV cache dtype (#45720) | paths:  | notes: 
- L3-cee0f92c02 2026-09-03 PR #54682: [ROCm][Perf] Optimize MiniMax-M3 decode indexer and top-k (#54682) | paths:  | notes: NEW artifact not in L3 registry; MiniMax-M3 sparse attention is a model-specific L3 attention implementation.
- L3-f09c52a587 2026-09-13 PR #56170: [ROCm][Performance] Avoid blocking MiniMax M3 scalar upload (#56170) | paths:  | notes: NEW MiniMax sparse attention artifact not in registry.
- L3-dfa1984e58 2026-09-13 PR #55235: [ROCm][Perf] Tune MiniMax-M3 decode top-k for short contexts (#55235) | paths:  | notes: NEW MiniMax sparse attention artifact not in registry.

## NEW:triton_reshape_and_cache_flash  (4 events)
- L3-100b630a60 2025-09-23 PR #24503: [V1][Kernel] Add triton implementation for `reshape_and_cache_flash` (#24503) | paths: vllm/attention/ops/triton_reshape_and_cache_flash.py; vllm/v1/attention/backends/triton_attn.py | notes: NEW artifact is the Triton cache-write kernel added by this PR.
- L3-4f2954f724 2025-09-23 PR #25522: Fix triton_reshape_and_cache_flash.py triton import (#25522) | paths: vllm/attention/ops/triton_reshape_and_cache_flash.py | notes: NEW artifact follows PR #24503 triton reshape/cache kernel.
- L3-a3ae45a38c 2025-09-29 PR #25825: [Misc] fix tests failure by using current_platform (#25825) | paths: vllm/attention/ops/triton_reshape_and_cache_flash.py | notes: Registry lacks a Triton reshape_and_cache_flash artifact; used NEW for cache op wrapper.
- L3-53a2088675 2026-05-28 PR #43330: Allow native KV cache dtype in Triton cache update (#43330) | paths: vllm/v1/attention/ops/triton_reshape_and_cache_flash.py | notes: NEW artifact is Triton FlashAttention cache update op not in registry.

## NEW:deepgemm_sparse_attn_indexer  (4 events)
- L3-79028d4388 2026-02-05 PR #33568: [Perf] Disable clean_logits in deepgemm fp8_mqa_logits kernel (#33568) | paths:  | notes: NEW artifact: sparse attention indexer/DeepGEMM helper is not in L3 registry.
- L3-9fa5b25a23 2026-02-24 PR #35075: [Bug][DSV3.2] Always prepare metadata for DeepGEMM Sparse Attention (#35075) | paths: vllm/v1/attention/backends/mla/indexer.py | notes: Registry lacks sparse/DeepGEMM indexer artifact; used NEW id from stage-1.
- L3-7e08c22b8c 2026-02-28 PR #35271: [Feat] Add CUDA torch fallbacks for fp8_mqa_logits/fp8_paged_mqa_logits_torch function (#35271) | paths: vllm/v1/attention/backends/mla/indexer.py | notes: Registry lacks DeepGEMM sparse indexer artifact; stage-1 NEW retained.
- L3-28ef9ba399 2026-03-03 PR #34552: [BugFix] Add support for MTP num_speculative_tokens > 1 with sparse MLA (#34552) | paths: vllm/v1/attention/backends/mla/indexer.py; vllm/v1/worker/gpu_model_runner.py; vllm/v1/worker/utils.py | notes: Registry lacks sparse MLA indexer artifact; stage-1 NEW retained.

## NEW:kv_cache_interface  (3 events)
- L3-9607d5eb44 2025-09-19 PR #25101: [Hybrid Allocator] Support full attention with different hidden size  (#25101) | paths:  | notes: NEW artifact is KV-cache interface support path, not an existing registry artifact.
- L3-8ce53a616e 2026-07-20 PR #47574: [Bugfix] Zero new KV blocks for quantized + sliding-window hybrid caches (#47574) | paths:  | notes: NEW artifact is KV-cache policy, not a kernel body.
- L3-48ebd6f2f1 2026-07-26 PR #49226: [Bugfix][KVConnector] Disable cross-layer KV blocks for per-token-head quant (#49226) | paths:  | notes: completes issue deferred after #48411.

## NEW:mla_indexer  (3 events)
- L3-1e50f1be70 2025-10-02 PR #25999: [Deepseek v3.2] Support indexer prefill chunking (#25999) | paths: vllm/v1/attention/backends/mla/indexer.py | notes: Registry lacks MLA indexer artifact; used NEW.
- L3-a966aaed30 2026-04-29 PR #39277: [Bugfix][MLA] Size arange_buffer to max_num_batched_tokens to prevent CUDA IMA (#39277) | paths: vllm/v1/attention/backends/mla/indexer.py | notes: NEW artifact is MLA indexer metadata builder/buffer contract.
- L3-ee58665aac 2026-05-15 PR #42135: [Bugfix] Fix DeepGEMM context lens contiguity in MLA indexer (#42135) | paths: vllm/v1/attention/backends/mla/indexer.py | notes: NEW artifact is MLA indexer metadata helper not in registry.

## NEW:dsv4_qnorm_rope_kv_insert  (3 events)
- L3-6ab6ffb428 2026-05-26 PR #43162: [Feat][DSV4] Fuse q pad into deepseek v4 fused kernel (#43162) | paths: vllm/models/deepseek_v4/nvidia/flashmla.py | notes: NEW artifact is fused DeepSeek V4 qnorm/rope/KV insert kernel.
- L3-59d0236193 2026-06-04 PR #44365: [10b/n] Migrate custom all-reduce, DeepSeek V4 fused MLA, MiniMax reduce-RMS, and MXFP8 MoE to libto | paths: CMakeLists.txt; csrc/ops.h; csrc/torch_bindings.cpp | notes: NEW artifact matches fused DeepSeek V4 qnorm/rope/KV insert kernel introduced earlier in batch.
- L3-be1cb9834b 2026-09-10 PR #56215: [Kernel] Optional Q-norm in fused DSv4 MLA epilogue; group_size=32 for packed FP8 quant (#56215) | paths:  | notes: NEW DSv4 fused epilogue artifact not in registry.

## NEW:hpc_attention_backend  (3 events)
- L3-8cf7c4d8ad 2026-06-30 PR #46020: [Attention Backend] add HPC-Ops Attention backend (#46020) | paths: vllm/model_executor/layers/attention/attention.py; vllm/v1/attention/backends/hpc_attn.py; vllm/v1/attention/backends/registry.py; vllm/config/compilation.py | notes: Uses NEW artifact because registry has no HPC-Ops attention backend artifact.
- L3-95a248faed 2026-07-05 PR #47433: [Attention Backend] HPC_ATTN backend support mtp and dynamic scheduled attention (#47433) | paths: vllm/v1/attention/backends/hpc_attn.py | notes: Uses same NEW HPC-Ops backend artifact introduced earlier in this batch.
- L3-470297c143 2026-08-06 PR #50980: [HPC Attention Backend] hpc attention backend support bf16 kv cache with fp8 weight  (#50980) | paths: vllm/v1/attention/backends/hpc_attn.py | notes: NEW artifact because HPC_ATTN is not in the L3 registry.

## NEW:gather_cached_kv  (2 events)
- L3-e3cec88aa5 2023-04-10 PR #29: Memcpy kernel for flash attention (#29) | paths: csrc/cache_kernels.cu | notes: Temporary prefix-sharing cache kernel; removed in PR #3043.
- L3-d6e4a130b0 2024-02-26 PR #3043: [Minor] Remove gather_cached_kv kernel (#3043) | paths: csrc/cache_kernels.cu | notes: 

## NEW:attention_adapter  (2 events)
- L3-cf5f000d21 2025-01-10 PR #11677: [torch.compile] Hide KV cache behind torch.compile boundary (#11677) | paths: vllm/attention/layer.py | notes: NEW:attention_adapter is the forward_context Attention API adapter named by stage-1.
- L3-1f18adb245 2025-01-14 PR #12038: [Kernel] Revert the API change of Attention.forward (#12038) | paths: vllm/attention/layer.py | notes: Partial API-name revert, not a full revert commit.

## NEW:dual_chunk_flash_attn_backend  (2 events)
- L3-60f7624334 2025-05-12 PR #11844: Implements dual-chunk-flash-attn backend for dual chunk attention with sparse attention support (#11 | paths: csrc/attention/vertical_slash_index.cu; vllm/attention/backends/dual_chunk_flash_attn.py; CMakeLists.txt; csrc/ops.h | notes: Not in registry; kept NEW artifact from stage-1 dossier.
- L3-7c734ee09b 2025-07-23 PR #21364: [Bugfix][Qwen][DCA] fixes bug in dual-chunk-flash-attn backend for qwen 1m models. (#21364) | paths: vllm/attention/backends/dual_chunk_flash_attn.py | notes: No registry artifact exists for DCA; diff removes block_table argument passed to flash_attn_varlen_func.

## NEW:mla_dual_rms_norm_fusion  (2 events)
- L3-fb5635d3f9 2026-04-20 PR #39242: [ROCm] Add MLA dual RMS norm fusion (Q, KV) pass for DeepSeek/Kimi-K2 (#39242) | paths: vllm/compilation/passes/fusion/rocm_aiter_fusion.py; vllm/compilation/passes/pass_manager.py; vllm/config/compilation.py; vllm/config/vllm.py | notes: NEW:mla_dual_rms_norm_fusion absent from registry.
- L3-21b086d0aa 2026-04-20 PR #40386: [ROCm] Hotfix: guard MLA dual RMS norm fusion against older AITer versions (#40386) | paths: vllm/config/vllm.py | notes: NEW:mla_dual_rms_norm_fusion absent from registry.

## NEW:flash_attn_diffkv_backend  (2 events)
- L3-e8ee2a78db 2026-04-24 PR #40045: [Attention] use diff kv backend for mimo v2 flash (#40045) | paths: vllm/model_executor/layers/attention/attention.py; vllm/v1/attention/backends/fa_utils.py; vllm/v1/attention/backends/flash_attn_diffkv.py; vllm/vllm_flash_attn/flash_attn_interface.py | notes: NEW artifact is the new FlashAttentionDiffKV backend path.
- L3-66dfee7121 2026-05-03 PR #40737: [Bugfix] Fix degenerate KV cache stride causing TMA cudaErrorIllegalInstruction (#40737) | paths: vllm/v1/attention/backends/flash_attn.py; vllm/v1/attention/backends/flash_attn_diffkv.py; vllm/v1/attention/backends/flashinfer.py | notes: 

## NEW:block_table_kv_contract  (2 events)
- L3-51fda1ba44 2026-04-29 PR #40648: [Model Runner v2] Fix block table IMA issue (#40648) | paths:  | notes: NEW artifact is the block-table/KV metadata contract.
- L3-38e16678ba 2026-05-06 PR #39324: [Bugfix] Align block table for TRTLLM MLA edge-case (#39324) | paths: vllm/v1/worker/block_table.py; vllm/v1/worker/gpu/block_table.py | notes: 

## NEW:dsv4_sparse_compressor  (2 events)
- L3-adaa5e455a 2026-05-26 PR #43710: [DSv4] Refactor compressor & Fix ROCm compatibility (#43710) | paths:  | notes: NEW artifact is DeepSeek V4 sparse-attention compressor dispatch code.
- L3-035733515f 2026-06-02 PR #44161: [Kernel][DSv4] Optimize sparse FP8 compressor kernels (#44161) | paths:  | notes: NEW dsv4_sparse_compressor carried from previous refactor but not in registry.

## NEW:mla_sparse_swa  (2 events)
- L3-f5a8d73377 2026-07-01 PR #46995: [Spec Decode] DSpark (#46995) | paths: vllm/v1/attention/backends/mla/sparse_swa.py | notes: Uses NEW artifact because registry lacks MLA sparse SWA artifact.
- L3-f70caef48b 2026-07-06 PR #47474: [Perf] Cache `token_to_req_indices` for dsv4, 5x~6x kernel performance improvement (#47474) | paths: vllm/v1/attention/backend.py; vllm/v1/attention/backends/mla/sparse_swa.py | notes: Uses NEW mla_sparse_swa from PR #46995 because registry lacks this artifact.

## NEW:minimax_m3_rocm_aiter_sparse_pa  (2 events)
- L3-ee5a89f4d7 2026-07-12 PR #47287: [ROCm][MiniMax-M3] Add AITER sparse paged attention (#47287) | paths: vllm/_custom_ops.py | notes: Registry has no MiniMax-M3 artifact; kept stage-1 NEW artifact.
- L3-95aab66e95 2026-07-14 PR #47984: [ROCm][MiniMax-M3][Spec Decode] Support speculative decode with AITER sparse PA (#47984) | paths:  | notes: 

## NEW:kimi_k3_mla_kv_concat  (2 events)
- L3-903d2efe7e 2026-08-12 PR #51772: [Attention][MLA] Fuse Kimi-K3 chunked-context K/V packing (#51772) | paths: vllm/model_executor/layers/attention/mla_attention.py; vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py; vllm/v1/attention/backends/mla/prefill/base.py; vllm/v1/attention/backends/mla/prefill/flash_attn.py | notes: NEW fused Kimi-K3 kernel not in registry.
- L3-2e2ffd104b 2026-08-13 PR #52210: [CI Failure] Fix CUDA wheel build for the Kimi K3 fused MLA kernel (#52210) | paths:  | notes: 

## NEW:deepseek_v32_fused_attention  (2 events)
- L3-2cf82bcdd1 2026-08-31 PR #50005: [Bugfix][DCP] Fix NVIDIA DeepSeek-V3.2 / GLM-5.2 fused attention (#50005) | paths:  | notes: NEW artifact retained from stage1; model-specific fused attention is not in L3 registry.
- L3-19c018ec05 2026-09-04 PR #54908: [Bugfix][DCP] Materialize prefill keys on non-owner ranks (#54908) | paths:  | notes: NEW artifact retained from stage1; model-specific fused attention not in registry.

## NEW:hisparse_sparse_mla_cache  (2 events)
- L3-d43bb2f37f 2026-09-12 PR #53781: [3/N] HiSparse: host-resident sparse-MLA decode hot-buffering (#53781) | paths: vllm/model_executor/layers/attention/mla_attention.py; vllm/model_executor/layers/attention/sparse_mla_attention.py; vllm/model_executor/layers/mla.py; vllm/v1/attention/backend.py | notes: NEW HiSparse artifact not in registry.
- L3-e19a3e172e 2026-09-12 PR #56629: [5/N] Share HiSparse host cache across TP ranks (reopens #52760) (#56629) | paths:  | notes: 

## NEW:sparse_indexer_topk  (2 events)
- L3-c6fa1f05d1 2026-09-15 PR #56346: [Perf][Kernel] Add sampled filtering for persistent top-k (#56346) | paths:  | notes: NEW sparse indexer top-k artifact not in registry.
- L3-dffbb714e4 2026-09-15 PR #56743: [ROCm][Perf] Optimize DSV4.1 K=512 decode top-k on gfx950 (#56743) | paths: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py | notes: 

## NEW:early_flashattention_prompt_adapter  (1 events)
- L3-3e9f991d6a 2023-03-01 PR #4: Use FlashAttention for `multi_query_kv_attention` (#4) | paths:  | notes: Early prompt adapter predates registry FlashAttention backend; kept as NEW artifact.

## NEW:cache_block_copy  (1 events)
- L3-0f40557af6 2023-04-07 PR #32: Implement block copy kernel to optimize beam search (#32) | paths: csrc/cache_kernels.cu | notes: Auxiliary cache kernel not listed in registry; NEW artifact.

## NEW:early_xformers_prompt_adapter  (1 events)
- L3-c9d5b6d4a8 2023-05-05 PR #70: Replace FlashAttention with xformers (#70) | paths:  | notes: Early prompt backend before registry xFormers backend; NEW artifact.

## NEW:early_xformers_adapter  (1 events)
- L3-2a4ec90854 2023-08-23 PR #834: Fix for breaking changes in xformers 0.0.21 (#834) | paths: vllm/model_executor/layers/attention.py | notes: 

## NEW:early_xformers_flashattn_selection  (1 events)
- L3-75471386de 2023-08-29 PR #877: use flash-attn via xformers (#877) | paths: vllm/model_executor/layers/attention.py | notes: Early selection path not represented as registry artifact.

## NEW:early_rocm_attention_selection  (1 events)
- L3-0580aab02f 2024-02-10 PR #2768: [ROCm] support Radeon™ 7900 series (gfx1100) without using flash-attention (#2768) | paths: vllm/model_executor/layers/attention.py; setup.py; Dockerfile.rocm | notes: Selection path predates formal platform selector artifact; NEW artifact.

## NEW:differential_flash_attn  (1 events)
- L3-2c11a738b3 2025-07-12 PR #20702: [Model] New model support for microsoft/Phi-4-mini-flash-reasoning (#20702) | paths: vllm/attention/backends/blocksparse_attn.py; vllm/attention/backends/differential_flash_attn.py; vllm/attention/backends/dual_chunk_flash_attn.py; vllm/attention/backends/flash_attn.py | notes: No registry artifact existed; coded as NEW:differential_flash_attn.

## NEW:mla_kv_transfer_contract  (1 events)
- L3-80608ba5af 2025-09-30 PR #25902: [NIXL] Add support for MLA caches with different latent dim (#25902) | paths:  | notes: Registry has no KV-transfer artifact for MLA cache contract.

## NEW:mla_kv_scale_loading  (1 events)
- L3-a26917332f 2025-10-03 PR #25968: [Quantization/NVFP4] Speed up TRTLLM NVFP4 MOE weight loading and fix K/V scale loading for MLA Attn | paths:  | notes: Mostly weight-loading/MoE; kept because MLA K/V scale loading is explicit but not registry artifact.

## NEW:cascade_attention_kv_scheduler  (1 events)
- L3-cd9890544b 2025-10-08 PR #23485: fix(v1/kv_cache): resolve async KV transfer bug in cascade attention (#23485) | paths:  | notes: Scheduler artifact is not in registry; used NEW.

## NEW:gather_indexer_k_quant_cache  (1 events)
- L3-127c8b782a 2025-10-08 PR #25931: Add gather_indexer_k_quant_cache kernel (#25931) | paths: csrc/cache_kernels.cu | notes: New helper op not present in registry.

## NEW:trtllm_ragged_mla_prefill  (1 events)
- L3-0f67d4d962 2025-10-24 PR #26397: [Attention] Add MLA prefill backend: trtllm_ragged_attention_deepseek (#26397) | paths: vllm/v1/attention/backends/mla/common.py; vllm/envs.py | notes: NEW artifact retained because registry has TRTLLM decode but no ragged MLA prefill artifact.

## NEW:context_parallel_lse_common  (1 events)
- L3-2d9ee28cab 2025-11-24 PR #29338: [CI/Test Fix] Fix CP tests on Blackwell (#29338) | paths: vllm/attention/ops/common.py | notes: Uses NEW artifact from stage-1 because registry has no context-parallel LSE helper artifact.

## NEW:cache_copy_blocks_ops  (1 events)
- L3-4ed11105d7 2025-12-23 PR #30967: [Misc] Remove unused custom ops `copy_blocks` and `copy_blocks_mla` (#30967) | paths: csrc/cache_kernels.cu | notes: NEW artifact retained from stage1 because registry has no durable copy-block artifact ID.

## NEW:static_sink_attention  (1 events)
- L3-3f52fa5aa2 2025-12-30 PR #28775: [Model] Add support for openPangu moe model (#28775) | paths: vllm/attention/backends/registry.py; vllm/attention/layer.py; vllm/attention/layers/static_sink_attention.py; vllm/attention/ops/triton_reshape_and_cache_flash.py | notes: Added NEW:flash_diffkv_attention because the diff introduces a separate FlashDiffkv backend not named in stage1 artifact

## NEW:flash_diffkv_attention  (1 events)
- L3-3f52fa5aa2 2025-12-30 PR #28775: [Model] Add support for openPangu moe model (#28775) | paths: vllm/attention/backends/registry.py; vllm/attention/layer.py; vllm/attention/layers/static_sink_attention.py; vllm/attention/ops/triton_reshape_and_cache_flash.py | notes: Added NEW:flash_diffkv_attention because the diff introduces a separate FlashDiffkv backend not named in stage1 artifact

## NEW:fused_rope_mla_cache  (1 events)
- L3-80fead8bf6 2026-01-09 PR #25774: Fuse RoPE and MLA KV-cache write (#25774) | paths: csrc/cache_kernels_fused.cu; CMakeLists.txt; csrc/torch_bindings.cpp; vllm/_custom_ops.py | notes: PR says kernel is added but not integrated by default.

## NEW:v1_attention_common_ops  (1 events)
- L3-3a6d5cbefd 2026-01-27 PR #33102: [Perf] Optimize dcp allocate tensor (#33102) | paths: vllm/v1/attention/ops/common.py | notes: NEW artifact is an attention common op helper not in registry.

## NEW:cascade_attention_selection  (1 events)
- L3-8f5d51203b 2026-01-30 PR #32561: Disable Cascade Attention for Batch Invariance (#32561) | paths: vllm/config/vllm.py | notes: NEW artifact used because cascade-attention selection policy is not separately registered.

## NEW:topk_per_row_sparse_attention  (1 events)
- L3-afdce12c89 2026-02-10 PR #33680: [Perf][Kernel] Add faster topKperRow decode kernel for DeepSeek-V3.2 sparse attention (#33680) | paths: vllm/v1/attention/backends/mla/indexer.py | notes: Stage1 said introduce; coded as port because source explicitly identifies SGLang/TileLang adaptation.

## NEW:concat_mla_q  (1 events)
- L3-580864d81e 2026-03-09 PR #34917: [Attention][Perf][Kernel] Replace torch.cat with vectorized CUDA kernel MLA query concat - DeepSeek- | paths: csrc/cache_kernels.cu; vllm/v1/attention/backends/mla/flashmla_sparse.py; csrc/torch_bindings.cpp; vllm/_custom_ops.py | notes: New kernel artifact not in registry; NEW retained.

## NEW:flash_attn_diffkv  (1 events)
- L3-93b3ec1585 2026-03-30 PR #36466: feat(attention): extract KV-cache update from FlashAttentionDiffKV ba… (#36466) | paths: vllm/v1/attention/backends/flash_attn_diffkv.py | notes: NEW flash_attn_diffkv variant absent from registry.

## NEW:turboquant_attention_backend  (1 events)
- L3-2cc008e7b4 2026-04-27 PR #40941: [Attention][TurboQuant] Share dequant buffers, eliminate float16_copy (#40941) | paths: vllm/model_executor/layers/attention/attention.py; vllm/v1/attention/backends/turboquant_attn.py; vllm/v1/attention/ops/triton_decode_attention.py; vllm/v1/attention/ops/triton_turboquant_decode.py | notes: NEW artifact is TurboQuant attention backend.

## NEW:dcp_alltoall_attention  (1 events)
- L3-4f7bde572a 2026-05-01 PR #41160: [Kernel] Pack output and LSE in DCP A2A (#41160) | paths: vllm/v1/attention/ops/dcp_alltoall.py | notes: NEW artifact is distributed context-parallel all-to-all attention path.

## NEW:mla_prefill_backends  (1 events)
- L3-f3fef12350 2026-05-01 PR #32623: [Attention] Abstract the MLA prefill backends and eliminate cuDNN (#32623) | paths: vllm/model_executor/layers/attention/mla_attention.py; vllm/v1/attention/backends/mla/prefill/__init__.py; vllm/v1/attention/backends/mla/prefill/base.py; vllm/v1/attention/backends/mla/prefill/flash_attn.py | notes: NEW artifact is MLA prefill backend selector/registry family.

## NEW:rocm_triton_sparse_mla_dsv4  (1 events)
- L3-7863fff6e5 2026-05-11 PR #41812: [ROCm][DSv4] implement flash sparse mla with triton kernels (#41812) | paths: vllm/v1/attention/backends/mla/flashmla_sparse.py; vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py; vllm/v1/attention/backends/mla/sparse_swa.py; vllm/v1/attention/ops/rocm_aiter_mla_sparse.py | notes: NEW artifact is ROCm Triton sparse MLA DSV4 backend/kernel.

## NEW:mla_rope_kvcache_cat_fusion  (1 events)
- L3-a51376b3f0 2026-05-11 PR #40392: [Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA (#40392) | paths: csrc/cache_kernels_fused.cu; vllm/model_executor/layers/attention/mla_attention.py; vllm/compilation/passes/fusion/matcher_utils.py; vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py | notes: NEW artifact is compiler fusion pass plus fused MLA cache kernel path.

## NEW:tokenspeed_mla_backend  (1 events)
- L3-0d2732dd91 2026-05-13 PR #41778: [MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell ( | paths: vllm/model_executor/layers/attention/mla_attention.py; vllm/v1/attention/backends/mla/prefill/registry.py; vllm/v1/attention/backends/mla/prefill/selector.py; vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py | notes: NEW artifact is TokenSpeed MLA backend.

## NEW:mla_prefill_trtllm_ragged  (1 events)
- L3-2317682f95 2026-05-14 PR #42112: [Bugfix] Fix TRTLLM ragged MLA prefill workspace warmup (#42112) | paths: vllm/v1/attention/backends/mla/prefill/flashinfer.py; vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py | notes: NEW artifact is TRTLLM ragged MLA prefill backend.

## NEW:attention_layer_kv_cache_config  (1 events)
- L3-852f567444 2026-05-15 PR #42782: [Bugfix] Respect explicit --kv-cache-dtype over checkpoint kv_cache_scheme (#42782) | paths: vllm/model_executor/layers/attention/attention.py | notes: NEW artifact is per-layer attention KV-cache config handling not represented in registry.

## NEW:vertical_slash_attention_index  (1 events)
- L3-00e20e76f7 2026-05-18 PR #42767: [Refactor] Remove dead cuda kernels (#42767) | paths: csrc/attention/vertical_slash_index.cu | notes: NEW artifact is dead vertical/slash index conversion CUDA kernels; peripheral to attention lineage.

## NEW:dsv4_indexer_q_cutedsl  (1 events)
- L3-3ca8db2ef8 2026-05-18 PR #42899: add cutedsl dsv4 indexer fp8 kernel (#42899) | paths: vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py; vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py; vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py | notes: NEW artifact is DeepSeek V4 CuTe DSL FP8 indexer op not in registry.

## NEW:turboquant_attn_backend  (1 events)
- L3-b29cbf0652 2026-05-21 PR #42988: [Perf] `zeros` -> `empty` to remove additional fill (#42988) | paths: vllm/v1/attention/backends/turboquant_attn.py | notes: Mostly adjacent backend/quant utilities; NEW TurboQuant attention backend not in L3 registry.

## NEW:mla_prefill_registry  (1 events)
- L3-aa6138169f 2026-05-26 PR #43325: [MLA][Attention] Add OOT MLA prefill backend registration mechanism (#43325) | paths: vllm/v1/attention/backends/mla/prefill/__init__.py; vllm/v1/attention/backends/mla/prefill/registry.py | notes: NEW registry mechanism for OOT MLA prefill backends.

## NEW:triton_diffkv_attention  (1 events)
- L3-f81daf8880 2026-06-11 PR #41797: [Attention] add triton diff-kv backend for mimo (#41797) | paths: vllm/v1/attention/backends/flash_attn_diffkv.py; vllm/v1/attention/backends/registry.py; vllm/v1/attention/backends/triton_attn_diffkv.py; vllm/v1/attention/ops/triton_unified_attention_diffkv.py | notes: 

## NEW:mla_trtllm_ragged_prefill  (1 events)
- L3-9d4dc4ca2f 2026-06-16 PR #43525: [Kernel] Support GLM-5 dimensions for TRT-LLM ragged MLA prefill (#43525) | paths: vllm/v1/attention/backends/mla/prefill/base.py; vllm/v1/attention/backends/mla/prefill/flashinfer.py; vllm/v1/attention/backends/mla/prefill/selector.py; vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py | notes: NEW artifact for TRTLLM ragged MLA prefill selector/backend not in registry.

## NEW:prefill_prefix_lm_attention  (1 events)
- L3-a52205bccf 2026-06-16 PR #43098: [Model] Add HrmTextForCausalLM (Hierarchical Reasoning Model — Text) (#43098) | paths: vllm/model_executor/layers/attention/__init__.py; vllm/model_executor/layers/attention/prefill_prefix_lm_attention.py | notes: NEW wrapper artifact for HRM PrefixLM attention.

## NEW:minimax_m3_sparse_indexer  (1 events)
- L3-6691f087a6 2026-06-23 PR #45892: [Minimax-M3] BF16/FP8 Indexer using MSA (#45892) | paths: cmake/external_projects/fmha_sm100.cmake; setup.py | notes: NEW artifact for MiniMax-specific sparse indexer.

## NEW:sparse_attention_topk_indexer  (1 events)
- L3-855cd4d787 2026-06-23 PR #43008: [Perf][DSv4/DSv3.2] Add cluster-cooperative topK kernel for low-latency scenarios (#43008) | paths:  | notes: NEW artifact for sparse attention topK indexer, not registry attention backend.

## NEW:minimax_m3_rocm_sparse_attn  (1 events)
- L3-c63cd4906c 2026-06-25 PR #46546: [ROCm][ [Perf] sparse attention optimization on minimax-m3  (#46546) | paths:  | notes: Uses NEW artifact because registry has no MiniMax-M3 ROCm sparse attention artifact.

## NEW:mla_rocm_aiter_fa_prefill  (1 events)
- L3-5ecae3266c 2026-06-28 PR #45033: [ROCm][Perf][MLA] Add AITER FlashAttention MLA prefill backend (`ROCM_AITER_FA`) (#45033) | paths: vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py; vllm/v1/attention/backends/mla/prefill/registry.py; vllm/v1/attention/backends/mla/prefill/selector.py | notes: Uses NEW artifact because registry lacks a separate ROCm AITER MLA prefill artifact.

## NEW:sparse_attention_indexer  (1 events)
- L3-9a08a5118e 2026-06-30 PR #47164: fix: skip cooperative top-K on SM120 (#47164) | paths:  | notes: Uses NEW artifact because sparse_attention_indexer is not in L3 registry.

## NEW:rocm_aiter_mla_norm_quant_fusion  (1 events)
- L3-09663abde0 2026-07-02 PR #44977: [ROCm][MLA] Fuse MLA q/kv RMSNorm + FP8 per-token quant in the FP8 attention path (#44977) | paths: vllm/compilation/passes/fusion/rocm_aiter_fusion.py | notes: Uses NEW artifact because registry lacks this fusion artifact.

## NEW:minimax_m3_msa_sparse_attention  (1 events)
- L3-9021589498 2026-07-07 PR #47502: [Minimax-M3] Using tok_sparse_select from MSA instead of triton kernels (#47502) | paths: cmake/external_projects/fmha_sm100.cmake | notes: Registry has no MiniMax-M3 artifact; kept stage-1 NEW artifact.

## NEW:triton_turboquant_store  (1 events)
- L3-b12cca6a23 2026-07-10 PR #39988: [Bugfix] Fix turboquant FP8 cast failure for BF16 models on Ampere GPUs (#39988) | paths: vllm/v1/attention/ops/triton_turboquant_store.py | notes: Turboquant attention KV-store is not in L3 registry; kept NEW artifact.

## NEW:mla_tokenspeed_backend  (1 events)
- L3-7fc97042c3 2026-07-13 PR #48180: Add DCP + Eagle support for Tokenspeed MLA backends (#48180) | paths: vllm/utils/flashinfer.py; vllm/v1/attention/backends/flashinfer.py; vllm/v1/attention/backends/mla/tokenspeed_mla.py; requirements/cuda.txt | notes: Tokenspeed MLA is not in L3 registry; kept stage-1 NEW artifact.

## NEW:kv_transfer_layout  (1 events)
- L3-7154856f3d 2026-07-26 PR #47791: [Bugfix] Fix handling 5D KV cache in kv_postprocess_layout_on_receive (#47791) | paths:  | notes: repairs layout break after #42095.

## NEW:sparse_mla_masked_mha  (1 events)
- L3-82ae4164ee 2026-07-31 PR #48770: [2/N][Attention] Enable masked MHA for sparse MLA prefills (#48770) | paths: cmake/external_projects/vllm_flash_attn.cmake; vllm/model_executor/layers/attention/mla_attention.py; vllm/model_executor/layers/attention/sparse_mla_attention.py; vllm/model_executor/layers/attention/sparse_mla_mask.py | notes: optimized_from #47327; integrates flash-attention#155.

## NEW:dsv4_fused_kv_insert  (1 events)
- L3-df71917cf1 2026-07-31 PR #49236: [DSv4 Perf] Optimize workspace reuse for eager break, 3.9% E2E TTFT improvement. (#49236) | paths: vllm/models/deepseek_v4/nvidia/flashmla.py | notes: Deep-study notes later reverted by PR #52836 outside this batch.

## NEW:minimax_m3_msa_cutlass_decode  (1 events)
- L3-0055b8bfa3 2026-08-02 PR #50032: [Attention][MiniMax-M3] Add MSA speculative decode verification (#50032) | paths: vllm/v1/attention/backends/registry.py; cmake/external_projects/fmha_sm100.cmake | notes: integrates vllm-project/MSA#10.

## NEW:dsa_fused_q_cutedsl  (1 events)
- L3-3756bf1e2b 2026-08-04 PR #49792: [Kernel][SM100] Add a CuTeDSL fused query kernel (#49792) | paths:  | notes: SM100 bf16 fixed head dims and quantized MQA required.

## NEW:tokenspeed_mla  (1 events)
- L3-8ae8337ffa 2026-08-04 PR #50911: [Spec Decode] Enable fused non-causal TokenSpeed MLA for DSpark (#50911) | paths: vllm/v1/attention/backends/mla/tokenspeed_mla.py | notes: DSpark draft positions are produced in one non-causal forward pass.

## NEW:dsa_decode_kernels  (1 events)
- L3-2dfb8ba590 2026-08-06 PR #50230: [Perf][CUDA] Programmatic dependent launch for the DSA decode kernels (#50230) | paths:  | notes: NEW artifact because DSA decode kernels are not in L3 registry.

## NEW:dcp_lse_reduce  (1 events)
- L3-63ac04a61e 2026-08-10 PR #50484: [Kimi-K3] DCP support (#50484) | paths: vllm/model_executor/layers/attention/mla_attention.py; vllm/model_executor/layers/attention/sparse_mla_attention.py; vllm/v1/attention/backends/mla/flashinfer_mla.py; vllm/v1/attention/backends/mla/tokenspeed_mla.py | notes: NEW DCP LSE kernel not in registry.

## NEW:turboquant_rocm_decode  (1 events)
- L3-5426311d91 2026-08-11 PR #47896: [Kernel][ROCm][Perf] FlyDSL decode-attention kernel for 4-bit TurboQuant KV cache  (#47896) | paths: vllm/v1/attention/backends/turboquant_attn.py; vllm/v1/attention/ops/flydsl_kernels/__init__.py; vllm/v1/attention/ops/flydsl_kernels/tq_decode.py; vllm/v1/attention/ops/flydsl_kernels/tq_decode_gqa6.py | notes: NEW artifacts because TurboQuant attention is not registered in L3.

## NEW:rocm_sparse_indexer  (1 events)
- L3-466855a2bf 2026-08-12 PR #47017: [ROCm] Enable DeepSeek-V4 on gfx11 (#47017) | paths:  | notes: NEW sparse indexer adapter not in L3 registry.

## NEW:dsa_native_decode  (1 events)
- L3-63a9a5010a 2026-08-14 PR #52164: [Attention][DSA] Take the native decode path for MTP=3 on SM90 (#52164) | paths: vllm/v1/attention/backends/mla/indexer.py | notes: NEW DSA native decode path not in registry.

## NEW:dsv4_sparse_topk_metadata  (1 events)
- L3-836aac92ff 2026-08-16 PR #52084: [Perf][DSV4] Optimize sparse top-k metadata kernels for higher prefill throughput (#52084) | paths:  | notes: NEW metadata-kernel artifact not in registry.

## NEW:dsv4_sparse_topk_indexer  (1 events)
- L3-83f591d7f6 2026-08-16 PR #51967: [Perf][DSV4] Optimize global top-k index kernel with compile-time constants (#51967) | paths:  | notes: NEW artifact: DeepSeek V4 sparse-attention global top-k metadata kernel.

## NEW:jit_warmup_infrastructure  (1 events)
- L3-49905ad94d 2026-08-17 PR #50174: [3/N][Feat][Perf] Add new warmup infrastructure for JITs. Add provider registry and orchestration fo | paths:  | notes: NEW artifact: shared JIT warmup infrastructure.

## NEW:dcp_prefix_cache_block_table  (1 events)
- L3-0db502c8d8 2026-08-17 PR #50493: [Kimi-K3] support DCP partial prefix cache hit (#50493) | paths:  | notes: NEW artifact: DCP prefix-cache block-table geometry not registered.

## NEW:deepseek_v4_attention_input_prep  (1 events)
- L3-f1178f3a06 2026-08-18 PR #52836: Revert DSv4 eager workspace reuse (#52836) | paths: vllm/models/deepseek_v4/nvidia/flashmla.py | notes: NEW artifact: DeepSeek V4 attention input-preparation scratch/workspace path.

## NEW:sparse_mla_mask  (1 events)
- L3-8e46accab2 2026-08-18 PR #52217: [Attention] Vectorize sparse MLA mask loads (#52217) | paths: vllm/model_executor/layers/attention/sparse_mla_attention.py; vllm/model_executor/layers/attention/sparse_mla_mask.py | notes: NEW artifact: sparse MLA mask callback/path not separately registered.

## NEW:deepseek_v4_sparse_mla  (1 events)
- L3-e6f35d3c69 2026-08-21 PR #52823: [DSv4 Perf] Adaptive topk width for dsv4, making #50004 back (#52823) | paths:  | notes: NEW artifact: DeepSeek V4 sparse MLA C128A top-k metadata path.

## NEW:dcp_slot_mapping  (1 events)
- L3-0ecc284790 2026-08-24 PR #51031: [Bugfix][Kernel] Handle kernel block sizes in V2 DCP slot mapping (#51031) | paths:  | notes: NEW artifact: DCP slot mapping helper not registered.

## NEW:pcp_cuda_graph_manager  (1 events)
- L3-b1fbbc2ade 2026-08-26 PR #53515: BugFix(PCP): use persistent input buffers for PIECEWISE CUDA graphs (#53515) | paths:  | notes: NEW artifact: PCP CUDA graph manager not registered.

## NEW:hpc_rope_norm_attention_adapter  (1 events)
- L3-2267d3b112 2026-08-26 PR #53705: [AttentionBackend][HPC-ops] update hpc rope norm to support stride kv cache (#53705) | paths:  | notes: Thin PR body; kept as stage-1 positive but evidence is limited and artifact is not in registry.

## NEW:deepseek_v4_rocm_swa_qkv_norm  (1 events)
- L3-32ad1400d7 2026-08-27 PR #53540: [ROCm][Perf] Fuse SWA q/kv RMSNorm and q FP8 group quant for DeepSeek-V4 (#53540) | paths:  | notes: NEW artifact: ROCm DeepSeek V4 SWA q/kv normalization and quantization pipeline.

## NEW:hy_v4_flashmla_sparse_adapter  (1 events)
- L3-b2f685834a 2026-08-29 PR #54160: [Hy4] support Hy4-preview model (#54160) | paths: vllm/models/hy_v4/nvidia/flashmla_sparse.py | notes: NEW artifact because registry has FlashMLA sparse family but no HY V4 model-local sink adapter.

## NEW:sparse_attention_cooperative_topk  (1 events)
- L3-f5c3cc240b 2026-08-31 PR #53382: [Perf][Kernel] Tune cooperative topk for medium batch-sizes (#53382) | paths:  | notes: Retune changes cluster-size thresholds/limits, not kernel structure; NEW artifact not in L3 registry.

## NEW:b12x_paged_attention_backend  (1 events)
- L3-0d4ad47981 2026-09-01 PR #52017: [Kernel] Add B12X causal paged attention backend (#52017) | paths: vllm/v1/attention/backends/b12x.py; vllm/v1/attention/backends/registry.py; setup.py; vllm/utils/b12x.py | notes: NEW artifact because B12X backend is not in current L3 registry.

## NEW:qsa_indexer_prefill_decode  (1 events)
- L3-003e34341a 2026-09-02 PR #54513: [Qwen3.8-Flash-Next] Separate prefill and decode paths for QSA indexer (#54513) | paths:  | notes: NEW artifact from stage1; QSA indexer is model-specific and not in L3 registry.

## NEW:kimi_k3_mla_concat_cache  (1 events)
- L3-9509fc8ae6 2026-09-03 PR #54896: [Perf][Kimi-K3] Cut MLA decode concat/cache epilogue latency (#54896) | paths:  | notes: NEW artifact from stage1; model-specific Kimi-K3 fused epilogue is not in registry.

## NEW:fused_qk_rmsnorm  (1 events)
- L3-bc2ee48073 2026-09-03 PR #55020: [Perf] Prefetch the weight before the PDL wait in fused_q_kv_rmsnorm (#55020) | paths:  | notes: NEW artifact from stage1; fused q/kv RMSNorm is model attention-front-end kernel not in registry.

## NEW:deepseek_v4_dequant_gather_k_cache  (1 events)
- L3-69cf055936 2026-09-03 PR #55061: [Performance][DSv4] Size dequant gather launch grid by rows (#55061) | paths:  | notes: NEW artifact from stage1; DSv4 model-specific gather kernel not in registry.

## NEW:kimi_k3_amd_mla_wrapper  (1 events)
- L3-3ff4f02dfe 2026-09-04 PR #52494: [AMD][kimik3][ROCm][Perf] Fuse MLA q/kv RMSNorm in AMD Kimi-K3 MLA wrapper (#52494) | paths:  | notes: NEW wrapper noted because model-local AMD Kimi-K3 wrapper is not a registry artifact.

## NEW:deepseek_v4_sparse_index_workspace  (1 events)
- L3-7fbd44cbe0 2026-09-05 PR #55299: [Bugfix][DSv4] Seed the -1 sentinel in the prefill sparse index workspace (#55299) | paths:  | notes: NEW artifact from stage1; workspace helper is model-specific.

## NEW:dsv4_dcp_attention_ops  (1 events)
- L3-1713b9866a 2026-09-07 PR #53564: [2/N][warmup][DSv4] Migrate sequence and DCP kernels (#53564) | paths: vllm/v1/attention/ops/common.py; vllm/v1/attention/ops/dcp.py | notes: NEW artifact from stage1; warmup migration unit not in registry.

## NEW:fused_qknorm_rope  (1 events)
- L3-9c297e3b8b 2026-09-07 PR #55755: [Kernel] PDL enablement for fusedQKNormRopeKernel (#55755) | paths:  | notes: NEW artifact from stage1; fused QK norm/RoPE kernel not in registry.

## NEW:deepseek_v4_common_attention_kernels  (1 events)
- L3-6ddbab03de 2026-09-08 PR #50176: [4/N][warmup][DSv4] Migrate common attention kernels (#50176) | paths:  | notes: NEW artifact from stage1; common DSv4 warmup migration not in registry.

## NEW:minimax_m3_aiter_sparse_attention  (1 events)
- L3-83252ea899 2026-09-09 PR #52664: [Performance][ROCm]  Integrate aiter indexer scoring and top-k kernels into MiniMax-M3 sparse attent | paths:  | notes: NEW artifact because MiniMax-M3 AITER sparse indexer/attention is model-specific and absent from L3 registry.

## NEW:dcp_a2a_pack_lse_mask  (1 events)
- L3-6ee5bb0a0b 2026-09-10 PR #54889: [DCP][Kernel][Perf] Fuse the empty-shard LSE mask into the A2A pack kernel (#54889) | paths: vllm/v1/attention/ops/dcp.py | notes: 

## NEW:sparse_attn_indexer  (1 events)
- L3-980c16c8e4 2026-09-10 PR #56107: [PCP][Spec Decode] Adds PCP support for single-module MTP and replicated DSpark. (#56107) | paths:  | notes: 

## NEW:dsv4_cutedsl_attention_warmup  (1 events)
- L3-120ec4ebd2 2026-09-12 PR #53566: [5/N][warmup][DSv4] Migrate NVIDIA CuTeDSL attention kernels (#53566) | paths: vllm/v1/attention/backends/mla/compressor_utils.py; vllm/v1/attention/backends/mla/indexer.py; vllm/v1/attention/backends/mla/sparse_swa.py; vllm/v1/attention/backends/mla/sparse_utils.py | notes: 

## NEW:mla_metadata_triton  (1 events)
- L3-13e221f830 2026-09-12 PR #56562: [Perf] Fuse DSV4.1 input metadata preparation with Triton (#56562) | paths: vllm/v1/attention/backend.py; vllm/v1/attention/backends/mla/indexer.py; vllm/v1/attention/ops/metadata.py | notes: 

## NEW:dsa_indexer_topk  (1 events)
- L3-fa008bdccf 2026-09-13 PR #56464: [Perf][Kernel] Integrate DeepSelect TopK for the DSA sparse indexer (#56464) | paths: CMakeLists.txt; cmake/external_projects/deepselect.cmake; setup.py | notes: NEW DSA top-k artifact not in registry.

## NEW:mla_prefill_flash_attn  (1 events)
- L3-238cb2b191 2026-09-14 PR #55738: [Perf][GLM-5.3-Flash] Dense/masked-MHA sparse prefill for the NoPE (256, 0, 256) layout + skip the N | paths: vllm/model_executor/layers/attention/mla_attention.py; vllm/model_executor/layers/attention/sparse_mla_attention.py; vllm/v1/attention/backends/mla/prefill/flash_attn.py | notes: NEW MLA prefill FlashAttention artifact not in registry.

## NEW:dsv4_rocm_csa_attention  (1 events)
- L3-a6c5d6d0fc 2026-09-14 PR #51794: [ROCm][Perf] Enable CSA multi-stream overlap for DeepSeek-V4 (#51794) | paths:  | notes: NEW ROCm CSA attention artifact not in registry.

## NEW:dsa_sparse_mqa_logits  (1 events)
- L3-1257512305 2026-09-15 PR #56254: [DSA] Wire DeepGEMM sparse MQA logits into the DeepSeek V4.1 indexer (#56254) | paths: vllm/v1/attention/backends/mla/indexer.py; vllm/v1/attention/backends/mla/sparse_indexer.py | notes: NEW sparse MQA logits artifact not in registry.

## NEW:hisparse_config_dispatch  (1 events)
- L3-df42d112ee 2026-09-15 PR #57041: [Config] Infer HiSparse attention config from HiSparseConnector (#57041) | paths:  | notes: NEW HiSparse config dispatch artifact not in registry.

## NEW:flashmla_mega_attn_dsv41  (1 events)
- L3-d6a1677d55 2026-09-16 PR #56935: [Model][DSv4.1] FlashMLA mega attention and the NVFP4 compressed KV cache (#56935) | paths: vllm/models/deepseek_v41/nvidia/flashmla.py; vllm/v1/attention/backends/mla/sparse_swa.py; vllm/v1/attention/backends/registry.py | notes: NEW FlashMLA mega-attention artifact not in registry.
