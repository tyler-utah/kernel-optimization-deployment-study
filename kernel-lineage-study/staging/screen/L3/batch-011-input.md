### L3-03a49bb8f0  (L3, 2026-03-05, sha 03a49bb8f0c8, PR #36047)
TITLE: [Feature] Add --distributed-timeout-seconds CLI option (#36047)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/parallel.py (+7/-1); vllm/engine/arg_utils.py (+6/-0); vllm/v1/worker/gpu_worker.py (+12/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Add a `--distributed-timeout-seconds` CLI option that allows users to configure the timeout for `torch.distributed.init_process_group`. This is useful for multi-node setups where model downloads or node startup may be slow, causing PyTorch's default 600s NCCL timeout to expire prematurely. ⏎  ⏎ When set, the value is passed as a `timedelta` to `init_process_group`. When not set (`None`), PyTorch's default timeout is used. ⏎  ⏎ ## Test Plan ⏎  ⏎ - …[truncated]

### L3-ebed80a7c8  (L3, 2026-03-06, sha ebed80a7c8c6, PR #35384)
TITLE: [Performance] Extract KV-cache update from TreeAttention backend (#35384)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.tree_attention
FILES: vllm/v1/attention/backends/tree_attn.py (+28/-19)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎ - Separate the KV-cache write (`reshape_and_cache_flash`) from the attention `forward()` in the TreeAttention backend ⏎ - Part of #32335 — decoupling KV-cache updates from all attention backends ⏎  ⏎ ## Changes ⏎ - Set `forward_includes_kv_cache_update = False` on `TreeAttentionBackend` ⏎ - Add `do_kv_cache_update()` method to `TreeAttentionImpl` ⏎ - Remove KV cache write logic from `forward()` ⏎  ⏎ ## Benchmarks ⏎  ⏎ Hardware: 1x NVIDIA A100 80GB, vLLM v …[truncated]

### L3-807d680337  (L3, 2026-03-06, sha 807d6803376f, PR #35553)
TITLE: [ROCm][CI] Fix tool use test stability - disable skinny GEMM, prefix caching, eliminate batch variance (#35553)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+8/-0); docs/design/attention_backends.md (+1/-1); requirements/rocm-test.txt (+2/-0); tests/entrypoints/openai/test_completion_with_function_calling.py (+8/-16); tests/utils.py (+14/-0)
LABELS: documentation, rocm, ready, ci/build, v1
BODY: Fixes intermittent tool use test failures on ROCm by applying platform-specific overrides to `tests/entrypoints/openai/test_tool_calls.py`. ⏎  ⏎ ## Changes ⏎  ⏎ - Disable `VLLM_ROCM_USE_SKINNY_GEMM` (`=0`) via `env_dict` to avoid non-deterministic results from atomic reductions in the `wvSplitKrc` skinny GEMM kernel. ⏎ - Disable prefix caching (`--no-enable-prefix-caching`) and eliminate batch variance (`--max-num-seqs 1`) to reduce flakiness on ROCm. ⏎  ⏎ ##  …[truncated]

### L3-1a9718085c  (L3, 2026-03-06, sha 1a9718085c79, PR #36042)
TITLE: Fix CUDA graph decode capture crash in AITER FlashAttention (#36042)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+3/-4)
LABELS: rocm, ready, v1, nvidia, meta-exported, fb-exported
BODY: Summary: ⏎ Upstream vLLM commit vllm-project/vllm#32877 added speculative decoding ⏎ support to the ROCm AITER FA backend by routing `decode_max_query_len > 1` ⏎ through `unified_attention` (a Triton kernel). However, it also added ⏎ `self.sliding_window[0] != -1` to the same condition, which causes the ⏎ `unified_attention` path to be taken during CUDA graph decode FULL capture ⏎ even for non-speculative workloads. ⏎  ⏎ `unified_attention` is not CUDA-graph-cap …[truncated]

### L3-225d1090a0  (L3, 2026-03-06, sha 225d1090a099, PR #35253)
TITLE: Enabling some B200-specific tests on MI355 (#35253)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+71/-110); tests/evals/gsm8k/configs/Qwen3-Next-FP8-EP2_MI355.yaml (+9/-0); tests/evals/gsm8k/configs/models-mi355.txt (+5/-0); tests/kernels/attention/test_attention_selector.py (+4/-0)
LABELS: ready, ci/build
BODY: Enabling some B200-specific tests on MI355: ⏎  ⏎ 1. mi355_1: V1 Test attention (B200) ⏎ Technically, this testing has already been performed, but now we're separating MI355 and MI325 to track future changes in definitions of the  "V1 Test attention (B200)" & "V1 Test attention (H100)" TGs, respectively. ⏎  ⏎ 2. mi355_1: Blackwell Test (MI355) ⏎ Among many un-supported commands, this group has still two compatible & successful testing steps: ⏎ pytest -v -s test …[truncated]

### L3-c188749bcd  (L3, 2026-03-06, sha c188749bcdaa, PR #35850)
TITLE: [ROCm] Support MLA with nhead<16 and FP8 KV cache for TP=8 (Kimi K2.5/Linear) (#35850)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+19/-3)
LABELS: rocm, ready, v1
ISSUES: #35641 [Bug]: ROCm MI355X Kimi K2.5 AITER TP8 MLA kernel Error (num_head=8)
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎  ⏎ - **Support AITER MLA for num_heads < 16** (e.g., TP=8 with Kimi K2.5 giving 8 heads/rank, or Kimi-Linear giving 4 heads/rank). Uses head-repeat to expand to 16 heads before calling the AITER MLA decode kernel, then contracts the output back. This reuses the existing optimized gqa_ratio=16 ASM kernel without requiring new kernel variants. ⏎ - **Relax head-count assertion** to accept num_heads of 4, 8, or any multiple of 16 in [16, 128]. …[truncated]

### L3-24a03915f5  (L3, 2026-03-07, sha 24a03915f525, PR #36282)
TITLE: mla: don't update kv cache on dummy forwards (#36282)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+4/-0)
LABELS: ready, nvidia
BODY: ## Purpose ⏎ Before #34627, MLA only wrote KV inside forward_impl, after checking attn_metadata is not None. ⏎ With #34627, we started calling unified_mla_kv_cache_update unconditionally, so warmup/profile runs could still write into KV pages. ⏎  ⏎ This breaks prefix cache after elastic ep reconfigure, since it involves a dummy run which can now overwrite KV pages without invalidating the prefix-cache entries. ⏎  ⏎ This PR restores the original logic (skip K …[truncated]

### L3-fc4657756f  (L3, 2026-03-07, sha fc4657756ff0, PR #36174)
TITLE: [ROCm][CI] Enable AITER for failing `test_gpt_oss` test case on MI355 (#36174)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/models/quantization/test_gpt_oss.py (+8/-1)
LABELS: rocm, ready, gpt-oss
BODY: This test case is passing on MI325 but failing on MI350: ⏎ `pytest -v -s tests/models/quantization/test_gpt_oss.py::test_gpt_oss_attention_quantization[amd/gpt-oss-20b-MoE-Quant-W-MXFP4-A-FP8-KV-FP8-0.89-1]` ⏎  ⏎ It fails with the following error: ⏎ ``` ⏎ (EngineCore_DP0 pid=143860)   File "/projects/vllm/vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py", line 85, in _moe_forward ⏎ (EngineCore_DP0 pid=143860)     return layer.runner.forward_ …[truncated]

### L3-379689d533  (L3, 2026-03-07, sha 379689d53364, PR #35891)
TITLE: [Perf] Support FP8 KV cache for Flashinfer MLA Sparse (#35891)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_sparse, L3.mla.flashinfer_sparse
FILES: vllm/model_executor/layers/attention/mla_attention.py (+30/-5); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+7/-0); vllm/v1/attention/backends/mla/flashmla_sparse.py (+7/-0); docs/design/attention_backends.md (+1/-1); tests/v1/attention/test_mla_backends.py (+18/-2); tests/v1/attention/test_sparse_mla_backends.py (+11/-1); tools/pre_commit/generate_attention_backend_docs.py (+15/-1); vllm/model_executor/models/config.py (+0/-7)
LABELS: documentation, ready, v1, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ This PR enables fp8 kv cache for Flashinfer MLA Sparse attention backend (tracked by #35805). ⏎  ⏎ ## Test Plan ⏎ Added fp8 unit tests for flashinfer to `tests/v1/attention/test_sparse_mla_backends.py` ⏎  ⏎ ## Test Result ⏎ For deepseek v3.2 nvfp4, see **14% speedup** in throughput using flashinfer backend compared to flashMLA under DEP8 and concurrency 256. ⏎  ⏎ ``` ⏎ vllm serve nvidia/DeepSeek-V3.2-NVFP4 \ ⏎     --trust-remote-code \ ⏎     --max-num-seqs  …[truncated]

### L3-63298ee173  (L3, 2026-03-07, sha 63298ee17350, PR #35931)
TITLE: [Bugfix][LMCache][KVConnector] fix potential memory leak in LMCache multiprocess mode (#35931)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py (+28/-0)
LABELS: bug, ready, kv-connector
BODY: ## Purpose ⏎  ⏎   Fix a potential memory leak in LMCache multiprocess (MP) mode caused by lookup locks not being freed for chunks that vLLM computes itself rather than retrieving from LMCache. ⏎  ⏎   When a prefetch lookup returns hits but vLLM decides to compute some (or all) of those blocks itself, the corresponding LMCache chunk locks were never released. Over time, these leaked locks prevent LMCache from evicting stale entries, leading to unbounded m …[truncated]

### L3-5d6aae4577  (L3, 2026-03-07, sha 5d6aae457759, PR #35831)
TITLE: [LMCache MP Patch]: Race Condition + Duplicated Block Ids (#35831)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py (+23/-4)
LABELS: ready, kv-connector
BODY: @ApostaC this is accompanied by fix on LMCache side: https://github.com/LMCache/LMCache/pull/2671 ⏎ There are two bugs in `LMCacheMPConnector` that caused the MP correctness test to fail at high concurrency. ⏎  ⏎ 1. **APC race condition**: `GetRetrieveMetadata` computes `skip_first_n_tokens` from APC overlap so the LMCache server knows not to overwrite APC-shared blocks during retrieve. Without this, the server writes on its own CUDA stream while concu …[truncated]

### L3-0a6a3a1290  (L3, 2026-03-08, sha 0a6a3a12906b, PR #35986)
TITLE: Add support for ModelOpt MXFP8 MoE models (#35986)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_ocp_mx_moe.py (+185/-2); vllm/model_executor/layers/fused_moe/layer.py (+9/-0); vllm/model_executor/layers/fused_moe/oracle/mxfp8.py (+44/-0); vllm/model_executor/layers/quantization/modelopt.py (+359/-16)
LABELS: ready, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ Follow up to these PRs: ⏎ https://github.com/vllm-project/vllm/pull/33786 ⏎ https://github.com/vllm-project/vllm/pull/35053 ⏎  ⏎ vLLM recently upgraded flashinfer to version 0.6.4. ⏎  ⏎ Flashinfer 0.6.4 includes a new MXFP8 TRTLLM MoE kernel: ⏎ https://github.com/flashinfer-ai/flashinfer/pull/2505 ⏎  ⏎ This PR adds support for ModelOpt MXFP8 MoE models using the new kernel. ⏎  ⏎ At this time only gated MoE is supported. ⏎  ⏎ ## Test Plan ⏎  ⏎ Use the following MoE …[truncated]

### L3-747431044d  (L3, 2026-03-08, sha 747431044df6, PR #36263)
TITLE: feat(attention): extract KV-cache update from FlexAttention backend (#36263)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+25/-11)
LABELS: v1
BODY: ## Summary                                                                                                                                                                                   ⏎                                                                  ⏎   Extracts KV-cache update from FlexAttention, one of the last backends still doing it inline. Part of #32335.                                                                                 ⏎       …[truncated]

### L3-b0906d8b02  (L3, 2026-03-09, sha b0906d8b0268, PR #36472)
TITLE: [MM Encoder] Default to use TORCH_SDPA backend for ViT on Volta/Turing GPU (#36472)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: tests/kernels/attention/test_mha_attn.py (+15/-0); vllm/platforms/cuda.py (+15/-7)
LABELS: ready, nvidia
ISSUES: #36357 [Bug]: Multimodal encoder memory profiling hangs indefinitely on V100 (SM 7.0) in v0.17
BODY: ## Purpose ⏎ - Fix #36357 ⏎ - Let's investigate whether we should intergrate trtllm prefill kernel in the future ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest -s -v tests/kernels/attention/test_mha_attn.py::test_mha_attn_platform ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Tests should pass ⏎  ⏎ --- ⏎ [details omitted]

### L3-77a73458e3  (L3, 2026-03-09, sha 77a73458e3ae, PR #35122)
TITLE: Reapply [Attention] Refactor `check_and_update_config` (#35122)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashinfer_sparse, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/model_executor/layers/attention/attention.py (+0/-3); vllm/model_executor/layers/attention/chunked_local_attention.py (+6/-9); vllm/model_executor/layers/attention/cross_attention.py (+0/-3); vllm/model_executor/layers/attention/encoder_only_attention.py (+0/-3); vllm/model_executor/layers/attention/mla_attention.py (+12/-8); vllm/model_executor/layers/attention/static_sink_attention.py (+3/-8); vllm/v1/attention/backend.py (+14/-9); vllm/v1/attention/backends/mla/flashattn_mla.py (+1/-1); vllm/v1/attention/backends/mla/flashinfer_mla.py (+1/-1); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+1/-1); (+22 more)
LABELS: rocm, speculative-decoding, ready, v1, multi-modality, cpu, kv-connector, nvidia, ready-run-all-tests
DEEP_STUDY: deep-study revert record: reland of PR(s) 33600 reason=other
BODY: **Recommend marking with `ready-run-all-tests`** ⏎  ⏎ ## Purpose ⏎ Reapplies #33600 with fixes from #34818 and #34970, plus some additional cleanup ⏎  ⏎ `check_and_update_config` unnecessarily duplicates much of the logic from the attention selector in order to set an approproate block size. This PR refactors `check_and_update_config` to use the selector, which will be simpler to maintain going forward. ⏎  ⏎ * If the user specifies `--block-size` and `--attent …[truncated]

### L3-2b28b9b269  (L3, 2026-03-09, sha 2b28b9b269e1, PR #35290)
TITLE: [Attention][Perf] Optimize cp_gather_and_upconvert_fp8_kv_cache - DeepSeek-v3.2 (#35290)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+62/-69); benchmarks/kernels/bench_cp_gather_fp8.py (+153/-0); tests/kernels/test_cp_gather_fp8.py (+363/-0)
LABELS: performance, ready, deepseek, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ``` ⏎ ====================================================================== ⏎ CP_GATHER_AND_UPCONVERT_FP8_KV_CACHE BENCHMARKS ⏎ ====================================================================== ⏎ Cache entry: 656 bytes (512 FP8 + 16 scales + 128 RoPE) ⏎ Output row:  576 BF16 = 1152 bytes ⏎ Per token:   1808 bytes (read + write) ⏎ Block size:  64 tokens/block ⏎ ====================================================================== ⏎  ⏎ --- 1-req: 1 request(s),  …[truncated]

### L3-580864d81e  (L3, 2026-03-09, sha 580864d81eb0, PR #34917)
TITLE: [Attention][Perf][Kernel] Replace torch.cat with vectorized CUDA kernel MLA query concat - DeepSeek-V3.2 (#34917)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.mla.flashmla_sparse
FILES: csrc/cache_kernels.cu (+41/-0); csrc/torch_bindings.cpp (+4/-0); vllm/_custom_ops.py (+15/-0); vllm/v1/attention/backends/mla/flashmla_sparse.py (+13/-4); .buildkite/test_areas/kernels.yaml (+2/-1); benchmarks/kernels/bench_concat_mla_q.py (+98/-0); csrc/cache.h (+6/-0); csrc/concat_mla_q.cuh (+60/-0); csrc/cuda_vec_utils.cuh (+37/-10); tests/kernels/test_concat_mla_q.py (+139/-0)
LABELS: performance, ready, ci/build, v1, deepseek, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: PR #35105 needs to be merged first ⏎  ⏎ ## Perf eval on B300 ⏎  ⏎ Speedup improves with the number of tokens, peaking at `~11.8×` w.r.t. `torch.cat`. ⏎  ⏎ ``` ⏎ --- Non-contiguous nope inputs (transposed BMM output) --- ⏎ concat_mla_q-transposed: ⏎     num_tokens  torch.cat (Latency (us))  concat_mla_q (v8) (Latency (us)) ⏎ 1          8.0                  5.055390                          1.892126 ⏎ 2         16.0                  7.916890                          2.0 …[truncated]

### L3-70485a11bd  (L3, 2026-03-09, sha 70485a11bd83, PR #36253)
TITLE: [ROCM] Optimize the fused_topk_bias to use aiter instead of fallback torch ops. (#36253)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py (+38/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ When running MiniMax M2.5, currently the topk uses fallback pytorch ops, which is really slow. ⏎ <img width="1883" height="815" alt="image" src="https://github.com/user-attachments/assets/1678ebff-f2c8-40d2-b38f-f1dcb09caf1f" /> ⏎ This PR add a fast path for models like MiniMax M2.5 on ROCm by leveraging aiter's `biased_grouped_topk` kernel when sigmoid scoring with bias correction is used in `fused_topk_bias`. ⏎  ⏎ The aiter kernel has a har …[truncated]

### L3-c174d54f86  (L3, 2026-03-09, sha c174d54f86aa, PR #36292)
TITLE: [ROCm][CI] Fix ROCm attention backend validation for head sizes, block sizes, and compute capability checks (#36292)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_unified, L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+4/-0); vllm/v1/attention/backends/mla/triton_mla.py (+15/-0); vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+6/-0); vllm/v1/attention/backends/rocm_attn.py (+6/-0); vllm/v1/attention/backends/triton_attn.py (+6/-0); docs/design/attention_backends.md (+1/-1); tests/v1/attention/test_rocm_attention_backends_selection.py (+17/-8)
LABELS: documentation, rocm, ready, v1
BODY: This PR fixes several issues in the ROCm attention backend selection tests and underlying validation logic introduced by the move to validation-based backend selection in #35246 and refined in #36185. ⏎  ⏎ MLA backends (TritonMLA, AiterMLA, AiterTritonMLA) were incorrectly rejecting common head sizes like 128 because they inherited a restrictive head size list from the base MLA class. These backends now correctly accept any head size. The AITER FA co …[truncated]

### L3-4ff9b045fe  (L3, 2026-03-09, sha 4ff9b045fe7a, PR #36025)
TITLE: [ROCm][CI] Prep Tests For Change To ROCM_ATTN As New Default Backend On ROCm (#36025)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/lm-eval-harness/test_lm_eval_correctness.py (+8/-2); .buildkite/scripts/scheduled_integration_test/qwen3_next_mtp_async_eplb.sh (+1/-1); .buildkite/test-amd.yaml (+2/-2); tests/entrypoints/openai/test_tensorizer_entrypoint.py (+3/-0); tests/models/language/pooling_mteb_test/test_gte.py (+7/-1); tests/models/multimodal/generation/test_common.py (+3/-0); tests/test_regression.py (+3/-1); tests/v1/e2e/test_kv_sharing_fast_prefill.py (+1/-0); tests/v1/e2e/test_spec_decode.py (+3/-0); tests/v1/sample/test_logprobs.py (+1/-3)
LABELS: rocm, ready, ci/build, v1, multi-modality
BODY: This PR is a prerequisite to https://github.com/vllm-project/vllm/pull/33271. In this PR, we fix some tests to use TRITON_ATTN to prepare our CI for the switch to ROCM_ATTN as the default attention backend for AMD hardware. The change to using ROCM_ATTN as the default is motivated by the overall performance benefits over TRITON_ATTN. ⏎  ⏎ The LM Eval test required two changes. First of all, the accuracy score in this test was eventually compared agai …[truncated]

### L3-483463f735  (L3, 2026-03-09, sha 483463f735c4, PR #35959)
TITLE: [MRV2] Extensible CG dispatch rework  (#35959)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/compilation.py (+3/-0); vllm/v1/worker/gpu/block_table.py (+24/-8); vllm/v1/worker/gpu/cudagraph_utils.py (+285/-316); vllm/v1/worker/gpu/dp_utils.py (+53/-49); vllm/v1/worker/gpu/input_batch.py (+4/-1); vllm/v1/worker/gpu/model_runner.py (+76/-57); vllm/v1/worker/gpu/model_states/default.py (+5/-2); vllm/v1/worker/gpu/spec_decode/eagle/cudagraph.py (+52/-175); vllm/v1/worker/gpu/spec_decode/eagle/speculator.py (+43/-28)
LABELS: ready, v1, nvidia
BODY: ## Design ⏎  ⏎ In order to cleanly support full cudagraphs, the "shape" of the batch must match what was captured. This is primarily due to increased use of ahead-of-time schedulers in attention backends that create scheduling metadata as a function of num_tokens, num_reqs, etc., and with LoRA where the number of active adapters affects graph compatibility. This means the approach of mapping from token count to a specific graph is less ideal as it do …[truncated]

### L3-006aea17d7  (L3, 2026-03-09, sha 006aea17d7de, PR #36553)
TITLE: [BugFix] Remove incorrect assert in split_decodes_and_prefills (#36553)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+0/-1)
LABELS: bug, ready, v1
BODY: The assert statement does not consider the `query_lens == 0` case.

### L3-d0cd736caa  (L3, 2026-03-09, sha d0cd736caada, PR #36557)
TITLE: [Bugfix] Fix `RuntimeError: Already borrowed` that degrades VLM serving throughput under concurrent load. (#36557)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/renderers/base.py (+9/-1)
LABELS: bug, ready
BODY: ### Root cause ⏎  ⏎ `BaseRenderer.__init__` passes the same HF tokenizer instance to both the `AsyncMicrobatchTokenizer` (API-side tokenization) and the multimodal processor (via `create_processor`). These run on different threads: ⏎  ⏎ ``` ⏎ Request A (main thread, Step 3 — MM processing): ⏎   process_for_engine → _process_multimodal → mm_processor.apply ⏎     → call_hf_processor → hf_processor() → tokenizer.encode() ⏎     → _encode_plus → set_truncation_and_pa …[truncated]

### L3-156e33553c  (L3, 2026-03-09, sha 156e33553ccd, PR #36494)
TITLE: Fix: Re-Enable EP for trtllm MoE FP8 backend (#36494)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+0/-6)
LABELS: ready, nvidia
BODY: ## Purpose ⏎ Remove check to verify EP is not enabled when using trtllm MoE backend for FP8, as it works. ⏎   ⏎ ## Test Plan ⏎ Verify `lm_eval` on https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 with EP enabled on 4xB200. ⏎  ⏎ ## Test Result ⏎ Flashinfer 0.6.4 installed, so fp8 default to trtllm MoE backend ⏎ MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 ⏎  ⏎ Eval command: ⏎ ``` ⏎ lm_eval \ ⏎   --model local-completions \ ⏎   --model_args base_url=http:// …[truncated]

### L3-4ff8c3c8f9  (L3, 2026-03-10, sha 4ff8c3c8f9ec, PR #35219)
TITLE: [BUGFIX][Mamba][Qwen3.5] Zero freed SSM cache blocks on GPU (#35219)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+20/-0); vllm/utils/math_utils.py (+5/-0); vllm/v1/core/kv_cache_manager.py (+7/-0); vllm/v1/core/sched/output.py (+5/-0); vllm/v1/core/sched/scheduler.py (+10/-8); vllm/v1/core/single_type_kv_cache_manager.py (+11/-0); vllm/v1/kv_cache_interface.py (+8/-0); vllm/v1/worker/gpu_model_runner.py (+27/-0); vllm/v1/worker/gpu_worker.py (+8/-0); vllm/v1/worker/utils.py (+186/-0)
LABELS: bug, ready, v1, qwen, ready-run-all-tests
ISSUES: #35138 [Bug]: Qwen/Qwen3.5-397B-A17B-FP8 and Qwen/Qwen3.5-397B-A17B has accuracy issues when running with Flashinfer Attention backend on Blackwell.
BODY: ### Essential problem ⏎  ⏎ Fixes https://github.com/vllm-project/vllm/issues/35138 ⏎ Workaround for https://github.com/Dao-AILab/flash-attention/issues/1974 ⏎  ⏎ Hybrid models (e.g. Qwen3.5-397B-A17B) share a unified block pool between attention (fp8/fp16) and Mamba/SSM (fp32) layers. When a block previously used by Mamba (fp32 state) is reallocated to an attention layer with a smaller dtype, leftover fp32 bit patterns can appear as NaN/Inf in the new dtyp …[truncated]

### L3-721ae79f50  (L3, 2026-03-10, sha 721ae79f50c5, PR #34304)
TITLE: Improvements to wvSplitKrc skinny GEMM solution (#34304)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/rocm/skinny_gemms.cu (+150/-84); tests/kernels/quantization/test_rocm_skinny_gemms.py (+7/-4); vllm/model_executor/layers/utils.py (+11/-9)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: This checkin adjusts wvSplitKrc to use a static reduce buffer that each kernel call cleans back to zero after use. This avoids the need for a .zero_() call, improving perf. ⏎ Additionally the header has been adjusted to match torch.linear() and avoid need for reshape() calls. ⏎ In addition activation padding support is introduced, and test coverage added. ⏎  ⏎ **Tok/s Performance on MI350** ⏎ RUN: ⏎ vllm bench serve --model openai/gpt-oss-120b \ ⏎     --percen …[truncated]

### L3-9095cbbfb6  (L3, 2026-03-10, sha 9095cbbfb6f6, PR #36519)
TITLE: [Bugfix][Sparse MLA] report indexer CG support properly (#36519)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+15/-3)
LABELS: bug, ready, v1
DEEP_STUDY: deep-study: this PR was reverted by PR 38076 (confirmed_revert, reason=unstated)
BODY: ## Purpose ⏎ Right now, the indexer reports its CG support as `UNIFORM_BATCH`, but this is only true when DeepGEMM is available. The torch fallback, `fp8_paged_mqa_logits_torch`, is not cudagraph compatible. This causes errors like https://github.com/vllm-project/vllm/pull/30515#issuecomment-4023776086 ⏎  ⏎ This PR reports the CG support correctly. ⏎  ⏎ ## Test Plan ⏎ Without DeepGEMM installed, ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek-V3.2 ⏎ ``` ⏎  ⏎ ## Test Result ⏎ ## …[truncated]

### L3-82f3f30e26  (L3, 2026-03-10, sha 82f3f30e266e, PR #35719)
TITLE: [ROCm][Perf] Enable `sparse_mla`'s cudagraph on ROCm platform (#35719)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+3/-1); vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+0/-3)
LABELS: rocm, ready, v1, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ This PR re-enabled cudagraph for models that adopt sparse_mla backend on ROCm platform since the fixed PR in aiter have updated in newest nightly docker. ⏎  ⏎ ## Test Plan ⏎ gsm8k ⏎ ## Test Result ⏎  ⏎ gsm8k with 20shot ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|    20|exact_match|↑  |0.9522|±  |0.0059| ⏎ |     …[truncated]

### L3-195d1ca3e8  (L3, 2026-03-10, sha 195d1ca3e8b1, PR #36609)
TITLE: [Minor] Enhance error message for TRTLLM decode uniformity check (#36609)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+2/-1)
LABELS: ready, v1, nvidia
BODY: 

### L3-5f77ef15ae  (L3, 2026-03-10, sha 5f77ef15aedc, PR #36673)
TITLE: [Misc][Attention] Clean up unused method in `CPU_ATTN` (#36673)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+0/-4)
LABELS: ready, v1, cpu
BODY: ## Purpose ⏎ `get_supported_dtypes` is dead code, not called anywhere within vLLM. The attention selector refers to the ClassVar `supported_dtypes`, which is properly implemented. ⏎  ⏎ ## Test Plan ⏎ CI ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-76c6e6da08  (L3, 2026-03-10, sha 76c6e6da08db, PR #36458)
TITLE: [XPU] Support block fp8 moe by fallback to TritonExpert on XPU (#36458)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/fused_moe.py (+5/-3); vllm/model_executor/layers/fused_moe/oracle/fp8.py (+5/-0)
LABELS: ready
BODY: ## Purpose ⏎ Support block fp8 moe by fallback to TritonExpert on XPU ⏎  ⏎ ## Test Plan ⏎ python3 examples/offline_inference/basic/generate.py --model Qwen/Qwen3-30B-A3B-FP8  --enforce-eager ⏎  ⏎ ## Test Result ⏎ output make sense. ⏎  ⏎ --- ⏎ [details omitted]

### L3-8ab3d7427c  (L3, 2026-03-11, sha 8ab3d7427cf5, PR #36691)
TITLE: [Bugfix] Fix DeepSeek V3.2 OOM during CG memory profiling (#36691)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+5/-7)
LABELS: bug, ready, v1, deepseek
BODY: ## Purpose ⏎ The cudagraph memory profiler added in #30515 did not account for `UniformTypeKVCacheSpecs` in `init_minimal_kv_cache_for_profiling`, so the `page_size` was being improperly multiplied by the `group_size`, causing an allocation that was 61x too large. This PR fixes this and takes advantage of the existing `num_blocks` override mechanism instead of spoofing the available memory, so it should be more robust. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve  …[truncated]

### L3-9d07a3d6e4  (L3, 2026-03-11, sha 9d07a3d6e472, PR #36658)
TITLE: Add: Eagle3 support for Qwen3.5 (#36658)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_5.py (+11/-0); vllm/model_executor/models/qwen3_next.py (+14/-2)
LABELS: ready, qwen
BODY: This PR adds support for EAGLE-3 speculative decoding to `Qwen3.5`, enabling faster inference with draft models like `BLR2/Qwen3.5-9B-Eagle3-ShareGPT`. ⏎  ⏎ ## Changes ⏎  ⏎ ### Modified Files ⏎ - `vllm/model_executor/models/qwen3_next.py` ⏎ - `vllm/model_executor/models/qwen3_5.py` ⏎  ⏎ ### Implementation Details ⏎  ⏎ 1. **Updated `Qwen3NextModel` (`qwen3_next.py`)** ⏎    - Added `aux_hidden_state_layers` attribute to track which layers output auxiliary hidden states ⏎  …[truncated]

### L3-5353c9b016  (L3, 2026-03-11, sha 5353c9b01605, PR #36665)
TITLE: platforms: Fix Ray DP startup crash (#36665)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/platforms/interface.py (+5/-0)
LABELS: ready
BODY: ## Purpose ⏎ PR #35122 broke Ray DP startup because ⏎ `self.collective_rpc(_update_block_size)` pickles the nested callback. ⏎  ⏎ When pickle serializes that callback, it checks dunder methods like ⏎ `__getstate__` on `current_platform`. `Platform` was returning `None` for ⏎ missing dunder methods, which pickle treats like a valid value, so it tried ⏎ to call `None` and crashed with `'NoneType' object is not callable`. ⏎  ⏎ Raise `AttributeError` for missing dunde …[truncated]

### L3-9c34e9d24f  (L3, 2026-03-11, sha 9c34e9d24fcd, PR #36318)
TITLE: Disable cascade attention by default (#36318)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/model.py (+5/-4)
LABELS: ready
BODY: ### Motivation ⏎ - Make cascade attention opt-in to avoid potential numerical issues by default, requiring users to explicitly enable it. ⏎  ⏎ ### Description ⏎ - Change `ModelConfig.disable_cascade_attn` default from `False` to `True` in `vllm/config/model.py`. ⏎ - Update the `disable_cascade_attn` docstring to explain the new opt-in behavior and note that heuristics still gate usage. ⏎ - The change is limited to the config default and documentation; runtim …[truncated]

### L3-fa0d353acf  (L3, 2026-03-11, sha fa0d353acfa3, PR #35194)
TITLE: [Bugfix] Surface exceptions from non-blocking execute_model in UniProcExecutor to avoid DP deadlocks (#35194)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/engine/core.py (+4/-3); vllm/v1/executor/uniproc_executor.py (+6/-1)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ Fixes issue #35193 by preventing distributed-parallel (DP) hangs. Exceptions raised during async model execution are now surfaced immediately instead of being silently stuck in a `Future`. ⏎  ⏎ Changes in `vllm/v1/executor/uniproc_executor.py` when `non_block=True`: ⏎  ⏎ - Calls `res.exception(timeout=0)` to check if the future has already failed. ⏎ - If the future isn’t done, `TimeoutError` is ignored and the future is returned as before. ⏎ - If …[truncated]

### L3-eac2dc2b41  (L3, 2026-03-11, sha eac2dc2b410d, PR #35765)
TITLE: AITER MLA backend: Avoid CPU sync in _build_decode (#35765)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+46/-15)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: This uses the same triton kernel as flashinfer.py to prevent a CPU sync when preparing the page_indices buffer. ⏎  ⏎ NOTE: This is my first time working on this type of code, and I'd appreciate thorough reviews. ⏎  ⏎ ## Purpose ⏎ Fix a long sync when running _build_decode. This was found during profiling. ⏎  ⏎ Before the change: ⏎ <img width="1165" height="217" alt="image" src="https://github.com/user-attachments/assets/38b87662-3d1d-4d46-8984-a3a447d1dbde" /> ⏎  ⏎  …[truncated]

### L3-a40ee486f2  (L3, 2026-03-11, sha a40ee486f273, PR #35923)
TITLE: [Bugfix] Add Multiple of 16 block_size to triton fallback on rocm Attention to support qwen3_5 (#35923)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+10/-22); docs/design/attention_backends.md (+1/-1)
LABELS: bug, documentation, rocm, ready, v1, qwen
BODY: This PR adds  multiple of 16 to the list of supported kernel block sizes in RocmAttentionBackend ⏎  ⏎ When running Qwen3.5 models using the ROCM_ATTN backend, the model produces broken, nonsensical outputs (e.g., repeating exclamation marks like !!!!!!!!!!). This happens because Qwen3.5 utilizes a non-standard block size of 1056. Since this size was not explicitly permitted, the model failed to correctly route the value_cache through the optimized Tr …[truncated]

### L3-545d18d81b  (L3, 2026-03-11, sha 545d18d81bf1, PR #36321)
TITLE: [Bugfix] Support other quantization methods in glm41v (#36321)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/glm4_1v.py (+6/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎ - In glm4.1v, If we use other quantization methods, there will be unsupported cases in '{prefix}.qkv_proj' . ⏎  ⏎ ## Test Plan ⏎ - We used the 'Ascend/msit' quantization method to test the w8a8 weights. ⏎  ⏎ ## Test Result ⏎ - Successfully ran on NPU using vllm-ascend by the w8a8 weights. ⏎  ⏎ --- ⏎ [details omitted]

### L3-a5d06dc557  (L3, 2026-03-11, sha a5d06dc557f9, PR #36161)
TITLE: Add 320 dimension size support to MLA (#36161)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.mla.common_v1
FILES: csrc/cache_kernels.cu (+19/-6); vllm/model_executor/layers/attention/mla_attention.py (+1/-1); tests/kernels/attention/test_cache.py (+5/-2)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ This PR adds the dimension 320 support to MLA. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-5573894737  (L3, 2026-03-11, sha 557389473755, PR #36361)
TITLE: Kimi k2.5 MLA based eagle3 (#36361)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/speculative.py (+4/-0); vllm/model_executor/models/deepseek_eagle3.py (+419/-0); vllm/model_executor/models/deepseek_v2.py (+39/-6); tests/models/registry.py (+12/-0); vllm/model_executor/models/kimi_k25.py (+14/-1); vllm/model_executor/models/registry.py (+2/-0); vllm/transformers_utils/config.py (+1/-0); vllm/v1/spec_decode/eagle.py (+8/-1)
LABELS: new-model, speculative-decoding, ready, v1, deepseek
BODY: ## Purpose ⏎ @IzzyPutterman is original author. ⏎  ⏎ This allows for Eagles that share MLA instead of GQA for attention, so one can train Eagle3s for Kimi and Deepseek and use them across TRTLLM, SGL, and vLLM.  ⏎  ⏎ ## Test Plan ⏎  ⏎ Acc benchmark: ⏎ ``` ⏎ lm_eval \ ⏎   --model local-completions \ ⏎   --model_args base_url=http://my_server:8001/v1/completions,model=/trt_llm_ci/data/llm-models/Kimi-K2.5-NVFP4,num_concurrent=16,tokenized_requests=False,trust_remote_cod …[truncated]

### L3-9040cd40af  (L3, 2026-03-11, sha 9040cd40af6b, PR #36723)
TITLE: [DSV3.2][MTP] Optimize Indexer MTP handling (#36723)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+7/-3)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎  ⏎ Usage of `repeat_interleave` in MLA indexer MTP batch expansion logic causes gpu/cpu synchronization which breaks async scheduling, causing massive performance degredation. ~When the requests in the batch have uniform query length (common case for speculative decoding) the shapes are consistent and we can use a nonblocking implementation.~ ⏎  ⏎ Alternative implementation suggested by @MatthewBonanni is to pass output_size to repeat_interl …[truncated]

### L3-82b110d50e  (L3, 2026-03-11, sha 82b110d50ee0, PR #36719)
TITLE: [ci] Bound nvidia-cudnn-frontend version (#36719)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: requirements/cuda.txt (+3/-0)
LABELS: ready, ci/build, nvidia
BODY: `nvidia-cudnn-frontend` seems to only look for CUDA 13 shared object files and it breaks for CUDA 12.9 environment in CI.

### L3-428bc718bd  (L3, 2026-03-11, sha 428bc718bd4a, PR #36274)
TITLE: [Bugfix][ROCm] Strip block_size before attention backend validation (#36274)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+2/-0)
LABELS: bug, rocm, ready
BODY: ## Purpose ⏎ The ROCm attention backend refactor (#35246) introduced validate_configuration calls that reject irregular block_size because it is not in the BlockSize type (1, 8, 16, 32, 64, 128, 256). The CUDA platform avoids this by stripping block_size before validation. Apply the same fix to the ROCm platform. ⏎  ⏎ ## Test Plan ⏎ Can start Qwen3 Next correctly ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-a9e532afe2  (L3, 2026-03-11, sha a9e532afe2a1, PR #36681)
TITLE: [ROCm][Perf] Allow MTP lens > 1 in Sparse MLA (#36681)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/spec_decode/eagle.py (+4/-0)
LABELS: rocm, speculative-decoding, ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ This PR adds ROCMAiterMLASparseMetadata to the list of allowed attention metadatas for MTP lens > 1. Attention backends where whitelisted 2 months ago so every new one must be added to the list. ⏎  ⏎ This PR does not guarantee MTP functionality on every possible case with sparse MLA, but it allows the user to enable it if found working on their case. ⏎  ⏎ ## Test Plan ⏎ gsm8k on DS3.2 ⏎ ## Test Result ⏎ With 3 speculated tokens: ⏎ |Tasks|Version|      …[truncated]

### L3-afebeffbfb  (L3, 2026-03-11, sha afebeffbfbf2, PR #36163)
TITLE: Add support to Mistral large 3 eagle with dense layers (#36163)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/mistral_large_3_eagle.py (+5/-1); vllm/transformers_utils/configs/mistral.py (+23/-0)
LABELS: speculative-decoding, ready
BODY: ## Purpose ⏎  ⏎ This PR adds support to Dense layers for Mistral Large 3 eagle. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-a3ea760ea5  (L3, 2026-03-11, sha a3ea760ea59a, PR #36238)
TITLE: Add 'none' reasoning effort to ChatCompletionRequest (#36238)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/entrypoints/openai/chat_completion/protocol.py (+8/-1); vllm/entrypoints/openai/chat_completion/serving.py (+3/-1); vllm/entrypoints/serve/render/serving.py (+3/-0)
LABELS: frontend, ready
BODY: ## Purpose ⏎  ⏎ This PR adds support to 'none' reasoning effort. ⏎  ⏎ This is something that is actually supported by the [OpenAI client](https://github.com/openai/openai-python/blob/main/src/openai/types/shared/reasoning_effort.py#L8). ⏎  ⏎ It is added to the ChatCompletionRequest but not to harmony.

### L3-e584dce52b  (L3, 2026-03-11, sha e584dce52b95, PR #33230)
TITLE: Add XPU MLA Sparse backend for DeepSeek v3.2 (#33230)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.dispatch.registry
FILES: vllm/platforms/xpu.py (+2/-1); vllm/v1/attention/backends/mla/xpu_mla_sparse.py (+257/-0); vllm/v1/attention/backends/registry.py (+1/-0); vllm/v1/attention/ops/xpu_mla_sparse.py (+265/-0); docs/design/attention_backends.md (+1/-0); tests/kernels/attention/test_xpu_mla_sparse.py (+118/-0); vllm/_xpu_ops.py (+245/-0); vllm/model_executor/layers/sparse_attn_indexer.py (+47/-22); vllm/triton_utils/__init__.py (+4/-1)
LABELS: documentation, ready, v1, deepseek
BODY: ## Purpose ⏎  ⏎ This is to add Sparse MLA backend for DeepSeek V3.2 on XPU device ⏎  ⏎ ## Test Plan ⏎  ⏎ Added op test for triton based sparse mla implementation. ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-12001f2ebc  (L3, 2026-03-11, sha 12001f2ebc60, PR #36129)
TITLE: [LMCache] Pass TP size in lookup for MLA multi-reader locking (#36129)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/multi_process_adapter.py (+6/-0); vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py (+16/-0)
LABELS: ready, kv-connector
BODY: ## Summary ⏎  ⏎ When MLA is enabled, the world_size passed to LMCache is divided by tp_size (since all TP ranks share the same KV cache in MLA models). However, the read lock count during lookup still needs the original TP size to acquire the correct number of locks for all workers that will independently retrieve the same cached chunks. ⏎  ⏎ This PR adds a tp_size parameter to LMCacheMPSchedulerAdapter and propagates it through IPCCacheEngineKey so the  …[truncated]

### L3-24062b704f  (L3, 2026-03-11, sha 24062b704fea, PR #36499)
TITLE: [ROCm][CI/Build] Add gfx1152/gfx1153 (Krackan) to HIP supported architectures (#36499)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1)
LABELS: rocm, ready, ci/build
BODY: Without these entries, the CMake build silently drops gfx1152/gfx1153 from PYTORCH_ROCM_ARCH even though they are valid RDNA 3.5 targets, resulting in wheels that lack kernel images for these GPUs. ⏎  ⏎ I have a validated a local vLLM build with this fix on gfx1153.

### L3-3e64fe4a18  (L3, 2026-03-12, sha 3e64fe4a183a, PR #36599)
TITLE: [Bugfix] Warm up Triton autotuner for GDN layers during V1 profiling (#36599)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_next.py (+98/-1)
LABELS: bug, ready, qwen
BODY: ## Purpose ⏎  ⏎ Fix Triton autotuner OOM for Qwen3.5 / Qwen3-Next models with Gated Delta Net (GDN) linear attention layers. ⏎  ⏎ As is mentioned in #36598, during V1 profile runs, `_forward_core` in `Qwen3NextGatedDeltaNet` returns early when `attn_metadata is None`, so the Triton-autotuned kernels used by GDN (`solve_tril`, `chunk_scaled_dot_kkt`, `chunk_gated_delta_rule_fwd_h`, `chunk_fwd_o`) are never invoked. After profiling, vLLM allocates KV cache …[truncated]

### L3-f0d3658c0f  (L3, 2026-03-12, sha f0d3658c0f10, PR #36605)
TITLE: [MM][OOT] Support CPU `seq_lens` for OOT MMEncoderAttention kernels (#36605)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/mm_encoder_attention.py (+33/-18); tests/kernels/attention/test_mha_attn.py (+6/-9); vllm/model_executor/custom_op.py (+6/-0); vllm/model_executor/models/qwen3_omni_moe_thinker.py (+4/-6); vllm/model_executor/models/qwen3_vl.py (+3/-7)
LABELS: ready, qwen
BODY: ## Purpose ⏎ PR https://github.com/vllm-project/vllm/pull/34580 has introduced some interfaces for `MMEncoderAttention` to customize the preparation for variables needed by the kernel, such as `cu_seqlens`. ⏎  ⏎ In this PR, we want to support OOT customization for these interfaces. To achieve this, we need to get OOT class first then call the class method like `oot_class.classmethod_name()`, since `CustomOp` only replace the instance (object) for the o …[truncated]

### L3-2ef69456f5  (L3, 2026-03-12, sha 2ef69456f5a0, PR #36586)
TITLE: [LMCache] Fault Tolerance Mechanism (#36586)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py (+38/-6)
LABELS: ready, kv-connector
BODY: ## Purpose ⏎  ⏎ Add Fault Tolerance Mechanism for vllm in case LMCache server crashes. Need to be merged after LMCache side. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-2e693f48e7  (L3, 2026-03-12, sha 2e693f48e7bd, PR #36307)
TITLE: [Perf] Add TRTLLM FP8 MoE Modular Kernel (#36307)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer.py (+2/-2); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+176/-55); vllm/model_executor/layers/fused_moe/oracle/fp8.py (+52/-51)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Add TRTLLM FP8 MoE Modular Kernel. The trtllm monolithic kernel has restrictions over the routing method of the model. This PR enables more models (e.g., minimax m2) to use the trtllm moe backend by adding the modular version. ⏎  ⏎ ## Test Plan ⏎ Tested with Minimax-m2.5 (which cannot use trtllm moe monolithic backend due to routing method restriction) ⏎ ``` ⏎ vllm serve MiniMaxAI/MiniMax-M2.5 \ ⏎     --trust-remote-code \ ⏎     --tensor-parallel-si …[truncated]

### L3-53ec16a705  (L3, 2026-03-12, sha 53ec16a705f2, PR #36145)
TITLE: [Hardware] Replace torch.cuda.device_count/current_device/set_device API (#36145)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: benchmarks/attention_benchmarks/mla_runner.py (+1/-1); benchmarks/attention_benchmarks/runner.py (+1/-1); benchmarks/kernels/benchmark_cutlass_moe_fp8.py (+1/-1); benchmarks/kernels/benchmark_device_communicators.py (+1/-1); benchmarks/kernels/benchmark_fused_collective.py (+2/-2); benchmarks/kernels/benchmark_grouped_gemm_cutlass.py (+1/-1); benchmarks/kernels/benchmark_w8a8_block_fp8.py (+2/-2); docs/configuration/conserving_memory.md (+1/-1); docs/usage/troubleshooting.md (+3/-3); examples/online_serving/new_weight_syncing/rlhf_http_ipc.py (+1/-1); (+79 more)
LABELS: documentation, performance, speculative-decoding, ready, v1, multi-modality, kv-connector, nvidia, ready-run-all-tests
BODY: ## Purpose ⏎ part of https://github.com/vllm-project/vllm/issues/30679  ⏎ will push 3 commit one by one : ⏎ 1. torch.cuda.device_count => torch.accelerator.device_count ⏎ 2. torch.cuda.current_device => torch.accelerator.current_device ⏎ 3. torch.cuda.set_device => torch.accelerator.set_device  ⏎  ⏎ ## Test Plan ⏎ CI ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-a1257fd1ea  (L3, 2026-03-12, sha a1257fd1ea93, PR #34597)
TITLE: [Kernel] Add FP8 KV cache support to Triton MLA decode attention (#34597)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.triton.decode_attention
FILES: vllm/v1/attention/backends/mla/triton_mla.py (+10/-7); vllm/v1/attention/ops/triton_decode_attention.py (+47/-0); docs/design/attention_backends.md (+1/-1); tests/kernels/attention/test_triton_decode_attention.py (+134/-0)
LABELS: documentation, ready, v1
BODY: Enable fp8/fp8_e4m3 KV cache for the Triton MLA attention backend, which is the only MLA backend available on sm120 GPUs. ⏎  ⏎ - Add fp8 and fp8_e4m3 to TritonMLABackend.supported_kv_cache_dtypes ⏎ - Thread k_scale/v_scale through decode attention kernel launch path ⏎ - Add FP8 dequant-on-load in both stage1 Triton kernels (MHA and grouped/MLA) ⏎ - Set supports_quant_query_input=False for FP8 (BF16 queries + FP8 KV) ⏎ - Add FP8-specific parametrized test cas …[truncated]

### L3-f444c05c32  (L3, 2026-03-12, sha f444c05c3267, PR #34732)
TITLE: [Attention] Use FA4 for MLA prefill (#34732)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_build, L3.flash_attn.fa_utils, L3.mla.common_v1
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/config/attention.py (+2/-2); vllm/model_executor/layers/attention/mla_attention.py (+8/-7); vllm/v1/attention/backends/fa_utils.py (+3/-0); benchmarks/attention_benchmarks/benchmark.py (+103/-29); benchmarks/attention_benchmarks/common.py (+2/-0); benchmarks/attention_benchmarks/configs/mla_prefill.yaml (+89/-25); benchmarks/attention_benchmarks/configs/mla_sparse_prefill.yaml (+62/-0); benchmarks/attention_benchmarks/mla_runner.py (+143/-14)
LABELS: documentation, performance, ready, ci/build, v1, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Depends on https://github.com/vllm-project/flash-attention/pull/123 and https://github.com/vllm-project/flash-attention/pull/126 ⏎  ⏎ Enable using FA4 as an MLA prefill (`forward_mha`) backend. This PR makes FA4 the default MLA prefill backend on Blackwell due to its improved performance over TRT-LLM, especially for extend operations relevant to the chunked prefill performed in vLLM ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ python benchmarks/attention_benchmarks …[truncated]

### L3-cc16b24b17  (L3, 2026-03-12, sha cc16b24b1798, PR #36768)
TITLE: Update Flashinfer to 0.6.6 (#36768)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); docker/Dockerfile.nightly_torch (+2/-2); docker/versions.json (+1/-1); requirements/cuda.txt (+1/-1); tools/pre_commit/update-dockerfile-graph.sh (+1/-1)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ Update Flashinfer to 0.6.6 to allow use of new kernels. ⏎  ⏎ Also fixed a minor issue in `update-dockerfile-graph.sh` to allow running from a path that includes a symlink. ⏎  ⏎ ## Test Plan ⏎  ⏎ Test the accuracy of a model that uses Flashinfer: ⏎ `vllm serve Qwen/Qwen3-30B-A3B-FP8` ⏎  ⏎ ## Test Result ⏎  ⏎ Model: Qwen/Qwen3-30B-A3B-FP8 ⏎ Running with default FLASHINFER attention and FLASHINFER_TRTLLM Fp8 MoE on Nvidia B200. ⏎  ⏎ Before: ⏎ |Tasks|Version|     Filt …[truncated]

### L3-a79c1c2c80  (L3, 2026-03-12, sha a79c1c2c806c, PR #36086)
TITLE: [AMD][Build] Add DeepEP to ROCm Dockerfile (#36086)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/test-amd.yaml (+16/-0); docker/Dockerfile.rocm (+33/-0)
LABELS: rocm, ready, ci/build
BODY: ## Purpose ⏎ Update the ROCm Docker image to include the ROCm port of DeepEP.  ⏎  ⏎ ## Test Plan ⏎ `pytest tests/kernels/moe/test_deepep_moe.py` ⏎  ⏎ ## Test Result ⏎ Before: ⏎ > ======================= 56 skipped, 5 warnings in 8.32s ======================== ⏎  ⏎ After: ⏎ > ================= 56 passed, 5 warnings in 1149.36s (0:19:09) ================== ⏎  ⏎ --- ⏎ [details omitted]

### L3-5e1a373d2e  (L3, 2026-03-13, sha 5e1a373d2e62, PR #36940)
TITLE: [BUG] Fix rank calculation in NCCLWeightTransferEngine (#36940)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/weight_transfer/nccl_engine.py (+1/-1)
LABELS: bug, ready
ISSUES: #36932 [Bug]: DP>1 doesn't work with weight syncing
BODY: ## Purpose ⏎ closes https://github.com/vllm-project/vllm/issues/36932 ⏎ Rank calculations were not being made correctly, `parallel_config.data_parallel_rank` for non moe models is always 0, switched to using `self.parallel_config.data_parallel_index` instead. ⏎  ⏎ --- ⏎ [details omitted]

### L3-f296a1966d  (L3, 2026-03-13, sha f296a1966dca, PR #36876)
TITLE: [Bugfix] Fix FlashInfer GDN warmup ValueError on SM90 GPUs (#36876)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_next.py (+8/-2)
LABELS: bug, ready, qwen
BODY: ## Summary ⏎ - PR #36599 added Triton autotuner warmup for GDN layers during V1 profiling, which also exercises the FlashInfer path on SM90 GPUs ⏎ - FlashInfer's `chunk_gated_delta_rule` returns a single tensor when `output_final_state=False`, but `fi_chunk_gated_delta_rule` always unpacked two values, causing a `ValueError` ⏎ - Fix: handle the return value based on `output_final_state` — unpack the tuple when `True`, use the single tensor when `False` …[truncated]

### L3-a4ad9db541  (L3, 2026-03-13, sha a4ad9db54169, PR #35786)
TITLE: Enable RoPE+KV cache fusion for ROCm AITER FA (non-shuffle layout) (#35786)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+46/-1); tests/compile/passes/test_rope_kvcache_fusion.py (+1/-0)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Enables the rope_kvcache_fusion pass for the ROCm AITER Flash Attention backend when shuffle KV cache layout is not used. ⏎  ⏎ **Changes:** ⏎ - **rocm_aiter_fa.py**: Add `fused_rope_kvcache_supported()` (returns true only when AITER is enabled and shuffle KV cache is disabled) and `do_rope_and_kv_cache_update()` using `rocm_aiter_ops.triton_rope_and_cache` with NHD layout. ⏎ - **test_rope_kvcache_fusion.py**: Add `ROCM_AITER_FA` to the parametrized backe …[truncated]

### L3-82f836d976  (L3, 2026-03-13, sha 82f836d976f3, PR #36962)
TITLE: [XPU] Support LoRA via torch.compile on XPU platform (#36962)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/platforms/xpu.py (+1/-3)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎ VLLM_USE_V1=1 CCL_ZE_IPC_EXCHANGE=drmfd VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn vllm bench throughput  --model meta-llama/Llama-3.2-3B-Instruct --backend vllm --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000 --max-loras 1 --max-lora-rank 8 --enable-lora --lora-path "jeeejeee/llama32-3b-text2sql-spider"  --gpu-memory-utilization 0.95 ⏎ ## Test Result ⏎  ⏎ without this pr: ⏎  ⏎ Asser …[truncated]

### L3-367cf5cd3e  (L3, 2026-03-13, sha 367cf5cd3eb2, PR #36931)
TITLE: [Feat][Bugfix] Enable additional dimension for Flashinfer MLA and fix routing dtype (#36931)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer
FILES: vllm/model_executor/models/deepseek_v2.py (+15/-2); vllm/v1/attention/backends/mla/flashinfer_mla.py (+3/-3)
LABELS: bug, ready, v1, deepseek, nvidia
DEEP_STUDY: deep-study correctness case vllm:367cf5cd3e: class=numerical_precision; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Purpose ⏎  ⏎ This PR implements the following: ⏎ - Enable additional `qk_nope_head_dim=64` for Flashinfer MLA, supported starting 0.6.6 ⏎ - Fix routing logits dtype in Flashinfer MoE for models not using `DeepseekV3` routing, e.g. Mistral Large 3 that uses `Renormalize` ⏎ - Fix a minor bug in torch compilation, when both `input_ids` and `inputs_embeds` are `None` in `DeepseekV2Model.forward()` ⏎  ⏎ ## Test Plan ⏎  ⏎ Test bugfix by running inference on `mistrala …[truncated]

### L3-6341d43043  (L3, 2026-03-13, sha 6341d4304351, PR #35316)
TITLE: [ROCm][Quantization] add quark w4a8 mxfp4_fp8 for LinearLayer (#35316)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/_aiter_ops.py (+53/-0); vllm/model_executor/layers/quantization/quark/quark.py (+32/-0); vllm/model_executor/layers/quantization/quark/schemes/__init__.py (+8/-1); vllm/model_executor/layers/quantization/quark/schemes/quark_w4a8_mxfp4_fp8.py (+218/-0)
LABELS: rocm, ready, gpt-oss
BODY: weights: mxfp4 with static scales ⏎ activations: fp8 ⏎  ⏎ 1. Supports Eager + Torch.compile mode ⏎ 2. Supports Emulation mode ⏎ 3. Updated test results below. ~~Tested on [this script](https://gist.github.com/divakar-amd/aad32a1fe87088392fb1f44dff26219b#file-quark_quantize_gpt-oss_wmxfp4_afp8_linearlayeronly-sh) for quantized gpt-oss LinearLayer only (with MoE quant_confit set to None)~~ ⏎  ⏎ ~~- [x] Remove temporary hack: Force MoE to use un-quantized path in …[truncated]

### L3-6d53efd2a5  (L3, 2026-03-13, sha 6d53efd2a582, PR #34695)
TITLE: [Bugfix] Fix MLA attention crash with AWQ/GPTQ quantized models (#34695)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+10/-5)
LABELS: bug, ready
ISSUES: #34561 [Bug]: GLM-4.7-Flash-AWQ fails with AttributeError: 'ColumnParallelLinear' object has no attribute 'weight'
BODY: ## Purpose ⏎  ⏎ Fix `AttributeError: 'ColumnParallelLinear' object has no attribute 'weight'` when running MLA models with AWQ/GPTQ quantization (e.g., `cyankiwi/GLM-4.7-Flash-AWQ-4bit`). ⏎  ⏎ Closes #34561. ⏎  ⏎ **Root cause**: MLA attention code accesses `self.kv_b_proj.weight.dtype` in 3 places, but AWQ/GPTQ-quantized `ColumnParallelLinear` layers store weights as `qweight` (packed int32), not `weight`. The code only accounted for unquantized and FP8-quan …[truncated]

### L3-2754231ba3  (L3, 2026-03-15, sha 2754231ba3a7, PR #36022)
TITLE: [Kernel] Add FlashInfer MoE A2A Kernel (#36022)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+11/-2); docs/design/moe_kernel_features.md (+2/-1); docs/serving/expert_parallel_deployment.md (+2/-1); tests/kernels/moe/modular_kernel_tools/mk_objects.py (+38/-5); vllm/config/parallel.py (+5/-2); vllm/distributed/device_communicators/all2all.py (+132/-6); vllm/distributed/device_communicators/cuda_communicator.py (+17/-4); vllm/distributed/device_communicators/mnnvl_compat.py (+7/-7); vllm/model_executor/layers/fused_moe/all2all_utils.py (+21/-4); vllm/model_executor/layers/fused_moe/config.py (+16/-4); (+9 more)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, ready, ci/build, v1, multi-modality
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ This PR is a port of PR #32217 to the vLLM top-of-tree after the modular kernel refactors in #32564. It adds the latest TRT-LLM gen A2A kernel from flashinfer's MoE-A2A API (one sided all-to-all) as added in (https://github.com/flashinfer-ai/flashinfer/pull/2102). This should perform better than the older A2A kernel from https://github.com/vllm-project/vllm/pull/21003 (formerly flashinfer_all2allv) in large batch size. ⏎  ⏎ The new kernel …[truncated]

### L3-697e4ff352  (L3, 2026-03-16, sha 697e4ff3528c, PR #36647)
TITLE: [GDN] add a config for gdn kernel selection (#36647)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/engine/arg_utils.py (+11/-0); vllm/model_executor/models/qwen3_next.py (+36/-4)
LABELS: ready, qwen
ISSUES: #36584 [Bug]: Qwen3.5-35B-A3B FlashInfer JIT compilation fails with C++17 feature errors (e.g., std::is_unsigned_v) when using vLLM 0.17.0
BODY: ## Purpose ⏎ Fix https://github.com/vllm-project/vllm/issues/36584 ⏎ Although the GDN kernel offers better performance, it requires C++17 for JIT compilation. So we've added a switching mechanism here to handle it. ⏎  ⏎ ``` ⏎ vllm serve --gdn-prefill-backend flashinfer ⏎ ``` ⏎ ``` ⏎ vllm serve --gdn-prefill-backend triton ⏎ ``` ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-8374387bd8  (L3, 2026-03-16, sha 8374387bd8ef, PR #36987)
TITLE: [FlashInfer] Revert block_size 16 + head_size 256 workaround on Blackwell (#36987)
SOURCES: path_core, subject_keyword, corpus:confirmed-reverts, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+0/-9); vllm/model_executor/models/config.py (+0/-12)
LABELS: ready, v1, qwen, nvidia
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 27994 reason=build_or_dependency
BODY: ## Purpose ⏎  ⏎ Revert the workaround introduced in [vllm-project/vllm#27994](https://github.com/vllm-project/vllm/pull/27994). ⏎  ⏎ PR #27994 was originally merged to work around a FlashInfer bug tracked in [flashinfer-ai/flashinfer#1993](https://github.com/flashinfer-ai/flashinfer/issues/1993), which reported that the combination of `block_size=16` and `head_size=256` produced incorrect results on Blackwell. The workaround forced `kernel_block_alignmen …[truncated]

### L3-116ed130f4  (L3, 2026-03-16, sha 116ed130f4d3, PR #34871)
TITLE: [Bugfix] Fix GDN attention crash with mixed decode/spec-decode batches (#34871)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_gdn_metadata_builder.py (+191/-0); vllm/v1/attention/backends/gdn_attn.py (+10/-0)
LABELS: bug, ready, v1
ISSUES: #34845 [Bug]: Qwen3-Next MTP fails when paired with chunked prefill
BODY: ## Purpose ⏎  ⏎ Fixes #34845.  ⏎  ⏎ When chunked prefill is enabled with speculative decoding (MTP) on Qwen3-Next, the GDN attention backend crashes with: ⏎  ⏎ ``` ⏎ AssertionError: num_decodes: 1, num_spec_decodes: 2 ⏎ ``` ⏎  ⏎ **Root cause**: The GDN attention backend asserts that `num_decodes` and `num_spec_decodes` are mutually exclusive. With chunked prefill + speculative decoding, a batch can violate this: a request's last prefill chunk may have `query_len=1`  …[truncated]

### L3-5ae685c1c8  (L3, 2026-03-16, sha 5ae685c1c85b, PR #34158)
TITLE: [Bugfix] Relax TRTLLM KV cache contiguity assertion for cross-layer layout (#34158)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+27/-2)
LABELS: bug, ready, v1, nvidia
ISSUES: #33572 [Bug]: GPT-OSS with CPU KV cache offload break with FlashInfer
BODY: Fixes #33572 ⏎  ⏎ The `is_strictly_contiguous` assertion on `kv_cache_permute` rejects the non-canonical strides views, and #33192 worked around it by just disabling TRTLLM entirely when KV transfer is enabled. ⏎  ⏎ I dug into the FlashInfer kernel source to check if full-tensor contiguity is actually needed. Turns out the TRTLLM kernels in `csrc/trtllm_fmha_kernel_launcher.cu` read actual tensor strides: ⏎  ⏎ ```cpp ⏎ int kv_stride_keys_values = key_cache.str …[truncated]

### L3-68e1b711f1  (L3, 2026-03-16, sha 68e1b711f1cf, PR #36612)
TITLE: [XPU] Add deepseek_scaling_rope fused kernel (#36612)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/_xpu_ops.py (+50/-0); vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py (+17/-0)
LABELS: ready, deepseek
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ [XPU] Add the usage of the fused deepseek_scaling_rope kernel in [PR](https://github.com/vllm-project/vllm-xpu-kernels/pull/34) for DeepseekScalingRotaryEmbedding. Previously, it ran with forward_native(). ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ Verified lm_eval with 4xBMG for DeepSeek-V2-Lite-Chat functionality locally. ⏎  ⏎ --- ⏎ [details omitted]

### L3-93f3c8e531  (L3, 2026-03-16, sha 93f3c8e53157, PR #37199)
TITLE: [Misc] Add `float16` to `CacheDType` (#37199)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashinfer_sparse, L3.mla.rocm_aiter, L3.mla.rocm_aiter_sparse, L3.dispatch.abstract_interface, L3.flex_attention, L3.tree_attention
FILES: vllm/v1/attention/backend.py (+5/-1); vllm/v1/attention/backends/flash_attn.py (+6/-1); vllm/v1/attention/backends/flashinfer.py (+1/-0); vllm/v1/attention/backends/flex_attention.py (+5/-1); vllm/v1/attention/backends/mla/cutlass_mla.py (+1/-0); vllm/v1/attention/backends/mla/flashattn_mla.py (+1/-0); vllm/v1/attention/backends/mla/flashinfer_mla.py (+1/-0); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+1/-0); vllm/v1/attention/backends/mla/flashmla.py (+1/-0); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+1/-0); (+9 more)
LABELS: documentation, rocm, ready, v1, nvidia
BODY: ## Purpose ⏎ This PR adds `float16` as an enumerated `CacheDType`. Many attention backends support `float16` in `supported_dtypes`, and also support `auto` in `supported_kv_cache_dtypes`, but don't list `float16` in `supported_kv_cache_dtypes`. This technically means that if someone tried to run a float16 model with `--kv-cache-dtype float16`, it would fail with no available attention backend. ⏎  ⏎ More importantly, this is a necessary step along the w …[truncated]

### L3-c88ea8338b  (L3, 2026-03-16, sha c88ea8338b9a, PR #36982)
TITLE: [MTP][Sparse MLA] Take advantage of native MTP support in indexer when possible (#36982)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+23/-12); csrc/sampler.cu (+1/-1)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎ PR #34552 added support for MTP > 1 with sparse MLA by unconditionally flattening all requests into single-token decodes. The indexer kernel does support MTP = 1, though, and future iterations of the kernel will support other token counts e.g. MTP = 3. This PR takes advantage of this support by only flattening when the specified `num_speculative_tokens` is not natively supported by the kernel. ⏎  ⏎ It does this by turning on `require_unifo …[truncated]

### L3-18be11fd59  (L3, 2026-03-16, sha 18be11fd59cd, PR #35594)
TITLE: [BUGFIX]fix CUDA OOM ERROR : invalid argument at cumem_allocator.cpp:119 (#35594)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/cumem_allocator.cpp (+8/-6)
LABELS: bug, ready, nvidia
BODY: solve https://github.com/vllm-project/vllm/issues/35612 ⏎ ## Purpose ⏎ when launch vllm serve "Qwen/Qwen3-4B" --enable-sleep-mode, it response CUDA Error:invalid argument at cumem_allocator.cpp:119, the root cause is cuDeviceGetAttribute is being wrong wrapped by CUDA_CHECK, and it cause the wrong error message. ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ image this is the error message before fix, after the fix code is implemented, it runs normal. image ⏎ <img widt …[truncated]

### L3-911355e216  (L3, 2026-03-16, sha 911355e216d3, PR #36845)
TITLE: [ROCm] Fix KV copy methods and auto-select attention backend for ROCm (#36845)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+24/-0); tests/v1/kv_connector/nixl_integration/spec_decode_acceptance_test.sh (+51/-17)
LABELS: rocm, ready, v1, kv-connector
BODY: - Added `insert_blocks_to_device` and `swap_out_blocks_to_host` to  `RocmPlatform`. These were only defined on `CudaPlatform`, causing a `TypeError: 'NoneType' object is not callable` crash when `NixlConnector` tried to copy KV blocks between GPU and CPU buffers during prefill/decode disaggregation on ROCm. ⏎  ⏎ - Updated `spec_decode_acceptance_test.sh` to auto-select the attention backend based on the detected GPU platform: `TRITON_ATTN` on ROCm, ` …[truncated]

### L3-6c1cfbad32  (L3, 2026-03-16, sha 6c1cfbad3250, PR #36867)
TITLE: Support non-contiguous KV cache in TRTLLM fp8 dequant kernel (#36867)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+57/-26); tests/kernels/attention/test_trtllm_kvfp8_dequant.py (+434/-0)
LABELS: ready, v1, nvidia
BODY: ## Summary ⏎  ⏎ Fix the `trtllm_prefill_attn_kvfp8_dequant` Triton kernel to support non-contiguous KV cache tensors (e.g. cross-layer unified allocation used by KV offloading). ⏎  ⏎ The kernel previously computed flat pointer offsets from tensor shape, assuming contiguity. With cross-layer KV caches the per-layer view is non-contiguous (strides skip over other layers), causing incorrect memory reads. Now the kernel uses actual tensor strides for the pag …[truncated]

### L3-0a0a1a198b  (L3, 2026-03-16, sha 0a0a1a198be8, PR #37181)
TITLE: Add ability to replace oot ops when using lora (#37181)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/mm_encoder_attention.py (+3/-3); vllm/lora/layers/column_parallel_linear.py (+4/-3); vllm/lora/layers/replicated_linear.py (+2/-1); vllm/lora/layers/row_parallel_linear.py (+2/-1); vllm/lora/layers/vocal_parallel_embedding.py (+2/-1); vllm/model_executor/custom_op.py (+3/-2)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Add ability to support lora when using custom ops. Otherwise, custom ops will not be replaced with Lora layers. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e5b807607c  (L3, 2026-03-16, sha e5b807607c84, PR #35448)
TITLE: [Quant][Feature] Support online MXFP8 quantization for MoE and dense models (#35448)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/quantization/test_mxfp8.py (+104/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+90/-21); vllm/model_executor/layers/fused_moe/oracle/fp8.py (+16/-1); vllm/model_executor/layers/fused_moe/oracle/mxfp8.py (+66/-23); vllm/model_executor/layers/fused_moe/utils.py (+1/-1); vllm/model_executor/layers/quantization/__init__.py (+3/-0); vllm/model_executor/layers/quantization/modelopt.py (+4/-5); vllm/model_executor/layers/quantization/mxfp8.py (+354/-0); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+101/-3); vllm/model_executor/layers/quantization/utils/quant_utils.py (+6/-0)
LABELS: ready, nvidia, quantization
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎ Add support for **online MXFP8 quantization** (`--quantization mxfp8`), enabling BF16/FP16 models to be dynamically quantized to MXFP8 (microscaling FP8 with block-32 scales) at load time — for both **linear layers** and **MoE expert layers**. ⏎  ⏎ This is powered by the FlashInfer kernels ⏎ - `trtllm_fp8_block_scale_moe` for MoE layers (see  #35986) ⏎ - `mm_mxfp8` for linear layers (see #35053 ) ⏎  ⏎ This PR implements part of the online quantiza …[truncated]

### L3-ca1954d58c  (L3, 2026-03-16, sha ca1954d58c49, PR #37090)
TITLE: [Bugfix] Disable cross-layer KV cache for MLA attention backends (#37090)
SOURCES: path_core, path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+6/-4); vllm/v1/attention/backends/mla/indexer.py (+3/-0); vllm/v1/worker/kv_connector_model_runner_mixin.py (+7/-2); tests/v1/kv_connector/unit/test_kv_cache_layout.py (+36/-0); vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py (+4/-2)
LABELS: bug, ready, v1, kv-connector
ISSUES: #37032 [Bug]: GLM 4.7-flash returns gibberish when native KV cache offloading is on
BODY: ## Purpose ⏎  ⏎ Fixes #37032 ⏎  ⏎ MLA models (e.g., GLM-4.7-Flash) produce garbage output with `--kv-offloading-size` because cross-layer KV cache allocation creates non-contiguous per-layer views. MLA decode kernels assume contiguous block layout, so they read wrong memory for block_id > 0. ⏎  ⏎ ### Fix (3 parts) ⏎  ⏎ 1. **`mla_attention.py`, `indexer.py`**: MLA backends return identity permutation `(0, 1, 2, 3)` from `get_kv_cache_stride_order(include_num_laye …[truncated]

### L3-fd4d96302a  (L3, 2026-03-16, sha fd4d96302a29, PR #37217)
TITLE: Fix eplb nvfp4 experts hook (#37217)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/cutlass_moe.py (+7/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+19/-4); vllm/model_executor/layers/fused_moe/flashinfer_cutedsl_moe.py (+4/-0); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+5/-0); vllm/model_executor/layers/fused_moe/layer.py (+11/-7); vllm/model_executor/layers/fused_moe/modular_kernel.py (+3/-0); vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (+6/-4); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+1/-0); vllm/model_executor/layers/quantization/modelopt.py (+1/-0); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+0/-10)
LABELS: ready, nvidia
DEEP_STUDY: deep-study correctness case vllm:fd4d96302a: class=numerical_precision; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Summary ⏎  ⏎ Two fixes for EPLB + NVFP4 quantization on FlashInfer modular kernels (CuteDSL/CUTLASS/TRTLLM): ⏎  ⏎ 1. **Exclude broadcast activation scales from EPLB**: `w13_input_scale` and `w2_input_scale` are global per-tensor activation scales shared across all experts (created via `.expand()` with stride 0). They are not per-expert weights and must not participate in expert rebalancing. ⏎ 2. **Fix stale quant config after EPLB rearrangement**: Absor …[truncated]

### L3-a3a51d20e7  (L3, 2026-03-16, sha a3a51d20e7d0, PR #37115)
TITLE: [Benchmark] Improvements to attention benchmark script (#37115)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/attention_benchmarks/benchmark.py (+54/-16); benchmarks/attention_benchmarks/common.py (+5/-0); benchmarks/attention_benchmarks/configs/mla_mixed_batch.yaml (+3/-3); benchmarks/attention_benchmarks/configs/mla_sparse_decode.yaml (+58/-0); benchmarks/attention_benchmarks/mla_runner.py (+132/-33); benchmarks/attention_benchmarks/runner.py (+59/-16)
LABELS: performance, ready
BODY: ## Purpose ⏎ This PR improves the attention benchmark in the following areas: ⏎ - Introduce **fp8 kv cache dtype** benchmarks for standard attention, MLA and MLA sparse. ⏎ - Adds **CUDA graph support** to the benchmark. Our investigation shows that the current measurements can be very off for decode benchmarks due to multiple kernel launches within attention backend and short kernel duration. ⏎ - Fix a bug related to `max_num_batched_tokens` that causes  …[truncated]

### L3-45f526d652  (L3, 2026-03-17, sha 45f526d65237, PR #36030)
TITLE: [BugFix] Correct max memory usage for multiple KV-cache groups (#36030)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_kv_cache_utils.py (+41/-0); vllm/v1/core/kv_cache_utils.py (+4/-2)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎ This PR fixes a calculation error in `_max_memory_usage_bytes_from_groups()` that led to underestimating total memory usage in multi-group cases. ⏎  ⏎ ### Root Cause ⏎ The original implementation **only calculated `blocks_needed` based on the first KV-Cache group (index `0`)**. In models with multiple KV-Cache groups (e.g., hybrid Mamba models), the block usage from other groups was neglected. This resulted in a lower-than-actual total memor …[truncated]

### L3-77d2a5f17b  (L3, 2026-03-17, sha 77d2a5f17b38, PR #36265)
TITLE: pick up tuned prefill configs for FP8 FA3 (#36265)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build, ready-run-all-tests
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎ Run CI for https://github.com/vllm-project/flash-attention/pull/125  ⏎ Benchmarking results are in FA PR

### L3-f63ed7b5ac  (L3, 2026-03-17, sha f63ed7b5aca6, PR #35243)
TITLE: [Bugfix] Fix DP MTP Dummy Run (#35243)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_worker.py (+2/-1)
LABELS: bug, ready, v1
ISSUES: #33899 [Bug]:  DeepSeek-R1-0528 AssertionError: tokens not padded correctly on GB200
BODY: ## Purpose ⏎  ⏎ FIX #33899 ⏎  ⏎ The DP dummy run issues a uniform_batch decode with num_reqs=1, num_tokens=1, and uniform_batch=True. This will pad the number of tokens to max_query_len==(1+num_speculative_tokens) when MTP is enabled. ⏎  ⏎ But when preparing num_scheduled_tokens (and then building query_start_loc), we do this: ⏎ ``` ⏎         elif uniform_decode: ⏎             assert not create_mixed_batch ⏎             num_reqs = min(max_num_reqs, cdiv(num_tokens,  …[truncated]

### L3-c25dbc2d27  (L3, 2026-03-17, sha c25dbc2d2728, PR #36955)
TITLE: [Bugfix] Fix unclean shutdown crash with AllReduce Fusion workspace (#36955)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/distributed/device_communicators/flashinfer_all_reduce.py (+19/-9)
LABELS: bug, ready, nvidia
ISSUES: #35686 [Bug][UX]: Unclean shutdown from ctrl-c with AR Fusion
BODY: Fixes #35686 ⏎  ⏎ ### Purpose ⏎ On B200 with allreduce fusion enabled, pressing ctrl-c causes FlashInfer `AllReduceFusionWorkspace.__del__` to crash with `ImportError: sys.meta_path is None, Python is likely shutting down`. ⏎  ⏎ The root cause is that global workspace objects (`_fi_ar_workspace`, `_fi_ar_quant_workspace`) rely on garbage collection `__del__` during interpreter shutdown, but Python's module/import system is already torn down by that point. ⏎  …[truncated]

### L3-bdb903bb5f  (L3, 2026-03-17, sha bdb903bb5f4b, PR #36674)
TITLE: [Bug] Fix FlashInfer MNNVL socket collisions under concurrent vLLM jobs (#36674)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/distributed/device_communicators/flashinfer_all_reduce.py (+16/-9)
LABELS: bug, ready, nvidia
DEEP_STUDY: deep-study correctness case vllm:bdb903bb5f: class=nondeterminism_race_sync; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎  ⏎ ```bash ⏎ [rank3]:[W310 15:54:45.576772833 ProcessGroupNCCL.cpp:1802] [PG ID 0 PG GUID 0 Rank 3] Failed to check the "should dump" flag on TCPStore, (maybe TCPStore server has shut down too early), with error: Failed to recv, got 0 bytes. Connection was likely closed. Did the remote server shutdown or crash? ⏎ [rank1]:[W310 15:54:45.574839167 TCPStore.cpp:125] [c10d] recvValue failed on SocketImpl(fd=86, addr=[localhost]:54074, remote=[lo …[truncated]

### L3-3ed7b1e6e0  (L3, 2026-03-17, sha 3ed7b1e6e0d4, PR #36846)
TITLE: [ROCm] Validate block_size for explicitly selected attention backends (#36846)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+0/-2)
LABELS: rocm, ready
BODY: #36274 stripped `block_size` from `attn_selector_config` before backend validation in `get_attn_backend_cls`, which was correct for auto-selection (block_size may not be finalized at that point). However, this also bypassed block_size validation for *explicitly* user-selected backends, breaking the contract established in #36292. ⏎  ⏎ - Add an explicit `supports_block_size` check for the selected-backend path, before the strip ⏎ - Auto-selection path i …[truncated]

### L3-e8f9dbc369  (L3, 2026-03-17, sha e8f9dbc369aa, PR #36720)
TITLE: [Bugfix][ROCm] Fix worker startup OOM on ROCm by skipping unreliable cudagraph memory profiling (#36720)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_worker.py (+5/-2)
LABELS: bug, rocm, ready, v1, nvidia
BODY: ## Problem ⏎  ⏎ On ROCm/HIP platforms, vLLM fails to start with: ⏎  ⏎     ValueError: Free memory on device cuda:0 (1.33/23.98 GiB) on startup is less ⏎     than desired GPU memory utilization (0.95, 22.79 GiB). ⏎  ⏎ This was introduced by the pr  ⏎ [UX][Startup] Account for CUDA graphs during memory profiling ⏎   #30515), which removed `profile_cudagraph_memory()` and ⏎ the explicit recalculation of `torch_peak_increase` and `non_kv_cache_memory`. ⏎  ⏎ ## Root Cause ⏎  ⏎ ` …[truncated]

### L3-68f783a727  (L3, 2026-03-17, sha 68f783a72749, PR #35673)
TITLE: [Torch 2.11] Guard torch._C._cpu attribute checks for forward compatibility (#35673)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+1/-1); benchmarks/kernels/cpu/benchmark_cpu_attn.py (+1/-1); benchmarks/kernels/cpu/benchmark_cpu_fused_moe.py (+1/-1); tests/kernels/attention/test_cpu_attn.py (+2/-4); tests/kernels/moe/test_cpu_fused_moe.py (+1/-1); vllm/model_executor/kernels/linear/mixed_precision/cpu.py (+1/-1); vllm/model_executor/layers/fused_moe/cpu_fused_moe.py (+1/-1); vllm/model_executor/layers/quantization/cpu_wna16.py (+1/-1); vllm/model_executor/layers/utils.py (+1/-1)
LABELS: performance, ready, v1, cpu
BODY: ## Summary ⏎  ⏎   - Migrate all torch._C._cpu._is_amx_tile_supported() calls to ⏎    the public API torch.cpu._is_amx_tile_supported() ⏎   - PyTorch 2.11 removed/renamed these private CPU ⏎   introspection attributes (see ⏎   https://github.com/pytorch/pytorch/pull/173433), causing ⏎   AttributeError crashes ⏎  ⏎   After: https://github.com/pytorch/pytorch/pull/173433 ⏎   Similar change on torchao: ⏎   https://github.com/pytorch/ao/pull/3845/ ⏎  ⏎   ## Failures on torch 2 …[truncated]

### L3-b36adfa349  (L3, 2026-03-17, sha b36adfa349cf, PR #37252)
TITLE: [Perf] Set Flashinfer sparse MLA as default backend for FP8 kv cache (#37252)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+31/-15); docs/design/attention_backends.md (+4/-2); tools/pre_commit/generate_attention_backend_docs.py (+26/-3)
LABELS: documentation, ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ This PR sets Flashinfer sparse MLA as default backend for FP8 KV cache for better performance. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎ Kernel microbenchmark results: https://github.com/vllm-project/vllm/issues/35807 ⏎  ⏎ E2E results with different TP (with EP enabled) ⏎ <img width="2384" height="1035" alt="nvidia_DeepSeek-V3 2-NVFP4_backend_cmp_isl8192_osl1024" src="https://github.com/user-attachments/assets/c0e284c1-b94f-43d4-bf50-e7f6f5b662d9" /> ⏎ -  …[truncated]

### L3-2660b9289c  (L3, 2026-03-17, sha 2660b9289c1f, PR #37178)
TITLE: Bugfix for offloading+prefetch for GLM-4.7-FP8 (#37178)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/offloader/prefetch.py (+42/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎ This fixes a bug when using offload+prefetch feature with GLM-4.7-FP8 on RTX PRO 6000 hardware, see #37176 . ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve "zai-org/GLM-4.7-FP8" \ ⏎ 	--served-model-name "zai-org/GLM-4.7-FP8" \ ⏎ 	--tensor-parallel-size 8 \ ⏎ 	--enable-expert-parallel \ ⏎ 	--offload-group-size 2 \ ⏎ 	--offload-num-in-group 1 \ ⏎ 	--offload-prefetch-step 1 \ ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ Without the bugfix, startup fails with "AttributeError: 'Attention' obje …[truncated]

### L3-ce2ef42fd3  (L3, 2026-03-18, sha ce2ef42fd3ae, PR #37335)
TITLE: [CI] Stabilize test_cpu_offloading by waiting for async offload before cache reset (#37335)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_offload/test_cpu_offloading.py (+46/-7)
LABELS: rocm, ready, v1
BODY: - `test_cpu_offloading[TRITON_ATTN-48]` was intermittently failing because `reset_prefix_cache()` was called while the async GPU-to-CPU offload was still in progress, returning `False` (silently ignored). This meant the GPU prefix cache was never actually cleared, no new CPU stored events were produced, and `assert subscriber.get_new_cpu_stored_events()` failed with an empty list. ⏎  ⏎ - Add `_wait_for_prefix_cache_reset()` that retries with a timeou …[truncated]

### L3-e6c4797704  (L3, 2026-03-18, sha e6c4797704e4, PR #36927)
TITLE: [ROCm][Quantization] add fp8xfp8 attn support for rocm_aiter_unified_attn (#36927)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+17/-5)
LABELS: rocm, ready, v1
BODY: As a follow-up of https://github.com/vllm-project/vllm/pull/35316,  ⏎ This PR adds support for fp8xfp8 attention for `rocm_aiter_unified_attn`  ⏎ For fp8xfp8 computation, the aiter triton kernel expects Q to be in fp8 as well, otherwise it descales it to fp32. [(LINK)](https://github.com/ROCm/aiter/blob/3a217fc9885a5d5f799c2f2aacecabf9f5e99614/aiter/ops/triton/_triton_kernels/attention/unified_attention.py#L270-L276). However, Q scales are not suppor …[truncated]

### L3-c373b5c00d  (L3, 2026-03-18, sha c373b5c00d1a, PR #37313)
TITLE: [Log] Reduce duplicate log (#37313)
SOURCES: path_core
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/model_executor/layers/attention/mm_encoder_attention.py (+3/-1); vllm/compilation/backends.py (+3/-1); vllm/config/scheduler.py (+2/-1); vllm/model_executor/models/qwen3_next.py (+4/-3); vllm/platforms/cuda.py (+2/-1); vllm/v1/executor/multiproc_executor.py (+2/-1); vllm/v1/worker/dp_utils.py (+2/-1); vllm/v1/worker/gpu_model_runner.py (+2/-1)
LABELS: ready, v1, qwen, nvidia
BODY: ## Purpose ⏎  ⏎ For example ⏎  ⏎ ```bash ⏎ INFO 03-17 15:12:19 [dp_utils.py:30] Using CPU all reduce to synchronize DP padding between ranks. ⏎ INFO 03-17 15:12:19 [dp_utils.py:30] Using CPU all reduce to synchronize DP padding between ranks. ⏎ INFO 03-17 15:12:19 [dp_utils.py:30] Using CPU all reduce to synchronize DP padding between ranks. ⏎ INFO 03-17 15:12:19 [dp_utils.py:30] Using CPU all reduce to synchronize DP padding between ranks. ⏎ ``` ⏎ ->  ⏎ ```bash ⏎ INFO  …[truncated]

### L3-58cde5c026  (L3, 2026-03-18, sha 58cde5c026ef, PR #37330)
TITLE: [ROCm][CI] Skip trtllm kvfp8 dequant tests on ROCm (#37330)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_trtllm_kvfp8_dequant.py (+6/-0)
LABELS: rocm, ready, nvidia
BODY: - Add module-level `pytest.skip` for ROCm in `test_trtllm_kvfp8_dequant.py` ⏎ - Follows the same skip pattern used in `test_flashinfer.py` ⏎  ⏎ cc @kenroche

### L3-0ef7f79054  (L3, 2026-03-18, sha 0ef7f79054b9, PR #37340)
TITLE: [Perf] Add tuned triton moe config for Qwen3.5 H200, 9.9% E2E throughput improvement (#37340)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_moe.py (+33/-9); vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json (+147/-0)
LABELS: performance, ready, qwen
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ 1. updating tuning script, adding support for Qwen3.5 ⏎ 2. add a tuned config for it ⏎  ⏎ Note: Computing resource is limited so just running for bs 128, which is the hot path in practice. ⏎  ⏎ ## Test ⏎  ⏎ `export MODEL="Qwen/Qwen3.5-35B-A3B-FP8"` ⏎ `vllm serve $MODEL  --port 9256 --enable-expert-parallel --enable-expert-parallel --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/home/yewentao256/profile_vllm` ⏎  ⏎ ### Acc ⏎  ⏎ `lm_eval  …[truncated]

### L3-0d81a1fe61  (L3, 2026-03-18, sha 0d81a1fe6190, PR #37195)
TITLE: [V0 Deprecation] Deprecate virtual engine (#37195)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/attention.py (+2/-2); vllm/model_executor/layers/attention/mla_attention.py (+2/-2); vllm/model_executor/layers/attention/static_sink_attention.py (+1/-2); tests/compile/passes/test_rope_kvcache_fusion.py (+2/-2); tests/v1/kv_connector/unit/test_decode_bench_connector.py (+1/-1); tests/v1/kv_connector/unit/test_lmcache_integration.py (+0/-1); tests/v1/kv_connector/unit/test_nixl_connector.py (+0/-8); tests/v1/kv_connector/unit/test_offloading_connector.py (+0/-1); vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py (+1/-1); vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py (+1/-3); (+13 more)
LABELS: ready, v1, qwen, kv-connector
BODY: ## Purpose ⏎  ⏎ Deprecate virtual engine, for V1 and V2 the virtual engine is always set to 0 so nothing effect on default behavior in main. ⏎  ⏎ Tests in CI

### L3-296839a1b0  (L3, 2026-03-18, sha 296839a1b07e, PR #30647)
TITLE: [Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE (#30647)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/fusions_e2e/conftest.py (+4/-0); tests/compile/fusions_e2e/models.py (+9/-0); tests/compile/fusions_e2e/test_tp2_ar_rms.py (+2/-1); vllm/model_executor/layers/fused_moe/fused_moe_method_base.py (+5/-0); vllm/model_executor/layers/fused_moe/runner/default_moe_runner.py (+4/-1); vllm/model_executor/layers/quantization/mxfp4.py (+16/-1)
LABELS: ready, ci/build, gpt-oss, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ - Depends on Flashinfer update #30993 ⏎ - Eliminated padding op before the MoE: by setting the alignment in flashinfer mxfp8 quant, the output quantized tensor will be padded. ⏎ - Eliminated slicing op after the MoE: by passing the output tensor with unpadded hidden size to MoE kernel, this depends on a Flashinfer PR: ⏎   - https://github.com/flashinfer-ai/flashinfer/pull/2217 ⏎   - This will also resolve the previous AR+Norm fusion broken by …[truncated]

### L3-17c47fb869  (L3, 2026-03-18, sha 17c47fb8691f, PR #37322)
TITLE: [Bugfix] Fix EP weight filter breaking EPLB and NVFP4 accuracy (#37322)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/model_loader/default_loader.py (+7/-0); vllm/model_executor/model_loader/ep_weight_filter.py (+5/-0)
LABELS: bug, ready
BODY: ## Summary ⏎  ⏎ PR #37136 introduced an EP weight filter that skips non-local expert tensors before reading them from disk, drastically reducing storage I/O for MoE models under EP. However, it has two bugs: ⏎  ⏎ **1. Breaks EPLB (catastrophic accuracy loss — gsm8k drops from ~0.95 to ~0.08)** ⏎  ⏎ When EPLB is enabled, redundant physical expert slots (e.g., 256 logical + 32 redundant = 288 physical for DeepSeek-R1) can map to *any* logical expert. The EP we …[truncated]

### L3-70b81c4f3d  (L3, 2026-03-18, sha 70b81c4f3d1a, PR #37449)
TITLE: [bugfix][async scheduling] fix extra cuda context in device 0 with EP/DP (#37449)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/executor/multiproc_executor.py (+23/-11)
LABELS: bug, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ See https://forums.developer.nvidia.com/t/when-a-thread-has-a-primary-cuda-context-does-the-child-thread-it-creates-automatically-inherit-the-cuda-context/362810 , a new thread does not have any cuda context, and later cuda runtime call might create a context in device 0. ⏎  ⏎ ## Test Plan ⏎  ⏎ Run vLLM serve with EP/DP: ⏎  ⏎ `vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 -dp 2 -ep  --port 8899` ⏎  ⏎ test with multiple requests: ⏎  ⏎ ```bash ⏎ vllm bench ser …[truncated]

### L3-577df69b26  (L3, 2026-03-18, sha 577df69b2649, PR #37054)
TITLE: [Bugfix] Fix KV scales inconsistency in fp8 MLA & FlashInfer kv_cache_dtype "auto" leading to gibberish (#37054)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.cutlass_v1_backend, L3.mla.flashinfer, L3.mla.flashinfer_sparse
FILES: vllm/v1/attention/backends/flashinfer.py (+6/-2); vllm/v1/attention/backends/mla/cutlass_mla.py (+5/-0); vllm/v1/attention/backends/mla/flashinfer_mla.py (+7/-2); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+6/-2); vllm/v1/attention/backends/mla/triton_mla.py (+1/-1); tests/v1/attention/test_mla_backends.py (+31/-28); tests/v1/attention/test_sparse_mla_backends.py (+9/-2); tests/v1/attention/test_trtllm_attention_integration.py (+6/-6)
LABELS: bug, documentation, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ This PR fixes the following issues: ⏎ - FlashInfer + kv_cache_dtype "auto" generates giberrish when layer._[qkv]_scale != 1.0 ⏎   - Bug is FI applies the layer._[qkv]_scale unconditionally, even when the QKV values are in unscaled bf16. ⏎   - This applies to both normal & MLA attention paths. ⏎ - KV cache scales not properly handed when using MLA + fp8. ⏎   - In MLA, the KV latents necessarily must use the same quantization scale for K & V, so  …[truncated]

### L3-ef2c4f778d  (L3, 2026-03-19, sha ef2c4f778df5, PR #37442)
TITLE: [Bugfix] Zero-init MLA attention output buffers to prevent NaN from CUDA graph padding (#37442)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.cutlass_v1_backend, L3.mla.flashinfer
FILES: vllm/v1/attention/backends/mla/cutlass_mla.py (+14/-1); vllm/v1/attention/backends/mla/flashinfer_mla.py (+44/-0)
LABELS: bug, ready, v1, nvidia
BODY: ## Summary ⏎  ⏎ When running CUDA graph decode with padding (e.g. batch of 1024 with 1 real request), unused slots have `seq_lens=0`. The MLA decode kernels (both CUTLASS MLA and FlashInfer TRT-LLM MLA) skip writing output for these slots, leaving stale data in the output buffer. If that stale data contains NaN (from a previous iteration or uninitialized memory), it propagates to real tokens via downstream per-tensor FP8 quantization (`amax` over the …[truncated]

### L3-6accb21f2a  (L3, 2026-03-19, sha 6accb21f2a9a, PR #37024)
TITLE: [bug] Fix deadlock with pause resume and collective_rpc (#37024)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/engine/core.py (+5/-1)
LABELS: bug, ready, v1
ISSUES: #36594 [Bug]: DPEngineCoreProc may re-arm DP wave while paused (START_DP_WAVE ignores pause state), causing collective timeout after pause_generation + collective_rpc
BODY: ## Purpose ⏎ closes https://github.com/vllm-project/vllm/issues/36594 ⏎  ⏎ ## Test Plan ⏎  ⏎ pending large scale training run ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e390742c59  (L3, 2026-03-19, sha e390742c5906, PR #37536)
TITLE: Fix KV Offloading + MLA AssertionError by using num_kv_heads=1 in cpu… (#37536)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/kv_offload/worker/cpu_gpu.py (+1/-1)
LABELS: ready, v1
BODY: …_gpu.py ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-4ca3fa6bb4  (L3, 2026-03-20, sha 4ca3fa6bb463, PR #37606)
TITLE: [ROCm][Bugfix] fix cache block size mismatch for aiter unified attention (#37606)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_unified, L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+0/-24); vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+7/-0)
LABELS: bug, rocm, ready, v1
ISSUES: #37548 [Bug][ROCm]: Aiter unified attention fails during compilation
BODY: This PR fixes the following issue (Resolves https://github.com/vllm-project/vllm/issues/37548): ⏎  ⏎ - We want to use cache block size of `64` for AITER_UNIFIED_ATTENTION. The current logic works as intended when we use the env variable `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1` ⏎ - However, cache block size still resolves to 16 when  `--attention-config '{"backend": "ROCM_AITER_UNIFIED_ATTN"}'` is used. ⏎  ⏎ This PR resolves this mismatch irrespective …[truncated]

### L3-ca1ac1a4b4  (L3, 2026-03-20, sha ca1ac1a4b44f, PR #37452)
TITLE: Fix DP coordinator ZMQ port TOCTOU (#37452)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/utils/network_utils.py (+1/-1); vllm/v1/engine/coordinator.py (+57/-7)
LABELS: ready, v1
BODY: Previously the parent selected the DP coordinator's TCP ZMQ ports with ⏎ `get_open_port()` before the coordinator actually bound them, leaving a ⏎ window where another socket could claim the ports. ⏎  ⏎ Fix this by letting the coordinator bind first and report the bound ZMQ ⏎ addresses back to the parent via pipe.

### L3-47b7af0d87  (L3, 2026-03-20, sha 47b7af0d8770, PR #37207)
TITLE: [Feat] Enable CompressedTensorW4A8Int for XPU (#37207)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/_xpu_ops.py (+54/-0); vllm/model_executor/kernels/linear/__init__.py (+3/-0); vllm/model_executor/kernels/linear/mixed_precision/__init__.py (+2/-0); vllm/model_executor/kernels/linear/mixed_precision/xpu.py (+113/-0)
LABELS: ready
BODY: ## Purpose ⏎ Allow executing models quantized to CompressedTensorsW4A8Int using int4_gemm_w4a8 for xpu. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ python3 vllm_local/examples/basic/offline_inference/generate.py --model tranhuonglan/qwen3-06B-base-smoothquant08-gptqmodifier-w4a8-linear ⏎ ``` ⏎ ## Test Result ⏎ Using just one of the prompts as outputs are long. ⏎ ``` ⏎ -------------------------------------------------- ⏎ Prompt: 'The future of AI is' ⏎ Generated text: ' not just commercia …[truncated]

### L3-0140eafb15  (L3, 2026-03-20, sha 0140eafb1546, PR #37461)
TITLE: [Bug] Fix FlashInfer allreduce fusion workspace uninitialized error (#37461)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+38/-39); vllm/distributed/device_communicators/flashinfer_all_reduce.py (+90/-86)
LABELS: bug, ready, nvidia
ISSUES: #37468 [Bug]: FlashInfer allreduce fusion workspace uninitialized error
DEEP_STUDY: deep-study correctness case vllm:0140eafb15: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎ Fix #37468 ⏎  ⏎ Currently, the FlashInfer allreduce fusion workspace is created in `AllReduceFusionPass.__init__`. However, when torch compile directly loads the compiled module from cache, it skips running the passes. Thus, the workspace will not be initialized, causing error when the kernel is called which expects the workspace to be in place. This PR fixes this by adding workspace initialization to the kernel code `call_trtllm_fused_all …[truncated]

### L3-1779c09898  (L3, 2026-03-20, sha 1779c09898e0, PR #34709)
TITLE: [ROCm] Enable wvSplitK skinny GEMM kernel for RDNA4/gfx1x decode (#34709)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: csrc/rocm/skinny_gemms.cu (+268/-92); tests/kernels/quantization/test_rocm_skinny_gemms.py (+5/-4); tests/model_executor/layers/test_rocm_unquantized_gemm.py (+89/-0); vllm/model_executor/layers/utils.py (+3/-3)
LABELS: new-model, rocm, ready, ci/build
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ Enable the `wvSplitK` and `wvSplitKQ` skinny GEMM kernels on RDNA (gfx11/gfx12) hardware for decode-phase GEMMs (M=1..4). Previously these kernels were gfx9-only (MI-series), with RDNA falling through to `torch.nn.functional.linear` (unquantized) or generic `torch._scaled_mm` (FP8). ⏎  ⏎ **~15% decode token/s improvement on AMD Radeon AI PRO R9700 (gfx1201).** ⏎  ⏎ ### Commit 1: `wvSplitK` — unquantized BF16/FP16 skinny GEMM ⏎  ⏎ #### Kernel chan …[truncated]

### L3-fb4e8bf442  (L3, 2026-03-20, sha fb4e8bf442c5, PR #37613)
TITLE: [ROCm][CI] Fix accuracy for llama-nemotron-vl pooling tests (#37613)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/pooling/test_llama_nemotron_vl.py (+8/-1)
LABELS: rocm, ready, multi-modality, llama
BODY: Follow-up for: ⏎ - #34839  ⏎  ⏎ Fixes small accuracy diff due to differences in HF and vLLM attention backends on ROCm in `mi250_1: Multi-Modal Models (Extended Pooling) ⏎ ` ⏎  ⏎ Motivation: https://buildkite.com/vllm/amd-ci/builds/6701/steps/canvas?sid=019d07a7-1a1e-445a-8480-1feaf029a19d&tab=output ⏎  ⏎ cc @kenroche

### L3-269bf46d99  (L3, 2026-03-20, sha 269bf46d99f1, PR #36708)
TITLE: fix: disambiguate multimodal prefix cache keys (#36708)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_kv_cache_utils.py (+10/-8); tests/v1/core/test_prefix_caching.py (+12/-4); vllm/v1/core/kv_cache_utils.py (+7/-4)
LABELS: ready, v1
BODY: ## Summary ⏎ Disambiguate multimodal prefix cache keys. ⏎  ⏎ ## Change ⏎ - adjust multimodal prefix cache key construction in `vllm/v1/core/kv_cache_utils.py`

### L3-e1d85e5c24  (L3, 2026-03-20, sha e1d85e5c2454, PR #37303)
TITLE: [Attention] Support distinguishing between short extends and decodes (#37303)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+6/-0); vllm/v1/attention/backends/utils.py (+49/-24); .buildkite/test_areas/engine.yaml (+12/-0); tests/v1/attention/test_batch_reordering.py (+86/-37); tests/v1/e2e/test_hybrid_chunked_prefill.py (+2/-2); vllm/v1/attention/backends/mamba_attn.py (+3/-1); vllm/v1/worker/gpu_input_batch.py (+7/-1); vllm/v1/worker/gpu_model_runner.py (+13/-28); vllm/v1/worker/mamba_utils.py (+0/-42)
LABELS: ready, ci/build, v1
BODY: Alternative to https://github.com/vllm-project/vllm/pull/35447, support distinguishing between short-extends/prefills and decodes via batch reordering; the batch order is now: ⏎  ⏎ ``` ⏎         decode:        (num_scheduled <= threshold AND is not prefilling) ⏎         short_extend:  (num_scheduled <= threshold AND is chunked prefilling) ⏎         long_extend:   (num_scheduled > threshold AND is chunked prefilling) ⏎         prefill:       (num_computed ==  …[truncated]

### L3-4f16ebbbd3  (L3, 2026-03-20, sha 4f16ebbbd35e, PR #37605)
TITLE: [Bugfix] Disable monolithic TRTLLM MoE for Renormalize routing (#37591) (#37605)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/lm_eval.yaml (+16/-0); tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml (+8/-0); tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml (+9/-0); tests/evals/gsm8k/configs/models-qwen35-blackwell.txt (+1/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+8/-5)
LABELS: bug, ready, ci/build, qwen, nvidia
ISSUES: #37591 [Bug]: FlashInfer TRTLLM monolithic MoE produces 0% accuracy for Qwen3.5-35B/122B FP8
BODY: ## Summary ⏎  ⏎ The FlashInfer TRTLLM monolithic MoE kernel (`trtllm_fp8_block_scale_moe`) produces incorrect expert routing when all router logits are negative. This causes 0% GSM8K accuracy for Qwen3.5-35B-A3B-FP8 and Qwen3.5-122B-A10B-FP8 with DP+EP. ⏎  ⏎ This PR removes `Renormalize` and `RenormalizeNaive` from `TrtLlmFp8ExpertsMonolithic._supports_routing_method()`, causing the backend selection to fall through to `TrtLlmFp8ExpertsModular`. The  …[truncated]

### L3-6951fcd44f  (L3, 2026-03-20, sha 6951fcd44fdd, PR #37634)
TITLE: [XPU] Automatically detect target platform as XPU in build. (#37634)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: setup.py (+3/-0)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ Before VLLM_TARGET_DEVICE=xpu is required to build vLLM for XPU, this patch will automatically detect XPU as target platform.  ⏎ ## Test Plan ⏎ pip install -r requirements/xpu.txt && \ ⏎ pip install --no-build-isolation . ⏎ ## Test Result

### L3-e5ed6c6c13  (L3, 2026-03-20, sha e5ed6c6c134f, PR #37475)
TITLE: [BugFix] Allow qk_nope_head_dim=192 in FlashInfer MLA backend checks (#37475)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer, L3.mla.flashinfer_sparse
FILES: vllm/v1/attention/backends/mla/flashinfer_mla.py (+4/-4); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+4/-4)
LABELS: bug, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ Align vLLM's FlashInfer MLA backend gating with [FlashInfer](https://github.com/flashinfer-ai/flashinfer/pull/2607) support for `qk_nope_head_dim=192' for glm-5 model. ⏎  ⏎ ## Test Plan ⏎  ⏎ Run and check its accuracy ⏎ `vllm serve models/GLM-5-NVFP4   -tp 8   --trust-remote-code   --enable-auto-tool-choice   --tool-call-parser glm47   --reasoning-parser glm45   --enable-chunked-prefill   --max-num-batched-tokens 131072 --gpu-memory-utilization …[truncated]

### L3-5a4a179591  (L3, 2026-03-20, sha 5a4a1795916a, PR #37611)
TITLE: [ROCm][CI] Fix granite_speech test for gfx90a by selecting compatible attention backend (#37611)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_granite_speech.py (+5/-1)
LABELS: rocm, ready, multi-modality
BODY: Follow-up for: ⏎ - #34839  ⏎  ⏎ Fixes incompatible (AITER) attention backend issue in `mi250_1: Multi-Modal Models (Extended Generation 1)` ⏎  ⏎ Motivation: https://buildkite.com/vllm/amd-ci/builds/6701/steps/canvas?sid=019d07a7-1a1b-45c0-9cb9-e9f1d9f2bc33&tab=output ⏎  ⏎ cc @kenroche

### L3-79eb9369c5  (L3, 2026-03-20, sha 79eb9369c5ba, PR #37426)
TITLE: fix CUDAGraph memory being counted twice (#37426)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_worker.py (+2/-3)
LABELS: ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ Fix twice-counted CUDAGraph memory ⏎  ⏎ When `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`, CUDAGraph memory appears to be counted twice in the KV cache recommendation path. ⏎  ⏎ In `determine_available_memory` , `self.peak_activation_memory` already includes the estimated CUDAGraph memory: ⏎  ⏎ ``` ⏎ cudagraph_memory_estimate_applied = ( ⏎     cudagraph_memory_estimate ⏎     if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS ⏎     else 0 ⏎ ) ⏎ self.peak_activa …[truncated]

### L3-87bd91892f  (L3, 2026-03-21, sha 87bd91892f8c, PR #37128)
TITLE: [MoE Refactor] Mxfp4 oracle rebased (#37128)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/design/moe_kernel_features.md (+2/-2); tests/compile/fusions_e2e/conftest.py (+4/-1); tests/kernels/moe/test_gpt_oss_triton_kernels.py (+8/-0); tests/kernels/moe/test_ocp_mx_moe.py (+1/-1); tests/kernels/quantization/test_mxfp4_triton_ep.py (+0/-83); vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py (+352/-0); vllm/model_executor/layers/fused_moe/fused_marlin_moe.py (+3/-1); vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py (+151/-20); vllm/model_executor/layers/fused_moe/layer.py (+0/-28); vllm/model_executor/layers/fused_moe/oracle/mxfp4.py (+847/-0); (+8 more)
LABELS: documentation, rocm, ready, gpt-oss, nvidia
BODY: ## Purpose ⏎ Rebased and improve version of #34983 ⏎ Ongoing MXFP4 MoE refactor ⏎  ⏎ - Refactor MXFP4 MoE from a monolithic 1299-line `Mxfp4MoEMethod` class to the oracle pattern used by FP8 and NvFP4 ⏎   - Create `oracle/mxfp4.py` with backend selection, weight conversion, quant config, and kernel assembly ⏎   - Create `TrtLlmMxfp4ExpertsMonolithic` wrapping `trtllm_fp4_block_scale_moe()` (both BF16 and MXFP8 input modes) ⏎   - Create `OAITritonMxfp4ExpertsMo …[truncated]

### L3-61e381dcf0  (L3, 2026-03-21, sha 61e381dcf01f, PR #37756)
TITLE: [Perf] Add SM 10.3 (B300/GB300) all-reduce communicator tuning (#37756)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+10/-0); vllm/distributed/device_communicators/all_reduce_utils.py (+12/-0); vllm/distributed/device_communicators/symm_mem.py (+1/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Add benchmarked optimal all-reduce communicator config values for SM 10.3 (B300/GB300). ⏎  ⏎ Depends on #37755 for allreduce fusion to be auto-enabled on SM 10.3. ⏎  ⏎ ## Config Values ⏎  ⏎ | Config | ws=2 | ws=4 | ws=6 | ws=8 | ⏎ |--------|------|------|------|------| ⏎ | `CUSTOM_ALL_REDUCE_MAX_SIZES` | 4 MiB | 4 MiB | 8 MiB | 4 MiB | ⏎ | `SYMM_MEM_ALL_REDUCE_MAX_SIZES` | 4 MiB | 32 MiB | 32 MiB | 64 MiB | ⏎ | `_WORLD_SIZES_MULTIMEM` | -- |  …[truncated]

### L3-4383f1532e  (L3, 2026-03-22, sha 4383f1532e87, PR #35927)
TITLE: [MoE] Move PF Methods to Folder (#35927)
SOURCES: corpus:production-kernel-provenance, body_keyword
ARTIFACT_HINTS: -
FILES: docs/design/moe_kernel_features.md (+4/-4); tests/kernels/moe/modular_kernel_tools/mk_objects.py (+4/-4); tests/kernels/moe/parallel_utils.py (+2/-2); tests/kernels/moe/test_deepep_deepgemm_moe.py (+2/-2); tests/kernels/moe/test_deepep_moe.py (+2/-2); vllm/model_executor/layers/fused_moe/all2all_utils.py (+8/-8); vllm/model_executor/layers/fused_moe/prepare_finalize/__init__.py (+3/-0); vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py (+0/-0); vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py (+0/-0); vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py (+0/-0); (+1 more)
LABELS: documentation, ready, nvidia
BODY: ## Summary ⏎  ⏎ - Moves `deepep_ht_prepare_finalize.py` → `prepare_finalize/deepep_ht.py` ⏎ - Moves `deepep_ll_prepare_finalize.py` → `prepare_finalize/deepep_ll.py` ⏎ - Moves `flashinfer_a2a_prepare_finalize.py` → `prepare_finalize/flashinfer_a2a_prepare_finalize.py` ⏎ - Updates all importers (`all2all_utils.py` and four test files) to use the new paths ⏎ - No functional changes — pure file reorganization ⏎  ⏎ This is part of the ongoing effort to organize `vll …[truncated]

### L3-66f927f205  (L3, 2026-03-22, sha 66f927f205fd, PR #37775)
TITLE: [Bugfix] Fix pooling non-determinism from pinned prompt_lens aliasing (#37775)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_input_batch.py (+62/-0); vllm/v1/worker/gpu_input_batch.py (+1/-1)
LABELS: bug, rocm, ready, v1
BODY: - PR #37303 changed `num_prompt_tokens` in `InputBatch` from a plain `np.zeros()` array to a pinned-memory-backed numpy view (`torch.zeros(..., pin_memory=True).numpy()`). ⏎ - `get_pooling_metadata()` calls `torch.from_numpy(self.num_prompt_tokens[:self.num_reqs])`, which creates a tensor that shares the underlying pinned buffer rather than copying the data. ⏎ - Because pinned memory is used for async GPU transfers, the shared buffer can be modifie …[truncated]

### L3-77d24c4bfe  (L3, 2026-03-22, sha 77d24c4bfedc, PR #37718)
TITLE: [Bug] Fix fp8 deepgemm batch invariant (#37718)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/utils/fp8_utils.py (+5/-0)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ Selecting kernels between `run_flashinfer_deepgemm_swapAB` and `run_deepgemm` will break batch invaraince, this PR fixes the issue. ⏎  ⏎ ## Test ⏎  ⏎ `VLLM_TEST_MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8" pytest tests/v1/determinism/test_batch_invariance.py -svx` ⏎  ⏎ ```bash ⏎ # now ⏎ ============== 10 passed, 28 warnings in 458.02s (0:07:38) =============== ⏎ # original ⏎ ======================= short test summary info ==================== …[truncated]

### L3-eaf4978621  (L3, 2026-03-22, sha eaf4978621ac, PR #37719)
TITLE: [Test] Only Run MLA model when user explicitly set for batch invariance (#37719)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/determinism/test_batch_invariance.py (+6/-9); tests/v1/determinism/test_online_batch_invariance.py (+3/-4); tests/v1/determinism/utils.py (+14/-14)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Originally, if we are testing models like `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`, we are actually running a MLA model  for MLA backend, which is not we want. Now this PR fixes the issue

### L3-63f49b8bd4  (L3, 2026-03-22, sha 63f49b8bd46b, PR #35162)
TITLE: [Model Runner V2] Enable piecewise CUDA graphs for pipeline parallelism (#35162)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/cudagraph_utils.py (+54/-15); vllm/v1/worker/gpu/model_runner.py (+48/-31)
LABELS: ready, v1, nvidia
BODY: ## Summary ⏎  ⏎ Add piecewise CUDA graph capture/replay support for PP in V2 model runner. ⏎  ⏎ Related: #33960 ⏎  ⏎ **model_runner.py:** ⏎ - Enable PP cudagraph mode handling ⏎ - Add `IntermediateTensors` buffer during graph replay ⏎ - Copy received tensors into the buffer at runtime ⏎  ⏎ **cudagraph_utils.py:** ⏎ - Add `intermediate_tensors` through capture pipeline ⏎ - Handle `IntermediateTensors` output on non-last PP ranks ⏎ - Fix `num_reqs` divisibility for uniform qu …[truncated]

### L3-f85e479e66  (L3, 2026-03-23, sha f85e479e6616, PR #35963)
TITLE: [Feature] ViT Full CUDA Graph (#35963)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/cudagraph/test_encoder_cudagraph.py (+451/-0); vllm/config/compilation.py (+32/-0); vllm/model_executor/models/interfaces.py (+141/-0); vllm/model_executor/models/qwen3_vl.py (+270/-30); vllm/v1/worker/gpu/mm/encoder_cudagraph.py (+576/-0); vllm/v1/worker/gpu/mm/encoder_cudagraph_defs.py (+66/-0); vllm/v1/worker/gpu_model_runner.py (+48/-1)
LABELS: performance, ready, v1, multi-modality, qwen, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Add full CUDA graph for the ViT to reduce kernel launch overheads. ⏎  ⏎ **Features:** ⏎   - **Budget-based graphs with a maximum batch size**: ⏎       - Capture CUDA graphs at configurable token budgets (e.g., `[256, 512, 1024, 2048, 4096]`). ⏎       - Pad sequence metadata (e.g. cu_seqlen) so that we can use the same budget-based graph for various number of images ⏎   during replays. ⏎   - **Greedy bin-packing**: ⏎       - Sort images in a …[truncated]

### L3-e99fb98867  (L3, 2026-03-23, sha e99fb98867c2, PR #36100)
TITLE: [ROCm] Fix fused_moe_fake signature mismatch and other AITER bugs (#36100)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+5/-4); vllm/_aiter_ops.py (+5/-1); vllm/model_executor/layers/quantization/quark/quark_moe.py (+1/-1); vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py (+5/-20)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study correctness case vllm:e99fb98867: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Summary ⏎  ⏎ Fix a real logic bug and several related issues in ROCm AITER ops: ⏎  ⏎ ### Main bugfix: `_rocm_aiter_fused_moe_fake` signature mismatch ⏎  ⏎ The fake/meta implementation for the `rocm_aiter_fused_moe` custom op was missing 4 parameters (`hidden_pad`, `intermediate_pad`, `bias1`, `bias2`) that the real implementation has. Since `direct_register_custom_op` derives the op schema from the impl's signature, callers can pass these arguments to the …[truncated]

### L3-e85f8f0932  (L3, 2026-03-23, sha e85f8f0932f6, PR #36728)
TITLE: [Bug][MoE] Strengthen _supports_current_device() checks in the TRTLLM FP8, NVFP4, and FlashInfer CuteDSL MoE experts (#36728)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+1/-1); vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py (+6/-1); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+6/-2); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+6/-1)
LABELS: bug, documentation, rocm, frontend, ready, ci/build, v1, multi-modality, tool-calling, llama
BODY: ## Purpose ⏎  ⏎ Strengthened `_supports_current_device()` checks in TRTLLM FP8/NvFP4/BF16 and FlashInfer CuteDSL MoE experts to verify FlashInfer runtime availability before selecting those backends. This helps preventing crashes on platforms where FlashInfer TRTLLM or CuteDSL kernels are not installed. ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ pytest -s -v tests/kernels/moe/test_unquantized_backend_selection.py ⏎ pytest -s -v tests/kernels/moe/test_cutedsl_moe.py ⏎ pytest -s  …[truncated]

### L3-dc6908ac6a  (L3, 2026-03-23, sha dc6908ac6a18, PR #35007)
TITLE: [Bugfix] Register VLLM_BATCH_INVARIANT in envs.py to fix spurious unknown env var warning (#35007)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fa_utils, L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.unified_attention, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashattn, L3.flex_attention
FILES: vllm/model_executor/layers/attention/attention.py (+1/-2); vllm/model_executor/layers/attention/mla_attention.py (+2/-3); vllm/utils/flashinfer.py (+1/-4); vllm/v1/attention/backends/fa_utils.py (+2/-2); vllm/v1/attention/backends/flash_attn.py (+5/-7); vllm/v1/attention/backends/flashinfer.py (+2/-5); vllm/v1/attention/backends/flex_attention.py (+2/-4); vllm/v1/attention/backends/mla/flashattn_mla.py (+3/-5); vllm/v1/attention/backends/mla/flashmla.py (+2/-4); vllm/v1/attention/backends/mla/triton_mla.py (+2/-4); (+20 more)
LABELS: bug, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ Register `VLLM_BATCH_INVARIANT` in `envs.py` to suppress spurious warning. ⏎  ⏎ `VLLM_BATCH_INVARIANT` is read via `os.getenv()` in `batch_invariant.py` but is never registered in `envs.py`'s `environment_variables` dict. This causes `validate_environ()` to emit an "Unknown vLLM environment variable detected" warning every time the feature is used: ⏎  ⏎ ``` ⏎ WARNING 02-20 23:18:09 [envs.py:1656] Unknown vLLM environment variable detected: VLLM …[truncated]

### L3-c59a132f96  (L3, 2026-03-23, sha c59a132f9674, PR #37487)
TITLE: [V0 Deprecation] Refactor kv cache from list to element (#37487)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa, L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/attention.py (+2/-5); vllm/model_executor/layers/attention/mla_attention.py (+3/-8); vllm/model_executor/layers/attention/static_sink_attention.py (+1/-1); vllm/v1/attention/backends/rocm_aiter_fa.py (+3/-7); tests/compile/passes/test_fusion_attn.py (+1/-1); tests/compile/passes/test_rope_kvcache_fusion.py (+3/-3); tests/v1/e2e/general/test_mamba_prefix_cache.py (+3/-3); tests/v1/worker/test_gpu_model_runner.py (+17/-17); tests/v1/worker/test_utils.py (+10/-10); vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py (+2/-4); (+17 more)
LABELS: rocm, ready, v1, qwen, deepseek, kv-connector
BODY: ## Purpose ⏎  ⏎ A follow up for https://github.com/vllm-project/vllm/pull/37195 of removing the virtual engine, this PR further refactor the kv cache from list to element to clean the code ⏎  ⏎ Tests in CI

### L3-c07e2ca6e0  (L3, 2026-03-24, sha c07e2ca6e0ac, PR #37728)
TITLE: Fix Mamba state corruption from referencing stale block table entries (#37728) (#37728) (#37728)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/block_table.py (+10/-0); vllm/v1/worker/gpu_input_batch.py (+1/-0); vllm/v1/worker/gpu_model_runner.py (+6/-0)
LABELS: ready, v1, nvidia, meta-exported, fb-exported
BODY: Summary: ⏎ we saw zero-token-id response for a linear attention model. This diff fixes it. ⏎  ⏎ Root cause is due to using stale mamba block, and this is triggered by DP dummy_run. It happens when one rank finishes a batch while other ranks are still running.  dummy_run(1) generates seq_len of [1,0,0,0..] to match other ranks' num_decode. The seq len 0 value indirectly maps to a stale mamba block gdn_attn.py. This padding is only needed for DP full cud …[truncated]

### L3-0f0e03890e  (L3, 2026-03-24, sha 0f0e03890ed7, PR #37233)
TITLE: [UX] Add flashinfer-cubin as CUDA default dep (#37233)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+2/-4); requirements/cuda.txt (+1/-0)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ Since we rely on cubins now for most Blackwell deployments, it makes sense to consider making `flashinfer-cubin` a default package. It is roughly ~250MB so not the worst bloat for other CUDA arches. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-8c47fdfdb1  (L3, 2026-03-24, sha 8c47fdfdb17c, PR #37692)
TITLE: [FlexAttention] allow custom mask mod (#37692)
SOURCES: path_core
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+70/-16); tests/kernels/test_flex_attention.py (+51/-0)
LABELS: ready, v1
BODY: updating FlexAttention impl to accept custom mask mod from users

### L3-42e9547976  (L3, 2026-03-25, sha 42e95479761a, PR #37640)
TITLE: [ROCm][Test] Fix ROCM_AITER_UNIFIED_ATTN attn+quant fusion test (#37640)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/passes/test_fusion_attn.py (+7/-1)
LABELS: rocm, ready
BODY: ## Purpose ⏎ fixes the block size used for ROCM_AITER_UNIFIED_ATTN in the FP8 attention+quant fusion test, following #37606. The backend requires block_size=64 but the test was using 16. ⏎  ⏎ ## Test Plan ⏎ ```bash ⏎ pytest -s tests/compile/passes/test_fusion_attn.py ⏎ ``` ⏎ ## Test Result ⏎ ```log ⏎ ================== 24 passed, 7 warnings in 122.79s (0:02:02) ================== ⏎ ``` ⏎  ⏎ --- ⏎ [details omitted]

### L3-09c3dc9186  (L3, 2026-03-25, sha 09c3dc91862e, PR #37968)
TITLE: [Revert] Remove CUDA torch fallbacks for fp8_mqa_logits/fp8_paged_mqa_logits_torch function (#37968)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+2/-1); vllm/model_executor/layers/sparse_attn_indexer.py (+22/-50); vllm/utils/deep_gemm.py (+0/-121)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 35271 reason=other
BODY: ## Purpose ⏎  ⏎ Revet https://github.com/vllm-project/vllm/pull/35271 ⏎ The original PR #35271 was intended to allow **dsv3.2** to run even when **deep_gemm** is not installed or on lower-end GPUs such as **A800**. ⏎  ⏎ @youkaichao believes that if the model vendor itself does not support the hardware, we should clearly state that it is **not supported**. Simply making it run doesn’t really add value. ⏎  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [detai …[truncated]

### L3-4a76ad12e0  (L3, 2026-03-25, sha 4a76ad12e001, PR #37725)
TITLE: [Bugfix] Preserve CUDA arch suffix (a/f) for SM12x — fixes NVFP4 NaN on desktop Blackwell (#37725)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+2/-2); cmake/utils.cmake (+4/-2)
LABELS: bug, ready, ci/build, nvidia
BODY: ## Summary ⏎  ⏎ vLLM's cmake build strips the architecture-specific suffix (`a`/`f`) from CUDA gencode flags, causing SM12x (DGX Spark, RTX 5090) to compile as plain `sm_120` instead of `sm_120a`/`sm_121a`. ⏎  ⏎ Without the suffix, `__CUDA_ARCH_FAMILY_SPECIFIC__` is not defined. This causes NVIDIA's own `cuda_fp4.hpp` (via `__CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__`) to disable the native `cvt.rn.satfinite.e2m1x2.f32` PTX instruction and fa …[truncated]

### L3-e7221180e1  (L3, 2026-03-25, sha e7221180e1d6, PR #37970)
TITLE: [Kernel] Optimize SM120 CUTLASS blockwise FP8 GEMM (#37970)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh (+36/-5)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎ The SM120 blockwise FP8 GEMM dispatch used a single `Shape<_128, _128, _128>` / `KernelScheduleAuto` configuration for all problem sizes , leaving significant performance on the table during decode (small-M) workloads. ⏎  ⏎ To address this, this PR splits the dispatch logic based on the $M$ dimension: ⏎   - **Small/medium M** (`M<=256`): `Shape<_64,_128,_128>` + `KernelTmaWarpSpecializedPingpong` — reduces tile overhead, improves SM util …[truncated]

### L3-bf4cc9ed2d  (L3, 2026-03-25, sha bf4cc9ed2d1f, PR #36058)
TITLE: [2/n] Migrate per_token_group_quant to torch stable ABI (#36058)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flash_attn.fork_inline_cmake
FILES: csrc/cache_kernels.cu (+2/-1); CMakeLists.txt (+5/-4); csrc/layernorm_kernels.cu (+1/-1); csrc/layernorm_quant_kernels.cu (+1/-1); csrc/libtorch_stable/dispatch_utils.h (+60/-0); csrc/libtorch_stable/ops.h (+21/-0); csrc/libtorch_stable/quantization/vectorization.cuh (+2/-2); csrc/libtorch_stable/quantization/vectorization_utils.cuh (+0/-0); csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu (+50/-46); csrc/libtorch_stable/quantization/w8a8/int8/per_token_group_quant.cu (+12/-0); (+12 more)
LABELS: ready, ci/build, nvidia
BODY: [See https://github.com/vllm-project/vllm/pull/36058/changes/c3bd0530831a06e0f5298c34e18e66e3e58a5644 to view the changes in this PR](https://github.com/vllm-project/vllm/pull/36058/changes/c3bd0530831a06e0f5298c34e18e66e3e58a5644) (otherwise github renders moved files with more than 50% changed lines as deleted and then added) ⏎  ⏎ ## Purpose ⏎ https://github.com/vllm-project/vllm/issues/26946 ⏎  ⏎ Stacked on https://github.com/vllm-project/vllm/pull …[truncated]

### L3-678b3c99e8  (L3, 2026-03-25, sha 678b3c99e82e, PR #38050)
TITLE: [MoE Kernel] Flashinfer nvfp4 cutedsl moe kernel integration (#38050)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+18/-0); tests/kernels/moe/test_cutedsl_moe.py (+1/-1); vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_batched_moe.py (+353/-0); vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py (+64/-244); vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (+46/-2); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+95/-1)
LABELS: ready, nvidia
DEEP_STUDY: deep-study: this PR was reverted by PR 38169 (confirmed_revert, reason=crash_or_hang)
BODY: ## Purpose ⏎ Kernel [Link](https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/fused_moe/cute_dsl) ⏎  ⏎ ## Test Plan ⏎ ```bash ⏎ vllm serve nvidia/Kimi-K2.5-NVFP4 \ ⏎   -dp 4 -ep --port 8000 --trust-remote-code \ ⏎   --moe-backend=flashinfer_cutedsl \ ⏎   --max-model-len 9216 --language-model-only --kv-cache-dtype fp8  ⏎ ``` ⏎  ⏎ gsm8k ⏎ ``` ⏎ lm_eval --model local-completions \ ⏎   --model_args "model=nvidia/Kimi-K2.5-NVFP4,base_url=http://0.0.0. …[truncated]

### L3-189ddefbfd  (L3, 2026-03-25, sha 189ddefbfdf7, PR #36702)
TITLE: [ROCm] Attention selector reordering (#36702)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.triton.chunked_prefill_paged_decode, L3.rocm.v1_rocm_attn, L3.platform.rocm_selection
FILES: vllm/envs.py (+0/-5); vllm/platforms/rocm.py (+8/-23); vllm/v1/attention/backends/rocm_attn.py (+4/-1); vllm/v1/attention/ops/chunked_prefill_paged_decode.py (+7/-0); .buildkite/scripts/hardware_ci/run-amd-test.sh (+1/-1); docs/design/attention_backends.md (+1/-1); tests/v1/attention/test_rocm_attention_backends_selection.py (+6/-23); vllm/_aiter_ops.py (+2/-2)
LABELS: documentation, rocm, ready, ci/build, v1
BODY: With the unit tests now able to handle this change following ⏎ #36025 #35334 and others ⏎ Changing the priorities of ROCm attention backends to ⏎ 1. ROCM_ATTN - when applicable it is the most performant backend today ⏎ 2. AITER_MHA - when explicitly selected ⏎ 3. AITER_UNIFIED - a variation of TRITON_ATTN, specifically tuned for ROCm. Will fall back here when the model requires sinks (GPT-OSS) ⏎ 4. TRITON_ATTN - the most versatile and as of now the least per …[truncated]

### L3-14771f7150  (L3, 2026-03-25, sha 14771f715085, PR #37143)
TITLE: [XPU] support MLA model on Intel GPU (#37143)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+4/-0); vllm/platforms/xpu.py (+0/-12); vllm/_xpu_ops.py (+1/-0); vllm/model_executor/layers/quantization/input_quant_fp8.py (+10/-0)
LABELS: ready
BODY: ## Purpose ⏎ before this PR, we can enable MLA model by  `export VLLM_MLA_DISABLE=1`, which will always fall back to MHA backend.  ⏎ this PR will use FLASH_ATTN for prefill and TRITON_MLA for decode. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ python3 examples/basic/offline_inference/generate.py --model deepseek-ai/DeepSeek-V2-Lite  --enforce-eager --temperature 0  -tp 2   ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-52069012fe  (L3, 2026-03-26, sha 52069012fe53, PR #38083)
TITLE: [Bugfix] Fix DeepGemm E8M0 accuracy degradation for Qwen3.5 FP8 on Blackwell (#38083)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml (+2/-1); tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml (+2/-1); tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2.yaml (+9/-0); tests/evals/gsm8k/configs/models-qwen35-blackwell.txt (+2/-0); tests/evals/gsm8k/test_gsm8k_correctness.py (+4/-6); vllm/config/vllm.py (+19/-0); vllm/model_executor/layers/quantization/fp8.py (+7/-2); vllm/model_executor/layers/quantization/input_quant_fp8.py (+1/-0); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+5/-1); vllm/utils/deep_gemm.py (+18/-0)
LABELS: bug, ready, qwen
ISSUES: #37804 [Bug] DeepGemm E8M0 scale format causes accuracy degradation for Qwen3.5 FP8 on Blackwell
DEEP_STUDY: deep-study correctness case vllm:52069012fe: class=numerical_precision; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Summary ⏎  ⏎ Fixes #37804 — `Qwen/Qwen3.5-35B-A3B-FP8` accuracy drops ~12pp on Blackwell (B200) when DeepGemm is active. DeepGemm's mandatory E8M0 (power-of-2 ceiling) scale format loses precision per FP8 block-quantized layer, compounding across all ~80 attention-projection and shared-expert FFN layers in Qwen3.5. ⏎  ⏎  ⏎ ## Changes ⏎  ⏎ - **Auto-disable DeepGemm for Qwen3.5 on Blackwell.** Adds a model-type exclusion list (`qwen3_5_text`, `qwen3_5_ …[truncated]

### L3-87f05d6880  (L3, 2026-03-26, sha 87f05d6880d0, PR #38076)
TITLE: [Revert] Remove DeepGEMM availability check in DeepseekV32IndexerMetadataBuilder (#38076)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+0/-7)
LABELS: ready, v1, deepseek
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 36519 reason=unstated
BODY: ## Purpose ⏎  ⏎ revert https://github.com/vllm-project/vllm/pull/36519 ⏎  ⏎  ⏎ see https://github.com/vllm-project/vllm/pull/37968 ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-978fc18bf0  (L3, 2026-03-26, sha 978fc18bf000, PR #36574)
TITLE: [ROCm] Utilize persistent MLA kernel from AITER (#36574)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+107/-2); vllm/_aiter_ops.py (+47/-0)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎ Add support for aiter's persistent mode MLA decode kernel on ROCm. The persistent ⏎ kernel stays resident on GPU CUs and processes work items from pre-computed ⏎ scheduling metadata, avoiding per-batch kernel launch overhead. ⏎ - Pre-allocates six persistent scheduling buffers (`work_meta_data`, `work_indptr`, ⏎   `work_info_set`, `reduce_indptr`, `reduce_final_map`, `reduce_partial_map`) during ⏎   `AiterMLAMetadataBuilder` initialization …[truncated]

### L3-be1a85b7a2  (L3, 2026-03-26, sha be1a85b7a292, PR #38169)
TITLE: Revert "[MoE Kernel] Flashinfer nvfp4 cutedsl moe kernel integration" (#38050) (#38169)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+0/-18); tests/kernels/moe/test_cutedsl_moe.py (+1/-1); vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_batched_moe.py (+0/-353); vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py (+244/-64); vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (+2/-46); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+1/-95)
LABELS: ci-failure, nvidia
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 38050 reason=crash_or_hang
BODY: ## Revert of https://github.com/vllm-project/vllm/pull/38050 ⏎  ⏎ This reverts commit 678b3c99e82e1b1dd6cc95ff98c114393b788be4. ⏎  ⏎ ### Reason ⏎ The original PR caused **1 new CI failure** in nightly build [#58103](https://buildkite.com/vllm/ci/builds/58103): ⏎ - **Kernels (B200)**: CUDA coredump + `Fatal Python error: Aborted` in `test_flashinfer_cutedsl_moe_masked` (`tests/kernels/moe/test_cutedsl_moe.py`) ⏎  ⏎ The crash occurs in `break_fp4_bytes` / `dequant …[truncated]

### L3-a4cf9b22ba  (L3, 2026-03-26, sha a4cf9b22ba8f, PR #37228)
TITLE: [ROCM][Bugfix] Use correct stride in cp_mha_gather_cache_kernel for hybrid model (#37228) (#37228)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+25/-6)
LABELS: bug, rocm, ready, v1, meta-exported, fb-exported
BODY: **Problem** ⏎ [_update_hybrid_attention_mamba_layout](https://github.com/vllm-project/vllm/blob/e5b807607c8493155e6eccd665772d4c19b2114e/vllm/v1/worker/gpu_model_runner.py#L6359) uses as_strided_() to reorder KV blocks into a interleaved pattern. But cp_mha_gather_cache_kernel Triton kernel used hardcoded pointer arithmetic assuming contiguous memory. For hybrid models, this reads from wrong memory locations, producing garbage values and NaN in att …[truncated]

### L3-cb2263218e  (L3, 2026-03-26, sha cb2263218e66, PR #35886)
TITLE: [Bugfix][Minor] Fix potential NameError in mamba backend selector and misc typos (#35886)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+4/-1); vllm/v1/attention/backends/utils.py (+1/-1); vllm/v1/attention/selector.py (+2/-2); vllm/model_executor/models/kimi_k25.py (+1/-1); vllm/v1/attention/backends/rocm_aiter_fa.py (+1/-1)
LABELS: bug, rocm, ready, v1
BODY: ## Summary ⏎  ⏎ Fix a potential `NameError` bug in the mamba attention backend selector and several minor typos/grammar issues across the codebase. ⏎  ⏎ ### Bug fix ⏎  ⏎ In `_cached_get_mamba_attn_backend` (`vllm/v1/attention/selector.py`), if `MAMBA_TYPE_TO_BACKEND_MAP[mamba_type]` raises a `KeyError`, the variable `backend_name` is never assigned. The `except` handler then references `backend_name` in the error message, which would raise a `NameError` inst …[truncated]

### L3-0aac2048bf  (L3, 2026-03-26, sha 0aac2048bf3a, PR #35175)
TITLE: [Bugfix] Restore CUDA graph persistent buffers for FP8 FlashMLA decode (#35175)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_v1_adapter
FILES: vllm/v1/attention/backends/mla/flashmla.py (+15/-0)
LABELS: bug, ready, v1, nvidia
ISSUES: #33638 [Bug]: DeepSeekV3.1 with fp8 kvcache in v0.15.0 produces garbled output
BODY: ## Purpose ⏎  ⏎ Fix #33638: DeepSeek-V3.1 with `--kv-cache-dtype fp8` produces garbled output in v0.15.0. ⏎  ⏎ The PR #32810 refactored the FlashMLA interface to use `FlashMLASchedMeta` objects. The non-FP8 path was adapted correctly - `FlashMLASchedMeta` manages its own internal buffers for CUDA graph safety. However, the FP8 path calls `get_mla_metadata_dense_fp8()` which allocates fresh raw tensors every call, and these were set directly on the `F …[truncated]

### L3-f26fcdfb9e  (L3, 2026-03-26, sha f26fcdfb9e50, PR #37547)
TITLE: [Bugfix][ROCm] Fix lru_cache on paged_mqa_logits_module (#37547)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+39/-38)
LABELS: bug, rocm, ready, v1
BODY: Fix lru_cache on paged_mqa_logits_module by moving to module scope ⏎  ⏎ ## Purpose ⏎  ⏎ The function `paged_mqa_logits_module ` was defined inside `rocm_fp8_paged_mqa_logits`, causing a new function object (and thus a new empty cache) to be created on every call, defeating the purpose of lru_cache. ⏎ As a consequence, the module was reimported on each call. ⏎  ⏎ The issue is in the ROCm Sparse MLA implementation, affecting DeepSeek v3.2. ⏎  ⏎ ## Test Plan ⏎  ⏎ Benchma …[truncated]

### L3-0904b6550d  (L3, 2026-03-26, sha 0904b6550d8e, PR #38136)
TITLE: Fix multi-node allreduce fusion (#38136)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/distributed/device_communicators/flashinfer_all_reduce.py (+61/-2); vllm/envs.py (+2/-7)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎ The flashinfer trtllm allreduce backend does not work for multi-node setup (See issue https://github.com/flashinfer-ai/flashinfer/issues/2006). As a result, running allreduce fusion results in hang in multi-node. This PR resolves this by auto-selecting mnnvl backend for flashinfer all reduce in multi-node setup. In single node setup, the default backend remains trtllm due to issue #35772 ⏎  ⏎ ## Test Plan ⏎ Test with nvidia/DeepSeek-R1-N …[truncated]

### L3-98e7f223b9  (L3, 2026-03-27, sha 98e7f223b9fb, PR #33695)
TITLE: enable skipping of SW attention layers when using FP8 KV cache (#33695)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/attention.py (+25/-0); tests/quantization/test_fp8.py (+23/-0); vllm/config/cache.py (+3/-0); vllm/engine/arg_utils.py (+7/-0)
LABELS: ready, quantization
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎ This PR enables us to keep Sliding Window Attention layers in BF16, whilst quantizing the Full Attention layers. The idea is that there is not much latency / memory to be saved in SW layers, but the quantization overheads are paid nonetheless. Furthermore, from an accuracy perspective, skipping the SW layers is a more conservative approach and does minimize the risks of accuracy degradation. ⏎ We thus expect slight ITL gains. ⏎  ⏎ It's the l …[truncated]

### L3-7cc302dd87  (L3, 2026-03-27, sha 7cc302dd87cb, PR #37853)
TITLE: [kv_offload+HMA][7/N]: Support register_kv_caches for hybrid models (#37853)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+2/-1); tests/v1/kv_connector/unit/offloading_connector/__init__.py (+0/-0); tests/v1/kv_connector/unit/offloading_connector/conftest.py (+7/-0); tests/v1/kv_connector/unit/offloading_connector/test_metrics.py (+151/-0); tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py (+341/-0); tests/v1/kv_connector/unit/offloading_connector/test_worker.py (+504/-0); tests/v1/kv_connector/unit/offloading_connector/utils.py (+15/-487); tests/v1/kv_connector/unit/test_nixl_connector.py (+3/-0); tests/v1/kv_offload/test_cpu_gpu.py (+88/-127); vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py (+227/-15); (+3 more)
LABELS: ready, v1, kv-connector
BODY: This PR extends the offloading connector `register_kv_caches` function to support KV caches used in hybrid models. ⏎  ⏎ We define a new `CanonicalKVCaches` class which captures: ⏎ 1. The unique set of KV cache tensors (as tensors maybe shared by multiple layers) ⏎ 2. Mapping each group to its relevant KV cache data (given by a tensor pointer + page size). The canonical tensors are each of dtype int8 and shape `(num_blocks, page_size)`. ⏎  ⏎ This PR als …[truncated]

### L3-384e4d5f48  (L3, 2026-03-27, sha 384e4d5f48ce, PR #38311)
TITLE: [Model Runner V2] Rebuild attention metadata before eagle decode full… (#38311)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/spec_decode/eagle/speculator.py (+97/-55)
LABELS: ready, v1, mrv2
BODY: # Purpose ⏎ This PR addresses the low quality draft tokens produced at positions > 0 (after the draft prefill) that result from not rebuilding the attention metadata during FULL cudagraph. Rebuilding is necessary to update the state of the attention metadata builders/backend so that it doesn't contain stale values from previous runs. Doing so seems to improve the acceptance rates of draft tokens at positions > 0, as shown in the HTML link below. ⏎  …[truncated]

### L3-97d19197bc  (L3, 2026-03-27, sha 97d19197bcd0, PR #38126)
TITLE: [NVIDIA] Fix DGX Spark logic (#38126)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+6/-6); cmake/external_projects/qutlass.cmake (+4/-4); cmake/utils.cmake (+37/-2)
LABELS: ready, ci/build, nvidia
BODY: ## [Bugfix] Add SM121 (12.1) arch guards for desktop Blackwell, fixes missing NVFP4/scaled_mm/MLA kernels on DGX Spark ⏎  ⏎ ### Summary ⏎  ⏎ PR #37725 correctly preserved the CUDA arch suffix (`a`/`f`) during gencode extraction, but the downstream arch guard checks in `CMakeLists.txt` only recognize `12.0a`/`12.0f` — they don't match `12.1`/`12.1a`/`12.1f` (SM121/DGX Spark). This causes all SM12x-family kernels to be silently skipped when building wi …[truncated]

### L3-88149b635e  (L3, 2026-03-27, sha 88149b635e3b, PR #31201)
TITLE: Add nvidia h800 moe config (#31201)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/configs/E=128,N=192,device_name=NVIDIA_H800,dtype=fp8_w8a8.json (+147/-0); vllm/model_executor/layers/fused_moe/configs/E=128,N=384,device_name=NVIDIA_H100_80GB_HBM3.json (+147/-0)
LABELS: stale, nvidia
BODY: ## Purpose ⏎  ⏎ We get this result in NVIDIA H800 device use follower command  ⏎  ⏎ ```python ⏎ python3 /vllm-workspace/benchmarks/kernels/benchmark_moe.py --model /data/models/Qwen/Qwen3-235B-A22B --tensor-parallel-size 8 --dtype fp8_w8a8 --tune ⏎ ``` ⏎  ⏎ - H100 ⏎ ``` ⏎ python3 ./benchmarks/kernels/benchmark_moe.py --model  /home/jovyan/qwen3-30b-a3b --tensor-parallel-size 2 --tune ⏎ ``` ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-2bf5b70ae8  (L3, 2026-03-28, sha 2bf5b70ae862, PR #38391)
TITLE: [CI Bugfix] Pre-download missing FlashInfer headers in Docker build (#38391)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+19/-0)
LABELS: bug, ready, ci/build, ci-failure
ISSUES: #38110 [Bug]: `flashinfer-cubin` does not include all cubins/headers
BODY: ## Summary ⏎ - **NOTE: This is a temporary fix to unblock our CI** - the proper fix should be upstream in flashinfer, hopefully https://github.com/flashinfer-ai/flashinfer/pull/2903 ⏎ - Pre-download FlashInfer TRTLLM BMM headers at Docker build time to fix air-gapped/offline runtime failures ⏎ - The `flashinfer-cubin` package ships headers at the artifact hash path (`cubins/b55211623.../include/trtllmGen_bmm_export/`), but runtime code (`download_tr …[truncated]

### L3-43cc5138e5  (L3, 2026-03-28, sha 43cc5138e514, PR #38450)
TITLE: [ROCm][CI] Fix cross-attention dispatch for encoder-decoder models (#38450)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+22/-6); vllm/v1/attention/backends/rocm_aiter_fa.py (+6/-5); vllm/v1/attention/backends/rocm_attn.py (+7/-2); docs/design/attention_backends.md (+2/-2); tests/entrypoints/openai/speech_to_text/test_transcription_validation_whisper.py (+52/-3); tools/pre_commit/generate_attention_backend_docs.py (+1/-1)
LABELS: documentation, rocm, ready, v1
DEEP_STUDY: deep-study correctness case vllm:43cc5138e5: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: Cross-attention layers in encoder-decoder models (Whisper, BART, etc.) produce incorrect beam search results on `ROCM_ATTN` and `ROCM_AITER_FA`. This PR removes `ENCODER_DECODER` from their `supports_attn_type` so the backend ⏎ dispatch selects a working backend for cross-attention layers instead. ⏎  ⏎ ### Motivation ⏎  ⏎ The test `test_whisper_beam_search_single_beam` fails on ROCm because single-beam beam search does not match greedy decoding. The m …[truncated]

### L3-b4a2f3ac36  (L3, 2026-03-30, sha b4a2f3ac3690, PR #38423)
TITLE: [NVIDIA] Bugfix NVFP4 DGX Spark and RTX50 (#38423)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1); docker/Dockerfile (+4/-1); docker/Dockerfile.nightly_torch (+5/-2); docker/versions.json (+1/-1); requirements/cuda.txt (+2/-2); csrc/quantization/fp4/nvfp4_quant_entry.cu (+29/-0); csrc/quantization/fp4/nvfp4_scaled_mm_entry.cu (+13/-1); csrc/quantization/machete/machete_mainloop.cuh (+1/-0); tests/kernels/moe/test_unquantized_backend_selection.py (+0/-4); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+5/-0); (+5 more)
LABELS: bug, ready, ci/build, nvidia, ready-run-all-tests
BODY: ## Summary ⏎  ⏎ Fix `cudaErrorIllegalInstruction` when running NVFP4 models (e.g. `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`) on SM12x GPUs (RTX 50 series SM120, DGX Spark SM121). ⏎  ⏎ ### Root causes ⏎  ⏎ 1. **CUTLASS v4.2.2 lacks SM12x NVFP4 tile constraints** — The bundled CUTLASS was missing SM120f family-level compilation support for NVFP4/MX Grouped GEMM and SM121-specific tile configurations (DGX Spark). This caused `IllegalInstruction` durin …[truncated]

### L3-2c734ed0e0  (L3, 2026-03-30, sha 2c734ed0e06a, PR #38562)
TITLE: [Bugfix][MLA] Change default SM100 MLA prefill backend back to TRT-LLM (#38562)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/attention.py (+1/-1)
LABELS: bug, ready
ISSUES: #36763 [Bug]: Kimi-K2.5 outputs only '!!!!!!!!!!' in reasoning field, content is always null
BODY: FIX: #36763 ⏎  ⏎ ## Purpose ⏎ On SM100, FA4 MLA prefill appears to cause unusable output on Kimi-K2.5. This PR changes the default MLA prefill backend back to TRTLLM while we resolve the issues with FA4. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-8e6293e838  (L3, 2026-03-30, sha 8e6293e838f9, PR #35753)
TITLE: [Mamba] Add stochastic rounding support (#35753)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/mamba/test_mamba_ssm.py (+54/-0); vllm/config/cache.py (+35/-1); vllm/engine/arg_utils.py (+13/-0); vllm/model_executor/layers/mamba/mamba_mixer.py (+2/-0); vllm/model_executor/layers/mamba/mamba_mixer2.py (+2/-0); vllm/model_executor/layers/mamba/ops/mamba_ssm.py (+58/-1); vllm/model_executor/models/plamo2.py (+2/-0)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ Add stochastic rounding support in SSM's selective state update kernel. ⏎  ⏎ ## Test Plan ⏎  ⏎ Add tests comparing the selective_state_update kernel's output with stochastic rounding enabled with the reference implementation in FP32, with some tolerance. ⏎ Add e2e lm-eval test for Nemotron 3 Nano with stochastic rounding enabled. ⏎  ⏎ ## Test Result ⏎  ⏎ New tests pass, e2e test passes with same thresholds as existing test that has stochastic rounding  …[truncated]

### L3-bcc6f67447  (L3, 2026-03-30, sha bcc6f67447e9, PR #35431)
TITLE: [Bugfix] Use null block (0) for padded block table entries (#35431)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/mla/indexer.py (+0/-6); vllm/v1/attention/backends/utils.py (+1/-0); csrc/mamba/mamba_ssm/selective_scan.h (+1/-1); csrc/mamba/mamba_ssm/selective_scan_fwd.cu (+15/-7); csrc/ops.h (+1/-1); csrc/torch_bindings.cpp (+1/-1); tests/kernels/mamba/test_causal_conv1d.py (+24/-16); tests/kernels/mamba/test_mamba_ssm.py (+29/-22); vllm/_custom_ops.py (+2/-2); vllm/model_executor/layers/lightning_attn.py (+5/-2); (+5 more)
LABELS: bug, ready, v1, ready-run-all-tests
ISSUES: #33664 [Bug]: DeepSeek-V3.1 with fp8 KV Cache causes illegal memory access at concurrency ≥ 5 in `serve_benchmark` | #35336 [Refactor]: Make SSM backends use the null block (0) for padded requests instead of -1
BODY: FIX #35336 ⏎ FIX #33664 ⏎  ⏎ Alternative to #35969  ⏎  ⏎ Block table CUDAGraph padding now uses `0` (null block) instead of `-1` (`PAD_SLOT_ID`), aligning SSM/Mamba backends with the null block convention. ⏎  ⏎   - `PAD_SLOT_ID = -1` is for **slot mappings** (indexing individual slots), not block tables (indexing blocks) ⏎   - Block 0 is already reserved as the null block via `free_block_queue.popleft()` with `is_null = True` ⏎   - Per @WoosukKwon: "PAD_S …[truncated]

### L3-494636b29d  (L3, 2026-03-30, sha 494636b29d3b, PR #36847)
TITLE: [Feat][Spec Decode] DFlash (#36847)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+14/-0); vllm/v1/attention/backends/flash_attn.py (+4/-0); vllm/v1/attention/selector.py (+9/-1); tests/models/registry.py (+8/-0); tests/v1/e2e/spec_decode/test_spec_decode.py (+166/-6); tests/v1/spec_decode/test_eagle.py (+152/-3); vllm/config/speculative.py (+21/-6); vllm/config/vllm.py (+20/-0); vllm/model_executor/models/qwen3.py (+1/-0); vllm/model_executor/models/qwen3_dflash.py (+619/-0); (+7 more)
LABELS: new-model, speculative-decoding, ready, v1, qwen, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ ### Overview ⏎  ⏎ DFlash works much like P-EAGLE (see #32887), but with a major architectural change: it uses bidirectional attention between the query tokens (the last sampled token from the base model plus a bunch of placeholder mask tokens) and the context states, which are the target model's hidden states from the prefill or accepted tokens. ⏎  ⏎ To implement this, I introduce an extra operation that lives outside of the main model  …[truncated]

### L3-85c0950b1f  (L3, 2026-03-30, sha 85c0950b1f64, PR #37529)
TITLE: [ROCm] Enable MORI EP for unquantized MoE with AITER backend (#37529)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/all2all_utils.py (+12/-5); vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py (+12/-4)
LABELS: rocm, ready
BODY: ## Purpose ⏎ Fix silent degradation when using MORI EP (`--all2all-backend mori`) with unquantized (BF16) MoE models and the AITER expert backend on ROCm. ⏎  ⏎ Currently, `UnquantizedFusedMoEMethod.maybe_make_prepare_finalize()` returns `None` when AITER is the backend, which silently skips MORI dispatch/combine. Tokens never get sent to remote GPUs, each GPU only runs its local experts, dropping ~87.5% of expert contributions on EP8. The model appe …[truncated]

### L3-3b1dbaad4e  (L3, 2026-03-30, sha 3b1dbaad4e59, PR #37467)
TITLE: [HMA]Fix corner case when hybrid page_size can not be evenly divided issue (blk_size=64,tp=4) (#37467)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+4/-0); vllm/v1/attention/backends/flash_attn.py (+7/-0); tests/v1/worker/test_gpu_model_runner.py (+2/-0); vllm/config/cache.py (+5/-0); vllm/model_executor/models/config.py (+7/-142); vllm/platforms/interface.py (+169/-28); vllm/platforms/xpu.py (+0/-10); vllm/v1/attention/backends/gdn_attn.py (+4/-0); vllm/v1/attention/backends/linear_attn.py (+4/-0); vllm/v1/attention/backends/mamba1_attn.py (+4/-0); (+2 more)
LABELS: intel-gpu, ready, v1
BODY: ## Purpose ⏎  ⏎ -- ⏎  ⏎ Issue description: ⏎  ⏎ Testing `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-bf16` with default block_size = 64. ⏎  ⏎ if FA block_size = 16 w/ TP4 => PASS: 540672 / 8192 = 66 ⏎ ``` ⏎ [kv_cache_utils.py:934] layer_spec.block_size=16, layer_spec.page_size_bytes=8192 ⏎ [kv_cache_utils.py:934] layer_spec.block_size=262144, layer_spec.page_size_bytes=540672 ⏎ ``` ⏎  ⏎ if FA block_size = 64 w/ TP4 => ERROR!!! SSM_page_size will not be evenly divide …[truncated]

### L3-4ac227222f  (L3, 2026-03-30, sha 4ac227222fc2, PR #36070)
TITLE: [Bugfix][DCP] Fix CUDA graph capture for Decode Context Parallelism (#36070)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+34/-8)
LABELS: bug, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ DCP produces incorrect results under FULL CUDA graph capture (the default decode mode in v1's `FULL_AND_PIECEWISE`). `dcp_context_kv_lens` is computed each step in `FlashAttentionMetadataBuilder.build()`, ⏎   so its tensor address gets baked into the graph and becomes stale on replay. ⏎  ⏎   ## Test Plan ⏎  ⏎ ```bash ⏎ pytest tests/distributed/test_context_parallel.py -v -x -k "Qwen and parallel_setup2" ⏎ ``` ⏎  ⏎   ## Test Result ⏎  ⏎ GB200 NVL72 (4 GPUs, …[truncated]

### L3-d9c7db18da  (L3, 2026-03-30, sha d9c7db18da98, PR #38381)
TITLE: [ROCm][CI] Pin test_hybrid test to TRITON_ATTN on ROCm (#38381)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/language/generation/test_hybrid.py (+11/-2)
LABELS: rocm, ready
BODY: After the attention backend priority reordering in https://github.com/vllm-project/vllm/pull/36702, we've seen some flakiness in `tests/models/language/generation/test_hybrid.py` (example: https://buildkite.com/vllm/amd-ci/builds/6991/steps/canvas?jid=019d2de0-ff54-4b0d-b038-c26e76e0d80c&tab=output). We pin these tests to TRITON_ATTN to mitigate batch variance effects and reduce flakiness on ROCm.

### L3-93b3ec1585  (L3, 2026-03-30, sha 93b3ec15859a, PR #36466)
TITLE: feat(attention): extract KV-cache update from FlashAttentionDiffKV ba… (#36466)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/flash_attn_diffkv.py (+34/-27)
LABELS: ready, v1
BODY: …ckend ⏎  ⏎ [Core] extract KV-cache update from FlashAttentionDiffKV backend ⏎  ⏎ ## Purpose ⏎   ⏎ Extract the KV-cache write out of `FlashAttentionDiffKVImpl.forward()` into a dedicated      ⏎   `do_kv_cache_update()` method, as part of issue #32335. ⏎  ⏎   - Added `FlashAttentionDiffKVImpl.do_kv_cache_update()` which calls ⏎   `triton_reshape_and_cache_flash_diffkv` directly on the combined KV cache tensor (no ⏎   `.unbind(0)` split, since DiffKV stores K and V conc …[truncated]

### L3-757068dc65  (L3, 2026-03-31, sha 757068dc65f6, PR #38556)
TITLE: [Bugfix][Async] Fix async spec decoding with hybrid models (#38556)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/spec_decode/test_backup_token_async_spec.py (+147/-0); tests/v1/spec_decode/test_eagle.py (+4/-12); tests/v1/spec_decode/test_extract_hidden_states.py (+2/-12); vllm/v1/spec_decode/eagle.py (+1/-2); vllm/v1/spec_decode/extract_hidden_states.py (+1/-2); vllm/v1/worker/gpu_model_runner.py (+22/-8)
LABELS: bug, speculative-decoding, ready, v1
ISSUES: #38098 [CI Failure]: LM Eval Large Models (H200)
BODY: co-authored by @SandishKumarHN  ⏎  ⏎ FIX: #38098 ⏎  ⏎ ## Purpose ⏎ Incorporates 2 fixes: ⏎  ⏎ ### Fix 1 ⏎ Posted earlier as #38419, incorporated into this PR. ⏎  ⏎ In async mode, `seq_lens_cpu` is inflated by optimistic draft token placeholders. When `prepare_next_token_ids_padded` uses this inflated value to call `get_token_id()`, it reads past the end of the committed tokens and returns -1. Use `num_tokens_no_spec - 1` (the actual last committed token po …[truncated]

### L3-7d65463528  (L3, 2026-03-31, sha 7d65463528a0, PR #38584)
TITLE: [WIP][CI][Bugfix] Fix `test_run_eagle_dp` (#38584)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+5/-2); tests/v1/distributed/test_eagle_dp.py (+1/-3)
LABELS: bug, ready, v1
ISSUES: #31913 [Bug]: test_eagle_dp test is flaky | #38234 Test Failure: test_run_eagle_dp[FLASH_ATTN] produces non-deterministic outputs with EAGLE speculative decoding
BODY: FIX: #38234 ⏎ FIX: #31913 ⏎ Revert: #31915 ⏎  ⏎ ## Purpose ⏎ Fixes flaky test by disabling AOT scheduling when `VLLM_BATCH_INVARIANT` is enabled. ⏎  ⏎ ## Test Plan ⏎ Distributed DP Tests (2 GPUs) ⏎ `pytest tests/v1/distributed/test_eagle_dp.py::test_run_eagle_dp[FLASH_ATTN]` ⏎  ⏎ ## Test Result ⏎ Should pass in CI ⏎  ⏎ --- ⏎ [details omitted]

### L3-598190aac3  (L3, 2026-03-31, sha 598190aac38a, PR #36540)
TITLE: [fix] Remove trtllm ragged mla prefills (#36540)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.merge.triton_lse, L3.merge.cuda_lse, L3.mla.common_v1
FILES: csrc/attention/merge_attn_states.cu (+48/-19); csrc/ops.h (+5/-6); csrc/torch_bindings.cpp (+2/-1); vllm/_custom_ops.py (+8/-1); vllm/model_executor/layers/attention/mla_attention.py (+14/-2); vllm/v1/attention/ops/merge_attn_states.py (+43/-2); vllm/v1/attention/ops/triton_merge_attn_states.py (+39/-2); tests/kernels/attention/test_merge_attn_states.py (+26/-2)
LABELS: ready, v1
BODY: ## Purpose ⏎ Removing output and workspace_buffer prefills for `run_prefill_context_chunk_trtllm_ragged`. This kernel reads and writes only tokens having history. Empty buffers may cause numeric issues for tokens w/o context. This PR modifies `merge_attn_states` to be aware of this. ⏎  ⏎ ## Test Plan ⏎ - [+]  `test_merge_attn_states` now accepts `prefills_with_context` ⏎ - [+] accuracy (`lm_eval`) ⏎ - [+] performance (`vllm bench serve`) ⏎  ⏎ ## Test Result ⏎ `VLL …[truncated]
