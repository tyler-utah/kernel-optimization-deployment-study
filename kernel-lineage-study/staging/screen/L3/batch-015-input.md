### L3-f6a708ab2b  (L3, 2026-06-05, sha f6a708ab2bfd, PR #44435)
TITLE: [Doc] Add Llama-3.2-3B-Instruct to batch-invariance tested models (#44435)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/features/batch_invariance.md (+2/-5)
LABELS: documentation, ready, llama
BODY: ## Purpose ⏎  ⏎ Adds `meta-llama/Llama-3.2-3B-Instruct` to the batch-invariance Tested Models list — the gap in the Llama 3.2 line (`3.1-8B` and `3.2-1B` were already listed). Discussed in #27433 (@yewentao256 gave the go-ahead). ⏎  ⏎ ## Validation ⏎  ⏎ Validated on an **RTX 4080 (Ada Lovelace, SM 8.9)** with `VLLM_BATCH_INVARIANT=1`, vLLM v0.22.0, default V1 model runner: ⏎  ⏎ | Test | Backend | Result | ⏎ | --- | --- | --- | ⏎ | `test_logprobs_bitwise_batch_invari …[truncated]

### L3-b593396c7a  (L3, 2026-06-05, sha b593396c7a70, PR #44621)
TITLE: Upgrade tpu-inference to v0.21.0 (#44621)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/tpu.txt (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ Upgrade tpu-inference to latest stable release v0.21.0 ⏎  ⏎ ## Test Plan ⏎ Verified on tpu-inference CI.  ⏎  ⏎ ## Test Result ⏎ Success.  ⏎  ⏎ --- ⏎ [details omitted]

### L3-a50e675b0d  (L3, 2026-06-05, sha a50e675b0d86, PR #44021)
TITLE: [Cohere] fix RoutingMethodType (#44021)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py (+2/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+1/-1); vllm/model_executor/layers/fused_moe/router/custom_routing_router.py (+4/-2)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎ Cohere models have optional renormalize in router function, fix the `RoutingMethodType` if the router doesn't have renormalize ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-7fe7800fa4  (L3, 2026-06-05, sha 7fe7800fa4ce, PR #43150)
TITLE: [BUG] Fix FP64 Gumbel precision coverage (#43150)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/sample/test_rejection_sampler.py (+64/-1); tests/v1/sample/test_topk_topp_sampler.py (+39/-1); tests/v1/spec_decode/test_eagle.py (+2/-1); tests/v1/spec_decode/test_llm_base_proposer_sampling.py (+70/-0); tools/gumbel_precision/prove_exponential_race_precision.py (+141/-0); vllm/config/model.py (+4/-3); vllm/v1/sample/ops/topk_topp_sampler.py (+36/-7); vllm/v1/sample/rejection_sampler.py (+13/-2); vllm/v1/sample/sampler.py (+7/-2); vllm/v1/spec_decode/llm_base_proposer.py (+11/-3); (+1 more)
LABELS: bug, speculative-decoding, ready, v1
BODY: The existing --use-fp64-gumbel flag only covered the explicit Triton Gumbel sampler. V1 sampling and spec decode also use the equivalent exponential-race form q.exponential_(); probs / q; argmax, so those paths still used fp32 exponential noise even when the precision flag was enabled. ⏎  ⏎ Thread use_fp64_gumbel through the Python V1 sampler, TopKTopPSampler, rejection sampler recovery sampling, and LLM draft proposer sampling. When enabled, these …[truncated]

### L3-f87df1df9e  (L3, 2026-06-05, sha f87df1df9e93, PR #44613)
TITLE: [Bugfix][MoE] Snapshot max_cudagraph_capture_size into FusedMoEConfig (#44613)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/config.py (+2/-0); vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py (+1/-4); vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py (+1/-5); vllm/model_executor/layers/fused_moe/layer.py (+1/-0); vllm/model_executor/layers/quantization/mxfp4.py (+2/-7)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎  ⏎ Several MoE kernels/quant-methods read ⏎ `get_current_vllm_config().compilation_config.max_cudagraph_capture_size` ⏎ inside their `__init__`: ⏎  ⏎ - `FlashInferExperts.__init__` (`fused_moe/experts/flashinfer_cutlass_moe.py`) ⏎ - `TrtLlmMxfp4ExpertsBase.__init__` (`fused_moe/experts/trtllm_mxfp4_moe.py`) ⏎ - `GptOssMxfp4MoEMethod.__init__` and `Mxfp4MoEMethod.__init__` ⏎   (`quantization/mxfp4.py`) ⏎  ⏎ `UnquantizedFusedMoEMethod.process_weights_after_ …[truncated]

### L3-4765f0f189  (L3, 2026-06-05, sha 4765f0f189fd, PR #44130)
TITLE: [Bugfix] Fix `sequence_parallel_chunk_impl` custom op aliasing its input (#44130)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/utils.py (+4/-1)
LABELS: bug, ready
BODY: ## Root cause ⏎  ⏎ `sequence_parallel_chunk_impl()` returns `torch.narrow(y, ...)`, which is a *view* of its input whenever the sequence length is already divisible by `tp_size` (the no-pad branch sets `y = x`). A functional custom op must not return an output that aliases one of its inputs, so under the AOT-autograd `torch.compile` path the runtime aliasing guard rejects it during the memory-profiling dummy run and engine startup crashes. It trigg …[truncated]

### L3-00d1fb7747  (L3, 2026-06-06, sha 00d1fb774755, PR #43684)
TITLE: [Bugfix][ROCm] `ApplyRotaryEmb`: fall back to native when flash_attn rotary grid would exceed the HIP per-dim limit (#43684)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rotary_embedding/common.py (+23/-0)
LABELS: bug, rocm, ready
BODY: ## Summary ⏎  ⏎ On AMD ROCm/HIP the per-dimension launch grid is capped at **65,535**. ⏎ `vllm/model_executor/layers/rotary_embedding/common.py::ApplyRotaryEmb.forward_hip` ⏎ delegates to the `flash_attn` Triton rotary kernel, whose grid is ⏎ `(cdiv(nheads, BLOCK_H), cdiv(seq_len, BLOCK_M), batch)`. With ⏎ `BLOCK_M = 8` (the default when `rotary_dim <= 128`) any `seq_len` ⏎ above `65535 * 8 = 524,280` overflows `gridY` and ⏎ `hipModuleLaunchKernel` abort …[truncated]

### L3-062b05ff3a  (L3, 2026-06-06, sha 062b05ff3af4, PR #44075)
TITLE: [ROCm][Perf] Fused MoE W4A16 HIP kernel for AMD RDNA3 (gfx1100) (#44075)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+2/-1); csrc/rocm/moe_q_gemm_rdna3.cu (+639/-0); csrc/rocm/ops.h (+9/-0); csrc/rocm/torch_bindings.cpp (+9/-0); tests/kernels/quantization/test_rdna3_compile_guards.py (+503/-0); tests/kernels/quantization/test_rdna3_moe_w4a16.py (+367/-0); vllm/_custom_ops.py (+53/-0); vllm/model_executor/layers/fused_moe/layer.py (+1/-0); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py (+14/-4); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_rdna3.py (+261/-0); (+1 more)
LABELS: rocm, ready, ci/build, verified
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ### Summary ⏎  ⏎ Native HIP kernel for W4A16 MoE on RDNA3 (gfx1100), replacing the Triton ⏎ `fused_moe_kernel_gptq_awq` path. It uses the same dequant + dot primitives as ⏎ the dense W4A16 kernel: `v_dot2_f32_f16` / `v_dot2_f32_bf16`, exllama bit-trick ⏎ dequant, and 64-bit CAS atomic output. ⏎  ⏎ ### What it does ⏎  ⏎ - **Fused HIP kernel** (`csrc/rocm/moe_q_gemm_rdna3.cu`): expert routing + W4A16 ⏎   GEMM in a single kernel launch. Templated on `BLOCK_SI …[truncated]

### L3-bc5745a00f  (L3, 2026-06-06, sha bc5745a00f58, PR #42838)
TITLE: [ROCm][MLA] Replace torch.cat in sparse-MLA forward_mqa with fused concat_mla_q (#42838)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+12/-2); csrc/libtorch_stable/concat_mla_q.cuh (+0/-3)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ In the ROCm sparse-MLA backend, `forward_mqa` builds `q = torch.cat([ql_nope, q_pe], dim=-1)` once per layer per forward pass. On DeepSeek-V3.2 (61 sparse-MLA layers on the decode path) this adds up to **61 fresh tensor allocations and 61 `aten::CatArrayBatchedCopy` kernel launches per decode step**, ≈ 360 µs at MC=4 (≈ 4% of a decode step measured at TP=4, ISL=1000, OSL=100 on MI355X). ⏎  ⏎ The fused `ops.concat_mla_q(ql_nope, q_pe, q_ou …[truncated]

### L3-2a983c79ac  (L3, 2026-06-06, sha 2a983c79acdb, PR #44699)
TITLE: [DSV4] Decouple DS V4 Sparse MLA Metadata from DS V3.2 (#44699)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.flashmla_sparse, L3.dispatch.registry
FILES: vllm/models/deepseek_v4/nvidia/flashmla.py (+7/-39); vllm/v1/attention/backends/mla/flashmla_sparse.py (+3/-272); vllm/v1/attention/backends/registry.py (+1/-1); docs/design/attention_backends.md (+1/-1); vllm/models/deepseek_v4/amd/rocm.py (+8/-10); vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py (+13/-10); vllm/models/deepseek_v4/sparse_mla.py (+416/-0)
LABELS: documentation, ready, v1, nvidia
BODY: 

### L3-32f34d3935  (L3, 2026-06-06, sha 32f34d393524, PR #44420)
TITLE: [feature] add index share feature for DSA MTP (#44420)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/deepseek_mtp.py (+24/-2); vllm/model_executor/models/deepseek_v2.py (+16/-15); vllm/transformers_utils/model_arch_config_convertor.py (+33/-1); vllm/v1/spec_decode/llm_base_proposer.py (+32/-3); vllm/v1/worker/gpu/spec_decode/eagle/utils.py (+9/-4)
LABELS: speculative-decoding, ready, v1, deepseek
BODY: ## Summary ⏎  ⏎ This PR implements IndexCache top-k index reuse for DeepSeek Sparse MLA's MTP Layer, inspired by [IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201). ⏎  ⏎ The core idea is to avoid running the sparse attention indexer in every layer or every MTP speculative step when the top-k token selections can be reused. The implementation adds support for carrying `topk_indices` through the De …[truncated]

### L3-fa27d4e9cf  (L3, 2026-06-06, sha fa27d4e9cf3c, PR #44700)
TITLE: [PERF] [Qwen3.5] Split mixed prefill+decode batches: route decodes to the recurrent kernel (#44700)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2-MTP.yaml (+12/-0); tests/evals/gsm8k/configs/models-qwen35-blackwell.txt (+2/-1); tests/kernels/mamba/test_gdn_forward_core_split.py (+296/-0); vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+75/-23); vllm/v1/attention/backends/gdn_attn.py (+41/-7)
LABELS: ready, v1, qwen
DEEP_STUDY: deep-study performance PR ()
BODY: # Description ⏎  ⏎ This PR has 3 changes: ⏎  ⏎ 1. **Main: split mixed prefill+decode batches** (see below). ⏎ 2. **Add MTP eval** config for Qwen3.5-397B NVFP4. ⏎ 3. **Rename** `fast_kernel` → `aiter_kernel` (`fast_kernel` was too general). ⏎  ⏎ ## Problem ⏎  ⏎ GDN attention ran **all** non-spec tokens through `chunk_gated_delta_rule`. Each decode becomes its own chunk padded to `FLA_CHUNK_SIZE=64`, so `D` decodes cost `D` near-empty 64-token chunks of was …[truncated]

### L3-6ac69203e8  (L3, 2026-06-07, sha 6ac69203e8eb, PR #44417)
TITLE: [videoloader] implement glm46v video loader (#44417)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/multimodal/test_video.py (+133/-0); vllm/multimodal/video.py (+126/-0)
LABELS: ready, multi-modality
BODY: echo #44126 implement a video loader for GLM-4.6V VideoProcessor

### L3-228bcc436b  (L3, 2026-06-07, sha 228bcc436b0f, PR #44674)
TITLE: [ROCm][Kernel] Enable permute_cols for ROCm (#44674)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-3); csrc/libtorch_stable/ops.h (+1/-1); csrc/libtorch_stable/torch_bindings.cpp (+1/-4)
LABELS: rocm, ready, ci/build
BODY: `pytest -sv tests/kernels/core/test_permute_cols.py` ⏎  ⏎ ================ 3 passed, 16 warnings in 1.81s ========================

### L3-810966453a  (L3, 2026-06-07, sha 810966453a25, PR #36423)
TITLE: [XPU] Support  cpu kv offloading and tiering offloading on XPU platform (#36423)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/intel_jobs/misc_intel.yaml (+23/-0); tests/v1/kv_connector/unit/offloading_connector/test_worker.py (+5/-2); tests/v1/kv_connector/unit/test_offloading_connector.py (+8/-5); vllm/_custom_ops.py (+9/-4); vllm/v1/kv_offload/cpu/gpu_worker.py (+28/-13); vllm/v1/kv_offload/cpu/shared_offload_region.py (+10/-6); vllm/v1/kv_offload/cpu/spec.py (+3/-2)
LABELS: documentation, intel-gpu, ready, ci/build, v1, kv-connector
BODY: ## Purpose ⏎ Support CPU KV offloading with XPU [swap_blocks](https://github.com/vllm-project/vllm-xpu-kernels/pull/157) kernel  on XPU platform  ⏎  ⏎ Need XPU kernel support cross layer KV layout with layer dimension:  https://github.com/vllm-project/vllm-xpu-kernels/pull/335 ⏎  ⏎ ## Test Plan ⏎ pytest -s -v tests/v1/kv_offload ⏎ pytest -s -v tests/v1/kv_connector/unit/offloading_connector/test_worker.py ⏎ pytest -s -v tests/v1/kv_connector/unit/offload …[truncated]

### L3-3bb46975bd  (L3, 2026-06-07, sha 3bb46975bd31, PR #37149)
TITLE: [XPU][Feature] transparent sleep mode support for XPU platform (#37149)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/intel_jobs/basic_correctness.yaml (+22/-0); .buildkite/test-amd.yaml (+2/-2); .buildkite/test_areas/basic_correctness.yaml (+2/-2); tests/basic_correctness/test_mem.py (+22/-21); vllm/config/model.py (+1/-1); vllm/device_allocator/__init__.py (+48/-0); vllm/device_allocator/cumem.py (+1/-11); vllm/device_allocator/xpumem.py (+295/-0); vllm/platforms/interface.py (+1/-1); vllm/v1/worker/gpu_worker.py (+16/-9)
LABELS: documentation, intel-gpu, ready, ci/build, v1
BODY: ## Purpose ⏎ **Note that this depends on `XPUPluggableAllocator` and `use_mem_pool` API in torch-xpu 2.11.** ⏎ This PR is to enable transparent sleep mode on XPU platform. It introduces changes: ⏎  ⏎ - Add XpuMemoryAllocator and provide sleep/wake up interface for cuda parity ⏎ - `get_mem_allocator` to support both cuda_alike path and xpu path ⏎ - Add memory allocator UT coverage on XPU ⏎  ⏎  ⏎ ## Test Plan ⏎ ```./tests/basic_correctness/test_mem.py```

### L3-4dcd10eb0d  (L3, 2026-06-07, sha 4dcd10eb0d22, PR #44454)
TITLE: [1/N][KV-Cache Layout Refactor] Refactor DSV4 KV cache config construction (#44454)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/core/kv_cache_utils.py (+38/-49)
LABELS: ready, v1
BODY: PR #42374 (first part of RFC #42082) has been split into 4 PRs: ⏎  ⏎ -> #44454 [1/N][KV-Cache Layout Refactor] Refactor DSV4 KV cache config ⏎ #44455 [2/N][KV-Cache Layout Refactor] Pack K/V into the content dim across attention backends ⏎ #44456 [3/N][KV-Cache Layout Refactor] Standardize Mamba cache; drop `get_transfer_cache_regions` ⏎ #44458 [4/N][KV-Cache Layout Refactor] Standardize KV cache layout ⏎  ⏎  ⏎ ## Summary ⏎  ⏎ Extracts the per-page-size la …[truncated]

### L3-66ecfd0568  (L3, 2026-06-07, sha 66ecfd05681b, PR #42599)
TITLE: [Dependency] Remove stale cuDNN frontend upper bound (#42599)
SOURCES: dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: requirements/cuda.txt (+1/-3)
LABELS: ready, ci/build, nvidia
BODY: ## Summary ⏎  ⏎ Remove the stale upper bound on `nvidia-cudnn-frontend` since the breaking changes in `1.19.0` have been fixed. The constraint is now `>=1.19.1`, which excludes `1.19.0` while letting vLLM pick up newer cuDNN frontend fixes and improvements. This is compatible with FlashInfer's `>=1.13.0` requirement.

### L3-6124a98a9b  (L3, 2026-06-07, sha 6124a98a9bb9, PR #44215)
TITLE: [Bugfix] Fix FunASR-Nano crash during initialization (#44215)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/funasr.py (+4/-0)
LABELS: bug, ready
BODY: ## Purpose ⏎ Fix an engine-initialization crash when serving `FunASRForConditionalGeneration` ⏎ (e.g. `allendou/Fun-ASR-Nano-2512-vllm`) on vLLM >= 0.22 / current main: ⏎ ``` ⏎ NotImplementedError: No language model found in FunASRForConditionalGeneration! ⏎ ``` ⏎  ⏎ Root cause: #39805 added a `get_language_model()` call in `gpu_model_runner.load_model()` to unwrap VLM wrappers into their underlying MoE language model for EPLB. It runs for every non-MoE …[truncated]

### L3-2ed0a9627b  (L3, 2026-06-08, sha 2ed0a9627b50, PR #42736)
TITLE: [Kernel][Test] Make kernel tests for mamba dual-HW (CUDA + XPU) (#42736)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/mamba/test_causal_conv1d.py (+12/-4); tests/kernels/mamba/test_mamba_ssm.py (+24/-7); tests/kernels/mamba/test_mamba_ssm_ssd.py (+13/-3)
LABELS: intel-gpu, ready, nvidia, verified
BODY: ## Purpose ⏎  ⏎ Make the MambaMixer Triton kernel tests under `tests/kernels/mamba/` runnable on Intel XPU in addition to CUDA/ROCm. The underlying kernels (causal_conv1d, selective_state_update, mamba_chunk_scan_combined_varlen and its sub-kernels) are pure Triton and already work on XPU - only the test harness was pinning `device="cuda"`. ⏎  ⏎ Changes in three files: ⏎  ⏎ - `tests/kernels/mamba/test_mamba_ssm_ssd.py` ⏎ - `tests/kernels/mamba/test_mamb …[truncated]

### L3-eebce65756  (L3, 2026-06-08, sha eebce65756f0, PR #42953)
TITLE: [XPU]feat: add DeepSeek-V4 XPU attention decode path (#42953)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/xpu_mla_sparse.py (+8/-2); vllm/model_executor/kernels/linear/scaled_mm/xpu.py (+15/-0); vllm/model_executor/layers/mhc.py (+74/-0); vllm/models/deepseek_v4/__init__.py (+9/-8); vllm/models/deepseek_v4/compressor.py (+1/-1); vllm/models/deepseek_v4/xpu/__init__.py (+2/-0); vllm/models/deepseek_v4/xpu/model.py (+1340/-0); vllm/models/deepseek_v4/xpu/mtp.py (+511/-0); vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py (+159/-0); vllm/models/deepseek_v4/xpu/xpu_sparse.py (+350/-0); (+1 more)
LABELS: intel-gpu, ready, v1, deepseek, verified
BODY: ## Summary ⏎ Add XPU-specific decode implementation for DeepSeek-V4 MLA sparse attention, including Triton kernels for FP8 KV cache operations. ⏎  ⏎ ## Changes ⏎ - **mhc.py**: add `forward_xpu` for MHC Pre/Post/Fuse processors ⏎ - **xpu_qnorm_rope_kv_fp8_insert.py**: Triton kernel for fused QK norm + RoPE + FP8 KV insert ⏎ - **xpu_sparse_decode_fp8.py**: Triton kernel for FP8 sparse MLA decode ⏎  ⏎ ## Testing ⏎ Tested with DeepSeek-V4 inference (prefill + …[truncated]

### L3-ba94a3b998  (L3, 2026-06-08, sha ba94a3b99896, PR #40470)
TITLE: [Attention] Extract KV-cache update from CPU attention backend (#40470)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+31/-22)
LABELS: v1, cpu
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Part of #32335 ⏎  ⏎ ## Summary ⏎  ⏎ - Extract the KV-cache update from `CPUAttentionBackendImpl.forward()` into a separate `do_kv_cache_update()` method, as part of issue #32335. ⏎ - Set `forward_includes_kv_cache_update = False` on `CPUAttentionBackend` so the attention layer calls `do_kv_cache_update()` before `forward()`. ⏎ - The CPU backend uses `ops.cpu_attn_reshape_and_cache` (not `reshape_and_cache_flash`), which requires an ISA parameter — this is de …[truncated]

### L3-05cb606cad  (L3, 2026-06-08, sha 05cb606cad30, PR #44809)
TITLE: [ROCm][CI] Re-route NixlConnector jobs (#44809)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+10/-10); tests/v1/kv_connector/nixl_integration/config_sweep_accuracy_test.sh (+5/-1)
LABELS: rocm, ready, ci/build, v1, kv-connector
BODY: Re-route ROCm NixlConnector CI jobs that use KV connectors through `TRITON_ATTN` instead of forcing `ROCM_ATTN`. This updates the AMD Buildkite pipeline entries for the NixlConnector accuracy and spec decode acceptance jobs on MI250, MI300, and MI355, and teaches the config sweep helper to accept a generic `ATTENTION_BACKEND=...` override. ⏎  ⏎ ## Changed Test Groups ⏎  ⏎ - MI250: `NixlConnector PD + Spec Decode acceptance (2 GPUs)` ⏎ - MI250: `Distri …[truncated]

### L3-5add018beb  (L3, 2026-06-08, sha 5add018bebbd, PR #44854)
TITLE: [Connector] Remove `P2pNcclConnector` (#44854)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/disagg_benchmarks/disagg_overhead_benchmark.sh (+0/-143); benchmarks/disagg_benchmarks/disagg_performance_benchmark.sh (+0/-157); benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py (+0/-260); benchmarks/disagg_benchmarks/round_robin_proxy.py (+0/-63); benchmarks/disagg_benchmarks/visualize_benchmark_results.py (+0/-47); docs/design/p2p_nccl_connector.md (+0/-319); docs/features/disagg_prefill.md (+0/-1); examples/disaggregated/disaggregated_prefill.py (+0/-127); examples/disaggregated/disaggregated_prefill.sh (+0/-125); examples/disaggregated/p2p_nccl_xpyd/disagg_example_p2p_nccl_xpyd.sh (+0/-245); (+8 more)
LABELS: documentation, performance, ready, kv-connector
BODY: After some more recent dicussions, for the same reasons listed here https://github.com/vllm-project/vllm/issues/33115, we've decided to remove the `P2pNcclConnector`. ⏎  ⏎ I would still like to thank @Abatom and everyone who was contributed to the connector as this helped PD take off the ground in its early stages. ⏎  ⏎ Mind that similar connectors can still be implemented OOT leveraging the current ConnectorAPI.

### L3-540aaf2140  (L3, 2026-06-08, sha 540aaf21406b, PR #44264)
TITLE: [Bugfix][Model] Qwen3-Omni: move cu_seqlens to GPU before VIT attention (#44264)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_omni_moe_thinker.py (+3/-0)
LABELS: bug, documentation, ready, qwen
ISSUES: #44180 [Bug]: v0.22.0 fails to load Qwen/Qwen3-Omni-30B-A3B-Thinking on H20: RuntimeError: cu_seqlens_q must be on CUDA
BODY: ## Purpose ⏎  ⏎ Fixes #44180. ⏎  ⏎ During `profile_run` the multimodal `grid_thw` arrives on CPU, so the `cu_seqlens` tensor built from it inherits the CPU device. Passing this CPU tensor into the FA3 vit attention path raises `RuntimeError: cu_seqlens_q must be on CUDA` and crashes engine init when loading `Qwen/Qwen3-Omni-30B-A3B-Thinking` (and the Instruct variant). ⏎  ⏎ Move `cu_seqlens` to `self.device` after construction so the FA3 wrapper receiv …[truncated]

### L3-baacbfcebf  (L3, 2026-06-08, sha baacbfcebf5e, PR #42978)
TITLE: [ROCm][MLA][Bugfix] Reserve FP8 prefill workspace before lock for Kimi-K2.5 (#42978)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+34/-5)
LABELS: bug, rocm, ready, v1
BODY: ## Summary ⏎  ⏎ Fixes a ROCm AITER dense MLA FP8 prefill crash after #42509 where the per-call scratch workspace introduced for `mla_prefill_ps_asm_fwd` could be first requested after the global workspace manager had already been locked. ⏎  ⏎ The fix reserves the maximum FP8 prefill scratch shapes while initializing the FP8 persistent-scheduling metadata buffers, before `lock_workspace()` runs. The reservation is sized from the AITER metadata capacit …[truncated]

### L3-dc10e467a9  (L3, 2026-06-09, sha dc10e467a985, PR #44983)
TITLE: [Bugfix] Fix minimax_qk_norm_fusion (#44983)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py (+41/-9)
LABELS: bug, ready
ISSUES: #44003 [Bug]: [Regression] MiniMax-M2.5-int4 fails with cudaErrorPeerAccessUnsupported since PR #43410
BODY: ## Purpose ⏎  ⏎ -   Fixes https://github.com/vllm-project/vllm/issues/44003 ⏎ -   Fix the fallback of minimax_qk_norm_fusion when VLLM_ALLREDUCE_USE_FLASHINFER=1 is set, which raises the error below. This is caused by the variance in minimax_qk_norm being fp32, which isn't compatible with FlashInfer's AllReduce. ⏎ ```bash ⏎ orker_TP3 pid=2256615) ERROR 06-09 08:04:17 [multiproc_executor.py:987]     out = fi_ar_comm.all_reduce(input_) ⏎ (Worker_TP3 pid= …[truncated]

### L3-9f153aa781  (L3, 2026-06-09, sha 9f153aa781f0, PR #40576)
TITLE: [MM][Perf][CG] Support ViT full CUDA graph for glm4_1v image and video inference  (#40576)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/design/cuda_graphs_multimodal.md (+1/-0); examples/generate/multimodal/vision_language_offline.py (+1/-0); tests/models/multimodal/generation/test_vit_cudagraph.py (+22/-0); vllm/model_executor/models/glm4_1v.py (+456/-25)
LABELS: documentation, ready, multi-modality, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Following https://github.com/vllm-project/vllm/issues/38175, this PR implements ViT CUDA graph support for glm4_1v models image and video inference . The implementation draws references from https://github.com/vllm-project/vllm/pull/35963 (image) and https://github.com/vllm-project/vllm/pull/38061 (video) ⏎  ⏎ 1. Functional Test ⏎ 2. Benchmark in some scenarios: ⏎ - no DP VIT + eager vs no DP VIT + cuda graph. ⏎ - DP VIT + eager vs DP VIT  …[truncated]

### L3-80e2c4462d  (L3, 2026-06-09, sha 80e2c4462dd9, PR #42864)
TITLE: [ROCm][Compile] Fuse AR + RMSNorm + per-group FP8 quant (+ DSv3.2 indexer fan-out) (#42864)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/passes/distributed/test_fusion_all_reduce.py (+276/-0); vllm/_aiter_ops.py (+196/-0); vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+323/-2)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ Two new patterns in `RocmAiterAllReduceFusionPass` that fuse the `all_reduce -> RMSNorm[+add] -> per-group FP8 quant [-> bf16 indexer GEMM]` chain into one AITER call, eliminating ~535 us / decode step of standalone `triton_per_token_group_quant_fp8` launches on DeepSeek V3.2 MI355X TP4. ⏎  ⏎ This is the AR-side analogue of #41825 (which fixed the quant-only half) and the ROCm port of the flashinfer `AllReduceFusedRMSNormStaticQuantFP8Pat …[truncated]

### L3-c9c1540e61  (L3, 2026-06-09, sha c9c1540e61b2, PR #44936)
TITLE: [ROCm][V2] Fix failed assertion in Llama models when using EAGLE with `ROCM_AITER_FA` (#44936)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+4/-3)
LABELS: rocm, ready, v1, llama
BODY: This test fails on ROCm ever since https://github.com/vllm-project/vllm/pull/43458 enabled MRV2 for Llama: ⏎ `pytest -v -s tests/v1/e2e/spec_decode/test_spec_decode.py::test_eagle_correctness_heavy[ROCM_AITER_FA-llama3_eagle]` ⏎  ⏎ The test is failing the assertion `assert common_attn_metadata.seq_lens_cpu_upper_bound is not None` in `split_decodes_prefills_and_extends`, which is currently only being used in the `ROCM_AITER_FA` backend. The early re …[truncated]

### L3-2ee5106372  (L3, 2026-06-09, sha 2ee51063722c, PR #39425)
TITLE: Remove `raw_inputs` from transformers backend (#39425)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/transformers/multimodal.py (+3/-12)
LABELS: new-model, ready, v1, verified
BODY: Transformers backend models doesn't need anymore to pass arbitrary kwargs to model's `forward`, specifically token-types. vLLM's attention backend infers multimodal toke types at runtime and computes correct mask based on `config.model_type` ⏎  ⏎ Verified that the backend also goes through the "bidirectional image mask" path with correct `mm_range` values

### L3-d7607ad273  (L3, 2026-06-09, sha d7607ad2730f, PR #44914)
TITLE: [Bug] Fix deepseek v4 OOM issue (#44914)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/quant_config.py (+10/-3)
LABELS: bug, ready, deepseek
BODY: ## Purpose ⏎  ⏎ On H200 ⏎  ⏎ ```bash ⏎ vllm serve deepseek-ai/DeepSeek-V4-Pro   --trust-remote-code   --kv-cache-dtype fp8   --block-size 256   --enable-expert-parallel   --tensor-parallel-size 8   --max-model-len 800000   --gpu-memory-utilization 0.95   --max-num-seqs 512   --max-num-batched-tokens 512   --no-enable-flashinfer-autotune   --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}' ⏎ ``` ⏎  ⏎ Will raise ⏎  ⏎ ```bash ⏎ (Worker_TP …[truncated]

### L3-70db1488c5  (L3, 2026-06-09, sha 70db1488c5d5, PR #44144)
TITLE: [DSV4][XPU] Add MHC fused_post_pre support (#44144)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mhc.py (+72/-3); vllm/models/deepseek_v4/xpu/model.py (+40/-14)
LABELS: intel-gpu, ready, v1, verified
BODY: ## Summary ⏎  ⏎ Add `MHCFusedPostPreOp` XPU support for DeepSeek-V4 on Intel XPU, enabling the fused MHC post+pre path in the decoder loop (matching the AMD/CUDA pattern). ⏎  ⏎ ## Changes ⏎  ⏎ - **`vllm/model_executor/layers/mhc.py`**: Implement `forward_native` for `MHCFusedPostPreOp` (decomposes into `mhc_post_torch` + `mhc_pre_torch`); add `forward_xpu` delegating to `forward_native`. ⏎ - **`vllm/models/deepseek_v4/xpu/model.py`**: Update decoder loop to us …[truncated]

### L3-6deb05e0e4  (L3, 2026-06-09, sha 6deb05e0e4f6, PR #42175)
TITLE: [Core][Model] Gemma4: Unified FA4 for all layers + FlashAttention mm_prefix support (#42175)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fa4_cutedsl, L3.triton.v1_backend, L3.mla.flashmla_v1_adapter, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashinfer_sparse, L3.dispatch.abstract_interface, L3.flex_attention
FILES: vllm/v1/attention/backend.py (+8/-0); vllm/v1/attention/backends/flash_attn.py (+89/-3); vllm/v1/attention/backends/flex_attention.py (+1/-0); vllm/v1/attention/backends/mla/flashattn_mla.py (+1/-0); vllm/v1/attention/backends/mla/flashinfer_mla.py (+1/-0); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+1/-0); vllm/v1/attention/backends/mla/flashmla.py (+1/-0); vllm/v1/attention/backends/mla/tokenspeed_mla.py (+1/-0); vllm/v1/attention/backends/triton_attn.py (+14/-36); vllm/v1/attention/backends/utils.py (+30/-0); (+4 more)
LABELS: documentation, speculative-decoding, ready, ci/build, v1, kv-connector, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Enable Flash Attention 4 (FA4) as the default attention backend for Gemma4 models on Hopper (SM90) GPUs, and add multimodal bidirectional attention (mm_prefix / PrefixLM) support to the FlashAttention backend. ⏎  ⏎ Gemma4 uses heterogeneous head dimensions across layers: `head_dim=256` for sliding-window attention and `global_head_dim=512` for full attention. Previously, the `Gemma4Config` gate detected this mismatch and forced `TRITO …[truncated]

### L3-a1ec011a83  (L3, 2026-06-10, sha a1ec011a833e, PR #39498)
TITLE: [Bugfix] Add deepseek_v32 to Quark dynamic MXFP4 model type check (#39498)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/quark/quark.py (+32/-0)
LABELS: bug, rocm, ready, deepseek
BODY: ## Summary ⏎  ⏎ Add `deepseek_v32` to `_DEEPSEEK_V3_FAMILY_MODEL_TYPES` so that Quark correctly enables `dynamic_mxfp4_quant` for [amd/DeepSeek-V3.2-mxfp4](https://huggingface.co/amd/DeepSeek-V3.2-mxfp4). ⏎  ⏎ Without this fix, excluded layers (e.g. `self_attn`) are not dynamically re-quantized when serving MXFP4-quantized DeepSeek-V3.2 checkpoints, causing silent correctness degradation (all-zero or garbage output tokens). ⏎  ⏎ ## Root cause ⏎  ⏎ `_DEEPSEEK_V3_ …[truncated]

### L3-9ad08c4d15  (L3, 2026-06-10, sha 9ad08c4d1513, PR #44683)
TITLE: [Bugfix][Rust Frontend] Fix missing added tokens in hf/fastokens tokenizer (#44683)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: rust/src/chat/src/multimodal.rs (+1/-1); rust/src/server/src/routes/tests.rs (+18/-5); rust/src/tokenizer/src/hf.rs (+38/-2); rust/src/tokenizer/src/hf/added_tokens.rs (+158/-0)
LABELS: bug, ready, rust
BODY: ## Purpose ⏎ - For [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct), its image_token_id `"<|image_pad|>"` located at [tokenizer_config.json](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/main/tokenizer_config.json#L101) but not `tokenizer.json` ⏎ - So it will cause an error when create chat/text backends: ⏎ ``` ⏎ $ vllm-rs serve /mnt/data0/LLM/Qwen2-VL-2B-Instruct/ ⏎ (RustFrontend pid=1068478) INFO 06-06 02:52:24 [ …[truncated]

### L3-6ec7dcd641  (L3, 2026-06-10, sha 6ec7dcd64125, PR #44448)
TITLE: [Frontend][Metrics] Add `vllm:tool_call_parser_invocations_total` Prometheus metric (#44448)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/entrypoints/openai/api_server.py (+8/-1); vllm/parser/abstract_parser.py (+51/-19); vllm/parser/metrics.py (+108/-0)
LABELS: frontend, ready
BODY: ## Purpose ⏎  ⏎ Add a metric for tool parser activity so operators can see how often the parser runs and whether an invocation produced a tool call. This makes it easier to spot tool-calling regressions during model rollouts or runtime changes. ⏎  ⏎ This PR adds the `vllm:tool_call_parser_invocations_total` counter and records it in `DelegatingParser` for both non-streaming and streaming tool parser calls, with labels for mode (streaming vs non-strea …[truncated]

### L3-5b6b536fdc  (L3, 2026-06-10, sha 5b6b536fdc41, PR #44679)
TITLE: [ROCm][Bugfix] Make intermediate_pad TP-aware in rocm_aiter_fused_experts (#44679)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py (+12/-2)
LABELS: bug, rocm, ready
BODY: ## Purpose ⏎  ⏎ Fix the GSM8K accuracy regression on the AITER MoE path at TP>1 introduced by #42098. This is because FlyDSL and CKTile MoE follow different conventions for hidden_pad/intermediate_pad in AITER, and the `aiter.fused_moe` dispatcher uses FlyDSL on TP1 but CKTile on TP8. This will partially be fixed on the AITER side in 0.1.15 by https://github.com/ROCm/aiter/pull/3401. ⏎  ⏎ On openai/gpt-oss-120b BF16 at TP=8 on MI355X, lm_eval GSM8K ⏎  …[truncated]

### L3-ccc05de038  (L3, 2026-06-10, sha ccc05de03888, PR #45073)
TITLE: [Bugfix] Fix missing sequence_lengths in EXAONE-4.5 vision encoder (#45073)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/exaone4_5.py (+7/-0)
LABELS: bug, ready
ISSUES: #45071 [Bug]: EXAONE-4.5 Vision — unexpected keyword argument 'sequence_lengths' in Exaone4_5_VisionBlock.forward()
BODY: ## Purpose ⏎  ⏎ PR #42787 made the Qwen2.5-VL vision backbone pass `sequence_lengths` (FlashInfer CuDNN metadata) to every vision block, but the EXAONE-4.5 overrides of the vision block and attention kept the pre-#42787 signature. Since EXAONE-4.5 inherits `Qwen2_5_VisionTransformer.forward`, any multimodal request now fails with: ⏎  ⏎     TypeError: Exaone4_5_VisionBlock.forward() got an unexpected ⏎     keyword argument 'sequence_lengths' ⏎  ⏎ Thread  …[truncated]

### L3-e2db0222e9  (L3, 2026-06-10, sha e2db0222e9b2, PR #45074)
TITLE: [Perf][Attention] Pin MLA chunked-context metadata tensors so H2D copies are truly non-blocking (#45074)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+5/-3)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Fix a hidden host-side stall in the MLA chunked-prefill metadata build that serializes CPU metadata prep with GPU execution on every scheduler step. ⏎  ⏎ In `MLACommonMetadataBuilder.build()`, the chunked-context section creates three small CPU tensors and ships them to the GPU with `.to(device, non_blocking=True)` when assembling `ChunkedContextMetadata`: ⏎  ⏎ - `cu_seq_lens_cpu` — allocated with `pin_memory=True` ✅ ⏎ - `chunk_starts` — pageab …[truncated]

### L3-fe1d923afc  (L3, 2026-06-10, sha fe1d923afccb, PR #45110)
TITLE: [BUGFIX][XPU] fix xpu `flash_attn_varlen_func` interface (#45110)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/_xpu_ops.py (+3/-0)
LABELS: bug, intel-gpu, ready
BODY: ## Purpose ⏎ xpu path broke after https://github.com/vllm-project/vllm/pull/42175, which introduce a new paramter in `flash_attn_varlen_func`  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-6e64c1bab1  (L3, 2026-06-10, sha 6e64c1bab187, PR #44565)
TITLE: [10c/n] Migrate MoE kernels to torch stable ABI  (#44565)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: .gitignore (+1/-1); .pre-commit-config.yaml (+1/-1); CMakeLists.txt (+61/-20); csrc/libtorch_stable/dispatch_utils.h (+22/-0); csrc/libtorch_stable/moe/dsv3_router_gemm_bf16_out.cu (+1/-4); csrc/libtorch_stable/moe/dsv3_router_gemm_entry.cu (+50/-32); csrc/libtorch_stable/moe/dsv3_router_gemm_float_out.cu (+1/-4); csrc/libtorch_stable/moe/grouped_topk_kernels.cu (+48/-40); csrc/libtorch_stable/moe/marlin_moe_wna16/.gitignore (+0/-0); csrc/libtorch_stable/moe/marlin_moe_wna16/generate_kernels.py (+1/-1); (+24 more)
LABELS: rocm, ready, ci/build, nvidia
BODY: ## Purpose ⏎ This PR continues the libtorch stable ABI migration (see #26946) for vLLM MoE CUDA kernels by introducing _moe_C_stable_libtorch and moving all of the MoE ops (topk, align, permute/unpermute, grouped topk, and related headers) into csrc/libtorch_stable/moe/. ⏎  ⏎ Note: started using the [10x/n] label to indicate that they could be merged in any order (theoretically, there could still be merge conflicts because of CMakeLists.txt, ops.h,  …[truncated]

### L3-2ba68d9bf7  (L3, 2026-06-11, sha 2ba68d9bf704, PR #44946)
TITLE: [Test] Fix one-sided MNNVL alltoall test workspace under-reservation (#44946)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/distributed/test_mnnvl_alltoall.py (+5/-0)
LABELS: bug, ready
BODY: ## Summary ⏎  ⏎ `tests/distributed/test_mnnvl_alltoall.py::test_one_sided_dispatch_combine` ⏎ initializes the FlashInfer one-sided `MoeAlltoAll` workspace **without ⏎ declaring the fp8 block-scale payload it later dispatches**. The worker ⏎ dispatches four payloads: ⏎  ⏎ - `a1q` — nvfp4 hidden states, `(tokens, hidden // 2)` → `hidden // 2` B/token ⏎ - `a1q_scale` — fp8 block scales, `(tokens, hidden // 16)` → `hidden // 16` B/token ⏎ - `topk_ids`, `topk_weights`  …[truncated]

### L3-40e065e86a  (L3, 2026-06-11, sha 40e065e86a91, PR #45204)
TITLE: [Docker] Fix CUTLASS DSL cu13 install order in Dockerfile (#45204)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+35/-0); docs/assets/contributing/dockerfile-stages-dependency.png (+0/-0)
LABELS: documentation, ready, ci/build, nvidia
BODY: ## Summary ⏎  ⏎ Fix CUDA 13 Docker builds by force-reinstalling `nvidia-cutlass-dsl-libs-cu13` last, avoiding uv install-order races with overlapping CUTLASS DSL sibling wheels

### L3-750aab5b8e  (L3, 2026-06-11, sha 750aab5b8e7f, PR #44424)
TITLE: [Bugfix] Fix CPU memory leak related to not cleaning up old remotes data (#44424)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector.py (+118/-0); tests/v1/kv_connector/unit/test_nixl_connector_hma.py (+1/-0); vllm/distributed/kv_transfer/kv_connector/utils.py (+5/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py (+68/-8)
LABELS: bug, ready, v1, kv-connector
BODY: ## Problem ⏎  ⏎ When a Prefill pod dies, the Decode pod never cleans up per-engine state accumulated during handshake (D has no way to know when a P dies/is not used anymore right now). ⏎ These grow monotonically until D shuts down, leaking both NIXL resources and memory. ⏎  ⏎ ## Approach ⏎  ⏎ This PR adds a **TTL-based eviction** policy: if D hasn't read from a remote engine for `engine_ttl` seconds (default 600, configurable via kv_connector_extra_con …[truncated]

### L3-f1d8d99717  (L3, 2026-06-11, sha f1d8d99717b6, PR #43495)
TITLE: [Bugfix] CohereModel.load_weights: skip modelopt _quantizer.* keys (#43495)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/commandr.py (+5/-1)
LABELS: bug, ready
ISSUES: #41925 [Bug] Cohere2ForCausalLM fails to load ModelOpt NVFP4 quantized models
BODY: ## Description ⏎  ⏎ ModelOpt NVFP4 exports of Cohere / Command-A models include raw quantizer-module state keys named `*.weight_quantizer._double_scale` (and similar). These are modelopt calibration-time artifacts — superseded at runtime by the exported `weight_scale` / `weight_scale_2` parameters — but they are not registered parameters on the vLLM-side `CohereModel`. `CohereModel.load_weights` then raises a `KeyError` when it tries to dispatch them …[truncated]

### L3-c2b4cd39ac  (L3, 2026-06-11, sha c2b4cd39acca, PR #37047)
TITLE: [Doc][Attention] Fix MLA top-of-file comments (#37047)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+10/-10)
LABELS: ready
BODY: ## Summary ⏎ Fix several obvious issues in the MLA top-of-file explanatory comment and pseudocode. ⏎  ⏎ ## Details ⏎ - describe prefill as having a larger `Sq / Skv` ratio, close to `1` ⏎ - describe decode as having a smaller `Sq / Skv` ratio, close to `0` ⏎ - rename `spda_o` to `sdpa_o` in the pseudocode examples ⏎ - fix the `ql_nope` pseudocode line to use `q_nope` ⏎ - fix `casual` to `causal` in the chunked prefill pseudocode ⏎ - remove the stray `@ self.num_he …[truncated]

### L3-1f60771c74  (L3, 2026-06-11, sha 1f60771c7448, PR #42679)
TITLE: fix: guard flash-attn rotary import (#42679)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rotary_embedding/common.py (+7/-5)
LABELS: ready
ISSUES: #42675 [Bug]: FA4 causes `no module named 'flash_attn.ops'`
BODY: Fixes #42675. ⏎  ⏎ ## Summary ⏎ - import `flash_attn.ops.triton.rotary` directly when vLLM is not on CPU ⏎ - treat a missing rotary module as unavailable instead of crashing during `ApplyRotaryEmb` construction ⏎ - keep the existing fallback path when FA4 no longer provides the old rotary module ⏎  ⏎ FA4 can leave the `flash_attn` root package importable while moving or removing `flash_attn.ops.triton.rotary`. Checking only the root package therefore selects a …[truncated]

### L3-f81daf8880  (L3, 2026-06-11, sha f81daf888063, PR #41797)
TITLE: [Attention] add triton diff-kv backend for mimo (#41797)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.dispatch.registry
FILES: vllm/v1/attention/backends/flash_attn_diffkv.py (+24/-2); vllm/v1/attention/backends/registry.py (+3/-0); vllm/v1/attention/backends/triton_attn_diffkv.py (+261/-0); vllm/v1/attention/ops/triton_unified_attention_diffkv.py (+529/-0); .buildkite/test_areas/kernels.yaml (+13/-0); docs/design/attention_backends.md (+1/-0); tests/kernels/attention/test_triton_unified_attention_diffkv.py (+189/-0); vllm/model_executor/models/mimo_v2.py (+21/-7)
LABELS: documentation, ready, ci/build, v1
ISSUES: #41519 [Bug]: Xiaomi MiMo v2.5 broken on SM12x
BODY: ## Purpose ⏎  ⏎ Fix https://github.com/vllm-project/vllm/issues/41519 ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ vllm serve XiaomiMiMo/MiMo-V2.5 -tp 4 --trust-remote-code ⏎ vllm serve XiaomiMiMo/MiMo-V2.5 -tp 4 --trust-remote-code --attention-backend TRITON_ATTN_DIFFKV ⏎ ``` ⏎ ``` ⏎ lm_eval --model local-completions --model_args "model=XiaomiMiMo/MiMo-V2.5,base_url=http://0.0.0.0:8000/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=256,timeo …[truncated]

### L3-5a6c7b7ab5  (L3, 2026-06-11, sha 5a6c7b7ab569, PR #45052)
TITLE: [Bug] Fix test flashmla for DSv4 (#45052)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_flashmla_sparse.py (+5/-3)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ `pytest tests/kernels/attention/test_flashmla_sparse.py` ⏎  ⏎ Will raise error ⏎  ⏎ ```bash ⏎ ============================================================= short test summary info ============================================================= ⏎ FAILED tests/kernels/attention/test_flashmla_sparse.py::test_sparse_flashmla_metadata_smoke - AttributeError: 'FlashMLASchedMeta' object has no attribute 'dtype' ⏎ FAILED tests/kernels/attention/test …[truncated]

### L3-b8142294b7  (L3, 2026-06-11, sha b8142294b7e7, PR #45251)
TITLE: [Bugfix] Restrict FlashInfer cuDNN FP8 ViT attention gate to Blackwell (SM 100) (#45251)
SOURCES: path_core, path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/model_executor/layers/attention/mm_encoder_attention.py (+3/-2); vllm/utils/flashinfer.py (+11/-4)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎  ⏎ `is_flashinfer_cudnn_fp8_prefill_attn_supported()` currently returns `True` on SM 90 GPUs (Hopper, e.g. H100 / H200 / H20), but the underlying cuDNN FP8 SDPA forward path requires Blackwell (SM 100) when the output dtype is `bf16`/`fp16` — which is exactly the configuration `MMEncoderAttention._forward_flashinfer` uses (`o_data_type=self.dtype`, where `self.dtype` is the model dtype, typically `bf16`). ⏎  ⏎ As a result, on Hopper the gate …[truncated]

### L3-7021be66e8  (L3, 2026-06-11, sha 7021be66e8c3, PR #45176)
TITLE: [11a/n]  Migrate Marlin kernels to torch stable ABI (#45176)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+139/-139); csrc/libtorch_stable/moe/marlin_moe_wna16/kernel.h (+2/-2); csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h (+4/-4); csrc/libtorch_stable/quantization/gptq_allspark/allspark_utils.cuh (+1/-1); csrc/libtorch_stable/quantization/marlin/.gitignore (+0/-0); csrc/libtorch_stable/quantization/marlin/awq_marlin_repack.cu (+43/-37); csrc/libtorch_stable/quantization/marlin/dequant.h (+0/-0); csrc/libtorch_stable/quantization/marlin/generate_kernels.py (+1/-1); csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu (+50/-41); csrc/libtorch_stable/quantization/marlin/kernel.h (+0/-0); (+9 more)
LABELS: rocm, ready, ci/build
BODY: ## Purpose ⏎ his PR continues the libtorch stable ABI migration (see #26946) for vLLM Marlin CUDA kernels by moving Marlin host ops (marlin_gemm, gptq_marlin_repack, awq_marlin_repack, and marlin_int4_fp8_preprocess), generated template kernels (sm75/sm80/sm89), and shared device headers into csrc/libtorch_stable/quantization/marlin/, registering them on _C_stable_libtorch and removing them from legacy _C. ⏎  ⏎ This PR is stacked on top of #44565 ⏎  …[truncated]

### L3-6fbfdd1831  (L3, 2026-06-11, sha 6fbfdd183145, PR #44583)
TITLE: [NIXL] Per-region KV transfer classification for mixed full-attn + MLA groups (#44583)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector.py (+81/-0); tests/v1/kv_connector/unit/test_tp_mapping.py (+11/-1); vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py (+142/-67)
LABELS: ready, v1, kv-connector
BODY: ## Purpose ⏎  ⏎ Within a single KV-cache group the NIXL connector may need to transfer regions of two kinds: ⏎  ⏎ - **Full-attention (GQA)** layers, whose KV is head-sharded across TP → **SPLIT** (each rank reads its head slice from the remote at a per-rank offset). ⏎ - **MLA** layers, whose latent KV is replicated on every rank and is key-only → **REPLICATE** (whole block read from one rank at offset 0, no V stream). ⏎  ⏎ Previously this was handled by scatte …[truncated]

### L3-eb28452b10  (L3, 2026-06-11, sha eb28452b10a1, PR #45163)
TITLE: [Model] Add DiffusionGemma Support (#45163)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_build, L3.flash_attn.fa4_cutedsl, L3.flash_attn.fa_utils, L3.triton.unified_attention, L3.triton.v1_backend, L3.dispatch.abstract_interface
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/v1/attention/backend.py (+4/-2); vllm/v1/attention/backends/fa_utils.py (+6/-0); vllm/v1/attention/backends/flash_attn.py (+35/-3); vllm/v1/attention/backends/triton_attn.py (+8/-1); vllm/v1/attention/ops/triton_attention_helpers.py (+46/-11); vllm/v1/attention/ops/triton_unified_attention.py (+63/-7); vllm/v1/attention/ops/triton_unified_attention_diffkv.py (+1/-0); benchmarks/kernels/benchmark_moe.py (+6/-0); docs/design/attention_backends.md (+1/-1); (+42 more)
LABELS: documentation, performance, new-model, structured-output, speculative-decoding, ready, ci/build, v1, tool-calling, kv-connector
BODY: See: https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it ⏎  ⏎ Docker images are available under: `vllm-openai:gemma-cu130`

### L3-fcf5115c45  (L3, 2026-06-12, sha fcf5115c45b9, PR #44899)
TITLE: [ROCm][DSv4][Perf] Flash-decode split-K decode attention kernel (#44899)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+535/-10); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+140/-0)
LABELS: rocm, ready, v1, verified
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ This PR optimizes the DeepSeek-V4 sparse **decode** attention path on ROCm (`_sparse_attn_decode_ragged_kernel`), which is GPU-occupancy-bound at small batch sizes: DSV4's 16 TP-local heads collapse to a single head-block, so the whole decode runs on just `batch` workgroups and leaves most of a 256-CU gfx950 device idle (latency flat at ~277us across batch 4 to 128). The fix rewrites it as a flash-decode split-K pipeline with a wave-a …[truncated]

### L3-a014dddbaa  (L3, 2026-06-12, sha a014dddbaa67, PR #45304)
TITLE: [11b/n] Migrate Machete kernels to torch stable ABI (#45304)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+69/-70); csrc/cutlass_extensions/vllm_cutlass_library_extension.py (+7/-7); csrc/libtorch_stable/quantization/machete/Readme.md (+0/-0); csrc/libtorch_stable/quantization/machete/generate.py (+18/-18); csrc/libtorch_stable/quantization/machete/machete_collective_builder.cuh (+0/-0); csrc/libtorch_stable/quantization/machete/machete_interleaving_utils.cuh (+0/-0); csrc/libtorch_stable/quantization/machete/machete_mainloop.cuh (+0/-0); csrc/libtorch_stable/quantization/machete/machete_mm_kernel.cuh (+30/-27); csrc/libtorch_stable/quantization/machete/machete_mm_launcher.cuh (+80/-0); csrc/libtorch_stable/quantization/machete/machete_prepack_kernel.cuh (+3/-2); (+7 more)
LABELS: rocm, ready, ci/build, nvidia
BODY: ## Purpose ⏎ This PR continues the libtorch stable ABI migration (see #26946) for vLLM Machete CUDA kernels by moving the Machete tree (host entry point, launchers, CUTLASS/CuTe device headers, and codegen) into `csrc/libtorch_stable/quantization/machete/`, registering `machete_mm`, `machete_prepack_B`, and `machete_supported_schedules` on `_C_stable_libtorch`, and removing them from legacy `_C`. CMake Machete generation and SM90a gencode are swit …[truncated]

### L3-88ed636218  (L3, 2026-06-12, sha 88ed63621866, PR #35264)
TITLE: [KV Connector]: Support KV push from Prefill to Decode node using Nixl KV Connector (#35264)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/design/nixl_kv_push_connector.md (+256/-0); examples/disaggregated/disaggregated_serving/disagg_proxy_pushconnector_demo.py (+429/-0); tests/v1/kv_connector/unit/test_bidirectional_kv_transfer.py (+4/-4); tests/v1/kv_connector/unit/test_multi_connector.py (+5/-2); tests/v1/kv_connector/unit/test_nixl_connector.py (+43/-36); tests/v1/kv_connector/unit/test_nixl_connector_hma.py (+6/-2); tests/v1/kv_connector/unit/test_nixl_push_connector.py (+815/-0); tests/v1/kv_connector/unit/test_nixl_simple_cpu_offload.py (+1/-1); tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py (+3/-1); tests/v1/kv_connector/unit/utils.py (+63/-0); (+16 more)
LABELS: documentation, ready, v1, kv-connector
BODY: RFC https://github.com/vllm-project/vllm/issues/36923 ⏎  ⏎ Implemented KV push feature where Prefill node pushes ⏎ its KV blocks to Decode node as soon as the model executor ⏎ completes the forward pass and finishes request. ⏎ The implementation supports heterogeneous TP and ⏎ heterogeneous block sizes between P and D nodes ⏎  ⏎ And it covers both the scenarios: ⏎ Scenario 1: D registers blocks with P before P finishes generating KV ⏎ Scenario 2: P has the …[truncated]

### L3-b7f9b6ab27  (L3, 2026-06-12, sha b7f9b6ab271f, PR #42206)
TITLE: [Metrics] Add group-aware KV cache capacity to vllm:cache_config_info (#42206)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/entrypoints/serve/instrumentator/test_metrics.py (+11/-0); tests/v1/core/test_kv_cache_utils.py (+6/-0); vllm/config/cache.py (+10/-0); vllm/v1/core/kv_cache_utils.py (+19/-24); vllm/v1/engine/__init__.py (+3/-0); vllm/v1/engine/core.py (+12/-0); vllm/v1/engine/core_client.py (+14/-2)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎   Addresses the Prometheus vs. startup-log discrepancy for KV cache capacity in #42024. ⏎  ⏎   The startup log already reports the correct group-aware KV cache capacity for  hybrid models, but Prometheus did not expose matching info in 'vllm:cache_config_info`. This PR adds two: ⏎  ⏎   - `kv_cache_size_tokens` : Per-DP-engine KV cache capacity in tokens (group-aware). Uses group-aware capacity since `num_gpu_blocks * block_size` can be  …[truncated]

### L3-efe7adb5e1  (L3, 2026-06-12, sha efe7adb5e145, PR #45322)
TITLE: [Perf] Use native DSA indexer decode path for next_n > 2 on SM100 (#45322)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+15/-11)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ On SM100, the DeepGEMM paged MQA logits kernels support any `next_n` natively (multi-atom decomposition, FP8 and FP4 indexer caches), so the FP4 indexer cache no longer needs the flattening fallback for `next_n > 2` — flattening is kept only outside the SM100 family, where the FP8 kernel requires `next_n ∈ (1, 2)` (deepgemm `smxx_fp8_fp4_paged_mqa_logits.hpp:233`). Implements the in-tree `TODO (matt): integrate kernel with next_n =  …[truncated]

### L3-9eaacb23ec  (L3, 2026-06-12, sha 9eaacb23ec18, PR #45295)
TITLE: [Kernel] Consolidate Marlin thread-tile padding across all dense Marlin paths (#45295)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/quantization/test_marlin_tile_padding.py (+470/-0); vllm/model_executor/kernels/linear/mixed_precision/marlin.py (+68/-23); vllm/model_executor/layers/quantization/awq_marlin.py (+5/-2); vllm/model_executor/layers/quantization/modelopt.py (+1/-0); vllm/model_executor/layers/quantization/utils/marlin_utils.py (+117/-9); vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py (+64/-14); vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py (+45/-18)
LABELS: ready, quantization
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Marlin GEMM/repack require the rank-local (N, K) to match a thread-tile family: (n%64, k%128) or (n%128, k%64). TP sharding can produce shapes satisfying neither — FP4/FP8 paths crash at repack (e.g. `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` at TP4) and GPTQ/AWQ/WNA16 reject Marlin and fall back to slower kernels. ⏎  ⏎ Fix: zero-pad weights/scales to the nearest valid tile with one shared mechanism (`marlin_padded_nk` + pad helpers …[truncated]

### L3-4171ae406c  (L3, 2026-06-12, sha 4171ae406cdc, PR #39457)
TITLE: [V1][Metrics] Add MLA attention metrics for DeepSeek MFU estimation (#39457)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/metrics/test_perf_metrics.py (+315/-0); vllm/v1/metrics/perf.py (+285/-0)
LABELS: ready, v1, deepseek
BODY: ## Purpose ⏎   ⏎ Adds `MLAAttentionMetrics` to the V1 performance metrics module to correctly estimate FLOPs and memory bandwidth for models using Multi-Latent Attention (MLA), such as DeepSeek-V2, DeepSeek-V3, and DeepSeek-R1. ⏎   ⏎ The existing `AttentionMetrics` assumes standard MHA or GQA attention patterns with separate K/V projections and large KV cache representations. MLA introduces a fundamentally different architecture: ⏎   ⏎ - Compressed KV  …[truncated]

### L3-04cec9e4d8  (L3, 2026-06-12, sha 04cec9e4d846, PR #45240)
TITLE: [XPU][DeepSeek-V4] Fix MTP: sync with upstream fixes #44821 and #43746 (#45240)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/xpu/mtp.py (+29/-18)
LABELS: intel-gpu, ready, deepseek
BODY: ## Summary ⏎  ⏎ Sync the XPU DeepSeek V4 MTP implementation with two upstream fixes: ⏎  ⏎ 1. **#44821** — `fix: prefix DeepSeek V4 MTP projections` ⏎    - Pass explicit `prefix=` to `e_proj` and `h_proj` `ReplicatedLinear` layers so compressed-tensors can match artifact-side ignore/target rules. ⏎  ⏎ 2. **#43746** — `[Model Refactoring] Remove torch compile dependency in DSv4` ⏎    - Remove `@support_torch_compile` decorator, align with breakable cudagraph path  …[truncated]

### L3-badddd254f  (L3, 2026-06-12, sha badddd254f74, PR #45103)
TITLE: [ROCm][DSV4][Perf] Fuse inverse-RoPE and cache bf16 wo_a in o-projection (#45103)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+126/-59); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+215/-0)
LABELS: rocm, ready, v1, verified, DSv4
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ The ROCm DeepSeek V4 o-projection (rocm_inv_rope_einsum, run once per decode layer) ran the inverse GPT-J RoPE as ~10 small PyTorch kernels  (clone, index_select, 2x repeat_interleave, neg, stack, cat, casts) and re-dequantized the static fp8 wo_a weight every step (fp8->fp32->*scale ->bf16), which showed up in the decode profile as the two largest copy/mul kernels. ⏎     - Add `_inverse_rope_gptj_kernel` (Triton): single-launch invers …[truncated]

### L3-78e7293bb1  (L3, 2026-06-13, sha 78e7293bb157, PR #45277)
TITLE: [Build] Fix CUDA arch build coverage gaps (#45277)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake
FILES: .buildkite/release-pipeline.yaml (+16/-3); .github/workflows/scripts/build.sh (+5/-2); CMakeLists.txt (+95/-88); cmake/external_projects/qutlass.cmake (+22/-10); cmake/utils.cmake (+2/-2); csrc/libtorch_stable/cuda_vec_utils.cuh (+1/-1); csrc/libtorch_stable/moe/dsv3_router_gemm_entry.cu (+1/-2); csrc/libtorch_stable/quantization/fp4/mxfp4_experts_quant.cu (+63/-12); csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh (+4/-4); csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu (+9/-3); (+4 more)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ This PR fixes several CUDA architecture build/dispatch gaps found while auditing ⏎ the CUDA arch matrix in #45260. ⏎  ⏎ The main theme is making top-level arch requests, per-component CMake arch ⏎ filters, and runtime support checks agree with each other. Before this change, ⏎ some paths either carried misleading PTX settings that were filtered out, built ⏎ for arch families that the runtime never used, or advertised runtime support for ⏎  …[truncated]

### L3-9fd737badc  (L3, 2026-06-14, sha 9fd737badcc5, PR #45487)
TITLE: [Bugfix][DCP] Fix illegal memory access in DCP a2a decode under full CUDA graphs (#45487)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/dcp_alltoall.py (+10/-11)
LABELS: bug, ready, v1, nvidia
BODY: ## Summary ⏎  ⏎ `--dcp-comm-backend a2a` crashes every worker with a CUDA illegal memory access on the first request when full CUDA graphs are enabled (cudagraph_mode=FULL_AND_PIECEWISE) for DCP4 when running model `nvidia/Kimi-K2.5-NVFP4`. The DCP a2a combine drew its all-to-all send/recv buffers from the shared, growable WorkspaceManager. A full CUDA graph bakes in those buffer addresses at capture; vLLM's post-capture warmup then regrows the wor …[truncated]

### L3-8760f972ca  (L3, 2026-06-14, sha 8760f972caf5, PR #45391)
TITLE: [CPU] Refine CPU attention frontend (#45391)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+88/-202); csrc/cpu/cpu_attn_impl.hpp (+8/-7); csrc/cpu/generate_cpu_attn_dispatch.py (+1/-1); tests/kernels/attention/test_cpu_attn.py (+287/-7)
LABELS: ready, v1, cpu
BODY: ## Purpose ⏎  ⏎ - Remove SDPA ⏎ - Move ISA check to the init ⏎ - Add head_dim 48 ⏎  ⏎ ## Test Plan ⏎  ⏎ CI tests ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-b8336c3c7c  (L3, 2026-06-14, sha b8336c3c7c29, PR #45564)
TITLE: [Bugfix][V1] Split V2 model-runner attention groups on num_heads_q (#45564)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/attn_utils.py (+7/-3)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ The V2 model runner's `init_attn_backend` (`vllm/v1/worker/gpu/attn_utils.py`) keys ⏎ attention metadata-builder groups on `(backend, kv_cache_spec)` only. The V1 runner ⏎ (`gpu_model_runner.initialize_attn_backend`) additionally keys on per-rank ⏎ `num_heads_q`, with a comment explaining why: layers with different Q-head counts ⏎ (e.g. a spec-decode draft head vs its target) must get **separate** metadata builders ⏎ because a builder's scratch …[truncated]

### L3-48df95c43e  (L3, 2026-06-15, sha 48df95c43e05, PR #45458)
TITLE: [Feature][Frontend] Report multimodal token counts in usage.prompt_tokens_details (#45458)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/entrypoints/openai/chat_completion/test_serving_chat.py (+37/-1); vllm/entrypoints/openai/chat_completion/serving.py (+51/-13); vllm/entrypoints/openai/engine/protocol.py (+5/-0)
LABELS: frontend, ready
BODY: ## Purpose ⏎  ⏎ `usage.prompt_tokens` already includes the image / audio / video placeholder tokens, but clients cannot tell how many tokens each modality contributed. This adds `image_tokens`, `audio_tokens` and `video_tokens` to `usage.prompt_tokens_details`, following sgl-project/sglang#27122. `audio_tokens` mirrors the OpenAI field of the same name; `image_tokens` and `video_tokens` are multimodal extensions beyond the OpenAI schema. ⏎  ⏎ The per …[truncated]

### L3-5ed15f42b9  (L3, 2026-06-15, sha 5ed15f42b93d, PR #43557)
TITLE: Fix the E8M0 scale computation in the MXFP4 (W4A4) MOE CUTLASS kernel (#43557)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh (+33/-16); tests/kernels/moe/test_mxfp4_moe.py (+219/-0); vllm/model_executor/kernels/linear/mxfp4/flashinfer.py (+1/-1)
LABELS: bug, ready, nvidia, quantization, verified
BODY: ## Purpose ⏎  ⏎ Fix the E8M0 scale computation in the MXFP4 (W4A4) MOE CUTLASS kernel, which ⏎ currently produces severely incorrect quantization results. ⏎  ⏎ Related PR: #37463 ⏎  ⏎ ## Root Cause ⏎  ⏎ The original code computes the E8M0 scale factor via: ⏎  ⏎ ```c++ ⏎ // Old (buggy): ⏎ float SFValue = vecMax * reciprocal_approximate_ftz(6.0f);  // vecMax / 6.0 ⏎ uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23; ⏎ fp8SFVal = tmp & 0xff;  // scale_exp  …[truncated]

### L3-fa63bb9db6  (L3, 2026-06-15, sha fa63bb9db6f4, PR #43914)
TITLE: Remove redundant Triton KV cache dtype asserts and enforce architectural support (fp8 >= sm89) (#43914)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+20/-0); vllm/v1/attention/ops/triton_reshape_and_cache_flash.py (+13/-39)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ On the Triton attention backend (`TRITON_ATTN`), an fp8 KV cache is stored via ⏎ `current_platform.fp8_dtype()` == `float8_e4m3fn` (Triton `fp8e4nv`), which has ⏎ no lowering before **SM89**; a `bfloat16` KV cache needs native bf16 (**SM80+**). ⏎ Neither is caught early today: on older GPUs the engine loads and then dies deep ⏎ in inductor autotuning with `type fp8e4nv not supported in this architecture`, ⏎ surfacing only as an opaque *"Engine  …[truncated]

### L3-e18fe932ca  (L3, 2026-06-15, sha e18fe932ca61, PR #45061)
TITLE: [Perf] Optimize DSv4 prefill chunk planning, 4.0% E2E Throughput Improvement (#45061)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/nvidia/flashmla.py (+12/-20); vllm/v1/attention/backends/mla/sparse_swa.py (+100/-3); tests/kernels/attention/test_flashmla_sparse.py (+21/-0)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Optimize DSv4 prefill chunk planning, instead of fixed 4, we can pour in as much requests as possible for one chunk ⏎  ⏎ For example ⏎  ⏎ ```bash ⏎ seq_lens   = [256, 320, 448, 640, 1024, 1536, 3072, 4096, 6144, 8192] ⏎ query_lens = [4,   8,   8,   8,   16,   16,   32,   32,   32,   32] ⏎  ⏎ compress_ratio = 4 ⏎ window_size = 64 ⏎ PREFILL_CHUNK_SIZE = 4 ⏎ max_model_len = 16384 ⏎ max_num_batched_tokens = 4096 ⏎ swa_only = False ⏎ ``` ⏎  ⏎ We will ge …[truncated]

### L3-76a373eff4  (L3, 2026-06-15, sha 76a373eff47a, PR #45588)
TITLE: [Frontend] Replace legacy Gemma4 parsers with engine-based implementation (#45588)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/parser/engine/replay_harness.py (+9/-4); tests/parser/engine/test_delegating_replay.py (+18/-9); tests/parser/engine/test_gemma4_streaming_reasoning.py (+1201/-0); tests/parser/engine/test_parser_engine.py (+97/-0); tests/parser/engine/test_replay.py (+85/-1); tests/parser/engine/test_token_id_scanner.py (+528/-124); tests/parser/engine/trace_builder.py (+113/-2); tests/reasoning/test_gemma4_reasoning_parser.py (+4/-4); tests/tool_parsers/test_gemma4_tool_parser.py (+138/-51); tests/tool_use/test_gemma4_responses_adjust_request.py (+13/-6); (+10 more)
LABELS: ready, tool-calling, qwen, rust
BODY: ## Purpose ⏎  ⏎ Migrate Gemma4 from separate hand-coded reasoning and tool parsers to the unified ParserEngine framework introduced for Qwen3. A single state machine in `vllm/parser/gemma4.py` now handles both channel-based reasoning extraction and custom tool call parsing with declarative configuration. The streaming engine parser adapters are registered under the existing `gemma4` name so no user-facing configuration changes are needed. ⏎  ⏎ This c …[truncated]

### L3-ab8b0fe338  (L3, 2026-06-15, sha ab8b0fe338d0, PR #45606)
TITLE: nixl_ep: Skip post-receive quantization for NVFP4 (#45606)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_batched_moe.py (+6/-3); vllm/model_executor/layers/fused_moe/prepare_finalize/nixl_ep.py (+2/-7)
LABELS: ready, kv-connector, nvidia
BODY: After #44992, NIXL EP checks the current vLLM config to decide whether NVFP4 receive quantization should be skipped. That breaks when the receive path runs without a current config. ⏎  ⏎ Instead of relying on --moe-backend, make the decision from the MoE quant dtype. This matches non-hybrid DeepEP low-latency behavior: NVFP4 receives a regular activation tensor, and FlashInfer CUTEDSL quantizes the input inside the expert path. ⏎  ⏎ Also, ensure VLLM …[truncated]

### L3-f4359a70f9  (L3, 2026-06-16, sha f4359a70f9e0, PR #44892)
TITLE: [DSV4][Minor] Fix supported KV cache dtypes (#44892)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/design/attention_backends.md (+2/-2); vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py (+6/-5); vllm/models/deepseek_v4/sparse_mla.py (+0/-1)
LABELS: documentation, ready, nvidia
BODY: Fixes a bug introduced in #44699  ⏎ FlashMLA doesn't support bfloat16, and FalshInfer doesn't support `fp8_ds_mla`

### L3-0a1c5034f5  (L3, 2026-06-16, sha 0a1c5034f5e4, PR #45381)
TITLE: [Model] Add MiniMax M3 support (#45381)
SOURCES: path_core, dependency_pin, corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.registry
FILES: CMakeLists.txt (+2/-0); cmake/external_projects/fmha_sm100.cmake (+50/-0); requirements/common.txt (+1/-0); requirements/test/cuda.txt (+1/-0); requirements/test/rocm.txt (+2/-0); requirements/test/xpu.txt (+1/-0); setup.py (+19/-0); .gitignore (+3/-0); csrc/libtorch_stable/activation_kernels.cu (+77/-41); csrc/libtorch_stable/fp32_router_gemm.cu (+44/-41); (+98 more)
LABELS: documentation, new-model, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, gpt-oss, nvidia
ISSUES: #45360 [Feature]: Support for MiniMax Sparse Attention (MSA)
BODY: ## Summary ⏎  ⏎ - Add MiniMax M3 model support across config, processors, model registry, AMD/NVIDIA model implementations, MTP, sparse attention, and warmup paths. ⏎ - Add MiniMax M3 reasoning and tool parsers, including Rust frontend registrations and Python-facing parser wrappers. ⏎ - Add supporting kernels, quantization paths, router GEMM shape support, and targeted tests. ⏎  ⏎ ## Duplicate-work check ⏎  ⏎ - Open PR searches for `MiniMax M3` and `min …[truncated]

### L3-6607a80dab  (L3, 2026-06-16, sha 6607a80dabfa, PR #45553)
TITLE: [Bugfix][Gemma4] Fix offline parser truncation, adjust_request token leak, and chat template sync (#45553)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: examples/tool_chat_template_gemma4.jinja (+64/-40); tests/reasoning/test_gemma4_reasoning_parser.py (+1/-1); tests/renderers/test_gemma4_chat_template.py (+2/-2); vllm/parser/gemma4.py (+20/-3); vllm/tool_parsers/gemma4_utils.py (+7/-28)
LABELS: bug, documentation, ready, tool-calling
ISSUES: #39069 [Bug]: gemma4_utils._parse_tool_arguments truncates string values containing internal quotes | #39130 [Bug]: `--reasoning-parser gemma4` silently disables structured output (xgrammar) when `enable_thinking=false`
BODY: ## Purpose ⏎  ⏎ Fixes remaining Gemma4 tool calling bugs not addressed by #45588 (ParserEngine rewrite): offline parser truncation, reasoning token leak when thinking is disabled, and chat template sync with upstream HuggingFace model PRs. ⏎  ⏎ Fixes #39069 ⏎ Fixes #39130 ⏎  ⏎ ### Bugs fixed ⏎  ⏎ - **#39069** — Offline `_parse_tool_arguments` in `gemma4_utils.py` truncates string values containing internal double quotes. The function replaced `<|"|>` with …[truncated]

### L3-d53f4593ce  (L3, 2026-06-16, sha d53f4593cec9, PR #44528)
TITLE: [KV Connector][Mooncake] Pipeline-parallel support for PD-disaggregated serving with Mooncake connector (#44528)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_mooncake_connector.py (+503/-5); tests/v1/kv_connector/unit/test_mooncake_connector_hma.py (+20/-4); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py (+149/-24); vllm/distributed/kv_transfer/kv_transfer_state.py (+8/-1)
LABELS: ready, v1, kv-connector
BODY: ## Purpose ⏎  ⏎ This PR adds Mooncake-specific support for pipeline-parallel prefill in PD-disaggregated serving. It is intended for long-context workloads where the prefill side needs PP to fit or run efficiently on H20-class devices. ⏎ Decode-side PP is intentionally not part of this PR. ⏎  ⏎ ## Test Plan ⏎  ⏎ ### Unit tests ⏎  ⏎ Run the Mooncake unit suites from a Kubernetes validation pod using the PR source checkout: ⏎  ⏎ ```bash ⏎ python3 -m pytest -v  …[truncated]

### L3-bf5149b516  (L3, 2026-06-16, sha bf5149b51606, PR #36616)
TITLE: [Bugfix] Fix FlashMLA sparse accuracy with topk_length and zero-init padding (#36616)
SOURCES: path_core, subject_keyword, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.mla.flashmla_sparse
FILES: vllm/v1/attention/backends/mla/flashmla_sparse.py (+13/-3)
LABELS: bug, ready, v1
ISSUES: #36524 [Bug]: Accuracy Issue with FlashMLA Sparse on DeepSeek V3.2
DEEP_STUDY: deep-study correctness case vllm:bf5149b516: class=shape_alignment_edge; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: Pass topk_length to flash_mla_sparse_fwd for precise attention masking and use new_zeros instead of new_empty for BF16 head padding. ⏎  ⏎ Closes #36524

### L3-eb04c769d3  (L3, 2026-06-16, sha eb04c769d38d, PR #43050)
TITLE: feat: MLA prefill enable FA4 fp8 output (#43050)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fa4_cutedsl, L3.mla.common_v1, L3.mla.rocm_aiter, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/mla_attention.py (+38/-8); vllm/v1/attention/backend.py (+1/-0); vllm/v1/attention/backends/mla/prefill/base.py (+9/-0); vllm/v1/attention/backends/mla/prefill/flash_attn.py (+23/-0); vllm/v1/attention/backends/mla/prefill/flashinfer.py (+2/-0); vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py (+2/-0); vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py (+2/-0); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+7/-0); vllm/vllm_flash_attn/flash_attn_interface.py (+8/-0); benchmarks/attention_benchmarks/benchmark.py (+82/-1); (+3 more)
LABELS: performance, rocm, ready, ci/build, v1, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: Completes FlashAttn x Static FP8 in https://github.com/vllm-project/vllm/issues/35792 ⏎  ⏎ ## Purpose ⏎ - Enables fused static FP8 output in FA4 backend ⏎ - currently points at https://github.com/vllm-project/flash-attention/pull/135 ⏎  ⏎ ## Test Plan ⏎  ⏎  ⏎  ⏎ ## Test Result ⏎  ⏎ Eval ⏎ ``` ⏎ ============================================ ⏎   EVAL SUMMARY ⏎ ============================================ ⏎  ⏎ Config:      mla_fa4_fp8_output/b200_dscoder_v2_lite_fp8_ev …[truncated]

### L3-3d34f8cbdc  (L3, 2026-06-16, sha 3d34f8cbdcc9, PR #44178)
TITLE: [ROCm][Cleanup] Remove stale AITER FA hybrid KV-cache TODO (#44178)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+0/-2)
LABELS: documentation, rocm, ready, v1
BODY: ## Summary ⏎  ⏎ Remove a stale TODO from the ROCm AITER FA shuffled KV-cache update path. ⏎  ⏎ The TODO said correct hybrid-model KV-cache handling still needed to be added because Mamba state could make the KV cache non-contiguous. Since I think #43660, the ROCm AITER FA backend uses block-first KV-cache layout and the shuffled writer uses the actual key/value cache block strides, so the TODO no longer describes the current implementation. ⏎  ⏎ ## Tes …[truncated]

### L3-a8c86eeb16  (L3, 2026-06-16, sha a8c86eeb1695, PR #45306)
TITLE: [Quant] Support modelopt_mixed on Ampere (SM80/SM86) (#45306)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/modelopt.py (+7/-1)
LABELS: ready, verified
BODY: # Support modelopt_mixed on Ampere (SM80/SM86) ⏎  ⏎ ## Purpose ⏎  ⏎ Enables `modelopt_mixed` checkpoints (NVFP4 routed experts + FP8 weight-only ⏎ dense layers) on Ampere — A100/SM80 and RTX 30-series/SM86. ⏎  ⏎ `ModelOptMixedPrecisionConfig` rejected these GPUs at engine-config validation ⏎ ("Minimum capability: 89"), even though `modelopt_mixed` does not require ⏎ native FP8 tensor cores. Its layers run on Marlin, which needs only cc ≥ 7.5: ⏎  ⏎ - NVFP4 r …[truncated]

### L3-b8bd773fe4  (L3, 2026-06-16, sha b8bd773fe415, PR #45758)
TITLE: [XPU] Fix Triton attn fp8/bf16 check failing (#45758)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+23/-20); vllm/v1/attention/ops/triton_reshape_and_cache_flash.py (+2/-2)
LABELS: intel-gpu, ready, v1
BODY: ## Purpose ⏎  ⏎ The compute capability checks added in https://github.com/vllm-project/vllm/pull/43914 are CUDA-specific and incorrectly reject XPU devices. This PR skip these checks on XPU. ⏎  ⏎  ⏎ ## Test Plan ⏎ `python3 examples/basic/offline_inference/generate.py --model Qwen/Qwen3-0.6B   --enforce-eager -tp 1 --attention-backend TRITON_ATTN --kv-cache-dtype fp8` ⏎  ⏎ ## Test Result ⏎ ``` ⏎ -------------------------------------------------- ⏎ Prompt: 'H …[truncated]

### L3-a7fdfeef72  (L3, 2026-06-16, sha a7fdfeef7232, PR #45690)
TITLE: [CPU] Support Gemma Diffusion (#45690)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+16/-5); csrc/cpu/cpu_attn.cpp (+14/-25); csrc/cpu/cpu_attn_impl.hpp (+62/-27); csrc/cpu/cpu_fused_moe.cpp (+5/-7); csrc/cpu/torch_bindings.cpp (+10/-8); tests/kernels/attention/test_cpu_attn.py (+100/-25); vllm/_custom_ops.py (+6/-3)
LABELS: ready, v1, cpu
BODY: ## Purpose ⏎  ⏎ - Add dynamic causal support in CPU attention ⏎ - Optimize CPU fused moe ```gelu_tanh``` ⏎  ⏎ ## Test Plan ⏎  ⏎ CI tests ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-ced32bb474  (L3, 2026-06-16, sha ced32bb474fe, PR #42425)
TITLE: [Perf] Add VLLM_TRITON_FORCE_FIRST_CONFIG to skip Triton autotuning (#42425)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/test_force_first_config.py (+93/-0); vllm/env_override.py (+15/-0); vllm/envs.py (+9/-0); vllm/triton_utils/force_first_config.py (+92/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Triton's `@triton.autotune` benchmarks every candidate config on the first launch per `key`, then caches the winner. Three things make this painful when debugging or measuring: ⏎  ⏎ 1. **Autotune is timing-driven, so its winner is non-deterministic.** Run-to-run jitter can promote a different (BLOCK_M, num_warps, ...) tuple each time, and a different tuple reduces partial sums in a different order — so identical inputs produce numeric …[truncated]

### L3-f2beaa80c8  (L3, 2026-06-16, sha f2beaa80c8d6, PR #45725)
TITLE: [ROCm][Quant] mxfp8 moe/linear gfx950 tuning for MiniMax-M3 (#45725)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/mxfp8/rocm_native.py (+9/-2); vllm/model_executor/layers/fused_moe/experts/mxfp8_native_moe.py (+30/-5)
LABELS: rocm, ready, verified
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ native mxfp8 moe/linea tuning to improve perf ⏎  ⏎ The native MXFP8 dot_scaled MoE/linear kernels used one hardcoded tile config ⏎ (block_m=64, BLOCK_N=128, num_warps=8), which under-tiles long prefills. Gate ⏎ the tiles on token count (threshold 1024): ⏎   - prefill (>=1024): block_m=128, BLOCK_N=256, num_warps=8, num_stages=2 ⏎   - decode  (<1024) : block_m=64,  BLOCK_N=64,  num_warps=4, num_stages=2 ⏎  ⏎  ⏎ ## Test Plan ⏎  ⏎ **Serve (identica …[truncated]

### L3-71bc19dbdd  (L3, 2026-06-16, sha 71bc19dbdd0d, PR #45589)
TITLE: [Bugfix] Fix MoE model load OOM in FlashInfer_TRTLLM  backend with sleep mode (#45589)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+0/-2); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+43/-30)
LABELS: bug, ready, nvidia
BODY: ## Summary ⏎  ⏎ This PR fixes the load-time OOM reported in [#43951](https://github.com/vllm-project/vllm/issues/43951) for large BF16 MoE models using `--enable-sleep-mode` with the FlashInfer TRTLLM MoE backend. ⏎  ⏎ Instead of bypassing the `max_split_size_mb=20` allocator policy introduced by [#41268](https://github.com/vllm-project/vllm/pull/41268), **this PR reduces the transient GPU memory created by FlashInfer TRTLLM BF16 MoE weight conversio …[truncated]

### L3-2785a5e0e6  (L3, 2026-06-16, sha 2785a5e0e6ee, PR #44912)
TITLE: [Bugfix][ROCm] Fix FP8 per-tensor scale rank mismatch causing Inductor assertion failure (#44912)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/scaled_mm/pytorch.py (+6/-0)
LABELS: bug, rocm, ready
BODY: ## Purpose ⏎  ⏎ Fix `AssertionError` in Inductor's aten._scaled_mm lowering due to mismatched scale tensor ranks when AITER is enabled (`VLLM_ROCM_USE_AITER=1`)  and skinny GEMMs are disabled ( `VLLM_ROCM_USE_SKINNY_GEMM=0`) on ROCm. ⏎  ⏎ `requantize_with_max_scale` returns a 0-D  weight scale, but AITER's `_rocm_aiter_per_tensor_quant_fake` always returns a 1-D activation scale. When torch.compile lowers aten._scaled_mm, it asserts that scale_a and  …[truncated]

### L3-7b5d60cc37  (L3, 2026-06-16, sha 7b5d60cc3733, PR #45195)
TITLE: [Bugfix][V1] Clean up compiled-model bytecode hooks on VllmRunner exit (#45195)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/conftest.py (+1/-0); tests/v1/shutdown/test_delete.py (+28/-1); vllm/compilation/wrapper.py (+9/-1); vllm/v1/engine/llm_engine.py (+23/-0)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ This is a rebased continuation of #35676, originally opened by @zou3519. The old PR is still open, but the thread confirmed that opening a new PR is the preferred handoff. ⏎  ⏎ `VllmRunner` can run V1 in-process with `VLLM_ENABLE_V1_MULTIPROCESSING=0`. In that mode, compiled models can stay pinned after runner exit because `TorchCompileWithNoGuardsWrapper` registers a global TorchDynamo bytecode hook with a bound method. ⏎  ⏎ This PR st …[truncated]

### L3-b9684d99e9  (L3, 2026-06-16, sha b9684d99e9ba, PR #45795)
TITLE: [Bugfix] Gemma4: skip forced JSON for required/named tool choice (#45795)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/tool_use/test_gemma4_responses_adjust_request.py (+92/-1); vllm/tool_parsers/gemma4_engine_tool_parser.py (+28/-0)
LABELS: bug, ready, tool-calling
BODY: ## Purpose ⏎  ⏎ Fix a regression from #45588: with the engine-based Gemma4 parser ⏎ (`supports_required_and_named=False`), `tool_choice="required"` and named tool ⏎ choice return the forced `FunctionDefinition` JSON as `content` with an empty ⏎ `tool_calls` list, both streaming and non-streaming. ⏎  ⏎ Root cause: the base `ToolParser.adjust_request` constrains the model to that ⏎ JSON via structured outputs, but the native Gemma4 parser only reads the ⏎ model's na …[truncated]

### L3-9d4dc4ca2f  (L3, 2026-06-16, sha 9d4dc4ca2fef, PR #43525)
TITLE: [Kernel] Support GLM-5 dimensions for TRT-LLM ragged MLA prefill (#43525)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/prefill/base.py (+23/-4); vllm/v1/attention/backends/mla/prefill/flashinfer.py (+12/-3); vllm/v1/attention/backends/mla/prefill/selector.py (+24/-21); vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py (+12/-3); vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py (+17/-3); docs/design/attention_backends.md (+3/-3); tests/v1/attention/test_mla_backends.py (+10/-2); tests/v1/attention/test_mla_prefill_registry.py (+0/-2); tests/v1/attention/test_mla_prefill_selector.py (+44/-25); tools/pre_commit/generate_attention_backend_docs.py (+55/-11)
LABELS: documentation, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ This PR lets `TRTLLM_RAGGED` MLA prefill accept GLM-5 dimensions: ⏎  ⏎ - `DeepSeek-R1`: `(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128)` ⏎ - `GLM-5`: `(qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256)` ⏎  ⏎ It also replaces the old DeepSeek-R1-only validation flag with per-backend supported `MLADimensions` declarations, so `FLASHINFER` and `TOKENSPEED_MLA` stay limited to `(qk_nope_head_dim=128, qk_rope_head_dim=64 …[truncated]

### L3-4c62663315  (L3, 2026-06-16, sha 4c6266331598, PR #45744)
TITLE: [M3] Enable FP8 sparse GQA (#45744)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: cmake/external_projects/fmha_sm100.cmake (+8/-10); csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu (+80/-40); csrc/libtorch_stable/ops.h (+5/-1); csrc/libtorch_stable/torch_bindings.cpp (+2/-1); setup.py (+1/-1); tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py (+49/-13); vllm/_custom_ops.py (+5/-2); vllm/models/minimax_m3/amd/model.py (+7/-63); vllm/models/minimax_m3/common/sparse_attention.py (+16/-4); vllm/models/minimax_m3/nvidia/model.py (+6/-43)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Reland of #45680 after M3 is merged ⏎  ⏎ Add support for FP8 sparse GQA on NVIDIA ⏎ - Q is not quantized, only KV is ⏎ - Update the fused QKNorm+RoPE+insert to support FP8 KV cache ⏎ - Prefill kernel: MSA for sm100, Triton otherwise ⏎ - Decode kernel: Triton ⏎  ⏎ Note: `--attention-backend TRITON_ATTN` must be used since `FLASH_ATTN` doesn't support FP8 KV cache, and FlashInfer FP8 doesn't support page size 128 ⏎  ⏎ ``` ⏎           vllm serve  …[truncated]

### L3-a52205bccf  (L3, 2026-06-16, sha a52205bccfc7, PR #43098)
TITLE: [Model] Add HrmTextForCausalLM (Hierarchical Reasoning Model — Text) (#43098)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/__init__.py (+4/-0); vllm/model_executor/layers/attention/prefill_prefix_lm_attention.py (+86/-0); docs/models/supported_models.md (+1/-0); tests/models/registry.py (+4/-0); vllm/model_executor/models/hrm_text.py (+527/-0); vllm/model_executor/models/registry.py (+1/-0); vllm/v1/engine/core.py (+22/-0); vllm/v1/kv_cache_interface.py (+13/-0)
LABELS: documentation, new-model, ready, v1, verified
BODY: HRM-Text shipped in [transformers 5.9.0](https://pypi.org/project/transformers/5.9.0/) (released 2026-05-20, merged to `main` via [huggingface/transformers#46025](https://github.com/huggingface/transformers/pull/46025)). The model performs a hierarchical recurrent forward over two transformer stacks (`H` slow, `L` fast) inside nested H/L cycle loops; each recurrence step consumes a distinct KV cache slot, matching transformers' `cycle_offset` for …[truncated]

### L3-4bf699d310  (L3, 2026-06-16, sha 4bf699d31030, PR #45473)
TITLE: [Kernel] Support DS Mamba tail copy for MTP align mode (#45473)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_mamba_utils.py (+115/-0); vllm/model_executor/layers/mamba/mamba_utils.py (+6/-10); vllm/v1/worker/mamba_utils.py (+61/-24)
LABELS: ready, v1, verified
BODY: ## Purpose ⏎  ⏎ Add support for speculative decoding with `mamba_cache_mode=align` when Mamba conv state uses DS layout (`[num_blocks, dim, state_len]`). ⏎  ⏎   For DS layout and accepted-token offset > 0, the conv-state tail is strided and cannot use the existing contiguous memcpy path. This PR adds a DS-tail copy spec and a Triton copy kernel for: ⏎  ⏎   ```text ⏎   dst[:, : state_len - offset] = src[:, offset:] ⏎   ``` ⏎  ⏎   The kernel keeps pointer of …[truncated]

### L3-b831374cf1  (L3, 2026-06-17, sha b831374cf1db, PR #45832)
TITLE: [Bugfix][Gemma4] Fix parsing when thinking is disabled (#45832)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/tool_use/test_gemma4_responses_adjust_request.py (+57/-21); vllm/parser/gemma4.py (+13/-7)
LABELS: bug, ready, tool-calling
BODY: ## Purpose ⏎  ⏎ `Gemma4Parser.adjust_request` (added in #45553) returns early when thinking is ⏎ disabled (`enable_thinking=False`), so `skip_special_tokens` is never set to ⏎ `False`. With thinking disabled and tools active, the `<|tool_call>` delimiters ⏎ get stripped before the parser sees them, so tool calling breaks: `tool_calls` ⏎ is empty and the raw `call:fn{...}` body leaks into `content`, both streaming ⏎ and non-streaming. ⏎  ⏎ The engine-based Gemma4 p …[truncated]

### L3-efd15e192a  (L3, 2026-06-17, sha efd15e192a1a, PR #45720)
TITLE: [Bugfix][ROCm] Fix MiniMax-M3 FP8 KV cache dtype (#45720)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_minimax_m3.py (+44/-0); vllm/models/minimax_m3/common/ops/sparse_attn.py (+8/-2); vllm/models/minimax_m3/common/sparse_attention.py (+8/-3)
LABELS: bug, rocm, ready, verified
ISSUES: #45562 [Bug]: ROCm MI300X FP8 KV cache MiniMax-M3-MXFP8 accuracy issues
BODY: ## Purpose ⏎  ⏎ Fixes #45562. ⏎  ⏎ MiniMax-M3's sparse-attention backend reinterprets the byte-backed FP8 KV ⏎ cache as `torch.float8_e4m3fn` for every E4M3 configuration. That is incorrect ⏎ on ROCm gfx942, where `current_platform.fp8_dtype()` is ⏎ `torch.float8_e4m3fnuz`. ⏎  ⏎ FN and FNUZ use different encodings. Reinterpreting FNUZ cache bytes as FN ⏎ changes the K/V values before the sparse-attention kernels consume them. The ⏎ prefill and decode wrappers also omi …[truncated]

### L3-3c6084bb0d  (L3, 2026-06-17, sha 3c6084bb0d51, PR #45852)
TITLE: [Bugfix][Gemma4] Pre-initialise streaming reasoning state when prompt ends inside an open `<|channel>` (fixes #45834) (#45852)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/parser/engine/test_gemma4_streaming_reasoning.py (+208/-0); vllm/parser/abstract_parser.py (+7/-0); vllm/parser/engine/adapters.py (+3/-0); vllm/parser/engine/parser_engine.py (+12/-0); vllm/parser/gemma4.py (+28/-0); vllm/reasoning/abs_reasoning_parsers.py (+12/-0)
LABELS: bug, ready, tool-calling
ISSUES: #45834 [Bug]: [Gemma4] Post-tool reasoning can leak after final tool response with enable_thinking=true
BODY: ## Purpose ⏎  ⏎ Fixes #45834. ⏎  ⏎ After #45553 the Gemma4 tool chat template inserts `<|channel>thought\n` into ⏎ the generation prompt when continuing a turn after a tool response with ⏎ `enable_thinking=True`. The prompt token IDs therefore end inside an **open ⏎ reasoning channel** (a `<|channel>` start with no matching `<channel|>` close). ⏎  ⏎ `Gemma4Parser`'s engine, however, always starts in `ParserState.CONTENT`. As a ⏎ result the first generated  …[truncated]

### L3-0a7bacdcac  (L3, 2026-06-17, sha 0a7bacdcacc5, PR #45863)
TITLE: [DSv4 Perf] DSv4 flashinfer sparse index cache for metadata, 2%~4% TTFT improvement (#45863)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/sparse_swa.py (+4/-1); tests/kernels/attention/test_flashmla_sparse.py (+147/-0); vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py (+33/-17)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Part of https://github.com/vllm-project/vllm/issues/45861 ⏎  ⏎ We rebuild the same metadata for every layer, this can be easily optimized using a small cache ⏎  ⏎ ## Test ⏎  ⏎ `vllm serve deepseek-ai/DeepSeek-V4-Pro   --trust-remote-code   --kv-cache-dtype fp8   --block-size 256   --enable-expert-parallel -dp 8   --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}'   --attention_config.use_fp4_indexer_cache …[truncated]

### L3-556b063e45  (L3, 2026-06-17, sha 556b063e45a7, PR #44468)
TITLE: [XPU] Fix test_spec_decode_logprobs: use FLASH_ATTN for XPU in GPU_DETERMINISM_KWARGS (#44468)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/intel_jobs/misc_intel.yaml (+2/-1); tests/v1/sample/test_logprobs.py (+7/-6)
LABELS: intel-gpu, ready, ci/build, v1
BODY: ## Summary ⏎  ⏎ Use `FLASH_ATTN` for XPU in `GPU_DETERMINISM_KWARGS` to avoid logprob mismatch beyond `abs_tol=0.1` in spec-decode tests. ⏎  ⏎ ## Changes ⏎ - `tests/v1/sample/test_logprobs.py`: rename `ROCM_DETERMINISM_KWARGS` → `GPU_DETERMINISM_KWARGS`, add XPU→FLASH_ATTN case ⏎ - `.buildkite/intel_jobs/misc_intel.yaml`: enable `test_logprobs.py` on XPU CI

### L3-d537122398  (L3, 2026-06-17, sha d537122398df, PR #45782)
TITLE: [ROCm][Bugfix]: Fallback GFX942 sparse MLA ops to Triton (#45782)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+15/-34)
LABELS: bug, rocm, ready, v1
BODY: ## Purpose ⏎ This PR resolves severe generation accuracy degradation, numerical instability occurring in sparse MLA-based models on ROCm platforms. ⏎  ⏎ During our internal evaluations on the main branch, we identified a critical regression: ⏎ - GLM-5.1-FP8 suffered from a massive accuracy collapse (dropping to ~55% on GSM8K). ⏎ - DeepSeek-V3.2 experienced a slight but noticeable accuracy degradation (~3% drop on GSM8K). ⏎  ⏎ ### Key Changes: ⏎ Fallback  …[truncated]

### L3-20a5f8b43b  (L3, 2026-06-17, sha 20a5f8b43ba5, PR #45232)
TITLE: [FlexAttention] make custom mask mods fully cudagraphable (#45232)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+31/-1); tests/kernels/test_flex_attention.py (+67/-0)
LABELS: ready, v1, nvidia
BODY: **sumary** ⏎  ⏎ previously i added https://github.com/vllm-project/vllm/pull/37692 which allowed users to pass in a custom mask mod, but after https://github.com/vllm-project/vllm/pull/36298 landed enabling full cudagraphs, we need to change where we build the masks.  ⏎  ⏎ instead of building in the forward per layer, we do so before in the `build()` function that creates `FlexAttentionMetadata` ⏎  ⏎ same thing for the native KV cache spec, we need to  …[truncated]

### L3-e28e8c8782  (L3, 2026-06-17, sha e28e8c87820d, PR #45854)
TITLE: [ROCm][Quant] Minimax-M3:  Enable fp8_per_channel for bf16 weights on mi300x (#45854)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/model_executor/layers/fused_moe/config.py (+4/-0); vllm/model_executor/layers/fused_moe/oracle/fp8.py (+2/-0); vllm/model_executor/layers/quantization/fp8.py (+2/-0); vllm/model_executor/layers/quantization/online/fp8.py (+2/-0); vllm/platforms/rocm.py (+1/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Improve the perf of Minimax-M3 bf16 model on MI300x (gfx942)/ ⏎  ⏎ The fp8 w8a8 MoE quant config dropped the SwiGLU-OAI alpha/beta that models ⏎ such as MiniMax-M3 pass to FusedMoE (swiglu_alpha=1.702, swiglu_beta=1.0). ⏎ Only swiglu_limit was forwarded, so the silu_and_mul_with_clamp kernel ran ⏎ with its default alpha=1.0/beta=0.0 and produced garbage (gsm8k 0.00) on both ⏎ the serialized (Fp8MoEMethod) and online (_Fp8OnlineMoEBase) fp8  …[truncated]

### L3-8b2b566ea7  (L3, 2026-06-17, sha 8b2b566ea710, PR #43853)
TITLE: Feature: Enable Flashinfer non-gated MoE bf16 (#43853)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py (+10/-3); vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+12/-0); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+8/-5)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ Add support for flashinfer non-gated MoE bf16. Note that this PR makes flashinfer-trtllm backend the new default for non-gated MoE bf16 models for sm100. ⏎  ⏎ Shows perf gain of ~15% e2e for below benchmark. ⏎  ⏎ ## Test Result ⏎  ⏎ MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 ⏎ on 1 gb200 node. ⏎  ⏎ ``` ⏎ perf cmd: ⏎ vllm bench serve --backend vllm --host 0.0.0.0 --port 8000 --model $MODEL --num-prompts 256 --trust-remote-code --ignore-eos  …[truncated]

### L3-eb0fdeb1e8  (L3, 2026-06-17, sha eb0fdeb1e834, PR #45831)
TITLE: [Bugfix][PD] Fix DSV4 disaggregated serving (#45831)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+4/-1)
LABELS: bug, ready, kv-connector
BODY: ### Bug  ⏎ Recent changes broke DSV4 PD disagg.  ⏎  ⏎ The `is_mla_region` check in `base_worker.py` only recognizes MLAAttentionSpec, but DSV4's compressor layers are `SlidingWindowMLASpec`.  ⏎  ⏎ ### Fix ⏎ Include `SlidingWindowMLASpec` in the isinstance check so these regions are treated as MLA region. ⏎  ⏎ ### Test ###  ⏎ - DSV4 Flash 4P4D on 8xH100 starts and serves requests after this fix. ⏎ - PR #42310 adds test cases for DSV4 PD disagg to nightly.

### L3-5e27b2baf4  (L3, 2026-06-17, sha 5e27b2baf481, PR #45917)
TITLE: [Bugfix] Pass TP group to FlashInfer all-reduce fusion (#45917)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+1/-8); vllm/distributed/device_communicators/flashinfer_all_reduce.py (+1/-0)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎  ⏎ ### Symptom ⏎ Serving Nemotron-3-Ultra-NVFP4 with `--tensor-parallel-size > 1` and `--data-parallel-size > 1` hangs at startup during CUDA-graph capture / warmup: workers freeze and `EngineCore` repeatedly logs `No available shared memory broadcast block found in 60 seconds`. ⏎ The server never becomes ready. ⏎  ⏎ Bug reproduced on both H100 and GB200. ⏎  ⏎ Using `TP>1/DP=1`, `TP=1/DP>1`, and `TP=1/DP=1` works (with expert parallelism on  …[truncated]

### L3-091386a99b  (L3, 2026-06-17, sha 091386a99b95, PR #45794)
TITLE: [Bugfix] MiniMax-M3 (AMD): add packed_modules_mapping and pass swiglu… (#45794)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/quark/quark_moe.py (+3/-0); vllm/models/minimax_m3/amd/model.py (+12/-2)
LABELS: bug, rocm, ready, verified
BODY: ## Purpose ⏎  ⏎ Running MiniMax-M3 on AMD with Quark MXFP4 quantization currently fails at ⏎ weight loading with: ⏎  ⏎ AssertionError ⏎ ... param_data.shape != loaded_weight.shape ⏎  ⏎ The root cause is that the fused `qkv_proj` / `gate_up_proj` modules are not ⏎ declared as packed modules, so the loader tries to copy a full fused-weight ⏎ tensor into a single shard's parameter, producing a shape mismatch. This PR ⏎ fixes the loading path and a couple of re …[truncated]

### L3-58b2e89642  (L3, 2026-06-17, sha 58b2e896423f, PR #45867)
TITLE: [Bugfix][Gemma4] Render reasoning on assistant turns without tool_calls (#45867)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: examples/tool_chat_template_gemma4.jinja (+2/-2)
LABELS: bug, documentation, ready, tool-calling
BODY: ## Purpose ⏎  ⏎ Fixes a template regression introduced by #45553: the thinking channel guard `and message.get('tool_calls')` silently dropped reasoning content on assistant messages without tool calls — e.g. the final answer after a tool chain where the model reasons about the tool result before responding. ⏎  ⏎ Fix: remove the `tool_calls` guard. Reasoning is now rendered whenever `thinking_text` and `thinking_gate` are both true, regardless of whet …[truncated]

### L3-b4092176b9  (L3, 2026-06-18, sha b4092176b9bc, PR #45448)
TITLE: [Bugfix] Complete one-shot fused all-reduce PDL at end to avoid NaN (#45448)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+11/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ The FlashInfer one-shot Lamport all-reduce signals programmatic-dependent- launch (PDL) completion before committing its output buffer when trigger_completion_at_end=False, so the next PDL-launched kernel reads the uninitialized buffer and produces NaN. The fused AR+RMSNorm pass set the flag to `num_tokens <= PDL_ADVANCE_LAUNCH_TOKENS` (#43103) -- exactly the batch=1 / spec-decode shapes, where the one-shot path is always selected - …[truncated]

### L3-5fd3b276f8  (L3, 2026-06-18, sha 5fd3b276f8fa, PR #45444)
TITLE: [Mooncake] Skip KV lookup for non-reachable SWA blocks (#45444)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py (+10/-10); tests/v1/kv_connector/unit/test_mooncake_store_worker.py (+61/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py (+40/-7); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py (+9/-1)
LABELS: ready, v1, kv-connector
BODY: ## Purpose ⏎ This PR adds some optimizations for reducing overhead in Mooncake KV offloading. ⏎ - In `lookup`, skip SWA blocks that are not eligible to be considered cache hit, using kv cache group's `reachable_block_mask`. ⏎ - Use `None` as all-True mask for `store_mask`, saving list construction overhead for full cache: `masks.append([True] * num_chunks if mask is None else mask)`. ⏎  ⏎ ## Performance benchmark: ⏎ DeepSeek v4 TP4 on 4 x GB300: ⏎  ⏎ <im …[truncated]

### L3-e2352c2974  (L3, 2026-06-18, sha e2352c29743a, PR #45706)
TITLE: [ROCm][Spec Decode] Fix probabilistic draft probs test attention backend (#45706)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/misc.yaml (+6/-0); tests/v1/spec_decode/test_eagle.py (+6/-2)
LABELS: rocm, speculative-decoding, ready, ci/build, v1
BODY: ## Purpose ⏎  ⏎ `test_propose_stores_probabilistic_draft_probs` hardcodes the `FLASH_ATTN` ⏎ attention backend, which builds `FlashAttentionMetadata`. On ROCm, the ⏎ speculative-decoding proposer only accepts Triton/Rocm/AITER metadata ⏎ (`allowed_attn_types` is built under `current_platform.is_rocm()` in ⏎ `vllm/v1/spec_decode/llm_base_proposer.py`), so the test fails on AMD MI ⏎ architectures (gfx942 / MI325, gfx950 / MI355) with: ⏎  ⏎ ``` ⏎ ValueError:  …[truncated]

### L3-afdcbd5d39  (L3, 2026-06-18, sha afdcbd5d39ea, PR #45681)
TITLE: [ROCm][DSv4] Functional fixes for DeepSeek V4 on MI300X/MI325X (#45681)
SOURCES: path_core, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+47/-14); vllm/v1/attention/ops/triton_fp8_mqa_logits.py (+262/-0); csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu (+11/-4); tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py (+147/-24); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+22/-3); vllm/models/deepseek_v4/amd/rocm.py (+4/-0); vllm/models/deepseek_v4/common/ops/cache_utils.py (+49/-6); vllm/models/deepseek_v4/nvidia/ops/o_proj.py (+3/-1)
LABELS: rocm, ready, v1, deepseek, DSv4
DEEP_STUDY: deep-study correctness case vllm:afdcbd5d39: class=numerical_precision; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Purpose ⏎  ⏎ This PR fixes DeepSeek V4 / DeepSeek V4 Flash functional issues on ROCm gfx942 (MI300X), while preserving the gfx950 path. ⏎  ⏎ The main fixes are: ⏎  ⏎ - Avoid unsupported FP8 arithmetic when converting UE8M0 block scales. ⏎ - Make the fused DeepSeek V4 qnorm/RoPE/KV-insert path emit the correct FP8 bytes on ROCm: ⏎   - gfx942 uses FNUZ encoding. ⏎   - gfx950 uses OCP encoding. ⏎   - FNUZ uses max value `224.0`. ⏎ - Propagate the FNUZ/OCP c …[truncated]

### L3-79ca54d221  (L3, 2026-06-18, sha 79ca54d2215b, PR #45040)
TITLE: [Bugfix][Quantization] Don't reject fp8_e5m2 KV cache for non-fp8 quantized checkpoints (#45040)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/attention.py (+16/-2)
LABELS: bug, ready
ISSUES: #39137 [Bug]: fp8_e5m2 kv-cache gate in _init_kv_cache_quant fires on any quantized checkpoint, not only fp8 checkpoints
BODY: ## Purpose ⏎  ⏎ Closes #39137. ⏎  ⏎ `_init_kv_cache_quant` rejects `--kv-cache-dtype fp8_e5m2` for every quantized checkpoint with "fp8_e5m2 kv-cache is not supported with fp8 checkpoints". But weight-only checkpoints (INT4/INT8 compressed-tensors, AWQ, GPTQ) carry no fp8 KV scales and are not fp8. On Ampere, where `fp8_e5m2` is the only fp8 KV cache dtype that runs at all (`fp8_e4m3` is unsupported by the hardware), this makes fp8 KV cache unreachab …[truncated]

### L3-4583630b56  (L3, 2026-06-18, sha 4583630b5621, PR #45466)
TITLE: [Bugfix][Kernel] Check output alignment in vectorize_with_alignment (fixes misaligned-address crash for non-multiple-of-8 head sizes) (#45466)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/quantization/vectorization_utils.cuh (+24/-6); tests/kernels/attention/test_cache.py (+37/-0)
LABELS: bug, ready
ISSUES: #41257 [Bug]: vLLM + FlexAttention crashes with torch._dynamo.exc.InternalTorchDynamoError: AcceleratorError: CUDA error: misaligned address
BODY: ## Purpose ⏎  ⏎ Fixes #41257. ⏎  ⏎ `FLEX_ATTENTION` with `head_size=46` (e.g. the `inferno-project/vllm-mixtral-2` fuzz model) crashes with `CUDA error: misaligned address`. Root-cause analysis (details in [the issue comment](https://github.com/vllm-project/vllm/issues/41257#issuecomment-4695960083)): the crash is not in FlexAttention, CUDA graphs, or Ada drivers — it is `reshape_and_cache_flash_kernel` issuing vectorized stores to misaligned KV-cache ro …[truncated]

### L3-2a6c6b9429  (L3, 2026-06-18, sha 2a6c6b94293e, PR #46001)
TITLE: [DeepSeek-V4] Support TEP=16 for the block-FP8 shared expert (#46001)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/nvidia/model.py (+44/-0)
LABELS: ready, deepseek, verified
BODY: ## Purpose ⏎ Currently TEP=16 is not supported for DeepSeek-V4. ⏎ The shared expert is a block-FP8 MLP (`weight_block_size=[128,128]`, ⏎ `moe_intermediate_size=3072`). A per-rank TP shard must be a whole number of ⏎ 128-blocks (a block's FP8 scale can't be split). At TP=16, `3072/16 = 192` is not ⏎ a multiple of 128, so the standard linears fail to build: ⏎ `Weight input_size_per_partition = 192 is not divisible by block_k = 128`. ⏎  ⏎ ## Layout (`Deepse …[truncated]

### L3-021cdf72bc  (L3, 2026-06-18, sha 021cdf72bc22, PR #43179)
TITLE: Fix _riscv_supports_rvv_vlen128() to detect RVV on hardware without zvl flags (#43179)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+16/-3); csrc/cpu/cpu_attn.cpp (+11/-0); csrc/cpu/torch_bindings.cpp (+3/-0)
LABELS: ready, v1, cpu
BODY: ## Purpose ⏎  ⏎ Fix `_riscv_supports_rvv_vlen128()` so that RISC-V hardware without `zvl<N>b` flags in `/proc/cpuinfo` can still correctly detect RVV support. ⏎  ⏎ On some RISC-V hardware (e.g. SG2044), `/proc/cpuinfo` advertises the V extension (`rv64imafdcv`) and `zve*` / `zvfh` flags but does **not** report any `zvl<N>b` VLEN hint. The existing Python-only check required `zvl128b` to be present: ⏎  ⏎ ```python ⏎ if "zvl128b" not in cpuinfo: ⏎     retu …[truncated]

### L3-35e4dd4a69  (L3, 2026-06-18, sha 35e4dd4a69b6, PR #45659)
TITLE: [KV Connector][Mooncake] Async lookup to reduce scheduler overhead (#45659)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/features/mooncake_store_connector_usage.md (+1/-0); tests/v1/kv_connector/unit/test_mooncake_store_connector.py (+126/-1); tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py (+8/-1); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py (+1/-1); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py (+19/-6); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py (+42/-2)
LABELS: documentation, ready, v1, kv-connector
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Looking up keys in Mooncake currently happens synchronously inside `get_num_new_matched_tokens`, on the scheduler's critical path. Each lookup costs ~1–2 ms per request on average, and that latency is paid serially during scheduling. This PR adds an **optional** async lookup mode that offloads the lookup to a background thread so its latency overlaps with the current scheduling step; results are consumed on a later step. ⏎  ⏎ When async m …[truncated]

### L3-b9a7cd464c  (L3, 2026-06-19, sha b9a7cd464c9a, PR #45415)
TITLE: [12/n]  final _C library kernel migration (#45415)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.platform.cuda_selection
FILES: CMakeLists.txt (+56/-72); cmake/external_projects/qutlass.cmake (+29/-5); setup.py (+4/-1); csrc/libtorch_stable/core/math.hpp (+0/-0); csrc/libtorch_stable/moe/moe_align_sum_kernels.cu (+1/-1); csrc/libtorch_stable/ops.h (+28/-0); csrc/libtorch_stable/quantization/activation_kernels.cu (+63/-55); csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_kernels.cu (+1/-1); csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu (+1/-1); csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh (+1/-1); (+7 more)
LABELS: rocm, ready, ci/build, nvidia
BODY: ## Purpose ⏎ This PR continues the libtorch stable ABI migration (see https://github.com/vllm-project/vllm/issues/26946) for vLLM and is the final `_C` library kernels to move to the `_C_stable_libtorch` library. The PR moves `csrc/quantization/activation_kernels.cu` to `csrc/libtorch_stable/quantization/activation_kernels.cu`, along with the `weak_ref_tensor` (defined in ops.h), `silu_and_mul_quant`, and `persistent_masked_m_silu_mul_quant` kerne …[truncated]

### L3-ab66606993  (L3, 2026-06-19, sha ab666069935c, PR #45895)
TITLE: [bugfix]Indexer init skip and MTP TopK share for iteration (#45895)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_sparse, L3.mla.flashinfer_sparse, L3.mla.rocm_aiter_sparse
FILES: vllm/model_executor/layers/attention/mla_attention.py (+6/-0); vllm/model_executor/layers/mla.py (+1/-0); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+7/-3); vllm/v1/attention/backends/mla/flashmla_sparse.py (+6/-2); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+7/-3); vllm/v1/attention/backends/mla/xpu_mla_sparse.py (+7/-3); vllm/model_executor/models/deepseek_mtp.py (+6/-2); vllm/model_executor/models/deepseek_v2.py (+22/-17); vllm/v1/spec_decode/llm_base_proposer.py (+7/-0)
LABELS: bug, rocm, intel-gpu, speculative-decoding, ready, v1, deepseek, nvidia
BODY: ## Purpose ⏎ fix GLM-5.2 BF16 init Indexer in skip_topk layer ⏎ fix GLM-5.2 MTP Port-Norm cycle ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ # setup with ⏎ VLLM_DEEP_GEMM_WARMUP=skip  vllm serve zai-org/GLM-5.2-FP8 \ ⏎   --trust-remote-code \ ⏎   --tensor-parallel-size 8 \ ⏎   --tool-call-parser glm47 \ ⏎   --enable-auto-tool-choice \ ⏎   --reasoning-parser glm45 \ ⏎   --speculative-config.method mtp \ ⏎   --speculative-config.num_speculative_tokens 5 \ ⏎   --max-num-seqs 32 \ …[truncated]

### L3-01192139bf  (L3, 2026-06-19, sha 01192139bf02, PR #44577)
TITLE: [DSv4] Pack KV caches into contiguous per-block allocations for DeepSeek V4 (#44577)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_contiguous_kv_packing.py (+135/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+98/-0); vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py (+9/-5); vllm/v1/core/kv_cache_utils.py (+12/-1); vllm/v1/kv_cache_interface.py (+2/-0); vllm/v1/worker/gpu/attn_utils.py (+41/-7); vllm/v1/worker/gpu_model_runner.py (+47/-9)
LABELS: ready, ci/build, v1, deepseek, kv-connector, nvidia
BODY: For DeepSeek V4, pack all layer data contiguously per block so that KV connectors can send/receive one region per block. ⏎  ⏎ Full-attention MLA + SWA/compressor caches share one contiguous allocation per block. Each layer gets an `as_strided` view with storage_offset into the packed backing tensor. ⏎  ⏎ The packed backing tensor and block_stride are passed through the cross-layer KV cache registration API so NIXL registers one region with block_len= …[truncated]

### L3-4a083cc858  (L3, 2026-06-19, sha 4a083cc858f0, PR #46180)
TITLE: [ROCm][CI] Pin `test_rocm_compressed_tensors_w8a8` to TRITON_ATTN (#46180)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/kernels.yaml (+1/-0); tests/kernels/quantization/test_triton_scaled_mm.py (+4/-2)
LABELS: rocm, ready, ci/build
BODY: `AMD: Kernels Quantization Test 2 (mi325_1)` is failing on main ever since https://github.com/vllm-project/vllm/pull/44446 was merged. There is a seg fault in the ROCM_ATTN backend, so we're pinning to TRITON_ATTN to unblock CI while I investigate the fix (see https://github.com/vllm-project/vllm/issues/46179).

### L3-dced290769  (L3, 2026-06-20, sha dced2907693e, PR #46024)
TITLE: [Hardware][AMD][CI] Fix e2e core test group (#46024)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+1/-14); .buildkite/test_areas/engine.yaml (+10/-0); tests/v1/e2e/general/test_cascade_attention.py (+7/-0)
LABELS: rocm, ready, ci/build, v1
BODY: ## Purpose ⏎ This PR fixes the e2e core test group. Cascade attention is only supported on the `FLASH_ATTN` and `FLASHINFER` backends, neither of which is currently supported on AMD. ⏎  ⏎ ## Test Plan ⏎ `pytest -v -s v1/e2e/general --ignore v1/e2e/general/test_async_scheduling.py` ⏎  ⏎ ## Test Result ⏎ The test group passes. It is run as part of AMD CI. ⏎  ⏎ cc @AndreasKaratzas  ⏎  ⏎ --- ⏎ [details omitted]

### L3-ebfbcfe46a  (L3, 2026-06-20, sha ebfbcfe46aa8, PR #45026)
TITLE: Stop setting CUDA_VISIBLE_DEVICES internally in vLLM, add device_ids arg (#45026)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.cutlass_sm100, L3.platform.cuda_selection
FILES: csrc/libtorch_stable/attention/mla/sm100_cutlass_mla_kernel.cu (+8/-3); tests/engine/test_arg_utils.py (+193/-0); tests/entrypoints/openai/test_dp_supervisor.py (+2/-2); vllm/config/parallel.py (+9/-0); vllm/distributed/device_communicators/all2all.py (+8/-1); vllm/distributed/device_communicators/all_reduce_utils.py (+26/-7); vllm/distributed/device_communicators/custom_all_reduce.py (+10/-8); vllm/distributed/device_communicators/quick_all_reduce.py (+4/-6); vllm/distributed/device_communicators/shm_broadcast.py (+7/-1); vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py (+5/-4); (+14 more)
LABELS: frontend, ready, v1, kv-connector, nvidia
ISSUES: #32569 [Feature]: Support GPU UUID in `CUDA_VISIBLE_DEVICES` | #44556 [Bug]: External DP Load Balancing plus DeepGEMM MegaMoE
BODY: This PR changes the way vLLM interacts with the `CUDA_VISIBLE_DEVICES` (CVD) environment variable: ⏎ * With this PR vLLM no longer sets CVD to control which GPU a worker uses ⏎ * This PR adds a `--device-ids` argument so that users don't need to set  ⏎  ⏎ This has a few benefits: ⏎ * vLLM currently does not support the UUID format for device IDs which is problematic for using it with MIG (should fix https://github.com/vllm-project/vllm/issues/32569) ⏎  …[truncated]

### L3-1bdf9810aa  (L3, 2026-06-20, sha 1bdf9810aae3, PR #46222)
TITLE: [ROCm] [Bugfix] Bugfix ROCm Sparse Indexer (#46222)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+6/-2)
LABELS: bug, rocm, ready, v1, DSv4
BODY: ## Purpose ⏎  ⏎ Cause: PR https://github.com/vllm-project/vllm/pull/44577 exposed a ROCm AITER sparse-indexer bug. The packed KV layout makes per-block strides large enough that `block_id * stride` can exceed 32-bit range.  ⏎  ⏎ Fix: widened the physical block id to tl.int64 before cache-stride arithmetic in both indexer cache insert and gather paths ⏎    ⏎    ⏎   Error log ⏎    ⏎ ``` ⏎   (Worker_TP1 pid=318032) ERROR 06-20 05:55:22 [multiproc_executor.py: …[truncated]

### L3-ab7fcbdd5d  (L3, 2026-06-20, sha ab7fcbdd5dbc, PR #45969)
TITLE: [Perf][KVConnector][Mooncake] Compact chunk-hash keys and zero-copy lookup wire format (#45969)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py (+5/-2); tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py (+6/-7); tests/v1/kv_connector/unit/test_mooncake_store_worker.py (+35/-3); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py (+12/-12); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py (+78/-9); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/protocol.py (+4/-1); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py (+24/-19)
LABELS: ready, v1, kv-connector
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Two related optimizations to the `MooncakeStoreConnector` prefix-lookup path. Both reduce the cost of moving block hashes around, which becomes acute when the model's `block_size` is much larger than the connector's `hash_block_size`. ⏎  ⏎ ### 1. Compact chunk-hash keys ⏎  ⏎ When `block_size > hash_block_size`, a single `block_size` chunk previously keyed Mooncake by **concatenating all of its fine-grained sub-hashes** (`BlockHashListWithBloc …[truncated]

### L3-7df3d7dada  (L3, 2026-06-20, sha 7df3d7dada84, PR #45424)
TITLE: [Core] Ensure memory is pinned prior to async h2d copy (#45424)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.flashmla_sparse, L3.mla.flashinfer_sparse, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.flex_attention
FILES: vllm/model_executor/layers/attention/mla_attention.py (+10/-8); vllm/model_executor/layers/attention/mm_encoder_attention.py (+2/-1); vllm/v1/attention/backends/flashinfer.py (+2/-4); vllm/v1/attention/backends/flex_attention.py (+6/-2); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+2/-2); vllm/v1/attention/backends/mla/flashmla_sparse.py (+2/-2); vllm/v1/attention/backends/utils.py (+16/-19); tests/v1/logits_processors/test_correctness.py (+0/-2); tests/v1/streaming_input/test_gpu_model_runner_streaming.py (+0/-1); tests/v1/worker/test_gpu_input_batch.py (+0/-6); (+39 more)
LABELS: structured-output, intel-gpu, speculative-decoding, ready, v1, multi-modality, qwen, cpu, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: To avoid possible gpu/cpu stream syncs. ⏎  ⏎ See for example https://github.com/vllm-project/vllm/pull/45074. ⏎  ⏎ Also: Generalize `async_tensor_h2d` utility function and standardize how `is_pin_memory_available()` is used.

### L3-9c450b1027  (L3, 2026-06-21, sha 9c450b102788, PR #45361)
TITLE: [Kernel][Bugfix] Fix INT8 per-token-head KV cache rounding in Triton reshape-and-cache (#45361)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/triton_reshape_and_cache_flash.py (+12/-2); tests/quantization/test_per_token_kv_cache.py (+53/-7)
LABELS: bug, rocm, ready, v1, verified
BODY: ### What this PR does ⏎  ⏎ This PR fixes the Triton reshape-and-cache path for `kv_cache_dtype=int8_per_token_head`. ⏎  ⏎ The per-token-head KV cache kernel computes one absmax scale per `(token, head)`, but the INT8 path previously clamped the scaled K/V values and then relied on the implicit float-to-int8 store. That store truncates toward zero, while INT8 absmax quantization should round to nearest before storing. ⏎  ⏎ This change makes the INT8 path expl …[truncated]

### L3-2cac89f9da  (L3, 2026-06-21, sha 2cac89f9da86, PR #45181)
TITLE: [Spec Decode] Support mixed KV page sizes for DFlash (#45181)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+32/-0); tests/v1/core/test_kv_cache_utils.py (+102/-7); tests/v1/worker/test_attn_utils.py (+242/-0); vllm/v1/core/kv_cache_utils.py (+24/-9); vllm/v1/kv_cache_interface.py (+13/-3); vllm/v1/worker/gpu/attn_utils.py (+77/-41); vllm/v1/worker/gpu_model_runner.py (+17/-52); vllm/v1/worker/kv_connector_model_runner_mixin.py (+4/-29)
LABELS: speculative-decoding, ready, v1, qwen, nvidia
BODY: ## Summary ⏎  ⏎ This PR addresses the KV-cache infrastructure gap needed by DFlash-style speculative decoding when the target and draft models have different KV page sizes. ⏎  ⏎ It adds: ⏎ - A padding fallback in `unify_kv_cache_spec_page_size` for non-divisible page sizes. ⏎ - Padded KV-cache reshape handling for FlashAttention-style layouts where the block dimension is not the first physical dimension. ⏎ - A shared attention KV-cache reshape helper so padded …[truncated]

### L3-db32b53e30  (L3, 2026-06-22, sha db32b53e302e, PR #43081)
TITLE: [SpecDecode] Support DFlash with FlashInfer  (#43081)
SOURCES: path_core, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+60/-13); docs/design/attention_backends.md (+2/-2); tests/kernels/attention/test_attention_selector.py (+1/-1)
LABELS: documentation, ready, v1, nvidia, verified, dflash
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ This PR is split out from #39995. ⏎ It adds  FlashInfer backend support needed for DFlash. ⏎  ⏎ **Note:** On RTX 4090, using FP8 KV cache requires the combination of DFlash + FlashInfer. ⏎ ## Changes: ⏎ - Allow FlashInfer to be selected for non-causal DFlash attention. ⏎ - Route non-causal DFlash batches through FlashInfer native prefill with `causal=False`. ⏎ - Keep non-causal FlashInfer away from unsupported DCP / TRTLLM / NVFP4 paths. ⏎  ⏎  …[truncated]

### L3-f3df7a7231  (L3, 2026-06-22, sha f3df7a7231f3, PR #45955)
TITLE: [ROCm][CI] Enable kv_connector unit tests on ROCm (#45955)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/scripts/install-kv-connectors.sh (+5/-0); .buildkite/test_areas/misc.yaml (+6/-0); tests/v1/kv_connector/unit/test_multi_connector.py (+1/-0); tests/v1/kv_connector/unit/test_offloading_connector.py (+4/-0)
LABELS: rocm, ready, ci/build, v1, kv-connector
BODY: We are seeing failures in `test_offloading_connector` after https://github.com/vllm-project/vllm/pull/43458 enabled MRV2 for Llama models by default.  ⏎  ⏎ here is an example from https://buildkite.com/vllm/amd-ci/builds/9582/list?sid=019ecfa8-eaee-4930-90f1-9c79dac59f06&tab=output ⏎ ``` ⏎ =========================== short test summary info ============================ ⏎ FAILED v1/kv_connector/unit/test_offloading_connector.py::test_cpu_offloading[met …[truncated]

### L3-aa4990a9a2  (L3, 2026-06-22, sha aa4990a9a202, PR #45111)
TITLE: [Attention] Re-enable cross-layer KV cache layout for MLA via stride-aware kernels (#45111)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.triton.decode_attention, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_sm100, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer
FILES: csrc/libtorch_stable/attention/mla/sm100_cutlass_mla_kernel.cu (+5/-1); vllm/model_executor/layers/attention/mla_attention.py (+3/-3); vllm/v1/attention/backends/mla/cutlass_mla.py (+8/-0); vllm/v1/attention/backends/mla/flashattn_mla.py (+8/-0); vllm/v1/attention/backends/mla/flashinfer_mla.py (+8/-0); vllm/v1/attention/backends/mla/flashmla.py (+8/-0); vllm/v1/attention/backends/mla/triton_mla.py (+8/-0); vllm/v1/attention/ops/triton_decode_attention.py (+34/-7); csrc/libtorch_stable/cache_kernels.cu (+6/-7); tests/kernels/attention/test_cutlass_mla_decode.py (+66/-0); (+3 more)
LABELS: ready, v1, kv-connector, nvidia
BODY: ## Purpose ⏎  ⏎ #37090 disabled the cross-layer (block-major) KV cache layout for all MLA backends after #37032 (GLM-4.7-Flash garbage output with KV offloading), attributing the bug to "MLA kernels requiring contiguous per-layer KV cache views". The actual cause is narrower: a few kernels computed page addresses from `block_size * entry_size` instead of reading the cache tensor's block-dim stride. The MLA write path (`concat_and_cache_mla`), the pre …[truncated]

### L3-44d95069e9  (L3, 2026-06-22, sha 44d95069e9d6, PR #43477)
TITLE: Enable DeepSeek V4 and GLM-5.1 on SM120 (#43477)
SOURCES: path_core, symbol_pickaxe, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashinfer_sparse, L3.dispatch.registry, L3.dispatch.abstract_interface, L3.platform.cuda_selection
FILES: cmake/external_projects/deepgemm.cmake (+58/-28); cmake/external_projects/qutlass.cmake (+38/-16); vllm/model_executor/layers/attention/mla_attention.py (+29/-14); vllm/utils/flashinfer.py (+33/-9); vllm/v1/attention/backend.py (+1/-1); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+128/-37); vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py (+155/-0); vllm/v1/attention/backends/mla/sparse_swa.py (+74/-15); vllm/v1/attention/backends/registry.py (+5/-1); vllm/v1/attention/ops/flashmla.py (+1/-1); (+27 more)
LABELS: documentation, speculative-decoding, ready, ci/build, v1, deepseek, nvidia, verified
BODY: ## Summary ⏎  ⏎ This draft PR brings up the DeepSeek V4 and GLM-5.1 SM120 path on consumer Blackwell: ⏎  ⏎ - routes DSv4 sparse MLA through the FlashInfer SM120 sparse MLA wrapper/backend ⏎ - adds SM120 sparse MLA decode/prefill handling for sparse MLA models ⏎ - wires DSv4-specific kernel warmups, including FlashInfer decode autotune during warmup ⏎ - enables DeepGEMM MXFP4 MoE paths and related grouped-GEMM heuristics for SM120 ⏎ - updates CMake plumbi …[truncated]

### L3-9037498c22  (L3, 2026-06-22, sha 9037498c2289, PR #44517)
TITLE: [DSV4][XPU] Pass gemm1_clamp_limit to XpuFusedMoe (#44517)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/xpu_moe.py (+2/-0)
LABELS: intel-gpu, ready, v1
BODY: ## Summary ⏎  ⏎ Pass `quant_config.gemm1_clamp_limit` to `XpuFusedMoe` so that the SwiGLU clamp limit is applied during MoE expert computation on XPU. ⏎  ⏎ ## Dependencies ⏎  ⏎ This PR depends on: ⏎ - https://github.com/vllm-project/vllm/pull/42953 ⏎ - https://github.com/vllm-project/vllm-xpu-kernels/pull/395

### L3-183b5f27ea  (L3, 2026-06-22, sha 183b5f27eafa, PR #44053)
TITLE: [Bugfix][V1][TurboQuant] Reserve workspace before CUDA graph capture (#44053)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/turboquant_attn.py (+39/-0); tests/quantization/test_turboquant.py (+124/-0)
LABELS: bug, ready, v1, nvidia, quantization
BODY: ## Summary ⏎  ⏎ Supersedes #40798. This PR fixes a TurboQuant CUDA graph workspace bug: TurboQuant can lazily request larger decode / continuation-prefill scratch buffers after CUDA graph capture has already locked the v1 workspace, which triggers runtime locked-workspace assertions on long-context requests. ⏎  ⏎ The fix reserves the maximum TurboQuant workspace from the TurboQuant attention backend initialization path, before CUDA graph capture locks th …[truncated]

### L3-d1a38c2762  (L3, 2026-06-22, sha d1a38c276202, PR #42235)
TITLE: [Kernel][Performance] Add FlashInfer cutedsl NVFP4 GEMM backend (#42235)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py (+3/-1); tests/models/quantization/test_nvfp4.py (+7/-0); vllm/compilation/passes/fusion/collective_fusion.py (+9/-0); vllm/config/kernel.py (+2/-0); vllm/model_executor/kernels/linear/__init__.py (+6/-0); vllm/model_executor/kernels/linear/nvfp4/flashinfer.py (+66/-0)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Adds `flashinfer-cutedsl` for dense NVFP4 GEMM and makes it the highest-priority CUDA backend when supported on SM10x. In serving benchmarks, cutedsl is fastest across concurrency 1-512 and improves tok/s/user by up to 27.07% over the tested FlashInfer backends. ⏎  ⏎ ## Performance Comparison ⏎  ⏎ Setup: ⏎  ⏎ - Model: `nvidia/Llama-3.1-8B-Instruct-NVFP4` ⏎ - Device: SM103 ⏎ - Dataset: random ⏎ - Input/output length: 512 input tokens, 512 out …[truncated]

### L3-fbf9ff7cf4  (L3, 2026-06-22, sha fbf9ff7cf4a4, PR #46401)
TITLE: [CI][ROCm] Restrict MLA cross-layer KV cache test to supported backends on ROCm (#46401)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_kv_cache_layout.py (+7/-0)
LABELS: rocm, ready, v1, kv-connector
BODY: This PR Fixes `test_verified_mla_backends_support_cross_layer_kv_cache[FlashAttnMLABackend]` on ROCm (MI300/MI325). ⏎  ⏎ ## Test Plan ⏎ On MI300/MI325: ⏎ ```bash ⏎ pytest -v tests/v1/kv_connector/unit/test_kv_cache_layout.py ⏎  ⏎  ⏎ Test Result ⏎ Before (ROCm): ⏎  ⏎ FAILED ...test_verified_mla_backends_support_cross_layer_kv_cache[...FlashAttnMLABackend] ⏎ 1 failed, 3 passed, 1 skipped ⏎ After (ROCm): ⏎  ⏎ 3 passed

### L3-3c8e49596c  (L3, 2026-06-22, sha 3c8e49596c3f, PR #46108)
TITLE: [Model] ColQwen3.5: fix retrieval correctness (bias + bidirectional) (#46108)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/attention.py (+8/-1); docs/models/pooling_models/token_embed.md (+1/-1); examples/pooling/score/colqwen3_5_rerank_online.py (+17/-1); tests/models/multimodal/pooling/test_colqwen3_5.py (+18/-0); vllm/model_executor/models/colqwen3_5.py (+9/-1); vllm/model_executor/models/config.py (+15/-1); vllm/model_executor/models/qwen3_next.py (+11/-0)
LABELS: documentation, ready, multi-modality, qwen, build-docs
BODY: ## Summary ⏎  ⏎ Follow-up to #36887. The in-tree `ColQwen3_5` model deviates from the native ⏎ colpali `ColQwen3_5Processor` inference pipeline in three silent ways, none caught ⏎ by the current sanity-level tests. Measured cost: **~2.5 ndcg@10** on Vidore3. ⏎ This affects **every** ColQwen3.5 checkpoint — e.g. `athrael-soju/colqwen3.5-4.5B-v3` ⏎ and `athrael-soju/VultronRetrieverPrime-Qwen3.5-8B` — both of which MTEB loads ⏎ through the one `ColQwen3_5Wrapper …[truncated]

### L3-91ba720b75  (L3, 2026-06-22, sha 91ba720b75f0, PR #46148)
TITLE: [ROCm][CI] Only require q_scale==1.0 for fp8 query in RocmAttention (#46148)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+12/-3)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎  ⏎ On ROCm, `RocmAttentionImpl.forward()` asserted `layer._q_scale_float == 1.0` ⏎ whenever the KV cache is quantized (fp8): ⏎  ⏎ ```python ⏎ if is_quantized_kv_cache(self.kv_cache_dtype): ⏎     key_cache = key_cache.view(self.fp8_dtype) ⏎     value_cache = value_cache.view(self.fp8_dtype) ⏎     assert layer._q_scale_float == 1.0, ( ⏎         "A non 1.0 q_scale is not currently supported." ⏎     ) ⏎ ``` ⏎  ⏎ The ROCm MHA path (`chunked_prefill_pag …[truncated]

### L3-56e5797511  (L3, 2026-06-22, sha 56e57975112b, PR #45375)
TITLE: [Quant] Enable modelopt_mixed on Turing (SM75) (#45375)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+6/-1); docs/design/attention_backends.md (+1/-1); vllm/model_executor/layers/quantization/modelopt.py (+8/-7)
LABELS: documentation, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ Extend `modelopt_mixed` (NVFP4 routed experts + FP8 weight-only dense) inference to Turing (SM75) by lowering `ModelOptMixedPrecisionConfig.get_min_capability()` from 80 to 75. Marlin already supports SM75, so the kernels are in place — this gate was the only thing blocking it. The payoff is NVFP4 inference on widely available platforms, including a free Google Colab T4. ⏎  ⏎ The per-layer paths already run on SM75: NVFP4 routed exper …[truncated]

### L3-6691f087a6  (L3, 2026-06-23, sha 6691f087a65b, PR #45892)
TITLE: [Minimax-M3] BF16/FP8 Indexer using MSA (#45892)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen
FILES: cmake/external_projects/fmha_sm100.cmake (+26/-1); setup.py (+8/-0); csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu (+152/-51); tests/kernels/attention/test_minimax_m3.py (+325/-1); tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py (+96/-0); vllm/envs.py (+3/-4); vllm/models/minimax_m3/amd/model.py (+28/-3); vllm/models/minimax_m3/common/indexer.py (+60/-8); vllm/models/minimax_m3/common/ops/index_topk.py (+30/-12); vllm/models/minimax_m3/common/sparse_attention.py (+22/-15); (+3 more)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ Integrating MSA indexer prefill kernel. Decode remain triton for performance reason ⏎  ⏎ ## Test Plan ⏎ Minimax M3 gsm8k, AIME25, GPQA in TP4 ⏎  ⏎ ## Test Result ⏎ with FP8 attention and FP8 Indexer cache.  ⏎ gsm8k: 92 ⏎ GPQA-D: 93.6 ⏎ AIME25: 92.5 ⏎  ⏎ --- ⏎ [details omitted]

### L3-7c2e08451a  (L3, 2026-06-23, sha 7c2e08451a47, PR #46517)
TITLE: [Docker] Remove redundant flashinfer download-cubin step (#46517)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+0/-7)
LABELS: ready, ci/build, nvidia
BODY: ## Summary ⏎  ⏎ The `flashinfer-cubin` wheel — a default CUDA dep in `requirements/cuda.txt` since #37233 — already ships all precompiled cubins into the package directory that FlashInfer's runtime loader reads from (`jit/env.py:_get_cubin_dir` resolves to the package dir when `flashinfer-cubin` is installed). The `flashinfer download-cubin` step in `vllm-base` re-downloads the same files from NVIDIA's artifactory **unconditionally** (`download_artif …[truncated]

### L3-11b56b2ff2  (L3, 2026-06-23, sha 11b56b2ff28e, PR #46393)
TITLE: [Kernel] Add FlashInferCutedslMxfp8LinearKernel (cute-dsl mm_mxfp8) (#46393)
SOURCES: path_integration+keyword, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/kernel.py (+1/-1); vllm/model_executor/kernels/linear/__init__.py (+4/-0); vllm/model_executor/kernels/linear/mxfp8/flashinfer.py (+82/-0)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Add an MXFP8 W8A8 linear GEMM that drives FlashInfer's mm_mxfp8(..., backend="cute-dsl"), sibling to the existing CUTLASS kernel. The cute-dsl backend consumes the same 1D swizzled F8_128x4 scales the CUTLASS path already produces, so weight/activation prep is identical and output is bit-identical; only the backend string and support gate differ. ⏎  ⏎ Gate to sm_100/sm_103 (matching FlashInfer's supported_compute_capability [100, 103]) plus has_fla …[truncated]

### L3-9f5117820f  (L3, 2026-06-23, sha 9f5117820fb0, PR #46135)
TITLE: [HARDWARE][POWER] Enable fp16 support for PowerPC (#46135)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: csrc/cpu/mla_decode.cpp (+0/-8); csrc/cpu/cpu_attn_impl.hpp (+0/-2); csrc/cpu/cpu_attn_vsx.hpp (+10/-1); csrc/cpu/cpu_types_vsx.hpp (+168/-56); csrc/cpu/pos_encoding.cpp (+154/-1); vllm/platforms/cpu.py (+1/-1)
LABELS: ready, cpu, verified
BODY: ## Purpose ⏎  ⏎ Enable FP16 (half precision) inference support for PowerPC systems.

### L3-3cc871aaf1  (L3, 2026-06-23, sha 3cc871aaf155, PR #46422)
TITLE: [Perf] Skip detokenization in online beam search (#46422)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/entrypoints/generate/beam_search/online.py (+1/-0)
LABELS: frontend, ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## What & why ⏎  ⏎ Online beam search builds an internal per-step `SamplingParams` requesting ⏎ `logprobs = 2 * beam_width`. The engine then **detokenizes every one of those ⏎ `2 * beam_width` logprob token ids into strings on every decode step** ⏎ (`LogprobsProcessor` → `tokenizer.decode`). Beam search never uses those ⏎ strings — it ranks candidates by `cum_logprob` and detokenizes only the final ⏎ selected sequences itself (`tokenizer.decode(tokens)` …[truncated]

### L3-855cd4d787  (L3, 2026-06-23, sha 855cd4d78760, PR #43008)
TITLE: [Perf][DSv4/DSv3.2] Add cluster-cooperative topK kernel for low-latency scenarios (#43008)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: .buildkite/test_areas/kernels.yaml (+2/-0); CMakeLists.txt (+30/-0); csrc/libtorch_stable/cooperative_topk.cu (+146/-0); csrc/libtorch_stable/cooperative_topk.cuh (+593/-0); csrc/libtorch_stable/ops.h (+8/-0); csrc/libtorch_stable/persistent_topk.cuh (+49/-18); csrc/libtorch_stable/topk_histogram_4096.cuh (+554/-0); csrc/libtorch_stable/torch_bindings.cpp (+9/-0); tests/kernels/test_top_k_per_row.py (+151/-26); vllm/model_executor/layers/sparse_attn_indexer.py (+26/-1)
LABELS: ready, ci/build, deepseek, DSv4
ISSUES: #40902 [Roadmap] DeepSeek V4
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎ Adds a cluster-cooperative topK kernel for low-latency cases, which uses TMA and DSMEM, meaning that SM90+ arch is required. This new kernel has been extensively tuned to cover the whole low-latency regime (i.e. bs ≤32) with a cluster-level cooperation via cluster.sync() and distributed SMEM histogram reduction, eliminating the complexity of the persistent scheduler and multi-CTA spin-barrier coordination in persistent_topK v1 (PR #37 …[truncated]

### L3-20b5af55c1  (L3, 2026-06-23, sha 20b5af55c1c9, PR #43673)
TITLE: [ROCm][Perf] DSv3.2: fuse MLA Q concat+fp8-quant in forward_mqa (#43673)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+12/-7)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: On the ROCm sparse-MLA fp8-KV decode path, the query tensor was being built by two back-to-back HBM-bound kernels: `ConcatMLAQKernel<bf16, 512>` (concatenate `q_nope` + `q_pe` into bf16, ~5.2 µs) followed by `scaled_fp8_quant` (~5 µs). Both are pure copy-with-scale, no compute. ⏎  ⏎ This PR routes `forward_mqa` through `layer._decode_concat_quant_fp8_op(ql_nope, q_pe, q_scale)` (already used by `MLACommonImpl`), which `torch.compile` lowers to a si …[truncated]

### L3-556bc4e3a0  (L3, 2026-06-23, sha 556bc4e3a089, PR #46568)
TITLE: Upgrade tpu-inference to v0.23.0 (#46568)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/tpu.txt (+1/-1)
LABELS: ci/build
BODY: ## Purpose ⏎  ⏎ Upgrade tpu-inference to latest stable release v0.23.0 ⏎  ⏎ ## Test Plan ⏎  ⏎ Verified on tpu-inference CI. ⏎  ⏎ ## Test Result ⏎  ⏎ Success. ⏎  ⏎ --- ⏎ [details omitted]

### L3-9d6fdc2901  (L3, 2026-06-23, sha 9d6fdc2901df, PR #46385)
TITLE: [Kernel] GLM5 Router GEMM (#46385)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/moe/dsv3_router_gemm_bf16_out.cu (+49/-0); csrc/libtorch_stable/moe/dsv3_router_gemm_entry.cu (+51/-29); csrc/libtorch_stable/moe/dsv3_router_gemm_float_out.cu (+49/-0); vllm/model_executor/layers/fused_moe/router/gate_linear.py (+9/-3)
LABELS: ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Borrrow idea from https://github.com/NVIDIA/TensorRT-LLM/pull/13740 ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ### GSM8K ⏎  ⏎ - main branch ⏎  ⏎ ``` ⏎ local-completions ({'model': 'zai-org/GLM-5.2-FP8', 'base_url': 'http://0.0.0.0:8000/v1/completions', 'tokenized_requests': False, 'tokenizer_backend': None, 'num_concurrent': 256}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 1 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   …[truncated]

### L3-80e511772f  (L3, 2026-06-23, sha 80e511772f3e, PR #44434)
TITLE: [ROCm][Bugfix][Perf] enable shared expert fusion for Qwen3.5 (#44434)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_5.py (+19/-0); vllm/model_executor/models/qwen3_next.py (+29/-4)
LABELS: bug, rocm, ready, qwen
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎  ⏎ The existing FSE (Fused Shared Expert) support (#39280) works for Qwen3-Next but fails on Qwen3.5 models because `qwen3_5.py`'s `load_weights` does not remap shared expert checkpoint weights to the fused expert slot. This causes shared expert weights to silently fail to load, producing garbage output when `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`. ⏎  ⏎ ## Root Cause ⏎  ⏎ When FSE is enabled, `Qwen3NextSparseMoeBlock.__init__` (which …[truncated]

### L3-d86c66c981  (L3, 2026-06-23, sha d86c66c98100, PR #46167)
TITLE: [Feat] Add runtime monitor for post-warmup CuTeDSL compilation (#46167)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/engine/test_arg_utils.py (+11/-0); tests/test_jit_monitor.py (+113/-18); vllm/config/observability.py (+4/-1); vllm/engine/arg_utils.py (+22/-13); vllm/triton_utils/jit_monitor.py (+0/-135); vllm/utils/jit_monitor.py (+212/-0); vllm/v1/worker/gpu_worker.py (+4/-5)
LABELS: ready, v1
BODY: ### Warn mode ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek-V2-Lite \ ⏎ ... ⏎   --jit-monitor-mode warn \ ⏎   --jit-monitor-verbose \ ⏎   --attention-config '{"flash_attn_version":4,"mla_prefill_backend":"FLASH_ATTN"}' ⏎ ``` ⏎  ⏎ `(EngineCore pid=2386340) ... [jit_monitor.py:71] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.` ⏎  ⏎ `(EngineCore pid=2275828) WARNING 06-19 16:22:25 [jit_monitor.py:124] CuTeDSL JIT compi …[truncated]

### L3-dc0d318177  (L3, 2026-06-24, sha dc0d318177e1, PR #46189)
TITLE: [Attention] Add FLASH_ATTN_MLA_SPARSE backend for Hopper sparse MLA (#46189)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.flashattn_sparse, L3.dispatch.registry, L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+1/-0); vllm/v1/attention/backends/mla/flashattn_mla_sparse.py (+287/-0); vllm/v1/attention/backends/registry.py (+3/-0); docs/design/attention_backends.md (+1/-0)
LABELS: documentation, ready, v1, nvidia, verified
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ This PR adds a Python-only `FLASH_ATTN_MLA_SPARSE` backend for sparse MLA ⏎ models on Hopper GPUs. The backend reuses vLLM's existing FlashAttention MLA ⏎ wrapper and sparse MLA index conversion path, then calls ⏎ `flash_attn_varlen_func(..., fa_version=3)` for BF16 KV-cache sparse decode. ⏎  ⏎ The tested GLM-5.1-FP8 checkpoint has: ⏎  ⏎ - `num_attention_heads = 64` ⏎ - `kv_lora_rank = 512` ⏎ - `qk_rope_head_dim = 64` ⏎ - compressed KV dim =  …[truncated]

### L3-563c628968  (L3, 2026-06-24, sha 563c628968c0, PR #46607)
TITLE: [XPU] bump up vllm_xpu_kernels to v0.1.10.1 (#46607)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/xpu.txt (+1/-1)
LABELS: intel-gpu, ready, ci/build
BODY: ## Purpose ⏎ bump up vllm_xpu_kernels to v0.1.10.1 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-84c62e1cbd  (L3, 2026-06-24, sha 84c62e1cbdef, PR #46535)
TITLE: [Model Runner V2][MM] Support EVS (#46535)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/diffusion_gemma.py (+9/-9); vllm/model_executor/models/interfaces.py (+6/-5); vllm/model_executor/models/qwen2_5_vl.py (+10/-6); vllm/model_executor/models/qwen3_vl.py (+14/-10); vllm/v1/worker/gpu/mm/rope.py (+17/-0); vllm/v1/worker/gpu/model_runner.py (+4/-10); vllm/v1/worker/gpu/model_states/default.py (+25/-8); vllm/v1/worker/gpu/model_states/encoder_decoder.py (+4/-1); vllm/v1/worker/gpu/model_states/interface.py (+21/-1); vllm/v1/worker/gpu/model_states/mm_pruning.py (+135/-0)
LABELS: ready, v1, qwen
BODY: This also changes the Qwen VL `recompute_mrope_positions` methods to accept input_ids as a device tensor rather than python list, to avoid cpu roundtrip.

### L3-70749fdcca  (L3, 2026-06-24, sha 70749fdcca7d, PR #40835)
TITLE: [Feature] Triton INT4 per-token-head KV cache quantization (#40835)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.triton.unified_attention, L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+33/-16); vllm/v1/attention/ops/int4_per_token_head.py (+1155/-0); vllm/v1/attention/ops/triton_reshape_and_cache_flash.py (+21/-2); vllm/v1/attention/ops/triton_unified_attention.py (+43/-1); docs/design/attention_backends.md (+1/-1); tests/models/quantization/test_per_token_kv_cache.py (+2/-1); tests/quantization/test_per_token_kv_cache.py (+185/-70); vllm/config/cache.py (+1/-0); vllm/utils/torch_utils.py (+1/-0); vllm/v1/kv_cache_interface.py (+21/-21)
LABELS: documentation, ready, v1
BODY: Based on PR: ⏎  ⏎ https://github.com/vllm-project/vllm/pull/40633 ⏎  ⏎  ⏎ Changed to only INT4_PER_TOKEN_HEAD ⏎  ⏎ INT4_PER_TOKEN_HEAD ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.912|±  |0.0273| ⏎ |     |       |strict-match    |     5|exact_match|↑  | 0.91|±  |0.0273|

### L3-b3a688cb9e  (L3, 2026-06-24, sha b3a688cb9eb7, PR #46548)
TITLE: [ROCm] Fix OOB During Model Warmup With `ROCM_ATTN` and MRV2 (#46548)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+1/-1); tests/kernels/attention/test_attention.py (+5/-3); tests/kernels/quantization/test_triton_scaled_mm.py (+2/-4)
LABELS: rocm, ready
ISSUES: #46179 [Bug]: `vllm serve neuralmagic/Llama-3.2-1B-quantized.w8a8` Failing On ROCm w/ MRV2
BODY: Resolves https://github.com/vllm-project/vllm/issues/46179 (as well as numerous other failing tests in AMD CI, e.g. `pytest -v -s tests/compile/fullgraph/test_full_graph.py::test_full_graph[neuralmagic/Llama-3.2-1B-Instruct-FP8-dynamic-model_kwargs1-2]`). ⏎  ⏎ This PR resolves a latent seg fault in the `ROCM_ATTN` that was exposed when Llama was added to the default architectures for MRV2 in https://github.com/vllm-project/vllm/pull/43458. ⏎  ⏎ The s …[truncated]

### L3-3c43237233  (L3, 2026-06-24, sha 3c43237233a8, PR #46560)
TITLE: [Bugfix][Model Runner V2][Spec Decode] Fix int32 offset overflow in sampler kernels (#46560)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/sample/test_logprobs.py (+34/-0); tests/v1/sample/test_topk_topp_sampler.py (+40/-0); vllm/v1/sample/ops/topk_topp_triton.py (+1/-1); vllm/v1/worker/gpu/sample/bad_words.py (+1/-1); vllm/v1/worker/gpu/sample/gumbel.py (+3/-3); vllm/v1/worker/gpu/sample/logit_bias.py (+1/-1); vllm/v1/worker/gpu/sample/logprob.py (+2/-2); vllm/v1/worker/gpu/sample/min_p.py (+1/-1); vllm/v1/worker/gpu/sample/penalties.py (+1/-1); vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py (+6/-6)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ DFlash (and any speculative method that drafts many tokens) crashes the V1 ⏎ engine at startup on the Model Runner V2 path: ⏎  ⏎ ``` ⏎ RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered ⏎   ... vllm/v1/worker/gpu/warmup.py  worker_sample_tokens(None) ⏎ ``` ⏎  ⏎ Minimal repro (current main, single H200): ⏎  ⏎ ```python ⏎ from vllm import LLM, SamplingParams ⏎ llm = LLM( ⏎     model="Qwen/Qwen3-8B", ⏎     speculative_config={"method": "dfl …[truncated]

### L3-e7df232288  (L3, 2026-06-24, sha e7df23228895, PR #46252)
TITLE: [KV Offload] Gate packed HMA KV cache on cross-layer config (#46252)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/v1/core/test_contiguous_kv_packing.py (+13/-10); vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py (+8/-7); vllm/envs.py (+0/-6); vllm/v1/core/kv_cache_utils.py (+21/-10)
LABELS: ready, v1, kv-connector
BODY: ## Summary ⏎  ⏎ - Use `kv_connector_extra_config["enable_cross_layers_blocks"]` to opt multi-group HMA layouts into packed KV allocation as laid out in https://docs.vllm.ai/en/stable/features/nixl_connector_usage/#cross-layers-blocks ⏎ - Keep DeepSeek V4-style `UniformTypeKVCacheSpecs` layouts on the packed path by default. ⏎ - Remove the extra `VLLM_USE_PACKED_HMA_KV_CACHE` environment flag. ⏎ - Canonicalize packed KV caches in the offloading worker  …[truncated]

### L3-d7ab9be775  (L3, 2026-06-24, sha d7ab9be77552, PR #46408)
TITLE: [Bugfix] Support -1 (invalid/non-local) slots in topk_ids for Triton MoE (#46408)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py (+88/-7)
LABELS: bug, ready, gpt-oss
DEEP_STUDY: deep-study correctness case vllm:d7ab9be775: class=shape_alignment_edge; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Purpose ⏎  ⏎ Under expert parallelism, dispatch hands the experts a `topk_ids` tensor whose entries can be `-1`, marking invalid / non-local slots (a token's routed expert lives on another rank). The Triton unfused MoE path (`gpt_oss_triton_kernels_moe.py`) mishandled these in two places: ⏎  ⏎ 1. **`expert_map[topk_ids]`** (global→local remap) indexed the map with `-1`, which wraps to `expert_map[-1]` — a valid local id — and misroutes the slot. Repla …[truncated]

### L3-ede54b926e  (L3, 2026-06-24, sha ede54b926ebe, PR #46555)
TITLE: set AttentionCGSupport.UNIFORM_BATCH for fa2 on xpu (#46555)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+1/-1)
LABELS: intel-gpu, ready, v1
BODY: ## Purpose ⏎  ⏎ FA2 on XPU will decide grid size by max_query_len, it won't support ALWAYS with mix-prefill-decode batch. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-f889325c51  (L3, 2026-06-24, sha f889325c511b, PR #45850)
TITLE: [KV Offload] Use background thread for mmap / cpu_tensors pinning (#45850)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/kv_offload/cpu/gpu_worker.py (+89/-38)
LABELS: ready, v1
DEEP_STUDY: deep-study: this PR was reverted by PR 46958 (confirmed_revert, reason=unstated)
BODY: ## Purpose ⏎ This pull request introduces improvements to how CPU tensors / mmap are pinned for CUDA transfers in `vllm/v1/kv_offload/cpu/gpu_worker.py`, including asynchronous pinning for better startup performance. The changes add a dedicated function for pinning CPU tensors, enable background pinning on CUDA platforms, and ensure logging and synchronization of the pinning process at the end of `CpuGpuOffloadingHandlers.__init__`. ⏎  ⏎ Partial #33 …[truncated]

### L3-9e88e969c0  (L3, 2026-06-24, sha 9e88e969c08e, PR #45971)
TITLE: [Perf][KVConnector][Mooncake] Parallelize KV load with a receive-thread pool (#45971)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/v1/kv_connector/unit/test_mooncake_store_worker.py (+5/-2); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py (+40/-25); vllm/envs.py (+9/-0)
LABELS: ready, v1, kv-connector
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ In `MooncakeStoreConnector`, each KV-load request pays control overhead before any RDMA transfer begins: Python-side request preparation (token/block walk, slice list construction, tp-rank rotation, disk-offload budgeting) plus the master key lookup that runs inside `batch_get_into_multi_buffers`. With a single receive thread, that overhead is serial dead time on the NIC between consecutive transfers — the link sits idle while th …[truncated]

### L3-fc7fc421e9  (L3, 2026-06-24, sha fc7fc421e988, PR #46518)
TITLE: [Kernel][MoE] Allow FlashInfer MXINT4 MoE for gated SiLU (#46518)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_marlin_vs_trtllm_mxint4.py (+12/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_mxint4_moe.py (+4/-2)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ `TrtLlmMxint4ExpertsMonolithic` currently rejects `MoEActivation.SILU`, so vLLM does not select the FlashInfer TRT-LLM MXINT4 MoE backend for Kimi-K2.5-style gated-SiLU MoE layers. ⏎  ⏎ The underlying FlashInfer `trtllm_mxint4_block_scale_moe` path supports the standard gated SiLU computation. This PR updates the Python-side activation support predicate to accept `MoEActivation.SILU` in addition to the existing `MoEActivation.SWIGLUOA …[truncated]

### L3-49f2104c53  (L3, 2026-06-24, sha 49f2104c53c9, PR #44044)
TITLE: [Feature] Support DCP with FP8 KV cache in MLA decode path (#44044)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_v1_adapter
FILES: vllm/model_executor/layers/attention/mla_attention.py (+68/-14); vllm/v1/attention/backends/mla/flashmla.py (+4/-1); tests/kernels/attention/test_cache.py (+72/-0); tests/v1/attention/test_mla_backends.py (+215/-0)
LABELS: ready, v1
ISSUES: #32010 [Feature]: Support DCP with FP8 KV Cache
BODY: ## Purpose ⏎  ⏎ Fix DCP + FP8 KV cache support in MLA. ⏎ Previously, MLA decode rejected this config with: `DCP not support fp8 kvcache now.` This PR allows MLA decode to all-gather the already-quantized FP8 query tensor across DCP ranks when FP8 query input is supported. Non-FP8 behavior is unchanged. ⏎  ⏎ This also updates FlashMLA FP8 decode metadata to account for gathered DCP query heads, and fixes the DCP + standard FP8 MLA chunked prefill conte …[truncated]

### L3-84c2f9f0fb  (L3, 2026-06-24, sha 84c2f9f0fb3f, PR #46344)
TITLE: [Frontend] Fix Kimi K2 tool call IDs for required tool choice (#46344)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/entrypoints/openai/chat_completion/test_completion_with_function_calling.py (+1/-74); tests/entrypoints/openai/responses/test_parsable_context_unit.py (+9/-1); tests/parser/test_parse.py (+149/-2); tests/parser/test_streaming.py (+106/-2); vllm/entrypoints/openai/chat_completion/serving.py (+17/-68); vllm/entrypoints/openai/responses/context.py (+0/-14); vllm/entrypoints/openai/responses/serving.py (+1/-6); vllm/entrypoints/openai/responses/utils.py (+1/-7); vllm/parser/abstract_parser.py (+46/-2); vllm/parser/engine/parser_engine.py (+11/-2); (+1 more)
LABELS: frontend, ready, tool-calling
BODY: ## Purpose ⏎ [Frontend] Fix Kimi K2 tool call IDs for required tool choice ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve /mnt/data3/models/moonshotai/Kimi-K2.5   --tensor-parallel-size 8 --mm-encoder-tp-mode data   --trust-remote-code  --tool-call-parser kimi_k2  --enable-auto-tool-choice   --reasoning-parser kimi_k2  ⏎  ⏎ ``` ⏎  ⏎ ``` ⏎ #!/usr/bin/env python3 ⏎ """Reproduction script for vLLM issue #31501""" ⏎  ⏎ from openai import OpenAI ⏎  ⏎ k2_client = OpenAI(base_url …[truncated]

### L3-dc55936f64  (L3, 2026-06-24, sha dc55936f6477, PR #46650)
TITLE: [AMD][CI] Fix Pipeline + Context Parallelism test group (#46650)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/distributed/test_pp_cudagraph.py (+7/-6)
LABELS: rocm, ready, nvidia
BODY: ## Purpose ⏎ The `Pipeline + Context Parallelism (4 GPUs)` group fails on ROCm at `test_pp_cudagraph[FLASH_ATTN-2-JackFram/llama-160m]`. ⏎ The test hardcodes `--attention-backend=FLASH_ATTN`, which resolves to the unified CUDA `FlashAttentionBackend` on every platform. That backend is not runnable on ROCm: ⏎ - `get_flash_attn_version()` intentionally returns `None` on ROCm ("ROCm doesn't use vllm_flash_attn"), but `FlashAttentionImpl.forward()` asse …[truncated]

### L3-fc61c6fc26  (L3, 2026-06-24, sha fc61c6fc26a2, PR #46392)
TITLE: [Perf] Enable + tune FlashInfer fused allreduce at world_size=16 on SM 10.3 (GB300) (#46392)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_fused_collective.py (+10/-3); vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+2/-0); vllm/config/compilation.py (+1/-1)
LABELS: performance, ready, verified
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎  ⏎ FlashInfer fused allreduce+RMSNorm is gated by world size in two places, and ⏎ neither currently admits `world_size=16`: ⏎  ⏎ - `PassConfig.flashinfer_max_size()` returns `None` unless the world size is in ⏎   `FI_SUPPORTED_WORLD_SIZES` (was `[2, 4, 8]`), and ⏎ - it then looks up `FI_ALLREDUCE_FUSION_MAX_SIZE_MB[capability]`, which for SM 10.3 ⏎   (GB300) had no `16` entry. ⏎  ⏎ When either is missing, `flashinfer_max_size(16)` is `None`, s …[truncated]

### L3-710ebaa189  (L3, 2026-06-25, sha 710ebaa1897e, PR #46114)
TITLE: [ROCm][Bugfix] Fix chunk alignment when using context parallelism with TRITON_MLA (#46114)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+6/-9); tests/distributed/test_context_parallel.py (+47/-30)
LABELS: bug, rocm, ready
BODY: There is a bug in mla_attention when using context parallelism on ROCm. `max_context_chunk` is being aligned properly on CUDA because of the `self.aot_schedule` path (which is CUDA-only). The chunk misalignment causes zero accuracy in `test_context_parallel.py` on ROCm with dcp_size=4 using `TRITON_MLA`. ⏎  ⏎ ``` ⏎ tests/distributed/test_context_parallel.py::test_cp_generation[deepseek-ai/DeepSeek-V2-Lite-Chat-parallel_setup0-mp-auto-test_options0] …[truncated]

### L3-1273a8f05a  (L3, 2026-06-25, sha 1273a8f05a1a, PR #36559)
TITLE: [Kernel] Add swap AB optimization to fused_moe_kernel (#36559)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/fused_moe.py (+49/-14); vllm/model_executor/layers/fused_moe/utils.py (+10/-0)
LABELS: performance, ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ This PR add swap AB optimization to `fused_moe_kernel` to make better use of WGMMA. Similar as #20396. ⏎  ⏎ For small BLOCK_SIZE_M cases (BLOCK_SIZE_M < 64), change the gemm from `a[M, K] @ b[K, N]-> c[M, N]` to `b[N, K] @ a[K, M] -> c[N, M]`. ⏎  ⏎ 2% to 4% e2e output token throughput improvement. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest -s -v tests/kernels/moe/test_moe.py ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ Unit tests passed. ⏎  ⏎ ## Micro bench ⏎  ⏎ Micro benc …[truncated]

### L3-15be78732b  (L3, 2026-06-25, sha 15be78732bac, PR #45019)
TITLE: [NIXL][Mamba] Add Mamba1 support to NIXL P/D disaggregation (#45019)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/nixl_integration/config_sweep_accuracy_test.sh (+2/-0); tests/v1/kv_connector/nixl_integration/test_accuracy.py (+1/-0); tests/v1/kv_connector/unit/test_nixl_connector_hma.py (+45/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+14/-8); vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py (+42/-27)
LABELS: ready, v1, kv-connector
BODY: ## Purpose ⏎ Add Mamba1 hybrid-model support (e.g. Jamba) to the NIXL connector's conv-state transfer for prefill/decode disaggregation, extending the existing Mamba2/GDN path (#41869, #37635). ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎ Successfully passed - ⏎ `pytest tests/v1/kv_connector/unit/test_nixl_connector_hma.py -k derive_mamba_conv_split -v` ⏎  ⏎ ### Accuracy Test ⏎  ⏎ ## Accuracy Results ⏎  ⏎   **Model:** `ai21labs/AI21-Jamba2-Mini` (Mamba1) ⏎   **Benchm …[truncated]

### L3-8fa36fbbeb  (L3, 2026-06-25, sha 8fa36fbbebdf, PR #46506)
TITLE: [Bugfix] FLASHINFER_MLA_SPARSE_SM120 compatibility with GLM-5 NVFP4 (#46506)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer_sparse
FILES: vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py (+6/-4)
LABELS: bug, ready, v1, nvidia
BODY: Models with an index_topk skip pattern (e.g. GLM-5.2, index_topk_freq=4) ⏎ build a sparse-MLA indexer only on a subset of backbone layers; the ⏎ remaining "skip" layers are constructed with indexer=None and instead ⏎ receive the shared top-k indices buffer via MLAAttention ⏎ (extra_impl_args["topk_indices_buffer"], see mla_attention.py). ⏎  ⏎ FLASHINFER_MLA_SPARSE_SM120 asserted `indexer is not None` and only read ⏎ `indexer.topk_indices_buffer`, so eve …[truncated]

### L3-e53a17232c  (L3, 2026-06-25, sha e53a17232c31, PR #46692)
TITLE: [ROCm]: Bump aiter to 0.1.16.post2 (#46692)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm_base (+2/-2)
LABELS: rocm, ready, ci/build
BODY: AITER 0.1.16.post2; perf uplifts+bugfixes for gpt-oss, DSV4, Minimax M3, etc. ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e8e7b592d1  (L3, 2026-06-25, sha e8e7b592d11d, PR #46642)
TITLE: [Kernel][MoE] Tune block-FP8 fused MoE for low-batch decode (#46642)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_moe.py (+10/-7); vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json (+10/-10); vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json (+9/-9); vllm/model_executor/layers/fused_moe/configs/E=160,N=640,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json (+7/-7); vllm/model_executor/layers/fused_moe/configs/E=256,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json (+13/-13); vllm/model_executor/layers/fused_moe/configs/E=256,N=384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json (+7/-7); vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json (+7/-7); vllm/model_executor/layers/fused_moe/configs/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json (+12/-12); vllm/model_executor/layers/fused_moe/configs/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json (+9/-9); vllm/model_executor/layers/fused_moe/fused_moe.py (+26/-7)
LABELS: performance, ready
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Justification ⏎  ⏎ Block-quantized FP8 fused MoE was locked to `BLOCK_SIZE_N = block_shape[0]` (128) at every batch size — in both the runtime heuristic (`get_default_config`) and the tuner search space. At decode the gate-up GEMM is SM-bound: a 128-wide N tile produces too few thread blocks to fill the GPU. ⏎  ⏎ This was never a kernel requirement. The Triton kernel indexes block scales per N element (`offs_bn // group_n`), so a narrower N tile that  …[truncated]
