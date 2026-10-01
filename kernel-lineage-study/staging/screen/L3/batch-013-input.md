### L3-cf8a613a87  (L3, 2026-04-23, sha cf8a613a8726, PR #37892)
TITLE: Support only half types for concat_mla_q kernel (#37892)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+4/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ This PR fixes `concat_mla_q` kernel for float32 dtype ⏎  ⏎ * Currently in `concat_mla_q` kernel, `q_pe` is casted to int* (4 bytes), so `ld32_cs` loads the rope section in 32 * 4 = 128 bytes, see https://github.com/vllm-project/vllm/blob/v0.18.0/csrc/concat_mla_q.cuh#L50-L55. This works correctly if DType is bf16 or half, because 64 rope elements take 64 * 2 = 128 bytes ⏎ * However, if DType is float32, 64 rope elements take 64 * 4 = 2 …[truncated]

### L3-fe85a92e86  (L3, 2026-04-24, sha fe85a92e8621, PR #40654)
TITLE: [Core] Avoid seq_lens_cpu GPU->CPU sync (#40654)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_sparse, L3.dispatch.abstract_interface, L3.flex_attention
FILES: vllm/model_executor/layers/attention/cross_attention.py (+12/-4); vllm/model_executor/layers/attention/mla_attention.py (+10/-5); vllm/v1/attention/backend.py (+6/-0); vllm/v1/attention/backends/flex_attention.py (+5/-4); vllm/v1/attention/backends/mla/flashmla_sparse.py (+4/-1); vllm/v1/attention/backends/mla/indexer.py (+6/-2); vllm/v1/attention/backends/utils.py (+7/-1); tests/v1/attention/utils.py (+1/-0); tests/v1/spec_decode/test_tree_attention.py (+3/-1); vllm/v1/spec_decode/dflash.py (+7/-0); (+9 more)
LABELS: speculative-decoding, ready, v1
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:7d3195ea9f (fix PR 40772) || deep-study performance PR (system_performance)
BODY: With help from claude

### L3-e9f331d72e  (L3, 2026-04-24, sha e9f331d72e90, PR #40746)
TITLE: [MRV2] Ensure warmup covers prefill path (#40746)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/warmup.py (+9/-6)
LABELS: ready, v1
BODY: Ensure dummy prefill warmup batch is not misclassified as a uniform decode batch.

### L3-7d3195ea9f  (L3, 2026-04-24, sha 7d3195ea9fc8, PR #40772)
TITLE: [Bugfix] Fix IMA in DSA + MTP (#40772)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+14/-7)
LABELS: bug, ready
DEEP_STUDY: deep-study correctness case vllm:7d3195ea9f: class=memory_safety_oob; symptom=illegal_memory_access; introducing=#40654
BODY: This PR fixes the IMA bug when using DSA with MTP. The bug was introduced in #40654

### L3-fa4b70555b  (L3, 2026-04-24, sha fa4b70555bc1, PR #39999)
TITLE: [ROCm] Cast score correction bias tensor during model construction for DeepSeek/Kimi-K2 (#39999)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/_aiter_ops.py (+2/-0); vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py (+1/-1); vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py (+1/-1); vllm/model_executor/models/deepseek_v2.py (+15/-0)
LABELS: rocm, ready, deepseek
BODY: ## Purpose ⏎ The moe score correction bias tensor was being cast to the gate output dtype on every forward pass. The datatype that this tensor needs to be cast to is known at model construction time and never changes beyond that. So this repeated cast is redundant work that launches an extra GPU kernel per MoE layer per forward call. ⏎  ⏎ This PR moves the cast to the model construction thereby eliminating the per-forward-pass overhead. ⏎  ⏎ ## Summar …[truncated]

### L3-9f771b3ab9  (L3, 2026-04-24, sha 9f771b3ab92d, PR #34556)
TITLE: [Quantization] add humming quantization kernel (#34556)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/config/model.py (+4/-0); vllm/envs.py (+32/-0); vllm/model_executor/layers/fused_moe/fused_humming_moe.py (+690/-0); vllm/model_executor/layers/fused_moe/layer.py (+5/-1); vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py (+202/-0); vllm/model_executor/layers/linear.py (+13/-6); vllm/model_executor/layers/quantization/__init__.py (+3/-0); vllm/model_executor/layers/quantization/humming.py (+962/-0); vllm/model_executor/layers/quantization/utils/humming_moe_utils.py (+35/-0); vllm/model_executor/parameter.py (+2/-2)
LABELS: performance, ready, gpt-oss, nvidia, quantization
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Introduction ⏎  ⏎ [Humming](https://github.com/inclusionAI/humming/) is a highly flexible JIT quantization kernel library. ⏎  ⏎ It supports inference for the vast majority of quantization types and offers performance superior to the Marlin kernel. ⏎  ⏎ - **Supported Data Types:** W{1-8}A{16/8/4} ⏎ - **Currently Compatible Weight Types:** GPTQ / AWQ / FP8 / MXFP4 / NVFP4 / BITNET ⏎ - **Unsupported Types (No support planned):** GPTQ with `desc_act=True` …[truncated]

### L3-e8ee2a78db  (L3, 2026-04-24, sha e8ee2a78dbc0, PR #40045)
TITLE: [Attention] use diff kv backend for mimo v2 flash (#40045)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fa4_cutedsl, L3.flash_attn.fa_utils
FILES: vllm/model_executor/layers/attention/attention.py (+1/-0); vllm/v1/attention/backends/fa_utils.py (+22/-3); vllm/v1/attention/backends/flash_attn_diffkv.py (+18/-4); vllm/vllm_flash_attn/flash_attn_interface.py (+1/-0); docs/design/attention_backends.md (+1/-1); tools/pre_commit/generate_attention_backend_docs.py (+41/-8); vllm/model_executor/models/mimo_v2_flash.py (+14/-8); vllm/v1/kv_cache_interface.py (+14/-0)
LABELS: documentation, ready, v1
BODY: ## Purpose ⏎  ⏎ ### Diff kv ⏎ A key characteristic of mimo v2 flash architecture is that the attention layer uses different head dimensions for keys and values (v_head_dim != head_dim) ⏎  ⏎ We use `FlashAttentionDiffKVBackend` to avoid padding ⏎  ⏎ ### Attention sink ⏎  ⏎ MiMo V2 Flash uses sliding-window attention (SWA) layers with an attention sink mechanism, passing sink biases via the sinks (`s_aux`) parameter to the FlashAttention kernel. ⏎  ⏎ When sin …[truncated]

### L3-ce6a199ecc  (L3, 2026-04-24, sha ce6a199ecc09, PR #40715)
TITLE: [BE][Bugfix] Respect TORCH_COMPILE_DISABLE env var at the vLLM config level for torch 2.12 (#40715)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/test_config.py (+16/-0); vllm/config/vllm.py (+7/-0)
LABELS: bug, ready
ISSUES: #181247 [vllm] [2.12 regression][CPU] torch.compile fullgraph=True raises "found no compiled frames" under Intel SDE (TORCH_COMPILE_DISABLE=1 not honored?)
BODY: ## Purpose ⏎ Fixes https://github.com/pytorch/pytorch/issues/181247 ⏎  ⏎ - PyTorch https://github.com/pytorch/pytorch/pull/177809 (landed 2026-04-14, by William Wen) added a runtime check that errors when fullgraph=True but no frame was actually compiled: "torch.compile with fullgraph=True found no compiled frames". This breaks any code that sets TORCH_COMPILE_DISABLE=1 while vLLM still calls torch.compile(..., fullgraph=True). ⏎ - Previously, TORCH_ …[truncated]

### L3-4c34b2f6fc  (L3, 2026-04-24, sha 4c34b2f6fc63, PR #39466)
TITLE: [XPU] Enable torch.compile for XPU GDN attention (#39466)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/_xpu_ops.py (+73/-0); vllm/config/compilation.py (+1/-0); vllm/model_executor/layers/mamba/gdn_linear_attn.py (+7/-41)
LABELS: intel-gpu, ready, v1
BODY: ## Purpose ⏎ Enable torch.compile for XPU GDN attention. ⏎  ⏎ ### Issue descrption ⏎ Dynamo failed to trace the XPU GDN kernel call. ⏎ Code: https://github.com/vllm-project/vllm/blob/1b19bd758936496751432eccabf8adb7b5d8936a/vllm/model_executor/layers/mamba/gdn_linear_attn.py#L622-L669 ⏎  ⏎ Traced graph **miss XPU GDN kernel**: ⏎ ``` ⏎ # File: /home/yuwenzho/qwen3.5_tc/vllm-pr/vllm/model_executor/layers/mamba/gdn_linear_attn.py:612 in forward_xpu, code: co …[truncated]

### L3-914d0464c1  (L3, 2026-04-24, sha 914d0464c1b1, PR #40631)
TITLE: [Refactor] Unify 2D/3D kernels in triton_unified_attention (#40631)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/v1/attention/ops/triton_attention_helpers.py (+383/-0); vllm/v1/attention/ops/triton_unified_attention.py (+321/-888)
LABELS: ready, v1
BODY: This PR is the refactor-only half of the split that @tdoublep and @bringlein requested on #39074. There are no new features and no behavior changes here. The goal is simply to remove duplication by collapsing two nearly identical attention kernels into one. ⏎  ⏎ On main, triton_unified_attention.py contains two kernels that are almost the same. kernel_unified_attention_2d processes the full sequence in a single tile loop and writes directly to the  …[truncated]

### L3-095d2f87e8  (L3, 2026-04-24, sha 095d2f87e851, PR #40763)
TITLE: [Bug] Fix GLM-5.1 running error on ROCm platform (#40763)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.rocm_aiter, L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+55/-24); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+10/-3); vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+3/-0)
LABELS: bug, rocm, ready, v1
BODY: 1. AITER MLA implementation requires num_heads>=16. Add padding when necessary for both regular MLA and sparse MLA; ⏎ 2. An AITER bug will cause divided_by_zero error when num_heads is too small. Set it to num_heads to circumvent, and will remove it when corresponding AITER PR gets merged. ⏎  ⏎ ## Test Plan ⏎ __To verify sparse mla padding works correctly:__ ⏎ 1. **Padding added since MLA num_heads==8:** ⏎ VLLM_ROCM_USE_AITER=1 vllm serve /data/GLM-5.1 …[truncated]

### L3-95995bbef8  (L3, 2026-04-25, sha 95995bbef812, PR #38503)
TITLE: [ROCm][Engine] Fix GPU memory leaks in engine shutdown and test workaround for async KV prefix cache reset (#38503)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm (+14/-5); tests/entrypoints/openai/completion/test_shutdown.py (+1/-1); tests/v1/kv_connector/unit/test_rixl_gpu_mem_diag.py (+222/-0); tests/v1/kv_offload/test_cpu_offloading.py (+12/-3); vllm/utils/func_utils.py (+25/-4); vllm/v1/engine/core.py (+7/-0); vllm/v1/worker/gpu_model_runner.py (+14/-0); vllm/v1/worker/gpu_worker.py (+5/-0)
LABELS: rocm, ready, ci/build, v1, kv-connector
BODY: When running in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), shutting down the engine leaked GPU memory because: ⏎  ⏎ - **`supports_kw` LRU cache pinned model weights**: The `@lru_cache` on `supports_kw` was keyed on bound methods (e.g. `model.forward`), which held strong references to the model instance and all its GPU tensors for the lifetime of the cache. Fixed by unwrapping bound methods to `__func__` before caching. ⏎ - **`gc.freeze()` was nev …[truncated]

### L3-12a3f6454b  (L3, 2026-04-25, sha 12a3f6454b97, PR #40865)
TITLE: [Bugfix][MoE] Only unpad routed output before shared expert add or routed output transform (#40865)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/runner/moe_runner.py (+5/-1)
LABELS: bug, ready
BODY: Only trim padded routed output before shared+routed add / latent moe up-proj. No-shared MoE keeps late truncation. ⏎  ⏎ Not duplicate: #40853 (draft auto-pr) reverts #40794; this preserves the shared-output fix. ⏎  ⏎ I've reproduced the following error locally before this pr's change, and the test passes with the change: ⏎ ``` ⏎ python -m pytest -sv "./tests/evals/gpt_oss/test_gpqa_correctness.py::test_gpqa_correctness[gpt-oss-20b-flashinfer-mxfp4-bf16 …[truncated]

### L3-9558f43903  (L3, 2026-04-26, sha 9558f43903fa, PR #40893)
TITLE: [Bugfix] Size FlashInfer NVLink MNNVL workspace to EP group (#40893)
SOURCES: subject_keyword, release_notes, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/distributed/device_communicators/all2all.py (+13/-7)
LABELS: bug, ready, tool-calling, deepseek, kv-connector
DEEP_STUDY: deep-study correctness case vllm:9558f43903: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Summary ⏎ Fix `FlashInferNVLinkOneSidedManager` and `FlashInferNVLinkTwoSidedManager` so the MNNVL workspace is sized to the EP group instead of the DP group. ⏎  ⏎ In both managers, the flashinfer-allocated workspace tensor's leading dimension is determined by the comm_backend's group size (via `MnnvlMemory.set_comm_from_config` -> `comm_backend.Split(...)` -> `Get_size`; vLLM's `CustomCommunicator.Split` is a no-op that returns `self`), while th …[truncated]

### L3-8cd174fa35  (L3, 2026-04-26, sha 8cd174fa3583, PR #40338)
TITLE: [LoRA] MoE LoRA Refactor (#40338)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/lora/layers/fused_moe.py (+36/-297); vllm/lora/layers/utils.py (+5/-4); vllm/lora/ops/triton_ops/utils.py (+17/-0); vllm/lora/punica_wrapper/punica_base.py (+62/-0); vllm/lora/punica_wrapper/punica_gpu.py (+236/-0); vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py (+52/-3); vllm/model_executor/layers/fused_moe/fused_marlin_moe.py (+106/-6); vllm/model_executor/layers/fused_moe/fused_moe.py (+48/-1); vllm/model_executor/layers/fused_moe/lora_context.py (+44/-0); vllm/model_executor/layers/fused_moe/lora_experts_mixin.py (+111/-0); (+4 more)
LABELS: rocm, intel-gpu, ready, gpt-oss, nvidia
BODY: ## Motivation ⏎  ⏎  ⏎ Currently, MoE LoRA is wired in by monkey-patching methods on the modular kernel at construction time. `FusedMoEWithLoRA._inject_lora_into_fused_moe` wraps `FusedMoEKernel.apply`, `TritonExperts.activation`, and `TritonExperts.moe_sum` with `fwd_decorator` / `act_decorator` / `moe_sum_decorator`, and smuggles tensors between them through  `moe_state_dict`. This has several concrete problems: ⏎  ⏎ ### 1. Hacky and hard to maintain …[truncated]

### L3-b39c266dae  (L3, 2026-04-26, sha b39c266dae8c, PR #40346)
TITLE: [KV Offload] Offload all KV blocks when doing prefill in P/D (#40346)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py (+56/-0); tests/v1/kv_connector/unit/offloading_connector/utils.py (+7/-1); vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py (+8/-1)
LABELS: ready, v1, kv-connector
BODY: ## Purpose ⏎ Support P/D (Prefill/Decode) disaggregation in the OffloadingConnector by ensuring the prefill instance offloads all KV blocks to CPU — not just the incremental delta since the last store. ⏎  ⏎ When kv_transfer_params["do_remote_decode"] is set, _get_reqs_to_store now resets start_block_idx to 0, so the full KV cache is available on CPU for a remote decode node to consume. ⏎  ⏎ Without this change, blocks that were already tracked by next …[truncated]

### L3-32e45636e3  (L3, 2026-04-26, sha 32e45636e3d7, PR #38373)
TITLE: [torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation (#38373)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/correctness_e2e/test_sequence_parallel.py (+2/-0); tests/compile/passes/distributed/test_async_tp.py (+17/-0); tests/compile/passes/distributed/test_sequence_parallelism.py (+19/-0); tests/compile/test_config.py (+118/-1); vllm/compilation/passes/fusion/collective_fusion.py (+7/-10); vllm/compilation/passes/fusion/sequence_parallelism.py (+16/-26); vllm/config/compilation.py (+19/-0); vllm/config/vllm.py (+17/-28); vllm/v1/worker/utils.py (+8/-15)
LABELS: ready, v1, verified
ISSUES: #35771 [RFC][torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation
BODY: ## Purpose ⏎ Disable Sequence Parallelism for piecewise compilation in v1, and only keep SP enabled for full-graph compilation paths. ⏎  ⏎ This change adds an explicit config-level guard so that when v1 is using Dynamo FX-level graph splitting, `enable_sp` and `fuse_gemm_comms` are turned off early. The SP-related compilation passes are updated to match the same contract, and the v1 worker-side residual handling is adjusted to follow the new assumpt …[truncated]

### L3-4d51588e23  (L3, 2026-04-26, sha 4d51588e2381, PR #40860)
TITLE: [Feat] DeepSeek V4 Rebased  (#40860)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.mla.common_v1, L3.mla.flashmla_build, L3.mla.flashmla_sparse, L3.dispatch.abstract_interface
FILES: CMakeLists.txt (+5/-2); cmake/external_projects/deepgemm.cmake (+6/-1); cmake/external_projects/flashmla.cmake (+1/-1); requirements/cuda.txt (+2/-0); csrc/cpu/pos_encoding.cpp (+6/-1); csrc/cpu/torch_bindings.cpp (+2/-1); csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu (+477/-0); csrc/layernorm_kernels.cu (+15/-7); csrc/layernorm_quant_kernels.cu (+28/-10); csrc/moe/moe_ops.h (+9/-0); (+140 more)
LABELS: documentation, new-model, frontend, speculative-decoding, ci/build, v1, tool-calling, deepseek, cpu, gpt-oss
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:124fac10cb (fix PR 42379)
BODY: ## Purpose ⏎  ⏎ Rebased version of #40760  ⏎  ⏎ Roadmap: https://github.com/vllm-project/vllm/issues/40902 ⏎  ⏎ Co-authored by: Bugen Zhao, Giancarlo Delfin, Jie Li, Kaichao You, Roy Wang, Woosuk Kwon, Yifan Qiao, Yongye Zhu, Zhewen Li, Zijing Liu, Zixi Qi ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-5d5c776444  (L3, 2026-04-27, sha 5d5c7764446d, PR #38065)
TITLE: [Perf] FP8 FlashInfer Attn for ViT (#38065)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/config/model.py (+12/-0); vllm/config/multimodal.py (+51/-0); vllm/engine/arg_utils.py (+28/-0); vllm/model_executor/layers/attention/mm_encoder_attention.py (+336/-15); vllm/utils/flashinfer.py (+35/-0); vllm/v1/attention/ops/vit_attn_wrappers.py (+29/-2); benchmarks/kernels/benchmark_vit_fp8_attn.py (+324/-0); docs/features/quantization/README.md (+1/-0); docs/features/quantization/fp8_vit_attn.md (+109/-0); tests/config/test_multimodal_config.py (+18/-0); (+8 more)
LABELS: documentation, performance, ready, v1, multi-modality, qwen, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ For visual understanding workloads with large images and relatively short text prompts/generation, the ViT encoder attention becomes a significant bottleneck, especially when the text model is quantized, e.g. with nvfp4. This PR adds optional FP8 quantization for ViT encoder attention via the FlashInfer cuDNN backend. ⏎  ⏎ This is part of the efforts of NVIDIA MLPerf Inference Submission V6.0 @wangshangsam. Credit to @ybgao-nvidia for …[truncated]

### L3-2cc008e7b4  (L3, 2026-04-27, sha 2cc008e7b491, PR #40941)
TITLE: [Attention][TurboQuant] Share dequant buffers, eliminate float16_copy (#40941)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.triton.decode_attention
FILES: vllm/model_executor/layers/attention/attention.py (+0/-48); vllm/v1/attention/backends/turboquant_attn.py (+83/-40); vllm/v1/attention/ops/triton_decode_attention.py (+5/-1); vllm/v1/attention/ops/triton_turboquant_decode.py (+10/-3)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: # TurboQuant Backend Optimizations ⏎  ⏎ **Builds on top of**: shared-decode-buffer PR (#40655) ⏎  ⏎ ## Summary ⏎  ⏎ Backend-level optimizations for the TurboQuant KV cache attention backend, reducing memory usage and improving prefill latency for long-context workloads. ⏎  ⏎ **Key results** (L4 TP=4, GPTQ-Int4 + TQ k8v4): ⏎ - **Saves 57 GB memory at 1M context** — shared dequant buffers via WorkspaceManager in the **continuation prefill path** (60× reduction; the  …[truncated]

### L3-c8bbe05189  (L3, 2026-04-27, sha c8bbe05189ba, PR #39141)
TITLE: [Perf] Update TRTLLM supported MoE routing methods (#39141)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/config.py (+14/-7); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+9/-43); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+3/-29); vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (+0/-1); vllm/model_executor/models/deepseek_v2.py (+0/-12)
LABELS: ready, deepseek, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Flashinfer v0.6.8 recently adds more support for different MoE routing methods (e.g., minimax m2). This PR updates the routing methods enums corresponding to flashinfer, as well as the supported routing methods in Trtllm MoE fp8 and nvfp4. ⏎  ⏎ In addition, the requirement for fp32 router logits for deepseek MoE is now removed. ⏎  ⏎ ## Test Plan ⏎ Test e2e accuracy with Deepseek R1 and Minimax M2.5 fp8 and nvfp4, and Deepseek V4. ⏎  ⏎ ## Tes …[truncated]

### L3-985961345a  (L3, 2026-04-27, sha 985961345a13, PR #39855)
TITLE: [Bugfix] Install libcublas-dev in Dockerfile for FlashInfer CuTe DSL JIT (#39855)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1)
LABELS: bug, ready, ci/build
BODY: ## Summary ⏎ - The runtime image's JIT compilation dependencies section installs `libcublas` (runtime-only) instead of `libcublas-dev`, causing FlashInfer's `flashinfer_cutedsl` MoE backend to fail with `fatal error: cublasLt.h: No such file or directory` when JIT-compiling the `moe_utils` module at startup. ⏎ - Fix: change `libcublas-${CUDA_VERSION_DASH}` to `libcublas-dev-${CUDA_VERSION_DASH}` (the `-dev` package includes `cublasLt.h` and depends …[truncated]

### L3-fd74c90d9c  (L3, 2026-04-27, sha fd74c90d9c3b, PR #39930)
TITLE: [Attention][Spec Decode] Allow independent drafter attention backend selection (#39930)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/config/attention.py (+3/-0); vllm/config/speculative.py (+15/-1); vllm/v1/attention/selector.py (+1/-6); tests/kernels/attention/test_attention_selector.py (+87/-0); vllm/v1/spec_decode/dflash.py (+12/-0); vllm/v1/spec_decode/llm_base_proposer.py (+18/-4)
LABELS: speculative-decoding, ready, v1
BODY: ## Purpose ⏎  ⏎ Currently, when running with speculative decoding, the drafter's attention backend is pegged to be identical to that of the target model. This is problematic when these have incompatible requirements (e.g. MLA vs GQA) or the drafter has restrictive requirements (e.g. DFlash, which requires non-causal support). ⏎  ⏎ This PR adds `--speculative-config.attention_backend` (analogous to `--speculative-config.moe_backend`) to allow independ …[truncated]

### L3-ed57f77192  (L3, 2026-04-27, sha ed57f7719237, PR #40859)
TITLE: [Bugfix ] fix bailing_moe_linear (#40859)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/mamba_utils.py (+0/-3); vllm/model_executor/models/bailing_moe_linear.py (+15/-13)
LABELS: bug, ready, verified
BODY: ## Purpose ⏎ 1. fix decode idx err vllm v1 reorders batches as decode-first, but _decode_infer was ported from SGLang which uses prefill-first ordering. In mixed batches, this caused decode requests to read q/k/v from wrong positions and update wrong KV cache state slots, corrupting the recurrent state and leading to degenerate output (repeated newlines) in long-form generation. ⏎  ⏎     The bug is silently masked in decode-only batches (num_prefill …[truncated]

### L3-407b34be26  (L3, 2026-04-28, sha 407b34be2633, PR #41019)
TITLE: [xpu] bump up vllm-xpu-kernel v0.1.7 (#41019)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/intel_jobs/lora_intel.yaml (+0/-1); requirements/xpu.txt (+1/-1)
LABELS: intel-gpu, ready, ci/build
BODY: ## Purpose ⏎ bump up vllm-xpu-kernel v0.1.7 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-a60883644b  (L3, 2026-04-28, sha a60883644be0, PR #41134)
TITLE: [Build] Defer flashinfer cubin download to avoid ~2.5 GB (decompressed) layer duplication (#41134)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+8/-3)
LABELS: ready, ci/build
BODY: Defer `flashinfer download-cubin` to after all pip installs, eliminating ~2.5 GB of cross-layer file duplication in the Docker image. ⏎  ⏎ ## Purpose ⏎  ⏎ While looking at reducing image size with `Dive` and Claude Opus, I find few things. Here is the first one; ⏎  ⏎ Running `flashinfer download-cubin` in the flashinfer-jit-cache install layer (L586) writes ~10,600 precompiled cubin files into the `flashinfer_cubin` package directory. When the vLLM whe …[truncated]

### L3-de3fe8dc62  (L3, 2026-04-28, sha de3fe8dc62f3, PR #40449)
TITLE: [Bugfix] release KV blocks for skipped P-ranks to prevent invalid KV errors and timeouts when P_tp > D_tp and MLA (#40449)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector.py (+119/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py (+1/-1)
LABELS: bug, ready, v1, kv-connector
BODY: ## Purpose ⏎ Fix KV block delayed release caused by incorrect request ID when notifying skipped P ranks in MLA PD-disaggregated serving with $P_{tp} > D_{tp}$ (e.g., P: tp8, D: tp4). ⏎  ⏎ Root Cause: ⏎ In MLA architecture, KV Cache is fully replicated across all TP ranks on the P side — D only needs to read from one P rank to obtain a complete KV Cache. _read_blocks_for_req correctly breaks after reading remote_ranks[0], but then sends a "free notifi …[truncated]

### L3-7fd05e05ae  (L3, 2026-04-28, sha 7fd05e05aeb3, PR #40842)
TITLE: uncomment flex backend for batch invariant mode (#40842)
SOURCES: path_core
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+7/-9); tests/v1/determinism/utils.py (+1/-0)
LABELS: ready, v1
BODY: re-enabling Flex Attention as an accepted backend for batch invariant mode. to fix the IMA issue, we need to use the strides of the persistent buffer instead of the current tensor so that the memory layout matches what was captured by the cuda graphs.  ⏎  ⏎ `python -m pytest tests/v1/determinism/test_batch_invariance.py -v -k "FLEX_ATTENTION" -s` ⏎ before fix:  ⏎ <img width="2614" height="1406" alt="image" src="https://github.com/user-attachments/ass …[truncated]

### L3-fa1b9840f6  (L3, 2026-04-28, sha fa1b9840f6d8, PR #40845)
TITLE: [BE][Torch 2.12] Remove workaround code for fixed cublas issue (#40845)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/batch_invariant.py (+11/-11)
LABELS: ready
ISSUES: #181248 [vllm] [2.12 regression][B200] test_batch_invariance: nondeterministic outputs 3/5 trials with FLASH_ATTN (B200 only, H100 passes)
BODY: ## Purpose ⏎  ⏎ Fixes https://github.com/pytorch/pytorch/issues/181248 ⏎  ⏎ To do so, we modify vllm/model_executor/layers/batch_invariant.py:enable_batch_invariant_mode() — restrict the Triton persistent matmul override (mm/addmm/matmul/linear) to SM80 (Ampere) only. Hopper and Blackwell now share the same path                                     ⏎  ⏎ The SM100 branch was added as a workaround for a torch-2.9-era cuBLAS bug where B200 selected a non-b …[truncated]

### L3-2ae73c758c  (L3, 2026-04-28, sha 2ae73c758cee, PR #41135)
TITLE: [Bugfix] fix inductor error for dpsk v4 (#41135)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py (+106/-36)
LABELS: bug, ready, v1
ISSUES: #41106 [Bug] Deepseek v4 :torch._inductor.exc.InductorError: AssertionError:
BODY: ## Purpose ⏎  ⏎ Fix https://github.com/vllm-project/vllm/issues/41106 ⏎  ⏎ also see https://github.com/pytorch/pytorch/issues/181735 ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek-V4-Flash   --trust-remote-code   --kv-cache-dtype fp8   --block-size 256   --enable-expert-parallel   --data-parallel-size 1 --tensor-parallel-size 8   --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}'   --max-num-batched-tokens 8 …[truncated]

### L3-856b15c62c  (L3, 2026-04-29, sha 856b15c62c8a, PR #41072)
TITLE: [CI][AMD][BugFix] Patch has_flashinfer decorator for test_select_rocm_aiter_backend  (#41072)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_unquantized_backend_selection.py (+1/-1)
LABELS: bug, rocm, ready
BODY: ## Purpose ⏎ The `has_flashinfer` function has been moved from `vllm.model_executor.layers.fused_moe.oracle.unquantized` to  ⏎ ` vllm.utils.flashinfer.has_flashinfer ` but the patch decorator [here ](https://github.com/vllm-project/vllm/blob/main/tests/kernels/moe/test_unquantized_backend_selection.py#L88) hasn't been updated.  This PR updates the patch decorator so the test now passes. ⏎ ## Test Plan ⏎ ` pytest -sv kernels/moe/test_unquantized_backe …[truncated]

### L3-3f1a4bb639  (L3, 2026-04-29, sha 3f1a4bb639a9, PR #40653)
TITLE: build: embed image provenance metadata in vLLM containers (#40653)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/image_build/image_build.sh (+1/-0); .buildkite/release-pipeline.yaml (+108/-8); .buildkite/scripts/docker-build-metadata-args.sh (+54/-0); .buildkite/test_areas/docker.yaml (+16/-0); docker/Dockerfile (+16/-0); docker/Dockerfile.cpu (+1/-0); docker/docker-bake.hcl (+28/-2); tests/tools/test_docker_build_metadata_args.py (+152/-0)
LABELS: ready, ci/build, cpu
BODY: ## Summary ⏎  ⏎ Add a light-touch provenance contract to CUDA `vllm-openai` container images: ⏎  ⏎ - Embed build provenance as both `VLLM_*` env vars and OCI/custom labels. ⏎ - Record whether an image came from a local, CI, nightly, or release build. ⏎ - Record source repo, commit, Buildkite pipeline/build URL, public image ref, staging image ref, CUDA/Ubuntu/base-image inputs, torch arch list, FlashInfer AOT, and KV connector settings. ⏎ - Wire Buildkite rele …[truncated]

### L3-68dd7db810  (L3, 2026-04-29, sha 68dd7db81001, PR #34668)
TITLE: [Reasoning] Support for speculative decoding with thinking budget (#34668)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/entrypoints/openai/chat_completion/test_thinking_token_budget.py (+181/-4); tests/v1/logits_processors/test_correctness.py (+338/-85); tests/v1/sample/utils.py (+23/-2); tests/v1/streaming_input/test_gpu_model_runner_streaming.py (+0/-1); vllm/v1/sample/logits_processor/__init__.py (+0/-3); vllm/v1/sample/logits_processor/builtin.py (+1/-262); vllm/v1/sample/metadata.py (+6/-0); vllm/v1/sample/rejection_sampler.py (+16/-3); vllm/v1/sample/sampler.py (+16/-1); vllm/v1/sample/thinking_budget_state.py (+528/-0); (+2 more)
LABELS: frontend, ready, v1, verified
BODY: This PR provide support and and compatibility for thinking budget with speculative decoding. ⏎ For further details on implementation please refer : https://github.com/vllm-project/vllm/pull/34668#issuecomment-4066967952 ⏎  ⏎ The PR extends to refactor the thinking budget design by moving it out of the logitsprocessor.  ⏎ This PR provides the following compatibilities with thinking budget:

### L3-39a7f4f4e2  (L3, 2026-04-29, sha 39a7f4f4e263, PR #41163)
TITLE: [Perf] Optimize `AllPool.forward` by slicing first, 51% faster in the method level benchmark (#41163)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/pooler/tokwise/methods.py (+11/-7)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Part of https://github.com/vllm-project/vllm/issues/35631 ⏎  ⏎ ## Test ⏎  ⏎ ### Acc ⏎  ⏎ Covered in current CI ⏎  ⏎ ### Perf ⏎  ⏎ ```bash ⏎ import gc ⏎ import statistics ⏎ import time ⏎  ⏎ import numpy as np ⏎ import torch ⏎  ⏎ from vllm.config import VllmConfig, set_current_vllm_config ⏎ from vllm.model_executor.layers.pooler.tokwise.methods import AllPool ⏎ from vllm.pooling_params import PoolingParams ⏎ from vllm.v1.pool.metadata import PoolingMetada …[truncated]

### L3-51fda1ba44  (L3, 2026-04-29, sha 51fda1ba44ff, PR #40648)
TITLE: [Model Runner v2] Fix block table IMA issue (#40648)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/block_table.py (+21/-12); vllm/v1/worker/gpu/model_runner.py (+3/-0); vllm/v1/worker/gpu_model_runner.py (+3/-0); vllm/v1/worker/gpu_worker.py (+3/-10)
LABELS: ready, v1, mrv2
BODY: ## Purpose ⏎  ⏎ Part of the https://github.com/vllm-project/vllm/pull/39337 ⏎  ⏎ `VLLM_USE_V2_MODEL_RUNNER=1 pytest tests/basic_correctness/test_cumem.py -k "test_end_to_end and opt-125m" -sv` ⏎  ⏎ Originally ⏎  ⏎ ```bash ⏎ (EngineCore pid=2694707)   File "/home/yewentao256/vllm-source/vllm/v1/engine/core.py", line 1205, in _process_engine_step ⏎ (EngineCore pid=2694707)     outputs, model_executed = self.step_fn() ⏎ (EngineCore pid=2694707)                 …[truncated]

### L3-8a8c9b564e  (L3, 2026-04-29, sha 8a8c9b564ef0, PR #39186)
TITLE: [KV Offload] Per-job store completion for CPU offloading connector (#39186)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py (+134/-0); tests/v1/kv_connector/unit/offloading_connector/test_worker_metadata.py (+32/-0); tests/v1/kv_connector/unit/offloading_connector/utils.py (+13/-21); tests/v1/kv_connector/unit/utils.py (+4/-0); vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py (+50/-5); vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py (+156/-48); vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py (+35/-71); vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py (+6/-0)
LABELS: ready, v1, kv-connector
BODY: When GPU→CPU KV cache stores complete, the scheduler currently only learns about it when the entire request finishes generating (EOS). Blocks that are physically on CPU remain invisible (`ref_cnt=-1`) for many steps. Future requests sharing the same prefix cannot reuse those blocks and must recompute. ⏎  ⏎ This PR reports store completions **per-job, per-step** so the scheduler can call `complete_store` on individual block groups as soon as their D …[truncated]

### L3-1628239eb2  (L3, 2026-04-29, sha 1628239eb234, PR #41246)
TITLE: [Multimodal][Render] Skip mm processor initialization and warmup for text-only mode (#41246)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/renderers/base.py (+1/-1)
LABELS: ready
ISSUES: #40365 [Bug]: Multi-modal warmup is run even in `language_model_only` mode
BODY: ## Purpose ⏎ - Fix https://github.com/vllm-project/vllm/issues/40365#issuecomment-4343691140 ⏎ - I think there is no need to initailize any mm processor with `--language-model-only` flag ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve /mnt/data0/LLM/gemma-4-E4B-it/ --language-model-only --max-model-len 8192 ⏎ ``` ⏎  ⏎ ## Test Result ⏎ mm processor no longer warmup, and text inputs can work normally: ⏎ ``` ⏎ (APIServer pid=3074039) INFO 04-29 22:18:07 [api_server.py:598 …[truncated]

### L3-b58669cb42  (L3, 2026-04-29, sha b58669cb427e, PR #41043)
TITLE: [Perf][Spec Decode] Avoid per-step numpy allocation in prepare_next_t… (#41043)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/spec_decode/llm_base_proposer.py (+5/-9)
LABELS: speculative-decoding, ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Description ⏎  ⏎ ### Purpose ⏎ Remove an unnecessary `np.array()` allocation that runs every decode step in the speculative decoding path. ⏎  ⏎ ### Background ⏎  ⏎ After the target model produces sampled tokens, the drafter needs the **next token ID** for each request to start drafting. There are two cases: ⏎  ⏎ 1. **Normal request**: The next token is the last sampled token from the target model ⏎ 2. **Discarded request** (e.g., chunked prefill not yet …[truncated]

### L3-faab189554  (L3, 2026-04-29, sha faab18955407, PR #37735)
TITLE: [Feature]: IndexCache support for DSA models (#37735)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mla.py (+8/-4); docs/features/index_cache.md (+54/-0); vllm/model_executor/models/deepseek_v2.py (+21/-1)
LABELS: documentation, ready, deepseek
ISSUES: #37684 [Feature]: IndexCache support for DSA models
BODY: ## Purpose ⏎ FIX https://github.com/vllm-project/vllm/issues/37684 ⏎  ⏎ ## Test ⏎ ``` ⏎  ⏎ vllm serve /mnt/data4/models/deepseek-ai/DeepSeek-V3___2 -tp=8  \ ⏎ --tokenizer-mode deepseek_v32 --enable-auto-tool-choice \ ⏎ --tool-call-parser deepseek_v32 --reasoning-parser deepseek_v3  \ ⏎ --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' ⏎  ⏎ ``` ⏎  ⏎  ⏎ ## Test Result ⏎  ⏎  ⏎ ``` ⏎ vllm bench serve --backend vllm  --endpoint /v1/completions --dataset- …[truncated]

### L3-6841f5dc77  (L3, 2026-04-29, sha 6841f5dc77e9, PR #39987)
TITLE: [ROCm] Add env flags to disable dynamic MXFP4 quant and enable AITER tuned GEMMs for Attention Projection Layers (#39987)
SOURCES: path_integration+keyword, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+1/-0); tests/quantization/test_quark_maybe_update_config.py (+0/-63); vllm/_aiter_ops.py (+7/-0); vllm/model_executor/layers/quantization/quark/quark.py (+4/-29); vllm/model_executor/layers/utils.py (+16/-11)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ For models in the DeepSeek family, dynamic MXFP4 quantization is enabled by default for the attention projection layers. This behavior may be suboptimal in certain deployment scenarios. ⏎  ⏎ This PR introduces two opt-in environment variables that provide finer control over this path: ⏎  ⏎ 1. The ability to disable dynamic MXFP4 quantization for attention projections. ⏎ 2. The ability to route unquantized attention GEMMs through AITER's ne …[truncated]

### L3-a966aaed30  (L3, 2026-04-29, sha a966aaed30b9, PR #39277)
TITLE: [Bugfix][MLA] Size arange_buffer to max_num_batched_tokens to prevent CUDA IMA (#39277)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+4/-1)
LABELS: bug, v1, nvidia, verified, DSv4
ISSUES: #40983 [Bug]: arange_buffer size mismatch
BODY: Hi from [novita.ai](https://novita.ai/) team 👋 ⏎  ⏎ ## Purpose ⏎  ⏎ Fix a CUDA illegal memory access (IMA) crash in the MLA indexer's decode flattening path when using data parallelism + speculative decoding (MTP). ⏎  ⏎ ### What changed ⏎  ⏎ `arange_buffer` in `DeepseekV32IndexerMetadataBuilder` was allocated with size `max_num_seqs * next_n`, while the other flattening buffers (`expanded_seq_lens_buffer`, `expanded_block_table_buffer`) correctly use `ma …[truncated]

### L3-296741d025  (L3, 2026-04-29, sha 296741d02571, PR #41015)
TITLE: [DSv4] Use `cvt` PTX for FP32->FP4 conversion (#41015)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py (+6/-6); vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py (+20/-35); tests/kernels/test_compressor_kv_cache.py (+228/-4); tests/kernels/test_fused_indexer_q_rope_quant.py (+90/-17)
LABELS: performance, ready, v1, deepseek
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ Currently FP4 quantization logic for `_fused_indexer_q_rope_mxfp4_kernel()` and `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn()` does a linear search, which is inefficient, and more importantly, does not do round-to-nearest-even correctly. This PR switches to inline PTX `cvt.rn.satfinite.e2m1x2.f32` which is cleaner and should be faster too. ⏎  ⏎ ## Test Plan ⏎  ⏎ - Extend `test_fused_indexer_q_rope_quant_matches_unfused` to i …[truncated]

### L3-6d7d4da99e  (L3, 2026-04-29, sha 6d7d4da99e41, PR #41185)
TITLE: [Bugfix] BailingMoeV2.5: rotate full qk_rope_head_dim in MLA RoPE (#41185)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/bailing_moe_linear.py (+8/-2)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎   `BailingMoeV25MLAAttention` builds its RoPE by passing the model's `rope_parameters`                                         ⏎   (which includes `partial_rotary_factor=0.5`) into `get_rope` together with ⏎   `head_size=qk_rope_head_dim=64`. `get_rope` then computes                                                                    ⏎   `rotary_dim = int(64 * 0.5) = 32`, so **only half of the rope head dimensions are ⏎   rotated** in ev …[truncated]

### L3-b92ef9ec5a  (L3, 2026-04-29, sha b92ef9ec5a04, PR #40376)
TITLE: [Perf] Enable FlashInfer top-k/top-p sampler by default (#40376)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: .buildkite/test_areas/samplers.yaml (+3/-1); tests/models/language/generation/test_hybrid.py (+1/-1); tests/v1/sample/test_topk_topp_sampler.py (+301/-0); vllm/envs.py (+5/-3); vllm/v1/sample/ops/topk_topp_sampler.py (+26/-18)
LABELS: ready, ci/build, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ FlashInfer top-k/top-p has better performance compared to Triton's implementation used by default now. ⏎  ⏎ Some time ago FI top-k/top-p sampler implementation was disabled by default in [Disable FlashInfer sampler by default#26859](https://github.com/vllm-project/vllm/pull/26859) because of a number of issues reported here: ⏎ * [[Bug]: illegal memory access when there are multiple concurrent request #23814](https://github.com/vllm-pro …[truncated]

### L3-22524f7a92  (L3, 2026-04-29, sha 22524f7a92b7, PR #39445)
TITLE: [Feat] CPU fp8 attn for AMX/AVX-512 (#39445)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+38/-5); csrc/cpu/cpu_attn.cpp (+78/-25); csrc/cpu/cpu_attn_amx.hpp (+171/-46); csrc/cpu/cpu_attn_fp8.hpp (+214/-0); csrc/cpu/cpu_attn_impl.hpp (+31/-7); csrc/cpu/cpu_attn_neon.hpp (+5/-4); csrc/cpu/cpu_attn_neon_bfmmla.hpp (+2/-1); csrc/cpu/cpu_attn_vec.hpp (+105/-28); csrc/cpu/cpu_attn_vec16.hpp (+3/-3); csrc/cpu/cpu_attn_vxe.hpp (+4/-3); (+9 more)
LABELS: documentation, ready, v1, cpu, verified
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎ Add FP8 E4M3 and E5M2 KV cache quantization support to the CPU attention backend (x86 AVX-512 / Intel AMX paths). Removes the hard error and silent `cache_dtype` downgrade in `CpuPlatform` that previously blocked FP8 KV cache on CPU entirely. ⏎ ### Key design points ⏎  ⏎ - **Two new C++ kernels**: `cpu_attn_reshape_and_cache_fp8` (quantises K/V during cache writes) and `cpu_attention_with_kv_cache_fp8` (dequantises during QK / PV matmuls …[truncated]

### L3-121dbe7a22  (L3, 2026-04-30, sha 121dbe7a221d, PR #39721)
TITLE: [ROCm] ROCm DeepEP API updated to latest (#39721)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm (+15/-8); vllm/distributed/device_communicators/all2all.py (+4/-11); vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py (+23/-41)
LABELS: rocm, ready, ci/build
BODY: Enablement of all the DeepEP API parameters ⏎  ⏎ ## Purpose ⏎ This PR is to match ROCm DeepEP API with the vllm  ⏎  ⏎ co-authored by : @lcskrishna  ⏎  ⏎ The following changes are preformed ⏎  ⏎ Code ⏎ - DeepEP buffer initialization arguments ⏎ - DeepEP dispatch call arguments support for low latency ⏎  ⏎ ROCm Docker test stage ⏎ - Updates to the DeepEP and ROCShmem Commit IDs  ⏎ - Installation of rdma-core v62 required for latest ROCShmem ⏎  ⏎ ## Test Plan ⏎ **STE …[truncated]

### L3-3179e53135  (L3, 2026-04-30, sha 3179e53135db, PR #32553)
TITLE: [P/D] Prefill compute optimizations with bi-directional KV cache transfers between P and D nodes (#32553)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/multi_turn/benchmark_serving_multi_turn.py (+5/-0); examples/online_serving/disaggregated_serving/disagg_proxy_multiturn.py (+562/-0); tests/v1/kv_connector/unit/test_bidirectional_kv_transfer.py (+915/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py (+88/-9)
LABELS: documentation, performance, frontend, ready, v1, kv-connector
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Prefill worker can pull KV cache from remote engines to eliminate redundant prefill computation ⏎      ⏎ Benefits: ⏎     (1) Multi-turn conversations: Prefill worker loads larger KV cache from decode worker ⏎     (2) Cache eviction recovery: Prefill worker retrieves evicted blocks still available on decode worker ⏎  ⏎ ## Purpose ⏎ To avoid redundant compute cycles on Prefill nodes and hence improve prefill throughput. ⏎  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result …[truncated]

### L3-b4806c8ee1  (L3, 2026-04-30, sha b4806c8ee12d, PR #40960)
TITLE: [DSV4] Add BF16 and MXFP8 A2A support for flashinfer a2a one sided (#40960)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/design/moe_kernel_features.md (+1/-1); vllm/distributed/device_communicators/all2all.py (+8/-2); vllm/model_executor/layers/fused_moe/all2all_utils.py (+22/-8); vllm/model_executor/layers/fused_moe/config.py (+4/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py (+17/-36); vllm/model_executor/layers/fused_moe/oracle/mxfp4.py (+12/-5); vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py (+22/-9); vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_two_sided.py (+1/-0); vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py (+1/-0); vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py (+1/-0); (+2 more)
LABELS: documentation, performance, new-model, frontend, speculative-decoding, ready, ci/build, v1, tool-calling, deepseek
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:54f548e9e5 (fix PR 41646)
BODY: ## Purpose ⏎ Originally Flashinfer one sided a2a only supports nvfp4 dispatch. Add BF16 and MXFP8 dispatch.  ⏎  ⏎ ## Test Plan ⏎ gsm8k on V4-Flash ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9583|±  |0.0055| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0 …[truncated]

### L3-9c61864bf8  (L3, 2026-04-30, sha 9c61864bf8a9, PR #41300)
TITLE: [DeepSeek] Use torch.mm for bf16xbf16->fp32 gemm (#41300)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+0/-1); csrc/moe/moe_ops.h (+0/-4); csrc/moe/router_gemm.cu (+0/-52); csrc/moe/torch_bindings.cpp (+0/-4); vllm/_custom_ops.py (+0/-17); vllm/model_executor/layers/deepseek_v4_attention.py (+8/-5); vllm/model_executor/layers/fused_moe/router/gate_linear.py (+1/-1); vllm/model_executor/layers/utils.py (+0/-7)
LABELS: ready, ci/build, deepseek
BODY: It turns out that `torch.mm` natively supports bf16xbf16->fp32 gemm, so we don't need to wrap cublas ourselves. The following script confirms that `nvjet_sm100_tss_16x64_64x16_4x1_v_bz_TNN` is being launched under the hood. ⏎  ⏎ ```python ⏎ import torch ⏎ import torch.nn as nn ⏎  ⏎  ⏎ with torch.inference_mode(): ⏎     import torch.autograd.profiler as profiler ⏎  ⏎     x = torch.randn(8, 1024, dtype=torch.bfloat16, device="cuda") ⏎     l = nn.Linear(1024,  …[truncated]

### L3-4d5c89295b  (L3, 2026-04-30, sha 4d5c89295b76, PR #41363)
TITLE: (bugfix): block_size check for flex attn (#41363)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+5/-0); docs/design/attention_backends.md (+1/-1)
LABELS: bug, documentation, ready, v1
ISSUES: #41339 [Bug]: block_size < 16 silently falls back to FLEX_ATTENTION, then crashes in Triton compilation
BODY: ## Purpose ⏎ Closes #41339 ⏎  ⏎ ## Description ⏎ Previously, when a user set block_size < 16 (e.g., block_size=8), vLLM would trigger a cryptic Triton compilation error during the runtime, which was difficult for users to debug and understand. ⏎  ⏎ ## Changes ⏎ Added block_size compatibility checks for flex attention backend. ⏎  ⏎ Verification ⏎ Now, instead of a Triton crash, users will see a clear and actionable error message: ⏎  ⏎ ``` ⏎ ValueError: No vali …[truncated]

### L3-1adaa5056b  (L3, 2026-04-30, sha 1adaa5056b0e, PR #41341)
TITLE: [ROCm][CI] Add ROCm score absolute tolerance floor (#41341)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py (+17/-3)
LABELS: rocm, ready
BODY: This PR keeps the existing per-backend relative tolerances for the online vision cross-encoder score tests, but adds a small backend-specific absolute tolerance floor for `ROCM_AITER_FA` and `FLEX_ATTENTION`. ⏎  ⏎ The affected failures are low-probability text-vs-text scores on ROCm 7.2/gfx950. Those scores drift by only about 0.005 to 0.006 in absolute terms, but that is enough to exceed the current relative tolerance because the expected value is …[truncated]

### L3-b542bdf7fb  (L3, 2026-04-30, sha b542bdf7fb36, PR #40808)
TITLE: [Bugfix] Disable FlashInfer CUTLASS MoE on SM110 (Jetson Thor AGX) (#40808)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+1/-1)
LABELS: bug, ready, nvidia
BODY: flashinfer <= 0.6.8.post1 ships no SM110 MoE cubins. On SM110 the SM100 artifact is reused at runtime and the TMA-WS dispatcher picks tile `256x256x128` which has no SM100 registration, raising `Unsupported tile shape config 256256128 for MoE gemm` at `moe_gemm_template_dispatch_tma_ws.h:486` when running Nemotron-H unquantized BF16 MoE. Drop SM110 from the `FlashInferExperts` whitelist so the oracle falls back to Triton, analogous to #39825 for  …[truncated]

### L3-6b6ac6c3c7  (L3, 2026-05-01, sha 6b6ac6c3c737, PR #41050)
TITLE: [Kernel][MoE] Support GELU on TRT-LLM NvFP4 fused MoE for Gemma4 (#41050)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_trtllm_nvfp4_moe.py (+207/-0); tests/kernels/moe/utils.py (+2/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+8/-4)
LABELS: ready, nvidia, verified
BODY: ## Summary                                                                                                                                                                                  ⏎                                                                                                                                                                                               ⏎ Enables `MoEActivation.GELU` (gated GELU, GeGLU) on the FlashInfer TRT …[truncated]

### L3-947138b6c2  (L3, 2026-05-01, sha 947138b6c22f, PR #40177)
TITLE: Add nvfp4 kv cache support (#40177)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+114/-15); docs/design/attention_backends.md (+1/-1); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+146/-50); tests/v1/attention/test_trtllm_attention_integration.py (+198/-23); tools/pre_commit/generate_attention_backend_docs.py (+15/-0); vllm/config/vllm.py (+12/-0); vllm/model_executor/layers/quantization/modelopt.py (+7/-7); vllm/v1/kv_cache_interface.py (+10/-0)
LABELS: documentation, ready, ci/build, v1, nvidia
BODY: This is a follow up of https://github.com/vllm-project/vllm/pull/37332 to fully enable nvfp4 kv cache using flashinfer trtllm backend. ⏎ This builds on the existing NVFP4 infrastructure (packed layout, `reshape_and_cache_flash` kernel, `KVQuantMode.NVFP4`) and wires up the read/attention side so that end-to-end NVFP4 KV cache inference works through FlashInfer on Blackwell (SM100). ⏎  ⏎ Key changes: ⏎  ⏎ - **FlashInfer backend**: Add `"nvfp4"` to `sup …[truncated]

### L3-c3868bbbe4  (L3, 2026-05-01, sha c3868bbbe4b1, PR #39505)
TITLE:  [compile] Add FlashInfer FP8 async TP fusion and preserve allreduce fusion ordering #27893   (#39505)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+33/-0); tests/compile/conftest.py (+17/-3); tests/compile/fusions_e2e/conftest.py (+6/-0); tests/compile/fusions_e2e/test_tp2_async_tp.py (+0/-11); vllm/compilation/passes/fusion/collective_fusion.py (+333/-16); vllm/compilation/passes/fusion/sequence_parallelism.py (+9/-1)
LABELS: performance, ready, torch.compile, nvidia, model-bash
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ https://github.com/vllm-project/vllm/issues/27893 ⏎   ⏎ This PR enables AsyncTP fusion for FlashInfer FP8 GEMMs by rewriting: ⏎ - `all_gather + vllm.bmm_fp8` ⏎ - `vllm.bmm_fp8 + reduce_scatter` ⏎  ⏎ into SymmetricMemory-backed fused ops using `bmm_fp8_out`. ⏎  ⏎ ## Test Plan ⏎ https://paste.ubuntu.com/p/MKynCPPj5B/ ⏎ ## Test Result ⏎ <img width="1200" height="264" alt="image" src="https://github.com/user-attachments/assets/11c9e7c0-c947-4843-870 …[truncated]

### L3-4f7bde572a  (L3, 2026-05-01, sha 4f7bde572ad0, PR #41160)
TITLE: [Kernel] Pack output and LSE in DCP A2A (#41160)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/dcp_alltoall.py (+249/-154); tests/distributed/test_dcp_a2a.py (+306/-3)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Optimize the DCP A2A attention backend by packing partial attention output and fp32 LSE into a single collective payload. The packed payload is exchanged with one `all_to_all_single`, then unpacked and combined with the existing exact LSE-weighted reduction semantics. ⏎  ⏎ Temporary send/recv staging buffers use `WorkspaceManager` to avoid repeated hot-path allocations and keep the path compatible with CUDA graph workspace locking. Th …[truncated]

### L3-7075df79b3  (L3, 2026-05-01, sha 7075df79b309, PR #34726)
TITLE: [ROCm] Enable DBO (Dynamic Batch Optimization) on ROCm (#34726)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+12/-3); vllm/v1/worker/gpu_ubatch_wrapper.py (+2/-2)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: This PR enables DBO (`--enable-dbo`) on AMD GPUs by making two minimal changes to `vllm/v1/worker/gpu_ubatch_wrapper.py`: ⏎  ⏎ ## Changes ⏎  ⏎ 1. **Relax the CUDA-only assertion in `SMControlContextManager`** — The HIP runtime exposes CU (Compute Unit) count via `torch.cuda.get_device_properties().multi_processor_count`, so the existing SM control logic works unchanged on ROCm. The assertion is updated to accept both CUDA and ROCm platforms. ⏎  ⏎ 2. **Platfo …[truncated]

### L3-f3fef12350  (L3, 2026-05-01, sha f3fef123504d, PR #32623)
TITLE: [Attention] Abstract the MLA prefill backends and eliminate cuDNN (#32623)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/config/attention.py (+57/-2); vllm/model_executor/layers/attention/mla_attention.py (+58/-595); vllm/platforms/interface.py (+3/-0); vllm/v1/attention/backends/mla/prefill/__init__.py (+11/-0); vllm/v1/attention/backends/mla/prefill/base.py (+125/-0); vllm/v1/attention/backends/mla/prefill/flash_attn.py (+180/-0); vllm/v1/attention/backends/mla/prefill/flashinfer.py (+211/-0); vllm/v1/attention/backends/mla/prefill/registry.py (+53/-0); vllm/v1/attention/backends/mla/prefill/selector.py (+183/-0); vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py (+178/-0); (+6 more)
LABELS: documentation, rocm, ready, v1, nvidia
BODY: ## Purpose ⏎ Abstracts the MLA prefill backends to simplify `mla_attention.py` and introduces a selection mechanism similar to that of the decode backends, via `--attention-config.mla_prefill_backend`. Old `AttentionConfig` arguments (`use_cudnn_prefill`, `use_trtllm_ragged_deepseek_prefill`, and `disable_flashinfer_prefill`) are retained (with deprecation warnings) for backwards compatibility. ⏎  ⏎ Also eliminates cuDNN for simplicity since this ba …[truncated]

### L3-0c99629ede  (L3, 2026-05-01, sha 0c99629ede51, PR #41476)
TITLE: [Build] Make bundled DeepGEMM wheel portable across Python versions (#41476)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: cmake/external_projects/deepgemm.cmake (+23/-4)
LABELS: bug, ready, ci/build
DEEP_STUDY: deep-study: this PR was reverted by PR 41512 (confirmed_revert, reason=build_or_dependency)
BODY: ## Purpose ⏎  ⏎ The bundled `_deep_gemm_C` is built with `WITH_SOABI` only, producing `_C.cpython-312-x86_64-linux-gnu.so` (since the wheel-build container is Python 3.12). That `.so` won't load on any other CPython, so users on Python 3.10/3.11/3.13 install vLLM and silently fall through to `DeepGEMM backend is not available` in `vllm/utils/deep_gemm.py`, unless they manually run `tools/install_deepgemm.sh`. ⏎  ⏎ Mirrors the pattern used in `cmake/e …[truncated]

### L3-c3e64696cd  (L3, 2026-05-01, sha c3e64696cdea, PR #41375)
TITLE: [Perf] Warmup forward_native sampler kernel (#41375)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+20/-0)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ The default-on FlashInfer top-k/top-p sampler from #40376 exposed a warmup gap in `_dummy_sampler_run`: the existing dummy call uses `generators={}`, which under the new default routes through `flashinfer_sample` and leaves the `forward_native` Triton fallback path JIT-cold. That fallback is taken at runtime whenever a request has a seed (`SamplingMetadata.generators` non-empty), so the first such request pays ~1s of JIT cost on its …[truncated]

### L3-5737770c6c  (L3, 2026-05-01, sha 5737770c6c34, PR #41458)
TITLE: Re-enable allreduce rms fusion for DP / PP (#41458)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+0/-6)
LABELS: ready
ISSUES: #34458 [Bug]: AR+rms broken for TP=2 DP=2 | #35426 [Bug]: AllReduceRMSFusionPass crashes with PP
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Re-enable allreduce + rms fusion for PP & DP. ⏎  ⏎ Previous issues: ⏎ - PP: https://github.com/vllm-project/vllm/issues/35426 ⏎ - DP: https://github.com/vllm-project/vllm/issues/34458 ⏎  ⏎ Both issues have been fixed by https://github.com/flashinfer-ai/flashinfer/pull/2662 that was included in flashinfer 0.6.7. We are currently on 0.6.8 so this can be re-enabled.  ⏎  ⏎ ## Test Plan ⏎  ⏎ Tested manually the reproducers. Turning the config on b …[truncated]

### L3-bc635fad23  (L3, 2026-05-01, sha bc635fad2389, PR #41217)
TITLE: [ROCm][Deepseek] dsv3.2 further optimization (#41217)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.rocm_aiter, L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/indexer.py (+1/-1); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+4/-0); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+227/-29); vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+22/-19); docs/design/attention_backends.md (+1/-1); vllm/model_executor/models/deepseek_v2.py (+38/-23)
LABELS: documentation, rocm, ready, v1, deepseek
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ This PR will replace https://github.com/vllm-project/vllm/pull/32649.  ⏎ The main optimization include the following part: ⏎ - shuffle indexer cache layout for kv cache update and fetch  ⏎ - rocm gluon version of ·paged_mqa_logits` integration ⏎ - build ragged metadata for mla at metadata_builder instead of runtime ⏎ - fp8 sparse mla support  ⏎  ⏎ please help to review cc @tjtanaa  ⏎ ## Test Plan ⏎ gsm8k score: ⏎ ``` ⏎ |Tasks|Version|     Filter …[truncated]

### L3-529c671e80  (L3, 2026-05-01, sha 529c671e8075, PR #37646)
TITLE: [ROCm][FEAT] AITER Fused Allreduce + RMSNorm (#37646)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+1/-0); tests/compile/fusions_e2e/test_tp2_ar_rms.py (+19/-3); tests/compile/passes/distributed/test_fusion_all_reduce.py (+77/-12); vllm/_aiter_ops.py (+115/-0); vllm/compilation/passes/fusion/act_quant_fusion.py (+1/-0); vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+209/-1); vllm/compilation/passes/pass_manager.py (+7/-1); vllm/config/vllm.py (+16/-2); vllm/distributed/parallel_state.py (+9/-1)
LABELS: rocm, ready, ci/build, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ This PR integrates AITER fused all-reduce + RMSNorm kernel for ROCm platforms,  ⏎  ⏎ introduces: ⏎ -  `RocmAiterAllReduceFusionPass` that fuses the all-reduce collective with the subsequent RMSNorm (and fused-add-RMSNorm)  ⏎  ⏎ AITER's `v0.1.10.post3` version exposes two new function `custom_fused_ar_rms` and `custom_fused_ar_rms_quant` in `CustomAllReduce` class. In this PR we are only utilizing `custom_fused_ar_rms` from AITER's Custom …[truncated]

### L3-c51df43005  (L3, 2026-05-03, sha c51df4300572, PR #41524)
TITLE: Disable flashinfer autotune temporarily due to correctness issues (#41524)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+6/-2)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎ We have observed correctness bugs with flashinfer autotuning, as seen in issue https://github.com/flashinfer-ai/flashinfer/issues/3197. Kernel-level reproduction is available. ⏎  ⏎ While this is pending for fix, this PR disables flashinfer autotuning by default for now for O1 and O2 optimization levels. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-db9a84e0cd  (L3, 2026-05-03, sha db9a84e0cd0e, PR #41424)
TITLE: [Bugfix] Fix FP8 Bias Loading (#41424)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/model_loader/test_reload.py (+28/-0); vllm/model_executor/model_loader/reload/meta.py (+1/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎ Fixes the underlying cause of https://github.com/vllm-project/vllm/issues/41284 ⏎  ⏎ The issue is that when layers have `bias=True`, we do the following: ⏎ - Initialize the weights on meta device + wrap its own weight loader, allocate the bias (not on meta device) ⏎ - The params are generally yielded alphabetically when we are loading them. This means: ⏎      - First, we load the bias normally ⏎      - Then, we load the weight, which needs  …[truncated]

### L3-66dfee7121  (L3, 2026-05-03, sha 66dfee7121df, PR #40737)
TITLE: [Bugfix] Fix degenerate KV cache stride causing TMA cudaErrorIllegalInstruction (#40737)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flash_attn.py (+23/-1); vllm/v1/attention/backends/flash_attn_diffkv.py (+24/-1); vllm/v1/attention/backends/flashinfer.py (+31/-19); tests/v1/attention/test_kv_head_stride_canonicalization.py (+162/-0); vllm/utils/cpu_resource_utils.py (+1/-1); vllm/utils/torch_utils.py (+26/-0)
LABELS: bug, ready, v1, cpu, nvidia, verified
BODY: ## Purpose ⏎  ⏎ Fix cudaErrorIllegalInstruction → NCCL allgather hang → EngineDeadError that occurs when num_kv_heads_per_rank == 1 (e.g. Qwen3.5-397B with --tensor-parallel-size 8) and prefix caching is enabled. ⏎  ⏎ Root cause: When prefix-cached KV blocks are freed and reallocated, PyTorch can produce tensor views with a degenerate stride on the singleton num_kv_heads dimension. is_contiguous() returns True for any stride on a size-1 dimension, so …[truncated]

### L3-e724b0ea8d  (L3, 2026-05-04, sha e724b0ea8d3b, PR #41386)
TITLE: [ROCm] ROCm7.2.2 + profiler fix + AITER 0.1.12.post2 (#41386)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/release-pipeline.yaml (+1/-1); docker/Dockerfile.rocm_base (+24/-4)
LABELS: rocm, ready, ci/build
BODY: A combination of base libraries that works ⏎  ⏎ Including the profiler fix through rocm runtime

### L3-1cb0838721  (L3, 2026-05-04, sha 1cb08387214f, PR #41569)
TITLE: [ROCm][CI] Fix MLA prefill scale for DeepSeek GSM8K (#41569)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+33/-1); tests/v1/attention/test_mla_backends.py (+5/-3); tests/v1/attention/test_mla_prefill_selector.py (+61/-0)
LABELS: rocm, ready, v1, deepseek
BODY: This restores the MLA prefill softmax scale used by DeepSeek-style models. The prefill backend refactor in `f3fef12350` (#32623) started constructing MLA prefill backends with `model_config.get_head_size() ** -0.5`. For DeepSeek-Coder-V2-Lite, that head size is the 576-dim latent KV cache, not the 192-dim query/key attention head. It also missed the DeepSeek YaRN mscale correction applied by the model attention module. ⏎  ⏎ The fix derives prefill  …[truncated]

### L3-be5983b874  (L3, 2026-05-04, sha be5983b874bd, PR #41643)
TITLE: [Docs] Add non-causal support to attention backend docs (#41643)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: docs/design/attention_backends.md (+30/-29); tools/pre_commit/generate_attention_backend_docs.py (+9/-0)
LABELS: documentation, ready
BODY: ## Purpose ⏎ Given the popularity of DFlash, we should report non-causal support in the attention backend documentation. This PR adds this column.  ⏎  ⏎ ## Test Plan ⏎ Docs build should succeed (docs-only PR) ⏎  ⏎ ## Test Result ⏎ TBD ⏎  ⏎ --- ⏎ [details omitted]

### L3-420b0a5c95  (L3, 2026-05-04, sha 420b0a5c9518, PR #40451)
TITLE: [Hardware][Power]Add Power VSX Attention Backend and fix l2 Cache Crash (#40451)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+12/-3); csrc/cpu/cpu_attn.cpp (+4/-0); csrc/cpu/cpu_attn_impl.hpp (+4/-1); csrc/cpu/cpu_attn_vec.hpp (+2/-2); csrc/cpu/cpu_attn_vsx.hpp (+359/-0); csrc/cpu/cpu_types_vsx.hpp (+4/-0); csrc/cpu/generate_cpu_attn_dispatch.py (+13/-2); csrc/cpu/utils.hpp (+1/-1)
LABELS: ready, v1, cpu
BODY: ## Purpose ⏎ This PR adds native PowerPC (ppc64le) VSX support for the vLLM CPU backend and resolves a initialization crash caused by `IndexError: unordered_map::at`. ⏎  ⏎ ## Test Plan ⏎ ```bash ⏎ vllm serve ibm-granite/granite-3.3-8b-instruct --port 8000 --max-model-len 512 --max-num-batched-tokens 8192 --dtype bfloat16  ⏎ ``` ⏎  ⏎ ## Test Result ⏎ ```bash ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     100   …[truncated]

### L3-3e1ad4435f  (L3, 2026-05-05, sha 3e1ad4435f7c, PR #41288)
TITLE: [Bug] Fix `tests/compile/test_config.py` AttributeError: 'NoneType' object has no attribute 'dtype' (#41288)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+1/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ `pytest tests/compile/test_config.py::test_sequence_parallelism_requires_full_graph_compilation[PIECEWISE-False-True-FULL-False-expected_capture_sizes0-4] -xvs` ⏎  ⏎ Originally ⏎  ⏎ ```bash ⏎ ============================================= FAILURES ============================================= ⏎ _ test_sequence_parallelism_requires_full_graph_compilation[PIECEWISE-False-True-FULL-False-expected_capture_sizes0-4] _ ⏎  ⏎ cudagraph_mode = <CUDAG …[truncated]

### L3-e1e4646b06  (L3, 2026-05-05, sha e1e4646b06f2, PR #41162)
TITLE: [Model Runner V2] Rebuild attn metadata between draft decode steps (#41162)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/sample/gumbel.py (+18/-4); vllm/v1/worker/gpu/spec_decode/eagle/speculator.py (+176/-115); vllm/v1/worker/gpu/spec_decode/probabilistic_rejection_sampler_utils.py (+4/-2)
LABELS: ready, v1, mrv2
BODY: ## Context ⏎ Investigating DSV4 "invalid memory access" crash that happens when MTP > 2. I believe that crash is caused by not rebuilding the attention metadata in between draft decode steps. DSV4's attention metadata builders have position-dependent state that must be updated whenever the position is advanced. For example: https://github.com/vllm-project/vllm/blob/e9f8f31e9a4c31d6842ca1adffe2619ed204fafb/vllm/v1/attention/backends/mla/sparse_swa. …[truncated]

### L3-685bf811d6  (L3, 2026-05-05, sha 685bf811d65b, PR #37481)
TITLE: [XPU] enable is_act_and_mul for xpu (#37481)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/xpu_moe.py (+2/-1); vllm/model_executor/layers/fused_moe/layer.py (+4/-2)
LABELS: intel-gpu, ready
BODY: ## Purpose ⏎  ⏎ Testing `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-bf16` on XPU and enable `relu2_no_mul` ⏎  ⏎ dependencies: ⏎ 1. https://github.com/vllm-project/vllm-xpu-kernels/pull/232 -> Add 'relu2_no_mul' kernel ⏎  ⏎ ## Test Plan ⏎  ⏎ ``` ⏎ lm_eval   \ ⏎ --model vllm   \ ⏎ --model_args pretrained=/mnt/data/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-bf16,tensor_parallel_size=4,block_size=16,trust_remote_code=True,enable_expert_parallel=True,attention_backend=TRIT …[truncated]

### L3-416f9cdede  (L3, 2026-05-05, sha 416f9cdede96, PR #41433)
TITLE: [Perf][2/n] Eliminate GPU<->CPU syncs in pooling code (#41433)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/pooler/seqwise/methods.py (+10/-8); vllm/model_executor/layers/pooler/special.py (+34/-4); vllm/model_executor/layers/pooler/tokwise/methods.py (+14/-15)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: Second batch of unnecessary gpu/cpu syncs, found via https://github.com/vllm-project/vllm/pull/40561.

### L3-628c436301  (L3, 2026-05-05, sha 628c43630155, PR #40871)
TITLE: [New Model][ROCm] Add AMD support for DeepSeek V4 (#40871)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.mla.rocm_aiter_sparse, L3.platform.rocm_selection
FILES: vllm/v1/attention/backends/mla/sparse_swa.py (+2/-1); vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py (+3/-1); vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+528/-60); CMakeLists.txt (+6/-6); csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu (+36/-2); csrc/moe/topk_softplus_sqrt_kernels.cu (+32/-21); csrc/moe/torch_bindings.cpp (+1/-2); csrc/torch_bindings.cpp (+0/-2); requirements/rocm.txt (+3/-0); tests/kernels/moe/test_topk_softplus_sqrt.py (+4/-2); (+12 more)
LABELS: documentation, performance, new-model, rocm, speculative-decoding, ready, ci/build, v1, tool-calling, deepseek
BODY: ## Purpose ⏎ This PR adds support of DeepSeek V4 for AMD. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎ docker image: docker pull rocm/vllm-dev:deepseek-v4-mi35x ⏎ machine: mi355x ⏎ environment setting: ⏎ ``` ⏎ # enter docker, do: ⏎ pip uninstall vllm ⏎ git clone https://github.com/vllm-project/vllm.git ⏎ cd vllm ⏎ git fetch origin pull/40871/head:pr_dsv4 ⏎ git checkout pr_dsv4 ⏎ python3 setup.py develop ⏎ ``` ⏎ ### Deepseek-V4-Flash ⏎ Launch command: ⏎ ``` ⏎ max_num_seqs= …[truncated]

### L3-8b9ea2f881  (L3, 2026-05-05, sha 8b9ea2f881ea, PR #40137)
TITLE: [Feature] Add Triton kernel JIT compilation monitor for inference (#40137)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/engine.yaml (+2/-1); tests/test_jit_monitor.py (+240/-0); vllm/triton_utils/jit_monitor.py (+113/-0); vllm/v1/worker/gpu_worker.py (+8/-0)
LABELS: ready, ci/build, v1
BODY: ### Purpose ⏎  ⏎ Enables Triton JIT compilation and autotuning warnings during inference by default. After warmup completes, any such event is logged as a WARNING, letting developers quickly spot warmup/inference path divergences that cause latency spikes. ⏎  ⏎ Recent cases like #37338 and #39169 were found only after time-consuming investigation of 1st-vs-2nd benchmark performance gaps. This monitor makes such issues immediately visible in server lo …[truncated]

### L3-b786ec8e74  (L3, 2026-05-05, sha b786ec8e744b, PR #38099)
TITLE: [Bugfix] Suggest upgrading Transformers for tokenizer class errors (#38099)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/tokenizers/hf.py (+4/-1)
LABELS: bug, ready
ISSUES: #38024 Tokenizer error with Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated model - TokenizersBackend not found
BODY: ## Summary ⏎  ⏎ Fixes https://github.com/vllm-project/vllm/issues/38024 ⏎  ⏎ When `AutoTokenizer.from_pretrained()` fails because the installed Transformers version does not recognize a tokenizer class, vLLM currently tells users to try `trust_remote_code`. For models produced with newer Transformers versions, such as tokenizer configs containing `"tokenizer_class": "TokenizersBackend"`, `trust_remote_code` does not resolve the issue because the missing  …[truncated]

### L3-2228fe6868  (L3, 2026-05-05, sha 2228fe68687d, PR #40815)
TITLE: [Attention] Move FA3→FA4 upgrade into get_flash_attn_version() (#40815)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fa_utils
FILES: vllm/v1/attention/backends/fa_utils.py (+18/-11); vllm/v1/attention/backends/flash_attn.py (+0/-8)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Follow https://github.com/vllm-project/vllm/pull/40045#discussion_r3132033201. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-2d7d6cf765  (L3, 2026-05-05, sha 2d7d6cf765d0, PR #41752)
TITLE: [Spec Decode] Allow multimodal models with a warning (#41752)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/spec_decode/dflash.py (+1/-2); vllm/v1/spec_decode/llm_base_proposer.py (+5/-4)
LABELS: speculative-decoding, ready, v1
BODY: ## Purpose ⏎ Allow server to start up when speculative decoding with parallel drafting is enabled for multimodal models like Gemma-4  ⏎  ⏎ Previously, `_raise_if_multimodal()` unconditionally blocked parallel drafting for any model registered as multimodal, even when only text inference was needed. This change downgrades the hard error to a warning (`_warn_if_multimodal()`), enabling models like `google/gemma-4-31B-it` to use EAGLE3 with parallel dr …[truncated]

### L3-01b9b5af67  (L3, 2026-05-05, sha 01b9b5af67b7, PR #41744)
TITLE: [Attention] Minor refactor: layer takes ownership of the MLA prefill backend (#41744)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+24/-52); vllm/v1/attention/backends/mla/prefill/base.py (+0/-4); vllm/v1/attention/backends/mla/prefill/flash_attn.py (+0/-4); vllm/v1/attention/backends/mla/prefill/flashinfer.py (+34/-23); vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py (+0/-4); tests/v1/attention/test_mla_backends.py (+16/-2); tests/v1/attention/test_mla_prefill_selector.py (+0/-61)
LABELS: ready, v1, nvidia
BODY: ## Purpose ⏎ Moves the ownership of the MLA prefill backend from the `MLACommonMetadataBuilder` to the `MLAAttention` layer. This allows the removal of `get_mla_prefill_scale` so that the layer's scale is the single source of truth. The `MLACommonMetadataBuilder` does still hold a reference to the prefill backend, though, because it needs to include it in the metadata for use in the impl. This will be resolved in the upcoming broader attention ref …[truncated]

### L3-38e16678ba  (L3, 2026-05-06, sha 38e16678ba7e, PR #39324)
TITLE: [Bugfix] Align block table for TRTLLM MLA edge-case (#39324)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/block_table.py (+7/-0); vllm/v1/worker/gpu/block_table.py (+5/-0)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ When running with unusual max-model-len, such as `9416`, TRTLLM MLA crashes: ⏎ ``` ⏎ (Worker_TP0 pid=646061) ERROR 04-07 23:47:19 [multiproc_executor.py:932] ValueError: Expected block_num % (128 / block_size) == 0, got block_num=295 and block_size=32` ⏎ ``` ⏎  ⏎ It seems like a straightforward fix to pad the block table's max_num_blocks by a block or two to avoid this case.  ⏎  ⏎ ## Testing ⏎  ⏎ Ran Kimi K2.5 with the max-model-len and it w …[truncated]

### L3-22a3cbe152  (L3, 2026-05-06, sha 22a3cbe1520b, PR #38296)
TITLE: [ROCm] aiter_unified_attn fp8 q scale refactor (#38296)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+3/-21)
LABELS: rocm, ready, v1
BODY: (To be merged after https://github.com/ROCm/aiter/pull/2360 - aiter PR cleans up fp8 attention & adds native support for q scale) ⏎  ⏎ The absorption logic for fp8xfp8 attention computation introduced in https://github.com/vllm-project/vllm/pull/36927 can now be natively handled by the aiter kernel. ⏎  ⏎ 1. The aiter kernel now has support for q_scale ⏎ 2. We would no longer need to absorb ` layer._q_scale_float * layer._k_scale_float in softmax_scale …[truncated]

### L3-80d5e7d103  (L3, 2026-05-06, sha 80d5e7d103ee, PR #41665)
TITLE: [Bugfix] Fix condition to clear persistent topk so that it can be captured regardless (#41665)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/topk.cu (+17/-8)
LABELS: bug, ready, deepseek
ISSUES: #41483 [Bug]: h200 deepseekv4 pro mtp
DEEP_STUDY: deep-study correctness case vllm:80d5e7d103: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=#41444
BODY: ## Purpose ⏎ External memset is introduced in #41444 but only when `need_cooperative` is true. `need_cooperative` only trigger when `max_seq_len` is greater than a certain threshold. And this means that the trigger will never fire at cuda graph capture time.  ⏎  ⏎ This PR remove `need_cooperative` statement so that the memset kernel is always trigger at capture time.  ⏎  ⏎ ## Test Plan ⏎ gsm8k v4-pro DEP4 on B300 MTP2 ⏎  ⏎ ## Test Result ⏎ ``` ⏎ |Tasks|Ver …[truncated]

### L3-20cac26b19  (L3, 2026-05-06, sha 20cac26b197a, PR #40549)
TITLE: [ROCm] Enable SimpleCPUOffloadConnector on ROCm (#40549)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+27/-9); tests/v1/simple_kv_offload/test_integration.py (+2/-2); tests/v1/simple_kv_offload/test_scheduler.py (+7/-1); vllm/v1/simple_kv_offload/cuda_mem_ops.py (+54/-9)
LABELS: rocm, ready, v1, nvidia
ISSUES: #40397 [Feature]: Add ROCm support for simple offload connector
BODY: ## Purpose ⏎ Fix https://github.com/vllm-project/vllm/issues/40397 ⏎  ⏎ Enable `SimpleCPUOffloadConnector`  on ROCm backend. ⏎ Also enabled the related tests on ROCm. ⏎  ⏎ ## Test Plan ⏎ Using the upstream nightly docker as the base (where rocm is v7.2.1) and build vllm from source. ⏎ Tested on MI350 ⏎  ⏎ (1) unit test: ⏎ ``` ⏎ vllm/tests/v1/simple_kv_offload# pytest- ⏎ ``` ⏎ (2) integration test: ⏎ ``` ⏎ vllm/tests/v1/simple_kv_offload# pytest test_integration. …[truncated]

### L3-27e0057aed  (L3, 2026-05-06, sha 27e0057aeda6, PR #41745)
TITLE: [Spec Decode] Add Gemma4 MTP speculative decoding support (#41745)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/registry.py (+6/-0); tests/v1/e2e/spec_decode/test_spec_decode.py (+35/-12); vllm/config/speculative.py (+20/-0); vllm/model_executor/models/gemma4_mtp.py (+603/-0); vllm/model_executor/models/registry.py (+1/-0); vllm/transformers_utils/model_arch_config_convertor.py (+13/-0); vllm/v1/spec_decode/gemma4.py (+335/-0); vllm/v1/spec_decode/llm_base_proposer.py (+84/-54); vllm/v1/worker/gpu_model_runner.py (+24/-6)
LABELS: new-model, speculative-decoding, ready, v1
BODY: ## Purpose ⏎  ⏎ Add Multi-Token Prediction (MTP) speculative decoding support for Gemma 4 assistant models. This enables draft-based speculative decoding for all Gemma 4 model variants (E2B, E4B, 26B-A4B, 31B) using their corresponding assistant checkpoints. ⏎  ⏎ The Gemma 4 assistant is a lightweight decoder with Q-only attention layers that share KV cache with the target model. It supports an optional centroids masking optimization (enabled automat …[truncated]

### L3-d4b0048404  (L3, 2026-05-07, sha d4b00484040c, PR #41713)
TITLE: Eliminate redundant MoE buffer copies in AITER fused experts (without dependency on AITER changes) (#41713)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/modular_kernel.py (+19/-0); vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py (+14/-1); vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py (+4/-0)
LABELS: performance, rocm, ready, deepseek
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: Addressing the issues raised in previous PRs.  ⏎ It follows the draft PR on this topic here: https://github.com/vllm-project/vllm/pull/41020  ⏎ We copied ideas from other works, mostly based on https://github.com/vllm-project/vllm/pull/38597 and similar ideas in https://github.com/ROCm/aiter/commit/da318d0c10038cff73d5a6c38c471b9601fccaeb and https://github.com/vllm-project/vllm/pull/41020.  ⏎  ⏎  ⏎ ## Purpose ⏎  ⏎ Eliminate redundant __amd_rocclr_copyB …[truncated]

### L3-003159d98b  (L3, 2026-05-07, sha 003159d98b25, PR #41534)
TITLE: [ROCm][CI] Avoid duplicate ROCm AITER norm-quant patterns (#41534)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/compilation/passes/fusion/rocm_aiter_fusion.py (+12/-1)
LABELS: rocm, ready
BODY: This fixes a ROCm AITER compile-time failure where Inductor rejects duplicate RMSNorm plus dynamic FP8 quant fusion patterns. ⏎  ⏎ Test group fixed: ⏎ - `v1/e2e/spec_decode`: `test_eagle_correctness_heavy[ROCM_AITER_FA-llama3_eagle]` ⏎  ⏎ Details: ⏎ - The failing reference LLM enables ROCm AITER norm-quant fusion without enabling the `quant_fp8` custom op. ⏎ - In that mode, both the AITER and native dynamic quant matchers trace through `QuantFP8`'s nati …[truncated]

### L3-8189a15914  (L3, 2026-05-07, sha 8189a15914ca, PR #39917)
TITLE: [Core] Replace routing replay with device cache and async D2H pipeline (#39917)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/training/routed_experts_replay.md (+289/-0); tests/model_executor/test_routed_experts_capture.py (+132/-215); vllm/config/vllm.py (+51/-0); vllm/entrypoints/openai/chat_completion/protocol.py (+4/-0); vllm/entrypoints/openai/chat_completion/serving.py (+15/-0); vllm/entrypoints/openai/completion/protocol.py (+4/-0); vllm/entrypoints/openai/completion/serving.py (+11/-0); vllm/model_executor/layers/fused_moe/layer.py (+7/-0); vllm/model_executor/layers/fused_moe/routed_experts_capturer.py (+760/-298); vllm/model_executor/layers/fused_moe/runner/moe_runner.py (+8/-0); (+5 more)
LABELS: documentation, frontend, ready, v1, nvidia
DEEP_STUDY: deep-study: this PR was reverted by PR 42434 (confirmed_revert, reason=premature_or_process) || deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ Replace upstream vLLM routing replay with a device-cache approach that works correctly with CUDA graphs, multi-node TP, and data parallelism. This PR focuses on the core architecture change — monolithic kernel support and prefix caching are in a follow-up PR. ⏎  ⏎ RFC: #39701 ⏎  ⏎ ## What this PR does ⏎  ⏎ Replaces the SharedMemory-based routing replay with: ⏎ - Pre-allocated `(L, N, K)` int16 device buffer with per-layer views ⏎ - Async D2H pipeline …[truncated]

### L3-51f22dcfd0  (L3, 2026-05-07, sha 51f22dcfd068, PR #41025)
TITLE: [Feat][CPU] Enable Gated DeltaNet Attention (Qwen 3.5 / 3.6) (#41025)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+5/-0); .buildkite/scripts/hardware_ci/run-cpu-test-arm.sh (+15/-0); docs/design/attention_backends.md (+1/-1); vllm/model_executor/layers/linear.py (+3/-0); vllm/model_executor/layers/mamba/gdn_linear_attn.py (+60/-3); vllm/model_executor/layers/mamba/ops/cpu/__init__.py (+0/-0); vllm/model_executor/layers/mamba/ops/cpu/causal_conv1d.py (+88/-0); vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py (+208/-0); vllm/model_executor/layers/mamba/ops/cpu/recurrent_gated_delta_rule.py (+223/-0); vllm/platforms/cpu.py (+10/-3); (+1 more)
LABELS: documentation, ready, ci/build, v1, qwen, cpu
ISSUES: #35950 [Bug]: ValueError: too many values to unpack (expected 2)
BODY: ## Purpose ⏎  ⏎ Enables Qwen 3.5 and Qwen 3.6 on all CPUs. ⏎  ⏎ - Initial enablement of Gated Delta-Net Attentoin for CPU Backend ⏎ - Will follow up with future PRs to sequeeze perf out of this with  fusion / varlen kernels ⏎  ⏎ Fixes: https://github.com/vllm-project/vllm/issues/35950 ⏎  ⏎  ⏎ ## torch.compile gotcha ⏎  ⏎  Qwen 3.5 / 3.6 currently fail in compile mode (i.e. the default) due to https://github.com/vllm-project/vllm/issues/40972 whose root-cause …[truncated]

### L3-3af561ec0a  (L3, 2026-05-07, sha 3af561ec0a00, PR #41972)
TITLE: [ROCm] Fix AITER AR+RMSNorm no-residual fusion (#41972)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+1/-1)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ Fix the ROCm AITER allreduce + RMSNorm fusion for the no-residual pattern. ⏎  ⏎ `AiterAllreduceFusedRMSNormPattern` replaces an allreduce followed by RMSNorm without a residual input. However, the AITER fused kernel computes RMSNorm over `allreduce(input) + residual`, so the synthetic residual for this pattern must be zero. ⏎  ⏎ The AITER replacement used `torch.empty_like(input)`, which can add uninitialized memory into the layer outpu …[truncated]

### L3-c936548ce6  (L3, 2026-05-07, sha c936548ce6b0, PR #41835)
TITLE: [ROCm][DeepSeek] Enable V3.2 TP4 AITER MLA (#41835)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/model_executor/models/deepseek_v2.py (+11/-9); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+1/-1)
LABELS: rocm, ready, v1, deepseek
BODY: ## Purpose ⏎  ⏎ This PR enables the remaining DeepSeek-V3.2 TP4-specific pieces needed for ROCm AITER MLA serving. ⏎  ⏎ DeepSeek-V3.2 TP4 reaches local MLA `num_heads=32`. Current ROCm AITER MLA rejects that head count through `_AITER_UNSUPPORTED_HEADS = [32]`, so the model can fail before reaching the HIP graph accuracy path with `unsupported head_num: 32`. This PR removes that stale block now that the AITER-side MLA kernel support for this shape ex …[truncated]

### L3-cd58e30872  (L3, 2026-05-07, sha cd58e30872c2, PR #41681)
TITLE: [Perf] Use numpy zero-copy path for embedding float response serialization (#41681)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/entrypoints/pooling/test_utils.py (+59/-0); vllm/entrypoints/pooling/embed/serving.py (+35/-0); vllm/entrypoints/pooling/utils.py (+11/-0)
LABELS: frontend, ready, verified
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ When `encoding_format=float` and `ORJSONResponse` is active, the `/v1/embeddings` response builder currently calls `.tolist()` on each embedding tensor, converting every float individually through Python. This PR replaces that with a zero-copy `.numpy()` view that ORJSON serializes natively. ⏎  ⏎ **Scope:** Only the float-format OpenAI embedding response path is changed. Base64, bytes, Cohere, non-ORJSON fallback, model execution, schedul …[truncated]

### L3-989c176c0a  (L3, 2026-05-07, sha 989c176c0a14, PR #41434)
TITLE: [Perf][3/n] Eliminate GPU<->CPU syncs in attention impls (#41434)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.dispatch.abstract_interface, L3.flex_attention, L3.tree_attention
FILES: vllm/v1/attention/backends/flashinfer.py (+2/-1); vllm/v1/attention/backends/flex_attention.py (+30/-18); vllm/v1/attention/backends/tree_attn.py (+52/-6); vllm/v1/attention/backends/triton_attn.py (+4/-5); vllm/v1/attention/backends/turboquant_attn.py (+37/-9); vllm/v1/attention/backends/utils.py (+15/-4); vllm/utils/torch_utils.py (+9/-3); vllm/v1/attention/backends/mamba_attn.py (+18/-19); vllm/v1/worker/gpu/buffer_utils.py (+1/-1); vllm/v1/worker/gpu/sample/penalties.py (+1/-2)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Unnecessary gpu/cpu syncs in attention implementations, found via https://github.com/vllm-project/vllm/pull/40561. ⏎  ⏎ ### TurboQuant benchmark ⏎  ⏎ Each scenario runs vLLM with `--tensor-parallel-size 1 --distributed-executor-backend uni` (UniProcExecutor) on a single NVIDIA GB200 GPU. Model: `Qwen/Qwen3-0.6B`. Each side (without / with change) is the mean ± population std across **3 timed runs** sharing one server process; each run uses its own se …[truncated]

### L3-baf068d8be  (L3, 2026-05-07, sha baf068d8be09, PR #41990)
TITLE: enable persistent mla for sparse mla backend (#41990)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter, L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+0/-3); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+120/-2)
LABELS: rocm, ready, v1
BODY: After commit https://github.com/vllm-project/vllm/commit/628c43630 Deepseek 3.2 TP4 started suffering from Memory Access Faults on ROCm based runs. ⏎  ⏎ This Bugfix is to properly implement the metadata allocation when using the sparse mla attention backend on ROCm systems. ⏎  ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-ed582b6a4c  (L3, 2026-05-07, sha ed582b6a4cae, PR #40711)
TITLE: [Aiter][ROCm] gdn_linear_attn kernel fusion (#40711)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/_aiter_ops.py (+19/-0); vllm/model_executor/layers/fla/ops/chunk.py (+11/-0); vllm/model_executor/layers/fla/ops/chunk_o.py (+8/-1); vllm/model_executor/layers/mamba/gdn_linear_attn.py (+410/-51)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Overview: ⏎ This change merges various triton compiled kernels into single kernels. For all backends, triton kernels are merged into one. For AITER, the triton kernels are further merged into optimized kernels for AMD. ⏎  ⏎ Fusions remove 20-26us in launch overhead per layer (36 GDN layers and 12 attention layers per decode step) which corresponds to 5-8% improvements in TPOT and throughput depending on workload size and input/output size rations. ⏎  …[truncated]

### L3-2c6b59b807  (L3, 2026-05-08, sha 2c6b59b80771, PR #39280)
TITLE: [ROCm][Perf] Add Fused Shared Expert (FSE) support for Qwen3-Next (#39280)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/_aiter_ops.py (+69/-2); vllm/model_executor/layers/fused_moe/layer.py (+4/-0); vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py (+49/-0); vllm/model_executor/layers/fused_moe/router/aiter_shared_routed_fused_moe_router.py (+143/-0); vllm/model_executor/layers/fused_moe/router/router_factory.py (+25/-2); vllm/model_executor/layers/fused_moe/runner/moe_runner.py (+29/-1); vllm/model_executor/models/qwen3_next.py (+26/-4); vllm/model_executor/models/qwen3_next_mtp.py (+14/-1)
LABELS: rocm, ready, qwen
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Fuse shared expert into the AITER MoE kernel as an extra expert slot when `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`, eliminating the separate shared expert MLP forward pass and greatly improving decode throughput. ⏎  ⏎ The router gate `[num_experts, hidden]` and shared expert gate `[num_shared, hidden]` weight matrices are fused into a single `[num_experts + num_shared, hidden]` matrix at init. One `F.linear` call produces combine …[truncated]

### L3-8bcd8a260c  (L3, 2026-05-08, sha 8bcd8a260cd1, PR #42089)
TITLE: [Bugfix] Fix FlashInfer CUTLASS MXFP4-MXFP8 MoE by restoring swizzled scale (#42089)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_trtllm_nvfp4_moe.py (+1/-1); tests/kernels/moe/utils.py (+2/-2); vllm/model_executor/layers/fused_moe/config.py (+10/-6); vllm/model_executor/layers/fused_moe/oracle/fp8.py (+1/-1); vllm/model_executor/layers/fused_moe/oracle/mxfp4.py (+4/-0); vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (+1/-1); vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py (+1/-1); vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py (+2/-5); vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_two_sided.py (+2/-2); vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py (+2/-2); (+2 more)
LABELS: bug, ready, nvidia
ISSUES: #42093 [CI Failure]: GPQA Eval (GPT-OSS) (B200) - gpt-oss-20b-flashinfer-mxfp4-mxfp8-cutlass
BODY: ## Purpose ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/42093 ⏎  ⏎ #40960 hardcoded `is_sf_swizzled_layout=False` in the mxfp8 branch of `moe_kernel_quantize_input`, which broke the FlashInfer CUTLASS MXFP4-MXFP8 path. The CUTLASS kernel requires swizzled scales, so the model produced garbage output and never terminated, manifesting as the `GPQA Eval (GPT-OSS) (B200)` nightly failing for several days. ⏎  ⏎ This PR restores the parameter as a pa …[truncated]

### L3-b1728c1e66  (L3, 2026-05-08, sha b1728c1e660d, PR #42121)
TITLE: [Attention][Cleanup] Remove tree attention (#42121)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.dispatch.registry, L3.tree_attention
FILES: vllm/v1/attention/backends/registry.py (+0/-1); vllm/v1/attention/backends/tree_attn.py (+0/-488); .buildkite/intel_jobs/test-intel.yaml (+1/-1); docs/design/attention_backends.md (+0/-1); tests/utils.py (+1/-1); tests/v1/attention/test_attention_backends.py (+0/-1); tests/v1/e2e/spec_decode/test_spec_decode.py (+0/-5); tests/v1/spec_decode/test_eagle.py (+0/-160); tests/v1/spec_decode/test_tree_attention.py (+0/-506); vllm/config/speculative.py (+4/-21); (+1 more)
LABELS: documentation, speculative-decoding, ready, ci/build, v1
BODY: TreeAttention spec-decode is currently not fully supported and there's no plans to support it in the short to medium term (if we support it in the long term we can always add it back). Remove it to make refactoring the attention backends easier and remove cruft.

### L3-be0dcc29dc  (L3, 2026-05-09, sha be0dcc29dcfa, PR #40356)
TITLE: [XPU] remove q/k/v force contiguous for flash_attn (#40356)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/_xpu_ops.py (+1/-6)
LABELS: intel-gpu, ready
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-530d371302  (L3, 2026-05-09, sha 530d37130278, PR #41428)
TITLE: [DSv4] Improved fused Indexer Q quant kernel (#41428)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py (+45/-24); vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py (+423/-0); tests/kernels/test_fused_indexer_q_rope_quant.py (+1/-1); vllm/utils/import_utils.py (+5/-0)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Replace `_fused_indexer_q_rope_mxfp4_kernel` Triton kernel with a CuteDSL version to utilize 256-bit loads. Initially I wrote this in CUDA C++, but couldn't build vLLM from source, so asked Codex to port it over to CuteDSL. Hopefully this will be the first of many CuteDSL kernels to come in vLLM. ⏎  ⏎ Update: I keep the original Triton implementation for fallback (potentially for ROCm). Also put CuteDSL kernel in a separate file and a …[truncated]

### L3-986edc858a  (L3, 2026-05-09, sha 986edc858a13, PR #42169)
TITLE: [Bugfix] Fix DeepSeek v4 topk numerical issue for unaligned max-model-len (#42169)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: csrc/topk.cu (+2/-2)
LABELS: bug, ready, deepseek
BODY: ## Purpose: ⏎ It is observed that DeepSeek v4 has numerical issues when passing certain values of `--max-model-len` (e.g., 900000). For instance, the following figure shows the distribution of the number of completion tokens with 100 responses sampled for a selected question from GPQA. It can be seen that the number of generated tokens has a noticeable distribution shift when `--max-model-len 900000` is passed, compared to that not passed.  ⏎  ⏎ <im …[truncated]

### L3-bc5fdc1e6a  (L3, 2026-05-10, sha bc5fdc1e6a71, PR #41882)
TITLE: Add NVFP4 all-gather GEMM fusion for AsyncTP (#41882)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+42/-0); tests/compile/correctness_e2e/test_async_tp.py (+73/-0); tests/compile/correctness_e2e/test_sequence_parallel.py (+44/-5); tests/compile/fullgraph/test_toy_llama.py (+2/-1); tests/compile/fusions_e2e/test_tp2_async_tp.py (+65/-0); vllm/compilation/passes/fusion/collective_fusion.py (+243/-0); vllm/compilation/passes/fusion/sequence_parallelism.py (+136/-0)
LABELS: ready, llama, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ #27893 ⏎  ⏎ wires the NVFP4 FlashInfer all-gather + GEMM path into AsyncTP. ⏎  ⏎ It adds NVFP4 coverage for SP + AsyncTP by fusing: ⏎  ⏎ `all_gather(fp4 activation) + all_gather(group scales) + flashinfer_mm_fp4` ⏎  ⏎ into: ⏎  ⏎ `fused_all_gather_flashinfer_fp4_matmul` ⏎  ⏎ The reduce-scatter side is intentionally not enabled for NVFP4 in this PR. ⏎  ⏎ ## PyTorch Gap ⏎  ⏎ PyTorch does not currently provide an NVFP4-aware fused GEMM + reduce-scatter p …[truncated]

### L3-00b0618a03  (L3, 2026-05-10, sha 00b0618a0391, PR #39306)
TITLE: Use CU_MEMCPY_SRC_ACCESS_ORDER_ANY for batch KV cache swaps (#39306)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+8/-2); csrc/cache.h (+2/-1); csrc/torch_bindings.cpp (+2/-1); vllm/_custom_ops.py (+10/-1); vllm/v1/kv_offload/cpu/gpu_worker.py (+13/-1)
LABELS: ready, v1
BODY: Use `CU_MEMCPY_SRC_ACCESS_ORDER_ANY` instead of `CU_MEMCPY_SRC_ACCESS_ORDER_STREAM` in the `cuMemcpyBatchAsync` call used for batched KV cache swap copies. ⏎  ⏎ This relaxes the source access ordering constraint, allowing the CUDA driver to pipeline reads more aggressively. The safety of this change relies on the fact that source data is always fully written before the batch copy begins — the offloading handler synchronizes via stream events (`stre …[truncated]

### L3-0a309b5ee9  (L3, 2026-05-10, sha 0a309b5ee948, PR #38502)
TITLE: [ROCm] Cap Triton paged attention block size to fix ROCm shared memory OOM (#38502)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode, L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+8/-4); vllm/v1/attention/ops/chunked_prefill_paged_decode.py (+27/-8); .buildkite/test-amd.yaml (+9/-6)
LABELS: rocm, ready, ci/build, v1
BODY: Hybrid Mamba models (e.g. Jamba) inflate block_size to 2048 to align attention and Mamba page sizes. When the ROCm custom paged attention kernel rejects this (it only supports 16/32), the Triton fallback kernel_paged_attention_2d used 2048 as its tile size, requesting 262144 bytes of shared memory and thus exceeding the MI325X hardware limit of 65536 bytes. Cap TRITON_BLOCK_SIZE at 128. The kernel already decouples tile size from physical block s …[truncated]

### L3-21943d4c25  (L3, 2026-05-10, sha 21943d4c2589, PR #41499)
TITLE: [Performance] Make safetensors checkpoint prefetch settings configurable (#41499)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/load.py (+12/-0); vllm/engine/arg_utils.py (+17/-1); vllm/model_executor/model_loader/default_loader.py (+6/-0); vllm/model_executor/model_loader/weight_utils.py (+52/-14)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ This PR makes safetensors checkpoint prefetch settings configurable while preserving the existing defaults. ⏎  ⏎ Previously, `--safetensors-load-strategy=prefetch` used hardcoded values in `weight_utils.py`: ⏎  ⏎ - `num_prefetch_threads = 8` ⏎ - `block_size = 16 * 1024 * 1024` ⏎  ⏎ This change adds `LoadConfig` fields and CLI args for those values: ⏎  ⏎ - `--safetensors-prefetch-num-threads` ⏎ - `--safetensors-prefetch-block-size` ⏎  ⏎ The defaults remain un …[truncated]

### L3-171019ab19  (L3, 2026-05-10, sha 171019ab1923, PR #41536)
TITLE: add fused mhc_post_pre kernel (#41536)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_mhc_kernels.py (+142/-0); vllm/model_executor/layers/mhc.py (+343/-0); vllm/model_executor/models/deepseek_v4.py (+48/-11)
LABELS: ready, ci/build, deepseek
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ This PR adds a new mHC kernel, which fuses the `hc_post` operation with the `prenorm_gemm` portion of `hc_pre`. The approach is adapted from TRTLLM, and performs the GEMM using FMA (rather than tensor cores), improving speed at low concurrency. ⏎  ⏎ ### Benchmarks ⏎ Benchmark results with deepseek-ai/DeepSeek-V4-Flash at concurrency 4. ⏎  ⏎ Before this PR: ⏎ ``` ⏎ ================= Serving Benchmark Result ================= ⏎ Successful reque …[truncated]

### L3-5536fc0c01  (L3, 2026-05-11, sha 5536fc0c019d, PR #41188)
TITLE: [Misc] Replace mamba_type string literals with MambaAttentionBackendEnum (#41188)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L3.dispatch.selector, L3.dispatch.registry
FILES: vllm/v1/attention/backends/registry.py (+0/-10); vllm/v1/attention/selector.py (+4/-15); docs/contributing/model/basic.md (+1/-1); tests/kernels/mamba/test_ssu_dispatch.py (+10/-2); tests/v1/attention/test_attention_backends_selection.py (+13/-8); vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py (+2/-1); vllm/model_executor/layers/kda.py (+3/-2); vllm/model_executor/layers/mamba/abstract.py (+2/-1); vllm/model_executor/layers/mamba/gdn_linear_attn.py (+3/-2); vllm/model_executor/layers/mamba/linear_attn.py (+3/-2); (+8 more)
LABELS: documentation, ready, v1, kv-connector
BODY: ## Purpose ⏎ Convert `mamba_type` across all mamba-like layers from string literals (e.g., "mamba1", "gdn_attention") to `MambaAttentionBackendEnum`. ⏎  ⏎ Simplifies `vllm/v1/attention/selector.py` by removing the redundant `MAMBA_TYPE_TO_BACKEND_MAP` dictionary and lookup logic. ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-b1b59720b2  (L3, 2026-05-11, sha b1b59720b252, PR #38895)
TITLE: bugfix(flashinfer,dcp): remove kv_cache_layout for BatchDCPPrefillWrapper._new_tokens. (#38895)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+1/-3)
LABELS: bug, ready, v1, nvidia
BODY: ## Purpose ⏎ This PR fixes the compatibility issues with the HND kvcache layout when the FlashInfer backend is used with DCP enabled. ⏎  ⏎ While adapting the nixl connector for DCP, I discovered that the FlashInfer backend would crash during the warmup process after specifying the HND layout. The reason is that the `_new_tokens` wrapper was initialized with a specific kvcache layout, but the actual `key`/`value` tensors received by the wrapper durin …[truncated]

### L3-7863fff6e5  (L3, 2026-05-11, sha 7863fff6e591, PR #41812)
TITLE: [ROCm][DSv4] implement flash sparse mla with triton kernels (#41812)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.flashmla_sparse, L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/flashmla_sparse.py (+2/-2); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py (+682/-0); vllm/v1/attention/backends/mla/sparse_swa.py (+6/-0); vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+758/-164); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+377/-0); vllm/model_executor/layers/deepseek_v4_attention.py (+24/-46)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ This PR replaces ROCm's torch reference implementation of deepseek v4 sparse mla with triton kernels to support larger concurrency and improve performance. ⏎  ⏎ ## Test Plan ⏎ Model: DeepSeek-V4-Pro ⏎ Machine: MI355x ⏎ Accuracy: gsm8k with 8 shot ⏎ Performance: profiling ⏎  ⏎ ## Test Result ⏎ ### Accuracy: ⏎ ``` ⏎ lm_eval --model local-completions --model_args model=$MODEL,base_url=http://0.0.0.0:8001/v1/completions,num_concurrent=8,max_retries= …[truncated]

### L3-724ed2fc35  (L3, 2026-05-11, sha 724ed2fc352b, PR #42236)
TITLE: [DSv4] Improved dequant gather K cache kernel (#42236)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py (+30/-1); vllm/v1/attention/ops/deepseek_v4_ops/cutedsl_utils.py (+145/-0); vllm/v1/attention/ops/deepseek_v4_ops/dequant_gather_k_cutedsl.py (+334/-0); vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q_cutedsl.py (+8/-92); tests/kernels/test_compressor_kv_cache.py (+141/-7)
LABELS: ready, v1, verified, DSv4
BODY: ## Purpose ⏎  ⏎ When I did an nsys profile of DSv4 prefill, several memory-bound kernels show up as taking quite a bit of time. So I did microbenchmarks on some of them, finding out that `dequantize_and_gather_k_cache()` is currently having very low bandwidth utilization (it can only reach ~60GB/s for a single request). Hence, this PR introduces a CuteDSL version to improve it. ⏎  ⏎ This kernel somewhat follows the Triton kernel design: launch `(num_ …[truncated]

### L3-7f95e66a11  (L3, 2026-05-11, sha 7f95e66a11f5, PR #41119)
TITLE: [ROCm][Bugfix]: dynamically align BLOCK_DMODEL with Lv in MLA decode kernel (#41119)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.triton.decode_attention
FILES: vllm/v1/attention/ops/triton_decode_attention.py (+20/-11)
LABELS: bug, rocm, ready, v1
ISSUES: #40966 [Bug]: Triton MLA decode kernel shape mismatch for Mistral-Small on ROCm when TP > 1
BODY: ## Purpose ⏎  ⏎ Fixes #40966  ⏎  ⏎ This PR resolves a Triton compilation error, `ValueError('Cannot make_shape_compatible: incompatible dimensions at index 1: 256 and 512')`  occurring in MLA-based models Mistral-Small on ROCm. ⏎  ⏎ **Key Changes:** ⏎  ⏎ **Dynamic Dimension Alignment:** Updated the triton_decode_attention kernel to align BLOCK_DMODEL with the local latent rank (Lv) instead of the total head dimension (Lk) when is_mla=True. This ensures s …[truncated]

### L3-0d453e2336  (L3, 2026-05-11, sha 0d453e233647, PR #40408)
TITLE: [Perf] Batch invariance with Cutlass fp8 support, 28.9% E2E latency improvement (#40408)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8.cu (+9/-0); csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8_dispatch.cuh (+52/-0); csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu (+9/-0); csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh (+42/-0); csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8.cu (+9/-0); csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh (+53/-0); csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x.cu (+9/-0); csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x_sm89_fp8_dispatch.cuh (+39/-0); tests/v1/determinism/test_cutlass_batch_invariance.py (+68/-0); vllm/model_executor/layers/quantization/fp8.py (+9/-3); (+1 more)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Use cutlass fp8 to avoid quantize/dequantize overhead ⏎  ⏎ ## Test ⏎  ⏎ ### Perf ⏎  ⏎ `VLLM_BATCH_INVARIANT=1 vllm bench latency   --model=Qwen/Qwen3-1.7B   --quantization=fp8   --attention-backend=TRITON_ATTN   --input-len=1024   --output-len=128   --batch-size=8` ⏎  ⏎ ```bash ⏎ # now ⏎ Avg latency: 0.5149560342853268 seconds ⏎ 10% percentile latency: 0.5113491305150092 seconds ⏎ 25% percentile latency: 0.5118766154628247 seconds ⏎ 50% percenti …[truncated]

### L3-a51376b3f0  (L3, 2026-05-11, sha a51376b3f05a, PR #40392)
TITLE: [Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA (#40392)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: csrc/cache_kernels_fused.cu (+75/-60); vllm/compilation/passes/fusion/matcher_utils.py (+84/-0); vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py (+271/-0); vllm/compilation/passes/pass_manager.py (+4/-0); vllm/compilation/passes/utility/fix_functionalization.py (+39/-0); vllm/config/compilation.py (+10/-1); vllm/config/vllm.py (+13/-0); vllm/model_executor/layers/attention/mla_attention.py (+19/-24); tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py (+413/-0); tests/compile/passes/test_rope_kvcache_fusion.py (+4/-11); (+2 more)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Reland updated version of #35245 #35879 #38646, to fuse MLA RoPE and KV Cache ops. Adding some pattern matching fixes/minimization on top of #35879. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ # server startup ⏎ vllm serve amd/DeepSeek-R1-0528-MXFP4 --tensor-parallel-size=8 --gpu-memory-utilization=0.94 --dtype=auto --kv-cache-dtype=fp8 --max-num-batched-tokens=8192 --attention-backend ROCM_AITER_MLA -cc.pass_config.fuse_rope_kvcache_cat_mla=True -cc.use_in …[truncated]

### L3-a721315488  (L3, 2026-05-11, sha a72131548842, PR #41825)
TITLE: [ROCm][Perf] Fix RMSNorm+Quant fusion for gfx950 (non-fnuz) (#41825)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/passes/test_double_aiter_rms_quant_fusion.py (+166/-0); vllm/compilation/passes/fusion/matcher_utils.py (+1/-6); vllm/compilation/passes/fusion/rocm_aiter_fusion.py (+175/-1)
LABELS: rocm, ready, v1, deepseek
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ `RocmAiterRMSNormQuantFusionPass` was silently skipping the fused AITER RMSNorm+GroupedQuantFP8 kernel on gfx950 (non-fnuz hardware) due to two issues: ⏎  ⏎ 1. **`matcher_utils.py`**: `MatcherQuantFP8` guarded `get_group_quant_op()` behind `is_fp8_fnuz()`. On gfx950 this is `False`, so the matcher always selected `triton_per_token_group_quant_fp8` instead — a different op target that the fusion pattern does not match. Fix: always use  …[truncated]

### L3-bbee532988  (L3, 2026-05-11, sha bbee5329880b, PR #41429)
TITLE: [Perf][1/n] Eliminate various GPU<->CPU syncs (#41429)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/mamba_mixer.py (+1/-0); vllm/v1/sample/ops/bad_words.py (+2/-1); vllm/v1/sample/sampler.py (+17/-14); vllm/v1/spec_decode/ngram_proposer_gpu.py (+3/-2); vllm/v1/worker/dp_utils.py (+7/-5); vllm/v1/worker/gpu/sample/penalties.py (+4/-2); vllm/v1/worker/gpu_model_runner.py (+15/-9)
LABELS: speculative-decoding, ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: Fix first batch of unnecessary gpu/cpu syncs, found via https://github.com/vllm-project/vllm/pull/40561: ⏎  ⏎ Should benefit the following features: ⏎  ⏎ - MRv1 specific `logprob_token_ids` impl ⏎ - MRv1 `bad_words` sampling parameter impl ⏎ - MRv1 fast prefill ⏎ - MRv1 pooling (preprocessing) ⏎ - GPU ngram spec decoding (batch reorder) ⏎ - DP native all-reduce ⏎ - MRv2 penalties ⏎ - Mamba-1 prefill

### L3-5f1b313900  (L3, 2026-05-11, sha 5f1b313900fd, PR #41942)
TITLE: [ROCm] Clean up a bit the AITER FA backend (#41942)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+15/-50)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ Some of the meta data is not used and doesn't need to be computed. ⏎ Also, avoid a CPU sync when batch contains decode only. (In that case, it's not needed to copy seq_lens from GPU to CPU.) ⏎  ⏎ ## Test Plan ⏎ Run an exemplary model with the AITER FA backend and check lm_eval results. ⏎ Also, run a latency benchmark to gauge impact of the change. ⏎  ⏎ ## Test Result ⏎  ⏎ ### `vllm bench latency` ⏎ ``` ⏎ python -m vllm.entrypoints.cli.main bench …[truncated]

### L3-1ff9d33535  (L3, 2026-05-12, sha 1ff9d335355c, PR #42387)
TITLE: [CI] Migrate remaining B200 jobs to b200-k8s with test fixes (#42387)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/lm_eval.yaml (+13/-1); .buildkite/test_areas/spec_decode.yaml (+2/-2); tests/evals/gsm8k/configs/models-blackwell-ep.txt (+3/-0); tests/evals/gsm8k/configs/models-blackwell.txt (+0/-3); tests/v1/e2e/spec_decode/test_spec_decode.py (+13/-1)
LABELS: ci/build, v1
BODY: ## Summary ⏎ - Migrates the last 3 `device: b200` jobs to `b200-k8s`, with fixes for pre-existing test failures discovered during validation in #42356 ⏎ - **Spec Decode Eagle** (`spec-decode-eagle-nightly-b200`): Skip `test_eagle_correctness_light` on Blackwell — DeepSeek `head_dim=(192,192)` is not supported on SM100/SM110 with FLASH_ATTN ⏎ - **Spec Decode Speculators+MTP** (`spec-decode-speculators-mtp-nightly-b200`): Skip `deepseek` MTP variant on B …[truncated]

### L3-4e498b5e5c  (L3, 2026-05-12, sha 4e498b5e5c07, PR #40657)
TITLE: [Bugfix][Performance Improvement] Improve penalties triton kernel performance (#40657)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/sample/penalties.py (+8/-15); vllm/v1/worker/gpu/sample/sampler.py (+0/-1)
LABELS: bug, ready, v1
ISSUES: #180908 [vllm] [triton 3.7] PassManager::run failed in make_ttgir when compiling vLLM _penalties_kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: Replace tl.static_range(MAX_SPEC_LEN) with tl.range(pos) in _penalties_kernel to avoid a Triton 3.7.0 TritonGPURemoveLayoutConversions pass crash on sm_89 (L4) GPUs when num_speculative_tokens >= 10. ⏎  ⏎ The unrolled scf.if blocks from tl.static_range produced IR too complex for the pass to handle. tl.range emits a compact scf.for loop instead. ⏎  ⏎ We also make the following optimization: Instead of accumulating into a fresh draft_counts = tl.zeros …[truncated]

### L3-ef34592a1a  (L3, 2026-05-12, sha ef34592a1ac7, PR #41382)
TITLE: [Bugfix] Fix double reduce in flashinfer_nvlink_two_sided and flashinfer_nvlink_one_sided backends (#41382)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py (+1/-1); vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_two_sided.py (+1/-1)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎ Fix accuracy degradation when using `--all2all-backend flashinfer_nvlink_two_sided` or `--all2all-backend flashinfer_nvlink_one_sided`. ⏎  ⏎ Apparently reduce is already performed in flashinfer, see https://github.com/flashinfer-ai/flashinfer/blob/v0.6.8.post1/flashinfer/comm/trtllm_alltoall.py#L663 , so no need to perform it again in vLLM. This double reduce seemed to have caused the accuracy degradation, that was measured as following …[truncated]

### L3-a7b801e26d  (L3, 2026-05-12, sha a7b801e26d6b, PR #41664)
TITLE: [MXFP4] Support for linear layers + compressed-tensors integration (#41664)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+41/-2); tests/quantization/test_compressed_tensors.py (+29/-0); vllm/model_executor/kernels/linear/__init__.py (+68/-0); vllm/model_executor/kernels/linear/mxfp4/__init__.py (+12/-0); vllm/model_executor/kernels/linear/mxfp4/base.py (+67/-0); vllm/model_executor/kernels/linear/mxfp4/flashinfer.py (+74/-0); vllm/model_executor/kernels/linear/mxfp4/marlin.py (+52/-0); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py (+2/-2); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py (+1/-0); vllm/model_executor/layers/quantization/compressed_tensors/schemes/__init__.py (+2/-2); (+1 more)
LABELS: ready, nvidia, quantization
BODY: ## Purpose ⏎ - Update scheme name from a16 to a4 to be agnostic for activation quantization  ⏎ - Extend `flashinfer_mm_fp4` and `flashinfer_scaled_fp4_mm`  to take in a configurable `block_size` and boolean flag `use_nvfp4` to enable the mxfp4 linear forward pass based on https://github.com/flashinfer-ai/flashinfer/blob/393e83ea8497ff9fb9ad61e170b89797a6b682a3/flashinfer/gemm/gemm_base.py#L5511 ⏎ - Follow the pattern of linear kernels to allow selec …[truncated]

### L3-dd6b3a5ef5  (L3, 2026-05-12, sha dd6b3a5ef548, PR #42153)
TITLE: [Perf] Use 2D-grid to eliminate divmod in W8W8 group quant (#42153)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu (+69/-40)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Replace the 1D-grid `(global_group_id % padded_gpr, / padded_gpr)` divmod with a 2D grid + template-constant index unpack.  Since the tile sizes are compile-time constants, the compiler can lower the coordinate math to simple bit operations instead of emitting runtime division/modulo setup. ⏎  ⏎ Micro-benchmark on DeepSeek V4 Pro shape ⏎  ⏎ |    K |    M | baseline us | opt us | speedup | ⏎ |-----:|-----:|------------:|---------:|--------: …[truncated]

### L3-8f89381fc6  (L3, 2026-05-12, sha 8f89381fc6b2, PR #39822)
TITLE: [Hybrid] Warmup Mamba2 SSD kernel (#39822)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/model.py (+2/-2); vllm/model_executor/layers/mamba/gdn_linear_attn.py (+2/-1); vllm/model_executor/layers/mamba/mamba_mixer2.py (+106/-1)
LABELS: ready
BODY: ## Summary ⏎  ⏎ Triton's auto-tuner for the Mamba2 SSD kernels currently runs lazily on the first inference request, causing a large latency spike. This PR adds a `_warmup_ssd_kernels()` method to `MambaMixer2` that triggers auto-tuning during vLLM's profile phase (before SSM cache allocation), shifting the cost into server startup. ⏎  ⏎ - Runs a minimal `mamba_chunk_scan_combined_varlen` forward pass with dummy tensors during the V1 profile run ⏎ - Covers …[truncated]

### L3-c8a6e272e0  (L3, 2026-05-12, sha c8a6e272e0d3, PR #42225)
TITLE: [CPU] Fix rotary embedding for CPU without flash-attn ops (#42225)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rotary_embedding/common.py (+2/-1)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Fixes CPU inference for modern models (Llama 3.x, Qwen 2.x/3.x) that use rotary embeddings by handling missing flash-attn ops module in CPU-only environments. ⏎  ⏎ **Problem**: CPU inference crashes with `ModuleNotFoundError: No module named 'flash_attn.ops'` when loading models with rotary embeddings (Llama 3.x, Qwen 2.x/3.x). ⏎  ⏎ **Root cause**: The code checks if flash-attn package is installed (`find_spec("flash_attn")`), but in CP …[truncated]

### L3-184577ae46  (L3, 2026-05-12, sha 184577ae46f5, PR #42429)
TITLE: [Build] DeepGEMM: trim comments, add integration notes + TODOs (#42429)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: cmake/external_projects/deepgemm.cmake (+27/-14); docker/Dockerfile (+3/-4); tools/build_deepgemm_C.py (+5/-7); tools/setup_deepgemm_pythons.sh (+5/-18)
LABELS: ready, ci/build
BODY: ## Summary ⏎  ⏎ Follow-up to #41516 per review feedback (asking for background + TODOs on the DeepGEMM integration). No behavior change. ⏎  ⏎ - Adds a short integration-notes block at the top of `if(DEEPGEMM_ARCHS)` in `cmake/external_projects/deepgemm.cmake` explaining why we build `_C` per-Python and listing the two cleanups we'd like to make next (TORCH_LIBRARY+shim binding, AOT kernel compile). ⏎ - Trims verbose inline how-it-works prose across `deepge …[truncated]

### L3-ebeb09d822  (L3, 2026-05-12, sha ebeb09d82261, PR #40900)
TITLE: [KV Transfer] Add MooncakeStoreConnector for KV cache offloading via Mooncake distributed store (#40900)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/features/mooncake_store_connector_usage.md (+161/-0); tests/v1/kv_connector/unit/test_mooncake_store_connector.py (+258/-0); tests/v1/kv_connector/unit/test_mooncake_store_worker.py (+300/-0); vllm/distributed/kv_transfer/kv_connector/factory.py (+5/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_utils.py (+10/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/__init__.py (+2/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py (+229/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py (+276/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py (+380/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py (+979/-0)
LABELS: documentation, ready, v1, kv-connector
BODY: ## Summary ⏎  ⏎ Add `MooncakeStoreConnector`, a new KV connector that uses [MooncakeDistributedStore](https://github.com/kvcache-ai/Mooncake) as a shared KV cache pool. Unlike the existing `MooncakeConnector` (which does direct point-to-point KV transfer between prefiller and decoder), `MooncakeStoreConnector` offloads KV cache to an external distributed store, enabling: ⏎  ⏎   - **CPU/Disk offloading**: Extend effective KV cache capacity by offloadi …[truncated]

### L3-3d635c58c0  (L3, 2026-05-12, sha 3d635c58c058, PR #42460)
TITLE: [Perf] Optimize MLA `compute_prefill_context` memory allocation (#42460)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+16/-12)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Originally: ⏎  ⏎ ```bash ⏎ chunk1 -> output1 ⏎ merge(output1, chunk2) -> allocate output2 ⏎ merge(output2, chunk3) -> allocate output3 ⏎ merge(output3, chunk4) -> allocate output4 ⏎ ``` ⏎  ⏎ Now ⏎  ⏎ ```bash ⏎ chunk1 -> outputA ⏎ merge(outputA, chunk2) -> outputB ⏎ merge(outputB, chunk3) -> outputA ⏎ merge(outputA, chunk4) -> outputB ⏎ # Only two allocations for the whole process ⏎ ``` ⏎  ⏎ For example ⏎  ⏎ ```bash ⏎ case (1024, 128, 128, 16 chunks): ⏎  ⏎  …[truncated]

### L3-67c89fe40a  (L3, 2026-05-12, sha 67c89fe40ac5, PR #42333)
TITLE: [Model][Bugfix] Fix Step3-VL image_embeds input path (#42333)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/processing/test_step3_vl_image_embeds.py (+58/-0); vllm/model_executor/models/step3_vl.py (+14/-10)
LABELS: bug, ready, multi-modality, verified
BODY: ## Purpose ⏎  ⏎ Fix the Step3-VL `image_embeds` input path, which is currently broken. ⏎  ⏎ `Step3VLImageEmbeddingInputs` declares a required `TensorSchema` field named `data`, but `_parse_and_validate_image_input()` previously constructed the object with `image_embeds=...`. This causes `TensorSchema.validate()` to raise `ValueError: Required field 'data' is missing`. ⏎  ⏎ Additionally, `_process_image_input()` had a control-flow bug: the `image_embeds …[truncated]

### L3-dcacdf9a88  (L3, 2026-05-12, sha dcacdf9a8860, PR #41052)
TITLE: [Attention] Sync FA with upstream (#41052)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ Partner of https://github.com/vllm-project/flash-attention/pull/134 ⏎  ⏎ ## Test Plan ⏎ CI ⏎  ⏎ ## Test Result ⏎ TBD ⏎  ⏎ --- ⏎ [details omitted]

### L3-0ddaf6dffa  (L3, 2026-05-13, sha 0ddaf6dffa0d, PR #38896)
TITLE: [XPU] [CT] Enable CT W4A4MxFp4 path and add xpu kernel (#38896)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/__init__.py (+6/-0); vllm/model_executor/kernels/linear/mxfp4/xpu.py (+53/-0)
LABELS: intel-gpu, ready
BODY: 1. add a  mxfp4 xpu gemm kernel  ⏎  ⏎ Test: ⏎ accuracy passed with Yi30/Llama-3.2-1B-Instruct-MXFP4-llmc

### L3-85b2fecab7  (L3, 2026-05-13, sha 85b2fecab777, PR #42339)
TITLE: [5/n] Migrate CUTLASS MLA, hadamard, awq, allspark and DSV3 fused a gemm to torch stable ABI (continued) (#42339)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.mla.cutlass_sm100
FILES: CMakeLists.txt (+71/-70); csrc/libtorch_stable/attention/mla/sm100_cutlass_mla_kernel.cu (+55/-56); csrc/ops.h (+0/-20); csrc/torch_bindings.cpp (+0/-51); csrc/core/scalar_type.hpp (+26/-18); csrc/libtorch_stable/attention/mla/cutlass_sm100_mla/device/sm100_mla.hpp (+0/-0); csrc/libtorch_stable/attention/mla/cutlass_sm100_mla/kernel/sm100_fmha_mla_reduction.hpp (+0/-0); csrc/libtorch_stable/attention/mla/cutlass_sm100_mla/kernel/sm100_fmha_mla_tma_warpspecialized.hpp (+0/-0); csrc/libtorch_stable/attention/mla/cutlass_sm100_mla/kernel/sm100_mla_tile_scheduler.hpp (+0/-0); csrc/libtorch_stable/dsv3_fused_a_gemm.cu (+38/-33); (+10 more)
LABELS: ready, ci/build, nvidia
BODY: This is a continuation of the PR #38671 ⏎  ⏎ cc @janeyx99  ⏎  ⏎  ⏎  ⏎  ⏎ ## Purpose ⏎  ⏎ https://github.com/vllm-project/vllm/issues/26946 ⏎  ⏎ ## Test Plan ⏎  ⏎ On A100       ⏎ ```                                                                                                                       ⏎   python -m pytest tests/kernels/quantization/test_allspark_gemm.py           ⏎ ```    ⏎  ⏎ On H100   ⏎ ```                                                             …[truncated]

### L3-97c4317bf5  (L3, 2026-05-13, sha 97c4317bf5c6, PR #42329)
TITLE: [Bugfix][Frontend] Default max_tokens server-side on /inference/v1/generate (#42329)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/entrypoints/openai/test_openai_schema.py (+1/-0); tests/entrypoints/serve/disagg/test_protocol.py (+70/-0); tests/entrypoints/serve/disagg/test_serving_tokens.py (+30/-0); vllm/entrypoints/serve/disagg/protocol.py (+40/-1); vllm/entrypoints/serve/disagg/serving.py (+27/-1)
LABELS: bug, frontend, ready
BODY: ## Summary ⏎  ⏎ `/inference/v1/generate` hands the client-supplied `SamplingParams` to the engine verbatim. Because `SamplingParams.max_tokens` defaults to `16` at the dataclass level (`vllm/sampling_params.py:231`), any client that omits `max_tokens` silently gets a 16-token completion — long enough to start a sentence and stop mid-word. ⏎  ⏎ `/v1/chat/completions` (`vllm/entrypoints/openai/chat_completion/serving.py:280`) and `/v1/completions` (`vllm/e …[truncated]

### L3-3c413a5481  (L3, 2026-05-13, sha 3c413a548177, PR #40327)
TITLE: Triton attention: add USE_TD constexpr for tensor descriptor Q/K/V load/store (#40327)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.triton.unified_attention, L3.triton.v1_backend
FILES: vllm/envs.py (+11/-0); vllm/v1/attention/backends/triton_attn.py (+17/-0); vllm/v1/attention/ops/triton_unified_attention.py (+368/-70); tests/kernels/attention/test_triton_unified_attention.py (+179/-0)
LABELS: ready, v1, verified
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ Add an optional tensor-descriptor (TD) load/store path to the Triton ⏎ unified attention kernel in `vllm/v1/attention/ops/triton_unified_attention.py`. ⏎ The TD path uses `tl.make_tensor_descriptor` for Q/K/V loads and the ⏎ output store, which enables HW 2D block reads on Intel Xe2/Xe3. ⏎ The non-TD branch is preserved and stays bit-for-bit identical — both ⏎ `USE_TD` and `USE_TD_QO` are `tl.constexpr` so the dead branch is ⏎ eliminated at Trito …[truncated]

### L3-6b5c389ee3  (L3, 2026-05-13, sha 6b5c389ee326, PR #41252)
TITLE: expose flex block size for batch invariant mode (#41252)
SOURCES: path_core
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/model_executor/layers/attention/attention.py (+23/-0); vllm/v1/attention/backends/flex_attention.py (+59/-7); tests/v1/determinism/test_batch_invariance.py (+11/-1); vllm/config/attention.py (+21/-0)
LABELS: ready, v1
BODY: under VLLM_BATCH_INVARIANT_MODE, expose Flex Attention block sizes through additional env vars so that the user can set them. previously this was hardcoded to 16 (default with this PR is still 16). ⏎  ⏎ testing:  ⏎ parametrized block size N/M in batch invariant test and also added test for invalid numbers (ie not powers of 2).

### L3-92def124bc  (L3, 2026-05-13, sha 92def124bcb7, PR #42151)
TITLE: [MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5 (#42151)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/design/cuda_graphs_multimodal.md (+2/-1); examples/generate/multimodal/vision_language_offline.py (+93/-1); tests/models/multimodal/generation/test_vit_cudagraph.py (+15/-3); vllm/model_executor/models/qwen3_5.py (+2/-0)
LABELS: documentation, ready, multi-modality, qwen, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Support ViT full CUDA graph for Qwen3.5 (reusing Qwen3-VL ViT implementation). ⏎  ⏎ ## Test Plan ⏎  ⏎ Functional test: ⏎  ⏎ ```bash ⏎ python examples/generate/multimodal/vision_language_offline.py -m qwen3_5 --modality "image" --enable-vit-cuda-graph ⏎ python examples/generate/multimodal/vision_language_offline.py -m qwen3_5 --modality "video" --enable-vit-cuda-graph ⏎ ``` ⏎  ⏎ CI test: ⏎  ⏎ ``` ⏎ pytest -sv tests/models/multimodal/generation/tes …[truncated]

### L3-0d2732dd91  (L3, 2026-05-13, sha 0d2732dd919b, PR #41778)
TITLE: [MLA Attention Backend] Add TOKENSPEED_MLA backend for DSR1/Kimi K25 prefill + decode on Blackwell (#41778)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.dispatch.registry, L3.platform.cuda_selection
FILES: requirements/cuda.txt (+3/-0); vllm/model_executor/layers/attention/mla_attention.py (+1/-0); vllm/platforms/cuda.py (+4/-0); vllm/v1/attention/backends/mla/prefill/registry.py (+4/-0); vllm/v1/attention/backends/mla/prefill/selector.py (+1/-0); vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py (+180/-0); vllm/v1/attention/backends/mla/tokenspeed_mla.py (+277/-0); vllm/v1/attention/backends/registry.py (+3/-0); benchmarks/attention_benchmarks/configs/mla_decode.yaml (+1/-0); benchmarks/attention_benchmarks/configs/mla_prefill.yaml (+2/-0); (+4 more)
LABELS: documentation, performance, ready, ci/build, v1, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ Wires the tokenspeed_mla CuTe DSL kernels into vLLM as a new MLA backend, covering both prefill (tokenspeed_mla_prefill) and decode (tokenspeed_mla_decode). Targets Blackwell (SM100) with FP8 KV cache and DeepSeek R1 MLA dimensions;  ⏎  ⏎ Enable by adding ⏎ ``` ⏎ -ac {\"backend\":\"TOKENSPEED_MLA\",\"mla_prefill_backend\":\"TOKENSPEED_MLA\",\"use_prefill_query_quantization\":true}  ⏎ ``` ⏎ To the engine config ⏎  ⏎ This kernel is optimized fo …[truncated]

### L3-b3c69595a6  (L3, 2026-05-14, sha b3c69595a63f, PR #41736)
TITLE: [MM][CG] Support ViT CG for Qwen2-VL (#41736)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/design/cuda_graphs_multimodal.md (+2/-1); examples/generate/multimodal/vision_language_offline.py (+1/-0); tests/models/multimodal/generation/test_vit_cudagraph.py (+12/-0); vllm/model_executor/models/qwen2_vl.py (+300/-20)
LABELS: documentation, ready, multi-modality, qwen, nvidia
BODY: ## Purpose ⏎ Enable Cudagraph for ViT for Qwen2.5-VL following the precedence from https://github.com/vllm-project/vllm/pull/35963. ⏎  ⏎ ## Test Plan ⏎ Added record in the file `tests/models/multimodal/generation/test_vit_cudagraph.py` ⏎  ⏎ ## Test Result ⏎ **E2E** ⏎ Test on H100 ⏎ Engine command ⏎ ``` ⏎ vllm serve Qwen/Qwen2-VL-7B-Instruct \ ⏎     --max-model-len 8192 \ ⏎     --no-enable-prefix-caching \ ⏎     --max-num-batched-tokens 4096 \ ⏎     --max-num-se …[truncated]

### L3-751b9f14bd  (L3, 2026-05-14, sha 751b9f14bd5c, PR #41918)
TITLE: [XPU][CT] Support mxfp8 moe model (#41918)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/xpu_moe.py (+44/-0); vllm/model_executor/layers/fused_moe/oracle/fp8.py (+2/-1); vllm/model_executor/layers/fused_moe/oracle/mxfp8.py (+2/-0)
LABELS: intel-gpu, ready
BODY: ## Purpose ⏎ add a new `XPUExpertsMxfp8` class for MXFP8 moe model.  ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ python3 examples/basic/offline_inference/generate.py --model INCModel/Qwen3-30B-A3B-MXFP8-LLMC   --enforce-eager --temperature 0 -tp 2 --max-model-len 1024 ⏎ ``` ⏎  ⏎ ## Test Result ⏎ output make sense. ⏎  ⏎ --- ⏎ [details omitted]

### L3-2317682f95  (L3, 2026-05-14, sha 2317682f9511, PR #42112)
TITLE: [Bugfix] Fix TRTLLM ragged MLA prefill workspace warmup (#42112)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/prefill/flashinfer.py (+6/-7); vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py (+3/-8)
LABELS: bug, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ Fix `TRTLLM_RAGGED` MLA prefill failing after CUDA graph capture. ⏎  ⏎ After the MLA prefill backend refactor in #32623, the FlashInfer workspace used by `TRTLLM_RAGGED` MLA prefill is allocated lazily. The first real request after `lock_workspace()` could hit: ⏎  ⏎ ```text ⏎ Workspace is locked but allocation from 'trtllm_ragged.py:66:_get_workspace_buffer' requires 394.00 MB, current size is 0.00 MB. Workspace growth is not allowed aft …[truncated]

### L3-ae4f59f0ec  (L3, 2026-05-14, sha ae4f59f0ece8, PR #39337)
TITLE: [Model Runner v2] Oracle for model runner v2 - qwen3 dense model by default [1/N] (#39337)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+1/-1); tests/test_config.py (+99/-0); vllm/config/vllm.py (+102/-12); vllm/envs.py (+4/-4); vllm/v1/core/sched/scheduler.py (+1/-2); vllm/v1/worker/gpu_worker.py (+1/-1)
LABELS: structured-output, ready, v1, qwen, kv-connector, nvidia, mrv2
BODY: ## Purpose ⏎  ⏎ Oracle for model runner v2 - dense model by default  ⏎  ⏎ Now the env function: ⏎  ⏎ - Not set: using our oracle ⏎ - set to 1: force v2 ⏎ - set to 0: force v1 ⏎  ⏎ We are testing "Qwen/Qwen3-0.6B" and "facebook/opt-125m" since they cover the most current v1 unit test. ⏎  ⏎ Should land after https://github.com/vllm-project/vllm/pull/39353 ⏎  ⏎ ## Test ⏎  ⏎ Covered in unit test

### L3-9898f94abe  (L3, 2026-05-14, sha 9898f94abe00, PR #42555)
TITLE: [Attention] Remove deprecated MLA prefill arguments (#42555)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/config/attention.py (+1/-52); vllm/model_executor/layers/attention/mla_attention.py (+2/-2); benchmarks/attention_benchmarks/mla_runner.py (+18/-39); tests/engine/test_arg_utils.py (+0/-12); tests/v1/attention/test_mla_prefill_selector.py (+6/-21)
LABELS: performance, ready, v1
BODY: Land after 0.21 is cut. ⏎  ⏎ ## Purpose ⏎ These arguments were slated for removal in 0.22, because following the refactor (#32623) the MLA prefill backend is now specified by `--attention-config.mla_prefill_backend`. This PR removes them. ⏎  ⏎ ## Test Plan ⏎ CI: V1 Attention (H100) and V1 Attention (B200) ⏎  ⏎ ## Test Result ⏎ TBD ⏎  ⏎ --- ⏎ [details omitted]

### L3-c7560af424  (L3, 2026-05-14, sha c7560af42487, PR #39568)
TITLE: [RFC] Replace shared-memory routed experts with ModelRunnerOutput transfer and HTTP support (#39568)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/model_executor/test_routed_experts_capture.py (+12/-7); tests/v1/core/test_async_scheduler.py (+1/-0); tests/v1/core/test_scheduler.py (+1/-0); vllm/config/vllm.py (+24/-0); vllm/entrypoints/serve/disagg/protocol.py (+10/-0); vllm/entrypoints/serve/disagg/serving.py (+16/-0); vllm/model_executor/layers/fused_moe/routed_experts_capturer.py (+242/-264); vllm/sampling_params.py (+7/-0); vllm/v1/core/sched/scheduler.py (+91/-61); vllm/v1/engine/output_processor.py (+13/-7); (+2 more)
LABELS: performance, frontend, ready, v1
ISSUES: #38079 [RFC] Redesign enable_return_routed_experts to avoid blocking EngineCore event loop
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Background ⏎  ⏎ The previous `routed_experts_capturer` relied on `SharedMemory + fcntl.flock + /tmp` lock files for cross-process transport. The worker was forced into a `cpu().numpy()` on every step, and the scheduler performed a `flock` + shm read for every finished request. Consequences: synchronous CUDA calls on the hot path, frequent NCCL timeouts on cross-node EP, an outright `AssertionError` under SP + modular-kernel, and routed-experts d …[truncated]

### L3-f887aa1a53  (L3, 2026-05-14, sha f887aa1a53e2, PR #40710)
TITLE: [Aiter][ROCm] RMSNormGated+GroupedQuantFP8 fusion (#40710)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/passes/test_fusion.py (+240/-1); vllm/_aiter_ops.py (+58/-0); vllm/compilation/passes/fusion/matcher_utils.py (+62/-1); vllm/compilation/passes/fusion/rocm_aiter_fusion.py (+124/-1); vllm/compilation/passes/vllm_inductor_pass.py (+27/-0); vllm/model_executor/layers/layernorm.py (+44/-29)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: This PR adds a compilation fusion pass (AiterRMSNormGatedFp8GroupQuantPattern) that fuses the decomposed RMSNormGated + reshape + group FP8 quantization sequence into a single AITER Triton kernel call (fused_rms_gated_fp8_group_quant). This pattern appears in GatedDeltaNetAttention layers (e.g., Qwen3-Next) where each attention head's output goes through gated RMS normalization, is reshaped back to the full hidden dimension, and then group-quanti …[truncated]

### L3-f07b1da797  (L3, 2026-05-14, sha f07b1da797cc, PR #42062)
TITLE: [ROCm] Enable gluon paged MQA logits on gfx950 (MI355X) (#42062)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+3/-2)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ### Summary ⏎  ⏎ `rocm_fp8_paged_mqa_logits` in `rocm_aiter_mla_sparse.py` had two separate branches: ⏎  ⏎ - `_ON_GFX942` (MI300X, MI325X): called `deepgemm_fp8_paged_mqa_logits` → gluon single-kernel path - everything else (including MI355X / gfx950): fell through to `deepgemm_fp8_paged_mqa_logits_stage1` → slow two-kernel path (stage1 + `sum(dim=0)`) ⏎  ⏎ The split was introduced in 628c436 (#40871). With Triton >= 3.5.0, `deepgemm_fp8_paged_mqa_logi …[truncated]

### L3-3b6a204789  (L3, 2026-05-14, sha 3b6a2047899b, PR #42444)
TITLE: [Model Runner V2][Bug Fix][DSV4] Ensure lazy attention state initializations happen during cudagraph capture (#42444)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/cudagraph_utils.py (+7/-1)
LABELS: bug, ready, v1, nvidia, mrv2
BODY: # Context ⏎ DeepSeek V4 Flash with full CUDA graph capture in Model Runner V2 produces corrupted output (gibberish or repetition loops). The root cause is that `CudaGraphManager.capture()` runs a warmup forward pass before the actual graph capture, and both passes share the same `FlashMLASchedMeta `objects. ⏎  ⏎ FlashMLA's C++ decode kernel (`sparse_decode_fwd` / `dense_decode_fwd`) has a conditional initialization path: if `tile_scheduler_metadata` …[truncated]

### L3-24337fb860  (L3, 2026-05-14, sha 24337fb860a8, PR #41869)
TITLE: PD disagg with NIXL Connector: GDN support (Qwen3.5) (#41869)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/disaggregated.yaml (+1/-1); tests/v1/kv_connector/nixl_integration/config_sweep_accuracy_test.sh (+3/-0); tests/v1/kv_connector/nixl_integration/test_accuracy.py (+1/-0); tests/v1/kv_connector/unit/test_nixl_connector_hma.py (+126/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py (+3/-13); vllm/distributed/kv_transfer/kv_connector/v1/ssm_conv_transfer_utils.py (+110/-69)
LABELS: ready, ci/build, v1, qwen, kv-connector
ISSUES: #41886 [Feature]: NIXL P/D Disaggregation: GDN support (Qwen3.5)
BODY: Closes https://github.com/vllm-project/vllm/issues/41886 ⏎  ⏎ ## Summary ⏎ - Add GDN (Gated Delta Net) conv-state layout support for NIXL KV transfer ⏎ - Fix heterogeneous TP kernel block matching for mamba hybrid models ⏎ - Handle physical_blocks_per_logical mismatch between P and D in disaggregated serving ⏎  ⏎ ## Test plan ⏎  ⏎  ⏎ ## Accuracy Results ⏎  ⏎ **Model:** `Qwen/Qwen3.5-0.8B` (GDN) ⏎ **Benchmark:** GSM8K `exact_match,strict-match` (5-shot) ⏎ **Stan …[truncated]

### L3-0d4d334eaa  (L3, 2026-05-14, sha 0d4d334eaa58, PR #42150)
TITLE: Bump llguidance to 1.7 (#42150)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/common.txt (+1/-1); requirements/test/rocm.txt (+1/-1)
LABELS: structured-output, ready, ci/build
BODY: Bump llguidance from the 1.3 minor range to the 1.7 minor range. ⏎  ⏎ This keeps vLLM on a bounded llguidance minor release while allowing newer guidance-compatible dependency stacks to resolve cleanly with vLLM. ⏎  ⏎ This is also needed by vllm-project/vllm-metal: `mlx-vlm>=0.5.0` fixes Qwen3-VL deepstack reference outputs, but it requires `llguidance>=1.7.0`. Without this bump, vllm-metal cannot safely move past `mlx-vlm<0.5.0` without conflicting  …[truncated]

### L3-a7737cb4f3  (L3, 2026-05-14, sha a7737cb4f339, PR #38040)
TITLE: [Fix] Misc Fixes in ViT CUDA Graph (#38040)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/cudagraph/test_encoder_cudagraph.py (+172/-0); vllm/config/compilation.py (+8/-0); vllm/model_executor/models/qwen3_vl.py (+10/-9); vllm/v1/worker/encoder_cudagraph.py (+52/-12)
LABELS: performance, ready, v1, multi-modality, qwen, nvidia
BODY: ## Purpose ⏎  ⏎ 1. Previously `max_batch_size = max_budget // min_budget` could exceed `min_budget`, causing `prepare_encoder_cudagraph_capture_inputs` to compute `per_image_output = token_budget // max_batch_size = 0` for small budgets, leading to a reshape crash on empty tensors in `Qwen3_VisionPatchEmbed.forward`. Fixed by capping to `min(max_budget // min_budget, min_budget)` if both budgets and max batch size are auto-inferred. For the paths w …[truncated]

### L3-ccde9540be  (L3, 2026-05-15, sha ccde9540bed0, PR #42604)
TITLE: DeepSeekV4-Pro enable cuda graph full and piecewise mode (#42604)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse_dsv4.py (+73/-0); vllm/model_executor/layers/mhc.py (+0/-3)
LABELS: rocm, ready, v1, deepseek, gpt-oss, nvidia
BODY: ## Purpose ⏎ Enable "FULL_AND_PIECEWISE" cuda graph mode on Deepseekv4 pro ⏎  ⏎ ## Test Plan ⏎ env ⏎ ``` ⏎ vllm/vllm-openai-rocm:nightly ( a0e392a1cdeb ) ⏎ ``` ⏎  ⏎ server.sh ⏎ ``` ⏎ export HF_HOME=/data/huggingface-cache ⏎ export VLLM_ROCM_USE_AITER=1 ⏎ export VLLM_ROCM_USE_AITER_LINEAR=1 ⏎ export HSA_TOOLS_DISABLE_REGISTER=1 ⏎ export VLLM_DSV4_ROCM_GRAPH_SAFE=1 ⏎  ⏎ vllm serve /dev/shm/models/DeepSeek-V4-Pro \ ⏎   --host localhost \ ⏎   --port 8001 \ ⏎   --dtype a …[truncated]

### L3-491e8d8539  (L3, 2026-05-15, sha 491e8d8539ba, PR #42561)
TITLE: [Perf] Optimize MLA attention `_v_up_proj` bmm by removing additional copy (#42561)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+2/-13)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Originally ⏎  ⏎ ```bash ⏎ x:   (2, 3, 4)   -> transpose -> (3, 2, 4) ⏎ out: (2, 6)      -> view      -> (2, 3, 2) ⏎                   -> transpose -> (3, 2, 2) ⏎  ⏎ bmm: (3, 2, 4) x (3, 4, 2) -> (3, 2, 2) ⏎  ⏎ (3, 2, 2) -> transpose -> (2, 3, 2) -> reshape -> (2, 6) -> copy to output ⏎ ``` ⏎  ⏎ Now ⏎  ⏎ ```bash ⏎ x:   (2, 3, 4)   -> transpose -> (3, 2, 4) ⏎ out: (2, 6)      -> view      -> (2, 3, 2) ⏎  ⏎ bmm write to out.transpose(0, 1) = (3, 2, 2) ⏎  …[truncated]

### L3-0fe7550254  (L3, 2026-05-15, sha 0fe755025467, PR #42692)
TITLE: [Bugfix] DFlash FP8 KV-Cache (#42692)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_dflash.py (+3/-1); vllm/v1/spec_decode/llm_base_proposer.py (+1/-1)
LABELS: bug, speculative-decoding, ready, v1, qwen, dflash
BODY: ## Purpose ⏎  ⏎ There are a few crashes that happen when we run DFlash with FP8 KV Cache, here's why: ⏎  ⏎ - We are not currently propagating the Attention `quant_config` or `cache_config` to the DFlash decoder layer. ⏎ - The `self.token_arange_np` has a dtype mismatch with `self.arange` in the DFlash setup kernel, so I fix the type for consistency. ⏎  ⏎ ## Testing ⏎  ⏎ Currently, there are no supported attention backend combinations with non-causal + FP8 …[truncated]

### L3-f351455f0f  (L3, 2026-05-15, sha f351455f0f06, PR #40119)
TITLE:  [CPU][RISC-V] Add RVV-optimized attention kernels for RISC-V Vector Extension (#40119)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+28/-0); benchmarks/kernels/cpu/benchmark_cpu_attn.py (+9/-11); cmake/cpu_extension.cmake (+3/-1); csrc/cpu/cpu_arch_macros.h (+16/-0); csrc/cpu/cpu_attn.cpp (+4/-0); csrc/cpu/cpu_attn_impl.hpp (+4/-1); csrc/cpu/cpu_attn_rvv.hpp (+445/-0); csrc/cpu/cpu_types_riscv_defs.hpp (+1/-1); csrc/cpu/cpu_types_riscv_impl.hpp (+29/-6); csrc/cpu/generate_cpu_attn_dispatch.py (+31/-4); (+1 more)
LABELS: performance, ready, ci/build, v1, cpu
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Add RISC-V Vector Extension (RVV) optimized attention kernels for vLLM's ⏎   CPU backend, enabling efficient LLM inference on RISC-V platforms. ⏎  ⏎   **Key changes:** ⏎   - Add `cpu_attn_rvv.hpp`: RVV-specific tiled GEMM micro-kernels using  `vfmacc_vf` (scalar-broadcast FMA) with Mx8 tiles and K- ⏎      unroll-by-4 ⏎   - Add RVV ISA dispatch throughout the CPU attention pipeline   (`cpu_attn.cpp`, `cpu_attn_impl.hpp`,   ⏎      generate_cpu_attn_dispat …[truncated]

### L3-b2c58ee942  (L3, 2026-05-15, sha b2c58ee9427f, PR #42685)
TITLE: [FlashAttn] Fix supports_kv_cache_dtype() accepting unhandled fp8 kv-cache dtype variants (#42685)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fa_utils
FILES: vllm/v1/attention/backends/fa_utils.py (+0/-9); vllm/v1/attention/backends/flash_attn.py (+10/-23); vllm/v1/attention/backends/flash_attn_diffkv.py (+3/-5); tests/kernels/attention/test_attention_selector.py (+32/-0); tests/models/quantization/test_fp8.py (+8/-2); tools/pre_commit/generate_attention_backend_docs.py (+1/-13)
LABELS: ready, v1
ISSUES: #42587 [Bug]: vllm serve crashes at engine init for several --kv-cache-dtype values accepted by CLI (fp8_ds_mla, fp8_e5m2, fp8_inc, bfloat16)
BODY: ## Purpose ⏎  ⏎ Fix a regression where `FlashAttentionBackend.supports_kv_cache_dtype()` incorrectly claimed support for kv-cache dtype variants it cannot handle (`fp8_e5m2`, `fp8_ds_mla`, `fp8_inc`, `nvfp4`, `fp8_per_token_head`, `int8_per_token_head`). ⏎  ⏎ `supports_kv_cache_dtype()` returned `True` for any `is_quantized_kv_cache()` type whenever `flash_attn_supports_fp8()` was `True` (H100/A100 hardware). This caused FLASH_ATTN (the highest-prior …[truncated]

### L3-852f567444  (L3, 2026-05-15, sha 852f567444cf, PR #42782)
TITLE: [Bugfix] Respect explicit --kv-cache-dtype over checkpoint kv_cache_scheme (#42782)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/attention.py (+7/-2)
LABELS: bug, ready, quantization
DEEP_STUDY: deep-study correctness case vllm:852f567444: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎  ⏎ When a compressed-tensors checkpoint declares a kv_cache_scheme, the per-layer Attention init was unconditionally forcing kv_cache_dtype to "fp8" and mutating cache_config.cache_dtype, even when the user passed an explicit --kv-cache-dtype on the CLI. That broke user overrides and also cascaded to draft models in speculative decoding (e.g. dflash non-causal attention, which has no FP8 KV backend on Blackwell). ⏎  ⏎ Validated with pool …[truncated]

### L3-be7a03ea65  (L3, 2026-05-15, sha be7a03ea65fa, PR #42409)
TITLE: [ROCm] Widen AITER fused AR RMSNorm 1-stage gate (#42409)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/_aiter_ops.py (+5/-1)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Replace the hardcoded `hidden_dim` allowlist used for AITER fused allreduce+RMSNorm 1-stage dispatch with the kernel's actual pack-size constraint. ⏎  ⏎ Today, vLLM only enables the AITER 1-stage AR+RMS path for this fixed set of hidden sizes: `512, 1024, 2048, 4096, 7168`. ⏎  ⏎ That allowlist is narrower than the kernel constraint and excludes valid model shapes. One concrete example is GPT-OSS, where the relevant hidden size is `2880` …[truncated]

### L3-4d67d3bde2  (L3, 2026-05-15, sha 4d67d3bde25f, PR #42072)
TITLE: [ROCm] Restore fast top_k_per_row kernels for sparse MLA when topk_tokens=2048 (#42072)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+77/-4)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ### Summary ⏎ [#40871](https://github.com/vllm-project/vllm/pull/40871) replaced the C++ `top_k_per_row_decode` / `top_k_per_row_prefill` kernels with a generic `torch.topk`-based fallback (`_topk_indices_torch`) in `rocm_aiter_sparse_attn_indexer_native`, so that non-2048 `topk_tokens` values needed for DeepSeek-V4 would work. The fallback runs an unfused `torch.topk` + `torch.where` + dtype cast over a `[num_tokens, max_model_len]` logits tensor …[truncated]

### L3-ee58665aac  (L3, 2026-05-15, sha ee58665aac60, PR #42135)
TITLE: [Bugfix] Fix DeepGEMM context lens contiguity in MLA indexer (#42135)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+16/-18)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ Fix an MLA indexer crash in the DeepGEMM metadata path: ⏎  ⏎ ```text ⏎ RuntimeError: Assertion error (/workspace/.deps/deepgemm-src/csrc/apis/attention.hpp:201): context_lens.is_contiguous() ⏎ ``` ⏎  ⏎ This can happen in the native MTP/spec decode path when `max_decode_len < next_n`. The MLA indexer used to slice the 2D `decode_seq_lens_buffer` as `[:, :max_decode_len]`, which can produce a non-contiguous `seq_lens` view. ⏎  ⏎ DeepGEMM requ …[truncated]

### L3-46a95815d3  (L3, 2026-05-15, sha 46a95815d344, PR #42509)
TITLE: [ROCm][MLA] FP8 ASM prefill for AITER dense MLA backend on gfx950 (#42509)
SOURCES: path_core, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+369/-0)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎  ⏎ Adds FP8 ASM prefill for the dense AITER MLA backend on gfx950 (MI355X). ⏎  ⏎ When AITER ships `mla_prefill_ps_asm_fwd` + `mla_reduce_v1`, dense MLA prefill batches dispatch through the persistent-scheduling FP8 ASM kernel; otherwise the backend silently falls back to `flash_attn_varlen_func`. No env knob required. ⏎  ⏎ ## Changes ⏎  ⏎ Single file: `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` (+337 lines, 0 deletions). ⏎  ⏎ - `_fp8_ml …[truncated]

### L3-0867497368  (L3, 2026-05-16, sha 0867497368f3, PR #41711)
TITLE: [CI/Build] Bump flashinfer to v0.6.11.post2 (#41711)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); docker/Dockerfile.nightly_torch (+2/-2); docker/versions.json (+1/-1); requirements/cuda.txt (+2/-2); tests/kernels/moe/test_cutedsl_moe.py (+6/-2); vllm/model_executor/layers/fused_moe/oracle/mxfp4.py (+18/-15)
LABELS: ready, ci/build, nvidia, ready-run-all-tests
BODY: ## Purpose ⏎  ⏎ * Bump FlashInfer from `v0.6.8.post1` to `v0.6.11.post2`. ⏎ * ~Adjust installation to use `flashinfer-python[cu13]` extra for cu13 users.~ Upd. Not needed anymore. PR #42438 fixed that issue ⏎ * Fix tensor shapes in `tests/kernels/moe/test_cutedsl_moe.py` ⏎  ⏎ --- ⏎ [details omitted]

### L3-504a26ce2b  (L3, 2026-05-16, sha 504a26ce2be2, PR #41680)
TITLE: Support bf16 for mamba ssm cache (#41680)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/cache.py (+1/-1)
LABELS: ready
BODY: Will be used in tpu-inference ⏎  ⏎ ## Purpose ⏎ Adding the option to choose bf16 for mamba ssm cache. This will be consumed by tpu-inference and other use cases. ⏎  ⏎ ## Test Plan ⏎ Validate in tpu-inference. ⏎  ⏎ ## Test Result ⏎ Validated in tpu-inference

### L3-8a56da3845  (L3, 2026-05-16, sha 8a56da384527, PR #42304)
TITLE: [Experimental] Breakable CUDA graph (#42304)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.common_v1, L3.mla.rocm_aiter_sparse
FILES: vllm/model_executor/layers/attention/mla_attention.py (+2/-0); vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+2/-0); .buildkite/test_areas/cuda.yaml (+2/-1); tests/v1/cudagraph/test_breakable_cudagraph.py (+367/-0); vllm/compilation/breakable_cudagraph.py (+424/-0); vllm/config/vllm.py (+12/-1); vllm/envs.py (+5/-0); vllm/model_executor/layers/deepseek_v4_attention.py (+2/-0); vllm/model_executor/layers/sparse_attn_indexer.py (+2/-0); vllm/v1/cudagraph_dispatcher.py (+5/-0); (+1 more)
LABELS: rocm, intel-gpu, ci/build, v1, deepseek, nvidia, ready-run-all-tests
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Remove the piecewise cudagraph dependency of torch compile ⏎  ⏎ How to enable: set `VLLM_USE_BREAKABLE_CUDAGRAPH=1` ⏎  ⏎ ~~**Note: it has startup time regression when enabling breakable cudagraph and torch.compile**~~ ⏎  ⏎  ⏎ --- ⏎  ⏎ **update in 2026.05.16** ⏎  ⏎ - remove the breakable CG support for other backends, only keep it for deepseek models ⏎ - make breakable cudagraph and torch.compile fullgraph exclusive. No dynamo and inductor invol …[truncated]

### L3-599e75f432  (L3, 2026-05-17, sha 599e75f432e5, PR #42810)
TITLE: [ROCm] [Bugfix] Fix DeepSeek V4 Functionality and Accuracy (#42810)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+33/-114); vllm/model_executor/layers/mhc.py (+48/-40); vllm/model_executor/layers/sparse_attn_indexer.py (+5/-22); vllm/model_executor/models/deepseek_v4.py (+2/-1)
LABELS: bug, rocm, ready, v1, deepseek
BODY: ## Purpose ⏎  ⏎ This PR addresses the following issues ⏎  ⏎ 1. Functionality broken after this PR https://github.com/vllm-project/vllm/pull/41263 (fixed by removing `x = self.ffn_norm(x)`) ⏎ 2. Fix the DeepSeek V4 accuracy degradation at high concurrency. ⏎  ⏎ I have added self-review comments to explain the changes. ⏎  ⏎ ### Point 2 ⏎ It is a compounded issue. ⏎ 1. AITER MHC has accuracy issue when number of tokens is large. New AITER version is coming soo …[truncated]

### L3-990f49bdcb  (L3, 2026-05-17, sha 990f49bdcb8f, PR #42224)
TITLE: [MM][CG] Enable encoder Cudagraph for Step3VL (#42224)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/design/cuda_graphs_multimodal.md (+2/-0); examples/generate/multimodal/vision_language_offline.py (+1/-0); tests/models/multimodal/generation/test_vit_cudagraph.py (+12/-0); vllm/model_executor/models/interfaces.py (+21/-0); vllm/model_executor/models/step3_vl.py (+323/-2); vllm/model_executor/models/step_vl.py (+1/-0); vllm/model_executor/models/utils.py (+16/-0); vllm/v1/worker/encoder_cudagraph.py (+8/-20)
LABELS: documentation, ready, v1, multi-modality, nvidia, verified
BODY: ## Progress ⏎  ⏎  ⏎ ## Test Plan ⏎ Engin serve command  ⏎ ``` ⏎ vllm serve stepfun-ai/Step3-VL-10B \ ⏎     --max-model-len 8192 \ ⏎     --max-num-batched-tokens 8192 \ ⏎     --distributed-executor-backend uni \ ⏎     --limit-mm-per-prompt '{"image": 2, "video": 0}' \ ⏎     --dtype bfloat16 \ ⏎     --no-enable-prefix-caching \ ⏎     --compilation-config '{"cudagraph_mm_encoder": true, ⏎ 	"encoder_cudagraph_token_budgets": [338, 1536],  ⏎ 	"encoder_cudagraph_max_v …[truncated]

### L3-998714b21b  (L3, 2026-05-18, sha 998714b21b41, PR #42849)
TITLE: [Perf] Add do_not_specialize in fused FP8 RoPE kernel (#42849)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py (+1/-1)
LABELS: performance, ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ This PR add `do_not_specialize` in `_fused_inv_rope_fp8_quant_per_head` kernel, to prevent unnecessary recompilation of the kernel during inference when num_tokens are seen for the first time. ⏎  ⏎ e2e TTFT improves 9% on H200. ⏎  ⏎ ## Benchmark ⏎  ⏎ ``` ⏎ rm -rf ~/.triton/cache ⏎  ⏎ vllm serve deepseek-ai/DeepSeek-V4-Flash \ ⏎   --tensor-parallel-size 8 \ ⏎   --max-num-seqs 8 \ ⏎   --kv-cache-dtype fp8 \ ⏎   --block-size 256 \ ⏎   --no-enable-fl …[truncated]

### L3-b4601ad43f  (L3, 2026-05-18, sha b4601ad43ff7, PR #42707)
TITLE: [CPU] Add fused GDN support for AMX CPU platform (#42707)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/cpu/sgl-kernels/conv.cpp (+21/-10); csrc/cpu/sgl-kernels/fla.cpp (+5/-1); csrc/cpu/torch_bindings.cpp (+80/-0); vllm/_custom_ops.py (+129/-0); vllm/model_executor/layers/linear.py (+0/-3); vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py (+140/-0); vllm/model_executor/layers/utils.py (+13/-0); vllm/platforms/cpu.py (+19/-0)
LABELS: ready, cpu
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Add fused GDN support for AMX CPU platform ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-cac81b6eda  (L3, 2026-05-18, sha cac81b6eda41, PR #42666)
TITLE: [CPU Backend] Improve cpu thread utilization (#42666)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/cpu/cpu_attn_impl.hpp (+1/-1); vllm/utils/ompmultiprocessing.py (+3/-5)
LABELS: ready, cpu
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ 1. Reduce attention work per thread to allow better utilization of arbitrary number of threads (e.g.: 31) ⏎ 2. Tune KMP-related env variables. KMP_BLOCKTIME=1 is short and causes threads to go to sleep too quickly. BARRIER_PATTERN settings are more beneficial when OMP regions cross numa nodes, which is not the recommended/default behavior. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm bench throughput \ ⏎ 	--model RedHatAI/Meta-Llama-3.1-8B-Instruct-quant …[truncated]

### L3-88a860d754  (L3, 2026-05-18, sha 88a860d7545a, PR #41922)
TITLE: [CPU] Add MXFP4 W4A16 MoE support (#41922)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/hardware_tests/cpu.yaml (+3/-3); csrc/cpu/sgl-kernels/common.h (+7/-0); csrc/cpu/sgl-kernels/gemm.h (+49/-9); csrc/cpu/sgl-kernels/gemm_fp8.cpp (+83/-3); csrc/cpu/sgl-kernels/moe.cpp (+78/-9); csrc/cpu/sgl-kernels/moe.h (+106/-0); csrc/cpu/sgl-kernels/moe_fp8.cpp (+92/-56); csrc/cpu/sgl-kernels/vec.h (+15/-2); csrc/cpu/torch_bindings.cpp (+12/-2); tests/kernels/moe/test_cpu_fp8_fused_moe.py (+0/-256); (+6 more)
LABELS: ready, ci/build, cpu, verified
BODY: ## Purpose ⏎ Adapted from https://github.com/sgl-project/sglang/pull/16775 ⏎ - Sync csrc/cpu/sgl-kernels/ with upstream SGLang sgl-kernel.  ⏎ - Add MXFP4 W4A16 fused MoE experts for CPU, targeting openai/gpt-oss-20b ⏎  ⏎  ⏎ ## Test Plan ⏎ python -m pytest tests/kernels/moe/test_cpu_mxfp4_fused_moe.py -v ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-b50646e5ef  (L3, 2026-05-18, sha b50646e5effd, PR #42909)
TITLE: [ROCm][CI] Stabilize ROCm pooling and multimodal CI (#42909)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/language/pooling/test_gritlm.py (+9/-3); tests/models/language/pooling/test_max_tokens_per_doc.py (+31/-12); tests/models/multimodal/generation/test_qwen2_5_vl.py (+3/-2); vllm/model_executor/models/transformers/base.py (+8/-0)
LABELS: rocm, ready, multi-modality, qwen
DEEP_STUDY: deep-study: this PR was reverted by PR 42923 (partial_revert, reason=premature_or_process)
BODY: Addresses two ROCm CI groups that were failing or exposing cache-dependent behavior: ⏎  ⏎ - `AMD: Language Models Test (Extended Pooling) (mi300_1)` ⏎ - `AMD: Multi-Modal Models (Standard) 2: qwen3 + gemma (mi300_1)` ⏎  ⏎ The fixes keep model correctness checks intact. They avoid pinning model revisions where cache refreshes can legitimately change tokenizer/checkpoint details, and they make ROCm test setup explicit where defaults previously let unrel …[truncated]

### L3-8c296de63b  (L3, 2026-05-18, sha 8c296de63b47, PR #42857)
TITLE: [Perf] Re-enable flashinfer autotune by default and cleanup (#42857)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/config/vllm.py (+2/-6); vllm/utils/flashinfer.py (+0/-1); vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py (+18/-21); vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py (+31/-37); vllm/model_executor/warmup/kernel_warmup.py (+61/-15)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ This PR re-enables flashinfer autotune by default as previous correctness issues are now fixed: https://github.com/flashinfer-ai/flashinfer/pull/3227.  ⏎  ⏎ In addition, did some cleanup: ⏎ - Remove `_is_fi_autotuning` wrapper as not longer needed. ⏎ - Make autotuning done on rank 0 only, and the chosen tactics are broadcasted to other ranks, ensuring all ranks running the same tactics. ⏎  ⏎ ## Test Plan ⏎ - GSM8k on Deepseek v4 TP, TEP, DEP …[truncated]

### L3-00e20e76f7  (L3, 2026-05-18, sha 00e20e76f775, PR #42767)
TITLE: [Refactor] Remove dead cuda kernels (#42767)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: csrc/attention/vertical_slash_index.cu (+0/-401); CMakeLists.txt (+0/-1); csrc/moe/torch_bindings.cpp (+0/-10); csrc/ops.h (+0/-26); csrc/torch_bindings.cpp (+0/-24); vllm/_custom_ops.py (+1/-143)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ Remove cuda dead kernels ⏎  ⏎ - `marlin_gemm_moe` ⏎ - `convert_vertical_slash_indexes` ⏎ - `convert_vertical_slash_indexes_mergehead`

### L3-8fc1c284b9  (L3, 2026-05-18, sha 8fc1c284b946, PR #42880)
TITLE: [ROCm] Guard AITER GDN decode fast path by layout (#42880)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/gdn_linear_attn.py (+8/-4)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ Fix the ROCm Qwen3.5 accuracy regression apparently introduced by #40711 by skipping the optional AITER Triton GDN decode fast-path for non-interleaved GDN projection layouts. ⏎  ⏎ The fused AITER decode path added in #40711 is intended for the Qwen3-Next interleaved GQA layout. Qwen3.5 uses a non-interleaved `[q, k, v, z]` and `[b, a]` projection layout, so sending those packed tensors directly to the fused reshape kernel can read th …[truncated]
