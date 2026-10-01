### L3-013b54088c  (L3, 2026-01-02, sha 013b54088c2b, PR #31612)
TITLE: [ROCm][CI] Fix ModernBERT token classification test (#31612)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/language/pooling/test_token_classification.py (+18/-7)
LABELS: rocm, ready
BODY: Fixes ModernBERT token classification test failure on ROCm by using eager attention for HuggingFace inference. ⏎  ⏎ ## Problem ⏎  ⏎ On ROCm, HuggingFace Transformers' flash attention implementation produces numerically different results compared to vLLM's FlexAttention backend, causing the `test_modernbert_models` test to fail with `atol=1e-2`. ⏎  ⏎ This is a known HF Transformers accuracy issue on ROCm (see [#30167](https://github.com/vllm-project/vllm/issu …[truncated]

### L3-a3f2f40947  (L3, 2026-01-02, sha a3f2f40947cf, PR #31504)
TITLE: [MoE Refactor] Explicit construct mk for flashinfer bf16 kernel (#31504)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py (+32/-36)
LABELS: ready
BODY: Explicitly construct modular kernels for flashinfer kernel and make it compatible with `self.kernel` calling convention.  ⏎  ⏎ ## Testing ⏎ Test `gsm8k` eval before and after the change ⏎  ⏎ Launch command ⏎ ``` ⏎ VLLM_USE_FLASHINFER_MOE_FP16=1 vllm serve Qwen/Qwen3-30B-A3B -tp 4 --enable-expert-parallel ⏎ ``` ⏎  ⏎ Benchmark command ⏎ ``` ⏎     lm_eval \ ⏎             --model local-completions \ ⏎             --tasks gsm8k \ ⏎             --model_args "model=Qwen/Qwen3-30B-A3 …[truncated]

### L3-6ef770df7c  (L3, 2026-01-02, sha 6ef770df7c3f, PR #31596)
TITLE: [MoE] Fix output_shape calculation in Attention layer to handle 3D query inputs (#31596)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+4/-1); vllm/model_executor/layers/quantization/fp8.py (+13/-1)
LABELS: rocm, ready
BODY: Fixes a regression introduced in [#28775](https://github.com/vllm-project/vllm/pull/28775) where the `output_shape` calculation in `Attention.forward()` assumes 2D query input, causing failures when models pass 3D query tensors. ⏎  ⏎ ## Problem ⏎  ⏎ The new code added in #28775: ⏎ ```python ⏎ if output_shape is None: ⏎     output_shape = torch.Size((*query.shape[:-1], self.num_heads * self.head_size_v)) ⏎ ``` ⏎  ⏎ This breaks when `query` is 3D `[num_tokens, num_hea …[truncated]

### L3-268b1c55ad  (L3, 2026-01-03, sha 268b1c55ade3, PR #31533)
TITLE: [MoE Refactor][13/N] Convert FI to Use PFNoEP (#31533)
SOURCES: corpus:kernel-correctness-cases(introducing), body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/all2all_utils.py (+13/-3); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py (+17/-5); vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py (+0/-1); vllm/model_executor/layers/fused_moe/layer.py (+2/-2); vllm/model_executor/layers/fused_moe/modular_kernel.py (+5/-75); vllm/model_executor/layers/quantization/fp8.py (+40/-73); vllm/model_executor/layers/quantization/modelopt.py (+1/-9)
LABELS: ready, llama, nvidia
ISSUES: #31609 [Bug][ModelOpt]: FlashInfer CUTLASS MoE Accuracy Degraded (Llama4)
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:f6c0009afa (fix PR 31742) || deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ * flashinfer has custom implementation of prepare finalize for the tp case, convert to use the standard version ⏎ * as a result, we no longer have to run shared expert stream overlap inside the `mk.ModularKernelMethod` (#28879). This is because now in the TP case we no longer use `mk.ModularKernelMethod`, simplifying the code ⏎ * this is also a performance optimization, since the implementation of shared expert overlap in the non-mk case i …[truncated]

### L3-367856de14  (L3, 2026-01-05, sha 367856de1402, PR #31665)
TITLE: [CI/Build] Revive skipped reward models e2e test (#31665)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/fixtures/qwen2_5_math_prm_reward_step.json (+1/-0); tests/models/language/pooling/test_reward.py (+63/-4)
LABELS: ready
BODY: ## Purpose ⏎ - Reward models test have been skipped since transformers v4.53 update, causing that reward models test missing for a long time. ⏎ - This PR revive the broken test to use gold fixture from `hf_outputs` to make sure that we have at least one working e2e test for reward entrypoint. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest -s -v tests/models/language/pooling/test_reward.py ⏎ ``` ⏎  ⏎ ## Test Result ⏎ ``` ⏎ tests/models/language/pooling/test_reward.py::test_prm_model …[truncated]

### L3-d8e38d4939  (L3, 2026-01-05, sha d8e38d493907, PR #30687)
TITLE: Triton Attention: Support cross-layers blocks (#30687)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+13/-0); tests/v1/kv_offload/test_cpu_offloading.py (+1/-3)
LABELS: ready, v1
BODY: This PR enables cross-layer blocks for the triton attention backend. ⏎ An accuracy test is added to the CPU KV offloading test.

### L3-f6c0009afa  (L3, 2026-01-05, sha f6c0009afa36, PR #31742)
TITLE: [Bugfix] Fix Broken ModelOpt NVFP4 MoE (#31742)
SOURCES: corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/all2all_utils.py (+5/-13); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+3/-1); vllm/model_executor/layers/quantization/fp8.py (+9/-1); vllm/model_executor/layers/quantization/modelopt.py (+15/-0)
LABELS: bug, ready, llama, nvidia
DEEP_STUDY: deep-study correctness case vllm:f6c0009afa: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=#31533
BODY: ## Purpose ⏎ - https://github.com/vllm-project/vllm/pull/31533 broke nvfp4 modelopt models ⏎ - this fixes it ⏎  ⏎ ## Test Plan ⏎  ⏎ ```bash ⏎ pytest -s -v -x tests/quantization/test_blackwell_moe.py -k test_llama4_nvfp4_moe_flashinfer_cutlass ⏎ ``` ⏎  ⏎ ```bash ⏎ # modelopt ⏎ MODEL_NVFP4 := "nvidia/Qwen3-30B-A3B-NVFP4" ⏎ MODEL_TENSOR := "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8" ⏎  ⏎ GPUS := "1" ⏎ PORT := "8000" ⏎  ⏎ launch_nvfp4: ⏎ 	VLLM_USE_FLASHINFER_MOE_FP4=0 VLLM_FLASHINFER_MOE …[truncated]

### L3-6aa5b18e1d  (L3, 2026-01-06, sha 6aa5b18e1dfd, PR #31406)
TITLE: [v1] Add encoder-only/cross attention support to Triton Attention backend (#31406)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.triton.v1_backend, L3.platform.rocm_selection
FILES: vllm/attention/ops/triton_prefill_attention.py (+271/-0); vllm/platforms/rocm.py (+0/-9); vllm/v1/attention/backends/triton_attn.py (+73/-4); tests/kernels/attention/test_triton_prefill_attention.py (+225/-0); tests/models/multimodal/generation/test_whisper.py (+1/-1); tests/v1/attention/test_attention_backends.py (+57/-0)
LABELS: rocm, ready, v1, multi-modality
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ ### Motivation ⏎ 1. We don't have Whisper support for Turing/Volta GPUs because of FA's cc limit. ⏎ 2. Furthermore, FlexAttention's encoder-only attention's speed is still not ideal for FP32. ⏎ 3. After xformers deprecation (https://github.com/vllm-project/vllm/pull/29262), we want to add a Triton MMEncoderAttention backend to give a balanced solution between FA and SDPA for incompatable `head_size`. ⏎  ⏎ ### Introduction ⏎ - This PR introduced an …[truncated]

### L3-ee2e69d6cd  (L3, 2026-01-06, sha ee2e69d6cda8, PR #31776)
TITLE: [Bugfix][CI/Build] Fix failing pooling models test due to Triton kernel accuracy diff (#31776)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/language/pooling/test_token_classification.py (+1/-1)
LABELS: ready
BODY: ## Purpose ⏎ - Fix https://github.com/vllm-project/vllm/pull/31406#issuecomment-3713125841 ⏎ - Triton kernel has minor accuracy (~0.001) error comparing previous FlexAttention backend ⏎ - Increase `rtol` to fix failing pooling test ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest -s -v tests/models/language/pooling/test_token_classification.py::test_modernbert_models[float-disham993/electrical-ner-ModernBERT-base] ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Can't reproduce locally, hope this make CI …[truncated]

### L3-51e38a8e30  (L3, 2026-01-06, sha 51e38a8e3010, PR #31725)
TITLE: [Misc] Enable Paligemma's PrefixLM attention mask computation (#31725)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_common.py (+0/-4); vllm/config/model.py (+1/-3)
LABELS: ready, multi-modality
BODY: ## Purpose ⏎ - Previously, we disabled Paligemma's PrefixLM attention because FlexAttention doesn't support soft cap ⏎ - After https://github.com/vllm-project/vllm/pull/30386, we can enable it through Triton Attention backend with soft cap now. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ pytest -s -v tests/models/multimodal/generation/test_common.py -k paligemma ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Tests pass now. ⏎  ⏎ --- ⏎ [details omitted]

### L3-e0327c9db2  (L3, 2026-01-06, sha e0327c9db206, PR #31773)
TITLE: [Attention][1/n] Remove usage of deprecated `seq_lens_cpu` and `num_computed_tokens_cpu` CommonAttentionMetadata properties (#31773)
SOURCES: path_core, corpus:kernel-correctness-cases(introducing), body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.mla.common_v1, L3.mla.flashmla_sparse, L3.dispatch.abstract_interface, L3.flex_attention
FILES: vllm/v1/attention/backends/flashinfer.py (+3/-1); vllm/v1/attention/backends/flex_attention.py (+1/-3); vllm/v1/attention/backends/mla/common.py (+3/-1); vllm/v1/attention/backends/mla/flashmla_sparse.py (+1/-1); vllm/v1/attention/backends/triton_attn.py (+1/-1); vllm/v1/attention/backends/utils.py (+9/-0); tests/v1/attention/test_attention_backends.py (+2/-2); tests/v1/attention/test_mla_backends.py (+2/-2); tests/v1/attention/test_sparse_mla_backends.py (+1/-1)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study: this PR was reverted by PR 33771 (partial_revert, reason=performance_regression) || deep-study: introduced the defect fixed in case vllm:179ae7da8f (fix PR 33771)
BODY: Update tests, MLA backends and CUDA full-attention backends

### L3-e9717801bd  (L3, 2026-01-06, sha e9717801bdc5, PR #31714)
TITLE: [Bugfix][ROCm] Fix Unsupported attention metadata type for speculative decoding in `eagle.py` (#31714)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/spec_decode/eagle.py (+6/-2)
LABELS: rocm, speculative-decoding, ready, v1
BODY: ## Purpose ⏎ this PR fixes the issue mentioned here https://github.com/vllm-project/vllm/pull/30811#issuecomment-3668013162 ⏎  ⏎ ## Test Plan ⏎ Only testing VLLM_ATTENTION_BACKEND=ROCM_ATTN ⏎  ⏎ 1. run vllm serve: ⏎ ` ⏎ VLLM_USE_V1=1 ⏎ VLLM_ROCM_USE_AITER=0 ⏎ VLLM_ATTENTION_BACKEND=ROCM_ATTN ⏎ vllm serve meta-llama/Llama-3.3-70B-Instruct ⏎ -tp 4 ⏎ --speculative-config '{"model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B", "num_speculative_tokens": 3, "method":"eagle3", "draft …[truncated]

### L3-7101e0851f  (L3, 2026-01-06, sha 7101e0851f73, PR #31738)
TITLE: [Models]: Use `MMEncoderAttention` for MoonViT (#31738)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/kimi_vl.py (+1/-1); vllm/model_executor/models/moonvit.py (+71/-157)
LABELS: ready
BODY: ## Purpose ⏎ - We missed MoonViT for Kimi-VL in https://github.com/vllm-project/vllm/pull/30125https://github.com/vllm-project/vllm/pull/30125 ⏎ - This PR updates its attention interface, and also adds ViT data parallel support.  ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ python examples/offline_inference/vision_language.py -m kimi_vl ⏎ ``` ⏎  ⏎ ## Test Result ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ --- ⏎ [details omitted]

### L3-799b5721f6  (L3, 2026-01-06, sha 799b5721f6bb, PR #31720)
TITLE: [cpu][bench] Add CPU paged attention benchmarks (#31720)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/cpu/benchmark_cpu_attn.py (+272/-0)
LABELS: performance, ready, cpu
ISSUES: #30374 [Feature][CPU Backend]: Add Paged Attention Benchmarks for CPU backend
BODY: ## Purpose ⏎  ⏎ Add CPU paged attention benchmarks ⏎ Fixes: #30374 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-1fb0209bbc  (L3, 2026-01-06, sha 1fb0209bbc47, PR #31177)
TITLE: [Bugfix][Hardware][AMD] Fix exception types in AITER MLA FP8 check (#31177)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/rocm/aiter/test_mla_fp8_support_check.py (+118/-0); vllm/_aiter_ops.py (+11/-1)
LABELS: rocm, ready
BODY: ## Summary ⏎  ⏎ Replace broad `except Exception:` with specific exception types in `_check_aiter_mla_fp8_support()` to avoid masking unexpected errors during AITER MLA FP8 parameter detection. ⏎  ⏎ ## Changes ⏎  ⏎ **File**: `vllm/_aiter_ops.py` ⏎  ⏎ Before: ⏎ ```python ⏎ except Exception: ⏎     _AITER_MLA_SUPPORTS_FP8 = False ⏎ ``` ⏎  ⏎ After: ⏎ ```python ⏎ except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError): ⏎     # ImportError/ModuleNotFoundError: a …[truncated]

### L3-af8fd73051  (L3, 2026-01-06, sha af8fd7305123, PR #31593)
TITLE: [MoE Refactor][14/N] Clean Up FI Quant Config Smuggling (#31593)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer.py (+12/-7); vllm/model_executor/layers/fused_moe/config.py (+7/-4); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+4/-4); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py (+4/-3); vllm/model_executor/layers/quantization/fp8.py (+74/-10); vllm/model_executor/layers/quantization/modelopt.py (+38/-13); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+34/-43)
LABELS: ready, llama, nvidia
BODY: ## Purpose ⏎ * moe refactor --- flashinfer smuggles scales for certain kernels via nvpf4 global scales --- clean up ⏎ * potentially fixes issues with flashinfer per tensor for non-modelopt --- it avoids crashing mixtral but we still have 0% accuracy on mixtral. Will address this in another PR. ⏎  ⏎ ## Test Plan ⏎  ⏎ ```bash ⏎ # autofp8 ⏎ MODEL_BLOCK := "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8" ⏎ # MODEL_TENSOR := "amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV" ⏎  ⏎ # modelopt ⏎ M …[truncated]

### L3-02809af1e7  (L3, 2026-01-06, sha 02809af1e7cd, PR #31806)
TITLE: [Bugfix]: Fix cross attention backend selection for Turing GPU (#31806)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/attention/layers/cross_attention.py (+9/-5)
LABELS: ready
BODY: ## Purpose ⏎ - The default Attention backend for Turing is FlashInfer, so cross attn initialization will encounter error due to missing `attn_type=AttentionType.ENCODER_DECODER` when calling `get_attn_backend`: ⏎ ``` ⏎ (EngineCore_DP0 pid=185092) ERROR 01-06 12:32:35 [core.py:888]   File "/kaggle/working/vllm/vllm/model_executor/models/whisper.py", line 577, in <lambda> ⏎ (EngineCore_DP0 pid=185092) ERROR 01-06 12:32:35 [core.py:888]     lambda prefix: W …[truncated]

### L3-9a1d20a89c  (L3, 2026-01-07, sha 9a1d20a89c3b, PR #31183)
TITLE: [CI] Add warmup run in test_fusion_attn (#31183)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/test_fusion_attn.py (+16/-8)
LABELS: ready
ISSUES: #31044 [CI Failure]: Blackwell Fusion Tests
BODY: `tests/compile/test_fusion_attn.py::test_attention_quant_pattern` seems to be failing in CI. This is sort of difficult to repro but I was finally able to repro it when running this command: ⏎ ``` ⏎ TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 pytest -sv "tests/compile/test_fusion_attn.py::test_attention_quant_pattern[AttentionBackendEnum.TRITON_ATTN-nvidia/Llama-4-Scout-17B-16E-Instruct-FP8-TestAttentionFp8StaticQuantPatternModel--quant_fp8-dtype0-256-128-64 …[truncated]

### L3-0a2c2dc3f1  (L3, 2026-01-07, sha 0a2c2dc3f146, PR #31465)
TITLE: fixed mypy warnings for files vllm/v1/attention with TEMPORARY workaround (#31465)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.flashattn, L3.mla.flashmla_sparse, L3.mla.rocm_aiter, L3.mla.aiter_triton, L3.mla.rocm_aiter_sparse, L3.dispatch.abstract_interface, L3.flex_attention, L3.tree_attention
FILES: vllm/attention/backends/abstract.py (+9/-3); vllm/model_executor/layers/attention_layer_base.py (+3/-1); vllm/v1/attention/backends/flash_attn.py (+34/-8); vllm/v1/attention/backends/flash_attn_diffkv.py (+9/-1); vllm/v1/attention/backends/flashinfer.py (+6/-5); vllm/v1/attention/backends/flex_attention.py (+1/-1); vllm/v1/attention/backends/mla/aiter_triton_mla.py (+1/-1); vllm/v1/attention/backends/mla/common.py (+21/-12); vllm/v1/attention/backends/mla/flashattn_mla.py (+9/-2); vllm/v1/attention/backends/mla/flashmla_sparse.py (+4/-2); (+8 more)
LABELS: rocm, ready, v1, nvidia
BODY: ## Purpose ⏎ Follow-up to https://github.com/vllm-project/vllm/pull/26448 ⏎  ⏎ This PR improves type checking coverage for the attention module by: ⏎ - Moving `vllm/v1/attention` from SEPARATE_GROUPS to FILES in mypy configuration ⏎ - Fixing type errors in the attention module to enable strict type checking ⏎ - This ensures better code quality and catches type-related bugs earlier ⏎  ⏎ This is part of the effort to enable comprehensive type checking across the v …[truncated]

### L3-6409004b26  (L3, 2026-01-07, sha 6409004b2656, PR #31816)
TITLE: [ROCm][AITER] bugfix accuracy regression in ROCM_AITER_TRITON_MLA backend (#31816)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.aiter_triton
FILES: vllm/v1/attention/backends/mla/aiter_triton_mla.py (+2/-10)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎ Running deepseek using the AITER_TRITON_MLA backend results in low accuracy as follow: ⏎ command: ⏎ ` ⏎ export VLLM_USE_V1=1 ⏎ export SAFETENSORS_FAST_GPU=1 ⏎ export VLLM_ROCM_USE_AITER=1 ⏎ export VLLM_ATTENTION_BACKEND=ROCM_AITER_TRITON_MLA ⏎ vllm serve deepseek-ai/DeepSeek-V3 -tp 8  ⏎ ` ⏎ lm_eval score: ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ | …[truncated]

### L3-59fe6f298e  (L3, 2026-01-07, sha 59fe6f298e16, PR #31762)
TITLE: [XPU]fallback to TRITON_ATTN on xpu when use float32 dtype (#31762)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/platforms/xpu.py (+7/-0)
LABELS: ready, ci/build, v1
BODY: ## Purpose ⏎ We found that the XPU kernel does not support FLASH_ATTN with FP32 , so we designed a fallback mechanism to use TRITON_ATTN when testing with FLASH_ATTN with FP32 on the XPU platform. ⏎ ## Test Plan ⏎ cd tests ⏎ pytest -sv v1/entrypoints/openai/test_completion.py ⏎ ## Test Result ⏎ == 38 passed, 2 warnings in 93.52s (0:01:33) == ⏎ --- ⏎ [details omitted]

### L3-0790f07695  (L3, 2026-01-07, sha 0790f07695c7, PR #30593)
TITLE: [Misc] Improve error messages for unsupported types and parameters (#30593)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+4/-1); benchmarks/cutlass_benchmarks/sparse_benchmarks.py (+3/-1); vllm/config/lora.py (+1/-1); vllm/distributed/device_communicators/pynccl_wrapper.py (+4/-1); vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py (+4/-1); vllm/model_executor/layers/quantization/auto_round.py (+6/-6); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py (+4/-1); vllm/model_executor/layers/quantization/mxfp4.py (+4/-1); vllm/model_executor/models/ernie45_vl.py (+5/-1); vllm/model_executor/models/granite_speech.py (+3/-1); (+1 more)
LABELS: performance, ready, kv-connector, nvidia
BODY: ## Purpose ⏎ Improve developer experience by clarifying an existing error message for an invalid configuration. ⏎ The updated message explicitly includes the relevant parameter names, the invalid value and the expected constraint, making the failure easier to understand and fix without changing runtime behavior. ⏎  ⏎ ## Test ⏎ No behavior change. Existing tests passed. ⏎  ⏎ --- ⏎ [details omitted]

### L3-05f47bd8d2  (L3, 2026-01-07, sha 05f47bd8d2f3, PR #31868)
TITLE: [Doc] Fix: Correct vLLM announcing blog post link in docs (#31868)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/README.md (+1/-1)
LABELS: documentation
BODY: ## Purpose ⏎ Fix incorrect link for the vLLM announcing blog post in the documentation homepage. ⏎ The link was pointing to https://vllm.ai (the main website) instead of the actual blog post at https://blog.vllm.ai/2023/06/20/vllm.html which introduces PagedAttention. ⏎ ## Test Plan ⏎ No testing required for documentation link fixes. The change can be verified by: ⏎ 1. Visiting the documentation at https://docs.vllm.ai ⏎ 2. Clicking on the "vLLM announcing b …[truncated]

### L3-cc6dafaef2  (L3, 2026-01-07, sha cc6dafaef2bc, PR #29213)
TITLE: [Perf][Kernels] Enable FlashInfer DeepGEMM swapAB on SM90 (for W8A8 Linear Op) (#29213)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+57/-0); tests/kernels/quantization/test_block_fp8.py (+51/-0); vllm/envs.py (+6/-0); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+143/-1)
LABELS: documentation, performance, ready, v1, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ### **Purpose** ⏎  ⏎ Per https://github.com/vllm-project/vllm/issues/28427, https://github.com/vllm-project/vllm/issues/28316, TRTLLM has `fp8_gemm_kernel_swapAB` kernel for blockscale FP8 gemm in linear layer. This PR https://github.com/vllm-project/vllm/pull/29213 (and FlashInfer PR https://github.com/flashinfer-ai/flashinfer/pull/2131) brings the gemm kernel and quant kernel (to get FP8 scaling) in VLLM. ⏎  ⏎ Description: ⏎ M < 32 uses DeepGEMM swapAB f …[truncated]

### L3-41cfa50632  (L3, 2026-01-07, sha 41cfa50632c2, PR #31880)
TITLE: [ROCm][AITER] fix wrong argument passed to  AITER `flash_attn_varlen_func` (#31880)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter, L3.mla.aiter_triton
FILES: vllm/v1/attention/backends/mla/aiter_triton_mla.py (+1/-1); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+1/-1)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎ This https://github.com/vllm-project/vllm/pull/31465 PR changed arguments name into wrong one that doesn't exist in `aiter package`. ⏎ This PR fixes the issue. ⏎  ⏎ ## Test Plan ⏎ commands: ⏎  ⏎ ROCM_AITER_MLA backend ⏎  ⏎ ` ⏎ export VLLM_USE_V1=1 ⏎ export SAFETENSORS_FAST_GPU=1 ⏎ export VLLM_ROCM_USE_AITER=1 ⏎ export VLLM_ATTENTION_BACKEND=ROCM_AITER_MLA ⏎ vllm serve deepseek-ai/DeepSeek-V3 \ ⏎ --block-size 128 \ ⏎ -tp 8 > ⏎ ` ⏎  ⏎ ROCM_AITER_TRITON_MLA  ⏎ ` ⏎ export VLLM_U …[truncated]

### L3-c7a79d41a0  (L3, 2026-01-07, sha c7a79d41a03f, PR #31850)
TITLE: [Attention][3/n] Remove usage of deprecated `seq_lens_cpu` and `num_computed_tokens_cpu` CommonAttentionMetadata properties (#31850)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+2/-2); vllm/v1/attention/backends/rocm_attn.py (+1/-1)
LABELS: rocm, ready, v1
BODY: Update rocm backends

### L3-b665bbc2d4  (L3, 2026-01-07, sha b665bbc2d427, PR #31891)
TITLE: [Chore] Migrate V0 attention utils (#31891)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_sparse, L3.mla.rocm_aiter_sparse
FILES: vllm/attention/backends/utils.py (+0/-33); vllm/v1/attention/backends/mla/common.py (+22/-2); vllm/v1/attention/backends/mla/flashmla_sparse.py (+1/-2); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+1/-4); tests/kernels/mamba/test_causal_conv1d.py (+1/-1); tests/kernels/mamba/test_mamba_ssm.py (+1/-1); vllm/model_executor/layers/mamba/ops/causal_conv1d.py (+1/-1); vllm/model_executor/layers/mamba/ops/mamba_ssm.py (+1/-1); vllm/v1/attention/backends/gdn_attn.py (+1/-1); vllm/v1/worker/gpu/block_table.py (+1/-1)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎  ⏎ Refactoring to remove V0 code ⏎  ⏎ - `vllm.attention.backends.utils.PAD_SLOT_ID -> vllm.v1.attention.backends.utils.PAD_SLOT_ID` (existing) ⏎ - `vllm.attention.backends.utils.get_mla_dims -> vllm.v1.attention.backends.mla.common.get_mla_dims` ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e7596371a4  (L3, 2026-01-07, sha e7596371a403, PR #30808)
TITLE: [Refactor][TPU] Remove torch_xla path and use tpu-inference (#30808)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.dispatch.registry
FILES: vllm/attention/backends/registry.py (+0/-1); vllm/attention/layers/mm_encoder_attention.py (+0/-25); docs/design/moe_kernel_features.md (+0/-1); tests/tpu/__init__.py (+0/-0); tests/tpu/lora/__init__.py (+0/-0); tests/tpu/lora/test_lora.py (+0/-139); tests/tpu/test_compilation.py (+0/-86); tests/tpu/test_custom_dispatcher.py (+0/-34); tests/tpu/test_moe_pallas.py (+0/-88); tests/tpu/test_quantization_accuracy.py (+0/-52); (+36 more)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, tpu, speculative-decoding, ready, ci/build
BODY: ## Purpose ⏎ Removes torch_xla related code paths as this backend is now deprecated. To run vLLM on TPU, users should now install and use tpu-inference. ⏎  ⏎ ## Test Plan ⏎ - Triggered tpu-inference CI/CD pipeline. ⏎ - Verified that removal does not impact non-TPU backends. ⏎  ⏎ ## Test Result ⏎ - tpu-inference CI/CD: https://buildkite.com/tpu-commons/tpu-inference-ci/builds/71 ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted]

### L3-30399cc725  (L3, 2026-01-07, sha 30399cc72530, PR #31899)
TITLE: UX: add vLLM env info in '/server_info' (#31899)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/entrypoints/serve/instrumentator/server_info.py (+19/-3)
LABELS: frontend, ready
BODY: ## Purpose ⏎  ⏎ ```bash ⏎ VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \ ⏎      --served-model-name base-model  ⏎ ``` ⏎ ```python ⏎ import requests ⏎  ⏎ respones = requests.get("http://0.0.0.0:8000/server_info") ⏎ print(respones.json()) ⏎ ``` ⏎  ⏎ [details omitted] ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-be6a81f31b  (L3, 2026-01-07, sha be6a81f31b04, PR #30460)
TITLE: [chore] Update FA commit (#30460)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
BODY: Update vLLM-FA commit to include: ⏎  ⏎ https://github.com/vllm-project/flash-attention/pull/110 ⏎ https://github.com/vllm-project/flash-attention/pull/112

### L3-0d7667419f  (L3, 2026-01-08, sha 0d7667419f73, PR #31924)
TITLE: [0/N][Attention] Fix miscellaneous pre-commit issues (#31924)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.flash_attn.fa_utils, L3.triton.prefix_prefill, L3.triton.decode_attention, L3.mla.flashmla_v0_adapter
FILES: vllm/attention/layers/static_sink_attention.py (+1/-1); vllm/attention/ops/flashmla.py (+5/-2); vllm/attention/ops/paged_attn.py (+6/-2); vllm/attention/ops/prefix_prefill.py (+2/-2); vllm/attention/ops/rocm_aiter_mla_sparse.py (+3/-3); vllm/attention/ops/triton_decode_attention.py (+1/-4); vllm/attention/ops/triton_prefill_attention.py (+3/-3); vllm/attention/utils/fa_utils.py (+14/-8)
LABELS: rocm, ready
BODY: ## Purpose ⏎ CI on #31916 surfaced some preexisting pre-commit issues. This PR fixes them. ⏎  ⏎ ## Test Plan ⏎ Should pass pre-commit in CI ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-2972a05473  (L3, 2026-01-08, sha 2972a05473d6, PR #31950)
TITLE: [MM Encoder]: Make MMEncoderAttention's `scale` takes effect properly  (#31950)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layers/mm_encoder_attention.py (+2/-0); vllm/attention/ops/vit_attn_wrappers.py (+21/-8); vllm/model_executor/models/dots_ocr.py (+1/-0); vllm/model_executor/models/ernie45_vl.py (+1/-0); vllm/model_executor/models/glm4_1v.py (+1/-0); vllm/model_executor/models/glmasr.py (+1/-0); vllm/model_executor/models/isaac.py (+1/-0); vllm/model_executor/models/keye.py (+1/-0); vllm/model_executor/models/moonvit.py (+1/-0); vllm/model_executor/models/paddleocr_vl.py (+1/-0); (+3 more)
LABELS: ready, qwen
BODY: ## Purpose ⏎ - `scale` in `MMEncoderAttention` doesn't take effect exactly, because we don't pass it to vit wrapper ops before. ⏎ - This PR fixes it, and also standardize Qwen-VL-style MMEncoderAttention usage to pass `scale` even if they have `scale=head_dim**-0.5` ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-087a138963  (L3, 2026-01-08, sha 087a13896319, PR #31928)
TITLE: [ROCm][CI] Fix attention backend test flakiness from uninitialized KV cache memory (#31928)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_attention_backends.py (+1/-1)
LABELS: rocm, ready, v1
BODY: Fixes intermittent test failures in `test_attention_backends.py` caused by uninitialized memory in the KV cache. ⏎  ⏎ ## Problem ⏎  ⏎ The `create_and_prepopulate_kv_cache` function uses `torch.empty()` to allocate the KV cache, which leaves memory uninitialized. Only the blocks actually used by test sequences are populated with valid data. ⏎  ⏎ This manifests as intermittent failures, particularly on ROCm where uninitialized GPU memory is less likely to cont …[truncated]

### L3-39d82005f7  (L3, 2026-01-08, sha 39d82005f7a5, PR #31286)
TITLE: fix(rocm): add early return in get_flash_attn_version for ROCm (#31286)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fa_utils
FILES: vllm/attention/utils/fa_utils.py (+3/-0)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ Prevents spurious "libcudart.so.12 not found" errors by skipping the CUDA-specific vllm_flash_attn import on ROCm platform. ⏎  ⏎ ## Test Result ⏎  ⏎ No error tracebacks.

### L3-5f2a473ff3  (L3, 2026-01-08, sha 5f2a473ff324, PR #31833)
TITLE: [ROCm][CI] v1 cpu offloading attention backend fix (#31833)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_offload/test_cpu_offloading.py (+4/-2)
LABELS: rocm, ready, v1
BODY: This PR fixes a regression caused by https://github.com/vllm-project/vllm/pull/30687 on ROCm. The underlying cause is that `FLASH_ATTN` is not supported on ROCm.

### L3-6cdf015c3c  (L3, 2026-01-08, sha 6cdf015c3cd8, PR #31747)
TITLE: [Misc] Fix `Current vLLM config is not set.` warnings, assert to avoid issues in the future (#31747)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.fa_utils
FILES: vllm/attention/utils/fa_utils.py (+6/-3); tests/compile/distributed/test_async_tp.py (+29/-24); tests/compile/test_config.py (+1/-1); tests/conftest.py (+11/-0); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+1/-1); tests/kernels/attention/test_mha_attn.py (+3/-1); tests/kernels/core/test_activation.py (+2/-0); tests/kernels/core/test_fused_qk_norm_rope.py (+1/-0); tests/kernels/core/test_fused_quant_layernorm.py (+1/-0); tests/kernels/core/test_layernorm.py (+1/-0); (+38 more)
LABELS: ready, v1, multi-modality, cpu, kv-connector, nvidia, ready-run-all-tests
ISSUES: #30859 [Bug]: set_current_vllm_config() is only done during the initialization stage but not the runtime stage
BODY: https://github.com/vllm-project/vllm/pull/30531 and https://github.com/vllm-project/vllm/pull/29575 introduced accesses to `get_current_vllm_config` on boot that are outside of `set_current_vllm_config` contexts leading to repeated logs  ⏎ ``` ⏎ (EngineCore_DP0 pid=1647618) WARNING 01-05 16:41:49 [vllm.py:1447] Current vLLM config is not set. ⏎ (EngineCore_DP0 pid=1647618) INFO 01-05 16:41:49 [scheduler.py:231] Chunked prefill is enabled with max_num_b …[truncated]

### L3-107cf8e92f  (L3, 2026-01-08, sha 107cf8e92f88, PR #31712)
TITLE:  fix(rocm): Add get_supported_kernel_block_sizes() to ROCM_ATTN (#31712)
SOURCES: path_core, subject_keyword, corpus:kernel-correctness-cases(introducing)
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+8/-0)
LABELS: documentation, rocm, ready, v1
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:e27078ea80 (fix PR 32336)
BODY: ## Purpose ⏎  ⏎ ROCM_ATTN kernel only supports block sizes 16 and 32. Previously, the backend used the default which claims any block size is supported, causing select_common_block_size() to return large framework block sizes (e.g., 256 for Mamba alignment) that the kernel cannot handle. ⏎      ⏎ This fix enables the kernel block size mechanism to correctly select 32 as the kernel block size for hybrid models like Nemotron-H.

### L3-b8112c1d85  (L3, 2026-01-08, sha b8112c1d8544, PR #31960)
TITLE: [Bugfix] Fix vllm serve failure with Nemotron Nano V3 FP8 (#31960)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/utils/fp8_utils.py (+4/-3)
LABELS: ready
BODY: ## Purpose ⏎ Fix bug: ⏎ https://github.com/vllm-project/vllm/issues/31957 ⏎  ⏎ The fix in file `fp8_utils.py` is required to fix this error (raised from `flashinfer_cutlass_fused_moe`): ⏎ ``` ⏎ RuntimeError: Check failed: fc1_dequant.ndim() == 1 (0 vs. 1) : fc1_dequant must be a 1D tensor ⏎ ``` ⏎  ⏎ ## Test Plan ⏎ vLLM serve should not fail. ⏎  ⏎ Run vLLM serve based on this recipe: ⏎ https://docs.vllm.ai/projects/recipes/en/latest/NVIDIA/Nemotron-3-Nano-30B-A3B.html#laun …[truncated]

### L3-d5ec6c056f  (L3, 2026-01-09, sha d5ec6c056f2f, PR #29450)
TITLE: [UX] Add vLLM model inspection view (#29450)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/entrypoints/llm.py (+16/-0); vllm/envs.py (+7/-0); vllm/model_executor/layers/rotary_embedding/common.py (+1/-1); vllm/model_executor/model_loader/base_loader.py (+14/-0); vllm/model_inspection.py (+136/-0); vllm/v1/worker/worker_base.py (+6/-0)
LABELS: frontend, ready, v1
BODY: ## Purpose ⏎  ⏎ Initial start to the kernels view request in https://github.com/vllm-project/vllm/issues/28085 by making a general modules view where we can see attention backends and quant_methods used at first. ⏎  ⏎ You can see the model inspection if you add `VLLM_LOG_MODEL_INSPECTION=1` during the engine creation or by simply printing the `LLM` object. ⏎  ⏎ ## Test Result ⏎  ⏎ #### LLM class ⏎  ⏎ ```python ⏎ >>> from vllm import LLM ⏎ >>> model = LLM("Qwen/Qwen3-0.6 …[truncated]

### L3-f9e2a75a1e  (L3, 2026-01-09, sha f9e2a75a1ee1, PR #32001)
TITLE: [fix] add cutedsl to global sf (#32001)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+1/-0)
LABELS: ready, nvidia
ISSUES: #31918 [Bug]: nvidia/DeepSeek-R1-NVFP4-v2 accuracy issue with NVFP4 dispatch (CUTEDSL MoE + DeepEP LL)
BODY: ## Purpose ⏎ Add flashinfer cutedsl to global sf list, fixes https://github.com/vllm-project/vllm/issues/31918 ⏎ ## Test Plan ⏎ ``` ⏎ VLLM_DEEPEPLL_NVFP4_DISPATCH=1 VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_USE_STANDALONE_COMPILE=0 VLLM_FLASHINFER_MOE_BACKEND="masked_gemm" VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ALL2ALL_BACKEND="deepep_low_latency" lm_eval --model vllm --model_args pretrained=nvidia/DeepSeek-R1-0528-FP4-v2,data_parallel_size=4,enable_expert_par …[truncated]

### L3-2612ba9285  (L3, 2026-01-09, sha 2612ba9285d8, PR #31916)
TITLE: [1/N][Attention] Restructure attention: move files (#31916)
SOURCES: path_core, symbol_pickaxe, corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.flash_attn.v1_backend, L3.flash_attn.fa_utils, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.prefix_prefill, L3.triton.decode_attention, L3.triton.chunked_prefill_paged_decode, L3.triton.unified_attention, L3.triton.v1_backend, L3.merge.triton_lse, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.mla.common_v1, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashmla_sparse, L3.mla.rocm_aiter, L3.mla.rocm_aiter_sparse, L3.dispatch.selector, L3.dispatch.registry, L3.dispatch.abstract_interface, L3.flex_attention, L3.tree_attention
FILES: .buildkite/test-amd.yaml (+1/-1); .buildkite/test-pipeline.yaml (+1/-1); .buildkite/test_areas/kernels.yaml (+1/-1); .github/CODEOWNERS (+4/-4); .github/mergify.yml (+2/-2); benchmarks/kernels/benchmark_reshape_and_cache_flash.py (+3/-3); docs/contributing/model/basic.md (+1/-1); docs/design/custom_op.md (+1/-1); docs/design/plugin_system.md (+2/-2); examples/offline_inference/basic/embed.py (+1/-1); (+185 more)
LABELS: documentation, performance, rocm, tpu, speculative-decoding, ready, ci/build, v1, multi-modality, llama
BODY: ## Purpose ⏎ Implement step 1 of #31919. This PR consists solely of file renaming and movement, and the necessary updates to imports. ⏎  ⏎ * Move vllm/attention/layers to vllm/model_executor/layers/attention ⏎ * Move vllm/attention/backends/abstract.py to vllm/v1/attention/backend.py  ⏎ * Move vllm/attention/backends/registry.py to vllm/v1/attention/backends/registry.py  ⏎ * Eliminate vllm/attention/backends folder ⏎ * Move vllm/attention/utils/fa_utils.py to  …[truncated]

### L3-4505849b30  (L3, 2026-01-09, sha 4505849b309b, PR #29304)
TITLE: [ROCm][PD] add moriio kv connector. (#29304)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen
FILES: docker/Dockerfile.rocm_base (+18/-1); docs/getting_started/installation/gpu.rocm.inc.md (+17/-2); examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py (+320/-0); tests/v1/kv_connector/unit/test_moriio_connector.py (+545/-0); vllm/distributed/kv_transfer/kv_connector/factory.py (+6/-0); vllm/distributed/kv_transfer/kv_connector/v1/moriio/__init__.py (+0/-0); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py (+321/-0); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py (+1515/-0); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py (+609/-0); vllm/envs.py (+18/-0)
LABELS: documentation, rocm, frontend, ready, ci/build, v1, kv-connector
BODY: ## Purpose ⏎  ⏎ This PR introduces the mori-io KV connector for AMD devices. Built on top of the [MORI](https://github.com/ROCm/mori) project, the mori-io connector supports both PULL and PUSH modes for KV Cache transfer. Key features include: ⏎  ⏎ - Mori backend integration.​ ⏎ - Mori-related components (buffer merge &session cache management &batch io).​ ⏎ - PULL mode (Serial interaction of prefill and decode).​ ⏎ - PUSH mode (Parallel interaction of prefill …[truncated]

### L3-08d954f036  (L3, 2026-01-09, sha 08d954f03659, PR #30886)
TITLE: [Doc] Add developer guide for CustomOp (#30886)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layers/mm_encoder_attention.py (+3/-0); vllm/model_executor/layers/mla.py (+3/-0); docs/design/custom_op.md (+318/-0); vllm/config/compilation.py (+2/-1); vllm/model_executor/custom_op.py (+6/-3); vllm/model_executor/layers/activation.py (+33/-0); vllm/model_executor/layers/conv.py (+6/-0); vllm/model_executor/layers/fused_moe/fused_moe.py (+3/-0); vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py (+3/-0); vllm/model_executor/layers/fused_moe/layer.py (+3/-0); (+14 more)
LABELS: documentation, rocm, ready
BODY: ## Purpose ⏎  ⏎ Currently, there are more and more `CustomOp` defined both in vLLM and other OOT plugin devices. Following https://github.com/vllm-project/vllm/pull/30125#issuecomment-3648916991, I have written a doc about the principle and usage of `CustomOp`. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted] ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > Introduces a developer guide for `CustomOp` explaining registration, enable/disable semantics, backend dispatch, and OOT  …[truncated]

### L3-80fead8bf6  (L3, 2026-01-09, sha 80fead8bf650, PR #25774)
TITLE: Fuse RoPE and MLA KV-cache write (#25774)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-0); csrc/cache_kernels_fused.cu (+279/-0); csrc/torch_bindings.cpp (+16/-0); vllm/_custom_ops.py (+26/-0); csrc/cache.h (+7/-0); tests/kernels/core/test_rotary_embedding_mla_cache_fused.py (+161/-0)
LABELS: ready, ci/build, v1
BODY: ## Purpose ⏎ It fuses RoPE and MLA KV-cache into a single kernel. ⏎ It only adds kernel without integrating it. ⏎ Numbers showed below were obtained with kernel integrated. ⏎ ## Test Plan ⏎ added test_rotary_embedding_mla_cache_fused.py ⏎ ## Test Result ⏎ tests pass ⏎ ## Benchmark ⏎ ``` ⏎ vllm bench serve --model deepseek-ai/DeepSeek-V3  --dataset-name sharegpt --sharegpt-output-len 100  --port 9020 --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --backen …[truncated]

### L3-1a19e9cd87  (L3, 2026-01-09, sha 1a19e9cd87b6, PR #31380)
TITLE: [Bugfix][ROCm]Fix Qwen3-Next-80B-A3B-Thinking inference and optimize non-standard block size (544) support under rocm_atten (#31380)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.triton.prefix_prefill, L3.triton.chunked_prefill_paged_decode, L3.rocm.v1_rocm_attn
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+83/-28); vllm/attention/ops/prefix_prefill.py (+80/-32); vllm/attention/ops/triton_reshape_and_cache_flash.py (+52/-12); vllm/v1/attention/backends/rocm_attn.py (+35/-10); tests/kernels/attention/test_prefix_prefill.py (+33/-2)
LABELS: bug, rocm, ready, v1, qwen
ISSUES: #26473 [Bug][ROCm]: Failed to send request to Qwen3-Next
BODY: ## Purpose ⏎  ⏎ Fixes #26473 ⏎  ⏎ This PR refactors the rocm_attn backend kernels to support models with non-power-of-2 block sizes, specifically the Qwen/Qwen3-Next-80B-A3B-Thinking model. ⏎  ⏎ The core of this update is a Dynamic Dispatching Mechanism:Standard Path ($2^n$): For models with power-of-2 block sizes (16, 32, 64, 128, etc.), the kernel retains the legacy bitwise-optimization logic to ensure maximum performance and zero regression.Generalized Pa …[truncated]

### L3-e02706d2d2  (L3, 2026-01-09, sha e02706d2d27c, PR #32000)
TITLE: [ROCm][CI][V1] Fix `nixl_connector` test failure and achieve CUDA parity in `test_async_scheduling` (#32000)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/e2e/test_async_scheduling.py (+3/-23); tests/v1/kv_connector/unit/test_nixl_connector.py (+20/-16); vllm/v1/spec_decode/eagle.py (+5/-0)
LABELS: rocm, speculative-decoding, ready, v1, kv-connector, nvidia
BODY: This PR adds FlexAttention backend support for ROCm in the EAGLE speculative decoding proposer, removing platform-specific attention backend restrictions and merging with the CUDA data flow of this test. It also fixes `test_abort_timeout_on_prefiller[ray]` failing on ROCm platforms. ⏎  ⏎ ## Changes ⏎  ⏎ - Added `FlexAttentionMetadata` to the allowed attention types for ROCm in `eagle.py` ⏎ - Removed ROCm-specific backend overrides that were workarounds for …[truncated]

### L3-da6709c9fe  (L3, 2026-01-09, sha da6709c9fe69, PR #32074)
TITLE: [Misc] Delay deprecation of CommonAttentionMetadata properties (#32074)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+3/-3)
LABELS: v1
BODY: Tracking progress here: https://github.com/vllm-project/vllm/issues/32072 ⏎  ⏎ Unfortunately the last 2 PR ⏎ - https://github.com/vllm-project/vllm/pull/31852 ⏎ - https://github.com/vllm-project/vllm/pull/32073 ⏎  ⏎ Are a bit trickier and do not appear to be on track to v0.14; delay just in case ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > Delays removal timeline for deprecated CPU-based fields and accessors in attention metadata. ⏎ >  ⏎ > - Update deprecation notes/comments in `CommonAtt …[truncated]

### L3-ea6d067a2a  (L3, 2026-01-09, sha ea6d067a2aeb, PR #30709)
TITLE: [Misc][LLaMa4] Compile LLaMa Vision Encoder (#30709)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/mm_encoder_attention.py (+3/-3); vllm/v1/attention/ops/vit_attn_wrappers.py (+3/-3); tests/compile/fullgraph/test_multimodal_compile.py (+37/-0); vllm/config/compilation.py (+3/-2); vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py (+5/-2); vllm/model_executor/models/llama.py (+5/-1); vllm/model_executor/models/mllama4.py (+29/-9)
LABELS: ready, v1, llama
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ We want to speedup up inference for mllama4 by applying `torch.compile` to the intensive workload, similar to what is done in #23207.  We start by enabling the VisionEncoder + PixelShuffle ⏎  ⏎ ## Test Plan ⏎ ### Unit Test ⏎ ``` ⏎ with-proxy pytest tests/compile/fullgraph/test_multimodal_compile.py::test_mllama4_vit_compilation ⏎ ``` ⏎ Result: ⏎ ``` ⏎  1 passed, 27 warnings in 176.88s (0:02:56)  ⏎ ``` ⏎  ⏎ ### Offline Test ⏎ ``` ⏎ with-proxy VLLM_USE_V1=1 python  …[truncated]

### L3-8e27663b6a  (L3, 2026-01-09, sha 8e27663b6a2b, PR #31968)
TITLE: [CPU] Add head sizes 80 and 112 with vec16 fallback (#31968)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+7/-3); csrc/cpu/cpu_attn.cpp (+3/-0); csrc/cpu/cpu_attn_amx.hpp (+1/-1); csrc/cpu/cpu_attn_neon.hpp (+1/-1)
LABELS: ready, v1, cpu
BODY: ## Purpose ⏎ Reintroduce support for head dimensions 80 and 112 in CPU attention backend which were previously removed in #27954 but these head dimensions are commonly used by granite models deployed on Z archs. Since these heads are not friendly for  Intel AMX instruction set. The implementation now falls back to vec16. ⏎ ## Test Plan ⏎ Build Docker image and test using `ibm-granite/granite-3b-code-base-2k` model which has head size of 80. ⏎ ## Test Res …[truncated]

### L3-0308901975  (L3, 2026-01-10, sha 03089019759b, PR #32052)
TITLE: [2/N][Attention] Fix pre-commit errors (#32052)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.flash_attn.fa_utils
FILES: vllm/v1/attention/backends/fa_utils.py (+4/-8); vllm/v1/attention/ops/paged_attn.py (+2/-6); tools/pre_commit/mypy.py (+0/-2)
LABELS: ready, v1
BODY: ## Purpose ⏎ Another step in the attention restructuring, #31919  ⏎ This PR fixes pre-commit errors which arose when moving files into regions of higher mypy scrutiny ⏎  ⏎ ## Test Plan ⏎ Pre-commit should pass ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted] ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > <sup>[Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) is generating a summary for commit b9743968bf2e3f20de1e03e51544ccaef025b11f. Configure [here](https://cursor.com/dashboard?tab=bugb …[truncated]

### L3-e6c6f2c79d  (L3, 2026-01-10, sha e6c6f2c79d2c, PR #31926)
TITLE: [Quant] Support MXFP4 W4A16 for compressed-tensors dense models (#31926)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py (+23/-0); vllm/model_executor/layers/quantization/compressed_tensors/schemes/__init__.py (+2/-0); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_mxfp4.py (+106/-0)
LABELS: ready, nvidia
BODY: ## Purpose ⏎  ⏎ Adds a compressed-tensors backend for dense models that have MXFP4 compression. At the moment it will ignore any activation quantization to just run weight-only with Marlin since that kernel is well tested, but we will expand in later PRs to also support W4A4 with flashinfer or qutlass kernels. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ vllm serve nm-testing/Meta-Llama-3-8B-Instruct-MXFP4 ⏎ python tests/evals/gsm8k/gsm8k_eval.py --port 9000     …[truncated]

### L3-1c46dea001  (L3, 2026-01-10, sha 1c46dea0017a, PR #31617)
TITLE: Revert "[Kernels][FI] Skip trtllm attention when num_kv_heads=1 (#308… (#31617)
SOURCES: path_core, corpus:confirmed-reverts, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+1/-21); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+0/-35)
LABELS: ready, nvidia
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 30842 reason=crash_or_hang
BODY: This reverts commit a100152288c8ec50336aea842f0b3d8e36624024. ⏎  ⏎ This PR causes GPT-OSS-120B TP8 has functional issue(NotImplementedError: FlashInfer backend currently does not support attention sinks). ⏎  ⏎ ## Purpose ⏎ Detail please see bug https://github.com/vllm-project/vllm/issues/30919. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted] ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > Re-enables TRTLLM attention for MQA configs and removes the guard/test that rejected `num_kv_ …[truncated]

### L3-e15a5ff07b  (L3, 2026-01-10, sha e15a5ff07b89, PR #32008)
TITLE: [MISC] Add strict contiguity check for FlashInfer attention tensors (#32008)
SOURCES: path_core, path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/torch_utils.py (+30/-0); vllm/v1/attention/backends/flashinfer.py (+11/-10)
LABELS: ready, v1, nvidia
BODY: Early check of potential error as in #30842. See also #31617, https://github.com/flashinfer-ai/flashinfer/issues/2232 ⏎  ⏎ Updates FlashInfer attention path to use stricter contiguous check, preventing potential CUDA kernel memory access issues. ⏎ Introduces `is_strictly_contiguous()` utility to detect tensors with degenerate strides  that PyTorch's `is_contiguous()` reports as contiguous.  ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > Strengthens memory layout validation for Fla …[truncated]

### L3-b8bf5c45bb  (L3, 2026-01-10, sha b8bf5c45bbbb, PR #31984)
TITLE: [Kernel] Optimize Sliding Window Attention in 3D Triton Kernel (#31984)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/v1/attention/ops/triton_unified_attention.py (+26/-3)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ This pull request improves the efficiency of sliding window attention in the 3D kernel by applying the same optimization used in the 2D kernel. It ensures that only tiles within the defined window are processed, resulting in better performance and reduced computational overhead. ⏎  ⏎ ## Performance ⏎ The following results were obtained for `openai/gpt-oss-20b` on an NVIDIA H100 GPU, by running ⏎  ⏎ <pre>$ VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm …[truncated]

### L3-20228cb851  (L3, 2026-01-12, sha 20228cb8514e, PR #32054)
TITLE: [3/N][Attention] Move AttentionMetadata-related code from utils.py to backend.py (#32054)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashmla_sparse, L3.mla.rocm_aiter, L3.mla.rocm_aiter_sparse, L3.dispatch.abstract_interface, L3.flex_attention, L3.tree_attention
FILES: vllm/model_executor/layers/attention/chunked_local_attention.py (+4/-2); vllm/model_executor/layers/attention/cross_attention.py (+1/-1); vllm/model_executor/layers/attention/encoder_only_attention.py (+1/-1); vllm/model_executor/layers/attention/static_sink_attention.py (+1/-1); vllm/v1/attention/backend.py (+287/-0); vllm/v1/attention/backends/cpu_attn.py (+2/-2); vllm/v1/attention/backends/flash_attn.py (+3/-1); vllm/v1/attention/backends/flashinfer.py (+3/-3); vllm/v1/attention/backends/flex_attention.py (+2/-4); vllm/v1/attention/backends/mla/common.py (+2/-2); (+27 more)
LABELS: documentation, rocm, speculative-decoding, ready, v1, cpu, nvidia, ready-run-all-tests
BODY: ## Purpose ⏎ Step 3 of #31919. Moves chunks of code from utils.py to backend.py (unchanged) and updates imports accordingly. The following objects are moved: ⏎ * `CommonAttentionMetadata` ⏎ * `AttentionMetadataBuilder` ⏎ * `AttentionCGSupport` ⏎  ⏎ ## Test Plan ⏎ CI (run all tests) ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted] ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > Consolidates attention metadata types into `vllm.v1.attention.backend` for clearer ownership and reuse. ⏎ >  ⏎ > - Moves `Common …[truncated]

### L3-9dbe1fe960  (L3, 2026-01-12, sha 9dbe1fe960f9, PR #32149)
TITLE: [Bugfix] Fix missing scale passing for encoder Triton Attention implementation  (#32149)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+1/-0); vllm/v1/attention/ops/triton_prefill_attention.py (+12/-11); examples/offline_inference/basic/embed.py (+0/-8); examples/offline_inference/basic/score.py (+0/-8)
LABELS: documentation, ready, v1
BODY: ## Purpose ⏎ - A small fix for Triton Encoder only attention, otherwise `TritonAttentionImpl.scale` won't take effect exactly. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted] ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > Ensures encoder Triton attention applies the configured softmax scale. ⏎ >  ⏎ > - Passes `softmax_scale=self.scale` when invoking `context_attention_fwd` in encoder path of `triton_attn.py` ⏎ > - Extends `context_attention_fwd` signature in `triton_prefill_att …[truncated]

### L3-3f72639d36  (L3, 2026-01-12, sha 3f72639d36a1, PR #31528)
TITLE: [FIX] Add NO_MUL activation support for modular kernel path (#31528)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_triton_moe_no_act_mul.py (+201/-0); vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py (+3/-1); vllm/model_executor/layers/fused_moe/cutlass_moe.py (+16/-4); vllm/model_executor/layers/fused_moe/deep_gemm_moe.py (+12/-6); vllm/model_executor/layers/fused_moe/fallback.py (+1/-0); vllm/model_executor/layers/fused_moe/flashinfer_cutedsl_moe.py (+1/-0); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+1/-0); vllm/model_executor/layers/fused_moe/fused_batched_moe.py (+9/-3); vllm/model_executor/layers/fused_moe/fused_marlin_moe.py (+2/-0); vllm/model_executor/layers/fused_moe/fused_moe.py (+17/-34); (+7 more)
LABELS: rocm, ready, gpt-oss, nvidia
BODY: This PR adds support for `*_no_mul` activations (e.g., `relu2_no_mul`) in the modular  ⏎ kernel MoE path (`TritonExperts`). ⏎  ⏎ ### Problem ⏎ The modular kernel path assumed all activations use gate/up multiplication (like SiLU, GELU), ⏎ where output size is `N/2`. For `*_no_mul` activations, which apply activation directly  ⏎ without gating, output size should equal input size (`N`). This caused assertion failures  ⏎ and buffer size mismatches. ⏎  ⏎ Attional Cha …[truncated]

### L3-8fb2c135be  (L3, 2026-01-12, sha 8fb2c135be35, PR #32118)
TITLE: [Bugfix] Fix stale SSM state for new Mamba requests scheduled as decode (#32118)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+3/-3); tests/v1/attention/test_batch_reordering.py (+21/-0)
LABELS: ready, v1
DEEP_STUDY: deep-study correctness case vllm:8fb2c135be: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Purpose ⏎ Fix stale SSM state corruption when new Mamba requests are scheduled with only 1 token due to token budget exhaustion. ⏎  ⏎ ## Problem ⏎ When the scheduler's token budget is nearly exhausted, new requests may be allocated only 1 token. A request with 1 token could be classified as decode rather than prefill. This causes a prompt to first be decoded on a stale SSM state (meaning it had previous values - this happens after the I believe most o …[truncated]

### L3-5b68107411  (L3, 2026-01-12, sha 5b681074119b, PR #31988)
TITLE: [Misc][PD] Fix `get_attn_backend` usage in transfer connectors (#31988)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector.py (+1/-1); vllm/distributed/kv_transfer/kv_connector/utils.py (+26/-2); vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py (+7/-9); vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py (+5/-8)
LABELS: ready, v1, kv-connector
BODY: This PR fixes the use of `get_attn_backend` in the context discussed here https://vllm-dev.slack.com/archives/C07R5Q1Q2BB/p1767810349936949. ⏎ In particular, it turns out that modification to the interface of this shared function can cause unintended backend retrieval (as a partial configuration was passed in), leading to cases such as ⏎ ``` ⏎ VLLM_LOGGING_LEVEL=DEBUG vllm serve google/gemma-3-4b-it --port 8004 --enforce-eager --kv-transfer-config '{"k …[truncated]

### L3-ad8818bb5e  (L3, 2026-01-12, sha ad8818bb5eb8, PR #31748)
TITLE: [Misc][BE] Type coverage for vllm/compilation [3/3] (#31748)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/compilation/activation_quant_fusion.py (+33/-28); vllm/compilation/collective_fusion.py (+111/-102); vllm/compilation/fusion.py (+58/-38); vllm/compilation/fusion_attn.py (+24/-20); vllm/compilation/inductor_pass.py (+1/-1); vllm/compilation/matcher_utils.py (+23/-22); vllm/compilation/qk_norm_rope_fusion.py (+16/-10); vllm/compilation/rocm_aiter_fusion.py (+38/-36); vllm/compilation/sequence_parallelism.py (+22/-18); vllm/distributed/parallel_state.py (+4/-4); (+1 more)
LABELS: ready, nvidia
BODY: ## Purpose ⏎ We want to provide better type hint coverage in vllm/compilation to improve maintainability, readability, and reduce silent errors ⏎  ⏎ This PR should be applied on top of #31744 ⏎  ⏎ ## Test Plan ⏎ `mypy vllm/compilation` ⏎  ⏎ ## Test Result ⏎ ``` ⏎ Success: no issues found in 28 source files ⏎ ``` ⏎ --- ⏎ [details omitted] ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > Improves type coverage and API clarity across compilation passes with minimal logic adjustments. ⏎ >  ⏎ > - Add explicit r …[truncated]

### L3-5e714f7ff4  (L3, 2026-01-12, sha 5e714f7ff416, PR #32233)
TITLE: [ROCm][CI] Fix HuggingFace flash_attention_2 accuracy issue in Isaac vision encoder (#32233)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/conftest.py (+19/-0); tests/models/multimodal/generation/vlm_utils/model_utils.py (+8/-0)
LABELS: rocm, ready, multi-modality
BODY: Fixes Isaac model test failures on ROCm by forcing SDPA (Scaled Dot-Product Attention) for the vision encoder. HuggingFace's `flash_attention_2` implementation produces incorrect results on ROCm, causing vision embeddings to diverge significantly from CUDA. ⏎  ⏎ ## Problem ⏎  ⏎ Isaac model tests were failing on ROCm with completely different outputs compared to CUDA: ⏎  ⏎ | Platform | Output | ⏎ |----------|--------| ⏎ | CUDA | "This image captures a vibrant str …[truncated]

### L3-15b33ff064  (L3, 2026-01-13, sha 15b33ff06473, PR #32226)
TITLE: [Misc] improve warning/assert messages (#32226)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.fa_utils
FILES: vllm/v1/attention/backends/fa_utils.py (+1/-1); vllm/compilation/compiler_interface.py (+2/-2); vllm/config/compilation.py (+4/-4); vllm/config/model.py (+1/-1); vllm/config/vllm.py (+10/-10); vllm/lora/ops/triton_ops/utils.py (+1/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Fix missing whitespaces and typos and polish sentences in various warning/assert messages. ⏎  ⏎ ## Test Plan ⏎  ⏎ CI green ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-98f60e5acb  (L3, 2026-01-13, sha 98f60e5acbcb, PR #32215)
TITLE: [6/N][Attention] Move utils to more appropriate locations (#32215)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/chunked_local_attention.py (+1/-1); vllm/model_executor/layers/attention/cross_attention.py (+0/-2); vllm/model_executor/layers/attention/encoder_only_attention.py (+0/-2); vllm/model_executor/layers/attention/static_sink_attention.py (+0/-2); vllm/v1/attention/backend.py (+25/-1); vllm/v1/attention/backends/utils.py (+1/-159); tests/v1/attention/test_attention_splitting.py (+4/-2); vllm/model_executor/models/whisper_utils.py (+1/-3); vllm/v1/worker/gpu/cudagraph_utils.py (+1/-1); vllm/v1/worker/gpu/spec_decode/eagle.py (+1/-1); (+4 more)
LABELS: ready, v1, nvidia, ready-run-all-tests
BODY: ## Purpose ⏎ Step 6 of #31919: ⏎ * Move `_make_metadata_with_slice`, `slice_query_start_locs split_attn_metadata` to `vllm/v1/worker/ubatch_utils.py` ⏎ * Move `subclass_attention_backend`, `subclass_attention_backend_with_overrides` to `vllm/v1/attention/backend.py` ⏎  ⏎ ## Test Plan ⏎ CI ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted] ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > Reorganizes utility functions for clearer ownership and reuse. ⏎ >  ⏎ > - Move `subclass_attention_backend` and `subcla …[truncated]

### L3-2263d44b68  (L3, 2026-01-13, sha 2263d44b6889, PR #32060)
TITLE: [4/N][Attention] Move MLA common to model_executor (#32060)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashmla_sparse, L3.mla.rocm_aiter, L3.mla.rocm_aiter_sparse
FILES: vllm/model_executor/layers/attention/mla_attention.py (+0/-0); vllm/v1/attention/backends/mla/cutlass_mla.py (+6/-6); vllm/v1/attention/backends/mla/flashattn_mla.py (+8/-8); vllm/v1/attention/backends/mla/flashinfer_mla.py (+7/-7); vllm/v1/attention/backends/mla/flashmla.py (+8/-8); vllm/v1/attention/backends/mla/flashmla_sparse.py (+4/-1); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+2/-2); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+4/-1); vllm/v1/attention/backends/mla/triton_mla.py (+5/-5); tests/v1/attention/test_mla_backends.py (+1/-1); (+4 more)
LABELS: rocm, speculative-decoding, ready, v1, kv-connector, nvidia, ready-run-all-tests
BODY: ## Purpose ⏎ Step 4 of #31919. Moves `vllm/v1/attention/backends/mla/common.py` to `vllm/model_executor/layers/attention/mla_attention.py` and updates imports accordingly. ⏎  ⏎ ## Test Plan ⏎ CI ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted] ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > <sup>[Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) is generating a summary for commit 86bf729e00a9e8ab17cd792981816017d8dbcddf. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup> ⏎  ⏎ - …[truncated]

### L3-a5bbbd2f24  (L3, 2026-01-13, sha a5bbbd2f2498, PR #29867)
TITLE: [Quantization] fix: overflow with static per-tensor scaling (#29867)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+10/-54); vllm/model_executor/layers/quantization/utils/quant_utils.py (+61/-2)
LABELS: bug, ready, v1, deepseek
BODY: ## Purpose ⏎  ⏎ When dequantizing weights with the `eye` method, the static scaling factor may actually push the 1s out of float8 range. ⏎  ⏎ Don't use that method when there's static scaling factors. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted] ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > Addresses weight dequantization overflow with static per‑tensor FP8 scaling and centralizes logic. ⏎ >  ⏎ > - Add `get_attribute_fallback` and `get_and_maybe_dequant_weights` in `quant_utils …[truncated]

### L3-8ef50d9a6b  (L3, 2026-01-13, sha 8ef50d9a6b91, PR #30885)
TITLE: [Kernel][Performance] Enable smaller Scaling Factor tiling for NVFP4 small-batch decoding (#30885)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+54/-1); .buildkite/test-pipeline.yaml (+3/-1); tests/kernels/quantization/nvfp4_utils.py (+24/-2); tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py (+26/-8); tests/models/quantization/test_nvfp4.py (+26/-0); vllm/_custom_ops.py (+30/-16); vllm/envs.py (+6/-1); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py (+7/-2); vllm/model_executor/layers/quantization/modelopt.py (+1/-1)
LABELS: performance, ready, ci/build, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎ This PR adds an opt-in NVFP4 backend variant that uses smaller scaling-factor tiling (8x4 SF layout). The change targets small-concurrency decode workloads and delivers ~25–35% higher output token throughput compared to the current best NVFP4 backend at small batch sizes. ⏎  ⏎ > **Note:** this backend is not recommended for medium or large batch sizes. Benefits are limited to the small-batch decode regime - see below. ⏎  ⏎ The backend is autom …[truncated]

### L3-9d0d7f48d5  (L3, 2026-01-14, sha 9d0d7f48d55a, PR #32281)
TITLE: [ROCm][CI] Handle missing vision_config in Isaac model attention patch (#32281)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/vlm_utils/model_utils.py (+23/-1)
LABELS: rocm, ready, multi-modality
BODY: Follow-up to #32233. Adds error handling to `isaac_patch_hf_runner` for models that don't have the expected `vision_config` attribute in their attention layers. ⏎  ⏎ ## Problem ⏎  ⏎ Isaac-0.1 (`PerceptronAI/Isaac-0.1`) uses `Siglip2VariableLengthAttention` which doesn't have a `vision_config` attribute, causing test failures: ⏎ ``` ⏎ AttributeError: 'Siglip2VariableLengthAttention' object has no attribute 'vision_config' ⏎ ``` ⏎  ⏎ The newer Isaac-0.2-2B-Preview h …[truncated]

### L3-b8199f6049  (L3, 2026-01-14, sha b8199f604931, PR #32167)
TITLE: [Model] Re-implement Qwen3Omni Audio Encoder (#32167)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_omni_moe_thinker.py (+428/-29)
LABELS: ready, qwen
BODY: ## Purpose ⏎ Re-implement Qwen3-Omni Audio Encoder with vLLM primitives with some vectorization improvements. - roughly 10% speedup at high batch sizes according to profiling run with TP=1. ⏎  ⏎ main ⏎ ``` ⏎ (EngineCore_DP0 pid=2620374) INFO 01-12 23:47:30 [gpu_model_runner.py:4696] Encoder cache will be initialized with a budget of 8192 tokens, and profiled with 21 audio items of the maximum feature size. ⏎ (EngineCore_DP0 pid=2620374) Audio processing time …[truncated]

### L3-e27078ea80  (L3, 2026-01-14, sha e27078ea807c, PR #32336)
TITLE: [Bugfix][ROCm][performance] Resolve the performance regression issue of the Qwen3-Next-80B-A3B-Thinking under rocm_atten (#32336)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+10/-1)
LABELS: bug, rocm, ready, v1, qwen
DEEP_STUDY: deep-study correctness case vllm:e27078ea80: class=perf_regression_as_correctness; symptom=performance_or_availability; introducing=#31712 || deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎ This PR resolves a significant performance regression for models using non-standard block sizes (specifically block size 544, such as Qwen3-Next-80B-A3B-Thinking) under the ROCM_ATTN backend. ⏎  ⏎ Background: Recently, PR #31712 restricted the supported kernel block sizes for ROCM_ATTN to only 16 and 32. While this was intended to prevent unsupported configurations, it inadvertently throttled models that rely on larger, non-standard block  …[truncated]

### L3-d86fc23bdd  (L3, 2026-01-15, sha d86fc23bdd8a, PR #32366)
TITLE: [Misc] Remove redundant line (#32366)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+0/-1)
LABELS: ready
BODY: ## Purpose ⏎ The line https://github.com/vllm-project/vllm/blob/8471b27df97c3eb79f891802fc0e858f8f7ac6a0/vllm/attention/layer.py#L366 is never went through; we should remove it to avoid some confusion ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-6218034dd7  (L3, 2026-01-15, sha 6218034dd7f9, PR #32348)
TITLE: [Model Runner V2] Support FlashInfer backend & Fix CUDA Graph bug [1/2] (#32348)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/cudagraph_utils.py (+9/-5); vllm/v1/worker/gpu/model_runner.py (+8/-2)
LABELS: bug, v1, nvidia
BODY: Add FlashInfer to the supported backends and fix a bug in supporting the `FULL_DECODE_ONLY` mode ⏎  ⏎ NOTE: We still need to fix FlashInfer attention metadata builder has race conditions

### L3-8ebfacaa75  (L3, 2026-01-15, sha 8ebfacaa7524, PR #32339)
TITLE: [Attention][MLA] Make `FLASHINFER_MLA` the default MLA backend on Blackwell, and TRTLLM the default prefill (#32339)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:confirmed-reverts(reverted), body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.platform.cuda_selection
FILES: vllm/config/attention.py (+1/-1); vllm/model_executor/layers/attention/mla_attention.py (+11/-10); vllm/platforms/cuda.py (+4/-4)
LABELS: ready, nvidia
DEEP_STUDY: deep-study: this PR was reverted by PR 32484 (confirmed_revert, reason=correctness_or_accuracy)
BODY: ## Purpose ⏎ Make `FLASHINFER_MLA` the default MLA decode backend on Blackwell (currently `CUTLASS_MLA`), and TRTLLM the default MLA prefill backend. ⏎  ⏎ Also adds log line to indicate which prefill backend is being used. ⏎  ⏎ ## Test Plan ⏎ On a Blackwell system, ⏎ `vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat` ⏎  ⏎ ## Test Result ⏎ FLASHINFER_MLA and TRTLLM are selected: ⏎ ``` ⏎ (EngineCore_DP0 pid=2563240) INFO 01-14 11:45:05 [cuda.py:315] Using AttentionBackendEnum …[truncated]

### L3-c36ba69bda  (L3, 2026-01-15, sha c36ba69bda26, PR #32362)
TITLE: [BugFix] Fix `assert x_s.shape[-1] == x_q.shape[-1] // group_shape[1]` in Blackwell Quantized MoE Test (#32362)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/utils/quant_utils.py (+2/-2)
LABELS: bug, ready
BODY: https://github.com/vllm-project/vllm/pull/32361 broke: ⏎  ⏎ ``` ⏎ python -m pytest tests/compile/fullgraph/test_full_graph.py::test_fp8_kv_scale_compile[deepseek-ai/DeepSeek-V2-Lite-AttentionBackendEnum.FLASHINFER_MLA-0] ⏎ ``` ⏎  ⏎ Add more robust handling of scalar weights to fix it

### L3-31c29257c8  (L3, 2026-01-15, sha 31c29257c852, PR #31827)
TITLE: [MoE Refactor][17/N] Apply Refactor to Bf16 (#31827)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/evals/gsm8k/configs/moe-refactor-dp-ep/Qwen3-30B-A3B-BF16-triton.yaml (+5/-0); tests/evals/gsm8k/configs/moe-refactor-dp-ep/config-b200.txt (+1/-0); tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-BF16-fi-cutlass.yaml (+7/-0); tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-BF16-triton.yaml (+6/-0); tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-fi-cutlass.yaml (+7/-0); tests/evals/gsm8k/configs/moe-refactor/Mixtral-8x7B-BF16-triton.yaml (+5/-0); tests/evals/gsm8k/configs/moe-refactor/Qwen3-30B-A3B-BF16-fi-cutlass.yaml (+7/-0); tests/evals/gsm8k/configs/moe-refactor/Qwen3-30B-A3B-BF16-triton.yaml (+5/-0); tests/evals/gsm8k/configs/moe-refactor/config-b200.txt (+4/-0); tests/evals/gsm8k/configs/moe-refactor/config-h100.txt (+2/-0); (+2 more)
LABELS: ready, nvidia
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:73f635a75f (fix PR 32438)
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ gsm8k result for Triton kernel, flashinfer_cutlass kernel and aiter rocm kernel for `Qwen/Qwen3-30B-A3B` in TP(triton), TEP (flashinfer cutlass) and rocm.  ⏎  ⏎ ``` ⏎ lm_eval \ ⏎                 --model local-completions \ ⏎                 --tasks gsm8k \ ⏎                 --model_args "model=Qwen/Qwen3-30B-A3B,base_url=http://localhost:8000/v1/completions,num_concurrent=1000,tokenized_requests=False" ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Triton ⏎  ⏎ `` …[truncated]

### L3-8c11001ba2  (L3, 2026-01-15, sha 8c11001ba211, PR #32238)
TITLE: [ROCM] DSfp4 mla projection gemms weight dynamic quantization (#32238)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.common_v1
FILES: vllm/envs.py (+6/-0); vllm/model_executor/layers/attention/mla_attention.py (+47/-80); vllm/_aiter_ops.py (+30/-0); vllm/model_executor/layers/quantization/quark/utils.py (+14/-0)
LABELS: rocm, ready, v1
BODY: ### Commands: ⏎  ⏎ Server: ⏎ > VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4BMM=1 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ROCM_USE_AITER_MLA=1 AMDGCN_USE_BUFFER_OPS=1 SAFETENSORS_FAST_GPU=1 vllm serve /data/models/DeepSeek-R1-0528-MXFP4-Preview --host localhost --port 8000 --tensor-parallel-size 8 --max-num-batched-tokens 32768 --trust-remote-code --no-enable-prefix-caching --disable-log-requests --gpu_memory_utilization 0.8 --async-scheduling --block-size 16 - …[truncated]

### L3-130d6c9514  (L3, 2026-01-15, sha 130d6c951485, PR #29887)
TITLE: [ROCm][Perf] Enable shuffle kv cache layout and assembly paged attention kernel for `AiterFlashAttentionBackend` (#29887)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.rocm.aiter_fa
FILES: vllm/envs.py (+5/-0); vllm/v1/attention/backends/rocm_aiter_fa.py (+299/-84); vllm/_aiter_ops.py (+7/-0)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ This PR add assembly paged attention kernel in aiter to `AiterFlashAttentionBackend`. We verify this implementation on `Qwen3-30B-A3B-FP8` Mi308 and observed at least about 20% thoughput gain and obvious latency reduction on tpot. ⏎  ⏎ This PR add this flag `USING_SHUFFLE_LAYOUT` to control whether to enable assembly paged attention, and set it False by default to prevent any unexpected circumstance. This flag will be removed after assembl …[truncated]

### L3-047413375c  (L3, 2026-01-15, sha 047413375c9d, PR #30361)
TITLE: [Attention][AMD] Make flash-attn optional (#30361)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fa_utils
FILES: vllm/v1/attention/backends/fa_utils.py (+7/-5)
LABELS: rocm, speculative-decoding, ready, v1
BODY: If `flash-attn` is not installed, vLLM on ROCm (with any attention backend) would [fail with](https://github.com/vllm-project/vllm/blob/9db78f34dce03d149f3571d45a2d2f259bdc7d15/vllm/attention/utils/fa_utils.py#L25) `Rocm platform requires upstream flash-attn to be installed` due to the unconditional import of `vllm.attention.utils.fa_utils` in `vllm.v1.spec_decode.eagle` which in turn is unconditionally imported by `vllm.v1.worker.gpu_model_runne …[truncated]

### L3-46f8a982b1  (L3, 2026-01-16, sha 46f8a982b191, PR #32431)
TITLE: [ROCm][CI] Enable AITER Unified Attention On ROCm For gpt-oss Test (#32431)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/entrypoints/openai/test_serving_chat.py (+15/-2)
LABELS: rocm, ready, gpt-oss
BODY: https://github.com/vllm-project/vllm/pull/26291 introduced a gpt-oss test which currently fails on ROCm: ⏎ ``` ⏎ pytest -v -s tests/entrypoints/openai/test_serving_chat.py::TestGPTOSSSpeculativeChat::test_gpt_oss_speculative_reasoning_leakage[with_tool_parser-exclude_tools_when_tool_choice_none] ⏎ ``` ⏎  ⏎ On ROCm, gpt-oss requires the `ROCM_AITER_UNIFIED_ATTN` backend to be enabled. It is expected that `TRITON_ATTN` should work, however, so I have filed a …[truncated]

### L3-41c544f78a  (L3, 2026-01-16, sha 41c544f78a4d, PR #32264)
TITLE: [ROCm] [CI] [Release] Rocm wheel pipeline with sccache (#32264)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm (+150/-4); docker/Dockerfile.rocm_base (+104/-2); requirements/rocm-test.txt (+2/-0); requirements/rocm.txt (+0/-1); .buildkite/release-pipeline.yaml (+362/-0); .buildkite/scripts/annotate-rocm-release.sh (+74/-0); .buildkite/scripts/cache-rocm-base-wheels.sh (+140/-0); .buildkite/scripts/generate-nightly-index.py (+69/-9); .buildkite/scripts/upload-rocm-wheels.sh (+151/-0); tools/vllm-rocm/pin_rocm_dependencies.py (+221/-0)
LABELS: rocm, ready, ci/build
BODY: ## Purpose ⏎ If we picked to merge this PR first, then we don't need to merge PR https://github.com/vllm-project/vllm/pull/30395 . ⏎  ⏎ It optimized the build process of `docker/Dockerfile.rocm_base` and `docker/Dockerfile.rocm` of PR https://github.com/vllm-project/vllm/pull/30395 . ⏎ It is built on top of https://github.com/vllm-project/vllm/pull/30395 by integration `sccache` to optimize the release pipeline. ⏎  ⏎  `Dockerfile.rocm_base` build time reduce …[truncated]

### L3-7a1030431a  (L3, 2026-01-16, sha 7a1030431ade, PR #29843)
TITLE: Atomics Reduce Counting Optimization for SplitK Skinny GEMMs. (#29843)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/rocm/ops.h (+4/-0); csrc/rocm/skinny_gemms.cu (+545/-9); csrc/rocm/torch_bindings.cpp (+6/-0); tests/kernels/quantization/test_rocm_skinny_gemms.py (+53/-0); vllm/_custom_ops.py (+6/-0); vllm/model_executor/layers/utils.py (+21/-1)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎ Optimize N=16-128 M=128,640 range and K=2880 cases -- targeting decode GEMMs observed in low concurrency gpt-oss. After more testing can be expanded to all GEMMs where N=16-128 and **(M/16)*(K/512)** is less than SIMD count.  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-2e7c89e708  (L3, 2026-01-17, sha 2e7c89e7084d, PR #32484)
TITLE: Revert "[Attention][MLA] Make `FLASHINFER_MLA` the default MLA backen… (#32484)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:confirmed-reverts
ARTIFACT_HINTS: L3.mla.common_v1, L3.platform.cuda_selection
FILES: vllm/config/attention.py (+1/-1); vllm/model_executor/layers/attention/mla_attention.py (+10/-11); vllm/platforms/cuda.py (+4/-4)
LABELS: ready, nvidia
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 32339 reason=correctness_or_accuracy
BODY: …d on Blackwell, and TRTLLM the default prefill (#32339)" ⏎  ⏎ This reverts commit 8ebfacaa7524dd9641cef9c5e4151214e0912dfd. ⏎  ⏎ ## Purpose ⏎  ⏎ The FI prefill backend has correctness issues (https://github.com/vllm-project/vllm/issues/27684) which caused CI failure. Will make a separate PR to improve CI coverage as this should have been detected in the original PR. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-2b99f210f5  (L3, 2026-01-17, sha 2b99f210f53f, PR #32411)
TITLE: [Misc] Fix typo: seperator -> separator in flashmla_sparse.py (#32411)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_sparse
FILES: vllm/v1/attention/backends/mla/flashmla_sparse.py (+7/-7)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎   Fix a typo in `flashmla_sparse.py`: rename `FP8SeperatePrefillDecode` to `FP8SeparatePrefillDecode` (seperator -> separator). ⏎  ⏎   ## Test Plan ⏎  ⏎   No functional changes - this is a pure rename for code readability. ⏎  ⏎   ## Test Result

### L3-9e078d0582  (L3, 2026-01-17, sha 9e078d058253, PR #31492)
TITLE: [CI/Build][Docker] Add centralized version manifest for Docker builds (#31492)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+33/-6); docker/versions.json (+92/-0); .pre-commit-config.yaml (+7/-0); tools/generate_versions_json.py (+139/-0)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Introduces `docker/versions.json` as a centralized, machine-readable version manifest for all pinned dependencies in Docker builds. ⏎  ⏎ ## Problem ⏎  ⏎ Version information is currently scattered across `docker/Dockerfile` and various install scripts, making it hard to: ⏎ - Track version changes across releases ⏎ - Programmatically query versions for CI/tooling ⏎ - Maintain consistency ⏎  ⏎ ## Solution ⏎  ⏎ **Dockerfile = Source of Truth** with auto-genera …[truncated]

### L3-8cc26acd8b  (L3, 2026-01-17, sha 8cc26acd8b77, PR #32403)
TITLE: [Performance] Improve Triton prefill attention kernel's performance  (#32403)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/utils/math_utils.py (+4/-0); vllm/v1/attention/ops/triton_prefill_attention.py (+27/-46); tests/models/language/pooling/test_token_classification.py (+2/-2)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Refine prefill attention kernel with FA2 implementation based on https://github.com/ModelTC/LightLLM/blob/2fbd2b8a7b180570129bd539bb34cbe0a5dbb22a/lightllm/common/basemodel/triton_kernel/att/prefill_att/context_flashattention_nopad.py#L287-L288 ⏎ - In previous implementation, we use `float("-inf")` in attention mask, which needs additional invalid values mask for SWA. This PR replaces it with extremely large values $-10^8$ to overflow. ⏎ - …[truncated]

### L3-c826c72a96  (L3, 2026-01-18, sha c826c72a9633, PR #32511)
TITLE: [Model] Support Step1 Model (#32511)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.unified_attention, L3.triton.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/layer.py (+11/-1); vllm/v1/attention/backend.py (+4/-0); vllm/v1/attention/backends/triton_attn.py (+7/-1); vllm/v1/attention/ops/triton_unified_attention.py (+25/-2); docs/models/supported_models.md (+2/-1); tests/models/registry.py (+3/-0); tests/models/test_initialization.py (+4/-1); vllm/model_executor/models/registry.py (+1/-0); vllm/model_executor/models/step1.py (+415/-0)
LABELS: documentation, new-model, ready, v1
BODY: ## Purpose ⏎ Added native support for Step1 model in vLLM. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-16de822c71  (L3, 2026-01-18, sha 16de822c719d, PR #32433)
TITLE: [Refactor] Remove unused file `pallas_kv_cache_update.py` (#32433)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/pallas_kv_cache_update.py (+0/-130)
LABELS: tpu, ready, v1
BODY: ## Purpose ⏎  ⏎ Seems introduced in https://github.com/vllm-project/vllm/pull/19928 ⏎  ⏎ But this function now is not used anywhere, not sure if we can delete it ⏎  ⏎ CC: @yaochengji

### L3-6101a26dc9  (L3, 2026-01-18, sha 6101a26dc956, PR #32417)
TITLE: [BUGFIX]  Fix degenerate strides in TRTLLM query tensors for FlashInfer backend. Fixes issue #32353 (#32417)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+10/-4)
LABELS: bug, ready, v1, nvidia
BODY: ### Summary ⏎  ⏎ This PR fixes an issue #32353 with degenerate strides in query tensors when using the TRTLLM kernels in the FlashInfer attention backend. The `.contiguous()` call alone doesn't fix degenerate strides when a dimension has size 1, which can cause issues with kernel execution. ⏎  ⏎ ### Problem ⏎  ⏎ Query tensors can have degenerate strides and `.contiguous()` doesn't fix it. In #32353: ⏎ ``` ⏎ Shape: torch.Size([1, 32, 128]) ⏎ Stride: (4608, 128, 1) ⏎  …[truncated]

### L3-38bf2ffb21  (L3, 2026-01-18, sha 38bf2ffb21d5, PR #32540)
TITLE: [Bugfix] Fix GLM-ASR audio encoder RoPE dim (#32540)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: examples/offline_inference/audio_language.py (+28/-28); vllm/model_executor/models/glmasr.py (+12/-2)
LABELS: bug, documentation, ready
ISSUES: #32445 [Bug]: GLMASR rope error
BODY: ## Purpose ⏎ - Fix https://github.com/vllm-project/vllm/issues/32445 ⏎ - GLM-ASR audio encoder's RoPE should be half rotary. ⏎ - This issue only occured in native code path, because vllm_flash_attn triton RoPE will always allocate `q_rot`/`k_rot` based on `cos`/`sin`'s last dim. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-c0a350ca73  (L3, 2026-01-19, sha c0a350ca73fa, PR #32363)
TITLE: [ROCm][CI] Add ROCm attention backend support for EAGLE DP tests (#32363)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/distributed/test_eagle_dp.py (+28/-6)
LABELS: rocm, ready, v1
BODY: This PR adds ROCm platform support to the EAGLE data parallel tests by parametrizing attention backends based on the detected platform. ⏎  ⏎ ## Changes ⏎  ⏎ - Add platform detection using `current_platform.is_rocm()` to select appropriate attention backends ⏎ - Parametrize `test_run_eagle_dp` across platform-specific backends: ⏎   - **ROCm**: `ROCM_ATTN`, `ROCM_AITER_FA`, `TRITON_ATTN`, `FLEX_ATTENTION` ⏎   - **CUDA**: `FLASH_ATTN` ⏎ - Conditionally apply `VLLM_ …[truncated]

### L3-0727cc9ecf  (L3, 2026-01-19, sha 0727cc9ecf2e, PR #32529)
TITLE: [BUGFIX] Fix `test_mla_backends.py`. Scale MLA projection weights to prevent numerical instability  (#32529)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_mla_backends.py (+8/-0)
LABELS: bug, ready, v1
BODY: V1 Test attention (B200) CI run fails. Here #32484 was committed workaround.  ⏎  ⏎ This PR make a root cause fix.  ⏎  ⏎ Apply `1/sqrt(kv_lora_rank)` scaling to `kv_b_proj_weight` to produce outputs with unit-variance. ⏎  ⏎ Without scaling, random weight matrices with `kv_lora_rank=512` inputs produce projection outputs with std ≈ √512 ≈ 22.6. These large magnitudes cause extreme attention scores, which destabilize LSE (log-sum-exp) merging across chunks and  …[truncated]

### L3-7350331718  (L3, 2026-01-19, sha 73503317188e, PR #32349)
TITLE: [BugFix] Fix TRT-LLM NVFP4 DP/EP (#32349)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/evals/gsm8k/configs/moe-refactor-dp-ep/Qwen3-30B-A3B-NvFp4-ModelOpt-fi-trtllm.yaml (+8/-0); tests/evals/gsm8k/configs/moe-refactor-dp-ep/config-b200.txt (+1/-0); vllm/model_executor/layers/fused_moe/layer.py (+1/-8); vllm/model_executor/layers/quantization/modelopt.py (+11/-3)
LABELS: bug, ready, nvidia
DEEP_STUDY: deep-study correctness case vllm:7350331718: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#31692
BODY: ## Purpose ⏎ Flashinfer trtllm nvfp4 was broke by https://github.com/vllm-project/vllm/pull/31692 .  ⏎ ``` ⏎ (EngineCore_DP2 pid=545)   File "/usr/local/lib/python3.12/dist-packages/torch/_ops.py", line 841, in __call__ ⏎ (EngineCore_DP2 pid=545)     return self._op(*args, **kwargs) ⏎ (EngineCore_DP2 pid=545)            ^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ (EngineCore_DP2 pid=545)   File "/scratch/vllm/vllm/model_executor/layers/fused_moe/layer.py", line 2146, in moe …[truncated]

### L3-4a5299c93f  (L3, 2026-01-19, sha 4a5299c93ff9, PR #24322)
TITLE: feat: spec decode with draft models (#24322)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+11/-1); vllm/v1/attention/backends/utils.py (+32/-0); examples/offline_inference/spec_decode.py (+17/-2); examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py (+1/-1); tests/v1/e2e/test_spec_decode.py (+306/-9); tests/v1/worker/test_utils.py (+35/-0); vllm/benchmarks/datasets.py (+10/-9); vllm/benchmarks/lib/ready_checker.py (+6/-0); vllm/config/parallel.py (+4/-0); vllm/config/speculative.py (+33/-6); (+11 more)
LABELS: documentation, performance, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality
BODY: ## Purpose ⏎ Enabling draft models for speculative decoding (SD). ⏎ E.g. `Qwen3-1.7B` as draft model and `Qwen3-32B` as target model. ⏎ This type of SD requires no special trained heads (like EAGLE, or Medusa). ⏎  ⏎ Example usage: ⏎ ```shell ⏎ vllm serve \ ⏎     --model=Qwen/Qwen3-4B \ ⏎     --speculative-config '{"model": "Qwen/Qwen3-0.6B", "method": "draft_model", "num_speculative_tokens": 3, "max-model-len": 2000, "disable_padded_drafter_batch": true}' \ ⏎     -- …[truncated]

### L3-1a1fc3bbc0  (L3, 2026-01-19, sha 1a1fc3bbc062, PR #32615)
TITLE: [Attention][MLA] Make FLASHINFER_MLA the default MLA backend on Blackwell, and TRTLLM the default prefill (#32615)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.platform.cuda_selection
FILES: vllm/config/attention.py (+1/-1); vllm/model_executor/layers/attention/mla_attention.py (+11/-10); vllm/platforms/cuda.py (+4/-4); .buildkite/test-amd.yaml (+1/-1); .buildkite/test-pipeline.yaml (+1/-1); .buildkite/test_areas/attention.yaml (+1/-1)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎ #32339 was reverted because it caused a CI failure. That failure was fixed by #32529. Consequently, this PR re-applies the default change and updates the CI to use the default backend on each platform. ⏎  ⏎ ## Test Plan ⏎ V1 Attention Test (B200) should pass in CI with FlashInfer MLA and TRTLLM selected. ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-aa7f37ccfa  (L3, 2026-01-19, sha aa7f37ccfa16, PR #30802)
TITLE: Add support for LoRA adapters in Nemotron-H models (#30802)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/lora/test_layers.py (+297/-0); vllm/lora/layers/__init__.py (+2/-0); vllm/lora/layers/column_parallel_linear.py (+85/-4); vllm/lora/layers/fused_moe.py (+25/-17); vllm/lora/lora_weights.py (+26/-3); vllm/lora/model_manager.py (+47/-3); vllm/lora/utils.py (+6/-0); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+5/-0); vllm/model_executor/models/interfaces.py (+1/-0); vllm/model_executor/models/nemotron_h.py (+3/-0)
LABELS: ready, nvidia
BODY: ## Purpose ⏎ Add support for LoRA adapters in Nemotron-H models (such as https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8). ⏎  ⏎ No support for `VLLM_USE_FLASHINFER_MOE_FP8` at this time. ⏎  ⏎ ## Test Plan ⏎ Create LoRA adapters using `peft` with `target_modules="all-linear"`. ⏎ Tested vLLM generation with a Nemotron-H model: ⏎ * With/without LoRA adapters enabled. ⏎ * With/without TP. ⏎ * With/without CUDA graphs. ⏎ * Raise an error if LoRA is used wi …[truncated]

### L3-9ab4388cd3  (L3, 2026-01-20, sha 9ab4388cd3dc, PR #32709)
TITLE: [Model Runner V2] Support FLASHINFER_MLA backend (#32709)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/attn_utils.py (+1/-1); vllm/v1/worker/gpu/model_runner.py (+1/-1)
LABELS: v1
BODY: 

### L3-148117ea2e  (L3, 2026-01-20, sha 148117ea2e68, PR #27814)
TITLE: [Refactor] Make FP8 Linear Ops use kernel abstraction (#27814)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/lm-eval-harness/configs/models-small-rocm.txt (+5/-0); tests/compile/distributed/test_fusion_all_reduce.py (+17/-27); tests/compile/distributed/test_sequence_parallelism.py (+17/-26); tests/compile/test_functionalization.py (+19/-22); tests/compile/test_fusion.py (+182/-146); tests/compile/test_fusion_attn.py (+20/-21); tests/compile/test_silu_mul_quant_fusion.py (+44/-27); tests/kernels/quantization/test_scaled_mm_kernel_selection.py (+24/-22); tests/quantization/test_compressed_tensors.py (+1/-1); tests/utils.py (+127/-0); (+20 more)
LABELS: rocm, ready, ci/build, cpu, nvidia, ready-run-all-tests
BODY: ## Purpose ⏎ This PR refactors the FP8 linear kernel integration in vLLM to improve code clarity, maintainability, and path consistency.  ⏎  ⏎ Changes: ⏎ 1- A more generic [ScaledMMLinearKernel](11785) interface. ⏎ 2- Introduce `FP8ScaledMMLinearKernel` and `Int8ScaledMMLinearKernel`. ⏎ 3- Remove `FP8LinearOp` and replace with different `FP8ScaledMMLinearKernel` implementations. ⏎ 4- Update unit tests to use [ScaledMMLinearKernel](11785) interface. ⏎  ⏎ ### Follow …[truncated]

### L3-4ca62a0dbd  (L3, 2026-01-20, sha 4ca62a0dbdba, PR #32331)
TITLE: [PluggableLayer][1/N] Define PluggableLayer (#32331)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mla.py (+8/-11); docs/design/custom_op.md (+0/-9); tests/model_executor/test_enabled_custom_ops.py (+5/-5); vllm/model_executor/custom_op.py (+86/-14)
LABELS: documentation, ready
ISSUES: #23786 [RFC]: Design a new Layer-Pluggable abstraction to work together with CustomOp
DEEP_STUDY: deep-study: this PR was reverted by PR 32725 (confirmed_revert, reason=ci_or_test_failure)
BODY: **EDIT**: Reverted in #32725, reapplied in #32744 ⏎  ⏎ ## Purpose ⏎ First implementation of RFC https://github.com/vllm-project/vllm/issues/23786: Define PluggableLayer and apply to MLA as an example. ⏎  ⏎ ## Test Plan ⏎ New ut waitted to be added. ⏎  ⏎ ## Test Result ⏎ All ci should pass. ⏎  ⏎ ## CC List ⏎ @ProExpertProg @wangxiyuan @jgong5 @Yikun  ⏎  ⏎ --- ⏎ [details omitted]

### L3-04a9e064db  (L3, 2026-01-20, sha 04a9e064db4d, PR #32687)
TITLE: [Bugfix] fix the ima issue of qwen-vit (#32687)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen2_5_vl.py (+1/-0); vllm/model_executor/models/qwen3_vl.py (+2/-2)
LABELS: bug, ready, qwen
BODY: After changing the default value of Qwen3VL `_MAX_FRAMES_PER_VIDEO`, I encounted the ima issues at the forward of Qwen-ViT (profiling stage). I found it's a corner case when `bs=1`. ⏎  ⏎ https://github.com/vllm-project/vllm/blob/13f6630a9ea78bee4bd80bb6e842e55e374eec9a/vllm/model_executor/models/qwen2_5_vl.py#L384-L386 ⏎  ⏎ When `b=1`, `qk_reshaped` is not contiguous and cause the ima issue. So this pr just add one line. 🫡 ⏎  ⏎ [details omitted]

### L3-c78ee240b3  (L3, 2026-01-21, sha c78ee240b30d, PR #32725)
TITLE: Revert "[PluggableLayer][1/N] Define PluggableLayer" (#32725)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mla.py (+11/-8); docs/design/custom_op.md (+9/-0); tests/model_executor/test_enabled_custom_ops.py (+5/-5); vllm/model_executor/custom_op.py (+14/-86)
LABELS: documentation, ready
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 32331 reason=ci_or_test_failure
BODY: Reverts vllm-project/vllm#32331

### L3-7013e9ac8f  (L3, 2026-01-21, sha 7013e9ac8f65, PR #29087)
TITLE: OffloadingConnector: Prevent redundant loads (#29087)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_offloading_connector.py (+67/-1); vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py (+36/-3); vllm/v1/kv_offload/abstract.py (+4/-2); vllm/v1/kv_offload/arc_manager.py (+1/-1); vllm/v1/kv_offload/lru_manager.py (+1/-1)
LABELS: ready, v1, kv-connector
BODY: When handling concurrent requests hitting the same CPU blocks, multiple concurrent CPU->GPU transfers will be issued, one per each request. If the GPU prefix cache is enabled, this will create an unnecessary duplication of KV data in the GPU prefix cache. ⏎ This PR changes the OffloadingConnector to detect such cases, and delay loading requests which have some of their blocks already being loaded by other requests. ⏎ This results in reducing the unne …[truncated]

### L3-42135d6898  (L3, 2026-01-21, sha 42135d689830, PR #32414)
TITLE: [MoE Refactor] Oracle Select FP8+NVFP4 Kernels In Priority (#32414)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: .buildkite/test-pipeline.yaml (+40/-0); benchmarks/kernels/benchmark_cutlass_moe_fp8.py (+7/-5); benchmarks/kernels/benchmark_cutlass_moe_nvfp4.py (+3/-4); benchmarks/kernels/benchmark_grouped_gemm_cutlass.py (+13/-12); benchmarks/kernels/benchmark_moe.py (+29/-1); docs/design/moe_kernel_features.md (+2/-2); tests/compile/test_fusion_attn.py (+3/-3); tests/compile/test_silu_mul_quant_fusion.py (+3/-3); tests/evals/gsm8k/configs/moe-refactor-dp-ep/Llama-4-Scout-Fp8-ModelOpt-triton.yaml (+3/-0); tests/evals/gsm8k/configs/moe-refactor-dp-ep/Qwen3-30B-A3B-Fp8-AutoFp8-deepgemm-deepep-ll.yaml (+0/-1); (+72 more)
LABELS: documentation, performance, rocm, ready, ci/build, v1, llama, qwen, gpt-oss, nvidia
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:6a9cceb219 (fix PR 37418) || deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ * update oracles to have notion of priority of kernels ⏎ * update kernels to have standard interface for construction ⏎ * update kernels to have registration of supported features ⏎  ⏎ This allows us to: ⏎ * autoselect kernels across hardware and model architecture (enabling us to opt-into kernels like TRTLLM) ⏎ * verify that kernels support the deployment at init time rather than runtime (enabling clearer error messages) ⏎ * improve code reuse ⏎  ⏎ ##  …[truncated]

### L3-808d6fd7b9  (L3, 2026-01-21, sha 808d6fd7b97f, PR #30993)
TITLE: Bump Flashinfer to v0.6.1 (#30993)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: docker/Dockerfile (+1/-1); docker/Dockerfile.nightly_torch (+2/-3); docker/versions.json (+1/-1); requirements/cuda.txt (+1/-1); vllm/v1/attention/backends/flashinfer.py (+13/-5); tests/kernels/moe/test_ocp_mx_moe.py (+0/-26); tests/v1/sample/test_topk_topp_sampler.py (+1/-0); vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py (+0/-7); vllm/model_executor/layers/fused_moe/trtllm_moe.py (+0/-1); vllm/model_executor/layers/quantization/mxfp4.py (+1/-2); (+2 more)
LABELS: ready, ci/build, v1, nvidia
BODY: ## Purpose ⏎ Bump Flashinfer to v0.6.1 when it is released. ⏎ API change: argument `tile_tokens_dim` has been removed from all TRTLLM MoE kernels. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted] ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > <sup>[Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) is generating a summary for commit a5e35ec5cd52bd4ca0c9e5a6fbb3a3b491e21ffa. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup> ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > **Upgrade Flas …[truncated]

### L3-6bb2bc71e2  (L3, 2026-01-21, sha 6bb2bc71e299, PR #32749)
TITLE: [Bugfix] Force using spawn multiprocess method when it's the WSL platform (#32749)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/utils/system_utils.py (+4/-0)
LABELS: bug, ready
ISSUES: #32559 [Bug]: NVML initialization failure even when running the basic example in the WSL platform
BODY: ## Purpose ⏎  ⏎ This is to fix https://github.com/vllm-project/vllm/issues/32559 where the NVML initialization failed in the WSL platform even when running the basic example. ⏎  ⏎ ```bash ⏎ (vllm) $ python examples/offline_inference/basic/basic.py ⏎ ... ⏎ (EngineCore_DP0 pid=29657) ERROR 01-17 21:03:52 [core.py:935] vllm.third_party.pynvml.NVMLError_NotSupported: Not Supported ⏎ ... ⏎ ``` ⏎  ⏎ All the root cause analysis can be found in [that issue description](https: …[truncated]

### L3-1861ae8aae  (L3, 2026-01-21, sha 1861ae8aae18, PR #32744)
TITLE: [PluggableLayer][1/N] Define PluggableLayer (Fix ci) (#32744)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mla.py (+8/-11); benchmarks/kernels/benchmark_activation.py (+5/-5); docs/design/custom_op.md (+0/-9); tests/kernels/utils.py (+2/-2); tests/model_executor/test_enabled_custom_ops.py (+5/-5); vllm/config/compilation.py (+2/-2); vllm/model_executor/custom_op.py (+86/-14)
LABELS: documentation, performance, ready, ready-run-all-tests
BODY: Reapplies #32331 that was reverted in #32725. ⏎  ⏎ ## Purpose ⏎ First implementation of RFC https://github.com/vllm-project/vllm/issues/23786: Define PluggableLayer and apply to MLA as an example. ⏎ ## Test Plan ⏎ All ci should pass. ⏎ ## Test Result ⏎ All ci should pass. ⏎ --- ⏎ [details omitted]

### L3-b4f64e5b02  (L3, 2026-01-21, sha b4f64e5b02a9, PR #32491)
TITLE: Update FlashMLA (#32491)
SOURCES: path_core, subject_keyword, symbol_pickaxe, dependency_pin, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_build, L3.mla.flashmla_sparse
FILES: cmake/external_projects/flashmla.cmake (+40/-10); vllm/v1/attention/backends/mla/flashmla_sparse.py (+64/-20); tests/kernels/attention/test_flashmla_sparse.py (+1/-1); tests/v1/attention/test_sparse_mla_backends.py (+64/-11)
LABELS: ready, ci/build, v1, ready-run-all-tests
BODY: WIP Update FlashMLA, vLLM side of https://github.com/vllm-project/FlashMLA/pull/12

### L3-24dc30f7ff  (L3, 2026-01-21, sha 24dc30f7ff4b, PR #32799)
TITLE: [ModelRunner V2] Don't pin reused flashinfer tensors (#32799)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+6/-1)
LABELS: v1, nvidia
BODY: Since we do not have explicit synchronization in ModelRunnerV2, we do not pin reused CPU buffers to avoid a race condition between step N async copies to GPU and step N+1 buffer updates.

### L3-5e00b561cd  (L3, 2026-01-21, sha 5e00b561cddd, PR #32820)
TITLE: [Model Runner V2] Do not error on attention backends (#32820)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/model_runner.py (+0/-10)
LABELS: v1
BODY: 

### L3-889722f3bf  (L3, 2026-01-21, sha 889722f3bfda, PR #32810)
TITLE: [FlashMLA] Update FlashMLA to expose new arguments (#32810)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.mla.flashmla_v1_adapter, L3.mla.flashmla_build, L3.mla.flashmla_sparse
FILES: cmake/external_projects/flashmla.cmake (+19/-2); setup.py (+10/-0); vllm/third_party/flashmla/__init__.py (+1/-0); vllm/v1/attention/backends/mla/flashmla.py (+45/-50); vllm/v1/attention/backends/mla/flashmla_sparse.py (+8/-29); vllm/v1/attention/ops/flashmla.py (+47/-135); .gitignore (+3/-0); tests/v1/attention/test_sparse_mla_backends.py (+0/-1)
LABELS: ci/build, v1, ready-run-all-tests
BODY: The FlashMLA update contains new arguments; update the interface to expose these

### L3-a810299838  (L3, 2026-01-21, sha a81029983873, PR #32835)
TITLE: [ROCm][CI][Docs] Add comment explaining TRITON_ATTN fallback for ROCm (#32835)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/spec_decode/test_acceptance_length.py (+2/-0)
LABELS: rocm, speculative-decoding, v1
BODY: This PR adds a clarifying comment as requested in #32787 by @DarkLight1337. ⏎  ⏎ The comment explains why `TRITON_ATTN` is used as the fallback attention backend for ROCm platforms.

### L3-63227accf5  (L3, 2026-01-21, sha 63227accf5af, PR #31246)
TITLE: [Kernel] Add topk_sigmoid kernel (#31246)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_fused_topk.py (+99/-0); csrc/moe/moe_ops.h (+7/-1); csrc/moe/topk_softmax_kernels.cu (+242/-101); csrc/moe/torch_bindings.cpp (+9/-1); tests/kernels/moe/test_fused_topk.py (+137/-0); tests/model_executor/test_enabled_custom_ops.py (+17/-3); vllm/_aiter_ops.py (+39/-0); vllm/_custom_ops.py (+25/-1); vllm/model_executor/layers/fused_moe/config.py (+2/-2); vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py (+97/-1); (+3 more)
LABELS: performance, ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ This PR add `topk_sigmoid` kernel. Currently fused topk only support softmax function, this PR add sigmoid function support, also support `e_score_correction_bias`. ⏎  ⏎ * Support `sigmoid` function ⏎ * Support optional `bias`. If `bias` is not null, use biased values as weights for selection, but return the raw score as `topk_weights`, following torch reference implementation [here](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/mo …[truncated]

### L3-6c20e89c02  (L3, 2026-01-21, sha 6c20e89c0209, PR #29287)
TITLE: [ROCm][Deepseekv3.2] Refactor Sparse Indexer as CustomOp (#29287)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse, L3.platform.rocm_selection
FILES: vllm/v1/attention/backends/mla/indexer.py (+6/-0); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+110/-10); vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+518/-80); vllm/_aiter_ops.py (+12/-0); vllm/config/compilation.py (+1/-0); vllm/model_executor/layers/sparse_attn_indexer.py (+318/-0); vllm/model_executor/models/deepseek_v2.py (+14/-233); vllm/platforms/rocm.py (+3/-0)
LABELS: rocm, ready, v1, deepseek
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎ This PR optimize the deepseekv3.2's performance on AMD's device, and separate `SparseAttnIndexer` out as a `CustomOp` as it contains lots of heavy kernels like `fp8_mqa_logits` or `fp8_paged_mqa_logits`. The solution might vary on different platform for this indexer op in order to achieve optimal performance on vllm. The main change include: ⏎ - Separate `SparseAttnIndexer` out as `CustomOp`. ⏎ - Integrate `mla_decode_fwd` to `AiterMLASpar …[truncated]

### L3-6437ff1fb9  (L3, 2026-01-22, sha 6437ff1fb9dd, PR #32812)
TITLE: [Deprecation] Remove deprecated environment variables (#32812)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.platform.rocm_selection
FILES: .buildkite/test-amd.yaml (+2/-2); tests/v1/spec_decode/test_acceptance_length.py (+1/-1); vllm/config/attention.py (+0/-46); vllm/envs.py (+0/-67); vllm/platforms/rocm.py (+4/-1); vllm/usage/usage_lib.py (+0/-1)
LABELS: rocm, speculative-decoding, ready, ci/build, v1
BODY: ## Purpose ⏎  ⏎ As v0.14.0 has been out, we can remove these deprecated envs. ⏎  ⏎ CC: @MatthewBonanni

### L3-421012b63a  (L3, 2026-01-22, sha 421012b63ae2, PR #30692)
TITLE: OffloadingConnector: Support kernel_block_size != block_size (#30692)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+1/-4); csrc/cache.h (+1/-0); csrc/torch_bindings.cpp (+2/-1); tests/kernels/attention/test_cache.py (+29/-6); tests/v1/kv_offload/test_cpu_gpu.py (+51/-45); vllm/_custom_ops.py (+25/-2); vllm/v1/kv_offload/worker/cpu_gpu.py (+47/-56)
LABELS: ready, v1
BODY: This PR enables the offloading connector to work for cases where the kernel block size is different than vLLM's logical block size. ⏎  ⏎ To efficiently copy blocks when the logical block size is greater ⏎ than the block size, we add an additional parameter to the swap_blocks ⏎ function that determines the size per each contiguous copy. ⏎  ⏎ This fix addresses the same issue that was fixed in the NixlConnector in #28677. ⏎  ⏎ --- ⏎  ⏎ > [!NOTE] ⏎ > <sup>[Cursor Bugbot] …[truncated]

### L3-44f08af3a7  (L3, 2026-01-22, sha 44f08af3a75e, PR #30141)
TITLE: Add llmcompressor fp8 kv-cache quant (per-tensor and per-attn_head) (#30141)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: csrc/cache_kernels.cu (+33/-12); vllm/attention/layer.py (+37/-36); vllm/v1/attention/backend.py (+1/-0); vllm/v1/attention/backends/flash_attn.py (+15/-6); docs/features/quantization/quantized_kvcache.md (+139/-114); tests/kernels/attention/test_cache.py (+34/-13); tests/quantization/test_compressed_tensors.py (+21/-3); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py (+168/-22); vllm/model_executor/layers/quantization/input_quant_fp8.py (+4/-1); vllm/model_executor/layers/quantization/utils/quant_utils.py (+47/-24); (+8 more)
LABELS: documentation, speculative-decoding, ready, v1, llama
BODY: **TLDR**: this PR adds support to load and run `llm-compressor` models with FP8 KV-cache and attention quantization. In addition to the standard "per-tensor" quantization, it adds support for "per-attention-head" quantization. ⏎  ⏎ ## Summary ⏎ 1. enable using the existing pathway of "per-tensor" KV-cache (and query) FP8 quantization with scales calibrated through `llm-compressor` ⏎ 2. Flash Attention v3 backend supports "finer-grained" scales, i.e. one  …[truncated]

### L3-eb1629da24  (L3, 2026-01-22, sha eb1629da2496, PR #32346)
TITLE: [ROCm][CI] Fix AITER test flakiness by using explicit attention backend (#32346)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+1/-1); tests/models/language/generation/test_common.py (+5/-1); vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=AMD_Instinct_MI325X.json (+4/-4)
LABELS: rocm, ready, ci/build
BODY: ## Summary ⏎  ⏎ This PR fixes test failures for `TitanML/tiny-mixtral` when running AITER tests on ROCm as part of the full test suite. ⏎  ⏎ ## Problem ⏎  ⏎ When running the test suite as a group, the following tests were failing: ⏎ - `test_models[True-True-5-32-TitanML/tiny-mixtral]` ⏎ - `test_models[False-True-5-32-TitanML/tiny-mixtral]` ⏎  ⏎ **Notably, these tests pass when run individually.** The failures only occur when running multiple tests together, showing  …[truncated]

### L3-955b43a5a5  (L3, 2026-01-22, sha 955b43a5a514, PR #32795)
TITLE: [Bugfix][Attention] Explicitly report support for kv_cache_dtype bfloat16 (#32795)
SOURCES: path_core, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashmla_sparse, L3.dispatch.abstract_interface, L3.flex_attention
FILES: vllm/v1/attention/backend.py (+2/-2); vllm/v1/attention/backends/flash_attn.py (+1/-1); vllm/v1/attention/backends/flashinfer.py (+1/-0); vllm/v1/attention/backends/flex_attention.py (+1/-1); vllm/v1/attention/backends/mla/cutlass_mla.py (+1/-0); vllm/v1/attention/backends/mla/flashattn_mla.py (+4/-1); vllm/v1/attention/backends/mla/flashinfer_mla.py (+1/-0); vllm/v1/attention/backends/mla/flashmla.py (+1/-0); vllm/v1/attention/backends/mla/flashmla_sparse.py (+5/-1); vllm/v1/attention/backends/mla/triton_mla.py (+4/-1); (+3 more)
LABELS: bug, ready, v1, cpu, nvidia
DEEP_STUDY: deep-study correctness case vllm:955b43a5a5: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎ Attention backends all support bfloat16, but only report support for `auto`. In cases where the automatically resolved dtype is not bfloat16, this can lead to failures. ⏎  ⏎ Related issue: #32732 - setting `--kv-cache-dtype=bfloat16` should work as a workaround, but does not, because TRITON_MLA does not report support for bfloat16. ⏎  ⏎ ## Test Plan ⏎ `vllm serve nvidia/DeepSeek-R1-0528-NVFP4 --attention-backend=TRITON_MLA --kv-cache-dtype=bfloa …[truncated]

### L3-5e4e0e51f4  (L3, 2026-01-22, sha 5e4e0e51f4fb, PR #32806)
TITLE: [torch.compile] Compile `CustomOp.forward_native` for `SiluAndMul` and `QuantFP8` to avoid raw torch ops inside opaque custom ops (#32806)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/test_silu_mul_quant_fusion.py (+2/-1); tests/kernels/core/test_activation.py (+1/-1); vllm/compilation/matcher_utils.py (+1/-0); vllm/model_executor/custom_op.py (+42/-7); vllm/model_executor/layers/activation.py (+2/-2); vllm/model_executor/layers/quantization/input_quant_fp8.py (+3/-1); vllm/model_executor/layers/rotary_embedding/common.py (+1/-1)
LABELS: ready, torch.compile, ready-run-all-tests
ISSUES: #32059 [Feature]: Use platform op if inside a custom op
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ When `CustomOp.forward_native` is invoked from within another opaque torch custom op (e.g. `fused_moe`, `unified_attention`), it is hidden from the model-level torch.compile. In this case, executing `forward_native` directly executes eager PyTorch ops, which can significantly hurt performance. ⏎  ⏎ In this PR, we compile forward_native manually to avoid this scenario. Existing `CustomOp` invocations visible to model-level compilation are n …[truncated]

### L3-5206e5e28c  (L3, 2026-01-23, sha 5206e5e28c25, PR #30877)
TITLE: [V1][Hybrid] Mamba Prefix Caching with align mode (#30877)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+38/-0); tests/v1/core/test_single_type_kv_cache_manager.py (+18/-7); tests/v1/e2e/test_mamba_prefix_cache.py (+764/-0); vllm/config/cache.py (+10/-0); vllm/config/vllm.py (+11/-0); vllm/engine/arg_utils.py (+6/-0); vllm/model_executor/layers/mamba/abstract.py (+1/-0); vllm/model_executor/layers/mamba/mamba_mixer.py (+3/-3); vllm/model_executor/layers/mamba/mamba_mixer2.py (+6/-6); vllm/model_executor/layers/mamba/mamba_utils.py (+95/-0); (+32 more)
LABELS: documentation, ready, v1, qwen
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: The cleaned-up version of #29272  ⏎  ⏎ ## Purpose ⏎  ⏎ This PR enhances the design of #28176 , adopting the same memory layout as FullAttention while adding support for decode caching and speculative decoding. ⏎  ⏎ The core idea of this Mamba Prefix-Caching implementation (referred to as LPC) is to directly cache Mamba states through **block-aligned scheduling**. This approach enables rapid support for Prefix-caching in Mamba models **without modifications t …[truncated]

### L3-68b0a6c1ba  (L3, 2026-01-23, sha 68b0a6c1baf3, PR #30443)
TITLE: [CI][torch nightlies] Use main Dockerfile with flags for nightly torch tests (#30443)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+146/-19); docker/Dockerfile.nightly_torch (+8/-0); docs/assets/contributing/dockerfile-stages-dependency.png (+0/-0); use_existing_torch.py (+49/-13)
LABELS: documentation, ready, ci/build, ready-run-all-tests
BODY: Use standard Docker image instead of torch_nightly image for PyTorch nightlies testing and CI runs. ⏎  ⏎ Moving this from https://github.com/vllm-project/ci-infra/pull/239 to a branch on upstream for testing purposes outlined at https://github.com/vllm-project/ci-infra?tab=readme-ov-file#how-to-test-changes-in-this-repo ⏎  ⏎ Tests to confirm: ⏎ 1. Baseline (my `vllm` fork matching HEAD, no `ci-infra` changes) at https://buildkite.com/vllm/ci/builds/42874/s …[truncated]

### L3-1fb648bf10  (L3, 2026-01-23, sha 1fb648bf107e, PR #32886)
TITLE: [Bugfix] Fix FP8 MoE EP Weight Loading for ModelOpt Llama4 (#32886)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/llama4.py (+21/-1)
LABELS: bug, ready, llama
BODY: ## Purpose ⏎ #32862 ⏎   Add a version-guarded fallback in Llama4 MoE weight loading to avoid CPU FP8 indexing on older PyTorch releases. For torch < 2.11, weights are temporarily cast to FP16 for indexing and then cast back, preventing index_cpu errors while leaving newer versions unchanged. ⏎ ## Test Plan ⏎ ``` ⏎  VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel \ ⏎ VLLM_USE_FLASHINFER_MOE_FP8=0 \ ⏎ vllm bench throughput --model=nvidia/Llama-4-Scout-17 …[truncated]

### L3-160c6fa387  (L3, 2026-01-23, sha 160c6fa3872a, PR #32698)
TITLE: [Misc] Add `get_name` to missing AttentionBackends (#32698)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+4/-0); vllm/v1/attention/backends/gdn_attn.py (+4/-0); vllm/v1/attention/backends/linear_attn.py (+4/-0); vllm/v1/attention/backends/mamba1_attn.py (+4/-0); vllm/v1/attention/backends/mamba2_attn.py (+8/-1); vllm/v1/attention/backends/short_conv_attn.py (+4/-0)
LABELS: ready, v1
BODY: There's a few attention backends that are missing this meta-method, most notably the Mamba ones. ⏎ Albeit the `get_name()` descriptor isn't a functional one, I would expect every backend to define a simple unique tag for identification, adhering to the new structure of AttentionBackends. ⏎  ⏎ I don't have a strong opinion on how this should be carried out (I am also fine with baseclass providing a default), but I tried adding a basic decorator util to  …[truncated]

### L3-7ef5873752  (L3, 2026-01-23, sha 7ef587375280, PR #32722)
TITLE: [CI] Fix mypy for `vllm/v1/structured_output` (#32722)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fa_utils, L3.mla.rocm_aiter, L3.mla.aiter_triton
FILES: vllm/v1/attention/backends/fa_utils.py (+6/-4); vllm/v1/attention/backends/mla/aiter_triton_mla.py (+1/-1); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+1/-1); tools/pre_commit/mypy.py (+1/-1); vllm/reasoning/abs_reasoning_parsers.py (+2/-2); vllm/reasoning/basic_parsers.py (+2/-2); vllm/reasoning/deepseek_v3_reasoning_parser.py (+1/-1); vllm/reasoning/gptoss_reasoning_parser.py (+1/-1); vllm/reasoning/holo2_reasoning_parser.py (+1/-1); vllm/reasoning/hunyuan_a13b_reasoning_parser.py (+1/-1); (+8 more)
LABELS: rocm, structured-output, ready, v1, deepseek
BODY: ## Purpose ⏎  ⏎ Part of the https://github.com/vllm-project/vllm/issues/26533 ⏎  ⏎ ## Test ⏎  ⏎ At first: ⏎  ⏎ ```bash ⏎ (yewentao256) [yewentao256@nma-h200-isolated-0-preserve vllm-source]$ pre-commit run --hook-stage manual mypy-3.10 -a ⏎ [INFO] Initializing environment for local:dockerfile-parse. ⏎ Run mypy for Python 3.10.................................................Failed ⏎ - hook id: mypy-3.10 ⏎ - exit code: 1 ⏎  ⏎ $ mypy --python-version 3.10  ⏎ vllm/v1/structured_out …[truncated]

### L3-243e78c20f  (L3, 2026-01-23, sha 243e78c20fd7, PR #32927)
TITLE: [Benchmark][Bugfix] Fix race condtion when starting server for sweep benchmark (#32927)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/benchmarks/sweep/serve.py (+13/-0); vllm/benchmarks/sweep/server.py (+24/-0)
LABELS: bug, performance, ready
BODY: ## Purpose ⏎ - Currently, there is a race condition that bench command is executed too early before server become ready for `vllm bench sweep serve`: ⏎ ``` ⏎ [BEGIN SERVER] ⏎ Server overrides: {'mm_encoder_attn_backend': 'FLASH_ATTN'} ⏎ Server command: ['vllm', 'serve', '/home/mozf/LLM/Qwen3-VL-4B-Instruct/', '--enforce-eager', '--max-model-len', '32768', '--mm-processor-cache-gb=0', '--media-io-kwargs.video.num_frames', '-1', '--mm-encoder-attn-backend',  …[truncated]

### L3-586a57ad7e  (L3, 2026-01-23, sha 586a57ad7ede, PR #32614)
TITLE: fix: Add glm4_moe_lite to MLA detection (#32614)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashinfer, L3.platform.cuda_selection
FILES: vllm/model_executor/layers/attention/mla_attention.py (+23/-4); vllm/platforms/cuda.py (+19/-6); vllm/v1/attention/backends/mla/flashinfer_mla.py (+26/-0); vllm/transformers_utils/model_arch_config_convertor.py (+2/-0)
LABELS: ready, v1, nvidia
BODY: ## Summary ⏎  ⏎ - Add `glm4_moe_lite` and `glm4_moe_lite_mtp` to `is_deepseek_mla()` check in `model_arch_config_convertor.py` ⏎  ⏎ GLM-4.7-Flash (`glm4_moe_lite`) uses Multi-head Latent Attention (MLA) via `Glm4MoeLiteMLAAttention` (which inherits from `DeepseekV2MLAAttention`) but was missing from the MLA detection. ⏎  ⏎ Without this fix, vLLM falls back to standard KV caching instead of efficient MLA caching, resulting in ~4x higher KV cache memory usage. …[truncated]

### L3-13d8746c54  (L3, 2026-01-23, sha 13d8746c5455, PR #32815)
TITLE: [Feature]: Remove DtoH Copy for lfm2_vl On Default Stream (#32815)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+8/-4); vllm/model_executor/models/lfm2_siglip2.py (+89/-92); vllm/model_executor/models/lfm2_vl.py (+129/-51); vllm/v1/attention/backends/gdn_attn.py (+24/-4); vllm/v1/attention/backends/mamba_attn.py (+10/-7)
LABELS: ready, v1
BODY: Purpose ⏎ This PR removes all CUDA Device-to-Host (DtoH) memcpy/syncs observed during lfm2_vl preprocess on the default compute stream. ⏎  ⏎ Main changes: ⏎  ⏎ Run LFM2-VL’s SigLIP2 vision path fully packed (unpadded) end-to-end so we don’t trigger tiny syncs from CUDA nonzero / padding logic. ⏎ Avoid host sync in the vision attention path by keeping max_seqlen on CPU (prevent .item()-style sync in FA wrapper). ⏎ Remove remaining preprocess DtoH in the ShortCo …[truncated]

### L3-a8eb1182f1  (L3, 2026-01-23, sha a8eb1182f172, PR #32885)
TITLE: [CI][Models] Add VLM Support for Sequence Classification Conversion (#32885)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+3/-1); vllm/model_executor/layers/layernorm.py (+39/-15); vllm/model_executor/models/adapters.py (+113/-23)
LABELS: ready, v1
BODY: This PR enables Vision-Language Models (VLMs) like Gemma 3 to be converted to sequence classifiers using the `no_post_processing` and `from_2_way_softmax` methods. Additionally, it fixes two PyTorch compiler warnings that were causing noise in the logs. ⏎  ⏎ ## Changes ⏎  ⏎ ### 1. `vllm/model_executor/models/adapters.py` ⏎  ⏎ - Added `_get_language_model_for_seq_cls()` helper function to correctly retrieve the inner language model component from VLMs ⏎ - Updat …[truncated]

### L3-a28b94e6ef  (L3, 2026-01-23, sha a28b94e6ef60, PR #25954)
TITLE: [Performance] Split FlashAttn attention and cache update (#25954)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/layer.py (+85/-16); vllm/model_executor/layers/attention/cross_attention.py (+48/-5); vllm/utils/torch_utils.py (+2/-1); vllm/v1/attention/backend.py (+3/-0); vllm/v1/attention/backends/flash_attn.py (+45/-26); vllm/v1/worker/gpu/attn_utils.py (+12/-0); vllm/v1/worker/gpu/cudagraph_utils.py (+12/-4); vllm/v1/worker/gpu/model_runner.py (+10/-0); vllm/v1/worker/gpu_model_runner.py (+127/-14); vllm/v1/worker/gpu_ubatch_wrapper.py (+8/-0); (+11 more)
LABELS: documentation, speculative-decoding, ready, v1, qwen, kv-connector, nvidia, ready-run-all-tests
DEEP_STUDY: deep-study performance PR ()
BODY: This PR creates codepaths for separating KV Cache update and Attention forward op. It also implements this split for FlashAttn backend. This separation facilitates future unwrapping. ⏎  ⏎ #### E2E tests: ⏎ ran inference on Blackwell machine with ⏎ ``` ⏎ llm = LLM(model="deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True) ⏎ ``` ⏎ and both `VLLM_MLA_DISABLE=1` (to test the split) and `VLLM_MLA_DISABLE=0` (to test if this PR does not affect backends that don't …[truncated]

### L3-0b9a735e11  (L3, 2026-01-24, sha 0b9a735e113e, PR #32981)
TITLE: [Tests] Clarify pytest skip reasons with actionable context (#32981)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/samplers/test_beam_search.py (+2/-2); tests/v1/sample/test_topk_topp_sampler.py (+5/-1)
LABELS: ready, v1
BODY: ## Summary ⏎ Replace vague FIXME comments in test skip markers with more descriptive reasons: ⏎  ⏎ - `tests/v1/sample/test_topk_topp_sampler.py`: Changed from "FIXME: This test is failing right now" to explain the FlashInfer top-k/top-p renorm comparison issue ⏎ - `tests/samplers/test_beam_search.py`: Changed from "FIXME: This fails on V1 right now" to clarify that V1 engine does not yet support beam search ⏎  ⏎ ## Test Plan ⏎ - Run `pytest --collect-only` on  …[truncated]

### L3-97ef11dd34  (L3, 2026-01-24, sha 97ef11dd3406, PR #32944)
TITLE: [ROCm][ViT] Enable Flash Attention Triton backend on RDNA3/RDNA4 (#32944)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+34/-1)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ Flash Attention's CK backend only supports CDNA GPUs (gfx90a/gfx942/gfx950). On RDNA3/RDNA4, vLLM falls back to PyTorch SDPA for vision attention, which is significantly slower. Flash Attention 2.8.3+ includes a Triton backend that supports RDNA architectures. ⏎  ⏎ ## Changes ⏎  ⏎ - Added `flash_attn_triton_available()` in `vllm/platforms/rocm.py` to detect RDNA GPUs and verify Triton backend availability ⏎ - Updated `get_vit_attn_backend()` to …[truncated]

### L3-da5e7b12be  (L3, 2026-01-24, sha da5e7b12beb6, PR #32950)
TITLE: [MLA] Fuse cat and qaunt for fp8 kv-cache (#32950)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+41/-20)
LABELS: documentation, performance, ready, deepseek
ISSUES: #32055 [Feature]: Copy Ops in FP8 MLA
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Main: ⏎ <img width="976" height="71" alt="Screenshot 2026-01-23 at 2 25 12 PM" src="https://github.com/user-attachments/assets/a63e6a69-e08e-4925-b61a-df53133fe862" /> ⏎ PR: ⏎ <img width="950" height="68" alt="Screenshot 2026-01-23 at 2 25 24 PM" src="https://github.com/user-attachments/assets/329aa1d7-c3c7-4a19-98d0-ec88e6c1314f" />

### L3-fcb9df99bd  (L3, 2026-01-24, sha fcb9df99bd7d, PR #32520)
TITLE: [Perf][Kernel] Optimize FP4 quantization kernels (SM100F) (#32520)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/bench_nvfp4_quant.py (+55/-22); csrc/ops.h (+2/-1); csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu (+59/-20); csrc/quantization/fp4/nvfp4_experts_quant.cu (+4/-4); csrc/quantization/fp4/nvfp4_quant_entry.cu (+6/-3); csrc/quantization/fp4/nvfp4_quant_kernels.cu (+141/-47); csrc/quantization/fp4/nvfp4_utils.cuh (+171/-25); csrc/torch_bindings.cpp (+2/-1); tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py (+6/-2); tests/kernels/quantization/test_nvfp4_quant.py (+28/-0); (+8 more)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎ This PR introduces further optimizations to the FP4 quantization kernels in vLLM, delivering substantial performance improvements across all tested shapes and batch sizes. ⏎  ⏎ The main change is the use of **256-bit `load.global` instructions** enabled via **PTX 8.8**, which significantly reduces global memory instruction count and improves effective bandwidth. This optimization is **hardware- and toolchain-gated** and applies **only to S …[truncated]

### L3-9ad7f89f55  (L3, 2026-01-24, sha 9ad7f89f55b7, PR #31972)
TITLE: [Models]: Make Multimodal config implicit in ViT implementation (#31972)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/mm_encoder_attention.py (+0/-9); vllm/model_executor/models/clip.py (+4/-24); vllm/model_executor/models/deepencoder.py (+0/-3); vllm/model_executor/models/deepseek_ocr.py (+0/-1); vllm/model_executor/models/dots_ocr.py (+5/-34); vllm/model_executor/models/eagle2_5_vl.py (+0/-1); vllm/model_executor/models/ernie45_vl.py (+1/-12); vllm/model_executor/models/glm4_1v.py (+5/-30); vllm/model_executor/models/hunyuan_vision.py (+5/-26); vllm/model_executor/models/hyperclovax_vision.py (+1/-6); (+28 more)
LABELS: speculative-decoding, ready, qwen, deepseek
BODY: ## Purpose ⏎ - Currently, we are passing `MultimodalConfig` through whole MMEncoder only to enable data parallel and attention backend overrides ⏎ - This PR makes it implicit through `get_current_vllm_config`. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-151e5451c2  (L3, 2026-01-25, sha 151e5451c2eb, PR #33016)
TITLE: [Doc] Add Qwen2.5 models to batch invariance tested models (#33016)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/features/batch_invariance.md (+1/-0)
LABELS: documentation, ready, qwen
BODY: ## Description ⏎ Add Qwen2.5 model family to the validated batch invariance models list: ⏎ - Qwen2.5-0.5B-Instruct ⏎ - Qwen2.5-1.5B-Instruct ⏎ - Qwen2.5-3B-Instruct ⏎ - Qwen2.5-7B-Instruct ⏎ - Qwen2.5-14B-Instruct ⏎ - Qwen2.5-32B-Instruct ⏎  ⏎ Tested on H200 with both FLASH_ATTN and FLASHINFER backends. ⏎  ⏎ Related to #27433 ⏎  ⏎ ## Purpose ⏎  ⏎ Add Qwen2.5 model family to the batch invariance documentation as validated models. ⏎  ⏎ This addresses the "Help needed for validation …[truncated]

### L3-566cdb6cfb  (L3, 2026-01-25, sha 566cdb6cfb89, PR #33033)
TITLE: [CI] Fix MHA attention test failure (AttributeError when model_config is None in ViT attention backend) (#33033)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/vision.py (+4/-2)
LABELS: ready
ISSUES: #33028 [CI Failure]: MultiModal Tests
BODY: Fixes `AttributeError: 'NoneType' object has no attribute 'multimodal_config'` in `get_vit_attn_backend()` and `is_vit_use_data_parallel()` functions. ⏎  ⏎ Breaking Commit: `9ad7f89f5` - [Models]: Make Multimodal config implicit in ViT implementation (#31972) ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/33028 ⏎  ⏎ ## Root Cause ⏎  ⏎ The breaking commit introduced new functions `get_vit_attn_backend()` and `is_vit_use_data_parallel()` in `vllm/model_execu …[truncated]

### L3-f4a0921c9c  (L3, 2026-01-26, sha f4a0921c9c11, PR #32873)
TITLE: [Performance] Tune Mamba selective scan kernel for B200 (#32873)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/mamba_mixer2.py (+5/-0); vllm/model_executor/layers/mamba/ops/mamba_ssm.py (+21/-11)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎  ⏎ Tune config of `selective_state_update` for **Nemotron Nano NVFP4** on **B200 (TP1)**. ⏎  ⏎ ## Test Plan ⏎  ⏎ Compare vLLM performance (Nano NVFP4, B200, TP1) with main branch and this branch/PR. ⏎  ⏎ ## Test Result ⏎  ⏎ Server command: ⏎ ``` ⏎ export VLLM_USE_FLASHINFER_MOE_FP4=1 ⏎ export VLLM_FLASHINFER_MOE_BACKEND=throughput ⏎  ⏎ export MAMBA_CDTYPE=float32 ⏎  ⏎ vllm serve $MODEL_PATH \ ⏎ --served-model-name my_model \ ⏎ --async-scheduling \ ⏎ --dtype auto --kv-cache …[truncated]

### L3-19ab0f7ce5  (L3, 2026-01-26, sha 19ab0f7ce56a, PR #33073)
TITLE: [Bugfix] Fix Voxtral streaming slot_mapping (#33073)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/whisper_causal.py (+40/-0)
LABELS: bug, ready
BODY: PR https://github.com/vllm-project/vllm/pull/25954 broke custom attention backend registered in vllm that manipulate slot_mapping. This PR fixes it by "replaying" the strategy used for CrossAttn to deal with the same issue in the linked PR.

### L3-67fe677c53  (L3, 2026-01-26, sha 67fe677c53e7, PR #31099)
TITLE:  [FIX] Always support TP > 4 for FP4 Gemm (#31099)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py (+26/-4); vllm/model_executor/layers/quantization/modelopt.py (+21/-2); vllm/model_executor/layers/quantization/utils/quant_utils.py (+67/-0)
LABELS: ready
BODY: ## Summary ⏎  ⏎ This PR enables FP4 (NVFP4) quantization to work with TP  >=  4 ⏎  ⏎ ## Background ⏎  ⏎ Previously, FP4 quantized models would fail to initialize with TP=4/8 due to kernel alignment requirements. The FlashInfer-CUTLASS FP4 GEMM kernels require the N/K-dimension to be divisible by 32. ⏎  ⏎ ## Changes ⏎  ⏎ This PR makes the following changes to support TP >= 4, by padding the weights accordingly. In addition, in cases where we pad the K dim of the weig …[truncated]

### L3-8caffd92df  (L3, 2026-01-26, sha 8caffd92dfb0, PR #33104)
TITLE: [Bugfix][MXFP4] Call `trtllm_fp4_block_scale_moe` with kwargs (#33104)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/mxfp4.py (+26/-26)
LABELS: bug, ready
BODY: ## Purpose ⏎ We are maintaining our own version of flashinfer in our own infra, PR #30993 breaks our internal integration because positioned arg tile_tokens_dim is removed from trtllm_fp4_block_scale_moe.  ⏎ ``` ⏎ [0;36m(Worker_TP0 pid=1547236)^[[0;0m ERROR 01-25 13:49:16 [multiproc_executor.py:839] RuntimeError: Error in function 'TrtllmGenBatchedGemmRunner' at flashinfer/csrc/trtllm_batched_gemm_runner.cu:126: No kernel found for the given options: m …[truncated]

### L3-5a93b9162b  (L3, 2026-01-27, sha 5a93b9162bfe, PR #32567)
TITLE: [MoE Refactor] Integrate Naive Prepare Finalize into MK (#32567)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+1/-1); .buildkite/test-pipeline.yaml (+1/-1); .buildkite/test_areas/kernels.yaml (+1/-1); benchmarks/kernels/benchmark_cutlass_moe_nvfp4.py (+2/-2); docs/design/moe_kernel_features.md (+1/-2); tests/kernels/moe/modular_kernel_tools/common.py (+8/-4); tests/kernels/moe/modular_kernel_tools/mk_objects.py (+3/-25); tests/kernels/moe/test_flashinfer.py (+1/-6); tests/kernels/moe/test_flashinfer_moe.py (+1/-6); tests/kernels/moe/test_nvfp4_moe.py (+1/-1); (+36 more)
LABELS: documentation, performance, rocm, ready, ci/build, llama, cpu, nvidia, ready-run-all-tests
BODY: #### SUMMARY ⏎ * integrate Naive A2A kernels into the modular kernel framework ⏎ * unify how FP8 and NVFP4 quant methods "own" the Modular Kernels for both TP and DP/EP. Now, the Quant Methods "own" the Modular Kernel for both cases ⏎  ⏎ #### Sets up: ⏎ * removing naive Dispatch/Combine from `FusedMoE.forward()` ⏎ * removing `ModularKernelMethod` ⏎ * unifying structure of Kernels owned by Quant Methods ⏎  ⏎ #### How: ⏎ * update `reducescatter_allgather` and `naive`  …[truncated]

### L3-58996f3589  (L3, 2026-01-27, sha 58996f358943, PR #32976)
TITLE: [AMD][Kernel][BugFix] Use correct scale in concat_and_cache_ds_mla_kernel when on gfx942 (#32976)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+9/-7)
LABELS: bug, rocm, ready
BODY: ## Purpose ⏎ This PR updates `concat_and_cache_ds_mla_kernel` to use the correct scale divisor when running on `gfx942` architectures on ROCm.  The `tile_size` divisor was `448.0` which does not work on AMD platforms with arch of `gfx942`, e.g. MI300, MI325.  This PR updates the divisor to be `224.0 `on `gfx942`. ⏎  ⏎ Additionally, I consolidated the scale divisor into a constexpr float. ⏎ ## Test Plan ⏎ Use lm_eval to check accuracy on `DeepSeek-R1 `model …[truncated]

### L3-da8d0c441a  (L3, 2026-01-27, sha da8d0c441aad, PR #32042)
TITLE: [AMD][QWEN3-NEXT] FP8 Tunings (#32042)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_w8a8_block_fp8.py (+2/-2); vllm/model_executor/layers/quantization/utils/configs/N=1024,K=2048,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json (+146/-0); vllm/model_executor/layers/quantization/utils/configs/N=12288,K=2048,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json (+146/-0); vllm/model_executor/layers/quantization/utils/configs/N=2048,K=4096,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json (+146/-0); vllm/model_executor/layers/quantization/utils/configs/N=9216,K=2048,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json (+146/-0)
LABELS: performance, rocm, ready, qwen
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎   Add pre-tuned FP8 block-scaled matmul kernel configs for Qwen3-Next-80B on 1 MI300X to improve the latency. ⏎  ⏎   **Configs added:** ⏎   - `N=12288,K=2048,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json` ⏎   - `N=2048,K=4096,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json` ⏎   - `N=1024,K=2048,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json` ⏎   - `N=9216,K=2048,devic …[truncated]

### L3-a608b4c6c2  (L3, 2026-01-27, sha a608b4c6c2b9, PR #32064)
TITLE: [5/N][Attention] Finish eliminating `vllm/attention` folder (#32064)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.rocm.aiter_fa, L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/__init__.py (+26/-0); vllm/model_executor/layers/attention/attention.py (+42/-315); vllm/model_executor/layers/attention/chunked_local_attention.py (+1/-1); vllm/model_executor/layers/attention/cross_attention.py (+1/-1); vllm/model_executor/layers/attention/encoder_only_attention.py (+1/-1); vllm/model_executor/layers/attention/kv_transfer_utils.py (+1/-1); vllm/model_executor/layers/attention/mla_attention.py (+354/-17); vllm/model_executor/layers/attention/static_sink_attention.py (+1/-1); vllm/model_executor/layers/mla.py (+1/-1); .buildkite/test-amd.yaml (+2/-1); (+141 more)
LABELS: documentation, rocm, speculative-decoding, ready, ci/build, v1, llama, qwen, deepseek, gpt-oss
BODY: Merge https://github.com/vllm-project/vllm/pull/32060 before this. ⏎  ⏎ ## Purpose ⏎ Step 5 of #31919: This PR finishes eliminating the `vllm/attention` folder by doing the following: ⏎ * Split `vllm/attention/layer.py` into `vllm/model_executor/layers/attention/mla_attention.py` (`MLAAttention`, `unified_mla_attention`) and `vllm/model_executor/layers/attention/attention.py` (`Attention`, `unified_attention`) ⏎ * Move `vllm/attention/utils/kv_sharing_util …[truncated]

### L3-c568581ff3  (L3, 2026-01-27, sha c568581ff38a, PR #33112)
TITLE: Fix IndexError with encoder-decoder models when using Custom Paged Attention (#33112)
SOURCES: path_core, subject_keyword, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode, L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+10/-3); vllm/v1/attention/ops/chunked_prefill_paged_decode.py (+3/-2)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study correctness case vllm:c568581ff3: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎ Fix `IndexError: Dimension out of range` error when running encoder-decoder models (e.g., Whisper) when using the ROCm Attention (Custom Paged Attention) backend. ⏎ ``` ⏎ IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1) ⏎ ``` ⏎ During encoder-decoder cross-attention, key and value are `None` on decode steps (KV is already cached from the encoder). The ROCm Attention backend was unconditionally passing these to …[truncated]

### L3-3a6d5cbefd  (L3, 2026-01-27, sha 3a6d5cbefd97, PR #33102)
TITLE: [Perf] Optimize dcp allocate tensor (#33102)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/common.py (+3/-7)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ There is no need to allocate an additional tensor for view_as, we can optimize through that.

### L3-1cbccb6dba  (L3, 2026-01-27, sha 1cbccb6dbabd, PR #33177)
TITLE: [Attention] Use `has_flashinfer` helper (#33177)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+4/-11)
LABELS: ready
BODY: ## Purpose ⏎ Small PR to remove `is_flashinfer_available()` from `mla_attention.py` in favor of `vllm/utils/flashinfer.py`'s `has_flashinfer()` ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-1f3a2c2944  (L3, 2026-01-27, sha 1f3a2c2944f2, PR #33164)
TITLE: [Bugfix] Disable CG for Whisper+FA2 (#33164)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+20/-0)
LABELS: bug, ready, v1
BODY: Temporary fix to address https://github.com/vllm-project/vllm/issues/33091 for a tentative v0.15.0 release. As the issue is purely in accuracy and does not produce a crash, I believe it's better to patch it until fixed as to not mislead users deploying Whisper. ⏎  ⏎ This PR disables CG for encoder-decoder models when FA2 is detected. ⏎  ⏎ ## Test with ⏎  ⏎ ``` ⏎ python -m pytest -v -s -x tests/entrypoints/openai/test_transcription_validation_whisper.py ⏎  ⏎ # On F …[truncated]

### L3-e82fa448c4  (L3, 2026-01-28, sha e82fa448c40b, PR #26835)
TITLE: Add attention benchmarking tools (#26835)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/attention_benchmarks/README.md (+266/-0); benchmarks/attention_benchmarks/__init__.py (+44/-0); benchmarks/attention_benchmarks/batch_spec.py (+231/-0); benchmarks/attention_benchmarks/benchmark.py (+886/-0); benchmarks/attention_benchmarks/common.py (+503/-0); benchmarks/attention_benchmarks/configs/mla_decode.yaml (+61/-0); benchmarks/attention_benchmarks/configs/mla_mixed_batch.yaml (+60/-0); benchmarks/attention_benchmarks/configs/reorder_threshold.yaml (+88/-0); benchmarks/attention_benchmarks/configs/speculative_decode.yaml (+62/-0); benchmarks/attention_benchmarks/configs/standard_attention.yaml (+40/-0); (+2 more)
LABELS: performance, ready, nvidia
BODY: ## Purpose ⏎ Add tools for benchmarking attention backends. These can be used to perform parameter tuning as well as selecting optimal backends for particular configurations. These tools were built with substantial use of Claude Code. ⏎  ⏎ These were used to perform the tuning in #26846, #27363, and #27368, and will be useful for further studies in the future ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-2eb673a088  (L3, 2026-01-28, sha 2eb673a088a1, PR #33191)
TITLE: Add flake8-implicit-str-concat rules to Ruff (#33191)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.flashattn
FILES: vllm/v1/attention/backends/mla/flashattn_mla.py (+1/-1); .buildkite/performance-benchmarks/scripts/convert-results-json-to-markdown.py (+1/-1); csrc/quantization/machete/generate.py (+3/-3); examples/offline_inference/automatic_prefix_caching.py (+1/-1); examples/others/lmcache/disagg_prefill_lmcache_v1/disagg_proxy_server.py (+4/-4); pyproject.toml (+2/-0); tests/entrypoints/openai/tool_parsers/test_llama4_pythonic_tool_parser.py (+1/-1); tests/tool_parsers/test_deepseekv31_tool_parser.py (+8/-8); tests/utils.py (+1/-1); vllm/benchmarks/datasets.py (+1/-2); (+6 more)
LABELS: documentation, performance, frontend, ready, ci/build, v1, tool-calling, llama, deepseek, kv-connector
BODY: Inspired by https://github.com/vllm-project/vllm/pull/31694 ⏎  ⏎ Add https://docs.astral.sh/ruff/rules/#flake8-implicit-str-concat-isc rules to our Ruff config

### L3-36d450e3b8  (L3, 2026-01-28, sha 36d450e3b884, PR #33058)
TITLE: Adds FunAudioChat multimodal audio model support (#2) (#33058)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: examples/offline_inference/audio_language.py (+37/-0); tests/models/registry.py (+3/-0); vllm/model_executor/models/funaudiochat.py (+1083/-0); vllm/model_executor/models/registry.py (+4/-0); vllm/multimodal/audio.py (+8/-0); vllm/transformers_utils/config.py (+1/-0); vllm/transformers_utils/configs/__init__.py (+4/-0); vllm/transformers_utils/configs/funaudiochat.py (+124/-0)
LABELS: documentation, new-model, ready, multi-modality
BODY: supports full-functional FunAudioChat ST2T only inference mode in VLLM ⏎  ⏎ ## Purpose ⏎  ⏎ Add vLLM support for **FunAudioChatForConditionalGeneration** (multimodal **audio → text**). ⏎  ⏎ This PR: ⏎ - Implements the FunAudioChat model in `vllm/model_executor/models/funaudiochat.py` (multimodal embeddings + audio towers + LM integration). ⏎ - Registers the model/config in the vLLM model registry and Transformers config registry so it can be loaded via `--model` …[truncated]

### L3-c4e744dbd4  (L3, 2026-01-28, sha c4e744dbd41f, PR #32892)
TITLE: [Perf] Optimize `moe_permute` for CUTLASS FP8 (#32892)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/moe/moe_permute_unpermute_op.cu (+24/-9); csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.h (+2/-1); csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl (+21/-34)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Optimize `moe_permute` kernel using `aligned_expert_first_token_offset` ⏎  ⏎ - Reduce calculation ⏎ - Reduce shared memory usage ⏎  ⏎ ## Test ⏎  ⏎ ### Acc ⏎  ⏎ ```bash ⏎ tests/kernels/moe/test_moe_permute_unpermute.py ........................... [  6%] ⏎ ........................................................................... [ 23%] ⏎ ........................................................................... [ 40%] ⏎ ....................................... …[truncated]

### L3-22ad649501  (L3, 2026-01-28, sha 22ad64950119, PR #33106)
TITLE: [ROCm] Enabling forward_includes_kv_cache on ROCm MHA backends (#33106)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+34/-22); vllm/v1/attention/backends/rocm_attn.py (+55/-40); vllm/v1/attention/backends/triton_attn.py (+38/-25)
LABELS: rocm, ready, v1
BODY: Add the following backends support for #32335 ⏎ - RocmAiterUnifiedAttention ⏎ - RocmAttention ⏎ - TritonAttention ⏎  ⏎ rocm_attn is also covered in #32543 from what I can tell

### L3-77c4f45c6c  (L3, 2026-01-28, sha 77c4f45c6c94, PR #32477)
TITLE: [7/N][Attention][Docs] Add documentation for attention backends (#32477)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .pre-commit-config.yaml (+4/-0); docs/configuration/optimization.md (+7/-1); docs/design/attention_backends.md (+212/-0); tools/pre_commit/generate_attention_backend_docs.py (+1258/-0)
LABELS: documentation, ready
BODY: ## Purpose ⏎ Part 7 of #31919: Adds documentation for attention backends, including auto-generated tables of priority lists and feature support. This will be generated as part of pre-commit. ⏎  ⏎ cc @mgoin @LucasWilkinson @hmellor  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-59bcc5b6f2  (L3, 2026-01-28, sha 59bcc5b6f2e6, PR #30976)
TITLE: Use aiter triton fused_add_rmsnorm_pad for gpt-oss (#30976)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/test_fuse_act_padding.py (+131/-0); tests/compile/test_fusion.py (+2/-2); vllm/_aiter_ops.py (+46/-0); vllm/compilation/pass_manager.py (+6/-2); vllm/compilation/rocm_aiter_fusion.py (+104/-1); vllm/config/compilation.py (+16/-0); vllm/config/vllm.py (+16/-0); vllm/model_executor/layers/utils.py (+5/-5); vllm/model_executor/models/gpt_oss.py (+1/-1)
LABELS: rocm, ready, gpt-oss
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Adds fused padding op before router GEMM on ROCm, eliminating this unfused pad after the GEMM before the fused_moe: https://github.com/ROCm/vllm/blob/main/vllm/model_executor/layers/fused_moe/layer.py#1603 ⏎  ⏎ Before: ⏎ <img width="1370" height="30" alt="image" src="https://github.com/user-attachments/assets/11db1bf1-e8b0-49c5-b180-3d1c730f07ca" /> ⏎ After: ⏎ <img width="1462" height="34" alt="image" src="https://github.com/user-attachments/as …[truncated]

### L3-ab597c869a  (L3, 2026-01-28, sha ab597c869a78, PR #33269)
TITLE: [Bugfix] Add missing encoder only guard for do_kv_cache_update (#33269)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.triton.v1_backend
FILES: vllm/v1/attention/backends/triton_attn.py (+4/-0)
LABELS: bug, rocm, ready, v1
DEEP_STUDY: deep-study correctness case vllm:ab597c869a: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: A follow up for #33106 ⏎ do_kv_cache_update is called in a broader set of circumstances than the attention forward function. Need to prevent it from being ran in the encoder mode

### L3-e01ff5c070  (L3, 2026-01-29, sha e01ff5c070f4, PR #32669)
TITLE: Bugfix: Pass router logits dtype in nemotron shared experts (#32669)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/nemotron_h.py (+3/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎ A change introduced in [this PR](https://github.com/vllm-project/vllm/pull/31055) , requires passing `router_logits_dtype` to MoE layer. ⏎  ⏎ When running with `dp > 1` and flashinfer cutlass MoE kernel in nvfp4, the following error happens: ⏎ ``` ⏎ assert self.batched_router_logits.dtype == full_router_logits.dtype, ( ⏎ ERROR 01-19 05:53:49 [multiproc_executor.py:839]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ ERROR …[truncated]

### L3-0493d897c4  (L3, 2026-01-29, sha 0493d897c4b7, PR #32954)
TITLE: [NVIDIA] [feat] Integrate flashinfer Trtllmgen bf16 moe (#32954)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+3/-0); vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py (+103/-0); vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+35/-6); vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py (+74/-11); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+75/-0)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ - Integrate flashinfer trtllm-gen BF16 moe to supported models ⏎ - This is a rebased version of PR [28238](https://github.com/vllm-project/vllm/pull/28238) by @jiahanc and includes adaptation to the latest moe refactoring changes. I have further verified that the accuracy issues discussed in [28238](https://github.com/vllm-project/vllm/pull/28238) are solved. ⏎  ⏎ ## Test Plan ⏎  ⏎ ``VLLM_USE_FLASHINFER_MOE_FP16=1 VLLM_FLASHINFER_MOE_BACKEND=lat …[truncated]

### L3-23591e631e  (L3, 2026-01-29, sha 23591e631e19, PR #33326)
TITLE: [Bugfix][Kernel] Fix negative memory offset in GDN Triton kernel (#33326)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fla/ops/fused_recurrent.py (+18/-15)
LABELS: bug, ready
ISSUES: #31186 [Bug]: Qwen3-Next MTP Crash
BODY: ## Purpose ⏎  ⏎ Fix illegal memory access (`cudaErrorIllegalAddress`) when running Qwen3-Next models with MTP speculative decoding and CUDA Graphs enabled. ⏎  ⏎ Fixes #31186 ⏎  ⏎ cc @vadiklyutiy ⏎ ### Root Cause ⏎  ⏎ The Triton kernel `fused_recurrent_gated_delta_rule_fwd_kernel` in GDN attention did not check for `PAD_SLOT_ID = -1` values in `ssm_state_indices`. When CUDA Graph pads unused slots with `-1`, the kernel computes negative memory offsets, causing ille …[truncated]

### L3-53fc166402  (L3, 2026-01-29, sha 53fc16640281, PR #33262)
TITLE: [BugFix] Fix EPLB fail for MoeFP4 model with Marlin backend (#33262)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/utils.py (+10/-2)
LABELS: bug, ready
BODY: ## Purpose ⏎ PR fixes a bug when `nvidia/DeepSeek-R1-0528-FP4-v2` crashes when we enable EPLB with Flashinfer MOE FP4 disabled.  ⏎  ⏎ In marlin Moe_FP4 backend we don't use activations scales so we replace them in `replace_parameter` util with None. But  it doesn't set the parameter to None, it creates an empty parameter on cpu which triggers EPLB error: ⏎  ⏎ ``` ⏎ (EngineCore_DP1 pid=3244402)   File "/home/sagemoore/git/nm-vllm/vllm/distributed/eplb/eplb_st …[truncated]

### L3-8e2a469b3b  (L3, 2026-01-29, sha 8e2a469b3b2f, PR #32804)
TITLE: Add Triton fused MoE config for B200 (Nemotron Nano) (#32804)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/configs/E=128,N=1856,device_name=NVIDIA_B200.json (+139/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎  ⏎ When running Nemotron Nano on B200 the following warning appears: ⏎ ``` ⏎ Using default MoE config. Performance might be sub-optimal! ⏎ Config file not found at .../vllm/model_executor/layers/fused_moe/configs/E=128,N=1856,device_name=NVIDIA_B200.json ⏎ ``` ⏎  ⏎ I used the `benchmark_moe.py` to create a JSON file for this use-case: ⏎ ``` ⏎ export MODEL_PATH=/my_home/hf_models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 ⏎  ⏎ python benchmarks/kernels/bench …[truncated]

### L3-1a7894dbdf  (L3, 2026-01-30, sha 1a7894dbdfd0, PR #33332)
TITLE: [Misc] Replace Optional[X] with X | None syntax (#33332)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.mla.flashmla_sparse, L3.mla.rocm_aiter_sparse, L3.platform.cuda_selection, L3.tree_attention
FILES: vllm/beam_search.py (+2/-2); vllm/distributed/kv_transfer/kv_connector/factory.py (+2/-2); vllm/distributed/kv_transfer/kv_connector/v1/base.py (+6/-6); vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py (+2/-2); vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py (+2/-2); vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py (+2/-2); vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py (+5/-5); vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py (+2/-2); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py (+2/-2); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py (+3/-3); (+46 more)
LABELS: rocm, frontend, ready, v1, multi-modality, cpu, kv-connector, nvidia
BODY: ## Purpose ⏎  ⏎ Modernize type annotations by replacing `Optional[X]` with the more readable `X | None` union type syntax (PEP 604) that has been available since Python 3.10. ⏎  ⏎ Changes: ⏎ - Replace `Optional[X]` with `X | None` across 55 files ⏎ - Remove unused `Optional` imports from typing ⏎  ⏎ ## Test Plan ⏎  ⏎ ```bash ⏎ # Run pre-commit hooks ⏎ pre-commit run -a ⏎  ⏎ # Run mypy type checking ⏎ pre-commit run --hook-stage manual mypy-3.12 ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ ruff c …[truncated]

### L3-8f5d51203b  (L3, 2026-01-30, sha 8f5d51203b56, PR #32561)
TITLE: Disable Cascade Attention for Batch Invariance (#32561)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+12/-0); tests/v1/determinism/test_batch_invariance.py (+11/-3); tests/v1/determinism/utils.py (+6/-4); vllm/model_executor/layers/batch_invariant.py (+3/-1); vllm/model_executor/layers/linear.py (+14/-1); vllm/model_executor/layers/utils.py (+14/-0)
LABELS: ready, v1
ISSUES: #32481 [Bug]: Batch Invariance fails under more diverse workloads
BODY: ## Purpose ⏎ This PR disables cascade attention when batch invariance is enabled. Cascade attention is conditionally enabled on some input batches which occasionally causes numerical differences in the output logprobs. ⏎  ⏎ This PR additionally updates the batch invariance test suite by having more varied requests for `test_logprobs_bitwise_batch_invariance_bs1_vs_bsN` and slightly tweaking the padding logic for longer prompts. ⏎  ⏎ This closes #32481. ⏎  ⏎ ## …[truncated]

### L3-c3a9752b0c  (L3, 2026-01-30, sha c3a9752b0c11, PR #32437)
TITLE: [Hardware][SM100] Add TRTLLM Kernel for INT4 W4A16 Kernel. (#32437)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+2/-2); tests/kernels/moe/test_marlin_vs_trtllm_mxint4.py (+272/-0); vllm/envs.py (+8/-3); vllm/model_executor/layers/fused_moe/layer.py (+9/-5); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+170/-13); vllm/model_executor/layers/quantization/utils/flashinfer_mxint4_moe.py (+266/-0)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎ Adds INT4 W4A16 kernels from trtllm. Use `VLLM_USE_FLASHINFER_MOE_INT4=1` to use the kernel. ⏎   ⏎ ## Test Plan ⏎ Local test, gsm8k ⏎  ⏎ ## Test Result ⏎ ``` ⏎ vllm serve /tmp/moonshotai-Kimi-K2-Thinking   --tensor-parallel-size 4   --enable-auto-tool-choice   --tool-call-parser kimi_k2   --reasoning-parser kimi_k2    --trust-remote-code ⏎ root@gb-nvl-081-compute06:/workspace/pm-vllm# python3 tests/evals/gsm8k/gsm8k_eval.py  ⏎ Downloading from https://r …[truncated]

### L3-67ebaff528  (L3, 2026-01-30, sha 67ebaff528be, PR #33201)
TITLE: Refactor NVFP4 Linear utils for ModelOpt and CT (#33201)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/modular_kernel_tools/mk_objects.py (+1/-1); tests/quantization/test_compressed_tensors.py (+1/-1); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py (+1/-12); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_mxfp4.py (+2/-2); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py (+6/-21); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py (+29/-153); vllm/model_executor/layers/quantization/modelopt.py (+34/-162); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+3/-1); vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py (+9/-7); vllm/model_executor/layers/quantization/utils/nvfp4_moe_support.py (+1/-1); (+2 more)
LABELS: ready, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ There have been several linear kernel backends for nvfp4 integrated in vLLM, some added to ModelOpt and some added to compressed-tensors. This PR consolidates all the kernel backends and preparation logic into shared utils that both quant methods can use. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ Tested all backends for both checkpoint types, noting that `flashinfer-cudnn` seems broken on main already ⏎  ⏎ ### ModelOpt ⏎  ⏎ ``` ⏎ VLLM_NVFP4_GEMM_BACKEND=f …[truncated]

### L3-aaa901ad55  (L3, 2026-01-30, sha aaa901ad55ee, PR #33284)
TITLE: [Attention] Move MLA `forward` from backend to layer (#33284)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashmla_sparse, L3.mla.rocm_aiter, L3.mla.rocm_aiter_sparse, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/attention.py (+2/-2); vllm/model_executor/layers/attention/mla_attention.py (+443/-440); vllm/v1/attention/backend.py (+82/-13); vllm/v1/attention/backends/mla/cutlass_mla.py (+1/-1); vllm/v1/attention/backends/mla/flashattn_mla.py (+1/-1); vllm/v1/attention/backends/mla/flashinfer_mla.py (+1/-1); vllm/v1/attention/backends/mla/flashmla.py (+1/-1); vllm/v1/attention/backends/mla/flashmla_sparse.py (+24/-70); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+1/-1); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+21/-77); (+3 more)
LABELS: rocm, ready, v1, nvidia
BODY: ## Purpose ⏎ Refactor MLA by moving `forward` from the backend to the layer to facilitate prefill-decode splitting. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-f0bca83ee4  (L3, 2026-01-30, sha f0bca83ee4b6, PR #33174)
TITLE: Add support for Mistral Large 3 inference with Flashinfer MoE (#33174)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); docker/Dockerfile.nightly_torch (+2/-2); docker/versions.json (+1/-1); requirements/cuda.txt (+1/-1); benchmarks/kernels/benchmark_moe.py (+38/-11); tests/kernels/moe/test_flashinfer.py (+2/-0); vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json (+147/-0); vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_B200.json (+147/-0); vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json (+147/-0); vllm/model_executor/layers/fused_moe/configs/E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json (+147/-0); (+6 more)
LABELS: performance, ready, ci/build, deepseek, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ Allow inference of Mistral Large 3 on Blackwell with Flashinfer TRTLLM (`latency`) backend for better performance. ⏎  ⏎ This PR updates Flashinfer to 0.6.2 that includes fixed kernels for blockwise quantized FP8 MoE and makes small changes to the vLLM code to allow calling this code. It also allows calling Flashinfer for per-tensor quantized models (`fp8` quantization). ⏎  ⏎ Also, optimized Triton configurations are added for Mistral Large 3  …[truncated]

### L3-5b55c0bea7  (L3, 2026-01-31, sha 5b55c0bea741, PR #33427)
TITLE: [Attention] Clarify comment explaining attn_logits +1 dimension (#33427)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/triton_mla.py (+2/-2)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ The original comment admitted uncertainty about why the +1 exists. Replace it with an accurate explanation: the extra slot stores the LogSumExp (LSE) value that are computed by the stage1 kernel and later used by the the stage2 kernel to merge the partial attention outputs across KV splits. ⏎  ⏎ CC: @LucasWilkinson

### L3-15f40b20aa  (L3, 2026-01-31, sha 15f40b20aadf, PR #33441)
TITLE: [fix][torch.compile] Fix cold-start compilation time increase by adding kv cache update to splitting ops (#33441)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/test_cold_start.py (+48/-0); tests/compile/test_graph_partition.py (+62/-0); vllm/compilation/backends.py (+8/-1); vllm/config/compilation.py (+9/-0)
LABELS: ready, ready-run-all-tests
BODY: ## Purpose ⏎ As described in #33267, a cold start time regression was introduced by #25954. This fixes that (temporarily) by adding `unified_kv_cache_update` to `splitting_ops`. The PR also includes a fix to make sure `unified_kv_cache_update` and `unified_attention` end up in the same splitting graph to reduce piecewise backend overhead. ⏎  ⏎ ## Startup Results ⏎ Numbers for compile/total startup time (clearly our compilation time reporting is imperfect …[truncated]

### L3-1618e25492  (L3, 2026-01-31, sha 1618e2549281, PR #33122)
TITLE: [CPU][Feat] Enable KleidiAI accelerated int4 dynamic quant with BF16 activations on Arm CPUs (#33122)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/kernels/mixed_precision/dynamic_4bit.py (+18/-1)
LABELS: ready
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ [CPU][Feat] Enable KleidiAI accelerated int4 dynamic quant with BF16 activations ⏎ BF16 act, int4 wei KleidiAI kernels were enabled in PyTorch by https://github.com/pytorch/pytorch/pull/158250 and are available in torch==2.10.0 ⏎  ⏎ Currently, our int4 dynamic quant path enabled by https://github.com/vllm-project/vllm/pull/17112 only supports FP32 activations - i.e. all ops but dense layers run in FP32 and our KV cache will be in FP32.  ⏎ Thi …[truncated]

### L3-0a3c71e7e5  (L3, 2026-01-31, sha 0a3c71e7e5f0, PR #33360)
TITLE: [BugFix] Fix whisper FA2 + full cudagraphs (#33360)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+0/-12); vllm/v1/worker/gpu_model_runner.py (+12/-0)
LABELS: bug, ready, v1, nvidia
ISSUES: #33091 [Bug]: Whisper accuracy issue with FA2+CG
DEEP_STUDY: deep-study correctness case vllm:0a3c71e7e5: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: Fix: https://github.com/vllm-project/vllm/issues/33091 ⏎  ⏎ `CrossAttentionBuilder.build()` overrides `max_seq_len` with encoder_seq_lens (`new_metadata.max_seq_len = max(encoder_seq_lens_cpu)`) this leads to a CG capture with `max_seq_len == 0` and an incorrect graph

### L3-079781177a  (L3, 2026-01-31, sha 079781177ae4, PR #33417)
TITLE: fix: Add SM120 (RTX Blackwell) support for FlashInfer CUTLASS NVFP4 MoE kernels (#33417)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/cutlass_moe.py (+6/-1); vllm/model_executor/layers/fused_moe/flashinfer_cutedsl_moe.py (+2/-1); vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py (+14/-13); vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py (+0/-1); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+0/-27); vllm/model_executor/layers/quantization/utils/nvfp4_moe_support.py (+0/-67)
LABELS: ready, ci/build, nvidia, quantization
ISSUES: #33416 [Bug] NVFP4 MoE kernels fail on RTX Blackwell (SM12.0) - device capability family check missing SM120
BODY: ## Summary ⏎  ⏎ This PR adds SM120 (RTX Blackwell) device capability family support to the NVFP4 MoE kernel backend selection code. The NVFP4 quantization kernels check for specific GPU architecture families, but currently only recognize SM9.0 (Hopper) and SM10.x (B100/B200 data center Blackwell), missing SM12.0 (RTX Blackwell workstation GPUs). ⏎  ⏎ ## Problem ⏎  ⏎ On RTX Blackwell GPUs (e.g., RTX PRO 6000 Blackwell Workstation Edition with compute capabili …[truncated]
