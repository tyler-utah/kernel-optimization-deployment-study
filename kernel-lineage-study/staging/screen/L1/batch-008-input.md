### L1-895e56097c  (L1, 2026-03-16, sha 895e56097cc2, PR #19382)
TITLE: Add NPU basic function testcases (#19382)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/nightly-test-npu.yml (+37/-17); python/sglang/test/ascend/disaggregation_utils.py (+153/-0); python/sglang/test/ascend/test_ascend_utils.py (+495/-54); test/registered/ascend/basic_function/HiCache/test_npu_hierarchical_cache.py (+127/-0); test/registered/ascend/basic_function/HiCache/test_npu_hierarchical_cache_mla.py (+97/-0); test/registered/ascend/basic_function/HiCache/test_npu_hierarchical_cache_mutually_exclusive.py (+68/-0); test/registered/ascend/basic_function/HiCache/test_npu_hierarchical_cache_ttft_mha.py (+89/-0); test/registered/ascend/basic_function/HiCache/test_npu_radix_cache.py (+132/-0); test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_deepseek_v3_2_w8a8.py (+108/-0); test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_qwen3_480b.py (+128/-0); (+77 more)
LABELS: Multi-modal, deepseek, npu, run-ci
BODY: ## Motivation ⏎  ⏎ This PR aims to comprehensively improve the test coverage of the Ascend (NPU) backend for the SGLang framework. A large number of targeted test cases have been added, and some existing test files have been optimized and adjusted. In total,48 files are involved (35 new files, 13 optimized files). The detailed changes are as follows: ⏎  ⏎ 1、Test Infrastructure Optimization ⏎ Adjusted the nightly-test-npu.yml workflow configuration, ad …[truncated]

### L1-15097c5c3b  (L1, 2026-03-16, sha 15097c5c3b52, PR #20440)
TITLE: Release sglang kernel 0.4.0 (#20440)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+4/-4); python/pyproject.toml (+1/-1); .github/workflows/release-whl-kernel.yml (+1/-1); 3rdparty/amd/wheel/README.md (+5/-5); 3rdparty/amd/wheel/sglang/pyproject.toml (+2/-2); docs/developer_guide/contribution_guide.md (+5/-5); docs/get_started/install.md (+3/-3); docs/platforms/ascend_contribution_guide.md (+4/-4); python/sglang/check_env.py (+1/-1); python/sglang/jit_kernel/benchmark/bench_awq_marlin_moe_repack.py (+7/-8); (+26 more)
LABELS: documentation, high priority, amd, dependencies, sgl-kernel, npu, run-ci, mthreads, jit-kernel
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎ 3. Trigge …[truncated]

### L1-385a35bd11  (L1, 2026-03-17, sha 385a35bd118d, PR #20647)
TITLE: [AMD][MORI] Fix MTP crash with FP4/FP8 dispatch and add NEXTN dispatch env vars. (#20647)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+9/-7); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-1); python/sglang/srt/layers/moe/moe_runner/base.py (+1/-0); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+30/-17); docs/references/environment_variables.md (+2/-0); python/sglang/srt/models/deepseek_v2.py (+1/-0)
LABELS: documentation, deepseek, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 20797 (confirmed_revert, reason=ci_or_test_failure)
BODY: This PR is new version of old PR #20453 with a fix to address DeepEP CI failure by only passing `is_nextn` to `MaybeTboDeepEPDispatcher` when `a2a_backend.is_mori`. ⏎  ⏎ ## Motivation ⏎  ⏎ When using a quark MXFP4 model (e.g. `DeepSeek-R1-0528-MXFP4`) with EAGLE speculative decoding (MTP layer) and MORI FP4/FP8 dispatch enabled (`SGLANG_MORI_FP4_DISP=True` or `SGLANG_MORI_FP8_DISP=True`), the MTP layer crashes during CUDA graph capture: ⏎  ⏎ ``` ⏎ Runti …[truncated]

### L1-cb1e63aba4  (L1, 2026-03-17, sha cb1e63aba45e, PR #20303)
TITLE: bump fa4 to official released fa4 pkg (#20303)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+3/-4); python/pyproject.toml (+3/-4); python/sglang/jit_kernel/flash_attention_v4.py (+2/-2); scripts/ci/cuda/ci_install_dependency.sh (+3/-15)
LABELS: dependencies, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ - fa4 integration with official fa4 pkg. ⏎  ⏎ ## Modifications ⏎  ⏎ - bump fa4 pkg ⏎ - bump nvidia-cutlass-dsl pkg ⏎ - bump quack-kernels pkg ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [ …[truncated]

### L1-c5d2528bff  (L1, 2026-03-17, sha c5d2528bff98, PR #20797)
TITLE: Revert "[AMD][MORI] Fix MTP crash with FP4/FP8 dispatch and add NEXTN dispatch env vars." (#20797)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+7/-9); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-6); python/sglang/srt/layers/moe/moe_runner/base.py (+0/-1); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+17/-30); docs/references/environment_variables.md (+0/-2); python/sglang/srt/models/deepseek_v2.py (+0/-1)
LABELS: documentation, deepseek
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 20647 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#20647 ⏎  ⏎ This breaks https://github.com/sgl-project/sglang/actions/runs/23083366420/job/67452685234?pr=20469

### L1-b5f3eaecbc  (L1, 2026-03-17, sha b5f3eaecbc05, PR #19913)
TITLE: [NPU] Support dequant_swiglu_quant & moe_init_routing_v2 & npu_moe_token_unpermute for W8A8 MoE decode (#19913)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+116/-14)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ Support moe_init_routing_v2 & dequant_swiglu_quant & npu_moe_token_unpermute for W8A8 MoE decode. ⏎  ⏎ ## Modifications ⏎  ⏎ - Replace npu_moe_init_routing with npu_moe_init_routing_v2 in W8A8 MoE decode stage. ⏎ - Replace npu_swiglu and npu_dynamic_quant with npu_dequant_swiglu_quant in W8A8 MoE decode stage. ⏎ - Replace npu_moe_finalize_routing with npu_moe_token_unpermute in W8A8 MoE decode stage. ⏎  ⏎ ## Accuracy Tests ⏎ gsm8k, 200 qu …[truncated]

### L1-97d5386a21  (L1, 2026-03-17, sha 97d5386a211c, PR #19889)
TITLE: Use TRTLLM allreduce fusion for Qwen 3.5 (#19889)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/layernorm.py (+63/-48); python/sglang/srt/models/qwen2_moe.py (+11/-2); python/sglang/srt/models/qwen3_5.py (+12/-2); python/sglang/srt/server_args.py (+2/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Before: ⏎ 21.5 us ⏎  ⏎ <img width="1652" height="1424" alt="image" src="https://github.com/user-attachments/assets/d9b92a62-dfc1-4593-b86a-7fadbd8b8701" /> ⏎  ⏎ After ⏎ 10.4 us ⏎  ⏎ <img width="1994" height="1050" alt="image" src="https://github.com/user-attachments/assets/76f3c5c5-22e9-479f-b0f1-e80c76cc339e" /> ⏎  ⏎ This PR is mainly authored by @vincentzed

### L1-f0d7a3f427  (L1, 2026-03-18, sha f0d7a3f42732, PR #19888)
TITLE: [AMD][TBO] Fix mori ep dual stream accuracy (#19888)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+57/-21); python/sglang/srt/batch_overlap/two_batch_overlap.py (+2/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ This patch is to fix the accuracy issue for mori ep dual stream through adding the instance id to pair the mori op to according stream. ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎  ⏎ [details omitted] ⏎  ⏎  ⏎ GSM8K accuracy check: ⏎ ```bash ⏎ python /billhe/sglang-sa-tbo/benchmark/gsm8k/bench_sglang.py --num-questions 200 --port 30100 ⏎ 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████ …[truncated]

### L1-6b8a6545b2  (L1, 2026-03-18, sha 6b8a6545b231, PR #20708)
TITLE: Add Mistral Small 4 (Pixtral) support (#20708)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+1/-1); benchmark/mmmu/bench_sglang.py (+49/-10); benchmark/mmmu/eval_utils.py (+8/-0); python/sglang/srt/configs/deepseek_ocr.py (+2/-2); python/sglang/srt/configs/deepseekvl2.py (+3/-3); python/sglang/srt/configs/janus_pro.py (+12/-12); python/sglang/srt/configs/jet_nemotron.py (+12/-12); python/sglang/srt/entrypoints/openai/protocol.py (+1/-1); python/sglang/srt/entrypoints/openai/serving_chat.py (+32/-13); python/sglang/srt/function_call/mistral_detector.py (+17/-9); (+8 more)
LABELS: deepseek, blackwell, run-ci
BODY: ## Summary ⏎ - Add Mistral Small 4 (119B) model support, reusing the MistralLarge3/DeepSeekV3 backend with Pixtral vision encoder ⏎ - Handle Mistral-native config format (`params.json`) for Mistral Small 4 and LeanStral model variants ⏎ - Add Mistral reasoning parser (`[THINK]`/`[/THINK]` format) with `reasoning_effort="high"` gating ⏎ - Fix Pixtral vision processor: proper `spatial_merge_size` handling, `rope_parameters` compatibility, and fallback  …[truncated]

### L1-532470bcca  (L1, 2026-03-18, sha 532470bcca52, PR #20245)
TITLE: [NPU] add new fusion operator DispatchFFNCombine (#20245)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+53/-14); python/sglang/srt/layers/moe/token_dispatcher/fuseep.py (+1/-0); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/hardware_backend/npu/utils.py (+5/-0); python/sglang/srt/server_args.py (+9/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ The fusion operator DispatchFFNCombine is added ⏎  ⏎ ## Modifications ⏎  ⏎ 1、Add the environment variable SGLANG_NPU_FUSED_MOE_MODE; ⏎ 2、When SGLANG_NPU_FUSED_MOE_MODE is set to 1, dispatch_gmm_combine_decode is executed；When SGLANG_NPU_FUSED_MOE_MODE is set to 2, dispatch_ffn_combine is executed. ⏎ 3、ascend_fuseep support hybrid deployment when SGLANG_NPU_FUSED_MOE_MODE=2 ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ qwen3-235B: ⏎ <img width="924" height="1 …[truncated]

### L1-7f6f1a3ab1  (L1, 2026-03-18, sha 7f6f1a3ab1c4, PR #19711)
TITLE: [LoRA][II] Add fused MOE LoRA Triton kernel and tests (#19711)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/lora/triton_ops/fused_moe_lora_kernel.py (+690/-0); python/sglang/srt/lora/triton_ops/__init__.py (+2/-0); test/registered/lora/test_fused_moe_lora_kernel.py (+380/-0)
LABELS: lora, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Split this PR https://github.com/sgl-project/sglang/pull/14105 into 3 parts - Part II ⏎  ⏎ Add Triton-based fused MoE LoRA kernel for combined expert routing and LoRA computation: ⏎ - fused_moe_lora_kernel.py: Triton kernels for fused_moe_lora_shrink and fused_moe_lora_expand ⏎ - test_fused_moe_lora_kernel.py: Unit tests validating kernel correctness against PyTorch reference ⏎ - Update triton_ops/__init__.py to export fused_moe_lora ⏎  ⏎  ⏎  ⏎  ⏎ ## Motiv …[truncated]

### L1-4e8829e4cd  (L1, 2026-03-18, sha 4e8829e4cd98, PR #20302)
TITLE: Replace topk_ids with curr_topk_ids in fused_moe.py (#20302)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-2)
LABELS: run-ci
BODY: Fixed the topk id bug when the num_tokens>chunk_size ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/b …[truncated]

### L1-8d4fcf2f7b  (L1, 2026-03-18, sha 8d4fcf2f7bff, PR #12555)
TITLE: [CPU] Fix MoE layer support for DeepSeek-OCR models (#12555)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek.py (+31/-9); python/sglang/srt/models/deepseek_ocr.py (+34/-1)
LABELS: deepseek, intel, cpu, run-ci
BODY: This PR adds the fix for DeepSeek-OCR running on CPUs. ⏎  ⏎ Without this PR, default `deepseek.py` will use **triton** [fusemoe module ](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek.py#L179-L185), which will throw errors on CPUs. ⏎  ⏎ Instead, this PR adds frontends of CPU amx weight loading/packing and calls into CPU optimized kernel `torch.ops.sgl_kernel.fused_experts_cpu` ⏎  ⏎  ⏎ **Funtionality test with this PR:* …[truncated]

### L1-574572b21b  (L1, 2026-03-19, sha 574572b21bf7, PR #20492)
TITLE: [BugFix] bug fix for DeepSeek eagle3 in Attn-DP mode (#20492)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+2/-2)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ The current DeepSeek Eagle3 implementation has an issue in DP scenarios when using tensor_model_parallel_all_gather. The gather should be performed across the attention_tp group instead of the tensor parallel group. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ tp 16 ⏎ ```sh ⏎ python -m sglang.launch_server --skip-server-warmup  \ ⏎     --model-path /xxx/Kimi-K2.5-w4a8 --quantization modelslim --dtype bfloat16 \ ⏎     --host  …[truncated]

### L1-9419453713  (L1, 2026-03-20, sha 941945371314, PR #18684)
TITLE: [AMD] Add MoE weights and scales padding (#18684)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+14/-2); python/sglang/srt/layers/moe/utils.py (+30/-0); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py (+18/-4); python/sglang/srt/layers/quantization/fp8.py (+37/-20); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py (+23/-5); python/sglang/srt/model_executor/model_runner.py (+5/-1)
LABELS: run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 21067 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Motivation ⏎  ⏎ Right now, Aiter MoE requires weights and scales to align with a fixed number. Since some models have intermediate sizes that don't fit this rule, we need to add extra padding to the weights so they can be processed by the Fused MoE. ⏎  ⏎ ## Modifications ⏎  ⏎ Add padding for the weights. Below listed are the models and configurations that has been verified with: ⏎ 1. Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 : TP=8 ⏎ 2. amd/Qwen3-235B-A2 …[truncated]

### L1-048d90e165  (L1, 2026-03-20, sha 048d90e1651a, PR #21067)
TITLE: Revert "[AMD] Add MoE weights and scales padding" (#21067)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-14); python/sglang/srt/layers/moe/utils.py (+0/-30); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py (+4/-18); python/sglang/srt/layers/quantization/fp8.py (+20/-37); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py (+5/-23); python/sglang/srt/model_executor/model_runner.py (+1/-5)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 18684 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#18684 ⏎  ⏎ Caused CI failure: https://github.com/sgl-project/sglang/actions/runs/23367673750/job/67984876644

### L1-c076968c52  (L1, 2026-03-21, sha c076968c52cc, PR #21075)
TITLE: [CI] Remove obsolete AOT-only jit-kernel benchmarks after sgl-kernel 4.0 (#21075)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/benchmark/bench_awq_marlin_moe_repack.py (+0/-125); python/sglang/jit_kernel/benchmark/bench_awq_marlin_repack.py (+0/-110); python/sglang/jit_kernel/benchmark/bench_gptq_marlin.py (+0/-129); python/sglang/jit_kernel/benchmark/bench_gptq_marlin_repack.py (+0/-97); python/sglang/jit_kernel/benchmark/bench_moe_wna16_marlin.py (+0/-240)
LABELS: run-ci, jit-kernel
BODY: ## Summary ⏎  ⏎ This PR cleans up `python/sglang/jit_kernel/benchmark` for the current `sgl-kernel` setup. ⏎  ⏎ Several benchmark scripts were still comparing against legacy `sgl-kernel` AOT kernels that are no longer exported after the `sgl-kernel 4.0` upgrade. ⏎  ⏎ This PR removes the benchmark scripts that only benchmark JIT against the removed AOT kernels and no longer have a meaningful reference baseline. ⏎  ⏎ ## Changes ⏎  ⏎ Removed these benchmark s …[truncated]

### L1-a0862f00c2  (L1, 2026-03-21, sha a0862f00c246, PR #17121)
TITLE: dbrx instruct npu support (#17121)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/dbrx.py (+8/-2); test/registered/ascend/llm_models/test_ascend_dbrx_instruct.py (+26/-0)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ dbrx instruct npu support ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Reason: dbrx instruct do not support npu, sgl_moe_align_block_size only exists in sgl-kernel, not in sgl-kernel-npu ⏎  ⏎ error: NameError: name 'sgl_moe_align_block_size' is not defined ⏎  ⏎ ## Accuracy Tests ⏎ NPU ⏎ <img width="607" height="85" alt="image" src="https://github.com/user-attachments/assets/94c10d42-27b9-452a-b4cc-da4977aa89d6" /> ⏎  ⏎ GPU ⏎ <img width="298" height="67" al …[truncated]

### L1-ce0541404f  (L1, 2026-03-22, sha ce0541404fec, PR #20214)
TITLE: [FlashInfer v0.6.6][RL] Support fp8-last-n-bf16 RL for `flashinfer_trtllm_routed` moe backend (#20214)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+10/-0); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+88/-32); python/sglang/srt/layers/quantization/unquant.py (+47/-1); python/sglang/srt/server_args.py (+2/-1); test/registered/backends/test_flashinfer_trtllm_gen_moe_backend.py (+7/-1); test/registered/rl/test_update_weights_from_disk_mxfp8.py (+165/-0)
LABELS: documentation, quant, run-ci
BODY: ## Motivation ⏎ @humansand ⏎  ⏎ This PR is the last missing piece of Miles Blackwell mxfp8 RL training: https://github.com/radixark/miles/issues/615 , https://github.com/radixark/miles/pull/614 . ⏎  ⏎  ⏎  ⏎ Dependencies: ⏎ - FlashInfer `trtllm_bf16_routed_moe`: https://github.com/flashinfer-ai/flashinfer/pull/2594 ⏎ - Earlier FlashInfer routed moe integration: https://github.com/sgl-project/sglang/pull/19537 ⏎ - Mixed mxfp8 + bf16 serving: https://github.c …[truncated]

### L1-766d225fcc  (L1, 2026-03-22, sha 766d225fccf0, PR #20910)
TITLE: Add SGLang CUDA crash API logging inspired by FlashInfer (#20910)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm, L1.runner.marlin, L1.ep.other_dispatchers
FILES: python/sglang/jit_kernel/moe_wna16_marlin.py (+2/-0); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+14/-3); python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+3/-0); .claude/skills/debug-cuda-crash/SKILL.md (+657/-0); docs/diffusion/environment_variables.md (+12/-0); docs/references/environment_variables.md (+5/-0); python/sglang/jit_kernel/awq_marlin_repack.py (+3/-0); python/sglang/jit_kernel/debug_utils.py (+45/-0); python/sglang/jit_kernel/diffusion/triton/norm.py (+55/-2); python/sglang/jit_kernel/diffusion/triton/rmsnorm_onepass.py (+5/-1); (+36 more)
LABELS: documentation, quant, deepseek, hicache, sgl-kernel, blackwell, run-ci, diffusion, jit-kernel
BODY: ## Motivation ⏎  ⏎ This PR adds SGLang-native API-level CUDA crash logging for LLM and diffusion kernel call boundaries. ⏎  ⏎ The implementation is inspired by FlashInfer's API logging utility: ⏎ https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/api_logging.py ⏎  ⏎ This version keeps the scope focused on crash debugging and level-10 dump capture. Replay-related code was intentionally not included so the implementation stays smaller and al …[truncated]

### L1-814202704b  (L1, 2026-03-23, sha 814202704bf2, PR #21187)
TITLE: ci: unify PR test suite naming (#21187)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .claude/skills/write-sglang-test/SKILL.md (+12/-12); .github/actions/wait-for-jobs/action.yml (+1/-1); .github/workflows/pr-test-amd-rocm720.yml (+30/-30); .github/workflows/pr-test-amd.yml (+38/-38); .github/workflows/pr-test.yml (+28/-28); python/sglang/jit_kernel/tests/test_moe_lora_align_block_size.py (+1/-1); scripts/ci/utils/slash_command_handler.py (+19/-19); test/README.md (+19/-19); test/registered/amd/disaggregation/test_disaggregation_basic.py (+2/-2); test/registered/attention/test_chunk_gated_delta_rule.py (+1/-1); (+273 more)
LABELS: documentation, quant, amd, lora, Multi-modal, deepseek, speculative-decoding, hicache, npu, run-ci
BODY: ## Summary ⏎  ⏎ This change renames CI **job IDs** and registered-test `suite=` strings to a single, predictable pattern so stage / GPU count / size are ordered consistently. ⏎  ⏎ ### Naming pattern ⏎  ⏎ - **Before:** mixed order, e.g. `stage-b-test-large-1-gpu`, `stage-a-test-small-1-gpu`, `stage-a-cpu-only` ⏎ - **After:** `stage-{a|b|c}-test-{N}-gpu-{small|large|h200|b200}` (with existing suffixes such as `-amd`, `-mi35x`, etc. preserved), CPU suite a …[truncated]

### L1-4641e5a3d2  (L1, 2026-03-23, sha 4641e5a3d2bb, PR #17695)
TITLE: [NPU] enhance accuracy for model minimaxm2 from 16.5% to 95.5% (#17695)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+1/-1); python/sglang/test/ascend/test_ascend_utils.py (+1/-0); test/registered/ascend/llm_models/test_ascend_minimax_m2.py (+43/-0)
LABELS: npu, run-ci
BODY: ## Motivation ⏎ Previously, the accuracy for npu for model minimaxm2 is no more than 16.5%. ⏎ <img width="2232" height="136" alt="image" src="https://github.com/user-attachments/assets/aea8e378-ca93-472d-a90e-35138ed116ee" /> ⏎  ⏎ ## Modifications ⏎ I pinpointed the problem in fused_topk_npu(), operator--npu_moe_gating_topk_softmax. ⏎  ⏎ ## Accuracy Tests ⏎ <img width="2648" height="122" alt="image" src="https://github.com/user-attachments/assets/fbee341 …[truncated]

### L1-a32e0d57e7  (L1, 2026-03-24, sha a32e0d57e7f0, PR #14105)
TITLE: [LoRA][III] Add LoRA support for MoE layers and enable TP (#14105)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.runner.framework
FILES: python/sglang/srt/layers/moe/moe_runner/runner.py (+38/-18); python/sglang/srt/lora/lora_moe_runners.py (+585/-0); python/sglang/srt/lora/triton_ops/fused_moe_lora_kernel.py (+12/-1); python/sglang/srt/lora/layers.py (+190/-0); python/sglang/srt/lora/lora.py (+0/-1); python/sglang/srt/lora/lora_manager.py (+49/-1); python/sglang/srt/lora/mem_pool.py (+205/-40); python/sglang/srt/lora/utils.py (+6/-1); python/sglang/test/lora_utils.py (+27/-0); test/registered/lora/test_lora_moe_tp_logprob_diff.py (+172/-0); (+1 more)
LABELS: documentation, quant, amd, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel, blackwell
BODY: ## Motivation ⏎  ⏎ This PR adds support for LoRA serving on the expert layers for Mixture-of-Expert models. This is the initial development PR, which was split into two other smaller PRs, for maintenance reasons: https://github.com/sgl-project/sglang/pull/19710 and https://github.com/sgl-project/sglang/pull/19711 ⏎  ⏎ ## Modifications ⏎  ⏎ 1) Adds `FusedMoEWithLoRA`, which is a wrapper around `FusedMoE` and enables LoRA on `FusedMoE`. `FusedMoEWithLoRA …[truncated]

### L1-1046dbe038  (L1, 2026-03-24, sha 1046dbe03865, PR #21343)
TITLE: [Fix] Fix trtllm fp4 moe kernel not found error (#21343)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-10)
BODY: ## Motivation ⏎ https://github.com/sgl-project/sglang/actions/runs/23512497713/job/68437804154?pr=21330#step:6:3087 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS]( …[truncated]

### L1-61a902ce88  (L1, 2026-03-24, sha 61a902ce88ea, PR #21040)
TITLE: [AMD][MoRI] Auto-select dispatch quantization type from MoE weight dtype. (#21040)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+71/-37); docs/references/environment_variables.md (+1/-2); python/sglang/srt/layers/quantization/fp8.py (+6/-7); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py (+4/-0); test/registered/amd/test_moriep_small.py (+8/-8)
LABELS: documentation, high priority, amd, run-ci
BODY: ## Motivation ⏎  ⏎ Previously, MoRI EP dispatch quantization type (BF16/FP8/FP4) was controlled entirely by environment variables (`SGLANG_MORI_FP8_DISP` / `SGLANG_MORI_FP4_DISP`), requiring users to manually set them to match the model's weight type. This was error-prone and inconvenient. ⏎  ⏎ This PR makes MoRI EP automatically detect the dispatch quantization type from the loaded MoE weight dtype, so that the correct dispatch path is selected with …[truncated]

### L1-dfc15b78b0  (L1, 2026-03-25, sha dfc15b78b09f, PR #21325)
TITLE: [misc] clean up kernel API (#21325)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm, L1.runner.marlin
FILES: python/sglang/jit_kernel/moe_wna16_marlin.py (+2/-2); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+8/-15); python/sglang/jit_kernel/all_reduce.py (+2/-0); python/sglang/jit_kernel/awq_marlin_repack.py (+3/-3); python/sglang/jit_kernel/debug_utils.py (+0/-45); python/sglang/jit_kernel/diffusion/triton/rmsnorm_onepass.py (+2/-3); python/sglang/jit_kernel/flash_attention_v4.py (+3/-3); python/sglang/jit_kernel/fused_store_index_cache.py (+2/-2); python/sglang/jit_kernel/gptq_marlin.py (+2/-2); python/sglang/jit_kernel/gptq_marlin_repack.py (+2/-2); (+17 more)
LABELS: quant, hicache, blackwell, run-ci, diffusion, jit-kernel
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Remove `maybe_wrap_jit_kernel_debug`. Always use `debug_kernel_api` in JIT kernel. ⏎ 2. Make `debug_kernel_api` compatible with `register_custom_op` and improve readability (see the example below) ⏎ 3. Make the import of `kernel_api_logging` module safe. When the environment var is not valid, it will not raise an Error, but instead give a warning and fallback to default value. ⏎  ⏎ example: ⏎  ⏎ ```python ⏎ im …[truncated]

### L1-dbe871efdd  (L1, 2026-03-25, sha dbe871efdd97, PR #21430)
TITLE: Rollback flashmla to older version [1/2] (#21430)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/cmake/flashmla.cmake (+11/-47); sgl-kernel/python/sgl_kernel/flash_mla.py (+0/-9)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study revert record: explicit_rollback of PR(s)  reason=correctness_or_accuracy || deep-study: this PR was reverted by PR 21922 (reland, reason=unstated)
BODY: ## Motivation ⏎  ⏎ Temporarily avoid #21291 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and …[truncated]

### L1-f142608408  (L1, 2026-03-25, sha f142608408ed, PR #21296)
TITLE: [MUSA] apply_vocab_mask support musa device (#21296)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/constrained/xgrammar_backend.py (+1/-5)
LABELS: run-ci
BODY: ## Motivation ⏎ The xgrammar backend currently supports several hardware accelerators, including CUDA, NPU, and XPU, for constrained decoding. This pull request adds support for Moore Threads (MUSA) devices to enable efficient constrained decoding features using the xgrammar backend on MUSA hardware. ⏎  ⏎ ## Modifications ⏎  ⏎ - Updated the device check logic in xgrammar_backend.py to include musa as a supported device type. ⏎ - This change ensures tha …[truncated]

### L1-fd535942ac  (L1, 2026-03-26, sha fd535942acfe, PR #21421)
TITLE: [AMD]Integrate aiter's fused_topk for softmax scoring in topk function (#21421)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+19/-6)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Enable AIter-backed paths for ROCm/HIP to fuse softmax+topk: **MoE TopK**. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ When aiter is enabled, by default use `aiter.fused_topk` ⏎ ## Accuracy Tests ⏎  ⏎ **Before** ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9719|±  |0.0045| ⏎ |     |       |st …[truncated]

### L1-6d48719e31  (L1, 2026-03-27, sha 6d48719e31aa, PR #21439)
TITLE: [1/n] lora support - Auto detect lora target modules (#21439)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/lora/lora_manager.py (+16/-10); python/sglang/srt/lora/utils.py (+48/-5); test/registered/lora/test_lora_qwen3_8b_logprob_diff.py (+202/-0)
LABELS: lora, run-ci
BODY: ## Motivation ⏎  ⏎ support auto detect lora target modules ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Auto-detect LoRA target modules ⏎ When adapter uses target_modules="all-linear"/"all", instead of raising an error requiring --lora-target-modules, now auto-resolves by scanning the base model for LoRA-compatible linear modules (qkv_proj, o_proj, gate_up_proj, down_proj, lm_head). ⏎ Adds auto_detect_lora_target_modules() in lora/utils.py; updates lora_manager.py to c …[truncated]

### L1-6ef4318ec0  (L1, 2026-03-27, sha 6ef4318ec07b, PR #21585)
TITLE: [CI] Move v32 cp test to deepep running suite (#21585)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: test/registered/cp/test_deepseek_v32_cp_single_node.py (+1/-1)
LABELS: deepseek
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-f0303fd07e  (L1, 2026-03-29, sha f0303fd07eb0, PR #18461)
TITLE: [Intel GPU] Enable DeepSeek R1 inference on XPU (#18461)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/token_dispatcher/standard.py (+9/-3); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+9/-4); benchmark/kernels/quantization/tuning_block_wise_kernel.py (+22/-20); python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py (+2/-1); python/sglang/srt/models/deepseek_common/utils.py (+2/-0); python/sglang/srt/models/deepseek_v2.py (+2/-0)
LABELS: deepseek, ready-to-merge, intel, xpu, run-ci
BODY: ## Motivation ⏎  ⏎ Enable DeepSeek-R1 model inference for XPU ⏎ use FP8 precision through triton ⏎  ⏎ ## Modifications ⏎  ⏎ 1. run bmm in bf16 for absorb ⏎ 2. Add XPU support for benchmarking ⏎ 3. Add EP support ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Not Applicable ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ Not Applicable ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge On calls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/m …[truncated]

### L1-6da8f5f69e  (L1, 2026-03-29, sha 6da8f5f69e0f, PR #14702)
TITLE: fix topk softmax performance issue (#14702)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: sgl-kernel/csrc/cpu/topk.cpp (+3/-5)
LABELS: sgl-kernel, intel, cpu, run-ci
BODY: This is a minor change to fix performance issue of topk softmax. ⏎  ⏎ orginal code uses full sort but we only need topk. ⏎  ⏎ ## Checklist

### L1-7119d59747  (L1, 2026-03-30, sha 7119d5974798, PR #14162)
TITLE: DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication (#14162)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:confirmed-reverts(reverted), corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+15/-7); python/sglang/srt/layers/moe/ep_moe/kernels.py (+73/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-3); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+1/-1); python/sglang/srt/layers/quantization/w4afp8.py (+2/-1)
LABELS: run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 21719 (confirmed_revert, reason=unstated) || deep-study: this PR was reverted by PR 22316 (reland, reason=unstated) || deep-study performance PR (precision_format)
BODY: ## Motivation ⏎ profiling: ⏎ deepseek-R1-0528: ⏎ <img width="3170" height="510" alt="ME1764483666809" src="https://github.com/user-attachments/assets/d2a2c921-7c41-4308-8420-a75bdf40cb83" /> ⏎  ⏎ deepseek-R1-0528-w4afp8: ⏎ <img width="3092" height="576" alt="ME1764483641467" src="https://github.com/user-attachments/assets/6489b43f-839a-4d28-be20-1dd7cef3b564" /> ⏎  ⏎ 1.When DeepEP is enabled, the communication latency of the DeepSeek-R1-0508-W4AFP8 model …[truncated]

### L1-505eb312ec  (L1, 2026-03-31, sha 505eb312ec4d, PR #21719)
TITLE: Revert "DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication" (#21719)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+7/-15); python/sglang/srt/layers/moe/ep_moe/kernels.py (+0/-73); python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-3); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+1/-1); python/sglang/srt/layers/quantization/w4afp8.py (+1/-2)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 14162 reason=unstated
BODY: Reverts sgl-project/sglang#14162

### L1-3c91ebdf55  (L1, 2026-03-31, sha 3c91ebdf5526, PR #21466)
TITLE: [2/n] lora - Shared outer experts and support qwen3_30b_a3b_instruct (#21466)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/lora/lora_moe_runners.py (+16/-11); python/sglang/srt/lora/layers.py (+60/-11); python/sglang/srt/lora/lora.py (+25/-2); python/sglang/srt/lora/lora_manager.py (+51/-9); python/sglang/srt/lora/mem_pool.py (+124/-54); python/sglang/srt/lora/triton_ops/sgemm_lora_b.py (+4/-3); python/sglang/srt/server_args.py (+9/-0); test/registered/lora/test_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py (+151/-0)
LABELS: lora, run-ci
BODY: ## Motivation ⏎  ⏎ <img width="1376" height="768" alt="Generated_image" src="https://github.com/user-attachments/assets/c968e763-c613-4d61-ad97-3ba40dc90204" /> ⏎  ⏎ - Support: hared outer expert LoRA ⏎  ⏎ - Support:  Qwen3-30B-A3B-Instruct-2507 ⏎  ⏎ ## Modifications ⏎  ⏎ - Shared outer expert LoRA: Supports MoE adapters where gate_up lora_A and down lora_B are shared across all experts (expert_dim=1) instead of per-expert. The memory pool allocates collap …[truncated]

### L1-f60f2ccc10  (L1, 2026-03-31, sha f60f2ccc10cf, PR #21780)
TITLE: [Fix] Fall back to triton MOE for GPT-OSS on Blackwell with driver >= 595 (#21780)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/check_env.py (+5/-15); python/sglang/cli/killall.py (+5/-13); python/sglang/srt/server_args.py (+16/-4); python/sglang/srt/utils/common.py (+35/-0)
BODY: ## Summary ⏎ - The `triton_kernels` external package segfaults (SIGSEGV, exit code -11) on B200 (Blackwell) GPUs with NVIDIA driver >= 595.58.03 during MoE forward pass (`matmul_ogs` / `routing` / `topk` kernels) ⏎ - This causes `test_gpt_oss_4gpu.py::test_bf16_120b` to consistently fail on `b200-dgx-020-*` CI runners (driver 595.58.03) while passing on `b200-novita-1` (driver 590.48.01) and all H100 runners ⏎ - Adds shared `get_nvidia_driver_version_s …[truncated]

### L1-1f7cee81da  (L1, 2026-03-31, sha 1f7cee81da9a, PR #21786)
TITLE: [moe] add customized option to moe-a2a-backend (#21786)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+4/-0)
LABELS: run-ci
BODY: ## Motivation ⏎ Add customized option so that `require_mlp_tp_gather` could be dealed correctly without side effect (e.g., deepep stuff) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2.  …[truncated]

### L1-c7adca9992  (L1, 2026-03-31, sha c7adca99929c, PR #21752)
TITLE: Fix kimi-linear launch server error (#21752)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+5/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Main kimi-linear was broken due to self.scaling being deleted. ⏎ ``` ⏎ self.scaling = 1 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim) ⏎ ``` ⏎ It causes the server launch error for kimi-linear model. ⏎ ``` ⏎ ➜  python git:(main) ✗ sglang serve --model-path moonshotai/Kimi-Linear-48B-A3B-Instruct --tp-size 2 --trust-remote --device cuda --host 127.0.0.1 --port 30000 ⏎ Warning: You are sending unauthenticated requests to the …[truncated]

### L1-ca3ba05a7a  (L1, 2026-03-31, sha ca3ba05a7aa5, PR #21422)
TITLE: chore: bump flashinfer version to 0.6.7 (#21422)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/jit_kernel/benchmark/diffusion/bench_fused_norm_scale_shift.py (+5/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1); python/sglang/test/lora_utils.py (+3/-0); test/registered/lora/test_lora_tp.py (+2/-1); test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py (+18/-1)
LABELS: high priority, dependencies, lora, sgl-kernel, run-ci, jit-kernel
ISSUES: #18980 [Bug] GLM 5 Crashes at nsa_backend on B200 | #18989 [Bug] deepseek 3.2 nvfp4 moe-runner-backend=flashinfer_trtllm illegal memory access | #19081 [Bug] Deepseek 3.2 nvfp4 specv2 nsa-decode-backend=trttlm kernel crash
BODY: ## Summary ⏎  ⏎ This PR bumps the flashinfer version to `0.6.7` across all relevant files. ⏎  ⏎ Fix these bugs: ⏎  ⏎ Fix https://github.com/sgl-project/sglang/issues/19081  ⏎ Fix https://github.com/sgl-project/sglang/issues/18989  ⏎ Fix https://github.com/sgl-project/sglang/issues/18980 ⏎  ⏎ , this version include this commit: https://github.com/flashinfer-ai/flashinfer/pull/2726 ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/s …[truncated]

### L1-8950d129bd  (L1, 2026-04-01, sha 8950d129bdee, PR #21233)
TITLE: [refactor] Clean up duplicate flashinfer trtllm moe code (#21233)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_trtllm, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+0/-20); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-319); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+2/-4); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-1)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎ There was redundant code paths for flashinfer_trtllm moe runner backend which is harder to maintain and to add new features for (example: routed moe support for fp4). ⏎  ⏎ https://github.com/sgl-project/sglang/issues/8715 ⏎  ⏎ ## Modifications ⏎  ⏎ Now we always use FusedMoE class which uses the fused func fused_experts_none_to_flashinfer_trtllm. ⏎ FlashInferFP4MoE was redundant and is removed. ⏎ FlashInferFusedMoE was unused (dead code) …[truncated]

### L1-835e19656f  (L1, 2026-04-01, sha 835e19656fcf, PR #21397)
TITLE: Bug fix for llama eagle3 (#21397)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/llama.py (+7/-2); python/sglang/srt/models/llama_eagle3.py (+6/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ After upgrading transformers to version 5.3.0, all existing Eagle checkpoints from earlier versions can no longer properly read or use rope_theta and rope_scaling from the config. For example, this affects models such as lightseekorg/kimi-k2.5-eagle3. Therefore, we would like to add compatibility support for these legacy checkpoints for the time being. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ 1. In llama.py, add fallback logic for RoPE con …[truncated]

### L1-d24ea24e18  (L1, 2026-04-01, sha d24ea24e18cc, PR #20394)
TITLE: [NVIDIA] Enable fp8 flashinfer_trtllm_routed MoE for MiniMax-M2.5 (#20394)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+3/-1); python/sglang/srt/layers/quantization/fp8.py (+6/-2); python/sglang/srt/model_executor/model_runner.py (+2/-0)
LABELS: documentation, quant, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Modifications ⏎  ⏎ * The kernel will always output in bf16, so we might have to cast it back to the hidden states dtype. ⏎ * Enable align_fp8_moe_weights_for_flashinfer_trtllm for the routed version ⏎ * Use same fused_func as non-routed and just use topk output checker to determine which one to run. ⏎ * Fix issue where `getattr` doesn't use the default value when the attr is set to none. ⏎ * Add comments to enable autotune, remove copy when flashinf …[truncated]

### L1-c7d03a6215  (L1, 2026-04-02, sha c7d03a6215f1, PR #21922)
TITLE: Revert "Rollback flashmla to older version [1/2]" (#21922)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/cmake/flashmla.cmake (+47/-11); sgl-kernel/python/sgl_kernel/flash_mla.py (+9/-0)
LABELS: sgl-kernel
DEEP_STUDY: deep-study revert record: reland of PR(s) 21430 reason=unstated
BODY: Reverts sgl-project/sglang#21430

### L1-939cf398a9  (L1, 2026-04-02, sha 939cf398a9ec, PR #17985)
TITLE: [MUSA][9/N] Add FA3 attention backend support through MATE (MUSA AI Tensor Engine) (#17985)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/pyproject_other.toml (+3/-0); python/sglang/srt/configs/model_config.py (+6/-0); python/sglang/srt/environ.py (+3/-0); python/sglang/srt/hardware_backend/musa/__init__.py (+1/-0); python/sglang/srt/hardware_backend/musa/attention/__init__.py (+14/-0); python/sglang/srt/hardware_backend/musa/attention/flash_attention.py (+254/-0); python/sglang/srt/layers/attention/attention_registry.py (+16/-7); python/sglang/srt/layers/attention/flashattention_backend.py (+230/-39); python/sglang/srt/server_args.py (+8/-0)
LABELS: dependencies, run-ci, mthreads
DEEP_STUDY: deep-study: this PR was reverted by PR 22002 (confirmed_revert, reason=ci_or_test_failure) || deep-study performance PR (new_kernel_or_fusion)
BODY: ### Motivation ⏎ This PR is the 9th in a series of pull requests (tracked in https://github.com/sgl-project/sglang/issues/16565) to add full support for [Moore Threads](https://en.mthreads.com/) GPUs, leveraging MUSA (Meta-computing Unified System Architecture) to accelerate LLM inference. ⏎  ⏎ ### Modifications ⏎ This commit adds support for the fa3 attention backend powered by [MATE (MUSA AI Tensor Engine)](https://github.com/MooreThreads/mate). ⏎  …[truncated]

### L1-34ddf135fd  (L1, 2026-04-02, sha 34ddf135fd2d, PR #19163)
TITLE: [Feature] Stronger transformers modeling backend with TP, PP, MoE, VLMs, and torch compile (#19163)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+12/-3); python/sglang/srt/disaggregation/encode_receiver.py (+5/-0); python/sglang/srt/managers/io_struct.py (+2/-0); python/sglang/srt/managers/multimodal_processor.py (+30/-2); python/sglang/srt/managers/scheduler.py (+43/-14); python/sglang/srt/managers/tokenizer_manager.py (+10/-1); python/sglang/srt/model_executor/model_runner.py (+10/-0); python/sglang/srt/model_loader/utils.py (+141/-15); python/sglang/srt/models/qwen2.py (+1/-0); python/sglang/srt/models/transformers.py (+1492/-149); (+4 more)
LABELS: Multi-modal, run-ci
BODY: ## Motivation ⏎  ⏎ Adds a generic modeling backend that uses HF transformers models directly via AutoModel.from_config(), enabling any model with a tp_plan, pp_plan and custom attention support to run on SGLang without a dedicated model implementation. ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  Architecture ⏎  ⏎   Mixin-based design that composes capabilities: ⏎   • `TransformersBase` - core class: meta-device init, recursive module replacement (Linear -> TP, RMSN …[truncated]

### L1-991f3aa5b3  (L1, 2026-04-03, sha 991f3aa5b3e7, PR #19652)
TITLE: [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+) (#19652)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.marlin
FILES: python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h (+21/-32); python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh (+2/-0); python/sglang/jit_kernel/moe_wna16_marlin.py (+23/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+33/-10); python/sglang/srt/layers/moe/moe_runner/marlin.py (+9/-1); docs/references/environment_variables.md (+2/-0); python/sglang/jit_kernel/csrc/gemm/marlin/marlin_template.h (+8/-17); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py (+32/-1); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py (+66/-8); (+6 more)
LABELS: documentation, high priority, quant, sgl-kernel, blackwell, run-ci, jit-kernel
DEEP_STUDY: deep-study: this PR was reverted by PR 22047 (confirmed_revert, reason=premature_or_process) || deep-study performance PR (precision_format)
BODY: ## Motivation ⏎ Related Issue: #19491 ⏎ NVFP4-quantized models (e.g., `nvidia/Llama-3.1-8B-Instruct-NVFP4`, `nvidia/DeepSeek-V3-0324-FP4`, `mistralai/Minimax-M2.5-NVFP4`) crash immediately on non-Blackwell GPUs because `get_min_capability()` returned 100. ⏎ This forces users on A100/A40/H100/RTX 3090 to fall back to less accurate quantization (AWQ/GPTQ) or switch to vLLM, which already supports this via Marlin fallback. This PR brings equivalent fun …[truncated]

### L1-6aafe756b9  (L1, 2026-04-03, sha 6aafe756b969, PR #22047)
TITLE: Revert "[Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+… (#22047)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.marlin
FILES: python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h (+32/-21); python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh (+0/-2); python/sglang/jit_kernel/moe_wna16_marlin.py (+1/-23); python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+10/-33); python/sglang/srt/layers/moe/moe_runner/marlin.py (+1/-9); docs/references/environment_variables.md (+0/-2); python/sglang/jit_kernel/csrc/gemm/marlin/marlin_template.h (+17/-8); python/sglang/srt/environ.py (+0/-1); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py (+1/-32); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py (+8/-66); (+6 more)
LABELS: documentation, quant, sgl-kernel, blackwell, jit-kernel
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 19652 reason=premature_or_process
BODY: …) (#19652)" ⏎  ⏎ This reverts commit 991f3aa5b3e7ecea5f525d57cda564722b4a6f27. ⏎  ⏎ Firstly it introduces merge conflicts through at least 2 places (I haven't checked the kernel code, only python. thus there could be more), so I'm uncertain if it has problem in other places...

### L1-84118acf50  (L1, 2026-04-03, sha 84118acf50b3, PR #22009)
TITLE: chore: bump sglang-kernel version to 0.4.1 (#22009)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sglang-kernel` version to `0.4.1` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.4.1` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the `sglang-kernel` dependency are updat …[truncated]

### L1-ac1e437f6a  (L1, 2026-04-03, sha ac1e437f6a63, PR #22078)
TITLE: Revert "[Feature] JIT activation and update skills (by codex)" (#22078)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.runner.triton, L1.runner.deep_gemm, L1.runner.openai_triton_kernels, L1.runner.marlin, L1.upstream.openai_triton_kernels, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+1/-1); python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+2/-6); python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+1/-2); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-3); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+1/-7); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+1/-1); python/sglang/srt/layers/moe/moe_runner/triton.py (+17/-16); python/sglang/srt/lora/lora_moe_runners.py (+5/-4); .claude/skills/add-jit-kernel/SKILL.md (+22/-41); python/sglang/jit_kernel/activation.py (+0/-66); (+6 more)
LABELS: documentation, lora, diffusion, jit-kernel
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 21766 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#21766 for this failure: ⏎ https://github.com/sgl-project/sglang/actions/runs/23958698449/job/69895069178?pr=21913

### L1-ad0516d9c1  (L1, 2026-04-03, sha ad0516d9c1f8, PR #19246)
TITLE: [NPU] optimize glm4.7 (#19246)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/utils.py (+64/-0); python/sglang/srt/layers/quantization/modelslim/modelslim.py (+2/-2); python/sglang/srt/models/glm4_moe.py (+61/-11); python/sglang/srt/models/glm4_moe_nextn.py (+19/-2)
LABELS: npu, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ Optimize glm4.7 performance on NPU. ⏎  ⏎  ⏎ ## Modifications ⏎ 1. Enable dual stream on deepep, one stream for routed experts, the other for shared experts. ⏎ 2. Use rmsnorm_bias to inplace rmsnorm and add bias. ⏎ 3. Use single split_qkv_rmsnorm_rope op to inplace qkv_rmsnorm and rope ops. ⏎  ⏎ ## Accuracy Tests ⏎ Accuracy on gsm8k dataset: ⏎  ⏎ -Accuracy: 0.915 ⏎ -Invalid: 0.000 ⏎ -Latency: 86.270 s ⏎ -Output throughput: 318.951 token/s ⏎  ⏎ ## B …[truncated]

### L1-5118295f7b  (L1, 2026-04-03, sha 5118295f7b0b, PR #22081)
TITLE: [CI] Support CPU stage and auto-batch same-stage files in `/rerun-test` (#22081)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/rerun-test.yml (+55/-2); scripts/ci/utils/slash_command_handler.py (+138/-78)
BODY: ## Summary ⏎ - `/rerun-test` now supports CPU-only tests (files with `register_cpu_ci()`) by dispatching to `ubuntu-latest` runner ⏎ - When multiple test files are specified, files targeting the same `(runner_label, use_deepep, is_cpu)` are batched into a single workflow run instead of one run per file ⏎  ⏎ ## Changes ⏎ **`slash_command_handler.py`** ⏎ - `detect_cuda_suite()` → `detect_suite()`: tries `register_cuda_ci` first, falls back to `register_cpu_ci` …[truncated]

### L1-b7ae3b5a9a  (L1, 2026-04-03, sha b7ae3b5a9a57, PR #21851)
TITLE: GLM-4.7 and GLM-4.7-Flash Loading and import format (#21851)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/glm4_moe.py (+130/-57); python/sglang/srt/models/glm4_moe_lite.py (+9/-29)
LABELS: run-ci
BODY: The main issues addressed were: ⏎  ⏎ 1.	GLM-4.7-Flash does not have an Eagle implementation, so it should be removed. ⏎ 2.	Some code import locations and comments needed adjustment, such as non-standard model naming. ⏎ 3.	Some code and conditional logic in glm4_moe.py are not yet aligned with the latest version of deepseek_v2.py, and this part can be updated.

### L1-990c7590b8  (L1, 2026-04-03, sha 990c7590b835, PR #21280)
TITLE: [RL] Support mxfp8 DeepSeek V3 (#21280)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+12/-7); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+86/-38); python/sglang/srt/layers/quantization/fp8.py (+7/-0)
LABELS: high priority, deepseek, run-ci
BODY: ## Motivation ⏎ @humansand ⏎  ⏎ Support Blackwell mxfp8 DeepSeek RL. ⏎  ⏎ Since the `kv_b_proj` can have different contraction axis in absorbed vs non-absorbed MLA mode while mxfp8 is 1d quantization, for better train-inference consistency and to avoid requantization behavior I have decided to keep it always bf16. For DeepSeek V3, the size of bf16 `kv_b_proj` is `32768 x 512 x 2bytes x 61layers = 1.90625gb`, not a big overhead. ⏎  ⏎  ⏎ ## Modifications ⏎  …[truncated]

### L1-44e5d35703  (L1, 2026-04-03, sha 44e5d357037c, PR #21766)
TITLE: [Feature][JIT Kernel] JIT activation and update skills (by codex) (#21766)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.runner.triton, L1.runner.deep_gemm, L1.runner.openai_triton_kernels, L1.runner.marlin, L1.upstream.openai_triton_kernels, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+1/-1); python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+6/-2); python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+2/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+3/-1); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+7/-1); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+1/-1); python/sglang/srt/layers/moe/moe_runner/triton.py (+16/-17); python/sglang/srt/lora/lora_moe_runners.py (+4/-5); .claude/skills/add-jit-kernel/SKILL.md (+41/-22); python/sglang/jit_kernel/activation.py (+66/-0); (+6 more)
LABELS: documentation, lora, run-ci, diffusion, jit-kernel
DEEP_STUDY: deep-study: this PR was reverted by PR 22078 (confirmed_revert, reason=ci_or_test_failure) || deep-study: this PR was reverted by PR 22094 (reland, reason=correctness_or_accuracy)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎ (TL;DR: Performance gain is mostly from PDL and vectorization) ⏎  ⏎ H200: ⏎  ⏎ <img width="1073" height="2010" alt="image" src="https://github.com/user-attachments/assets/b242efc6-ec2a-471f-ab6c-746800f168de" /> ⏎  ⏎ B200: ⏎  ⏎ <img width="1044" height="1957" alt="image" src="https://github.com/user-attachments/assets/fafa5067-9e18-4f4d-b76c-dfa1f577fc95 …[truncated]

### L1-ff8e47edf9  (L1, 2026-04-04, sha ff8e47edf93a, PR #21647)
TITLE: [5/n] Lora support cuda graph (#21647)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/jit_kernel/moe_lora_align.py (+12/-6); python/sglang/srt/lora/lora_moe_runners.py (+80/-30); python/sglang/srt/lora/triton_ops/fused_moe_lora_kernel.py (+1/-1); python/sglang/srt/lora/backend/base_backend.py (+83/-4); python/sglang/srt/lora/layers.py (+16/-14); python/sglang/srt/lora/lora_manager.py (+44/-4); python/sglang/srt/lora/mem_pool.py (+24/-0); python/sglang/srt/lora/utils.py (+10/-1); python/sglang/srt/model_executor/cuda_graph_runner.py (+3/-0); python/sglang/srt/model_executor/model_runner.py (+35/-0); (+5 more)
LABELS: high priority, lora, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ MoE LoRA inference does not support CUDA graph because the forward path dynamically allocates intermediate tensors (torch.empty()) on every call. CUDA graph requires fixed tensor addresses between capture and replay, so these dynamic allocations break graph replay. ⏎  ⏎ ## Modifications ⏎  ⏎ - Pre-allocate MoE CG buffers (Phase 1): Add init_cuda_graph_moe_buffers() to BaseLoRABackend, implemented in TritonLoRABackend and ChunkedSgmvL …[truncated]

### L1-46bf19cdab  (L1, 2026-04-04, sha 46bf19cdab3b, PR #22097)
TITLE: chore: bump flashinfer version to 0.6.7.post2 (#22097)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the flashinfer version to `0.6.7.post2` across all relevant files. ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎ - python/sglang/srt/utils/common.py ⏎  ⏎ 🤖 Generated with GitHub Actions

### L1-f407461ec8  (L1, 2026-04-05, sha f407461ec843, PR #22006)
TITLE: Tiny fix trtllm_fp8_per_tensor_scale_moe_wrapper router_logits dtype (#22006)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+8/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ https://github.com/flashinfer-ai/flashinfer/blob/fe0539318dcc31c76a33a7ed2ab0ee3c94fe6bad/csrc/trtllm_fused_moe_kernel_launcher.cu#L1789 ⏎  ⏎ the dtype of router_logits should be float32 for deepseek routing method ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://githu …[truncated]

### L1-2813cb6d9a  (L1, 2026-04-06, sha 2813cb6d9a5b, PR #21952)
TITLE: [New Model] Gemma 4 (#21952)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=352,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=352,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=704,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); .codespellrc (+1/-1); benchmark/kernels/fused_moe_triton/common_utils.py (+4/-0); benchmark/mmlu/bench_hf.py (+151/-0); python/sglang/lang/chat_template.py (+17/-8); python/sglang/srt/configs/model_config.py (+18/-2); python/sglang/srt/entrypoints/openai/serving_chat.py (+13/-2); (+25 more)
LABELS: quant, Multi-modal, run-ci
BODY: ## Motivation ⏎  ⏎ Add Gemma 4 model support to SGLang. Gemma 4 is Google's next-generation family of open models featuring Dense and MoE architectures, multimodal support (text, image, audio), hybrid reasoning, and native tool calling. ⏎  ⏎ **Supported Models:** ⏎  ⏎ | Model | Architecture | Parameters | ⏎ |-------|-------------|------------| ⏎ | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | Dense | ~2B | ⏎ | [google/gemma-4-E4B-it](http …[truncated]

### L1-490fa9fa44  (L1, 2026-04-07, sha 490fa9fa44c3, PR #21771)
TITLE: [Perf] Restore torch.compile fusion for topk postprocessing (#21771)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+3/-2)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Motivation ⏎  ⏎ PR #16945 reorganized topk logic into `_post_process_topk_ids` but inlined `topk_ids_logical_to_physical` and `_mask_topk_ids_padded_region` instead of calling the existing `@torch.compile`-decorated `_biased_grouped_topk_postprocess`. This was flagged during review by @fzyzcjy ([comment](https://github.com/sgl-project/sglang/pull/16945#discussion_r2682016393)): ⏎  ⏎ > qq: does this mean this will launch a kernel while this should  …[truncated]

### L1-ae38b24cc3  (L1, 2026-04-07, sha ae38b24cc358, PR #20919)
TITLE: [NPU] Support dp-attention for MiniMax2.5 (#20919)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+22/-1); python/sglang/srt/models/minimax_m2.py (+82/-39)
LABELS: run-ci
BODY: ## Motivation ⏎ Support dp-attention for MiniMax2.5 ⏎  ⏎  ⏎ ## Modifications ⏎ Support dp-attention for MiniMax2.5 ⏎ Resolve the issue that fused_topk_native does not support num_token_non_padded is not None. ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ gsm8k test: ⏎ Accuracy: 0.955 ⏎ Invalid: 0.000 ⏎ Latency: 34.479 s ⏎ Output throughput: 566.869 token/s ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎ npu cmd: ⏎ ``` ⏎ export SGLANG_NPU_FUSED_MOE_MODE=2 ⏎ export SGLANG_DEEPEP_NUM_MAX_DISPA …[truncated]

### L1-7546d04c81  (L1, 2026-04-07, sha 7546d04c81e3, PR #21240)
TITLE: [NVIDIA] Enable FP4 flashinfer trtllm routed moe (#21240)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+109/-56); python/sglang/srt/layers/quantization/modelopt_quant.py (+5/-0)
LABELS: quant, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ Enable FP4 support for `--moe-runner-backend=flashinfer_trtllm_routed` ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ sglang serve --model-path nvidia/DeepSeek-R1-0528-FP4-v2 --tp 4 --moe-runner-backend flashinfer_trtllm_routed ⏎ Accuracy: 0.983 ⏎ Invalid: 0.000 ⏎ Latency: 27.109 s ⏎ Output throughput: 4489.171 token/s ⏎  ⏎ sglang serve --model-path /trevor2/saved_models_MiniMax-M2_5_nvfp4_mlp_only --served-model-name MiniMax-M2. …[truncated]

### L1-86e4542f35  (L1, 2026-04-07, sha 86e4542f35fb, PR #22309)
TITLE: Use dedicated runner label for deepep 8-GPU tests (#22309)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+1/-1)
BODY: ## Summary ⏎ - Change `stage-c-test-deepep-8-gpu-h200` job from `runs-on: 8-gpu-h200` to `runs-on: 8-gpu-h200-deepep` ⏎ - DeepEP tests require working RDMA/nvshmem for inter-GPU communication ⏎ - Ion H200 machines have intermittent RDMA port failures (PORT_DOWN states), causing `ibv_modify_qp` timeouts and nvshmem initialization failures ⏎  ⏎ ## Root cause ⏎ Ion H200 machines have RDMA ports in PORT_DOWN state: ⏎ - ion-3: `mlx5_3` PORT_DOWN ⏎ - ion-4: `rocep68s0 …[truncated]

### L1-6131fb5882  (L1, 2026-04-08, sha 6131fb588273, PR #22024)
TITLE: [NPU] enable mla prepare fused kernel only when being mla attn (#22024)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+2/-2)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ MLAPO is only applicable to MLA-based models. Currently, when it is used together with an Eagle draft model, the draft model fails to save its KV cache correctly. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ ```sh ⏎ python3 -m sglang.test.few_shot_gsm8k \ ⏎     --num-questions 200 \ ⏎     --num-shots 5 \ ⏎     --data-path /xxx/gsm8k.jsonl \ ⏎     --max-new-tokens 512 \ ⏎     --parallel 200 \ ⏎     --host http://0.0.0.0 \ ⏎     --p …[truncated]

### L1-db30a63a13  (L1, 2026-04-08, sha db30a63a1395, PR #21610)
TITLE: [sgl-kernel] support > 1024 experts in moe_align_block_size kernel (#21610)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.align.cuda_jit
FILES: python/sglang/jit_kernel/csrc/moe/moe_align_kernel.cu (+580/-0); python/sglang/jit_kernel/moe_align.py (+46/-0); python/sglang/jit_kernel/tests/test_moe_align_block_size.py (+349/-0)
LABELS: high priority, sgl-kernel, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎   The existing `moe_align_block_size` CUDA kernel uses CUB block-level scan primitives that are limited to 1024 threads, which caps the maximum number of experts at 1024. Models with virtual/merged LoRA experts or very large MoE configurations (e.g., 2048 or 4096 experts) hit this limit. This PR adds a v2 kernel path that supports up to 4096 experts. ⏎  ⏎   ## Modifications ⏎  ⏎   - **New `moe_align_block_size_kernel_v2` kernel** (`s …[truncated]

### L1-df3275bd6c  (L1, 2026-04-08, sha df3275bd6c55, PR #22382)
TITLE: chore: bump flashinfer version to 0.6.7.post3 (#22382)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the flashinfer version to `0.6.7.post3` across all relevant files. ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎ - python/sglang/srt/utils/common.py ⏎  ⏎ 🤖 Generated with GitHub Actions

### L1-57ffc55fb6  (L1, 2026-04-09, sha 57ffc55fb647, PR #20089)
TITLE: feat: [1/2] [DeepEP] Fuse shared expert into MoE dispatch under EP (#20089)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+20/-16); python/sglang/srt/layers/moe/topk.py (+82/-8); python/sglang/srt/layers/moe/utils.py (+6/-0); python/sglang/srt/models/deepseek_v2.py (+84/-25); python/sglang/srt/server_args.py (+7/-0)
LABELS: documentation, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ In DeepSeek V3/R1 with expert parallelism (EP), each rank currently computes the shared expert **separately** from routed experts. This means shared expert is not part of the DeepEP dispatch/compute/combine pipeline — it runs as a standalone forward pass on every rank. ⏎  ⏎ This PR fuses the shared expert into the MoE path by treating it as an additional routed expert (the "9th expert" per rank). Instead of computing shared exp …[truncated]

### L1-1df9f4e2f6  (L1, 2026-04-09, sha 1df9f4e2f6ec, PR #22329)
TITLE: [AMD] Add prealloc token env for mori-ep (#22329)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+28/-4); docs/references/environment_variables.md (+1/-0); python/sglang/srt/server_args.py (+1/-1)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ This patch is adding : ⏎ 1. Add `SGLANG_MORI_PREALLOC_MAX_RECV_TOKENS` as new environment to allow user to configure the mori ep token preallocation. ⏎ 2. Add `check_mori_compatibility` to provide backward compatibility  ⏎ 3. Some minor doc fixes ⏎  ⏎ cc @Duyi-Wang  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ UT `test/registered/amd/test_moriep_small.py` passed ⏎ <img width="1587" height="287" alt="image" src="https://githu …[truncated]

### L1-9d905efa2c  (L1, 2026-04-09, sha 9d905efa2c0d, PR #22322)
TITLE: [Docker] Fix Trivy CVEs, cubin download 403s, and kernels command order (#22322)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+35/-3)
BODY: ## Summary ⏎ - **Fix Trivy-reported CVEs**: Add targeted `apt-get install --only-upgrade` layers to patch vulnerable OS packages. Uses `--only-upgrade` to avoid accidentally upgrading NVIDIA CUDA packages. ⏎ - **Remove redundant flashinfer cubin download**: The `python3 -m flashinfer --download-cubin` step was downloading ~10,564 cubins individually from NVIDIA's CDN, causing 403 rate-limiting errors that fail the Docker release CI. The `flashinfer_c …[truncated]

### L1-aa103eab8d  (L1, 2026-04-09, sha aa103eab8df4, PR #22160)
TITLE: [Docker] Optimize Dockerfile for BuildKit layer caching (#22160)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+328/-162)
BODY: ## Summary ⏎  ⏎ - Restructure Dockerfile into parallel multi-stage builds (`torch_deps`, `deepep_builder`, `flashinfer_cache`, `devtools_builder`) so independent work executes concurrently via BuildKit ⏎ - Install dependencies from `pyproject.toml` only (with constraints file), push source `COPY` and editable install to the very last stage -- any Python source change now only invalidates the final layers ⏎ - Build DeepEP as a wheel in an isolated stage,  …[truncated]

### L1-7603b226ce  (L1, 2026-04-09, sha 7603b226ce67, PR #22440)
TITLE: Upgrade sglang-torch-profiler-analysis SKILLS (#22440)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .claude/skills/sglang-torch-profiler-analysis/SKILL.md (+78/-112); .claude/skills/sglang-torch-profiler-analysis/references/fuse-overlap-catalog.md (+120/-8); .claude/skills/sglang-torch-profiler-analysis/references/overlap-catalog.md (+71/-6); .claude/skills/sglang-torch-profiler-analysis/references/trace-workflow.md (+0/-119); .claude/skills/sglang-torch-profiler-analysis/references/validated-workflows.md (+0/-263); .claude/skills/sglang-torch-profiler-analysis/scripts/analyze_sglang_torch_profile.py (+218/-176); .claude/skills/sglang-torch-profiler-analysis/scripts/profile_common.py (+0/-45); .claude/skills/sglang-torch-profiler-analysis/scripts/triage_kernel_helpers.py (+907/-306); .claude/skills/sglang-torch-profiler-analysis/scripts/triage_overlap_helpers.py (+4/-216)
LABELS: documentation
BODY: Update `.claude/skills/sglang-torch-profiler-analysis` to the latest skill layout and profiling workflow. ⏎  ⏎ ## Summary ⏎ - Upgrade `sglang-torch-profiler-analysis` to the new triage-only workflow. ⏎ - Keep the latest deterministic kernel / overlap / fuse rendering logic, including the updated source-attribution heuristics and the `>=1%` render threshold. ⏎ - Refresh the profiler references/catalogs and carry the latest H100 render for `Qwen/Qwen3.5 …[truncated]

### L1-c554dc5c64  (L1, 2026-04-10, sha c554dc5c64b6, PR #21339)
TITLE: Add dedicated FlashInferCuteDslMoE layer for standard-path FP4 MoE (#21339)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:production-kernel-provenance, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.runner.framework, L1.runner.flashinfer_cutedsl
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+353/-0); python/sglang/srt/layers/moe/moe_runner/runner.py (+2/-0); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+9/-4); python/sglang/srt/layers/quantization/modelopt_quant.py (+113/-37); python/sglang/srt/model_executor/model_runner.py (+1/-0); python/sglang/srt/server_args.py (+20/-0); test/registered/backends/test_deepseek_v3_fp4_cutedsl_moe.py (+149/-0); test/registered/moe/test_cutedsl_moe.py (+605/-146)
LABELS: quant, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ We want to have the option of having a more standard `--moe-runner-backend flashinfer_cutedsl` backend that is not specified to DeepEP.  This PR integrates the Wrapper API exposed here: https://github.com/flashinfer-ai/flashinfer/pull/2398. ⏎  ⏎ ## Modifications ⏎  ⏎ ### Server ⏎  ⏎ - Add `flashinfer_cutedsl` as a modular `moe_runner` backend for `--moe-runner-backend flashinfer_cutedsl` with `modelopt_fp4` quantization on the standard …[truncated]

### L1-7dbd0dd9f0  (L1, 2026-04-10, sha 7dbd0dd9f01a, PR #20067)
TITLE: MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn (#20067)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/layernorm.py (+4/-0); python/sglang/srt/models/minimax_m2.py (+25/-6); test/registered/8-gpu-models/test_minimax_m25.py (+10/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ * Enables dp attention for MiniMax-M2.5 which is useful for high thoughput use cases. ⏎ I also added these performance improvements: ⏎ * For DEP, use reduce-scatter after MoE instead of AR + slice ⏎ * For DEP, use fp4 allgather when applicable (quantize before comm) ⏎ * For TP/TEP, also enables flash infer all reduce fusion between MoE and attention of next layer ⏎  ⏎ ## Modifications ⏎  ⏎ Attention modules in MiniMax-M2.5 model definiti …[truncated]

### L1-f7a1740101  (L1, 2026-04-10, sha f7a174010127, PR #22051)
TITLE: [MUSA][9/N] Add FA3 attention backend support through MATE (MUSA AI Tensor Engine) (#22051)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/pyproject_other.toml (+4/-1); python/sglang/srt/configs/model_config.py (+6/-0); python/sglang/srt/environ.py (+3/-0); python/sglang/srt/hardware_backend/musa/__init__.py (+1/-0); python/sglang/srt/hardware_backend/musa/attention/__init__.py (+3/-0); python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py (+913/-0); python/sglang/srt/layers/attention/attention_registry.py (+23/-9); python/sglang/srt/server_args.py (+5/-1)
LABELS: dependencies, run-ci, mthreads, jit-kernel
BODY: ## Motivation ⏎  ⏎ This PR fixes the Flash Attention backend support that was previously merged in PR #17985 but later reverted in PR #22002 due to a bug. The original commit 2373552 caused CI failures (see [failed CI job](https://github.com/sgl-project/sglang/actions/runs/23928333410/job/69789912493)). ⏎  ⏎ Previously, the MUSA-adapted flash attention implementation had a bug in the `_forward_extend_impl` method. The code was missing a proper mechan …[truncated]

### L1-18f41ac427  (L1, 2026-04-10, sha 18f41ac42746, PR #22316)
TITLE: [Reland] DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication (#22316)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:confirmed-reverts, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+15/-7); python/sglang/srt/layers/moe/ep_moe/kernels.py (+73/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+0/-3); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+1/-1); python/sglang/srt/layers/quantization/w4afp8.py (+2/-1)
LABELS: run-ci
DEEP_STUDY: deep-study revert record: reland of PR(s) 14162 reason=unstated || deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ## Motivation ⏎ profiling: ⏎ deepseek-R1-0528: ⏎ <img width="3170" height="510" alt="ME1764483666809" src="https://github.com/user-attachments/assets/d2a2c921-7c41-4308-8420-a75bdf40cb83" /> ⏎  ⏎ deepseek-R1-0528-w4afp8: ⏎ <img width="3092" height="576" alt="ME1764483641467" src="https://github.com/user-attachments/assets/6489b43f-839a-4d28-be20-1dd7cef3b564" /> ⏎  ⏎ 1.When DeepEP is enabl …[truncated]

### L1-2ab141547d  (L1, 2026-04-10, sha 2ab141547d59, PR #22413)
TITLE: [CPU] Add apply_routed_scaling_factor_on_output support for biased_grouped_topk fusion (#22413)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py, L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/layers/moe/topk.py (+1/-2); sgl-kernel/csrc/cpu/topk.cpp (+55/-26); sgl-kernel/csrc/cpu/common.h (+35/-34); test/srt/cpu/test_topk.py (+34/-10)
LABELS: sgl-kernel, run-ci
BODY: This PR: ⏎  ⏎ 1.  removes the limit of `apply_routed_scaling_factor_on_output` for CPU path of the fusion biased_grouped_topk_cpu ⏎ 2.  add fp32 dtype support for gating_output ⏎ 3.  refine topk expert numebers

### L1-8da1cfb30d  (L1, 2026-04-11, sha 8da1cfb30d12, PR #21858)
TITLE: [lora][moe] Decoupled LoRA MoE backend with Marlin support (#21858)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.runner.framework, L1.runner.triton, L1.runner.deep_gemm, L1.runner.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/moe_runner/base.py (+6/-2); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+2/-1); python/sglang/srt/layers/moe/moe_runner/runner.py (+49/-17); python/sglang/srt/layers/moe/moe_runner/triton.py (+22/-4); python/sglang/srt/layers/moe/moe_runner/triton_kernels.py (+2/-1); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+7/-7); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py (+17/-0); python/sglang/srt/lora/lora_moe_runner_marlin.py (+206/-0); python/sglang/srt/lora/lora_moe_runners.py (+361/-500); python/sglang/srt/lora/layers.py (+31/-8); (+2 more)
LABELS: high priority, lora, run-ci
BODY: ## Motivation ⏎  ⏎ LoRA adapters applied to MoE layers currently couple LoRA injection logic into each backend-specific runner subclass (e.g., `TritonRunnerCoreWithLoRA`), making it difficult to add new backends. This PR: ⏎  ⏎ 1. Refactors the LoRA MoE runner architecture from per-backend subclasses to a **generic hook-based injection** pattern, decoupling LoRA logic from backend-specific code. ⏎ 2. Adds a **Marlin int4/int8 MoE backend** with LoRA su …[truncated]

### L1-701a0e0c25  (L1, 2026-04-12, sha 701a0e0c2551, PR #22491)
TITLE: [CI/Docker] Clean up redundant flashinfer cubin downloads (#22491)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-4); scripts/ci/cuda/ci_download_flashinfer_cubin.sh (+0/-62); scripts/ci/cuda/ci_install_dependency.sh (+1/-3)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ Follow-up to #22322.  ⏎  ⏎ Cleans up redundant flashinfer cubin download steps across the CI workflows and the main Dockerfile that were missed in the previous PR.  ⏎  ⏎ cc @Kangyan-Zhou

### L1-5593539942  (L1, 2026-04-12, sha 559353994266, PR #22204)
TITLE: [RL] Refactor NVFP4 shuffling/swizzling to in-place replacement (#22204)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+17/-34); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py (+9/-26); python/sglang/srt/layers/quantization/modelopt_quant.py (+6/-7); python/sglang/srt/server_args.py (+2/-1); test/registered/backends/test_flashinfer_trtllm_gen_moe_backend.py (+55/-0); test/registered/rl/test_update_weights_from_disk_blackwell.py (+68/-35)
LABELS: high priority, quant, blackwell, npu, run-ci
BODY: ## Motivation ⏎ @humansand ⏎  ⏎ https://github.com/sgl-project/sglang/pull/18085 is an earlier fix for nvfp4 weight update but it did not fix trtllm backend, since trtllm backend used `*_weights_fp4_shuffled` tensors, which requires broader refactoring for in-pace replacement. ⏎  ⏎ This PR replaces all `*_weights_fp4_shuffled` with the original weight tensor and conducts swizzling/shuffling with in-place replacement. ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - …[truncated]

### L1-3f4fbc165d  (L1, 2026-04-12, sha 3f4fbc165d83, PR #21441)
TITLE: Upgrade CI default CUDA version from 12.9 to 13.0 (#21441)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: scripts/ci/cuda/ci_install_deepep.sh (+2/-7); .github/workflows/pr-test.yml (+14/-17); python/pyproject.toml (+4/-4); scripts/ci/cuda/ci_download_flashinfer_jit_cache.sh (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+12/-4)
LABELS: high priority, dependencies, sgl-kernel, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 22727 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Summary ⏎ - Upgrade CI default CUDA from 12.9 to 13.0 to match Torch 2.11's default ⏎ - All CI runners already have driver 580+ (CUDA 13.0 capable) ⏎ - B200 Novita upgraded to driver 590 (CUDA 13.1) ⏎  ⏎ ## Changes ⏎ - `ci_install_dependency.sh`: `CU_VERSION` cu129 → cu130 ⏎ - `docker/Dockerfile`: default `CUDA_VERSION` 12.9.1 → 13.0.2, added 13.0.2 to version maps ⏎ - `python/pyproject.toml`: `cuda-python` >=13.0, torch index cu129 → cu130 ⏎ - `sgl-kernel/Dock …[truncated]

### L1-90ef8ce54d  (L1, 2026-04-13, sha 90ef8ce54de7, PR #22653)
TITLE: [Docker] Remove flashinfer cache copy (#22653)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-2)
BODY: ## Summary ⏎  ⏎ Fix https://github.com/sgl-project/sglang/actions/runs/24320223003/job/71004907738#step:8:8520. After #22491 removed the manual flashinfer cubin download steps, `/root/.cache/flashinfer` is no longer populated. ⏎  ⏎ cc @Kangyan-Zhou

### L1-1e9eecfa36  (L1, 2026-04-13, sha 1e9eecfa36ce, PR #22417)
TITLE: [Intel GPU] Enable sgl-kernel-xpu fused_experts MoE kernel path for GPT-OSS bf16 models. (#22417)
SOURCES: path_integration+keyword, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/unquant.py (+2/-0)
LABELS: quant, intel, xpu, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Enable sgl-kernel-xpu fused_experts MoE kernel path for GPT-OSS bf16 models. ⏎  ⏎ ## Modifications ⏎  ⏎ Pass the gemm1_alpha and gemm1_limit parameter to the fused_experts kernel call. ⏎  ⏎ ## Accuracy Tests ⏎ - Ran GSM8K test and it's similar to results on Nvidia A100. ⏎ ``` ⏎ # Run server ⏎ SGLANG_USE_SGL_XPU=1 python3 -m sglang.launch_server --model lmsys/gpt-oss-20b-bf16 --disable-radix-cache --tp 4 --attention-backend triton ⏎  ⏎ # Run  …[truncated]

### L1-b441317aa4  (L1, 2026-04-13, sha b441317aa430, PR #22727)
TITLE: Revert "Upgrade CI default CUDA version from 12.9 to 13.0" (#22727)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: scripts/ci/cuda/ci_install_deepep.sh (+7/-2); .github/workflows/pr-test.yml (+17/-14); python/pyproject.toml (+4/-4); scripts/ci/cuda/ci_download_flashinfer_jit_cache.sh (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+4/-12)
LABELS: dependencies
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 21441 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#21441

### L1-f4f9e68189  (L1, 2026-04-13, sha f4f9e6818916, PR #21097)
TITLE: [AMD] Add MoE weights and scales padding (#21097)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+18/-12); python/sglang/srt/layers/moe/utils.py (+51/-0); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py (+18/-4); python/sglang/srt/layers/quantization/fp8.py (+37/-20); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py (+20/-5); python/sglang/srt/model_executor/model_runner.py (+5/-1)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ Right now, Aiter MoE requires weights and scales to align with a fixed number. Since some models have intermediate sizes that don't fit this rule, we need to add extra padding to the weights so they can be processed by the Fused MoE. ⏎  ⏎ ## Modifications ⏎  ⏎ Add padding for the weights. Below listed are the models and configurations that has been verified with: ⏎  ⏎ 1. Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 : TP=8 ⏎ 2. amd/Qwen3-235B- …[truncated]

### L1-ff13dfee45  (L1, 2026-04-13, sha ff13dfee45df, PR #22122)
TITLE: [lora][moe] Virtual experts for LoRA MoE (#22122)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.runner.framework
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+36/-13); python/sglang/srt/layers/moe/moe_runner/runner.py (+9/-16); python/sglang/srt/lora/lora_moe_runners.py (+267/-76); python/sglang/srt/lora/layers.py (+2/-0); python/sglang/srt/lora/lora_manager.py (+2/-1); python/sglang/srt/lora/triton_ops/__init__.py (+2/-0); python/sglang/srt/lora/triton_ops/virtual_experts.py (+662/-0); python/sglang/srt/server_args.py (+14/-0); test/registered/lora/test_lora_moe_runner.py (+153/-0); test/registered/lora/test_marlin_lora_correctness.py (+1/-0)
LABELS: high priority, lora, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ **NOTE: depends on via the hooks-based architecture in #21858** ⏎  ⏎ This PR introduces **virtual expert computation** for LoRA+MoE: instead of iterating over each LoRA adapter separately (one alignment + kernel call per adapter), we treat `[num_loras, num_experts]` weight combinations as a flat `[virtual_num_experts]` space. This allows LoRA deltas to be computed in a single fused MoE kernel call by reusing the existing `invoke_fu …[truncated]

### L1-657945c338  (L1, 2026-04-13, sha 657945c3380b, PR #22642)
TITLE: Replace all-reduce + dp_scatter with reduce_scatterv for DP attention (#22642)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/__init__.py (+2/-0); python/sglang/srt/layers/moe/utils.py (+15/-0); python/sglang/srt/layers/communicator.py (+9/-2); python/sglang/srt/models/qwen2_moe.py (+2/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ For DP attention with Expert Parallelism (EP), the default MoE communication path performs two separate operations after the MoE kernel: ⏎  ⏎ 1. `tensor_model_parallel_all_reduce` — reduces expert outputs across all DP workers ⏎ 2. `dp_scatter` — extracts each worker's local token slice from the global result ⏎  ⏎ This is functionally equivalent to a single `reduce_scatterv`, which fuses the reduce and scatter into one NCCL collective …[truncated]

### L1-3cb3f7c018  (L1, 2026-04-14, sha 3cb3f7c01814, PR #22525)
TITLE: fix: EPLB dispatch OOB when shared experts fusion enabled under DeepEP (#22525)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+15/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ When --enforce-shared-experts-fusion and --init-expert-location are both enabled with DeepEP backend, biased_grouped_topk_gpu appends shared expert columns (value = n_routed_experts = 256) to topk_ids. The subsequent EPLB dispatch in _biased_grouped_topk_postprocess uses topk_ids as an index into the logical-to-physical dispatch table, which only has 256 entries (0-255). This causes a CUDA device-side assert (index out of bound …[truncated]

### L1-e15401ee0e  (L1, 2026-04-14, sha e15401ee0eb4, PR #22537)
TITLE: Add runai-model-streamer into Python packages installed in Dockerfile and fix NotADirectoryError Docker regression (#22537)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+9/-8)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Currently, when using `--load-format=runai_streamer` to load a model from Google Cloud Storage (`gs://...`) using the official SGLang Docker image, the server crashes with the following error: ⏎  ⏎ ```bash ⏎   File "/usr/local/lib/python3.12/dist-packages/runai_model_streamer/s3_utils/s3_utils.py", line 137, in gcs_pull_files ⏎     raise ImportError("GCS files module not found. Please install the required package.") ⏎ ImportError: GCS …[truncated]

### L1-ea05ea5abe  (L1, 2026-04-14, sha ea05ea5abed1, PR #20736)
TITLE: [AMD] Enable share expert fusion with router experts for Qwen3.5 BF16 & FP8 (#20736)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen2_moe.py (+108/-5); python/sglang/srt/models/qwen3_5.py (+110/-3)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Qwen2 MoE and Qwen3.5 MoE models use a **shared expert** in addition to routed experts. When `shared_expert_intermediate_size == moe_intermediate_size`, the shared expert can be fused with routed experts so that each token attends to top-k routed experts plus one shared expert (topk+1) in a single MoE dispatch, reducing kernel launches and improving inference efficiency. This PR adds shared expert fusion support for Qwen2 MoE (wh …[truncated]

### L1-adb310b976  (L1, 2026-04-14, sha adb310b976d6, PR #22820)
TITLE: Cleanup server_args.py and minor code tidying (#22820)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-0); python/sglang/srt/managers/scheduler.py (+23/-14); python/sglang/srt/server_args.py (+37/-40)
LABELS: run-ci
BODY: ## Summary ⏎ - Inline `MAMBA_SSM_DTYPE_CHOICES` directly into `add_cli_args` and remove the unused constant and `add_mamba_ssm_dtype_choices()` function ⏎ - Reorder constants and helper functions in `server_args.py` for better grouping ⏎ - Move import to top-level in `fused_moe.py` ⏎  ⏎ ## Test plan ⏎ - No behavioral changes — pure cleanup (moves, inlining, import reorder) ⏎ - CI should pass as-is

### L1-454228e071  (L1, 2026-04-14, sha 454228e071aa, PR #20016)
TITLE: hicache storage backend mooncake support ascend hixl (#20016)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py (+2/-1); python/sglang/srt/model_executor/model_runner.py (+3/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ mooncake_master ⏎  ⏎ ``` ⏎ ./mooncake_master --enable_http_metadata_server=true  --http_metadata_server_port=20000  --http_metadata_server_host=0.0.0.0 ⏎ ``` ⏎  ⏎ prefill ⏎ ``` ⏎ source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ⏎  ⏎ echo $ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE ⏎  ⏎ export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True ⏎ export HCCL_INTRA_ROCE_ENABLE=1 ⏎ export GLOO_SOCKET_IFNAME=enp61s0f0 ⏎ export HCCL_SOCKET_IFNAM …[truncated]

### L1-b2af34be54  (L1, 2026-04-14, sha b2af34be5404, PR #22844)
TITLE: [AMD] Optimize _append_shared_to_topk_output by a single fused Triton kernel for Qwen3.5 (#22844)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+76/-0); python/sglang/srt/models/qwen2_moe.py (+10/-13)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ - `_append_shared_to_topk_output` previously used 4 separate kernel launches (2 elementwise + 2 concat) to append shared expert IDs and weights. ⏎ - This path is on the critical MoE routing path and adds avoidable launch overhead. ⏎ - Qwen3.5 requires per-token shared weights from learned `shared_expert_gate`, so the optimization must preserve token-wise weights. ⏎  ⏎ ## Modifications ⏎  ⏎ - Added a fused Triton kernel in `python/sglan …[truncated]

### L1-4e480d5785  (L1, 2026-04-15, sha 4e480d5785b4, PR #21776)
TITLE: Harden FlashInfer FP4 imports in standard dispatcher (#21776)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/token_dispatcher/standard.py (+15/-10)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The `flashinfer_cutlass` FP4 all-gather path already depends on FlashInfer block-scale interleaving, so keeping a separate JIT fallback for activation quantization in `standard.py` was misleading and effectively dead. This change makes that dependency explicit and avoids failing with a raw `ImportError` before the code can report a clearer message. ⏎  ⏎ ## Modifications ⏎  ⏎ - remove the dead SM120-specific `fp4_quantize` fallback lo …[truncated]

### L1-2b0f349927  (L1, 2026-04-15, sha 2b0f349927bf, PR #22903)
TITLE: ci: clarify srt-slurm issue filing for incompatible flag combos (#22903)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: scripts/ci/slurm/analyze_logs_with_modal.py (+9/-11); scripts/ci/slurm/log_analysis_prompt.md (+167/-80)
LABELS: documentation
BODY: ## Summary ⏎ - Clarifies in the log analyzer prompt that when a recipe passes an incompatible combination of flags to SGLang, that's a recipe bug and should be filed against `NVIDIA/srt-slurm` — even if the error surfaces in SGLang code ⏎  ⏎ Follow-up to #22899. In the first test run, the analyzer correctly identified that `flashinfer_cutedsl` + `deepep` is an unsupported combination, but didn't file an issue against srt-slurm because the error appeare …[truncated]

### L1-8c190f6b91  (L1, 2026-04-16, sha 8c190f6b9183, PR #22952)
TITLE: [AMD] Add SGLANG_MORI_MOE_MAX_INPUT_TOKENS to truncate dispatch before MoE. (#22952)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+18/-1); docs/references/environment_variables.md (+1/-0)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ In the MoriEP MoE path, `dispatch_a1` (the dispatch output hidden states) has a fixed first dimension of `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK * ep_size` (or `SGLANG_MORI_PREALLOC_MAX_RECV_TOKENS`). However, the actual number of valid tokens (`totalRecvTokenNum`) is often much smaller than this buffer size. ⏎  ⏎ `aiter.fused_moe` uses the first dimension of the input tensor (`dispatch_a1.shape[0]`) to select the kernel disp …[truncated]

### L1-5f7aee726a  (L1, 2026-04-17, sha 5f7aee726a10, PR #23019)
TITLE: refactor(moe): de-duplicate triton MoE runner path into shared helpers (#23019)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:production-kernel-provenance, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.triton.moe_align, L1.routing.topk_py, L1.runner.triton
FILES: python/sglang/srt/layers/moe/fused_moe_triton/__init__.py (+6/-25); python/sglang/srt/layers/moe/moe_runner/triton.py (+59/-316); python/sglang/srt/layers/moe/moe_runner/triton_utils/__init__.py (+36/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_1_0/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_1_0/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_1_0/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_1_0/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_1_0/E=1,N=3072,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_1_0/E=1,N=3072,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_1_0/E=1,N=3072,device_name=NVIDIA_H100_80GB_HBM3.json (+0/-0); (+312 more)
LABELS: documentation, quant, amd, lora, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ `TritonRunnerCore.run` (in `moe_runner/triton.py`) and `fused_experts_impl` (in `fused_moe_triton/fused_moe.py`) had grown to ~95% identical logic — same two-kernel + activation + combine pipeline, same platform dispatch ladders, same activation variants — but with subtle divergences (runner missed `filter_expert`, TMA, `enable_fused_moe_sum_all_reduce`, non-gated activations, the sgl-kernel `moe_sum_reduce` path, and the HIP small …[truncated]

### L1-6ecd6f84db  (L1, 2026-04-19, sha 6ecd6f84dbf9, PR #23119)
TITLE: [CI] Add per-job uv venv isolation and upgrade CI version to Cuda 13 (#23119)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+17/-10); scripts/ci/cuda/ci_install_deepep.sh (+43/-10); .github/workflows/pr-test-multimodal-gen.yml (+5/-5); .github/workflows/pr-test-sgl-kernel.yml (+4/-4); .github/workflows/pr-test.yml (+75/-22); python/sglang/jit_kernel/tests/test_pos_enc.py (+2/-2); python/sglang/multimodal_gen/configs/models/dits/wanvideo.py (+23/-1); python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py (+1/-0); python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py (+7/-1); python/sglang/multimodal_gen/runtime/loader/fsdp_load.py (+3/-1); (+29 more)
LABELS: high priority, quant, dependencies, lora, Multi-modal, hicache, sgl-kernel, run-ci, diffusion, jit-kernel
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-1cff871c67  (L1, 2026-04-19, sha 1cff871c67ef, PR #23185)
TITLE: [Bugfix] Fix DeepEP timeout when compiling DeepGeMM in EP+DP+TP (#23185)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+14/-1); python/sglang/compile_deep_gemm.py (+40/-14)
BODY: ## Motivation ⏎  ⏎ When compiling DeepGeMM for Kimi K2 EP+DP+TP 32, I hit a DeepEP timeout: ⏎  ⏎ ``` ⏎ File "/root/code/.venv/lib/python3.12/site-packages/sglang/srt/layers/moe/token_dispatcher/deepep.py", line 458, in _dispatch_core ⏎     ) = buffer.dispatch( ⏎         ^^^^^^^^^^^^^^^^ ⏎   File "/root/code/.venv/lib/python3.12/site-packages/deep_ep/buffer.py", line 376, in dispatch ⏎     return self.internode_dispatch(x, handle, num_tokens_per_rank, num_tokens_pe …[truncated]

### L1-b4bb036b73  (L1, 2026-04-20, sha b4bb036b7308, PR #22925)
TITLE: fix legacy deepep path for flashinfer_cutedsl (#22925)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+18/-5); python/sglang/srt/layers/moe/utils.py (+8/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+70/-8); python/sglang/srt/model_executor/model_runner.py (+11/-0); test/registered/moe/test_cutedsl_moe.py (+557/-180)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎ The recent PR  https://github.com/sgl-project/sglang/pull/21339  accidentally made it impossible to use the existing cutedsl moe backend + deepep a2a (`--moe-runner-backend flashinfer_cutedsl --moe-a2a-backend deepep`). ⏎  ⏎ This manifested here: ⏎ [Recipe bug: flashinfer_cutedsl moe-runner-backend incompatible with deepep a2a-backend #39](https://github.com/NVIDIA/srt-slurm/issues/39). ⏎  ⏎ This PR restores the previous DeepEP behavi …[truncated]

### L1-09b1d10d59  (L1, 2026-04-21, sha 09b1d10d5938, PR #23156)
TITLE: [AMD] prepare for MI300x PR runner pool: registry mirror, runner routing, threshold tuning (#23156)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/nightly-test-amd-rocm720.yml (+2/-0); .github/workflows/nightly-test-amd.yml (+2/-0); .github/workflows/pr-test-amd-rocm720.yml (+2/-0); .github/workflows/pr-test-amd.yml (+46/-51); scripts/ci/amd/amd_ci_start_container.sh (+19/-19); scripts/ci/amd/amd_ci_start_container_disagg.sh (+15/-19); test/registered/amd/test_deepseek_v32_basic.py (+1/-1); test/registered/amd/test_deepseek_v3_mtp.py (+2/-1); test/registered/amd/test_moriep_small.py (+1/-1); test/registered/moe/test_torch_compile_moe.py (+4/-0); (+4 more)
LABELS: quant, amd, lora, Multi-modal, deepseek, run-ci
BODY: ## Motivation ⏎ Prepare the AMD CI to run PR tests on the new `linux-mi300-*gpu-sglang` runner pool while keeping scheduled runs on `linux-mi325-*gpu-sglang`. This involves three coupled changes: ⏎ 1. **In-network image mirror** so PR jobs don't hammer Docker Hub ⏎ 2. **Runner pool routing** that picks MI300x for PRs, MI325x for schedules, user-selectable for `workflow_dispatch` ⏎ 3. **MI300x-specific test budget tuning** — perf thresholds, est_time, …[truncated]

### L1-c560326884  (L1, 2026-04-21, sha c56032688404, PR #22911)
TITLE: [perf] support return_routed_experts with overlap scheduling (#22911)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/routed_experts_capturer.py (+72/-22); python/sglang/srt/managers/scheduler_output_processor_mixin.py (+6/-0); python/sglang/srt/managers/tp_worker.py (+1/-0); python/sglang/srt/managers/utils.py (+7/-0); python/sglang/srt/model_executor/model_runner.py (+5/-2); python/sglang/srt/speculative/eagle_worker_v2.py (+1/-0); python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py (+1/-0)
LABELS: high priority, run-ci
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Before, ⏎  ⏎ <img width="1013" height="802" alt="image" src="https://github.com/user-attachments/assets/ce3be8cd-13d7-4c3c-a539-9e81952eafb1" /> ⏎  ⏎ After, ⏎  ⏎ <img width="854" height="695" alt="image" src="https://github.com/user-attachments/assets/7f59c741-bc3a-43c7-8c64-6489448ff406" /> ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎ h200 ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-pat …[truncated]

### L1-6cf0b004ca  (L1, 2026-04-21, sha 6cf0b004ca78, PR #22791)
TITLE: [MoE] Add LFM2 MoE tuning support + tuned configs for H100/B200/MI325X (#22791)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=32,N=1792,device_name=NVIDIA_B200.json (+154/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=32,N=1792,device_name=NVIDIA_H100_80GB_HBM3.json (+154/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=32,N=224,device_name=NVIDIA_B200.json (+154/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=32,N=224,device_name=NVIDIA_H100_80GB_HBM3.json (+154/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=32,N=448,device_name=NVIDIA_B200.json (+154/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=32,N=448,device_name=NVIDIA_H100_80GB_HBM3.json (+154/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=32,N=896,device_name=NVIDIA_B200.json (+154/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=32,N=896,device_name=NVIDIA_H100_80GB_HBM3.json (+154/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=64,N=1536,device_name=NVIDIA_B200.json (+154/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=64,N=1536,device_name=NVIDIA_H100_80GB_HBM3.json (+154/-0); (+15 more)
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Summary ⏎  ⏎ Adds `Lfm2MoeForCausalLM` to the MoE tuning script and ships tuned fused MoE triton kernel configs for LFM2-8B-A1B and LFM2-24B-A2B at TP=1,2,4,8 on NVIDIA H100, B200, and AMD Instinct MI325X. Up to **+47% throughput** over default configs at high concurrency on NVIDIA. ⏎  ⏎ ## Motivation ⏎  ⏎ LFM2 MoE models (`LiquidAI/LFM2-8B-A1B`, `LiquidAI/LFM2-24B-A2B`) use `num_experts` / `moe_intermediate_size` config keys. The default Mixtral fa …[truncated]

### L1-bf5e71dcec  (L1, 2026-04-21, sha bf5e71dcec80, PR #23361)
TITLE: [MUSA][19/N] Support HiCache with pin_memory allocator (#23361)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/mem_cache/memory_pool_host.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ (tracked in https://github.com/sgl-project/sglang/issues/16565) ⏎  ⏎ HiCache currently falls back to the default host allocation path on MUSA, which uses `cudaHostRegister` when `pin_memory=True`. That path is CUDA-specific and is not appropriate for MUSA. ⏎  ⏎ This change aligns MUSA with the existing NPU behavior by using PyTorch's built-in `pin_memory` allocation path for host buffers. ⏎  ⏎ Error log: ⏎  ⏎ ``` ⏎ [2026-04-20 08:09:39] A …[truncated]

### L1-929e00eeab  (L1, 2026-04-21, sha 929e00eeab0e, PR #22933)
TITLE: [CPU] expand the interface of shared_expert without scaling factor (#22933)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/moe.cpp (+25/-119); sgl-kernel/csrc/cpu/moe.h (+173/-0); sgl-kernel/csrc/cpu/moe_fp8.cpp (+6/-136); sgl-kernel/csrc/cpu/moe_int4.cpp (+10/-176); sgl-kernel/csrc/cpu/moe_int8.cpp (+6/-108); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+3/-3); test/srt/cpu/test_moe.py (+8/-27); test/srt/cpu/test_shared_expert.py (+64/-50); test/srt/cpu/utils.py (+18/-4)
LABELS: sgl-kernel, intel, cpu, run-ci
BODY: ## Motivation ⏎  ⏎ This patch expands `shared_expert` kernel when scaling factor is None, which maps exactly the behavior of MoE when expert equals to one. ⏎  ⏎ Also simplifies moe kernel source codes by extracting common vectorized stubs into `moe.h`. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Major changes in `sgl-kernel/csrc/cpu/moe_xxx.cpp` ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ `python /test/srt/cpu/test_shared_expert.py` ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process …[truncated]

### L1-14ac14287c  (L1, 2026-04-22, sha 14ac14287cf3, PR #23492)
TITLE: [CI] /rerun-stage: auto-include wheel build when PR modifies sgl-kernel/ (#23492)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+66/-25); scripts/ci/utils/slash_command_handler.py (+26/-14)
LABELS: documentation, sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎ `/rerun-stage <stage-name>` on a PR that touches `sgl-kernel/` silently runs the target stage against the **PyPI `sgl-kernel` wheel** instead of the PR's changes, because `sgl-kernel-build-wheels` unconditionally skips in `target_stage` mode. The existing `validate-target-stage` safety-step only fires when `pr_head_sha` is passed to the workflow, which the handler previously omitted for non-fork PRs, so the guard silently passed an …[truncated]

### L1-ad0fc88810  (L1, 2026-04-22, sha ad0fc8881038, PR #22685)
TITLE: [CPU] [Quantization] Add GPTQ/AWQ 4bits quantization support for CPU  (#22685)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/update_config.py (+12/-7); python/sglang/srt/layers/amx_utils.py (+65/-31); python/sglang/srt/layers/linear.py (+3/-0); python/sglang/srt/layers/quantization/__init__.py (+28/-1); python/sglang/srt/layers/quantization/awq.py (+0/-0); python/sglang/srt/layers/quantization/awq_cpu.py (+133/-0); python/sglang/srt/layers/quantization/gptq.py (+10/-1); python/sglang/srt/layers/quantization/gptq_cpu.py (+375/-0); sgl-kernel/csrc/cpu/gemm.h (+10/-0); sgl-kernel/csrc/cpu/gemm_int4.cpp (+105/-25); (+4 more)
LABELS: sgl-kernel, intel, cpu, run-ci
BODY: This PR contains https://github.com/sgl-project/sglang/pull/8225  (and #8225 will be closed) ⏎  ⏎  ⏎ This PR aims to add both AWQ/GPTQ format (4bits) support for CPU platform, mainly including: ⏎ 1. Unpack from gptq format (except varients like marlin/exllama) ⏎ 2. Unpack from awq format ⏎ 3. Repack gptq/awq to cpu amx format, and calls the 4bit sgl-kernels ⏎ 4. Also, TP padding fix for gptq/awq  weight loading (e.g., CPU TP=3/6)

### L1-b9e33d6a5b  (L1, 2026-04-22, sha b9e33d6a5be7, PR #22809)
TITLE: Dual MoE CUDA graph capture for lora/nolora batches (#22809)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+25/-0); python/sglang/srt/lora/lora_moe_runners.py (+16/-6); python/sglang/srt/model_executor/cuda_graph_runner.py (+76/-22); python/sglang/srt/server_args.py (+9/-0)
LABELS: high priority, lora, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎ When LoRA is enabled with a triton MoE backend, capture two sets of CUDA graphs per batch size: one with LoRA kernels recorded and one without. At replay time, batches without active adapters use the faster nolora graph, avoiding LoRA kernel overhead entirely. Controlled by `--record-nolora-graph` (default True), auto-disabled for non-triton MoE backends. ⏎  ⏎ - **server_args.py**: Add `--record-nolora-graph` / `--no-record-nolora-graph …[truncated]

### L1-917d2aa1dc  (L1, 2026-04-22, sha 917d2aa1dc2a, PR #23178)
TITLE: [LoRA] Fix EP + per-expert MoE LoRA illegal memory access (#23178)
SOURCES: path_core, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/csrc/lora/moe_lora_align_kernel.cu (+3/-1); python/sglang/srt/lora/mem_pool.py (+169/-69); test/registered/unit/lora/test_mem_pool_ep_unit.py (+611/-0)
LABELS: lora, run-ci, jit-kernel
DEEP_STUDY: deep-study correctness case sglang:917d2aa1dc: class=memory_safety_oob; symptom=illegal_memory_access; introducing=unknown
BODY: ## Summary ⏎  ⏎ Two related bugs prevent `--tp N --ep M --enable-lora --lora-backend triton --moe-runner-backend triton` from running on any MoE model with per-expert LoRA adapters; both reliably crash in the first forward pass with `CUDA error: an illegal memory access was encountered`. This PR fixes both and adds unit coverage. ⏎  ⏎ ### Bug 1 — `LoRAMemoryPool` sizes per-expert buffers for the global expert count ⏎  ⏎ `FusedMoEWithLoRA` allocates LoR …[truncated]

### L1-c689f774a4  (L1, 2026-04-22, sha c689f774a419, PR #23510)
TITLE: [CI] /rerun-stage: fix workflow-run URL lookup for sgl-kernel PRs (#23510)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: scripts/ci/utils/slash_command_handler.py (+5/-1)
BODY: ## Motivation ⏎  ⏎ PR #23492 introduced an auto-enabled `include_wheel_build` path for non-fork PRs that touch `sgl-kernel/`. That path adds `pr_head_sha` to the `workflow_dispatch` inputs so the build gate's `filter-api` check sees the kernel changes. However, it forgot to update the **local** `pr_head_sha` variable in `handle_rerun_stage` — the variable that is later passed to `find_workflow_run_url()` for the success-comment URL lookup. ⏎  ⏎ As a resu …[truncated]

### L1-887d380ace  (L1, 2026-04-22, sha 887d380acedb, PR #23270)
TITLE: [MUSA] Resolve output garbage in Context Parallel on MusaFlashAttentionBackend (#23270)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: 3rdparty/amd/wheel/sglang/pyproject.toml (+4/-1); python/pyproject_other.toml (+9/-9); python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py (+57/-50); python/sglang/srt/hardware_backend/musa/layers/utils/__init__.py (+0/-0); python/sglang/srt/hardware_backend/musa/layers/utils/cp_utils.py (+57/-0); sgl-kernel/pyproject_musa.toml (+1/-1)
LABELS: dependencies, sgl-kernel, run-ci, mthreads
BODY: ## Motivation ⏎  ⏎ Fix Context Parallel (CP) attention forward extension for the MUSA backend. The original `cp_attn_forward_extend` function from `cp_utils.py` was incompatible with the MUSA FA Attention backend, causing CP workloads to fail on MUSA devices. ⏎  ⏎ ## Modifications ⏎ - **Fix context parallel**:  Modify the original cp_attn_forward_extend function to be able to switch the backend's _current_prefix in order to obtain the correct schedule …[truncated]

### L1-f3b88e080a  (L1, 2026-04-23, sha f3b88e080aeb, PR #23281)
TITLE: chore: bump flashinfer version to 0.6.8.post1 (#23281)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1)
LABELS: high priority, dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the flashinfer version to `0.6.8.post1` across all relevant files. ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎ - python/sglang/srt/utils/common.py ⏎  ⏎ 🤖 Generated with GitHub Actions

### L1-bb962b0046  (L1, 2026-04-23, sha bb962b0046ee, PR #23545)
TITLE: Fix MoE no_combine: skip router weight in down projection (#23545)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+1/-1); python/sglang/srt/layers/moe/token_dispatcher/base.py (+1/-1)
LABELS: high priority, run-ci
BODY: ## Summary ⏎ - When `no_combine` mode is enabled, the down projection kernel should not apply router weights (the combine step handles weighting). Previously only `apply_router_weight_on_input` was checked; now `no_combine` is also checked. ⏎ - Change `BaseDispatcher.quant_config` default from `None` to `{}` to avoid AttributeError when accessed before explicit assignment. ⏎  ⏎ ## Test plan

### L1-000a2525e1  (L1, 2026-04-23, sha 000a2525e196, PR #23585)
TITLE: Move expert_mask_gpu from FusedMoE layer to StandardDispatcher (#23585)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-12); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+19/-7); python/sglang/srt/layers/quantization/fp8.py (+2/-2); python/sglang/srt/layers/quantization/mxfp4.py (+2/-2); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py (+1/-1); python/sglang/srt/layers/quantization/unquant.py (+1/-1)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎ The aiter `expert_mask_gpu` tensor is a pure function of the dispatcher's `local_expert_mapping` and `num_local_experts`, but today it lives on `FusedMoE` and is computed in `FusedMoE.forward_impl`. Having the layer reach into `self.dispatcher.local_expert_mapping` to build state that it then exposes back to quant kernels is a leaky abstraction. This PR moves ownership to `StandardDispatcher`, where the inputs already live. ⏎  ⏎ ## Mod …[truncated]

### L1-d9c72bdd2b  (L1, 2026-04-23, sha d9c72bdd2b8b, PR #23493)
TITLE: Skip unselected experts in flashinfer_trtllm (#23493)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+1/-2)
LABELS: run-ci
BODY: Removed masking of packed tokens with -1 expert ids. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project …[truncated]

### L1-74c2e5bacd  (L1, 2026-04-23, sha 74c2e5bacd0d, PR #17946)
TITLE: [MUSA][8/N] Port CUDA kernels that are compatible with MUSA (#17946)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_softmax, L1.routing.fused_gate
FILES: sgl-kernel/csrc/moe/moe_fused_gate_musa.cu (+840/-0); sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu (+2/-0); sgl-kernel/csrc/allreduce/custom_all_reduce.cuh (+206/-25); sgl-kernel/csrc/common_extension_musa.cc (+246/-12); sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.mu (+529/-0); sgl-kernel/csrc/elementwise/utils.cuh (+7/-0); sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu (+4/-0); sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu (+4/-0); sgl-kernel/csrc/gemm/per_token_group_quant_8bit_v2.cu (+9/-0); sgl-kernel/csrc/kvcacheio/transfer.cu (+2/-2); (+5 more)
LABELS: quant, dependencies, sgl-kernel, run-ci, mthreads
BODY: ### Motivation ⏎  ⏎ This PR continues the ongoing effort (tracked in #16565) to add full support for **Moore Threads GPUs** in SGLang by leveraging **MUSA (Meta-computing Unified System Architecture)** for LLM inference. ⏎  ⏎ The primary goal of this submission is to enable core kernel functionality on MUSA by porting CUDA kernels that are compatible with the MUSA programming model, while keeping the codebase unified across CUDA, ROCm, and MUSA backe …[truncated]

### L1-b35213be11  (L1, 2026-04-23, sha b35213be11c7, PR #22774)
TITLE: [MUSA][16/N] Add MUSA backend support for layers and DeepSeek models (V2/V3/R1) (#22774)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.runner.deep_gemm, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+11/-3); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+7/-4); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+9/-1); python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py (+3/-2); python/sglang/srt/layers/moe/topk.py (+12/-10); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/layers/activation.py (+13/-0); python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py (+14/-3); python/sglang/srt/layers/deep_gemm_wrapper/configurer.py (+11/-2); python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py (+3/-2); (+17 more)
LABELS: quant, deepseek, run-ci, mthreads
BODY: ## Motivation ⏎  ⏎ Enable SGLang to run DeepSeek models on Moore Threads MUSA GPUs. This PR adds MUSA backend support across the inference stack, including layers, quantization, MoE, attention, speculative decoding, and custom op registration. ⏎  ⏎ ## Modifications ⏎  ⏎ **Core Infrastructure:** ⏎  ⏎ - **Custom op registration** (`utils/common.py`): Register custom ops on the `MUSA` dispatch key in `direct_register_custom_op`; extend `get_device_sm()` to  …[truncated]

### L1-54e21bb3a5  (L1, 2026-04-23, sha 54e21bb3a585, PR #23060)
TITLE: [fix] Fix dynamic chunking profiling crash on GLM-5 models (#23060)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/scheduler_pp_mixin.py (+3/-0)
LABELS: run-ci
ISSUES: #23057 [Bug] Dynamic chunking profiling crashes on GLM-5 model   (AttributeError: _is_extend_in_batch)
BODY: ## Motivation ⏎  ⏎ Fixes #23057  ⏎  ⏎ When `--enable-dynamic-chunking` is used with GLM-5  (have DeepEP ), the profiling phase crashes with `AttributeError: _is_extend_in_batch`. This silently disables dynamic chunking and also causes KV cache memory leaks and NCCL timeouts on non-first PP ranks. ⏎  ⏎ ## Modifications ⏎  ⏎ - `python/sglang/srt/managers/scheduler_pp_mixin.py` (`profile_and_init_predictor`): ⏎   - Call `set_is_extend_in_batch(True)` before  …[truncated]

### L1-23e4d381f0  (L1, 2026-04-24, sha 23e4d381f0ae, PR #23528)
TITLE: [CPU] remove RECORD_FUNCTION (#23528)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: sgl-kernel/csrc/cpu/moe.cpp (+0/-5); sgl-kernel/csrc/cpu/topk.cpp (+0/-5); sgl-kernel/csrc/cpu/activation.cpp (+0/-3); sgl-kernel/csrc/cpu/bmm.cpp (+0/-2); sgl-kernel/csrc/cpu/common.h (+0/-1); sgl-kernel/csrc/cpu/conv3d.cpp (+0/-2); sgl-kernel/csrc/cpu/decode.cpp (+0/-5); sgl-kernel/csrc/cpu/extend.cpp (+0/-16); sgl-kernel/csrc/cpu/flash_attn.cpp (+0/-4); sgl-kernel/csrc/cpu/gemm.cpp (+0/-4); (+11 more)
LABELS: sgl-kernel, intel, cpu, run-ci
BODY: ## Motivation ⏎  ⏎ remove `RECORD_FUNCTION` in cpu C++ source files. Since we are now registering kernels with `torch.ops.sgl_kernel.xxx(...)`. No longer need to put `RECORD_FUNCTION` in the C++ kernels. ⏎  ⏎ Otherwise, each kernel will be recorded twice, one from dispatch logic and the other from the internal implementation by `RECORD_FUNCTION`, like: ⏎ ``` ⏎ --------------------------------------------------  ------------  ------------  ------------  …[truncated]

### L1-6d03861476  (L1, 2026-04-24, sha 6d038614760f, PR #23533)
TITLE: support Hy3 preview (#23533)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_py, L1.routing.fused_gate
FILES: python/sglang/jit_kernel/csrc/moe/grouped_topk.cuh (+267/-0); python/sglang/jit_kernel/grouped_topk.py (+89/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_B200,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20,dtype=fp8_w8a8_down.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8_down.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20-3e.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20-3e_down.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20.json (+146/-0); (+15 more)
LABELS: documentation, run-ci, jit-kernel
BODY: ## Summary ⏎  ⏎ Add support for Tencent Hunyuan V3 (Hy3-preview) models in sglang. ⏎  ⏎ ### Components ⏎  ⏎ - **Model**: `python/sglang/srt/models/hunyuan_v3.py` (+ MTP variant `hunyuan_v3_nextn.py`) ⏎ - **Tool-call parser**: `python/sglang/srt/function_call/hunyuan_detector.py` — streaming HYV3 tool parser with `<tool_calls>/<tool_call>/<tool_sep>/<arg_key>/<arg_value>` format, schema-aware type coercion, and char-by-char streaming for string args ⏎ - **Reasoni …[truncated]

### L1-465abadd3c  (L1, 2026-04-24, sha 465abadd3cb0, PR #23682)
TITLE: Add fused moe triton config for Qwen3.5-397B-A17B-FP8 (#23682)
SOURCES: path_config_only, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=512,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Qwen3.5-397B-A17B-FP8 is not tuned on moe kernel and H100 gpu, which leads to a sub-optimal performance of throughput. ⏎  ⏎ This PR runs the script that [tunes Triton MoE Kernel](https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/fused_moe_triton/README.md#1-tuning_fused_moe_tritonpy), benchmarks the decode throughput of tuned moe kernel. ⏎  ⏎ Tuning instruction: ⏎ ```bash ⏎ python benchmark/kernels/fused_moe_triton/ …[truncated]

### L1-fb272d27db  (L1, 2026-04-24, sha fb272d27dbe1, PR #23611)
TITLE: [AMD] Optimize MiniMax-M2.5 - use aiter biased_grouped_topk for sigmoid scoring in MoE routing (#23611)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+18/-7)
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ - For models using sigmoid scoring with correction bias (e.g., MiniMax-M2.5), ⏎   use `aiter.biased_grouped_topk` (ASM kernel) instead of `sgl_kernel.topk_sigmoid` ⏎   on AMD GPUs. ⏎ - The aiter kernel runs at ~6 us/call vs ~9.3 us/call for the sgl_kernel variant, ⏎   reducing MoE routing overhead by ~35% per call. ⏎ - Benchmarked on MI355X with MiniMax-M2.5 FP8 (TP=4, ISL=8192, OSL=1024): ⏎   +2.0% output throughput at conc=64, +2.4%  …[truncated]

### L1-adc59325bc  (L1, 2026-04-24, sha adc59325bc67, PR #23620)
TITLE: [AMD] Optimize MiniMax-M2.5 - enable fused Triton kernel for FP8 KV cache write in aiter decode path (#23620)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+17/-0)
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ - On AMD GPUs with FP8 KV cache (`--kv-cache-dtype fp8_e4m3`) and unified ⏎   attention enabled, the decode KV cache write previously required two ⏎   separate kernel launches: a bf16→fp8 dtype cast (`float8_copy_kernel`) ⏎   followed by a paged store (`store_kvcache`). ⏎ - This PR adds a branch in `AiterAttnBackend.forward_decode` that uses ⏎   `launch_reshape_and_cache_flash` (an existing Triton kernel already used ⏎   for SWA models …[truncated]

### L1-82254bd9c5  (L1, 2026-04-24, sha 82254bd9c5bb, PR #22094)
TITLE: [JIT Kernel] Reland JIT activation (#22094)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.deep_gemm, L1.runner.openai_triton_kernels, L1.runner.marlin, L1.upstream.openai_triton_kernels, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+1/-1); python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+6/-2); python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+2/-1); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+7/-1); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+1/-1); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+3/-1); .claude/skills/add-jit-kernel/SKILL.md (+41/-22); python/sglang/jit_kernel/activation.py (+85/-0); python/sglang/jit_kernel/benchmark/bench_activation.py (+86/-0); python/sglang/jit_kernel/csrc/elementwise/activation.cuh (+135/-0); (+4 more)
LABELS: documentation, lora, run-ci, diffusion, jit-kernel
DEEP_STUDY: deep-study revert record: reland of PR(s) 21766 reason=correctness_or_accuracy
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Fix `num_token = 0` case (see #22078) ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/. …[truncated]

### L1-046c14a3ed  (L1, 2026-04-25, sha 046c14a3edd2, PR #17883)
TITLE: [NPU] Support GGUF quantization for Ascend NPU (dense + MoE) (#17883)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+109/-0); docs/advanced_features/quantization.md (+1/-1); docs/platforms/ascend/ascend_npu_quantization.md (+5/-1); python/sglang/srt/layers/linear.py (+5/-1); python/sglang/srt/layers/quantization/gguf.py (+465/-2); python/sglang/srt/model_loader/loader.py (+2/-0); python/sglang/srt/model_loader/weight_utils.py (+73/-9); python/sglang/srt/models/qwen2_moe.py (+1/-0); python/sglang/srt/models/qwen3_moe.py (+7/-1); python/sglang/test/ascend/test_ascend_utils.py (+6/-0); (+2 more)
LABELS: documentation, quant, npu, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎  ⏎ Enable GGUF quantized models (e.g., Q4_K_M, Q8_0, Q5_K_M) to run on Ascend NPU hardware. GGUF is a popular format for quantized LLM models, and this PR adds native NPU support with optimized performance. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎  ⏎ 1. Add GGUF quantization methods for NPU (python/sglang/srt/layers/quantization/gguf.py) ⏎ - GGUFLinearAscendMethod: Linear layer support with pre-dequantization at load time ⏎ - GGUFMoEAscendMeth …[truncated]

### L1-ba4e9d2ac2  (L1, 2026-04-25, sha ba4e9d2ac20b, PR #23732)
TITLE: Apply should_use_dp_reduce_scatterv guard to remaining MoE models (follow-up to #23731) (#23732)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/bailing_moe.py (+2/-0); python/sglang/srt/models/bailing_moe_linear.py (+7/-1); python/sglang/srt/models/deepseek_v2.py (+3/-0); python/sglang/srt/models/exaone_moe.py (+6/-2); python/sglang/srt/models/glm4_moe.py (+3/-0); python/sglang/srt/models/hunyuan_v3.py (+7/-4); python/sglang/srt/models/llada2.py (+10/-2); python/sglang/srt/models/llama4.py (+6/-1); python/sglang/srt/models/mimo_v2_flash.py (+2/-0); python/sglang/srt/models/minimax_m2.py (+2/-0); (+3 more)
LABELS: high priority, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Follow-up to #23731 (Qwen3 MoE). Supersedes #23431 (same diff for the 12 files there) by also fixing `hunyuan_v3.py`. ⏎  ⏎ PR #22642 introduced `should_use_dp_reduce_scatterv()`, which fuses the post-MoE all-reduce with `dp_scatter` into a single `reduce_scatterv` call inside `LayerCommunicator`. To avoid a double-reduce, the model-side `tensor_model_parallel_all_reduce` (or `moe_*_all_reduce`) on `final_hidden_states` must be skipped  …[truncated]

### L1-049f1bf6fb  (L1, 2026-04-25, sha 049f1bf6fb42, PR #23725)
TITLE: docs(DeepSeek-V4): add GB200 platform to cookbook recipe (#23725)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+6/-2); docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+52/-6)
LABELS: deepseek
BODY: ## Summary ⏎ - Add NVIDIA GB200 (FP4, Grace Blackwell, 4 GPU/node NVL4) as a new hardware platform in the DeepSeek-V4 cookbook ⏎ - Flash (285B): TP=4, single-node — all 4 recipes verified ⏎ - Pro (1.6T): TP=8, 2-node multinode — low-latency, balanced, max-throughput verified; cp unverified; pd-disagg TBD ⏎ - Pro low-latency uses pure TP (no DP-attn/DeepEP), unlike H200 big ⏎ - Pro balanced/max-throughput: `mem-fraction-static=0.78`, `cuda-graph-max-bs=64` ⏎  …[truncated]

### L1-3cfd1561df  (L1, 2026-04-25, sha 3cfd1561df78, PR #23742)
TITLE: docs(DeepSeek-V4): add h200|big verified recipes + tune H200 Pro parameters (#23742)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+22/-8)
LABELS: deepseek
BODY: ## Summary ⏎ - Mark `h200|big|low-latency`, `h200|big|balanced`, `h200|big|max-throughput` as verified in the interactive command generator. ⏎ - Tune H200 Pro (big) parameters based on testing: ⏎   - `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`: 256 → 128 (balanced & max-throughput) ⏎   - `--cuda-graph-max-bs`: 32 → 8, `--max-running-requests`: 64 → 32 (low-latency) ⏎   - `--mem-fraction-static`: 0.82 → 0.88 (low-latency / balanced / max-throughput) ⏎    …[truncated]

### L1-9003f24e2b  (L1, 2026-04-25, sha 9003f24e2b88, PR #23733)
TITLE: chore: bump sglang-kernel version to 0.4.1.post1 (#23733)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); test/registered/4-gpu-models/test_qwen35_hicache.py (+0/-5); test/registered/hicache/test_hicache_storage.py (+0/-5); test/registered/hicache/test_hicache_storage_3fs_backend.py (+0/-2); test/registered/hicache/test_hicache_storage_file_backend.py (+0/-2); test/registered/hicache/test_hicache_storage_mooncake_backend.py (+3/-3); test/registered/hicache/test_hicache_storage_runtime_attach_detach.py (+0/-2); test/registered/hicache/test_hicache_variants.py (+0/-2)
LABELS: dependencies, hicache, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sglang-kernel` version to `0.4.1.post1` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.4.1.post1` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the `sglang-kernel` dependen …[truncated]

### L1-c7878dbb6d  (L1, 2026-04-26, sha c7878dbb6ddf, PR #23707)
TITLE: [MoE] Deprecate act_and_mul_triton; fold filter_expert into JIT silu/gelu_and_mul (#23707)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+14/-19); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py (+0/-106); python/sglang/jit_kernel/activation.py (+50/-8); python/sglang/jit_kernel/benchmark/bench_activation.py (+71/-0); python/sglang/jit_kernel/csrc/elementwise/activation.cuh (+60/-17); python/sglang/jit_kernel/tests/test_activation.py (+80/-0)
LABELS: run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ `act_and_mul_triton` (in `fused_moe_triton_kernels.py`) duplicates `silu_and_mul` / `gelu_and_mul`. The only difference is that it skips rows whose routed expert id is `-1` (the `filter_expert=True` MoE path used under EP). The JIT CUDA `silu_and_mul` / `gelu_and_mul` kernels already exist and are faster — the consolidation removes ~100 lines of Triton and a redundant kernel. ⏎  ⏎ ## Modifications ⏎  ⏎ **JIT activation kernel (CUDA)** ⏎ - `p …[truncated]

### L1-10fd0faccd  (L1, 2026-04-26, sha 10fd0faccd85, PR #19484)
TITLE: [CPU] Add Qwen3.5 model optimization for CPU (#19484)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/moe.cpp (+7/-5); sgl-kernel/csrc/cpu/moe_fp8.cpp (+7/-5); sgl-kernel/csrc/cpu/moe_int8.cpp (+8/-6); python/sglang/srt/configs/update_config.py (+178/-75); python/sglang/srt/layers/attention/fla/fused_norm_gate.py (+19/-11); python/sglang/srt/layers/attention/mamba/mamba.py (+18/-2); python/sglang/srt/layers/attention/vision.py (+2/-1); python/sglang/srt/layers/linear.py (+17/-1); python/sglang/srt/model_executor/cpu_graph_runner.py (+12/-0); python/sglang/srt/model_loader/weight_utils.py (+35/-1); (+10 more)
LABELS: Multi-modal, sgl-kernel, intel, cpu, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: This PR (work with @blzheng ) adds support for Qwen3.5 series with cpu optimized performance, including changes: ⏎  ⏎ 1. Dtype support in fusion of `fused_sigmoid_gating_delta_rule_update` ⏎ 2. Continues support for fusion of `fused_qkvzba_split_reshape_cat_contiguous_cpu ` ⏎ 3. TP cases padding support for both bf16 and fp8 ⏎ 4. Refinements for logging CPU padding logic. ⏎  ⏎ Note that this PR depends on previous CPU mrope kernel support https://github …[truncated]

### L1-da175b964d  (L1, 2026-04-26, sha da175b964d15, PR #23785)
TITLE: chore: update CI test est_time values (#23785)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/4-gpu-models/test_gpt_oss_4gpu.py (+2/-2); test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py (+1/-1); test/registered/4-gpu-models/test_qwen35_fp4_triton.py (+1/-1); test/registered/4-gpu-models/test_qwen3_30b.py (+1/-1); test/registered/4-gpu-models/test_qwen3_next_models.py (+1/-1); test/registered/8-gpu-models/test_deepseek_v32_indexcache.py (+1/-1); test/registered/8-gpu-models/test_deepseek_v3_basic.py (+1/-1); test/registered/8-gpu-models/test_deepseek_v3_mtp.py (+1/-1); test/registered/8-gpu-models/test_dsa_models_basic.py (+1/-1); test/registered/8-gpu-models/test_dsa_models_mtp.py (+1/-1); (+258 more)
LABELS: quant, lora, Multi-modal, deepseek, speculative-decoding, blackwell, npu
BODY: ## Summary ⏎  ⏎ Updates `est_time` values in CI test registration calls based on the 90th percentile of the last 15 successful executions from scheduled PR Test runs on main. ⏎  ⏎ This keeps the LPT load-balancing algorithm accurate for partitioning tests across parallel CI jobs. ⏎  ⏎ ### Significant est_time changes (26 of 269 updates) ⏎  ⏎ | File | Suite | Old (s) | New (s) | Δ | ⏎ | --- | --- | ---: | ---: | ---: | ⏎ | `test_awq.py` | `stage-b-test-1-gpu-large` | …[truncated]

### L1-85376a6119  (L1, 2026-04-26, sha 85376a61190a, PR #23748)
TITLE: refactor(moe): centralize post-experts all-reduce skip predicate (#23748)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/__init__.py (+2/-0); python/sglang/srt/layers/moe/utils.py (+33/-0); python/sglang/srt/models/bailing_moe.py (+5/-8); python/sglang/srt/models/bailing_moe_linear.py (+5/-6); python/sglang/srt/models/deepseek_v2.py (+9/-13); python/sglang/srt/models/exaone_moe.py (+7/-5); python/sglang/srt/models/glm4_moe.py (+9/-13); python/sglang/srt/models/hunyuan_v3.py (+13/-7); python/sglang/srt/models/llada2.py (+4/-5); python/sglang/srt/models/llama4.py (+4/-5); (+7 more)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ The post-experts EP and TP all-reduce paths in MoE models gate on the same growing list of "downstream will absorb the all-reduce" predicates: ⏎  ⏎ - `should_allreduce_fusion` — `LayerCommunicator` will fuse with next layer's residual all-reduce ⏎ - `use_reduce_scatter` — `LayerCommunicator`'s post-attention scatter does reduce-scatter ⏎ - `should_use_dp_reduce_scatterv()` — DP reduce-scatterv combine path (#22642) ⏎ - `should_use_flashinfer …[truncated]

### L1-32c3513816  (L1, 2026-04-27, sha 32c3513816b0, PR #20918)
TITLE: [NPU] Support MTP for Qwen3.5 (#20918)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+2/-0); python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py (+425/-0); python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py (+280/-0); python/sglang/srt/hardware_backend/npu/memory_pool_npu.py (+24/-0); python/sglang/srt/layers/attention/attention_registry.py (+17/-5); python/sglang/srt/layers/attention/mamba/mamba2_metadata.py (+1/-0); python/sglang/srt/layers/layernorm.py (+6/-3); python/sglang/srt/mem_cache/memory_pool.py (+9/-0); python/sglang/srt/models/qwen3_5_mtp.py (+23/-1); python/sglang/srt/models/qwen3_next_mtp.py (+22/-1)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ Adapt the MTP (Multi-Token Prediction) speculative decoding feature for the Qwen3.5 model on the Ascend NPU platform, fix inference errors, and ensure stable and efficient model operation. ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Add a dedicated GDN attention backend tailored for Ascend NPU, designed to address hardware-specific compatibility and performance needs; ⏎ 2. Complete end-to-end MTP speculative decoding adaptation for the Qwen3.5 mode …[truncated]

### L1-5f47cae1a0  (L1, 2026-04-27, sha 5f47cae1a081, PR #23719)
TITLE: add H100 configs for GLM-4.7-Flash (#23719)
SOURCES: path_config_only, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=65,N=1536,device_name=NVIDIA_H100_80GB_HBM3.json (+154/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=65,N=1536,device_name=NVIDIA_H100_80GB_HBM3_down.json (+154/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎ SGLang currently falls back to the default Triton fused MoE config for `GLM-4.7-Flash` on H100 for the missing shapes below: ⏎  ⏎ - `E=65,N=1536,device_name=NVIDIA_H100_80GB_HBM3.json` ⏎ - `E=65,N=1536,device_name=NVIDIA_H100_80GB_HBM3_down.json` ⏎  ⏎ That fallback showed up in the server logs and profiler triage during the GLM-4.7-Flash H100 optimization loop, and it left measurable latency on the table. ⏎  ⏎ This PR adds the missing H …[truncated]

### L1-9a53ab3d6d  (L1, 2026-04-28, sha 9a53ab3d6d09, PR #15771)
TITLE: [6/N] (Elastic EP) Recover failed ranks (#15771)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/advanced_features/server_arguments.md (+1/-0); python/sglang/srt/distributed/parallel_state.py (+32/-6); python/sglang/srt/elastic_ep/elastic_ep.py (+131/-1); python/sglang/srt/eplb/eplb_manager.py (+3/-0); python/sglang/srt/eplb/expert_location.py (+46/-0); python/sglang/srt/managers/data_parallel_controller.py (+27/-1); python/sglang/srt/model_executor/model_runner.py (+84/-1); python/sglang/srt/server_args.py (+13/-0)
LABELS: documentation, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ As a follow-up to #11657, this PR enables SGLang to dynamically add back previously failed processes, recovering the optimal throughput. ⏎  ⏎ The core idea is as follows. When a node fails, the system admin can relaunch the node with an additional flag `--elastic-ep-rejoin`. Meanwhile, the remaining healthy processes continue serving ongoing inference requests and periodically poll the status of the relaunched process. Once the …[truncated]

### L1-f34222da1b  (L1, 2026-04-28, sha f34222da1b22, PR #23851)
TITLE: [Docs] add cookbook for MiMo-V2.5 family (#23851)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx (+626/-0); docs_new/cookbook/autoregressive/intro.mdx (+1/-1); docs_new/docs.json (+1/-0); docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx (+397/-0)
BODY: ## Summary ⏎  ⏎ Adds a unified cookbook for [XiaomiMiMo/MiMo-V2.5-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro) (1.02T MoE, text-only) and [XiaomiMiMo/MiMo-V2.5](https://huggingface.co/XiaomiMiMo/MiMo-V2.5) (310B MoE, native omnimodal) under `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx`. ⏎  ⏎ - **Interactive deployment generator** (variant × hardware × recipe) as a JSX snippet — H200 / H100 / B200 / GB300, low-latency / balanced / max-t …[truncated]

### L1-1a55646dcd  (L1, 2026-04-28, sha 1a55646dcdf0, PR #23808)
TITLE: [Feature] Xiaomi MiMo-V2.5-Pro day0 support (#23808)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+20/-5); python/sglang/srt/models/mimo_v2.py (+27/-8); python/sglang/srt/models/mimo_v2_nextn.py (+21/-6); python/sglang/srt/server_args.py (+11/-4); python/sglang/srt/utils/common.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ MiMo-V2.5-Pro is a Mixture-of-Experts (MoE) language model with 1.02T total parameters and 42B active parameters. It utilizes hybrid attention architecture and 3-layers Multi-Token Prediction (MTP) described in MiMo-V2-Flash. The context length is up to 1M tokens. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎ ### Benchmark command ⏎  ⏎ ``` ⏎ python3 -m sglang.bench_serving \ ⏎   --backend sgl …[truncated]

### L1-ddcacaf1bd  (L1, 2026-04-28, sha ddcacaf1bd4e, PR #23874)
TITLE: Fix failing `test_nvidia_nemotron_3_nano` by fixing `test_grouped_topk` (#23874)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: python/sglang/jit_kernel/csrc/moe/grouped_topk.cuh (+11/-18); python/sglang/jit_kernel/tests/test_grouped_topk.py (+210/-0); python/sglang/srt/models/nemotron_h.py (+2/-0); test/registered/models/test_nvidia_nemotron_3_nano.py (+0/-1)
LABELS: documentation, high priority, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ Fix the Nemotron-3-Nano FP8 failure exposed after enabling the JIT grouped-topk path. ⏎  ⏎ Nemotron now selects the JIT `grouped_topk` path because its router config matches the kernel constraints: one expert group, `topk_group=1`, 128 routed experts, `topk=6`, and correction bias. That exposed two separate issues. ⏎  ⏎ First, the grouped-topk kernel did not order negative choice scores correctly. Nemotron's correction bias can make  …[truncated]

### L1-4e1ef6b3cf  (L1, 2026-04-28, sha 4e1ef6b3cf9b, PR #23943)
TITLE: [Docs] Add single-node H200 DeepSeek-V4-Pro low-latency recipe (#23943)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+32/-0)
LABELS: deepseek
BODY: ## Summary ⏎ - Add a TP=8 single-node variant (Marlin backend) for H200 + DeepSeek-V4-Pro low-latency cookbook recipe ⏎ - The existing multi-node (2 nodes, TP=16, DP-Attn + DeepEP) command is preserved below the new single-node command ⏎  ⏎ ## Test plan ⏎  ⏎ 🤖 Generated with [Claude Code](https://claude.com/claude-code)

### L1-8327270c72  (L1, 2026-04-29, sha 8327270c7263, PR #21321)
TITLE: [Kernel] Support FlashInfer TRTLLM-Gen fused MoE for non-gated FP4 & FP8 (Nemotron) (#21321)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/flashinfer_trtllm_moe.py (+21/-0); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+274/-31); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py (+6/-0); python/sglang/srt/layers/quantization/fp8.py (+7/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+14/-14); python/sglang/srt/layers/quantization/utils.py (+8/-7); python/sglang/srt/server_args.py (+9/-1); python/sglang/srt/models/nemotron_h.py (+2/-0)
LABELS: quant, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎  ⏎ - Add support for non-gated (relu2) activation in FlashInfer TRTLLM-Gen FP4 and FP8 MoE kernels, enabling NemotronH-120B models (`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` and `-FP8`) ⏎ - Add FP4/FP8/MXFP8 weight alignment padding for TP > 1 (non-gated needs 128 alignment) - will integrate MXFP8 soon.  ⏎ - Set `routing_method_type=DeepSeekV3` for NemotronH MoE routing ⏎  ⏎  ⏎ ## Evaluation ⏎  ⏎ ### GSM8K 4-shot accuracy (lm_eval, B20 …[truncated]

### L1-d9270b8c6a  (L1, 2026-04-29, sha d9270b8c6ad3, PR #24004)
TITLE: fix(moe): relocate orphan tuned configs after #23019 (#24004)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_B200,dtype=fp8_w8a8.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20,dtype=fp8_w8a8.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20,dtype=fp8_w8a8_down.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8_down.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20-3e.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20-3e_down.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=192,N=192,device_name=NVIDIA_H20_down.json (+0/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=32,N=1792,device_name=NVIDIA_B200.json (+0/-0); (+25 more)
LABELS: documentation
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎ After #23019 moved the MoE config loader and the configs/ tree from `fused_moe_triton/` to `moe_runner/triton_utils/`, two later PRs unknowingly added 33 tuned-config JSONs to the OLD path: ⏎  ⏎ - #22791 (LFM2)        — 24 files (E=32/64, H100/B200/MI325X) ⏎ - #23533 (Hy3 preview) —  9 files (E=192,N=192 incl. _down, ⏎                                     H20/H20-3e/B200) ⏎  ⏎ The runtime loader anchors its search via ⏎ os.path.dirname(o …[truncated]

### L1-13afe8acdf  (L1, 2026-04-29, sha 13afe8acdff3, PR #23619)
TITLE: [codex] Enable Qwen3-Next MoE all-reduce fusion (#23619)
SOURCES: subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen3_next.py (+46/-23)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ - Route Qwen3-Next decoder MLP outputs through `LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer`, matching the existing MoE all-reduce fusion pattern used by related Qwen MoE model code. ⏎ - Mark fused-path MoE outputs with `_sglang_needs_allreduce_fusion` so the next layer can use all-reduce + RMSNorm fusion when supported. ⏎ - Keep the PR code change scoped to `python/sglang/srt/models/qwen3_next.py`; the earlier unit test a …[truncated]

### L1-08699bb1b2  (L1, 2026-04-29, sha 08699bb1b2d3, PR #23815)
TITLE: [NPU] Fix DeepEP LL dispatch BF16 flag and skip triton kernel on NPU for Qwen3.5 (#23815)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+5/-1); python/sglang/srt/models/qwen3_5.py (+7/-1)
LABELS: run-ci
BODY: ## Motivation ⏎ Fix two bugs that block NPU inference for Qwen3.5 MoE models with DeepEP backend. ⏎  ⏎ ## Modifications ⏎  ⏎ ### 1. DeepEP low-latency dispatch ignores `SGLANG_DEEPEP_BF16_DISPATCH` (`deepep.py`) ⏎ The normal dispatch path respects `SGLANG_DEEPEP_BF16_DISPATCH` to skip FP8 quantization, but the low-latency dispatch path (`_dispatch_core`) did not check this flag. It always quantized hidden states to INT8/FP8 during all-to-all communicat …[truncated]

### L1-0ac23cffac  (L1, 2026-04-29, sha 0ac23cffacc2, PR #12771)
TITLE: Add intel_xpu as backend for GptOssForCausalLM, enabled for bf16 models with torch native backend (#12771)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_native.py (+39/-1); python/sglang/srt/server_args.py (+15/-0)
LABELS: ready-to-merge, intel, xpu, run-ci
BODY: ## Motivation ⏎  ⏎ Enable GptOssForCausalLM on Intel XPU for bf16 dtype. ⏎  ⏎ ## Modifications ⏎  ⏎ Add intel_xpu as supported backend for GptOssForCausalLM model. ⏎ Update forward_moe_native function to add following changes required for GPT-OSS MoE architetecture: ⏎  - Bias addition support in linear layers of MoE ⏎  - swiglu (with alpha and limit) activation support ⏎  ⏎ ## Accuracy Tests ⏎ To verify the correctness of the changes I ran the following test …[truncated]

### L1-1376761841  (L1, 2026-04-29, sha 13767618412a, PR #24069)
TITLE: fix(moe): repair dead import in fused_moe_native after MoE refactor (#24069)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_native.py (+4/-2); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+2/-2)
LABELS: run-ci
BODY: ## Summary ⏎ - `fused_moe_native.py:10` still imported `swiglu_with_alpha_and_limit` from `sglang.srt.layers.moe.fused_moe_triton.fused_moe`, but that symbol was renamed to `_swiglu_gpt_oss_sigmoid_alpha` in #18084 and the whole module was deleted in #23019 — so any `torch.compile` + MoE path with `gemm1_alpha` set raised `ModuleNotFoundError` on server startup. ⏎ - Promote the helper to public (`swiglu_gpt_oss_sigmoid_alpha`) in `moe_runner/triton_u …[truncated]

### L1-6c7b242181  (L1, 2026-04-29, sha 6c7b2421816c, PR #23936)
TITLE: mimo v2.5 pro sglang-jax cookbook (#23936)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx (+36/-0); docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx (+78/-16)
BODY: ## Motivation ⏎  ⏎ Add a TPU deployment recipe for **MiMo-V2.5-Pro** to the cookbook, served via [sgl-jax](https://github.com/sgl-project/sglang-jax). This documents the verified v7x (16 chips / 4 nodes) and v6e (64 chips / 16 nodes) topologies and lets users generate the launch command from the existing variant/hardware panel instead of copy-pasting from a separate sglang-jax doc. ⏎  ⏎ ## Modifications ⏎  ⏎ - `docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2 …[truncated]

### L1-c8c1c9261d  (L1, 2026-04-29, sha c8c1c9261d72, PR #23594)
TITLE: LoRA support for qwen3.5 and nemotron3 (#23594)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/lora/lora_moe_runners.py (+14/-9); python/sglang/srt/lora/backend/ascend_backend.py (+3/-3); python/sglang/srt/lora/backend/chunked_backend.py (+4/-3); python/sglang/srt/lora/backend/torch_backend.py (+2/-2); python/sglang/srt/lora/backend/triton_backend.py (+5/-3); python/sglang/srt/lora/layers.py (+123/-66); python/sglang/srt/lora/lora.py (+161/-10); python/sglang/srt/lora/lora_manager.py (+2/-0); python/sglang/srt/lora/mem_pool.py (+15/-4); python/sglang/srt/lora/triton_ops/chunked_sgmv_expand.py (+8/-7); (+11 more)
LABELS: lora, npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Adding missing LoRA support for Qwen3.5 and Nemotron3. In particular supporting ungated mlp/moe in lora-moe, and mamba2/GDN projections. ⏎ This also fixes an issue in ReplicatedLinearWithLoRA with 2 slices (kimi/deepseek's `fused_qkv_a_proj_with_mqa)` causing wrong results/nans when the loaded-lora-rank<max_lora_rank ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ### Overview ⏎ - Generalize `run_qkv_lora` to arbitrary num_slices in all backends, t …[truncated]

### L1-3f7c95d6cc  (L1, 2026-04-29, sha 3f7c95d6cccb, PR #23833)
TITLE: [JIT Kernel][1/2]Migrate MXFP8 Group GEMM & Quant into JIT (#23833)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/csrc/moe/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cuh (+461/-0); python/sglang/jit_kernel/csrc/moe/expert_specialization/es_sm100_mxfp8_blockscaled_moe_group_gemm.cuh (+217/-0); python/sglang/jit_kernel/csrc/moe/expert_specialization/es_sm100_mxfp8_blockscaled_moe_group_gemm_functor.cuh (+64/-0); python/sglang/jit_kernel/csrc/moe/expert_specialization/es_sm100_mxfp8_blockscaled_moe_group_gemm_traits.cuh (+123/-0); python/sglang/jit_kernel/benchmark/bench_mxfp8_moe.py (+290/-0); python/sglang/jit_kernel/mxfp8.py (+136/-0); python/sglang/jit_kernel/tests/test_mxfp8_moe.py (+153/-0)
LABELS: quant, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎ Migrate MXFP8 Group GEMM & Quant into JIT. **Note that this is not a simple transplant.** I used MoE Group GEMM instead of the normal Group GEMM([Reference CUTLASS Example](https://github.com/NVIDIA/cutlass/blob/main/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_blockscaled_rcgrouped.cu)). This implementation reduces TMA descriptor updates. In addition, we used a 2SM strategy to reduce data copying and improve performance in …[truncated]

### L1-692979a8d9  (L1, 2026-04-29, sha 692979a8d981, PR #23929)
TITLE: [AMD] Support sdma path for moriep (#23929)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+21/-5)
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ This patch is to enable sdma path for moriep through env `MORI_ENABLE_SDMA`.  ⏎ cc @Duyi-Wang @HaiShaw  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Added `enable_sdma` parameter to `init_mori_op()`, which activates MoRI's low-latency mode when SDMA is enabled. ⏎ - Added `MORI_ENABLE_SDMA` env var (default `false`) in `_MoriEPDispatcherImplBase` to toggle the SDMA path at runtime. ⏎ - In `_MoriEPDispatcherImplNormal`, when SDMA is enabled: ⏎   - **D …[truncated]

### L1-71e89e9003  (L1, 2026-04-30, sha 71e89e9003f5, PR #23654)
TITLE: [MUSA][19/N] Support qwen series models (#23654)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py, L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/musa/kernels/topk.py (+300/-0); python/sglang/srt/layers/moe/topk.py (+25/-3); 3rdparty/amd/wheel/sglang/pyproject.toml (+1/-1); python/pyproject_other.toml (+1/-1); python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py (+30/-12); python/sglang/srt/layers/attention/vision.py (+12/-2); python/sglang/srt/layers/quantization/fp8_kernel.py (+0/-5); python/sglang/srt/layers/quantization/fp8_utils.py (+13/-7); python/sglang/srt/layers/rotary_embedding/base.py (+2/-1); python/sglang/srt/layers/utils/multi_platform.py (+1/-4); (+10 more)
LABELS: dependencies, Multi-modal, sgl-kernel, run-ci, mthreads
BODY: ## Motivation ⏎  ⏎ This PR is part of the MUSA (Moore Threads GPU) backend support series (19/N). The goal is to enable Qwen series models on the MUSA platform by: ⏎ 1. Adding MUSA-specific MoE fused gate and top-k kernels. ⏎ 2. Fixing compatibility issues in vision attention, FP8 quantization, and multi-platform dispatch for MUSA. ⏎  ⏎ ## Modifications ⏎  ⏎ ### 1. MUSA Top-K Kernels (New File) ⏎ - **`python/sglang/srt/hardware_backend/musa/kernels/topk.p …[truncated]

### L1-577dbc4ab9  (L1, 2026-04-30, sha 577dbc4ab976, PR #21126)
TITLE: [4/N] Quantization Refactor: AWQ schemes and Kernel call and weight init split (#21126)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/gpu/quantization/awq_kernels.py (+255/-0); python/sglang/srt/hardware_backend/npu/quantization/awq_kernels.py (+156/-0); python/sglang/srt/layers/linear.py (+0/-3); python/sglang/srt/layers/quantization/__init__.py (+2/-3); python/sglang/srt/layers/quantization/auto_round.py (+5/-2); python/sglang/srt/layers/quantization/awq.py (+0/-966); python/sglang/srt/layers/quantization/awq/__init__.py (+32/-0); python/sglang/srt/layers/quantization/awq/awq.py (+484/-0); python/sglang/srt/layers/quantization/awq/awq_triton.py (+0/-0); python/sglang/srt/layers/quantization/awq/schemes/__init__.py (+19/-0); (+10 more)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎ Add schemes to awq instead of storing all classes in a single file, and split kernel call and weight init. Follow up to https://github.com/sgl-project/sglang/pull/17503. ⏎ Images and motivation for this PR can be viewed in our roadmap: https://github.com/sgl-project/sglang/issues/15194. ⏎  ⏎  ⏎ ## Modifications ⏎  Refactored AWQ to align with the scheme-based quantization structure used by modelslim and compressed_tensors. ⏎  ⏎   Moved  …[truncated]

### L1-108bfd8b6a  (L1, 2026-04-30, sha 108bfd8b6a0d, PR #23597)
TITLE: [MoE] Add Aiter MoE runner backend and purge aiter.fused_moe from quant methods (#23597)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:production-kernel-provenance, body_keyword
ARTIFACT_HINTS: L1.runner.framework, L1.runner.aiter, L1.upstream.aiter_moe
FILES: python/sglang/srt/layers/moe/moe_runner/aiter.py (+94/-0); python/sglang/srt/layers/moe/moe_runner/runner.py (+5/-0); python/sglang/srt/layers/moe/utils.py (+4/-0); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py (+33/-35); python/sglang/srt/layers/quantization/fp8.py (+46/-69); python/sglang/srt/layers/quantization/mxfp4.py (+59/-59); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py (+25/-29); python/sglang/srt/layers/quantization/quark_int4fp8_moe.py (+25/-22); python/sglang/srt/layers/quantization/unquant.py (+24/-37); python/sglang/srt/server_args.py (+1/-0)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎ Part of the MoE refactor roadmap tracked in #8715. ⏎  ⏎ Today, ``aiter.fused_moe`` is called directly from seven quantization methods (``fp8.py``, ``mxfp4.py``, ``unquant.py``, ``quark_int4fp8_moe.py``, ``quark/schemes/quark_w4a4_mxfp4_moe.py``, ``compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py``). Each call site duplicates roughly the same ~15 arguments, re-implements ``apply_router_weight_on_input`` and ``topk_weights`` …[truncated]

### L1-8975479f87  (L1, 2026-04-30, sha 8975479f87dc, PR #24171)
TITLE: [LoRA][MOE] Fix EP correctness in MoE LoRA slicing and virtual-experts kernels (#24171)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/lora/lora_manager.py (+6/-0); python/sglang/srt/lora/mem_pool.py (+26/-2); python/sglang/srt/lora/triton_ops/virtual_experts.py (+42/-12); test/registered/lora/test_virtual_experts_kernels.py (+259/-0); test/registered/unit/lora/test_mem_pool_ep_unit.py (+119/-0)
LABELS: lora, run-ci
BODY: ## Motivation ⏎  ⏎ Two correctness gaps in the LoRA MoE path under expert parallelism, plus small cleanups uncovered while tracing them: ⏎  ⏎ 1. **`tp_size > ep_size > 1` crashes during adapter load.** `LoRAMemoryPool.load_lora_weight_to_buffer` passes the outer `tp_rank` to `slice_moe_lora_{a,b}_weights`, but per-expert MoE weights are sharded along `moe_tp_size = tp/ep/dp`. On `tp=4 ep=2`, outer ranks 2,3 slice past `intermediate_size` and `load_lo …[truncated]

### L1-651af06a0b  (L1, 2026-05-01, sha 651af06a0b5e, PR #23811)
TITLE: [Feature] Xiaomi MiMo-V2.5 day0 support (#23811)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+36/-23); python/sglang/srt/layers/attention/vision.py (+61/-17); python/sglang/srt/layers/linear.py (+1/-1); python/sglang/srt/managers/io_struct.py (+2/-2); python/sglang/srt/managers/mm_utils.py (+38/-9); python/sglang/srt/models/mimo_audio.py (+1350/-0); python/sglang/srt/models/mimo_v2.py (+222/-13); python/sglang/srt/models/mimo_v2_nextn.py (+12/-7); python/sglang/srt/models/mimo_vl.py (+507/-0); python/sglang/srt/multimodal/processors/mimo_v2.py (+2039/-0); (+6 more)
LABELS: high priority, Multi-modal, new-model, run-ci
BODY: ## Summary ⏎  ⏎ Adds day-0 support for `XiaomiMiMo/MiMo-V2.5` in SGLang. ⏎  ⏎ - Registers `MiMoV2ForCausalLM` and the `MiMoV2MTP` draft model while keeping the legacy `MiMoV2FlashForCausalLM` name loadable. ⏎ - Adds MiMo-V2 multimodal model pieces for image, video, and audio via the checkpoint's `vision_config` / `audio_config`. ⏎ - Adds the MiMo-V2 multimodal processor for image, video, audio, and video+audio request inputs. ⏎ - Supports the FP8 fused-QKV che …[truncated]

### L1-88bb5dffe4  (L1, 2026-05-02, sha 88bb5dffe496, PR #21247)
TITLE: [Dependency] Upgrade to Torch 2.11.0 (#21247)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+4/-21); scripts/ci/cuda/ci_install_deepep.sh (+4/-1); .github/workflows/pr-test-amd-rocm720.yml (+20/-0); .github/workflows/pr-test-amd.yml (+20/-0); .github/workflows/pr-test-jit-kernel.yml (+63/-4); .github/workflows/pr-test-multimodal-gen.yml (+24/-4); .github/workflows/pr-test-npu.yml (+56/-26); .github/workflows/pr-test.yml (+13/-10); python/pyproject_other.toml (+2/-2); (+11 more)
LABELS: documentation, high priority, amd, dependencies, Multi-modal, deepseek, sgl-kernel, npu, run-ci, diffusion
BODY: ## Motivation ⏎  ⏎ github.com/pytorch/pytorch/releases/tag/v2.11.0 ⏎  ⏎ ## Modifications ⏎  ⏎ github.com/sgl-project/sglang/pull/18862 ⏎  ⏎ <!-- Detail the changes made in this pull request. --

### L1-53df43d0a3  (L1, 2026-05-03, sha 53df43d0a3ab, PR #24325)
TITLE: rerun-test: route deepep h200 suite to deepep runner (#24325)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/rerun-test.yml (+3/-0); scripts/ci/utils/slash_command_handler.py (+1/-1)
BODY: Match the runner that pr-test.yml uses for stage-c-test-deepep-8-gpu-h200 (8-gpu-h200-deepep), so /rerun-test on this suite lands on the same hardware as the original PR job.

### L1-00d620b77d  (L1, 2026-05-03, sha 00d620b77d1b, PR #24328)
TITLE: introduce arg_groups/ with nemotron_h hook (#24328)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/__init__.py (+0/-0); python/sglang/srt/arg_groups/nemotron_h_hook.py (+51/-0); python/sglang/srt/server_args.py (+4/-39)
BODY: Seed the per-arch hook framework called out in RFC #20481, with a minimal example: extract the \`NemotronHForCausalLM\` model-specific block out of \`server_args.py\` into \`arg_groups/nemotron_h_hook.py\`. ⏎  ⏎ This is a flat, simplified version of the RFC structure — \`arg_groups/<model>_hook.py\` instead of nested \`arg_groups/handlers/model_specific/<family>.py\`. The 3-level RFC layout is overkill before the broader mixin / handler split lands;  …[truncated]

### L1-52b4609789  (L1, 2026-05-04, sha 52b46097894c, PR #23593)
TITLE: [Docker] Prep for torch 2.11: cu129 fix, image validator, dep cleanup (#23593)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+21/-34); docker/diffusion.Dockerfile (+0/-104); .github/workflows/release-docker-dev.yml (+6/-4); .github/workflows/release-docker-runtime.yml (+3/-3); .github/workflows/release-docker.yml (+4/-4); .github/workflows/trivy-scan-dev.yml (+1/-1); scripts/ci/utils/docker_build_metadata_args.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ Torch 2.11 makes cu130 wheels the PyPI default, which surfaces two install-time bugs in the cu129 Docker image build path. This PR fixes those, adds a post-build validator to gate future CUDA-variant regressions, and cleans up NVIDIA package overrides that torch 2.11 now ships at equal or newer versions — mirroring the CI-script cleanup already done in #21247. ⏎  ⏎ Companion to **#21247** (torch 2.11 upgrade), which handles `python/pyp …[truncated]

### L1-de08c804c5  (L1, 2026-05-04, sha de08c804c58b, PR #24348)
TITLE: Add release workflow for sgl-deep-gemm wheels (#24348)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/sgl-deep-gemm.Dockerfile (+35/-0); .github/workflows/release-whl-deepgemm.yml (+196/-0); scripts/build_sgl_deep_gemm.sh (+66/-0); scripts/rename_sgl_deep_gemm_whl.sh (+66/-0); scripts/update_deepgemm_whl_index.py (+45/-0)
BODY: ## Summary ⏎ - Add `.github/workflows/release-whl-deepgemm.yml` to build and publish `sgl-deep-gemm` wheels. Inputs: `version`, `target` (`all`/`cu129`/`cu130`, default `all`), `branch` (default `release-0426`). ⏎ - Build matrix `[cu129, cu130] × [x86_64, aarch64]` on the same self-hosted nodes used by `release-whl-kernel.yml`. Each job checks out `sgl-project/DeepGEMM` at the given branch, writes the input version into `sgl_deep_gemm/VERSION`, then  …[truncated]

### L1-62a4df0067  (L1, 2026-05-04, sha 62a4df006799, PR #24234)
TITLE: [docker] Fix silently-masked cubin download failure; skip prebuilt cubins on aarch64 (#24234)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+18/-9)
BODY: ## Motivation ⏎  ⏎ Fix two compounded issues in `framework_final` of `docker/Dockerfile` that have been silently breaking aarch64 nightly Docker builds since #22160 (2026-04-09): ⏎  ⏎ 1. **Silent-failure bug.** The cubin-download retry block ends with `... || true` (intended only to guard the trailing `find` cleanup), but bash's left-associative `&&`/`||` precedence makes that swallow the entire chain — including the `[ "$success" = "1" ]` fail-fast chec …[truncated]

### L1-8ffd39e140  (L1, 2026-05-04, sha 8ffd39e1402d, PR #24374)
TITLE: [CI] Exclude flaky h20 stage from check-stage-health root cause set (#24374)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/actions/check-stage-health/action.yml (+6/-0)
BODY: ## Summary ⏎  ⏎ - The `8-gpu-h20` runners regularly enter jobs with leftover GPU memory from a prior run that cleanup can't reclaim, so `stage-c-test-8-gpu-h20` fails at `Install dependencies` before any test runs (see [run 25316228022](https://github.com/sgl-project/sglang/actions/runs/25316228022/job/74268805394) — `ERROR: memory >=10%` across all 8 GPUs, 8 unkillable PIDs). ⏎ - When that happens, every other stage-c job (h200, b200, deepep, …) fast- …[truncated]

### L1-0f283a515a  (L1, 2026-05-04, sha 0f283a515a50, PR #24369)
TITLE: [Docker] fix: install nixl stub alongside nixl-cuXX binary (#24369)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2)
BODY: ## Motivation ⏎  ⏎ Follow-up to #23593. That PR dropped the `nixl` Python stub package because its wheel METADATA carries an unconditional `Requires-Dist: nixl-cu12>=1.0.1` — installing plain `nixl` would pull cu12 binaries onto cu13 images. The cleanup went one step too far: sglang code still imports `from nixl._api import nixl_agent, nixl_agent_config`, and that import path is owned **only** by the stub package, not by `nixl-cu12` / `nixl-cu13`. ⏎  ⏎ # …[truncated]

### L1-4743cf6051  (L1, 2026-05-04, sha 4743cf6051f5, PR #24384)
TITLE: misc: add marlin to moe runner choices; drop dead env var doc (#24384)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+1/-0); docs_new/docs/references/environment_variables.mdx (+0/-5)
LABELS: documentation, run-ci
BODY: Two unrelated tiny cleanups bundled: ⏎  ⏎ 1. `server_args.py`: `MOE_RUNNER_BACKEND_CHOICES` was missing `marlin` even though `MoeRunnerBackend.MARLIN` enum exists since #14554. `docs_new/.../deepseek-v4-deployment.jsx` already instructs users to pass `--moe-runner-backend marlin`, but argparse rejects it. ⏎ 2. `docs_new/.../environment_variables.mdx`: drop the `SGLANG_DISABLE_REQUEST_LOGGING` row -- the env var has no reader anywhere in the codebase.

### L1-2b769d37a4  (L1, 2026-05-04, sha 2b769d37a41d, PR #24246)
TITLE: (2/n - prefill optimize)perf(lora): remove GPU-CPU sync barrier (.item()) in MoE LoRA path and remove duplicate code (#24246)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/lora/lora_moe_runners.py (+4/-125)
LABELS: high priority, lora, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ based on https://github.com/sgl-project/sglang/pull/24007 ⏎  ⏎ - Remove `.item()` call in `_add_lora_gate_up_delta` that caused unnecessary GPU-CPU synchronization (`cudaStreamSynchronize`) on every MoE layer forward pass during eager execution ⏎ - Clean up duplicate code blocks (duplicate function definitions, dataclass fields, variable assignments) ⏎  ⏎ ### Root Cause ⏎  ⏎ The original code used `.any().item()` on GPU tensors to check wh …[truncated]

### L1-e6f252e9b8  (L1, 2026-05-05, sha e6f252e9b8d7, PR #24156)
TITLE: Cache FlashInfer autotune configs (#24156)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/model_runner.py (+43/-5)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎ - Persist FlashInfer autotune configs across launches by passing a cache path to `flashinfer.autotune`. ⏎ - Scope cache files by FlashInfer version, CUDA arch, model/backend key, and TP/PP/DP rank. ⏎ - Store autotune configs under `SGLANG_CACHE_DIR/flashinfer/autotune/...`. ⏎  ⏎ ## Testing ⏎ - `uvx ruff==0.15.1 check --select=F401,F821 python/sglang/srt/model_executor/model_runner.py` ⏎ - `uvx isort==7.0.0 --check-only python/sglang/srt/model_exec …[truncated]

### L1-6279aee716  (L1, 2026-05-05, sha 6279aee7163a, PR #24367)
TITLE: [docs] Update B300 Pro cookbook with accuracy-verified serving configs (#24367)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+108/-11)
LABELS: documentation, deepseek
BODY: ## Summary ⏎ - Update DeepSeek-V4 cookbook JSX command generator for B200|big (= B300 Pro) with configs verified via SimpleQA-Verified (1,000 factual QA samples). All 3 configs match the official 57.9% Pass@1 benchmark within ±2%. ⏎ - **Low Latency**: chunked-prefill-size 4096→8192, mem-fraction 0.88→0.90, add swa-full-tokens-ratio 0.1 ⏎ - **Balanced**: switch MoE backend from deepep to flashinfer_mxfp4, add chunked-prefill-size 32768, mem-fraction 0.8 …[truncated]

### L1-c4c0376fcb  (L1, 2026-05-05, sha c4c0376fcb24, PR #24403)
TITLE: consolidate routed-experts capturer onto reusable base (#24403)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py, L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+5/-4); python/sglang/srt/layers/moe/routed_experts_capturer.py (+49/-272); python/sglang/srt/layers/moe/topk.py (+5/-4); python/sglang/srt/layers/topk_capturer_base.py (+178/-0); python/sglang/srt/managers/scheduler_output_processor_mixin.py (+4/-1); python/sglang/srt/managers/utils.py (+2/-2); python/sglang/srt/model_executor/model_runner.py (+9/-8)
LABELS: high priority, run-ci
BODY: Pure refactor of the routed-experts capturer; prep for an upcoming indexer-topk capturer (stacked PR) that reuses the same base. ⏎  ⏎ - Drop `RoutedExpertsCapturer` ABC + `_RoutedExpertsCapturerNoop` — `create()` returns `Optional[RoutedExpertsCapturer]`, callers use `is not None` checks. ⏎ - Extract `BaseTopkCapturer` / `BaseHostCache` / `BaseDeviceCache` / `TopkCaptureOutput` to `python/sglang/srt/layers/topk_capturer_base.py`. ⏎ - `RoutedExpertsCaptur …[truncated]

### L1-08d4c2072b  (L1, 2026-05-05, sha 08d4c2072b50, PR #24450)
TITLE: move topk capturers to srt/state_capturer/ (#24450)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py, L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+1/-1); python/sglang/srt/layers/moe/topk.py (+1/-1); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+3/-3); python/sglang/srt/managers/scheduler_output_processor_mixin.py (+4/-4); python/sglang/srt/managers/utils.py (+1/-1); python/sglang/srt/model_executor/model_runner.py (+11/-11); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+3/-3); python/sglang/srt/state_capturer/__init__.py (+0/-0); python/sglang/srt/state_capturer/base.py (+0/-0); python/sglang/srt/state_capturer/indexer_topk.py (+1/-1); (+3 more)
LABELS: run-ci
BODY: Move the three capturer files out of `layers/` into a new top-level dir `srt/state_capturer/` — capture is observability infrastructure (side-effect observers tapping into producer outputs), not a layer (compute component). Aligns with sglang's existing observer modules (`eplb/`, `metrics/`, `debug_utils/`) which all sit at `srt/` root. ⏎  ⏎ ## Moves ⏎  ⏎ ``` ⏎ layers/topk_capturer_base.py            -> state_capturer/base.py ⏎ layers/moe/routed_experts_capt …[truncated]

### L1-6764155914  (L1, 2026-05-05, sha 6764155914ef, PR #24457)
TITLE: chore: bump sgl-kernel version to 0.4.2.post1 (#24457)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+0/-54); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_musa.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
LABELS: amd, dependencies, sgl-kernel, mthreads
BODY: ## Summary ⏎  ⏎ This PR bumps the sgl-kernel version to `0.4.2.post1` across all relevant files. ⏎  ⏎ ## Files Updated ⏎ - sgl-kernel/pyproject.toml ⏎ - sgl-kernel/pyproject_cpu.toml ⏎ - sgl-kernel/pyproject_musa.toml ⏎ - sgl-kernel/pyproject_rocm.toml ⏎ - sgl-kernel/python/sgl_kernel/version.py ⏎  ⏎ 🤖 Generated with GitHub Actions

### L1-b2420d72ff  (L1, 2026-05-05, sha b2420d72ff3e, PR #16859)
TITLE: [RL] DeepEP support for `--enable-return-routed-experts` (#16859)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/state_capturer/routed_experts.py (+31/-1); test/registered/rl/test_return_routed_experts.py (+35/-27)
LABELS: run-ci
BODY: Linked issue: https://github.com/THUDM/slime/issues/1316 ⏎  ⏎ ## Summary ⏎ - Add DeepEP a2a support to `RoutedExpertsCapturer` (rebased onto `srt/state_capturer/` after #24450) ⏎ - Each attn-TP rank only sees its scattered slice of `topk_ids` under DeepEP, so all-gather across attn-TP at capture time before staging into `device_cache` ⏎ - Alternative late-gather impl in #17892 by @ocss884 (gather at D2H sync); this PR keeps the early-gather approach ⏎ - Repl …[truncated]
