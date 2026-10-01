### L1-cc25f9df50  (L1, 2026-01-11, sha cc25f9df50e1, PR #16835)
TITLE: Update est_time for stage-b-test-small-1-gpu tests (#16835)
SOURCES: body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: test/registered/attention/test_torch_native_attention_backend.py (+1/-1); test/registered/backends/test_torch_compile.py (+1/-1); test/registered/core/test_deterministic.py (+1/-1); test/registered/core/test_gpt_oss_1gpu.py (+1/-1); test/registered/cuda_graph/test_piecewise_cuda_graph_small_1_gpu.py (+1/-1); test/registered/dllm/test_llada2_mini.py (+1/-1); test/registered/hicache/test_hicache_variants.py (+1/-1); test/registered/lora/test_lora_update.py (+1/-1); test/registered/mla/test_flashmla.py (+1/-1); test/registered/mla/test_mla_int8_deepseek_v3.py (+1/-1); (+14 more)
LABELS: lora, Multi-modal, deepseek, speculative-decoding, hicache, npu, run-ci
BODY: ## Summary ⏎ - Updated `est_time` for 24 tests in `stage-b-test-small-1-gpu` suite based on actual elapsed times from 3 CI runs ⏎ - Only updated tests where difference between estimated and actual time was >50 seconds ⏎  ⏎ **CI runs analyzed:** ⏎ - https://github.com/sgl-project/sglang/actions/runs/20851379702 ⏎ - https://github.com/sgl-project/sglang/actions/runs/20861026466 ⏎ - https://github.com/sgl-project/sglang/actions/runs/20842927997 ⏎  ⏎ **Tests with unde …[truncated]

### L1-9f5cd80a8d  (L1, 2026-01-12, sha 9f5cd80a8d21, PR #16019)
TITLE: Re-introduce the unit test of test_mooncake_ep_small (#16019)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/mooncake.py (+10/-16); python/sglang/srt/eplb/eplb_algorithms/__init__.py (+1/-1); test/srt/ep/test_mooncake_ep_small.py (+5/-2); test/srt/run_suite.py (+1/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ The unit test was temporarily disabled in #12969 because Mooncake EP was not fully compatible with torch 2.9. ⏎  ⏎ We now add it back due to the Mooncake EP library has upgraded. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Moved test_mooncake_ep_small.py from manual/ to srt/ ⏎ - A couple of fixes that weren't discovered until the CI has been enabled. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ The unit tests should pass. ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  …[truncated]

### L1-075c5a5789  (L1, 2026-01-13, sha 075c5a5789e7, PR #16982)
TITLE: Code clean up for fp8 quantization (#16982)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/bench_one_batch_server.py (+5/-5); python/sglang/srt/constrained/xgrammar_backend.py (+5/-5); python/sglang/srt/layers/linear.py (+2/-3); python/sglang/srt/layers/parameter.py (+47/-2); python/sglang/srt/layers/quantization/fp8.py (+198/-172); python/sglang/srt/layers/quantization/fp8_kernel.py (+3/-7); python/sglang/srt/layers/quantization/fp8_utils.py (+15/-36); python/sglang/srt/metrics/collector.py (+13/-1); python/sglang/srt/utils/common.py (+16/-1)
LABELS: high priority, sgl-kernel, run-ci
BODY: 

### L1-5af84c8af5  (L1, 2026-01-14, sha 5af84c8af554, PR #7392)
TITLE: [AMD][Quantization] Add `int4fp8_moe` online quantization on ROCm (#7392)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/advanced_features/quantization.md (+8/-0); python/sglang/srt/configs/model_config.py (+1/-0); python/sglang/srt/layers/int4fp8_utils.py (+73/-0); python/sglang/srt/layers/linear.py (+1/-0); python/sglang/srt/layers/quantization/__init__.py (+2/-0); python/sglang/srt/layers/quantization/quark_int4fp8_moe.py (+443/-0); python/sglang/srt/model_loader/utils.py (+7/-1); python/sglang/srt/model_loader/weight_utils.py (+16/-14); python/sglang/srt/server_args.py (+1/-0); python/sglang/srt/utils/common.py (+6/-0); (+2 more)
LABELS: documentation, quant, run-ci
BODY: As per title, this PR supersedes https://github.com/sgl-project/sglang/pull/6238. ⏎  ⏎ This PR implements loading MOE models checkpoints in high-precision (fp16, bf16), quantizing online the MOE experts to int4 and the attention projections to float8. ⏎  ⏎ During inference the int4 moe weights are upcasted to float8 in order to use fp8 math. ⏎  ⏎ Runnable with ⏎ ```bash ⏎ export MODEL_ID="/large_models/mistralai_Mixtral-8x7B-Instruct-v0.1" ⏎ python3 -m sg …[truncated]

### L1-424a380077  (L1, 2026-01-15, sha 424a38007735, PR #14504)
TITLE: [NPU] NPU quantization refactoring & more quantization formats support (#14504)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.ep.layer
FILES: python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+136/-451); python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-4); .github/CODEOWNERS (+1/-0); docs/platforms/ascend_npu_deepseek_example.md (+0/-5); docs/platforms/ascend_npu_quantization.md (+21/-0); python/sglang/srt/configs/model_config.py (+27/-2); python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py (+51/-129); python/sglang/srt/layers/quantization/__init__.py (+2/-9); python/sglang/srt/layers/quantization/awq.py (+4/-4); python/sglang/srt/layers/quantization/base_config.py (+0/-4); (+20 more)
LABELS: documentation, quant, deepseek, hicache, npu, run-ci
ISSUES: #15391 [Bug] [Ascend] [AWQ] AWQ quantization RuntimeError with aclnnAddRmsNorm operator on ascend backend
BODY: ## Motivation ⏎  ⏎ Related to https://github.com/sgl-project/sglang/issues/14424 (you can found class diagramm [here](https://github.com/sgl-project/sglang/issues/14424)). Follows https://github.com/sgl-project/sglang/issues/13664 ⏎  ⏎ Continuation of the refactoring started in https://github.com/sgl-project/sglang/pull/13359 and feature supporting started in https://github.com/sgl-project/sglang/pull/11984. To simplify the support of various quantiz …[truncated]

### L1-000ad42225  (L1, 2026-01-15, sha 000ad4222595, PR #17075)
TITLE: chore: bump sgl-kernel version to 0.3.21 (#17075)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+4/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: high priority, dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.21` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.21` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are updated accord …[truncated]

### L1-69822c7271  (L1, 2026-01-15, sha 69822c727123, PR #17176)
TITLE: Disable unit-test-deepep-8-gpu (#17176)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+49/-48); test/srt/run_suite.py (+6/-3)
LABELS: quant, run-ci
BODY: ## Summary ⏎  ⏎ Disables `unit-test-deepep-8-gpu` job and `test_deepep_large.py` due to IBGDA/cudaHostRegister environment issues on the 8-GPU runner. ⏎  ⏎ The 4-GPU DeepEP tests (`unit-test-deepep-4-gpu`) provide sufficient coverage. ⏎  ⏎ ## CI Failure ⏎  ⏎ https://github.com/sgl-project/sglang/actions/runs/21041416426/job/60528556805 ⏎  ⏎ ## Related ⏎  ⏎ Temporarily bypass the issue in #17175

### L1-146b5fcc84  (L1, 2026-01-15, sha 146b5fcc8410, PR #16826)
TITLE: [CI] Reorganize stage-b 1-GPU tests for 5090 compatibility (#16826)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+15/-58); scripts/ci/slash_command_handler.py (+0/-1); test/registered/attention/test_create_kvindices.py (+0/-1); test/registered/attention/test_mamba_unittest.py (+0/-1); test/registered/attention/test_radix_attention.py (+0/-1); test/registered/attention/test_radix_cache_unit.py (+0/-1); test/registered/attention/test_swa_unittest.py (+1/-1); test/registered/attention/test_torch_native_attention_backend.py (+0/-1); test/registered/attention/test_triton_attention_backend.py (+1/-1); test/registered/attention/test_triton_attention_kernels.py (+1/-1); (+126 more)
LABELS: high priority, quant, amd, lora, Multi-modal, deepseek, speculative-decoding, hicache, npu, run-ci
BODY: ## Summary ⏎  ⏎ This PR reorganizes the 1-GPU test suites to fully integrate RTX 5090 runners. ⏎  ⏎ ### Test Distribution ⏎  ⏎ | Suite | Runner | GPU | Tests | Description | ⏎ |-------|--------|-----|-------|-------------| ⏎ | `stage-b-test-small-1-gpu` | `1-gpu-5090` | RTX 5090 (32GB, SM120) | **67** | Tests that pass on 5090 | ⏎ | `stage-b-test-large-1-gpu` | `1-gpu-runner` | H200 (80GB, SM90) | **56** | Tests incompatible with 5090 | ⏎  ⏎ ### Changes ⏎  ⏎ 1. **`stage-b …[truncated]

### L1-7f8353aff3  (L1, 2026-01-16, sha 7f8353aff3d9, PR #16925)
TITLE: [BugFix]: Fix `sglang.bench_one_batch` (#16925)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/bench_one_batch.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ On the main branch, model final logits from the following commands do not match. ⏎  ⏎ - `python3 scripts/playground/reference_hf.py --model-path meta-llama/Llama-3.1-8B --dtype bfloat16` ⏎  ⏎ - `python3 -m sglang.bench_one_batch --correctness-test --model meta-llama/Llama-3.1-8B`  ⏎  ⏎ The reason is one line of code casts `req.prefix_indices` from `torch.int64` to `torch.int32`, which is not what the `write_req_to_token_pool_triton` ke …[truncated]

### L1-daea51385d  (L1, 2026-01-16, sha daea51385d2e, PR #13216)
TITLE: Add AFMoE model implementation (#13216)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/supported_models/generative_models.md (+1/-0); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/afmoe.py (+102/-0); python/sglang/srt/models/afmoe.py (+633/-0); python/sglang/srt/utils/hf_transformers_utils.py (+2/-0)
LABELS: documentation, run-ci
BODY: ## Purpose ⏎ This PR adds architecture implementation of upcoming Arcee AI AFMoE (trinity) models. ⏎  ⏎ # Test Plan ⏎ The model is not public yet, verified serving of AFMoE across configs.

### L1-82a1b645ba  (L1, 2026-01-17, sha 82a1b645bad3, PR #17133)
TITLE: [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab (#17133)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json (+164/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json (+164/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+2/-2); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py (+337/-215)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎ 1. Performance tuning based on the code after fused moe swapab #16723. The optimal configuration of fused MoE changes when swapab is taken into consideration. ⏎  ⏎ 2. Optimize the tuning script `tuning_fused_moe_triton_sep.py`:  CUDA Graph is used to encapsulate the kernel to avoid inaccurate performance evaluation in small-token scenarios. In addition, a total of 100 sample data are divided into 10 iterations with 10 data executed p …[truncated]

### L1-a04675892e  (L1, 2026-01-17, sha a04675892eba, PR #15551)
TITLE: Update flashinfer to 0.6.1 (#15551)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_trtllm, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-2); python/pyproject.toml (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-1); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+0/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+0/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+0/-1); python/sglang/srt/layers/quantization/mxfp4.py (+0/-1); python/sglang/srt/utils/common.py (+1/-0); scripts/ci/ci_install_dependency.sh (+1/-1)
LABELS: documentation, high priority, quant, amd, dependencies, lora, Multi-modal, deepseek, speculative-decoding, hicache
BODY: ## Motivation ⏎ flashinfer -> 0.6.1 ⏎ flashinfer-cubin -> 0.6.1 ⏎  ⏎ PRs dependent on this upgrade: ⏎ #15546 ⏎ #15422 ⏎ #15514 ⏎ #15347 ⏎ #14668 ⏎ #16232 ⏎ #16279 ⏎ #16892 ⏎ #16534 ⏎ #12787 ⏎ ... ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d36f6f043c  (L1, 2026-01-17, sha d36f6f043ce8, PR #16824)
TITLE: [Fix] `flashinfer_trtllm` `intermediate_size` assertion with Qwen3 + TP=8 (#16824)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+22/-2); python/sglang/srt/server_args.py (+14/-25); python/sglang/srt/utils/common.py (+8/-0)
LABELS: ready-to-merge, run-ci
DEEP_STUDY: deep-study correctness case sglang:d36f6f043c: class=shape_alignment_edge; symptom=crash_or_exception; introducing=unknown
BODY: ## Motivation ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path Qwen/Qwen3-Next-80B-A3B-Instruct --trust-remote-code --tp 8 ⏎ ``` ⏎  ⏎ Since the  ⏎  ⏎ ``` ⏎ { ⏎   "architectures": [ ⏎     "Qwen3NextForCausalLM" ⏎ ... ⏎   "moe_intermediate_size": 512, ⏎ ... ⏎ ``` ⏎  ⏎ We will run into this issue when TP = 8 (512 / 8 = 64): ⏎ ``` ⏎ 026-01-08 16:44:05 TP2] Scheduler hit an exception: Traceback (most recent call last): ⏎   File "/sgl-workspace/sglang/python/sglang …[truncated]

### L1-d3eafc7357  (L1, 2026-01-18, sha d3eafc7357f3, PR #17235)
TITLE: [GLM 4.7] Add RTX 6000 Pro aka sm120 (#17235)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=161,N=192,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition,dtype=fp8_w8a8,per_channel_quant=True.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=161,N=192,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition.json (+146/-0)
LABELS: quant
BODY: ## Motivation ⏎  ⏎ GLM 4.7 fails to run on RTX Pro 6000 because it uses the default Hopper kernels which allocates more SMEM than is available on sm120. ⏎  ⏎ ## Modifications ⏎  ⏎ Add moe json configs. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-reque …[truncated]

### L1-93433726eb  (L1, 2026-01-19, sha 93433726eb72, PR #16649)
TITLE: [Refactor] Split out deepseek v2 weight loader function into mixin (#16649)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py (+657/-0); python/sglang/srt/models/deepseek_common/utils.py (+53/-1); python/sglang/srt/models/deepseek_nextn.py (+2/-5); python/sglang/srt/models/deepseek_v2.py (+9/-594)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎ DeepseekV2 code has been developed fast and a lot of historical code and be more orgranized, including the weight loading part. ⏎  ⏎ Issue related: https://github.com/sgl-project/sglang/issues/16291 ⏎  ⏎ ## Modifications ⏎ This PR just **moves** the weight loader function into a mixin with some documentations. ⏎  ⏎ The further refactors of splitting the weight loading internal will come after this PR get merged. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Se …[truncated]

### L1-2ea02f0642  (L1, 2026-01-19, sha 2ea02f06420d, PR #17116)
TITLE: [AMD CI] Migrate and Add More Testcases (#17116)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-amd.yml (+81/-47); test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py (+1/-1); test/registered/amd/test_deepseek_v3_basic.py (+84/-0); test/registered/amd/test_deepseek_v3_mtp.py (+116/-0); test/registered/attention/test_wave_attention_kernels.py (+1/-1); test/registered/core/test_deterministic.py (+5/-1); test/registered/hicache/test_hicache_storage_3fs_backend.py (+2/-1); test/registered/hicache/test_hicache_storage_file_backend.py (+2/-1); test/registered/openai_server/basic/test_serving_rerank.py (+1/-1); test/registered/quant/test_awq_dequant.py (+1/-1); (+9 more)
LABELS: quant, amd, lora, deepseek, hicache, run-ci
BODY: ## Motivation ⏎ Cleans up and reorganizes AMD CI test infrastructure. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ - Renamed suite: `stage-a-test-1` → `stage-a-test-1-amd` (5 files) ⏎ - Update suite name: `stage-b-test-small-1-gpu` → `stage-b-test-small-1-gpu-amd` (2 files) ⏎ - Add 3 test cases :  ⏎  `test/registered/core/test_deterministic.py` ⏎  `test/registered/hicache/test_hicache_storage_file_backend.py` ⏎  `test_hicache_storage_3fs_backend.py` ⏎ - Removed 2 legacy jo …[truncated]

### L1-84c8390514  (L1, 2026-01-19, sha 84c8390514d5, PR #15347)
TITLE: Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate` (#15347)
SOURCES: path_core, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+66/-4); test/registered/kernels/test_fused_topk_deepseek.py (+97/-0); test/srt/test_deepseek_v3_mtp.py (+2/-8)
LABELS: performance, deepseek, run-ci, nvidia
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ flashinfer has an optimized routing kernel for DeepSeek V3:  https://github.com/flashinfer-ai/flashinfer/pull/2099 ⏎ The API was renamed to `fused_topk_deepseek` here:  https://github.com/flashinfer-ai/flashinfer/pull/2181  ⏎  ⏎ ## Modifications ⏎  ⏎ Replace the call to `moe_fused_gate` with `fused_topk_deepseek`. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Server Command: ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path nvidia/DeepSeek-R1-0528-FP4-V …[truncated]

### L1-9fe56cd0fb  (L1, 2026-01-19, sha 9fe56cd0fb77, PR #17325)
TITLE: Fix kernel selection in biased_grouped_topk_gpu (#17325)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+0/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ The ```moe_fused_gate``` has supported the case ```num_fused_shared_experts > 0``` since PR https://github.com/sgl-project/sglang/pull/5440, but PR https://github.com/sgl-project/sglang/pull/15347 incorrectly added a check for ```num_fused_shared_experts == 0```, which leads to performance degradation for some models (such as DeepSeek v3.1). ⏎  ⏎ <img width="813" height="284" alt="image" src="https://github.com/user-attachments/ass …[truncated]

### L1-5c02217746  (L1, 2026-01-19, sha 5c0221774633, PR #17158)
TITLE: Inclusion of nvfp4 blockscale in EPLB Rebalance (#17158)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+0/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ #13715 exclude block_scale in the EPLB rebalance. This PR add it back since it's expert dependent. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ benchmark: ⏎   type: "gpqa" ⏎   num_examples: 198 ⏎   repeat: 8 ⏎   num_threads: 128 ⏎   max_tokens: 65536 ⏎  ⏎ Without EPLB:  ⏎ ``` ⏎ Repeat: 8, mean: 0.785 ⏎ Scores: ['0.803', '0.793', '0.783', '0.778', '0.783', '0.798', '0.768', '0.778'] ⏎ ==================== ⏎ Writing report to /tmp/gpqa_deep …[truncated]

### L1-8fb45523f3  (L1, 2026-01-19, sha 8fb45523f339, PR #15325)
TITLE: feat: support bitsandbytes quantization algorithm (#15325)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/__init__.py (+2/-0); python/sglang/srt/layers/quantization/bitsandbytes.py (+620/-0)
BODY: bitsandbytes was removed by this PR (https://github.com/sgl-project/sglang/pull/12268) to reduce the dependency on vLLM.  ⏎  ⏎ For now(e.g. sglang0.5.6), if you run ⏎ `python3 -m sglang.bench_one_batch --model unsloth/llama-3-8b-bnb-4bit --load-format bitsandbytes`  ⏎ u will get an error: ⏎ `ValueError: Unknown quantization method: bitsandbytes. Must be one of ['fp8', 'blockwise_int8', 'modelopt', 'modelopt_fp8', 'modelopt_fp4', 'w8a8_int8', 'w8a8_fp8 …[truncated]

### L1-db2425a00b  (L1, 2026-01-20, sha db2425a00b03, PR #17409)
TITLE: [Fix]: correctly fetch ds32 config  in tuning_fused_moe_triton (#17409)
SOURCES: subject_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/common_utils.py (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎ fix "deepseek_v32" KeyError ⏎ ``` ⏎ Traceback (most recent call last): ⏎   File "/opt/conda/lib/python3.10/site-packages/transformers/models/auto/configuration_auto.py", line 1360, in from_pretrained ⏎     config_class = CONFIG_MAPPING[config_dict["model_type"]] ⏎   File "/opt/conda/lib/python3.10/site-packages/transformers/models/auto/configuration_auto.py", line 1048, in __getitem__ ⏎     raise KeyError(key) ⏎ KeyError: 'deepseek_v3 …[truncated]

### L1-76b06bee03  (L1, 2026-01-20, sha 76b06bee03e8, PR #17247)
TITLE: [New Model] GLM4.7-Flash (#17247)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+19/-9); python/sglang/srt/entrypoints/openai/serving_chat.py (+2/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py (+7/-2); python/sglang/srt/models/glm4_moe.py (+3/-1); python/sglang/srt/models/glm4_moe_lite.py (+808/-0); python/sglang/srt/server_args.py (+3/-0)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎ Coauthored with Xinyuan ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎  ⏎ ## Usage  ⏎ Install sglang with this PR: ⏎ ``` ⏎ uv pip install sglang==0.3.2.dev9039+pr-17247.g90c446848 \                                                  ⏎       --extra-index-url https://sgl-project.github.io/whl/pr/ ⏎ ``` ⏎ Need to install transformers on this commit: https://github.com/huggingface/transformers/commit/76732b4e7120808ff989edbd16401f61fa6a0afa ⏎  ⏎ ```sh ⏎ uv pip ins …[truncated]

### L1-6ea491e439  (L1, 2026-01-21, sha 6ea491e4392d, PR #17289)
TITLE: Overlap shared experts with deepep dispatch for single batch overlap on Blackwell (#17289)
SOURCES: path_integration+keyword, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+46/-0); docs/references/environment_variables.md (+1/-0); python/sglang/srt/batch_overlap/single_batch_overlap.py (+5/-1); python/sglang/srt/environ.py (+1/-0)
LABELS: documentation, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ cherry-picked from https://github.com/sgl-project/sglang/tree/gb200-spec ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-proj …[truncated]

### L1-95f59c13fd  (L1, 2026-01-21, sha 95f59c13fd08, PR #17493)
TITLE: [Chore] include all jit files in building packages (#17493)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-10); python/pyproject_cpu.toml (+2/-10); python/pyproject_other.toml (+2/-10); python/pyproject_xpu.toml (+2/-10)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ To include `jit_kernel/include/sgl_kernel/impl/`, and `jit_kernel/include/sgl_kernel/gemm/`. ⏎  ⏎ I'm not very familiar with packaging, need help in reviewing this cc @merrymercy @Kangyan-Zhou @zhyncs . Before this PR, I found that the original packaging code already include all files uner `jit_kernel`. However, this is **unexpected**, since `jit_kernel/include/sgl_kernel/*.cuh` shouldn't match files like `jit_kernel/include/sgl_ …[truncated]

### L1-2ff0880a0e  (L1, 2026-01-21, sha 2ff0880a0ed1, PR #17166)
TITLE: [Fix] GLM 4.7 + NVFP4 + MTP (#17166)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+17/-8); python/sglang/srt/model_loader/loader.py (+14/-0); python/sglang/srt/model_loader/weight_utils.py (+38/-0); python/sglang/srt/models/glm4_moe.py (+5/-1); python/sglang/srt/server_args.py (+22/-0); python/sglang/srt/utils/common.py (+18/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ A few issues reported by @ynwang007 @JustinTong0323 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ 1. We will face an error in loading the draft model with NVFP4 due to some inheritance relationship. Instead of deleting the detection from `modelopt -> modelopt_fp4` in https://github.com/sgl-project/sglang/pull/16581 I think this detection might be a better method. ⏎ 2. Hardcode a fix for GLM 4.7 FP4 checkpoint + MTP: Essentially, `safetensors.index.j …[truncated]

### L1-2b2f317383  (L1, 2026-01-22, sha 2b2f317383a4, PR #17532)
TITLE: fix gpt-oss launch failure with piecewise cuda graph (#17532)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/gpt_oss.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎ recent NPU support introduced a small bug which make the gpt-oss fail to launch with piecewise cuda graph. ⏎ This one line change is to fix the bug ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ `python3 -m sglang.launch_server --model-path /shared/public/elr-models/openai/gpt-oss-120b-new/ --trust-remote-code --tp 4 --reasoning-parser gpt-oss --enable-piecewise-cuda-graph` ⏎  ⏎ Before: ⏎ ``` ⏎     combine_input = self.run_moe_core( ⏎   …[truncated]

### L1-d7dd0b8832  (L1, 2026-01-23, sha d7dd0b883229, PR #17438)
TITLE: Re-enable unit-test-deepep-8-gpu and unit-test-backend-4-gpu-gb200 (#17438)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+100/-102); test/srt/run_suite.py (+3/-6)
LABELS: high priority
BODY: ## Summary ⏎ - Re-enable `unit-test-deepep-8-gpu` (8-GPU H200 runner fixed, #17175) ⏎ - Re-enable `unit-test-backend-4-gpu-gb200` (GB200 runner repaired, #17367) ⏎  ⏎ Successful GB200 run: https://github.com/sgl-project/sglang/actions/runs/21139104876/job/60851136785

### L1-628ab5d57b  (L1, 2026-01-23, sha 628ab5d57b33, PR #17053)
TITLE: [MUSA][2/N] sgl-kernel build (#17053)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .gitignore (+14/-0); docs/platforms/mthreads_gpu.md (+25/-0); sgl-kernel/csrc/common_extension_musa.cc (+51/-0); sgl-kernel/pyproject_musa.toml (+33/-0); sgl-kernel/setup_musa.py (+205/-0)
LABELS: documentation, dependencies, sgl-kernel, run-ci, mthreads
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is the second in a series of pull requests (tracked in https://github.com/sgl-project/sglang/issues/16565) to add full support for [Moore Threads](https://en.mthreads.com/) GPUs, leveraging MUSA (Meta-computing Unified System Architecture) to accelerate LLM inference. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Following the AMD approach, we add a small set of MUSA-specific files: ⏎  ⏎ 1. `pyproject_musa.toml`: used later during the Docker  …[truncated]

### L1-2c2c4e446b  (L1, 2026-01-24, sha 2c2c4e446b99, PR #14668)
TITLE: [NVIDIA] Add flashinfer all-to-all MOE dispatcher (#14668)
SOURCES: path_core, symbol_pickaxe, release_notes, corpus:production-kernel-provenance
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+9/-0); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+6/-0); python/sglang/srt/layers/moe/token_dispatcher/base.py (+19/-0); python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+263/-0); python/sglang/srt/layers/moe/token_dispatcher/flashinfer_utils.py (+47/-0); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+1/-0); python/sglang/srt/layers/moe/utils.py (+5/-0); docs/advanced_features/expert_parallelism.md (+1/-0); docs/references/environment_variables.md (+1/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+23/-14); (+4 more)
LABELS: documentation, quant, deepseek, run-ci, nvidia, Grace Blackwell
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ~Draft PR since https://github.com/flashinfer-ai/flashinfer/pull/2102 is not yet merged into flashinfer.~ ⏎  ⏎ https://github.com/flashinfer-ai/flashinfer/pull/2102 is now merged. ⏎  ⏎ ## Motivation ⏎  ⏎ This PR integrates the latest TRT-LLM moe all-to-all kernels into sglang (AKA nvlink one sided allotall or mnnvlthroughput alltoall): ⏎  ⏎ > NVLINK one-sided comm AllToAll strategy for throughput scenarios. ⏎ >  ⏎ > This implementation utilizes symmetric m …[truncated]

### L1-d1042e0d62  (L1, 2026-01-24, sha d1042e0d62d2, PR #17584)
TITLE: [Refactore] [CI] Remove redundant CI test runs step 2 (#17584)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: scripts/ci/cuda/ci_install_deepep.sh (+1/-1); .github/workflows/ci-coverage-overview.yml (+5/-5); .github/workflows/execute-notebook.yml (+1/-1); .github/workflows/nightly-test-amd.yml (+69/-69); .github/workflows/nightly-test-npu.yml (+6/-6); .github/workflows/nightly-test-nvidia.yml (+15/-15); .github/workflows/pr-benchmark-rust.yml (+2/-2); .github/workflows/pr-test-amd.yml (+44/-43); .github/workflows/pr-test-npu.yml (+5/-5); .github/workflows/pr-test-rust.yml (+3/-3); (+26 more)
LABELS: amd, npu, run-ci
ISSUES: #17583 [CI] Redundant CI test runs step 2
DEEP_STUDY: deep-study: this PR was reverted by PR 17701 (partial_revert, reason=unstated)
BODY: ## Motivation ⏎  ⏎ related to https://github.com/sgl-project/sglang/issues/17583 ⏎  ⏎ ## Modifications ⏎  ⏎ CI scripts moved to corresponding folder ⏎ PR file filter check only related hardware CI scripts ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https …[truncated]

### L1-1a19b3987d  (L1, 2026-01-25, sha 1a19b3987dca, PR #15679)
TITLE: [Model] Add Ernie4.5 VL model support (#15679)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/supported_models/multimodal_language_models.md (+1/-0); python/sglang/srt/configs/model_config.py (+1/-0); python/sglang/srt/layers/rotary_embedding.py (+256/-0); python/sglang/srt/models/ernie45_moe_vl.py (+552/-0); python/sglang/srt/models/ernie45_vl.py (+845/-0); python/sglang/srt/multimodal/processors/ernie45_vl.py (+417/-0)
LABELS: documentation, Multi-modal, run-ci
BODY: ## Motivation ⏎  ⏎ Add Baidu Ernie4.5 VL model support ⏎  ⏎ ## Modifications ⏎  ⏎ `ernie45_moe_vl.py` the text backbone ⏎ `ernie45_vl.py` the vit ⏎ `processors/ernie45_vl.py` the processor ⏎ `rotary_embedding.py::Ernie4_5_VLRotaryEmbedding` the 3d_rope (hwhwhw...ttt..) ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ```bash ⏎ python3 -m sglang.launch_server --model-path baidu/ERNIE-4.5-VL-28B-A3B-PT \ ⏎  --served-model-name ERNIE-45-VL-28B \  ⏎  --port 8301 \  ⏎  --trust-remote-code …[truncated]

### L1-a883906a24  (L1, 2026-01-26, sha a883906a248a, PR #16892)
TITLE: Support mxint4 flashinfer_trtllm moe gemm (#16892)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-0); python/sglang/srt/layers/moe/moe_runner/base.py (+6/-1); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+13/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+341/-5)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ We introduced an MXINT4 flashinfer_trtllm MoE GEMM kernel on SM100. We report both accuracy and performance results of the Kimi-K2-Thinking model on B200.  Compared with Marlin MoE, the new kernel delivers higher prefill throughput, and improves decoding throughput for most batch sizes. This kernel requires Flashinfer v0.6.0. @ispobock @BBuf @FlamingoPg @AniZpZ  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ # launch server ⏎ python3 -m sglang.laun …[truncated]

### L1-f6f1b6d000  (L1, 2026-01-26, sha f6f1b6d000b6, PR #17700)
TITLE: Bump FI version (#17700)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎ This PR bumps FlashInfer version, to incorporate latest fix to Mamba's `selective_scan_update` kernel ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other re …[truncated]

### L1-7106f6c8e1  (L1, 2026-01-26, sha 7106f6c8e150, PR #17582)
TITLE: [GLM-OCR] Support GLM-OCR Model (#17582)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/supported_models/multimodal_language_models.md (+1/-0); python/sglang/srt/configs/model_config.py (+7/-1); python/sglang/srt/layers/attention/vision.py (+49/-19); python/sglang/srt/models/glm4.py (+18/-6); python/sglang/srt/models/glm4v.py (+0/-2); python/sglang/srt/models/glm_ocr.py (+435/-0); python/sglang/srt/models/glm_ocr_nextn.py (+162/-0); python/sglang/srt/multimodal/processors/glm4v.py (+6/-1); python/sglang/srt/utils/common.py (+1/-0)
LABELS: documentation, high priority, Multi-modal, run-ci
BODY: Support for GLM-OCR Model, transformers PR [here](https://github.com/huggingface/transformers/pull/43391) ⏎ Need transformer>=5.0.0dev0, but not 5.0.0, so also changed min requirements in glm4v.

### L1-d578b41bad  (L1, 2026-01-27, sha d578b41badc8, PR #17615)
TITLE: [NPU] Adapt cann 8.5: use sfa and lightning indexer op from cann and CI update (#17615)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+13/-23); .github/workflows/nightly-test-npu.yml (+4/-8); .github/workflows/pr-test-npu.yml (+4/-8); .github/workflows/release-docker-npu-nightly.yml (+2/-2); .github/workflows/release-docker-npu.yml (+2/-2); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+9/-3); python/sglang/srt/hardware_backend/npu/utils.py (+0/-8); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+7/-8); scripts/ci/npu/npu_ci_install_dependency.sh (+11/-22)
LABELS: npu, run-ci
BODY: ## Motivation ⏎ There are some bugs of sparse flash attention and lightning indexer in custom_ops package. Fixed ops are intergrated in CANN 8.5 so we update these ops and call them using torch_npu. ⏎  ⏎ lightning_indexer: https://gitcode.com/cann/ops-transformer/blob/master/attention/lightning_indexer/README.md ⏎ sfa: https://gitcode.com/cann/ops-transformer/blob/master/attention/sparse_flash_attention/README.md ⏎  ⏎ Upgrading the CI  to the correspon …[truncated]

### L1-8acd4d7d7e  (L1, 2026-01-28, sha 8acd4d7d7e6f, PR #17600)
TITLE: Make flashMLA work on: Cu13, B300 (#17600)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/cmake/flashmla.cmake (+53/-0)
LABELS: sgl-kernel, run-ci
BODY: Here are the steps to **build local.** ⏎  ⏎ Also, the cuda packaged cu13 wheel will now work. ⏎  ⏎ Steps: ⏎  ⏎ `sudo ln -s /usr/local/cuda/include/cccl/cuda ⏎ /usr/local/cuda/include/cuda` (it moved). ⏎  ⏎ This is validated in sglang docker. ⏎  ⏎ Important: when .venv/ is present in sglang dir, it breaks compilation (even if not activated) ⏎  ⏎ `nohup bash -c 'CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") make build' > bu …[truncated]

### L1-1953efb60e  (L1, 2026-01-28, sha 1953efb60e82, PR #17863)
TITLE: [AMD] ROCm: route W4A16 MoE to Triton and fix packed-weight loading (#17863)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-1); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+69/-0)
LABELS: amd, run-ci
BODY: ## Motivation ⏎  As issue #17854  ⏎ On ROCm, `CompressedTensorsWNA16MoEMethod` currently routes to Marlin kernels by default. Marlin is NVIDIA‑only, which breaks Kimi‑K2.5 (native INT4) on MI300X. This patch dispatches ROCm to Triton and fixes the weight‑loading transpose path to avoid shape mismatches. ⏎  ⏎  ⏎   ## Modifications ⏎ - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` ⏎   - Add `CompressedTensorsWNA16Tri …[truncated]

### L1-e9d727cb92  (L1, 2026-01-28, sha e9d727cb9218, PR #17499)
TITLE: [MUSA][7/N] Enhance CUDA / PyNccl wrapper to support MTLink connectivity detection (#17499)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/pyproject_other.toml (+1/-1); python/sglang/srt/distributed/device_communicators/cuda_wrapper.py (+5/-1); python/sglang/srt/distributed/device_communicators/custom_all_reduce.py (+15/-3); python/sglang/srt/distributed/device_communicators/custom_all_reduce_ops.py (+4/-3); python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py (+8/-1); python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py (+6/-4)
LABELS: documentation, dependencies, mthreads
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is the seventh in a series of pull requests (tracked in https://github.com/sgl-project/sglang/issues/16565) to add full support for [Moore Threads](https://en.mthreads.com/) GPUs, leveraging MUSA (Meta-computing Unified System Architecture) to accelerate LLM inference. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ This commit updates the runtime wrappers and communication utilities to improve MUSA backend compatibility across SGLang’s distr …[truncated]

### L1-67fb492c9a  (L1, 2026-01-28, sha 67fb492c9a8a, PR #17844)
TITLE: [CI] Fix test_moe_fused_gate error (#17844)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: sgl-kernel/tests/test_moe_fused_gate.py (+116/-1)
LABELS: sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎ 3. Trigge …[truncated]

### L1-0998de088b  (L1, 2026-01-28, sha 0998de088b20, PR #17891)
TITLE: [Perf] Tune Llama-4-Scout-17B-16E-Instruct fused moe kernel (#17891)
SOURCES: path_config_only, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=16,N=2048,device_name=NVIDIA_B200.json (+146/-0)
LABELS: ready-to-merge
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Llama-4-Scout-17B-16E-Instruct is not tuned on `fused_moe_kernel` and b200 gpu, which leads to a sub-optimal performance of throughput. ⏎  ⏎ This PR runs the script that [tunes Triton MoE Kernel](https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/fused_moe_triton/README.md#1-tuning_fused_moe_tritonpy), benchmarks the decode throughput and profiles the performance of tuned `fused_moe_kernel`. ⏎  ⏎ The `fused_moe_ker …[truncated]

### L1-f1384f5293  (L1, 2026-01-28, sha f1384f5293ce, PR #17012)
TITLE: Integration mori backend for EP a2a data communication (#17012)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+143/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+13/-0); python/sglang/srt/layers/moe/rocm_moe_utils.py (+73/-0); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+8/-0); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+461/-0); python/sglang/srt/layers/moe/utils.py (+4/-0); docs/advanced_features/expert_parallelism.md (+3/-2); docs/references/environment_variables.md (+10/-0); python/sglang/srt/layers/attention/utils.py (+1/-1); python/sglang/srt/managers/scheduler_metrics_mixin.py (+2/-0); (+5 more)
LABELS: documentation, amd, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Co-author: @Duyi-Wang @billishyahao ⏎  ⏎ ## Motivation ⏎ MORI-EP is a high-performance all-to-all communication kernel for AMD GPUs. For more details, see the [MORI project](https://github.com/ROCm/mori). MORI also provides support for CUDA Graph.   ⏎ AMD have deliverd a Large-EP solution based on MORI-EP, more details in [BLOG](https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html) and [inferenceMAX](https://inferencema …[truncated]

### L1-d3cdee0a04  (L1, 2026-01-28, sha d3cdee0a040b, PR #17246)
TITLE: [MUSA][4/N] Add common device utilities, distributed backend, and custom op wiring (#17246)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/multimodal_gen/runtime/layers/custom_op.py (+8/-0); python/sglang/srt/configs/device_config.py (+1/-1); python/sglang/srt/distributed/parallel_state.py (+6/-0); python/sglang/srt/layers/rotary_embedding.py (+3/-0); python/sglang/srt/layers/utils/multi_platform.py (+10/-0); python/sglang/srt/model_executor/model_runner.py (+4/-2); python/sglang/srt/utils/common.py (+77/-8)
LABELS: documentation, dependencies, run-ci, diffusion, mthreads
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is the fourth in a series of pull requests (tracked in https://github.com/sgl-project/sglang/issues/16565) to add full support for [Moore Threads](https://en.mthreads.com/) GPUs, leveraging MUSA (Meta-computing Unified System Architecture) to accelerate LLM inference. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ - **`python/sglang/multimodal_gen/runtime/layers/custom_op.py`**: Add `forward_musa()` method and MUSA platform dispatch in `Cust …[truncated]

### L1-b77b0ffd60  (L1, 2026-01-29, sha b77b0ffd6021, PR #15904)
TITLE: [NPU] NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix (#15904)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.ep.layer
FILES: python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+6/-21); python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-12); docs/platforms/ascend_npu_deepseek_example.md (+0/-2); docs/platforms/ascend_npu_qwen3_examples.md (+0/-2); python/sglang/srt/layers/quantization/unquant.py (+15/-5); python/sglang/srt/models/qwen3_moe.py (+10/-7); test/registered/ascend/test_ascend_memory_consumption.py (+76/-0)
LABELS: documentation, quant, deepseek, npu, run-ci
DEEP_STUDY: deep-study correctness case sglang:b77b0ffd60: class=perf_regression_as_correctness; symptom=performance_or_availability; introducing=unknown || deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ The part of closed PR https://github.com/sgl-project/sglang/pull/11984.  ⏎ Adding weight conversion from ND to FRACTAL_NZ speeds up the GroupedMatmul kernel ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Add NZ conversion for non-quantized MOE models (11% speedup an average), update NZ documentaion ⏎ 2. Move transpose(1,2) from forward_npu() to process_weights_after_loading() ⏎ 3. Fix double memory consumption bug for Qwen MOE models on ascend, remove o …[truncated]

### L1-ef1c512754  (L1, 2026-01-29, sha ef1c5127545e, PR #17735)
TITLE: Add aiter bias moe support in gpt-oss mxfp4 model (#17735)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4.py (+108/-1); python/sglang/srt/server_args.py (+7/-0)
LABELS: amd, aiter, run-ci
BODY: ## Motivation ⏎  ⏎ Optimized MoE performance in ROCm platform when running the gpt-oss mxfp4 model ⏎  ⏎ ## Modifications ⏎  ⏎ mxfp4.py => Add aiter path for weight processing and use aiter fusedMoe function to handle MoE op ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ `sglang# python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --port 8000 ⏎ 100%|████████████████████████████████████████████████████████████████████████████████████████████████████ …[truncated]

### L1-3c9cc44ff5  (L1, 2026-01-29, sha 3c9cc44ff5da, PR #17449)
TITLE: Add mxfp8 support for online quantization, Triton dense linear, and CUTLASS MoE (#17449)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+100/-5); docs/advanced_features/server_arguments.md (+1/-1); python/sglang/srt/layers/quantization/__init__.py (+1/-0); python/sglang/srt/layers/quantization/fp8.py (+235/-19); python/sglang/srt/layers/quantization/fp8_kernel.py (+139/-0); python/sglang/srt/layers/quantization/fp8_utils.py (+127/-0); python/sglang/srt/model_loader/weight_utils.py (+3/-0); python/sglang/srt/server_args.py (+18/-5); python/sglang/test/test_block_fp8.py (+101/-1)
LABELS: documentation, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎ @humansand ⏎  ⏎ https://github.com/sgl-project/sglang/issues/17093 ⏎ This PR adds mxfp8 quantization support to SGLang, using Triton for dense linear layer, and existing mxfp8 CUTLASS groped GEMM kernel in sgl-kernel from https://github.com/sgl-project/sglang/pull/13731 for MoE. ⏎  ⏎ Online mxfp8 quantization from bf16 checkpoints and serving mxfp8 checkpoints directly are both supported. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  …[truncated]

### L1-71e4d3b6bc  (L1, 2026-01-29, sha 71e4d3b6bc46, PR #10858)
TITLE: [Intel GPU] fix import error to run DeepSeek-V2-Lite model with BF16 on XPU (#10858)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+12/-1)
LABELS: ready-to-merge, xpu, run-ci
BODY: ## Motivation ⏎  ⏎ Enable deepseek model on XPU ⏎  ⏎ ## Modifications ⏎  ⏎ import cuda specific kernels when cuda is available ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ `python -m sglang.bench_one_batch  --batch-size 8  --input 1024  --output 1024  --model "deepseek-ai/DeepSeek-V2-Lite"  --dtype bfloat16 --tp 1  --trust-remote-code  --mem-fraction-static 0.8  --device xpu --correctness --json-model-override-args '{"num_hidden_layers": 8}'` ⏎ above command gives garbage o …[truncated]

### L1-336dc4579e  (L1, 2026-01-29, sha 336dc4579e94, PR #12525)
TITLE: [CPU] Optimize Qwen3-next model on CPU (#12525)
SOURCES: path_core
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: sgl-kernel/csrc/cpu/topk.cpp (+6/-0); python/sglang/srt/configs/qwen3_next.py (+7/-0); python/sglang/srt/configs/update_config.py (+43/-0); python/sglang/srt/layers/amx_utils.py (+49/-7); python/sglang/srt/layers/attention/fla/layernorm_gated.py (+27/-11); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+35/-10); python/sglang/srt/layers/attention/intel_amx_backend.py (+10/-2); python/sglang/srt/layers/attention/mamba/mamba.py (+41/-1); python/sglang/srt/mem_cache/memory_pool.py (+17/-1); python/sglang/srt/model_loader/weight_utils.py (+23/-3); (+3 more)
LABELS: sgl-kernel, intel, cpu, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: This PR adds unified CPU optimizations for Qwen3-next models, including: ⏎  ⏎ 1.  Add CPU paths to call optimized kernels, which is depending on below sgl-kernels: ⏎      a. chunk_gated_delta_rule  https://github.com/sgl-project/sglang/pull/12441 ⏎      b. fused_sigmoid_gating_delta_rule_update and fused_gdn_gating https://github.com/sgl-project/sglang/pull/12324 ⏎      c. fused_qkvzba_split_reshape_cat  https://github.com/sgl-project/sglang/pull/1233 …[truncated]

### L1-c35aa0238c  (L1, 2026-01-29, sha c35aa0238c73, PR #8226)
TITLE: [CPU][INT4] Add INT4 kernels for CPU  (#8226)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/moe.cpp (+83/-30); sgl-kernel/csrc/cpu/moe_int4.cpp (+484/-0); python/sglang/srt/layers/amx_utils.py (+9/-0); python/sglang/srt/layers/quantization/fp8.py (+7/-5); python/sglang/srt/layers/quantization/unquant.py (+7/-5); python/sglang/srt/layers/quantization/w8a8_int8.py (+7/-5); python/sglang/srt/models/deepseek_v2.py (+0/-2); sgl-kernel/csrc/cpu/gemm.h (+69/-0); sgl-kernel/csrc/cpu/gemm_int4.cpp (+811/-0); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+24/-10); (+4 more)
LABELS: quant, deepseek, sgl-kernel, intel, cpu, run-ci
BODY: This PR implements CPU int4 kernels, which are called by CPU AWQ frontend https://github.com/sgl-project/sglang/pull/8225/files ⏎  ⏎ - Including: AWQLinear and AWQMoE

### L1-c04efe030a  (L1, 2026-01-30, sha c04efe030acc, PR #16294)
TITLE: [Model] Add K-EXAONE model support (#16294)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+4/-0); python/sglang/srt/models/exaone_moe.py (+881/-0); python/sglang/srt/models/exaone_moe_mtp.py (+106/-0); python/sglang/srt/server_args.py (+9/-7)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR integrates the [K-EXAONE-236B-A23B](https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B) model, recently released by [LG AI Research](https://huggingface.co/LGAI-EXAONE). Our model's github link is here: [K-EXAONE github](https://github.com/LG-AI-EXAONE/K-EXAONE). ⏎  ⏎ - Goal: To provide native support for K-EXAONE within SGLang's high-performance inference framework. ⏎  ⏎ - Details: Implementation includes the EXAONE-M …[truncated]

### L1-c52578c7fd  (L1, 2026-01-30, sha c52578c7fd19, PR #17940)
TITLE: 【docs】【NPU】Update Expert Parallelism docs for Ascend NPU (#17940)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docs/advanced_features/attention_backend.md (+1/-1); docs/advanced_features/expert_parallelism.md (+41/-1)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ Update Expert Parallelism Docs for Ascend NPU to append some parameter guidance ⏎  ⏎ ## Modifications ⏎  ⏎ Update Expert Parallelism Docs for Ascend NPU to append some parameter guidance ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-r …[truncated]

### L1-ee3058c6e8  (L1, 2026-01-31, sha ee3058c6e867, PR #18017)
TITLE: [NPU] fix sgl-kernel-npu package url error in npu.Dockerfile (#18017)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+2/-2)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ npu nightly docker image build failed for url error ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ change sgl-kernel-npu url ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ NA ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎ NA ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://git …[truncated]

### L1-22498e10c0  (L1, 2026-01-31, sha 22498e10c06a, PR #17965)
TITLE: [Fix] Triton TP MoE Dpsk V3/Qwen3 Coder with SwapAB (#17965)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+114/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json (+128/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=80,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json (+164/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+4/-16); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py (+17/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Enable SwapAB on H200, which has been verified in this PR. And also retune the configuration of TP=8 EP=1 DeepseekV3 and TP=8 EP=2 Qwen3 Coder, in the latency scenario. ⏎  ⏎ Also enable tuning with EP. ⏎  ⏎ Feel free to follow these steps for any model: ⏎  ⏎ ```diff ⏎ +++ b/python/sglang/srt/models/qwen3_moe.py ⏎ @@ -301,6 +301,28 @@ class Qwen3MoeSparseMoeBlock(nn.Module): ⏎          # router_logits: (num_tokens, n_experts) ⏎          rou …[truncated]

### L1-9951a1ae07  (L1, 2026-01-31, sha 9951a1ae074d, PR #17858)
TITLE: Fix: Remove duplicate assignment for use_w4afp8 (#17858)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+0/-1)
BODY: ## Motivation ⏎ It was discovered that the assignment of use_w4afp8 is duplicated. ⏎  ⏎  ⏎ ## Modifications ⏎ Remove redundant assignment of use_w4afp8 to False. ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals  …[truncated]

### L1-a0bae4c343  (L1, 2026-01-31, sha a0bae4c34349, PR #17299)
TITLE: Migrate 4-GPU/8-GPU workflow jobs to stage-c and add CI registry decorators (#17299)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+35/-63); scripts/ci/utils/slash_command_handler.py (+7/-0); test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py (+3/-0); test/registered/4-gpu-models/test_gpt_oss_4gpu.py (+4/-0); test/registered/4-gpu-models/test_qwen3_next_models.py (+3/-0); test/registered/4-gpu-models/test_qwen3_next_models_mtp.py (+3/-0); test/registered/8-gpu-models/test_deepseek_v32_basic.py (+3/-0); test/registered/8-gpu-models/test_deepseek_v32_mtp.py (+3/-0); test/registered/8-gpu-models/test_deepseek_v3_basic.py (+3/-0); test/registered/8-gpu-models/test_deepseek_v3_mtp.py (+3/-0); (+29 more)
LABELS: high priority, deepseek, blackwell, run-ci
BODY: ## Summary ⏎  ⏎ Migrate ALL 4-GPU/8-GPU CUDA tests from `test/srt/` to `test/registered/` with CI registry decorators. ⏎  ⏎ **Changes:** ⏎ - Move 24+ test files to `test/registered/` organized by feature (8-gpu-models, disaggregation, ep, kernels, models, quant, stage-c, rl, scheduler, layers, profiling) ⏎ - Add new suites to `test/run_suite.py`: stage-c-test-4-gpu-h100, stage-c-test-8-gpu-h200, stage-c-test-8-gpu-h20, stage-c-test-4-gpu-b200, stage-c-test-4 …[truncated]

### L1-e5ac6229e1  (L1, 2026-01-31, sha e5ac6229e186, PR #18050)
TITLE: Fix installation script for H200 runners (#18050)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: scripts/ci/cuda/ci_install_deepep.sh (+36/-6); scripts/ci/cuda/ci_install_dependency.sh (+24/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎ 3. Trigge …[truncated]

### L1-0fe282543f  (L1, 2026-02-01, sha 0fe282543fbb, PR #17025)
TITLE: [NPU] support the Enable return routed experts (#17025)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py, L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+7/-0); python/sglang/srt/layers/moe/topk.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ NPU support the Enable return routed experts ⏎ ## Modifications ⏎  ⏎  ⏎ The following code is added to the fused_topk_npu method to obtain topk: ⏎ ```python ⏎         get_global_experts_capturer().capture( ⏎             layer_id=layer_id, ⏎             topk_ids=topk_ids, ⏎         ) ⏎ ``` ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎ #### Scheduler ⏎ ```bash ⏎ python -m sglang.launch_server \ ⏎ --model-path /home/weights/DeepSeek-R1_w8a8 \ ⏎ --host  …[truncated]

### L1-cbf1500390  (L1, 2026-02-02, sha cbf150039037, PR #17634)
TITLE: [MiMoV2Flash] [feat]: support two batch overlap (#17634)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/batch_overlap/operations_strategy.py (+84/-0); python/sglang/srt/models/mimo_v2_flash.py (+208/-8)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ support mimo_v2_flash two batch overlap: ⏎ p: ⏎ ```bash ⏎ python3 -m sglang.launch_server \ ⏎     --model-path /mnt/mify-gw-model-alicn3/models/global_step_84-FP8-Block \ ⏎     --pp-size 1 --dp-size 2 --tp-size 8 \ ⏎     --enable-dp-attention \ ⏎     --moe-a2a-backend deepep \ ⏎     --deepep-mode normal \ ⏎     --disaggregation-mode prefill \ ⏎     --page-size 1 \ ⏎     --host 0.0.0.0 \ ⏎     --port 30010 \ ⏎     --trust-remote-code \ ⏎     --mo …[truncated]

### L1-78bf13db44  (L1, 2026-02-02, sha 78bf13db4447, PR #16685)
TITLE: MoE Refactor: Refactor `modelopt_quant.py` -> `flashinfer_trllm.py` (#16685)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+277/-15); python/sglang/srt/layers/quantization/fp8.py (+1/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+64/-176)
LABELS: quant, run-ci, format
BODY: ## Motivation ⏎  ⏎ Followup on https://github.com/sgl-project/sglang/pull/15151#event-21910322181, and part of #8715

### L1-980d2936cd  (L1, 2026-02-03, sha 980d2936cd9a, PR #18084)
TITLE: model: support Step-3.5-Flash (#18084)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.runner.triton
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+20/-5); python/sglang/srt/layers/moe/moe_runner/triton.py (+8/-5); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/model_config.py (+24/-0); python/sglang/srt/configs/step3p5.py (+97/-0); python/sglang/srt/function_call/function_call_parser.py (+1/-0); python/sglang/srt/model_executor/model_runner.py (+2/-0); python/sglang/srt/model_loader/loader.py (+3/-0); python/sglang/srt/models/mimo_v2_flash_nextn.py (+1/-0); python/sglang/srt/models/step3p5.py (+1037/-0); (+5 more)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ add Step-3.5-Flash model support ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWN …[truncated]

### L1-495290aefd  (L1, 2026-02-03, sha 495290aefd1f, PR #11712)
TITLE: enable ut test for xpu devices (#11712)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+3/-1); python/sglang/test/runners.py (+17/-13); python/sglang/test/test_utils.py (+40/-0); test/manual/test_expert_location_updater.py (+4/-1); test/manual/test_forward_split_prefill.py (+2/-1); test/manual/test_get_weights_by_name.py (+7/-5); test/manual/test_triton_moe_wna16.py (+17/-10); test/registered/attention/test_create_kvindices.py (+9/-8); test/registered/attention/test_wave_attention_kernels.py (+64/-39); test/registered/core/test_hidden_states.py (+3/-3); (+10 more)
LABELS: documentation, high priority, quant, amd, dependencies, lora, Multi-modal, deepseek, speculative-decoding, hicache
BODY: (Please be kindly informed that this PR encompasses a large scope. It will be split into smaller PRs as requested.) ⏎  ⏎ This PR is to enable Sglang UTs on XPU. What we do in this PR:  ⏎ 1. Enabling multi-hardware config in test/runners.py ⏎ 2. Enabling multi-hardware config in test/test_utils.py ⏎ 3. Enabling multi-hardware on test_*.py as required.  ⏎  ⏎ How to Run UTs on XPU: ⏎ Apply this PR/diff on Sglang main,  and then build sglang env via docker/D …[truncated]

### L1-d48bbe3bed  (L1, 2026-02-03, sha d48bbe3beda7, PR #18173)
TITLE: [CI][NPU] Bugfix import sgl-kernel error (#18173)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+5/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ <img width="908" height="275" alt="image" src="https://github.com/user-attachments/assets/33c25ec3-d744-4307-8c4c-636485ede244" /> ⏎  ⏎ All NPU CIs failed due to an `sgl-kernel` import error introduced in #16685. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Conditionally import `sgl-kernel` when device is cuda alike. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ shall be covered by ci ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1 …[truncated]

### L1-e484c90cc7  (L1, 2026-02-03, sha e484c90cc7aa, PR #18091)
TITLE: Add triton_fused_moe config for GLM-4.7-FP8 tp8 H20 H20-3e (#18091)
SOURCES: path_config_only, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=161,N=192,device_name=NVIDIA_H20,dtype=fp8_w8a8,per_channel_quant=True.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=161,N=192,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,per_channel_quant=True.json (+146/-0)
LABELS: quant
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎ Many tenant use it for coding agent ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNE …[truncated]

### L1-99fab2ce67  (L1, 2026-02-03, sha 99fab2ce673e, PR #18065)
TITLE: [Bugfix] Fix Mistral Large 3 NVFP4 TRTLLM MoE (#18065)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+94/-103); test/registered/8-gpu-models/test_mistral_large3.py (+21/-8)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ TRTLLM MoE refactoring PR(#15151) broke Mistral Large 3 NVFP4 MoE support(#15049), this PR is trying to fix the issue. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     8|exact_match|↑  |0.9174|±  |0.0076| ⏎ |     |       |strict-match    | …[truncated]

### L1-efbf39583e  (L1, 2026-02-04, sha efbf39583e7a, PR #18195)
TITLE: Add MoE fused config for Qwen3-Coder-Next-FP8 on H100 TP=2 (#18195)
SOURCES: path_config_only, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎ Add optimized Triton MoE kernel configuration for Qwen3-Coder-Next-FP8 on H100 with TP=2. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎ `python -m sglang.bench_serving --backend sglang-oai-chat --model Qwen/Qwen3-Coder-Next-FP8 --dataset-name random --random-input-len 1024 --random-output-len 1024 --num-prompts 100 --max-concurrency 8` ⏎  ⏎ Summary: ⏎ ``` ⏎ Metric                                 B …[truncated]

### L1-a72f4f839c  (L1, 2026-02-04, sha a72f4f839c4d, PR #18243)
TITLE: Tiny fix for fp8 moe backend flashinfer_trtllm naming (#18243)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8_utils.py (+2/-2)
BODY: ## Motivation ⏎  ⏎ renames the function name for clarity.  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODE …[truncated]

### L1-368936a62b  (L1, 2026-02-04, sha 368936a62bdd, PR #13561)
TITLE: [XPU] Integrate MoE and minor improvements in XPU attention backend (#13561)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.runner.triton
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+34/-1); python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py (+3/-2); python/sglang/srt/layers/moe/moe_runner/triton.py (+13/-3); python/sglang/srt/layers/moe/topk.py (+1/-0); python/sglang/srt/layers/quantization/unquant.py (+49/-0); python/sglang/srt/utils/common.py (+11/-1); test/srt/run_suite.py (+1/-0); test/srt/xpu/test_deepseek_ocr.py (+121/-0)
LABELS: quant, deepseek, intel, xpu, run-ci
BODY: ## Motivation ⏎  ⏎ 1. Current ```fused_moe.fused_experts``` is a mixture implementation of triton/cutlass/aiter, the PR add sgl-kernel-xpu implementation as the high efficiency path for XPU. This is an experimental feature. Users need to enable sgl-kernel-xpu MoE by setting environmental variable ```SGLANG_USE_SGL_XPU=0``` ⏎ 2. Update test cases with DeepSeekOCR which is the smallest MoE model currently ⏎ 3. Minor improvements on XPU attention side,  …[truncated]

### L1-3e7ecb78a6  (L1, 2026-02-05, sha 3e7ecb78a60f, PR #18145)
TITLE: model: support interns1-pro (#18145)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+1/-0); python/sglang/srt/entrypoints/openai/protocol.py (+8/-0); python/sglang/srt/layers/rotary_embedding.py (+199/-0); python/sglang/srt/models/interns1pro.py (+252/-0); python/sglang/srt/models/qwen3_vl_moe.py (+8/-2); python/sglang/srt/multimodal/processors/interns1pro.py (+118/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ support [internlm/Intern-S1-Pro](https://huggingface.co/internlm/Intern-S1-Pro) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/s …[truncated]

### L1-8f8c1724ae  (L1, 2026-02-05, sha 8f8c1724ae2f, PR #18298)
TITLE: docker: add patch to increase GPU deepep timeout (#18298)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+3/-0)
BODY: This patch along with `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1` should solve https://github.com/sgl-project/sglang/issues/9867#issuecomment-3336551174

### L1-107958a489  (L1, 2026-02-09, sha 107958a48946, PR #17828)
TITLE: Make compressed-tensors MoEs support ignored layers (#17828)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+50/-20); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+30/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ When the MoE layer is not fully quantized, certain layers must be ignored during weight loading. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Extract the common function `get_scheme_dict` to provide a unified interface for both `Linear` and `FusedMoE` layers, determining whether a rollback is needed and matching the appropriate target to return the corresponding `scheme_dict`. ⏎ - Add `FusedMoE` to the `target_scheme_map`, then match normally by  …[truncated]

### L1-bec7fe9e65  (L1, 2026-02-10, sha bec7fe9e6523, PR #18362)
TITLE: [sgl-kernel] upgrade deepgemm (#18362)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+14/-3); sgl-kernel/build.sh (+2/-0); sgl-kernel/csrc/elementwise/concat_mla.cu (+1/-0)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 18562 (confirmed_revert, reason=unstated)
BODY: ## Motivation ⏎  ⏎ Fix DeepGEMM compilation error after updating to commit `9b680f42`: remove `USE_SABI` from the `deep_gemm_cpp` target's `Python_add_library` call. The new DeepGEMM commit uses pybind11 features that require the full CPython API (e.g., `PyTuple_SET_ITEM`, `Py_buffer`, `PyTypeObject` internals), which are unavailable when Python Stable ABI (`Py_LIMITED_API`) is enabled. Dropping `USE_SABI` allows pybind11 to access the complete CPy …[truncated]

### L1-2d38b8aca0  (L1, 2026-02-11, sha 2d38b8aca016, PR #18562)
TITLE: Revert "[sgl-kernel] upgrade deepgemm" (#18562)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+3/-14); sgl-kernel/build.sh (+0/-2); sgl-kernel/csrc/elementwise/concat_mla.cu (+0/-1)
LABELS: sgl-kernel
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 18362 reason=unstated
BODY: Reverts sgl-project/sglang#18362 ⏎ Also fix the commit to the latest one on DeepGemm's `sgl-release` branch

### L1-5875ef0a34  (L1, 2026-02-11, sha 5875ef0a3474, PR #18531)
TITLE: Clean up noisy startup log messages and refactor loader.py (#18531)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+0/-5); python/sglang/launch_server.py (+3/-0); python/sglang/srt/entrypoints/openai/serving_chat.py (+7/-1); python/sglang/srt/mem_cache/mamba_radix_cache.py (+0/-2); python/sglang/srt/model_executor/model_runner.py (+0/-5); python/sglang/srt/model_loader/loader.py (+34/-29); python/sglang/srt/utils/common.py (+16/-1)
LABELS: run-ci
BODY: ## Summary ⏎ This PR cleans up noisy/redundant log messages during server startup and does minor refactoring in the model loader. ⏎  ⏎ ## Log cleanup ⏎ - Suppress `FutureWarning` from deprecated `cuda.cudart` and `cuda.nvrtc` modules (in both main process and scheduler subprocess) ⏎ - Remove `MOE_RUNNER_BACKEND is not initialized, the backend will be automatically selected` info log (auto-selection still happens silently) ⏎ - Remove `Beginning to load weight …[truncated]

### L1-20554a0a4f  (L1, 2026-02-11, sha 20554a0a4fb6, PR #17799)
TITLE: [AMD] rocm 7.2 image release, PR test, Nightly Test (#17799)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.runner.triton
FILES: docker/rocm720.Dockerfile (+502/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+43/-12); python/sglang/srt/layers/moe/moe_runner/triton.py (+16/-2); .github/workflows/nightly-test-amd-rocm720.yml (+868/-0); .github/workflows/pr-test-amd-rocm720.yml (+793/-0); .github/workflows/pr-test-amd.yml (+20/-2); .github/workflows/release-docker-amd-rocm720-nightly-preview.yml (+82/-0); python/sglang/srt/layers/layernorm.py (+14/-1); python/sglang/srt/layers/quantization/fp8_kernel.py (+45/-4); python/sglang/srt/layers/quantization/unquant.py (+8/-2); (+16 more)
LABELS: quant, amd, Multi-modal, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Enable Rocm 7.2  ⏎  ⏎ ## Modifications ⏎  ⏎ ROCM 7.2 ⏎ - PR Test ⏎ - Nightly Test ⏎ - cleanup vllm dependencies ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Nightly All green : https://github.com/sgl-project/sglang/actions/runs/21875099676 ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pu …[truncated]

### L1-ded068a76e  (L1, 2026-02-12, sha ded068a76e00, PR #17997)
TITLE: Add LMF2 MoE model architecture (#17997)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/lfm2_moe.py (+192/-0); python/sglang/srt/model_executor/model_runner.py (+4/-1); python/sglang/srt/models/lfm2_moe.py (+679/-0); test/registered/openai_server/function_call/test_tool_choice.py (+36/-0)
LABELS: run-ci
BODY: This PR introduces Liquid Foundation Model Mixture of Experts architecture. ⏎  ⏎ Example model using this architecture: [LFM2-8B-A1B]( https://huggingface.co/LiquidAI/LFM2-8B-A1B) ⏎  ⏎ ## How to run ⏎  ⏎ ```bash ⏎ sglang serve --model-path LiquidAI/LFM2-8B-A1B --tool-call-parser lfm2   ⏎ ``` ⏎  ⏎ ## Benchmarks ⏎  ⏎ **GPQA Dimond:** 34.04 vs. 29.29 reported ⏎ **IFBench:** 26.53 vs. 25.85 reported ⏎  ⏎ Integration test for function calling: ⏎ ```bash ⏎ pytest test/ …[truncated]

### L1-2bd8363486  (L1, 2026-02-12, sha 2bd8363486e4, PR #18405)
TITLE: [PCG] GPT OSS Triton Kernel Support (#18405)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+22/-20); python/sglang/srt/compilation/piecewise_context_manager.py (+6/-0); python/sglang/srt/model_executor/model_runner.py (+6/-0); python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py (+12/-2); python/sglang/srt/models/gpt_oss.py (+21/-4); python/sglang/srt/server_args.py (+1/-6)
LABELS: run-ci, piecewise-cuda-graph
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Support Backend for GPT-OSS ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) an …[truncated]

### L1-454676811e  (L1, 2026-02-12, sha 454676811ec7, PR #18500)
TITLE: [Flashinfer Autotune] Fix FlashInfer FP4 MoE autotuning crash by removing incorrect flatten on hidden_states_scale (#18500)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-1)
LABELS: run-ci
ISSUES: #18499 [Bug] FlashInfer 0.6.3 trtllm_fp4_block_scale_moe AssertionError during autotune warmup with PP4
BODY: ## Motivation ⏎  ⏎ When using `flashinfer_trtllm` MoE backend with FP4 quantization (`modelopt_fp4`), the FlashInfer autotuner crashes with: ⏎  ⏎ ``` ⏎ File ".../model_runner.py", line 1730, in _flashinfer_autotune ⏎     self._dummy_run(batch_size=self.req_to_token_pool.size) ⏎ File ".../model_runner.py", line 1974, in _dummy_run ⏎     run_once() ⏎   ... ⏎ File ".../deepseek_v2.py", line 720, in forward_normal ⏎     final_hidden_states = self.experts(...) ⏎  …[truncated]

### L1-d97eb111a3  (L1, 2026-02-13, sha d97eb111a368, PR #18598)
TITLE: Support LingV2_5 model (#18598)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/bailing_hybrid.py (+188/-0); python/sglang/srt/configs/model_config.py (+20/-0); python/sglang/srt/layers/attention/attention_registry.py (+3/-0); python/sglang/srt/layers/attention/fla/layernorm_gated.py (+26/-5); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+369/-0); python/sglang/srt/layers/attention/linear/lightning_attn.py (+767/-0); python/sglang/srt/layers/attention/linear/linear_metadata.py (+70/-0); python/sglang/srt/layers/attention/linear/seg_la.py (+909/-0); python/sglang/srt/model_executor/model_runner.py (+14/-1); (+6 more)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎ This is used for support Bailing models, Ling-2.5 and Ring-2.5 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang …[truncated]

### L1-98ad284ebf  (L1, 2026-02-13, sha 98ad284ebf96, PR #18480)
TITLE: Added cuda availability guard (#18480)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.deep_gemm
FILES: python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+10/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `sgl_kernel` function imported without checking CUDA availability breaks CPU runs. ⏎ Raised from: https://sgl-fru7574.slack.com/archives/C07PEP77X6F/p1770521834654589 ⏎  ⏎ ## Modifications ⏎  ⏎ Added `torch.cuda.is_available()` guard to the import. ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-re …[truncated]

### L1-1be41e9036  (L1, 2026-02-14, sha 1be41e9036e1, PR #18448)
TITLE: [FlashInfer] Bump FlashInfer version from 0.6.2 to 0.6.3 (#18448)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); .github/workflows/release-docker-cu13-framework.yml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/server_args.py (+2/-2); python/sglang/srt/utils/common.py (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎ 3. Trigge …[truncated]

### L1-fa0ef6e4f7  (L1, 2026-02-14, sha fa0ef6e4f7f7, PR #18782)
TITLE: [VLM][LLM] Optimize fused_moe triton kernel tma (#18782)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+67/-8)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎  ⏎ During profiling Qwen3-VL-30B-A3B, we found that fused_moe had significant GPU bubbles in prefill phase. One reason is CUDA Graph is not enabled by default, many small kernels launch introduces overhead. Some other reason can be scheduler's CPU participates some small ops' calculation, which makes kernel launch delayed. This PR focus on fused_moe optimzation. ⏎  ⏎ <img width="2676" height="1636" alt="image" src="https://github.co …[truncated]

### L1-922fbc21e2  (L1, 2026-02-15, sha 922fbc21e25d, PR #18851)
TITLE: [Perf] Tune MiniMax M2 fused moe kernel on H100 GPU (#18851)
SOURCES: path_config_only, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=256,N=384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
LABELS: ready-to-merge
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎  ⏎ MiniMax-M2 is not tuned on fused_moe_kernel and H100 gpu, which leads to a sub-optimal performance of throughput. ⏎  ⏎ This PR runs the script that [tunes Triton MoE Kernel](https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/fused_moe_triton/README.md#1-tuning_fused_moe_tritonpy), benchmarks the decode throughput and profiles the performance of tuned fused_moe_kernel. ⏎  ⏎ The fused_moe_kernel is tuned by the followi …[truncated]

### L1-ad1bdb93df  (L1, 2026-02-15, sha ad1bdb93df02, PR #18833)
TITLE: perf: add minimax-2.5 fused_moe tuning config for h20 (#18833)
SOURCES: path_config_only, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=256,N=384,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎ Add h20 fused MoE tuning config for minimax 2.5 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ Before ⏎ ![before_pr](https://github.com/user-attachments/assets/569f930e-48b2-4a53-9e12-af163bfed9b2) ⏎  ⏎ After ⏎ ![after_pr](https://github.com/user-attachments/assets/cec4e2d9-9562-4aff-bca3-52de4e481a07) ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow …[truncated]

### L1-4e162d4b1b  (L1, 2026-02-15, sha 4e162d4b1bf5, PR #18835)
TITLE: change npu.dockerfile (#18835)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+3/-3); .github/workflows/release-docker-npu-nightly.yml (+1/-1)
LABELS: npu
BODY: ## Motivation ⏎  ⏎ change npu.dockerfile ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other  …[truncated]

### L1-5ddc84e33e  (L1, 2026-02-15, sha 5ddc84e33e64, PR #18437)
TITLE: [AMD] MORI-EP inter kernel type switch (#18437)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+15/-3); docker/rocm.Dockerfile (+1/-1); docker/rocm720.Dockerfile (+1/-1); docs/references/environment_variables.md (+1/-0)
LABELS: documentation, amd, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎  ⏎ MORI has recently implemented a new low-latency inter-node kernel, `InterNodeV1LL`, which delivers better performance than the existing `InterNodeV1` when the number of tokens per rank is below 256. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Since MORI does not yet support runtime automatic kernel switching, the inter-kernel is selected at initialization time based on `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK`. The switching threshold is con …[truncated]

### L1-07a24f1a38  (L1, 2026-02-16, sha 07a24f1a3846, PR #18860)
TITLE: update pre-commit config (#18860)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: .github/workflows/lint.yml (+2/-2); .pre-commit-config.yaml (+6/-6); 3rdparty/amd/tuning/benchmark_moe_rocm.py (+2/-4); benchmark/fla/benchmark_layernorm_gated.py (+3/-1); benchmark/tip_suggestion/bench_other.py (+2/-8); benchmark/tip_suggestion/bench_sglang.py (+2/-8); benchmark/tip_suggestion/lmql_funcs.py (+2/-8); docs/advanced_features/lora.ipynb (+10/-20); docs/advanced_features/structured_outputs.ipynb (+0/-1); docs/advanced_features/structured_outputs_for_reasoning_models.ipynb (+0/-1); (+125 more)
LABELS: documentation, quant, amd, lora, Multi-modal, deepseek, sgl-kernel, npu, piecewise-cuda-graph, diffusion
BODY: ## Motivation ⏎  ⏎ To stay up-to-date with the latest security/style patches and ensure the codebase remains consistent with current best practices. ⏎  ⏎ ## Modifications ⏎  ⏎ Synchronized pre-commit hook versions and performed a global run (--all-files) to maintain codebase consistency. ⏎ `pre-commit autoupdate`  + `pre-commit run --all-files` ⏎  ⏎ ## Accuracy Tests ⏎ ? ⏎  ⏎ ## Benchmarking and Profiling ⏎ ? ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping M …[truncated]

### L1-7a607c4900  (L1, 2026-02-16, sha 7a607c49009d, PR #18459)
TITLE: fix_get_quant_method_in_fused_moe_condition (#18459)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/moe_wna16.py (+6/-1)
LABELS: quant, run-ci
BODY: Purpose: to fix this bug: ⏎  ⏎ - https://github.com/sgl-project/sglang/issues/18457

### L1-597d17dd18  (L1, 2026-02-16, sha 597d17dd18af, PR #18009)
TITLE: Use ephemeral nccl port via get_free_port() (#18009)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+10/-17); test/registered/core/test_server_args.py (+35/-57)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ This improves the reliability of the nccl port selection logic by using the existing get_free_port() utility function instead of manually scanning for available ports. The get_free_port() function lets the OS assign an ephemeral port without risk of conflicts when running multiple instances of SGLang on a single host. ⏎  ⏎ ## Modifications ⏎  ⏎ - Replace manual port scanning loop with get_free_port() call ⏎ - Update tests to mock get_ …[truncated]

### L1-0d30896015  (L1, 2026-02-16, sha 0d3089601547, PR #18750)
TITLE: fix(sgl-kernel): use >= 120 for SM12x CUDA kernel dispatch (#18750)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.nvfp4
FILES: sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cu (+1/-1); sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu (+1/-1); sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu (+1/-1)
LABELS: sgl-kernel, blackwell
BODY: ## Summary ⏎ - Fix SM12x variant dispatch in three CUDA kernel files where `getSMVersion() == 120` exact match misses SM121a (DGX Spark, which returns 121) ⏎ - Changed `== 120` to `>= 120` to cover SM120, SM120a, SM121a, and future SM12x variants ⏎ - Consistent with `fp8_gemm_kernel.cu` which already correctly uses `>= 120` ⏎  ⏎ ## Files Changed ⏎ | File | Line | Change | ⏎ |------|------|--------| ⏎ | `sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu` | 451 |  …[truncated]

### L1-0ffd0a3995  (L1, 2026-02-16, sha 0ffd0a3995e5, PR #18389)
TITLE: Nsa trtllm mla sparse fp8 support with Deepseek v3.2 NVFP4 (#18389)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/advanced_features/server_arguments.md (+2/-2); docs/basic_usage/deepseek_v32.md (+4/-0); python/sglang/srt/layers/attention/nsa_backend.py (+172/-66); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+13/-97); python/sglang/srt/layers/attention/utils.py (+99/-0); python/sglang/srt/mem_cache/memory_pool.py (+14/-18); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+40/-0); python/sglang/srt/models/deepseek_v2.py (+6/-0); test/registered/hicache/test_nsa_pool_host_unit.py (+1/-0); test/registered/kernels/test_nsa_indexer.py (+1/-0)
LABELS: documentation, high priority, deepseek, blackwell, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ #17655  ⏎  ⏎ - support Deepseek v3.2 NVFP4 with trtllm mla sparse fp8 attention backend ⏎  ⏎ ## Modifications ⏎  ⏎ - update the nsa backend to support trtllm sparse fp8 attention backend ⏎ - update the deepseek v2 to make sure the cos_sin_cache pass to trtllm kernels ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ### GSM8K ⏎ ```bash ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 200 --parallel 100 --port 30000 ⏎ 100%|████████████████████ …[truncated]

### L1-f9c3def7fe  (L1, 2026-02-16, sha f9c3def7fe46, PR #18887)
TITLE: Fix CI: add flashinfer --download-cubin to install dependencies (#18887)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: scripts/ci/cuda/ci_download_flashinfer_cubin.sh (+32/-0); scripts/ci/cuda/ci_install_dependency.sh (+3/-0)
LABELS: run-ci
BODY: ## Summary ⏎ - Add `python3 -m flashinfer --download-cubin` step to `scripts/ci/cuda/ci_install_dependency.sh` after flashinfer-jit-cache installation ⏎ - Fixes `TypeError: 'NoneType' object is not callable` for `trtllm_fp4_block_scale_moe` on B200 CI runner ⏎ - Mirrors the existing pattern in the Dockerfile (line 220) ⏎  ⏎ failure example: https://github.com/sgl-project/sglang/actions/runs/22018874131/job/63624809610?pr=14105#logs ⏎  ⏎ ## Root Cause ⏎ T …[truncated]

### L1-eba6af385d  (L1, 2026-02-16, sha eba6af385d17, PR #17503)
TITLE: [2/N] Quantization Refactor: Compressed tensors MoE schemes (#17503)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-3); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+20/-15); python/sglang/srt/layers/moe/kt_ep_wrapper.py (+1/-1); python/sglang/srt/layers/quantization/base_scheme.py (+99/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+209/-11); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+0/-2190); python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py (+24/-2); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_scheme.py (+64/-5); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py (+358/-0); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py (+2/-2); (+9 more)
LABELS: blackwell, run-ci
BODY: ## Motivation ⏎  ⏎ Add MoE schemes to compressed-tensors instead of storing all classes in a single file. This should make it easier to implement additional functionality as well as clearly show that compressed-tensors also supports NPU hardware in some cases. ⏎  ⏎ Images and motivation for this PR can be viewed in our roadmap: #15194 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ Moved all classes from `compressed_tensors_moe.py` to new schemes in quantization/compressed …[truncated]

### L1-1b659bcb08  (L1, 2026-02-16, sha 1b659bcb088f, PR #18804)
TITLE: Fix GLM-5 fused shared expert (#18804)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/glm4_moe.py (+2/-1)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ The MoE parts of GLM-5 consists of 256 routing experts and 1 shared expert, but currently the code fully inherits from `DeepseekV2ForCausalLM`, which causes the fused shared expert to be disabled by default when the architecture parameter is not explicitly passed. This leads to inefficient computation, and can not use the specific fused MoE kernel configs. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and  …[truncated]

### L1-355127c2e9  (L1, 2026-02-17, sha 355127c2e98e, PR #18940)
TITLE: Fix benchmark_sglang_fused_moe_triton.py (#18940)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py (+14/-4)
LABELS: run-ci
BODY: ## Motivation ⏎ Fix benchmark_sglang_fused_moe_triton.py ⏎  ⏎ Currently it doesn't work out of the box. Planning to update this script to compare against matmul_ogs as a follow_up ⏎  ⏎  ⏎ ## Modifications ⏎ get_model_config already handles TP / EP config, world size / model_parallel should just be 1 ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎ Before ⏎ ``` ⏎ python benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py ⏎ <f …[truncated]

### L1-150ed881be  (L1, 2026-02-18, sha 150ed881be2c, PR #18252)
TITLE: [4/N] Quantization Refactor: Quark MoE schemes (#18252)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-2); python/sglang/srt/layers/quantization/quark/quark.py (+96/-10); python/sglang/srt/layers/quantization/quark/schemes/__init__.py (+11/-2); python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py (+65/-4); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py (+2/-2); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py (+213/-0); python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py (+2/-2); python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8_moe.py (+5/-221)
LABELS: run-ci
BODY: ## Motivation ⏎ Add MoE schemes to quark instead of storing all classes in a single file. Follow up to https://github.com/sgl-project/sglang/pull/17503 ⏎  ⏎ Images and motivation for this PR can be viewed in our roadmap: https://github.com/sgl-project/sglang/issues/15194 ⏎  ⏎  ⏎ ## Modifications ⏎ Moved all classes from `quark_moe.py` to new schemes in quantization/quark/schemes/ ⏎ Removed `quark_moe.py` file ⏎ Added `get_moe_scheme` function to `quark.py …[truncated]

### L1-44ab752b7a  (L1, 2026-02-19, sha 44ab752b7aea, PR #18318)
TITLE: Add SDAR model support (#18318)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/supported_models/text_generation/diffusion_language_models.md (+2/-0); python/sglang/srt/dllm/config.py (+11/-6); python/sglang/srt/models/sdar.py (+589/-0); python/sglang/srt/models/sdar_moe.py (+746/-0); test/registered/dllm/test_sdar.py (+91/-0)
LABELS: documentation, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 19032 (confirmed_revert, reason=unstated)
BODY: ## Motivation ⏎  ⏎ Add **SDAR (`SDARForCausalLM`) model support** to SGLang so SDAR-style diffusion/LLM models can be served and evaluated via the existing SRT server + DLLM workflow (e.g., `--dllm-algorithm LowConfidence`). This enables running SDAR models with SGLang’s standard serving, CI, and evaluation tooling. ⏎  ⏎ ## Modifications ⏎  ⏎ * Implement / integrate SDAR model support (`SDARForCausalLM`, `SDARMoeForCausalLM`) into the model registry /  …[truncated]

### L1-73a7f0d049  (L1, 2026-02-19, sha 73a7f0d04997, PR #19032)
TITLE: Revert "Add SDAR model support" (#19032)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/supported_models/text_generation/diffusion_language_models.md (+0/-2); python/sglang/srt/dllm/config.py (+6/-11); python/sglang/srt/models/sdar.py (+0/-589); python/sglang/srt/models/sdar_moe.py (+0/-746); test/registered/dllm/test_sdar.py (+0/-91)
LABELS: documentation
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 18318 reason=unstated
BODY: Reverts sgl-project/sglang#18318

### L1-295bc17576  (L1, 2026-02-19, sha 295bc175760a, PR #19044)
TITLE: Feature/sdar support (#19044)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/supported_models/text_generation/diffusion_language_models.md (+2/-0); python/sglang/srt/dllm/config.py (+11/-6); python/sglang/srt/models/sdar.py (+589/-0); python/sglang/srt/models/sdar_moe.py (+746/-0)
LABELS: documentation, run-ci
BODY: ## Motivation ⏎  ⏎ Add **SDAR (`SDARForCausalLM`) model support** to SGLang so SDAR-style diffusion/LLM models can be served and evaluated via the existing SRT server + DLLM workflow (e.g., `--dllm-algorithm LowConfidence`). This enables running SDAR models with SGLang’s standard serving, CI, and evaluation tooling. ⏎  ⏎ ## Modifications ⏎  ⏎ * Implement / integrate SDAR model support (`SDARForCausalLM`, `SDARMoeForCausalLM`) into the model registry /  …[truncated]

### L1-fbb6098487  (L1, 2026-02-20, sha fbb60984872e, PR #17953)
TITLE: [AMD] support two batch overlapping for mori ep (#17953)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+51/-23); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-14); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+4/-0); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+448/-47); docs/advanced_features/server_arguments.md (+1/-1); python/sglang/srt/batch_overlap/operations_strategy.py (+9/-2); python/sglang/srt/batch_overlap/two_batch_overlap.py (+5/-0); python/sglang/srt/layers/attention/aiter_backend.py (+162/-52); python/sglang/srt/models/deepseek_v2.py (+13/-1); python/sglang/srt/server_args.py (+7/-5); (+1 more)
LABELS: documentation, deepseek, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 19161 (confirmed_revert, reason=ci_or_test_failure) || deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ co-author with @kkHuang-amd @ZhaiFeiyue @Duyi-Wang   ⏎ cc @HaiShaw ⏎  ⏎ This patch is to support TBO aka two batch overlapping feature for mori ep. It can be divided into the following changes:  ⏎ (1) We introduce MORI async API to support CU-free method for low latency scenario. ⏎ (2) We introduce multi hip stream to enable communication-computation overlapping for high throughput scenario. ⏎ (3) The relation between sglang arguments an …[truncated]

### L1-4bffd3a232  (L1, 2026-02-20, sha 4bffd3a2323a, PR #18988)
TITLE: [GPT-OSS] support fp8 online quantization for gpt-oss bf16 (#18988)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+26/-0); python/sglang/srt/server_args.py (+5/-1)
LABELS: run-ci
BODY: ## Motivation ⏎ 1. Keep `moe_runner_backend` as `auto` when launch gpt-oss bf16 with online quantization (e.g. fp8) to pick up either `deep_gemm` or `triton` moe backend, since `triton_kernels` moe backend [doesn't support](https://github.com/sgl-project/sglang/blob/2f592c3b181b7d75dbab8cc8f95806ee78e6ff01/python/sglang/srt/layers/quantization/fp8.py#L1335) online quantization like fp8 yet.  ⏎ 2. Updated `FP8MoeMethod` to accept `with_bias` to supp …[truncated]

### L1-6448de7e96  (L1, 2026-02-23, sha 6448de7e9690, PR #16945)
TITLE: Reorganize topk logic to clean up code and expose logical experts (#16945)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+62/-78)
LABELS: run-ci
BODY: ## Motivation ⏎ Reorganize the topk processing logic to unify a place of processing logical experts. I'm case which makes #12162 compatible with eplb without adding the capture logic everywhere. This PR also clean-up the duplicated eplb and padding code right after each topk kernel. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the P …[truncated]

### L1-43f83525c0  (L1, 2026-02-23, sha 43f83525c0a2, PR #19161)
TITLE: Revert "[AMD] support two batch overlapping for mori ep #17953" (#19161)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+23/-51); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+14/-6); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+0/-4); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+47/-448); docs/advanced_features/server_arguments.md (+1/-1); python/sglang/srt/batch_overlap/operations_strategy.py (+2/-9); python/sglang/srt/batch_overlap/two_batch_overlap.py (+0/-5); python/sglang/srt/layers/attention/aiter_backend.py (+52/-162); python/sglang/srt/models/deepseek_v2.py (+1/-13); python/sglang/srt/server_args.py (+5/-7); (+1 more)
LABELS: documentation, deepseek
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 17953 reason=ci_or_test_failure
BODY: ## Motivation ⏎  ⏎ Fix broken CI https://github.com/sgl-project/sglang/actions/runs/22256775640/job/64445687327 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](htt …[truncated]

### L1-4f25a48d7a  (L1, 2026-02-24, sha 4f25a48d7a47, PR #)
TITLE: support xverse_moe on npu
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L1-750ecf4a45  (L1, 2026-02-24, sha 750ecf4a45af, PR #19201)
TITLE: Add server CUDA graph warmup CI step for cold H200 nodes (#19201)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+27/-6); scripts/ci/cuda/warmup_deep_gemm.py (+399/-0); scripts/ci/cuda/warmup_server.py (+313/-0); test/registered/8-gpu-models/test_ring_2_5_1t.py (+0/-1)
BODY: ## Summary ⏎ - On cold H200 nodes (new nodes or after container recreation), CUDA graph capture triggers Triton autotuning which takes ~330s per server launch (vs ~30-60s on warm nodes). This caused `test_deepseek_v32_basic.py` to [timeout (1200s)](https://github.com/sgl-project/sglang/actions/runs/13479756506/job/37664899503) on new `gmi-h200-wk03` — server startup consumed ~450s, and the test launches 2 servers. ⏎ - New `warmup_server.py` launches  …[truncated]

### L1-d7a03c7ebf  (L1, 2026-02-24, sha d7a03c7ebfd7, PR #19266)
TITLE: [MoE Refactor] Refactor FlashInferFusedMoE into FusedMoE and flashinfer_trtllm.py (#19266)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_trtllm, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-2); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-98); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+74/-0); python/sglang/srt/layers/quantization/unquant.py (+18/-5)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎ Refactor FlashInferFusedMoE into FusedMoE and flashinfer_trtllm.py according to https://github.com/sgl-project/sglang/issues/8715 ⏎  ⏎ ## Modifications ⏎  ⏎ Delete FlashInferFusedMoE and move logic into flashinfer_trtllm.py and use FusedMoE.forward. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://githu …[truncated]

### L1-31c7dc9d99  (L1, 2026-02-24, sha 31c7dc9d990d, PR #19003)
TITLE: [VLM] Introduce FlashInfer CUDNN Prefill as ViT Backend (#19003)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/vision.py (+152/-0); python/sglang/srt/models/qwen3_vl.py (+259/-13); python/sglang/srt/server_args.py (+9/-1); test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py (+258/-0)
LABELS: performance, Multi-modal, run-ci, vlm, flashinfer
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎  ⏎ FlashInfer CUDNN Prefill demonstrates strong performance. This PR is to introduce it to SGLang as one of VLM ViT attention backends. A new "flashinfer" mm attention backend is added. This PR supports Qwen3-VL. In the next PRs we will adapt more VLMs to support this backend. ⏎  ⏎ **The performance improved 11.6% vs FA3. (TTFT reduce)**  ⏎ 1054ms vs 931ms. ⏎  ⏎ ``` ⏎ Server: ⏎ ➜  sglang_dev2 git:(support_vit_fi_backend) ✗ CUDA_VISIBLE_D …[truncated]

### L1-60eeef7370  (L1, 2026-02-25, sha 60eeef73701a, PR #19216)
TITLE: [AMD][with CI Fix] support two batch overlapping for mori ep (#19216)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+51/-23); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-14); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+4/-0); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+448/-47); docs/advanced_features/server_arguments.md (+1/-1); python/sglang/srt/batch_overlap/operations.py (+1/-1); python/sglang/srt/batch_overlap/operations_strategy.py (+9/-2); python/sglang/srt/batch_overlap/two_batch_overlap.py (+5/-0); python/sglang/srt/layers/attention/aiter_backend.py (+172/-59); python/sglang/srt/models/deepseek_v2.py (+14/-1); (+2 more)
LABELS: documentation, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: This PR is new version of old PR https://github.com/sgl-project/sglang/pull/17953 ⏎ and with the commit https://github.com/sgl-project/sglang/pull/19216/changes/08880e6f87c5ae0e063630697e5ddbe974eaa155 to address deepep CI failure: ⏎ https://github.com/sgl-project/sglang/actions/runs/22256775640/job/64445687327 ⏎ https://github.com/sgl-project/sglang/actions/runs/22256775640/job/64445687329 ⏎  ⏎ cc @HaiShaw @Fridge003  ⏎  ⏎ ## Motivation ⏎ co-author with …[truncated]

### L1-f1088beb6a  (L1, 2026-02-25, sha f1088beb6ab0, PR #13158)
TITLE: [NPU]Optimization of `forward_npu` for `UnquantizedFusedMoEMethod` (#13158)
SOURCES: path_integration+keyword, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/unquant.py (+20/-22)
LABELS: quant, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Using of `torch_npu.npu_moe_init_routing_v2` instead of sequence of operations: `torch_npu.npu_moe_init_routing` + `torch_npu.npu_moe_compute_expert_tokens` improves performance of non quantized MoE models ⏎  ⏎ ## Modifications ⏎  ⏎ Implementation of `forward_npu` method of `UnquantizedFusedMoEMethod` class has been changed, `torch_npu.npu_moe_init_routing` + `torch_npu.npu_moe_compute_expert_tokens` sequence has been substituted by  …[truncated]

### L1-350190487b  (L1, 2026-02-25, sha 350190487be4, PR #15422)
TITLE: Flashinfer MOE FP8 support for Mistral Large 3. (#15422)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8_moe.py (+52/-12); python/sglang/srt/layers/quantization/utils.py (+6/-0); test/registered/8-gpu-models/test_mistral_large3.py (+2/-5)
LABELS: quant, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ This PR brings in Flashinfer MOE FP8 support for Mistral Large 3. ⏎  ⏎ It requires an upcoming release of flashinfer to work. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Without EP8: ⏎  ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     8|exact_match|↑  |0.9257|±  |0.0072| ⏎ |     |       | …[truncated]

### L1-d0bb140034  (L1, 2026-02-25, sha d0bb14003489, PR #18700)
TITLE: [NPU] bugfix for model Qwen3-Coder-Next at weight shape transpose for npu. (#18700)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+2/-2); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+1/-1)
LABELS: npu, run-ci
BODY: ## Motivation ⏎ 1. The original code contained an import error that affected the proper loading of weight-related modules. ⏎ 2. Duplicate dimension transformations were applied to weight.data: ⏎ The first transpose was executed during the weight loading post-processing step. ⏎ The second identical permute was repeated in the fused MOE routing weight operator. ⏎ This redundant dimension reshaping caused the input tensor passed to the fused operator to  …[truncated]

### L1-cdc411160b  (L1, 2026-02-25, sha cdc411160b08, PR #19287)
TITLE: [NPU] Fix a corner case where FusedMoE.top_k is not explicitly declared (#19287)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/unquant.py (+3/-4)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ For some models (e.g. Llama 4, OLMoE), `top_k`s are not set when initializing FusedMoE, leading to a TypeError as follows: ⏎  ⏎ ![topk](https://github.com/user-attachments/assets/3da77802-26e9-4bcf-9292-de6dd95d159b) ⏎  ⏎ This pr aims to solve this issue by extracting `top_k` info from `topk_ids` when `FusedMoE.top_k` is `None`. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Use `topk_ids.shape[1]` as `top_k` when `layer.top_k` is not set ⏎  ⏎ ## Ac …[truncated]

### L1-e55e65535e  (L1, 2026-02-26, sha e55e65535e59, PR #19418)
TITLE: [Bugfix] Add rids to the batch filtering for two batch overlap (#19418)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/batch_overlap/two_batch_overlap.py (+1/-0); python/sglang/srt/server_args.py (+1/-1)
BODY: ## Motivation ⏎ Fix https://github.com/sgl-project/sglang/actions/runs/22427405426/job/64964095275?pr=17374, ⏎ introduced by https://github.com/sgl-project/sglang/pull/19372 ⏎  ⏎ And since we switch deepep test from NVIDIA H100 80GB to NVIDIA H200, eagle TBO test will fail due to OOM, so I increase the reserved mem of EAGLE to ensure a better `--mem-fraction-static`. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling …[truncated]

### L1-beabaa8d37  (L1, 2026-02-26, sha beabaa8d3767, PR #19181)
TITLE: [Kernel Slimming] Migrate marlin moe kernel to JIT (#19181)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.marlin
FILES: python/sglang/jit_kernel/csrc/gemm/marlin_moe/kernel.h (+37/-0); python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h (+1896/-0); python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh (+1089/-0); python/sglang/jit_kernel/moe_wna16_marlin.py (+172/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+6/-4); python/sglang/jit_kernel/benchmark/bench_moe_wna16_marlin.py (+251/-0); python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py (+329/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ See https://github.com/sgl-project/sglang/issues/17865 ⏎  ⏎ ## Modifications ⏎  ⏎ New files: ⏎ - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` — JIT-compiled CUDA kernel ported from `sgl-kernel/csrc/moe/marlin_moe_wna16/ops.cu` ⏎ - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` — Marlin MoE kernel template (ported from sgl-kernel) ⏎ - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/kernel.h` — Ke …[truncated]

### L1-2ad475b4ed  (L1, 2026-02-26, sha 2ad475b4edfe, PR #18696)
TITLE: use flashinfer.sampling (#18696)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+0/-1); python/sglang/srt/layers/sampler.py (+4/-3); sgl-kernel/benchmark/bench_top_k_top_p_sampling.py (+3/-2); sgl-kernel/csrc/common_extension.cc (+0/-15); sgl-kernel/include/sgl_kernel_ops.h (+0/-29); sgl-kernel/python/sgl_kernel/__init__.py (+0/-4); sgl-kernel/python/sgl_kernel/sampling.py (+0/-371); sgl-kernel/tests/test_sampling.py (+7/-6)
LABELS: sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/issues/17865 ⏎ move (external) flashinfer/csrc/sampling.cu ⏎ Call flashinfer directly from python, instead of compiling the operators into sgl_kernel ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ unittest ```python -m pytest sgl-kernel/tests/test_sampling.py -s``` ⏎  ⏎ gsm8k ⏎ ``` ⏎ python -m sglang.launch_server --model /data/Qwen3-8B/ ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 - …[truncated]

### L1-a1ef8e2cc0  (L1, 2026-02-26, sha a1ef8e2cc097, PR #19228)
TITLE: [AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning  (#19228)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json (+164/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json (+164/-0); benchmark/kernels/fused_moe_triton/common_utils.py (+23/-5); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+63/-6); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py (+72/-12)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎  Kimi K2.5 fused_moe_triton use default config so the performance is poor. ⏎  ⏎ ## Modifications ⏎  ⏎ 1. optimize Kimi K2.5 fused_moe_triton performance by tuning  ⏎ 2. fix fused_moe_triton/tuning_fused_moe_triton.py support int4_w4a16 ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ python3 benchmark/gsm8k/bench_sglang.py --host http://127.0.0.1 --port 30000 --num-questions 2000 --parallel 2000 --num-shots 8 ⏎ 100%|███████████████████████████████████████ …[truncated]

### L1-52c8a3632a  (L1, 2026-02-26, sha 52c8a3632a97, PR #19400)
TITLE: Fix missing StandardCombineInput import in BF16 flashinfer_trtllm MoE (#19400)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+2/-0)
LABELS: Multi-modal
DEEP_STUDY: deep-study correctness case sglang:52c8a3632a: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Summary ⏎ - Add missing runtime import of `StandardCombineInput` in `fused_experts_none_to_flashinfer_trtllm_bf16()`, matching the pattern already used by the FP8 and FP4 variants in the same file ⏎ - Fixes `NameError: name 'StandardCombineInput' is not defined` that crashes both `test_flashinfer_trtllm_gen_attn_backend.py` and `test_flashinfer_trtllm_gen_moe_backend.py` in the nightly B200 CI ⏎ - may be related to #19266 ⏎  ⏎ ## Error link ⏎ https: …[truncated]

### L1-86eb80007e  (L1, 2026-02-26, sha 86eb80007e78, PR #19331)
TITLE: [NPU] support Kimi-K2.5 on NPU (#19331)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+8/-1); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+1/-0); python/sglang/srt/models/kimi_k25.py (+14/-2)
LABELS: run-ci
ISSUES: #18841 [Feature][NPU] Support  Kimi-K2.5 on Ascend NPU
BODY: ## Motivation ⏎ support Kimi-K2.5 on Ascend ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ gsm8k test: ⏎ ```sh ⏎ python -m sglang.launch_server \ ⏎     --model-path /mnt/share/weights/Kimi-K2.5 \ ⏎     --host 127.0.0.1 --port 8100 \ ⏎     --trust-remote-code --device npu --attention-backend ascend \ ⏎     --tp-size 16 \ ⏎     --base-gpu-id 0 --mem-fraction-static 0.82 \ ⏎     --disable-radix-cache --chunked-prefill-size -1 \ ⏎     --moe-a2a-backend de …[truncated]

### L1-af3eccc9ab  (L1, 2026-02-26, sha af3eccc9abf7, PR #15731)
TITLE: [Perf] Eliminate the slice op for Flashinfer `trtllm_fp4_block_scale_moe` (#15731)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4.py (+3/-7)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ Depends on Flashinfer 0.6.0 #15551 ⏎  ⏎ `trtllm_fp4_block_scale_moe` has supported output with unpadded hidden dim. This PR applies this optimization to eliminate the slice op before the MoE when output is padded. ⏎  ⏎ ### Accuracy ⏎  ⏎ PR ⏎ ``` ⏎ [{'eval_name': 'aime25', 'model_name': 'gpt-oss-120b-high_temp1.0_20251223_190538', 'metric': 0.8958333333333334}] ⏎ ``` ⏎  ⏎ main ⏎ ``` ⏎ [{'eval_name': 'aime25', 'model_name': 'gpt-oss-120b-high_t …[truncated]

### L1-9b2fbf7e6a  (L1, 2026-02-27, sha 9b2fbf7e6ad4, PR #19203)
TITLE: [AMD] Merge Dockerfiles for ROCm (#19203)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/rocm.Dockerfile (+157/-34); docker/rocm720.Dockerfile (+0/-503); .github/workflows/pr-test-amd-rocm720.yml (+2/-2); .github/workflows/release-docker-amd-rocm720-nightly.yml (+1/-1); .github/workflows/release-docker-amd.yml (+1/-4); scripts/ci/amd/amd_ci_start_container.sh (+1/-7)
LABELS: amd, run-ci
BODY: ## Motivation ⏎  ⏎ `rocm720.Dockerfile` and `rocm.Dockerfile` shares a lot in common. It is time to unify them. ⏎  ⏎ ## Modifications ⏎  ⏎ Making `rocm720.Dockerfile` a superset of `rocm.Dockerfile` was the design choice back in #17799 . This PR mostly replaces `rocm.Dockerfile` with `rocm720.Dockerfile`, and modifies all the users of it accordingly.  Refactoring workflow files is a reasonable next step but not include in this PR.  ⏎  ⏎ ## Accuracy Tests …[truncated]

### L1-403195d59d  (L1, 2026-02-27, sha 403195d59de0, PR #19443)
TITLE: [AMD] [MiniMax-M2.5 Day 0] Add MiniMax-M2.5 nightly accuracy test (#19443)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/nightly-test-amd-rocm720.yml (+68/-0); .github/workflows/nightly-test-amd.yml (+68/-0); docs/basic_usage/minimax_m2.md (+22/-3); docs/supported_models/text_generation/generative_models.md (+1/-1); test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py (+245/-0); test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py (+249/-0)
LABELS: documentation, amd, run-ci
BODY: ## Summary ⏎  ⏎ - Add MiniMax-M2.5 (`MiniMaxAI/MiniMax-M2.5`) GSM8K few-shot accuracy tests for AMD GPUs (8-GPU, TP=8 + EP=8) ⏎   - **MI30x** (MI325/MI300X): `nightly-8-gpu-minimax-m25` with aiter backend ⏎   - **MI35x**: `nightly-8-gpu-mi35x-minimax-m25` with aiter backend ⏎ - Register both tests in the AMD nightly workflow ⏎ - No model code changes required -- MiniMax-M2.5 is a pure softmax-attention MoE model whose components all already support AMD …[truncated]

### L1-98e433e305  (L1, 2026-02-27, sha 98e433e305c5, PR #19492)
TITLE: [PD-Disagg][Fix] Remove 'test_external_dp_routing' from Rust Router constructor parameters. (#19492)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-model-gateway/bindings/python/src/sglang_router/router.py (+1/-0)
LABELS: model-gateway
BODY: cc @hnyls2002  ⏎  ⏎ ## Motivation ⏎  ⏎ `test_external_dp_routing` param has been introduced in #19268 for `minilb` test and was not filtered out when transmitted to the Rust implementation. The error occured. ⏎  ⏎ ``` ⏎ python -m sglang_router.launch_router --pd-disaggregation --port 30000 --policy random --prefill-policy random --decode-policy random --prefill http://10.235.192.132:8000 --decode http://10.235.192.132:8000 ⏎ Both --prefill-policy and --d …[truncated]

### L1-054bd71086  (L1, 2026-02-27, sha 054bd71086e4, PR #19379)
TITLE: [sgl-kernel slimming] remove sgl-kernel moe-wna16-marlin (#19379)
SOURCES: path_core, path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.runner.marlin
FILES: sgl-kernel/CMakeLists.txt (+0/-1); sgl-kernel/csrc/common_extension.cc (+0/-17); sgl-kernel/csrc/moe/marlin_moe_wna16/generate_kernels.py (+0/-158); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel.h (+0/-40); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4.cuh (+0/-39); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4b8.cuh (+0/-47); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku8b128.cuh (+0/-47); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4.cuh (+0/-39); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4b8.cuh (+0/-47); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku8b128.cuh (+0/-47); (+6 more)
LABELS: sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎ Follow https://github.com/sgl-project/sglang/pull/19181 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob …[truncated]

### L1-9469ad089b  (L1, 2026-02-27, sha 9469ad089b67, PR #18085)
TITLE: Fix nvfp4 weight update (#18085)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+10/-12); python/sglang/srt/layers/quantization/modelopt_quant.py (+72/-50); python/sglang/srt/layers/utils/common.py (+17/-0)
LABELS: high priority, quant, run-ci
BODY: ## Motivation ⏎ @humansand ⏎  ⏎ The existing nvfp4 `/update_weights_from_disk` endpoint does not work. This PR fixes it. ⏎  ⏎ This feature is a pre-requisite of nvfp4 RL in miles: https://github.com/radixark/miles/pull/546 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Introduce a `copy_or_rebind_param` for in-place weight update to keep CUDA graph stable. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ Testing on nvidia/Qwen3-30B-A3B-NVFP4 ⏎ ``` ⏎ hf download nvidia/Qwen3-30B-A3B-NVFP4 --l …[truncated]

### L1-eef44ec916  (L1, 2026-02-27, sha eef44ec916fe, PR #19387)
TITLE: [NPU]kimi k2 thinking bugfix (#19387)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-0); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ kimi k2 thinking bugfix on npu ⏎  ⏎ ## Modifications ⏎  ⏎ 1.When running the Kimi-K2 Linking large model on NPU, the following error was encountered ⏎ The reason is that when using W4A16 quantization, moe_mediate_stze is passed, resulting in an error when selecting quantit_method for calculating size. Therefore, this parameter needs to be passed externally ⏎  ⏎ RuntimeError: The expanded size of the tensor (4) must match the existing si …[truncated]

### L1-8240a87306  (L1, 2026-02-28, sha 8240a8730624, PR #19578)
TITLE: [AMD] MORI-EP support for EP4. (#19578)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+3/-0); docker/rocm.Dockerfile (+1/-1)
LABELS: amd, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Add MORI-EP support for EP4. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ DEP4 DeepSeek-R1 ⏎ ``` ⏎ # python bench_sglang.py --port 8000 --num-questions 200 ⏎ 100%|██████████████████████| 200/200 [00:40<00:00,  4.91it/s] ⏎ Accuracy: 0.975 ⏎ Invalid: 0.000 ⏎ Latency: 40.729 s ⏎ Output throughput: 439.591 token/s ⏎ ``` ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR fl …[truncated]

### L1-5fa6633485  (L1, 2026-02-28, sha 5fa66334858a, PR #19498)
TITLE: [AMD] Fix MoRI EP warmup hang by restoring deepep_mode=normal default (#19498)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+3/-0)
LABELS: amd, run-ci
BODY: ## Motivation ⏎ PR https://github.com/sgl-project/sglang/pull/19216 removed the `self.deepep_mode = "normal"` auto-setting for MoRI EP, leaving it as "auto". This causes MoRI EP to initialize in `EpMode.LOW_LATENCY` mode, which hangs during warmup on MI325X 8-GPU configurations.https://github.com/sgl-project/sglang/actions/runs/22471492647/job/65100816457?pr=19113#step:7:12894 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ The fix set `deepep_mode="normal"` by default  …[truncated]

### L1-0e86977811  (L1, 2026-03-01, sha 0e8697781115, PR #18742)
TITLE: [RL] Support per-layer mixed FP8/BF16 serving for FP8 checkpoints (#18742)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+27/-7); python/sglang/srt/layers/quantization/utils.py (+6/-6)
LABELS: run-ci
BODY: ## Motivation ⏎ @humansand ⏎  ⏎ As noted in https://arxiv.org/abs/2509.25149: ⏎  ⏎ > Keep a few sensitive linear layers in higher precision ⏎  ⏎ can improve training convergence. ⏎  ⏎ https://github.com/radixark/miles/pull/614 adds `--first-last-layers-bf16` for mxfp8 RL training. ⏎  ⏎ This PR enables reliable serving of FP8 checkpoints with intentional per-layer BF16 retention ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ - Normalize `modules_to_not_convert` to include both  …[truncated]

### L1-6822941514  (L1, 2026-03-02, sha 682294151441, PR #19005)
TITLE: [FlashInfer] Bump FlashInfer version from 0.6.3 to 0.6.4 (#19005)
SOURCES: symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); .github/workflows/release-docker-cu13-framework.yml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/server_args.py (+4/-7); python/sglang/srt/utils/common.py (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎ - CuteDSL FP4 MoE for DeepSeek-R1 Performance ([#2398](https://github.com/flashinfer-ai/flashinfer/pull/2398)) ⏎ - TRTLLM-Gen MxFP8 MoE Integration ([#2505](https://github.com/flashinfer-ai/flashinfer/pull/2505)) ⏎ - GDN Decode CuteDSL Kernel ([#2498](https://github.com/flashinfer-ai/flashinfer/pull/2498)) ⏎ - TRTLLM-Gen Skip-Softmax Attention ([#2477](https://github.com/flashinfer-ai/flashinfer/pull/2477), [#2547](https://github.co …[truncated]

### L1-468e3dc56b  (L1, 2026-03-02, sha 468e3dc56bee, PR #19030)
TITLE: [Qwen3.5] Set full attn_backend to trtllm_mha on SM100 by default when possible (#19030)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+85/-77)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The default mha backend for hybrid qwen models is set to triton, which leads to worse performance when trtllm_mha can be used.  ⏎  ⏎ ## Modifications ⏎  ⏎ - Refactor out `_get_default_attn_backend` so it can be called in `_handle_model_specific_adjustments` ⏎ - Remove the duplicated logic to set default MoE backend for qwen family models ⏎ - Get the default attn_backend by calling `_get_default_attn_backend` and pass it to `_handle_mam …[truncated]

### L1-c64274c746  (L1, 2026-03-02, sha c64274c746f2, PR #16331)
TITLE: Piecewise Cuda Graph set default (#16331)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+31/-1); python/sglang/srt/compilation/compile.py (+2/-14); python/sglang/srt/compilation/piecewise_context_manager.py (+23/-5); python/sglang/srt/configs/model_config.py (+19/-0); python/sglang/srt/layers/attention/fla/layernorm_gated.py (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+1/-1); python/sglang/srt/layers/communicator.py (+1/-1); python/sglang/srt/layers/quantization/awq.py (+7/-12); python/sglang/srt/layers/quantization/fp8_utils.py (+29/-2); python/sglang/srt/managers/schedule_batch.py (+1/-1); (+24 more)
LABELS: quant, Multi-modal, deepseek, npu, run-ci, piecewise-cuda-graph
BODY: ## Motivation ⏎  ⏎ Work in progress ⏎  ⏎ ## Modifications ⏎  ⏎ Work in progress ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Work in progress ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ Work in progress ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sgla …[truncated]

### L1-4c95953b77  (L1, 2026-03-03, sha 4c95953b7733, PR #19433)
TITLE: Fix/nemotron mtp quantaized (#19433)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-1); python/sglang/srt/configs/model_config.py (+4/-1); python/sglang/srt/layers/quantization/unquant.py (+6/-0); python/sglang/srt/models/nemotron_h_mtp.py (+1/-1); test/registered/model_loading/test_modelopt_loader.py (+59/-0)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎ Fix code so nemotron+mtp works for quantized checkpoints ⏎  ⏎ ## Modifications ⏎  ⏎ * mtp layer prefix should be mtp ⏎ * fused_moe_triton should handle non gated moe correctly ⏎ * parsing modelopt quant config in hf quant config should consider quant algo ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md …[truncated]

### L1-9305f0e58d  (L1, 2026-03-03, sha 9305f0e58dca, PR #19718)
TITLE: Support `triton_kernels` for GPT-OSS on SM120 (#19718)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4.py (+33/-18); python/sglang/srt/server_args.py (+8/-2)
LABELS: run-ci
BODY: Tested on 2 x 5090: ⏎  ⏎ This PR is written by: @amittell https://github.com/sgl-project/sglang/pull/16975, I just rebased changes and tested code. ⏎  ⏎ Requires: ⏎ `pip install triton_kernels --no-deps` ⏎  ⏎ ``` ⏎ python -m sglang.launch_server \ ⏎   --model openai/gpt-oss-20b \ ⏎   --reasoning-parser gpt-oss \ ⏎   --tool-call-parser gpt-oss \ ⏎   --tp 2 ⏎ ``` ⏎  ⏎ Looks alright. Around 260 TPS ⏎  ⏎ ``` ⏎ python3 -m sglang.test.send_one --stream --max-new-tokens  …[truncated]

### L1-fb37c0a400  (L1, 2026-03-03, sha fb37c0a40070, PR #18492)
TITLE: [args] Add Expert Parallelism Argument To SRT Runner (#18492)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/test/runners.py (+2/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Add expert parallelism argument to SRT Runner so that expert parallelism can be tested using the SRT Runner. ⏎  ⏎ ## Modifications ⏎  ⏎ Pass the ep_size argument into the engine initialization code. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAIN …[truncated]

### L1-0c760c4cd7  (L1, 2026-03-03, sha 0c760c4cd73b, PR #15917)
TITLE: Add tuned triton==3.5.1 b200 tp2, tp4 for qwen 3 next (#15917)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=128,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_B200.json (+146/-0)
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ TP8 seems unnecessary. ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b8c71f895e  (L1, 2026-03-03, sha b8c71f895e97, PR #15948)
TITLE:  Add tuned triton==3.5.1 h200 tp2, tp4 for qwen 3 next (#15948)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=128,device_name=NVIDIA_H200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=512,N=256,device_name=NVIDIA_H200.json (+146/-0)
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3c01b44700  (L1, 2026-03-03, sha 3c01b4470070, PR #17804)
TITLE: [Fix] NPU deepep hccl buffer and fix IPC safe check (#17804)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/distributed/parallel_state.py (+23/-1); python/sglang/srt/utils/common.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ 1. In npu device, when deepep is enabled, it requires more hccl buffer to process token dispatch. So in order to not mix with other HCCL buffer, we config the ep pg seperately(We do not want every pg to be that big) .  ⏎  ⏎ <img width="1165" height="353" alt="image" src="https://github.com/user-attachments/assets/0a081fad-d956-4cb2-81b5-4c2c6ff7242c" /> ⏎  ⏎  ⏎ 2. enable ipc handle safe serialization for torch_npu  ⏎  ⏎ <img width=" …[truncated]

### L1-c7ffbf25e9  (L1, 2026-03-03, sha c7ffbf25e940, PR #19803)
TITLE: [CI] Fix `rerun-ut` workflow: add DeepEP install, RDMA env, Blackwell detection (#19803)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/rerun-ut.yml (+13/-2); scripts/ci/utils/slash_command_handler.py (+15/-4)
BODY: - Select `ci_install_deepep.sh` for DeepEP suites (`stage-c-test-8-gpu-h20`, `stage-c-test-deepep-*`) via new `use_deepep` workflow input ⏎ - Set `SGLANG_CI_RDMA_ALL_DEVICES` for `8-gpu-h20` runner (matches `pr-test.yml`) ⏎ - Add `SGLANG_JIT_DEEPGEMM_FAST_WARMUP` to global env (was missing vs `pr-test.yml`) ⏎ - Extend `IS_BLACKWELL` detection to cover B200 runners ⏎  ⏎ Made with [Cursor](https://cursor.com)

### L1-c18cff4f90  (L1, 2026-03-03, sha c18cff4f9029, PR #19806)
TITLE: [CI] Add DeepGEMM warmup to stage-c-test-deepep-4-gpu (#19806)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+12/-0)
BODY: ## Summary ⏎ - The `stage-c-test-deepep-4-gpu` job times out (20 min) because DeepGEMM JIT compilation consumes ~10 min across 3 server launches during the test ⏎ - The 8-GPU DeepEP test already has warmup steps that avoid this — this PR adds the same pattern to the 4-GPU test ⏎ - Adds two warmup steps before "Run test": `warmup_deep_gemm.py` and `warmup_server.py` with `lmsys/sglang-ci-dsv3-test:4` ⏎  ⏎ **Failure example**: https://github.com/sgl-project/ …[truncated]

### L1-6851613b93  (L1, 2026-03-03, sha 6851613b93e3, PR #19656)
TITLE: [Bugfix] For cp: Fixed hang problem in prefix cache and kvcache support fp8 in-seq-split mode (#19656)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/basic_usage/deepseek_v32.md (+1/-2); python/sglang/srt/layers/attention/nsa/utils.py (+2/-1); python/sglang/srt/server_args.py (+1/-2)
LABELS: documentation, deepseek
BODY: ## Motivation ⏎  ⏎ 1. For cp(round-robin-split mode) : ⏎  ⏎ Problem: When cp+prefix cache is enabled, it will cause a hang problem after a period of pressure testing. ⏎ <img width="942" height="916" alt="image" src="https://github.com/user-attachments/assets/a41de075-af7a-41c0-802f-d2f04d202e51" /> ⏎  ⏎ Reason: When the prefix-cache is triggered, the actual number of tokens that are not hit is < cp_size. In this case, the splitting will cause the follow …[truncated]

### L1-329817e262  (L1, 2026-03-04, sha 329817e26248, PR #19866)
TITLE: [AMD] Move get_global_server_args import out of CUDA-only block to fix NameError on AMD (#19866)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-2)
LABELS: run-ci
BODY: ## Motivation ⏎ PR #19672 placed the `get_global_server_args` import inside the `if _is_cuda:` block, but it is called unconditionally in `fused_experts_impl`, causing `NameError` on non-CUDA platforms. https://github.com/sgl-project/sglang/actions/runs/22653862256/job/65659059427#step:5:1138 ⏎  ⏎  ⏎ ## Modifications ⏎ Moved `from sglang.srt.server_args import get_global_server_args` out of the `if _is_cuda:` conditional block to a top-level unconditi …[truncated]

### L1-ee5ccde0ad  (L1, 2026-03-04, sha ee5ccde0ad9e, PR #19672)
TITLE: support fused_moe_triton and moe_sum_all_reduce kernel fusion[reduce … (#19672)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+31/-4); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+30/-5); python/sglang/srt/server_args.py (+6/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎ 1.Fusing the fused_moe_triton and moe_sum_all_reduce kernels eliminates redundant intermediate calculations and data handling. This optimization delivers a 20%–30% reduction in TTFT, greatly improving the real-time user experience for MoE model inference. ⏎ 2.It reduces costly GPU global memory access and eases memory bandwidth bottlenecks. ⏎ 3.It lowers kernel launch and scheduling overhead, improving GPU resource efficiency. ⏎  ⏎ ##  …[truncated]

### L1-a710b7d791  (L1, 2026-03-04, sha a710b7d7910f, PR #18938)
TITLE: [Sarvam] Add inference support for Sarvam MoE LLMs (#18938)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/supported_models/text_generation/generative_models.md (+1/-0); python/sglang/srt/configs/model_config.py (+17/-0); python/sglang/srt/models/sarvam_moe.py (+1525/-0)
LABELS: documentation, run-ci
BODY: Adds inference support for two Sarvam MoE models: ⏎ Sarvam 30B MoE -- GQA attention with QK norm, fused QK-norm-RoPE kernel, sparse MoE with shared experts, and HF checkpoint weight remapping. ⏎ Sarvam 105B MoE -- MLA (Multi-head Latent Attention) with weight absorption (kv_b_proj split into w_kc/w_vc BMMs), FP8 BMM support, multi-backend attention dispatch (FA3, FlashMLA, CutlassMLA, etc.), and optional fp8 MHA prefill path for full-head attention …[truncated]

### L1-6910c1b281  (L1, 2026-03-04, sha 6910c1b281fc, PR #16364)
TITLE: [Feature][NPU]: add runtime support for GPTQ-quantized MoE models (#16364)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/linear.py (+2/-0); python/sglang/srt/layers/quantization/gptq.py (+291/-5); test/srt/ascend/test_ascend_gptq_moe.py (+76/-0); test/srt/run_suite.py (+1/-0)
LABELS: npu, run-ci
BODY: ## Stacked PR Notice ⏎  ⏎ > **This PR depends on and includes changes from #15203.** ⏎  ⏎ --- ⏎  ⏎ ## PR Description ⏎  ⏎ This PR introduces support for running **GPTQ-quantized Mixture-of-Experts (MoE) models** on Huawei Ascend (NPU) hardware. ⏎  ⏎ Previously, GPTQ support was primarily optimized for CUDA backends. Loading MoE architectures on NPU would cause runtime errors (e.g., `GPTQ Method does not support MoE, please use gptq_marlin`) because the spe …[truncated]

### L1-88cfa6c11d  (L1, 2026-03-04, sha 88cfa6c11d27, PR #19813)
TITLE: [NPU]Releasing redundant memory of w13_weight and nz when the ascend_fuseep feature is enabled (#19813)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-6)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Releasing redundant memory of w13_weight and nz when the ascend_fuseep feature is enabled ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ Test model：qwen3-480B-w8a8 ⏎ P： ⏎ python -m sglang.launch_server --model-path ${MODEL_PATH}  --disaggregation-mode prefill --host ${P_IP[$i]} \ ⏎         --port 8000 --disaggregation-bootstrap-port $((8995+$i)) --trust-remote-code --nnodes 1 --node-rank 0 \ ⏎         --tp-size 16 --mem-fraction-stati …[truncated]

### L1-46dced64ea  (L1, 2026-03-05, sha 46dced64ea5a, PR #19174)
TITLE: Adjust padding size to improve triton_kernels moe performance (#19174)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+18/-4); python/sglang/srt/layers/quantization/mxfp4.py (+3/-5)
LABELS: high priority, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Accuracy Tests ⏎ [{'eval_name': 'gpqa', 'model_name': 'dummy-medium_temp1.0_20260223_054434', 'metric': 0.7222222222222222}] ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎ ### e2e performance ⏎ tp4, for bs64, 32000 input len, 2000 output len ⏎ before: 4921.79 token/s ⏎ after: 5085.14 token/s ⏎  ⏎ ### Kernel ⏎  ⏎  ⏎ Before ⏎ <img width="388" height="852" alt="image" src="https://github.com/user-attachments/assets/28140b8e-45df-497e-9bbe-b7ba508de79c" /> ⏎  ⏎ After ⏎ < …[truncated]

### L1-2bdd89a6cd  (L1, 2026-03-05, sha 2bdd89a6cd6e, PR #19437)
TITLE: [Kernel Slimming] Migrate NVFP4 kernels to JIT (#19437)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.cutlass.nvfp4, L1.runner.flashinfer_trtllm, L1.cutlass.adapters
FILES: python/sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh (+341/-157); python/sglang/srt/layers/moe/cutlass_moe.py (+5/-2); python/sglang/srt/layers/moe/cutlass_moe_params.py (+28/-10); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+1/-1); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+1/-1); sgl-kernel/python/sgl_kernel/moe.py (+1/-48); python/sglang/jit_kernel/benchmark/bench_nvfp4_blockwise_moe.py (+250/-0); python/sglang/jit_kernel/benchmark/bench_nvfp4_quant.py (+181/-0); python/sglang/jit_kernel/benchmark/bench_nvfp4_scaled_mm.py (+175/-0); python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_expert_quant.cuh (+127/-143); (+17 more)
LABELS: quant, sgl-kernel, blackwell, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 20005 (confirmed_revert, reason=unstated) || deep-study: this PR was reverted by PR 20012 (reland, reason=crash_or_hang)
BODY: ## Motivation ⏎ #17865 ⏎  ⏎ This PR moves NVFP4 kernels from AOT `sgl-kernel` to JIT (`python/sglang/jit_kernel`). ⏎  ⏎ _“Should we migrate this given its CUTLASS dependency and JIT compile overhead?”_ ⏎  ⏎ I think yes, because in almost all serving setups with FlashInfer always installed, these JIT CUTLASS kernels are not usually on the default path anymore. ⏎  ⏎ Exact runtime behavior that can still use these even with FlashInfer available: ⏎ - On non-SM …[truncated]

### L1-51e5dc845a  (L1, 2026-03-05, sha 51e5dc845a1b, PR #20005)
TITLE: Revert "[Kernel Slimming] Migrate NVFP4 kernels to JIT" (#20005)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.nvfp4, L1.runner.flashinfer_trtllm, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+2/-5); python/sglang/srt/layers/moe/cutlass_moe_params.py (+10/-28); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+1/-1); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+1/-1); sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cu (+157/-341); sgl-kernel/python/sgl_kernel/moe.py (+48/-1); python/sglang/jit_kernel/benchmark/bench_nvfp4_blockwise_moe.py (+0/-250); python/sglang/jit_kernel/benchmark/bench_nvfp4_quant.py (+0/-181); python/sglang/jit_kernel/benchmark/bench_nvfp4_scaled_mm.py (+0/-175); python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_quant_entry.cuh (+0/-68); (+17 more)
LABELS: quant, sgl-kernel, blackwell
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 19437 reason=unstated
BODY: Reverts sgl-project/sglang#19437

### L1-84aaa69795  (L1, 2026-03-06, sha 84aaa69795f3, PR #19843)
TITLE: [AMD] Use bfloat16 for correction_bias in AITER FP8 path to avoid runtime dtype conversion for dsv3 (#19843)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+12/-7)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎ In the AITER (AMD MI355X) path with FP8 quantization, ```e_score_correction_bias``` was initialized in float32, but at runtime in biased_grouped_topk it gets cast to match gating_output.dtype: ⏎ correction_bias.to(dtype=gating_output.dtype) ⏎ On the AITER path, aiter_dsv3_router_gemm returns logits in bfloat16 (via .to(hidden_states.dtype) at rocm_linear_utils.py:30). This means every forward pass triggers a float32 → bfloat16 type c …[truncated]

### L1-de1a0afcbc  (L1, 2026-03-06, sha de1a0afcbc7c, PR #18357)
TITLE: [MUSA][10/N] Add GGUF support (#18357)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/gguf.py (+6/-5)
LABELS: mthreads
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR continues the ongoing effort (tracked in #16565) to add full support for **Moore Threads GPUs** in SGLang by leveraging **MUSA (Meta-computing Unified System Architecture)** for LLM inference. ⏎  ⏎ This submission is to enable GGUF support. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/pull/17946 has brought GGUF kernels to MUSA and this one simply enables it in SGLang runtime level. ⏎  ⏎ ### Testing Done …[truncated]

### L1-759700c808  (L1, 2026-03-06, sha 759700c80856, PR #20040)
TITLE: Fix SM120 `triton_kernels` MXFP4 `block_k` for GPT-OSS (#20040)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4.py (+4/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ On SM120, the `triton_kernels` MXFP4 path can pick a tile that exceeds the per-block shared-memory budget and hits `assert num_stages >= 1` during GPT-OSS startup. This sets `block_k=128` for the SM120 MXFP4 path, which is the largest power-of-two tile that fits this kernel’s requirements and the SM120 shared-memory limit. ⏎  ⏎ [details omitted] ⏎  ⏎ @b8zhong  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ```shell ⏎ python -m gpt_oss.evals --model openai/g …[truncated]

### L1-61de303f0a  (L1, 2026-03-06, sha 61de303f0a14, PR #19189)
TITLE: Fix fallback to default tactic (flashinfer autotuner) with trtllm_fp4_block_scale_moe (#19189)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-2); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+3/-1); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py (+3/-1)
LABELS: blackwell, run-ci
BODY: The flashinfer autotuner expects the first dimension of the MoE tensors to be num_tokens. ⏎  ⏎  ⏎ Following the changes in https://github.com/vllm-project/vllm/pull/35088.  ⏎  ⏎ Credits to @danisereb. Depends on flashinfer fix: https://github.com/flashinfer-ai/flashinfer/pull/2617 ⏎ cc. @Fridge003 @nvpohanh  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎  …[truncated]

### L1-43d6a32045  (L1, 2026-03-07, sha 43d6a32045ec, PR #18902)
TITLE: [sgl-kernel] rebase FlashMLA 0217 (#18902)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/cmake/flashmla.cmake (+47/-11); python/sglang/srt/layers/attention/flashmla_backend.py (+108/-65); sgl-kernel/python/sgl_kernel/flash_mla.py (+9/-0)
LABELS: high priority, sgl-kernel, run-ci
BODY: ## Summary ⏎ This PR updates SGLang's FlashMLA integration to the latest validated rebase snapshot on the SGL-maintained FlashMLA branch. ⏎  ⏎ ## What changed ⏎ - Updated `sgl-kernel/cmake/flashmla.cmake` to use the latest FlashMLA commit: ⏎   - `GIT_TAG=9804b12079e4c873514d3457aa588d3ccf40da28` ⏎ - Keeps SGLang pinned to an immutable commit SHA instead of a moving branch ref. ⏎  ⏎ ## FlashMLA side included in this pin ⏎ The pinned FlashMLA commit includes the reb …[truncated]

### L1-f88acf8780  (L1, 2026-03-07, sha f88acf87805b, PR #20012)
TITLE: [JIT Kernel] Reland NVFP4 kernels to JIT (#20012)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.nvfp4, L1.runner.flashinfer_trtllm, L1.cutlass.adapters
FILES: python/sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh (+341/-157); python/sglang/srt/layers/moe/cutlass_moe.py (+5/-4); python/sglang/srt/layers/moe/cutlass_moe_params.py (+28/-10); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+1/-1); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+1/-1); sgl-kernel/python/sgl_kernel/moe.py (+1/-48); python/sglang/jit_kernel/benchmark/bench_nvfp4_blockwise_moe.py (+250/-0); python/sglang/jit_kernel/benchmark/bench_nvfp4_quant.py (+181/-0); python/sglang/jit_kernel/benchmark/bench_nvfp4_scaled_mm.py (+175/-0); python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_expert_quant.cuh (+127/-143); (+17 more)
LABELS: quant, sgl-kernel, blackwell, run-ci
DEEP_STUDY: deep-study revert record: reland of PR(s) 19437 reason=crash_or_hang || deep-study performance PR (precision_format)
BODY: ## Summary ⏎  ⏎ Reland #19437 after fixing some missed custom-op registration that broke PCG ⏎  ⏎ cc @Fridge003 @DarkSharpness @BBuf  ⏎  ⏎ ## Accuracy Tests ⏎ PCG here is turned on by default now ⏎ ```shell ⏎ SGLANG_MOE_NVFP4_DISPATCH=1 sglang serve \ ⏎   --model-path nvidia/DeepSeek-V3-0324-NVFP4 \ ⏎   --tensor-parallel-size 4 \ ⏎   --expert-parallel-size 4 \ ⏎   --attention-backend trtllm_mla \ ⏎   --moe-runner-backend flashinfer_cutlass \ ⏎   --quantization  …[truncated]

### L1-230fb55899  (L1, 2026-03-08, sha 230fb5589960, PR #17216)
TITLE: [Performance] Decode Offload improves the long texts performance 100% through dynamic block offload. (#17216)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/references/environment_variables.md (+2/-0); python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py (+88/-31); python/sglang/srt/disaggregation/kv_events.py (+17/-0); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/managers/cache_controller.py (+2/-0); python/sglang/srt/managers/scheduler_output_processor_mixin.py (+7/-1); test/registered/disaggregation/test_disaggregation_decode_offload.py (+169/-0); test/registered/disaggregation/test_specv2_kvcache_offloading.py (+4/-1)
LABELS: documentation, high priority, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Changing the offload process to dynamic on-the-fly offloading significantly increases the number of requests that decode nodes can process in parallel. On H20-96GiB devices, the concurrency count can be more than doubled, and the end-to-end performance is doubled. The deployed test model is DeepSeek-V3.2-W4AFP8, with the device being H20-96GiB, 1P (PP8) 1D (EP8). ⏎  ⏎ DeepSeek-V3.2 introduces sparse kv cache, which brings significa …[truncated]

### L1-96724f490c  (L1, 2026-03-08, sha 96724f490c6d, PR #15678)
TITLE: Add auto bind numa node (#15678)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+1/-0); python/sglang/srt/managers/scheduler.py (+9/-4); python/sglang/srt/utils/common.py (+29/-5); python/sglang/srt/utils/numa_utils.py (+7/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Hardware: H20 3e ⏎ Model: qwen3-235b-a22b-thinking-2507-fp8 ⏎  ⏎ SGLang Prefill: ⏎ /opt/conda/bin/python -m sglang.launch_server --model-path /home/admin/model/ --host 0.0.0.0 --port 8088 --trust-remote-code --disaggregation-mode prefill --disaggregation-transfer-backend mooncake --tp-size 4 --dp-size 2 --mem-fraction-static 0.85 --chunked-prefill-size 16384 --enable-cache-report --page-size 64 --attention-backend fa3 --disaggregatio …[truncated]

### L1-f0153ad225  (L1, 2026-03-09, sha f0153ad225bd, PR #19757)
TITLE: [AMD][Feature] support fp4 dispatch and fp8 combine in moriep (#19757)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+35/-7); python/sglang/srt/layers/moe/rocm_moe_utils.py (+140/-0); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+88/-16); docs/references/environment_variables.md (+2/-0)
LABELS: documentation, amd, deepseek, npu, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: @Duyi-Wang @ZhaiFeiyue @billishyahao  ⏎  ⏎ ## Motivation ⏎  ⏎ MXFP4 quantization further reduces communication throughput and load, thereby accelerating model inference speed. ⏎  ⏎ ## Modifications ⏎  ⏎ In MoriEP, MXFP4 dispatch were introduced to reduce communication latency. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ### amd/DeepSeek-R1-0528-MXFP4 ⏎  ⏎ #### FP8 dispatch + BF16 combine ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------ …[truncated]

### L1-11b76d24dc  (L1, 2026-03-09, sha 11b76d24dc11, PR #18485)
TITLE: [NPU] [DLLM]DLLM LLaDA2.x graph mode support with NPU speedup modifications (#18485)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+103/-2); python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py (+18/-5); python/sglang/srt/managers/scheduler.py (+21/-0); python/sglang/srt/model_executor/forward_batch_info.py (+1/-1); python/sglang/srt/models/llada2.py (+13/-1); python/sglang/srt/server_args.py (+6/-0); test/srt/ascend/test_llada2_mini_ascend.py (+87/-0); test/srt/run_suite.py (+1/-0)
LABELS: npu, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Support DLLM model LLaDA2.0 and LLaDA2.1 and speedup with graph mode on Ascend NPU (910b/c) ⏎  ⏎ ## Modifications ⏎  ⏎ - Supported PA with chunked prefill (dllm_extend) mode with ascend backend ⏎ - Modified npu_graph_runner.py to support DLLM model capture and replay ⏎ - Added block based batch result processing to speedup the DLLM batch result processing on NPU ⏎  ⏎ ## Accuracy Tests ⏎ ### LLaDA2.0-mini "LowConfidence"bs1-tp1 on 910b wit …[truncated]

### L1-76ee4bb98c  (L1, 2026-03-10, sha 76ee4bb98c64, PR #19537)
TITLE: [FlashInfer v0.6.4] [RL] Integrate FlashInfer mxfp8 gemm, MoE, and routed MoE (#19537)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.runner.framework, L1.runner.flashinfer_trtllm, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+8/-2); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+218/-46); python/sglang/srt/layers/moe/moe_runner/runner.py (+4/-1); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+4/-0); python/sglang/srt/layers/moe/topk.py (+42/-9); python/sglang/srt/layers/moe/utils.py (+4/-0); docs/advanced_features/expert_parallelism.md (+1/-0); docs/advanced_features/server_arguments.md (+1/-1); python/sglang/srt/layers/quantization/fp8.py (+95/-15); (+4 more)
LABELS: documentation, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎ @humansand ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ This PR integrates: ⏎  ⏎ - Expand existing `flashinfer.fused_moe.trtllm_fp8_block_scale_moe` with mxfp8 ⏎ - Add `flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe` which supports mxfp8 and deepseek fp8 ⏎ - Add `flashinfer.mm_mxfp8` ⏎ - Expand test coverage ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ The following expanded tests passed on B200: ⏎ - `test_flashinfer_trtllm_gen_moe_backend.py` ⏎     - mxfp8 `trtllm …[truncated]

### L1-5a7c1b8ec6  (L1, 2026-03-10, sha 5a7c1b8ec632, PR #)
TITLE: [NPU] replace swiglu with custom kernel
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L1-ab4b863546  (L1, 2026-03-11, sha ab4b86354643, PR #20380)
TITLE: fix ci by removing nvidia-cutlass-dsl-libs-base and force reinstall n… (#20380)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+5/-0); scripts/ci/cuda/ci_install_dependency.sh (+2/-0)
BODY: …vidia-cutlass-dsl 4.3.5 ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ Fix CI for nvidia-cutlass-dsl auto upgrading install along with nvidia-cutlass-dsl-libs-base 4.4.x error. ⏎  ⏎ - refer https://github.com/flashinfer-ai/flashinfer/pull/2760 ⏎ - refer https://github.com/sgl-project/sglang/pull/20309 ⏎ - refer https://github.com/sgl-project/sgl-flash-attn/pull/40 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ##  …[truncated]

### L1-680d9d98e4  (L1, 2026-03-11, sha 680d9d98e468, PR #20309)
TITLE: Fix cutedsl ci error (#20309)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); scripts/ci/cuda/ci_install_dependency.sh (+3/-0)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎ 3. Trigge …[truncated]

### L1-ed42af99a9  (L1, 2026-03-11, sha ed42af99a92f, PR #18924)
TITLE: [NPU] [Quantization] w4a4 MoE layer support (#18924)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+154/-0); docs/platforms/ascend_npu_quantization.md (+1/-0); python/sglang/srt/layers/quantization/modelslim/modelslim.py (+10/-2); python/sglang/srt/layers/quantization/modelslim/schemes/__init__.py (+2/-0); python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a4_int4_moe.py (+135/-0); python/sglang/test/ascend/test_ascend_utils.py (+3/-0); test/registered/ascend/llm_models/test_ascend_qwen3_30b_w4a4.py (+37/-0)
LABELS: documentation, quant, npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ For now support only mixed quantization, where w13 (up_gate_proj) in w4a4 and w2 (down_proj) in w8a8. In future will refactor this to add mixed quantization in https://github.com/sgl-project/sglang/pull/17361 PR. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Modify python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py and python/sglang/srt/layers/quantization/modelslim/modelslim_moe.py ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ Server: ⏎ ```ASCE …[truncated]

### L1-9991debde3  (L1, 2026-03-11, sha 9991debde37b, PR #19248)
TITLE: [Feature] Integrate Elastic NIXL-EP into SGLang (#19248)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+5/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-1); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+8/-0); python/sglang/srt/layers/moe/token_dispatcher/nixl.py (+465/-0); python/sglang/srt/layers/moe/utils.py (+4/-0); docs/advanced_features/expert_parallelism.md (+3/-2); docs/advanced_features/server_arguments.md (+1/-1); python/sglang/srt/batch_overlap/two_batch_overlap.py (+5/-0); python/sglang/srt/distributed/parallel_state.py (+61/-0); python/sglang/srt/distributed/utils.py (+36/-0); (+8 more)
LABELS: documentation, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Overview ⏎  ⏎ This PR introduces support for the NIXL-EP MoE backend in SGLang, enabling efficient expert parallelism through NVIDIA's [NIXL framework](https://github.com/ai-dynamo/nixl). This implementation leverages the elastic expert parallelism infrastructure being developed as part of the Elastic EP Support roadmap (PR #8961). ⏎  ⏎ ## What is NIXL-EP? ⏎  ⏎ [NIXL-EP](https://github.com/ai-dynamo/nixl/tree/main/examples/device/ep) is a complete i …[truncated]

### L1-2c03a5c6c7  (L1, 2026-03-12, sha 2c03a5c6c7fb, PR #20412)
TITLE: Fix global server args not set error in test_triton_moe_wna16.py (#20412)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/manual/test_triton_moe_wna16.py (+1/-2)
BODY: ---                                                                                                                                                                                                       ⏎   Title: Fix global server args not set error in test_triton_moe_wna16.py ⏎                                                                                                                                                                              …[truncated]

### L1-abc672e717  (L1, 2026-03-12, sha abc672e7177b, PR #20305)
TITLE: [Benchmark] use flashinfer bench_gpu_time instead of triton do_bench (#20305)
SOURCES: body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py (+5/-4); benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm_blackwell.py (+4/-3); benchmark/kernels/deepseek/benchmark_deepgemm_fp8_group_gemm.py (+4/-3); benchmark/kernels/elementwise/benchmark_concat_mla.py (+4/-4); benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py (+3/-2); benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py (+3/-2); benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py (+3/-2); benchmark/kernels/quantization/bench_fp4_quant.py (+5/-4); benchmark/kernels/quantization/bench_int8_quant.py (+5/-4); benchmark/kernels/scheduler_batch/benchmark_get_last_loc_triton.py (+6/-4); (+3 more)
LABELS: quant, run-ci
BODY: **Note:** Due to the number of modifications, this work is split into 2 PRs. This is PR 1 (benchmark/ folder only). ⏎  ⏎ **Related Issue:** #20212 ⏎  ⏎ ## Motivation ⏎  ⏎ Replace `triton.testing.do_bench` / `do_bench_cudagraph` with `flashinfer.testing.bench_gpu_time` in benchmark scripts for more accurate GPU kernel timing (see [b8zhong's issue](https://github.com/sgl-project/sglang/issues) and [FlashInfer docs](https://docs.flashinfer.ai/generated/fl …[truncated]

### L1-af2807e146  (L1, 2026-03-12, sha af2807e14616, PR #19710)
TITLE: [LoRA][I] Add MOE LoRA JIT alignment kernel and tests  (#19710)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/csrc/lora/moe_lora_align_kernel.cu (+618/-0); python/sglang/jit_kernel/moe_lora_align.py (+68/-0); python/sglang/jit_kernel/tests/test_moe_lora_align_block_size.py (+166/-0); python/sglang/jit_kernel/utils.py (+1/-0)
LABELS: lora, run-ci
BODY: Split this PR https://github.com/sgl-project/sglang/pull/14105 into 3 parts - Part I ⏎  ⏎ Add JIT-compiled CUDA kernels for MOE LoRA block size alignment: ⏎ - moe_lora_align.py: JIT wrapper for moe_lora_align_block_size ⏎ - moe_lora_align_kernel.cu: CUDA kernels for token alignment, sorting, and expert counting ⏎ - test_moe_lora_align_block_size.py: Unit tests for the alignment kernel ⏎  ⏎ Made-with: Cursor ⏎  ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications …[truncated]

### L1-70d4aabe42  (L1, 2026-03-12, sha 70d4aabe42b8, PR #12922)
TITLE: Add CLI args to conveniently support tuning more models (#12922)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmark/kernels/deepep/tuning_deepep.py (+8/-4)
LABELS: performance, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Add CLI args to conveniently support tuning more models. Refer to the latest deepep test script. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ add CLI args ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0eea80bc00  (L1, 2026-03-13, sha 0eea80bc001c, PR #20453)
TITLE: [AMD][MORI] Fix MTP crash with FP4/FP8 dispatch and add NEXTN dispatch env vars. (#20453)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+9/-7); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-0); python/sglang/srt/layers/moe/moe_runner/base.py (+1/-0); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+30/-17); docs/references/environment_variables.md (+2/-0); python/sglang/srt/models/deepseek_v2.py (+1/-0)
LABELS: documentation, deepseek, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 20602 (confirmed_revert, reason=unstated)
BODY: ## Motivation ⏎  ⏎ When using a quark MXFP4 model (e.g. `DeepSeek-R1-0528-MXFP4`) with EAGLE speculative decoding (MTP layer) and MORI FP4/FP8 dispatch enabled (`SGLANG_MORI_FP4_DISP=True` or `SGLANG_MORI_FP8_DISP=True`), the MTP layer crashes during CUDA graph capture: ⏎  ⏎ ``` ⏎ RuntimeError: Unsupported kernel config for moe heuristic dispatch ⏎ ``` ⏎  ⏎ The root cause is that the MTP layer's MoE weights are BF16 while the main model's MoE weights are …[truncated]

### L1-39008955ff  (L1, 2026-03-14, sha 39008955ffc5, PR #20602)
TITLE: Revert "[AMD][MORI] Fix MTP crash with FP4/FP8 dispatch and add NEXTN dispatch env vars." (#20602)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+7/-9); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-3); python/sglang/srt/layers/moe/moe_runner/base.py (+0/-1); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+17/-30); docs/references/environment_variables.md (+0/-2); python/sglang/srt/models/deepseek_v2.py (+0/-1)
LABELS: documentation, deepseek
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 20453 reason=unstated
BODY: Reverts sgl-project/sglang#20453

### L1-574dbe23b2  (L1, 2026-03-14, sha 574dbe23b25b, PR #18184)
TITLE: Add piecewise cuda graph for Qwen3-Next FP8 flashinfer_trtllm moe backend (#18184)
SOURCES: path_core, subject_keyword, symbol_pickaxe, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/flashinfer_trtllm_moe.py (+274/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+180/-4); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+123/-112)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Qwen3-next already has piecewise cuda graph support for NVFP4 and BF16, this PR adds pcg for Q3N FP8 on flashinfer_trtllm moe backend. Co-authored by @samuellees  ⏎  ⏎ nsys timeline shows cuda graph is enabled on FP8 flashinfer_trtllm moe layers ⏎ <img width="2613" height="1016" alt="image" src="https://github.com/user-attachments/assets/39465ab4-bfb6-4720-8b4a-39c338fd4092" /> ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Added custom op wrap …[truncated]

### L1-93afe15b43  (L1, 2026-03-14, sha 93afe15b4370, PR #20480)
TITLE: chore: bump flashinfer version to 0.6.6 (#20480)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the flashinfer version to `0.6.6` across all relevant files. ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎ - python/sglang/srt/utils/common.py ⏎ - scripts/ci/cuda/ci_install_dependency.sh ⏎  ⏎ 🤖 Generated with GitHub Actions
