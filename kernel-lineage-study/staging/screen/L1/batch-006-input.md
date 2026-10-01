### L1-2b8b9d8496  (L1, 2025-11-16, sha 2b8b9d8496af, PR #13388)
TITLE: [CI] use cached deepep installation in gb200 CI (#13388)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: scripts/ci/ci_install_deepep.sh (+3/-6)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d368c7451a  (L1, 2025-11-16, sha d368c7451a48, PR #12065)
TITLE: (1/n)support context parallel with deepseekv3.2-DSA (#12065)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: docs/advanced_features/server_arguments.md (+1/-0); docs/basic_usage/deepseek_v32.md (+20/-0); python/sglang/srt/distributed/device_communicators/pynccl.py (+28/-0); python/sglang/srt/distributed/parallel_state.py (+21/-0); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+221/-8); python/sglang/srt/layers/attention/nsa/utils.py (+305/-0); python/sglang/srt/layers/attention/nsa_backend.py (+28/-8); python/sglang/srt/layers/communicator_nsa_cp.py (+284/-0); python/sglang/srt/layers/dp_attention.py (+5/-1); python/sglang/srt/managers/schedule_policy.py (+7/-0); (+7 more)
LABELS: documentation, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Currently, under deepseek3.2-DSA, prefill-ttft of long text sequences takes a long time. Introducing context parallel can reduce ttft. ⏎ **Main design ideas：** ⏎ <img width="599" height="598" alt="image" src="https://github.com/user-attachments/assets/3827c03b-2448-43be-8f95-bfaf76b57a04" /> ⏎  ⏎ Taking TP=EP=4 and DP=2 as an example (CP_SIZE==ATTEN_TP_SIZE):  ⏎ Each DP accepts an independent request.  ⏎ Within each DP, after embedding …[truncated]

### L1-50691d7b49  (L1, 2025-11-16, sha 50691d7b4999, PR #13332)
TITLE: [opt kimi k2  2/n] apply kimi k2 thinking moe_fused_gate (#13332)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+6/-9)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ Follow https://github.com/sgl-project/sglang/pull/13287 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e389f91dec  (L1, 2025-11-17, sha e389f91decda, PR #13264)
TITLE: [NVIDIA] Fix broken fp8 MoE of deepseek v3 (#13264)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/nightly-test.yml (+22/-0); python/sglang/srt/layers/quantization/fp8.py (+1/-3); python/sglang/srt/models/deepseek_v2.py (+2/-0); test/srt/run_suite.py (+4/-1); test/srt/test_deepseek_r1_fp8_trtllm_backend.py (+88/-0)
LABELS: high priority, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ [This](https://github.com/sgl-project/sglang/pull/12543) breaks the fp8 moe of the deepseek v3, causing: ⏎ ``` ⏎   File "/scratch/repo/sglang/python/sglang/srt/layers/quantization/fp8.py", line 1225, in apply_with_router_logits                                                                                                                          ⏎     return trtllm_fp8_block_scale_moe(                                                …[truncated]

### L1-85ae508e8b  (L1, 2025-11-17, sha 85ae508e8b72, PR #13455)
TITLE: Add bfloat16 tuned fused moe config for Dpsk-MTP layer on B200 (#13455)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py (+18/-9)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎ When launching dpsk-r1-fp4 with MTP and TP4, the draft model will use bfloat16 fused moe triton kernels. ⏎ So it requires some tuning. ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ ```bash ⏎ # Launch ⏎ export SGLANG_ENABLE_SPEC_V2=1 ⏎ python3 -m sglang.launch_server \ ⏎     --model-path nvidia/DeepSeek-R1-0528-FP4-v2 \ ⏎     --trust-remote-code \ ⏎     --attention-backend trtllm_mla \ ⏎     --mo …[truncated]

### L1-9188feccca  (L1, 2025-11-17, sha 9188fecccad3, PR #13473)
TITLE: [model-gateway] use worker startup time out for worker registration (#13473)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/bindings/python/sglang_router/router.py (+1/-1); sgl-router/bindings/python/sglang_router/router_args.py (+2/-2); sgl-router/src/config/types.rs (+4/-4); sgl-router/src/core/workflow/steps/worker_registration.rs (+4/-3); sgl-router/src/main.rs (+1/-1)
LABELS: run-ci, model-gateway
BODY: ## Checklist

### L1-7e88b9c128  (L1, 2025-11-18, sha 7e88b9c128f8, PR #13157)
TITLE: [BugFix] Accuracy and function Issue when run ptpc quant model (#13157)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+29/-19)
LABELS: amd, run-ci
BODY: ## Modifications ⏎ Both function and precision have issues with original code。 ⏎ Modified the CompressedTensorsW8A8Fp8MoEMethod class, primarily revising the logic and interface calls for QuantizationStrategy.CHANNEL and aiter is active, to support unified fused_moe operator invocation under the ptpc quantization mode. ⏎ ## Accuracy Tests ⏎ Llama-4-Maverick-17B-128E-Instruct-FP8 : ⏎ ``` ⏎ SGLANG_USE_AITER=1 \ ⏎ CUDA_VISIBLE_DEVICES=4,5,6,7 \ ⏎ python3 -m …[truncated]

### L1-92ad2ff9ce  (L1, 2025-11-18, sha 92ad2ff9ce0e, PR #13489)
TITLE: Flashinfer TRTLLM-GEN-MoE + Qwen3 (#13489)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen3_moe.py (+2/-0); python/sglang/srt/server_args.py (+41/-1)
LABELS: run-ci
BODY: ``` ⏎ SGLANG_ENABLE_JIT_DEEPGEMM=false SGLANG_ENABLE_FLASHINFER_FP8_GEMM=1 python3 -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --moe-runner-backend flashinfer_trtllm --quantization fp8 ⏎ ``` ⏎  ⏎ This cmd needs to work, https://github.com/sgl-project/sglang/pull/12543 only make it work for Qwen2 MoE (qwen3 next) ⏎  ⏎ Also set better default, st: ⏎  ⏎ ``` ⏎ CUDA_VISIBLE_DEVICES=7 SGLANG_ENABLE_JIT_DEEPGEMM=false SGLANG_ENABLE_ …[truncated]

### L1-820e13c9c1  (L1, 2025-11-18, sha 820e13c9c135, PR #13374)
TITLE: [opt kimi k2 3/n] opt kimi_k2 moe_fused_gate kernel (#13374)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu (+130/-173)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎  ⏎ ## Kimi K2 Acc test ⏎  ⏎ <img width="1013" height="626" alt="图片" src="https://github.com/user-attachments/assets/8671e094-ed3b-432c-9689-a72386076653" /> ⏎  ⏎  ⏎ ### main branch ⏎  ⏎ ```shell ⏎ ➜  sglang git:(add_kimi_k2_moe_fused_gate) ✗ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 ⏎ /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is depre …[truncated]

### L1-bfaf0b8607  (L1, 2025-11-19, sha bfaf0b860727, PR #13570)
TITLE: chore: bump sgl-kernel version to 0.3.17.post2 (#13570)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.17.post2` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.17.post2` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are up …[truncated]

### L1-d4a4dcdfb3  (L1, 2025-11-19, sha d4a4dcdfb30f, PR #13567)
TITLE: [NPU] Adapt pr-gate for pr-test workflow & workflows refresh (#13567)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+4/-10); .github/labeler.yml (+7/-0); .github/workflows/pr-test-npu.yml (+47/-25); .github/workflows/release-docker-npu-nightly.yml (+1/-1); .github/workflows/release-docker-npu.yml (+2/-4); test/srt/run_suite.py (+6/-5)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Ever since #12710 introduced auto-labeling, there has been a huge gap that npu-related pull-requests can't be automatically recognized and correctly labled. Now we are filling this gap up with this pr. ⏎  ⏎ Plus, this one also adapts pr-gate mechanism that helps to improve npu ci efficiency. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Following #13436, adapts `pr-gate` on npu ci ⏎ - Adds `npu` auto-labler ⏎ - Fixes an issue that breaks docker i …[truncated]

### L1-e72cf13693  (L1, 2025-11-20, sha e72cf136932a, PR #13049)
TITLE: Support moe topk sigmoid kernel (#13049)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.topk_sigmoid
FILES: sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/csrc/common_extension.cc (+5/-0); sgl-kernel/csrc/common_extension_rocm.cc (+5/-0); sgl-kernel/csrc/moe/moe_topk_sigmoid_kernels.cu (+592/-0); sgl-kernel/include/sgl_kernel_ops.h (+7/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/python/sgl_kernel/moe.py (+26/-0); sgl-kernel/benchmark/bench_moe_topk_sigmoid.py (+171/-0); sgl-kernel/setup_rocm.py (+1/-0); sgl-kernel/tests/test_moe_topk_sigmoid.py (+183/-0)
LABELS: amd, sgl-kernel, ready-to-merge, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ This PR introduces a `topk_sigmoid` CUDA kernel to support MiniMax-M2 that require sigmoid-based expert routing. Our previously workaround was to use `grouped_topk` with `group_size=1`. ⏎  ⏎  ⏎ Previous: 8 kernels were launched. ⏎  ⏎ <img width="2042" height="848" alt="Clipboard_Screenshot_1762836857" src="https://github.com/user-attachments/assets/7c521fd0-7ed9-4037-8c77-7715f22702ce" /> ⏎  ⏎ Now: Only 1 kernel is launched. ⏎  ⏎ <img wid …[truncated]

### L1-6bc3062894  (L1, 2025-11-20, sha 6bc306289465, PR #13666)
TITLE: Fix launch of `Olmo3` (#13666)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/olmo2.py (+1/-0)
BODY: ## Motivation ⏎  ⏎ Olmo 3 is broken. Applies to non thinking too. ⏎  ⏎ Without the fix ⏎ ``` ⏎ python -m sglang.launch_server --model-path allenai/Olmo-3-32B-Think --mem-fraction-static 0.8 ⏎ [2025-11-20 17:49:37] INFO utils.py:148: Note: detected 248 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable. ⏎ [2025-11-20 17:49:37] INFO utils.py:151: Note: NumExpr detected 248 cores but "NUMEXPR_MAX_THREADS" not se …[truncated]

### L1-db2d362d04  (L1, 2025-11-20, sha db2d362d0471, PR #12672)
TITLE: [NVIDIA] Add cutedsl e2e test to GB200 CI (#12672)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/srt/run_suite.py (+1/-3); test/srt/test_deepseek_v3_cutedsl_4gpu.py (+0/-0)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Enable the e2e test with the cutedsl moe to the GB200 CI. Previously the test failed in the CI because the deepep low latency mode requires some "priveleged" access to the resources. Tracked in this [issue](https://github.com/sgl-project/sglang/issues/12533). The issue is fixed now. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6d0e0b9bfc  (L1, 2025-11-21, sha 6d0e0b9bfc15, PR #13327)
TITLE: [11/N] MoE Refactor: Simplifying SBO Implementation with Dispatcher Hooks (#13327)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.triton, L1.runner.deep_gemm, L1.ep.layer, L1.ep.deepep_dispatcher, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+10/-23); python/sglang/srt/layers/moe/fused_moe_native.py (+10/-4); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+17/-4); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+4/-1); python/sglang/srt/layers/moe/moe_runner/triton.py (+4/-1); python/sglang/srt/layers/moe/token_dispatcher/base.py (+200/-3); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+53/-20); python/sglang/srt/layers/moe/token_dispatcher/mooncake.py (+12/-11); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+83/-16); python/sglang/srt/layers/quantization/modelopt_quant.py (+14/-54); (+5 more)
LABELS: quant, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR introduces dispatcher hooks to avoid modifying quantization methods and moe modules when implementing SBO. This change streamlines the MoE layer interfaces and enhances modularity, allowing for more dynamic control over SBO behavior through registered hooks. Some minor related refactors are summarized by @gemini-code-assist below. @Fridge003 helped verifying the correctness and efficiency of this PR. ⏎  ⏎ ## Modificatio …[truncated]

### L1-eff7df6d0a  (L1, 2025-11-21, sha eff7df6d0a49, PR #13705)
TITLE: [AMD] Enable fused shared expert append and flatten quant for fp8 deepseekR1 model (#13705)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+71/-0); python/sglang/srt/layers/moe/topk.py (+11/-25); python/sglang/srt/models/deepseek_v2.py (+10/-1)
LABELS: deepseek, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎ Co-author : @yctseng0211  ⏎  ⏎ This PR introduces two performance improvements for DeepseekR1-0528 (fp8): ⏎  ⏎ 1. Support aiter fused_flatten_fp8_group_quant for fp8 DeepSeek model ⏎ Adds a fused path for fp8 output projection input using fused_flatten_fp8_group_quant, extending parity with existing MXFP4 support. ⏎  ⏎ 2. Fused shared-expert routing in MoE ⏎ The original PyTorch implementation used multiple kernels (arange, full, cat) to a …[truncated]

### L1-85ffce30af  (L1, 2025-11-21, sha 85ffce30af54, PR #13466)
TITLE: [Piecewise CUDA Graph] Support Kimi-K2 (non-Thinking) (#13466)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+23/-0)
LABELS: run-ci, piecewise-cuda-graph
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Currently, this only supports `moonshotai/Kimi-K2-Instruct-0905` (there are some additional complication w/ some torch.cuda non tensor ops in the prefill stage that I will remove later, so support regular Kimi for now). ⏎  ⏎ ``` ⏎ python -m sglang.launch_server --model-path moonshotai/Kimi-K2-Instruct-0905 --tp 8 --trust-remote-code --tool-call-parser kimi_k2 --enable-piecewise-cuda-graph --piecewise-cuda-graph-max-tokens 8192 ⏎ ``` ⏎  ⏎ The KV cache s …[truncated]

### L1-45c572c58f  (L1, 2025-11-21, sha 45c572c58ff6, PR #12949)
TITLE: Support torch 12.9 + DeepEP by removing custom nvshmem (#12949)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+4/-27)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ I can confirm this works (remove custom nvshmem) ⏎  ⏎ ``` ⏎ (cd /home/sgl-tom && enroot import docker://lmsysorg/sglang:nightly-dev-cu13-20251106-bb6a21cd) ⏎  ⏎ srun \ ⏎     --kill-on-bad-exit \ ⏎     --mpi=pmix \ ⏎     --container-image "/home/sgl-tom/lmsysorg+sglang+nightly-dev-cu13-20251106-bb6a21cd.sqsh" \ ⏎     --nodes="1" \ ⏎     --ntasks="1" \ ⏎     --gpus-per-node="4" \ ⏎     --pty \ ⏎     /bin/bash ⏎  ⏎ rm -rf /sgl-workspace/nvshmem …[truncated]

### L1-fb04d43428  (L1, 2025-11-21, sha fb04d4342877, PR #13596)
TITLE: [kimi k2 thinking] Avoid useless torch.zeros_  (#13596)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.marlin
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+239/-0); sgl-kernel/python/sgl_kernel/fused_moe.py (+0/-232); python/sglang/srt/layers/quantization/awq.py (+4/-6); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+3/-12); python/sglang/srt/layers/quantization/gptq.py (+4/-4); python/sglang/test/test_marlin_moe.py (+1/-1); sgl-kernel/python/sgl_kernel/__init__.py (+1/-1)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎  ⏎ ```shell ⏎ ➜  sglang git:(overlap_fused_marlin_moe_silu_and_mul_zeros) ✗ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8                                                                 ⏎ /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please repo …[truncated]

### L1-bfcf15a129  (L1, 2025-11-21, sha bfcf15a129fd, PR #13587)
TITLE: [opt kimi k2 4 / n] Delete useless pad kernel in sgl_moe_align_block_size (#13587)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py (+1/-6)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ ```shell ⏎ ➜  sglang git:(main) ✗ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 ⏎ /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you. ⏎   import pynvml  # type: ignor …[truncated]

### L1-1b48e1b974  (L1, 2025-11-21, sha 1b48e1b97484, PR #12690)
TITLE: Feat/nemotron nano v3 support (#12690)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.runner.framework, L1.runner.triton, L1.runner.deep_gemm, L1.runner.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=1856,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=1856,device_name=NVIDIA_L40S.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=928,device_name=NVIDIA_L40S.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+20/-3); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+15/-6); python/sglang/srt/layers/moe/moe_runner/base.py (+3/-9); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+1/-8); python/sglang/srt/layers/moe/moe_runner/triton.py (+2/-0); python/sglang/srt/layers/moe/moe_runner/triton_kernels.py (+4/-0); benchmark/kernels/fused_moe_triton/common_utils.py (+1/-1); (+3 more)
LABELS: performance, quant, run-ci
BODY: ## Motivation ⏎  ⏎ Add support for upcoming NVIDIA Nemotron v3 models. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Add an MoE layer to the NemotronH modeling code. ⏎ Add support for un-gated MoE in the triton codepath. ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a56f770277  (L1, 2025-11-21, sha a56f770277c7, PR #13484)
TITLE: Fix global scaling factor loading hang (#13484)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-1); python/sglang/srt/eplb/expert_location.py (+15/-9)
LABELS: quant, run-ci
BODY: cc @Fridge003  ⏎ The root-cause of hang is that each thread reading and write to 256 slots. There will be race condition. ⏎ ref. https://github.com/sgl-project/sglang/pull/13348/commits ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-589d9ad55b  (L1, 2025-11-21, sha 589d9ad55bd7, PR #13647)
TITLE: [NPU] chore: bump to CANN 8.3.RC1 and Pytorch 2.8.0 (#13647)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+26/-25); .github/workflows/pr-test-npu.yml (+4/-4); .github/workflows/release-docker-npu-nightly.yml (+2/-2); .github/workflows/release-docker-npu.yml (+2/-2); docs/platforms/ascend_npu.md (+7/-24); python/pyproject_other.toml (+1/-0); python/sglang/check_env.py (+7/-1); python/sglang/srt/layers/attention/ascend_backend.py (+1/-1); python/sglang/test/test_utils.py (+2/-2); scripts/ci/npu_ci_install_dependency.sh (+14/-22)
LABELS: documentation, dependencies, npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Bump CANN version and PTA version for NPU backend ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ n/a ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ n/a ⏎  ⏎ ## Checklist

### L1-59b4d7f8d6  (L1, 2025-11-21, sha 59b4d7f8d61c, PR #13746)
TITLE: Fix B200 Nightly tests and move one manual test back to unit test to prevent the same issue (#13746)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-1); test/srt/run_suite.py (+1/-0); test/srt/test_triton_fused_moe.py (+1/-1)
LABELS: run-ci
BODY: Fixing error: ⏎  ⏎   File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl ⏎     return forward_call(*args, **kwargs) ⏎   File "/actions-runner/_work/sglang/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py", line 1036, in forward ⏎     dispatch_output=StandardDispatchOutput( ⏎ TypeError: StandardDispatchOutput.__new__() missing 1 required positional argument: 'hidden_states_scale' ⏎  ⏎ Verified i …[truncated]

### L1-a92afb00c6  (L1, 2025-11-22, sha a92afb00c675, PR #12759)
TITLE: [Ascend] support Kimi-K2-Thinking (#12759)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+62/-130); python/sglang/srt/layers/quantization/w8a8_int8.py (+480/-39); python/sglang/srt/model_executor/model_runner.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+6/-0)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Kimi-K2-Think model Day 0 support on SGLang on Ascend NPU backend. ⏎  ⏎ Ascend has completed the adaptation of the INT4 Weight (A16W4, pergroup=32) quantization format for the K2 Think model, balancing inference speed and precision. Meanwhile, the supporting GMM (GroupedMatmul) kernel has been fully open-sourced, providing a more flexible engineering foundation for low-bit inference of large models. This optimization not only reduc …[truncated]

### L1-618ca23802  (L1, 2025-11-23, sha 618ca2380293, PR #13687)
TITLE: [Deepseek] Refactor deepseek server_args _handle_model_specific_adjustments (#13687)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+79/-74)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Deepseek v3.2 shares the same MoE architecture with other Deepseek MoE models, so that part of the logics can be reused for all `DeepseekV3ForCausalLM` models. ⏎  ⏎ ## Modifications ⏎  ⏎ Refactor the Deepseek part of `_handle_model_specific_adjustments` so that we can share the MoE adjustments across `DeepseekV3ForCausalLM` models. ⏎ `--moe-runner-backend flashinfer_trtllm  --quantization fp8` will be set automatically for DS models o …[truncated]

### L1-2892265d4c  (L1, 2025-11-23, sha 2892265d4cd8, PR #13815)
TITLE: Tune fp8_w8a8 fused triton moe for GLM-4.6-FP8 (#13815)
SOURCES: path_config_only, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=160,N=384,device_name=NVIDIA_B200,dtype=fp8_w8a8.json (+146/-0)
LABELS: high priority
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎  ⏎ Before this pr, 58 token/s (bs=1). ⏎ After this pr, 68 token/s (bs=1). ⏎  ⏎ ## Modifications ⏎ Before, ⏎  ⏎ <img width="640" height="549" alt="image" src="https://github.com/user-attachments/assets/568dbdcb-ef91-499a-8bc4-5c6667cc3922" /> ⏎  ⏎ After, ⏎ <img width="392" height="345" alt="image" src="https://github.com/user-attachments/assets/08130e8a-a31a-449f-bf8f-cb09aa598b30" /> ⏎  ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Pr …[truncated]

### L1-4683e244fe  (L1, 2025-11-23, sha 4683e244fe62, PR #13601)
TITLE: [1/2] Refactor DeepGeem requant for FP8 Linear on Blackwell  (#13601)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+29/-0); python/sglang/srt/models/deepseek_nextn.py (+0/-1); python/sglang/srt/models/deepseek_v2.py (+9/-77); python/sglang/test/test_block_fp8_deep_gemm_blackwell.py (+1/-1)
LABELS: high priority, deepseek, run-ci
BODY: ## Motivation ⏎ Co-Author: @fy1214  ⏎ Based on pr: https://github.com/sgl-project/sglang/pull/13067 ⏎  ⏎ Refactor the messy codes related to deepgemm requant, also fix the bug in #12878 ⏎ Refactor for MoE requant will be left to the next PR ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ To run DeepGemm on Blackwell, the input scale factor needs to be requantized to ue8m0. The prior codes only consider this requantization for deepseek model classes, and the codes are quit …[truncated]

### L1-b964ce61d6  (L1, 2025-11-23, sha b964ce61d6ee, PR #13787)
TITLE: [DeepEP] Add SGLANG_DEEPEP_BF16_DISPATCH env var in Normal mode (#13787)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ DeepEP Normal mode (prefill) ignores the `SGLANG_DEEPEP_BF16_DISPATCH` environment variable and always quantizes activations to FP8, even when the flag is set to use BF16 dispatch. ⏎  ⏎ This breaks Marlin MoE quantization, which requires FP16/BF16 activations and does not support FP8 activations. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9ea1953331  (L1, 2025-11-23, sha 9ea1953331c1, PR #13820)
TITLE: [Doc] Refine fused_moe_triton configs doc (#13820)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/README.md (+36/-11)
LABELS: documentation, run-ci
BODY: ## Motivation ⏎  ⏎ preview: ⏎  ⏎ <img width="2320" height="1444" alt="图片" src="https://github.com/user-attachments/assets/dbef419f-02fc-4d13-8b57-9bcd65ffca60" /> ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-04b52fa8d6  (L1, 2025-11-23, sha 04b52fa8d6f8, PR #13751)
TITLE: [chore]Upgrade flashinfer to 0.5.3 (#13751)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
ISSUES: #13748 [Feature] Upgrade flashinfer to 0.5.3
BODY: ## Motivation ⏎  ⏎ Close #13748 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9535015d05  (L1, 2025-11-24, sha 9535015d055a, PR #10027)
TITLE: [Perf] Optimize DeepSeek-R1 w4afp8 glue kernels (#10027)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.ep.layer, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+21/-17); python/sglang/srt/layers/moe/ep_moe/kernels.py (+227/-54); python/sglang/srt/layers/quantization/w4afp8.py (+5/-6)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR improves DeepSeek-R1 w4afp8 TP8/EP8 ITL performance. It is motivated by the following profiling trace, obtained by running DeepSeek-R1 w4afp8 TP8 with a concurrency level (i.e., batch size) of 4: ⏎  ⏎ <img width="1676" height="133" alt="image" src="https://github.com/user-attachments/assets/1e745037-30c7-4996-becc-763d950efa21" /> ⏎  ⏎ The trace reveals several performance gaps: ⏎ - The grouped GEMM kernel is a major bottl …[truncated]

### L1-94216a9cc4  (L1, 2025-11-24, sha 94216a9cc45d, PR #13853)
TITLE: Fix quantized moe checker fail for Qwen3 dense fp8 model (#13853)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/model_runner.py (+5/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f56b9b42e6  (L1, 2025-11-24, sha f56b9b42e668, PR #13829)
TITLE: [Bugfix] Add jit kernel files in packaging (#13829)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+3/-0)
LABELS: high priority, dependencies
BODY: ## Motivation ⏎  ⏎  ⏎ PR https://github.com/sgl-project/sglang/pull/13764 is missing jit kernel files when packaging. This PR fix it. ⏎  ⏎ The reason why github ci didn't fail is because there were cache jit kernel files un-deleted in the folder. So the new CI happened to reuse them. ⏎  ⏎ ``` ⏎ $SGLANG_VLM_CACHE_SIZE_MB=2048 python -m sglang.launch_server --model-path /home/admin/Qwen3-VL-2B-Thinking --host 0.0.0.0 --port 8188 --trust-remote-code --tp-si …[truncated]

### L1-fafaa2ccea  (L1, 2025-11-24, sha fafaa2cceace, PR #13864)
TITLE: [BugFix] fix outplace_fused_experts missing is_gated (#13864)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎ This [change](https://github.com/sgl-project/sglang/pull/12690/files) recently introduced is_gated, but missed passing it for `outplace_fused_experts` ⏎  ⏎ `python3 -m sglang.launch_server --model /shared/public/elr-models/xai-org/grok-2/ --tokenizer-path /shared/public/elr-models/xai-org/grok-2/tokenizer.tok.json --tp 8 --quantization fp8 --attention-backend triton` ⏎  ⏎ Before: ⏎  ⏎ ``` ⏎   File "/home/jobuser/zminglei/sglang/venv/lib/p …[truncated]

### L1-bf10869203  (L1, 2025-11-24, sha bf10869203b9, PR #13783)
TITLE: [Doc] Add an Introduction to Expert Parallelism (#13783)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docs/advanced_features/expert_parallelism.md (+141/-0)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4b45d556a7  (L1, 2025-11-24, sha 4b45d556a7e6, PR #13786)
TITLE: Overlap glm moe gemms in two cuda streams (#13786)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/glm4_moe.py (+47/-3)
LABELS: high priority, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Modifications ⏎  ⏎ Before this pr ⏎  ⏎ <img width="1008" height="415" alt="image" src="https://github.com/user-attachments/assets/ea6ab7f3-a3d1-420d-8aa8-14a9da9824a1" /> ⏎  ⏎ After this pr ⏎ <img width="785" height="469" alt="image" src="https://github.com/user-attachments/assets/da36c4b9-4891-4c0f-8915-94ce6aab0896" /> ⏎  ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ``` ⏎ /sgl-workspace/sglang# python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions …[truncated]

### L1-b0a26ba624  (L1, 2025-11-24, sha b0a26ba6249f, PR #10275)
TITLE: Add support for bf16 x bf16 cutlass fused MoE (#10275)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-7); python/sglang/srt/layers/moe/utils.py (+4/-0); python/sglang/srt/layers/quantization/unquant.py (+34/-1); python/sglang/srt/server_args.py (+3/-2); python/sglang/test/test_cutlass_w16a16_moe.py (+118/-0)
LABELS: high priority, quant, blackwell, run-ci, nvidia, hopper
BODY: ## Motivation ⏎  ⏎ Add efficient fused MoE layer for bf16 weights and bf16 activations. ⏎ Usage: Add `--moe-runner-backend flashinfer_cutlass` server arg to use the cutlass MoE backend. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ # python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319 ⏎ Accuracy: 0.933 ⏎ Invalid: 0.000 ⏎ Latency: 75.776 s ⏎ Output throughput: 2424.322 token/s ⏎ ``` ⏎  ⏎ ## Benchmarking and Prof …[truncated]

### L1-760c20b360  (L1, 2025-11-25, sha 760c20b36041, PR #13848)
TITLE: update flashinfer_cubin==0.5.3 (#13848)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎ flashinfer_cubin 0.5.3 ready in pypi, ref: https://github.com/flashinfer-ai/flashinfer/issues/2133#issuecomment-3569604925 ⏎ @Fridge003  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-36b1bcd242  (L1, 2025-11-25, sha 36b1bcd2424d, PR #12969)
TITLE: [chore] update torch version to 2.9 (#12969)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: scripts/ci/ci_install_deepep.sh (+2/-29); .github/workflows/pr-test-pd-router.yml (+1/-1); .github/workflows/pr-test.yml (+0/-1); docker/Dockerfile (+1/-13); python/pyproject.toml (+2/-2); python/sglang/multimodal_gen/test/server/testcase_configs.py (+27/-31); scripts/ci/ci_install_dependency.sh (+2/-0); sgl-kernel/CMakeLists.txt (+1/-1); sgl-kernel/build.sh (+4/-7); test/manual/ep/test_mooncake_ep_small.py (+0/-0); (+3 more)
LABELS: high priority, quant, dependencies, lora, sgl-kernel, run-ci, diffusion, format
BODY: ## Motivation ⏎  ⏎  ⏎ As titled, update torch version to 2.9 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c53e729d45  (L1, 2025-11-25, sha c53e729d4535, PR #13951)
TITLE: chore: bump sgl-kernel version to 0.3.18.post1 (#13951)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.18.post1` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.18.post1` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are up …[truncated]

### L1-432ecf841e  (L1, 2025-11-25, sha 432ecf841e26, PR #12078)
TITLE: [Ascend] qwen optimization (#12078)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+137/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+12/-0); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+2/-0); python/sglang/srt/layers/moe/token_dispatcher/fuseep.py (+97/-0); python/sglang/srt/layers/moe/topk.py (+5/-5); python/sglang/srt/layers/moe/utils.py (+4/-0); docker/npu.Dockerfile (+9/-10); python/sglang/srt/layers/attention/ascend_backend.py (+85/-45); python/sglang/srt/layers/quantization/w8a8_int8.py (+29/-11); python/sglang/srt/layers/rotary_embedding.py (+13/-0); (+6 more)
LABELS: npu, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ related to #10337  ⏎  ⏎ ## Modifications ⏎ -bugfix: ⏎  ⏎ 1.memory bugfix(w8a8_int8.py): in previous code, both layer.w13_weight and layer.w2_weight occupied double memory. now we solve it. ⏎ 2.Cache Management Operation(CMO) bugfix(common.py):in some circumstances(BS=1,2), deadlock situations may occur due to issues with stream sync and waiting. now we solve it. ⏎ 3.eplb bugfix:The eplb index operator is also introduced when the map path  …[truncated]

### L1-5e70880e64  (L1, 2025-11-26, sha 5e70880e64e2, PR #13766)
TITLE: [model-gateway] Add PostgreSQL support to binding (#13766)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/bindings/python/sglang_router/router.py (+2/-0); docs/advanced_features/router.md (+1/-0); sgl-router/README.md (+8/-0); sgl-router/bindings/python/sglang_router/router_args.py (+15/-0)
LABELS: documentation, run-ci, model-gateway
BODY: ## Motivation ⏎  ⏎ #13758 - Postgres connector supported but not added into binding. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Included the postgres connector related params. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ - start sgl-router use this pararm ⏎   ``` ⏎   python3 -m sglang_router.launch_server \ ⏎     --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \ ⏎     --host 0.0.0.0 \ ⏎     --port 30000 \ ⏎     --router-backend openai \ ⏎     --router-history-backend postgres \ ⏎     --router-p …[truncated]

### L1-e0e8a99630  (L1, 2025-11-26, sha e0e8a9963043, PR #13892)
TITLE: fix: correct usage of minimax-m2 deepep moe forward (#13892)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/minimax_m2.py (+3/-7)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0a186924ba  (L1, 2025-11-26, sha 0a186924ba8e, PR #13990)
TITLE: [Code sync] Fix registration of some ops in grok & Fix oss sync scripts (#13990)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+29/-24); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/layers/elementwise.py (+24/-3); python/sglang/srt/layers/quantization/modelopt_quant.py (+2/-1); python/sglang/srt/utils/common.py (+15/-0); scripts/code_sync/copy_from_oss.py (+4/-1); scripts/code_sync/copy_to_oss.py (+4/-1)
LABELS: quant, run-ci
BODY: 

### L1-91e8dc371a  (L1, 2025-11-26, sha 91e8dc371a8b, PR #13761)
TITLE: [Feat][NVFP4] Enable NVFP4 MoE for Qwen series models (eg. Qwen3-Next) #13761 (#13761)
SOURCES: path_core, corpus:kernel-correctness-cases(introducing)
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+13/-3); python/sglang/srt/server_args.py (+2/-2); test/nightly/test_qwen3_fp4_trtllm_gen_moe.py (+61/-0); test/run_suite_nightly.py (+1/-0)
LABELS: blackwell, run-ci, nvidia
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:922756aaa1 (fix PR 14350) || deep-study performance PR (precision_format)
BODY: ## PR Dependency ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ Enable NVFP4 MoE for Qwen series models (eg. Qwen3-Next) on Blackwell GPUs ⏎  ⏎ ## TODO ⏎  ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ ```bash ⏎ # Qwen3-Next: NVFP4 linear + NVFP4 MoE + FP8 Attention ⏎ export SGL_ENABLE_JIT_DEEPGEMM=false ⏎ python3 -m sglang.launch_server --model-path qwen3-next-80b-a3b-instruct-nvfp4-all --chunked-prefill-size 16384 --max-prefill-tokens 16384 --max-running-requests 512 --tp-size 4 --ep-size 4 --mem- …[truncated]

### L1-21b0582d4b  (L1, 2025-11-26, sha 21b0582d4bb0, PR #12588)
TITLE: [feature] Initial block diffusion language model support (#12588)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/dllm/algorithm/__init__.py (+39/-0); python/sglang/srt/dllm/algorithm/base.py (+18/-0); python/sglang/srt/dllm/algorithm/low_confidence.py (+59/-0); python/sglang/srt/dllm/config.py (+40/-0); python/sglang/srt/layers/attention/flashinfer_backend.py (+8/-1); python/sglang/srt/layers/logits_processor.py (+18/-1); python/sglang/srt/managers/schedule_batch.py (+42/-1); python/sglang/srt/managers/scheduler.py (+18/-1); python/sglang/srt/managers/scheduler_output_processor_mixin.py (+30/-0); python/sglang/srt/managers/tp_worker.py (+17/-0); (+3 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The current implementation focuses only on the minimal functionalities required for dLLM serving： ⏎  ⏎   ⏎ Those feature will be implemented in seperate PRs in the future: ⏎  ⏎  ⏎ ##  Our From-Scratch Diffusion Language Models ⏎ LLaDA2.0-flash-preview: https://huggingface.co/inclusionAI/LLaDA2.0-flash-preview ⏎ LLaDA2.0-mini-preview: https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview ⏎  ⏎ ## Contributor ⏎ **Tiwei Bie tiwei.btw@antgroup. …[truncated]

### L1-6330d6641b  (L1, 2025-11-26, sha 6330d6641ba3, PR #14028)
TITLE: Fix flashinfer cutlass MoE output shape for non-FP4-packed inputs (#14028)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/modelopt_quant.py (+5/-1)
LABELS: quant, run-ci
BODY: ## Summary ⏎  ⏎ Fixes `nightly-test-perf-4-gpu-b200` failure caused by `ValueError: Invalid shape of output: expected (512, 7168), got torch.Size([512, 14336])` when starting DeepSeek-V3-FP4 with flashinfer_cutlass MoE backend. ⏎  ⏎ ## Root Cause ⏎  ⏎ PR #13327 introduced a regression in `ModelOptNvFp4FusedMoEMethod.apply()` when refactoring the MoE dispatcher implementation. The output tensor allocation was changed from: ⏎  ⏎ ```python ⏎ symm_output = torch.empty …[truncated]

### L1-b12c9e5c0a  (L1, 2025-11-26, sha b12c9e5c0ab2, PR #14033)
TITLE: Fix installation for nvidia-nvshmem-cu12 (#14033)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: scripts/ci/ci_install_deepep.sh (+0/-3); scripts/ci/ci_install_dependency.sh (+4/-0)
LABELS: quant
BODY: Force reinstall nvidia-nvshmem-cu12 version 3.4.5. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-7ab548ef64  (L1, 2025-11-27, sha 7ab548ef641b, PR #13960)
TITLE: [2/2] Refactor DeepGeem requant for FP8 FusedMoE on Blackwell (#13960)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+27/-4); python/sglang/srt/models/deepseek_v2.py (+0/-35); python/sglang/srt/server_args.py (+5/-0)
LABELS: deepseek, run-ci
ISSUES: #13680 [Bug] Lower gsm8k accuracy in Deepseek V3.2 with moe_backend = deep_gemm
BODY: ## Motivation ⏎  ⏎ Following #13601 Close #13680 ⏎ Currently when `--moe-runner-backend deep_gemm` is applied, the accuracy will drop to 0 for any non-deepseek model. It's caused by the missing requantization process that converts weight/scale to ue8m0 format. This requantization is required for DeepGemm kernels, since DeepGemm takes ue8m0 input datatype. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ All cases are tested on B200 with gsm8k …[truncated]

### L1-63b056213f  (L1, 2025-11-27, sha 63b056213fc4, PR #13946)
TITLE: Remove disused B300 Dockerfile (#13946)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/b300.Dockerfile (+0/-55)
BODY: ## Motivation ⏎  ⏎ I think this B300 Dockerfile can be removed already since it's not being used anymore and just causes confusion like in #13900? ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-21af8e73ad  (L1, 2025-11-27, sha 21af8e73ad57, PR #14048)
TITLE: Super tiny add comments to SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK (#14048)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+1/-0)
BODY: ## Motivation ⏎  ⏎ I see at least twice from different person dnk this affects mem a lot, thus add a tiny comment ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e12c78aab6  (L1, 2025-11-28, sha e12c78aab6da, PR #14036)
TITLE: [sgl-kernel][1/2] Fused qk_norm_rope for Qwen3-MoE (#14036)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/moe/fused_qknorm_rope_kernel.cu (+408/-0); sgl-kernel/python/sgl_kernel/moe.py (+36/-0); sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/csrc/common_extension.cc (+8/-0); sgl-kernel/include/sgl_kernel_ops.h (+17/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/tests/test_fused_qk_norm_rope.py (+225/-0)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎  ⏎ This is the CUDA kernel to fuse Q K normalization and apply_rope_embedding. ⏎ More details refer to https://github.com/sgl-project/sglang/pull/13998. ⏎  ⏎ ``` ⏎ ➜  sglang_dev git:(fused_qk_norm_rope) ✗ python -m pytest ./sgl-kernel/tests/test_fused_qk_norm_rope.py ⏎ =============================================================================================================== test session starts ===================================== …[truncated]

### L1-a102a0507a  (L1, 2025-11-28, sha a102a0507aff, PR #14111)
TITLE: Disable Deepep 2 GPU tests (#14111)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: test/srt/ep/test_moe_ep.py (+2/-0); test/srt/run_suite.py (+3/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0fe74af563  (L1, 2025-11-28, sha 0fe74af563cb, PR #14113)
TITLE: Remove incorrect deep_gemm assertions from server_args.py (#14113)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+0/-5); test/srt/ep/test_moe_ep.py (+0/-2); test/srt/run_suite.py (+1/-3)
LABELS: run-ci
BODY: Removed assertions related to DeepGemm MoE runner backend. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3339c81072  (L1, 2025-11-29, sha 3339c8107292, PR #14135)
TITLE: fix RuntimeError: RMSNorm failed with error code an illegal memory access was encountered (#14135)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ issue: https://github.com/sgl-project/sglang/issues/14120 ⏎  ⏎ https://github.com/sgl-project/sglang/commit/91e8dc371a8b631475cb620aae35d5628ed27155#diff-d7e3ce3bc8e85e0a99fbcde0049eedd948dcf84167802cf3342cc02d0beeab1aL1121 ⏎  ⏎ After fix, the test/srt/test_deepseek_v3_fp4_4gpu.py test could pass. ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9872a677b4  (L1, 2025-11-30, sha 9872a677b48e, PR #14166)
TITLE: [ci]fix deepep import error on H20 action (#14166)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎ fix https://github.com/sgl-project/sglang/actions/runs/19791424706/job/56706686018 ⏎  ⏎  ⏎ ## Modifications ⏎ <img width="1404" height="334" alt="image" src="https://github.com/user-attachments/assets/9000694b-d9fc-4820-8f8a-2a1a88754067" /> ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-7b03cc6482  (L1, 2025-11-30, sha 7b03cc6482ba, PR #14065)
TITLE: [Minor]Raise Error when deepep num dispatch token per rank is smaller than cuda graph bs (#14065)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+10/-0)
LABELS: run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 14171 (confirmed_revert, reason=unstated)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f1115cf58d  (L1, 2025-11-30, sha f1115cf58db5, PR #14171)
TITLE: Revert "[Minor]Raise Error when deepep num dispatch token per rank is smaller than cuda graph bs" (#14171)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+0/-10)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 14065 reason=unstated
BODY: Reverts sgl-project/sglang#14065

### L1-bd0e690857  (L1, 2025-11-30, sha bd0e69085760, PR #12181)
TITLE: [Feature] Enable PTPC FP8 for compressed tensors moe (aiter kernel) (#12181)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py (+23/-11); python/sglang/srt/layers/quantization/fp8_utils.py (+35/-0); python/sglang/srt/models/deepseek_v2.py (+15/-7)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Deepseek-R1 PTPC FP8 model for llm-compressor quantized model can be supported on MI308. ⏎  ⏎ ## Accuracy Tests ⏎ Here is my script. ⏎ ```shell ⏎ model=/data/models/Deepseek-R1-FP8-Dynamic ⏎ TP=8 ⏎ EP=1 ⏎  ⏎ python3 -m sglang.launch_server \ ⏎     --model-path ${model} \ ⏎     --host localhost \ ⏎     --port 9000 \ ⏎     --tp-size ${TP} \ ⏎     --ep-size ${EP} \ ⏎     --trust-remote-code \ ⏎     --chunked-prefill-size 16384 \ ⏎     --mem-fraction …[truncated]

### L1-41b7aab848  (L1, 2025-12-01, sha 41b7aab848e2, PR #14152)
TITLE: Disable Deepep 8 GPU tests (#14152)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+27/-27)
LABELS: run-ci
BODY: 

### L1-d9dca28247  (L1, 2025-12-01, sha d9dca28247a3, PR #)
TITLE: Update pr-test.yml to fix unknown job name deepep-8-gpu
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: 
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L1-982db4ebac  (L1, 2025-12-01, sha 982db4ebac26, PR #13873)
TITLE: Feat: GLM-4.6 supports shared experts fusion (#13873)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.runner.triton
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py (+19/-2); python/sglang/srt/layers/moe/moe_runner/triton.py (+1/-0); benchmark/kernels/fused_moe_triton/common_utils.py (+7/-3); python/sglang/srt/models/glm4_moe.py (+74/-19); python/sglang/srt/models/glm4_moe_nextn.py (+4/-0)
LABELS: quant, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Hi from [novita.ai](https://novita.ai/) team 👋 ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎ Fuse shared experts with routed experts for better performance. ⏎  ⏎ ## Modifications ⏎ The changes involve modifying the model initialization to treat the shared expert as a regular expert, updating weight loading to remap shared expert weights. ⏎  ⏎ ## Accuracy Tests ⏎ ``` ⏎ python3 -m sglang.test.run_eval --port 30000 --eval-name mmlu --num-examples 200 ⏎ ChatCompletionSampler initia …[truncated]

### L1-eb5008846a  (L1, 2025-12-01, sha eb5008846ae9, PR #14247)
TITLE: [CI] Fix test_deepep_large.py (#14247)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+28/-28); test/srt/ep/test_deepep_large.py (+10/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Fix hanging caused by DeepGemm precompilation ⏎ https://github.com/sgl-project/sglang/actions/runs/19809465132/job/56759305082 ⏎  ⏎ After appending the SGLANG_DEEPGEMM_JIT_PRECOMPILE=0 flag, the first subtest can pass ⏎ The second subtest will fail due to jit latency (but it can pass on local machine), thus skipped ⏎ https://github.com/sgl-project/sglang/actions/runs/19837242549/job/56839051765?pr=14247 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎  …[truncated]

### L1-02af51e4fc  (L1, 2025-12-01, sha 02af51e4fc3e, PR #13794)
TITLE: Support fp4 fp8 non gated moe (#13794)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-6); python/sglang/srt/layers/quantization/modelopt_quant.py (+173/-29); python/sglang/srt/server_args.py (+22/-3)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎ Add support for FP8 and NVFP4 quantization for the upcoming NVIDIA Nemotron v3 models. ⏎  ⏎ ## Modifications ⏎  ⏎ Add support for FP8 and NVFP4 in non-gated MoE models ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-03888b9de5  (L1, 2025-12-01, sha 03888b9de5ec, PR #13968)
TITLE: [Minor] Upgrade cutedsl version in Dockerfile (#13968)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+3/-4)
LABELS: dependencies, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fa9021b21f  (L1, 2025-12-01, sha fa9021b21f92, PR #14173)
TITLE: fix: Increase FlashInfer workspace size for Qwen3VL models (#14173)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/flashinfer_backend.py (+4/-0)
BODY: ```shell ⏎ @sglang  ⏎ ➜  sglang git:(main) ✗ CUDA_VISIBLE_DEVICES=7 python -m sglang.launch_server \      --model Qwen/Qwen3-VL-32B-Instruct-FP8 \                                                 ⏎     --tp 1 \ ⏎     --quantization fp8 \ ⏎     --trust-remote-code   ⏎ /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml direct …[truncated]

### L1-3de09aadbc  (L1, 2025-12-01, sha 3de09aadbc03, PR #14122)
TITLE: Add new moe wna16 marlin gemm (#14122)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.runner.marlin
FILES: sgl-kernel/csrc/common_extension.cc (+2/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/generate_kernels.py (+44/-15); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel.h (+5/-6); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4.cuh (+16/-67); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4b8.cuh (+20/-83); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku8b128.cuh (+20/-83); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4.cuh (+16/-67); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4b8.cuh (+20/-83); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku8b128.cuh (+20/-83); sgl-kernel/csrc/moe/marlin_moe_wna16/marlin_template.h (+359/-265); (+5 more)
LABELS: quant, sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Unit-Test ⏎  ⏎ <img width="1996" height="918" alt="图片" src="https://github.com/user-attachments/assets/bc7bf020-e37b-461d-aa56-27f27396d8d1" /> ⏎  ⏎ ## Kimi-K2-Thinking Acc ⏎  ⏎ ```shell ⏎  ⏎  python3 -m sglang.launch_server --model-path moonshotai/Kimi-K2-Thinking --tp 8 --trust-remote-code  --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 ⏎  ⏎   sglang git:(add_moe_wna16_marlin_gemm_v2) ✗ python3 benchmark/gsm8k/bench_sglang.py --num-questions 20 …[truncated]

### L1-63b9300f00  (L1, 2025-12-01, sha 63b9300f00fe, PR #14244)
TITLE: chore: bump sgl-kernel version to 0.3.18.post2 (#14244)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.marlin, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+10/-16); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.18.post2` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.18.post2` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are up …[truncated]

### L1-3dabd609fb  (L1, 2025-12-02, sha 3dabd609fb03, PR #14047)
TITLE: Optimize topk sigmoid in minimax_m2 (#14047)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+38/-10); python/sglang/srt/models/minimax_m2.py (+0/-3)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR optimizes the topk sigmoid in minimax_m2, using the `topk_sigmoid` kernel implementation from #13049. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Make the `TopK` to support the `scoring_func` parameter. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ We have validated the correctness of this change on MiniMax-M2, achieving an accuracy of 0.9249 on GSM8K. Previous PR #13049 results was 0.93 on GSM8K and 0.803 on AIME2025. Original AIME2025 score was 0.78. …[truncated]

### L1-ca52ed425f  (L1, 2025-12-02, sha ca52ed425f7c, PR #14317)
TITLE: Clean up imports and move files (#14317)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.ep.deepep_dispatcher, L1.ep.other_dispatchers, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-2); python/sglang/srt/layers/moe/token_dispatcher/base.py (+1/-1); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+1/-1); python/sglang/srt/layers/moe/token_dispatcher/mooncake.py (+1/-1); python/sglang/srt/layers/moe/topk.py (+12/-7); python/pyproject.toml (+15/-14); python/sglang/compile_deep_gemm.py (+1/-1); python/sglang/srt/batch_overlap/operations.py (+0/-0); python/sglang/srt/batch_overlap/operations_strategy.py (+211/-0); python/sglang/srt/batch_overlap/single_batch_overlap.py (+0/-0); (+18 more)
LABELS: quant, dependencies, deepseek, run-ci
BODY: - Do not put direct files under `srt` folder, move most files into a subfolder of `srt` ⏎   - move single_batch_overlap.py, two_batch_overlap.py, operations.py, operations_strategy.py under a new folder srt/batch_overlap ⏎   - python/sglang/srt/warmup.py -> python/sglang/srt/entrypoints/warmup.py ⏎ - Clean imports and fake registration

### L1-c5947ecd85  (L1, 2025-12-02, sha c5947ecd8595, PR #14133)
TITLE: Opt moe align block size kernel (#14133)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+65/-46); sgl-kernel/benchmark/bench_moe_align_block_size.py (+29/-9); sgl-kernel/tests/test_moe_align.py (+2/-0)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: Need bump sgl-kernel version and apply https://github.com/sgl-project/sglang/pull/14134 . ⏎  ⏎ ## Kimi K2 Thinking acc ⏎  ⏎ ```shell ⏎ python3 -m sglang.launch_server --model-path moonshotai/Kimi-K2-Thinking --tp 8 --trust-remote-code  --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 ⏎  ⏎ ➜  sglang git:(add_moe_wna16_marlin_gemm_v2) ✗ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8  ⏎  ⏎ /usr/local/lib/pytho …[truncated]

### L1-77512ae0d7  (L1, 2025-12-03, sha 77512ae0d72a, PR #14333)
TITLE: [bugfix] Fix prefill tbo disabled when --deepep-mode=auto (#14333)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/scheduler_dp_attn_mixin.py (+2/-0); test/srt/ep/test_deepep_small.py (+9/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ When TBO is enabled and deepep is set to auto, TBO is actually not activated for prefill batches. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ We found that in `prepare_mlp_sync_batch_raw`, `tbo_preparer.prepare_all_gather` does not return the correct `local_can_run_tbo`. The issue is that `deepep_mode.resolve(local_batch.is_extend_in_batch)` produces an incorrect result because `local_batch.is_extend_in_batch` is not set properly. ⏎  ⏎ ## Accur …[truncated]

### L1-20aad5b5ab  (L1, 2025-12-03, sha 20aad5b5abca, PR #9660)
TITLE: Single Batch Overlap for MoE Models (#9660)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.runner.deep_gemm, L1.ep.deepep_dispatcher
FILES: docker/Dockerfile (+7/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+15/-4); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+22/-1); python/sglang/srt/layers/moe/moe_runner/runner.py (+23/-1); python/sglang/srt/layers/moe/token_dispatcher/base.py (+5/-5); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+35/-9); python/sglang/srt/batch_overlap/single_batch_overlap.py (+33/-12); python/sglang/srt/batch_overlap/two_batch_overlap.py (+6/-0); python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py (+23/-8); python/sglang/srt/models/deepseek_v2.py (+57/-3)
LABELS: deepseek, sgl-kernel, run-ci
BODY: # 1. Motivation ⏎ The optimization effect of Two-Batch Overlap (TBO) is suboptimal for the Decode phase on low-compute-power cards (i.e., H20). This is due to two main factors: First, on the Hopper architecture, the WGMMA block_m is 64. Consequently, when TBO is enabled with a small Decode batch size, the MLP GEMM suffers from redundant computations. A positive throughput gain is only observed at larger batch sizes (e.g., 64, 128). Second, at thes …[truncated]

### L1-16d8de2284  (L1, 2025-12-03, sha 16d8de2284ed, PR #14295)
TITLE: [bugfix] NpuFuseEPMoE miss initialization parameters (#14295)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-0)
LABELS: run-ci
DEEP_STUDY: deep-study correctness case sglang:16d8de2284: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Motivation ⏎  ⏎ bugfix： ⏎   NpuFuseEPMoE miss initialization parameters. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ <img width="1501" height="109" alt="image" src="https://github.com/user-attachments/assets/d6ded161-1999-44bc-adde-119c7ab83e52" /> ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e3ab23c1a6  (L1, 2025-12-04, sha e3ab23c1a688, PR #14399)
TITLE: Try to fix B200 DeepEP error (#14399)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2)
BODY: ## Motivation ⏎  ⏎ * https://github.com/pytorch/pytorch/blob/v2.9.0/.github/scripts/generate_binary_build_matrix.py#L57: torch 2.9.0 => nvshmem ==3.3.20 ⏎ * https://github.com/deepseek-ai/DeepEP/blob/main/third-party/README.md DeepEP => >=3.3.9 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-922756aaa1  (L1, 2025-12-04, sha 922756aaa1c2, PR #14350)
TITLE: [FIX] trtllm-moe-fp4-renorm for Qwen series models (#14350)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-1); python/sglang/srt/server_args.py (+0/-1)
LABELS: run-ci
DEEP_STUDY: deep-study correctness case sglang:922756aaa1: class=numerical_precision; symptom=wrong_output_or_accuracy; introducing=#13761/#14135
BODY: ## Motivation ⏎ **trtllm-moe-fp4-renorm for Qwen series models** ⏎ [PR13761](https://github.com/sgl-project/sglang/pull/13761 ) introduced bug for DeepSeek nvfp4 models. ⏎ [PR14135](https://github.com/sgl-project/sglang/pull/14135) Fixed it, but introduced bug for Qwen3 nvfp4 models. ⏎  ⏎ This PR is to re-fix both of them. After this fix, `test/nightly/test_qwen3_fp4_trtllm_gen_moe.py` could pass. ⏎ Also a PR for flashinfer repo will be created to requ …[truncated]

### L1-894c0dc57c  (L1, 2025-12-04, sha 894c0dc57cfd, PR #13359)
TITLE: [NPU][1/N] NPU basic functions refactor and new modelslim quant type (#13359)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_py, L1.hardware.cpu_npu_musa, L1.ep.layer
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+79/-0); python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+916/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-27); python/sglang/srt/layers/moe/topk.py (+9/-78); .github/CODEOWNERS (+1/-3); python/sglang/srt/environ.py (+3/-0); python/sglang/srt/eplb/expert_distribution.py (+2/-8); python/sglang/srt/hardware_backend/npu/allocator_npu.py (+7/-9); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+3/-1); python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py (+19/-24); (+33 more)
LABELS: lora, deepseek, npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Due to the underlying structural difference between gpgpus and npus, we have introduced a lot of `is_npu` branches in current repository from previous commits. Though literarlly it helps the out-of-box experience for our end-users and matches our rapid development pace, this way of orignizing codes breaks readability and of cource maintainability of the whole sglang project. We believe this is not a long-term solution and a h …[truncated]

### L1-b5d3998508  (L1, 2025-12-04, sha b5d3998508c0, PR #14421)
TITLE: Rename secrets.WHL_TOKEN -> secrets.GH_PAT_FOR_WHL_RELEASE (#14421)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: scripts/ci/ci_install_deepep.sh (+1/-1); .github/MAINTAINER.md (+1/-1); .github/pull_request_template.md (+1/-1); .github/workflows/cancel-all-pending-pr-test-runs.yml (+7/-3); .github/workflows/release-whl-kernel.yml (+2/-2); docs/developer_guide/contribution_guide.md (+1/-1); scripts/ci/ci_install_dependency.sh (+1/-1)
LABELS: documentation, run-ci
BODY: 

### L1-8428078436  (L1, 2025-12-04, sha 842807843671, PR #14213)
TITLE: Add Mistral Large 3 support. (#14213)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/common_utils.py (+7/-1); python/sglang/srt/configs/model_config.py (+5/-0); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+11/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+17/-5); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+81/-1); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py (+127/-63); python/sglang/srt/layers/quantization/fp8.py (+1/-5); python/sglang/srt/layers/quantization/fp8_utils.py (+43/-0); python/sglang/srt/models/deepseek_v2.py (+61/-12); python/sglang/srt/models/mistral_large_3.py (+81/-0); (+6 more)
LABELS: high priority, quant, Multi-modal, deepseek, blackwell, run-ci, vlm, model-gateway
ISSUES: #12751 [Bug] bench_sglang fails due to get_model_info endpoint of SGLang PDRouter not being implemented
BODY: ## Motivation ⏎  ⏎ This PR introduces support for model Mistral Large 3. ⏎  ⏎ ## Modifications ⏎  ⏎ To enable the model, several key modifications were made. ⏎  ⏎ * Two new models are supported: MistralLarge3ForCausalLM and PixtralForConditionalGeneration. ⏎ * The latter incorporates VLM support. ⏎ * Per-tensor scale MOE support was added to `fp8.py`. ⏎ * As ML3 is not yet supported in AutoConfig, a separate code-path for it in `hf_transformers_utils.py` wa …[truncated]

### L1-4c5074eb78  (L1, 2025-12-04, sha 4c5074eb786b, PR #14463)
TITLE: Add AMD stage support to /rerun-stage command and fix related bugs (#14463)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-amd.yml (+106/-19); .github/workflows/pr-test.yml (+133/-76); scripts/ci/slash_command_handler.py (+43/-17)
LABELS: amd, run-ci
BODY: ## Summary ⏎  ⏎ Fixes the `/rerun-stage <stage-name>` slash command to properly isolate and run only the targeted stage. ⏎  ⏎ **Bugs fixed:** ⏎  ⏎ 1. **Missing `inputs.target_stage == ''` check**: Added this check to all targetable jobs so non-matching jobs skip when a specific stage is targeted. ⏎  ⏎ 2. **`always()` at wrong level**: Moved `always()` from inside the second branch to the **outer level** of the condition. In GitHub Actions, when a dependency is s …[truncated]

### L1-498ea41ca6  (L1, 2025-12-05, sha 498ea41ca64b, PR #13861)
TITLE: dockerfile: add runtime stage + ubuntu 24.04 (#13861)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+376/-163); .github/workflows/release-docker.yml (+109/-4); docs/get_started/install.md (+17/-4)
LABELS: documentation
BODY: Will merge this in after next release ⏎  ⏎ Multi-stage Dockerfile splits SGLang builds into base, framework, and runtime stages. Runtime cuts image size roughly in half. ⏎  ⏎ --- ⏎  ⏎ ```bash ⏎ sglang                            framework-test     be66a8e51a09   39.3GB ⏎ sglang                            runtime-test          a4dac91fe030    20GB ⏎ ``` ⏎  ⏎ --- ⏎  ⏎ Tests ⏎ 1. cu13 arm - https://github.com/ishandhanani/srt-slurm/blob/main/recipies/gb300-fp4/1p2 …[truncated]

### L1-49dfa1d891  (L1, 2025-12-05, sha 49dfa1d891f4, PR #14312)
TITLE: [model-gateway] change sgl-router to sgl-model-gateway (#14312)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: .github/labeler.yml (+2/-2); .github/workflows/lint.yml (+3/-3); .github/workflows/nightly-release-gateway.yml (+4/-4); .github/workflows/pr-benchmark-rust.yml (+13/-13); .github/workflows/pr-test-pd-router.yml (+7/-7); .github/workflows/pr-test-rust.yml (+23/-21); .github/workflows/release-docker-gateway.yml (+2/-2); .github/workflows/release-pypi-gateway.yml (+5/-5); .pre-commit-config.yaml (+2/-2); docker/Dockerfile (+3/-3); (+421 more)
LABELS: documentation, dependencies, Multi-modal, deepseek, run-ci, model-gateway
BODY: rename sgl-router to sgl-model-gateway ⏎ less confusion ⏎  ⏎ ## Checklist

### L1-205f041e96  (L1, 2025-12-05, sha 205f041e9619, PR #14466)
TITLE: Add Mistral Large 3 Eagle Support (#14466)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+11/-1); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+2/-1); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py (+6/-10); python/sglang/srt/layers/quantization/fp8.py (+161/-36); python/sglang/srt/models/deepseek_v2.py (+14/-6); python/sglang/srt/models/mistral_large_3.py (+0/-3); python/sglang/srt/models/mistral_large_3_eagle.py (+105/-0); python/sglang/srt/server_args.py (+7/-3); python/sglang/srt/utils/mistral_utils.py (+7/-2)
LABELS: deepseek, blackwell, run-ci
BODY: ## Motivation ⏎ Support Mistral Large 3 Eagle. The eagle checkpoint `mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle` is using the FP8 per-tensor quantization while the standard FP8/NVFP4 checkpoint `mistralai/Mistral-Large-3-675B-Instruct-2512[-NVFP4]` is using compressed tensors quantization. In this PR, we support ⏎ - the functionality of FP8 + eagle for the Mistral Large 3 model ⏎ - Flashinfer TRTLLM FP8 per-tensor quant MoE, this can be used …[truncated]

### L1-ea177372bd  (L1, 2025-12-06, sha ea177372bd8c, PR #13115)
TITLE: support mtp with deepseek r1 nvfp4 model (#13115)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+32/-0); docs/advanced_features/server_arguments.md (+3/-2); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+17/-32); python/sglang/srt/model_executor/forward_batch_info.py (+6/-1); python/sglang/srt/models/deepseek_v2.py (+3/-1); python/sglang/srt/server_args.py (+11/-1); python/sglang/srt/speculative/eagle_info.py (+3/-0); python/sglang/srt/speculative/eagle_worker.py (+12/-6); python/sglang/srt/speculative/eagle_worker_v2.py (+24/-11); python/sglang/srt/speculative/standalone_worker.py (+6/-3); (+1 more)
LABELS: documentation, high priority, deepseek, speculative-decoding, blackwell, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: collabrate with @trevor-m  ⏎  ⏎ ## Motivation ⏎  ⏎ Support large scale EP deployment for the DS R1 fp4 model with eagle spec decoding.  ⏎  ⏎ ## Modifications ⏎  ⏎ - add the custom moe a2a backend for speculative decoding ⏎ - fix the request padding during the forward batch ⏎ - remove the request padding from trtllm-mla attn backend when draft_extend/verify ⏎  ⏎ ## Test Scripts ⏎  ⏎ **Prefill** ⏎  ⏎ ```bash ⏎ TORCH_CUDA_ARCH_LIST=10.0  NVSHMEM_IB_ENABLE_IBGDA=0 NV …[truncated]

### L1-9dfa01a435  (L1, 2025-12-06, sha 9dfa01a43573, PR #14538)
TITLE: [Misc]Register and refactor some environs for dpsk-fp4 and DeepEp (#14538)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-2); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+5/-5); python/sglang/srt/layers/moe/token_dispatcher/fuseep.py (+3/-3); python/sglang/srt/models/deepseek_v2.py (+3/-2); docs/references/environment_variables.md (+5/-1); python/sglang/srt/batch_overlap/single_batch_overlap.py (+7/-4); python/sglang/srt/environ.py (+10/-0)
LABELS: documentation, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d2b42477c7  (L1, 2025-12-06, sha d2b42477c788, PR #14518)
TITLE: chore: bump sgl-kernel version to 0.3.18.post3 (#14518)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.18.post3` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.18.post3` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are up …[truncated]

### L1-673c11ba73  (L1, 2025-12-07, sha 673c11ba7302, PR #14586)
TITLE: [Minor] Temporarily skipping deepep large mtp test (#14586)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: test/srt/ep/test_deepep_large.py (+1/-0)
BODY: ## Motivation ⏎  ⏎ @rainj-me is working on its fix ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e5135b73f4  (L1, 2025-12-07, sha e5135b73f433, PR #14544)
TITLE: Add CUDA kernel size analysis tool for sgl-kernel optimization (#14544)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-1); sgl-kernel/README.md (+22/-0); sgl-kernel/analyze_whl_kernel_sizes.py (+259/-0)
LABELS: documentation, sgl-kernel
BODY: ## Motivation ⏎  ⏎ This PR adds a kernel size analysis tool to help identify and optimize large CUDA kernels in sgl-kernel wheel packages. ⏎  ⏎ - Extracts and analyzes all CUDA kernels from wheel files ⏎ - Groups kernels by name prefix (before `<`) to identify bloat from template instantiation ⏎ - Sorts kernels by size to find optimization targets ⏎ - Handles files without CUDA code gracefully ⏎ - Outputs both human-readable text reports and machine-read …[truncated]

### L1-ae6a6630e4  (L1, 2025-12-07, sha ae6a6630e4fe, PR #13725)
TITLE: Add Expert Parallelism (EP) support for kimi-k2-thinking (#13725)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+12/-0)
LABELS: run-ci
BODY: ```shell ⏎ python3 -m sglang.launch_server --model-path moonshotai/Kimi-K2-Thinking --tp 8 --ep 4 --trust-remote-code  --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 ⏎ ``` ⏎  ⏎ Result: ⏎  ⏎ ```shell ⏎ ➜  sglang git:(ep_for_kimi_k2_thinking) ✗ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 ⏎ /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. …[truncated]

### L1-cf0478d602  (L1, 2025-12-07, sha cf0478d602ce, PR #14585)
TITLE: [Glm46v] Bug fix for accuracy drop and unable to launch server (#14585)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docs/advanced_features/dp_for_multi_modal_encoder.md (+1/-0); docs/basic_usage/glm45.md (+70/-0); docs/basic_usage/glmv.md (+136/-0); docs/basic_usage/popular_model_usage.rst (+3/-1); python/pyproject.toml (+1/-1); python/sglang/srt/configs/qwen3_omni.py (+0/-4); python/sglang/srt/configs/qwen3_vl.py (+0/-5); python/sglang/srt/models/glm4_moe.py (+1/-0); python/sglang/srt/models/glm4v.py (+19/-3); python/sglang/srt/models/glm4v_moe.py (+68/-15); (+2 more)
LABELS: documentation, high priority, dependencies, run-ci
ISSUES: #14582 The GLM-4.5V series model outputs garbled text.
BODY: ## Motivation ⏎ Adapted from https://github.com/sgl-project/sglang/pull/14568 and fixed https://github.com/sgl-project/sglang/issues/14582 ⏎  ⏎ Next follow-up: ⏎ - Fix https://github.com/sgl-project/sglang/issues/14582. Currently i just disabled shared expert fusion feature to bypass it as workaround ⏎ - Add GLM46v into CI or nightly build test ⏎ - Merge https://github.com/sgl-project/sglang/pull/14584 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  …[truncated]

### L1-32f8b6064e  (L1, 2025-12-08, sha 32f8b6064e0b, PR #14457)
TITLE: improve default glm mtp setting (#14457)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=161,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=161,N=192,device_name=NVIDIA_H200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=161,N=384,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=20,N=1536,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=20,N=1536,device_name=NVIDIA_H200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=40,N=1536,device_name=NVIDIA_H200,dtype=fp8_w8a8,per_channel_quant=True.json (+146/-0); python/sglang/srt/server_args.py (+1/-0)
LABELS: quant, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ### GLM 4.6 ⏎ Since it's MTP set it like Deepseek (topk > 1 is not so good). ⏎ BF16 TP = 8, MTP (5 4 8) ⏎ ``` ⏎ +-------------+--------+------------+-----------------+ ⏎ | Latency (s) | Tokens | Acc Length | Speed (token/s) | ⏎ +-------------+--------+------------+-----------------+ ⏎ |    6.171    |  512   |   2.081    |      82.97      | ⏎ +-------------+--------+------------+-----------------+ ⏎ ``` ⏎  ⏎ BF16 TP = 8 MTP (1 3 4) (+16%) ⏎ ``` ⏎ +------------ …[truncated]

### L1-2de98010b5  (L1, 2025-12-08, sha 2de98010b5ad, PR #14649)
TITLE: chore: bump sgl-kernel version to 0.3.19 (#14649)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.19` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.19` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are updated accord …[truncated]

### L1-f0e948a0f1  (L1, 2025-12-09, sha f0e948a0f1c9, PR #14601)
TITLE: fix the deepep 8 gpu unit test (#14601)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+1/-1); test/srt/ep/test_deepep_large.py (+4/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Fix the unit test failure for test/srt/ep/test_deepep_large.py ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-25e97380e3  (L1, 2025-12-10, sha 25e97380e386, PR #14854)
TITLE: Fix CUDA version handling in ci_install_deepep.sh (#14854)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: scripts/ci/ci_install_deepep.sh (+7/-1)
LABELS: run-ci
BODY: 

### L1-8642dbe416  (L1, 2025-12-10, sha 8642dbe41601, PR #14554)
TITLE: Refactor Marlin MoeRunner (#14554)
SOURCES: path_core, symbol_pickaxe, corpus:production-kernel-provenance
ARTIFACT_HINTS: L1.runner.framework, L1.runner.marlin
FILES: python/sglang/srt/layers/moe/moe_runner/marlin.py (+125/-0); python/sglang/srt/layers/moe/moe_runner/runner.py (+2/-0); python/sglang/srt/layers/moe/utils.py (+4/-0); python/sglang/srt/layers/quantization/awq.py (+22/-35); python/sglang/srt/layers/quantization/gptq.py (+21/-34); test/srt/quant/test_awq.py (+33/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Refactor Marlini MoE runner integration into `marlin.py` per https://github.com/sgl-project/sglang/issues/8715 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Implemented `MarlinRunnerInput`, `MarlinRunnerOutput`, `MarlinMoeQuantInfo`, `MarlinRunnerCore`, `pre_permute_standard_to_marlin`, and `post_permute_marlin_to_standard` ⏎  ⏎ - Added `TestAWQMarlinFloat16` to `test_awq.py` ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ - Passed `sglang/test/srt/quant/test_awq. …[truncated]

### L1-e54307f26a  (L1, 2025-12-10, sha e54307f26a61, PR #14313)
TITLE: [6/n] Fix `num_token_non_padded` computation in prefill (#14313)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/cuda_graph_runner.py (+0/-2); python/sglang/srt/model_executor/forward_batch_info.py (+42/-0); python/sglang/srt/model_executor/input_buffers.py (+11/-11); python/sglang/srt/model_executor/model_runner.py (+12/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `num_token_non_padded` is used for masking padded tokens in MoE compute. In the **prefill** path when `dp < tp`, the current implementation still overestimates the number of non-padded tokens, because it uses the token size in the local DP rank while tokens are sharded across the DP group under DeepEP. Additionally, the **decode without CUDA-graph** path is not covered by #9107 either. This PR applies the same correction from #91 …[truncated]

### L1-b8cfa02c01  (L1, 2025-12-10, sha b8cfa02c013b, PR #14806)
TITLE: [NPU] bug fix for mtp and w4a8 (#14806)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.ep.layer
FILES: python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+11/-9); python/sglang/srt/layers/moe/ep_moe/layer.py (+6/-1); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+2/-0)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5b5571a8da  (L1, 2025-12-11, sha 5b5571a8da7a, PR #14829)
TITLE: Apply back moe_sum_reduce for fused_marlin_moe (#14829)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.marlin
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+10/-4)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Previously we applied this change in https://github.com/sgl-project/sglang/pull/12888 but revert it to debug IMA crash issue. ⏎ Now the marlin kernel crash issue is solved, so we can apply this change back. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ ``` ⏎ Accuracy: 0.941 ⏎ Invalid: 0.000 ⏎ Latency: 88.846 s ⏎ Output throughput: 1451.369 token/s ⏎ ``` ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e9e7f15eb5  (L1, 2025-12-11, sha e9e7f15eb50e, PR #13730)
TITLE: [bugfix] fix TBO crashes when attn_tp_size > 1 (#13730)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/batch_overlap/operations.py (+10/-8); python/sglang/srt/batch_overlap/two_batch_overlap.py (+39/-6); python/sglang/srt/layers/communicator.py (+14/-1); python/sglang/srt/model_executor/forward_batch_info.py (+9/-0); python/sglang/srt/models/bailing_moe.py (+4/-0); python/sglang/srt/models/deepseek_v2.py (+2/-0); python/sglang/srt/models/falcon_h1.py (+3/-1); python/sglang/srt/models/glm4_moe.py (+2/-0); python/sglang/srt/models/gpt_oss.py (+2/-0); python/sglang/srt/models/llada2.py (+2/-0); (+10 more)
LABELS: deepseek, run-ci
ISSUES: #8033 [Bug] TBO is not compatible with Dense FFN under TP
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Issues [#12757](https://github.com/sgl-project/sglang/issues/12757) and [#10863](https://github.com/sgl-project/sglang/issues/10863) indicate that the current TBO implementation still has several bugs. Although PRs such as [#11423](https://github.com/sgl-project/sglang/pull/11423) and [#13082](https://github.com/sgl-project/sglang/pull/13082) address part of the problem, some configurations remain unfixed. ⏎  ⏎ For example, a Q …[truncated]

### L1-388018a5bd  (L1, 2025-12-11, sha 388018a5bd41, PR #14541)
TITLE: [NPU] adapt dsv3.2 nsa prefill context parallel (#14541)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/distributed/parallel_state.py (+3/-6); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+111/-22); python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py (+9/-0); python/sglang/srt/hardware_backend/npu/utils.py (+8/-0); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+117/-94); python/sglang/srt/layers/attention/nsa/utils.py (+25/-4); python/sglang/srt/layers/communicator_nsa_cp.py (+7/-8); test/srt/ascend/test_ascend_tp4_bf16.py (+1/-0)
LABELS: deepseek, npu, run-ci
BODY: ## Motivation ⏎  ⏎ Now, sglang already has the --enable-nsa-prefill-context-parallel option proposed  to support CP parallelism, [(1/n)support context parallel with deepseekv3.2-DSA ](https://github.com/sglang-npu/sglang_npu/commit/76504b8b078a7bf0a6a527632d697c1b7734feba), but NPU not supports DSV 3.2 context parallelism, This PR is intended to solve this problem. ⏎  ⏎ ## Modifications ⏎  ⏎ This feature currently reuses the TP domain for CP, and emplo …[truncated]

### L1-d7ed8a8c24  (L1, 2025-12-11, sha d7ed8a8c24ce, PR #13798)
TITLE: [NVIDIA] Enable TRTLLM BF16 MoE on Blackwell GPUs (#13798)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+13/-18); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+56/-15); python/sglang/srt/layers/quantization/unquant.py (+70/-1); python/sglang/srt/models/deepseek_v2.py (+7/-2); python/sglang/srt/server_args.py (+6/-5); test/nightly/test_flashinfer_trtllm_gen_moe_backend.py (+47/-1)
LABELS: quant, deepseek, blackwell, run-ci, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Dependency ⏎ **flashinfer-python >= 0.5.3** ⏎  ⏎ [Merged] This PR could be merged after [PR14350](https://github.com/sgl-project/sglang/pull/14350) ⏎  ⏎ ## Motivation ⏎ Enable TRTLLM BF16 MoE on Blackwell GPUs ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ TRTLLM MoE ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     8|exact_match|↑  | …[truncated]

### L1-c05d3afb5d  (L1, 2025-12-12, sha c05d3afb5d8b, PR #14572)
TITLE: [NPU] optimization for dsv3.2 (#14572)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+4/-11); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+6/-2); python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py (+34/-18); python/sglang/srt/hardware_backend/npu/quantization/modelslim.py (+5/-3); python/sglang/srt/hardware_backend/npu/utils.py (+8/-0); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+51/-15); python/sglang/srt/layers/layernorm.py (+1/-13); python/sglang/srt/layers/linear.py (+5/-0); python/sglang/srt/layers/quantization/compressed_tensors/utils.py (+1/-1); python/sglang/srt/models/deepseek_nextn.py (+1/-1); (+1 more)
LABELS: deepseek, npu, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Co-author: @jiaming1130 ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ In this PR, we optimized the performance of the DeepSeek-V3.2 on NPU  to improve the performance of the network in the decode phase, and also adapted the dynamic W8A8 quantization for kv_b. ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Implemented multi-stream parallelism for the shared expert and MOE. ⏎ 2. Implemented multi-stream parallelism for the calculation of q, k and weight in the indexer. ⏎ 3. Implemented multi- …[truncated]

### L1-44fd701732  (L1, 2025-12-12, sha 44fd70173265, PR #15020)
TITLE: Tune triton fused moe for the case of glm-4.6-fp8 b200 tp4 (#15020)
SOURCES: path_config_only, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=161,N=384,device_name=NVIDIA_B200,dtype=fp8_w8a8,per_channel_quant=True.json (+146/-0)
LABELS: quant
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4eda4194f2  (L1, 2025-12-12, sha 4eda4194f2c5, PR #15002)
TITLE: [Fix] Disable trtllm moe backend for draft model for a qucik fix (#15002)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+15/-0); test/srt/test_deepseek_v3_fp4_4gpu.py (+1/-1)
LABELS: high priority, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ A quick fix for draft model acc regression of DeepSeek V3. I'll have a investigation on this deeper after this PR. ⏎ cc @b8zhong  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ python3 -m sglang.launch_server   --model-path nvidia/DeepSeek-V3-0324-FP4   --speculative-algorithm EAGLE   --tp 4   --quantization modelopt_fp4 ⏎ +-------------+--------+------------+-----------------+ ⏎ | Latency (s) | Tokens | Acc Length | Speed (token/s …[truncated]

### L1-d36299ad77  (L1, 2025-12-13, sha d36299ad774f, PR #14423)
TITLE: [NPU] perf update with kvcache nz & w4a8 quant (#14423)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+56/-47); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+56/-22); python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py (+33/-5); python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py (+38/-23); python/sglang/srt/layers/rotary_embedding.py (+38/-8); python/sglang/srt/model_executor/forward_batch_info.py (+0/-5)
LABELS: deepseek, npu, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ 1、Use the nz format for kv cache, thie method accelerates the FIA operator.; ⏎ 2、Moe's w4a8 uses per-channel quantization; ⏎ 3、Accelerating preprocessing of MHA in prefill using the npu_interleave_rope operator； ⏎ 4、bugfix num_token_non_padded_cpu; ⏎  ⏎ ## Modifications ⏎  ⏎ Use `export SGLANG_USE_FIA_NZ=1` to enable FIA NZ, and this feature must be turned on together with mlapo `export SGLANG_NPU_USE_MLAPO=1`. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ < …[truncated]

### L1-3b8a824b8b  (L1, 2025-12-13, sha 3b8a824b8b2e, PR #14422)
TITLE: [VLM] Support VLM ViT Piecewise CUDA Graph (#14422)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/vision.py (+73/-30); python/sglang/srt/models/qwen2_5_vl.py (+83/-1); python/sglang/srt/multimodal/vit_cuda_graph_runner.py (+263/-0); test/manual/nightly/test_vlms_vit_cuda_graph.py (+271/-0)
LABELS: performance, Multi-modal, run-ci, vlm, piecewise-cuda-graph
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎  ⏎ Previously VLM supports Piecewise CUDA Graph in LLM. ⏎ But ViT was not included inside due to some Tensors addresses in ViT are not fixed. This PR is to support Piecewise CUDA Graph for ViT. Currently Triton Attention and FA3 as MM-Attention are supported. Qwen2.5-VL is supported as the first target VLM model. ⏎  ⏎ Co-author: @kousakawang  ⏎  ⏎ Notes: ⏎ TP>1 is supported now due to custom all-reduce is disabled by default. ⏎  ⏎ Before  …[truncated]

### L1-0e4108ba29  (L1, 2025-12-14, sha 0e4108ba29f4, PR #15052)
TITLE: feat(gateway): Add server-side TLS support (#15052)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-model-gateway/bindings/python/sglang_router/router.py (+2/-0); sgl-model-gateway/Cargo.toml (+3/-0); sgl-model-gateway/bindings/python/sglang_router/router_args.py (+28/-0); sgl-model-gateway/bindings/python/src/lib.rs (+12/-0); sgl-model-gateway/py_test/e2e_http/test_tls.py (+153/-0); sgl-model-gateway/src/config/builder.rs (+58/-0); sgl-model-gateway/src/config/types.rs (+8/-0); sgl-model-gateway/src/main.rs (+8/-1); sgl-model-gateway/src/server.rs (+40/-7)
LABELS: dependencies, run-ci, model-gateway
BODY: ## Motivation ⏎  ⏎  This PR adds server-side TLS support to the SGLang Model Gateway, allowing it to serve traffic over HTTPS.  ⏎  ⏎ ## Modifications ⏎  ⏎       ⏎ - Adds `--tls-cert-path` and `--tls-key-path` arguments to the Rust binary and Python launchers (`launch_router`, `launch_server`). ⏎  -   Updates the server startup logic in `src/server.rs` to use `axum-server` with `rustls` when certificate and key paths are provided. ⏎  -   Defaults to the ex …[truncated]

### L1-1ab9b8e0a3  (L1, 2025-12-14, sha 1ab9b8e0a3cc, PR #14764)
TITLE: Enable TRT AllReduce Fusion by default (#14764)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+27/-13); python/sglang/srt/utils/common.py (+2/-2)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: It can work on SM90+. ⏎  ⏎ Note that it will increase the startup time about +-3 minutes. ⏎  ⏎ H200 ⏎ ``` ⏎ python -m sglang.bench_one_batch \ ⏎   --model-path /opt/dlami/nvme/models/DeepSeek-V3.1/ \ ⏎   --tp-size 8 \ ⏎   --batch-size 16 \ ⏎   --input-len 512 \ ⏎   --output-len 512 \ ⏎   --mem-fraction-static 0.8 \ ⏎   --enable-flashinfer-allreduce-fusion ⏎    ⏎ Prefill. latency: 0.28894 s, throughput:  28351.92 token/s ⏎ Decode 0. Batch size: 16, latency: 0.015 …[truncated]

### L1-a9ce1623cd  (L1, 2025-12-14, sha a9ce1623cddd, PR #13969)
TITLE: [kernel][moe] add moe topk fast (#13969)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_softmax
FILES: sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu (+128/-12); sgl-kernel/tests/test_moe_topk_softmax.py (+55/-11); test/srt/quant/test_block_int8.py (+6/-4)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ The original moeTopK kernel reads data from global memory topk times, resulting in significant performance overhead. By finding the top-2 elements in each iteration, we can reduce global memory accesses to topk/2 times, substantially improving kernel efficiency. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ To minimize global memory access, we modified each thread to maintain its local top-2 elements. These partial results are then reduced in shared  …[truncated]

### L1-0261c4aff7  (L1, 2025-12-16, sha 0261c4aff784, PR #14857)
TITLE: [misc] Upgrade cutedsl to 4.3.1 (#14857)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+0/-3); python/pyproject.toml (+1/-1)
LABELS: dependencies, deepseek, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 15293 (confirmed_revert, reason=build_or_dependency)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0861dca81f  (L1, 2025-12-16, sha 0861dca81faf, PR #15293)
TITLE: Revert "[misc] Upgrade cutedsl to 4.3.1 (#14857)" (#15293)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+3/-0); python/pyproject.toml (+1/-1)
LABELS: dependencies
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 14857 reason=build_or_dependency
BODY: This reverts commit 0261c4aff784389d9d50a0ae00cbeb8d4570981f. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ pytest [sgl-kernel/tests/test_flash_attention_4.py](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/tests/test_flash_attention_4.py) ⏎  ⏎ ``` ⏎ cutlass.base_dsl.common.DSLCudaRuntimeError: DSLCudaRuntimeError: (<cudaError_t.cudaSuccess: 0>, b'cudaErrorInsufficientDriver') (error code: 35) ⏎ ``` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Be …[truncated]

### L1-c8c64876a7  (L1, 2025-12-16, sha c8c64876a772, PR #15280)
TITLE: [NVIDIA] Fixes for NVFP4 all-gather with spec decoding (#15280)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+9/-2); python/sglang/srt/layers/quantization/modelopt_quant.py (+5/-1)
LABELS: quant, nvidia
BODY: This PR fixes two issues with the FP4-allgather path. ⏎ 1. Since https://github.com/sgl-project/sglang/pull/13327 it became necessary to set `SGLANG_MOE_NVFP4_DISPATCH=1` for the fp4 allgather to work, otherwise the input global scale would not be set for the dispatcher. Now, you won't need to set this environment variable. ⏎  ⏎ Previously seen error message: ⏎ ``` ⏎   File "/trevor/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py", line  …[truncated]

### L1-b399e3ac4f  (L1, 2025-12-16, sha b399e3ac4f86, PR #15100)
TITLE: Support piecewise cuda graph for fused marlin moe (#15100)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.marlin
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+14/-3); python/sglang/srt/layers/moe/moe_runner/marlin.py (+4/-2); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+2/-2); python/sglang/srt/layers/quantization/gptq.py (+0/-29); test/srt/test_piecewise_cuda_graph.py (+35/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Support piecewise cuda graph for models using `fused_marlin_moe` like moe models with gptq/awq quantization, kimi-k2-thinking model. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-GPTQ-Int4 --trust-remote-code --enable-piecewise-cuda-graph ⏎  ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 1400 --parallel 1400 ⏎  ⏎ Accuracy: 0.903 ⏎ Invalid: 0.000 ⏎ Latency: 17.246 s ⏎ Output throu …[truncated]

### L1-435d1c83c1  (L1, 2025-12-16, sha 435d1c83c1f6, PR #14357)
TITLE: [Perf] Enable Flashinfer autotune by default (#14357)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-0); docs/advanced_features/server_arguments.md (+1/-0); python/sglang/srt/layers/quantization/fp8.py (+3/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-0); python/sglang/srt/model_executor/model_runner.py (+7/-3); python/sglang/srt/server_args.py (+4/-4); test/srt/test_deepseek_v3_fp4_4gpu.py (+1/-1)
LABELS: documentation, quant, deepseek, run-ci
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎ This PR enable Flashinfer autotune by default to achieve possible perf gain. ⏎ Following #12306. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9970ee34e8  (L1, 2025-12-18, sha 9970ee34e8e1, PR #15049)
TITLE: Mistral Large 3 NVFP4 TRTLLM MoE support (#15049)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-1); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+193/-21); python/sglang/srt/layers/quantization/modelopt_quant.py (+2/-125); python/sglang/srt/layers/quantization/utils.py (+140/-0); python/sglang/srt/server_args.py (+2/-1); python/sglang/srt/utils/mistral_utils.py (+1/-2)
LABELS: quant, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ Support Mistral Large 3 NVFP4 TRTLLM MoE. ⏎ Tested with cmd from #14485 + `--moe-runner-backend flashinfer_trtllm`: ⏎  ⏎ ### Accuracy ⏎ TRTLLM MoE: ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     8|exact_match|↑  |0.9249|±  |0.0073| ⏎ |     |       |strict-match    |     8|exact_match|↑  |0.7 …[truncated]

### L1-56d12b4aea  (L1, 2025-12-18, sha 56d12b4aea9a, PR #15306)
TITLE: Fix warp illegal instruction in kimi k2 thinking PCG (#15306)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu (+12/-4)
LABELS: sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎ The `kimi_k2_moe_fused_gate` kernels fail to initialize `indices_ptr` when no valid expert is found (when `expert_id` is out of bounds or `max_expert == -1`), leaving uninitialized garbage values that propagate to `moe_align_block_size_kernel` and cause out-of-bounds memory access in `atomicAdd(&shared_counts[expert_id], 1)`, triggering a "Warp Illegal Instruction" error. ⏎  ⏎ ### Kimi K2 Thinking piecewise cuda graph ⏎  ⏎ ```shell ⏎  …[truncated]

### L1-4792d1f452  (L1, 2025-12-18, sha 4792d1f45203, PR #15141)
TITLE: [sgl-kernel][1/2] Fused qk_norm_rope for GLM4.6 (#15141)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/moe/fused_qknorm_rope_kernel.cu (+59/-39); sgl-kernel/python/sgl_kernel/moe.py (+2/-0); sgl-kernel/csrc/common_extension.cc (+2/-1); sgl-kernel/include/sgl_kernel_ops.h (+2/-1); sgl-kernel/tests/test_fused_qk_norm_rope.py (+11/-1)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ ref https://github.com/sgl-project/sglang/pull/14952 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ ``` ⏎ ➜  sgl-kernel git:(pr/sgl-kernel-partial-rotary) pytest tests/test_fused_qk_norm_rope.py    ⏎ =================================================================================================================================================== test session starts ================================================================ …[truncated]

### L1-793c96c3d2  (L1, 2025-12-18, sha 793c96c3d269, PR #12921)
TITLE: [perf]optimize w4afp8 kernel on deepseek-v3-0324 (#12921)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.w4a8
FILES: sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu (+64/-236); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu (+96/-27); python/sglang/srt/layers/quantization/w4afp8.py (+0/-1)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ we use w4afp8 deepseekv3-0324 online, and we find its performance is not good enough when decode batch size < 32 ⏎  ⏎ ## Modifications ⏎  ⏎ fine-grained tiling config ⏎ and based on https://github.com/sgl-project/sglang/pull/10027/files ⏎ I use cuda-int4 memory access to decrease memory-access pressure ⏎  ⏎ ## Accuracy Tests ⏎ deepseek-v3-0324 w4afp8 ⏎ ``` ⏎ aime25          0.4/0.4 ⏎ aime24          0.5/0.65/0.55 ⏎ mmlu              0.8947 ⏎ ` …[truncated]

### L1-ef908aeb40  (L1, 2025-12-19, sha ef908aeb401d, PR #15022)
TITLE: fixed trtllm nvfp4 backend for moe (#15022)
SOURCES: path_core, symbol_pickaxe, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+12/-4); python/sglang/srt/layers/quantization/modelopt_quant.py (+31/-9)
LABELS: high priority, quant, run-ci
DEEP_STUDY: deep-study correctness case sglang:ef908aeb40: class=numerical_precision; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Motivation ⏎  ⏎ The  --moe-runner-backend flashinfer_trtllm backend for NVFP4 quantized models had some bugs, specifically for baseten-admin/Qwen3-235B-A22B-Instruct-2507-FP4.  ⏎  ⏎ ## Modifications ⏎  ⏎ ModelOpt FP4 config now handles kv_cache_scheme provided as either dict or string (maps to FP8/NVFP4/auto), improving config compatibility. ⏎  ⏎ NVFP4 weight scale handling ensures per-16 block layout (with logging) and backend-specific validation. ⏎  …[truncated]

### L1-160a06cab2  (L1, 2025-12-19, sha 160a06cab23f, PR #15207)
TITLE: [Feature] Xiaomi `MiMo-V2-Flash` day0 support (#15207)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/batch_overlap/two_batch_overlap.py (+2/-0); python/sglang/srt/configs/load_config.py (+3/-0); python/sglang/srt/configs/model_config.py (+66/-13); python/sglang/srt/disaggregation/decode.py (+7/-0); python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py (+5/-5); python/sglang/srt/disaggregation/mooncake/conn.py (+8/-2); python/sglang/srt/disaggregation/prefill.py (+7/-0); python/sglang/srt/entrypoints/http_server.py (+1/-1); python/sglang/srt/function_call/function_call_parser.py (+2/-0); python/sglang/srt/function_call/mimo_detector.py (+281/-0); (+28 more)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ MiMo-V2-Flash is a Mixture-of-Experts (MoE) language model with 309B total parameters and 15B active parameters. Designed for high-speed reasoning and agentic workflows, it utilizes a novel hybrid attention architecture and Multi-Token Prediction (MTP) to achieve state-of-the-art performance while significantly reducing inference costs. ⏎  ⏎ See it on HF: https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash ⏎ LMSys blog: https://lmsys. …[truncated]

### L1-9d0347b33a  (L1, 2025-12-20, sha 9d0347b33aff, PR #14164)
TITLE: EP Support for Piecewise Cuda Graph (#14164)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+22/-3); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+85/-1); python/sglang/srt/layers/moe/topk.py (+5/-14); python/sglang/srt/compilation/backend.py (+1/-11); python/sglang/srt/compilation/compilation_config.py (+7/-0); python/sglang/srt/compilation/piecewise_context_manager.py (+10/-1); python/sglang/srt/layers/communicator.py (+1/-0); python/sglang/srt/model_executor/model_runner.py (+22/-2); python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py (+26/-23); test/srt/run_suite.py (+1/-1); (+1 more)
LABELS: run-ci, piecewise-cuda-graph
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Support EP for Piecewise Cuda Graph ⏎  ⏎ ## Modifications ⏎  ⏎ Following discussions with @ch-wan, @ispobock, and @BBuf, the Piecewise CUDA Graph wrapper for the MoE layer has been scoped to focus on Fused MoE and its derived classes. Currently, we primarily support standard top-k output, handling the necessary packing and unpacking operations for Torch library registration. ⏎  ⏎ Also credit to: @byjiang1996 ⏎  ⏎ ## Accuracy Tests & Benc …[truncated]

### L1-050f108c29  (L1, 2025-12-20, sha 050f108c29f0, PR #15526)
TITLE: Optimize Bailing-MoE with FlashInfer Fused All-Reduce (#15526)
SOURCES: subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/bailing_moe.py (+58/-20)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is to make Bailing-MoE model leverage FlashInfer fused_allreduce to fuse allreduce+rmsnorm+residual_add. ⏎ In B200 the inclusionAI/Ling-plus model E2E TTFT reduce 7.2%. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ PR: ⏎ ``` ⏎ ➜  sglang git:(main) python3 benchmark/gsm8k/bench_sglang.py --num-questions 200 --parallel 128 --num-shots 8 --port 30000 ⏎ 100%|██████████████████████████████████████████████████████████████████ …[truncated]

### L1-165f5c04cb  (L1, 2025-12-20, sha 165f5c04cbc2, PR #15464)
TITLE: Optimize MiMo-V2-Flash by flashinfer fused allreduce (#15464)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/mimo_v2_flash.py (+66/-10)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is to make MiMo-V2-Flash model leverage FlashInfer fused_allreduce to fuse allreduce+rmsnorm+residual_add. ⏎ The E2E TTFT reduce 5.1%. ⏎  ⏎ ``` ⏎ ➜  sglang git:(main) python3 -m sglang.launch_server --model XiaomiMiMo/MiMo-V2-Flash --tp-size 4 --port 30000 --attention-backend triton --disable-radix-cache --trust-remote-code --enable-flashinfer-allreduce-fusion ⏎ ...... ⏎ [2025-12-19 07:41:50] DeepGemm is enabled but the scale …[truncated]

### L1-7fa4906f4f  (L1, 2025-12-21, sha 7fa4906f4ff4, PR #15552)
TITLE: [sgl-kernel] Streamline kernel size report (Top 20 only) and clean up (#15552)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: sgl-kernel/README.md (+6/-4); sgl-kernel/analyze_whl_kernel_sizes.py (+28/-66)
LABELS: documentation, sgl-kernel
BODY: ## Motivation ⏎  ⏎ Follow https://github.com/sgl-project/sglang/pull/14544 ⏎  ⏎ Cleans up `sgl-kernel/analyze_whl_kernel_sizes.py` by removing noisy non-essential prints/comments and keeping the output focused on the text report, which summarizes only the Top 20 kernel name-prefix groups and Top 20 individual kernels (with an aggregated “Other” row for the rest). ⏎  ⏎ In h200: ⏎  ⏎ ```shell ⏎ =============================================================== …[truncated]

### L1-4b351f6b95  (L1, 2025-12-21, sha 4b351f6b9582, PR #14134)
TITLE: Apply new moe align block size kernel (#14134)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py (+4/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Follow https://github.com/sgl-project/sglang/pull/14133 and need a new sgl-kernel version. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-350fbbf4dc  (L1, 2025-12-21, sha 350fbbf4dc1e, PR #14901)
TITLE: fix ds3.2 nsa backend prefill TBO (#14901)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/tbo_backend.py (+3/-0); python/sglang/srt/models/deepseek_v2.py (+8/-1); python/sglang/srt/server_args.py (+9/-0); test/srt/ep/test_deepep_large.py (+55/-0); test/srt/run_suite.py (+1/-1)
LABELS: deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ After applying the fixes, tbo can be enabled on ds3.2. ⏎ And resolving ⏎ • https://github.com/sgl-project/sglang/issues/12698 ⏎ • https://github.com/sgl-project/sglang/issues/14594 ⏎  ⏎ ## Modifications ⏎  ⏎ 1. fix some bug that make nsa backend compatible with tbo ⏎ 2. add an an assertion when tbo is enabled but not set moe-e2e-backend  ⏎ 3. add test for ds3.2 with tbo ⏎  ⏎ ## Accuracy Tests ⏎ ``` ⏎ Accuracy: 0.957 ⏎ Invalid: 0.000 ⏎ Latency: 20 …[truncated]

### L1-254de6d2fd  (L1, 2025-12-21, sha 254de6d2fdcf, PR #15569)
TITLE: Add triton_fused_moe config for GLM-4.6-FP8 tp8 blackwell (#15569)
SOURCES: path_config_only, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=161,N=192,device_name=NVIDIA_B200,dtype=fp8_w8a8,per_channel_quant=True.json (+146/-0)
LABELS: quant
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fc05acc2c7  (L1, 2025-12-21, sha fc05acc2c7b7, PR #15432)
TITLE: [FusedMoE] Fix fused w13 tp sharded weight loading (#15432)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+26/-3)
LABELS: run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 15579 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Motivation ⏎ For fused w13 weights [w1; w3], we need to shard w1 and w3 independently, then concatenate them back to maintain the ⏎ [w1_shard; w3_shard] structure per TP rank, because our gelu_and_mul kernel is assuming this kind of layout: ⏎  ⏎ https://github.com/sgl-project/sglang/blob/4b4050e27e219a4d5bfadc4b4846414a093ed56a/python/sglang/srt/layers/elementwise.py#L530-L532 ⏎  ⏎ This is an RFC as I don't know how the broader community is using w1 …[truncated]

### L1-8fe3e37468  (L1, 2025-12-21, sha 8fe3e3746832, PR #15531)
TITLE: Support piecewise cuda graph for dsv3 fp4 (#15531)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+64/-10); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+3/-1); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py (+1/-2); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+11/-1); test/srt/run_suite.py (+1/-1); test/srt/test_deepseek_v3_fp4_4gpu.py (+67/-0)
LABELS: quant, deepseek, blackwell, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/issues/11490 ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model nvidia/DeepSeek-R1-0528-FP4-v2 --tp 8 --trust-remote --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' --enable-piecewise-cuda-graph --quantization modelopt_fp4 ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 1400 --parallel 1400 ⏎  ⏎ Accuracy: 0.947 ⏎ Invalid: 0.000 ⏎ Late …[truncated]

### L1-bed301a5ac  (L1, 2025-12-21, sha bed301a5acaa, PR #12162)
TITLE: [Feature] Enable return routed experts (#12162)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/routed_experts_capturer.py (+289/-0); python/sglang/srt/layers/moe/topk.py (+11/-1); python/sglang/bench_serving.py (+6/-0); python/sglang/srt/entrypoints/engine.py (+2/-0); python/sglang/srt/managers/detokenizer_manager.py (+18/-0); python/sglang/srt/managers/io_struct.py (+14/-0); python/sglang/srt/managers/multi_tokenizer_mixin.py (+3/-0); python/sglang/srt/managers/schedule_batch.py (+12/-0); python/sglang/srt/managers/scheduler.py (+1/-0); python/sglang/srt/managers/scheduler_output_processor_mixin.py (+18/-0); (+17 more)
LABELS: documentation, high priority, quant, amd, dependencies, Multi-modal, deepseek, sgl-kernel, blackwell, npu
BODY: ## Motivation ⏎  ⏎ As per the request from the RL community, this PR enables sglang to return routed experts (topk) during fwd for later usage in training phase. Thanks the MiMo team for proprosing [R3](https://arxiv.org/abs/2510.11370v1) to help stablizing MoE Reinforcement Learning. This method has been used in [MiMo-V2-Flash](https://github.com/XiaomiMiMo/MiMo-V2-Flash/blob/main/paper.pdf) and [DeepSeek-V3.2](huggingface.co/deepseek-ai/DeepSeek- …[truncated]

### L1-393e2f9b62  (L1, 2025-12-22, sha 393e2f9b62ff, PR #15579)
TITLE: Revert "[FusedMoE] Fix fused w13 tp sharded weight loading" (#15579)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-26)
LABELS: run-ci
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 15432 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#15432 ⏎  ⏎ Errors: https://github.com/sgl-project/sglang/actions/runs/20405657770/job/58655183248?pr=15509#step:5:3959

### L1-34013d9d5a  (L1, 2025-12-22, sha 34013d9d5a59, PR #15590)
TITLE: chore: bump sgl-kernel version to 0.3.20 (#15590)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.20` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.20` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are updated accord …[truncated]

### L1-061f41affc  (L1, 2025-12-22, sha 061f41affc70, PR #15539)
TITLE: MoE: Skip SiLU/GELU activation for masked experts (#15539)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+28/-4); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+104/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ When `filter_expert` is enabled, some tokens are routed to masked experts (`expert_id == -1`) and should skip activation computation. This PR adds Triton SiLU/GELU multiply kernels that correctly bypass compute for these cases while still producing valid outputs, for both sorted and non-sorted routing layouts. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ gsm8k's accuracy is correct under different concurrency. ⏎  ⏎ ## Benchmarki …[truncated]

### L1-ff903a7eea  (L1, 2025-12-23, sha ff903a7eeaf4, PR #15704)
TITLE: Simplify server args (#15704)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+9/-12)
LABELS: run-ci
BODY: 

### L1-45adad37d0  (L1, 2025-12-25, sha 45adad37d0f5, PR #15586)
TITLE: Add manual routing policy for router (#15586)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-model-gateway/bindings/python/sglang_router/router.py (+1/-0); sgl-model-gateway/benches/request_processing.rs (+2/-0); sgl-model-gateway/bindings/python/sglang_router/router_args.py (+10/-3); sgl-model-gateway/bindings/python/src/lib.rs (+2/-0); sgl-model-gateway/src/config/types.rs (+4/-0); sgl-model-gateway/src/config/validation.rs (+1/-0); sgl-model-gateway/src/observability/metrics.rs (+13/-0); sgl-model-gateway/src/policies/bucket.rs (+39/-0); sgl-model-gateway/src/policies/cache_aware.rs (+15/-4); sgl-model-gateway/src/policies/factory.rs (+8/-1); (+25 more)
LABELS: run-ci, model-gateway
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0c39730b18  (L1, 2025-12-25, sha 0c39730b18f9, PR #11469)
TITLE: DP: support piggyback server load report (#11469)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/references/environment_variables.md (+1/-0); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/managers/data_parallel_controller.py (+40/-14); python/sglang/srt/managers/detokenizer_manager.py (+7/-1); python/sglang/srt/managers/io_struct.py (+9/-0); python/sglang/srt/managers/scheduler.py (+1/-3); python/sglang/srt/managers/scheduler_metrics_mixin.py (+1/-0); python/sglang/srt/managers/scheduler_output_processor_mixin.py (+16/-1); python/sglang/srt/managers/tokenizer_manager.py (+10/-19); python/sglang/srt/server_args.py (+0/-7)
LABELS: documentation, run-ci
BODY: ## Motivation ⏎  ⏎ This PR implements a piggyback load reporting mechanism, as issue [11186](https://github.com/sgl-project/sglang/issues/11186) has mentioned. ⏎ Also，I have tested this piggyback load report mechanism upon shortest_queue scheduler and used a new method to count request queue length. ⏎  ⏎ ## Modifications ⏎  ⏎ ### piggyback load reporting ⏎ Load information is generated inside process_batch_result, and send back all the way to tokenizer m …[truncated]

### L1-de03b0cd30  (L1, 2025-12-25, sha de03b0cd305b, PR #15815)
TITLE: [Nemotron 3 Nano] Add triton MoE configs (#15815)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1856,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1856,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=232,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=232,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=464,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=464,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=928,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=928,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=16,N=1856,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=16,N=1856,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); (+10 more)
BODY: ## Motivation ⏎  ⏎  ⏎ Add Triton fused MoE configs for Nemotron 3 Nano in different parallelization schemes for H100 and B200 GPUs ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4edee6954a  (L1, 2025-12-26, sha 4edee6954a9a, PR #15850)
TITLE: [model-gateway] add JWT/OIDC authentication for control plane APIs (#15850)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-model-gateway/bindings/python/sglang_router/router.py (+68/-0); sgl-model-gateway/Cargo.toml (+2/-0); sgl-model-gateway/README.md (+116/-0); sgl-model-gateway/bindings/python/sglang_router/router_args.py (+109/-0); sgl-model-gateway/bindings/python/src/lib.rs (+158/-0); sgl-model-gateway/src/auth/audit.rs (+357/-0); sgl-model-gateway/src/auth/config.rs (+345/-0); sgl-model-gateway/src/auth/jwks.rs (+515/-0); sgl-model-gateway/src/auth/jwt.rs (+532/-0); sgl-model-gateway/src/auth/middleware.rs (+399/-0); (+7 more)
LABELS: documentation, dependencies, run-ci, model-gateway
BODY: Add secure authentication and authorization for control plane APIs (workers, wasm, tokenizers management) while keeping inference APIs under the existing simple API key authentication. ⏎  ⏎ New features: ⏎ - JWT/OIDC integration with external IDPs (Azure AD, Okta, etc.) ⏎ - JWKS auto-discovery and caching ⏎ - Role-based access control (admin role required for control plane) ⏎ - API keys with roles for service accounts ⏎ - Audit logging for control plane …[truncated]

### L1-cf34d0ab32  (L1, 2025-12-26, sha cf34d0ab3292, PR #15881)
TITLE: [Fix] assert error in log_prefill_stats (#15881)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/prefill.py (+2/-0)
BODY: ## Motivation ⏎  ⏎ This PR fix #15858. The prefill server crashes on assertion of log_prefill_stats. ⏎ ``` ⏎ [2025-12-26 16:40:13] End of prefill disaggregation mode warmup with status 200, resp: [{'text': '/*', 'output_ids': [1057], 'meta_info': {'id': '53787acedd9944f88667aea71f1bf629', 'finish_reason': {'type': 'length', 'length': 0}, 'prompt_tokens': 4, 'weight_version': 'default', 'total_retractions': 0, 'completion_tokens': 1, 'cached_tokens':  …[truncated]

### L1-171912a9e3  (L1, 2025-12-26, sha 171912a9e36f, PR #15907)
TITLE: [model-gateway] Add consistent hashing for ManualPolicy routing (#15907)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-model-gateway/bindings/python/sglang_router/router.py (+1/-0); sgl-model-gateway/bindings/python/src/lib.rs (+2/-0); sgl-model-gateway/src/config/types.rs (+12/-3); sgl-model-gateway/src/config/validation.rs (+4/-1); sgl-model-gateway/src/core/mod.rs (+1/-1); sgl-model-gateway/src/core/worker_registry.rs (+147/-2); sgl-model-gateway/src/observability/metrics.rs (+9/-0); sgl-model-gateway/src/policies/consistent_hashing.rs (+527/-0); sgl-model-gateway/src/policies/factory.rs (+16/-2); sgl-model-gateway/src/policies/manual.rs (+378/-150); (+5 more)
LABELS: run-ci, model-gateway
BODY: Implement header-based routing with consistent hashing support: ⏎  ⏎ ## HashRing Implementation (worker_registry.rs) ⏎ - Add HashRing struct with O(log n) worker selection via binary search ⏎ - Use 150 virtual nodes per worker for even key distribution ⏎ - Cache rings per-model, rebuild only on worker add/remove ⏎  ⏎ ## ManualPolicy Routing (manual.rs) ⏎ - X-SMG-Target-Worker: Direct routing by worker index (0-based) ⏎ - X-SMG-Routing-Key: Consistent hash …[truncated]

### L1-3645ed0f73  (L1, 2025-12-27, sha 3645ed0f737c, PR #15935)
TITLE: [model-gateway] Add PrefixHash load balancing policy for KV cache-aware routing (#15935)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-model-gateway/bindings/python/sglang_router/router.py (+1/-0); sgl-model-gateway/Cargo.toml (+2/-0); sgl-model-gateway/bindings/python/src/lib.rs (+5/-0); sgl-model-gateway/src/config/types.rs (+25/-0); sgl-model-gateway/src/config/validation.rs (+20/-0); sgl-model-gateway/src/main.rs (+15/-3); sgl-model-gateway/src/observability/metrics.rs (+9/-0); sgl-model-gateway/src/policies/factory.rs (+13/-1); sgl-model-gateway/src/policies/mod.rs (+5/-0); sgl-model-gateway/src/policies/prefix_hash.rs (+409/-0); (+4 more)
LABELS: dependencies, run-ci, model-gateway
BODY: A lightweight alternative to the cache_aware radix tree policy. Routes requests based on prefix token hash for cache locality. ⏎  ⏎ Algorithm: ⏎ - Extract first N tokens from request (configurable, default: 256) ⏎ - Hash token sequence using xxhash for fast, stable hashing ⏎ - Use consistent hash ring to find target worker (O(log n) lookup) ⏎ - If worker is overloaded (load > avg * load_factor), select least loaded ⏎ - Fall back to least loaded worker i …[truncated]

### L1-60a230b1fd  (L1, 2025-12-27, sha 60a230b1fda7, PR #14736)
TITLE: [NPU] Support w4a8 with activation clip (#14736)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+105/-7); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+12/-0); python/sglang/srt/hardware_backend/npu/quantization/modelslim.py (+8/-1)
LABELS: npu, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ This PR introduces an optimized W4A8 quantization implementation for MoE.  ⏎  ⏎ - Weight Quantization (W4): Static Per-Channel Int4 quantization is applied to expert weights. ⏎  ⏎ - Activation Quantization (A8): Dynamic Per-Token Int8 quantization is applied to activations. ⏎  ⏎ Compared to W8A8, this implementation achieves a significant reduction in the memory footprint of expert weights (approximately 2× less), while maintaining com …[truncated]

### L1-c457aad54a  (L1, 2025-12-28, sha c457aad54ae7, PR #16001)
TITLE: Update test parameters for deepep_large test (#16001)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: test/srt/ep/test_deepep_large.py (+5/-4)
BODY: ## Motivation ⏎  ⏎  ⏎ Fix the timeout issue for `test_deepep_large.py` ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e6d5a213ad  (L1, 2025-12-28, sha e6d5a213aded, PR #15998)
TITLE: Fix metrics (#15998)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+3/-1); python/sglang/srt/managers/data_parallel_controller.py (+2/-2); python/sglang/srt/managers/detokenizer_manager.py (+3/-3); python/sglang/srt/managers/scheduler_metrics_mixin.py (+7/-1); python/sglang/srt/managers/tokenizer_manager.py (+42/-26); python/sglang/srt/model_loader/ci_weight_validation.py (+0/-2); python/sglang/srt/model_loader/weight_utils.py (+31/-42)
LABELS: run-ci
BODY: 

### L1-656f4d69a1  (L1, 2025-12-28, sha 656f4d69a1bc, PR #15353)
TITLE: Refactor fp8 nextn layer for DeepSeek nvfp4 checkpoint (#15353)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: docs/references/environment_variables.md (+1/-0); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/layers/quantization/fp8.py (+6/-0); python/sglang/srt/models/deepseek_nextn.py (+3/-3); python/sglang/srt/models/deepseek_v2.py (+83/-109); python/sglang/srt/server_args.py (+21/-0)
LABELS: documentation, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ In DeepSeek-nvfp4 checkpoint, the moe weights of nextn layer is stored in bf16 precision. We have some logics that quantize nextn moe layer to fp8 optionally, but the codes are a little bit messy. ⏎  ⏎ This PR refactors this part of codes. With flag `SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE` enabled, dpsk fp4 can be launched with fp8 MTP correctly. ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ### B200 ⏎  ⏎ Launch: ⏎ ``` ⏎ # First we need to apply https://github …[truncated]

### L1-0294844f04  (L1, 2025-12-28, sha 0294844f04e2, PR #15891)
TITLE: [fix]deepgemm precompile when warmup (#15891)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py (+2/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ I encountered a compilation error when starting the P-node with the --enable-dp-attention parameter. Specifically: ⏎ ``` bash ⏎   File "/sgl-workspace/sglang/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py", line 256, in deep_gemm_execution_hook ⏎     _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups) ⏎   File "/sgl-workspace/sglang/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py", line 108 …[truncated]

### L1-208e6a9dac  (L1, 2025-12-28, sha 208e6a9dacb0, PR #16013)
TITLE: [Doc]Update MTP moe backends for EP document (#16013)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docs/advanced_features/expert_parallelism.md (+14/-0)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2ec6fa3c54  (L1, 2025-12-29, sha 2ec6fa3c54a8, PR #14280)
TITLE: feat PD: add eagle3 support for DeepSeek V3 in EP mode (#14280)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+12/-1)
LABELS: deepseek, run-ci
BODY: When try to enable EP MOE for DeepSeek V3 with EAGLE3,  ⏎  ⏎ SGL_ENABLE_JIT_DEEPGEMM_BMM=1 SGL_ENABLE_JIT_DEEPGEMM=1 \ ⏎ TORCHINDUCTOR_CACHE_DIR=/home/admin/inductor_root_cache /opt/conda/bin/python -m sglang.launch_server --model-path /home/admin/model/ --host 0.0.0.0 --port 8089 --disaggregation-mode decode --disaggregation-transfer-backend mooncake --attention-backend fa3 --speculative-attention-mode decode --trust-remote-code --mem-fraction-stat …[truncated]

### L1-45f3ad2f52  (L1, 2025-12-31, sha 45f3ad2f5236, PR #16175)
TITLE: [Refactor] Rename CustomOp -> MultiPlatformOp (#16175)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-2); python/sglang/srt/layers/activation.py (+6/-6); python/sglang/srt/layers/attention/mamba/mixer2_rms_norm_gated.py (+2/-2); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+2/-2); python/sglang/srt/layers/layernorm.py (+5/-5); python/sglang/srt/layers/model_parallel.py (+1/-1); python/sglang/srt/layers/quantization/unquant.py (+2/-2); python/sglang/srt/layers/rotary_embedding.py (+3/-3); python/sglang/srt/layers/utils/__init__.py (+1/-0); python/sglang/srt/layers/utils/multi_platform.py (+3/-7); (+2 more)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ #15995  ⏎  ⏎ Rename CustomOp -> MultiPlatformOp for clarity. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/b …[truncated]

### L1-e0e5084802  (L1, 2026-01-01, sha e0e508480247, PR #14085)
TITLE: Fix parse args from file(#13911) (#14085)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+6/-18); python/sglang/srt/server_args_config_parser.py (+52/-26); test/manual/test_config_integration.py (+9/-2)
LABELS: run-ci
ISSUES: #13911 [Bug]  Is `_add_boolean_arg` code  has a problem?
BODY: ## Motivation ⏎ fix(#13911) ⏎  ⏎ The `argparse` will directly generate an instance of the _StoreTrueAction class based on `action="store_true"`, and will not store the `action="store_true"` field on the instance. ⏎  ⏎ So the code `hasattr(action, "action")` will never return True.  You can see the test results I added in the issue(#13911). ⏎  ⏎  ⏎ If the code there had issues, why did it still run successfully? This is because the usage of `boolean_actio …[truncated]

### L1-f6f7af4068  (L1, 2026-01-01, sha f6f7af406822, PR #15995)
TITLE: [Refactor] Clean up custom op (#15995)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.runner.marlin, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+6/-2); python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+3/-35); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+5/-82); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+13/-50); python/sglang/srt/layers/moe/moe_runner/marlin.py (+2/-4); python/sglang/srt/layers/moe/rocm_moe_utils.py (+7/-31); python/sglang/jit_kernel/utils.py (+1/-120); python/sglang/srt/distributed/parallel_state.py (+41/-93); python/sglang/srt/layers/dp_attention.py (+3/-3); python/sglang/srt/layers/elementwise.py (+3/-17); (+13 more)
LABELS: quant, amd, blackwell, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ We observe that there're many direct use `torch.ops.sglang` in code, which is not friendly to IDE (especially for type checking). This is a result of custom op registration. Currently, registering a custom op requires: ⏎  ⏎ 1. Define the custom op `example_op`. ⏎ 2. Define a fake implementation. ⏎ 3. Call `direct_register_custom_op`. ⏎ 4. After that, we can only use `torch.ops.sglang.example_op`. ⏎  ⏎ The fake impl is mainly for torch …[truncated]

### L1-b021332339  (L1, 2026-01-02, sha b0213323397c, PR #16227)
TITLE: [NemotronH] Add latent MoE support (#16227)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=2688,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=128,N=2688,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=256,N=1344,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=256,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=256,N=2688,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=256,N=2688,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=256,N=672,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=256,N=672,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); (+13 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Future NemotronH models will (conditionally) have a linear layer before (and after) the MoE layer, letting the MoE operate in a smaller hidden size. This PR enables it. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAIN …[truncated]

### L1-0d244116d2  (L1, 2026-01-02, sha 0d244116d28a, PR #13959)
TITLE: [DeepSeek v3.2] opt Context Parallelism: support fused moe, multi batch and fp8 kvcache (#13959)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_nextn.py (+5/-6); python/sglang/srt/models/deepseek_v2.py (+10/-16); python/sglang/srt/server_args.py (+15/-3); docs/advanced_features/server_arguments.md (+1/-0); docs/basic_usage/deepseek_v32.md (+12/-0); python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py (+5/-5); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+45/-68); python/sglang/srt/layers/attention/nsa/utils.py (+209/-5); python/sglang/srt/layers/attention/nsa_backend.py (+149/-20); python/sglang/srt/layers/communicator.py (+14/-4); (+4 more)
LABELS: documentation, deepseek, npu, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎  ⏎ The original default token splitting scheme of cp does not support prefill multi-batch. A new token splitting method is introduced to enable multi-batch support, fused MoE compatibility, and FP8 KV-cache support. Compared with the original DeepEP scheme, the combination of the tuned fused MoE backend and the new token splitting method **reduces TTFT by 8.9% (for inputs ≥16K tokens) to 32% (for 1K token inputs)** in 8× H20(141GB …[truncated]

### L1-f07e76b229  (L1, 2026-01-03, sha f07e76b229db, PR #16305)
TITLE: Multiple refactors of DeepSeek V32 and context parallel (#16305)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+2/-2); docs/basic_usage/deepseek_v32.md (+30/-21); python/sglang/srt/server_args.py (+16/-5); test/srt/run_suite.py (+3/-2); test/srt/test_deepseek_v32_basic.py (+56/-1); test/srt/test_deepseek_v32_cp_single_node.py (+2/-3); test/srt/test_deepseek_v32_mtp.py (+81/-1)
LABELS: documentation, deepseek
BODY: ## Motivation ⏎  ⏎ - Refine the handler in `server_args.py` ⏎ - Move dpsk v32 cp test to pr-test, and add pure tp tests ⏎ - Update the document for better readibility ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-me …[truncated]

### L1-6bc5a52fd2  (L1, 2026-01-03, sha 6bc5a52fd2d4, PR #16164)
TITLE: [NPU] Adapt qwen3-next W8A8 on NPU (#16164)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen3_next.py (+18/-5)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Fix the bugs on NPU when running TP+EP on the qwen3-next model, improve the performance, and adapt to W8A8 quantization. ⏎  ⏎ ## Modifications ⏎  ⏎ The qwen3-next model was missing the prefix parameters corresponding to each module; this PR adds the corresponding prefix string parameters. ⏎ A loose padding condition that could trigger a bug during TP+EP was fixed. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ gsm8k BF16 TP4EP4 on A3 NPU ⏎ <img width="1536"  …[truncated]

### L1-ff0f370f85  (L1, 2026-01-04, sha ff0f370f8531, PR #16127)
TITLE: ci: migrate MoE tests to test/registered/moe/ (#16127)
SOURCES: body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: test/registered/moe/test_cutedsl_moe.py (+4/-0); test/registered/moe/test_fused_moe.py (+5/-0); test/registered/moe/test_glm4_moe_models.py (+4/-0); test/registered/moe/test_moe_ep.py (+4/-0); test/registered/moe/test_torch_compile_moe.py (+4/-0); test/registered/moe/test_triton_fused_moe.py (+4/-0); test/registered/moe/test_triton_moe_channel_fp8_kernel.py (+4/-0); test/srt/run_suite.py (+0/-8)
LABELS: run-ci
BODY: ## Summary ⏎ - Migrate 7 MoE test files from `test/srt/` to `test/registered/moe/` ⏎ - Add CI registry decorators for automatic test discovery ⏎ - Remove migrated test entries from legacy `test/srt/run_suite.py` ⏎  ⏎ Part of #13808 (CI suites organization) ⏎  ⏎ ## Files Migrated ⏎  ⏎ **1-GPU Tests (stage-b-test-small-1-gpu):** ⏎ - `test_fused_moe.py` (80s, + AMD stage-a-test-1 at 30s) ⏎ - `test_triton_fused_moe.py` (12s) ⏎ - `test_triton_moe_channel_fp8_kernel.py` (16s) …[truncated]

### L1-0ff3747ca1  (L1, 2026-01-04, sha 0ff3747ca1f4, PR #16430)
TITLE: [model-gateway]: move unit tests to bindings/python/tests/ (#16430)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: sgl-model-gateway/bindings/python/src/sglang_router/router.py (+0/-0); .github/workflows/pr-test-rust.yml (+2/-2); .github/workflows/release-docker-gateway.yml (+1/-1); sgl-model-gateway/bindings/python/README.md (+20/-14); sgl-model-gateway/bindings/python/pyproject.toml (+11/-1); sgl-model-gateway/bindings/python/src/sglang_router/__init__.py (+0/-0); sgl-model-gateway/bindings/python/src/sglang_router/__main__.py (+0/-0); sgl-model-gateway/bindings/python/src/sglang_router/cli.py (+0/-0); sgl-model-gateway/bindings/python/src/sglang_router/launch_router.py (+0/-0); sgl-model-gateway/bindings/python/src/sglang_router/launch_server.py (+0/-0); (+8 more)
LABELS: documentation, dependencies, run-ci, model-gateway
BODY: Move Python binding unit tests from py_test/unit/ to bindings/python/tests/ where they logically belong with the Python binding package. ⏎  ⏎ Changes: ⏎ - Move test_validation.py, test_arg_parser.py, test_router_config.py, test_startup_sequence.py to bindings/python/tests/ ⏎ - Add __init__.py and conftest.py for pytest configuration ⏎ - Update pyproject.toml with pytest settings and dev dependency ⏎ - Update CI workflow to use new test path ⏎  ⏎  ⏎  ⏎ ## C …[truncated]

### L1-0fee6bc632  (L1, 2026-01-05, sha 0fee6bc6323e, PR #15836)
TITLE: [JIT kernel] Apply jit per_tensor_quant_fp8 kernel (#15836)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+5/-11); python/sglang/jit_kernel/csrc/gemm/per_tensor_quant_fp8.cuh (+43/-24); python/sglang/jit_kernel/per_tensor_quant_fp8.py (+13/-4); python/sglang/jit_kernel/tests/test_per_tensor_quant_fp8.py (+16/-0); python/sglang/srt/layers/quantization/fp8_kernel.py (+5/-5)
LABELS: quant, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-45ef834412  (L1, 2026-01-05, sha 45ef8344128d, PR #16280)
TITLE: Add MoE Integration Tests For CUTLASS Coverage (#16280)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: test/manual/layers/moe/test_moe_runners_1gpu.py (+23/-3); test/manual/layers/moe/test_moe_runners_4gpu.py (+116/-0)
BODY: ## Motivation ⏎  ⏎ Add 4 more MoE integration tests to achieve complete CUTLASS path coverage.  ⏎  ⏎ ## Modifications ⏎  ⏎ Adds coverage for these 4 CUTLASS kernels: `cutlass_fused_experts_fp8`, `cutlass_w4a8_moe`, `cutlass_w4a8_moe_deepep_normal`, `cutlass_w4a8_moe_deepep_ll`. ⏎  ⏎ `cutlass_moe_fp4` is already covered by an existing test case. Thus, all 5 CUTLASS paths are tested. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ `moe_runner_cutlass_fp8`: ⏎  ⏎ ```bash ⏎ Score: 0.60 …[truncated]

### L1-3e73e12458  (L1, 2026-01-07, sha 3e73e12458e3, PR #16676)
TITLE: Revert "Add SwapAB Optimization for triton fused_moe_kernel on SM90." (#16676)
SOURCES: path_core, subject_keyword, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+1/-40)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 15712 reason=hardware_specific_breakage
BODY: Reverts sgl-project/sglang#15712 ⏎  ⏎ pr [#15712](https://github.com/sgl-project/sglang/pull/15712) merged cause AMD CI failed [stage-a-test-1-amd (linux-mi325-gpu-1)](https://github.com/sgl-project/sglang/actions/runs/20787129171/job/59699719610#logs).  ⏎ @Fridge003  ⏎  ⏎ CC: @HaiShaw @Kangyan-Zhou

### L1-24b30f7757  (L1, 2026-01-07, sha 24b30f7757f8, PR #15151)
TITLE: MoE Refactor: Refactor `fp8.py` -> `flashinfer_trllm.py` (#15151)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:production-kernel-provenance
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-2); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+238/-0); python/sglang/srt/layers/moe/moe_runner/runner.py (+9/-0); python/sglang/srt/layers/quantization/fp8.py (+58/-200)
LABELS: run-ci
BODY: Part of https://github.com/sgl-project/sglang/issues/8715. There are many bugs recently related to Flashinfer MoE, partially bc this file and `modelopt_quant.py` is getting too complex.

### L1-ee4d2287ab  (L1, 2026-01-07, sha ee4d2287ab64, PR #15712)
TITLE: Add SwapAB Optimization for triton fused_moe_kernel on SM90. (#15712)
SOURCES: path_core, subject_keyword, corpus:confirmed-reverts(reverted), corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+40/-1)
LABELS: run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 16676 (confirmed_revert, reason=hardware_specific_breakage) || deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎  ⏎ In case of a small M dimension and  using fp8_w8a8 on SM90, SwapAB brings significant benefit by transposing input A, B to make better use of `WGMMA`. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ SwapAB is enabled under all the following conditions: ⏎ - use_fp8_w8a8 ⏎ - SM90 is supported ⏎ - config["BLOCK_SIZE_M"] < 64 and config["BLOCK_SIZE_N"] >= 64 ⏎  ⏎ If SwapAB is enabled, `a`, `b`, `a_scale`, `b_scale`, `accumulator` will be transposed before `tl …[truncated]

### L1-fb7609f1dd  (L1, 2026-01-08, sha fb7609f1ddbe, PR #16622)
TITLE: Fix FP8 MoE NaN with DeepGEMM on Blackwell (#16622)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+20/-8)
LABELS: run-ci
BODY: ## Motivation ⏎ Fix NaN issue in FP8 quantized MoE models (e.g., Qwen3-MoE) when using DeepGEMM on Blackwell GPUs. The root cause is that the UE8M0 scale requantization condition in process_weights_after_loading was not triggered correctly when MOE_RUNNER_BACKEND is AUTO but DeepGEMM is actually used (via --moe-a2a-backend deepep/mooncake). ⏎  ⏎ ## Modifications ⏎ Fixed DeepGEMM detection logic: Added logic in Fp8MoEMethod.process_weights_after_loadi …[truncated]

### L1-e46f79431b  (L1, 2026-01-08, sha e46f79431b80, PR #16458)
TITLE: Fix external_models import path and migrate model loading tests (#16458)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.deep_gemm
FILES: python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+1/-1); .github/workflows/pr-test.yml (+2/-2); python/sglang/test/external_models/custom_qwen2_vl.py (+0/-0); test/registered/model_loading/test_external_models.py (+6/-2); test/registered/model_loading/test_modelopt_export.py (+3/-0); test/registered/model_loading/test_modelopt_loader.py (+9/-11); test/registered/model_loading/test_utils_update_weights.py (+3/-0); test/srt/run_suite.py (+0/-5)
LABELS: run-ci
BODY: ## Summary ⏎ Fix ModuleNotFoundError on AMD CI by moving external_models to `python/sglang/test/external_models/` and migrate model loading tests to `test/registered/model_loading/`. ⏎  ⏎ ## Test plan ⏎ CI should pass with the new test locations.

### L1-d6d5c3fdea  (L1, 2026-01-09, sha d6d5c3fdea70, PR #11349)
TITLE: [AMD] Clean up vllm dependencies in moe_runner/triton.py (#11349)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L1.runner.triton
FILES: python/sglang/srt/layers/moe/moe_runner/triton.py (+15/-13)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ CC: @HaiShaw @kkHuang-amd

### L1-d27f16f38a  (L1, 2026-01-10, sha d27f16f38a6f, PR #13715)
TITLE: Fix EPLB + FP4 Quantization Compatibility Issue (#13715)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+12/-0); python/sglang/srt/models/bailing_moe.py (+4/-0); python/sglang/srt/models/deepseek_v2.py (+7/-1); python/sglang/srt/models/glm4_moe.py (+4/-0); python/sglang/srt/models/gpt_oss.py (+4/-0); python/sglang/srt/models/longcat_flash.py (+4/-0); python/sglang/srt/models/qwen2_moe.py (+7/-1); python/sglang/srt/models/qwen3_moe.py (+7/-1)
LABELS: deepseek, run-ci
BODY: ----------------------------------------------------------------------------------------- ⏎ Note:  The following description was generated by AI and may not be accurate. ⏎ ----------------------------------------------------------------------------------------- ⏎ # EPLB + FP4 Quantization Compatibility Issue Analysis and Fix ⏎  ⏎ ## Problem Description ⏎  ⏎ When using EPLB (Expert-based Load Balancing) for expert rebalancing, the system crashes with an  …[truncated]

### L1-67b61a4e8d  (L1, 2026-01-10, sha 67b61a4e8d0d, PR #16723)
TITLE: [Rework] Add SwapAB Optimization for triton fused_moe_kernel on SM90. (#16723)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+46/-1)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎ Rework of reverted pr https://github.com/sgl-project/sglang/pull/15712, with AMD CI failures fixed. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS …[truncated]

### L1-9fd2358cc2  (L1, 2026-01-10, sha 9fd2358cc2fc, PR #16838)
TITLE: Update Cutedsl version and pin cuda-python version (#16838)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-2); python/pyproject.toml (+2/-2)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎ In prior efforts of upgrading cutedsl to >= 4.3.1, the driver will throw the following error: ⏎ ``` ⏎ Error Code: 35 ⏎  ⏎ 🔍 Additional Context: ⏎ - Error name: (<cudaError_t.cudaSuccess: 0>, b'cudaErrorInsufficientDriver') ⏎ - Error code: 35 ⏎ - CUDA_TOOLKIT_PATH: not set ⏎ - Target SM ARCH: not set ⏎  ⏎ 📊 GPU Information: ⏎ - CUDA devices available: 8 (current: <CUdevice 0>) ⏎ - Architecture: Blackwell (sm_100a) ⏎ - Compatible SM archs: sm_1 …[truncated]

### L1-3c16c58619  (L1, 2026-01-11, sha 3c16c58619d1, PR #16300)
TITLE: [model-gateway] Add Redis support as a history backend (#16300)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-model-gateway/bindings/python/src/sglang_router/router.py (+20/-0); docs/advanced_features/sgl_model_gateway.md (+17/-0); sgl-model-gateway/Cargo.toml (+2/-0); sgl-model-gateway/README.md (+21/-0); sgl-model-gateway/bindings/python/src/lib.rs (+50/-0); sgl-model-gateway/bindings/python/src/sglang_router/router_args.py (+29/-1); sgl-model-gateway/src/config/builder.rs (+16/-2); sgl-model-gateway/src/config/types.rs (+52/-0); sgl-model-gateway/src/data_connector/factory.rs (+50/-4); sgl-model-gateway/src/data_connector/mod.rs (+1/-0); (+2 more)
LABELS: documentation, dependencies, run-ci, model-gateway
BODY: ## Motivation ⏎ #16157  ⏎  ⏎ - Added Redis as a history backend ⏎ - Critical for use cases that requires rapid creation & retrieval of conversation history ⏎  ⏎ ## Modifications ⏎  ⏎ ### Main Structures ⏎ - `RedisStore` ⏎ - `RedisConversationStorage` ⏎ - `RedisConversationItemStorage` ⏎ - `RedisResponseStorage` ⏎  ⏎ ### Data Schema ⏎  ⏎ Entity | Redis Key Pattern | Data Type | Description ⏎ -- | -- | -- | -- ⏎ **Conversation** | conversation:{id} | Hash | Stores m …[truncated]
