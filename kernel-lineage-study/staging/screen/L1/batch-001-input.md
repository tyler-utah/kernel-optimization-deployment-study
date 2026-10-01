### L1-6b0aeb58fd  (L1, 2025-02-20, sha 6b0aeb58fd53, PR #3692)
TITLE: [moe] optim: reduce memory consumption in fused_moe (#3692)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+6/-5)
BODY: ## Motivation ⏎  ⏎  ⏎ This PR can partially address #3633. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ We reuse the memory of `intermediate_cache1` to create `intermediate_cache3`. ⏎  ⏎ Here is the test script ⏎  ⏎ ```python ⏎ import torch ⏎ from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe ⏎  ⏎ N = 64 * 1024 ⏎ E = 8 ⏎ H = 4096 ⏎ I = 8192 ⏎  ⏎ torch.manual_seed(0) ⏎  ⏎ x = torch.randn((N, H), device="cuda", dtype=torch.float16) / 32 ⏎ w1 = torch.randn((E, I * 2,  …[truncated]

### L1-5c54ef0352  (L1, 2025-02-21, sha 5c54ef0352dc, PR #3747)
TITLE: AMD/ROCm: update AITER repo to ROCm/aiter (#3747)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-2); docker/Dockerfile.rocm (+5/-4); python/sglang/srt/layers/quantization/fp8.py (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Code Formatting with Pre-Commit](https://docs.sglang.ai/references/contribution_guide.html#code-formatting-with-pre-commit). ⏎ - [+] Add unit tests as outlined in the [Running Unit Tests](https://docs.sglang.ai/references/contribution_guide.html#running-unit-tests-adding-to-ci). ⏎ - [+] Update documentation / docstrings / example tutorials as neede …[truncated]

### L1-d5d80ab477  (L1, 2025-02-21, sha d5d80ab477de, PR #3705)
TITLE: [Bugfix] Fix scores mask for moe topk (#3705)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+3/-1)
BODY: ## Motivation ⏎  ⏎ In the implementation of grouped topk in MoE layer, the scores of masked groups are set to 0, which may leads to select incorrect experts in certain scenarios. ⏎  ⏎ ## Modifications ⏎  ⏎ In SGLang, the mask scores are set to 0. This configuration may result in the selection of experts in masked groups if the scores in the unmasked groups are negative. This behavior can lead to incorrect or suboptimal selections in certain scenarios. …[truncated]

### L1-0c227ee373  (L1, 2025-02-21, sha 0c227ee373ac, PR #3680)
TITLE: feat: update grouped_topk to support softmax and sigmoid (#3680)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+10/-3)
DEEP_STUDY: deep-study: this PR was reverted by PR 4505 (confirmed_revert, reason=other)
BODY: ### Motivation ⏎ This PR is to support both softmax and sigmoid scoring functions in grouped_topk. Also, verified DeepSeek V2/V3/R1 uses biased_grouped_top and updated the corresponding comments. ⏎ Ref https://github.com/sgl-project/sglang/issues/2739 ⏎  ⏎ ### Modifications ⏎ #### Checklist

### L1-1df6eabd5d  (L1, 2025-02-21, sha 1df6eabd5d36, PR #3740)
TITLE: feat: Add SageMaker support (#3740)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.sagemaker (+78/-0); docker/serve (+31/-0); python/sglang/srt/entrypoints/http_server.py (+12/-0); test/srt/test_sagemaker_server.py (+178/-0)
BODY: ## Motivation ⏎  ⏎ SageMaker Endpoints support /ping for healthchecks and /invocations for invocation payloads however sglang currently doesn't support this invocation pattern to make the package usable on SageMaker Endpoints. ⏎  ⏎ ## Modifications ⏎  ⏎ This pull request adds two endpoints for `/ping`/ and `/invocations` in `http_server.py`. ⏎  ⏎ `/ping` provides the same functionality as `/health`. At present `/invocations` acts the same as `/v1/chat/co …[truncated]

### L1-27a46317b6  (L1, 2025-02-24, sha 27a46317b648, PR #3813)
TITLE: Fix dependency (#3813)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+35/-13); python/sglang/srt/constrained/outlines_backend.py (+3/-9); python/sglang/srt/layers/sampler.py (+3/-3); python/sglang/srt/server_args.py (+0/-4); test/lang/test_srt_backend.py (+1/-1); test/srt/models/test_qwen_models.py (+1/-1)
BODY: One item per row instead of packing multiple together

### L1-1a6e97577a  (L1, 2025-02-24, sha 1a6e97577acb, PR #3730)
TITLE: Feature DeepSeek V3/R1 INT8 Quantization (block-wise) (#3730)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+46/-5); python/sglang/srt/layers/linear.py (+1/-0); python/sglang/srt/layers/quantization/__init__.py (+2/-0); python/sglang/srt/layers/quantization/blockwise_int8.py (+406/-0); python/sglang/srt/layers/quantization/int8_kernel.py (+327/-0); python/sglang/srt/layers/quantization/int8_utils.py (+73/-0); python/sglang/srt/models/deepseek_v2.py (+15/-0); test/srt/run_suite.py (+1/-0); test/srt/test_block_int8.py (+221/-0)
BODY: ## Motivation ⏎ Support block-wise INT8 quantization for DeepSeek V3/R1.  ⏎ INT8 is a friendly type for most hardware platforms. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Fork FP8 implementation and support the block-wise quantization on INT8. ⏎ - INT8 R1 weights is released in https://huggingface.co/meituan/DeepSeek-R1-Block-INT8. ⏎  ⏎  ⏎  ⏎ ## Performance ⏎ In A100*32 with TP exclusively, we observe **no accuracy loss** and up tp **33%** performance enhancement ⏎ | Mo …[truncated]

### L1-21463e321a  (L1, 2025-02-26, sha 21463e321ad8, PR #3602)
TITLE: Expert Parallelism (EP) Support for DeepSeek V3/R1 (#3602)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+59/-8); python/sglang/srt/layers/moe/ep_moe/layer.py (+128/-27); python/sglang/test/test_block_fp8_ep.py (+361/-0)
BODY: ## Motivation ⏎ Expert Parallelism (EP) Support for DeepSeek  V3/R1。 ⏎  ⏎ ## Modifications ⏎ * the group GEMM operator supports FP8 ⏎ *  supports DeepSeek V3 parameter loading. ⏎  ⏎ ## Performence ⏎ The performance improved by approximately 5% on a single H200 machine. ⏎ <!DOCTYPE html> ⏎ H200*8 | Input token throughput (tok/s) | Output token throughput (tok/s) ⏎ -- | -- | -- ⏎ EP=8 | 677.82 | 1468.48 ⏎ TP=8 | 647.60 | 1403.02 ⏎  ⏎  ⏎ *test command* ⏎ ``` ⏎ python …[truncated]

### L1-564bdf29f7  (L1, 2025-02-27, sha 564bdf29f7ef, PR #3934)
TITLE: upgrade flashinfer v0.2.2.post1 (#3934)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); docs/start/install.md (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b0df5d240b  (L1, 2025-02-27, sha b0df5d240b4f, PR #3922)
TITLE: Tuning Script for Feature DeepSeek V3/R1 INT8 Quantization (block-wise) (#3922)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=128,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+22/-4); benchmark/kernels/quantization/tuning_block_wise_kernel.py (+63/-24); python/sglang/srt/layers/quantization/configs/N=1536,K=1536,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=1536,K=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=2048,K=512,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=2304,K=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=24576,K=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=256,K=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=32768,K=512,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); (+6 more)
BODY: ## Motivation ⏎ Continue #3730 , tuning script does not support block-wise INT8 yet. ⏎  ⏎  ⏎ ## Modifications ⏎ Add script for tuning and provide configs for A100. ⏎  ⏎  ⏎ ## Performance ⏎ In A100x8x2nodes, output throughput (qps=128) improves **9%** and output throughput improves **50%** ⏎ | Tune | Output Throughputs(qps=128) | Output Throughputs(bs=1) | ⏎ |------|-----------------------------|--------------------------| ⏎ | w/  | 2426.61 (**+9%**)          …[truncated]

### L1-1c96fa86cf  (L1, 2025-02-27, sha 1c96fa86cfa2, PR #3613)
TITLE: [MOE] enable efficient moe_alignment multi-blocks execution (3x~6x) (#3613)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.align.cuda_aot
FILES: sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+266/-52); benchmark/kernels/fused_moe_triton/benchmark_deepseekv3_moe_align_blocks.py (+81/-38); sgl-kernel/src/sgl-kernel/include/utils.h (+30/-0); sgl-kernel/tests/test_moe_align.py (+2/-2)
DEEP_STUDY: deep-study: this PR was reverted by PR 3982 (confirmed_revert, reason=premature_or_process)
BODY: ## Motivation ⏎  ⏎ The new algorithm is the adpation and the follow up of [moe-align-with-multiple-blocks-execution](https://github.com/yiakwy-xpu-ml-framework-team/AMD-sglang-benchmark-fork/blob/try_to_optimize_moe_align_block_size_multiblocks_cuda_kernel/sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu) in [PR#3137](https://github.com/sgl-project/sglang/pull/3137) and [PR#2970](https://github.com/sgl-project/sglang/pull/2970)  ⏎  ⏎ |    base (mas …[truncated]

### L1-90bc26a813  (L1, 2025-02-27, sha 90bc26a813ac, PR #3950)
TITLE: set a strict sgl-kernel version (#3950)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎ Required by mercy. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f3b99f73b3  (L1, 2025-02-28, sha f3b99f73b391, PR #)
TITLE: update flashinfer-python version
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L1-18bb216c28  (L1, 2025-02-28, sha 18bb216c28e6, PR #3982)
TITLE: Revert "[MOE] enable efficient moe_alignment multi-blocks execution (3x~6x)" (#3982)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.align.cuda_aot
FILES: sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+52/-266); benchmark/kernels/fused_moe_triton/benchmark_deepseekv3_moe_align_blocks.py (+38/-81); sgl-kernel/src/sgl-kernel/include/utils.h (+0/-30); sgl-kernel/tests/test_moe_align.py (+2/-2)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 3613 reason=premature_or_process
BODY: Reverts sgl-project/sglang#3613 ⏎  ⏎ Will merge it after further modification from @BBuf

### L1-ac2387279e  (L1, 2025-03-03, sha ac2387279ea9, PR #3988)
TITLE: Support penalty in overlap mode; return logprob with chunked prefill; improve benchmark scripts (#3988)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+67/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+12/-0); python/sglang/srt/layers/moe/fused_moe_native.py (+2/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+31/-10); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+18/-2); benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py (+10/-1); benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py (+9/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+9/-0); docs/backend/native_api.ipynb (+2/-3); docs/backend/server_arguments.md (+0/-1); (+76 more)
BODY: - Support penalty in overlap mode ⏎ - Support chunked prefill + input logprob ⏎ - Improve benchmark script and profiler ⏎ - rename "token_ids" to "output_ids" in the return value when using `--skip-tokenizer-init` ⏎  ⏎  ⏎ ``` ⏎  ⏎ ```

### L1-66301e124f  (L1, 2025-03-03, sha 66301e124f19, PR #4021)
TITLE: Improve code styles (#4021)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+0/-1); benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py (+5/-5); python/sglang/bench_serving.py (+1/-1); python/sglang/lang/backend/runtime_endpoint.py (+1/-6); python/sglang/srt/layers/logits_processor.py (+1/-1); python/sglang/srt/managers/data_parallel_controller.py (+0/-3); python/sglang/srt/managers/io_struct.py (+1/-1); python/sglang/srt/managers/schedule_batch.py (+2/-5); python/sglang/srt/managers/scheduler.py (+0/-8); python/sglang/srt/metrics/collector.py (+0/-114); (+4 more)
BODY: 

### L1-6b45a21d16  (L1, 2025-03-03, sha 6b45a21d16a3, PR #4025)
TITLE: Reorganize c++ source files in sgl-kernel with multiple folders  (#4025)
SOURCES: path_core
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/src/sgl-kernel/csrc/moe/moe_align_kernel.cu (+0/-0); sgl-kernel/setup.py (+17/-11); sgl-kernel/setup_rocm.py (+2/-2); sgl-kernel/src/sgl-kernel/csrc/activation/fused_add_rms_norm_kernel.cu (+0/-0); sgl-kernel/src/sgl-kernel/csrc/allreduce/custom_all_reduce.hip (+0/-0); sgl-kernel/src/sgl-kernel/csrc/allreduce/custom_all_reduce_hip.cuh (+0/-0); sgl-kernel/src/sgl-kernel/csrc/allreduce/trt_reduce_internal.cu (+0/-0); sgl-kernel/src/sgl-kernel/csrc/allreduce/trt_reduce_kernel.cu (+0/-0); sgl-kernel/src/sgl-kernel/csrc/gemm/cublas_grouped_gemm.cu (+0/-0); sgl-kernel/src/sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu (+0/-0); (+10 more)
BODY: Organize them into different folders  ⏎ - allreduce ⏎ - gemm ⏎ - moe ⏎ - ...

### L1-110e006673  (L1, 2025-03-03, sha 110e0066735a, PR #4027)
TITLE: Reorganize python source files in sgl-kernel with multiple files  (#4027)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-kernel/src/sgl-kernel/ops/moe.py (+24/-0); sgl-kernel/src/sgl-kernel/__init__.py (+33/-101); sgl-kernel/src/sgl-kernel/ops/__init__.py (+0/-677); sgl-kernel/src/sgl-kernel/ops/activation.py (+153/-0); sgl-kernel/src/sgl-kernel/ops/allreduce.py (+78/-0); sgl-kernel/src/sgl-kernel/ops/attention.py (+8/-0); sgl-kernel/src/sgl-kernel/ops/gemm.py (+111/-0); sgl-kernel/src/sgl-kernel/ops/sampling.py (+211/-0); sgl-kernel/src/sgl-kernel/ops/speculative.py (+84/-0); sgl-kernel/src/sgl-kernel/ops/utils.py (+2/-2); (+1 more)
BODY: Do not write implementation code in `__init__.py`. ⏎  ⏎ Organize them into different files  ⏎ - allreduce.py ⏎ - gemm.py ⏎ - moe.py ⏎ - ...

### L1-935cda944b  (L1, 2025-03-03, sha 935cda944b82, PR #4032)
TITLE: Misc clean up; Remove the support of jump forward (#4032)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); docs/backend/function_calling.ipynb (+1/-1); docs/backend/sampling_params.md (+257/-46); docs/backend/server_arguments.md (+1/-2); docs/backend/speculative_decoding.ipynb (+3/-3); docs/references/contribution_guide.md (+1/-1); docs/references/multi_node.md (+1/-1); docs/start/install.md (+7/-7); examples/runtime/engine/offline_batch_inference_eagle.py (+1/-1); python/sglang/README.md (+0/-1); (+31 more)
BODY: - Remove jump forward to simplify the code maintenance  ⏎ - Rename `function_call` to `parse_function_call` ⏎ - Rename python/sglang/srt/layers/attention/__init__.py  -> python/sglang/srt/layers/attention/base_attn_backend.py ⏎ - Revert https://github.com/sgl-project/sglang/pull/3260. We need good type annotation and examples to demonstrate how to use these parameters. ⏎ - Do not import `from sglang.lang.chat_template import get_chat_template_by_mode …[truncated]

### L1-11eea69e70  (L1, 2025-03-03, sha 11eea69e70aa, PR #4049)
TITLE: Fix assert options.num_stages != 0 error in the latest ROCm build image (#4049)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=14336,device_name=AMD_Instinct_MI300X.json (+18/-18); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=14336,device_name=AMD_Instinct_MI325X.json (+18/-18); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=14336,device_name=AMD_Radeon_Graphics.json (+18/-18); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=1792,device_name=AMD_Instinct_MI300X.json (+18/-18); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=1792,device_name=AMD_Instinct_MI325X.json (+18/-18); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=1792,device_name=AMD_Radeon_Graphics.json (+18/-18); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=3584,device_name=AMD_Instinct_MI300X.json (+18/-18); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=3584,device_name=AMD_Instinct_MI325X.json (+18/-18); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=3584,device_name=AMD_Radeon_Graphics.json (+18/-18); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=4096,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json (+16/-16); (+8 more)
BODY: ## Motivation ⏎  ⏎ With the new ROCm sglang image release and the triton version upgraded to 3.2, the related tuning configuration file needs adjustments to avoid assertion errors. ⏎  ⏎ ## Modifications ⏎  ⏎ Modify the related configuration files in the sglang folder ⏎  ⏎ ## Checklist

### L1-51d25405a7  (L1, 2025-03-04, sha 51d25405a7cc, PR #4053)
TITLE: ROCm: update aiter and its usage to fused moe (bloat16, fp8, fp8 block-quant) (#4053)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+16/-11); python/sglang/srt/layers/quantization/fp8.py (+64/-27); docker/Dockerfile.rocm (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎ Performance boost (DSv3 +25~35%) ⏎  ⏎ ``` ⏎ /sgl-workspace/sglang# RCCL_MSCCL_ENABLE=0 CK_MOE=1 python -m sglang.bench_one_batch --batch-size 64 --input 2048 --output 256 --model /data/deepseek-ai/DeepSeek-V3/ --tp 8 --trust-remote-code --quantization fp8 ⏎ Benchmark ... ⏎ Prefill. latency: 8.12619 s, throughput:  16129.57 token/s ⏎ Decode.  latency: 0.04280 s, throughput:   1495.39 token/s ⏎ Decode.  latency: 0.04312 s, throughput:   …[truncated]

### L1-71ab0dabe0  (L1, 2025-03-05, sha 71ab0dabe0de, PR #4081)
TITLE: Fix the moe padding conditional logic (#4081)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+6/-1)
BODY: ## Motivation ⏎  ⏎  ⏎ Fix logic error. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-13bc39c5d6  (L1, 2025-03-06, sha 13bc39c5d631, PR #4152)
TITLE: ROCm: enable trillion-parameter MoE models with INT4-FP8 single node (#4152)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+12/-0); python/sglang/srt/layers/quantization/fp8.py (+110/-22); python/sglang/srt/utils.py (+2/-1)
BODY: INT4 MoE weights, FP8 compute ⏎  ⏎ credits: @shengnxu, @coderfeli, @carlushuang, @kkHuang-amd , @leishaoSC, @valarLip, @HaiShaw   ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ Enable models with more than 1.2 trillion parameters on single node of `8xMI300/MI308`. ⏎ Speedup decoding performance from INT4 weight, lowered memory bandwidth. ⏎ Use the latest FP8 Tensor Core for computation (available to MI300, MI308). ⏎  ⏎ Model used can be accessed at `https://huggingface.co/am …[truncated]

### L1-c7f254468f  (L1, 2025-03-06, sha c7f254468fca, PR #3888)
TITLE: [Feature] DeepSeek V3/R1 INT8 Quantization (channel-wise)  (#3888)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+37/-6); python/sglang/srt/layers/quantization/w8a8_int8.py (+152/-3); python/sglang/srt/models/deepseek_v2.py (+16/-12); test/srt/run_suite.py (+1/-0); test/srt/test_int8_kernel.py (+163/-0)
BODY: ## Motivation ⏎  ⏎ Support channel-wise INT8 quantization for DeepSeek V3/R1. ⏎ INT8 is a friendly type for most hardware platforms. ⏎  ⏎ ## Modifications ⏎  ⏎ Co-author: @yych0745 @sleepcoo @b0urnee ⏎  ⏎ - Moe: Fused moe triton kernel supports channel-wise int8 quantization. ⏎ - Norm Linear: Use the cutlass implementation of w8a8 int8. ⏎ - Quantization config: Support `W8A8Int8MoEMethod ` in w8a8 int8 config. ⏎ - Unit test: Add unit test of channel-wise int …[truncated]

### L1-0beea4503f  (L1, 2025-03-07, sha 0beea4503f4f, PR #4178)
TITLE: ROCm: Flex Attention Enablement with custom backends (#4178)
SOURCES: path_core
ARTIFACT_HINTS: L1.align.hip_variant
FILES: sgl-kernel/src/sgl-kernel/csrc/moe/moe_align_kernel.hip (+118/-0); docker/Dockerfile.rocm (+3/-2); python/sglang/srt/layers/attention/aiter_backend.py (+605/-0); python/sglang/srt/layers/attention/aiter_decode_backend.py (+535/-0); python/sglang/srt/model_executor/model_runner.py (+59/-27); python/sglang/srt/server_args.py (+17/-7); sgl-kernel/src/sgl-kernel/include/utils_hip.h (+98/-0)
DEEP_STUDY: deep-study: this PR was reverted by PR 4186 (confirmed_revert, reason=premature_or_process)
BODY: Credits: @poyenc , @amd-hhashemi , @linsun12 , @tenpercent , @carlushuang , @HaiShaw  ⏎  ⏎ ## Motivation ⏎  ⏎ Add ROCm custom attention backends to enable Flex Attn. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ `--attention-backend aiter`: prefill: flashinfer-mod + decode: aiter decode (flex attn feasible) ⏎ `--attention-backend aiter_decode`:  prefill: triton + decode: aiter decode ⏎  ⏎ ROCm default backend: triton (no need to assign, flex attn infeasible) ⏎ `--attention …[truncated]

### L1-eb61f5c9af  (L1, 2025-03-07, sha eb61f5c9af73, PR #4186)
TITLE: Revert "ROCm: Flex Attention Enablement with custom backends (#4178)" (#4186)
SOURCES: path_core
ARTIFACT_HINTS: L1.align.hip_variant
FILES: sgl-kernel/src/sgl-kernel/csrc/moe/moe_align_kernel.hip (+0/-118); docker/Dockerfile.rocm (+2/-3); python/sglang/srt/layers/attention/aiter_backend.py (+0/-605); python/sglang/srt/layers/attention/aiter_decode_backend.py (+0/-535); python/sglang/srt/model_executor/model_runner.py (+27/-59); python/sglang/srt/server_args.py (+7/-17); sgl-kernel/src/sgl-kernel/include/utils_hip.h (+0/-98)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 4178 reason=premature_or_process
BODY: This reverts commit 0beea4503f4f23f6f4348748c610af4c4233d2fc. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d052f4c8a9  (L1, 2025-03-07, sha d052f4c8a9fb, PR #4194)
TITLE: New clang format for sgl kernel (#4194)
SOURCES: path_core
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/src/sgl-kernel/csrc/moe/moe_align_kernel.cu (+37/-16); python/upload_pypi.sh (+0/-6); sgl-kernel/.clang-format (+7/-0); sgl-kernel/src/sgl-kernel/csrc/activation/fused_add_rms_norm_kernel.cu (+9/-4); sgl-kernel/src/sgl-kernel/csrc/allreduce/custom_all_reduce_hip.cuh (+72/-44); sgl-kernel/src/sgl-kernel/csrc/allreduce/trt_reduce_internal.cu (+29/-14); sgl-kernel/src/sgl-kernel/csrc/allreduce/trt_reduce_kernel.cu (+20/-10); sgl-kernel/src/sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu (+31/-15); sgl-kernel/src/sgl-kernel/csrc/cutlass_extensions/epilogue/epilogue_per_row_per_col_scale.h (+41/-22); sgl-kernel/src/sgl-kernel/csrc/cutlass_extensions/gemm/dispatch_policy.hpp (+12/-8); (+15 more)
BODY: 

### L1-b93ef5e56d  (L1, 2025-03-07, sha b93ef5e56d5e, PR #4164)
TITLE: Remove the vllm dependency from the moe_align function  (#4164)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/src/sgl-kernel/csrc/moe/moe_align_kernel.cu (+10/-8); sgl-kernel/tests/test_moe_align.py (+5/-3)
BODY: Remove the vllm dependency from the moe_align function and support the sgl_moe_align_block_size with expert sizes that are multiples of 32. ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Test Case

### L1-4a893d142d  (L1, 2025-03-08, sha 4a893d142ded, PR #3749)
TITLE: Refactor Dockerfile: unify CUDA logic and reduce image size by ~2.6 GB (#3749)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+7/-32)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR refactors the Dockerfile to unify the handling of various CUDA versions into a more concise approach. Additionally, it removes redundant layers and streamlines installations, resulting in a reduced final Docker image size by approximately 2.6 GB. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Consolidated multiple if-else statements for CUDA version checks into a single RUN block with exported variables. ⏎ - Removed duplicate or unneede …[truncated]

### L1-8abf74e3c9  (L1, 2025-03-08, sha 8abf74e3c935, PR #4213)
TITLE: Rename files in sgl kernel to avoid nested folder structure (#4213)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+0/-0); sgl-kernel/python/sgl_kernel/moe.py (+1/-2); .github/workflows/release-pypi-kernel.yml (+1/-1); .github/workflows/release-whl-kernel.yml (+2/-2); python/sglang/srt/_custom_ops.py (+15/-15); sgl-kernel/Makefile (+5/-5); sgl-kernel/README.md (+6/-7); sgl-kernel/csrc/allreduce/custom_all_reduce.hip (+0/-0); sgl-kernel/csrc/allreduce/custom_all_reduce_hip.cuh (+0/-0); sgl-kernel/csrc/allreduce/trt_reduce_internal.cu (+0/-0); (+37 more)
BODY: - Create a new folder `sgl_kernel` for the python package. ⏎ - Reduce the nested folder structure. Use a a more flat structure. ⏎      - rename `sgl-kernel/src/sgl-kernel/csrc` to `sgl-kernel/csrc` ⏎      - rename `sgl-kernel/src/sgl-kernel/ops` to `sgl-kernel/python/sgl_kernel` ⏎ - Move `torch_extension.cu` under `csrc` ⏎ - Rename `activation.cu` to `elementwise.cu` ⏎ - Rename `sgl_kernels_ops.h` to `sgl_kernel_ops.h` ⏎ - Rename `torch.ops.sgl_kernels` …[truncated]

### L1-89ccb533ad  (L1, 2025-03-08, sha 89ccb533ad39, PR #4224)
TITLE: use sgl-kernel 0.0.4 (#4224)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-df84ab2a5b  (L1, 2025-03-09, sha df84ab2a5b87, PR #4228)
TITLE: update sgl-kernel 3rdparty (#4228)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+0/-3); sgl-kernel/3rdparty/cutlass (+1/-1); sgl-kernel/3rdparty/turbomind (+0/-1); sgl-kernel/README.md (+0/-1); sgl-kernel/setup.py (+0/-3)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c553e1604c  (L1, 2025-03-10, sha c553e1604c4e, PR #4165)
TITLE: DeepGemm integrate to sgl-kernel (#4165)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+3/-0); sgl-kernel/3rdparty/deepgemm (+1/-0); sgl-kernel/build.sh (+4/-3); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/setup.py (+52/-1); sgl-kernel/tests/test_deep_gemm.py (+263/-0)
LABELS: high priority
BODY: ## Motivation ⏎ Integrate DeepGemm in setup. ⏎ Linear usage: #4199 . ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ ## Checklist

### L1-00d25a7f5e  (L1, 2025-03-10, sha 00d25a7f5e2f, PR #4258)
TITLE: Fix quantization and nightly tests (#4258)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+0/-1); python/sglang/srt/layers/quantization/__init__.py (+88/-68); python/sglang/srt/model_executor/model_runner.py (+4/-0); python/sglang/test/test_utils.py (+4/-1); test/srt/run_suite.py (+1/-0); test/srt/test_awq.py (+44/-0); test/srt/test_nightly_gsm8k_eval.py (+1/-0)
BODY: Monkey patch functions like `AWQMoEMethod` from vllm to align the arguments ⏎  ⏎ ``` ⏎ python -m sglang.launch_server --model-path cognitivecomputations/DeepSeek-V3-AWQ --tp-size 8 ⏎ ```

### L1-5a6400eec5  (L1, 2025-03-10, sha 5a6400eec5f3, PR #4256)
TITLE: Test no vllm custom allreduce (#4256)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); .github/workflows/pr-test.yml (+1/-1); python/sglang/srt/server_args.py (+2/-2); scripts/ci_install_dependency.sh (+1/-1)
BODY: 

### L1-d3ecd63204  (L1, 2025-03-11, sha d3ecd6320433, PR #4136)
TITLE: Add A800 tuning configs support DeepSeek V3/R1 BF16 and INT8(block-wise) (#4136)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=160,N=192,device_name=NVIDIA_A800-SXM4-80GB.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=128,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=64,device_name=NVIDIA_A800-SXM4-80GB.json (+146/-0); python/sglang/srt/layers/quantization/configs/N=1536,K=1536,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=1536,K=7168,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=2048,K=512,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=2304,K=7168,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=24576,K=7168,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=256,K=7168,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=32768,K=512,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); (+6 more)
BODY: ## Motivation ⏎  ⏎ Support DeepSeek V3/R1 BF16 and INT8(block-wise) configs.  ⏎ See https://github.com/sgl-project/sglang/issues/3748 ⏎  ⏎ ## Modifications ⏎  ⏎ - DeepSeek V3/R1 BF16 config ⏎ - DeepSeek V2.5 BF16 config ⏎ - DeepSeek INT8(block-wise) configs ⏎  ⏎ ## Checklist

### L1-4d27eb9ad1  (L1, 2025-03-11, sha 4d27eb9ad1f1, PR #4291)
TITLE: update sgl-kernel 0.0.4.post2 (#4291)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6a02b32d07  (L1, 2025-03-11, sha 6a02b32d0785, PR #4287)
TITLE: Add A100 tuning configs for DeepSeek R1/V3 channel-wise INT8  (#4287)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=128,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8.json (+146/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+6/-1)
BODY: ## Motivation ⏎ tuning script does not support per-channel INT8 yet. ⏎  ⏎ ## Modifications ⏎ Add script for tuning and provide configs for A100. ⏎  ⏎ ## Performance ⏎ In A100x8x2nodes, output throughput (qps=128) improves 23% and output throughput improves 12% ⏎  ⏎ Tune | Output Throughputs(qps=128) | Output Throughputs(bs=1) ⏎ -- | -- | -- ⏎ w/ | 3121.85 (+23%) | 38.30 (+12%) ⏎ w/o | 2517.91 | 34.17 ⏎  ⏎  ⏎ ## Checklist

### L1-690e1f2371  (L1, 2025-03-11, sha 690e1f23716c, PR #4311)
TITLE: [AMD] Fix rocm sgl-kernel missing modules error (#4311)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); sgl-kernel/csrc/torch_extension_rocm.cc (+1/-1); sgl-kernel/setup_rocm.py (+1/-1)
BODY: ## Motivation ⏎ [Bug] missing allreduce from sgl_kernel module ⏎ [Bug] repeated nv sgl-kernel installation on amd platform ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0f2a2e3c19  (L1, 2025-03-11, sha 0f2a2e3c19ef, PR #4220)
TITLE: Add H20 tuning configs support DeepSeek V3/R1 INT8(block-wise) (#4220)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=256,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=24576,K=7168,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=3072,K=1536,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=3072,K=7168,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=32768,K=512,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=4096,K=512,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=4608,K=7168,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=512,K=7168,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=576,K=7168,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=16384,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128, 128].json (+146/-0); (+4 more)
BODY: ## Motivation ⏎  ⏎ Support DeepSeek V3/R1 INT8(block-wise) configs on H20. ⏎ See https://github.com/sgl-project/sglang/pull/3922 ⏎  ⏎ ## Modifications ⏎ DeepSeek INT8(block-wise) configs ⏎ ## Performance ⏎  ⏎ `python3 -m sglang.launch_server --model /path/to/DeepSeek-R1-INT8 --tp 8  --trust-remote --enable-torch-compile --torch-compile-max-bs 8 --enable-flashinfer-mla` ⏎  ⏎ In H20x8, output throughput (batch_size=1) improves 50.9% ⏎  ⏎ Tune | Output Throughpu …[truncated]

### L1-1cf63485c1  (L1, 2025-03-11, sha 1cf63485c1ef, PR #4317)
TITLE: upgrade flashinfer 0.2.3 (#4317)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); docs/start/install.md (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d1da58e275  (L1, 2025-03-11, sha d1da58e275e3, PR #4321)
TITLE: unify is_cuda and is_hip (#4321)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+2/-1); python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+9/-9); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-7); python/sglang/srt/custom_op.py (+5/-3); python/sglang/srt/distributed/device_communicators/custom_all_reduce.py (+18/-17); python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+6/-6); python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py (+3/-3); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+4/-4); python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py (+3/-3); (+8 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-7140ba3573  (L1, 2025-03-11, sha 7140ba357384, PR #4323)
TITLE: Add A800 tuning configs for DeepSeek R1/V3 channel-wise INT8 (#4323)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=128,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8.json (+146/-0)
BODY: ## Motivation ⏎ Add A800 tuning configs for DeepSeek R1/V3 channel-wise INT8. ⏎ See https://github.com/sgl-project/sglang/pull/4287 ⏎  ⏎ ## Modifications ⏎ Add config `E=256,N=128,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8.json`. ⏎  ⏎ ## Performance ⏎ In A800x8x2nodes, output throughput (qps=128) improves 23.6% and output throughput improves 9.7% ⏎  ⏎ | Tune | Output Throughputs(qps=128) | Output Throughputs(qps=1) | ⏎ |---------|---------|---------| …[truncated]

### L1-7130a7cea9  (L1, 2025-03-11, sha 7130a7cea9ce, PR #4327)
TITLE: refine sgl_moe_align_block_size_benchmark (#4327)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: sgl-kernel/benchmark/bench_moe_align_block_size.py (+77/-30)
BODY: Part of https://github.com/sgl-project/sglang/issues/2965 ⏎  ⏎ h200: ⏎  ⏎ ```shell ⏎ INFO 03-12 04:02:20 __init__.py:190] Automatically detected platform cuda. ⏎ ✅ SGL and Triton implementations match ⏎ ✅ SGL and VLLM implementations match ⏎ moe-align-block-size-performance: ⏎      num_tokens  num_experts  topk        SGL      Triton        VLLM ⏎ 0          16.0         32.0   2.0  16.736001   25.312001   16.832000 ⏎ 1          16.0         32.0   4.0  16. …[truncated]

### L1-e0917e6bd0  (L1, 2025-03-12, sha e0917e6bd0fb, PR #4215)
TITLE: Remove vllm ops scaled fp8 quant and accelerate per token quant by 20-28% (#4215)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+22/-8); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+14/-6); python/sglang/srt/custom_op.py (+59/-0); python/sglang/test/test_custom_ops.py (+88/-0); sgl-kernel/csrc/gemm/per_token_quant_fp8.cu (+19/-23)
LABELS: high priority
BODY: ## Motivation ⏎ Part of initiative: https://github.com/sgl-project/sglang/issues/2965 ⏎  ⏎ ### Goal:  ⏎ - Optimize the cuda kernel when applicable ⏎  ⏎ The optimized implementation achieved a time reduction of up to 28.72%. The improvement is most pronounced for larger inputs (e.g., 128 tokens, 4096 hidden_dim), suggesting that the optimization (likely vectorization or alignment fixes) effectively leverages parallelism and memory access efficiency at s …[truncated]

### L1-ed91561f79  (L1, 2025-03-12, sha ed91561f7972, PR #4334)
TITLE: upgrade sgl-kernel 0.0.4.post3 (#4334)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-817d43705c  (L1, 2025-03-12, sha 817d43705cd7, PR #4348)
TITLE: feat: support ep size < 32 for sgl kernel (#4348)
SOURCES: path_core
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+28/-6); sgl-kernel/benchmark/bench_moe_align_block_size.py (+54/-29)
BODY: ## Motivation ⏎ Support ep size < 32 for sgl kernel. ⏎ related pr https://github.com/sgl-project/sglang/pull/4249 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2f6bacee03  (L1, 2025-03-12, sha 2f6bacee0318, PR #3679)
TITLE: [moe] fix: correct the cache size in the last chunk (#3679)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+3/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This code confronts AssertionError: ⏎  ⏎ ```python ⏎ import torch ⏎ from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe ⏎  ⏎ N = 64 * 1024 + 10 ⏎ E = 8 ⏎ H = 1024 ⏎ I = 4096 ⏎  ⏎ x = torch.randn((N, H), device="cuda", dtype=torch.float16) ⏎ w1 = torch.randn((E, I * 2, H), device="cuda", dtype=torch.float16) ⏎ w2 = torch.randn((E, H, I), device="cuda", dtype=torch.float16) ⏎  ⏎ gating_output = torch.randn((N, E), device="c …[truncated]

### L1-3623b6a7f5  (L1, 2025-03-13, sha 3623b6a7f581, PR #4381)
TITLE: upgrade sgl-kernel 0.0.5 (#4381)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c6d7f8d370  (L1, 2025-03-13, sha c6d7f8d37080, PR #4398)
TITLE: Add some fused elementwise kernels for grok-1 (#4398)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.router_py
FILES: python/sglang/srt/layers/moe/router.py (+342/-0); python/sglang/srt/layers/elementwise.py (+411/-0); python/sglang/srt/models/grok.py (+374/-119)
BODY: Fuse small ops such as back-to-back RMSNorms and softcap ⏎  ⏎ Done by the co-authors: ⏎  ⏎ ``` ⏎  ⏎ ```

### L1-977d7cd26a  (L1, 2025-03-14, sha 977d7cd26abd, PR #4400)
TITLE: cleanup deps 1/n (#4400)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+23/-32); .github/workflows/pr-test-amd.yml (+2/-0); python/sglang/srt/layers/rotary_embedding.py (+2/-1)
BODY: ## Motivation ⏎  ⏎ close https://github.com/sgl-project/sglang/pull/4249/files ⏎  ⏎ ``` ⏎  ⏎ ``` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-61e4433caf  (L1, 2025-03-14, sha 61e4433cafd6, PR #4302)
TITLE: Add moe topk softmax templated from vllm (#4302)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_softmax
FILES: sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu (+505/-0); sgl-kernel/csrc/torch_extension.cc (+5/-0); sgl-kernel/include/sgl_kernel_ops.h (+6/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-1); sgl-kernel/python/sgl_kernel/moe.py (+11/-0); sgl-kernel/setup.py (+1/-0); sgl-kernel/benchmark/bench_moe_topk_softmax.py (+120/-0); sgl-kernel/include/utils.h (+14/-5); sgl-kernel/tests/test_moe_topk_softmax.py (+53/-0)
BODY: ## Motivation ⏎ #2965  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ 1) Cherry picked current vllm MoE topk softmax kernel template (with a fix on naming typo for `token_expert_indices`) ⏎ 2) Polish util func `warpReduceMax` / `blockReduceMax` for handle AMD use case as well. ⏎  ⏎  ⏎ ## Tests ⏎ Unit tests + benchmarking aligned with vllm counterpart ⏎  ⏎ ## Checklist

### L1-ad1ae7f7cd  (L1, 2025-03-14, sha ad1ae7f7cd0e, PR #4439)
TITLE: use topk_softmax with sgl-kernel (#4439)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L1.routing.topk_py, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/topk.py (+21/-8); .github/workflows/execute-notebook.yml (+1/-1); .github/workflows/experiment-runner.yml (+1/-1); .github/workflows/lint.yml (+1/-1); .github/workflows/nightly-test.yml (+1/-1); .github/workflows/pr-test-amd.yml (+2/-2); .github/workflows/pr-test-rust.yml (+2/-2); .github/workflows/pr-test-sgl-kernel.yml (+1/-1); .github/workflows/pr-test.yml (+9/-9); (+8 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8ec2ce0726  (L1, 2025-03-15, sha 8ec2ce072648, PR #4459)
TITLE: perf: update fused moe config (#4459)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎ ## Modifications ⏎  ⏎ ## Checklist

### L1-81f431eded  (L1, 2025-03-15, sha 81f431eded8a, PR #4449)
TITLE: feat: Add FlashMLA submodule (#4449)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+3/-0); sgl-kernel/setup.py (+49/-0); sgl-kernel/tests/test_flash_mla.py (+153/-0)
LABELS: high priority
DEEP_STUDY: deep-study: this PR was reverted by PR 4470 (confirmed_revert, reason=build_or_dependency)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9971dc2283  (L1, 2025-03-16, sha 9971dc2283ed, PR #4470)
TITLE: Revert "feat: Add FlashMLA submodule (#4449)" (#4470)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+0/-3); sgl-kernel/setup.py (+0/-49); sgl-kernel/tests/test_flash_mla.py (+0/-153)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 4449 reason=build_or_dependency
BODY: This reverts commit 81f431eded8a634b80f6c9fa4e9e0b016bd1fac1. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ - missing `git submodule add https://github.com/deepseek-ai/FlashMLA sgl-kernel/3rdparty/flashmla` ⏎ - It's AOT not JIT ⏎ - I have previously manually installed flash_mla locally, so I can pass through ut, but the original pr simply does not work ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1b859295f4  (L1, 2025-03-16, sha 1b859295f422, PR #4363)
TITLE: [Eagle] Remove the greedy branch and some redundant code (#4363)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/http_server.py (+1/-1); python/sglang/srt/managers/scheduler.py (+0/-2); python/sglang/srt/model_executor/cuda_graph_runner.py (+10/-11); python/sglang/srt/server_args.py (+0/-1); python/sglang/srt/speculative/build_eagle_tree.py (+7/-347); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+30/-5); python/sglang/srt/speculative/eagle_utils.py (+204/-250); python/sglang/srt/speculative/eagle_worker.py (+111/-46); python/sglang/srt/utils.py (+11/-0); (+4 more)
BODY: - Use faster kernel for temp=0 ⏎ - Support cuda graph padding ⏎ - Simplify redundant python code ⏎  ⏎ llama 2 7b: 390 token/s -> 400 token/s with this PR ⏎  ⏎ ``` ⏎  ⏎ ```

### L1-9b8333d992  (L1, 2025-03-16, sha 9b8333d99204, PR #4448)
TITLE: [ROCm] enable moe topk softmax in amd (#4448)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/torch_extension_rocm.cc (+4/-0); sgl-kernel/setup_rocm.py (+1/-0)
BODY: ## Motivation ⏎  ⏎  ⏎ Follow up of https://github.com/sgl-project/sglang/pull/4302 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - bench test ⏎ <img width="911" alt="截屏2025-03-15 13 24 28" src="https://github.com/user-attachments/assets/363c1b8b-16bc-45ac-8129-9a180d3e7a5d" /> ⏎  ⏎ ## Checklist

### L1-0f52fb55ec  (L1, 2025-03-16, sha 0f52fb55ecd9, PR #4493)
TITLE: config: Update fused moe config (#4493)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=1024,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+11/-11)
BODY: ## Motivation ⏎  ⏎ ## Modifications ⏎  ⏎ ## Checklist

### L1-75b656488a  (L1, 2025-03-17, sha 75b656488a64, PR #4418)
TITLE: Support serving DeepSeek-R1-Channel-INT8 with 32 L40S. (#4418)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=64,device_name=NVIDIA_L20,dtype=int8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=64,device_name=NVIDIA_L40S,dtype=int8_w8a8.json (+146/-0); benchmark/deepseek_v3/README.md (+27/-0); docs/references/deepseek.md (+2/-0); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+14/-5); sgl-kernel/csrc/gemm/int8_gemm_kernel.cu (+153/-5); sgl-kernel/tests/test_int8_gemm.py (+1/-1)
LABELS: high priority, quant
BODY: ## Motivation ⏎  ⏎ Add L40S support for serving channel quant model: [meituan/DeepSeek-R1-Channel-INT8](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8). ⏎  ⏎ ## Modifications ⏎  ⏎ * Modify extend attention BLOCK_M/N for sm_89; Fix test/srt/test_triton_attention_kernels.py test failure on L40S/L20. ⏎ * Add fused_moe_triton config for DSv3 & L40S. ⏎ * Modify int8 gemm kernel to support sm_89 gemm according to vllm implemetation. ⏎ * Add int8 gemm t …[truncated]

### L1-3ded4b215d  (L1, 2025-03-17, sha 3ded4b215df3, PR #4505)
TITLE: Revert "feat: update grouped_topk to support softmax and sigmoid" (#4505)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+3/-10)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 3680 reason=other
BODY: Reverts sgl-project/sglang#3680 ⏎  ⏎ Related to issue https://github.com/sgl-project/sglang/issues/4497. ⏎  ⏎ Currently `grouped_topk` is only used for DeepSeek-V2/V2.5, `biased_grouped_topk` is used for DeepSeek-V3/R1. For DeepSeek-V2, the scoring function is always `softmax`, so don't need to add new option.

### L1-f81a27f65e  (L1, 2025-03-17, sha f81a27f65ec1, PR #4522)
TITLE: upgrade sgl-kernel 0.0.5.post3 (#4522)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9b81f9bd34  (L1, 2025-03-17, sha 9b81f9bd349c, PR #4507)
TITLE: sglang quant module remove vllm dependency (#4507)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/__init__.py (+180/-123); python/sglang/srt/layers/quantization/blockwise_int8.py (+1/-1); python/sglang/srt/layers/quantization/fp8.py (+64/-27); python/sglang/srt/layers/quantization/fp8_utils.py (+95/-83); python/sglang/srt/layers/quantization/gptq.py (+24/-3); python/sglang/srt/layers/quantization/kv_cache.py (+98/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+9/-7); python/sglang/srt/layers/quantization/utils.py (+442/-0)
BODY: @zhyncs  ⏎  ⏎ End2end deepseek r1 test to ensure that the changes in the PR do not affect the normal functionality of the system. ⏎  ⏎ ```shell ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000 ⏎ python3 -m sglang.bench_serving --backend sglang --num-prompts 1000 --request-rate 8 ⏎ ``` ⏎  ⏎ ![图片](https://github.com/user-attachments/assets/c2171edb-5117-4cf2-a139-26dc0500e741) ⏎  ⏎ ![图片](https://github. …[truncated]

### L1-f44db16c8e  (L1, 2025-03-19, sha f44db16c8e0f, PR #4232)
TITLE: [Feature] Integrate DeepEP into SGLang (#4232)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: docker/Dockerfile.deepep (+77/-0); python/sglang/srt/layers/moe/ep_moe/kernels.py (+112/-11); python/sglang/srt/layers/moe/ep_moe/layer.py (+274/-0); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+533/-0); python/sglang/srt/model_executor/model_runner.py (+7/-0); python/sglang/srt/models/deepseek_v2.py (+143/-20); python/sglang/srt/server_args.py (+12/-0); docs/backend/server_arguments.md (+1/-0); python/sglang/srt/layers/linear.py (+13/-2); python/sglang/srt/layers/parameter.py (+1/-1); (+2 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Intergrate [DeepEP](https://github.com/deepseek-ai/DeepEP) into SGLang framework. Still WIP but could use '--enable-dp-attention **--enable-deepep-moe**' to trigger DeepEP intranode / internode, please follow the [install guide of NVSHMEM dependency](https://github.com/deepseek-ai/DeepEP/blob/main/third-party/README.md), also provide a Dockerfile.deepep based on SGLang image. ⏎  ⏎ Co-auther @xutizhou  ⏎  ⏎ Note: ⏎ - Currently need …[truncated]

### L1-38f25e87fc  (L1, 2025-03-22, sha 38f25e87fc9f, PR #4665)
TITLE: Correcting default configuration when benchmarking fused_moe (#4665)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+1/-0)
BODY: ## Motivation ⏎  ⏎ For benchmarking block-wise quantization, the block shape needs to be provided when obtaining the default configuration. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c2bd094d6e  (L1, 2025-03-22, sha c2bd094d6eb6, PR #4643)
TITLE: Optimize Permute Kernel in DeepEP (#4643)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+47/-49); python/sglang/srt/layers/moe/ep_moe/layer.py (+12/-15); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+39/-164); python/sglang/srt/models/deepseek_v2.py (+3/-2)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ The current performance of DeepEP is suboptimal due to the low efficiency of PyTorch's native permute function, which is used for formatting data before and after DeepEP communication. To address this limitation, we have implemented high-efficiency Triton kernels that significantly improve overall performance. ⏎  ⏎ ``` ⏎  ⏎ ``` ⏎ ### Performance on H20 ⏎ #### Single Node ⏎ Command ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path deep …[truncated]

### L1-c6d549e773  (L1, 2025-03-22, sha c6d549e7731e, PR #4608)
TITLE: Multiple tiny code cleanups (#4608)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+1/-2); python/sglang/srt/models/deepseek_v2.py (+2/-6)
BODY: ## Motivation ⏎  ⏎ When doing https://github.com/sgl-project/sglang/pull/4068, it seems these several lines are not needed, and thus remove them to make that PR neater ⏎  ⏎ EDIT: One more tiny cleanup suggested by @ch-wan in https://github.com/sgl-project/sglang/pull/4610#discussion_r2008986291 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-ca75741e86  (L1, 2025-03-22, sha ca75741e8605, PR #4610)
TITLE: Support async in DeepEP (#4610)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+24/-15); python/sglang/srt/models/deepseek_v2.py (+1/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ When doing https://github.com/sgl-project/sglang/pull/4068, DeepEP needs to be async. This PR enables that in a minimal way. ⏎  ⏎ (This is a separate PR because #4068 may not be done in a day, and I hope less merge conflicts happen, so extract this part first) ⏎  ⏎ Related: https://github.com/sgl-project/sglang/pull/4232 (Initial DeepEP support) ⏎  ⏎ When viewing diff, please subtract from change in https://github.com/sgl-project/sglan …[truncated]

### L1-64129fa632  (L1, 2025-03-24, sha 64129fa632c0, PR #4737)
TITLE: Add DeepEP tests into CI (#4737)
SOURCES: subject_keyword, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+6/-0)
DEEP_STUDY: deep-study: this PR was reverted by PR 4751 (confirmed_revert, reason=build_or_dependency)
BODY: ## Motivation ⏎  ⏎ test_moe_deepep.py seems not to be in CI, thus this super tiny PR adds it ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-199bb01d00  (L1, 2025-03-24, sha 199bb01d00af, PR #4435)
TITLE: Add endpoints to dump selected expert ids (#4435)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+6/-0); docs/backend/native_api.ipynb (+64/-0); python/sglang/srt/entrypoints/http_server.py (+30/-0); python/sglang/srt/managers/io_struct.py (+6/-0); python/sglang/srt/managers/scheduler.py (+15/-1); python/sglang/srt/managers/tokenizer_manager.py (+13/-0); python/sglang/srt/managers/utils.py (+78/-1); python/sglang/srt/models/deepseek_v2.py (+4/-0); test/srt/run_suite.py (+1/-0); test/srt/test_expert_distribution.py (+111/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ When optimizing the performance of MoE models, understanding the expert id distribution helps us to identify the performance bottlenecks and come up with a plan to fix performance issues.  Such information can be captured in `python/sglang/srt/layers/moe/topk.py`. ⏎  ⏎ ## Modifications ⏎  ⏎ - Created a singleton class in `python/sglang/srt/managers/utils.py` to record the layer id, expert id, and the topk id in a data structure. ⏎ - C …[truncated]

### L1-e45ae444db  (L1, 2025-03-25, sha e45ae444db7c, PR #4751)
TITLE: Revert "Add DeepEP tests into CI (#4737)" (#4751)
SOURCES: subject_keyword, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+0/-6)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 4737 reason=build_or_dependency
BODY: Need to have CI use Dockerfile.deepep infrastructure before enabling this ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-04e3ff6975  (L1, 2025-03-26, sha 04e3ff69753f, PR #4743)
TITLE: Support compressed tensors fp8w8a8 (#4743)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/sglang/srt/layers/moe/fused_moe_native.py (+2/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-2); python/sglang/srt/layers/moe/topk.py (+7/-6); .github/workflows/vllm-dependency-test.yml (+45/-0); python/pyproject.toml (+0/-1); python/sglang/srt/configs/model_config.py (+16/-3); python/sglang/srt/distributed/parallel_state.py (+4/-1); python/sglang/srt/layers/linear.py (+2/-3); python/sglang/srt/layers/quantization/__init__.py (+8/-7); python/sglang/srt/layers/quantization/base_config.py (+5/-0); (+20 more)
LABELS: high priority
BODY: 

### L1-8bf6d7f406  (L1, 2025-03-27, sha 8bf6d7f40614, PR #4706)
TITLE: support cmake for sgl-kernel (#4706)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: docker/Dockerfile.rocm (+2/-0); sgl-kernel/CMakeLists.txt (+166/-0); sgl-kernel/csrc/moe/moe_align_kernel.cu (+0/-1); .github/workflows/pr-test-amd.yml (+2/-2); .github/workflows/pr-test-sgl-kernel.yml (+2/-0); sgl-kernel/3rdparty/flashinfer (+1/-1); sgl-kernel/Makefile (+3/-2); sgl-kernel/build.sh (+2/-2); sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu (+1/-1); sgl-kernel/csrc/gemm/cublas_grouped_gemm.cu (+0/-1); (+8 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-7f19e083c1  (L1, 2025-03-27, sha 7f19e083c1d2, PR #4770)
TITLE: Support (1 <= dp < tp) in the dp attention in DeepEP (#4770)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/cuda_graph_runner.py (+7/-6); python/sglang/srt/model_executor/model_runner.py (+0/-3); python/sglang/srt/models/deepseek_v2.py (+146/-25); python/sglang/srt/server_args.py (+11/-6); docs/backend/server_arguments.md (+2/-2); python/sglang/srt/distributed/device_communicators/custom_all_reduce.py (+1/-1); python/sglang/srt/distributed/parallel_state.py (+22/-1); python/sglang/srt/layers/dp_attention.py (+12/-1); python/sglang/srt/managers/scheduler.py (+1/-1); test/srt/test_moe_deepep.py (+36/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ Previously, DeepEP only supported cases where dp = tp in the dp attention. This PR extends support to the case where 1 <= dp < tp.  ⏎ ## Modifications ⏎  ⏎  ⏎ We apply reduce-scatter operation to distribute tokens before the MOE layer, and an all-gather afterward to restore the tensor. ⏎ ## Checklist

### L1-31dfff7da7  (L1, 2025-03-27, sha 31dfff7da7ad, PR #4835)
TITLE: use default for torch.ops (#4835)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: sgl-kernel/python/sgl_kernel/moe.py (+2/-2); sgl-kernel/python/sgl_kernel/allreduce.py (+15/-15); sgl-kernel/python/sgl_kernel/attention.py (+1/-1); sgl-kernel/python/sgl_kernel/elementwise.py (+10/-8); sgl-kernel/python/sgl_kernel/gemm.py (+14/-12); sgl-kernel/python/sgl_kernel/sampling.py (+5/-5); sgl-kernel/python/sgl_kernel/speculative.py (+4/-4)
BODY: ## Motivation ⏎  ⏎ suggested by @yinfan98  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-74e0ac1dbd  (L1, 2025-03-28, sha 74e0ac1dbd19, PR #4834)
TITLE: Clean up `import vllm` in quantization/__init__.py (#4834)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/sglang/srt/layers/moe/topk.py (+1/-1); .github/workflows/pr-test.yml (+4/-8); .github/workflows/vllm-dependency-test.yml (+4/-8); python/pyproject.toml (+7/-1); python/sglang/srt/configs/model_config.py (+3/-16); python/sglang/srt/layers/quantization/__init__.py (+132/-163); python/sglang/srt/layers/quantization/awq.py (+1/-1); python/sglang/srt/layers/quantization/fp8_kernel.py (+2/-1); python/sglang/srt/layers/quantization/gptq.py (+30/-40); python/sglang/srt/managers/tp_worker.py (+3/-2); (+4 more)
LABELS: high priority
BODY: - Raise meaningful error messages when the vllm dependency is needed. ⏎ - Move all imports into a single place, to minimize the use of try-catch

### L1-4db29e82ec  (L1, 2025-03-28, sha 4db29e82ec10, PR #4864)
TITLE: [Feat] support deepgemm for cmake (#4864)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+16/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-044c315970  (L1, 2025-03-28, sha 044c3159706c, PR #4749)
TITLE: Make torch compile configurable for biased_grouped_topk (#4749)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+29/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Make torch compile configurable for biased_grouped_topk in order to test the kernel #4530 in eager mode without duplicating the functions. (torch compile has slight perf diff with eager mode and the cuda kernel added in #4530 matches the eager mode results) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2bb0e7cf43  (L1, 2025-03-28, sha 2bb0e7cf43a3, PR #4871)
TITLE: fix sampling issue (#4871)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d8a136a113  (L1, 2025-03-28, sha d8a136a11332, PR #4873)
TITLE: upgrade sgl-kernel 0.0.5.post4 (#4873)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-45dcfc2e76  (L1, 2025-03-29, sha 45dcfc2e762d, PR #4530)
TITLE: Add deepseek style fused moe group gate selection kernel (#4530)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/csrc/moe/moe_fused_gate.cu (+447/-0); sgl-kernel/csrc/torch_extension.cc (+5/-0); sgl-kernel/include/sgl_kernel_ops.h (+3/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-1); sgl-kernel/python/sgl_kernel/moe.py (+12/-0); sgl-kernel/setup.py (+1/-0); sgl-kernel/benchmark/bench_moe_fused_gate.py (+74/-0); sgl-kernel/tests/test_moe_fused_gate.py (+72/-0)
LABELS: high priority
BODY: ## Motivation ⏎ PR adapted and improved from #3191  ⏎ Rewrite Macro. Extended to support all power of 2 `# expert` & `# expert group`, also all `# topk_group` & `# topk` use cases + dtype support `fp16/bf16/fp32`. ⏎  ⏎ Pick up the closed pr #4445  ⏎  ⏎ Remove deepgemm test as not needed. ⏎  ⏎ NOTE: ⏎ - Current CUDA kernel matches the eager mode results for supported cases, but torch compile with static or dynamic will cause original torch implementation r …[truncated]

### L1-8e7b31546c  (L1, 2025-03-29, sha 8e7b31546c72, PR #4898)
TITLE: quick fix: add default for new kernel (#4898)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: sgl-kernel/python/sgl_kernel/moe.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0d7fe866f9  (L1, 2025-03-29, sha 0d7fe866f97d, PR #4890)
TITLE: [Misc] Clean m.def and add Development Tips (#4890)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: sgl-kernel/README.md (+41/-0); sgl-kernel/csrc/torch_extension.cc (+37/-151); sgl-kernel/python/sgl_kernel/elementwise.py (+8/-8)
DEEP_STUDY: deep-study: this PR was reverted by PR 4944 (partial_revert, reason=crash_or_hang)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Update TORCH_LIBRARY define ⏎  ⏎ From: ⏎  ⏎ ```cpp ⏎ m.def( ⏎       "init_custom_ar(int rank_id, int world_size, Tensor rank_data, int[] buffers, int[] tmp_result_buffers, int[] " ⏎       "barrier_in, int[] barrier_out) -> int"); ⏎   m.impl("init_custom_ar", torch::kCUDA, &init_custom_ar); ⏎ ``` ⏎  ⏎ To: ⏎  ⏎ ```cpp ⏎ m.def("init_custom_ar", init_custom_ar); ⏎ ``` ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4ede6770cd  (L1, 2025-03-30, sha 4ede6770cd94, PR #4914)
TITLE: Fix retract for page size > 1 (#4914)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+2/-44); python/sglang/srt/constrained/base_grammar_backend.py (+5/-1); python/sglang/srt/managers/schedule_batch.py (+6/-3); python/sglang/srt/metrics/collector.py (+23/-53); python/sglang/srt/server_args.py (+12/-8); python/sglang/test/test_utils.py (+0/-4); test/srt/models/lora/test_lora_tp.py (+3/-3); test/srt/run_suite.py (+13/-3); test/srt/test_dp_attention.py (+4/-0); test/srt/test_metrics.py (+0/-1)
BODY: - Fix retract for page size > 1 ⏎ ``` ⏎                 last_uncached_pos = len(req.prefix_indices) ⏎                 last_uncached_pos = ( ⏎                     (len(req.prefix_indices) + server_args.page_size - 1) ⏎                     // server_args.page_size ⏎                     * server_args.page_size ⏎                 ) ⏎ ``` ⏎ - Improve server_args ⏎ - Improve test cases

### L1-37c66ec856  (L1, 2025-03-30, sha 37c66ec8563d, PR #4902)
TITLE: [feat] add fa3 in sgl-kernel (#4902)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+54/-0); sgl-kernel/README.md (+30/-0); sgl-kernel/csrc/torch_extension.cc (+5/-0); sgl-kernel/include/sgl_kernel_ops.h (+47/-0); sgl-kernel/include/sgl_kernel_torch_shim.h (+122/-0); sgl-kernel/python/sgl_kernel/flash_attn.py (+201/-0); sgl-kernel/tests/test_flash_attention.py (+841/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ add fa3 in sgl-kernel ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4814ecaff9  (L1, 2025-03-30, sha 4814ecaff988, PR #4933)
TITLE: cleanup sgl-kernel (#4933)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+2/-10); sgl-kernel/3rdparty/cccl (+0/-1); sgl-kernel/3rdparty/cutlass (+0/-1); sgl-kernel/3rdparty/deepgemm (+0/-1); sgl-kernel/3rdparty/flashinfer (+1/-1); sgl-kernel/README.md (+1/-1); sgl-kernel/THIRDPARTYNOTICES.txt (+33/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c7457191a0  (L1, 2025-03-31, sha c7457191a0ef, PR #4944)
TITLE: [Fix] revert clean m.def for cudagraph (#4944)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: sgl-kernel/README.md (+13/-3); sgl-kernel/csrc/torch_extension.cc (+182/-37)
DEEP_STUDY: deep-study revert record: partial_revert of PR(s) 4890 reason=crash_or_hang
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Revert and update m.def PR https://github.com/sgl-project/sglang/pull/4890, it will cause cudagraph/torch.compile fail. ⏎  ⏎ Cauz m.def with schema will help torch inductor to find out custom ops. ⏎  ⏎ Update sgl-kernel docs. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1c63e79756  (L1, 2025-03-31, sha 1c63e7975604, PR #4954)
TITLE: use fa3 in sgl-kernel (#4954)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-23c764b18a  (L1, 2025-04-01, sha 23c764b18aeb, PR #4767)
TITLE: [Feature] Support DeepEP Low Latency (#4767)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+142/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+81/-78); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+137/-124); python/sglang/srt/model_executor/model_runner.py (+2/-1); python/sglang/srt/models/deepseek_v2.py (+56/-25); python/sglang/srt/server_args.py (+18/-0); docs/backend/server_arguments.md (+1/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ Support DeepEP low latency dispatch / combine, introduce a new command-line argument `--deepep-mode` to specify DeepEP mode (`auto`, `normal` and `low_latency`). Additionally, we believe DeepEP is particularly well-suited for PD disaggregation. Also in low-latency mode, the CUDA Graph feature functions seamlessly. ⏎  ⏎ **deepep mode option** ⏎  ⏎ ```sh ⏎ # auto (default mode): use normal dispatch / combine for non decode and low_lat …[truncated]

### L1-e41549c3d6  (L1, 2025-04-03, sha e41549c3d6d8, PR #4727)
TITLE: fix: fix illegal cuda memory access at fused_moe_kernel (#4727)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-0)
ISSUES: #4572 [BUG] deepseek using dp encounter CUDA illegal memory access on 2nodes(each with 8xH20)
BODY: ## Motivation ⏎ When the Deepseek's prompt length is very long (>40k) and the --chunked-prefill-size is large (>64k), fused_moe_kernel will encounter a cuda illegal memory access error. This pull request aims to fix this error. ⏎  ⏎ In python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py: ⏎ `def fused_moe_kernel`, we have: ⏎ ``` ⏎ offs_token = tl.load(sorted_token_ids_ptr + offs_token_id) ⏎ ... ⏎ c_ptrs = c_ptr + stride_cm * offs_token[:, None] + s …[truncated]

### L1-8e10fec9a8  (L1, 2025-04-03, sha 8e10fec9a8c8, PR #4992)
TITLE: Small refactor DeepEPMode to clean up code a bit (#4992)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+6/-10); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+11/-15); python/sglang/srt/models/deepseek_v2.py (+3/-3); python/sglang/srt/server_args.py (+2/-2); python/sglang/srt/utils.py (+22/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b8b6008f47  (L1, 2025-04-03, sha b8b6008f47de, PR #5036)
TITLE: [Fix] fix fa3 build at cu118 (#5036)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+85/-50); sgl-kernel/cmake/utils.cmake (+21/-0); sgl-kernel/csrc/common_extension.cc (+1/-40); sgl-kernel/csrc/flash_extension.cc (+62/-0); sgl-kernel/include/sgl_flash_kernel_ops.h (+85/-0); sgl-kernel/include/sgl_kernel_ops.h (+0/-47); sgl-kernel/python/sgl_kernel/flash_attn.py (+15/-4); sgl-kernel/tests/test_flash_attention.py (+19/-1)
ISSUES: #4941 [Feature] use different lib so for fa3 in sgl-kernel
DEEP_STUDY: deep-study correctness case sglang:b8b6008f47: class=hardware_compiler_specific; symptom=compile_or_build_failure; introducing=unknown
BODY: ## Motivation ⏎  ⏎  ⏎ Fix cu118 for fa3 compile error ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e53bf190bc  (L1, 2025-04-03, sha e53bf190bce3, PR #5049)
TITLE: upgrade sgl-kernel v0.0.7 (#5049)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d95269f9b3  (L1, 2025-04-03, sha d95269f9b3a2, PR #4625)
TITLE: [2/3] fix dsv3 awq issue  (#4625)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+392/-39); benchmark/deepseek_v3/README.md (+3/-2); python/sglang/srt/configs/model_config.py (+1/-0); python/sglang/srt/layers/quantization/__init__.py (+2/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+501/-0); python/sglang/srt/layers/quantization/utils.py (+1/-1); python/sglang/srt/server_args.py (+1/-0); test/srt/test_triton_moe_wna16.py (+238/-0)
LABELS: high priority, quant
BODY: ## Motivation ⏎  ⏎ related issue: https://github.com/sgl-project/sglang/issues/4462 ⏎ The dsv3 awq issue requires a few PRs to fix, this is a part of it. ⏎ Currently dsv3 awq suffers speed problem because lacking of efficient moe w4a16 kernel ⏎  ⏎ Author: @laixinn @huangtingwei9988 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ Add moe wna16 kernel to sglang ⏎  ⏎  ⏎ ## Checklist

### L1-31035dda44  (L1, 2025-04-03, sha 31035dda44b8, PR #5057)
TITLE: Add H20 fused MoE kernel tuning configs for DeepSeek V3/R1 (#5057)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=128,device_name=NVIDIA_H20,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎ Add H20 fused MoE kernel tuning configs for DeepSeek V3/R1 ⏎  ⏎ ## Modifications ⏎  ⏎ Add `E=256,N=256,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128, 128].json` ⏎  ⏎ ## Checklist

### L1-febe21ce03  (L1, 2025-04-04, sha febe21ce031d, PR #4994)
TITLE: Small refactor DeepEPDispatcher into subclasses (#4994)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+291/-162); python/sglang/srt/models/deepseek_v2.py (+18/-29)
BODY: ## Motivation ⏎  ⏎ Please subtract diff from https://github.com/sgl-project/sglang/pull/4992 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-77e929a1a2  (L1, 2025-04-04, sha 77e929a1a2f3, PR #4995)
TITLE: Support async DeepEP by splitting into two stages (#4995)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+86/-32)
BODY: ## Motivation ⏎  ⏎ Please subtract code diff from https://github.com/sgl-project/sglang/pull/4994 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6ff9c6a5e7  (L1, 2025-04-04, sha 6ff9c6a5e71f, PR #4996)
TITLE: Cleanup unused resources after DeepEP operation (#4996)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+3/-12)
BODY: ## Motivation ⏎  ⏎ Please subtract diff from https://github.com/sgl-project/sglang/pull/4995 ⏎  ⏎ Benchmark ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --trust-remote-code --tp 16 --dp 16 --host 0.0.0.0 --port 32000 --enable-deepep-moe --deepep-mode low_latency --max-running-requests 16 --chunked-prefill-size 2048 --max-prefill-tokens 128 --disable-radix-cache --stream-output --cuda-graph-max-bs 128 --dist-init-addr 1 …[truncated]

### L1-924ca7c92c  (L1, 2025-04-04, sha 924ca7c92c86, PR #4918)
TITLE: Add DeepSeek V3/R1 shared experts fusion (#4918)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+10/-8); python/sglang/srt/layers/moe/topk.py (+49/-3); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+12/-1); python/sglang/bench_serving.py (+31/-6); python/sglang/srt/layers/quantization/__init__.py (+2/-1); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+2/-1); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+34/-10); python/sglang/srt/managers/schedule_batch.py (+2/-0); (+4 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ I gain idea mainly from https://github.com/vllm-project/vllm/pull/15502 , thanks for the author's work.I will add references in the modifications for `grouped_topk` function and DeepSeek v2 model `weight_loader`. And sgl-kernel's `moe_align_kernel` kernel has already supported `num_experts > 256` in [pr](https://github.com/sgl-project/sglang/pull/4348), so it's easy to implement fuse shared experts into 256 experts now. ⏎  ⏎ ## Con …[truncated]

### L1-4c54f44202  (L1, 2025-04-04, sha 4c54f4420217, PR #5072)
TITLE: [deepep] fix: shared experts are not initialized when shared experts fusion is enabled (#5072)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+2/-2)
BODY: PR #4918 sets the default value of `n_share_experts_fusion` as None, which is not compatible with [this line](https://github.com/sgl-project/sglang/blob/924ca7c92c86fa3a6a321e7944e2fdd193f30c50/python/sglang/srt/models/deepseek_v2.py#L221). ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-aba5ca154d  (L1, 2025-04-05, sha aba5ca154d4c, PR #5080)
TITLE: python transfer custom allreduce from trt kernel to vllm kernel (#5080)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/_custom_ops.py (+59/-92); python/sglang/srt/distributed/device_communicators/custom_all_reduce.py (+25/-77); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎ ref https://github.com/sgl-project/sglang/pull/5079 ⏎  ⏎ ```python ⏎ lm_eval --model sglang --model_args pretrained=meta-llama/Llama-3.1-8B,tp_size=x,dtype=auto --tasks gsm8k --batch_size 16 ⏎ lm_eval --model sglang --model_args pretrained=meta-llama/Llama-3.1-70B,tp_size=x,dtype=auto --tasks gsm8k --batch_size 16 ⏎ ``` ⏎ Llama3.1-8B ⏎  ⏎ tp = 2 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|---- …[truncated]

### L1-f04c80dc42  (L1, 2025-04-07, sha f04c80dc42be, PR #5092)
TITLE: Add Llama4 support (#5092)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_native.py (+5/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=512,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=144,N=512,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=1024,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=1024,device_name=NVIDIA_H200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=2048,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=20,N=2048,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=24,N=1024,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+13/-3); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-0); (+17 more)
LABELS: high priority
BODY: ## Motivation ⏎ - Add LM support for meta-llama/Llama4 model family. ⏎  ⏎  ⏎ ## Modifications ⏎ - Add conversation template support for Llama4 models @ispobock  ⏎ - Implement Llama4 language model @ch-wan  ⏎ - Implement local attention supported by FA3 @CatherineSue  ⏎ - Support tuned MoE configs @fzyzcjy  ⏎ - Benchmark @ch-wan @ispobock  ⏎  ⏎  ⏎ ## Benchmark Results ⏎  ⏎ ### Llama-4-Scout-17B-16E-Instruct ⏎  ⏎ - Command ⏎  ⏎ ``` ⏎ python -m sglang.launch_server -- …[truncated]

### L1-afb752bcbe  (L1, 2025-04-07, sha afb752bcbeb1, PR #5140)
TITLE: [AMD] Fix missing per_token_group_quant_fp8 for ROCm (#5140)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-0)
LABELS: bug, high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎  ⏎ Fix the issue https://github.com/sgl-project/sglang/issues/5138 introduced by https://github.com/sgl-project/sglang/pull/3702 for  AMD GPUs. ⏎  ⏎ Once https://github.com/sgl-project/sglang/pull/3702 is merged, AMD can use sglang.srt.layers.quantization.fp8_kernel.sglang_per_token_group_quant_fp8 like NVIDIA GPUs. ⏎  ⏎ CC: @HaiShaw

### L1-5a144a8ab9  (L1, 2025-04-07, sha 5a144a8ab985, PR #5147)
TITLE: Fix run time error in ROCm platform (#5147)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.router_py
FILES: python/sglang/srt/layers/moe/router.py (+7/-1); python/sglang/srt/layers/elementwise.py (+15/-2); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-0)
BODY: Obsolete: #5124  ⏎  ⏎ ## Motivation ⏎  ⏎ ``` ⏎  ⏎ ``` ⏎  ⏎ When running the latest docker image "lmsysorg/sglang:v0.4.4.post4-rocm630", there are some errors happened. ⏎ **Error 1** ⏎ ``` ⏎ File "/sgl-workspace/sglang/python/sglang/srt/models/grok.py", line 320, in forward ⏎     fused_rmsnorm( ⏎   File "/sgl-workspace/sglang/python/sglang/srt/layers/elementwise.py", line 260, in fused_rmsnorm ⏎     fused_rmsnorm_kernel[(bs,)]( ⏎   File "/usr/local/lib/python3.12 …[truncated]

### L1-a73c4df438  (L1, 2025-04-08, sha a73c4df4387a, PR #5150)
TITLE: Add optimized native kernels in sgl-kernel (#5150)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: sgl-kernel/csrc/cpu/moe.cpp (+1247/-0); sgl-kernel/csrc/cpu/moe_int8.cpp (+830/-0); sgl-kernel/csrc/cpu/topk.cpp (+406/-0); sgl-kernel/csrc/cpu/activation.cpp (+79/-0); sgl-kernel/csrc/cpu/bmm.cpp (+122/-0); sgl-kernel/csrc/cpu/common.h (+164/-0); sgl-kernel/csrc/cpu/decode.cpp (+1119/-0); sgl-kernel/csrc/cpu/extend.cpp (+621/-0); sgl-kernel/csrc/cpu/gemm.cpp (+507/-0); sgl-kernel/csrc/cpu/gemm.h (+130/-0); (+10 more)
LABELS: high priority, sgl-kernel, intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This pull request is a follow up on https://github.com/sgl-project/sglang/issues/2807 to enable and optimize sglang performance on CPU devices. In this patch, optimized C++ kernels are provided including: ⏎ * activations ⏎ * layernorms ⏎ * gemm (bfloat16, int8) ⏎ * extend attention (bfloat16) ⏎ * decode attention (bfloat16) ⏎ * allreduce and allgather ⏎ * moe (bfloat16, int8) ⏎ * rope ⏎  ⏎ Specifically, we are are targeting at optimizi …[truncated]

### L1-bc3f6db2dd  (L1, 2025-04-08, sha bc3f6db2dd6a, PR #5068)
TITLE: [Fix] DeepEP Compatibility with Low Latency (#5068)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+145/-118); python/sglang/srt/model_executor/forward_batch_info.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+1/-1); python/sglang/srt/server_args.py (+1/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ As title, make DeepEP normal buffer compatible with low_latency buffer, simply tested both intra-node and inter-node can run successfully. ⏎  ⏎ Support matrix ⏎ | Type | DeepEP Auto | DeepEP Normal (disable CUDA Graph)| DeepEP Low Latency | ⏎ |------------------|------------------|------------------|------------------| ⏎ | Intra-node      | ✓         | ✓           | ✓                | ⏎ | Inter-node      | ✓         | ✓           | …[truncated]

### L1-90caf06c00  (L1, 2025-04-08, sha 90caf06c0064, PR #5180)
TITLE: fix: use DeepEPDispatcher on CUDA (#5180)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+2/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-76c48a0913  (L1, 2025-04-08, sha 76c48a0913b9, PR #5179)
TITLE: [DeepEP] fix: import buffer error (#5179)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-87eddedfa2  (L1, 2025-04-09, sha 87eddedfa2d9, PR #5102)
TITLE: [ci] fix ci test fused_moe op (#5102)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: test/srt/run_suite.py (+1/-0); test/srt/test_fused_moe.py (+42/-22)
BODY: ## Motivation ⏎  ⏎ I found that the errors related to "failed to run tuning_fused_moe_triton.py" in https://github.com/sgl-project/sglang/issues/4991 are due to a particular commit in the CI that mistakenly deleted test_fused_moe.py. This deletion resulted in a lack of tests to check for potential circular import issues caused by the line `from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe`. The test has been added back. Additio …[truncated]

### L1-456b008bd8  (L1, 2025-04-09, sha 456b008bd8f1, PR #5196)
TITLE: Add H20 dtype fp8_w8a8 fused MoE kernel tuning configs for DeepSeek V3/R1 (#5196)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: # Add H20 fp8_w8a8 Fused MoE Kernel Configurations for DeepSeek V3/R1 ⏎  ⏎ ## Motivation ⏎ - Currently missing optimized kernel configurations for fused MoE layers when using `fp8_w8a8` datatype on NVIDIA H20 GPUs ⏎  ⏎ ## Changes ⏎ 1. Added new fused MoE kernel tuning configurations for H20 GPUs with `fp8_w8a8` precision ⏎ 2. Parameters were derived from systematic benchmarking using `main/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` ⏎  …[truncated]

### L1-e3c4bd3153  (L1, 2025-04-09, sha e3c4bd315302, PR #5190)
TITLE: Fix DeepSeek error when using DeepEP mode (#5190)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+8/-6)
BODY: command ⏎  ⏎ ``` ⏎ python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3-0324 --trust-remote-code --tp 16 --dp 16 --enable-dp-attention --enable-deepep-moe --deepep-mode normal --disable-cuda-graph --enable-flashmla --disable-radix-cache --decode-log-interval 1 --host 0.0.0.0 --port 20000 --nnodes 2 --dist-init-addr 10.10.38.10:23456 --chunked-prefill-size 16384 --node-rank 0 ⏎ # and rank 1 ⏎  ⏎ (cd /host_home/primary_synced/sglang && pyt …[truncated]

### L1-f730362ee2  (L1, 2025-04-09, sha f730362ee207, PR #5086)
TITLE: reduce moe_align_block_size_kernel small batch mode overhead (#5086)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.align.cuda_aot
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-1); sgl-kernel/csrc/moe/moe_align_kernel.cu (+111/-44); sgl-kernel/benchmark/bench_moe_align_block_size.py (+31/-10); sgl-kernel/tests/test_moe_align.py (+0/-1)
BODY: ## Motivation ⏎  ⏎  ⏎ ## Acc test ⏎  ⏎ I set `token_cnts_buffer` and `cumsum_buffer` to `torch.empty` in `fused_moe.py`: ⏎  ⏎ ```python ⏎ token_cnts_buffer = torch.empty( ⏎             (num_experts + 1) * num_experts, ⏎             dtype=torch.int32, ⏎             device=topk_ids.device, ⏎         ) ⏎         cumsum_buffer = torch.empty( ⏎             num_experts + 1, dtype=torch.int32, device=topk_ids.device ⏎         ) ⏎ ``` ⏎  ⏎ Acc result: ⏎  ⏎ ```shell ⏎ ➜  sglan …[truncated]

### L1-4065248214  (L1, 2025-04-09, sha 406524821457, PR #5194)
TITLE: Support Llama4 fp8 inference (#5194)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+33/-18); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+4/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+66/-45); python/sglang/srt/layers/quantization/fp8_utils.py (+9/-0); python/sglang/srt/layers/quantization/w8a8_fp8.py (+154/-4); python/sglang/srt/layers/quantization/w8a8_int8.py (+1/-0); python/sglang/srt/model_loader/loader.py (+10/-3); python/sglang/srt/model_loader/weight_utils.py (+4/-1); python/sglang/srt/models/deepseek_v2.py (+24/-16); python/sglang/srt/models/llama4.py (+1/-1); (+4 more)
LABELS: high priority, quant
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Support llama4 fp8 inference for [meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 ⏎ ](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8) @zhyncs @ispobock @zhaochenyang20  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Benchmark  ⏎  ⏎ ``` ⏎ # launch ⏎ python -m sglang.launch_server --model-path meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --tp 8 ⏎  ⏎ # gsm8k and mmlu ⏎ python benchmark/gsm8k/bench_sglang.py --num-quest …[truncated]

### L1-60bcbf2a35  (L1, 2025-04-11, sha 60bcbf2a35e2, PR #5298)
TITLE: remove moe_align_block_size torch.zeros in small batch/expert mode (#5298)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ Follow [pr 5086](https://github.com/sgl-project/sglang/pull/5086) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f774a0d275  (L1, 2025-04-11, sha f774a0d27557, PR #5302)
TITLE: feat: add blackwell Dockerfile (#5302)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile.blackwell (+19/-0); python/pyproject.toml (+11/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6f8593799b  (L1, 2025-04-11, sha 6f8593799bd1, PR #5303)
TITLE: feat: add blackwell workflow (#5303)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+4/-3); .github/workflows/release-docker-blackwell.yml (+36/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f65b8d5c89  (L1, 2025-04-11, sha f65b8d5c896c, PR #5142)
TITLE: Blackwell Cutlass MLA kernel (#5142)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+4/-1); sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+207/-0); sgl-kernel/csrc/common_extension.cc (+5/-0); sgl-kernel/include/sgl_kernel_ops.h (+8/-1); sgl-kernel/python/sgl_kernel/__init__.py (+5/-1); sgl-kernel/python/sgl_kernel/attention.py (+61/-0); sgl-kernel/tests/test_cutlass_mla.py (+81/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Adds blackwell cutlass MLA kernel to sgl-kernel. Requires cutlass 3.9 ⏎ Thanks to @kaixih for kernel and test code   ⏎  ⏎ I will open a second PR soon to add the cutlass mla attention backend. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2eb55770f9  (L1, 2025-04-11, sha 2eb55770f99c, PR #5311)
TITLE: misc: cleanup 3rdparty (#5311)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+0/-4); sgl-kernel/3rdparty/flashinfer (+0/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-115ae2e728  (L1, 2025-04-11, sha 115ae2e728bc, PR #5317)
TITLE: chore: bump sgl-kernel v0.0.8.post2 (#5317)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-690ec20587  (L1, 2025-04-12, sha 690ec2058795, PR #5321)
TITLE: Delete python/sglang/srt/layers/moe/fused_moe_triton/configs/E=257,N=… (#5321)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+0/-146)
BODY: ## Motivation ⏎  ⏎ This PR removes the obsolete config used for experimentation in shared_experts_fusion. cc @zhyncs  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3e4794aad8  (L1, 2025-04-12, sha 3e4794aad8a2, PR #5294)
TITLE: refine fused_moe tuning docs (#5294)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-1); benchmark/kernels/fused_moe_triton/README.md (+10/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4879e50c6d  (L1, 2025-04-12, sha 4879e50c6d46, PR #5327)
TITLE: [Feat] Add sparse attn to sgl-kernel (#5327)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+30/-14); sgl-kernel/csrc/common_extension.cc (+22/-0); sgl-kernel/include/sgl_kernel_ops.h (+50/-0); sgl-kernel/python/sgl_kernel/sparse_flash_attn.py (+175/-0); sgl-kernel/tests/test_sparse_flash_attn.py (+348/-0)
BODY: ## Motivation ⏎  ⏎  ⏎ Adapt from: https://github.com/sgl-project/sgl-attn/pull/1 ⏎ Add sparse attn kernel to sgl-kernel.  ⏎ Co-author: @minedec @minminsun ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b371f7cd36  (L1, 2025-04-12, sha b371f7cd3626, PR #5332)
TITLE: chore: bump sgl-kernel v0.0.8.post3 (#5332)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c138025731  (L1, 2025-04-12, sha c13802573162, PR #5341)
TITLE: misc: update sagemaker Dockerfile (#5341)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.sagemaker (+1/-73)
BODY: ## Motivation ⏎  ⏎ @andjsmi ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f58b929a51  (L1, 2025-04-13, sha f58b929a5185, PR #5342)
TITLE: chore: upgrade sgl-kernel 0.0.8.post3 (#5342)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/quantization/fp8_kernel.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎ unblock https://github.com/sgl-project/sglang/pull/5113 @Fridge003  ⏎  ⏎ ``` ⏎ flash_attn_with_varlen_func ⏎ ``` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-adca585bfb  (L1, 2025-04-13, sha adca585bfb59, PR #5277)
TITLE: [DeepEP] Reduce routed scaling overhead (#5277)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+9/-10)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ In the `--deepep-mode=low_latency mode`, `final_hidden_states` is a tensor with the shape `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]`, and most of its values are masked. Applying `routed_scaling_factor` to the entire tensor would result in substantial memory access overhead. The introduction of `masked_scale` avoids scaling the masked portions, thereby reducing memory access and lowering latency. ⏎  …[truncated]

### L1-38076dea84  (L1, 2025-04-14, sha 38076dea8425, PR #5371)
TITLE: apply fused moe gate in ds v3/r1 (#5371)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+37/-16)
BODY: ## torch profile ⏎  ⏎ ```shell ⏎ python3 -m sglang.bench_serving --backend sglang --num-prompts 2 --request-rate 1 --port 30001 --flush-cache --warmup-requests 1 --profile ⏎ ``` ⏎  ⏎ ### main branch ⏎  ⏎ ![图片](https://github.com/user-attachments/assets/6e5d87ca-1090-47da-b05e-1e1124583048) ⏎  ⏎ ### pr ⏎  ⏎ ![图片](https://github.com/user-attachments/assets/e09ede25-c6c6-4481-8428-30d053c96e4e) ⏎  ⏎ Only one kernel now. ⏎  ⏎ 36us->8us. ⏎  ⏎  ⏎ ## fused_moe_gate gsm8k  …[truncated]

### L1-61e7c4dd21  (L1, 2025-04-14, sha 61e7c4dd21f5, PR #5368)
TITLE: Add A800 shared experts fused MoE kernel tuning configs for DeepSeek V3/R1 (#5368)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=128,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=64,device_name=NVIDIA_A800-SXM4-80GB.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=288,N=64,device_name=NVIDIA_A800-SXM4-80GB.json (+146/-0); benchmark/kernels/fused_moe_triton/README.md (+8/-0)
BODY: ## Motivation ⏎ Ref: https://github.com/sgl-project/sglang/pull/4918 ⏎  ⏎ ## Modifications ⏎ - Modify `benchmark/kernels/fused_moe_triton/README.md` ⏎ - Added tuning config files corresponding to bf16 and channel-wise int8 ⏎   - bf16 TP32 on A800, with 32 or 16 shared experts ⏎   - channel-wise int8 TP16 on A800, with 16 or 8 shared experts ⏎  ⏎ ## Checklist

### L1-2dd6489468  (L1, 2025-04-14, sha 2dd648946833, PR #5291)
TITLE: Add H20 dtype fp8_w8a8 shared experts fused MoE kernel tuning configs for DeepSeek V3/R1 (#5291)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎ https://github.com/sgl-project/sglang/pull/4918 ⏎ Currently missing optimized kernel configurations for shared experts fused MoE kernel tuning configs when using fp8_w8a8 datatype on NVIDIA H20 GPUs ⏎  ⏎  ⏎ ## Modifications ⏎ 1. Added new fused MoE kernel tuning configurations for H20 GPUs with fp8_w8a8 precision ⏎ 2. Parameters were derived from systematic benchmarking using `main/benchmark/kernels/fused_moe_triton/tuning_fused_moe_trit …[truncated]

### L1-ee9d6ca677  (L1, 2025-04-14, sha ee9d6ca67723, PR #5279)
TITLE: [fix/misc] remove duplicate row in deepseek v2 model (#5279)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+0/-1)
BODY: remove duplicate row `self.routed_scaling_factor = config.routed_scaling_factor`, line 175 and line 183

### L1-e940dc4f06  (L1, 2025-04-14, sha e940dc4f06a3, PR #5400)
TITLE: chore: bump sgl-kernel 0.0.9 (#5400)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/Makefile (+2/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8aab7fdb21  (L1, 2025-04-14, sha 8aab7fdb21e8, PR #5401)
TITLE: chore: upgrade sgl-kernel 0.0.9 (#5401)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-838fa0f218  (L1, 2025-04-15, sha 838fa0f21855, PR #5420)
TITLE: [minor] cleanup cmakelists.txt (#5420)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+17/-11); .github/workflows/pr-test.yml (+0/-2); sgl-kernel/build.sh (+2/-0)
BODY: 

### L1-88defc4d89  (L1, 2025-04-15, sha 88defc4d89b7, PR #5434)
TITLE: fix: solve release issue (#5434)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); .github/workflows/release-pypi-kernel.yml (+0/-44); .github/workflows/release-whl-kernel-cu118.yml (+3/-3); .github/workflows/release-whl-kernel.yml (+44/-10); sgl-kernel/build.sh (+0/-2)
BODY: ## Motivation ⏎  ⏎ ref https://github.com/sgl-project/sglang/actions/runs/14477264110 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8ec0bb7d55  (L1, 2025-04-15, sha 8ec0bb7d558d, PR #5436)
TITLE: chore: upgrade sgl-kernel 0.0.9.post1 (#5436)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-177320a582  (L1, 2025-04-16, sha 177320a582ec, PR #5467)
TITLE: Clean up imports (#5467)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+12/-26); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+12/-19); python/sglang/srt/layers/moe/topk.py (+10/-20); python/sglang/__init__.py (+2/-4); python/sglang/bench_serving.py (+0/-4); python/sglang/lang/__init__.py (+0/-0); python/sglang/lang/backend/anthropic.py (+0/-4); python/sglang/lang/backend/base_backend.py (+1/-1); python/sglang/lang/backend/openai.py (+1/-1); python/sglang/lang/backend/vertexai.py (+0/-1); (+41 more)
BODY: - Deprecate `python/sglang/srt/server.py` ⏎ - Make the import order cleaner ⏎ - Clean up all `import vllm` ⏎ - Move `scaled_fp8_quant` from `custom_ops.py` to `fp8_kernel.py` ⏎ - Clean up the usage of `from sglang.srt.custom_op import scaled_fp8_quant as sgl_scaled_fp8_quant`. Do not use if/else over this.

### L1-8beb356f0d  (L1, 2025-04-17, sha 8beb356f0daa, PR #5205)
TITLE: Refactor DeepSeek decoder layer branches (#5205)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+47/-21)
BODY: ## Motivation ⏎  ⏎ This code depends on https://github.com/sgl-project/sglang/pull/5190 and please subtract diff from there ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-06d0a3d92b  (L1, 2025-04-17, sha 06d0a3d92b25, PR #5496)
TITLE: [Bug fix] use correct func path in deepseek (#5496)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek.py (+1/-1)
BODY: Running with deepseek-moe-16b-chat，error coours: ⏎  ⏎ final_hidden_states = fused_moe( ⏎ TypeError: 'module' object is not callable ⏎   ⏎ Fix it by using correct path ⏎  ⏎ ## Checklist

### L1-c08a717c77  (L1, 2025-04-17, sha c08a717c7728, PR #5500)
TITLE: [Feat] Update sgl-kernel flashinfer to latest main version (#5500)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+2/-2); sgl-kernel/csrc/common_extension.cc (+12/-17); sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu (+5/-1); sgl-kernel/csrc/speculative/speculative_sampling.cuh (+4/-4); sgl-kernel/include/sgl_kernel_ops.h (+16/-26); sgl-kernel/python/sgl_kernel/elementwise.py (+108/-8); sgl-kernel/python/sgl_kernel/sampling.py (+213/-38); sgl-kernel/tests/test_sampling.py (+33/-37)
BODY: ## Motivation ⏎  ⏎  ⏎ Update flashinfer. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8e09b37077  (L1, 2025-04-17, sha 8e09b370777c, PR #5440)
TITLE: Sgl kernel fused_moe_gate support n_shared_experts (#5440)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/csrc/common_extension.cc (+2/-1); sgl-kernel/csrc/moe/moe_fused_gate.cu (+81/-28); sgl-kernel/include/sgl_kernel_ops.h (+8/-2); sgl-kernel/python/sgl_kernel/moe.py (+18/-2); sgl-kernel/tests/test_moe_fused_gate.py (+31/-5)
BODY: ## Motivation ⏎  ⏎  ⏎ ```shell ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30001 ⏎  ⏎ ➜  sglang python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 --port 30001 ⏎ 100%|███████████████████████████████████████████████████████| 1319/1319 [00:56<00:00, 23.30it/s] ⏎ Accuracy: 0.955 ⏎ Invalid: 0.000 ⏎ Latency: 58.947 s ⏎ Output throughput: 2358.942 token/s ⏎ ```  ⏎  ⏎ ## Mo …[truncated]

### L1-f28d82997a  (L1, 2025-04-17, sha f28d82997ac5, PR #5518)
TITLE: chore: bump sgl-kernel 0.0.9.post2 (#5518)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-bed05878f6  (L1, 2025-04-18, sha bed05878f6c8, PR #5461)
TITLE: fix kimi vl running bug after rebase main (#5461)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-0)
DEEP_STUDY: deep-study correctness case sglang:bed05878f6: class=shape_alignment_edge; symptom=crash_or_exception; introducing=unknown
BODY: ## Motivation ⏎  ⏎ Fix: ⏎  ⏎ ```shell ⏎ File "/eightT/open_source/sglang/python/sglang/srt/models/deepseek_v2.py", line 1177, in forward_normal ⏎     hidden_states = self.mlp(hidden_states) ⏎                     ^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/eightT/open_source/sglang/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl ⏎     return self._call_impl(*args, **kwargs) ⏎            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ …[truncated]

### L1-2c11f9c2eb  (L1, 2025-04-18, sha 2c11f9c2eba4, PR #5540)
TITLE: chore: upgrade sgl-kernel 0.0.9.post2 (#5540)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/sampler.py (+3/-7); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1e0806f30b  (L1, 2025-04-18, sha 1e0806f30b99, PR #5340)
TITLE: Fix DeepGEMM masked cannot be run on groups not being multiple or 4 (#5340)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+0/-3)
BODY: ## 2025.04.17 ⏎  ⏎ Originally I added some assertions to ensure we do not accidentally enter the slow branch of deepgemm preparation, but @ch-wan reviewed and suggested to remove it, thus the new code does not contain this. ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d58e354472  (L1, 2025-04-19, sha d58e35447203, PR #5504)
TITLE: simplify the control logic for using shared experts fusion (#5504)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-0); python/sglang/srt/layers/moe/fused_moe_native.py (+4/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-0); python/sglang/srt/layers/moe/topk.py (+15/-10); python/sglang/srt/layers/quantization/__init__.py (+1/-0); python/sglang/srt/layers/quantization/blockwise_int8.py (+2/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+4/-0); python/sglang/srt/layers/quantization/fp8.py (+2/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+2/-0); (+6 more)
BODY: ## Motivation ⏎  ⏎ Follow [pr 5440](https://github.com/sgl-project/sglang/pull/5440) , and following  @merrymercy ‘s suggestion, simplify the control logic for using shared experts fusion. Setting `--n-shared-experts-fusion=0` by default means it is not enabled, while other values indicate it is enabled. In the parameters, it is noted that setting it to the current `--tp-size` can achieve the best performance. However, this optimization lacks a tun …[truncated]

### L1-d07e797ace  (L1, 2025-04-20, sha d07e797ace35, PR #5149)
TITLE: Fix bench_one_batch producing unnatural results for expert parallel (#5149)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/bench_one_batch.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ Without fix ⏎  ⏎ ![image](https://github.com/user-attachments/assets/89471fa6-872f-4546-ba82-427088156848) ⏎  ⏎ With fix ⏎  ⏎ ![image](https://github.com/user-attachments/assets/2815b208-fddb-4606-908d-fd6d224f9533) ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d9dd529854  (L1, 2025-04-20, sha d9dd529854f7, PR #5571)
TITLE: enable DeepSeek V3 shared_experts_fusion in sm90 (#5571)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+12/-0)
BODY: ## H200 Benchmark ⏎  ⏎ I tested the benchmark using the command provided by https://github.com/sgl-project/sglang/issues/5514. To avoid warmup, I turned off deepgemm. Below are the results and detailed test records. ⏎  ⏎ <img width="676" alt="图片" src="https://github.com/user-attachments/assets/617075f5-aea8-432a-b4e0-da33995ab775" /> ⏎  ⏎ **The difference in the data set of 5000-1000 was due to fluctuations. I retested it once, and there was no differe …[truncated]

### L1-463d4b7400  (L1, 2025-04-20, sha 463d4b7400e4, PR #5567)
TITLE: Fix DeepEP cannot run on latest master (#5567)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-0)
BODY: ## Motivation ⏎  ⏎ waiting for CI ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-417b44eba8  (L1, 2025-04-20, sha 417b44eba8b9, PR #5417)
TITLE: [Feat] upgrade pytorch2.6 (#5417)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); .github/workflows/pr-test-sgl-kernel.yml (+1/-1); benchmark/deepseek_v3/README.md (+1/-1); docs/start/install.md (+1/-1); python/sglang/srt/layers/dp_attention.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
LABELS: high priority, dependencies
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-dc86f25a57  (L1, 2025-04-21, sha dc86f25a57eb, PR #5021)
TITLE: Tiny remove duplicated code (#5021)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+0/-7)
BODY: ## Motivation ⏎  ⏎ seems we have identical code  ⏎  ⏎ ![image](https://github.com/user-attachments/assets/c1f76fd1-b4e3-4969-bf12-68acd14d6e49) ⏎ ![image](https://github.com/user-attachments/assets/bb72ece2-1eea-4fb0-b270-144de6c2aa8c) ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2aa3f5e2d0  (L1, 2025-04-22, sha 2aa3f5e2d084, PR #5641)
TITLE: [feature] Add H20 fp8_w8a8 FusedMoE config for --n-share-experts-fusion=16 (#5641)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎ The original fusedMoE config on H20 will become invalid after enabling the new feature --n-share-experts-fusion=16(here tp=16). So we added relevant config to improve the performance of the fusedMoE. ⏎  ⏎ ## Modifications ⏎  ⏎ add E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json ⏎  ⏎ ## Tuning Command ⏎ ``` ⏎ python sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \ ⏎ --model /sgl-workspace …[truncated]

### L1-e62c49557d  (L1, 2025-04-22, sha e62c49557dfc, PR #5281)
TITLE: [1/2] Add FP8 Blockscale MoE CUTLASS kernel for Blackwell (#5281)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/csrc/common_extension.cc (+5/-0); sgl-kernel/csrc/moe/cutlass_moe_helper.cu (+142/-0); sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+386/-0); sgl-kernel/include/sgl_kernel_ops.h (+14/-0); sgl-kernel/python/sgl_kernel/__init__.py (+6/-1); sgl-kernel/python/sgl_kernel/moe.py (+30/-0); sgl-kernel/tests/test_fp8_blockwise_moe.py (+148/-0)
LABELS: high priority
BODY: ## Motivation ⏎ Functionality integration for FP8 blockscale MoE CUTLASS kernels on Blackwell.  ⏎ Huge thanks to @depaulmillz for CUTLASS library support.  ⏎   ⏎ cc @kushanam  ⏎  ⏎ ## Modifications  ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-711efe7814  (L1, 2025-04-23, sha 711efe781426, PR #5435)
TITLE: Integrating PD disaggregation with DP attention and DeepEP (#5435)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/decode.py (+46/-5); python/sglang/srt/disaggregation/prefill.py (+16/-0); python/sglang/srt/managers/data_parallel_controller.py (+10/-3)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Support DP attention and DeepEP in PD disaggregation. Discussed with @ByronHsu @ShangmingCai @whybeyoung. Tested by @liz-badada. ⏎  ⏎  ⏎ ## Evaluation ⏎  ⏎  ⏎  ⏎ - Prepare configuration files ⏎  ⏎ ```txt ⏎ # pd_node0.json ⏎ { ⏎     "local_hostname": "10.10.37.16", ⏎     "metadata_server": "http://10.10.37.16:8998/metadata", ⏎     "protocol": "rdma", ⏎     "device_name": "mlx5_7" ⏎ } ⏎  ⏎ # pd_node1.json ⏎ { ⏎     "local_hostname": "10.10.38.1", …[truncated]

### L1-7d0edf3cae  (L1, 2025-04-23, sha 7d0edf3caed4, PR #5688)
TITLE: chore: bump sgl-kernel 0.1.0 (#5688)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-43fb95c2fa  (L1, 2025-04-25, sha 43fb95c2facb, PR #5078)
TITLE: [Model] Support `ArcticForCausalLM` architecture (Snowflake/snowflake-arctic-instruct) (#5078)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/supported_models/generative_models.md (+1/-0); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/arctic.py (+127/-0); python/sglang/srt/hf_transformers_utils.py (+2/-0); python/sglang/srt/models/arctic.py (+634/-0)
DEEP_STUDY: deep-study: this PR was reverted by PR 5754 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Motivation ⏎  ⏎ Support the https://huggingface.co/Snowflake/snowflake-arctic-instruct ⏎  ⏎ ## Modifications ⏎  ⏎ Add to model config and models ⏎  ⏎ ## Checklist

### L1-5641a09458  (L1, 2025-04-25, sha 5641a0945831, PR #5754)
TITLE: Revert "[Model] Support `ArcticForCausalLM` architecture (Snowflake/snowflake-arctic-instruct)" (#5754)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/supported_models/generative_models.md (+0/-1); python/sglang/srt/configs/__init__.py (+0/-2); python/sglang/srt/configs/arctic.py (+0/-127); python/sglang/srt/hf_transformers_utils.py (+0/-2); python/sglang/srt/models/arctic.py (+0/-634)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 5078 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#5078 ⏎  ⏎ @b8zhong can you fix this error? https://github.com/sgl-project/sglang/actions/runs/14674277098/job/41187542685#step:4:25 ⏎  ⏎ I will revert this for now.

### L1-18ce468d56  (L1, 2025-04-25, sha 18ce468d56aa, PR #5740)
TITLE: update triton 3.2.0 h200 fused moe triton config and add warning about triton fused_moe_kernel performance degradation due to different Triton versions. (#5740)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+41/-41); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+4/-1)
BODY: ## Motivation ⏎  ⏎ <img width="839" alt="图片" src="https://github.com/user-attachments/assets/06f8db13-b826-4bc1-8523-ff2894478cb4" /> ⏎  ⏎ Reproduction commands : ⏎  ⏎ ```shell ⏎ SGL_ENABLE_JIT_DEEPGEMM=0 python3 -m sglang.launch_server --model /DeepSeek-V3 --tp 8 --trust-remote-code --enable-dp-attention --dp-size 8 ⏎  ⏎ # Random 1k, 2k ⏎ python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 100 --request-rate 10 --dataset-name random --rando …[truncated]

### L1-133ded039a  (L1, 2025-04-26, sha 133ded039a70, PR #5716)
TITLE: perf: update H20 fused_moe_triton kernel config to get higher throughput during prefilling (#5716)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+27/-27)
BODY: ## Motivation ⏎ We have provided fused_moe_triton config in PR(https://github.com/sgl-project/sglang/pull/5641) to improve model performance. ⏎ However, we found that after using this config, the throughput of the model during **prefilling period** dropped by about 3%. ⏎ After careful analysis, we found that the fused_moe_triton using this config will show different performance under different triton versions. ⏎ In the environment of triton==3.1.0 an …[truncated]

### L1-a086a11305  (L1, 2025-04-26, sha a086a113050f, PR #4971)
TITLE: Use sgl-kernel sgl_per_token_group_quant_int8 (#4971)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+7/-1); python/sglang/srt/layers/quantization/int8_kernel.py (+32/-1)
BODY: ## Motivation ⏎ Based on https://github.com/sgl-project/sglang/pull/4396, use the `sgl_per_token_group_quant_int8` method in the new version of sgl-kernel. ⏎  ⏎ ## Modifications ⏎ Modify `int8_kernel.py` and `fused_moe.py`, and add `sglang_per_token_group_quant_int8` method and related calls. ⏎  ⏎ ## Checklist

### L1-621e96bf9b  (L1, 2025-04-27, sha 621e96bf9b28, PR #5769)
TITLE: [CI] Fix ci tests (#5769)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_native.py (+2/-4); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+10/-16); python/sglang/bench_one_batch.py (+6/-0); python/sglang/srt/model_executor/model_runner.py (+11/-9); python/sglang/srt/models/llama4.py (+0/-1); python/sglang/srt/server_args.py (+10/-24); python/sglang/srt/torch_memory_saver_adapter.py (+10/-1); python/sglang/srt/utils.py (+1/-1); python/sglang/test/test_utils.py (+32/-19); test/srt/run_suite.py (+21/-21); (+8 more)
BODY: 

### L1-f0365820e8  (L1, 2025-04-27, sha f0365820e807, PR #)
TITLE: [Misc] add structure logging, write to file and log tracing for SGL Router
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L1-41ac0c6d48  (L1, 2025-04-27, sha 41ac0c6d4839, PR #5690)
TITLE: chore: upgrade sgl-kernel 0.1.0 (#5690)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+9/-0); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0045f4b2af  (L1, 2025-04-28, sha 0045f4b2af46, PR #5833)
TITLE: feat: Add fused moe triton config for qwen3 moe on h100 (#5833)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=192,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+5/-0)
BODY: ## Motivation ⏎ <img width="843" alt="image" src="https://github.com/user-attachments/assets/8e6913c7-f4d6-4f05-a5bc-1e5ed3aa6577" /> ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e132cba2a8  (L1, 2025-04-28, sha e132cba2a883, PR #5842)
TITLE: fused moe triton tuning script support qwen3 (#5842)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/README.md (+7/-0); benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py (+6/-1); benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py (+5/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-92ab0a2055  (L1, 2025-04-28, sha 92ab0a205588, PR #5839)
TITLE: feat: Add fused moe triton config for qwen3bf16 moe on h20 (#5839)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=192,device_name=NVIDIA_H20.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=384,device_name=NVIDIA_H20.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=768,device_name=NVIDIA_H20.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=96,device_name=NVIDIA_H20.json (+146/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ configs include TP1,2,4,8 for 30B, TP8 for 235B ⏎  ⏎ ref: https://github.com/sgl-project/sglang/pull/5740 ⏎  ⏎ | H20/30B/TP1         |                                              |                                             | ⏎ | ------------------- | -------------------------------------------- | ------------------------------------------- | ⏎ | Input-output length | before Output token throughput (tok/s):    …[truncated]

### L1-74cb12a878  (L1, 2025-04-28, sha 74cb12a8786f, PR #5846)
TITLE: [config] qwen3moe_tune_h20 fp8 tp4 (#5846)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=384,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: configs includes TP4  for 235B-FP8 ⏎  ⏎ ref: #5740 ⏎  ⏎  ⏎ # H20/235B-FP8/TP4 ⏎  ⏎ | Input-Output Length | Before Output Token Throughput (tok/s) | After Output Token Throughput (tok/s) | ⏎ |:--------------------|:---------------------------------------|:--------------------------------------| ⏎ | 1000-2000            | 1733.18                                | 1966.39                               | ⏎ | 5000-1000            | 835.71                         …[truncated]

### L1-d73ddeb196  (L1, 2025-04-28, sha d73ddeb196fa, PR #5850)
TITLE: feat: Add fused moe triton config for qwen3-30b-fp8 moe on h20 (#5850)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=768,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: configs include TP1 for 30B FP8 ⏎  ⏎ ref: https://github.com/sgl-project/sglang/pull/5740 ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ H20-30B-TP1-FP8 |   |   ⏎ -- | -- | -- ⏎ Input-output length | before Output token throughput (tok/s): | after Output token throughput (tok/s): ⏎ 1000-2000 | 3034.82 | 3221.53 ⏎ 5000-1000 | 1477.37 | 1484.21 ⏎ 10000-500 | 460.97 | 466.79 ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d364b9b0f2  (L1, 2025-04-28, sha d364b9b0f26d, PR #5816)
TITLE: ROCm: update AITER (#5816)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+15/-17); .github/workflows/pr-test-amd.yml (+6/-6); 3rdparty/amd/tuning/benchmark_moe_rocm.py (+1/-1); docker/Dockerfile.rocm (+2/-2); python/sglang/srt/layers/quantization/fp8.py (+20/-22); python/sglang/srt/layers/quantization/fp8_utils.py (+2/-2)
BODY: `co-author: kkHuang-amd` ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎ AITER update to v0.1.1 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a0251a3fd6  (L1, 2025-04-28, sha a0251a3fd648, PR #5849)
TITLE: add fused moe config for qwen3moe fp8/bf16 (#5849)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=192,device_name=NVIDIA_H200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=384,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=384,device_name=NVIDIA_H200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=768,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=768,device_name=NVIDIA_H200.json (+146/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+1/-6)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Reproduction commands : ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path xxx --tp xxx ⏎  ⏎ # Random 1k, 2k ⏎ python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 100 --request-rate 10 --dataset-name random --random-input-len 1000 --random-output-len 2000 --random-range-ratio 1 --warmup-requests 5 ⏎  ⏎ # Random 5k, 1k ⏎ python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 100 --r …[truncated]

### L1-1cc326032d  (L1, 2025-04-28, sha 1cc326032db6, PR #5801)
TITLE: simplify fused_moe config logging (#5801)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+8/-6); benchmark/kernels/fused_moe_triton/README.md (+1/-1)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist
