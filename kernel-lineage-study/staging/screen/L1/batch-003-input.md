### L1-f18a8fddd4  (L1, 2025-07-01, sha f18a8fddd418, PR #7698)
TITLE: chore: upgrade flashinfer v0.2.7.post1 (#7698)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ fix flashinfer.comm ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8e03b641ba  (L1, 2025-07-01, sha 8e03b641baf1, PR #7683)
TITLE: [1/n] apply wna16marlin kernel in moe weight only quantization (#7683)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.runner.marlin
FILES: sgl-kernel/csrc/moe/marlin_moe_wna16/awq_marlin_repack.cu (+255/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/core/registration.h (+25/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/generate_kernels.py (+106/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/gptq_marlin/marlin.cuh (+96/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/gptq_marlin/marlin_dtypes.cuh (+83/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/gptq_marlin_repack.cu (+333/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel.h (+40/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4.cu (+89/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4b8.cu (+109/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku8b128.cu (+109/-0); (+17 more)
BODY: ## Motivation ⏎  ⏎  ⏎ This is the sgl-kernel part of https://github.com/sgl-project/sglang/pull/5639. A series of PRs will remove the wna16 quantization dependency on vllm and optimize the feature. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-88f484ce4c  (L1, 2025-07-02, sha 88f484ce4c73, PR #7677)
TITLE: Apply dsv3 router gemm kernel for deepseek-r1 fp4 (#7677)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+21/-3)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Apply kernel added in #7627 to dpsk-r1 fp4 model. ⏎  ⏎ Currently we have to convert the dtype of router logits back to bf16, otherwise there will be cuda illegal memory bug. We should modify the router gemm kernel so that it outputs bf16 logits. ⏎  ⏎ This PR can improve performance on bs = 1, but for larger bs like 4 or 16, the performance will drop due to the extra dtype conversions. ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎ ```bash ⏎ python3 benchma …[truncated]

### L1-82f021e22e  (L1, 2025-07-02, sha 82f021e22e09, PR #6512)
TITLE: [router] add --log-level to sgl-router (#6512)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+3/-3); sgl-router/README.md (+1/-1); sgl-router/py_src/sglang_router/launch_router.py (+10/-8); sgl-router/py_test/test_launch_router.py (+1/-0); sgl-router/src/config/types.rs (+3/-3); sgl-router/src/lib.rs (+6/-6); sgl-router/src/server.rs (+12/-6)
BODY: ## Motivation ⏎  ⏎  ⏎ Add `--log-level` to router so that we can control the amount of output log. ⏎  ⏎ Thank you for your time on reviewing this PR :) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2c4feaf308  (L1, 2025-07-02, sha 2c4feaf30875, PR #7278)
TITLE: Add CUTLASS FP8 Blockscale MoE kernel for Hopper architecture (#7278)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+239/-6); sgl-kernel/benchmark/bench_fp8_blockwise_group_gemm.py (+330/-0); sgl-kernel/tests/test_fp8_blockwise_moe.py (+9/-3)
BODY: ## Motivation ⏎  ⏎  ⏎ Using the benchmark we provided in the PR, under some common configs, the CUTLASS FP8 Blockscale MoE kernel has ~5%-6% speedup over DeepGEMM. ⏎  ⏎ We carried out further analysis on the CUTLASS FP8 Blockscale MoE kernel, and the primary conclusions are as follows: ⏎  ⏎ 1. When using the schedule of Pingpong, there is no requirement to synchronize Warp Groups after the execution of the Mainloop. In comparison with the Cooperative ap …[truncated]

### L1-1e0e549766  (L1, 2025-07-03, sha 1e0e549766a0, PR #7722)
TITLE: Ascend attention backend(PA&MLA) (#7722)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+5/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+38/-0); python/sglang/srt/layers/moe/topk.py (+5/-0); docs/backend/attention_backend.md (+6/-0); python/sglang/srt/layers/attention/ascend_backend.py (+219/-0); python/sglang/srt/layers/rotary_embedding.py (+2/-2); python/sglang/srt/managers/schedule_batch.py (+5/-1); python/sglang/srt/mem_cache/allocator.py (+161/-0); python/sglang/srt/mem_cache/memory_pool.py (+148/-0); python/sglang/srt/model_executor/forward_batch_info.py (+9/-2); (+7 more)
LABELS: npu
BODY: ## Motivation ⏎  ⏎ Currently only `torch_native` backend works for NPU, it has cycle by batch that works slow for bigger batch, in this MR new Ascend Attention backend for NPU device was implemented. It has big performance improvement for big batch size. ⏎  ⏎ ## Modifications ⏎  ⏎ - PA ⏎     - New attention backend - `AscendAttnBackend`: support paged attention with page size 128 only. ⏎      ⏎        New option for server argument `attention-backend` - " …[truncated]

### L1-9fcc9a80e7  (L1, 2025-07-03, sha 9fcc9a80e7bd, PR #7647)
TITLE: [CPU] refine CPU integration code (#7647)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-6); python/sglang/srt/layers/amx_utils.py (+86/-0); python/sglang/srt/layers/linear.py (+4/-3); python/sglang/srt/layers/logits_processor.py (+2/-2); python/sglang/srt/layers/quantization/fp8.py (+6/-5); python/sglang/srt/layers/quantization/w8a8_int8.py (+6/-5); python/sglang/srt/layers/vocab_parallel_embedding.py (+2/-6); python/sglang/srt/models/deepseek_v2.py (+29/-20); python/sglang/srt/utils.py (+2/-69)
LABELS: intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎ 1. As discussed here https://github.com/sgl-project/sglang/commit/7eb47b0f3d0cd69ff3e0dd140cb377e15d6b148a#r161090822, we only need to check the `weight_block_size` if the model is employing `block_quant`. In addition, this check is the requirement of CPU kernels so I limit the check to happen only when `_is_cpu and _is_cpu_amx_available`. ⏎  ⏎ 2. When checking the supported dims for prepack, fix the dim check when weight is 3D o …[truncated]

### L1-1dce6c480f  (L1, 2025-07-03, sha 1dce6c480fac, PR #6771)
TITLE: [CPU] support the case where num_attention_heads or intermediate_size is not divisible by the TP size (#6771)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+33/-9); python/sglang/srt/configs/update_config.py (+119/-0); python/sglang/srt/layers/linear.py (+80/-11); python/sglang/srt/layers/parameter.py (+67/-7); python/sglang/srt/layers/vocab_parallel_embedding.py (+9/-1); python/sglang/srt/managers/scheduler.py (+9/-3); python/sglang/srt/model_executor/model_runner.py (+6/-1); python/sglang/srt/model_loader/weight_utils.py (+54/-0); python/sglang/srt/models/mllama4.py (+13/-7); python/sglang/srt/models/qwen2.py (+7/-1); (+1 more)
LABELS: intel, cpu
BODY: ## Motivation ⏎ Support the case where num_attention_heads or intermediate_size is not divisible by the TP size, for example running TP = 6 on the below machine with 6 numa nodes: ⏎  ⏎ ```sh ⏎ NUMA:                     ⏎   NUMA node(s):          6 ⏎   NUMA node0 CPU(s):     0-39,240-279 ⏎   NUMA node1 CPU(s):     40-79,280-319 ⏎   NUMA node2 CPU(s):     80-119,320-359 ⏎   NUMA node3 CPU(s):     120-159,360-399 ⏎   NUMA node4 CPU(s):     160-199,400-439 ⏎    …[truncated]

### L1-c01a1df588  (L1, 2025-07-03, sha c01a1df5888c, PR #7723)
TITLE: [Bug] add flashinfer bool check for fusedmoe in Qwen moe models (#7723)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen2_moe.py (+9/-0); python/sglang/srt/models/qwen3_moe.py (+9/-0)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎ The current Qwen MoE models (`qwen2_moe.py` and `qwen3_moe.py`) do not pass the `enable_flashinfer_moe` flag, causing `FuseMoE` to default to `enable_flashinfer_moe=False`. ⏎  ⏎ ## Modifications ⏎  ⏎ Added checks to ensure the `enable_flashinfer_moe` flag is correctly passed to the `FuseMoE` initializer.

### L1-2998c4bdf4  (L1, 2025-07-03, sha 2998c4bdf4fe, PR #7744)
TITLE: [optimize] fuse renormalize into moe_topk_softmax (#7744)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_softmax
FILES: sgl-kernel/csrc/common_extension.cc (+1/-3); sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu (+157/-81); sgl-kernel/csrc/torch_extension_rocm.cc (+1/-3); sgl-kernel/include/sgl_kernel_ops.h (+1/-4); sgl-kernel/python/sgl_kernel/moe.py (+2/-2); sgl-kernel/benchmark/bench_moe_topk_softmax.py (+0/-4); sgl-kernel/tests/test_moe_topk_softmax.py (+92/-4)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ fuse normalize into moe took softmax kernel. ⏎ This PR modifies topk_softmax interface, we need also modify python code. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-aca1101a13  (L1, 2025-07-03, sha aca1101a1396, PR #7755)
TITLE: chore: bump sgl-kernel 0.2.2 (#7755)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-489934be0a  (L1, 2025-07-03, sha 489934be0ad3, PR #7751)
TITLE: fuse renormal into moe topk softmax kernel python code (#7751)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L1.routing.topk_py, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/topk.py (+1/-25); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎ fuse renormal into moe topk softmax kernel python code, need merge #7744 and update sgl-kernel first ⏎  ⏎  ⏎  ⏎ ``` ⏎ accuracy ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 1000 ⏎ Accuracy: 0.951 ⏎ Invalid: 0.000 ⏎ Latency: 132.842 s ⏎ Output throughput: 1027.763 token/s ⏎ ``` ⏎  ⏎ before ⏎ | index | max_concurrency | input_throughput | output_throughput | mean_ttft_ms | median_ttft_ms | p99_ttft_ms | mean_tpot_ms | median_tpot_ms | p …[truncated]

### L1-8b1942c6cc  (L1, 2025-07-03, sha 8b1942c6cc08, PR #7759)
TITLE: Remove type conversion and fix id map in topk (#7759)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-1)
BODY: ## Motivation ⏎  ⏎ Remove type conversion and fix id map in topk.  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-da3890e82a  (L1, 2025-07-04, sha da3890e82a97, PR #7772)
TITLE: [1/n]: add cutlass W4A8 moe kernel for hopper architecture (#7772)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes, corpus:kernel-correctness-cases(introducing)
ARTIFACT_HINTS: L1.cutlass.w4a8
FILES: sgl-kernel/CMakeLists.txt (+3/-0); sgl-kernel/csrc/common_extension.cc (+19/-0); sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu (+91/-0); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh (+92/-0); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu (+240/-0); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh (+276/-0); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu (+79/-0); sgl-kernel/include/sgl_kernel_ops.h (+29/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/python/sgl_kernel/cutlass_moe.py (+112/-0); (+6 more)
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:de4990a5b2 (fix PR 9392)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This is the sgl-kernel part of https://github.com/sgl-project/sglang/pull/7762. A series of PRs will enable running [DeepSeek-R1-W4AFP8](https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8) using sglang. ⏎  ⏎ Additionally, this kernel supports other MoE models with INT4 MoE weight and FP8 activation quantization. ⏎  ⏎ Co-author: yicwang <yichen.wang@bytedance.com> ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ In this kernel implementation, we focus on …[truncated]

### L1-8e9fb43d82  (L1, 2025-07-04, sha 8e9fb43d8255, PR #7782)
TITLE: Optimize Hopper CUTLASS FP8 Blockwise Grouped GEMM Kernel in Small K Scenario (#7782)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+86/-38)
LABELS: high priority
BODY: ## Motivation ⏎ Follow https://github.com/sgl-project/sglang/pull/7278 ⏎ When K in GEMM problem size is small, CUTLASS Kernel shows suboptimal performance. The reason is that when K is small, the execution time of Mainloop is very short and Epilogue is difficult to be overlapped by Mainloop. In CUTLASS, Epilogue performs unnecessary LinearCombination, which further degrades performance. ⏎ We compared the nsight-compute profile reports of DeepGEMM an …[truncated]

### L1-c797322280  (L1, 2025-07-04, sha c797322280b4, PR #7444)
TITLE: fix: fix apply_shuffle_mul_sum (#7444)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/prepare_moe_input.cu (+67/-43)
LABELS: ready-to-merge
BODY: ## Motivation ⏎ Some bugs in `apply_shuffle_mul_sum_kernel`: ⏎ 1. some experts outputs were wrongly truncated by `src_row >= m`, where src_row is mistakenly taken as original ⏎    token index ( [0, m) ) ⏎ 2. in original impl, the thread-count assigned to a block is insufficient to cover row_stride(hidden_size, 4096). Modify with a strided loop ⏎  ⏎ 3. vectorize loading, increase blockDim ⏎  ⏎  ⏎ |                     | gsm8k                  |  ⏎ |-------- …[truncated]

### L1-4fece12be9  (L1, 2025-07-05, sha 4fece12be982, PR #7784)
TITLE: chore: bump sgl-kernel v0.2.3 (#7784)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8fc910db03  (L1, 2025-07-05, sha 8fc910db0310, PR #7222)
TITLE: DP Attention with Auto DeepEP Dispatch (#7222)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+5/-3); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+15/-13); python/sglang/srt/model_executor/forward_batch_info.py (+2/-0); python/sglang/srt/models/deepseek_v2.py (+7/-7); python/sglang/srt/models/qwen3_moe.py (+7/-9); python/sglang/srt/server_args.py (+0/-4); python/sglang/srt/two_batch_overlap.py (+7/-3); python/sglang/srt/disaggregation/decode.py (+1/-1); python/sglang/srt/disaggregation/prefill.py (+2/-2); python/sglang/srt/managers/schedule_batch.py (+3/-0); (+3 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR enables auto DeepEP dispatch for DP attention. Integration with TBO will be supported in future PRs. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-77cfea689d  (L1, 2025-07-05, sha 77cfea689d41, PR #7786)
TITLE: chore: upgrade sgl-kernel v0.2.3 (#7786)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f200af0d8c  (L1, 2025-07-05, sha f200af0d8cde, PR #7800)
TITLE: chore: bump sgl-kernel v0.2.4 (#7800)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-62f5522ffe  (L1, 2025-07-05, sha 62f5522ffe3f, PR #7801)
TITLE: chore: upgrade sgl-kernel v0.2.4 (#7801)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e00715eb66  (L1, 2025-07-06, sha e00715eb66aa, PR #5246)
TITLE: [AMD] Add test_fused_moe.py and test_rope_rocm.py to AMD CI (#5246)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: test/srt/run_suite.py (+2/-0); test/srt/test_fused_moe.py (+32/-5); test/srt/test_rope_rocm.py (+116/-0)
BODY: ## Motivation ⏎ To have better coverage of AMD CI ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - test_fused_moe.py ⏎ - test_rope_rocm.py ⏎  ⏎ This PR is to enable test_fused_moe.py following the newly-merged PR: https://github.com/sgl-project/sglang/pull/5102. The main difference on ROCm is the part for the conversion from e4m3fn to e4m3fnuz. ⏎  ⏎ To test it on ROCm locally, please run ⏎ ``` ⏎ SGLANG_USE_AITER=1 python3 test/srt/test_fused_moe.py  ⏎ SGLANG_USE_AITER=0 SGLANG …[truncated]

### L1-253454de9b  (L1, 2025-07-06, sha 253454de9b53, PR #7689)
TITLE: Integrate triton moe kernel (#7689)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+95/-54); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+176/-0); python/sglang/srt/server_args.py (+6/-0); benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py (+271/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0); test/srt/test_triton_fused_moe.py (+146/-0)
BODY: ## Motivation ⏎  ⏎ This PR is to follow up https://github.com/sgl-project/sglang/issues/7287. ⏎ The main purpose is to replace fused_moe with Triton v3.4.0 matmul_ogs kernel, in order to improve fused_moe's performance. ⏎  ⏎ ``` ⏎ #python ./benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py --use-cuda-graph ⏎ INFO 07-01 07:27:05 [__init__.py:244] Automatically detected platform cuda. ⏎ shape_configs={'num_experts': 128, 'topk': 8, 'h …[truncated]

### L1-a3398d8478  (L1, 2025-07-07, sha a3398d8478e9, PR #7794)
TITLE: Optimize moe align block size kernel (#7794)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+94/-63); sgl-kernel/include/utils.h (+6/-0)
BODY: ## Motivation ⏎  ⏎ - Optimize prefix sum with Blelloch scan (O(logN)) ⏎ - Reduce global memory access ⏎  ⏎ main: ⏎ ``` ⏎     num_tokens  num_experts  topk        SGL  SGL Fusion      Triton ⏎ 0          1.0        128.0   1.0  22.688000   20.624001   32.384001 ⏎ 1          1.0        128.0   2.0  22.688000   20.800000   32.512002 ⏎ 2          1.0        128.0   4.0  22.688000   20.768000   32.543998 ⏎ 3          1.0        128.0   8.0  22.528000   20.640001 …[truncated]

### L1-cb9d91ea8a  (L1, 2025-07-07, sha cb9d91ea8a71, PR #7762)
TITLE: feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode (#7762)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+215/-0); python/sglang/srt/layers/moe/ep_moe/kernels.py (+58/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+140/-2); python/sglang/srt/configs/model_config.py (+12/-1); python/sglang/srt/layers/quantization/__init__.py (+2/-0); python/sglang/srt/layers/quantization/fp8.py (+27/-6); python/sglang/srt/layers/quantization/w4afp8.py (+264/-0); python/sglang/srt/models/deepseek_v2.py (+6/-0); python/sglang/srt/server_args.py (+1/-0); python/sglang/test/test_cutlass_w4a8_moe.py (+281/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ This PR supports running [DeepSeek-R1-W4AFP8](https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8) model with ep-moe mode(deepep mode support is on the way~) ⏎ Due to the reduced space required for model weights and decreased bandwidth usage, DeepSeek R1 models can now be run on a single H2O or H100, leading to improved throughput and latency. ⏎  ⏎ ## Usage: ⏎ Run without mtp: ⏎ ``` ⏎ SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m sglang.launch …[truncated]

### L1-659907e32b  (L1, 2025-07-08, sha 659907e32b95, PR #7129)
TITLE: Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang (#7129)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+39/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+244/-1); python/sglang/srt/models/mllama4.py (+360/-79)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Enable ModelOpt Llama4 fp8 checkpoint deployment in SGLang, as part of our efforts to promote ModelOpt in SGLang. See https://github.com/sgl-project/sglang/issues/5251 ⏎  ⏎ Resolve request in https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/203  ⏎  ⏎ ## Modifications ⏎  ⏎ - Introduced `ModelOptFp8MoEMethod` to support Llama 4 FP8 MoE: ⏎   - Handles weight and scale creation, post-processing, and kernel invocation.  ⏎ - Enhanced  …[truncated]

### L1-d379bda4fa  (L1, 2025-07-08, sha d379bda4fade, PR #7853)
TITLE: [Bugfix] Fix two batch overlap with auto DeepEP Dispatch (#7853)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/two_batch_overlap.py (+1/-0)
BODY: ## Motivation ⏎ Fix https://sgl-fru7574.slack.com/archives/C08QGMU93GX/p1751871263127489 ⏎  ⏎ CC: @zhyncs @fzyzcjy  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-128f16a817  (L1, 2025-07-08, sha 128f16a81728, PR #7818)
TITLE: [CPU]convert topk_weights to fp32 for INT8 and FP8 paths (for llama4) and fix LmHead weight pack (#7818)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-3); sgl-kernel/csrc/cpu/moe.cpp (+10/-5); python/sglang/srt/layers/vocab_parallel_embedding.py (+9/-3)
LABELS: intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎ 1. Convert `topk_weights` to fp32 for INT8 and FP8 paths since the CPU kernel requires it to be fp32. ⏎ Previously we've fixed this issue in BF16 path for `llama4`. The same fix is needed for INT8 and FP8. ⏎ https://github.com/sgl-project/sglang/blob/3646f6bb3e42ef31b29e4bae3244a14333a2ba9b/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L320-L322 ⏎  ⏎ 2. For the weight pack check in `ParallelLMHead`, we previously checked t …[truncated]

### L1-d487555f84  (L1, 2025-07-09, sha d487555f84ee, PR #7872)
TITLE: [CI] Add deepep tests to CI (#7872)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+2/-1); scripts/ci_install_deepep.sh (+75/-0); .github/workflows/pr-test.yml (+44/-5); test/srt/run_suite.py (+6/-5); test/srt/test_deepep_large.py (+145/-0); test/srt/test_deepep_small.py (+384/-0); test/srt/test_dp_attention.py (+0/-81)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ The followup PR of #5655 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d389bedf72  (L1, 2025-07-09, sha d389bedf72a6, PR #7838)
TITLE: [CPU][Qwen3 MoE] Enable fused_topk CPU fusion and enhance FP8 TP padding (#7838)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+7/-1); python/sglang/srt/layers/parameter.py (+19/-3)
LABELS: ready-to-merge, intel, cpu
BODY: This PR contains the following fixes when run with fp8 MoE models on CPUs like https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8   ⏎  ⏎ 1. Enable fused_topk CPU fusion ⏎    ->  `fused_topk_cpu ` could cover the original `fused_topk `func on CPU path ⏎ 2. fix `load_qkv_weight `loading when odd TP size ⏎    -> When running with TP size that is not dividable, refering to other weight loader, adding `narrow_padded_param_and_loaded_weight `for `load_qkv_weight …[truncated]

### L1-766392c6bd  (L1, 2025-07-10, sha 766392c6bda2, PR #7791)
TITLE: [feature]Ascend quantization support (#7791)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-1); python/sglang/srt/layers/moe/topk.py (+1/-1); python/sglang/srt/configs/model_config.py (+3/-1); python/sglang/srt/layers/linear.py (+10/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+1/-2); python/sglang/srt/layers/quantization/w8a8_int8.py (+738/-14); python/sglang/srt/mem_cache/memory_pool.py (+4/-2); python/sglang/srt/model_loader/loader.py (+23/-12); python/sglang/srt/models/llama.py (+2/-0); (+3 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ support W8A8 quantized models inference on Ascend servers. ⏎  ⏎ ## Modifications ⏎  ⏎ - quant ⏎  ⏎     Extended W8A8 quantization support with NPU specific quantization configurations for static and dynamic cases. ⏎  ⏎     Added NPU specific quantization support interface classes for Linear layer utilized by respective configurations: ⏎     `NPU_W8A8LinearMethod`, `NPU_W8A8DynamicLinearMethod` ⏎  ⏎     Added NPU specific quantized Linear la …[truncated]

### L1-191d836ff6  (L1, 2025-07-11, sha 191d836ff616, PR #7953)
TITLE: fix: minor fix for modelopt weight load compatibility (#7953)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-1)
BODY: ## Motivation ⏎ fix compatibility when loading weight with different pack method. ⏎ related PR: https://github.com/sgl-project/sglang/pull/7129 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ only apply dim check when using modelopt quantization ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-475a249bb8  (L1, 2025-07-11, sha 475a249bb86a, PR #7961)
TITLE: temporarily disable deepep-8-gpu and activate two small tests (#7961)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+21/-21); test/srt/test_deepep_small.py (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8f3173d0b0  (L1, 2025-07-11, sha 8f3173d0b072, PR #7964)
TITLE: chore: bump sgl-kernel v0.2.5 (#7964)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-732fc8e405  (L1, 2025-07-11, sha 732fc8e405df, PR #7971)
TITLE: chore: upgrade sgl-kernel 0.2.5 (#7971)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5f6756b038  (L1, 2025-07-12, sha 5f6756b038ff, PR #7814)
TITLE: [BugFix] fix pre_reorder_triton_kernel default int32 issue (#7814)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+4/-2)
BODY: …cause overflow with large grid size ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎ [Bugfix] fix pre_reorder_triton_kernel default int32 issue which may cause overflow with large grid size ⏎  ⏎ ## Modifications ⏎ Modified the default index type for src and dst in pre_reorder_triton_kernel from int32 to int64 to avoid overflow caused by excessively large grid sizes. ⏎ ## Checklist

### L1-c07f647c9f  (L1, 2025-07-14, sha c07f647c9f3f, PR #8021)
TITLE: perf: add kimi k2 fused_moe tuning config for h30_3e (#8021)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Add h20_3e fused MoE tuning config for Kimi K2 (E=384) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a562c8a35c  (L1, 2025-07-14, sha a562c8a35c93, PR #7902)
TITLE: [Dockerfile] Multi-arch support for ROCm (#7902)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile.rocm (+95/-35); python/pyproject.toml (+0/-1); python/sglang/srt/layers/quantization/fp8_utils.py (+2/-2)
BODY: ## Motivation ⏎  ⏎ To support multi-arch for ROCm ⏎  ⏎ ## Modifications ⏎  ⏎ - Dockerfile ⏎ - pyproject.toml ⏎  ⏎ ## Checklist

### L1-38216cf049  (L1, 2025-07-15, sha 38216cf04950, PR #7943)
TITLE: concurrently load weights of DeepseekV2ForCausalLM (#7943)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+148/-127)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ The previous patch https://github.com/sgl-project/sglang/pull/7277 supported multi-thread model weight loading (from NVMe to CPU memory), significantly reducing the model weight loading time by fully utilizing the NVMe bandwidth. Building on this, the current patch modifies the load_weight() function of DeepseekV2ForCausalLM to enable multi-threaded concurrent loading of weights (from CPU memory to device memory), further sho …[truncated]

### L1-14f1f1514b  (L1, 2025-07-15, sha 14f1f1514be5, PR #8047)
TITLE: H20 tune config for Kimi (#8047)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: add tune config for kimi on H20 , performance improvement will be update soon..  ⏎  ⏎ ## Performance ⏎  ⏎ * origin Kimi Performance ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang ⏎ Traffic request rate:                    inf ⏎ Max request concurrency:                 32 ⏎ Successful requests:                     128 ⏎ Benchmark duration (s):                  222.17 ⏎ Total input tokens:           …[truncated]

### L1-d9eb5efc71  (L1, 2025-07-16, sha d9eb5efc71b1, PR #8098)
TITLE: [misc] update nvshmem and pin deepEP commit hash (#8098)
SOURCES: subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+6/-6)
LABELS: dependencies
BODY: ## Motivation ⏎  ⏎ Update the Docker build to use the latest NVSHMEM version 3.3.9 and improve reproducibility by pinning the DeepEP dependency to a specific commit hash.  ⏎  ⏎ ## Modifications ⏎  ⏎ - **Updated NVSHMEM version**: Upgraded from 3.2.5 to 3.3.9 ⏎   - Changed download URL to use `nvshmem_src_cuda12-all-all-3.3.9.tar.gz` ⏎   - Updated cleanup commands to match new filename ⏎ - **Pinned DeepEP commit**: Added explicit commit hash `b6ce310bb0b75 …[truncated]

### L1-c28ad1990d  (L1, 2025-07-16, sha c28ad1990d29, PR #7992)
TITLE: [1/n] chore: decouple quantization implementation from vLLM dependency (#7992)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.marlin
FILES: python/sglang/srt/layers/moe/fused_moe_triton/__init__.py (+4/-1); sgl-kernel/python/sgl_kernel/fused_moe.py (+2/-1); python/sglang/srt/layers/quantization/__init__.py (+2/-4); python/sglang/srt/layers/quantization/gptq.py (+491/-119); python/sglang/srt/layers/quantization/marlin_utils.py (+781/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+30/-0); python/sglang/srt/layers/quantization/quant_utils.py (+0/-166); python/sglang/srt/layers/quantization/scalar_type.py (+0/-0); python/sglang/srt/layers/quantization/utils.py (+162/-1); sgl-kernel/tests/test_marlin_repack.py (+2/-4); (+3 more)
LABELS: high priority
BODY: ## Motivation ⏎ The primary goal of this change is to enhance the consistency and stability of SGLang's quantization features. By decoupling the quantization implementation from its vLLM dependency, we aim to make the module easier to maintain and more portable. ⏎ Full realization of this goal will involve several subsequent PRs; this particular PR addresses the GPTQ feature. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5c08a36cbf  (L1, 2025-07-16, sha 5c08a36cbfae, PR #8110)
TITLE: [Fix] ensure DeepGEMM is only enabled for FP8_W8A8 models (#8110)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+6/-0)
BODY: ## Motivation ⏎  ⏎ If a user erroneously enables the ENABLE_JIT_DEEPGEMM environment variable in non-FP8 model scenarios (e.g., when using Qwen3-235B-FP16), sglang will enter a failed state during launch and sglang‘s rank scheduler will stuck at the following stack: ⏎  ⏎  ⏎ Thread 66311 (active+gil): "Thread-3 (forward_thread_func)" ⏎     dispatch (deep_ep/buffer.py:349) ⏎     _dispatch_core (ep_moe/token_dispatcher.py:349) ⏎     dispatch_b (ep_moe/token …[truncated]

### L1-02404a1e35  (L1, 2025-07-17, sha 02404a1e35d9, PR #8105)
TITLE: [ci] recover 8-gpu deepep test (#8105)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: scripts/ci_install_deepep.sh (+11/-18); .github/workflows/pr-test.yml (+21/-21); test/srt/test_deepep_large.py (+11/-9); test/srt/test_deepep_small.py (+9/-11)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Recover 8-gpu deepep test and update installation file to make the ci environment align with the default development env updated in #8098. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-49b8777460  (L1, 2025-07-17, sha 49b8777460b7, PR #7989)
TITLE: Refactor: move all quantization-related code to `srt/layer/quantization` (#7989)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.runner.marlin, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-324); python/sglang/srt/layers/moe/fused_moe_triton/__init__.py (+0/-3); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+8/-367); python/sglang/srt/layers/moe/topk.py (+1/-5); python/sglang/srt/layers/linear.py (+13/-103); python/sglang/srt/layers/quantization/__init__.py (+4/-96); python/sglang/srt/layers/quantization/awq.py (+9/-7); python/sglang/srt/layers/quantization/base_config.py (+77/-9); python/sglang/srt/layers/quantization/blockwise_int8.py (+10/-27); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+9/-10); (+12 more)
LABELS: ready-to-merge
ISSUES: #7919 [Bug] install sglang by pip install sglang[all]>=0.4.9, work well when run llama3 model, but raise "most likely due to a circular import" error when check ColumnParallelLinear op..
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Several users have encountered circular import errors when they directly import some functions or classes in SGLang (e.g., #2386, #7919). These stem from the disorganized structure between our linear/MoE layers and quantization methods. Some quantization methods are implemented in layer files, which may incur import paths like layer -> quant -> layer. While #2387 offered a partial workaround by implementing dynamic class comp …[truncated]

### L1-af1cc8fe2d  (L1, 2025-07-17, sha af1cc8fe2dd8, PR #7884)
TITLE: [kernel] opt moe align block kernel by block/warp scan algorithm (#7884)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+51/-42)
DEEP_STUDY: deep-study: this PR was reverted by PR 8457 (confirmed_revert, reason=hardware_specific_breakage)
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is to introduce block / warp scan algorithm in fused MoE path **moe_align_block_size_kernel**  which gains approximately 10% speedup. ⏎  ⏎ Here is the benchmark result. ⏎ Note: num_experts >= 128 appliable to this PR. num_experts < 128 appliable to **moe_align_block_size_small_batch_expert_kernel**  kernel. ⏎  ⏎ This PR: ⏎ ``` ⏎ $python ./sgl-kernel/benchmark/bench_moe_align_block_size.py ⏎ INFO 07-09 13:15:28 [__init__.py:244] …[truncated]

### L1-48c1fa7bb6  (L1, 2025-07-17, sha 48c1fa7bb695, PR #7889)
TITLE: [CPU][Llama4] Fix Llama4 MoE inputs with "apply_router_weight_on_input"  (#7889)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+13/-0); python/sglang/srt/configs/update_config.py (+3/-1); python/sglang/srt/layers/quantization/fp8.py (+6/-0); python/sglang/srt/layers/quantization/unquant.py (+8/-3); python/sglang/srt/layers/quantization/w8a8_int8.py (+5/-0)
LABELS: high priority
DEEP_STUDY: deep-study correctness case sglang:48c1fa7bb6: class=numerical_precision; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: This PR fixes the support of "apply_router_weight_on_input" for llama4 model for the CPU path. ⏎ Differing from other MoE models (Qwen3/DeepSeek), llama4 applies the topk weight [before MoE calculation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py#L153). Here we follow the logic to revise the current CPU path, and added TODO to fuse this processing in MoE kernels next.

### L1-7891bac16b  (L1, 2025-07-17, sha 7891bac16b0a, PR #7820)
TITLE: [Quantization][w8a8_int8] Fix weight loading issue for w8a8_int8 path with "ignore" layer list in quantization config (#7820)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/unquant.py (+1/-1); python/sglang/srt/layers/quantization/w8a8_int8.py (+21/-15)
LABELS: high priority
BODY: This PR is referring to the way of [compressed_tensors](https://github.com/CatherineSue/sglang/blob/a41fcad7d9c53b3045bc8e8e9db3e6ad519b4f9d/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py#L132) loading format from the quantization config, adding `ignore`  list for `w8a8_int8` path, which is necessary since models may not be fully quantized.  ⏎ For example, W8A8 models will contain the `ignore` list: https://huggingf …[truncated]

### L1-1f76fc8747  (L1, 2025-07-18, sha 1f76fc874759, PR #8113)
TITLE: [3/n] chore: decouple AWQ implementation from vLLM dependency (#8113)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: benchmark/deepseek_v3/README.md (+9/-0); python/sglang/srt/layers/quantization/__init__.py (+7/-15); python/sglang/srt/layers/quantization/awq.py (+582/-2); python/sglang/srt/layers/quantization/utils.py (+84/-1); python/sglang/srt/models/deepseek_v2.py (+3/-1); python/sglang/test/test_marlin_moe.py (+286/-0); python/sglang/test/test_marlin_utils.py (+171/-0); test/srt/test_gptqmodel_dynamic.py (+1/-1)
LABELS: high priority, ready-to-merge
BODY: ## Motivation ⏎ follow https://github.com/sgl-project/sglang/pull/7992 ⏎ The primary goal of this change is to enhance the consistency and stability of SGLang's quantization features. By decoupling the quantization implementation from its vLLM dependency, we aim to make the module easier to maintain and more portable. ⏎ Full realization of this goal will involve several subsequent PRs; this particular PR addresses the AWQ feature. ⏎ ## Modifications …[truncated]

### L1-cfab0ff6e2  (L1, 2025-07-18, sha cfab0ff6e291, PR #8157)
TITLE: Add GB200 wide-EP docker (#8157)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+357/-0)
BODY: ## Motivation ⏎  ⏎ Creating a special dockerfile to build and reproduce the SGLang GB200 wideEP study ⏎  ⏎ ## Modifications ⏎  ⏎ In additional to the base changes in https://github.com/sgl-project/sglang/pull/7721 ⏎ * Use a fork of DeepEP ⏎ * Install mooncake from source ⏎ * Use CUDA_VERSION=12.8.1 as default ⏎  ⏎  ⏎ ## Checklist

### L1-15ad6c9086  (L1, 2025-07-19, sha 15ad6c908670, PR #7966)
TITLE: [1/N] MoE Refactor: refactor `select_experts` (#7966)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+13/-74); python/sglang/srt/layers/moe/fused_moe_native.py (+7/-47); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+7/-38); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-29); python/sglang/srt/layers/moe/topk.py (+171/-5); python/sglang/srt/layers/quantization/__init__.py (+9/-23); python/sglang/srt/layers/quantization/awq.py (+8/-31); python/sglang/srt/layers/quantization/base_config.py (+14/-7); python/sglang/srt/layers/quantization/blockwise_int8.py (+7/-28); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+21/-71); (+29 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This pull request extracts the `select_experts` computation from within `FusedMoE` and `EPMoE`, moving it outside these modules. This refactoring offers three key benefits: ⏎  ⏎ - Enable gate-router fusion.  ⏎ - Simplifying MoE's input: reducing input number from 16 to 7. ⏎ - Unifying API with DeepEPMoE. ⏎  ⏎ This PR temporarily disables `triton_kernel_moe`, which will be added back later. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f98e88b9fb  (L1, 2025-07-19, sha f98e88b9fbbb, PR #8165)
TITLE: chore: bump sgl-kernel v0.2.6 (#8165)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-561dd7b2ce  (L1, 2025-07-19, sha 561dd7b2ce2b, PR #8166)
TITLE: chore: upgrade sgl-kernel 0.2.6 (#8166)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-abda2542d5  (L1, 2025-07-19, sha abda2542d5cd, PR #8175)
TITLE: Fix tuning_fused_moe_triton.py (#8175)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+9/-5)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0f9b11e310  (L1, 2025-07-19, sha 0f9b11e3101b, PR #8176)
TITLE: feat: add h200 tp 16 kimi k2 moe config (#8176)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f62d75b6a1  (L1, 2025-07-19, sha f62d75b6a17d, PR #8178)
TITLE: feat: add b200 tp 16 kimi k2 moe config (#8178)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a589a07167  (L1, 2025-07-19, sha a589a0716774, PR #7825)
TITLE: fix moe gate dtype, fix tbo, fix fake dispatch (#7825)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+1/-1); python/sglang/srt/eplb/expert_location_dispatch.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Fix TBO after #7222 import is_extend_in_batch, add this to TBO batch filter ⏎  ⏎ Fix dtype of e_score_correction_bias, which need to be float32 but bfloat16 now. ⏎  ⏎ reference: https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main?show_file_info=model-00001-of-000163.safetensors ⏎ ![image](https://github.com/user-attachments/assets/f4181632-e424-4fbe-9dba-849f5dff79b6) ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Other changes: ⏎  ⏎ Change fake d …[truncated]

### L1-bbcfbc1a02  (L1, 2025-07-19, sha bbcfbc1a0249, PR #8183)
TITLE: feat: add h200 tp 16 kimi k2 moe config (#8183)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-429bb0efa2  (L1, 2025-07-20, sha 429bb0efa203, PR #8200)
TITLE: chore: bump sgl-kernel v0.2.6.post1 (#8200)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-465968b2e3  (L1, 2025-07-21, sha 465968b2e328, PR #8197)
TITLE: Fix dtype error in CI (#8197)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/actions/runs/16400875083/job/46340127530#step:5:489 ⏎ ``` ⏎   File "/sglang-checkout/python/sglang/srt/layers/moe/topk.py", line 526, in biased_grouped_topk_gpu ⏎     aiter_biased_grouped_topk( ⏎   File "/sgl-workspace/aiter/aiter/jit/core.py", line 631, in wrapper ⏎     return op(*args, **kwargs) ⏎            ^^^^^^^^^^^^^^^^^^^ ⏎ RuntimeError: gating_output.dtype() == correction_bias.dtype() ⏎ ``` …[truncated]

### L1-74f59ae555  (L1, 2025-07-21, sha 74f59ae55557, PR #8202)
TITLE: chore: upgrade sgl-kernel 0.2.6.post1 (#8202)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c9e8613c97  (L1, 2025-07-21, sha c9e8613c9708, PR #8193)
TITLE: Apply fused sorted token ids padding (#8193)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+7/-2)
BODY: ## Motivation ⏎  ⏎ Apply https://github.com/sgl-project/sglang/pull/7437. ⏎  ⏎ ## Accuracy ⏎ For `DeepSeek-V3-0324`. ⏎ ``` ⏎ Accuracy: 0.936 ⏎ Invalid: 0.000 ⏎ Latency: 32.717 s ⏎ Output throughput: 4104.390 token/s ⏎ ``` ⏎  ⏎ ## Benchmark ⏎ main branch: ⏎ ``` ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |    …[truncated]

### L1-e50109f2ed  (L1, 2025-07-21, sha e50109f2edfe, PR #7484)
TITLE: [AMD] Remove vllm's scaled_fp8_quant and moe_sum when SGLANG_USE_AITER=1 (#7484)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-4); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+21/-5); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+3/-2); python/sglang/srt/layers/quantization/fp8.py (+1/-2); python/sglang/srt/layers/quantization/fp8_kernel.py (+115/-46); python/sglang/srt/layers/quantization/unquant.py (+0/-1); python/sglang/srt/layers/quantization/utils.py (+3/-2); python/sglang/test/test_custom_ops.py (+12/-7)
LABELS: high priority, ready-to-merge
BODY: ## Motivation ⏎  ⏎ This is part of the efforts to remove vllm's dependency on ROCm.  ⏎ The vLLM's `scaled_fp8_quant` and `moe_sum` will be replaced with the aiter's counterparts when `SGLANG_USE_AITER=1` ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎  ⏎ ### To run the unit tests for `scaled_fp8_quant`: ⏎ ``` ⏎ /sgl-workspace/sglang/python/test# SGLANG_USE_AITER=1 pytest test_custom_ops.py ⏎ /sgl-workspace/sglang/python/test# SGLANG_USE_AITER=0 pytest test_ …[truncated]

### L1-0f8b538614  (L1, 2025-07-22, sha 0f8b5386145c, PR #8059)
TITLE: [fix] benchmark : routed_scaling_factor is None (#8059)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: sgl-kernel/benchmark/bench_moe_fused_gate.py (+5/-2)
BODY: ## Motivation ⏎  ⏎ Fix benchmark for bench_moe_fused_gate.py ⏎  ⏎  ⏎ Previous, when running `python sgl-kernel/benchmark/bench_moe_fused_gate.py`:  ⏎  ⏎ ``` ⏎ Traceback (most recent call last): ⏎ ..... ⏎   File "/peter/sglang/sgl-kernel/benchmark/bench_moe_fused_gate.py", line 13, in biased_grouped_topk_org ⏎     return biased_grouped_topk( ⏎            ^^^^^^^^^^^^^^^^^^^^ ⏎   File "/peter/sglang/python/sglang/srt/layers/moe/topk.py", line 318, in biased_gro …[truncated]

### L1-4c605235aa  (L1, 2025-07-23, sha 4c605235aa83, PR #8302)
TITLE: fix: workaround for deepgemm warmup issue (#8302)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/CMakeLists.txt (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4953f4ca9a  (L1, 2025-07-23, sha 4953f4ca9a3a, PR #8304)
TITLE: chore: upgrade sgl-kernel 0.2.7 (#8304)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8d1c5b948e  (L1, 2025-07-24, sha 8d1c5b948ed0, PR #8301)
TITLE: chore: upgrade flashinfer v0.2.9rc1 (#8301)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: high priority
BODY: up flashinfer version

### L1-a167fd0bcb  (L1, 2025-07-24, sha a167fd0bcb9e, PR #8310)
TITLE: [code style] Clean dead triton kernel code in fused_moe and useless vllm_ops import (#8310)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+25/-224); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+2/-9); python/sglang/srt/layers/quantization/utils.py (+0/-9)
BODY: ## Motivation ⏎  ⏎ Follow https://github.com/sgl-project/sglang/pull/7794 . ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-12cb760a37  (L1, 2025-07-25, sha 12cb760a3773, PR #8344)
TITLE: Add H20-3e fused MoE kernel tuning configs for Qwen3-Coder-480B-A35B-Instruct (#8344)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=160,N=320,device_name=NVIDIA_H20-3e.json (+146/-0)
BODY: ## Motivation ⏎  ⏎ Add H20-3e fused MoE kernel tuning configs for Qwen3-Coder-480B-A35B-Instruct ⏎  ⏎ ## Modifications ⏎  ⏎ Add H20-3e fused MoE kernel tuning configs for Qwen3-Coder-480B-A35B-Instruct ⏎  ⏎ Result (without Moe config): ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang     ⏎ Traffic request rate:                    inf        ⏎ Max request concurrency:                 10         ⏎ Succe …[truncated]

### L1-ed2e313eb6  (L1, 2025-07-25, sha ed2e313eb667, PR #8332)
TITLE: Clean up server_args, triton cache manager (#8332)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+3/-4); python/sglang/srt/entrypoints/engine.py (+0/-6); python/sglang/srt/entrypoints/http_server.py (+20/-32); python/sglang/srt/managers/scheduler.py (+5/-6); python/sglang/srt/model_executor/forward_batch_info.py (+8/-7); python/sglang/srt/model_executor/model_runner.py (+0/-1); python/sglang/srt/server_args.py (+59/-49); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+0/-1); python/sglang/srt/utils.py (+0/-65); test/srt/test_deepep_large.py (+1/-1); (+2 more)
BODY: - clean up server args to make them more organized ⏎ - remove triton cache manager since the monkey patch is not needed anymore

### L1-58c468f404  (L1, 2025-07-25, sha 58c468f4045e, PR #8333)
TITLE: Fix FP4 MoE accuracy from missing routed_scaling_factor (#8333)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/modelopt_quant.py (+8/-4); python/sglang/srt/server_args.py (+0/-4)
ISSUES: #7166 [Bug] Deepseek R1 FP4 model quality drop
BODY: ## Motivation ⏎  ⏎ This PR fixes the accuracy issues with FP4 MoE. ⏎  ⏎ Fixes https://github.com/sgl-project/sglang/issues/7166 ⏎  ⏎ Before: ⏎ ``` ⏎ sglang (pretrained=nvidia/DeepSeek-R1-0528-FP4,trust_remote_code=True,quantization=modelopt_fp4,tp_size=8,max_model_len=32768,add_bos_token=True,enable_flashinfer_moe=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 512 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Std …[truncated]

### L1-9045cc1eb8  (L1, 2025-07-25, sha 9045cc1eb8da, PR #8353)
TITLE: [torch.compile bug] avoid biased_grouped_topk_impl func repeatedly triggering `torch.compile` in forward pass (#8353)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-9); docs/references/hardware.rst (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e6312d271d  (L1, 2025-07-26, sha e6312d271d86, PR #8356)
TITLE: Uodate Dockerfile.gb200 to latest sglang (#8356)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+28/-39)
BODY: ## Motivation ⏎  ⏎ Improve the Dockerfile.gb200 build to include latest sglang improvements. ⏎  ⏎ ## Modifications ⏎  ⏎ - Bump sglang version in Dockerfile.gb200 to commit: a167fd0bcb9ef4b0f4331a109e40c8cdc770b026, ⏎ this is a work-around because flashinfer v0.2.9rc1 doesn't build for aarch64. ⏎ - Use sgl kernel 0.2.7 ⏎ - Move NVSHMEM to 3.3.9 ⏎ - pip install mooncake `0.3.5 ` as oppose to install from source ⏎  ⏎ ## Checklist

### L1-da0c026084  (L1, 2025-07-26, sha da0c0260841e, PR #8381)
TITLE: Tiny assert EPLB is used together with expert parallel (#8381)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+3/-0)
ISSUES: #8379 [Bug] ScatterGatherKernel.cu:144: operator(): block: [0,0,0], thread: [8,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"` failed.
BODY: ## Motivation ⏎  ⏎ Close https://github.com/sgl-project/sglang/issues/8379 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8af145b7dc  (L1, 2025-07-26, sha 8af145b7dcb2, PR #8374)
TITLE: Fix test_moe_fused_gate_combined sgl-kernel ci test (#8374)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: sgl-kernel/tests/test_moe_fused_gate.py (+0/-1)
BODY: ## Motivation ⏎  ⏎ fix https://github.com/sgl-project/sglang/actions/runs/16533593515/job/46765376007?pr=8241#step:5:10875 after https://github.com/sgl-project/sglang/pull/8353 change.

### L1-85486b6f6f  (L1, 2025-07-27, sha 85486b6f6f72, PR #8036)
TITLE: [NVIDIA] Add Flashinfer MoE blockscale fp8 backend (#8036)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+102/-7); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+9/-7); python/sglang/srt/layers/quantization/modelopt_quant.py (+5/-5); python/sglang/srt/models/deepseek_v2.py (+44/-20); python/sglang/srt/models/qwen2_moe.py (+2/-2); python/sglang/srt/models/qwen3_moe.py (+2/-2); python/sglang/srt/server_args.py (+13/-3); python/sglang/srt/managers/schedule_batch.py (+2/-1)
LABELS: high priority
BODY: Enable flashinfer moe blockscale fp8 backend for low latency scenario. The e2e perf shows up to 3x improvement (see [here](https://github.com/sgl-project/sglang/pull/8036#issuecomment-3104285508)). ⏎  ⏎ cc. @kushanam @pavanimajety

### L1-2ab97023e3  (L1, 2025-07-27, sha 2ab97023e316, PR #8395)
TITLE: [router] add different policies for p node and d node (#8395)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+8/-0); sgl-router/README.md (+13/-1); sgl-router/py_src/sglang_router/launch_router.py (+65/-2); sgl-router/src/config/types.rs (+190/-0); sgl-router/src/config/validation.rs (+101/-0); sgl-router/src/lib.rs (+31/-16); sgl-router/src/policies/cache_aware.rs (+5/-1); sgl-router/src/routers/factory.rs (+21/-7); sgl-router/src/routers/pd_router.rs (+97/-55); sgl-router/tests/test_pd_routing.rs (+6/-0)
BODY: ## Motivation ⏎  ⏎ The SGLang router's PD (Prefill-Decode) disaggregated mode previously only supported a single routing policy for both prefill and decode nodes. This limitation prevented users from optimizing routing strategies based on the different characteristics of prefill and decode workloads. ⏎  ⏎ For example, prefill operations benefit from cache-aware routing to maximize cache hits, while decode operations might perform better with power-of …[truncated]

### L1-2a1936de96  (L1, 2025-07-27, sha 2a1936de96dd, PR #8351)
TITLE: Add A800 fused MoE kernel tuning configs for Qwen3-Coder-480B-A35B-Instruct (#8351)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_2_0/E=160,N=320,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8.json (+146/-0)
BODY: ## Motivation ⏎  ⏎ Add A800 fused MoE kernel tuning configs for Qwen3-Coder-480B-A35B-Instruct on channel-wise INT8. ⏎  ⏎ ## Modifications ⏎  ⏎ Add fused MoE kernel tuning config file. ⏎  ⏎ ## Benchmark ⏎  ⏎ Run command: ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path /path/to/model --trust-remote-code --host 0.0.0.0 --port 30000 --max-running-requests 128 --quantization w8a8_int8 --tp 8 ⏎  ⏎ ``` ⏎  ⏎ main: ⏎ ``` ⏎ ============ Serving Benchmark Result ====== …[truncated]

### L1-bf0f448fe5  (L1, 2025-07-27, sha bf0f448fe5b5, PR #8397)
TITLE: [2/N] MoE Refactor: Unify weight loader and quant methods (#8397)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+87/-217); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+31/-43); python/sglang/srt/layers/quantization/fp8.py (+25/-247); python/sglang/srt/layers/quantization/unquant.py (+10/-66); python/sglang/srt/layers/quantization/w4afp8.py (+68/-17)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4d921f2b79  (L1, 2025-07-27, sha 4d921f2b7916, PR #8405)
TITLE: [hotfix] fix merge conflicts in FlashInferEPMoE (#8405)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-10ee89559e  (L1, 2025-07-27, sha 10ee89559ee4, PR #8406)
TITLE: chore: upgrade flashinfer v0.2.9rc2 (#8406)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b3eac168e7  (L1, 2025-07-27, sha b3eac168e7de, PR #8258)
TITLE: Support triton kernels v3.4.0 for fused_moe (#8258)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+11/-8); python/sglang/srt/layers/moe/topk.py (+84/-22); python/sglang/srt/layers/quantization/unquant.py (+14/-10)
LABELS: high priority
BODY: ## Motivation  ⏎  ⏎ This PR is to follow up https://github.com/sgl-project/sglang/pull/7966 [1/N] MoE Refactor: refactor select_experts. It supports triton_kernels v3.4.0 for fused moe. ⏎ This PR involves a change to the TopKOutput data structure.  ⏎  ⏎  ⏎  ⏎ Note: it requires pytorch-triton. ⏎ ``` ⏎ Name: pytorch-triton ⏎ Version: 3.4.0+gitae848267 ⏎ ``` ⏎  ⏎ The result is as expected. ⏎ ``` ⏎ ➜  python git:(support_triton_moe) ✗ python3 -m sglang.launch_server …[truncated]

### L1-fe6a445d1e  (L1, 2025-07-27, sha fe6a445d1e1f, PR #8415)
TITLE: [router] improve router logs and request id header (#8415)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+5/-0); sgl-router/README.md (+13/-0); sgl-router/py_src/sglang_router/launch_router.py (+10/-0); sgl-router/src/config/types.rs (+7/-0); sgl-router/src/core/worker.rs (+4/-1); sgl-router/src/lib.rs (+7/-0); sgl-router/src/middleware.rs (+111/-0); sgl-router/src/policies/cache_aware.rs (+3/-5); sgl-router/src/routers/pd_router.rs (+62/-30); sgl-router/src/routers/router.rs (+63/-46); (+7 more)
LABELS: enhancement, feature, router
BODY: ## Motivation ⏎  ⏎ The router was generating verbose logs that made it difficult to debug issues in production environments. Additionally, there was no consistent way to track requests across the distributed system, making it challenging to correlate logs when debugging customer issues. This PR addresses these problems by: ⏎  ⏎ 1. Implementing request ID tracking for distributed tracing ⏎ 2. Reducing log verbosity while maintaining important operation …[truncated]

### L1-6d6a8bc278  (L1, 2025-07-27, sha 6d6a8bc278ea, PR #8224)
TITLE: GLM-4.5 Model Support (#8224)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py (+5/-1); benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py (+5/-1); python/sglang/srt/configs/model_config.py (+3/-0); python/sglang/srt/function_call/ebnf_composer.py (+10/-3); python/sglang/srt/function_call/function_call_parser.py (+2/-0); python/sglang/srt/function_call/glm4_moe_detector.py (+165/-0); python/sglang/srt/models/glm4_moe.py (+1034/-0); python/sglang/srt/models/glm4_moe_nextn.py (+167/-0); python/sglang/srt/reasoning_parser.py (+1/-0); python/sglang/srt/server_args.py (+2/-1); (+4 more)
LABELS: high priority
BODY: The SGLang version of the complete implementation of the GLM-4.5 model, which includes: ⏎ 1. Model implementation (with MTP and without MTP) ⏎ 2. Tool call and Reasoning parser ⏎ 3. transformers lib should use 4.54.0 or higher ⏎  ⏎ Everything is ready.

### L1-2262369905  (L1, 2025-07-28, sha 226236990588, PR #8457)
TITLE: Revert "[kernel] opt moe align block kernel by block/warp scan algorithm" (#8457)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+42/-51)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 7884 reason=hardware_specific_breakage
BODY: Reverts sgl-project/sglang#7884 ⏎  ⏎ PR has bug, and it caused a lot of test failture in ci.

### L1-134fa43e19  (L1, 2025-07-28, sha 134fa43e1940, PR #8453)
TITLE: [NVIDIA] Change to use `num_local_experts` (#8453)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-1); docs/backend/server_arguments.md (+2/-1)
BODY: The recent [change](https://github.com/sgl-project/sglang/commit/bf0f448fe5b549cc80bc86a505e0ceb040e0f613) breaks the usage of trtllm-moe-fp8 kernel. This PR fixes it. ⏎  ⏎ cc @kushanam @zhyncs

### L1-9c138a0445  (L1, 2025-07-28, sha 9c138a044514, PR #8421)
TITLE: [3/N] MoE Refactor: Simplify DeepEP Output (#8421)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+150/-30); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+69/-118); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+0/-0); python/sglang/srt/layers/moe/token_dispatcher/base_dispatcher.py (+48/-0); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+19/-0); python/sglang/srt/models/deepseek_v2.py (+13/-56); python/sglang/srt/models/qwen3_moe.py (+12/-69); python/sglang/srt/two_batch_overlap.py (+8/-3)
BODY: - Introduce `DispatchOutput` to maintain dispatcher's results. ⏎ - Move DeepEP's `dispatch` and `combine` operations from model files the moe layer file. ⏎  ⏎ After this PR, all forward functions of MoE share the same logic: dispatch -> layout transfer -> grouped-gemm. ⏎  ⏎ ## Checklist

### L1-74e7e45710  (L1, 2025-07-28, sha 74e7e457103a, PR #8469)
TITLE: Fix DEEPEP BF16 compatibility for Deepseek Style model like GLM 4.5 (#8469)
SOURCES: path_core, subject_keyword, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-6)
DEEP_STUDY: deep-study correctness case sglang:74e7e45710: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Motivation ⏎ Fix DEEPEP BF16 compatibility surfaced by running ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model /shared/public/elr-models/GLM-4.5/ --tp-size 8 --trust-remote-code --enable-deepep-moe --deepep-mode=normal ⏎ ``` ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3a04aa4be7  (L1, 2025-07-28, sha 3a04aa4be712, PR #8478)
TITLE: chore: add glm4 fp8 tp8 config (#8478)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=160,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8.json (+146/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8240a6b013  (L1, 2025-07-28, sha 8240a6b0132f, PR #8480)
TITLE: chore: add glm 4.5 fp8 tp4 config (#8480)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=160,N=384,device_name=NVIDIA_H200,dtype=fp8_w8a8.json (+146/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a9dd3ec3e9  (L1, 2025-07-28, sha a9dd3ec3e961, PR #8125)
TITLE: fix:reorder topk experts to ensure shared expert replaces minimal score (#8125)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+6/-2)
DEEP_STUDY: deep-study correctness case sglang:a9dd3ec3e9: class=other; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Motivation ⏎  ⏎ In the MoE (Mixture of Experts) layer, in the original design, to achieve integration between the shared expert and routed experts, when selecting top-k experts for each token, we need to reorder the selected experts to guarantee the last expert in the list has the smallest score. The original implementation uses torch.topk with sorted=False, which means the returned experts aren't necessarily in score order. When we replace the  …[truncated]

### L1-59d0bf012f  (L1, 2025-07-28, sha 59d0bf012f46, PR #8426)
TITLE: Tiny add warnings for DeepEP when it is suboptimal (#8426)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+14/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4d16c88b6e  (L1, 2025-07-29, sha 4d16c88b6e96, PR #8535)
TITLE: Update cutlass_moe.py (#8535)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+2/-1)
BODY: Minor change regarding cutlass MoE ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎ Fix a minor bug ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5973675bc3  (L1, 2025-07-29, sha 5973675bc30d, PR #8531)
TITLE: Fix moe align kernel test (#8531)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: sgl-kernel/tests/test_moe_align.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ Fix https://github.com/sgl-project/sglang/actions/runs/16483200176/job/46602410480#step:5:200.

### L1-9effeb5bdd  (L1, 2025-07-29, sha 9effeb5bddf2, PR #8448)
TITLE: Support EPLB in FusedMoE (#8448)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/eplb/expert_distribution.py (+5/-0); python/sglang/srt/eplb/expert_location.py (+17/-6); python/sglang/srt/eplb/expert_location_dispatch.py (+1/-0); python/sglang/srt/eplb/expert_location_updater.py (+2/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+16/-3); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+44/-1); python/sglang/srt/models/deepseek_v2.py (+2/-0); python/sglang/srt/models/glm4_moe.py (+3/-1); python/sglang/srt/models/grok.py (+3/-0); python/sglang/srt/models/llama4.py (+3/-0); (+5 more)
BODY: ## Motivation ⏎  ⏎ Fix #8398 ⏎  ⏎ ## Modifications ⏎  ⏎ ## Checklist

### L1-e3f08c77bc  (L1, 2025-07-29, sha e3f08c77bc8e, PR #8545)
TITLE: Update cutlass_moe.py (#8545)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+1/-1)
BODY: Another small fix for cutlass moe ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ Simple bug fix for FP8 cutlass moe ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a730ce8162  (L1, 2025-07-30, sha a730ce816214, PR #6869)
TITLE: [feature] [sgl-router] Add a dp-aware routing strategy (#6869)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+8/-0); docs/router/router.md (+8/-0); sgl-router/py_src/sglang_router/launch_router.py (+17/-0); sgl-router/py_test/run_suite.py (+1/-1); sgl-router/py_test/test_launch_router.py (+47/-0); sgl-router/py_test/test_launch_server.py (+279/-0); sgl-router/src/config/types.rs (+14/-0); sgl-router/src/config/validation.rs (+8/-0); sgl-router/src/core/error.rs (+5/-0); sgl-router/src/core/worker.rs (+25/-3); (+9 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ This PR add an additional dp-aware routing strategy on top of the  ⏎ sgl-router's hybrid "cache-aware load-balancing" routing strategy. ⏎  ⏎ The strategy can be enabled by setting the `--dp-aware` flag when ⏎ starting the router.  When enabled, the router will try to contact the ⏎ workers to get the `dp_size` of each worker, and add the new workers ⏎ in the **dp_rank** level. ⏎  ⏎ In such a case, the router will apply the cache-aware rou …[truncated]

### L1-66a398f49d  (L1, 2025-07-30, sha 66a398f49ddf, PR #8479)
TITLE: [router] migrate router from actix to axum (#8479)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+17/-2); sgl-router/Cargo.toml (+13/-13); sgl-router/py_src/sglang_router/launch_router.py (+37/-0); sgl-router/py_test/test_launch_router.py (+3/-2); sgl-router/src/config/types.rs (+14/-0); sgl-router/src/lib.rs (+12/-1); sgl-router/src/middleware.rs (+255/-51); sgl-router/src/routers/mod.rs (+29/-21); sgl-router/src/routers/pd_router.rs (+363/-332); sgl-router/src/routers/router.rs (+268/-277); (+8 more)
LABELS: enhancement, router
BODY: ## Motivation ⏎  ⏎ This PR migrates the SGLang router from Actix-web to Axum, addressing several long-standing architectural limitations and improving the overall maintainability and performance of the router. The migration was necessary to: ⏎  ⏎ 1. **Leverage Tower middleware ecosystem** - Axum's integration with Tower provides better middleware composition and reusability ⏎ 2. **Improve type safety** - Axum's type-driven approach reduces runtime err …[truncated]

### L1-9b9e82539b  (L1, 2025-07-30, sha 9b9e82539b77, PR #8564)
TITLE: [Fix]Fix index oob in get_group_gemm_starts kernel. (#8564)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/cutlass_moe_helper.cu (+6/-6)
BODY: ## Motivation ⏎ Using `int` as offset or stride may sometimes cause OOB problems. For example, when M=128, N=8192, K=8192, and groups=256, executing the following code will cause OOB:https://github.com/sgl-project/sglang/blob/55ecdc0a8e62ac56bb475f128d2b1fc728953a28/sgl-kernel/csrc/moe/cutlass_moe_helper.cu#L56 ⏎ So I fix it. ⏎  ⏎  ⏎ ## Modifications ⏎ sgl-kernel/csrc/moe/cutlass_moe_helper.cu ⏎  ⏎  ⏎ ## Accuracy Test ⏎ Test M=128, N=8192, K=8192, groups=2 …[truncated]

### L1-a5f5ab4030  (L1, 2025-07-30, sha a5f5ab4030a1, PR #8514)
TITLE: update sgl-kernel for EP: kernel part  (#8514)
SOURCES: path_core
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+6/-7); sgl-kernel/python/sgl_kernel/moe.py (+0/-2); sgl-kernel/benchmark/bench_moe_align_block_size.py (+0/-10); sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/csrc/torch_extension_rocm.cc (+1/-1); sgl-kernel/include/sgl_kernel_ops.h (+0/-1); sgl-kernel/tests/test_moe_align.py (+4/-10)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ In EP, we set the expert ids for filtered experts as -1. We update sgl-kernel to handle this case. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-43118f5f2a  (L1, 2025-07-30, sha 43118f5f2ad3, PR #8599)
TITLE: chore: bump sgl-kernel v0.2.8 (#8599)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e179e0b797  (L1, 2025-07-31, sha e179e0b79738, PR #8550)
TITLE: update sgl-kernel for EP: python part (#8550)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+4/-9); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎ See #8514  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3bdcdd134b  (L1, 2025-07-31, sha 3bdcdd134b1c, PR #8461)
TITLE: [Hot-Fix] moe_aligned_block_size CI failed in AMD (#8461)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/moe/moe_align_kernel.cu (+65/-6)
BODY: ## Motivation ⏎ This PR is to re-introduce the improvement of https://github.com/sgl-project/sglang/pull/7884, which was reverted in https://github.com/sgl-project/sglang/pull/8457 due to AMD CI failed constantly. ⏎  ⏎ This PR is to diverge the branches for HIP and CUDA. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-32fa1e9cc2  (L1, 2025-07-31, sha 32fa1e9cc286, PR #8515)
TITLE: [4/N] MoE Refactor: Unified Triton Kernel for FusedMoE and EPMoE (#8515)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+15/-648); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+22/-4); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+32/-12); python/sglang/srt/layers/quantization/fp8.py (+0/-18); python/sglang/srt/layers/quantization/unquant.py (+0/-8); python/sglang/srt/layers/quantization/w4afp8.py (+1/-0)
ISSUES: #8402 [Bug] DeepSeek-V3 model gets bad accuracy result on gsm8k benchmark when EP is enabled | #8427 [Feature] Support per channel quant EPMOE Triton kernel
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-51c38163c1  (L1, 2025-07-31, sha 51c38163c19c, PR #8583)
TITLE: model: support Step3V (#8583)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/backend/server_arguments.md (+1/-1); python/sglang/srt/configs/__init__.py (+8/-0); python/sglang/srt/configs/model_config.py (+3/-0); python/sglang/srt/configs/step3_vl.py (+172/-0); python/sglang/srt/conversation.py (+23/-0); python/sglang/srt/function_call/function_call_parser.py (+2/-0); python/sglang/srt/function_call/step3_detector.py (+436/-0); python/sglang/srt/hf_transformers_utils.py (+2/-0); python/sglang/srt/jinja_template_utils.py (+4/-1); python/sglang/srt/managers/template_manager.py (+62/-19); (+6 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ This PR adds the support for Step3VModel. ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Add Stepv3 model ⏎ - Add tool_call_parser `Step3VDetector` ⏎ - Add reasoning parser registry for `stepv3` ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-7a1f7fc504  (L1, 2025-07-31, sha 7a1f7fc5049d, PR #8590)
TITLE: [Feature] Hybrid EP and TP (#8590)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+21/-25); assets/logo.svg (+1/-1); assets/logo_square.svg (+1/-1); python/sglang/bench_one_batch.py (+3/-0); python/sglang/srt/distributed/parallel_state.py (+86/-1); python/sglang/srt/entrypoints/engine.py (+2/-0); python/sglang/srt/managers/data_parallel_controller.py (+2/-0); python/sglang/srt/managers/scheduler.py (+11/-1); python/sglang/srt/managers/tp_worker.py (+4/-0); (+4 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Dependency: ⏎ - #8514  ⏎ - #8515  ⏎ - #8550  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path /dev/shm/GLM-4.5-Air-FP8 --trust-remote-code --tp 4 --enable-ep-moe --base-gpu-id 4 --ep-size 2 ⏎ python3 few_shot_gsm8k.py ⏎ ``` ⏎  ⏎ Accuracy: 93.5% ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b7170cc820  (L1, 2025-07-31, sha b7170cc82062, PR #8630)
TITLE: [bugfix] Fix flashinfer cutlass EP moe after MoE refactor (#8630)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-1); python/sglang/srt/server_args.py (+5/-0)
LABELS: bug, high priority
BODY: ## Motivation ⏎  ⏎ Accuracy went to 0 for flashinfer cutlass MoE after recent MoE refactor and improvements. ⏎  ⏎ ## Modifications ⏎  ⏎ * Fix issue from https://github.com/sgl-project/sglang/pull/8515 - cutlass moe does not need to map topkid ids using expert map gpu. ⏎ * Fix issue from https://github.com/sgl-project/sglang/pull/8590 - hybrid EP/TP not yet supported for this path ⏎  ⏎ ## Accuracy Test ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path n …[truncated]

### L1-aa4c66b564  (L1, 2025-07-31, sha aa4c66b564b7, PR #8450)
TITLE: [NVIDIA] Enable Flashinfer MoE blockscale fp8 backend for TP MoE (#8450)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+19/-34); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+54/-1); python/sglang/srt/layers/quantization/fp8.py (+52/-0); python/sglang/srt/models/deepseek_v2.py (+3/-4); python/sglang/srt/models/glm4_moe.py (+3/-3); python/sglang/srt/server_args.py (+0/-4)
LABELS: high priority
BODY: A followup PR to enable Flashinfer MoE blockscale fp8 backend for TP MoE. ⏎  ⏎ The previous [PR](https://github.com/sgl-project/sglang/pull/8036) is doing the same but for the EP MoE. ⏎  ⏎ cc. @kushanam

### L1-c8d3a402c1  (L1, 2025-08-01, sha c8d3a402c1ca, PR #8511)
TITLE: Bug: apply final_hidden_states*=self.routed_scaling_factor at MoE lay… (#8511)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-1)
BODY: …er if epmoe is enabled ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎ Issue: https://github.com/sgl-project/sglang/issues/8402 ⏎  ⏎ Also noticed significant regression when running epmoe during recent GLM4.5 support work: GSM8K accuracy drops from 0.965 to 0.745 when EPMOE is enabled. Accuracy is good for TP & DeepEP. ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model  /shared/public/elr-models/zai-org/GLM-4.5 --tp-size 8 --trust-remote-code ⏎  ⏎ python3 benchmark/gsm8k/be …[truncated]

### L1-6c88f6c8d9  (L1, 2025-08-01, sha 6c88f6c8d908, PR #8658)
TITLE: [5/N] MoE Refactor: Update MoE parallelism arguments (#8658)
SOURCES: path_core, symbol_pickaxe, corpus:production-kernel-provenance
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+9/-35); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-5); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+23/-0); python/sglang/srt/layers/moe/token_dispatcher/base_dispatcher.py (+12/-1); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+8/-15); python/sglang/srt/layers/moe/utils.py (+43/-0); docker/k8s-sglang-distributed-sts.yaml (+1/-2); docs/backend/pd_disaggregation.md (+8/-8); docs/backend/server_arguments.md (+1/-2); docs/references/disaggregation/lws-examples/d.yaml (+4/-6); (+28 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR introduces `--moe-a2a-backend` and deprecates `--enable-ep-moe` and `--enable-deepep-moe`. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2ae95d17e8  (L1, 2025-08-01, sha 2ae95d17e807, PR #8647)
TITLE: Disable tp for shared experts under expert parallelism for GLM4.5 model (#8647) (#8647)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/glm4_moe.py (+73/-5)
BODY: ## Motivation ⏎  ⏎  ⏎ Currently when we enable **EPMoe** of the **glm4_moe** model, it still TP splits the shared experts, which blocks us to launch **FP8 per block quant GLM4.5** with **--tp 8 --enable-ep-moe**  ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server   --model /shared/public/elr-models/GLM-4.5-fp8-0725/   --tp-size 8   --trust-remote-code  --ep-size 8 ⏎  ⏎ File "/home/jobuser/zminglei/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py", l …[truncated]

### L1-1fe691a429  (L1, 2025-08-01, sha 1fe691a429be, PR #8648)
TITLE: Fix FP8 block quantization when N or K is not multiples of 128 (#8648)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/moe.cpp (+10/-10); test/srt/cpu/test_moe.py (+10/-4); test/srt/cpu/utils.py (+19/-4)
LABELS: ready-to-merge
DEEP_STUDY: deep-study correctness case sglang:1fe691a429: class=shape_alignment_edge; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ This PR is to add support of FP8 block quantize when N or K is not multiples of 128. ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f642524fd9  (L1, 2025-08-01, sha f642524fd992, PR #8364)
TITLE: [1/2] sgl-kernel: Fuse routed scaling factor into select_experts (#8364)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/csrc/moe/moe_fused_gate.cu (+20/-7); sgl-kernel/include/sgl_kernel_ops.h (+2/-1); sgl-kernel/python/sgl_kernel/moe.py (+9/-2); sgl-kernel/tests/test_moe_fused_gate.py (+6/-1)
LABELS: high priority
DEEP_STUDY: deep-study: this PR was reverted by PR 8706 (confirmed_revert, reason=unstated)
BODY: ## Motivation ⏎  ⏎ Follow up to https://github.com/sgl-project/sglang/pull/8333 ⏎ Fuse the multiply by routed_scaling_factor into select_experts, following example of TRT-LLM: ⏎ https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/models/modeling_deepseekv3.py#L323 ⏎ https://github.com/NVIDIA/TensorRT-LLM/blob/738ab615930fd08dccb94fa388bd74dc91c5f235/cpp/tensorrt_llm/kernels/noAuxTcKernels.cu#L651 ⏎  ⏎ For the non-FP4 paths, the routed_s …[truncated]

### L1-b27b11919c  (L1, 2025-08-01, sha b27b11919cfa, PR #8694)
TITLE: chore(gb200): update dockerfile to handle fp4 disaggregation (#8694)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+11/-4)
BODY: This dockerfile allows for FP4 disaggregation with DSR1. The commands are as follows  ⏎  ⏎ prefill ⏎  ⏎ ```bash ⏎ NCCL_MNNVL_ENABLE=1 \ ⏎ NCCL_CUMEM_ENABLE=1 \ ⏎ SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \ ⏎ PYTHONUNBUFFERED=1 \ ⏎ python3 -m sglang.launch_server \ ⏎ --disaggregation-transfer-backend nixl \ ⏎ --disaggregation-mode decode \ ⏎ --host 0.0.0.0 \ ⏎ --decode-log-interval 1 \ ⏎ --max-running-requests 1536 \ ⏎ --context-length 4224 \ ⏎ --max-total-tokens=20 …[truncated]

### L1-89caf7a3c6  (L1, 2025-08-01, sha 89caf7a3c6cd, PR #8688)
TITLE: [bugfix] Apply routed scaling factor to cutlass_fused_experts_fp8 (#8688)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+5/-1)
BODY: ## Motivation ⏎  ⏎ Similar to https://github.com/sgl-project/sglang/pull/8333 we weren't applyign routed scaling factor for cutlass_fused_experts_fp8 (SGLANG_CUTLASS_MOE=1) path. ⏎  ⏎ https://github.com/sgl-project/sglang/pull/8364 will fuse this into select_experts, but let's get the accuracy fixed first since that requires an sgl-kernel change and won't be merged for a while. ⏎  ⏎ ## Modifications ⏎  ⏎ Multiply MOE output by factor. ⏎  ⏎ ## Accuracy Test …[truncated]

### L1-82e6c3a65a  (L1, 2025-08-01, sha 82e6c3a65ab3, PR #8238)
TITLE: Add support for NCCL symmetric memory for TP allreduces (#8238)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+25/-18); docs/backend/server_arguments.md (+1/-0); python/sglang/srt/distributed/device_communicators/pynccl.py (+7/-0); python/sglang/srt/distributed/device_communicators/pynccl_allocator.py (+133/-0); python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py (+42/-3); python/sglang/srt/distributed/parallel_state.py (+11/-0); python/sglang/srt/entrypoints/engine.py (+3/-2); python/sglang/srt/layers/linear.py (+7/-1); python/sglang/srt/layers/vocab_parallel_embedding.py (+7/-1); python/sglang/srt/managers/schedule_batch.py (+1/-0); (+3 more)
DEEP_STUDY: deep-study: this PR was reverted by PR 10210 (partial_revert, reason=premature_or_process) || deep-study: this PR was reverted by PR 10238 (reland, reason=unstated)
BODY: ## Motivation ⏎  ⏎ Add support for [NCCL symmetric memory](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-registration). ⏎ This PR only enables TP allreduce. Follow up will enable other collectives. ⏎  ⏎ NCCL symmetric memory uses a separate memory pool to allocate tensors that will be used for communications, therefore it can lead to more memory fragmentation. This new feature is off by default and can be turned …[truncated]

### L1-0a56b721d5  (L1, 2025-08-02, sha 0a56b721d553, PR #8713)
TITLE: chore: bump sgl-kernel v0.2.9 (#8713)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8ada1ab6c7  (L1, 2025-08-02, sha 8ada1ab6c791, PR #8705)
TITLE: Fix triton moe error caused by TopK refactor (#8705)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+0/-31)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f9f0138f80  (L1, 2025-08-02, sha f9f0138f80a3, PR #8706)
TITLE: Revert "[1/2] sgl-kernel: Fuse routed scaling factor into select_experts" (#8706)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/csrc/moe/moe_fused_gate.cu (+7/-20); sgl-kernel/include/sgl_kernel_ops.h (+1/-2); sgl-kernel/python/sgl_kernel/moe.py (+2/-9); sgl-kernel/tests/test_moe_fused_gate.py (+1/-6)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 8364 reason=unstated
BODY: Reverts sgl-project/sglang#8364

### L1-d9def43dcd  (L1, 2025-08-02, sha d9def43dcdfd, PR #8722)
TITLE: [Perf]Use Cooperative Schedule for H100 & H200 & H800 in fp8_blockwise_scaled_grouped_mm (#8722)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+3/-2)
BODY: ## Motivation ⏎ Fix https://github.com/sgl-project/sglang/pull/7278#discussion_r2225518835 ⏎ Migrating fp8_blockwise_scaled_grouped_mm from H20 to H100, H200, and H800 resulted in a performance regression. After in-depth profiling, we identified several key causes of the regression: ⏎  ⏎ 1. The reduction in Mainloop execution time may not be enough to cover the Epilogue overhead. ⏎ 2. The fp8 blockwise calculation logic is to execute 4 Tensor Core MMA …[truncated]

### L1-a31b7a7024  (L1, 2025-08-03, sha a31b7a702456, PR #8547)
TITLE: feat: Add new moe triton for NVIDIA RTX 6000 Ada (#8547)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=128,N=352,device_name=NVIDIA_RTX_6000_Ada_Generation,dtype=fp8_w8a8.json (+146/-0)
BODY: ## Motivation ⏎  ⏎ See #8513 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3435a24e81  (L1, 2025-08-03, sha 3435a24e8157, PR #8676)
TITLE: [RL] fix update weight for FusedMoE with EP (#8676)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+11/-3)
LABELS: high priority
DEEP_STUDY: deep-study correctness case sglang:3435a24e81: class=integration_backend_cudagraph; symptom=performance_or_availability; introducing=unknown
BODY: ## Motivation ⏎  ⏎  ⏎ During colocated RL, we will use `/release_memory_occupation` to release all GPU allocations within model init, and we should not create GPU tensors that cannot be loaded through weight_loader. ⏎  ⏎ This PR moves the creation of `self.expert_map_gpu` to `forward` and make sure `self.expert_map_cpu` is created on CPU (which was by default created on GPU). ⏎  ⏎ Thank you for your time on reviewing this PR :) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  …[truncated]

### L1-0242bb9c74  (L1, 2025-08-03, sha 0242bb9c7437, PR #8732)
TITLE: Fix triton kernels topk with keyword arguments (#8732)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+22/-3)
BODY: ## Motivation ⏎  ⏎ For the triton kernels topk routing function, the meaning of `sm_first` is not exactly the same as `renormalize`. Use keyword arguments to avoid confusion and unknown bugs. ⏎ - If `sm_first` = True, the softmax will before the topk, so the result can be unnormalized before sort. `renormalize` is not supported in triton kernels rounting funciton. ⏎ - If `sm_first` = False, the softmax will after the topk, so the result is already no …[truncated]

### L1-e67276ecb3  (L1, 2025-08-03, sha e67276ecb305, PR #8678)
TITLE: feat: support cutlass_moe_fp8 kernel for fusedmoe in sm90 (#8678)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+20/-6); python/sglang/srt/layers/quantization/fp8.py (+3/-3); python/sglang/srt/layers/utils.py (+9/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ### reproduce ⏎ #### Qwen3-MOE, H20, TP4 ⏎ * server ⏎ ``` ⏎ SGL_ENABLE_JIT_DEEPGEMM=1 SGLANG_CUTLASS_MOE=1 python3 -m sglang.launch_server --model-path /data/models/Qwen3-235B-A22B-FP8 --tp 4 --trust-remote-code --host 0.0.0.0 --port 8080 --mem-fraction-static 0.7  --max-running-requests 128 --context-length 4096 --chunked-prefill-size 4096 ⏎ ``` ⏎ * gsm8k ⏎ ``` ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 320 --parallel …[truncated]

### L1-9f47d686e5  (L1, 2025-08-03, sha 9f47d686e521, PR #8709)
TITLE: Fix fused MoE when `routed_scaling_factor is None` (#8709)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-1)
BODY: Reproduction: ⏎  ⏎ ```bash ⏎ python3 -m sglang.compile_deep_gemm --host=0.0.0.0 --port=8000 --model=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 --tp=8 --trust-remote-code --enable-ep-moe --tool-call-parser=qwen25 --reasoning-parser=qwen3 --context-length=131072 ⏎ ``` ⏎  ⏎ Issue: ⏎  ⏎ ```bash ⏎   File "/root/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py", line 283, in forward_deepgemm ⏎     return output * self.routed_scaling_factor ⏎ TypeError: unsupported …[truncated]

### L1-b102353f8f  (L1, 2025-08-03, sha b102353f8f2d, PR #8735)
TITLE: [MoE] Enable `renormalize=False` in Triton kernels (#8735)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+4/-22)
BODY: ## Motivation ⏎  ⏎  ⏎ Fix incorrect fix in #8732. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fee0ab0fba  (L1, 2025-08-03, sha fee0ab0fba12, PR #8294)
TITLE: [CI] Ascend NPU CI enhancement (#8294)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+10/-2); .github/workflows/pr-test-npu.yml (+67/-3); scripts/npu_ci_install_dependency.sh (+36/-24); test/srt/run_suite.py (+8/-2); test/srt/test_ascend_attention_backend.py (+0/-62); test/srt/test_ascend_mla_backend.py (+0/-96); test/srt/test_ascend_mla_w8a8int8.py (+100/-0); test/srt/test_ascend_tp1_bf16.py (+96/-0); test/srt/test_ascend_tp2_bf16.py (+98/-0)
LABELS: ready-to-merge, npu
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ To enhance ascend npu ci by covering all the features we have already supported on sglang. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Add a new test case covering Ascend-specific W8A8-int8 quant method ⏎ - Add a new test case covering tp=2 scenario ⏎ - Merge attention backend test case into tp=1 bf16 scenario ⏎ - Merge attention mla backend test case into W8A8-int8 scenario with DeepSeek-V2-Lite model ⏎ - **_Fix a functional issue on Ascend NP …[truncated]

### L1-915140fd18  (L1, 2025-08-04, sha 915140fd18c9, PR #8552)
TITLE: [NVIDIA] Add Low Latency NVFP4 decode kernels from Flashinfer (#8552)
SOURCES: path_core, symbol_pickaxe, corpus:kernel-correctness-cases(introducing)
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+18/-7); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+173/-16); python/sglang/srt/layers/moe/utils.py (+16/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+260/-63); python/sglang/srt/managers/schedule_batch.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+25/-24); python/sglang/srt/models/glm4_moe.py (+2/-4); python/sglang/srt/server_args.py (+7/-0)
LABELS: high priority
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:b01eeb80f8 (fix PR 8779) || deep-study: introduced the defect fixed in case sglang:6d0646da11 (fix PR 8773)
BODY: ## Motivation ⏎  ⏎ Bring best low latency NVFP4 kernels for Blackwell MoE. Currently enabling DSR1.  ⏎  ⏎ ## Modifications ⏎  ⏎ Changing some weight preprocessing logic as well as exposing these kernels. Plus various piping to make it work.    ⏎  ⏎ ## Accuracy Test ⏎  ⏎ Ran accuracy tests, find description and repro below.  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎ Added below with repros.  ⏎  ⏎ ## Checklist

### L1-9bd4872a34  (L1, 2025-08-04, sha 9bd4872a343e, PR #8768)
TITLE: [bugfix] Fix typo in modelopt quant: 'FusedMoE' object has no attribute 'local_num_experts' (#8768)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ After https://github.com/sgl-project/sglang/pull/8552 was merged, I get this error `AttributeError: 'FusedMoE' object has no attribute 'local_num_experts'`. It should be `num_local_experts` instead.

### L1-fc8c8e5041  (L1, 2025-08-04, sha fc8c8e504156, PR #8762)
TITLE: Integrate triton_kernels in sgl-kernel (#8762)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+16/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6d0646da11  (L1, 2025-08-04, sha 6d0646da1145, PR #8773)
TITLE: [NVIDIA] Fix breakage of using trtllm-gen fp8 moe (#8773)
SOURCES: path_core, symbol_pickaxe, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-62); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+14/-1)
LABELS: bug, high priority
DEEP_STUDY: deep-study correctness case sglang:6d0646da11: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=#8552
BODY: This PR fixes a breakage caused by [this PR](https://github.com/sgl-project/sglang/pull/8552). ⏎  ⏎ cc. @kushanam

### L1-d4bf5a8524  (L1, 2025-08-04, sha d4bf5a852482, PR #8255)
TITLE: Support OCP MXFP4 quantization on AMD GPUs (#8255)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+2/-0); python/sglang/srt/layers/quantization/__init__.py (+13/-1); python/sglang/srt/layers/quantization/fp4.py (+822/-0); python/sglang/srt/layers/quantization/quark/__init__.py (+0/-0); python/sglang/srt/layers/quantization/quark/schemes/__init__.py (+6/-0); python/sglang/srt/layers/quantization/quark/schemes/quark_scheme.py (+55/-0); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py (+118/-0); python/sglang/srt/layers/quantization/quark/utils.py (+107/-0); python/sglang/srt/model_loader/weight_utils.py (+10/-0); python/sglang/srt/models/deepseek_v2.py (+14/-0); (+2 more)
BODY: Co-author: @fxmarty-amd  ⏎  ⏎ ## Motivation ⏎  ⏎ With the new instructions support on the new hardware platforms, AMD new GPUs can run the OCP MXFP4 computation in MoE and linear layers. Besides dynamic quantization from bf16/fp16 to mxfp4, this also supported the quark quantized mxfp4 static model.  ⏎  ⏎ ## Modifications ⏎  ⏎ This PR adds a new quantization scheme, fp4 to support ocp mxfp4 models. ⏎  ⏎ ## Accuracy check on Grok1 ⏎ SGLANG_USE_AITER=1 python …[truncated]

### L1-08f8f49016  (L1, 2025-08-04, sha 08f8f4901650, PR #8212)
TITLE: [CPU][sgl-kernel] biased_grouped_topk: fix correction_bias dtype to float32 (#8212)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: sgl-kernel/csrc/cpu/topk.cpp (+28/-25); sgl-kernel/csrc/cpu/common.h (+39/-0); sgl-kernel/csrc/cpu/vec.h (+19/-0); test/srt/cpu/test_topk.py (+8/-3)
LABELS: sgl-kernel, ready-to-merge, intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/pull/7825 has fixed the dtype of `e_score_correction_bias` to float32 instead of bfloat16. We need to fix the support in the topk kernel. ⏎ UT has been updated accordingly.

### L1-1ea94d3b92  (L1, 2025-08-04, sha 1ea94d3b926c, PR #8780)
TITLE: chore: upgrade flashinfer v0.2.9 (#8780)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b01eeb80f8  (L1, 2025-08-04, sha b01eeb80f840, PR #8779)
TITLE: [NVIDIA]Fix local_num_experts for EP (#8779)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+2/-1)
LABELS: bug, high priority
DEEP_STUDY: deep-study: this PR was reverted by PR 8797 (confirmed_revert, reason=unstated) || deep-study correctness case sglang:b01eeb80f8: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#8552
BODY: ## Motivation ⏎  ⏎ This PR fixes a bug in https://github.com/sgl-project/sglang/pull/8552. At create_weight, the num_experts and num_loca_experts should both be passed in for EP case. ⏎  ⏎ @kaixih @kushanam  ⏎  ⏎ cc. @kushanam ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5e91fed1c5  (L1, 2025-08-04, sha 5e91fed1c593, PR #8797)
TITLE: Revert "[NVIDIA]Fix local_num_experts for EP (#8779)" (#8797)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-2); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-2)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 8779 reason=unstated
BODY: This reverts commit b01eeb80f8406cba569af5deb40f394293f9950d. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8e8545caf6  (L1, 2025-08-05, sha 8e8545caf6e0, PR #8817)
TITLE: fix: update cmake (#8817)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+3/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4f4e0e4162  (L1, 2025-08-05, sha 4f4e0e4162ac, PR #8827)
TITLE: chore: upgrade flashinfer 0.2.10 (#8827)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c1d2061f97  (L1, 2025-08-05, sha c1d2061f97ae, PR #8824)
TITLE: Add initial support for gpt-oss (#8824)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+134/-8); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+178/-3); python/sglang/srt/layers/attention/triton_backend.py (+85/-14); python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+17/-0); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+36/-8); python/sglang/srt/layers/linear.py (+0/-5); python/sglang/srt/layers/quantization/fp8_utils.py (+29/-0); python/sglang/srt/layers/quantization/mxfp4_tensor.py (+133/-0); python/sglang/srt/layers/quantization/unquant.py (+52/-7); python/sglang/srt/managers/schedule_batch.py (+4/-2); (+2 more)
LABELS: high priority
BODY: Future progress will be tracked here: https://github.com/sgl-project/sglang/issues/8833 ⏎  ⏎ **This PR only works for FP8/BF16 ckpt. The FP8/BF16 ckpt has been uploaded to:** ⏎ `lmsys/gpt-oss-20b-bf16` and `lmsys/gpt-oss-120b-bf16` ⏎  ⏎ Install SGLang: ⏎ 1. local build: `pip install -e "python[all]"` ⏎ 2. normal install should also be fine (you might see version conflict complaints, but should be fine) ⏎  ⏎ Additional install for gpt-oss: ⏎ ``` ⏎ pip3 insta …[truncated]

### L1-3ae8e3ea8f  (L1, 2025-08-05, sha 3ae8e3ea8f32, PR #8836)
TITLE: chore: upgrade torch 2.8.0 (#8836)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+8/-8); .github/workflows/vllm-dependency-test.yml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-168033d5fb  (L1, 2025-08-06, sha 168033d5fb1e, PR #8843)
TITLE: Support mxfp4 for GPT-OSS (#8843)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+58/-6); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+25/-15); python/sglang/srt/layers/quantization/__init__.py (+12/-2); python/sglang/srt/layers/quantization/fp4.py (+28/-293); python/sglang/srt/layers/quantization/mxfp4.py (+443/-0); python/sglang/srt/layers/quantization/unquant.py (+2/-0); python/sglang/srt/models/gpt_oss.py (+209/-9); python/sglang/srt/server_args.py (+10/-0); python/sglang/srt/utils.py (+4/-0)
BODY: 

### L1-01c99a9959  (L1, 2025-08-06, sha 01c99a9959e0, PR #8872)
TITLE: chore: update Dockerfile (#8872)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+8/-5); .github/workflows/vllm-dependency-test.yml (+2/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-288ae41f7a  (L1, 2025-08-06, sha 288ae41f7ae9, PR #8811)
TITLE: [NVIDIA] Fix num_experts in modelopt_quant (#8811)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+2/-4)
DEEP_STUDY: deep-study correctness case sglang:288ae41f7a: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: A hotfix for https://github.com/sgl-project/sglang/pull/8779.  ⏎ The trtllm_fp4_block_scale_moe API checks for the routing_logits dim to be consistent with num_experts(global). But previously at create_weight, the num_experts is overwritten by num_local_experts.  ⏎ cc. @kushanam  @zhyncs  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎ With EP enabled: ⏎ Accuracy: 0.963 ⏎ Invalid: 0.000 ⏎ Latency: 697.093 s ⏎ Output throughput: 139. …[truncated]

### L1-4373df5525  (L1, 2025-08-06, sha 4373df55258e, PR #8847)
TITLE: add flashinfer mxfp4 (#8847)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+20/-2); python/sglang/srt/layers/quantization/mxfp4.py (+195/-19); python/sglang/srt/server_args.py (+15/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-47824c1488  (L1, 2025-08-07, sha 47824c14881b, PR #8898)
TITLE: [Perf] Auto enable best flashinfer mxfp4  kernel in b200 (#8898)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-4); python/sglang/srt/layers/quantization/mxfp4.py (+24/-27); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/models/gpt_oss.py (+9/-5); python/sglang/srt/server_args.py (+10/-12)
BODY: ### GPT-OSS-20B ⏎  ⏎ - SGLANG_USE_FLASHINFER_MXFP4_BF16_MOE=1 ⏎  ⏎ ```markdown ⏎ CUDA_VISIBLE_DEVICES=1 SGLANG_USE_FLASHINFER_MXFP4_BF16_MOE=1 python3 -m sglang.launch_server --model-path openai/gpt-oss-20b --tp-size 1 --port 30001 ⏎  ⏎ curl http://127.0.0.1:30001/flush_cache ⏎ python3 -m sglang.bench_serving --backend sglang-oai  --dataset-name random --random-input-len 512 --random-output-len 1024 --random-range-ratio 1 --num-prompts 20 --max-concurren …[truncated]

### L1-39fd178831  (L1, 2025-08-07, sha 39fd1788311c, PR #8720)
TITLE: refactor: Move scalar_types.py to sgl-kernel to avoid circular import (#8720)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: sgl-kernel/python/sgl_kernel/fused_moe.py (+1/-1); sgl-kernel/python/sgl_kernel/scalar_type.py (+352/-0); sgl-kernel/tests/test_marlin_repack.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ This PR addresses a Python circular import issue encountered in  https://github.com/sgl-project/sglang/pull/8112.  ⏎  ⏎ We believe that placing the `scalar_types`  inside the sgl-kernel will be better.  ⏎  ⏎ Please merge this PR first to ensure the dependent PR works as expected. ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-76915d68a8  (L1, 2025-08-07, sha 76915d68a8f8, PR #8950)
TITLE: Fix enable flashinfer mxfp4 moe  bf16 check (#8950)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+9/-8)
BODY: ## Motivation ⏎  ⏎ Fix running lmsys/gpt-oss-20b-bf16, which should bypass fused_moe with mxfp4 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0d1e27a0c5  (L1, 2025-08-08, sha 0d1e27a0c572, PR #8953)
TITLE: Better optimization log for gpt-oss model (#8953)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4.py (+5/-4); python/sglang/srt/server_args.py (+6/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1d24db8348  (L1, 2025-08-08, sha 1d24db834803, PR #8944)
TITLE: Expert Parallelism for GPT-OSS (#8944)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+6/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+101/-12); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-2); python/sglang/srt/layers/quantization/mxfp4.py (+80/-52); python/sglang/srt/layers/quantization/unquant.py (+9/-2); python/sglang/srt/models/gpt_oss.py (+54/-47); python/sglang/srt/server_args.py (+10/-4); python/sglang/srt/utils.py (+5/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ - What's in this PR: ⏎   - Enable GPT-OSS launch without triton-kernels ⏎   - Support expert parallelism for GPT-OSS ⏎  ⏎ Example: ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model openai/gpt-oss-120b --tp 4 --ep 2 ⏎ ``` ⏎  ⏎ - TODO: ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1132547496  (L1, 2025-08-08, sha 113254749659, PR #7657)
TITLE: Add ernie4.py for ERNIE-4.5  (#7657)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/supported_models/generative_models.md (+1/-0); python/sglang/srt/configs/model_config.py (+5/-0); python/sglang/srt/models/ernie4.py (+426/-0); python/sglang/srt/models/ernie4_eagle.py (+203/-0)
LABELS: high priority, new-model
ISSUES: #7668 [Feature] Ernie4.5 and Ernie4.5MoE Model Support
BODY: ## Motivation ⏎  ⏎ - To close: https://github.com/sgl-project/sglang/issues/7668 ⏎ * Support [ERNIE-4.5 models](https://huggingface.co/collections/baidu/ernie-45-6861cd4c9be84540645f35c9) ⏎  ⏎ ## Modifications ⏎  ⏎ * Add ernie4.py for ERNIE-4.5 (Ernie4_5_ForCausalLM and Ernie4_5_MoeForCausalLM) ⏎  ⏎ ## Benchmark ⏎  ⏎ GSM8K (200) Benchmark (using sglang/benchmark/gsm8k/bench_sglang.py) ⏎  ⏎ | Model                                          | GSM8K (200) | ⏎ |:-- …[truncated]

### L1-b4c9f38a76  (L1, 2025-08-08, sha b4c9f38a76bd, PR #8955)
TITLE: [NVIDIA] Fix missing `get_col_major_tma_aligned_tensor` for Blackwell deepgemm in EpMoE (#8955)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+49/-4)
BODY: EpMoE on Blackwell with deepgemm currently fails with the following error due to the absence of get_col_major_tma_aligned_tensor: ⏎  ⏎ ```  ⏎  File "/sgl-workspace/sglang/python/sglang/srt/layers/moe/ep_moe/layer.py", line 204, in forward_deepgemm ⏎     deep_gemm_wrapper.get_col_major_tma_aligned_tensor(gateup_input_scale), ⏎     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ AttributeError: module 'sglang.srt.layers.quantization.deep_gemm_wrappe …[truncated]

### L1-54ea57f245  (L1, 2025-08-08, sha 54ea57f2451c, PR #8957)
TITLE: chore: bump sgl-kernel v0.3.3 (#8957)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-41357e511b  (L1, 2025-08-08, sha 41357e511b15, PR #8958)
TITLE: chore: update flashinfer (#8958)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4e7f025219  (L1, 2025-08-08, sha 4e7f02521920, PR #8772)
TITLE: chore(gb200): update to CUDA 12.9 and improve build process (#8772)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+19/-13); docker/Dockerfile.gb200 (+36/-24); .github/workflows/release-docker-gb200.yml (+3/-3); .github/workflows/release-whl-kernel-aarch64.yml (+7/-7); python/sglang/srt/entrypoints/engine.py (+1/-1); sgl-kernel/build.sh (+7/-0); sgl-kernel/rename_wheels.sh (+13/-2)
BODY: This PR does a couple different things ⏎ 1. Breaks apart our deepep/nvshmem installation for cleaner dockerfile practices ⏎ 2. Change GB200 dockerfile to cuda 12.9 because there doesnt seem to be torch 2.8+cu128 for aarch64 ⏎ 3. Denotes the proper arm runner so the `sgl-kernel` can be built for aarch ⏎ 4. Makes the wheel rename script more robust so we can properly rename for cuda 129/128 and cp39/310 ⏎  ⏎ SGL kernel for aarch64 has been officially rel …[truncated]

### L1-591c232f7c  (L1, 2025-08-08, sha 591c232f7c9e, PR #8770)
TITLE: [1/2][resubmit] sgl-kernel: Fuse routed scaling factor into moe_fused_gate (select_experts) (#8770)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.fused_gate
FILES: python/sglang/srt/layers/moe/topk.py (+24/-0); sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/csrc/moe/moe_fused_gate.cu (+20/-7); sgl-kernel/include/sgl_kernel_ops.h (+2/-1); sgl-kernel/python/sgl_kernel/moe.py (+9/-2); sgl-kernel/tests/test_moe_fused_gate.py (+6/-1)
DEEP_STUDY: deep-study: this PR was reverted by PR 9035 (confirmed_revert, reason=unstated)
BODY: ## Motivation ⏎  ⏎ Resubmit of https://github.com/sgl-project/sglang/pull/8364 which was missing changes required for the unit test. ⏎  ⏎ https://github.com/sgl-project/sglang/pull/8690 will enable using this fusion for deepseek. ⏎  ⏎ Prefill: ⏎ 10.46% speedup at BS 1 ⏎ 1.86% speedup at BS 128 ⏎ Decode: ⏎ 1.22% speedup at BS1 ⏎ 0.26% speedup at BS128

### L1-706bd69cc5  (L1, 2025-08-08, sha 706bd69cc58a, PR #8983)
TITLE: Clean up server_args.py to have a dedicated function for model specific adjustments (#8983)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: .github/workflows/execute-notebook.yml (+1/-4); .github/workflows/pr-test-pd-router.yml (+1/-1); .github/workflows/pr-test.yml (+38/-34); .github/workflows/vllm-dependency-test.yml (+4/-3); .gitmodules (+0/-0); README.md (+5/-4); docs/backend/server_arguments.md (+2/-2); docs/index.rst (+3/-3); python/pyproject.toml (+4/-5); python/sglang/srt/configs/model_config.py (+5/-7); (+14 more)
BODY: 

### L1-3f2e315f6e  (L1, 2025-08-09, sha 3f2e315f6e5b, PR #8962)
TITLE: optimize: reduce shulffle and quantization overhead in cutlass_moe sm90 (#8962)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+11/-16); python/sglang/srt/layers/quantization/fp8_kernel.py (+59/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ In https://github.com/sgl-project/sglang/pull/8678, `per_token_group_quant_fp8_hopper_moe_mn_major` is called after `shuffle_rows`. This approach shuffles hidden_states in bf16 and increases the tokens to quantize by top-k fold,  which is inefficient. We optimize this workflow by first quantizing the hidden_states prior to shuffling, and then utilizing the `per_group_transpose` Triton kernel to handle scale transposition. ⏎ ## …[truncated]

### L1-f29aba8c6e  (L1, 2025-08-09, sha f29aba8c6e38, PR #8798)
TITLE: Support glm4.1v and glm4.5v (#8798)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: benchmark/mmmu/README.md (+12/-0); benchmark/mmmu/bench_sglang.py (+23/-2); benchmark/mmmu/eval_utils.py (+7/-0); docs/supported_models/multimodal_language_models.md (+1/-0); python/sglang/lang/chat_template.py (+18/-0); python/sglang/srt/configs/model_config.py (+2/-0); python/sglang/srt/function_call/ebnf_composer.py (+1/-0); python/sglang/srt/function_call/glm4_moe_detector.py (+1/-1); python/sglang/srt/jinja_template_utils.py (+6/-0); python/sglang/srt/layers/rotary_embedding.py (+230/-1); (+11 more)
LABELS: high priority, Multi-modal, new-model
ISSUES: #7993 [Feature] Any plan to support glm-4.1v-thinking
BODY: ## Motivation ⏎  ⏎ Close https://github.com/sgl-project/sglang/issues/7993.  ⏎  ⏎ Adapted from https://github.com/sgl-project/sglang/pull/8015 and depend on https://github.com/huggingface/transformers/pull/39805 ⏎  ⏎ Covers https://github.com/sgl-project/sglang/pull/8948 ⏎  ⏎ ## Modifications ⏎  ⏎ This pull request introduces support for the GLM-4V model within the sglang framework. It includes the necessary model files, configuration adjustments, and a ne …[truncated]

### L1-326a901df4  (L1, 2025-08-09, sha 326a901df448, PR #8998)
TITLE: chore: upgrade sgl-kernel 0.3.3 (#8998)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-137e75daa1  (L1, 2025-08-09, sha 137e75daa1d3, PR #8355)
TITLE: [Feature] Optimize DeepSeek's DeepEP on Ascend NPU (#8355)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.topk_py, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+60/-2); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+61/-24); python/sglang/srt/layers/moe/topk.py (+2/-1); python/sglang/srt/layers/quantization/w8a8_int8.py (+39/-31); python/sglang/srt/distributed/parallel_state.py (+4/-2); python/sglang/srt/layers/attention/ascend_backend.py (+3/-0); python/sglang/srt/layers/rotary_embedding.py (+41/-1)
LABELS: high priority, ready-to-merge
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Following our roadmap #8004, this is the first pr to support deepep expert parallelism and speed up with some fusion kernels. Now it is possible to activate `--enable-deepep-moe` with `--deepep-mode low_latency` only under pd disaggregation scenario on decode nodes. However, we only support W8A8 int8 quant method currently. ⏎  ⏎ More info about our DeepEP-compatible kernels, check out our roadmap in [sgl-kernel-npu](https://git …[truncated]

### L1-20cfc5a251  (L1, 2025-08-09, sha 20cfc5a251c1, PR #9010)
TITLE: [perf] add kimi-k2 b200 fused moe config (#9010)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9a44b643c6  (L1, 2025-08-09, sha 9a44b643c67d, PR #9012)
TITLE: Fix CI (#9012)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-0); .github/workflows/vllm-dependency-test.yml (+9/-3); python/sglang/srt/entrypoints/engine.py (+2/-2); python/sglang/srt/entrypoints/openai/tool_server.py (+4/-3); python/sglang/srt/layers/quantization/__init__.py (+4/-2); python/sglang/srt/layers/quantization/modelopt_quant.py (+2/-7); python/sglang/srt/managers/multimodal_processor.py (+1/-1); python/sglang/srt/models/registry.py (+1/-1); test/srt/test_utils_update_weights.py (+0/-1)
BODY: fix vllm dependency tests and remove some warnings

### L1-ef48d5547e  (L1, 2025-08-09, sha ef48d5547ec9, PR #9013)
TITLE: Fix CI (#9013)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-0); .github/workflows/cancel-all-pending-pr-test-runs.yml (+25/-11); .github/workflows/execute-notebook.yml (+1/-1); .github/workflows/pr-test-xeon.yml (+1/-1); .github/workflows/pr-test.yml (+7/-3); test/srt/run_suite.py (+64/-48); test/srt/test_bench_serving.py (+1/-1); test/srt/test_gpt_oss_1gpu.py (+6/-6); test/srt/test_gpt_oss_common.py (+13/-4)
BODY: - Fix the threshold in CI ⏎ - Reorganize the CI pipeline ⏎ - Fix the CPU CI

### L1-3817a37d87  (L1, 2025-08-09, sha 3817a37d8761, PR #9019)
TITLE: [router] upgrade to latest sgl kernel for router ci (#9019)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-pd-router.yml (+1/-1)
LABELS: router, ci
BODY: ## Motivation ⏎  ⏎ upgrade to latest router kernel ⏎  ⏎ ## Checklist

### L1-2c7f01bc89  (L1, 2025-08-10, sha 2c7f01bc899a, PR #9027)
TITLE: Reorganize CI and test files (#9027)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: scripts/ci/ci_install_deepep.sh (+1/-1); .github/workflows/execute-notebook.yml (+1/-1); .github/workflows/experiment-runner.yml (+1/-1); .github/workflows/nightly-test-amd.yml (+3/-3); .github/workflows/nightly-test.yml (+1/-1); .github/workflows/pr-benchmark-rust.yml (+2/-2); .github/workflows/pr-test-amd.yml (+42/-42); .github/workflows/pr-test-npu.yml (+3/-3); .github/workflows/pr-test-pd-router.yml (+4/-4); .github/workflows/pr-test-rust.yml (+2/-2); (+56 more)
BODY: - move ci related scripts under `scripts`  to `scripts/ci` ⏎ - start to introduce more subfolders under `test/srt`

### L1-dd949ace23  (L1, 2025-08-10, sha dd949ace23d6, PR #9035)
TITLE: Revert "[1/2][resubmit] sgl-kernel: Fuse routed scaling factor into m… (#9035)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.fused_gate
FILES: python/sglang/srt/layers/moe/topk.py (+0/-24); sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/csrc/moe/moe_fused_gate.cu (+7/-20); sgl-kernel/include/sgl_kernel_ops.h (+1/-2); sgl-kernel/python/sgl_kernel/moe.py (+2/-9); sgl-kernel/tests/test_moe_fused_gate.py (+1/-6)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 8770 reason=unstated
BODY: …oe_fused_gate (select_experts) (#8770)" ⏎  ⏎ This reverts commit 591c232f7c9ef959c8620567789fe9d918a58bb4. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-dd001a5477  (L1, 2025-08-10, sha dd001a54772b, PR #9036)
TITLE: chore: upgrade flashinfer 0.2.11 (#9036)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
DEEP_STUDY: deep-study: this PR was reverted by PR 9057 (confirmed_revert, reason=crash_or_hang)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-067068f271  (L1, 2025-08-10, sha 067068f27173, PR #8997)
TITLE: [router] regular router circuit breaker (#8997)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+22/-0); package-lock.json (+6/-0); sgl-router/README.md (+33/-0); sgl-router/py_src/sglang_router/launch_router.py (+94/-0); sgl-router/py_src/sglang_router/launch_server.py (+1/-0); sgl-router/py_test/test_launch_router.py (+11/-0); sgl-router/py_test/test_launch_server.py (+25/-0); sgl-router/src/config/types.rs (+43/-0); sgl-router/src/config/validation.rs (+79/-0); sgl-router/src/core/circuit_breaker.rs (+15/-5); (+12 more)
LABELS: enhancement, router
BODY: ### Overview ⏎ - Implemented circuit-breaker-aware retries with exponential backoff and jitter for the regular router. ⏎ - Centralized retry orchestration in a reusable executor. ⏎ - Added metrics and unit tests. ⏎ - Part of #8917  ⏎  ⏎ ### What changed ⏎  ⏎ - Config ⏎   - Added `jitter_factor` to `RetryConfig` (defaults to 0.1) in `sgl-router/src/config/types.rs`. ⏎  ⏎ - Core retry API (`sgl-router/src/core/retry.rs`) ⏎   - `BackoffCalculator`: computes exp …[truncated]

### L1-84cb449eec  (L1, 2025-08-11, sha 84cb449eeccf, PR #9057)
TITLE: Revert "chore: upgrade flashinfer 0.2.11 (#9036)" (#9057)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 9036 reason=crash_or_hang
BODY: This reverts commit dd001a54772b3f164f8ec359f77168109262bb01. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f4ae50e97c  (L1, 2025-08-11, sha f4ae50e97ce5, PR #)
TITLE: fix: use flashinfer v0.2.11.post1
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml
PR_RECORD: missing (use git/gh if needed)
BODY: 
