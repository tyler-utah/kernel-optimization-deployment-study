### L1-b819381fec  (L1, 2025-06-05, sha b819381feca4, PR #6838)
TITLE: AITER backend extension and workload optimizations (#6838)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.triton.fused_moe; AITER flag/workload changes alter fused_moe_triton AMD backend behavior.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-3); .github/workflows/pr-test-amd.yml (+1/-1); docs/references/environment_variables.md (+1/-1); python/sglang/srt/layers/attention/aiter_backend.py (+488/-124); python/sglang/srt/layers/layernorm.py (+27/-2); python/sglang/srt/layers/quantization/fp8.py (+16/-16); python/sglang/srt/layers/quantization/fp8_utils.py (+6/-10); python/sglang/srt/model_executor/model_runner.py (+10/-0); python/sglang/srt/models/deepseek_v2.py (+17/-0); scripts/amd_ci_exec.sh (+13/-7); test/srt/test_nightly_gsm8k_eval_amd.py (+1/-1)
LABELS: high priority, aiter
BODY: `Co-author: @kkHuang-amd ` ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎ - DeepSeek optimization ⏎ - 1 simple flag: `SGLANG_USE_AITER` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-61ce91ed28  (L1, 2025-06-06, sha 61ce91ed2822, PR #6934)
TITLE: Tiny support customize DeepEP max dispatch tokens per rank (#6934)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L1.ep.deepep_dispatcher; DeepEP dispatcher supports custom max dispatch tokens per rank.
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+4/-2)
BODY: ## Motivation ⏎  ⏎ qiaolin as well as some other cases I meet needs it ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2f715f51cc  (L1, 2025-06-07, sha 2f715f51cc41, PR #6944)
TITLE: Minor compile fused topk (#6944)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L1.routing.topk_py; topk.py adjusts fused topk for torch compile compatibility.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+17/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8b5f83ed3b  (L1, 2025-06-07, sha 8b5f83ed3b7d, PR #6369)
TITLE: reduce torch.zeros overhead in moe align block size kernel (#6369)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: optimize; artifacts=L1.align.cuda_aot,L1.triton.moe_align; moe_align path reduces torch.zeros overhead around align buffers.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.align.cuda_aot
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+58/-6); sgl-kernel/csrc/moe/moe_align_kernel.cu (+0/-2)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ origin: ⏎  ⏎ <img width="918" alt="图片" src="https://github.com/user-attachments/assets/ec8bcd33-c14a-41a8-a548-0e32f1c955c0" /> ⏎  ⏎ pr: ⏎  ⏎ <img width="475" alt="图片" src="https://github.com/user-attachments/assets/177625ea-aa45-41fa-b93b-fd33868b000c" /> ⏎  ⏎ ## Benchmark ⏎  ⏎ main: ⏎  ⏎  ⏎ ```shell ⏎ fused-moe-performance: ⏎      batch_size  vllm_fused_moe_triton  sglang_fused_moe_triton ⏎ 0           1.0               0.355232                 0.306560 ⏎ 1           2.0               0.345664                 0.303840 ⏎ 2           3.0               0.343264                 0.299968 ⏎ 3           4.0               0.345696                 0.304192 ⏎ 4           5.0               0.347824                 0.305152 ⏎ 5           6.0               0.347136                 0.304768 ⏎ 6           7.0               0.347840                 0.308256 ⏎ 7           8.0               0.349984                 0.308576 ⏎ 8           9.0               0.349504                 0.308288 ⏎ 9          10.0               0.350080                 0.308576 ⏎ 10         11.0               0.350560                 0.308112 ⏎ 11         12.0               0.355456                 0.307680 ⏎ 12         13.0               0.366944                 0.309232 ⏎ 13         14.0               0.378208                 0.306624 ⏎ 14         15.0               0.381824                 0.309808 ⏎ 15         16.0               0.469888                 0.306464 ⏎ 16         17.0               0.475616                 0.305440 ⏎ ``` ⏎  ⏎  ⏎ pr: ⏎  ⏎  ⏎ ```shell ⏎      batch_size  vllm_fused_moe_triton  sglang_fused_moe_triton ⏎ 0           1.0               0.344144                 0.283072 ⏎ 1           2.0               0.340368                 0.280368 ⏎ 2           3.0               0.338864                 0.277792 ⏎ 3           4.0               0.342304                 0.283440 ⏎ 4           5.0               0.347936                 0.284928 ⏎ 5           6.0               0.340288                 0.281808 ⏎ 6           7.0               0.341248                 0.282720 ⏎ 7           8.0               0.347392                 0.284640 ⏎ 8           9.0               0.342144                 0.284608 ⏎ 9          10.0               0.350176                 0.287008 ⏎ 10         11.0               0.349024                 0.284832 ⏎ 11         12.0               0.348352                 0.286816 ⏎ 12         13.0               0.366480                 0.285312 ⏎ 13         14.0               0.378784                 0.284448 ⏎ 14         15.0               0.380800                 0.286848 ⏎ 15         16.0               …[truncated]

### L1-515ef4facb  (L1, 2025-06-07, sha 515ef4facbc8, PR #6220)
TITLE: Fuse routed scaling factor in topk_reduce kernel (#6220)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts(reverted)
STAGE1: optimize; artifacts=L1.triton.fused_moe; Fuses routed_scaling_factor into topk_reduce/fused_moe path.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+124/-8); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-0); python/sglang/srt/layers/quantization/blockwise_int8.py (+1/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+1/-0); python/sglang/srt/layers/quantization/fp8.py (+1/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+1/-0); python/sglang/srt/layers/quantization/w8a8_fp8.py (+1/-0); python/sglang/srt/layers/quantization/w8a8_int8.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+1/-1); benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py (+199/-0)
DEEP_STUDY: deep-study: this PR was reverted by PR 6968 (confirmed_revert, reason=unstated)
PERF_LINES: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [00:42<00:00, 31.11it/s] | Latency: 45.708 s | Output throughput: 3127.153 token/s
BODY: ## Motivation ⏎  ⏎ ### gsm8k acc in h200 ⏎  ⏎ ```shell ⏎ ➜  sglang git:(fuse_routed_scaling_factor_in_deepseek) ✗ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8  ⏎  ⏎ 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [00:42<00:00, 31.11it/s] ⏎ Accuracy: 0.947 ⏎ Invalid: 0.000 ⏎ Latency: 45.708 s ⏎ Output throughput: 3127.153 token/s ⏎ ``` ⏎  ⏎ ### benchmark in H200 ⏎  ⏎ command: ⏎  ⏎ ```shell ⏎ python3 -m sglang.launch_server --model /DeepSeek-V3 --tp 8 --trust-remote-code --enable-dp-attention --dp-size 8  ⏎  ⏎ # 256, 4k ⏎ python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 100 --request-rate 10 --dataset-name random --random-input-len 256 --random-output-len 4096 --random-range-ratio 1 --warmup-requests 5 ⏎  ⏎ # Random 1k, 2k ⏎ python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 100 --request-rate 10 --dataset-name random --random-input-len 1000 --random-output-len 2000 --random-range-ratio 1 --warmup-requests 5 ⏎  ⏎ # Random 5k, 1k ⏎ python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 100 --request-rate 10 --dataset-name random --random-input-len 5000 --random-output-len 1000 --random-range-ratio 1  --warmup-requests 5 ⏎  ⏎ # Random 10k, 500 ⏎ python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 100 --request-rate 10 --dataset-name random --random-input-len 10000 --random-output-len 500 --random-range-ratio 1 --warmup-requests 5 ⏎  ⏎  ⏎ ``` ⏎  ⏎ <img width="752" alt="图片" src="https://github.com/user-attachments/assets/a0d4a2bc-5da4-4d7a-81dd-906fe231f3c3" /> ⏎  ⏎  ⏎ ### Micro benchmark in h200 ⏎  ⏎ micro benchmark in h200: ⏎  ⏎ ```shell ⏎ Running correctness verification... ⏎ ✅ All implementations match ⏎  ⏎ Running performance benchmark... ⏎ sum_scaled_performance: ⏎     num_tokens    Original  TorchCompile  TritonKernel ⏎ 0          1.0    9.376000      5.600000      8.896000 ⏎ 1          2.0    9.888000      5.664000      8.960000 ⏎ 2          4.0   10.000000      5.824000      9.056000 ⏎ 3          8.0   10.112000      6.272000      9.248000 ⏎ 4         16.0   10.656000      7.456000      9.568000 ⏎ 5         32.0   10.912000      9.056000      9.920000 ⏎ 6         64.0   11.776000     12.704000     10.400000 ⏎ 7        128.0   13.440000     20.000000     10.944000 ⏎ 8        256.0   18.432001     34.559999     13.312000 ⏎ 9        512.0   28.640000     63.936003     18.304000 ⏎ 10      1024.0   46.592001    122.047998     30.400001 ⏎ 11      2048.0   79.648003    238.655999     50.271999 ⏎ 12      4096.0  144.927993    471.055984     …[truncated]

### L1-3e56f557fd  (L1, 2025-06-07, sha 3e56f557fdd1, PR #6916)
TITLE: Add a CUDA kernel for fusing mapping and weighted sum for MoE. (#6916)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.cutlass.fp8_blockwise,L1.cutlass.adapters; CUTLASS MoE adds fusion kernel for mapping and weighted sum.
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+6/-5); sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+6/-6); sgl-kernel/csrc/moe/prepare_moe_input.cu (+114/-0); sgl-kernel/python/sgl_kernel/moe.py (+11/-0); sgl-kernel/csrc/common_extension.cc (+2/-1); sgl-kernel/include/sgl_kernel_ops.h (+6/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0)
PERF_LINES: Cutlass fused_experts time: 0.069 ms (median) [0.069 - 0.069] | Triton  fused_experts time: 0.128 ms (median) [0.128 - 0.128] | Cutlass fused_experts time: 0.107 ms (median) [0.107 - 0.107] | Triton  fused_experts time: 0.186 ms (median) [0.186 - 0.186] | Cutlass fused_experts time: 0.297 ms (median) [0.296 - 0.297] | Triton  fused_experts time: 0.334 ms (median) [0.333 - 0.337] | Cutlass fused_ex
BODY: ## Motivation ⏎  ⏎ A fusion kernel that improves the overall CUTLASS MOE layer perf for B200. ⏎ Fuses this single line:  ⏎ `(c2[c_map].view(m, topk, k) * topk_weights.view(m, topk, 1).to(out_dtype)).sum(dim=1)` ⏎  ⏎ ## Modifications ⏎ Previous perf: https://github.com/sgl-project/sglang/pull/5694 ⏎ After the change ⏎ ``` ⏎ --- Batch Size: 1 --- ⏎ Config: E=256, topk=8, H=7168, I_shard=512, dtype=torch.bfloat16, block_shape=[128, 128] ⏎ Warming up... ⏎ Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /usr/local/lib/python3.12/dist-packages/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=256,device_name=NVIDIA_Graphics_Device,dtype=fp8_w8a8,block_shape=[128, 128].json, you can create them with https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton ⏎ Benchmarking Cutlass fused_experts... ⏎ Benchmarking Triton fused_experts... ⏎ Cutlass fused_experts time: 0.069 ms (median) [0.069 - 0.069] ⏎ Triton  fused_experts time: 0.128 ms (median) [0.128 - 0.128] ⏎  ⏎ --- Batch Size: 4 --- ⏎ Config: E=256, topk=8, H=7168, I_shard=512, dtype=torch.bfloat16, block_shape=[128, 128] ⏎ Warming up... ⏎ Benchmarking Cutlass fused_experts... ⏎ Benchmarking Triton fused_experts... ⏎ Cutlass fused_experts time: 0.107 ms (median) [0.107 - 0.107] ⏎ Triton  fused_experts time: 0.186 ms (median) [0.186 - 0.186] ⏎  ⏎ --- Batch Size: 16 --- ⏎ Config: E=256, topk=8, H=7168, I_shard=512, dtype=torch.bfloat16, block_shape=[128, 128] ⏎ Warming up... ⏎ Benchmarking Cutlass fused_experts... ⏎ Benchmarking Triton fused_experts... ⏎ Cutlass fused_experts time: 0.297 ms (median) [0.296 - 0.297] ⏎ Triton  fused_experts time: 0.334 ms (median) [0.333 - 0.337] ⏎  ⏎ --- Batch Size: 32 --- ⏎ Config: E=256, topk=8, H=7168, I_shard=512, dtype=torch.bfloat16, block_shape=[128, 128] ⏎ Warming up... ⏎ Benchmarking Cutlass fused_experts... ⏎ Benchmarking Triton fused_experts... ⏎ Cutlass fused_experts time: 0.465 ms (median) [0.460 - 0.487] ⏎ Triton  fused_experts time: 0.495 ms (median) [0.476 - 0.526] ⏎  ⏎ --- Batch Size: 64 --- ⏎ Config: E=256, topk=8, H=7168, I_shard=512, dtype=torch.bfloat16, block_shape=[128, 128] ⏎ Warming up... ⏎ Benchmarking Cutlass fused_experts... ⏎ Benchmarking Triton fused_experts... ⏎ Cutlass fused_experts time: 0.539 ms (median) [0.536 - 0.624] ⏎ Triton  fused_experts time: 0.630 ms (median) [0.603 - 0.654] ⏎  ⏎ --- Batch Size: 128 --- ⏎ Config: E=256, topk=8, H=7168, I_shard=512, dtype=torch.bfloat16, block_shape=[128, 128] ⏎ Warming up... ⏎ Benchmarking Cutlass fused_experts... ⏎ Benchmarking Triton fused_experts... ⏎ Cutlass fused_experts time: 0.6 …[truncated]

### L1-c2c4f57f63  (L1, 2025-06-07, sha c2c4f57f6311, PR #6853)
TITLE: [DeepseekR1-FP4] Add Support for nvidia/DeepSeekR1-FP4 model (#6853)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L1.triton.fused_moe; fused_moe_triton layer adds DeepSeek R1 FP4 support.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+19/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+334/-7); python/sglang/srt/models/deepseek_v2.py (+33/-5)
LABELS: high priority
BODY: ## Motivation ⏎ Adds Model support for DeepSeek R1 FP4 Model. (Functional enablement - kernel optimizations underway) ⏎  ⏎ ## Modifications ⏎ Adds ModelOptFP4FusedMoEMethod and CutlassMoEParams dataclass to initialize the parameters required by Cutlass MoE methods. ⏎  ⏎ ## Checklist

### L1-1fb76ebb93  (L1, 2025-06-07, sha 1fb76ebb9389, PR #6968)
TITLE: Revert "Fuse routed scaling factor in topk_reduce kernel (#6220)" (#6968)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts
STAGE1: revert; artifacts=L1.triton.fused_moe; Reverts routed scaling fusion in topk_reduce/fused_moe path.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+8/-124); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-1); python/sglang/srt/layers/quantization/blockwise_int8.py (+0/-1); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+0/-1); python/sglang/srt/layers/quantization/fp8.py (+0/-1); python/sglang/srt/layers/quantization/moe_wna16.py (+0/-1); python/sglang/srt/layers/quantization/w8a8_fp8.py (+0/-1); python/sglang/srt/layers/quantization/w8a8_int8.py (+0/-1); python/sglang/srt/models/deepseek_v2.py (+1/-1); benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py (+0/-199)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 6220 reason=unstated
BODY: This reverts commit 515ef4facbc89cd7c093c198386a8817fce856d6. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3712abfaf9  (L1, 2025-06-08, sha 3712abfaf920, PR #6970)
TITLE: Fuse routed scaling factor in deepseek (#6970)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: reland; artifacts=L1.triton.fused_moe; Relands routed_scaling_factor fusion in DeepSeek fused_moe path.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+130/-14); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-0); python/sglang/srt/layers/quantization/blockwise_int8.py (+1/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+1/-0); python/sglang/srt/layers/quantization/fp8.py (+1/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+1/-0); python/sglang/srt/layers/quantization/w8a8_fp8.py (+1/-0); python/sglang/srt/layers/quantization/w8a8_int8.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+2/-1); benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py (+199/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-56ccd3c22c  (L1, 2025-06-09, sha 56ccd3c22c71, PR #6958)
TITLE: chore: upgrade flashinfer v0.2.6.post1 jit (#6958)
SOURCES: path_core, dependency_pin
STAGE1: integrate; artifacts=L1.upstream.flashinfer_moe,L1.triton.fused_moe; FlashInfer JIT upgrade plus fused_moe config/layer support for GB200.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+8/-6); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-0); .github/workflows/vllm-dependency-test.yml (+1/-1); lmms-eval (+1/-0); python/sglang/srt/entrypoints/engine.py (+2/-2); python/sglang/srt/layers/multimodal.py (+3/-3); python/sglang/srt/layers/quantization/__init__.py (+2/-2); python/sglang/test/test_utils.py (+0/-1); scripts/ci_install_dependency.sh (+12/-3); test/srt/run_suite.py (+2/-2); test/srt/test_bench_serving.py (+2/-2); test/srt/test_srt_engine.py (+2/-2); test/srt/test_vlm_input_format.py (+7/-3)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Integrate latest FlashInfer into SGLang for GB200 NVL72 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a979daac3b  (L1, 2025-06-09, sha a979daac3b60, PR #7013)
TITLE: Fallback to lower triton version for unfound fused moe configs (#7013)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: change_default; artifacts=L1.triton.fused_moe; fused_moe config lookup falls back to older Triton-version configs.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+21/-3)
BODY: ## Motivation ⏎  ⏎ Followup of #5955 ⏎ After upgrading to torch 2.7, the new version of triton (3.3.1) lacks most of the configs tuned before. ⏎ This PR can let the fused moe use config with older triton version, when current triton version doesn't include tuned config. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-81372f3bef  (L1, 2025-06-09, sha 81372f3bef65, PR #7029)
TITLE: Fix fused_moe triton configs (#7029)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_build_dependency; artifacts=L1.triton.fused_moe; pyproject package-data fix ensures fused_moe_triton configs are delivered.
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎ fix package-data for fused_moe triton configs. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fcde67b016  (L1, 2025-06-10, sha fcde67b016eb, PR #6833)
TITLE: CPU: map changes from developing branch in sgl-kernel (#6833)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.hardware.cpu_npu_musa; CPU sgl-kernel maps topk API changes from developing branch.
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: sgl-kernel/csrc/cpu/moe.cpp (+13/-4); sgl-kernel/csrc/cpu/moe_fp8.cpp (+19/-17); sgl-kernel/csrc/cpu/topk.cpp (+34/-2); sgl-kernel/csrc/cpu/decode.cpp (+503/-145); sgl-kernel/csrc/cpu/extend.cpp (+107/-9); sgl-kernel/csrc/cpu/gemm.h (+4/-0); sgl-kernel/csrc/cpu/gemm_fp8.cpp (+88/-91); sgl-kernel/csrc/cpu/interface.cpp (+2/-2); sgl-kernel/csrc/cpu/norm.cpp (+10/-4); sgl-kernel/csrc/cpu/qkv_proj.cpp (+79/-3); sgl-kernel/csrc/cpu/shm.cpp (+18/-11); sgl-kernel/csrc/cpu/shm.h (+1/-1); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+40/-4); sgl-kernel/csrc/cpu/vec.h (+115/-1); test/srt/cpu/test_mla.py (+155/-0); test/srt/cpu/test_moe.py (+1/-1); test/srt/cpu/test_norm.py (+3/-2); test/srt/cpu/test_qkv_proj_with_rope.py (+95/-9); test/srt/cpu/test_topk.py (+12/-1); test/srt/cpu/utils.py (+8/-0)
LABELS: sgl-kernel, intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is to map changes from developing branch in sgl-kernel, including: ⏎  ⏎ - optimize decode perf by mapping algo from FlashMLA ⏎ - optimize fp8 performance. ⏎ - Add API of fuse q_a_proj and kv_a_proj ⏎ - Support non-contiguous input in norm ⏎ - Update topk API along with SGLang main branch ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-84727a5139  (L1, 2025-06-11, sha 84727a51396a, PR #6919)
TITLE: [sgl-kernel] Add cuda kernel for moe_ep_silu_and_mul (#6919)
SOURCES: path_core
STAGE1: introduce; artifacts=L1.ep.reorder_aot; Adds CUDA moe_ep_silu_and_mul kernel replacing Triton helper.
ARTIFACT_HINTS: L1.ep.reorder_aot
FILES: sgl-kernel/csrc/moe/ep_moe_silu_and_mul_kernel.cu (+115/-0); sgl-kernel/python/sgl_kernel/moe.py (+18/-0); sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/benchmark/bench_moe_silu_and_mul.py (+92/-0); sgl-kernel/csrc/common_extension.cc (+4/-0); sgl-kernel/include/sgl_kernel_ops.h (+8/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/tests/test_ep_moe_silu_and_mul_kernel.py (+142/-0)
LABELS: high priority, ready-to-merge
PERF_LINES: Rewrite moe_ep_silu_and_mul Triton kernel in CUDA. Gains 10% performance improvement.
BODY: ## Motivation ⏎ Rewrite moe_ep_silu_and_mul Triton kernel in CUDA. Gains 10% performance improvement. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-da47621ccc  (L1, 2025-06-13, sha da47621ccc4f, PR #7058)
TITLE: Minor speedup topk postprocessing (#7058)
SOURCES: path_core, symbol_pickaxe
STAGE1: optimize; artifacts=L1.routing.topk_py; topk.py postprocessing change gives reported speedup.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+16/-8)
LABELS: high priority
PERF_LINES: 1.4% speedup in my test case
BODY: (cherry picked from commit c234231e6d625a4273dda47f414560895f2be8ff) ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ 1.4% speedup in my test case ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b04df75acd  (L1, 2025-06-13, sha b04df75acdda, PR #7154)
TITLE: Fix DeepEP error in some environments (#7154)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.deepep_dispatcher; DeepEP dispatcher workaround fixes environment-specific DeepEP error.
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+2/-0)
BODY: ## Motivation ⏎  ⏎ Use with https://github.com/deepseek-ai/DeepEP/pull/193 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8b8f2e7463  (L1, 2025-06-13, sha 8b8f2e74630c, PR #7153)
TITLE: Support new DeepGEMM input format in silu_and_mul_masked_post_quant_fwd (#7153)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L1.ep.layer; EP MoE kernel accepts new DeepGEMM input format.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+5/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b4c41f7276  (L1, 2025-06-13, sha b4c41f7276e2, PR #7150)
TITLE: Refactor DeepGEMM integration (#7150)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.ep.layer,L1.ep.deepep_dispatcher; Refactors DeepGEMM integration across EP kernels/layer/dispatcher wrapper.
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+1/-5); python/sglang/srt/layers/moe/ep_moe/layer.py (+35/-32); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+5/-5); python/sglang/srt/layers/quantization/deep_gemm_wrapper/__init__.py (+1/-0); python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py (+22/-76); python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py (+26/-0); python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py (+95/-0); python/sglang/srt/layers/quantization/fp8_kernel.py (+6/-10); python/sglang/srt/layers/quantization/fp8_utils.py (+3/-4); python/sglang/srt/model_executor/model_runner.py (+6/-6); python/sglang/srt/models/deepseek_v2.py (+3/-7); python/sglang/srt/two_batch_overlap.py (+4/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-93cec4335f  (L1, 2025-06-13, sha 93cec4335fed, PR #7172)
TITLE: Support new DeepGEMM (#7172)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L1.ep.layer,L1.ep.deepep_dispatcher,L1.triton.fused_moe; Updates EP/fused_moe paths for new DeepGEMM integration.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+8/-1); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+2/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-4); python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py (+7/-1); python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py (+18/-8); python/sglang/srt/layers/quantization/fp8_kernel.py (+20/-3); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-349bb2c92a  (L1, 2025-06-14, sha 349bb2c92af4, PR #7198)
TITLE: Fix error when disabling new DeepGEMM (#7198)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.ep.deepep_dispatcher; DeepEP dispatcher fixes disabling new DeepGEMM mode.
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+4/-2); python/sglang/srt/models/deepseek_v2.py (+4/-1)
BODY: ## Motivation ⏎  ⏎ local test on 8xB200 passes gsm8k ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5ca07eed90  (L1, 2025-06-16, sha 5ca07eed9018, PR #7247)
TITLE: [fix] fix DeepGEMM blackwell input quant & ut & fix style and log (#7247)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.ep.layer,L1.ep.deepep_dispatcher; Fixes DeepGEMM Blackwell input quant path used by EP MoE.
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-2); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+2/-2); python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py (+6/-9); python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py (+7/-7); python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py (+3/-3); python/sglang/srt/layers/quantization/fp8_kernel.py (+4/-3); python/sglang/srt/layers/quantization/fp8_utils.py (+4/-3); python/sglang/srt/models/deepseek_v2.py (+4/-2); python/sglang/test/test_block_fp8.py (+1/-0); python/sglang/test/test_block_fp8_deep_gemm_blackwell.py (+252/-0)
LABELS: bug
PERF_LINES: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [01:05<00:00, 20.13it/s] | Latency: 66.251 s | Output throughput: 2047.444 token/s | 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [00:49<00:00, 26.71it/s] | Latency: 49.735 s | Output t
BODY: ## Motivation ⏎  ⏎  ⏎ acc ⏎ ``` ⏎ # launch ⏎ python3 -m sglang.launch_server --model /dev/shm/DeepSeek-V3-0324 --tp 8 ⏎  ⏎ # test ⏎ while true; do (python3 benchmark/gsm8k/bench_sglang.py --num-questions 1319 --parallel 1319); done ⏎ 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [01:05<00:00, 20.13it/s] ⏎ Accuracy: 0.933 ⏎ Invalid: 0.000 ⏎ Latency: 66.251 s ⏎ Output throughput: 2047.444 token/s ⏎ 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [00:49<00:00, 26.71it/s] ⏎ Accuracy: 0.933 ⏎ Invalid: 0.000 ⏎ Latency: 49.735 s ⏎ Output throughput: 2722.960 token/s ⏎ 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [00:53<00:00, 24.80it/s] ⏎ Accuracy: 0.935 ⏎ Invalid: 0.000 ⏎ Latency: 53.538 s ⏎ Output throughput: 2522.791 token/s ⏎ 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [00:47<00:00, 28.01it/s] ⏎ Accuracy: 0.939 ⏎ Invalid: 0.000 ⏎ Latency: 47.419 s ⏎ Output throughput: 2833.372 token/s ⏎ 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [00:48<00:00, 27.30it/s] ⏎ Accuracy: 0.935 ⏎ Invalid: 0.000 ⏎ Latency: 48.655 s ⏎ Output throughput: 2790.393 token/s ⏎ ``` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8e2363dc15  (L1, 2025-06-16, sha 8e2363dc15ea, PR #7125)
TITLE: fix amd EP MoE FP8 issue (#7125)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.layer; AMD EP MoE FP8 path adds dtype transfer fix.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+30/-0)
PERF_LINES: 100%|██████████████████████████████████████████████████████████████| 1319/1319 [03:52<00:00,  5.66it/s] | Latency: 233.496 s | Output throughput: 571.187 token/s
BODY: ## Motivation ⏎  ⏎ To fix AMD EP MoE FP8 issue. See https://github.com/sgl-project/sglang/issues/7055. ⏎  ⏎ ## Modifications ⏎  ⏎ Add dtype transfer in EP path. ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Accuracy check ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server \ ⏎     --trust-remote-code \ ⏎     --chunked-prefill-size 131072 \ ⏎     --disable-cuda-graph \ ⏎     --disable-custom-all-reduce \ ⏎     --tp-size 8 \ ⏎     --enable-ep-moe \ ⏎     --model deepseek-ai/DeepSeek-V3 ⏎ ``` ⏎  ⏎ ``` ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 ⏎ 100%|██████████████████████████████████████████████████████████████| 1319/1319 [03:52<00:00,  5.66it/s] ⏎ Accuracy: 0.944 ⏎ Invalid: 0.000 ⏎ Latency: 233.496 s ⏎ Output throughput: 571.187 token/s ⏎ ```

### L1-405780bcf0  (L1, 2025-06-16, sha 405780bcf0d1, PR #7160)
TITLE: [amd] Opt dsv3 moe (#7160)
SOURCES: symbol_pickaxe
STAGE1: optimize; artifacts=L1.runner.aiter; AITER MoE implementation used to improve DS FP8 MoE performance.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+10/-8)
PERF_LINES: The throughput can increase from 1.73 req/s to 1.88 req/s | Request throughput (req/s):              1.88 | Input token throughput (tok/s):          6021.25 | Output token throughput (tok/s):         1505.31 | Total token throughput (tok/s):          7526.56 | ----------------End-to-End Latency---------------- | Mean E2E Latency (ms):                   66487.79 | Median E2E Latency (ms):          
BODY: ## Motivation ⏎  ⏎ Use new aiter moe implementation to enhance the performance of MoE fp8 computation of ds model ⏎  ⏎ The throughput can increase from 1.73 req/s to 1.88 req/s ⏎  ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang ⏎ Traffic request rate:                    inf ⏎ Max request concurrency:                 128 ⏎ Successful requests:                     500 ⏎ Benchmark duration (s):                  265.73 ⏎ Total input tokens:                      1600000 ⏎ Total generated tokens:                  400000 ⏎ Total generated tokens (retokenized):    398605 ⏎ Request throughput (req/s):              1.88 ⏎ Input token throughput (tok/s):          6021.25 ⏎ Output token throughput (tok/s):         1505.31 ⏎ Total token throughput (tok/s):          7526.56 ⏎ Concurrency:                             125.11 ⏎ ----------------End-to-End Latency---------------- ⏎ Mean E2E Latency (ms):                   66487.79 ⏎ Median E2E Latency (ms):                 67496.35 ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          12757.98 ⏎ Median TTFT (ms):                        12824.62 ⏎ P99 TTFT (ms):                           23708.40 ⏎ ---------------Inter-Token Latency---------------- ⏎ Mean ITL (ms):                           67.25 ⏎ Median ITL (ms):                         54.23 ⏎ P95 ITL (ms):                            55.63 ⏎ P99 ITL (ms):                            56.14 ⏎ Max ITL (ms):                            23758.53 ⏎ ================================================== ⏎ ``` ⏎  ⏎ ## Modifications ⏎  ⏎ ## Checklist ⏎  ⏎ - [✓] Format your code according to the [Code Formatting with Pre-Commit](https://docs.sglang.ai/references/contribution_guide.html#code-formatting-with-pre-commit). ⏎ - [✓] Add unit tests as outlined in the [Running Unit Tests](https://docs.sglang.ai/references/contribution_guide.html#running-unit-tests-adding-to-ci). ⏎ - [✓] Update documentation / docstrings / example tutorials as needed, according to [Writing Documentation](https://docs.sglang.ai/references/contribution_guide.html#writing-documentation-running-docs-ci). ⏎ - [✓] Provide throughput / latency benchmark results and accuracy evaluation results as needed, according to [Benchmark and Profiling](https://docs.sglang.ai/references/benchmark_and_profiling.html) and [Accuracy Results](https://docs.sglang.ai/references/accuracy_evaluation.html). ⏎ - [✓] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging th …[truncated]

### L1-094c116f7d  (L1, 2025-06-17, sha 094c116f7dc4, PR #6614)
TITLE: Update python API of activation, topk, norm and rope and remove vllm dependency (#6614)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.routing.topk_py,L1.triton.fused_moe; Updates topk and fused_moe Python APIs, removing vLLM dependency.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+6/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-1); python/sglang/srt/layers/moe/topk.py (+91/-4); docker/Dockerfile.xeon (+1/-0); python/sglang/srt/custom_op.py (+5/-1); python/sglang/srt/layers/activation.py (+20/-3); python/sglang/srt/layers/layernorm.py (+28/-2); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+5/-2); python/sglang/srt/layers/quantization/fp8.py (+5/-1); python/sglang/srt/layers/quantization/utils.py (+4/-2); python/sglang/srt/layers/rotary_embedding.py (+41/-2); python/sglang/srt/model_executor/model_runner.py (+2/-1); python/sglang/srt/models/deepseek_v2.py (+9/-2); python/sglang/srt/utils.py (+14/-1); test/srt/cpu/test_activation.py (+1/-1); test/srt/cpu/test_gemm.py (+5/-5); test/srt/cpu/test_moe.py (+3/-5); test/srt/cpu/test_norm.py (+4/-4); test/srt/cpu/test_qkv_proj_with_rope.py (+12/-12); test/srt/cpu/test_rope.py (+2/-2); test/srt/cpu/test_shared_expert.py (+3/-3); test/srt/cpu/test_topk.py (+2/-2); test/srt/run_suite.py (+2/-0)
LABELS: high priority, ready-to-merge, intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is to update python API of activation, topk, norm and rope calling by using torch.ops.sgl_kernel.xxx_cpu in CPU part, which includes optimized implementation. And we also remove vllm dependency and use sgl_kernel implementation instead. ⏎  ⏎ We also refactor `is_cpu` and use `SGLANG_USE_CPU_ENGINE=1` env to indicate run on CPU device before runtime. When `is_cpu()` is true, `cpu_has_amx_support()` is false, it will go into torch native path, which is compatible with AMD platform and legacy intel platform. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5962e70d8d  (L1, 2025-06-22, sha 5962e70d8d37, PR #7327)
TITLE: FlashInfer NVFP4 MoE with EP & 2-stream shared expert (#7327)
SOURCES: path_core, symbol_pickaxe
STAGE1: integrate; artifacts=L1.runner.flashinfer_cutlass,L1.ep.layer; Integrates FlashInfer NVFP4 MoE with EP and layer adapter.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+62/-10); python/sglang/srt/layers/quantization/modelopt_quant.py (+62/-8); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+39/-1); python/sglang/srt/server_args.py (+15/-1)
LABELS: high priority
PERF_LINES: Latency: 33.545 s | Output throughput: 4141.189 token/s | Latency: 26.736 s | Output throughput: 4804.049 token/s | Latency: 27.856 s | Output throughput: 4691.077 token/s | Latency: 36.977 s | Output throughput: 3718.960 token/s
BODY: ## Motivation ⏎  ⏎ Integrates Flashinfer NVFP4 cutlass MoE kernel from https://github.com/flashinfer-ai/flashinfer/pull/1113  ⏎ You can enable it with `--enable-flashinfer-moe`. `--enable-ep-moe` is also supported for this backend. ⏎  ⏎ ## Example usage ⏎  ⏎ TP 8 ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path nvidia/DeepSeek-R1-FP4 --trust-remote-code --quantization modelopt_fp4 --tp 8 --enable-flashinfer-moe ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319 --port=30000 ⏎ Accuracy: 0.918 ⏎ Invalid: 0.000 ⏎ Latency: 33.545 s ⏎ Output throughput: 4141.189 token/s ⏎ ``` ⏎  ⏎ TP 8 + MoE EP 8  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path nvidia/DeepSeek-R1-FP4 --trust-remote-code --quantization modelopt_fp4 --tp 8  --enable-flashinfer-moe --enable-ep-moe ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319 --port=30000 ⏎ Accuracy: 0.928 ⏎ Invalid: 0.000 ⏎ Latency: 26.736 s ⏎ Output throughput: 4804.049 token/s ⏎ ``` ⏎  ⏎ TP 8 + MoE EP 8 + DP attention ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path nvidia/DeepSeek-R1-FP4 --trust-remote-code --quantization modelopt_fp4 --tp 8 --enable-flashinfer-moe --enable-ep-moe --dp 8 --enable-dp-attention ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319 --port=30000 ⏎ Accuracy: 0.932 ⏎ Invalid: 0.000 ⏎ Latency: 27.856 s ⏎ Output throughput: 4691.077 token/s ⏎ ``` ⏎  ⏎ Reference (existing FusedMoE path) ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path nvidia/DeepSeek-R1-FP4 --trust-remote-code --quantization modelopt_fp4 --tp 8 --disable-shared-experts-fusion ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319 --port=30000 ⏎ Accuracy: 0.898 ⏎ Invalid: 0.000 ⏎ Latency: 36.977 s ⏎ Output throughput: 3718.960 token/s ⏎ ``` ⏎  ⏎ ## Modifications ⏎  ⏎ The new kernel is integrated into ModelOptNvFp4FusedMoEMethod, with some changes to the weight loader. The EP changes are mostly in FusedMoE's weight loading. ⏎  ⏎ ## Checklist

### L1-bd4f581896  (L1, 2025-06-22, sha bd4f58189669, PR #7391)
TITLE: Fix torch compile run (#7391)
SOURCES: path_core, symbol_pickaxe, corpus:kernel-correctness-cases
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; fused_moe_triton layer fix for torch compile run.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-1); docker/Dockerfile.rocm (+1/-1); python/sglang/srt/layers/quantization/fp8.py (+8/-8); scripts/amd_ci_start_container.sh (+1/-1)
LABELS: bug, high priority, aiter
DEEP_STUDY: deep-study correctness case sglang:bd4f581896: class=hardware_compiler_specific; symptom=compile_or_build_failure; introducing=unknown
BODY: ## Motivation ⏎  ⏎ Run the ci test "test/srt/test_mla_deepseek_v3.py" hit the torch-compile errors ⏎  ⏎ ## Modifications ⏎  ⏎ Fix the problematic part that reported by TORCH_LOGS="+dynamo,+recompiles" TORCHDYNAMO_VERBOSE=1 debug log ⏎  ⏎ ## Checklist ⏎  ⏎ - [✓] Format your code according to the [Code Formatting with Pre-Commit](https://docs.sglang.ai/references/contribution_guide.html#code-formatting-with-pre-commit).

### L1-506c4928f5  (L1, 2025-06-23, sha 506c4928f546, PR #6821)
TITLE: feat: integrate deepgemm into EPMoE (#6821)
SOURCES: path_core, symbol_pickaxe
STAGE1: integrate; artifacts=L1.ep.layer,L1.runner.deep_gemm; Integrates DeepGEMM as an option in EPMoE kernels/layer.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+159/-2); python/sglang/srt/layers/moe/ep_moe/layer.py (+174/-1); python/sglang/test/test_block_fp8_ep.py (+1/-0); sgl-kernel/benchmark/bench_moe_ep_post_reorder.py (+1/-0); sgl-kernel/tests/test_ep_moe_post_reorder_kernel.py (+1/-0)
LABELS: high priority
PERF_LINES: | Batch Size | Test Group | Output Throughput (tok/s) | | | | Diff | +22.2% | | | | Diff | +8.4% | | | | Diff | +7.5% | | | | Diff | +8.6% | | | | Diff | +8.6% |
BODY: ## Motivation ⏎  ⏎  ⏎ For normal EPMoE (no DeepEP), integrate DeepGEMM as an option. ⏎  ⏎ This PR builds upon the work from [pr5805](https://github.com/sgl-project/sglang/pull/5805) Credit goes to @TianQiLin666666, who authored most of the code presented here. ⏎ ``` ⏎  ⏎ ``` ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Performance ⏎ two h20 node  ⏎  ⏎ node 1 ⏎ ``` ⏎ python -m sglang.launch_server --model-path /path/to/DeepSeek-V3/    --trust-remote-code --tp-size 16 --enable-dp-attention --dp-size 16 --mem-fraction-static 0.8 --enable-ep-moe --disable-radix-cache --dist-init-addr 10.6.131.5:5000 --nnodes 2 --node-rank 0 ⏎ ``` ⏎  ⏎ node 2 ⏎ ``` ⏎ python -m sglang.launch_server --model-path /path/to/DeepSeek-V3/    --trust-remote-code --tp-size 16 --enable-dp-attention --dp-size 16 --mem-fraction-static 0.8 --enable-ep-moe --disable-radix-cache  --dist-init-addr 10.6.131.5:5000 --nnodes 2 --node-rank 1 ⏎ ``` ⏎  ⏎ test ⏎ ``` ⏎ python3 -m sglang.bench_one_batch_server --model /path/to/DeepSeek-V3/ --base-url http://localhost:30000 --batch-size 1 16 32 64 128 --input-len 1024 --output-len 1024 ⏎ ``` ⏎ | Batch Size | Test Group | Output Throughput (tok/s) | ⏎ |------------|---------------|---------------------------| ⏎ | 1 | w/o deepgemm | 24.95 | ⏎ | | w deepgemm | 30.49 | ⏎ | | Diff | +22.2% | ⏎ | 16 | w/o deepgemm | 309.32 | ⏎ | | w deepgemm | 335.31 | ⏎ | | Diff | +8.4% | ⏎ | 32 | w/o deepgemm | 553.30 | ⏎ | | w deepgemm | 595.04 | ⏎ | | Diff | +7.5% | ⏎ | 64 | w/o deepgemm | 1003.07 | ⏎ | | w deepgemm | 1088.89 | ⏎ | | Diff | +8.6% | ⏎ | 128 | w/o deepgemm | 1644.28 | ⏎ | | w deepgemm | 1786.10 | ⏎ | | Diff | +8.6% | ⏎  ⏎ ## Checklist

### L1-15f3401343  (L1, 2025-06-23, sha 15f34013432f, PR #7376)
TITLE: Fix MTP with Deepseek R1 Fp4 (#7376)
SOURCES: path_core, symbol_pickaxe
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; fused_moe_triton layer fixes DeepSeek R1 FP4 MTP quant config.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-0); python/sglang/srt/models/deepseek_nextn.py (+6/-0); python/sglang/srt/models/deepseek_v2.py (+8/-1)
LABELS: bug, high priority
PERF_LINES: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ | Latency: 141.432 s | Output throughput: 1161.636 token/s | [2025-06-20 04:35:25 TP0] Decode batch. #running-req: 4, #token: 3121, token usage: 0.00, accept len: 2.74, cuda graph: True, gen throughput (token/s): 396.76, 
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/issues/7365 ⏎  ⏎ ## Modifications ⏎  ⏎ Update quant config to None for MTP module in Deepseek modelopt fp4. ⏎  ⏎ It's not quantized. See https://github.com/NVIDIA/TensorRT-LLM/blob/5d4ab47d5b333296d33d62dde095310d7eba0b48/tensorrt_llm/_torch/models/modeling_deepseekv3.py#L682 ⏎  ⏎ Tested as follow: ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --port=7080 --model-path=/root/.cache/huggingface/hub/models--nvidia--DeepSeek-R1-FP4/snapshots/4bedb8a695a119b1a38d16a675c4665e58708aea/  --trust-remote-code --tp=8 --host=0.0.0.0 --speculative-algorithm=EAGLE --speculative-num-steps=3 --speculative-eagle-topk=1 --speculative-num-draft-tokens=4 ⏎ ``` ⏎  ⏎ GSM8k ⏎ ``` ⏎ root@predictor-resource-pool-1907436070400688128-bcc64d749-kccns:/sgl-workspace/sglang# python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319 --port=7080 ⏎ 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [02:20<00:00,  9.37it/s] ⏎ Accuracy: 0.873 ⏎ Invalid: 0.000 ⏎ Latency: 141.432 s ⏎ Output throughput: 1161.636 token/s ⏎ ``` ⏎  ⏎ Accept rate is  ⏎ ``` ⏎ [2025-06-20 04:35:25 TP0] Decode batch. #running-req: 4, #token: 3121, token usage: 0.00, accept len: 2.74, cuda graph: True, gen throughput (token/s): 396.76, #queue-req: 0 ⏎ ``` ⏎  ⏎ ## Checklist

### L1-e984d5073b  (L1, 2025-06-24, sha e984d5073bc8, PR #7423)
TITLE: enable aiter_biased_grouped_topk kernel (#7423)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: integrate; artifacts=L1.routing.topk_py,L1.runner.aiter; topk.py enables AITER biased_grouped_topk kernel path.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+26/-0); python/sglang/srt/model_executor/cuda_graph_runner.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+2/-1)
BODY: ## Motivation ⏎  ⏎ ## Modifications ⏎  ⏎ ## Checklist

### L1-7151194bb2  (L1, 2025-06-24, sha 7151194bb2f9, PR #7439)
TITLE: Remove cumsum_buffer initilization (#7439)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.triton.fused_moe; fused_moe_triton removes unnecessary cumsum_buffer initialization.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+7/-2)
LABELS: high priority
PERF_LINES: Latency: 104.687 s | Output throughput: 1772.852 token/s | |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |    | |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |   
BODY: ## Motivation ⏎  ⏎ `cumsum_buffer` is not needed to be initialized here. ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model Qwen/Qwen3-235B-A22B-FP8 --tp 4 ⏎  ⏎ Accuracy: 0.944 ⏎ Invalid: 0.000 ⏎ Latency: 104.687 s ⏎ Output throughput: 1772.852 token/s ⏎ ``` ⏎ mian: ⏎ ``` ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |   p99_tpot_ms |   per_user_throughput | ⏎ +====+===================+====================+=====================+================+==================+===============+================+==================+===============+=======================+ ⏎ |  1 |             1.000 |             66.590 |              66.590 |        295.734 |          198.118 |       636.961 |         14.732 |           14.497 |        15.677 |                66.590 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |  1 |             4.000 |            222.213 |             222.213 |        599.022 |          405.376 |      1974.015 |         17.415 |           17.488 |        18.503 |                55.553 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |  3 |            16.000 |            650.278 |             650.278 |        815.932 |          872.726 |       986.677 |         23.801 |           23.767 |        25.208 |                40.642 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |  4 |            32.000 |           1105.222 |            1105.222 |       1230.355 |         1416.241 |      1799.283 |         27.741 |           27.565 |        29.724 |                34.538 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ ``` ⏎ this PR: ⏎ ``` ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+--------- …[truncated]

### L1-755f314785  (L1, 2025-06-24, sha 755f314785fc, PR #7268)
TITLE: [AMD] add aiter fused moe in DeepEP path (#7268)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L1.ep.layer,L1.runner.aiter; EP layer adds AITER fused_moe path for DeepEP performance.
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+79/-12); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+19/-2)
PERF_LINES: 100%|██████████████████████████████████████████████████████████████| 1319/1319 [05:20<00:00,  4.12it/s] | Latency: 321.660 s | Output throughput: 416.608 token/s | 100%|██████████████████████████████████████████████████████████████| 1319/1319 [02:13<00:00,  9.90it/s] | Latency: 135.440 s | Output throughput: 990.551 token/s
BODY: ## Motivation ⏎  ⏎ For better performance in DeepEP path ⏎  ⏎ ## Modifications ⏎  ⏎ add aiter fused moe in DeepEP path ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Accuracy and performance ⏎  ⏎ without aiter fused_moe: ⏎ ``` ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 ⏎ 100%|██████████████████████████████████████████████████████████████| 1319/1319 [05:20<00:00,  4.12it/s] ⏎ Accuracy: 0.946 ⏎ Invalid: 0.000 ⏎ Latency: 321.660 s ⏎ Output throughput: 416.608 token/s ⏎ ``` ⏎  ⏎ with aiter fused_moe ⏎ ``` ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000  ⏎ 100%|██████████████████████████████████████████████████████████████| 1319/1319 [02:13<00:00,  9.90it/s] ⏎ Accuracy: 0.945 ⏎ Invalid: 0.000 ⏎ Latency: 135.440 s ⏎ Output throughput: 990.551 token/s ⏎ ``` ⏎  ⏎ We see significant performance uplift and same accuracy results.

### L1-57ab776910  (L1, 2025-06-24, sha 57ab77691038, PR #7437)
TITLE: Fuse sorted_token_ids padding to moe_align_block_size kernel (#7437)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: optimize; artifacts=L1.align.cuda_aot; CUDA moe_align fuses sorted_token_ids padding into align kernel.
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/common_extension.cc (+2/-1); sgl-kernel/csrc/moe/moe_align_kernel.cu (+57/-5); sgl-kernel/csrc/torch_extension_rocm.cc (+2/-1); sgl-kernel/include/sgl_kernel_ops.h (+2/-1); sgl-kernel/python/sgl_kernel/moe.py (+2/-0); sgl-kernel/benchmark/bench_moe_align_block_size.py (+53/-48); sgl-kernel/tests/test_moe_align.py (+42/-11)
BODY: ## Motivation ⏎  ⏎ Fuse `sorted_token_ids` padding to `moe_align_block_size`. In some shapes, it can perform better, so make it optional. ⏎  ⏎ ## Benchmark ⏎ ``` ⏎      num_tokens  num_experts  topk         SGL  SGL Fusion       Triton ⏎ 0           1.0          8.0   1.0   19.088000   17.216001    53.504001 ⏎ 1           1.0          8.0   2.0   19.072000   17.535999    32.768000 ⏎ 2           1.0          8.0   4.0   19.152001   17.503999    31.872001 ⏎ 3           1.0          8.0   8.0   20.032000   17.792000    31.840000 ⏎ 4           1.0         32.0   1.0   22.175999   19.584000    30.128000 ⏎ 5           1.0         32.0   2.0   22.175999   19.648001    39.391998 ⏎ 6           1.0         32.0   4.0   22.560000   19.808000    36.880001 ⏎ 7           1.0         32.0   8.0   22.143999   20.032000    29.440001 ⏎ 8           1.0         64.0   1.0   26.528001   23.903999    32.928001 ⏎ 9           1.0         64.0   2.0   26.559999   23.952000    34.111999 ⏎ 10          1.0         64.0   4.0   26.591999   24.032000    31.744000 ⏎ 11          1.0         64.0   8.0   26.464000   24.160000    40.592000 ⏎ 12          1.0        128.0   1.0   26.144000   23.328001    36.192000 ⏎ 13          1.0        128.0   2.0   26.144000   23.328001    36.224000 ⏎ 14          1.0        128.0   4.0   26.144000   23.328001    36.224000 ⏎ 15          1.0        128.0   8.0   26.112000   23.456000    36.320001 ⏎ 16          1.0        256.0   1.0   30.144000   27.488001    52.320000 ⏎ 17          1.0        256.0   2.0   30.080000   27.488001    52.288000 ⏎ 18          1.0        256.0   4.0   30.112000   27.424000    52.576002 ⏎ 19          1.0        256.0   8.0   30.112000   27.551999    52.416001 ⏎ 20          8.0          8.0   1.0   20.032000   17.824000    37.440000 ⏎ 21          8.0          8.0   2.0   20.160001   17.920000    41.120000 ⏎ 22          8.0          8.0   4.0   18.719999   17.999999    39.264001 ⏎ 23          8.0          8.0   8.0   20.447999   18.464001    40.064000 ⏎ 24          8.0         32.0   1.0   22.143999   20.064000    30.080000 ⏎ 25          8.0         32.0   2.0   22.431999   20.640001    30.832000 ⏎ 26          8.0         32.0   4.0   22.048000   21.215999    31.647999 ⏎ 27          8.0         32.0   8.0   22.496000   22.368001    30.784000 ⏎ 28          8.0         64.0   1.0   26.464000   24.160000    38.047999 ⏎ 29          8.0         64.0   2.0   26.432000   24.576001    32.095999 ⏎ 30          8.0         64.0   4.0   26.800000   25.024001    32.320000 ⏎ 31          8.0         64.0   8.0   26.432000   25.696000    37.535999 ⏎ 32          8.0        128.0   1. …[truncated]

### L1-7eb47b0f3d  (L1, 2025-06-25, sha 7eb47b0f3d0c, PR #6641)
TITLE: [CPU] [BF16] Call fused_experts_cpu, weight_packed_linear and bmm_cpu kernel in DeepSeek model (#6641)
SOURCES: path_core, symbol_pickaxe
STAGE1: change_default; artifacts=L1.hardware.cpu_npu_musa; CPU DeepSeek path selects fused_experts_cpu and related kernels.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_native.py (+7/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+73/-14); python/sglang/srt/layers/linear.py (+18/-1); python/sglang/srt/layers/logits_processor.py (+12/-3); python/sglang/srt/layers/vocab_parallel_embedding.py (+14/-1); python/sglang/srt/models/deepseek_v2.py (+145/-1); python/sglang/srt/utils.py (+71/-0); sgl-kernel/csrc/cpu/gemm.cpp (+2/-2); test/srt/cpu/test_gemm.py (+1/-1)
LABELS: intel, cpu
BODY: ## Motivation ⏎  ⏎ When CPU has AMX support, replace `moe_forward_native` with `fused_experts_cpu`, replace `F.linear` and `torch.matmul` with `weight_packed_linear`, add `AttnForwardMethod.MLA_FUSED_ROPE_CPU` to optimize the performance. ⏎  ⏎ https://github.com/sgl-project/sglang/pull/6408 (merged), https://github.com/sgl-project/sglang/pull/6614 and https://github.com/sgl-project/sglang/pull/6833 need to be landed first and the current PR will work then. ⏎  ⏎ ## Modifications ⏎ When CPU has AMX support, ⏎ 1. MoE ⏎ - pack the weight of MoE ⏎ - use the `fused_experts_cpu` kernel for better performance ⏎ - fix the input parameters of `moe_forward_native`. ⏎  ⏎ 2. Linear ⏎ - pack the weight of linear, MoEGate and lm_head ⏎  - Add a `PackWeightMethod` to handle weight packing ⏎ - use the `weight_packed_linear` kernel for better performance ⏎ - disable expert location update on CPU. Otherwise both the original weight and the packed weight of MoE are saved into the module, which makes MoE memory usage doubled. ⏎  ⏎ 3. attention ⏎ - add `AttnForwardMethod.MLA_FUSED_ROPE_CPU` ⏎ - use the `torch.ops.sgl_kernel.fused_qkv_proj_with_rope` kernel to optimize the performance (the kernel is added in https://github.com/sgl-project/sglang/pull/6833)

### L1-a5317b2fd3  (L1, 2025-06-27, sha a5317b2fd3dd, PR #6769)
TITLE: [CPU] add optimizations for INT8 and FP8 DeepSeek (#6769)
SOURCES: path_core, symbol_pickaxe
STAGE1: change_default; artifacts=L1.hardware.cpu_npu_musa; CPU DeepSeek INT8/FP8 paths call fused_experts_cpu/shared_expert kernels.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-1); python/sglang/srt/layers/quantization/fp8.py (+43/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+1/-1); python/sglang/srt/layers/quantization/w8a8_int8.py (+51/-1); python/sglang/srt/models/deepseek_v2.py (+83/-0)
LABELS: intel, cpu
BODY: ## Motivation ⏎ Call the below kernels in DeepSeek to optimize INT8 and FP8: ⏎ INT8 linear: `int8_scaled_mm_with_quant` ⏎ FP8 linear: `fp8_scaled_mm` ⏎ INT8 and FP8 shared_expert: `shared_expert_cpu` ⏎ INT8 and FP8 MoE: `fused_experts_cpu` ⏎  ⏎ For bmm, currently we don't support weight to be FP8, we convert weight to BF16 (only for CPU).

### L1-82eccae44e  (L1, 2025-06-28, sha 82eccae44e86, PR #7309)
TITLE: Let ep_scatter support arbitrary strides / ue8m0 format (#7309)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.ep.layer; EP scatter kernel supports arbitrary strides and ue8m0 format.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+22/-6)
LABELS: high priority, ready-to-merge
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-00c7b1ad07  (L1, 2025-06-28, sha 00c7b1ad0788, PR #7310)
TITLE: Let EP prefill support new DeepGEMM (#7310)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.ep.layer,L1.ep.deepep_dispatcher; EP prefill path supports new DeepGEMM format.
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+24/-7); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+7/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ please merge #7309 first (o/w this PR will error) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-22352d47a9  (L1, 2025-06-29, sha 22352d47a93e, PR #7632)
TITLE: Improve streaming, log_level, memory report, weight loading, and benchmark script (#7632)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.routing.router_py; MoE router kernel gains correction_bias and topk<=2 support.
ARTIFACT_HINTS: L1.routing.router_py
FILES: python/sglang/srt/layers/moe/router.py (+60/-22); docs/backend/server_arguments.md (+1/-1); python/sglang/bench_one_batch_server.py (+14/-2); python/sglang/bench_serving.py (+0/-1); python/sglang/srt/configs/internvl.py (+2/-3); python/sglang/srt/entrypoints/http_server.py (+23/-6); python/sglang/srt/layers/elementwise.py (+76/-12); python/sglang/srt/managers/configure_logging.py (+1/-1); python/sglang/srt/managers/io_struct.py (+3/-3); python/sglang/srt/managers/mm_utils.py (+52/-0); python/sglang/srt/managers/multimodal_processor.py (+0/-1); python/sglang/srt/managers/schedule_batch.py (+5/-51); python/sglang/srt/managers/scheduler.py (+36/-20); python/sglang/srt/managers/scheduler_output_processor_mixin.py (+8/-2); python/sglang/srt/managers/tokenizer_manager.py (+124/-25); python/sglang/srt/mem_cache/memory_pool.py (+3/-0); python/sglang/srt/model_executor/model_runner.py (+5/-2); python/sglang/srt/model_loader/loader.py (+38/-0); python/sglang/srt/models/mistral.py (+1/-1); python/sglang/srt/server_args.py (+9/-2); python/sglang/srt/warmup.py (+12/-3); scripts/playground/replay_request_dump.py (+150/-0); test/srt/run_suite.py (+2/-1); test/srt/test_vision_chunked_prefill.py (+1/-1)
BODY: - support disable streaming in bench_one_batch_server.py ⏎ - add more log_request_level ⏎ - support cpu quantization for weight loading  ⏎ - simplify tokenizer manager ⏎ - support crash dump and replay ⏎ - fix CI

### L1-7248272ccc  (L1, 2025-06-29, sha 7248272ccc21, PR #7627)
TITLE: Add dsv3 router gemm kernel (#7627)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: introduce; artifacts=NEW:dsv3_router_gemm; Adds new DSV3 router GEMM CUDA kernel adapted from TRT-LLM.
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/csrc/common_extension.cc (+3/-0); sgl-kernel/include/sgl_kernel_ops.h (+1/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/benchmark/bench_dsv3_router_gemm.py (+56/-0); sgl-kernel/csrc/gemm/dsv3_router_gemm.cu (+286/-0); sgl-kernel/python/sgl_kernel/gemm.py (+18/-0); sgl-kernel/tests/test_dsv3_router_gemm.py (+32/-0)
PERF_LINES: Adapt router gemm kernel from [trtllm](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu)
BODY: ## Motivation ⏎  ⏎ Adapt router gemm kernel from [trtllm](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu) ⏎  ⏎ Note that this kernel takes two bf16 tensors as input, and outputs a fp32 tensor. ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ For next PR: Integrate into dspk-r1 fp4 ⏎  ⏎  ⏎ ## Accuracy Test ⏎ ```bash ⏎ python3 tests/test_dsv3_router_gemm.py ⏎ ``` ⏎ <img width="1259" alt="截屏2025-06-29 17 08 17" src="https://github.com/user-attachments/assets/fd3d7cf1-a058-4649-bc53-ce66d70a255e" /> ⏎  ⏎ ## Benchmark ⏎ ```bash ⏎ python3 bench/bench_dsv3_router_gemm.py ⏎ ``` ⏎ <img width="483" alt="截屏2025-06-29 17 13 00" src="https://github.com/user-attachments/assets/8a5299f7-1eb7-4d8f-bcae-1e49e85d2755" /> ⏎  ⏎ ## Checklist

### L1-8e03b641ba  (L1, 2025-07-01, sha 8e03b641baf1, PR #7683)
TITLE: [1/n] apply wna16marlin kernel in moe weight only quantization (#7683)
SOURCES: path_core, symbol_pickaxe
STAGE1: introduce; artifacts=L1.runner.marlin; Adds marlin_moe_wna16 CUDA sources for MoE weight-only quantization.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.runner.marlin
FILES: sgl-kernel/csrc/moe/marlin_moe_wna16/awq_marlin_repack.cu (+255/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/core/registration.h (+25/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/generate_kernels.py (+106/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/gptq_marlin/marlin.cuh (+96/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/gptq_marlin/marlin_dtypes.cuh (+83/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/gptq_marlin_repack.cu (+333/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel.h (+40/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4.cu (+89/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4b8.cu (+109/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku8b128.cu (+109/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4.cu (+89/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4b8.cu (+109/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku8b128.cu (+109/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/marlin_template.h (+1804/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/ops.cu (+1112/-0); sgl-kernel/python/sgl_kernel/fused_moe.py (+223/-0); python/sglang/srt/layers/quantization/quant_utils.py (+166/-0); sgl-kernel/CMakeLists.txt (+9/-0); sgl-kernel/csrc/common_extension.cc (+18/-1); sgl-kernel/include/scalar_type.hpp (+328/-0); sgl-kernel/include/sgl_kernel_ops.h (+37/-0); sgl-kernel/python/sgl_kernel/__init__.py (+6/-0); sgl-kernel/python/sgl_kernel/marlin.py (+44/-0); sgl-kernel/python/sgl_kernel/scalar_type.py (+352/-0); sgl-kernel/tests/test_marlin_repack.py (+138/-0); test/srt/test_int4_kernel.py (+301/-0); test/srt/test_w4a8.py (+14/-0)
BODY: ## Motivation ⏎  ⏎  ⏎ This is the sgl-kernel part of https://github.com/sgl-project/sglang/pull/5639. A series of PRs will remove the wna16 quantization dependency on vllm and optimize the feature. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2c4feaf308  (L1, 2025-07-02, sha 2c4feaf30875, PR #7278)
TITLE: Add CUTLASS FP8 Blockscale MoE kernel for Hopper architecture (#7278)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: optimize; artifacts=L1.cutlass.fp8_blockwise; Changes CUTLASS FP8 blockscale MoE kernel for Hopper performance.
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+239/-6); sgl-kernel/benchmark/bench_fp8_blockwise_group_gemm.py (+330/-0); sgl-kernel/tests/test_fp8_blockwise_moe.py (+9/-3)
PERF_LINES: Using the benchmark we provided in the PR, under some common configs, the CUTLASS FP8 Blockscale MoE kernel has ~5%-6% speedup over DeepGEMM.
BODY: ## Motivation ⏎  ⏎  ⏎ Using the benchmark we provided in the PR, under some common configs, the CUTLASS FP8 Blockscale MoE kernel has ~5%-6% speedup over DeepGEMM. ⏎  ⏎ We carried out further analysis on the CUTLASS FP8 Blockscale MoE kernel, and the primary conclusions are as follows: ⏎  ⏎ 1. When using the schedule of Pingpong, there is no requirement to synchronize Warp Groups after the execution of the Mainloop. In comparison with the Cooperative approach, this asynchronous method can efficiently and continuously utilize TensorCore. ⏎ 2. DeepGEMM outperforms CUTLASS in the Small K scenario. The reason is that CUTLASS uses inefficient TensorCore instructions (CUTLASS 64x128x32 vs. DeepGEMM 64x160x32) and executes a lot of unnecessary FFMA instructions in Epilogue. Therefore, we will skip these Small K scenarios in the code and encourage users to use other implementations. ⏎  ⏎ Consequently, in ep and some tp + prefill configurations, the cutlass kernel can enhance performance when compared to DeepGEMM. ⏎  ⏎ More profiling details will be shown in a blog. ⏎  ⏎  ⏎ `The results of python3 sgl-kernel/benchmark/bench_fp8_blockwise_group_gemm.py` ⏎  ⏎ ![image](https://github.com/user-attachments/assets/ce299801-125d-4a0d-abf8-6b95cf307244) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Add a  CUTLASS MoE kernel for Hopper architecture ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎ ## Future work

### L1-9fcc9a80e7  (L1, 2025-07-03, sha 9fcc9a80e7bd, PR #7647)
TITLE: [CPU] refine CPU integration code (#7647)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.hardware.cpu_npu_musa; CPU integration fixes MoE layer prepack/block-quant checks for CPU kernels.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-6); python/sglang/srt/layers/amx_utils.py (+86/-0); python/sglang/srt/layers/linear.py (+4/-3); python/sglang/srt/layers/logits_processor.py (+2/-2); python/sglang/srt/layers/quantization/fp8.py (+6/-5); python/sglang/srt/layers/quantization/w8a8_int8.py (+6/-5); python/sglang/srt/layers/vocab_parallel_embedding.py (+2/-6); python/sglang/srt/models/deepseek_v2.py (+29/-20); python/sglang/srt/utils.py (+2/-69)
LABELS: intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎ 1. As discussed here https://github.com/sgl-project/sglang/commit/7eb47b0f3d0cd69ff3e0dd140cb377e15d6b148a#r161090822, we only need to check the `weight_block_size` if the model is employing `block_quant`. In addition, this check is the requirement of CPU kernels so I limit the check to happen only when `_is_cpu and _is_cpu_amx_available`. ⏎  ⏎ 2. When checking the supported dims for prepack, fix the dim check when weight is 3D or need to be transposed. ⏎ 3. make `getattr(sth, "use_intel_amx_backend", False)` as a method in `utils.py` to address https://github.com/sgl-project/sglang/pull/6641#discussion_r2175389691 ⏎ 4. move CPU weight pack functions to `layers/amx_utils.py` and rename them to address https://github.com/sgl-project/sglang/pull/6641#discussion_r2175408594

### L1-1dce6c480f  (L1, 2025-07-03, sha 1dce6c480fac, PR #6771)
TITLE: [CPU] support the case where num_attention_heads or intermediate_size is not divisible by the TP size (#6771)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.hardware.cpu_npu_musa,L1.triton.fused_moe; Allows CPU MoE path when dimensions are not divisible by TP size.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+33/-9); python/sglang/srt/configs/update_config.py (+119/-0); python/sglang/srt/layers/linear.py (+80/-11); python/sglang/srt/layers/parameter.py (+67/-7); python/sglang/srt/layers/vocab_parallel_embedding.py (+9/-1); python/sglang/srt/managers/scheduler.py (+9/-3); python/sglang/srt/model_executor/model_runner.py (+6/-1); python/sglang/srt/model_loader/weight_utils.py (+54/-0); python/sglang/srt/models/mllama4.py (+13/-7); python/sglang/srt/models/qwen2.py (+7/-1); python/sglang/srt/utils.py (+2/-0)
LABELS: intel, cpu
BODY: ## Motivation ⏎ Support the case where num_attention_heads or intermediate_size is not divisible by the TP size, for example running TP = 6 on the below machine with 6 numa nodes: ⏎  ⏎ ```sh ⏎ NUMA:                     ⏎   NUMA node(s):          6 ⏎   NUMA node0 CPU(s):     0-39,240-279 ⏎   NUMA node1 CPU(s):     40-79,280-319 ⏎   NUMA node2 CPU(s):     80-119,320-359 ⏎   NUMA node3 CPU(s):     120-159,360-399 ⏎   NUMA node4 CPU(s):     160-199,400-439 ⏎   NUMA node5 CPU(s):     200-239,440-479 ⏎ ``` ⏎  ⏎ https://github.com/sgl-project/sglang/pull/6549 needs to be landed first. ⏎  ⏎ ## Modifications ⏎ 1. For CPU, we will pad num_attention_heads, intermediate_size and vocab_size to be divisible by the PT size. In addition, for FP8, the size after padding needs to be divisible by weight block size as well. ⏎ For padded values, we will set them to zero so that the computation result is not impacted. ⏎  ⏎    If the device is not CPU, the behavior is same as before. ⏎  ⏎ 2. Fixes the below error due to the change in https://github.com/sgl-project/sglang/pull/7632. We only call `self.tp_worker.worker.model_runner.cuda_graph_mem_usage` if `not _is_cpu`. ⏎ ``` ⏎ sglang/python/sglang/srt/managers/scheduler.py", line 2119, in get_internal_state ⏎     self.tp_worker.worker.model_runner.cuda_graph_mem_usage, 2 ⏎     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ AttributeError: 'ModelRunner' object has no attribute 'cuda_graph_mem_usage' ⏎ ```

### L1-c01a1df588  (L1, 2025-07-03, sha c01a1df5888c, PR #7723)
TITLE: [Bug] add flashinfer bool check for fusedmoe in Qwen moe models (#7723)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L1.upstream.flashinfer_moe; Qwen MoE models now pass enable_flashinfer_moe so the intended FlashInfer MoE path can run.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen2_moe.py (+9/-0); python/sglang/srt/models/qwen3_moe.py (+9/-0)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎ The current Qwen MoE models (`qwen2_moe.py` and `qwen3_moe.py`) do not pass the `enable_flashinfer_moe` flag, causing `FuseMoE` to default to `enable_flashinfer_moe=False`. ⏎  ⏎ ## Modifications ⏎  ⏎ Added checks to ensure the `enable_flashinfer_moe` flag is correctly passed to the `FuseMoE` initializer.

### L1-2998c4bdf4  (L1, 2025-07-03, sha 2998c4bdf4fe, PR #7744)
TITLE: [optimize] fuse renormalize into moe_topk_softmax (#7744)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L1.routing.topk_softmax; Fuses renormalization into moe_topk_softmax kernel and changes its interface.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_softmax
FILES: sgl-kernel/csrc/common_extension.cc (+1/-3); sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu (+157/-81); sgl-kernel/csrc/torch_extension_rocm.cc (+1/-3); sgl-kernel/include/sgl_kernel_ops.h (+1/-4); sgl-kernel/python/sgl_kernel/moe.py (+2/-2); sgl-kernel/benchmark/bench_moe_topk_softmax.py (+0/-4); sgl-kernel/tests/test_moe_topk_softmax.py (+92/-4)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ fuse normalize into moe took softmax kernel. ⏎ This PR modifies topk_softmax interface, we need also modify python code. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-489934be0a  (L1, 2025-07-03, sha 489934be0ad3, PR #7751)
TITLE: fuse renormal into moe topk softmax kernel python code (#7751)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes
STAGE1: adapt_framework; artifacts=L1.routing.topk_py,L1.routing.topk_softmax; Updates topk.py to consume fused-renormalization topk_softmax API and dependency version.
ARTIFACT_HINTS: L1.routing.topk_py, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/topk.py (+1/-25); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: high priority
PERF_LINES: Latency: 132.842 s | Output throughput: 1027.763 token/s | | index | max_concurrency | input_throughput | output_throughput | mean_ttft_ms | median_ttft_ms | p99_ttft_ms | mean_tpot_ms | median_tpot_ms | p99_tpot_ms | p | | index | max_concurrency | input_throughput | output_throughput | mean_ttft_ms | median_ttft_ms | p99_ttft_ms | mean_tpot_ms | median_tpot_ms | p99_tpot_ms | p
BODY: ## Motivation ⏎ fuse renormal into moe topk softmax kernel python code, need merge #7744 and update sgl-kernel first ⏎  ⏎  ⏎  ⏎ ``` ⏎ accuracy ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 1000 ⏎ Accuracy: 0.951 ⏎ Invalid: 0.000 ⏎ Latency: 132.842 s ⏎ Output throughput: 1027.763 token/s ⏎ ``` ⏎  ⏎ before ⏎ | index | max_concurrency | input_throughput | output_throughput | mean_ttft_ms | median_ttft_ms | p99_ttft_ms | mean_tpot_ms | median_tpot_ms | p99_tpot_ms | per_user_throughput | ⏎ |-------|-----------------|------------------|-------------------|--------------|----------------|-------------|--------------|----------------|-------------|---------------------| ⏎ | 0     | 1.000           | 86.102           | 86.102            | 168.054      | 173.137        | 178.307     | 11.454       | 11.456         | 11.462      | 86.102              | ⏎ | 1     | 4.000           | 285.029          | 285.029           | 317.742      | 318.506        | 452.252     | 13.724       | 13.672         | 14.006      | 71.257              | ⏎ | 2     | 16.000          | 869.277          | 869.277           | 909.634      | 711.676        | 2098.242    | 17.506       | 17.498         | 18.628      | 54.330              | ⏎ | 3     | 32.000          | 1423.474         | 1423.474          | 1028.146     | 1113.705       | 1413.942    | 21.463       | 21.572         | 22.595      | 44.484              | ⏎  ⏎ after ⏎ | index | max_concurrency | input_throughput | output_throughput | mean_ttft_ms | median_ttft_ms | p99_ttft_ms | mean_tpot_ms | median_tpot_ms | p99_tpot_ms | per_user_throughput | ⏎ |-------|-----------------|------------------|-------------------|--------------|----------------|-------------|--------------|----------------|-------------|---------------------| ⏎ | 0     | 1.000           | 87.534           | 87.534            | 159.558      | 158.196        | 164.858     | 11.272       | 11.271         | 11.279      | 87.534              | ⏎ | 1     | 4.000           | 289.031          | 289.031           | 319.455      | 306.669        | 418.004     | 13.528       | 13.445         | 13.825      | 72.258              | ⏎ | 2     | 16.000          | 882.732          | 882.732           | 762.041      | 701.809        | 1363.294    | 17.373       | 17.572         | 18.171      | 55.171              | ⏎ | 3     | 32.000          | 1420.820         | 1420.820          | 1007.886     | 1099.283       | 1405.840    | 21.526       | 21.771         | 22.658      | 44.401              | ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8b1942c6cc  (L1, 2025-07-03, sha 8b1942c6cc08, PR #7759)
TITLE: Remove type conversion and fix id map in topk (#7759)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.routing.topk_py; Fixes top-k ID map and removes an unsafe type conversion in select_experts logic.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-1)
BODY: ## Motivation ⏎  ⏎ Remove type conversion and fix id map in topk.  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-da3890e82a  (L1, 2025-07-04, sha da3890e82a97, PR #7772)
TITLE: [1/n]: add cutlass W4A8 moe kernel for hopper architecture (#7772)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes, corpus:kernel-correctness-cases(introducing)
STAGE1: introduce; artifacts=L1.cutlass.w4a8; Adds W4A8 CUTLASS MoE CUDA kernels and Python bindings for Hopper.
ARTIFACT_HINTS: L1.cutlass.w4a8
FILES: sgl-kernel/CMakeLists.txt (+3/-0); sgl-kernel/csrc/common_extension.cc (+19/-0); sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu (+91/-0); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh (+92/-0); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu (+240/-0); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh (+276/-0); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu (+79/-0); sgl-kernel/include/sgl_kernel_ops.h (+29/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/python/sgl_kernel/cutlass_moe.py (+112/-0); sgl-kernel/csrc/cutlass_extensions/detail/collective/mixed_input_utils.hpp (+482/-0); sgl-kernel/csrc/cutlass_extensions/gemm/collective/builders/sm90_gmma_builder_mixed_input.inl (+278/-0); sgl-kernel/csrc/cutlass_extensions/gemm/collective/collective_builder_mixed_input.hpp (+52/-0); sgl-kernel/csrc/cutlass_extensions/gemm/collective/collective_mma_array_mixed_input.hpp (+53/-0); sgl-kernel/csrc/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp (+1535/-0); sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py (+260/-0)
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:de4990a5b2 (fix PR 9392)
PERF_LINES: - `w4a8_grouped_mm_c3x.cu`:Responsible for dispatching grouped GEMM operations. | - `w4a8_grouped_mm_c3x.cuh`: Contains the key computation logic. | - The tuning results are integrated into the dispatch logic in `w4a8_grouped_mm_c3x.cu`
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This is the sgl-kernel part of https://github.com/sgl-project/sglang/pull/7762. A series of PRs will enable running [DeepSeek-R1-W4AFP8](https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8) using sglang. ⏎  ⏎ Additionally, this kernel supports other MoE models with INT4 MoE weight and FP8 activation quantization. ⏎  ⏎ Co-author: yicwang <yichen.wang@bytedance.com> ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ In this kernel implementation, we focus on two main areas: ⏎ **1. Function Implementation**:  ⏎   - `scaled_mm_entry.cu`: This serves as the entry point for the CUTLASS W4A8 MoE. ⏎   - `w4a8_moe_data.cu`: Contains the logic for computing `expert_offsets` and `problem_size` ⏎   - `w4a8_grouped_mm_c3x.cu`:Responsible for dispatching grouped GEMM operations. ⏎   - `w4a8_grouped_mm_c3x.cuh`: Contains the key computation logic. ⏎   - `cutlass_extensions/`: This section is adapted from [trtllm](https://github.com/NVIDIA/TensorRT-LLM/pull/5027/files) and contains the separate CUTLASS components needed for the W4A8 grouped GEMM. ⏎  ⏎ **2. Tailored Performance Tuning for the DeepSeek R1 Model:** ⏎    - We explore the optimal CUTLASS template configuration for the MoE layer of DeepSeek R1 to maximize performance. ⏎    - The tuning results are integrated into the dispatch logic in `w4a8_grouped_mm_c3x.cu` ⏎   ⏎ ## Checklist

### L1-8e9fb43d82  (L1, 2025-07-04, sha 8e9fb43d8255, PR #7782)
TITLE: Optimize Hopper CUTLASS FP8 Blockwise Grouped GEMM Kernel in Small K Scenario (#7782)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.cutlass.fp8_blockwise; Optimizes CUTLASS FP8 blockwise kernel for small-K scenarios by changing kernel implementation.
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+86/-38)
LABELS: high priority
BODY: ## Motivation ⏎ Follow https://github.com/sgl-project/sglang/pull/7278 ⏎ When K in GEMM problem size is small, CUTLASS Kernel shows suboptimal performance. The reason is that when K is small, the execution time of Mainloop is very short and Epilogue is difficult to be overlapped by Mainloop. In CUTLASS, Epilogue performs unnecessary LinearCombination, which further degrades performance. ⏎ We compared the nsight-compute profile reports of DeepGEMM and CUTLASS after aligning TileShape. The results showed that in the Small K scenario, the number of FFMA instructions executed by CUTLASS far exceeded that of DeepGEMM.  ⏎ ![PR示意图](https://github.com/user-attachments/assets/6e8fc23a-3e05-4f6f-8d0c-549add7b67dd) ⏎ Therefore, we optimized the unnecessary LinearCombination in Epilogue. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ Only sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu: ⏎ + Use CUTLASS Sm90EVT to define an identity op in Epilogue ⏎ + When K is extremely small, the Cooperative Kernel has better performance, probably because TMA data copying is more efficient. Therefore, the Cooperative Kernel is used when K < 256. ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎ Lint done: ⏎ ![Small_K Code lint](https://github.com/user-attachments/assets/6c96b9cf-0dff-4d45-ad54-7ed965252d7a) ⏎  ⏎ Unitest passed: ⏎ ![单元测试通过](https://github.com/user-attachments/assets/523b5488-a41b-44ac-9840-106cf02f4be1) ⏎  ⏎ Before optimization: ⏎ ![Small_K优化前](https://github.com/user-attachments/assets/43c311bb-abec-4da6-8bb5-646d4072cc6b) ⏎ After optimization: ⏎ ![Small_K分区间优化](https://github.com/user-attachments/assets/0486c933-f115-47cd-b567-b2abae68e93e)

### L1-c797322280  (L1, 2025-07-04, sha c797322280b4, PR #7444)
TITLE: fix: fix apply_shuffle_mul_sum (#7444)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.cutlass.fp8_blockwise; Fixes prepare_moe_input shuffle/mul/sum truncation and adds strided/vectorized handling.
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/prepare_moe_input.cu (+67/-43)
LABELS: ready-to-merge
BODY: ## Motivation ⏎ Some bugs in `apply_shuffle_mul_sum_kernel`: ⏎ 1. some experts outputs were wrongly truncated by `src_row >= m`, where src_row is mistakenly taken as original ⏎    token index ( [0, m) ) ⏎ 2. in original impl, the thread-count assigned to a block is insufficient to cover row_stride(hidden_size, 4096). Modify with a strided loop ⏎  ⏎ 3. vectorize loading, increase blockDim ⏎  ⏎  ⏎ |                     | gsm8k                  |  ⏎ |---------------------|------------------------| ⏎ | shuffle_mul_sum     | 0.815 2010.813   token/s     |  ⏎ | shuffle + mul + sum(before) | 0.818 2092.073 token/s |  ⏎ |modified (bf16) |0.813  3081 token/s | ⏎ | modified (float) | 0.818 3139 token/s | ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8fc910db03  (L1, 2025-07-05, sha 8fc910db0310, PR #7222)
TITLE: DP Attention with Auto DeepEP Dispatch (#7222)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: change_default; artifacts=L1.ep.deepep_dispatcher,L1.ep.layer; Enables automatic DeepEP dispatch for DP attention configurations.
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+5/-3); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+15/-13); python/sglang/srt/model_executor/forward_batch_info.py (+2/-0); python/sglang/srt/models/deepseek_v2.py (+7/-7); python/sglang/srt/models/qwen3_moe.py (+7/-9); python/sglang/srt/server_args.py (+0/-4); python/sglang/srt/two_batch_overlap.py (+7/-3); python/sglang/srt/disaggregation/decode.py (+1/-1); python/sglang/srt/disaggregation/prefill.py (+2/-2); python/sglang/srt/managers/schedule_batch.py (+3/-0); python/sglang/srt/managers/scheduler.py (+3/-4); python/sglang/srt/utils.py (+4/-4); test/srt/test_hybrid_dp_ep_tp_mtp.py (+80/-40)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR enables auto DeepEP dispatch for DP attention. Integration with TBO will be supported in future PRs. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist
