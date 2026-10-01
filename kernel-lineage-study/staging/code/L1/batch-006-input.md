### L1-c4500233ff  (L1, 2025-08-22, sha c4500233ff20, PR #9456)
TITLE: Add Qwen3-30B-A3B-Thinking-2507 support on AMD GPUs. (#9456)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L1.triton.fused_moe; Triton fused-MoE code was extended for Qwen3-30B-A3B on AMD GPUs.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+18/-7)
PERF_LINES: Latency: 103.769 s | Output throughput: 2743.774 token/s
BODY: ## Motivation ⏎ Add Qwen3-30B-A3B-Thinking model support on AMD GPUs.  ⏎  ⏎  ⏎ ## Modifications ⏎ Enabling triton backend on AMD GPU. ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ SGLANG_USE_AITER=0 python3 -m sglang.launch_server --model-path Qwen3-30B-A3B-Thinking-2507/ --tp 8 --trust-remote-code --chunked-prefill-size 130172 --max-running-requests 128 --mem-fraction-static 0.85 --attention-backend aiter --enable-torch-compile ⏎  ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 ⏎ Accuracy: 0.858 ⏎ Invalid: 0.000 ⏎ Latency: 103.769 s ⏎ Output throughput: 2743.774 token/s ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-ccd3fb946e  (L1, 2025-08-23, sha ccd3fb946e04, PR #9473)
TITLE: [fix] Fix mxfp4 triton MoE tp bug (#9473)
SOURCES: path_core, symbol_pickaxe
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; MXFP4 Triton MoE tensor-parallel bug fixed fused-MoE layer integration.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-6); python/sglang/srt/layers/quantization/mxfp4.py (+7/-0); python/sglang/srt/models/gpt_oss.py (+5/-6)
LABELS: high priority
PERF_LINES: 8 x H100 tp8
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/pull/9433 broke GPT-OSS on Hopper because the MoE intermediate size was not padded after sharding on Hopper. Adding padding would fix the issue. However, it's only a workaround. The correct fix is to change the way of the weight sharding for block scaling data formats (mxfp4, nvfp4 etc) to the following before creating the weights and then add padding on top of that: ⏎  ⏎ ``` ⏎         intermediate_size_block = intermediate_size // mxfp4_block ⏎         per_rank_intermediate_size_block = math.ceil( ⏎             intermediate_size_block / moe_tp_size ⏎         ) ⏎         per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block ⏎ ``` ⏎ This is a bigger change and will require more testing. I will do this in a followup PR. ⏎  ⏎ ## Modifications ⏎  ⏎ Added intermediate size padding to triton MoE ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ 8 x H100 tp8 ⏎ ``` ⏎ before ⏎ [{'eval_name': 'gpqa', 'model_name': 'gpt-oss-120b-high_temp1.0_20250822_210411', 'metric': 0.7556818181818182}] ⏎  ⏎ After ⏎ [{'eval_name': 'gpqa', 'model_name': 'gpt-oss-120b-high_temp1.0_20250823_015440', 'metric': 0.7986111111111112}, ⏎ {'eval_name': 'mmlu', 'model_name': 'gpt-oss-120b-high_temp1.0_20250823_024653', 'metric': 0.8889759293547927},  ⏎ {'eval_name': 'aime25', 'model_name': 'gpt-oss-120b-high_temp1.0_20250823_024653', 'metric': 0.9041666666666667}] ⏎ ``` ⏎  ⏎ Perf on Hopper is going to regress a little bit because of padding, which means more computation than necessary. But the padding is necessary. Previously the weights were not even fully loaded so the computation is less that what it should have been. ⏎  ⏎ ## Checklist

### L1-86d10d220f  (L1, 2025-08-23, sha 86d10d220f66, PR #9532)
TITLE: Update grok.py and tiktoken tokenizer (#9532)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.routing.router_py; MoE router Triton kernel handles zero softcapping and fixes top2 boolean logic.
ARTIFACT_HINTS: L1.routing.router_py
FILES: python/sglang/srt/layers/moe/router.py (+15/-9); python/sglang/srt/constrained/xgrammar_backend.py (+10/-6); python/sglang/srt/hf_transformers_utils.py (+5/-0); python/sglang/srt/layers/attention/triton_backend.py (+16/-2); python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+31/-0); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+18/-0); python/sglang/srt/layers/elementwise.py (+94/-0); python/sglang/srt/layers/radix_attention.py (+6/-0); python/sglang/srt/models/grok.py (+376/-47); python/sglang/srt/tokenizer/tiktoken_tokenizer.py (+161/-0)
BODY: 

### L1-0374304a2c  (L1, 2025-08-23, sha 0374304a2cb6, PR #9004)
TITLE: Add enable_flashinfer_mxfp4_bf16_moe for higher precision and slower moe backend (#9004)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L1.runner.flashinfer_mxfp4; Added BF16 higher-precision FlashInfer MXFP4 MoE backend flag and scheduling integration.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4.py (+27/-5); python/sglang/srt/server_args.py (+9/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0)
PERF_LINES: mxfp4: 381 tok/s | mxfp4 bf16: 364 tok/s
BODY: ## Motivation ⏎  ⏎ ``` ⏎ CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path openai/gpt-oss-20b --tp-size 1 --port 30000 --enable-flashinfer-mxfp4-moe ⏎  ⏎ CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server --model-path openai/gpt-oss-20b --tp-size 1 --port 31000 --enable-flashinfer-mxfp4-bf16-moe ⏎  ⏎ python3 -m sglang.bench_serving --backend sglang-oai  --dataset-name random --random-input-len 512 --random-output-len 1024 --random-range-ratio 1 --num-prompts 10 --max-concurrency 1 --output-file res.jsonl --port 30000 ⏎ python3 -m sglang.bench_serving --backend sglang-oai  --dataset-name random --random-input-len 512 --random-output-len 1024 --random-range-ratio 1 --num-prompts 10 --max-concurrency 1 --output-file res.jsonl --port 31000 ⏎ ``` ⏎  ⏎ mxfp4: 381 tok/s ⏎ mxfp4 bf16: 364 tok/s ⏎  ⏎ gpqa ⏎ mxfp4: low=55.4, medium=66.0 ⏎ mxfp4_bf16: low=56.8, medium=66.2 ⏎ not sure whether real diff or just randomness ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fda4792620  (L1, 2025-08-24, sha fda47926208e, PR #9559)
TITLE: Update CUTLASS 4.2 & Enable K-Major Scale Factor for SM90 FP8 Blockwise Group GEMM (#9559)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.cutlass.fp8_blockwise,L1.cutlass.adapters; CUTLASS FP8 blockwise MoE enabled K-major scale factors and updated kernel wrapper.
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+0/-7); sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+49/-40); python/sglang/test/test_cutlass_moe.py (+33/-28); sgl-kernel/CMakeLists.txt (+1/-1); sgl-kernel/tests/test_fp8_blockwise_moe.py (+20/-57)
LABELS: high priority
PERF_LINES: 1. CUTLASS 4.2 supports scale factors in the K-Major format, allowing us to unify the code path with Blackwell, thus avoiding some format conversion kernel call
BODY: ## Motivation ⏎ 1. CUTLASS 4.2 supports scale factors in the K-Major format, allowing us to unify the code path with Blackwell, thus avoiding some format conversion kernel calls(per_group_transpose).  ⏎ 2. In addition, we optimized scenes with smaller M by swapping the A/B matrices. ⏎ 3. Use the ATen interface to determine whether the current device is H20 to avoid performance impact of `cudaGetDeviceProperties`. ⏎  ⏎ TODO: ⏎ 1. Test performance on H20 ⏎  ⏎  ⏎ ## Modifications ⏎ 1. sgl-kernel/CMakeLists.txt ⏎ 2. sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu ⏎ 3. sgl-kernel/tests/test_fp8_blockwise_moe.py ⏎ 4. python/sglang/srt/layers/moe/cutlass_moe.py ⏎ 5. python/sglang/test/test_cutlass_moe.py ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ Test sgl-kernel/tests/test_fp8_blockwise_moe.py on H200: ⏎ <img width="3456" height="2112" alt="image" src="https://github.com/user-attachments/assets/5f668a70-16de-4ee1-a3a3-1367aaad2b88" /> ⏎ Test python/sglang/test/test_cutlass_moe.py on H200: ⏎ <img width="3456" height="2112" alt="image" src="https://github.com/user-attachments/assets/984829d5-94ae-4f53-be60-a834428237b6" /> ⏎  ⏎ ## Benchmarking and Profiling ⏎ Test sgl-kernel/benchmark/bench_fp8_blockwise_group_gemm.py on H200: ⏎ <img width="3456" height="2112" alt="image" src="https://github.com/user-attachments/assets/dcb3c881-4ccb-4b38-a5e6-b7c0c5a3889a" /> ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f92b729d52  (L1, 2025-08-25, sha f92b729d5240, PR #8328)
TITLE: [new feat] ascend backend support fia fusion kernel (#8328)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.routing.topk_py,L1.hardware.cpu_npu_musa; Ascend NPU top-k condition changed renormalize requirement for npu_moe_gating_top_k.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+1/-1); .github/workflows/pr-test-npu.yml (+3/-3); python/sglang/srt/layers/attention/ascend_backend.py (+218/-111); python/sglang/srt/mem_cache/memory_pool.py (+73/-14); python/sglang/srt/models/deepseek_v2.py (+9/-1); test/srt/ascend/test_ascend_mla_fia_w8a8int8.py (+103/-0); test/srt/ascend/test_ascend_mla_w8a8int8.py (+1/-0); test/srt/ascend/test_ascend_tp2_fia_bf16.py (+101/-0); test/srt/run_suite.py (+2/-0)
LABELS: high priority, ready-to-merge, npu
BODY: ## Motivation ⏎ In this MR, we implemented the NPU fusion kernels npu_fused_infer_attention_score in the Qwen2.5-7b, deepseek-v2-lite and deepseek-v3 models, this fusion kernel is  suitable for the graph mode. One needs to export **ASCEND_USE_FIA=ture** to activate this  fusion kernel. ⏎  ⏎ ## Modifications ⏎ Ascend Backend: support npu_fused_infer_attention_score kernel ⏎ Add unittest: test_ascend_tp_fia_bf16.py and test_ascend_mla_fia_w8a8int8.py ⏎  ⏎ > Exclusive support for paged attention, currently ONLY support page size 128. ⏎ > You can activate this feature by setting export ASCEND_USE_FIA=ture and --attention-backend ascend. ⏎  ⏎ ## Memory Management Advancement ⏎ We modify the AscendMLAPagedTokenToKVPool class, split the kvbuffer to the k_buffer and v_buffer, in order to remove the split op in MLA attention. ⏎  ⏎ ## Testing Framework ⏎ Unit tests for the Ascend attention backend have been added and can be found at: /test/srt/ascend/test_ascend_tp_fia_bf16.py and /test/srt/ascend/test_ascend_mla_fia_w8a8int8.py ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Accuracy and performance result ⏎ `python -m unittest test_npu_mla_backend.TestNpuMlaBackend.test_gsm8k` ⏎ <img width="1498" height="185" alt="1314ci2" src="https://github.com/user-attachments/assets/7d427455-804b-40f5-814c-4667c0d6f18e" /> ⏎  ⏎ ## Pre-commit check ⏎ <img width="400" height="181" alt="precom" src="https://github.com/user-attachments/assets/1727ab6c-c947-40cf-a6b4-b38a12df2d1a" />

### L1-79e6a8a6ac  (L1, 2025-08-26, sha 79e6a8a6acd8, PR #9495)
TITLE: support cuda 13.0 and trtllm kernel by Aug 25 2025 (#9495)
SOURCES: path_core, dependency_pin
STAGE1: repair_build_dependency; artifacts=L1.runner.marlin; CUDA 13/TRTLLM support changed Marlin MoE kernel generation/build files.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_softmax, L1.runner.marlin
FILES: sgl-kernel/CMakeLists.txt (+23/-9); sgl-kernel/csrc/moe/marlin_moe_wna16/generate_kernels.py (+25/-2); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel.h (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4b8.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku8b128.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4b8.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku8b128.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_marlin.cuh (+10/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/marlin_template.h (+2/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/ops.cu (+1/-0); sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu (+13/-3)
PERF_LINES: - fix cub::Sum cub::Max issue and let it support both cuda12x and 130
BODY: ## Motivation ⏎  ⏎ #9490  ⏎  ⏎ - Support cuda130 with custom flashinfer and trtllm kernel [Aug 25 2025](https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/80c7f8677a063859c07760dc98fe235db0ea22a8/fmha/) ⏎ - Support sm_110 and sm_121 on cuda 130 ⏎ - Support --compress-mode=size on cuda 130 ⏎ - Keep sm_101  support on cuda 128/129 ⏎  ⏎  ⏎ ### Test ⏎ - Step 1, use nvidia pytorch 25.08 image  ⏎ ```bash ⏎ docker pull nvcr.io/nvidia/pytorch:25.08-py3 ⏎ ``` ⏎  ⏎ - Step 2, run container with bash ⏎ - Step 3, git clone the change ⏎ - Step 4, comment out the torch dependency in sgl-kernel/pyproject.toml, patch like ⏎ ```bash ⏎ diff --git a/sgl-kernel/pyproject.toml b/sgl-kernel/pyproject.toml ⏎ index 52ee620e4..177e49e57 100644 ⏎ --- a/sgl-kernel/pyproject.toml ⏎ +++ b/sgl-kernel/pyproject.toml ⏎ @@ -1,7 +1,7 @@ ⏎  [build-system] ⏎  requires = [ ⏎    "scikit-build-core>=0.10", ⏎ -  "torch>=2.8.0", ⏎ +  # "torch>=2.8.0", ⏎    "wheel", ⏎  ] ⏎  ⏎ ``` ⏎  ⏎ - Step 5, patch the python/pyproject.toml with the torch version from the container, in my container the patch is like ⏎ ```bash ⏎ diff --git a/python/pyproject.toml b/python/pyproject.toml ⏎ index c23efbc2e..b29789d45 100644 ⏎ --- a/python/pyproject.toml ⏎ +++ b/python/pyproject.toml ⏎ @@ -49,7 +49,7 @@ runtime_common = [ ⏎      "scipy", ⏎      "timm==1.0.16", ⏎      "tiktoken", ⏎ -    "torchao==0.9.0", ⏎ +    "torchao==0.12.0+git", ⏎      "transformers==4.55.2", ⏎      "uvicorn", ⏎      "uvloop", ⏎ @@ -59,21 +59,19 @@ runtime_common = [ ⏎  srt = [ ⏎      "sglang[runtime_common]", ⏎      "sgl-kernel==0.3.5", ⏎ -    "torch==2.8.0", ⏎ -    "torchaudio==2.8.0", ⏎ +    "torch==2.8.0a0+34c6371d24.nv25.8", ⏎      "torchvision", ⏎      "cuda-python", ⏎ -    "flashinfer_python==0.2.11.post3", ⏎ +    "flashinfer_python==0.2.14.post1", ⏎  ] ⏎   ⏎  blackwell = [ ⏎      "sglang[runtime_common]", ⏎      "sgl-kernel", ⏎ -    "torch==2.8.0", ⏎ -    "torchaudio==2.8.0", ⏎ +    "torch==2.8.0a0+34c6371d24.nv25.8", ⏎      "torchvision", ⏎      "cuda-python", ⏎ -    "flashinfer_python==0.2.11.post3", ⏎ +    "flashinfer_python==0.2.14.post1", ⏎  ] ⏎ ``` ⏎  ⏎ - Step 6, install sgl-kernel ⏎  ⏎ ```bash ⏎ CUDA_VERSION=13.0 CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" SKBUILD_BUILD_DIR=./build CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"  pip install -v . ⏎ ``` ⏎  ⏎ - Step 7, install sglang ⏎ - Step 8, git clone [flashinfer with commit 018b5518](https://github.com/flashinfer-ai/flashinfer/commit/018b551825c8e5579206e6eb9d3229fa679202b3) and install with editable pkg ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - use custom flashinfer to support cuda130 and load trtllm kernel Aug 21 2025 ⏎ - fix cub::Sum cub::Ma …[truncated]

### L1-aa3eba8eb4  (L1, 2025-08-27, sha aa3eba8eb42c, PR #9340)
TITLE: [sgl-kernel] misc: update deepgemm version for sgl-kernel (#9340)
SOURCES: path_core, dependency_pin
STAGE1: repair_build_dependency; artifacts=L1.upstream.deepgemm,L1.runner.marlin,L1.ep.layer; DeepGEMM update changed sgl-kernel build and MoE/Marlin integration files.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_softmax, L1.runner.marlin, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-7); sgl-kernel/CMakeLists.txt (+48/-44); sgl-kernel/csrc/moe/marlin_moe_wna16/generate_kernels.py (+2/-25); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel.h (+0/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4.cu (+0/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4b8.cu (+0/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku8b128.cu (+0/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4.cu (+0/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4b8.cu (+0/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku8b128.cu (+0/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_marlin.cuh (+0/-10); sgl-kernel/csrc/moe/marlin_moe_wna16/marlin_template.h (+0/-2); sgl-kernel/csrc/moe/marlin_moe_wna16/ops.cu (+0/-1); sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu (+3/-13); .github/workflows/pr-test-sgl-kernel.yml (+2/-0); python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py (+133/-235); python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py (+5/-7); python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py (+5/-23); python/sglang/srt/layers/quantization/fp8_kernel.py (+2/-2); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-1); python/sglang/srt/layers/quantization/mxfp4_tensor.py (+3/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
LABELS: bug, enhancement, high priority, dependencies
BODY: ## Motivation ⏎  ⏎ DeepGEMM updated for unify cuda version. ⏎ So we need upd for TORCH LIBRARY. ⏎  ⏎ It depends on: https://github.com/sgl-project/sglang/pull/9167 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6b39f9cf8c  (L1, 2025-08-28, sha 6b39f9cf8c51, PR #9721)
TITLE: Support compile sgl-kernel on cuda 13.0 (#9721)
SOURCES: path_core
STAGE1: repair_build_dependency; artifacts=L1.runner.marlin; CUDA 13 build support changed Marlin MoE generated kernels and build metadata.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_softmax, L1.runner.marlin
FILES: sgl-kernel/csrc/moe/marlin_moe_wna16/generate_kernels.py (+25/-2); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel.h (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku4b8.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_bf16_ku8b128.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku4b8.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_fp16_ku8b128.cuh (+1/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/kernel_marlin.cuh (+10/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/marlin_template.h (+2/-0); sgl-kernel/csrc/moe/marlin_moe_wna16/ops.cu (+1/-0); sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu (+13/-3); sgl-kernel/CMakeLists.txt (+20/-9)
PERF_LINES: - fix cub::Sum cub::Max issue and let it support both cuda12x and 130
BODY: ## Motivation ⏎  ⏎  ⏎ #9490  ⏎ [PR 9495](https://github.com/sgl-project/sglang/pull/9495) ⏎  ⏎ - Support cuda130 with custom flashinfer and trtllm kernel [Aug 25 2025](https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/80c7f8677a063859c07760dc98fe235db0ea22a8/fmha/) ⏎ - Support sm_110 and sm_121 on cuda 130 ⏎ - Support --compress-mode=size on cuda 130 ⏎ - Keep sm_101  support on cuda 128/129 ⏎  ⏎  ⏎ ### Test ⏎ - Step 1, use nvidia pytorch 25.08 image  ⏎ ```bash ⏎ docker pull nvcr.io/nvidia/pytorch:25.08-py3 ⏎ ``` ⏎  ⏎ - Step 2, run container with bash ⏎ - Step 3, git clone the change ⏎ - Step 4, comment out the torch dependency in sgl-kernel/pyproject.toml, patch like ⏎ ```bash ⏎ diff --git a/sgl-kernel/pyproject.toml b/sgl-kernel/pyproject.toml ⏎ index 52ee620e4..177e49e57 100644 ⏎ --- a/sgl-kernel/pyproject.toml ⏎ +++ b/sgl-kernel/pyproject.toml ⏎ @@ -1,7 +1,7 @@ ⏎  [build-system] ⏎  requires = [ ⏎    "scikit-build-core>=0.10", ⏎ -  "torch>=2.8.0", ⏎ +  # "torch>=2.8.0", ⏎    "wheel", ⏎  ] ⏎  ⏎ ``` ⏎  ⏎ - Step 5, patch the python/pyproject.toml with the torch version from the container, in my container the patch is like ⏎ ```bash ⏎ diff --git a/python/pyproject.toml b/python/pyproject.toml ⏎ index c23efbc2e..b29789d45 100644 ⏎ --- a/python/pyproject.toml ⏎ +++ b/python/pyproject.toml ⏎ @@ -49,7 +49,7 @@ runtime_common = [ ⏎      "scipy", ⏎      "timm==1.0.16", ⏎      "tiktoken", ⏎ -    "torchao==0.9.0", ⏎ +    "torchao==0.12.0+git", ⏎      "transformers==4.55.2", ⏎      "uvicorn", ⏎      "uvloop", ⏎ @@ -59,21 +59,19 @@ runtime_common = [ ⏎  srt = [ ⏎      "sglang[runtime_common]", ⏎      "sgl-kernel==0.3.5", ⏎ -    "torch==2.8.0", ⏎ -    "torchaudio==2.8.0", ⏎ +    "torch==2.8.0a0+34c6371d24.nv25.8", ⏎      "torchvision", ⏎      "cuda-python", ⏎ -    "flashinfer_python==0.2.11.post3", ⏎ +    "flashinfer_python==0.2.14.post1", ⏎  ] ⏎   ⏎  blackwell = [ ⏎      "sglang[runtime_common]", ⏎      "sgl-kernel", ⏎ -    "torch==2.8.0", ⏎ -    "torchaudio==2.8.0", ⏎ +    "torch==2.8.0a0+34c6371d24.nv25.8", ⏎      "torchvision", ⏎      "cuda-python", ⏎ -    "flashinfer_python==0.2.11.post3", ⏎ +    "flashinfer_python==0.2.14.post1", ⏎  ] ⏎ ``` ⏎  ⏎ - Step 6, install sgl-kernel ⏎  ⏎ ```bash ⏎ CUDA_VERSION=13.0 CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" SKBUILD_BUILD_DIR=./build CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"  pip install -v . ⏎ ``` ⏎  ⏎ - Step 7, install sglang ⏎ - Step 8, git clone [flashinfer with commit 018b5518](https://github.com/flashinfer-ai/flashinfer/commit/018b551825c8e5579206e6eb9d3229fa679202b3) and install with editable pkg ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - use custom flashinfer to support cuda …[truncated]

### L1-74dd4249ac  (L1, 2025-08-28, sha 74dd4249ac60, PR #9355)
TITLE: [Feature] Support NPUGraph for DeepSeek on Ascend NPU (#9355)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.routing.topk_py,L1.ep.layer; Ascend NPUGraph support changed NPU TopK and EP MoE layer graph behavior.
ARTIFACT_HINTS: L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+12/-6); python/sglang/srt/layers/moe/topk.py (+12/-2); python/sglang/srt/disaggregation/ascend/conn.py (+75/-0); python/sglang/srt/layers/attention/ascend_backend.py (+183/-88); python/sglang/srt/layers/quantization/w8a8_int8.py (+7/-3); python/sglang/srt/mem_cache/memory_pool.py (+4/-0); python/sglang/srt/models/deepseek_v2.py (+14/-6)
LABELS: high priority, ready-to-merge, npu
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ this pr improves deepseek model (mla to be exact) performance with npugraph support ⏎ check initial npugraph support on mha model here #9399 and #8030 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - support concurrent kvcache transfer following mooncake transfer engine ⏎ - add mla graph support in ascend attention backend ⏎ - some performance and accuracy improvements ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ <img width="925" height="72" alt="87DF5544-5E77-4E8C-F895-71FDFDFEEEEB" src="https://github.com/user-attachments/assets/86274a5f-d4bf-4d2b-a4d3-04f208b36efc" /> ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ Reusing performance result [here](https://github.com/sgl-project/sglang/pull/8328#issuecomment-3222527169) ⏎ FIA result: sglang/test/srt/ascend/test_ascend_mla_fia_w8a8int8.py (tp == 2) ⏎ <img width="1542" height="191" alt="image" src="https://github.com/user-attachments/assets/c48da661-9c92-4272-95e6-f0d3a84b75b0" /> ⏎  ⏎ After this pr, with cuda graph enabled ⏎ ```bash ⏎ python3 -m sglang.launch_server --model DeepSeek-V2-Lite-W8A8 --tp 2 --trust-remote-code --quantization w8a8_int8  --device npu --attention-backend ascend --cuda-graph-bs 1 2 4 6 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 --disable-radix-cache --chunked-prefill-size 32768 --mem-fraction-static 0.8 ⏎ ``` ⏎ <img width="921" height="79" alt="D0A19C18-A5A9-4ED0-B5B6-8B5F1BCB3246" src="https://github.com/user-attachments/assets/2683a94c-ee77-4e74-a4ec-817fcfe58b4e" /> ⏎  ⏎  ⏎ ## Checklist

### L1-5ad296bda1  (L1, 2025-08-28, sha 5ad296bda140, PR #8750)
TITLE: Optimize prefill performance on cpu backend (#8750)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.hardware.cpu_npu_musa; CPU MoE kernels were optimized with AMX-int8 and thread blocking.
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/moe.cpp (+39/-69); sgl-kernel/csrc/cpu/moe_fp8.cpp (+51/-46); sgl-kernel/csrc/cpu/moe_int8.cpp (+334/-96); sgl-kernel/csrc/cpu/common.h (+118/-1); sgl-kernel/csrc/cpu/gemm.cpp (+30/-14); sgl-kernel/csrc/cpu/gemm.h (+4/-3); sgl-kernel/csrc/cpu/gemm_fp8.cpp (+34/-31); sgl-kernel/csrc/cpu/gemm_int8.cpp (+70/-12); sgl-kernel/csrc/cpu/qkv_proj.cpp (+1/-2)
LABELS: high priority, sgl-kernel, ready-to-merge, intel, cpu
PERF_LINES: gemm_bf16(native): 4.644 ms, gemm_fp8(opt): 4.375 ms, gemm_int8(opt): 8.945 ms, gemm_bf16(opt): 14.843 ms | gemm_bf16(native): 4.805 ms, gemm_fp8(opt): 3.483 ms, gemm_int8(opt): 2.048 ms, gemm_bf16(opt): 3.975 ms | ### fused_experts: M = 4, N = 384, K = 7168, E = 256, TopK = 8: bfloat16: 2.124 ms; int8: 1.057 ms; fp8: 1.066 ms | ### fused_experts: M = 3929, N = 384, K = 7168, E = 256, TopK = 8: bf
BODY: ## Motivation ⏎  ⏎ This PR aims at improving prefill performance for CPU backend, it will improve `bfloat16`, `int8_w8a8` and `fp8_w8a8` paths. This one has no effective on decoding performance. ⏎  ⏎ ## Modifications ⏎  ⏎ * **amx-int8**: enable **amx-int8** for GEMM and MoE kernels. As SGLang pops up with PyTorch 2.7, we are free to use amx-int8 on SGLang CPU backend. ⏎ * **thread blocking**: the original code base utilizes a simple blocking scheme for parallel, due to the limited time frame allowed for this project. Now i added more complexed blocking scheme. to use `parallel_2d` to replace `parallel_for` (which a 1d parallel template). Which means that for a 2d problem size of [M, N], the available threads will also be chunked to `num_threads_m * num_threads_n`, where the problem allocated to each thread will be as square as possible - so as to minimize memory load globally. ⏎ * **cache blocking**: use `loop_2d` template to alter the loop order from [MB, NB] to [NB / cache_blocks_nb, MB, cache_blocks_nb] -  so as to maximize the usage of L2 cache as much as possible. ⏎ * **fp8 dequant blocking**: fp8 dequant is super slow on CPU, so here I manage to dequant only once per thread and cache the bf16 weights in L2 and reuse it as much as possible. ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎ we can test with: ⏎ ```bash ⏎ python -m unittest -v test_gemm.py ⏎ python -m unittest -v test_moe.py ⏎ ``` ⏎  ⏎ this PR is purely backend performance improvement and has no effect on end2end accuracy. ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎ i tested the PR on Intel 4th gen Xeon, (codename SPR), single socket with 40 cores. (the optimization in this PR also applies to 5th and 6th gen Xeon, e.g. EMR and GNR). ⏎  ⏎ to reproduce the performance, use [sgl-cpu-test](https://github.com/mingfeima/sgl-cpu-tests) ⏎  ⏎ ```bash ⏎ ./run_bench_cpu.sh bench_gemm.py ⏎ ./run_bench_cpu.sh bench_moe.py ⏎ ``` ⏎  ⏎ ### Performance Improvement of GEMM ⏎ ``` ⏎ ### before ⏎ ### gemm_fp8 benchmark: M = 1024, N = 14336, K = 4096, has_bias = False ⏎ gemm_bf16(native): 4.644 ms, gemm_fp8(opt): 4.375 ms, gemm_int8(opt): 8.945 ms, gemm_bf16(opt): 14.843 ms ⏎  ⏎ ### after ⏎ ### gemm_fp8 benchmark: M = 1024, N = 14336, K = 4096, has_bias = False ⏎ gemm_bf16(native): 4.805 ms, gemm_fp8(opt): 3.483 ms, gemm_int8(opt): 2.048 ms, gemm_bf16(opt): 3.975 ms ⏎ ``` ⏎  ⏎ ### Performance Improvement of MoE ⏎ ``` ⏎ ### before ⏎ ### fused_experts: M = 4, N = 384, K = 7168, E = 256, TopK = 8: bfloat16: 2.124 ms; int8: 1.057 ms; fp8: 1.066 ms ⏎ ### fused_experts: M = 3929, N = 384, K = 7168, E = 256, TopK = 8: bfloat16: 73.836 ms; int8: 53.745 ms; fp8: 60.054 ms ⏎  ⏎ ### after ⏎ ### f …[truncated]

### L1-7a16db9bd9  (L1, 2025-08-28, sha 7a16db9bd9ab, PR #9789)
TITLE: Make sm100 fp8 kernels available on sm103 (#9789)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.cutlass.fp8_blockwise; CUTLASS FP8 blockwise MoE kernel made SM100 path available on SM103/B300.
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+6/-2); sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu (+5/-1); sgl-kernel/csrc/gemm/fp8_gemm_kernel.cu (+5/-1)
BODY: ## Motivation ⏎  ⏎ The tests failed on B300 without the fix. ⏎  ⏎ ## Accuracy Tests ⏎ On B300 ⏎ <img width="1718" height="618" alt="image" src="https://github.com/user-attachments/assets/c1dd80eb-c0ca-432f-8e27-4244d727115d" /> ⏎  ⏎  ⏎ ## Checklist

### L1-9970e3bf32  (L1, 2025-08-30, sha 9970e3bf328a, PR #9822)
TITLE: chore: upgrade sgl-kernel 0.3.7.post1 with deepgemm fix (#9822)
SOURCES: dependency_pin
STAGE1: repair_build_dependency; artifacts=L1.upstream.deepgemm; sgl-kernel bump is explicitly for a DeepGEMM fix.
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+2/-2); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5e194b2143  (L1, 2025-08-30, sha 5e194b21437f, PR #9824)
TITLE: [Model] Support Meituan LongCat-Flash && LongCat-Flash-MTP (#9824)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L1.routing.topk_py,L1.ep.layer; LongCat-Flash support changed TopK routing and EP MoE kernels.
ARTIFACT_HINTS: L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+74/-0); python/sglang/srt/layers/moe/topk.py (+23/-10); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/longcat_flash.py (+104/-0); python/sglang/srt/configs/model_config.py (+12/-0); python/sglang/srt/hf_transformers_utils.py (+2/-0); python/sglang/srt/layers/quantization/utils.py (+13/-0); python/sglang/srt/model_executor/model_runner.py (+4/-1); python/sglang/srt/models/longcat_flash.py (+1015/-0); python/sglang/srt/models/longcat_flash_nextn.py (+691/-0)
LABELS: enhancement, high priority
BODY: ## Motivation ⏎  ⏎ Support Meituan LongCat-Flash && LongCat-Flash-MTP ⏎  ⏎ ## Modifications ⏎  ⏎ [python/sglang/srt/models/longcat_flash.py](https://github.com/sgl-project/sglang/pull/9824/files/be711b71a60b8dd894c14b45ed2108ece803d6d8#diff-37eb91f71e6f2207973471f07977fb97e6ba216d834f8c490b689733d3950e7b) ⏎ [python/sglang/srt/models/longcat_flash_nextn.py](https://github.com/sgl-project/sglang/pull/9824/files/be711b71a60b8dd894c14b45ed2108ece803d6d8#diff-ea03a0b5b41ce627a0675e54d8a52d9374c6e6bd586fe318e9cfb4692e836741) ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d4a938417d  (L1, 2025-09-01, sha d4a938417d2c, PR #8118)
TITLE: [feat] Support tp mode for DeepSeek-R1-W4AFP8 (#8118)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.cutlass.w4a8,L1.cutlass.adapters; W4AFP8 DeepSeek TP mode adds Cutlass W4A8 kernel tiles and adapter routing.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.cutlass.w4a8, L1.ep.layer, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+1/-9); python/sglang/srt/layers/moe/ep_moe/layer.py (+0/-3); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-2); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh (+1/-1); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu (+206/-60); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh (+7/-6); python/sglang/srt/configs/model_config.py (+2/-1); python/sglang/srt/layers/quantization/w4afp8.py (+30/-25); python/sglang/srt/models/deepseek_v2.py (+5/-0); python/sglang/test/test_cutlass_w4a8_moe.py (+24/-9); sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py (+10/-4)
LABELS: high priority
PERF_LINES: Request throughput (req/s):              1.61 | Input token throughput (tok/s):          1610.09 | Output token throughput (tok/s):         1610.09 | Total token throughput (tok/s):          3220.18 | ----------------End-to-End Latency---------------- | Mean E2E Latency (ms):                   79201.47 | Median E2E Latency (ms):                 78956.21 | Mean TTFT (ms):                          6
BODY: ## Motivation ⏎  ⏎ Support tp mode for DeepSeek w4a8 model, which has a better performace than ep mode. ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Add W4AFp8MoEMethod and associated `create_weights`, `process_weights_after_loading` function and `apply` function. In the apply function, we use the same cutlass_w4a8_moe kernel as ep moe uses. ⏎ 2. Add some tile shape and cluster shape config for tp moe in `cutlass_w4a8_moe` kernel. ⏎ 3. Add a router logic in w4afp8 quant config and method. When "enable_ep_moe" found in global_server_args_dict, we use ep mode, else tp. ⏎  ⏎ Co-author: @yuhyao <827623970@qq.com> ⏎  ⏎ ## Benchmark ⏎ We run DeepSeek-R1-W4AFP8 on 8*H20 with tp8, comparing to run DeepSeek-R1 on 8*H20 with ep8. ⏎ Test configuration:  ⏎  ⏎ ### ISL1000, OSL1000 ⏎ input/output len = 1000/1000, qps=128, max_concurrency=128, num_prompt=256. ⏎ The results are shown below: ⏎  ⏎ TP ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang     ⏎ Traffic request rate:                    128.0      ⏎ Max request concurrency:                 128        ⏎ Successful requests:                     256        ⏎ Benchmark duration (s):                  159.00     ⏎ Total input tokens:                      256000     ⏎ Total generated tokens:                  256000     ⏎ Total generated tokens (retokenized):    254696     ⏎ Request throughput (req/s):              1.61       ⏎ Input token throughput (tok/s):          1610.09    ⏎ Output token throughput (tok/s):         1610.09    ⏎ Total token throughput (tok/s):          3220.18    ⏎ Concurrency:                             127.52     ⏎ ----------------End-to-End Latency---------------- ⏎ Mean E2E Latency (ms):                   79201.47   ⏎ Median E2E Latency (ms):                 78956.21   ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          6547.98    ⏎ Median TTFT (ms):                        6612.11    ⏎ P99 TTFT (ms):                           11687.82   ⏎ ---------------Inter-Token Latency---------------- ⏎ Mean ITL (ms):                           72.73      ⏎ Median ITL (ms):                         68.05      ⏎ P95 ITL (ms):                            72.42      ⏎ P99 ITL (ms):                            73.05      ⏎ Max ITL (ms):                            11148.65   ⏎ ================================================== ⏎ ``` ⏎  ⏎ While EP: ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang     ⏎ Traffic request rate:                    128.0      ⏎ Max request concurrency:                 128        ⏎ Successful requests: …[truncated]

### L1-b9eb0d9c2b  (L1, 2025-09-02, sha b9eb0d9c2bac, PR #9844)
TITLE: Change tensor alignment method to mn major (#9844)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.ep.layer; EP MoE layer changed tensor alignment method to mn-major.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-4)
BODY: ## Modifications ⏎  ⏎ Change tensor alignment method to mn major (from changes on #9340).

### L1-1db649ac02  (L1, 2025-09-02, sha 1db649ac0204, PR #9879)
TITLE: [feat] apply deep_gemm compile_mode to skip launch (#9879)
SOURCES: dependency_pin
STAGE1: optimize; artifacts=L1.runner.deep_gemm,L1.upstream.deepgemm; DeepGEMM compile_mode skip-launch support changed runner/dependency integration.
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+2/-2); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py (+8/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ do not merged until next version of sgl-kernel bumped ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-adf73175d6  (L1, 2025-09-05, sha adf73175d617, PR #9567)
TITLE: Forbid DeepEP racing condition when too many tokens (#9567)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.deepep_dispatcher; DeepEP dispatcher added guard against racing condition with too many tokens.
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+3/-0)
BODY: ## Motivation ⏎  ⏎ todo: pass CI ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0e78c63c0e  (L1, 2025-09-05, sha 0e78c63c0ec9, PR #10097)
TITLE: Revert "[1/N][Bug] Fix w4afp8 MoE NaN issue (sgl-kernel) (#9953)" (#10097)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts
STAGE1: revert; artifacts=L1.cutlass.w4a8; Reverts the W4AFP8 NaN kernel change in Cutlass W4A8 MoE.
ARTIFACT_HINTS: L1.cutlass.w4a8
FILES: sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh (+2/-2)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 9953 reason=ci_or_test_failure
BODY: This reverts commit f78b7fd16dbfe32c2ee73c1f3fef49fc1257b27f. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3fa62da78c  (L1, 2025-09-05, sha 3fa62da78c12, PR #9269)
TITLE: [7/N] MoE Refactor: the implementation of new framework (#9269)
SOURCES: path_core, symbol_pickaxe
STAGE1: introduce; artifacts=L1.runner.framework,L1.runner.triton,L1.ep.deepep_dispatcher; Introduces MoE runner framework with dispatch/core/combine split and Triton runner.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.runner.framework, L1.runner.triton, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/__init__.py (+2/-1); python/sglang/srt/layers/moe/fused_moe_native.py (+5/-3); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+5/-2); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+48/-29); python/sglang/srt/layers/moe/moe_runner/__init__.py (+2/-1); python/sglang/srt/layers/moe/moe_runner/base.py (+284/-1); python/sglang/srt/layers/moe/moe_runner/runner.py (+84/-0); python/sglang/srt/layers/moe/moe_runner/triton.py (+442/-0); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+16/-2); python/sglang/srt/layers/moe/token_dispatcher/base.py (+68/-7); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+29/-2); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+44/-2); python/sglang/srt/layers/moe/utils.py (+6/-4); python/sglang/srt/eplb/expert_distribution.py (+14/-9); python/sglang/srt/eplb/expert_location.py (+8/-3); python/sglang/srt/layers/quantization/awq.py (+19/-7); python/sglang/srt/layers/quantization/base_config.py (+11/-6); python/sglang/srt/layers/quantization/blockwise_int8.py (+38/-27); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+50/-30); python/sglang/srt/layers/quantization/fp8.py (+67/-39); python/sglang/srt/layers/quantization/gptq.py (+25/-17); python/sglang/srt/layers/quantization/modelopt_quant.py (+60/-35); python/sglang/srt/layers/quantization/moe_wna16.py (+21/-18); python/sglang/srt/layers/quantization/mxfp4.py (+64/-40); python/sglang/srt/layers/quantization/quark/quark_moe.py (+32/-27); python/sglang/srt/layers/quantization/unquant.py (+67/-43); python/sglang/srt/layers/quantization/w4afp8.py (+26/-17); python/sglang/srt/layers/quantization/w8a8_fp8.py (+35/-20); python/sglang/srt/layers/quantization/w8a8_int8.py (+71/-31); python/sglang/srt/managers/schedule_batch.py (+0/-1); python/sglang/srt/model_loader/__init__.py (+9/-3); python/sglang/srt/model_loader/loader.py (+18/-4); python/sglang/test/test_cutlass_moe.py (+24/-5); test/srt/test_mla_deepseek_v3.py (+37/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR implements a new MoE framework introduced in #8715 to streamline the integration of new all-to-all backends or grouped-gemm backends.  ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ One MoE forward function can be decoupled into three parts: dispatch -> pre-permute -> core runner -> post-permute -> combine. With this PR, a developer can implement dispatch/combine in the FusedMoE module and define their customized moe runner. The detailed workflow is illustrated in #8715. ⏎  ⏎ We offer two modes for implementing layer permutation: ⏎  ⏎ - Mode 1: `register_fused_func` for static fused_op for each pair of a2a x runner backend. This mode is compatible with torch.compile. ⏎ - Mode 2: `register_pre_permute` and `register_post_permute` for static runner + dynamic permutation functions. This mode ensures better extensibility. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a5a03209e9  (L1, 2025-09-06, sha a5a03209e959, PR #10107)
TITLE: Fix circular import (#10107)
SOURCES: path_core
STAGE1: repair_build_dependency; artifacts=L1.runner.framework,L1.runner.triton; MoE runner files were reorganized to fix circular imports.
ARTIFACT_HINTS: L1.runner.framework, L1.runner.triton, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/moe_runner/base.py (+8/-18); python/sglang/srt/layers/moe/moe_runner/runner.py (+1/-5); python/sglang/srt/layers/moe/moe_runner/triton.py (+10/-4); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+5/-5); python/sglang/srt/layers/quantization/w4afp8.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5f1eb20484  (L1, 2025-09-06, sha 5f1eb2048427, PR #9956)
TITLE: [chore] Remove unused ep_moe cuda kernels (#9956)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: remove; artifacts=L1.ep.reorder_aot; Deletes unused AOT EP MoE reorder/SwiGLU CUDA kernels.
ARTIFACT_HINTS: L1.ep.reorder_aot
FILES: sgl-kernel/CMakeLists.txt (+0/-2); sgl-kernel/csrc/common_extension.cc (+0/-12); sgl-kernel/csrc/moe/ep_moe_reorder_kernel.cu (+0/-181); sgl-kernel/csrc/moe/ep_moe_silu_and_mul_kernel.cu (+0/-115); sgl-kernel/include/sgl_kernel_ops.h (+0/-29); sgl-kernel/python/sgl_kernel/__init__.py (+0/-3); sgl-kernel/python/sgl_kernel/moe.py (+0/-64); sgl-kernel/benchmark/bench_moe_ep_post_reorder.py (+4/-22); sgl-kernel/benchmark/bench_moe_ep_pre_reorder.py (+0/-103); sgl-kernel/benchmark/bench_moe_silu_and_mul.py (+0/-92); sgl-kernel/tests/test_ep_moe_post_reorder_kernel.py (+0/-164); sgl-kernel/tests/test_ep_moe_pre_reorder_kernel.py (+0/-181); sgl-kernel/tests/test_ep_moe_silu_and_mul_kernel.py (+0/-142)
BODY: ## Motivation ⏎  ⏎ Clean up some unused kernels causing accuracy issues in https://github.com/sgl-project/sglang/issues/9944, benchmarks scripts, and unit tests (some of which fail on B200/B300 due to larger atol needed). ⏎  ⏎ ## Checklist

### L1-cb3918a091  (L1, 2025-09-07, sha cb3918a09127, PR #9477)
TITLE: Optimize moe_sum_reduce_kernel (#9477)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L1.triton.helper_kernels; Triton moe_sum_reduce helper kernel was optimized.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+23/-20); benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py (+24/-21)
PERF_LINES: This PR is to optimize moe_sum_reduce_kernel and get speedup up to 17.6%. | DeepSeek-V3 E2E throughput speedup 2-5%. | 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ | Latency: 68.177 s | Output throughput: 300.540 token/s | Second round on-ward, throughput increase 10%. | 100%|███████
BODY: ## Motivation ⏎  ⏎ During the Prefill stage of TP MoE, the moe_sum_reduce_kernel operator accounts for a relatively large proportion. ⏎ <img width="2996" height="1606" alt="image" src="https://github.com/user-attachments/assets/c49f154e-92dc-41a5-a0e5-dbf5fc645b87" /> ⏎  ⏎ This PR is to optimize moe_sum_reduce_kernel and get speedup up to 17.6%. ⏎ DeepSeek-V3 E2E throughput speedup 2-5%. ⏎  ⏎ ============Main============ ⏎ TritonKernel column is the concerned kernel. ⏎ ``` ⏎ Running correctness verification... ⏎ ✅ All implementations match ⏎  ⏎ Running performance benchmark... ⏎ sum_scaled_performance: ⏎     num_tokens    Original  TorchCompile  TritonKernel ⏎ 0          1.0    9.920000      5.936000      7.840000 ⏎ 1          2.0   10.368000      6.080000      8.032000 ⏎ 2          4.0   10.560000      6.432000      7.904000 ⏎ 3          8.0   10.752000      7.072000      8.128000 ⏎ 4         16.0   11.008000      8.704000      8.608000 ⏎ 5         32.0   11.360000     12.000000      8.960000 ⏎ 6         64.0   12.544000     18.015999      9.440000 ⏎ 7        128.0   15.807999     30.432001     10.688000 ⏎ 8        256.0   21.407999     55.328000     14.048000 ⏎ 9        512.0   34.688000    104.128003     21.888001 ⏎ 10      1024.0   58.591999    202.575997     35.840001 ⏎ 11      2048.0  101.712000    398.575991     59.519999 ⏎ 12      4096.0  189.536005    792.576015    104.032002 ⏎ ``` ⏎  ⏎ ============This PR============ ⏎ ``` ⏎ Running correctness verification... ⏎ ✅ All implementations match ⏎  ⏎ Running performance benchmark... ⏎ sum_scaled_performance: ⏎     num_tokens    Original  TorchCompile  TritonKernel ⏎ 0          1.0    9.536000      5.824000      6.688000 ⏎ 1          2.0    9.984000      5.856000      6.816000 ⏎ 2          4.0   10.144000      6.336000      6.784000 ⏎ 3          8.0   10.336000      6.976000      7.008000 ⏎ 4         16.0   10.592000      8.576000      7.328000 ⏎ 5         32.0   10.976000     11.616000      7.680000 ⏎ 6         64.0   12.128000     17.824000      8.480000 ⏎ 7        128.0   15.456000     30.336000     10.016000 ⏎ 8        256.0   21.088000     54.848000     14.304000 ⏎ 9        512.0   34.336001    104.128003     22.752000 ⏎ 10      1024.0   58.240000    202.207997     36.959998 ⏎ 11      2048.0  102.527998    400.736004     61.983999 ⏎ 12      4096.0  189.152002    791.040003    109.664001 ⏎ ``` ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎ ``` ⏎ $python3 -m sglang.launch_server --model /home/admin/DeepSeek-V3 --tp-size 8 --port 30000 --mem-fraction-static 0.9 ⏎  ⏎ This PR: ⏎  ⏎ First round warmup, is not high, ⏎ $pyth …[truncated]

### L1-5a7e10fe4c  (L1, 2025-09-07, sha 5a7e10fe4c1e, PR #10144)
TITLE: [MoE] fix: incorrect weight initialization for  cutlass_fused_experts_fp8 (#10144)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.cutlass.fp8_blockwise; Cutlass fused-experts FP8 MoE weight initialization was corrected.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+1/-1)
ISSUES: #10138 [Bug] Qwen3 w8a8 generate garbage output with all MoE backend
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ fix #10138  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-ee0b3c5bad  (L1, 2025-09-07, sha ee0b3c5bad6c, PR #10108)
TITLE: [1/N][Bug] Fix w4afp8 MoE NaN issue (sgl-kernel, fixed) (#10108)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.cutlass.w4a8; Fixed reland changes W4AFP8 grouped GEMM output type to resolve MoE NaNs.
ARTIFACT_HINTS: L1.cutlass.w4a8
FILES: sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh (+2/-2); sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py (+5/-6)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR provides the sgl-kernel changes for [PR#9918](https://github.com/sgl-project/sglang/pull/9918). It fixes [PR#9953](https://github.com/sgl-project/sglang/pull/9953). ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Update the w4afp8 grouped gemm kernel to use bfloat16 as the output type. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ See [PR#9918](https://github.com/sgl-project/sglang/pull/9918). ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ See [PR#9918](https://github.com/sgl-project/sglang/pull/9918). ⏎  ⏎ ## Checklist

### L1-2286e85e77  (L1, 2025-09-10, sha 2286e85e7758, PR #10241)
TITLE: pass a_scale from fp8 quant result instead of hard code to 1.0f  (#10241)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.cutlass.w4a8,L1.cutlass.adapters; W4A8 Cutlass MoE now passes real fp8 scale instead of hard-coded 1.0.
ARTIFACT_HINTS: L1.cutlass.w4a8, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+3/-3); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh (+1/-1); sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py (+30/-25)
PERF_LINES: test_cutlass_w4a8_moe_mm.py ............................................................................................................................... [ 31 | ........................................................................................................................................................... [ 69 | ..........................................................................
BODY: coauthor with @yicwang  @ayrnb  ⏎  ⏎ ## Motivation ⏎  ⏎ Issue: #10215  ⏎  ⏎ Fix the w4afp8 test case failures on hopper ⏎  ⏎ ## Modifications ⏎  ⏎ - Fix the w4afp8 kernel pass the quantization a_scale instead of hard coding to 1.0f ⏎ - Fix the failed test cases ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ - Unit tests ⏎  ⏎ ```bash ⏎  python test_cutlass_w4a8_moe_mm.py ⏎ ======================================================================= test session starts ======================================================================= ⏎ platform linux -- Python 3.12.11, pytest-8.4.1, pluggy-1.6.0 ⏎ rootdir: /sgl-workspace/sglang/sgl-kernel ⏎ configfile: pyproject.toml ⏎ plugins: typeguard-4.4.4, anyio-4.10.0 ⏎ collected 405 items                                                                                                                                                ⏎  ⏎ test_cutlass_w4a8_moe_mm.py ............................................................................................................................... [ 31%] ⏎ ........................................................................................................................................................... [ 69%] ⏎ ...........................................................................................................................                                 [100%] ⏎  ⏎ ======================================================================= 405 passed in 3.09s ======================================================================= ⏎ ``` ⏎  ⏎ ```bash ⏎ curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "/data01/models/DeepSeek-R1-W4AFP8", "temperature": 0, "messages": [ {"role": "user", "content": "How to travel to New York from San Francisco"}], "max_tokens": 200, "temperature": 0}' ⏎ {"id":"726e6ae588d944d38454749054b44dd4","object":"chat.completion","created":1757473837,"model":"/data01/models/DeepSeek-R1-W4AFP8","choices":[{"index":0,"message":{"role":"assistant","content":"Okay, so I need to figure out how to travel from San Francisco to New York. Let me start by thinking about the different ways people usually travel between cities. The main options are flying, driving, taking a train, or a bus. Maybe there are other less common methods too, like cycling or walking, but those seem impractical for such a long distance. Let me go through each option one by one.\n\nFirst, flying. That's probably the fastest and most common way. I should check how long the flight takes. I think a direct flight from San Francisco International Airport (SFO) to New York's airports like JF …[truncated]

### L1-5b64f006ec  (L1, 2025-09-10, sha 5b64f006ec2e, PR #9881)
TITLE: [Feature] Support DeepEP normal & Redundant Experts on NPU (#9881)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: extend_support; artifacts=L1.ep.deepep_dispatcher,L1.ep.layer; Adds NPU DeepEP normal and redundant-expert MoE support.
ARTIFACT_HINTS: L1.routing.topk_py, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/eplb/eplb_manager.py (+2/-2); python/sglang/srt/eplb/expert_distribution.py (+12/-4); python/sglang/srt/eplb/expert_location_updater.py (+1/-1); python/sglang/srt/layers/moe/ep_moe/layer.py (+108/-48); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+0/-2); python/sglang/srt/layers/moe/token_dispatcher/base.py (+0/-11); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+8/-35); python/sglang/srt/layers/moe/topk.py (+8/-0); .github/workflows/pr-test-npu.yml (+36/-0); .github/workflows/release-docker-npu-nightly.yml (+1/-0); .github/workflows/release-docker-npu.yml (+1/-3); python/sglang/srt/layers/attention/ascend_backend.py (+10/-3); scripts/ci/npu_ci_install_dependency.sh (+6/-0); test/srt/ascend/test_ascend_deepep.py (+121/-0); test/srt/run_suite.py (+3/-0)
LABELS: high priority
PERF_LINES: --deepep-mode low_latency \
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR adds support on Ascend NPU for: ⏎  ⏎ 1. DeepEP normal mode ⏎ 2. Redundant Experts ⏎  ⏎ Along with previously merged #8355, we are now allowing both prefill and decode to run with expert parallelism on Altas 800I A3. This also means running large-scale moe models without PD disaggregation is also possible if HBM capacity allows. ⏎  ⏎ Checkout our roadmap [here](https://github.com/sgl-project/sgl-kernel-npu/issues/47) about DeepEP-Ascend ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Fix a bug where FIA kernel argues not supporting `input_seq_len` smaller than `tp_size` ⏎ - Remove `AscendDeepEPLLOutput` due to DeepEP-Ascend now aligns output variables with DeepSeek's DeepEP ⏎ - Support intranode dispatch/combine (deepep normal mode) ⏎ - Support expert distribution recorder & redundant experts ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ <img width="1087" height="110" alt="image" src="https://github.com/user-attachments/assets/2e30aa08-3bcc-4e82-a1ce-e757dc03488b" /> ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ```shell ⏎ # Prefill ⏎ # NOTE: should increase the number of P instances as D is definitely not fullfilled ⏎ export HCCL_BUFFSIZE=1536 ⏎ python3 -m sglang.launch_server \ ⏎     --model-path <deepseek-model-path> \ ⏎     --trust-remote-code \ ⏎     --attention-backend ascend \ ⏎     --mem-fraction-static 0.85 \ ⏎     --quantization w8a8_int8 \ ⏎     --disable-radix-cache \ ⏎     --chunked-prefill-size 32768 \ ⏎     --tp-size 16 \ ⏎     --dp-size 1 \ ⏎     --ep-size 16 \ ⏎     --moe-a2a-backend deepep \ ⏎     --deepep-mode normal \ ⏎     --nnodes 1 \ ⏎     --node-rank 0 \ ⏎     --disaggregation-mode prefill \ ⏎     --disaggregation-transfer-backend ascend \ ⏎     --ep-num-redundant-experts 16 \ ⏎     --ep-dispatch-algorithm static \ ⏎     --init-expert-location <location-file> ⏎  ⏎ # Decode ⏎ export HCCL_BUFFSIZE=500 ⏎ export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32 ⏎ python3 -m sglang.launch_server \ ⏎     --model-path <deepseek-model-path> \ ⏎     --max-running-requests 512 \ ⏎     --trust-remote-code \ ⏎     --attention-backend ascend \ ⏎     --mem-fraction-static 0.9 \ ⏎     --quantization w8a8_int8 \ ⏎     --disable-radix-cache \ ⏎     --chunked-prefill-size 32768 \ ⏎     --cuda-graph-bs 8 16 24 32 \ ⏎     --tp-size 16 \ ⏎     --dp-size 2 \ ⏎     --enable-dp-attention \ ⏎     --ep-size 16 \ ⏎     --moe-a2a-backend deepep \ ⏎     --deepep-mode low_latency \ ⏎     --nnodes 1 \ ⏎     --node-rank 0 \ ⏎     --disaggregation-mode decode \ ⏎     --disaggregation-transfer-backend ascend \ ⏎     --ep-num-redundant-experts 16 \ ⏎     --ep-dispatch-algorithm static \ ⏎     --init-expert-location <location-file> ⏎ ``` ⏎ < …[truncated]

### L1-37367da639  (L1, 2025-09-10, sha 37367da6390f, PR #10299)
TITLE: [fix CI] Fix logical condition in fused MoE layer for compressed tensor quantization (#10299)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; Fused-MoE layer fixed compressed-tensor quantization enable condition.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-2)
LABELS: high priority
BODY: ### Background ⏎  ⏎ The bug is caused by https://github.com/sgl-project/sglang/pull/8118 cc @chenxijun1029  ⏎  ⏎ The current code in `fused_moe_triton/layer.py` has a logical operator precedence issue in the condition check for input scales validation (lines 615-620). The problematic condition was: ⏎  ⏎ ```python ⏎ if ( ⏎     "compressed" in self.quant_method.__class__.__name__.lower() ⏎     or "w4afp8" in self.quant_config.get_name() ⏎     and (param.data[expert_id] != 1).any() ⏎     and ((param.data[expert_id] - loaded_weight).abs() > 1e-5).any() ⏎ ): ⏎ ``` ⏎  ⏎ Due to operator precedence (`and` has higher precedence than `or`), this condition is actually parsed as: ⏎  ⏎ ```python ⏎ if ( ⏎     "compressed" in self.quant_method.__class__.__name__.lower() ⏎     or ( ⏎         "w4afp8" in self.quant_config.get_name() ⏎         and (param.data[expert_id] != 1).any() ⏎         and ((param.data[expert_id] - loaded_weight).abs() > 1e-5).any() ⏎     ) ⏎ ): ⏎ ``` ⏎  ⏎ This means that **any** model using compressed tensors quantization will unconditionally trigger the ValueError, regardless of whether the input scales are actually equal or not. ⏎  ⏎ ### Error Observed ⏎  ⏎ When loading models with compressed tensors quantization (e.g., `neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8`), the following error occurs: ⏎  ⏎ ``` ⏎ ValueError: input_scales of w1 and w3 of a layer must be equal. But got 1.0 vs. tensor([0.1011], device='cuda:1', dtype=torch.bfloat16) ⏎ ``` ⏎  ⏎ ### Solution ⏎  ⏎ Add parentheses to ensure correct operator precedence: ⏎  ⏎ ```python ⏎ if ( ⏎     ("compressed" in self.quant_method.__class__.__name__.lower() ⏎     or "w4afp8" in self.quant_config.get_name()) ⏎     and (param.data[expert_id] != 1).any() ⏎     and ((param.data[expert_id] - loaded_weight).abs() > 1e-5).any() ⏎ ): ⏎ ``` ⏎  ⏎ Now the condition correctly checks: ⏎ - The quantization method is either `compressed` OR `w4afp8`  ⏎ - AND the parameter data is not equal to 1 ⏎ - AND the difference between existing and loaded weights exceeds the threshold ⏎  ⏎ This ensures the validation only runs when appropriate and doesn't incorrectly fail for valid compressed tensor models. ⏎  ⏎ ### Testing ⏎  ⏎ This fix resolves the model loading failure for compressed tensor quantized models like `neuralmagic/Mixtral-8x7B-Instruct-v0.1-FP8` in the nightly GSM8K evaluation tests.

### L1-c5d2b01cea  (L1, 2025-09-11, sha c5d2b01cea6c, PR #10303)
TITLE: [LongCat] Optimize zero_experts_compute_triton by changing mask (#10303)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.ep.layer; EP MoE zero_experts_compute_triton mask changed for LongCat optimization.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+1/-1)
PERF_LINES: Regarding the zero expert proposed by longcat_flash, we found that there may be an optimization when masking the expert. The original masking of the ID to 0 may
BODY: update moe expert mask @Orchard-DT . Sincerely invite you to help check. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ Regarding the zero expert proposed by longcat_flash, we found that there may be an optimization when masking the expert. The original masking of the ID to 0 may cause confusion with the expert whose ID was originally 0. We modified this and found that the throughput of our model in the prefill phase increased by 10%. ⏎  ⏎ ## Modifications ⏎  ⏎ modify the expert mask in function `zero_experts_compute_triton` in sglang/python/srt/layer/moe/ep_moe/kernel.py. ⏎ ``` ⏎ #expert_indices[normal_expert_mask] = 0 ⏎ expert_indices[normal_expert_mask] = -1 ⏎ ``` ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3df05f4d6a  (L1, 2025-09-11, sha 3df05f4d6ab8, PR #9199)
TITLE: [NVIDIA] [3/N] Nvfp4 Masked Gemm: Add flashinfer grouped_gemm_nt_masked  (#9199)
SOURCES: path_core, symbol_pickaxe
STAGE1: integrate; artifacts=L1.runner.flashinfer_cutedsl,L1.ep.layer,L1.ep.deepep_dispatcher; Adds FlashInfer CuteDSL grouped_gemm_nt_masked MoE adapter and EP integration.
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+18/-0); python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py (+156/-0); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+2/-1); python/sglang/srt/layers/moe/utils.py (+4/-0); docs/references/environment_variables.md (+5/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+41/-1); python/sglang/srt/models/deepseek_v2.py (+6/-2); python/sglang/srt/server_args.py (+12/-0); python/sglang/test/test_fp4_moe.py (+370/-1); python/sglang/test/test_utils.py (+3/-0); test/srt/test_cutedsl_flashinfer_8gpu.py (+77/-0)
PERF_LINES: --enable-deepep-moe --deepep-mode low_latency | Latency: 288.874 s | Output throughput: 93.390 token/s | nemo-run_1/0 pass@1          | 500         | 5496       | 2760        | 98.20%           | 0.00% | Request throughput (req/s):              5.47 | Input token throughput (tok/s):          5597.82 | Output token throughput (tok/s):         5597.82 | Total token throughput (tok/s):          11195
BODY: @kaixih @kushanam @fzyzcjy ⏎  ⏎ ## Motivation ⏎  ⏎ Add  [grouped_gemm_nt_masked](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/cute_dsl/blockscaled_gemm.py#L2708) from flashinfer to support nvfp4 MoE. This PR exposes 2 APIs: `flashinfer_cutedsl_grouped_gemm_nt_masked` and `flashinfer_cutedsl_grouped_gemm_nt_masked`. ⏎ Depends on [9200](https://github.com/sgl-project/sglang/pull/9200/files) ⏎ The next step is to integrate into EpMoE.  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ ``` ⏎ SGLANG_DEEPEP_BF16_DISPATCH=true python3 -m sglang.launch_server \ ⏎   --model-path nvidia/DeepSeek-R1-0528-FP4 \ ⏎   --trust-remote-code \ ⏎   --disable-radix-cache \ ⏎   --max-running-requests 256 \ ⏎   --chunked-prefill-size 1024 \ ⏎   --mem-fraction-static 0.89 \ ⏎   --max-prefill-tokens 16384 \ ⏎   --disable-cuda-graph \ ⏎   --tp 8 \ ⏎   --dp 8 \ ⏎   --enable-dp-attention \ ⏎   --load-format dummy \ ⏎   --enable-ep-moe \ ⏎   --quantization modelopt_fp4 \ ⏎   --enable-flashinfer-cutedsl-moe \ ⏎   --enable-deepep-moe --deepep-mode low_latency  ⏎ python3 benchmark/gsm8k/bench_sglang.py   --num-questions 256   --parallel 32   --num-shots 8 ⏎ ``` ⏎ Accuracy: 0.980 ⏎ Invalid: 0.000 ⏎ Latency: 288.874 s ⏎ Output throughput: 93.390 token/s ⏎  ⏎  ⏎ math-500 data-set: ⏎ this PR ⏎ ``` ⏎ 64k output-len ⏎ nemo-run_1/0 --------------------------------------- math-500 -------------------------------------- ⏎ nemo-run_1/0 evaluation_mode | num_entries | avg_tokens | gen_seconds | symbolic_correct | no_answer ⏎ nemo-run_1/0 pass@1          | 500         | 5496       | 2760        | 98.20%           | 0.00%     ⏎ nemo-run_1/0  ⏎ ``` ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎ ``` ⏎ python3 -m sglang.bench_serving \ ⏎   --model nvidia/DeepSeek-R1-0528-FP4 \ ⏎   --dataset-name random \ ⏎   --backend sglang-oai \ ⏎   --random-range-ratio 1 \ ⏎   --random-input-len 1024 \ ⏎   --random-output-len 1024 \ ⏎   --max-concurrency 256 \ ⏎   --num-prompts 512 \ ⏎   --base-url http://127.0.0.1:30000 ⏎ ``` ⏎ ``` ⏎ This PR ⏎  ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang-oai ⏎ Traffic request rate:                    inf        ⏎ Max request concurrency:                 256        ⏎ Successful requests:                     512        ⏎ Benchmark duration (s):                  93.66      ⏎ Total input tokens:                      524288     ⏎ Total generated tokens:                  524288     ⏎ Total generated tokens (retokenized):    521116     ⏎ Request throughput (req/s):              5.47       ⏎ Input token throughput (tok/s):          5597.82    ⏎ Output token throughput (tok/s):         5597. …[truncated]

### L1-c7e85f5378  (L1, 2025-09-11, sha c7e85f537870, PR #10296)
TITLE: fix: flashinfer_cutlass_moe: Use max of global expert scales instead of local for input scale (#10296)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.runner.flashinfer_cutlass; FlashInfer Cutlass MoE input scale now uses max global expert scale.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+2/-2)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ To fix accuracy issues. ⏎  ⏎ ## Modifications ⏎  ⏎ This matches trt-llm usage ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ SGL_ENABLE_JIT_DEEPGEMM=0 python -m sglang.launch_server   --max-running-requests 1024   --disable-radix-cache   --disable-shared-experts-fusion   --tp-size 8   --dp-size 8   --ep-size 8   --enable-dp-attention   --chunked-prefill-size $((4096 * 8))   --moe-dense-tp-size 1   --enable-dp-lm-head   --model-path nvidia/DeepSeek-R1-0528-FP4   --trust-remote-code   --port 40001   --mem-fraction-static 0.84   --quantization modelopt_fp4   --disable-chunked-prefix-cache   --attention-backend trtllm_mla   --moe-runner-backend flashinfer_cutlass  ⏎ ``` ⏎ Both ran using cherry-pick https://github.com/sgl-project/sglang/pull/10178 ⏎  ⏎ GPQA before ⏎ ``` ⏎ ./nemo_skills_output_20250911042610_4040570/eval-results/gpqa/metrics.json ⏎       "symbolic_correct": 66.66666666666667, ⏎ ./nemo_skills_output_20250911042610_4040570/eval-results/metrics.json ⏎       "symbolic_correct": 66.66666666666667, ⏎ ./nemo_skills_output_20250911042613_6210320/eval-results/gpqa/metrics.json ⏎       "symbolic_correct": 65.65656565656566, ⏎ ./nemo_skills_output_20250911042613_6210320/eval-results/metrics.json ⏎       "symbolic_correct": 65.65656565656566, ⏎ ./nemo_skills_output_20250911042616_4874585/eval-results/gpqa/metrics.json ⏎       "symbolic_correct": 79.29292929292929, ⏎ ./nemo_skills_output_20250911042616_4874585/eval-results/metrics.json ⏎       "symbolic_correct": 79.29292929292929, ⏎ ./nemo_skills_output_20250911042619_3060768/eval-results/gpqa/metrics.json ⏎       "symbolic_correct": 74.74747474747475, ⏎ ./nemo_skills_output_20250911042619_3060768/eval-results/metrics.json ⏎       "symbolic_correct": 74.74747474747475, ⏎ ./nemo_skills_output_20250911042622_5684787/eval-results/gpqa/metrics.json ⏎       "symbolic_correct": 80.8080808080808, ⏎ ./nemo_skills_output_20250911042622_5684787/eval-results/metrics.json ⏎       "symbolic_correct": 80.8080808080808, ⏎ ./nemo_skills_output_20250911042625_1518254/eval-results/gpqa/metrics.json ⏎       "symbolic_correct": 72.22222222222223, ⏎ ./nemo_skills_output_20250911042625_1518254/eval-results/metrics.json ⏎       "symbolic_correct": 72.22222222222223, ⏎ ./nemo_skills_output_20250911042628_7418138/eval-results/gpqa/metrics.json ⏎       "symbolic_correct": 75.25252525252525, ⏎ ./nemo_skills_output_20250911042628_7418138/eval-results/metrics.json ⏎       "symbolic_correct": 75.25252525252525, ⏎ ./nemo_skills_output_20250911042631_2525420/eval-results/gpqa/metrics.json ⏎       "symbolic_correct": 76.76767676767676, ⏎ ./nemo_skills_output_20250911042631_2525 …[truncated]

### L1-4aa39d72c4  (L1, 2025-09-11, sha 4aa39d72c422, PR #10356)
TITLE: fix the break in FlashInferFusedMoE (#10356)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; Fused-MoE layer fix repaired FlashInferFusedMoE breakage.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-2)
BODY: This PR fixed the break due to the recent changes to API apply_with_router_logits() in https://github.com/sgl-project/sglang/pull/9269.

### L1-2df532ef20  (L1, 2025-09-14, sha 2df532ef20cb, PR #10369)
TITLE: Fix the global scale fix does not support EPLB and improve enabling condition (#10369)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; Fused-MoE global-scale fix adjusted EPLB enable condition.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-10); python/sglang/srt/layers/quantization/modelopt_quant.py (+2/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fa46e2bd40  (L1, 2025-09-14, sha fa46e2bd4003, PR #9948)
TITLE: Support offloading in fp8 (#9948)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.ep.layer; EP MoE layer gained FP8 offloading support.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+52/-10); python/sglang/srt/layers/quantization/fp8_utils.py (+7/-2); python/sglang/srt/models/deepseek_v2.py (+9/-2); python/sglang/srt/offloader.py (+27/-3)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ (multi code change to be extracted) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2f8ba6fe82  (L1, 2025-09-14, sha 2f8ba6fe82be, PR #10429)
TITLE: [Fix] MoE: fix w8a8_fp8 MoE and add tests to cover this code path (#10429)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=NEW:w8a8_fp8_quant_moe_adapter; w8a8_fp8 quantized MoE code path was fixed and covered by tests.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/w8a8_fp8.py (+1/-1); test/srt/quant/test_w8a8_quantization.py (+43/-7)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR fixes the issue mentioned in [this comment](https://github.com/sgl-project/sglang/pull/9269#discussion_r2342544472). It also adds two cases to cover E2E w8a8_fp8 test. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-258d02c86d  (L1, 2025-09-14, sha 258d02c86d93, PR #10426)
TITLE: Fix correction bias undefined behavior for nvfp4 models (#10426)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.routing.fused_gate; moe_fused_gate now guards correction-bias undefined behavior for NVFP4 models.
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/csrc/moe/moe_fused_gate.cu (+2/-0); python/sglang/srt/models/deepseek_v2.py (+3/-1)
BODY: ## Motivation ⏎  ⏎ TODO tomorrow: (1) add assertions at kernel level (2) check shall we make it bf16 etc (3) test e2e ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4844fac91d  (L1, 2025-09-14, sha 4844fac91d07, PR #9338)
TITLE: Refactor TopK to ensure readability and extensibility (#9338)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.routing.topk_py; TopK routing abstraction was refactored to centralize recent fixes and extensibility.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-4); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-10); python/sglang/srt/layers/moe/topk.py (+30/-9); python/sglang/srt/managers/schedule_batch.py (+0/-1); python/sglang/srt/models/bailing_moe.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+7/-12); python/sglang/srt/models/ernie4.py (+1/-1); python/sglang/srt/models/glm4_moe.py (+1/-1); python/sglang/srt/models/gpt_oss.py (+1/-1); python/sglang/srt/models/longcat_flash.py (+2/-2); python/sglang/srt/models/qwen2_moe.py (+1/-1); python/sglang/srt/models/qwen3_moe.py (+1/-1); python/sglang/srt/models/qwen3_next.py (+2/-2); python/sglang/srt/models/step3_vl.py (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Some recent fixes and optimizations are hardcoded in `deepseek_v2.py`. This PR slightly adjust the code structure and rename some variables to ensure readability and extensibility. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-059c13de5c  (L1, 2025-09-15, sha 059c13de5cba, PR #10440)
TITLE: Fix trtllm_moe wrong correction bias (#10440)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.routing.topk_py; TRTLLM MoE correction-bias handling moved out of TopK to avoid uninitialized values.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+0/-8); python/sglang/srt/models/deepseek_v2.py (+14/-2)
LABELS: run-ci
DEEP_STUDY: deep-study correctness case sglang:059c13de5c: class=memory_safety_oob; symptom=nan_inf; introducing=unknown
BODY: ## Motivation ⏎  ⏎ before fix: comes from torch.empty, can be zero or even nan ⏎  ⏎ <img width="1740" height="410" alt="image" src="https://github.com/user-attachments/assets/94d11287-0447-4ef1-8b71-88a8257ddafd" /> ⏎  ⏎ after fix: ⏎  ⏎ <img width="1143" height="305" alt="image" src="https://github.com/user-attachments/assets/10c57390-0f09-4ece-a02a-6cd8064a0e98" /> ⏎  ⏎ (trtllm_gen requires bf16 dtype, though we know it is good to be fp32 since the value diff is small) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9b876889b7  (L1, 2025-09-16, sha 9b876889b7d1, PR #10491)
TITLE: Update CUTLASS. Refine KernelSchedule for fp8 (grouped) gemm. (#10491)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.cutlass.fp8_blockwise; CUTLASS FP8 blockwise MoE kernel schedule was refined to CUTLASS grouped-GEMM examples.
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+3/-3); sgl-kernel/CMakeLists.txt (+1/-1); sgl-kernel/csrc/cutlass_extensions/gemm/fp8_blockwise_gemm_sm90_dispatch.cuh (+1/-1)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎ Updated CUTLASS to tag version 4.2. Aligned the fp8 (grouped) gemm kernel schedule to CUTLASS Example: ⏎  ⏎ - https://github.com/NVIDIA/cutlass/tree/main/examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling ⏎ - https://github.com/NVIDIA/cutlass/tree/main/examples/68_hopper_fp8_warp_specialized_grouped_gemm_with_blockwise_scaling ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - sgl-kernel/CMakeLists.txt ⏎ - sgl-kernel/csrc/cutlass_extensions/gemm/fp8_blockwise_gemm_sm90_dispatch.cuh ⏎ - sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e07b21ceaf  (L1, 2025-09-18, sha e07b21ceaf1b, PR #10624)
TITLE: update deepep version for qwen3-next deepep moe (#10624)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L1.upstream.deepep; DeepEP version updated for Qwen3-Next 512-expert MoE support.
ARTIFACT_HINTS: -
FILES: scripts/ci/ci_install_deepep.sh (+1/-1); docker/Dockerfile (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Support qwen3-next expert number = 512 since https://github.com/deepseek-ai/DeepEP/pull/403 ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-388c05d544  (L1, 2025-09-18, sha 388c05d54435, PR #10579)
TITLE: Fix bias handling in TritonMoeQuantInfo within quantization/mxfp4.py (#10579)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; TritonMoeQuantInfo bias fields were corrected for MXFP4 MoE quantization.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4.py (+2/-2)
LABELS: ready-to-merge, run-ci
BODY: ## Motivation ⏎  ⏎ This PR fixes a field mismatch when constructing `TritonMoeQuantInfo` in `quantization/mxfp4.py`. ⏎ - Issue: The call site passed `w13_weight_bias` and `w2_weight_bias`, but `TritonMoeQuantInfo` defines these as optional fields `b13` and `b2`. This could raise a TypeError and/or leave biases unwired. ⏎ - Fix: Pass biases via `b13` and `b2`, sourcing them from the layer with `getattr(..., None)` to safely handle layers without bias parameters. ⏎ - Impact: Prevents constructor errors and correctly wires optional biases in the MoE quantization path. No behavioral change when biases are absent; scope limited to `quantization/mxfp4.py`. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4039c626e2  (L1, 2025-09-19, sha 4039c626e27a, PR #8274)
TITLE: fix deepep assert when PD disaggregation == null (#8274)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.deepep_dispatcher; DeepEP dispatcher avoids upstream assert when PD disaggregation is null.
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+11/-2)
BODY: when `--pd-disaggregation` == `null` deepep will assert:  ⏎  ⏎ https://github.com/deepseek-ai/DeepEP/blob/bdd119f8b249953cab366f4d737ad39d4246fd7e/csrc/kernels/internode.cu#L386 ⏎  ⏎ deepep PR: https://github.com/deepseek-ai/DeepEP/pull/181 ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-616a3e20df  (L1, 2025-09-19, sha 616a3e20df5f, PR #10321)
TITLE: [sgl-kernel] Support moe_sum_reduce cuda kernel (#10321)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: introduce; artifacts=L1.reduce.moe_sum_reduce; Introduces CUDA moe_sum_reduce kernel replacing Triton/TorchCompile combine for MoE.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.reduce.moe_sum_reduce
FILES: sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/csrc/common_extension.cc (+2/-0); sgl-kernel/csrc/moe/moe_sum_reduce.cu (+303/-0); sgl-kernel/include/sgl_kernel_ops.h (+2/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/python/sgl_kernel/moe.py (+12/-0); benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py (+25/-10)
LABELS: run-ci
PERF_LINES: In bs=4096, the memory bandwidth of the Triton is nearly 3TB/s, 75% throughput which is close to the theoretical upper limit. That said, the Cuda Kernel is bett
BODY: ## Motivation ⏎  ⏎  ⏎ moe_sum_reduce kernel in MoE is a Triton kernel (combine with TorchCompile version for small batch). This PR is to rewrite it in CUDA for better performance. ⏎  ⏎ Currently the CudaKernel outperforms both TorchCompile and TritonKernel in small num_tokens. In large token_num the performs has not drop. ⏎ ``` ⏎ root@blackwell-research:/sgl-workspace/sglang_dev# python ./benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py ⏎ Running correctness verification... ⏎  ⏎ ✅ All implementations match ⏎  ⏎ Running performance benchmark... ⏎ sum_scaled_performance: ⏎     num_tokens    Original  TorchCompile  TritonKernel  CudaKernel ⏎ 0          1.0   14.368000     20.160001     10.240000    7.488000 ⏎ 1          2.0   14.112000     24.256000     10.080000    7.456000 ⏎ 2          4.0   14.464000     24.064001     10.112000    7.680000 ⏎ 3          8.0   14.624000     23.647999     10.240000    8.256000 ⏎ 4         16.0   15.888000     23.968000     10.176000    8.160000 ⏎ 5         32.0   15.488000     21.888001     10.304000    9.504000 ⏎ 6         64.0   16.416000     27.712001     10.368000   10.208000 ⏎ 7        128.0   16.448000     42.208001     12.192000   12.224000 ⏎ 8        256.0   20.288000     75.680003     12.320000   14.208000 ⏎ 9        512.0   26.591999    143.135995     17.312000   17.408000 ⏎ 10      1024.0   41.760001    278.495997     26.496001   26.784001 ⏎ 11      2048.0   66.848002    547.936022     40.160000   40.128000 ⏎ 12      4096.0  114.335999   1087.664008     68.704002   68.576001 ⏎ ```  ⏎  ⏎ In bs=4096, the memory bandwidth of the Triton is nearly 3TB/s, 75% throughput which is close to the theoretical upper limit. That said, the Cuda Kernel is better than TritonKernel in lower batch, and be equal to TritonKernel in large batch which shows solid improvement. ⏎  ⏎ <img width="3254" height="1696" alt="image" src="https://github.com/user-attachments/assets/22b735d3-eef8-4d6e-ac4e-152db87368d9" /> ⏎  ⏎ <img width="3352" height="1692" alt="image" src="https://github.com/user-attachments/assets/1a04515c-0c4c-4075-a05c-771aff9f4351" /> ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9c53dad809  (L1, 2025-09-22, sha 9c53dad80993, PR #10758)
TITLE: Fix MTP MoE weight loading with NVFP4 target model. (#10758)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; Fused-MoE layer fixed MTP NVFP4 target-model weight loading.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-1)
LABELS: bug, high priority, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0c9174108a  (L1, 2025-09-28, sha 0c9174108abe, PR #10701)
TITLE: Unify SGL Kernel Releases (#10701)
SOURCES: dependency_pin
STAGE1: repair_build_dependency; artifacts=L1.align.cuda_aot,L1.routing.topk_softmax; Unify SGL Kernel Releases
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+50/-12); .github/workflows/pr-test.yml (+13/-17); sgl-kernel/python/sgl_kernel/__init__.py (+178/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Dynamically load common_ops module for SM90 (compiled with fast math) and SM 100 (without fast math) during initialization based on compute architecture so that one SGLang kernel binary can cover all versions with different configurations.  ⏎  ⏎ CMake will compile two common_ops libraries and install them in two different locations. ⏎  ⏎ Update images used in CI to be 12.9. Remove the steps to build kernel for 12.4 and 12.8. ⏎  ⏎ ## Tests ⏎  ⏎ Ran CI before updating the image (using [12.4](https://github.com/sgl-project/sglang/pull/10701/checks?sha=9c87ad5c8ed13d67b9f09e0791ca6cac45900b4e)) and after updating image ([12.9](https://github.com/sgl-project/sglang/pull/10701/checks?sha=09d873b2e9459aa9954edeb020dbcdebdd67a515)) ⏎  ⏎ ## Checklist

### L1-1237aa19ce  (L1, 2025-09-30, sha 1237aa19ce63, PR #11099)
TITLE: [Auto Sync] Update fused_moe_triton_config.py (20250930) (#11099)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: adapt_framework; artifacts=L1.upstream.vllm.fused_topk,L1.triton.fused_moe; [Auto Sync] Update fused_moe_triton_config.py (20250930)
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py (+6/-2)
LABELS: run-ci
BODY: Sync changes from commit `d078e69d`. ⏎  ⏎ **Files Changed:** ⏎ - python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py ⏎  ⏎ Author: Cheng Wan <54331508+ch-wan@users.noreply.github.com> ⏎  ⏎ --- ⏎  ⏎ *This is an automated PR created by scripts/copy_from_oss.py.*

### L1-a6cc86df9d  (L1, 2025-09-30, sha a6cc86df9d3e, PR #11081)
TITLE: Fix DSR1 accuracy for flashinfer_trtllm MoE with FP8 quantization (#11081)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.runner.flashinfer_trtllm; Fix DSR1 accuracy for flashinfer_trtllm MoE with FP8 quantization
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-3); python/sglang/srt/server_args.py (+1/-1)
LABELS: run-ci
PERF_LINES: Latency: 73.347 s | Output throughput: 9207.244 token/s | Latency: 25.951 s | Output throughput: 5488.445 token/s
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/pull/10758 introduced a change to only apply w13 -> w31 weight mapping with flashinfer_trtllm for NVFP4 quantization, however FP8 quantization is also supported and needs this mapping too, otherwise GSM8K accuracy went to 0. ⏎  ⏎ ## Modifications ⏎  ⏎ Apply w13 -> w31 for FP8 quantization also. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Command ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-0528  --trust-remote-code --tp 8 --moe-runner-backend flashinfer_trtllm --quantization fp8 ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319 --port=30000 ⏎ ``` ⏎  ⏎ Before fix ⏎ ``` ⏎ Accuracy: 0.025 ⏎ Invalid: 0.067 ⏎ Latency: 73.347 s ⏎ Output throughput: 9207.244 token/s ⏎ ``` ⏎  ⏎ After fix ⏎ ``` ⏎ Accuracy: 0.959 ⏎ Invalid: 0.000 ⏎ Latency: 25.951 s ⏎ Output throughput: 5488.445 token/s ⏎ ``` ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ N/A ⏎  ⏎ ## Checklist

### L1-0b9dfba787  (L1, 2025-10-02, sha 0b9dfba78700, PR #10263)
TITLE: Support dispatch low latency (#10263)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.runner.flashinfer_cutedsl,L1.ep.layer,L1.ep.deepep_dispatcher; Support dispatch low latency
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+11/-0); python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py (+38/-25); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+19/-3); python/sglang/srt/layers/quantization/modelopt_quant.py (+11/-1); python/sglang/srt/models/deepseek_v2.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ previous PR: https://github.com/sgl-project/sglang/pull/10120 ⏎ co-author: @kaixih (who did 10120) ⏎  ⏎ (for the one who merges this PR - please add @kaixih as co-author since he did 10120) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5e786cca3a  (L1, 2025-10-02, sha 5e786cca3a9a, PR #10422)
TITLE: Support single batch overlap (#10422)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.runner.flashinfer_cutedsl,L1.ep.layer,L1.ep.deepep_dispatcher; Support single batch overlap
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+16/-3); python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py (+14/-0); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+47/-14); python/sglang/srt/layers/moe/utils.py (+10/-0); docs/advanced_features/server_arguments.md (+1/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+11/-0); python/sglang/srt/models/deepseek_v2.py (+12/-3); python/sglang/srt/server_args.py (+6/-0); python/sglang/srt/single_batch_overlap.py (+151/-0)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ This PR is mainly for my case (nvlink deepep and cutedsl gemm), while https://github.com/sgl-project/sglang/pull/9660 focus on H20 (rdma deepep and deepgemm). I try to make the code somehow general s.t. after this one is merged, 9660 can be merged without code duplication. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-12d6818380  (L1, 2025-10-02, sha 12d681838048, PR #11130)
TITLE: Tiny fix ep_gather behavior different in CI (#11130)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.ep.layer; Tiny fix ep_gather behavior different in CI
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist
