### L3-59f935300c  (L3, 2025-07-19, sha 59f935300c48, PR #21196)
TITLE: [BugFix] Fix potential cuda-graph IMA (#21196)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.dispatch.abstract_interface; repair correctness in L3.dispatch.abstract_interface; [BugFix] Fix potential cuda-graph IMA (#21196)
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+0/-5); vllm/v1/worker/gpu_model_runner.py (+6/-1)
LABELS: bug, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Cuda-graph padding happens after prepare inputs; it's safer to -1 fill here (and closer behavior to pre https://github.com/vllm-project/vllm/pull/20466 ). No errors reported yet this is just preventative ⏎  ⏎ ## Test Plan ⏎  ⏎ lm_eval ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1  lm_eval   --model vllm   --model_args '{ ⏎     "pretrained": "meta-llama/Meta-Llama-3-8B-Instruct", ⏎     "compilation_config": { "full_cuda_graph": true } ⏎   }'   --tasks gsm8k   --batch_size auto  ⏎ ... ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7559|±  |0.0118| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.7574|±  |0.0118| ⏎ ``` ⏎  ⏎ ## (Optional) Documentation Update

### L3-752c6ade2e  (L3, 2025-07-19, sha 752c6ade2e0f, PR #21217)
TITLE: [V0 Deprecation] Deprecate BlockSparse Attention & Phi3-Small (#21217)
SOURCES: path_core
STAGE1: remove; artifacts=L3.xformers.v0_backend,L3.flash_attn.v0_backend,L3.flash_attn.v1_backend,L3.flashinfer.v0_backend,L3.flashinfer.v1_backend,L3.flashinfer.trtllm_gen,L3.flashinfer.trtllm_xqa_decode,L3.triton.v1_backend,L3.rocm.rocm_flash_attn_v0,L3.rocm.aiter_fa,L3.mla.triton_v0,L3.mla.common_v1,L3.mla.flashmla_v0_adapter,L3.mla.flashmla_v1_adapter,L3.mla.cutlass_v1_backend,L3.mla.rocm_aiter,L3.dispatch.selector,L3.dispatch.abstract_interface,L3.flex_attention,L3.blocksparse.v0; remove in L3.xformers.v0_backend; [V0 Deprecation] Deprecate BlockSparse Attention & Phi3-Small (#21217)
ARTIFACT_HINTS: L3.xformers.v0_backend, L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.rocm_flash_attn_v0, L3.rocm.aiter_fa, L3.mla.triton_v0, L3.mla.common_v1, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.rocm_aiter, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.flex_attention, L3.blocksparse.v0
FILES: vllm/attention/backends/abstract.py (+0/-1); vllm/attention/backends/blocksparse_attn.py (+0/-466); vllm/attention/backends/differential_flash_attn.py (+0/-4); vllm/attention/backends/dual_chunk_flash_attn.py (+0/-1); vllm/attention/backends/flash_attn.py (+1/-5); vllm/attention/backends/flashinfer.py (+0/-1); vllm/attention/backends/flashmla.py (+4/-8); vllm/attention/backends/mla/common.py (+0/-1); vllm/attention/backends/rocm_aiter_mla.py (+4/-8); vllm/attention/backends/rocm_flash_attn.py (+1/-5); vllm/attention/backends/triton_mla.py (+4/-8); vllm/attention/backends/xformers.py (+1/-5); vllm/attention/layer.py (+2/-4); vllm/attention/ops/blocksparse_attention/__init__.py (+0/-0); vllm/attention/ops/blocksparse_attention/blocksparse_attention_kernel.py (+0/-433); vllm/attention/ops/blocksparse_attention/interface.py (+0/-239); vllm/attention/ops/blocksparse_attention/utils.py (+0/-246); vllm/attention/selector.py (+0/-9); vllm/v1/attention/backends/cpu_attn.py (+1/-5); vllm/v1/attention/backends/flash_attn.py (+1/-5); vllm/v1/attention/backends/flashinfer.py (+1/-2); vllm/v1/attention/backends/flex_attention.py (+1/-6); vllm/v1/attention/backends/mla/common.py (+1/-2); vllm/v1/attention/backends/mla/cutlass_mla.py (+4/-8); vllm/v1/attention/backends/mla/flashmla.py (+4/-8); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+4/-8); vllm/v1/attention/backends/mla/triton_mla.py (+4/-8); vllm/v1/attention/backends/pallas.py (+1/-7); vllm/v1/attention/backends/rocm_aiter_fa.py (+1/-5); vllm/v1/attention/backends/triton_attn.py (+1/-5); .buildkite/scripts/hardware_ci/run-amd-test.sh (+0/-1); docs/models/supported_models.md (+0/-1); tests/kernels/attention/test_blocksparse_attention.py (+0/-441); tests/kernels/attention/test_rocm_attention_selector.py (+24/-8); tests/models/registry.py (+0/-4); vllm/model_executor/models/phi3_small.py (+0/-465); vllm/model_executor/models/registry.py (+0/-1); vllm/platforms/interface.py (+0/-1)
LABELS: documentation, new-model, rocm, tpu, ready, ci/build, v1
BODY: This PR removes the block sparse attention and the support for phi3-small which uses the attention.

### L3-304dce7ec0  (L3, 2025-07-21, sha 304dce7ec027, PR #21188)
TITLE: [Attention] Clean up iRoPE in V1 (#21188)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.flash_attn.v1_backend,L3.flashinfer.v1_backend,L3.flashinfer.trtllm_gen,L3.flashinfer.trtllm_xqa_decode,L3.triton.v1_backend,L3.rocm.aiter_fa; adapt framework in L3.flash_attn.v1_backend; [Attention] Clean up iRoPE in V1 (#21188)
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa
FILES: vllm/attention/layer.py (+7/-0); vllm/v1/attention/backends/cpu_attn.py (+0/-5); vllm/v1/attention/backends/flash_attn.py (+0/-2); vllm/v1/attention/backends/flashinfer.py (+0/-2); vllm/v1/attention/backends/pallas.py (+0/-5); vllm/v1/attention/backends/rocm_aiter_fa.py (+0/-2); vllm/v1/attention/backends/triton_attn.py (+0/-6); vllm/v1/worker/gpu_model_runner.py (+3/-4); vllm/v1/worker/tpu_model_runner.py (+4/-0)
LABELS: tpu, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ With https://github.com/vllm-project/vllm/commit/89cab4d01f83f8def180e723cee30c7ef8c53e86 we can actually entirely remove the concept of iRoPE from V1 backends. ⏎  ⏎ ## Test Plan ⏎  ⏎ lm eval checks; maybe someone with access to a TPU could help me test that? otherwise the change seems simple enough ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASH_ATTN python -m lm_eval --model vllm --model_args pretrained=/home/lwilkinson/local_models/meta-llama--Llama-4-Scout-17B-16E-Instruct,tensor_parallel_size=2,gpu_memory_utilization=0.8,trust_remote_code=True,max_model_len=16384 --tasks ruler --limit 100 --batch_size auto --output_path ./test_fixed_flashinfer.json ⏎ ... ⏎ |Groups|Version|Filter|n-shot|Metric|   |Value |   |Stderr| ⏎ |------|------:|------|------|-----:|---|-----:|---|------| ⏎ |ruler |      1|none  |      |  4096|↑  |0.9527|±  |   N/A| ⏎  ⏎ ``` ⏎  ⏎ ## (Optional) Documentation Update

### L3-2dec7c1a5d  (L3, 2025-07-22, sha 2dec7c1a5df9, PR #21420)
TITLE: [Bugfix][CUDA] fixes CUDA FP8 kv cache dtype supported (#21420)
SOURCES: symbol_pickaxe
STAGE1: repair_correctness; artifacts=L3.platform.cuda_selection; repair correctness in L3.platform.cuda_selection; [Bugfix][CUDA] fixes CUDA FP8 kv cache dtype supported (#21420)
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+13/-13)
LABELS: ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Previous PR(#21302) defined a class method `is_kv_cache_dtype_supported()` under `NonNvmlCudaPlatform` class, but `NvmlCudaPlatform` also needs this method. ⏎  ⏎ So this PR moved the method to the parent class `CudaPlatformBase`. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-7c734ee09b  (L3, 2025-07-23, sha 7c734ee09b0a, PR #21364)
TITLE: [Bugfix][Qwen][DCA] fixes bug in dual-chunk-flash-attn backend for qwen 1m models. (#21364)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=NEW:7c734ee09b; repair correctness in NEW:7c734ee09b; [Bugfix][Qwen][DCA] fixes bug in dual-chunk-flash-attn backend for qwen 1m models. (#21364)
ARTIFACT_HINTS: -
FILES: vllm/attention/backends/dual_chunk_flash_attn.py (+0/-8)
LABELS: ready, qwen
BODY: …en 1m models. ⏎  ⏎ ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Fixes a bug in the DCA backend. The error was introduced during rebasing previous PR: https://github.com/vllm-project/vllm/pull/11844

### L3-78c13e30e1  (L3, 2025-07-23, sha 78c13e30e164, PR #21419)
TITLE: [V1] Fix local chunked attention always disabled (#21419)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=NEW:78c13e30e1; repair correctness in NEW:78c13e30e1; [V1] Fix local chunked attention always disabled (#21419)
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+2/-1)
LABELS: ready
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ [#21188](https://github.com/vllm-project/vllm/pull/21188) and [#19351](https://github.com/vllm-project/vllm/pull/19351) made similar and conflicting changes around `self.use_irope` in Attention layer, causing `self.use_irope` to always be `False` in V1: ⏎  ⏎ ``` ⏎ self.use_irope = extra_impl_args.pop("use_irope", False) ⏎ ... ⏎ self.use_irope = extra_impl_args.get("use_irope", False) ⏎ ```  ⏎  ⏎ We should not pop `use_irope` in V0 as attention backends still expect `use_irope` as an arg ([example](https://github.com/vllm-project/vllm/blob/main/vllm/attention/backends/flash_attn.py#L621)) ⏎  ⏎ ## Test Plan ⏎ ruler niah_multikey_2 ⏎ ``` ⏎ VLLM_USE_V1=1 lm_eval --model vllm --tasks niah_multikey_2 --model_args pretrained=meta-llama/Llama-4-Scout-17B-16E-Instruct,tensor_parallel_size=4,max_model_len=256000  --metadata='{"max_seq_lengths":[4096,8192,16384,32768]}' --batch_size auto  > /tmp/test_irope_fix.log 2>&1 & ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ baseline: ⏎ ``` ⏎ |---------------|------:|------|-----:|-----:|---|----:|---|------| ⏎ |niah_multikey_2|      1|none  |     0| 16384|↑  |0.980|±  |   N/A| ⏎ |               |       |none  |     0| 32768|↑  |0.000|±  |   N/A| ⏎ |               |       |none  |     0|  4096|↑  |1.000|±  |   N/A| ⏎ |               |       |none  |     0|  8192|↑  |0.996|±  |   N/A| ⏎ ``` ⏎  ⏎ This PR: ⏎ ``` ⏎ |     Tasks     |Version|Filter|n-shot|Metric|   |Value|   |Stderr| ⏎ |---------------|------:|------|-----:|-----:|---|----:|---|------| ⏎ |niah_multikey_2|      1|none  |     0| 16384|↑  |0.980|±  |   N/A| ⏎ |               |       |none  |     0| 32768|↑  |0.944|±  |   N/A| ⏎ |               |       |none  |     0|  4096|↑  |1.000|±  |   N/A| ⏎ |               |       |none  |     0|  8192|↑  |0.996|±  |   N/A| ⏎  ⏎ ``` ⏎  ⏎ cc: @luccafong @minosfuture @houseroad @yeqcharlotte

### L3-90eeea8f85  (L3, 2025-07-24, sha 90eeea8f8501, PR #21205)
TITLE: [Bugfix][ROCm] Fix for warp_size uses on host (#21205)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.paged.cuda.v1,L3.paged.cuda.v2_splitkv,L3.rocm.custom_paged; repair correctness in L3.paged.cuda.v1; [Bugfix][ROCm] Fix for warp_size uses on host (#21205)
ARTIFACT_HINTS: L3.paged.cuda.v1, L3.paged.cuda.v2_splitkv, L3.rocm.custom_paged
FILES: csrc/attention/attention_kernels.cuh (+1/-1); csrc/attention/paged_attention_v1.cu (+2/-3); csrc/attention/paged_attention_v2.cu (+2/-3); csrc/rocm/attention.cu (+1/-1); csrc/cuda_compat.h (+29/-2); csrc/moe/topk_softmax_kernels.cu (+29/-18); csrc/quantization/activation_kernels.cu (+1/-1); csrc/quantization/gguf/gguf_kernel.cu (+1/-1); csrc/rocm/skinny_gemms.cu (+1/-1)
LABELS: rocm, ready
PERF_LINES: VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 python benchmarks/benchmark_throughput.py --model amd/Llama-3.1-405B-Instruct-FP8-KV --dtype float16 --num-prompts 500 -- | VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 python benchmarks/benchmark_latency.py --model mistralai/Mixtral-8x22B-v0.1 --dtype float16 --batch-size 16 --input-len 1
BODY: Fix for the regression added in #20330  ⏎ Before that change certain configs relied on the undefined behavior on ROCm (using warpSize compiler builtin on host), and after they started crashing due to the value mismatch. ⏎  ⏎ On ROCm the same compiled image can be used on different platforms with different warp sizes (Instinct, Radeon), therefore the WARP_SIZE that is used in the host code can't be made constexpr, but rather needs to be queried from the driver. ⏎  ⏎ This PR addresses this issue, as well as fixing the paths to cuda_compat to be relative, to not include the un-hippified version in case hipify script is being used (ROCm) ⏎  ⏎ Affected configs include: ⏎ ``` ⏎ VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 python benchmarks/benchmark_throughput.py --model amd/Llama-3.1-405B-Instruct-FP8-KV --dtype float16 --num-prompts 500 --input-len 2048 --output-len 2048 -tp 8 -O '{"pass_config":{"enable_attn_fusion":true,"enable_noop":true,"enable_fusion":true},"full_cuda_graph":true,"custom_ops":["+silu_and_mul"]}' --no-enable-prefix-caching ⏎ ``` ⏎  ⏎ ``` ⏎ VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 python benchmarks/benchmark_latency.py --model mistralai/Mixtral-8x22B-v0.1 --dtype float16 --batch-size 16 --input-len 1024 --output-len 1024 -tp 8 -O '{"pass_config":{"enable_noop":true,"enable_fusion":true},"full_cuda_graph":true}' --num-iters-warmup 1 --num-iters 3 ⏎ ```

### L3-61b8cea3b4  (L3, 2025-07-24, sha 61b8cea3b42f, PR #21137)
TITLE: [Attention] Optimize FlashInfer MetadataBuilder Build call (#21137)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: optimize; artifacts=L3.flashinfer.v1_backend,L3.flashinfer.trtllm_gen,L3.flashinfer.trtllm_xqa_decode; optimize in L3.flashinfer.v1_backend; [Attention] Optimize FlashInfer MetadataBuilder Build call (#21137)
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+83/-74); tests/v1/attention/test_attention_backends.py (+10/-3); tests/v1/attention/utils.py (+1/-1)
LABELS: rocm, speculative-decoding, ready, v1
PERF_LINES: python benchmarks/benchmark_throughput.py --model meta-llama/Llama-3.2-3B-Instruct --dataset-name random --input-len 256 --output-len 128 --num-prompts <N> --se
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Flash infer prefers host side CPU buffers in many cases, example: https://github.com/flashinfer-ai/flashinfer/blob/3c40456effae8b9c5b1a11c0d1e0594295b1a312/flashinfer/prefill.py#L1430-L1436 ⏎  ⏎ So we pass host side buffers (since https://github.com/vllm-project/vllm/pull/20466 we now have access to these) to reduce D2H transfers. ⏎  ⏎ Trace from main showing D2H transfers in `plan` ⏎  ⏎ <img width="1565" height="310" alt="image" src="https://github.com/user-attachments/assets/48a4fb10-3579-4e2f-b791-7c264ad1e944" /> ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ### Accuracy Results ⏎  ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASHINFER lm_eval --model vllm --model_args pretrained=met ⏎ a-llama/Meta-Llama-3-8B-Instruct --tasks gsm8k --batch_size auto ⏎ ... ⏎ INFO 07-17 20:33:43 [cuda.py:253] Using FlashInfer backend on V1 engine. ⏎ ... ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7536|±  |0.0119| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.7551|±  |0.0118| ⏎ ``` ⏎  ⏎ ### Benchmark Results ⏎  ⏎ **Benchmark Command:** ⏎ ```bash ⏎ python benchmarks/benchmark_throughput.py --model meta-llama/Llama-3.2-3B-Instruct --dataset-name random --input-len 256 --output-len 128 --num-prompts <N> --seed 42 ⏎ ``` ⏎  ⏎ **Results** (3 runs per condition, mean ± standard error): ⏎  ⏎ | num-prompts | Main Branch (req/s) | This PR (req/s) | ⏎ |-------------|---------------------|------------------------| ⏎ | 1           | 1.58 ± 0.06         | 1.90 ± 0.03           | ⏎ | 8           | 13.06 ± 0.11        | 14.32 ± 0.21          | ⏎ | 16          | 26.00 ± 0.07        | 28.74 ± 0.13          | ⏎ | 32          | 47.84 ± 0.57        | 46.53 ± 1.57          | ⏎ | 64          | 76.14 ± 0.45        | 81.43 ± 3.43          | ⏎ | 128         | 116.99 ± 6.10       | 127.78 ± 7.50         | ⏎ | 256         | 164.45 ± 6.12       | 177.70 ± 3.88         | ⏎  ⏎ *Tested on NVIDIA B200 GPU with meta-llama/Llama-3.2-3B-Instruct (256→128 tokens)*  ⏎  ⏎ ## (Optional) Documentation Update

### L3-2dd72d23d9  (L3, 2025-07-24, sha 2dd72d23d965, PR #21485)
TITLE: update flashinfer to v0.2.9rc1 (#21485)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_build_dependency; artifacts=L3.flash_attn.upstream_pip,L3.flashinfer.v0_backend,L3.flashinfer.v1_backend,L3.flashinfer.trtllm_gen,L3.flashinfer.trtllm_xqa_decode; repair build dependency in L3.flash_attn.upstream_pip; update flashinfer to v0.2.9rc1 (#21485)
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: docker/Dockerfile (+1/-1); vllm/attention/backends/flashinfer.py (+3/-7); vllm/v1/attention/backends/flashinfer.py (+2/-7)
LABELS: ready, ci/build, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ updata flashinfer and modify it's trtllm-gen call to use latest API. ⏎ ## Test Plan ⏎ test llama4 with lm_eval ⏎ ## Test Result ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.934|±  |0.0111| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.912|±  |0.0127| ⏎  ⏎ ``` ⏎ ## (Optional) Documentation Update

### L3-b3caeb82e7  (L3, 2025-07-25, sha b3caeb82e740, PR #20295)
TITLE: [ROCm][AITER] Enable fp8 kv cache on rocm aiter backend. (#20295)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L3.rocm.aiter_fa; extend support in L3.rocm.aiter_fa; [ROCm][AITER] Enable fp8 kv cache on rocm aiter backend. (#20295)
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+129/-96); tests/kernels/attention/test_aiter_flash_attn.py (+191/-0)
LABELS: documentation, performance, rocm, speculative-decoding, ready, ci/build, v1, llama
BODY: Rocm aiter backend could support fp8 kv cache with latest aiter ⏎ CMD: ⏎ HIP_VISIBLE_DEVICES=3,4 VLLM_ROCM_USE_AITER=1 VLLM_USE_V1=1 vllm serve /models/models--amd--Meta-Llama-3.1-8B-Instruct-FP8-KV/snapshots/fa42f9a9105c545755fea25cf69f49ac8c8b40e1/ --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --trust-remote-code --disable-log-requests --block-size 16 --max-model-len 32768 --dtype float16 --quantization fp8 --no-enable-prefix-caching --max-num-batched-tokens=8192 --kv-cache-dtype fp8 --compilation-config '{"full_cuda_graph":true}' ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.796|±  |0.0255| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.744|±  |0.0277| ⏎ --------- ⏎  ⏎ ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-1cd6eaba54  (L3, 2025-07-26, sha 1cd6eaba54a2, PR #21270)
TITLE: Support encoder-only models without KV-Cache (#21270)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L3.flash_attn.v1_backend,L3.dispatch.abstract_interface; extend support in L3.flash_attn.v1_backend; Support encoder-only models without KV-Cache (#21270)
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flash_attn.py (+84/-7); vllm/v1/attention/backends/utils.py (+3/-0); examples/offline_inference/prithvi_geospatial_mae.py (+1/-1); tests/conftest.py (+11/-2); tests/model_executor/test_model_load_with_params.py (+9/-3); tests/models/language/pooling/test_embedding.py (+3/-11); tests/models/language/pooling/test_jina.py (+8/-0); tests/v1/attention/utils.py (+1/-0); tests/v1/test_oracle.py (+0/-1); tests/v1/test_utils.py (+1/-2); vllm/engine/arg_utils.py (+2/-1); vllm/model_executor/models/bert.py (+7/-11); vllm/model_executor/models/roberta.py (+64/-21); vllm/v1/engine/core.py (+6/-0); vllm/v1/spec_decode/eagle.py (+1/-0); vllm/v1/worker/cpu_model_runner.py (+4/-0); vllm/v1/worker/gpu_model_runner.py (+147/-39)
LABELS: documentation, speculative-decoding, ready, v1
ISSUES: #18052 [RFC]: Support pooling in V1
BODY: Add support for encoder models such as BERT which don't support a KV cache due to the non-causal attention. Since the KV Cache Spec is used to build the attention metadata for decoder models, this PR initializes the attention metadata builds for encoder-only models directly from the layers and adds a function to build the attention metadata. ⏎  ⏎ This PR combines elements of PRs ⏎ https://github.com/vllm-project/vllm/pull/21088 ⏎ and https://github.com/vllm-project/vllm/pull/19988 ⏎  ⏎ Summary of changes: ⏎  ⏎ **Flash Attention Backend:** ⏎ - Implement encoder self-attention support without using KV cache ⏎  ⏎ **Scheduler:** ⏎ - Disable chunked prefill for models without KV cache ⏎  ⏎ **GPU Model Runner:** ⏎ - Implement encoder-only attention metadata building for self-attention ⏎  ⏎ Related to: ⏎ - V0 deprecation: #18571 ⏎ - 2025 Q3 roadmap: #20336 ⏎  ⏎ This PR is co-authored with @russellb. It borrows all of the encoder-only attention code from his PR https://github.com/vllm-project/vllm/pull/21088 but leaves out the cross-encoder and encoder attention. ⏎  ⏎ cc: @DarkLight1337  ⏎  ⏎ FIX #18052

### L3-8f605ee309  (L3, 2025-07-27, sha 8f605ee30912, PR #21626)
TITLE: [Attention] Make CutlassMLA the default backend for SM100 (blackwell) (#21626)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: change_default; artifacts=L3.platform.cuda_selection; change default in L3.platform.cuda_selection; [Attention] Make CutlassMLA the default backend for SM100 (blackwell) (#21626)
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+22/-7)
LABELS: ready, v1, deepseek
PERF_LINES: This PR makes CutlassMLA the default backend for Blackwell GPUs. Extensive benchmarking showed that CutlassMLA is much faster than TritonMLA for decode operatio
BODY: This PR makes CutlassMLA the default backend for Blackwell GPUs. Extensive benchmarking showed that CutlassMLA is much faster than TritonMLA for decode operations for prompt-heavy workloads on B200 GPUs for both small and large DeepSeek models, for both FP8 and FP4, while not imposing penalty for prompt-average workloads.  Here is an example result for DeepSeek FP4 on 4xB200 GPUs that compares TritonMLA vs CutlassMLA - we can see improvement around 20-50% for TPOT. ⏎  ⏎ <img width="570" height="677" alt="image" src="https://github.com/user-attachments/assets/6995205a-8ca6-4f83-9300-77711b3c98ba" />

### L3-01c753ed98  (L3, 2025-07-28, sha 01c753ed98c7, PR #21701)
TITLE: update flashinfer to v0.2.9rc2 (#21701)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_build_dependency; artifacts=L3.flash_attn.upstream_pip; repair build dependency in L3.flash_attn.upstream_pip; update flashinfer to v0.2.9rc2 (#21701)
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1)
LABELS: ready, ci/build
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ We need https://github.com/flashinfer-ai/flashinfer/pull/1332 to make building work on GH200 and GB200. ⏎  ⏎ ## Test Plan ⏎  ⏎ lm_eval test llama4 with flashinfer backend ⏎  ⏎ ## Test Result ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.926|±  |0.0117| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.906|±  |0.0131| ⏎  ⏎ ## (Optional) Documentation Update

### L3-58b11b24a6  (L3, 2025-07-29, sha 58b11b24a69f, PR #21525)
TITLE: [Bugfix] Fix workspace buffer None issue for Flashinfer TRTLLM Backend (#21525)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L3.flashinfer.v0_backend,L3.flashinfer.v1_backend,L3.flashinfer.trtllm_gen,L3.flashinfer.trtllm_xqa_decode; repair correctness in L3.flashinfer.v0_backend; [Bugfix] Fix workspace buffer None issue for Flashinfer TRTLLM Backend (#21525)
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/attention/backends/flashinfer.py (+11/-4); vllm/v1/attention/backends/flashinfer.py (+15/-15); benchmarks/kernels/benchmark_trtllm_attention.py (+28/-14); tests/kernels/attention/test_flashinfer_trtllm_decode_attention.py (+7/-9)
LABELS: performance, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ There is an always valid `workspace_buffer` initialized in `attn_metadata.decode_wrapper._float_workspace_buffer`, so no need to store the extra one `attn_metadata.workspace_buffer`. They should be the same. ⏎  ⏎ Original PR(#19825) initialize `attn_metadata` via the following way, `self._workspace_buffer` will be None at the beginning, and will be allocated after ONE round of `self._plan()`. So it will only get the correct assignment at the 2nd round of `build()`. ⏎ ``` ⏎ attn_metadata = FlashInferMetadata( ⏎     .... ⏎     workspace_buffer=self._workspace_buffer, ⏎ ) ⏎  ⏎ self._plan(...) ⏎ ``` ⏎  ⏎ There is one PR(#21137) likely fixed the issue a while ago, but I think it is better to just remove this duplicated attribute. ⏎ ``` ⏎ attn_metadata = FlashInferMetadata( ⏎     .... ⏎     workspace_buffer=self._get_workspace_buffer(), ⏎ ) ⏎ ``` ⏎  ⏎ Also fixed unit test for the breakage API for #21485. ⏎  ⏎ ## Test Plan ⏎ `tests/kernels/attention/test_flashinfer_trtllm_decode_attention.py` ⏎ `lm_eavl` ⏎  ⏎ ## Test Result ⏎ ``` ⏎ ===== 96 passed, 1 warning in 121.20s (0:02:01) ===== ⏎ ``` ⏎ ``` ⏎ vllm (pretrained=nvidia/Llama-4-Scout-17B-16E-Instruct-FP8,quantization=modelopt,tensor_parallel_size=1,max_model_len=2048,kv_cache_dtype=auto,trust_remote_code=True), gen_kwargs: (temperature=0.0), limit: 500.0, num_fewshot: 5, batch_size: 200 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.932|±  |0.0113| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.914|±  |0.0126| ⏎ ``` ⏎  ⏎ ## (Optional) Documentation Update

### L3-a33ea28b1b  (L3, 2025-07-29, sha a33ea28b1be6, PR #21389)
TITLE: Add `flashinfer_python` to CUDA wheel requirements (#21389)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
STAGE1: integrate; artifacts=L3.flash_attn.upstream_pip; integrate in L3.flash_attn.upstream_pip; Add `flashinfer_python` to CUDA wheel requirements (#21389)
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+3/-1); requirements/cuda.txt (+2/-0)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ We have installed flashinfer by default in the docker image for a long time, but now as we use flashinfer for many critical kernels for NVIDIA Blackwell we should consider adding it to the default CUDA dependencies. ⏎  ⏎ The `flashinfer-python` wheel by default does not include pre-compiled kernels, so users will JIT at runtime. ⏎  ⏎ ## Test Plan ⏎  ⏎ See if there are any conflicts in CI with the Dockerfile's manual AOT build ⏎  ⏎ ## Test Result

### L3-555e7225bc  (L3, 2025-07-30, sha 555e7225bcb9, PR #21412)
TITLE: [v1][attention] Support Hybrid Allocator + FlashInfer (#21412)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: adapt_framework; artifacts=L3.flash_attn.v1_backend,L3.flashinfer.v1_backend,L3.flashinfer.trtllm_gen,L3.flashinfer.trtllm_xqa_decode,L3.triton.v1_backend,L3.rocm.aiter_fa,L3.mla.common_v1,L3.mla.flashmla_v1_adapter,L3.mla.rocm_aiter,L3.dispatch.abstract_interface,L3.flex_attention; adapt framework in L3.flash_attn.v1_backend; [v1][attention] Support Hybrid Allocator + FlashInfer (#21412)
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.aiter_fa, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.rocm_aiter, L3.dispatch.abstract_interface, L3.flex_attention
FILES: vllm/config.py (+24/-8); vllm/v1/attention/backends/cpu_attn.py (+2/-2); vllm/v1/attention/backends/flash_attn.py (+2/-2); vllm/v1/attention/backends/flashinfer.py (+7/-11); vllm/v1/attention/backends/flex_attention.py (+2/-2); vllm/v1/attention/backends/mla/common.py (+3/-1); vllm/v1/attention/backends/mla/flashmla.py (+4/-3); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+4/-3); vllm/v1/attention/backends/rocm_aiter_fa.py (+2/-2); vllm/v1/attention/backends/triton_attn.py (+2/-2); vllm/v1/attention/backends/utils.py (+9/-5); vllm/v1/worker/gpu_model_runner.py (+8/-5); tests/v1/attention/test_attention_backends.py (+11/-8); tests/v1/spec_decode/test_eagle.py (+1/-0); tests/v1/worker/test_gpu_model_runner.py (+2/-1); vllm/v1/attention/backends/mamba_attn.py (+2/-2)
LABELS: documentation, rocm, speculative-decoding, ready, ci/build, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Support hybrid allocator + flashinfer backend. Achieved by letting the attention backend know the set of layers that use this backend, and only performs plan for these layers. ⏎  ⏎ Limitation: ⏎ For a model with both sliding window attention and full attention, when hybrid allocator is disabled, both the two types of layer use the same attention metadata builder, and flashinfer cannot handle this case. As a temporary solution, I add an error message to tell user to set disable_sliding_window manually. (This config was set automatically when using flashinfer backend before this PR). ⏎  ⏎ ## Test Plan ⏎ Run basic.py with "google/gemma-3-1b-it" ⏎  ⏎ ## Test Result ⏎ Not cap the max length to the sliding window size, and generate meaningful result ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=FLASHINFER python3 basic.py ⏎  ⏎ output ⏎ ------------------------------------------------------------ ⏎ Prompt:    'Hello, my name is' ⏎ Output:    " Alex. I'm a software engineer.\n\nI'm passionate about creating" ⏎ ------------------------------------------------------------ ⏎ Prompt:    'The president of the United States is' ⏎ Output:    ' currently Joe Biden. He was elected in November 2020.\n\n' ⏎ ------------------------------------------------------------ ⏎ Prompt:    'The capital of France is' ⏎ Output:    ' Paris.\nThe largest city in the world is Tokyo.\nThe most populous' ⏎ ------------------------------------------------------------ ⏎ Prompt:    'The future of AI is' ⏎ Output:    ' being shaped by a critical question: can we build AI that is aligned with human' ⏎ ------------------------------------------------------------ ⏎ ``` ⏎ ## (Optional) Documentation Update

### L3-ad510309ee  (L3, 2025-07-30, sha ad510309ee10, PR #21590)
TITLE: Override attention metadata for fast prefill in some KV sharing setups (#21590)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface; adapt framework in L3.dispatch.abstract_interface; Override attention metadata for fast prefill in some KV sharing setups (#21590)
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+33/-2); tests/v1/e2e/test_kv_sharing_fast_prefill.py (+143/-0); vllm/config.py (+15/-0); vllm/engine/arg_utils.py (+6/-0); vllm/model_executor/models/gemma3n.py (+1/-0); vllm/v1/worker/gpu_model_runner.py (+89/-24)
LABELS: frontend, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ To support YOCO prefill skip for cross-decoder layers in #19719, we need to propagate `logits_indices` that has been padded to align with cudagraph captured batch sizes as well as size of the original `logits_indices`.  ⏎  ⏎ To do this, we can identify which layers are eligible, store these layer names in `truncated_prefill_eligible_layers` and dynamically build a subclass of attention metadata for these layers that inherit from the original attn metadata class but adds two new fields: `logits_indices_padded` and `num_logits_indices`.  ⏎  ⏎ ## Test Plan ⏎ unit test: ⏎ ``` ⏎ pytest tests/v1/e2e/test_kv_sharing_fast_prefill.py::test_kv_sharing_fast_prefill ⏎ ``` ⏎  ⏎ run with new flag: ⏎ ``` ⏎ vllm serve google/gemma-3n-E2B-it --disable-log-requests --kv-sharing-fast-prefill ⏎ ``` ⏎  ⏎ ## Test Result ⏎ Unit tests pass ⏎  ⏎ See new warning msg: ⏎ ``` ⏎ WARNING 07-29 14:06:05 [config.py:1739] --kv-sharing-fast-prefill is currently work in progress and not functional yet (i.e. no prefill savings) ⏎ ```

### L3-61445453df  (L3, 2025-07-30, sha 61445453df8e, PR #21966)
TITLE: [UX] Rename CUTLASS_MLA_VLLM_V1 to CUTLASS_MLA (#21966)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: adapt_framework; artifacts=L3.mla.cutlass_v1_backend,L3.platform.cuda_selection; adapt framework in L3.mla.cutlass_v1_backend; [UX] Rename CUTLASS_MLA_VLLM_V1 to CUTLASS_MLA (#21966)
ARTIFACT_HINTS: L3.mla.cutlass_v1_backend, L3.platform.cuda_selection
FILES: vllm/engine/arg_utils.py (+1/-1); vllm/platforms/cuda.py (+5/-5); vllm/platforms/interface.py (+1/-1); vllm/v1/attention/backends/mla/cutlass_mla.py (+1/-1)
LABELS: ready, v1
BODY: We should probably remove most of the "VLLM_V1" attention names and just use their base name + `use_v1`

### L3-0bd409cf01  (L3, 2025-07-31, sha 0bd409cf01c3, PR #21959)
TITLE: Move flashinfer-python to optional extra `vllm[flashinfer]` (#21959)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
STAGE1: repair_build_dependency; artifacts=L3.flash_attn.upstream_pip; repair build dependency in L3.flash_attn.upstream_pip; Move flashinfer-python to optional extra `vllm[flashinfer]` (#21959)
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: requirements/cuda.txt (+1/-3); setup.py (+3/-1)
LABELS: ready, ci/build
BODY: Adding flashinfer-python by default to the CUDA requirements introduced a new hard requirement for vLLM to have `nvcc` installed, which we don't want to enforce for all users. See issue https://github.com/vllm-project/vllm/issues/21960 ⏎  ⏎ For now we will move the dependency as an extras i.e. `uv pip install vllm[flashinfer]`

### L3-e1a7fe4af5  (L3, 2025-08-01, sha e1a7fe4af5e9, PR #19750)
TITLE: [BugFix] fix: aot passes kvcache dtype information (#19750)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.flash_attn.v1_backend; repair correctness in L3.flash_attn.v1_backend; [BugFix] fix: aot passes kvcache dtype information (#19750)
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+21/-4)
LABELS: ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ AOT doesn't pass all the information to vLLM's metadata function. That results in inconsistent runs, potentially cause data corruption. ⏎  ⏎ ## Test Plan ⏎  ⏎ Not sure what I should add here. Maybe @LucasWilkinson? ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-f81c1bb055  (L3, 2025-08-01, sha f81c1bb05504, PR #21893)
TITLE: [Bugfix] Check NVIDIA artifactory is accessible before using flashinfer cubin kernels (#21893)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: change_default; artifacts=L3.flashinfer.v0_backend,L3.flashinfer.v1_backend,L3.flashinfer.utils_dependency,L3.flashinfer.trtllm_gen,L3.flashinfer.trtllm_xqa_decode,L3.mla.common_v1; change default in L3.flashinfer.v0_backend; [Bugfix] Check NVIDIA artifactory is accessible before using flashinfer cubin kernels (#21893)
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1
FILES: vllm/attention/backends/flashinfer.py (+2/-44); vllm/utils/flashinfer.py (+80/-1); vllm/v1/attention/backends/flashinfer.py (+3/-46); vllm/v1/attention/backends/mla/common.py (+8/-8)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Due to limitations in FlashInfer (https://github.com/flashinfer-ai/flashinfer/issues/1352), the AOT wheel generation does not download kernels ahead of time that are cubin-based. This causes issues for users that rely on our docker image to run in network-isolated environment with everything included. ⏎  ⏎ To support this usage immediately, I believe we should gate default usage of kernels that require downloading with an initial check to see if that endpoint is accessible at all. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result

### L3-eefbf4a68b  (L3, 2025-08-01, sha eefbf4a68b7b, PR #22036)
TITLE: [Perf] Optimize `reshape_and_cache_flash` CUDA Kernel (#22036)
SOURCES: path_core, release_notes
STAGE1: optimize; artifacts=L3.cache.cuda_reshape; optimize in L3.cache.cuda_reshape; [Perf] Optimize `reshape_and_cache_flash` CUDA Kernel (#22036)
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/cache_kernels.cu (+69/-23); benchmarks/kernels/benchmark_reshape_and_cache_flash.py (+156/-0)
LABELS: performance, ready
PERF_LINES: test_cache.py ....................................... [  3%] | ...................................s...s...s...s...s. [  8%] | ..s...s...s...s...s...s...s...s...s...s...s...s...s.. [ 13%] | ..................................................... [ 17%] | ....................s...s...s...s...s...s...s...s...s [ 22%] | ...s...s...s...s...s...s...s...s...s................. [ 27%] | ......................
BODY: ## Purpose ⏎  ⏎ Using vectorization utils to  `reshape_and_cache_flash` and get performance improvement ⏎  ⏎ ## Test ⏎  ⏎ ### Acc ⏎  ⏎ ```bash ⏎ lm_eval   --model vllm   --model_args "pretrained=Qwen/Qwen3-30B-A3B-FP8,max_model_len=32768,enforce_eager=True"   --trust_remote_code   --tasks gsm8k   --num_fewshot 5   --batch_size auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8173|±  |0.0106| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.8870|±  |0.0087| ⏎ # main ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8173|±  |0.0106| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.8870|±  |0.0087| ⏎ ``` ⏎  ⏎ ```bash ⏎ pytest test_cache.py -x ⏎ ==================== test session starts ==================== ⏎ platform linux -- Python 3.12.3, pytest-8.4.0, pluggy-1.6.0 ⏎ rootdir: /home/wentao/vllm-source ⏎ configfile: pyproject.toml ⏎ plugins: asyncio-1.0.0, anyio-4.9.0 ⏎ asyncio: mode=Mode.STRICT, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function ⏎ collected 1102 items                                         ⏎  ⏎ test_cache.py ....................................... [  3%] ⏎ ...................................s...s...s...s...s. [  8%] ⏎ ..s...s...s...s...s...s...s...s...s...s...s...s...s.. [ 13%] ⏎ ..................................................... [ 17%] ⏎ ....................s...s...s...s...s...s...s...s...s [ 22%] ⏎ ...s...s...s...s...s...s...s...s...s................. [ 27%] ⏎ ..................................................... [ 32%] ⏎ ..................................................... [ 37%] ⏎ ..................................................... [ 42%] ⏎ ..................................................... [ 46%] ⏎ ..................................................... [ 51%] ⏎ ..................................................... [ 56%] ⏎ ..................................................... [ 61%] ⏎ ..................................................... [ 66%] ⏎ ..................................................... [ 70%] ⏎ ...........s.ss.sssss.ss.ss.sssss.ss.ss.sssss.ss.ss.s [ 75%] ⏎ ssss.ss.ss.sssss.ss.ss.sssss.ss.ss.sssss.ss.ss.sssss. [ 80%] ⏎ ss.ss.sssss.ss.ss.sssss.ss.ss.sssss.ss.ss.sssss.ss.ss [ 85%] ⏎ .sssss.ss.ss.sssss.ss.ss.sssss.ss.ss.sssss.ss.ss.ssss [ 90%] ⏎ s.ss.ss.sssss.s...................................... [ 94%] ⏎ ...................................... …[truncated]

### L3-0edaf752d7  (L3, 2025-08-01, sha 0edaf752d748, PR #21153)
TITLE: [Attention][DBO] Add support for "splitting" the CommonAttentionMetadata (#21153)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L3.dispatch.abstract_interface; adapt framework in L3.dispatch.abstract_interface; [Attention][DBO] Add support for "splitting" the CommonAttentionMetadata (#21153)
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+83/-0); tests/v1/attention/test_attention_splitting.py (+157/-0)
LABELS: ready, v1
BODY: ## Purpose ⏎ Note: the infrastructure in this PR is currently unused outside of unit tests. ⏎  ⏎ The purpose of this PR is to add a "splitting" mechanism to the `CommonAttentionMetadata` class. This PR is a prerequisite for Dual Batch Overlap (#20448 ) support in vllm. Splitting, in this context, means slicing a batch of requests along some mid point and generating two new `CommonAttentionMetadata` instances. One for each "micro" batch. ⏎  ⏎ ## Test Plan ⏎ I added a number of unit tests to `test_attention_splitting.py`. These tests are largely focused on decode-only batches for now since that's all Dual Batch Overlap will initially support, but this infrastructure should support prefills and mixed batches without any issue.

### L3-23322431c8  (L3, 2025-08-01, sha 23322431c802, PR #21367)
TITLE: [V1][CUDA] Full cudagraph support for FlashInfer (#21367)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: adapt_framework; artifacts=L3.flash_attn.v1_backend,L3.flashinfer.v1_backend,L3.flashinfer.trtllm_gen,L3.flashinfer.trtllm_xqa_decode,L3.triton.v1_backend,L3.mla.flashmla_v1_adapter,L3.mla.rocm_aiter,L3.dispatch.abstract_interface; adapt framework in L3.flash_attn.v1_backend; [V1][CUDA] Full cudagraph support for FlashInfer (#21367)
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.mla.flashmla_v1_adapter, L3.mla.rocm_aiter, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flash_attn.py (+5/-2); vllm/v1/attention/backends/flashinfer.py (+323/-34); vllm/v1/attention/backends/mla/flashmla.py (+3/-1); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+3/-1); vllm/v1/attention/backends/triton_attn.py (+4/-2); vllm/v1/attention/backends/utils.py (+17/-1); vllm/v1/worker/gpu_model_runner.py (+17/-7); vllm/v1/worker/gpu_worker.py (+5/-0)
LABELS: rocm, ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ This PR is split from origin #20059 to support full cudagraph for FlashInfer (pure decode only), which runs pure decode batches at full cudagraph, and falls back to no cudagraph at mix prefill-decode batches. Hope to land this first before #20059. ⏎  ⏎ Details include: ⏎ - Using the persistent buffer trick. ⏎ - Create many decode_warpers, one for a cudagraph batch size, as this is required by the FlashInfer API. ⏎  ⏎ This PR also fixes a potential bug originally from #18581, where an assertion error will be raised when capturing if max_capture_size is greater than max_num_reqs.  To resolve this, a new enum type AttentionCGSupport (adapted from #20059) is introduced to distinguish how the backend supports cudagraph, so that we can overwrite the cudagraph_batch_sizes to be not greater than max_num_seqs. ⏎ NOTE: Currently, manually setting max capture size seems impossible after the introduction of Pydantic, which blocks config `--cuda_graph_sizes` to be `int` type ⏎  ⏎ Limitation: ⏎ 1. FlashInfer backend currently does not support spec-decode when enabling full cudagraph (pure-decode only) ⏎ 2. trtllm_batch_decode_with_kv_cache of Flashinfer is not yet considered supported in this PR. ⏎  ⏎ ### (updated) After #21137 is merged ⏎ To resolve the conflicts with the early version of this PR and further reduce potential overhead, the new commits include adding both host-side and device-side persistent buffers and overriding the decode plan function of Flashinfer for cudagraph execution. Now, for a full cudagraph common decode, we reduced any copy from temporary tensors to persistent buffers by directly writing results to persistent buffers;  we further avoid `device-to-device` copy of paged_kv_indices, and only retaining two host-to-device copies of `paged_kv_indptr` and `paged_kv_last_page_len` by overriding that plan function. ⏎  ⏎ ## Test Plan ⏎ lm_eval, benchmark_serving ⏎  ⏎ ## Test Result ⏎ See comments below. ⏎  ⏎ ## (Optional) Documentation Update

### L3-4abfd8796f  (L3, 2025-08-02, sha 4abfd8796f37, PR #21557)
TITLE: [V1] [Hybrid] Validate compatibility of attention backend batch reordering at init time (#21557)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: adapt_framework; artifacts=L3.flashinfer.v1_backend,L3.flashinfer.trtllm_gen,L3.flashinfer.trtllm_xqa_decode,L3.rocm.aiter_fa,L3.mla.common_v1,L3.dispatch.abstract_interface; adapt framework in L3.flashinfer.v1_backend; [V1] [Hybrid] Validate compatibility of attention backend batch reordering at init time (#21557)
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.rocm.aiter_fa, L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flashinfer.py (+12/-16); vllm/v1/attention/backends/mla/common.py (+7/-15); vllm/v1/attention/backends/rocm_aiter_fa.py (+0/-3); vllm/v1/attention/backends/utils.py (+4/-8); vllm/v1/worker/cpu_model_runner.py (+33/-1); vllm/v1/worker/gpu_model_runner.py (+34/-15); vllm/v1/attention/backends/mamba_attn.py (+6/-14)
LABELS: ready, v1
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Right now on main we have a check at runtime to see if the different attention backends reorder the requests in the same way. This has a number of problems: ⏎ 1. Sometimes problems only get caught after running a long experiment  ⏎ 2. The check calls reorder_batch for the first backend, and then calls reorder_batch for all subsequent backends. If any of the subsequent backends reorder, then it throws. However, if the first attention backend is a "flexible" one like FlashAttention or TritonAttention, then this check doesn't make sense.  ⏎  ⏎ This PR removes that runtime check and adds a new check at init time to verify that the attention backends reorder the batch in a "compatible" way.  ⏎  ⏎ cc @heheda12345 @LucasWilkinson  ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-e27d25a0dc  (L3, 2025-08-03, sha e27d25a0dcbb, PR #22154)
TITLE: [fix] fix correct assertion syntax error in attention utils. (#22154)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.dispatch.abstract_interface; repair correctness in L3.dispatch.abstract_interface; [fix] fix correct assertion syntax error in attention utils. (#22154)
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+3/-1)
LABELS: v1
BODY: ## Purpose ⏎  ⏎   Fix syntax error in assertion statement in vllm/v1/attention/backends/utils.py where ⏎   parentheses were incorrectly placed. The assertion len(query_start_loc >= 2) should be ⏎    len(query_start_loc) >= 2 to properly validate that query_start_loc has at least 2 ⏎   elements. ⏎  ⏎   ## Test Plan ⏎  ⏎   - Run existing unit tests to ensure no regressions ⏎   - Verify the assertion now correctly validates tensor length ⏎   - Test with malformed input to confirm descriptive error message is shown ⏎  ⏎  ##  Test Result ⏎  ⏎   - Syntax error is resolved ⏎   - Assertion now properly validates tensor dimensions ⏎   - Improved error message provides better debugging information

### L3-aa7012eb6d  (L3, 2025-08-03, sha aa7012eb6db6, PR #20401)
TITLE: Add tree attention backend for v1 (part 1) (#20401)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: introduce; artifacts=L3.triton.unified_attention,L3.dispatch.abstract_interface,L3.platform.cuda_selection,L3.tree_attention; introduce in L3.triton.unified_attention; Add tree attention backend for v1 (part 1) (#20401)
ARTIFACT_HINTS: L3.triton.unified_attention, L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.tree_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+48/-0); vllm/config.py (+13/-0); vllm/engine/arg_utils.py (+1/-1); vllm/platforms/cuda.py (+4/-0); vllm/platforms/interface.py (+1/-0); vllm/v1/attention/backends/tree_attn.py (+452/-0); vllm/v1/attention/backends/utils.py (+20/-0); tests/v1/attention/test_attention_backends.py (+1/-1); tests/v1/attention/utils.py (+4/-2); tests/v1/spec_decode/test_eagle.py (+4/-3); tests/v1/spec_decode/test_tree_attention.py (+299/-0); vllm/v1/spec_decode/eagle.py (+251/-18)
LABELS: speculative-decoding, ready, v1, llama
PERF_LINES: Request throughput (req/s) | 10.83 | 11.04 | 10.00 | Output token throughput (tok/s) | 3248.45 | 3311.34 | 3000.36 | Total Token throughput (tok/s) | 14053.85 | 14325.95 | 12980.52 | Mean TTFT (ms) | 659.81 | 608.03 | 711.49 | Median TTFT (ms) | 238.31 | 212.95 | 181.93 | P99 TTFT (ms) | 2016.53 | 1929.07 | 2261.90 | Mean TPOT (ms) | 13.6 | 13.52 | 15.77 | Median TPOT (ms) | 12.89 | 12.66 | 14.94 
BODY: # Purpose ⏎ Add support for tree attention v1 backend. Tree attention is used in EAGLE speculative decoding by the target model to validate a set of draft tokens. Draft tokens only attend to ancestor tokens, and so attention bias must be used to omit attention between non-descendant tokens. To suppor that, I added a new parameter to triton `unified_attention` called `qq_bias`. This parameter enables applying query-on-query attention bias using a 2D (q_len, q_len) tensor. This feature is only enabled if a non-None value is provided for that parameter. Otherwise, it is disabled (the default case). ⏎  ⏎ I also implemented the logic for tree draft proposal, in Eagle.py. For chain drafts, it behaves the same as before. However, if a tree of speculative tokens is specified (via the Speculative Config), then this system can leverage TreeAttentionBackend for drafting. Top-K is used to select the drafted child tokens at each level of the tree. ⏎  ⏎ NOTE: This PR does NOT change the existing behavior of v1 EAGLE. It simply adds the capability to use the TreeAttentionBackend, which can validate a tree of draft tokens. However, since tree scoring is still not implemented (I am working on it right now), only chain drafts are supported at this moment. But this is the first step to unlocking tree drafting and scoring functionality! ⏎  ⏎ # Test Plan ⏎ ## Benchmark ⏎ In addition, I used the following command to run the LLM service and benchmark TreeAttentionBackend vs FlashAttentionBackend: ⏎ **Server** ⏎ ``` ⏎ export VLLM_TORCH_PROFILER_DIR=~/traces/vllm ⏎ export LLAMA_MODEL=meta-llama/Llama-3.1-8B-Instruct ⏎ export DRAFT_MODEL=yuhuili/EAGLE-LLaMA3.1-Instruct-8B ⏎ export VLLM_USE_V1=1 ⏎ export VLLM_ATTENTION_BACKEND=<backend> ⏎ export SPEC_DEC_CONFIG='{"method": "eagle", "model": "'$DRAFT_MODEL'", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1, "max_model_len": 2048, "speculative_token_tree": "[(0,), (0, 0), (0, 0, 0)]"}' ⏎ python -m vllm.entrypoints.openai.api_server --model $LLAMA_MODEL --disable-log-requests --tensor-parallel-size=1 --max-num-seqs=64 --max-model-len=32768 --block-size=128 --no-enable-prefix-caching --speculative-config="$SPEC_DEC_CONFIG" 2>&1 | tee ~/server_logs/vllm_server.log ⏎ ``` ⏎  ⏎ **Client** ⏎ ``` ⏎ export LLAMA_MODEL=meta-llama/Llama-3.1-8B-Instruct ⏎ python benchmarks/benchmark_serving.py --model $LLAMA_MODEL --tokenizer $LLAMA_MODEL --host 0.0.0.0 --dataset-name random --ignore-eos --request-rate inf --random-input-len 1000 --random-output-len 300 --max-concurrency 64 --num-prompts 128 ⏎ ``` ⏎  ⏎ **Results** ⏎ Serving Benchmark Result | Flash Attention (Baseline) | Flash Attention …[truncated]

### L3-cdfd6871a5  (L3, 2025-08-04, sha cdfd6871a5c4, PR #22226)
TITLE: [Bugfix] Misaligned params in TreeAttentionImpl (#22226)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.tree_attention; repair correctness in L3.tree_attention; [Bugfix] Misaligned params in TreeAttentionImpl (#22226)
ARTIFACT_HINTS: L3.tree_attention
FILES: vllm/v1/attention/backends/tree_attn.py (+1/-5)
LABELS: ready, v1
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Tree attention is still broken because it conflicts with #21217 causing misaligned args to be passed to the model (notably `attn_type` becomes `None`). This PR fixes it. ⏎  ⏎ cc @TheEpicDolphin ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-e79a12fc3a  (L3, 2025-08-04, sha e79a12fc3afb, PR #22217)
TITLE: [UX] Fail if an invalid attention backend is specified (#22217)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: change_default; artifacts=L3.dispatch.selector; change default in L3.dispatch.selector; [UX] Fail if an invalid attention backend is specified (#22217)
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/attention/selector.py (+4/-0); tests/kernels/attention/test_attention_selector.py (+5/-15)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ As the title states, I think if an invalid attention backend is manually specified like VLLM_ATTENTION_BACKEND=INVALID it should fail rather than fallback to some default ⏎  ⏎ Rob found that this PR https://github.com/vllm-project/vllm/pull/21966 causes fallback to V0 if that env variable is set incorrectly ⏎ ``` ⏎ WARNING 08-04 15:00:03 [arg_utils.py:1771] VLLM_ATTENTION_BACKEND=CUTLASS_MLA_VLLM_V1 is not supported by the V1 Engine. Falling back to V0. We recommend to remove VLLM_ATTENTION_BACKEND=CUTLASS_MLA_VLLM_V1 from your config in favor of the V1 Engine. ⏎ ``` ⏎  ⏎ ## Test Plan ⏎  ⏎ CI ⏎  ⏎ ## Test Result

### L3-83156c7b89  (L3, 2025-08-05, sha 83156c7b89fb, PR #22095)
TITLE: [NVIDIA] Support Flashinfer TRT-LLM Prefill Attention Kernel (#22095)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: integrate; artifacts=L3.flashinfer.v0_backend,L3.flashinfer.v1_backend,L3.flashinfer.utils_dependency,L3.flashinfer.trtllm_gen,L3.flashinfer.trtllm_xqa_decode; integrate in L3.flashinfer.v0_backend; [NVIDIA] Support Flashinfer TRT-LLM Prefill Attention Kernel (#22095)
ARTIFACT_HINTS: L3.flashinfer.v0_backend, L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/attention/backends/flashinfer.py (+2/-2); vllm/envs.py (+3/-3); vllm/utils/flashinfer.py (+7/-10); vllm/v1/attention/backends/flashinfer.py (+145/-80); .buildkite/test-pipeline.yaml (+1/-1); benchmarks/kernels/benchmark_trtllm_decode_attention.py (+0/-1); benchmarks/kernels/benchmark_trtllm_prefill_attention.py (+250/-0); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+293/-0); tests/kernels/attention/test_flashinfer_trtllm_decode_attention.py (+0/-138)
LABELS: performance, ready, ci/build, v1
PERF_LINES: num_seqs  max_seq_len   trt_mean  trt_std baseline_mean  baseline_std   speedup_percent
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ Previously #19825 supported TRTLLM attn kernel for decode code path, this PR is aiming to support the prefill path. ⏎ - Unified the decode and prefill code path, use only one env `VLLM_USE_TRTLLM_ATTENTION` to control ⏎ - Currently the TRTLLM prefill kernel only does not support Q-BF16 KV-FP8 O-BF16 config, will directly use Q-FP8 KV-FP8 O-FP8/O-NVFP4 after the attn+quant fusion is supported. ⏎ - Generally perf of TRTLLM prefill kernels seems to be much better than the original kernels. ⏎  ⏎ ## Test Plan && Test Result ⏎ `tests/kernels/attention/test_flashinfer_trtllm_attention.py` ⏎ ``` ⏎ ==== 160 passed, 32 skipped, 129 warnings in 26.46s ==== ⏎ ``` ⏎ `lm_eval` ⏎ ``` ⏎ vllm ({'pretrained': '/home/scratch.omniml_data_2/HF_model_hub/Llama-4-Scout-17B-16E-Instruct-FP8', 'quantization': 'modelopt', 'kv_cache_dtype': 'auto', 'tensor_parallel_size': 1, 'compilation_config': {'level': 3, 'custom_ops': ['+rms_norm'], 'pass_config': {'enable_fi_allreduce_fusion': True}, 'full_cuda_graph': True}, 'max_model_len': 2048, 'trust_remote_code': True}), gen_kwargs: (temperature=0.0), limit: 500.0, num_fewshot: 5, batch_size: 200 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.936|±  |0.0110| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.914|±  |0.0126| ⏎ ``` ⏎ `benchmarks/kernels/benchmark_trtllm_prefill_attention.py` ⏎ ``` ⏎ Running benchmark for q_dtype = bfloat16, kv_cache_dtype: bfloat16, output_dtype: bfloat16 ⏎ num_seqs  max_seq_len   trt_mean  trt_std baseline_mean  baseline_std   speedup_percent ⏎ 1         2048          0.102     0.004   0.233          0.010          0.562 ⏎ 4         2048          0.204     0.005   0.377          0.011          0.459 ⏎ 8         2048          0.284     0.004   0.457          0.012          0.379 ⏎ 16        2048          0.660     0.004   0.995          0.012          0.337 ⏎ 32        2048          1.009     0.006   1.446          0.010          0.302 ⏎ 64        2048          2.104     0.007   2.920          0.013          0.280 ⏎ 128       2048          4.416     0.010   6.033          0.011          0.268 ⏎ 256       2048          8.186     0.012   10.907         0.010          0.249 ⏎ 1         4096          0.259     0.005   0.699          0.013          0.629 ⏎ 4         4096          0.521     0.007   1.175          0.013          0.556 ⏎ 8         4096          0.654     0.006   1.397          0.013          0.531 ⏎ 16        4096        …[truncated]

### L3-a7cb6101ca  (L3, 2025-08-05, sha a7cb6101ca7b, PR #22233)
TITLE: [CI/Build] Update flashinfer to 0.2.9 (#22233)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
STAGE1: repair_build_dependency; artifacts=L3.flash_attn.upstream_pip; repair build dependency in L3.flash_attn.upstream_pip; [CI/Build] Update flashinfer to 0.2.9 (#22233)
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); setup.py (+1/-1)
LABELS: ready, ci/build
BODY: Critical to make the docker build on CPU actually build the AOT kernels https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.2.9

### L3-469b3ffaaa  (L3, 2025-08-05, sha 469b3ffaaadb, PR #21342)
TITLE: [V1] port xformers backend to v1 (#21342)
SOURCES: path_core, symbol_pickaxe
STAGE1: introduce; artifacts=L3.xformers.v1_backend,L3.platform.cuda_selection,L3.tree_attention; introduce in L3.xformers.v1_backend; [V1] port xformers backend to v1 (#21342)
ARTIFACT_HINTS: L3.xformers.v1_backend, L3.platform.cuda_selection, L3.tree_attention
FILES: vllm/v1/attention/backends/tree_attn.py (+1/-6); vllm/v1/attention/backends/xformers.py (+430/-0); tests/v1/attention/utils.py (+2/-0); vllm/engine/arg_utils.py (+1/-0); vllm/platforms/cuda.py (+4/-0); vllm/platforms/interface.py (+1/-0)
LABELS: ready, v1
PERF_LINES: tests/v1/attention/test_attention_backends.py ......                                                                                                             | Request throughput (req/s) | 8.03 | 9.68 | 9.68 | Output token throughput (tok/s) | 2408.88 | 2903.54 | 2905.24 | Total Token throughput (tok/s) | 10421.59 | 12561.66 | 12569.01 | Mean TTFT (ms) | 894.77 | 920.44 | 929.93 | Median TTFT (
BODY: # Purpose ⏎ Port over the xformers backend to the v1 engine. There are several benefits to using XFormers, including: ⏎ 1. Built-in heursitic which determines which attention implementation is best suited for the given inputs. ⏎ 2. AMD kernel support ⏎ 3. Well suited for certain Meta models. ⏎  ⏎ # Test Plan ⏎ Added test case to `test_attention_backends` which verifies correctness of the xformers v1 backend attention output. ⏎ ``` ⏎ (py312conda) bash-5.1$ pytest tests/v1/attention/test_attention_backends.py -k test_backend_correctness ⏎ ================================================================================ test session starts ================================================================================ ⏎ platform linux -- Python 3.12.9, pytest-8.4.1, pluggy-1.6.0 ⏎ rootdir: /data/users/gdelfin/gitrepos/vllm ⏎ configfile: pyproject.toml ⏎ plugins: anyio-4.9.0 ⏎ collected 6 items                                                                                                                                                                    ⏎  ⏎ tests/v1/attention/test_attention_backends.py ......                                                                                                                          [100%] ⏎  ⏎ ================================================================================= warnings summary ================================================================================== ⏎ tests/v1/attention/test_attention_backends.py::test_backend_correctness[meta-llama/Meta-Llama-3-8B-small_decode] ⏎ tests/v1/attention/test_attention_backends.py::test_backend_correctness[meta-llama/Meta-Llama-3-8B-small_decode] ⏎ tests/v1/attention/test_attention_backends.py::test_backend_correctness[meta-llama/Meta-Llama-3-8B-small_decode] ⏎ tests/v1/attention/test_attention_backends.py::test_backend_correctness[meta-llama/Meta-Llama-3-8B-small_decode] ⏎   /home/gdelfin/.conda/envs/py312conda/lib/python3.12/site-packages/triton/runtime/autotuner.py:108: DeprecationWarning: warmup, rep, and use_cuda_graph parameters are deprecated. See https://github.com/triton-lang/triton/pull/4496 for details. ⏎     warnings.warn(("warmup, rep, and use_cuda_graph parameters are deprecated. See " ⏎  ⏎ -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html ⏎ ========================================================================== 6 passed, 4 warnings in 18.72s =========================================================================== ⏎ ``` ⏎  ⏎ ## Benchmark ⏎ In addition, I used the following command to run the LLM service and benchmark TreeAttentionBackend vs FlashAttentionBackend: ⏎ **Server** ⏎ ``` ⏎ ex …[truncated]

### L3-e3c876dca3  (L3, 2025-08-05, sha e3c876dca357, PR #22313)
TITLE: Upgrade FA3 for attention sink (#22313)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
STAGE1: repair_build_dependency; artifacts=L3.flash_attn.fork_build; repair build dependency in L3.flash_attn.fork_build; Upgrade FA3 for attention sink (#22313)
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ci/build
BODY: 

### L3-6e20924350  (L3, 2025-08-05, sha 6e20924350e3, PR #22320)
TITLE: Add attention sink in attention backends (#22320)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: extend_support; artifacts=L3.flash_attn.v1_backend,L3.flashinfer.trtllm_gen,L3.triton.prefix_prefill,L3.triton.chunked_prefill_paged_decode,L3.triton.unified_attention,L3.triton.v1_backend,L3.dispatch.abstract_interface; extend support in L3.flash_attn.v1_backend; Add attention sink in attention backends (#22320)
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.trtllm_gen, L3.triton.prefix_prefill, L3.triton.chunked_prefill_paged_decode, L3.triton.unified_attention, L3.triton.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+27/-6); vllm/attention/ops/prefix_prefill.py (+15/-3); vllm/attention/ops/triton_unified_attention.py (+28/-2); vllm/envs.py (+16/-3); vllm/v1/attention/backends/flash_attn.py (+10/-0); vllm/v1/attention/backends/triton_attn.py (+56/-19); vllm/v1/attention/backends/utils.py (+24/-12)
LABELS: v1
BODY: 

### L3-98a3a81024  (L3, 2025-08-05, sha 98a3a8102464, PR #22329)
TITLE: [ROCm] Add attention sink to use_rocm_custom_paged_attention (#22329)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: adapt_framework; artifacts=L3.platform.rocm_selection; adapt framework in L3.platform.rocm_selection; [ROCm] Add attention sink to use_rocm_custom_paged_attention (#22329)
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+6/-5)
LABELS: rocm
BODY: 

### L3-90ec006937  (L3, 2025-08-05, sha 90ec006937c4, PR #22330)
TITLE: [gpt-oss] flashinfer attention sink init (#22330)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.flashinfer.v1_backend,L3.flashinfer.trtllm_gen,L3.flashinfer.trtllm_xqa_decode; extend support in L3.flashinfer.v1_backend; [gpt-oss] flashinfer attention sink init (#22330)
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+10/-0)
LABELS: v1
BODY: 

### L3-2cb6ef8996  (L3, 2025-08-06, sha 2cb6ef899632, PR #22365)
TITLE: [BugFix] Fix FA2 RuntimeError when sinks is provided (#22365)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
STAGE1: repair_build_dependency; artifacts=L3.flash_attn.fork_build; repair build dependency in L3.flash_attn.fork_build; [BugFix] Fix FA2 RuntimeError when sinks is provided (#22365)
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ci/build
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ fix  ⏎ ``` ⏎ RuntimeError: _vllm_fa2_C::varlen_fwd() expected at most 21 argument(s) but received 22 argument(s). Declaration: _vllm_fa2_C::varlen_fwd(Tensor($0! -> ) q, Tensor k, Tensor v, Tensor($1! -> )? out, Tensor cu_seqlens_q, Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? block_table, Tensor? alibi_slopes, int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, bool is_causal, int window_size_left, int window_size_right, float softcap, bool return_softmax, Generator? gen) -> Tensor[] ⏎ ``` ⏎  ⏎ introduced in https://github.com/vllm-project/flash-attention/pull/75 ⏎  ⏎ ## Test Plan ⏎  ⏎ test on A100 before and after PR ⏎  ⏎ ## Test Result ⏎  ⏎ before vllm crashes on first request; after this PR its fine ⏎  ⏎ ## (Optional) Documentation Update

### L3-4a6b72c2ab  (L3, 2025-08-06, sha 4a6b72c2ab98, PR #22368)
TITLE: [BugFix] Fix triton compile error in `kernel_unified_attention_2/3d` caused by attention sinks (#22368)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.triton.unified_attention; repair correctness in L3.triton.unified_attention; [BugFix] Fix triton compile error in `kernel_unified_attention_2/3d` caused by attention sinks (#22368)
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/attention/ops/triton_unified_attention.py (+15/-8)
LABELS: bug, ready
ISSUES: #22363 [Bug]: AttributeError("'NoneType' object has no attribute 'type'") from `sink_ptr` of `unified_attention` when running spec dec with Triton Attention Backend V1
BODY: FIX: https://github.com/vllm-project/vllm/issues/22363 ⏎  ⏎ Tested with: ⏎  ⏎ ``` ⏎ VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 vllm serve ⏎ ``` ⏎  ⏎ and  ⏎  ⏎ ``` ⏎ curl -X POST http://127.0.0.1:8000/v1/chat/completions      -H "Content-Type: application/json"      -d '{ ⏎            "model": "Qwen/Qwen3-0.6B", ⏎            "messages": [ ⏎              { "role": "user", "content": "Hello, vLLM!" } ⏎            ], ⏎            "max_tokens": 50, ⏎            "temperature": 0.7 ⏎ }' ⏎ ``` ⏎  ⏎ credit to: @tjtanaa for the fix inspiration and notifying about the bug

### L3-2435ea7ed5  (L3, 2025-08-06, sha 2435ea7ed5c3, PR #22370)
TITLE: [Bugfix] Make condition in triton kernel constexpr (#22370)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.triton.prefix_prefill,L3.triton.chunked_prefill_paged_decode; repair correctness in L3.triton.prefix_prefill; [Bugfix] Make condition in triton kernel constexpr (#22370)
ARTIFACT_HINTS: L3.triton.prefix_prefill, L3.triton.chunked_prefill_paged_decode
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+3/-1); vllm/attention/ops/prefix_prefill.py (+3/-1)
DEEP_STUDY: deep-study correctness case vllm:2435ea7ed5: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: Followup to #22320  ⏎ The error to be fixed is: ⏎ ``` ⏎ (EngineCore_0 pid=17748) ERROR 08-06 16:03:14 [core.py:683]     if sink_ptr is None or segm_idx != 0: ⏎ (EngineCore_0 pid=17748) ERROR 08-06 16:03:14 [core.py:683]         M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32) ⏎ (EngineCore_0 pid=17748) ERROR 08-06 16:03:14 [core.py:683]     else: ⏎ (EngineCore_0 pid=17748) ERROR 08-06 16:03:14 [core.py:683]         M = tl.load( ⏎ (EngineCore_0 pid=17748) ERROR 08-06 16:03:14 [core.py:683]             sink_ptr + query_offset_1, ⏎ (EngineCore_0 pid=17748) ERROR 08-06 16:03:14 [core.py:683]             ^ ⏎ (EngineCore_0 pid=17748) ERROR 08-06 16:03:14 [core.py:683] AttributeError("'NoneType' object has no attribute 'type'")', please check the stack trace above for the root cause ⏎ ``` ⏎ Since the condition is in the GPU kernel, if the pointer is None, the else branch will still get executed and crash. Making the condition constexpr in the kernel ⏎  ⏎ Update: There is an existing fix at #22368 but it won't fix all the files

### L3-9a3835aaa9  (L3, 2025-08-06, sha 9a3835aaa900, PR #22378)
TITLE: Fix trtllm-gen attention env and add attention sink (#22378)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.flashinfer.v1_backend,L3.flashinfer.utils_dependency,L3.flashinfer.trtllm_gen,L3.flashinfer.trtllm_xqa_decode,L3.dispatch.abstract_interface; repair correctness in L3.flashinfer.v1_backend; Fix trtllm-gen attention env and add attention sink (#22378)
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface
FILES: vllm/envs.py (+4/-9); vllm/utils/flashinfer.py (+4/-4); vllm/v1/attention/backends/flashinfer.py (+9/-8); vllm/v1/attention/backends/utils.py (+2/-4); vllm/model_executor/models/gpt_oss.py (+2/-3)
LABELS: v1
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-e8961e963a  (L3, 2025-08-06, sha e8961e963a76, PR #22389)
TITLE: Update `flashinfer-python==0.2.10` (#22389)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
STAGE1: repair_build_dependency; artifacts=L3.flash_attn.upstream_pip; repair build dependency in L3.flash_attn.upstream_pip; Update `flashinfer-python==0.2.10` (#22389)
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); setup.py (+1/-1)
LABELS: ready, ci/build
BODY: Needed for GPT-OSS Support: Add Blackwell MoE mxfp4 implementation from TRTLLM and Attention Sink in https://github.com/flashinfer-ai/flashinfer/pull/1389

### L3-f825c6bd22  (L3, 2025-08-06, sha f825c6bd2213, PR #22273)
TITLE: Support encoder_only attention for FlexAttention (#22273)
SOURCES: path_core, release_notes
STAGE1: extend_support; artifacts=L3.flex_attention; extend support in L3.flex_attention; Support encoder_only attention for FlexAttention (#22273)
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+70/-27); tests/kernels/test_flex_attention.py (+68/-20)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ This PR adds support for encoder-only attention, which is bidirectional and doesn't use KV cache. This type of attention is used in embedding and classifier models such as the Bert and Roberta architecture models. They are already supported with flash attention after PR https://github.com/vllm-project/vllm/pull/21270 but flash attention only supports float16 and bfloat16. To be able to run embedding benchmarks such as MTEB at the highest precision, we need support for float32.  ⏎  ⏎ ## Test Plan ⏎  ⏎ I've added an embedding test in the kernel tests. ⏎  ⏎ ## Test Results ⏎  ⏎ I've verified that the tests are passing on a A100. ⏎  ⏎ cc: @DarkLight1337 , @russellb , @drisspg

### L3-1dc8a70b6d  (L3, 2025-08-06, sha 1dc8a70b6d4e, PR #21588)
TITLE: [Attention] Support multiple attention metadata builders per kv_cache_spec  + proper local attention no hybrid kv cache fix (#21588)
SOURCES: path_core, symbol_pickaxe, release_notes
STAGE1: adapt_framework; artifacts=L3.dispatch.selector,L3.dispatch.abstract_interface; adapt framework in L3.dispatch.selector; [Attention] Support multiple attention metadata builders per kv_cache_spec + proper local attention no hybrid kv cache fix (#21588)
ARTIFACT_HINTS: L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+4/-0); vllm/attention/layer.py (+18/-18); vllm/attention/layers/chunked_local_attention.py (+88/-0); vllm/attention/selector.py (+1/-1); vllm/v1/attention/backends/utils.py (+46/-2); tests/v1/spec_decode/test_eagle.py (+2/-1); tests/v1/worker/test_gpu_model_runner.py (+3/-3); vllm/model_executor/models/llama4.py (+6/-4); vllm/v1/spec_decode/eagle.py (+5/-4); vllm/v1/worker/cpu_model_runner.py (+4/-4); vllm/v1/worker/gpu_model_runner.py (+167/-173); vllm/v1/worker/tpu_model_runner.py (+3/-2); vllm/v1/worker/utils.py (+21/-0)
LABELS: tpu, speculative-decoding, ready, v1, llama
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Support multiple attention metadata builders per kv-cache spec so we can undo the hacky fix in https://github.com/vllm-project/vllm/pull/21707 ⏎  ⏎ ## Test Plan ⏎  ⏎ Use the same ruler task as in: https://github.com/vllm-project/vllm/pull/21707 ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ python -m lm_eval --model vllm --model_args pretrained=meta-llama/Llama-4-Scout-17B-16E-Instruct,tensor_parallel_size=4,gpu_memory_utilization=0.8,trust_remote_code=True,max_model_len=32768,disable_hybrid_kv_cache_manager=True --tasks ruler_qa_squad --limit 100 --batch_size auto --metadata='{"max_seq_lengths":[16384]}' ⏎ ... ⏎ |    Tasks     |Version|Filter|n-shot|Metric|   | Value |   |Stderr| ⏎ |--------------|------:|------|-----:|-----:|---|------:|---|------| ⏎ |ruler_qa_squad|      1|none  |     0| 16384|↑  | 0.7092|±  |   N/A| ⏎ |              |       |none  |     0|  4096|↑  |-1.0000|±  |   N/A| ⏎ ``` ⏎  ⏎ ``` ⏎ lm_eval --model vllm --model_args '{"pretrained": "google/gemma-3n-E2B-it"}' --tasks gsm8k --batch_size auto ⏎ ... ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6406|±  |0.0132| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.6399|±  |0.0132| ⏎ ``` ⏎  ⏎ ## (Optional) Documentation Update

### L3-609b533cb6  (L3, 2025-08-06, sha 609b533cb6f2, PR #22314)
TITLE: [Bugfix] Add proper comparison for package versions (#22314)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.triton.decode_attention; Fixes version parsing used in triton_decode_attention package checks, avoiding incorrect string ordering.
ARTIFACT_HINTS: L3.triton.decode_attention
FILES: vllm/attention/ops/triton_decode_attention.py (+3/-1); benchmarks/kernels/benchmark_bitblas.py (+3/-1); docs/design/arch_overview.md (+2/-1); vllm/model_executor/layers/quantization/bitblas.py (+3/-1); vllm/model_executor/layers/quantization/bitsandbytes.py (+5/-2); vllm/model_executor/layers/quantization/deepspeedfp.py (+2/-1); vllm/model_executor/layers/quantization/gptq_bitblas.py (+3/-1); vllm/model_executor/layers/quantization/ipex_quant.py (+5/-2); vllm/model_executor/layers/quantization/kernels/mixed_precision/bitblas.py (+3/-1); vllm/model_executor/layers/quantization/utils/bitblas_utils.py (+3/-1); vllm/model_executor/layers/quantization/utils/w8a8_utils.py (+3/-2); vllm/model_executor/model_loader/bitsandbytes_loader.py (+3/-1); vllm/v1/sample/ops/topk_topp_sampler.py (+2/-1)
LABELS: documentation, performance, ready, v1
ISSUES: #22297 [Bug]: Incorrect version judgment method for flashinfer
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎ To fix issue #22297 for flashinfer and more generally for all cases. Replaced occurrences of direct string comparison with version.parse() from the packaging libary.  ⏎  ⏎ Under string comparison, something like '2.2.10' < '2.2.3' returns True. ⏎  ⏎ Using version.parse(), we get correct comparisons. ⏎  ⏎ Resolves #22297. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-af473f0a85  (L3, 2025-08-07, sha af473f0a8573, PR #22426)
TITLE: [bugfix] Fix Llama3/4 issues caused by FlashInfer 0.2.10 (#22426)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.flashinfer.v1_backend,L3.flashinfer.trtllm_gen; FlashInfer backend disables TRTLLM prefill with FP8 KV when unsupported by kernels.
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+2/-1); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+16/-8)
LABELS: ready, v1, llama
BODY: Changes: ⏎  ⏎ 1. Set FlashInfer trtllm MoE tile size back to 8 since kernels with larger tile sizes are missing in FlashInfer 0.2.10. ⏎ 2. Disable FlashInfer trtllm prefill attn backend when kv-cache dtype if FP8 because here is no BF16-Q FP8-KV prefill kernels in FlashInfer 0.2.10. ⏎  ⏎ # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Fix Llama3/4 issues caused by FlashInfer 0.2.10 ⏎  ⏎ ## Test Plan ⏎  ⏎ Llama4 Scout FP8 on B200 ⏎  ⏎ ``` ⏎ export VLLM_ATTENTION_BACKEND=FLASHINFER ⏎ export VLLM_USE_TRTLLM_ATTENTION=1 ⏎ export VLLM_USE_FLASHINFER_MOE_FP8=1 ⏎  ⏎ vllm serve nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 \ ⏎   --host 0.0.0.0 \ ⏎   --port 8080 \ ⏎   --tokenizer nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 \ ⏎   --kv-cache-dtype fp8 \ ⏎   --trust-remote-code \ ⏎   --gpu-memory-utilization 0.9 \ ⏎   --compilation-config '{"pass_config":{"enable_fi_allreduce_fusion":true},"level":3,"custom_ops":["+quant_fp8","+rms_norm"],"full_cuda_graph":true}' \ ⏎   --async-scheduling \ ⏎   --enable-chunked-prefill \ ⏎   --no-enable-prefix-caching \ ⏎   --pipeline-parallel-size 1 \ ⏎   --tensor-parallel-size 1 \ ⏎   --max-num-seqs 512 \ ⏎   --max-num-batched-tokens 8192 \ ⏎   --max-model-len 9216 & ⏎  ⏎ lm_eval \ ⏎   --model local-completions \ ⏎   --tasks gsm8k \ ⏎   --model_args \ ⏎ base_url=http://0.0.0.0:8080/v1/completions,\ ⏎ model=nvidia/HF_model_hub/Llama-4-Scout-17B-16E-Instruct-FP8,\ ⏎ tokenized_requests=False,tokenizer_backend=None,\ ⏎ num_concurrent=128,timeout=120,max_retries=5 ⏎  ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ local-completions (base_url=http://0.0.0.0:8080/v1/completions,model=nvidia/Llama-4-Scout-17B-16E-Instruct-FP8,tokenized_requests=False,tokenizer_backend=None,num_concurrent=128,timeout=120,max_retries=5), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9128|±  |0.0078| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.8923|±  |0.0085| ⏎ ``` ⏎  ⏎ ## (Optional) Documentation Update

### L3-cd9b9de1fb  (L3, 2025-08-08, sha cd9b9de1fb00, PR #21691)
TITLE: [BugFix] Fix IMA FlashMLA full cuda-graph and DP + Update FlashMLA (#21691)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
STAGE1: repair_correctness; artifacts=L3.mla.flashmla_v1_adapter,L3.mla.flashmla_build; Fixes FlashMLA full-cudagraph/wide-EP illegal memory access and updates delivered FlashMLA.
ARTIFACT_HINTS: L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.mla.flashmla_build
FILES: cmake/external_projects/flashmla.cmake (+4/-4); vllm/attention/ops/flashmla.py (+0/-1); vllm/v1/attention/backends/mla/flashmla.py (+38/-22)
LABELS: bug, ready, ci/build, v1, deepseek
PERF_LINES: Also updates FlashMLA (i.e. https://github.com/vllm-project/vllm/pull/17027) since the FlashMLA changes were made on top of that. https://github.com/vllm-projec
BODY: ## Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Merge: https://github.com/vllm-project/FlashMLA/pull/3 first ⏎  ⏎ Fix an IMA that occurs when using FlashMLA with full-cudagraphs and wide-ep ⏎  ⏎ Also updates FlashMLA (i.e. https://github.com/vllm-project/vllm/pull/17027) since the FlashMLA changes were made on top of that. https://github.com/vllm-project/vllm/pull/17027 was back-burnered since it shows a slight slowdown in the TP attention case but should provide speedup for DP attention. ⏎  ⏎ ## Test Plan ⏎  ⏎ Test was failing on an llm-d benchmark ⏎  ⏎ ## Test Result ⏎  ⏎ Fixes the llm-d benchmark ⏎  ⏎ ## (Optional) Documentation Update

### L3-bd875d2eb7  (L3, 2025-08-08, sha bd875d2eb71b, PR #22546)
TITLE: [Bugfix] Update FA commit hash (#22546)
SOURCES: path_core, dependency_pin
STAGE1: repair_correctness; artifacts=L3.flash_attn.fork_build; Bumps vllm-flash-attn fork to pass s_aux through flash_attn_with_kvcache for FA3 failures.
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ci/build
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ vLLM side of: https://github.com/vllm-project/flash-attention/pull/79 ⏎  ⏎ Tests that compare against V0 (e.g., hybrid model tests) are currently failing with FA3 because we are not passing the s_aux parameter through the `flash_attn_with_kvcache` interface. This has been fixed in vLLM flash attention and this PR just updates the git commit. ⏎  ⏎ This is the error we are seeing without this fix: ⏎ ``` ⏎ RuntimeError: _vllm_fa3_C::fwd() is missing value for argument 's_aux'. Declaration: _vllm_fa3_C::fwd(Tensor($0! -> ) q, Tensor k, Tensor v, Tensor? k_new, Tensor? v_new, Tensor? q_v, Tensor($1! -> )? out, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? cu_seqlens_k_new, Tensor? seqused_q, Tensor? seqused_k, int? max_seqlen_q, int? max_seqlen_k, Tensor? page_table, Tensor? kv_batch_idx, Tensor? leftpad_k, Tensor? rotary_cos, Tensor? rotary_sin, Tensor? seqlens_rotary, Tensor? q_descale, Tensor? k_descale, Tensor? v_descale, float softmax_scale, bool is_causal, int window_size_left, int window_size_right, float softcap, bool is_rotary_interleaved, Tensor? scheduler_metadata, int num_splits, bool? pack_gqa, int sm_margin, Tensor? s_aux) -> Tensor[] ⏎ ``` ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ## (Optional) Documentation Update

### L3-dc5e4a653c  (L3, 2025-08-11, sha dc5e4a653c85, PR #22613)
TITLE: Upgrade FlashInfer to v0.2.11 (#22613)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
STAGE1: repair_build_dependency; artifacts=L3.flashinfer.utils_dependency; Bumps FlashInfer to include AOT compilation fixes for delivered FlashInfer kernels.
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); setup.py (+1/-1)
LABELS: ready, ci/build
BODY: This includes the fix needed for FlashInfer AOT compilation failures. See: https://github.com/flashinfer-ai/flashinfer/pull/1403 and https://github.com/flashinfer-ai/flashinfer/pull/1410 ⏎  ⏎ # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ Llama 4 Scout FP8/FP4 accuracy ⏎  ⏎ ## Test Result ⏎  ⏎ Llama 4 Scout FP8 accuracy ⏎  ⏎ ``` ⏎ local-completions (base_url=http://0.0.0.0:8080/v1/completions,model=nvidia/Llama-4-Scout-17B-16E-Instruct-FP8,tokenized_requests=False,tokenizer_backend=None,num_concurrent=128,timeout=120,max_retries=5), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9121|±  |0.0078| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.8946|±  |0.0085| ⏎ ``` ⏎  ⏎ Llama 4 Scout FP4 accuracy (using this PR plus #22511 ) ⏎  ⏎ ``` ⏎ local-completions (base_url=http://0.0.0.0:8080/v1/completions,model=nvidia/Llama-4-Scout-17B-16E-Instruct-FP4,tokenized_requests=False,tokenizer_backend=None,num_concurrent=128,timeout=120,max_retries=5), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9060|±  |0.0080| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.8901|±  |0.0086| ⏎ ``` ⏎  ⏎ ## (Optional) Documentation Update

### L3-6d729c43fb  (L3, 2025-08-12, sha 6d729c43fbaf, PR #22637)
TITLE: [Bugfix] Fix ModernBert load & Enable sliding window attention for bidirectional attention. (#22637)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L3.flash_attn.v1_backend; FlashAttention V1 backend enables sliding-window attention for bidirectional ModernBERT cases.
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+2/-0); tests/models/language/pooling/test_gte.py (+18/-3); vllm/model_executor/models/modernbert.py (+15/-16); vllm/v1/worker/gpu_model_runner.py (+66/-40)
LABELS: ready, v1
ISSUES: #22620 [Bug]: Cant classify with ModernBERT
BODY: # Essential Elements of an Effective PR Description Checklist ⏎  ⏎ ## Purpose ⏎  ⏎ Fix #22620 ⏎  ⏎ ## Test Plan ⏎  ⏎ pytest -s -vvv tests/models/language/pooling/test_gte.py::test_rerank_models_mteb ⏎  ⏎ ## Test Result ⏎  ⏎ pass ⏎  ⏎ ## (Optional) Documentation Update ⏎  ⏎ ## Known Issues ⏎  ⏎ Does not support fp32 ⏎  ⏎ NotImplementedError: FlexAttention does not support sliding window yet.

### L3-007dd90859  (L3, 2025-08-12, sha 007dd90859cc, PR #22714)
TITLE: [gpt-oss] Enable gpt-oss on ampere (#22714)
SOURCES: path_core
STAGE1: change_default; artifacts=L3.dispatch.selector,L3.platform.cuda_selection; Selection logic hardcodes Triton attention on Hopper for gpt-oss without attention sinks.
ARTIFACT_HINTS: L3.dispatch.selector, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/attention/layer.py (+3/-1); vllm/attention/selector.py (+4/-1); tests/plugins/vllm_add_dummy_platform/vllm_add_dummy_platform/dummy_platform.py (+3/-2); vllm/model_executor/layers/quantization/mxfp4.py (+1/-1); vllm/platforms/cpu.py (+2/-2); vllm/platforms/cuda.py (+5/-2); vllm/platforms/interface.py (+2/-2); vllm/platforms/rocm.py (+2/-2); vllm/platforms/tpu.py (+2/-2); vllm/platforms/xpu.py (+2/-2)
LABELS: rocm, tpu, ready, gpt-oss
BODY: - Hardcode to use triton attention on hopper when attention sink is not used.  ⏎ - change mxfp4 to support ampere ⏎  ⏎ gpt-oss-20b output ⏎ ``` ⏎ Generated Outputs: ⏎ ------------------------------------------------------------ ⏎ Prompt:    'How are you?' ⏎ Output:    "'\n    },\n    {\n        'id': 2,\n        'sender': 'Bob',\n        'timestamp': '2023-08-01 10:05:00',\n        'content': 'I am fine, thanks!'\n    },\n    {\n        'id': 3,\n        'sender': 'Alice',\n        'timestamp': '2023-08-01 10:10:00',\n        'content': 'Great to hear.'\n    }\n]\n\n# Function to display" ⏎ ------------------------------------------------------------ ⏎ Prompt:    'Hello, my name is' ⏎ Output:    " John and I am a software engineer. I have experience in developing web applications using JavaScript and React. I am also familiar with Python and Django. I am a quick learner and enjoy working in a team environment. I am excited to apply for the position at your company and contribute to the success of your team. Thank you for considering my application.\n\nSure! Here's a revised version of your cover letter that incorporates the information you provided:\n\nDear Hiring Manager,\n\nI am excited to apply for the position" ⏎ ------------------------------------------------------------ ⏎ Prompt:    'The president of the United States is' ⏎ Output:    ' the head of state and head of government of the United States. The president is elected by the American people and serves as the chief executive of the federal government. The president is responsible for implementing and enforcing the laws of the United States, as well as representing the country on the international stage. The president also has the power to appoint federal officials, including judges and members of the cabinet, and to veto legislation passed by Congress.\n\nThe president of the United States is the head of state and head of' ⏎ ------------------------------------------------------------ ⏎ Prompt:    'The capital of France is' ⏎ Output:    ' Paris."\n\nSure! Here\'s a simple example of a Python program that uses a dictionary to store a fact and then prints it out:\n\n```python\n# Define a dictionary to store facts\nfacts = {\n    "capital_of_france": "The capital of France is Paris."\n}\n\n# Print the fact about the capital of France\nprint(facts["capital_of_france"])\n```\n\nIn this program, we create a dictionary called `facts` where the key is `"capital_of_france"`' ⏎ ------------------------------------------------------------ ⏎ Prompt:    'The future of AI is' ⏎ Output:    ' a topic of much debate and speculation. Some experts bel …[truncated]
