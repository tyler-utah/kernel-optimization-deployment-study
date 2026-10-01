### L2-1c63e79756  (L2, 2025-03-31, sha 1c63e7975604, PR #4954)
TITLE: use fa3 in sgl-kernel (#4954)
SOURCES: dependency_pin
STAGE1: repair_build_dependency; artifacts=L2.backend.fa3_fa4_mla,L2.upstream.flashattention_mla; Switches FA3 backend to sgl-kernel packaged FlashAttention implementation.
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e8999b13b7  (L2, 2025-04-03, sha e8999b13b7c3, PR #5005)
TITLE: Replace enable_flashinfer_mla argument with attention_backend (#5005)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts(reverted)
STAGE1: adapt_framework; artifacts=L2.backend.flashinfer_mla,L2.dispatch.server_args_defaults; Replaces --enable-flashinfer-mla flow with attention_backend dispatch for FlashInfer MLA.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+0/-2); python/sglang/srt/model_executor/model_runner.py (+6/-3); python/sglang/srt/models/deepseek_v2.py (+1/-2); python/sglang/srt/server_args.py (+2/-2); docs/backend/server_arguments.md (+3/-3); docs/references/deepseek.md (+2/-2); python/sglang/srt/managers/schedule_batch.py (+1/-2); test/srt/test_mla_flashinfer.py (+6/-4)
DEEP_STUDY: deep-study: this PR was reverted by PR 5048 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Motivation ⏎  ⏎ Currently there are many mla backends, making the arguments a little messy. ⏎ After this PR, the functionality of `--enable-flashinfer-mla` can be replaced by `--attention-backend flashinfer`. `--enable-flashinfer-mla` can still be used as before, but it's supposed to be deprecated in following versions. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-b8b6008f47  (L2, 2025-04-03, sha b8b6008f47de, PR #5036)
TITLE: [Fix] fix fa3 build at cu118 (#5036)
SOURCES: dependency_pin
STAGE1: repair_build_dependency; artifacts=L2.backend.fa3_fa4_mla,L2.upstream.flashattention_mla; Fixes CUDA 11.8 build for sgl-kernel FA3 library used by FA3 backend.
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+85/-50); sgl-kernel/cmake/utils.cmake (+21/-0); sgl-kernel/csrc/common_extension.cc (+1/-40); sgl-kernel/csrc/flash_extension.cc (+62/-0); sgl-kernel/include/sgl_flash_kernel_ops.h (+85/-0); sgl-kernel/include/sgl_kernel_ops.h (+0/-47); sgl-kernel/python/sgl_kernel/flash_attn.py (+15/-4); sgl-kernel/tests/test_flash_attention.py (+19/-1)
ISSUES: #4941 [Feature] use different lib so for fa3 in sgl-kernel
DEEP_STUDY: deep-study correctness case sglang:b8b6008f47: class=hardware_compiler_specific; symptom=compile_or_build_failure; introducing=unknown
BODY: ## Motivation ⏎  ⏎  ⏎ Fix cu118 for fa3 compile error ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-74885a848b  (L2, 2025-04-03, sha 74885a848bb6, PR #5048)
TITLE: Revert "Replace enable_flashinfer_mla argument with attention_backend" (#5048)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts
STAGE1: revert; artifacts=L2.backend.flashinfer_mla,L2.dispatch.server_args_defaults; Reverts FlashInfer MLA attention_backend flag replacement because it broke CI.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+2/-0); python/sglang/srt/model_executor/model_runner.py (+3/-6); python/sglang/srt/models/deepseek_v2.py (+2/-1); python/sglang/srt/server_args.py (+2/-2); docs/backend/server_arguments.md (+3/-3); docs/references/deepseek.md (+2/-2); python/sglang/srt/managers/schedule_batch.py (+2/-1); test/srt/test_mla_flashinfer.py (+4/-6)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 5005 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#5005 because it breaks CI

### L2-efbae697b3  (L2, 2025-04-05, sha efbae697b370, PR #5052)
TITLE: [Revision] Replace enable_flashinfer_mla argument with attention_backend (#5052)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: adapt_framework; artifacts=L2.backend.flashinfer_mla,L2.dispatch.server_args_defaults; Revises FlashInfer MLA attention_backend replacement and fixes default backend bug.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+0/-2); python/sglang/srt/model_executor/model_runner.py (+49/-38); python/sglang/srt/models/deepseek_v2.py (+1/-2); python/sglang/srt/server_args.py (+3/-7); python/sglang/srt/speculative/eagle_worker.py (+24/-22); docs/backend/server_arguments.md (+3/-3); docs/references/deepseek.md (+2/-2); python/sglang/srt/managers/schedule_batch.py (+4/-2); test/srt/test_mla_flashinfer.py (+6/-4)
BODY: ## Motivation ⏎ Fixing the default backend bug caused by #5005. Now w/wo mla, backend is set to triton/flashinfer by default. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f65b8d5c89  (L2, 2025-04-11, sha f65b8d5c896c, PR #5142)
TITLE: Blackwell Cutlass MLA kernel (#5142)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
STAGE1: introduce; artifacts=L2.kernel.cutlass_mla; Adds Blackwell CUTLASS MLA kernel, Python wrapper, extension binding, and tests.
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/CMakeLists.txt (+4/-1); sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+207/-0); sgl-kernel/csrc/common_extension.cc (+5/-0); sgl-kernel/include/sgl_kernel_ops.h (+8/-1); sgl-kernel/python/sgl_kernel/__init__.py (+5/-1); sgl-kernel/python/sgl_kernel/attention.py (+61/-0); sgl-kernel/tests/test_cutlass_mla.py (+81/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Adds blackwell cutlass MLA kernel to sgl-kernel. Requires cutlass 3.9 ⏎ Thanks to @kaixih for kernel and test code   ⏎  ⏎ I will open a second PR soon to add the cutlass mla attention backend. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-57de7c6b5f  (L2, 2025-04-12, sha 57de7c6b5fb3, PR #5210)
TITLE: feat: use fa3 mla by default on hopper (#5210)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: change_default; artifacts=L2.backend.fa3_fa4_mla,L2.dispatch.server_args_defaults; Switches Hopper DeepSeek MLA default path to FA3 and adds DP MLA support.
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+12/-7); python/sglang/srt/model_executor/model_runner.py (+21/-4); python/sglang/srt/utils.py (+9/-0)
LABELS: high priority
PERF_LINES: Latency: 96.392 s | Output throughput: 325.570 token/s
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ - [Support DP MLA for FA3](https://github.com/sgl-project/sglang/pull/5210/commits/77f31ed842c71f580c6288e2caf19ae7ad99342c)  ⏎  ⏎ ## Benchmark ⏎ ```bash ⏎ python3 -m sglang.launch_server --model-path /tmp/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205/ --trust-remote-code --tp 8 --enable-dp-attention --dp 8 --host 127.0.0.1 --attention-backend fa3   ⏎  ⏎ python /home/jobuser/sglang/benchmark/gsm8k/bench_sglang.py ⏎  ⏎ Accuracy: 0.975 ⏎ Invalid: 0.000 ⏎ Latency: 96.392 s ⏎ Output throughput: 325.570 token/s ⏎ ``` ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-812e82f35e  (L2, 2025-04-12, sha 812e82f35e09, PR #5331)
TITLE: fix: solve cu118 issue for cutlass mla (#5331)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:kernel-correctness-cases
STAGE1: repair_build_dependency; artifacts=L2.kernel.cutlass_mla; Fixes cu118 build failure in the CUTLASS MLA kernel source and CI build.
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+4/-0); .github/workflows/pr-test-sgl-kernel.yml (+12/-6); .github/workflows/release-whl-kernel.yml (+1/-1)
DEEP_STUDY: deep-study correctness case sglang:812e82f35e: class=hardware_compiler_specific; symptom=compile_or_build_failure; introducing=unknown
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e9fc2ac7b6  (L2, 2025-04-14, sha e9fc2ac7b611, PR #5384)
TITLE: [PD Bug] fix  MLA get_contiguous_buf_infos error (#5384)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L2.pool.mla_token_kv; Fixes MLA memory pool get_contiguous_buf_infos error for PD disaggregation.
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/memory_pool.py (+4/-9)
BODY: ## Motivation ⏎ [PD Bug] fix  MLA get_contiguous_buf_infos error ⏎ ## Modifications

### L2-e8f62b20ca  (L2, 2025-04-15, sha e8f62b20ca88, PR #5431)
TITLE: BLackwell cutlass mla: Add check for bad page size/block num combinations (#5431)
SOURCES: subject_keyword, release_notes
STAGE1: extend_support; artifacts=L2.kernel.cutlass_mla; Extends CUTLASS MLA wrapper to allow non-128 page sizes with block-number guard.
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/python/sgl_kernel/attention.py (+5/-3); sgl-kernel/tests/test_cutlass_mla.py (+6/-1)
BODY: ## Motivation ⏎  ⏎ This PR allows the blackwell cutlass mla kernel to support page sizes other than 128, as long as `block_num % (128 / PAGE_SIZE) == 0` ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-fa909dc3c4  (L2, 2025-04-15, sha fa909dc3c40a, PR #5344)
TITLE: feat: update model_specific_adjustment (#5344)
SOURCES: symbol_pickaxe
STAGE1: repair_correctness; artifacts=L2.backend.fa3_fa4_mla; Adjusts FA3 backend/model-specific metadata to fix mixed-chunk batch-size mismatch.
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+1/-1); python/sglang/srt/model_executor/forward_batch_info.py (+8/-4); python/sglang/srt/model_executor/model_runner.py (+16/-11); python/sglang/srt/utils.py (+26/-1)
LABELS: high priority
PERF_LINES: Latency: 99.928 s | Output throughput: 1343.914 token/s
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Fixed the issue: "RuntimeError: batch_size must be equal to batch_size_k" when `--enable-mixed-chunk` is enabled ⏎  ⏎ ```bash ⏎ python3 -m sglang.launch_server --model  /shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/ --chunked-prefill-size 32 --log-level debug --enable-mixed-chunk --attention-backend fa ⏎ 3 ⏎  ⏎  ⏎ Accuracy: 0.794 ⏎ Invalid: 0.000 ⏎ Latency: 99.928 s ⏎ Output throughput: 1343.914 token/s ⏎ ``` ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-a42736bbb8  (L2, 2025-04-15, sha a42736bbb8fe, PR #5113)
TITLE: Support MHA with chunked prefix cache for DeepSeek chunked prefill (#5113)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L2.model.deepseek_v2_mla,L2.optimization.weight_absorption,L2.backend.fa3_fa4_mla; Adds chunked-prefix MHA path for DeepSeek MLA prefill to avoid slow absorbed long-prefix computation.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+80/-34); python/sglang/srt/model_executor/forward_batch_info.py (+181/-0); python/sglang/srt/model_executor/model_runner.py (+11/-0); python/sglang/srt/models/deepseek_v2.py (+174/-9); python/sglang/srt/server_args.py (+6/-0); docs/backend/server_arguments.md (+1/-0); docs/references/deepseek.md (+3/-1); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/test/attention/test_prefix_chunk_info.py (+224/-0); test/srt/test_fa3.py (+53/-2)
LABELS: high priority, performance, deepseek
PERF_LINES: Latency: 92.794 s | Output throughput: 1544.409 token/s | Total latency: 144.079 | All benchmarks are tested on 8*H200 with DeepGemm disabled.  Higher throughput can be obtained through enabling and warming up DeepGemm. | | Throughput (tok/s)| After PR | Before PR  | | | Mean TTFT (ms) | 185310.57 | 259006.92 | | | Mean TPOT (ms) | 214.13 | 280.73  | | | Throughput (tok/s)| After PR | Before PR  |
BODY: ## Motivation ⏎  ⏎ The current implementation of MLA is slow when when handling long prefix lengths, such as sequences with 32k input length, as highlighted in issues like #5031. ⏎  ⏎ Profiling revealed that the attention kernel's slow runtime is the primary bottleneck.  Currently, we perform absorption during chunked prefilling, but this approach is inefficient for long prefix lengths, as MLA with absorption has overly large computational intensity when `q` contains many tokens. Conversely, while multi-head attention (MHA) is computationally efficient, materializing key-value (KV) heads often leads to out-of-memory (OOM) errors.  ⏎  ⏎ we propose dividing long prefix KV caches into multiple chunks. The output tensors and log-sum-exp (LSE) states for each chunk can be merged into a single accumulated output and LSE state. This approach enables MHA during chunked prefilling while avoiding OOM issues. ⏎  ⏎ Inspired by solutions in [vllm](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py) and [flashinfer](https://docs.flashinfer.ai/tutorials/recursive_attention.html), we propose dividing long prefix KV caches into multiple chunks.The output tensors and log-sum-exp (LSE) states for each chunk can be merged into a single accumulated output and LSE state. This approach enables MHA during chunked prefilling while avoiding OOM issues. ⏎  ⏎ In our design, the chunked prefix cache is enabled by default for DeepSeek models (unless page_size > 1). Users can disable it by passing the server argument `--disable-chunked-prefix-cache`. Currently, the chunked prefix cache is supported only on the FA3 backend. ⏎  ⏎ For short input cases (e.g., 128 or 256 tokens), this optimization introduces overhead that degrades performance. Therefore, it is activated only when the total prefix cache length exceeds a threshold (currently set to 8,192 tokens). In the future, a more refined trigger condition should be developed. ⏎  ⏎ Really want to thank @ispobock for helping with debugging, and thank @DefTruth for implementing merge_state kernel~ ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Add prefix chunk preparing and management features for `ForwardBatch`. ⏎ - Change the logic of forward function dispatching in deepseek models. ⏎ - Add a new forward method for deepseek model, which handles chunked prefilling with chunked prefix caches. ⏎ - Add MHA logic in FlashAttention backend with `flash_attn_with_varlen_func` API. ⏎ - Add a test for checking chunking logic. ⏎  ⏎ ## TODO List ⏎  ⏎  ⏎ ## Accuracy ⏎  ⏎ ### Launch ⏎ ```bash ⏎ python3 -m sglang.launch_server --model /dev/shm/DeepSeek-V …[truncated]

### L2-4fb05583ef  (L2, 2025-04-17, sha 4fb05583ef38, PR #5481)
TITLE: Deprecate disable-mla (#5481)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: deprecate; artifacts=L2.dispatch.server_args_defaults,L2.model.deepseek_v2_mla; Removes disable-MLA flag paths, deprecating non-MLA DeepSeek dispatch control.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+1/-3); python/sglang/srt/model_executor/model_runner.py (+1/-5); python/sglang/srt/models/deepseek_nextn.py (+63/-67); python/sglang/srt/models/deepseek_v2.py (+93/-290); python/sglang/srt/server_args.py (+0/-6); docs/backend/server_arguments.md (+0/-1); docs/references/deepseek.md (+1/-1); python/sglang/srt/managers/schedule_batch.py (+0/-1); python/sglang/srt/models/minicpm3.py (+29/-201)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-6fb29ffd9e  (L2, 2025-04-17, sha 6fb29ffd9e7b, PR #5480)
TITLE: Deprecate enable-flashinfer-mla and enable-flashmla (#5480)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: deprecate; artifacts=L2.dispatch.server_args_defaults,L2.backend.flashinfer_mla,L2.backend.flashmla; Deprecates FlashInfer-MLA and FlashMLA boolean flags in favor of attention-backend selection.
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/model_executor/model_runner.py (+7/-11); python/sglang/srt/server_args.py (+6/-8); docs/backend/server_arguments.md (+0/-1); docs/references/deepseek.md (+1/-1); python/sglang/srt/managers/schedule_batch.py (+1/-2); scripts/playground/bench_speculative.py (+3/-8)
BODY: ## Motivation ⏎  ⏎ Deprecate two arguments: `--enable-flashinfer-mla` and `--enable-flashmla` for clarity. ⏎ They should be replaced by `--attention-backend flashinfer` and `--attention-backend flashmla` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-bfa3922451  (L2, 2025-04-18, sha bfa392245159, PR #5476)
TITLE: Avoid computing lse in Ragged Prefill when there's no prefix. (#5476)
SOURCES: path_core
STAGE1: optimize; artifacts=L2.backend.flashinfer_mla; Changes FlashInfer MLA ragged prefill to avoid computing LSE when no prefix exists.
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-1); docs/backend/server_arguments.md (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+17/-10)
DEEP_STUDY: deep-study: this PR was reverted by PR 5544 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Motivation ⏎ Small tweak to save a bit of compute.  ⏎ cc @Fridge003  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-a6f892e5d0  (L2, 2025-04-18, sha a6f892e5d08f, PR #5544)
TITLE: Revert "Avoid computing lse in Ragged Prefill when there's no prefix.… (#5544)
SOURCES: path_core
STAGE1: revert; artifacts=L2.backend.flashinfer_mla; Reverts the FlashInfer MLA ragged-prefill LSE skip after reward-model test failure.
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-1); docs/backend/server_arguments.md (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+10/-17)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 5476 reason=ci_or_test_failure
BODY: … (#5476)" ⏎  ⏎ This reverts commit bfa392245159147a2b7dbd67178c825e5035c329. ⏎  ⏎ fix `python3 models/test_reward_models.py` ⏎  ⏎ @Edenzzzz May you submit a fix for this PR again? @Fridge003  ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-613b197e57  (L2, 2025-04-19, sha 613b197e5790, PR #5549)
TITLE: Remove one kernel in per_tensor_quant_mla_fp8 (#5549)
SOURCES: symbol_pickaxe
STAGE1: optimize; artifacts=L2.model.deepseek_v2_mla,L2.optimization.weight_absorption; Removes an extra kernel in DeepSeek FP8 MLA per-tensor quantization path.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/layers/quantization/fp8_kernel.py (+9/-7); python/sglang/srt/models/deepseek_nextn.py (+8/-2); python/sglang/srt/models/deepseek_v2.py (+32/-9); python/sglang/srt/utils.py (+13/-0)
PERF_LINES: Prefill. latency: 0.04585 s, throughput:   2791.45 token/s | Decode.  latency: 0.00547 s, throughput:    182.68 token/s | Decode.  latency: 0.00540 s, throughput:    185.02 token/s | Decode.  latency: 0.00539 s, throughput:    185.43 token/s | Decode.  latency: 0.00538 s, throughput:    185.93 token/s | Decode.  latency: 0.00535 s, throughput:    186.90 token/s | Decode.  median latency: 0.00529 s
BODY: ## Motivation ⏎  ⏎ Thanks @Alcanderian for discussing it is acceptable to change APIs in the caller site ⏎  ⏎ ### Accuracy ⏎  ⏎ ``` ⏎ python -m sglang.launch_server --model-path /dev/shm/DeepSeek-V3-0324 --trust-remote-code --tp 16 --dp 16 --enable-dp-attention --enable-deepep-moe --deepep-mode normal --disable-cuda-graph --disable-radix-cache --disable-overlap-schedule --decode-log-interval 1 --host 0.0.0.0 --port 20000 --dist-init-addr 10.10.38.8:15000 --nnodes 2 --node-rank 0 ⏎ (cd /host_home/primary_synced/sglang && while true; do python3 benchmark/gsm8k/bench_sglang.py --port 20000 --parallel 1400 --num-questions 1400; done) ⏎ ``` ⏎  ⏎ PR: 93.3 ⏎  ⏎ (baseline: roughly 93 when I tested in https://github.com/sgl-project/sglang/pull/5295) ⏎  ⏎ Another acc test ⏎  ⏎ ``` ⏎ (cd /host_home/primary_synced/sglang && python3 test/srt/test_mla_deepseek_v3.py) ⏎ ``` ⏎  ⏎ baseline (#5370): 0.64~0.67 ⏎  ⏎ PR: 0.645 ⏎  ⏎ ### Speed ⏎  ⏎ Using command in https://github.com/sgl-project/sglang/pull/5370: ⏎  ⏎ ``` ⏎ python3 -m sglang.bench_one_batch --model RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8 --batch-size 1 --input-len 128 --output-len 128 --trust-remote-code ⏎ ``` ⏎  ⏎ baseline ⏎  ⏎ ``` ⏎ Benchmark ... ⏎ Prefill. latency: 0.04585 s, throughput:   2791.45 token/s ⏎ Decode.  latency: 0.00547 s, throughput:    182.68 token/s ⏎ Decode.  latency: 0.00540 s, throughput:    185.02 token/s ⏎ Decode.  latency: 0.00539 s, throughput:    185.43 token/s ⏎ Decode.  latency: 0.00538 s, throughput:    185.93 token/s ⏎ Decode.  latency: 0.00535 s, throughput:    186.90 token/s ⏎ Decode.  median latency: 0.00529 s, median throughput:    188.94 token/s ⏎ ``` ⏎  ⏎ PR ⏎  ⏎ ``` ⏎ Benchmark ... ⏎ Prefill. latency: 0.05106 s, throughput:   2506.72 token/s ⏎ Decode.  latency: 0.00543 s, throughput:    184.17 token/s ⏎ Decode.  latency: 0.00534 s, throughput:    187.36 token/s ⏎ Decode.  latency: 0.00531 s, throughput:    188.17 token/s ⏎ Decode.  latency: 0.00530 s, throughput:    188.84 token/s ⏎ Decode.  latency: 0.00529 s, throughput:    189.15 token/s ⏎ Decode.  median latency: 0.00522 s, median throughput:    191.53 token/s ⏎ ``` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-99456bcacb  (L2, 2025-04-20, sha 99456bcacb81, PR #5432)
TITLE: [perf] introduce deep gemm group_gemm_masked as bmm (#5432)
SOURCES: symbol_pickaxe
STAGE1: optimize; artifacts=L2.model.deepseek_v2_mla,L2.optimization.weight_absorption; Replaces FP8 bmm in MLA path with DeepGEMM grouped_gemm_masked for performance.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/layers/quantization/fp8_kernel.py (+108/-4); python/sglang/srt/models/deepseek_v2.py (+86/-16); python/sglang/test/test_block_fp8.py (+167/-0)
LABELS: high priority
PERF_LINES: NOTE: Multi-batch throughput is hardly comparable because expert loads differ significantly with difference quant method. | Optimize out bmm fp8, which takes 7.9 us. The quant kenel with 2 stages takes 1.65+1.14 us and the extra zeros kernel takes 893 ns. | Deepgemm bmm kernel takes 5.7 us And the quant kernel takes 2.7 us. | Bmm fp8 takes 6.4 us. The quant kenel with 2 stages takes 2.5+1.7 us and
BODY: ## Motivation ⏎  ⏎ per-token-group quant+deep_gemm's grouped_gemm_masked is generally faster than per-tensor quant+bmm_fp8 ⏎  ⏎ NOTE: Multi-batch throughput is hardly comparable because expert loads differ significantly with difference quant method. ⏎  ⏎ TODO in futher PR: Optimize `_per_token_group_quant_mla_deep_gemm_masked_fp8` with CudaC ⏎  ⏎ Profile files: https://drive.google.com/drive/folders/1guY2rDdd6LZ6qtLb0L3SQjYwj0pIHJjy?usp=sharing ⏎  ⏎ ### Profile on 8xH200 DeepSeekV3 1-batch ⏎  ⏎ Optimize out bmm fp8, which takes 7.9 us. The quant kenel with 2 stages takes 1.65+1.14 us and the extra zeros kernel takes 893 ns. ⏎  ⏎ ![image](https://github.com/user-attachments/assets/02f714e1-61be-4c56-a796-95a18e1ae291) ⏎  ⏎ Deepgemm bmm kernel takes 5.7 us And the quant kernel takes 2.7 us. ⏎  ⏎ ![image](https://github.com/user-attachments/assets/50732d40-8bcb-4224-911b-2f542fb19976) ⏎  ⏎ ### Profile on 8xH200 DeepSeekV3 50-batch ⏎  ⏎ Bmm fp8 takes 6.4 us. The quant kenel with 2 stages takes 2.5+1.7 us and the extra zeros kernel takes 992 ns. ⏎  ⏎ ![image](https://github.com/user-attachments/assets/50afb31a-e3ce-4f4a-81eb-94468bd944e8) ⏎  ⏎ Deepgemm bmm kernel takes 5.7 us And the quant kernel takes 2.7 us. ⏎  ⏎ ![image](https://github.com/user-attachments/assets/4e3b2ee4-92ac-4444-a952-dba025ec3b34) ⏎  ⏎  ⏎  ⏎ ## Perf on 8xH200 ⏎  ⏎ About 3% improvement on decode phase ⏎  ⏎ ``` ⏎ SGL_ENABLE_JIT_DEEPGEMM=1 SGL_ENABLE_JIT_DEEPGEMM_BMM=0 python3 -m \ ⏎ sglang.bench_one_batch --model-path /DeepSeek-V3 --trust-remote-code --tp 8 \ ⏎ --batch-size 1 --input-len 1000 --output-len 100 ⏎  ⏎ Benchmark ... ⏎ Prefill. latency: 0.17955 s, throughput:   5569.44 token/s ⏎ Decode 0.  latency: 0.01748 s, throughput:     57.20 token/s ⏎ Decode 1.  latency: 0.01744 s, throughput:     57.35 token/s ⏎ Decode 2.  latency: 0.01717 s, throughput:     58.25 token/s ⏎ Decode 3.  latency: 0.01722 s, throughput:     58.09 token/s ⏎ Decode 4.  latency: 0.01742 s, throughput:     57.40 token/s ⏎ Decode.  median latency: 0.01720 s, median throughput:     58.15 token/s ⏎ Total. latency:  1.887 s, throughput:    582.93 token/s ⏎ ``` ⏎  ⏎ - deep gemm bmm ON ⏎ ``` ⏎ SGL_ENABLE_JIT_DEEPGEMM=1 SGL_ENABLE_JIT_DEEPGEMM_BMM=1 python3 -m \ ⏎ sglang.bench_one_batch --model-path /DeepSeek-V3 --trust-remote-code --tp 8 \ ⏎ --batch-size 1 --input-len 1000 --output-len 100 ⏎  ⏎ Benchmark ... ⏎ Prefill. latency: 0.16662 s, throughput:   6001.70 token/s ⏎ Decode 0.  latency: 0.01687 s, throughput:     59.27 token/s ⏎ Decode 1.  latency: 0.01663 s, throughput:     60.14 token/s ⏎ Decode 2.  latency: 0.01662 s, throughput:     60.18 token/s ⏎ Decode 3.  latency: 0.01 …[truncated]

### L2-11b23ae97b  (L2, 2025-04-21, sha 11b23ae97bba, PR #5578)
TITLE: Remove extra copy in deepseek forward absorb (#5578)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: optimize; artifacts=L2.model.deepseek_v2_mla,L2.optimization.weight_absorption; Removes an extra copy in DeepSeek forward_absorb and reports improved throughput.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+9/-13); .github/workflows/pr-test-amd.yml (+7/-7); python/sglang/srt/layers/rotary_embedding.py (+2/-1)
LABELS: high priority
PERF_LINES: latency: 10.73 s | output throughput: 47.70 token/s | (input + output) throughput: 95.40 token/s | latency: 20.00 s | output throughput: 819.37 token/s | (input + output) throughput: 1638.74 token/s | latency: 10.67 s | output throughput: 47.97 token/s | (input + output) throughput: 95.95 token/s | latency: 19.69 s | output throughput: 832.17 token/s | (input + output) throughput: 1664.35 token/s 
BODY: ## Benchmark ⏎  ⏎ ### Performance ⏎  ⏎ main branch: ⏎ ``` ⏎ batch size: 1 ⏎ latency: 10.73 s ⏎ output throughput: 47.70 token/s ⏎ (input + output) throughput: 95.40 token/s ⏎  ⏎ batch size: 32 ⏎ latency: 20.00 s ⏎ output throughput: 819.37 token/s ⏎ (input + output) throughput: 1638.74 token/s ⏎ ``` ⏎  ⏎ this PR: ⏎ ``` ⏎ batch size: 1 ⏎ latency: 10.67 s ⏎ output throughput: 47.97 token/s ⏎ (input + output) throughput: 95.95 token/s ⏎  ⏎ batch size: 32 ⏎ latency: 19.69 s ⏎ output throughput: 832.17 token/s ⏎ (input + output) throughput: 1664.35 token/s ⏎ ``` ⏎  ⏎ 0.5% improvement for bs 1 and 1.5% improvement for bs 32. ⏎  ⏎ main branch: ⏎ <img width="779" alt="image" src="https://github.com/user-attachments/assets/6dc4cc57-0e9c-4d34-afaa-52495f8e5dc5" /> ⏎  ⏎ this PR: ⏎ <img width="740" alt="image" src="https://github.com/user-attachments/assets/1018d709-0446-44c7-8ddf-fc256fb6914f" /> ⏎  ⏎ removed 3 copy with 2 cat added (can overlap one later), duration 40μm -> 38μm ⏎  ⏎ ## Accuracy ⏎ ``` ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3-0324 --tp 8 --trust-remote-code --attention-backend fa3 --disable-radix ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 1400 --parallel 1400 --num-shots 8 ⏎ ``` ⏎ main barnch: ⏎ ``` ⏎ Accuracy: 0.947 ⏎ Invalid: 0.000 ⏎ Latency: 194.902 s ⏎ Output throughput: 737.433 token/s ⏎ ``` ⏎ this PR: ⏎ ``` ⏎ Accuracy: 0.949 ⏎ Invalid: 0.000 ⏎ Latency: 195.182 s ⏎ Output throughput: 729.388 token/s ⏎ ``` ⏎ this PR with deepgemm and deepgemm_bmm: ⏎ ``` ⏎ Accuracy: 0.944 ⏎ Invalid: 0.000 ⏎ Latency: 361.141 s ⏎ Output throughput: 398.598 token/s ⏎ ```

### L2-4418f599a5  (L2, 2025-04-22, sha 4418f599a546, PR #5624)
TITLE: Fix FA3 DeepSeek prefill performance regression (#5624)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_performance; artifacts=L2.model.deepseek_v2_mla,L2.backend.fa3_fa4_mla; Fixes FA3 DeepSeek prefill performance regression by changing forward_absorb code.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+6/-2)
PERF_LINES: Significantly boost 20% at context len 1024, the longer context, the more benifitions | Prefill. latency: 0.32884 s, throughput:  24911.50 token/s | Decode. Batch size: 8, latency: 0.01430 s, throughput:    559.60 token/s | Decode. Batch size: 8, latency: 0.01438 s, throughput:    556.49 token/s | Decode. Batch size: 8, latency: 0.01415 s, throughput:    565.55 token/s | Decode. Batch size: 8, lat
BODY: ## Motivation ⏎  ⏎ Fix FA3 DeepSeek prefill performance regression ⏎  ⏎ Significantly boost 20% at context len 1024, the longer context, the more benifitions ⏎  ⏎ cmd: `python3 -m sglang.bench_one_batch --model lmsys/sglang-ci-dsv3-test --batch-size 8 --input-len 1024 --output-len 128 --trust-remote-code` ⏎  ⏎ - before ⏎ ``` ⏎ Prefill. latency: 0.32884 s, throughput:  24911.50 token/s ⏎ Decode. Batch size: 8, latency: 0.01430 s, throughput:    559.60 token/s ⏎ Decode. Batch size: 8, latency: 0.01438 s, throughput:    556.49 token/s ⏎ Decode. Batch size: 8, latency: 0.01415 s, throughput:    565.55 token/s ⏎ Decode. Batch size: 8, latency: 0.01433 s, throughput:    558.10 token/s ⏎ Decode. Batch size: 8, latency: 0.01465 s, throughput:    546.01 token/s ⏎ Decode.  median latency: 0.01490 s, median throughput:    536.79 token/s ⏎ Total. latency:  2.217 s, throughput:   4156.30 token/s ⏎ ``` ⏎  ⏎ - after ⏎  ⏎ ``` ⏎ Benchmark ... ⏎ Prefill. latency: 0.27612 s, throughput:  29668.45 token/s ⏎ Decode. Batch size: 8, latency: 0.01418 s, throughput:    564.37 token/s ⏎ Decode. Batch size: 8, latency: 0.01412 s, throughput:    566.39 token/s ⏎ Decode. Batch size: 8, latency: 0.01405 s, throughput:    569.45 token/s ⏎ Decode. Batch size: 8, latency: 0.01408 s, throughput:    568.10 token/s ⏎ Decode. Batch size: 8, latency: 0.01437 s, throughput:    556.84 token/s ⏎ Decode.  median latency: 0.01411 s, median throughput:    566.95 token/s ⏎ Total. latency:  2.071 s, throughput:   4449.77 token/s ⏎ ``` ⏎  ⏎ =================================================== ⏎  ⏎ MORE advice: we should consider both context len and prefix len to select MLA or MHA CHUNKED KV. Since they both affect the FLOPS of the core attention ⏎  ⏎ ATTN FLOPS = bh * sq * sk * (dq + dv) * 2 ⏎ MHA:  128b * context * (context + prefix) * (192 + 128) * 2 ⏎ MLA: 128b * context * (context + prefix) * (576 + 512) * 2 ⏎  ⏎ Extract KV FLOPS: b * prefix * kvlora * (h * dqn) * 2 = bh * prefix * kvlora * dqn * 2 = 128b * prefix * 512 * 128 * 2 ⏎  ⏎ Goal: MHA + Extact KV < MLA ⏎ 1. context * (context + prefix) * 320 * 2 + prefix * 65536 * 2 < context * (context + prefix) * 1088 * 2 ⏎ 2. prefix < 0.01171875 * (context + prefix) * context ⏎ 3. set C = 0.01171875 ⏎ 4. prefix < C * (context^2 + prefix*context) ⏎  ⏎ We can impl the filter like this? @Fridge003  ⏎ `sum(prefix) < C * sum(context^2 + prefix*context)` ⏎  ⏎ Maybe have some mistake, welcome to point out ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2ed96c7a8a  (L2, 2025-04-22, sha 2ed96c7a8a20, PR #5272)
TITLE: fix flashmla bug (#5272)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L2.backend.flashmla; Fixes FlashMLA backend bug linked to issue 5154.
ARTIFACT_HINTS: L2.backend.flashmla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashmla_backend.py (+8/-11)
BODY: ## Motivation ⏎  ⏎ fig bug https://github.com/sgl-project/sglang/issues/5154 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-6b6e748775  (L2, 2025-04-22, sha 6b6e7487750b, PR #5638)
TITLE: Remove q concat in FA3 backend for DeepSeek decode (#5638)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L2.backend.fa3_fa4_mla,L2.model.deepseek_v2_mla; Removes q concatenation in FA3 DeepSeek decode because backend accepts q_nope/q_rope separately.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla
FILES: python/sglang/srt/layers/attention/base_attn_backend.py (+3/-0); python/sglang/srt/layers/attention/flashattention_backend.py (+22/-6); python/sglang/srt/layers/radix_attention.py (+8/-1); python/sglang/srt/models/deepseek_v2.py (+7/-2)
PERF_LINES: Latency: 252.757 s | Output throughput: 567.307 token/s | Latency: 246.356 s | Output throughput: 583.927 token/s | latency: 10.38 s | output throughput: 49.30 token/s | (input + output) throughput: 98.61 token/s | latency: 18.97 s | output throughput: 863.68 token/s | (input + output) throughput: 1727.35 token/s | latency: 10.29 s | output throughput: 49.75 token/s | (input + output) throughput: 
BODY: ## Motivation ⏎  ⏎ concat for q in FA3 backend can be removed since the interface for q_nope and q_rope are separate. ⏎  ⏎ ### Accuracy ⏎ ``` ⏎ python3 -m sglang.launch_server --model /dev/shm/DeepSeek-V3 --tp 8 --trust-remote-code --attention-backend fa3 --disable-radix ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 1400 --parallel 160 --num-shots 8 ⏎ ``` ⏎ main: ⏎ ``` ⏎ Accuracy: 0.944 ⏎ Invalid: 0.000 ⏎ Latency: 252.757 s ⏎ Output throughput: 567.307 token/s ⏎ ``` ⏎ this PR: ⏎ ``` ⏎ Accuracy: 0.947 ⏎ Invalid: 0.000 ⏎ Latency: 246.356 s ⏎ Output throughput: 583.927 token/s ⏎ ``` ⏎ ### Performance ⏎ ``` ⏎ python3 -m sglang.launch_server --model /dev/shm/DeepSeek-V3 --tp 8 --trust-remote-code --attention-backend fa3 --disable-radix ⏎ python3 -m sglang.bench_one_batch_server --model None --base-url http://0.0.0.0:30000 --batch-size 1 --input-len 512 --output-len 512 ⏎ ``` ⏎ main: ⏎ ``` ⏎ batch size: 1 ⏎ latency: 10.38 s ⏎ output throughput: 49.30 token/s ⏎ (input + output) throughput: 98.61 token/s ⏎  ⏎ batch size: 32 ⏎ latency: 18.97 s ⏎ output throughput: 863.68 token/s ⏎ (input + output) throughput: 1727.35 token/s ⏎ ``` ⏎ this PR: ⏎ ``` ⏎ batch size: 1 ⏎ latency: 10.29 s ⏎ output throughput: 49.75 token/s ⏎ (input + output) throughput: 99.51 token/s ⏎  ⏎ batch size: 32 ⏎ latency: 18.15 s ⏎ output throughput: 902.83 token/s ⏎ (input + output) throughput: 1805.66 token/s ⏎ ``` ⏎ 1% improvement for bs 1 and 4% for bs 32. ⏎ ### Profile ⏎ main: ⏎ <img width="564" alt="image" src="https://github.com/user-attachments/assets/0ec0cc52-9c31-4baf-a739-c670aacd33d1" /> ⏎  ⏎ this PR: ⏎ <img width="575" alt="image" src="https://github.com/user-attachments/assets/b311f5ca-2ccf-47e3-9deb-14579a2dcea4" /> ⏎  ⏎ 2.5μm reduced for bs 1.

### L2-93c6fb12c7  (L2, 2025-04-25, sha 93c6fb12c773, PR #5723)
TITLE: Fix: deepseek forward absorb (#5723)
SOURCES: subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L2.model.deepseek_v2_mla,L2.optimization.weight_absorption; Restores contiguous handling needed after forward_absorb copy removal broke test_mla.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/layernorm.py (+41/-4)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ #5578 will cause test_mla.py fail in CI after #5646 (revert #5510) layernorm.py to the orginal.   ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ The problematic modifications are likely comes from .contiguous() call in deepseek_v2.py ⏎ so add contiguous check used before #5578. ⏎  ⏎ ## Checklist

### L2-799c4bb502  (L2, 2025-04-26, sha 799c4bb50218, PR #5748)
TITLE: Fuse MLA set kv cache kernel (#5748)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: optimize; artifacts=L2.pool.mla_token_kv,L2.kernel.set_mla_kv_buffer,L2.backend.fa3_fa4_mla; Fuses MLA set-KV-cache work and removes k concatenation for FA3 backend.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla, L2.pool.mla_token_kv
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+6/-4); python/sglang/srt/layers/radix_attention.py (+5/-2); python/sglang/srt/mem_cache/memory_pool.py (+87/-0); python/sglang/srt/models/deepseek_v2.py (+2/-3)
LABELS: high priority, ready-to-merge
PERF_LINES: Latency: 133.934 s | Output throughput: 1070.970 token/s | latency: 8.21 s | output throughput: 62.39 token/s | (input + output) throughput: 124.79 token/s | latency: 16.52 s | output throughput: 991.67 token/s | (input + output) throughput: 1983.34 token/s | latency: 26.52 s | output throughput: 2471.57 token/s | latency: 8.10 s | output throughput: 63.23 token/s | (input + output) throughput: 12
BODY: ## Motivation ⏎  ⏎ Fuse MLA set kv cache kernel and remove k concat operation. Currently only support FA3 backend. Can be applied to other backend with subsequent verification. ⏎  ⏎ ## Benchmark ⏎  ⏎ ### Accuracy ⏎ ``` ⏎ Accuracy: 0.946 ⏎ Invalid: 0.000 ⏎ Latency: 133.934 s ⏎ Output throughput: 1070.970 token/s ⏎ ``` ⏎  ⏎ ### Performance ⏎ main branch: ⏎ ``` ⏎ batch size: 1 ⏎ latency: 8.21 s ⏎ output throughput: 62.39 token/s ⏎ (input + output) throughput: 124.79 token/s ⏎  ⏎ batch size: 32 ⏎ latency: 16.52 s ⏎ output throughput: 991.67 token/s ⏎ (input + output) throughput: 1983.34 token/s ⏎  ⏎ batch size: 128 ⏎ latency: 26.52 s ⏎ output throughput: 2471.57 token/s ⏎ ``` ⏎ this PR: ⏎ ``` ⏎ batch size: 1 ⏎ latency: 8.10 s ⏎ output throughput: 63.23 token/s ⏎ (input + output) throughput: 126.47 token/s ⏎  ⏎ batch size: 32 ⏎ latency: 16.25 s ⏎ output throughput: 1008.51 token/s ⏎ (input + output) throughput: 2017.01 token/s ⏎  ⏎ batch size: 128 ⏎ latency: 25.38 s ⏎ output throughput: 2581.93 token/s ⏎ (input + output) throughput: 5163.85 token/s ⏎ ``` ⏎  ⏎ ### Profile ⏎ #### bs 1 ⏎ main branch: ⏎ <img width="562" alt="image" src="https://github.com/user-attachments/assets/d4a9f365-0974-4baf-9a5b-d706fc2d1b88" /> ⏎  ⏎ this PR: ⏎ <img width="445" alt="image" src="https://github.com/user-attachments/assets/c91bda91-6e8e-478d-b395-c39e44a383c5" /> ⏎ 5.2μm -> 1.2μm ⏎ #### bs 16 ⏎ main branch: ⏎ <img width="452" alt="image" src="https://github.com/user-attachments/assets/f214636c-5e8e-4895-a13c-f310cf96da18" /> ⏎  ⏎ this PR: ⏎ <img width="401" alt="image" src="https://github.com/user-attachments/assets/402e4d38-519d-49ee-b225-f744fa90adda" /> ⏎ 6.5μm -> 1.6μm

### L2-84810da4ae  (L2, 2025-04-27, sha 84810da4ae42, PR #5390)
TITLE: Add Cutlass MLA attention backend (#5390)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: integrate; artifacts=L2.backend.cutlass_mla,L2.kernel.cutlass_mla,L2.dispatch.server_args_defaults; Adds cutlass_mla attention backend option wiring the Blackwell CUTLASS MLA kernel into DeepSeek.
ARTIFACT_HINTS: L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+278/-0); python/sglang/srt/layers/attention/utils.py (+1/-1); python/sglang/srt/model_executor/model_runner.py (+7/-0); python/sglang/srt/server_args.py (+14/-1); docs/backend/server_arguments.md (+1/-1); python/sglang/srt/managers/schedule_batch.py (+1/-0); sgl-kernel/python/sgl_kernel/attention.py (+3/-0)
LABELS: high priority, ready-to-merge
PERF_LINES: Request throughput (req/s):              5.22 | Input token throughput (tok/s):          5223.67 | Output token throughput (tok/s):         5223.67 | Total token throughput (tok/s):          10447.34 | ----------------End-to-End Latency---------------- | Mean E2E Latency (ms):                   572768.28 | Median E2E Latency (ms):                 572918.73 | Mean TTFT (ms):                        
BODY: ## Motivation ⏎  ⏎ Enables use of the [blackwell cutlass MLA decode kernel](https://github.com/sgl-project/sglang/pull/5142) with deepseek models. ⏎  ⏎ ## Modifications ⏎  ⏎ Adds "cutlass_mla" option for attention backend. ⏎  ⏎ ## Deepseek-R1 Benchmarks ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --host 0.0.0.0 --port 30000 --tp 8 --model-path deepseek-ai/DeepSeek-R1 --trust-remote-code --enable-dp-attention --attention-backend cutlass_mla --dtype float16 --dp 8 --page-size 128 ⏎ python3 -m sglang.bench_serving --backend sglang --model deepseek-ai/DeepSeek-R1 --num-prompts 3000 --dataset-name random --random-input-len 1000 --random-output-len 1000 --random-range-ratio 1 ⏎ ``` ⏎  ⏎ Using `--attention-backend cutlass_mla` ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang ⏎ Traffic request rate:                    inf ⏎ Max reqeuest concurrency:                not set ⏎ Successful requests:                     3000 ⏎ Benchmark duration (s):                  574.31 ⏎ Total input tokens:                      3000000 ⏎ Total generated tokens:                  3000000 ⏎ Total generated tokens (retokenized):    2990164 ⏎ Request throughput (req/s):              5.22 ⏎ Input token throughput (tok/s):          5223.67 ⏎ Output token throughput (tok/s):         5223.67 ⏎ Total token throughput (tok/s):          10447.34 ⏎ Concurrency:                             2991.95 ⏎ ----------------End-to-End Latency---------------- ⏎ Mean E2E Latency (ms):                   572768.28 ⏎ Median E2E Latency (ms):                 572918.73 ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          134177.24 ⏎ Median TTFT (ms):                        133475.88 ⏎ P99 TTFT (ms):                           261181.99 ⏎ ---------------Inter-Token Latency---------------- ⏎ Mean ITL (ms):                           439.04 ⏎ Median ITL (ms):                         309.17 ⏎ P95 ITL (ms):                            392.38 ⏎ P99 ITL (ms):                            607.01 ⏎ Max ITL (ms):                            249723.06 ⏎ ================================================== ⏎ ``` ⏎  ⏎ Baseline `--attention-backend triton` ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang ⏎ Traffic request rate:                    inf ⏎ Max reqeuest concurrency:                not set ⏎ Successful requests:                     3000 ⏎ Benchmark duration (s):                  729.28 ⏎ Total input tokens:                      3000000 ⏎ Total generated tokens:                  3000000 ⏎ Total generated tokens (retok …[truncated]

### L2-8d463fe351  (L2, 2025-04-28, sha 8d463fe351c4, PR #5868)
TITLE: Cutlass MLA decode - fix dtype error (#5868)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L2.backend.cutlass_mla; Fixes CUTLASS MLA backend dtype error found after integration.
ARTIFACT_HINTS: L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ After #5390 was merged, I ran into the following error. I bisected it and found #5578 to be the cause. ⏎  ⏎ ``` ⏎   File "/trevor/sglang/python/sglang/srt/models/deepseek_v2.py", line 632, in forward ⏎     return self.forward_absorb( ⏎            ^^^^^^^^^^^^^^^^^^^^ ⏎   File "/trevor/sglang/python/sglang/srt/models/deepseek_v2.py", line 741, in forward_absorb ⏎     attn_output = self.attn_mqa(q, k, k_nope, forward_batch) ⏎                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl ⏎     return self._call_impl(*args, **kwargs) ⏎            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1762, in _call_impl ⏎     return forward_call(*args, **kwargs) ⏎            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/trevor/sglang/python/sglang/srt/layers/radix_attention.py", line 97, in forward ⏎     return forward_batch.attn_backend.forward( ⏎            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/trevor/sglang/python/sglang/srt/layers/attention/base_attn_backend.py", line 68, in forward ⏎     return self.forward_decode( ⏎            ^^^^^^^^^^^^^^^^^^^^ ⏎   File "/trevor/sglang/python/sglang/srt/layers/attention/cutlass_mla_backend.py", line 270, in forward_decode ⏎     o = cutlass_mla_decode( ⏎         ^^^^^^^^^^^^^^^^^^^ ⏎   File "/usr/local/lib/python3.12/dist-packages/sgl_kernel/attention.py", line 85, in cutlass_mla_decode ⏎     assert q_nope_and_q_pe.dtype in ( ⏎            ^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ AssertionError: q_nope_and_q_pe.dtype needs to be fp16 or bf16 but got torch.float32. ⏎ ``` ⏎  ⏎ ## Modifications ⏎  ⏎ Fix by casting q to the correct dtype. ⏎  ⏎ ## Checklist

### L2-8e5a6d3441  (L2, 2025-04-29, sha 8e5a6d3441d5, PR #5875)
TITLE: [Fix] Fix a bug for flashmla to run R1 model (#5875)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L2.backend.flashmla; Guards FlashMLA invalid zero-length parameter that caused illegal memory access on R1.
ARTIFACT_HINTS: L2.backend.flashmla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashmla_backend.py (+3/-0)
BODY: ## Motivation ⏎ When using sglang to run R1 Model by TP = 16, there is a bug for flashmla. ⏎ I investigated this issue and found that it causes illegal access to the HMB when the seq_length is zero, leading to a program crash. ⏎  ⏎ So, we should not pass a invalid parameter to flashmla kernel. I chose the number 1024 because it is 2 to the power of 10. Other numbers would also work. ⏎  ⏎  ⏎  ⏎ Related PR: https://github.com/sgl-project/sglang/pull/5272 ⏎ Thanks for @sleepcoo ⏎  ⏎ ``` ⏎ root:/sgl-workspace/pengcuo_work/op_test/FlashMLA# python3 tests/test_flash_mla.py   ⏎ b=16, s_q=1, mean_sk=0, h_q=16, h_kv=1, d=576, dv=512, causal=True, varlen=False ⏎ q : torch.Size([16, 1, 16, 576]) torch.bfloat16 ⏎ blocked_k : torch.Size([0, 64, 1, 576]) torch.bfloat16 ⏎ block_table : torch.Size([16, 0]) torch.int32 ⏎ cache_seqlens : torch.Size([16]) torch.int32 ⏎ cache_seqlens : tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0', ⏎        dtype=torch.int32) ⏎ num_splits : tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16], ⏎        device='cuda:0', dtype=torch.int32) ⏎ Traceback (most recent call last): ⏎   File "/mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_pengcuoze/work/op_test/FlashMLA/tests/test_flash_mla.py", line 168, in <module> ⏎     main(torch_dtype) ⏎   File "/mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_pengcuoze/work/op_test/FlashMLA/tests/test_flash_mla.py", line 149, in main ⏎     test_flash_mla(b, s_q, s, h_q, h_kv, d, dv, causal, varlen) ⏎   File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context ⏎     return func(*args, **kwargs) ⏎   File "/mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_pengcuoze/work/op_test/FlashMLA/tests/test_flash_mla.py", line 110, in test_flash_mla ⏎     out_torch, lse_torch = ref_mla() ⏎   File "/mnt/gemininjceph2/geminicephfs/mm-base-plt2/user_pengcuoze/work/op_test/FlashMLA/tests/test_flash_mla.py", line 96, in ref_mla ⏎     end = begin + cache_seqlens[i] ⏎   File "/usr/local/lib/python3.10/dist-packages/torch/utils/_device.py", line 104, in __torch_function__ ⏎     return func(*args, **kwargs) ⏎ RuntimeError: CUDA error: an illegal memory access was encountered ⏎ CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect. ⏎ For debugging consider passing CUDA_LAUNCH_BLOCKING=1 ⏎ Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions. ⏎ ``` ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-799789afed  (L2, 2025-04-29, sha 799789afedcf, PR #5870)
TITLE: Bump Flashinfer to 0.2.5 (#5870)
SOURCES: path_core, dependency_pin
STAGE1: adapt_framework; artifacts=L2.backend.flashinfer_mla,L2.upstream.flashinfer_mla; Bumps FlashInfer and updates FlashInfer MLA backend code for the new API.
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+27/-16); .github/workflows/pr-test.yml (+0/-2); docs/start/install.md (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+107/-82)
BODY: ## Motivation ⏎ This pull requests is just little modification on the basis of #5538, which fixes conflicts and lints. ⏎ Thanks @AkazaAkane for contribution! ⏎  ⏎ Ref: #5855, #4905, #5023 ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-22da3d978f  (L2, 2025-05-05, sha 22da3d978f8a, PR #5555)
TITLE: Fix "Avoid computing lse in Ragged Prefill when there's no prefix match" (#5555)
SOURCES: path_core
STAGE1: reland; artifacts=L2.backend.flashinfer_mla; Relands fixed ragged-prefill LSE skip for FlashInfer MLA after reverted attempt.
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-1); docs/backend/server_arguments.md (+2/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+20/-12)
LABELS: ready-to-merge
BODY: ## Motivation ⏎ Reopens #5476 as per #5544. ⏎ Locally passed `python3 test/srt/models/test_reward_models.py`, but we can wait for all CIs before merging.  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f6f96b0521  (L2, 2025-05-08, sha f6f96b0521ca, PR #6123)
TITLE: [sgl-kernel] fix: fix cu118 compile error (#6123)
SOURCES: path_core, symbol_pickaxe
STAGE1: repair_build_dependency; artifacts=L2.kernel.cutlass_mla; Fixes cu118 compile/symbol error in CUTLASS MLA kernel source.
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+16/-1); sgl-kernel/csrc/grammar/apply_token_bitmask_inplace_cuda.cu (+6/-2)
ISSUES: #5100 [Bug] common_ops.abi3.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb
BODY: ## Motivation ⏎  ⏎  ⏎ See issue : https://github.com/sgl-project/sglang/issues/5100 ⏎ https://github.com/sgl-project/sglang/pull/5686. After this PR, it will cause some symbol error when compile at cu 118 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-0ab3f437ab  (L2, 2025-05-08, sha 0ab3f437aba7, PR #6101)
TITLE: Cutlass MLA: Disable split kv due to https://github.com/NVIDIA/cutlass/issues/2274 (#6101)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L2.kernel.cutlass_mla; Disables split-KV in CUTLASS MLA because upstream CUTLASS bug caused incorrect outputs.
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+4/-1); sgl-kernel/tests/test_cutlass_mla.py (+1/-1)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎ A bug https://github.com/NVIDIA/cutlass/issues/2274 was found in the cutlass MLA kernel which can cause incorrect outputs. The test cases didn't catch this because the random input values were too low. ⏎  ⏎ ## Modifications ⏎  ⏎ Disable split kv for now until the issue is fixed. Increase random input values so that bug can be detected properly - new values will cause test to fail when using split kv. Thanks @kaixih  ⏎  ⏎ ## Checklist

### L2-e30c273bc9  (L2, 2025-05-08, sha e30c273bc9bb, PR #5822)
TITLE: opt flashinfer mla cat (#5822)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L2.backend.flashinfer_mla,L2.model.deepseek_v2_mla; Optimizes FlashInfer MLA by removing q and k concatenation following FA3 changes.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+59/-13); python/sglang/srt/models/deepseek_v2.py (+1/-1)
PERF_LINES: Latency: 228.672 s | Output throughput: 554.173 token/s | {"run_name": "default", "batch_size": 1, "input_len": 1024, "output_len": 1024, "latency": 14.4687, "output_throughput": 70.77, "overall_throughput": 141.55} | {"run_name": "default", "batch_size": 16, "input_len": 1024, "output_len": 1024, "latency": 28.8723, "output_throughput": 567.47, "overall_throughput": 1134.93} | {"run_name": "defau
BODY: ## Motivation ⏎  ⏎  ⏎ Base on  #5748 and #5638 , for flashinfer mla, remove q and k cat. ⏎  ⏎ ### Accuracy ⏎ ``` ⏎ Accuracy: 0.951 ⏎ Invalid: 0.000 ⏎ Latency: 228.672 s ⏎ Output throughput: 554.173 token/s ⏎ ``` ⏎  ⏎ ### Performance ⏎ main branch: ⏎ ``` ⏎ {"run_name": "default", "batch_size": 1, "input_len": 1024, "output_len": 1024, "latency": 14.4687, "output_throughput": 70.77, "overall_throughput": 141.55} ⏎  ⏎ {"run_name": "default", "batch_size": 16, "input_len": 1024, "output_len": 1024, "latency": 28.8723, "output_throughput": 567.47, "overall_throughput": 1134.93} ⏎  ⏎ {"run_name": "default", "batch_size": 32, "input_len": 1024, "output_len": 1024, "latency": 38.0349, "output_throughput": 861.52, "overall_throughput": 1723.05} ⏎ ``` ⏎  ⏎ this PR: ⏎ ``` ⏎ {"run_name": "default", "batch_size": 1, "input_len": 1024, "output_len": 1024, "latency": 14.5066, "output_throughput": 70.59, "overall_throughput": 141.18} ⏎  ⏎ {"run_name": "default", "batch_size": 16, "input_len": 1024, "output_len": 1024, "latency": 28.4372, "output_throughput": 576.15, "overall_throughput": 1152.29} ⏎  ⏎ {"run_name": "default", "batch_size": 32, "input_len": 1024, "output_len": 1024, "latency": 37.2972, "output_throughput": 878.57, "overall_throughput": 1757.13} ⏎ ``` ⏎  ⏎ ### Profile ⏎  ⏎ #### Prefill ⏎ main branch: ⏎ ![image](https://github.com/user-attachments/assets/b5e21f35-b5da-4c03-8961-a24afdf36402) ⏎  ⏎ this PR: ⏎ ![image](https://github.com/user-attachments/assets/c916a15b-0236-44a0-8330-1ce8575d0e34) ⏎  ⏎ 47 us to 3us ⏎  ⏎ #### Decode  ⏎ main branch bs=1 cuda graph+torch compile: ⏎ ![image](https://github.com/user-attachments/assets/cb82a756-e01b-4324-b8dd-48d2ba09de52) ⏎  ⏎ this PR bs=1 cuda graph+torch compile: ⏎ ![image](https://github.com/user-attachments/assets/c80eb6e2-03a1-4294-8429-7297b44739bf) ⏎ bs=1, cuda graph+torch compile, almost the same ⏎  ⏎ main branch bs=1 cuda graph+ without torch compile fused with other ops: ⏎ ![image](https://github.com/user-attachments/assets/6999123c-faad-49a4-a8e5-5302ae7e156c) ⏎  ⏎ this PR bs=1 cuda graph+ without torch compile fused with other ops: ⏎ ![image](https://github.com/user-attachments/assets/9b27b16c-058f-4f76-8c3e-e7bba0645067) ⏎ bs=1, cuda graph, without torch compile, 6~7 us -> 1us ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ - Update deepseek_v2 code, remove q and k cat. ⏎ - In flashinfer_mla_backend:  ⏎ > Cat when ragged, no cat in other scenes.  ⏎ > Use set_mla_kv_buffer when k_rope is not empty ⏎  ⏎ ## Checklist

### L2-b29a026e14  (L2, 2025-05-09, sha b29a026e14b9, PR #6016)
TITLE: KV‑Cache (MHA, MLA): add missing start_layer / end_layer fields to MHATokenToKVPoolHost and MLATokenToKVPoolHost (#6016)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L2.pool.mla_token_kv; Adds missing start_layer/end_layer fields to MLATokenToKVPoolHost for hierarchical cache.
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/memory_pool.py (+2/-0)
ISSUES: #6005 [Bug] AttributeError: 'MHATokenToKVPoolHost' object has no attribute 'start_layer' when using --enable-hierarchical-cache
BODY: ## Motivation ⏎  ⏎ Fix #6005  ⏎  ⏎ ## Modifications ⏎  ⏎ both MHATokenToKVPoolHost.__init__ and MLATokenToKVPoolHost.__init__ ⏎ ``` ⏎ self.start_layer = start_layer or device_pool.start_layer ⏎ self.end_layer   = end_layer   or device_pool.end_layer ⏎ ``` ⏎ makes the host shard self‑contained before any loader thread starts. ⏎ No other files are touched; public APIs and CLI flags remain unchanged. ⏎  ⏎ Notes for Reviewers ⏎ This follows the same pattern used by MHATokenToKVPoolHost, keeping the ⏎ two host‑pool implementations symmetrical and avoiding the need to touch ⏎ every layer_id - self.start_layer call site. ⏎  ⏎ ## Checklist

### L2-2e4babdb0a  (L2, 2025-05-15, sha 2e4babdb0a19, PR #6109)
TITLE: [Feat] Support FlashMLA backend with MTP and FP8 KV cache (#6109)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: extend_support; artifacts=L2.backend.flashmla,L2.backend.flashinfer_mla; Extends FlashMLA backend for MTP decode and FP8 KV cache support.
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.flashmla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+8/-4); python/sglang/srt/layers/attention/flashmla_backend.py (+340/-78); python/sglang/srt/layers/attention/utils.py (+2/-2); python/sglang/srt/model_executor/cuda_graph_runner.py (+5/-1); python/sglang/srt/speculative/eagle_worker.py (+13/-0); docs/backend/attention_backend.md (+7/-1); docs/references/deepseek.md (+1/-1); test/srt/test_flashmla.py (+68/-0)
PERF_LINES: The speedup of MTP + FP8 KV cache is about 30% with KV cache usage reducing by 50%:
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR improves flashmla backend by accelerating decode stage with mtp. The implementation utilizes the feature that flashmla can handle `seq_len_q > 1`. To use flashmla with mtp, an example server args can be `--attention-backend flashmla --speculative-algorithm NEXTN --speculative-num-steps  1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`. ⏎  ⏎ This PR also supports flashmla backend with fp8 kv-cache, which halves kv cache usage and enables larger concurrency with longer input sequence lengths when memory is limited. To enable fp8 kv-cache, an additional server arg need to be added: `--kv-cache-dtype fp8_e4m3`. ⏎  ⏎ The speedup of MTP + FP8 KV cache is about 30% with KV cache usage reducing by 50%: ⏎  ⏎ <img width="696" alt="Screenshot 2025-05-14 at 20 44 28" src="https://github.com/user-attachments/assets/e5322bce-0be2-45c2-872c-5062ce1ae85a" /> ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - FlashMLA backend supports MTP, compatiable with cuda graph ⏎ - FlashMLA backend supports FP8 KV cache ⏎  ⏎ ## Checklist

### L2-a38376fa99  (L2, 2025-05-24, sha a38376fa9913, PR #6477)
TITLE: Refactor attention into multiple stages (#6477)
SOURCES: symbol_pickaxe
STAGE1: adapt_framework; artifacts=L2.model.deepseek_v2_mla,L2.optimization.weight_absorption; Splits DeepSeek MLA attention into prepare/core stages for framework operation staging.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+117/-24); python/sglang/srt/operations_strategy.py (+4/-2)
BODY: ## Motivation ⏎  ⏎ dep #6476, plz merge that first and subtract diff from that ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-183d9f969c  (L2, 2025-05-27, sha 183d9f969c24, PR #6638)
TITLE: DeepSeek: enable none block-quant FP8 quantizations (#6638)
SOURCES: symbol_pickaxe
STAGE1: extend_support; artifacts=L2.model.deepseek_v2_mla,L2.optimization.weight_absorption; Extends DeepSeek MLA path to non-block FP8 quantization modes.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+55/-43)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ Enable broader quantization - like none-block FP8 quant for DeepSeek based models ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-51cdd81f97  (L2, 2025-05-29, sha 51cdd81f9720, PR #6265)
TITLE: [fix][RL] Fix DeepSeekV3ForCausalLM.post_load_weights for multiple update weight (#6265)
SOURCES: symbol_pickaxe
STAGE1: repair_correctness; artifacts=L2.model.deepseek_v2_mla,L2.optimization.weight_absorption; Fixes repeated post_load_weights corrupting absorbed w_kv weights during RL updates.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+39/-14); python/sglang/srt/utils.py (+8/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ When doing RL, which will call `post_load_weights` a second time apart from the intialization, the origin `post_load_weights` will somehow corrupt the weight when recreate w_kv instead of `copy_`. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-a2cb5913a0  (L2, 2025-06-02, sha a2cb5913a009, PR #6805)
TITLE: Add draft extend CUDA graph for flashinfer backend  (#6805)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L2.backend.flashinfer_mla,L2.runner.cuda_graph_mla; Adds draft-extend CUDA graph support for FlashInfer MLA/backend speculative path.
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+32/-0); python/sglang/srt/layers/attention/flashinfer_backend.py (+40/-0); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+3/-1); python/sglang/srt/speculative/eagle_worker.py (+10/-1); test/srt/test_eagle_infer.py (+85/-1)
LABELS: ready-to-merge
PERF_LINES: |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |    | |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |    | |    |   max_concurrency |   input_throughput |   output_throughput |   me
BODY: ## Motivation ⏎ Follow up of https://github.com/sgl-project/sglang/pull/6606. ⏎  ⏎ ### Flashinfer ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --speculative-algorithm EAGLE --speculative-draft-model-path lmsys/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --trust-remote-code --dtype float16 --attention-backend flashinfer ⏎ ``` ⏎ main branch: ⏎ ``` ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |    |   max_concurrency |   input_throughput |   output_throughput |   mean_ttft_ms |   median_ttft_ms |   p99_ttft_ms |   mean_tpot_ms |   median_tpot_ms |   p99_tpot_ms |   per_user_throughput | ⏎ +====+===================+====================+=====================+================+==================+===============+================+==================+===============+=======================+ ⏎ |  0 |             1.000 |            201.726 |             201.726 |         42.829 |           37.038 |        67.022 |          4.918 |            4.770 |         5.844 |               201.726 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |  1 |             4.000 |            743.703 |             743.703 |         53.703 |           42.985 |       115.351 |          5.070 |            5.137 |         6.097 |               185.926 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |  2 |            16.000 |           2116.391 |            2116.391 |        110.257 |           45.905 |       459.640 |          6.511 |            6.472 |         8.899 |               132.274 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+-----------------------+ ⏎ |  3 |            32.000 |           3485.321 |            3485.321 |        154.138 |           48.667 |       854.407 |          8.471 |            8.312 |        11.995 |               108.916 | ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------- …[truncated]

### L2-b819381fec  (L2, 2025-06-05, sha b819381feca4, PR #6838)
TITLE: AITER backend extension and workload optimizations (#6838)
SOURCES: symbol_pickaxe
STAGE1: optimize; artifacts=L2.backend.aiter_mla,L2.model.deepseek_v2_mla; Extends AITER backend with DeepSeek MLA workload optimizations and model-runner wiring.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.aiter_mla
FILES: .github/workflows/pr-test-amd.yml (+1/-1); docs/references/environment_variables.md (+1/-1); python/sglang/srt/layers/attention/aiter_backend.py (+488/-124); python/sglang/srt/layers/layernorm.py (+27/-2); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-3); python/sglang/srt/layers/quantization/fp8.py (+16/-16); python/sglang/srt/layers/quantization/fp8_utils.py (+6/-10); python/sglang/srt/model_executor/model_runner.py (+10/-0); python/sglang/srt/models/deepseek_v2.py (+17/-0); scripts/amd_ci_exec.sh (+13/-7); test/srt/test_nightly_gsm8k_eval_amd.py (+1/-1)
LABELS: high priority, aiter
BODY: `Co-author: @kkHuang-amd ` ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎ - DeepSeek optimization ⏎ - 1 simple flag: `SGLANG_USE_AITER` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-18efb5e8e0  (L2, 2025-06-08, sha 18efb5e8e0ed, PR #6929)
TITLE: [perf][sgl-kernel] extend cutlass_mla_decode to support num_head < 128 (#6929)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L2.kernel.cutlass_mla; Rewrites/extends CUTLASS MLA decode to support num_head under 128 and improve page-size-128 performance.
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+52/-22); sgl-kernel/csrc/attention/cutlass_sm100_mla/device/sm100_mla.hpp (+358/-0); sgl-kernel/csrc/attention/cutlass_sm100_mla/kernel/sm100_fmha_mla_reduction.hpp (+198/-0); sgl-kernel/csrc/attention/cutlass_sm100_mla/kernel/sm100_fmha_mla_tma_warpspecialized.hpp (+2018/-0); sgl-kernel/csrc/attention/cutlass_sm100_mla/kernel/sm100_mla_tile_scheduler.hpp (+160/-0); sgl-kernel/benchmark/bench_cutlass_mla.py (+133/-0); sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/include/sgl_kernel_ops.h (+4/-2); sgl-kernel/python/sgl_kernel/attention.py (+18/-8); sgl-kernel/tests/test_cutlass_mla.py (+17/-4)
PERF_LINES: 3. improve performance up to 1.8x for page_size == 128 | Tables below shows GB/s of  cases
BODY: ## Motivation ⏎  ⏎ An inelegant workaround, thus the kernel have to be rewrited to support num_head != 128 ⏎  ⏎ 1. add benchmark for cutlass_mla_decode ⏎ 2. extend num_head support range ⏎ 3. improve performance up to 1.8x for page_size == 128 ⏎ 4. fix split_kv thanks to https://github.com/flashinfer-ai/flashinfer/pull/1109, but still has some issue, ref: https://github.com/sgl-project/sglang/pull/6929/files#diff-3aef9e714fbacce71c7a8d812964d83355a6fa2953340ebc3f2087e9f0920d6eR215 ⏎  ⏎ Limitation: ⏎ 1. Auto dynamic split_kv is not compatitable with cuda graph ⏎ 2. While static split_kv will hang with spec. batch_size and seq_len combination ⏎  ⏎ TODO futher: ⏎ Try per batch spilt kv schedule ⏎  ⏎ Tables below shows GB/s of  cases ⏎  ⏎ - This PR ⏎ ``` ⏎ block_size=1, num_kv_splits=-1:  ⏎ cutlass mla: ⏎     batch_size  seq_len    128 heads     64 heads     32 heads     16 heads ⏎ 0          1.0      1.0    18.775741     9.323621     7.066667     5.914849 ⏎ 1          1.0     64.0    18.037941     8.492891     6.369953     5.546251 ⏎ 2          1.0    128.0    18.086956     9.333334     7.302476     5.933333 ⏎ 3          1.0    256.0    19.248121    11.767562     9.880312     8.944444 ⏎ 4          1.0    512.0    30.251950    20.342857    18.383585    17.413024 ⏎ 5          1.0   1024.0    47.565761    35.746746    33.859496    32.944443 ⏎ 6          1.0   2048.0    88.256958    65.945943    65.831741    63.242601 ⏎ 7          1.0   4096.0   152.649073   124.944850   123.052637   125.459454 ⏎ 8          1.0   8192.0   262.871008   227.231584   225.579344   225.437931 ⏎ 9          8.0      1.0   144.695649    56.619274    44.741961    37.504525 ⏎ 10         8.0     64.0   144.892515    58.609976    44.485247    36.867314 ⏎ 11         8.0    128.0   144.695649    57.853107    45.039005    37.815768 ⏎ 12         8.0    256.0   149.645091    78.089209    65.518332    59.431869 ⏎ 13         8.0    512.0   226.133339   130.754661   119.121392   112.588323 ⏎ 14         8.0   1024.0   377.765798   232.612351   221.660749   215.636362 ⏎ 15         8.0   2048.0   432.146795   325.333342   316.513943   311.895788 ⏎ 16         8.0   4096.0   709.012474   542.171414   534.399986   530.751216 ⏎ 17         8.0   8192.0  1036.674388   852.300472   852.170655   848.441399 ⏎ 18        32.0      1.0   577.214113   231.599354   176.064888   147.588667 ⏎ 19        32.0     64.0   577.214113   234.631753   178.086961   152.210434 ⏎ 20        32.0    128.0   577.997268   238.139541   175.779761   147.948058 ⏎ 21        32.0    256.0   510.632211   282.115667   237.178928   214.806508 ⏎ 22        32.0    512.0   545.104826   …[truncated]

### L2-e58423b2b9  (L2, 2025-06-09, sha e58423b2b9f2, PR #6998)
TITLE: Fix cutlass MLA gets almost zero accuracy (#6998)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases
STAGE1: repair_correctness; artifacts=L2.backend.cutlass_mla,L2.runner.cuda_graph_mla; Fixes near-zero accuracy in CUTLASS MLA CUDA-graph backend integration.
ARTIFACT_HINTS: L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+2/-19)
DEEP_STUDY: deep-study correctness case sglang:e58423b2b9: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: Fix issues in cuda graph and now it is roughly same as non-cuda-graph or slightly lower. But seems non-cuda-graph still has bug, thus may need to check separately. ⏎  ⏎ test ⏎  ⏎ ``` ⏎ while true; do python3 benchmark/gsm8k/bench_sglang.py --parallel 10000 --num-questions 1400; done ⏎ ``` ⏎  ⏎ master + no cudagraph ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model /dev/shm/DeepSeek-V3-0324 --tp 8 --enable-dp-attention --dp 8 --trust-remote-code --attention-backend cutlass_mla --context-length 4096 --cuda-graph-bs 64 128 --max-running-requests 128 --disable-cuda-graph ⏎  ⏎ Accuracy: 0.917 ⏎ Accuracy: 0.917 ⏎ Accuracy: 0.910 ⏎ Accuracy: 0.920 ⏎ Accuracy: 0.916 ⏎ ``` ⏎  ⏎ master + cuda graph ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model /dev/shm/DeepSeek-V3-0324 --tp 8 --enable-dp-attention --dp 8 --trust-remote-code --attention-backend cutlass_mla --context-length 4096 --cuda-graph-bs 64 128 --max-running-requests 128 ⏎  ⏎ Accuracy: 0.060 ⏎ ``` ⏎  ⏎ pr + cuda graph ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model /dev/shm/DeepSeek-V3-0324 --tp 8 --enable-dp-attention --dp 8 --trust-remote-code --attention-backend cutlass_mla --context-length 4096 --cuda-graph-bs 64 128 --max-running-requests 128 ⏎  ⏎ Accuracy: 0.906 ⏎ Accuracy: 0.914 ⏎ Accuracy: 0.917 ⏎ Accuracy: 0.913 ⏎ Accuracy: 0.917 ⏎ ``` ⏎  ⏎ FYI ⏎  ⏎ baseline + triton ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model /dev/shm/DeepSeek-V3-0324 --tp 8 --enable-dp-attention --dp 8 --trust-remote-code --context-length 4096 --cuda-graph-bs 64 128 --max-running-requests 128 ⏎  ⏎ Accuracy: 0.933 ⏎ Accuracy: 0.933 ⏎ Accuracy: 0.937 ⏎ Accuracy: 0.927 ⏎ ``` ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-6716b41786  (L2, 2025-06-09, sha 6716b4178690, PR #7023)
TITLE: Update default settings for blackwell (#7023)
SOURCES: symbol_pickaxe, dependency_pin
STAGE1: change_default; artifacts=L2.dispatch.server_args_defaults,L2.backend.flashinfer_mla; Updates Blackwell default attention backend to FlashInfer, affecting DeepSeek MLA default selection.
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=257,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/model_executor/model_runner.py (+5/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Set flashinfer as default attention backend for blackwell ⏎ - Tune FusedMoE config for Dpsk v3 on blackwell ⏎ - Update flashinfer in blackwell docker image ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-19995dd78e  (L2, 2025-06-10, sha 19995dd78efd, PR #7057)
TITLE: Tiny fix cutlass_mla_get_workspace_size stub incorrect signature (#7057)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_build_dependency; artifacts=L2.kernel.cutlass_mla; Fixes incorrect cutlass_mla_get_workspace_size stub signature in the kernel source.
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-aa46ed34d2  (L2, 2025-06-13, sha aa46ed34d257, PR #7145)
TITLE: Remove 200us slow concat kernel (part 1: kernel) (#7145)
SOURCES: path_core
STAGE1: optimize; artifacts=L2.kernel.cutlass_mla; Changes CUTLASS MLA kernel API to remove the slow concat-kernel path.
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+29/-20); sgl-kernel/benchmark/bench_cutlass_mla.py (+16/-5); sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/include/sgl_kernel_ops.h (+2/-1); sgl-kernel/python/sgl_kernel/attention.py (+26/-20); sgl-kernel/tests/test_cutlass_mla.py (+5/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-c49c1d9226  (L2, 2025-06-13, sha c49c1d9226ad, PR #7020)
TITLE: Remove 200us slow concat kernel (part 2: srt) (#7020)
SOURCES: path_core, symbol_pickaxe
STAGE1: optimize; artifacts=L2.backend.cutlass_mla,L2.kernel.cutlass_mla,L2.model.deepseek_v2_mla; Updates SRT Cutlass MLA backend/model path to use concat-free kernel interface.
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+34/-10); python/sglang/srt/models/deepseek_v2.py (+5/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ gsm8k and mmlu ok after integrating this pr to dev branch ⏎  ⏎ maybe firstly have a brief look at this, then I will split into kernel pr and srt pr and remove temp code ⏎  ⏎ before ⏎  ⏎ ![image](https://github.com/user-attachments/assets/f5a71871-9490-449d-b28a-f1e3847caf25) ⏎  ⏎ after ⏎  ⏎ ![image](https://github.com/user-attachments/assets/852607aa-dad5-4810-aae6-a6d1201b3322) ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-ab1a4fa5cb  (L2, 2025-06-14, sha ab1a4fa5cbb0, PR #7184)
TITLE: [fix] fix cutlass_mla_backend with cuda_graph and add sm_scale for sgl-kernel cutlass_mla (#7184)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L2.kernel.cutlass_mla,L2.backend.cutlass_mla,L2.runner.cuda_graph_mla; Fixes CUTLASS MLA CUDA-graph issue and adds sm_scale argument through kernel/backend.
ARTIFACT_HINTS: L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+3/-2); sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+10/-9); sgl-kernel/benchmark/bench_cutlass_mla.py (+1/-0); sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/include/sgl_kernel_ops.h (+6/-2); sgl-kernel/python/sgl_kernel/attention.py (+7/-2); sgl-kernel/tests/test_cutlass_mla.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ set default num_kv_splits to 1, avoiding cuda graph issue ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-ed89837cf4  (L2, 2025-06-14, sha ed89837cf42e, PR #7186)
TITLE: chore: upgrade sgl-kernel v0.1.8.post2 (#7186)
SOURCES: path_core, dependency_pin
STAGE1: repair_build_dependency; artifacts=L2.kernel.cutlass_mla,L2.backend.cutlass_mla; Upgrades sgl-kernel after CUTLASS MLA fix and updates backend call site.
ARTIFACT_HINTS: L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.runner.cuda_graph_mla
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/attention/cutlass_mla_backend.py (+1/-0); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py (+5/-1)
BODY: ## Motivation ⏎  ⏎ should be merged after https://github.com/sgl-project/sglang/pull/7184 and new version of sgl-kernel bumped ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-5f1ab32717  (L2, 2025-06-14, sha 5f1ab3271762, PR #7163)
TITLE: [EAGLE] Refactor code for page size > 1 & more simplifications (#7163)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L2.backend.flashinfer_mla,L2.pool.mla_token_kv,L2.runner.cuda_graph_mla; Changes draft KV-cache layout/page-size handling consumed by FlashInfer backend; later reverted.
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.pool.mla_token_kv, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+2/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+1/-2); python/sglang/srt/layers/attention/triton_backend.py (+1/-2); python/sglang/srt/mem_cache/memory_pool.py (+61/-0); python/sglang/srt/speculative/eagle_utils.py (+385/-99); python/sglang/srt/speculative/eagle_worker.py (+131/-45); test/srt/test_eagle_infer_b.py (+66/-0)
DEEP_STUDY: deep-study: this PR was reverted by PR 7210 (confirmed_revert, reason=ci_or_test_failure)
BODY: - Unify the draft kv cache layout for page size > 1 and page size =1. ⏎ - Simplify the verify function. ⏎ - Now it supports page size > 1 and top-k > 1 for the flashinfer backend correctly. This combination is still not fully optimized due to some extra page operations and device sync. ⏎ - There is still one remaining todo item to support page size >1 and topk > 1 for other attention backends that really runs page size > 1.  https://github.com/sgl-project/sglang/blob/5f1ab327176297b984cfb957fc84aa7db1b4856f/python/sglang/srt/speculative/eagle_worker.py#L407-L414

### L2-fff10809bf  (L2, 2025-06-15, sha fff10809bfd7, PR #7210)
TITLE: Revert "[EAGLE] Refactor code for page size > 1 & more simplifications" (#7210)
SOURCES: path_core
STAGE1: revert; artifacts=L2.backend.flashinfer_mla,L2.pool.mla_token_kv,L2.runner.cuda_graph_mla; Reverts EAGLE page-size KV-layout changes affecting FlashInfer MLA/backend page tables.
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.pool.mla_token_kv, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+2/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+2/-1); python/sglang/srt/layers/attention/triton_backend.py (+2/-1); python/sglang/srt/mem_cache/memory_pool.py (+0/-61); python/sglang/srt/speculative/eagle_utils.py (+99/-385); python/sglang/srt/speculative/eagle_worker.py (+45/-131); test/srt/test_eagle_infer_b.py (+0/-66)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 7163 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#7163 because it failed some test cases

### L2-21615cc3fe  (L2, 2025-06-16, sha 21615cc3fe7c, PR #7228)
TITLE: Minor style and doc fix (#7228)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L2.backend.flashmla,L2.runner.cuda_graph_mla; Sets FlashMLA CUDA-graph sequence-length fill value while cleaning backend/docs.
ARTIFACT_HINTS: L2.backend.flashmla, L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.backend.fa3_fa4_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+0/-3); python/sglang/srt/layers/attention/flashmla_backend.py (+1/-4); docs/backend/attention_backend.md (+10/-7); python/sglang/srt/layers/attention/flashattention_backend.py (+0/-1)
BODY: - Set get_cuda_graph_seq_len_fill_value as 1 in python/sglang/srt/layers/attention/flashmla_backend.py ⏎ - Fix docs ⏎ - remove unused imports
