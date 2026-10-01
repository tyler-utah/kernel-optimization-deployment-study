### L2-804d9f2e4c  (L2, 2025-04-07, sha 804d9f2e4c49, PR #4760)
TITLE: Add unit test on page_size > 1 and mla and  integration test for Flash Attention 3 (#4760)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+7/-5); python/sglang/srt/layers/quantization/__init__.py (+7/-4); python/sglang/test/attention/test_flashattn_backend.py (+259/-221); python/sglang/test/attention/test_flashattn_mla_backend.py (+285/-0); test/srt/run_suite.py (+1/-0); test/srt/test_fa3.py (+180/-0)
BODY: ## Motivation ⏎  ⏎ Adding more unit tests and integration test for Flash Attention 3 as part of roadmap: #4709 . ⏎  ⏎ Test Results: ⏎ Using Meta-Llama-3.1-8B-Instruct ⏎ ``` ⏎ [2025-03-25 16:37:06 TP0] Load weight end. type=LlamaForCausalLM, dtype=torch.bfloat16, avail mem=63.48 GB, mem usage=15.11 GB. ⏎ [2025-03-25 16:37:06 TP0] KV Cache is allocated. #tokens: 442750, K size: 27.02 GB, V size: 27.02 GB ⏎ [2025-03-25 16:37:06 TP0] Memory pool end. avail me …[truncated]

### L2-f774a0d275  (L2, 2025-04-11, sha f774a0d27557, PR #5302)
TITLE: feat: add blackwell Dockerfile (#5302)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+19/-0); python/pyproject.toml (+11/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-6f8593799b  (L2, 2025-04-11, sha 6f8593799bd1, PR #5303)
TITLE: feat: add blackwell workflow (#5303)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+4/-3); .github/workflows/release-docker-blackwell.yml (+36/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f65b8d5c89  (L2, 2025-04-11, sha f65b8d5c896c, PR #5142)
TITLE: Blackwell Cutlass MLA kernel (#5142)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/CMakeLists.txt (+4/-1); sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+207/-0); sgl-kernel/csrc/common_extension.cc (+5/-0); sgl-kernel/include/sgl_kernel_ops.h (+8/-1); sgl-kernel/python/sgl_kernel/__init__.py (+5/-1); sgl-kernel/python/sgl_kernel/attention.py (+61/-0); sgl-kernel/tests/test_cutlass_mla.py (+81/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Adds blackwell cutlass MLA kernel to sgl-kernel. Requires cutlass 3.9 ⏎ Thanks to @kaixih for kernel and test code   ⏎  ⏎ I will open a second PR soon to add the cutlass mla attention backend. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2eb55770f9  (L2, 2025-04-11, sha 2eb55770f99c, PR #5311)
TITLE: misc: cleanup 3rdparty (#5311)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+0/-4); sgl-kernel/3rdparty/flashinfer (+0/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-115ae2e728  (L2, 2025-04-11, sha 115ae2e728bc, PR #5317)
TITLE: chore: bump sgl-kernel v0.0.8.post2 (#5317)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-57de7c6b5f  (L2, 2025-04-12, sha 57de7c6b5fb3, PR #5210)
TITLE: feat: use fa3 mla by default on hopper (#5210)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+12/-7); python/sglang/srt/model_executor/model_runner.py (+21/-4); python/sglang/srt/utils.py (+9/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ - [Support DP MLA for FA3](https://github.com/sgl-project/sglang/pull/5210/commits/77f31ed842c71f580c6288e2caf19ae7ad99342c)  ⏎  ⏎ ## Benchmark ⏎ ```bash ⏎ python3 -m sglang.launch_server --model-path /tmp/DeepSeek-V3/1d044fd82b15f1cedb197a288e50cc96a2c27205/ --trust-remote-code --tp 8 --enable-dp-attention --dp 8 --host 127.0.0.1 --attention-backend fa3   ⏎  ⏎ python /home/jobuser/sglang/benchmark/gsm8k/bench_sgl …[truncated]

### L2-4879e50c6d  (L2, 2025-04-12, sha 4879e50c6d46, PR #5327)
TITLE: [Feat] Add sparse attn to sgl-kernel (#5327)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+30/-14); sgl-kernel/csrc/common_extension.cc (+22/-0); sgl-kernel/include/sgl_kernel_ops.h (+50/-0); sgl-kernel/python/sgl_kernel/sparse_flash_attn.py (+175/-0); sgl-kernel/tests/test_sparse_flash_attn.py (+348/-0)
BODY: ## Motivation ⏎  ⏎  ⏎ Adapt from: https://github.com/sgl-project/sgl-attn/pull/1 ⏎ Add sparse attn kernel to sgl-kernel.  ⏎ Co-author: @minedec @minminsun ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-812e82f35e  (L2, 2025-04-12, sha 812e82f35e09, PR #5331)
TITLE: fix: solve cu118 issue for cutlass mla (#5331)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+4/-0); .github/workflows/pr-test-sgl-kernel.yml (+12/-6); .github/workflows/release-whl-kernel.yml (+1/-1)
DEEP_STUDY: deep-study correctness case sglang:812e82f35e: class=hardware_compiler_specific; symptom=compile_or_build_failure; introducing=unknown
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-b371f7cd36  (L2, 2025-04-12, sha b371f7cd3626, PR #5332)
TITLE: chore: bump sgl-kernel v0.0.8.post3 (#5332)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-b62e7e99b8  (L2, 2025-04-12, sha b62e7e99b80e, PR #5337)
TITLE: feat: adapt merge_state (#5337)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-sgl-kernel.yml (+7/-1); sgl-kernel/CMakeLists.txt (+4/-0); sgl-kernel/csrc/attention/cascade.cu (+55/-0); sgl-kernel/csrc/common_extension.cc (+2/-0); sgl-kernel/include/sgl_kernel_ops.h (+2/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/python/sgl_kernel/attention.py (+15/-2); sgl-kernel/tests/test_merge_state.py (+138/-0)
BODY: ## Motivation ⏎  ⏎ unblock https://github.com/sgl-project/sglang/pull/5318 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-c138025731  (L2, 2025-04-12, sha c13802573162, PR #5341)
TITLE: misc: update sagemaker Dockerfile (#5341)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.sagemaker (+1/-73)
BODY: ## Motivation ⏎  ⏎ @andjsmi ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f58b929a51  (L2, 2025-04-13, sha f58b929a5185, PR #5342)
TITLE: chore: upgrade sgl-kernel 0.0.8.post3 (#5342)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/quantization/fp8_kernel.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎ unblock https://github.com/sgl-project/sglang/pull/5113 @Fridge003  ⏎  ⏎ ``` ⏎ flash_attn_with_varlen_func ⏎ ``` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-defede5073  (L2, 2025-04-14, sha defede5073fe, PR #5367)
TITLE: Fix DeepSeek DP Attention + torch compile (#5367)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/dp_attention.py (+2/-4); test/srt/parse_results.py (+3/-2); test/srt/test_dp_attention.py (+3/-0)
BODY: ## Motivation ⏎  ⏎ seems because: graph break => need to get torch seed => cuda graph err ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e940dc4f06  (L2, 2025-04-14, sha e940dc4f06a3, PR #5400)
TITLE: chore: bump sgl-kernel 0.0.9 (#5400)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/Makefile (+2/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-8aab7fdb21  (L2, 2025-04-14, sha 8aab7fdb21e8, PR #5401)
TITLE: chore: upgrade sgl-kernel 0.0.9 (#5401)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e9fc2ac7b6  (L2, 2025-04-14, sha e9fc2ac7b611, PR #5384)
TITLE: [PD Bug] fix  MLA get_contiguous_buf_infos error (#5384)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/memory_pool.py (+4/-9)
BODY: ## Motivation ⏎ [PD Bug] fix  MLA get_contiguous_buf_infos error ⏎ ## Modifications

### L2-838fa0f218  (L2, 2025-04-15, sha 838fa0f21855, PR #5420)
TITLE: [minor] cleanup cmakelists.txt (#5420)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+17/-11); .github/workflows/pr-test.yml (+0/-2); sgl-kernel/build.sh (+2/-0)
BODY: 

### L2-88defc4d89  (L2, 2025-04-15, sha 88defc4d89b7, PR #5434)
TITLE: fix: solve release issue (#5434)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); .github/workflows/release-pypi-kernel.yml (+0/-44); .github/workflows/release-whl-kernel-cu118.yml (+3/-3); .github/workflows/release-whl-kernel.yml (+44/-10); sgl-kernel/build.sh (+0/-2)
BODY: ## Motivation ⏎  ⏎ ref https://github.com/sgl-project/sglang/actions/runs/14477264110 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e8f62b20ca  (L2, 2025-04-15, sha e8f62b20ca88, PR #5431)
TITLE: BLackwell cutlass mla: Add check for bad page size/block num combinations (#5431)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/python/sgl_kernel/attention.py (+5/-3); sgl-kernel/tests/test_cutlass_mla.py (+6/-1)
BODY: ## Motivation ⏎  ⏎ This PR allows the blackwell cutlass mla kernel to support page sizes other than 128, as long as `block_num % (128 / PAGE_SIZE) == 0` ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-fa909dc3c4  (L2, 2025-04-15, sha fa909dc3c40a, PR #5344)
TITLE: feat: update model_specific_adjustment (#5344)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+1/-1); python/sglang/srt/model_executor/forward_batch_info.py (+8/-4); python/sglang/srt/model_executor/model_runner.py (+16/-11); python/sglang/srt/utils.py (+26/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Fixed the issue: "RuntimeError: batch_size must be equal to batch_size_k" when `--enable-mixed-chunk` is enabled ⏎  ⏎ ```bash ⏎ python3 -m sglang.launch_server --model  /shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/ --chunked-prefill-size 32 --log-level debug --enable-mixed-chunk --attention-backend fa ⏎ 3 ⏎  ⏎  ⏎ Accuracy: 0.794 ⏎ Invalid: 0.000 ⏎ Latency: …[truncated]

### L2-8ec0bb7d55  (L2, 2025-04-15, sha 8ec0bb7d558d, PR #5436)
TITLE: chore: upgrade sgl-kernel 0.0.9.post1 (#5436)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-ffde65a094  (L2, 2025-04-15, sha ffde65a0942b, PR #5415)
TITLE: [PD] Fix dynamic port support and MLA buffer for Mooncake (#5415)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/base/conn.py (+8/-1); python/sglang/srt/disaggregation/decode.py (+4/-1); python/sglang/srt/disaggregation/mooncake/conn.py (+117/-175); python/sglang/srt/disaggregation/prefill.py (+6/-1); python/sglang/srt/managers/scheduler.py (+1/-0); python/sglang/srt/utils.py (+33/-0)
BODY: This PR fixes dynamic port implementation and the MLA buffer issue to support more use cases, such as multi-node inference (TP > 8) with DeepSeek.

### L2-a42736bbb8  (L2, 2025-04-15, sha a42736bbb8fe, PR #5113)
TITLE: Support MHA with chunked prefix cache for DeepSeek chunked prefill (#5113)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+80/-34); python/sglang/srt/model_executor/forward_batch_info.py (+181/-0); python/sglang/srt/model_executor/model_runner.py (+11/-0); python/sglang/srt/models/deepseek_v2.py (+174/-9); python/sglang/srt/server_args.py (+6/-0); docs/backend/server_arguments.md (+1/-0); docs/references/deepseek.md (+3/-1); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/test/attention/test_prefix_chunk_info.py (+224/-0); test/srt/test_fa3.py (+53/-2)
LABELS: high priority, performance, deepseek
BODY: ## Motivation ⏎  ⏎ The current implementation of MLA is slow when when handling long prefix lengths, such as sequences with 32k input length, as highlighted in issues like #5031. ⏎  ⏎ Profiling revealed that the attention kernel's slow runtime is the primary bottleneck.  Currently, we perform absorption during chunked prefilling, but this approach is inefficient for long prefix lengths, as MLA with absorption has overly large computational intensity  …[truncated]

### L2-4fb05583ef  (L2, 2025-04-17, sha 4fb05583ef38, PR #5481)
TITLE: Deprecate disable-mla (#5481)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+1/-3); python/sglang/srt/model_executor/model_runner.py (+1/-5); python/sglang/srt/models/deepseek_nextn.py (+63/-67); python/sglang/srt/models/deepseek_v2.py (+93/-290); python/sglang/srt/server_args.py (+0/-6); docs/backend/server_arguments.md (+0/-1); docs/references/deepseek.md (+1/-1); python/sglang/srt/managers/schedule_batch.py (+0/-1); python/sglang/srt/models/minicpm3.py (+29/-201)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-6fb29ffd9e  (L2, 2025-04-17, sha 6fb29ffd9e7b, PR #5480)
TITLE: Deprecate enable-flashinfer-mla and enable-flashmla (#5480)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/model_executor/model_runner.py (+7/-11); python/sglang/srt/server_args.py (+6/-8); docs/backend/server_arguments.md (+0/-1); docs/references/deepseek.md (+1/-1); python/sglang/srt/managers/schedule_batch.py (+1/-2); scripts/playground/bench_speculative.py (+3/-8)
BODY: ## Motivation ⏎  ⏎ Deprecate two arguments: `--enable-flashinfer-mla` and `--enable-flashmla` for clarity. ⏎ They should be replaced by `--attention-backend flashinfer` and `--attention-backend flashmla` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-8beb356f0d  (L2, 2025-04-17, sha 8beb356f0daa, PR #5205)
TITLE: Refactor DeepSeek decoder layer branches (#5205)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+47/-21)
BODY: ## Motivation ⏎  ⏎ This code depends on https://github.com/sgl-project/sglang/pull/5190 and please subtract diff from there ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-c08a717c77  (L2, 2025-04-17, sha c08a717c7728, PR #5500)
TITLE: [Feat] Update sgl-kernel flashinfer to latest main version (#5500)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+2/-2); sgl-kernel/csrc/common_extension.cc (+12/-17); sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu (+5/-1); sgl-kernel/csrc/speculative/speculative_sampling.cuh (+4/-4); sgl-kernel/include/sgl_kernel_ops.h (+16/-26); sgl-kernel/python/sgl_kernel/elementwise.py (+108/-8); sgl-kernel/python/sgl_kernel/sampling.py (+213/-38); sgl-kernel/tests/test_sampling.py (+33/-37)
BODY: ## Motivation ⏎  ⏎  ⏎ Update flashinfer. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f28d82997a  (L2, 2025-04-17, sha f28d82997ac5, PR #5518)
TITLE: chore: bump sgl-kernel 0.0.9.post2 (#5518)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-bfa3922451  (L2, 2025-04-18, sha bfa392245159, PR #5476)
TITLE: Avoid computing lse in Ragged Prefill when there's no prefix. (#5476)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-1); docs/backend/server_arguments.md (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+17/-10)
DEEP_STUDY: deep-study: this PR was reverted by PR 5544 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Motivation ⏎ Small tweak to save a bit of compute.  ⏎ cc @Fridge003  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-a6f892e5d0  (L2, 2025-04-18, sha a6f892e5d08f, PR #5544)
TITLE: Revert "Avoid computing lse in Ragged Prefill when there's no prefix.… (#5544)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-1); docs/backend/server_arguments.md (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+10/-17)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 5476 reason=ci_or_test_failure
BODY: … (#5476)" ⏎  ⏎ This reverts commit bfa392245159147a2b7dbd67178c825e5035c329. ⏎  ⏎ fix `python3 models/test_reward_models.py` ⏎  ⏎ @Edenzzzz May you submit a fix for this PR again? @Fridge003  ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2c11f9c2eb  (L2, 2025-04-18, sha 2c11f9c2eba4, PR #5540)
TITLE: chore: upgrade sgl-kernel 0.0.9.post2 (#5540)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/sampler.py (+3/-7); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-613b197e57  (L2, 2025-04-19, sha 613b197e5790, PR #5549)
TITLE: Remove one kernel in per_tensor_quant_mla_fp8 (#5549)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/layers/quantization/fp8_kernel.py (+9/-7); python/sglang/srt/models/deepseek_nextn.py (+8/-2); python/sglang/srt/models/deepseek_v2.py (+32/-9); python/sglang/srt/utils.py (+13/-0)
BODY: ## Motivation ⏎  ⏎ Thanks @Alcanderian for discussing it is acceptable to change APIs in the caller site ⏎  ⏎ ### Accuracy ⏎  ⏎ ``` ⏎ python -m sglang.launch_server --model-path /dev/shm/DeepSeek-V3-0324 --trust-remote-code --tp 16 --dp 16 --enable-dp-attention --enable-deepep-moe --deepep-mode normal --disable-cuda-graph --disable-radix-cache --disable-overlap-schedule --decode-log-interval 1 --host 0.0.0.0 --port 20000 --dist-init-addr 10.10.38.8:1500 …[truncated]

### L2-99456bcacb  (L2, 2025-04-20, sha 99456bcacb81, PR #5432)
TITLE: [perf] introduce deep gemm group_gemm_masked as bmm (#5432)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/layers/quantization/fp8_kernel.py (+108/-4); python/sglang/srt/models/deepseek_v2.py (+86/-16); python/sglang/test/test_block_fp8.py (+167/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ per-token-group quant+deep_gemm's grouped_gemm_masked is generally faster than per-tensor quant+bmm_fp8 ⏎  ⏎ NOTE: Multi-batch throughput is hardly comparable because expert loads differ significantly with difference quant method. ⏎  ⏎ TODO in futher PR: Optimize `_per_token_group_quant_mla_deep_gemm_masked_fp8` with CudaC ⏎  ⏎ Profile files: https://drive.google.com/drive/folders/1guY2rDdd6LZ6qtLb0L3SQjYwj0pIHJjy?usp=sharing ⏎  ⏎ ### Pr …[truncated]

### L2-417b44eba8  (L2, 2025-04-20, sha 417b44eba8b9, PR #5417)
TITLE: [Feat] upgrade pytorch2.6 (#5417)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); .github/workflows/pr-test-sgl-kernel.yml (+1/-1); benchmark/deepseek_v3/README.md (+1/-1); docs/start/install.md (+1/-1); python/sglang/srt/layers/dp_attention.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
LABELS: high priority, dependencies
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-11b23ae97b  (L2, 2025-04-21, sha 11b23ae97bba, PR #5578)
TITLE: Remove extra copy in deepseek forward absorb (#5578)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+9/-13); .github/workflows/pr-test-amd.yml (+7/-7); python/sglang/srt/layers/rotary_embedding.py (+2/-1)
LABELS: high priority
BODY: ## Benchmark ⏎  ⏎ ### Performance ⏎  ⏎ main branch: ⏎ ``` ⏎ batch size: 1 ⏎ latency: 10.73 s ⏎ output throughput: 47.70 token/s ⏎ (input + output) throughput: 95.40 token/s ⏎  ⏎ batch size: 32 ⏎ latency: 20.00 s ⏎ output throughput: 819.37 token/s ⏎ (input + output) throughput: 1638.74 token/s ⏎ ``` ⏎  ⏎ this PR: ⏎ ``` ⏎ batch size: 1 ⏎ latency: 10.67 s ⏎ output throughput: 47.97 token/s ⏎ (input + output) throughput: 95.95 token/s ⏎  ⏎ batch size: 32 ⏎ latency: 19.69 s …[truncated]

### L2-4418f599a5  (L2, 2025-04-22, sha 4418f599a546, PR #5624)
TITLE: Fix FA3 DeepSeek prefill performance regression (#5624)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+6/-2)
BODY: ## Motivation ⏎  ⏎ Fix FA3 DeepSeek prefill performance regression ⏎  ⏎ Significantly boost 20% at context len 1024, the longer context, the more benifitions ⏎  ⏎ cmd: `python3 -m sglang.bench_one_batch --model lmsys/sglang-ci-dsv3-test --batch-size 8 --input-len 1024 --output-len 128 --trust-remote-code` ⏎  ⏎ - before ⏎ ``` ⏎ Prefill. latency: 0.32884 s, throughput:  24911.50 token/s ⏎ Decode. Batch size: 8, latency: 0.01430 s, throughput:    559.60 token/ …[truncated]

### L2-2ed96c7a8a  (L2, 2025-04-22, sha 2ed96c7a8a20, PR #5272)
TITLE: fix flashmla bug (#5272)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.backend.flashmla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashmla_backend.py (+8/-11)
BODY: ## Motivation ⏎  ⏎ fig bug https://github.com/sgl-project/sglang/issues/5154 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-6b6e748775  (L2, 2025-04-22, sha 6b6e7487750b, PR #5638)
TITLE: Remove q concat in FA3 backend for DeepSeek decode (#5638)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla
FILES: python/sglang/srt/layers/attention/base_attn_backend.py (+3/-0); python/sglang/srt/layers/attention/flashattention_backend.py (+22/-6); python/sglang/srt/layers/radix_attention.py (+8/-1); python/sglang/srt/models/deepseek_v2.py (+7/-2)
BODY: ## Motivation ⏎  ⏎ concat for q in FA3 backend can be removed since the interface for q_nope and q_rope are separate. ⏎  ⏎ ### Accuracy ⏎ ``` ⏎ python3 -m sglang.launch_server --model /dev/shm/DeepSeek-V3 --tp 8 --trust-remote-code --attention-backend fa3 --disable-radix ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 1400 --parallel 160 --num-shots 8 ⏎ ``` ⏎ main: ⏎ ``` ⏎ Accuracy: 0.944 ⏎ Invalid: 0.000 ⏎ Latency: 252.757 s ⏎ Output throughput: 567 …[truncated]

### L2-7d0edf3cae  (L2, 2025-04-23, sha 7d0edf3caed4, PR #5688)
TITLE: chore: bump sgl-kernel 0.1.0 (#5688)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-93c6fb12c7  (L2, 2025-04-25, sha 93c6fb12c773, PR #5723)
TITLE: Fix: deepseek forward absorb (#5723)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/layernorm.py (+41/-4)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ #5578 will cause test_mla.py fail in CI after #5646 (revert #5510) layernorm.py to the orginal.   ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ The problematic modifications are likely comes from .contiguous() call in deepseek_v2.py ⏎ so add contiguous check used before #5578. ⏎  ⏎ ## Checklist

### L2-799c4bb502  (L2, 2025-04-26, sha 799c4bb50218, PR #5748)
TITLE: Fuse MLA set kv cache kernel (#5748)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla, L2.pool.mla_token_kv
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+6/-4); python/sglang/srt/layers/radix_attention.py (+5/-2); python/sglang/srt/mem_cache/memory_pool.py (+87/-0); python/sglang/srt/models/deepseek_v2.py (+2/-3)
LABELS: high priority, ready-to-merge
BODY: ## Motivation ⏎  ⏎ Fuse MLA set kv cache kernel and remove k concat operation. Currently only support FA3 backend. Can be applied to other backend with subsequent verification. ⏎  ⏎ ## Benchmark ⏎  ⏎ ### Accuracy ⏎ ``` ⏎ Accuracy: 0.946 ⏎ Invalid: 0.000 ⏎ Latency: 133.934 s ⏎ Output throughput: 1070.970 token/s ⏎ ``` ⏎  ⏎ ### Performance ⏎ main branch: ⏎ ``` ⏎ batch size: 1 ⏎ latency: 8.21 s ⏎ output throughput: 62.39 token/s ⏎ (input + output) throughput: 124.79 to …[truncated]

### L2-84810da4ae  (L2, 2025-04-27, sha 84810da4ae42, PR #5390)
TITLE: Add Cutlass MLA attention backend (#5390)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+278/-0); python/sglang/srt/layers/attention/utils.py (+1/-1); python/sglang/srt/model_executor/model_runner.py (+7/-0); python/sglang/srt/server_args.py (+14/-1); docs/backend/server_arguments.md (+1/-1); python/sglang/srt/managers/schedule_batch.py (+1/-0); sgl-kernel/python/sgl_kernel/attention.py (+3/-0)
LABELS: high priority, ready-to-merge
BODY: ## Motivation ⏎  ⏎ Enables use of the [blackwell cutlass MLA decode kernel](https://github.com/sgl-project/sglang/pull/5142) with deepseek models. ⏎  ⏎ ## Modifications ⏎  ⏎ Adds "cutlass_mla" option for attention backend. ⏎  ⏎ ## Deepseek-R1 Benchmarks ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --host 0.0.0.0 --port 30000 --tp 8 --model-path deepseek-ai/DeepSeek-R1 --trust-remote-code --enable-dp-attention --attention-backend cutlass_mla --dtype float16 -- …[truncated]

### L2-41ac0c6d48  (L2, 2025-04-27, sha 41ac0c6d4839, PR #5690)
TITLE: chore: upgrade sgl-kernel 0.1.0 (#5690)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+9/-0); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-8d463fe351  (L2, 2025-04-28, sha 8d463fe351c4, PR #5868)
TITLE: Cutlass MLA decode - fix dtype error (#5868)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ After #5390 was merged, I ran into the following error. I bisected it and found #5578 to be the cause. ⏎  ⏎ ``` ⏎   File "/trevor/sglang/python/sglang/srt/models/deepseek_v2.py", line 632, in forward ⏎     return self.forward_absorb( ⏎            ^^^^^^^^^^^^^^^^^^^^ ⏎   File "/trevor/sglang/python/sglang/srt/models/deepseek_v2.py", line 741, in forward_absorb ⏎     attn_output = self.attn_mqa(q, k, k_nope, forward_batch) ⏎               …[truncated]

### L2-8e5a6d3441  (L2, 2025-04-29, sha 8e5a6d3441d5, PR #5875)
TITLE: [Fix] Fix a bug for flashmla to run R1 model (#5875)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.backend.flashmla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashmla_backend.py (+3/-0)
BODY: ## Motivation ⏎ When using sglang to run R1 Model by TP = 16, there is a bug for flashmla. ⏎ I investigated this issue and found that it causes illegal access to the HMB when the seq_length is zero, leading to a program crash. ⏎  ⏎ So, we should not pass a invalid parameter to flashmla kernel. I chose the number 1024 because it is 2 to the power of 10. Other numbers would also work. ⏎  ⏎  ⏎  ⏎ Related PR: https://github.com/sgl-project/sglang/pull/5272 ⏎  …[truncated]

### L2-f4c191a712  (L2, 2025-04-29, sha f4c191a712f8, PR #5894)
TITLE: chore: update Dockerfile (#5894)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+5/-8); .github/workflows/release-docker.yml (+3/-3)
BODY: ## Motivation ⏎  ⏎ The GPU installed by default on Lambda is cu128. If running with the image lmsysorg/sglang:latest, an error will occur due to the version issue of nccl that comes with torch 2.6, which needs to be upgraded to the latest version. There doesn't seem to be a need for cu118 at the moment, and in fact cu121 and cu125 were using the same cu124 image before. The presence of srt and all often makes it difficult for people to distinguish  …[truncated]

### L2-799789afed  (L2, 2025-04-29, sha 799789afedcf, PR #5870)
TITLE: Bump Flashinfer to 0.2.5 (#5870)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+27/-16); .github/workflows/pr-test.yml (+0/-2); docs/start/install.md (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+107/-82)
BODY: ## Motivation ⏎ This pull requests is just little modification on the basis of #5538, which fixes conflicts and lints. ⏎ Thanks @AkazaAkane for contribution! ⏎  ⏎ Ref: #5855, #4905, #5023 ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-08acdb5c3d  (L2, 2025-04-30, sha 08acdb5c3db3, PR #5912)
TITLE: [Feat] Scale up fa3 kernel to sm8x arch (#5912)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+15/-9); sgl-kernel/README.md (+8/-0); sgl-kernel/python/sgl_kernel/flash_attn.py (+12/-4); sgl-kernel/tests/test_flash_attention.py (+17/-9)
BODY: ## Motivation ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/issues/5911 ⏎ More detail plz refer fa hopper/ folder. ⏎  ⏎ TODO: ⏎ - test it on A100(Done)/L20/L40/L40s/A*0/3090 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-d353d08b4e  (L2, 2025-04-30, sha d353d08b4e89, PR #5932)
TITLE: chore: bump sgl-kernel 0.1.1 (#5932)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-9a6ad8916d  (L2, 2025-04-30, sha 9a6ad8916dcd, PR #5933)
TITLE: chore: upgrade sgl-kernel 0.1.1 (#5933)
SOURCES: symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); .github/workflows/vllm-dependency-test.yml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/__init__.py (+2/-2); python/sglang/srt/model_executor/model_runner.py (+6/-3); python/sglang/srt/utils.py (+8/-5); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-bf203cb7a2  (L2, 2025-05-04, sha bf203cb7a231, PR #5992)
TITLE: [Fix] Suppress dynamo logging when using flashinfer backend with torch compile (#5992)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+2/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+2/-1)
BODY: ## Motivation ⏎  ⏎ Currently when using flashinfer backend (v0.2.5) with `enable-torch-compile`, the program can run well, but there will be some annoying logs from torch dynamo: ⏎  ⏎ <img width="1267" alt="useless logs" src="https://github.com/user-attachments/assets/1576f7cf-b505-4d9e-97e9-171696394efa" /> ⏎  ⏎ This PR removes these logs by manually setting the logging level of torch dynamo. ⏎  ⏎  ⏎ Reproducing: ⏎ python3 -m sglang.launch_server --model  …[truncated]

### L2-b8559764f6  (L2, 2025-05-05, sha b8559764f64e, PR #5587)
TITLE: [Test] Add flashmla attention backend test (#5587)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: scripts/ci_install_dependency.sh (+3/-0); test/srt/run_suite.py (+1/-0); test/srt/test_flash_mla_attention_backend.py (+64/-0)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎ Test flashmla attention backend ⏎  ⏎ ## Modifications ⏎  ⏎ Test latency and mmlu ability using deepseek-v2 lite model in flashmla backend, ⏎  ⏎ There are 2 different tests: ⏎ - latency ⏎ - mmlu ⏎  ⏎ All passed ⏎  ⏎ Coauthor: @sleepcoo @yyihuang  ⏎  ⏎ ## Checklist

### L2-22da3d978f  (L2, 2025-05-05, sha 22da3d978f8a, PR #5555)
TITLE: Fix "Avoid computing lse in Ragged Prefill when there's no prefix match" (#5555)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-1); docs/backend/server_arguments.md (+2/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+20/-12)
LABELS: ready-to-merge
BODY: ## Motivation ⏎ Reopens #5476 as per #5544. ⏎ Locally passed `python3 test/srt/models/test_reward_models.py`, but we can wait for all CIs before merging.  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-bdd17998e6  (L2, 2025-05-06, sha bdd17998e60f, PR #6045)
TITLE: [Fix] Fix and rename flashmla CI test (#6045)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: test/srt/run_suite.py (+1/-1); test/srt/test_flash_mla_attention_backend.py (+0/-64); test/srt/test_flashmla.py (+86/-0)
BODY: ## Motivation ⏎  ⏎ Current CI test for flashmla is unstable. This PR tries to fix it by rewriting it. Also the name of test file is changed from `test_flash_mla_attention_backend.py` to `test_flashmla.py` for clarity. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-8f508cc77f  (L2, 2025-05-07, sha 8f508cc77f4c, PR #6034)
TITLE: Update doc for MLA attention backends (#6034)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docs/backend/server_arguments.md (+1/-1); docs/references/deepseek.md (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f6f96b0521  (L2, 2025-05-08, sha f6f96b0521ca, PR #6123)
TITLE: [sgl-kernel] fix: fix cu118 compile error (#6123)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+16/-1); sgl-kernel/csrc/grammar/apply_token_bitmask_inplace_cuda.cu (+6/-2)
ISSUES: #5100 [Bug] common_ops.abi3.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb
BODY: ## Motivation ⏎  ⏎  ⏎ See issue : https://github.com/sgl-project/sglang/issues/5100 ⏎ https://github.com/sgl-project/sglang/pull/5686. After this PR, it will cause some symbol error when compile at cu 118 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-6578cf27de  (L2, 2025-05-08, sha 6578cf27de8b, PR #6131)
TITLE: chore: bump sgl-kernel 0.1.2 (#6131)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-0ab3f437ab  (L2, 2025-05-08, sha 0ab3f437aba7, PR #6101)
TITLE: Cutlass MLA: Disable split kv due to https://github.com/NVIDIA/cutlass/issues/2274 (#6101)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+4/-1); sgl-kernel/tests/test_cutlass_mla.py (+1/-1)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎ A bug https://github.com/NVIDIA/cutlass/issues/2274 was found in the cutlass MLA kernel which can cause incorrect outputs. The test cases didn't catch this because the random input values were too low. ⏎  ⏎ ## Modifications ⏎  ⏎ Disable split kv for now until the issue is fixed. Increase random input values so that bug can be detected properly - new values will cause test to fail when using split kv. Thanks @kaixih  ⏎  ⏎ ## Checklist

### L2-e30c273bc9  (L2, 2025-05-08, sha e30c273bc9bb, PR #5822)
TITLE: opt flashinfer mla cat (#5822)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+59/-13); python/sglang/srt/models/deepseek_v2.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎ Base on  #5748 and #5638 , for flashinfer mla, remove q and k cat. ⏎  ⏎ ### Accuracy ⏎ ``` ⏎ Accuracy: 0.951 ⏎ Invalid: 0.000 ⏎ Latency: 228.672 s ⏎ Output throughput: 554.173 token/s ⏎ ``` ⏎  ⏎ ### Performance ⏎ main branch: ⏎ ``` ⏎ {"run_name": "default", "batch_size": 1, "input_len": 1024, "output_len": 1024, "latency": 14.4687, "output_throughput": 70.77, "overall_throughput": 141.55} ⏎  ⏎ {"run_name": "default", "batch_size": 16, "input_ …[truncated]

### L2-b29a026e14  (L2, 2025-05-09, sha b29a026e14b9, PR #6016)
TITLE: KV‑Cache (MHA, MLA): add missing start_layer / end_layer fields to MHATokenToKVPoolHost and MLATokenToKVPoolHost (#6016)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/memory_pool.py (+2/-0)
ISSUES: #6005 [Bug] AttributeError: 'MHATokenToKVPoolHost' object has no attribute 'start_layer' when using --enable-hierarchical-cache
BODY: ## Motivation ⏎  ⏎ Fix #6005  ⏎  ⏎ ## Modifications ⏎  ⏎ both MHATokenToKVPoolHost.__init__ and MLATokenToKVPoolHost.__init__ ⏎ ``` ⏎ self.start_layer = start_layer or device_pool.start_layer ⏎ self.end_layer   = end_layer   or device_pool.end_layer ⏎ ``` ⏎ makes the host shard self‑contained before any loader thread starts. ⏎ No other files are touched; public APIs and CLI flags remain unchanged. ⏎  ⏎ Notes for Reviewers ⏎ This follows the same pattern used by …[truncated]

### L2-45b4dcf037  (L2, 2025-05-11, sha 45b4dcf0375a, PR #6195)
TITLE: chore: bump sgl-kernel v0.1.2.post1 (#6195)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-230106304d  (L2, 2025-05-11, sha 230106304db3, PR #6196)
TITLE: chore: upgrade sgl-kernel v0.1.2.post1 (#6196)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/deep_gemm.py (+57/-67); scripts/ci_install_dependency.sh (+1/-1); scripts/ci_install_dependency_8_gpu.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎ Merged With PR https://github.com/sgl-project/sglang/pull/6194 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e8e18dcdcc  (L2, 2025-05-12, sha e8e18dcdcca0, PR #6244)
TITLE: Revert "fix some typos" (#6244)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla
FILES: 3rdparty/amd/profiling/PROFILING.md (+1/-1); 3rdparty/amd/tuning/TUNING.md (+7/-7); README.md (+1/-1); benchmark/benchmark_vllm_060/README.md (+3/-3); benchmark/deepseek_v3/README.md (+8/-8); benchmark/gsm8k/README.md (+6/-6); benchmark/hellaswag/README.md (+5/-5); benchmark/kernels/fused_moe_triton/README.md (+2/-2); benchmark/mmlu/README.md (+7/-7); benchmark/mtbench/README.md (+5/-5); (+85 more)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 6209 reason=premature_or_process
BODY: Reverts sgl-project/sglang#6209 because the capitalization are minor and it will introduce merge conflicts.

### L2-d738ab52f8  (L2, 2025-05-13, sha d738ab52f86f, PR #6209)
TITLE: fix some typos (#6209)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla
FILES: 3rdparty/amd/profiling/PROFILING.md (+1/-1); 3rdparty/amd/tuning/TUNING.md (+7/-7); README.md (+1/-1); benchmark/benchmark_vllm_060/README.md (+3/-3); benchmark/deepseek_v3/README.md (+8/-8); benchmark/gsm8k/README.md (+6/-6); benchmark/hellaswag/README.md (+5/-5); benchmark/kernels/fused_moe_triton/README.md (+2/-2); benchmark/mmlu/README.md (+7/-7); benchmark/mtbench/README.md (+5/-5); (+85 more)
DEEP_STUDY: deep-study: this PR was reverted by PR 6244 (confirmed_revert, reason=premature_or_process)
BODY: ## Motivation ⏎  ⏎ First off, apologies for the very long PR. But mostly just a systematic find and replace of typos, some capitalization of things like sglang -> SGLang, cuda -> CUDA, lora -> LoRA, etc. (in comments/docs only). As well, various issues to make, the documentation clearer. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2e4babdb0a  (L2, 2025-05-15, sha 2e4babdb0a19, PR #6109)
TITLE: [Feat] Support FlashMLA backend with MTP and FP8 KV cache (#6109)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.flashmla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+8/-4); python/sglang/srt/layers/attention/flashmla_backend.py (+340/-78); python/sglang/srt/layers/attention/utils.py (+2/-2); python/sglang/srt/model_executor/cuda_graph_runner.py (+5/-1); python/sglang/srt/speculative/eagle_worker.py (+13/-0); docs/backend/attention_backend.md (+7/-1); docs/references/deepseek.md (+1/-1); test/srt/test_flashmla.py (+68/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR improves flashmla backend by accelerating decode stage with mtp. The implementation utilizes the feature that flashmla can handle `seq_len_q > 1`. To use flashmla with mtp, an example server args can be `--attention-backend flashmla --speculative-algorithm NEXTN --speculative-num-steps  1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2`. ⏎  ⏎ This PR also supports flashmla backend with fp8 kv-cache, which ha …[truncated]

### L2-839fb31e5f  (L2, 2025-05-16, sha 839fb31e5f6e, PR #6334)
TITLE: [Fix] Improve dependencies for Blackwell image (#6334)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+8/-7)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Move decord out of runtime_common dependency ⏎ - Add flashinfer dependency for blackwell  ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-3d7f7a43c8  (L2, 2025-05-17, sha 3d7f7a43c87f, PR #6368)
TITLE: chore: bump sgl-kernel v0.1.3 (#6368)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/Makefile (+1/-0); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f07c6a009b  (L2, 2025-05-17, sha f07c6a009b42, PR #6377)
TITLE: chore: upgrade sgl-kernel v0.1.3 (#6377)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-155214952b  (L2, 2025-05-18, sha 155214952b05, PR #6323)
TITLE: refactor: Extract repeated member variables in KVCache subclasses to base class. (#6323)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/memory_pool.py (+63/-47)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎  ⏎ Reduce type annotation warning in python/sglang/srt/mem_cache/memory_pool.py ⏎  ⏎ ## Modifications ⏎  ⏎ Extract repeated member variables and initialization code in KVCache subclasses to base class. ⏎ Correct device_pool type annotations in HostKVCache and its derived classes ⏎  ⏎  ⏎ ## Checklist

### L2-d6e1d28c8a  (L2, 2025-05-21, sha d6e1d28c8aff, PR #6476)
TITLE: Refactor DeepSeek attention dispatching (#6476)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+27/-19)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-d71f3f0a2a  (L2, 2025-05-22, sha d71f3f0a2a55, PR #6522)
TITLE: chore: bump sgl-kernel v0.1.4 (#6522)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ unblock https://github.com/sgl-project/sglang/pull/6521 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-0b07c4a99f  (L2, 2025-05-22, sha 0b07c4a99f8a, PR #6532)
TITLE: chore: upgrade sgl-kernel v0.1.4 (#6532)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); test/srt/run_suite.py (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-a6ae3af15e  (L2, 2025-05-22, sha a6ae3af15e84, PR #6059)
TITLE: Support XiaomiMiMo inference with mtp (#6059)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/backend/speculative_decoding.ipynb (+54/-0); python/sglang/srt/configs/model_config.py (+3/-0); python/sglang/srt/model_executor/model_runner.py (+9/-6); python/sglang/srt/models/mimo.py (+0/-0); python/sglang/srt/models/mimo_mtp.py (+220/-0); test/srt/models/test_mtp_models.py (+58/-0)
BODY: ## Motivation ⏎  ⏎ Support XiaomiMiMo inference with mtp ⏎  ⏎ ## Modifications ⏎  ⏎ Add new model support. ⏎ Add corresponding MTP accuracy & latency test ⏎ ### How to start server ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path XiaomiMiMo/MiMo-7B-RL --trust-remote-code \ ⏎ --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 \ ⏎ --speculative-num-draft-tokens 2  --mem-fraction 0.5 ⏎ ``` ⏎  ⏎  ⏎ ## Test Result ⏎ ### test/send_on …[truncated]

### L2-a38376fa99  (L2, 2025-05-24, sha a38376fa9913, PR #6477)
TITLE: Refactor attention into multiple stages (#6477)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+117/-24); python/sglang/srt/operations_strategy.py (+4/-2)
BODY: ## Motivation ⏎  ⏎ dep #6476, plz merge that first and subtract diff from that ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-183d9f969c  (L2, 2025-05-27, sha 183d9f969c24, PR #6638)
TITLE: DeepSeek: enable none block-quant FP8 quantizations (#6638)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+55/-43)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ Enable broader quantization - like none-block FP8 quant for DeepSeek based models ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-0b9557fcd7  (L2, 2025-05-27, sha 0b9557fcd7b2, PR #6380)
TITLE: Disable compiling arch below sm_90 in aarch64 by default (#6380)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+32/-11)
BODY: ## Motivation ⏎ For aarch64, it doesn't need to use arch like sm_7x or sm_8x in most cases, so we disable it by default to make the compile more fast. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-51cdd81f97  (L2, 2025-05-29, sha 51cdd81f9720, PR #6265)
TITLE: [fix][RL] Fix DeepSeekV3ForCausalLM.post_load_weights for multiple update weight (#6265)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+39/-14); python/sglang/srt/utils.py (+8/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ When doing RL, which will call `post_load_weights` a second time apart from the intialization, the origin `post_load_weights` will somehow corrupt the weight when recreate w_kv instead of `copy_`. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-b520d02888  (L2, 2025-05-31, sha b520d0288863, PR #6794)
TITLE: chore: bump sgl-kernel v0.1.5 (#6794)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation  ⏎  ⏎ ref https://github.com/sgl-project/sglang/pull/6788 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-34c63731fc  (L2, 2025-05-31, sha 34c63731fcd3, PR #6795)
TITLE: chore: upgrade sgl-kernel v0.1.5 (#6795)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-1da8d23051  (L2, 2025-06-01, sha 1da8d2305124, PR #6800)
TITLE: chore: update blackwell docker (#6800)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+23/-12)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-a2cb5913a0  (L2, 2025-06-02, sha a2cb5913a009, PR #6805)
TITLE: Add draft extend CUDA graph for flashinfer backend  (#6805)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+32/-0); python/sglang/srt/layers/attention/flashinfer_backend.py (+40/-0); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+3/-1); python/sglang/srt/speculative/eagle_worker.py (+10/-1); test/srt/test_eagle_infer.py (+85/-1)
LABELS: ready-to-merge
BODY: ## Motivation ⏎ Follow up of https://github.com/sgl-project/sglang/pull/6606. ⏎  ⏎ ### Flashinfer ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --speculative-algorithm EAGLE --speculative-draft-model-path lmsys/sglang-EAGLE-LLaMA3-Instruct-8B --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --trust-remote-code --dtype float16 --attention-backend flashinfer ⏎ ``` ⏎ main branc …[truncated]

### L2-562f279a2d  (L2, 2025-06-05, sha 562f279a2d80, PR #6458)
TITLE: [CPU] enable CI for PRs, add Dockerfile and auto build task (#6458)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.xeon (+44/-0); python/pyproject.toml (+1/-1); .github/workflows/pr-test-xeon.yml (+86/-0); .github/workflows/release-docker-xeon.yml (+35/-0); python/sglang/test/test_utils.py (+63/-1); test/srt/run_suite.py (+10/-0)
LABELS: high priority, intel, cpu
BODY: ## Motivation ⏎  ⏎ The PR is for enabling the docker env setup and CI processes for running SGLang on Xeon CPU servers. ⏎  ⏎ ## Modifications ⏎  ⏎ Add the dockerfile, and yml files for CPU part of CI process and auto docker image build-up. ⏎ The test files are updated to enable device specific tests, as well as the CPU test cases. ⏎  ⏎ ## Checklist

### L2-b819381fec  (L2, 2025-06-05, sha b819381feca4, PR #6838)
TITLE: AITER backend extension and workload optimizations (#6838)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.aiter_mla
FILES: .github/workflows/pr-test-amd.yml (+1/-1); docs/references/environment_variables.md (+1/-1); python/sglang/srt/layers/attention/aiter_backend.py (+488/-124); python/sglang/srt/layers/layernorm.py (+27/-2); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-3); python/sglang/srt/layers/quantization/fp8.py (+16/-16); python/sglang/srt/layers/quantization/fp8_utils.py (+6/-10); python/sglang/srt/model_executor/model_runner.py (+10/-0); python/sglang/srt/models/deepseek_v2.py (+17/-0); (+2 more)
LABELS: high priority, aiter
BODY: `Co-author: @kkHuang-amd ` ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎ - DeepSeek optimization ⏎ - 1 simple flag: `SGLANG_USE_AITER` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-5f91c82526  (L2, 2025-06-06, sha 5f91c82526c1, PR #6930)
TITLE: [Feature] Support Flashinfer fmha on Blackwell (#6930)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+5/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+5/-1); python/sglang/srt/layers/quantization/fp8.py (+1/-1); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-6); python/sglang/srt/layers/utils.py (+6/-0)
ISSUES: #6906 [Bug] FMHA using flashinfer cutlass on Blackwell has low accuracy result
BODY: ## Motivation ⏎ Support Flashinfer's FMHA on Blackwell. ⏎ https://github.com/sgl-project/sglang/issues/5855 ⏎ https://github.com/flashinfer-ai/flashinfer/pull/1039 ⏎ Flashinfer needs to be pulled from latest github main and installed from source. ⏎ Adding `--disable-radix` is needed for Deepseek models to use fmha. ⏎  ⏎  ⏎ ## Modifications ⏎ Use `is_sm100_supported()` to decide whether use "cutlass" as fmha backend. ⏎ Move is_sm100_supported() to layers di …[truncated]

### L2-d664ca18f2  (L2, 2025-06-07, sha d664ca18f242, PR #6943)
TITLE: chore: bump sgl-kernel v0.1.6 (#6943)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-6153f2ff6e  (L2, 2025-06-07, sha 6153f2ff6e1b, PR #6945)
TITLE: chore: upgrade sgl-kernel v0.1.6 (#6945)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/deep_gemm.py (+42/-56)
BODY: ## Motivation ⏎  ⏎ Open a new PR due to no write permissions to https://github.com/sgl-project/sglang/pull/6893 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f5599ef124  (L2, 2025-06-07, sha f5599ef12421, PR #6866)
TITLE: Refactor global_server_args_dict (#6866)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/schedule_batch.py (+31/-26); python/sglang/srt/model_executor/model_runner.py (+7/-27)
BODY: ## Motivation ⏎  ⏎ wait for ci ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-8db3ac55a9  (L2, 2025-06-07, sha 8db3ac55a975, PR #6955)
TITLE: chore: bump sgl-kernel v0.1.6.post1 (#6955)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-23881fa60c  (L2, 2025-06-07, sha 23881fa60ce5, PR #6957)
TITLE: chore: upgrade sgl-kernel v0.1.6.post1 (#6957)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-6c0a48282a  (L2, 2025-06-08, sha 6c0a48282a81, PR #6963)
TITLE: chore: bump sgl-kernel v0.1.7 (#6963)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); .github/workflows/pr-test-sgl-kernel.yml (+2/-3); sgl-kernel/build.sh (+3/-7); sgl-kernel/pyproject.toml (+2/-2); sgl-kernel/pyproject_cpu.toml (+2/-2); sgl-kernel/pyproject_rocm.toml (+2/-2); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-18efb5e8e0  (L2, 2025-06-08, sha 18efb5e8e0ed, PR #6929)
TITLE: [perf][sgl-kernel] extend cutlass_mla_decode to support num_head < 128 (#6929)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+52/-22); sgl-kernel/csrc/attention/cutlass_sm100_mla/device/sm100_mla.hpp (+358/-0); sgl-kernel/csrc/attention/cutlass_sm100_mla/kernel/sm100_fmha_mla_reduction.hpp (+198/-0); sgl-kernel/csrc/attention/cutlass_sm100_mla/kernel/sm100_fmha_mla_tma_warpspecialized.hpp (+2018/-0); sgl-kernel/csrc/attention/cutlass_sm100_mla/kernel/sm100_mla_tile_scheduler.hpp (+160/-0); sgl-kernel/benchmark/bench_cutlass_mla.py (+133/-0); sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/include/sgl_kernel_ops.h (+4/-2); sgl-kernel/python/sgl_kernel/attention.py (+18/-8); sgl-kernel/tests/test_cutlass_mla.py (+17/-4)
BODY: ## Motivation ⏎  ⏎ An inelegant workaround, thus the kernel have to be rewrited to support num_head != 128 ⏎  ⏎ 1. add benchmark for cutlass_mla_decode ⏎ 2. extend num_head support range ⏎ 3. improve performance up to 1.8x for page_size == 128 ⏎ 4. fix split_kv thanks to https://github.com/flashinfer-ai/flashinfer/pull/1109, but still has some issue, ref: https://github.com/sgl-project/sglang/pull/6929/files#diff-3aef9e714fbacce71c7a8d812964d83355a6fa29 …[truncated]

### L2-56ccd3c22c  (L2, 2025-06-09, sha 56ccd3c22c71, PR #6958)
TITLE: chore: upgrade flashinfer v0.2.6.post1 jit (#6958)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+8/-6); .github/workflows/vllm-dependency-test.yml (+1/-1); lmms-eval (+1/-0); python/sglang/srt/entrypoints/engine.py (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-0); python/sglang/srt/layers/multimodal.py (+3/-3); python/sglang/srt/layers/quantization/__init__.py (+2/-2); python/sglang/test/test_utils.py (+0/-1); scripts/ci_install_dependency.sh (+12/-3); (+4 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Integrate latest FlashInfer into SGLang for GB200 NVL72 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e58423b2b9  (L2, 2025-06-09, sha e58423b2b9f2, PR #6998)
TITLE: Fix cutlass MLA gets almost zero accuracy (#6998)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+2/-19)
DEEP_STUDY: deep-study correctness case sglang:e58423b2b9: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: Fix issues in cuda graph and now it is roughly same as non-cuda-graph or slightly lower. But seems non-cuda-graph still has bug, thus may need to check separately. ⏎  ⏎ test ⏎  ⏎ ``` ⏎ while true; do python3 benchmark/gsm8k/bench_sglang.py --parallel 10000 --num-questions 1400; done ⏎ ``` ⏎  ⏎ master + no cudagraph ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model /dev/shm/DeepSeek-V3-0324 --tp 8 --enable-dp-attention --dp 8 --trust-remote-code --attention …[truncated]

### L2-dc0705a504  (L2, 2025-06-09, sha dc0705a504fc, PR #6987)
TITLE: Simplify prepare_extend_after_decode (#6987)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/managers/schedule_batch.py (+10/-4); python/sglang/srt/managers/scheduler.py (+3/-4); python/sglang/srt/model_executor/cuda_graph_runner.py (+13/-10); python/sglang/srt/server_args.py (+4/-4); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+3/-1); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+11/-3); python/sglang/srt/speculative/eagle_utils.py (+41/-130); python/sglang/srt/speculative/eagle_worker.py (+53/-18); test/srt/test_full_deepseek_v3.py (+2/-2)
LABELS: high priority
BODY: - Remove redundant padding in prepare_extend_after_decode ⏎ - Remove some redundant device sync ⏎  ⏎ before ⏎ ``` ⏎ acc_length=2.88 ⏎ speed=310.63 token/s ⏎ ``` ⏎  ⏎ after ⏎ ``` ⏎ acc_length=2.88 ⏎ speed=322.92 token/s ⏎ ```

### L2-6716b41786  (L2, 2025-06-09, sha 6716b4178690, PR #7023)
TITLE: Update default settings for blackwell (#7023)
SOURCES: symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=257,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/model_executor/model_runner.py (+5/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Set flashinfer as default attention backend for blackwell ⏎ - Tune FusedMoE config for Dpsk v3 on blackwell ⏎ - Update flashinfer in blackwell docker image ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-4f723edd3b  (L2, 2025-06-10, sha 4f723edd3baf, PR #7038)
TITLE: chore: bump v0.4.7 (#7038)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+6/-2); docker/Dockerfile.rocm (+1/-1); python/pyproject.toml (+1/-1); Makefile (+1/-1); benchmark/deepseek_v3/README.md (+1/-1); docs/references/setup_github_runner.md (+2/-2); docs/start/install.md (+6/-6); python/sglang/version.py (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-cef6655b26  (L2, 2025-06-10, sha cef6655b2694, PR #7045)
TITLE: fix 24.12 docker (#7045)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+7/-7)
BODY: ## Motivation ⏎  ⏎ fix https://github.com/sgl-project/sglang/pull/7043 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2dae104dca  (L2, 2025-06-10, sha 2dae104dca6f, PR #6999)
TITLE: Minor cleanup of fa3 backend (#6999)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.fa3_fa4_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+15/-16); python/sglang/srt/layers/attention/flashattention_backend.py (+48/-48)
LABELS: high priority
BODY: - Remove unnecessary `.to()` ⏎ - Extract common subfunctions

### L2-6406408a70  (L2, 2025-06-10, sha 6406408a7085, PR #7037)
TITLE: Clean up server_args.py (#7037)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/quantization/deep_gemm.py (+1/-1); python/sglang/srt/managers/schedule_batch.py (+7/-6); python/sglang/srt/model_executor/cuda_graph_runner.py (+68/-37); python/sglang/srt/server_args.py (+183/-171); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+3/-0); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+3/-0); python/sglang/srt/utils.py (+13/-0)
BODY: - Move all EP related things into a single group ⏎ - Sort them correctly

### L2-19995dd78e  (L2, 2025-06-10, sha 19995dd78efd, PR #7057)
TITLE: Tiny fix cutlass_mla_get_workspace_size stub incorrect signature (#7057)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-6b12d6a8d5  (L2, 2025-06-10, sha 6b12d6a8d581, PR #7054)
TITLE: Simplify the heuristics for setting --mem-fraction-static (#7054)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+0/-5); python/sglang/srt/model_executor/cuda_graph_runner.py (+1/-1); python/sglang/srt/server_args.py (+56/-44)
BODY: see comments in the code

### L2-7046e0fab7  (L2, 2025-06-12, sha 7046e0fab791, PR #7119)
TITLE: feat: update blackwell setup (#7119)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+2/-2); sgl-kernel/CMakeLists.txt (+10/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-aa46ed34d2  (L2, 2025-06-13, sha aa46ed34d257, PR #7145)
TITLE: Remove 200us slow concat kernel (part 1: kernel) (#7145)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+29/-20); sgl-kernel/benchmark/bench_cutlass_mla.py (+16/-5); sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/include/sgl_kernel_ops.h (+2/-1); sgl-kernel/python/sgl_kernel/attention.py (+26/-20); sgl-kernel/tests/test_cutlass_mla.py (+5/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-8ab7d93c2e  (L2, 2025-06-13, sha 8ab7d93c2e0d, PR #7152)
TITLE: chore: bump v0.1.8.post1 (#7152)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-bec3e48402  (L2, 2025-06-13, sha bec3e484022a, PR #7155)
TITLE: Support new DeepGEMM format in per token group quant (part 2: srt) (#7155)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/fp8_kernel.py (+17/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-c49c1d9226  (L2, 2025-06-13, sha c49c1d9226ad, PR #7020)
TITLE: Remove 200us slow concat kernel (part 2: srt) (#7020)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+34/-10); python/sglang/srt/models/deepseek_v2.py (+5/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ gsm8k and mmlu ok after integrating this pr to dev branch ⏎  ⏎ maybe firstly have a brief look at this, then I will split into kernel pr and srt pr and remove temp code ⏎  ⏎ before ⏎  ⏎ ![image](https://github.com/user-attachments/assets/f5a71871-9490-449d-b28a-f1e3847caf25) ⏎  ⏎ after ⏎  ⏎ ![image](https://github.com/user-attachments/assets/852607aa-dad5-4810-aae6-a6d1201b3322) ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-ab1a4fa5cb  (L2, 2025-06-14, sha ab1a4fa5cbb0, PR #7184)
TITLE: [fix] fix cutlass_mla_backend with cuda_graph and add sm_scale for sgl-kernel cutlass_mla (#7184)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+3/-2); sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+10/-9); sgl-kernel/benchmark/bench_cutlass_mla.py (+1/-0); sgl-kernel/csrc/common_extension.cc (+1/-1); sgl-kernel/include/sgl_kernel_ops.h (+6/-2); sgl-kernel/python/sgl_kernel/attention.py (+7/-2); sgl-kernel/tests/test_cutlass_mla.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ set default num_kv_splits to 1, avoiding cuda graph issue ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-4473320380  (L2, 2025-06-14, sha 44733203800d, PR #7189)
TITLE: chore: bump v0.1.8.post2 (#7189)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); .github/workflows/pr-test-sgl-kernel.yml (+3/-4); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-ed89837cf4  (L2, 2025-06-14, sha ed89837cf42e, PR #7186)
TITLE: chore: upgrade sgl-kernel v0.1.8.post2 (#7186)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.runner.cuda_graph_mla
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/attention/cutlass_mla_backend.py (+1/-0); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py (+5/-1)
BODY: ## Motivation ⏎  ⏎ should be merged after https://github.com/sgl-project/sglang/pull/7184 and new version of sgl-kernel bumped ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-a023856b12  (L2, 2025-06-14, sha a023856b128b, PR #7200)
TITLE: Move host memory pools into a separate file (#7200)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/managers/cache_controller.py (+2/-1); python/sglang/srt/mem_cache/hiradix_cache.py (+4/-2); python/sglang/srt/mem_cache/memory_pool.py (+1/-377); python/sglang/srt/mem_cache/memory_pool_host.py (+380/-0); test/srt/test_hicache_page.py (+1/-1)
BODY: 

### L2-38af4f68a9  (L2, 2025-06-14, sha 38af4f68a90c, PR #7204)
TITLE: Fix grammar abort & Minor style fixes (#7204)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.backend.flashinfer_mla, L2.backend.flashmla, L2.pool.mla_token_kv, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+5/-6); python/sglang/srt/layers/attention/flashmla_backend.py (+1/-3); python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+2/-2); python/sglang/srt/layers/attention/triton_backend.py (+5/-4); python/sglang/srt/layers/radix_attention.py (+2/-3); python/sglang/srt/managers/scheduler.py (+2/-1); python/sglang/srt/mem_cache/memory_pool.py (+0/-3); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+2/-2)
BODY: 

### L2-5f1ab32717  (L2, 2025-06-14, sha 5f1ab3271762, PR #7163)
TITLE: [EAGLE] Refactor code for page size > 1 & more simplifications (#7163)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.pool.mla_token_kv, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+2/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+1/-2); python/sglang/srt/layers/attention/triton_backend.py (+1/-2); python/sglang/srt/mem_cache/memory_pool.py (+61/-0); python/sglang/srt/speculative/eagle_utils.py (+385/-99); python/sglang/srt/speculative/eagle_worker.py (+131/-45); test/srt/test_eagle_infer_b.py (+66/-0)
DEEP_STUDY: deep-study: this PR was reverted by PR 7210 (confirmed_revert, reason=ci_or_test_failure)
BODY: - Unify the draft kv cache layout for page size > 1 and page size =1. ⏎ - Simplify the verify function. ⏎ - Now it supports page size > 1 and top-k > 1 for the flashinfer backend correctly. This combination is still not fully optimized due to some extra page operations and device sync. ⏎ - There is still one remaining todo item to support page size >1 and topk > 1 for other attention backends that really runs page size > 1.  https://github.com/sgl-p …[truncated]

### L2-fff10809bf  (L2, 2025-06-15, sha fff10809bfd7, PR #7210)
TITLE: Revert "[EAGLE] Refactor code for page size > 1 & more simplifications" (#7210)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.pool.mla_token_kv, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+2/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+2/-1); python/sglang/srt/layers/attention/triton_backend.py (+2/-1); python/sglang/srt/mem_cache/memory_pool.py (+0/-61); python/sglang/srt/speculative/eagle_utils.py (+99/-385); python/sglang/srt/speculative/eagle_worker.py (+45/-131); test/srt/test_eagle_infer_b.py (+0/-66)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 7163 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#7163 because it failed some test cases

### L2-21615cc3fe  (L2, 2025-06-16, sha 21615cc3fe7c, PR #7228)
TITLE: Minor style and doc fix (#7228)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashmla, L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.backend.fa3_fa4_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+0/-3); python/sglang/srt/layers/attention/flashmla_backend.py (+1/-4); docs/backend/attention_backend.md (+10/-7); python/sglang/srt/layers/attention/flashattention_backend.py (+0/-1)
BODY: - Set get_cuda_graph_seq_len_fill_value as 1 in python/sglang/srt/layers/attention/flashmla_backend.py ⏎ - Fix docs ⏎ - remove unused imports

### L2-b1286a116a  (L2, 2025-06-16, sha b1286a116aa2, PR #7213)
TITLE: [EAGLE] Refactor code for page size > 1 & more simplifications (#7213)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.flashmla, L2.pool.mla_token_kv, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+2/-2); python/sglang/srt/layers/attention/flashmla_backend.py (+0/-3); python/sglang/srt/layers/attention/flashinfer_backend.py (+1/-2); python/sglang/srt/layers/attention/triton_backend.py (+1/-2); python/sglang/srt/mem_cache/memory_pool.py (+61/-0); python/sglang/srt/speculative/eagle_utils.py (+385/-99); python/sglang/srt/speculative/eagle_worker.py (+131/-48); test/srt/test_eagle_infer_b.py (+66/-0)
BODY: - Unify the kv cache layout during the decode of the draft model for page size > 1 and page size =1. It changed the draft kv cache layout in both `forward_batch.out_cache_loc` and `forward_batch.req_to_token_pool`. This only affects the draft decode phase. If any attention backend builds the page table based on `forward_batch.out_cache_loc` or `forward_batch.req_to_token_pool.req_to_token`, they need to be updated accordingly. ⏎   -  The layout of …[truncated]

### L2-53a525bf33  (L2, 2025-06-16, sha 53a525bf3356, PR #7231)
TITLE: [Eagle] Fix kernel call after updating speculative sampling kernels (#7231)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/speculative/build_eagle_tree.py (+1/-1); python/sglang/srt/speculative/eagle_utils.py (+12/-21); python/sglang/srt/speculative/eagle_worker.py (+1/-1); test/srt/test_fa3.py (+7/-7)
BODY: Following https://github.com/sgl-project/sglang/pull/7207, we change the kernel calls. ⏎  ⏎ - Use a different coin for the final sampling in speculative sampling ⏎ - Remove some uncessary `to(torch.int32)` ⏎  ⏎ ``` ⏎  ⏎ ```

### L2-c64290dcb5  (L2, 2025-06-16, sha c64290dcb59d, PR #7233)
TITLE: Use seq_len_fill_value in the cuda graph runners (#7233)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.fa3_fa4_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+1/-1); python/sglang/srt/model_executor/cuda_graph_runner.py (+3/-3); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+3/-4); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+6/-5); python/sglang/srt/speculative/eagle_worker.py (+4/-4)
BODY: - Use seq_len_fill_value in the cuda graph runners ⏎  ⏎ Thanks @stslxg-nv for pointing this out

### L2-10d60cd41b  (L2, 2025-06-17, sha 10d60cd41bb5, PR #6081)
TITLE: feat: mtp support dp-attention (#6081)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.backend.flashmla, L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.backend.fa3_fa4_mla, L2.backend.aiter_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+1/-0); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+6/-3); python/sglang/srt/layers/attention/flashmla_backend.py (+5/-2); python/sglang/srt/layers/attention/aiter_backend.py (+5/-2); python/sglang/srt/layers/attention/base_attn_backend.py (+1/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+3/-3); python/sglang/srt/layers/attention/flashinfer_backend.py (+8/-5); python/sglang/srt/layers/attention/tbo_backend.py (+3/-3); python/sglang/srt/layers/attention/triton_backend.py (+19/-11); python/sglang/srt/layers/dp_attention.py (+8/-0); (+12 more)
LABELS: high priority
ISSUES: #6080 [Feature] mtp support dp-attention
BODY: ## Motivation ⏎ mtp support dp-attention ⏎  ⏎ -   implemented  MTP  for DP-attention,  also fixed related bugs #4783 #4847  . ⏎ -   Enabled CUDA Graph support for both target and draft models at dp-attention. ⏎ -   Performance Optimizations: Refined gathered_buffer memory allocation during MTP on dp-attention, eliminates redundant GPU allocation (previously scaled by --speculative-num-draft-tokens multiplier),prevents unnecessary memory usage in both  …[truncated]

### L2-20a503c7d1  (L2, 2025-06-18, sha 20a503c7d16e, PR #7331)
TITLE: fix: resolve blackwell deepep image issue (#7331)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+9/-13)
BODY: ## Motivation ⏎  ⏎ - reduce size ⏎ - use ubuntu 22.04 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-906dbc34f1  (L2, 2025-06-19, sha 906dbc34f164, PR #7343)
TITLE: [Docker] optimize dockerfile  remove deepep and blackwell merge it to… (#7343)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+89/-42); docker/Dockerfile.blackwell (+0/-215); docker/Dockerfile.deepep (+0/-114); .github/workflows/release-docker-blackwell.yml (+0/-36); .github/workflows/release-docker-deepep.yml (+0/-47); .github/workflows/release-docker.yml (+15/-4)
BODY: ## Motivation ⏎  ⏎ reduce docker image nums and its size

### L2-5ea5d22170  (L2, 2025-06-22, sha 5ea5d221703d, PR #7409)
TITLE: Fix CPU offloading for MLA memory pool (#7409)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/memory_pool.py (+44/-8)
BODY: 

### L2-50f1b6d6b1  (L2, 2025-06-22, sha 50f1b6d6b164, PR #7441)
TITLE: Remove copy after bmm (#7441)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+18/-4)
BODY: ## Motivation ⏎  ⏎ Remove copy after bmm in dsv3 forward absorb. ⏎  ⏎ main: ⏎ <img width="595" alt="image" src="https://github.com/user-attachments/assets/bcdc1ca3-108f-4bfc-b229-a540b5876bd2" /> ⏎  ⏎ this PR: ⏎ <img width="540" alt="image" src="https://github.com/user-attachments/assets/f6a294a9-8960-457b-9920-46a1663ede28" /> ⏎  ⏎ ## Benchmark ⏎  ⏎ main:  ⏎ ``` ⏎ +----+-------------------+--------------------+---------------------+----------------+---------- …[truncated]

### L2-55e03b10c4  (L2, 2025-06-23, sha 55e03b10c456, PR #7457)
TITLE: Fix a bug in BatchTokenIDOut & Misc style and dependency updates (#7457)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/pyproject.toml (+6/-8); sgl-kernel/CMakeLists.txt (+10/-10); .github/workflows/pr-test.yml (+5/-1); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/managers/scheduler.py (+8/-1); python/sglang/srt/managers/tokenizer_manager.py (+1/-1); python/sglang/srt/server_args.py (+2/-3); python/sglang/srt/utils.py (+3/-7); sgl-kernel/python/sgl_kernel/sampling.py (+1/-1)
BODY: 

### L2-e846d95ef6  (L2, 2025-06-23, sha e846d95ef6ba, PR #7490)
TITLE: chore: bump sgl-kernel v0.2.0 (#7490)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/Makefile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-7eb47b0f3d  (L2, 2025-06-25, sha 7eb47b0f3d0c, PR #6641)
TITLE: [CPU] [BF16] Call fused_experts_cpu, weight_packed_linear and bmm_cpu kernel in DeepSeek model (#6641)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/layers/linear.py (+18/-1); python/sglang/srt/layers/logits_processor.py (+12/-3); python/sglang/srt/layers/moe/fused_moe_native.py (+7/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+73/-14); python/sglang/srt/layers/vocab_parallel_embedding.py (+14/-1); python/sglang/srt/models/deepseek_v2.py (+145/-1); python/sglang/srt/utils.py (+71/-0); sgl-kernel/csrc/cpu/gemm.cpp (+2/-2); test/srt/cpu/test_gemm.py (+1/-1)
LABELS: intel, cpu
BODY: ## Motivation ⏎  ⏎ When CPU has AMX support, replace `moe_forward_native` with `fused_experts_cpu`, replace `F.linear` and `torch.matmul` with `weight_packed_linear`, add `AttnForwardMethod.MLA_FUSED_ROPE_CPU` to optimize the performance. ⏎  ⏎ https://github.com/sgl-project/sglang/pull/6408 (merged), https://github.com/sgl-project/sglang/pull/6614 and https://github.com/sgl-project/sglang/pull/6833 need to be landed first and the current PR will work …[truncated]

### L2-e21aa1df67  (L2, 2025-06-25, sha e21aa1df674c, PR #6793)
TITLE: [PD] Add different TP sizes support for no-MLA models (#6793)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/base/conn.py (+2/-0); python/sglang/srt/disaggregation/mooncake/conn.py (+260/-27); python/sglang/srt/disaggregation/prefill.py (+3/-0)
BODY: ## Motivation ⏎  ⏎ Inspired by https://github.com/sgl-project/sglang/pull/5922, this PR add support for different TP sizes per DP rank for no-MLA models.  ⏎ This enhancement aims to provide more flexibility in configuring PD setups. ⏎  ⏎ In SGLang, the KV cache is managed in pages. When Prefill and Decode stages have differing TP sizes, the transfer of paged KV cache needs careful handling: ⏎  ⏎ - MLA Models or Aligned TP Sizes (Decode TP == Prefill TP  …[truncated]

### L2-a5317b2fd3  (L2, 2025-06-27, sha a5317b2fd3dd, PR #6769)
TITLE: [CPU] add optimizations for INT8 and FP8 DeepSeek (#6769)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-1); python/sglang/srt/layers/quantization/fp8.py (+43/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+1/-1); python/sglang/srt/layers/quantization/w8a8_int8.py (+51/-1); python/sglang/srt/models/deepseek_v2.py (+83/-0)
LABELS: intel, cpu
BODY: ## Motivation ⏎ Call the below kernels in DeepSeek to optimize INT8 and FP8: ⏎ INT8 linear: `int8_scaled_mm_with_quant` ⏎ FP8 linear: `fp8_scaled_mm` ⏎ INT8 and FP8 shared_expert: `shared_expert_cpu` ⏎ INT8 and FP8 MoE: `fused_experts_cpu` ⏎  ⏎ For bmm, currently we don't support weight to be FP8, we convert weight to BF16 (only for CPU).

### L2-392e441ad1  (L2, 2025-06-30, sha 392e441ad17c, PR #7663)
TITLE: chore: upgrade flashinfer v0.2.7 jit (#7663)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-637bfee448  (L2, 2025-06-30, sha 637bfee448a8, PR #7675)
TITLE: chore: bump sgl-kernel v0.2.1 (#7675)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f9eb04ddb2  (L2, 2025-07-01, sha f9eb04ddb26a, PR #7676)
TITLE: upgrade sgl kernel to 0.2.1 for main (#7676)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ Making sure compatibility with latest kernel update. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f18a8fddd4  (L2, 2025-07-01, sha f18a8fddd418, PR #7698)
TITLE: chore: upgrade flashinfer v0.2.7.post1 (#7698)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ fix flashinfer.comm ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-1e0e549766  (L2, 2025-07-03, sha 1e0e549766a0, PR #7722)
TITLE: Ascend attention backend(PA&MLA) (#7722)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.pool.mla_token_kv, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/ascend_backend.py (+219/-0); python/sglang/srt/mem_cache/memory_pool.py (+148/-0); python/sglang/srt/model_executor/forward_batch_info.py (+9/-2); python/sglang/srt/model_executor/model_runner.py (+60/-8); python/sglang/srt/models/deepseek_v2.py (+3/-1); python/sglang/srt/server_args.py (+7/-0); docs/backend/attention_backend.md (+6/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+5/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+38/-0); python/sglang/srt/layers/moe/topk.py (+5/-0); (+7 more)
LABELS: npu
BODY: ## Motivation ⏎  ⏎ Currently only `torch_native` backend works for NPU, it has cycle by batch that works slow for bigger batch, in this MR new Ascend Attention backend for NPU device was implemented. It has big performance improvement for big batch size. ⏎  ⏎ ## Modifications ⏎  ⏎ - PA ⏎     - New attention backend - `AscendAttnBackend`: support paged attention with page size 128 only. ⏎      ⏎        New option for server argument `attention-backend` - " …[truncated]

### L2-aca1101a13  (L2, 2025-07-03, sha aca1101a1396, PR #7755)
TITLE: chore: bump sgl-kernel 0.2.2 (#7755)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-489934be0a  (L2, 2025-07-03, sha 489934be0ad3, PR #7751)
TITLE: fuse renormal into moe topk softmax kernel python code (#7751)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/moe/topk.py (+1/-25)
LABELS: high priority
BODY: ## Motivation ⏎ fuse renormal into moe topk softmax kernel python code, need merge #7744 and update sgl-kernel first ⏎  ⏎  ⏎  ⏎ ``` ⏎ accuracy ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 1000 ⏎ Accuracy: 0.951 ⏎ Invalid: 0.000 ⏎ Latency: 132.842 s ⏎ Output throughput: 1027.763 token/s ⏎ ``` ⏎  ⏎ before ⏎ | index | max_concurrency | input_throughput | output_throughput | mean_ttft_ms | median_ttft_ms | p99_ttft_ms | mean_tpot_ms | median_tpot_ms | p …[truncated]

### L2-da3890e82a  (L2, 2025-07-04, sha da3890e82a97, PR #7772)
TITLE: [1/n]: add cutlass W4A8 moe kernel for hopper architecture (#7772)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+3/-0); sgl-kernel/csrc/common_extension.cc (+19/-0); sgl-kernel/csrc/cutlass_extensions/detail/collective/mixed_input_utils.hpp (+482/-0); sgl-kernel/csrc/cutlass_extensions/gemm/collective/builders/sm90_gmma_builder_mixed_input.inl (+278/-0); sgl-kernel/csrc/cutlass_extensions/gemm/collective/collective_builder_mixed_input.hpp (+52/-0); sgl-kernel/csrc/cutlass_extensions/gemm/collective/collective_mma_array_mixed_input.hpp (+53/-0); sgl-kernel/csrc/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp (+1535/-0); sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu (+91/-0); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh (+92/-0); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu (+240/-0); (+6 more)
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:de4990a5b2 (fix PR 9392)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This is the sgl-kernel part of https://github.com/sgl-project/sglang/pull/7762. A series of PRs will enable running [DeepSeek-R1-W4AFP8](https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8) using sglang. ⏎  ⏎ Additionally, this kernel supports other MoE models with INT4 MoE weight and FP8 activation quantization. ⏎  ⏎ Co-author: yicwang <yichen.wang@bytedance.com> ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ In this kernel implementation, we focus on …[truncated]

### L2-4fece12be9  (L2, 2025-07-05, sha 4fece12be982, PR #7784)
TITLE: chore: bump sgl-kernel v0.2.3 (#7784)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-77cfea689d  (L2, 2025-07-05, sha 77cfea689d41, PR #7786)
TITLE: chore: upgrade sgl-kernel v0.2.3 (#7786)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f200af0d8c  (L2, 2025-07-05, sha f200af0d8cde, PR #7800)
TITLE: chore: bump sgl-kernel v0.2.4 (#7800)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-62f5522ffe  (L2, 2025-07-05, sha 62f5522ffe3f, PR #7801)
TITLE: chore: upgrade sgl-kernel v0.2.4 (#7801)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-8f3173d0b0  (L2, 2025-07-11, sha 8f3173d0b072, PR #7964)
TITLE: chore: bump sgl-kernel v0.2.5 (#7964)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-732fc8e405  (L2, 2025-07-11, sha 732fc8e405df, PR #7971)
TITLE: chore: upgrade sgl-kernel 0.2.5 (#7971)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-42fc44100a  (L2, 2025-07-12, sha 42fc44100a27, PR #7988)
TITLE: [minor] Add server_args check for Llama4 with hybrid (#7988)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+4/-0)
BODY: 

### L2-9379da77de  (L2, 2025-07-13, sha 9379da77de43, PR #7367)
TITLE: SWA Prefix Cache (#7367)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/configs/model_config.py (+2/-3); python/sglang/srt/disaggregation/decode.py (+9/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+30/-0); python/sglang/srt/managers/schedule_batch.py (+110/-48); python/sglang/srt/managers/schedule_policy.py (+68/-27); python/sglang/srt/managers/scheduler.py (+186/-40); python/sglang/srt/managers/tp_worker.py (+14/-0); python/sglang/srt/managers/tp_worker_overlap_thread.py (+11/-0); python/sglang/srt/mem_cache/allocator.py (+1/-16); python/sglang/srt/mem_cache/base_prefix_cache.py (+14/-2); (+6 more)
LABELS: high priority
BODY: ## Motivation ⏎ - Support hybrid LLM architecture prefix cache - interleaved full attention and sliding window attention layers ⏎  ⏎  ⏎ ## Modifications ⏎ - Depends on SWA mem pool: https://github.com/sgl-project/sglang/pull/6563 ⏎ - python/sglang/srt/mem_cache/swa_radix_cache.py:  ⏎   - swa prefix cache logic and swa_memory_pool api usage ⏎   - use a LRU linked list to manage eviction order ⏎ - swa_radix_cache usage integration ⏎  ⏎  ⏎ ## Testing ⏎ Loogle: ⏎  …[truncated]

### L2-a562c8a35c  (L2, 2025-07-14, sha a562c8a35c93, PR #7902)
TITLE: [Dockerfile] Multi-arch support for ROCm (#7902)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.rocm (+95/-35); python/pyproject.toml (+0/-1); python/sglang/srt/layers/quantization/fp8_utils.py (+2/-2)
BODY: ## Motivation ⏎  ⏎ To support multi-arch for ROCm ⏎  ⏎ ## Modifications ⏎  ⏎ - Dockerfile ⏎ - pyproject.toml ⏎  ⏎ ## Checklist

### L2-194841e329  (L2, 2025-07-15, sha 194841e3292e, PR #8058)
TITLE: remove kv_a.congigous in DeepseekV2AttentionMLA (#8058)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎ I noticed that `sgl_kernl.rmsnorm` has already support stride input, so tensor.contiguous is not neede anymore. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ remove `kv_a.contiguous() ` in DeepseekV2 ⏎  ⏎ ## Checklist

### L2-1403ea5694  (L2, 2025-07-18, sha 1403ea56949e, PR #7931)
TITLE: [PD] Support non-MLA models PD different TP with DP attention (#7931)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/mooncake/conn.py (+41/-53)
BODY: ## Motivation ⏎ Fix slice logics to repair the accuracy when using different TP with DP attention for PD disaggregation. ⏎  ⏎ Before this PR: ⏎  ⏎ ```bash ⏎ python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 --host 192.168.0.137 --disaggregation-mode prefill  --tp-size 4 --trust-remote-code --page-size 32 --base-gpu-id 0 --chunked-prefill-size 393216  --enable-dp-attention --dp-size 4 ⏎ python -m sglang.launch_server --model-path Qwe …[truncated]

### L2-cfab0ff6e2  (L2, 2025-07-18, sha cfab0ff6e291, PR #8157)
TITLE: Add GB200 wide-EP docker (#8157)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+357/-0)
BODY: ## Motivation ⏎  ⏎ Creating a special dockerfile to build and reproduce the SGLang GB200 wideEP study ⏎  ⏎ ## Modifications ⏎  ⏎ In additional to the base changes in https://github.com/sgl-project/sglang/pull/7721 ⏎ * Use a fork of DeepEP ⏎ * Install mooncake from source ⏎ * Use CUDA_VERSION=12.8.1 as default ⏎  ⏎  ⏎ ## Checklist

### L2-f98e88b9fb  (L2, 2025-07-19, sha f98e88b9fbbb, PR #8165)
TITLE: chore: bump sgl-kernel v0.2.6 (#8165)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-561dd7b2ce  (L2, 2025-07-19, sha 561dd7b2ce2b, PR #8166)
TITLE: chore: upgrade sgl-kernel 0.2.6 (#8166)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-bb0e8a32b5  (L2, 2025-07-19, sha bb0e8a32b579, PR #8161)
TITLE: Clean up server args (#8161)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: .github/CODEOWNERS (+10/-9); docs/backend/server_arguments.md (+85/-43); python/sglang/srt/configs/model_config.py (+4/-4); python/sglang/srt/managers/scheduler.py (+0/-3); python/sglang/srt/model_loader/utils.py (+4/-4); python/sglang/srt/server_args.py (+281/-275); python/sglang/test/runners.py (+2/-2); test/srt/models/test_transformers_models.py (+3/-3)
BODY: - Reorganize the server args better ⏎ - Rename `--impl` to `--model-impl` ⏎ - Update docs

### L2-429bb0efa2  (L2, 2025-07-20, sha 429bb0efa203, PR #8200)
TITLE: chore: bump sgl-kernel v0.2.6.post1 (#8200)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-74f59ae555  (L2, 2025-07-21, sha 74f59ae55557, PR #8202)
TITLE: chore: upgrade sgl-kernel 0.2.6.post1 (#8202)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-4c605235aa  (L2, 2025-07-23, sha 4c605235aa83, PR #8302)
TITLE: fix: workaround for deepgemm warmup issue (#8302)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/CMakeLists.txt (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-4953f4ca9a  (L2, 2025-07-23, sha 4953f4ca9a3a, PR #8304)
TITLE: chore: upgrade sgl-kernel 0.2.7 (#8304)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-8d1c5b948e  (L2, 2025-07-24, sha 8d1c5b948ed0, PR #8301)
TITLE: chore: upgrade flashinfer v0.2.9rc1 (#8301)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: high priority
BODY: up flashinfer version

### L2-ed2e313eb6  (L2, 2025-07-25, sha ed2e313eb667, PR #8332)
TITLE: Clean up server_args, triton cache manager (#8332)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/entrypoints/engine.py (+0/-6); python/sglang/srt/entrypoints/http_server.py (+20/-32); python/sglang/srt/layers/moe/topk.py (+3/-4); python/sglang/srt/managers/scheduler.py (+5/-6); python/sglang/srt/model_executor/forward_batch_info.py (+8/-7); python/sglang/srt/model_executor/model_runner.py (+0/-1); python/sglang/srt/server_args.py (+59/-49); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+0/-1); python/sglang/srt/utils.py (+0/-65); test/srt/test_deepep_large.py (+1/-1); (+2 more)
BODY: - clean up server args to make them more organized ⏎ - remove triton cache manager since the monkey patch is not needed anymore

### L2-e6312d271d  (L2, 2025-07-26, sha e6312d271d86, PR #8356)
TITLE: Uodate Dockerfile.gb200 to latest sglang (#8356)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+28/-39)
BODY: ## Motivation ⏎  ⏎ Improve the Dockerfile.gb200 build to include latest sglang improvements. ⏎  ⏎ ## Modifications ⏎  ⏎ - Bump sglang version in Dockerfile.gb200 to commit: a167fd0bcb9ef4b0f4331a109e40c8cdc770b026, ⏎ this is a work-around because flashinfer v0.2.9rc1 doesn't build for aarch64. ⏎ - Use sgl kernel 0.2.7 ⏎ - Move NVSHMEM to 3.3.9 ⏎ - pip install mooncake `0.3.5 ` as oppose to install from source ⏎  ⏎ ## Checklist

### L2-10ee89559e  (L2, 2025-07-27, sha 10ee89559ee4, PR #8406)
TITLE: chore: upgrade flashinfer v0.2.9rc2 (#8406)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2810338401  (L2, 2025-07-28, sha 2810338401d0, PR #6338)
TITLE: [feat] Support different attention backends for prefill and decode  (#6338)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: docs/backend/server_arguments.md (+2/-0); python/sglang/srt/layers/attention/hybrid_attn_backend.py (+100/-0); python/sglang/srt/managers/schedule_batch.py (+9/-5); python/sglang/srt/model_executor/model_runner.py (+68/-14); python/sglang/srt/models/deepseek_v2.py (+18/-8); python/sglang/srt/server_args.py (+39/-2); python/sglang/test/runners.py (+4/-0); test/srt/run_suite.py (+1/-0); test/srt/test_hybrid_attn_backend.py (+109/-0)
LABELS: high priority
BODY: ## Motivation ⏎ Follow up on #6151 ⏎  ⏎ Note that speculative decoding is not yet supported in this PR. ⏎  ⏎  ⏎  ⏎ ## Profile ⏎  ⏎ ``` ⏎ python3 -m sglang.bench_offline_throughput --model-path meta-llama/Llama-3.1-8B-Instruct --num-prompts 32 --random-input-len 3500 --random-output-len 25 --random-range-ratio 1 --trust-remote-code --disable-radix-cache --mem-fraction-static 0.9 --profile --reasoning-parser qwen3 --prefill-attention-backend fa3 --decode-att …[truncated]

### L2-43118f5f2a  (L2, 2025-07-30, sha 43118f5f2ad3, PR #8599)
TITLE: chore: bump sgl-kernel v0.2.8 (#8599)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e179e0b797  (L2, 2025-07-31, sha e179e0b79738, PR #8550)
TITLE: update sgl-kernel for EP: python part (#8550)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+4/-9)
BODY: ## Motivation ⏎  ⏎  ⏎ See #8514  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-4b04998d38  (L2, 2025-07-31, sha 4b04998d3854, PR #8632)
TITLE: TRTLLM Gen MLA Decode Kernel Integration (same as #7938) (#8632)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:production-kernel-provenance
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+372/-0); python/sglang/srt/layers/attention/utils.py (+6/-1); python/sglang/srt/model_executor/model_runner.py (+7/-0); python/sglang/srt/models/deepseek_v2.py (+1/-0); python/sglang/srt/server_args.py (+18/-0); docs/backend/attention_backend.md (+9/-0); docs/references/deepseek.md (+3/-3); python/sglang/test/attention/test_trtllm_mla_backend.py (+945/-0)
LABELS: high priority
BODY: Creating a new PR since not able to merge https://github.com/sgl-project/sglang/pull/7938 one due to outdated unresolved conversation. ⏎  ⏎ Commit history is the same nothing new.

### L2-6a7528e623  (L2, 2025-08-01, sha 6a7528e6232f, PR #8685)
TITLE: [bugfix] Fix page size for create_flashmla_kv_indices_triton() for cutlass mla (#8685)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+3/-3); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+6/-6)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/pull/8632 added a new arg to `create_flashinfer_kv_indices_triton` which affected the arg order, so page size was no longer passed correctly to the function in the cutlass mla backend. This caused the accuracy to go to 0. ⏎  ⏎ ## Modifications ⏎  ⏎ Use named arg for page size so the order change won't affect it. ⏎  ⏎ ## Accuracy Test ⏎  ⏎ BEFORE ⏎ ``` ⏎ Accuracy: 0.000 ⏎ Invalid: 1.000 ⏎ Latency: 121.637 …[truncated]

### L2-b27b11919c  (L2, 2025-08-01, sha b27b11919cfa, PR #8694)
TITLE: chore(gb200): update dockerfile to handle fp4 disaggregation (#8694)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+11/-4)
BODY: This dockerfile allows for FP4 disaggregation with DSR1. The commands are as follows  ⏎  ⏎ prefill ⏎  ⏎ ```bash ⏎ NCCL_MNNVL_ENABLE=1 \ ⏎ NCCL_CUMEM_ENABLE=1 \ ⏎ SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \ ⏎ PYTHONUNBUFFERED=1 \ ⏎ python3 -m sglang.launch_server \ ⏎ --disaggregation-transfer-backend nixl \ ⏎ --disaggregation-mode decode \ ⏎ --host 0.0.0.0 \ ⏎ --decode-log-interval 1 \ ⏎ --max-running-requests 1536 \ ⏎ --context-length 4224 \ ⏎ --max-total-tokens=20 …[truncated]

### L2-0a56b721d5  (L2, 2025-08-02, sha 0a56b721d553, PR #8713)
TITLE: chore: bump sgl-kernel v0.2.9 (#8713)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-1ea94d3b92  (L2, 2025-08-04, sha 1ea94d3b926c, PR #8780)
TITLE: chore: upgrade flashinfer v0.2.9 (#8780)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-40e3b2beeb  (L2, 2025-08-05, sha 40e3b2beebef, PR #8782)
TITLE: feat: add trtllm-gen mha from direct call (#8782)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/trtllm_mha_backend.py (+321/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/model_executor/model_runner.py (+11/-0); python/sglang/srt/server_args.py (+18/-0)
LABELS: high priority, ready-to-merge
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎ benchmark with ⏎ `python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1000 --parallel 1000` ⏎  ⏎ Results of the updated kernel: ⏎ `python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct    --trust-remote   --attention-backend trtllm_mha --page-size 64` ⏎  ⏎ > Accuracy: 0.797 ⏎ > Invalid: 0.001 ⏎ > Latency: 6.904 s ⏎ > Output …[truncated]

### L2-4f4e0e4162  (L2, 2025-08-05, sha 4f4e0e4162ac, PR #8827)
TITLE: chore: upgrade flashinfer 0.2.10 (#8827)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-c1d2061f97  (L2, 2025-08-05, sha c1d2061f97ae, PR #8824)
TITLE: Add initial support for gpt-oss (#8824)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+17/-0); python/sglang/srt/layers/attention/triton_backend.py (+85/-14); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+36/-8); python/sglang/srt/layers/linear.py (+0/-5); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+134/-8); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+178/-3); python/sglang/srt/layers/quantization/fp8_utils.py (+29/-0); python/sglang/srt/layers/quantization/mxfp4_tensor.py (+133/-0); python/sglang/srt/layers/quantization/unquant.py (+52/-7); python/sglang/srt/managers/schedule_batch.py (+4/-2); (+2 more)
LABELS: high priority
BODY: Future progress will be tracked here: https://github.com/sgl-project/sglang/issues/8833 ⏎  ⏎ **This PR only works for FP8/BF16 ckpt. The FP8/BF16 ckpt has been uploaded to:** ⏎ `lmsys/gpt-oss-20b-bf16` and `lmsys/gpt-oss-120b-bf16` ⏎  ⏎ Install SGLang: ⏎ 1. local build: `pip install -e "python[all]"` ⏎ 2. normal install should also be fine (you might see version conflict complaints, but should be fine) ⏎  ⏎ Additional install for gpt-oss: ⏎ ``` ⏎ pip3 insta …[truncated]
