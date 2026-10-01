### L2-9d61205dac  (L2, 2025-10-22, sha 9d61205dac19, PR #11922)
TITLE: [lint] improve ruff check (#11922)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.pool.mla_token_kv, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+4/-1); .pre-commit-config.yaml (+3/-1); benchmark/kernels/minmax-text-01-lightning_attention/benchmark_lightning_attention_decode.py (+1/-0); benchmark/kernels/minmax-text-01-lightning_attention/benchmark_lightning_attention_prefill.py (+4/-0); python/sglang/bench_one_batch_server.py (+3/-0); python/sglang/srt/disaggregation/common/conn.py (+2/-2); python/sglang/srt/disaggregation/mooncake/conn.py (+1/-1); python/sglang/srt/entrypoints/openai/serving_responses.py (+2/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+20/-19); (+9 more)
LABELS: run-ci
BODY: Prev PR #11685 added `python/sglang` to the Ruff lint scope, but the pre-commit args were written as: ⏎  ⏎ ``` ⏎ args: [--select=F401,F821, --fixable=F401] ⏎ ``` ⏎  ⏎ These arguments are interpreted as `--select=F401 F821 --fixable=F401`, which leads to  ⏎  ⏎ ``` ⏎ warning: Failed to lint F821: No such file or directory (os error 2) ⏎ ``` ⏎  ⏎ This PR changes the arguments to ⏎ ``` ⏎ args: ⏎   - --select=F401,F821 ⏎   - --fix ⏎ ``` ⏎  ⏎ - Remove `--fixable` since ` …[truncated]

### L2-c23eda8589  (L2, 2025-10-22, sha c23eda8589f6, PR #11985)
TITLE: Fix incorrect KV indices creation when page_size=32 in TRTLLM MLA backend (#11985)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+6/-7); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+4/-6); python/sglang/srt/layers/attention/utils.py (+11/-7); python/sglang/test/attention/test_trtllm_mla_backend.py (+49/-53)
LABELS: run-ci
BODY: ### Problem ⏎ `create_flashmla_kv_indices_triton` was computing iteration count from tokens instead of pages, causing KV indices to be incompletely processed when `page_size=32`. ⏎  ⏎ Root cause: The loop count was calculated using a fixed `BLOCK_SIZE` (4096 tokens): ⏎ ```python ⏎ num_pages_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)  # Wrong: uses tokens ⏎ ``` ⏎  ⏎ This works only for `page_size=64` (where 64 pages × 64 tokens/page = 4096 tokens) but  …[truncated]

### L2-36a4cad7b0  (L2, 2025-10-23, sha 36a4cad7b0ab, PR #11821)
TITLE: Support overlap-spec-v2 with trtllm_mla attention backend (#11821)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+12/-9)
LABELS: high priority, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ To open this feature, use `--enable-beta-spec` ⏎ This pr can increase the peformance of bs=1 from 250 to 280 token/s ⏎  ⏎ After this pr, it's still not fully overlap (figure 1) because these three kernels are too slow (figure 2). Will fix this in the future pr. ⏎  ⏎ <img width="389" height="445" alt="image" src="https://github.com/user-attachments/assets/9498ab97-6b63-46a6-9d0d-7b5dcf98cc2e" /> ⏎  ⏎ <img width="203 …[truncated]

### L2-9a71500cfb  (L2, 2025-10-23, sha 9a71500cfb26, PR #12009)
TITLE: Fixed aarch64 flash-mla (#12009)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-4)
BODY: ## Motivation ⏎ DSV3.2 is unrunnable without flash-mla. flash-mla is buildable on aarch64, guarding `"$TARGETARCH" = "amd64"` is unnecessary. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-47e12e082e  (L2, 2025-10-23, sha 47e12e082e07, PR #12003)
TITLE: Enable Llama 4 + TRTLLM MHA (#12003)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+7/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ I don't know why it was blocked originally from being used (TBD maybe it was either not in yet, or was untested). A lot better than triton (currently) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Unblock it. ⏎  ⏎ ``` ⏎ python -m sglang.launch_server \ ⏎                                                                                                                                                          --model-path /root/.cache/huggingface/hub/models …[truncated]

### L2-4793ec7d1a  (L2, 2025-10-23, sha 4793ec7d1af7, PR #10953)
TITLE: Opt MHA chunked prefix: merge prefix and extend kv cache to run mha once  (#10953)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.backend.fa3_fa4_mla, L2.pool.mla_token_kv, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+18/-10); python/sglang/srt/layers/attention/flashattention_backend.py (+12/-2); python/sglang/srt/layers/attention/utils.py (+78/-0); python/sglang/srt/mem_cache/memory_pool.py (+82/-0); python/sglang/srt/model_executor/forward_batch_info.py (+35/-0); python/sglang/srt/models/deepseek_v2.py (+136/-39)
LABELS: ready-to-merge, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ For fa3 and flashinfer backends, when seq_lens ≤ 128K, fuse prefix and extended kv and perform only one `attn_mha` calculation, and optimize performance through fused operators to avoid multiple copies and type conversions. **MHA performance is generally better than MLA performance**. It is recommended to directly switch to MHA through env `SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0`.  ⏎  ⏎  ⏎ T: SGL_CHUNKED_PREFIX_CACHE_THRESHOLD ⏎ Conf …[truncated]

### L2-f4b78d137c  (L2, 2025-10-24, sha f4b78d137cc2, PR #12000)
TITLE: [1/2] deepseek deterministic: support deterministic inference for deepseek arch models on a single GPU (#12000)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py (+31/-2); python/sglang/srt/models/deepseek_v2.py (+9/-1); python/sglang/srt/server_args.py (+24/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Part of this Issue: https://github.com/sgl-project/sglang/issues/10278 ⏎  ⏎ As part of deepseek deterministic inference support, this change ensures deterministic inference results for deepseek arch models on a single GPU.  ⏎  ⏎ ## Modifications ⏎ 1. Fixed the AttnForwardMethod as MLA instead of determining it at runtime based on batch status.  ⏎ 2. Replace torch.bmm with batch_invariant_bmm when enable deterministic inference. ⏎  ⏎ Curr …[truncated]

### L2-e51046beaa  (L2, 2025-10-24, sha e51046beaa67, PR #12093)
TITLE: perf: trtllm_mla attention backend spec decoding speedup w/ cuda graph (#12093)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+1/-13)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎ The `trtllm_mla` backend was selecting kernels during CUDA graph capture using very small sequence lengths. ⏎ This often led to inefficient kernel choices in most runtime cases, degrading overall performance. ⏎  ⏎ ## Modifications ⏎  ⏎ Now the kernel selection uses `max_context_len` instead of the real input sequence length during graph capture. ⏎ Although finer-grained CUDA graph capture depends on max seq len should be introduced, th …[truncated]

### L2-7ef5d8afd4  (L2, 2025-10-24, sha 7ef5d8afd4d8, PR #12049)
TITLE: Revise POINTSV15Chat model (#12049)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/multimodal/processors/points_v15_chat.py (+2/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Fix the following error: ⏎ ``` ⏎ [2025-10-23 21:06:52] Ignore import error when loading sglang.srt.multimodal.processors.points_v15_chat: cannot import name 'Qwen2_5VLImageProcessor' from 'sglang.srt.multimodal.processors.qwen_vl' (/usr/local/lib/python3.10/dist-packages/sglang/srt/multimodal/processors/qwen_vl.py) ⏎ ``` ⏎ #10911 changed Qwen2_5VLImageProcessor to QwenVLImageProcessor ⏎ Revise POINTSV15Chat model accordingly to avoi …[truncated]

### L2-4b0ac1d52a  (L2, 2025-10-25, sha 4b0ac1d52a6b, PR #12125)
TITLE: Update sgl-kernel version to 0.3.16.post4 (#12125)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎ Fix CI break in  https://github.com/sgl-project/sglang/actions/runs/18796162478/job/53637188677?pr=12098#step:5:558  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-7b36c47b3b  (L2, 2025-10-25, sha 7b36c47b3be2, PR #12136)
TITLE: Clean up attention backend selection code & Other minor rename (#12136)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/entrypoints/http_server.py (+5/-0); python/sglang/srt/managers/scheduler.py (+12/-9); python/sglang/srt/managers/scheduler_metrics_mixin.py (+15/-12); python/sglang/srt/model_executor/model_runner.py (+0/-145); python/sglang/srt/server_args.py (+177/-31); python/sglang/srt/utils/common.py (+73/-78)
LABELS: run-ci
BODY: - move the default attention backend selection logic from `model_runner.py` to `server_args.py`, so we can put all attention backend related things into a single place ⏎ - rename `spec_num_total_accepted_tokens` -> `spec_num_accepted_tokens `, `cum_spec_accept_length` -> `spec_total_num_accepted_tokens`

### L2-cadfae666d  (L2, 2025-10-26, sha cadfae666d10, PR #12170)
TITLE: fix broken deepep/flashmla install in container by adding `--no-build-isolation` (#12170)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2)
LABELS: run-ci
BODY: changes -> https://github.com/sgl-project/sglang/pull/12170#pullrequestreview-3381440603

### L2-88596739a4  (L2, 2025-10-27, sha 88596739a463, PR #11708)
TITLE: Support running FP4 Deepseek on SM120. (#11708)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+2/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+2/-2); python/sglang/srt/layers/quantization/fp8_utils.py (+2/-2); python/sglang/srt/layers/quantization/modelopt_quant.py (+8/-7); python/sglang/srt/models/deepseek_v2.py (+1/-5); python/sglang/srt/models/gpt_oss.py (+1/-10); python/sglang/srt/server_args.py (+4/-3); python/sglang/srt/utils/common.py (+10/-1); sgl-kernel/tests/test_fp8_blockwise_moe.py (+3/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ sm120 is not runnable for deepseek because of sm check and invalid kernels. ⏎  ⏎ ## Modifications ⏎ - add `is_blackwell_supported` as an extend to `is_sm100_supported` ⏎ - bypass dsv3_fused_a_gemm on sm120 as it doesn't work. ⏎ - use flashinfer `fp4_quantize` to replace native `scaled_fp4_quant` as it doesn't work on sm120.  ⏎  ⏎ with the change, we can run on SM120 by: ⏎ ``` ⏎ SGLANG_USE_CUTLASS_BACKEND_FOR_FP4_GEMM=1 \ ⏎ python3 -m sgl …[truncated]

### L2-285a8e6986  (L2, 2025-10-27, sha 285a8e698609, PR #11517)
TITLE: docker: add CUDA13 support in dockerfile and update GDRCopy/NVSHMEM for blackwell support (#11517)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+82/-26); python/pyproject.toml (+3/-22); .github/workflows/release-docker-cu13.yml (+118/-0); .github/workflows/release-docker-dev.yml (+11/-1); docs/get_started/install.md (+3/-1); scripts/ci/ci_install_deepep.sh (+6/-6); scripts/ci/ci_install_dependency.sh (+2/-2)
LABELS: high priority, run-ci
BODY: CUDA13 represents a major bump. Because of this - we do not want to fully default to using it. Instead this container will be used by the team (and others) to develop on CU13 friendly platforms like B/GB300.  ⏎  ⏎ This PR adds ⏎ 1. Support to official dockerfile to build for gb and b300 ⏎ 2. A manual pr trigger that can be used to release a cu13 image for x86/arm ⏎ 3. Updated gdrcopy and nvshmem versions ⏎  ⏎ Will wait for https://github.com/sgl-project …[truncated]

### L2-81a632ace6  (L2, 2025-10-27, sha 81a632ace647, PR #11655)
TITLE: [DeepseekV32] Enable flashmla_prefill kernel with fp8 kvcache (#11655)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/nsa/dequant_k_cache.py (+138/-6); python/sglang/srt/layers/attention/nsa/quant_k_cache.py (+44/-12); python/sglang/srt/layers/attention/nsa_backend.py (+156/-20); python/sglang/srt/server_args.py (+29/-6)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ Add logics to dequant the kvcache from fp8 to bf16 in a separate kernel and use `flashmla_prefill` kernel with fp8 kvcache. ⏎  ⏎  ⏎ flashmla_decode (before) ⏎  ⏎ <img width="2258" height="162" alt="image" src="https://github.com/user-attachments/assets/daa7789a-65de-4343-bd6c-4d66b955460b" /> ⏎  ⏎ flashmla_prefill with no kvcache reuse or chunked prefill (after) ⏎ <img width="1950" height="110" alt="image" src="https://github.com/user-att …[truncated]

### L2-8d6ab1cb88  (L2, 2025-10-28, sha 8d6ab1cb88b5, PR #12295)
TITLE: fix seqlen bug for trtllm_mla's draft_extend (#12295)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+10/-2)
LABELS: high priority, run-ci
BODY: In trtllm_mla's draft extend, we have `seq_lens = forward_batch.seq_lens.to(torch.int32)` which is passed to`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`. `forward_batch.seq_lens.to(torch.int32)` does not take into account the padding we do to the queries with `pad_draft_extend_query`. This leads to mismatched q's and kv's when we call `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`. We should instead have ⏎ ``` ⏎ seq_lens = ( …[truncated]

### L2-42e1a72efb  (L2, 2025-10-28, sha 42e1a72efb78, PR #12294)
TITLE: [Deepseek V3.2] Enable flashmla_auto with MTP (#12294)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+1/-3)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Follow up to https://github.com/sgl-project/sglang/pull/11655 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ With fp8 kvcache on B200, flashmla_sparse can be used in the extend (not draft_extend) phases in both normal and speculative (eagle/MTP) settings.  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ With fp4 checkpoint on 4xB200: ⏎ ``` ⏎  python -m sglang.launch_server --model $MODEL --tp 4 --dp 4 --enable-dp-attention --reasoning-parser deepseek-v3 --kv-cache-dtype fp8_e4 …[truncated]

### L2-bacb3825fe  (L2, 2025-10-29, sha bacb3825fe09, PR #12347)
TITLE: fix: llama 4 + trtllm gen + fp8 kv cache incompatibility (#12347)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+7/-0)
LABELS: run-ci
BODY: I spent way too long trying to debug this issue yesterday bc it used to work fine in the integration PR... ⏎  ⏎ ``` ⏎         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/usr/local/lib/python3.12/dist-packages/flashinfer/prefill.py", line 3482, in trtllm_batch_context_with_kv_cache ⏎     run_func( ⏎   File "python/tvm_ffi/cython/function.pxi", line 678, in core.Function.__call__ ⏎ RuntimeError: Error in function 'trtllm_paged_attent …[truncated]

### L2-a18161875c  (L2, 2025-10-29, sha a18161875c8a, PR #12325)
TITLE: Fix Flashinfer Backend for SM120 Usage (#12325)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+5/-3); python/sglang/srt/layers/attention/flashinfer_backend.py (+2/-2)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ As trtllm backend don't support sm120, we need to use flashinfer backend instead. Current prefill in flashinfer backend for sm120 have to use paged attention as the cutlass backend will fail otherwise. But paged attention only support weight absorb path for MLA. For prefill it's better to not absorb the weights for perf. ⏎  ⏎  ⏎ ## Modifications ⏎ Use fa2 backend in flashinfer MLA for sm120 as cutlass backend don't support sm120. With  …[truncated]

### L2-7ed8ba05cb  (L2, 2025-10-29, sha 7ed8ba05cb38, PR #12182)
TITLE: [CI] Add Llama 3.1 8B FP4 to B200 CI (#12182)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/srt/run_suite.py (+2/-1); test/srt/test_llama31_fp4.py (+58/-0)
LABELS: run-ci
BODY: There is FP4 coverage but only through MLA, so cover it through MHA case as well. ⏎  ⏎ ``` ⏎ Accuracy: 0.620 ⏎ Invalid: 0.000 ⏎ Latency: 2.518 s ⏎ Output throughput: 3138.540 token/s ⏎ {'accuracy': np.float64(0.62), 'invalid': np.float64(0.0), 'latency': 2.518050184000458, 'output_throughput': 3138.539513713902} ⏎ ok ⏎  ⏎ ---------------------------------------------------------------------- ⏎ Ran 1 test in 33.638s ⏎  ⏎ OK ⏎ ``` ⏎  ⏎ Out of 8 runs, the throughpu …[truncated]

### L2-621dfb8886  (L2, 2025-10-29, sha 621dfb88864c, PR #12135)
TITLE: Import flash_mla from sgl-kernel (#12135)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.flashmla, L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla
FILES: docker/Dockerfile (+0/-13); python/sglang/srt/layers/attention/flashmla_backend.py (+1/-1); python/sglang/srt/layers/attention/nsa_backend.py (+3/-3); .github/workflows/pr-test.yml (+1/-1); docs/basic_usage/deepseek_v32.md (+1/-7); scripts/ci/ci_install_dependency.sh (+0/-17); test/srt/run_suite.py (+1/-1); test/srt/test_flashmla.py (+2/-20)
LABELS: run-ci
BODY: ## Motivation ⏎ Following #11717 ⏎ - Import flash_mla from sgl-kernel instead of flash_mla library ⏎ - Remove flash_mla installation operations in Dockerfile/ci scripts ⏎ - Enable test_flashmla.py on CI ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-6a63a9852e  (L2, 2025-10-30, sha 6a63a9852e7c, PR #12403)
TITLE: minor code sync (#12403)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/managers/io_struct.py (+10/-3); python/sglang/srt/metrics/collector.py (+3/-3); python/sglang/srt/server_args.py (+14/-4); scripts/ci/ci_install_dependency.sh (+5/-17)
LABELS: run-ci
BODY: 

### L2-ce6b17c0f9  (L2, 2025-10-30, sha ce6b17c0f94e, PR #11897)
TITLE: [Feature] Support DeepSeek MTP on NPU (#11897)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: .github/workflows/pr-test-npu.yml (+6/-2); python/sglang/srt/layers/attention/ascend_backend.py (+233/-5); python/sglang/srt/managers/schedule_batch.py (+7/-1); python/sglang/srt/model_executor/npu_graph_runner.py (+7/-3); python/sglang/srt/models/deepseek_nextn.py (+11/-2); python/sglang/srt/models/deepseek_v2.py (+7/-2); python/sglang/srt/speculative/draft_utils.py (+16/-0); python/sglang/srt/speculative/eagle_info.py (+42/-36); python/sglang/srt/speculative/eagle_info_v2.py (+68/-25); python/sglang/srt/speculative/eagle_utils.py (+261/-16); (+6 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This pr primarily aims to support deepseek's mtp on ascend npus. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Introduces NPU support for newest eagle framework ⏎ - Includes ascend specific ops for draft tree build/verify ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ <img width="1195" height="149" alt="pr" src="https://github.com/user-attachments/assets/cf9e4d00-322c-4cdc-841f-27299acdd08a" /> ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-a076ec1a7a  (L2, 2025-10-30, sha a076ec1a7af0, PR #12437)
TITLE: Revert "fix llama4 kv cache layout" (#12437)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs/advanced_features/attention_backend.md (+1/-1); python/sglang/srt/server_args.py (+0/-7)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s)  reason=other
BODY: Because https://github.com/sgl-project/sglang/pull/12307 is actually the correct solution ⏎  ⏎ + Update FP8 column for trtllm mha.

### L2-c0652d907b  (L2, 2025-10-31, sha c0652d907b2e, PR #12413)
TITLE: Clean up sgl kernel (#12413)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: sgl-kernel/CMakeLists.txt (+26/-40); sgl-kernel/cmake/flashmla.cmake (+2/-0); python/sglang/srt/entrypoints/openai/protocol.py (+5/-1); sgl-kernel/csrc/common_extension.cc (+52/-49); sgl-kernel/csrc/common_extension_rocm.cc (+9/-9); sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu (+129/-16); sgl-kernel/include/sgl_kernel_ops.h (+43/-35); sgl-kernel/python/sgl_kernel/__init__.py (+3/-226); sgl-kernel/python/sgl_kernel/load_utils.py (+224/-0); sgl-kernel/python/sgl_kernel/moe.py (+20/-2)
LABELS: run-ci
BODY: - minor clean up sgl kernel python and c++ files ⏎ - add some arguments to `topk_softmax`

### L2-a4bf5c6ad2  (L2, 2025-10-31, sha a4bf5c6ad25d, PR #12469)
TITLE: Support Kimi Linear (#12469)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.pool.mla_token_kv, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+3/-0); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/kimi_linear.py (+160/-0); python/sglang/srt/configs/mamba_utils.py (+66/-0); python/sglang/srt/configs/model_config.py (+7/-0); python/sglang/srt/layers/attention/fla/chunk_delta_h.py (+61/-32); python/sglang/srt/layers/attention/fla/fused_recurrent.py (+17/-4); python/sglang/srt/layers/attention/fla/kda.py (+1359/-0); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+223/-0); python/sglang/srt/layers/attention/triton_backend.py (+4/-1); (+8 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Support Kimi Linear model (https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct). ⏎  ⏎ Major work is done by @yizhang2077 .  Thanks @zhiyuan1i for valuable discussion. ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model moonshotai/Kimi-Linear-48B-A3B-Instruct --tp 4 --trust-remote ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 1319 --parallel 1319 ⏎  ⏎ Accuracy: 0.895 ⏎ Invalid: 0.000 ⏎ Latency: 46.696 s ⏎ Output through …[truncated]

### L2-d5b6e50fe8  (L2, 2025-10-31, sha d5b6e50fe814, PR #12435)
TITLE: perf: trtllm mla performance minor improvements (#12435)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+70/-65)
LABELS: high priority, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ The current `trtllm_mla` backend has several minor inefficiencies and confusing parts: ⏎  ⏎ 1. Element-wise operations on `seq_lens` are repeatedly executed inside CUDA Graph replay, instead of being done once during metadata initialization.   ⏎ 2. A redundant `torch.cat` operation (`k` + `k_rope`) exists in `draft_extend`, `draft`, and `verify` paths. This operation has been moved so it only happens during prefill.   ⏎ 3. Some logic …[truncated]

### L2-756ad9ceb1  (L2, 2025-11-01, sha 756ad9ceb14b, PR #12238)
TITLE: Reduce docker image size. mount cache when use pip/cargo build (#12238)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+21/-21)
LABELS: run-ci
BODY: When building an image with Docker, for installations via pip and cargo， and  apt , you should use mount cache to prevent the built image from becoming excessively large. ⏎ CC @zhyncs  @slin1237

### L2-6f858930c8  (L2, 2025-11-01, sha 6f858930c8fa, PR #12488)
TITLE: [Bug] test_flashattn_mla_backend errors in Hopper #12487 (#12488)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/test/attention/test_flashattn_mla_backend.py (+58/-13)
LABELS: run-ci
BODY: ## Motivation ⏎ https://github.com/sgl-project/sglang/issues/12487 ⏎  ⏎  ⏎ ## Modifications ⏎   Summary of All Fixes ⏎  ⏎   1. ✅ Added server_args mock with required attributes: ⏎     - kv_cache_dtype ⏎     - speculative_eagle_topk ⏎     - speculative_num_draft_tokens ⏎     - enable_deterministic_inference ⏎   2. ✅ Added missing kv_cache_dtype and is_hybrid attributes to MockModelRunner ⏎   3. ✅ Fixed MLA tensor format by splitting into k_nope and k_rope comp …[truncated]

### L2-76196b3cbf  (L2, 2025-11-01, sha 76196b3cbf8a, PR #10078)
TITLE: feat: Add FP4 (E2M1) KV Cache Support with Quantization Utilities for MLA (#10078)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.pool.mla_token_kv, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+29/-13); python/sglang/srt/mem_cache/memory_pool.py (+120/-148); python/sglang/srt/model_executor/model_runner.py (+27/-0); python/sglang/srt/server_args.py (+4/-4); python/sglang/srt/layers/quantization/kvfp4_tensor.py (+112/-0); python/sglang/srt/mem_cache/utils.py (+210/-0); python/sglang/srt/utils/common.py (+6/-0); python/sglang/test/test_kvfp4_quant_dequant.py (+116/-0)
LABELS: high priority, quant, run-ci
BODY: ## Summary ⏎ This PR introduces FP4 (E2M1) support for Multi-Head Latent Attention (MLA) KV cache in SGLang, enabling low-precision caching to reduce memory usage and improve inference efficiency. It integrates FP4 quantization utilities, Triton kernels, and unit tests while remaining backward compatible with FP16/FP8. See #10083, points 1-1, for more context. ⏎  ⏎  ⏎  ⏎ ## Usage ⏎ Added `--kv-cache-dtype=fp4_e2m1` option. ⏎ ```  ⏎ $ python3 -m sglang.lau …[truncated]

### L2-15ed27d7d4  (L2, 2025-11-02, sha 15ed27d7d415, PR #12453)
TITLE: [Fix] `concat_mla_absorb_q_kernel` fails for long inputs (#12453)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.kernel.concat_mla
FILES: sgl-kernel/csrc/elementwise/concat_mla.cu (+6/-6)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/issues/12250 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ This is caused by **int32 overflow** in addressing like `BBufType* base_addr = reinterpret_cast<BBufType*>(out + idx_0 * out_stride_0 + idx_1 * out_stride_1 + A_LAST_DIM);`.  ⏎  ⏎ Specifically, the `out_stride_0` can be 128\*576=73728, for S=30000, the  `idx_0 * out_stride_0` can be 30000\*73728=2,211,840,000, which exceeds int32 limit (2,147,483,647). ⏎  …[truncated]

### L2-9434a0e50f  (L2, 2025-11-02, sha 9434a0e50f53, PR #12502)
TITLE: [Refact] Remove hardcoded KV cache dimension in MLATokenToKVPool (#12502)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/memory_pool.py (+17/-2)
LABELS: run-ci
BODY: ## Motivation ⏎ Remove the [hardcode](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py#L1422) MLA KV Cache for NSATokenToKVPool ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎   The MLATokenToKVPool class contained a hardcoded value 656 for kv_cache_dim when using NSA (Neural Sparse Attention) with FP8 quantization. This hardcoded value made the code inflexible and error-prone when: ⏎   1. Using different model configurations wit …[truncated]

### L2-0c3543d7d5  (L2, 2025-11-02, sha 0c3543d7d507, PR #12523)
TITLE: chore: upgrade flashinfer 0.5.0 (#12523)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+3/-1); python/sglang/check_env.py (+2/-0); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+3/-1); scripts/ci/ci_install_dependency.sh (+2/-1); sgl-kernel/build.sh (+1/-1)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-6e29446e45  (L2, 2025-11-02, sha 6e29446e45e8, PR #12530)
TITLE: [hotfix] Remove flashinfer-jit-cache from pyproject (#12530)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+0/-1); scripts/ci/ci_install_dependency.sh (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎ flashinfer-jit-cache cannot be installed without extra-url-index, which might hurt user experience ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-b419e20c5b  (L2, 2025-11-04, sha b419e20c5b62, PR #8784)
TITLE: [Dockerfile] Speed up docker image building (#8784)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+38/-24)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ It may be very slow or competely unaccessable when build docker image in some special network environment, it's a block if we want to build the image ourselves. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ 1. Introduce `UBUNTU_MIRROR` to allow replacing default Ubuntu package sources, e.g. `http://mirrors.aliyun.com` ⏎ 2. Introduce `PIP_DEFAULT_INDEX` to enable setting a custom PyPI index URL, e.g. `https://mirrors.aliyun.com/pypi/simple/` ⏎ 3. Intr …[truncated]

### L2-dc4f541823  (L2, 2025-11-05, sha dc4f54182374, PR #12687)
TITLE: fix trtllm_mla attention backend when disabling cuda graph. (#12687)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+2/-2)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ When `spec_v2` is enabled, the draft extend path does not use CUDA Graph. However, in `init_forward_metadata`, `seq_lens` is not converted to `torch.int32`, which causes incorrect computation in `trtllm_batch_decode_with_kv_cache_mla`, leading to a drop in acceptance length when bs>1. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ -  serving command ⏎ ``` ⏎ SGLANG_ENABLE_SPEC_V2=[0/1] \ ⏎ python3 -m sglang.launch_server \ ⏎   --mo …[truncated]

### L2-97be66c358  (L2, 2025-11-05, sha 97be66c35830, PR #12723)
TITLE: fix sgl-kernel version (#12723)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ sgl kernel version had been bumped in Dockerfile, but it would be overwritten when sgl be installed. ⏎  ⏎  ⏎ ``` ⏎ #14 2.391 Collecting sgl-kernel==0.3.16.post5+cu130 ⏎ #14 2.764   Downloading https://github.com/sgl-project/whl/releases/download/v0.3.16.post5/sgl_kernel-0.3.16.post5+cu130-cp310-abi3-manylinux2014_aarch64.whl (252.3 MB) ⏎ #14 5.681      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 252.3/252.3 MB 87.9 MB/s  0:00:02 ⏎ #14 5.93 …[truncated]

### L2-f235498eca  (L2, 2025-11-05, sha f235498eca7a, PR #11892)
TITLE: DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill (#11892)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+84/-0); python/sglang/srt/layers/attention/nsa_backend.py (+61/-2); python/sglang/srt/models/deepseek_v2.py (+43/-2)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ For DeepSeek-V3.2 models, using MLA (Multi-Latent Attention) uniformly across all sequence lengths during prefill is suboptimal. For short sequences, the overhead of MLA's compression/decompression and absorbed attention mechanism outweighs any potential benefits, making standard MHA (Multi-Head Attention) more efficient. This PR implements adaptive attention mechanism selection based on sequence length to optimize inference perf …[truncated]

### L2-a119363f08  (L2, 2025-11-06, sha a119363f0863, PR #12782)
TITLE: ignore the deepgemm check when the model weight with nvfp4 and moe ba… (#12782)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+9/-1)
LABELS: run-ci
BODY: …ckend is flashinfer cutedsl ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ support DS R1 FP4 model deploy to GB200 with deepep and flashinfer-cutedsl moe backend. ⏎  ⏎ ## Modifications ⏎  ⏎ ease the DeepGEMM check when the model weight is fp4 quantitized and the moe backend is flashinfer-cutedsl ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ For the decode ⏎  ⏎ **Rank 0** ⏎  ⏎ ```bash ⏎ NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_SOCKET_FAMILY=AF_INET GLOO_SOCKET_IFNAME=eth0 …[truncated]

### L2-7257525cce  (L2, 2025-11-06, sha 7257525ccea6, PR #12788)
TITLE: [DeepSeek-V3.2][NSA] Enable MHA Pathway for Short Sequence Prefill on B200 (SM100) (#12788)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+46/-2); python/sglang/srt/models/deepseek_v2.py (+7/-4)
LABELS: deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Follow up this PR: [DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill](https://github.com/sgl-project/sglang/pull/11892). Enable and optimize MHA on B200 (SM100) in the NSA backend by TRT-LLM ragged attention. ⏎  ⏎ ## Modifications ⏎  ⏎ - **Add TRT-LLM ragged attention for SM100**: Integrate `flashinfer.prefill.trtllm_ragged_attention_deepseek` for Blackwell (B200) architecture to provide better accuracy th …[truncated]

### L2-bef37d6de8  (L2, 2025-11-07, sha bef37d6de86a, PR #12816)
TITLE: [Deepseek V3.2] Only skip Indexer logits computation when is_extend_without_speculative (#12816)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+7/-4); python/sglang/srt/model_executor/forward_batch_info.py (+7/-0); python/sglang/srt/models/deepseek_v2.py (+6/-14)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Fixed a bug in https://github.com/sgl-project/sglang/pull/12788 ⏎  ⏎ ## Modifications ⏎  ⏎ We should only skip Indexer logits computation when forward_mode.is_extend_without_speculative() returns true, meaning that no cuda graph is involved. The `is_extend_without_speculative` was only added to the MHA path, but not the MLA path. So when MLA is used (on B200 for example) and mtp is on,, the bug is triggered. ⏎  ⏎ ## Accuracy Tests ⏎ on  …[truncated]

### L2-0f76976c3c  (L2, 2025-11-07, sha 0f76976c3ccf, PR #12801)
TITLE: remove the fa4 page_size hardcode to 128 restriction on mla model arch (#12801)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+2/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ When prefill attn backend set to FA4, the page size is hardcode to 128, which lead to the DS R1 FP4 model large scale EP deployment could not deploy with the following combination due to the trtllm-mla only support 16, 32, 64 page size.  ⏎ - Prefill: FA4 attn backend, flashinfer_trtllm moe backend ⏎ - Decode: trtllm-mla attn backend, flashinfer_cutedsl moe backend + deepep low_latency a2a ⏎  ⏎ ## Modifications ⏎  ⏎ remove the page size …[truncated]

### L2-55e8e3999c  (L2, 2025-11-07, sha 55e8e3999ca1, PR #12851)
TITLE: add back flashinfer jit cache to dev docker (#12851)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+4/-0); .github/workflows/release-docker-dev.yml (+1/-0)
LABELS: run-ci
BODY: for ease of development & since the dev docker does not have a slim requirement

### L2-190002c613  (L2, 2025-11-08, sha 190002c613bd, PR #12868)
TITLE: [Docs][DeepseekV3.2] Update deepseekv3.2 docs for mha short seq prefill (#12868)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docs/basic_usage/deepseek_v32.md (+3/-2)
LABELS: documentation, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Update DeepSeek V3.2 documentation to document the adaptive MHA short-sequence prefill mechanism, helping users understand the new attention pathway selection logic. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Update `docs/basic_usage/deepseek_v32.md` ⏎ - Add a new bullet point "Short-sequence MHA prefill (adaptive)" explaining: ⏎   - Default threshold of 2048 tokens ⏎   - H200 (SM90) uses FlashAttention varlen ⏎   - B200 (SM100) uses TRT-LLM ragged  …[truncated]

### L2-5f02b918ec  (L2, 2025-11-08, sha 5f02b918ec79, PR #12361)
TITLE: [Fix] Fix trtllm-mla backend when chunked prefix cache is disabled (#12361)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+19/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-db24d34603  (L2, 2025-11-10, sha db24d34603d3, PR #11812)
TITLE: Support piecewise cuda graph for MLA (#11812)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/compilation/compile.py (+3/-0); python/sglang/srt/compilation/piecewise_context_manager.py (+1/-2); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+5/-0); python/sglang/srt/layers/radix_attention.py (+23/-3); python/sglang/srt/model_executor/model_runner.py (+15/-8); python/sglang/srt/models/deepseek_v2.py (+29/-12); python/sglang/srt/layers/rotary_embedding.py (+0/-1); python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py (+77/-60); test/srt/run_suite.py (+1/-1); test/srt/test_piecewise_cuda_graph.py (+40/-0)
LABELS: deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Support piecewise cuda graph for MLA. ⏎  ⏎ ### Triton ⏎ ``` ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V2-Lite --enable-piecewise-cuda-graph --piecewise-cuda-graph-max-tokens 8192 --attention-backend triton ⏎ python3 benchmark/gsm8k/bench_sglang.py --parallel 1319 --num-questions 1319 ⏎  ⏎ Accuracy: 0.387 ⏎ Invalid: 0.005 ⏎ Latency: 16.465 s ⏎ Output throughput: 9663.436 token/s ⏎ ``` ⏎ ### Flashinfer ⏎ ``` ⏎ python3 -m sgl …[truncated]

### L2-3594815a8b  (L2, 2025-11-10, sha 3594815a8b29, PR #12885)
TITLE: Re-enable Flashinfer TRTLLM GEN MHA and Add Unit Test (#12885)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+2/-1); test/srt/nightly/test_flashinfer_trtllm_gen_attn_backend.py (+62/-0); test/srt/run_suite.py (+1/-0)
LABELS: run-ci, nvidia
BODY: ## Motivation ⏎ Re-enable Flashinfer TRTLLM GEN MHA BF16. And add unit test for it. It was disabled non-intentionally. ⏎ [Related PR](https://github.com/sgl-project/sglang/pull/11138) cc @DomBrown  ⏎ [Related PR](https://github.com/sgl-project/sglang/pull/10909) cc @netanel-haber  ⏎  ⏎ cc @yizhang2077  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-37c40a87a8  (L2, 2025-11-10, sha 37c40a87a8cb, PR #12966)
TITLE: chore: bump sgl-kernel version to 0.3.17 (#12966)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.17` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.17` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are updated accord …[truncated]

### L2-4eda9969e8  (L2, 2025-11-12, sha 4eda9969e8b9, PR #12215)
TITLE: [DeepseekV32]: use `_concat_mla_absorb_q_general` to replace `torch.cat` (#12215)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+7/-6)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/issues/11989  ⏎ `torch.cat([q_nope, q_rope], dim=-1)` is heavily used in `nsa_backend`, which is less efficient. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Use existed kernel `_concat_mla_absorb_q_general` to replace. ⏎  ⏎ ## Checklist

### L2-2d531946db  (L2, 2025-11-12, sha 2d531946dbbd, PR #12214)
TITLE: [Ascend][feature] support L1+ L2 radixcache on ascend (#12214)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs/advanced_features/server_arguments.md (+2/-2); python/sglang/srt/layers/attention/ascend_backend.py (+5/-0); python/sglang/srt/managers/cache_controller.py (+16/-11); python/sglang/srt/mem_cache/memory_pool_host.py (+94/-4); python/sglang/srt/server_args.py (+21/-2); python/sglang/srt/utils/common.py (+5/-0); scripts/ci/npu_ci_install_dependency.sh (+1/-1); test/srt/ascend/test_ascend_hicache_mha.py (+98/-0); test/srt/run_suite.py (+1/-0)
LABELS: documentation, hicache, run-ci
ISSUES: #10844 [Bug] radix-cache can not work on ascend-npu | #11055 NPU  use mooncake as kvcahe but error
BODY: ## Motivation ⏎ The sglang prefix cache feature already supports a three-level caching strategy (L1: HBM, L2: DRAM, L3: storage) on GPUs. However, on NPUs, prefix cache is currently not supported. Therefore, our plan is to first enable L1 + L2 prefix cache support on NPUs in this PR, and add L3 support in the coming weeks. ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ 1. In `server_args.py`, we introduced two new parameters — **`kernel_ascend`** and **`page_fi …[truncated]

### L2-706502ff6c  (L2, 2025-11-12, sha 706502ff6cff, PR #13075)
TITLE: [VLM] Support PP for Qwen2.5-VL (#13075)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/mm_utils.py (+39/-36); python/sglang/srt/managers/scheduler.py (+5/-3); python/sglang/srt/models/qwen2_5_vl.py (+44/-15)
LABELS: feature, performance, Multi-modal, run-ci, vlm
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is to support PP for Qwen2.5-VL model. ⏎  ⏎ ``` ⏎ [root  /root] 二 11月 11 20:23:51  ⏎ $python3 -m sglang.launch_server --model /home/admin/Qwen2.5-VL-7B-Instruct --tp 2 --pp-size=2 ⏎ INFO 11-11 20:29:05 [__init__.py:216] Automatically detected platform cuda. ⏎ [2025-11-11 20:29:05] WARNING server_args.py:1183: Attention backend not explicitly specified. Use flashinfer backend by default. ⏎ [2025-11-11 20:29:05] WARNING server_a …[truncated]

### L2-6664083522  (L2, 2025-11-13, sha 666408352243, PR #12376)
TITLE: Replace [silu_and_mul_]scaled_fp4_group_quant by Flashinfer equivalent (#12376)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmark/kernels/quantization/bench_fp4_quant.py (+11/-8); docs/references/environment_variables.md (+1/-0); python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py (+8/-8); sgl-kernel/tests/test_fp4_quantize.py (+10/-11); test/srt/test_cutedsl_moe.py (+6/-6); test/srt/test_fp4_moe.py (+6/-6)
LABELS: documentation, performance, quant, sgl-kernel, run-ci
BODY: ## Motivation ⏎ Flashinfer Introduced[1927](https://github.com/flashinfer-ai/flashinfer/pull/1927) 2 nvfp4 quantization APIs: ⏎ `silu_and_mul_scaled_nvfp4_experts_quantize` and `scaled_nvfp4_grouped_quantize` which are equivalent in implementation with sglang's `silu_and_mul_scaled_fp4_grouped_quant` and `scaled_fp4_grouped_quant` respectively. This PR made the replacement for future maintenance. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ `` …[truncated]

### L2-2966367a31  (L2, 2025-11-13, sha 2966367a316e, PR #13087)
TITLE: [sgl-kernel] support custom fp8 flashmla kernel (#13087)
SOURCES: path_core, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: sgl-kernel/cmake/flashmla.cmake (+6/-1); sgl-kernel/csrc/flashmla_extension.cc (+9/-0); sgl-kernel/python/sgl_kernel/flash_mla.py (+43/-14); sgl-kernel/include/sgl_kernel_ops.h (+18/-0); sgl-kernel/tests/test_flashmla.py (+138/-2)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎  ⏎ Add custom fp8 flashmla kernel, which is already used in sglang. But we not support it. ⏎ Stack PR: https://github.com/sgl-project/FlashMLA/pull/1 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ - Add new fp8 kernel ⏎ - Add fp8 kernel test ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e9c0c55833  (L2, 2025-11-13, sha e9c0c5583389, PR #12392)
TITLE: [sgl-kernel] clean up fa fetch in CMakeLists.txt (#12392)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+1/-10)
LABELS: sgl-kernel, run-ci
BODY: ## Motivation ⏎ Unify fa build, cauz sgl-attn has cute folder ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f8d3d80f63  (L2, 2025-11-14, sha f8d3d80f6374, PR #13242)
TITLE: chore: bump flashinfer v0.5.2 (#13242)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎ As mentioned by @yzh119, flashinfer v0.6.0 will be released soon, so let's first upgrade to the latest version v0.5.2. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-a7002e614b  (L2, 2025-11-14, sha a7002e614bbd, PR #13236)
TITLE: [Deepseek V3.2] Clean up MTP (#13236)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+7/-1); python/sglang/srt/layers/attention/nsa_backend.py (+59/-73); test/srt/test_deepseek_v32_mtp.py (+2/-2)
LABELS: deepseek, run-ci
BODY: ## Modifications ⏎  ⏎ - Fix an accuracy bug in MTP draft_extend stage. The `seqlens_int32_cpu` used in the calculations of  `seqlens_expanded` is incorrect. We should use `seq_lens_cpu`, instead of `seqlens_int32_cpu = [self.speculative_num_draft_tokens + kv_len for kv_len in seq_lens_cpu.tolist()]`. ⏎ - Make the draft_extend stage use `_get_topk_paged` instead of `_get_topk_ragged`, which avoids copies of the k cache from the paged to contiguous. M …[truncated]

### L2-5ae0ac4244  (L2, 2025-11-14, sha 5ae0ac424465, PR #13274)
TITLE: [NVIDIA] Fix use case of SGLANG_ENABLE_FLASHINFER_GEMM (#13274)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: docs/references/environment_variables.md (+1/-1); python/sglang/srt/environ.py (+4/-1); python/sglang/srt/layers/quantization/fp8_utils.py (+5/-4); python/sglang/srt/models/deepseek_v2.py (+3/-1)
LABELS: documentation, deepseek, run-ci, format
BODY: ## Motivation ⏎  ⏎ When SGLANG_ENABLE_FLASHINFER_GEMM is enabled, we observed that the accuracy test fails. (by @gracehonv ) ⏎ The root cause is that the model execution switches to use e8m0 scales for Blackwell GPUs even when DeepGEMM is not in use. ⏎ This PR fixes the issue by ensuring that the correct scaling mode is applied. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ script: ⏎ ``` ⏎ set -x ⏎ if [[ "$1" == "server" ]]; then ⏎ model_str=/model/dee …[truncated]

### L2-eae59b337e  (L2, 2025-11-15, sha eae59b337e7c, PR #13045)
TITLE: Piecewise Cuda Graph Support for gpt-oss model (#13045)
SOURCES: path_core
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-3); python/sglang/srt/compilation/piecewise_context_manager.py (+16/-0); python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py (+4/-17); python/sglang/srt/models/deepseek_v2.py (+1/-3); python/sglang/srt/server_args.py (+6/-1)
LABELS: deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Support Piecewise cuda graph for gpt-oss series model. ⏎  ⏎ ## Modifications ⏎  ⏎ - MoE backend Select: With piecewise cuda graph, we can achieve similar performance with auto backend compared with triton backend. ⏎ - Adjust the position of `enable_piecewise_cudagraph` to avoid circular import ⏎  ⏎ ## Accuracy Tests ⏎ In benchmark & profilling section ⏎  ⏎ ## Benchmarking and Profiling ⏎ For gsm 8k test: ⏎ - piecewise cuda graph support with …[truncated]

### L2-1ca205f6da  (L2, 2025-11-15, sha 1ca205f6da0c, PR #13358)
TITLE: chore: bump sgl-kernel version to 0.3.17.post1 (#13358)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.17.post1` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.17.post1` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are up …[truncated]

### L2-d5fa58c4dd  (L2, 2025-11-16, sha d5fa58c4ddf3, PR #13386)
TITLE: fix nightly docker build (#13386)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+5/-3); python/pyproject.toml (+1/-1)
LABELS: dependencies, run-ci
BODY: Keep these aligned

### L2-d368c7451a  (L2, 2025-11-16, sha d368c7451a48, PR #12065)
TITLE: (1/n)support context parallel with deepseekv3.2-DSA (#12065)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: docs/advanced_features/server_arguments.md (+1/-0); docs/basic_usage/deepseek_v32.md (+20/-0); python/sglang/srt/distributed/device_communicators/pynccl.py (+28/-0); python/sglang/srt/distributed/parallel_state.py (+21/-0); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+221/-8); python/sglang/srt/layers/attention/nsa/utils.py (+305/-0); python/sglang/srt/layers/attention/nsa_backend.py (+28/-8); python/sglang/srt/layers/communicator_nsa_cp.py (+284/-0); python/sglang/srt/layers/dp_attention.py (+5/-1); python/sglang/srt/managers/schedule_policy.py (+7/-0); (+7 more)
LABELS: documentation, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Currently, under deepseek3.2-DSA, prefill-ttft of long text sequences takes a long time. Introducing context parallel can reduce ttft. ⏎ **Main design ideas：** ⏎ <img width="599" height="598" alt="image" src="https://github.com/user-attachments/assets/3827c03b-2448-43be-8f95-bfaf76b57a04" /> ⏎  ⏎ Taking TP=EP=4 and DP=2 as an example (CP_SIZE==ATTEN_TP_SIZE):  ⏎ Each DP accepts an independent request.  ⏎ Within each DP, after embedding …[truncated]

### L2-e389f91dec  (L2, 2025-11-17, sha e389f91decda, PR #13264)
TITLE: [NVIDIA] Fix broken fp8 MoE of deepseek v3 (#13264)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: .github/workflows/nightly-test.yml (+22/-0); python/sglang/srt/layers/quantization/fp8.py (+1/-3); python/sglang/srt/models/deepseek_v2.py (+2/-0); test/srt/run_suite.py (+4/-1); test/srt/test_deepseek_r1_fp8_trtllm_backend.py (+88/-0)
LABELS: high priority, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ [This](https://github.com/sgl-project/sglang/pull/12543) breaks the fp8 moe of the deepseek v3, causing: ⏎ ``` ⏎   File "/scratch/repo/sglang/python/sglang/srt/layers/quantization/fp8.py", line 1225, in apply_with_router_logits                                                                                                                          ⏎     return trtllm_fp8_block_scale_moe(                                                …[truncated]

### L2-85ae508e8b  (L2, 2025-11-17, sha 85ae508e8b72, PR #13455)
TITLE: Add bfloat16 tuned fused moe config for Dpsk-MTP layer on B200 (#13455)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=512,device_name=NVIDIA_B200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py (+18/-9)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎ When launching dpsk-r1-fp4 with MTP and TP4, the draft model will use bfloat16 fused moe triton kernels. ⏎ So it requires some tuning. ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ ```bash ⏎ # Launch ⏎ export SGLANG_ENABLE_SPEC_V2=1 ⏎ python3 -m sglang.launch_server \ ⏎     --model-path nvidia/DeepSeek-R1-0528-FP4-v2 \ ⏎     --trust-remote-code \ ⏎     --attention-backend trtllm_mla \ ⏎     --mo …[truncated]

### L2-bfaf0b8607  (L2, 2025-11-19, sha bfaf0b860727, PR #13570)
TITLE: chore: bump sgl-kernel version to 0.3.17.post2 (#13570)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.17.post2` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.17.post2` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are up …[truncated]

### L2-d4a4dcdfb3  (L2, 2025-11-19, sha d4a4dcdfb30f, PR #13567)
TITLE: [NPU] Adapt pr-gate for pr-test workflow & workflows refresh (#13567)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+4/-10); .github/labeler.yml (+7/-0); .github/workflows/pr-test-npu.yml (+47/-25); .github/workflows/release-docker-npu-nightly.yml (+1/-1); .github/workflows/release-docker-npu.yml (+2/-4); test/srt/run_suite.py (+6/-5)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Ever since #12710 introduced auto-labeling, there has been a huge gap that npu-related pull-requests can't be automatically recognized and correctly labled. Now we are filling this gap up with this pr. ⏎  ⏎ Plus, this one also adapts pr-gate mechanism that helps to improve npu ci efficiency. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Following #13436, adapts `pr-gate` on npu ci ⏎ - Adds `npu` auto-labler ⏎ - Fixes an issue that breaks docker i …[truncated]

### L2-c8ede0e93c  (L2, 2025-11-20, sha c8ede0e93c3a, PR #13617)
TITLE: [ROCM] Optimized deepseek-r1 fp8 model with + triton_gemm_a8w8 + batch_gemm_a8w8 + fused set_mla_kv_buffer kernel (#13617)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/layers/quantization/fp8_utils.py (+4/-1); python/sglang/srt/models/deepseek_v2.py (+57/-14)
LABELS: amd, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎ Co-author : @yichiche   ⏎ To optimize the deepseek-r1 model performance on ROCM. ⏎ This PR improves the performance of a8w8 GEMM and enable batched_gemm and mla_kv_buffer feature on MI355x. ⏎  ⏎ - Replace aiter gemm_a8w8_blockscale with the Triton implementation (gemm_a8w8_blockscale_triton). ⏎ Profiling shows Triton’s version provides better latency at concurrency 1–8 and significantly improves elementwise–GEMM fusion opportunities. ⏎ - …[truncated]

### L2-6bc3062894  (L2, 2025-11-20, sha 6bc306289465, PR #13666)
TITLE: Fix launch of `Olmo3` (#13666)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/olmo2.py (+1/-0)
BODY: ## Motivation ⏎  ⏎ Olmo 3 is broken. Applies to non thinking too. ⏎  ⏎ Without the fix ⏎ ``` ⏎ python -m sglang.launch_server --model-path allenai/Olmo-3-32B-Think --mem-fraction-static 0.8 ⏎ [2025-11-20 17:49:37] INFO utils.py:148: Note: detected 248 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable. ⏎ [2025-11-20 17:49:37] INFO utils.py:151: Note: NumExpr detected 248 cores but "NUMEXPR_MAX_THREADS" not se …[truncated]

### L2-6be65ae462  (L2, 2025-11-21, sha 6be65ae46289, PR #13555)
TITLE: Fix target MLA with eagle3 support for PD disaggregation (#13555)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/mooncake/conn.py (+3/-5)
LABELS: run-ci
BODY: When using EAGLE3 for target model with MLA Arch in PD disaggregation, the draft cache has write with wrong format. ⏎ this lead two critical issues: ⏎ 1. accept length reduced from 2.8 to 2.1 in spec args 3 1 4 ⏎ 2. transfer engine may crash with  addresses error ⏎ I1112 18:32:25.549899  3909 transfer_metadata_dump.cpp:51] Failed to get segment descriptor for segment 33.184.124.216:16483 address 0x7f5db26ec000--0x7f5db26fc000 ⏎ I1112 18:32:25.549937   …[truncated]

### L2-589d9ad55b  (L2, 2025-11-21, sha 589d9ad55bd7, PR #13647)
TITLE: [NPU] chore: bump to CANN 8.3.RC1 and Pytorch 2.8.0 (#13647)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+26/-25); .github/workflows/pr-test-npu.yml (+4/-4); .github/workflows/release-docker-npu-nightly.yml (+2/-2); .github/workflows/release-docker-npu.yml (+2/-2); docs/platforms/ascend_npu.md (+7/-24); python/pyproject_other.toml (+1/-0); python/sglang/check_env.py (+7/-1); python/sglang/srt/layers/attention/ascend_backend.py (+1/-1); python/sglang/test/test_utils.py (+2/-2); scripts/ci/npu_ci_install_dependency.sh (+14/-22)
LABELS: documentation, dependencies, npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Bump CANN version and PTA version for NPU backend ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ n/a ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ n/a ⏎  ⏎ ## Checklist

### L2-8bfce9b08d  (L2, 2025-11-22, sha 8bfce9b08dbe, PR #13756)
TITLE: [Tiny] Renaming environ for NVFP4 dispatch (#13756)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/references/environment_variables.md (+1/-1); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+4/-6); test/srt/test_deepseek_v3_cutedsl_4gpu.py (+2/-2)
LABELS: documentation, quant, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Renaming `SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH` to `SGLANG_MOE_NVFP4_DISPATCH`. ⏎ Since cutlass moe backend also can dispatch with nvfp4 precision ⏎ Following #13327  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ Launching with this flag and flashinfer cutlass moe: ⏎ ```bash ⏎ SGLANG_MOE_NVFP4_DISPATCH=1 \ ⏎ python3 -m sglang.launch_server \ ⏎     --model-path nvidia/DeepSeek-R1-0528-FP4-v2 \ ⏎    …[truncated]

### L2-3990b84bd3  (L2, 2025-11-22, sha 3990b84bd36f, PR #13547)
TITLE: Refactor MHA & MLA KV caches to support FP4 (#13547)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/memory_pool.py (+329/-226); python/sglang/srt/model_executor/model_runner.py (+66/-30)
LABELS: run-ci
BODY: ## Motivation ⏎ Per the discussion in #12612 with @Fridge003, this PR refactors the FP4 implementation introduced by #12612 (KV4 MHA) and #10078 (KV4 MLA). ⏎  ⏎ ## Description ⏎ This PR refactors the KV cache implementations for both MHA and MLA layers to better support FP4 (float4_e2m1fn_x2) dtype: ⏎ - Overall, isolates FP4 logic into dedicated classes and improves maintainability and readability of KV cache code. ⏎ - Introduce dedicated subclasses MH …[truncated]

### L2-53fffefd5d  (L2, 2025-11-23, sha 53fffefd5dae, PR #13718)
TITLE: Upgrade flashmla kernel for NSA tp support (#13718)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: sgl-kernel/cmake/flashmla.cmake (+1/-1)
LABELS: sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-618ca23802  (L2, 2025-11-23, sha 618ca2380293, PR #13687)
TITLE: [Deepseek] Refactor deepseek server_args _handle_model_specific_adjustments (#13687)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+79/-74)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Deepseek v3.2 shares the same MoE architecture with other Deepseek MoE models, so that part of the logics can be reused for all `DeepseekV3ForCausalLM` models. ⏎  ⏎ ## Modifications ⏎  ⏎ Refactor the Deepseek part of `_handle_model_specific_adjustments` so that we can share the MoE adjustments across `DeepseekV3ForCausalLM` models. ⏎ `--moe-runner-backend flashinfer_trtllm  --quantization fp8` will be set automatically for DS models o …[truncated]

### L2-18403f6bfe  (L2, 2025-11-23, sha 18403f6bfe65, PR #13802)
TITLE: make trtllm attn backend's init_forward_metadat non blocking (#13802)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+3/-1)
LABELS: blackwell, run-ci
BODY: ## Motivation ⏎  ⏎ This is a minor performance fix. The original implementation caused a CPU-to-GPU copy, which in turn introduced a host–device synchronization point. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-4683e244fe  (L2, 2025-11-23, sha 4683e244fe62, PR #13601)
TITLE: [1/2] Refactor DeepGeem requant for FP8 Linear on Blackwell  (#13601)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/layers/quantization/fp8.py (+29/-0); python/sglang/srt/models/deepseek_nextn.py (+0/-1); python/sglang/srt/models/deepseek_v2.py (+9/-77); python/sglang/test/test_block_fp8_deep_gemm_blackwell.py (+1/-1)
LABELS: high priority, deepseek, run-ci
BODY: ## Motivation ⏎ Co-Author: @fy1214  ⏎ Based on pr: https://github.com/sgl-project/sglang/pull/13067 ⏎  ⏎ Refactor the messy codes related to deepgemm requant, also fix the bug in #12878 ⏎ Refactor for MoE requant will be left to the next PR ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ To run DeepGemm on Blackwell, the input scale factor needs to be requantized to ue8m0. The prior codes only consider this requantization for deepseek model classes, and the codes are quit …[truncated]

### L2-04b52fa8d6  (L2, 2025-11-23, sha 04b52fa8d6f8, PR #13751)
TITLE: [chore]Upgrade flashinfer to 0.5.3 (#13751)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
ISSUES: #13748 [Feature] Upgrade flashinfer to 0.5.3
BODY: ## Motivation ⏎  ⏎ Close #13748 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f56b9b42e6  (L2, 2025-11-24, sha f56b9b42e668, PR #13829)
TITLE: [Bugfix] Add jit kernel files in packaging (#13829)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+3/-0)
LABELS: high priority, dependencies
BODY: ## Motivation ⏎  ⏎  ⏎ PR https://github.com/sgl-project/sglang/pull/13764 is missing jit kernel files when packaging. This PR fix it. ⏎  ⏎ The reason why github ci didn't fail is because there were cache jit kernel files un-deleted in the folder. So the new CI happened to reuse them. ⏎  ⏎ ``` ⏎ $SGLANG_VLM_CACHE_SIZE_MB=2048 python -m sglang.launch_server --model-path /home/admin/Qwen3-VL-2B-Thinking --host 0.0.0.0 --port 8188 --trust-remote-code --tp-si …[truncated]

### L2-b0a26ba624  (L2, 2025-11-24, sha b0a26ba6249f, PR #10275)
TITLE: Add support for bf16 x bf16 cutlass fused MoE (#10275)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-7); python/sglang/srt/layers/moe/utils.py (+4/-0); python/sglang/srt/layers/quantization/unquant.py (+34/-1); python/sglang/srt/server_args.py (+3/-2); python/sglang/test/test_cutlass_w16a16_moe.py (+118/-0)
LABELS: high priority, quant, blackwell, run-ci, nvidia, hopper
BODY: ## Motivation ⏎  ⏎ Add efficient fused MoE layer for bf16 weights and bf16 activations. ⏎ Usage: Add `--moe-runner-backend flashinfer_cutlass` server arg to use the cutlass MoE backend. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ # python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319 ⏎ Accuracy: 0.933 ⏎ Invalid: 0.000 ⏎ Latency: 75.776 s ⏎ Output throughput: 2424.322 token/s ⏎ ``` ⏎  ⏎ ## Benchmarking and Prof …[truncated]

### L2-760c20b360  (L2, 2025-11-25, sha 760c20b36041, PR #13848)
TITLE: update flashinfer_cubin==0.5.3 (#13848)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎ flashinfer_cubin 0.5.3 ready in pypi, ref: https://github.com/flashinfer-ai/flashinfer/issues/2133#issuecomment-3569604925 ⏎ @Fridge003  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-5eed5fc0b0  (L2, 2025-11-25, sha 5eed5fc0b091, PR #13544)
TITLE: [DeepSeekV3.2] Centralize NSA dispatch logic in NativeSparseAttnBackend (#13544)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+69/-42); python/sglang/srt/models/deepseek_v2.py (+5/-36)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ NativeSparseAttnBackend currently spreads dispatch logic for NSA prefill/decode implementations and MHA vs. MLA selection across multiple places: ⏎  ⏎ - Global `NSA_PREFILL_IMPL` / `NSA_DECODE_IMPL` variables that are mutated in `__init__`. ⏎ - MHA vs. MLA decisions partly in `deepseek_v2.handle_attention_nsa()` and partly in the backend. ⏎ - FP8-specific dequantization flags (`using_mha_one_shot_fp8_dequant`) set from the model side …[truncated]

### L2-fcccaf9001  (L2, 2025-11-25, sha fcccaf9001ab, PR #13421)
TITLE: Add Llama4 attention backend auto-selection (#13421)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs/basic_usage/llama4.md (+9/-0); python/sglang/srt/server_args.py (+15/-5)
LABELS: documentation, run-ci, format
BODY: ## Motivation ⏎  ⏎ When using Llama4 models without explicitly specifying the `--attention-backend` parameter, user encounter an `AssertionError` during server initialization: `AssertionError: fa3, aiter, triton, or trtllm_mha is required for Llama4 model` ⏎  ⏎ ## Modifications ⏎  ⏎ **Added platform-aware auto-selection**: Implemented comprehensive auto-selection logic for different platforms: ⏎    - **Blackwell GPUs (SM100)**: `trtllm_mha` ⏎    - **Hopp …[truncated]

### L2-c53e729d45  (L2, 2025-11-25, sha c53e729d4535, PR #13951)
TITLE: chore: bump sgl-kernel version to 0.3.18.post1 (#13951)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.18.post1` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.18.post1` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are up …[truncated]

### L2-5a8adca900  (L2, 2025-11-25, sha 5a8adca90079, PR #13963)
TITLE: Turn off PREBUILD aiter in MI355 (#13963)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docker/rocm.Dockerfile (+1/-1)
LABELS: amd
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ There are some errors in aiter PREBUILD mode, which cause accuracy drop to 0% on DS-MXFP4 model. The errors have been solved but not yet merge into aiter main. We may use jit build to make the docker run correctly for now.  ⏎ - Prebuild kernel dispatching issue: fixed in [fix_fp4_aot_post2](https://github.com/ROCm/aiter/tree/wjx/fix_fp4_aot_post2). ⏎ - MLA issue: fixed in [mla_fix](https://github.com/ROCm/aiter/pull/1445). ⏎  ⏎ # …[truncated]

### L2-18fb51583f  (L2, 2025-11-26, sha 18fb51583f49, PR #7725)
TITLE: Support FlashAttention3 page_size > 1 and topk > 1 case with paged attn and spec decode (#7725)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.dispatch.server_args_defaults
FILES: benchmark/mtbench/bench_sglang_eagle.py (+2/-2); python/sglang/srt/layers/attention/flashattention_backend.py (+133/-23); python/sglang/srt/layers/quantization/kv_cache.py (+1/-2); python/sglang/srt/server_args.py (+1/-1); python/sglang/srt/speculative/eagle_worker.py (+37/-14); python/sglang/srt/speculative/spec_utils.py (+71/-38); python/sglang/test/attention/test_flashattn_backend.py (+70/-1); python/sglang/test/speculative/test_spec_utils.py (+348/-0); test/srt/test_eagle_infer_a.py (+43/-5)
LABELS: high priority, speculative-decoding, run-ci
BODY: ## Motivation ⏎  ⏎ Fully enable flash attention 3 backend with speculative decoding and paged attention use case where topk > 1 and page_size > 1.  ⏎  ⏎ The current implementation is still a fake support for page size > 1. In the `assign_draft_cache_locs`. we directly move the indices instead of the real kv cache. This only works when the kernel backend runs with page size = 1. If the kernel backend runs with page size > 1, we need to duplicate the r …[truncated]

### L2-262c3c1fde  (L2, 2025-11-27, sha 262c3c1fdeec, PR #12491)
TITLE: [Ascend] Support enable-mixed-chunk in non-MLA scenarios (#12491)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/ascend_backend.py (+236/-36); python/sglang/srt/layers/attention/base_attn_backend.py (+24/-0); python/sglang/srt/managers/schedule_policy.py (+1/-1)
LABELS: npu, run-ci
ISSUES: #10091 [Bug] i test qwen3-235b-a22b on 2* 910b，where request concurrency is 7,sglang will be hang up。export STREAMS_PER_DEVICE=32 already set
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ This pull request aims to support the `enable-mixed-chunk` feature in non-MLA scenarios on Ascend NPUs. It merges the current decode requests with upcoming prefill requests into the same batch.  ⏎  ⏎ ## Modifications ⏎  ⏎ We implement a new `forward_mixed` method within `AscendAttnBackend` to specifically handle mixed-chunk attention, leveraging` torch_npu._npu_paged_attention_splitfuse` for optimized NPU operations. ⏎ In addition, we …[truncated]

### L2-7ab548ef64  (L2, 2025-11-27, sha 7ab548ef641b, PR #13960)
TITLE: [2/2] Refactor DeepGeem requant for FP8 FusedMoE on Blackwell (#13960)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/quantization/fp8.py (+27/-4); python/sglang/srt/models/deepseek_v2.py (+0/-35); python/sglang/srt/server_args.py (+5/-0)
LABELS: deepseek, run-ci
ISSUES: #13680 [Bug] Lower gsm8k accuracy in Deepseek V3.2 with moe_backend = deep_gemm
BODY: ## Motivation ⏎  ⏎ Following #13601 Close #13680 ⏎ Currently when `--moe-runner-backend deep_gemm` is applied, the accuracy will drop to 0 for any non-deepseek model. It's caused by the missing requantization process that converts weight/scale to ue8m0 format. This requantization is required for DeepGemm kernels, since DeepGemm takes ue8m0 input datatype. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ All cases are tested on B200 with gsm8k …[truncated]

### L2-63b056213f  (L2, 2025-11-27, sha 63b056213fc4, PR #13946)
TITLE: Remove disused B300 Dockerfile (#13946)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/b300.Dockerfile (+0/-55)
BODY: ## Motivation ⏎  ⏎ I think this B300 Dockerfile can be removed already since it's not being used anymore and just causes confusion like in #13900? ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-848ee57067  (L2, 2025-11-29, sha 848ee57067f5, PR #12306)
TITLE: feat: support flashinfer kernel autotune (#12306)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+3/-1); python/sglang/srt/layers/quantization/mxfp4.py (+3/-1); python/sglang/srt/model_executor/model_runner.py (+303/-2); python/sglang/srt/server_args.py (+7/-0)
LABELS: high priority, run-ci
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎ Flashinfer MoE requires autotune to select a most performant kernel. This should be done before cudagraph captures. ⏎  ⏎ ## Modifications ⏎  ⏎ This PR added a kernel warmup stage before the cudagraph captures. In the warmup stage, it will run a dummy forward for the model under the Flashinfer autotune context. ⏎  ⏎ ## Accuracy Tests ⏎ `lm_eval --model local-completions --tasks gsm8k --model_args model=openai/gpt-oss-120b,base_url=http:/ …[truncated]

### L2-c72f0756d2  (L2, 2025-11-30, sha c72f0756d282, PR #13841)
TITLE: Fix: fix flashmla fp8 kv cache acc error (#13841)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.backend.flashmla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashmla_backend.py (+65/-12)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/issues/13832 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ Fixed. ⏎ <img width="1577" height="235" alt="Image" src="https://github.com/user-attachments/assets/8ef02724-3b6a-4a50-adf6-6417ccbce385" /> ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-decb48965d  (L2, 2025-11-30, sha decb48965dd1, PR #13646)
TITLE: [DeepSeekV3.2] Enable pure TP & Partial DP Attention (#13646)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+95/-14); python/sglang/srt/layers/attention/nsa_backend.py (+72/-7); python/sglang/srt/server_args.py (+6/-1); docs/basic_usage/deepseek_v32.md (+6/-2); test/manual/nightly/test_deepseek_v32_perf.py (+25/-0); test/nightly/test_deepseek_v32_nsabackend.py (+57/-0); test/nightly/test_deepseek_v32_perf.py (+25/-0)
LABELS: documentation, deepseek, sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎ DeepSeekV3.2 NSA currently has rough edges when running in **pure TP mode** (`dp_size < tp_size`): ⏎  ⏎ - FlashMLA sparse can see an invalid `num_heads` per rank after TP sharding. ⏎ - NSA's get_mla_metadata crashes with TP + large bs (Fixed with [Fix FlashMLA Shared-Memory Overflow in SGLang's Pure-TP Mode with Low-SMEM Fallback Scheduler #2](https://github.com/sgl-project/FlashMLA/pull/2)) ⏎ - The NSA fp8 MQA indexer may OOM when b …[truncated]

### L2-03888b9de5  (L2, 2025-12-01, sha 03888b9de5ec, PR #13968)
TITLE: [Minor] Upgrade cutedsl version in Dockerfile (#13968)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+3/-4)
LABELS: dependencies, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-fa9021b21f  (L2, 2025-12-01, sha fa9021b21f92, PR #14173)
TITLE: fix: Increase FlashInfer workspace size for Qwen3VL models (#14173)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_backend.py (+4/-0)
BODY: ```shell ⏎ @sglang  ⏎ ➜  sglang git:(main) ✗ CUDA_VISIBLE_DEVICES=7 python -m sglang.launch_server \      --model Qwen/Qwen3-VL-32B-Instruct-FP8 \                                                 ⏎     --tp 1 \ ⏎     --quantization fp8 \ ⏎     --trust-remote-code   ⏎ /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml direct …[truncated]

### L2-236a7c2370  (L2, 2025-12-01, sha 236a7c237002, PR #13738)
TITLE: fix trtllm mla spec (#13738)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+0/-3)
LABELS: blackwell, run-ci
BODY: In theory  ⏎  ⏎ `test/srt/test_deepseek_v3_fp4_4gpu.py` should pass

### L2-63b9300f00  (L2, 2025-12-01, sha 63b9300f00fe, PR #14244)
TITLE: chore: bump sgl-kernel version to 0.3.18.post2 (#14244)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+10/-16)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.18.post2` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.18.post2` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are up …[truncated]

### L2-d122e32467  (L2, 2025-12-03, sha d122e32467ec, PR #13980)
TITLE: [NPU] bug fix: w_vc need contiguous for NPU batch_matmul_transpose ops (#13980)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+3/-3); python/sglang/srt/models/deepseek_v2.py (+4/-3); test/srt/ascend/test_ascend_deepep.py (+2/-0)
LABELS: deepseek, npu, run-ci
BODY: NPU batch_matmul_transpose ops need w_vc to be contiguous. ⏎  ⏎ ## Motivation ⏎ Bug fix: w_vc need contiguous for NPU batch_matmul_transpose ops ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Before fix bug: ⏎ <img width="3072" height="417" alt="image" src="https://github.com/user-attachments/assets/924536d6-a183-4e0e-832e-8e57b0eeccba" /> ⏎ After fix bug: ⏎ <img width="3072" height="453" alt="image" src="https://github.com/user-attachments/assets …[truncated]

### L2-7dfcc78155  (L2, 2025-12-04, sha 7dfcc78155b6, PR #14325)
TITLE: [DeepseekV3.2][NSA][Indexer] Fix PAGED top-k transform for NSA indexer chunked execution on H200 (#14325)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+26/-5); test/nightly/test_deepseek_v32_nsabackend.py (+0/-58); test/nightly/test_deepseek_v32_tp.py (+169/-0)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ In `extend` mode with NSA indexer enabled, H200 setups that select **PAGED** top-k transform (`flashmla_kv` + FP8 KV cache) may trigger chunked execution in `_get_topk_ragged` when the logits matrix becomes large. ⏎  ⏎ Previously, chunked execution only sliced **logits / lengths / ks**, but continued to use full-batch `page_table_1` and full-batch query-sequence metadata. ⏎ This causes `fast_topk_transform_fused` to receive mismatch …[truncated]

### L2-922756aaa1  (L2, 2025-12-04, sha 922756aaa1c2, PR #14350)
TITLE: [FIX] trtllm-moe-fp4-renorm for Qwen series models (#14350)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-1); python/sglang/srt/server_args.py (+0/-1)
LABELS: run-ci
DEEP_STUDY: deep-study correctness case sglang:922756aaa1: class=numerical_precision; symptom=wrong_output_or_accuracy; introducing=#13761/#14135
BODY: ## Motivation ⏎ **trtllm-moe-fp4-renorm for Qwen series models** ⏎ [PR13761](https://github.com/sgl-project/sglang/pull/13761 ) introduced bug for DeepSeek nvfp4 models. ⏎ [PR14135](https://github.com/sgl-project/sglang/pull/14135) Fixed it, but introduced bug for Qwen3 nvfp4 models. ⏎  ⏎ This PR is to re-fix both of them. After this fix, `test/nightly/test_qwen3_fp4_trtllm_gen_moe.py` could pass. ⏎ Also a PR for flashinfer repo will be created to requ …[truncated]

### L2-894c0dc57c  (L2, 2025-12-04, sha 894c0dc57cfd, PR #13359)
TITLE: [NPU][1/N] NPU basic functions refactor and new modelslim quant type (#13359)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.npu_mla, L2.pool.mla_token_kv, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py (+19/-24); python/sglang/srt/layers/attention/attention_registry.py (+3/-1); .github/CODEOWNERS (+1/-3); python/sglang/srt/environ.py (+3/-0); python/sglang/srt/eplb/expert_distribution.py (+2/-8); python/sglang/srt/hardware_backend/npu/allocator_npu.py (+7/-9); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+3/-1); python/sglang/srt/hardware_backend/npu/cmo.py (+54/-0); python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_extend_npu_graph_runner.py (+0/-0); python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py (+0/-0); (+33 more)
LABELS: lora, deepseek, npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Due to the underlying structural difference between gpgpus and npus, we have introduced a lot of `is_npu` branches in current repository from previous commits. Though literarlly it helps the out-of-box experience for our end-users and matches our rapid development pace, this way of orignizing codes breaks readability and of cource maintainability of the whole sglang project. We believe this is not a long-term solution and a h …[truncated]

### L2-8428078436  (L2, 2025-12-04, sha 842807843671, PR #14213)
TITLE: Add Mistral Large 3 support. (#14213)
SOURCES: path_core
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+11/-0); benchmark/kernels/fused_moe_triton/common_utils.py (+7/-1); python/sglang/srt/configs/model_config.py (+5/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+17/-5); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+81/-1); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py (+127/-63); python/sglang/srt/layers/quantization/fp8.py (+1/-5); python/sglang/srt/layers/quantization/fp8_utils.py (+43/-0); python/sglang/srt/models/deepseek_v2.py (+61/-12); python/sglang/srt/models/mistral_large_3.py (+81/-0); (+6 more)
LABELS: high priority, quant, Multi-modal, deepseek, blackwell, run-ci, vlm, model-gateway
ISSUES: #12751 [Bug] bench_sglang fails due to get_model_info endpoint of SGLang PDRouter not being implemented
BODY: ## Motivation ⏎  ⏎ This PR introduces support for model Mistral Large 3. ⏎  ⏎ ## Modifications ⏎  ⏎ To enable the model, several key modifications were made. ⏎  ⏎ * Two new models are supported: MistralLarge3ForCausalLM and PixtralForConditionalGeneration. ⏎ * The latter incorporates VLM support. ⏎ * Per-tensor scale MOE support was added to `fp8.py`. ⏎ * As ML3 is not yet supported in AutoConfig, a separate code-path for it in `hf_transformers_utils.py` wa …[truncated]

### L2-498ea41ca6  (L2, 2025-12-05, sha 498ea41ca64b, PR #13861)
TITLE: dockerfile: add runtime stage + ubuntu 24.04 (#13861)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+376/-163); .github/workflows/release-docker.yml (+109/-4); docs/get_started/install.md (+17/-4)
LABELS: documentation
BODY: Will merge this in after next release ⏎  ⏎ Multi-stage Dockerfile splits SGLang builds into base, framework, and runtime stages. Runtime cuts image size roughly in half. ⏎  ⏎ --- ⏎  ⏎ ```bash ⏎ sglang                            framework-test     be66a8e51a09   39.3GB ⏎ sglang                            runtime-test          a4dac91fe030    20GB ⏎ ``` ⏎  ⏎ --- ⏎  ⏎ Tests ⏎ 1. cu13 arm - https://github.com/ishandhanani/srt-slurm/blob/main/recipies/gb300-fp4/1p2 …[truncated]

### L2-16e8463a90  (L2, 2025-12-05, sha 16e8463a9096, PR #14460)
TITLE: Add Mistral Large 3 basic test to PR CI (#14460)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+27/-0); test/srt/run_suite.py (+3/-1); test/srt/test_mistral_large3_basic.py (+87/-0)
LABELS: run-ci
BODY: ## Summary ⏎ - Add basic functionality test for `mistralai/Mistral-Large-3-675B-Instruct-2512` to PR CI ⏎ - Test runs on `per-commit-8-gpu-h200` suite ⏎ - Includes GSM8K accuracy check and single batch speed test ⏎  ⏎ ## Configuration ⏎ - TP=8 ⏎ - `--attention-backend trtllm_mla` ⏎ - `--model-loader-extra-config '{"enable_multithread_load": true}'` ⏎ - `--chat-template mistral` ⏎ - `SGLANG_ENABLE_JIT_DEEPGEMM=0` ⏎  ⏎ ## Related PR ⏎ - Nightly perf test: #14459 ⏎  ⏎ ## Test pl …[truncated]

### L2-3d1b591aa1  (L2, 2025-12-05, sha 3d1b591aa15b, PR #14291)
TITLE: Tiny use trtllm_mha as default when possible (#14291)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+5/-1); test/srt/test_flash_attention_4.py (+2/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ wait for ci to see perf regression or errors ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-205f041e96  (L2, 2025-12-05, sha 205f041e9619, PR #14466)
TITLE: Add Mistral Large 3 Eagle Support (#14466)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+2/-1); python/sglang/srt/configs/model_config.py (+11/-1); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py (+6/-10); python/sglang/srt/layers/quantization/fp8.py (+161/-36); python/sglang/srt/models/deepseek_v2.py (+14/-6); python/sglang/srt/models/mistral_large_3.py (+0/-3); python/sglang/srt/models/mistral_large_3_eagle.py (+105/-0); python/sglang/srt/server_args.py (+7/-3); python/sglang/srt/utils/mistral_utils.py (+7/-2)
LABELS: deepseek, blackwell, run-ci
BODY: ## Motivation ⏎ Support Mistral Large 3 Eagle. The eagle checkpoint `mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle` is using the FP8 per-tensor quantization while the standard FP8/NVFP4 checkpoint `mistralai/Mistral-Large-3-675B-Instruct-2512[-NVFP4]` is using compressed tensors quantization. In this PR, we support ⏎ - the functionality of FP8 + eagle for the Mistral Large 3 model ⏎ - Flashinfer TRTLLM FP8 per-tensor quant MoE, this can be used …[truncated]

### L2-662809874c  (L2, 2025-12-05, sha 662809874ceb, PR #14459)
TITLE: Add Mistral Large 3 to nightly CI tests (#14459)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/nightly-test-nvidia.yml (+21/-0); python/sglang/srt/model_loader/weight_utils.py (+97/-82); test/nightly/test_mistral_large3_perf.py (+105/-0)
LABELS: documentation
BODY: ## Summary ⏎ - Add nightly performance test for `mistralai/Mistral-Large-3-675B-Instruct-2512` ⏎ - Test configuration: ⏎   - TP=8 ⏎   - `--attention-backend trtllm_mla` ⏎   - `--model-loader-extra-config '{"enable_multithread_load": true}'` ⏎   - `--chat-template mistral` ⏎   - `SGLANG_ENABLE_JIT_DEEPGEMM=0` ⏎  ⏎ ## Test plan

### L2-ea177372bd  (L2, 2025-12-06, sha ea177372bd8c, PR #13115)
TITLE: support mtp with deepseek r1 nvfp4 model (#13115)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+17/-32); docs/advanced_features/server_arguments.md (+3/-2); python/sglang/srt/layers/moe/utils.py (+32/-0); python/sglang/srt/model_executor/forward_batch_info.py (+6/-1); python/sglang/srt/models/deepseek_v2.py (+3/-1); python/sglang/srt/server_args.py (+11/-1); python/sglang/srt/speculative/eagle_info.py (+3/-0); python/sglang/srt/speculative/eagle_worker.py (+12/-6); python/sglang/srt/speculative/eagle_worker_v2.py (+24/-11); python/sglang/srt/speculative/standalone_worker.py (+6/-3); (+1 more)
LABELS: documentation, high priority, deepseek, speculative-decoding, blackwell, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: collabrate with @trevor-m  ⏎  ⏎ ## Motivation ⏎  ⏎ Support large scale EP deployment for the DS R1 fp4 model with eagle spec decoding.  ⏎  ⏎ ## Modifications ⏎  ⏎ - add the custom moe a2a backend for speculative decoding ⏎ - fix the request padding during the forward batch ⏎ - remove the request padding from trtllm-mla attn backend when draft_extend/verify ⏎  ⏎ ## Test Scripts ⏎  ⏎ **Prefill** ⏎  ⏎ ```bash ⏎ TORCH_CUDA_ARCH_LIST=10.0  NVSHMEM_IB_ENABLE_IBGDA=0 NV …[truncated]

### L2-d2b42477c7  (L2, 2025-12-06, sha d2b42477c788, PR #14518)
TITLE: chore: bump sgl-kernel version to 0.3.18.post3 (#14518)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.18.post3` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.18.post3` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are up …[truncated]

### L2-3c7886ec4c  (L2, 2025-12-06, sha 3c7886ec4cbe, PR #14560)
TITLE: Fix attention backend logic for Qwen3-Next on SM100 (#14560)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+15/-0)
LABELS: run-ci
BODY: ## Motivation ⏎ After https://github.com/sgl-project/sglang/pull/14291 set trtllm_mha as the default attention backend, but mamba cache requires `page_size = 1`, while `trtllm_mha` only supports `page_size > 1.` This can lead to issues  https://github.com/sgl-project/sglang/issues/14527 ⏎  ⏎ This PR prevents such conflicts. ⏎  ⏎ ## Modifications ⏎ 1. use `triton` backend by default for Qwen3-Next on SM100. ⏎ 2. disable radix-cache when `trtllm_mha` is s …[truncated]

### L2-125e17efd5  (L2, 2025-12-07, sha 125e17efd547, PR #14576)
TITLE: Add small model test for spec v2 + dp + trtllm_mla (#14576)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/forward_batch_info.py (+2/-3); test/manual/test_eagle_infer_beta_dp_attention.py (+96/-39); test/srt/ep/test_deepep_large.py (+4/-0)
LABELS: deepseek, run-ci
BODY: Also, add this small model, which is for future `per-commit` tests, and the large model will be moved to nightly tests. ⏎  ⏎ Also, this tiny model is better for debugging. ⏎  ⏎ cc @Fridge003 @rainj-me @alisonshao  ⏎  ⏎ related #14551

### L2-f72a77038f  (L2, 2025-12-08, sha f72a77038f7f, PR #14625)
TITLE: modify the sgl-kernel to be compatible with transformers 5.x. (#14625)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+35/-4); scripts/ci/ci_install_dependency.sh (+21/-0); sgl-kernel/python/sgl_kernel/_fa4_interface.py (+3/-3)
LABELS: sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-948b6acee8  (L2, 2025-12-08, sha 948b6acee802, PR #13573)
TITLE: [BugFix] fix prefixcache performance and accuracy on ascend (#13573)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+131/-6); python/sglang/srt/managers/cache_controller.py (+1/-1); python/sglang/srt/mem_cache/memory_pool_host.py (+54/-21); test/srt/ascend/test_ascend_hicache_mla.py (+102/-0); test/srt/run_suite.py (+1/-0)
LABELS: deepseek, hicache, npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ We have fixed some performance and accuracy problems in the MLA model on Ascend. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ For performance problems: ⏎  ⏎ 1. We moved **device_indices** to the CPU in advance in *cache_controller.py*. ⏎ 2. In *memory_pool_host.py*, we added a new variable **NEED_HOST_REGISTER** and **self.host_register** to indicate whether **cudaHostRegister** is required for pinning memory. If it is not needed, tensors are all …[truncated]

### L2-36361adcbf  (L2, 2025-12-08, sha 36361adcbf52, PR #14203)
TITLE: [DLLM] Add initial cuda graph support (#14203)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_backend.py (+45/-1); python/sglang/srt/layers/logits_processor.py (+8/-0); python/sglang/srt/managers/schedule_batch.py (+3/-1); python/sglang/srt/model_executor/cuda_graph_runner.py (+17/-1); python/sglang/srt/model_executor/forward_batch_info.py (+10/-1); python/sglang/srt/server_args.py (+10/-4)
LABELS: blackwell, npu, run-ci
BODY: ## Motivation ⏎  ⏎ Add cuda graph support for diffusion LLMs. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add a `DLLM_EXTEND` forward mode for dllm. ⏎ - Add cuda graph support for the `DLLM_EXTEND` forward mode. ⏎ - Extend flashinfer backend to support cuda graph for the `DLLM_EXTEND` forward mode. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-c106b54b57  (L2, 2025-12-08, sha c106b54b57d8, PR #13147)
TITLE: Aiter fp8 kv cache (#13147)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.aiter_mla
FILES: Makefile (+0/-49); python/sglang/srt/layers/attention/aiter_backend.py (+564/-41); python/sglang/srt/layers/quantization/fp8.py (+3/-0); python/sglang/srt/layers/quantization/quark/quark.py (+2/-0); python/sglang/srt/layers/rocm_linear_utils.py (+2/-1); python/sglang/srt/model_executor/model_runner.py (+4/-3); python/sglang/srt/models/deepseek_v2.py (+19/-2)
LABELS: amd, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ Support fp8 kv cache in aiter-backend of AMD. ⏎  ⏎ Aiter backend only support mla decode fp8 computation. ⏎  ⏎ Other attention function still do bf16 computation ⏎  ⏎ ## Modifications ⏎  ⏎ Aiter backend and model runner. ⏎  ⏎ ## Next actions ⏎  ⏎ 1. Support other attention function also do fp8 computation ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ `sglang# python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --port 8000 ⏎ Downloading fro …[truncated]

### L2-08da4c2618  (L2, 2025-12-08, sha 08da4c261888, PR #14627)
TITLE: [Bugfix] Fix KeyError for Mistral-Large-3 rope_scaling config (#14627)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+4/-3)
BODY: ## Summary ⏎ - Fix `KeyError: 'factor'` when loading Mistral-Large-3 model ⏎ - Use `.get()` to safely access `rope_scaling["factor"]` which may not exist in Mistral-Large-3's config structure ⏎  ⏎ ## Root Cause ⏎ Mistral-Large-3's HuggingFace config uses a different `rope_scaling` structure (based on `rope_parameters`) that doesn't have a top-level `"factor"` key. The code at `model_config.py:361` assumed all MLA models with `rope_scaling` have this key. ⏎  ⏎  …[truncated]

### L2-0e0b0c0566  (L2, 2025-12-08, sha 0e0b0c0566fe, PR #14676)
TITLE: Revert "[Bug] fix not desired disable fused share experts caused by r… (#14676)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+2/-4)
LABELS: high priority, deepseek, run-ci
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 14432 reason=correctness_or_accuracy
BODY: …ocm logic (#14432)" ⏎  ⏎ This reverts commit 2ecee7571cdfe10a124ae313fe020166ad5c56b8. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ Recently — especially over the past week — the stability of SGLang OSS has declined significantly. Some very basic accuracy guarantees have not been upheld. The main reason is CI flakiness: many CI jobs, especially those on B200, either didn’t run or were merged before completing, which led to several issues that should have been easily c …[truncated]

### L2-2de98010b5  (L2, 2025-12-08, sha 2de98010b5ad, PR #14649)
TITLE: chore: bump sgl-kernel version to 0.3.19 (#14649)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.19` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.19` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are updated accord …[truncated]

### L2-f832994c32  (L2, 2025-12-11, sha f832994c328b, PR #14525)
TITLE: [CI] Add Mistral Large 3 Eagle nightly performance test (#14525)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/nightly-test-nvidia.yml (+2/-0); test/nightly/test_mistral_large3_perf.py (+99/-0)
BODY: ## Summary ⏎ - Add nightly CI test for `mistralai/Mistral-Large-3-675B-Instruct-2512` with Eagle speculative decoding ⏎ - Eagle test is integrated into the existing `test_mistral_large3_perf.py` file as a separate test class ⏎ - Test includes benchmark and MGSM accuracy evaluation ⏎ - Runs on 8-gpu-b200 in the nightly test workflow ⏎  ⏎ ## Eagle-specific configuration ⏎ - `--speculative-algorithm EAGLE` ⏎ - `--speculative-draft-model-path mistralai/Mistral-Large …[truncated]

### L2-10146af099  (L2, 2025-12-11, sha 10146af099f7, PR #14467)
TITLE: Check KV4 compatibility with attention backends and add KV4 support to the attention_backend doc (#14467)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs/advanced_features/attention_backend.md (+29/-25); python/sglang/srt/server_args.py (+79/-0)
LABELS: documentation, run-ci
BODY: ## Motivation ⏎  ⏎ Prevent users from using KV4 with incompatible attention backends by clearly documenting supported backends and enforcing runtime checks. ⏎  ⏎ Improves reliability and reduces runtime errors by ensuring users cannot accidentally run KV4 with unsupported attention backends. ⏎  ⏎ ## Modifications ⏎  ⏎ Ensure code executes after default settings are set by placing it in server_args.py instead of server_args.py. ⏎  ⏎ Description / Changes: ⏎  …[truncated]

### L2-171b442ad3  (L2, 2025-12-12, sha 171b442ad3ac, PR #14989)
TITLE: Add KV4-capable backend flashmla and update server args (#14989)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+1/-0); docs/advanced_features/attention_backend.md (+1/-1)
LABELS: documentation, run-ci
BODY: ## Motivation ⏎ #14467 only tested on SM100 and thus didn't test flashmla and fa3. This PR adds the attention backend, flashmla, that can run with FP4 KV cache on SM90. ⏎  ⏎ - Added the new KV4-capable backend flashmla and documented it in attention_backend.md ⏎ - Updated server_args.py accordingly ⏎  ⏎ # Experiments ⏎ Tested flashmla and fa3 on H20 with DeepSeek-R1-W4AFP8 (MLA) and Qwen3-235B-A22B (MHA). ⏎  ⏎  ⏎ ## Checklist

### L2-0e7d7969d5  (L2, 2025-12-13, sha 0e7d7969d50d, PR #15027)
TITLE: [PP Prefill][NIXL] Fix PP mode transfer completion tracking to wait for all ranks (#15027)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/nixl/conn.py (+40/-16)
LABELS: high priority, run-ci
BODY: Collaborated with @hlu1 on root cause analysis and fix ⏎  ⏎ ## Motivation ⏎  ⏎ Fix NIXL PP mode correctness bug: decode server prematurely considers KV transfer "complete" after receiving chunks from only one PP rank (instead of all ranks), causing accuracy drop. ⏎  ⏎ **Root cause:** `TransferStatus` used `Set[int]` for chunk IDs without distinguishing PP ranks. Overlapping chunk IDs (0,1,2...) from different PP ranks got deduplicated. ⏎  ⏎ ## Modificati …[truncated]

### L2-f6031adf08  (L2, 2025-12-13, sha f6031adf0875, PR #14485)
TITLE: Mistral Large 3 NVFP4 support (#14485)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+42/-1); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+245/-2); python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py (+2/-0); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py (+168/-0); python/sglang/srt/layers/quantization/compressed_tensors/utils.py (+1/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+6/-30); python/sglang/srt/layers/quantization/utils.py (+29/-0); python/sglang/srt/models/deepseek_v2.py (+4/-0); python/sglang/srt/models/mistral_large_3.py (+1/-1); python/sglang/srt/models/mistral_large_3_eagle.py (+2/-0); (+1 more)
LABELS: quant, deepseek, blackwell, run-ci
BODY: Support Mistral Large 3 NVFP4. ⏎  ⏎ Depends on https://github.com/sgl-project/sglang/pull/14466. ⏎  ⏎ * GSM8K test results: ⏎  ⏎ ``` ⏎ SGLANG_ENABLE_JIT_DEEPGEMM=0 \ ⏎ python3 -m sglang.launch_server \ ⏎ --model mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4 \ ⏎ --kv-cache-dtype fp8_e4m3 \ ⏎ --tensor-parallel-size 8 \ ⏎ --disable-radix-cache \ ⏎ --stream-interval 20 \ ⏎ --mem-fraction-static 0.9 \ ⏎ --attention-backend trtllm_mla \ ⏎ --model-loader-extra-con …[truncated]

### L2-d36299ad77  (L2, 2025-12-13, sha d36299ad774f, PR #14423)
TITLE: [NPU] perf update with kvcache nz & w4a8 quant (#14423)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.npu_mla
FILES: python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py (+33/-5); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+56/-22); python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py (+38/-23); python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+56/-47); python/sglang/srt/layers/rotary_embedding.py (+38/-8); python/sglang/srt/model_executor/forward_batch_info.py (+0/-5)
LABELS: deepseek, npu, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ 1、Use the nz format for kv cache, thie method accelerates the FIA operator.; ⏎ 2、Moe's w4a8 uses per-channel quantization; ⏎ 3、Accelerating preprocessing of MHA in prefill using the npu_interleave_rope operator； ⏎ 4、bugfix num_token_non_padded_cpu; ⏎  ⏎ ## Modifications ⏎  ⏎ Use `export SGLANG_USE_FIA_NZ=1` to enable FIA NZ, and this feature must be turned on together with mlapo `export SGLANG_NPU_USE_MLAPO=1`. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ < …[truncated]

### L2-3b8a824b8b  (L2, 2025-12-13, sha 3b8a824b8b2e, PR #14422)
TITLE: [VLM] Support VLM ViT Piecewise CUDA Graph (#14422)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/vision.py (+73/-30); python/sglang/srt/models/qwen2_5_vl.py (+83/-1); python/sglang/srt/multimodal/vit_cuda_graph_runner.py (+263/-0); test/manual/nightly/test_vlms_vit_cuda_graph.py (+271/-0)
LABELS: performance, Multi-modal, run-ci, vlm, piecewise-cuda-graph
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎  ⏎ Previously VLM supports Piecewise CUDA Graph in LLM. ⏎ But ViT was not included inside due to some Tensors addresses in ViT are not fixed. This PR is to support Piecewise CUDA Graph for ViT. Currently Triton Attention and FA3 as MM-Attention are supported. Qwen2.5-VL is supported as the first target VLM model. ⏎  ⏎ Co-author: @kousakawang  ⏎  ⏎ Notes: ⏎ TP>1 is supported now due to custom all-reduce is disabled by default. ⏎  ⏎ Before  …[truncated]

### L2-3f0482174a  (L2, 2025-12-14, sha 3f0482174aa4, PR #15117)
TITLE: Fix Mamba2-based models' default attention backend (#15117)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+36/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Currently, when running on Blackwell GPUs, Mamba2-based models require either disabling radix cache, or specifying an attention backend, as the default attention backend uses a page size different from 1, which the Mamba radix cache requires. ⏎  ⏎ This PR fixes it so that these models work without extra arguments. It also fixes a bug where NemotronH models were running with overlap scheduler enabled when radix cache is enabled, w …[truncated]

### L2-2ea844ec81  (L2, 2025-12-15, sha 2ea844ec81f4, PR #14862)
TITLE: Fused two elementwise kernels for k_nope and k_pe concat (#14862)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+3/-0)
LABELS: deepseek, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Reduce time cost in concat k_nope and k_pe before doing MHA attention ⏎  ⏎ ## Modifications ⏎  ⏎ Use the triton kernel to replace the naive torch operations ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ root@mia1-p01-g07:/sgl-workspace/sglang# python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --port 8000 ⏎ 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████ …[truncated]

### L2-30da2f0598  (L2, 2025-12-16, sha 30da2f0598e8, PR #14820)
TITLE: [NPU][eagle3] support qwen eagle3 on NPU (#14820)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs/platforms/ascend_npu_qwen3_examples.md (+29/-0); python/sglang/srt/configs/model_config.py (+8/-1); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+152/-91); python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_extend_npu_graph_runner.py (+3/-0); python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py (+36/-3); python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py (+16/-6); python/sglang/srt/hardware_backend/npu/memory_pool_npu.py (+1/-1); python/sglang/srt/server_args.py (+19/-2); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+5/-1); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+6/-1); (+1 more)
LABELS: documentation, npu, run-ci
BODY: ## Motivation ⏎  ⏎ Enable sglang eagle3 on NPU platform ⏎  ⏎ Tested models: ⏎ Qwen3-32B-Int8 ⏎  ⏎ ## Modifications ⏎  ⏎ 1、Add a MHA attn op in forward_mtp to support non-MLA model. ⏎ 2、Modify eagle_draft_npu_graph_runner.py to support speculative-num-steps > 1 in eagle3 scenario ⏎ 3、A parameter '--speculative-draft-model-quantization' has been added to handle cases where the target and draft models use different quantization method. ⏎  ⏎ Rules for `speculativ …[truncated]

### L2-99401e7b1a  (L2, 2025-12-16, sha 99401e7b1ab0, PR #14936)
TITLE: Fix accuracy issue when using a16w16 mla_decode_fwd (#14936)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+9/-2)
LABELS: amd, run-ci
BODY: ## Motivation ⏎ Current persist mla_decode_fwd kernel does not support nhead=128 a16w16. ⏎ So it caused the accuracy problem in the PR#13147. ⏎  ⏎ We need to fall back to non-persist mla_decode_fwd kernel ⏎  ⏎ ## Modifications ⏎ aiter_backend.py to check some condition to fall back to the non-persist mla_decode_fwd kernel ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ **Before this PR** ⏎ Accuracy: 0.009 ⏎ Invalid: 0.058 ⏎ Latency: 133.917 s ⏎  ⏎ **this PR** ⏎ Accuracy: 0.935 ⏎ In …[truncated]

### L2-0261c4aff7  (L2, 2025-12-16, sha 0261c4aff784, PR #14857)
TITLE: [misc] Upgrade cutedsl to 4.3.1 (#14857)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+0/-3); python/pyproject.toml (+1/-1)
LABELS: dependencies, deepseek, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 15293 (confirmed_revert, reason=build_or_dependency)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-930705863f  (L2, 2025-12-16, sha 930705863f2d, PR #15242)
TITLE: [sgl-kernel] Update flashmla to include fp8 sparse_mla optimizations (#15242)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts(reverted), corpus:performance-pr-population
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: sgl-kernel/cmake/flashmla.cmake (+1/-1)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 16678 (confirmed_revert, reason=correctness_or_accuracy) || deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/issues/15211 ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ gsm8k 20shots: ⏎  ⏎ Accuracy: 0.955 ⏎ Invalid: 0.000 ⏎ Latency: 175.612 s ⏎ Output throughput: 727.763 token/s ⏎  ⏎ ## Benchmarking and Profiling ⏎ ``` ⏎ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2 --tp 8 --dp 8 --enable-dp-attention --disable-radix-cache ⏎  ⏎ python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 51 …[truncated]

### L2-0861dca81f  (L2, 2025-12-16, sha 0861dca81faf, PR #15293)
TITLE: Revert "[misc] Upgrade cutedsl to 4.3.1 (#14857)" (#15293)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+3/-0); python/pyproject.toml (+1/-1)
LABELS: dependencies
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 14857 reason=build_or_dependency
BODY: This reverts commit 0261c4aff784389d9d50a0ae00cbeb8d4570981f. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ pytest [sgl-kernel/tests/test_flash_attention_4.py](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/tests/test_flash_attention_4.py) ⏎  ⏎ ``` ⏎ cutlass.base_dsl.common.DSLCudaRuntimeError: DSLCudaRuntimeError: (<cudaError_t.cudaSuccess: 0>, b'cudaErrorInsufficientDriver') (error code: 35) ⏎ ``` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Be …[truncated]

### L2-435d1c83c1  (L2, 2025-12-16, sha 435d1c83c1f6, PR #14357)
TITLE: [Perf] Enable Flashinfer autotune by default (#14357)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs/advanced_features/server_arguments.md (+1/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-0); python/sglang/srt/layers/quantization/fp8.py (+3/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-0); python/sglang/srt/model_executor/model_runner.py (+7/-3); python/sglang/srt/server_args.py (+4/-4); test/srt/test_deepseek_v3_fp4_4gpu.py (+1/-1)
LABELS: documentation, quant, deepseek, run-ci
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎ This PR enable Flashinfer autotune by default to achieve possible perf gain. ⏎ Following #12306. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-888594333e  (L2, 2025-12-17, sha 888594333e60, PR #15233)
TITLE: Fix gpu-fault when running mtp in eager mode (#15233)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+74/-30); python/sglang/srt/layers/attention/utils.py (+98/-0)
LABELS: amd, run-ci
BODY: ## Motivation ⏎  ⏎ Fix gpu fault when running request number is over the maximum capture batch size ⏎  ⏎ ## Modifications ⏎  ⏎ The accepted length is not a fix size. Add padding for Q tensor to avoid the memory access violation error ⏎  ⏎ ## Accuracy Tests ⏎ `Accuracy: 0.945 ⏎ Invalid: 0.000 ⏎ Latency: 116.504 s ⏎ Output throughput: 1136.755 token/s` ⏎  ⏎ ## Checklist

### L2-e0026f7c92  (L2, 2025-12-18, sha e0026f7c92c9, PR #14781)
TITLE: [Performance] optimize NSA backend metadata computation for multi-step speculative decoding (#14781)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py (+324/-0); python/sglang/srt/layers/attention/nsa/utils.py (+5/-0); python/sglang/srt/layers/attention/nsa_backend.py (+111/-16)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ This PR optimizes metadata initialization in the NSA backend for multi-step speculative decoding. ⏎  ⏎ The key idea is to replace repeated per-backend metadata computation with a **precompute-once-copy-many** strategy. Shared metadata is computed once and then copied/reused across backend instances, reducing decode-path overhead without changing model computation or speculative decoding behavior. ⏎  ⏎ ## Approach ⏎  ⏎ The old path compute …[truncated]

### L2-793c96c3d2  (L2, 2025-12-18, sha 793c96c3d269, PR #12921)
TITLE: [perf]optimize w4afp8 kernel on deepseek-v3-0324 (#12921)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/w4afp8.py (+0/-1); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu (+64/-236); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu (+96/-27)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ we use w4afp8 deepseekv3-0324 online, and we find its performance is not good enough when decode batch size < 32 ⏎  ⏎ ## Modifications ⏎  ⏎ fine-grained tiling config ⏎ and based on https://github.com/sgl-project/sglang/pull/10027/files ⏎ I use cuda-int4 memory access to decrease memory-access pressure ⏎  ⏎ ## Accuracy Tests ⏎ deepseek-v3-0324 w4afp8 ⏎ ``` ⏎ aime25          0.4/0.4 ⏎ aime24          0.5/0.65/0.55 ⏎ mmlu              0.8947 ⏎ ` …[truncated]

### L2-6559e43f30  (L2, 2025-12-19, sha 6559e43f3068, PR #14395)
TITLE: Support FP8 MLA prefill and 128k context. (#14395)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+78/-34)
LABELS: blackwell, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎ MLA prefill always use 16-bit despite set `kv-cache-dtype` to fp8. We have fp8 kernel now, so this fixed it. ⏎  ⏎  ⏎ ## Modifications ⏎ - convert qkv to fp8 to make attention faster. ⏎ - increase workspace size to make 128k runable. ⏎ - ~~remove `tile_tokens_dim` to be compatible with newer flashinfer.~~ ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|--------------- …[truncated]

### L2-9e0ef04e5b  (L2, 2025-12-19, sha 9e0ef04e5bb2, PR #14843)
TITLE: Support using different attention backend for draft decoding. (#14843)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/model_executor/model_runner.py (+10/-0); python/sglang/srt/server_args.py (+7/-0); python/sglang/srt/speculative/draft_utils.py (+6/-1)
LABELS: run-ci
BODY: ## Motivation + Modifications ⏎  ⏎ Support using different attention backend for drafter decoding by setting SGLANG_DRAFT_ATTN_BACKEND or --speculative-draft-attention-backend. ⏎ e.g. Use trtllm_mla for main deepseek target model and use trtllm_mha or flashinfer for eagle heads. In local test, using trtllm_mla + trtllm_mha for drafting can provide up to 15% better TPOT than using flashinfer only. ⏎  ⏎ Also allow DeepseekV3 model to use EAGLE3. ⏎  ⏎ ## T …[truncated]

### L2-9a3bdf2c95  (L2, 2025-12-20, sha 9a3bdf2c9516, PR #15436)
TITLE: [CI] Migrate CUDA Graph tests to test/registered/cuda_graph/ (#15436)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+42/-4); scripts/ci/slash_command_handler.py (+2/-1); test/registered/cuda_graph/test_piecewise_cuda_graph_2_gpu.py (+4/-0); test/registered/cuda_graph/test_piecewise_cuda_graph_large_1_gpu.py (+42/-85); test/registered/cuda_graph/test_piecewise_cuda_graph_small_1_gpu.py (+96/-45); test/registered/lora/test_lora_tp.py (+1/-1); test/run_suite.py (+2/-1); test/srt/run_suite.py (+0/-3)
LABELS: documentation, lora, run-ci
BODY: ## Summary ⏎ Part of #13808 ⏎  ⏎ - Migrate piecewise CUDA graph tests from `test/srt/` to `test/registered/cuda_graph/` ⏎ - Split tests by GPU memory requirements (small 24GB vs large 80GB) ⏎ - Add `stage-b-test-large-1-gpu` suite and workflow job ⏎ - Rename `stage-b-test-2-gpu` to `stage-b-test-large-2-gpu` for consistency ⏎ - Fix `test_lora_tp.py` suite name (was using invalid `stage-b-test-small-2-gpu`) ⏎  ⏎ ## Test Organization ⏎  ⏎ | File | GPU Requirement | Est. …[truncated]

### L2-8fe3e37468  (L2, 2025-12-21, sha 8fe3e3746832, PR #15531)
TITLE: Support piecewise cuda graph for dsv3 fp4 (#15531)
SOURCES: path_core
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+3/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+64/-10); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py (+1/-2); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+11/-1); test/srt/run_suite.py (+1/-1); test/srt/test_deepseek_v3_fp4_4gpu.py (+67/-0)
LABELS: quant, deepseek, blackwell, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/issues/11490 ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model nvidia/DeepSeek-R1-0528-FP4-v2 --tp 8 --trust-remote --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' --enable-piecewise-cuda-graph --quantization modelopt_fp4 ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 1400 --parallel 1400 ⏎  ⏎ Accuracy: 0.947 ⏎ Invalid: 0.000 ⏎ Late …[truncated]

### L2-34013d9d5a  (L2, 2025-12-22, sha 34013d9d5a59, PR #15590)
TITLE: chore: bump sgl-kernel version to 0.3.20 (#15590)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.20` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.20` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are updated accord …[truncated]

### L2-72a980c6d5  (L2, 2025-12-24, sha 72a980c6d566, PR #15798)
TITLE: ci: migrate MLA tests to test/registered/mla/ (#15798)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+2/-2); test/registered/mla/test_flashmla.py (+5/-1); test/registered/mla/test_mla.py (+9/-0); test/registered/mla/test_mla_deepseek_v3.py (+9/-0); test/registered/mla/test_mla_flashinfer.py (+4/-0); test/registered/mla/test_mla_fp8.py (+4/-0); test/registered/mla/test_mla_int8_deepseek_v3.py (+4/-0); test/srt/run_suite.py (+0/-9)
LABELS: deepseek
BODY: ## Summary ⏎ - Move 6 MLA test files from `test/srt/` to `test/registered/mla/` with CI registry decorators ⏎ - Remove MLA test entries from legacy `test/srt/run_suite.py` ⏎  ⏎ ## Test Files Migrated ⏎  ⏎ | Test File | CUDA Suite | AMD Suite | ⏎ |-----------|------------|-----------| ⏎ | test_mla.py | stage-b-test-small-1-gpu (194s) | stage-a-test-1 (disabled #13107) | ⏎ | test_mla_deepseek_v3.py | stage-b-test-small-1-gpu (442s) | stage-a-test-1 (disabled #12574) …[truncated]

### L2-9d878c1f3e  (L2, 2025-12-25, sha 9d878c1f3e1e, PR #15522)
TITLE: Optimize FP8 MLA KV cache writes with Triton kernel (#15522)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.pool.mla_token_kv
FILES: python/sglang/srt/layers/attention/nsa/quant_k_cache.py (+199/-1); python/sglang/srt/mem_cache/memory_pool.py (+20/-7); python/sglang/srt/mem_cache/utils.py (+22/-1)
LABELS: quant, run-ci
ISSUES: #15104 [Perf][DeepSeekV3.2][Decode] Reduce small-kernel overhead in _get_topk_paged
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: # Updated Summary ⏎  ⏎   Optimize FP8 MLA KV cache writes in NSA decode by: ⏎   - skipping `torch.cat([k_nope, k_rope])` in the FP8 path, and ⏎   - reusing the existing `set_mla_kv_buffer_triton` two-tensor write kernel. ⏎  ⏎   **No new Triton write kernel is added in the final version.** ⏎  ⏎   ## Motivation ⏎  ⏎   In DeepSeek-V3.2 decode, each step writes FP8 MLA KV cache per layer. The baseline path incurs noticeable CPU-side overhead when using PyTorch …[truncated]

### L2-cb1812954a  (L2, 2025-12-26, sha cb1812954af4, PR #15821)
TITLE: Introduce `ModelRunnerKVCacheMixin` to simplify the code. (#15821)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/model_runner.py (+7/-646); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+663/-0)
LABELS: run-ci
BODY: As the title describes. Preparation for future decouple of memory initialization from other model runner logic.

### L2-656f4d69a1  (L2, 2025-12-28, sha 656f4d69a1bc, PR #15353)
TITLE: Refactor fp8 nextn layer for DeepSeek nvfp4 checkpoint (#15353)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: docs/references/environment_variables.md (+1/-0); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/layers/quantization/fp8.py (+6/-0); python/sglang/srt/models/deepseek_nextn.py (+3/-3); python/sglang/srt/models/deepseek_v2.py (+83/-109); python/sglang/srt/server_args.py (+21/-0)
LABELS: documentation, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ In DeepSeek-nvfp4 checkpoint, the moe weights of nextn layer is stored in bf16 precision. We have some logics that quantize nextn moe layer to fp8 optionally, but the codes are a little bit messy. ⏎  ⏎ This PR refactors this part of codes. With flag `SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE` enabled, dpsk fp4 can be launched with fp8 MTP correctly. ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ### B200 ⏎  ⏎ Launch: ⏎ ``` ⏎ # First we need to apply https://github …[truncated]

### L2-7380ec9d55  (L2, 2025-12-29, sha 7380ec9d5572, PR #16096)
TITLE: [CI] fix test_mla_deepseek_v3.py (#16096)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+1/-1)
BODY: ## Motivation ⏎ test_mla_deepseek_v3.py was moved from 'test/srt/test_mla_deepseek_v3.py' to 'test/registered/mla/test_mla_deepseek_v3.py', so update its path in the CI workflow. ⏎ ![Uploading image.png…]() ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e47afa0237  (L2, 2025-12-31, sha e47afa023775, PR #16178)
TITLE: [DP]Fix sync bubble in adjust_num_token_non_padded_for_attn_tp (#16178)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/forward_batch_info.py (+8/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Currently in function `adjust_num_token_non_padded_for_attn_tp` (https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py#L543) ⏎  ⏎ Two on-device tensors `self.num_token_non_padded` and `num_tokens_per_dp` are passed into `compute_local_num_token_non_padded` function, where the torch.clamp function (https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forwar …[truncated]

### L2-e0e5084802  (L2, 2026-01-01, sha e0e508480247, PR #14085)
TITLE: Fix parse args from file(#13911) (#14085)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+6/-18); python/sglang/srt/server_args_config_parser.py (+52/-26); test/manual/test_config_integration.py (+9/-2)
LABELS: run-ci
ISSUES: #13911 [Bug]  Is `_add_boolean_arg` code  has a problem?
BODY: ## Motivation ⏎ fix(#13911) ⏎  ⏎ The `argparse` will directly generate an instance of the _StoreTrueAction class based on `action="store_true"`, and will not store the `action="store_true"` field on the instance. ⏎  ⏎ So the code `hasattr(action, "action")` will never return True.  You can see the test results I added in the issue(#13911). ⏎  ⏎  ⏎ If the code there had issues, why did it still run successfully? This is because the usage of `boolean_actio …[truncated]

### L2-f0195627a9  (L2, 2026-01-02, sha f0195627a927, PR #16308)
TITLE: Fix sgl-kernel jobs to skip when target_stage is specified (#16308)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+18/-6)
BODY: ## Summary ⏎ - Add `!inputs.target_stage` check to all sgl-kernel jobs to skip them when `/rerun-stage` targets a specific stage ⏎  ⏎ **Jobs fixed:** ⏎ - `sgl-kernel-build-wheels` ⏎ - `sgl-kernel-build-wheels-arm` ⏎ - `sgl-kernel-unit-test` ⏎ - `sgl-kernel-mla-test` ⏎ - `sgl-kernel-benchmark-test` ⏎ - `sgl-kernel-b200-test` ⏎  ⏎ This fixes an issue where Build Wheel and kernel tests were incorrectly running when using `/rerun-stage` to target a specific stage. ⏎  ⏎ **Note …[truncated]

### L2-6256936d09  (L2, 2026-01-02, sha 6256936d0908, PR #16324)
TITLE: Fix: Allow build-wheels to run when target_stage is set (#16324)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+2/-6)
BODY: ## Summary ⏎ Revert the `!inputs.target_stage` check from `sgl-kernel-build-wheels` and `sgl-kernel-build-wheels-arm` jobs. ⏎  ⏎ **Problem:** PR #16308 incorrectly added `!inputs.target_stage` to the build-wheels jobs, causing them to be skipped when `/rerun-stage` is used. This broke stage jobs that depend on the built wheel artifacts. ⏎  ⏎ **Error:** ⏎ ``` ⏎ + ls -alh sgl-kernel/dist ⏎ ls: cannot access 'sgl-kernel/dist': No such file or directory ⏎ ``` ⏎  ⏎ **Fix:* …[truncated]

### L2-0d244116d2  (L2, 2026-01-02, sha 0d244116d28a, PR #13959)
TITLE: [DeepSeek v3.2] opt Context Parallelism: support fused moe, multi batch and fp8 kvcache (#13959)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.npu_mla, L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: docs/advanced_features/server_arguments.md (+1/-0); docs/basic_usage/deepseek_v32.md (+12/-0); python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py (+5/-5); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+45/-68); python/sglang/srt/layers/attention/nsa/utils.py (+209/-5); python/sglang/srt/layers/attention/nsa_backend.py (+149/-20); python/sglang/srt/layers/communicator.py (+14/-4); python/sglang/srt/layers/communicator_nsa_cp.py (+60/-133); python/sglang/srt/managers/schedule_policy.py (+3/-3); python/sglang/srt/models/deepseek_nextn.py (+5/-6); (+4 more)
LABELS: documentation, deepseek, npu, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎  ⏎ The original default token splitting scheme of cp does not support prefill multi-batch. A new token splitting method is introduced to enable multi-batch support, fused MoE compatibility, and FP8 KV-cache support. Compared with the original DeepEP scheme, the combination of the tuned fused MoE backend and the new token splitting method **reduces TTFT by 8.9% (for inputs ≥16K tokens) to 32% (for 1K token inputs)** in 8× H20(141GB …[truncated]

### L2-8b869e326c  (L2, 2026-01-03, sha 8b869e326c7a, PR #15560)
TITLE: [AMD] feat: add DLLM support for AMD GPUs with LLaDA2 testing (#15560)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: .github/workflows/pr-test-amd.yml (+39/-1); python/sglang/srt/model_executor/forward_batch_info.py (+4/-2); python/sglang/srt/server_args.py (+13/-1); test/srt/dllm/test_llada2_mini_amd.py (+87/-0); test/srt/run_suite.py (+1/-0)
LABELS: amd, run-ci
BODY: enable  basic text diffusion model for amd cards. ⏎  ⏎  ⏎ ## Launch Command ⏎  ⏎ ```bash ⏎ python -m sglang.launch_server \ ⏎ --model-path inclusionAI/LLaDA2.0-mini \ ⏎ --dllm-algorithm LowConfidence \ ⏎ --trust-remote-code \ ⏎ --mem-fraction-static 0.9 \ ⏎ --max-running-requests 1 \ ⏎ --attention-backend triton \ ⏎ --host 0.0.0.0 \ ⏎ --port 30000 ⏎ ``` ⏎  ⏎ ## Test Command ⏎  ⏎ ```bash ⏎ curl -s http://127.0.0.1:30000/v1/completions \ ⏎   -H "Content-Type: applicati …[truncated]

### L2-7d757d6f17  (L2, 2026-01-07, sha 7d757d6f17ef, PR #15938)
TITLE: Clean Some Environment Variables for DeepSeek V32 (#15938)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: docs/references/environment_variables.md (+10/-0); python/sglang/srt/environ.py (+4/-0); python/sglang/srt/layers/attention/nsa/dequant_k_cache.py (+2/-7); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+1/-3); python/sglang/srt/layers/attention/nsa/quant_k_cache.py (+6/-42); python/sglang/srt/layers/attention/nsa/utils.py (+0/-24); python/sglang/srt/layers/attention/nsa_backend.py (+8/-19); python/sglang/srt/server_args.py (+8/-13)
LABELS: documentation, quant, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Move `SGLANG_NSA_FUSE_TOPK` and `SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA` to environ.py ⏎ - Deprecate `SGLANG_NSA_DUAL_STREAM`, `SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8`, `SGLANG_NSA_QUANT_K_CACHE_FAST`,  `SGLANG_NSA_DEQUANT_K_CACHE_FAST` ⏎ - Fix some comments by replacing NSA with DSA ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-4935344fcd  (L2, 2026-01-07, sha 4935344fcd57, PR #16531)
TITLE: [AMD] Fix aiter page-size handling, DeepSeek MLA tuple inputs, and HiCache/FA3 decode-backend override (#16531)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.aiter_mla, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+1/-2); python/sglang/srt/models/deepseek_v2.py (+9/-2); python/sglang/srt/server_args.py (+23/-13)
LABELS: amd, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ I hit a few issues while running DeepSeek-R1 MXFP4 on ROCm with aiter + hierarchical cache: ⏎  ⏎ - aiter MLA metadata creation was effectively hard-coding `page_size=1`, and the draft backend also asserted `page_size==1`, which breaks configs like `--page-size 64`. ⏎ - `DeepseekV2AttentionMLA` can receive `hidden_states` as a tuple in some quantization paths, which makes shape-based allocations fail. ⏎ - When HiCache kernel I/O is en …[truncated]

### L2-38dc5839dd  (L2, 2026-01-08, sha 38dc5839dd8d, PR #16306)
TITLE: [1/n]deepseek_v2.py Refactor: attention backend handlers and forward method definition (#16306)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/__init__.py (+0/-0); python/sglang/srt/models/deepseek_common/attention_backend_handler.py (+182/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py (+32/-0); python/sglang/srt/models/deepseek_common/utils.py (+23/-0); python/sglang/srt/models/deepseek_v2.py (+18/-228)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎ 3. Trigge …[truncated]

### L2-12a0292bfd  (L2, 2026-01-08, sha 12a0292bfdde, PR #16678)
TITLE: Revert "[sgl-kernel] Update flashmla to include fp8 sparse_mla optimizations" (#16678)
SOURCES: path_core, subject_keyword, corpus:confirmed-reverts, body_keyword
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: sgl-kernel/cmake/flashmla.cmake (+1/-1)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 15242 reason=correctness_or_accuracy
BODY: Reverts sgl-project/sglang#15242 ⏎  ⏎ This update is breaking the nvfp4 Deepseek checkpoint, leading to 0 accuracy with gsm8k 8 shots. It works fine with the fp8 checkpoint, however. ⏎  ⏎ Debug notes: ⏎ With gsm8k, 0 shots with running max batch_size = 1 works. If I do not limit the max batch_size, I get 0 accuracy. ⏎  ⏎ Comparing the upstream flash_mla (main branch) flash_mla_with_kvcache kernel results to sgl_kernel.flash_mla when running gsm8k 8 shot …[truncated]

### L2-08636f72b5  (L2, 2026-01-09, sha 08636f72b5d9, PR #16810)
TITLE: [Fix CI] Fix test_mamba_unittest.py (#16810)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/attention/test_mamba_unittest.py (+5/-1)
LABELS: ci
BODY: ## Motivation ⏎  ⏎  ⏎ Fix CI test_mamba_unittest.py ⏎ https://github.com/sgl-project/sglang/pull/16768 introduced server_args, but not initialized it in test_mamba unit test. This PR fixed it. ⏎  ⏎ https://github.com/sgl-project/sglang/actions/runs/20844699378/job/59914340819?pr=16732 ⏎  ⏎ ``` ⏎ ============================================================ ⏎ Test Summary: 9/12 passed ⏎ ============================================================ ⏎ ✓ PASSED: ⏎  …[truncated]

### L2-9fd2358cc2  (L2, 2026-01-10, sha 9fd2358cc2fc, PR #16838)
TITLE: Update Cutedsl version and pin cuda-python version (#16838)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-2); python/pyproject.toml (+2/-2)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎ In prior efforts of upgrading cutedsl to >= 4.3.1, the driver will throw the following error: ⏎ ``` ⏎ Error Code: 35 ⏎  ⏎ 🔍 Additional Context: ⏎ - Error name: (<cudaError_t.cudaSuccess: 0>, b'cudaErrorInsufficientDriver') ⏎ - Error code: 35 ⏎ - CUDA_TOOLKIT_PATH: not set ⏎ - Target SM ARCH: not set ⏎  ⏎ 📊 GPU Information: ⏎ - CUDA devices available: 8 (current: <CUdevice 0>) ⏎ - Architecture: Blackwell (sm_100a) ⏎ - Compatible SM archs: sm_1 …[truncated]

### L2-145bd54f1b  (L2, 2026-01-10, sha 145bd54f1b94, PR #15927)
TITLE: Piecewise Cuda Graph Memory Usage (#15927)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/compilation/cuda_piecewise_backend.py (+3/-6); python/sglang/srt/server_args.py (+18/-13)
LABELS: run-ci, piecewise-cuda-graph
BODY: ## Motivation ⏎ ### Memory Components ⏎  ⏎ We decompose GPU memory usage into three components: ⏎  ⏎ - **Non-Torch Memory**   ⏎   GPU memory not managed by PyTorch’s CUDA caching allocator. For Piecewise CUDA Graph (PCG), this mainly corresponds to CUDA Graph metadata and runtime bookkeeping. ⏎  ⏎ - **Torch Dynamic Pool Memory**   ⏎   Memory allocated from PyTorch’s default CUDA caching allocator during eager execution. ⏎  ⏎ - **Torch Private Pool Memory**  …[truncated]

### L2-3fd88ea9b5  (L2, 2026-01-10, sha 3fd88ea9b54b, PR #15790)
TITLE: [MTP][spec_v2] Fix TRTLLM MLA backend crash in EAGLE draft_extend mode  (#15790)
SOURCES: path_core, path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+65/-5)
LABELS: blackwell, run-ci
BODY: ## Motivation ⏎  ⏎ Fix a crash in TRTLLM MLA backend during EAGLE speculative decoding in `draft_extend` mode when sequences have varying `accept_length` values. ⏎  ⏎ This can be reproduced by the following agg command (and also intermittency issues in the disagg setup in benchmark section): ⏎ ``` ⏎ SGLANG_ENABLE_SPEC_V2=1 SGLANG_FLASHINFER_FP4_GEMM_BACKEND=cutlass SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 python3 -m sglang.launch_server     --model-p …[truncated]

### L2-cc25f9df50  (L2, 2026-01-11, sha cc25f9df50e1, PR #16835)
TITLE: Update est_time for stage-b-test-small-1-gpu tests (#16835)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/attention/test_torch_native_attention_backend.py (+1/-1); test/registered/backends/test_torch_compile.py (+1/-1); test/registered/core/test_deterministic.py (+1/-1); test/registered/core/test_gpt_oss_1gpu.py (+1/-1); test/registered/cuda_graph/test_piecewise_cuda_graph_small_1_gpu.py (+1/-1); test/registered/dllm/test_llada2_mini.py (+1/-1); test/registered/hicache/test_hicache_variants.py (+1/-1); test/registered/lora/test_lora_update.py (+1/-1); test/registered/mla/test_flashmla.py (+1/-1); test/registered/mla/test_mla_int8_deepseek_v3.py (+1/-1); (+14 more)
LABELS: lora, Multi-modal, deepseek, speculative-decoding, hicache, npu, run-ci
BODY: ## Summary ⏎ - Updated `est_time` for 24 tests in `stage-b-test-small-1-gpu` suite based on actual elapsed times from 3 CI runs ⏎ - Only updated tests where difference between estimated and actual time was >50 seconds ⏎  ⏎ **CI runs analyzed:** ⏎ - https://github.com/sgl-project/sglang/actions/runs/20851379702 ⏎ - https://github.com/sgl-project/sglang/actions/runs/20861026466 ⏎ - https://github.com/sgl-project/sglang/actions/runs/20842927997 ⏎  ⏎ **Tests with unde …[truncated]

### L2-000ad42225  (L2, 2026-01-15, sha 000ad4222595, PR #17075)
TITLE: chore: bump sgl-kernel version to 0.3.21 (#17075)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+4/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: high priority, dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.21` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.21` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are updated accord …[truncated]

### L2-146b5fcc84  (L2, 2026-01-15, sha 146b5fcc8410, PR #16826)
TITLE: [CI] Reorganize stage-b 1-GPU tests for 5090 compatibility (#16826)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: .github/workflows/pr-test.yml (+15/-58); scripts/ci/slash_command_handler.py (+0/-1); test/registered/attention/test_create_kvindices.py (+0/-1); test/registered/attention/test_mamba_unittest.py (+0/-1); test/registered/attention/test_radix_attention.py (+0/-1); test/registered/attention/test_radix_cache_unit.py (+0/-1); test/registered/attention/test_swa_unittest.py (+1/-1); test/registered/attention/test_torch_native_attention_backend.py (+0/-1); test/registered/attention/test_triton_attention_backend.py (+1/-1); test/registered/attention/test_triton_attention_kernels.py (+1/-1); (+126 more)
LABELS: high priority, quant, amd, lora, Multi-modal, deepseek, speculative-decoding, hicache, npu, run-ci
BODY: ## Summary ⏎  ⏎ This PR reorganizes the 1-GPU test suites to fully integrate RTX 5090 runners. ⏎  ⏎ ### Test Distribution ⏎  ⏎ | Suite | Runner | GPU | Tests | Description | ⏎ |-------|--------|-----|-------|-------------| ⏎ | `stage-b-test-small-1-gpu` | `1-gpu-5090` | RTX 5090 (32GB, SM120) | **67** | Tests that pass on 5090 | ⏎ | `stage-b-test-large-1-gpu` | `1-gpu-runner` | H200 (80GB, SM90) | **56** | Tests incompatible with 5090 | ⏎  ⏎ ### Changes ⏎  ⏎ 1. **`stage-b …[truncated]

### L2-a04675892e  (L2, 2026-01-17, sha a04675892eba, PR #15551)
TITLE: Update flashinfer to 0.6.1 (#15551)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-2); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-1); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+0/-1); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+0/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+0/-1); python/sglang/srt/layers/quantization/mxfp4.py (+0/-1); python/sglang/srt/utils/common.py (+1/-0); scripts/ci/ci_install_dependency.sh (+1/-1)
LABELS: documentation, high priority, quant, amd, dependencies, lora, Multi-modal, deepseek, speculative-decoding, hicache
BODY: ## Motivation ⏎ flashinfer -> 0.6.1 ⏎ flashinfer-cubin -> 0.6.1 ⏎  ⏎ PRs dependent on this upgrade: ⏎ #15546 ⏎ #15422 ⏎ #15514 ⏎ #15347 ⏎ #14668 ⏎ #16232 ⏎ #16279 ⏎ #16892 ⏎ #16534 ⏎ #12787 ⏎ ... ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-8b9e9357fe  (L2, 2026-01-17, sha 8b9e9357fe28, PR #16817)
TITLE: [2/n] deepseek_v2.py Refactor: Migrate MHA forward method in deepseek_v2.py (#16817)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py (+232/-0); python/sglang/srt/model_executor/forward_batch_info.py (+4/-220); python/sglang/srt/models/deepseek_common/attention_backend_handler.py (+3/-1); python/sglang/srt/models/deepseek_common/attention_forward_methods/__init__.py (+7/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py (+1/-1); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py (+493/-0); python/sglang/srt/models/deepseek_v2.py (+6/-407)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Part of #16255. This PR moves all the mha forward functions to a single file `forward_mha.py`. ⏎ Checklist: ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests (H200) ⏎  ⏎ ### GSM8K ⏎  ⏎ ``` ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319 ⏎ ``` ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3-0324 --tp 8 --trust-remote-code ⏎  ⏎ Main branch: ⏎  ⏎ Accuracy: 0.940 ⏎ Invalid: 0.0 …[truncated]

### L2-a45e0e5df4  (L2, 2026-01-18, sha a45e0e5df4e6, PR #16974)
TITLE: [SPEC_V2] Enable cudagraph draft_extend for trtllm_mla_backend and Acclen Fix for DP under cudagraph mode (#16974)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+15/-3); python/sglang/srt/speculative/eagle_worker_v2.py (+7/-0)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ 1. Enable CUDA graph draft extend support for `trtllm_mla_backend` ⏎ 2. Fix accept length degradation (~35% drop) when draft extend CUDA graph is enabled with DP Attention ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Add `TRTLLMMLAMultiStepDraftBackend` support in `eagle_worker_v2.py` to enable CUDA graph capture for draft extend ⏎ 2. Fix `num_tokens_for_logprob_per_batch` / `global_num_tokens_for_logprob_gpu` to match `num_tokens_per_batch` in CUDA  …[truncated]

### L2-93433726eb  (L2, 2026-01-19, sha 93433726eb72, PR #16649)
TITLE: [Refactor] Split out deepseek v2 weight loader function into mixin (#16649)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py (+657/-0); python/sglang/srt/models/deepseek_common/utils.py (+53/-1); python/sglang/srt/models/deepseek_nextn.py (+2/-5); python/sglang/srt/models/deepseek_v2.py (+9/-594)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎ DeepseekV2 code has been developed fast and a lot of historical code and be more orgranized, including the weight loading part. ⏎  ⏎ Issue related: https://github.com/sgl-project/sglang/issues/16291 ⏎  ⏎ ## Modifications ⏎ This PR just **moves** the weight loader function into a mixin with some documentations. ⏎  ⏎ The further refactors of splitting the weight loading internal will come after this PR get merged. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Se …[truncated]

### L2-d2105d4abd  (L2, 2026-01-19, sha d2105d4abda6, PR #16961)
TITLE: [DeepSeek v3.2] Opt MTP decode cuda batch sizes and nsa implementation (#16961)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+14/-5); python/sglang/srt/model_executor/cuda_graph_runner.py (+12/-7)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ 1、`draft_extend` and `target_verify` use `nsa_decode_backend` as backend implementation instead of `nsa_prefill_backend` ⏎ In the MTP scenario, the NSA attention implementation selected for `prefill`, `draft_extend`, and `target_verify` is all `nsa_prefill_backend`, which is typically set to `flashmla_sparse`. For `draft_extend` and `target_verify` scenarios with a small number of tokens, we observed that `fa3` delivers better perfo …[truncated]

### L2-6988a0f570  (L2, 2026-01-19, sha 6988a0f5706c, PR #17327)
TITLE: Disable mla persistent kernel when not using fp8 kv_cache (#17327)
SOURCES: path_integration+keyword, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+3/-1)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: cc @HaiShaw, @kkHuang-amd  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ The default value of `_use_mla_ps_kernel` is `True` ([PR#13147](https://github.com/sgl-project/sglang/pull/13147/changes#diff-b6bc9b31b10dc03f374a72a56e4004a339b83073e25948c6fff71ec1a5a92e58R49)), which forces the use of persistent kernel even when fp8 kv cache is not enabled, causing performance degradation. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Disable `_use_mla_ps_kernel` when fp8 kv cache is not used. ⏎  …[truncated]

### L2-95f59c13fd  (L2, 2026-01-21, sha 95f59c13fd08, PR #17493)
TITLE: [Chore] include all jit files in building packages (#17493)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-10); python/pyproject_cpu.toml (+2/-10); python/pyproject_other.toml (+2/-10); python/pyproject_xpu.toml (+2/-10)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ To include `jit_kernel/include/sgl_kernel/impl/`, and `jit_kernel/include/sgl_kernel/gemm/`. ⏎  ⏎ I'm not very familiar with packaging, need help in reviewing this cc @merrymercy @Kangyan-Zhou @zhyncs . Before this PR, I found that the original packaging code already include all files uner `jit_kernel`. However, this is **unexpected**, since `jit_kernel/include/sgl_kernel/*.cuh` shouldn't match files like `jit_kernel/include/sgl_ …[truncated]

### L2-d6e2b88288  (L2, 2026-01-22, sha d6e2b88288ff, PR #16890)
TITLE: Add Liquid Foundation Model (LFM2) (#16890)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/lfm2.py (+102/-0); python/sglang/srt/configs/mamba_utils.py (+9/-6); python/sglang/srt/function_call/function_call_parser.py (+2/-0); python/sglang/srt/function_call/lfm2_detector.py (+387/-0); python/sglang/srt/model_executor/model_runner.py (+2/-1); python/sglang/srt/models/lfm2.py (+566/-0); python/sglang/srt/server_args.py (+27/-0); test/registered/function_call/test_function_call_parser.py (+358/-0); test/registered/models/test_generation_models.py (+10/-0); (+1 more)
LABELS: run-ci
BODY: ## Summary ⏎ - Add support for LiquidAI's LFM2 hybrid architecture (attention + ShortConv layers) ⏎ - LFM2 uses gated 1D causal convolution (kernel=3) instead of attention in some layers, requiring SGLang's hybrid caching system ⏎ - Tested with [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) and [LiquidAI/LFM2-2.6B-Exp](https://huggingface.co/LiquidAI/LFM2-2.6B-Exp) ⏎  ⏎ ## Changes ⏎ | File | Purpose | ⏎ |------|--- …[truncated]

### L2-283a2daeaa  (L2, 2026-01-22, sha 283a2daeaa88, PR #17591)
TITLE: [hotfix] Reenable all reduce fusion on sm100 (#17591)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/communicator.py (+2/-3)
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Motivation ⏎  ⏎ Following #17474 ⏎  ⏎ After benchmarking on dpsk fp4 + B200 + tp4 ⏎  ⏎ Even with broken allreduce+norm fusion kernel, it's still faster than nccl allreduce ⏎ So this PR adds this kernel back to sm100 machine ⏎  ⏎  ⏎ ``` ⏎ SGLANG_ENABLE_SPEC_V2=1 python3 -m sglang.launch_server   --model-path nvidia/DeepSeek-R1-0528-FP4-v2   --tp 4   --kv-cache-dtype fp8_e4m3   --attention-backend trtllm_mla   --moe-runner-backend flashinfer_trtllm   --qua …[truncated]

### L2-628ab5d57b  (L2, 2026-01-23, sha 628ab5d57b33, PR #17053)
TITLE: [MUSA][2/N] sgl-kernel build (#17053)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .gitignore (+14/-0); docs/platforms/mthreads_gpu.md (+25/-0); sgl-kernel/csrc/common_extension_musa.cc (+51/-0); sgl-kernel/pyproject_musa.toml (+33/-0); sgl-kernel/setup_musa.py (+205/-0)
LABELS: documentation, dependencies, sgl-kernel, run-ci, mthreads
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is the second in a series of pull requests (tracked in https://github.com/sgl-project/sglang/issues/16565) to add full support for [Moore Threads](https://en.mthreads.com/) GPUs, leveraging MUSA (Meta-computing Unified System Architecture) to accelerate LLM inference. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Following the AMD approach, we add a small set of MUSA-specific files: ⏎  ⏎ 1. `pyproject_musa.toml`: used later during the Docker  …[truncated]
