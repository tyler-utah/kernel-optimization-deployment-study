### L2-22085081bb  (L2, 2024-01-08, sha 22085081bb24, PR #)
TITLE: release initial code
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules; python/pyproject.toml
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L2-dafafe5b11  (L2, 2024-01-18, sha dafafe5b111d, PR #42)
TITLE: Use HTTP link in 3rdparty module (#42)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+1/-1)
BODY: Replace the SSH link of FlashInfer with HTTP link, so that users don't need to deploy the SSH key of their Github account for cloning, as deploying SSH key could be tedious in cloud instances or docker containers.

### L2-26c3494152  (L2, 2024-02-06, sha 26c349415213, PR #156)
TITLE: [Submodule] Change FlashInfer to import (#156)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+0/-3); 3rdparty/flashinfer (+0/-1); docs/flashinfer.md (+5/-3); python/sglang/srt/layers/radix_attention.py (+0/-8); python/sglang/srt/managers/router/model_runner.py (+12/-9)
BODY: This PR removes submodule flashinfer from this repo. As flashinfer is now officially released, we can directly install it via pip. However, I didn't add it to pyproject.toml but just updated the README due to two reasons: ⏎ 1. flashinfer is not available on PYPI yet. ⏎ 2. It supports limited CUDA version and requires to manually select the wheel path. ⏎  ⏎ Meanwhile, the latest version of flashinfer has some interface changes. I modified them accordi …[truncated]

### L2-33b242df30  (L2, 2024-05-11, sha 33b242df303e, PR #380)
TITLE: Compat with latest VLLM 0.4.2 main + fork.number rename + Flashinfer 0.0.4 (#380)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/lang/interpreter.py (+7/-7); python/sglang/lang/tracer.py (+6/-4); python/sglang/srt/layers/logits_processor.py (+1/-1); python/sglang/srt/managers/router/model_rpc.py (+5/-2); python/sglang/srt/managers/router/model_runner.py (+6/-14); python/sglang/srt/models/commandr.py (+19/-18); python/sglang/srt/models/dbrx.py (+20/-19); python/sglang/srt/models/gemma.py (+18/-17); python/sglang/srt/models/llama2.py (+19/-18); (+10 more)
BODY: Reason for PR: ⏎  ⏎ * Compat with VLLM main ⏎ * Rename `fork(number=N)` param to `fork(size=N)` for clarity.  ⏎ * Complete flashinfer 0.0.4 todo

### L2-2d580e7a89  (L2, 2024-05-12, sha 2d580e7a8991, PR #430)
TITLE: Fix flashinfer (#430)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/managers/router/model_rpc.py (+2/-1); python/sglang/srt/managers/router/model_runner.py (+3/-3)
BODY: 

### L2-ac11388756  (L2, 2024-07-04, sha ac113887560c, PR #588)
TITLE: Add docker file (#588)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+6/-0); README.md (+3/-0)
BODY: 

### L2-f6b29f6920  (L2, 2024-07-16, sha f6b29f692083, PR #629)
TITLE: Update docker file (#629)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+36/-5)
BODY: 

### L2-8832ecb1e4  (L2, 2024-07-16, sha 8832ecb1e451, PR #632)
TITLE: Reduce docker size (#632)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+8/-5); README.md (+10/-0)
BODY: Reduce the docker size and add the docker usage in readme

### L2-e1eae1fd15  (L2, 2024-08-05, sha e1eae1fd15ed, PR #905)
TITLE: Support MLA for DeepSeek-V2 with Triton - step 1 (#905)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.pool.mla_token_kv, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/radix_attention.py (+22/-9); python/sglang/srt/mem_cache/memory_pool.py (+65/-24); python/sglang/srt/model_executor/model_runner.py (+46/-17); python/sglang/srt/models/deepseek_v2.py (+198/-16); python/sglang/srt/server_args.py (+6/-0); benchmark/gsm8k/download_data.sh (+0/-0); python/sglang/srt/layers/extend_attention.py (+59/-7); python/sglang/srt/layers/token_attention.py (+28/-2); python/sglang/srt/managers/schedule_batch.py (+4/-3); python/sglang/srt/model_config.py (+11/-0)
BODY: ## Motivation ⏎  ⏎ MLA implementation. ⏎  ⏎ ## Modification

### L2-c31f084c71  (L2, 2024-08-07, sha c31f084c713c, PR #966)
TITLE: chore: update vllm to 0.5.4 (#966)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); .github/workflows/e2e-test.yml (+1/-2); .github/workflows/unit-test.yml (+1/-2); README.md (+2/-2); python/sglang/check_env.py (+1/-0); test/srt/models/test_causal_models.py (+1/-3); test/srt/run_suite.py (+1/-1); test/srt/test_chunked_prefill.py (+1/-1); test/srt/test_eval_accuracy.py (+1/-1); (+4 more)
BODY: Thank you for your contribution, we really appreciate it. The following instructions will help improve your pull request and make it easier to receive feedback. If there are any items you don't understand, don't worry. Just submit the pull request and ask the maintainers for help. ⏎  ⏎ ## Motivation ⏎  ⏎ Please explain the motivation behind this PR and the goal you aim to achieve with it. ⏎  ⏎ ## Modification ⏎  ⏎ Briefly describe the changes made in thi …[truncated]

### L2-cb99ba4fc6  (L2, 2024-08-12, sha cb99ba4fc619, PR #1033)
TITLE: feat: update Dockerfile (#1033)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+34/-19); .github/workflows/release-docker.yml (+25/-13)
BODY: ## Motivation ⏎  ⏎ Inspired by https://github.com/sgl-project/sglang/pull/999 so I co-authored with @vhain  ⏎  ⏎  ⏎ ## Modification ⏎  ⏎ 1. update Python 3.10 on Ubuntu 20.04 ⏎ 2. use GitHub Matrix to support SRT only (all build tasks are parallel) ⏎ 3. support cu118 ⏎  ⏎ ## Checklist ⏎  ⏎ 1. Ensure pre-commit `pre-commit run --all-files` or other linting tools are used to fix potential lint issues. ⏎ 4. Confirm that modifications are covered by complete unit t …[truncated]

### L2-df191254ab  (L2, 2024-08-19, sha df191254abc0, PR #1138)
TITLE: Optimize MLA/GQA/MQA Triton decoding (#1138)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/decode_attention.py (+337/-49)
LABELS: high priority, performance
BODY: ## Motivation ⏎  ⏎ Optimize memory access for MLA/GQA/MQA decoding. ⏎  ⏎ ## Modification ⏎  ⏎ One block handle `BLOCK_H` q heads with shared k/v head. Inspired by https://github.com/InternLM/lmdeploy/pull/1649.

### L2-2c615d120f  (L2, 2024-08-25, sha 2c615d120fa5, PR #1204)
TITLE: [Feature] Support fp8 e5m2 kv cache with flashinfer (#1204)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/radix_attention.py (+3/-4); python/sglang/srt/mem_cache/memory_pool.py (+82/-8); python/sglang/srt/model_executor/forward_batch_info.py (+4/-0); python/sglang/srt/model_executor/model_runner.py (+19/-4); python/sglang/srt/server_args.py (+8/-0)
LABELS: feature
BODY: ## Motivation ⏎  ⏎ Support fp8 e5m2 kv cache with flashinfer. ⏎  ⏎ ## Usage ⏎ Add `--kv-cache-dtype fp8_e5m2` to enable this feature. Currently it only works when flashinfer is not disabled. ⏎  ⏎ ## Performance & Accuracy ⏎ Tested with llama2-13b-chat on A100, the throughput increased by **17.8%** without accuracy degradation. ⏎ |  Enable fp8_e5m2 kv cache| Throughput | MMLU (nsub=10) Avg Accuracy | gsm8k Accuracy | ⏎ | --------------------- | ---------- | …[truncated]

### L2-f414352ae6  (L2, 2024-08-30, sha f414352ae678, PR #1261)
TITLE: Transpose mla weight offline (#1261)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+13/-7)
BODY: ## Motivation ⏎  ⏎ This change will boost performance slightly and reduce runtime cuda memory usage. ⏎  ⏎ ## Modifications ⏎  ⏎ Preprocess weight after weight loading.

### L2-54772f784a  (L2, 2024-09-01, sha 54772f784adb, PR #1285)
TITLE: feat: fix fp8 for MLA and support bmm fp8 for DeepSeek V2 (#1285)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+41/-19)
LABELS: enhancement
BODY: ## Motivation ⏎  ⏎ Paired programming with @ispobock , completed this feature, gsm8k fp16 and fp8 are both normal, will continue to use nvtx and nsys for performance analysis and optimization. ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --enable-mla --trust-remote-code --disable-radix --mem-frac 0.85 ⏎ python3 -m sglang.launch_server --model neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8 --enable-mla  …[truncated]

### L2-6cb32ef92c  (L2, 2024-09-01, sha 6cb32ef92c99, PR #1286)
TITLE: Support Triton fp8 e5m2 kv cache (#1286)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/extend_attention.py (+12/-4); python/sglang/srt/model_executor/model_runner.py (+1/-7)
LABELS: enhancement
BODY: ## Motivation ⏎  ⏎ Previously fp8 kv cache for Flashinfer was supported in https://github.com/sgl-project/sglang/pull/1204. ⏎ This PR support this feature for Triton runtime. ⏎  ⏎ ## Evaluation ⏎ DeepSeek-Coder-V2-Lite-Instruct ⏎ | backend    | kv cache dtype | gsm8k flexible-extract | gsm8k strict-match | ⏎ | ---------- | -------------- | ---------------------- | ------------------ | ⏎ | Flashinfer | bf16           | 0.7703                 | 0.7627       …[truncated]

### L2-46094e0c1b  (L2, 2024-09-10, sha 46094e0c1b9c, PR #1380)
TITLE: Deprecate --disable-flashinfer and introduce --attention-backend (#1380)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: README.md (+1/-1); docs/en/install.md (+1/-1); python/sglang/srt/layers/radix_attention.py (+6/-2); python/sglang/srt/layers/sampler.py (+6/-2); python/sglang/srt/managers/schedule_batch.py (+6/-4); python/sglang/srt/managers/tp_worker.py (+1/-1); python/sglang/srt/model_executor/forward_batch_info.py (+3/-3); python/sglang/srt/model_executor/model_runner.py (+13/-10); python/sglang/srt/server.py (+1/-1); python/sglang/srt/server_args.py (+51/-24); (+3 more)
BODY: We plan to build better abstraction around the attention backends, so we can easily switch between them and share the upper-level scheduling and optimizations (e.g., cuda graph). ⏎  ⏎ We will use `--attention-backend triton` and `--attention-backend flashinfer` instead of `--disable-flashinfer`.

### L2-15c75e4146  (L2, 2024-09-11, sha 15c75e41462d, PR #1389)
TITLE: [Fix] Fix --disable-flashinfer (#1389)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+2/-0)
BODY: https://github.com/sgl-project/sglang/pull/1380#discussion_r1753078941

### L2-fec185ce0c  (L2, 2024-09-11, sha fec185ce0cba, PR #1381)
TITLE: Refactor attention backend (#1381)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention_backend.py (+383/-0); python/sglang/srt/layers/flashinfer_utils.py (+35/-37); python/sglang/srt/layers/radix_attention.py (+7/-168); python/sglang/srt/layers/triton_attention/decode_attention.py (+3/-4); python/sglang/srt/layers/triton_attention/extend_attention.py (+12/-19); python/sglang/srt/managers/schedule_batch.py (+1/-1); python/sglang/srt/model_executor/cuda_graph_runner.py (+46/-108); python/sglang/srt/model_executor/forward_batch_info.py (+26/-108); python/sglang/srt/model_executor/model_runner.py (+28/-97); python/sglang/srt/sampling/sampling_batch_info.py (+3/-5); (+6 more)
BODY: - Introduce a base class `AttentionBackend` to abstract away the different attention kernel backends ⏎ - Now we have `FlashInferAttnBackend` and `TritonAttnBackend`

### L2-3efa798116  (L2, 2024-09-12, sha 3efa79811641, PR #1401)
TITLE: Support cuda graph in the triton attention backend (#1401)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention_backend.py (+95/-13); python/sglang/srt/layers/flashinfer_utils.py (+10/-10); python/sglang/srt/layers/triton_attention/decode_attention.py (+19/-25); python/sglang/srt/model_executor/cuda_graph_runner.py (+13/-6); python/sglang/srt/model_executor/model_runner.py (+0/-6); test/srt/test_serving_throughput.py (+10/-0)
BODY: ## Llama 3 8B (1.3x faster) ⏎  ⏎ ``` ⏎ # triton w/ cuda graph ⏎ # Decode.  median latency: 0.00706 s, median throughput:    141.63 token/s ⏎ python3 -m sglang.bench_latency --model meta-llama/Meta-Llama-3-8B --batch-size 1 --input 128 --output 8 --attention-backend triton ⏎  ⏎ # triton w/o cuda graph ⏎ # Decode.  median latency: 0.00928 s, median throughput:    107.79 token/s ⏎ python3 -m sglang.bench_latency --model meta-llama/Meta-Llama-3-8B --batch-siz …[truncated]

### L2-9463bc1385  (L2, 2024-09-14, sha 9463bc13856a, PR #1422)
TITLE: Enable torch.compile for triton backend (#1422)
SOURCES: release_notes
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: README.md (+12/-11); python/sglang/bench_latency.py (+2/-1); python/sglang/srt/layers/attention_backend.py (+16/-13); python/sglang/srt/layers/triton_attention/decode_attention.py (+11/-19); python/sglang/test/test_utils.py (+39/-1); test/srt/test_bench_latency.py (+9/-62); test/srt/test_bench_serving.py (+8/-8); test/srt/test_moe_eval_accuracy_large.py (+3/-3); test/srt/test_triton_attn_backend.py (+33/-20)
BODY: ## w/ torch.compile ⏎ ``` ⏎ python3 -m sglang.bench_latency --model meta-llama/Meta-Llama-3-8B --batch-size 1 --input 128 --output 8 --attention-backend triton --enable-torch-compile ⏎ Decode.  median latency: 0.00616 s, median throughput:    162.40 token/s ⏎ ``` ⏎  ⏎ ## w/o torch.compile ⏎ ``` ⏎ python3 -m sglang.bench_latency --model meta-llama/Meta-Llama-3-8B --batch-size 1 --input 128 --output 8 --attention-backend triton ⏎ Decode.  median latency: 0. …[truncated]

### L2-76524b70d1  (L2, 2024-09-17, sha 76524b70d1f8, PR #1442)
TITLE: Fix torch compile for deepseek-v2 (#1442)
SOURCES: release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/model_executor/cuda_graph_runner.py (+12/-1); python/sglang/srt/models/deepseek_v2.py (+1/-0); python/sglang/srt/server_args.py (+7/-0)
BODY: ## Motivation ⏎ Fix issue https://github.com/sgl-project/sglang/issues/1372. ⏎  ⏎ ## Bench Latency ⏎ ### w/ torch compile ⏎ ``` ⏎ python3 -m sglang.bench_latency --model /workdir/llm_models/DeepSeek-V2-Lite/ --disable-radix --trust-remote-code --input-len 128 --output-len 8 --batch 1 --enable-mla --enable-torch-compile ⏎  ⏎ Decode.  median latency: 0.00742 s, median throughput:    134.82 token/s ⏎ Total. latency:  0.095 s, throughput:   1428.89 token/s ⏎ ` …[truncated]

### L2-3a6e04185b  (L2, 2024-09-17, sha 3a6e04185b8d, PR #1420)
TITLE: [Feature, Hardware] Enable SGLang on AMD GPUs via PyTorch for ROCm (#1420)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/activation.py (+12/-0); python/sglang/srt/layers/attention_backend.py (+11/-7); python/sglang/srt/layers/fused_moe/layer.py (+27/-7); python/sglang/srt/layers/layernorm.py (+12/-0); python/sglang/srt/layers/sampler.py (+10/-6); python/sglang/srt/lora/lora_manager.py (+5/-2); python/sglang/srt/models/deepseek_v2.py (+5/-1); python/sglang/srt/models/minicpm3.py (+5/-1); python/sglang/srt/server.py (+5/-0); python/sglang/srt/server_args.py (+7/-0); (+1 more)
BODY: ## Motivation ⏎ - Enable SGLang on AMD GPUs ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ - Bypass `FlashInfer` backend untill it is available on AMD/ROCm ⏎ - Add proper fix for AMD FP8 `e4m3fnuz` to support Fused_MoE ⏎ - Dependency over `vLLM>=0.5.5`, I modified `pyproject.toml` just to confirm that it works up to 0.6.0 as well. ⏎ - Misc. ⏎  ⏎ - TODO: follow-up to address one error (below) when `cuda-graph` is enabled. ⏎ ``` ⏎ File "/sglang/python/sglang/srt/layers/sampler. …[truncated]

### L2-c6b6d2e71b  (L2, 2024-09-17, sha c6b6d2e71b2b, PR #1447)
TITLE: Enable MLA by default (#1447)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/model_executor/model_runner.py (+3/-3); python/sglang/srt/models/deepseek_v2.py (+2/-2); python/sglang/srt/server_args.py (+7/-7); README.md (+0/-1); docs/en/backend.md (+0/-1); python/sglang/srt/managers/schedule_batch.py (+1/-1); python/sglang/srt/models/minicpm3.py (+2/-2); test/srt/test_nightly_gsm8k_eval.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ Enable MLA by default. Add server arg `--disable-mla` to switch it.

### L2-b3710d2c93  (L2, 2024-09-17, sha b3710d2c93b6, PR #1448)
TITLE: Fix attention backend (#1448)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/model_executor/model_runner.py (+8/-0); python/sglang/srt/server_args.py (+0/-4)
BODY: Fix attention backend change in https://github.com/sgl-project/sglang/pull/1447.

### L2-b8ccaf4d73  (L2, 2024-09-21, sha b8ccaf4d737a, PR #1484)
TITLE: Add MLA gsm8k eval (#1484)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+1/-1); python/sglang/test/test_utils.py (+2/-2); test/srt/test_mla.py (+12/-0)
BODY: ## Motivation ⏎  ⏎ Add MLA gsm8k eval in pr test & nightly eval.

### L2-42a2d82ba7  (L2, 2024-09-23, sha 42a2d82ba71d, PR #1494)
TITLE: minor: add mla fp8 test (#1494)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+1/-0); python/sglang/test/test_utils.py (+1/-0); test/srt/test_mla_fp8.py (+50/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ``` ⏎ Writing report to /tmp/mgsm_en_neuralmagic_DeepSeek-Coder-V2-Lite-Instruct-FP8.html ⏎ {'en': 0.852, 'en:std': 0.35509998591945907, 'group_latin': 0.852, 'group_latin:std': 0.35509998591945907, 'score:std': 0.35509998591945907, 'score': 0.852} ⏎ Writing results to /tmp/mgsm_en_neuralmagic_DeepSeek-Coder-V2-Lite-Instruct-FP8.json ⏎ Total latency: 48.074 s ⏎ Score: 0.852 ⏎ . ⏎ ----------------------------------------------------- …[truncated]

### L2-99ec439da4  (L2, 2024-09-30, sha 99ec439da476, PR #1547)
TITLE: Organize Attention Backends (#1547)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+0/-0); python/sglang/srt/layers/attention/__init__.py (+49/-0); python/sglang/srt/layers/attention/flashinfer_backend.py (+2/-195); python/sglang/srt/layers/attention/flashinfer_utils.py (+0/-0); python/sglang/srt/layers/attention/triton_backend.py (+161/-0); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+3/-1); python/sglang/srt/layers/attention/triton_ops/prefill_attention.py (+0/-0); python/sglang/srt/model_executor/forward_batch_info.py (+1/-1); python/sglang/srt/model_executor/model_runner.py (+2/-1); scripts/deprecated/test_flashinfer.py (+3/-3); (+2 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ To better split flashinfer backends and triton backends, for future different features on these two backends. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-061e546313  (L2, 2024-10-14, sha 061e54631352, PR #1459)
TITLE: Support double sparsity (#1459)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/double_sparsity_backend.py (+281/-0); python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py (+772/-0); python/sglang/srt/mem_cache/memory_pool.py (+58/-0); python/sglang/srt/model_executor/model_runner.py (+49/-1); python/sglang/srt/server_args.py (+45/-0); test/srt/Llama-3.1-8B-Instruct.json (+1/-0); test/srt/run_suite.py (+1/-0); test/srt/test_double_sparsity.py (+62/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ - Support double sparsity (post-training sparse attention) for long context inference in SGLang ⏎ - See [paper](https://arxiv.org/pdf/2408.07092) ⏎  ⏎ ## Modifications ⏎  ⏎ - Add triton implementation in `sglang/python/sglang/srt/layers/sparse_decode_attention.py`  ⏎ - Add serving-related parts ⏎  ⏎ ## Speedup Evaluation ⏎  ⏎ Run double sparsity with: ⏎ ```bash ⏎ python -m sglang.bench_latency --model-path lmsys/longchat-7b-v1.5-32k \ ⏎     - …[truncated]

### L2-c77762d57f  (L2, 2024-10-27, sha c77762d57f41, PR #1819)
TITLE: Fix Triton decode kernel & ut (#1819)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+101/-30); python/sglang/srt/layers/attention/triton_ops/prefill_attention.py (+1/-1); test/srt/run_suite.py (+2/-1); test/srt/test_triton_attention_backend.py (+0/-0); test/srt/test_triton_attention_kernels.py (+112/-8)
BODY: ## Motivation ⏎ - Set upper-bound limit for BLOCK_H in grouped decode Triton kernel to avoid OOM ⏎ - Fix and add Triton attention kernel unit tests

### L2-2d4ce1b792  (L2, 2024-10-30, sha 2d4ce1b7928d, PR #1845)
TITLE: [Performance, Triton Kernel Args] _decode_grouped_softmax_reducev_fwd… (#1845)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+9/-0)
BODY: … speedup on ROCm ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ Speedup `_decode_grouped_softmax_reducev_fwd`. ⏎ Test shows ~1.0% improvement to median decode throughput on MI300x with Grok-1 and FP8 (b32/i1024/o256) ⏎  ⏎ ## Modifications ⏎  ⏎ Setting optimal kernel arguments to `_fwd_grouped_kernel_stage2` on ROCm. ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/contributor_guide.md). ⏎ - [ …[truncated]

### L2-087ab83223  (L2, 2024-11-10, sha 087ab832236e, PR #1980)
TITLE: [Performance, Triton] Optimize over mask compute to tl.load in fused_moe_kernel (#1980)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+7/-0); python/sglang/srt/layers/fused_moe/fused_moe.py (+23/-7)
BODY: ## Motivation ⏎  ⏎ Test shows `~0.5%` boost to prefill, `~1.0%` boost to median decode throughput over Grok-1 with `b32/i1023/o256` settings. ⏎  ⏎ ## Modifications ⏎  ⏎ `fused_moe_kernel`: simplify the mask part for even K BLOCK sizes ⏎ `decode_attention`: adjust kernel args ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Add unit …[truncated]

### L2-976bc302e5  (L2, 2024-11-16, sha 976bc302e52b, PR #1970)
TITLE: Support DP MLA (#1970)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/triton_backend.py (+7/-3); python/sglang/srt/model_executor/forward_batch_info.py (+21/-0); python/sglang/srt/model_executor/model_runner.py (+8/-0); python/sglang/srt/models/deepseek_v2.py (+147/-44); python/sglang/srt/server_args.py (+16/-0); .github/workflows/pr-test.yml (+1/-0); python/sglang/srt/managers/data_parallel_controller.py (+43/-8); python/sglang/srt/managers/schedule_batch.py (+24/-5); python/sglang/srt/managers/scheduler.py (+55/-3); python/sglang/srt/managers/tp_worker.py (+7/-0); (+2 more)
BODY: ## Motivation ⏎  ⏎ Support data parallel on MLA for DeepSeek model to reduce replicated KV cache.  ⏎ ``` ⏎ python -m sglang.launch_server --model-path neuralmagic/DeepSeek-Coder-V2-Instruct-FP8 --trust-remote-code --tp 8 --dp 8 --enable-dp-attention ⏎ ``` ⏎ ## Modifications ⏎ - Add `--enable-dp-attention` option. When it is turned on, DP and TP share the same workers. ⏎ - Add `IDLE` forward mode for workers that do not have sequence to forward but need T …[truncated]

### L2-11f881d173  (L2, 2024-11-17, sha 11f881d173c4, PR #2065)
TITLE: Deprecate --disable-flashinfer and --disable-flashinfer-sampling (#2065)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+22/-26); python/sglang/srt/utils.py (+2/-0); test/srt/test_torch_compile_moe.py (+1/-2)
BODY: 

### L2-62832bb272  (L2, 2024-11-17, sha 62832bb2728e, PR #2061)
TITLE: Support cuda graph for DP attention (#2061)
SOURCES: release_notes
ARTIFACT_HINTS: L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/managers/schedule_batch.py (+10/-0); python/sglang/srt/managers/scheduler.py (+23/-9); python/sglang/srt/managers/tp_worker.py (+0/-3); python/sglang/srt/managers/tp_worker_overlap_thread.py (+1/-4); python/sglang/srt/model_executor/cuda_graph_runner.py (+44/-6); python/sglang/srt/model_executor/forward_batch_info.py (+2/-0); python/sglang/srt/model_executor/model_runner.py (+3/-0); python/sglang/srt/server_args.py (+3/-2); scripts/playground/reference_hf.py (+2/-2)
BODY: ## Motivation ⏎  ⏎ Support cuda graph for DP attention (https://github.com/sgl-project/sglang/pull/1970).

### L2-ebaa2f3199  (L2, 2024-11-17, sha ebaa2f31996e, PR #2066)
TITLE: Rename arguments `--disable-nan-detection` to `--enable-nan-detection` (#2066)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/sampler.py (+1/-1); python/sglang/srt/managers/schedule_batch.py (+1/-1); python/sglang/srt/model_executor/model_runner.py (+5/-1); python/sglang/srt/models/gemma2.py (+1/-0); python/sglang/srt/server_args.py (+9/-17)
BODY: 

### L2-a9e90b4bce  (L2, 2024-11-17, sha a9e90b4bcecd, PR #2068)
TITLE: [Minor] Fix styles for overlap mode (#2068)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/scheduler.py (+2/-2); python/sglang/srt/managers/tp_worker_overlap_thread.py (+1/-10); python/sglang/srt/model_executor/model_runner.py (+0/-4); test/srt/test_triton_attention_backend.py (+5/-1)
BODY: 

### L2-62a4a339eb  (L2, 2024-11-22, sha 62a4a339ebc1, PR #2077)
TITLE: docs: fix module docstrings and copyright headers (#2077)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+13/-15); LICENSE (+1/-1); benchmark/lora/lora_bench.py (+13/-14); python/sglang/srt/configs/model_config.py (+13/-14); python/sglang/srt/constrained/__init__.py (+13/-14); python/sglang/srt/constrained/base_grammar_backend.py (+13/-15); python/sglang/srt/constrained/outlines_backend.py (+13/-15); python/sglang/srt/constrained/outlines_jump_forward.py (+13/-15); python/sglang/srt/constrained/xgrammar_backend.py (+13/-15); python/sglang/srt/conversation.py (+13/-15); (+73 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ As per the title, fix module docstrings and copyright headers. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ 1. Use multiline comments instead of triple quoted strings for copyright at the top of the file. ⏎  ⏎ Before: ⏎  ⏎ ```python ⏎ In [1]: import sglang.srt.layers.activation ⏎   ⏎ In [2]: print(sglang.srt.layers.activation.__doc__) ⏎  ⏎ Copyright 2023-2024 SGLang Team ⏎ Licensed under the Apache License, Version 2.0 (the "License"); ⏎ you may not use  …[truncated]

### L2-a78d8f8db3  (L2, 2024-11-23, sha a78d8f8db380, PR #2137)
TITLE: [CI] Fix test cases (#2137)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+4/-2); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+3/-1); python/sglang/srt/model_executor/model_runner.py (+11/-9); test/srt/test_srt_engine.py (+9/-9)
BODY: 

### L2-c5f865013e  (L2, 2024-11-23, sha c5f865013e72, PR #2134)
TITLE: Fix grid size in Triton decoding kernel (#2134)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+34/-38)
BODY: ## Motivation ⏎ Fix issue mentioned in https://github.com/sgl-project/sglang/discussions/1935. ⏎  ⏎ ``` ⏎ python -m sglang.bench_one_batch --batch-size 128 --input 128 --output 128 --model meta-llama/Llama-3.1-8B-Instruct --attention-backend triton ⏎  ⏎ Prefill. latency: 0.37300 s, throughput:  43925.00 token/s ⏎ Decode.  latency: 0.01022 s, throughput:  12529.66 token/s ⏎ Decode.  latency: 0.01028 s, throughput:  12455.53 token/s ⏎ Decode.  latency: 0.01 …[truncated]

### L2-fae4e5e99a  (L2, 2024-11-30, sha fae4e5e99a93, PR #2259)
TITLE: chore: bump v0.3.6.post3 (#2259)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+19/-18); docker/Dockerfile.rocm (+1/-1); python/pyproject.toml (+2/-2); Makefile (+27/-1); docs/developer/setup_github_runner.md (+2/-2); docs/start/install.md (+7/-13); python/sglang/version.py (+1/-1); scripts/ci_install_dependency.sh (+1/-2)
BODY: ## Motivation ⏎  ⏎ install with one-click, set `flashinfer` in deps ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-62c516ac45  (L2, 2024-12-01, sha 62c516ac45a7, PR #2241)
TITLE: Add a simple torch native attention backend (#2241)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/torch_native_backend.py (+285/-0); python/sglang/srt/managers/schedule_batch.py (+18/-14); python/sglang/srt/model_executor/forward_batch_info.py (+9/-4); python/sglang/srt/model_executor/model_runner.py (+3/-0); python/sglang/srt/server_args.py (+14/-8); test/srt/run_suite.py (+1/-0); test/srt/test_torch_native_attention_backend.py (+58/-0)
BODY: ## Motivation ⏎  ⏎ Add a torch_native attention backend, which only relies on PyTorch native operations and fully bypass triton and flashinfer. ⏎ This backend can be used for debug or as an reference impl for devices that not support triton/flashinfer. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-7301a39b13  (L2, 2024-12-01, sha 7301a39b13c7, PR #2305)
TITLE: fix: resolve CodeQL cpp issue (#2305)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.dev (+7/-1); sgl-kernel/CMakeLists.txt (+47/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-ec52464dde  (L2, 2024-12-05, sha ec52464ddeab, PR #2349)
TITLE: MLA prefill w/o weight absorption (#2349)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/__init__.py (+5/-2); python/sglang/srt/layers/attention/double_sparsity_backend.py (+22/-8); python/sglang/srt/layers/attention/flashinfer_backend.py (+20/-5); python/sglang/srt/layers/attention/torch_native_backend.py (+22/-8); python/sglang/srt/layers/attention/triton_backend.py (+22/-8); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+3/-0); python/sglang/srt/layers/radix_attention.py (+4/-2); python/sglang/srt/models/deepseek_v2.py (+68/-3)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ For large batch sizes, not using weight absorption in the MLA prefill phase will be more efficient.  ⏎  ⏎ ``` ⏎ python3 -m sglang.bench_one_batch --batch-size 128 --input 512 --output 1 --model neuralmagic/DeepSeek-Coder-V2-Instruct-FP8 --trust-remote-code --tp 8 --disable-cuda-graph ⏎  ⏎ # prefill w/o absorb (this PR) ⏎ Prefill. latency: 2.17135 s, throughput:  30182.18 token/s ⏎  ⏎ # prefill w/ absorb (main) ⏎ Prefill. latency: 3.29770  …[truncated]

### L2-2db4469808  (L2, 2024-12-05, sha 2db446980815, PR #2350)
TITLE: minor: limit the range of vllm versions (#2350)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/__init__.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-4a63c181f1  (L2, 2024-12-06, sha 4a63c181f190, PR #2364)
TITLE: Fix AWQ with enable MLA (#2364)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+14/-1)
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/issues/2336

### L2-7dc66fcb40  (L2, 2024-12-08, sha 7dc66fcb40aa, PR #2394)
TITLE: Optimize Triton decoding kernel for long context (#2394)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+287/-342); python/sglang/srt/layers/attention/triton_backend.py (+13/-8); python/sglang/srt/server_args.py (+7/-0); test/srt/test_triton_attention_kernels.py (+21/-10)
BODY: ## Motivation ⏎  ⏎ As mentioned in https://github.com/sgl-project/sglang/issues/2271, the original triton decoding kernel has significant performance degradation on long context. We refactored the kernel and adapted the flash decoding implementation from [lightllm](https://github.com/ModelTC/lightllm). Currently, the long context speed decay has been alleviated a lot. ⏎  ⏎ ## Benchmark ⏎ Tested for input 128, output 2048. ⏎  ⏎ Triton (this PR) num_kv_sp …[truncated]

### L2-61dec545b0  (L2, 2024-12-08, sha 61dec545b044, PR #2401)
TITLE: Remove unused vars in the triton backend (#2401)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+10/-10); python/sglang/srt/layers/attention/triton_backend.py (+4/-17); test/srt/test_triton_attention_kernels.py (+0/-6)
BODY: ## Motivation ⏎  ⏎ - Remove unsed vars ⏎ - Add hint log for triton error message

### L2-2f9bd0fafd  (L2, 2024-12-14, sha 2f9bd0fafd7b, PR #2479)
TITLE: Fix correctness issue for triton decoding kernel (#2479)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+24/-14); test/srt/test_triton_attention_kernels.py (+6/-4)
ISSUES: #2465 [Bug] potential correctness with triton-attention-num-kv-splits > 1
BODY: ## Motivation ⏎  ⏎ Fix https://github.com/sgl-project/sglang/issues/2465. The issue is for short prompts.

### L2-e04d3f2897  (L2, 2024-12-15, sha e04d3f289753, PR #2481)
TITLE: adapt tensorrt llm custom all reduce to sgl-kernel (#2481)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+47/-19); sgl-kernel/Makefile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/setup.py (+25/-1); sgl-kernel/src/sgl-kernel/__init__.py (+7/-2); sgl-kernel/src/sgl-kernel/csrc/trt_reduce.cc (+13/-0); sgl-kernel/src/sgl-kernel/csrc/trt_reduce_internal.cu (+282/-0); sgl-kernel/src/sgl-kernel/csrc/trt_reduce_internal.cuh (+91/-0); sgl-kernel/src/sgl-kernel/csrc/trt_reduce_kernel.cu (+102/-0); sgl-kernel/src/sgl-kernel/csrc/utils.hpp (+36/-0); (+3 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ - move tensorrt custom allreduce algoithm to sgl-kernel, make adaption for python, add test for custom allreduce ⏎ - we **do not use twoshot allreduce kernel** from tensorrt llm since it is disabled [here](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/plugins/ncclPlugin/allreducePlugin.cpp#L192) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎  ⏎ ## Next

### L2-4b83db24f1  (L2, 2024-12-19, sha 4b83db24f128, PR #2517)
TITLE: fix: continue to use flashinfer 0.1.6 temporarily (#2517)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎ If using the latest 0.2.0, CI cannot pass. Temporarily fix it to 0.1.6 and wait for FlashInfer to release 0.2.0.post1 or 0.2.1 before lifting the restriction again. ⏎  ⏎ ref https://github.com/sgl-project/sglang/actions/runs/12407074179 ⏎  ⏎ cc @yzh119 @merrymercy  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-60e2fdcf4f  (L2, 2024-12-26, sha 60e2fdcf4fdb, PR #2581)
TITLE: use sgl-kernel moe_align_block_size (#2581)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+20/-3); python/sglang/srt/model_executor/model_runner.py (+6/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-efc52f85e2  (L2, 2024-12-26, sha efc52f85e2d5, PR #2582)
TITLE: chore: bump v0.4.1 (#2582)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.rocm (+1/-1); python/pyproject.toml (+2/-2); docs/developer/setup_github_runner.md (+2/-2); docs/start/install.md (+5/-5); python/sglang/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-635a042623  (L2, 2024-12-26, sha 635a04262396, PR #2592)
TITLE: docs: update deepseek v3 example (#2592)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); benchmark/deepseek_v3/README.md (+30/-0); python/sglang/srt/layers/moe/topk.py (+14/-0); python/sglang/srt/layers/quantization/fp8_kernel.py (+14/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-6e5305158c  (L2, 2024-12-28, sha 6e5305158cde, PR #2617)
TITLE: update sgl_moe_align_block_size usage (#2617)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+10/-2); python/sglang/srt/model_executor/model_runner.py (+0/-6)
BODY: 

### L2-b02da24a5b  (L2, 2024-12-30, sha b02da24a5b8c, PR #2642)
TITLE: Refactor sgl-kernel build (#2642)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+6/-23); sgl-kernel/setup.py (+34/-67); sgl-kernel/src/sgl-kernel/__init__.py (+11/-1); sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+2/-6); sgl-kernel/src/sgl-kernel/csrc/sgl_kernel_ops.cu (+32/-0); sgl-kernel/src/sgl-kernel/csrc/trt_reduce.cc (+0/-13); sgl-kernel/src/sgl-kernel/csrc/warp_reduce.cc (+0/-14); sgl-kernel/src/sgl-kernel/csrc/warp_reduce_kernel.cu (+2/-1); sgl-kernel/src/sgl-kernel/ops/__init__.py (+21/-1)
BODY: ## Motivation ⏎  ⏎ Involve all ops into one target. Reduce redundant build code.

### L2-c5210dfa38  (L2, 2024-12-30, sha c5210dfa3802, PR #2667)
TITLE: AMD DeepSeek_V3 FP8 Numerical fix (#2667)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+34/-7)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Together with core changes from: https://github.com/sgl-project/sglang/pull/2637 ⏎  ⏎ ``` ⏎ # python3 -m sglang.launch_server --model /data2/DeepSeek-V3/ --tp 8 --trust-remote-code ⏎  ⏎ # python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 ⏎ 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ …[truncated]

### L2-b4403985d0  (L2, 2024-12-31, sha b4403985d009, PR #2676)
TITLE: Add cutlass submodule for sgl-kernel (#2676)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+3/-0); sgl-kernel/CMakeLists.txt (+4/-0); sgl-kernel/3rdparty/cutlass (+1/-0); sgl-kernel/setup.py (+6/-0)
BODY: ## Motivation ⏎  ⏎ Include CUTLASS (v3.6.0) for kernels dev.

### L2-b6b57fc200  (L2, 2024-12-31, sha b6b57fc20075, PR #2679)
TITLE: minor: cleanup sgl-kernel (#2679)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+0/-1); sgl-kernel/setup.py (+0/-37); sgl-kernel/src/sgl-kernel/__init__.py (+0/-2); sgl-kernel/src/sgl-kernel/csrc/sgl_kernel_ops.cu (+0/-10); sgl-kernel/src/sgl-kernel/csrc/warp_reduce_kernel.cu (+0/-90); sgl-kernel/src/sgl-kernel/ops/__init__.py (+0/-5)
BODY: ## Motivation ⏎  ⏎ Warp reduce was initially added as an example and is no longer necessary. The hack logic for renaming in setup is also redundant now. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-c7ae474a49  (L2, 2025-01-02, sha c7ae474a49f9, PR #2601)
TITLE: [Feature, Hardware] Enable DeepseekV3 on AMD GPUs (#2601)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+4/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+5/-5)
LABELS: bug, high priority, amd
BODY: ## Motivation ⏎ - Support DeepseekV3 on AMD Instinct MI300X GPU ⏎  ⏎ ## Modifications ⏎ - Add proper fix for AMD FP8 ```e4m3fnuz``` to support DeepseekV3 FP8 model ⏎ - Bypass ```FlashInfer backend bmm_fp8``` to cast FP8 to BF16 in MLA ⏎ - Add AMD ```triton stages``` config ⏎ ### TODO ⏎  ⏎  ⏎  ⏎ ## How to run ⏎ **build env** ⏎ ``` ⏎ cd sglang/docker ⏎  ⏎ docker build –t sglang-rocm:latest –f Dockerfile.rocm . ⏎   ⏎ docker run -it --ipc=host \  ⏎                --cap- …[truncated]

### L2-2f0d386496  (L2, 2025-01-06, sha 2f0d38649623, PR #2713)
TITLE: chore: bump v0.4.1.post4 (#2713)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.rocm (+1/-1); python/pyproject.toml (+2/-2); docs/developer/setup_github_runner.md (+2/-2); docs/start/install.md (+5/-5); python/sglang/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ - EAGLE-2 ⏎ - DeepSeek V3 36 tokens/s ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-0f3eb1d294  (L2, 2025-01-06, sha 0f3eb1d29404, PR #2752)
TITLE: Support cutlass Int8 gemm (#2752)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/benchmark/bench_int8_gemm.py (+55/-0); sgl-kernel/setup.py (+2/-0); sgl-kernel/src/sgl-kernel/__init__.py (+2/-0); sgl-kernel/src/sgl-kernel/csrc/cutlass_extensions/epilogue/epilogue_per_row_per_col_scale.h (+278/-0); sgl-kernel/src/sgl-kernel/csrc/cutlass_extensions/gemm/gemm_universal_base_compat.h (+346/-0); sgl-kernel/src/sgl-kernel/csrc/cutlass_extensions/gemm/gemm_with_epilogue_visitor.h (+456/-0); sgl-kernel/src/sgl-kernel/csrc/int8_gemm_kernel.cu (+209/-0); sgl-kernel/src/sgl-kernel/csrc/sgl_kernel_ops.cu (+7/-0); sgl-kernel/src/sgl-kernel/csrc/utils.hpp (+10/-0); (+2 more)
BODY: ## Motivation ⏎  ⏎ Support fused int8 gemm for W8A8 quantization. ⏎  ⏎ Tested on A100 with benchmark script `benchmark/bench_int8_gemm.py` (measured with GB/s): ⏎ N = 4096, K = 8192 ⏎ ``` ⏎    batch_size  vllm int8 gemm  sgl-kernel int8 gemm ⏎ 0         1.0     1502.728206           1604.403733 ⏎ 1        16.0    24009.266126          26119.766434 ⏎ 2        32.0    48732.887699          51184.391083 ⏎ 3        64.0    91677.707776          94582.986851 ⏎ 4  …[truncated]

### L2-bdc1acf6cd  (L2, 2025-01-07, sha bdc1acf6cdad, PR #2761)
TITLE: Misc fix for min_p_sampling, --cuda-graph-bs (#2761)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/pyproject.toml (+9/-3); .github/workflows/pr-test.yml (+3/-1); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+1/-0); python/sglang/bench_serving.py (+4/-1); python/sglang/srt/layers/logits_processor.py (+5/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+16/-5); python/sglang/srt/layers/quantization/__init__.py (+1/-2); python/sglang/srt/managers/data_parallel_controller.py (+2/-0); python/sglang/srt/managers/scheduler.py (+3/-2); python/sglang/srt/metrics/collector.py (+22/-30); (+7 more)
BODY: - Support `--cuda-graph-bs` so you can specify the batch size to capture the cuda graph. ⏎ - Fix merge_batch for min_p_sampling ⏎ - Add more utility functions and remove redundant code.

### L2-51caee740f  (L2, 2025-01-07, sha 51caee740fee, PR #2771)
TITLE: Host memory pool for hierarchical caching (#2771)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/memory_pool.py (+206/-1); python/sglang/srt/utils.py (+24/-0)
BODY: ## Motivation ⏎ This is one of the breakdown PR of the [hierarchical caching proposal](https://github.com/sgl-project/sglang/pull/2693). ⏎  ⏎ ## Modifications ⏎ - Interfaces for data transfer between host and devices. ⏎ - `MemoryStateInt` and functions specify memory state transfers. ⏎ - `MLATokenToKVPoolHost` that backs up KV caches in a `MLATokenToKVPoo`. ⏎  ⏎ ## Checklist

### L2-e2b16c4716  (L2, 2025-01-12, sha e2b16c4716f2, PR #2846)
TITLE: add sampling_scaling_penalties kernel (#2846)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/setup.py (+1/-0); sgl-kernel/src/sgl-kernel/__init__.py (+2/-0); sgl-kernel/src/sgl-kernel/csrc/sampling_scaling_penalties.cu (+64/-0); sgl-kernel/src/sgl-kernel/csrc/sgl_kernel_ops.cu (+5/-0); sgl-kernel/src/sgl-kernel/csrc/vectorization.cuh (+30/-0); sgl-kernel/src/sgl-kernel/ops/__init__.py (+7/-0); sgl-kernel/tests/test_sampling_scaling_penalties.py (+39/-0)
BODY: Add a sampling_scaling_penalties kernel to reduce the GPU memory usage during the sampling phase when the scaling_penalties sampling parameter is applied (https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/sampling/sampling_batch_info.py#L247). This can eliminate the need for storing multiple intermediate logits in memory, which is particularly useful when serving long sequences or handling large batch sizes on GPUs with limited me …[truncated]

### L2-67008f4b32  (L2, 2025-01-13, sha 67008f4b320d, PR #2858)
TITLE: Use only one GPU for MLA CI tests (#2858)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+3/-5); test/srt/run_suite.py (+2/-0); test/srt/test_mla.py (+34/-1); test/srt/test_mla_fp8.py (+0/-2)
BODY: 

### L2-6249e4a19e  (L2, 2025-01-13, sha 6249e4a19ed6, PR #2866)
TITLE: Revert "Integration of TurboMind AWQ" (#2866)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/configs/model_config.py (+1/-9); python/sglang/srt/layers/linear.py (+0/-1); python/sglang/srt/layers/quantization/__init__.py (+0/-2); python/sglang/srt/layers/quantization/awq_turbomind.py (+0/-287); python/sglang/srt/layers/quantization/turbomind_utils.py (+0/-63); python/sglang/srt/server_args.py (+0/-1); test/srt/test_turbomind_awq.py (+0/-47)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 2828 reason=build_or_dependency
BODY: Reverts sgl-project/sglang#2828 because it introduces extra dependency and breaks some internal stuff

### L2-17de02f98d  (L2, 2025-01-13, sha 17de02f98d8f, PR #2828)
TITLE: Integration of TurboMind AWQ (#2828)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/configs/model_config.py (+9/-1); python/sglang/srt/layers/linear.py (+1/-0); python/sglang/srt/layers/quantization/__init__.py (+2/-0); python/sglang/srt/layers/quantization/awq_turbomind.py (+287/-0); python/sglang/srt/layers/quantization/turbomind_utils.py (+63/-0); python/sglang/srt/server_args.py (+1/-0); test/srt/test_turbomind_awq.py (+47/-0)
LABELS: enhancement, high priority, quant
DEEP_STUDY: deep-study: this PR was reverted by PR 2866 (confirmed_revert, reason=build_or_dependency)
BODY: ## Motivation ⏎  ⏎ Come from this [issue](https://github.com/sgl-project/sglang/issues/2788) ⏎  ⏎ ## Usage ⏎ ``` ⏎ # use turbomind ⏎ python examples/runtime/engine/offline_batch_inference.py --model=${Meta-Llama-3-8B-Instruct-hf-AWQ} --quantization=awq_turbomind ⏎  ⏎ # use marlin ⏎ python examples/runtime/engine/offline_batch_inference.py --model=${Meta-Llama-3-8B-Instruct-hf-AWQ} ⏎ ``` ⏎  ⏎ ## Checklist

### L2-d08c77c434  (L2, 2025-01-13, sha d08c77c43498, PR #2870)
TITLE: Sampling penalties memory interface (#2870)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); benchmark/kernels/fused_moe_triton/benchmark_deepseekv3_moe_align_blocks.py (+2/-1); python/sglang/srt/sampling/penaltylib/penalizers/repetition_penalty.py (+15/-5); python/sglang/srt/sampling/sampling_batch_info.py (+14/-5); python/sglang/srt/utils.py (+4/-0); sgl-kernel/benchmark/benchmark_sampling_scaling_penalties.py (+159/-0); sgl-kernel/tests/test_moe_align.py (+61/-34)
BODY: Related [pr](https://github.com/sgl-project/sglang/pull/2846) ⏎  ⏎ Result of `benchmark_sampling_scaling_penalties.py` : ⏎  ⏎ performace: ⏎  ⏎ ![图片](https://github.com/user-attachments/assets/122d33ee-bcd8-426b-9357-dc76f9f7f903) ⏎  ⏎ peak memory: ⏎  ⏎ ![图片](https://github.com/user-attachments/assets/72dc772d-8dce-4ec3-b414-2d4d22b5c1a6) ⏎  ⏎ - Fix a bug in `benchmark_moe_align_blocks.py` . ⏎ - Refine sgl-kernel `test_moe_align.py` .

### L2-f005758f2b  (L2, 2025-01-14, sha f005758f2bcf, PR #2887)
TITLE: introduce CUB in sgl-kernel (#2887)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+3/-0); sgl-kernel/CMakeLists.txt (+2/-0); sgl-kernel/3rdparty/cub (+1/-0)
BODY: ![图片](https://github.com/user-attachments/assets/89bde132-d625-4f8a-adf0-df86fe3b2f2a) ⏎  ⏎ @zhyncs

### L2-767c9dec03  (L2, 2025-01-16, sha 767c9dec03e3, PR #2511)
TITLE: adapt custom allreduce for tensorrt llm (#2511)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/_custom_ops.py (+22/-27); python/sglang/srt/distributed/device_communicators/custom_all_reduce.py (+53/-39); test/srt/run_suite.py (+1/-0); test/srt/test_custom_allreduce.py (+164/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ adapt for tensorrt llm custom allreduce, currently still use vllm distributed.  ⏎ After this pr is merged and sgl-kernel is stable, we only need replace vllm.distribued to sglang.srt.distributed, and add a monkey patch, then we can remove vllm distributed ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-7596417732  (L2, 2025-01-16, sha 75964177327c, PR #2919)
TITLE: minor: use bear for compilation database (#2919)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+0/-65); sgl-kernel/Makefile (+8/-5)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-63051738a9  (L2, 2025-01-16, sha 63051738a91e, PR #2806)
TITLE: Enable CPU device on SGLang (#2806)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: python/pyproject.toml (+6/-0); python/sglang/srt/configs/device_config.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_native.py (+69/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+4/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+26/-2); python/sglang/srt/layers/rotary_embedding.py (+248/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/managers/scheduler.py (+2/-0); python/sglang/srt/managers/tp_worker_overlap_thread.py (+2/-0); python/sglang/srt/model_executor/model_runner.py (+9/-3); (+3 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR enables CPU device on SGLang. ⏎ Currently we fallback attention and MoE to the torch native backend and make the functionality work on CPU. ⏎ We will submit follow-up PRs to provide optimized kernels to further improvement the performance. ⏎  ⏎ For vllm installation for CPU, users could follow the instruction provided by vllm [here](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html). ⏎  ⏎ ## Modific …[truncated]

### L2-d33cbb7e58  (L2, 2025-01-19, sha d33cbb7e5857, PR #2976)
TITLE: remove cub and add cccl (#2976)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+3/-3); sgl-kernel/3rdparty/cccl (+1/-0); sgl-kernel/3rdparty/cub (+0/-1)
BODY: ## Motivation ⏎  ⏎ The cub repo has been archived, use cccl instead. cc @BBuf  ⏎  ⏎ ``` ⏎ sglang git:(zhyncs/sub) git submodule status ⏎  b5fe509fd11a925f90d6495176707cc1184eed9d sgl-kernel/3rdparty/cccl (v2.7.0) ⏎  b78588d1630aa6643bf021613717bafb705df4ef sgl-kernel/3rdparty/cutlass (v3.7.0) ⏎ ``` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2c05f81f15  (L2, 2025-01-20, sha 2c05f81f157f, PR #2988)
TITLE: fix custom op version compatibility (#2988)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/rotary_embedding.py (+3/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-5a0d680a14  (L2, 2025-01-21, sha 5a0d680a14fc, PR #3033)
TITLE: feat: add flashinfer as 3rdparty and use rmsnorm as example (#3033)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+3/-0); .github/workflows/pr-test-sgl-kernel.yml (+1/-0); .gitignore (+2/-0); sgl-kernel/3rdparty/flashinfer (+1/-0); sgl-kernel/THIRDPARTYNOTICES.txt (+225/-0); sgl-kernel/setup.py (+19/-2); sgl-kernel/src/sgl-kernel/__init__.py (+2/-0); sgl-kernel/src/sgl-kernel/csrc/norm.cu (+28/-0); sgl-kernel/src/sgl-kernel/csrc/sgl_kernel_ops.cu (+5/-0); sgl-kernel/src/sgl-kernel/ops/__init__.py (+18/-0); (+1 more)
BODY: ## Motivation ⏎  ⏎ ``` ⏎ sglang git:(zhyncs/flashinfer) git submodule status ⏎  b5fe509fd11a925f90d6495176707cc1184eed9d sgl-kernel/3rdparty/cccl (v2.7.0) ⏎  b78588d1630aa6643bf021613717bafb705df4ef sgl-kernel/3rdparty/cutlass (v3.7.0) ⏎  a0e99a3a820109763d9a757138a5cdf7bbcd1f85 sgl-kernel/3rdparty/flashinfer (v0.0.2-422-ga0e99a3) ⏎ ``` ⏎  ⏎ ``` ⏎ sgl-kernel git:(zhyncs/flashinfer) pytest tests/test_rmsnorm.py ⏎ ============================================= …[truncated]

### L2-153b414e83  (L2, 2025-01-24, sha 153b414e835e, PR #3105)
TITLE: minor: sync flashinfer and add turbomind as 3rdparty (#3105)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+3/-0); sgl-kernel/3rdparty/flashinfer (+1/-1); sgl-kernel/3rdparty/turbomind (+1/-0); sgl-kernel/developer_guide.md (+1/-0)
BODY: ## Motivation ⏎  ⏎ cc @lzhangzz @lvhan028 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-7e0976133c  (L2, 2025-01-26, sha 7e0976133ca4, PR #3150)
TITLE: udpate sgl-kernel version for srt (#3150)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2f79f58873  (L2, 2025-01-27, sha 2f79f58873b0, PR #3179)
TITLE: feat: use sgl-kernel 0.0.3 in sglang (#3179)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/activation.py (+5/-5); python/sglang/srt/layers/layernorm.py (+5/-5); python/sglang/srt/layers/sampler.py (+4/-8); python/sglang/srt/models/deepseek_v2.py (+3/-3); python/sglang/srt/models/minicpm3.py (+3/-3)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-cf142b6eb8  (L2, 2025-01-27, sha cf142b6eb87a, PR #3181)
TITLE: fix: update Dockerfile for cu118 (#3181)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+3/-0)
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/actions/runs/12992119766 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-b49d6d0fee  (L2, 2025-01-31, sha b49d6d0fee3c, PR #3231)
TITLE: support 12.5 CUDA runtime (#3231)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+6/-0); .github/workflows/release-docker.yml (+4/-2)
BODY: ## Motivation ⏎  ⏎ follow-up for https://github.com/sgl-project/sglang/pull/3230 ⏎  ⏎ https://github.com/sgl-project/sglang/actions/runs/13072042144 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-34e405e01f  (L2, 2025-02-01, sha 34e405e01f7f, PR #3238)
TITLE: update sgl-kernel version for sglang (#3238)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-d9eb9358cc  (L2, 2025-02-01, sha d9eb9358ccf8, PR #3255)
TITLE: Tune paged attention parameters for AMD GPU. (#3255)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+9/-2); python/sglang/srt/server_args.py (+4/-0)
LABELS: good first issue
BODY: ## Motivation ⏎  ⏎ Fine tune SGLang page attention kernel performance on AMD MI GPU for LLM. ⏎  ⏎ ## Modifications ⏎  ⏎ Changes: ⏎ - num_kv_splits : 8 -> 16 ⏎ - BLOCK : 64 -> 8 ⏎ - num_warps : 2 -> 1 when kv_group_num is more than 1 ⏎ - waves_per_cu : 4 -> 1 in grouped paged attention kernel ⏎  ⏎ These knobs have been tested with a couple of workloads on AMD ROCm platform.

### L2-566d61d90f  (L2, 2025-02-03, sha 566d61d90fd5, PR #3259)
TITLE: ROCm: bump 6.3.0 (#3259)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.rocm (+2/-2); python/pyproject.toml (+8/-10); .github/workflows/release-docker-amd.yml (+3/-3); docs/developer/setup_github_runner.md (+2/-2); docs/start/install.md (+3/-3); python/sglang/srt/constrained/outlines_backend.py (+9/-1); python/sglang/srt/custom_op.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ Latest ROCm ⏎  ⏎ ## Modifications ⏎  ⏎ As it is. ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Code Formatting with Pre-Commit](https://docs.sglang.ai/references/contribution_guide.html#code-formatting-with-pre-commit). ⏎ - [+] Add unit tests as outlined in the [Running Unit Tests](https://docs.sglang.ai/references/contribution_guide.html#running-unit-tests-adding-to-ci). ⏎ - [+] Update documentation / docstrings / exampl …[truncated]

### L2-d39899e85c  (L2, 2025-02-04, sha d39899e85c5c, PR #3288)
TITLE: upgrade flashinfer v0.2.0.post2 (#3288)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.backend.flashinfer_general_mla
FILES: python/pyproject.toml (+1/-1); .github/workflows/pr-test.yml (+8/-8); python/sglang/srt/entrypoints/engine.py (+2/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+20/-36); python/sglang/srt/speculative/eagle_utils.py (+5/-0); python/sglang/srt/speculative/eagle_worker.py (+2/-0); scripts/ci_install_dependency.sh (+4/-3); test/srt/run_suite.py (+0/-1)
BODY: ## Motivation ⏎  ⏎ - The flashinfer package name has been changed to flashinfer_python. cc @yzh119  ⏎ - Resolve the eagle-related issue. cc @pankajroark ⏎ - Temporarily disable the fp8 kv cache test (will be fixed in a follow-up) to facilitate the upgrade. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2c1a695ff1  (L2, 2025-02-04, sha 2c1a695ff111, PR #3287)
TITLE: ROCm: sgl-kernel enablement starting with sgl_moe_align_block (#3287)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.rocm (+3/-0); python/pyproject.toml (+1/-1); docs/start/install.md (+3/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+3/-11); sgl-kernel/setup_rocm.py (+92/-0); sgl-kernel/src/sgl-kernel/torch_extension_rocm.cc (+29/-0)
BODY: ## Motivation ⏎  ⏎ 1. Enable sgl-kernel on ROCm, make it easy to add individual kernels ⏎ 2. Use sgl_moe_align_block kernel for fused_moe for performance ⏎  ⏎ ## Modifications ⏎  ⏎ As they are. ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Code Formatting with Pre-Commit](https://docs.sglang.ai/references/contribution_guide.html#code-formatting-with-pre-commit). ⏎ - [+] Add unit tests as outlined in the [Running Unit Tests](https://docs.sgl …[truncated]

### L2-a07364ccc5  (L2, 2025-02-04, sha a07364ccc500, PR #3292)
TITLE: Update Triton decode backend interface (#3292)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+44/-57); python/sglang/srt/layers/attention/triton_backend.py (+71/-7); test/srt/test_triton_attention_kernels.py (+14/-13)
BODY: ## Motivation ⏎  ⏎ Use `kv_indptr` and `kv_indices` for more flexible kv cache selection and get unified with flashinfer interface.

### L2-6186a8f889  (L2, 2025-02-05, sha 6186a8f8897e, PR #3293)
TITLE: update flashinfer install index url (#3293)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+8/-8); docs/start/install.md (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f287037673  (L2, 2025-02-07, sha f287037673a3, PR #3374)
TITLE: update sgl-kernel version (#3374)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-85986bb978  (L2, 2025-02-10, sha 85986bb97808, PR #3435)
TITLE: compatible with new outlines (#3435)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/constrained/outlines_backend.py (+4/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-cddb1cdf8f  (L2, 2025-02-10, sha cddb1cdf8fd8, PR #3459)
TITLE: chore: bump v0.4.2.post4 (#3459)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.rocm (+1/-1); python/pyproject.toml (+2/-2); benchmark/deepseek_v3/README.md (+1/-1); docs/developer/setup_github_runner.md (+2/-2); docs/start/install.md (+6/-6); python/sglang/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ - bunch of EAGLE 2 fixes ⏎ - FlashInfer cleanup and enable ragged FA3 by default ⏎ - update base image for better multi node support ⏎ - compatible with new outlines ⏎  ⏎ BTW v0.4.3 in on the way :) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-4fe92bfca5  (L2, 2025-02-10, sha 4fe92bfca551, PR #3469)
TITLE: fix mla test (#3469)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-amd.yml (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ fix https://github.com/sgl-project/sglang/actions/runs/13237584638/job/36949256746 ⏎  ⏎ ref https://github.com/pytorch/pytorch/blob/6f15a609d395e56ffa433f8c9efa231579bb5fee/.github/workflows/auto_request_review.yml#L9 ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-45e3a7bc41  (L2, 2025-02-12, sha 45e3a7bc41d7, PR #3493)
TITLE: use sgl_per_token_group_quant_fp8 kernel (#3493)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+8/-1); python/sglang/srt/layers/quantization/fp8_kernel.py (+34/-0)
BODY: end2end perfomance: ⏎  ⏎ ```shell ⏎ ```shell ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000 ⏎  ⏎ # Run 2 times commands: ⏎  ⏎ python3 -m sglang.bench_serving --backend sglang --num-prompts 1000 --request-rate 8 ⏎ ``` ⏎  ⏎  ⏎ ### first time ⏎  ⏎ main: ⏎  ⏎ ```shell ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang     ⏎ Traffic request rate:                …[truncated]

### L2-98eecbda54  (L2, 2025-02-13, sha 98eecbda54d5, PR #3529)
TITLE: integrate blockwise fp8 kernel (#3529)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/quantization/fp8_kernel.py (+90/-18); python/sglang/srt/layers/quantization/fp8_utils.py (+33/-4)
BODY: ## Motivation ⏎  ⏎  ⏎ Integrate #3267 into python side to optimize deepseekv3, merge it after #3267 has merged and sgl kernel is released ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Test Method ⏎ Following #3486, test on gsm8k and mmlu ⏎ |    | latency | accuracy | ⏎ |  ----  | ----  | --- | ⏎ | gsm8k (before) | 97.148s | 0.953 | ⏎ | gsm8k (after) | 82.708s | 0.958 | ⏎ | mmlu (before) | 250.839s | 0.871| ⏎ | mmlu (after) | 240.800s | 0.871| ⏎  ⏎ ## TODO ⏎  ⏎  ⏎ ## Checklist

### L2-70f894b810  (L2, 2025-02-14, sha 70f894b810c0, PR #3550)
TITLE: feat: support flashinfer mla attention for deepseek v3 (#3550)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults, L2.backend.flashinfer_general_mla
FILES: python/pyproject.toml (+3/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+234/-109); python/sglang/srt/model_executor/model_runner.py (+12/-2); python/sglang/srt/models/deepseek_v2.py (+13/-7); python/sglang/srt/server_args.py (+7/-0); .github/workflows/pr-test.yml (+8/-8); python/sglang/global_config.py (+2/-0); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/utils.py (+7/-0); (+2 more)
BODY: ## Motivation ⏎  ⏎ Kudos to @yzh119 Throughout the integration process, we have identified and resolved numerous issues with the exceptional support from the FlashInfer team. Currently, **SGLang is the first open-source LLM inference engine to incorporate FlashInfer's new MLA Attention into the LLM engine among all frameworks.** ⏎  ⏎ ref https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.2.1 ⏎  ⏎ **This version should use `--enable-flashinfer- …[truncated]

### L2-e0b9a423c8  (L2, 2025-02-14, sha e0b9a423c841, PR #3556)
TITLE: chore: bump v0.4.3 (#3556)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.rocm (+1/-1); python/pyproject.toml (+2/-2); benchmark/deepseek_v3/README.md (+1/-1); docs/developer/setup_github_runner.md (+2/-2); docs/start/install.md (+6/-6); python/sglang/version.py (+1/-1); test/srt/test_eagle_infer.py (+2/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-ac963be234  (L2, 2025-02-14, sha ac963be234cb, PR #3557)
TITLE: update flashinfer-python (#3557)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+8/-8); benchmark/deepseek_v3/README.md (+1/-1); docs/start/install.md (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-714f3e6362  (L2, 2025-02-18, sha 714f3e636279, PR #3643)
TITLE: feat: support flashinfer mla with prefix cache (#3643)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_backend.py (+101/-30); python/sglang/srt/model_executor/model_runner.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+5/-2); python/sglang/srt/managers/schedule_batch.py (+1/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ This version should only use `--enable-flashinfer-mla`. ⏎  ⏎ For other LLM engines, if you refer to this PR, please include "Adapted from https://github.com/sgl-project/sglang/pull/3643/files", thank you :-) ⏎  ⏎ ref https://github.com/sgl-project/sglang/pull/3550 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-75d171a9c5  (L2, 2025-02-18, sha 75d171a9c59b, PR #3644)
TITLE: chore: update flashinfer v0.2.1.post2 (#3644)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); docs/start/install.md (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e5ce395a6c  (L2, 2025-02-18, sha e5ce395a6cb8, PR #3676)
TITLE: Fix draft decode max batch size (#3676)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+3/-0); python/sglang/srt/layers/attention/flashinfer_backend.py (+1/-1); python/sglang/srt/layers/attention/triton_backend.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ For draft decode in spec, the max batch size should be req_to_token_pool.size*topk.

### L2-1df6eabd5d  (L2, 2025-02-21, sha 1df6eabd5d36, PR #3740)
TITLE: feat: Add SageMaker support (#3740)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.sagemaker (+78/-0); docker/serve (+31/-0); python/sglang/srt/entrypoints/http_server.py (+12/-0); test/srt/test_sagemaker_server.py (+178/-0)
BODY: ## Motivation ⏎  ⏎ SageMaker Endpoints support /ping for healthchecks and /invocations for invocation payloads however sglang currently doesn't support this invocation pattern to make the package usable on SageMaker Endpoints. ⏎  ⏎ ## Modifications ⏎  ⏎ This pull request adds two endpoints for `/ping`/ and `/invocations` in `http_server.py`. ⏎  ⏎ `/ping` provides the same functionality as `/health`. At present `/invocations` acts the same as `/v1/chat/co …[truncated]

### L2-27a46317b6  (L2, 2025-02-24, sha 27a46317b648, PR #3813)
TITLE: Fix dependency (#3813)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/pyproject.toml (+35/-13); python/sglang/srt/constrained/outlines_backend.py (+3/-9); python/sglang/srt/layers/sampler.py (+3/-3); python/sglang/srt/server_args.py (+0/-4); test/lang/test_srt_backend.py (+1/-1); test/srt/models/test_qwen_models.py (+1/-1)
BODY: One item per row instead of packing multiple together

### L2-b110084654  (L2, 2025-02-24, sha b110084654a1, PR #3785)
TITLE: Refactor flashinfer logic for deepseek v3 and fix accuracy bug (#3785)
SOURCES: path_core, symbol_pickaxe, corpus:production-kernel-provenance
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+521/-0); python/sglang/srt/configs/model_config.py (+20/-0); python/sglang/srt/model_executor/model_runner.py (+5/-2); python/sglang/srt/models/deepseek_v2.py (+19/-17)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ `flashinfer_backend.py` for attention is too complex, this PR extract the logic of MLA and creates a new `flashinfer_mla_backend.py` ⏎  ⏎ Also, #3716 #3751 reports an accuracy bug when enabling flashinfer mla. This PR solves this bug by correctly handling rope scaling with yarn, with the help of @yzh119. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Define `FlashInferMLAAttnBackend` in `flashinfer_mla_backend.py` by removing codes irrelevant to MLA …[truncated]

### L2-6ce9dbe828  (L2, 2025-02-24, sha 6ce9dbe82882, PR #3237)
TITLE: [ROCm] Enable Fused MLA Triton kernel for DeepSeekV3  (#3237)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.kernel.rocm_mla_decode_rope
FILES: python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py (+446/-0); python/sglang/srt/models/deepseek_v2.py (+159/-1); test/srt/test_triton_attention_rocm_mla.py (+258/-0)
BODY: This PR introduces the concept of fusing MLA rope into the grouped attention on ROCm.  ⏎ To use this feature use the env variable : SGLANG_ROCM_FUSED_DECODE_MLA=1.  ⏎  ⏎ Triton Kernel authors: @juuso-oskari (Korhonen, Juuso), @Chi-Chu319 (Tianxing Wu) and @vgokhale (Gokhale Vinayak)

### L2-8b681d7724  (L2, 2025-02-26, sha 8b681d7724a6, PR #3898)
TITLE: [Rocm] Fix to the rocm_mla_decode_rope.py returning random result (#3898)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.rocm_mla_decode_rope
FILES: python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ This PR fixes the roped fused decode returning random result. A additional condition is added to the sequence length loop in the kernel ⏎  ⏎ ## Checklist

### L2-71ed01833d  (L2, 2025-02-26, sha 71ed01833dd7, PR #3907)
TITLE: [doc] Update document for flashinfer mla (#3907)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docs/backend/server_arguments.md (+1/-0); docs/references/deepseek.md (+1/-1)
BODY: ## Motivation ⏎  ⏎ Follow up pr of #3785 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Add instruction for `--enable-flashinfer-mla` argument ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-3e02526b1f  (L2, 2025-02-27, sha 3e02526b1ff7, PR #3925)
TITLE: [Doc] Add experimental tag for flashinfer mla (#3925)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docs/backend/server_arguments.md (+1/-1); docs/references/deepseek.md (+1/-1)
BODY: ## Motivation ⏎  ⏎ Add an experimental tag for `enable-flashinfer-mla` feature. ⏎ `In Experiment` tag should be removed in the next PR. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-564bdf29f7  (L2, 2025-02-27, sha 564bdf29f7ef, PR #3934)
TITLE: upgrade flashinfer v0.2.2.post1 (#3934)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); docs/start/install.md (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-90bc26a813  (L2, 2025-02-27, sha 90bc26a813ac, PR #3950)
TITLE: set a strict sgl-kernel version (#3950)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎ Required by mercy. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-77a6c9d229  (L2, 2025-02-28, sha 77a6c9d22927, PR #3963)
TITLE: Remove unused imports from rocm mla kernel.  (#3963)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.kernel.rocm_mla_decode_rope
FILES: python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py (+0/-7)
BODY: Based on #3938  ⏎ Removes unused imports in the ROCm fused mla triton kernel.

### L2-f3b99f73b3  (L2, 2025-02-28, sha f3b99f73b391, PR #)
TITLE: update flashinfer-python version
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L2-90a4b7d98a  (L2, 2025-02-28, sha 90a4b7d98a5c, PR #3967)
TITLE: [Feature]Support ragged prefill in flashinfer mla backend (#3967)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_backend.py (+122/-314); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+59/-81); python/sglang/srt/model_executor/model_runner.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+3/-2); python/sglang/srt/server_args.py (+6/-0); docs/backend/server_arguments.md (+2/-1); python/sglang/srt/managers/schedule_batch.py (+1/-0); test/srt/run_suite.py (+1/-0); test/srt/test_mla_flashinfer.py (+104/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Current flashinfer mla backend only uses MLA paged wrapper for both prefilling and decoding. This PR supports the usage of Ragged prefill wrapper during prefilling with empty cache, and adds an argument that controls this usage. ⏎  ⏎  ⏎  ⏎ ## Usage ⏎  ⏎ By default, ragged prefill wrapper is used in flashinfer mla backend. ⏎ To turn it off and use only mla paged wrapper: ⏎ ```bash ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepS …[truncated]

### L2-fa56106731  (L2, 2025-03-02, sha fa5610673196, PR #3987)
TITLE: Add fast decode plan for flashinfer mla (#3987)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/__init__.py (+2/-4); python/sglang/srt/layers/attention/flashinfer_backend.py (+4/-2); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+118/-38); python/sglang/srt/layers/attention/triton_backend.py (+4/-2); python/sglang/srt/model_executor/cuda_graph_runner.py (+12/-4); python/sglang/srt/model_executor/forward_batch_info.py (+5/-0); docs/backend/server_arguments.md (+1/-1); docs/references/deepseek.md (+1/-1); python/sglang/srt/managers/schedule_batch.py (+9/-0)
LABELS: high priority
DEEP_STUDY: deep-study: this PR was reverted by PR 4008 (confirmed_revert, reason=performance_regression)
BODY: ## Motivation ⏎  ⏎ When using flashinfer mla backend and cuda graph together, graph replay will be hanged due to transmission of indptr tensors between cpu and gpu in `BatchMLAPagedAttentionWrapper.plan`. ⏎  ⏎ This PR fixes this issue by adding a new `decode_seq_len_cpu` in forward batch and customizing a faster decode plan for graph replaying.  ⏎  ⏎ Also, some issues (#3906, #3917) points out current flashinfer mla backend behaves worse than triton in …[truncated]

### L2-9e1014cf99  (L2, 2025-03-02, sha 9e1014cf9941, PR #4008)
TITLE: Revert "Add fast decode plan for flashinfer mla" (#4008)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/__init__.py (+4/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+2/-4); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+38/-118); python/sglang/srt/layers/attention/triton_backend.py (+2/-4); python/sglang/srt/model_executor/cuda_graph_runner.py (+4/-12); python/sglang/srt/model_executor/forward_batch_info.py (+0/-5); docs/backend/server_arguments.md (+1/-1); docs/references/deepseek.md (+1/-1); python/sglang/srt/managers/schedule_batch.py (+0/-9)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 3987 reason=performance_regression
BODY: Reverts sgl-project/sglang#3987 because it introduces regression for other cases.

### L2-935cda944b  (L2, 2025-03-03, sha 935cda944b82, PR #4032)
TITLE: Misc clean up; Remove the support of jump forward (#4032)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-2); docs/backend/function_calling.ipynb (+1/-1); docs/backend/sampling_params.md (+257/-46); docs/backend/server_arguments.md (+1/-2); docs/backend/speculative_decoding.ipynb (+3/-3); docs/references/contribution_guide.md (+1/-1); docs/references/multi_node.md (+1/-1); docs/start/install.md (+7/-7); examples/runtime/engine/offline_batch_inference_eagle.py (+1/-1); (+31 more)
BODY: - Remove jump forward to simplify the code maintenance  ⏎ - Rename `function_call` to `parse_function_call` ⏎ - Rename python/sglang/srt/layers/attention/__init__.py  -> python/sglang/srt/layers/attention/base_attn_backend.py ⏎ - Revert https://github.com/sgl-project/sglang/pull/3260. We need good type annotation and examples to demonstrate how to use these parameters. ⏎ - Do not import `from sglang.lang.chat_template import get_chat_template_by_mode …[truncated]

### L2-d3d4d76758  (L2, 2025-03-05, sha d3d4d76758b1, PR #3986)
TITLE: [Eagle] Refactor eagle speculative decoding (#3986)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.dispatch.server_args_defaults, L2.backend.flashinfer_general_mla
FILES: python/sglang/bench_one_batch.py (+2/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+81/-58); python/sglang/srt/layers/attention/triton_backend.py (+1/-0); python/sglang/srt/managers/cache_controller.py (+2/-2); python/sglang/srt/managers/schedule_batch.py (+26/-24); python/sglang/srt/managers/schedule_policy.py (+19/-14); python/sglang/srt/managers/scheduler.py (+31/-26); python/sglang/srt/managers/tokenizer_manager.py (+0/-17); python/sglang/srt/managers/tp_worker.py (+6/-1); python/sglang/srt/managers/tp_worker_overlap_thread.py (+1/-1); (+12 more)
LABELS: high priority
BODY: Prefix caching and chunked prefill will be compatible with eagle speculative decoding after this PR.

### L2-fc91d08a8f  (L2, 2025-03-05, sha fc91d08a8f0e, PR #4012)
TITLE: [Revision] Add fast decode plan for flashinfer mla  (#4012)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/base_attn_backend.py (+1/-0); python/sglang/srt/layers/attention/flashinfer_backend.py (+2/-0); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+116/-32); python/sglang/srt/layers/attention/triton_backend.py (+2/-0); python/sglang/srt/model_executor/cuda_graph_runner.py (+8/-0); python/sglang/srt/model_executor/forward_batch_info.py (+5/-0); docs/backend/server_arguments.md (+1/-1); docs/references/deepseek.md (+1/-1); python/sglang/srt/managers/schedule_batch.py (+9/-0)
BODY: ## Motivation ⏎  ⏎ Revision of #3987  ⏎  ⏎ @merrymercy @zhyncs  @Ying1123  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-98c73d71cb  (L2, 2025-03-06, sha 98c73d71cb7d, PR #4132)
TITLE: [Minor] make the `__init__` function of model_runner.py shorter (#4132)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.runner.cuda_graph_mla
FILES: python/sglang/srt/model_executor/cuda_graph_runner.py (+1/-1); python/sglang/srt/model_executor/model_runner.py (+86/-69)
BODY: 

### L2-0beea4503f  (L2, 2025-03-07, sha 0beea4503f4f, PR #4178)
TITLE: ROCm: Flex Attention Enablement with custom backends (#4178)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.aiter_mla, L2.dispatch.server_args_defaults
FILES: docker/Dockerfile.rocm (+3/-2); python/sglang/srt/layers/attention/aiter_backend.py (+605/-0); python/sglang/srt/layers/attention/aiter_decode_backend.py (+535/-0); python/sglang/srt/model_executor/model_runner.py (+59/-27); python/sglang/srt/server_args.py (+17/-7); sgl-kernel/src/sgl-kernel/csrc/moe/moe_align_kernel.hip (+118/-0); sgl-kernel/src/sgl-kernel/include/utils_hip.h (+98/-0)
DEEP_STUDY: deep-study: this PR was reverted by PR 4186 (confirmed_revert, reason=premature_or_process)
BODY: Credits: @poyenc , @amd-hhashemi , @linsun12 , @tenpercent , @carlushuang , @HaiShaw  ⏎  ⏎ ## Motivation ⏎  ⏎ Add ROCm custom attention backends to enable Flex Attn. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ `--attention-backend aiter`: prefill: flashinfer-mod + decode: aiter decode (flex attn feasible) ⏎ `--attention-backend aiter_decode`:  prefill: triton + decode: aiter decode ⏎  ⏎ ROCm default backend: triton (no need to assign, flex attn infeasible) ⏎ `--attention …[truncated]

### L2-eb61f5c9af  (L2, 2025-03-07, sha eb61f5c9af73, PR #4186)
TITLE: Revert "ROCm: Flex Attention Enablement with custom backends (#4178)" (#4186)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.aiter_mla, L2.dispatch.server_args_defaults
FILES: docker/Dockerfile.rocm (+2/-3); python/sglang/srt/layers/attention/aiter_backend.py (+0/-605); python/sglang/srt/layers/attention/aiter_decode_backend.py (+0/-535); python/sglang/srt/model_executor/model_runner.py (+27/-59); python/sglang/srt/server_args.py (+7/-17); sgl-kernel/src/sgl-kernel/csrc/moe/moe_align_kernel.hip (+0/-118); sgl-kernel/src/sgl-kernel/include/utils_hip.h (+0/-98)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 4178 reason=premature_or_process
BODY: This reverts commit 0beea4503f4f23f6f4348748c610af4c4233d2fc. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-08c4d764a5  (L2, 2025-03-08, sha 08c4d764a51e, PR #4200)
TITLE: lazy import attn backends (#4200)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/triton_backend.py (+1/-3); python/sglang/srt/model_executor/cuda_graph_runner.py (+1/-1); python/sglang/srt/model_executor/model_runner.py (+18/-6); test/srt/test_eagle_infer.py (+1/-1)
BODY: 

### L2-4a893d142d  (L2, 2025-03-08, sha 4a893d142ded, PR #3749)
TITLE: Refactor Dockerfile: unify CUDA logic and reduce image size by ~2.6 GB (#3749)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+7/-32)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR refactors the Dockerfile to unify the handling of various CUDA versions into a more concise approach. Additionally, it removes redundant layers and streamlines installations, resulting in a reduced final Docker image size by approximately 2.6 GB. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Consolidated multiple if-else statements for CUDA version checks into a single RUN block with exported variables. ⏎ - Removed duplicate or unneede …[truncated]

### L2-ee132a4515  (L2, 2025-03-08, sha ee132a451551, PR #4222)
TITLE: use latest sgl-kernel for mla test (#4222)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-sgl-kernel.yml (+32/-1)
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/actions/runs/13745325002 ⏎  ⏎ Since dsv3 is crucial, add this to ensure that the latest sgl-kernel does not disrupt dsv3. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-89ccb533ad  (L2, 2025-03-08, sha 89ccb533ad39, PR #4224)
TITLE: use sgl-kernel 0.0.4 (#4224)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-9fb48f951f  (L2, 2025-03-09, sha 9fb48f951f86, PR #4218)
TITLE: Support nextn for flashinfer mla attention backend (#4218)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+317/-57); python/sglang/srt/models/deepseek_v2.py (+2/-0); python/sglang/srt/speculative/eagle_worker.py (+10/-0); docs/references/deepseek.md (+1/-1); test/srt/test_mla_flashinfer.py (+63/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Support the compatibility of nextn and flashinfer mla attention backend. Currently topk can only be set to 1 due to lack of custom mask support for flashinfer MLA wrapper. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Implement class `FlashInferMLAMultiStepDraftBackend` for draft model when using flashinfer mla and eagle together. ⏎ - Update some methods of `FlashInferMLABackend` so draft extend and target verify batches can be handled. ⏎ - Update  …[truncated]

### L2-df84ab2a5b  (L2, 2025-03-09, sha df84ab2a5b87, PR #4228)
TITLE: update sgl-kernel 3rdparty (#4228)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+0/-3); sgl-kernel/3rdparty/cutlass (+1/-1); sgl-kernel/3rdparty/turbomind (+0/-1); sgl-kernel/README.md (+0/-1); sgl-kernel/setup.py (+0/-3)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-9dfafa743c  (L2, 2025-03-09, sha 9dfafa743cf5, PR #4237)
TITLE: Fix test of flashinfer mla with nextn (#4237)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: test/srt/test_mla_flashinfer.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ #4217 changes the constraint on parameters for speculative decoding. ⏎ This PR fix the test parameters in #4218 to satisfy this constraint. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-c553e1604c  (L2, 2025-03-10, sha c553e1604c4e, PR #4165)
TITLE: DeepGemm integrate to sgl-kernel (#4165)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+3/-0); sgl-kernel/3rdparty/deepgemm (+1/-0); sgl-kernel/build.sh (+4/-3); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/setup.py (+52/-1); sgl-kernel/tests/test_deep_gemm.py (+263/-0)
LABELS: high priority
BODY: ## Motivation ⏎ Integrate DeepGemm in setup. ⏎ Linear usage: #4199 . ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ ## Checklist

### L2-5a6400eec5  (L2, 2025-03-10, sha 5a6400eec5f3, PR #4256)
TITLE: Test no vllm custom allreduce (#4256)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/pyproject.toml (+1/-1); .github/workflows/pr-test.yml (+1/-1); python/sglang/srt/server_args.py (+2/-2); scripts/ci_install_dependency.sh (+1/-1)
BODY: 

### L2-4d27eb9ad1  (L2, 2025-03-11, sha 4d27eb9ad1f1, PR #4291)
TITLE: update sgl-kernel 0.0.4.post2 (#4291)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-690e1f2371  (L2, 2025-03-11, sha 690e1f23716c, PR #4311)
TITLE: [AMD] Fix rocm sgl-kernel missing modules error (#4311)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); sgl-kernel/csrc/torch_extension_rocm.cc (+1/-1); sgl-kernel/setup_rocm.py (+1/-1)
BODY: ## Motivation ⏎ [Bug] missing allreduce from sgl_kernel module ⏎ [Bug] repeated nv sgl-kernel installation on amd platform ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-1cf63485c1  (L2, 2025-03-11, sha 1cf63485c1ef, PR #4317)
TITLE: upgrade flashinfer 0.2.3 (#4317)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); docs/start/install.md (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-d1da58e275  (L2, 2025-03-11, sha d1da58e275e3, PR #4321)
TITLE: unify is_cuda and is_hip (#4321)
SOURCES: path_core
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.kernel.triton_decode_lightllm, L2.optimization.weight_absorption, L2.kernel.rocm_mla_decode_rope, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+6/-6); python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py (+3/-3); python/sglang/srt/custom_op.py (+5/-3); python/sglang/srt/distributed/device_communicators/custom_all_reduce.py (+18/-17); python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py (+3/-3); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+4/-4); python/sglang/srt/layers/moe/ep_moe/kernels.py (+2/-1); python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+9/-9); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-7); (+8 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-ed91561f79  (L2, 2025-03-12, sha ed91561f7972, PR #4334)
TITLE: upgrade sgl-kernel 0.0.4.post3 (#4334)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-71046fcd71  (L2, 2025-03-12, sha 71046fcd7160, PR #4086)
TITLE: [XPU][CPU] Enable the native path of DeepSeek (#4086)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: docs/start/install.md (+9/-0); python/pyproject.toml (+1/-1); python/sglang/srt/layers/quantization/base_config.py (+12/-2); python/sglang/srt/layers/quantization/blockwise_int8.py (+16/-2); python/sglang/srt/layers/quantization/fp8.py (+92/-3); python/sglang/srt/layers/quantization/fp8_kernel.py (+182/-89); python/sglang/srt/layers/quantization/gptq.py (+44/-3); python/sglang/srt/layers/quantization/modelopt_quant.py (+16/-1); python/sglang/srt/layers/quantization/w8a8_fp8.py (+15/-2); python/sglang/srt/layers/quantization/w8a8_int8.py (+16/-1); (+6 more)
DEEP_STUDY: deep-study: this PR was reverted by PR 4367 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Motivation ⏎  ⏎ Currently the native path of deepseek v3 is broken, so fix the broken native path which will benefit CPU and all other accelerators ⏎  ⏎ Has verified on CPU and XPU: ⏎  ⏎ ``` ⏎ TRITON_AVAILABLE=0 python3 -m sglang.bench_one_batch --batch-size 1 --input 32 --output 32 --model deepseek-ai/DeepSeek-R1 --trust-remote-code --device cpu --attention-backend torch_native --disable-radix --disable-mla ⏎ python3 -m sglang.bench_one_batch --batch …[truncated]

### L2-45de89719c  (L2, 2025-03-12, sha 45de89719c33, PR #4367)
TITLE: Revert "[XPU][CPU] Enable the native path of DeepSeek" (#4367)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: docs/start/install.md (+0/-9); python/pyproject.toml (+1/-1); python/sglang/srt/layers/quantization/base_config.py (+2/-12); python/sglang/srt/layers/quantization/blockwise_int8.py (+2/-16); python/sglang/srt/layers/quantization/fp8.py (+3/-92); python/sglang/srt/layers/quantization/fp8_kernel.py (+89/-182); python/sglang/srt/layers/quantization/gptq.py (+3/-44); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-16); python/sglang/srt/layers/quantization/w8a8_fp8.py (+2/-15); python/sglang/srt/layers/quantization/w8a8_int8.py (+1/-16); (+6 more)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 4086 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#4086 because it breaks the ci tests https://github.com/sgl-project/sglang/actions/runs/13827716560/job/38687533656#step:5:434

### L2-3623b6a7f5  (L2, 2025-03-13, sha 3623b6a7f581, PR #4381)
TITLE: upgrade sgl-kernel 0.0.5 (#4381)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-8e66fbecee  (L2, 2025-03-13, sha 8e66fbecee8b, PR #4390)
TITLE: Improve DP attention (#4390)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/dp_attention.py (+30/-2); python/sglang/srt/layers/logits_processor.py (+1/-0); python/sglang/srt/managers/data_parallel_controller.py (+1/-1); python/sglang/srt/managers/scheduler.py (+51/-18); python/sglang/srt/model_executor/cuda_graph_runner.py (+59/-16); python/sglang/srt/model_executor/forward_batch_info.py (+13/-4); python/sglang/srt/models/deepseek_v2.py (+180/-177); python/sglang/srt/server_args.py (+5/-5); test/srt/test_dp_attention.py (+2/-0)
BODY: - Use a better padding strategy for cuda graph. If TP=8, DP=8, when batch size = 1, the previous implementation will pad it to global batch size 8. The new implementation will allow running global batch size 1, so it is faster at low bs range.  It is 1.15x faster then the old implementation for `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` at TP=8 and bs=1. ⏎ - Support TP != DP. Now you need to explicitly specify `--dp` and `--tp`. The constraint  …[truncated]

### L2-0e0ec70200  (L2, 2025-03-13, sha 0e0ec702007b, PR #4009)
TITLE: Hierarchical Caching supports MLA (#4009)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/memory_pool.py (+160/-26); python/sglang/srt/managers/cache_controller.py (+2/-5); python/sglang/srt/mem_cache/hiradix_cache.py (+11/-5); test/srt/test_hierarchical_mla.py (+56/-0)
BODY: ## Motivation ⏎ I am deeply grateful to @xiezhq-hermann for implementing the Hierarchical Caching feature, which has expanded the storage capacity of the KV cache. However, the version only supports MHA and not MLA.  This PR introduces support for Hierarchical Caching in the context of MLA. ⏎ At present, there might currently be a bug when tp > 1, and @xiezhq-hermann is fixing it. ⏎  ⏎ ## Modifications ⏎ I have abstracted a base class named BaseTokenT …[truncated]

### L2-ad1ae7f7cd  (L2, 2025-03-14, sha ad1ae7f7cd0e, PR #4439)
TITLE: use topk_softmax with sgl-kernel (#4439)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); .github/workflows/execute-notebook.yml (+1/-1); .github/workflows/experiment-runner.yml (+1/-1); .github/workflows/lint.yml (+1/-1); .github/workflows/nightly-test.yml (+1/-1); .github/workflows/pr-test-amd.yml (+2/-2); .github/workflows/pr-test-rust.yml (+2/-2); .github/workflows/pr-test-sgl-kernel.yml (+1/-1); .github/workflows/pr-test.yml (+9/-9); .github/workflows/release-docker-amd-nightly.yml (+1/-1); (+8 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-81f431eded  (L2, 2025-03-15, sha 81f431eded8a, PR #4449)
TITLE: feat: Add FlashMLA submodule (#4449)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: -
FILES: .gitmodules (+3/-0); sgl-kernel/setup.py (+49/-0); sgl-kernel/tests/test_flash_mla.py (+153/-0)
LABELS: high priority
DEEP_STUDY: deep-study: this PR was reverted by PR 4470 (confirmed_revert, reason=build_or_dependency)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-9971dc2283  (L2, 2025-03-16, sha 9971dc2283ed, PR #4470)
TITLE: Revert "feat: Add FlashMLA submodule (#4449)" (#4470)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: -
FILES: .gitmodules (+0/-3); sgl-kernel/setup.py (+0/-49); sgl-kernel/tests/test_flash_mla.py (+0/-153)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 4449 reason=build_or_dependency
BODY: This reverts commit 81f431eded8a634b80f6c9fa4e9e0b016bd1fac1. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ - missing `git submodule add https://github.com/deepseek-ai/FlashMLA sgl-kernel/3rdparty/flashmla` ⏎ - It's AOT not JIT ⏎ - I have previously manually installed flash_mla locally, so I can pass through ut, but the original pr simply does not work ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-1b859295f4  (L2, 2025-03-16, sha 1b859295f422, PR #4363)
TITLE: [Eagle] Remove the greedy branch and some redundant code (#4363)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/http_server.py (+1/-1); python/sglang/srt/managers/scheduler.py (+0/-2); python/sglang/srt/model_executor/cuda_graph_runner.py (+10/-11); python/sglang/srt/server_args.py (+0/-1); python/sglang/srt/speculative/build_eagle_tree.py (+7/-347); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+30/-5); python/sglang/srt/speculative/eagle_utils.py (+204/-250); python/sglang/srt/speculative/eagle_worker.py (+111/-46); python/sglang/srt/utils.py (+11/-0); (+4 more)
BODY: - Use faster kernel for temp=0 ⏎ - Support cuda graph padding ⏎ - Simplify redundant python code ⏎  ⏎ llama 2 7b: 390 token/s -> 400 token/s with this PR ⏎  ⏎ ``` ⏎  ⏎ ```

### L2-a53fe428f9  (L2, 2025-03-16, sha a53fe428f9f5, PR #4472)
TITLE: Support FlashMLA backend (#4472)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:production-kernel-provenance
ARTIFACT_HINTS: L2.backend.flashmla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashmla_backend.py (+128/-0); python/sglang/srt/layers/attention/utils.py (+54/-0); python/sglang/srt/model_executor/model_runner.py (+8/-0); python/sglang/srt/server_args.py (+8/-0); python/sglang/srt/managers/schedule_batch.py (+5/-1); scripts/playground/bench_speculative.py (+6/-0)
BODY: ## Motivation ⏎ Integrate flashmla for decoding, and the accuracy test is currently okay. The current implementation is quite simple, directly integrating flashmla as the backend. Later, we need to abstract a fastmla_backend, using fa3 for prefill and flashmla for decode ⏎ ## Modifications ⏎ * FlashMLABackend inherits from FlashInferMLAAttnBackend, using FlashInferMLAAttnBackend for prefill and FlashMLABackend for decoding. ⏎ * Add the create_flashml …[truncated]

### L2-f81a27f65e  (L2, 2025-03-17, sha f81a27f65ec1, PR #4522)
TITLE: upgrade sgl-kernel 0.0.5.post3 (#4522)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-c0e9a36c5f  (L2, 2025-03-18, sha c0e9a36c5f5e, PR #4553)
TITLE:  Optimize Triton decoding kernel for dynamic workload (#4553)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+2/-0); python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+92/-34); python/sglang/srt/layers/attention/base_attn_backend.py (+1/-0); python/sglang/srt/layers/attention/flashinfer_backend.py (+2/-0); python/sglang/srt/layers/attention/triton_backend.py (+142/-15); python/sglang/srt/model_executor/cuda_graph_runner.py (+10/-0); test/srt/test_triton_attention_kernels.py (+28/-8)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ The current implementation of triton decode attention uses a static split for KV, which makes it unable to adapt to dynamically changing context lengths and batch sizes in an online environment, leading to performance degradation. Therefore, I refactored the code to improve performance under different workloads. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Split `attn_out` into `attn_out` and `attn_lse` to ensure that `attn_out` is aligned with …[truncated]

### L2-90532b7627  (L2, 2025-03-18, sha 90532b762777, PR #4557)
TITLE: [Fix] Fix raw_bs bug when using flashinfer mla and eagle (#4557)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+11/-0)
BODY: ## Motivation ⏎  ⏎ Fix the bug mentioned in #4536: when `bs != raw_bs` when eagle replays cuda graph, `seq_len_cpu` needed by flashinfer mla needs to be handled. ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Fix bug ⏎  ⏎  ⏎  ⏎ ## Accuracy ⏎  ⏎ ### Launch ⏎ ```bash ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --speculative-algo EAGLE --speculative-draft lmsys/DeepSeek-V3-NextN --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft- …[truncated]

### L2-b6944f97a6  (L2, 2025-03-19, sha b6944f97a616, PR #4514)
TITLE: Support FlashMLA backend cuda graph (#4514)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.backend.flashmla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashmla_backend.py (+184/-30); python/sglang/srt/layers/attention/utils.py (+0/-1); python/sglang/srt/server_args.py (+4/-1)
LABELS: high priority
BODY: ## Motivation ⏎ Support FlashMLA backend cuda graph. Optimize index calculation, complete the calculation in init_forward ⏎ ## Modifications ⏎ * Optimize the FlashInfer block table calculation logic to compute only once during the forward pass. ⏎ * Support FlashMLA backend CUDA Graph. ⏎ * Automatically set page=64 when launching FlashMLA. ⏎ ## Test ⏎ **deepseekV3 accuracy test** ⏎ GSM8K  Accuracy: 0.980 ⏎ MMLU   Average accuracy: 0.878 ⏎  ⏎ todo ⏎ * performa …[truncated]

### L2-df7014a8d2  (L2, 2025-03-19, sha df7014a8d230, PR #4577)
TITLE: avoid cudaStreamSynchronize in DeepSeekV2AttentionMLA (#4577)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎ I profiled deepseek and observed some bubbles in timeline, finally found the cause: ⏎ ![image](https://github.com/user-attachments/assets/b7235a07-580b-4b50-837c-babe92d4c342) ⏎ This is caused by D2H copy in DeepSeekV2AttentionMLA. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Using forward_batch.extend_prefix_lens_cpu directy instead of forward_batch.extend_prefix_lens, this can decrease TTFT ⏎  ⏎ ## Checklist

### L2-9e93ef3f8e  (L2, 2025-03-20, sha 9e93ef3f8e82, PR #4571)
TITLE: [fix] fix illegal mem access and clean up triton attention backend (#4571)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+0/-2); python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+15/-10); python/sglang/srt/layers/attention/base_attn_backend.py (+0/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+0/-2); python/sglang/srt/layers/attention/triton_backend.py (+103/-97); python/sglang/srt/model_executor/cuda_graph_runner.py (+0/-10); test/srt/test_triton_attention_kernels.py (+6/-3)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Clean up and verify issue at https://github.com/sgl-project/sglang/pull/4553#issuecomment-2735283306 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Extract `attn_lse` from `attn_logits` and add `ForwardMetadata` class for clearerdefinition. ⏎ 2. Fix `get_num_kv_splits` with MTP ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-5d7edc8e55  (L2, 2025-03-23, sha 5d7edc8e55d7, PR #4680)
TITLE: Support FA3 as Attention backend by using `--attention-backend fa3` (#4680)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.dispatch.server_args_defaults
FILES: python/sglang/bench_serving.py (+2/-0); python/sglang/srt/layers/attention/flashattention_backend.py (+295/-0); python/sglang/srt/model_executor/model_runner.py (+13/-0); python/sglang/srt/server_args.py (+1/-1); python/sglang/test/attention/test_flashattn_backend.py (+311/-0)
LABELS: high priority, performance
BODY: Co-authored with @qingquansong  ⏎  ⏎ Roadmap Issue is [here](https://github.com/sgl-project/sglang/issues/4709) ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ Support FA3 as attention backend by using fa3's `flash_attn_with_kvcache` ⏎  ⏎ Conclusion: ⏎ - Prefill Throughput are on par than current baseline ⏎ - Decode Throughput are slightly higher than current baseline ⏎ - Accuracy is slightly better than current baseline ⏎ - Flashinfer will OOM when batch size, input size is la …[truncated]

### L2-57eec0bfbc  (L2, 2025-03-24, sha 57eec0bfbce9, PR #4691)
TITLE: fix FlashMLA cudagraph config (#4691)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.backend.flashmla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashmla_backend.py (+4/-2)
BODY: ## Motivation ⏎ For changes and performance testing, you can refer to https://github.com/sgl-project/sglang/pull/4591.

### L2-8bf6d7f406  (L2, 2025-03-27, sha 8bf6d7f40614, PR #4706)
TITLE: support cmake for sgl-kernel (#4706)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.rocm (+2/-0); sgl-kernel/CMakeLists.txt (+166/-0); .github/workflows/pr-test-amd.yml (+2/-2); .github/workflows/pr-test-sgl-kernel.yml (+2/-0); sgl-kernel/3rdparty/flashinfer (+1/-1); sgl-kernel/Makefile (+3/-2); sgl-kernel/build.sh (+2/-2); sgl-kernel/csrc/attention/lightning_attention_decode_kernel.cu (+1/-1); sgl-kernel/csrc/gemm/cublas_grouped_gemm.cu (+0/-1); sgl-kernel/csrc/moe/moe_align_kernel.cu (+0/-1); (+8 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-668ecc6c5b  (L2, 2025-03-27, sha 668ecc6c5b37, PR #4813)
TITLE: Fix ut mla-test-1-gpu-amd (#4813)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-amd.yml (+1/-0); python/sglang/srt/layers/rotary_embedding.py (+12/-0)
DEEP_STUDY: deep-study: this PR was reverted by PR 4959 (confirmed_revert, reason=crash_or_hang)
BODY: ## Motivation ⏎  ⏎  ⏎ Fix ut ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-4db29e82ec  (L2, 2025-03-28, sha 4db29e82ec10, PR #4864)
TITLE: [Feat] support deepgemm for cmake (#4864)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+16/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2bb0e7cf43  (L2, 2025-03-28, sha 2bb0e7cf43a3, PR #4871)
TITLE: fix sampling issue (#4871)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-20c90be23d  (L2, 2025-03-28, sha 20c90be23de7, PR #4831)
TITLE: [Feature] Support FA3 backend for MLA (#4831)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+171/-73); python/sglang/srt/model_executor/model_runner.py (+5/-1); python/sglang/srt/models/deepseek_v2.py (+4/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Support FlashAttention 3 backend for MLA. FA3 official example of MLA can be found in [this file](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/benchmark_mla_decode.py). ⏎  ⏎ TODO: ⏎  ⏎  ⏎ Maybe in next PR: ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Support MLA for flash attention 3 backend by with `flash_attn_with_kvcache` function ⏎  ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎ ### Launch ⏎ ```bash ⏎ python3 -m sglang.launch_server --model deepseek-ai/Dee …[truncated]

### L2-d8a136a113  (L2, 2025-03-28, sha d8a136a11332, PR #4873)
TITLE: upgrade sgl-kernel 0.0.5.post4 (#4873)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-9adf178cc2  (L2, 2025-03-30, sha 9adf178cc2ac, PR #4930)
TITLE: Fix 2-gpu CI test and suppress some warnings (#4930)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+5/-3); python/sglang/srt/utils.py (+4/-4); test/srt/run_suite.py (+11/-11); test/srt/test_eagle_infer.py (+1/-1)
BODY: 

### L2-37c66ec856  (L2, 2025-03-30, sha 37c66ec8563d, PR #4902)
TITLE: [feat] add fa3 in sgl-kernel (#4902)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+54/-0); sgl-kernel/README.md (+30/-0); sgl-kernel/csrc/torch_extension.cc (+5/-0); sgl-kernel/include/sgl_kernel_ops.h (+47/-0); sgl-kernel/include/sgl_kernel_torch_shim.h (+122/-0); sgl-kernel/python/sgl_kernel/flash_attn.py (+201/-0); sgl-kernel/tests/test_flash_attention.py (+841/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ add fa3 in sgl-kernel ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e62d60fe6d  (L2, 2025-03-30, sha e62d60fe6d7c, PR #4932)
TITLE: [Fix] avoid stream sync and torch compile in prefill for fa3 backend (#4932)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.flashmla, L2.backend.fa3_fa4_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-1); python/sglang/srt/layers/attention/flashmla_backend.py (+1/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+1/-1); python/sglang/srt/managers/schedule_batch.py (+12/-13); python/sglang/srt/model_executor/cuda_graph_runner.py (+2/-2); python/sglang/srt/model_executor/forward_batch_info.py (+8/-12); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+5/-5)
BODY: ## Motivation ⏎  ⏎ During profiling, I found two potential places for improvement when utilizing fa3 backend: ⏎ ![fig1](https://github.com/user-attachments/assets/8aa4607a-1bb1-42d0-b26c-1a8422012fd2) ⏎  ⏎ As shown in the figure,  ⏎ - circle a is a cuda stream synchronization bubble, caused by `.item()` operation in `init_forward_metadata`. This overhead happens during every prefill/extend batch and costs ~100ms. ⏎ - circle b is the long overhead caused …[truncated]

### L2-4814ecaff9  (L2, 2025-03-30, sha 4814ecaff988, PR #4933)
TITLE: cleanup sgl-kernel (#4933)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+2/-10); sgl-kernel/3rdparty/cccl (+0/-1); sgl-kernel/3rdparty/cutlass (+0/-1); sgl-kernel/3rdparty/deepgemm (+0/-1); sgl-kernel/3rdparty/flashinfer (+1/-1); sgl-kernel/README.md (+1/-1); sgl-kernel/THIRDPARTYNOTICES.txt (+33/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-1c63e79756  (L2, 2025-03-31, sha 1c63e7975604, PR #4954)
TITLE: use fa3 in sgl-kernel (#4954)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e8999b13b7  (L2, 2025-04-03, sha e8999b13b7c3, PR #5005)
TITLE: Replace enable_flashinfer_mla argument with attention_backend (#5005)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+0/-2); python/sglang/srt/model_executor/model_runner.py (+6/-3); python/sglang/srt/models/deepseek_v2.py (+1/-2); python/sglang/srt/server_args.py (+2/-2); docs/backend/server_arguments.md (+3/-3); docs/references/deepseek.md (+2/-2); python/sglang/srt/managers/schedule_batch.py (+1/-2); test/srt/test_mla_flashinfer.py (+6/-4)
DEEP_STUDY: deep-study: this PR was reverted by PR 5048 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Motivation ⏎  ⏎ Currently there are many mla backends, making the arguments a little messy. ⏎ After this PR, the functionality of `--enable-flashinfer-mla` can be replaced by `--attention-backend flashinfer`. `--enable-flashinfer-mla` can still be used as before, but it's supposed to be deprecated in following versions. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-b8b6008f47  (L2, 2025-04-03, sha b8b6008f47de, PR #5036)
TITLE: [Fix] fix fa3 build at cu118 (#5036)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+85/-50); sgl-kernel/cmake/utils.cmake (+21/-0); sgl-kernel/csrc/common_extension.cc (+1/-40); sgl-kernel/csrc/flash_extension.cc (+62/-0); sgl-kernel/include/sgl_flash_kernel_ops.h (+85/-0); sgl-kernel/include/sgl_kernel_ops.h (+0/-47); sgl-kernel/python/sgl_kernel/flash_attn.py (+15/-4); sgl-kernel/tests/test_flash_attention.py (+19/-1)
ISSUES: #4941 [Feature] use different lib so for fa3 in sgl-kernel
DEEP_STUDY: deep-study correctness case sglang:b8b6008f47: class=hardware_compiler_specific; symptom=compile_or_build_failure; introducing=unknown
BODY: ## Motivation ⏎  ⏎  ⏎ Fix cu118 for fa3 compile error ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-74885a848b  (L2, 2025-04-03, sha 74885a848bb6, PR #5048)
TITLE: Revert "Replace enable_flashinfer_mla argument with attention_backend" (#5048)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+2/-0); python/sglang/srt/model_executor/model_runner.py (+3/-6); python/sglang/srt/models/deepseek_v2.py (+2/-1); python/sglang/srt/server_args.py (+2/-2); docs/backend/server_arguments.md (+3/-3); docs/references/deepseek.md (+2/-2); python/sglang/srt/managers/schedule_batch.py (+2/-1); test/srt/test_mla_flashinfer.py (+4/-6)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 5005 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#5005 because it breaks CI

### L2-e53bf190bc  (L2, 2025-04-03, sha e53bf190bce3, PR #5049)
TITLE: upgrade sgl-kernel v0.0.7 (#5049)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-7ed77d6b9e  (L2, 2025-04-04, sha 7ed77d6b9e5c, PR #4535)
TITLE: fix dummy-load deepseekv2 (#4535)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/model_loader/loader.py (+8/-0); python/sglang/srt/models/deepseek_v2.py (+79/-73)
LABELS: high priority
ISSUES: #4405 [Bug] Can't benchmark deepseek_v2 with dummy weights
BODY: ## Motivation ⏎ fix https://github.com/sgl-project/sglang/issues/4405  ⏎  ⏎ ## Modifications ⏎  ⏎ Load weight is divided into two parts: ⏎  ⏎ - Purely load logic. ⏎ - Post-processing logic, including some reshaping and assignment of member variables. ⏎ Dummy loader only needs to handle the second logic, not skip everything. ⏎  ⏎ ## Checklist

### L2-efbae697b3  (L2, 2025-04-05, sha efbae697b370, PR #5052)
TITLE: [Revision] Replace enable_flashinfer_mla argument with attention_backend (#5052)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+0/-2); python/sglang/srt/model_executor/model_runner.py (+49/-38); python/sglang/srt/models/deepseek_v2.py (+1/-2); python/sglang/srt/server_args.py (+3/-7); python/sglang/srt/speculative/eagle_worker.py (+24/-22); docs/backend/server_arguments.md (+3/-3); docs/references/deepseek.md (+2/-2); python/sglang/srt/managers/schedule_batch.py (+4/-2); test/srt/test_mla_flashinfer.py (+6/-4)
BODY: ## Motivation ⏎ Fixing the default backend bug caused by #5005. Now w/wo mla, backend is set to triton/flashinfer by default. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-aba5ca154d  (L2, 2025-04-05, sha aba5ca154d4c, PR #5080)
TITLE: python transfer custom allreduce from trt kernel to vllm kernel (#5080)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/_custom_ops.py (+59/-92); python/sglang/srt/distributed/device_communicators/custom_all_reduce.py (+25/-77); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎ ref https://github.com/sgl-project/sglang/pull/5079 ⏎  ⏎ ```python ⏎ lm_eval --model sglang --model_args pretrained=meta-llama/Llama-3.1-8B,tp_size=x,dtype=auto --tasks gsm8k --batch_size 16 ⏎ lm_eval --model sglang --model_args pretrained=meta-llama/Llama-3.1-70B,tp_size=x,dtype=auto --tasks gsm8k --batch_size 16 ⏎ ``` ⏎ Llama3.1-8B ⏎  ⏎ tp = 2 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|---- …[truncated]
