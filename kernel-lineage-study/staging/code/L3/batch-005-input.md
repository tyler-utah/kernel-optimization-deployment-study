### L3-ed6ea06577  (L3, 2025-03-05, sha ed6ea06577ec, PR #14244)
TITLE: [Hardware] Update the flash attn tag to support Blackwell (#14244)
SOURCES: path_core, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
STAGE1: extend_support; artifacts=L3.flash_attn.fork_build; Dossier shows L3.flash_attn.fork_build behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+2/-2); vllm/attention/backends/utils.py (+6/-1)
LABELS: ready, ci/build
BODY: Update vllm's flash attention tag to update automatically for blackwell platform.

### L3-6bd1dd9d26  (L3, 2025-03-06, sha 6bd1dd9d2625, PR #14152)
TITLE: [Kernel] [V1] Improved performance for V1 Triton (ROCm) backend  (#14152)
SOURCES: path_core, symbol_pickaxe
STAGE1: optimize; artifacts=L3.triton.prefix_prefill,L3.triton.chunked_prefill_paged_decode,L3.rocm.v1_rocm_attn; Dossier shows L3.triton.prefix_prefill behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.triton.prefix_prefill, L3.triton.chunked_prefill_paged_decode, L3.rocm.v1_rocm_attn
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+289/-0); vllm/attention/ops/prefix_prefill.py (+13/-1); vllm/v1/attention/backends/rocm_attn.py (+20/-17); tests/kernels/test_prefix_prefill.py (+76/-59)
LABELS: rocm, ready, v1
PERF_LINES: Request throughput (req/s):              9.32 | Output token throughput (tok/s):         1838.05 | Total Token throughput (tok/s):          3842.98 | Mean TTFT (ms):                          10264.30 | Median TTFT (ms):                        8952.08 | P99 TTFT (ms):                           24269.34 | Mean TPOT (ms):                          408.87 | Median TPOT (ms):                        261.
BODY: In this PR we target performance improvement for the V1 `ROCmAttentionBackend` (which should be renamed `TritonAttentionBackend` once [this](https://github.com/vllm-project/vllm/pull/14071) PR is merged).  ⏎  ⏎ cc @SageMoore  ⏎  ⏎ ### Performance  ⏎  ⏎ All of the below results are for `meta-llama/Llama-3.1-8B-Instruct` on an NVIDIA H100 GPU. ⏎  ⏎ The `ROCmAttention` backend on `main` (we have to hack it a bit to make this happen on NVIDIA) currently produces the following results for the serving benchmark: ⏎ ``` ⏎ $ python benchmarks/benchmark_serving.py \ ⏎     --model meta-llama/Llama-3.1-8B-Instruct \ ⏎     --dataset-name sharegpt \ ⏎     --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     1000       ⏎ Benchmark duration (s):                  107.33     ⏎ Total input tokens:                      215196     ⏎ Total generated tokens:                  197284     ⏎ Request throughput (req/s):              9.32       ⏎ Output token throughput (tok/s):         1838.05    ⏎ Total Token throughput (tok/s):          3842.98    ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          10264.30   ⏎ Median TTFT (ms):                        8952.08    ⏎ P99 TTFT (ms):                           24269.34   ⏎ -----Time per Output Token (excl. 1st token)------ ⏎ Mean TPOT (ms):                          408.87     ⏎ Median TPOT (ms):                        261.25     ⏎ P99 TPOT (ms):                           1208.50    ⏎ ---------------Inter-token Latency---------------- ⏎ Mean ITL (ms):                           210.67     ⏎ Median ITL (ms):                         172.46     ⏎ P99 ITL (ms):                            1308.82    ⏎ ================================================== ⏎ ``` ⏎  ⏎ Whereas, using the default `FlashAttention` backend produces: ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     1000       ⏎ Benchmark duration (s):                  20.47      ⏎ Total input tokens:                      215196     ⏎ Total generated tokens:                  198001     ⏎ Request throughput (req/s):              48.86      ⏎ Output token throughput (tok/s):         9673.74    ⏎ Total Token throughput (tok/s):          20187.58   ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          3473.32    ⏎ Median TTFT (ms):                        3405.53    ⏎ P99 TTFT (ms):                           6302.31    ⏎ -----Time per Output Token (excl. 1st token)------ ⏎ Mean TPOT (ms):                          82.06      ⏎ Median TPOT (ms):                    …[truncated]

### L3-9f1710f1ac  (L3, 2025-03-06, sha 9f1710f1ace3, PR #13897)
TITLE: Fix mla prefill context performance (#13897)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: optimize; artifacts=L3.mla.triton_v0,L3.mla.common_v1; Dossier shows L3.mla.triton_v0 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.mla.triton_v0, L3.mla.common_v1
FILES: vllm/attention/backends/mla/common.py (+1/-1); vllm/v1/attention/backends/mla/common.py (+1/-1)
LABELS: ready, v1
BODY: `kv_c_normed` unsqeezed leads to the following `kv_b_proj` slowed down.

### L3-6b2ef5cd17  (L3, 2025-03-06, sha 6b2ef5cd17c5, PR #14313)
TITLE: [Bug] Fix Attention when ignored in by quant_method (#14313)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.dispatch.abstract_interface; Dossier shows L3.dispatch.abstract_interface behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: -
FILES: vllm/attention/layer.py (+3/-1)
LABELS: bug
BODY: There was a bug in the Attention module when a quantized model has the attention module ignored. Many `quant_config.get_quant_method` calls will return `UnquantizedLinearMethod` rather than `None` if the layer is ignored. Maybe we should be returning `UnquantizedAttentionMethod` to be more clear, but that isn't functionally important ⏎  ⏎ On main we hit `assert isinstance(quant_method, BaseKVCacheMethod)`

### L3-6832707e90  (L3, 2025-03-06, sha 6832707e90d4, PR #14221)
TITLE: [V1][Bugfix] Standardize quantized kv cache rejection for attention backends (#14221)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.flash_attn.v0_backend,L3.flash_attn.v1_backend,L3.mla.triton_v0,L3.mla.flashmla_v0_adapter,L3.mla.flashmla_v1_adapter,L3.dispatch.abstract_interface; Dossier shows L3.flash_attn.v0_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.mla.triton_v0, L3.mla.flashmla_v0_adapter, L3.mla.flashmla_v1_adapter, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+4/-0); vllm/attention/backends/flash_attn.py (+8/-1); vllm/attention/backends/flashmla.py (+6/-3); vllm/attention/backends/hpu_attn.py (+6/-1); vllm/attention/backends/ipex_attn.py (+3/-2); vllm/attention/backends/pallas.py (+3/-2); vllm/attention/backends/torch_sdpa.py (+6/-2); vllm/attention/backends/triton_mla.py (+6/-3); vllm/v1/attention/backends/flash_attn.py (+5/-1); vllm/v1/attention/backends/mla/flashmla.py (+6/-4); vllm/v1/attention/backends/mla/triton_mla.py (+6/-1)
LABELS: bug, ready, v1
ISSUES: #11329 [Bug]: FP8 kvcache causes RuntimeError in v1 engine | #13133 [Bug]: [V1] wrong output when using kv cache fp8
PERF_LINES: INFO 03-04 17:10:36 [loggers.py:79] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache
BODY: Many users have been reporting FP8 KV cache "hasn't been working right" in V1, but in reality we have just been ignoring the parameter since the only (non-MLA) attention backend in V1 is FlashAttention, which doesn't support it. This PR audits existing backends to raise an exception quickly if quantized kv cache is not supported ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/13133  ⏎ FIX https://github.com/vllm-project/vllm/issues/11329 ⏎  ⏎ Before (the server starts even though FP8 isn't supported): ⏎ ``` ⏎ VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct --port 9000 --kv-cache-dtype fp8 ⏎ ... ⏎ INFO:     Started server process [327532] ⏎ INFO:     Waiting for application startup. ⏎ INFO:     Application startup complete. ⏎ INFO 03-04 17:10:36 [loggers.py:79] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0% ⏎ ``` ⏎  ⏎ After: ⏎ ``` ⏎ VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct --port 9000 --kv-cache-dtype fp8 ⏎ ... ⏎ ERROR 03-04 17:08:57 [core.py:303]   File "/home/mgoin/code/vllm/vllm/v1/attention/backends/flash_attn.py", line 185, in __init__ ⏎ ERROR 03-04 17:08:57 [core.py:303]     raise NotImplementedError( ⏎ ERROR 03-04 17:08:57 [core.py:303] NotImplementedError: FlashAttention V1 with FP8 KV cache not yet supported ⏎ ```

### L3-dae6896977  (L3, 2025-03-06, sha dae68969774e, PR #14384)
TITLE: [Perf] Reduce MLA CPU overheads in V1 (#14384)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts(reverted)
STAGE1: optimize; artifacts=L3.mla.common_v1; Dossier shows L3.mla.common_v1 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+11/-4); vllm/model_executor/layers/rotary_embedding.py (+7/-2)
LABELS: performance, ready, v1
DEEP_STUDY: deep-study: this PR was reverted by PR 14471 (confirmed_revert, reason=correctness_or_accuracy)
BODY: Some temporary hacks to reduce CPU overheads in MLA caused by rotary embeddings (not in torch.compile, or a cuda-graph) ⏎  ⏎ Main ⏎  ⏎ <img width="794" alt="Screenshot 2025-03-06 at 3 14 37 PM" src="https://github.com/user-attachments/assets/f7f46f74-86dd-47b3-8642-b34d25712d1d" /> ⏎  ⏎ This PR ⏎  ⏎ <img width="408" alt="image" src="https://github.com/user-attachments/assets/da3b88e9-10a3-431f-9b9d-e65974cb9e9b" />

### L3-333681408f  (L3, 2025-03-07, sha 333681408fea, PR #14462)
TITLE: [Bugfix][V1] Handle MLA in kv_cache_interface (#14462)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.mla.common_v1; Dossier shows L3.mla.common_v1 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+3/-2); vllm/v1/worker/tpu_model_runner.py (+4/-3); vllm/v1/kv_cache_interface.py (+8/-5)
LABELS: ready, v1
BODY: main:  ⏎ ```INFO 03-07 22:50:53 [kv_cache_utils.py:537] GPU KV cache size: 249,584 tokens``` ⏎ this PR:  ⏎ ```INFO 03-07 22:54:52 [kv_cache_utils.py:537] GPU KV cache size: 499,184 tokens```

### L3-ca7a2d5f28  (L3, 2025-03-07, sha ca7a2d5f28ea, PR #14471)
TITLE: Revert "[Perf] Reduce MLA CPU overheads in V1 (#14384)" (#14471)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts
STAGE1: revert; artifacts=L3.mla.common_v1; Dossier shows L3.mla.common_v1 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+4/-11); vllm/model_executor/layers/rotary_embedding.py (+2/-7)
LABELS: v1
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 14384 reason=correctness_or_accuracy
BODY: Running ⏎ ``` ⏎ VLLM_USE_V1=1 vllm serve deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --tensor_parallel_size=2 --port 8192 --trust-remote-code ⏎ ``` ⏎ and then ⏎ ``` ⏎ lm_eval --model local-completions --tasks gsm8k --model_args model=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct,base_url=http://127.0.0.1:8192/v1/completions,num_concurrent=5,max_retries=3,tokenized_requests=False --limit 100 ⏎ ``` ⏎  ⏎ On current main we see: ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.06|±  |0.0239| ⏎ |     |       |strict-match    |     5|exact_match|↑  | 0.00|±  |0.0000| ⏎ ``` ⏎  ⏎ This PR: ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.77|±  |0.0423| ⏎ |     |       |strict-match    |     5|exact_match|↑  | 0.77|±  |0.0423| ⏎ ```

### L3-db84f5eb3b  (L3, 2025-03-08, sha db84f5eb3bd2, PR #14476)
TITLE: [Bugfix] DeepSeek Accuracy (#14476)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.mla.common_v1; Dossier shows L3.mla.common_v1 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+7/-5)
LABELS: ready, v1
BODY: Fix accuracy issue introduced by https://github.com/vllm-project/vllm/pull/14384 , on main `forward_native` is called (due to `enabled()` be false in `CustomOp(nn.Module)` on main) the PR updated this cuda, changing the behavior slightly. We should probably investigate further to figure out why `forward_native` differs form `forward_cuda` (on an Nvidia GPUs). This is potentially due to `q_pe` and `k_pe` being different shapes in MLA.  ⏎  ⏎ Main ⏎ ``` ⏎ local-completions (model=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct,base_url=http://127.0.0.1:8192/v1/completions,num_concurrent=5,max_retries=3,tokenized_requests=False), gen_kwargs: (None), limit: 100.0, num_fewshot: None, batch_size: 1 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.06|±  |0.0239| ⏎ |     |       |strict-match    |     5|exact_match|↑  | 0.00|±  |0.0000| ⏎ ``` ⏎  ⏎ This PR: ⏎ ``` ⏎ local-completions (model=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct,base_url=http://127.0.0.1:8192/v1/completions,num_concurrent=5,max_retries=3,tokenized_requests=False), gen_kwargs: (None), limit: 100.0, num_fewshot: None, batch_size: 1 ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.78|±  |0.0416| ⏎ |     |       |strict-match    |     5|exact_match|↑  | 0.77|±  |0.0423| ⏎ ```

### L3-b0d541947a  (L3, 2025-03-08, sha b0d541947ab5, PR #14451)
TITLE: [Attention] Default to FlashMLA backend for MLA (#14451)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: change_default; artifacts=L3.platform.cuda_selection; Dossier shows L3.platform.cuda_selection behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+24/-16)
LABELS: ready
BODY: Numbers from @simon-mo , seems like a consistent enough win to have on by default ⏎  ⏎ ![image](https://github.com/user-attachments/assets/2a2c83e7-33e7-41a3-b58e-31a9ca33130c)

### L3-fb0acb6c72  (L3, 2025-03-10, sha fb0acb6c7287, PR #14540)
TITLE: [Perf] Improve MLA on V1 (#14540)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: optimize; artifacts=L3.mla.common_v1; Dossier shows L3.mla.common_v1 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+41/-27)
LABELS: ready, v1
PERF_LINES: 2. Reordered some operation in the `build` function, which ended up costing quite a bit overhead in my timing (p99 tail latency up to 1ms) | We are still a bit worse on the short range but we became significantly better on longer range. 64% boost for 6k input. | VLLM_USE_V1=1 python benchmarks/benchmark_throughput.py --model /home/vllm-dev/DeepSeek-R1 --load-format dummy --trust-remote-code --inpu
BODY: This PR helps V1 to _mostly match_ and exceed (in most cases) V0's performance for MLA. Mostly by two things ⏎ 1. Fix @LucasWilkinson's `rotary_emb` specialization (#14384, #14471, #14476) to reduce CPU overhead. ⏎   * Identified that the cause of 0 GSM8K score comes from the cuda kernel needs the input to be continuous. ⏎   * Fixed it by make the input contiguous if possible. A better fix will be to change the kernel (help wanted).  ⏎ 2. Reordered some operation in the `build` function, which ended up costing quite a bit overhead in my timing (p99 tail latency up to 1ms)  ⏎   * This is by ensuring there is not GPU -> CPU communication. CPU -> GPU is fine.  ⏎  ⏎ All the following ran in 8xH200.  ⏎  ⏎ ## Performance Test (R1) ⏎ We are still a bit worse on the short range but we became significantly better on longer range. 64% boost for 6k input.  ⏎  ⏎ VLLM_USE_V1=1 python benchmarks/benchmark_throughput.py --model /home/vllm-dev/DeepSeek-R1 --load-format dummy --trust-remote-code --input-len 3000 --output-len 1000 --num-prompts 50 --tensor-parallel-size 8 ⏎ Throughput: 1.09 requests/s, 4342.27 total tokens/s, **1085.57 output tokens/s** ⏎  ⏎ VLLM_USE_V1=0 python benchmarks/benchmark_throughput.py --model /home/vllm-dev/DeepSeek-R1 --load-format dummy --trust-remote-code --input-len 3000 --output-len 1000 --num-prompts 50 --tensor-parallel-size 8 ⏎ Throughput: 1.13 requests/s, 4536.67 total tokens/s, **1134.17 output tokens/s** ⏎  ⏎ VLLM_USE_V1=1 python benchmarks/benchmark_throughput.py --model /home/vllm-dev/DeepSeek-R1 --load-format dummy --trust-remote-code --input-len 6000 --output-len 1000 --num-prompts 50 --tensor-parallel-size 8 ⏎ Throughput: 0.87 requests/s, 6060.61 total tokens/s, **865.80 output tokens/s** ⏎  ⏎ VLLM_USE_V1=0 python benchmarks/benchmark_throughput.py --model /home/vllm-dev/DeepSeek-R1 --load-format dummy --trust-remote-code --input-len 6000 --output-len 1000 --num-prompts 50 --tensor-parallel-size 8 ⏎ Throughput: 0.53 requests/s, 3692.82 total tokens/s, **527.55 output tokens/s** ⏎  ⏎ ## Performance Test (Small) ⏎ We are 15% better for small model for 3k input.  ⏎  ⏎ VLLM_USE_V1=1 python benchmarks/benchmark_throughput.py --model deepseek-ai/DeepSeek-V2-Lite --load-format dummy --trust-remote-code --input-len 3000 --output-len 1000 --num-prompts 50  ⏎ Throughput: 3.84 requests/s, 15364.27 total tokens/s, **3841.07 output tokens/s** ⏎  ⏎ VLLM_USE_V1=0 python benchmarks/benchmark_throughput.py --model deepseek-ai/DeepSeek-V2-Lite --load-format dummy --trust-remote-code --input-len 3000 --output-len 1000 --num-prompts 50  ⏎ Throughput: 3.32 requests/s, 13275.67 total tokens/s, **3318.92 output …[truncated]

### L3-916836bbfb  (L3, 2025-03-12, sha 916836bbfb7e, PR #14664)
TITLE: [FEAT] [ROCm] [Embedding] Add encoder-only model support into ROCm Flash Attention to enable embedding models. (#14664)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L3.flash_attn.fork_inline_cmake,L3.rocm.rocm_flash_attn_v0; Dossier shows L3.flash_attn.fork_inline_cmake behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.rocm.rocm_flash_attn_v0
FILES: CMakeLists.txt (+4/-0); vllm/attention/backends/rocm_flash_attn.py (+68/-44); csrc/moe/torch_bindings.cpp (+1/-0); tests/models/embedding/language/test_cls_models.py (+15/-2); tests/models/embedding/language/test_embedding.py (+11/-2); tests/models/embedding/language/test_gritlm.py (+2/-2); tests/models/embedding/vision_language/test_llava_next.py (+17/-0)
LABELS: rocm, ready, ci/build
ISSUES: #14062 [Bug]: Alibaba-NLP/gte-Qwen2-7B-instruct on AMD MI300X | #14583 [Bug]: ERROR 03-11 07:47:00 [engine.py:141] AttributeError: Invalid attention type encoder-only
PERF_LINES: The same embedding model is run on A100, L40 and MI300X, the embedding output is compared: | | Comparison | Mean Difference | Abs Mean | Std | Min | Max | Cosine Similarity | L2 Norm Avg | % Dims with Rel Diff >1% | | | **A100 vs L40** | 4.42e-07 | 3.39e-05 | 4.51e-05 | -4.58e-04 | 3.66e-04 | 0.999996 | 0.00270 | 16.95% | | | **A100 vs MI300X** | 9.57e-08 | 3.22e-05 | 4.31e-05 | -3.36e-04 | 3.05e-
BODY: # Description ⏎ This PR add the logic to enable ENCODER_ONLY model support to ROCm Flash Attention. Thus, enabling language embedding models and some vision language embedding models. ⏎  ⏎ FIX https://github.com/vllm-project/vllm/issues/14062 ⏎ FIX #14583 ⏎  ⏎ # File changes: ⏎ * `vllm/attention/backends/rocm_flash_attn.py`: Add ENCODER_ONLY code path ⏎ * `tests/models/embedding/language/test_embedding.py`: Fix the code logic to enable unit tests for ROCm support. Re-group the embedding model to their correct category. ⏎ * `tests/models/embedding/language/test_cls_models.py`: Only test "half" datatype for ROCm support as ROCm Flash Attention does not support Float32 attention. ⏎ * `tests/models/embedding/language/test_gritlm.py`: Fix the bug where the tests are not skipped even if `xformers` package is not installed. ⏎  ⏎ # Tests ⏎ * Locally ran tests: ⏎   *  `tests/models/embedding/language` [Passed] ⏎   * `tests/models/embedding/vision_language/test_dse_qwen2_vl.py` [Passed] ⏎   * `tests/models/embedding/vision_language/test_llava_next.py`. This test is skipped as this HF model only works on CUDA, failed on CPU and ROCm. ⏎   * `tests/models/embedding/vision_language/test_phi3v.py` failed because of this BUG https://github.com/vllm-project/vllm/issues/14677 ⏎    ⏎  ⏎ * Since there is a change in the ROCm Flash Attention module, I have also ensure all the unittests below passed ⏎   * `tests/models/decoder_only/language/test_models.py` [Passed] ⏎  ⏎ # Experimental Results: ⏎  ⏎ The same embedding model is run on A100, L40 and MI300X, the embedding output is compared: ⏎  ⏎ Model launch command: `VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve Alibaba-NLP/gte-Qwen2-7B-instruct --dtype float16 --max-num-seqs 512 --gpu-memory-utilization 0.95 --hf-overrides '{"is_causal": false}' --trust-remote-code` ⏎  ⏎ ## Comparison of Embedding Model Performance Across Different Hardware ⏎  ⏎ | Comparison | Mean Difference | Abs Mean | Std | Min | Max | Cosine Similarity | L2 Norm Avg | % Dims with Rel Diff >1% | ⏎ |------------|----------------|----------|-----|-----|-----|-------------------|-------------|--------------------------| ⏎ | **A100 vs L40** | 4.42e-07 | 3.39e-05 | 4.51e-05 | -4.58e-04 | 3.66e-04 | 0.999996 | 0.00270 | 16.95% | ⏎ | **A100 vs MI300X** | 9.57e-08 | 3.22e-05 | 4.31e-05 | -3.36e-04 | 3.05e-04 | 0.999997 | 0.00256 | 15.33% | ⏎ | **L40 vs MI300X** | -3.46e-07 | 3.28e-05 | 4.35e-05 | -2.75e-04 | 3.66e-04 | 0.999997 | 0.00257 | 15.64% | ⏎ | **MI300X Triton vs CK** | -6.02e-07 | 2.78e-05 | 3.69e-05 | -2.44e-04 | 3.05e-04 | 0.999998 | 0.00220 | 13.74% | ⏎  ⏎ *Note: All p-values from paired t-tests were >0.05, indicating no statistical …[truncated]

### L3-d9f83d6206  (L3, 2025-03-12, sha d9f83d62068b, PR #14316)
TITLE: [ROCm] Enable chunked prefill/paged attention in MLA on ROCm (#14316)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: change_default; artifacts=L3.mla.triton_v0; Dossier shows L3.mla.triton_v0 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+2/-16); vllm/config.py (+2/-2)
LABELS: rocm, ready
BODY: This PR is largely just removing the guards in config.py to allow chunked prefill and paged attention in MLA. The LSE computation in the triton kernel doesn't work so we always fall back to flash attention in this case. ⏎  ⏎ I ran `lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct,trust_remote_code=True,enable_chunked_prefill=True --tasks gsm8k --num_fewshot 5 --batch_size auto` and got  ⏎  ⏎ ``` ⏎ vllm (pretrained=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct,trust_remote_code=True,enable_chunked_prefill=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7612|±  |0.0117| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.7437|±  |0.0120| ⏎ ``` ⏎  ⏎ CC: @LucasWilkinson

### L3-fb4c7f8ef0  (L3, 2025-03-13, sha fb4c7f8ef016, PR #14431)
TITLE: [Kernel] [V1] Further optimizations to ROCm (Triton) Backend to better handle GQA. (#14431)
SOURCES: path_core
STAGE1: optimize; artifacts=L3.triton.chunked_prefill_paged_decode; Dossier shows L3.triton.chunked_prefill_paged_decode behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode
FILES: vllm/attention/ops/chunked_prefill_paged_decode.py (+63/-40)
LABELS: ready
PERF_LINES: **TLDR:** This PR adds some further optimizations to `chunked_prefill_paged_decode` op to better handle models with GQA. Serving benchmarks using V1 indicate th | Request throughput (req/s):              48.15 | Output token throughput (tok/s):         9534.13 | Total Token throughput (tok/s):          19896.23 | Mean TTFT (ms):                          3563.95 | Median TTFT (ms):                 
BODY: **TLDR:** This PR adds some further optimizations to `chunked_prefill_paged_decode` op to better handle models with GQA. Serving benchmarks using V1 indicate that with these changes, we see a **25% improvement in throughput** for `llama3.1-8b` on an H100 vs. the current Triton implementation. With these changes, the throughput of the Triton implementation is only **8% worse than the V1 CUDA backend** (FlashAttention). ⏎  ⏎ Using `FlashAttentionBackend` from main on H100: ⏎ ``` ⏎ $ python benchmarks/benchmark_serving.py \ ⏎     --model meta-llama/Llama-3.1-8B-Instruct \ ⏎     --dataset-name sharegpt \ ⏎     --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     1000       ⏎ Benchmark duration (s):                  20.77      ⏎ Total input tokens:                      215196     ⏎ Total generated tokens:                  198001     ⏎ Request throughput (req/s):              48.15      ⏎ Output token throughput (tok/s):         9534.13    ⏎ Total Token throughput (tok/s):          19896.23   ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          3563.95    ⏎ Median TTFT (ms):                        3386.57    ⏎ P99 TTFT (ms):                           6444.51    ⏎ -----Time per Output Token (excl. 1st token)------ ⏎ Mean TPOT (ms):                          84.11      ⏎ Median TPOT (ms):                        48.47      ⏎ P99 TPOT (ms):                           213.98     ⏎ ---------------Inter-token Latency---------------- ⏎ Mean ITL (ms):                           37.38      ⏎ Median ITL (ms):                         23.74      ⏎ P99 ITL (ms):                            216.44     ⏎ ================================================== ⏎ ``` ⏎ Using `ROCmAttentionBackend` from main on H100: ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     1000       ⏎ Benchmark duration (s):                  28.22      ⏎ Total input tokens:                      215196     ⏎ Total generated tokens:                  197281     ⏎ Request throughput (req/s):              35.44      ⏎ Output token throughput (tok/s):         6991.12    ⏎ Total Token throughput (tok/s):          14617.11   ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          3483.50    ⏎ Median TTFT (ms):                        3607.34    ⏎ P99 TTFT (ms):                           6616.79    ⏎ -----Time per Output Token (excl. 1st token)------ ⏎ Mean TPOT (ms):                          99.14      ⏎ Median TPOT (ms):                        63.35      ⏎ P99 TPOT (ms):          …[truncated]

### L3-d47807ba08  (L3, 2025-03-13, sha d47807ba0806, PR #14769)
TITLE: [Attention] Remove slow setattr in MLA (#14769)
SOURCES: subject_keyword, release_notes
STAGE1: optimize; artifacts=L3.mla.common_v1; Dossier shows L3.mla.common_v1 behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rotary_embedding.py (+7/-2)
LABELS: ready
BODY: <img width="516" alt="image" src="https://github.com/user-attachments/assets/bc36adfb-fd5e-4dc7-9d34-f50eea1bf8d1" /> ⏎  ⏎ Partial undo of https://github.com/vllm-project/vllm/pull/14471 which was a revert of https://github.com/vllm-project/vllm/pull/14384

### L3-9532c49836  (L3, 2025-03-13, sha 9532c49836ad, PR #14770)
TITLE: [Attention] MLA get rid of materialization (#14770)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: optimize; artifacts=L3.flashinfer.trtllm_gen,L3.mla.triton_v0,L3.mla.common_v1; Dossier shows L3.flashinfer.trtllm_gen behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.triton_v0, L3.mla.common_v1
FILES: vllm/attention/backends/mla/common.py (+57/-210); vllm/envs.py (+0/-19); vllm/v1/attention/backends/mla/common.py (+58/-213); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+2/-57)
LABELS: ready, v1
PERF_LINES: It's actually better to just not materialize the absorbed `W_Q_UK` and `W_UV_O` as it reduces memory usage (and total flops) and instead compute using sequentia
BODY: Based on these calculations: ⏎  ⏎ https://docs.google.com/spreadsheets/d/17eoqEbhblvtNsRRlFSjCQnEXZiBxtLgZGKD4IgZUz38/edit?usp=sharing ⏎  ⏎ It's actually better to just not materialize the absorbed `W_Q_UK` and `W_UV_O` as it reduces memory usage (and total flops) and instead compute using sequential matmuls. One issue is that we do not have an FP8 bmm (which is needed if not materializing the absorbed matrix, materializing  absorbing allowed us to bypass this), so we instead decompress the matrices involved in the bmm to fp16/bf16. This also has the added benefit of dramatically reducing complexity. ⏎  ⏎ This PR is needed for DP attention since without it the weight materialization eats up too much of the GPU memory to make DP beneficial. ⏎  ⏎ This PR (minor regression in short context but seems worth it given the saved memory boosts long-context and enables DP-attention, also the short context measurements are a bit noisy) ⏎ ``` ⏎   backend  input_tokens  output_tokens  output_toks/s     req/s  median_itl_ms  median_ttft_ms ⏎ 3    vllm          1000           1000    1323.397915  1.323398      29.755860     2307.676603 ⏎ 2    vllm          5000           1000    1041.455043  1.041455      33.205620     5491.457423 ⏎ 4    vllm         10000           1000     874.563079  0.874563      36.871404     8508.498527 ⏎ 1    vllm         32000           1000     190.195698  0.190196      35.948243   108055.433401 ⏎ ``` ⏎  ⏎ Baseline (https://github.com/vllm-project/vllm/pull/14769) ⏎ ``` ⏎   backend  input_tokens  output_tokens  output_toks/s     req/s  median_itl_ms  median_ttft_ms ⏎ 3    vllm          1000           1000    1380.973383  1.380973      30.029079     2223.776098 ⏎ 2    vllm          5000           1000    1047.717832  1.047718      33.303024     5499.557093 ⏎ 4    vllm         10000           1000     586.460476  0.586460      36.855329     8512.637936 ⏎ 1    vllm         32000           1000     162.816157  0.162816      42.981104   115515.806204 ⏎ ``` ⏎  ⏎ correctness tests: ⏎  ⏎ ``` ⏎ VLLM_USE_V1=1  vllm serve /home/vllm-dev/DeepSeek-R1 --tensor-parallel-size 8 --trust-remote-code --disable-log-requests ⏎  ⏎ lm_eval --model local-completions --tasks gsm8k --model_args model=/home/vllm-dev/DeepSeek-R1,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=5,max_retries=3,tokenized_request ⏎ s=False --limit 100 ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.96|±  |0.0197| ⏎ |     |       |strict-match    |     5|exact_match|↑  | 0.96|±  |0.0197| ⏎ `` …[truncated]

### L3-d4d93db2c5  (L3, 2025-03-14, sha d4d93db2c54a, PR #13726)
TITLE: [V1] V1 Enablement Oracle  (#13726)
SOURCES: path_core, symbol_pickaxe
STAGE1: change_default; artifacts=L3.flash_attn.v1_backend,L3.platform.cuda_selection; Dossier shows L3.flash_attn.v1_backend behavior, dispatch, support, or dependency changed.
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: .buildkite/lm-eval-harness/configs/Minitron-4B-Base-FP8.yaml (+2/-2); .buildkite/lm-eval-harness/test_lm_eval_correctness.py (+5/-0); .buildkite/test-pipeline.yaml (+18/-16); tests/async_engine/conftest.py (+11/-0); tests/async_engine/test_api_server.py (+5/-1); tests/async_engine/test_async_llm_engine.py (+9/-0); tests/basic_correctness/test_chunked_prefill.py (+9/-0); tests/basic_correctness/test_cpu_offload.py (+7/-0); tests/basic_correctness/test_preemption.py (+9/-0); tests/compile/conftest.py (+14/-0); tests/conftest.py (+20/-0); tests/core/conftest.py (+11/-0); tests/detokenizer/__init__.py (+0/-0); tests/detokenizer/conftest.py (+10/-0); tests/detokenizer/test_disable_detokenization.py (+1/-0); tests/detokenizer/test_stop_checker.py (+0/-0); tests/detokenizer/test_stop_reason.py (+0/-0); tests/detokenizer/test_stop_strings.py (+141/-0); tests/distributed/test_pipeline_parallel.py (+12/-0); tests/encoder_decoder/test_e2e_correctness.py (+9/-0); tests/engine/conftest.py (+11/-0); tests/engine/test_multi_step_output_processor.py (+1/-1); tests/engine/test_stop_strings.py (+0/-165); tests/entrypoints/llm/test_lazy_outlines.py (+9/-0); tests/entrypoints/openai/test_chat_echo.py (+0/-3); tests/entrypoints/openai/test_root_path.py (+0/-3); tests/kernels/test_attention_selector.py (+21/-7); tests/kernels/test_encoder_decoder_attn.py (+10/-0); tests/kernels/test_rocm_attention_selector.py (+2/-1); tests/lora/test_llama_tp.py (+4/-0); tests/lora/test_lora_functions.py (+2/-2); tests/lora/test_lora_manager.py (+3/-0); tests/metrics/test_metrics.py (+9/-0); tests/models/decoder_only/language/test_gguf.py (+10/-10); tests/models/decoder_only/language/test_hybrid.py (+6/-8); tests/models/decoder_only/language/test_mamba.py (+0/-7); tests/models/decoder_only/language/test_mistral.py (+11/-10); tests/models/decoder_only/language/test_models.py (+9/-7); tests/models/decoder_only/vision_language/test_awq.py (+6/-1); tests/models/decoder_only/vision_language/test_models.py (+60/-29); (+56 more)
LABELS: documentation, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality
BODY: SUMMARY: ⏎ * adds oracle for V1 feature enablement ⏎ * removes usage of `envs.VLLM_USE_V1` from code base (now only used in tests + in oracle) ⏎ * enables V1 by default ⏎  ⏎ BEHAVIOR ⏎ ``` ⏎ # * If VLLM_USE_V1 is unset, we enable V1 for "supported features" ⏎ #   and fall back to V0 for experimental or unsupported features. ⏎ # * If VLLM_USE_V1=1, we enable V1 for supported + experimental ⏎ #   features and raise error for unsupported features. ⏎ # * If VLLM_USE_V1=0, we disable V1. ⏎ ``` ⏎  ⏎ UX: ⏎ - if the user specifies `.from_vllm_config`, we can autodetect the engine and it works nicely ⏎  ⏎ ```python ⏎ from vllm.engine.async_llm_engine import AsyncLLMEngine ⏎ args = AsyncEngineArgs(...) ⏎  ⏎ # sets VLLM_USE_V1 to 0 or 1 if unset by the user ⏎ config = args.create_engine_config()  ⏎  ⏎ # autoselects V1 engine if VLLM_USE_V1 is 1 ⏎ async_llm_engine = AsyncLLMEngine.from_vllm_config(config, ...) ⏎ ``` ⏎  ⏎ - if the user uses `AsyncLLMEngine.__init__()` this will not work ⏎ ```python ⏎ from vllm.engine.async_llm_engine import AsyncLLMEngine ⏎ args = AsyncEngineArgs(...) ⏎  ⏎ # sets VLLM_USE_V1 to 0 or 1 if unset by the user ⏎ config = args.create_engine_config() ⏎  ⏎ # this uses the V0 constructor, raises ValueError ⏎ async_llm_engine = AsyncLLMEngine(config, ...) ⏎  ⏎ # >> raise ValueError( ⏎ # >>                "Using V1 LLMEngine, but envs.VLLM_USE_V1=True. " ⏎ # >>                "This should not happen. As a workaround, try using " ⏎ # >>                "LLMEngine.from_vllm_config(...) or explicitly set " ⏎ # >>                "VLLM_USE_V1=0 or 1 and report this issue on Github.") ⏎ ``` ⏎  ⏎ TODOs: ⏎ * get CI green

### L3-5952d8ab61  (L3, 2025-03-15, sha 5952d8ab61a3, PR #14842)
TITLE: [Attention] Get rid of mla cache alignment (#14842)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: adapt_framework; artifacts=L3.mla.flashmla_v1_adapter,L3.mla.triton_v0; Removed MLA cache-alignment padding/env path used by MLA attention cache handling.
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+0/-10); vllm/utils.py (+0/-6); vllm/worker/cache_engine.py (+3/-39); tests/kernels/test_cache.py (+11/-28)
LABELS: ready
PERF_LINES: INFO 03-14 18:57:59 [executor_base.py:116] Maximum concurrency for 163840 tokens per request: 1.85x | INFO 03-14 20:43:52 [executor_base.py:116] Maximum concurrency for 163840 tokens per request: 2.06x | INFO 03-14 21:50:28 [executor_base.py:116] Maximum concurrency for 163840 tokens per request: 2.06x
BODY: With FlashMLA now being the default on Nvidia GPU and the fact that it [seemingly doesnt help the Triton backend anymore/ever-did](https://github.com/vllm-project/vllm/pull/12676#issuecomment-2652494624) (bit of a mystery). I think we can go ahead and rip this out reclaiming the memory lost to padding: ⏎  ⏎ ``` ⏎ VLLM_MLA_CUDA_MEM_ALIGN_KV_CACHE=1 VLLM_USE_V1=0 VLLM_USE_FLASHINFER_SAMPLER=1 vllm serve /home/vllm-dev/DeepSeek-R1 --tensor-parallel-size 8 --trust-remote-code --disable-log-requests ⏎  ⏎ INFO 03-14 18:57:59 [executor_base.py:111] # cuda blocks: 4739, # CPU blocks: 859 ⏎ INFO 03-14 18:57:59 [executor_base.py:116] Maximum concurrency for 163840 tokens per request: 1.85x ⏎  ⏎ Data Preview: ⏎   backend  input_tokens  output_tokens  output_toks/s     req/s  median_itl_ms  median_ttft_ms ⏎ 3    vllm          1000           1000     854.219899  0.854220      52.914861     3398.488862 ⏎ 2    vllm          5000           1000     712.867089  0.712867      50.757768    13790.549284 ⏎ 4    vllm         10000           1000     153.534461  0.153534     142.423406    19432.205255 ⏎ 1    vllm         32000           1000      49.799586  0.049800     142.251487   361420.859763 ⏎  ⏎ VLLM_MLA_CUDA_MEM_ALIGN_KV_CACHE=0 VLLM_USE_V1=0 VLLM_USE_FLASHINFER_SAMPLER=1 vllm serve /home/vllm-dev/DeepSeek-R1 --tensor-parallel-size 8 --trust-remote-code --disable-log-requests ⏎  ⏎ INFO 03-14 20:43:52 [executor_base.py:111] # cuda blocks: 5266, # CPU blocks: 954 ⏎ INFO 03-14 20:43:52 [executor_base.py:116] Maximum concurrency for 163840 tokens per request: 2.06x ⏎  ⏎ Data Preview: ⏎   backend  input_tokens  output_tokens  output_toks/s     req/s  median_itl_ms  median_ttft_ms ⏎ 3    vllm          1000           1000     835.217510  0.835218      51.893893     4393.823143 ⏎ 2    vllm          5000           1000     736.631281  0.736631      51.266942    13791.848651 ⏎ 4    vllm         10000           1000     155.732726  0.155733     139.090127    20673.930987 ⏎ 1    vllm         32000           1000      55.991667  0.055992     139.235539   362350.173109 ⏎ ``` ⏎  ⏎ This PR: ⏎ ``` ⏎ INFO 03-14 21:50:28 [executor_base.py:111] # cuda blocks: 5266, # CPU blocks: 954 ⏎ INFO 03-14 21:50:28 [executor_base.py:116] Maximum concurrency for 163840 tokens per request: 2.06x ⏎  ⏎ Data Preview: ⏎   backend  input_tokens  output_tokens  output_toks/s     req/s  median_itl_ms  median_ttft_ms ⏎ 3    vllm          1000           1000     846.707105  0.846707      53.101287     3406.700477 ⏎ 0    vllm          5000              5       1.085574  0.217115      79.162302      848.571256 ⏎ 2    vllm          5000           1000     717.566549  0.717567      53.10 …[truncated]

### L3-1e799b7ec1  (L3, 2025-03-17, sha 1e799b7ec1b1, PR #14910)
TITLE: [BugFix] Fix MLA + V1 + TP==1 causing reinitialization of cuda context (#14910)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.platform.cuda_selection; CUDA platform MLA guard avoids reinitializing CUDA context for TP=1 MLA runs.
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+1/-1)
LABELS: bug, ready
BODY: on main: ⏎  ⏎ ``` ⏎ # SPDX-License-Identifier: Apache-2.0 ⏎  ⏎ from vllm import LLM, SamplingParams ⏎  ⏎ # Sample prompts. ⏎ prompts = [ ⏎     "Hello, my name is", ⏎     "The president of the United States is", ⏎     "The capital of France is", ⏎     "The future of AI is", ⏎ ] ⏎ # Create a sampling params object. ⏎ sampling_params = SamplingParams(temperature=0.8, top_p=0.95) ⏎  ⏎ # Create an LLM. ⏎ llm = LLM( ⏎     model="deepseek-ai/DeepSeek-V2-Lite", ⏎     trust_remote_code=True, ⏎ ) ⏎ # Generate texts from the prompts. The output is a list of RequestOutput objects ⏎ # that contain the prompt, generated text, and other information. ⏎ outputs = llm.generate(prompts, sampling_params) ⏎ # Print the outputs. ⏎ for output in outputs: ⏎     prompt = output.prompt ⏎     generated_text = output.outputs[0].text ⏎     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}") ⏎ ``` ⏎  ⏎ is raising  ⏎  ⏎ ``` ⏎ ERROR 03-17 01:09:24 [core.py:337] RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method ⏎ ``` ⏎  ⏎ this fixes it by making sure we perform minimal imports in `check_and_update_config` (`backends.flashmla` imports alot more, and the function we need is actually defined in `ops.flashmla`, the narrower import works) ⏎  ⏎ TODO: this area of the code needs a refactor but this works as a temporary fix

### L3-89fca671fb  (L3, 2025-03-17, sha 89fca671fbb5, PR #14921)
TITLE: [V1] Default MLA to V1 (#14921)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: change_default; artifacts=L3.mla.common_v1; Changes MLA models to default toward the V1 engine/MLA path.
ARTIFACT_HINTS: -
FILES: vllm/engine/arg_utils.py (+1/-5)
LABELS: ready
BODY: 1. This is stable enough to recommend it. ⏎ 2. V0 is currently broken when serving a local method with trust remote code ⏎  ⏎ ``` ⏎ INFO 03-17 04:22:35 [llm_engine.py:241] Initializing a V0 LLM engine (v0.7.4.dev497+ga73e183e) with config: model='/home/vllm-dev/DeepSeek-R1', speculative_config=None, tokenizer='/home/vllm-dev/DeepSeek-R1', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=163840, download_dir=None, load_format=dummy, tensor_parallel_size=8, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=fp8, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar', reasoning_backend=None), observability_config=ObservabilityConfig(show_hidden_metrics=False, otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=None, served_model_name=/home/vllm-dev/DeepSeek-R1, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=False, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=True,  ⏎ WARNING 03-17 04:22:36 [multiproc_worker_utils.py:310] Reducing Torch parallelism from 64 threads to 1 to avoid unnecessary CPU contention. Set OMP_NUM_THREADS in the external environment to tune this value as needed. ⏎ INFO 03-17 04:22:36 [custom_cache_manager.py:19] Setting Triton cache manager to: vllm.triton_utils.custom_cache_manager:CustomCacheManager ⏎ ERROR 03-17 04:22:36 [engine.py:443] Can't pickle <class 'transformers_modules.DeepSeek-R1.configuration_deepseek.DeepseekV3Config'>: it's not the same object as transformers_modules.DeepSeek-R1.configuration_deepseek.DeepseekV3Config ⏎ ERROR 03-17 04:22:36 [engine.py:443] Traceback (most recent call last): ⏎ ERROR 03-17 04:22:36 [engine.py:443]   File "/home/vllm-dev/.cache/uv/archive-v0/ShHQ1oC1bVhD39JGW6Vk6/lib/python3.10/site-packages/vllm/engine/multiprocessing/engine.py", line 431, in run_mp_engine ⏎ ERROR 03-17 04:22:36 [engine.py:443]     engine = MQLLMEngine.from_vllm_config( ⏎ ERROR 03-17 04:22:36 [engine.py:443]   File "/home/vllm-dev/.cache/uv/archive-v0/ShHQ1oC1bVhD39JGW6Vk6/lib/python3.10/site-packages/vllm/engi …[truncated]

### L3-a597a57595  (L3, 2025-03-20, sha a597a57595b5, PR #14570)
TITLE: [Attention] Flash Attention 3 - fp8 (#14570)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
STAGE1: extend_support; artifacts=L3.flash_attn.v0_backend,L3.flash_attn.v1_backend,L3.flash_attn.fork_build; Adds FA3 FP8 KV-cache support and selection logic for FlashAttention backends.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.flash_attn.fork_build, L3.flashinfer.trtllm_gen, L3.mla.triton_v0, L3.dispatch.abstract_interface, L3.platform.cuda_selection
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/attention/backends/abstract.py (+1/-0); vllm/attention/backends/flash_attn.py (+68/-11); vllm/attention/backends/mla/common.py (+1/-1); vllm/attention/backends/utils.py (+0/-34); vllm/attention/layer.py (+7/-2); vllm/envs.py (+6/-1); vllm/fa_utils.py (+42/-0); vllm/platforms/cuda.py (+12/-9); vllm/v1/attention/backends/flash_attn.py (+40/-1); vllm/v1/worker/gpu_model_runner.py (+1/-1); tests/kernels/test_flash_attn.py (+68/-8); vllm/attention/__init__.py (+8/-4); vllm/model_executor/layers/quantization/kv_cache.py (+13/-3); vllm/v1/executor/multiproc_executor.py (+4/-0)
LABELS: ready, ci/build, v1
BODY: This PR add support for FP8 KV cache with FlashAttention3 (related PR in flash-attn [here](https://github.com/vllm-project/flash-attention/pull/50)) cc @LucasWilkinson Please do not merge this PR as long as it's not referencing vllm-project/flash-attention yet. ⏎  ⏎ FlashAttention (contrary to FlashInfer) does attention with all Q, K and V in FP8. ⏎ The performance is usually better than FlashInfer FP8 KV and FlashAttention 3 with bf16. ⏎  ⏎ I added support for v0 and v1 + some unit testing. ⏎  ⏎ Note that I've added a trick for checkpoints not providing q_scale and reuse the k_scale (with is something TRTLLM does fwiw). ⏎  ⏎ Also: I added a small QoS improvement when debugging v1: workers send back their traceback when they raise an exception.

### L3-2b22290ce0  (L3, 2025-03-20, sha 2b22290ce01b, PR #15243)
TITLE: [V1] Add flag to disable cascade attention (#15243)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: change_default; artifacts=L3.flash_attn.v1_backend; Adds a V1 flag to disable cascade attention, changing attention path selection.
ARTIFACT_HINTS: -
FILES: vllm/config.py (+2/-0); vllm/engine/arg_utils.py (+12/-0); vllm/v1/worker/gpu_model_runner.py (+9/-5)
LABELS: ready, v1
BODY: This PR adds a flag to disable cascade attention in V1. This could be useful when potential numerical issues are concerned.

### L3-0032903a5b  (L3, 2025-03-20, sha 0032903a5bb7, PR #15231)
TITLE: [Bugfix] detect alibi and revert to FA2 (#15231)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases
STAGE1: repair_correctness; artifacts=L3.flash_attn.v0_backend,L3.flash_attn.fa_utils; Detects ALiBi and falls back from unsupported FA3 to FA2.
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+2/-1); vllm/fa_utils.py (+9/-3)
LABELS: ready
ISSUES: #13810 [Usage]: vllm v0.7.2 can not support baichuan2 model
DEEP_STUDY: deep-study correctness case vllm:0032903a5b: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: On Hopper, Flash Attention 3 is enabled by default (since the v0.7.0 release), but FA3 does not support ALiBi positional encodings. This PR will revert to FA2 if alibi_slopes are passed in to `FlashAttentionBackend`. ⏎  ⏎ In the v0.8.1 release, the error when attempting to use a model with FA3 with ALiBi (such as [bigscience/bloom-1b1](https://huggingface.co/bigscience/bloom-1b1)) is: ⏎ ``` ⏎ RuntimeError: If cu_seqlens_k is passed in, then page table is not supported ⏎ ``` ⏎  ⏎ Now there is [change in vllm_flash_attn](https://github.com/vllm-project/flash-attention/pull/50/files#diff-30bbb250b1fdc0d0d0ada8e34377e835bbb92f4b48ca42807fe10524fe5a3730R199) that will raise an AssertionError if alibi is used with FA3. This improves the error message, but the server would still fail to come up. With this PR, vLLM will fall back to FA2 instead of erroring. ⏎  ⏎ FIX #13810

### L3-f8a08cb90d  (L3, 2025-03-21, sha f8a08cb90dc0, PR #14071)
TITLE: [V1] Enable Triton(ROCm) Attention backend for Nvidia GPUs (#14071)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:production-kernel-provenance
STAGE1: integrate; artifacts=L3.triton.v1_backend,L3.platform.cuda_selection; Enables the Triton attention backend for NVIDIA GPUs and platform selection.
ARTIFACT_HINTS: L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/engine/arg_utils.py (+1/-1); vllm/platforms/cuda.py (+8/-3); vllm/platforms/interface.py (+1/-0); vllm/platforms/rocm.py (+3/-2); vllm/v1/attention/backends/triton_attn.py (+10/-10)
LABELS: rocm, ready, v1
BODY: Related issue: #12724  ⏎ - Rename v1 `ROCmAttention` to `TritonAttention` and allow user to use it on Nvidia GPUs through `VLLM_ATTENTION_BACKEND=triton_attn_vllm_v1` ⏎ - Since v1 ROCm attn backend is implemented with Triton, it can be used on Nvidia GPUs too for triton kernel development. ⏎  ⏎ --- ⏎  ⏎ [details omitted]

### L3-91ca929dc7  (L3, 2025-03-21, sha 91ca929dc7aa, PR #15280)
TITLE: [V1] Fix wrong import path of get_flash_attn_version (#15280)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases
STAGE1: repair_build_dependency; artifacts=L3.mla.common_v1; Fixes MLA V1 import path for get_flash_attn_version.
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+1/-1)
LABELS: ready, v1
ISSUES: #15265 [Bug]: V1 with MLA enable throw error `cannot import name 'get_flash_attn_version' from 'vllm.attention.backends.utils'`
DEEP_STUDY: deep-study correctness case vllm:91ca929dc7: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: FIX #15265

### L3-dccf535f8e  (L3, 2025-03-23, sha dccf535f8edb, PR #15191)
TITLE: [V1] Enable V1 Fp8 cache for FA3 in the oracle (#15191)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: extend_support; artifacts=L3.flash_attn.v1_backend,L3.platform.cuda_selection; Enables V1 FP8 KV-cache use for FA3 on Hopper.
ARTIFACT_HINTS: L3.flash_attn.v0_backend, L3.flash_attn.v1_backend, L3.mla.triton_v0, L3.mla.common_v1, L3.platform.cuda_selection
FILES: vllm/attention/backends/flash_attn.py (+12/-4); vllm/attention/backends/mla/common.py (+1/-1); vllm/config.py (+0/-4); vllm/engine/arg_utils.py (+14/-3); vllm/platforms/cuda.py (+3/-5); vllm/v1/attention/backends/flash_attn.py (+6/-4); vllm/v1/attention/backends/mla/common.py (+1/-1); vllm/vllm_flash_attn/fa_utils.py (+6/-0); .gitignore (+2/-1)
LABELS: ready, v1
BODY: Now that FA3 Fp8 support has landed (https://github.com/vllm-project/vllm/pull/14570) we can enable Fp8 KV-caches for Hopper devices in V1 ⏎  ⏎ Update the oracle to allow this plus basic refactors ⏎  ⏎ Test script ⏎ ``` ⏎ # SPDX-License-Identifier: Apache-2.0 ⏎  ⏎ from vllm import LLM, SamplingParams ⏎  ⏎ # Sample prompts. ⏎ prompts = [ ⏎     "Hello, my name is", ⏎     "The president of the United States is", ⏎     "The capital of France is", ⏎     "The future of AI is", ⏎ ] ⏎ # Create a sampling params object. ⏎ sampling_params = SamplingParams(temperature=0.8, top_p=0.95) ⏎  ⏎ if __name__ == '__main__': ⏎     # Create an LLM. ⏎     llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct", kv_cache_dtype="fp8") ⏎     # Generate texts from the prompts. The output is a list of RequestOutput objects ⏎     # that contain the prompt, generated text, and other information. ⏎     outputs = llm.generate(prompts, sampling_params) ⏎     # Print the outputs. ⏎     for output in outputs: ⏎         prompt = output.prompt ⏎         generated_text = output.outputs[0].text ⏎         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}") ⏎ ``` ⏎  ⏎ Results: ⏎ ``` ⏎ (vllm) lwilkinson@beaker:~/code/vllm$ python examples/offline_inference/basic/basic.py  ⏎ .... ⏎ Prompt: 'Hello, my name is', Generated text: " Rachel and I'm a software developer. I'm new to the tech industry and" ⏎ Prompt: 'The president of the United States is', Generated text: ' the head of state and government of the United States. The president is elected by' ⏎ Prompt: 'The capital of France is', Generated text: ' Paris. You can visit the Eiffel Tower, the Louvre, and' ⏎ Prompt: 'The future of AI is', Generated text: ' a complex and multifaceted topic that spans various fields, including philosophy, ethics' ⏎ [rank0]:[W320 06:15:16.475614640 ProcessGroupNCCL.cpp:1496] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator()) ⏎  ⏎ (vllm) lwilkinson@beaker:~/code/vllm$ VLLM_USE_V1=0 python examples/offline_inference/basic/basic.py  ⏎ .... ⏎ Prompt: 'Hello, my name is', Generated text: " Alex. I've been spending a lot of time online lately, and I'm" ⏎ Prompt: 'The president of the United States is', Generated text: ' elected by the people through the Electoral College system. This system was established by the' ⏎ Prompt: 'The capital of France is', Generated text: ' Paris.\nThe capital of Germany is Berlin.\nThe capital of the United States is' ⏎ Prompt: 'The future of AI is', Generated text: " here, and it's making a huge impact on our lives. Fro …[truncated]

### L3-051da7efe3  (L3, 2025-03-25, sha 051da7efe395, PR #15160)
TITLE: Fix CUDA kernel index data type in vllm/csrc/quantization/gptq_marlin/awq_marlin_repack.cu +10 (#15160)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.rocm.custom_paged; Changes ROCm attention kernel index types to match CUDA built-ins.
ARTIFACT_HINTS: L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+26/-26); csrc/quantization/gptq_marlin/awq_marlin_repack.cu (+6/-6); csrc/quantization/gptq_marlin/gptq_marlin.cu (+14/-14); csrc/quantization/gptq_marlin/gptq_marlin_repack.cu (+8/-8); csrc/quantization/marlin/dense/marlin_cuda_kernel.cu (+5/-5); csrc/quantization/marlin/qqq/marlin_qqq_gemm_kernel.cu (+7/-7); csrc/quantization/marlin/sparse/marlin_24_cuda_kernel.cu (+7/-7)
LABELS: ready
BODY: Summary: ⏎ CUDA kernel variables matching the type `(thread|block|grid).(Idx|Dim).(x|y|z)` [have the data type `uint`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-variables). ⏎  ⏎ Many programmers mistakenly use implicit casts to turn these data types into `int`. In fact, the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) it self is inconsistent and incorrect in its use of data types in programming examples. ⏎  ⏎ The result of these implicit casts is that our kernels may give unexpected results when exposed to large datasets, i.e., those exceeding >~2B items. ⏎  ⏎ While we now have linters in place to prevent simple mistakes (D71236150), our codebase has many problematic instances. This diff fixes some of them. ⏎  ⏎ Differential Revision: D71355454

### L3-33437bc6e7  (L3, 2025-03-25, sha 33437bc6e7af, PR #15492)
TITLE: [BugFix] Fix nightly MLA failure (FA2 + MLA chunked prefill, i.e. V1, producing bad results) (#15492)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.merge.triton_lse; Fixes merge_attn_states handling for zero-length contexts in MLA chunked prefill.
ARTIFACT_HINTS: L3.merge.triton_lse
FILES: vllm/attention/ops/triton_merge_attn_states.py (+9/-0)
LABELS: ready
BODY: For MLA chunked prefill we use `merge_attn_states` to compute the context for a chunked-prefill, this sometimes leads to 0 length contexts when regular prefills and chunked-prefills are mixed in the same batch. FA2 and FA3 have differences in what the return when the sum-exp is 0, (i.e. what to return for log(0) which is technically undefined), FA3 returns -inf while FA2 returns inf. The code was written for FA3, i.e. -inf (I think this is also just the more logical choice) resulting in failures when running on A100s that default to FA2. This PR updates `merge_attn_states` to handle both cases (easier and less error prone than trying to update FA2).

### L3-ecff8309a3  (L3, 2025-03-26, sha ecff8309a3ca, PR #15557)
TITLE: [ROCm] Env variable to trigger custom PA (#15557)
SOURCES: path_core
STAGE1: change_default; artifacts=L3.rocm.rocm_flash_attn_v0; Restores environment control for selecting/disabling ROCm custom paged attention.
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+2/-1); vllm/envs.py (+6/-0)
LABELS: ready
BODY: Returning the trigger env for the custom PA that got lost during upstreaming of this kernel ⏎ This would allow to force disable custom PA kernel and fall back to the default implementation

### L3-8958217ad5  (L3, 2025-03-27, sha 8958217ad5a6, PR #15211)
TITLE: [Bugfix] Fix use_cascade_attention handling for Alibi-based models on vllm/v1 (#15211)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.flash_attn.v1_backend; Fixes cascade-attention enablement for ALiBi models in V1.
ARTIFACT_HINTS: -
FILES: vllm/utils.py (+13/-1); vllm/v1/worker/gpu_model_runner.py (+5/-2)
LABELS: ready, v1
BODY: When using Alibi-based models like MPT, the following assertion error causes. ⏎ ```bash ⏎ AssertionError: Cascade attention does not support ALiBi. ⏎ ``` ⏎  ⏎ This is because that the determination logic for `use_cascade` in `gpu_model_runner.py` incorrectly uses a hard-coded `use_alibi=False`. Therefore, `cascade_attention` is wrongly enabled. ⏎  ⏎ This PR allows setting `use_alibi` based on the alibi configuration specified in the `config.json` for MPT models.

### L3-e73ff24e31  (L3, 2025-04-02, sha e73ff24e31d2, PR #15720)
TITLE: [ROCM][KERNEL] Paged attention for V1 (#15720)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: integrate; artifacts=L3.rocm.custom_paged,L3.triton.chunked_prefill_paged_decode,L3.platform.rocm_selection; Adapts ROCm custom paged attention for V1 and fallback/prefix paths.
ARTIFACT_HINTS: L3.paged.python_wrapper, L3.triton.prefix_prefill, L3.triton.chunked_prefill_paged_decode, L3.triton.v1_backend, L3.rocm.custom_paged, L3.rocm.rocm_flash_attn_v0, L3.platform.rocm_selection
FILES: csrc/rocm/attention.cu (+73/-32); vllm/_custom_ops.py (+4/-2); vllm/attention/backends/rocm_flash_attn.py (+6/-18); vllm/attention/ops/chunked_prefill_paged_decode.py (+103/-54); vllm/attention/ops/paged_attn.py (+2/-0); vllm/attention/ops/prefix_prefill.py (+1/-0); vllm/platforms/rocm.py (+20/-1); vllm/v1/attention/backends/triton_attn.py (+1/-0); csrc/rocm/ops.h (+3/-2); csrc/rocm/torch_bindings.cpp (+3/-1); requirements/common.txt (+1/-1); requirements/test.in (+1/-1); tests/kernels/test_prefix_prefill.py (+4/-0)
LABELS: ready, ci/build, v1
PERF_LINES: Request throughput (req/s):              36.35 | Output token throughput (tok/s):         7164.57 | Total Token throughput (tok/s):          14987.32 | Mean TTFT (ms):                          5068.86 | Median TTFT (ms):                        4666.36 | P99 TTFT (ms):                           10794.20 | Mean TPOT (ms):                          59.98 | Median TPOT (ms):                        58.6
BODY: Adopting ROCM Paged Attention to be use in V1 FA as alternative to Triton kernel. Perf I see: ⏎  ⏎ Baseline (VLLM_ROCM_CUSTOM_PAGED_ATTN=0 and --no-enable-prefix-caching): ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     1000 ⏎ Benchmark duration (s):                  27.51 ⏎ Total input tokens:                      215196 ⏎ Total generated tokens:                  197090 ⏎ Request throughput (req/s):              36.35 ⏎ Output token throughput (tok/s):         7164.57 ⏎ Total Token throughput (tok/s):          14987.32 ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          5068.86 ⏎ Median TTFT (ms):                        4666.36 ⏎ P99 TTFT (ms):                           10794.20 ⏎ -----Time per Output Token (excl. 1st token)------ ⏎ Mean TPOT (ms):                          59.98 ⏎ Median TPOT (ms):                        58.69 ⏎ P99 TPOT (ms):                           87.44 ⏎ ---------------Inter-token Latency---------------- ⏎ Mean ITL (ms):                           43.91 ⏎ Median ITL (ms):                         34.99 ⏎ P99 ITL (ms):                            95.14 ⏎ ================================================== ⏎ ``` ⏎  ⏎ With change (--no-enable-prefix-caching): ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Successful requests:                     1000 ⏎ Benchmark duration (s):                  23.97 ⏎ Total input tokens:                      215196 ⏎ Total generated tokens:                  197119 ⏎ Request throughput (req/s):              41.73 ⏎ Output token throughput (tok/s):         8225.24 ⏎ Total Token throughput (tok/s):          17204.79 ⏎ ---------------Time to First Token---------------- ⏎ Mean TTFT (ms):                          4853.87 ⏎ Median TTFT (ms):                        4488.42 ⏎ P99 TTFT (ms):                           10301.76 ⏎ -----Time per Output Token (excl. 1st token)------ ⏎ Mean TPOT (ms):                          56.14 ⏎ Median TPOT (ms):                        54.89 ⏎ P99 TPOT (ms):                           82.79 ⏎ ---------------Inter-token Latency---------------- ⏎ Mean ITL (ms):                           39.92 ⏎ Median ITL (ms):                         32.00 ⏎ P99 ITL (ms):                            94.70 ⏎ ================================================== ⏎ ``` ⏎  ⏎ latency: ⏎ ``` ⏎ python benchmarks/benchmark_latency.py --model /data/models/Llama-3.1-8B-Instruct --input-len 2048 --output-len 2048 --batch-size 64 --num-iters-warmup 1 --num-iters 3 ⏎ ``` ⏎ V0 ⏎   Avg latency: 40.9744499316439 seconds ⏎  ⏎ V1 ⏎   upstream (VLLM_ROCM_CUSTOM_PAGED_ATTN=0): ⏎   Avg latency: 46.44388557784259 seconds ⏎  ⏎   with change: …[truncated]

### L3-40a36ccfeb  (L3, 2025-04-04, sha 40a36ccfeb49, PR #15717)
TITLE: [ROCm][Bugfix] Use platform specific FP8 dtype (#15717)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.triton.prefix_prefill; Prefix prefill uses platform-specific FP8 dtype on ROCm.
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+1/-1)
LABELS: ready
BODY: 

### L3-620fc2d09e  (L3, 2025-04-05, sha 620fc2d09ed8, PR #16112)
TITLE: [Model] fix model testing for TeleChat2ForCausalLM and V0 llama4 (#16112)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.flash_attn.v0_backend; Adds iRoPE parameter/warning support to V0 FlashAttention path for Llama4.
ARTIFACT_HINTS: L3.flash_attn.v0_backend
FILES: vllm/attention/backends/flash_attn.py (+5/-0); vllm/model_executor/models/telechat2.py (+6/-2)
BODY: 1. `TeleChat2ForCausalLM` inherits `LlamaForCausalLM` which requires a change to `_init_model` signature ⏎ 2. irope is not supported for V0 attention, we need to add the parameter and warning.

### L3-e1a2c699dd  (L3, 2025-04-08, sha e1a2c699dda8, PR #16209)
TITLE: [BugFix] Fix Llama4 - Index Error When Single Request Near Max Context (#16209)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.flash_attn.v1_backend; Fixes V1 FlashAttention index error near max context for Llama4.
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+1/-1)
LABELS: ready, v1
BODY: Fix for: https://github.com/vllm-project/vllm/issues/16157

### L3-2976dc27e9  (L3, 2025-04-08, sha 2976dc27e9dc, PR #16198)
TITLE: [Bug] [ROCm] Fix Llama 4 Enablement Bug on ROCm: V0 ROCmFlashAttentionImpl and Triton Fused MoE bugs (#16198)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.rocm.rocm_flash_attn_v0; Fixes ROCmFlashAttentionImpl use_irope constructor compatibility.
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+5/-1); vllm/utils.py (+9/-8); vllm/model_executor/layers/fused_moe/fused_moe.py (+2/-0)
LABELS: ready
BODY: # Description ⏎ This PR fixes two bugs: ⏎  ⏎ 1. `TypeError: ROCmFlashAttentionImpl.__init__() got an unexpected keyword argument 'use_irope'` ⏎ 2. Fix the `topk_weights` in `invoke_fused_moe_kernel` being not contiguous under V1 + ROCm + torch.compile + Dynamo + hipgraph mode.

### L3-e9528f6dc6  (L3, 2025-04-11, sha e9528f6dc614, PR #16173)
TITLE: [Kernel] support merge_attn_states CUDA kernel, 3x speedup (#16173)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: introduce; artifacts=L3.merge.cuda_lse; Introduces CUDA merge_attn_states kernel and binding as faster LSE merge.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_inline_cmake, L3.merge.cuda_lse, L3.mla.triton_v0, L3.mla.common_v1
FILES: CMakeLists.txt (+1/-0); csrc/attention/merge_attn_states.cu (+173/-0); csrc/ops.h (+9/-0); csrc/torch_bindings.cpp (+15/-0); vllm/_custom_ops.py (+11/-0); vllm/attention/backends/mla/common.py (+1/-2); vllm/attention/ops/merge_attn_states.py (+42/-0); vllm/v1/attention/backends/flash_attn.py (+1/-1); vllm/v1/attention/backends/mla/common.py (+1/-1); tests/kernels/test_merge_attn_states.py (+265/-0)
LABELS: ready, ci/build, v1
PERF_LINES: Use CUDA kernel instead of Triton to minimize CPU overhead. Compared to the Triton kernel, the CUDA kernel implemented in this PR can achieve a maximum speedup  | | tokens | heads | headsize | dtype | device | torch | triton | cuda | speedup | | | 256 | 16 | 128 | float32 | L20 | 0.15258ms | 0.05647ms | 0.01638ms | 3.4475x | | | 512 | 16 | 128 | float32 | L20 | 0.14940ms | 0.05417ms | 0.01654ms | 
BODY: base on [vllm/attention/ops/triton_merge_attn_states.py](https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_merge_attn_states.py) ⏎  ⏎ Use CUDA kernel instead of Triton to minimize CPU overhead. Compared to the Triton kernel, the CUDA kernel implemented in this PR can achieve a maximum speedup of over `3x`. @WoosukKwon, End2End performance improved for R1 with PP=3 + TP=8 on L20,  4K IN:1K OUT (TTFT 5687.80 ms -> 5654.02 ms). The performance of inference will not degrade. ⏎  ⏎ ## Performance ⏎  ⏎ cases for MLA with TP=8, num query heads per rank is 16, headsize is 128. ⏎  ⏎ | tokens | heads | headsize | dtype | device | torch | triton | cuda | speedup | ⏎ | --- | --- | --- | --- | --- | --- | --- | --- | --- | ⏎ | 256 | 16 | 128 | float32 | L20 | 0.15258ms | 0.05647ms | 0.01638ms | 3.4475x | ⏎ | 512 | 16 | 128 | float32 | L20 | 0.14940ms | 0.05417ms | 0.01654ms | 3.2749x | ⏎ | 613 | 16 | 128 | float32 | L20 | 0.14996ms | 0.05386ms | 0.01628ms | 3.3080x | ⏎ | 1024 | 16 | 128 | float32 | L20 | 0.14817ms | 0.05432ms | 0.01618ms | 3.3583x | ⏎ | 1536 | 16 | 128 | float32 | L20 | 0.14960ms | 0.05878ms | 0.01643ms | 3.5774x | ⏎ | 4096 | 16 | 128 | float32 | L20 | 0.38063ms | 0.12160ms | 0.06748ms | 1.8021x | ⏎ | 256 | 16 | 128 | float16 | L20 | 0.14776ms | 0.05509ms | 0.01567ms | 3.5165x | ⏎ | 512 | 16 | 128 | float16 | L20 | 0.14807ms | 0.05524ms | 0.01551ms | 3.5621x | ⏎ | 613 | 16 | 128 | float16 | L20 | 0.14843ms | 0.05380ms | 0.01557ms | 3.4564x | ⏎ | 1024 | 16 | 128 | float16 | L20 | 0.14945ms | 0.05437ms | 0.01556ms | 3.4938x | ⏎ | 1536 | 16 | 128 | float16 | L20 | 0.15718ms | 0.05836ms | 0.01557ms | 3.7494x | ⏎ | 4096 | 16 | 128 | float16 | L20 | 0.31852ms | 0.08372ms | 0.01955ms | 4.2813x | ⏎ | 256 | 16 | 128 | bfloat16 | L20 | 0.14842ms | 0.05381ms | 0.01546ms | 3.4801x | ⏎ | 512 | 16 | 128 | bfloat16 | L20 | 0.14782ms | 0.05356ms | 0.01536ms | 3.4858x | ⏎ | 613 | 16 | 128 | bfloat16 | L20 | 0.14848ms | 0.05320ms | 0.01547ms | 3.4398x | ⏎ | 1024 | 16 | 128 | bfloat16 | L20 | 0.14935ms | 0.05376ms | 0.01562ms | 3.4423x | ⏎ | 1536 | 16 | 128 | bfloat16 | L20 | 0.15765ms | 0.05934ms | 0.01572ms | 3.7736x | ⏎ | 4096 | 16 | 128 | bfloat16 | L20 | 0.31912ms | 0.08524ms | 0.01925ms | 4.4283x | ⏎  ⏎ ## Correctness  ⏎  ⏎ - float16 (performance & correctness) ⏎  ⏎ ```bash ⏎ pytest -s test_merge_attn_states.py ⏎ ---------------------------------------------------------------------------------------------------- ⏎ NUM_TOKENS:512, NUM_HEADS:16, HEAD_SIZE:128, DTYPE: torch.float16, Device: NVIDIA L20 ⏎  Torch time: 0.149299ms ⏎ Triton time: 0.050995ms ⏎   CUDA time: 0.015722ms, Performance: 3.24364x ⏎ -------------------- …[truncated]

### L3-280d62b8a2  (L3, 2025-04-15, sha 280d62b8a2cf, PR #16123)
TITLE: [Kernel] Remove redundant Exp calculations (#16123)
SOURCES: path_core
STAGE1: optimize; artifacts=L3.merge.triton_lse; Removes redundant exp calculations in Triton merge_attn_states.
ARTIFACT_HINTS: L3.merge.triton_lse
FILES: vllm/attention/ops/triton_merge_attn_states.py (+6/-3)
LABELS: ready
BODY: Remove redundant Exp calculations

### L3-e82ee40de3  (L3, 2025-04-16, sha e82ee40de336, PR #16693)
TITLE: [Bugfix][Kernel] fix potential cuda graph broken for merge_attn_states kernel (#16693)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: repair_correctness; artifacts=L3.merge.cuda_lse; Binds CUDA merge_attn_states to stream for CUDA graph correctness.
ARTIFACT_HINTS: L3.merge.cuda_lse
FILES: csrc/attention/merge_attn_states.cu (+15/-10)
LABELS: ready
BODY: Fix potential CUDA graph broken for the merge_attn_states kernel. A CUDA graph error related to merge_state was observed in sglang (https://github.com/sgl-project/sglang/issues/5404) and fixed in https://github.com/sgl-project/sglang/pull/5419. Since the merge_attn_states kernel is often active as a fundamental kernel in many scenarios, it would be better to bind the merge_attn_states kernel to the CUDA stream, as required by the CUDA graph. This binding won't affect the performance.

### L3-183dad7a85  (L3, 2025-04-17, sha 183dad7a8548, PR #13111)
TITLE: [Attention] Update to lastest FA3 code (#13111)
SOURCES: path_core, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, corpus:kernel-correctness-cases(introducing)
STAGE1: optimize; artifacts=L3.flash_attn.v1_backend,L3.flash_attn.fork_build,L3.mla.common_v1; Updates FA3 fork pin and adapts FlashAttention/MLA backends for latest FA3 code.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_build, L3.mla.triton_v0, L3.mla.common_v1
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/attention/backends/mla/common.py (+92/-90); vllm/attention/backends/utils.py (+25/-1); vllm/v1/attention/backends/flash_attn.py (+58/-1); vllm/v1/attention/backends/mla/common.py (+65/-25)
LABELS: ready, ci/build, v1
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:d0da99fb70 (fix PR 16998) || deep-study: introduced the defect fixed in case vllm:0f87d8f7b2 (fix PR 17574)
BODY: NOTE: Tested MLA on AMD V0 is working, V1 is broken but is also broken on main ⏎  ⏎ Perf: https://docs.google.com/spreadsheets/d/1U5lsoCKuWq99Cz1QbWkc0dBn1bij1Ifb3tphE2UXJj0/edit?usp=sharing ⏎   ⏎ # Main: ⏎  ⏎ ``` ⏎ -------------------------------------- ⏎ Full Command: ⏎ VLLM_USE_V1=0 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,max_num_batched_tokens=1024,enable_chunked_prefill=1 --task gsm8k --num_fewshot 5 --limit 10 ⏎  ⏎ Extracted Result Table: ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |  0.8|±  |0.1333| ⏎ |     |       |strict-match    |     5|exact_match|↑  |  0.8|±  |0.1333| ⏎ Log file saved at: logs/deepseek_v0_chunked_20250326_005420.log ⏎  ⏎ -------------------------------------- ⏎ Full Command: ⏎ VLLM_USE_V1=0 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384 --task gsm8k --num_fewshot 5 --limit 10 ⏎  ⏎ Extracted Result Table: ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |  0.8|±  |0.1333| ⏎ |     |       |strict-match    |     5|exact_match|↑  |  0.8|±  |0.1333| ⏎ Log file saved at: logs/deepseek_v0_nchunked_20250326_005531.log ⏎  ⏎ -------------------------------------- ⏎ Full Command: ⏎ VLLM_USE_V1=1 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384,max_num_batched_tokens=1024,enable_chunked_prefill=1 --task gsm8k --num_fewshot 5 --limit 10 ⏎  ⏎ Extracted Result Table: ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |  0.8|±  |0.1333| ⏎ |     |       |strict-match    |     5|exact_match|↑  |  0.8|±  |0.1333| ⏎ Log file saved at: logs/deepseek_v1_chunked_20250326_005722.log ⏎  ⏎ -------------------------------------- ⏎ Full Command: ⏎ VLLM_USE_V1=1 lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384 --task gsm8k --num_fewsho …[truncated]

### L3-0377b8310b  (L3, 2025-04-17, sha 0377b8310b28, PR #16673)
TITLE: [MLA] Simplification to batch P/D reordering (#16673)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: optimize; artifacts=L3.mla.common_v1; Moves MLA prefill/decode reordering to reduce duplicate sampling metadata work.
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/v1/attention/backends/mla/common.py (+5/-7); vllm/v1/worker/gpu_model_runner.py (+7/-9)
LABELS: ready, v1
BODY: I noticed that we're unnecessarily re-creating the sampling metadata twice when reordering the batch requests into prefill and decode groups for MLA. ⏎  ⏎ This moves the reorder op from the start of the `_prepare_inputs()` method to the end of the `_update_stats()` method (which is called right before).

### L3-aaec845f8e  (L3, 2025-04-18, sha aaec845f8ed7, PR #16431)
TITLE: [ROCm] [Attention] Cleanup ROCm output passing (#16431)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L3.rocm.rocm_flash_attn_v0; Cleans ROCm attention output passing to reduce copies and prepare quant fusion.
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0
FILES: vllm/attention/backends/rocm_flash_attn.py (+18/-23)
LABELS: rocm, ready
BODY: This PR cleans up the many copies & reallocations of outputs in the ROCm attention backend. This is needed for the attention+quant fusion described in #16220.

### L3-986537f1c3  (L3, 2025-04-22, sha 986537f1c3c8, PR #16684)
TITLE: [V1] V1 FlashInfer Attention (#16684)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:production-kernel-provenance
STAGE1: introduce; artifacts=L3.flashinfer.v1_backend; Introduces V1 FlashInfer backend with separate prefill/decode metadata.
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.platform.cuda_selection
FILES: vllm/engine/arg_utils.py (+10/-3); vllm/platforms/cuda.py (+3/-0); vllm/v1/attention/backends/flash_attn.py (+3/-4); vllm/v1/attention/backends/flashinfer.py (+639/-0); vllm/v1/attention/backends/mla/common.py (+3/-4); vllm/v1/worker/gpu_model_runner.py (+1/-1); tests/v1/e2e/test_cascade_attention.py (+9/-1)
LABELS: ready, v1
PERF_LINES: python benchmarks/benchmark_throughput.py --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 1000 --input-len 1024 --output-len 128 | Throughput: 25.63 requests/s, 30776.11 total tokens/s, 3280.51 output tokens/s | Throughput: 15.51 requests/s, 18616.93 total tokens/s, 1985.48 output tokens/s | Throughput: 25.09 requests/s, 30112.70 total tokens/s, 3212.02 output tokens/s | python benchmarks/b
BODY: Carrying on @aurickq work from here https://github.com/vllm-project/vllm/pull/14061. Thanks to @LucasWilkinson for helping debug qo_indptr issues. ⏎  ⏎ There are some performance issues in the original PR due to using `BatchPrefillWithPagedKVCacheWrapper` for all prefill and decode tokens. This PR separates prefill and decode tokens in V1 using the `reorder_batch()` functionality added for MLA, where the requests in the `input_batch` is reshuffled such that all decode tokens are at the front and all prefill tokens are at the back. This makes it easy to split the input/output to the attention implementation to contiguous chunks for decode and prefill. ⏎  ⏎ With this new implementation FlashInfer 0.2.1.post2 is close to within the performance of FA3. ⏎  ⏎ ### Evaluations  ⏎  ⏎ Evaluations on GSM8k: ⏎  ⏎ ``` ⏎ export VLLM_ATTENTION_BACKEND=FLASHINFER ⏎ lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto ⏎ vllm (pretrained=meta-llama/Llama-3.2-1B-Instruct,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.3351|±  | 0.013| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.3351|±  | 0.013| ⏎  ⏎ lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-7B-Instruct --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto ⏎ vllm (pretrained=Qwen/Qwen2.5-7B-Instruct,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8264|±  |0.0104| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.7885|±  |0.0112| ⏎  ⏎ lm_eval --model vllm --model_args pretrained=RedHatAI/QwQ-32B-FP8-dynamic,tensor_parallel_size=2 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto ⏎ vllm (pretrained=RedHatAI/QwQ-32B-FP8-dynamic,tensor_parallel_size=2,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.4321|±  |0.0136| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.7369|± …[truncated]

### L3-f961d7f6ef  (L3, 2025-04-22, sha f961d7f6ef14, PR #16973)
TITLE: [BugFix] Pass in correct VLLM config in FlashInfer backend (#13207) (#16973)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L3.flashinfer.v0_backend; Passes correct VLLM config into FlashInfer backend during CUDA graph capture.
ARTIFACT_HINTS: L3.flashinfer.v0_backend
FILES: vllm/attention/backends/flashinfer.py (+3/-3)
LABELS: ready
ISSUES: #13207 [Bug]: VLLM config not set when using Flash Infer backend.
BODY: This PR fixes the issue of the flashinfer backend complaining "Current VLLM config is not set" during CUDA graph capturing and closes #13207. The fix follows the proposed method in that issue.

### L3-30bc3e0f66  (L3, 2025-04-22, sha 30bc3e0f665e, PR #15893)
TITLE: [FEAT][ROCm]: Support AITER MLA (#15893)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: integrate; artifacts=NEW:mla_rocm_aiter_v0; Integrates AITER MLA ops into the V0 ROCm MLA backend path.
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.triton_v0, L3.platform.rocm_selection
FILES: vllm/attention/backends/mla/common.py (+18/-3); vllm/attention/backends/rocm_aiter_mla.py (+412/-0); vllm/attention/ops/rocm_aiter_mla.py (+42/-0); vllm/config.py (+1/-1); vllm/envs.py (+6/-0); vllm/platforms/interface.py (+1/-0); vllm/platforms/rocm.py (+31/-3); tests/kernels/test_attention_selector.py (+128/-21); tests/kernels/test_rocm_attention_selector.py (+28/-1)
LABELS: ready, ci/build
PERF_LINES: | Request throughput (req/s)              | 8.26                            | 3.78                                | 9.55                                   | | | Output token throughput (tok/s)         | 323.13                          | 150.96                              | 388.66                                 | | | Total Token throughput (tok/s)          | 8777.14                         | 4025
BODY: # Description ⏎ This PR integrates the AITER ops to improve the MLA functionality from [AITER flash_attn_varlen_func](https://github.com/ROCm/aiter/blob/86256916eee13a94c2213be2b7fde27a145d7103/aiter/ops/mha.py#L1155) and [AITER mla_decode_fwd](https://github.com/ROCm/aiter/blob/86256916eee13a94c2213be2b7fde27a145d7103/aiter/mla.py#L74) into vLLM, and will allow any up-coming optimizations in AITER kernel to be directly used and evaluated within the vLLM framework. ⏎  ⏎ ### Implementation ⏎ `ROCM_AITER_MLA` is introduced as an additional attention backend type for ROCm platform. ⏎ To support this backend the modules below are implemented `vllm/attention/backends/rocm_aiter_mla.py` ⏎  - `AiterMLABackend` inherits from `MLACommonBackend`. ⏎  - `AiterMLAMetadata` inherits from `MLACommonMetadata`: note that from this class the `advance_step` function utilizes `advance_step_flashinfer` function from VLLM cutom ops. ⏎  - `AiterMLAMetadataBuilder` inherits from `MLACommonMetadataBuilder`. ⏎  - `AiterMLAState` inherits from `MLACommonState`. ⏎  - `AiterMLAImpl` class inherits from `CommonMLAImpl`:  ⏎     Important notes for this class: ⏎     - `flash_attn_varlen_func` (FA function) used in this class is  AITER FA implementation (`flash_attn_varlen_func` from AITER package).  ⏎     - `_forward_decode` function in this class uses `mla_decode_fwd` kernel from AITER package. ⏎  ⏎ The MLACommon module has been refactored to reduce code duplication in its subclasses for  `advance_step`  function by invoking ops attention `ops.advance_step_flashattn` in a separate function `_ops_advance_step` that can be overridden by subclass.  ⏎  ⏎ To enable the backed the environment variable `VLLM_ATTN_BACKEND` can be set to `ROCM_AITER_MLA`.  ⏎ In case that the backend is not specified the `rocm.py` in `vllm/platforms` verifies whether `VLLM_ROCM_USE_AITER` and `VLLM_ROCM_USE_AITER_MLA` are both enabled or not to utilize this backend. Otherwise the selected backend is `TRITON_MLA`. ⏎  ⏎ Important Notes: ⏎  - AITER MLA currently only supports `block_size=1` and the variable `max_model_len=32768` has to be set. ⏎  - AITER MLA is suitable for DeepSeek models. ⏎  ⏎ ### Testing ⏎ In order to ensure correct attention backend is selected.  ⏎ MLA backend env backends has been added into the test cases in `tests/kernels/test_attention_selector.py`  ⏎  ⏎ ### Performance ⏎ Benchmark Serving Results Comparison ⏎  ⏎ | Metric                                       | Triton MLA (ROCm Flash Attention) | Triton MLA (Triton Flash Attention) | ROCm AITER MLA (AITER Flash Attention) | ⏎ |----------------------------------------------|----------------------------- …[truncated]

### L3-bc7c4d206b  (L3, 2025-04-22, sha bc7c4d206bbf, PR #13305)
TITLE: [Kernel][ROCM] Upstream prefix prefill speed up for vLLM V1 (#13305)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: optimize; artifacts=L3.triton.prefix_prefill; Reworks ROCm prefix-prefill Triton kernel with vectorization, online softmax, autotuning.
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/attention/ops/prefix_prefill.py (+821/-813); tests/core/block/e2e/test_correctness.py (+3/-3)
LABELS: rocm, ready, ci/build, v1
PERF_LINES: Speed up prefix prefill with vLLM V1 on AMG GPUs
BODY: Speed up prefix prefill with vLLM V1 on AMG GPUs ⏎  ⏎ Improvements: ⏎ 1. Vectorization in the context loop (most complex one as k cache shape is very specific) ⏎ 2. Refactoring for online softmax computation ⏎ 3. Refactoring to the kernel so autotune might select the best configs per shape ⏎ 4. Plus adding new spectrum of unrolling/staging in autotuner ⏎  ⏎ More details on triton kernel tunning: https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html ⏎  ⏎ see last comments

### L3-7e081ba7ca  (L3, 2025-04-22, sha 7e081ba7cad2, PR #17022)
TITLE: [BugFix] Revert ROCm Custom Paged Attention Env Flag Check (#17022)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:confirmed-reverts
STAGE1: revert; artifacts=L3.platform.rocm_selection; Restores ROCm custom paged-attention env guard removed by prior integration.
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+1/-0)
DEEP_STUDY: deep-study revert record: partial_revert of PR(s) 15001 reason=api_or_compat_break
BODY: #15001 Removes `envs.VLLM_ROCM_CUSTOM_PAGED_ATTN` from the `use_rocm_custom_paged_attention` check. This might cause vllm to use rocm custom paged attention even when the flag is not set. This PR reverts the check to maintain correctness.

### L3-047797ef90  (L3, 2025-04-22, sha 047797ef904f, PR #16902)
TITLE: [Bugfix] Triton FA function takes no keyword arguments (#16902)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L3.mla.triton_v0; Fixes Triton MLA call site when Triton flash-attention rejects keyword args.
ARTIFACT_HINTS: L3.mla.triton_v0
FILES: vllm/attention/backends/mla/common.py (+8/-1)
LABELS: rocm, ready
BODY: This PR resolves an existing bug in serving Deepseek model using MLA backend with triton flash attention when running the command below: ⏎  ⏎ `VLLM_MLA_DISABLE=0 VLLM_ATTENTION_BACKEND=TRITON_MLA VLLM_USE_TRITON_FLASH_ATTN=1 vllm serve deepseek-ai/DeepSeek-V3 --trust-remote-code --swap-space 16 --disable-log-requests -tp 8` ⏎  ⏎ Throws the error below: ⏎  ⏎ ""ERROR 04-21 04:42:55 [engine.py:448]   File "/app/vllm/vllm/attention/backends/mla/common.py", line 1431, in forward ⏎ ERROR 04-21 04:42:55 [engine.py:448]     output[:num_prefill_tokens] = self._forward_prefill( ⏎ ERROR 04-21 04:42:55 [engine.py:448]                                   ^^^^^^^^^^^^^^^^^^^^^^ ⏎ ERROR 04-21 04:42:55 [engine.py:448]   File "/app/vllm/vllm/attention/backends/mla/common.py", line 1315, in _forward_prefill ⏎ ERROR 04-21 04:42:55 [engine.py:448]     output = self._flash_attn_varlen_diff_headdims( ⏎ ERROR 04-21 04:42:55 [engine.py:448]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ ERROR 04-21 04:42:55 [engine.py:448]   File "/app/vllm/vllm/attention/backends/mla/common.py", line 1089, in _flash_attn_varlen_diff_headdims ⏎ ERROR 04-21 04:42:55 [engine.py:448]     attn_out = self.triton_fa_func( ⏎ ERROR 04-21 04:42:55 [engine.py:448]                ^^^^^^^^^^^^^^^^^^^^ ⏎ ERROR 04-21 04:42:55 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/autograd/function.py", line 575, in apply ⏎ ERROR 04-21 04:42:55 [engine.py:448]     return super().apply(*args, **kwargs)  # type: ignore[misc] ⏎ ERROR 04-21 04:42:55 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎ ERROR 04-21 04:42:55 [engine.py:448] TypeError: apply() takes no keyword arguments ⏎ ERROR 04-21 04:42:56 [multiproc_worker_utils.py:120] Worker VllmWorkerProcess pid 38646 died, exit code: -15 ⏎ INFO 04-21 04:42:56 [multiproc_worker_utils.py:124] Killing local vLLM worker processes ⏎ ""

### L3-d0da99fb70  (L3, 2025-04-22, sha d0da99fb70ba, PR #16998)
TITLE: [BugFix] llama4 fa3 fix - RuntimeError: scheduler_metadata must have shape (metadata_size) (#16998)
SOURCES: path_core, subject_keyword, release_notes, corpus:kernel-correctness-cases
STAGE1: repair_correctness; artifacts=L3.flash_attn.v1_backend; Fixes FA3 scheduler metadata shape and head-count handling for Llama4.
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/flash_attn.py (+48/-28)
LABELS: ready, v1
ISSUES: #16948 [Bug]: Run Llama4 Scout 16E w/ 10000 input length trigger vllm crashing, but run fine if use FA2. | #16997 [Bug]: Qwen2.5-VL-72B Inference
DEEP_STUDY: deep-study correctness case vllm:d0da99fb70: class=shape_alignment_edge; symptom=crash_or_exception; introducing=#13111
BODY: FIX https://github.com/vllm-project/vllm/issues/16948 ⏎ FIX #16997 ⏎  ⏎ Cause by: https://github.com/vllm-project/vllm/pull/13111 ⏎  ⏎ `+` fix for numheads (did not affect accuracy)

### L3-41ca7eb491  (L3, 2025-04-24, sha 41ca7eb49192, PR #16864)
TITLE: [Attention] FA3 decode perf improvement - single mma warp group support for head dim 128 (#16864)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
STAGE1: optimize; artifacts=L3.flash_attn.fork_build; Bumps vllm-flash-attn to include FA3 single-MMA warp-group decode improvement.
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build, v1
BODY: vLLM side of https://github.com/vllm-project/flash-attention/pull/63 ⏎  ⏎ Perf Results: ⏎  ⏎ 1000 in / 100 out ⏎  ⏎ https://docs.google.com/spreadsheets/d/1r7Hdgy1OGK7tU9DD4QWVt8IlA-688h0mHlYFeoD6Jtk/edit?usp=sharing ⏎  ⏎ 4xH100 ⏎ ![meta-llama_Llama-4-Scout-17B-16E](https://github.com/user-attachments/assets/d4bf3170-8f4e-4d15-b92c-37c1abe1c031) ⏎  ⏎ 1xH100 ⏎ ![mistralai_Mistral-Small-24B-Instruct-2501](https://github.com/user-attachments/assets/ffea227a-a3da-4aa7-a79b-5321d6029820)

### L3-a41351f363  (L3, 2025-04-25, sha a41351f363f3, PR #15734)
TITLE: [Quantization][FP8] Add support for FP8 models with input_scale for output projection and QK quantization (#15734)
SOURCES: path_core
STAGE1: extend_support; artifacts=L3.rocm.rocm_flash_attn_v0,L3.dispatch.abstract_interface; Adds FP8 output/QK quantization scale plumbing to ROCm FlashAttention backend.
ARTIFACT_HINTS: L3.rocm.rocm_flash_attn_v0, L3.dispatch.abstract_interface
FILES: vllm/attention/backends/abstract.py (+1/-0); vllm/attention/backends/rocm_flash_attn.py (+7/-0); vllm/attention/layer.py (+1/-0); vllm/config.py (+11/-0); vllm/engine/arg_utils.py (+17/-0); vllm/model_executor/layers/quantization/fp8.py (+5/-0); vllm/model_executor/layers/quantization/kv_cache.py (+36/-0); vllm/model_executor/layers/quantization/quark/quark.py (+27/-20)
LABELS: ready
BODY: This PR adds fp8 quantization support for: ⏎  ⏎ 1) Preserving FP8 quantization after FA output, so that output of FA into the next layer will be FP8 ⏎      - this uses the model parameter self_attn.q/k/v_proj.input_scale and passes it as _out_scale to the FA kernel ⏎ 2) During execution of FA loop, the quantity softmax(QK^T) can also be quantized as FP8 ⏎     - this uses the model parameter self_attn.prob_output_scale ⏎  ⏎ These are using Quark quantized models, in particular using Llama-3.1-8B-Instruct-FP8-QKV-Prob ⏎  ⏎ So far, only Llama is being supported, but more can be added later.
