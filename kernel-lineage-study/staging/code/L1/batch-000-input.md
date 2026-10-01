### L1-09de730dee  (L1, 2024-05-27, sha 09de730dee31, PR #484)
TITLE: Improve benchmark scripts & add more models (#484)
SOURCES: path_core, symbol_pickaxe
STAGE1: introduce; artifacts=L1.triton.fused_moe,L1.upstream.vllm.fused_topk; Adds in-tree Triton fused_moe copied from vLLM for routed experts.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe.py (+485/-0); benchmark/latency_throughput/bench_throughput.py (+8/-6); benchmark/mmlu/bench_other.py (+1/-1); python/sglang/srt/models/grok.py (+669/-0); python/sglang/srt/utils.py (+4/-3); python/sglang/test/test_utils.py (+6/-1)
BODY: 

### L1-53a7ebd89a  (L1, 2024-06-17, sha 53a7ebd89a0b, PR #553)
TITLE: Update fused_moe (#553)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: integrate; artifacts=L1.triton.fused_moe,L1.upstream.vllm.fused_topk; Splits fused_topk and routes MoE gating through vLLM topk_softmax.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe.py (+220/-183)
BODY: 

### L1-9465b668b9  (L1, 2024-06-24, sha 9465b668b9d3, PR #561)
TITLE: Allow running with vllm==0.4.3 (#561)
SOURCES: path_core, symbol_pickaxe
STAGE1: repair_build_dependency; artifacts=L1.triton.fused_moe,L1.upstream.vllm.fused_topk; Adds topk_softmax fallback for vLLM 0.4.3 compatibility.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe.py (+38/-3); python/sglang/srt/constrained/__init__.py (+11/-5)
BODY: There are some wired errors with fp8 and vllm==0.5.0.

### L1-2e6e62e156  (L1, 2024-06-26, sha 2e6e62e1562d, PR #567)
TITLE: Increase the number of thread limitation for tp worker managers. (#567)
SOURCES: path_core
STAGE1: retune; artifacts=L1.triton.fused_moe; Changes default FP8 Triton fused_moe block sizes and warps.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe.py (+30/-11); benchmark/latency_throughput/bench_throughput.py (+9/-4); benchmark/latency_throughput/test_latency.py (+3/-2); benchmark/mmlu/bench_sglang.py (+47/-55); playground/load_tokenizer.py (+10/-5); python/sglang/srt/constrained/fsm_cache.py (+2/-1); python/sglang/srt/hf_transformers_utils.py (+43/-3); python/sglang/srt/managers/controller/manager_single.py (+3/-2); python/sglang/srt/managers/controller/tp_worker.py (+1/-1)
BODY: - Increase the number of thread limitation for tp worker managers. ⏎ - Improve MMLU benchmark scripts ⏎ - Tune block sizes for fp8 fused_moe kernels ⏎ - Support sentencepiece tokenizer

### L1-ad3e4f1619  (L1, 2024-08-13, sha ad3e4f16199a, PR #1081)
TITLE: Update the mixtral to use the better FusedMoE layer (#1081)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: change_default; artifacts=L1.triton.fused_moe; Changes Mixtral model to use the newer FusedMoE layer by default.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/mixtral.py (+55/-253); python/sglang/srt/models/mixtral_quant.py (+0/-3); docs/en/model_support.md (+1/-1); test/srt/test_moe_serving_throughput.py (+1/-1)
BODY: 

### L1-a59636bb5e  (L1, 2024-08-14, sha a59636bb5e68, PR #1095)
TITLE: Update grok 1 model (#1095)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.triton.fused_moe; Repackages fused_moe into package with layer adapter for Grok/MoE use.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe/__init__.py (+1/-0); python/sglang/srt/layers/fused_moe/fused_moe.py (+165/-108); python/sglang/srt/layers/fused_moe/layer.py (+587/-0); benchmark/gsm8k/bench_sglang.py (+3/-0); python/sglang/bench_latency.py (+1/-0); python/sglang/srt/layers/activation.py (+0/-1); python/sglang/srt/layers/logits_processor.py (+4/-4); python/sglang/srt/model_executor/model_runner.py (+2/-2); python/sglang/srt/models/grok.py (+49/-395); python/sglang/srt/models/mixtral.py (+0/-1); python/sglang/srt/utils.py (+1/-2)
BODY: 

### L1-3a6e04185b  (L1, 2024-09-17, sha 3a6e04185b8d, PR #1420)
TITLE: [Feature, Hardware] Enable SGLang on AMD GPUs via PyTorch for ROCm (#1420)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.triton.fused_moe; Adds ROCm/AMD handling around FusedMoE layer and FP8 support.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe/layer.py (+27/-7); python/sglang/srt/layers/activation.py (+12/-0); python/sglang/srt/layers/attention_backend.py (+11/-7); python/sglang/srt/layers/layernorm.py (+12/-0); python/sglang/srt/layers/sampler.py (+10/-6); python/sglang/srt/lora/lora_manager.py (+5/-2); python/sglang/srt/models/deepseek_v2.py (+5/-1); python/sglang/srt/models/minicpm3.py (+5/-1); python/sglang/srt/server.py (+5/-0); python/sglang/srt/server_args.py (+7/-0); python/sglang/srt/utils.py (+5/-0)
PERF_LINES: root@x:/sglang# VLLM_MOE_PADDING=0 python -m sglang.bench_latency --batch-size 32 --input 1024 --output 8 --model dummy_half_grok1/ --tokenizer-path Xenova/grok | Prefill. latency: 25.79838 s, throughput:   1270.16 token/s | Decode.  latency: 0.53607 s, throughput:     59.69 token/s | Decode.  latency: 0.31325 s, throughput:    102.15 token/s | Decode.  latency: 0.04105 s, throughput:    779.47 to
BODY: ## Motivation ⏎ - Enable SGLang on AMD GPUs ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ - Bypass `FlashInfer` backend untill it is available on AMD/ROCm ⏎ - Add proper fix for AMD FP8 `e4m3fnuz` to support Fused_MoE ⏎ - Dependency over `vLLM>=0.5.5`, I modified `pyproject.toml` just to confirm that it works up to 0.6.0 as well. ⏎ - Misc. ⏎  ⏎ - TODO: follow-up to address one error (below) when `cuda-graph` is enabled. ⏎ ``` ⏎ File "/sglang/python/sglang/srt/layers/sampler.py", line 164, in top_k_top_p_min_p_sampling_from_probs_torch ⏎     min_p_thresholds = probs_sort[:, 0] * min_ps ⏎     TypeError: unsupported operand type(s) for *: 'Tensor' and 'NoneType' (where min_ps is None) ⏎ ``` ⏎  ⏎ ## How to run ⏎ - An example on one MI3xx (not performance benchmark) ⏎ ``` ⏎ root@x:/sglang# VLLM_MOE_PADDING=0 python -m sglang.bench_latency --batch-size 32 --input 1024 --output 8 --model dummy_half_grok1/ --tokenizer-path Xenova/grok-1-tokenizer --load-format dummy --tp 8 --quant fp8  --attention-backend triton --sampling-backend  pytorch --disable-cuda-graph ⏎ Warmup ... ⏎ Prefill. latency: 25.79838 s, throughput:   1270.16 token/s ⏎ Decode.  latency: 0.53607 s, throughput:     59.69 token/s ⏎ Decode.  latency: 0.31325 s, throughput:    102.15 token/s ⏎ Decode.  latency: 0.04105 s, throughput:    779.47 token/s ⏎ Decode.  latency: 0.04075 s, throughput:    785.31 token/s ⏎ Decode.  median latency: 0.17715 s, median throughput:    180.64 token/s ⏎ Total. latency: 26.730 s, throughput:   1230.70 token/s ⏎ Benchmark ... ⏎ Prefill. latency: 1.11868 s, throughput:  29291.72 token/s ⏎ Decode.  latency: 0.02593 s, throughput:   1234.10 token/s ⏎ Decode.  latency: 0.02575 s, throughput:   1242.87 token/s ⏎ Decode.  latency: 0.02559 s, throughput:   1250.46 token/s ⏎ Decode.  latency: 0.02646 s, throughput:   1209.23 token/s ⏎ Decode.  latency: 0.02574 s, throughput:   1243.42 token/s ⏎ Decode.  median latency: 0.02574 s, median throughput:   1243.15 token/s ⏎ Total. latency:  1.325 s, throughput:  24925.60 token/s ⏎ root@x:/sglang# ⏎ ``` ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/en/contributor_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/en/contributor_guide.md). ⏎ - [+] Update documentation as needed, including docstrings or example tutorials.

### L1-8d4ed42ad5  (L1, 2024-09-24, sha 8d4ed42ad51d, PR #1497)
TITLE: MoE torch compile (#1497)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.triton.fused_moe_patch; Introduces MoE torch.compile monkey patch adapter for fused_moe.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.fused_moe_patch
FILES: python/sglang/srt/layers/fused_moe/patch.py (+117/-0); python/sglang/srt/model_executor/cuda_graph_runner.py (+9/-5)
PERF_LINES: ## Bench Latency | python3 -m sglang.bench_latency --model deepseek-ai/DeepSeek-V2-Lite --disable-radix --trust-remote-code --input-len 128 --output-len 8 --batch 1 --enable-torch | Decode.  median latency: 0.00921 s, median throughput:    108.61 token/s | Total. latency:  0.101 s, throughput:   1352.80 token/s | Decode.  median latency: 0.00735 s, median throughput:    136.13 token/s | Total. lat
BODY: ## Motivation ⏎  ⏎ Temporarily workaround MoE torch compile with monkey patch. ⏎  ⏎ ## Bench Latency ⏎ ```bash ⏎ python3 -m sglang.bench_latency --model deepseek-ai/DeepSeek-V2-Lite --disable-radix --trust-remote-code --input-len 128 --output-len 8 --batch 1 --enable-torch-compile --max-torch-compile-bs 1 ⏎  ⏎ # bs=1, w/o torch compile ⏎ Decode.  median latency: 0.00921 s, median throughput:    108.61 token/s ⏎ Total. latency:  0.101 s, throughput:   1352.80 token/s ⏎  ⏎ # bs=1, w/ torch compile, skip moe (main) ⏎ Decode.  median latency: 0.00735 s, median throughput:    136.13 token/s ⏎ Total. latency:  0.086 s, throughput:   1574.70 token/s ⏎  ⏎ # bs=1, w/ torch compile + moe (this PR) ⏎ Decode.  median latency: 0.00682 s, median throughput:    146.71 token/s ⏎ Total. latency:  0.083 s, throughput:   1632.26 token/s ⏎ ``` ⏎  ⏎ ## Evaluation ⏎ ```bash ⏎ python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --port 30000 --trust-remote-code --host 0.0.0.0  --enable-torch-compile --max-torch-compile-bs 1 --max-running-requests 1 ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 200 ⏎  ⏎ Accuracy: 0.825 ⏎ Invalid: 0.000 ⏎ Latency: 189.962 s ⏎ Output throughput: 136.559 token/s ⏎ ```

### L1-5f65e2b830  (L1, 2024-10-30, sha 5f65e2b830a4, PR #1836)
TITLE: [Performance, Hardware] MoE weights padding to AMD MI300x GPUs (#1836)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.triton.fused_moe; Adds optional AMD MoE weight padding performance path.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe/fused_moe.py (+4/-3); python/sglang/srt/layers/fused_moe/layer.py (+28/-0)
PERF_LINES: Test shows approximate performance boost of prefill +2.2%, decode +3.0% for Grok-1 on setting: b32/i1024/o512
BODY: ## Motivation ⏎  ⏎ Padding MoE weights (last dim) to minimize Memory Channel Contention (only to AMD Instinct GPUs) ⏎ Test shows approximate performance boost of prefill +2.2%, decode +3.0% for Grok-1 on setting: b32/i1024/o512 ⏎  ⏎ ## Modifications ⏎  ⏎ As mentioned: fused_moe.py and layer.py ⏎ To enable this feature, set binary flag `MOE_PADDING=1` at command line, or `export MOE_PADDING=1` in console. ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/contributor_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/contributor_guide.md). ⏎ - [+] Update documentation as needed, including docstrings or example tutorials.

### L1-087ab83223  (L1, 2024-11-10, sha 087ab832236e, PR #1980)
TITLE: [Performance, Triton] Optimize over mask compute to tl.load in fused_moe_kernel (#1980)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: optimize; artifacts=L1.triton.fused_moe; Optimizes fused_moe_kernel mask computation before tl.load.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe/fused_moe.py (+23/-7); python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+7/-0)
PERF_LINES: Test shows `~0.5%` boost to prefill, `~1.0%` boost to median decode throughput over Grok-1 with `b32/i1023/o256` settings.
BODY: ## Motivation ⏎  ⏎ Test shows `~0.5%` boost to prefill, `~1.0%` boost to median decode throughput over Grok-1 with `b32/i1023/o256` settings. ⏎  ⏎ ## Modifications ⏎  ⏎ `fused_moe_kernel`: simplify the mask part for even K BLOCK sizes ⏎ `decode_attention`: adjust kernel args ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Update documentation as needed, including docstrings or example tutorials.

### L1-c3eac1b010  (L1, 2024-11-14, sha c3eac1b010b3, PR #2033)
TITLE: Fix torch.compile for MoE (#2033)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe_patch; Fixes MoE torch.compile patch and adds compile regression test.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.fused_moe_patch
FILES: python/sglang/srt/layers/fused_moe/patch.py (+4/-2); python/sglang/test/test_utils.py (+3/-2); test/srt/run_suite.py (+1/-0); test/srt/test_data_parallelism.py (+1/-1); test/srt/test_double_sparsity.py (+1/-1); test/srt/test_eval_accuracy_mini.py (+1/-1); test/srt/test_retract_decode.py (+1/-1); test/srt/test_torch_compile.py (+3/-3); test/srt/test_torch_compile_moe.py (+73/-0); test/srt/test_triton_attention_backend.py (+1/-1)
ISSUES: #2029 [Bug] Does Mixtral currently not support torch compile?
DEEP_STUDY: deep-study correctness case sglang:c3eac1b010: class=integration_backend_cudagraph; symptom=compile_or_build_failure; introducing=unknown
BODY: Fix https://github.com/sgl-project/sglang/issues/2029

### L1-f35cb46cc3  (L1, 2024-11-21, sha f35cb46cc376, PR #2111)
TITLE: ROCm: Fix MoE padding for none FP8 cases (#2111)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; Fixes ROCm MoE padding for non-FP8 cases.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe/fused_moe.py (+11/-4)
BODY: ## Motivation ⏎  ⏎ As mentioned. ⏎  ⏎ ## Modifications ⏎  ⏎ Separate padding size handling to FP8 and None FP8 cases. ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Update documentation as needed, including docstrings or example tutorials.

### L1-be0124bda0  (L1, 2024-11-24, sha be0124bda09d, PR #2163)
TITLE: Rename triton_fused_moe -> fused_moe_triton (#2163)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: introduce; artifacts=L1.triton.fused_moe,L1.triton.grok_variant; Renames fused_moe_triton and creates Grok-specific fused MoE variant.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.grok_variant
FILES: python/sglang/srt/layers/fused_moe/__init__.py (+0/-1); python/sglang/srt/layers/fused_moe_grok/__init__.py (+1/-0); python/sglang/srt/layers/fused_moe_grok/configs/E=8,N=4096,device_name=AMD_Instinct_MI300X,dtype=float8.json (+0/-0); python/sglang/srt/layers/fused_moe_grok/configs/E=8,N=8192,device_name=AMD_Instinct_MI300X,dtype=float8.json (+0/-0); python/sglang/srt/layers/fused_moe_grok/fused_moe.py (+0/-0); python/sglang/srt/layers/fused_moe_grok/layer.py (+3/-3); python/sglang/srt/layers/fused_moe_triton/__init__.py (+3/-3); python/sglang/srt/layers/fused_moe_triton/configs/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=1,N=3072,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=1,N=3072,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=1,N=3072,device_name=NVIDIA_H100_80GB_HBM3.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=1,N=3584,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=1,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=1,N=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=1,N=7168,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=1344,device_name=NVIDIA_A100-SXM4-40GB.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=1344,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=14336,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=14336,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=1792,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=1792,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=2688,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=2688,device_name=NVIDIA_H100_80GB_HBM3.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=3072,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=3072,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=3200,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=3584,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=6400,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=7168,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=16,N=800,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=64,N=1280,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=64,N=1280,device_name=NVIDIA_H100_80GB_HBM3.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=64,N=640,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); (+36 more)
BODY: 

### L1-b509db5832  (L1, 2024-11-24, sha b509db5832c9, PR #2153)
TITLE: feat: remove the dependency on FusedMoE (#2153)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: adapt_framework; artifacts=L1.triton.fused_moe; Copies Triton FusedMoE in-tree to remove external FusedMoE dependency.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/quantization/__init__.py (+15/-5); python/sglang/srt/layers/triton_fused_moe/__init__.py (+44/-0); python/sglang/srt/layers/triton_fused_moe/configs/README (+10/-0); python/sglang/srt/layers/triton_fused_moe/fused_moe.py (+858/-0); python/sglang/srt/layers/triton_fused_moe/layer.py (+631/-0); python/sglang/srt/models/deepseek_v2.py (+1/-1); python/sglang/srt/utils.py (+43/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e3938b2f9c  (L1, 2024-11-24, sha e3938b2f9c96, PR #2156)
TITLE: feat: update other MoE models deps (#2156)
SOURCES: path_core, symbol_pickaxe
STAGE1: adapt_framework; artifacts=L1.triton.fused_moe; Repoints MoE model imports from vLLM to SGLang triton_fused_moe.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe/layer.py (+1/-6); python/sglang/srt/layers/triton_fused_moe/fused_moe.py (+2/-0); python/sglang/srt/layers/triton_fused_moe/layer.py (+4/-2); python/sglang/srt/models/dbrx.py (+1/-1); python/sglang/srt/models/deepseek.py (+1/-1); python/sglang/srt/models/mixtral.py (+1/-1); python/sglang/srt/models/olmoe.py (+1/-1); python/sglang/srt/models/qwen2_moe.py (+1/-1); python/sglang/srt/models/xverse_moe.py (+1/-1); python/sglang/srt/utils.py (+15/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fa27161380  (L1, 2024-11-24, sha fa271613809b, PR #2161)
TITLE: fix: use torch.sum for compatible (#2161)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; Restores torch.sum combine path for vLLM compatibility issue.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/triton_fused_moe/fused_moe.py (+3/-2)
ISSUES: #2160 [Bug] FusedMoE compatible with vllm 0.6.3.post1
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ fix https://github.com/sgl-project/sglang/issues/2160 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-55842eb81a  (L1, 2024-11-25, sha 55842eb81a78, PR #2174)
TITLE: feat: fused_moe fp8 monkey patch (#2174)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
STAGE1: adapt_framework; artifacts=L1.triton.fused_moe; Monkey-patches FP8 MoE quantization to call SGLang fused_experts.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/__init__.py (+68/-18)
PERF_LINES: Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s] | Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:01<00:03,  1.19s/it] | Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:02<00:02,  1.29s/it] | Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:03<00:01,  1.30s/it] | Loading safetensors checkpoint shards: 100% Completed | 4/4 
BODY: ## Motivation ⏎  ⏎ as titled cc @ispobock  ⏎  ⏎ ``` ⏎ root@id:/sgl-workspace/sglang/test/srt# python3 test_mla_fp8.py ⏎ [2024-11-25 00:47:01] server_args=ServerArgs(model_path='neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8', tokenizer_path='neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8', tokenizer_mode='auto', skip_tokenizer_init=False, load_format='auto', trust_remote_code=True, dtype='auto', kv_cache_dtype='fp8_e5m2', quantization=None, context_length=None, device='cuda', served_model_name='neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8', chat_template=None, is_embedding=False, host='127.0.0.1', port=2157, mem_fraction_static=0.87, max_running_requests=None, max_total_tokens=None, chunked_prefill_size=8192, max_prefill_tokens=16384, schedule_policy='lpm', schedule_conservativeness=1.0, cpu_offload_gb=0, tp_size=2, stream_interval=1, random_seed=128310829, constrained_json_whitespace_pattern=None, watchdog_timeout=300, download_dir=None, base_gpu_id=0, log_level='info', log_level_http=None, log_requests=False, show_time_cost=False, enable_metrics=False, decode_log_interval=40, api_key=None, file_storage_pth='SGLang_storage', enable_cache_report=False, dp_size=1, load_balance_method='round_robin', dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, lora_paths=None, max_loras_per_batch=8, attention_backend='flashinfer', sampling_backend='flashinfer', grammar_backend='outlines', disable_radix_cache=False, disable_jump_forward=False, disable_cuda_graph=False, disable_cuda_graph_padding=False, disable_disk_cache=False, disable_custom_all_reduce=False, disable_mla=False, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_torch_compile=False, torch_compile_max_bs=32, cuda_graph_max_bs=160, torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, num_continuous_decode_steps=1, delete_ckpt_after_loading=False) ⏎ [2024-11-25 00:47:11 TP0] MLA optimization is turned on. Use triton backend. ⏎ [2024-11-25 00:47:11 TP0] Init torch distributed begin. ⏎ [2024-11-25 00:47:11 TP1] MLA optimization is turned on. Use triton backend. ⏎ [2024-11-25 00:47:11 TP1] Init torch distributed begin. ⏎ [2024-11-25 00:47:16 TP0] Load weight begin. avail mem=77.17 GB ⏎ [2024-11-25 00:47:16 TP1] Load weight begin. avail mem=77.18 GB ⏎ INFO 11-25 00:47:16 config.py:107] Replacing legacy 'type' key with 'rope_type' …[truncated]

### L1-dd5eba4c88  (L1, 2024-11-27, sha dd5eba4c8899, PR #2223)
TITLE: Remove fused_moe_grok (#2223)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: remove; artifacts=L1.triton.grok_variant; Deletes separate fused_moe_grok implementation and unifies on fused_moe_triton.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.grok_variant
FILES: python/sglang/srt/layers/fused_moe_grok/__init__.py (+0/-1); python/sglang/srt/layers/fused_moe_grok/fused_moe.py (+0/-692); python/sglang/srt/layers/fused_moe_grok/layer.py (+0/-630); python/sglang/srt/layers/fused_moe_triton/configs/E=8,N=4096,device_name=AMD_Instinct_MI300X,dtype=float8.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=8,N=8192,device_name=AMD_Instinct_MI300X,dtype=float8.json (+0/-0); python/sglang/srt/models/grok.py (+11/-48); 3rdparty/amd/tuning/benchmark_moe_rocm.py (+1/-1)
BODY: We do not want to maintain a separate fused moe folder. This unifies them

### L1-07ec07ad1f  (L1, 2024-12-03, sha 07ec07ad1fa5, PR #2327)
TITLE: Improve torch compile for fused moe (#2327)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: adapt_framework; artifacts=L1.triton.fused_moe_patch; Changes MoE torch.compile policy to bs=1 and dynamic=False.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.fused_moe_patch
FILES: python/sglang/srt/layers/fused_moe_patch.py (+20/-11); python/sglang/srt/model_executor/cuda_graph_runner.py (+16/-7); python/sglang/srt/model_executor/model_runner.py (+1/-1); benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py (+5/-2); test/srt/test_srt_engine.py (+1/-1); test/srt/test_torch_compile_moe.py (+2/-2)
BODY: Following the discussion from https://github.com/sgl-project/sglang/issues/2278. ⏎  ⏎ The final solution: ⏎ - Only do torch.compile for moe layers at bs=1 ⏎ - Skip torch.compile for moe layers when bs > 1 ⏎ - Add `dynamic=False` in places where the shape should be known

### L1-3d32e4a32c  (L1, 2024-12-06, sha 3d32e4a32c4c, PR #2371)
TITLE: Resubmit MoE-EP (#2371)
SOURCES: path_core, symbol_pickaxe, release_notes
STAGE1: introduce; artifacts=L1.ep.layer; Adds EP MoE layer and Triton helper kernels.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/ep_moe/__init__.py (+0/-0); python/sglang/srt/layers/ep_moe/kernels.py (+349/-0); python/sglang/srt/layers/ep_moe/layer.py (+661/-0); .github/workflows/pr-test.yml (+6/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/model_executor/model_runner.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+5/-3); python/sglang/srt/models/mixtral.py (+13/-5); python/sglang/srt/server_args.py (+23/-0); test/srt/test_moe_ep.py (+113/-0)
BODY: Resubmit the PR for MoE-EP.  Please refer to the details in the previous PR: https://github.com/sgl-project/sglang/pull/2203.

### L1-95f93f493a  (L1, 2024-12-07, sha 95f93f493a60, PR #2388)
TITLE: Fp8 MoE optimizations on AMD (#2388)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.triton.fused_moe; Restores AMD FP8 fused_moe_triton optimizations.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe_triton/fused_moe.py (+64/-21); python/sglang/srt/layers/quantization/fp8.py (+33/-1)
BODY: ## Motivation ⏎  ⏎  ⏎ Recover AMD optimizations over fused_moe_triton ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/issues/2347 ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Update documentation as needed, including docstrings or example tutorials.

### L1-5f2595be43  (L1, 2024-12-15, sha 5f2595be4302, PR #2485)
TITLE: hotfix: checking for HIP (#2485)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; Fixes HIP detection guard in fused_moe_triton layer.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe_triton/layer.py (+1/-1); python/sglang/srt/utils.py (+1/-7)
BODY: ## Motivation ⏎  ⏎ https://pytorch.org/docs/stable/notes/hip.html ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b532a5fd16  (L1, 2024-12-16, sha b532a5fd16d0, PR #2489)
TITLE: fix moe-ep accuracy issue for fp8 (#2489)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.ep.layer; Fixes EP MoE FP8 accuracy issue.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/ep_moe/layer.py (+4/-0)
PERF_LINES: Latency: 243.824 s | Output throughput: 1027.530 token/s
BODY: ## Motivation ⏎  ⏎  fix moe ep bug, when load fp8 model.  Links to related issues [link](https://github.com/sgl-project/sglang/issues/2482) ⏎  ⏎ Test model : neuralmagic/DeepSeek-Coder-V2-Instruct-FP8 ⏎ ``` ⏎ Accuracy: 0.932 ⏎ Invalid: 0.000 ⏎ Latency: 243.824 s ⏎ Output throughput: 1027.530 token/s ⏎ ``` ⏎  ⏎ cc: @ispobock  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ ## Checklist

### L1-e835a50021  (L1, 2024-12-24, sha e835a50021e0, PR #2563)
TITLE: Reorg moe code (#2563)
SOURCES: path_core, symbol_pickaxe, corpus:production-kernel-provenance
STAGE1: adapt_framework; artifacts=L1.triton.fused_moe,L1.routing.topk_py,L1.ep.layer; Reorganizes MoE code and introduces shared topk.py routing layer.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.ep.layer, L1.triton.fused_moe_patch
FILES: python/sglang/srt/layers/fused_moe_patch.py (+0/-133); python/sglang/srt/layers/moe/ep_moe/__init__.py (+0/-0); python/sglang/srt/layers/moe/ep_moe/kernels.py (+0/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+14/-39); python/sglang/srt/layers/moe/fused_moe_native.py (+46/-0); python/sglang/srt/layers/moe/fused_moe_triton/__init__.py (+3/-7); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=3072,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=3072,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=3072,device_name=NVIDIA_H100_80GB_HBM3.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=3584,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=7168,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=1344,device_name=NVIDIA_A100-SXM4-40GB.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=1344,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=14336,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=14336,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=1792,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=1792,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=2688,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=2688,device_name=NVIDIA_H100_80GB_HBM3.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=3072,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=3072,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=3200,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=3584,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=6400,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=7168,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=16,N=800,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=1280,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py (+3/-1); benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py (+3/-1); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+2/-2); (+48 more)
BODY: ## Motivation ⏎  ⏎ Reorg moe code and reuse common part. ⏎  ⏎ ## Checklist

### L1-53aed988cb  (L1, 2024-12-26, sha 53aed988cbaa, PR #2575)
TITLE: Refactor MoE (#2575)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.triton.fused_moe; Adds block-wise FP8 support in fused_moe_triton refactor.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+78/-8); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-1); python/sglang/srt/configs/model_config.py (+4/-1); python/sglang/srt/layers/linear.py (+20/-2); python/sglang/srt/layers/quantization/fp8.py (+159/-25); python/sglang/srt/layers/quantization/fp8_kernel.py (+278/-0); python/sglang/srt/layers/quantization/fp8_utils.py (+90/-1); python/sglang/srt/models/deepseek_v2.py (+36/-11); python/sglang/test/test_block_fp8.py (+341/-0)
BODY: Support w8a8 fp8 block-wise quantization.

### L1-31548116a8  (L1, 2024-12-26, sha 31548116a8dc, PR #2579)
TITLE: fix moe_align_block_size_kernel for shared memory issue (#2579)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: introduce; artifacts=L1.align.cuda_aot; Adds sgl-kernel CUDA moe_align_block_size implementation.
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/setup.py (+20/-0); sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+151/-0); sgl-kernel/src/sgl-kernel/__init__.py (+8/-1); sgl-kernel/src/sgl-kernel/ops/__init__.py (+19/-0); sgl-kernel/tests/test_moe_align.py (+26/-0)
BODY: ## Motivation ⏎  ⏎ thanks @fengyang95 for the solution ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-60e2fdcf4f  (L1, 2024-12-26, sha 60e2fdcf4fdb, PR #2581)
TITLE: use sgl-kernel moe_align_block_size (#2581)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
STAGE1: change_default; artifacts=L1.align.cuda_aot,L1.triton.fused_moe; Switches fused_moe_triton to use sgl-kernel moe_align_block_size.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+20/-3); python/sglang/srt/model_executor/model_runner.py (+6/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2dccecf432  (L1, 2024-12-26, sha 2dccecf43261, PR #2590)
TITLE: fix: only enable moe_align_block_size for now (#2590)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: adapt_framework; artifacts=L1.align.cuda_aot; Restricts sgl-kernel exports to moe_align_block_size for initial use.
ARTIFACT_HINTS: -
FILES: sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/src/sgl-kernel/__init__.py (+1/-11); sgl-kernel/src/sgl-kernel/ops/__init__.py (+0/-20)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-7722c11c1d  (L1, 2024-12-26, sha 7722c11c1d2a, PR #2606)
TITLE: Regression fix to AMD/ROCm from recent change (#2606)
SOURCES: path_core, symbol_pickaxe
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; Fixes AMD/ROCm regression in fused_moe_triton after align changes.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+8/-3)
BODY: ## Motivation ⏎  ⏎  ⏎ As it is. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Update documentation as needed, including docstrings or example tutorials.

### L1-77d1210b36  (L1, 2024-12-27, sha 77d1210b3610, PR #2615)
TITLE: fix moe_align_block_size (#2615)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_correctness; artifacts=L1.align.cuda_aot; Fixes CUDA moe_align_block_size kernel implementation.
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+4/-16); sgl-kernel/src/sgl-kernel/ops/__init__.py (+4/-0); sgl-kernel/tests/test_moe_align.py (+15/-1)
BODY: 

### L1-6e5305158c  (L1, 2024-12-28, sha 6e5305158cde, PR #2617)
TITLE: update sgl_moe_align_block_size usage (#2617)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes
STAGE1: adapt_framework; artifacts=L1.align.cuda_aot,L1.triton.fused_moe; Updates fused_moe_triton usage of sgl_moe_align_block_size.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+10/-2); python/sglang/srt/model_executor/model_runner.py (+0/-6)
BODY: 

### L1-9254a33ad4  (L1, 2024-12-28, sha 9254a33ad46d, PR #2624)
TITLE: avoid fused_moe_triton `padding` circular import (#2624)
SOURCES: path_integration+keyword, subject_keyword, release_notes
STAGE1: repair_build_dependency; artifacts=L1.triton.fused_moe; Fixes fused_moe_triton padding circular import.
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+4/-1)
BODY: avoid fused_moe_triton `padding` circular import.

### L1-7863e4368a  (L1, 2024-12-28, sha 7863e4368abf, PR #2628)
TITLE: add configs for block fp8 related kernels (#2628)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.triton.fused_moe; Adds block FP8 fused_moe support with block_shape configs.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+47/-26); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+51/-8); python/sglang/srt/layers/quantization/configs/N=1536,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=1536,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=2048,K=512,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=2048,K=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=2304,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=2304,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=24576,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=24576,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=3072,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=3072,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=32768,K=512,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=32768,K=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=36864,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=36864,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=4096,K=512,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=4096,K=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=4608,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=4608,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=576,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=576,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=1024,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=1024,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=1152,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=1152,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=16384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=16384,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=18432,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=18432,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=2048,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=2304,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=2304,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/fp8_kernel.py (+69/-16)
BODY: ## Motivation ⏎  ⏎ tune by @HandH1998 I help verify. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b02da24a5b  (L1, 2024-12-30, sha b02da24a5b8c, PR #2642)
TITLE: Refactor sgl-kernel build (#2642)
SOURCES: path_core, dependency_pin
STAGE1: repair_build_dependency; artifacts=L1.align.cuda_aot; Moves moe_align binding into unified sgl-kernel build target.
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/CMakeLists.txt (+6/-23); sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+2/-6); sgl-kernel/setup.py (+34/-67); sgl-kernel/src/sgl-kernel/__init__.py (+11/-1); sgl-kernel/src/sgl-kernel/csrc/sgl_kernel_ops.cu (+32/-0); sgl-kernel/src/sgl-kernel/csrc/trt_reduce.cc (+0/-13); sgl-kernel/src/sgl-kernel/csrc/warp_reduce.cc (+0/-14); sgl-kernel/src/sgl-kernel/csrc/warp_reduce_kernel.cu (+2/-1); sgl-kernel/src/sgl-kernel/ops/__init__.py (+21/-1)
BODY: ## Motivation ⏎  ⏎ Involve all ops into one target. Reduce redundant build code.

### L1-148254d4db  (L1, 2025-01-02, sha 148254d4db8b, PR #2705)
TITLE: Improve moe reduce sum kernel performance (#2705)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: optimize; artifacts=L1.triton.fused_moe; Replaces torch.sum combine with moe_sum for fused_moe performance.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+11/-5); docker/Dockerfile.rocm (+1/-1)
BODY: ## Motivation ⏎  ⏎ torch.sum could not use GPU core efficiency, implement specific kernel to enhance the performance ⏎  ⏎ ## Modifications ⏎  ⏎ change the base docker image and modify torch.sum to ops.moe_sum in fused_moe.py ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Update documentation as needed, including docstrings or example tutorials.

### L1-bdf946bf81  (L1, 2025-01-02, sha bdf946bf8101, PR #2716)
TITLE: Support loading pre-sharded moe weights (#2716)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.triton.fused_moe; Adds pre-sharded MoE weight loading support in fused_moe layer.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+14/-6); python/sglang/srt/models/grok.py (+97/-26)
BODY: 

### L1-c7ae474a49  (L1, 2025-01-02, sha c7ae474a49f9, PR #2601)
TITLE: [Feature, Hardware] Enable DeepseekV3 on AMD GPUs (#2601)
SOURCES: path_core
STAGE1: extend_support; artifacts=L1.triton.fused_moe; Adds AMD DeepSeek-V3 FP8 handling in fused_moe_triton.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+5/-5); python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+4/-0)
LABELS: bug, high priority, amd
PERF_LINES: - Support DeepseekV3 on AMD Instinct MI300X GPU | Prefill. latency: 6.46569 s, throughput:    633.50 token/s | Decode.  latency: 2.58990 s, throughput:     12.36 token/s | Decode.  latency: 0.07421 s, throughput:    431.21 token/s | Decode.  latency: 0.07358 s, throughput:    434.90 token/s | Decode.  latency: 0.07341 s, throughput:    435.91 token/s | Decode.  latency: 0.07385 s, throughput:    4
BODY: ## Motivation ⏎ - Support DeepseekV3 on AMD Instinct MI300X GPU ⏎  ⏎ ## Modifications ⏎ - Add proper fix for AMD FP8 ```e4m3fnuz``` to support DeepseekV3 FP8 model ⏎ - Bypass ```FlashInfer backend bmm_fp8``` to cast FP8 to BF16 in MLA ⏎ - Add AMD ```triton stages``` config ⏎ ### TODO ⏎  ⏎  ⏎  ⏎ ## How to run ⏎ **build env** ⏎ ``` ⏎ cd sglang/docker ⏎  ⏎ docker build –t sglang-rocm:latest –f Dockerfile.rocm . ⏎   ⏎ docker run -it --ipc=host \  ⏎                --cap-add=SYS_PTRACE \ ⏎                --network=host \  ⏎                --device=/dev/kfd --device=/dev/dri \ ⏎                --security-opt seccomp=unconfined \  ⏎                --group-add video \ ⏎                --privileged \ ⏎                -w /workspace sglang-rocm:latest  ⏎ ``` ⏎  ⏎ **offline:**   ⏎ ``` ⏎ python -m sglang.bench_one_batch --batch-size 32 --input 128 --output 32 --model /data/DeepSeek-V3-Base/ --tp 8 --trust-remote-code ⏎  ⏎ Warmup ... ⏎ Prefill. latency: 6.46569 s, throughput:    633.50 token/s ⏎ Decode.  latency: 2.58990 s, throughput:     12.36 token/s ⏎ Decode.  latency: 0.07421 s, throughput:    431.21 token/s ⏎ Decode.  latency: 0.07358 s, throughput:    434.90 token/s ⏎ Decode.  latency: 0.07341 s, throughput:    435.91 token/s ⏎ Decode.  latency: 0.07385 s, throughput:    433.30 token/s ⏎ Decode.  median latency: 0.07383 s, median throughput:    433.44 token/s ⏎ Total. latency:  9.498 s, throughput:    458.19 token/s ⏎ Benchmark ... ⏎ Prefill. latency: 0.54745 s, throughput:   7482.01 token/s ⏎ Decode.  latency: 0.07250 s, throughput:    441.41 token/s ⏎ Decode.  latency: 0.07399 s, throughput:    432.46 token/s ⏎ Decode.  latency: 0.07309 s, throughput:    437.84 token/s ⏎ Decode.  latency: 0.07335 s, throughput:    436.27 token/s ⏎ Decode.  latency: 0.07333 s, throughput:    436.38 token/s ⏎ Decode.  median latency: 0.07358 s, median throughput:    434.88 token/s ⏎ Total. latency:  2.828 s, throughput:   1810.38 token/s ⏎ ``` ⏎  ⏎ **server:** ⏎ ``` ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3-Base --tp 8 --trust-remote-code ⏎  ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 ⏎  ⏎ Accuracy: 0.950 ⏎ Invalid: 0.000 ⏎ ``` ⏎ ## Issues ⏎ - If you get the error like `raise OutOfResources(self.metadata.shared, max_shared, "shared memory")`, same with https://github.com/sgl-project/sglang/issues/2384 ⏎ **Solved with** ```python/sglang/srt/layers/attention/triton_ops/decode_attention.py +410``` ⏎ - If you get an error like `ImportError: cannot import name 'build_regex_from_schema' from 'outlines.fsm.json_schema'`, same with https://github.com/sgl-project/sglang/issues/25 …[truncated]

### L1-ba5112ff69  (L1, 2025-01-02, sha ba5112ff691d, PR #2712)
TITLE: feat: support moe_align_block_size_triton (#2712)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
STAGE1: introduce; artifacts=L1.triton.moe_align; Adds optional Triton moe_align_block_size implementation.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+177/-25)
PERF_LINES: batch size 1/8/32, input/output 128/256, **around 20% improvement for online cases** | Prefill. latency: 2.03397 s, throughput:     62.93 token/s | Decode.  latency: 2.15392 s, throughput:      0.46 token/s | Decode.  latency: 0.02824 s, throughput:     35.41 token/s | Decode.  latency: 0.02772 s, throughput:     36.07 token/s | Decode.  latency: 0.02776 s, throughput:     36.02 token/s | Decode. 
BODY: ## Motivation ⏎  ⏎ ``` ⏎ # enable Triton implementation ⏎ export ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON=1 ⏎ ``` ⏎  ⏎ batch size 1/8/32, input/output 128/256, **around 20% improvement for online cases** ⏎  ⏎ TODO @BBuf will continue to optimize the CUDA version. ⏎  ⏎ ``` ⏎ Prefill. latency: 2.03397 s, throughput:     62.93 token/s ⏎ Decode.  latency: 2.15392 s, throughput:      0.46 token/s ⏎ Decode.  latency: 0.02824 s, throughput:     35.41 token/s ⏎ Decode.  latency: 0.02772 s, throughput:     36.07 token/s ⏎ Decode.  latency: 0.02776 s, throughput:     36.02 token/s ⏎ Decode.  latency: 0.02775 s, throughput:     36.03 token/s ⏎ Decode.  median latency: 0.02776 s, median throughput:     36.02 token/s ⏎ Total. latency:  4.355 s, throughput:     31.23 token/s ⏎ Benchmark ... ⏎ Prefill. latency: 0.14870 s, throughput:    860.77 token/s ⏎ Decode.  latency: 0.02796 s, throughput:     35.76 token/s ⏎ Decode.  latency: 0.02771 s, throughput:     36.09 token/s ⏎ Decode.  latency: 0.02771 s, throughput:     36.09 token/s ⏎ Decode.  latency: 0.02771 s, throughput:     36.09 token/s ⏎ Decode.  latency: 0.02772 s, throughput:     36.07 token/s ⏎ Decode.  median latency: 0.02796 s, median throughput:     35.76 token/s ⏎ Total. latency:  7.268 s, throughput:     52.84 token/s ⏎  ⏎ Prefill. latency: 5.71712 s, throughput:    179.11 token/s ⏎ Decode.  latency: 2.32011 s, throughput:      3.45 token/s ⏎ Decode.  latency: 0.03317 s, throughput:    241.20 token/s ⏎ Decode.  latency: 0.03296 s, throughput:    242.74 token/s ⏎ Decode.  latency: 0.03353 s, throughput:    238.57 token/s ⏎ Decode.  latency: 0.03384 s, throughput:    236.42 token/s ⏎ Decode.  median latency: 0.03384 s, median throughput:    236.42 token/s ⏎ Total. latency:  8.239 s, throughput:    132.06 token/s ⏎ Benchmark ... ⏎ Prefill. latency: 0.18112 s, throughput:   5653.82 token/s ⏎ Decode.  latency: 0.03253 s, throughput:    245.91 token/s ⏎ Decode.  latency: 0.03272 s, throughput:    244.47 token/s ⏎ Decode.  latency: 0.03296 s, throughput:    242.74 token/s ⏎ Decode.  latency: 0.03349 s, throughput:    238.89 token/s ⏎ Decode.  latency: 0.03379 s, throughput:    236.73 token/s ⏎ Decode.  median latency: 0.03415 s, median throughput:    234.25 token/s ⏎ Total. latency:  8.888 s, throughput:    345.62 token/s ⏎  ⏎ Prefill. latency: 7.90844 s, throughput:    517.93 token/s ⏎ Decode.  latency: 1.95407 s, throughput:     16.38 token/s ⏎ Decode.  latency: 0.03557 s, throughput:    899.60 token/s ⏎ Decode.  latency: 0.03624 s, throughput:    882.95 token/s ⏎ Decode.  latency: 0.03713 s, throughput:    861.73 token/s ⏎ Decode.  latency: 0.03770 s, throughput:    8 …[truncated]

### L1-ded9fcd09a  (L1, 2025-01-06, sha ded9fcd09a43, PR #2735)
TITLE: improve moe_align_kernel for deepseek v3 (#2735)
SOURCES: path_core, subject_keyword, release_notes
STAGE1: optimize; artifacts=L1.align.cuda_aot; Optimizes CUDA moe_align_kernel for DeepSeek-V3.
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+32/-50); sgl-kernel/tests/test_trt_reduce.py (+0/-1)
PERF_LINES: Latency: 83.593 s | Output throughput: 1676.993 token/s | Latency: 78.651 s | Output throughput: 1782.633 token/s
BODY: In H200 ⏎  ⏎ ```shell ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code ⏎  ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 ⏎ ``` ⏎  ⏎ main branch: ⏎  ⏎ ```shell ⏎ Accuracy: 0.951 ⏎ Invalid: 0.000 ⏎ Latency: 83.593 s ⏎ Output throughput: 1676.993 token/s ⏎ ``` ⏎  ⏎ pr: ⏎  ⏎ ```shell ⏎ 0.952 ⏎ Invalid: 0.000 ⏎ Latency: 78.651 s ⏎ Output throughput: 1782.633 token/s ⏎ ``` ⏎  ⏎ I gain idea mainly from @zhyncs [issue](https://github.com/sgl-project/sglang/issues/2732#issuecomment-2571513609) ⏎  ⏎ And maybe we can get a better optimized version recently.

### L1-bdc1acf6cd  (L1, 2025-01-07, sha bdc1acf6cdad, PR #2761)
TITLE: Misc fix for min_p_sampling, --cuda-graph-bs (#2761)
SOURCES: path_core, dependency_pin
STAGE1: optimize; artifacts=L1.triton.fused_moe; Special-cases top-k combine for k=1 and k=2 instead of torch.sum.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+9/-3); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+16/-5); .github/workflows/pr-test.yml (+3/-1); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+1/-0); python/sglang/bench_serving.py (+4/-1); python/sglang/srt/layers/logits_processor.py (+5/-0); python/sglang/srt/layers/quantization/__init__.py (+1/-2); python/sglang/srt/managers/data_parallel_controller.py (+2/-0); python/sglang/srt/managers/scheduler.py (+3/-2); python/sglang/srt/metrics/collector.py (+22/-30); python/sglang/srt/model_executor/cuda_graph_runner.py (+8/-1); python/sglang/srt/model_executor/model_runner.py (+8/-1); python/sglang/srt/sampling/sampling_batch_info.py (+1/-0); python/sglang/srt/server.py (+4/-6); python/sglang/srt/server_args.py (+7/-0); python/sglang/srt/utils.py (+6/-5); python/sglang/test/test_utils.py (+35/-6)
BODY: - Support `--cuda-graph-bs` so you can specify the batch size to capture the cuda graph. ⏎ - Fix merge_batch for min_p_sampling ⏎ - Add more utility functions and remove redundant code.

### L1-8a6906127a  (L1, 2025-01-07, sha 8a6906127a81, PR #2784)
TITLE: Improve linear.py to load sharded weights & remove the dependency of Parameters from vllm (#2784)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L1.triton.fused_moe; Updates fused_moe layer after Parameter dependency removal.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-3); 3rdparty/amd/tuning/benchmark_moe_rocm.py (+4/-1); python/sglang/srt/layers/attention/__init__.py (+8/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+4/-2); python/sglang/srt/layers/linear.py (+165/-57); python/sglang/srt/layers/parameter.py (+431/-0); python/sglang/srt/layers/quantization/fp8.py (+1/-1); python/sglang/srt/layers/vocab_parallel_embedding.py (+1/-1); python/sglang/srt/managers/session_controller.py (+1/-1); python/sglang/srt/model_executor/forward_batch_info.py (+3/-0); python/sglang/srt/model_executor/model_runner.py (+1/-2); python/sglang/srt/models/grok.py (+25/-16); python/sglang/srt/server.py (+7/-2); python/sglang/srt/speculative/eagle_utils.py (+1/-1); scripts/killall_sglang.sh (+1/-0)
BODY: - remove the dependency of Parameters from vllm ⏎ - improve the weight loading of linear.py

### L1-72c7776355  (L1, 2025-01-13, sha 72c777635593, PR #2851)
TITLE: Fix linear.py and improve weight loading (#2851)
SOURCES: path_core
STAGE1: repair_correctness; artifacts=L1.routing.topk_py; Fixes topk torch_native path when custom routing function is present.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+4/-2); benchmark/deepseek_v3/README.md (+4/-3); docs/references/supported_models.md (+1/-1); python/sglang/srt/layers/linear.py (+36/-98); python/sglang/srt/layers/parameter.py (+24/-16); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-1); python/sglang/srt/layers/vocab_parallel_embedding.py (+15/-2); python/sglang/srt/managers/scheduler.py (+4/-0); python/sglang/srt/mem_cache/memory_pool.py (+19/-0); python/sglang/srt/server.py (+3/-0); test/srt/test_moe_eval_accuracy_large.py (+1/-1)
BODY: 

### L1-e808c1df3e  (L1, 2025-01-13, sha e808c1df3e04, PR #2854)
TITLE: Integrate ROCm ater package for ck moe function feasibility (#2854)
SOURCES: path_core, symbol_pickaxe
STAGE1: integrate; artifacts=L1.triton.fused_moe; Integrates ROCm CK/AITER MoE feasibility path in fused_moe layer.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+36/-9); docker/Dockerfile.rocm (+9/-0); python/sglang/srt/layers/quantization/fp8.py (+98/-45); python/sglang/srt/utils.py (+19/-0)
BODY: ## Motivation ⏎  ⏎ ROCm platform has moe function implementation from ck module, it fused two gemm and activation kernel into one to decrease the launched overhead. ⏎  ⏎ ## Modifications ⏎  ⏎ Major modification in the layer.py of fused_moe ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contribution_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contribution_guide.md). ⏎ - [+] Update documentation as needed, including docstrings or example tutorials.

### L1-63051738a9  (L1, 2025-01-16, sha 63051738a91e, PR #2806)
TITLE: Enable CPU device on SGLang (#2806)
SOURCES: path_core, symbol_pickaxe
STAGE1: extend_support; artifacts=L1.hardware.cpu_npu_musa,L1.triton.fused_moe; Adds CPU device fallback for MoE through native fused_moe path.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/sglang/srt/layers/moe/fused_moe_native.py (+69/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+4/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+26/-2); python/pyproject.toml (+6/-0); python/sglang/srt/configs/device_config.py (+1/-1); python/sglang/srt/layers/rotary_embedding.py (+248/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/managers/scheduler.py (+2/-0); python/sglang/srt/managers/tp_worker_overlap_thread.py (+2/-0); python/sglang/srt/model_executor/model_runner.py (+9/-3); python/sglang/srt/models/deepseek_v2.py (+3/-1); python/sglang/srt/server_args.py (+1/-1); python/sglang/srt/utils.py (+4/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR enables CPU device on SGLang. ⏎ Currently we fallback attention and MoE to the torch native backend and make the functionality work on CPU. ⏎ We will submit follow-up PRs to provide optimized kernels to further improvement the performance. ⏎  ⏎ For vllm installation for CPU, users could follow the instruction provided by vllm [here](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html). ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ The main modifications include: ⏎ - Add a native implementation for MoE (`moe_forward_native`) following the original implementation in the model: [moe_infer in deepseek](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/e0828e3cc0a03408724b80c3cc92c8e072db8d01/modeling_deepseek.py#L589). This performs better than the existing [fused_moe_forward_native](https://github.com/sgl-project/sglang/blob/b5fb4ef58a6bbe6c105d533b69e8e8bc2bf4fc3c/python/sglang/srt/layers/moe/fused_moe_native.py#L39-L46) on CPU. ⏎ - For CPU, we won't call the code to set the number of threads to 1 anymore: [link to the current code in SGLang](https://github.com/sgl-project/sglang/blob/46d44318894a13dc6d018892b32dd4a7e09f20f7/python/sglang/srt/model_executor/model_runner.py#L260), otherwise, only 1 CPU core will be used when running the workload. This change will improve the performance on CPU. ⏎ - For the rotary embedding part, in the `DeepseekScalingRotaryEmbedding` class defined in vllm, the device has been hard-coded to `"cuda"` in these two places: [_compute_inv_freq](https://github.com/vllm-project/vllm/blob/8a1f938e6f02052df0f4953c149410605a2d56d8/vllm/model_executor/layers/rotary_embedding.py#L646), [_compute_cos_sin_cache](https://github.com/vllm-project/vllm/blob/8a1f938e6f02052df0f4953c149410605a2d56d8/vllm/model_executor/layers/rotary_embedding.py#L665). We temporarily port the related code into SGLang to make it compatible with the CPU version. We will add an optimized rotary embedding kernel for CPU and will remove the ported code then. ⏎  ⏎ ## Example ⏎ Below are some example command lines to use on CPU with this PR. We only support `--disable-mla` for now. ⏎ Supposing we want to use 40 CPU cores on the NUMA node 0: ⏎  ⏎ ### Bench one batch ⏎ ``` ⏎ numactl --physcpubind=0-39 --membind=0 python3 -m sglang.bench_one_batch --batch-size 1 --input 1024 --output 8 --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --trust-remote-code --device cpu --attention-backend torch_native --disable-mla ⏎ ``` ⏎ ###  Server mode ⏎ Command line on server side: ⏎ ```sh ⏎ numactl --physcpubind=0-39 --membind=0 python3 -m sglang.launch_server --model …[truncated]

### L1-5dc54f1a62  (L1, 2025-01-17, sha 5dc54f1a627a, PR #2907)
TITLE: feat: remove vllm distributed (#2907)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L1.triton.fused_moe,L1.ep.layer; Updates MoE layers after removing vLLM distributed dependency.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-4); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-3); python/sglang/srt/layers/activation.py (+3/-3); python/sglang/srt/layers/dp_attention.py (+2/-1); python/sglang/srt/layers/linear.py (+2/-2); python/sglang/srt/layers/logits_processor.py (+2/-2); python/sglang/srt/layers/parameter.py (+2/-1); python/sglang/srt/layers/quantization/fp8.py (+1/-1); python/sglang/srt/layers/vocab_parallel_embedding.py (+2/-2); python/sglang/srt/model_executor/cuda_graph_runner.py (+2/-2); python/sglang/srt/model_executor/model_runner.py (+9/-5); python/sglang/srt/model_loader/loader.py (+8/-6); python/sglang/srt/model_loader/weight_utils.py (+1/-1); python/sglang/srt/models/baichuan.py (+4/-4); python/sglang/srt/models/chatglm.py (+1/-1); python/sglang/srt/models/commandr.py (+3/-3); python/sglang/srt/models/dbrx.py (+4/-4); python/sglang/srt/models/deepseek.py (+3/-3); python/sglang/srt/models/deepseek_v2.py (+3/-3); python/sglang/srt/models/exaone.py (+1/-1); python/sglang/srt/models/gemma.py (+1/-1); python/sglang/srt/models/gemma2.py (+1/-1); python/sglang/srt/models/gpt2.py (+2/-1); python/sglang/srt/models/gpt_bigcode.py (+1/-1); python/sglang/srt/models/granite.py (+1/-1); python/sglang/srt/models/grok.py (+3/-3); python/sglang/srt/models/internlm2.py (+1/-1); python/sglang/srt/models/llama.py (+4/-4); python/sglang/srt/models/minicpm.py (+1/-1); python/sglang/srt/models/minicpm3.py (+1/-1); python/sglang/srt/models/mixtral.py (+3/-3); python/sglang/srt/models/mixtral_quant.py (+3/-3); python/sglang/srt/models/mllama.py (+2/-2); python/sglang/srt/models/olmo.py (+1/-1); python/sglang/srt/models/olmo2.py (+4/-4); python/sglang/srt/models/olmoe.py (+4/-4); python/sglang/srt/models/phi3_small.py (+1/-1); python/sglang/srt/models/qwen.py (+1/-1); python/sglang/srt/models/qwen2.py (+1/-1); python/sglang/srt/models/qwen2_moe.py (+3/-3); (+5 more)
PERF_LINES: Benchmark offline throughput (w/ FP8) | Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s] | Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
BODY: ## Motivation ⏎  ⏎ ref https://github.com/sgl-project/sglang/actions/runs/12797036847/job/35678064629 ⏎  ⏎ cc @yizhang2077  ⏎  ⏎ ``` ⏎ Benchmark offline throughput (w/ FP8) ⏎ Run cd test/srt ⏎ [2025-0[1](https://github.com/sgl-project/sglang/actions/runs/12797036847/job/35678064160#step:6:1)-15 21:16:58] server_args=ServerArgs(model_path='neuralmagic/Meta-Llama-3.1-8B-FP8', tokenizer_path='neuralmagic/Meta-Llama-3.1-8B-FP8', tokenizer_mode='auto', load_format='auto', trust_remote_code=False, dtype='auto', kv_cache_dtype='auto', quantization_param_path=None, quantization=None, context_length=None, device='cuda', served_model_name='neuralmagic/Meta-Llama-3.1-8B-FP8', chat_template=None, is_embedding=False, revision=None, skip_tokenizer_init=False, host='127.0.0.1', port=61[5](https://github.com/sgl-project/sglang/actions/runs/12797036847/job/35678064160#step:6:6)7, mem_fraction_static=0.88, max_running_requests=None, max_total_tokens=None, chunked_prefill_size=8192, max_prefill_tokens=16384, schedule_policy='lpm', schedule_conservativeness=1.0, cpu_offload_gb=0, prefill_only_one_req=False, tp_size=1, stream_interval=1, random_seed=447459774, constrained_json_whitespace_pattern=None, watchdog_timeout=300, download_dir=None, base_gpu_id=0, log_level='info', log_level_http=None, log_requests=False, show_time_cost=False, enable_metrics=False, decode_log_interval=40, api_key=None, fi ⏎ [2025-01-15 21:17:11 TP0] Init torch distributed begin. ⏎ [2025-01-15 21:17:11 TP0] Load weight begin. avail mem=78.81 GB ⏎ [2025-01-15 21:17:12 TP0] Using model weights format ['*.safetensors'] ⏎ Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s] ⏎ [2025-01-15 21:17:12 TP0] Scheduler hit an exception: Traceback (most recent call last): ⏎   File "/public_sglang_ci/runner-b-gpu-67/_work/sglang/sglang/python/sglang/srt/managers/scheduler.py", line 1641, in run_scheduler_process ⏎     scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank, dp_rank) ⏎   File "/public_sglang_ci/runner-b-gpu-[6](https://github.com/sgl-project/sglang/actions/runs/12797036847/job/35678064160#step:6:7)7/_work/sglang/sglang/python/sglang/srt/managers/scheduler.py", line 209, in __init__ ⏎     self.tp_worker = TpWorkerClass( ⏎   File "/public_sglang_ci/runner-b-gpu-6[7](https://github.com/sgl-project/sglang/actions/runs/12797036847/job/35678064160#step:6:8)/_work/sglang/sglang/python/sglang/srt/managers/tp_worker_overlap_thread.py", line 63, in __init__ ⏎     self.worker = TpModelWorker(server_args, gpu_id, tp_rank, dp_rank, nccl_port) ⏎   File "/public_sglang_ci/runner-b-gpu-67/_work/sglang/sgl …[truncated]

### L1-6ada05d0ed  (L1, 2025-01-19, sha 6ada05d0ed52, PR #2984)
TITLE: feat: check for is_cuda for sgl_kernel import (#2984)
SOURCES: path_core, symbol_pickaxe
STAGE1: repair_correctness; artifacts=L1.triton.fused_moe; Guards sgl_kernel import by CUDA availability for fused_moe_triton.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+10/-10)
BODY: ## Motivation ⏎  ⏎ cc @HaiShaw @chunyuan-w  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-52c03f16b9  (L1, 2025-01-27, sha 52c03f16b914, PR #3170)
TITLE: Add activation parameters to fused_moe (#3170)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
STAGE1: extend_support; artifacts=L1.triton.fused_moe,L1.ep.layer; Adds activation parameter so fused_moe can use GELU for Grok.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-0); python/sglang/srt/layers/moe/fused_moe_native.py (+17/-3); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+18/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+9/-0); python/sglang/srt/layers/quantization/fp8.py (+4/-1); python/sglang/srt/models/grok.py (+1/-0); test/srt/test_fp8_kernel.py (+0/-2)
BODY: This is because grok uses gelu while other models use silu.

### L1-53cef81587  (L1, 2025-01-27, sha 53cef81587de, PR #3174)
TITLE: Improve weight loading and code style (#3174)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L1.ep.layer; Updates EP MoE layer for revised weight loading and scheduler args.
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+17/-12); python/sglang/srt/layers/linear.py (+24/-9); python/sglang/srt/layers/parameter.py (+16/-7); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/managers/scheduler.py (+11/-5); python/sglang/srt/model_executor/model_runner.py (+1/-1); python/sglang/srt/model_loader/weight_utils.py (+14/-4); python/sglang/srt/server_args.py (+6/-0); python/sglang/srt/utils.py (+1/-1); python/sglang/test/test_utils.py (+76/-22); sgl-kernel/setup.py (+2/-2)
BODY: 

### L1-1ebe1d6de5  (L1, 2025-02-01, sha 1ebe1d6de5e0, PR #3236)
TITLE: Optimize MoE topk with torch compile (#3236)
SOURCES: path_core
STAGE1: optimize; artifacts=L1.routing.topk_py; Applies torch.compile to MoE topk routing by default.
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+4/-0)
PERF_LINES: Prefill. latency: 1.82421 s, throughput:     70.17 token/s | Decode.  latency: 1.75760 s, throughput:      0.57 token/s | Decode.  latency: 0.02740 s, throughput:     36.49 token/s | Decode.  latency: 0.02703 s, throughput:     37.00 token/s | Decode.  latency: 0.02711 s, throughput:     36.88 token/s | Decode.  latency: 0.02711 s, throughput:     36.89 token/s | Decode.  median latency: 0.02711 s
BODY: ## Motivation ⏎  ⏎ 2 token/s faster by applying torch compile for topk function by default. ⏎  ⏎ ``` ⏎ python3 -m sglang.bench_one_batch --batch-size 1 --input 128 --output 256 --model deepseek-ai/DeepSeek-V3 --trust-remote-code --tp 8 ⏎  ⏎ main branch: ⏎ Prefill. latency: 1.82421 s, throughput:     70.17 token/s ⏎ Decode.  latency: 1.75760 s, throughput:      0.57 token/s ⏎ Decode.  latency: 0.02740 s, throughput:     36.49 token/s ⏎ Decode.  latency: 0.02703 s, throughput:     37.00 token/s ⏎ Decode.  latency: 0.02711 s, throughput:     36.88 token/s ⏎ Decode.  latency: 0.02711 s, throughput:     36.89 token/s ⏎ Decode.  median latency: 0.02711 s, median throughput:     36.88 token/s ⏎ Total. latency:  3.745 s, throughput:     36.32 token/s ⏎ Benchmark ... ⏎ Prefill. latency: 0.16921 s, throughput:    756.48 token/s ⏎ Decode.  latency: 0.02716 s, throughput:     36.81 token/s ⏎ Decode.  latency: 0.02713 s, throughput:     36.86 token/s ⏎ Decode.  latency: 0.02713 s, throughput:     36.86 token/s ⏎ Decode.  latency: 0.02714 s, throughput:     36.85 token/s ⏎ Decode.  latency: 0.02716 s, throughput:     36.82 token/s ⏎ Decode.  median latency: 0.02719 s, median throughput:     36.78 token/s ⏎ Total. latency:  7.107 s, throughput:     54.03 token/s ⏎  ⏎ this pr: ⏎ Prefill. latency: 1.79881 s, throughput:     71.16 token/s ⏎ Decode.  latency: 0.93208 s, throughput:      1.07 token/s ⏎ Decode.  latency: 0.02643 s, throughput:     37.84 token/s ⏎ Decode.  latency: 0.02598 s, throughput:     38.49 token/s ⏎ Decode.  latency: 0.02605 s, throughput:     38.38 token/s ⏎ Decode.  latency: 0.02609 s, throughput:     38.33 token/s ⏎ Decode.  median latency: 0.02605 s, median throughput:     38.38 token/s ⏎ Total. latency:  2.887 s, throughput:     47.10 token/s ⏎ Benchmark ... ⏎ Prefill. latency: 0.17031 s, throughput:    751.59 token/s ⏎ Decode.  latency: 0.02589 s, throughput:     38.62 token/s ⏎ Decode.  latency: 0.02599 s, throughput:     38.47 token/s ⏎ Decode.  latency: 0.02640 s, throughput:     37.88 token/s ⏎ Decode.  latency: 0.02594 s, throughput:     38.56 token/s ⏎ Decode.  latency: 0.02594 s, throughput:     38.55 token/s ⏎ Decode.  median latency: 0.02613 s, median throughput:     38.26 token/s ⏎ Total. latency:  6.822 s, throughput:     56.29 token/s ⏎ ```

### L1-4eb4b401cc  (L1, 2025-02-01, sha 4eb4b401cc55, PR #3249)
TITLE: update and simplify CustomOp (#3249)
SOURCES: path_core
STAGE1: adapt_framework; artifacts=L1.triton.fused_moe,L1.ep.layer; Updates MoE layers to new simplified CustomOp protocol.
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-3); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-3); python/sglang/srt/custom_op.py (+40/-0); python/sglang/srt/layers/activation.py (+1/-5); python/sglang/srt/layers/custom_op_util.py (+0/-25); python/sglang/srt/layers/layernorm.py (+1/-5); python/sglang/srt/layers/rotary_embedding.py (+1/-3); python/sglang/srt/model_executor/cuda_graph_runner.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist
