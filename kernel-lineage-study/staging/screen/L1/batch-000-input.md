### L1-22085081bb  (L1, 2024-01-08, sha 22085081bb24, PR #)
TITLE: release initial code
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: .gitmodules; python/pyproject.toml
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L1-dafafe5b11  (L1, 2024-01-18, sha dafafe5b111d, PR #42)
TITLE: Use HTTP link in 3rdparty module (#42)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+1/-1)
BODY: Replace the SSH link of FlashInfer with HTTP link, so that users don't need to deploy the SSH key of their Github account for cloning, as deploying SSH key could be tedious in cloud instances or docker containers.

### L1-f6bfe3aaff  (L1, 2024-02-03, sha f6bfe3aaff6f, PR #134)
TITLE: Release 0.1.11 (#134)
SOURCES: release_notes
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: README.md (+4/-0); python/pyproject.toml (+1/-1); python/sglang/__init__.py (+1/-1)
BODY: 

### L1-26c3494152  (L1, 2024-02-06, sha 26c349415213, PR #156)
TITLE: [Submodule] Change FlashInfer to import (#156)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+0/-3); 3rdparty/flashinfer (+0/-1); docs/flashinfer.md (+5/-3); python/sglang/srt/layers/radix_attention.py (+0/-8); python/sglang/srt/managers/router/model_runner.py (+12/-9)
BODY: This PR removes submodule flashinfer from this repo. As flashinfer is now officially released, we can directly install it via pip. However, I didn't add it to pyproject.toml but just updated the README due to two reasons: ⏎ 1. flashinfer is not available on PYPI yet. ⏎ 2. It supports limited CUDA version and requires to manually select the wheel path. ⏎  ⏎ Meanwhile, the latest version of flashinfer has some interface changes. I modified them accordi …[truncated]

### L1-2af565b3bb  (L1, 2024-03-28, sha 2af565b3bb22, PR #337)
TITLE: [model] DBRX-instruct support (#337)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/lang/chat_template.py (+19/-0); python/sglang/srt/model_config.py (+7/-0); python/sglang/srt/models/dbrx.py (+416/-0); python/sglang/srt/models/dbrx_config.py (+281/-0); python/sglang/srt/models/stablelm.py (+3/-4)
BODY: 

### L1-33b242df30  (L1, 2024-05-11, sha 33b242df303e, PR #380)
TITLE: Compat with latest VLLM 0.4.2 main + fork.number rename + Flashinfer 0.0.4 (#380)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/lang/interpreter.py (+7/-7); python/sglang/lang/tracer.py (+6/-4); python/sglang/srt/layers/logits_processor.py (+1/-1); python/sglang/srt/managers/router/model_rpc.py (+5/-2); python/sglang/srt/managers/router/model_runner.py (+6/-14); python/sglang/srt/models/commandr.py (+19/-18); python/sglang/srt/models/dbrx.py (+20/-19); python/sglang/srt/models/gemma.py (+18/-17); python/sglang/srt/models/llama2.py (+19/-18); (+10 more)
BODY: Reason for PR: ⏎  ⏎ * Compat with VLLM main ⏎ * Rename `fork(number=N)` param to `fork(size=N)` for clarity.  ⏎ * Complete flashinfer 0.0.4 todo

### L1-2d580e7a89  (L1, 2024-05-12, sha 2d580e7a8991, PR #430)
TITLE: Fix flashinfer (#430)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/managers/router/model_rpc.py (+2/-1); python/sglang/srt/managers/router/model_runner.py (+3/-3)
BODY: 

### L1-0fafc5606b  (L1, 2024-05-21, sha 0fafc5606b0d, PR #460)
TITLE: port fp8 mixtral (#460)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/router/model_rpc.py (+1/-8); python/sglang/srt/managers/router/model_runner.py (+16/-12); python/sglang/srt/models/mixtral.py (+240/-101); python/sglang/srt/models/mixtral_quant.py (+371/-0); python/sglang/srt/server_args.py (+7/-0); python/sglang/srt/utils.py (+1/-0)
BODY: 

### L1-09de730dee  (L1, 2024-05-27, sha 09de730dee31, PR #484)
TITLE: Improve benchmark scripts & add more models (#484)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe.py (+485/-0); benchmark/latency_throughput/bench_throughput.py (+8/-6); benchmark/mmlu/bench_other.py (+1/-1); python/sglang/srt/models/grok.py (+669/-0); python/sglang/srt/utils.py (+4/-3); python/sglang/test/test_utils.py (+6/-1)
BODY: 

### L1-fb9296f0ed  (L1, 2024-06-12, sha fb9296f0ed07, PR #540)
TITLE: Higher priority for user input of max_prefill_tokens & format (#540)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe.py (+125/-119); benchmark/gsm8k/bench_other.py (+1/-1); benchmark/latency_throughput/bench_throughput.py (+24/-10); benchmark/mmlu/bench_other.py (+1/-1); python/sglang/__init__.py (+1/-1); python/sglang/backend/litellm.py (+2/-1); python/sglang/backend/openai.py (+26/-15); python/sglang/lang/interpreter.py (+1/-1); python/sglang/lang/ir.py (+2/-4); python/sglang/launch_server.py (+1/-1); (+40 more)
BODY: 

### L1-53a7ebd89a  (L1, 2024-06-17, sha 53a7ebd89a0b, PR #553)
TITLE: Update fused_moe (#553)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe.py (+220/-183)
BODY: 

### L1-9465b668b9  (L1, 2024-06-24, sha 9465b668b9d3, PR #561)
TITLE: Allow running with vllm==0.4.3 (#561)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe.py (+38/-3); python/sglang/srt/constrained/__init__.py (+11/-5)
BODY: There are some wired errors with fp8 and vllm==0.5.0.

### L1-2e6e62e156  (L1, 2024-06-26, sha 2e6e62e1562d, PR #567)
TITLE: Increase the number of thread limitation for tp worker managers. (#567)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe.py (+30/-11); benchmark/latency_throughput/bench_throughput.py (+9/-4); benchmark/latency_throughput/test_latency.py (+3/-2); benchmark/mmlu/bench_sglang.py (+47/-55); playground/load_tokenizer.py (+10/-5); python/sglang/srt/constrained/fsm_cache.py (+2/-1); python/sglang/srt/hf_transformers_utils.py (+43/-3); python/sglang/srt/managers/controller/manager_single.py (+3/-2); python/sglang/srt/managers/controller/tp_worker.py (+1/-1)
BODY: - Increase the number of thread limitation for tp worker managers. ⏎ - Improve MMLU benchmark scripts ⏎ - Tune block sizes for fp8 fused_moe kernels ⏎ - Support sentencepiece tokenizer

### L1-ac11388756  (L1, 2024-07-04, sha ac113887560c, PR #588)
TITLE: Add docker file (#588)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+6/-0); README.md (+3/-0)
BODY: 

### L1-dc1b8bcfaa  (L1, 2024-07-05, sha dc1b8bcfaac5, PR #593)
TITLE: Format (#593)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe.py (+181/-167); benchmark/latency_throughput/bench_one.py (+1/-1); benchmark/latency_throughput/bench_serving.py (+3/-2); benchmark/line_retrieval/gen_data.py (+3/-3); benchmark/mmlu/bench_sglang.py (+9/-5); python/sglang/bench_latency.py (+28/-12); python/sglang/global_config.py (+1/-0); python/sglang/lang/ir.py (+4/-2); python/sglang/srt/constrained/__init__.py (+3/-2); python/sglang/srt/hf_transformers_utils.py (+7/-3); (+11 more)
BODY: 

### L1-f6b29f6920  (L1, 2024-07-16, sha f6b29f692083, PR #629)
TITLE: Update docker file (#629)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+36/-5)
BODY: 

### L1-8832ecb1e4  (L1, 2024-07-16, sha 8832ecb1e451, PR #632)
TITLE: Reduce docker size (#632)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+8/-5); README.md (+10/-0)
BODY: Reduce the docker size and add the docker usage in readme

### L1-2d96da813e  (L1, 2024-07-19, sha 2d96da813e3a, PR #655)
TITLE: refactor model loader [unreachable code]: initial refactor (#655)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/linear.py (+869/-0); python/sglang/srt/layers/quantization/__init__.py (+49/-0); python/sglang/srt/layers/quantization/fp8.py (+662/-0); python/sglang/srt/model_loader/model_loader.py (+276/-0); python/sglang/srt/model_loader/utils.py (+260/-0)
BODY: 

### L1-eedc12e12e  (L1, 2024-07-21, sha eedc12e12ed3, PR #689)
TITLE: Support Deepseek MoE Model (#689)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/controller/cuda_graph_runner.py (+38/-19); python/sglang/srt/managers/controller/model_runner.py (+4/-3); python/sglang/srt/models/deepseek.py (+430/-0); python/sglang/srt/server.py (+1/-1); python/sglang/srt/utils.py (+46/-0)
BODY: 

### L1-679ebcbbdc  (L1, 2024-07-26, sha 679ebcbbdc37, PR #693)
TITLE: Deepseek v2 support (#693)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hf_transformers_utils.py (+1/-1); python/sglang/srt/managers/controller/model_runner.py (+8/-5); python/sglang/srt/model_config.py (+5/-0); python/sglang/srt/models/deepseek_v2.py (+517/-0); python/sglang/srt/server_args.py (+7/-0)
BODY: To use deepseek v2, please sepcify the `--context-length` or `--max-num-reqs` to avoid oom. The context length for deepseek is quite large, for the current static `req_to_token` layout, we cannot support large requests num and large context length at the same time.

### L1-dd7e8b9421  (L1, 2024-07-28, sha dd7e8b9421f3, PR #790)
TITLE: chore: add copyright for srt (#790)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe.py (+15/-0); python/sglang/srt/constrained/__init__.py (+15/-0); python/sglang/srt/constrained/base_cache.py (+15/-0); python/sglang/srt/constrained/fsm_cache.py (+15/-0); python/sglang/srt/constrained/jump_forward.py (+15/-0); python/sglang/srt/conversation.py (+15/-0); python/sglang/srt/flush_cache.py (+15/-0); python/sglang/srt/hf_transformers_utils.py (+15/-0); python/sglang/srt/layers/context_flashattention_nopad.py (+15/-0); python/sglang/srt/layers/extend_attention.py (+15/-0); (+51 more)
BODY: Thank you for your contribution, we really appreciate it. The following instructions will help improve your pull request and make it easier to receive feedback. If there are any items you don't understand, don't worry. Just submit the pull request and ask the maintainers for help. ⏎  ⏎ ## Motivation ⏎  ⏎ Please explain the motivation behind this PR and the goal you aim to achieve with it. ⏎  ⏎ ## Modification ⏎  ⏎ Briefly describe the changes made in thi …[truncated]

### L1-c31f084c71  (L1, 2024-08-07, sha c31f084c713c, PR #966)
TITLE: chore: update vllm to 0.5.4 (#966)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); .github/workflows/e2e-test.yml (+1/-2); .github/workflows/unit-test.yml (+1/-2); README.md (+2/-2); python/sglang/check_env.py (+1/-0); test/srt/models/test_causal_models.py (+1/-3); test/srt/run_suite.py (+1/-1); test/srt/test_chunked_prefill.py (+1/-1); test/srt/test_eval_accuracy.py (+1/-1); (+4 more)
BODY: Thank you for your contribution, we really appreciate it. The following instructions will help improve your pull request and make it easier to receive feedback. If there are any items you don't understand, don't worry. Just submit the pull request and ask the maintainers for help. ⏎  ⏎ ## Motivation ⏎  ⏎ Please explain the motivation behind this PR and the goal you aim to achieve with it. ⏎  ⏎ ## Modification ⏎  ⏎ Briefly describe the changes made in thi …[truncated]

### L1-fb1f28cbbb  (L1, 2024-08-12, sha fb1f28cbbbd3, PR #1047)
TITLE: Clean up the comments and names under python/sglang/srt/layers (#1047)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/activation.py (+2/-0); python/sglang/srt/layers/decode_attention.py (+9/-5); python/sglang/srt/layers/extend_attention.py (+6/-1); python/sglang/srt/layers/layernorm.py (+2/-0); python/sglang/srt/layers/linear.py (+0/-884); python/sglang/srt/layers/prefill_attention.py (+5/-0); python/sglang/srt/layers/quantization/__init__.py (+0/-64); python/sglang/srt/layers/quantization/fp8.py (+0/-677); python/sglang/srt/layers/radix_attention.py (+2/-2)
BODY: - Rename `token_attention.py` -> `decode_attention.py`, `context_flashattention_nopad.py` -> `prefill_attention.py` ⏎ - Delete the unused `linear.py` and `quantization`

### L1-cb99ba4fc6  (L1, 2024-08-12, sha cb99ba4fc619, PR #1033)
TITLE: feat: update Dockerfile (#1033)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+34/-19); .github/workflows/release-docker.yml (+25/-13)
BODY: ## Motivation ⏎  ⏎ Inspired by https://github.com/sgl-project/sglang/pull/999 so I co-authored with @vhain  ⏎  ⏎  ⏎ ## Modification ⏎  ⏎ 1. update Python 3.10 on Ubuntu 20.04 ⏎ 2. use GitHub Matrix to support SRT only (all build tasks are parallel) ⏎ 3. support cu118 ⏎  ⏎ ## Checklist ⏎  ⏎ 1. Ensure pre-commit `pre-commit run --all-files` or other linting tools are used to fix potential lint issues. ⏎ 4. Confirm that modifications are covered by complete unit t …[truncated]

### L1-ad3e4f1619  (L1, 2024-08-13, sha ad3e4f16199a, PR #1081)
TITLE: Update the mixtral to use the better FusedMoE layer (#1081)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/mixtral.py (+55/-253); python/sglang/srt/models/mixtral_quant.py (+0/-3); docs/en/model_support.md (+1/-1); test/srt/test_moe_serving_throughput.py (+1/-1)
BODY: 

### L1-a59636bb5e  (L1, 2024-08-14, sha a59636bb5e68, PR #1095)
TITLE: Update grok 1 model (#1095)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe/__init__.py (+1/-0); python/sglang/srt/layers/fused_moe/fused_moe.py (+165/-108); python/sglang/srt/layers/fused_moe/layer.py (+587/-0); benchmark/gsm8k/bench_sglang.py (+3/-0); python/sglang/bench_latency.py (+1/-0); python/sglang/srt/layers/activation.py (+0/-1); python/sglang/srt/layers/logits_processor.py (+4/-4); python/sglang/srt/model_executor/model_runner.py (+2/-2); python/sglang/srt/models/grok.py (+49/-395); python/sglang/srt/models/mixtral.py (+0/-1); (+1 more)
BODY: 

### L1-f6af3a6561  (L1, 2024-08-24, sha f6af3a6561b2, PR #1194)
TITLE: Cleanup readme, llava examples, usage examples and nccl init (#1194)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: README.md (+18/-21); docs/en/sampling_params.md (+7/-2); examples/frontend_language/quick_start/anthropic_example_chat.py (+0/-0); examples/frontend_language/quick_start/anthropic_example_complete.py (+0/-0); examples/frontend_language/quick_start/azure_openai_example_chat.py (+0/-0); examples/frontend_language/quick_start/gemini_example_chat.py (+0/-0); examples/frontend_language/quick_start/gemini_example_complete.py (+0/-0); examples/frontend_language/quick_start/gemini_example_multimodal_chat.py (+0/-0); examples/frontend_language/quick_start/images/cat.jpeg (+0/-0); examples/frontend_language/quick_start/images/dog.jpeg (+0/-0); (+55 more)
BODY: - Clean up readme ⏎ - Clean up llava examples. Deprecate the old llava v1.5/1.6 examples. Use llava-onevision in most examples. ⏎ - Separate examples in `examples` folder. Create two subfolder `frontend_language` and `runtime`. ⏎ - Fix the bug in #1015 ⏎ - Other code style improvements

### L1-c33d82a211  (L1, 2024-09-12, sha c33d82a21114, PR #1397)
TITLE: Add Support for XVERSE Models (Dense and MoE) to sglang (#1397)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: README.md (+1/-0); python/sglang/srt/models/xverse.py (+380/-0); python/sglang/srt/models/xverse_moe.py (+449/-0)
BODY: ## Motivation ⏎  ⏎ Hello, I am willhe from the [XVERSE](https://huggingface.co/xverse) Infra Team. ⏎ We want to integrate our XVERSE models into the sglang framework, including: ⏎ - XVERSE Dense Model, which is based on the LLaMA architecture. ⏎ - XVERSE MoE Model, which follows the structure of the ^FDeepseek-v1 model. ⏎  ⏎ XVERSE models are critical for serving various AI applications, and integrating them into sglang will allow us to efficiently depl …[truncated]

### L1-3a6e04185b  (L1, 2024-09-17, sha 3a6e04185b8d, PR #1420)
TITLE: [Feature, Hardware] Enable SGLang on AMD GPUs via PyTorch for ROCm (#1420)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe/layer.py (+27/-7); python/sglang/srt/layers/activation.py (+12/-0); python/sglang/srt/layers/attention_backend.py (+11/-7); python/sglang/srt/layers/layernorm.py (+12/-0); python/sglang/srt/layers/sampler.py (+10/-6); python/sglang/srt/lora/lora_manager.py (+5/-2); python/sglang/srt/models/deepseek_v2.py (+5/-1); python/sglang/srt/models/minicpm3.py (+5/-1); python/sglang/srt/server.py (+5/-0); python/sglang/srt/server_args.py (+7/-0); (+1 more)
BODY: ## Motivation ⏎ - Enable SGLang on AMD GPUs ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ - Bypass `FlashInfer` backend untill it is available on AMD/ROCm ⏎ - Add proper fix for AMD FP8 `e4m3fnuz` to support Fused_MoE ⏎ - Dependency over `vLLM>=0.5.5`, I modified `pyproject.toml` just to confirm that it works up to 0.6.0 as well. ⏎ - Misc. ⏎  ⏎ - TODO: follow-up to address one error (below) when `cuda-graph` is enabled. ⏎ ``` ⏎ File "/sglang/python/sglang/srt/layers/sampler. …[truncated]

### L1-8d4ed42ad5  (L1, 2024-09-24, sha 8d4ed42ad51d, PR #1497)
TITLE: MoE torch compile (#1497)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.fused_moe_patch
FILES: python/sglang/srt/layers/fused_moe/patch.py (+117/-0); python/sglang/srt/model_executor/cuda_graph_runner.py (+9/-5)
BODY: ## Motivation ⏎  ⏎ Temporarily workaround MoE torch compile with monkey patch. ⏎  ⏎ ## Bench Latency ⏎ ```bash ⏎ python3 -m sglang.bench_latency --model deepseek-ai/DeepSeek-V2-Lite --disable-radix --trust-remote-code --input-len 128 --output-len 8 --batch 1 --enable-torch-compile --max-torch-compile-bs 1 ⏎  ⏎ # bs=1, w/o torch compile ⏎ Decode.  median latency: 0.00921 s, median throughput:    108.61 token/s ⏎ Total. latency:  0.101 s, throughput:   1352. …[truncated]

### L1-8cdc76f6d4  (L1, 2024-10-02, sha 8cdc76f6d4cd, PR #1554)
TITLE: [Performance, Hardware] MoE tuning on AMD MI300x GPUs (#1554)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/fused_moe/configs/E=8,N=4096,device_name=AMD_Instinct_MI300X,dtype=float8.json (+57/-0)
BODY: ## Motivation ⏎ Optimize MoE kernel performance for AMD platform ⏎  ⏎  ⏎ ## Modifications ⏎ Add configuration file ⏎  ⏎  ⏎ ## Checklist

### L1-e11ab79e68  (L1, 2024-10-10, sha e11ab79e68c1, PR #1619)
TITLE: [Performance, hardware] MoE tuning update to AMD MI300x GPUs (#1619)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/fused_moe/configs/E=8,N=4096,device_name=AMD_Instinct_MI300X,dtype=float8.json (+178/-57); python/sglang/srt/layers/fused_moe/configs/E=8,N=8192,device_name=AMD_Instinct_MI300X,dtype=float8.json (+175/-0)
BODY: ## Motivation ⏎  ⏎ Tuning update to optimize MoE kernels for AMD MI300x. ⏎  ⏎ ## Modifications ⏎  ⏎ Updated configuration files. ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/en/contributor_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/en/contributor_guide.md). ⏎ - [+] Update documentation a …[truncated]

### L1-5f65e2b830  (L1, 2024-10-30, sha 5f65e2b830a4, PR #1836)
TITLE: [Performance, Hardware] MoE weights padding to AMD MI300x GPUs (#1836)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe/fused_moe.py (+4/-3); python/sglang/srt/layers/fused_moe/layer.py (+28/-0)
BODY: ## Motivation ⏎  ⏎ Padding MoE weights (last dim) to minimize Memory Channel Contention (only to AMD Instinct GPUs) ⏎ Test shows approximate performance boost of prefill +2.2%, decode +3.0% for Grok-1 on setting: b32/i1024/o512 ⏎  ⏎ ## Modifications ⏎  ⏎ As mentioned: fused_moe.py and layer.py ⏎ To enable this feature, set binary flag `MOE_PADDING=1` at command line, or `export MOE_PADDING=1` in console. ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code accordin …[truncated]

### L1-087ab83223  (L1, 2024-11-10, sha 087ab832236e, PR #1980)
TITLE: [Performance, Triton] Optimize over mask compute to tl.load in fused_moe_kernel (#1980)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe/fused_moe.py (+23/-7); python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+7/-0)
BODY: ## Motivation ⏎  ⏎ Test shows `~0.5%` boost to prefill, `~1.0%` boost to median decode throughput over Grok-1 with `b32/i1023/o256` settings. ⏎  ⏎ ## Modifications ⏎  ⏎ `fused_moe_kernel`: simplify the mask part for even K BLOCK sizes ⏎ `decode_attention`: adjust kernel args ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Add unit …[truncated]

### L1-00ffde206f  (L1, 2024-11-11, sha 00ffde206f89, PR #1999)
TITLE: setup router python binding ci (#1999)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: rust/py_src/router.py (+48/-0); .github/workflows/release-pypi-router.yml (+104/-0); rust/Cargo.lock (+1/-1); rust/Cargo.toml (+3/-3); rust/MANIFEST.in (+3/-0); rust/README.md (+71/-0); rust/demo.py (+0/-0); rust/dp_demo.py (+0/-0); rust/py_src/__init__.py (+5/-0); rust/pyproject.toml (+18/-8); (+3 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ attempt to publish python binding for router. The proof-of-concept was achieved at  https://github.com/ByronHsu/sglang-rust-test ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-86c37d010a  (L1, 2024-11-11, sha 86c37d010aea, PR #2005)
TITLE: fix sglang_router not found (#2005)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: rust/py_src/sglang_router/router.py (+0/-0); rust/demo.py (+1/-3); rust/py_src/sglang_router/__init__.py (+1/-1); rust/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ `import sglang_router` caused not found error because we should create a folder called `sglang_router` under `py_src` ⏎  ⏎ After adding that, the import succeeded locally ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c3eac1b010  (L1, 2024-11-14, sha c3eac1b010b3, PR #2033)
TITLE: Fix torch.compile for MoE (#2033)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.fused_moe_patch
FILES: python/sglang/srt/layers/fused_moe/patch.py (+4/-2); python/sglang/test/test_utils.py (+3/-2); test/srt/run_suite.py (+1/-0); test/srt/test_data_parallelism.py (+1/-1); test/srt/test_double_sparsity.py (+1/-1); test/srt/test_eval_accuracy_mini.py (+1/-1); test/srt/test_retract_decode.py (+1/-1); test/srt/test_torch_compile.py (+3/-3); test/srt/test_torch_compile_moe.py (+73/-0); test/srt/test_triton_attention_backend.py (+1/-1)
ISSUES: #2029 [Bug] Does Mixtral currently not support torch compile?
DEEP_STUDY: deep-study correctness case sglang:c3eac1b010: class=integration_backend_cudagraph; symptom=compile_or_build_failure; introducing=unknown
BODY: Fix https://github.com/sgl-project/sglang/issues/2029

### L1-f35cb46cc3  (L1, 2024-11-21, sha f35cb46cc376, PR #2111)
TITLE: ROCm: Fix MoE padding for none FP8 cases (#2111)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe/fused_moe.py (+11/-4)
BODY: ## Motivation ⏎  ⏎ As mentioned. ⏎  ⏎ ## Modifications ⏎  ⏎ Separate padding size handling to FP8 and None FP8 cases. ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Update documentat …[truncated]

### L1-cbedd1db1d  (L1, 2024-11-23, sha cbedd1db1d8b, PR #2114)
TITLE: [router] cache-aware load-balancing router v1 (#2114)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: rust/py_src/sglang_router/router.py (+20/-10); benchmark/multi_turn_chat/long_prompt_multi_turn.py (+42/-11); python/sglang/bench_serving.py (+2/-2); python/sglang/test/few_shot_gsm8k.py (+7/-3); rust/Cargo.lock (+23/-1); rust/Cargo.toml (+2/-0); rust/README.md (+3/-0); rust/demo.py (+0/-10); rust/dp_demo.py (+0/-156); rust/py_src/sglang_router/launch_router.py (+204/-0); (+7 more)
BODY: ## Motivation ⏎  ⏎ Related to https://github.com/sgl-project/sglang/issues/1732 ⏎  ⏎ This PR finishes the first version of cache-aware load-balancing router. For long shared prefix data, It can achieve 2x throughput compared with existing round-robin DP controller.  ⏎  ⏎ ## Usage ⏎ The router offers two modes: ⏎  ⏎ ### 1. Co-launch workers and router ⏎ This will be a drop-in replacement for the existing `--dp-size`. This part of code will be moved into sgl …[truncated]

### L1-5652c56535  (L1, 2024-11-24, sha 5652c565352c, PR #2159)
TITLE: Update CI threshold & Improve code style (#2159)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.fused_moe_patch
FILES: python/sglang/srt/layers/fused_moe_patch.py (+5/-0); .github/workflows/pr-test.yml (+34/-22); python/sglang/bench_one_batch.py (+1/-0); python/sglang/srt/managers/schedule_batch.py (+14/-9); python/sglang/srt/managers/scheduler.py (+6/-3); python/sglang/srt/model_executor/cuda_graph_runner.py (+1/-1); test/srt/test_bench_serving.py (+6/-6); test/srt/test_srt_endpoint.py (+59/-0)
BODY: - Increase the threshold for the performance tests in CI because the perf is better now. ⏎ - Improve the code style of ScheduleBatch

### L1-be0124bda0  (L1, 2024-11-24, sha be0124bda09d, PR #2163)
TITLE: Rename triton_fused_moe -> fused_moe_triton (#2163)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.grok_variant
FILES: python/sglang/srt/layers/fused_moe/__init__.py (+0/-1); python/sglang/srt/layers/fused_moe_grok/__init__.py (+1/-0); python/sglang/srt/layers/fused_moe_grok/configs/E=8,N=4096,device_name=AMD_Instinct_MI300X,dtype=float8.json (+0/-0); python/sglang/srt/layers/fused_moe_grok/configs/E=8,N=8192,device_name=AMD_Instinct_MI300X,dtype=float8.json (+0/-0); python/sglang/srt/layers/fused_moe_grok/fused_moe.py (+0/-0); python/sglang/srt/layers/fused_moe_grok/layer.py (+3/-3); python/sglang/srt/layers/fused_moe_triton/__init__.py (+3/-3); python/sglang/srt/layers/fused_moe_triton/configs/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); (+66 more)
BODY: 

### L1-b509db5832  (L1, 2024-11-24, sha b509db5832c9, PR #2153)
TITLE: feat: remove the dependency on FusedMoE (#2153)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/quantization/__init__.py (+15/-5); python/sglang/srt/layers/triton_fused_moe/__init__.py (+44/-0); python/sglang/srt/layers/triton_fused_moe/configs/README (+10/-0); python/sglang/srt/layers/triton_fused_moe/fused_moe.py (+858/-0); python/sglang/srt/layers/triton_fused_moe/layer.py (+631/-0); python/sglang/srt/models/deepseek_v2.py (+1/-1); python/sglang/srt/utils.py (+43/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9e8f8fbf95  (L1, 2024-11-24, sha 9e8f8fbf95a1, PR #2155)
TITLE: feat: update gitignore and add tuning config for FusedMoE (#2155)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+146/-0); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB.json (+146/-0); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+218/-0); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB.json (+218/-0); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=3072,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+218/-0); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=3072,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json (+218/-0); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=3072,device_name=NVIDIA_H100_80GB_HBM3.json (+218/-0); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=3584,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+218/-0); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json (+218/-0); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+218/-0); (+48 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d90c3d6b8b  (L1, 2024-11-24, sha d90c3d6b8bcc, PR #2157)
TITLE: fix: resolve end-of-file-fixer (#2157)
SOURCES: path_config_only
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB.json (+1/-1); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+1/-1); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB.json (+1/-1); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=3072,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+1/-1); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=3072,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json (+1/-1); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=3072,device_name=NVIDIA_H100_80GB_HBM3.json (+1/-1); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=3584,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+1/-1); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json (+1/-1); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+1/-1); python/sglang/srt/layers/triton_fused_moe/configs/E=1,N=7168,device_name=NVIDIA_A100-SXM4-80GB.json (+1/-1); (+15 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e3938b2f9c  (L1, 2024-11-24, sha e3938b2f9c96, PR #2156)
TITLE: feat: update other MoE models deps (#2156)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe/layer.py (+1/-6); python/sglang/srt/layers/triton_fused_moe/fused_moe.py (+2/-0); python/sglang/srt/layers/triton_fused_moe/layer.py (+4/-2); python/sglang/srt/models/dbrx.py (+1/-1); python/sglang/srt/models/deepseek.py (+1/-1); python/sglang/srt/models/mixtral.py (+1/-1); python/sglang/srt/models/olmoe.py (+1/-1); python/sglang/srt/models/qwen2_moe.py (+1/-1); python/sglang/srt/models/xverse_moe.py (+1/-1); python/sglang/srt/utils.py (+15/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fa27161380  (L1, 2024-11-24, sha fa271613809b, PR #2161)
TITLE: fix: use torch.sum for compatible (#2161)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/triton_fused_moe/fused_moe.py (+3/-2)
ISSUES: #2160 [Bug] FusedMoE compatible with vllm 0.6.3.post1
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ fix https://github.com/sgl-project/sglang/issues/2160 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4b0a1c9365  (L1, 2024-11-24, sha 4b0a1c9365ef, PR #2170)
TITLE: Replace prob based with threshold based load balancing  (#2170)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: rust/py_src/sglang_router/router.py (+9/-6); python/sglang/bench_serving.py (+36/-23); rust/README.md (+40/-25); rust/py_src/sglang_router/launch_router.py (+16/-7); rust/src/lib.rs (+10/-5); rust/src/main.rs (+15/-6); rust/src/router.rs (+109/-91)
BODY: ## Motivation ⏎  ⏎  ⏎ Prob based LB can disturb cache aware when the load is actually balanced. Switching to threshold based LB can help improve the cache hit rate and also makes the behavior more deterministic. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ 1. `bench_serving`: for generated shared prefix dataset, Remove argument-based data caching and use auto caching with the key as the dataset params, so users don't have to manually configure the arguments ⏎ 2. Short …[truncated]

### L1-dd44173dad  (L1, 2024-11-25, sha dd44173dad4e, PR #2167)
TITLE: [Fused moe] add tuning fused configs for qwen2 57b and mixtral 8x7b (#2167)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/fused_moe_triton/configs/E=64,N=640,device_name=NVIDIA_GeForce_RTX_4090,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=8,N=3584,device_name=NVIDIA_GeForce_RTX_4090,dtype=fp8_w8a8.json (+146/-0)
BODY: Using FP8 quantization for inference on RTX 4090 can significantly improve the performance of both Qwen2-57B and Mixtral 8x7B models.

### L1-4d62bca542  (L1, 2024-11-25, sha 4d62bca54294, PR #2183)
TITLE: [router] Replace print with logger (#2183)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: rust/py_src/sglang_router/router.py (+3/-0); rust/Cargo.lock (+102/-0); rust/Cargo.toml (+3/-0); rust/py_src/sglang_router/launch_router.py (+28/-1); rust/py_src/sglang_router/launch_server.py (+35/-14); rust/src/lib.rs (+14/-8); rust/src/main.rs (+12/-2); rust/src/router.rs (+5/-5); rust/src/server.rs (+43/-14); rust/src/tree.rs (+5/-4)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Looks something like this ⏎ ``` ⏎ [Router (Python)] 2024-11-25 21:28:33 - INFO - Launching DP server process 0 on port 31000 ⏎ [Router (Python)] 2024-11-25 21:28:33 - INFO - Launching DP server process 1 on port 31814 ⏎ [Router (Rust)] 2024-11-25 21:29:40 - INFO - Tenant: http://127.0.0.1:31814, Size: 0 ⏎ [Router (Rust)] 2024-11-25 21:29:40 - INFO - Tenant: http://127.0.0.1:31000, Size: 0 ⏎ [Router (Rust)] 2 …[truncated]

### L1-55842eb81a  (L1, 2024-11-25, sha 55842eb81a78, PR #2174)
TITLE: feat: fused_moe fp8 monkey patch (#2174)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/__init__.py (+68/-18)
BODY: ## Motivation ⏎  ⏎ as titled cc @ispobock  ⏎  ⏎ ``` ⏎ root@id:/sgl-workspace/sglang/test/srt# python3 test_mla_fp8.py ⏎ [2024-11-25 00:47:01] server_args=ServerArgs(model_path='neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8', tokenizer_path='neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8', tokenizer_mode='auto', skip_tokenizer_init=False, load_format='auto', trust_remote_code=True, dtype='auto', kv_cache_dtype='fp8_e5m2', quantization=None, context_l …[truncated]

### L1-dd5eba4c88  (L1, 2024-11-27, sha dd5eba4c8899, PR #2223)
TITLE: Remove fused_moe_grok (#2223)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.grok_variant
FILES: python/sglang/srt/layers/fused_moe_grok/__init__.py (+0/-1); python/sglang/srt/layers/fused_moe_grok/fused_moe.py (+0/-692); python/sglang/srt/layers/fused_moe_grok/layer.py (+0/-630); python/sglang/srt/layers/fused_moe_triton/configs/E=8,N=4096,device_name=AMD_Instinct_MI300X,dtype=float8.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=8,N=8192,device_name=AMD_Instinct_MI300X,dtype=float8.json (+0/-0); python/sglang/srt/models/grok.py (+11/-48); 3rdparty/amd/tuning/benchmark_moe_rocm.py (+1/-1)
BODY: We do not want to maintain a separate fused moe folder. This unifies them

### L1-cd51758fad  (L1, 2024-11-27, sha cd51758fade4, PR #2228)
TITLE: Rename tuned MI300X config files for fused_moe_triton (#2228)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/fused_moe_triton/configs/E=8,N=4096,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json (+0/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=8,N=8192,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json (+0/-0)
BODY: ## Motivation ⏎  ⏎  ⏎ Rename tuned config files from fused_moe_triton changes ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ As it is. ⏎  ⏎ This recovers most performance. ⏎ Still ~3.0-4.x% perf drop from v0.3.6.post2, which will be looked into further. ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guid …[truncated]

### L1-a9ca297d76  (L1, 2024-11-28, sha a9ca297d769b, PR #2191)
TITLE: [3rdparty, document] Updated Documentation that for triton fused_moe kernel tuning for AMD Instinct GPUs (#2191)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: 3rdparty/amd/tuning/TUNING.md (+17/-0); 3rdparty/amd/tuning/benchmark_moe_rocm.py (+377/-0)
BODY: ## Motivation ⏎  ⏎ Updated Documentation for triton fused_moe kernel tuning for AMD Instinct GPUs. ⏎  ⏎ ## Modifications ⏎  ⏎ - Upload a tuning script file ⏎ - introduce the tuning parameters setting ⏎ - Provided example bash commands for tuning script run ⏎  ⏎ ## Checklist ⏎  ⏎ - [O] Update documentation as needed, including docstrings or example tutorials.

### L1-262e370f78  (L1, 2024-11-29, sha 262e370f78c0, PR #2225)
TITLE: [benchmark] Add fused_moe_triton benchmark and tuning tools (#2225)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/server_args.py (+4/-3); benchmark/kernels/fused_moe_triton/README.md (+45/-0); benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py (+237/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+446/-0)
BODY: The tuning scripts is related to [this pr](https://github.com/sgl-project/sglang/pull/2167), and it successfull produced efficient `fused_moe_triton` kernel config for qwen2-57b and mixtral 8x7b in both tp4 `fp8_w8a8` condition. ⏎  ⏎ In GTX 4090, I have checked benchmark script in `Qwen/Qwen2-57B-A14B-Instruct-FP8` model's `fused_moe_triton` kernel, the result: ⏎  ⏎ ```shell ⏎ python benchmark/kernels/fused_moe_triton/fused_moe_triton/benchmark_vllm_v …[truncated]

### L1-fae4e5e99a  (L1, 2024-11-30, sha fae4e5e99a93, PR #2259)
TITLE: chore: bump v0.3.6.post3 (#2259)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+19/-18); docker/Dockerfile.rocm (+1/-1); python/pyproject.toml (+2/-2); Makefile (+27/-1); docs/developer/setup_github_runner.md (+2/-2); docs/start/install.md (+7/-13); python/sglang/version.py (+1/-1); scripts/ci_install_dependency.sh (+1/-2)
BODY: ## Motivation ⏎  ⏎ install with one-click, set `flashinfer` in deps ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-7301a39b13  (L1, 2024-12-01, sha 7301a39b13c7, PR #2305)
TITLE: fix: resolve CodeQL cpp issue (#2305)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.dev (+7/-1); sgl-kernel/CMakeLists.txt (+47/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-33deca81b5  (L1, 2024-12-02, sha 33deca81b5e3, PR #2314)
TITLE: Add more fused moe benchmark utilities (#2314)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py (+275/-0); benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py (+6/-15); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+13/-8)
BODY: Add a benchmark against the torch.compile

### L1-07ec07ad1f  (L1, 2024-12-03, sha 07ec07ad1fa5, PR #2327)
TITLE: Improve torch compile for fused moe (#2327)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.fused_moe_patch
FILES: python/sglang/srt/layers/fused_moe_patch.py (+20/-11); python/sglang/srt/model_executor/cuda_graph_runner.py (+16/-7); python/sglang/srt/model_executor/model_runner.py (+1/-1); benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py (+5/-2); test/srt/test_srt_engine.py (+1/-1); test/srt/test_torch_compile_moe.py (+2/-2)
BODY: Following the discussion from https://github.com/sgl-project/sglang/issues/2278. ⏎  ⏎ The final solution: ⏎ - Only do torch.compile for moe layers at bs=1 ⏎ - Skip torch.compile for moe layers when bs > 1 ⏎ - Add `dynamic=False` in places where the shape should be known

### L1-2db4469808  (L1, 2024-12-05, sha 2db446980815, PR #2350)
TITLE: minor: limit the range of vllm versions (#2350)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/__init__.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3d32e4a32c  (L1, 2024-12-06, sha 3d32e4a32c4c, PR #2371)
TITLE: Resubmit MoE-EP (#2371)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/ep_moe/__init__.py (+0/-0); python/sglang/srt/layers/ep_moe/kernels.py (+349/-0); python/sglang/srt/layers/ep_moe/layer.py (+661/-0); .github/workflows/pr-test.yml (+6/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/model_executor/model_runner.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+5/-3); python/sglang/srt/models/mixtral.py (+13/-5); python/sglang/srt/server_args.py (+23/-0); test/srt/test_moe_ep.py (+113/-0)
BODY: Resubmit the PR for MoE-EP.  Please refer to the details in the previous PR: https://github.com/sgl-project/sglang/pull/2203.

### L1-84d96b3ae5  (L1, 2024-12-06, sha 84d96b3ae52e, PR #2370)
TITLE: Move FP8 to SGLang (#2370)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/__init__.py (+2/-2); python/sglang/srt/layers/quantization/fp8.py (+559/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ cc @HaiShaw  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d332aa3b0c  (L1, 2024-12-07, sha d332aa3b0c0a, PR #2387)
TITLE: fix: resolve fp8 moe issue (#2387)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/__init__.py (+2/-47); python/sglang/srt/layers/quantization/fp8.py (+25/-9)
LABELS: bug, high priority
ISSUES: #2386 [Bug] circular import error in fused_moe_triton
BODY: ## Motivation ⏎  ⏎ fix https://github.com/sgl-project/sglang/issues/2386 https://github.com/sgl-project/sglang/pull/2370 https://github.com/sgl-project/sglang/pull/2366 ⏎ cc @BBuf @HaiShaw  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-95f93f493a  (L1, 2024-12-07, sha 95f93f493a60, PR #2388)
TITLE: Fp8 MoE optimizations on AMD (#2388)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe_triton/fused_moe.py (+64/-21); python/sglang/srt/layers/quantization/fp8.py (+33/-1)
BODY: ## Motivation ⏎  ⏎  ⏎ Recover AMD optimizations over fused_moe_triton ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/issues/2347 ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide …[truncated]

### L1-3844feb9bb  (L1, 2024-12-08, sha 3844feb9bb1c, PR #2416)
TITLE: Add a unittest for fused_moe (#2416)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/fused_moe_triton/configs/E=64,N=1280,device_name=NVIDIA_A800-SXM4-80GB.json (+146/-0); python/sglang/srt/layers/fused_moe_triton/configs/E=64,N=640,device_name=NVIDIA_A800-SXM4-80GB.json (+146/-0); benchmark/kernels/fused_moe_triton/README.md (+6/-2); test/srt/run_suite.py (+1/-0); test/srt/test_fused_moe.py (+126/-0)
BODY: When I wan't to deploy qwen2-57b-a14b model in A800 with fp8, the error happens: ⏎  ⏎ <img width="1337" alt="图片" src="https://github.com/user-attachments/assets/21b4d0e6-9ecc-4d04-b3fa-aea89b052d11"> ⏎  ⏎ The reason is that in Triton, the Ampere architecture currently doesn't support the `fp8e4nv` dtype. To detect this situation early, I added the `fused_moe` test mentioned above, and verify whether the `fused_moe` operator can work properly on the c …[truncated]

### L1-2ac36b9a7b  (L1, 2024-12-11, sha 2ac36b9a7bd6, PR #2444)
TITLE: Make request payload size configurable (#2444)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: rust/py_src/sglang_router/router.py (+3/-0); .github/workflows/pr-test-rust.yml (+1/-0); rust/py_src/sglang_router/launch_router.py (+10/-1); rust/py_test/test_launch_router.py (+1/-0); rust/py_test/test_launch_server.py (+55/-2); rust/src/lib.rs (+5/-0); rust/src/server.rs (+7/-0)
BODY: ## Motivation ⏎ For scenarios where we have long context lengths (e.g. 32k) and have multiple prompts in a single completion request, the size of the http request payload exceeds the default 2MB that rust json has which fails to send the request to the engine. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ This PR makes the request payload size configurable by adding a new `max_request_payload_size` field to the `Config` struct. ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e3b3acfa6f  (L1, 2024-12-12, sha e3b3acfa6fff, PR #2464)
TITLE: Rename rust folder to sgl-router (#2464)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+0/-0); .github/CODEOWNERS (+1/-1); .github/workflows/pr-test-rust.yml (+6/-6); .github/workflows/release-pypi-router.yml (+5/-5); docs/router/router.md (+1/-1); sgl-router/Cargo.lock (+0/-0); sgl-router/Cargo.toml (+0/-0); sgl-router/MANIFEST.in (+0/-0); sgl-router/README.md (+0/-0); sgl-router/py_src/sglang_router/__init__.py (+0/-0); (+12 more)
BODY: ## Motivation ⏎ Rename rust folder to sgl-router to be consistent with other sub packages. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5f2595be43  (L1, 2024-12-15, sha 5f2595be4302, PR #2485)
TITLE: hotfix: checking for HIP (#2485)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/fused_moe_triton/layer.py (+1/-1); python/sglang/srt/utils.py (+1/-7)
BODY: ## Motivation ⏎  ⏎ https://pytorch.org/docs/stable/notes/hip.html ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e04d3f2897  (L1, 2024-12-15, sha e04d3f289753, PR #2481)
TITLE: adapt tensorrt llm custom all reduce to sgl-kernel (#2481)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+47/-19); sgl-kernel/Makefile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/setup.py (+25/-1); sgl-kernel/src/sgl-kernel/__init__.py (+7/-2); sgl-kernel/src/sgl-kernel/csrc/trt_reduce.cc (+13/-0); sgl-kernel/src/sgl-kernel/csrc/trt_reduce_internal.cu (+282/-0); sgl-kernel/src/sgl-kernel/csrc/trt_reduce_internal.cuh (+91/-0); sgl-kernel/src/sgl-kernel/csrc/trt_reduce_kernel.cu (+102/-0); sgl-kernel/src/sgl-kernel/csrc/utils.hpp (+36/-0); (+3 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ - move tensorrt custom allreduce algoithm to sgl-kernel, make adaption for python, add test for custom allreduce ⏎ - we **do not use twoshot allreduce kernel** from tensorrt llm since it is disabled [here](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/plugins/ncclPlugin/allreducePlugin.cpp#L192) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎  ⏎ ## Next

### L1-b532a5fd16  (L1, 2024-12-16, sha b532a5fd16d0, PR #2489)
TITLE: fix moe-ep accuracy issue for fp8 (#2489)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/ep_moe/layer.py (+4/-0)
BODY: ## Motivation ⏎  ⏎  fix moe ep bug, when load fp8 model.  Links to related issues [link](https://github.com/sgl-project/sglang/issues/2482) ⏎  ⏎ Test model : neuralmagic/DeepSeek-Coder-V2-Instruct-FP8 ⏎ ``` ⏎ Accuracy: 0.932 ⏎ Invalid: 0.000 ⏎ Latency: 243.824 s ⏎ Output throughput: 1027.530 token/s ⏎ ``` ⏎  ⏎ cc: @ispobock  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ ## Checklist

### L1-4b83db24f1  (L1, 2024-12-19, sha 4b83db24f128, PR #2517)
TITLE: fix: continue to use flashinfer 0.1.6 temporarily (#2517)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎ If using the latest 0.2.0, CI cannot pass. Temporarily fix it to 0.1.6 and wait for FlashInfer to release 0.2.0.post1 or 0.2.1 before lifting the restriction again. ⏎  ⏎ ref https://github.com/sgl-project/sglang/actions/runs/12407074179 ⏎  ⏎ cc @yzh119 @merrymercy  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e835a50021  (L1, 2024-12-24, sha e835a50021e0, PR #2563)
TITLE: Reorg moe code (#2563)
SOURCES: path_core, symbol_pickaxe, corpus:production-kernel-provenance
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.ep.layer, L1.triton.fused_moe_patch
FILES: python/sglang/srt/layers/fused_moe_patch.py (+0/-133); python/sglang/srt/layers/moe/ep_moe/__init__.py (+0/-0); python/sglang/srt/layers/moe/ep_moe/kernels.py (+0/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+14/-39); python/sglang/srt/layers/moe/fused_moe_native.py (+46/-0); python/sglang/srt/layers/moe/fused_moe_triton/__init__.py (+3/-7); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); (+78 more)
BODY: ## Motivation ⏎  ⏎ Reorg moe code and reuse common part. ⏎  ⏎ ## Checklist

### L1-53aed988cb  (L1, 2024-12-26, sha 53aed988cbaa, PR #2575)
TITLE: Refactor MoE (#2575)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+78/-8); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-1); python/sglang/srt/configs/model_config.py (+4/-1); python/sglang/srt/layers/linear.py (+20/-2); python/sglang/srt/layers/quantization/fp8.py (+159/-25); python/sglang/srt/layers/quantization/fp8_kernel.py (+278/-0); python/sglang/srt/layers/quantization/fp8_utils.py (+90/-1); python/sglang/srt/models/deepseek_v2.py (+36/-11); python/sglang/test/test_block_fp8.py (+341/-0)
BODY: Support w8a8 fp8 block-wise quantization.

### L1-9a23c48456  (L1, 2024-12-26, sha 9a23c4845627, PR #2560)
TITLE: h100 tuning fused_moe_triton for qwen2 moe (#2560)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=1280,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=1280,device_name=NVIDIA_H100_80GB_HBM3.json (+31/-31); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=2560,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=320,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=320,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=640,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=640,device_name=NVIDIA_H100_80GB_HBM3.json (+21/-21); benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py (+6/-1); benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py (+6/-1); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+6/-1)
BODY: Tuning qwen2 moe(57b) in H100:

### L1-31548116a8  (L1, 2024-12-26, sha 31548116a8dc, PR #2579)
TITLE: fix moe_align_block_size_kernel for shared memory issue (#2579)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/setup.py (+20/-0); sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+151/-0); sgl-kernel/src/sgl-kernel/__init__.py (+8/-1); sgl-kernel/src/sgl-kernel/ops/__init__.py (+19/-0); sgl-kernel/tests/test_moe_align.py (+26/-0)
BODY: ## Motivation ⏎  ⏎ thanks @fengyang95 for the solution ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-60e2fdcf4f  (L1, 2024-12-26, sha 60e2fdcf4fdb, PR #2581)
TITLE: use sgl-kernel moe_align_block_size (#2581)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+20/-3); python/sglang/srt/model_executor/model_runner.py (+6/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-efc52f85e2  (L1, 2024-12-26, sha efc52f85e2d5, PR #2582)
TITLE: chore: bump v0.4.1 (#2582)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile.rocm (+1/-1); python/pyproject.toml (+2/-2); docs/developer/setup_github_runner.md (+2/-2); docs/start/install.md (+5/-5); python/sglang/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-dc3bee4815  (L1, 2024-12-26, sha dc3bee481518, PR #2598)
TITLE: Fix test and benchmark scripts (#2598)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/README (+2/-0); .github/workflows/nightly-test.yml (+7/-13); .github/workflows/pr-test.yml (+2/-2); docs/references/benchmark_and_profiling.md (+2/-0); python/sglang/bench_serving.py (+11/-3); sgl-kernel/tests/.gitkeep (+0/-0); test/lang/run_suite.py (+1/-1); test/srt/run_suite.py (+1/-1); test/srt/test_triton_attention_backend.py (+1/-1)
BODY: Organize tests as suites in `run_suite.py` We will have `per-commit` and `nightly`

### L1-2dccecf432  (L1, 2024-12-26, sha 2dccecf43261, PR #2590)
TITLE: fix: only enable moe_align_block_size for now (#2590)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/src/sgl-kernel/__init__.py (+1/-11); sgl-kernel/src/sgl-kernel/ops/__init__.py (+0/-20)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-635a042623  (L1, 2024-12-26, sha 635a04262396, PR #2592)
TITLE: docs: update deepseek v3 example (#2592)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.routing.topk_py, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/topk.py (+14/-0); benchmark/deepseek_v3/README.md (+30/-0); python/sglang/srt/layers/quantization/fp8_kernel.py (+14/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-7722c11c1d  (L1, 2024-12-26, sha 7722c11c1d2a, PR #2606)
TITLE: Regression fix to AMD/ROCm from recent change (#2606)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+8/-3)
BODY: ## Motivation ⏎  ⏎  ⏎ As it is. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Update documentation as needed, including docstrings or example tutorials.

### L1-7ca751ff7d  (L1, 2024-12-26, sha 7ca751ff7d8d, PR #2612)
TITLE: Fused moe triton cfg opt for rocm (#2612)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=4096,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json (+14/-14)
BODY: ## Motivation ⏎  ⏎ As it is. ⏎  ⏎ ## Modifications ⏎  ⏎ Change the triton moe configuration file ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Update documentation as needed, includ …[truncated]

### L1-77d1210b36  (L1, 2024-12-27, sha 77d1210b3610, PR #2615)
TITLE: fix moe_align_block_size (#2615)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+4/-16); sgl-kernel/src/sgl-kernel/ops/__init__.py (+4/-0); sgl-kernel/tests/test_moe_align.py (+15/-1)
BODY: 

### L1-6e5305158c  (L1, 2024-12-28, sha 6e5305158cde, PR #2617)
TITLE: update sgl_moe_align_block_size usage (#2617)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+10/-2); python/sglang/srt/model_executor/model_runner.py (+0/-6)
BODY: 

### L1-9254a33ad4  (L1, 2024-12-28, sha 9254a33ad46d, PR #2624)
TITLE: avoid fused_moe_triton `padding` circular import (#2624)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+4/-1)
BODY: avoid fused_moe_triton `padding` circular import.

### L1-7863e4368a  (L1, 2024-12-28, sha 7863e4368abf, PR #2628)
TITLE: add configs for block fp8 related kernels (#2628)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+47/-26); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+51/-8); python/sglang/srt/layers/quantization/configs/N=1536,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=1536,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=2048,K=512,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=2048,K=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=2304,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=2304,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); (+27 more)
BODY: ## Motivation ⏎  ⏎ tune by @HandH1998 I help verify. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3815b23ccb  (L1, 2024-12-29, sha 3815b23ccb3d, PR #2638)
TITLE: Clean up wrapper in flashinfer backend (#2638)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/bench_offline_throughput.py (+1/-0); python/sglang/srt/configs/model_config.py (+1/-3); python/sglang/srt/layers/attention/__init__.py (+0/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+54/-41); python/sglang/srt/layers/logits_processor.py (+30/-2); python/sglang/srt/managers/schedule_batch.py (+2/-2); python/sglang/srt/managers/scheduler.py (+18/-10); python/sglang/srt/model_executor/forward_batch_info.py (+42/-3); python/sglang/srt/models/llama.py (+11/-0); python/sglang/srt/server.py (+2/-2); (+2 more)
BODY: - Explicitly specify all wrappers in flashinfer_backend.py ⏎ - Some preliminary components for eagle speculative decoding

### L1-afa0341e57  (L1, 2024-12-29, sha afa0341e57ec, PR #2641)
TITLE: Update Triton configs for block fp8 kernels (#2641)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+14/-14); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+11/-11); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-2); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+1/-2); python/sglang/srt/layers/quantization/configs/N=1536,K=1536,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=1536,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+33/-33); python/sglang/srt/layers/quantization/configs/N=1536,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+30/-30); python/sglang/srt/layers/quantization/configs/N=2048,K=512,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+31/-31); python/sglang/srt/layers/quantization/configs/N=2048,K=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+35/-35); python/sglang/srt/layers/quantization/configs/N=2304,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+30/-30); (+33 more)
BODY: Update Triton configs for block fp8 kernels

### L1-b02da24a5b  (L1, 2024-12-30, sha b02da24a5b8c, PR #2642)
TITLE: Refactor sgl-kernel build (#2642)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/CMakeLists.txt (+6/-23); sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+2/-6); sgl-kernel/setup.py (+34/-67); sgl-kernel/src/sgl-kernel/__init__.py (+11/-1); sgl-kernel/src/sgl-kernel/csrc/sgl_kernel_ops.cu (+32/-0); sgl-kernel/src/sgl-kernel/csrc/trt_reduce.cc (+0/-13); sgl-kernel/src/sgl-kernel/csrc/warp_reduce.cc (+0/-14); sgl-kernel/src/sgl-kernel/csrc/warp_reduce_kernel.cu (+2/-1); sgl-kernel/src/sgl-kernel/ops/__init__.py (+21/-1)
BODY: ## Motivation ⏎  ⏎ Involve all ops into one target. Reduce redundant build code.

### L1-b4403985d0  (L1, 2024-12-31, sha b4403985d009, PR #2676)
TITLE: Add cutlass submodule for sgl-kernel (#2676)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+3/-0); sgl-kernel/CMakeLists.txt (+4/-0); sgl-kernel/3rdparty/cutlass (+1/-0); sgl-kernel/setup.py (+6/-0)
BODY: ## Motivation ⏎  ⏎ Include CUTLASS (v3.6.0) for kernels dev.

### L1-b6b57fc200  (L1, 2024-12-31, sha b6b57fc20075, PR #2679)
TITLE: minor: cleanup sgl-kernel (#2679)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+0/-1); sgl-kernel/setup.py (+0/-37); sgl-kernel/src/sgl-kernel/__init__.py (+0/-2); sgl-kernel/src/sgl-kernel/csrc/sgl_kernel_ops.cu (+0/-10); sgl-kernel/src/sgl-kernel/csrc/warp_reduce_kernel.cu (+0/-90); sgl-kernel/src/sgl-kernel/ops/__init__.py (+0/-5)
BODY: ## Motivation ⏎  ⏎ Warp reduce was initially added as an example and is no longer necessary. The hack logic for renaming in setup is also redundant now. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-286cad3ee3  (L1, 2024-12-31, sha 286cad3ee31b, PR #2689)
TITLE: h200 tuning  fused_moe_triton config for  Mixtral 8x7B/8x22B and Qwen2 57BA14B (#2689)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=1280,device_name=NVIDIA_H200,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=1280,device_name=NVIDIA_H200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=2560,device_name=NVIDIA_H200,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=2560,device_name=NVIDIA_H200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=320,device_name=NVIDIA_H200,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=320,device_name=NVIDIA_H200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=64,N=640,device_name=NVIDIA_H200.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=14336,device_name=NVIDIA_H200,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=14336,device_name=NVIDIA_H200.json (+146/-0); (+11 more)
BODY: related issue: https://github.com/sgl-project/sglang/issues/2471 ⏎  ⏎ add h200 tuning `fused_moe_triton` kernel config for   mixtral 8x7b/8x22b and qwen2 57ba14b. ⏎  ⏎ - For mixtral 8x7b, H200 can serving with tp1/tp2/tp4/tp8 in BF16 and FP8.  ⏎ - For Miaxtral 8x22b, H200 can serving with tp4/tp8 in BF16, and tp2/tp4/tp8 in FP8. ⏎ - For Qwen257BA14B, H200 can serving with tp1/tp2/tp4/tp8 in BF16 and FP8. ⏎  ⏎ In total, there are 8+2+3+8=**21** configs.

### L1-148254d4db  (L1, 2025-01-02, sha 148254d4db8b, PR #2705)
TITLE: Improve moe reduce sum kernel performance (#2705)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+11/-5); docker/Dockerfile.rocm (+1/-1)
BODY: ## Motivation ⏎  ⏎ torch.sum could not use GPU core efficiency, implement specific kernel to enhance the performance ⏎  ⏎ ## Modifications ⏎  ⏎ change the base docker image and modify torch.sum to ops.moe_sum in fused_moe.py ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contributor_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](http …[truncated]

### L1-bdf946bf81  (L1, 2025-01-02, sha bdf946bf8101, PR #2716)
TITLE: Support loading pre-sharded moe weights (#2716)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+14/-6); python/sglang/srt/models/grok.py (+97/-26)
BODY: 

### L1-c7ae474a49  (L1, 2025-01-02, sha c7ae474a49f9, PR #2601)
TITLE: [Feature, Hardware] Enable DeepseekV3 on AMD GPUs (#2601)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+5/-5); python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+4/-0)
LABELS: bug, high priority, amd
BODY: ## Motivation ⏎ - Support DeepseekV3 on AMD Instinct MI300X GPU ⏎  ⏎ ## Modifications ⏎ - Add proper fix for AMD FP8 ```e4m3fnuz``` to support DeepseekV3 FP8 model ⏎ - Bypass ```FlashInfer backend bmm_fp8``` to cast FP8 to BF16 in MLA ⏎ - Add AMD ```triton stages``` config ⏎ ### TODO ⏎  ⏎  ⏎  ⏎ ## How to run ⏎ **build env** ⏎ ``` ⏎ cd sglang/docker ⏎  ⏎ docker build –t sglang-rocm:latest –f Dockerfile.rocm . ⏎   ⏎ docker run -it --ipc=host \  ⏎                --cap- …[truncated]

### L1-ba5112ff69  (L1, 2025-01-02, sha ba5112ff691d, PR #2712)
TITLE: feat: support moe_align_block_size_triton (#2712)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+177/-25)
BODY: ## Motivation ⏎  ⏎ ``` ⏎ # enable Triton implementation ⏎ export ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON=1 ⏎ ``` ⏎  ⏎ batch size 1/8/32, input/output 128/256, **around 20% improvement for online cases** ⏎  ⏎ TODO @BBuf will continue to optimize the CUDA version. ⏎  ⏎ ``` ⏎ Prefill. latency: 2.03397 s, throughput:     62.93 token/s ⏎ Decode.  latency: 2.15392 s, throughput:      0.46 token/s ⏎ Decode.  latency: 0.02824 s, throughput:     35.41 token/s ⏎ Decode.  late …[truncated]

### L1-ded9fcd09a  (L1, 2025-01-06, sha ded9fcd09a43, PR #2735)
TITLE: improve moe_align_kernel for deepseek v3 (#2735)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+32/-50); sgl-kernel/tests/test_trt_reduce.py (+0/-1)
BODY: In H200 ⏎  ⏎ ```shell ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code ⏎  ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 ⏎ ``` ⏎  ⏎ main branch: ⏎  ⏎ ```shell ⏎ Accuracy: 0.951 ⏎ Invalid: 0.000 ⏎ Latency: 83.593 s ⏎ Output throughput: 1676.993 token/s ⏎ ``` ⏎  ⏎ pr: ⏎  ⏎ ```shell ⏎ 0.952 ⏎ Invalid: 0.000 ⏎ Latency: 78.651 s ⏎ Output throughput: 1782.633 token/s ⏎ ``` ⏎  ⏎ I gain idea …[truncated]

### L1-2f0d386496  (L1, 2025-01-06, sha 2f0d38649623, PR #2713)
TITLE: chore: bump v0.4.1.post4 (#2713)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile.rocm (+1/-1); python/pyproject.toml (+2/-2); docs/developer/setup_github_runner.md (+2/-2); docs/start/install.md (+5/-5); python/sglang/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ - EAGLE-2 ⏎ - DeepSeek V3 36 tokens/s ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-06dd2eab84  (L1, 2025-01-06, sha 06dd2eab8438, PR #2751)
TITLE: Remove unused var in moe_align_kernel (#2751)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+0/-1)
BODY: ## Motivation ⏎  ⏎ Fix warning `variable "lane_id" was declared but never referenced`.

### L1-0f3eb1d294  (L1, 2025-01-06, sha 0f3eb1d29404, PR #2752)
TITLE: Support cutlass Int8 gemm (#2752)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/benchmark/bench_int8_gemm.py (+55/-0); sgl-kernel/setup.py (+2/-0); sgl-kernel/src/sgl-kernel/__init__.py (+2/-0); sgl-kernel/src/sgl-kernel/csrc/cutlass_extensions/epilogue/epilogue_per_row_per_col_scale.h (+278/-0); sgl-kernel/src/sgl-kernel/csrc/cutlass_extensions/gemm/gemm_universal_base_compat.h (+346/-0); sgl-kernel/src/sgl-kernel/csrc/cutlass_extensions/gemm/gemm_with_epilogue_visitor.h (+456/-0); sgl-kernel/src/sgl-kernel/csrc/int8_gemm_kernel.cu (+209/-0); sgl-kernel/src/sgl-kernel/csrc/sgl_kernel_ops.cu (+7/-0); sgl-kernel/src/sgl-kernel/csrc/utils.hpp (+10/-0); (+2 more)
BODY: ## Motivation ⏎  ⏎ Support fused int8 gemm for W8A8 quantization. ⏎  ⏎ Tested on A100 with benchmark script `benchmark/bench_int8_gemm.py` (measured with GB/s): ⏎ N = 4096, K = 8192 ⏎ ``` ⏎    batch_size  vllm int8 gemm  sgl-kernel int8 gemm ⏎ 0         1.0     1502.728206           1604.403733 ⏎ 1        16.0    24009.266126          26119.766434 ⏎ 2        32.0    48732.887699          51184.391083 ⏎ 3        64.0    91677.707776          94582.986851 ⏎ 4  …[truncated]

### L1-bdc1acf6cd  (L1, 2025-01-07, sha bdc1acf6cdad, PR #2761)
TITLE: Misc fix for min_p_sampling, --cuda-graph-bs (#2761)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+9/-3); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+16/-5); .github/workflows/pr-test.yml (+3/-1); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+1/-0); python/sglang/bench_serving.py (+4/-1); python/sglang/srt/layers/logits_processor.py (+5/-0); python/sglang/srt/layers/quantization/__init__.py (+1/-2); python/sglang/srt/managers/data_parallel_controller.py (+2/-0); python/sglang/srt/managers/scheduler.py (+3/-2); python/sglang/srt/metrics/collector.py (+22/-30); (+7 more)
BODY: - Support `--cuda-graph-bs` so you can specify the batch size to capture the cuda graph. ⏎ - Fix merge_batch for min_p_sampling ⏎ - Add more utility functions and remove redundant code.

### L1-380930a959  (L1, 2025-01-07, sha 380930a959ac, PR #2767)
TITLE: add  benchmark_moe_align_blocks (#2767)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/benchmark_moe_align_blocks.py (+289/-0)
BODY: h100: ⏎  ⏎ ```shell ⏎ python3 /opt/dlami/nvme/bbuf/sglang/benchmark/kernels/fused_moe_triton/benchmark_moe_align_blocks.py --save_path /opt/dlami/nvme/bbuf/configs ⏎ ✅ CUDA and Triton implementations match ⏎ moe-align-block-size-performance: ⏎      batch_size  seq_len         CUDA       Triton ⏎ 0           1.0      1.0    24.224000    73.408000 ⏎ 1           1.0      2.0    24.192000    72.127998 ⏎ 2           1.0      4.0    24.256000    71.648002 ⏎ 3    …[truncated]

### L1-8a6906127a  (L1, 2025-01-07, sha 8a6906127a81, PR #2784)
TITLE: Improve linear.py to load sharded weights & remove the dependency of Parameters from vllm (#2784)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-3); 3rdparty/amd/tuning/benchmark_moe_rocm.py (+4/-1); python/sglang/srt/layers/attention/__init__.py (+8/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+4/-2); python/sglang/srt/layers/linear.py (+165/-57); python/sglang/srt/layers/parameter.py (+431/-0); python/sglang/srt/layers/quantization/fp8.py (+1/-1); python/sglang/srt/layers/vocab_parallel_embedding.py (+1/-1); python/sglang/srt/managers/session_controller.py (+1/-1); python/sglang/srt/model_executor/forward_batch_info.py (+3/-0); (+5 more)
BODY: - remove the dependency of Parameters from vllm ⏎ - improve the weight loading of linear.py

### L1-e2b16c4716  (L1, 2025-01-12, sha e2b16c4716f2, PR #2846)
TITLE: add sampling_scaling_penalties kernel (#2846)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/setup.py (+1/-0); sgl-kernel/src/sgl-kernel/__init__.py (+2/-0); sgl-kernel/src/sgl-kernel/csrc/sampling_scaling_penalties.cu (+64/-0); sgl-kernel/src/sgl-kernel/csrc/sgl_kernel_ops.cu (+5/-0); sgl-kernel/src/sgl-kernel/csrc/vectorization.cuh (+30/-0); sgl-kernel/src/sgl-kernel/ops/__init__.py (+7/-0); sgl-kernel/tests/test_sampling_scaling_penalties.py (+39/-0)
BODY: Add a sampling_scaling_penalties kernel to reduce the GPU memory usage during the sampling phase when the scaling_penalties sampling parameter is applied (https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/sampling/sampling_batch_info.py#L247). This can eliminate the need for storing multiple intermediate logits in memory, which is particularly useful when serving long sequences or handling large batch sizes on GPUs with limited me …[truncated]

### L1-72c7776355  (L1, 2025-01-13, sha 72c777635593, PR #2851)
TITLE: Fix linear.py and improve weight loading (#2851)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+4/-2); benchmark/deepseek_v3/README.md (+4/-3); docs/references/supported_models.md (+1/-1); python/sglang/srt/layers/linear.py (+36/-98); python/sglang/srt/layers/parameter.py (+24/-16); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-1); python/sglang/srt/layers/vocab_parallel_embedding.py (+15/-2); python/sglang/srt/managers/scheduler.py (+4/-0); python/sglang/srt/mem_cache/memory_pool.py (+19/-0); (+2 more)
BODY: 

### L1-42f3909963  (L1, 2025-01-13, sha 42f390996317, PR #2856)
TITLE: Unify sglang coding style (#2856)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-4); python/sglang/srt/layers/quantization/fp8.py (+15/-14)
BODY: ## Motivation ⏎  ⏎ From #2854 suggestion, unify the coding style in #2854 changed files ⏎  ⏎ ## Modifications ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contribution_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contribution_guide.md). ⏎ - [+] Update documentation  …[truncated]

### L1-6249e4a19e  (L1, 2025-01-13, sha 6249e4a19ed6, PR #2866)
TITLE: Revert "Integration of TurboMind AWQ" (#2866)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/configs/model_config.py (+1/-9); python/sglang/srt/layers/linear.py (+0/-1); python/sglang/srt/layers/quantization/__init__.py (+0/-2); python/sglang/srt/layers/quantization/awq_turbomind.py (+0/-287); python/sglang/srt/layers/quantization/turbomind_utils.py (+0/-63); python/sglang/srt/server_args.py (+0/-1); test/srt/test_turbomind_awq.py (+0/-47)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 2828 reason=build_or_dependency
BODY: Reverts sgl-project/sglang#2828 because it introduces extra dependency and breaks some internal stuff

### L1-e808c1df3e  (L1, 2025-01-13, sha e808c1df3e04, PR #2854)
TITLE: Integrate ROCm ater package for ck moe function feasibility (#2854)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+36/-9); docker/Dockerfile.rocm (+9/-0); python/sglang/srt/layers/quantization/fp8.py (+98/-45); python/sglang/srt/utils.py (+19/-0)
BODY: ## Motivation ⏎  ⏎ ROCm platform has moe function implementation from ck module, it fused two gemm and activation kernel into one to decrease the launched overhead. ⏎  ⏎ ## Modifications ⏎  ⏎ Major modification in the layer.py of fused_moe ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Contributor Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/contribution_guide.md). ⏎ - [+] Add unit tests as outlined in the [Contrib …[truncated]

### L1-85b2e05770  (L1, 2025-01-13, sha 85b2e05770ea, PR #2848)
TITLE: Add int8 quant kernel (#2848)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: benchmark/kernels/quantization/bench_int8_quant.py (+94/-0); python/sglang/srt/layers/quantization/int8_kernel.py (+53/-0)
BODY: ## Motivation ⏎  ⏎ Add int8 per token quant kernel for w8a8 quantization. ⏎  ⏎ Kernel running time (ms) benchmark: ⏎ ``` ⏎    batch_size   vllm op    triton  torch.compile ⏎ 0         1.0  0.019568  0.010560       0.018944 ⏎ 1        16.0  0.018016  0.009280       0.030048 ⏎ 2        32.0  0.019008  0.010304       0.030304 ⏎ 3        64.0  0.019616  0.011744       0.029696 ⏎ 4       128.0  0.022624  0.014368       0.031808 ⏎ 5       256.0  0.033536  0.018464 …[truncated]

### L1-17de02f98d  (L1, 2025-01-13, sha 17de02f98d8f, PR #2828)
TITLE: Integration of TurboMind AWQ (#2828)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/configs/model_config.py (+9/-1); python/sglang/srt/layers/linear.py (+1/-0); python/sglang/srt/layers/quantization/__init__.py (+2/-0); python/sglang/srt/layers/quantization/awq_turbomind.py (+287/-0); python/sglang/srt/layers/quantization/turbomind_utils.py (+63/-0); python/sglang/srt/server_args.py (+1/-0); test/srt/test_turbomind_awq.py (+47/-0)
LABELS: enhancement, high priority, quant
DEEP_STUDY: deep-study: this PR was reverted by PR 2866 (confirmed_revert, reason=build_or_dependency)
BODY: ## Motivation ⏎  ⏎ Come from this [issue](https://github.com/sgl-project/sglang/issues/2788) ⏎  ⏎ ## Usage ⏎ ``` ⏎ # use turbomind ⏎ python examples/runtime/engine/offline_batch_inference.py --model=${Meta-Llama-3-8B-Instruct-hf-AWQ} --quantization=awq_turbomind ⏎  ⏎ # use marlin ⏎ python examples/runtime/engine/offline_batch_inference.py --model=${Meta-Llama-3-8B-Instruct-hf-AWQ} ⏎ ``` ⏎  ⏎ ## Checklist

### L1-d08c77c434  (L1, 2025-01-13, sha d08c77c43498, PR #2870)
TITLE: Sampling penalties memory interface (#2870)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); benchmark/kernels/fused_moe_triton/benchmark_deepseekv3_moe_align_blocks.py (+2/-1); python/sglang/srt/sampling/penaltylib/penalizers/repetition_penalty.py (+15/-5); python/sglang/srt/sampling/sampling_batch_info.py (+14/-5); python/sglang/srt/utils.py (+4/-0); sgl-kernel/benchmark/benchmark_sampling_scaling_penalties.py (+159/-0); sgl-kernel/tests/test_moe_align.py (+61/-34)
BODY: Related [pr](https://github.com/sgl-project/sglang/pull/2846) ⏎  ⏎ Result of `benchmark_sampling_scaling_penalties.py` : ⏎  ⏎ performace: ⏎  ⏎ ![图片](https://github.com/user-attachments/assets/122d33ee-bcd8-426b-9357-dc76f9f7f903) ⏎  ⏎ peak memory: ⏎  ⏎ ![图片](https://github.com/user-attachments/assets/72dc772d-8dce-4ec3-b414-2d4d22b5c1a6) ⏎  ⏎ - Fix a bug in `benchmark_moe_align_blocks.py` . ⏎ - Refine sgl-kernel `test_moe_align.py` .

### L1-f005758f2b  (L1, 2025-01-14, sha f005758f2bcf, PR #2887)
TITLE: introduce CUB in sgl-kernel (#2887)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+3/-0); sgl-kernel/CMakeLists.txt (+2/-0); sgl-kernel/3rdparty/cub (+1/-0)
BODY: ![图片](https://github.com/user-attachments/assets/89bde132-d625-4f8a-adf0-df86fe3b2f2a) ⏎  ⏎ @zhyncs

### L1-767c9dec03  (L1, 2025-01-16, sha 767c9dec03e3, PR #2511)
TITLE: adapt custom allreduce for tensorrt llm (#2511)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/_custom_ops.py (+22/-27); python/sglang/srt/distributed/device_communicators/custom_all_reduce.py (+53/-39); test/srt/run_suite.py (+1/-0); test/srt/test_custom_allreduce.py (+164/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ adapt for tensorrt llm custom allreduce, currently still use vllm distributed.  ⏎ After this pr is merged and sgl-kernel is stable, we only need replace vllm.distribued to sglang.srt.distributed, and add a monkey patch, then we can remove vllm distributed ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-7596417732  (L1, 2025-01-16, sha 75964177327c, PR #2919)
TITLE: minor: use bear for compilation database (#2919)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+0/-65); sgl-kernel/Makefile (+8/-5)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-63051738a9  (L1, 2025-01-16, sha 63051738a91e, PR #2806)
TITLE: Enable CPU device on SGLang (#2806)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/sglang/srt/layers/moe/fused_moe_native.py (+69/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+4/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+26/-2); python/pyproject.toml (+6/-0); python/sglang/srt/configs/device_config.py (+1/-1); python/sglang/srt/layers/rotary_embedding.py (+248/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/managers/scheduler.py (+2/-0); python/sglang/srt/managers/tp_worker_overlap_thread.py (+2/-0); python/sglang/srt/model_executor/model_runner.py (+9/-3); (+3 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR enables CPU device on SGLang. ⏎ Currently we fallback attention and MoE to the torch native backend and make the functionality work on CPU. ⏎ We will submit follow-up PRs to provide optimized kernels to further improvement the performance. ⏎  ⏎ For vllm installation for CPU, users could follow the instruction provided by vllm [here](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/index.html). ⏎  ⏎ ## Modific …[truncated]

### L1-5dc54f1a62  (L1, 2025-01-17, sha 5dc54f1a627a, PR #2907)
TITLE: feat: remove vllm distributed (#2907)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-4); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-3); python/sglang/srt/layers/activation.py (+3/-3); python/sglang/srt/layers/dp_attention.py (+2/-1); python/sglang/srt/layers/linear.py (+2/-2); python/sglang/srt/layers/logits_processor.py (+2/-2); python/sglang/srt/layers/parameter.py (+2/-1); python/sglang/srt/layers/quantization/fp8.py (+1/-1); python/sglang/srt/layers/vocab_parallel_embedding.py (+2/-2); python/sglang/srt/model_executor/cuda_graph_runner.py (+2/-2); (+35 more)
BODY: ## Motivation ⏎  ⏎ ref https://github.com/sgl-project/sglang/actions/runs/12797036847/job/35678064629 ⏎  ⏎ cc @yizhang2077  ⏎  ⏎ ``` ⏎ Benchmark offline throughput (w/ FP8) ⏎ Run cd test/srt ⏎ [2025-0[1](https://github.com/sgl-project/sglang/actions/runs/12797036847/job/35678064160#step:6:1)-15 21:16:58] server_args=ServerArgs(model_path='neuralmagic/Meta-Llama-3.1-8B-FP8', tokenizer_path='neuralmagic/Meta-Llama-3.1-8B-FP8', tokenizer_mode='auto', load_fo …[truncated]

### L1-033c715b46  (L1, 2025-01-17, sha 033c715b4662, PR #2948)
TITLE: cleanup models dependencies 1/n (#2948)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-1); python/sglang/srt/lora/lora.py (+1/-9); python/sglang/srt/models/baichuan.py (+5/-5); python/sglang/srt/models/gpt2.py (+1/-2); python/sglang/srt/models/minicpm3.py (+6/-6); python/sglang/srt/models/olmo2.py (+1/-1); python/sglang/srt/models/olmoe.py (+5/-6); python/sglang/srt/models/qwen2_vl.py (+2/-2); python/sglang/srt/models/xverse.py (+6/-6); python/sglang/srt/models/xverse_moe.py (+8/-8)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d33cbb7e58  (L1, 2025-01-19, sha d33cbb7e5857, PR #2976)
TITLE: remove cub and add cccl (#2976)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+3/-3); sgl-kernel/3rdparty/cccl (+1/-0); sgl-kernel/3rdparty/cub (+0/-1)
BODY: ## Motivation ⏎  ⏎ The cub repo has been archived, use cccl instead. cc @BBuf  ⏎  ⏎ ``` ⏎ sglang git:(zhyncs/sub) git submodule status ⏎  b5fe509fd11a925f90d6495176707cc1184eed9d sgl-kernel/3rdparty/cccl (v2.7.0) ⏎  b78588d1630aa6643bf021613717bafb705df4ef sgl-kernel/3rdparty/cutlass (v3.7.0) ⏎ ``` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6ada05d0ed  (L1, 2025-01-19, sha 6ada05d0ed52, PR #2984)
TITLE: feat: check for is_cuda for sgl_kernel import (#2984)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+10/-10)
BODY: ## Motivation ⏎  ⏎ cc @HaiShaw @chunyuan-w  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2c05f81f15  (L1, 2025-01-20, sha 2c05f81f157f, PR #2988)
TITLE: fix custom op version compatibility (#2988)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/rotary_embedding.py (+3/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0311ce8e1c  (L1, 2025-01-20, sha 0311ce8e1ccd, PR #3016)
TITLE: [router] Expose worker startup secs & Return error instead of panic for router init (#3016)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+3/-0); sgl-router/py_src/sglang_router/launch_router.py (+14/-6); sgl-router/py_src/sglang_router/launch_server.py (+27/-5); sgl-router/py_test/test_launch_router.py (+1/-0); sgl-router/src/lib.rs (+14/-6); sgl-router/src/router.rs (+43/-11); sgl-router/src/server.rs (+22/-19)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ From https://github.com/sgl-project/sglang/issues/2778, the worker startup secs should be adjustable ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ 1. Expose worker startup secs ⏎ 2. Also improve error handling. To be specific, originally we panic for error in router init, but py is not able to catch panic error. Therefore, I switched to return stderr to make py side able to catch and handle  ⏎  ⏎ ## Checklist

### L1-3a8428ecaa  (L1, 2025-01-20, sha 3a8428ecaa63, PR #3019)
TITLE: [router] Expose worker startup interval (#3019)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+3/-0); sgl-router/py_src/sglang_router/launch_router.py (+11/-0); sgl-router/py_test/test_launch_router.py (+1/-0); sgl-router/src/lib.rs (+7/-0); sgl-router/src/router.rs (+50/-13)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/issues/2778 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-10bfce71b3  (L1, 2025-01-20, sha 10bfce71b353, PR #3003)
TITLE: fix moe align blocks benchmark (#3003)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/benchmark_deepseekv3_moe_align_blocks.py (+30/-6)
BODY: ## Motivation ⏎  ⏎ suggested in https://github.com/sgl-project/sglang/pull/2970 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ benchmark_deepseekv3_moe_align_blocks.py ⏎  ⏎ - correct topk inputs ⏎  ⏎ NOTE the correction is disabled by default in benchmark tests (but enabled in **calculate_diff**) since it will be much longer for seq_len>16000 when correction used. ⏎  ⏎ ## Checklist

### L1-5a0d680a14  (L1, 2025-01-21, sha 5a0d680a14fc, PR #3033)
TITLE: feat: add flashinfer as 3rdparty and use rmsnorm as example (#3033)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+3/-0); .github/workflows/pr-test-sgl-kernel.yml (+1/-0); .gitignore (+2/-0); sgl-kernel/3rdparty/flashinfer (+1/-0); sgl-kernel/THIRDPARTYNOTICES.txt (+225/-0); sgl-kernel/setup.py (+19/-2); sgl-kernel/src/sgl-kernel/__init__.py (+2/-0); sgl-kernel/src/sgl-kernel/csrc/norm.cu (+28/-0); sgl-kernel/src/sgl-kernel/csrc/sgl_kernel_ops.cu (+5/-0); sgl-kernel/src/sgl-kernel/ops/__init__.py (+18/-0); (+1 more)
BODY: ## Motivation ⏎  ⏎ ``` ⏎ sglang git:(zhyncs/flashinfer) git submodule status ⏎  b5fe509fd11a925f90d6495176707cc1184eed9d sgl-kernel/3rdparty/cccl (v2.7.0) ⏎  b78588d1630aa6643bf021613717bafb705df4ef sgl-kernel/3rdparty/cutlass (v3.7.0) ⏎  a0e99a3a820109763d9a757138a5cdf7bbcd1f85 sgl-kernel/3rdparty/flashinfer (v0.0.2-422-ga0e99a3) ⏎ ``` ⏎  ⏎ ``` ⏎ sgl-kernel git:(zhyncs/flashinfer) pytest tests/test_rmsnorm.py ⏎ ============================================= …[truncated]

### L1-b2bd8f444c  (L1, 2025-01-22, sha b2bd8f444c61, PR #3054)
TITLE: minor: update header and use pytest (#3054)
SOURCES: path_core
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+1/-1); sgl-kernel/Makefile (+1/-1); sgl-kernel/src/sgl-kernel/csrc/int8_gemm_kernel.cu (+1/-1); sgl-kernel/src/sgl-kernel/csrc/sampling_scaling_penalties.cu (+1/-1); sgl-kernel/src/sgl-kernel/csrc/sgl_kernel_ops.cu (+1/-1); sgl-kernel/src/sgl-kernel/csrc/trt_reduce_internal.cuh (+1/-1); sgl-kernel/src/sgl-kernel/csrc/utils.h (+0/-0)
BODY: ## Motivation ⏎  ⏎ `make test` passed locally ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-153b414e83  (L1, 2025-01-24, sha 153b414e835e, PR #3105)
TITLE: minor: sync flashinfer and add turbomind as 3rdparty (#3105)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: .gitmodules (+3/-0); sgl-kernel/3rdparty/flashinfer (+1/-1); sgl-kernel/3rdparty/turbomind (+1/-0); sgl-kernel/developer_guide.md (+1/-0)
BODY: ## Motivation ⏎  ⏎ cc @lzhangzz @lvhan028 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-95f789adb0  (L1, 2025-01-26, sha 95f789adb0d6, PR #3143)
TITLE: minor: cleanup sgl-kernel (#3143)
SOURCES: path_core
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+1/-16); sgl-kernel/developer_guide.md (+4/-0); sgl-kernel/setup.py (+0/-2); sgl-kernel/src/sgl-kernel/csrc/fused_add_rms_norm.cu (+0/-92); sgl-kernel/src/sgl-kernel/csrc/lightning_attention_decode_kernel.cu (+1/-2); sgl-kernel/src/sgl-kernel/csrc/sampling_scaling_penalties.cu (+0/-61); sgl-kernel/src/sgl-kernel/csrc/trt_reduce_internal.cu (+1/-0); sgl-kernel/src/sgl-kernel/csrc/trt_reduce_kernel.cu (+1/-0); sgl-kernel/src/sgl-kernel/include/sgl_kernels_ops.h (+1/-5); sgl-kernel/src/sgl-kernel/include/trt_reduce_internal.cuh (+1/-2); (+3 more)
BODY: ## Motivation ⏎  ⏎ - remove unused kernels ⏎ - reorg the code structure ⏎  ⏎ ref https://github.com/sgl-project/sglang/pull/3133 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-7e0976133c  (L1, 2025-01-26, sha 7e0976133ca4, PR #3150)
TITLE: udpate sgl-kernel version for srt (#3150)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-52c03f16b9  (L1, 2025-01-27, sha 52c03f16b914, PR #3170)
TITLE: Add activation parameters to fused_moe (#3170)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-0); python/sglang/srt/layers/moe/fused_moe_native.py (+17/-3); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+18/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+9/-0); python/sglang/srt/layers/quantization/fp8.py (+4/-1); python/sglang/srt/models/grok.py (+1/-0); test/srt/test_fp8_kernel.py (+0/-2)
BODY: This is because grok uses gelu while other models use silu.

### L1-53cef81587  (L1, 2025-01-27, sha 53cef81587de, PR #3174)
TITLE: Improve weight loading and code style (#3174)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+17/-12); python/sglang/srt/layers/linear.py (+24/-9); python/sglang/srt/layers/parameter.py (+16/-7); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/managers/scheduler.py (+11/-5); python/sglang/srt/model_executor/model_runner.py (+1/-1); python/sglang/srt/model_loader/weight_utils.py (+14/-4); python/sglang/srt/server_args.py (+6/-0); python/sglang/srt/utils.py (+1/-1); python/sglang/test/test_utils.py (+76/-22); (+1 more)
BODY: 

### L1-351a72d40b  (L1, 2025-01-27, sha 351a72d40bee, PR #3146)
TITLE: add dsv3 mi300 triton config for block scale (#3146)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=256,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+43/-11); python/sglang/srt/layers/quantization/configs/N=1536,K=7168,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/quantization/configs/N=3072,K=1536,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/quantization/configs/N=4096,K=512,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/quantization/configs/N=4608,K=7168,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/quantization/configs/N=512,K=7168,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/quantization/configs/N=576,K=7168,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=2048,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=2304,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); (+1 more)
BODY: ## Motivation ⏎ **Tune MoE:** ⏎ ``` ⏎ python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py --model deepseek-ai/DeepSeek-V3 --tp-size 8 --dtype fp8_w8a8 --tune ⏎ ``` ⏎  ⏎ The offline perf would get a 20% uplift with configs. server perf no obvious change. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2f79f58873  (L1, 2025-01-27, sha 2f79f58873b0, PR #3179)
TITLE: feat: use sgl-kernel 0.0.3 in sglang (#3179)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/activation.py (+5/-5); python/sglang/srt/layers/layernorm.py (+5/-5); python/sglang/srt/layers/sampler.py (+4/-8); python/sglang/srt/models/deepseek_v2.py (+3/-3); python/sglang/srt/models/minicpm3.py (+3/-3)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-cf142b6eb8  (L1, 2025-01-27, sha cf142b6eb87a, PR #3181)
TITLE: fix: update Dockerfile for cu118 (#3181)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+3/-0)
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/actions/runs/12992119766 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-222ce6f1da  (L1, 2025-01-30, sha 222ce6f1da31, PR #3216)
TITLE: add tensorrt_llm common and cutlass_extensions as 3rdparty (#3216)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: sgl-kernel/3rdparty/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/collective/epilogue_moe_finalize.hpp (+550/-0); .clang-format-ignore (+1/-0); sgl-kernel/3rdparty/tensorrt_llm/common/CMakeLists.txt (+22/-0); sgl-kernel/3rdparty/tensorrt_llm/common/assert.cpp (+34/-0); sgl-kernel/3rdparty/tensorrt_llm/common/cublasMMWrapper.cpp (+360/-0); sgl-kernel/3rdparty/tensorrt_llm/common/cublasMMWrapper.h (+148/-0); sgl-kernel/3rdparty/tensorrt_llm/common/cublasVersionCheck.h (+35/-0); sgl-kernel/3rdparty/tensorrt_llm/common/cudaBf16Fallbacks.cuh (+313/-0); sgl-kernel/3rdparty/tensorrt_llm/common/cudaDriverWrapper.cpp (+187/-0); sgl-kernel/3rdparty/tensorrt_llm/common/cudaDriverWrapper.h (+138/-0); (+76 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e81d7f11de  (L1, 2025-01-30, sha e81d7f11dede, PR #3217)
TITLE: add tensorrt_llm moe_gemm as 3rdparty (#3217)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: sgl-kernel/3rdparty/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.h (+25/-0); sgl-kernel/3rdparty/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl (+96/-0); sgl-kernel/3rdparty/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_launcher_sm90.h (+37/-0); sgl-kernel/3rdparty/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_launcher_sm90.inl (+348/-0); sgl-kernel/3rdparty/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_hopper_input.cu (+131/-0); sgl-kernel/3rdparty/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h (+230/-0); sgl-kernel/3rdparty/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_bf16.cu (+24/-0); sgl-kernel/3rdparty/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_uint4.cu (+24/-0); sgl-kernel/3rdparty/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_uint8.cu (+24/-0); sgl-kernel/3rdparty/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_fp16.cu (+22/-0); (+10 more)
BODY: ## Motivation ⏎  ⏎ ref https://github.com/sgl-project/sglang/pull/3216 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3ee62235c6  (L1, 2025-01-31, sha 3ee62235c612, PR #3230)
TITLE: revert the MoE dependence (#3230)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: sgl-kernel/3rdparty/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/collective/epilogue_moe_finalize.hpp (+0/-550); sgl-kernel/3rdparty/tensorrt_llm/common/assert.cpp (+0/-34); sgl-kernel/3rdparty/tensorrt_llm/common/assert.h (+0/-92); sgl-kernel/3rdparty/tensorrt_llm/common/cublasMMWrapper.cpp (+0/-360); sgl-kernel/3rdparty/tensorrt_llm/common/cublasMMWrapper.h (+0/-148); sgl-kernel/3rdparty/tensorrt_llm/common/cublasVersionCheck.h (+0/-35); sgl-kernel/3rdparty/tensorrt_llm/common/cudaBf16Fallbacks.cuh (+0/-313); sgl-kernel/3rdparty/tensorrt_llm/common/cudaBf16Wrapper.h (+0/-21); sgl-kernel/3rdparty/tensorrt_llm/common/cudaDriverWrapper.cpp (+0/-187); sgl-kernel/3rdparty/tensorrt_llm/common/cudaDriverWrapper.h (+0/-138); (+84 more)
DEEP_STUDY: deep-study revert record: explicit_rollback of PR(s)  reason=build_or_dependency
BODY: ## Motivation ⏎  ⏎ follow-up: Add the CUDA 12.5 runtime Docker image and use the cuBLAS Grouped GEMM API. ⏎ https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/ ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b49d6d0fee  (L1, 2025-01-31, sha b49d6d0fee3c, PR #3231)
TITLE: support 12.5 CUDA runtime (#3231)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+6/-0); .github/workflows/release-docker.yml (+4/-2)
BODY: ## Motivation ⏎  ⏎ follow-up for https://github.com/sgl-project/sglang/pull/3230 ⏎  ⏎ https://github.com/sgl-project/sglang/actions/runs/13072042144 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1ebe1d6de5  (L1, 2025-02-01, sha 1ebe1d6de5e0, PR #3236)
TITLE: Optimize MoE topk with torch compile (#3236)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+4/-0)
BODY: ## Motivation ⏎  ⏎ 2 token/s faster by applying torch compile for topk function by default. ⏎  ⏎ ``` ⏎ python3 -m sglang.bench_one_batch --batch-size 1 --input 128 --output 256 --model deepseek-ai/DeepSeek-V3 --trust-remote-code --tp 8 ⏎  ⏎ main branch: ⏎ Prefill. latency: 1.82421 s, throughput:     70.17 token/s ⏎ Decode.  latency: 1.75760 s, throughput:      0.57 token/s ⏎ Decode.  latency: 0.02740 s, throughput:     36.49 token/s ⏎ Decode.  latency: 0.02 …[truncated]

### L1-34e405e01f  (L1, 2025-02-01, sha 34e405e01f7f, PR #3238)
TITLE: update sgl-kernel version for sglang (#3238)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4eb4b401cc  (L1, 2025-02-01, sha 4eb4b401cc55, PR #3249)
TITLE: update and simplify CustomOp (#3249)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-3); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-3); python/sglang/srt/custom_op.py (+40/-0); python/sglang/srt/layers/activation.py (+1/-5); python/sglang/srt/layers/custom_op_util.py (+0/-25); python/sglang/srt/layers/layernorm.py (+1/-5); python/sglang/srt/layers/rotary_embedding.py (+1/-3); python/sglang/srt/model_executor/cuda_graph_runner.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-566d61d90f  (L1, 2025-02-03, sha 566d61d90fd5, PR #3259)
TITLE: ROCm: bump 6.3.0 (#3259)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile.rocm (+2/-2); python/pyproject.toml (+8/-10); .github/workflows/release-docker-amd.yml (+3/-3); docs/developer/setup_github_runner.md (+2/-2); docs/start/install.md (+3/-3); python/sglang/srt/constrained/outlines_backend.py (+9/-1); python/sglang/srt/custom_op.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ Latest ROCm ⏎  ⏎ ## Modifications ⏎  ⏎ As it is. ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Code Formatting with Pre-Commit](https://docs.sglang.ai/references/contribution_guide.html#code-formatting-with-pre-commit). ⏎ - [+] Add unit tests as outlined in the [Running Unit Tests](https://docs.sglang.ai/references/contribution_guide.html#running-unit-tests-adding-to-ci). ⏎ - [+] Update documentation / docstrings / exampl …[truncated]

### L1-3c8ac78dc1  (L1, 2025-02-03, sha 3c8ac78dc143, PR #3268)
TITLE: optimize test_fused_moe style (#3268)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: test/srt/test_fused_moe.py (+90/-39)
BODY: Modify the method of generating Tensors in test_fused_moe to allow for higher precision when comparing with the baseline accuracy, and additionally, add a progress indicator for testing.

### L1-00fa7d0417  (L1, 2025-02-03, sha 00fa7d0417bf, PR #3270)
TITLE: add copyright for sgl-kernel (#3270)
SOURCES: path_core
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+15/-0); sgl-kernel/setup.py (+15/-0); sgl-kernel/src/sgl-kernel/csrc/cutlass_extensions/epilogue/epilogue_per_row_per_col_scale.h (+15/-0); sgl-kernel/src/sgl-kernel/csrc/cutlass_extensions/gemm/gemm_universal_base_compat.h (+15/-0); sgl-kernel/src/sgl-kernel/csrc/cutlass_extensions/gemm/gemm_with_epilogue_visitor.h (+15/-0); sgl-kernel/src/sgl-kernel/csrc/fp8_gemm_kernel.cu (+15/-0); sgl-kernel/src/sgl-kernel/csrc/fused_add_rms_norm_kernel.cu (+15/-0); sgl-kernel/src/sgl-kernel/csrc/int8_gemm_kernel.cu (+15/-0); sgl-kernel/src/sgl-kernel/csrc/lightning_attention_decode_kernel.cu (+15/-0); sgl-kernel/src/sgl-kernel/csrc/trt_reduce_internal.cu (+15/-0); (+6 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d54cee1441  (L1, 2025-02-04, sha d54cee144175, PR #3272)
TITLE: adding Triton configs for DeepSeekV3 on Blackwell (#3272)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=24576,K=7168,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=3072,K=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=3072,K=7168,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=32768,K=512,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=4096,K=512,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=4608,K=7168,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=512,K=7168,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=576,K=7168,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/quantization/configs/N=7168,K=16384,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); (+4 more)
BODY: This will add Triton configs to run optimally with DeepSeekV3 on Blackwell. ⏎  ⏎  ⏎  ⏎ To achieve better perf on Blackwell machines. Below are a few kernel benchmarks: ⏎  ⏎  Shape                      | old (us) | new (us) ⏎  -------------------|---------|-------| ⏎ (16, 7168, 2048)     | 38.2       | 14.2    | ⏎ (64, 7168, 2048)    | 38.2       | 25.6    | ⏎ (256, 7168, 2048)  | 62.2       | 57.5    |  ⏎  ⏎  ⏎  ⏎  ⏎ Only adding the configs.

### L1-c7256ca836  (L1, 2025-02-04, sha c7256ca836cf, PR #3294)
TITLE: [ROCm] Add tuning configs for AMD Radeon Graphics. (#3294)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=256,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=14336,device_name=AMD_Radeon_Graphics.json (+200/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=1792,device_name=AMD_Radeon_Graphics.json (+200/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=3584,device_name=AMD_Radeon_Graphics.json (+200/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=4096,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8.json (+178/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=7168,device_name=AMD_Radeon_Graphics.json (+200/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=8192,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8.json (+175/-0); python/sglang/srt/layers/quantization/configs/N=1536,K=7168,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/quantization/configs/N=3072,K=1536,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/quantization/configs/N=4096,K=512,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); (+6 more)
BODY: ## Motivation ⏎  ⏎ Enlarge supported SKUs for AMD ROCm platform. ⏎  ⏎ ## Modifications ⏎  ⏎ Add GEMM and MoE tuning configurations for generic AMD Radeon Graphics SKUs.

### L1-d39899e85c  (L1, 2025-02-04, sha d39899e85c5c, PR #3288)
TITLE: upgrade flashinfer v0.2.0.post2 (#3288)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); .github/workflows/pr-test.yml (+8/-8); python/sglang/srt/entrypoints/engine.py (+2/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+20/-36); python/sglang/srt/speculative/eagle_utils.py (+5/-0); python/sglang/srt/speculative/eagle_worker.py (+2/-0); scripts/ci_install_dependency.sh (+4/-3); test/srt/run_suite.py (+0/-1)
BODY: ## Motivation ⏎  ⏎ - The flashinfer package name has been changed to flashinfer_python. cc @yzh119  ⏎ - Resolve the eagle-related issue. cc @pankajroark ⏎ - Temporarily disable the fp8 kv cache test (will be fixed in a follow-up) to facilitate the upgrade. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2c1a695ff1  (L1, 2025-02-04, sha 2c1a695ff111, PR #3287)
TITLE: ROCm: sgl-kernel enablement starting with sgl_moe_align_block (#3287)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile.rocm (+3/-0); python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+3/-11); docs/start/install.md (+3/-1); sgl-kernel/setup_rocm.py (+92/-0); sgl-kernel/src/sgl-kernel/torch_extension_rocm.cc (+29/-0)
BODY: ## Motivation ⏎  ⏎ 1. Enable sgl-kernel on ROCm, make it easy to add individual kernels ⏎ 2. Use sgl_moe_align_block kernel for fused_moe for performance ⏎  ⏎ ## Modifications ⏎  ⏎ As they are. ⏎  ⏎ ## Checklist ⏎  ⏎ - [+] Format your code according to the [Code Formatting with Pre-Commit](https://docs.sglang.ai/references/contribution_guide.html#code-formatting-with-pre-commit). ⏎ - [+] Add unit tests as outlined in the [Running Unit Tests](https://docs.sgl …[truncated]

### L1-6186a8f889  (L1, 2025-02-05, sha 6186a8f8897e, PR #3293)
TITLE: update flashinfer install index url (#3293)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+8/-8); docs/start/install.md (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-ad3499858e  (L1, 2025-02-06, sha ad3499858e34, PR #3332)
TITLE: clean moe align block kernel code and add acc test (#3332)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.align.cuda_aot
FILES: sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+14/-26); benchmark/kernels/fused_moe_triton/benchmark_deepseekv3_moe_align_blocks.py (+1/-1); sgl-kernel/src/sgl-kernel/include/utils.h (+12/-0); sgl-kernel/tests/test_moe_align.py (+214/-57)
BODY: 

### L1-cdae77b03d  (L1, 2025-02-07, sha cdae77b03dfc, PR #3347)
TITLE: optimize moe_align_kernel cuda (#3347)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.align.cuda_aot
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-2); sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+23/-15); benchmark/kernels/fused_moe_triton/benchmark_deepseekv3_moe_align_blocks.py (+4/-4)
BODY: Thanks to @tim-zou help in https://github.com/sgl-project/sglang/issues/3339. ⏎  ⏎ ## DeepSeek V3 end2end benchmark ⏎  ⏎ ```shell ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 ⏎ ``` ⏎  ⏎ After starting the service, I ran `benchmark/gsm8k/bench_sglang.py` twice. The results shown below are from the second run. ⏎  ⏎ main  …[truncated]

### L1-f287037673  (L1, 2025-02-07, sha f287037673a3, PR #3374)
TITLE: update sgl-kernel version (#3374)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-849f58d617  (L1, 2025-02-08, sha 849f58d617e7, PR #3346)
TITLE: Update fused_moe's benchmark (#3346)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py (+75/-22)
BODY: ## Motivation ⏎  ⏎ i try to run benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py with deepseek-v2 and deepseek-v3 model ,But found it doesn't work. and it get the wrong shape for config.  ⏎ Also, it doesn't pass block_shape to fused_moe to support blockwise-fp8  ⏎  ⏎ ## Modifications ⏎  ⏎ I fix some shape bugs and pass block_shape to fused_moe ⏎  ⏎ Since version 0.6.4.post1 of vllm does not have the block_shape parameter in  …[truncated]

### L1-64c8713573  (L1, 2025-02-10, sha 64c871357359, PR #3433)
TITLE: remove activation dependency in fused_moe (#3433)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+18/-3)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-85986bb978  (L1, 2025-02-10, sha 85986bb97808, PR #3435)
TITLE: compatible with new outlines (#3435)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/constrained/outlines_backend.py (+4/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-cddb1cdf8f  (L1, 2025-02-10, sha cddb1cdf8fd8, PR #3459)
TITLE: chore: bump v0.4.2.post4 (#3459)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile.rocm (+1/-1); python/pyproject.toml (+2/-2); benchmark/deepseek_v3/README.md (+1/-1); docs/developer/setup_github_runner.md (+2/-2); docs/start/install.md (+6/-6); python/sglang/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ - bunch of EAGLE 2 fixes ⏎ - FlashInfer cleanup and enable ragged FA3 by default ⏎ - update base image for better multi node support ⏎ - compatible with new outlines ⏎  ⏎ BTW v0.4.3 in on the way :) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2f47d710ae  (L1, 2025-02-10, sha 2f47d710ae9c, PR #3473)
TITLE: refine some typo (#3473)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-1); python/sglang/srt/layers/moe/topk.py (+1/-1); benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py (+1/-1); python/sglang/srt/server_args.py (+1/-1)
BODY: 

### L1-fdf04a1426  (L1, 2025-02-10, sha fdf04a142668, PR #3418)
TITLE: [ROCm] Add ROCm tuning config to block gemm and Re-tune for AMD Radeon Graphics (#3418)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=256,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json (+2/-2); benchmark/kernels/quantization/tuning_block_wise_fp8.py (+59/-12); python/sglang/srt/layers/quantization/configs/N=1536,K=7168,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json (+30/-30); python/sglang/srt/layers/quantization/configs/N=3072,K=1536,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json (+29/-29); python/sglang/srt/layers/quantization/configs/N=4096,K=512,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json (+33/-33); python/sglang/srt/layers/quantization/configs/N=4608,K=7168,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json (+31/-31); python/sglang/srt/layers/quantization/configs/N=512,K=7168,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json (+27/-27); python/sglang/srt/layers/quantization/configs/N=576,K=7168,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json (+31/-31); python/sglang/srt/layers/quantization/configs/N=7168,K=2048,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json (+24/-24); python/sglang/srt/layers/quantization/configs/N=7168,K=2304,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json (+30/-30); (+1 more)
BODY: ## Motivation ⏎ Add ROCm block gemm tuning configs ⏎  ⏎  ⏎ ## Modifications ⏎ Modify config tunings for AMD Radeon Graphics ⏎  ⏎  ⏎ ## Checklist

### L1-cadd5dbe6a  (L1, 2025-02-11, sha cadd5dbe6a6a, PR #3492)
TITLE: Tune MI300X fused MoE Triton kernel JSON config. (#3492)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=256,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128, 128].json (+2/-2)
BODY: ## Modifications ⏎  ⏎ Align the JSON configs between MI300X and Radeon Graphics for BS=64, E=256, N=256, dtype=fp8_w8a8, block_shape=[128,128] case. The best tuned config was submitted in #3418 but it was only for Radeon Graphics. Let MI300X adopt the same configuration in this PR.

### L1-45e3a7bc41  (L1, 2025-02-12, sha 45e3a7bc41d7, PR #3493)
TITLE: use sgl_per_token_group_quant_fp8 kernel (#3493)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+8/-1); python/sglang/srt/layers/quantization/fp8_kernel.py (+34/-0)
BODY: end2end perfomance: ⏎  ⏎ ```shell ⏎ ```shell ⏎ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000 ⏎  ⏎ # Run 2 times commands: ⏎  ⏎ python3 -m sglang.bench_serving --backend sglang --num-prompts 1000 --request-rate 8 ⏎ ``` ⏎  ⏎  ⏎ ### first time ⏎  ⏎ main: ⏎  ⏎ ```shell ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang     ⏎ Traffic request rate:                …[truncated]

### L1-871a4aa1bf  (L1, 2025-02-12, sha 871a4aa1bf30, PR #3536)
TITLE: [ROCm] Add ROCm tuning configs for AMD Instinct MI325X. (#3536)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=256,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=14336,device_name=AMD_Instinct_MI325X.json (+200/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=1792,device_name=AMD_Instinct_MI325X.json (+200/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=3584,device_name=AMD_Instinct_MI325X.json (+200/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=4096,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8.json (+178/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=7168,device_name=AMD_Instinct_MI325X.json (+200/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=8,N=8192,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8.json (+175/-0); python/sglang/srt/layers/quantization/configs/N=1536,K=7168,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/quantization/configs/N=3072,K=1536,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/quantization/configs/N=4096,K=512,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); (+6 more)
BODY: ## Modifications ⏎  ⏎ Add JSON tuning configs for AMD Instinct MI325X for block wise GEMM and fused MoE. ⏎  ⏎ ## Checklist

### L1-8616357a97  (L1, 2025-02-12, sha 8616357a97c5, PR #3450)
TITLE: Fix deepseek awq v3 (#3450)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-0); python/sglang/srt/layers/linear.py (+12/-5); python/sglang/srt/layers/quantization/__init__.py (+51/-5); python/sglang/srt/models/deepseek_v2.py (+4/-0)
LABELS: high priority
BODY: ``` ⏎ python -m sglang.launch_server --model-path cognitivecomputations/DeepSeek-V3-AWQ --tp-size 8 --trust-remote --disable-mla ⏎ ```

### L1-98eecbda54  (L1, 2025-02-13, sha 98eecbda54d5, PR #3529)
TITLE: integrate blockwise fp8 kernel (#3529)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/quantization/fp8_kernel.py (+90/-18); python/sglang/srt/layers/quantization/fp8_utils.py (+33/-4)
BODY: ## Motivation ⏎  ⏎  ⏎ Integrate #3267 into python side to optimize deepseekv3, merge it after #3267 has merged and sgl kernel is released ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Test Method ⏎ Following #3486, test on gsm8k and mmlu ⏎ |    | latency | accuracy | ⏎ |  ----  | ----  | --- | ⏎ | gsm8k (before) | 97.148s | 0.953 | ⏎ | gsm8k (after) | 82.708s | 0.958 | ⏎ | mmlu (before) | 250.839s | 0.871| ⏎ | mmlu (after) | 240.800s | 0.871| ⏎  ⏎ ## TODO ⏎  ⏎  ⏎ ## Checklist

### L1-f076328bb7  (L1, 2025-02-13, sha f076328bb7a6, PR #3534)
TITLE: fix moe_align_kernel shm init not sync bug (#3534)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+2/-0)
BODY: 

### L1-70f894b810  (L1, 2025-02-14, sha 70f894b810c0, PR #3550)
TITLE: feat: support flashinfer mla attention for deepseek v3 (#3550)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+3/-2); .github/workflows/pr-test.yml (+8/-8); python/sglang/global_config.py (+2/-0); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+234/-109); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/model_executor/model_runner.py (+12/-2); python/sglang/srt/models/deepseek_v2.py (+13/-7); python/sglang/srt/server_args.py (+7/-0); python/sglang/srt/utils.py (+7/-0); (+2 more)
BODY: ## Motivation ⏎  ⏎ Kudos to @yzh119 Throughout the integration process, we have identified and resolved numerous issues with the exceptional support from the FlashInfer team. Currently, **SGLang is the first open-source LLM inference engine to incorporate FlashInfer's new MLA Attention into the LLM engine among all frameworks.** ⏎  ⏎ ref https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.2.1 ⏎  ⏎ **This version should use `--enable-flashinfer- …[truncated]

### L1-e0b9a423c8  (L1, 2025-02-14, sha e0b9a423c841, PR #3556)
TITLE: chore: bump v0.4.3 (#3556)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile.rocm (+1/-1); python/pyproject.toml (+2/-2); benchmark/deepseek_v3/README.md (+1/-1); docs/developer/setup_github_runner.md (+2/-2); docs/start/install.md (+6/-6); python/sglang/version.py (+1/-1); test/srt/test_eagle_infer.py (+2/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-ac963be234  (L1, 2025-02-14, sha ac963be234cb, PR #3557)
TITLE: update flashinfer-python (#3557)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+8/-8); benchmark/deepseek_v3/README.md (+1/-1); docs/start/install.md (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3efbdf68b9  (L1, 2025-02-14, sha 3efbdf68b91e, PR #3563)
TITLE: fix sgl-kernel codestyle (#3563)
SOURCES: path_core
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu (+8/-6); sgl-kernel/src/sgl-kernel/csrc/lightning_attention_decode_kernel.cu (+20/-15); sgl-kernel/src/sgl-kernel/csrc/per_token_group_quant_fp8.cu (+6/-8)
BODY: Refine sgl-kernel cu file code style, all the tests have been passed locally.

### L1-862dd76c76  (L1, 2025-02-15, sha 862dd76c7619, PR #3582)
TITLE: Support NextN (MTP) speculative decoding for DeepSeek-V3/R1 (#3582)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+1/-0); python/sglang/srt/models/deepseek_nextn.py (+295/-0); python/sglang/srt/models/deepseek_v2.py (+4/-1); python/sglang/srt/server_args.py (+6/-3); python/sglang/srt/speculative/eagle_worker.py (+7/-2); python/sglang/srt/speculative/spec_info.py (+11/-1); scripts/export_deepseek_nextn.py (+113/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ We implemented NextN (MTP) speculative decoding for DeepSeek-V3/R1 based on EAGLE 2 on Triton backend (https://github.com/sgl-project/sglang/pull/3466) and achieved **1.76x** speed up with CUDA Graph and Torch.compile compatibility. In current benchmark, we achieved **77 token/s** output throughput on batch size 1. ⏎  ⏎ In our implementation, we only use the 1 MTP module (NextN layer) from the official model checkpoint. We found it …[truncated]

### L1-ddf39d3fce  (L1, 2025-02-17, sha ddf39d3fcee2, PR #3567)
TITLE: [ROCm] Optimal MOE Tuning for AMD Radeon Graphics (#3567)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=256,device_name=AMD_Radeon_Graphics,dtype=fp8_w8a8,block_shape=[128, 128].json (+50/-50); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+1/-1)
BODY: ## Motivation ⏎ Optimal MOE tuning for AMD Radeon Graphics Card ⏎  ⏎  ⏎ ## Modifications ⏎ Modify MOE configs for AMD Radeon Graphics ⏎  ⏎  ⏎ ## Checklist

### L1-c38f3aed24  (L1, 2025-02-18, sha c38f3aed241a, PR #3639)
TITLE: support multi-gpu block-gemm tuning (#3639)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=256,N=128,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); benchmark/kernels/quantization/tuning_block_wise_fp8.py (+72/-19); python/sglang/srt/layers/quantization/configs/N=1536,K=1536,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128, 128].json (+26/-0); python/sglang/srt/layers/quantization/configs/N=1536,K=7168,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128, 128].json (+26/-0); python/sglang/srt/layers/quantization/configs/N=2048,K=512,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128, 128].json (+26/-0); python/sglang/srt/layers/quantization/configs/N=2304,K=7168,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128, 128].json (+26/-0); python/sglang/srt/layers/quantization/configs/N=24576,K=7168,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128, 128].json (+26/-0); python/sglang/srt/layers/quantization/configs/N=256,K=7168,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128, 128].json (+26/-0); python/sglang/srt/layers/quantization/configs/N=32768,K=512,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128, 128].json (+26/-0); python/sglang/srt/layers/quantization/configs/N=576,K=7168,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128, 128].json (+26/-0); (+5 more)
BODY: 2xh800 ⏎  ⏎ before tuning: ⏎  ⏎ ![图片](https://github.com/user-attachments/assets/3ea74e0d-6602-493d-958e-dd41f44d624b) ⏎  ⏎ after tuning: ⏎  ⏎ ![图片](https://github.com/user-attachments/assets/4d0e25e7-a0bb-42a5-8cae-46f345774b10)

### L1-75d171a9c5  (L1, 2025-02-18, sha 75d171a9c59b, PR #3644)
TITLE: chore: update flashinfer v0.2.1.post2 (#3644)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); docs/start/install.md (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist
