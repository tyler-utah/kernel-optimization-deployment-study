### L2-3ae8e3ea8f  (L2, 2025-08-05, sha 3ae8e3ea8f32, PR #8836)
TITLE: chore: upgrade torch 2.8.0 (#8836)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+8/-8); .github/workflows/vllm-dependency-test.yml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci_install_dependency.sh (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-01c99a9959  (L2, 2025-08-06, sha 01c99a9959e0, PR #8872)
TITLE: chore: update Dockerfile (#8872)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+8/-5); .github/workflows/vllm-dependency-test.yml (+2/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-6ad6c8c9e6  (L2, 2025-08-06, sha 6ad6c8c9e662, PR #8834)
TITLE: feat: openai oss attention sink support with trtllm-gen backend #8825 (#8834)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/trtllm_mha_backend.py (+19/-15); python/sglang/srt/model_executor/model_runner.py (+3/-3); python/sglang/srt/models/gpt_oss.py (+1/-1); python/sglang/srt/server_args.py (+14/-3)
BODY: ## Motivation ⏎  ⏎ Add attention sinks. #8833  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ throughput ⏎ `python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1000 --parallel 1000` ⏎ Output  ⏎  ⏎ lm_eval ⏎ `lm_eval --model local-chat-completions --model_args model=gpt-oss,base_url=http://127.0.0.1:30000/v1/chat/completions,num_concurrent=128,timeout=999999,max_gen_toks=2048 --tasks gsm8k --batch_size 128 --apply_chat_template --nu …[truncated]

### L2-399e7ec8b3  (L2, 2025-08-06, sha 399e7ec8b3bc, PR #8868)
TITLE: Refine naming (#8868)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+16/-16); python/sglang/srt/layers/attention/triton_backend.py (+4/-4); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+9/-9); python/sglang/srt/models/gpt_oss.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ unify sink interface name for better compatibility for more attention backends

### L2-444013585d  (L2, 2025-08-08, sha 444013585d6d, PR #8799)
TITLE: Fix typos and unify size(s)/stride(s) API calls (#8799)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+5/-5); sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu (+3/-3); sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu (+1/-1); sgl-kernel/csrc/gemm/fp8_gemm_kernel.cu (+1/-1); sgl-kernel/csrc/gemm/int8_gemm_kernel.cu (+1/-1); sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu (+23/-23)
BODY: ## Motivation ⏎  ⏎ Fix typos and unify size(s)/stride(s) API calls ⏎  ⏎ ## Modifications ⏎  ⏎ Corrected typos and unified `strides()[x]` and `sizes()[x]` calls to `stride(x)` and `size(x)` ⏎  ⏎ ## Accuracy Test ⏎  ⏎ No loss of accuracy ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎ No efficiency loss ⏎  ⏎ ## Checklist ⏎  ⏎ - [☑] Format your code according to the [Code Formatting with Pre-Commit](https://docs.sglang.ai/references/contribution_guide.html#code-formatting-with-pr …[truncated]

### L2-54ea57f245  (L2, 2025-08-08, sha 54ea57f2451c, PR #8957)
TITLE: chore: bump sgl-kernel v0.3.3 (#8957)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-41357e511b  (L2, 2025-08-08, sha 41357e511b15, PR #8958)
TITLE: chore: update flashinfer (#8958)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-53f7874ae6  (L2, 2025-08-08, sha 53f7874ae623, PR #7279)
TITLE: refine aiter_backend for mtp (#7279)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+370/-107); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/speculative/eagle_worker.py (+16/-0)
LABELS: high priority, amd, speculative-decoding, aiter
BODY: 1. refine aiter_backend, for mtp ⏎ 2. enable aiter_biased_grouped_topk kernel (addressed other place) ⏎ 3. take aiter get_rope back (addressed other place) ⏎ 4. enable aiter fp8 block scale quant (addressed other place) ⏎  ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-4e7f025219  (L2, 2025-08-08, sha 4e7f02521920, PR #8772)
TITLE: chore(gb200): update to CUDA 12.9 and improve build process (#8772)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+19/-13); docker/Dockerfile.gb200 (+36/-24); .github/workflows/release-docker-gb200.yml (+3/-3); .github/workflows/release-whl-kernel-aarch64.yml (+7/-7); python/sglang/srt/entrypoints/engine.py (+1/-1); sgl-kernel/build.sh (+7/-0); sgl-kernel/rename_wheels.sh (+13/-2)
BODY: This PR does a couple different things ⏎ 1. Breaks apart our deepep/nvshmem installation for cleaner dockerfile practices ⏎ 2. Change GB200 dockerfile to cuda 12.9 because there doesnt seem to be torch 2.8+cu128 for aarch64 ⏎ 3. Denotes the proper arm runner so the `sgl-kernel` can be built for aarch ⏎ 4. Makes the wheel rename script more robust so we can properly rename for cuda 129/128 and cp39/310 ⏎  ⏎ SGL kernel for aarch64 has been officially rel …[truncated]

### L2-706bd69cc5  (L2, 2025-08-08, sha 706bd69cc58a, PR #8983)
TITLE: Clean up server_args.py to have a dedicated function for model specific adjustments (#8983)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: .github/workflows/execute-notebook.yml (+1/-4); .github/workflows/pr-test-pd-router.yml (+1/-1); .github/workflows/pr-test.yml (+38/-34); .github/workflows/vllm-dependency-test.yml (+4/-3); .gitmodules (+0/-0); README.md (+5/-4); docs/backend/server_arguments.md (+2/-2); docs/index.rst (+3/-3); python/pyproject.toml (+4/-5); python/sglang/srt/configs/model_config.py (+5/-7); (+14 more)
BODY: 

### L2-326a901df4  (L2, 2025-08-09, sha 326a901df448, PR #8998)
TITLE: chore: upgrade sgl-kernel 0.3.3 (#8998)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-19bc77f05c  (L2, 2025-08-09, sha 19bc77f05ce3, PR #8991)
TITLE: [Fix] Fix hicache backend (#8991)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/scheduler.py (+1/-6); python/sglang/srt/model_executor/model_runner.py (+22/-4)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎ It is a known issue that FA3 kernels do not work well with hicache kernels.   ⏎ When running FA3 decode kernels while hicache kernels are writing from GPU to CPU, an `illegal memory access` error may occur. ⏎  ⏎ Previously, we worked around this by either: ⏎ - Avoiding the FA3 backend entirely, or   ⏎ - Using direct `cudaMemcpyAsync` instead of hicache kernels   ⏎  ⏎ However, these workarounds can lead to suboptimal performance. ⏎  ⏎ This …[truncated]

### L2-f2887498f0  (L2, 2025-08-10, sha f2887498f055, PR #9033)
TITLE: Simplify memory pool (#9033)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/model_runner.py (+35/-35)
BODY: 

### L2-dd001a5477  (L2, 2025-08-10, sha dd001a54772b, PR #9036)
TITLE: chore: upgrade flashinfer 0.2.11 (#9036)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
DEEP_STUDY: deep-study: this PR was reverted by PR 9057 (confirmed_revert, reason=crash_or_hang)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-84cb449eec  (L2, 2025-08-11, sha 84cb449eeccf, PR #9057)
TITLE: Revert "chore: upgrade flashinfer 0.2.11 (#9036)" (#9057)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 9036 reason=crash_or_hang
BODY: This reverts commit dd001a54772b3f164f8ec359f77168109262bb01. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f4ae50e97c  (L2, 2025-08-11, sha f4ae50e97ce5, PR #)
TITLE: fix: use flashinfer v0.2.11.post1
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L2-9f24dfefd1  (L2, 2025-08-11, sha 9f24dfefd156, PR #9079)
TITLE: chore(gb200): remove ToT flashinfer installation  (#9079)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+0/-8)
BODY: SGL now uses newer flashinfer that contains fix for fp4 quantization. We can remove ToT installation

### L2-f508cd3cb7  (L2, 2025-08-11, sha f508cd3cb78a, PR #8638)
TITLE: TRTLLM-MLA FP8 path (#8638)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+119/-22); python/sglang/srt/models/deepseek_v2.py (+26/-2); python/sglang/srt/server_args.py (+10/-1); docs/advanced_features/attention_backend.md (+5/-0); python/sglang/test/attention/test_trtllm_mla_backend.py (+186/-36)
BODY: ## Motivation ⏎  ⏎ Adds supports for converting MLA query to `fp8_e4m3` and run `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` in FP8 path. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add query fp8 conversion ⏎ - Change server args and docs ⏎  ⏎ ## Accuracy Test ⏎  ⏎ [details omitted] ⏎ [details omitted] ⏎ [details omitted] ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-3a9afe2a42  (L2, 2025-08-12, sha 3a9afe2a42eb, PR #9103)
TITLE: chore: bump sgl-kernel v0.3.4 (#9103)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-1f9ec65374  (L2, 2025-08-12, sha 1f9ec65374f8, PR #9118)
TITLE: fix(docker): update sgl_kernel version to 0.3.4 in Dockerfile.gb200 (#9118)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+1/-1)
BODY: 

### L2-c9ee738515  (L2, 2025-08-12, sha c9ee73851540, PR #9014)
TITLE: Fuse writing KV buffer into rope kernel (part 2: srt) (#9014)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); .github/workflows/pr-test-pd-router.yml (+1/-1); docker/Dockerfile.gb200 (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/rotary_embedding.py (+10/-0); python/sglang/srt/models/gpt_oss.py (+51/-2)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ Fuse  set_kv_buffer to sgl-kernel rope function, only for trtllm_mha attention ⏎  ⏎ ----- ⏎  ⏎ (below is from @fzyzcjy) ⏎  ⏎ speed may be suboptimal (I have not done any ncu profile or thorough optimization), but anyway it is faster than non-fused ⏎  ⏎ ``` ⏎ tests/test_rotary_embedding.py ..................                                                                  [100%] ⏎  ⏎ =================================================== 18 p …[truncated]

### L2-305b27c124  (L2, 2025-08-12, sha 305b27c12476, PR #9125)
TITLE: fix: update Dockerfile (#9125)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2ecbd8b8bf  (L2, 2025-08-12, sha 2ecbd8b8bf98, PR #8700)
TITLE: [feat] add ascend readme and docker release (#8700)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.npu (+81/-0); .github/workflows/pr-test-npu.yml (+21/-3); .github/workflows/release-docker-npu-nightly.yaml (+76/-0); .github/workflows/release-docker-npu.yaml (+77/-0); docs/basic_usage/deepseek.md (+2/-0); docs/platforms/ascend_npu.md (+203/-4); scripts/ci/npu_ci_install_dependency.sh (+7/-11)
LABELS: ready-to-merge, npu
BODY: ## Motivation ⏎  ⏎ In the past, we only had images for GPU and AMD, but this PR would try to build and push docker image for NPU hardware ⏎  ⏎ ## Modifications ⏎  ⏎ Add two new workflow and NPU related Dockerfile, both docker images will be published to [offical registry](https://hub.docker.com/r/lmsysorg/sglang): ⏎ 1. daily dev image for user to try and nightly test case, named as `sglang:main-cann8.2.rc1.alpha003-a3 ` ⏎ 2. release image when new tag ad …[truncated]

### L2-c81daf838d  (L2, 2025-08-12, sha c81daf838da5, PR #9129)
TITLE: fix: update Dockerfile (#9129)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-924827c3de  (L2, 2025-08-12, sha 924827c3ded3, PR #9130)
TITLE: chore: use cp310 (#9130)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-0ff6d1fce1  (L2, 2025-08-13, sha 0ff6d1fce122, PR #9028)
TITLE: Support FA3 backend for gpt-oss (#9028)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.dispatch.server_args_defaults
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+18/-0); python/sglang/srt/models/gpt_oss.py (+1/-1); python/sglang/srt/server_args.py (+4/-4)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Apply changes of https://github.com/sgl-project/sgl-attn/pull/4. ⏎  ⏎ ## Accuracy Test ⏎ `openai/gpt-oss-20b` mmlu 4k: ⏎ ``` ⏎ | model_name                                              |   ('metric', 'mmlu') | ⏎ |:--------------------------------------------------------|---------------------:| ⏎ | o4-mini-with-chat-completion-and-4k-gen_20250810_151719 |                0.835 | ⏎ ``` ⏎  ⏎ ## Benchmark & Profiling ⏎ ### `openai/gpt-oss-20b` T …[truncated]

### L2-71fb8c9527  (L2, 2025-08-13, sha 71fb8c9527cb, PR #9126)
TITLE: feat: update fa3 (#9126)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-4dbf43601d  (L2, 2025-08-14, sha 4dbf43601d49, PR #9065)
TITLE: fix: zero_init buffer (#9065)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-0); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+10/-3); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+1/-0); python/sglang/srt/layers/attention/trtllm_mha_backend.py (+8/-6); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=128,N=384,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=128,N=768,device_name=NVIDIA_H20.json (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=257,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+1/-1)
BODY: ## Motivation ⏎  ⏎ Flashinfer **v0.2.11.post3** feature: enable PDL, trtllm-gen attention zero_init buffer ⏎  ⏎ Fix crash at running gptoss introduced by this flashinfer update: ⏎ https://github.com/flashinfer-ai/flashinfer/pull/1463 ⏎  ⏎ How to re-produce the crash: ⏎ ``` ⏎ python3 -m sglang.launch_server --model openai/gpt-oss-120b --tp 8 --port 40000 --attention-backend trtllm_mha ⏎ OPENAI_API_KEY=“” python3 -m gpt_oss.evals --model openai/gpt-oss-120b  …[truncated]

### L2-1fea998a45  (L2, 2025-08-14, sha 1fea998a452f, PR #9185)
TITLE: chore: bump sgl-kernel v0.3.5 (#9185)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2cc9eeab01  (L2, 2025-08-14, sha 2cc9eeab015d, PR #9191)
TITLE: [4/n]decouple quantization implementation from vLLM dependency (#9191)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); .github/workflows/vllm-dependency-test.yml (+1/-6); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/__init__.py (+5/-32); python/sglang/srt/layers/quantization/awq.py (+10/-14); python/sglang/srt/layers/quantization/gptq.py (+10/-16); python/sglang/srt/layers/quantization/marlin_utils.py (+8/-3); test/srt/run_suite.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ follow https://github.com/sgl-project/sglang/pull/8112 ⏎ remove vllm dependency for AWQ and GPTQ quantization. ⏎  ⏎ co-author:@AniZpZ ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-27985c27aa  (L2, 2025-08-14, sha 27985c27aa74, PR #9202)
TITLE: feat: update model config (#9202)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+16/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-d4db9b028b  (L2, 2025-08-14, sha d4db9b028b86, PR #9208)
TITLE: fix: the store_dtype typo for ascend mla (#9208)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/memory_pool.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ fix: the store_dtype typo for ascend mla ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Change `store_dtype` to `self.store_dtype` ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-a3d99d6dcd  (L2, 2025-08-15, sha a3d99d6dcddd, PR #8790)
TITLE: [Misc] feat: Deepgemm update for sgl-kernel (#8790)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+46/-20)
LABELS: high priority
DEEP_STUDY: deep-study: this PR was reverted by PR 9260 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Update deepgemm sgl-kernel for nvrtc ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-c186feed7f  (L2, 2025-08-15, sha c186feed7fb7, PR #9220)
TITLE: chore: bump sgl-kernel v0.3.6 (#9220)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
LABELS: high priority
DEEP_STUDY: deep-study: this PR was reverted by PR 9247 (confirmed_revert, reason=unstated)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e52c3866eb  (L2, 2025-08-15, sha e52c3866eb77, PR #9243)
TITLE: chore(docker): update sgl_kernel version to 0.3.6 in Dockerfile.gb200 (#9243)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+1/-1)
DEEP_STUDY: deep-study: this PR was reverted by PR 9246 (confirmed_revert, reason=unstated)
BODY: 

### L2-5121af4627  (L2, 2025-08-15, sha 5121af4627ef, PR #9246)
TITLE: Revert "chore(docker): update sgl_kernel version to 0.3.6 in Dockerfi… (#9246)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+1/-1)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 9243 reason=unstated
BODY: …le.gb200 (#9243)" ⏎  ⏎ This reverts commit e52c3866eb7770076865da38618ce3cccc3af00f. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-87dab54824  (L2, 2025-08-15, sha 87dab548243b, PR #9247)
TITLE: Revert "chore: bump sgl-kernel v0.3.6 (#9220)" (#9247)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 9220 reason=unstated
BODY: This reverts commit c186feed7fb7604db59377e74d48bcc61053832e. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-0c8594e67d  (L2, 2025-08-15, sha 0c8594e67d1e, PR #9231)
TITLE: Optional extension for green context (#9231)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+12/-1); sgl-kernel/csrc/common_extension.cc (+0/-6); sgl-kernel/csrc/spatial/greenctx_stream.cu (+3/-7); sgl-kernel/csrc/spatial_extension.cc (+29/-0); sgl-kernel/python/sgl_kernel/__init__.py (+14/-1); sgl-kernel/python/sgl_kernel/spatial.py (+15/-5)
BODY: follow #9021

### L2-be1a3cd9b4  (L2, 2025-08-17, sha be1a3cd9b48c, PR #9279)
TITLE: Fix swa eagle verify accuracy for Triton backend (#9279)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+2/-2); python/sglang/srt/layers/attention/triton_backend.py (+60/-36); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+14/-2)
BODY: ## Motivation ⏎  ⏎ For sliding window layers, we should add offset to read custom mask in the sliding window. ⏎  ⏎ ## Accuracy Test ⏎ ``` ⏎ python3 -m sglang.launch_server --model openai/gpt-oss-20b --speculative-algorithm EAGLE3 --speculative-draft-model-path zhuyksir/EAGLE3-gpt-oss-20b-bf16 --speculative-num-steps 2 --speculative-eagle-topk 4 --speculative-num-draft-tokens 4 --attention-backend triton ⏎ OPENAI_BASE_URL=http://localhost:30000/v1 OPENAI …[truncated]

### L2-a1c7f742f9  (L2, 2025-08-17, sha a1c7f742f912, PR #9286)
TITLE: chore: bump sgl-kernel v0.3.6.post1 (#9286)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-4d98e48649  (L2, 2025-08-17, sha 4d98e4864998, PR #9260)
TITLE: Revert "[Misc] feat: Deepgemm update for sgl-kernel (#8790)" to fix kernel CI (#9260)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+20/-40)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 8790 reason=ci_or_test_failure
BODY: This reverts commit a3d99d6dcdddcfb6bde7b50c0830f0ad5534d74d. ⏎  ⏎ CI failes https://github.com/sgl-project/sglang/actions/runs/17002005000/job/48209894546 ⏎  ⏎ To reintroduce deepgemm upgrade, it depends on #9167 ⏎  ⏎ @fzyzcjy @FlamingoPg @zhyncs

### L2-64574ef8c0  (L2, 2025-08-21, sha 64574ef8c003, PR #9238)
TITLE: Enables speculative decoding for the trtllm_mla attention backend (#9238)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+39/-16); python/sglang/srt/server_args.py (+0/-5); python/sglang/srt/speculative/eagle_worker.py (+21/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Enables speculative decoding for the trtllm_mla attention backend ⏎  ⏎ ## Modifications ⏎  ⏎ Adds a new `TRTLLMMLAMultiStepDraftBackend` and allows trtllm_mla backend to be used when speculative decoding is enabled. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ [details omitted] ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-3cc3d9b950  (L2, 2025-08-21, sha 3cc3d9b950e4, PR #8593)
TITLE: Add Support for Page Size greater than 1 for Flashinfer MLA Backend (#8593)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+90/-72); python/sglang/srt/layers/attention/utils.py (+94/-15); python/sglang/srt/server_args.py (+4/-0); test/srt/test_create_kvindices.py (+59/-17); test/srt/test_mla_flashinfer.py (+44/-0)
LABELS: high priority, ready-for-review
DEEP_STUDY: deep-study: this PR was reverted by PR 9581 (confirmed_revert, reason=unstated)
BODY: ## Motivation ⏎  ⏎ This change modifies the create_flashinfe_kv_indices kernelto take in page size as input ⏎ and return the paged kv indices. Follows a similar style as FlashMLA for converting req_to_token ⏎ mapping to page index for the token/request. ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Changes the logic of the triton kernel `create_flashinfer_kv_indices_triton` ⏎ 2. Changes the plan metadata preparation for decode updater and prefill updater. ⏎  ⏎  ⏎  ⏎ ## …[truncated]

### L2-9708d353b7  (L2, 2025-08-21, sha 9708d353b756, PR #8616)
TITLE: Support MHA with chunked prefix cache for flashinfer/flashmla backend, support page size > 1 for MHA chunked prefix (#8616)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.backend.fa3_fa4_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+11/-7); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+137/-5); python/sglang/srt/model_executor/forward_batch_info.py (+3/-0); python/sglang/srt/model_executor/model_runner.py (+0/-3); python/sglang/srt/models/deepseek_v2.py (+32/-60); python/sglang/srt/managers/schedule_batch.py (+1/-0)
LABELS: high priority, ready-to-merge
BODY: ## Motivation ⏎  ⏎  ⏎ Based on #5113, for issue #6959, support flashinfer and flashmla (actually flashinfer as well) backend. remove the page-size=1 restriction: support page size > 1. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎ ```shell ⏎ export SGL_ENABLE_JIT_DEEPGEMM=1 ⏎ export TORCHINDUCTOR_CACHE_DIR=/home/admin/inductor_root_cache ⏎ export SGLANG_TORCH_PROFILER_DIR=/home/admin/torch_profiler ⏎ model_path=/home/models/deepseek-ai__DeepSeek-R1_64k …[truncated]

### L2-243e745d07  (L2, 2025-08-21, sha 243e745d0758, PR #9480)
TITLE: Add trtllm_mla and cutlass_mla for ragged fmha for chunked prefill  (#9480)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+2/-0)
BODY: ## Motivation ⏎  ⏎ Add trtllm_mla and cutlass_mla for ragged fmha for chunked prefill  since they inherit flashinfer for handling prefill for perf improvement ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-b6b2287e4b  (L2, 2025-08-21, sha b6b2287e4b9f, PR #9475)
TITLE: chore: bump sgl-kernel v0.3.6.post2 (#9475)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-9ec314c6ac  (L2, 2025-08-21, sha 9ec314c6ac05, PR #9331)
TITLE: Support speculative decoding in the trtllm_mha attention backend (#9331)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/trtllm_mha_backend.py (+388/-28); python/sglang/srt/server_args.py (+10/-5); python/sglang/srt/speculative/eagle_worker.py (+16/-0)
LABELS: high priority, speculative-decoding
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ ### Launch the server ⏎ ``` ⏎ python3 -m sglang.launch_server   --model openai/gpt-oss-120b   --speculative-algo EAGLE3   --speculative-draft zhuyksir/EAGLE3-gpt-oss-120b-bf16   --speculative-num-steps 2   --speculative-eagle-topk 1   --speculative-num-draft-tokens 3    --dtype float16 --attention-backend trtllm_mha    --tp 4 --port 40010 ⏎ ``` ⏎  ⏎ ### Run the accuracy test ⏎ ``` ⏎ OPENAI_ …[truncated]

### L2-f445a1d9a3  (L2, 2025-08-22, sha f445a1d9a3a3, PR #7699)
TITLE: [AMD] Fix Llama 4 FP8 accuracy issues on MI300X (#7699)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+0/-1); python/sglang/srt/layers/moe/rocm_moe_utils.py (+141/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+66/-15); python/sglang/srt/layers/quantization/fp8.py (+1/-0); python/sglang/srt/server_args.py (+4/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ - To fix Llama 4 FP8 accuracy issue on MI300X (https://github.com/sgl-project/sglang/issues/5362). ⏎ - This PR also includes the initial effort to streamline/modularize aiter's fused_moe implementations for various code paths. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ **Command to launch server** ⏎ ``` ⏎ SGLANG_USE_AITER=1 python -m sglang.launch_server --model-path /data/huggingface/meta-llama/Llama-4-Maverick-17B-128E-Instru …[truncated]

### L2-6078d5fcc0  (L2, 2025-08-22, sha 6078d5fcc009, PR #8865)
TITLE: [HiCacheStorage] backup optimization for MLA model (#8865)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/cache_controller.py (+26/-12); python/sglang/srt/mem_cache/hicache_storage.py (+2/-2); python/sglang/srt/mem_cache/memory_pool_host.py (+3/-2); python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py (+8/-4)
LABELS: ready-to-merge
BODY: ## Motivation ⏎ For the MLA model, the cache of each TP Rank is the same, and only the cache of Rank 0 needs to be written. The capacity of l3 can be expanded to the original tp_size times ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Test ⏎  ⏎  ⏎  ⏎ ## Benchmark & Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-86d10d220f  (L2, 2025-08-23, sha 86d10d220f66, PR #9532)
TITLE: Update grok.py and tiktoken tokenizer (#9532)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+31/-0); python/sglang/srt/constrained/xgrammar_backend.py (+10/-6); python/sglang/srt/hf_transformers_utils.py (+5/-0); python/sglang/srt/layers/attention/triton_backend.py (+16/-2); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+18/-0); python/sglang/srt/layers/elementwise.py (+94/-0); python/sglang/srt/layers/moe/router.py (+15/-9); python/sglang/srt/layers/radix_attention.py (+6/-0); python/sglang/srt/models/grok.py (+376/-47); python/sglang/srt/tokenizer/tiktoken_tokenizer.py (+161/-0)
BODY: 

### L2-bf863e3bbf  (L2, 2025-08-24, sha bf863e3bbfff, PR #9565)
TITLE: fix: use sgl-kernel 0.3.5 (#9565)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-938e986e15  (L2, 2025-08-25, sha 938e986e1584, PR #9578)
TITLE: chore: upgrade flashinfer 0.2.14.post1 (#9578)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-ebd9dbe71b  (L2, 2025-08-25, sha ebd9dbe71ba4, PR #9581)
TITLE: fix: revert #8593 (#9581)
SOURCES: path_core, symbol_pickaxe, corpus:confirmed-reverts
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+71/-89); python/sglang/srt/layers/attention/utils.py (+15/-94); python/sglang/srt/server_args.py (+0/-4); test/srt/test_create_kvindices.py (+17/-59); test/srt/test_mla_flashinfer.py (+0/-44)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 8593 reason=unstated
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f92b729d52  (L2, 2025-08-25, sha f92b729d5240, PR #8328)
TITLE: [new feat] ascend backend support fia fusion kernel (#8328)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.pool.mla_token_kv
FILES: .github/workflows/pr-test-npu.yml (+3/-3); python/sglang/srt/layers/attention/ascend_backend.py (+218/-111); python/sglang/srt/layers/moe/topk.py (+1/-1); python/sglang/srt/mem_cache/memory_pool.py (+73/-14); python/sglang/srt/models/deepseek_v2.py (+9/-1); test/srt/ascend/test_ascend_mla_fia_w8a8int8.py (+103/-0); test/srt/ascend/test_ascend_mla_w8a8int8.py (+1/-0); test/srt/ascend/test_ascend_tp2_fia_bf16.py (+101/-0); test/srt/run_suite.py (+2/-0)
LABELS: high priority, ready-to-merge, npu
BODY: ## Motivation ⏎ In this MR, we implemented the NPU fusion kernels npu_fused_infer_attention_score in the Qwen2.5-7b, deepseek-v2-lite and deepseek-v3 models, this fusion kernel is  suitable for the graph mode. One needs to export **ASCEND_USE_FIA=ture** to activate this  fusion kernel. ⏎  ⏎ ## Modifications ⏎ Ascend Backend: support npu_fused_infer_attention_score kernel ⏎ Add unittest: test_ascend_tp_fia_bf16.py and test_ascend_mla_fia_w8a8int8.py ⏎  …[truncated]

### L2-fd71b11b1d  (L2, 2025-08-27, sha fd71b11b1d96, PR #9679)
TITLE: move is_sm90_supported/is_sm100_supported to python/sglang/srt/utils.py (#9679)
SOURCES: path_core
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+5/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+5/-2); python/sglang/srt/layers/communicator.py (+1/-2); python/sglang/srt/layers/moe/cutlass_moe.py (+0/-8); python/sglang/srt/layers/quantization/fp8.py (+2/-1); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-1); python/sglang/srt/layers/quantization/mxfp4.py (+1/-2); python/sglang/srt/layers/utils.py (+0/-14); python/sglang/srt/model_executor/model_runner.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+3/-2); (+3 more)
BODY: 

### L2-aa3eba8eb4  (L2, 2025-08-27, sha aa3eba8eb42c, PR #9340)
TITLE: [sgl-kernel] misc: update deepgemm version for sgl-kernel (#9340)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+48/-44); .github/workflows/pr-test-sgl-kernel.yml (+2/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-7); python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py (+133/-235); python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py (+5/-7); python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py (+5/-23); python/sglang/srt/layers/quantization/fp8_kernel.py (+2/-2); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-1); python/sglang/srt/layers/quantization/mxfp4_tensor.py (+3/-1); sgl-kernel/csrc/moe/marlin_moe_wna16/generate_kernels.py (+2/-25); (+15 more)
LABELS: bug, enhancement, high priority, dependencies
BODY: ## Motivation ⏎  ⏎ DeepGEMM updated for unify cuda version. ⏎ So we need upd for TORCH LIBRARY. ⏎  ⏎ It depends on: https://github.com/sgl-project/sglang/pull/9167 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-b962a296ed  (L2, 2025-08-27, sha b962a296edbe, PR #9708)
TITLE: chore: upgrade sgl-kernel 0.3.7 (#9708)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); .github/workflows/vllm-dependency-test.yml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-bc80dc4ce0  (L2, 2025-08-27, sha bc80dc4ce0ae, PR #9716)
TITLE: chore: bump v0.5.1.post3 (#9716)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); python/pyproject.toml (+1/-1); benchmark/deepseek_v3/README.md (+1/-1); docs/get_started/install.md (+2/-2); docs/platforms/amd_gpu.md (+1/-1); docs/platforms/ascend_npu.md (+1/-1); python/sglang/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-ae7428a8a7  (L2, 2025-08-27, sha ae7428a8a737, PR #9678)
TITLE: fix mooncake store mla zero copy meta (#9678)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/mem_cache/memory_pool_host.py (+1/-2)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-fce7ae33f8  (L2, 2025-08-28, sha fce7ae33f883, PR #9745)
TITLE: [Sync] Update server_args.py (20250828) (#9745)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+80/-54)
BODY: Sync changes from commit `50c89afb`. ⏎  ⏎ **Relevant Files Changed:** ⏎ - python/sglang/srt/server_args.py

### L2-74dd4249ac  (L2, 2025-08-28, sha 74dd4249ac60, PR #9355)
TITLE: [Feature] Support NPUGraph for DeepSeek on Ascend NPU (#9355)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.pool.mla_token_kv
FILES: python/sglang/srt/disaggregation/ascend/conn.py (+75/-0); python/sglang/srt/layers/attention/ascend_backend.py (+183/-88); python/sglang/srt/layers/moe/ep_moe/layer.py (+12/-6); python/sglang/srt/layers/moe/topk.py (+12/-2); python/sglang/srt/layers/quantization/w8a8_int8.py (+7/-3); python/sglang/srt/mem_cache/memory_pool.py (+4/-0); python/sglang/srt/models/deepseek_v2.py (+14/-6)
LABELS: high priority, ready-to-merge, npu
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ this pr improves deepseek model (mla to be exact) performance with npugraph support ⏎ check initial npugraph support on mha model here #9399 and #8030 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - support concurrent kvcache transfer following mooncake transfer engine ⏎ - add mla graph support in ascend attention backend ⏎ - some performance and accuracy improvements ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ <img width="925" height="72" alt="87DF5544-5E77-4E8C …[truncated]

### L2-9f81d741a2  (L2, 2025-08-28, sha 9f81d741a286, PR #6287)
TITLE: fix: fix MLA for ShardedModelLoader/RemoteModelLoader (#6287)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: examples/runtime/engine/save_remote_state.py (+1/-2); python/sglang/srt/connector/__init__.py (+1/-1); python/sglang/srt/connector/base_connector.py (+1/-2); python/sglang/srt/connector/redis.py (+2/-2); python/sglang/srt/connector/serde/__init__.py (+1/-1); python/sglang/srt/connector/serde/safe_serde.py (+4/-3); python/sglang/srt/model_loader/loader.py (+15/-24); python/sglang/srt/model_loader/utils.py (+12/-0)
ISSUES: #6286 [Bug] MLA and serialization/deserialization bugs for RemoteModelLoader
BODY: ## Motivation ⏎ fix #6286  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-161e9dc51e  (L2, 2025-08-29, sha 161e9dc51e02, PR #9692)
TITLE: feat(hicache-3fs): 3FS-Store Backup Optimizations For MLA Model. (#9692)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py (+25/-5)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎ 3FS-Store Backup Optimizations For MLA Model. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-a23c30205d  (L2, 2025-08-29, sha a23c30205d18, PR #9784)
TITLE: Raise error when `topk>1` and `page>1` for paged attention backends. (#9784)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+9/-0)
BODY: related to #7725

### L2-3d8fc43400  (L2, 2025-08-29, sha 3d8fc43400ba, PR #9793)
TITLE: chore: upgrade flashinfer 0.3.0rc1 (#9793)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-ff9b561817  (L2, 2025-08-29, sha ff9b56181792, PR #9675)
TITLE: Fix TRTLLM MLA Cuda KV Blocks Causing accuracy drop (#9675)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+25/-10); python/sglang/test/attention/test_trtllm_mla_backend.py (+12/-3)
BODY: ## Motivation ⏎  ⏎  ⏎ Fix for issue https://github.com/sgl-project/sglang/pull/8638#issuecomment-3193478283 by setting number of `kv_indices` based on `self.max_context_len` ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ `max_blocks_per_seq = self._calc_padded_blocks(self.max_context_len)` ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ [details omitted] ⏎  ⏎  ⏎  ⏎ [details omitted] ⏎  ⏎  ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ Sanity benchmark WIP ag …[truncated]

### L2-9970e3bf32  (L2, 2025-08-30, sha 9970e3bf328a, PR #9822)
TITLE: chore: upgrade sgl-kernel 0.3.7.post1 with deepgemm fix (#9822)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-349b491c63  (L2, 2025-09-01, sha 349b491c635a, PR #9864)
TITLE: chore: upgrade flashinfer 0.3.0 (#9864)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-58d06fdc95  (L2, 2025-09-01, sha 58d06fdc9560, PR #9876)
TITLE: [HiCacheStorage]: Improve 3fs kvstore‘s performance and resolve mla issues (#9876)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/mem_cache/storage/hf3fs/mini_3fs_metadata_server.py (+61/-34); python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py (+27/-6)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-1db649ac02  (L2, 2025-09-02, sha 1db649ac0204, PR #9879)
TITLE: [feat] apply deep_gemm compile_mode to skip launch (#9879)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py (+8/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ do not merged until next version of sgl-kernel bumped ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-d631290e32  (L2, 2025-09-02, sha d631290e32b8, PR #9905)
TITLE: Remove annoying warnings in sgl kernel build (#9905)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+2/-2); sgl-kernel/CMakeLists.txt (+34/-32); sgl-kernel/Makefile (+1/-2); sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu (+1/-0); sgl-kernel/csrc/gemm/nvfp4_expert_quant.cu (+5/-0)
BODY: remove most nvcc warnings by either fixing the code or suppressing them

### L2-0dfd54d11d  (L2, 2025-09-02, sha 0dfd54d11d06, PR #9671)
TITLE: Optimized deepseek-v3/r1 model performance on mxfp4 run (#9671)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/layers/communicator.py (+41/-5); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py (+49/-30); python/sglang/srt/layers/quantization/quark/utils.py (+97/-0); python/sglang/srt/layers/quantization/rocm_mxfp4_utils.py (+13/-0); python/sglang/srt/layers/rocm_linear_utils.py (+44/-0); python/sglang/srt/models/deepseek_v2.py (+202/-27); python/sglang/srt/utils.py (+12/-0)
DEEP_STUDY: deep-study: this PR was reverted by PR 9959 (confirmed_revert, reason=hardware_specific_breakage)
BODY: ## Motivation ⏎  ⏎ In order to decrease the activation tensor quantized overhead, fused the quantized behavior into the different ops (activation, layernorm, gemm, flatten) ⏎  ⏎ ## Modifications ⏎ ## Accuracy Tests ⏎ ## Benchmarking and Profiling ⏎ Test below commands, we got about 10% throughput increased. ⏎ **Server:** ⏎ SGLANG_USE_AITER=1 python3 -m sglang.launch_server --model-path ams/DeepSeek-R1-WMXFP4-Preview --tp 8 --trust-remote-code --chunked-pr …[truncated]

### L2-1b2ff4fb7f  (L2, 2025-09-03, sha 1b2ff4fb7f05, PR #9959)
TITLE: Revert "Optimized deepseek-v3/r1 model performance on mxfp4 run (#9671)" (#9959)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/layers/communicator.py (+5/-41); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py (+30/-49); python/sglang/srt/layers/quantization/quark/utils.py (+0/-97); python/sglang/srt/layers/quantization/rocm_mxfp4_utils.py (+0/-13); python/sglang/srt/layers/rocm_linear_utils.py (+0/-44); python/sglang/srt/models/deepseek_v2.py (+27/-202); python/sglang/srt/utils.py (+0/-12)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 9671 reason=hardware_specific_breakage
BODY: This reverts commit 0dfd54d11d06eb8363bc7fb3cf9a1f464368caf8. ⏎  ⏎ Because #9671 breaks  ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model zai-org/GLM-4.5-Air --tp-size 4 --trust-remote-code --mem-fraction-static 0.85 --ep-size 4 ⏎ ``` ⏎  ⏎ on B200 ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e96973742c  (L2, 2025-09-04, sha e96973742c32, PR #10008)
TITLE: Optimized deepseek-v3/r1 model performance on mxfp4 run (#10008)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/layers/communicator.py (+36/-4); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py (+49/-30); python/sglang/srt/layers/quantization/quark/utils.py (+97/-0); python/sglang/srt/layers/quantization/rocm_mxfp4_utils.py (+13/-0); python/sglang/srt/layers/rocm_linear_utils.py (+44/-0); python/sglang/srt/models/deepseek_v2.py (+228/-32); python/sglang/srt/models/glm4_moe.py (+10/-1); python/sglang/srt/utils.py (+12/-0)
BODY: ## Motivation ⏎  ⏎ In order to decrease the activation tensor quantized overhead, fused the quantized behavior into the different ops (activation, layernorm, gemm, flatten) ⏎  ⏎ ## Modifications ⏎ ## Accuracy Tests ⏎ ## Benchmarking and Profiling ⏎ Test below commands, we got about 10% throughput increased. ⏎ Server: ⏎ SGLANG_USE_AITER=1 python3 -m sglang.launch_server --model-path ams/DeepSeek-R1-WMXFP4-Preview --tp 8 --trust-remote-code --chunked-prefil …[truncated]

### L2-918e3d4c27  (L2, 2025-09-04, sha 918e3d4c27c3, PR #8677)
TITLE: Fix accuracy drop of dsv3 run in dp enablement (#8677)
SOURCES: corpus:kernel-correctness-cases
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+93/-68); python/sglang/srt/models/deepseek_v2.py (+7/-1)
LABELS: high priority, ready-to-merge
DEEP_STUDY: deep-study correctness case sglang:918e3d4c27: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Motivation ⏎  ⏎ Got an accuracy issue of dsv3 model run on aiter backend with dp enablement. ⏎  ⏎ Issue link: https://github.com/sgl-project/sglang/issues/7692 ⏎  ⏎ ## Modifications ⏎  ⏎ For extend attention, use absorb attention to handle it, not use the MHA attention. ⏎  ⏎ ## Accuracy Test ⏎  ⏎ `/sgl-workspace/sglang# python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 ⏎ 100%|██████████████████████████████████████████████████████████████████████ …[truncated]

### L2-0e9387a95d  (L2, 2025-09-04, sha 0e9387a95ddc, PR #10052)
TITLE: fix: update gb200 dep (#10052)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2985090084  (L2, 2025-09-05, sha 298509008451, PR #10087)
TITLE: Update flashinfer to 0.3.1 for B300 support (#10087)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ Update flashinfer to 0.3.1 for initial B300 support ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ CI ⏎  ⏎ ## Checklist

### L2-bebd0576e5  (L2, 2025-09-05, sha bebd0576e5f0, PR #9801)
TITLE: Integrate trtllm ragged attention for prefill self-attention (#9801)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+16/-12); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+104/-24); python/sglang/srt/models/deepseek_v2.py (+9/-1); python/sglang/test/attention/test_trtllm_mla_backend.py (+169/-5)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ ### Accuracy ⏎ Accuracy: 0.961 ⏎ Invalid: 0.000 ⏎ Latency: 93.237 s ⏎ Output throughput: 1526.508 token/s ⏎  ⏎ ``` ⏎ SGLANG_CUTLASS_MOE=1 python3 -m sglang.launch_server --tokenizer-path deepseek-ai/DeepSeek-R1-0528 --trust-remote-code --enable-dp-attention --disable-radix-cache  --mem-fraction-static 0.8 --model-path=deepseek-ai/DeepSeek-R1-0528 --host 0.0.0.0 --port 8000 --tensor-parallel-size=8 --data-parallel-size=8  --attention-bac …[truncated]

### L2-4c22ebe2e8  (L2, 2025-09-06, sha 4c22ebe2e8ec, PR #10058)
TITLE: Disable kernel cutlass_mla_decode on SM103 (#10058)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.kernel.cutlass_mla
FILES: sgl-kernel/csrc/attention/cutlass_mla_kernel.cu (+5/-0); sgl-kernel/tests/test_cutlass_mla.py (+3/-2)
BODY: ## Motivation ⏎  ⏎ Half of the accuracy tests of cutlass_mla_decode is failing on SM103. Make it throw an exception on SM103 for now until the cause of the failure can be identified and fixed. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Full accuracy tests results are here: https://gist.github.com/hlu1/405371e28ea236150eae4a65c751c8f9 ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-90dfe3de4c  (L2, 2025-09-06, sha 90dfe3de4c1c, PR #9861)
TITLE: [NVIDIA] disable chunked prefix cache when dp and blackwell is used (#9861)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/model_runner.py (+11/-0)
BODY: This PR is to work around an accuracy issue found in https://github.com/sgl-project/sglang/issues/9806. ⏎  ⏎ It seems the chunked prefix cache doesn't work well with the DP. So, we disable it for now to recover the accuracy. ⏎  ⏎ cc @trevor-m @kushanam @zhyncs

### L2-7577f0e40f  (L2, 2025-09-07, sha 7577f0e40f56, PR #7843)
TITLE: Add graph runner support with torch compile on CPU (#7843)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-xeon.yml (+1/-1); docs/platforms/cpu_server.md (+6/-1); python/sglang/srt/distributed/parallel_state.py (+4/-3); python/sglang/srt/layers/attention/intel_amx_backend.py (+3/-0); python/sglang/srt/layers/quantization/fp8.py (+3/-0); python/sglang/srt/layers/quantization/w8a8_int8.py (+5/-7); python/sglang/srt/managers/scheduler.py (+4/-5); python/sglang/srt/managers/scheduler_metrics_mixin.py (+1/-1); python/sglang/srt/model_executor/cpu_graph_runner.py (+640/-0); python/sglang/srt/model_executor/forward_batch_info.py (+3/-0); (+6 more)
LABELS: high priority, intel, cpu
BODY: ## Motivation ⏎  ⏎ Inspired by https://github.com/mingfeima/sglang/pull/73. We add CPU graph runner with torch compile to reduce python overhead to speed up decoding on CPU. ⏎  ⏎ Profiling with disabling torch compile: ⏎ <img width="1042" height="325" alt="image" src="https://github.com/user-attachments/assets/567b3d68-a53c-4f7e-b5ba-0b39276e68c7" /> ⏎  ⏎ Profiling with enabling torch compile: ⏎ <img width="812" height="278" alt="image" src="https://gith …[truncated]

### L2-0096798ed6  (L2, 2025-09-08, sha 0096798ed60b, PR #10156)
TITLE: [1/2] Speed up prefill mla attention (#10156)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.kernel.concat_mla
FILES: sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/csrc/elementwise/concat_mla.cu (+117/-0); sgl-kernel/csrc/common_extension.cc (+2/-0); sgl-kernel/include/sgl_kernel_ops.h (+1/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/python/sgl_kernel/elementwise.py (+8/-0)
BODY: ## Motivation ⏎  ⏎ in nsys I see bandwidth is ~90%, and it is not yet full SOL, thus may have a bit of room for further improvement left as future work if having time ⏎  ⏎ also if needed I will generalize the code (currently it is specifically for dsv3) ⏎  ⏎ UT code: https://gist.github.com/fzyzcjy/bda0b3a6c5fec889b477de192cfc1f55 ⏎ too ugly, I will polish and put it to bench & test folder after the ddl ⏎  ⏎ UT bench (only focus on the "torch" and "cuda"  …[truncated]

### L2-8085aca791  (L2, 2025-09-08, sha 8085aca7913f, PR #9925)
TITLE: [Bug fix] Fix ascend mla in aclgraph (#9925)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/ascend_backend.py (+1/-1)
BODY: ## Motivation ⏎ Fix inference of DeepSeek-R1-0528 with batch > 1 on Ascend devices ⏎  ⏎  ⏎ ## Modifications ⏎ manually contiguous query input to avoid autocontiguous bug of mla operator in aclgraph mode ⏎  ⏎  ⏎ ## Error Log ⏎ ``` ⏎   File "/home/miniconda3/envs/hzh_sglang/lib/python3.11/site-packages/torch_npu/npu/graphs.py", line 75, in graph_task_update_end ⏎   File "/home/miniconda3/envs/hzh_sglang/lib/python3.11/threading.py", line 982, in run ⏎     _gra …[truncated]

### L2-148022fc36  (L2, 2025-09-08, sha 148022fc36f0, PR #9522)
TITLE: gb200: update dockerfile to latest kernel (#9522)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+6/-9)
BODY: fp8 disagg working

### L2-94fb4e9e54  (L2, 2025-09-09, sha 94fb4e9e54ef, PR #10205)
TITLE: feat: support fa cute in sgl-kernel (#10205)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-0); sgl-kernel/CMakeLists.txt (+19/-0); sgl-kernel/python/sgl_kernel/_fa4_interface.py (+376/-0); sgl-kernel/python/sgl_kernel/flash_attn.py (+42/-0); sgl-kernel/tests/test_flash_attention_4.py (+877/-0)
BODY: ## Motivation ⏎  ⏎ - extract fa cute sgl-kernel part from https://github.com/sgl-project/sglang/pull/9928, authored by @cicirori  ⏎ - also unify interface for https://github.com/sgl-project/sglang/pull/9428 ⏎  ⏎ After this pull request, a new sgl-kernel release will be updated. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-d3ee70985f  (L2, 2025-09-09, sha d3ee70985f3a, PR #10220)
TITLE: chore: upgrade v0.3.9 sgl-kernel (#10220)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-5); docker/Dockerfile.gb200 (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci/ci_install_dependency.sh (+1/-1)
DEEP_STUDY: deep-study: this PR was reverted by PR 10245 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-bcf1955f7e  (L2, 2025-09-09, sha bcf1955f7e42, PR #10245)
TITLE: Revert "chore: upgrade v0.3.9 sgl-kernel" (#10245)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+5/-2); docker/Dockerfile.gb200 (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci/ci_install_dependency.sh (+1/-1)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 10220 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#10220 ⏎  ⏎ because of the failed test case https://github.com/sgl-project/sglang/actions/runs/17579411785/job/49948124294#step:4:2936

### L2-bfe01a5eef  (L2, 2025-09-11, sha bfe01a5eef40, PR #10297)
TITLE: chore: upgrade v0.3.9.post2 sgl-kernel (#10297)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-5); docker/Dockerfile.gb200 (+1/-1); python/pyproject.toml (+2/-2); .github/workflows/pr-test-pd-router.yml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci/ci_install_dependency.sh (+3/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-70c0c1f926  (L2, 2025-09-11, sha 70c0c1f9262f, PR #10330)
TITLE: fix: trtllm-gen attention take zero-init workspace (#10330)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+2/-7)
DEEP_STUDY: deep-study correctness case sglang:70c0c1f926: class=memory_safety_oob; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Motivation ⏎  ⏎ trtllm_gen_mla took empty-init decode_cuda_graph_workspace. Replaced by zero_init self.workspace_buffer = global_zero_init_workspace_buffer. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ `python3 -m sglang.launch_server --model /dev/shm/DeepSeek-R1-0528-FP4 --trust-remote-code --tp 8 --attention-backend trtllm_mla` ⏎  ⏎ ``` ⏎ python3 benchmark/gsm8k/bench_sglang.py \ ⏎     --num-shots 5 \ ⏎     --num-questions 1319 \ ⏎     --parallel …[truncated]

### L2-36acd2ff16  (L2, 2025-09-12, sha 36acd2ff16d9, PR #10180)
TITLE: Fix chunked prefix cache for nvfp4 (#10180)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+19/-1); python/sglang/srt/model_executor/model_runner.py (+2/-1); python/sglang/srt/models/deepseek_v2.py (+27/-0)
LABELS: high priority, ready-to-merge
BODY: co-authored by @elfiegg  ⏎  ⏎ ## Motivation ⏎ To address issue  [9806](https://github.com/sgl-project/sglang/issues/9806) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ server.sh ⏎ python3 -m sglang.launch_server \ ⏎   --model-path nvidia/DeepSeek-R1-0528-FP4 \ ⏎   --trust-remote-code \ ⏎   --attention-backend trtllm_mla \ or flashinfer \ ⏎   --disable-radix-cache \ ⏎   --max-running-requests 256 \ ⏎   --chunked-prefill-size 2048 \ ⏎   --mem-fraction- …[truncated]

### L2-cef11e9a55  (L2, 2025-09-12, sha cef11e9a552c, PR #10343)
TITLE: fix: resolve gb200 image link (#10343)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+10/-1)
LABELS: bug
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-16cd550c85  (L2, 2025-09-12, sha 16cd550c8554, PR #10379)
TITLE: Support Qwen3-Next on Ascend NPU (#10379)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.dispatch.server_args_defaults
FILES: docker/Dockerfile.npu (+8/-3); .github/workflows/release-docker-npu-nightly.yml (+1/-1); .github/workflows/release-docker-npu.yml (+1/-1); python/sglang/srt/layers/attention/fla/layernorm_gated.py (+1/-1); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+22/-4); python/sglang/srt/mem_cache/memory_pool.py (+12/-3); python/sglang/srt/model_executor/model_runner.py (+16/-5); python/sglang/srt/models/qwen3_next.py (+6/-3); python/sglang/srt/server_args.py (+2/-1); scripts/ci/npu_ci_install_dependency.sh (+10/-4)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-b047b553c2  (L2, 2025-09-14, sha b047b553c26e, PR #10157)
TITLE: [2/2] Speed up prefill mla attention concat (#10157)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+13/-2)
BODY: ## Motivation ⏎  ⏎ depend on https://github.com/sgl-project/sglang/pull/10156, and please read post there ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-5afd036533  (L2, 2025-09-15, sha 5afd0365334c, PR #10465)
TITLE: feat: support pip install sglang (#10465)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.npu (+1/-1); docker/Dockerfile.rocm (+1/-0); docker/Dockerfile.xeon (+1/-0); python/pyproject.toml (+88/-132); .github/workflows/pr-test-xeon.yml (+1/-0); python/pyproject_other.toml (+174/-0); scripts/ci/amd_ci_install_dependency.sh (+2/-0); scripts/ci/npu_ci_install_dependency.sh (+1/-0)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ - We should support `pip install sglang` on NVIDIA GPU. ⏎ - If we don't back up `pyproject_other.toml`, other platforms will encounter errors during installation. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-3b25dc127a  (L2, 2025-09-15, sha 3b25dc127ae7, PR #10473)
TITLE: [1/2] Speed up trtllm_mla attention backend (>10% e2e) (#10473)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.kernel.concat_mla
FILES: sgl-kernel/csrc/elementwise/concat_mla.cu (+102/-0); sgl-kernel/csrc/common_extension.cc (+3/-0); sgl-kernel/include/sgl_kernel_ops.h (+1/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/python/sgl_kernel/elementwise.py (+12/-0); test/srt/models/test_generation_models.py (+0/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ EDIT: 17.5% e2e speedup for short seq len, thus even for long seq len I think it will be >10%. ⏎  ⏎ code really ugly b/c ddl is near, will beautify later. also it is not SOL, thus should optimize later. ⏎ will also add test and bench later ⏎  ⏎ ``` ⏎    num_tokens     torch  torch_compiled      cuda ⏎ 0         1.0  0.010048        0.038592  0.006176 ⏎ 1        16.0  0.014336        0.038880  0.006144 ⏎ 2       128.0  0.055328        0.05 …[truncated]

### L2-c0c6f543e4  (L2, 2025-09-16, sha c0c6f543e472, PR #10500)
TITLE: chore: upgrade sgl-kernel 0.3.10 (#10500)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); docker/Dockerfile.gb200 (+1/-1); python/pyproject.toml (+4/-4); python/pyproject_other.toml (+3/-3); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-311de47bb7  (L2, 2025-09-16, sha 311de47bb7a8, PR #10474)
TITLE: [2/2] Speed up trtllm_mla attention backend (#10474)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+10/-2)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ see https://github.com/sgl-project/sglang/pull/10473 for info ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-a2f7218a2e  (L2, 2025-09-16, sha a2f7218a2e23, PR #9928)
TITLE: support using fa4 on deepseek on blackwell (#9928)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+12/-0); python/sglang/srt/layers/attention/hybrid_attn_backend.py (+1/-0); python/sglang/srt/model_executor/model_runner.py (+10/-0); python/sglang/srt/models/deepseek_v2.py (+3/-0); python/sglang/srt/server_args.py (+1/-0); python/sglang/srt/entrypoints/engine.py (+7/-0); sgl-kernel/python/sgl_kernel/_fa4_interface.py (+102/-0)
LABELS: high priority, run-ci
BODY: This is an initial PR for FA4 backend. ⏎  ⏎ 1. Get FA4 from the original repository into the sgl kernel and share the interface with FA3, controlled by the `ver` parameter. ⏎ 2. Add support for using FA4 as the prefill backend with dpsk-like model + Blackwell. ⏎  ⏎ Based on the current interface compatibility of Flash Attention 4,I chose to share the implementation with the FA3 attention backend.

### L2-e1d45bc280  (L2, 2025-09-16, sha e1d45bc280e4, PR #10529)
TITLE: Fix decord dependency for aarch64 docker build (#10529)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+16/-4); python/pyproject.toml (+2/-1); .github/workflows/release-docker-gb200.yml (+1/-1)
BODY: ## Motivation ⏎  ⏎ `pyproject.toml` was modified to include `decord` as a dependency. ⏎ However, `decord` does not have an ARM package, hence it's breaking `Dockerfile.gb200` build. ⏎  ⏎ ## Modifications ⏎  ⏎ This PR creates a new `blackwell_aarch64` build target, and excludes `decord` dependency for that build target. It also adds support for BRANCH_TYPE argument in `Dockerfile.gb200. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ This is only a CI change. No impact on accur …[truncated]

### L2-124097fc5b  (L2, 2025-09-16, sha 124097fc5b22, PR #10459)
TITLE: enable prefix cache with dp (#10459)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/model_executor/model_runner.py (+0/-12); python/sglang/srt/models/deepseek_v2.py (+0/-18)
LABELS: high priority, run-ci
BODY: co-authored by @elfiegg @rainj-me  ⏎ ## Motivation ⏎ This PR is to finalize the accuracy in #9806 and only serves as a verification of comment [here](https://github.com/sgl-project/sglang/issues/9806#issuecomment-3289234555) by @rainj-me. The final fix should only include #10414 and #10178 and revert of #9861  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ ------------------------------- ⏎ python3 -m sglang.launch_server \ ⏎   --model-path  …[truncated]

### L2-1ba137e98f  (L2, 2025-09-17, sha 1ba137e98fa2, PR #10526)
TITLE: Enable trtllm mla prefix extend (#10526)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+40/-5)
LABELS: ready-to-merge, run-ci
BODY: co-authored by @elfiegg  ⏎ ## Motivation ⏎ Depends on [10459](https://github.com/sgl-project/sglang/pull/10459). This PR is supposed to be merged after 10459. This PR enables TRTLLM_MLA backend for chunked prefix cache extend. There is a issue with the kernel that might write out-of-bound for such case: ⏎ ``` ⏎ actual_seq_lens_kv:tensor([1280,    0]) ⏎ ``` ⏎ Comparing the corresponding lse from flashinfer cutlass backend and trtllm_mla: ⏎ ``` ⏎ lse by tr …[truncated]

### L2-2a2ff9a840  (L2, 2025-09-18, sha 2a2ff9a8407f, PR #10629)
TITLE: refactor: use registry for _get_attention_backend_from_str (#10629)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+192/-0); python/sglang/srt/model_executor/model_runner.py (+3/-148)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ Right now, `_get_attention_backend_from_str` is implemented as a large if-elif chain. Every time we add a new backend, we need to modify this function, which is a textbook violation of the Open/Closed Principle (OCP) — the code is not extension-friendly. In particular, if we want to add a private backend, we have to patch this function, even though the actual change is minimal. Introducing a registry mechanism would eliminate thi …[truncated]

### L2-e7bc600304  (L2, 2025-09-18, sha e7bc600304e9, PR #9873)
TITLE: [Feature] Speculative decoding support lookahead (#9873)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/pyproject.toml (+2/-0); python/sglang/bench_serving.py (+2/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+39/-14); python/sglang/srt/managers/schedule_batch.py (+12/-3); python/sglang/srt/managers/scheduler.py (+18/-7); python/sglang/srt/managers/tokenizer_manager.py (+11/-0); python/sglang/srt/model_executor/cuda_graph_runner.py (+25/-0); python/sglang/srt/model_executor/model_runner.py (+1/-1); python/sglang/srt/server_args.py (+91/-6); python/sglang/srt/speculative/cpp_lookahead/.clang-format (+1/-0); (+20 more)
LABELS: enhancement, high priority, speculative-decoding, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ For large language model with good output locality, using lookahead can achieve significant performance improvement at a relatively low cost. This PR references [#2790 ](https://github.com/sgl-project/sglang/pull/2790). ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ This feature has been running in multiple of our [business scenarios](https://www.wenxiaobai.com/) for over two months, and its stability and acceleration effect h …[truncated]

### L2-3fa3c22ae2  (L2, 2025-09-19, sha 3fa3c22ae2c4, PR #10634)
TITLE: Fix fast decode plan for flashinfer v0.4.0rc1 and upgrade sgl-kernel 0.3.11 (#10634)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.backend.flashinfer_general_mla
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/pyproject_other.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+2/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+3/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ In #10587, updating flashinfer to v0.4.0rc1 will cause some bugs, which is caused by some extra arguments added in https://github.com/flashinfer-ai/flashinfer/pull/1661 and  https://github.com/flashinfer-ai/flashinfer/pull/1675. ⏎  ⏎ To solve this, we just need to pass the default values of these extra arguments in `fast_decode_plan` of flashinfer backend. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and  …[truncated]

### L2-6f993e8b9e  (L2, 2025-09-19, sha 6f993e8b9e6c, PR #10671)
TITLE: chore: cleanup docker image (#10671)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+4/-3); .github/workflows/release-docker-dev.yml (+1/-7); .github/workflows/release-docker.yml (+2/-10)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `lmsysorg/sglang:dev` -> cu129, all ⏎ `lmsysorg/sglang:v0.5.3` -> release version, all ⏎ `lmsysorg/sglang:v0.5.3-cu126` -> hopper version, all (It should also works on hopper cu128) ⏎  ⏎ We should remove `lmsysorg/sglang:v0.5.3-cu126` in the near future. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-60e2a7cead  (L2, 2025-09-19, sha 60e2a7cead86, PR #10679)
TITLE: [Auto Sync] Update model_runner.py (20250920) (#10679)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/model_runner.py (+20/-11)
LABELS: run-ci
BODY: Sync changes from commit `f90d017e`. ⏎  ⏎ **Relevant Files Changed:** ⏎ - python/sglang/srt/model_executor/model_runner.py ⏎  ⏎ --- ⏎  ⏎ *This is an automated PR created by a script.*

### L2-b17e67df36  (L2, 2025-09-19, sha b17e67df36e7, PR #10683)
TITLE: [Auto Sync] Update deepseek_v2.py (20250920) (#10683)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+156/-112)
LABELS: high priority, run-ci
BODY: Sync changes from commit `239b2577`. ⏎  ⏎ **Relevant Files Changed:** ⏎ - python/sglang/srt/models/deepseek_v2.py ⏎  ⏎ --- ⏎  ⏎ *This is an automated PR created by a script.*

### L2-2b7417bf6a  (L2, 2025-09-20, sha 2b7417bf6a9e, PR #10673)
TITLE: fix(disagg): fix sending KV cache in case of MLA for NIXL backend (#10673)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/nixl/conn.py (+1/-1)
LABELS: high priority, ready-to-merge, run-ci
ISSUES: #10381 [Bug] AttributeError: 'KVArgs' object has no attribute 'kv_head_num'
BODY: ## Motivation ⏎  ⏎ When running with the NIXL backend and MLA enabled, KV cache transfers were handled incorrectly.   ⏎ The logic only allowed transfers KV cache fully when `decode_tp_size == attn_tp_size`, which does not apply in the MLA case -- KV cache sent by slices. ⏎ This caused KV cache synchronization issues. ⏎  ⏎ The correct behavior is consistent with the Mooncake backend implementation (see [mooncake/conn.py#L680](https://github.com/sgl-proj …[truncated]

### L2-cba0d8c309  (L2, 2025-09-20, sha cba0d8c3090b, PR #10651)
TITLE: [Feature] Support deterministic inference with FA3 backend (#10651)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+17/-0); python/sglang/srt/server_args.py (+8/-6)
LABELS: run-ci
BODY: ## Motivation ⏎ Part of https://github.com/sgl-project/sglang/issues/10278 ⏎ Reference: [Defeating Non-Determinism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) ⏎  ⏎ Thanks to the earlier work from @Fridge003, @Edenzzzz, and @Qiaolin-Yu in the following PRs: ⏎ - FlashInfer: https://github.com/flashinfer-ai/flashinfer/pull/1675   ⏎ - SGLang FlashInfer: https://github.com/sgl-project/sglang/pull/10645   ⏎ - …[truncated]

### L2-b1bb8e7490  (L2, 2025-09-22, sha b1bb8e7490d1, PR #10281)
TITLE: Enables TRT-LLM backend to be used for target_verify (#10281)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+134/-66)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ This change allows the TRT-LLM MLA backend to be used for the `target_verify` step in MTP, which should improve performance.  ⏎  ⏎ ## Modifications ⏎  ⏎ - Enables `forward_extend` in the TRT-LLM MLA backend to be used for `target_verify` in MTP. ⏎ - Also adds code to update the KV cache in `forward_extend` which was previously missing. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ### GSM8k: ⏎  ⏎ [details omitted] ⏎  ⏎ ### GPQA-Diamond: ⏎  ⏎ [details omitted] ⏎  …[truncated]

### L2-d27a6f7092  (L2, 2025-09-22, sha d27a6f7092cf, PR #10130)
TITLE: [Feature] Add MLAProcess for DeepSeek MLA on NPU (#10130)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/layers/attention/ascend_backend.py (+6/-1); python/sglang/srt/layers/attention/npu_ops/mla_preprocess.py (+300/-0); python/sglang/srt/models/deepseek_v2.py (+32/-3); docs/platforms/ascend_npu.md (+6/-5); python/sglang/srt/layers/rotary_embedding.py (+20/-14); python/sglang/srt/utils.py (+4/-0); test/srt/ascend/test_ascend_deepep.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This pr introduced a new fused op that warps whole `forward_absorb_prepare` function in mla. Codes for this op is open-sourced [here](https://github.com/sgl-project/sgl-kernel-npu/pull/51), docs on the way. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Add a new fused op ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ <img width="1089" height="110" alt="image" src="https://github.com/user-attachments/assets/221cbc1c-e623-4cd3-a5a7-48a2f4549a4e" /> ⏎  ⏎  ⏎ ## Benchmar …[truncated]

### L2-063c3791fe  (L2, 2025-09-22, sha 063c3791fe2e, PR #10777)
TITLE: Fix trtllm_mla slow concat kernel in MTP (#10777)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+9/-5)
LABELS: run-ci
BODY: before ⏎  ⏎ <img width="583" height="112" alt="image" src="https://github.com/user-attachments/assets/acb04d1b-9ab3-4738-b490-1cdd77dd630f" /> ⏎  ⏎ after ⏎ (the longest purple one becomes the concat kernel) ⏎  ⏎ <img width="365" height="63" alt="image" src="https://github.com/user-attachments/assets/19d39b72-06b0-4eda-a5b1-bd2c28fa3532" /> ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ##  …[truncated]

### L2-1c82d9db28  (L2, 2025-09-22, sha 1c82d9db2846, PR #10705)
TITLE: feat: unify dockerfiles (#10705)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+38/-24); .github/workflows/release-docker-dev.yml (+110/-4); .github/workflows/release-docker-gb200.yml (+0/-5); .github/workflows/release-docker.yml (+61/-28); .github/workflows/release-whl-kernel.yml (+7/-1)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ SGLang should be the best across all platforms!! This PR unifies the current Dockerfile and Dockerfile.gb200 to stem from the same file. This simplifies maintenance for version bumps and makes CI/release easier. ⏎  ⏎ In a follow up PR we should ⏎ 1. test the sgl-kernel release -> pypi directly ⏎ 2. update the dockerfile to pip install it vs installing from the repo ⏎ 3. delete the gb200 dockerfile ⏎  ⏎  ⏎ # Tests

### L2-8c1ef0f914  (L2, 2025-09-23, sha 8c1ef0f91444, PR #10782)
TITLE: chore: upgrade sgl-kernel 0.3.12 (#10782)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/pyproject_other.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-b06db198ba  (L2, 2025-09-23, sha b06db198ba9e, PR #10783)
TITLE: followup: clean up dockerfiles and release yamls  (#10783)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.gb200 (+0/-369); .github/workflows/release-docker-gb200.yml (+0/-31)
LABELS: run-ci
BODY: 

### L2-ea338676b5  (L2, 2025-09-23, sha ea338676b502, PR #10770)
TITLE: Clean up server args (#10770)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+190/-238)
LABELS: run-ci
BODY: - Do not use "Step 1", "Step 2", ... , as it will need lots of changes if we insert something in the middle ⏎ - rename `model_specific_adjustments` -> `_handle_model_specific_adjustments` ⏎ - Sort the names and functions better ⏎ - Deprecate old arguments `enable_ep_moe`, `enable_deepep_moe`, ..

### L2-adba172fd1  (L2, 2025-09-24, sha adba172fd155, PR #10786)
TITLE: ci: free space on workers for build (#10786)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); .github/workflows/release-docker-dev.yml (+19/-58); .github/workflows/release-docker.yml (+2/-2)
LABELS: run-ci
BODY: WIP work with @zhyncs to wire up x86 runner ⏎  ⏎ After swapping to custom aarch64 runner - builds are fast and complete (https://github.com/sgl-project/sglang/actions/runs/17965314447/job/51096758220)

### L2-592ddf374f  (L2, 2025-09-26, sha 592ddf374fe4, PR #10944)
TITLE: Add simple docker file for B300 (#10944)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.b300 (+55/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Add simple docker file for B300. It uses the ngc pytorch container as the base because the pytorch release doesn't have B300 support yet. ⏎ DeepEP, DeepGemm etc will be enabled in the future. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Tested sgl-kernel tests. I'll do more e2e tests.  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-0c9174108a  (L2, 2025-09-28, sha 0c9174108abe, PR #10701)
TITLE: Unify SGL Kernel Releases (#10701)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+50/-12); .github/workflows/pr-test.yml (+13/-17); sgl-kernel/python/sgl_kernel/__init__.py (+178/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Dynamically load common_ops module for SM90 (compiled with fast math) and SM 100 (without fast math) during initialization based on compute architecture so that one SGLang kernel binary can cover all versions with different configurations.  ⏎  ⏎ CMake will compile two common_ops libraries and install them in two different locations. ⏎  ⏎ Update images used in CI to be 12.9. Remove the steps to build kernel for 12.4 and 12.8. ⏎  ⏎ # …[truncated]

### L2-42245551ef  (L2, 2025-09-28, sha 42245551efd8, PR #10543)
TITLE: [sgl-kernel] Optimize concat_mla_k kernel (#10543)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.concat_mla
FILES: sgl-kernel/csrc/elementwise/concat_mla.cu (+39/-41); benchmark/kernels/elementwise/benchmark_concat_mla.py (+198/-0); sgl-kernel/csrc/elementwise/utils.cuh (+72/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Inspired by https://github.com/sgl-project/sglang/pull/10473, try to improve concat_mla_k kernel's performance and add benchmark test. ⏎  ⏎ ``` ⏎ ➜  sglang_dev git:(refactor_concat_mla_absorb) ✗ python benchmark_concat_mla.py ⏎ W0925 03:04:04.925000 322 torch/fx/experimental/symbolic_shapes.py:6823] [0/0] _maybe_guard_rel() was called on non-relation expression Eq(s60, 1) | Eq(s69 - 128, s60) ⏎ vector-add-performance: ⏎    num_tokens …[truncated]

### L2-5942fdb480  (L2, 2025-09-29, sha 5942fdb48009, PR #11054)
TITLE: chore: upgrade cutedsl 4.2.1 (#11054)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/pyproject_other.toml (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-3a641d9085  (L2, 2025-09-29, sha 3a641d90857c, PR #11056)
TITLE: chore: upgrade sgl-kernel 0.3.13 (#11056)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/pyproject_other.toml (+2/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ ``` ⏎ pip install sgl-kernel --upgrade ⏎ ``` ⏎  ⏎ One wheel for all. ⏎  ⏎ cu124/cu126/cu128/cu129 hopper ⏎ cu128/cu129 blackwell ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2bc61dd194  (L2, 2025-09-30, sha 2bc61dd19460, PR #10816)
TITLE: Remove hybrid_linear_attn attention backend and refactor attention registry (#10816)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+35/-29); python/sglang/srt/model_executor/model_runner.py (+9/-7); python/sglang/srt/server_args.py (+1/-2)
LABELS: run-ci, nvidia, hopper
BODY: Remove hybrid_linear_attn attention backend and refactor attention registry, so full attention backend could be set through cli for Hybrid GDN models like Qwen3-Next. ⏎  ⏎ Co-authored with @yizhang2077  ⏎  ⏎ ``` ⏎ # Before ⏎ # Attention backend parameters from cli will be override into hybrid_linear_attn. ⏎ # Also user can not set backend for full attention through cli, the --attention-backend parameter will not take effect. ⏎ python3 -m sglang.bench_one …[truncated]

### L2-ac1f2928ae  (L2, 2025-10-01, sha ac1f2928aef9, PR #10760)
TITLE: feat: add fast_decode_plan from flashinfer, flashinfer to 0.4.0rc3 (#10760)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.backend.flashinfer_general_mla
FILES: python/pyproject.toml (+1/-1); python/pyproject_other.toml (+3/-3); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+44/-186)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ We maintain fast_decode_plan at FI side. C++ API changes wont break sgl on fast_decode_plan function with this PR. Might be removed after we adopt GPU-based planning. ⏎  ⏎ https://github.com/sgl-project/sglang/pull/10634 ⏎  ⏎ https://github.com/flashinfer-ai/flashinfer/issues/1720 ⏎ https://github.com/flashinfer-ai/flashinfer/pull/1745 ⏎ https://github.com/flashinfer-ai/flashinfer/pull/1757 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests …[truncated]

### L2-73d4a5f879  (L2, 2025-10-01, sha 73d4a5f8791f, PR #10735)
TITLE: Organize spec-related data structures (#10735)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.flashmla, L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.aiter_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+3/-3); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+11/-15); python/sglang/srt/layers/attention/flashmla_backend.py (+3/-3); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+3/-3); python/sglang/srt/constrained/outlines_jump_forward.py (+1/-1); python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py (+1/-1); python/sglang/srt/layers/attention/aiter_backend.py (+9/-14); python/sglang/srt/layers/attention/ascend_backend.py (+4/-4); python/sglang/srt/layers/attention/base_attn_backend.py (+3/-3); python/sglang/srt/layers/attention/flashattention_backend.py (+5/-6); (+22 more)
LABELS: high priority, run-ci
BODY: A step to merge #9334 into main and clarify all data structures for separate decoding/overlap scheduling. ⏎  ⏎ Depends on #10715 ⏎  ⏎ *** ⏎  ⏎ - Introduce `SpecInput` class to unify `EagleDraftInput`, `EagleVerifyInput`, `LookaheadVerifyInput`, and more speculative algorithms' inputs ⏎ - Introduce `SpecInputType` to avoid direct assert `isinstance` in different attention backends and runners. ⏎ - Separate eagle info types out of `eagle_utils`. ⏎ - Move cu …[truncated]

### L2-2d62af6be5  (L2, 2025-10-01, sha 2d62af6be517, PR #11123)
TITLE: Fix metrics and request tracing (TimeStats) (#11123)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+20/-21); python/sglang/srt/disaggregation/decode.py (+17/-3); python/sglang/srt/disaggregation/prefill.py (+7/-2); python/sglang/srt/disaggregation/utils.py (+1/-1); python/sglang/srt/managers/schedule_batch.py (+10/-10); python/sglang/srt/managers/schedule_policy.py (+8/-4); python/sglang/srt/managers/scheduler.py (+82/-89); python/sglang/srt/managers/scheduler_metrics_mixin.py (+96/-33); python/sglang/srt/managers/scheduler_output_processor_mixin.py (+3/-2); python/sglang/srt/managers/tokenizer_communicator_mixin.py (+81/-0); (+3 more)
LABELS: run-ci
BODY: 

### L2-44b1fbe258  (L2, 2025-10-01, sha 44b1fbe258e0, PR #11149)
TITLE: Fix DeepSeek chunked prefill memory issue (#11149)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ try to save 8G on someone's case ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-e810077488  (L2, 2025-10-02, sha e810077488e1, PR #11138)
TITLE: Allow use of TRTLLM_MHA backend for hybrid attention on Blackwell (#11138)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+2/-1); python/sglang/srt/model_executor/model_runner.py (+1/-1)
LABELS: high priority, blackwell, run-ci, nvidia
BODY: Allows use of "trtllm_mha" attention backend for hybrid attention e.g. Qwen3 Next. ⏎  ⏎ ## Motivation ⏎  ⏎ TRTLLM_MHA kernels exist in SGLang but could not be used in hybrid attention. ⏎  ⏎ ## Modifications ⏎  ⏎ Modifies assertion to reflect that the new kernels can be used, and fixes a bug where page size was erroneously set to 1 instead of the value passed to the attention operator. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ See comparison to Triton backend, below. Model …[truncated]

### L2-f35def8652  (L2, 2025-10-02, sha f35def8652a2, PR #10779)
TITLE: Fuse quantize and rope in trtllm_mla MTP (#10779)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+33/-4); python/sglang/srt/models/deepseek_v2.py (+4/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ future work (will be done in separate pr, since this pr speed up is nontrivial and needs to get merged soon) ⏎  ⏎ - refactor trtllm_mla_backend.py to avoid code duplication (currently temporarily make it duplicated to follow style of existing code ⏎ - remove other 2 small elementwise kernels ⏎  ⏎ before ⏎  ⏎ <img width="936" height="106" alt="image" src="https://github.com/user-attachments/assets/56b4b197-8bfe-4c8f-808a-9ec4bf5599bf" /> …[truncated]

### L2-b00a0c786f  (L2, 2025-10-02, sha b00a0c786fd0, PR #11161)
TITLE: [Fix] Update to v0.1.5.post4 and refine HIP attention backend selection (#11161)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.rocm (+1/-1); python/sglang/srt/model_executor/model_runner.py (+1/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ This PR updates the aiter backend to the latest commit (`v0.1.5.post4`)  to include the newest MLA kernel implementation supported on AMD MI355x GPUs and meanwhile update sglang interface support speculative decoding and fixes the logic for selecting the HIP attention backend.  Previously, the condition for enabling `aiter` was more complex and could lead to unexpected fallbacks. This change makes the condition explicit: aiter wi …[truncated]

### L2-b65db0287b  (L2, 2025-10-02, sha b65db0287b55, PR #11163)
TITLE: Tiny cleanup deepseek_v2.py (#11163)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-6); python/sglang/srt/models/deepseek_v2.py (+31/-32)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ cc @merrymercy  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-666da3d59f  (L2, 2025-10-04, sha 666da3d59fa6, PR #11012)
TITLE: [fix]enable flashmla when using draft model P/D attention select (#11012)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.flashmla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashmla_backend.py (+4/-2); python/sglang/srt/speculative/eagle_worker.py (+7/-0); test/srt/test_flashmla.py (+3/-3)
LABELS: run-ci
BODY: ## Motivation ⏎ 1. on #9755 , when using flashmla , `draft_extend_attn_backend` will report an error when initialized, which is inconsistent with the pre-refactoring version. So I added unimplemented select for `_create_backend ` . ⏎ 2. add mutile token support for mla ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ ``` ⏎ python3 bench_sglang.py  --num-shots 8 --num-questions 1319 --parallel 50 --port 8080 ⏎  ⏎ Accuracy: 0.967 ⏎ ``` ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ ## Ben …[truncated]

### L2-f8924ad74b  (L2, 2025-10-05, sha f8924ad74b87, PR #11242)
TITLE: update sgl kernel version to 0.3.14.post1 (#11242)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/pyproject_other.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: run-ci
BODY: 

### L2-292a867ad9  (L2, 2025-10-05, sha 292a867ad93b, PR #11235)
TITLE: Add flashmla and fast hadamard transform to Dockerfile (#11235)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+24/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ For the purpose of running Dpsk-v3.2 on main docker ⏎ Ref: #11061 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-efbc687c28  (L2, 2025-10-06, sha efbc687c2817, PR #11061)
TITLE: Support DeepSeek V3.2 Exp (#11061)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters, L2.pool.mla_token_kv, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/attention_registry.py (+7/-0); python/sglang/srt/layers/attention/npu_ops/mla_preprocess.py (+94/-1); python/sglang/srt/configs/model_config.py (+30/-0); python/sglang/srt/disaggregation/ascend/transfer_engine.py (+47/-9); python/sglang/srt/layers/attention/ascend_backend.py (+98/-4); python/sglang/srt/layers/attention/base_attn_backend.py (+9/-0); python/sglang/srt/layers/attention/hybrid_attn_backend.py (+7/-0); python/sglang/srt/layers/attention/nsa/dequant_k_cache.py (+163/-0); python/sglang/srt/layers/attention/nsa/index_buf_accessor.py (+354/-0); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+761/-0); (+19 more)
LABELS: enhancement, high priority, amd, speculative-decoding, blackwell, npu, run-ci
BODY: We now support DeepSeek V3.2. Try this model in the tracking issue #11060  ⏎  ⏎ --- ⏎  ⏎ Authors: General: @fzyzcjy @DarkSharpness @hnyls2002 @hebiao064 @Fridge003; AMD: @HaiShaw @kkHuang-amd @hubertlu-tw; NPU: @ZhengdQin

### L2-4aeb193fbd  (L2, 2025-10-06, sha 4aeb193fbd12, PR #11274)
TITLE: disable sm100 for FlashMLA and fast-hadamard-transform in cuda12.6.1 (#11274)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+4/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ disable sm100 for FlashMLA and fast-hadamard-transform in cuda12.6.1 ⏎  ⏎  ⏎ PR in fast-hadamard-transform to disable sm100 in cuda12.6.1 ⏎ https://github.com/Dao-AILab/fast-hadamard-transform/commit/7fd811c2b47f63b0b08d2582619f939e14dad77c ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-2fcd56eaf6  (L2, 2025-10-07, sha 2fcd56eaf6d7, PR #11303)
TITLE: [router] add get server info and get model info in grpc server (#11303)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/entrypoints/grpc_server.py (+90/-0); python/sglang/srt/grpc/sglang_scheduler.proto (+59/-0); python/sglang/srt/grpc/sglang_scheduler_pb2.py (+11/-3); python/sglang/srt/grpc/sglang_scheduler_pb2.pyi (+62/-0); python/sglang/srt/grpc/sglang_scheduler_pb2_grpc.py (+88/-0); sgl-router/src/grpc_client/sglang_scheduler.rs (+24/-0); sgl-router/src/proto/sglang_scheduler.proto (+59/-0)
LABELS: router, run-ci
BODY: ### Changes ⏎ 1. add new methods for get server info and get model info ⏎ 2. updates protobuf to support those new endpoints ⏎  ⏎ ### Note ⏎ internal state is purposely ignored in get server info ⏎ this will be streamed back in each token generation in the future ⏎  ⏎ ### Validations ⏎ ```bash ⏎ grpcurl -plaintext localhost:30000 sglang.grpc.scheduler.SglangScheduler/GetModelInfo ⏎ { ⏎   "model_path": "/raid/models/meta-llama/Llama-3.1-8B-Instruct", ⏎   "toke …[truncated]

### L2-4f42c8cd3e  (L2, 2025-10-07, sha 4f42c8cd3e65, PR #11068)
TITLE: [sgl-kernel] Support float64 moe_sum_reduce cuda kernel (#11068)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: sgl-kernel/benchmark/bench_sum_scale.py (+53/-19); sgl-kernel/csrc/moe/moe_sum_reduce.cu (+171/-71)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ The deterministic feature introduced float64 data type in MLA test. The current moe_sum_reduce cuda kernel does not cover this data type. So as when using this new cuda kernel like in https://github.com/sgl-project/sglang/pull/10654 , the following srt test failure occurred: ⏎  ⏎ ``` ⏎ python test/srt/hicache/test_hicache_mla.py ⏎ ... ⏎ [2025-09-29 02:36:08] INFO:     127.0.0.1:35010 - "POST /v1/chat/completions HTTP/1.1" 200 OK ⏎ [2 …[truncated]

### L2-edefab0c64  (L2, 2025-10-08, sha edefab0c6498, PR #10937)
TITLE: [2/2] Support MHA prefill with FlashAttention 4. (#10937)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/attention/attention_registry.py (+0/-3); python/pyproject_other.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+0/-1); python/sglang/srt/model_executor/model_runner.py (+3/-9); python/sglang/srt/server_args.py (+28/-7)
LABELS: high priority, run-ci
BODY: (co-author @hyhieu) ⏎  ⏎ Updates: the kernel change has been checked in separately in: https://github.com/sgl-project/sglang/pull/10940  ⏎  ⏎ ## Motivation ⏎  ⏎ Add support for FA4 MHA prefill, changes mostly based on: https://github.com/sgl-project/sglang/pull/9428  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ ``` ⏎ lm_eval \ ⏎   --model local-chat-completions \ ⏎   --model_args model=gpt-oss,base_url=http://127.0.0.1:30010/v1/chat/completions,num_concu …[truncated]

### L2-a3c2ea4451  (L2, 2025-10-08, sha a3c2ea445194, PR #11327)
TITLE: fix: fix revision for sgl-flash-attn in sgl-kernel (#11327)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ https://github.com/sgl-project/sgl-flash-attn/commit/ec36bf54b90fcaa36a7358317a8b44df8ce739e9 ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-d6837aea4d  (L2, 2025-10-09, sha d6837aea4d2c, PR #10909)
TITLE: model: Support Hybrid Mamba2 NemotronHForCausalLM (nvidia/NVIDIA-Nemotron-Nano-9B-v2) (#10909)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+31/-19); docs/supported_models/generative_models.md (+3/-2); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/falcon_h1.py (+12/-58); python/sglang/srt/configs/mamba_utils.py (+117/-0); python/sglang/srt/configs/nemotron_h.py (+286/-0); python/sglang/srt/configs/qwen3_next.py (+11/-43); python/sglang/srt/layers/attention/fla/layernorm_gated.py (+47/-30); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+165/-59); python/sglang/srt/layers/attention/mamba/causal_conv1d.py (+1/-1); (+25 more)
LABELS: high priority, run-ci
BODY: Support the `NemotronHForCausalLM` architecture, which can include any combination of *Mamba2*, MLP and normal self-attention layers. ⏎  ⏎ ## Motivation ⏎ Support the `NemotronHForCausalLM` architecture, which includes https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2: ⏎ > The model uses a hybrid architecture consisting primarily of Mamba-2 and MLP layers combined with just four Attention layers. For the architecture, please refer to the [Nemo …[truncated]

### L2-44cb060785  (L2, 2025-10-09, sha 44cb060785b8, PR #11364)
TITLE: chore: upgrade flashinfer 0.4.0 (#11364)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-1); python/pyproject_other.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci/ci_install_dependency.sh (+2/-0)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f19613e6c3  (L2, 2025-10-10, sha f19613e6c362, PR #10734)
TITLE: Dedicated toml files for CPU/XPU (#10734)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.xeon (+8/-5); .github/workflows/pr-test-xeon.yml (+4/-1); docs/platforms/cpu_server.md (+4/-3); python/pyproject_cpu.toml (+123/-0); python/pyproject_xpu.toml (+123/-0)
LABELS: ready-to-merge, intel, cpu, run-ci
BODY: ## Motivation ⏎  ⏎ Split the `.toml` files for `sglang` package, to better facilitate the planned per-device `sgl-whl` building. ⏎ Originated from #10162 . ⏎  ⏎ ## Modifications ⏎  ⏎ Created dedicated `.toml` files for CPU and Intel XPU. Dockerfile and document of CPU are updated correspondingly. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ N/A ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ N/A ⏎  ⏎ ## Checklist

### L2-4b15fa00f0  (L2, 2025-10-12, sha 4b15fa00f082, PR #11500)
TITLE: move fla env check position (#11500)
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+2/-0); python/sglang/srt/layers/attention/fla/utils.py (+0/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ avoid fla check env in unrelated models ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-1bdd010291  (L2, 2025-10-12, sha 1bdd010291e4, PR #11520)
TITLE: Revert "Deprecate `global_server_args_dict`" (#11520)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+5/-5); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+4/-4); python/sglang/global_config.py (+3/-0); python/sglang/srt/distributed/device_communicators/pynccl_allocator.py (+2/-2); python/sglang/srt/eplb/expert_location_dispatch.py (+2/-2); python/sglang/srt/eplb/expert_location_updater.py (+2/-2); python/sglang/srt/layers/attention/double_sparsity_backend.py (+2/-2); python/sglang/srt/layers/attention/flashattention_backend.py (+2/-2); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+2/-2); python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py (+2/-2); (+44 more)
LABELS: run-ci
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 11331 reason=unstated
BODY: Reverts sgl-project/sglang#11331

### L2-a2b3d9b90b  (L2, 2025-10-12, sha a2b3d9b90b92, PR #11512)
TITLE: Update DeepSeek-R1-FP4 default config on blackwell (#11512)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+26/-1)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f49419061d  (L2, 2025-10-12, sha f49419061ddf, PR #11332)
TITLE: Move args from `global_config` to `environ` (#11332)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+2/-2); python/sglang/global_config.py (+1/-22); python/sglang/srt/environ.py (+9/-0); python/sglang/srt/layers/attention/flashinfer_backend.py (+10/-10); python/sglang/srt/managers/schedule_batch.py (+4/-4); python/sglang/srt/managers/scheduler.py (+8/-8)
LABELS: run-ci
BODY: `global_config` is old-fashioned and should be deprecated.

### L2-1083e7e3df  (L2, 2025-10-13, sha 1083e7e3df1e, PR #11331)
TITLE: Deprecate `global_server_args_dict` (#11331)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+5/-5); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+4/-4); python/sglang/global_config.py (+0/-3); python/sglang/srt/distributed/device_communicators/pynccl_allocator.py (+2/-2); python/sglang/srt/eplb/expert_location_dispatch.py (+2/-2); python/sglang/srt/eplb/expert_location_updater.py (+2/-2); python/sglang/srt/layers/attention/double_sparsity_backend.py (+2/-2); python/sglang/srt/layers/attention/flashattention_backend.py (+2/-2); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+2/-2); python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py (+2/-2); (+44 more)
LABELS: run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 11520 (confirmed_revert, reason=unstated)
BODY: Deprecate the `global_server_args_dict`, just use `global_server_args: ServerArgs`. ⏎  ⏎ - Only the scheduler process needs it ⏎ - Set the global value in `ModelRunner` ⏎ - Only access this global value by `get_global_server_args()` to avoid reading before setting the arguments.

### L2-c9cff2b984  (L2, 2025-10-13, sha c9cff2b98477, PR #11557)
TITLE: Fix DeepSeek-v3.2 default config (ValueError: not enough values to unpack (expected 4, got 3)) (#11557)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+1/-1)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ Changes to DeepSeek-R1-FP4 default config in #11512 are accidently being applied to DeepSeek-V3.2-Exp. On blackwell, this causes the attention backend for 3.2 to be set to `trtllm_mla` instead of `nsa` which leads to this error at runtime: ⏎ ``` ⏎   File "/trevor/sglang/python/sglang/srt/models/deepseek_v2.py", line 1299, in forward ⏎     return self.forward_core(s) ⏎            ^^^^^^^^^^^^^^^^^^^^ ⏎   File "/trevor/sglang/python/sgl …[truncated]

### L2-8e51049f56  (L2, 2025-10-13, sha 8e51049f5614, PR #11538)
TITLE: [CI Monitor] Ci monitor only deal with main branch in default (#11538)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: scripts/ci_monitor/ci_analyzer.py (+13/-3); scripts/ci_monitor/ci_analyzer_perf.py (+35/-11)
LABELS: run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 11846 (confirmed_revert, reason=unstated)
BODY: <img width="755" height="243" alt="图片" src="https://github.com/user-attachments/assets/0c298290-5daf-433e-9a35-4459f75d346a" /> ⏎  ⏎  ⏎ Use `build-test (all)` as example: ⏎  ⏎ main: ⏎  ⏎ Last Success: Run ⏎  ⏎ <img width="575" height="282" alt="图片" src="https://github.com/user-attachments/assets/3534f035-47da-43f9-ad43-2a7b8d291b73" /> ⏎  ⏎ It's a fork branch ⏎  ⏎ pr: ⏎  ⏎ Last Success: Run ⏎  ⏎ <img width="525" height="253" alt="图片" src="https://github.com/user- …[truncated]

### L2-516738b096  (L2, 2025-10-13, sha 516738b0963d, PR #11528)
TITLE: Depreate `global_server_args_dict` (#11528)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+5/-5); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+4/-4); python/sglang/global_config.py (+0/-3); python/sglang/srt/distributed/device_communicators/pynccl_allocator.py (+2/-2); python/sglang/srt/eplb/expert_location_dispatch.py (+2/-2); python/sglang/srt/eplb/expert_location_updater.py (+2/-2); python/sglang/srt/layers/attention/double_sparsity_backend.py (+2/-2); python/sglang/srt/layers/attention/flashattention_backend.py (+2/-2); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+2/-2); python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py (+2/-2); (+44 more)
LABELS: run-ci
BODY: 

### L2-c224a4c6cc  (L2, 2025-10-14, sha c224a4c6cc46, PR #11624)
TITLE: Fix log for chunked prefix cache (#11624)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/model_runner.py (+14/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Currently when we launch DeepSeek with attention backends not supporting chunked prefix cache, the log will print "Chunked prefix cache is turned on."  by mistake. This Pr fixes this logging. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-dc965db0e0  (L2, 2025-10-14, sha dc965db0e0c7, PR #10721)
TITLE: make radix cache deterministic (#10721)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/managers/scheduler.py (+3/-5); python/sglang/srt/mem_cache/radix_cache.py (+63/-4); python/sglang/srt/server_args.py (+0/-7); python/sglang/srt/utils/common.py (+13/-0); python/sglang/test/test_deterministic.py (+2/-1)
LABELS: run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 11728 (confirmed_revert, reason=unstated)
BODY: ## Motivation ⏎  ⏎ The patch (tries) adding determinism to the radix cache. Part of https://github.com/sgl-project/sglang/issues/10278. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ The patch mainly modifies the match_prefix logic: ⏎  ⏎ - If it's called from the scheduler, always return the value to the multiply of SPLIT_SIZE. ⏎ - Otherwise, insert into the tree as normal. ⏎  ⏎ The algorithm is two-pass: ⏎  ⏎ - In the first pass, do the prefix matching as before. ⏎ - If  …[truncated]

### L2-5ea96ac7cc  (L2, 2025-10-14, sha 5ea96ac7ccd1, PR #11561)
TITLE: Reduce one step decode for draft model. (#11561)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.flashmla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.aiter_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+2/-2); python/sglang/srt/layers/attention/flashmla_backend.py (+2/-2); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+1/-1); python/sglang/srt/layers/attention/aiter_backend.py (+3/-3); python/sglang/srt/layers/attention/flashattention_backend.py (+2/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+2/-2); python/sglang/srt/layers/attention/triton_backend.py (+4/-3); python/sglang/srt/layers/attention/trtllm_mha_backend.py (+2/-2); python/sglang/srt/speculative/draft_utils.py (+222/-0); python/sglang/srt/speculative/eagle_worker.py (+13/-194)
LABELS: run-ci
BODY: This PR introduces the DraftBackendFactor class to dispatch the mult-step draft decode backend. In any case, the draft backend only initializes attention backends for `speculative_num_steps - 1` times.

### L2-4c03dbaaef  (L2, 2025-10-15, sha 4c03dbaaef70, PR #9493)
TITLE: [CI][XPU]enable sglang CI on Intel XPU (#9493)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.xpu (+78/-0); .github/workflows/pr-test-xpu.yml (+99/-0); python/sglang/srt/layers/rotary_embedding.py (+16/-2); python/sglang/test/test_utils.py (+5/-0); test/srt/run_suite.py (+8/-0); test/srt/xpu/test_intel_xpu_backend.py (+60/-0)
LABELS: intel, xpu, run-ci
BODY: ## Motivation ⏎  ⏎ This PR aims to enable the CI pipeline for Sglang targeting the XPU architecture. This approach enhances our ability to promptly gate and block potential issues before they affect the broader codebase. Key improvements and features introduced in this PR include: ⏎  ⏎ - Add XPU Test Coverage ⏎  ⏎ - Expand test suite to cover XPU-specific code paths ⏎  ⏎ - Trigger XPU CI by default ⏎  ⏎  ⏎ ## Summary by CodeRabbit ⏎  ⏎ - New Features ⏎   - Initial …[truncated]

### L2-b0d1d717e1  (L2, 2025-10-16, sha b0d1d717e178, PR #11728)
TITLE: Revert "make radix cache deterministic" (#11728)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/managers/scheduler.py (+5/-3); python/sglang/srt/mem_cache/radix_cache.py (+4/-63); python/sglang/srt/server_args.py (+7/-0); python/sglang/srt/utils/common.py (+0/-13); python/sglang/test/test_deterministic.py (+1/-2)
LABELS: run-ci
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 10721 reason=unstated
BODY: Reverts sgl-project/sglang#10721

### L2-3cceaa381a  (L2, 2025-10-16, sha 3cceaa381ad3, PR #11510)
TITLE: [Bugfix] Fix Qwen3/DSV3/DSV3.2 model support (#11510)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: .github/workflows/pr-test-npu.yml (+33/-13); .github/workflows/release-docker-npu-nightly.yml (+1/-1); .github/workflows/release-docker-npu.yml (+1/-1); docker/Dockerfile.npu (+10/-2); python/sglang/srt/layers/attention/ascend_backend.py (+17/-0); python/sglang/srt/mem_cache/allocator_ascend.py (+1/-1); python/sglang/srt/mem_cache/common.py (+1/-5); python/sglang/srt/model_executor/npu_graph_runner.py (+2/-2); python/sglang/srt/models/deepseek_v2.py (+1/-0); python/sglang/srt/server_args.py (+20/-0); (+2 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This pr fixes very models that failed to start on npu after dsv3.2 support pr was merged. ci and image release infra was also improved. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Fix mla prefill padding mismatch with actual seq lens in ascend_backend ⏎ - Fix dtype issue in AscendTokenToKVPool that breaks save kv ops ⏎ - Fix deepseek nsa condition ⏎ - Adapt default value of chunked_prefill_size and cuda_graph_max_bs ⏎ - Add packages to support  …[truncated]

### L2-627974405d  (L2, 2025-10-17, sha 627974405dd3, PR #11685)
TITLE: [Lint] Add `python/sglang` to ruff F401 checks and remove unused imports in files (#11685)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/npu_ops/mla_preprocess.py (+1/-1); .pre-commit-config.yaml (+3/-3); python/sglang/srt/_custom_ops.py (+1/-1); python/sglang/srt/compilation/cuda_piecewise_backend.py (+0/-1); python/sglang/srt/configs/deepseekvl2.py (+0/-1); python/sglang/srt/configs/dots_vlm.py (+2/-7); python/sglang/srt/configs/falcon_h1.py (+1/-6); python/sglang/srt/configs/qwen3_next.py (+0/-1); python/sglang/srt/connector/remote_instance.py (+1/-1); python/sglang/srt/disaggregation/ascend/transfer_engine.py (+1/-1); (+141 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ This PR adds F401 to ruff checks and removes unused imports accross `python/sglang`, excluding `__init__.py`. ⏎  ⏎ ## Modifications ⏎  ⏎ - Added `F821` to ruff checks - this catches undefined names/missing imports ⏎ - Added `python/sglang` to ruff checks, excluded all __init__.py ⏎ - Tried to fix all unused imports ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-f4488e9dd9  (L2, 2025-10-18, sha f4488e9dd9ae, PR #11801)
TITLE: set default attention backend for deterministic inference (#11801)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+17/-2); python/sglang/srt/utils/common.py (+9/-0)
LABELS: run-ci, deterministic
BODY: ## Motivation ⏎ Set default deterministic compatible attention backend when deterministic enabled and no attention backend being set. ⏎  ⏎ Tested on a single H100 ⏎ Before: ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path /shared/public/elr-models/Qwen/Qwen3-8B/2069b3fae1114555f3c020c81410e51fa0f656f2 --enable-deterministic-inference ⏎ /home/jobuser/zminglei/sglang/venv/lib/python3.10/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pyn …[truncated]

### L2-f440baa136  (L2, 2025-10-18, sha f440baa136e9, PR #11540)
TITLE: [Feature] Reuse flashinfer workspace for PD-Multiplexing. (#11540)
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.attention_registry, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/attention_registry.py (+3/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+9/-1); python/sglang/srt/model_executor/model_runner.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ When PD-Multiplexing enabled, there are multiple decode attention backends. These backends won't be active at the same time, they can share a flashinfer workspace. Subsequent PRs of PD-Multiplexing can be found  [here](https://github.com/sgl-project/sglang/issues/10813). ⏎ ## Modifications ⏎  ⏎  ⏎ Add interface to reuse flashinfer workspace. ⏎  ⏎  ⏎ ## Checklist

### L2-3b80232d06  (L2, 2025-10-19, sha 3b80232d0669, PR #11815)
TITLE: [DeepseekV32] Add fast_topk_transform_ragged_fused kernel (#11815)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/common_extension.cc (+4/-0); sgl-kernel/csrc/elementwise/topk.cu (+81/-8); sgl-kernel/include/sgl_kernel_ops.h (+11/-6); sgl-kernel/python/sgl_kernel/__init__.py (+6/-1); sgl-kernel/python/sgl_kernel/top_k.py (+24/-1); sgl-kernel/tests/test_topk.py (+75/-4)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Add a fused kernel for `fast_topk_transform_ragged_fused`. The difference between this kernel and  `fast_topk_transform_fused` is that `fast_topk_transform_fused` outputs indices into the paged kvcache and `fast_topk_transform_ragged_fused` outputs indices into the ragged kv that's the input to the flashmla_prefill kernel. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Tested with https://github.com/sgl-project/sglang/pull/11655. ⏎ Before ⏎ Repeat: 4, me …[truncated]

### L2-efa473348b  (L2, 2025-10-19, sha efa473348bc9, PR #11652)
TITLE: [Spec Decoding] Support MTP for dsv3.2 (#11652)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/configs/model_config.py (+5/-1); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+23/-10); python/sglang/srt/layers/attention/nsa_backend.py (+385/-68); python/sglang/srt/speculative/draft_utils.py (+16/-0); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+8/-0); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+8/-0)
LABELS: high priority, ready-to-merge, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎  ⏎ Based on https://github.com/sgl-project/sglang/pull/11109 ⏎ We have implemented MTP support for DS v3.2 and ***cuda graph*** in our in-house maintained version of sglang. ⏎ Since the community has completed the MTP modification, we are ready to contribute this feature back to the community. ⏎  ⏎ `python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --attention-backend  nsa --nsa-prefill flashmla_prefill - …[truncated]

### L2-d383e6616e  (L2, 2025-10-19, sha d383e6616e51, PR #11396)
TITLE: [Model] Add Olmo 3 model support (#11396)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs/supported_models/generative_models.md (+1/-0); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/olmo3.py (+105/-0); python/sglang/srt/models/olmo2.py (+31/-4); python/sglang/srt/server_args.py (+26/-0); python/sglang/srt/utils/common.py (+1/-0); python/sglang/srt/utils/hf_transformers_utils.py (+2/-0); test/srt/models/test_generation_models.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Olmo3 will be released soon and has support in other libraries like transformers and vllm. This PR adds support for Olmo3 to sglang. ⏎  ⏎ @zhaochenyang20 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ This PR modifies the Olmo2 implementation to support Olmo3. This approach matches what was done for vllm, since the architectural changes from Olmo2 are fairly small: sliding window attention is now used for 3 out of 4 layers, and RoPE scaling is not …[truncated]

### L2-4fff1ec1d9  (L2, 2025-10-20, sha 4fff1ec1d9ad, PR #11147)
TITLE: Deterministic Mode: Add 1-stage triton kernel for prefill (#11147)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/triton_backend.py (+135/-0); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+539/-44); python/sglang/srt/server_args.py (+2/-2); python/sglang/test/test_deterministic.py (+3/-0); test/srt/test_triton_attention_kernels.py (+200/-0)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎ Part of this Issue: https://github.com/sgl-project/sglang/issues/10278 ⏎ Related PR: https://github.com/sgl-project/sglang/pull/10721 ⏎  ⏎ Inspired from @ispobock  ⏎  ⏎ Co-authored with @zminglei @byjiang1996  ⏎  ⏎ Before this PR: ⏎ ``` ⏎               K (Keys, total 9 tokens) ⏎         ╔═══════════════════╦═════════════════╗ ⏎         ║   Prefix (0-4)    ║  Extend (5-8)   ║ ⏎         ║   [from cache]    ║  [new tokens]   ║ ⏎     ════╬═════════ …[truncated]

### L2-b113c72e7a  (L2, 2025-10-21, sha b113c72e7add, PR #10656)
TITLE: Init attention backend for Intel XPU (#10656)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: docker/Dockerfile.xpu (+2/-7); python/sglang/srt/layers/attention/attention_registry.py (+7/-0); Makefile (+3/-1); docs/advanced_features/attention_backend.md (+8/-0); docs/index.rst (+1/-0); docs/platforms/xpu.md (+92/-0); python/pyproject_xpu.toml (+5/-3); python/sglang/bench_one_batch.py (+1/-1); python/sglang/srt/distributed/parallel_state.py (+4/-2); python/sglang/srt/layers/attention/fla/layernorm_gated.py (+3/-1); (+8 more)
LABELS: intel, xpu, run-ci
BODY: ## Motivation ⏎  ⏎ Add an attention backend for Intel XPU based on [sgl-kernel-xpu](https://github.com/sgl-project/sgl-kernel-xpu) ⏎ Depend on https://github.com/sgl-project/sglang/pull/10248 ⏎  ⏎ ## Modifications ⏎  ⏎ and clean some chores ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-d9a20fd28a  (L2, 2025-10-21, sha d9a20fd28ae6, PR #11664)
TITLE: Use trtllm_mla decode kernel for draft extend in speculative decoding (#11664)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+348/-18); python/sglang/test/attention/test_trtllm_mla_backend.py (+172/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ <img width="800" height="235" alt="image" src="https://github.com/user-attachments/assets/6a619bfe-231f-4054-b9bd-61b11531ed50" /> ⏎ Before this pr, draft extend is using flashinfer_mla kernel. However, since flashinfer_mla doesn't support fp8, when using fp8 kv cache, it needs to dequantize kv to bf16, which is very slow (shown in figure). ⏎  ⏎ After this pr, ⏎ <img width="621" height="149" alt="image" src="https://github.com/us …[truncated]

### L2-ebff4ee648  (L2, 2025-10-21, sha ebff4ee64836, PR #11844)
TITLE: Update sgl-kernel and remove fast hadamard depedency (#11844)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: docker/Dockerfile (+1/-9); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+1/-1); scripts/ci/ci_install_dependency.sh (+0/-8)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Follow up of  #11663 and #11733 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-ef4a8097b8  (L2, 2025-10-21, sha ef4a8097b8e1, PR #11876)
TITLE: Rename flashmla kernel options of nsa backend for better readability (#11876)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+19/-21); python/sglang/srt/server_args.py (+10/-10); docs/advanced_features/server_arguments.md (+2/-0)
LABELS: run-ci
BODY: ## Motivation ⏎ - Rename `--nsa-prefill` and `--nsa-decode` to `--nsa-prefill-backend` and `--nsa-decode-backend` ⏎ - Add  documents for `--nsa-prefill-backend` and `--nsa-decode-backend` arguments ⏎ - Rename `flashmla_prefill` and `flashmla_decode` kernel choices to `flashmla_sparse` and `flashmla_kv`, so misunderstanding can be avoided. ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-9792b9d7e3  (L2, 2025-10-21, sha 9792b9d7e368, PR #11933)
TITLE: chore: upgrade flashinfer 0.4.1 (#11933)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L2-23afdfd1c2  (L2, 2025-10-21, sha 23afdfd1c2b1, PR #11717)
TITLE: [sgl-kernel] support flashmla libtorch (#11717)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: sgl-kernel/CMakeLists.txt (+24/-15); sgl-kernel/cmake/flashmla.cmake (+60/-0); sgl-kernel/csrc/flashmla_extension.cc (+46/-0); sgl-kernel/python/sgl_kernel/flash_mla.py (+126/-0); sgl-kernel/include/sgl_kernel_ops.h (+45/-0); sgl-kernel/tests/test_flashmla.py (+518/-0)
LABELS: high priority, ready-for-review, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ support flashmla libtorch ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist
