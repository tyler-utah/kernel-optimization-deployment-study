### L1-8465f035d1  (L1, 2025-04-29, sha 8465f035d120, PR #5859)
TITLE: Add qwen3 30b fused moe config (#5859)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=384,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0)
BODY: ## Motivation ⏎ <img width="1116" alt="image" src="https://github.com/user-attachments/assets/0c519d3a-6a86-474d-a3da-58c0f7235682" /> ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-91dda4cd06  (L1, 2025-04-29, sha 91dda4cd067d, PR #5880)
TITLE: Add A800 fused moe config for qwen3 30b (#5880)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=768,device_name=NVIDIA_A800-SXM4-80GB.json (+146/-0)
BODY: ## Motivation ⏎ configs include TP1 for qwen3-30b-bfloat16 ⏎  ⏎ ## Performance ⏎ | Input-Output Length      |    Before Output Token Throughput (tok/s)    |    After Output Token Throughput (tok/s) | ⏎ |:------------:|:-------------:|:-----------:| ⏎ | 1000-2000        |     1192.96      |       1240.01 | ⏎ | 5000-1000        |     592.96       |       615.98 | ⏎ | 10000-500        |     279.48       |       298.77 | ⏎  ⏎ ## Modifications ⏎ Add A800 fused m …[truncated]

### L1-1468769bde  (L1, 2025-04-29, sha 1468769bde6f, PR #)
TITLE: [Misc] add service discovery for sgl router
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py
PR_RECORD: missing (use git/gh if needed)
BODY: 

### L1-f4c191a712  (L1, 2025-04-29, sha f4c191a712f8, PR #5894)
TITLE: chore: update Dockerfile (#5894)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+5/-8); .github/workflows/release-docker.yml (+3/-3)
BODY: ## Motivation ⏎  ⏎ The GPU installed by default on Lambda is cu128. If running with the image lmsysorg/sglang:latest, an error will occur due to the version issue of nccl that comes with torch 2.6, which needs to be upgraded to the latest version. There doesn't seem to be a need for cu118 at the moment, and in fact cu121 and cu125 were using the same cu124 image before. The presence of srt and all often makes it difficult for people to distinguish  …[truncated]

### L1-dd408ee481  (L1, 2025-04-29, sha dd408ee4815c, PR #5793)
TITLE: Auto set draft model path for MTP (#5793)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+7/-0); python/sglang/srt/managers/tp_worker.py (+1/-0); python/sglang/srt/model_executor/model_runner.py (+11/-2); python/sglang/srt/models/deepseek_nextn.py (+1/-257); python/sglang/srt/models/deepseek_v2.py (+74/-17); python/sglang/srt/server_args.py (+21/-11)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ - load nextn layer weights directly from target model ⏎ - refactor to reuse common weight loading logic for draft model and target model ⏎ - auto set draft model path if not given ⏎  ⏎ With this improvement, we don't need to specify `--speculative-draft-model-path` for DeepSeek-V3/R1 MTP since we load weights of the nextn layer directly from the original model (no extra export operation needed).  ⏎ This change is compatible with the p …[truncated]

### L1-cc4a80caf6  (L1, 2025-04-29, sha cc4a80caf604, PR #5830)
TITLE: [PD] Fix Assertion failed: /DeepEP/csrc/kernels/internode.cu:483, condition: ibgda_get_state()->num_rc_per_pe >= num_channels #134 (#5830)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+1/-3)
ISSUES: #134 Assertion failed: /DeepEP/csrc/kernels/internode.cu:483, condition: ibgda_get_state()->num_rc_per_pe >= num_channels
BODY: ## Motivation ⏎  ⏎ Fix https://github.com/deepseek-ai/DeepEP/issues/134

### L1-799789afed  (L1, 2025-04-29, sha 799789afedcf, PR #5870)
TITLE: Bump Flashinfer to 0.2.5 (#5870)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); .github/workflows/pr-test.yml (+0/-2); docs/start/install.md (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+107/-82); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+27/-16)
BODY: ## Motivation ⏎ This pull requests is just little modification on the basis of #5538, which fixes conflicts and lints. ⏎ Thanks @AkazaAkane for contribution! ⏎  ⏎ Ref: #5855, #4905, #5023 ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1698e94e67  (L1, 2025-04-29, sha 1698e94e676e, PR #5900)
TITLE: Add A800 fused moe config for qwen3 235b (#5900)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=128,N=192,device_name=NVIDIA_A800-SXM4-80GB.json (+146/-0)
BODY: ## Motivation ⏎ configs include TP8 for qwen3-235b-bfloat16 ⏎  ⏎ ## Performance ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path /path/to/Qwen3-235B-A22B --host 0.0.0.0 --port 30000  --tp 8 ⏎ ``` ⏎  ⏎ | Input-Output Length      |    Before Output Token Throughput (tok/s)    |    After Output Token Throughput (tok/s) | ⏎ |:------------:|:-------------:|:-----------:| ⏎ | 1000-2000        |     1048.28       |       1050.01 | ⏎ | 5000-1000        |     55 …[truncated]

### L1-e330f2b86c  (L1, 2025-04-30, sha e330f2b86cd2, PR #5917)
TITLE: [qwen3] support qwen3 ep moe (#5917)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen2_moe.py (+8/-3); python/sglang/srt/models/qwen3_moe.py (+8/-3)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Performance ⏎ Test in 8xH200 ⏎ Server launch and benchmarking command line: ⏎ ``` ⏎ python3 -m sglang.launch_server --model Qwen/Qwen3-235B-A22B-FP8  --tp 8  --reasoning-parser qwen3 --quantization fp8 --enable-ep-moe ⏎  ⏎ python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 50 --request-rate 10 --dataset-name random --random-input-len 1000 --random-output-len 2000 --random-range-ratio 1 ⏎ `` …[truncated]

### L1-d353d08b4e  (L1, 2025-04-30, sha d353d08b4e89, PR #5932)
TITLE: chore: bump sgl-kernel 0.1.1 (#5932)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9a6ad8916d  (L1, 2025-04-30, sha 9a6ad8916dcd, PR #5933)
TITLE: chore: upgrade sgl-kernel 0.1.1 (#5933)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); .github/workflows/vllm-dependency-test.yml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/__init__.py (+2/-2); python/sglang/srt/model_executor/model_runner.py (+6/-3); python/sglang/srt/utils.py (+8/-5); scripts/ci_install_dependency.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-11383cec3c  (L1, 2025-04-30, sha 11383cec3c08, PR #5724)
TITLE: [PP] Add pipeline parallelism (#5724)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/bench_one_batch.py (+2/-0); python/sglang/srt/entrypoints/engine.py (+36/-19); python/sglang/srt/layers/dp_attention.py (+5/-2); python/sglang/srt/layers/utils.py (+35/-0); python/sglang/srt/managers/data_parallel_controller.py (+52/-34); python/sglang/srt/managers/schedule_batch.py (+25/-15); python/sglang/srt/managers/scheduler.py (+262/-59); python/sglang/srt/managers/scheduler_output_processor_mixin.py (+1/-1); python/sglang/srt/managers/tp_worker.py (+50/-16); python/sglang/srt/managers/tp_worker_overlap_thread.py (+9/-3); (+15 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8ebde73f7d  (L1, 2025-05-03, sha 8ebde73f7d00, PR #5998)
TITLE: [perf] H100 DeepSeek-V3 fused moe tuned config (#5998)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c9abd7be01  (L1, 2025-05-06, sha c9abd7be011f, PR #5885)
TITLE: fix: deepep dockerfile, use pip install deepep. (#5885)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.deepep (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎ fix "NotImplementedError: Support for egg-based install has been removed." when building Dockerfile.deepep ⏎  ⏎ ``` ⏎  => ERROR [30/31] RUN NVSHMEM_DIR=/sgl-workspace/nvshmem/install python setup.py install                                                                                                                                     2.8s ⏎ ------                                                                                    …[truncated]

### L1-8a828666a3  (L1, 2025-05-06, sha 8a828666a3a9, PR #5655)
TITLE: Add DeepEP to CI PR Test (#5655)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/release-docker-deepep.yml (+36/-0); .github/workflows/pr-test.yml (+3/-3); python/sglang/test/test_deepep_utils.py (+219/-0); python/sglang/test/test_utils.py (+1/-0); scripts/ci_install_dependency_8_gpu.sh (+122/-0); test/srt/run_suite.py (+3/-0); test/srt/test_deepep_internode.py (+445/-0); test/srt/test_deepep_intranode.py (+379/-0); test/srt/test_deepep_low_latency.py (+325/-0); test/srt/test_moe_deepep_eval_accuracy_large.py (+74/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Ensure DeepEP work well with PR run, add intranode / low_latency test from deepseek-ai/DeepEP as well as DeepSeek-V3 DeepEP accuracy test. ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b70957fcf8  (L1, 2025-05-07, sha b70957fcf86a, PR #5993)
TITLE: [refactor] slightly tidy fp8 module (#5993)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+2/-5); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+2/-4); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py (+2/-1); python/sglang/srt/layers/quantization/fp8.py (+106/-91); python/sglang/srt/layers/quantization/fp8_kernel.py (+73/-55); python/sglang/srt/layers/quantization/fp8_utils.py (+36/-23); python/sglang/srt/layers/quantization/kv_cache.py (+3/-10); python/sglang/srt/layers/quantization/utils.py (+0/-5); python/sglang/srt/layers/quantization/w8a8_fp8.py (+8/-10); python/sglang/srt/models/deepseek_nextn.py (+1/-20); (+2 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ experimantal, waiting for CI ⏎  ⏎ To reduce the complexity of the core code and minimize the intrusiveness of cross-platform logic, we’ll extract the HIP-platform-specific logic—which currently comprises the largest portion of lines in the FP8 quantization module and unify the datatype logic ⏎  ⏎ Suggestion: Separate W4A8-FP8 into a stand-alone module ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-73600673bb  (L1, 2025-05-07, sha 73600673bb1d, PR #6079)
TITLE: Clean logs for DeepSeek-V3 launching (#6079)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+4/-1); python/sglang/srt/distributed/device_communicators/pynccl.py (+2/-1); python/sglang/srt/layers/quantization/fp8.py (+2/-4); python/sglang/srt/layers/quantization/fp8_kernel.py (+4/-3); python/sglang/srt/model_executor/model_runner.py (+35/-26); python/sglang/srt/models/deepseek_v2.py (+7/-4); python/sglang/srt/utils.py (+7/-0)
BODY: ## Motivation ⏎  ⏎ When launching Deepseek-V3 model, some of the logs only need to be printed once (on tp_rank=0).  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ This PR adds a `should_rank` util function. And print these logs only when `should_rank()` is true.  ⏎ Only logging behavior is affected by this PR. ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4c7b42424c  (L1, 2025-05-07, sha 4c7b42424cf5, PR #5014)
TITLE: Hint users DeepEP normal mode is incompatible with CUDA Graph (#5014)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+3/-0)
BODY: ## Motivation ⏎  ⏎ Otherwise users will see an error like: ⏎  ⏎ [details omitted] ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-cfca4e0ed2  (L1, 2025-05-07, sha cfca4e0ed2cf, PR #6111)
TITLE: adding Triton configs for DeepSeekV3 FusedMoE kernel on Blackwell (#6111)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
ISSUES: #6095 [Feature] Tune fp8 Gemm and fused moe kernel on B200
BODY: ## Motivation ⏎  ⏎ close #6095 ⏎  ⏎  ⏎  ⏎ ## Profile results ⏎ On batch size 16: ⏎ ```bash ⏎ python3 -m sglang.bench_one_batch --model-path /dev/shm/DeepSeek-V3 --tp 8 --batch 16 --input-len 1024 --output-len 2 --attention-backend triton --profile ⏎ ``` ⏎  ⏎ FusedMoE decoding latency before tuning: ⏎ ![截屏2025-05-07 23 04 06](https://github.com/user-attachments/assets/a5f19035-159a-4da5-8979-29c5efba78d5) ⏎  ⏎ FusedMoE decoding latency after tuning: ⏎ ![截屏2025-05 …[truncated]

### L1-acc816d8a2  (L1, 2025-05-08, sha acc816d8a24e, PR #5626)
TITLE:  DeepEP normal support deepgemm-contiguous (#5626)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+340/-2); python/sglang/srt/layers/moe/ep_moe/layer.py (+120/-1); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+97/-54); python/sglang/srt/layers/quantization/deep_gemm.py (+5/-0); python/sglang/srt/layers/quantization/fp8_kernel.py (+2/-2); python/sglang/srt/models/deepseek_v2.py (+4/-0)
BODY: ## Motivation ⏎ DeepEP normal support deepgemm-contiguous ⏎ The contiguous mode has passed debugging in normal mode, just finished testing on gsm8k, and accuracy is okay.  ⏎ ## Modifications ⏎  ⏎ ## TODO: ⏎ * There are still some details in code style that need adjustment. ⏎ * If fp8 quantization is performed before dispatch, a deep ep buffer error occurs. I will troubleshoot this later. ⏎ * Compatibility testing for auto mode. ⏎ * Performance testing --  …[truncated]

### L1-6578cf27de  (L1, 2025-05-08, sha 6578cf27de8b, PR #6131)
TITLE: chore: bump sgl-kernel 0.1.2 (#6131)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-17c36c5511  (L1, 2025-05-10, sha 17c36c55117e, PR #6186)
TITLE: [CI] Disabled deepep tests temporarily because it takes too much time. (#6186)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+22/-26); test/srt/run_suite.py (+5/-3)
BODY: We can re-enable them after doing the following enhancements. ⏎  ⏎ 1. Do not compile from scratch every time. Utilize the compilation cache. ⏎ 2. Use a smaller model `lmsys/sglang-ci-dsv3-test` for most tests ⏎ 3. Call `scripts/ci_install_dependency.sh` in `scripts/ci_install_dependency_8_gpu.sh` to reduce code duplication. ⏎  ⏎ The final goal is to run the 8-gpu test within 15 mins.

### L1-45b4dcf037  (L1, 2025-05-11, sha 45b4dcf0375a, PR #6195)
TITLE: chore: bump sgl-kernel v0.1.2.post1 (#6195)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2ce8793519  (L1, 2025-05-11, sha 2ce8793519af, PR #6179)
TITLE: Add typo checker in pre-commit (#6179)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.ep.layer, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+1/-1); python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-1); python/sglang/srt/layers/moe/topk.py (+1/-1); .pre-commit-config.yaml (+6/-0); 3rdparty/amd/tuning/TUNING.md (+1/-1); benchmark/hicache/bench_serving.py (+2/-2); benchmark/json_schema/bench_sglang.py (+2/-2); benchmark/line_retrieval/gen_data.py (+1/-1); benchmark/multi_document_qa/bench_other.py (+1/-1); (+89 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Fix typos everywhere ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Fix and add pre-commit the common https://github.com/codespell-project/codespell ⏎  ⏎ ## Checklist

### L1-e9a47f4cb5  (L1, 2025-05-11, sha e9a47f4cb58a, PR #6198)
TITLE: Add dev-deepep docker image (#6198)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/release-docker-dev-deepep.yml (+36/-0); docker/Dockerfile.dev-deepep (+80/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-230106304d  (L1, 2025-05-11, sha 230106304db3, PR #6196)
TITLE: chore: upgrade sgl-kernel v0.1.2.post1 (#6196)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/deep_gemm.py (+57/-67); scripts/ci_install_dependency.sh (+1/-1); scripts/ci_install_dependency_8_gpu.sh (+1/-1)
BODY: ## Motivation ⏎  ⏎ Merged With PR https://github.com/sgl-project/sglang/pull/6194 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-198b9056d1  (L1, 2025-05-14, sha 198b9056d18c, PR #6274)
TITLE: [AMD] Fix Llama 4 Scout and Maverick accuracy issues on MI300X (#6274)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+13/-0)
BODY: ## Motivation ⏎  ⏎ Fix the following models on AMD GPUs. ⏎ - meta-llama/Llama-4-Scout-17B-16E-Instruct ⏎ - meta-llama/Llama-4-Maverick-17B-128E-Instruct ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ - Fix `FusedMoE `with `SGLANG_AITER_MOE=1` path when `apply_router_weight_on_input=True` ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ### Llama-4-Scout-17B-16E-Instruct ⏎ ``` ⏎ SGLANG_AITER_MOE=1 python -m sglang.launch_server --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct/ --port 30000 --tp  …[truncated]

### L1-f194e14fb7  (L1, 2025-05-15, sha f194e14fb7ff, PR #6147)
TITLE: Reduce MoE memory usage (#6147)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+10/-2); python/sglang/srt/layers/moe/ep_moe/layer.py (+58/-35); python/sglang/srt/models/deepseek_v2.py (+3/-3); python/sglang/srt/utils.py (+4/-0)
BODY: ## Motivation ⏎  ⏎ I realized there is a more lightweight approach to do something similar as DisposableTensor, and please refer to the `dispose_tensor` function. ⏎  ⏎ For figures below, note that the aspect ratio is different for each figure, so the height of each box cannot be compared directly. Instead, can only check relative relationships between boxes. ⏎  ⏎ In addition, the "prefill" here means the default deepgemm approach, while the one in #508 …[truncated]

### L1-839fb31e5f  (L1, 2025-05-16, sha 839fb31e5f6e, PR #6334)
TITLE: [Fix] Improve dependencies for Blackwell image (#6334)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+8/-7)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Move decord out of runtime_common dependency ⏎ - Add flashinfer dependency for blackwell  ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6fc9357503  (L1, 2025-05-16, sha 6fc935750336, PR #5694)
TITLE: [2/2] Add python wrapper for CUTLASS FP8 Blockscale MoE Kernel.  (#5694)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:production-kernel-provenance
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+207/-0); python/sglang/srt/layers/quantization/fp8.py (+90/-0); python/sglang/srt/layers/quantization/fp8_utils.py (+6/-0); sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/csrc/common_extension.cc (+7/-3); sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+111/-36); sgl-kernel/csrc/moe/prepare_moe_input.cu (+128/-0); sgl-kernel/include/sgl_kernel_ops.h (+18/-1); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/python/sgl_kernel/moe.py (+36/-0); (+2 more)
LABELS: high priority
BODY: ### NOTE ⏎  ⏎ The current CUTLASS 3.9 in SGLang will experience: 1. Kernel hang 2. Perf slowdown for the this MoE kernel. ⏎ I'll update our CUTLASS dependency in another PR, as it breaks some of the existing sm90 templates.  ⏎  ⏎ ## Motivation ⏎ Using the benchmark we provided in the PR, we have found our fused_expert layer with CUTLASS 4.0 in CUDA graph mode has ~30%-40% speedup over Triton in CUDA graph mode on small batch sizes.  ⏎  ⏎  ⏎ For Deepseek V …[truncated]

### L1-4bd2952a37  (L1, 2025-05-16, sha 4bd2952a376a, PR #6121)
TITLE: feat: add dp attention support for Qwen 2/3 MoE models, fixes #6088 (#6121)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/bench_one_batch.py (+1/-0); python/sglang/srt/layers/dp_attention.py (+0/-10); python/sglang/srt/models/qwen2_moe.py (+227/-32); python/sglang/srt/models/qwen3_moe.py (+221/-28)
BODY: This is the prerequisites of EP, which is introduced in PR #5917 ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ As described in #6088, DP attention is not supported for Qwen MoE models, but #5917 introduces EP MoE for them. This PR introduces DP attention support for them. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Following DP Attention part in `deepseek_v2.py`, I modified `qwen2_moe.py` and `qwen3_moe.py`. ⏎  ⏎ ## Benchmark Results and Accuracy Results ⏎  ⏎ > The following benchmark …[truncated]

### L1-2df9d40aa6  (L1, 2025-05-16, sha 2df9d40aa672, PR #6324)
TITLE: Minor code cleanup refactor for DeepSeek models (#6324)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+10/-1); python/sglang/srt/models/deepseek_v2.py (+16/-35)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3d7f7a43c8  (L1, 2025-05-17, sha 3d7f7a43c87f, PR #6368)
TITLE: chore: bump sgl-kernel v0.1.3 (#6368)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/Makefile (+1/-0); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2716830802  (L1, 2025-05-17, sha 2716830802ae, PR #6175)
TITLE: Speed up when having padding tokens in DeepEP (#6175)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+38/-4); python/sglang/srt/model_executor/cuda_graph_runner.py (+4/-0); python/sglang/srt/model_executor/forward_batch_info.py (+4/-0); python/sglang/srt/models/deepseek_v2.py (+7/-5)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ test ⏎  ⏎ ``` ⏎ PYTHONUNBUFFERED=1 SGLANG_TORCH_PROFILER_DIR=/host_home/temp_sglang_server2local python3 -m sglang.launch_server --model-path /dev/shm/DeepSeek-R1 --trust-remote-code --dist-init-addr 192.168.0.55:5757 --nnodes 2 --node-rank ${MY_NODE_RANK} --tp-size ${num_gpu} --dp-size ${num_gpu} --enable-dp-attention --mem-fraction-static 0.8 --chunked-prefill-size $((128*${num_gpu})) --max-running-requests $((${num_gpu}*128)) --c …[truncated]

### L1-e3b8a72291  (L1, 2025-05-17, sha e3b8a72291af, PR #6348)
TITLE: [fix] illegal memory in _fwd_kernel_ep_scatter_2 and _fwd_kernel_ep_gather (#6348)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+23/-9)
BODY: ## Motivation ⏎  ⏎  ⏎ when deploying large scale ep deepseek, the prefill node sometimes meets illegal memory error with heavy workload as following ⏎ <img width="697" alt="image" src="https://github.com/user-attachments/assets/f4d5061b-59df-47c7-95cd-ede1713cfe77" /> ⏎ this is caused by the index overflow in _fwd_kernel_ep_scatter_2. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ replace int32 index with int64 to avoid overflow.  ⏎  ⏎ CUDA_LAUNCH_BLOCKING=1 python test.py …[truncated]

### L1-fd08c04821  (L1, 2025-05-17, sha fd08c0482129, PR #6257)
TITLE: Support custom DeepEP tuning config (#6257)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+35/-5); python/sglang/srt/model_executor/model_runner.py (+1/-0); python/sglang/srt/server_args.py (+7/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/utils.py (+7/-0); test/srt/test_moe_deepep.py (+28/-0)
BODY: ## Motivation ⏎  ⏎ Btw that test looks a bit stale, thus a few lines of diff are there to make that test run again ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f07c6a009b  (L1, 2025-05-17, sha f07c6a009b42, PR #6377)
TITLE: chore: upgrade sgl-kernel v0.1.3 (#6377)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5dd62c3a6f  (L1, 2025-05-18, sha 5dd62c3a6f97, PR #6339)
TITLE: Add fp8 shared_expert kernel for CPU in sgl-kernel and add UT (#6339)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/moe.cpp (+43/-0); sgl-kernel/csrc/cpu/moe_fp8.cpp (+205/-0); sgl-kernel/csrc/cpu/gemm.h (+35/-0); sgl-kernel/csrc/cpu/gemm_fp8.cpp (+40/-32); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+2/-0); sgl-kernel/setup_cpu.py (+1/-0); test/srt/cpu/test_shared_expert.py (+223/-0); test/srt/cpu/utils.py (+54/-0)
LABELS: high priority, sgl-kernel, intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is a follow-up on https://github.com/sgl-project/sglang/issues/2807 and https://github.com/sgl-project/sglang/pull/5150 to add **fp8** shared_expert kernel for CPU. The bf16 and int8 shared_expert kernel is already added in https://github.com/sgl-project/sglang/pull/5150. ⏎  ⏎ This PR also adds UTs for bf16, int8 and fp8 shared_expert kernels for CPU. ⏎  ⏎ ## Modifications ⏎ The main change is the C++ kernels for fp8 shared_ …[truncated]

### L1-72bfb0baf0  (L1, 2025-05-18, sha 72bfb0baf06f, PR #6325)
TITLE: Refactor DeepSeek MoE layer to unify the two forward branches (#6325)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+53/-49)
BODY: ## Motivation ⏎  ⏎ With this, it is easier to extract everything into ops for a unified TBO. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f0653886a5  (L1, 2025-05-19, sha f0653886a5e0, PR #4957)
TITLE: Expert distribution recording without overhead for EPLB (#4957)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+14/-0); python/sglang/srt/layers/moe/topk.py (+5/-4); docs/backend/native_api.ipynb (+2/-14); python/sglang/srt/managers/expert_distribution.py (+595/-56); python/sglang/srt/managers/expert_location.py (+273/-0); python/sglang/srt/managers/scheduler.py (+7/-6); python/sglang/srt/model_executor/model_runner.py (+47/-0); python/sglang/srt/models/deepseek_v2.py (+20/-9); python/sglang/srt/models/qwen2_moe.py (+18/-8); python/sglang/srt/server_args.py (+32/-0); (+2 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ For EPLB, and also for debugging/knowing details ⏎  ⏎ dep: #5219 ⏎  ⏎ NOTE: There are enhancements to this, but it currently in branch https://github.com/sgl-project/sglang/pull/5295 and not yet extracted to here. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1b19df4b2a  (L1, 2025-05-19, sha 1b19df4b2a14, PR #6321)
TITLE: Refactor communication logic of DeepSeek for extensibility and understandability (#6321)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/communicator.py (+451/-0); python/sglang/srt/models/deepseek_v2.py (+45/-188)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Make code clean, not error-prune, extensible ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d0443275f0  (L1, 2025-05-19, sha d0443275f059, PR #6326)
TITLE: Refactor DeepSeek logic into atomic operations (#6326)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+152/-76); python/sglang/srt/operations.py (+154/-0); python/sglang/srt/operations_strategy.py (+31/-0)
BODY: ## Motivation ⏎  ⏎ For unified TBO ⏎  ⏎ Please subtract diff from previous PRs ⏎  ⏎ Tests ⏎  ⏎ ``` ⏎ SGL_ENABLE_JIT_DEEPGEMM=0 python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V2-Lite --trust-remote-code --tp 4 --dp 2 --host 0.0.0.0 --port 12321 --enable-dp-attention --max-running-requests 2048 --disable-cuda-graph --enable-deepep-moe --deepep-mode normal ⏎ CUDA_VISIBLE_DEVICES=4,5,6,7 SGL_ENABLE_JIT_DEEPGEMM=0 python3 -m sglang.launch_ser …[truncated]

### L1-c471d39eb9  (L1, 2025-05-19, sha c471d39eb9e2, PR #6386)
TITLE: Support loading weights when physical experts are different from logical experts (#6386)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+23/-0); python/sglang/srt/managers/expert_location.py (+14/-1)
BODY: ## Motivation ⏎  ⏎ subtract diff from https://github.com/sgl-project/sglang/pull/4957 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-83f2d9d4ed  (L1, 2025-05-19, sha 83f2d9d4ed9d, PR #6429)
TITLE: [QuickFix] fix gptq model initialize (#6429)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/__init__.py (+5/-2); python/sglang/srt/layers/quantization/gptq.py (+298/-6)
LABELS: high priority
ISSUES: #6249 [Bug] KeyError: 'intermediate_size_full' when loading Qwen3 MoE GPTQ | #6313 [Bug] Unsupport for Qwen3-30B-A3B-GPTQ-Int4
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ When vllm version updated,`create_weights ` def changed. params `intermediate_size ` -> `intermediate_size_full `. Then we met some GPTQ error. ⏎  ⏎ This PR just for GPTQ quick fix, then for stack PR. We will remove and refactor quantization part, remove vllm. Add custom kernel to sgl-kernel ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e98afbe042  (L1, 2025-05-19, sha e98afbe042cf, PR #6385)
TITLE: Support dispatching logical to physical experts (#6385)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-0); python/sglang/srt/layers/moe/topk.py (+18/-0); python/sglang/srt/managers/expert_distribution.py (+2/-1); python/sglang/srt/managers/expert_location.py (+58/-3); python/sglang/srt/managers/expert_location_dispatch.py (+91/-0); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/model_executor/model_runner.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+2/-0); python/sglang/srt/server_args.py (+7/-0)
BODY: ## Motivation ⏎  ⏎ subtract diff from https://github.com/sgl-project/sglang/pull/4957 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-13feffd082  (L1, 2025-05-20, sha 13feffd0827d, PR #6447)
TITLE: Fix master CI for DeepSeek (#6447)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+9/-5)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a40aecc5a3  (L1, 2025-05-21, sha a40aecc5a3a5, PR #6468)
TITLE: Fix num_qps_per_rank computation when providing custom DeepEP configuration (#6468)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+21/-9)
BODY: ## Motivation ⏎  ⏎ Note: it is also made public, b/c I need to access it in TBO ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a071dc4084  (L1, 2025-05-21, sha a071dc4084ea, PR #6467)
TITLE: Tiny add stage assertions to DeepEPDispatcher to avoid misuse (#6467)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+20/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-ccfe5c009d  (L1, 2025-05-21, sha ccfe5c009d61, PR #6461)
TITLE: Support redundant experts in expert parallel (#6461)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/model_runner.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+8/-3); python/sglang/srt/server_args.py (+7/-0); python/sglang/srt/managers/expert_location.py (+1/-2); python/sglang/srt/managers/schedule_batch.py (+1/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-cfe48c5902  (L1, 2025-05-21, sha cfe48c590228, PR #6419)
TITLE: [CPU] Fix build issue (#6419)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/moe.cpp (+9/-9); sgl-kernel/csrc/cpu/CMakeLists.txt (+10/-41); sgl-kernel/csrc/cpu/bmm.cpp (+2/-1); sgl-kernel/csrc/cpu/gemm.cpp (+2/-1); sgl-kernel/csrc/cpu/gemm_fp8.cpp (+1/-1); sgl-kernel/csrc/cpu/gemm_int8.cpp (+2/-2); sgl-kernel/csrc/cpu/interface.cpp (+5/-6); sgl-kernel/csrc/cpu/qkv_proj.cpp (+6/-6); sgl-kernel/csrc/cpu/shm.h (+1/-1); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+94/-42); (+4 more)
LABELS: high priority, intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎  ⏎ 1. Simplify the `CMakeLists.txt` to automatically detect all `.cpp` files under the `csrc/cpu/` directory as `SOURCES`. ⏎ 2. Fix the issue where `sgl_kernel` cannot be imported properly. ⏎  ⏎ The following commands are expected to work correctly after this PR. ⏎  ⏎ ``` ⏎ cd sgl-kernel/ ⏎ cp pyproject_cpu.toml pyproject.toml ⏎ pip install -v . ⏎ python -c "import sgl_kernel" ⏎ ``` ⏎  ⏎ cc @chunyuan-w  ⏎  ⏎ ## Checklist

### L1-e9feb48838  (L1, 2025-05-21, sha e9feb4883830, PR #6308)
TITLE: [RL] Remove the w13 weight_scale and input_scale for UnquantizedEPMoE… (#6308)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+5/-14)
BODY: …Method ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎ When doing RL training, we may release all the parameters with `/release_memory_occupation` to free the memory occupied by the inference engine, which will also released all the `input_scale`s and `weight_scale`s. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ The origin `w13_weight_scale` in `UnquantizedEPMoEMethod` does not support reloading (as the shape should be `(num_experts_per_partition, 2)`). And I found that for the `Unqua …[truncated]

### L1-d71f3f0a2a  (L1, 2025-05-22, sha d71f3f0a2a55, PR #6522)
TITLE: chore: bump sgl-kernel v0.1.4 (#6522)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ unblock https://github.com/sgl-project/sglang/pull/6521 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fc0e3b9174  (L1, 2025-05-22, sha fc0e3b91744b, PR #6120)
TITLE: Support qwen3 deepep (#6120)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen2_moe.py (+4/-1); python/sglang/srt/models/qwen3_moe.py (+121/-7)
BODY: ## Motivation ⏎  ⏎ Support qwen3's deepep. For now, we've simply copied the deepep code from DS, and the accuracy test has passed. ⏎  ⏎ ## TODO ⏎ * test bf16 compatibility ⏎  ⏎ ## Test Command ⏎ ``` ⏎  python3 -m sglang.launch_server --model-path /workdir/huggingface.co/Qwen/Qwen3-235B-A22B-FP8/ --tp 4 --trust-remote --enable-torch-compile --torch-compile-max-bs 8 --host 0.0.0.0 --port 8418 --reasoning-parser qwen3 --tool-call-parser qwen25 --enable-deepe …[truncated]

### L1-0b07c4a99f  (L1, 2025-05-22, sha 0b07c4a99f8a, PR #6532)
TITLE: chore: upgrade sgl-kernel v0.1.4 (#6532)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); test/srt/run_suite.py (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3ded6235c9  (L1, 2025-05-23, sha 3ded6235c9e4, PR #6404)
TITLE: Add fp8 fused_experts kernel for CPU in sgl-kernel and add UT (#6404)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/moe.cpp (+76/-28); sgl-kernel/csrc/cpu/moe_fp8.cpp (+291/-12); sgl-kernel/csrc/cpu/gemm.h (+26/-0); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+4/-1); sgl-kernel/setup_cpu.py (+0/-116); test/srt/cpu/test_moe.py (+259/-0); test/srt/cpu/utils.py (+96/-0)
LABELS: high priority, sgl-kernel, intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is a follow-up on https://github.com/sgl-project/sglang/issues/2807 and https://github.com/sgl-project/sglang/pull/5150 to add **fp8** **fused_experts** kernel for CPU. The bf16 and int8 fused_experts kernel is already added in https://github.com/sgl-project/sglang/pull/5150. ⏎  ⏎ This PR also adds UTs for bf16, int8 and fp8 fused_experts kernels for CPU. ⏎  ⏎ ## Modifications ⏎ The main change is the C++ kernels for fp8 fus …[truncated]

### L1-2f42749184  (L1, 2025-05-23, sha 2f42749184ca, PR #6474)
TITLE: Fix topk inference performance reduce (#6474)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-0)
BODY: ## Motivation ⏎ When the following logic is added to `topk.py`, the inference performance will be significantly affected: ⏎ https://github.com/sgl-project/sglang/blob/66324895c6925c86c2b8c811ebb6dfb93ae42356/python/sglang/srt/layers/moe/topk.py#L267-L269 ⏎  ⏎ Run command: ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path /path/to/DeepSeek-V3-0324 --trust-remote-code --host 0.0.0.0 --port 30000 --attention-backend flashinfer --n-share-experts-fusion  …[truncated]

### L1-e6f113569e  (L1, 2025-05-23, sha e6f113569e51, PR #6533)
TITLE: support eplb for qwen3 (#6533)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+4/-2); python/sglang/srt/managers/expert_distribution.py (+3/-1); python/sglang/srt/models/qwen3_moe.py (+39/-22)
BODY: ## Motivation ⏎ support eplb for qwen3moe, ~~need merge #6120 first, then do other modification. (add ExpertLocationDispatchInfo)~~ ⏎  ⏎ simple test ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path Qwen/Qwen3-235B-A22B-FP8 --tp 8 --dp 2 --enable-dp-attention --trust-remote --enable-deepep-moe --deepep-mode normal --enable-eplb --expert-distribution-recorder-buffer-size 50  --expert-distribution-recorder-mode stat --disable-radix-cache --eplb-rebal …[truncated]

### L1-b2388433be  (L1, 2025-05-24, sha b2388433be8f, PR #6578)
TITLE: Add back DeepSeek non-TBO branches (#6578)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+119/-9)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0d47788025  (L1, 2025-05-24, sha 0d4778802576, PR #4068)
TITLE: Support overlapping two batches (#4068)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/tbo_backend.py (+241/-0); python/sglang/srt/layers/quantization/deep_gemm.py (+13/-0); python/sglang/srt/managers/schedule_batch.py (+8/-0); python/sglang/srt/managers/scheduler.py (+25/-1); python/sglang/srt/model_executor/cuda_graph_runner.py (+21/-1); python/sglang/srt/model_executor/forward_batch_info.py (+18/-1); python/sglang/srt/model_executor/model_runner.py (+18/-9); python/sglang/srt/models/deepseek_v2.py (+118/-92); python/sglang/srt/operations.py (+37/-2); python/sglang/srt/operations_strategy.py (+107/-24); (+3 more)
LABELS: high priority
BODY: ## Update ⏎  ⏎ If you want to try PD + EPLB + two-batch-overlap + ..., here is the branch that merges everything before they are merged into master: https://github.com/fzyzcjy/sglang/tree/feat/dev_branch ⏎  ⏎  ⏎ ## 2025.03.26 ⏎  ⏎ Just now I run some benchmark on 8xH200 and there seems to be performance improvements. Note that I have not done careful tuning, because still waiting for the kernels and features (e.g. DeepGEMM for grouped gemm, DeepEP low-l …[truncated]

### L1-1a39979993  (L1, 2025-05-24, sha 1a3997999356, PR #6537)
TITLE: Sgl-router Prometheus metrics endpoint and usage track metrics (#6537)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+6/-0); sgl-router/Cargo.toml (+3/-0); sgl-router/README.md (+12/-0); sgl-router/py_src/sglang_router/launch_router.py (+20/-0); sgl-router/py_test/test_launch_server.py (+10/-0); sgl-router/src/lib.rs (+21/-1); sgl-router/src/prometheus.rs (+40/-0); sgl-router/src/router.rs (+42/-1); sgl-router/src/server.rs (+13/-0)
BODY: ## Motivation ⏎ Adding a Prometheus metric endpoint exposes key usage statistics of sgl-router. With these metrics, operators can easily integrate with Prometheus/Grafana to monitor request rates, latencies, error counts, and other relevant information. ⏎  ⏎ ## Modifications ⏎ add prometheus http scrap endpoint on separate port   ⏎ add metrics track request duration, queue length and cache miss and hit ⏎  ⏎ ``` ⏎ curl localhost:9000 ⏎ # TYPE sgl_router_pr …[truncated]

### L1-0ca1811715  (L1, 2025-05-25, sha 0ca1811715ea, PR #6571)
TITLE: Support fake perfectly balanced EP dispatch algorithm (#6571)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+10/-0); python/sglang/srt/managers/expert_location_dispatch.py (+13/-1); python/sglang/srt/server_args.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ not tested yet ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-65f091310c  (L1, 2025-05-25, sha 65f091310ca6, PR #6581)
TITLE: refactor qwen moe code, use communicator to support tp+dp (#6581)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+1/-8); python/sglang/srt/models/qwen2_moe.py (+35/-185); python/sglang/srt/models/qwen3_moe.py (+33/-185); python/sglang/srt/utils.py (+8/-0); test/srt/test_disaggregation.py (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ refactor qwen2/qwen3moe tp+dp code, use communicator in #6321  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f9bab3d591  (L1, 2025-05-25, sha f9bab3d59100, PR #6598)
TITLE: qwen3moe support two batch overlap (#6598)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen2_moe.py (+17/-6); python/sglang/srt/models/qwen3_moe.py (+200/-11); python/sglang/srt/operations_strategy.py (+98/-7); python/sglang/srt/two_batch_overlap.py (+8/-4); test/srt/test_two_batch_overlap.py (+28/-0)
BODY: ## Motivation ⏎  ⏎  ⏎ Support two batch overlap for Qwen3, need merge #6581 first. Current overlap strategy is not stable, maybe we need change during tests. ⏎ ``` ⏎ # accuracy test ⏎ python3 -m sglang.launch_server --model-path /dev/shm/models/Qwen/Qwen3-235B-A22B-FP8/ --tp 8 --dp 8 --enable-dp-attention --trust-remote --enable-deepep-moe --deepep-mode normal --enable-two-batch-overlap ⏎  ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-que …[truncated]

### L1-a564e001b5  (L1, 2025-05-27, sha a564e001b532, PR #6673)
TITLE: Fix DeepEP error in Qwen 3 MoE models (#6673)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+9/-6)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c087ddd686  (L1, 2025-05-28, sha c087ddd6865a, PR #6627)
TITLE: Refine pre_reorder_triton_kernel slightly to improve performance (#6627)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+9/-4); benchmark/kernels/fused_moe_triton/benchmark_ep_pre_reorder_triton.py (+100/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ In ep_moe kernel _pre_reorder_triton_kernel_ and _post_reorder_triton_kernel_, every inner loop recomputes ⏎ offset = start_offset + tl.arange(...) ⏎  ⏎ The optimization is to create a constant once: ⏎ vec = tl.arange(0, BLOCK_SIZE) ⏎ and inside the loop use idx = start_offset + vec ⏎  ⏎ The benefit is one less instruction each iteration. The warp scheduler can vectorize the access pattern. ⏎ Per benchmark result the kernel gains 10-15%  …[truncated]

### L1-541a985f85  (L1, 2025-05-28, sha 541a985f85bc, PR #6710)
TITLE: Fuse routed_scaling_factor in DeepSeek (#6710)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+7/-3)
BODY: (cherry picked from commit a257203) ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6df81e8a39  (L1, 2025-05-29, sha 6df81e8a3919, PR #6742)
TITLE: Support tuning DeepEP configs (#6742)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: benchmark/kernels/deepep/deepep_utils.py (+218/-0); benchmark/kernels/deepep/tuning_deepep.py (+476/-0)
BODY: ## Motivation ⏎  ⏎ copied and modified from DeepEP unit tests ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b581b22504  (L1, 2025-05-30, sha b581b2250474, PR #6772)
TITLE: Fix one bug in the grouped-gemm triton kernel (#6772)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ When block quant is not in use, EPMoE returns incorrect responses. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-ced3c07afe  (L1, 2025-05-30, sha ced3c07afe02, PR #6782)
TITLE: Support token-level quantization for EP MoE (#6782)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+18/-2); python/sglang/srt/layers/moe/ep_moe/layer.py (+71/-23)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-094fbdacd5  (L1, 2025-05-31, sha 094fbdacd5bb, PR #6734)
TITLE: Fix incorrect LoRA weight loading for fused gate_up_proj (#6734)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/conversation.py (+2/-2); python/sglang/srt/lora/lora.py (+9/-1); python/sglang/srt/models/idefics2.py (+16/-9); python/sglang/srt/models/phi4mm.py (+2/-2)
BODY: ## Motivation ⏎  ⏎ During testing, I identified two bugs introduced in my previous PR for supporting phi-4-mm. ⏎  ⏎ 1. (Major impact) I introduced fused LoRA weight support in my prev PR, however, I did not correctly handle gate_up_proj shape. This problem was not caught earlier because pytorch implicitly handled shape mismatch through broadcast.  ⏎  ⏎ 3. (Minor impact) I had an incorrect understanding of the seqlens calculation for Idefics embedding.  …[truncated]

### L1-b520d02888  (L1, 2025-05-31, sha b520d0288863, PR #6794)
TITLE: chore: bump sgl-kernel v0.1.5 (#6794)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation  ⏎  ⏎ ref https://github.com/sgl-project/sglang/pull/6788 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-34c63731fc  (L1, 2025-05-31, sha 34c63731fcd3, PR #6795)
TITLE: chore: upgrade sgl-kernel v0.1.5 (#6795)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1da8d23051  (L1, 2025-06-01, sha 1da8d2305124, PR #6800)
TITLE: chore: update blackwell docker (#6800)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+23/-12)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-55444ed667  (L1, 2025-06-01, sha 55444ed66715, PR #6699)
TITLE: [EP] Add cuda kernel for moe_ep_pre_reorder (#6699)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.reorder_aot
FILES: sgl-kernel/csrc/moe/ep_moe_reorder_kernel.cu (+89/-0); sgl-kernel/python/sgl_kernel/moe.py (+24/-0); sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/benchmark/bench_moe_ep_pre_reorder.py (+100/-0); sgl-kernel/csrc/common_extension.cc (+4/-0); sgl-kernel/include/sgl_kernel_ops.h (+11/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0)
BODY: ## Motivation ⏎  ⏎ moe_pre_reorder is one of the important kernels in EP MoE. ⏎ Currently moe_pre_reorder is using triton kernel. This PR is to introduce cuda implementation for this kernel. ⏎ The new kernel gains 10-20% performance improvement. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-eb38c7d1ca  (L1, 2025-06-02, sha eb38c7d1cae1, PR #6093)
TITLE: [1/2] Add Kernel support for Cutlass based Fused FP4 MoE (#6093)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise, L1.cutlass.nvfp4, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+178/-1); sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cu (+471/-0); sgl-kernel/csrc/moe/prepare_moe_input.cu (+145/-19); sgl-kernel/python/sgl_kernel/moe.py (+55/-0); python/sglang/test/test_fp4_moe.py (+247/-0); sgl-kernel/CMakeLists.txt (+2/-0); sgl-kernel/csrc/common_extension.cc (+21/-2); sgl-kernel/csrc/gemm/nvfp4_expert_quant.cu (+431/-0); sgl-kernel/csrc/gemm/nvfp4_quant_entry.cu (+23/-0); sgl-kernel/include/sgl_kernel_ops.h (+24/-0); (+2 more)
LABELS: high priority
BODY: This kernel adds support for NVFP4 MoE kernels.  ⏎  ⏎ Currently measured perf:  ⏎ ``` ⏎ [--------------------------------------------------------------------------------------------- FP4 MOE vs FP8 Triton ---------------------------------------------------------------------------------------------] ⏎                                                                                                                        |  triton_moe  |  triton_moe_cuda_ …[truncated]

### L1-ff00895c46  (L1, 2025-06-02, sha ff00895c46a4, PR #6456)
TITLE: Add CPU optimized kernels for topk and rope fusions  (#6456)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: sgl-kernel/csrc/cpu/topk.cpp (+221/-0); sgl-kernel/csrc/cpu/norm.cpp (+77/-0); sgl-kernel/csrc/cpu/rope.cpp (+310/-93); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+25/-4); test/srt/cpu/test_norm.py (+14/-0); test/srt/cpu/test_rope.py (+103/-5); test/srt/cpu/test_topk.py (+83/-0)
LABELS: sgl-kernel, intel, cpu
BODY: ## Overview: ⏎  ⏎ This PR is adding the following CPU optimized sgl-kernels that are at least used in Qwen3/LLama1-4 models: ⏎ ``` ⏎ - TopK fusions:  ⏎       TopK+sigmoid ⏎       softmax+TopK ⏎ - Norm fusion: ⏎       L2norm ⏎ - RoPE fusions: ⏎       origin rope fusion (gpt_neox style and gptj style) ⏎ ```

### L1-8a5480528d  (L1, 2025-06-03, sha 8a5480528d71, PR #6735)
TITLE: [Refactor] Rename `n_share_experts_fusion` as `num_fused_shared_experts` (#6735)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_py, L1.routing.fused_gate
FILES: python/sglang/srt/layers/moe/topk.py (+15/-15); sgl-kernel/csrc/moe/moe_fused_gate.cu (+13/-13); sgl-kernel/python/sgl_kernel/moe.py (+3/-3); benchmark/kernels/fused_moe_triton/README.md (+4/-7); benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py (+3/-7); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+2/-10); python/sglang/srt/managers/schedule_batch.py (+1/-1); python/sglang/srt/model_executor/model_runner.py (+1/-1); python/sglang/srt/models/deepseek_nextn.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+24/-20); (+4 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-81964328b7  (L1, 2025-06-04, sha 81964328b7ed, PR #6736)
TITLE: Set `num_fused_shared_experts` as `num_shared_experts` when shared_experts fusion is not disabled (#6736)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.routing.fused_gate, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-0); python/sglang/srt/layers/moe/fused_moe_native.py (+4/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=257,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+2/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+9/-0); python/sglang/srt/layers/moe/topk.py (+1/-1); sgl-kernel/csrc/moe/moe_fused_gate.cu (+12/-4); sgl-kernel/python/sgl_kernel/moe.py (+2/-2); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+6/-3); (+12 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎  ⏎ ## TODO ⏎  ⏎ Finetune kernel configs ⏎  ⏎ ## Checklist

### L1-bd75690f4e  (L1, 2025-06-04, sha bd75690f4eef, PR #6858)
TITLE: fix ep_moe_reorder kernel bugs (#6858)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.reorder_aot
FILES: sgl-kernel/csrc/moe/ep_moe_reorder_kernel.cu (+35/-24); sgl-kernel/benchmark/bench_moe_ep_pre_reorder.py (+10/-7); sgl-kernel/tests/test_ep_moe_pre_reorder_kernel.py (+181/-0)
BODY: ## Motivation ⏎  ⏎  ⏎ ## benchmark in h100 ⏎  ⏎ ```shell ⏎ ep-moe-pre-reorder-performance: ⏎    batch_size  CUDA Kernel  Triton Kernel ⏎ 0        64.0     9.952000      15.584000 ⏎ 1       128.0    10.144000      15.712000 ⏎ 2       256.0    12.864000      17.440001 ⏎ 3       512.0    17.824000      22.528000 ⏎ 4       640.0    23.712000      24.831999 ⏎ 5       768.0    24.896000      27.456000 ⏎ 6      1024.0    29.247999      33.920001 ⏎ 7      2048.0    55.0 …[truncated]

### L1-499f5e620c  (L1, 2025-06-04, sha 499f5e620c24, PR #6878)
TITLE: Fix one missing arg in DeepEP (#6878)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+22/-19)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5aff1e9392  (L1, 2025-06-05, sha 5aff1e9392d0, PR #6820)
TITLE: Fix Qwen3MoE missing token padding optimization (#6820)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+3/-3); python/sglang/srt/models/qwen3_moe.py (+2/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0166403c20  (L1, 2025-06-05, sha 0166403c2029, PR #6868)
TITLE: Support Blackwell DeepEP docker images (#6868)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/release-docker-deepep.yml (+14/-3); .github/workflows/release-docker-dev-deepep.yml (+0/-36); docker/Dockerfile.deepep (+6/-2); docker/Dockerfile.dev (+3/-0); docker/Dockerfile.dev-deepep (+0/-80)
BODY: ## Motivation ⏎  ⏎ ~~do not merge, not tested yet~~ ⏎ tested blackwell one locally using ⏎  ⏎ ``` ⏎ (cd /home/innomatrix/tom/primary_synced/sglang && docker build . -f docker/Dockerfile.deepep --build-arg BASE_IMAGE=lmsysorg/sglang:blackwell -t lmsysorg/sglang:blackwell-deepep --no-cache) ⏎ ``` ⏎  ⏎ and it compiles ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-43baba649e  (L1, 2025-06-05, sha 43baba649e43, PR #6837)
TITLE: [EP] Add cuda kernel for moe_ep_post_reorder (#6837)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.reorder_aot
FILES: sgl-kernel/csrc/moe/ep_moe_reorder_kernel.cu (+83/-2); sgl-kernel/python/sgl_kernel/moe.py (+22/-0); sgl-kernel/benchmark/bench_moe_ep_post_reorder.py (+92/-0); sgl-kernel/csrc/common_extension.cc (+6/-2); sgl-kernel/include/sgl_kernel_ops.h (+10/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/tests/test_ep_moe_post_reorder_kernel.py (+163/-0)
LABELS: ready-to-merge
BODY: ## Motivation ⏎ moe_post_reorder is one of the important kernels in EP MoE. ⏎ Currently moe_post_reorder is using triton kernel. This PR is to introduce CUDA implementation for this kernel. ⏎ The new kernel is expected to gain performance improvement. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0df6765c83  (L1, 2025-06-05, sha 0df6765c83e2, PR #6887)
TITLE: [CUTLASS-FP4-MOE]  Introduce CutlassMoEParams class for easy initialization of Cutlass Grouped Gems Metadata (#6887)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+39/-54); python/sglang/srt/layers/moe/cutlass_moe_params.py (+169/-0); python/sglang/srt/layers/quantization/fp8.py (+2/-2); sgl-kernel/python/sgl_kernel/moe.py (+7/-11); python/sglang/test/test_cutlass_moe.py (+3/-3); python/sglang/test/test_fp4_moe.py (+10/-9)
BODY: ## Motivation ⏎  ⏎ Refactors Cutlass MoE to keep the interface cleaner. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Introduces `CutlassMoEParams` Class that creates all the cutlass metadata based on the shape of intermediate size and hidden shape ⏎  ⏎ ## Checklist ⏎  ⏎ - [N/A] Update documentation / docstrings / example tutorials as needed, according to [Writing Documentation](https://docs.sglang.ai/references/contribution_guide.html#writing-documentation-running-docs-c …[truncated]

### L1-562f279a2d  (L1, 2025-06-05, sha 562f279a2d80, PR #6458)
TITLE: [CPU] enable CI for PRs, add Dockerfile and auto build task (#6458)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile.xeon (+44/-0); python/pyproject.toml (+1/-1); .github/workflows/pr-test-xeon.yml (+86/-0); .github/workflows/release-docker-xeon.yml (+35/-0); python/sglang/test/test_utils.py (+63/-1); test/srt/run_suite.py (+10/-0)
LABELS: high priority, intel, cpu
BODY: ## Motivation ⏎  ⏎ The PR is for enabling the docker env setup and CI processes for running SGLang on Xeon CPU servers. ⏎  ⏎ ## Modifications ⏎  ⏎ Add the dockerfile, and yml files for CPU part of CI process and auto docker image build-up. ⏎ The test files are updated to enable device specific tests, as well as the CPU test cases. ⏎  ⏎ ## Checklist

### L1-b819381fec  (L1, 2025-06-05, sha b819381feca4, PR #6838)
TITLE: AITER backend extension and workload optimizations (#6838)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-3); .github/workflows/pr-test-amd.yml (+1/-1); docs/references/environment_variables.md (+1/-1); python/sglang/srt/layers/attention/aiter_backend.py (+488/-124); python/sglang/srt/layers/layernorm.py (+27/-2); python/sglang/srt/layers/quantization/fp8.py (+16/-16); python/sglang/srt/layers/quantization/fp8_utils.py (+6/-10); python/sglang/srt/model_executor/model_runner.py (+10/-0); python/sglang/srt/models/deepseek_v2.py (+17/-0); (+2 more)
LABELS: high priority, aiter
BODY: `Co-author: @kkHuang-amd ` ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎ - DeepSeek optimization ⏎ - 1 simple flag: `SGLANG_USE_AITER` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-61ce91ed28  (L1, 2025-06-06, sha 61ce91ed2822, PR #6934)
TITLE: Tiny support customize DeepEP max dispatch tokens per rank (#6934)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+4/-2)
BODY: ## Motivation ⏎  ⏎ qiaolin as well as some other cases I meet needs it ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f8eaaab817  (L1, 2025-06-06, sha f8eaaab81716, PR #6767)
TITLE: [fix] logical_to_all_physical_map index 256 is out of bounds in EP parallel. (#6767)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+15/-3)
LABELS: ready-to-merge
BODY: [fix] logical_to_all_physical_map index 256 is out of bounds in EP parallel. ⏎  ⏎ fix bug https://github.com/sgl-project/sglang/issues/6625. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c4ffbeca19  (L1, 2025-06-06, sha c4ffbeca1923, PR #6939)
TITLE: Add triton fused moe kernel config for E=257 on B200 (#6939)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=257,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎ After #6736, the default number of experts for DeepSeek-V3 changes from 264 to 257, which causes the performance of fused_moe kernel to decrease. ⏎ This PR tunes the kernel of E=257 on B200 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Adding config file. ⏎  ⏎ Benchmark diff: ⏎ ```bash ⏎ SGLANG_ENABLE_FLASHINFER_GEMM=1 python3 -m sglang.launch_server --model /dev/shm/DeepSeek-V3-0324 --tp 8 --trust-remote-code --attention-backend triton ⏎  ⏎ python3 -m  …[truncated]

### L1-22fe787852  (L1, 2025-06-06, sha 22fe7878520d, PR #6942)
TITLE: [sgl-kernel] update deepgemm (#6942)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+4/-1); sgl-kernel/CMakeLists.txt (+1/-1)
BODY: ## Motivation ⏎  ⏎ pre action for https://github.com/sgl-project/sglang/pull/6893 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d664ca18f2  (L1, 2025-06-07, sha d664ca18f242, PR #6943)
TITLE: chore: bump sgl-kernel v0.1.6 (#6943)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2f715f51cc  (L1, 2025-06-07, sha 2f715f51cc41, PR #6944)
TITLE: Minor compile fused topk (#6944)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+17/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-2a413829f4  (L1, 2025-06-07, sha 2a413829f42b, PR #5955)
TITLE: Add triton version as a fused_moe_triton config search key to avoid performace decrease in different Triton version (#5955)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/README (+3/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_1_0/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_1_0/E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_1_0/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_1_0/E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_1_0/E=1,N=3072,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_1_0/E=1,N=3072,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_1_0/E=1,N=3072,device_name=NVIDIA_H100_80GB_HBM3.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_1_0/E=1,N=3584,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_1_0/E=1,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json (+0/-0); (+148 more)
BODY: ## Motivation ⏎  ⏎ Follow https://github.com/sgl-project/sglang/pull/5740 & https://github.com/sgl-project/sglang/pull/5850 . ⏎  ⏎ <img width="1031" alt="图片" src="https://github.com/user-attachments/assets/afd9c522-7ff9-48e8-9a0d-f143ae0e0975" /> ⏎  ⏎ Ideally we need to re-tune all shapes with the new triton versions. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8b5f83ed3b  (L1, 2025-06-07, sha 8b5f83ed3b7d, PR #6369)
TITLE: reduce torch.zeros overhead in moe align block size kernel (#6369)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.align.cuda_aot
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+58/-6); sgl-kernel/csrc/moe/moe_align_kernel.cu (+0/-2)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ origin: ⏎  ⏎ <img width="918" alt="图片" src="https://github.com/user-attachments/assets/ec8bcd33-c14a-41a8-a548-0e32f1c955c0" /> ⏎  ⏎ pr: ⏎  ⏎ <img width="475" alt="图片" src="https://github.com/user-attachments/assets/177625ea-aa45-41fa-b93b-fd33868b000c" /> ⏎  ⏎ ## Benchmark ⏎  ⏎ main: ⏎  ⏎  ⏎ ```shell ⏎ fused-moe-performance: ⏎      batch_size  vllm_fused_moe_triton  sglang_fused_moe_triton ⏎ 0           1.0               0.355232               …[truncated]

### L1-6153f2ff6e  (L1, 2025-06-07, sha 6153f2ff6e1b, PR #6945)
TITLE: chore: upgrade sgl-kernel v0.1.6 (#6945)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/deep_gemm.py (+42/-56)
BODY: ## Motivation ⏎  ⏎ Open a new PR due to no write permissions to https://github.com/sgl-project/sglang/pull/6893 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-bae4fdc7ab  (L1, 2025-06-07, sha bae4fdc7abb0, PR #6924)
TITLE: add fbgemm moe grouped gemm kernel benchmark (#6924)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: benchmark/fbgemm/benchmark_fbgemm_grouped_gemm.py (+366/-0); benchmark/fbgemm/fbgemm_grouped_gemm.py (+1294/-0); benchmark/fbgemm/test_grouped_gemm.py (+323/-0)
BODY: ## H100 Benchmark FBGEMM GroupedGEMM Results ⏎  ⏎  ⏎ When running benchmarks with triton==3.2.0, the following warning appears: we can't use warp-specialized features, but persistent kernels and TMA load/store remain available. ⏎  ⏎ ```shell ⏎ /home/ubuntu/bbuf/sglang/benchmark/kernels/fbgemm/fbgemm_grouped_gemm.py:1104: UserWarning: Warp specialization is disabled as the Triton build in current environment doesn't have such support. Please build from  …[truncated]

### L1-f5599ef124  (L1, 2025-06-07, sha f5599ef12421, PR #6866)
TITLE: Refactor global_server_args_dict (#6866)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/schedule_batch.py (+31/-26); python/sglang/srt/model_executor/model_runner.py (+7/-27)
BODY: ## Motivation ⏎  ⏎ wait for ci ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-515ef4facb  (L1, 2025-06-07, sha 515ef4facbc8, PR #6220)
TITLE: Fuse routed scaling factor in topk_reduce kernel (#6220)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+124/-8); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-0); python/sglang/srt/layers/quantization/blockwise_int8.py (+1/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+1/-0); python/sglang/srt/layers/quantization/fp8.py (+1/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+1/-0); python/sglang/srt/layers/quantization/w8a8_fp8.py (+1/-0); python/sglang/srt/layers/quantization/w8a8_int8.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+1/-1); benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py (+199/-0)
DEEP_STUDY: deep-study: this PR was reverted by PR 6968 (confirmed_revert, reason=unstated)
BODY: ## Motivation ⏎  ⏎ ### gsm8k acc in h200 ⏎  ⏎ ```shell ⏎ ➜  sglang git:(fuse_routed_scaling_factor_in_deepseek) ✗ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8  ⏎  ⏎ 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [00:42<00:00, 31.11it/s] ⏎ Accuracy: 0.947 ⏎ Invalid: 0.000 ⏎ Latency: 45.708 s ⏎ Output throughput: 3127.153 token/s ⏎ ``` ⏎  ⏎  …[truncated]

### L1-62fec60d81  (L1, 2025-06-07, sha 62fec60d812b, PR #6885)
TITLE: Add H20 fused MoE kernel tuning configs for DeepSeek-R1/V3 (#6885)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/E=257,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎ Follow https://github.com/sgl-project/sglang/pull/6736, add "E=257" moe config file for DeepSeek-V3/R1 ⏎  ⏎ ## Modifications ⏎  ⏎ Tuning command: ⏎ ```bash ⏎ python3 /mnt/data/sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py --model /mnt/data/DeepSeek-V3-0324 --tp-size 8 --dtype fp8_w8a8 --tune ⏎ ``` ⏎  ⏎ Deploy DeepSeek model: ⏎ ```bash ⏎ python3 -m sglang.launch_server --model-path /mnt/data/DeepSeek-V3-0324 --disable- …[truncated]

### L1-3e56f557fd  (L1, 2025-06-07, sha 3e56f557fdd1, PR #6916)
TITLE: Add a CUDA kernel for fusing mapping and weighted sum for MoE. (#6916)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+6/-5); sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu (+6/-6); sgl-kernel/csrc/moe/prepare_moe_input.cu (+114/-0); sgl-kernel/python/sgl_kernel/moe.py (+11/-0); sgl-kernel/csrc/common_extension.cc (+2/-1); sgl-kernel/include/sgl_kernel_ops.h (+6/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0)
BODY: ## Motivation ⏎  ⏎ A fusion kernel that improves the overall CUTLASS MOE layer perf for B200. ⏎ Fuses this single line:  ⏎ `(c2[c_map].view(m, topk, k) * topk_weights.view(m, topk, 1).to(out_dtype)).sum(dim=1)` ⏎  ⏎ ## Modifications ⏎ Previous perf: https://github.com/sgl-project/sglang/pull/5694 ⏎ After the change ⏎ ``` ⏎ --- Batch Size: 1 --- ⏎ Config: E=256, topk=8, H=7168, I_shard=512, dtype=torch.bfloat16, block_shape=[128, 128] ⏎ Warming up... ⏎ Using d …[truncated]

### L1-8db3ac55a9  (L1, 2025-06-07, sha 8db3ac55a975, PR #6955)
TITLE: chore: bump sgl-kernel v0.1.6.post1 (#6955)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-23881fa60c  (L1, 2025-06-07, sha 23881fa60ce5, PR #6957)
TITLE: chore: upgrade sgl-kernel v0.1.6.post1 (#6957)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-c2c4f57f63  (L1, 2025-06-07, sha c2c4f57f6311, PR #6853)
TITLE: [DeepseekR1-FP4] Add Support for nvidia/DeepSeekR1-FP4 model (#6853)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+19/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+334/-7); python/sglang/srt/models/deepseek_v2.py (+33/-5)
LABELS: high priority
BODY: ## Motivation ⏎ Adds Model support for DeepSeek R1 FP4 Model. (Functional enablement - kernel optimizations underway) ⏎  ⏎ ## Modifications ⏎ Adds ModelOptFP4FusedMoEMethod and CutlassMoEParams dataclass to initialize the parameters required by Cutlass MoE methods. ⏎  ⏎ ## Checklist

### L1-1fb76ebb93  (L1, 2025-06-07, sha 1fb76ebb9389, PR #6968)
TITLE: Revert "Fuse routed scaling factor in topk_reduce kernel (#6220)" (#6968)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+8/-124); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-1); python/sglang/srt/layers/quantization/blockwise_int8.py (+0/-1); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+0/-1); python/sglang/srt/layers/quantization/fp8.py (+0/-1); python/sglang/srt/layers/quantization/moe_wna16.py (+0/-1); python/sglang/srt/layers/quantization/w8a8_fp8.py (+0/-1); python/sglang/srt/layers/quantization/w8a8_int8.py (+0/-1); python/sglang/srt/models/deepseek_v2.py (+1/-1); benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py (+0/-199)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 6220 reason=unstated
BODY: This reverts commit 515ef4facbc89cd7c093c198386a8817fce856d6. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6c0a48282a  (L1, 2025-06-08, sha 6c0a48282a81, PR #6963)
TITLE: chore: bump sgl-kernel v0.1.7 (#6963)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); .github/workflows/pr-test-sgl-kernel.yml (+2/-3); sgl-kernel/build.sh (+3/-7); sgl-kernel/pyproject.toml (+2/-2); sgl-kernel/pyproject_cpu.toml (+2/-2); sgl-kernel/pyproject_rocm.toml (+2/-2); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fa3592cfeb  (L1, 2025-06-08, sha fa3592cfeb9d, PR #6966)
TITLE: rebase h20 fused_moe config (#6966)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_2_0/E=257,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+0/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_2_0/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+0/-0); benchmark/kernels/fused_moe_triton/README.md (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/configs/README.md (+0/-0)
BODY: ## Motivation ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-20d3ad3b58  (L1, 2025-06-08, sha 20d3ad3b586f, PR #6974)
TITLE: Fix CI and triton moe Configs (#6974)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_2_0/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/managers/schedule_batch.py (+3/-3); python/sglang/srt/model_executor/forward_batch_info.py (+0/-1); test/srt/test_mla_flashinfer.py (+1/-1)
BODY: This PR copied some old triton moe configs from 3.1 folder to 3.2 folder. ⏎  ⏎ It fixed a regression in our CI caused by https://github.com/sgl-project/sglang/pull/5955 ⏎ Failed test case: https://github.com/sgl-project/sglang/actions/runs/15517438930/job/43686932213#step:6:1921

### L1-2fc1299562  (L1, 2025-06-08, sha 2fc129956207, PR #6965)
TITLE: Remove unnecessary kernels of num_token_non_padded (#6965)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/cuda_graph_runner.py (+14/-10); python/sglang/srt/model_executor/forward_batch_info.py (+19/-15); python/sglang/srt/two_batch_overlap.py (+0/-3)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3712abfaf9  (L1, 2025-06-08, sha 3712abfaf920, PR #6970)
TITLE: Fuse routed scaling factor in deepseek (#6970)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+130/-14); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-0); python/sglang/srt/layers/quantization/blockwise_int8.py (+1/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+1/-0); python/sglang/srt/layers/quantization/fp8.py (+1/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+1/-0); python/sglang/srt/layers/quantization/w8a8_fp8.py (+1/-0); python/sglang/srt/layers/quantization/w8a8_int8.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+2/-1); benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py (+199/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-de1350ea20  (L1, 2025-06-08, sha de1350ea2053, PR #6977)
TITLE: Minor remove one kernel for DeepSeek (#6977)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+5/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-56ccd3c22c  (L1, 2025-06-09, sha 56ccd3c22c71, PR #6958)
TITLE: chore: upgrade flashinfer v0.2.6.post1 jit (#6958)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+8/-6); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-0); .github/workflows/vllm-dependency-test.yml (+1/-1); lmms-eval (+1/-0); python/sglang/srt/entrypoints/engine.py (+2/-2); python/sglang/srt/layers/multimodal.py (+3/-3); python/sglang/srt/layers/quantization/__init__.py (+2/-2); python/sglang/test/test_utils.py (+0/-1); scripts/ci_install_dependency.sh (+12/-3); (+4 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Integrate latest FlashInfer into SGLang for GB200 NVL72 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a979daac3b  (L1, 2025-06-09, sha a979daac3b60, PR #7013)
TITLE: Fallback to lower triton version for unfound fused moe configs (#7013)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+21/-3)
BODY: ## Motivation ⏎  ⏎ Followup of #5955 ⏎ After upgrading to torch 2.7, the new version of triton (3.3.1) lacks most of the configs tuned before. ⏎ This PR can let the fused moe use config with older triton version, when current triton version doesn't include tuned config. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6716b41786  (L1, 2025-06-09, sha 6716b4178690, PR #7023)
TITLE: Update default settings for blackwell (#7023)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: docker/Dockerfile.blackwell (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=257,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/model_executor/model_runner.py (+5/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Set flashinfer as default attention backend for blackwell ⏎ - Tune FusedMoE config for Dpsk v3 on blackwell ⏎ - Update flashinfer in blackwell docker image ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f6ebba537a  (L1, 2025-06-09, sha f6ebba537ae2, PR #6964)
TITLE: Support both approximate and exact expert distribution collection (#6964)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/expert_distribution.py (+67/-43); python/sglang/srt/models/deepseek_v2.py (+19/-16); python/sglang/srt/models/qwen3_moe.py (+14/-11); python/sglang/srt/server_args.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-81372f3bef  (L1, 2025-06-09, sha 81372f3bef65, PR #7029)
TITLE: Fix fused_moe triton configs (#7029)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1)
BODY: ## Motivation ⏎  ⏎ fix package-data for fused_moe triton configs. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fcde67b016  (L1, 2025-06-10, sha fcde67b016eb, PR #6833)
TITLE: CPU: map changes from developing branch in sgl-kernel (#6833)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: sgl-kernel/csrc/cpu/moe.cpp (+13/-4); sgl-kernel/csrc/cpu/moe_fp8.cpp (+19/-17); sgl-kernel/csrc/cpu/topk.cpp (+34/-2); sgl-kernel/csrc/cpu/decode.cpp (+503/-145); sgl-kernel/csrc/cpu/extend.cpp (+107/-9); sgl-kernel/csrc/cpu/gemm.h (+4/-0); sgl-kernel/csrc/cpu/gemm_fp8.cpp (+88/-91); sgl-kernel/csrc/cpu/interface.cpp (+2/-2); sgl-kernel/csrc/cpu/norm.cpp (+10/-4); sgl-kernel/csrc/cpu/qkv_proj.cpp (+79/-3); (+10 more)
LABELS: sgl-kernel, intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is to map changes from developing branch in sgl-kernel, including: ⏎  ⏎ - optimize decode perf by mapping algo from FlashMLA ⏎ - optimize fp8 performance. ⏎ - Add API of fuse q_a_proj and kv_a_proj ⏎ - Support non-contiguous input in norm ⏎ - Update topk API along with SGLang main branch ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-4f723edd3b  (L1, 2025-06-10, sha 4f723edd3baf, PR #7038)
TITLE: chore: bump v0.4.7 (#7038)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+6/-2); docker/Dockerfile.rocm (+1/-1); python/pyproject.toml (+1/-1); Makefile (+1/-1); benchmark/deepseek_v3/README.md (+1/-1); docs/references/setup_github_runner.md (+2/-2); docs/start/install.md (+6/-6); python/sglang/version.py (+1/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-cef6655b26  (L1, 2025-06-10, sha cef6655b2694, PR #7045)
TITLE: fix 24.12 docker (#7045)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+7/-7)
BODY: ## Motivation ⏎  ⏎ fix https://github.com/sgl-project/sglang/pull/7043 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6406408a70  (L1, 2025-06-10, sha 6406408a7085, PR #7037)
TITLE: Clean up server_args.py (#7037)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/deep_gemm.py (+1/-1); python/sglang/srt/managers/schedule_batch.py (+7/-6); python/sglang/srt/model_executor/cuda_graph_runner.py (+68/-37); python/sglang/srt/server_args.py (+183/-171); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+3/-0); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+3/-0); python/sglang/srt/utils.py (+13/-0)
BODY: - Move all EP related things into a single group ⏎ - Sort them correctly

### L1-d7c3e8e93d  (L1, 2025-06-10, sha d7c3e8e93d05, PR #7060)
TITLE: [fix] libmlx5.so already in base image (#7060)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.deepep (+1/-1)
BODY: ## Motivation ⏎ `/usr/lib/x86_64-linux-gnu/libmlx5.so` already in base image, cause build error from `ln -s` ⏎ ``` ⏎ root@n199-204-219:/tmp/sglang# docker build --network host --build-arg BASE_IMAGE=lmsysorg/sglang:v0.4.7-cu124 -f docker/Dockerfile.deepep -t d --no-cache . ⏎ [+] Building 200.4s (21/35)                                                                                                                                                        …[truncated]

### L1-84727a5139  (L1, 2025-06-11, sha 84727a51396a, PR #6919)
TITLE: [sgl-kernel] Add cuda kernel for moe_ep_silu_and_mul (#6919)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.reorder_aot
FILES: sgl-kernel/csrc/moe/ep_moe_silu_and_mul_kernel.cu (+115/-0); sgl-kernel/python/sgl_kernel/moe.py (+18/-0); sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/benchmark/bench_moe_silu_and_mul.py (+92/-0); sgl-kernel/csrc/common_extension.cc (+4/-0); sgl-kernel/include/sgl_kernel_ops.h (+8/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/tests/test_ep_moe_silu_and_mul_kernel.py (+142/-0)
LABELS: high priority, ready-to-merge
BODY: ## Motivation ⏎ Rewrite moe_ep_silu_and_mul Triton kernel in CUDA. Gains 10% performance improvement. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-7046e0fab7  (L1, 2025-06-12, sha 7046e0fab791, PR #7119)
TITLE: feat: update blackwell setup (#7119)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+2/-2); sgl-kernel/CMakeLists.txt (+10/-2); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-da47621ccc  (L1, 2025-06-13, sha da47621ccc4f, PR #7058)
TITLE: Minor speedup topk postprocessing (#7058)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+16/-8)
LABELS: high priority
BODY: (cherry picked from commit c234231e6d625a4273dda47f414560895f2be8ff) ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ 1.4% speedup in my test case ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8ab7d93c2e  (L1, 2025-06-13, sha 8ab7d93c2e0d, PR #7152)
TITLE: chore: bump v0.1.8.post1 (#7152)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-bec3e48402  (L1, 2025-06-13, sha bec3e484022a, PR #7155)
TITLE: Support new DeepGEMM format in per token group quant (part 2: srt) (#7155)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/quantization/fp8_kernel.py (+17/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b04df75acd  (L1, 2025-06-13, sha b04df75acdda, PR #7154)
TITLE: Fix DeepEP error in some environments (#7154)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+2/-0)
BODY: ## Motivation ⏎  ⏎ Use with https://github.com/deepseek-ai/DeepEP/pull/193 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8b8f2e7463  (L1, 2025-06-13, sha 8b8f2e74630c, PR #7153)
TITLE: Support new DeepGEMM input format in silu_and_mul_masked_post_quant_fwd (#7153)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+5/-0)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b4c41f7276  (L1, 2025-06-13, sha b4c41f7276e2, PR #7150)
TITLE: Refactor DeepGEMM integration (#7150)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+1/-5); python/sglang/srt/layers/moe/ep_moe/layer.py (+35/-32); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+5/-5); python/sglang/srt/layers/quantization/deep_gemm_wrapper/__init__.py (+1/-0); python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py (+22/-76); python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py (+26/-0); python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py (+95/-0); python/sglang/srt/layers/quantization/fp8_kernel.py (+6/-10); python/sglang/srt/layers/quantization/fp8_utils.py (+3/-4); python/sglang/srt/model_executor/model_runner.py (+6/-6); (+2 more)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-93cec4335f  (L1, 2025-06-13, sha 93cec4335fed, PR #7172)
TITLE: Support new DeepGEMM (#7172)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+8/-1); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+2/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-4); python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py (+7/-1); python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py (+18/-8); python/sglang/srt/layers/quantization/fp8_kernel.py (+20/-3); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+2/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-ed54bf9d19  (L1, 2025-06-14, sha ed54bf9d19f1, PR #7181)
TITLE: [fix] fix dsv3 weight loader tqdm and simplify shared experts fusion (#7181)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+11/-96)
BODY: ## Motivation ⏎  ⏎ Using `weights = [w for w in weights_list if w[0] not in names_to_remove]` to update `weights` will change it from async weight iterator to a list. Thats will make the tqdm looks very fast but the weight are still loading in the backround. Making the progress bar confusing.  ⏎  ⏎ acc ⏎ ``` ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319 ⏎ 100%|███████████████████████████████████████████████ …[truncated]

### L1-4473320380  (L1, 2025-06-14, sha 44733203800d, PR #7189)
TITLE: chore: bump v0.1.8.post2 (#7189)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+1/-1); .github/workflows/pr-test-sgl-kernel.yml (+3/-4); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-55561e2553  (L1, 2025-06-14, sha 55561e25533f, PR #7180)
TITLE: [fix] fix determine_num_fused_shared_experts (#7180)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+29/-47)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-ed89837cf4  (L1, 2025-06-14, sha ed89837cf42e, PR #7186)
TITLE: chore: upgrade sgl-kernel v0.1.8.post2 (#7186)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/cutlass_mla_backend.py (+1/-0); python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py (+5/-1)
BODY: ## Motivation ⏎  ⏎ should be merged after https://github.com/sgl-project/sglang/pull/7184 and new version of sgl-kernel bumped ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-349bb2c92a  (L1, 2025-06-14, sha 349bb2c92af4, PR #7198)
TITLE: Fix error when disabling new DeepGEMM (#7198)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+4/-2); python/sglang/srt/models/deepseek_v2.py (+4/-1)
BODY: ## Motivation ⏎  ⏎ local test on 8xB200 passes gsm8k ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0ae1e9a755  (L1, 2025-06-15, sha 0ae1e9a75573, PR #7221)
TITLE: refine fused_moe benchmark (#7221)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/benchmark_ep_pre_reorder_triton.py (+0/-101); benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py (+5/-1); benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py (+5/-0)
BODY: 

### L1-cfceb83d05  (L1, 2025-06-16, sha cfceb83d0576, PR #7207)
TITLE: Fix sampling for speculative decoding & simplify kernels (#7207)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-kernel/python/sgl_kernel/top_k.py (+11/-0); sgl-kernel/csrc/common_extension.cc (+5/-2); sgl-kernel/csrc/speculative/eagle_utils.cu (+32/-32); sgl-kernel/csrc/speculative/packbit.cu (+7/-3); sgl-kernel/csrc/speculative/speculative_sampling.cu (+23/-16); sgl-kernel/csrc/speculative/speculative_sampling.cuh (+23/-15); sgl-kernel/include/sgl_kernel_ops.h (+7/-1); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/python/sgl_kernel/speculative.py (+4/-0); sgl-kernel/tests/speculative/test_eagle_utils.py (+5/-6); (+1 more)
BODY: - Use a different coin for the final sampling in speculative sampling ⏎ - Remove redundant int32 - int64 copy in tree kernels ⏎  ⏎  ⏎ ``` ⏎  ⏎ ```

### L1-53a525bf33  (L1, 2025-06-16, sha 53a525bf3356, PR #7231)
TITLE: [Eagle] Fix kernel call after updating speculative sampling kernels (#7231)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile.blackwell (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/speculative/build_eagle_tree.py (+1/-1); python/sglang/srt/speculative/eagle_utils.py (+12/-21); python/sglang/srt/speculative/eagle_worker.py (+1/-1); test/srt/test_fa3.py (+7/-7)
BODY: Following https://github.com/sgl-project/sglang/pull/7207, we change the kernel calls. ⏎  ⏎ - Use a different coin for the final sampling in speculative sampling ⏎ - Remove some uncessary `to(torch.int32)` ⏎  ⏎ ``` ⏎  ⏎ ```

### L1-91a066ec6a  (L1, 2025-06-16, sha 91a066ec6a4a, PR #7234)
TITLE: Tiny remove comments about DeepEP on H20 (#7234)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+0/-32)
BODY: ## Motivation ⏎  ⏎ b/c we have this now https://github.com/deepseek-ai/DeepEP/pull/213 ⏎  ⏎ this pr should be safe to merge since it only affects comments instead of code ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5ca07eed90  (L1, 2025-06-16, sha 5ca07eed9018, PR #7247)
TITLE: [fix] fix DeepGEMM blackwell input quant & ut & fix style and log (#7247)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-2); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+2/-2); python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py (+6/-9); python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py (+7/-7); python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py (+3/-3); python/sglang/srt/layers/quantization/fp8_kernel.py (+4/-3); python/sglang/srt/layers/quantization/fp8_utils.py (+4/-3); python/sglang/srt/models/deepseek_v2.py (+4/-2); python/sglang/test/test_block_fp8.py (+1/-0); python/sglang/test/test_block_fp8_deep_gemm_blackwell.py (+252/-0)
LABELS: bug
BODY: ## Motivation ⏎  ⏎  ⏎ acc ⏎ ``` ⏎ # launch ⏎ python3 -m sglang.launch_server --model /dev/shm/DeepSeek-V3-0324 --tp 8 ⏎  ⏎ # test ⏎ while true; do (python3 benchmark/gsm8k/bench_sglang.py --num-questions 1319 --parallel 1319); done ⏎ 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [01:05<00:00, 20.13it/s] ⏎ Accuracy: 0.933 ⏎ Invalid: 0.000 ⏎ Latency: 66.251 s ⏎ Output throughput: …[truncated]

### L1-8e2363dc15  (L1, 2025-06-16, sha 8e2363dc15ea, PR #7125)
TITLE: fix amd EP MoE FP8 issue (#7125)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+30/-0)
BODY: ## Motivation ⏎  ⏎ To fix AMD EP MoE FP8 issue. See https://github.com/sgl-project/sglang/issues/7055. ⏎  ⏎ ## Modifications ⏎  ⏎ Add dtype transfer in EP path. ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Accuracy check ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server \ ⏎     --trust-remote-code \ ⏎     --chunked-prefill-size 131072 \ ⏎     --disable-cuda-graph \ ⏎     --disable-custom-all-reduce \ ⏎     --tp-size 8 \ ⏎     --enable-ep-moe \ ⏎     --model deepseek-ai/DeepSeek-V3 ⏎ ``` ⏎  ⏎ ` …[truncated]

### L1-405780bcf0  (L1, 2025-06-16, sha 405780bcf0d1, PR #7160)
TITLE: [amd] Opt dsv3 moe (#7160)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+10/-8)
BODY: ## Motivation ⏎  ⏎ Use new aiter moe implementation to enhance the performance of MoE fp8 computation of ds model ⏎  ⏎ The throughput can increase from 1.73 req/s to 1.88 req/s ⏎  ⏎ ``` ⏎ ============ Serving Benchmark Result ============ ⏎ Backend:                                 sglang ⏎ Traffic request rate:                    inf ⏎ Max request concurrency:                 128 ⏎ Successful requests:                     500 ⏎ Benchmark duration (s):        …[truncated]

### L1-094c116f7d  (L1, 2025-06-17, sha 094c116f7dc4, PR #6614)
TITLE: Update python API of activation, topk, norm and rope and remove vllm dependency (#6614)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+6/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-1); python/sglang/srt/layers/moe/topk.py (+91/-4); docker/Dockerfile.xeon (+1/-0); python/sglang/srt/custom_op.py (+5/-1); python/sglang/srt/layers/activation.py (+20/-3); python/sglang/srt/layers/layernorm.py (+28/-2); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+5/-2); python/sglang/srt/layers/quantization/fp8.py (+5/-1); python/sglang/srt/layers/quantization/utils.py (+4/-2); (+13 more)
LABELS: high priority, ready-to-merge, intel, cpu
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is to update python API of activation, topk, norm and rope calling by using torch.ops.sgl_kernel.xxx_cpu in CPU part, which includes optimized implementation. And we also remove vllm dependency and use sgl_kernel implementation instead. ⏎  ⏎ We also refactor `is_cpu` and use `SGLANG_USE_CPU_ENGINE=1` env to indicate run on CPU device before runtime. When `is_cpu()` is true, `cpu_has_amx_support()` is false, it will go int …[truncated]

### L1-712bf9ec9b  (L1, 2025-06-18, sha 712bf9ec9b51, PR #7319)
TITLE: [pd] optimize dockerfile for  pd disaggregation (#7319)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.deepep (+75/-42)
BODY: ## Modifications ⏎ optimize dockerfile for  pd disaggregation

### L1-20a503c7d1  (L1, 2025-06-18, sha 20a503c7d16e, PR #7331)
TITLE: fix: resolve blackwell deepep image issue (#7331)
SOURCES: subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.blackwell (+9/-13)
BODY: ## Motivation ⏎  ⏎ - reduce size ⏎ - use ubuntu 22.04 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-09ae5b20f3  (L1, 2025-06-19, sha 09ae5b20f312, PR #7096)
TITLE: Merge PDLB (Prefill-Decode Load Balancer) into SGLang Router (#7096)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+12/-2); python/sglang/srt/disaggregation/mini_lb.py (+27/-3); sgl-router/Cargo.toml (+3/-1); sgl-router/py_src/sglang_router/launch_router.py (+107/-6); sgl-router/py_test/test_launch_router.py (+116/-2); sgl-router/src/lib.rs (+85/-18); sgl-router/src/openai_api_types.rs (+704/-0); sgl-router/src/pd_router.rs (+1002/-0); sgl-router/src/pd_types.rs (+245/-0); sgl-router/src/request_adapter.rs (+264/-0); (+3 more)
LABELS: high priority
BODY: This PR implements Phases 1 & 2 of the PDLB merger design, adding prefill-decode disaggregated routing capabilities to the SGLang Router while maintaining full backward compatibility. ⏎  ⏎ ##  Overview ⏎  ⏎ This PR implements Phases 1 & 2 of the PDLB merger design, adding prefill-decode disaggregated routing capabilities to the SGLang Router. ⏎  ⏎ ###  Architecture ⏎  ⏎ #### High-Level Design ⏎  ⏎ ```mermaid ⏎ graph LR ⏎     subgraph "Client Request" ⏎        …[truncated]

### L1-906dbc34f1  (L1, 2025-06-19, sha 906dbc34f164, PR #7343)
TITLE: [Docker] optimize dockerfile  remove deepep and blackwell merge it to… (#7343)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: -
FILES: .github/workflows/release-docker-deepep.yml (+0/-47); docker/Dockerfile (+89/-42); docker/Dockerfile.blackwell (+0/-215); docker/Dockerfile.deepep (+0/-114); .github/workflows/release-docker-blackwell.yml (+0/-36); .github/workflows/release-docker.yml (+15/-4)
BODY: ## Motivation ⏎  ⏎ reduce docker image nums and its size

### L1-5962e70d8d  (L1, 2025-06-22, sha 5962e70d8d37, PR #7327)
TITLE: FlashInfer NVFP4 MoE with EP & 2-stream shared expert (#7327)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+3/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+62/-10); python/sglang/srt/layers/quantization/modelopt_quant.py (+62/-8); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+39/-1); python/sglang/srt/server_args.py (+15/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Integrates Flashinfer NVFP4 cutlass MoE kernel from https://github.com/flashinfer-ai/flashinfer/pull/1113  ⏎ You can enable it with `--enable-flashinfer-moe`. `--enable-ep-moe` is also supported for this backend. ⏎  ⏎ ## Example usage ⏎  ⏎ TP 8 ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path nvidia/DeepSeek-R1-FP4 --trust-remote-code --quantization modelopt_fp4 --tp 8 --enable-flashinfer-moe ⏎ python3 benchmark/gsm8k/bench_sglang.py …[truncated]

### L1-bd4f581896  (L1, 2025-06-22, sha bd4f58189669, PR #7391)
TITLE: Fix torch compile run (#7391)
SOURCES: path_core, symbol_pickaxe, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-1); docker/Dockerfile.rocm (+1/-1); python/sglang/srt/layers/quantization/fp8.py (+8/-8); scripts/amd_ci_start_container.sh (+1/-1)
LABELS: bug, high priority, aiter
DEEP_STUDY: deep-study correctness case sglang:bd4f581896: class=hardware_compiler_specific; symptom=compile_or_build_failure; introducing=unknown
BODY: ## Motivation ⏎  ⏎ Run the ci test "test/srt/test_mla_deepseek_v3.py" hit the torch-compile errors ⏎  ⏎ ## Modifications ⏎  ⏎ Fix the problematic part that reported by TORCH_LOGS="+dynamo,+recompiles" TORCHDYNAMO_VERBOSE=1 debug log ⏎  ⏎ ## Checklist ⏎  ⏎ - [✓] Format your code according to the [Code Formatting with Pre-Commit](https://docs.sglang.ai/references/contribution_guide.html#code-formatting-with-pre-commit).

### L1-30f2a44a96  (L1, 2025-06-22, sha 30f2a44a9634, PR #7361)
TITLE: [misc] Add PD service discovery support in router (#7361)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+15/-3); sgl-router/Cargo.toml (+1/-0); sgl-router/README.md (+188/-9); sgl-router/py_src/sglang_router/launch_router.py (+42/-14); sgl-router/py_test/test_launch_router.py (+81/-8); sgl-router/src/lib.rs (+24/-6); sgl-router/src/pd_router.rs (+142/-7); sgl-router/src/pd_types.rs (+25/-0); sgl-router/src/router.rs (+154/-0); sgl-router/src/service_discovery.rs (+683/-66); (+1 more)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ Add Kubernetes service discovery for PD (Prefill-Decode) disaggregated routing with unified event handling and automatic pod classification. ⏎  ⏎ ## Modifications ⏎  ⏎ ### Implementation ⏎  ⏎ - Core Architecture: Unified pod event handler supports both regular and PD modes. Pod type detection via Kubernetes labels automatically classifies pods as Prefill, Decode, or Regular. ⏎ - Dynamic Worker Management: Real-time addition/removal of w …[truncated]

### L1-3cee035e99  (L1, 2025-06-22, sha 3cee035e99ec, PR #7445)
TITLE: add fused moe config for qwen3 in triton3.3.1 (#7445)
SOURCES: path_config_only, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=128,N=384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0)
BODY: ## Motivation ⏎  ⏎  ⏎ add fused moe triton 3.3.1 config for qwen3 moe. ⏎ co-auther @ispobock  ⏎  ⏎ ## benchmark ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path Qwen/Qwen3-235B-A22B-FP8 --tp 4 --disable-radix-cache ⏎  ⏎ python3 -m sglang.bench_serving --backend sglang-oai  --dataset-name random --random-input-len 1000 --random-output-len 1000 --random-range-ratio 1 --num-prompts 128 --max-concurrency 16 ⏎  ⏎ # before tuning ⏎ ============ Serving Benchmar …[truncated]

### L1-ac5010e0ba  (L1, 2025-06-22, sha ac5010e0ba14, PR #7451)
TITLE: Fix CUDA Graph Check under Deepep with DP FFN (#7451)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/cuda_graph_runner.py (+13/-11); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+14/-13); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+13/-16)
LABELS: ready-to-merge
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-506c4928f5  (L1, 2025-06-23, sha 506c4928f546, PR #6821)
TITLE: feat: integrate deepgemm into EPMoE (#6821)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+159/-2); python/sglang/srt/layers/moe/ep_moe/layer.py (+174/-1); python/sglang/test/test_block_fp8_ep.py (+1/-0); sgl-kernel/benchmark/bench_moe_ep_post_reorder.py (+1/-0); sgl-kernel/tests/test_ep_moe_post_reorder_kernel.py (+1/-0)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎  ⏎ For normal EPMoE (no DeepEP), integrate DeepGEMM as an option. ⏎  ⏎ This PR builds upon the work from [pr5805](https://github.com/sgl-project/sglang/pull/5805) Credit goes to @TianQiLin666666, who authored most of the code presented here. ⏎ ``` ⏎  ⏎ ``` ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Performance ⏎ two h20 node  ⏎  ⏎ node 1 ⏎ ``` ⏎ python -m sglang.launch_server --model-path /path/to/DeepSeek-V3/    --trust-remote-code --tp-size 16 --enable-d …[truncated]

### L1-55e03b10c4  (L1, 2025-06-23, sha 55e03b10c456, PR #7457)
TITLE: Fix a bug in BatchTokenIDOut & Misc style and dependency updates (#7457)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+6/-8); sgl-kernel/CMakeLists.txt (+10/-10); .github/workflows/pr-test.yml (+5/-1); python/sglang/srt/managers/schedule_batch.py (+1/-0); python/sglang/srt/managers/scheduler.py (+8/-1); python/sglang/srt/managers/tokenizer_manager.py (+1/-1); python/sglang/srt/server_args.py (+2/-3); python/sglang/srt/utils.py (+3/-7); sgl-kernel/python/sgl_kernel/sampling.py (+1/-1)
BODY: 

### L1-15f3401343  (L1, 2025-06-23, sha 15f34013432f, PR #7376)
TITLE: Fix MTP with Deepseek R1 Fp4 (#7376)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-0); python/sglang/srt/models/deepseek_nextn.py (+6/-0); python/sglang/srt/models/deepseek_v2.py (+8/-1)
LABELS: bug, high priority
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/issues/7365 ⏎  ⏎ ## Modifications ⏎  ⏎ Update quant config to None for MTP module in Deepseek modelopt fp4. ⏎  ⏎ It's not quantized. See https://github.com/NVIDIA/TensorRT-LLM/blob/5d4ab47d5b333296d33d62dde095310d7eba0b48/tensorrt_llm/_torch/models/modeling_deepseekv3.py#L682 ⏎  ⏎ Tested as follow: ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --port=7080 --model-path=/root/.cache/huggingface/hub/models--n …[truncated]

### L1-e846d95ef6  (L1, 2025-06-23, sha e846d95ef6ba, PR #7490)
TITLE: chore: bump sgl-kernel v0.2.0 (#7490)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/Makefile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e984d5073b  (L1, 2025-06-24, sha e984d5073bc8, PR #7423)
TITLE: enable aiter_biased_grouped_topk kernel (#7423)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+26/-0); python/sglang/srt/model_executor/cuda_graph_runner.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+2/-1)
BODY: ## Motivation ⏎  ⏎ ## Modifications ⏎  ⏎ ## Checklist

### L1-7151194bb2  (L1, 2025-06-24, sha 7151194bb2f9, PR #7439)
TITLE: Remove cumsum_buffer initilization (#7439)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+7/-2)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ `cumsum_buffer` is not needed to be initialized here. ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model Qwen/Qwen3-235B-A22B-FP8 --tp 4 ⏎  ⏎ Accuracy: 0.944 ⏎ Invalid: 0.000 ⏎ Latency: 104.687 s ⏎ Output throughput: 1772.852 token/s ⏎ ``` ⏎ mian: ⏎ ``` ⏎ +----+-------------------+--------------------+---------------------+----------------+------------------+---------------+----------------+------------------+---------------+---------------- …[truncated]

### L1-755f314785  (L1, 2025-06-24, sha 755f314785fc, PR #7268)
TITLE: [AMD] add aiter fused moe in DeepEP path (#7268)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+79/-12); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+19/-2)
BODY: ## Motivation ⏎  ⏎ For better performance in DeepEP path ⏎  ⏎ ## Modifications ⏎  ⏎ add aiter fused moe in DeepEP path ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Accuracy and performance ⏎  ⏎ without aiter fused_moe: ⏎ ``` ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 ⏎ 100%|██████████████████████████████████████████████████████████████| 1319/1319 [05:20<00:00,  4.12it/s] ⏎ Accuracy: 0.946 ⏎ Invalid: 0.000 ⏎ Latency: 321.660 s ⏎ Output throughput:  …[truncated]

### L1-57ab776910  (L1, 2025-06-24, sha 57ab77691038, PR #7437)
TITLE: Fuse sorted_token_ids padding to moe_align_block_size kernel (#7437)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.align.cuda_aot
FILES: sgl-kernel/csrc/common_extension.cc (+2/-1); sgl-kernel/csrc/moe/moe_align_kernel.cu (+57/-5); sgl-kernel/csrc/torch_extension_rocm.cc (+2/-1); sgl-kernel/include/sgl_kernel_ops.h (+2/-1); sgl-kernel/python/sgl_kernel/moe.py (+2/-0); sgl-kernel/benchmark/bench_moe_align_block_size.py (+53/-48); sgl-kernel/tests/test_moe_align.py (+42/-11)
BODY: ## Motivation ⏎  ⏎ Fuse `sorted_token_ids` padding to `moe_align_block_size`. In some shapes, it can perform better, so make it optional. ⏎  ⏎ ## Benchmark ⏎ ``` ⏎      num_tokens  num_experts  topk         SGL  SGL Fusion       Triton ⏎ 0           1.0          8.0   1.0   19.088000   17.216001    53.504001 ⏎ 1           1.0          8.0   2.0   19.072000   17.535999    32.768000 ⏎ 2           1.0          8.0   4.0   19.152001   17.503999    31.872001 ⏎  …[truncated]

### L1-7eb47b0f3d  (L1, 2025-06-25, sha 7eb47b0f3d0c, PR #6641)
TITLE: [CPU] [BF16] Call fused_experts_cpu, weight_packed_linear and bmm_cpu kernel in DeepSeek model (#6641)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_native.py (+7/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+73/-14); python/sglang/srt/layers/linear.py (+18/-1); python/sglang/srt/layers/logits_processor.py (+12/-3); python/sglang/srt/layers/vocab_parallel_embedding.py (+14/-1); python/sglang/srt/models/deepseek_v2.py (+145/-1); python/sglang/srt/utils.py (+71/-0); sgl-kernel/csrc/cpu/gemm.cpp (+2/-2); test/srt/cpu/test_gemm.py (+1/-1)
LABELS: intel, cpu
BODY: ## Motivation ⏎  ⏎ When CPU has AMX support, replace `moe_forward_native` with `fused_experts_cpu`, replace `F.linear` and `torch.matmul` with `weight_packed_linear`, add `AttnForwardMethod.MLA_FUSED_ROPE_CPU` to optimize the performance. ⏎  ⏎ https://github.com/sgl-project/sglang/pull/6408 (merged), https://github.com/sgl-project/sglang/pull/6614 and https://github.com/sgl-project/sglang/pull/6833 need to be landed first and the current PR will work …[truncated]

### L1-ce3a3e8783  (L1, 2025-06-27, sha ce3a3e878328, PR #7581)
TITLE: Move multimodal processors into a separate folder (#7581)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+1/-2); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-2); python/sglang/math_utils.py (+0/-8); python/sglang/srt/configs/janus_pro.py (+1/-1); python/sglang/srt/layers/quantization/fp8_kernel.py (+1/-1); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-2); python/sglang/srt/managers/io_struct.py (+1/-1); python/sglang/srt/managers/mm_utils.py (+0/-2); python/sglang/srt/managers/multimodal_processor.py (+2/-4); python/sglang/srt/models/llava.py (+5/-5); (+19 more)
BODY: - Redo https://github.com/sgl-project/sglang/pull/6495 ⏎ - remove unused imports

### L1-a5317b2fd3  (L1, 2025-06-27, sha a5317b2fd3dd, PR #6769)
TITLE: [CPU] add optimizations for INT8 and FP8 DeepSeek (#6769)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-1); python/sglang/srt/layers/quantization/fp8.py (+43/-0); python/sglang/srt/layers/quantization/moe_wna16.py (+1/-1); python/sglang/srt/layers/quantization/w8a8_int8.py (+51/-1); python/sglang/srt/models/deepseek_v2.py (+83/-0)
LABELS: intel, cpu
BODY: ## Motivation ⏎ Call the below kernels in DeepSeek to optimize INT8 and FP8: ⏎ INT8 linear: `int8_scaled_mm_with_quant` ⏎ FP8 linear: `fp8_scaled_mm` ⏎ INT8 and FP8 shared_expert: `shared_expert_cpu` ⏎ INT8 and FP8 MoE: `fused_experts_cpu` ⏎  ⏎ For bmm, currently we don't support weight to be FP8, we convert weight to BF16 (only for CPU).

### L1-82eccae44e  (L1, 2025-06-28, sha 82eccae44e86, PR #7309)
TITLE: Let ep_scatter support arbitrary strides / ue8m0 format (#7309)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+22/-6)
LABELS: high priority, ready-to-merge
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-00c7b1ad07  (L1, 2025-06-28, sha 00c7b1ad0788, PR #7310)
TITLE: Let EP prefill support new DeepGEMM (#7310)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+24/-7); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+7/-1)
LABELS: high priority
BODY: ## Motivation ⏎  ⏎ please merge #7309 first (o/w this PR will error) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0c9c6c75a8  (L1, 2025-06-29, sha 0c9c6c75a809, PR #7580)
TITLE: Move files related to EPLB (#7580)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-2); python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py (+1/-3); python/sglang/srt/layers/moe/topk.py (+3/-3); python/sglang/srt/eplb/__init__.py (+0/-0); python/sglang/srt/eplb/eplb_algorithms/__init__.py (+1/-1); python/sglang/srt/eplb/eplb_algorithms/deepseek.py (+0/-0); python/sglang/srt/eplb/eplb_algorithms/deepseek_vec.py (+0/-0); python/sglang/srt/eplb/eplb_manager.py (+2/-4); python/sglang/srt/eplb/eplb_simulator/__init__.py (+0/-0); python/sglang/srt/eplb/eplb_simulator/reader.py (+1/-1); (+12 more)
BODY: ## Motivation ⏎  ⏎ suggested by @merrymercy  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-22352d47a9  (L1, 2025-06-29, sha 22352d47a93e, PR #7632)
TITLE: Improve streaming, log_level, memory report, weight loading, and benchmark script (#7632)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.router_py
FILES: python/sglang/srt/layers/moe/router.py (+60/-22); docs/backend/server_arguments.md (+1/-1); python/sglang/bench_one_batch_server.py (+14/-2); python/sglang/bench_serving.py (+0/-1); python/sglang/srt/configs/internvl.py (+2/-3); python/sglang/srt/entrypoints/http_server.py (+23/-6); python/sglang/srt/layers/elementwise.py (+76/-12); python/sglang/srt/managers/configure_logging.py (+1/-1); python/sglang/srt/managers/io_struct.py (+3/-3); python/sglang/srt/managers/mm_utils.py (+52/-0); (+14 more)
BODY: - support disable streaming in bench_one_batch_server.py ⏎ - add more log_request_level ⏎ - support cpu quantization for weight loading  ⏎ - simplify tokenizer manager ⏎ - support crash dump and replay ⏎ - fix CI

### L1-7248272ccc  (L1, 2025-06-29, sha 7248272ccc21, PR #7627)
TITLE: Add dsv3 router gemm kernel (#7627)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/csrc/common_extension.cc (+3/-0); sgl-kernel/include/sgl_kernel_ops.h (+1/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/benchmark/bench_dsv3_router_gemm.py (+56/-0); sgl-kernel/csrc/gemm/dsv3_router_gemm.cu (+286/-0); sgl-kernel/python/sgl_kernel/gemm.py (+18/-0); sgl-kernel/tests/test_dsv3_router_gemm.py (+32/-0)
BODY: ## Motivation ⏎  ⏎ Adapt router gemm kernel from [trtllm](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu) ⏎  ⏎ Note that this kernel takes two bf16 tensors as input, and outputs a fp32 tensor. ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ For next PR: Integrate into dspk-r1 fp4 ⏎  ⏎  ⏎ ## Accuracy Test ⏎ ```bash ⏎ python3 tests/test_dsv3_router_gemm.py ⏎ ``` ⏎ <img width="1259" alt="截屏2025-06-29 17 08 17" s …[truncated]

### L1-392e441ad1  (L1, 2025-06-30, sha 392e441ad17c, PR #7663)
TITLE: chore: upgrade flashinfer v0.2.7 jit (#7663)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-637bfee448  (L1, 2025-06-30, sha 637bfee448a8, PR #7675)
TITLE: chore: bump sgl-kernel v0.2.1 (#7675)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-f9eb04ddb2  (L1, 2025-07-01, sha f9eb04ddb26a, PR #7676)
TITLE: upgrade sgl kernel to 0.2.1 for main (#7676)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ Making sure compatibility with latest kernel update. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Checklist
