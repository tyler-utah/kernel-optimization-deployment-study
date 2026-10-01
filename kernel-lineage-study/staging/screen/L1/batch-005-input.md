### L1-0c9174108a  (L1, 2025-09-28, sha 0c9174108abe, PR #10701)
TITLE: Unify SGL Kernel Releases (#10701)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+50/-12); .github/workflows/pr-test.yml (+13/-17); sgl-kernel/python/sgl_kernel/__init__.py (+178/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Dynamically load common_ops module for SM90 (compiled with fast math) and SM 100 (without fast math) during initialization based on compute architecture so that one SGLang kernel binary can cover all versions with different configurations.  ⏎  ⏎ CMake will compile two common_ops libraries and install them in two different locations. ⏎  ⏎ Update images used in CI to be 12.9. Remove the steps to build kernel for 12.4 and 12.8. ⏎  ⏎ # …[truncated]

### L1-5942fdb480  (L1, 2025-09-29, sha 5942fdb48009, PR #11054)
TITLE: chore: upgrade cutedsl 4.2.1 (#11054)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/pyproject_other.toml (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3a641d9085  (L1, 2025-09-29, sha 3a641d90857c, PR #11056)
TITLE: chore: upgrade sgl-kernel 0.3.13 (#11056)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/pyproject_other.toml (+2/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ ``` ⏎ pip install sgl-kernel --upgrade ⏎ ``` ⏎  ⏎ One wheel for all. ⏎  ⏎ cu124/cu126/cu128/cu129 hopper ⏎ cu128/cu129 blackwell ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-1237aa19ce  (L1, 2025-09-30, sha 1237aa19ce63, PR #11099)
TITLE: [Auto Sync] Update fused_moe_triton_config.py (20250930) (#11099)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py (+6/-2)
LABELS: run-ci
BODY: Sync changes from commit `d078e69d`. ⏎  ⏎ **Files Changed:** ⏎ - python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py ⏎  ⏎ Author: Cheng Wan <54331508+ch-wan@users.noreply.github.com> ⏎  ⏎ --- ⏎  ⏎ *This is an automated PR created by scripts/copy_from_oss.py.*

### L1-a6cc86df9d  (L1, 2025-09-30, sha a6cc86df9d3e, PR #11081)
TITLE: Fix DSR1 accuracy for flashinfer_trtllm MoE with FP8 quantization (#11081)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-3); python/sglang/srt/server_args.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/pull/10758 introduced a change to only apply w13 -> w31 weight mapping with flashinfer_trtllm for NVFP4 quantization, however FP8 quantization is also supported and needs this mapping too, otherwise GSM8K accuracy went to 0. ⏎  ⏎ ## Modifications ⏎  ⏎ Apply w13 -> w31 for FP8 quantization also. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Command ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek …[truncated]

### L1-47488cc353  (L1, 2025-10-01, sha 47488cc3538c, PR #11075)
TITLE: docker: x86 dev builds for hopper and blackwell (#11075)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+4/-3); .github/workflows/release-docker-dev.yml (+11/-12); .github/workflows/release-docker.yml (+11/-11); .github/workflows/release-whl-kernel.yml (+1/-1)
LABELS: run-ci
BODY: This allows you to do `docker pull lmsysorg:sglang:dev` and depending on your machine it will "just run". Also edit runner names and fix some typos

### L1-ac1f2928ae  (L1, 2025-10-01, sha ac1f2928aef9, PR #10760)
TITLE: feat: add fast_decode_plan from flashinfer, flashinfer to 0.4.0rc3 (#10760)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/pyproject_other.toml (+3/-3); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+44/-186)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ We maintain fast_decode_plan at FI side. C++ API changes wont break sgl on fast_decode_plan function with this PR. Might be removed after we adopt GPU-based planning. ⏎  ⏎ https://github.com/sgl-project/sglang/pull/10634 ⏎  ⏎ https://github.com/flashinfer-ai/flashinfer/issues/1720 ⏎ https://github.com/flashinfer-ai/flashinfer/pull/1745 ⏎ https://github.com/flashinfer-ai/flashinfer/pull/1757 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests …[truncated]

### L1-73d4a5f879  (L1, 2025-10-01, sha 73d4a5f8791f, PR #10735)
TITLE: Organize spec-related data structures (#10735)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-5); python/sglang/srt/constrained/outlines_jump_forward.py (+1/-1); python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py (+1/-1); python/sglang/srt/layers/attention/aiter_backend.py (+9/-14); python/sglang/srt/layers/attention/ascend_backend.py (+4/-4); python/sglang/srt/layers/attention/base_attn_backend.py (+3/-3); python/sglang/srt/layers/attention/cutlass_mla_backend.py (+3/-3); python/sglang/srt/layers/attention/flashattention_backend.py (+5/-6); python/sglang/srt/layers/attention/flashinfer_backend.py (+17/-19); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+11/-15); (+22 more)
LABELS: high priority, run-ci
BODY: A step to merge #9334 into main and clarify all data structures for separate decoding/overlap scheduling. ⏎  ⏎ Depends on #10715 ⏎  ⏎ *** ⏎  ⏎ - Introduce `SpecInput` class to unify `EagleDraftInput`, `EagleVerifyInput`, `LookaheadVerifyInput`, and more speculative algorithms' inputs ⏎ - Introduce `SpecInputType` to avoid direct assert `isinstance` in different attention backends and runners. ⏎ - Separate eagle info types out of `eagle_utils`. ⏎ - Move cu …[truncated]

### L1-2d62af6be5  (L1, 2025-10-01, sha 2d62af6be517, PR #11123)
TITLE: Fix metrics and request tracing (TimeStats) (#11123)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+20/-21); python/sglang/srt/disaggregation/decode.py (+17/-3); python/sglang/srt/disaggregation/prefill.py (+7/-2); python/sglang/srt/disaggregation/utils.py (+1/-1); python/sglang/srt/managers/schedule_batch.py (+10/-10); python/sglang/srt/managers/schedule_policy.py (+8/-4); python/sglang/srt/managers/scheduler.py (+82/-89); python/sglang/srt/managers/scheduler_metrics_mixin.py (+96/-33); python/sglang/srt/managers/scheduler_output_processor_mixin.py (+3/-2); python/sglang/srt/managers/tokenizer_communicator_mixin.py (+81/-0); (+3 more)
LABELS: run-ci
BODY: 

### L1-0b9dfba787  (L1, 2025-10-02, sha 0b9dfba78700, PR #10263)
TITLE: Support dispatch low latency (#10263)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+11/-0); python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py (+38/-25); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+19/-3); python/sglang/srt/layers/quantization/modelopt_quant.py (+11/-1); python/sglang/srt/models/deepseek_v2.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ previous PR: https://github.com/sgl-project/sglang/pull/10120 ⏎ co-author: @kaixih (who did 10120) ⏎  ⏎ (for the one who merges this PR - please add @kaixih as co-author since he did 10120) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5e786cca3a  (L1, 2025-10-02, sha 5e786cca3a9a, PR #10422)
TITLE: Support single batch overlap (#10422)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+16/-3); python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py (+14/-0); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+47/-14); python/sglang/srt/layers/moe/utils.py (+10/-0); docs/advanced_features/server_arguments.md (+1/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+11/-0); python/sglang/srt/models/deepseek_v2.py (+12/-3); python/sglang/srt/server_args.py (+6/-0); python/sglang/srt/single_batch_overlap.py (+151/-0)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ This PR is mainly for my case (nvlink deepep and cutedsl gemm), while https://github.com/sgl-project/sglang/pull/9660 focus on H20 (rdma deepep and deepgemm). I try to make the code somehow general s.t. after this one is merged, 9660 can be merged without code duplication. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b65db0287b  (L1, 2025-10-02, sha b65db0287b55, PR #11163)
TITLE: Tiny cleanup deepseek_v2.py (#11163)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-6); python/sglang/srt/models/deepseek_v2.py (+31/-32)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ cc @merrymercy  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-12d6818380  (L1, 2025-10-02, sha 12d681838048, PR #11130)
TITLE: Tiny fix ep_gather behavior different in CI (#11130)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-666da3d59f  (L1, 2025-10-04, sha 666da3d59fa6, PR #11012)
TITLE: [fix]enable flashmla when using draft model P/D attention select (#11012)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/flashmla_backend.py (+4/-2); python/sglang/srt/speculative/eagle_worker.py (+7/-0); test/srt/test_flashmla.py (+3/-3)
LABELS: run-ci
BODY: ## Motivation ⏎ 1. on #9755 , when using flashmla , `draft_extend_attn_backend` will report an error when initialized, which is inconsistent with the pre-refactoring version. So I added unimplemented select for `_create_backend ` . ⏎ 2. add mutile token support for mla ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ ``` ⏎ python3 bench_sglang.py  --num-shots 8 --num-questions 1319 --parallel 50 --port 8080 ⏎  ⏎ Accuracy: 0.967 ⏎ ``` ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ ## Ben …[truncated]

### L1-e0b2d3eebe  (L1, 2025-10-05, sha e0b2d3eebebd, PR #11194)
TITLE: [Feature] Add a fast-topk to sgl-kernel for DeepSeek v3.2 (#11194)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-kernel/python/sgl_kernel/top_k.py (+29/-0); sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/csrc/common_extension.cc (+7/-0); sgl-kernel/csrc/elementwise/topk.cu (+422/-0); sgl-kernel/include/sgl_kernel_ops.h (+8/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-1); sgl-kernel/tests/test_topk.py (+120/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎  ⏎ This kernel is written for DeepSeek v3.2 indexer. Indexer of Deepseek Sparse Attention requires a variable-length topk kernel, which is extremely crucial for  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ This is adapted from the [official tilelang implementation](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py), but we further optimized the performance a little and implemented some naive kernel fusion which can …[truncated]

### L1-41763ba079  (L1, 2025-10-05, sha 41763ba07945, PR #11237)
TITLE: Remove gdrcopy check in ci_install_deepep.sh (#11237)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: scripts/ci/ci_install_deepep.sh (+0/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-baee08601b  (L1, 2025-10-05, sha baee08601bea, PR #10048)
TITLE: [quantization] Enable aiter mxfp4 fused_moe for Quark (#10048)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/quark/quark_moe.py (+7/-1)
LABELS: run-ci
BODY: aiter's moe_sorting requires topk_weights to be FP32, insert cast to fp32 before calling fused_moe. ⏎  ⏎ Together with #10042 enables quark llama4 fp4 model. ⏎ Tested on ROCm MI355.

### L1-f8924ad74b  (L1, 2025-10-05, sha f8924ad74b87, PR #11242)
TITLE: update sgl kernel version to 0.3.14.post1 (#11242)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/pyproject_other.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: run-ci
BODY: 

### L1-efbc687c28  (L1, 2025-10-06, sha efbc687c2817, PR #11061)
TITLE: Support DeepSeek V3.2 Exp (#11061)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+120/-71); python/sglang/srt/configs/model_config.py (+30/-0); python/sglang/srt/disaggregation/ascend/transfer_engine.py (+47/-9); python/sglang/srt/layers/attention/ascend_backend.py (+98/-4); python/sglang/srt/layers/attention/attention_registry.py (+7/-0); python/sglang/srt/layers/attention/base_attn_backend.py (+9/-0); python/sglang/srt/layers/attention/hybrid_attn_backend.py (+7/-0); python/sglang/srt/layers/attention/npu_ops/mla_preprocess.py (+94/-1); python/sglang/srt/layers/attention/nsa/dequant_k_cache.py (+163/-0); python/sglang/srt/layers/attention/nsa/index_buf_accessor.py (+354/-0); (+19 more)
LABELS: enhancement, high priority, amd, speculative-decoding, blackwell, npu, run-ci
BODY: We now support DeepSeek V3.2. Try this model in the tracking issue #11060  ⏎  ⏎ --- ⏎  ⏎ Authors: General: @fzyzcjy @DarkSharpness @hnyls2002 @hebiao064 @Fridge003; AMD: @HaiShaw @kkHuang-amd @hubertlu-tw; NPU: @ZhengdQin

### L1-5ee777c98f  (L1, 2025-10-06, sha 5ee777c98ff5, PR #11219)
TITLE: [router] add ipv6 support across all components (#11219)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+1/-1); sgl-router/benches/request_processing.rs (+3/-8); sgl-router/py_src/sglang_router/router_args.py (+5/-5); sgl-router/py_test/unit/test_arg_parser.py (+1/-1); sgl-router/src/config/types.rs (+5/-5); sgl-router/src/core/worker.rs (+16/-11); sgl-router/src/core/worker_builder.rs (+20/-9); sgl-router/src/grpc_client/sglang_scheduler.rs (+16/-3); sgl-router/src/lib.rs (+1/-1); sgl-router/src/main.rs (+2/-2); (+4 more)
LABELS: feature, router, run-ci
BODY: ## Checklist

### L1-2fcd56eaf6  (L1, 2025-10-07, sha 2fcd56eaf6d7, PR #11303)
TITLE: [router] add get server info and get model info in grpc server (#11303)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/entrypoints/grpc_server.py (+90/-0); python/sglang/srt/grpc/sglang_scheduler.proto (+59/-0); python/sglang/srt/grpc/sglang_scheduler_pb2.py (+11/-3); python/sglang/srt/grpc/sglang_scheduler_pb2.pyi (+62/-0); python/sglang/srt/grpc/sglang_scheduler_pb2_grpc.py (+88/-0); sgl-router/src/grpc_client/sglang_scheduler.rs (+24/-0); sgl-router/src/proto/sglang_scheduler.proto (+59/-0)
LABELS: router, run-ci
BODY: ### Changes ⏎ 1. add new methods for get server info and get model info ⏎ 2. updates protobuf to support those new endpoints ⏎  ⏎ ### Note ⏎ internal state is purposely ignored in get server info ⏎ this will be streamed back in each token generation in the future ⏎  ⏎ ### Validations ⏎ ```bash ⏎ grpcurl -plaintext localhost:30000 sglang.grpc.scheduler.SglangScheduler/GetModelInfo ⏎ { ⏎   "model_path": "/raid/models/meta-llama/Llama-3.1-8B-Instruct", ⏎   "toke …[truncated]

### L1-cd4b39a900  (L1, 2025-10-07, sha cd4b39a90023, PR #11205)
TITLE: [quantization] Properly ignore quantization for layers excluded in quant_config (#11205)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-9); python/sglang/srt/layers/quantization/quark/quark.py (+3/-1)
LABELS: run-ci
BODY: 

### L1-4f42c8cd3e  (L1, 2025-10-07, sha 4f42c8cd3e65, PR #11068)
TITLE: [sgl-kernel] Support float64 moe_sum_reduce cuda kernel (#11068)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.reduce.moe_sum_reduce
FILES: sgl-kernel/csrc/moe/moe_sum_reduce.cu (+171/-71); sgl-kernel/benchmark/bench_sum_scale.py (+53/-19)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ The deterministic feature introduced float64 data type in MLA test. The current moe_sum_reduce cuda kernel does not cover this data type. So as when using this new cuda kernel like in https://github.com/sgl-project/sglang/pull/10654 , the following srt test failure occurred: ⏎  ⏎ ``` ⏎ python test/srt/hicache/test_hicache_mla.py ⏎ ... ⏎ [2025-09-29 02:36:08] INFO:     127.0.0.1:35010 - "POST /v1/chat/completions HTTP/1.1" 200 OK ⏎ [2 …[truncated]

### L1-3c06b673af  (L1, 2025-10-07, sha 3c06b673aff9, PR #11211)
TITLE: [8/N] MoE Refactor: deprecate `EPMoE` (#11211)
SOURCES: path_core, symbol_pickaxe, corpus:production-kernel-provenance
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.runner.deep_gemm, L1.ep.layer, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+18/-21); python/sglang/srt/layers/moe/ep_moe/kernels.py (+31/-452); python/sglang/srt/layers/moe/ep_moe/layer.py (+8/-286); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-2); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+304/-0); python/sglang/srt/layers/moe/moe_runner/runner.py (+3/-0); python/sglang/srt/layers/moe/utils.py (+7/-1); benchmark/kernels/fbgemm/README.md (+0/-29); benchmark/kernels/fbgemm/benchmark_fbgemm_grouped_gemm.py (+0/-516); docs/advanced_features/server_arguments.md (+1/-1); (+9 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-edefab0c64  (L1, 2025-10-08, sha edefab0c6498, PR #10937)
TITLE: [2/2] Support MHA prefill with FlashAttention 4. (#10937)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/pyproject_other.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/attention_registry.py (+0/-3); python/sglang/srt/layers/attention/flashattention_backend.py (+0/-1); python/sglang/srt/model_executor/model_runner.py (+3/-9); python/sglang/srt/server_args.py (+28/-7)
LABELS: high priority, run-ci
BODY: (co-author @hyhieu) ⏎  ⏎ Updates: the kernel change has been checked in separately in: https://github.com/sgl-project/sglang/pull/10940  ⏎  ⏎ ## Motivation ⏎  ⏎ Add support for FA4 MHA prefill, changes mostly based on: https://github.com/sgl-project/sglang/pull/9428  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ ``` ⏎ lm_eval \ ⏎   --model local-chat-completions \ ⏎   --model_args model=gpt-oss,base_url=http://127.0.0.1:30010/v1/chat/completions,num_concu …[truncated]

### L1-a3c2ea4451  (L1, 2025-10-08, sha a3c2ea445194, PR #11327)
TITLE: fix: fix revision for sgl-flash-attn in sgl-kernel (#11327)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ https://github.com/sgl-project/sgl-flash-attn/commit/ec36bf54b90fcaa36a7358317a8b44df8ce739e9 ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-44cb060785  (L1, 2025-10-09, sha 44cb060785b8, PR #11364)
TITLE: chore: upgrade flashinfer 0.4.0 (#11364)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/pyproject_other.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-1); scripts/ci/ci_install_dependency.sh (+2/-0)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9b8ebb2798  (L1, 2025-10-09, sha 9b8ebb2798e2, PR #11285)
TITLE: move more files under srt/utils (#11285)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer, L1.routing.router_py
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-1); python/sglang/srt/layers/moe/router.py (+51/-15); .github/workflows/pr-test.yml (+2/-2); python/sglang/srt/disaggregation/decode.py (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/entrypoints/grpc_server.py (+1/-1); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+4/-1); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-3); python/sglang/srt/lora/lora_registry.py (+1/-1); python/sglang/srt/managers/data_parallel_controller.py (+1/-1); (+18 more)
LABELS: run-ci
BODY: 

### L1-f19613e6c3  (L1, 2025-10-10, sha f19613e6c362, PR #10734)
TITLE: Dedicated toml files for CPU/XPU (#10734)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.xeon (+8/-5); .github/workflows/pr-test-xeon.yml (+4/-1); docs/platforms/cpu_server.md (+4/-3); python/pyproject_cpu.toml (+123/-0); python/pyproject_xpu.toml (+123/-0)
LABELS: ready-to-merge, intel, cpu, run-ci
BODY: ## Motivation ⏎  ⏎ Split the `.toml` files for `sglang` package, to better facilitate the planned per-device `sgl-whl` building. ⏎ Originated from #10162 . ⏎  ⏎ ## Modifications ⏎  ⏎ Created dedicated `.toml` files for CPU and Intel XPU. Dockerfile and document of CPU are updated correspondingly. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ N/A ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ N/A ⏎  ⏎ ## Checklist

### L1-8fdcd98efe  (L1, 2025-10-11, sha 8fdcd98efefc, PR #11019)
TITLE: [7/n] decouple quantization impl from vllm dependency - gguf kernel (#11019)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.reduce.moe_sum
FILES: sgl-kernel/csrc/moe/moe_sum.cu (+66/-0); sgl-kernel/csrc/quantization/gguf/moe.cuh (+1379/-0); sgl-kernel/csrc/quantization/gguf/moe_vec.cuh (+413/-0); sgl-kernel/python/sgl_kernel/moe.py (+10/-0); sgl-kernel/CMakeLists.txt (+3/-0); sgl-kernel/csrc/common_extension.cc (+37/-0); sgl-kernel/csrc/quantization/gguf/dequantize.cuh (+583/-0); sgl-kernel/csrc/quantization/gguf/ggml-common.h (+1029/-0); sgl-kernel/csrc/quantization/gguf/gguf_kernel.cu (+836/-0); sgl-kernel/csrc/quantization/gguf/mmq.cuh (+881/-0); (+9 more)
LABELS: ready-to-merge, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Support sglang native gguf kernel. ⏎ Will change srt in next PR ⏎  ⏎ ### Kernel List ⏎ - Done: ⏎ 1. ggml_mul_mat_vec_a8  ⏎ 2. ggml_mul_mat_a8  ⏎ 3. ggml_dequantize  ⏎ 4. ggml_moe_get_block_size  ⏎ 5. ggml_moe_a8  ⏎ 6. moe_sum  ⏎ 7. ggml_moe_a8_vec  ⏎ 8. moe_sum (https://github.com/sgl-project/sglang/blob/2a9d995c09dd16c06e457664028c732d7f3bc16a/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L598) ⏎  ⏎ - TODO: ⏎  ⏎ Adapted from: h …[truncated]

### L1-1bdd010291  (L1, 2025-10-12, sha 1bdd010291e4, PR #11520)
TITLE: Revert "Deprecate `global_server_args_dict`" (#11520)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-0); python/sglang/global_config.py (+3/-0); python/sglang/srt/distributed/device_communicators/pynccl_allocator.py (+2/-2); python/sglang/srt/eplb/expert_location_dispatch.py (+2/-2); python/sglang/srt/eplb/expert_location_updater.py (+2/-2); python/sglang/srt/layers/attention/double_sparsity_backend.py (+2/-2); python/sglang/srt/layers/attention/flashattention_backend.py (+2/-2); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+5/-5); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+2/-2); python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py (+2/-2); (+44 more)
LABELS: run-ci
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 11331 reason=unstated
BODY: Reverts sgl-project/sglang#11331

### L1-a2b3d9b90b  (L1, 2025-10-12, sha a2b3d9b90b92, PR #11512)
TITLE: Update DeepSeek-R1-FP4 default config on blackwell (#11512)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+26/-1)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-99a0704a36  (L1, 2025-10-12, sha 99a0704a36a5, PR #11465)
TITLE: bailingMoE: Fix Key error of deepep_mode (#11465)
SOURCES: subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/bailing_moe.py (+2/-2)
BODY: this patch fix erros below: ⏎  ⏎ ERROR 6524 [ DP0 TP2 EP2 scheduler.py:2864]     deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]], ⏎ ERROR 6524 [ DP0 TP2 EP2 scheduler.py:2864] KeyError: 'deepep_mode ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎ ## Summary by CodeRabbit ⏎  ⏎ - Refactor ⏎   - Streamlined dispatcher mode selection using a dedicated configuration …[truncated]

### L1-1083e7e3df  (L1, 2025-10-13, sha 1083e7e3df1e, PR #11331)
TITLE: Deprecate `global_server_args_dict` (#11331)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-2); python/sglang/global_config.py (+0/-3); python/sglang/srt/distributed/device_communicators/pynccl_allocator.py (+2/-2); python/sglang/srt/eplb/expert_location_dispatch.py (+2/-2); python/sglang/srt/eplb/expert_location_updater.py (+2/-2); python/sglang/srt/layers/attention/double_sparsity_backend.py (+2/-2); python/sglang/srt/layers/attention/flashattention_backend.py (+2/-2); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+5/-5); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+2/-2); python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py (+2/-2); (+44 more)
LABELS: run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 11520 (confirmed_revert, reason=unstated)
BODY: Deprecate the `global_server_args_dict`, just use `global_server_args: ServerArgs`. ⏎  ⏎ - Only the scheduler process needs it ⏎ - Set the global value in `ModelRunner` ⏎ - Only access this global value by `get_global_server_args()` to avoid reading before setting the arguments.

### L1-8e51049f56  (L1, 2025-10-13, sha 8e51049f5614, PR #11538)
TITLE: [CI Monitor] Ci monitor only deal with main branch in default (#11538)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: scripts/ci_monitor/ci_analyzer.py (+13/-3); scripts/ci_monitor/ci_analyzer_perf.py (+35/-11)
LABELS: run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 11846 (confirmed_revert, reason=unstated)
BODY: <img width="755" height="243" alt="图片" src="https://github.com/user-attachments/assets/0c298290-5daf-433e-9a35-4459f75d346a" /> ⏎  ⏎  ⏎ Use `build-test (all)` as example: ⏎  ⏎ main: ⏎  ⏎ Last Success: Run ⏎  ⏎ <img width="575" height="282" alt="图片" src="https://github.com/user-attachments/assets/3534f035-47da-43f9-ad43-2a7b8d291b73" /> ⏎  ⏎ It's a fork branch ⏎  ⏎ pr: ⏎  ⏎ Last Success: Run ⏎  ⏎ <img width="525" height="253" alt="图片" src="https://github.com/user- …[truncated]

### L1-cb8ed2c09a  (L1, 2025-10-13, sha cb8ed2c09a14, PR #11535)
TITLE: Make DeepEP combine recv do not overlap (#11535)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+3/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-516738b096  (L1, 2025-10-13, sha 516738b0963d, PR #11528)
TITLE: Depreate `global_server_args_dict` (#11528)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-2); python/sglang/global_config.py (+0/-3); python/sglang/srt/distributed/device_communicators/pynccl_allocator.py (+2/-2); python/sglang/srt/eplb/expert_location_dispatch.py (+2/-2); python/sglang/srt/eplb/expert_location_updater.py (+2/-2); python/sglang/srt/layers/attention/double_sparsity_backend.py (+2/-2); python/sglang/srt/layers/attention/flashattention_backend.py (+2/-2); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+5/-5); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+2/-2); python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py (+2/-2); (+44 more)
LABELS: run-ci
BODY: 

### L1-c7867b6702  (L1, 2025-10-13, sha c7867b67027a, PR #11201)
TITLE: [Fix] Add per_channel_quant parameter to MoE config functions (#11201)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py (+11/-3)
LABELS: run-ci
ISSUES: #11199 [Bug] tuning_fused_moe_triton.py failed with crash
BODY: ## Motivation ⏎  ⏎ Fixes #11199 (issue caused by #10915) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-eb8cac6fe2  (L1, 2025-10-14, sha eb8cac6fe279, PR #11453)
TITLE: [router] add py binding and readme for openai router and history backend (#11453)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+77/-3); sgl-router/README.md (+98/-0); sgl-router/py_src/sglang_router/router_args.py (+80/-0); sgl-router/src/config/validation.rs (+62/-2); sgl-router/src/lib.rs (+142/-2); sgl-router/src/routers/openai/conversations.rs (+3/-3); sgl-router/src/routers/openai/router.rs (+6/-7); sgl-router/src/server.rs (+20/-8)
LABELS: router, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ - add py binding and readme for openai router and history backend ⏎ - improve logs ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ oracle history backend- wallet ⏎ ``` ⏎ python -m sglang_router.launch_router     --backend openai  --history-backend oracle  --worker-urls https://api.openai.com --prometheus-port 30002 --oracle-wallet-path /home/ubuntu/wallet/Wallet_sglroutertestatp --oracle-tns-alias sglroutertestatp_low ⏎ 2025-10-11 05:5 …[truncated]

### L1-e4358a4585  (L1, 2025-10-14, sha e4358a458504, PR #11587)
TITLE: Add fused_moe_triton config: triton_3_4_0/E=256,N=256,device_name=NVIDIA_B200.json (#11587)
SOURCES: path_config_only, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=256,N=256,device_name=NVIDIA_B200.json (+146/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎  ⏎ Before: ⏎ <img width="743" height="242" alt="image" src="https://github.com/user-attachments/assets/63075495-f6d3-4c7f-984f-77fe48c50269" /> ⏎  ⏎ After: ⏎ <img width="752" height="290" alt="image" src="https://github.com/user-attachments/assets/dc77b70b-ae54-4dcb-886f-c1d5e138970b" /> ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a40229f6f8  (L1, 2025-10-14, sha a40229f6f828, PR #10423)
TITLE: [1/N] Introduce Mooncake Backend and Mooncake EP to Support Elastic EP (#10423)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-1); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+8/-0); python/sglang/srt/layers/moe/token_dispatcher/mooncake.py (+394/-0); python/sglang/srt/layers/moe/utils.py (+4/-0); scripts/ci/ci_install_deepep.sh (+4/-0); docs/advanced_features/server_arguments.md (+3/-1); python/sglang/srt/distributed/parallel_state.py (+20/-5); python/sglang/srt/model_executor/model_runner.py (+29/-12); python/sglang/srt/models/deepseek_v2.py (+6/-3); python/sglang/srt/server_args.py (+27/-4); (+3 more)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ **Why do we need a new backend?** ⏎  ⏎ The Mooncake Backend is a collective communication backend that provides fault tolerance while remaining seamlessly compatible with PyTorch. To the best of our knowledge, this is the first backend that supports fault tolerance for both: ⏎ - `torch.distributed` primitives, and ⏎ - EP all-to-all primitives. ⏎  ⏎ This lays the foundation for fault-tolerant distributed MoE inference (#8961). ⏎  ⏎ ## …[truncated]

### L1-825432fce6  (L1, 2025-10-14, sha 825432fce673, PR #8247)
TITLE: [1/N]Support  DeepSeek-R1 w4a8 normal deepep (#8247)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+196/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+21/-2); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+10/-4); python/sglang/srt/layers/moe/utils.py (+4/-0); python/sglang/srt/layers/quantization/w4afp8.py (+47/-1); python/sglang/srt/server_args.py (+1/-0); test/srt/quant/test_w4a8_deepseek_v3.py (+55/-0)
LABELS: high priority, ready-to-merge, run-ci
BODY: ## Motivation ⏎  ⏎ Support deepep normal mode for DeepSeek w4a8 model with @yangsijia-serena @rainj-me  ⏎  ⏎ ## Modifications ⏎  ⏎ add forward_cutlass_w4a8 for deepep normal mode ⏎  ⏎  ⏎ ## Command ⏎  ⏎ `SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m sglang.launch_server --model-path  /data/models/DeepSeek-R1-W4AFP8   --tp 8 --trust-remote-code --host 0.0.0.0 --port 8000  --mem-fraction-static 0.85  --max-running-requests 1024 --context-length 4096 --disable-cuda-gra …[truncated]

### L1-4c03dbaaef  (L1, 2025-10-15, sha 4c03dbaaef70, PR #9493)
TITLE: [CI][XPU]enable sglang CI on Intel XPU (#9493)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.xpu (+78/-0); .github/workflows/pr-test-xpu.yml (+99/-0); python/sglang/srt/layers/rotary_embedding.py (+16/-2); python/sglang/test/test_utils.py (+5/-0); test/srt/run_suite.py (+8/-0); test/srt/xpu/test_intel_xpu_backend.py (+60/-0)
LABELS: intel, xpu, run-ci
BODY: ## Motivation ⏎  ⏎ This PR aims to enable the CI pipeline for Sglang targeting the XPU architecture. This approach enhances our ability to promptly gate and block potential issues before they affect the broader codebase. Key improvements and features introduced in this PR include: ⏎  ⏎ - Add XPU Test Coverage ⏎  ⏎ - Expand test suite to cover XPU-specific code paths ⏎  ⏎ - Trigger XPU CI by default ⏎  ⏎  ⏎ ## Summary by CodeRabbit ⏎  ⏎ - New Features ⏎   - Initial …[truncated]

### L1-cd7e1bd591  (L1, 2025-10-15, sha cd7e1bd59117, PR #11686)
TITLE: Sync code and test CI; rename some env vars (#11686)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+4/-2); docs/references/environment_variables.md (+1/-1); python/sglang/srt/distributed/device_communicators/custom_all_reduce.py (+3/-3); python/sglang/srt/environ.py (+3/-3); python/sglang/srt/layers/logits_processor.py (+1/-1); python/sglang/srt/layers/sampler.py (+3/-3); python/sglang/srt/model_executor/model_runner.py (+3/-4); python/sglang/srt/model_loader/weight_utils.py (+2/-2); python/sglang/srt/server_args.py (+9/-7); python/sglang/srt/speculative/eagle_worker.py (+2/-2); (+7 more)
LABELS: run-ci
BODY: - Rename `RETURN_ORIGINAL_LOGPROB` to `SGLANG_RETURN_ORIGINAL_LOGPROB`. cc @narutolhy ⏎ - Rename `SGLANG_AMD_CI` -> `SGLANG_IS_IN_CI_AMD`

### L1-476c67d7fc  (L1, 2025-10-15, sha 476c67d7fcfe, PR #11692)
TITLE: Fix missing a2a backend init of GLM4.5 MoE Block (#11692)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/glm4_moe.py (+4/-2)
LABELS: run-ci
BODY: ## Motivation ⏎ Fix https://github.com/sgl-project/sglang/actions/runs/18546407737/job/52865228007#step:4:8006 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9b0f725b1d  (L1, 2025-10-17, sha 9b0f725b1dc6, PR #11730)
TITLE: add tuned fuse moe kernel for qwen3 235b fp8 on h200 (#11730)
SOURCES: path_config_only, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8.json (+146/-0)
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎ Add tuned fused_moe_triton kernel for Qwen/Qwen3-235B-A22B-Instruct-2507 fp8_w8a8 on h200. ⏎  ⏎ python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \ ⏎     --model Qwen/Qwen3-235B-A22B-Instruct-2507 \ ⏎     --tp-size 8 \ ⏎     --dtype fp8_w8a8 \ ⏎     --tune ⏎  ⏎ see ~8% improvement in inter token latency with this kernel ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ ``` ⏎ python ben …[truncated]

### L1-627974405d  (L1, 2025-10-17, sha 627974405dd3, PR #11685)
TITLE: [Lint] Add `python/sglang` to ruff F401 checks and remove unused imports in files (#11685)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.triton, L1.runner.flashinfer_cutedsl, L1.ep.layer, L1.ep.deepep_dispatcher, L1.ep.other_dispatchers, L1.cutlass.adapters
FILES: .pre-commit-config.yaml (+3/-3); python/sglang/srt/_custom_ops.py (+1/-1); python/sglang/srt/compilation/cuda_piecewise_backend.py (+0/-1); python/sglang/srt/configs/deepseekvl2.py (+0/-1); python/sglang/srt/configs/dots_vlm.py (+2/-7); python/sglang/srt/configs/falcon_h1.py (+1/-6); python/sglang/srt/configs/qwen3_next.py (+0/-1); python/sglang/srt/connector/remote_instance.py (+1/-1); python/sglang/srt/disaggregation/ascend/transfer_engine.py (+1/-1); python/sglang/srt/disaggregation/decode.py (+2/-6); (+141 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ This PR adds F401 to ruff checks and removes unused imports accross `python/sglang`, excluding `__init__.py`. ⏎  ⏎ ## Modifications ⏎  ⏎ - Added `F821` to ruff checks - this catches undefined names/missing imports ⏎ - Added `python/sglang` to ruff checks, excluded all __init__.py ⏎ - Tried to fix all unused imports ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-5b214b50b6  (L1, 2025-10-17, sha 5b214b50b65c, PR #11784)
TITLE: [Refactor] move `deep_gemm_wrapper` out of `quantization` (#11784)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.runner.deep_gemm, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-1); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+1/-1); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+1/-1); benchmark/kernels/quantization/bench_fp4_quant.py (+1/-1); python/sglang/bench_one_batch.py (+0/-1); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+1/-1); python/sglang/srt/layers/deep_gemm_wrapper/__init__.py (+0/-0); python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py (+1/-3); python/sglang/srt/layers/deep_gemm_wrapper/configurer.py (+0/-0); python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py (+2/-2); (+9 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ `deep_gemm_wrapper` is not quantization-specific. Many other files such as `deepseek_v2.py` and `moe_runner` also require it. This PR moves it out of the `quantization`  folder to avoid circular import. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-505329cab0  (L1, 2025-10-18, sha 505329cab00e, PR #11611)
TITLE: Support shared experts overlap in cutlass moe (#11611)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+13/-0); python/sglang/srt/models/deepseek_v2.py (+26/-4)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8af8491298  (L1, 2025-10-18, sha 8af8491298c3, PR #11613)
TITLE: Support casting bf16 NextN moe to fp8 (#11613)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_nextn.py (+17/-1); python/sglang/srt/models/deepseek_v2.py (+76/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-69fe3c9726  (L1, 2025-10-18, sha 69fe3c97268a, PR #11666)
TITLE: Manually flip deepep_mode for cuda_graph (#11666)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+9/-0); python/sglang/srt/model_executor/cuda_graph_runner.py (+28/-0); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+6/-0); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+6/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ When serving model with `--deepep-mode auto` and with cuda graph on, the current implementation won't flip the deepep mode from normal to low_latency and will trigger cuda illegal memory access in deepep or the following deepgemm kernel. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ This PR is trying to add a manual flip by storing the deepep mode during capture and set it during replay. ⏎  ⏎ Thank you for your time on reviewing this PR :) ⏎  ⏎ ## Accu …[truncated]

### L1-33e9bbec35  (L1, 2025-10-18, sha 33e9bbec350b, PR #11614)
TITLE: Make single-batch overlap compatible with offloading (#11614)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+13/-10); python/sglang/srt/models/deepseek_v2.py (+18/-8); python/sglang/srt/single_batch_overlap.py (+3/-5)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fda0cb2a30  (L1, 2025-10-18, sha fda0cb2a3043, PR #11773)
TITLE: Fix Dockerfile not installing correct version of DeepEP for arm build (#11773)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/release-docker-dev.yml (+3/-1); .github/workflows/release-docker.yml (+4/-0); docker/Dockerfile (+3/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ All the docker release github flows have been updated to use `BUILD_TYPE=all` for both x86/arm builds. ⏎ However, the Dockerfile was not updated to respect this change. ⏎ Hence, it's not installing the correct DeepEP version for arm builds. ⏎  ⏎ ## Modifications ⏎  ⏎ After discussion with @ishandhanani , the suggestion is to adopt the changes in https://github.com/sgl-project/sglang/pull/11517, where we introduce a new build arg GRACE_ …[truncated]

### L1-ce399e154c  (L1, 2025-10-19, sha ce399e154cb8, PR #11804)
TITLE: Make single-batch overlap compatible with NextN (#11804)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-0); python/sglang/srt/models/deepseek_v2.py (+2/-0); python/sglang/srt/single_batch_overlap.py (+5/-4)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-3b80232d06  (L1, 2025-10-19, sha 3b80232d0669, PR #11815)
TITLE: [DeepseekV32] Add fast_topk_transform_ragged_fused kernel (#11815)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-kernel/python/sgl_kernel/top_k.py (+24/-1); sgl-kernel/csrc/common_extension.cc (+4/-0); sgl-kernel/csrc/elementwise/topk.cu (+81/-8); sgl-kernel/include/sgl_kernel_ops.h (+11/-6); sgl-kernel/python/sgl_kernel/__init__.py (+6/-1); sgl-kernel/tests/test_topk.py (+75/-4)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Add a fused kernel for `fast_topk_transform_ragged_fused`. The difference between this kernel and  `fast_topk_transform_fused` is that `fast_topk_transform_fused` outputs indices into the paged kvcache and `fast_topk_transform_ragged_fused` outputs indices into the ragged kv that's the input to the flashmla_prefill kernel. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Tested with https://github.com/sgl-project/sglang/pull/11655. ⏎ Before ⏎ Repeat: 4, me …[truncated]

### L1-27a223aba4  (L1, 2025-10-19, sha 27a223aba48b, PR #11508)
TITLE: Improve Kernel Build Time (#11508)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: sgl-kernel/build.sh (+156/-4); sgl-kernel/kernel-runner-setup.sh (+150/-0)
LABELS: high priority, run-ci
BODY: ## Modifications ⏎  ⏎  ⏎  ⏎ - Use CCACHE to store all build results on runner local path ⏎ - Add profiles for both CMake and Ninja ⏎ - Add a setup script for new CPU runners that would just execute the build actions  ⏎ - Add time spent on each build action ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ Before (without cache): ⏎ [116/394 419.382s] Building CUDA object CMakeFiles/common_ops_sm90_build.dir/csrc/moe/prepare_moe_input.cu.o ⏎ [117/394 421.032s] Buildi …[truncated]

### L1-a8ba32798e  (L1, 2025-10-20, sha a8ba32798e1e, PR #11831)
TITLE: Fix triton_kernels import error on some hardwares (#11831)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/unquant.py (+1/-4)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9e3be1fa2a  (L1, 2025-10-20, sha 9e3be1fa2a27, PR #11810)
TITLE: Tiny bump DeepEP version in ARM blackwell (#11810)
SOURCES: subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ ~~not tested yet~~ ⏎ tested ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-bfc3b3f786  (L1, 2025-10-20, sha bfc3b3f78682, PR #11847)
TITLE: [9/N] MoE Refactor: cleanup dispatcher interfaces (#11847)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.ep.layer, L1.ep.deepep_dispatcher, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+3/-1); python/sglang/srt/layers/moe/ep_moe/layer.py (+69/-99); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+44/-35); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+2/-0); python/sglang/srt/layers/moe/token_dispatcher/base.py (+1/-1); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+86/-91); python/sglang/srt/layers/moe/token_dispatcher/mooncake.py (+37/-39); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+46/-0); python/sglang/srt/layers/moe/topk.py (+3/-2); python/sglang/srt/layers/dp_attention.py (+17/-0); (+14 more)
LABELS: run-ci
BODY: - Unified initialization for `StandardDispatcher` and `DeepEPDispatcher`. ⏎ - Added `get_is_extend_in_batch` so that there is no need to pass `forward_batch` to dispatcher. ⏎ - Added `set_quant_config` to each dispatch. Model files do not need to handle `input_global_scale`. ⏎ - Refactored all dispatchers to use `TopKOutput` objects instead of separate `topk_idx`/`topk_weights` parameters. ⏎ - Cleaned dead codes. ⏎ - Misc refactor for better consisten …[truncated]

### L1-7e6191c098  (L1, 2025-10-21, sha 7e6191c098e9, PR #11487)
TITLE: init support for KTransformers Heterogeneous Computing (#11487)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+25/-3); python/sglang/srt/environ.py (+8/-0); python/sglang/srt/layers/quantization/compressed_tensors/__init__.py (+7/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+10/-1); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+408/-8); python/sglang/srt/model_executor/cuda_graph_runner.py (+9/-0); python/sglang/srt/models/deepseek_v2.py (+21/-5); python/sglang/srt/models/qwen3_next.py (+2/-0); python/sglang/srt/server_args.py (+57/-0)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ KTransformers Integration to Support CPU/GPU Hybrid Inference for MoE Models ⏎  ⏎ Only support CompressedTensor format. Will support other formats later. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Hack num_gpu_experts to reduce VRAM usage and GPU calculation. Will refactor to use  experts_map instead of num_gpu_experts. ⏎  ⏎ Add dependency of kt_kernel (https://github.com/kvcache-ai/ktransformers) ⏎  ⏎ ## test command ⏎  ⏎ ``` ⏎ "args": [ ⏎         "- …[truncated]

### L1-0917c5da8c  (L1, 2025-10-21, sha 0917c5da8cf4, PR #11807)
TITLE: Support mixing cutedsl and deepgemm backend (#11807)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+5/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b113c72e7a  (L1, 2025-10-21, sha b113c72e7add, PR #10656)
TITLE: Init attention backend for Intel XPU (#10656)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.xpu (+2/-7); Makefile (+3/-1); docs/advanced_features/attention_backend.md (+8/-0); docs/index.rst (+1/-0); docs/platforms/xpu.md (+92/-0); python/pyproject_xpu.toml (+5/-3); python/sglang/bench_one_batch.py (+1/-1); python/sglang/srt/distributed/parallel_state.py (+4/-2); python/sglang/srt/layers/attention/attention_registry.py (+7/-0); python/sglang/srt/layers/attention/fla/layernorm_gated.py (+3/-1); (+8 more)
LABELS: intel, xpu, run-ci
BODY: ## Motivation ⏎  ⏎ Add an attention backend for Intel XPU based on [sgl-kernel-xpu](https://github.com/sgl-project/sgl-kernel-xpu) ⏎ Depend on https://github.com/sgl-project/sglang/pull/10248 ⏎  ⏎ ## Modifications ⏎  ⏎ and clean some chores ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-ebff4ee648  (L1, 2025-10-21, sha ebff4ee64836, PR #11844)
TITLE: Update sgl-kernel and remove fast hadamard depedency (#11844)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-9); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+1/-1); scripts/ci/ci_install_dependency.sh (+0/-8)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Follow up of  #11663 and #11733 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9792b9d7e3  (L1, 2025-10-21, sha 9792b9d7e368, PR #11933)
TITLE: chore: upgrade flashinfer 0.4.1 (#11933)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-23afdfd1c2  (L1, 2025-10-21, sha 23afdfd1c2b1, PR #11717)
TITLE: [sgl-kernel] support flashmla libtorch (#11717)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+24/-15); sgl-kernel/cmake/flashmla.cmake (+60/-0); sgl-kernel/csrc/flashmla_extension.cc (+46/-0); sgl-kernel/include/sgl_kernel_ops.h (+45/-0); sgl-kernel/python/sgl_kernel/flash_mla.py (+126/-0); sgl-kernel/tests/test_flashmla.py (+518/-0)
LABELS: high priority, ready-for-review, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ support flashmla libtorch ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e028af6998  (L1, 2025-10-22, sha e028af6998af, PR #11908)
TITLE: Fix mooncake dispatcher (#11908)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-1); python/sglang/srt/layers/moe/token_dispatcher/mooncake.py (+7/-1)
LABELS: ready-to-merge, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Keep the Mooncake dispatcher's implementation align with the codebase changes. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ 1. Both `a2a_backend.is_deepep()` and `a2a_backend.is_mooncake()` go to the branch `MaybeTboDeepEPDispatcher` ⏎ 2. `MooncakeDispatchOutput` should now accept `hidden_states` and `hidden_states_scale` ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ Unittests should remain okay ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-904655c5fd  (L1, 2025-10-22, sha 904655c5fd74, PR #10606)
TITLE: [2/N] Added the core structure of elastic EP and the eplb algorithm with faulty rank (#10606)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/mooncake.py (+2/-14); python/sglang/srt/elastic_ep/elastic_ep.py (+74/-0); python/sglang/srt/eplb/eplb_algorithms/__init__.py (+18/-1); python/sglang/srt/eplb/eplb_algorithms/elasticity_aware.py (+87/-0); python/sglang/srt/model_executor/model_runner.py (+37/-9); python/sglang/srt/server_args.py (+12/-0); test/srt/ep/test_mooncake_ep_small.py (+75/-218)
LABELS: ready-to-merge, run-ci
BODY: ## Motivation ⏎  ⏎ 1. Integrating Mooncake's fault-awareness, we need to adjust the eplb algorithm and model loading logic to enable the forward pass to bypass faulty ranks. ⏎  ⏎ base on #10423 ⏎ check our next pr #11657 and full draft #8961 (update on 10.16) to test the effect of fault redundancy ⏎  ⏎ The ut part is modified to facilitate testing on machines with different ibdev names ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ 1. Adding the core structure of the elastic …[truncated]

### L1-fdcb1d13c5  (L1, 2025-10-22, sha fdcb1d13c5ae, PR #11977)
TITLE: [BUG] AttributeError: 'DeepEPMoE' object has no attribute 'use_w4a… (#11977)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-0)
BODY: ## Motivation ⏎ AttributeError: 'DeepEPMoE' object has no attribute 'use_w4afp8' ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-9d61205dac  (L1, 2025-10-22, sha 9d61205dac19, PR #11922)
TITLE: [lint] improve ruff check (#11922)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+20/-19); .pre-commit-config.yaml (+3/-1); benchmark/kernels/minmax-text-01-lightning_attention/benchmark_lightning_attention_decode.py (+1/-0); benchmark/kernels/minmax-text-01-lightning_attention/benchmark_lightning_attention_prefill.py (+4/-0); python/sglang/bench_one_batch_server.py (+3/-0); python/sglang/srt/disaggregation/common/conn.py (+2/-2); python/sglang/srt/disaggregation/mooncake/conn.py (+1/-1); python/sglang/srt/entrypoints/openai/serving_responses.py (+2/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+1/-1); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+4/-1); (+9 more)
LABELS: run-ci
BODY: Prev PR #11685 added `python/sglang` to the Ruff lint scope, but the pre-commit args were written as: ⏎  ⏎ ``` ⏎ args: [--select=F401,F821, --fixable=F401] ⏎ ``` ⏎  ⏎ These arguments are interpreted as `--select=F401 F821 --fixable=F401`, which leads to  ⏎  ⏎ ``` ⏎ warning: Failed to lint F821: No such file or directory (os error 2) ⏎ ``` ⏎  ⏎ This PR changes the arguments to ⏎ ``` ⏎ args: ⏎   - --select=F401,F821 ⏎   - --fix ⏎ ``` ⏎  ⏎ - Remove `--fixable` since ` …[truncated]

### L1-eec9e471ca  (L1, 2025-10-22, sha eec9e471cad4, PR #11563)
TITLE: [NVIDIA] Update to leverage flashinfer trtllm FP4 MOE throughput kernel  (#11563)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-14); python/sglang/srt/layers/quantization/mxfp4.py (+1/-26); python/sglang/srt/layers/quantization/modelopt_quant.py (+3/-3); scripts/ci/ci_install_dependency.sh (+2/-2)
LABELS: high priority, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎ Use latest flashinfer trtllm FP4 MOE throughput kernel  ⏎  ⏎ ## Modifications ⏎ - Update flashinfer version to 0.4.1 to leverage latest flashinfer trtllm throughput moe kernel. ⏎ - Change use of falshinfer trtllm_fp4_block_scale_moe. Remove the tile_tokens_dim, pass None so flashinfer can calculate itself ⏎  ⏎ ## Accuracy Tests ⏎ `lm_eval --model local-completions --tasks gsm8k --model_args model=openai/gpt-oss-120b,base_url=http://0.0.0. …[truncated]

### L1-1d097aac87  (L1, 2025-10-22, sha 1d097aac8742, PR #11967)
TITLE: [Fix] Remove unused import from triton_kernels_moe.py (#11967)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+1/-26)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ It looks like unit-test-backend-1-gpu(3) was broken because this PR (https://github.com/sgl-project/sglang/pull/11922) imported a non-existent MicroscalingCtx. This symbol doesn’t exist in the Triton kernels. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8612811d85  (L1, 2025-10-22, sha 8612811d854e, PR #11990)
TITLE: Bump grace blackwell DeepEP version (#11990)
SOURCES: subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ This PR updates SGLang’s grace blackwell DeepEP version use the new `gb200_blog_part_2` branch from [fzyzcjy/DeepEP](https://github.com/fzyzcjy/DeepEP/tree/gb200_blog_part_2) ⏎  ⏎ As of 10/22 - this branch serves as the canonical reference for FP4 and FP8 performance on Grace-Blackwell and is the same codebase used in DeepSeek’s Blog Post 2 benchmarks. ⏎  ⏎ ## Background and Context ⏎  ⏎ Previously, two divergent DeepEP branches were floa …[truncated]

### L1-81fd2b0ee0  (L1, 2025-10-22, sha 81fd2b0ee0df, PR #11965)
TITLE: fix(deepep): resolve benchmark failure on 4×IB-card setup by aligning tuning config with DeepEP commit bdd119f8 (#11965)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: benchmark/kernels/deepep/tuning_deepep.py (+2/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ When running benchmark/kernels/deepep/tuning_deepep.py on a system equipped with 4 InfiniBand (IB) cards, the tuning script could fail due to unstable or invalid parameter combinations for nvl_chunk_size and rdma_chunk_size. ⏎ This issue mirrors the behavior reported in [DeepEP Issue #321](https://github.com/deepseek-ai/DeepEP/issues/321), which resulted in errors such as: ⏎  ⏎ ``` ⏎ assertion num_max_rdma_chunked_send_tokens >= num_ …[truncated]

### L1-d7e834d6ba  (L1, 2025-10-23, sha d7e834d6baee, PR #10750)
TITLE: [6/n]decouple quantization implementation from vLLM dependency (#10750)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L1.runner.marlin
FILES: python/sglang/srt/layers/quantization/__init__.py (+0/-52); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+8/-53); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+23/-65); python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py (+3/-0); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py (+4/-22); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py (+339/-0); python/sglang/srt/layers/quantization/marlin_utils.py (+12/-0)
LABELS: ready-to-merge, run-ci
BODY: ## Motivation ⏎ **Remove vLLM-dependency-test** ⏎  ⏎ Remove the compressed_tensors dependency from vLLM ⏎  ⏎ Now supported quant method: ⏎ - w8a8-fp8 ⏎ - w8a16fp8 ⏎ - wNa16 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-39c237f02c  (L1, 2025-10-23, sha 39c237f02cb3, PR #10158)
TITLE: Add AWQ quantization support for NPU.  (#10158)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/linear.py (+1/-0); python/sglang/srt/layers/quantization/awq.py (+176/-4); python/sglang/srt/layers/quantization/awq_triton.py (+29/-0); python/sglang/srt/layers/quantization/w8a8_int8.py (+28/-6); python/sglang/srt/model_loader/loader.py (+2/-0); python/sglang/srt/models/deepseek_v2.py (+5/-1); python/sglang/srt/utils/common.py (+2/-0)
LABELS: npu, run-ci
BODY: ## Motivation ⏎ This PR follows #9104 and Roadmap of NPU support #8004.  ⏎  ⏎  ⏎ ## Modifications ⏎ We mainly modified python/sglang/srt/layers/quantization/awq.py, add `AWQLinearAscendMethod` and `AWQMoEAscendMethod` to support AWQ.   ⏎  ⏎  ⏎ ## Accuracy and Benchmark Tests ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path /data/models/DeepSeek-AWQ --tp 8 --device npu --attention-backend ascend --port 8001 --disable-radix-cache --quantization awq ⏎ py …[truncated]

### L1-14a4d80e57  (L1, 2025-10-23, sha 14a4d80e575f, PR #11964)
TITLE: [8/n] decouple quantization impl from vllm dependency - gguf srt (#11964)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/__init__.py (+3/-5); python/sglang/srt/layers/quantization/gguf.py (+566/-0); python/sglang/srt/model_executor/model_runner.py (+0/-3); python/sglang/srt/utils/common.py (+1/-27); test/srt/run_suite.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Clean vllm dependency, as titled. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ```bash ⏎ python3 /sgl-workspace/sglang/test/srt/test_gguf.py ⏎ [Test Method] test_models ⏎ /sgl-workspace/sglang/python/sglang/srt/sampling/sampling_params.py:17: DeprecationWarning: module 'sre_parse' is deprecated ⏎   import sre_parse ⏎ `torch_dtype` is deprecated! Use `dtype` instead! ⏎ `torch_dtype` is deprecated! Use `dtype` instead! ⏎ [Gloo] …[truncated]

### L1-62eff37ba1  (L1, 2025-10-23, sha 62eff37ba19a, PR #11795)
TITLE: Refactor Triton-kernel MoE runner integration  (#11795)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.runner.framework, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+7/-4); python/sglang/srt/layers/moe/moe_runner/runner.py (+3/-0); python/sglang/srt/layers/moe/moe_runner/triton_kernels.py (+194/-0); python/sglang/srt/layers/moe/token_dispatcher/base.py (+6/-0); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+1/-1); python/sglang/srt/layers/moe/topk.py (+4/-4); python/sglang/srt/layers/moe/utils.py (+3/-3); python/sglang/srt/layers/quantization/mxfp4.py (+32/-38); python/sglang/srt/layers/quantization/unquant.py (+31/-46); (+1 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Refactor Triton-kernel MoE runner integration into `triton_kernels.py` per https://github.com/sgl-project/sglang/issues/8715 ⏎  ⏎ ## Modifications ⏎  ⏎ [Plan.md](https://github.com/user-attachments/files/22993701/Triton-Kernel-Plan.md) ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ `test/srt/test_triton_fused_moe.py` passes on Nvidia B200. ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ Not relevant as no kernels were modified. ⏎  ⏎ ## Checklist

### L1-f80371ff8c  (L1, 2025-10-23, sha f80371ff8cb7, PR #11816)
TITLE: Use flashinfer_trtllm moe runner backend to gain around 10% perf on b200 fp8 dpsk (#11816)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-1); python/sglang/srt/layers/quantization/fp8.py (+65/-59); python/sglang/srt/server_args.py (+10/-5)
LABELS: ready-to-merge, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Do 3 things ⏎  ⏎ 1. If on SM100 + FP8 for dpsk, use flashinfer_trtllm as default moe runner backend, replacing triton (current default). the fp4 path is already using this. Note that shared experts fusion is disabled here. ⏎  ⏎ The performance improvements are described: ⏎  ⏎ <img width="1022" height="464" alt="image" src="https://github.com/user-attachments/assets/b1faeda1-c921-418a-a24f-99898665c983" /> ⏎  ⏎ https://sgl-fru7574.slack.c …[truncated]

### L1-4060ed37cb  (L1, 2025-10-24, sha 4060ed37cb67, PR #11800)
TITLE: Refactoring GLM-4.5 and GLM-4.5V related implementations (#11800)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/glm4_moe.py (+322/-354); python/sglang/srt/models/glm4_moe_nextn.py (+4/-14); python/sglang/srt/models/glm4v_moe.py (+29/-196); python/sglang/srt/multimodal/processors/glm4v.py (+1/-1)
LABELS: run-ci
BODY: To resolve the inheritance conflict with the DeepSeek-V2 model, the GLM-4.5 model implementation has been refactored, streamlining the code and inference logic, with relevant benchmark validation completed.

### L1-14203432b4  (L1, 2025-10-24, sha 14203432b478, PR #12034)
TITLE: fix(compile_utils, ep_moe): update environment variable and dtype check (#12034)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-3); docs/references/environment_variables.md (+1/-1); python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py (+1/-1)
LABELS: run-ci
BODY: ## Overview ⏎ This PR corrects an environment variable key and updates the data type check in the `DeepEPMoE` layer to account for new conditions. ⏎  ⏎ ## Changes Made ⏎ - Updated environment variable key from `SGL_DG_CACHE_DIR` to `SGLANG_DG_CACHE_DIR` in `compile_utils.py`. ⏎ - Modified dtype assertion in `ep_moe/layer.py` to allow `torch.int32` when `DEEPGEMM_SCALE_UE8M0` is enabled. ⏎  ⏎ ## Testing ⏎ - Verify that the caching mechanism uses the updated envir …[truncated]

### L1-71d41212e4  (L1, 2025-10-24, sha 71d41212e411, PR #12063)
TITLE: Fix dpsk-r1-fp4 launching crash (#12063)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/mxfp4.py (+5/-7); python/sglang/srt/layers/quantization/unquant.py (+6/-13)
LABELS: high priority, run-ci
ISSUES: #12059 [Bug] DeepSeek FP4 Launching fail
BODY: ## Motivation ⏎ Closes #12059, which is introduced by #11795 ⏎  ⏎ This pr partly reverted the changes by #11795 for a quick fix. Maybe refine this in the future. ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-13bf565d60  (L1, 2025-10-24, sha 13bf565d60f8, PR #8464)
TITLE: [2/N]Support DeepSeek-R1 w4a8 low latency deepep (#8464)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.cutlass.w4a8, L1.ep.layer, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+138/-0); python/sglang/srt/layers/moe/ep_moe/kernels.py (+194/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+17/-0); python/sglang/srt/layers/quantization/w4afp8.py (+36/-0); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh (+72/-6); sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh (+4/-2); test/srt/quant/test_w4a8_deepseek_v3.py (+69/-0); test/srt/run_suite.py (+1/-1)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎ Follow https://github.com/sgl-project/sglang/pull/8247 https://github.com/sgl-project/sglang/pull/7762. Based on https://github.com/sgl-project/sglang/pull/8311. ⏎ Support deepep low latency mode for DeepSeek-R1 w4a8 model ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Add forward_cutlass_w4a8_masked for deepep low latency mode, and cudagraph can be enabled. ⏎  ⏎ ## Usage: ⏎ `  SGLANG_DEEPEP_BF16_DISPATCH=1 SGL_ENABLE_JIT_DEEPGEMM=1   python3 -m sglang.la …[truncated]

### L1-649949807f  (L1, 2025-10-24, sha 649949807fd9, PR #12054)
TITLE: [10/N] MoE Refactor: reorganize deepgemm runner in DeepEPMoE (#12054)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.deep_gemm, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+69/-280); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-1); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+287/-22); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+4/-4); python/sglang/srt/layers/moe/token_dispatcher/base.py (+5/-5); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+18/-14); python/sglang/srt/layers/quantization/fp8.py (+3/-4); python/sglang/srt/layers/quantization/w4afp8.py (+4/-4); python/sglang/srt/models/deepseek_v2.py (+2/-4); python/sglang/srt/models/qwen3_moe.py (+2/-4); (+1 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Deprecate `forward_deepgemm_contiguous` and `forward_deepgemm_masked`. Finalize `deep_gemm` backend. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ Server: ⏎ ``` ⏎ python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3-0324 --trust-remote-code --tp 8 --enable-dp-attention --dp 8 --moe-dense-tp-size 1 --enable-dp-lm-head --moe-a2a-backend deepep --enable-two-batch-overlap - …[truncated]

### L1-4b0ac1d52a  (L1, 2025-10-25, sha 4b0ac1d52a6b, PR #12125)
TITLE: Update sgl-kernel version to 0.3.16.post4 (#12125)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎ Fix CI break in  https://github.com/sgl-project/sglang/actions/runs/18796162478/job/53637188677?pr=12098#step:5:558  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-8e70064c37  (L1, 2025-10-25, sha 8e70064c3781, PR #12132)
TITLE: Clean up server launch code and multi tokenizer (#12132)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+0/-1); python/sglang/srt/configs/model_config.py (+1/-1); python/sglang/srt/entrypoints/engine.py (+17/-14); python/sglang/srt/entrypoints/http_server.py (+70/-94); python/sglang/srt/managers/multi_tokenizer_mixin.py (+17/-1); python/sglang/srt/managers/scheduler_metrics_mixin.py (+2/-2); python/sglang/srt/managers/tokenizer_manager.py (+11/-19); python/sglang/srt/server_args.py (+9/-2); python/sglang/srt/utils/common.py (+3/-7)
LABELS: run-ci
BODY: Simplify many redundant logics around server launch and multi-tokenizer

### L1-7ebc28f5d6  (L1, 2025-10-26, sha 7ebc28f5d657, PR #12129)
TITLE: [WIP] support MiniMax M2 model (#12129)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: docs/supported_models/generative_models.md (+1/-0); python/sglang/srt/function_call/function_call_parser.py (+2/-0); python/sglang/srt/function_call/minimax_m2.py (+367/-0); python/sglang/srt/models/minimax_m2.py (+922/-0); python/sglang/srt/parser/reasoning_parser.py (+28/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Supporting MiniMax M2, the SOTA model from MiniMax and we set up a standard for merging new LLMs. ⏎  ⏎ 1. Adding model support to SGLang, with minimal disturbance to SGLang's main logic. ⏎ 2. Adding docs to SGLang, in [LLM](https://github.com/sgl-project/sglang/blob/main/docs/supported_models/generative_models.md) and [VLM](https://github.com/sgl-project/sglang/blob/main/docs/supported_models/multimodal_language_models.md). ⏎ 4. Benc …[truncated]

### L1-6371f7af27  (L1, 2025-10-26, sha 6371f7af27c1, PR #11494)
TITLE: [quantization] AWQ Marlin doesn't work when dtype is bfloat16 (#11494)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: sgl-kernel/python/sgl_kernel/fused_moe.py (+6/-0); python/sglang/srt/layers/quantization/awq.py (+0/-3); python/sglang/srt/model_executor/model_runner.py (+1/-1); test/srt/quant/test_awq.py (+35/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ In [moe_wna16_marlin_gemm](https://github.com/sgl-project/sglang/blob/2db2cddd12a798aaf8b1efe7a88a2fa83d7ec05f/sgl-kernel/csrc/moe/marlin_moe_wna16/ops.cu#L1030-L1104), it implicitly assumes that `a` and `b_scales` are both `half` or both `bfloat16`. If the types are inconsistent, PyTorch will raise a RuntimeError when calling `b_scales.data_ptr<at::Half>()` or `b_scales.data_ptr<at::BFloat16>()`. ⏎  ⏎ * `b_scales` is created by `A …[truncated]

### L1-cadfae666d  (L1, 2025-10-26, sha cadfae666d10, PR #12170)
TITLE: fix broken deepep/flashmla install in container by adding `--no-build-isolation` (#12170)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2)
LABELS: run-ci
BODY: changes -> https://github.com/sgl-project/sglang/pull/12170#pullrequestreview-3381440603

### L1-285a8e6986  (L1, 2025-10-27, sha 285a8e698609, PR #11517)
TITLE: docker: add CUDA13 support in dockerfile and update GDRCopy/NVSHMEM for blackwell support (#11517)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+82/-26); python/pyproject.toml (+3/-22); scripts/ci/ci_install_deepep.sh (+6/-6); .github/workflows/release-docker-cu13.yml (+118/-0); .github/workflows/release-docker-dev.yml (+11/-1); docs/get_started/install.md (+3/-1); scripts/ci/ci_install_dependency.sh (+2/-2)
LABELS: high priority, run-ci
BODY: CUDA13 represents a major bump. Because of this - we do not want to fully default to using it. Instead this container will be used by the team (and others) to develop on CU13 friendly platforms like B/GB300.  ⏎  ⏎ This PR adds ⏎ 1. Support to official dockerfile to build for gb and b300 ⏎ 2. A manual pr trigger that can be used to release a cu13 image for x86/arm ⏎ 3. Updated gdrcopy and nvshmem versions ⏎  ⏎ Will wait for https://github.com/sgl-project …[truncated]

### L1-b1e13e7cea  (L1, 2025-10-28, sha b1e13e7cea7b, PR #12230)
TITLE: [hotfix] Incorrect CombineOverlapArgs in SBO (#12230)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+0/-1); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+12/-9); python/sglang/srt/single_batch_overlap.py (+4/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-fdd00295b5  (L1, 2025-10-28, sha fdd00295b53f, PR #12231)
TITLE: Fix 'BypassedTopKOutput' object has no attribute 'topk_weights' for DeepEP (#12231)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ A recent change was setting the moe runner backend to flashinfer_trtllm by default. With this backend, TopK is bypassed and returns a BypassedTopKOutput. ⏎  ⏎ This caused an error if DeepEP is used, since the DeepEP backend expects a standard topk output. ⏎  ⏎ ``` ⏎ [2025-10-27 12:31:57 DP19 TP19 EP19] Scheduler hit an exception: Traceback (most recent call last): ⏎   File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py" …[truncated]

### L1-77225d602a  (L1, 2025-10-28, sha 77225d602aa0, PR #11928)
TITLE: Use Flashinfer TRT-LLM as Llama 4 compatible MoE backend (#11928)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/modelopt_quant.py (+160/-4); python/sglang/srt/server_args.py (+9/-2); python/sglang/srt/configs/model_config.py (+1/-0); python/sglang/srt/model_loader/weight_utils.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Since the FI TRT-LLM backend was already added for blockscale MoE, add it for per-tensor FP8, specifically for Llama 4. ⏎ Only for SM100 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Modify the apply code of MoE methods. ⏎  ⏎ > Generally I only used ModelOpt checkpoints, so I didn't add the code in the normal FP8 file. ⏎ > There is some issues with ModelOpt loading on main, merge this in after https://github.com/sgl-project/sglang/pull/10154 ⏎  ⏎  ⏎  ⏎ ##  …[truncated]

### L1-813bd6f85c  (L1, 2025-10-28, sha 813bd6f85c3c, PR #10654)
TITLE: [2/2] Use moe_sum_reduce cuda kernel (#10654)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+3/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ This PR is to use moe_sum_reduce cuda kernel implemented in https://github.com/sgl-project/sglang/pull/10321. ⏎  ⏎  ⏎  ⏎ gsm8k result: ⏎ ``` ⏎ ➜  sglang_dev2 git:(use_moe_sum_reduce) ✗ python3 -m sglang.launch_server --model Qwen/Qwen3-30B-A3B --tp-size 8 --port 30000 --mem-fraction-static 0.85 --disable-radix-cache ⏎  ⏎ ➜  sglang_dev2 git:(use_moe_sum_reduce) ✗ python3 benchmark/gsm8k/bench_sglang.py ⏎ PR: ⏎ 100%|█████████████████████████ …[truncated]

### L1-d2b8c4123e  (L1, 2025-10-28, sha d2b8c4123ef5, PR #10567)
TITLE: Opt fused triton moe: add tma for down proj kernel (#10567)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.triton.moe_align
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json (+164/-0); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+65/-20); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py (+29/-3); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (+106/-26); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py (+818/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ In H20(96GB) TP8 prefill, during performance analysis, we observed that the latency of the second MOE in each layer (i.e., the downsampled Fused Triton MOE) was comparable to that of the first MOE (upsampled Fused Triton MOE), even though its weight data volume and computational cost were only half of the first MOE. This latency performance was unreasonable.  ⏎  ⏎ For the second MOE (downsampling MOE), we use TMA to encapsulate i …[truncated]

### L1-c143f416ce  (L1, 2025-10-28, sha c143f416ce47, PR #12308)
TITLE: fix: Llama 4 BF16 load on Blackwell (#12308)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+5/-4)
LABELS: run-ci
BODY: Small oversight in https://github.com/sgl-project/sglang/pull/11928. This fp8 moe kernel currently only support fp8 quantization input, therefore, it will choose this wrongly for bf16 or fp4 still. ⏎  ⏎ Fix it

### L1-83087247d1  (L1, 2025-10-28, sha 83087247d16e, PR #12259)
TITLE: [hotfix] missing `w13_weight_fp8` and `w2_weight_fp8` in UE8M0 requantization (#12259)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.runner.deep_gemm, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+0/-17); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+8/-7); python/sglang/srt/models/deepseek_v2.py (+20/-4); python/sglang/srt/models/longcat_flash.py (+2/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ - Missing `w13_weight_fp8` and `w2_weight_fp8` in UE8M0 requantization ⏎ - In correct assertion and scales roundup for deepep + deep_gemm on blackwell ⏎  ⏎ Fix an error reported by @ishandhanani. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-400bddf24c  (L1, 2025-10-29, sha 400bddf24cac, PR #12315)
TITLE: [router] fix router release workflow and add build test in PR (#12315)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+7/-2); .github/workflows/nightly-release-router.yml (+3/-3); .github/workflows/pr-test-pd-router.yml (+3/-1); .github/workflows/pr-test-rust.yml (+54/-5); .github/workflows/release-pypi-router.yml (+3/-3); sgl-router/Cargo.toml (+6/-1); sgl-router/MANIFEST.in (+1/-0); sgl-router/README.md (+12/-11); sgl-router/py_test/unit/test_arg_parser.py (+1/-1); sgl-router/py_test/unit/test_router_config.py (+1/-1); (+3 more)
LABELS: router, run-ci
BODY: ## 1. Fix PyPI Release Workflows ⏎  ⏎ **Problem**: Workflows still used `python -m build` with setuptools-rust, which no longer works after migrating to maturin backend. ⏎  ⏎ **Solution**: Updated all build commands across workflows: ⏎ - `release-pypi-router.yml` ✓ (already fixed) ⏎ - `nightly-release-router.yml` ✓ (already fixed) ⏎ - `pr-test-rust.yml` - pytest-rust and pytest-rust-2 jobs ⏎ - `pr-test-pd-router.yml` - build job ⏎  ⏎ **Changes**: ⏎ ```bash …[truncated]

### L1-e39628fd07  (L1, 2025-10-29, sha e39628fd07b3, PR #12095)
TITLE: [2/2] Deepseek deterministic: support deepseek v3 deterministic inference on 8 x H200 (#12095)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py (+14/-0); python/sglang/srt/models/deepseek_v2.py (+3/-0); test/srt/test_fused_moe.py (+2/-0)
LABELS: run-ci
BODY: ## Motivation ⏎ part of [roadmap](https://github.com/sgl-project/sglang/issues/10278) ⏎ Previous [PR](https://github.com/sgl-project/sglang/pull/12000) supported deepseek arch model's deterministic inference on a single Hopper GPU.  ⏎ This PR is to further support full deepseek v3 model's  deterministic inference on **8 x H200**.  ⏎  ⏎ This change also fixed this [issue](https://github.com/sgl-project/sglang/issues/12232) ⏎  ⏎  ⏎ ## Modifications ⏎ 1. Use …[truncated]

### L1-ed1044ac1b  (L1, 2025-10-29, sha ed1044ac1b89, PR #11737)
TITLE: support cutlass fp4 kernel in sm120 (#11737)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.nvfp4
FILES: sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cu (+259/-32); sgl-kernel/csrc/gemm/nvfp4_quant.cuh (+2/-2); sgl-kernel/csrc/gemm/nvfp4_quant_entry.cu (+6/-3); sgl-kernel/csrc/gemm/nvfp4_quant_kernels.cu (+5/-2); sgl-kernel/csrc/gemm/nvfp4_scaled_mm_entry.cu (+27/-2); sgl-kernel/csrc/gemm/nvfp4_scaled_mm_kernels.cu (+244/-10)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Add SM120 (GeForce RTX 50 series) support to the existing SM100/SM103 implementation for NVFP4 quantization. ⏎  ⏎ ## Modifications ⏎  ⏎ - Added SM120-Specific cutlass kernel in sgl-kernel:  cutlass_scaled_fp4_mm, cutlass_fp4_group_mm. ⏎ - Added SM120/SM121 support in nvfp4_quant,  nvfp4_export_quant. ⏎ - Updated is_blackwell() function: Now returns true for both compute capability 10.0 and 12.0 ⏎ - Restored original function naming: Cha …[truncated]

### L1-1357397a34  (L1, 2025-10-29, sha 1357397a34ab, PR #12276)
TITLE: feat: preview filename from tuning_fused_moe_triton.py (#12276)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+27/-15)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Since the `tuning_fused_moe_triton.py` script may be used when working backwards from the following warning, and the tuning process can take some time, it would be helpful to preview the filename of the generated config to ensure that the desired one is being created: ⏎ ```bash ⏎ [2025-10-28 09:42:18 TP0] Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /sgl-workspace/sglang/python/sglang/ …[truncated]

### L1-750940ae36  (L1, 2025-10-29, sha 750940ae3660, PR #12002)
TITLE: Eagle3 DP attention for Qwen3 MoE (#12002)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/communicator.py (+23/-1); python/sglang/srt/models/llama_eagle3.py (+11/-1); python/sglang/srt/models/qwen2_moe.py (+30/-15); python/sglang/srt/models/qwen3_moe.py (+16/-8); python/sglang/srt/server_args.py (+1/-1); python/sglang/srt/speculative/eagle_worker.py (+6/-1); python/sglang/test/test_utils.py (+2/-0); test/srt/run_suite.py (+1/-0); test/srt/test_eagle_dp_attention.py (+129/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Add eagle3 dp attention for Qwen3 MoE model for large scale EP deployment. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎ We test DP attention for Eagle3 in non-PD and PD scenarios. The results are all good. ⏎  ⏎ ### Single node of 8xH100 using Eagle and DP attention ⏎ engine command ⏎ ```bash ⏎ python3 -m sglang.launch_server --model-path ~/models/Qwen3-235B-A22B --trust-remote-code --disable-rad …[truncated]

### L1-52694b60da  (L1, 2025-10-29, sha 52694b60dab3, PR #12343)
TITLE: Triton fused_moe_kernel support ep moe tuning (#12343)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts(reverted), corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=32,N=768,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+146/-0); benchmark/kernels/fused_moe_triton/README.md (+10/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+101/-28); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py (+93/-25)
LABELS: run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 12377 (confirmed_revert, reason=correctness_or_accuracy) || deep-study performance PR (kernel_tuning_config)
BODY: ```shell ⏎  ⏎ CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py --model Qwen/Qwen3-30B-A3B-FP8 --ep-size 4 --tune --dtype fp8_w8a8 ⏎  ⏎ CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-FP8 --tp-size 4 --ep 4 ⏎  ⏎ python3 -m sglang.bench_serving --backend sglang-oai  --dataset-name random --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1 --n …[truncated]

### L1-e5ec976402  (L1, 2025-10-30, sha e5ec9764021b, PR #12362)
TITLE: [Bug fix][PP] fix deadlock with tie_word_embeddings (#12362)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen2.py (+1/-1); python/sglang/srt/models/qwen3.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Now when run Qwen model with tie_word_embeddings set  and the pp size > 2, the instance hangs like below: ⏎  ⏎ [2025-10-30 08:58:23 PP3] Init torch distributed ends. mem usage=0.16 GB                                                                                                                        ⏎ [2025-10-30 08:58:23 PP0] Init torch distributed ends. mem usage=0.16 GB                                                            …[truncated]

### L1-04e5b6faa7  (L1, 2025-10-30, sha 04e5b6faa7ee, PR #12377)
TITLE: Revert "Triton fused_moe_kernel support ep moe tuning" (#12377)
SOURCES: path_core, subject_keyword, release_notes, corpus:confirmed-reverts
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=32,N=768,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+0/-146); benchmark/kernels/fused_moe_triton/README.md (+0/-10); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+28/-101); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py (+25/-93)
LABELS: run-ci
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 12343 reason=correctness_or_accuracy
BODY: Reverts sgl-project/sglang#12343

### L1-b7fdde4bb4  (L1, 2025-10-30, sha b7fdde4bb499, PR #12375)
TITLE: [ci] Fix ci_install_deepep (#12375)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: scripts/ci/ci_install_deepep.sh (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-df5192cffa  (L1, 2025-10-30, sha df5192cffa02, PR #11806)
TITLE: Enable fast silu-and-mul-and-quant fused kernel (#11806)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.deep_gemm
FILES: python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+45/-26); python/sglang/srt/layers/quantization/fp8_kernel.py (+1/-1)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-cafebef154  (L1, 2025-10-30, sha cafebef1546e, PR #11969)
TITLE: [NPU] bugfix for Qwen3-Next and performance update (#11969)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+31/-6); .github/workflows/release-docker-npu-nightly.yml (+1/-1); .github/workflows/release-docker-npu.yml (+1/-1); python/sglang/srt/layers/attention/fla/layernorm_gated.py (+7/-1); python/sglang/srt/layers/attention/mamba/mamba.py (+20/-11); python/sglang/srt/models/qwen3_next.py (+7/-0); scripts/ci/npu_ci_install_dependency.sh (+1/-1)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR aims to solve model issues and improve model performance and throughput on Ascend NPUs. ⏎ We have reached a better performance result on a single Altas 800I A2 and half a single Atlas 800I A3 machine. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Provides improved triton kernels for decode ⏎ - Replaces torch-natively implemented topk with fused kernel api provided with torch_npu ⏎ - Fix a padding issue when DP-Attn is enabled ⏎  ⏎ ## Accur …[truncated]

### L1-c0652d907b  (L1, 2025-10-31, sha c0652d907b2e, PR #12413)
TITLE: Clean up sgl kernel (#12413)
SOURCES: path_core, symbol_pickaxe, dependency_pin, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_softmax
FILES: sgl-kernel/CMakeLists.txt (+26/-40); sgl-kernel/cmake/flashmla.cmake (+2/-0); sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu (+129/-16); sgl-kernel/python/sgl_kernel/moe.py (+20/-2); python/sglang/srt/entrypoints/openai/protocol.py (+5/-1); sgl-kernel/csrc/common_extension.cc (+52/-49); sgl-kernel/csrc/common_extension_rocm.cc (+9/-9); sgl-kernel/include/sgl_kernel_ops.h (+43/-35); sgl-kernel/python/sgl_kernel/__init__.py (+3/-226); sgl-kernel/python/sgl_kernel/load_utils.py (+224/-0)
LABELS: run-ci
BODY: - minor clean up sgl kernel python and c++ files ⏎ - add some arguments to `topk_softmax`

### L1-2d5605e89b  (L1, 2025-10-31, sha 2d5605e89bd8, PR #12449)
TITLE: Fix ci install to allow prerelease (#12449)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: sgl-kernel/python/sgl_kernel/fused_moe.py (+1/-1); scripts/ci/ci_install_dependency.sh (+1/-1)
LABELS: run-ci
BODY: 

### L1-82cfcd3bb8  (L1, 2025-10-31, sha 82cfcd3bb80b, PR #11224)
TITLE: [Refactor] tuning_fused_moe for MLLM and small refactor (#11224)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+60/-56)
LABELS: run-ci
BODY: ## Motivation ⏎ ~~blocks by #11199~~ ⏎ refactor tune moe script to nested config with `text_config` ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a4bf5c6ad2  (L1, 2025-10-31, sha a4bf5c6ad25d, PR #12469)
TITLE: Support Kimi Linear (#12469)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/kimi_linear.py (+160/-0); python/sglang/srt/configs/mamba_utils.py (+66/-0); python/sglang/srt/configs/model_config.py (+7/-0); python/sglang/srt/layers/attention/attention_registry.py (+3/-0); python/sglang/srt/layers/attention/fla/chunk_delta_h.py (+61/-32); python/sglang/srt/layers/attention/fla/fused_recurrent.py (+17/-4); python/sglang/srt/layers/attention/fla/kda.py (+1359/-0); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+223/-0); python/sglang/srt/layers/attention/triton_backend.py (+4/-1); (+8 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Support Kimi Linear model (https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct). ⏎  ⏎ Major work is done by @yizhang2077 .  Thanks @zhiyuan1i for valuable discussion. ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server --model moonshotai/Kimi-Linear-48B-A3B-Instruct --tp 4 --trust-remote ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 1319 --parallel 1319 ⏎  ⏎ Accuracy: 0.895 ⏎ Invalid: 0.000 ⏎ Latency: 46.696 s ⏎ Output through …[truncated]

### L1-756ad9ceb1  (L1, 2025-11-01, sha 756ad9ceb14b, PR #12238)
TITLE: Reduce docker image size. mount cache when use pip/cargo build (#12238)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+21/-21)
LABELS: run-ci
BODY: When building an image with Docker, for installations via pip and cargo， and  apt , you should use mount cache to prevent the built image from becoming excessively large. ⏎ CC @zhyncs  @slin1237

### L1-0afd68321b  (L1, 2025-11-01, sha 0afd68321bc7, PR #12391)
TITLE: Update Mooncake EP's a2a interface (#12391)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/mooncake.py (+16/-8); python/sglang/srt/layers/quantization/fp8.py (+2/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The interface of token_dispatcher has changed (specifically, the `combine` API). ⏎  ⏎ This PR updates the Mooncake's token_dispatcher to match the API change. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Existing unit tests should pass. ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0c3543d7d5  (L1, 2025-11-02, sha 0c3543d7d507, PR #12523)
TITLE: chore: upgrade flashinfer 0.5.0 (#12523)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+3/-1); python/sglang/check_env.py (+2/-0); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+3/-1); scripts/ci/ci_install_dependency.sh (+2/-1); sgl-kernel/build.sh (+1/-1)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6e29446e45  (L1, 2025-11-02, sha 6e29446e45e8, PR #12530)
TITLE: [hotfix] Remove flashinfer-jit-cache from pyproject (#12530)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+0/-1); scripts/ci/ci_install_dependency.sh (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎ flashinfer-jit-cache cannot be installed without extra-url-index, which might hurt user experience ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-ab8b83f71d  (L1, 2025-11-03, sha ab8b83f71d1f, PR #12541)
TITLE: chore: upgrade mooncake 0.3.7.post1 (#12541)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: scripts/ci/ci_install_deepep.sh (+0/-4); .github/workflows/pr-test-pd-router.yml (+1/-1); docker/Dockerfile (+1/-1); docker/b300.Dockerfile (+1/-1); scripts/ci/ci_install_dependency.sh (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-14d8064803  (L1, 2025-11-03, sha 14d806480308, PR #12536)
TITLE: fix: Fix KTransformers hybrid inference with int8 quantization and format (#12536)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+2/-4)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Add amx_method to support more KT-kernel quantization backend. ⏎  ⏎ Format routed_scaling_factor ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-7a21d8b276  (L1, 2025-11-03, sha 7a21d8b2765e, PR #12524)
TITLE: Reduce the overhead of nccl symmetric memory (#12524)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+8/-3); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+26/-10); python/sglang/srt/distributed/device_communicators/pynccl.py (+2/-1); python/sglang/srt/distributed/device_communicators/pynccl_allocator.py (+53/-51); python/sglang/srt/distributed/parallel_state.py (+7/-10); python/sglang/srt/entrypoints/engine.py (+9/-3); python/sglang/srt/layers/linear.py (+2/-2); python/sglang/srt/layers/quantization/fp8.py (+41/-26); python/sglang/srt/layers/quantization/modelopt_quant.py (+41/-23); python/sglang/srt/layers/quantization/mxfp4.py (+14/-7); (+4 more)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: - directly register the buffer in c++ code instead of calling `get_nccl_mem_pool().snapshot()`. This reduces the CPU overhead. ⏎ - apply some changes from #9358.  I ported the clean up for the TP part. The support of dp attention will be in another PR. ⏎  ⏎  ⏎  ⏎  ⏎ some issues: ⏎ 1. symmetric memory is slower than custom allreduce at small batch size (1). symmetric memory  is faster at large batch size (128). ⏎ 2. symmetric memory is not compatbile with  …[truncated]

### L1-dbcf85b7f0  (L1, 2025-11-04, sha dbcf85b7f0c1, PR #10183)
TITLE: Add --speculative-moe-runner-backend server arg (#10183)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/__init__.py (+0/-2); python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-2); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-6); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+3/-3); python/sglang/srt/layers/moe/topk.py (+2/-5); python/sglang/srt/layers/moe/utils.py (+33/-13); python/sglang/srt/layers/quantization/fp8.py (+8/-19); python/sglang/srt/layers/quantization/modelopt_quant.py (+9/-6); python/sglang/srt/models/deepseek_v2.py (+2/-2); python/sglang/srt/server_args.py (+22/-0); (+5 more)
LABELS: high priority, run-ci
ISSUES: #9441 [Feature] use cutlass moe for deepseek v3 mtp draft model | #9605 [Bug] [NVIDIA][B200] DSR1 fp8 server failes to launch with NCCL error
BODY: ## Motivation ⏎  ⏎ Allows moe runner backend to be configured separately for mtp draft model. ⏎  ⏎ Fixes https://github.com/sgl-project/sglang/issues/9441 ⏎  ⏎ ## Modifications ⏎  ⏎ * Add --speculative-moe-runner-backend to select moe runner backend to use for EAGLEWorker. Draft model init and forward are wrapped with a context that overrides the global setting for `get_moe_runner_backend()`. ⏎ * Replace env var `SGLANG_CUTLASS_MOE` with `--moe-runner-bac …[truncated]

### L1-211f4070e5  (L1, 2025-11-04, sha 211f4070e58f, PR #12641)
TITLE: fix: Lazy import mooncake-ep to fix extra gpu contexts being created (#12641)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/mooncake.py (+6/-8)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Recently @nvcastet noticed extra GPU contexts were accidentally being created and wasting gpu memory: ⏎  ⏎ ToT main: ⏎ ``` ⏎ +-----------------------------------------------------------------------------------------+ ⏎ | Processes:                                                                              | ⏎ |  GPU   GI   CI              PID   Type   Process name                        GPU Memory | ⏎ |        ID   ID                  …[truncated]

### L1-42889acbd0  (L1, 2025-11-04, sha 42889acbd0d0, PR #12642)
TITLE: [hotfix] Fix deepep w4a8 bug (#12642)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+3/-3)
LABELS: run-ci
BODY: ## Motivation ⏎ Ref: https://github.com/sgl-project/sglang/actions/runs/19078958193/job/54512657896?pr=12639 ⏎ Discussed with @trevor-m. This line can be safely removed. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-34f7564df0  (L1, 2025-11-04, sha 34f7564df037, PR #12640)
TITLE: [NVIDIA] Fix wrong symmetric sizes for fp4 cases (#12640)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/modelopt_quant.py (+6/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The recent [changes](https://github.com/sgl-project/sglang/commit/7a21d8b2765eabed74d8a8a3abced571b23492d0) seem to have broken the FP4 use cases. ⏎ The root cause is that the output shape of MoE should not be assumed to match the input, since FP4 inputs are stored in a packed format while the outputs are typically in higher precision (e.g., BF16). ⏎  ⏎ People might see errors like when using the flashinfer moe backend for nvfp4: ⏎ ` …[truncated]

### L1-2340798353  (L1, 2025-11-04, sha 2340798353bc, PR #12572)
TITLE: Register allgather/reducescatter buffers with symm memory (#12572)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-6); python/sglang/srt/layers/moe/topk.py (+21/-9); python/sglang/srt/distributed/device_communicators/pynccl_allocator.py (+16/-21); python/sglang/srt/distributed/parallel_state.py (+76/-19); python/sglang/srt/layers/communicator.py (+11/-1); python/sglang/srt/layers/dp_attention.py (+34/-11); python/sglang/srt/layers/linear.py (+4/-2); python/sglang/srt/layers/quantization/fp8.py (+13/-6); python/sglang/srt/layers/quantization/modelopt_quant.py (+30/-22); (+9 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Rebase version of https://github.com/sgl-project/sglang/pull/9358 on top main (including https://github.com/sgl-project/sglang/pull/12524) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-0711d1509b  (L1, 2025-11-04, sha 0711d1509b94, PR #12353)
TITLE: [NVIDIA] Fix cutedsl backend of MoE (#12353)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl
FILES: python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py (+9/-9); scripts/ci/ci_install_deepep.sh (+58/-8); .github/workflows/pr-test.yml (+2/-2); python/sglang/srt/layers/quantization/modelopt_quant.py (+6/-2); python/sglang/test/test_utils.py (+1/-1); test/srt/run_suite.py (+3/-0); test/srt/test_cutedsl_moe.py (+482/-0); test/srt/test_deepseek_v3_cutedsl_4gpu.py (+18/-6)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The cutedsl moe backend execution is broken. This PR fixes it. ⏎  ⏎ ## Modifications ⏎  ⏎ Fixed the execution path and added CI tests. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ The CI test results: ⏎ ``` ⏎ Accuracy: 0.951 ⏎ Invalid: 0.000 ⏎ Latency: 61.249 s ⏎ ``` ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b419e20c5b  (L1, 2025-11-04, sha b419e20c5b62, PR #8784)
TITLE: [Dockerfile] Speed up docker image building (#8784)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+38/-24)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ It may be very slow or competely unaccessable when build docker image in some special network environment, it's a block if we want to build the image ourselves. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ 1. Introduce `UBUNTU_MIRROR` to allow replacing default Ubuntu package sources, e.g. `http://mirrors.aliyun.com` ⏎ 2. Introduce `PIP_DEFAULT_INDEX` to enable setting a custom PyPI index URL, e.g. `https://mirrors.aliyun.com/pypi/simple/` ⏎ 3. Intr …[truncated]

### L1-fb2e816e83  (L1, 2025-11-05, sha fb2e816e83d7, PR #12696)
TITLE: Fix server args for gpt oss so users can override the moe runner backend (#12696)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+15/-18)
LABELS: run-ci
BODY: 

### L1-97be66c358  (L1, 2025-11-05, sha 97be66c35830, PR #12723)
TITLE: fix sgl-kernel version (#12723)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ sgl kernel version had been bumped in Dockerfile, but it would be overwritten when sgl be installed. ⏎  ⏎  ⏎ ``` ⏎ #14 2.391 Collecting sgl-kernel==0.3.16.post5+cu130 ⏎ #14 2.764   Downloading https://github.com/sgl-project/whl/releases/download/v0.3.16.post5/sgl_kernel-0.3.16.post5+cu130-cp310-abi3-manylinux2014_aarch64.whl (252.3 MB) ⏎ #14 5.681      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 252.3/252.3 MB 87.9 MB/s  0:00:02 ⏎ #14 5.93 …[truncated]

### L1-a889c85459  (L1, 2025-11-06, sha a889c854595e, PR #12455)
TITLE: [Grammar Fix] GLM-4-MOE self.first_k_dense_replace is undefined. (#12455)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/glm4_moe.py (+214/-65)
LABELS: run-ci
BODY: Simple bug fix.

### L1-2104d20eba  (L1, 2025-11-06, sha 2104d20eba2b, PR #12738)
TITLE: Temporarily fix missing routed_scaling_factor for CompressedTensorsWNA16MoEMethod (#12738)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+17/-3)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Temporarily add the missing `routed_scaling_factor` to `CompressedTensorsWNA16MoEMethod`. Later, we need to review all MoE methods to check for this factor—it's likely that other MoE methods are also missing it. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-a119363f08  (L1, 2025-11-06, sha a119363f0863, PR #12782)
TITLE: ignore the deepgemm check when the model weight with nvfp4 and moe ba… (#12782)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+9/-1)
LABELS: run-ci
BODY: …ckend is flashinfer cutedsl ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ support DS R1 FP4 model deploy to GB200 with deepep and flashinfer-cutedsl moe backend. ⏎  ⏎ ## Modifications ⏎  ⏎ ease the DeepGEMM check when the model weight is fp4 quantitized and the moe backend is flashinfer-cutedsl ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ For the decode ⏎  ⏎ **Rank 0** ⏎  ⏎ ```bash ⏎ NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_SOCKET_FAMILY=AF_INET GLOO_SOCKET_IFNAME=eth0 …[truncated]

### L1-fc84b0730c  (L1, 2025-11-06, sha fc84b0730ce7, PR #12440)
TITLE: [Refactor] Refactor fused_moe_triton tuning tools: extract shared utils, add EP/MLLM support, reduce overhead (#12440)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/README.md (+145/-11); benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py (+6/-59); benchmark/kernels/fused_moe_triton/benchmark_vllm_vs_sglang_fused_moe_triton.py (+5/-90); benchmark/kernels/fused_moe_triton/common_utils.py (+256/-0); benchmark/kernels/fused_moe_triton/tuning_client.py (+71/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+34/-245); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py (+45/-168); benchmark/kernels/fused_moe_triton/tuning_text.json (+1/-0)
LABELS: documentation, performance, run-ci
BODY: ## Summary ⏎ - Refactored tuning and benchmark scripts under [benchmark/kernels/fused_moe_triton/](cci:7://file:///home/yineng/bbuf/sglang/benchmark/kernels/fused_moe_triton:0:0-0:0) to remove duplication and improve maintainability. ⏎ - Added full Expert Parallelism (EP) support and MLLM model handling to tuning scripts. ⏎ - Updated README with EP usage guidance and detailed instructions for the separate-kernel tuning workflow. ⏎ - Fix tuning script …[truncated]

### L1-1fa788ec14  (L1, 2025-11-06, sha 1fa788ec1436, PR #12758)
TITLE: [Bugfix] Fix illegal memory access (#12758)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+9/-1); python/sglang/srt/layers/flashinfer_comm_fusion.py (+12/-3); python/sglang/srt/layers/quantization/mxfp4.py (+9/-1)
LABELS: high priority, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Fixed illegal memory access issue in #12695 and https://github.com/flashinfer-ai/flashinfer/issues/2034 ⏎ Caused by #12524 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-cd135bfe30  (L1, 2025-11-06, sha cd135bfe303b, PR #12778)
TITLE: Update dsv3 quantization auto setting for sm100 (#12778)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+22/-9)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ update quantization based on model quant_method

### L1-bc25ea6762  (L1, 2025-11-07, sha bc25ea6762a2, PR #12090)
TITLE: [MoE] Add Comprehensive MoE Integration Tests (#12090)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+1/-1); python/sglang/test/test_utils.py (+5/-1); test/srt/layers/moe/test_moe_runners.py (+193/-0); test/srt/run_suite.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ This PR adds a comphrensive suite of MoE integration tests to add test coverage for MoE models from beginning to end. Currently, MoE code is only tested tangentially through the testing of other features (i.e., quantization). ⏎  ⏎ ## Modifications ⏎  ⏎ Added test file `test/srt/layers/moe/test_moe_runners.py` with integration tests. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-6e6009fb6b  (L1, 2025-11-07, sha 6e6009fb6be7, PR #8243)
TITLE: [CPU] Fix TP padding case with weight block size (#8243)
SOURCES: path_core
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: sgl-kernel/csrc/cpu/topk.cpp (+3/-0); python/sglang/srt/configs/update_config.py (+23/-5); python/sglang/srt/layers/quantization/unquant.py (+1/-4)
LABELS: ready-to-merge, intel, cpu, run-ci
DEEP_STUDY: deep-study correctness case sglang:6e6009fb6b: class=shape_alignment_edge; symptom=crash_or_exception; introducing=unknown
BODY: ### Motivation ⏎ Fixes the [Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct) (FP8) TP=6 failure on CPU. ⏎  ⏎ ValueError: Weight output_partition_size = 2112 is not divisible by weight quantization block_n = 128. ⏎  ⏎ ### Modifications ⏎ Considering `weight_block_size `when padding TP, to make self.num_heads * self.qk_head_dim / tp_size divisible by `weight_block_size `in the below ColumnParallelLinear. ⏎ ``` ⏎   self.q_b_proj = Colum …[truncated]

### L1-0f76976c3c  (L1, 2025-11-07, sha 0f76976c3ccf, PR #12801)
TITLE: remove the fa4 page_size hardcode to 128 restriction on mla model arch (#12801)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+2/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ When prefill attn backend set to FA4, the page size is hardcode to 128, which lead to the DS R1 FP4 model large scale EP deployment could not deploy with the following combination due to the trtllm-mla only support 16, 32, 64 page size.  ⏎ - Prefill: FA4 attn backend, flashinfer_trtllm moe backend ⏎ - Decode: trtllm-mla attn backend, flashinfer_cutedsl moe backend + deepep low_latency a2a ⏎  ⏎ ## Modifications ⏎  ⏎ remove the page size …[truncated]

### L1-55e8e3999c  (L1, 2025-11-07, sha 55e8e3999ca1, PR #12851)
TITLE: add back flashinfer jit cache to dev docker (#12851)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+4/-0); .github/workflows/release-docker-dev.yml (+1/-0)
LABELS: run-ci
BODY: for ease of development & since the dev docker does not have a slim requirement

### L1-b8ddc296f4  (L1, 2025-11-07, sha b8ddc296f448, PR #12582)
TITLE: [sgl-kernel][Deepseek V3.2] Add row_starts to topk kernel (#12582)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-kernel/python/sgl_kernel/top_k.py (+61/-7); sgl-kernel/csrc/common_extension.cc (+3/-3); sgl-kernel/csrc/elementwise/topk.cu (+51/-24); sgl-kernel/include/sgl_kernel_ops.h (+9/-3); sgl-kernel/tests/test_topk.py (+85/-24)
LABELS: high priority, sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎ Part1 of the fix for bug in https://github.com/sgl-project/sglang/issues/11629 ⏎  ⏎ ## Modifications ⏎  ⏎ In the topk kernel for prefill, the q and k inputs are both ragged. We need to pass the correct start indices of k for each q token to the kernel. ⏎  ⏎ In `fast_topk_transform_interface`, I changed the `is_decode` criterion from `prefill_bs == B` to `!row_starts_opt.has_value() and prefill_bs == B`. Using `prefill_bs == B` can be a …[truncated]

### L1-44f594d832  (L1, 2025-11-09, sha 44f594d8325f, PR #12888)
TITLE: Apply moe_reduce_sum kernel for fused_marlin_moe (#12888)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: sgl-kernel/python/sgl_kernel/fused_moe.py (+12/-2)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ before apply (bs1): 4.6μs ⏎ <img width="349" height="77" alt="image" src="https://github.com/user-attachments/assets/4b8e20e8-4fc4-4005-bcff-076bf4ed7ffc" /> ⏎ after apply (bs1): 1.5μs ⏎ <img width="277" height="64" alt="image" src="https://github.com/user-attachments/assets/e36321e6-4dea-4fa7-801a-513c0a17b7de" />

### L1-ddd1440d0f  (L1, 2025-11-09, sha ddd1440d0f02, PR #12834)
TITLE: Refactor KTransformers heterogeneous compute with unified GPU-quantization backend (#12834)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+24/-19); python/sglang/srt/layers/moe/kt_ep_wrapper.py (+393/-0); python/sglang/srt/environ.py (+0/-10); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+25/-8); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+1/-411); python/sglang/srt/layers/radix_attention.py (+1/-0); python/sglang/srt/model_executor/cuda_graph_runner.py (+2/-2); python/sglang/srt/models/deepseek_v2.py (+5/-14); python/sglang/srt/models/glm4_moe.py (+37/-0); python/sglang/srt/server_args.py (+6/-43)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Replace scattered, hard-coded paths and env-var checks with one clean architecture that can load any GPU quant method. ⏎  ⏎ ## Modifications ⏎  ⏎ Introduced KTEPWrapperMethod a unify backend interface; all quant methods register their method/configs there. ⏎  ⏎ Removed legacy hard-coded branches and env-var. ⏎  ⏎ Added linear_fp8_config inside the compressed-tensor path to enable mixed-precision for DeepSeek block-fp8 (note: compressed-t …[truncated]

### L1-9cfe78dd30  (L1, 2025-11-09, sha 9cfe78dd3076, PR #12957)
TITLE: clean redundant code in previous PR (#12957)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/glm4_moe.py (+0/-37)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ #12834 add some redundant code in python/sglang/srt/models/glm4_moe.py. This PR remove it. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-611a4fd08b  (L1, 2025-11-10, sha 611a4fd08ba0, PR #11719)
TITLE: [router] bucket policy (#11719)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+1/-0); sgl-router/py_src/sglang_router/router_args.py (+8/-1); sgl-router/src/config/types.rs (+33/-0); sgl-router/src/config/validation.rs (+96/-0); sgl-router/src/core/workflow/steps/worker_registration.rs (+7/-0); sgl-router/src/lib.rs (+11/-0); sgl-router/src/policies/bucket.rs (+1167/-0); sgl-router/src/policies/factory.rs (+28/-6); sgl-router/src/policies/mod.rs (+19/-0); sgl-router/src/policies/registry.rs (+32/-2); (+2 more)
LABELS: feature, router, run-ci
BODY: ## Motivation ⏎  ⏎ In SGLang’s Prefill–Decode (PD) disaggregation architecture: ⏎ | Stage      | Runtime complexity vs. prompt length \( L \) | ⏎ |------------|---------------------------------------------| ⏎ | Attention  | \$( O(L^2) \)$                                | ⏎ | MoE        | \$( O(L) \)$                                  | ⏎  ⏎ Core Issue: When short prompts (e.g., 32-token) and long prompts (e.g., multi-kilotoken) share a micro-batch, the qu …[truncated]

### L1-2fe4e69fca  (L1, 2025-11-10, sha 2fe4e69fca89, PR #12218)
TITLE: [router] add postgres databases data connector (#12218)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-router/py_src/sglang_router/router.py (+12/-0); sgl-router/Cargo.toml (+3/-0); sgl-router/py_src/sglang_router/router_args.py (+1/-1); sgl-router/src/config/builder.rs (+10/-2); sgl-router/src/config/types.rs (+54/-0); sgl-router/src/data_connector/common.rs (+78/-0); sgl-router/src/data_connector/factory.rs (+51/-1); sgl-router/src/data_connector/mod.rs (+2/-0); sgl-router/src/data_connector/oracle.rs (+7/-78); sgl-router/src/data_connector/postgres.rs (+699/-0); (+2 more)
LABELS: high priority, dependencies, router, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/issues/10341 ⏎  ⏎ Postgres data store provider. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ requerment: we need a postgres database, and create a name is `sql_router` db. ⏎  ⏎ 1. start sgl-router use this pararm  ⏎ ``` ⏎ $ sglang-router  --backend openai --worker-urls http://10.20.100.240:31896 --history-backend postgres --postgres-db-url postgres://postgres:xxxxx@xxxx:xxxx/sql_router?ssl …[truncated]

### L1-37c40a87a8  (L1, 2025-11-10, sha 37c40a87a8cb, PR #12966)
TITLE: chore: bump sgl-kernel version to 0.3.17 (#12966)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.17` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.17` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are updated accord …[truncated]

### L1-58b12ccb46  (L1, 2025-11-10, sha 58b12ccb4629, PR #12996)
TITLE: Support piecewise cuda graph for deepseek v3 (#12996)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+24/-0); python/sglang/srt/server_args.py (+7/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Followup of https://github.com/sgl-project/sglang/pull/11812. ⏎  ⏎ ``` ⏎ python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-0528 --tp 8 --trust-remote-code --enable-piecewise-cuda-graph --piecewise-cuda-graph-max-tokens 8192 ⏎  ⏎ curl http://127.0.0.1:30000/flush_cache            ⏎ python3 -m sglang.bench_serving --backend sglang-oai  --dataset-name random --random-input-len 1024 --random-output-len 10 --random-range-r …[truncated]

### L1-f18ec927f3  (L1, 2025-11-11, sha f18ec927f360, PR #13027)
TITLE: fix tuning_fused_moe_triton_sep tool per_channel_quant bug (#13027)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py (+1/-2)
LABELS: performance, run-ci
BODY: ## Motivation ⏎  ⏎ <img width="2826" height="1020" alt="图片" src="https://github.com/user-attachments/assets/48a608ce-f22e-41f0-9096-0ee4003bcaba" /> ⏎  ⏎ <img width="2878" height="812" alt="图片" src="https://github.com/user-attachments/assets/63fb156e-8d3c-4863-b048-a199c49efe20" /> ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-99e25805f5  (L1, 2025-11-11, sha 99e25805f54e, PR #12866)
TITLE: [Fix] Fix nan error for large scale ep (#12866)
SOURCES: corpus:kernel-correctness-cases(introducing)
ARTIFACT_HINTS: -
FILES: python/sglang/srt/eplb/expert_location.py (+8/-4)
LABELS: run-ci
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:7aa443903d (fix PR 13162)
BODY: ## Motivation ⏎ Part of https://github.com/sgl-project/sglang/issues/12293/ ⏎  ⏎ This bug is introduced in https://github.com/sgl-project/sglang/pull/10874, which will wrongly remove some of the redundant experts from `logical_to_all_physical_map` (e.g., should be [0, 256] for logical expert 0, but wrongly set to [0, -1]) ⏎  ⏎ This only happens on the first and the last node. On these two nodes the values of `w13_input_scale` will become nan ⏎ ```bash …[truncated]

### L1-7ea5b42d70  (L1, 2025-11-11, sha 7ea5b42d7063, PR #12666)
TITLE: [sgl-kernel][5/N]Support Expert Specialization Grouped GEMM (#12666)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+76/-43); python/sglang/test/test_cutlass_moe.py (+8/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎ Add the flag parameter to CUTLASS MoE to enable the expert specialization kernel. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - python/sglang/srt/layers/moe/cutlass_moe.py ⏎ - python/sglang/test/test_cutlass_moe.py ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ <img width="1250" height="1842" alt="image" src="https://github.com/user-attachments/assets/fdfaa86d-071b-4c5d-972c-1c7d64efd248" /> ⏎  ⏎ [check.log](https://github.com/user-attachments/files/23350293/check.log) ⏎  ⏎  …[truncated]

### L1-7aa443903d  (L1, 2025-11-12, sha 7aa443903d18, PR #13162)
TITLE: Fix nan in global scaling factor for large scale nvfp4 EP (#13162)
SOURCES: path_core, corpus:kernel-correctness-cases(introducing), corpus:kernel-correctness-cases
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-1); python/sglang/srt/eplb/expert_location.py (+13/-9)
LABELS: run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 13348 (confirmed_revert, reason=crash_or_hang) || deep-study: introduced the defect fixed in case sglang:78a4b446c6 (fix PR 13348) || deep-study correctness case sglang:7aa443903d: class=memory_safety_oob; symptom=nan_inf; introducing=#12866
BODY: ## Motivation ⏎ Part of https://github.com/sgl-project/sglang/pull/12866(reverted in this PR), the root cause is that `w13[2]_input_scale` should not depends on `logical_to_all_physical_map`. In this fix, read in all physical experts' input scale regardless of its logical expert id. ⏎ cc @kaixih @Fridge003  ⏎  ⏎ The root cause is: ⏎ The `w13_input_scale` is of shape [288, ...] but if the map `logical_to_all_physical_map` is : ⏎ ``` ⏎ [ ⏎ [286], ⏎ [287], ⏎  …[truncated]

### L1-a1cb717d0b  (L1, 2025-11-12, sha a1cb717d0b4a, PR #13150)
TITLE: Opt kimi_k2_thinking biased topk module (#13150)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+71/-14)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ### launch server ⏎  ⏎ ```shell ⏎ python -m sglang.launch_server --model-path moonshotai/Kimi-K2-Thinking --tp 8 --trust-remote-code  --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 ⏎ ``` ⏎  ⏎ ### Acc ⏎  ⏎ ```shell ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 ⏎  ⏎ /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-p …[truncated]

### L1-706502ff6c  (L1, 2025-11-12, sha 706502ff6cff, PR #13075)
TITLE: [VLM] Support PP for Qwen2.5-VL (#13075)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/mm_utils.py (+39/-36); python/sglang/srt/managers/scheduler.py (+5/-3); python/sglang/srt/models/qwen2_5_vl.py (+44/-15)
LABELS: feature, performance, Multi-modal, run-ci, vlm
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is to support PP for Qwen2.5-VL model. ⏎  ⏎ ``` ⏎ [root  /root] 二 11月 11 20:23:51  ⏎ $python3 -m sglang.launch_server --model /home/admin/Qwen2.5-VL-7B-Instruct --tp 2 --pp-size=2 ⏎ INFO 11-11 20:29:05 [__init__.py:216] Automatically detected platform cuda. ⏎ [2025-11-11 20:29:05] WARNING server_args.py:1183: Attention backend not explicitly specified. Use flashinfer backend by default. ⏎ [2025-11-11 20:29:05] WARNING server_a …[truncated]

### L1-4edb240112  (L1, 2025-11-13, sha 4edb24011298, PR #12998)
TITLE: Fuse routed_scaling_factor to fused_marlin_moe (#12998)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+1/-11)
LABELS: deepseek, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Fuse routed_scaling_factor to fused_marlin_moe to reduce mul kernel launch.

### L1-6664083522  (L1, 2025-11-13, sha 666408352243, PR #12376)
TITLE: Replace [silu_and_mul_]scaled_fp4_group_quant by Flashinfer equivalent (#12376)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl
FILES: python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py (+8/-8); benchmark/kernels/quantization/bench_fp4_quant.py (+11/-8); docs/references/environment_variables.md (+1/-0); sgl-kernel/tests/test_fp4_quantize.py (+10/-11); test/srt/test_cutedsl_moe.py (+6/-6); test/srt/test_fp4_moe.py (+6/-6)
LABELS: documentation, performance, quant, sgl-kernel, run-ci
BODY: ## Motivation ⏎ Flashinfer Introduced[1927](https://github.com/flashinfer-ai/flashinfer/pull/1927) 2 nvfp4 quantization APIs: ⏎ `silu_and_mul_scaled_nvfp4_experts_quantize` and `scaled_nvfp4_grouped_quantize` which are equivalent in implementation with sglang's `silu_and_mul_scaled_fp4_grouped_quant` and `scaled_fp4_grouped_quant` respectively. This PR made the replacement for future maintenance. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ `` …[truncated]

### L1-aead0ef5e5  (L1, 2025-11-13, sha aead0ef5e553, PR #12201)
TITLE:  [FEAT][ROCM] enable fused shared expert for Rocm (#12201)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+27/-5); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+23/-5); python/sglang/srt/layers/moe/topk.py (+65/-12); python/sglang/srt/layers/quantization/fp8.py (+2/-1); python/sglang/srt/layers/quantization/unquant.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+34/-9)
LABELS: quant, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: # DeepSeek R1: ⏎ ## TP8 ⏎ ``` ⏎ TP=8 ⏎ EP=1 ⏎ python3 -m sglang.launch_server \ ⏎     --model-path ${model} \ ⏎     --host localhost \ ⏎     --port 9000 \ ⏎     --tp-size ${TP} \ ⏎     --ep-size ${EP} \ ⏎     --trust-remote-code \ ⏎     --chunked-prefill-size 196608 \ ⏎     --mem-fraction-static 0.9 \ ⏎     --disable-radix-cache \ ⏎     --num-continuous-decode-steps 4 \ ⏎     --max-prefill-tokens 196608 \ ⏎     --cuda-graph-max-bs 128 ⏎ ``` ⏎ ### acc ⏎ scripts: ⏎ ``` …[truncated]

### L1-67e9d287ee  (L1, 2025-11-13, sha 67e9d287eea0, PR #10485)
TITLE: [Quantization] Support Quark Dense + MoE FP8 & FP8 PTPC (#10485)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/__init__.py (+3/-11); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py (+1/-1); python/sglang/srt/layers/quantization/fp8_utils.py (+116/-220); python/sglang/srt/layers/quantization/quark/quark.py (+42/-1); python/sglang/srt/layers/quantization/quark/quark_moe.py (+296/-5); python/sglang/srt/layers/quantization/quark/schemes/__init__.py (+2/-1); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py (+8/-3); python/sglang/srt/layers/quantization/quark/schemes/quark_w8a8_fp8.py (+186/-0); python/sglang/srt/layers/quantization/quark/utils.py (+11/-1); python/sglang/srt/configs/model_config.py (+1/-0)
LABELS: run-ci
BODY: Includes #10396 and #10289.  ⏎  ⏎ Verified with Qwen3-30b-a3b-thinking-2507 quantized by Quark, running gsm8k. ⏎  ⏎ ``` ⏎ python bench_sglang.py --num-questions 2000 --parallel 2000 --port 8000 ⏎  ⏎ base bf16			0.864 ⏎ quark fp8 ptpc		0.857 ⏎ ```

### L1-0779c3d148  (L1, 2025-11-13, sha 0779c3d14895, PR #13211)
TITLE: docs: update fused MoE config path (#13211)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/README.md (+1/-1)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-bfe638f7e8  (L1, 2025-11-13, sha bfe638f7e87d, PR #13210)
TITLE: Fix broken Markdown formatting in DeepEP documentation (#13210)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: docs/references/environment_variables.md (+1/-0)
LABELS: documentation, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-e7e89349c9  (L1, 2025-11-13, sha e7e89349c901, PR #12543)
TITLE: Enable Flashinfer TRTLLM-GEN-MoE FP8 blockwise kernel for Qwen3-Next on Blackwell (#12543)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-1); python/sglang/srt/layers/moe/utils.py (+20/-1); python/sglang/srt/layers/quantization/fp8.py (+12/-7); python/sglang/srt/models/qwen2_moe.py (+2/-0); test/srt/nightly/test_flashinfer_trtllm_gen_moe_backend.py (+65/-0); test/srt/run_suite.py (+1/-0)
LABELS: blackwell, run-ci, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Dependency ⏎ **Require flashinfer-python >= 0.5.0** ⏎  ⏎ ## Usage ⏎ ```shell ⏎ export SGL_ENABLE_JIT_DEEPGEMM=false ⏎ python3 -m sglang.launch_server --model-path Qwen3-Next/Qwen3-Next-80B-A3B-Instruct-FP8 --tp-size 4 --ep-size 4 --cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 1024 --mem-fraction-static 0.7 --moe-runner-backend flashinfer_trtllm --attention-backend triton --quantization fp8 --mamba-ssm-dtype bfloat16  ⏎ ``` ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ### …[truncated]

### L1-e7b57b0d04  (L1, 2025-11-14, sha e7b57b0d04d0, PR #13113)
TITLE: [BugFix] weight load bug when checkpoint expert.gate and exepert.up_proj are not fused (#13113)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-0); python/sglang/srt/layers/attention/vision.py (+16/-10); python/sglang/srt/models/qwen3_vl_moe.py (+9/-31)
LABELS: run-ci
BODY: ## Motivation ⏎ 1.let ep moe layer to gracefully handle expert_ids that do not belong to local moe rank, in order to handle the case where model checkpoint expert.gate and exepert.up_proj are not fused. ⏎ 2.Update the method of referencing aiter lib in VisionAiterBackend. ⏎ ## Modifications ⏎ 1.Modify qwen_vl_moe and layer to gracefully handle expert_ids that do not belong to local moe rank. ⏎ 2.Modify VisionAiterBackend in VisionAttention ⏎  ⏎ ## Accur …[truncated]

### L1-f8d3d80f63  (L1, 2025-11-14, sha f8d3d80f6374, PR #13242)
TITLE: chore: bump flashinfer v0.5.2 (#13242)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎ As mentioned by @yzh119, flashinfer v0.6.0 will be released soon, so let's first upgrade to the latest version v0.5.2. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-b732ffa404  (L1, 2025-11-15, sha b732ffa404e0, PR #13295)
TITLE: [model-gateway] move python to binding folder (#13295)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: sgl-router/bindings/python/sglang_router/router.py (+0/-0); .github/workflows/nightly-release-router.yml (+8/-6); .github/workflows/pr-test-pd-router.yml (+1/-1); .github/workflows/pr-test-rust.yml (+10/-9); .github/workflows/release-docker-router.yml (+2/-2); .github/workflows/release-pypi-router.yml (+3/-1); sgl-router/.coveragerc (+0/-9); sgl-router/Cargo.toml (+4/-6); sgl-router/LICENSE (+1/-0); sgl-router/MANIFEST.in (+0/-7); (+18 more)
LABELS: documentation, dependencies, run-ci, model-gateway
BODY: we intended to introduce other language bindings including ⏎ 1. go ⏎ 2. node.js ⏎ thus re-organizing bindings to dedicated folder with language name ⏎  ⏎ ## Checklist

### L1-eae59b337e  (L1, 2025-11-15, sha eae59b337e7c, PR #13045)
TITLE: Piecewise Cuda Graph Support for gpt-oss model (#13045)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/compilation/piecewise_context_manager.py (+16/-0); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-3); python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py (+4/-17); python/sglang/srt/models/deepseek_v2.py (+1/-3); python/sglang/srt/server_args.py (+6/-1)
LABELS: deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Support Piecewise cuda graph for gpt-oss series model. ⏎  ⏎ ## Modifications ⏎  ⏎ - MoE backend Select: With piecewise cuda graph, we can achieve similar performance with auto backend compared with triton backend. ⏎ - Adjust the position of `enable_piecewise_cudagraph` to avoid circular import ⏎  ⏎ ## Accuracy Tests ⏎ In benchmark & profilling section ⏎  ⏎ ## Benchmarking and Profiling ⏎ For gsm 8k test: ⏎ - piecewise cuda graph support with …[truncated]

### L1-2a96e302cb  (L1, 2025-11-15, sha 2a96e302cbbb, PR #13314)
TITLE: Revert moe sum reduce for marlin moe (#13314)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:confirmed-reverts, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: sgl-kernel/python/sgl_kernel/fused_moe.py (+3/-9)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study revert record: explicit_rollback of PR(s)  reason=crash_or_hang
BODY: ## Motivation ⏎  ⏎ moe_sum_reduce may have launch issue in some case, reported in #13234

### L1-8e9f05ece1  (L1, 2025-11-15, sha 8e9f05ece168, PR #13322)
TITLE: Update marlin moe kernel interface (#13322)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: sgl-kernel/python/sgl_kernel/__init__.py (+1/-1); sgl-kernel/python/sgl_kernel/fused_moe.py (+54/-0)
LABELS: sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎ Expose `moe_wna16_marlin_gemm` interface, will move the fused_marlin_moe to python side for easier update and avoid too many sgl-kernel releases.

### L1-1d3d42bda0  (L1, 2025-11-15, sha 1d3d42bda0b2, PR #13287)
TITLE: [opt kimi k2 1 / n] Add kimi k2 moe fused gate (#13287)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: sgl-kernel/CMakeLists.txt (+1/-0); sgl-kernel/csrc/common_extension.cc (+6/-0); sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu (+354/-0); sgl-kernel/include/sgl_kernel_ops.h (+8/-0); sgl-kernel/python/sgl_kernel/__init__.py (+1/-0); sgl-kernel/python/sgl_kernel/moe.py (+35/-0); sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py (+117/-0); sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py (+124/-0)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Kimi K2 Acc test ⏎  ⏎ <img width="1013" height="626" alt="图片" src="https://github.com/user-attachments/assets/8671e094-ed3b-432c-9689-a72386076653" /> ⏎  ⏎  ⏎ ### main branch ⏎  ⏎ ```shell ⏎ ➜  sglang git:(add_kimi_k2_moe_fused_gate) ✗ python3 benchmark/gsm8k/bench_sglang.py --num-questions 2000 --parallel 2000 --num-shots 8 ⏎ /usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please inst …[truncated]

### L1-1ca205f6da  (L1, 2025-11-15, sha 1ca205f6da0c, PR #13358)
TITLE: chore: bump sgl-kernel version to 0.3.17.post1 (#13358)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sgl-kernel` version to `0.3.17.post1` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.3.17.post1` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The sgl-kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the kernel version are up …[truncated]

### L1-78a4b446c6  (L1, 2025-11-15, sha 78a4b446c6c6, PR #13348)
TITLE: Fix dpsk-r1-fp4 tp8 by reverting two commits (#13162 and #13341) (#13348)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-4); python/sglang/srt/eplb/expert_location.py (+9/-13); python/sglang/srt/layers/quantization/fp8.py (+1/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-0); python/sglang/srt/layers/quantization/mxfp4.py (+1/-0)
LABELS: high priority, quant, run-ci
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 13162;13341 reason=crash_or_hang || deep-study correctness case sglang:78a4b446c6: class=integration_backend_cudagraph; symptom=hang_deadlock; introducing=#13162/#13341
BODY: ## Motivation ⏎ Before this pr, this command will hang/fail. ⏎  ⏎ ``` ⏎ SGLANG_ENABLE_SPEC_V2=1 /root/.python/sglang/bin/python -m sglang.launch_server --model-path nvidia/DeepSeek-R1-0528-FP4-v2 --trust-remote-code --quantization modelopt_fp4 --tp 8  --speculative-algorithm=EAGLE  --port 40020   --kv-cache-dtype fp8_e4m3   --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' ⏎ ``` ⏎  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accur …[truncated]

### L1-0d116b9a0b  (L1, 2025-11-16, sha 0d116b9a0b0a, PR #13341)
TITLE: Clean up deprecated tile_tokens_dim for next flashinfer (#13341)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-1); python/sglang/srt/layers/quantization/fp8.py (+0/-1); python/sglang/srt/layers/quantization/modelopt_quant.py (+0/-1); python/sglang/srt/layers/quantization/mxfp4.py (+0/-1)
LABELS: quant
DEEP_STUDY: deep-study: this PR was reverted by PR 13348 (confirmed_revert, reason=crash_or_hang)
BODY: ## Motivation ⏎ When flashinfer 0.5.3 out, it will no longer be accepted ⏎ https://github.com/flashinfer-ai/flashinfer/pull/2086 ⏎ It already does nothing, and is passed as `None` (and unused flashinfer side) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist

### L1-d5fa58c4dd  (L1, 2025-11-16, sha d5fa58c4ddf3, PR #13386)
TITLE: fix nightly docker build (#13386)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+5/-3); python/pyproject.toml (+1/-1)
LABELS: dependencies, run-ci
BODY: Keep these aligned

### L1-e970892ffa  (L1, 2025-11-16, sha e970892ffa2e, PR #12874)
TITLE: fix import qwenvl error in RL engine (#12874)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/utils/common.py (+4/-1)
BODY: Solve https://github.com/sgl-project/sglang/issues/12830 ⏎  ⏎ When training qwen3VL RL with sgl.Engine, a `No processor registered for architecture: ['Qwen3VLForConditionalGeneration'] error` is raised. ⏎  ⏎ I tried printing the traceback: ⏎  ⏎ ``` ⏎  Traceback (most recent call last):  ⏎    File "/usr/local/lib/python3.12/dist-packages/sglang/srt/managers/multimodal_processor.py", line 20, in import_processors  ⏎      module = importlib.import_module(nam …[truncated]
