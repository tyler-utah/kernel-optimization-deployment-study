### L2-2fb328109f  (L2, 2026-01-23, sha 2fb328109fb9, PR #16758)
TITLE: [DeepSeek V3.2] Enable trtllm NSA with bf16 kvcache (#16758)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+54/-3); python/sglang/srt/server_args.py (+64/-28)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Enables NSA backend with trtllm kernels for sparse attention. This can be more efficient than FlashMLA when the head size isn't a multiple of 64 and hence requires padding. This PR enables with BF16 KVCache, FP8 will follow in another PR. ⏎  ⏎ ## Modifications ⏎  ⏎ This change interfaces with the new kernel added in flashinfer to use trtllm kernel for decode in NSA backend. There are also modifications made to `server_args.py` to ena …[truncated]

### L2-894928a951  (L2, 2026-01-24, sha 894928a95124, PR #16969)
TITLE: Refactor: Extract DeepSeek common utilities into shared module (#16969)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/utils.py (+41/-1); python/sglang/srt/models/deepseek_v2.py (+4/-21)
LABELS: documentation, quant, amd, dependencies, lora, Multi-modal, deepseek, hicache, blackwell, npu
BODY: ## Motivation ⏎  ⏎  ⏎ DeepseekV2 code implementation has grown quickly, making it difficult to maintain. This PR refactors these utilities into a dedicated, well-documented module with comprehensive test coverage. ⏎ Issue related: [#16701](https://github.com/sgl-project/sglang/issues/16701) ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ **Refactored utility functions into shared module:** ⏎ - Moved 4 utility functions from `deepseek_v2.py` to `deepseek_common/utils.py`: …[truncated]

### L2-1674b9ef44  (L2, 2026-01-25, sha 1674b9ef4494, PR #17662)
TITLE: [DeepSeek-V3.2] Fix TRT-LLM NSA in target_verify/draft_extend (#17662)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+18/-1)
BODY: ## Motivation ⏎  ⏎ Quick follow‑up to #16758: speculative decoding still fails with `--nsa-decode-backend trtllm`. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add TRT‑LLM handling in the extend path for speculative modes and guard it with a forward‑mode assertion (`target_verify`/`draft_extend` only). ⏎ - Require explicit `seq_lens` for `_forward_trtllm`, passing expanded `nsa_cache_seqlens_int32` for speculative paths and `cache_seqlens_int32` in normal decode. ⏎  ⏎ ## …[truncated]

### L2-f6f1b6d000  (L2, 2026-01-26, sha f6f1b6d000b6, PR #17700)
TITLE: Bump FI version (#17700)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎ This PR bumps FlashInfer version, to incorporate latest fix to Mamba's `selective_scan_update` kernel ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other re …[truncated]

### L2-5844cb2fd8  (L2, 2026-01-26, sha 5844cb2fd82c, PR #17645)
TITLE: refactor mamba radix cache logic in server_args (#17645)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+91/-116)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎ 3. Trigge …[truncated]

### L2-b56366f827  (L2, 2026-01-26, sha b56366f8275a, PR #15381)
TITLE: [NPU]DeepSeek-V3.2 support npu mlaprolog (#15381)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.npu_mla, L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py (+63/-1); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+3/-3); python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py (+116/-55); python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py (+8/-2); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+5/-0)
LABELS: quant, deepseek, npu, run-ci
BODY: ## Motivation ⏎  ⏎ we enabled the DSV3.2 to support npu mla_prolog_v3 to improve the performance ⏎  ⏎ ## Modifications ⏎  ⏎ - Added support for the mla_prolog_v3 custom operator, which includes operations such as Q/KV LoRA, RoPE, Norm, and KVCache update, and contains significant Cube/Vector parallelism space. ⏎ - Extracted the calls of the MLAPO operator and the MLA-PROLOG operator into the function `npu_mla_preprocess()`. ⏎ - Adjusted the weight of the …[truncated]

### L2-81c0f5c5ad  (L2, 2026-01-27, sha 81c0f5c5adf0, PR #8205)
TITLE: [Model] Add support for EXAONE-4.0 Model (#8205)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/exaone4.py (+719/-0); python/sglang/srt/server_args.py (+9/-0)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎  ⏎ Adding support for [LGAI](https://github.com/LG-AI-EXAONE)'s [EXAONE-4.0](https://github.com/LG-AI-EXAONE/EXAONE-4.0)  ⏎ The sliding window code was implemented by referencing [gemma2](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/gemma2.py) ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ Add exaone4.py ⏎  ⏎  ⏎ To use the model for reasoning, use the `--reasoning-parser deepseek-r1` argument. ⏎  ⏎ **Example:** ⏎ ``` ⏎ python -m sglan …[truncated]

### L2-d578b41bad  (L2, 2026-01-27, sha d578b41badc8, PR #17615)
TITLE: [NPU] Adapt cann 8.5: use sfa and lightning indexer op from cann and CI update (#17615)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: docker/npu.Dockerfile (+13/-23); .github/workflows/nightly-test-npu.yml (+4/-8); .github/workflows/pr-test-npu.yml (+4/-8); .github/workflows/release-docker-npu-nightly.yml (+2/-2); .github/workflows/release-docker-npu.yml (+2/-2); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+9/-3); python/sglang/srt/hardware_backend/npu/utils.py (+0/-8); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+7/-8); scripts/ci/npu/npu_ci_install_dependency.sh (+11/-22)
LABELS: npu, run-ci
BODY: ## Motivation ⏎ There are some bugs of sparse flash attention and lightning indexer in custom_ops package. Fixed ops are intergrated in CANN 8.5 so we update these ops and call them using torch_npu. ⏎  ⏎ lightning_indexer: https://gitcode.com/cann/ops-transformer/blob/master/attention/lightning_indexer/README.md ⏎ sfa: https://gitcode.com/cann/ops-transformer/blob/master/attention/sparse_flash_attention/README.md ⏎  ⏎ Upgrading the CI  to the correspon …[truncated]

### L2-8acd4d7d7e  (L2, 2026-01-28, sha 8acd4d7d7e6f, PR #17600)
TITLE: Make flashMLA work on: Cu13, B300 (#17600)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: sgl-kernel/cmake/flashmla.cmake (+53/-0)
LABELS: sgl-kernel, run-ci
BODY: Here are the steps to **build local.** ⏎  ⏎ Also, the cuda packaged cu13 wheel will now work. ⏎  ⏎ Steps: ⏎  ⏎ `sudo ln -s /usr/local/cuda/include/cccl/cuda ⏎ /usr/local/cuda/include/cuda` (it moved). ⏎  ⏎ This is validated in sglang docker. ⏎  ⏎ Important: when .venv/ is present in sglang dir, it breaks compilation (even if not activated) ⏎  ⏎ `nohup bash -c 'CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") make build' > bu …[truncated]

### L2-e9d727cb92  (L2, 2026-01-28, sha e9d727cb9218, PR #17499)
TITLE: [MUSA][7/N] Enhance CUDA / PyNccl wrapper to support MTLink connectivity detection (#17499)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/pyproject_other.toml (+1/-1); python/sglang/srt/distributed/device_communicators/cuda_wrapper.py (+5/-1); python/sglang/srt/distributed/device_communicators/custom_all_reduce.py (+15/-3); python/sglang/srt/distributed/device_communicators/custom_all_reduce_ops.py (+4/-3); python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py (+8/-1); python/sglang/srt/distributed/device_communicators/pynccl_wrapper.py (+6/-4)
LABELS: documentation, dependencies, mthreads
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is the seventh in a series of pull requests (tracked in https://github.com/sgl-project/sglang/issues/16565) to add full support for [Moore Threads](https://en.mthreads.com/) GPUs, leveraging MUSA (Meta-computing Unified System Architecture) to accelerate LLM inference. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ This commit updates the runtime wrappers and communication utilities to improve MUSA backend compatibility across SGLang’s distr …[truncated]

### L2-d3cdee0a04  (L2, 2026-01-28, sha d3cdee0a040b, PR #17246)
TITLE: [MUSA][4/N] Add common device utilities, distributed backend, and custom op wiring (#17246)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/multimodal_gen/runtime/layers/custom_op.py (+8/-0); python/sglang/srt/configs/device_config.py (+1/-1); python/sglang/srt/distributed/parallel_state.py (+6/-0); python/sglang/srt/layers/rotary_embedding.py (+3/-0); python/sglang/srt/layers/utils/multi_platform.py (+10/-0); python/sglang/srt/model_executor/model_runner.py (+4/-2); python/sglang/srt/utils/common.py (+77/-8)
LABELS: documentation, dependencies, run-ci, diffusion, mthreads
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is the fourth in a series of pull requests (tracked in https://github.com/sgl-project/sglang/issues/16565) to add full support for [Moore Threads](https://en.mthreads.com/) GPUs, leveraging MUSA (Meta-computing Unified System Architecture) to accelerate LLM inference. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ - **`python/sglang/multimodal_gen/runtime/layers/custom_op.py`**: Add `forward_musa()` method and MUSA platform dispatch in `Cust …[truncated]

### L2-84ab611af8  (L2, 2026-01-30, sha 84ab611af8b7, PR #17897)
TITLE: model: support DeepSeek-OCR-2 (#17897)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/basic_usage/deepseek_ocr.md (+54/-0); docs/index.rst (+1/-0); docs/supported_models/multimodal_language_models.md (+1/-0); python/sglang/srt/configs/deepseek_ocr.py (+32/-9); python/sglang/srt/configs/model_config.py (+9/-4); python/sglang/srt/model_loader/utils.py (+6/-2); python/sglang/srt/models/deepseek_ocr.py (+446/-116); python/sglang/srt/multimodal/processors/deepseek_ocr.py (+8/-0); python/sglang/srt/utils/hf_transformers_utils.py (+61/-9)
LABELS: documentation, Multi-modal, deepseek, run-ci
BODY: ## Motivation ⏎ #17833 ⏎ Config mismatch: OCR2 models were detected as DeepseekVL2Config lacking text_config, which broke model init. We explicitly load/override to the DeepSeek OCR config to ensure the correct model class and fields. ⏎ MLA vs non‑MLA attention: OCR2’s language config uses non‑MLA, which caused MLA‑only code paths to crash (e.g., ZeroDivision). We route OCR2 to DeepseekForCausalLM (non‑MLA) to avoid MLA‑specific assumptions. ⏎ Vision …[truncated]

### L2-c04efe030a  (L2, 2026-01-30, sha c04efe030acc, PR #16294)
TITLE: [Model] Add K-EXAONE model support (#16294)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/configs/model_config.py (+4/-0); python/sglang/srt/models/exaone_moe.py (+881/-0); python/sglang/srt/models/exaone_moe_mtp.py (+106/-0); python/sglang/srt/server_args.py (+9/-7)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR integrates the [K-EXAONE-236B-A23B](https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B) model, recently released by [LG AI Research](https://huggingface.co/LGAI-EXAONE). Our model's github link is here: [K-EXAONE github](https://github.com/LG-AI-EXAONE/K-EXAONE). ⏎  ⏎ - Goal: To provide native support for K-EXAONE within SGLang's high-performance inference framework. ⏎  ⏎ - Details: Implementation includes the EXAONE-M …[truncated]

### L2-ee3058c6e8  (L2, 2026-01-31, sha ee3058c6e867, PR #18017)
TITLE: [NPU] fix sgl-kernel-npu package url error in npu.Dockerfile (#18017)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+2/-2)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ npu nightly docker image build failed for url error ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ change sgl-kernel-npu url ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ NA ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎ NA ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://git …[truncated]

### L2-d0d9cecd1b  (L2, 2026-01-31, sha d0d9cecd1b34, PR #17766)
TITLE: Fix cuBLAS >=12.9 detection for cu12/cu13 package naming (#17766)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_common/utils.py (+2/-2); python/sglang/srt/utils/common.py (+6/-7)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ cu13 wheels ship cuBLAS as `nvidia-cublas` (not `nvidia-cublas-cu13`), so the cuBLAS 12.9+ gate can fail to detect newer installs. This updates the version check to work across cu12/cu13 packaging. ⏎  ⏎ ## Modifications ⏎  ⏎ - Rename the helper to a cu-version‑agnostic name. ⏎ - Check both `nvidia-cublas` and `nvidia-cublas-cu12` via the shared version helper and update call sites. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ `python3 test/registered/mla/ …[truncated]

### L2-9b1619c148  (L2, 2026-02-03, sha 9b1619c148a0, PR #17889)
TITLE: [Move sgl-kernel Kernel to JIT] Add JIT concat MLA kernels (#17889)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.kernel.concat_mla
FILES: python/sglang/jit_kernel/concat_mla.py (+65/-0); python/sglang/jit_kernel/csrc/elementwise/concat_mla.cuh (+325/-0); python/sglang/jit_kernel/benchmark/bench_concat_mla.py (+163/-0); python/sglang/jit_kernel/tests/test_concat_mla.py (+169/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Add JIT-compiled CUDA kernels for MLA tensor concatenation: ⏎ - concat_mla_k ⏎ - concat_mla_absorb_q ⏎  ⏎ ## Modifications ⏎  ⏎ - python/sglang/jit_kernel/concat_mla.py: Python interface ⏎ - python/sglang/jit_kernel/csrc/elementwise/concat_mla.cuh ⏎ - python/sglang/jit_kernel/tests/test_concat_mla.py: Unit tests ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Verified against PyTorch implementation and AOT sgl_kernel. ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ # …[truncated]

### L2-031a652b93  (L2, 2026-02-08, sha 031a652b936c, PR #18396)
TITLE: Fix TRT-LLM MLA backend applying k_scale to BF16 KV cache in BMM1 (#18396)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+36/-12)
LABELS: blackwell, run-ci
BODY: When a model checkpoint contains KV cache scaling factors (k_scale/v_scale) but the KV cache dtype is BF16 (not FP8), the TRT-LLM MLA backend unconditionally applies k_scale in the BMM1 attention score computation. This is incorrect because k_scale is a quantization compensation factor that should only be applied when KV cache values are actually FP8-quantized. ⏎  ⏎ For example, with k_scale=0.06, attention scores are scaled down by ~16x, producing …[truncated]

### L2-bec7fe9e65  (L2, 2026-02-10, sha bec7fe9e6523, PR #18362)
TITLE: [sgl-kernel] upgrade deepgemm (#18362)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.concat_mla
FILES: sgl-kernel/csrc/elementwise/concat_mla.cu (+1/-0); sgl-kernel/CMakeLists.txt (+14/-3); sgl-kernel/build.sh (+2/-0)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 18562 (confirmed_revert, reason=unstated)
BODY: ## Motivation ⏎  ⏎ Fix DeepGEMM compilation error after updating to commit `9b680f42`: remove `USE_SABI` from the `deep_gemm_cpp` target's `Python_add_library` call. The new DeepGEMM commit uses pybind11 features that require the full CPython API (e.g., `PyTuple_SET_ITEM`, `Py_buffer`, `PyTypeObject` internals), which are unavailable when Python Stable ABI (`Py_LIMITED_API`) is enabled. Dropping `USE_SABI` allows pybind11 to access the complete CPy …[truncated]

### L2-2d38b8aca0  (L2, 2026-02-11, sha 2d38b8aca016, PR #18562)
TITLE: Revert "[sgl-kernel] upgrade deepgemm" (#18562)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.concat_mla
FILES: sgl-kernel/csrc/elementwise/concat_mla.cu (+0/-1); sgl-kernel/CMakeLists.txt (+3/-14); sgl-kernel/build.sh (+0/-2)
LABELS: sgl-kernel
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 18362 reason=unstated
BODY: Reverts sgl-project/sglang#18362 ⏎ Also fix the commit to the latest one on DeepGemm's `sgl-release` branch

### L2-20554a0a4f  (L2, 2026-02-11, sha 20554a0a4fb6, PR #17799)
TITLE: [AMD] rocm 7.2 image release, PR test, Nightly Test (#17799)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docker/rocm720.Dockerfile (+502/-0); .github/workflows/nightly-test-amd-rocm720.yml (+868/-0); .github/workflows/pr-test-amd-rocm720.yml (+793/-0); .github/workflows/pr-test-amd.yml (+20/-2); .github/workflows/release-docker-amd-rocm720-nightly-preview.yml (+82/-0); python/sglang/srt/layers/layernorm.py (+14/-1); python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+43/-12); python/sglang/srt/layers/moe/moe_runner/triton.py (+16/-2); python/sglang/srt/layers/quantization/fp8_kernel.py (+45/-4); python/sglang/srt/layers/quantization/unquant.py (+8/-2); (+16 more)
LABELS: quant, amd, Multi-modal, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Enable Rocm 7.2  ⏎  ⏎ ## Modifications ⏎  ⏎ ROCM 7.2 ⏎ - PR Test ⏎ - Nightly Test ⏎ - cleanup vllm dependencies ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Nightly All green : https://github.com/sgl-project/sglang/actions/runs/21875099676 ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pu …[truncated]

### L2-9e9e949261  (L2, 2026-02-12, sha 9e9e94926162, PR #18586)
TITLE: speed up sgl-kernel build (#18586)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.concat_mla
FILES: sgl-kernel/csrc/elementwise/concat_mla.cu (+1/-0); .github/workflows/release-whl-kernel.yml (+4/-2); sgl-kernel/Dockerfile (+12/-2); sgl-kernel/build.sh (+77/-4)
LABELS: high priority, sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎ <img width="2280" height="1032" alt="e2b78bafeaf4f7e8e38ae375af65b70b" src="https://github.com/user-attachments/assets/83eaf7ef-dc5a-4362-bea8-8eb500809a4a" /> ⏎  ⏎ 1h10min->7min. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/M …[truncated]

### L2-d97eb111a3  (L2, 2026-02-13, sha d97eb111a368, PR #18598)
TITLE: Support LingV2_5 model (#18598)
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+3/-0); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/bailing_hybrid.py (+188/-0); python/sglang/srt/configs/model_config.py (+20/-0); python/sglang/srt/layers/attention/fla/layernorm_gated.py (+26/-5); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+369/-0); python/sglang/srt/layers/attention/linear/lightning_attn.py (+767/-0); python/sglang/srt/layers/attention/linear/linear_metadata.py (+70/-0); python/sglang/srt/layers/attention/linear/seg_la.py (+909/-0); python/sglang/srt/model_executor/model_runner.py (+14/-1); (+6 more)
LABELS: high priority, run-ci
BODY: ## Motivation ⏎ This is used for support Bailing models, Ling-2.5 and Ring-2.5 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang …[truncated]

### L2-1be41e9036  (L2, 2026-02-14, sha 1be41e9036e1, PR #18448)
TITLE: [FlashInfer] Bump FlashInfer version from 0.6.2 to 0.6.3 (#18448)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); .github/workflows/release-docker-cu13-framework.yml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/server_args.py (+2/-2); python/sglang/srt/utils/common.py (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎ 3. Trigge …[truncated]

### L2-34132d6da5  (L2, 2026-02-14, sha 34132d6da50e, PR #17554)
TITLE: Kernel: optimize decoding metadata in NSA multi-spec backend with fused kernels (#17554)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/jit_kernel/csrc/elementwise/fused_metadata_copy.cuh (+722/-0); python/sglang/jit_kernel/fused_metadata_copy.py (+316/-0); python/sglang/jit_kernel/tests/test_fused_metadata_copy.py (+1067/-0); python/sglang/srt/environ.py (+2/-0); python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py (+3/-3); python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py (+407/-0); python/sglang/srt/layers/attention/nsa_backend.py (+307/-51)
LABELS: documentation, quant, amd, dependencies, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎ Implement fused CUDA kernels to eliminate redundant metadata copies in Native Sparse Attention (NSA) backend during CUDA graph replay for speculative decoding. This optimization provides 3-5x speedup for multi-backend metadata operations.                                                               ⏎ ## Changes                                                                                       ⏎                                     …[truncated]

### L2-1ce3420784  (L2, 2026-02-15, sha 1ce3420784c2, PR #18040)
TITLE: Model: Support IBM Granite (Dense/Mamba + MoE)  (#18040)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs/supported_models/text_generation/generative_models.md (+1/-0); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/granitemoehybrid.py (+301/-0); python/sglang/srt/model_executor/model_runner.py (+12/-0); python/sglang/srt/models/granitemoehybrid.py (+737/-0); python/sglang/srt/server_args.py (+13/-0); python/sglang/srt/utils/hf_transformers_utils.py (+2/-0); test/registered/models/test_generation_models.py (+4/-0)
LABELS: documentation, run-ci
BODY: ## Motivation ⏎  ⏎ Add Support for [ibm-granite/granite-4.0-h-micro](https://huggingface.co/ibm-granite/granite-4.0-h-micro) and its [Dense variant](https://huggingface.co/ibm-granite/granite-4.0-micro) ⏎  ⏎ When I tried to run this model, i got the message: ⏎  ⏎ ```bash ⏎ ❯ python3 -m sglang.bench_one_batch --correct --model ibm-granite/granite-4.0-h-micro ⏎  ⏎ ... ⏎ ... ⏎  ⏎ [rank0]:   File "/run/media/blazingbhavneek/Common/Code/sglang/python/sglang/srt/m …[truncated]

### L2-b992828ad2  (L2, 2026-02-15, sha b992828ad272, PR #18604)
TITLE: fix: fix bug on kimi2.5 with dp2 and tp4 (#18604)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+1/-1)
LABELS: blackwell, run-ci
ISSUES: #18410 [Bug] Kimi 2.5 has a chance of crashing under TP4 DP2 configuration.
BODY: ## Motivation ⏎ This bug fix was a joint effort by me and my colleague @engineer1109 . Thanks to @engineer1109 for the collaboration! ⏎  ⏎ fix https://github.com/sgl-project/sglang/issues/18410 ⏎  ⏎ ## Modifications ⏎  ⏎ torch.empty to torch.zeros, so nan will not appear for pad tokens ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Proce …[truncated]

### L2-4e162d4b1b  (L2, 2026-02-15, sha 4e162d4b1bf5, PR #18835)
TITLE: change npu.dockerfile (#18835)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+3/-3); .github/workflows/release-docker-npu-nightly.yml (+1/-1)
LABELS: npu
BODY: ## Motivation ⏎  ⏎ change npu.dockerfile ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other  …[truncated]

### L2-0ffd0a3995  (L2, 2026-02-16, sha 0ffd0a3995e5, PR #18389)
TITLE: Nsa trtllm mla sparse fp8 support with Deepseek v3.2 NVFP4 (#18389)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.backend.sparse_mla_adapters, L2.pool.mla_token_kv, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+172/-66); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+13/-97); python/sglang/srt/layers/attention/utils.py (+99/-0); python/sglang/srt/mem_cache/memory_pool.py (+14/-18); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+40/-0); python/sglang/srt/models/deepseek_v2.py (+6/-0); docs/advanced_features/server_arguments.md (+2/-2); docs/basic_usage/deepseek_v32.md (+4/-0); test/registered/hicache/test_nsa_pool_host_unit.py (+1/-0); test/registered/kernels/test_nsa_indexer.py (+1/-0)
LABELS: documentation, high priority, deepseek, blackwell, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ #17655  ⏎  ⏎ - support Deepseek v3.2 NVFP4 with trtllm mla sparse fp8 attention backend ⏎  ⏎ ## Modifications ⏎  ⏎ - update the nsa backend to support trtllm sparse fp8 attention backend ⏎ - update the deepseek v2 to make sure the cos_sin_cache pass to trtllm kernels ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ### GSM8K ⏎ ```bash ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 200 --parallel 100 --port 30000 ⏎ 100%|████████████████████ …[truncated]

### L2-fbb6098487  (L2, 2026-02-20, sha fbb60984872e, PR #17953)
TITLE: [AMD] support two batch overlapping for mori ep (#17953)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.aiter_mla, L2.dispatch.server_args_defaults
FILES: docs/advanced_features/server_arguments.md (+1/-1); python/sglang/srt/batch_overlap/operations_strategy.py (+9/-2); python/sglang/srt/batch_overlap/two_batch_overlap.py (+5/-0); python/sglang/srt/layers/attention/aiter_backend.py (+162/-52); python/sglang/srt/layers/moe/ep_moe/layer.py (+51/-23); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-14); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+4/-0); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+448/-47); python/sglang/srt/models/deepseek_v2.py (+13/-1); python/sglang/srt/server_args.py (+7/-5); (+1 more)
LABELS: documentation, deepseek, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 19161 (confirmed_revert, reason=ci_or_test_failure) || deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ co-author with @kkHuang-amd @ZhaiFeiyue @Duyi-Wang   ⏎ cc @HaiShaw ⏎  ⏎ This patch is to support TBO aka two batch overlapping feature for mori ep. It can be divided into the following changes:  ⏎ (1) We introduce MORI async API to support CU-free method for low latency scenario. ⏎ (2) We introduce multi hip stream to enable communication-computation overlapping for high throughput scenario. ⏎ (3) The relation between sglang arguments an …[truncated]

### L2-43f83525c0  (L2, 2026-02-23, sha 43f83525c0a2, PR #19161)
TITLE: Revert "[AMD] support two batch overlapping for mori ep #17953" (#19161)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.aiter_mla, L2.dispatch.server_args_defaults
FILES: docs/advanced_features/server_arguments.md (+1/-1); python/sglang/srt/batch_overlap/operations_strategy.py (+2/-9); python/sglang/srt/batch_overlap/two_batch_overlap.py (+0/-5); python/sglang/srt/layers/attention/aiter_backend.py (+52/-162); python/sglang/srt/layers/moe/ep_moe/layer.py (+23/-51); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+14/-6); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+0/-4); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+47/-448); python/sglang/srt/models/deepseek_v2.py (+1/-13); python/sglang/srt/server_args.py (+5/-7); (+1 more)
LABELS: documentation, deepseek
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 17953 reason=ci_or_test_failure
BODY: ## Motivation ⏎  ⏎ Fix broken CI https://github.com/sgl-project/sglang/actions/runs/22256775640/job/64445687327 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](htt …[truncated]

### L2-d80c884a27  (L2, 2026-02-23, sha d80c884a279b, PR #18985)
TITLE: Use single mma warp group for short q_len in FA to optimize decoding performance (#18985)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+5/-5)
LABELS: high priority, sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: https://github.com/sgl-project/sgl-flash-attn/pull/34 ⏎  ⏎  ⏎ ## accuracy: ⏎  ⏎ [{'eval_name': 'gpqa', 'model_name': 'dummy-medium_temp1.0_20260218_210226', 'metric': 0.7051767676767676}] ⏎  ⏎  ⏎ ## Benchmarking ⏎ input len 32000, output len 2000 ⏎ ### bs1: ⏎ - before: 264.20 ⏎ - after: 273.64 ⏎ ### bs16: ⏎ - before: 2087.19 ⏎ - after: 2421.24 ⏎  ⏎ ### bs32:  ⏎ - before: 2951.04 ⏎ - after: 3486.82 ⏎  ⏎ ### bs64 ⏎ - before: 3801.22 ⏎ - after: 4907.68 ⏎  ⏎ ## profiling ⏎ bs …[truncated]

### L2-c193a52fa2  (L2, 2026-02-24, sha c193a52fa263, PR #18624)
TITLE: [AMD] DSR1/V3 use fp8 bmm in MLA for MI300X (#18624)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+13/-9)
LABELS: amd, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Improve TPOT by using use fp8 bmm in MLA on MI300X for DSR1/V3 ⏎ ## Modifications ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ ```shell ⏎ python3 benchmark/gsm8k/bench_sglang.py --host "http://127.0.0.1" --port 8000 --num-shots 8 --num-questions 1319 --parallel 1319  ⏎  ⏎ 100%|██████████| 1319/1319 [00:40<00:00, 32.27it/s]  ⏎ Accuracy: 0.952 ⏎ Invalid: 0.000 ⏎ Latency: 41.090 s ⏎ Output throughput: 3329.665 token/s ⏎ ``` ⏎ ## Benchmarking and Profiling ⏎  …[truncated]

### L2-b9cf1563de  (L2, 2026-02-24, sha b9cf1563de97, PR #18242)
TITLE: [ROCm] Optimize Deepseek R1 on MI300X (#18242)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+1/-0); python/sglang/srt/layers/quantization/fp8_utils.py (+1/-0); python/sglang/srt/models/deepseek_v2.py (+5/-2)
LABELS: amd, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ The DeepSeek-V3/R1 model on AMD MI300X (gfx942) was previously limited to fallback code paths for several MLA attention operations because the optimized kernels were gated behind `_use_aiter_gfx95` (gfx950-only). This left significant decode throughput on the table, as gfx942 could not leverage: ⏎  ⏎ 1. The fused RoPE + concatenation + KV cache write Triton kernel (`fused_qk_rope_cat_and_cache_mla`), falling back to separate `rotar …[truncated]

### L2-31c7dc9d99  (L2, 2026-02-24, sha 31c7dc9d990d, PR #19003)
TITLE: [VLM] Introduce FlashInfer CUDNN Prefill as ViT Backend (#19003)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/vision.py (+152/-0); python/sglang/srt/models/qwen3_vl.py (+259/-13); python/sglang/srt/server_args.py (+9/-1); test/manual/nightly/test_vlms_vit_flashinfer_cudnn.py (+258/-0)
LABELS: performance, Multi-modal, run-ci, vlm, flashinfer
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎  ⏎ FlashInfer CUDNN Prefill demonstrates strong performance. This PR is to introduce it to SGLang as one of VLM ViT attention backends. A new "flashinfer" mm attention backend is added. This PR supports Qwen3-VL. In the next PRs we will adapt more VLMs to support this backend. ⏎  ⏎ **The performance improved 11.6% vs FA3. (TTFT reduce)**  ⏎ 1054ms vs 931ms. ⏎  ⏎ ``` ⏎ Server: ⏎ ➜  sglang_dev2 git:(support_vit_fi_backend) ✗ CUDA_VISIBLE_D …[truncated]

### L2-e138f7960a  (L2, 2026-02-24, sha e138f7960ac9, PR #19247)
TITLE: [AMD] Fix accuracy while using --enable-dp-attention (#19247)
SOURCES: corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+4/-1)
LABELS: amd, run-ci
DEEP_STUDY: deep-study correctness case sglang:e138f7960a: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Motivation ⏎ When --enable-dp-attention is used with --tp-size 8 --dp-size 8, the attention TP size becomes 1 (i.e., 8 // 8), so each GPU sees all 128 heads instead of 16, causing the persistent MLA decode kernel to be incorrectly selected for a head count it does not support, leading to accuracy issues. In comments it mentioned persistent-kernel doesn't support head-size=128, but doesn't put include this condition. ⏎  ⏎ ## Modifications ⏎ Updated …[truncated]

### L2-60eeef7370  (L2, 2026-02-25, sha 60eeef73701a, PR #19216)
TITLE: [AMD][with CI Fix] support two batch overlapping for mori ep (#19216)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.aiter_mla, L2.dispatch.server_args_defaults
FILES: docs/advanced_features/server_arguments.md (+1/-1); python/sglang/srt/batch_overlap/operations.py (+1/-1); python/sglang/srt/batch_overlap/operations_strategy.py (+9/-2); python/sglang/srt/batch_overlap/two_batch_overlap.py (+5/-0); python/sglang/srt/layers/attention/aiter_backend.py (+172/-59); python/sglang/srt/layers/moe/ep_moe/layer.py (+51/-23); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-14); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+4/-0); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+448/-47); python/sglang/srt/models/deepseek_v2.py (+14/-1); (+2 more)
LABELS: documentation, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: This PR is new version of old PR https://github.com/sgl-project/sglang/pull/17953 ⏎ and with the commit https://github.com/sgl-project/sglang/pull/19216/changes/08880e6f87c5ae0e063630697e5ddbe974eaa155 to address deepep CI failure: ⏎ https://github.com/sgl-project/sglang/actions/runs/22256775640/job/64445687327 ⏎ https://github.com/sgl-project/sglang/actions/runs/22256775640/job/64445687329 ⏎  ⏎ cc @HaiShaw @Fridge003  ⏎  ⏎ ## Motivation ⏎ co-author with …[truncated]

### L2-b2c46fc60b  (L2, 2026-02-25, sha b2c46fc60b14, PR #18355)
TITLE: [AMD] Support Qwen3-Coder-Next on AMD platform (#18355)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+211/-72); python/sglang/srt/models/qwen3_next.py (+2/-2)
LABELS: amd, run-ci
BODY: ## Motivation ⏎ Enable Qwen3-Coder-Next model on AMD GPU platform. With this PR, we are able to support non-MTP (fp8 kv cache) and MTP on Qwen3-Coder-Next. ⏎  ⏎ ## Modifications ⏎  ⏎ - aiter_backend.py:  ⏎     - Handle v_head_dim correctly for MLA and hybrid linear models. Previously, v_head_dim was retrieved directly from token_to_kv_pool.get_value_buffer(0), which fails for models where layer 0 may not be a full attention layer. Now properly handles  …[truncated]

### L2-f75abb4521  (L2, 2026-02-25, sha f75abb4521cf, PR #19086)
TITLE: [Fix][Qwen3.5] Fix KV cache slice transfer for GQA models with replicated KV heads (#19086)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/base/conn.py (+1/-0); python/sglang/srt/disaggregation/mooncake/conn.py (+15/-5); python/sglang/srt/disaggregation/prefill.py (+3/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Merge with https://github.com/sgl-project/sglang/pull/19076 ⏎  ⏎ PD disaggregation with heterogeneous `attn_tp_size` (e.g., prefill TP4 with decode TP4/DP4/dp_attention where decode `attn_tp=1`) produces completely wrong outputs for non-MLA GQA models. This affects models like Qwen3.5-397B-A17B-FP8 which has only 2 KV heads — fewer than `tp_size`. ⏎  ⏎ With GQA replication (`num_kv_heads < tp_size`), per-rank `kv_head_num = max(1, to …[truncated]

### L2-c7c4a1cbbd  (L2, 2026-02-25, sha c7c4a1cbbd6c, PR #18622)
TITLE: refactor linear attention backend (#18622)
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+10/-4); python/sglang/srt/environ.py (+0/-2); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+3/-837); python/sglang/srt/layers/attention/linear/__init__.py (+0/-0); python/sglang/srt/layers/attention/linear/gdn_backend.py (+379/-0); python/sglang/srt/layers/attention/linear/kda_backend.py (+285/-0); python/sglang/srt/layers/attention/linear/kernels/__init__.py (+0/-0); python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl.py (+47/-0); python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py (+131/-0); python/sglang/srt/layers/attention/linear/kernels/kda_triton.py (+73/-0); (+4 more)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ main: commit 947927bdb55ae45469be7ea0e44541a940c78ec3 ⏎  ⏎ ``` ⏎ python -m sglang.launch_server --model-path moonshotai/Kimi-Linear-48B-A3B-Instruct --trust-remote-code ⏎  ⏎ python benchmark/mmlu/bench_sglang.py  ⏎  ⏎ Total latency: 235.497 ⏎ Average accuracy: 0.767 ⏎ ``` ⏎  ⏎ ``` ⏎ python -m sglang.launch_server --model-path Qwen/Qwen3-Next-80B-A3B-Instruct --tp 2  ⏎  ⏎ python benchmark/mmlu/bench_sglang.py  ⏎ Total latency: 101.721 ⏎ Average a …[truncated]

### L2-2ad475b4ed  (L2, 2026-02-26, sha 2ad475b4edfe, PR #18696)
TITLE: use flashinfer.sampling (#18696)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+0/-1); python/sglang/srt/layers/sampler.py (+4/-3); sgl-kernel/benchmark/bench_top_k_top_p_sampling.py (+3/-2); sgl-kernel/csrc/common_extension.cc (+0/-15); sgl-kernel/include/sgl_kernel_ops.h (+0/-29); sgl-kernel/python/sgl_kernel/__init__.py (+0/-4); sgl-kernel/python/sgl_kernel/sampling.py (+0/-371); sgl-kernel/tests/test_sampling.py (+7/-6)
LABELS: sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/issues/17865 ⏎ move (external) flashinfer/csrc/sampling.cu ⏎ Call flashinfer directly from python, instead of compiling the operators into sgl_kernel ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ unittest ```python -m pytest sgl-kernel/tests/test_sampling.py -s``` ⏎  ⏎ gsm8k ⏎ ``` ⏎ python -m sglang.launch_server --model /data/Qwen3-8B/ ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 - …[truncated]

### L2-4e843f1216  (L2, 2026-02-26, sha 4e843f121657, PR #19148)
TITLE: [DeepSeek-V3.2][JIT-kernel] Support nsa fuse store indexer k cache (#19148)
SOURCES: release_notes
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh (+124/-0); python/sglang/jit_kernel/fused_store_index_cache.py (+103/-0); python/sglang/jit_kernel/utils.py (+1/-0); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+79/-21)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎  ⏎ In DeepSeek v3.2, after the Indexer produces key in bf16 (roughly (N, 128)), it needs to populate NSA’s index_k_with_scale_buffer. The previous implementation used two steps: ⏎  ⏎ 1. Quantization: act_quant(key, ...) converts bf16 keys into k_fp8: FP8(E4M3) key bytes (128 dims) and k_scale: per-token FP32 scale (NSA uses one scale for the 128-d block) ⏎ 2. Store into cache: token_to_kv_pool.set_index_k_scale_buffer(layer_id, loc,  …[truncated]

### L2-9b2fbf7e6a  (L2, 2026-02-27, sha 9b2fbf7e6ad4, PR #19203)
TITLE: [AMD] Merge Dockerfiles for ROCm (#19203)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/rocm.Dockerfile (+157/-34); docker/rocm720.Dockerfile (+0/-503); .github/workflows/pr-test-amd-rocm720.yml (+2/-2); .github/workflows/release-docker-amd-rocm720-nightly.yml (+1/-1); .github/workflows/release-docker-amd.yml (+1/-4); scripts/ci/amd/amd_ci_start_container.sh (+1/-7)
LABELS: amd, run-ci
BODY: ## Motivation ⏎  ⏎ `rocm720.Dockerfile` and `rocm.Dockerfile` shares a lot in common. It is time to unify them. ⏎  ⏎ ## Modifications ⏎  ⏎ Making `rocm720.Dockerfile` a superset of `rocm.Dockerfile` was the design choice back in #17799 . This PR mostly replaces `rocm.Dockerfile` with `rocm720.Dockerfile`, and modifies all the users of it accordingly.  Refactoring workflow files is a reasonable next step but not include in this PR.  ⏎  ⏎ ## Accuracy Tests …[truncated]

### L2-9496bbd7b1  (L2, 2026-02-27, sha 9496bbd7b11c, PR #18319)
TITLE: [AMD] Use `tilelang` as default NSA attention backend dispatch on AMD Instinct (#18319)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+3/-1); python/sglang/srt/server_args.py (+4/-1)
LABELS: amd, run-ci
BODY: As per title. This PR adds a default on Instinct to `tilelang` for NSA attention backend. ⏎  ⏎ "flashmla_sparse" and "fa3" are not supported on ROCm afaik. Seems like part of this was added in https://github.com/sgl-project/sglang/pull/11061 ⏎  ⏎ See the following on MI355X in `rocm/sgl-dev:v0.5.7-rocm700-mi35x-20260121` with sglang from this branch and the environment ⏎ ``` ⏎ SGLANG_MOE_PADDING=1 ⏎ SGLANG_USE_AITER=1 ⏎ SGLANG_SET_CPU_AFFINITY=1 ⏎ SGLANG_ …[truncated]

### L2-403195d59d  (L2, 2026-02-27, sha 403195d59de0, PR #19443)
TITLE: [AMD] [MiniMax-M2.5 Day 0] Add MiniMax-M2.5 nightly accuracy test (#19443)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/nightly-test-amd-rocm720.yml (+68/-0); .github/workflows/nightly-test-amd.yml (+68/-0); docs/basic_usage/minimax_m2.md (+22/-3); docs/supported_models/text_generation/generative_models.md (+1/-1); test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py (+245/-0); test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py (+249/-0)
LABELS: documentation, amd, run-ci
BODY: ## Summary ⏎  ⏎ - Add MiniMax-M2.5 (`MiniMaxAI/MiniMax-M2.5`) GSM8K few-shot accuracy tests for AMD GPUs (8-GPU, TP=8 + EP=8) ⏎   - **MI30x** (MI325/MI300X): `nightly-8-gpu-minimax-m25` with aiter backend ⏎   - **MI35x**: `nightly-8-gpu-mi35x-minimax-m25` with aiter backend ⏎ - Register both tests in the AMD nightly workflow ⏎ - No model code changes required -- MiniMax-M2.5 is a pure softmax-attention MoE model whose components all already support AMD …[truncated]

### L2-e567215e44  (L2, 2026-02-27, sha e567215e4428, PR #19388)
TITLE: [bugfix]fix fa4 decoding (#19388)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+2/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+8/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Failed to use the fa4 attention backend on the official v0.5.9+cu130+arm64 image. ⏎  ⏎ ## Modifications ⏎  ⏎ Fixed the issue where `forward_decode` was incorrectly using `flash_attn_with_kvcache`. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER …[truncated]

### L2-1b75d0d1a9  (L2, 2026-02-27, sha 1b75d0d1a979, PR #15601)
TITLE: Fix BatchMLAPagedAttentionWrapper query/qo_inptr mismatch for EAGLE (#15601)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+7/-0)
LABELS: bug, blackwell, run-ci
BODY: ## Motivation ⏎ When the batch dimension of the query does not match `qo_indptr[-1]` in `BatchMLAPagedAttentionWrapper` (as described in FlashInfer issue #[2236](https://github.com/flashinfer-ai/flashinfer/issues/2236)), an invalid memory access can occur as below.  ⏎ ``` ⏎   File "/workspace/build/aot/generated/batch_mla_attention_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_ckv_512_head_dim_kpe_64_profiler_False/batch_mla_run.cu" …[truncated]

### L2-7e46aafebb  (L2, 2026-02-27, sha 7e46aafebb6d, PR #18526)
TITLE: [AMD] Enable cudagraph for aiter nsa backend and add aiter impl for nsa pr… (#18526)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa/triton_kernel.py (+60/-0); python/sglang/srt/layers/attention/nsa_backend.py (+70/-3)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎  ⏎ ``` ⏎ kv_indices = page_table_1[page_table_1 != -1]  ⏎ ``` ⏎ Bool mask produced dynamic shape that could not be captured into cuda graph. ⏎  ⏎ Capture NSA decode implemented by aiter backend into cuda graph for better performance. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ 1. Add a triton kernel get_valid_kv_indices to extract kv indices form transformed paged table. ⏎ 2. Add a simple implementation with aiter for nsa prefill phase. ⏎ ## Accuracy Tests …[truncated]

### L2-776709efe8  (L2, 2026-02-27, sha 776709efe893, PR #19122)
TITLE: [3/n] deepseek_v2.py Refactor: Migrate MLA forward method in deepseek_v2.py (#19122)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_backend_handler.py (+1/-1); python/sglang/srt/models/deepseek_common/attention_forward_methods/__init__.py (+6/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py (+1/-1); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+492/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py (+152/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py (+227/-0); python/sglang/srt/models/deepseek_v2.py (+22/-811); test/srt/cpu/test_qkv_proj_with_rope.py (+3/-3); test/srt/cpu/test_rope.py (+2/-2)
LABELS: amd, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Part of #16255 ⏎  ⏎ Split different MLA forward methods into separate files, including: ⏎ - `forward_mla.py` for Cuda/HIP mla aborption ⏎ - `forward_mla_fused_rope_cpu.py‎` for cpu optimization ⏎ - `forward_mla_fused_rope_rocm.py` for rocm optimization ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests (H200) ⏎  ⏎ ### GSM8K ⏎  ⏎ ``` ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319 ⏎ ``` ⏎  ⏎ ``` ⏎ pyth …[truncated]

### L2-b7f13a7b73  (L2, 2026-02-28, sha b7f13a7b7328, PR #19544)
TITLE: [NPU] bugs fix for Deepseek models (#19544)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.npu_mla
FILES: python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py (+3/-2); python/sglang/srt/managers/schedule_policy.py (+9/-9)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ When we run DeepSeek V3.1 and DeepSeek V3.2 models on Ascend NPU devices, we encounter three issues.  ⏎ 1.  when we export SGLANG_NPU_USE_MLAPO=1, this feature does not work. ⏎ 2. The error message is `Tensor self not implemented for DT_UINT64`.  It is about the data type of `bootstrap_room` in utils.py.  ⏎ <img width="1050" height="68" alt="image" src="https://github.com/user-attachments/assets/70428364-10b2-47a7-8f5c-3444df53c36 …[truncated]

### L2-80a6b32703  (L2, 2026-03-01, sha 80a6b32703db, PR #19536)
TITLE: [Perf] Optimize NSA backend metadata under MTP (#19536)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+24/-64); python/sglang/srt/layers/attention/utils.py (+61/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ This PR is authored by @Baidu-AIAK in https://github.com/sgl-project/sglang/pull/17647. I just tested the code, since the author hasn't replied in a while. ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Run with FlashMLA on SM100. It should be the same for `trtllm` ⏎ ``` ⏎ python3 -m sglang.launch_server \ ⏎   --model-path nvidia/DeepSeek-V3.2-NVFP4 \ ⏎   --trust-remote-code \ ⏎   --tp 8 \ ⏎   --quantization modelopt_fp4 \ ⏎   --kv-cache-dtype fp8_e4m3 \ …[truncated]

### L2-f51ddba131  (L2, 2026-03-02, sha f51ddba131c6, PR #18442)
TITLE: feat: add FA4 SM90 paged KV decode support & update attention docs (#18442)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs/advanced_features/attention_backend.md (+12/-1); python/sglang/jit_kernel/flash_attention/cute/flash_fwd.py (+41/-13); python/sglang/jit_kernel/flash_attention/cute/interface.py (+3/-2); python/sglang/srt/server_args.py (+5/-1)
LABELS: documentation, run-ci
BODY: ## Motivation ⏎  ⏎ Add paged KV cache support to FA4 on SM90 (Hopper), enabling FA4 decode on H100/H200 GPUs. Currently FA4 on SM90 is prefill-only because it lacks paged KV support, which is required for decode. SM100 (Blackwell) already has this support. ⏎  ⏎ This implements the sample code provided in #18265 (Part A). The core implementation is based on the diff provided by @b8zhong in the issue. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - **`flash_fwd.py`**: ⏎    …[truncated]

### L2-6822941514  (L2, 2026-03-02, sha 682294151441, PR #19005)
TITLE: [FlashInfer] Bump FlashInfer version from 0.6.3 to 0.6.4 (#19005)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); .github/workflows/release-docker-cu13-framework.yml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/server_args.py (+4/-7); python/sglang/srt/utils/common.py (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎ - CuteDSL FP4 MoE for DeepSeek-R1 Performance ([#2398](https://github.com/flashinfer-ai/flashinfer/pull/2398)) ⏎ - TRTLLM-Gen MxFP8 MoE Integration ([#2505](https://github.com/flashinfer-ai/flashinfer/pull/2505)) ⏎ - GDN Decode CuteDSL Kernel ([#2498](https://github.com/flashinfer-ai/flashinfer/pull/2498)) ⏎ - TRTLLM-Gen Skip-Softmax Attention ([#2477](https://github.com/flashinfer-ai/flashinfer/pull/2477), [#2547](https://github.co …[truncated]

### L2-468e3dc56b  (L2, 2026-03-02, sha 468e3dc56bee, PR #19030)
TITLE: [Qwen3.5] Set full attn_backend to trtllm_mha on SM100 by default when possible (#19030)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+85/-77)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The default mha backend for hybrid qwen models is set to triton, which leads to worse performance when trtllm_mha can be used.  ⏎  ⏎ ## Modifications ⏎  ⏎ - Refactor out `_get_default_attn_backend` so it can be called in `_handle_model_specific_adjustments` ⏎ - Remove the duplicated logic to set default MoE backend for qwen family models ⏎ - Get the default attn_backend by calling `_get_default_attn_backend` and pass it to `_handle_mam …[truncated]

### L2-c64274c746  (L2, 2026-03-02, sha c64274c746f2, PR #16331)
TITLE: Piecewise Cuda Graph set default (#16331)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+27/-1); python/sglang/srt/compilation/compile.py (+2/-14); python/sglang/srt/compilation/piecewise_context_manager.py (+23/-5); python/sglang/srt/configs/model_config.py (+19/-0); python/sglang/srt/layers/attention/fla/layernorm_gated.py (+1/-1); python/sglang/srt/layers/attention/flashinfer_backend.py (+1/-1); python/sglang/srt/layers/communicator.py (+1/-1); python/sglang/srt/layers/moe/topk.py (+31/-1); python/sglang/srt/layers/quantization/awq.py (+7/-12); python/sglang/srt/layers/quantization/fp8_utils.py (+29/-2); (+24 more)
LABELS: quant, Multi-modal, deepseek, npu, run-ci, piecewise-cuda-graph
BODY: ## Motivation ⏎  ⏎ Work in progress ⏎  ⏎ ## Modifications ⏎  ⏎ Work in progress ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Work in progress ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ Work in progress ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sgla …[truncated]

### L2-07b8d763ef  (L2, 2026-03-02, sha 07b8d763ef0b, PR #18882)
TITLE: feat: Add FP8 KV cache support for Triton attention backend (#18882)
SOURCES: path_core, symbol_pickaxe, corpus:kernel-correctness-cases(introducing), body_keyword
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+26/-15); python/sglang/srt/layers/attention/triton_backend.py (+63/-6); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+16/-6); test/registered/attention/test_triton_attention_kernels.py (+14/-0); test/registered/attention/test_wave_attention_kernels.py (+3/-0); test/registered/quant/test_fp8kv_triton.py (+58/-0)
LABELS: run-ci
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:c6850ac30c (fix PR 19736) || deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ The Triton attention backend does not correctly handle FP8 KV cache. It does not pass k_scale/v_scale to set_kv_buffer, causing scales to default to 1.0. In addition, the Triton kernels lack dequantization support. Unlike FlashInfer which handles scales internally, the Triton kernels receive raw FP8 values without applying the corresponding scales. ⏎  ⏎ This PR fuses FP8 dequantization scales directly into the Triton attention kern …[truncated]

### L2-441045a7bf  (L2, 2026-03-03, sha 441045a7bf24, PR #19362)
TITLE: [AMD] Fix EAGLE3 speculative decoding with aiter attention backend (#19362)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+87/-104); test/registered/spec/eagle/test_eagle3_basic.py (+20/-3)
LABELS: amd, run-ci
DEEP_STUDY: deep-study correctness case sglang:441045a7bf: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Motivation ⏎  ⏎ EAGLE3 speculative decoding crashes on the aiter backend for non-MLA models ⏎ (e.g. Llama). Two root causes: ⏎  ⏎ 1. The `target_verify` CUDA graph capture/replay paths had separate, ⏎    duplicated `qo_indptr`/`kv_indptr`/`kv_indices` setup for MLA vs non-MLA. ⏎    The non-MLA branch used different tensor objects than the ones updated ⏎    during replay, causing stale indices and garbled output. ⏎  ⏎ 2. Several `if _use_mla_ps_kernel:`  …[truncated]

### L2-facde4c6d3  (L2, 2026-03-03, sha facde4c6d3f3, PR #19765)
TITLE: [PD] Enable all CP ranks for KVCache transfer  (#19765)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/common/conn.py (+20/-12); python/sglang/srt/disaggregation/mooncake/conn.py (+25/-6); python/sglang/srt/disaggregation/mori/conn.py (+17/-1); python/sglang/srt/disaggregation/nixl/conn.py (+18/-1); python/sglang/srt/disaggregation/utils.py (+82/-1); python/sglang/srt/environ.py (+1/-0)
BODY: ## Motivation ⏎ Changes: ⏎  - Add an env var to allow all CP ranks for KVCache transfer instead of just cp rank 0. The env var is set to False by default since it has little performance improvement for MLA models. ⏎  - Remove `is_mla_backend` check, now we can support Qwen3 CP as well (This PR can unblock #18233). ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Pin …[truncated]

### L2-c6850ac30c  (L2, 2026-03-03, sha c6850ac30cbe, PR #19736)
TITLE: [AMD] Fix Qwen3-Coder-Next: Add missing k_scale/v_scale args to extend_attention_fwd in aiter_backend (#19736)
SOURCES: corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+2/-0)
LABELS: run-ci
DEEP_STUDY: deep-study correctness case sglang:c6850ac30c: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#18882
BODY: ## Summary ⏎  ⏎ - Fix `TypeError: extend_attention_fwd() missing 1 required positional argument: 'v_scale'` crash in aiter_backend when running non-MLA speculative decoding (target_verify / draft_extend) paths: https://github.com/sgl-project/sglang/actions/runs/22648816293/job/65643197339#step:5:6363 ⏎ - Add missing `k_scale=1.0, v_scale=1.0` positional args to the `extend_attention_fwd` call ⏎  ⏎ ## Motivation ⏎  ⏎ \#18882 added `k_scale` and `v_scale` …[truncated]

### L2-a710b7d791  (L2, 2026-03-04, sha a710b7d7910f, PR #18938)
TITLE: [Sarvam] Add inference support for Sarvam MoE LLMs (#18938)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/supported_models/text_generation/generative_models.md (+1/-0); python/sglang/srt/configs/model_config.py (+17/-0); python/sglang/srt/models/sarvam_moe.py (+1525/-0)
LABELS: documentation, run-ci
BODY: Adds inference support for two Sarvam MoE models: ⏎ Sarvam 30B MoE -- GQA attention with QK norm, fused QK-norm-RoPE kernel, sparse MoE with shared experts, and HF checkpoint weight remapping. ⏎ Sarvam 105B MoE -- MLA (Multi-head Latent Attention) with weight absorption (kv_b_proj split into w_kc/w_vc BMMs), FP8 BMM support, multi-backend attention dispatch (FA3, FlashMLA, CutlassMLA, etc.), and optional fp8 MHA prefill path for full-head attention …[truncated]

### L2-472eef4071  (L2, 2026-03-05, sha 472eef4071ed, PR #19727)
TITLE: fa4 cleanup (#19727)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-0); python/sglang/jit_kernel/flash_attention/cute/.flake8 (+0/-4); python/sglang/jit_kernel/flash_attention/cute/AUTHORS (+0/-5); python/sglang/jit_kernel/flash_attention/cute/LICENSE (+0/-29); python/sglang/jit_kernel/flash_attention/cute/README.md (+0/-0); python/sglang/jit_kernel/flash_attention/cute/__init__.py (+0/-21); python/sglang/jit_kernel/flash_attention/cute/ampere_helpers.py (+0/-103); python/sglang/jit_kernel/flash_attention/cute/barrier.py (+0/-71); python/sglang/jit_kernel/flash_attention/cute/benchmark.py (+0/-268); python/sglang/jit_kernel/flash_attention/cute/blackwell_helpers.py (+0/-753); (+36 more)
LABELS: documentation, dependencies, Multi-modal, sgl-kernel, run-ci, diffusion
BODY: ## Motivation ⏎  ⏎ #19447  ⏎  ⏎ cleanup fa4 by using sgl-fa4 pkg ⏎  ⏎ ## Modifications ⏎  ⏎ cleanup fa4 code copy ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/b …[truncated]

### L2-27053aa5ed  (L2, 2026-03-06, sha 27053aa5ed2d, PR #19902)
TITLE: Fix MLA decode path returning unwritten (padded) rows (#19902)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+2/-1)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ This PR fixes a correctness issue in the MLA decode path where we used `q_mask` from `pad_sequence_with_mask()` to select rows from the `mla_decode_fwd` output buffer. ⏎  ⏎ The underlying MLA kernel only writes the first `qo_indptr[-1]` rows (packed/compacted output), leaving the remaining padded rows **unwritten**.   ⏎  ⏎ Using `o[q_mask]` can therefore read **unwritten memory** (garbage / NaNs) and propagate NaNs into downstream compu …[truncated]

### L2-5471e4a492  (L2, 2026-03-06, sha 5471e4a49215, PR #19842)
TITLE: [NPU][Feature] eliminate dsv3 redundant rotary embed calculation (#19842)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.npu_mla
FILES: python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py (+12/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The sin and cos values at each layer are calculated repeatedly. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Only the first layer calculation is required; the same value is used for the subsequent layers. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pu …[truncated]

### L2-de1a0afcbc  (L2, 2026-03-06, sha de1a0afcbc7c, PR #18357)
TITLE: [MUSA][10/N] Add GGUF support (#18357)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/gguf.py (+6/-5)
LABELS: mthreads
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR continues the ongoing effort (tracked in #16565) to add full support for **Moore Threads GPUs** in SGLang by leveraging **MUSA (Meta-computing Unified System Architecture)** for LLM inference. ⏎  ⏎ This submission is to enable GGUF support. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/pull/17946 has brought GGUF kernels to MUSA and this one simply enables it in SGLang runtime level. ⏎  ⏎ ### Testing Done …[truncated]

### L2-50bbdcf8e9  (L2, 2026-03-06, sha 50bbdcf8e97d, PR #20068)
TITLE: Relax flaky test thresholds for MLA DeepSeek V3 and AutoRound (#20068)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: test/registered/mla/test_mla_deepseek_v3.py (+1/-1); test/registered/quant/test_autoround.py (+1/-1)
LABELS: deepseek
BODY: ## Summary ⏎ - `test_mla_deepseek_v3.py`: Lower GSM8K accuracy threshold from 0.62 to 0.60 for `TestMLADeepseekV3Fa3Fp8Kvcache` — test hit exact boundary (0.62 not > 0.62) ⏎ - `test_autoround.py`: Lower MMLU score threshold from 0.26 to 0.25 for Qwen2 model — scored 0.25 ⏎  ⏎ [Failure example](https://github.com/sgl-project/sglang/actions/runs/22786613432/job/66104943534?pr=19982)

### L2-925185f9ec  (L2, 2026-03-06, sha 925185f9ecd1, PR #20061)
TITLE: Fix flashinfer backend with pcg (#20061)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_backend.py (+4/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Previously, with prefix cache, flashinfer backend and pcg will produce obviously wrong output. This change follows the modification in `flashinfer_mla`, which disables ragged when using pcg. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob …[truncated]

### L2-43d6a32045  (L2, 2026-03-07, sha 43d6a32045ec, PR #18902)
TITLE: [sgl-kernel] rebase FlashMLA 0217 (#18902)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, body_keyword
ARTIFACT_HINTS: L2.backend.flashmla, L2.build.flashmla_sgl_kernel, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashmla_backend.py (+108/-65); sgl-kernel/cmake/flashmla.cmake (+47/-11); sgl-kernel/python/sgl_kernel/flash_mla.py (+9/-0)
LABELS: high priority, sgl-kernel, run-ci
BODY: ## Summary ⏎ This PR updates SGLang's FlashMLA integration to the latest validated rebase snapshot on the SGL-maintained FlashMLA branch. ⏎  ⏎ ## What changed ⏎ - Updated `sgl-kernel/cmake/flashmla.cmake` to use the latest FlashMLA commit: ⏎   - `GIT_TAG=9804b12079e4c873514d3457aa588d3ccf40da28` ⏎ - Keeps SGLang pinned to an immutable commit SHA instead of a moving branch ref. ⏎  ⏎ ## FlashMLA side included in this pin ⏎ The pinned FlashMLA commit includes the reb …[truncated]

### L2-f88acf8780  (L2, 2026-03-07, sha f88acf87805b, PR #20012)
TITLE: [JIT Kernel] Reland NVFP4 kernels to JIT (#20012)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/benchmark/bench_nvfp4_blockwise_moe.py (+250/-0); python/sglang/jit_kernel/benchmark/bench_nvfp4_quant.py (+181/-0); python/sglang/jit_kernel/benchmark/bench_nvfp4_scaled_mm.py (+175/-0); python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_expert_quant.cuh (+127/-143); python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_quant.cuh (+10/-32); python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_quant_entry.cuh (+68/-0); python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_quant_kernels.cuh (+53/-54); python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_scaled_mm_entry.cuh (+34/-0); python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_scaled_mm_kernels.cuh (+202/-159); python/sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh (+341/-157); (+17 more)
LABELS: quant, sgl-kernel, blackwell, run-ci
DEEP_STUDY: deep-study revert record: reland of PR(s) 19437 reason=crash_or_hang || deep-study performance PR (precision_format)
BODY: ## Summary ⏎  ⏎ Reland #19437 after fixing some missed custom-op registration that broke PCG ⏎  ⏎ cc @Fridge003 @DarkSharpness @BBuf  ⏎  ⏎ ## Accuracy Tests ⏎ PCG here is turned on by default now ⏎ ```shell ⏎ SGLANG_MOE_NVFP4_DISPATCH=1 sglang serve \ ⏎   --model-path nvidia/DeepSeek-V3-0324-NVFP4 \ ⏎   --tensor-parallel-size 4 \ ⏎   --expert-parallel-size 4 \ ⏎   --attention-backend trtllm_mla \ ⏎   --moe-runner-backend flashinfer_cutlass \ ⏎   --quantization  …[truncated]

### L2-d28f35240a  (L2, 2026-03-07, sha d28f35240a28, PR #20086)
TITLE: [V32/GLM5] Change default setting of V32 nvfp4 on TP4 (#20086)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+15/-6)
LABELS: run-ci
BODY: ## Motivation ⏎ After the flashmla is rebased to latest version, DeepSeekV32 fp4+tp4 will break on flashmla decode kernel, since the latest code doesn't support q_head=32 ⏎ error log: https://github.com/sgl-project/sglang/actions/runs/22779139862/job/66096508676?pr=18902 ⏎  ⏎ To unblock the upgrade of flashmla, we change the default setting of DeepSeekV32 fp4+tp4 to bf16 kv_cache_dtype, flashmla sparse prefill kernel, and trtllm decode kernel.  For D …[truncated]

### L2-36b557d2c9  (L2, 2026-03-08, sha 36b557d2c916, PR #20070)
TITLE: Fix streaming session with paged KV cache (SWA/MLA) (#20070)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/schedule_batch.py (+4/-1); python/sglang/srt/managers/scheduler_runtime_checker_mixin.py (+23/-4); python/sglang/srt/mem_cache/base_prefix_cache.py (+1/-0); python/sglang/srt/mem_cache/session_aware_cache.py (+27/-2); test/registered/sessions/test_session_latency.py (+10/-5)
LABELS: deepseek, run-ci
BODY: ## Summary ⏎ Fix streaming sessions crashing on models with `page_size > 1` (SWA, MLA, etc.). ⏎  ⏎ **Root cause:** `SessionAwareCache.match_prefix` returns `device_indices` of length `kv_committed_len` (not page-aligned), which was assigned to `req.cache_protected_len`, violating the page-alignment invariant. ⏎  ⏎ **Fix:** Pass `slot.cache_protected_len` (the page-aligned tree-inserted prefix length from turn 1) through a new `MatchResult.cache_protected_l …[truncated]

### L2-230fb55899  (L2, 2026-03-08, sha 230fb5589960, PR #17216)
TITLE: [Performance] Decode Offload improves the long texts performance 100% through dynamic block offload. (#17216)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/references/environment_variables.md (+2/-0); python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py (+88/-31); python/sglang/srt/disaggregation/kv_events.py (+17/-0); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/managers/cache_controller.py (+2/-0); python/sglang/srt/managers/scheduler_output_processor_mixin.py (+7/-1); test/registered/disaggregation/test_disaggregation_decode_offload.py (+169/-0); test/registered/disaggregation/test_specv2_kvcache_offloading.py (+4/-1)
LABELS: documentation, high priority, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Changing the offload process to dynamic on-the-fly offloading significantly increases the number of requests that decode nodes can process in parallel. On H20-96GiB devices, the concurrency count can be more than doubled, and the end-to-end performance is doubled. The deployed test model is DeepSeek-V3.2-W4AFP8, with the device being H20-96GiB, 1P (PP8) 1D (EP8). ⏎  ⏎ DeepSeek-V3.2 introduces sparse kv cache, which brings significa …[truncated]

### L2-be63f982b7  (L2, 2026-03-09, sha be63f982b7b7, PR #20062)
TITLE: [V32/GLM5] Control the threshold of applying dense attention with an environ (#20062)
SOURCES: symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: docs/references/environment_variables.md (+2/-0); python/sglang/srt/environ.py (+1/-2); python/sglang/srt/layers/attention/nsa_backend.py (+3/-46); python/sglang/srt/server_args.py (+26/-3); test/registered/quant/test_deepseek_v32_fp4_4gpu.py (+0/-4); test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py (+0/-4)
LABELS: documentation, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ - Add an environ `SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD`, for controlling whether to use dense MHA or sparse MLA kernel. It's set to index.topk by default, thus not breaking the original logic. ⏎ - For GLM-5 model on blackwell, this environ is set to 0 so as to avoid kernel issues. ⏎ - Remove the useless `SGLANG_VERIFY_FUSED_METADATA_COPY` environ ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎ Bench serving with V32  …[truncated]

### L2-0fd9a57d80  (L2, 2026-03-09, sha 0fd9a57d80dc, PR #20210)
TITLE: [Doc] Verify and Modify some attention backend specs (#20210)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/advanced_features/attention_backend.md (+35/-27)
LABELS: documentation
ISSUES: #20201 [Doc] Verify some attention backend specs
BODY: Closes #20201 ⏎  ⏎ ## Motivation ⏎  ⏎ The attention backend support matrix documentation (`docs/advanced_features/attention_backend.md`) contained several inaccuracies when compared against the actual codebase. This PR fixes 6 incorrect cells/claims found through code-level verification. ⏎  ⏎ ## Modifications ⏎  ⏎ ### Fix 1: AITER (ROCm) — Sliding Window: ❌ → ✅ ⏎  ⏎ [`aiter_backend.py:1788-1806`](https://github.com/sgl-project/sglang/blob/main/python/sglan …[truncated]

### L2-006bd44cf9  (L2, 2026-03-11, sha 006bd44cf920, PR #19319)
TITLE: [deepseekv3.2] fix get_k_and_s_triton kenel for 128K seqlen case bug (#19319)
SOURCES: release_notes
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.pool.mla_token_kv
FILES: python/sglang/srt/layers/attention/nsa/index_buf_accessor.py (+105/-48); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+29/-22); python/sglang/srt/mem_cache/memory_pool.py (+9/-2); test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py (+191/-0); test/manual/layers/attention/nsa/test_index_buf_accessor.py (+46/-9)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ fix get_k_and_s_triton kenel for 128K seqlen case bug ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/mai …[truncated]

### L2-ab4b863546  (L2, 2026-03-11, sha ab4b86354643, PR #20380)
TITLE: fix ci by removing nvidia-cutlass-dsl-libs-base and force reinstall n… (#20380)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+5/-0); scripts/ci/cuda/ci_install_dependency.sh (+2/-0)
BODY: …vidia-cutlass-dsl 4.3.5 ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ Fix CI for nvidia-cutlass-dsl auto upgrading install along with nvidia-cutlass-dsl-libs-base 4.4.x error. ⏎  ⏎ - refer https://github.com/flashinfer-ai/flashinfer/pull/2760 ⏎ - refer https://github.com/sgl-project/sglang/pull/20309 ⏎ - refer https://github.com/sgl-project/sgl-flash-attn/pull/40 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ##  …[truncated]

### L2-d093e70067  (L2, 2026-03-11, sha d093e700672d, PR #20326)
TITLE: [Doc] Add DSA/NSA attention backend to support matrix (#20326)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/advanced_features/attention_backend.md (+19/-1)
LABELS: documentation
ISSUES: #20314 [Docs] Missing DSA attention backend matrix
BODY: ## Summary ⏎ - Adds NSA (DSA) row to the MLA backends table with its capabilities (FP8 KV cache, speculative decoding topk=1, chunked prefix cache) ⏎ - Adds a new "DSA Attention Backend (NSA)" section documenting the NSA sub-backends (flashmla_sparse, flashmla_kv, flashmla_auto, fa3, trtllm, tilelang), their prefill/decode support, and default selection logic ⏎ - Includes launch command examples for DeepSeek V3.2 with NSA ⏎  ⏎ Fixes #20314 ⏎  ⏎ ## Test plan

### L2-680d9d98e4  (L2, 2026-03-11, sha 680d9d98e468, PR #20309)
TITLE: Fix cutedsl ci error (#20309)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); scripts/ci/cuda/ci_install_dependency.sh (+3/-0)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎ 3. Trigge …[truncated]

### L2-67f02681c9  (L2, 2026-03-11, sha 67f02681c9a3, PR #17450)
TITLE: [AMD] Support speculative decoding v2 for aiter backend on ROCm/HIP (#17450)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+316/-12); python/sglang/srt/speculative/eagle_info_v2.py (+1/-1); python/sglang/srt/speculative/eagle_worker_v2.py (+19/-8); test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py (+5/-3)
LABELS: amd, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ This PR supports `SGLANG_ENABLE_SPEC_V2=1` + `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` with aiter attention backend for AMD GPUs. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎ ### Server command:  ⏎ ``` ⏎ SGLANG_USE_AITER=1 \ ⏎ SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \ ⏎ SGLANG_INT4_WEIGHT=0 \ ⏎ SGLANG_MOE_PADDING=1 \ ⏎ SGLANG_SET_CPU_AFFINITY=1 \ ⏎ SGLANG_ROCM_FUSED_DECODE_MLA=1 \ ⏎ SGLANG_USE_R …[truncated]

### L2-1e2983c98e  (L2, 2026-03-12, sha 1e2983c98ebd, PR #19935)
TITLE: [AMD] Fix FP8 assertion failure in aiter MLA decode by falling back to self.k_scale (#19935)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+20/-8); test/registered/amd/accuracy/mi35x/test_kimi_k25_aiter_mla_eval_mi35x.py (+237/-0); test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py (+239/-0); test/registered/amd/test_kimi_k25_mxfp4.py (+117/-0)
LABELS: amd, run-ci
BODY: ## Motivation ⏎  ⏎ When `layer.k_scale` is `None` (the default in `RadixAttention`), the aiter ASM MLA kernel asserts `q_scale.has_value() && kv_scale.has_value()` for FP8 Q tensors. Fall back to `self.k_scale` (initialized to `tensor([1.0])`) at all 4 `mla_decode_fwd` call sites, matching the pattern used by `flashmla_backend`. ⏎ This issue was found in Kimi-K2.5 but not in DeepSeek-R1. It must be resolved to enable the FP8 KV cache for Kimi-K2.5 / …[truncated]

### L2-318a40fdfb  (L2, 2026-03-12, sha 318a40fdfb7f, PR #20399)
TITLE: [Bug-fix] Fix gpu fault when run the test with dp-attention-enabled and max-concurrency is over 256 (#20399)
SOURCES: symbol_pickaxe, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+5/-1)
LABELS: amd, run-ci
DEEP_STUDY: deep-study correctness case sglang:318a40fdfb: class=integration_backend_cudagraph; symptom=illegal_memory_access; introducing=unknown
BODY: ## Motivation ⏎  ⏎ Fix gpu fault issue when using mla_decode_fwd with dp-attention-enabled and max-concurrency is over 256. ⏎  ⏎ ## Modifications ⏎  ⏎ The major change is to change the mla_decode_fwd mode and that can get the correct metadata for the kernel to use. ⏎  ⏎ ## Accuracy Tests ⏎ **Server-command** ⏎ `python3 -m sglang.launch_server \ ⏎     --moe-a2a-backend none \ ⏎     --model-path amd/DeepSeek-R1-0528-MXFP4  \ ⏎     --tp-size 8 \ ⏎     --ep-size 8 …[truncated]

### L2-93afe15b43  (L2, 2026-03-14, sha 93afe15b4370, PR #20480)
TITLE: chore: bump flashinfer version to 0.6.6 (#20480)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the flashinfer version to `0.6.6` across all relevant files. ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎ - python/sglang/srt/utils/common.py ⏎ - scripts/ci/cuda/ci_install_dependency.sh ⏎  ⏎ 🤖 Generated with GitHub Actions

### L2-15097c5c3b  (L2, 2026-03-16, sha 15097c5c3b52, PR #20440)
TITLE: Release sglang kernel 0.4.0 (#20440)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+4/-4); python/pyproject.toml (+1/-1); .github/workflows/release-whl-kernel.yml (+1/-1); 3rdparty/amd/wheel/README.md (+5/-5); 3rdparty/amd/wheel/sglang/pyproject.toml (+2/-2); docs/developer_guide/contribution_guide.md (+5/-5); docs/get_started/install.md (+3/-3); docs/platforms/ascend_contribution_guide.md (+4/-4); python/sglang/check_env.py (+1/-1); python/sglang/jit_kernel/benchmark/bench_awq_marlin_moe_repack.py (+7/-8); (+26 more)
LABELS: documentation, high priority, amd, dependencies, sgl-kernel, npu, run-ci, mthreads, jit-kernel
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎ 3. Trigge …[truncated]

### L2-cb1e63aba4  (L2, 2026-03-17, sha cb1e63aba45e, PR #20303)
TITLE: bump fa4 to official released fa4 pkg (#20303)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+3/-4); python/pyproject.toml (+3/-4); python/sglang/jit_kernel/flash_attention_v4.py (+2/-2); scripts/ci/cuda/ci_install_dependency.sh (+3/-15)
LABELS: dependencies, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ - fa4 integration with official fa4 pkg. ⏎  ⏎ ## Modifications ⏎  ⏎ - bump fa4 pkg ⏎ - bump nvidia-cutlass-dsl pkg ⏎ - bump quack-kernels pkg ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [ …[truncated]

### L2-88c40ec16d  (L2, 2026-03-17, sha 88c40ec16d92, PR #20604)
TITLE: Use Flashinfer for target_verify in GDN model for SM120 (#20604)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+2/-1)
LABELS: run-ci
ISSUES: #20709 [Bug] Flashifier issue on Blackwell GPUs (SM12X and SM110)
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: When the context is long (32000 ISL, 1024 OSL) target verify will be very very slow, due to using Triton prefill which has no split-KV optimization (and, using decode Triton will also no help), in fact it will be a bit slower. ⏎  ⏎ Although there is trtllm-gen, it only support XQA decode (no prefill), so we can use Flashinfer with FA2 impl (for full attention). And now, it will select Flashinfer by default. Uncertain why this gate is intially creat …[truncated]

### L2-93422f27d6  (L2, 2026-03-18, sha 93422f27d6f4, PR #20409)
TITLE: [AMD][AITER] Guard _use_mla_ps_kernel with self.use_mla in draft_extend_v2 paths (#20409)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+2/-2)
LABELS: run-ci
ISSUES: #16027 [AMD][Bug] GLM-4.7 EAGLE speculative decoding fails with AITER attention backend on ROCm - missing `max_split_per_batch` attribute
BODY: ## Motivation ⏎  ⏎ This PR fixes #16027 — non-MLA models (e.g. GLM-4.7) crash with `AttributeError: 'AiterAttnBackend' object has no attribute 'max_split_per_batch'` when using EAGLE speculative decoding with the AITER attention backend on ROCm. ⏎  ⏎ The root cause: the global flag `_use_mla_ps_kernel` defaults to `True`, but `self.max_split_per_batch` is only initialized inside the `if self.use_mla:` block in `__init__`. Two code paths in `is_draft_ …[truncated]

### L2-cd22aa27a9  (L2, 2026-03-18, sha cd22aa27a941, PR #9744)
TITLE: [CPU] Add FP8 Bmm support (#9744)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+44/-28); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py (+2/-1); python/sglang/srt/models/bailing_moe_linear.py (+0/-8); python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py (+0/-10); python/sglang/srt/models/longcat_flash.py (+0/-12); python/sglang/srt/models/longcat_flash_nextn.py (+0/-4); sgl-kernel/csrc/cpu/bmm.cpp (+93/-11); sgl-kernel/csrc/cpu/gemm.h (+18/-0); sgl-kernel/csrc/cpu/gemm_fp8.cpp (+310/-1); sgl-kernel/csrc/cpu/qkv_proj.cpp (+4/-2); (+4 more)
LABELS: deepseek, sgl-kernel, intel, cpu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ This PR is a follow-up on https://github.com/sgl-project/sglang/issues/8281 to add FP8 BMM support in MLA weight absorb. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - The main change is the C++ kernels for fp8 bmm on CPU:  ⏎   - sgl-kernel/csrc/cpu/bmm.cpp ⏎   - sgl-kernel/csrc/cpu/gemm_fp8.cpp ⏎ - apply fp8 bmm in deepseek: python/sglang/srt/models/deepseek_v2.py ⏎ - unit test: test/srt/cpu/test_bmm.py ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ MMLU result on Deep …[truncated]

### L2-126cd5cfae  (L2, 2026-03-18, sha 126cd5cfae7a, PR #20392)
TITLE: gpt-oss decode performance optimization (#20392)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.aiter_mla, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+357/-88); python/sglang/srt/layers/attention/utils.py (+186/-0); python/sglang/srt/layers/quantization/unquant.py (+4/-0); python/sglang/srt/server_args.py (+2/-0)
LABELS: quant, amd, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ Improve the performance for gpt-oss model run ⏎  ⏎ ## Modifications ⏎  ⏎ Three parts for this PR ⏎ 1) linear operation optimization by using triton kernel to replace the naive a16w16 gemm ⏎ 2) fused the elementwise kernels of save kv into one triton kernel ⏎ 3) use unified_attention triton kernel to replace triton decode attention kernels ⏎  ⏎ ## Accuracy Tests ⏎ **Server command** ⏎ `SGLANG_USE_AITER=1 python3 -m sglang.launch_server --mod …[truncated]

### L2-574572b21b  (L2, 2026-03-19, sha 574572b21bf7, PR #20492)
TITLE: [BugFix] bug fix for DeepSeek eagle3 in Attn-DP mode (#20492)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+2/-2)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ The current DeepSeek Eagle3 implementation has an issue in DP scenarios when using tensor_model_parallel_all_gather. The gather should be performed across the attention_tp group instead of the tensor parallel group. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ tp 16 ⏎ ```sh ⏎ python -m sglang.launch_server --skip-server-warmup  \ ⏎     --model-path /xxx/Kimi-K2.5-w4a8 --quantization modelslim --dtype bfloat16 \ ⏎     --host  …[truncated]

### L2-6c91590e1b  (L2, 2026-03-20, sha 6c91590e1bab, PR #19669)
TITLE: [HiCache] refactor: hicache normalization flow and compatibility checks (#19669)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/managers/cache_controller.py (+4/-0); python/sglang/srt/mem_cache/hiradix_cache.py (+0/-7); python/sglang/srt/server_args.py (+86/-39); test/registered/unit/server_args/test_server_args.py (+94/-0)
LABELS: hicache, run-ci
BODY: ## Motivation ⏎ 1. Centralize HiCache compatibility normalization in ServerArgs and refactor `_handle_hicache` ⏎ 2. Add testcase for hicache compatibility normalization in `test_server_args.py` ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAIN …[truncated]

### L2-766d225fcc  (L2, 2026-03-22, sha 766d225fccf0, PR #20910)
TITLE: Add SGLang CUDA crash API logging inspired by FlashInfer (#20910)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.backend.flashinfer_general_mla
FILES: .claude/skills/debug-cuda-crash/SKILL.md (+657/-0); docs/diffusion/environment_variables.md (+12/-0); docs/references/environment_variables.md (+5/-0); python/sglang/jit_kernel/awq_marlin_repack.py (+3/-0); python/sglang/jit_kernel/debug_utils.py (+45/-0); python/sglang/jit_kernel/diffusion/triton/norm.py (+55/-2); python/sglang/jit_kernel/diffusion/triton/rmsnorm_onepass.py (+5/-1); python/sglang/jit_kernel/diffusion/triton/rotary.py (+19/-2); python/sglang/jit_kernel/diffusion/triton/scale_shift.py (+56/-3); python/sglang/jit_kernel/flash_attention_v4.py (+4/-0); (+36 more)
LABELS: documentation, quant, deepseek, hicache, sgl-kernel, blackwell, run-ci, diffusion, jit-kernel
BODY: ## Motivation ⏎  ⏎ This PR adds SGLang-native API-level CUDA crash logging for LLM and diffusion kernel call boundaries. ⏎  ⏎ The implementation is inspired by FlashInfer's API logging utility: ⏎ https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/api_logging.py ⏎  ⏎ This version keeps the scope focused on crash debugging and level-10 dump capture. Replay-related code was intentionally not included so the implementation stays smaller and al …[truncated]

### L2-4779755eb9  (L2, 2026-03-23, sha 4779755eb93f, PR #21219)
TITLE: Split pr-test.yml: extract sgl-kernel, jit-kernel, and multimodal-gen tests into separate workflow files (#21219)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/list-active-pr-runs.yml (+0/-0); .github/workflows/pr-test-jit-kernel.yml (+133/-0); .github/workflows/pr-test-multimodal-gen.yml (+191/-0); .github/workflows/pr-test-sgl-kernel.yml (+209/-0); .github/workflows/pr-test.yml (+44/-471)
LABELS: Multi-modal, run-ci
BODY: ## Summary ⏎ - Extract `sgl-kernel-unit-test`, `sgl-kernel-mla-test`, `sgl-kernel-benchmark-test`, `sgl-kernel-b200-test`, and `cuda13-kernel-smoke-test` into `pr-test-sgl-kernel.yml` ⏎ - Extract `jit-kernel-unit-test`, `jit-kernel-unit-test-nightly`, and `jit-kernel-benchmark-test` into `pr-test-jit-kernel.yml` ⏎ - Extract `multimodal-gen-test-1-gpu`, `multimodal-gen-test-2-gpu`, and `multimodal-gen-unit-test` into `pr-test-multimodal-gen.yml` ⏎ - Repla …[truncated]

### L2-6cb1c2d53d  (L2, 2026-03-24, sha 6cb1c2d53da3, PR #20294)
TITLE: [AMD] Add 4-GPU test suite for MI325 runners (#20294)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/nightly-test-amd-rocm720.yml (+34/-0); .github/workflows/nightly-test-amd.yml (+34/-0); .github/workflows/pr-test-amd-rocm720.yml (+58/-0); .github/workflows/pr-test-amd.yml (+58/-0); test/registered/debug_utils/test_engine_dumper_comparator_e2e.py (+7/-1); test/registered/distributed/test_dp_attention_large.py (+14/-1); test/registered/distributed/test_pp_single_node.py (+15/-2); test/registered/rl/test_multi_instance_release_memory_occupation.py (+6/-1); test/registered/rl/test_return_routed_experts.py (+6/-1); test/registered/spec/eagle/test_eagle_dp_attention.py (+13/-4); (+2 more)
LABELS: amd, run-ci
BODY: ## Summary ⏎  ⏎ Add 4-GPU AMD CI test coverage on MI325 runners: **+3 per-commit tests** and **+1 nightly test**. ⏎  ⏎ ### What this PR does ⏎  ⏎ 1. **Adds stage-c 4-GPU job** to both `pr-test-amd.yml` (ROCm 7.0) and `pr-test-amd-rocm720.yml` (ROCm 7.2) ⏎ 2. **Adds nightly 4-GPU jobs** to both `nightly-test-amd.yml` (ROCm 7.0) and `nightly-test-amd-rocm720.yml` (ROCm 7.2) ⏎ 3. **Registers AMD 4-GPU tests** with `register_amd_ci()` — enabled tests use `is …[truncated]

### L2-2b75fed0dd  (L2, 2026-03-24, sha 2b75fed0ddeb, PR #21337)
TITLE: Workaround of DSA performance drop on B200 + DP (#21337)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+11/-5)
BODY: ## Motivation ⏎  ⏎ #21291 ⏎ Route the config of glmfp8+B200+DP to bf16 kvcache+flashmla sparse prefill+trtllm decode ⏎ Note that this is just a workaround, not root fix. #21011 can be the potential root fix ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ # Launch ⏎ sglang serve --model-path zai-org/GLM-5-FP8 --tp 8 --trust-remote-code --dp 8 --enable-dp-attention  ⏎  ⏎ # Benchmark: 20-shots gsm8k ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-s …[truncated]

### L2-dbe871efdd  (L2, 2026-03-25, sha dbe871efdd97, PR #21430)
TITLE: Rollback flashmla to older version [1/2] (#21430)
SOURCES: path_core, subject_keyword, symbol_pickaxe, dependency_pin, corpus:confirmed-reverts, corpus:confirmed-reverts(reverted)
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: sgl-kernel/cmake/flashmla.cmake (+11/-47); sgl-kernel/python/sgl_kernel/flash_mla.py (+0/-9)
LABELS: sgl-kernel, run-ci
DEEP_STUDY: deep-study revert record: explicit_rollback of PR(s)  reason=correctness_or_accuracy || deep-study: this PR was reverted by PR 21922 (reland, reason=unstated)
BODY: ## Motivation ⏎  ⏎ Temporarily avoid #21291 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and …[truncated]

### L2-f142608408  (L2, 2026-03-25, sha f142608408ed, PR #21296)
TITLE: [MUSA] apply_vocab_mask support musa device (#21296)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/constrained/xgrammar_backend.py (+1/-5)
LABELS: run-ci
BODY: ## Motivation ⏎ The xgrammar backend currently supports several hardware accelerators, including CUDA, NPU, and XPU, for constrained decoding. This pull request adds support for Moore Threads (MUSA) devices to enable efficient constrained decoding features using the xgrammar backend on MUSA hardware. ⏎  ⏎ ## Modifications ⏎  ⏎ - Updated the device check logic in xgrammar_backend.py to include musa as a supported device type. ⏎ - This change ensures tha …[truncated]

### L2-d9e96153de  (L2, 2026-03-26, sha d9e96153de8a, PR #18032)
TITLE: [NPU] Support Hybrid KV Cache for Ascend backend (#18032)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+56/-3); python/sglang/srt/mem_cache/swa_memory_pool.py (+32/-5); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+56/-12)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ This PR implements Hybrid KV Cache support for Ascend NPU hardware. Hybrid KV Cache is essential for optimizing memory efficiency and inference throughput, especially for models using Sliding Window Attention (SWA). This modification enables Ascend users to leverage these memory optimizations, bridging the feature gap between CUDA and NPU backends in sglang. ⏎  ⏎ ## Modifications ⏎  ⏎ 1. In swa_memory_pool.py ⏎ Adapted SWATokenToKVPoo …[truncated]

### L2-4b5f63e1b8  (L2, 2026-03-26, sha 4b5f63e1b8ba, PR #20606)
TITLE: FIX: (NSA) Compute topk_indices_offset when NSA prefill flashmla_sparse is used with FP8 KV cache (#20606)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+20/-4)
BODY: ## Motivation ⏎  ⏎ When using the flashmla_sparse NSA prefill backend with FP8 KV cache, topk_indices_offset is never computed outside the normal EXTEND forward mode, causing a crash in forward_extend(). Rather than letting this silently crash the server, this PR ensures topk_indices_offset is always correctly computed whenever TopkTransformMethod.RAGGED is active, allowing inference to proceed normally. ⏎  ⏎  ⏎  ⏎ ## Root Cause ⏎ The bug is triggered b …[truncated]

### L2-d864622a68  (L2, 2026-03-27, sha d864622a6824, PR #18311)
TITLE: [Hicache & JIT_kernel] Support page first layout  & mla jit kernel (#18311)
SOURCES: subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/csrc/hicache.cuh (+153/-13); python/sglang/jit_kernel/hicache.py (+64/-0); python/sglang/jit_kernel/tests/test_hicache.py (+247/-0); python/sglang/srt/mem_cache/memory_pool_host.py (+151/-58)
LABELS: hicache, ready-to-merge, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎ 3. Trigge …[truncated]

### L2-9a91323c9f  (L2, 2026-03-27, sha 9a91323c9f97, PR #21561)
TITLE: test: point DSV3 int8 MLA CI models to lmsys Hugging Face org (#21561)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: test/registered/mla/test_mla_int8_deepseek_v3.py (+5/-5)
LABELS: deepseek
BODY: Updates Hugging Face model IDs in `test_mla_int8_deepseek_v3.py` from `sgl-project/*` to `lmsys/*` for channel/block int8 weights and the EAGLE draft model path. ⏎  ⏎ Made with [Cursor](https://cursor.com)

### L2-efebcab43e  (L2, 2026-03-28, sha efebcab43ed8, PR #19089)
TITLE: Support skip-softmax attention (#19089)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+4/-0); docs/references/environment_variables.md (+2/-0); python/sglang/bench_serving.py (+1/-0); python/sglang/benchmark/datasets/__init__.py (+2/-0); python/sglang/benchmark/datasets/longbench_v2.py (+104/-0); python/sglang/srt/environ.py (+4/-0); python/sglang/srt/layers/attention/nsa_backend.py (+2/-0); python/sglang/srt/layers/attention/trtllm_mha_backend.py (+3/-0)
LABELS: documentation, blackwell, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: To accelerating long context inference with skip-softmax attention. The work is originally proposed and implemented in [TRTLLM](https://github.com/bobboli/TensorRT-LLM/blob/user/lbo/skip_softmax_blog/docs/source/blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md). This PR enables the corresponding attention kernels and depends on flashinfer-python==0.6.4. ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Ac …[truncated]

### L2-c7adca9992  (L2, 2026-03-31, sha c7adca99929c, PR #21752)
TITLE: Fix kimi-linear launch server error (#21752)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+5/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Main kimi-linear was broken due to self.scaling being deleted. ⏎ ``` ⏎ self.scaling = 1 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim) ⏎ ``` ⏎ It causes the server launch error for kimi-linear model. ⏎ ``` ⏎ ➜  python git:(main) ✗ sglang serve --model-path moonshotai/Kimi-Linear-48B-A3B-Instruct --tp-size 2 --trust-remote --device cuda --host 127.0.0.1 --port 30000 ⏎ Warning: You are sending unauthenticated requests to the …[truncated]

### L2-ca3ba05a7a  (L2, 2026-03-31, sha ca3ba05a7aa5, PR #21422)
TITLE: chore: bump flashinfer version to 0.6.7 (#21422)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/jit_kernel/benchmark/diffusion/bench_fused_norm_scale_shift.py (+5/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1); python/sglang/test/lora_utils.py (+3/-0); test/registered/lora/test_lora_tp.py (+2/-1); test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py (+18/-1)
LABELS: high priority, dependencies, lora, sgl-kernel, run-ci, jit-kernel
ISSUES: #18980 [Bug] GLM 5 Crashes at nsa_backend on B200 | #18989 [Bug] deepseek 3.2 nvfp4 moe-runner-backend=flashinfer_trtllm illegal memory access | #19081 [Bug] Deepseek 3.2 nvfp4 specv2 nsa-decode-backend=trttlm kernel crash
BODY: ## Summary ⏎  ⏎ This PR bumps the flashinfer version to `0.6.7` across all relevant files. ⏎  ⏎ Fix these bugs: ⏎  ⏎ Fix https://github.com/sgl-project/sglang/issues/19081  ⏎ Fix https://github.com/sgl-project/sglang/issues/18989  ⏎ Fix https://github.com/sgl-project/sglang/issues/18980 ⏎  ⏎ , this version include this commit: https://github.com/flashinfer-ai/flashinfer/pull/2726 ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/s …[truncated]

### L2-9eb75211b1  (L2, 2026-04-01, sha 9eb75211b166, PR #21198)
TITLE: style refinement for hisparse (#21198)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/jit_kernel/csrc/hisparse.cuh (+76/-36); python/sglang/srt/managers/hisparse_coordinator.py (+8/-8); python/sglang/srt/managers/schedule_batch.py (+1/-1); python/sglang/srt/managers/scheduler.py (+29/-24); python/sglang/srt/server_args.py (+23/-0)
LABELS: run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎ 3. Trigge …[truncated]

### L2-5e12c4e08e  (L2, 2026-04-01, sha 5e12c4e08ec3, PR #21783)
TITLE: [DSA] Support trtllm sparse mla kernel for prefill batches  (#21783)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+9/-0); python/sglang/srt/server_args.py (+0/-11); python/sglang/test/run_eval.py (+3/-3)
BODY: ## Motivation ⏎  ⏎ Depends on flashinfer v0.6.7 #21422 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.gi …[truncated]

### L2-ed427e1299  (L2, 2026-04-01, sha ed427e1299d5, PR #21463)
TITLE: Migrate all callers from /get_server_info to /server_info (#21463)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/advanced_features/server_arguments.md (+1/-1); docs/advanced_features/sgl_model_gateway.md (+2/-2); docs/basic_usage/native_api.ipynb (+2/-2); docs/developer_guide/bench_serving.md (+1/-1); python/sglang/bench_serving.py (+2/-2); python/sglang/lang/backend/runtime_endpoint.py (+2/-2); python/sglang/profiler.py (+1/-1); python/sglang/test/bench_one_batch_server_internal.py (+2/-2); python/sglang/test/kits/cache_hit_kit.py (+1/-1); python/sglang/test/kl_test_utils.py (+2/-2); (+38 more)
LABELS: documentation, amd, deepseek, speculative-decoding, hicache, npu, run-ci, model-gateway
ISSUES: #21054 Deprecate and remove /get_server_info endpoint
BODY: ## Motivation ⏎  ⏎ Fix #21054. ⏎  ⏎ The `/server_info` endpoint was introduced as the canonical replacement for `/get_server_info`. The server-side deprecation wrapper (with warning log) is already in place. This PR migrates all remaining callers to use `/server_info`. ⏎  ⏎ ## Modifications ⏎  ⏎ - Update all client-side HTTP calls from `/get_server_info` to `/server_info` across: ⏎   - Python SDK (`runtime_endpoint.py`) ⏎   - Benchmarking scripts (`bench_serving.py` …[truncated]

### L2-fbc1f92453  (L2, 2026-04-02, sha fbc1f924534a, PR #21914)
TITLE: [DSA] Set trtllm kernels as nsa default for Blackwell (#21914)
SOURCES: release_notes
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+2/-7)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-c7d03a6215  (L2, 2026-04-02, sha c7d03a6215f1, PR #21922)
TITLE: Revert "Rollback flashmla to older version [1/2]" (#21922)
SOURCES: path_core, subject_keyword, symbol_pickaxe, dependency_pin, corpus:confirmed-reverts
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: sgl-kernel/cmake/flashmla.cmake (+47/-11); sgl-kernel/python/sgl_kernel/flash_mla.py (+9/-0)
LABELS: sgl-kernel
DEEP_STUDY: deep-study revert record: reland of PR(s) 21430 reason=unstated
BODY: Reverts sgl-project/sglang#21430

### L2-83c3158014  (L2, 2026-04-02, sha 83c315801474, PR #20648)
TITLE: [CI] Add Llama 3.1 8B Instruct FP4 CI test on SM120 (#20648)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/quant/test_nvfp4_gemm_sm120.py (+71/-0)
LABELS: blackwell, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Improves `stage-b-test-small-1-gpu` (SM120) coverage for quantized model tests. ⏎ One of many addressing #20600. ⏎  ⏎ Before this change, we already had: ⏎ - broader FP4 coverage on Blackwell in multi-GPU tests ⏎ - FP8/MoE/MLA accuracy tests in other suites and on other runners ⏎  ⏎ Missing single-5090 coverage for these tests. This PR addresses coverage for FP4-quantized models in particular. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Add per-PR CI te …[truncated]

### L2-51ad717089  (L2, 2026-04-02, sha 51ad717089ec, PR #20717)
TITLE: [CI] Add Per-Tensor, Blockwise FP8 Tests on SM120 (#20717)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/quant/test_fp8_gemm_sm120.py (+86/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Improves `stage-b-test-small-1-gpu` (SM120) coverage for quantized model tests. ⏎ One of many addressing #20600. ⏎  ⏎ Before this change, we already had: ⏎ - broader FP4 coverage on Blackwell in multi-GPU tests ⏎ - FP8/MoE/MLA accuracy tests in other suites and on other runners ⏎  ⏎ Missing single-5090 coverage for these tests. This PR addresses coverage for FP8-quantized models in particular. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Add per-PR CI te …[truncated]

### L2-939cf398a9  (L2, 2026-04-02, sha 939cf398a9ec, PR #17985)
TITLE: [MUSA][9/N] Add FA3 attention backend support through MATE (MUSA AI Tensor Engine) (#17985)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+16/-7); python/pyproject_other.toml (+3/-0); python/sglang/srt/configs/model_config.py (+6/-0); python/sglang/srt/environ.py (+3/-0); python/sglang/srt/hardware_backend/musa/__init__.py (+1/-0); python/sglang/srt/hardware_backend/musa/attention/__init__.py (+14/-0); python/sglang/srt/hardware_backend/musa/attention/flash_attention.py (+254/-0); python/sglang/srt/layers/attention/flashattention_backend.py (+230/-39); python/sglang/srt/server_args.py (+8/-0)
LABELS: dependencies, run-ci, mthreads
DEEP_STUDY: deep-study: this PR was reverted by PR 22002 (confirmed_revert, reason=ci_or_test_failure) || deep-study performance PR (new_kernel_or_fusion)
BODY: ### Motivation ⏎ This PR is the 9th in a series of pull requests (tracked in https://github.com/sgl-project/sglang/issues/16565) to add full support for [Moore Threads](https://en.mthreads.com/) GPUs, leveraging MUSA (Meta-computing Unified System Architecture) to accelerate LLM inference. ⏎  ⏎ ### Modifications ⏎ This commit adds support for the fa3 attention backend powered by [MATE (MUSA AI Tensor Engine)](https://github.com/MooreThreads/mate). ⏎  …[truncated]

### L2-efa7b2d5d3  (L2, 2026-04-02, sha efa7b2d5d358, PR #22002)
TITLE: Revert "[MUSA][9/N] Add FA3 attention backend support through MATE (MUSA AI Tensor Engine)" (#22002)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+7/-16); python/pyproject_other.toml (+0/-3); python/sglang/srt/configs/model_config.py (+0/-6); python/sglang/srt/environ.py (+0/-3); python/sglang/srt/hardware_backend/musa/__init__.py (+0/-1); python/sglang/srt/hardware_backend/musa/attention/__init__.py (+0/-14); python/sglang/srt/hardware_backend/musa/attention/flash_attention.py (+0/-254); python/sglang/srt/layers/attention/flashattention_backend.py (+39/-230); python/sglang/srt/server_args.py (+0/-8)
LABELS: dependencies
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 17985 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#17985 ⏎ Ref: https://github.com/sgl-project/sglang/actions/runs/23928333410/job/69789912493

### L2-7431db7392  (L2, 2026-04-03, sha 7431db7392fe, PR #21511)
TITLE: [AMD] Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 with TileLang backend (#21511)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters, L2.pool.mla_token_kv
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+79/-18); docker/rocm.Dockerfile (+1/-1); python/sglang/srt/layers/attention/nsa/tilelang_kernel.py (+307/-42); python/sglang/srt/mem_cache/memory_pool.py (+32/-15); python/sglang/srt/mem_cache/utils.py (+87/-0); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+11/-1)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 with TileLang backend. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Upgraded TileLang to [a55a823](https://github.com/tile-ai/tilelang/commit/a55a82302bf7f3c5af635b5c9146f728185cc900) to enable FP8 gemm support on AMD GPUs. ⏎ - Added an FP8 attention kernel, `sparse_mla_fwd_decode_partial_fp8`, while reusing the existing reduce kernel. ⏎ - Fused quantization path: ⏎   - On MI35 …[truncated]

### L2-cd75d54fc5  (L2, 2026-04-03, sha cd75d54fc573, PR #21987)
TITLE: [Bugfix] Fix CUDA graph replay issues in trtllm_mla draft_extend (#21987)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+13/-13); .github/workflows/nightly-test-nvidia.yml (+2/-1)
LABELS: blackwell, run-ci
DEEP_STUDY: deep-study correctness case sglang:cd75d54fc5: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Fix the test failure here https://github.com/sgl-project/sglang/actions/runs/23877953915/job/69624981292 ⏎ Same root cause and fix as PR #19807  ⏎  ⏎ **Root cause** ⏎  ⏎ `cu_seqlens_q` tensor layout differs in capture and replay path ⏎  ⏎ In `init_forward_metadata_capture_cuda_graph`, `cu_seqlens_q` is built with a **uniform stride** via `torch.arange`. ⏎ In `init_forward_metadata_replay_cuda_graph`, the replay path was building `cu_ …[truncated]

### L2-84118acf50  (L2, 2026-04-03, sha 84118acf50b3, PR #22009)
TITLE: chore: bump sglang-kernel version to 0.4.1 (#22009)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sglang-kernel` version to `0.4.1` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.4.1` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the `sglang-kernel` dependency are updat …[truncated]

### L2-8cb337c8ea  (L2, 2026-04-03, sha 8cb337c8ea48, PR #21906)
TITLE: [Bugfix] Temporarily skip TRTLLM attention on (G)B300 (SM103) to avoid high-concurrency hang (#21906)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+42/-24); python/sglang/srt/server_args.py (+27/-11); python/sglang/srt/utils/common.py (+17/-0)
LABELS: run-ci
ISSUES: #21904 [Bug] Any model hangs at high concurrency on (G)B300 (SM103) with TRTLLM attention
DEEP_STUDY: deep-study: this PR was reverted by PR 22098 (confirmed_revert, reason=build_or_dependency)
BODY: ## Summary ⏎  ⏎ Temporary workaround until FlashInfer fixes TRTLLM attention on SM103 (flashinfer-ai/flashinfer#2939). TRTLLM attention auto-selection is now gated on exact SM100 instead of the SM10X family. SM100 is unaffected and continues using TRTLLM attention as before. ⏎  ⏎ Fixes #21904

### L2-990c7590b8  (L2, 2026-04-03, sha 990c7590b835, PR #21280)
TITLE: [RL] Support mxfp8 DeepSeek V3 (#21280)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+12/-7); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+86/-38); python/sglang/srt/layers/quantization/fp8.py (+7/-0)
LABELS: high priority, deepseek, run-ci
BODY: ## Motivation ⏎ @humansand ⏎  ⏎ Support Blackwell mxfp8 DeepSeek RL. ⏎  ⏎ Since the `kv_b_proj` can have different contraction axis in absorbed vs non-absorbed MLA mode while mxfp8 is 1d quantization, for better train-inference consistency and to avoid requantization behavior I have decided to keep it always bf16. For DeepSeek V3, the size of bf16 `kv_b_proj` is `32768 x 512 x 2bytes x 61layers = 1.90625gb`, not a big overhead. ⏎  ⏎  ⏎ ## Modifications ⏎  …[truncated]

### L2-46bf19cdab  (L2, 2026-04-04, sha 46bf19cdab3b, PR #22097)
TITLE: chore: bump flashinfer version to 0.6.7.post2 (#22097)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the flashinfer version to `0.6.7.post2` across all relevant files. ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎ - python/sglang/srt/utils/common.py ⏎  ⏎ 🤖 Generated with GitHub Actions

### L2-bf984ae65d  (L2, 2026-04-04, sha bf984ae65d5d, PR #22098)
TITLE: Revert "[Bugfix] Temporarily skip TRTLLM attention on (G)B300 (SM103) to avoid high-concurrency hang" (#22098)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+24/-42); python/sglang/srt/server_args.py (+11/-27); python/sglang/srt/utils/common.py (+0/-17)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 21906 reason=build_or_dependency
BODY: Reverts sgl-project/sglang#21906 ⏎  ⏎ This temporary fix can be reverted due to release of flashinfer v0.6.7.post2 (#22097)

### L2-dd49127fe6  (L2, 2026-04-04, sha dd49127fe612, PR #21213)
TITLE: [AMD]: Support MLA with nhead<16 and FP8 KV cache for TP=8 (Kimi K2.5… (#21213)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+76/-62); test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py (+3/-12); test/registered/amd/test_kimi_k25_mxfp4.py (+2/-9)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Support AITER MLA for num_heads < 16 (e.g., TP=8 with Kimi K2.5 giving 8 heads/rank). Uses head-repeat to expand to 16 heads before calling the AITER MLA decode kernel, then contracts the output back. This reuses the existing optimized gqa_ratio=16 ASM kernel without requiring new kernel variants. ⏎ 2. Relax head-count assertion to accept num_heads of 4, 8, or any multiple of 16 in [16, 128]. Previously  …[truncated]

### L2-5a35316417  (L2, 2026-04-05, sha 5a3531641735, PR #21405)
TITLE: Enable IndexCache for DeepSeek V3.2 (#21405)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+27/-13); python/sglang/srt/models/deepseek_nextn.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+51/-6); test/registered/8-gpu-models/test_deepseek_v32_indexcache.py (+117/-0)
LABELS: high priority, deepseek, run-ci
ISSUES: #21286 [Feature] Implement IndexCache for GLM-5/DeepSeek V3.2
BODY: ## Motivation ⏎  ⏎ fix https://github.com/sgl-project/sglang/issues/21286 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ * Port https://github.com/THUDM/IndexCache ⏎ * add ut for deepseek-v3.2 ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ ``` ⏎ python3 -m sglang.launch_server   --model-path /ssd/hf_models/DeepSeek-V3.2-Exp --tp 8 --mem-fraction-static=0.9 --tool-call-parser deepseekv32  --reasoning-parser deepseek-v3 --json-model-override-args '{"index_topk_freq": 4}' ⏎ ``` ⏎ ``` ⏎ lm_eval -- …[truncated]

### L2-e835601fb7  (L2, 2026-04-05, sha e835601fb72e, PR #22143)
TITLE: Cache gfx95 quant format detection in DeepseekV2DecoderLayer (#22143)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+17/-28)
LABELS: deepseek, run-ci
BODY: ## Summary ⏎ - Extract the repeated quant_format detection logic from forward() into a dedicated _detect_gfx95_quant_format() method ⏎ - Cache the result so it is computed once instead of on every forward call ⏎ - On non-gfx95 platforms, the value is set to empty string immediately in __init__ with zero runtime overhead ⏎ - On gfx95, it is lazily computed on the first forward call (after weights are loaded) and cached ⏎  ⏎ ## Test plan

### L2-b311db2e49  (L2, 2026-04-05, sha b311db2e4994, PR #22179)
TITLE: [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation (#22179)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/basic_usage/deepseek_v32.md (+11/-12)
LABELS: documentation, deepseek
BODY: ## Summary ⏎  ⏎ Remove skip-softmax section (I think it's for dense attention only, not DSA, per flashinfer constraint below) and improve docs ⏎  ⏎ https://github.com/flashinfer-ai/flashinfer/blob/v0.6.7.post2/flashinfer/mla.py#L730-L731 ⏎  ⏎ ```python ⏎ if skip_softmax_threshold_scale_factor is not None and sparse_mla_top_k != 0: ⏎     raise ValueError("skip_softmax is not supported for sparse MLA") ⏎ ``` ⏎  ⏎ cc @Fridge003 for double check

### L2-dc125afffb  (L2, 2026-04-06, sha dc125afffbd5, PR #21921)
TITLE: Add staging buffer CI test and documentation for heterogeneous TP (#21921)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/advanced_features/pd_disaggregation.md (+52/-0); docs/references/environment_variables.md (+9/-0); python/sglang/srt/disaggregation/common/staging_buffer.py (+1/-1); python/sglang/srt/disaggregation/common/staging_handler.py (+4/-3); python/sglang/srt/disaggregation/decode.py (+9/-1); python/sglang/srt/disaggregation/fake/conn.py (+1/-0); python/sglang/srt/disaggregation/prefill.py (+5/-0); test/registered/distributed/test_disaggregation_different_tp.py (+162/-0)
LABELS: documentation, high priority, run-ci
BODY: - Add e2e test for disaggregation with staging buffer enabled (test_disaggregation_different_tp_staging.py), registered in stage-c-test-8-gpu-h20 suite. Covers MLA and MHA models with both prefill-larger and decode-larger TP configurations. ⏎ - Document staging buffer usage in pd_disaggregation.md: when to enable, environment variables, and usage example. ⏎ - Add staging buffer environment variables to environment_variables.md. ⏎  ⏎  ⏎  ⏎ ## Motivation …[truncated]

### L2-2813cb6d9a  (L2, 2026-04-06, sha 2813cb6d9a5b, PR #21952)
TITLE: [New Model] Gemma 4 (#21952)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: .codespellrc (+1/-1); benchmark/kernels/fused_moe_triton/common_utils.py (+4/-0); benchmark/mmlu/bench_hf.py (+151/-0); python/sglang/lang/chat_template.py (+17/-8); python/sglang/srt/configs/model_config.py (+18/-2); python/sglang/srt/entrypoints/openai/serving_chat.py (+13/-2); python/sglang/srt/function_call/function_call_parser.py (+2/-0); python/sglang/srt/function_call/gemma4_detector.py (+445/-0); python/sglang/srt/layers/attention/triton_backend.py (+85/-22); python/sglang/srt/layers/attention/triton_ops/prefill_attention.py (+4/-2); (+25 more)
LABELS: quant, Multi-modal, run-ci
BODY: ## Motivation ⏎  ⏎ Add Gemma 4 model support to SGLang. Gemma 4 is Google's next-generation family of open models featuring Dense and MoE architectures, multimodal support (text, image, audio), hybrid reasoning, and native tool calling. ⏎  ⏎ **Supported Models:** ⏎  ⏎ | Model | Architecture | Parameters | ⏎ |-------|-------------|------------| ⏎ | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | Dense | ~2B | ⏎ | [google/gemma-4-E4B-it](http …[truncated]

### L2-98f38b14df  (L2, 2026-04-07, sha 98f38b14df80, PR #21983)
TITLE: Add registration API for external linear attention backend (#21983)
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+16/-3); python/sglang/srt/configs/linear_attn_model_registry.py (+72/-0); python/sglang/srt/layers/attention/triton_backend.py (+1/-0); python/sglang/srt/managers/scheduler.py (+5/-0); python/sglang/srt/model_executor/model_runner.py (+18/-1); python/sglang/srt/server_args.py (+9/-0); test/registered/unit/configs/test_linear_attn_model_registry.py (+161/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ SGLang currently hardcodes hybrid model support (GDN, KDA, Mamba2, Lightning) across 5 core files via `isinstance` checks and architecture name lists. Adding a new linear attention hybrid model requires modifying all 5 files: ⏎  ⏎ - `model_runner.py` — `mambaish_config` property ⏎ - `attention_registry.py` — backend dispatch in `attn_backend_wrapper` ⏎ - `scheduler.py` — `is_hybrid_ssm` flag for MambaRadixCache ⏎ - `server_args.py` —  …[truncated]

### L2-0c204fbd57  (L2, 2026-04-07, sha 0c204fbd57a0, PR #21932)
TITLE: [HiSparse] Optimize the scheduling of decode backup. (#21932)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/hisparse_coordinator.py (+36/-9); python/sglang/srt/model_executor/model_runner.py (+6/-0)
LABELS: hicache, ready-to-merge, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ <img width="1211" height="749" alt="image" src="https://github.com/user-attachments/assets/ca0ba80d-a018-4098-a492-f834463bea81" /> ⏎ In overlap scheduling, the backup of decode tokens within the `prepare_for_decode` method currently requires waiting for the forward stream to complete, resulting in significant CPU bubbles. This PR shifts the timing of the backup operation to occur at the conclusion of the forward pass, and verifies  …[truncated]

### L2-1a8eb890f6  (L2, 2026-04-07, sha 1a8eb890f625, PR #20796)
TITLE: Kernels community fa3 (#20796)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.backend.sparse_mla_adapters
FILES: docker/Dockerfile (+7/-0); python/pyproject.toml (+4/-0); .gitignore (+1/-0); docs/references/environment_variables.md (+2/-0); python/sglang/jit_kernel/flash_attention.py (+286/-0); python/sglang/jit_kernel/flash_attention_v3.py (+222/-0); python/sglang/jit_kernel/flash_attention_v4.py (+0/-1); python/sglang/jit_kernel/tests/test_flash_attention_3.py (+1373/-0); python/sglang/jit_kernel/tests/test_flash_attention_4.py (+3/-1); python/sglang/multimodal_gen/runtime/layers/attention/backends/flash_attn.py (+4/-18); (+10 more)
LABELS: documentation, dependencies, Multi-modal, run-ci, piecewise-cuda-graph, diffusion, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Support flash-attn-3 from kernels community ⏎  ⏎ ## Modifications ⏎  ⏎ - Use the fa3 kernel from kernels community ⏎ - Docker image ⏎ - CI ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ 100%|████████████████████████████████████████████| 100/100 [00:05<00:00, 17.00it/s] ⏎ Accuracy: 0.950 ⏎ Invalid: 0.000 ⏎ Latency: 5.882 s ⏎ Output throughput: 2046.968 token/s ⏎ ``` ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Onca …[truncated]

### L2-3148742ddb  (L2, 2026-04-07, sha 3148742ddb2c, PR #22145)
TITLE: [Disagg][NIXL] Fix heterogeneous TP KV transfer for non-MLA models (same logic with mooncake, Step 1/2 for Qwen3.5 support) (#22145)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/nixl/conn.py (+20/-8)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ NIXL disaggregated serving with heterogeneous TP (prefill TP ≠ decode TP) on non-MLA models hangs indefinitely due to two bugs in `nixl/conn.py`: ⏎  ⏎ 1. **Notification key collision**: `_process_kvcache_transfer` uses `pp_rank` in RDMA notification tags. With PP=1, all prefill ranks share `pp_rank=0`, so `TransferStatus.received_kvs_per_pp` only records one key while `num_pp_ranks_expected > 1` → `is_done()` never returns `True` → …[truncated]

### L2-6131fb5882  (L2, 2026-04-08, sha 6131fb588273, PR #22024)
TITLE: [NPU] enable mla prepare fused kernel only when being mla attn (#22024)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+2/-2)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ MLAPO is only applicable to MLA-based models. Currently, when it is used together with an Eagle draft model, the draft model fails to save its KV cache correctly. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎ ```sh ⏎ python3 -m sglang.test.few_shot_gsm8k \ ⏎     --num-questions 200 \ ⏎     --num-shots 5 \ ⏎     --data-path /xxx/gsm8k.jsonl \ ⏎     --max-new-tokens 512 \ ⏎     --parallel 200 \ ⏎     --host http://0.0.0.0 \ ⏎     --p …[truncated]

### L2-df3275bd6c  (L2, 2026-04-08, sha df3275bd6c55, PR #22382)
TITLE: chore: bump flashinfer version to 0.6.7.post3 (#22382)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the flashinfer version to `0.6.7.post3` across all relevant files. ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎ - python/sglang/srt/utils/common.py ⏎  ⏎ 🤖 Generated with GitHub Actions

### L2-1e3f6ebea6  (L2, 2026-04-08, sha 1e3f6ebea685, PR #22384)
TITLE: [core] Extract pool sizing logic to pool_configurator.py (#22384)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/model_runner.py (+1/-1); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+31/-166); python/sglang/srt/model_executor/pool_configurator.py (+172/-0)
LABELS: high priority, run-ci
BODY: ## Summary ⏎  ⏎ - Extract `get_cell_size_per_token` and `resolve_hybrid_swa_tokens` from `model_runner_kv_cache_mixin.py` to `pool_configurator.py` as standalone functions ⏎ - Extract `_profile_available_bytes` from `profile_max_num_token` (returns bytes instead of tokens) ⏎ - Rename `_resolve_token_capacity` → `_apply_token_constraints` for clarity ⏎ - Fix missing page alignment in `_apply_token_constraints` ⏎  ⏎ Pure code movement, zero behavior change. Prep …[truncated]

### L2-9d905efa2c  (L2, 2026-04-09, sha 9d905efa2c0d, PR #22322)
TITLE: [Docker] Fix Trivy CVEs, cubin download 403s, and kernels command order (#22322)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+35/-3)
BODY: ## Summary ⏎ - **Fix Trivy-reported CVEs**: Add targeted `apt-get install --only-upgrade` layers to patch vulnerable OS packages. Uses `--only-upgrade` to avoid accidentally upgrading NVIDIA CUDA packages. ⏎ - **Remove redundant flashinfer cubin download**: The `python3 -m flashinfer --download-cubin` step was downloading ~10,564 cubins individually from NVIDIA's CDN, causing 403 rate-limiting errors that fail the Docker release CI. The `flashinfer_c …[truncated]

### L2-28ef6de091  (L2, 2026-04-09, sha 28ef6de09106, PR #22323)
TITLE: [Lora] Lora quat info re-factor and support deepseekv3 mla lora (#22323)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/base_config.py (+14/-0); python/sglang/srt/layers/quantization/blockwise_int8.py (+10/-7); python/sglang/srt/layers/quantization/fp8.py (+21/-20); python/sglang/srt/layers/quantization/moe_wna16.py (+13/-11); python/sglang/srt/layers/quantization/unquant.py (+9/-6); python/sglang/srt/layers/quantization/w8a8_fp8.py (+10/-7); python/sglang/srt/layers/quantization/w8a8_int8.py (+13/-10); python/sglang/srt/lora/backend/chunked_backend.py (+2/-1); python/sglang/srt/lora/backend/torch_backend.py (+7/-2); python/sglang/srt/lora/backend/triton_backend.py (+2/-1); (+6 more)
LABELS: high priority, quant, lora, deepseek, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - MoE quant-info refactor ⏎ Extracts get_triton_quant_info() into each quantization method (FP8, INT8, WNA16, etc.) so FusedMoEWithLoRA correctly receives quantization scales/flags, enabling LoRA on quantized MoE models. ⏎  ⏎ - DeepSeek-V3 MLA LoRA support ⏎ Adds ReplicatedLinearWithLoRA to handle the fused q_a_proj + kv_a_proj_with_mqa projection unique to MLA. Since the two sub-projections have unequal outpu …[truncated]

### L2-de441ac6bb  (L2, 2026-04-09, sha de441ac6bbb9, PR #22389)
TITLE: [core] Introduce `MemoryPoolConfigurator` class hierarchy (#22389)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/tp_worker.py (+1/-1); python/sglang/srt/model_executor/model_runner.py (+1/-1); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+27/-98); python/sglang/srt/model_executor/pool_configurator.py (+248/-132)
LABELS: high priority, run-ci
BODY: ## Summary ⏎  ⏎ - Introduce `MemoryPoolConfigurator` base class with unified coeff+bias interface (`calculate_pool_sizes` / `calculate_pool_sizes_from_max_tokens`) ⏎ - Add `DefaultPoolConfigurator` for MHA/MLA/NSA/FP4 — absorbs `get_cell_size_per_token`, num_layers deduction, DFLASH scaling ⏎ - Add `HybridSWAPoolConfigurator` for Gemma2/Command-R/MiMo — absorbs `resolve_hybrid_swa_tokens` with full/swa pool splitting ⏎ - Add `create_memory_pool_configurato …[truncated]

### L2-aa103eab8d  (L2, 2026-04-09, sha aa103eab8df4, PR #22160)
TITLE: [Docker] Optimize Dockerfile for BuildKit layer caching (#22160)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+328/-162)
BODY: ## Summary ⏎  ⏎ - Restructure Dockerfile into parallel multi-stage builds (`torch_deps`, `deepep_builder`, `flashinfer_cache`, `devtools_builder`) so independent work executes concurrently via BuildKit ⏎ - Install dependencies from `pyproject.toml` only (with constraints file), push source `COPY` and editable install to the very last stage -- any Python source change now only invalidates the final layers ⏎ - Build DeepEP as a wheel in an isolated stage,  …[truncated]

### L2-6d79c60995  (L2, 2026-04-09, sha 6d79c6099545, PR #22381)
TITLE: [Lora] Lora kimi support (#22381)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+13/-0); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py (+13/-10); python/sglang/srt/lora/layers.py (+8/-1); python/sglang/srt/lora/lora_manager.py (+4/-1); test/registered/lora/test_lora_kimi_k25_logprob_diff.py (+150/-0)
LABELS: high priority, quant, lora, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-6af34b95b6  (L2, 2026-04-10, sha 6af34b95b653, PR #21104)
TITLE: perf: precompute FA3 scheduler_metadata to eliminate per-layer prepare_varlen_num_blocks (#21104)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+107/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Call `get_scheduler_metadata` once per batch in decode metadata init (including CUDA graph capture/replay paths) and pass the result to `flash_attn_with_kvcache` so FA3 skips the internal `prepare_varlen_num_blocks` kernel on every layer. ⏎  ⏎ This eliminates 63 redundant GPU kernel calls per decode step (64 layers → 1 call via `get_scheduler_metadata`), matching what vLLM does.  ⏎  ⏎ ## Changes (1 file, Python only) ⏎  ⏎ **`flashattentio …[truncated]

### L2-f7a1740101  (L2, 2026-04-10, sha f7a174010127, PR #22051)
TITLE: [MUSA][9/N] Add FA3 attention backend support through MATE (MUSA AI Tensor Engine) (#22051)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+23/-9); python/pyproject_other.toml (+4/-1); python/sglang/srt/configs/model_config.py (+6/-0); python/sglang/srt/environ.py (+3/-0); python/sglang/srt/hardware_backend/musa/__init__.py (+1/-0); python/sglang/srt/hardware_backend/musa/attention/__init__.py (+3/-0); python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py (+913/-0); python/sglang/srt/server_args.py (+5/-1)
LABELS: dependencies, run-ci, mthreads, jit-kernel
BODY: ## Motivation ⏎  ⏎ This PR fixes the Flash Attention backend support that was previously merged in PR #17985 but later reverted in PR #22002 due to a bug. The original commit 2373552 caused CI failures (see [failed CI job](https://github.com/sgl-project/sglang/actions/runs/23928333410/job/69789912493)). ⏎  ⏎ Previously, the MUSA-adapted flash attention implementation had a bug in the `_forward_extend_impl` method. The code was missing a proper mechan …[truncated]

### L2-1c76f322df  (L2, 2026-04-10, sha 1c76f322df5c, PR #20977)
TITLE: [HiCache] Add CP support for HiCache (#20977)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/cache_controller.py (+6/-0); python/sglang/srt/managers/scheduler.py (+2/-0); python/sglang/srt/mem_cache/cache_init_params.py (+3/-0); python/sglang/srt/mem_cache/hicache_storage.py (+2/-0); python/sglang/srt/mem_cache/hiradix_cache.py (+6/-0); python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py (+13/-5)
LABELS: hicache, run-ci
BODY: ## Motivation ⏎ This PR is mostly for Qwen3 CP + Hicache. MLA models + Hicache will reuse cp 0's data for all ranks, so we don't need to distinguish the key. ⏎ CC: @whybeyoung  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull- …[truncated]

### L2-19cb918653  (L2, 2026-04-11, sha 19cb91865322, PR #21166)
TITLE: [Not-Merge][AMD] GLM-5 performance optimization (#21166)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+0/-3); python/sglang/srt/layers/attention/nsa_backend.py (+48/-5)
LABELS: amd
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎ Optimize GLM-5 model performance. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ 1. Pad IO tensor shape in framework level for Aiter sparse attn when nhead=8 per device (eg. glm-5 TP8). ⏎ 2. Delete 3 hardcode parameters in Aiter deepgemm_fp8_paged_mqa_logits api, and kernel will configure optimized parameters internally. ⏎ 3. Tune GLM-5 Moe gate/router gemm_a16w16_atomic triton kernel with shape (N=256, K=6144) in Aiter.https://github.com/ROCm/aiter/pull/ …[truncated]

### L2-bcc0c65aa8  (L2, 2026-04-12, sha bcc0c65aa836, PR #22372)
TITLE: [DSA] Hopper FP8 FlashMLA KV padding (#22372)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/nsa_backend.py (+39/-4); python/sglang/srt/server_args.py (+2/-2); docs/basic_usage/deepseek_v32.md (+2/-2)
LABELS: documentation, deepseek
BODY: ## Summary ⏎  ⏎ Adds q-head padding for `flashmla_kv` to support pure-TP configurations. FlashMLA requires q heads to be 64 or 128, but with tensor parallelism the per-GPU head count gets divided (e.g., GLM-5's 64 heads / TP8 = 8 heads; DeepSeek V3.2's 128 heads / TP8 = 16 heads). When using TP+DP attention, the effective attention TP size is 1, so no padding is needed. We pad to the next supported variant (64/128) for pure-TP cases and slice the o …[truncated]

### L2-701a0e0c25  (L2, 2026-04-12, sha 701a0e0c2551, PR #22491)
TITLE: [CI/Docker] Clean up redundant flashinfer cubin downloads (#22491)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-4); scripts/ci/cuda/ci_download_flashinfer_cubin.sh (+0/-62); scripts/ci/cuda/ci_install_dependency.sh (+1/-3)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ Follow-up to #22322.  ⏎  ⏎ Cleans up redundant flashinfer cubin download steps across the CI workflows and the main Dockerfile that were missed in the previous PR.  ⏎  ⏎ cc @Kangyan-Zhou

### L2-90ef8ce54d  (L2, 2026-04-13, sha 90ef8ce54de7, PR #22653)
TITLE: [Docker] Remove flashinfer cache copy (#22653)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-2)
BODY: ## Summary ⏎  ⏎ Fix https://github.com/sgl-project/sglang/actions/runs/24320223003/job/71004907738#step:8:8520. After #22491 removed the manual flashinfer cubin download steps, `/root/.cache/flashinfer` is no longer populated. ⏎  ⏎ cc @Kangyan-Zhou

### L2-934e19a610  (L2, 2026-04-13, sha 934e19a6104b, PR #21367)
TITLE: [CPU] Fix argument issues in qkv_proj_with_rope_fused_weight and bmm… (#21367)
SOURCES: path_core
ARTIFACT_HINTS: L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py (+2/-2); python/sglang/srt/model_executor/cpu_graph_runner.py (+1/-0)
LABELS: run-ci
BODY: …_cpu ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎ This PR fixes the following issue encountered when running DeepSeek-V3.1-Terminus: `w_scale` is only required for FP8 in the kernels `qkv_proj_with_rope_fused_weight` and `bmm_cpu`, so we updated the frontend logic to pass `w_scale=None` for other data types. ⏎  ⏎  ⏎ command: `python -m sglang.launch_server --model  IntervitensInc/DeepSeek-V3.1-Terminus-Channel-int8 --trust-remote-code --disable-overlap-schedule --dev …[truncated]

### L2-e15401ee0e  (L2, 2026-04-14, sha e15401ee0eb4, PR #22537)
TITLE: Add runai-model-streamer into Python packages installed in Dockerfile and fix NotADirectoryError Docker regression (#22537)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+9/-8)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Currently, when using `--load-format=runai_streamer` to load a model from Google Cloud Storage (`gs://...`) using the official SGLang Docker image, the server crashes with the following error: ⏎  ⏎ ```bash ⏎   File "/usr/local/lib/python3.12/dist-packages/runai_model_streamer/s3_utils/s3_utils.py", line 137, in gcs_pull_files ⏎     raise ImportError("GCS files module not found. Please install the required package.") ⏎ ImportError: GCS …[truncated]

### L2-adb310b976  (L2, 2026-04-14, sha adb310b976d6, PR #22820)
TITLE: Cleanup server_args.py and minor code tidying (#22820)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py (+1/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-0); python/sglang/srt/managers/scheduler.py (+23/-14); python/sglang/srt/server_args.py (+37/-40)
LABELS: run-ci
BODY: ## Summary ⏎ - Inline `MAMBA_SSM_DTYPE_CHOICES` directly into `add_cli_args` and remove the unused constant and `add_mamba_ssm_dtype_choices()` function ⏎ - Reorder constants and helper functions in `server_args.py` for better grouping ⏎ - Move import to top-level in `fused_moe.py` ⏎  ⏎ ## Test plan ⏎ - No behavioral changes — pure cleanup (moves, inlining, import reorder) ⏎ - CI should pass as-is

### L2-113d654152  (L2, 2026-04-15, sha 113d654152cd, PR #22723)
TITLE: [Fix] Fix accuracy bug in Flashmla sparse MLA kernel (#22723)
SOURCES: path_core, subject_keyword, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: sgl-kernel/cmake/flashmla.cmake (+1/-1)
LABELS: sgl-kernel, run-ci
ISSUES: #21291 [Bug] GLM-5 accuracy drop on B200 with flash_mla_with_kvcache kernel
DEEP_STUDY: deep-study correctness case sglang:113d654152: class=hardware_compiler_specific; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Motivation ⏎  ⏎ Close #21291 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and ot …[truncated]

### L2-e7ad7c587a  (L2, 2026-04-16, sha e7ad7c587a35, PR #22490)
TITLE: [EPD][VLM] Support Kimi VL EPD (#22490)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/encode_receiver.py (+44/-13); python/sglang/srt/disaggregation/encode_server.py (+70/-17); python/sglang/srt/models/kimi_vl.py (+23/-8); python/sglang/srt/multimodal/processors/kimi_common.py (+113/-0); python/sglang/srt/multimodal/processors/kimi_k25.py (+7/-63); python/sglang/srt/multimodal/processors/kimi_vl.py (+11/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Kimi VL (kimi-vl-a3b) shares the same MoonViT vision tower as Kimi K2.5 but was not yet supported under the EPD (Encode-Prefill-Decode) disaggregation pipeline. This PR extends the existing Kimi K2.5 EPD support to cover Kimi VL, enabling disaggregated serving where the vision encoder runs on a separate encode server and the language model runs on a decode server. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ **Model (`models/kimi_vl.py`)** ⏎ -  …[truncated]

### L2-44e67c6835  (L2, 2026-04-17, sha 44e67c6835b6, PR #23009)
TITLE: Remove deprecated double sparsity feature (#23009)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+2/-9); docs/advanced_features/server_arguments.md (+0/-10); docs/platforms/ascend/ascend_npu_support_features.md (+0/-6); python/sglang/srt/layers/attention/double_sparsity_backend.py (+0/-257); python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py (+0/-1106); python/sglang/srt/mem_cache/memory_pool.py (+0/-90); python/sglang/srt/model_executor/model_runner.py (+0/-33); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+0/-15); python/sglang/srt/server_args.py (+0/-45); test/manual/double-sparsity-config-Llama-3.1-8B-Instruct.json (+0/-1); (+2 more)
LABELS: documentation, npu, run-ci
BODY: ## Summary ⏎ - Remove the deprecated double sparsity attention optimization feature entirely ⏎ - Delete implementation files: DoubleSparseAttnBackend, triton kernels, DoubleSparseTokenToKVPool ⏎ - Remove all server arguments (--enable-double-sparsity, --ds-channel-config-path, --ds-heavy-channel-num, --ds-heavy-token-num, --ds-heavy-channel-type, --ds-sparse-decode-threshold) ⏎ - Clean up references in model_runner, attention_registry, kv_cache_mixin, an …[truncated]

### L2-6ecd6f84db  (L2, 2026-04-19, sha 6ecd6f84dbf9, PR #23119)
TITLE: [CI] Add per-job uv venv isolation and upgrade CI version to Cuda 13 (#23119)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+17/-10); .github/workflows/pr-test-multimodal-gen.yml (+5/-5); .github/workflows/pr-test-sgl-kernel.yml (+4/-4); .github/workflows/pr-test.yml (+75/-22); python/sglang/jit_kernel/tests/test_pos_enc.py (+2/-2); python/sglang/multimodal_gen/configs/models/dits/wanvideo.py (+23/-1); python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py (+1/-0); python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py (+7/-1); python/sglang/multimodal_gen/runtime/loader/fsdp_load.py (+3/-1); python/sglang/multimodal_gen/runtime/loader/transformer_load_utils.py (+78/-10); (+29 more)
LABELS: high priority, quant, dependencies, lora, Multi-modal, hicache, sgl-kernel, run-ci, diffusion, jit-kernel
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-7ca3566130  (L2, 2026-04-19, sha 7ca356613093, PR #21388)
TITLE: Multi platform Plugin (#21388)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.dispatch.server_args_defaults
FILES: docs/platforms/plugin.md (+414/-0); python/sglang/cli/serve.py (+4/-0); python/sglang/launch_server.py (+4/-0); python/sglang/srt/compilation/backend.py (+7/-1); python/sglang/srt/entrypoints/engine.py (+10/-0); python/sglang/srt/environ.py (+4/-0); python/sglang/srt/layers/utils/multi_platform.py (+22/-1); python/sglang/srt/managers/scheduler.py (+3/-0); python/sglang/srt/mem_cache/memory_pool.py (+12/-3); python/sglang/srt/model_executor/model_runner.py (+34/-10); (+12 more)
LABELS: documentation, high priority, run-ci, piecewise-cuda-graph
BODY: ## Summary ⏎  ⏎ Introduce a unified plugin framework for SGLang, inspired by vLLM's platform abstraction, enabling hardware vendors and advanced users to extend SGLang without forking or modifying the main repository. ⏎  ⏎ The framework provides two plugin types, both discovered via Python's standard setuptools `entry_points` mechanism: ⏎  ⏎ - **Platform Plugins** (`sglang.platform_plugins`): Register custom hardware platforms — device ops, KV cache po …[truncated]

### L2-5595f6e988  (L2, 2026-04-20, sha 5595f6e9888b, PR #22688)
TITLE: Fix trtllm mla chunked-prefill zero-length bug (#22291) (#22688)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+21/-1); python/sglang/jit_kernel/csrc/attention/fixup_zero_kv.cuh (+124/-0); python/sglang/jit_kernel/fixup_zero_kv.py (+44/-0); python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py (+10/-0)
LABELS: deepseek, blackwell, jit-kernel
ISSUES: #22291 [Bug] trtllm_mla giving wrong results with chunked-prefill on blackwell
DEEP_STUDY: deep-study correctness case sglang:5595f6e988: class=shape_alignment_edge; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Motivation ⏎  ⏎  ⏎ ### Background ⏎  ⏎ Issue #22291 reported significant accuracy degradation when using the `trtllm_mla` backend with chunked prefill to process multiple prompts in parallel on Blackwell GPUs. Benchmarks showed that certain sequences under `trtllm_mla` exhibited MSE values as high as 1.0–3.0, whereas the `fa4` and `flashinfer` backends produced MSE values consistently below 0.07 under identical conditions. Notably, the issue only m …[truncated]

### L2-6d47dc8f6d  (L2, 2026-04-20, sha 6d47dc8f6dc8, PR #23303)
TITLE: [CI][MLA] Enable deterministic inference for MGSM MLA FP8 test (#23303)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/mla/test_mla_fp8.py (+8/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `test/registered/mla/test_mla_fp8.py::TestMLA::test_mgsm_en` has been flaking at the 0.8 MGSM-EN threshold. A recent failing run (e.g. [actions run 24697455220](https://github.com/sgl-project/sglang/actions/runs/24697455220/job/72233745557)) scored 0.784 — ~1.6 percentage points below the threshold. The retry wrapper hit the same value, so it isn't pure RNG noise on a single run — it's server-side determinism that drifts across CI  …[truncated]

### L2-8589b92a89  (L2, 2026-04-21, sha 8589b92a891b, PR #23186)
TITLE: [AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4 (#23186)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+12/-0)
LABELS: lora, deepseek, run-ci, diffusion
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎   On ROCm targets with a BF16 DeepSeek model, the MLA QK layernorm path falls through to the PyTorch sequential ⏎   fallback instead of using aiter's fused kernel. The BF16 fused branch in forward_mla.py was gated by _use_aiter and ⏎   not _use_aiter_gfx95, which correctly excludes gfx950 from the FP8/MXFP4 quantized paths but incorrectly also ⏎   excludes it from the BF16 fused path. This PR fixes the condition so the fused BF16 ke …[truncated]

### L2-a58c7f381e  (L2, 2026-04-21, sha a58c7f381e7c, PR #23323)
TITLE: [PD] Fix clip logic when state indices lens are mismatch (#23323)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/mooncake/conn.py (+10/-6)
BODY: ## Motivation ⏎ Fix #18727. ⏎ In `MooncakeKVManager.maybe_send_extra`, the clip logic for SWA/NSA hybrid models only handled one direction of length mismatch between `prefill_state_indices` and `req.dst_state_indices`, and even that branch was effectively a no-op: ⏎ ```python ⏎ if len(prefill_state_indices) < len(req.dst_state_indices): ⏎     logger.warning(...) ⏎     prefill_state_indices = prefill_state_indices[: len(req.dst_state_indices)] ⏎ ``` ⏎ Two …[truncated]

### L2-e3782d04d2  (L2, 2026-04-21, sha e3782d04d248, PR #23139)
TITLE: fix: fallback to triton for attention-sink models (flashinfer unsupported) (#23139)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/configs/model_config.py (+21/-0); python/sglang/srt/server_args.py (+12/-4)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The FlashInfer attention backend does not accept a `sinks` kwarg, but hybrid-SWA models like `GptOssForCausalLM` and `MiMoV2FlashForCausalLM` (when `add_{swa,full}_attention_sink_bias=True`) pass per-head sink scalars to the attention call. Two auto-selection paths used to fall back to `flashinfer` without considering this, which could leave sink models on an unsupported backend. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add `ModelConfig.has_attention_ …[truncated]

### L2-887d380ace  (L2, 2026-04-22, sha 887d380acedb, PR #23270)
TITLE: [MUSA] Resolve output garbage in Context Parallel on MusaFlashAttentionBackend (#23270)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: 3rdparty/amd/wheel/sglang/pyproject.toml (+4/-1); python/pyproject_other.toml (+9/-9); python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py (+57/-50); python/sglang/srt/hardware_backend/musa/layers/utils/__init__.py (+0/-0); python/sglang/srt/hardware_backend/musa/layers/utils/cp_utils.py (+57/-0); sgl-kernel/pyproject_musa.toml (+1/-1)
LABELS: dependencies, sgl-kernel, run-ci, mthreads
BODY: ## Motivation ⏎  ⏎ Fix Context Parallel (CP) attention forward extension for the MUSA backend. The original `cp_attn_forward_extend` function from `cp_utils.py` was incompatible with the MUSA FA Attention backend, causing CP workloads to fail on MUSA devices. ⏎  ⏎ ## Modifications ⏎ - **Fix context parallel**:  Modify the original cp_attn_forward_extend function to be able to switch the backend's _current_prefix in order to obtain the correct schedule …[truncated]

### L2-fd88a1c562  (L2, 2026-04-23, sha fd88a1c5623f, PR #23382)
TITLE: [AMD] skip deterministic inference for MLA FP8 test (#23382)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/mla/test_mla_fp8.py (+17/-13)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ PR #23303 added `--enable-deterministic-inference` to `test/registered/mla/test_mla_fp8.py` to stabilize MGSM scores around the 0.8 threshold. The flag works on NVIDIA but breaks the AMD CI partition that runs the same test (`stage-b-test-1-gpu-small-amd`, partition 5). Failing run: ⏎  ⏎ - https://github.com/sgl-project/sglang/actions/runs/24713176882/job/72282791399 ⏎  ⏎ Server fails to start with: ⏎  ⏎ ``` ⏎ ValueError: Currently only …[truncated]

### L2-f3b88e080a  (L2, 2026-04-23, sha f3b88e080aeb, PR #23281)
TITLE: chore: bump flashinfer version to 0.6.8.post1 (#23281)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1)
LABELS: high priority, dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the flashinfer version to `0.6.8.post1` across all relevant files. ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎ - python/sglang/srt/utils/common.py ⏎  ⏎ 🤖 Generated with GitHub Actions

### L2-74c2e5bacd  (L2, 2026-04-23, sha 74c2e5bacd0d, PR #17946)
TITLE: [MUSA][8/N] Port CUDA kernels that are compatible with MUSA (#17946)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/allreduce/custom_all_reduce.cuh (+206/-25); sgl-kernel/csrc/common_extension_musa.cc (+246/-12); sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.mu (+529/-0); sgl-kernel/csrc/elementwise/utils.cuh (+7/-0); sgl-kernel/csrc/gemm/dsv3_fused_a_gemm.cu (+4/-0); sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu (+4/-0); sgl-kernel/csrc/gemm/per_token_group_quant_8bit_v2.cu (+9/-0); sgl-kernel/csrc/kvcacheio/transfer.cu (+2/-2); sgl-kernel/csrc/moe/moe_fused_gate_musa.cu (+840/-0); sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu (+2/-0); (+5 more)
LABELS: quant, dependencies, sgl-kernel, run-ci, mthreads
BODY: ### Motivation ⏎  ⏎ This PR continues the ongoing effort (tracked in #16565) to add full support for **Moore Threads GPUs** in SGLang by leveraging **MUSA (Meta-computing Unified System Architecture)** for LLM inference. ⏎  ⏎ The primary goal of this submission is to enable core kernel functionality on MUSA by porting CUDA kernels that are compatible with the MUSA programming model, while keeping the codebase unified across CUDA, ROCm, and MUSA backe …[truncated]

### L2-b35213be11  (L2, 2026-04-23, sha b35213be11c7, PR #22774)
TITLE: [MUSA][16/N] Add MUSA backend support for layers and DeepSeek models (V2/V3/R1) (#22774)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+6/-0); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/layers/activation.py (+13/-0); python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py (+14/-3); python/sglang/srt/layers/deep_gemm_wrapper/configurer.py (+11/-2); python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py (+3/-2); python/sglang/srt/layers/layernorm.py (+26/-1); python/sglang/srt/layers/moe/ep_moe/kernels.py (+11/-3); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+7/-4); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+9/-1); (+17 more)
LABELS: quant, deepseek, run-ci, mthreads
BODY: ## Motivation ⏎  ⏎ Enable SGLang to run DeepSeek models on Moore Threads MUSA GPUs. This PR adds MUSA backend support across the inference stack, including layers, quantization, MoE, attention, speculative decoding, and custom op registration. ⏎  ⏎ ## Modifications ⏎  ⏎ **Core Infrastructure:** ⏎  ⏎ - **Custom op registration** (`utils/common.py`): Register custom ops on the `MUSA` dispatch key in `direct_register_custom_op`; extend `get_device_sm()` to  …[truncated]

### L2-60bbb800db  (L2, 2026-04-24, sha 60bbb800db81, PR #22218)
TITLE: [Experimental] Breakable Piecewise Cuda Graph (#22218)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/radix_attention.py (+17/-3); python/sglang/srt/model_executor/breakable_cuda_graph/breakable_cuda_graph.py (+123/-135); python/sglang/srt/model_executor/breakable_cuda_graph/context.py (+39/-0); python/sglang/srt/model_executor/breakable_cuda_graph_runner.py (+402/-0); python/sglang/srt/model_executor/model_runner.py (+8/-1); python/sglang/srt/models/nemotron_h.py (+16/-1); python/sglang/srt/server_args.py (+6/-0); test/registered/breakable_cuda_graph/test_breakable_cuda_graph.py (+53/-7)
LABELS: high priority, deepseek, run-ci, piecewise-cuda-graph
BODY: ## Motivation ⏎  ⏎  ⏎ Inspired by #19102 and credit to @cctry, we implemented breakable piecewise CUDA graph which does not rely on torch compile backend.  ⏎  ⏎ This is still an experimental feature for simpler support of piecewise CUDA graph. ⏎  ⏎ Usage: `--enable-breakable-cuda-graph` ⏎  ⏎ mGSM8K Benchmark (200 questions) ⏎ | Config              | PCG score | PCG tput | PCG cap_GB | BCG score | BCG tput | BCG cap_GB |                                      …[truncated]

### L2-59724e90a9  (L2, 2026-04-24, sha 59724e90a9b8, PR #23454)
TITLE: model: support Moss-VL (#23454)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/configs/model_config.py (+3/-0); python/sglang/srt/layers/attention/flashinfer_backend.py (+11/-1); python/sglang/srt/managers/schedule_batch.py (+70/-0); python/sglang/srt/managers/tokenizer_manager.py (+12/-2); python/sglang/srt/model_executor/forward_batch_info.py (+1/-0); python/sglang/srt/model_executor/model_runner.py (+8/-0); python/sglang/srt/models/moss_vl.py (+1643/-0); python/sglang/srt/multimodal/processors/moss_vl.py (+612/-0); python/sglang/srt/parser/conversation.py (+29/-2); python/sglang/srt/server_args.py (+12/-1)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ This PR adds Python-side runtime support for Moss-VL in SRT. ⏎  ⏎ The changes include: ⏎ - add `MossVLForConditionalGeneration` model support ⏎ - add a Moss-VL multimodal processor ⏎ - register Moss-VL conversation template and model-type matching ⏎ - pass Moss-VL-specific multimodal metadata through schedule/forward paths ⏎ - support cross-attention custom masks in the FlashInfer prefill path ⏎ - update tokenizer/model runner flow to initi …[truncated]

### L2-880599cd43  (L2, 2026-04-25, sha 880599cd430f, PR #23698)
TITLE: docs(DeepSeek-V4): bump GB300 Pro PD decode --mem-fraction-static 0.83 → 0.9 (#23698)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+5/-3)
LABELS: deepseek
BODY: 

### L2-9003f24e2b  (L2, 2026-04-25, sha 9003f24e2b88, PR #23733)
TITLE: chore: bump sglang-kernel version to 0.4.1.post1 (#23733)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); test/registered/4-gpu-models/test_qwen35_hicache.py (+0/-5); test/registered/hicache/test_hicache_storage.py (+0/-5); test/registered/hicache/test_hicache_storage_3fs_backend.py (+0/-2); test/registered/hicache/test_hicache_storage_file_backend.py (+0/-2); test/registered/hicache/test_hicache_storage_mooncake_backend.py (+3/-3); test/registered/hicache/test_hicache_storage_runtime_attach_detach.py (+0/-2); test/registered/hicache/test_hicache_variants.py (+0/-2)
LABELS: dependencies, hicache, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sglang-kernel` version to `0.4.1.post1` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.4.1.post1` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the `sglang-kernel` dependen …[truncated]

### L2-da175b964d  (L2, 2026-04-26, sha da175b964d15, PR #23785)
TITLE: chore: update CI test est_time values (#23785)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: test/registered/4-gpu-models/test_gpt_oss_4gpu.py (+2/-2); test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py (+1/-1); test/registered/4-gpu-models/test_qwen35_fp4_triton.py (+1/-1); test/registered/4-gpu-models/test_qwen3_30b.py (+1/-1); test/registered/4-gpu-models/test_qwen3_next_models.py (+1/-1); test/registered/8-gpu-models/test_deepseek_v32_indexcache.py (+1/-1); test/registered/8-gpu-models/test_deepseek_v3_basic.py (+1/-1); test/registered/8-gpu-models/test_deepseek_v3_mtp.py (+1/-1); test/registered/8-gpu-models/test_dsa_models_basic.py (+1/-1); test/registered/8-gpu-models/test_dsa_models_mtp.py (+1/-1); (+258 more)
LABELS: quant, lora, Multi-modal, deepseek, speculative-decoding, blackwell, npu
BODY: ## Summary ⏎  ⏎ Updates `est_time` values in CI test registration calls based on the 90th percentile of the last 15 successful executions from scheduled PR Test runs on main. ⏎  ⏎ This keeps the LPT load-balancing algorithm accurate for partitioning tests across parallel CI jobs. ⏎  ⏎ ### Significant est_time changes (26 of 269 updates) ⏎  ⏎ | File | Suite | Old (s) | New (s) | Δ | ⏎ | --- | --- | ---: | ---: | ---: | ⏎ | `test_awq.py` | `stage-b-test-1-gpu-large` | …[truncated]

### L2-32c3513816  (L2, 2026-04-27, sha 32c3513816b0, PR #20918)
TITLE: [NPU] Support MTP for Qwen3.5 (#20918)
SOURCES: path_core
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+17/-5); python/sglang/srt/environ.py (+2/-0); python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py (+425/-0); python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py (+280/-0); python/sglang/srt/hardware_backend/npu/memory_pool_npu.py (+24/-0); python/sglang/srt/layers/attention/mamba/mamba2_metadata.py (+1/-0); python/sglang/srt/layers/layernorm.py (+6/-3); python/sglang/srt/mem_cache/memory_pool.py (+9/-0); python/sglang/srt/models/qwen3_5_mtp.py (+23/-1); python/sglang/srt/models/qwen3_next_mtp.py (+22/-1)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ Adapt the MTP (Multi-Token Prediction) speculative decoding feature for the Qwen3.5 model on the Ascend NPU platform, fix inference errors, and ensure stable and efficient model operation. ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Add a dedicated GDN attention backend tailored for Ascend NPU, designed to address hardware-specific compatibility and performance needs; ⏎ 2. Complete end-to-end MTP speculative decoding adaptation for the Qwen3.5 mode …[truncated]

### L2-bd448e51bd  (L2, 2026-04-29, sha bd448e51bd8b, PR #23962)
TITLE: [Spec] Split `accept_length` into `num_accepted_drafts` and `num_accepted_tokens` (#23962)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.aiter_mla, L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+1/-1); python/sglang/srt/layers/attention/aiter_backend.py (+5/-5); python/sglang/srt/layers/attention/flashattention_backend.py (+4/-4); python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py (+3/-2); python/sglang/srt/layers/attention/nsa_backend.py (+2/-2); python/sglang/srt/layers/attention/triton_backend.py (+2/-2); python/sglang/srt/layers/attention/trtllm_mha_backend.py (+4/-4); python/sglang/srt/layers/attention/wave_backend.py (+2/-2); python/sglang/srt/layers/utils/logprob.py (+4/-4); python/sglang/srt/managers/scheduler_output_processor_mixin.py (+4/-4); (+20 more)
LABELS: high priority, blackwell, run-ci
BODY: Follows up #23530. ⏎  ⏎ ## Summary ⏎ - Split the ambiguous `accept_length` into two explicit fields on `EagleDraftInput` / `NgramVerifyInput`: `num_accepted_drafts` (strict drafts-only) and `num_accepted_tokens` (includes the bonus token; equals drafts + 1 per req) ⏎ - Decouple the `accept_length.add_(1)` in-place mutation that flipped the variable's semantics mid-function ⏎ - Match the `accept`/`draft` naming convention from #23530: name contains `draft`  …[truncated]

### L2-3fce8f2009  (L2, 2026-04-29, sha 3fce8f200992, PR #23947)
TITLE: [Docs] add cookbook for Ling-2.6 family (#23947)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/InclusionAI/Ling-2.6.mdx (+223/-0); docs_new/docs.json (+1/-0); docs_new/src/snippets/autoregressive/ling-26-1t-deployment.jsx (+178/-0); docs_new/src/snippets/autoregressive/ling-26-flash-deployment.jsx (+160/-0)
BODY: ## Summary ⏎  ⏎ Combined cookbook covering the **Ling-2.6** family from inclusionAI: ⏎  ⏎ - **Ling-2.6-flash** — 104B total / 7.4B active BF16 MoE ⏎ - **Ling-2.6-1T** — ~1T FP8 (E4M3) MoE ⏎  ⏎ Both share the `1:7 MLA + Lightning Linear` hybrid attention backbone introduced in Ling-2.5, refined for token efficiency and agentic workloads (BFCL-V4, TAU2-bench, SWE-bench Verified, Claw-Eval, PinchBench). ⏎  ⏎ ## What's included ⏎  ⏎ - `docs_new/cookbook/autoregressive/Inc …[truncated]

### L2-1376761841  (L2, 2026-04-29, sha 13767618412a, PR #24069)
TITLE: fix(moe): repair dead import in fused_moe_native after MoE refactor (#24069)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/fused_moe_native.py (+4/-2); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+2/-2)
LABELS: run-ci
BODY: ## Summary ⏎ - `fused_moe_native.py:10` still imported `swiglu_with_alpha_and_limit` from `sglang.srt.layers.moe.fused_moe_triton.fused_moe`, but that symbol was renamed to `_swiglu_gpt_oss_sigmoid_alpha` in #18084 and the whole module was deleted in #23019 — so any `torch.compile` + MoE path with `gemm1_alpha` set raised `ModuleNotFoundError` on server startup. ⏎ - Promote the helper to public (`swiglu_gpt_oss_sigmoid_alpha`) in `moe_runner/triton_u …[truncated]

### L2-4c1eefca4f  (L2, 2026-04-29, sha 4c1eefca4fa9, PR #21685)
TITLE: [NPU] ascend backend support qwen3 moe attention cp (#21685)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/platforms/ascend/ascend_npu_qwen3_examples.md (+80/-0); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+156/-8); test/registered/ascend/llm_models/test_npu_qwen3_30b_attn_cp.py (+95/-0)
LABELS: documentation, npu, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Qwen3 MoE models on Ascend NPU already support Prefill Context Parallel (PCP) for the MLA attention path, but the standard (non-MLA) attention path lacked CP support. This PR completes CP support for standard attention in the Ascend backend, enabling Qwen3-30B-A3B to run correctly under co-located deployment (TP=4 / MOE_DP=2 / ATTN_CP=2), reducing peak HBM usage during long-sequence prefill and improving TTFT. ⏎  ⏎ ## Modifications …[truncated]
