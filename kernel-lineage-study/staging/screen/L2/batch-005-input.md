### L2-1279ae0787  (L2, 2026-04-29, sha 1279ae0787ba, PR #24027)
TITLE: Bugfix (#24027)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/Mistral/Mistral-Medium-3.5.mdx (+398/-0); docs_new/docs.json (+1/-0); docs_new/src/snippets/autoregressive/mistral-medium-3-5-deployment.jsx (+340/-0); python/sglang/srt/configs/model_config.py (+4/-1); python/sglang/srt/models/mistral.py (+102/-0); python/sglang/srt/models/pixtral.py (+16/-4); python/sglang/srt/utils/hf_transformers/common.py (+8/-0); python/sglang/srt/utils/hf_transformers/mistral_utils.py (+16/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Pixtral / Mistral3 currently does not support text configs that use plain GQA attention — the language-model branch is hardcoded to the MLA path, and `Mistral3ForConditionalGeneration` cannot load checkpoints in the transformers v5 weight layout. ⏎  ⏎ ## Modifications ⏎  ⏎ - `models/pixtral.py`: pick the language-model class from `text_config.model_type` (`deepseek_v3` → MLA backbone, otherwise → standard Llama-style backbone). ⏎ - `models/m …[truncated]

### L2-71e89e9003  (L2, 2026-04-30, sha 71e89e9003f5, PR #23654)
TITLE: [MUSA][19/N] Support qwen series models (#23654)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: 3rdparty/amd/wheel/sglang/pyproject.toml (+1/-1); python/pyproject_other.toml (+1/-1); python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py (+30/-12); python/sglang/srt/hardware_backend/musa/kernels/topk.py (+300/-0); python/sglang/srt/layers/attention/vision.py (+12/-2); python/sglang/srt/layers/moe/topk.py (+25/-3); python/sglang/srt/layers/quantization/fp8_kernel.py (+0/-5); python/sglang/srt/layers/quantization/fp8_utils.py (+13/-7); python/sglang/srt/layers/rotary_embedding/base.py (+2/-1); python/sglang/srt/layers/utils/multi_platform.py (+1/-4); (+10 more)
LABELS: dependencies, Multi-modal, sgl-kernel, run-ci, mthreads
BODY: ## Motivation ⏎  ⏎ This PR is part of the MUSA (Moore Threads GPU) backend support series (19/N). The goal is to enable Qwen series models on the MUSA platform by: ⏎ 1. Adding MUSA-specific MoE fused gate and top-k kernels. ⏎ 2. Fixing compatibility issues in vision attention, FP8 quantization, and multi-platform dispatch for MUSA. ⏎  ⏎ ## Modifications ⏎  ⏎ ### 1. MUSA Top-K Kernels (New File) ⏎ - **`python/sglang/srt/hardware_backend/musa/kernels/topk.p …[truncated]

### L2-da7f890788  (L2, 2026-05-01, sha da7f89078807, PR #23557)
TITLE: [Intel GPU] Integrate flash_mla_decode in Intel XPU attention backend (#23557)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/xpu_backend.py (+31/-62); python/sglang/srt/model_executor/model_runner.py (+1/-0); python/sglang/srt/models/deepseek_common/attention_backend_handler.py (+5/-0); python/sglang/srt/models/deepseek_common/utils.py (+1/-0); python/sglang/srt/server_args.py (+16/-3); test/srt/xpu/test_intel_xpu_backend.py (+14/-1)
LABELS: run-ci
BODY: used with --decode-attention-backend intel_xpu ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglan …[truncated]

### L2-88bb5dffe4  (L2, 2026-05-02, sha 88bb5dffe496, PR #21247)
TITLE: [Dependency] Upgrade to Torch 2.11.0 (#21247)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+4/-21); .github/workflows/pr-test-amd-rocm720.yml (+20/-0); .github/workflows/pr-test-amd.yml (+20/-0); .github/workflows/pr-test-jit-kernel.yml (+63/-4); .github/workflows/pr-test-multimodal-gen.yml (+24/-4); .github/workflows/pr-test-npu.yml (+56/-26); .github/workflows/pr-test.yml (+13/-10); python/pyproject_other.toml (+2/-2); python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py (+22/-1); (+11 more)
LABELS: documentation, high priority, amd, dependencies, Multi-modal, deepseek, sgl-kernel, npu, run-ci, diffusion
BODY: ## Motivation ⏎  ⏎ github.com/pytorch/pytorch/releases/tag/v2.11.0 ⏎  ⏎ ## Modifications ⏎  ⏎ github.com/sgl-project/sglang/pull/18862 ⏎  ⏎ <!-- Detail the changes made in this pull request. --

### L2-00d620b77d  (L2, 2026-05-03, sha 00d620b77d1b, PR #24328)
TITLE: introduce arg_groups/ with nemotron_h hook (#24328)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/__init__.py (+0/-0); python/sglang/srt/arg_groups/nemotron_h_hook.py (+51/-0); python/sglang/srt/server_args.py (+4/-39)
BODY: Seed the per-arch hook framework called out in RFC #20481, with a minimal example: extract the \`NemotronHForCausalLM\` model-specific block out of \`server_args.py\` into \`arg_groups/nemotron_h_hook.py\`. ⏎  ⏎ This is a flat, simplified version of the RFC structure — \`arg_groups/<model>_hook.py\` instead of nested \`arg_groups/handlers/model_specific/<family>.py\`. The 3-level RFC layout is overkill before the broader mixin / handler split lands;  …[truncated]

### L2-52b4609789  (L2, 2026-05-04, sha 52b46097894c, PR #23593)
TITLE: [Docker] Prep for torch 2.11: cu129 fix, image validator, dep cleanup (#23593)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+21/-34); docker/diffusion.Dockerfile (+0/-104); .github/workflows/release-docker-dev.yml (+6/-4); .github/workflows/release-docker-runtime.yml (+3/-3); .github/workflows/release-docker.yml (+4/-4); .github/workflows/trivy-scan-dev.yml (+1/-1); scripts/ci/utils/docker_build_metadata_args.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ Torch 2.11 makes cu130 wheels the PyPI default, which surfaces two install-time bugs in the cu129 Docker image build path. This PR fixes those, adds a post-build validator to gate future CUDA-variant regressions, and cleans up NVIDIA package overrides that torch 2.11 now ships at equal or newer versions — mirroring the CI-script cleanup already done in #21247. ⏎  ⏎ Companion to **#21247** (torch 2.11 upgrade), which handles `python/pyp …[truncated]

### L2-62a4df0067  (L2, 2026-05-04, sha 62a4df006799, PR #24234)
TITLE: [docker] Fix silently-masked cubin download failure; skip prebuilt cubins on aarch64 (#24234)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+18/-9)
BODY: ## Motivation ⏎  ⏎ Fix two compounded issues in `framework_final` of `docker/Dockerfile` that have been silently breaking aarch64 nightly Docker builds since #22160 (2026-04-09): ⏎  ⏎ 1. **Silent-failure bug.** The cubin-download retry block ends with `... || true` (intended only to guard the trailing `find` cleanup), but bash's left-associative `&&`/`||` precedence makes that swallow the entire chain — including the `[ "$success" = "1" ]` fail-fast chec …[truncated]

### L2-c2db19ffa4  (L2, 2026-05-05, sha c2db19ffa40e, PR #23146)
TITLE: [AMD] Enable EAGLE speculative decoding for Qwen3.5 FP8 and MXFP4 models with aiter's unified attention (#23146)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+478/-148); python/sglang/srt/layers/attention/triton_ops/aiter_unified_attention.py (+97/-0); python/sglang/srt/models/qwen3_5_mtp.py (+12/-0); python/sglang/srt/server_args.py (+1/-0)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Co-author: @kkHuang-amd, @sogalin ⏎  ⏎ ## Motivation ⏎  ⏎ Enable EAGLE speculative decoding on top of the aiter unified-attention decode path for Qwen3.5-397B-A17B FP8 and MXFP4 on AMD (gfx950 / MI355X), and fix loading for the Quark-MXFP4 EAGLE/MTP variant. ⏎  ⏎ Before this change: ⏎ - `SGLANG_USE_AITER_UNIFIED_ATTN=1` plus `--speculative-algorithm EAGLE` does not have a unified-attention target-verify path for the non-MLA, `topk == 1` EAGLE case. ⏎ - T …[truncated]

### L2-47a416fc62  (L2, 2026-05-05, sha 47a416fc6272, PR #24392)
TITLE: add indexer-topk capture (V3.2 NSA + infra) (#24392)
SOURCES: path_core
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+11/-2); python/sglang/srt/configs/model_config.py (+13/-0); python/sglang/srt/disaggregation/prefill.py (+3/-0); python/sglang/srt/layers/attention/indexer_topk_capturer.py (+103/-0); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+27/-15); python/sglang/srt/managers/detokenizer_manager.py (+2/-0); python/sglang/srt/managers/io_struct.py (+8/-0); python/sglang/srt/managers/multi_tokenizer_mixin.py (+3/-0); python/sglang/srt/managers/schedule_batch.py (+10/-0); python/sglang/srt/managers/scheduler.py (+1/-0); (+9 more)
LABELS: run-ci
BODY: Stacked on #24403. Adds the `IndexerTopkCapturer` (built on `BaseTopkCapturer` from #24403) and wires V3.2 NSA models as the first producer. ⏎  ⏎ **API** ⏎ - Server flag: `--enable-return-indexer-topk` (default off) ⏎ - Per-request flag: `return_indexer_topk: bool` on `GenerateReqInput` ⏎ - Response: `meta_info["indexer_topk"]` is a base64-encoded int32 tensor of shape `(seqlen, num_indexer_layers, index_topk)` ⏎  ⏎ **Activation gating** — `model_config.get_nu …[truncated]

### L2-08d4c2072b  (L2, 2026-05-05, sha 08d4c2072b50, PR #24450)
TITLE: move topk capturers to srt/state_capturer/ (#24450)
SOURCES: path_core
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+3/-3); python/sglang/srt/hardware_backend/npu/moe/topk.py (+1/-1); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+3/-3); python/sglang/srt/layers/moe/topk.py (+1/-1); python/sglang/srt/managers/scheduler_output_processor_mixin.py (+4/-4); python/sglang/srt/managers/utils.py (+1/-1); python/sglang/srt/model_executor/model_runner.py (+11/-11); python/sglang/srt/state_capturer/__init__.py (+0/-0); python/sglang/srt/state_capturer/base.py (+0/-0); python/sglang/srt/state_capturer/indexer_topk.py (+1/-1); (+3 more)
LABELS: run-ci
BODY: Move the three capturer files out of `layers/` into a new top-level dir `srt/state_capturer/` — capture is observability infrastructure (side-effect observers tapping into producer outputs), not a layer (compute component). Aligns with sglang's existing observer modules (`eplb/`, `metrics/`, `debug_utils/`) which all sit at `srt/` root. ⏎  ⏎ ## Moves ⏎  ⏎ ``` ⏎ layers/topk_capturer_base.py            -> state_capturer/base.py ⏎ layers/moe/routed_experts_capt …[truncated]

### L2-6764155914  (L2, 2026-05-05, sha 6764155914ef, PR #24457)
TITLE: chore: bump sgl-kernel version to 0.4.2.post1 (#24457)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+0/-54); sgl-kernel/pyproject.toml (+1/-1); sgl-kernel/pyproject_cpu.toml (+1/-1); sgl-kernel/pyproject_musa.toml (+1/-1); sgl-kernel/pyproject_rocm.toml (+1/-1); sgl-kernel/python/sgl_kernel/version.py (+1/-1)
LABELS: amd, dependencies, sgl-kernel, mthreads
BODY: ## Summary ⏎  ⏎ This PR bumps the sgl-kernel version to `0.4.2.post1` across all relevant files. ⏎  ⏎ ## Files Updated ⏎ - sgl-kernel/pyproject.toml ⏎ - sgl-kernel/pyproject_cpu.toml ⏎ - sgl-kernel/pyproject_musa.toml ⏎ - sgl-kernel/pyproject_rocm.toml ⏎ - sgl-kernel/python/sgl_kernel/version.py ⏎  ⏎ 🤖 Generated with GitHub Actions

### L2-1e404afec2  (L2, 2026-05-05, sha 1e404afec210, PR #24439)
TITLE: fix(req_pool): bump pool.size to match actual tensor row count after #24243 (#24439)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/disaggregation/decode.py (+6/-5); python/sglang/srt/mem_cache/memory_pool.py (+6/-7); python/sglang/srt/model_executor/model_runner.py (+2/-1); python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py (+3/-2)
LABELS: high priority, run-ci
BODY: ## Bug ⏎  ⏎ #24243 reserved slot 0 of `ReqToTokenPool` / `DecodeReqToTokenPool` as a zero padding row by: ⏎  ⏎ 1. expanding the underlying tensor by one row (`(size + 1, max_context_len)`), ⏎ 2. shifting `free_slots` to start at 1 (`range(1, size + 1)`). ⏎  ⏎ But `self.size` was left at the original `size`. Many attention backends size their per-request metadata buffers as `[pool.size]` (`flashinfer`, `triton`, `aiter`, `flashmla`, `nsa`, `wave`, `gdn`, ...).  …[truncated]

### L2-3da87902d7  (L2, 2026-05-06, sha 3da87902d75c, PR #23013)
TITLE: [HiSparse] Support FP8 KV cache by routing to flashmla_kv backend (#23013)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+29/-15)
LABELS: run-ci
BODY: flashmla_sparse does not accept FP8 input, so HiSparse was previously pinned to BF16 KV. The flashmla_kv kernel already supports native FP8 + sparse attention (is_fp8_kvcache=True + indices=...), and HiSparse's hot-buffer indices are drop-in compatible with its indices contract. ⏎  ⏎ Related resource: ⏎ https://github.com/sgl-project/sglang/pull/13841 ⏎ https://github.com/sgl-project/sglang/issues/13832 ⏎ https://github.com/sgl-project/FlashMLA/pull/1 …[truncated]

### L2-3fe8bc987e  (L2, 2026-05-06, sha 3fe8bc987e0d, PR #20479)
TITLE: Support Triton MLA FP8 KV cache (#20479)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/srt/layers/attention/triton_backend.py (+40/-4); python/sglang/srt/layers/attention/triton_ops/decode_attention.py (+47/-24); python/sglang/srt/layers/attention/triton_ops/extend_attention.py (+0/-2)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: Inspire by https://github.com/sgl-project/sglang/pull/18882 ⏎  ⏎ Recently some users of SM120 need this feature for FP8 KV cache (currently, Flashinfer will be slower and doesn't support FP8 Attention), XQA only support limited heaad dim (DP attention only and R1 head dim, not Kimi/etc.). Before, Triton backend will be really slow under MLA and long seq len (found by Claude). And add PDL to the stage 2 kernel. ⏎  ⏎ Credit to @koush from vLLM for this …[truncated]

### L2-ecb786c8d7  (L2, 2026-05-06, sha ecb786c8d719, PR #24268)
TITLE: [Kernel] Deprecate DeepGemm in sgl kernel and apply custom wheel sgl-deep-gemm (#24268)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/nsa/nsa_indexer.py (+10/-2); python/sglang/srt/layers/attention/nsa_backend.py (+19/-3); python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py (+10/-4); python/sglang/srt/layers/deep_gemm_wrapper/configurer.py (+2/-0); python/sglang/srt/layers/quantization/fp8_utils.py (+13/-0); scripts/ci/cuda/ci_install_dependency.sh (+6/-0); scripts/ci/cuda/warmup_deep_gemm.py (+8/-4); (+4 more)
LABELS: high priority, dependencies, deepseek, sgl-kernel, run-ci
BODY: ## Motivation ⏎ Ref:  ⏎ https://github.com/sgl-project/DeepGEMM/pull/26 ⏎ https://pypi.org/project/sgl-deep-gemm/ ⏎ #20745  ⏎  ⏎ Do the following one by one: ⏎  ⏎  ⏎ We will build a single wheel for deepgemm in sglang, rather than compiling it with sglang-kernel ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Mer …[truncated]

### L2-65ce9965ce  (L2, 2026-05-06, sha 65ce9965ce82, PR #22715)
TITLE: [Bug Fix] Fix RunAI streamer: corrupted weights, missing quant init, and broken URIs for multimodal models (#22715)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/utils/hf_transformers/common.py (+7/-0); python/sglang/srt/utils/hf_transformers/config.py (+2/-3); python/sglang/srt/utils/hf_transformers/processor.py (+3/-0); python/sglang/srt/utils/hf_transformers/tokenizer.py (+2/-3)
LABELS: deepseek, run-ci
ISSUES: #22701 [Bug] RunAI streamer (#17948): corrupted weights, missing quant init, and broken object-storage URIs for multimodal models
BODY: ## Motivation ⏎  ⏎ Fixes #22701. ⏎  ⏎ Found while testing #17948 (*Direct model loading from object storage with Runai Model Streamer*) on an 8×H200 cluster with Kimi-K2.5 (WNA16 Marlin MoE) and DeepSeek-V3-0324 (BF16 MoE) via `az://` paths with TP=8. ⏎  ⏎ Three independent bugs prevent RunAI streamer from working correctly with multi-GPU quantized / multimodal models: ⏎  ⏎ 1. **`RunaiModelStreamerLoader` omits `quant_config`** — quantized models initial …[truncated]

### L2-9dfb1d2ebe  (L2, 2026-05-07, sha 9dfb1d2ebece, PR #24372)
TITLE: [Intel GPU] Fix flash_mla_get_workspace_size call in intel_xpu (#24372)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/xpu_backend.py (+10/-4)
LABELS: intel, xpu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-35870d55ac  (L2, 2026-05-07, sha 35870d55aca7, PR #23882)
TITLE: Deepseek V4 (#23882)
SOURCES: path_core, symbol_pickaxe, dependency_pin, corpus:production-kernel-provenance, body_keyword
ARTIFACT_HINTS: L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: docker/Dockerfile (+10/-0); python/pyproject.toml (+2/-1); .github/workflows/pr-test.yml (+106/-0); docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+0/-4); python/sglang/jit_kernel/csrc/deepseek_v4/c128.cuh (+522/-0); python/sglang/jit_kernel/csrc/deepseek_v4/c128_online.cuh (+726/-0); python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh (+543/-0); python/sglang/jit_kernel/csrc/deepseek_v4/c4.cuh (+549/-0); python/sglang/jit_kernel/csrc/deepseek_v4/common.cuh (+208/-0); python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope.cuh (+254/-0); (+144 more)
LABELS: documentation, high priority, quant, dependencies, deepseek, speculative-decoding, sgl-kernel, blackwell, npu, benchmark
BODY: ## Rebase progress ⏎  ⏎ merge-base 0519b09 → target main ea794de (latest), 2880 commits across **29 batches** (~100/batch). **Done: 29 / 29. ✅** ⏎  ⏎ [details omitted] ⏎ [details omitted] ⏎ [details omitted] ⏎ [details omitted] ⏎ [details omitted] ⏎ [details omitted] ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details omitted] ⏎  ⏎ [details om …[truncated]

### L2-d9dddd4d7d  (L2, 2026-05-07, sha d9dddd4d7d52, PR #23336)
TITLE: [SPEC V2][2/N] feat: adaptive spec support spec v2 (#23336)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/scheduler_output_processor_mixin.py (+9/-1); python/sglang/srt/managers/utils.py (+1/-0); python/sglang/srt/speculative/adaptive_spec_params.py (+25/-14); python/sglang/srt/speculative/base_spec_worker.py (+8/-0); python/sglang/srt/speculative/eagle_info_v2.py (+6/-1); python/sglang/srt/speculative/eagle_worker.py (+24/-18); python/sglang/srt/speculative/eagle_worker_v2.py (+187/-8); python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py (+1/-0); python/sglang/srt/speculative/standalone_worker.py (+6/-0); python/sglang/srt/speculative/standalone_worker_v2.py (+6/-0); (+2 more)
LABELS: high priority, speculative-decoding, run-ci
BODY: ## Motivation ⏎ adaptive spec support spec v2: ⏎  ⏎ 1. launch sgl+adaptive spec+spec v2 ⏎ ``` ⏎ SGLANG_ENABLE_SPEC_V2=1 \ ⏎ python3 -m sglang.launch_server \ ⏎   --model-path /root/models/shakechen/Llama-2-7b-chat-hf \ ⏎   --trust-remote-code \ ⏎   --attention-backend triton \ ⏎   --speculative-algorithm EAGLE \ ⏎   --speculative-draft-model-path /root/models/lmsys/sglang-EAGLE-llama2-chat-7B \ ⏎   --speculative-num-steps 1 \ ⏎   --speculative-eagle-topk 1 \ …[truncated]

### L2-15e6572f21  (L2, 2026-05-07, sha 15e6572f2198, PR #23255)
TITLE: [MUSA][18/N] Add MUSA-optimized kernel implementations for hot ops (#23255)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/common_extension_musa.cc (+42/-0); sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.mu (+17/-8); sgl-kernel/csrc/musa/common.muh (+42/-0); sgl-kernel/csrc/musa/dtype.muh (+446/-0); sgl-kernel/csrc/musa/moe_gemv_swiglu.mu (+846/-0); sgl-kernel/csrc/musa/pos_encoding_contiguous.mu (+264/-0); sgl-kernel/csrc/musa/ternary.mu (+123/-0); sgl-kernel/csrc/musa/top_k_top_p_sampling.mu (+382/-0); sgl-kernel/include/musa/dispatch_utils.h (+34/-0); sgl-kernel/include/musa/integer_subbyte.h (+49/-0); (+5 more)
LABELS: quant, sgl-kernel, run-ci, mthreads
BODY: ## Motivation ⏎  ⏎ This PR continues the ongoing effort (tracked in https://github.com/sgl-project/sglang/issues/16565) to add full support for Moore Threads GPUs in SGLang by leveraging MUSA (Meta-computing Unified System Architecture) for LLM inference. ⏎  ⏎ The primary goal of this pr is to add MUSA-specific kernel support to sgl-kernel. ⏎  ⏎ It focuses only on the MUSA kernel path so that sgl-kernel can build successfully with setup_musa.py and be  …[truncated]

### L2-b22d3cd606  (L2, 2026-05-08, sha b22d3cd60640, PR #20319)
TITLE: [AMD] Support fp8 MLA for diffusion model (#20319)
SOURCES: subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/multimodal_gen/runtime/layers/attention/backends/aiter.py (+302/-1)
LABELS: amd, run-ci, diffusion
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ Replace the FP8 per-tensor flash attention path (`flash_attn_fp8_pertensor_func`) in the aiter DiT backend with the FP8 MLA prefill ASM kernel (`mla_prefill_ps_asm_fwd` + `mla_reduce_v1`), which delivers significantly better performance on MI355X.  ⏎  ⏎ When shapes are unsupported by the MLA kernel (v_head_dim != 128 or num_heads not divisible by 8), the code falls back gracefully to the BF16 flash attention path. ⏎  ⏎ Required: http …[truncated]

### L2-76a1f169b3  (L2, 2026-05-08, sha 76a1f169b3c6, PR #23955)
TITLE: [AMD] Add AMD FP8 MLA attention test for Wan2.2-T2V-A14B (#23955)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/amd/test_wan22_fp8_mla.py (+136/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Add registered AMD CI test covering the FP8 MLA attention path (`SGLANG_AITER_FP8_ATTN=1`) introduced in PR #20319. This ensures the FP8 MLA code path is exercised in nightly CI and catches regressions. ⏎  ⏎ ## Modifications ⏎  ⏎ Added `test/registered/amd/test_wan22_fp8_mla.py` with 4 test cases using Wan2.2-T2V-A14B: ⏎  ⏎ | Case ID | GPUs | torch.compile | ⏎ |---------|------|---------------| ⏎ | `wan2_2_t2v_a14b_fp8_mla_1gpu` | 1 | of …[truncated]

### L2-55224fff08  (L2, 2026-05-08, sha 55224fff0851, PR #22123)
TITLE: Add Arm64 CPU Phase 1A CI bootstrap (#22123)
SOURCES: dependency_pin
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docker/arm64.Dockerfile (+52/-0); .github/workflows/pr-test-arm64.yml (+118/-0); python/sglang/srt/server_args.py (+4/-1); sgl-kernel/csrc/cpu/CMakeLists.txt (+17/-0); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+29/-21); test/srt/cpu/test_server_args_backend.py (+35/-0); test/srt/run_suite.py (+17/-0)
LABELS: sgl-kernel, run-ci
BODY: ### Summary ⏎  ⏎ This PR adds **Arm64 CPU Phase 1A CI bootstrap** support. ⏎  ⏎ The goal is to give Arm64 a real PR-time build and functional test lane without disturbing the existing Xeon/x86 flow. This is an additive bootstrap step, not a full parity claim. ⏎  ⏎ ### What changed ⏎  ⏎ 1. Added a dedicated Arm64 PR workflow in `.github/workflows/pr-test-arm64.yml` ⏎ 2. Added a native Arm64 build image in `docker/arm64.Dockerfile` ⏎ 3. Defaulted Arm64 CPU b …[truncated]

### L2-5fbec0e445  (L2, 2026-05-08, sha 5fbec0e4455c, PR #24721)
TITLE: ci: prune per-commit CUDA tests — move 25 files + 13 testcases to test/manual/ (#24721)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: test/manual/4-gpu-models/test_qwen35_fp4_triton.py (+0/-0); test/manual/4-gpu-models/test_qwen35_models_archived.py (+168/-0); test/manual/4-gpu-models/test_qwen3_next_models.py (+0/-0); test/manual/4-gpu-models/test_qwen3_next_models_mtp_archived.py (+44/-0); test/manual/8-gpu-models/test_deepseek_v3_basic.py (+0/-0); test/manual/8-gpu-models/test_dsa_models_basic.py (+0/-0); test/manual/attention/test_fa3.py (+0/-0); test/manual/attention/test_local_attn.py (+0/-0); test/manual/core/test_gpt_oss_1gpu.py (+0/-0); test/manual/distributed/test_dp_attention_archived.py (+101/-0); (+35 more)
LABELS: high priority, quant, lora, deepseek, blackwell, run-ci
BODY: ## Summary ⏎  ⏎ Move tests marked **remove** in the 2026-05-07 working-session pruning catalog out of per-commit CUDA CI and into \`test/manual/\`, where they remain runnable as manual tests but are no longer auto-discovered by the registry parser. ⏎  ⏎ **Per-commit CUDA pipeline cut by ~174 minutes** of registered est_time: ⏎ - 8,622s from whole-file moves (143.7 min) ⏎ - 1,831s from subset removals (30.5 min) ⏎ - **10,453s total** ⏎  ⏎ 39 tests moved (25 whole f …[truncated]

### L2-62c2e091f6  (L2, 2026-05-08, sha 62c2e091f6ba, PR #22665)
TITLE: [PD] MORI-IO: Add state transfer, inline transfer model, and high-concurrency fixes (#22665)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/mori/conn.py (+590/-163); test/registered/amd/disaggregation/test_mori_transfer_engine_e2e.py (+179/-0)
LABELS: lora, deepseek, blackwell, run-ci, diffusion
BODY: ## Motivation ⏎  ⏎ Follow-up to #14626 which introduced MORI-IO as the RDMA-based KV transfer backend for PD disaggregation on AMD hardware. This PR addresses the known limitation (no state transfer) and resolves several performance bottlenecks and correctness issues discovered under high-concurrency workloads: ⏎  ⏎ 1. State data transfer was not implemented for hybrid models (Mamba, SWA, NSA). ⏎ 2. TP slice head mapping was incorrect for `prefill_tp_ …[truncated]

### L2-ef5e9f8aba  (L2, 2026-05-09, sha ef5e9f8abab1, PR #24793)
TITLE: [DSV4] Cherry pick missing commits from deepseek_v4 branch and enhance tests (#24793)
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: scripts/ci/cuda/ci_install_flash_mla.sh (+0/-35); .github/workflows/pr-test.yml (+2/-2); docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+0/-3); python/sglang/srt/entrypoints/openai/protocol.py (+5/-2); python/sglang/srt/function_call/deepseekv32_detector.py (+26/-10); python/sglang/srt/model_loader/loader.py (+5/-0); python/sglang/srt/model_loader/weight_utils.py (+33/-3); python/sglang/srt/server_args.py (+6/-0); scripts/ci/cuda/ci_install_dsv4_dep.sh (+161/-0); scripts/ci/utils/slash_command_handler.py (+6/-0); (+5 more)
LABELS: documentation, deepseek
BODY: ## Motivation ⏎ - Cherry-pick https://github.com/sgl-project/sglang/commit/dbdd494429678cdfa9d792d2c69b8a5f9f7961d7 ⏎ - Cherry-pick https://github.com/sgl-project/sglang/commit/c13abdc9fc7518e2155feaa45b2c64bc4eadbf33 ⏎ - Enhance per-commit v4 test with: b200 fp4 depep test, b200 fp4 megamoe test, h200 fp8 deepep test ⏎ - Add some tools for v4 tests, such as slash commands ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profi …[truncated]

### L2-fd636410a2  (L2, 2026-05-09, sha fd636410a209, PR #24097)
TITLE: Restrict fa_skip_kv_cache to non-MLA backends (#24097)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+6/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ PR #21971 added a `fa_skip_kv_cache` fast path for embedding models that bypasses the paged KV cache by passing raw K/V tensors to `flash_attn_varlen_func`. The flag is gated on `is_embedding`, `chunked_prefill_size == -1`, `disable_radix_cache`, and `kv_cache_dtype == "auto"`, but it is **not** gated on the attention architecture. ⏎  ⏎ This creates an asymmetry that silently corrupts MLA models if the same flag combination is used: ⏎  ⏎ - …[truncated]

### L2-12f42f2e7e  (L2, 2026-05-09, sha 12f42f2e7e75, PR #23976)
TITLE: Support Gemma3/4 + Eagle3 (#23976)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/gemma3_causal.py (+68/-6); python/sglang/srt/models/gemma3_mm.py (+16/-0); python/sglang/srt/models/gemma4_causal.py (+60/-3); python/sglang/srt/models/gemma4_mm.py (+34/-1); python/sglang/srt/models/llama_eagle3.py (+16/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ This PR supports Gemma3/4 model with Eagel3 and fixes multiple bugs in current eagle3 implementation. ⏎  ⏎ - Support aux layers embedding captures for both models. Fixes an issue when trying to capture the last layer ⏎ - Support an additional norm layer for each aux embedding. In practice, this could help stabilize the training and improve accept rate ⏎ - Gemma3/4 use nn.Embedding (Gemma3TextScaledWordEmbedding) which is not TP aware …[truncated]

### L2-cfd3fd00d0  (L2, 2026-05-09, sha cfd3fd00d048, PR #24854)
TITLE: [RL] Call torch.cuda.empty_cache() for `in-place` pause mode to avoid OOM (#24854)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/io_struct.py (+6/-1); python/sglang/srt/managers/scheduler.py (+9/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Post-weight-update processing (e.g. DeepSeek MLA w_kc/w_vc derivation, FP8 scale rebuild) creates transient CUDA allocations that fragment PyTorch's block cache. Without `empty_cache()`, reserved memory grows each iteration and eventually OOMs. ⏎  ⏎ | Pause mode | `flush_cache` called? | `empty_cache` before this PR | `empty_cache` after this PR | ⏎ |---|---|---|---| ⏎ | `abort` | Yes | Yes (via `flush_cache`) | Yes (via `flush_cache` + re …[truncated]

### L2-a87fb399de  (L2, 2026-05-09, sha a87fb399deaa, PR #24826)
TITLE: [spec decoding] support kimi-k2.5-eagle3-mla (#24826)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+1/-0); python/sglang/srt/models/kimi_k25_eagle3.py (+458/-0); python/sglang/srt/utils/hf_transformers/common.py (+6/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ tested with gpqa, accuracy 0.87, acc len 2.7 ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/mai …[truncated]

### L2-d08744238a  (L2, 2026-05-10, sha d08744238a64, PR #24859)
TITLE: [Spec V1] Split draft-extend phase from `EagleDraftInput` into new `EagleDraftExtendInput` (#24859)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/forward_batch_info.py (+5/-4); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+2/-2); python/sglang/srt/speculative/eagle_info.py (+161/-145); python/sglang/srt/speculative/eagle_worker.py (+56/-34); python/sglang/srt/speculative/frozen_kv_mtp_info.py (+20/-9); python/sglang/srt/speculative/frozen_kv_mtp_utils.py (+2/-4); python/sglang/srt/speculative/frozen_kv_mtp_worker.py (+30/-22); python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py (+2/-2); python/sglang/srt/speculative/multi_layer_eagle_worker.py (+45/-29); python/sglang/srt/speculative/spec_info.py (+4/-0)
LABELS: high priority, run-ci, bypass-fastfail
BODY: ## Summary ⏎ - Split the draft-extend phase out of `EagleDraftInput` into a new `EagleDraftExtendInput` dataclass, eliminating the phase-shifting overload where one instance was mutated across draft / draft-extend phases. ⏎ - V1 path only (`eagle_worker.py`, `multi_layer_eagle_worker.py`, `frozen_kv_mtp_worker.py`). V2 overlap worker still reuses one instance across phases — alignment is a follow-up. ⏎  ⏎ ## Background ⏎ - Pre-PR `EagleDraftInput.hidden_st …[truncated]

### L2-9150e77399  (L2, 2026-05-10, sha 9150e7739995, PR #24855)
TITLE: [Model] Add MiniCPM-V 4.6 support (#24855)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/benchmark/datasets/image.py (+1/-1); python/sglang/srt/configs/__init__.py (+3/-0); python/sglang/srt/configs/minicpmv4_6.py (+159/-0); python/sglang/srt/models/minicpmv.py (+301/-3); python/sglang/srt/models/minicpmv_vit.py (+526/-0); python/sglang/srt/multimodal/processors/minicpmv4_6.py (+548/-0); python/sglang/srt/server_args.py (+10/-0); python/sglang/srt/utils/hf_transformers/common.py (+4/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Add support for [MiniCPM-V 4.6](https://huggingface.co/openbmb/MiniCPM-V-4_6), the next iteration of the MiniCPM-V series. Compared to 4.5 (Qwen3 + SigLip + Perceiver-style resampler) the architecture changes are: ⏎  ⏎ - **Mid-ViT compression**: a 2x2 window-attention + 2x2 spatial fold inserted inside the SigLip encoder at `config.insert_layer_id`. ⏎ - **Post-encoder MLP merger** (`MiniCPMV_Merger`) replacing the legacy Perceiver r …[truncated]

### L2-d5e707f132  (L2, 2026-05-10, sha d5e707f1327e, PR #24914)
TITLE: Fix sgl-kernel-mla-test path after test was moved to test/manual (#24914)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-sgl-kernel.yml (+0/-35); .github/workflows/pr-test.yml (+11/-2)
LABELS: run-ci
BODY: Human remark: or maybe we should directly rm that section if it is confirmed no longer needed ⏎  ⏎ ## Summary ⏎  ⏎ #24721 (5fbec0e445 "ci: prune per-commit CUDA tests — move 25 files + 13 testcases to test/manual/") moved `test_mla_deepseek_v3.py` from `test/registered/mla/` to `test/manual/mla/`, but `.github/workflows/pr-test-sgl-kernel.yml:101` still hardcodes the old path. ⏎  ⏎ Result: every PR's `call-sgl-kernel-tests / sgl-kernel-mla-test` job fa …[truncated]

### L2-3ffb37789a  (L2, 2026-05-10, sha 3ffb37789a6a, PR #24799)
TITLE: [AMD] Fix DeepSeek import cascade by supporting both pre- and post-#2958 aiter `fused_qk_rmsnorm` APIs (#24799)
SOURCES: path_core
ARTIFACT_HINTS: L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+33/-3)
BODY: ## Motivation ⏎  ⏎ [ROCm/aiter#2958](https://github.com/ROCm/aiter/pull/2958) renamed the public `fused_qk_rmsnorm` in `aiter.ops.fused_qk_norm_rope_cache_quant` to a private `_fused_qk_rmsnorm` and introduced a new unified entry point in `aiter.ops.fused_qk_rmsnorm_group_quant` with a different (in-place, kwarg-only, no-return) signature. ⏎  ⏎ Because `forward_mla.py` imports this symbol at **module load time**, the rename causes an `ImportError` that c …[truncated]

### L2-da0eeb82f2  (L2, 2026-05-11, sha da0eeb82f232, PR #23675)
TITLE: perf: add --prefill-only-disable-kv-cache to skip KV pool allocation (#23675)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/mem_cache/memory_pool.py (+110/-0); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+67/-1); python/sglang/srt/server_args.py (+112/-0); test/registered/unit/server_args/test_server_args.py (+58/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ Adds an opt-in flag `--prefill-only-disable-kv-cache` that skips the physical KV cache pool allocation for prefill-only workloads. In those modes no layer ever reads or writes the KV cache — attention uses raw K/V via `flash_attn_varlen_func` — so the tens of GB of auto-sized KV buffers are pure waste. ⏎  ⏎ **What "prefill-only" covers today:** ⏎ - `--is-embedding` (CausalLM-as-embedding via FA3 `fa_skip_kv_cache`) ⏎ - SequenceClassification  …[truncated]

### L2-ce1736fcc6  (L2, 2026-05-11, sha ce1736fcc6cc, PR #25010)
TITLE: [Spec] Remove dead kernel params; fix stale comment in `trtllm_mla` (#25010)
SOURCES: path_core, path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/fla/kda.py (+0/-3); python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py (+0/-3); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+2/-2); test/registered/unit/layers/test_mamba_state_scatter_triton.py (+17/-17)
LABELS: blackwell, run-ci
BODY: Three independent cleanups in spec / attention scope: ⏎  ⏎ - `fla/kda.py`: drop `num_accepted_tokens` kwarg from `fused_recurrent_kda_fwd`. The param is never read inside the function (the only inner reference is commented out), and the lone caller hardcodes `num_accepted_tokens=None`. ⏎ - `mamba_state_scatter_triton._fused_mamba_state_scatter_with_mask_kernel`: drop the unused `total_requests` kernel param. `pid_req` comes from `tl.program_id`, bounds …[truncated]

### L2-74d70af09a  (L2, 2026-05-11, sha 74d70af09a19, PR #23449)
TITLE: [Apple Silicon] Add Metal kernel support in sgl-kernel (#23449)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs_new/docs/hardware-platforms/apple_metal.mdx (+22/-2); docs_new/docs/hardware-platforms/mthreads_gpu.mdx (+1/-1); docs_new/docs/sglang-diffusion/installation.mdx (+1/-1); python/sglang/__init__.py (+3/-1); sgl-kernel/csrc/metal/README.md (+21/-0); sgl-kernel/csrc/metal/placeholder.cpp (+0/-0); sgl-kernel/csrc/metal/placeholder.metal (+0/-0); sgl-kernel/python/sgl_kernel/__init__.py (+218/-206); sgl-kernel/python/sgl_kernel/metal.py (+30/-0); sgl-kernel/setup_metal.py (+298/-0)
LABELS: documentation, dependencies, sgl-kernel, run-ci, mthreads, apple-silicon
BODY: ## Motivation ⏎  ⏎  ⏎ [#22868](https://github.com/sgl-project/sglang/pull/22868) introduces custom Metal kernels to SGLang. After some discussion, we'd like to stay aligned with SGLang's existing convention of keeping custom kernels separate from SRT wiring. This PR therefore moves the Metal kernel build into `sgl-kernel` so it can be installed independently, mirroring how the CUDA / ROCm / MUSA backends are organized. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ 1. Ad …[truncated]

### L2-f3a8189e20  (L2, 2026-05-11, sha f3a8189e2090, PR #25014)
TITLE: [Spec] Internal rename per N2 v2 naming rule (#25014)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.aiter_mla, L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+4/-4); python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py (+6/-6); python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py (+3/-3); python/sglang/srt/layers/attention/aiter_backend.py (+4/-4); python/sglang/srt/layers/attention/flashattention_backend.py (+3/-3); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+4/-4); python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py (+6/-6); python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py (+1/-1); python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py (+2/-2); python/sglang/srt/layers/attention/nsa_backend.py (+2/-2); (+35 more)
LABELS: high priority, blackwell, npu, run-ci, bypass-fastfail
BODY: Pure internal identifier rename. **No external API change in this PR** — `meta_info` JSON keys, Prometheus names, trace_slice keys, and paper-aligned `spec_accept_rate` / `spec_accept_length` are all preserved. External-facing renames + backward-compat aliases follow in a separate PR. ⏎  ⏎ ## Scope ⏎  ⏎ **Rule 1 (drop `-ed`)**: ⏎ - `num_accepted_tokens*` → `num_accept_tokens*` ⏎ - `accepted_length`, `accepted_length_with_bonus` → `num_accept_tokens` ⏎ - `accep …[truncated]

### L2-a75b79e03b  (L2, 2026-05-11, sha a75b79e03bf8, PR #24663)
TITLE: Feat: Support newer EAGLE-3 drafters (#24663)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/llama_eagle3.py (+71/-34); python/sglang/srt/speculative/eagle_info.py (+16/-7)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ We will have a paper-release soon and we would like our models to have day-0 support with the paper/model release.  ⏎  ⏎ We will release checkpoints for gpt-oss-20b and gpt-oss-120b. ⏎  ⏎ ## Modifications ⏎  ⏎ 1) Multi-layer EAGLE-3: backwards compatible as we rename midlayer -> layers.0, this gives flexibility for users to run more models including vLLM supported ones. ⏎  ⏎ 2) Alternate norm positioning: this is about our upcoming paper …[truncated]

### L2-186eb42459  (L2, 2026-05-11, sha 186eb42459d7, PR #24664)
TITLE: Feat: Support SWA (Sliding Window Attention) for EAGLE-3 drafter (#24664)
SOURCES: release_notes
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/llama_eagle3.py (+20/-2); python/sglang/srt/server_args.py (+19/-16); python/sglang/srt/speculative/dflash_worker.py (+2/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Add sliding-window attention support to EAGLE series models. Related to #24663, in our upcoming paper we showed how SWA can help increase acceptance lengths if model is not trained to handle long context lengths. Some models are not usable without SWA as their training length is usually short (2-4K). ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Add SWA support to EAGLE via CLI flag `--speculative-draft-window-size`, which defaults to None to disabl …[truncated]

### L2-cfc41d5b15  (L2, 2026-05-11, sha cfc41d5b15fe, PR #25033)
TITLE: Fix kimi k2.5 mla eagle + dp attention (#25033)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/kimi_k25_eagle3.py (+15/-1)
BODY: ## Motivation ⏎ Similar as #21391 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and o …[truncated]

### L2-ecf5d844f5  (L2, 2026-05-12, sha ecf5d844f5bf, PR #22670)
TITLE: Migrate Intel CPU cases to the test/registered. (#22670)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-xeon.yml (+1/-1); .pre-commit-config.yaml (+1/-0); docs_new/index.mdx (+30/-30); scripts/ci/check_registered_tests.py (+2/-2); test/registered/cpu/test_activation.py (+59/-0); test/registered/cpu/test_binding.py (+30/-0); test/registered/cpu/test_bmm.py (+98/-0); test/registered/cpu/test_causal_conv1d.py (+330/-0); test/registered/cpu/test_cpu_graph.py (+91/-0); test/registered/cpu/test_decode.py (+172/-0); (+18 more)
LABELS: documentation, intel, cpu, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 25044 (confirmed_revert, reason=other)
BODY: # Summary ⏎ PR to improve CPU CI coverage and refine Xeon CI triggering. ⏎  ⏎ # Changes ⏎  ⏎ #### 1. Run `stage-b-test-cpu` through unified test runner. ⏎ #### 2. Update CPU per-commit suite to include `stage-b-test-cpu`. ⏎ #### 3. Register CPU tests ⏎   -   22 new CPU test files were added under `test/registered/cpu/`. ⏎   -   Coverage now includes key CPU kernel/operator paths: activation, binding, bmm, causal_conv1d, cpu_graph, decode/extend, flash_att …[truncated]

### L2-d5f3254ed1  (L2, 2026-05-12, sha d5f3254ed1f1, PR #24452)
TITLE: [Dependency] Flashinfer 0.6.8post1 -> 0.6.11 (#24452)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+4/-4); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/flashinfer_comm_fusion.py (+11/-13); python/sglang/srt/layers/quantization/fp4_utils.py (+7/-7); python/sglang/srt/utils/common.py (+1/-1); test/registered/moe/test_cutedsl_moe.py (+9/-6)
LABELS: high priority, dependencies, run-ci, bypass-fastfail
DEEP_STUDY: deep-study: this PR was reverted by PR 25310 (explicit_rollback, reason=crash_or_hang)
BODY: Commits of interest: ⏎  ⏎ https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.10 ⏎ https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.9 ⏎  ⏎ Note: if this is not cherry-picked, maybe just use 0.6.11, because it will break main and bad performance of this features ⏎  ⏎ Confirming not falling back from TRTLLM allreduce fusio  ⏎ <img width="1452" height="738" alt="Screenshot 2026-05-06 at 3 48 23 PM" src="https://github.com/user-attachm …[truncated]

### L2-4fb40bffac  (L2, 2026-05-12, sha 4fb40bffacfe, PR #25107)
TITLE: perf(nvfp4): free unused source scales after weight processing (#25107)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/modelopt_quant.py (+11/-1)
LABELS: high priority, quant, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ After NVFP4 `process_weights_after_loading`, several source-side scale tensors are bound to the layer but never read by `apply()`: ⏎  ⏎ **Linear (`ModelOptNvFp4LinearMethod`):** ⏎ - `layer.input_scale` — consumed into `alpha` and `input_scale_inv` ⏎ - `layer.weight_scale_2` — consumed into `alpha` ⏎ - `layer.weight_scale` — consumed into `weight_scale_interleaved` (both the FlashInfer-TRTLLM and CUTLASS branches converge on this) ⏎  ⏎ **MoE (`Mo …[truncated]

### L2-51a9403104  (L2, 2026-05-13, sha 51a94031042a, PR #25129)
TITLE: Update flashinfer to 0.6.11.post1 (#25129)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+2/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1)
LABELS: dependencies, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 25310 (explicit_rollback, reason=crash_or_hang)
BODY: ## Motivation ⏎ https://github.com/flashinfer-ai/flashinfer/compare/v0.6.11...v0.6.11.post1 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github. …[truncated]

### L2-839f7f2696  (L2, 2026-05-13, sha 839f7f2696f9, PR #24148)
TITLE: [AMD] Add _skip_rope_for_aiter_fused_mla method and check to avoid double rotating with gfx950 and Aiter backend (#24148)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+14/-0)
LABELS: run-ci
BODY: Aiter backend can be used for DeepSeek 3.2 and models with similar model architecture like the GLM5 to run the models on AMD Instinct. There is some differences in the optimizations and code path used for the MI30X and MI35X GPUs. The accuracy is fine when running on MI300X GPUs. However, there is currently an issue with the model output and accuracy when running on MI35X series GPUs. ⏎  ⏎ **Steps to reproduce the issue** ⏎  ⏎ Use some recent contain …[truncated]

### L2-fc20f5b114  (L2, 2026-05-13, sha fc20f5b114f9, PR #24125)
TITLE: [AMD] Skip redundant CatArrayBatchedCopy in GLM-5 NSA TileLang decode (#24125)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+48/-18); python/sglang/srt/layers/attention/nsa_backend.py (+12/-2)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ In GLM-5 NSA TileLang decode on ROCm, the fused-rope path dispatches a redundant `CatArrayBatchedCopy<OpaqueType<1u>, ...>` kernel once per layer per decode step that rebuilds an already-existing tensor. ⏎  ⏎ The cause: `fused_qk_rope_cat_and_cache_mla` produces a contiguous `q_cat` of shape `(M, num_heads, kv_lora_rank + qk_rope_head_dim)`. The pre-patch flow then ⏎  ⏎ 1. slices `q_cat` into `q_nope_fused` / `q_pe_fused`, ⏎ 2. passes …[truncated]

### L2-642ac9c916  (L2, 2026-05-13, sha 642ac9c916d5, PR #23893)
TITLE: [NPU]pp support mla kv transfer (#23893)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/ascend/conn.py (+64/-17); python/sglang/srt/disaggregation/base/conn.py (+4/-0); python/sglang/srt/disaggregation/decode.py (+1/-0); python/sglang/srt/disaggregation/prefill.py (+1/-0); python/sglang/srt/disaggregation/utils.py (+12/-4); python/sglang/srt/hardware_backend/npu/memory_pool_npu.py (+10/-1)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ pp support mla kv transfer ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ <img width="935" height="73" alt="image" src="https://github.com/user-attachments/assets/e0493592-1a4b-4720-a4c9-20f2bc399644" /> ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md# …[truncated]

### L2-22012ba1bc  (L2, 2026-05-13, sha 22012ba1bc21, PR #17392)
TITLE: Add BF16 support to EP-MoE for DeepGEMM (#17392)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py (+56/-0); python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py (+34/-0); python/sglang/srt/layers/moe/ep_moe/kernels.py (+141/-22); python/sglang/srt/layers/moe/ep_moe/layer.py (+5/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+4/-1); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+159/-13); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+21/-13); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+2/-1); python/sglang/srt/layers/quantization/unquant.py (+23/-2); python/sglang/srt/server_args.py (+5/-0)
LABELS: documentation, quant, run-ci, model-gateway, mthreads
BODY: ## Motivation ⏎ We attempted to enable EP-MoE on BF16 models and found that BF16 data type was not supported in the DeepGEMM backend. This PR implements BF16 EP-MoE support using DeepGEMM's grouped BF16 GEMM kernels. ⏎  ⏎ ## Modifications ⏎ Following the same computational paradigm as FP8, we have implemented EP-MoE for BF16 using `deep_gemm.m_grouped_bf16_gemm_nt_contiguous` and `deep_gemm.m_grouped_bf16_gemm_nt_masked` from DeepGEMM. ⏎  ⏎ The changes …[truncated]

### L2-01a225ac6f  (L2, 2026-05-13, sha 01a225ac6f4a, PR #25001)
TITLE: [LoRA] MLA attention LoRA: q_b_proj / kv_b_proj support (#25001)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+15/-0); python/sglang/srt/models/deepseek_v2.py (+4/-0); python/sglang/srt/lora/deepseek_mla_correction.py (+117/-0); python/sglang/srt/lora/triton_ops/__init__.py (+10/-0); python/sglang/srt/lora/triton_ops/kv_b_lora_absorbed.py (+849/-0); python/sglang/srt/lora/utils.py (+14/-0); python/sglang/srt/utils/common.py (+4/-0)
LABELS: amd, lora, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ DeepSeek-style MLA attention has four projections that an adapter checkpoint may target — `q_a_proj`, `kv_a_proj_with_mqa`, `q_b_proj`, `kv_b_proj` — but on `main` only the `*_a_*` pair is wired up (via the fused `fused_qkv_a_proj_with_mqa`, added in #22323). Adapters that include `q_b_proj` or `kv_b_proj` (e.g. `Kimi-K2.5` LoRA fine-tunes) either fail target-module validation at load time, or silently drop those modules. ⏎  ⏎ `kv_ …[truncated]

### L2-7618ad7075  (L2, 2026-05-13, sha 7618ad707568, PR #24925)
TITLE: [attn backend] Integrate tokenspeed_mla prefill/decode kernels (fp8 kv cache, blackwell) (#24925)
SOURCES: path_core, symbol_pickaxe, release_notes, corpus:production-kernel-provenance
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.backend.tokenspeed_mla, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/attention_registry.py (+11/-0); python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+247/-0); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+132/-91); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+1/-1); python/pyproject.toml (+1/-0); python/sglang/srt/model_executor/model_runner.py (+2/-0); python/sglang/srt/models/deepseek_common/attention_backend_handler.py (+7/-0); python/sglang/srt/models/deepseek_common/utils.py (+1/-0); python/sglang/srt/server_args.py (+20/-0); python/sglang/srt/speculative/draft_utils.py (+28/-0); (+1 more)
LABELS: high priority, dependencies, blackwell, run-ci, bypass-fastfail
BODY: ## Motivation ⏎ Some future works: ⏎ - don't need to init trtllm/flashinfer workspace first ⏎ - add more tests in ci ⏎ - add doc ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ needs to install first ⏎ ``` ⏎ pip install tokenspeed_mla ⏎ ``` ⏎  ⏎ verified with gpqa, gsm8k ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-proj …[truncated]

### L2-e2290b155a  (L2, 2026-05-13, sha e2290b155aa0, PR #24890)
TITLE: Port KV Compression V2 from deepseek_v4_dev (#24890)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.optimization.weight_absorption
FILES: python/sglang/jit_kernel/csrc/deepseek_v4/c128_online_v2.cuh (+875/-0); python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh (+208/-303); python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh (+405/-0); python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh (+827/-0); python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh (+419/-0); python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh (+629/-0); python/sglang/jit_kernel/csrc/deepseek_v4/rmsnorm.cuh (+15/-14); python/sglang/jit_kernel/deepseek_v4.py (+127/-2); python/sglang/jit_kernel/dsv4/__init__.py (+10/-0); python/sglang/jit_kernel/dsv4/compress.py (+349/-0); (+13 more)
LABELS: high priority, deepseek, run-ci, jit-kernel
BODY: Port the fused KV compression V2 kernels (c4, c128, online c128) from `deepseek_v4_dev` branch to main.

### L2-ff70aeac30  (L2, 2026-05-14, sha ff70aeac3051, PR #24491)
TITLE: [diffusion] feat: add performance mode server args (#24491)
SOURCES: path_core
ARTIFACT_HINTS: L2.kernel.rocm_mla_decode_rope, L2.pool.mla_token_kv
FILES: python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py (+1/-1); docs/diffusion/api/cli.md (+3/-0); docs/diffusion/api/openai_api.md (+7/-5); docs/diffusion/performance/deployment_cookbook.md (+94/-0); docs/diffusion/performance/index.md (+2/-0); python/sglang/benchmark/utils.py (+10/-7); python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/qwen_image.py (+1/-1); python/sglang/multimodal_gen/configs/models/dits/wanvideo.py (+1/-0); python/sglang/multimodal_gen/configs/pipeline_configs/base.py (+6/-0); python/sglang/multimodal_gen/configs/pipeline_configs/ltx_2.py (+9/-0); (+69 more)
LABELS: documentation, amd, lora, deepseek, npu, run-ci, diffusion, model-gateway
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ - Add `--performance-mode` / `--mode` for diffusion server defaults: `auto`, `throughput`, `memory`, and `balanced`, with `aggressive`, `conservative`, and `balance` aliases. ⏎ - Auto-select FSDP+CFG only for high-confidence multi-GPU Qwen/Wan CFG cases, gated by the least available memory across selected GPUs. ⏎ - Move the auto-tune decision logic into `runtime/server_args_auto_tune.py`; `ServerArgs` now only invokes the resolver. ⏎ - …[truncated]

### L2-426dd339da  (L2, 2026-05-14, sha 426dd339da2a, PR #25139)
TITLE: Migrate Intel CPU cases to the test/registered (#25139)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-xeon.yml (+1/-1); .pre-commit-config.yaml (+1/-0); scripts/ci/check_registered_tests.py (+2/-2); test/registered/cpu/test_activation.py (+59/-0); test/registered/cpu/test_binding.py (+30/-0); test/registered/cpu/test_bmm.py (+98/-0); test/registered/cpu/test_causal_conv1d.py (+330/-0); test/registered/cpu/test_cpu_graph.py (+91/-0); test/registered/cpu/test_decode.py (+172/-0); test/registered/cpu/test_extend.py (+225/-0); (+17 more)
LABELS: run-ci
BODY: # Summary ⏎ PR to improve CPU CI coverage and refine Xeon CI triggering. ⏎  ⏎ # Changes ⏎  ⏎ #### 1. Run `stage-b-test-cpu` through unified test runner. ⏎ #### 2. Update CPU per-commit suite to include `stage-b-test-cpu`. ⏎ #### 3. Register CPU tests ⏎   -   22 new CPU test files were added under `test/registered/cpu/`. ⏎   -   Coverage now includes key CPU kernel/operator paths: activation, binding, bmm, causal_conv1d, cpu_graph, decode/extend, flash_att …[truncated]

### L2-22dfcdaa04  (L2, 2026-05-14, sha 22dfcdaa0438, PR #25310)
TITLE: revert flashinfer 0.6.11 bumps (#25310)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+4/-4); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/flashinfer_comm_fusion.py (+13/-11); python/sglang/srt/layers/quantization/fp4_utils.py (+7/-7); python/sglang/srt/utils/common.py (+1/-1); test/registered/moe/test_cutedsl_moe.py (+6/-9)
LABELS: dependencies
DEEP_STUDY: deep-study revert record: explicit_rollback of PR(s) 24452;25129 reason=crash_or_hang
BODY: Reverts #24452 and #25129. The 0.6.11 bump causes a CUDA illegal-address crash in the triton_kernels mxfp4 MoE matmul (`_matmul_ogs_NNT_bf16xbf16xmxfp4_128x256x128x1_swiglu`) during piecewise CUDA graph capture for gpt-oss-120b on 4xH100. Same failure signature on the original PR CI and on main scheduled CI. Pin back to 0.6.8.post1 until upstream fixes the mxfp4 MoE path. ⏎  ⏎ Failure links: ⏎ - PR #24452 CI: https://github.com/sgl-project/sglang/actio …[truncated]

### L2-dca9ba6321  (L2, 2026-05-14, sha dca9ba63215d, PR #25311)
TITLE: perf(mla): TMA bulk-store set_mla_kv_buffer (up to 12× over baseline) (#25311)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.kernel.set_mla_kv_buffer
FILES: python/sglang/jit_kernel/csrc/elementwise/set_mla_kv_buffer.cuh (+249/-0); python/sglang/jit_kernel/set_mla_kv_buffer.py (+121/-0); python/sglang/jit_kernel/benchmark/bench_set_mla_kv_buffer.py (+127/-0); python/sglang/jit_kernel/tests/test_set_mla_kv_buffer.py (+126/-0); python/sglang/srt/mem_cache/utils.py (+55/-5)
LABELS: run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ The MLA paged-KV scatter-write (`set_mla_kv_buffer`) was a 1D Triton kernel (`BLOCK=128`, grid `(n_loc, ceil(total_dim/BLOCK))`). It is reasonable at very small batch sizes but degrades linearly with `n_loc` — **83.5 µs at bs=16384** on GB300. For DeepSeek-V4 prefill (61 layers × thousands of locs per step) this is a meaningful chunk of layer time. ⏎  ⏎ This PR introduces two changes, both backed by the new dispatcher in `set_mla_kv_bu …[truncated]

### L2-ad4994dc1d  (L2, 2026-05-14, sha ad4994dc1d6f, PR #25279)
TITLE: DeepseekV2MoE: defer shared experts when routed kernel is non-mutating (#25279)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+11/-3)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ When the routed-MoE kernel does **not** mutate its input (`inplace=False`, e.g. `flashinfer_trtllm_routed`), `hidden_states` is preserved across `self.experts(...)`. In that case the existing ordering — compute `shared_experts` **before** the routed call — forces the shared-experts activations and the routed-MoE workspace to coexist on-device, inflating transient peak memory at long prefill on Kimi-K2.5-NVFP4 / Blackwell. ⏎  ⏎ ## Modif …[truncated]

### L2-0c19540550  (L2, 2026-05-15, sha 0c19540550e1, PR #25335)
TITLE: [Fix] Fix gpt oss triton kernels and upgrade flashinfer back to 0.6.11.post1 (#25335)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+2/-2); python/pyproject.toml (+5/-5); python/sglang/srt/entrypoints/engine.py (+2/-2); python/sglang/srt/layers/flashinfer_comm_fusion.py (+11/-13); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+4/-3); python/sglang/srt/layers/moe/moe_runner/triton_kernels.py (+6/-2); python/sglang/srt/layers/moe/topk.py (+44/-1); python/sglang/srt/layers/quantization/fp4_utils.py (+7/-7); python/sglang/srt/layers/quantization/mxfp4.py (+46/-3); python/sglang/srt/utils/common.py (+10/-2); (+3 more)
LABELS: high priority, dependencies, deepseek, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ co-author: @b8zhong @mmangkad  ⏎ Modified upon #25312  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sg …[truncated]

### L2-ee93795476  (L2, 2026-05-15, sha ee93795476a4, PR #25333)
TITLE: perf(mla): hybrid Triton fused cat+FP8-quantize for MLA chunked-prefill K/V (#25333)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.kernel.mla_kv_pack_quantize_fp8
FILES: python/sglang/jit_kernel/mla_kv_pack_quantize_fp8.py (+295/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py (+28/-12); python/sglang/jit_kernel/benchmark/bench_mla_kv_pack_quantize_fp8.py (+214/-0); python/sglang/jit_kernel/tests/test_mla_kv_pack_quantize_fp8.py (+104/-0)
LABELS: quant, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ The MLA chunked-prefill path packs strided `k_nope` and broadcast `k_pe` into a contiguous K tensor and (in the FP8 path) FP8-quantizes K and V. The straightforward implementation is three sequential ops (concat + 2× per-tensor FP8 quantize), which dispatches the same `(s, num_heads)` work three times, round-trips intermediate BF16/FP16 values through gmem between the concat and the quantize, and cannot share PDL hand-off. ⏎  ⏎ This PR …[truncated]

### L2-34cb8e2842  (L2, 2026-05-15, sha 34cb8e28425d, PR #24130)
TITLE: fix(sgl-kernel): sm90 compile flashmla failed (#24130)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: sgl-kernel/cmake/flashmla.cmake (+26/-16); sgl-kernel/csrc/flashmla_extension.cc (+2/-0)
LABELS: sgl-kernel, run-ci
ISSUES: #24126 [Bug] sgl-kernel fails to compile FlashAttention on CUDA 12.8 with H20
BODY: ## Motivation ⏎  ⏎ ### close https://github.com/sgl-project/sglang/issues/24126 ⏎  ⏎ Fixes a FlashMLA build issue on Hopper GPUs such as H20 when buildingwith CUDA 12.8. ⏎  ⏎ In `sgl-kernel/cmake/flashmla.cmake`, the `sm100` Blackwell source files were always appended to `FlashMLA_SOURCES`, even when the build only targeted Hopper (`sm90a`). At the same time,`sm100` gencode was only added when `CUDA_VERSION > 12.8`. ⏎  ⏎ That mismatch meant a CUDA 12.8 H …[truncated]

### L2-aec4022e58  (L2, 2026-05-16, sha aec4022e58c6, PR #25424)
TITLE: [Spec] Clean up draft-window-size handling; extract spec arg setup to arg_groups (#25424)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/argparse_actions.py (+82/-0); python/sglang/srt/arg_groups/speculative_hook.py (+443/-0); python/sglang/srt/models/llama_eagle3.py (+12/-15); python/sglang/srt/server_args.py (+21/-473); python/sglang/srt/speculative/dflash_worker.py (+2/-3); python/sglang/srt/speculative/frozen_kv_mtp_worker.py (+1/-1); test/registered/unit/server_args/test_server_args.py (+13/-10)
LABELS: speculative-decoding, run-ci
BODY: ## Summary ⏎  ⏎ - Tighten `--speculative-draft-window-size` handling (validation, scope warning, deprecation alias) ⏎ - Clean up EAGLE-3 SWA wiring in `LlamaForCausalLMEagle3` (drop kwarg threading, post-init loop) ⏎ - Extract speculative arg setup from `server_args.py` to new `arg_groups/speculative_hook.py`, split per algorithm ⏎ - Move generic argparse `Action` classes to new `arg_groups/argparse_actions.py` ⏎  ⏎ References #24664. ⏎  ⏎ ## Changes ⏎  ⏎ **`--specula …[truncated]

### L2-2f81718773  (L2, 2026-05-16, sha 2f81718773ab, PR #25321)
TITLE: [attn backend] avoid initing parent class's workspace buffer (#25321)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.backend.tokenspeed_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+52/-36); python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+1/-0); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+17/-11)
LABELS: blackwell, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 25488 (confirmed_revert, reason=unstated)
BODY: ## Motivation ⏎ future works ⏎ - stop trtllm_mla from inheriting flashinfer_mla ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-proje …[truncated]

### L2-9869ef0849  (L2, 2026-05-16, sha 9869ef08495c, PR #25488)
TITLE: Revert "[attn backend] avoid initing parent class's workspace buffer" (#25488)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.backend.tokenspeed_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+36/-52); python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+0/-1); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+11/-17)
LABELS: blackwell
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 25321 reason=unstated
BODY: Reverts sgl-project/sglang#25321 ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: **Missing `run-ci` label** — add it to run CI tests. ⏎ Latest PR Test (Extra): :x: **Blocked** — `run-ci` is required first.

### L2-0f50ed86c9  (L2, 2026-05-16, sha 0f50ed86c91e, PR #25476)
TITLE: fix(pd): fix kv pools without end_layer (#25476)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/base/conn.py (+1/-1); python/sglang/srt/disaggregation/prefill.py (+1/-1)
BODY: PR #24704 unconditionally reads self.token_to_kv_pool.end_layer in PrefillBootstrap._init_kv_manager, but HybridLinearKVPool (used by Qwen3-Next) only defines start_layer, causing: ⏎  ⏎     AttributeError: 'HybridLinearKVPool' object has no attribute 'end_layer' ⏎  ⏎ prefill_end_layer is only meaningful for compressed-MLA pools (DeepSeek-V4); the downstream consumer in common/conn.py already uses getattr(..., None). Make the source side equally toler …[truncated]

### L2-4ef9bad223  (L2, 2026-05-16, sha 4ef9bad223b5, PR #25208)
TITLE: [AMD] ci: register 5 framework tests to run on AMD CI (#25208)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/core/test_engine_child_pids.py (+2/-1); test/registered/observability/test_tracing.py (+2/-1); test/registered/prefill_only/test_pooled_hidden_states.py (+9/-2); test/registered/sessions/test_session_control.py (+11/-2); test/registered/sessions/test_streaming_session.py (+2/-1)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ First batch of the NV→AMD coverage gap audit. On `main`, ~200 tests are registered for CUDA via `register_cuda_ci(...)` but not for AMD. This PR adds `register_amd_ci(...)` to 5 framework-level tests that have no NVIDIA-specific kernel paths in their test body, so they run on AMD CI per-commit. ⏎  ⏎ ## What now runs on AMD per-commit ⏎  ⏎ | AMD suite | File | est_time | ⏎ |---|---|---| ⏎ | `stage-b-test-1-gpu-small-amd` | `core/test_engine_child_ …[truncated]

### L2-7158a255eb  (L2, 2026-05-17, sha 7158a255ebec, PR #25525)
TITLE: [MoE Refactor] Migrate flashinfer_cutedsl + DeepEP to MoeRunner (#25525)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+7/-29); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+103/-13); python/sglang/srt/layers/moe/moe_runner/runner.py (+0/-2); python/sglang/srt/layers/quantization/modelopt_quant.py (+44/-101); test/registered/moe/test_cutedsl_moe.py (+2/-1)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎ Part of the MoE refactor roadmap (#8715). The roadmap entry ⏎  ⏎ > [ ] `DeepEPMoE.forward_*` ⏎ >   - [ ] `flashinfer_cutedsl.py` + `modelopt_quant.py` ⏎  ⏎ calls for retiring the legacy DeepEP fused-MoE forward path that bypasses the unified `MoeRunner` framework. This PR deprecates the last two pieces of that path: `DeepEPMoE.forward_flashinfer_cutedsl` and `ModelOptNvFp4FusedMoEMethod.apply_without_routing_weights`. ⏎  ⏎ ## Modifications ⏎  ⏎ The  …[truncated]

### L2-c67b287056  (L2, 2026-05-17, sha c67b2870569a, PR #25006)
TITLE: Enable trtllm_mha as gemma4 default attn backend. (#25006)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+6/-2)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ Enable `trtllm_mha` as the default attention backend for Gemma4 on SM100. ⏎  ⏎ When `--attention-backend` is not specified for `Gemma4ForConditionalGeneration`, SGLang now selects: ⏎  ⏎ - `trtllm_mha` on SM100 ⏎ - `triton` otherwise ⏎  ⏎ This keeps the existing non-SM100 behavior unchanged while enabling the Blackwell-optimized MHA backend for Gemma4 by default. ⏎  ⏎ ## Benchmark ⏎  ⏎ Same server flags otherwise, comparing `triton` vs `trtllm_ …[truncated]

### L2-866793c502  (L2, 2026-05-18, sha 866793c502b7, PR #24933)
TITLE: Amd/deepseek v4 rebase main 0509 (#24933)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.kernel.hip_flash_mla, L2.backend.sparse_mla_adapters, L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+17/-4); python/sglang/srt/layers/attention/hip_flash_mla.py (+197/-0); python/sglang/jit_kernel/deepseek_v4.py (+26/-0); python/sglang/srt/environ.py (+7/-0); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+1265/-0); python/sglang/srt/layers/attention/dsv4/compress_hip.py (+455/-0); python/sglang/srt/layers/attention/dsv4/compressor.py (+16/-2); python/sglang/srt/layers/attention/dsv4/indexer.py (+8/-11); python/sglang/srt/layers/attention/nsa/index_buf_accessor.py (+0/-4); python/sglang/srt/layers/attention/nsa/tilelang_kernel.py (+1214/-2); (+7 more)
LABELS: high priority, quant, amd, deepseek, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ Enable deepseek v4 model support (merge to main) to ROCm platform. ⏎  ⏎ This is a first PR to add ROCm deepseek v4 model support on SGLang main. ⏎  ⏎ After this PR merged, we can run the v4 flash/pro model on MI35x in eager mode. Then will have subsequent PRs to merge remaining DSv4 optimizations from amd/deepseek_v4 branch. ⏎  ⏎ Some following tasks is under cooking, included ⏎ - https://github.com/sgl-project/sglang/tree/amd/deepseek_ …[truncated]

### L2-b29e41e8b3  (L2, 2026-05-18, sha b29e41e8b3f1, PR #25547)
TITLE: Respect user override for Gemma4 attention backend (#25547)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/server_args.py (+22/-5)
BODY: ## Summary ⏎ - Follow-up to #25006. The Gemma4 default-backend block was unconditionally overwriting `self.attention_backend`, so `--attention-backend` was silently ignored. ⏎ - Only auto-select (`trtllm_mha` on sm100, `triton` otherwise) when no backend has been set by the user. ⏎ - If the user did pass `--attention-backend`, assert it is one of `trtllm_mha` or `triton` (the two Gemma4-supported backends). ⏎  ⏎ ## Test plan ⏎  ⏎ 🤖 Generated with [Claude Code] …[truncated]

### L2-f5049709b3  (L2, 2026-05-18, sha f5049709b323, PR #25454)
TITLE: fix(eagle3): drop +1 offset on aux layer ids when first id != 1 (#25454)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_v2.py (+6/-3)
LABELS: deepseek, run-ci
BODY: EAGLE3 drafts ship `eagle_aux_hidden_state_layer_ids` under two conventions: ⏎  ⏎ - "output-of-layer-X": first id == 1 (output of the first transformer block; embed-output is never used as aux). sglang's capture loop fires BEFORE layer i, so to capture output-of-X we need to set layers_to_capture to [X+1] — hence the historic `+1`. ⏎ - "input-to-layer-X" / "capture-before-X": first id == 2 (input to layer 2 = output of layer 1, same physical hidden  …[truncated]

### L2-1f185c6ba8  (L2, 2026-05-18, sha 1f185c6ba83c, PR #25489)
TITLE: Support draft extend cuda graph for tokenspeed_mla attention backend (#25489)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.tokenspeed_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+8/-14); python/sglang/srt/speculative/eagle_worker_v2.py (+2/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-b79e4b1e68  (L2, 2026-05-18, sha b79e4b1e687b, PR #25690)
TITLE: [Fix] Try to fix error caused by latest cutedsl packages  (#25690)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+2/-2); scripts/ci/cuda/ci_install_dependency.sh (+19/-2)
LABELS: dependencies, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Ref: https://github.com/sgl-project/sglang/actions/runs/26055697810/job/76604443741 ⏎ https://github.com/vllm-project/vllm/pull/40082#issuecomment-4349406309  ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAIN …[truncated]

### L2-d90bc65e30  (L2, 2026-05-19, sha d90bc65e3075, PR #25383)
TITLE: [NPU] Fix TypeError in get_state_buf_infos when index_head_dim is None on MLA (#25383)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/memory_pool_npu.py (+2/-0)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ In PD disaggregation scenarios, NPUMLATokenToKVPool.get_state_buf_infos() unconditionally accesses self.index_k_buffer , which is initialized to None when the model has no index head (i.e., index_head_dim is None ), causing TypeError: 'NoneType' object is not subscriptable . ⏎  ⏎ This issue was triggered by commit d7f4761a ([PD] Refactor hybrid state transfer), which added NPUMLATokenToKVPool to the isinstance check in setup_st …[truncated]

### L2-87c3c96bc8  (L2, 2026-05-19, sha 87c3c96bc8f7, PR #25699)
TITLE: [Bug][PD][NIXL] always send aux on is_last; only expects_state when truthy (#25699)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/nixl/conn.py (+22/-16)
LABELS: format, run-ci-extra
ISSUES: #25698 [Bug][PD][NIXL] disagg hangs on dense models on v0.5.12 — state-gated aux + decode expects_state mismatch (#24932 regression)
BODY: ## Motivation ⏎  ⏎ Fixes #25698. ⏎  ⏎ Two asymmetries introduced in #24932 (`[PD] Refactor hybrid state transfer`) hang every NIXL P/D disagg request for **dense** models (no Mamba / SWA / NSA — i.e. LLaMA, Qwen3, Gemma, GPT-OSS, …) on `v0.5.12`. Bisects cleanly to commit `d7f4761a4`. ⏎  ⏎ 1. `NixlKVManager.transfer_worker` gates the aux RDMA write inside `if kv_chunk.is_last and kv_chunk.state_indices:`. For dense models, `state_indices` is `[]` (falsy from …[truncated]

### L2-b9d470f4a2  (L2, 2026-05-19, sha b9d470f4a2f4, PR #24640)
TITLE: Support spec v2 for FlashMLA speculative decoding (#24640)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.flashmla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashmla_backend.py (+4/-3); python/sglang/srt/models/deepseek_v2.py (+1/-1); test/registered/mla/test_flashmla.py (+6/-8)
LABELS: deepseek, run-ci
ISSUES: #24637 Support spec decoding v2 for flashmla attention backend
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ Fixes #24637. ⏎ `test/registered/mla/test_flashmla.py` currently forces the FlashMLA MTP speculative decoding test to fall back to spec decoding v1 by setting `SGLANG_ENABLE_SPEC_V2=False`. FlashMLA should support spec decoding v2 and the registered CUDA CI test should cover that path explicitly. ⏎ ## Modifications ⏎ - Enable spec decoding v2 in `TestFlashMLAMTP` by changing `SGLANG_ENABLE_SPEC_V2.override(False)` to `SGLANG_ENABLE_SP …[truncated]

### L2-8131641bc6  (L2, 2026-05-20, sha 8131641bc66e, PR #25821)
TITLE: [Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename (#25821)
SOURCES: path_core, symbol_pickaxe, corpus:production-kernel-provenance
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.kernel.hip_flash_mla, L2.backend.sparse_mla_adapters, L2.kernel.set_mla_kv_buffer, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: .claude/skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md (+11/-11); .claude/skills/llm-torch-profiler-analysis/references/overlap-catalog.md (+1/-1); .claude/skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py (+9/-9); .github/workflows/nightly-test-amd-rocm720.yml (+2/-2); .github/workflows/nightly-test-amd.yml (+2/-2); docs/advanced_features/attention_backend.md (+4/-4); docs/advanced_features/hisparse_guide.md (+2/-2); docs/advanced_features/server_arguments.md (+9/-9); docs/basic_usage/deepseek_v32.md (+17/-17); docs/platforms/ascend/ascend_npu_best_practice.md (+2/-2); (+152 more)
LABELS: documentation, high priority, quant, amd, deepseek, npu, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ `NSA` (Native Sparse Attention) is a misnomer — this attention variant is specific to DeepSeek and should be called `DSA` (DeepSeek Sparse Attention). This PR performs a full rename from `nsa` to `dsa` across user-facing CLI/env/registry surfaces and internal files, while preserving backward-compatible aliases for the user-facing layer. ⏎  ⏎ ## Modifications ⏎  ⏎ **User-facing (with deprecated backward-compat aliases):** ⏎ - `ServerArgs` fie …[truncated]

### L2-1a17d753f1  (L2, 2026-05-20, sha 1a17d753f166, PR #25460)
TITLE: [perf] prepare_prefill_qkv hook + fp8 quantize jit kernel (#25460)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.backend.tokenspeed_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+128/-14); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+2/-2); python/sglang/jit_kernel/fp8_quantize.py (+157/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py (+18/-0)
LABELS: high priority, quant, blackwell, run-ci, jit-kernel, bypass-fastfail
DEEP_STUDY: deep-study performance PR ()
BODY: Adds a model-side `prepare_prefill_qkv` extension point on the MLA attention backends so backends can quantize/pack/rope Q/K/V *before* the prefill kernel call, and ships a tokenspeed_mla-side fp8 quantize jit kernel as one such producer: ⏎  ⏎ - python/sglang/jit_kernel/fp8_quantize.py (new) — fp8 e4m3 quantize helper used by the new prepare path. ⏎ - python/sglang/srt/layers/attention/tokenspeed_mla_backend.py — prepare_prefill_qkv path that lands  …[truncated]

### L2-f9f82d238c  (L2, 2026-05-20, sha f9f82d238ca9, PR #25646)
TITLE: fix deepseek v4 hisparse (#25646)
SOURCES: corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/dsv4/compressor_v2.py (+10/-1)
LABELS: run-ci
DEEP_STUDY: deep-study correctness case sglang:f9f82d238c: class=numerical_precision; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Motivation ⏎ gsm8k 200 examples 20shot ⏎ before（HiSparse + compressor_v2）: ⏎ ``` ⏎ 100%|██████████| 200/200 [01:44<00:00,  1.91it/s] ⏎ Accuracy: 0.825 ⏎ Invalid: 0.000 ⏎ Latency: 105.663 s ⏎ Output throughput: 211.806 token/s ⏎  ⏎ ``` ⏎ after （HiSparse + compressor_v2 + fix） ⏎ ``` ⏎ 100%|██████████| 200/200 [01:40<00:00,  1.98it/s] ⏎ Accuracy: 0.960 ⏎ Invalid: 0.000 ⏎ Latency: 101.371 s ⏎ Output throughput: 189.009 token/s ⏎  ⏎ ``` ⏎ HiSparse + SGLANG_OPT_USE_COM …[truncated]

### L2-c3f9bc9818  (L2, 2026-05-21, sha c3f9bc9818c9, PR #24376)
TITLE: Fix nixl mla key and backup skipping (#24376)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/mem_cache/storage/nixl/README.md (+85/-15); python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py (+14/-2); python/sglang/srt/mem_cache/storage/nixl/test_hicache_nixl_storage.py (+40/-2)
LABELS: documentation, hicache, run-ci
BODY: # Motivation ⏎  ⏎ Fix two bugs affecting MLA (Multi-head Latent Attention) models: ⏎  ⏎ 1. Fix incorrect MLA key denominator in batch_exists() ⏎  ⏎ 2. Missing MLA backup skip in NIXL backend. ⏎  ⏎ # Modifications ⏎  ⏎ 1. In `hicache_nixl.py`, add early-return guard in `batch_set_v1()` and `batch_set()` when `backup_skip` is True, returning success without performing any storage operation. ⏎  ⏎ 2. Fix inverted MLA key denominator in batch_exists():  ⏎  ⏎ The ze …[truncated]

### L2-19f55c0e6d  (L2, 2026-05-21, sha 19f55c0e6d6f, PR #25884)
TITLE: [Refactor] major JIT kernel clean up for dsv4 (#25884)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/jit_kernel/csrc/deepseek_v4/topk_1024.cuh (+0/-336); python/sglang/jit_kernel/csrc/deepseek_v4/topk_v1.cuh (+13/-9); python/sglang/jit_kernel/deepseek_v4.py (+0/-1036); python/sglang/jit_kernel/dsv4/__init__.py (+48/-1); python/sglang/jit_kernel/dsv4/attn.py (+195/-0); python/sglang/jit_kernel/dsv4/compress_old.py (+308/-0); python/sglang/jit_kernel/dsv4/elementwise.py (+158/-0); python/sglang/jit_kernel/dsv4/gemm.py (+24/-0); python/sglang/jit_kernel/dsv4/hisparse.py (+27/-0); python/sglang/jit_kernel/dsv4/moe.py (+216/-0); (+13 more)
LABELS: deepseek, jit-kernel
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Split deepseek v4.py into multiple files ⏎ 2. Reuse torch.mm instead of inline cpp cublas handler ⏎ 3. Unify topk.cuh and topk_1024.cuh (rename to topk_v1.cuh) ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINT …[truncated]

### L2-a449ee4822  (L2, 2026-05-21, sha a449ee4822ee, PR #25576)
TITLE: [Deps] Use cu13 extra for nvidia cutlass dsl (#25576)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1)
LABELS: dependencies, run-ci
ISSUES: #25564 [Bug] Qwen-3.5 on B300 (sm_103) crashes in flash-attn-4 cute kernel — assertion at flash_fwd_sm100.py:162 (fix exists in Dao-AILab/flash-attention#2572; sglang needs to bump flash-attn-4)
BODY: ## Summary ⏎  ⏎ Add `[cu13]` extra to `nvidia-cutlass-dsl` dependency since we already default to CUDA 13 and optionally upgrade to 4.5.1 ⏎  ⏎ Should fix #25564 ⏎  ⏎ ## Test Plan ⏎  ⏎ CI ⏎  ⏎  ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #26077315178](https://github.com/sgl-project/sglang/actions/runs/26077315178) ⏎ Latest PR Test (Extra): :warning: **Not enabled** -- add `run-ci-extra` label to opt in.

### L2-c5251a98a9  (L2, 2026-05-21, sha c5251a98a9d4, PR #25983)
TITLE: feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext (#25983)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.backend.flashmla, L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.tokenspeed_mla, L2.backend.aiter_mla, L2.backend.npu_mla, L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py (+9/-5); python/sglang/srt/layers/attention/cutlass_mla_backend.py (+3/-3); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+12/-13); python/sglang/srt/layers/attention/flashmla_backend.py (+4/-4); python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+1/-1); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+4/-4); python/sglang/srt/batch_overlap/operations.py (+62/-7); python/sglang/srt/batch_overlap/two_batch_overlap.py (+5/-9); python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py (+10/-16); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+64/-59); (+67 more)
LABELS: amd, deepseek, blackwell, npu, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ `ForwardBatch` carried four runtime-ref fields — `req_to_token_pool`, `token_to_kv_pool`, `attn_backend`, `hisparse_coordinator` — that have nothing to do with the per-batch input data the dataclass exists for. They were threaded into every model-layer file (`forward_mha.py`, `forward_mla.py`, `qwen3.py`, `deepseek_v4.py`, …), every attention backend, every hand-rolled `ForwardBatch(...)` constructor, and every speculative-decoding …[truncated]

### L2-4ea8282cb7  (L2, 2026-05-21, sha 4ea8282cb7ab, PR #25938)
TITLE: [Revert] nvidia-cutlass-dsl[cu13] 4.5.1 -> 4.5.0 (#25938)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1)
LABELS: high priority, dependencies, run-ci, bypass-fastfail
DEEP_STUDY: deep-study revert record: explicit_rollback of PR(s)  reason=build_or_dependency
BODY: ## Problem ⏎  ⏎ `nvidia-cutlass-dsl[cu13]` has **additive extras** on PyPI: both `-libs-base` AND `-libs-cu13` are installed together when `[cu13]` is requested. They write to the same `site-packages` paths with different content, causing a `GPUModuleOp TypeError` at kernel-compile time ([vllm-project/vllm#40082](https://github.com/vllm-project/vllm/issues/40082)). ⏎  ⏎ The correct libs package to keep depends on GPU family: ⏎  ⏎ | Runner | Required libs | W …[truncated]

### L2-7cf193fe1f  (L2, 2026-05-21, sha 7cf193fe1faf, PR #25753)
TITLE: feat: support HybridLinearKVPool in chunked prefix cache handling (#25753)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py (+7/-3)
LABELS: deepseek, run-ci
ISSUES: #25752 [Bug] Chunked prefix cache assertion fails for hybrid DeepSeek models using HybridLinearKVPool
BODY: ## Motivation ⏎ ## close https://github.com/sgl-project/sglang/issues/25752 ⏎  ⏎ Verified as an effective fix. ⏎                                                                                                                                                                                   ⏎ Hybrid models like Bailing-2.6-Flash use HybridLinearKVPool as token_to_kv_pool (because they have both Mamba/linear layers and full attention layers with MLA). T …[truncated]

### L2-caa9f08294  (L2, 2026-05-21, sha caa9f0829408, PR #25958)
TITLE: [CI] Force-reinstall nvidia-cutlass-dsl-libs-cu13 last to avoid wheel-mix TypeError (#25958)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: python/pyproject.toml (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+29/-0)
LABELS: dependencies, run-ci, bypass-fastfail
BODY: ## Root cause ⏎  ⏎ `nvidia-cutlass-dsl[cu13]` has **additive PyPI extras** — installing it pulls in both `nvidia-cutlass-dsl-libs-base` AND `nvidia-cutlass-dsl-libs-cu13`. The two wheels ship **intentionally-different content for the same paths**: ⏎  ⏎ | Path | `-libs-base` | `-libs-cu13` | ⏎ |------|--------------|--------------| ⏎ | `cutlass/_mlir/dialects/_gpu_ops_gen.py` | calls `super().__init__(self.build_generic(...))` (new-style single object) | call …[truncated]

### L2-c112f7623a  (L2, 2026-05-22, sha c112f7623a6b, PR #26017)
TITLE: Skip init_mha_chunk_metadata in trtllm_mla when not needed (#26017)
SOURCES: path_core, path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+8/-0)
LABELS: blackwell, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-d226f75669  (L2, 2026-05-22, sha d226f7566945, PR #26134)
TITLE: [refactor] unify cuda-graph capture/replay across attention backends (#26134)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+17/-23); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+6/-38); python/sglang/srt/layers/attention/flashinfer_backend.py (+89/-150); python/sglang/srt/layers/attention/triton_backend.py (+284/-327); python/sglang/srt/layers/attention/wave_backend.py (+59/-71)
LABELS: run-ci, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study: this PR was reverted by PR 26166 (confirmed_revert, reason=unstated)
BODY: ## Motivation ⏎  ⏎ \`init_forward_metadata_capture_cuda_graph\` and \`init_forward_metadata_replay_cuda_graph\` share substantial logic across every attention backend but were maintained as independent copies. This caused real divergence: \`WaveAttnBackend\` silently skipped \`get_num_kv_splits\` during capture, \`CutlassMLABackend\` had a spurious \`assert seq_lens_cpu is not None\` (the parameter was never read), and \`FlashInferAttnBackend\` dup …[truncated]

### L2-629b6c6a85  (L2, 2026-05-22, sha 629b6c6a85b9, PR #19918)
TITLE: correct allreduce fusion and dummy_run alignment in SCATTERED MLP mode (moe_dense_tp_size=1) (#19918)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/communicator.py (+5/-0); python/sglang/srt/model_executor/model_runner.py (+4/-2); test/registered/moe/test_hybrid_dp_ep_tp_mtp.py (+1/-0)
LABELS: run-ci
BODY: ## Problem ⏎  ⏎ When `--moe-dense-tp-size 1` is set, dense MLP layers run in **SCATTERED mode** — hidden states are distributed across `attn_tp_size` ranks via reduce-scatter/all-gather instead of a standard TP all-reduce. Two bugs exist in this path. ⏎  ⏎ --- ⏎  ⏎ ### Bug 1 — Garbled output from spurious allreduce fusion (`communicator.py`) ⏎  ⏎ `should_fuse_mlp_allreduce_with_next_layer()` returns `True` in SCATTERED mode, instructing the scheduler to  …[truncated]

### L2-fd3e11973b  (L2, 2026-05-22, sha fd3e11973b5f, PR #24587)
TITLE: [AMD][aiter] Fix cuda_graph_kv_indices OOB under page_size>1 (#24587)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+31/-13)
LABELS: amd, run-ci
DEEP_STUDY: deep-study correctness case sglang:fd3e11973b: class=memory_safety_oob; symptom=illegal_memory_access; introducing=unknown
BODY: ## Motivation ⏎  ⏎ The aiter attention backend crashes on ROCm with ⏎ `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` / ⏎ `HIP error: an illegal memory access was encountered` once the running ⏎ batch's `seq_lens_sum` grows past the cuda-graph KV-indices buffer. ⏎ The crash typically lands at a downstream `torch.cat` / ⏎ `synchronize()` so it looks unrelated, but the actual fault is an OOB ⏎ `tl.store` inside `create_flashinfer_kv_indices_triton`. ⏎  ⏎ Trigge …[truncated]

### L2-83a18e687d  (L2, 2026-05-23, sha 83a18e687d08, PR #26166)
TITLE: Revert "[refactor] unify cuda-graph capture/replay across attention backends (#26134)" (#26166)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+23/-17); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+38/-6); python/sglang/srt/layers/attention/flashinfer_backend.py (+150/-89); python/sglang/srt/layers/attention/triton_backend.py (+330/-284); python/sglang/srt/layers/attention/wave_backend.py (+71/-59)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 26134 reason=unstated
BODY: Reverts #26134. ⏎  ⏎ The conflict in `triton_backend.py` (from #26152 building on top of #26134) is resolved: the second commit re-applies the SWA fix in the context of the pre-#26134 code. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #26329216086](https://github.com/sgl-project/sglang/actions/runs/26329216086) ⏎ Latest PR Test (Extra): :x: [Run #26329216032](https://github.com/sgl-project/sglang/actions/runs/26329216032)

### L2-b0ce16d0c5  (L2, 2026-05-23, sha b0ce16d0c577, PR #23292)
TITLE: [CP] 1/N: Support MLA Prefill Context Parallel (#23292)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla, L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+128/-56); python/sglang/srt/model_executor/cuda_graph_runner.py (+14/-3); python/sglang/srt/model_executor/model_runner.py (+7/-0); python/sglang/srt/models/deepseek_common/attention_backend_handler.py (+7/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+2/-1); python/sglang/srt/models/deepseek_nextn.py (+31/-8); python/sglang/srt/models/deepseek_v2.py (+73/-14); python/sglang/srt/models/deepseek_v4.py (+1/-0); python/sglang/srt/models/deepseek_v4_nextn.py (+1/-0); python/sglang/srt/server_args.py (+57/-3); (+11 more)
LABELS: deepseek, run-ci, run-ci-extra
ISSUES: #22896 [Feature] Prefill Context Parallelism Support for MLA Models
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Part of Context Parallelism series https://github.com/sgl-project/sglang/issues/21788. Upon merge will close https://github.com/sgl-project/sglang/issues/22896 and https://github.com/sgl-project/sglang/issues/22692 ⏎  ⏎ Extend SGLang's prefill context parallelism (CP) to MLA-based models (DeepSeek V3 / R1, Kimi K2.5) on the `fa3` attention backend, unlocking multi-GPU long-prefill throughput for MLA architectures. ⏎  ⏎ **Insight.** M …[truncated]

### L2-cb7b57955d  (L2, 2026-05-23, sha cb7b57955d7b, PR #26170)
TITLE: fix tokenspeed_mla attn kernel jit (#26170)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.tokenspeed_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+3/-2)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-75427c9ca4  (L2, 2026-05-23, sha 75427c9ca426, PR #25843)
TITLE: Route concat MLA to JIT and remove unused downcast (#25843)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/utils.py (+1/-1); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py (+5/-1); python/sglang/jit_kernel/benchmark/bench_cast.py (+0/-106); python/sglang/jit_kernel/cast.py (+0/-52); python/sglang/jit_kernel/csrc/elementwise/cast.cuh (+0/-137); python/sglang/srt/models/sarvam_moe.py (+2/-1)
LABELS: quant, run-ci, jit-kernel, run-ci-extra
BODY: ## Summary ⏎  ⏎ - Route CUDA `concat_mla_k` / `concat_mla_absorb_q` runtime call sites through `sglang.jit_kernel.concat_mla`. ⏎ - Keep the MUSA `concat_mla_k` path on the existing `sgl_kernel` AOT implementation. ⏎ - Remove the unused JIT `downcast_fp8` wrapper, CUDA header, and JIT-only benchmark. ⏎ - Keep `mla_kv_pack_quantize_fp8` and MXFP8 JIT kernels untouched. ⏎  ⏎ ## Rationale ⏎  ⏎ `concat_mla` already has a JIT implementation with JIT-vs-AOT coverage. The …[truncated]

### L2-af8f66940e  (L2, 2026-05-23, sha af8f66940e9b, PR #25898)
TITLE: [AMD] Dsv4/pr1 fix run time issue (#25898)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/jit_kernel/csrc/deepseek_v4/c128_v2.cuh (+2/-2); python/sglang/jit_kernel/csrc/deepseek_v4/c4_v2.cuh (+2/-2); python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh (+21/-9); python/sglang/jit_kernel/csrc/deepseek_v4/fused_norm_rope_v2.cuh (+10/-4); python/sglang/jit_kernel/dsv4/attn.py (+13/-7); python/sglang/jit_kernel/dsv4/elementwise.py (+21/-4); python/sglang/jit_kernel/dsv4/gemm.py (+3/-3); python/sglang/jit_kernel/dsv4/moe.py (+32/-19); python/sglang/jit_kernel/dsv4/topk.py (+10/-4); python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/fp8_utils.cuh (+71/-2); (+22 more)
LABELS: amd, deepseek, sgl-kernel, run-ci, jit-kernel, run-ci-extra
BODY: ## Motivation ⏎  ⏎ DeepSeek-V4 (DSV4) on AMD MI300X/MI350X GPUs encounters multiple runtime failures when serving inference workloads: ⏎  ⏎ 1. **GPU fault (HSA_STATUS_ERROR_EXCEPTION)** — The core `CompressStatePool` is initialized without `swa_page_size`, defaulting to 0. This causes division-by-zero in `translate_from_swa_loc_to_state_loc()`, producing out-of-bounds `state_loc` indices (e.g., 525448 vs buffer size 470528) that trigger a hardware ex …[truncated]

### L2-ed179bf9b2  (L2, 2026-05-24, sha ed179bf9b297, PR #26239)
TITLE: [dsv4] fix multi-step draft on non-cuda-graph path (#26239)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/deepseek_v4_backend.py (+13/-1); python/sglang/srt/speculative/eagle_utils.py (+24/-0); python/sglang/srt/speculative/eagle_worker_v2.py (+10/-6)
LABELS: deepseek
BODY: ## Summary ⏎ - Fix DSv4 EAGLE multi-step draft on the non-cuda-graph path (`init_forward_metadata`); the assert `req_pool_indices.shape[0] == seq_lens.shape[0] == out_cache_loc.shape[0]` in `init_forward_metadata_decode` previously fired with `out_cache_loc.shape=[bs * topk * num_steps]` vs `seq_lens.shape=[bs]` ⏎ - Introduce a shared per-step layout helper so `EagleWorkerV2.draft_forward` and `DeepseekV4AttnBackend` agree on the slice convention ⏎  ⏎ ## …[truncated]

### L2-ec6fcb93cb  (L2, 2026-05-24, sha ec6fcb93cb8b, PR #26241)
TITLE: [perf][spec decoding] Skip common_template in TRTLLMMLAMultiStepDraftBackend init (#26241)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+19/-0)
LABELS: high priority, blackwell, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR ()
BODY: `FlashInferMLAMultiStepDraftBackend.{init_forward_metadata, init_forward_metadata_replay_cuda_graph}` go through `common_template`, which launches the `generate_draft_decode_kv_indices` triton kernel and slices `spec_info.kv_indptr/kv_indices` for each draft step. Both are unnecessary for the trtllm_mla path used by EAGLE: the per-step attention backends already prepare their own kv-indices from `forward_batch.req_pool_indices` + `forward_batch.s …[truncated]

### L2-2b9dd9c8b3  (L2, 2026-05-25, sha 2b9dd9c8b339, PR #22851)
TITLE: [FlashInfer v0.6.10] [RL] [DSv32] [GLM-5] Add `--dsa-topk-backend` and integrate FlashInfer and pytorch topk (#22851)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: docs/advanced_features/server_arguments.md (+1/-0); docs/references/environment_variables.md (+2/-0); docs_new/docs/advanced_features/server_arguments.mdx (+6/-0); docs_new/docs/references/environment_variables.mdx (+10/-0); python/sglang/srt/environ.py (+2/-0); python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py (+271/-0); python/sglang/srt/layers/attention/dsa_backend.py (+46/-53); python/sglang/srt/server_args.py (+12/-0); test/registered/kernels/test_dsa_indexer.py (+356/-1)
LABELS: documentation, run-ci, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎ @humansand ⏎  ⏎ Add `--dsa-topk-backend` for configurable topk backend implementation selection. ⏎  ⏎ `torch.topk` is used by GLM-5 for RL. ⏎ FlashInfer topk has determinism and configurable tie break (https://github.com/flashinfer-ai/flashinfer/pull/3095), and better long context performance. ⏎  ⏎  ⏎ ## Modifications ⏎ - Add `--dsa-topk-backend`, default to existing `sgl-kernel` ⏎ - Integrate flashinfer and torch topk for unfused code path …[truncated]

### L2-3f5e2c7688  (L2, 2026-05-25, sha 3f5e2c768825, PR #26208)
TITLE: [AMD] Dsv4/pr2 compressor opt (#26208)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.kernel.hip_flash_mla, L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/hip_flash_mla.py (+11/-4); docs/diffusion/compatibility_matrix.md (+0/-2); python/sglang/srt/environ.py (+4/-0); python/sglang/srt/layers/activation.py (+14/-0); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+13/-5); python/sglang/srt/layers/attention/dsv4/compress_hip.py (+5/-4); python/sglang/srt/layers/attention/dsv4/compressor.py (+37/-1); python/sglang/srt/layers/attention/dsv4/compressor_v2.py (+516/-25); python/sglang/srt/layers/attention/dsv4/fused_compress_triton.py (+954/-0); python/sglang/srt/layers/attention/dsv4/indexer.py (+131/-73); (+21 more)
LABELS: documentation, amd, deepseek, sgl-kernel, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ This PR improves DeepSeek-V4 inference performance on AMD ROCm by reducing decode/prefill hot-path overhead in compressor, indexer, and fused attention execution. ⏎ It also consolidates kernel options so we can enable high-performance fused paths with clearer runtime flags while maintaining numerical correctness checks. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add DSV4 fused compress implementations (`fused_compress_kernel.py`, `fused_compress_tr …[truncated]

### L2-b66f8e0b96  (L2, 2026-05-26, sha b66f8e0b96c9, PR #26132)
TITLE: Sgl flashmla (#26132)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: sgl-kernel/cmake/flashmla.cmake (+1/-3); sgl-kernel/csrc/flashmla_extension.cc (+72/-2); sgl-kernel/python/sgl_kernel/flash_mla.py (+186/-8); sgl-kernel/include/sgl_kernel_ops.h (+15/-4)
LABELS: high priority, deepseek, sgl-kernel, run-ci, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ Benchmark `deepseek-ai/DeepSeek-V4-Pro` on B200x8. ⏎  ⏎ Serving recipe: ⏎ - env: `SGLANG_DEFAULT_THINKING=1`, `SGLANG_DSV4_REASONING_EFFORT=max` ⏎ - `--tp 8` ⏎ - `--moe-runner-backend flashinfer_mxfp4` ⏎ - EAGLE: `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` ⏎ - `--chunked-prefill-size 4096` ⏎ - `--disable-flashinf …[truncated]

### L2-7ef06bfc06  (L2, 2026-05-26, sha 7ef06bfc06ec, PR #26088)
TITLE: GLM-4.7-Flash: standalone MLA impl and MLA NextN/MTP (#26088)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+7/-4); python/sglang/srt/model_loader/weight_utils.py (+7/-1); python/sglang/srt/models/glm4_moe_lite.py (+603/-81); python/sglang/srt/models/glm4_moe_lite_nextn.py (+182/-0)
LABELS: run-ci, run-ci-extra
BODY: ## Motivation ⏎    ⏎   `glm4_moe_lite` (GLM-4.7-Flash, MLA) had a few issues when running with the ⏎   native SGLang implementation: ⏎  ⏎   - It subclassed the `deepseek_v2.py` wrapper classes, so DSV4/DSA/CP changes ⏎     there kept leaking in and breaking it. ⏎   - The MoE gate computed routing logits in the model dtype (bf16), but GLM ⏎     requires an FP32 gate projection — wrong-precision routing corrupts output. ⏎   - With MTP/EAGLE speculative deco …[truncated]

### L2-dd6f073377  (L2, 2026-05-26, sha dd6f073377f3, PR #26397)
TITLE: Reland "[perf][spec decoding] Skip full-vocab softmax in EAGLE draft when topk == 1 (#26235)" (#26397)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+14/-2); python/sglang/srt/speculative/eagle_worker_v2.py (+23/-4)
LABELS: high priority, run-ci, bypass-fastfail
DEEP_STUDY: deep-study revert record: reland of PR(s) 26235 reason=other || deep-study performance PR ()
BODY: --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #26472585858](https://github.com/sgl-project/sglang/actions/runs/26472585858) ⏎ Latest PR Test (Extra): :white_check_mark: [Run #26472613859](https://github.com/sgl-project/sglang/actions/runs/26472613859)

### L2-e958f4561f  (L2, 2026-05-26, sha e958f4561f93, PR #15829)
TITLE: [feat] Support `extra_buffer` in Mamba2-based models (#15829)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs/advanced_features/server_arguments.md (+1/-1); python/sglang/srt/arg_groups/nemotron_h_hook.py (+1/-2); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+69/-29); python/sglang/srt/layers/attention/mamba/mamba.py (+15/-1); python/sglang/srt/layers/attention/mamba/mamba2_metadata.py (+10/-0); python/sglang/srt/managers/schedule_batch.py (+9/-8); python/sglang/srt/mem_cache/mamba_radix_cache.py (+6/-9); python/sglang/srt/models/falcon_h1.py (+1/-0); python/sglang/srt/models/granitemoehybrid.py (+1/-0); python/sglang/srt/models/nemotron_h.py (+1/-0); (+7 more)
LABELS: documentation, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎  ⏎ Recent updates to Qwen3-Next models enabled running them with both radix cache and overlap scheduler enabled. This PR does the same for Mamba2-based models. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Optionally return intermediate states in the `mamba_chunk_scan_combined` prefill kernel. ⏎ Fix writing locations of intermediate states in `selective_state_update` decode kernel. ⏎ Update MambaMixer2 to return prefill intermediate states, and provide  …[truncated]

### L2-bf5bc23431  (L2, 2026-05-27, sha bf5bc234310b, PR #26395)
TITLE: [AMD] [CI] Add DeepSeek-R1-0528 FP8 HiCache GSM8K test on MI35x (#26395)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/amd/test_deepseek_r1_hicache_mi35x.py (+171/-0)
LABELS: documentation, deepseek, hicache
BODY: ## Motivation ⏎  ⏎ Add per-commit PR-CI coverage for `deepseek-ai/DeepSeek-R1-0528` (native FP8, MLA, aiter attention backend) running on AMD MI35x with the full **L1 + L2 + L3 HiCache hierarchy** enabled (`--enable-hierarchical-cache --hicache-ratio 2 --hicache-storage-backend file`). ⏎  ⏎ The existing nightly DSR1-0528 AMD eval tests (`test_deepseek_r1_eval_mi35x.py`, `test_deepseek_r1_eval_amd.py`) intentionally pass `--disable-radix-cache`, so any re …[truncated]

### L2-14f81a67d9  (L2, 2026-05-27, sha 14f81a67d94c, PR #26421)
TITLE: chore: bump sglang-kernel version to 0.4.3 (#26421)
SOURCES: dependency_pin, release_notes
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci, run-ci-extra
BODY: ## Summary ⏎  ⏎ This PR bumps the `sglang-kernel` version to `0.4.3` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.4.3` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the `sglang-kernel` dependency are updat …[truncated]

### L2-e06058ed62  (L2, 2026-05-27, sha e06058ed624f, PR #26499)
TITLE: [Kernel] Import flash_mla kernels from sglang kernel for deepseek v4 (#26499)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.kernel.hip_flash_mla
FILES: python/sglang/srt/layers/attention/deepseek_v4_backend.py (+3/-3); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+2/-2); python/sglang/srt/layers/attention/hip_flash_mla.py (+1/-1)
LABELS: deepseek
BODY: ## Motivation ⏎ Switch to use sgl-flashmla instead of the upstream flashmla ⏎  ⏎ #26132  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/s …[truncated]

### L2-dea85c30f4  (L2, 2026-05-27, sha dea85c30f48e, PR #23837)
TITLE: Add Ling_2_6 (#23837)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+5/-0); python/sglang/srt/layers/attention/linear/lightning_backend.py (+16/-6); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e.json (+146/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20-3e_down.json (+164/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20.json (+146/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/E=256,N=512,device_name=NVIDIA_H20_down.json (+164/-0); python/sglang/srt/mem_cache/kv_cache_builder.py (+2/-0); python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py (+4/-3); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+80/-24); python/sglang/srt/models/bailing_moe_linear.py (+78/-34); (+2 more)
LABELS: deepseek, run-ci, run-ci-extra
BODY: ## Motivation ⏎ This is used for support Bailing models, Ling-2.6 ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Benchmarking and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review Process ⏎  ⏎ 1. Ping Merge Oncalls to start the PR flow. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.g …[truncated]

### L2-deaba74745  (L2, 2026-05-27, sha deaba74745d7, PR #26383)
TITLE: [AMD][DSV4] DSV4 MTP graph + sparse triton attn optimizations (#26383)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/environ.py (+1/-0); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+35/-26); python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_fused.py (+12/-1); python/sglang/srt/layers/attention/nsa/triton_decode/triton_mla_kernels_decode_optimized.py (+83/-25); python/sglang/srt/layers/sampler.py (+11/-2); python/sglang/srt/models/deepseek_common/amd/__init__.py (+1/-0); python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py (+158/-0); python/sglang/srt/models/deepseek_v4.py (+52/-7); python/sglang/srt/speculative/draft_utils.py (+17/-4); test/registered/ops/test_aiter_greedy_sample_amd.py (+289/-0)
LABELS: amd, deepseek, run-ci, run-ci-extra
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ The long-standing draft #25552 collected 85 commits of AMD / DSV4 work over several rebases. Most of the optimization PRs it tracked (compressor opts, fused topk, sparse MLA kernels, fused compress, etc.) have already been merged into `main` through smaller follow-up PRs (e.g. #26208, #25251, #25878, #25977, #26014). What remains is a small but meaningful set of changes needed to: ⏎  ⏎ 1. Make DSV4 + EAGLE/MTP speculative decoding  …[truncated]

### L2-50e0b3b77f  (L2, 2026-05-28, sha 50e0b3b77f27, PR #24737)
TITLE: Support Flashinfer Cute-DSL MLA attention (#24737)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.trtllm_mla, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/attention_registry.py (+9/-0); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+35/-9); python/sglang/srt/model_executor/model_runner.py (+2/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+2/-1); python/sglang/srt/models/deepseek_common/utils.py (+1/-0); python/sglang/srt/server_args.py (+30/-0); python/sglang/srt/speculative/draft_utils.py (+11/-2); docs_new/docs/advanced_features/attention_backend.mdx (+11/-1)
LABELS: documentation, blackwell, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ @nvpohanh ⏎  ⏎ Ref:  ⏎ https://github.com/flashinfer-ai/flashinfer/pull/2805 ⏎ https://github.com/flashinfer-ai/flashinfer/pull/2743 ⏎  ⏎ (closed, could need new/reopened PR) Flashinfer autotune PR: https://github.com/flashinfer-ai/flashinfer/pull/3086 ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ Add as a new backend (for the purposes of debugging and easily switching impls for now, **ideally**, in the future, the `trtllm_mla` backend will still be allo …[truncated]

### L2-8ca09a30f1  (L2, 2026-05-28, sha 8ca09a30f18d, PR #26515)
TITLE: Allow Optional key/value in unified_attention_with_output split-op (MLA absorb fix) (#26515)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/radix_attention.py (+6/-4)
BODY: ## Motivation ⏎  ⏎ `unified_attention_with_output` is the split-op entry point used by the piecewise / breakable CUDA graph runner (PCG/BCG) to dispatch through the active attention backend. MLA's absorb path calls `RadixAttention.forward(q, k=None, v=None, ...)` because compressed latent KV is read from the token-to-kv-pool inside the MLA backend rather than passed in. ⏎  ⏎ ### Bug ⏎  ⏎ The split-op required non-None `key` and `value`. When MLA was routed t …[truncated]

### L2-93445e6359  (L2, 2026-05-28, sha 93445e6359f8, PR #26506)
TITLE: [spec decoding] support kimi-k2.6-eagle3.1-mla draft (#26506)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/kimi_k25_eagle3.py (+38/-5)
LABELS: high priority, run-ci
BODY: ## Summary ⏎  ⏎ Adds support for the EAGLE3.1 variant of the Kimi MLA draft, e.g. ⏎ [`lightseekorg/kimi-k2.6-eagle3.1-mla`](https://huggingface.co/lightseekorg/kimi-k2.6-eagle3.1-mla). ⏎ EAGLE3.1 is the same single-layer MLA layout as EAGLE3 with two extra optional config flags: ⏎  ⏎ - `fc_norm`: per-chunk `RMSNorm` applied to each auxiliary hidden state before the `fc` projection. ⏎ - `norm_output`: emit post-norm (rather than pre-norm) hidden states as the a …[truncated]

### L2-435c4ffb30  (L2, 2026-05-28, sha 435c4ffb3081, PR #26609)
TITLE: [CI] Clean DeepSeek V4 tests and installation scripts (#26609)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+0/-8); .github/workflows/pr-test-extra.yml (+15/-0); .github/workflows/pr-test.yml (+12/-12); scripts/ci/cuda/ci_install_deepep.sh (+24/-1); scripts/ci/cuda/ci_install_dsv4_dep.sh (+0/-161); scripts/ci/runner_configs.yml (+2/-4); test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py (+1/-1); test/registered/disaggregation/test_disaggregation_dsv4.py (+1/-1); test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py (+2/-2); test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py (+2/-2); (+3 more)
LABELS: deepseek
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-f66f56c6bd  (L2, 2026-05-28, sha f66f56c6bd1a, PR #26517)
TITLE: Add attention-backend unit-test suite under test/registered/attention/unittest (#26517)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/test/kits/attention_unittest/__init__.py (+1/-0); python/sglang/test/kits/attention_unittest/attention_methods/__init__.py (+1/-0); python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py (+1244/-0); python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py (+1815/-0); python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py (+1575/-0); python/sglang/test/kits/attention_unittest/attention_methods/dual_chunk_attention.py (+1089/-0); python/sglang/test/kits/attention_unittest/attention_methods/gdn_attention.py (+1072/-0); python/sglang/test/kits/attention_unittest/attention_methods/kda_attention.py (+1132/-0); python/sglang/test/kits/attention_unittest/attention_methods/lightning_attention.py (+1038/-0); python/sglang/test/kits/attention_unittest/attention_methods/mamba2_attention.py (+1076/-0); (+59 more)
LABELS: documentation, deepseek, speculative-decoding, blackwell
BODY: ## Motivation ⏎  ⏎ `test/registered/attention/` currently has only server-level / end-to-end tests for attention backends. When a backend regresses, those tests fail late and don't isolate which `(backend × method × forward mode × shape × runner mode × layout)` combination broke. ⏎  ⏎ This PR adds a **module-level correctness suite** that exercises every supported combination in isolation, comparing against an independent PyTorch reference. A failure her …[truncated]

### L2-be32df33b9  (L2, 2026-05-28, sha be32df33b951, PR #26437)
TITLE: [MUSA] Fix startup with patched torchada (#26437)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: 3rdparty/amd/wheel/sglang/pyproject.toml (+1/-1); python/pyproject_other.toml (+1/-1); python/sglang/jit_kernel/utils.py (+7/-1); sgl-kernel/pyproject_musa.toml (+1/-1)
LABELS: dependencies, sgl-kernel, run-ci, mthreads, jit-kernel
BODY: ## Summary ⏎  ⏎ - Bump MUSA torchada requirements to `>=0.1.57` so SGLang picks up MooreThreads/torchada#70. ⏎ - Skip CUDA PDL arch probing on MUSA runtime. ⏎  ⏎ ## Motivation ⏎  ⏎ SGLang can fail during MUSA startup with: ⏎  ⏎ `AttributeError: module 'triton.language.extra' has no attribute 'cuda'` ⏎  ⏎ The torchada-side patch is available in https://github.com/MooreThreads/torchada/pull/70, so this updates the dependency and avoids the CUDA-specific PDL p …[truncated]

### L2-a42a7654a2  (L2, 2026-05-29, sha a42a7654a261, PR #25880)
TITLE: Update MooncakeStore batch tests to use v1 APIs (#25880)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/mem_cache/storage/mooncake_store/test_mooncake_store.py (+84/-78)
BODY: ## Motivation ⏎  ⏎ The existing MooncakeStore batch test was written against an older interface and no longer matched the current HiCache storage workflow.  ⏎  ⏎ This PR refreshes the test so it covers the current MooncakeStore batch API behavior for both MHA and MLA configurations. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #26613615937](https://github.com/sgl-project/sglang/actions/runs/26613615937) ⏎ Latest PR Test (Extra): :x: [Run #266136 …[truncated]

### L2-9062f583db  (L2, 2026-05-29, sha 9062f583db1a, PR #25463)
TITLE: [ROCm] Eliminate redundant contiguous copy in MLA attention on ROCm MXFP4 (#25463)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+21/-5)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ On the ROCm MXFP4 decode path (`_use_aiter_gfx95` with `w_vc.dtype == torch.uint8`), the `batched_gemm_afp4wfp4_pre_quant` kernel outputs `attn_bmm_output` in `(heads, batch, v_head_dim)` layout. The current code then does: ⏎  ⏎ ```python ⏎ attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2) ⏎ ``` ⏎  ⏎ The `.transpose(0, 1)` makes the tensor non-contiguous, so `.flatten(1, 2)` triggers an implicit `.contiguous()` copy — app …[truncated]

### L2-ec075d8bc5  (L2, 2026-05-29, sha ec075d8bc5ff, PR #26651)
TITLE: Fix DRAFT_EXTEND_V2 CG metadata: align test fixture and Triton with production seq_lens convention (#26651)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/triton_backend.py (+37/-4); python/sglang/test/kits/attention_unittest/runner_modes/speculative_draft_extend_runner.py (+12/-7); test/registered/attention/unittests/KNOWN_FAILURES.md (+0/-2); test/registered/attention/unittests/dense/README.md (+3/-13); test/registered/attention/unittests/dense/test_fa3.py (+0/-6); test/registered/attention/unittests/dense/test_fa4.py (+0/-6)
LABELS: documentation, speculative-decoding, run-ci
BODY: ## Summary ⏎  ⏎ - Make the DRAFT_EXTEND_V2 test fixture (`_set_draft_extend_v2_prefix_lens`) match production convention: `seq_lens = prefix + extend`, mirroring the bump that `eagle_info_v2.py` applies before calling `init_forward_metadata`. ⏎ - Fix the Triton CG capture/replay paths for DRAFT_EXTEND_V2: derive `extend_prefix_lens = seq_lens - extend_seq_lens` (from `spec_info.extend_seq_lens_tensor`) when building `kv_indptr/kv_indices`. Without this …[truncated]

### L2-3bdea78ad1  (L2, 2026-05-29, sha 3bdea78ad11d, PR #26565)
TITLE: model: support Step-3.7-Flash (#26565)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs_new/cookbook/autoregressive/StepFun/Step-3.7-Flash.mdx (+324/-0); docs_new/cookbook/autoregressive/StepFun/Step3.5.mdx (+1/-1); docs_new/docs.json (+1/-0); docs_new/src/snippets/autoregressive/step-37-flash-deployment.jsx (+394/-0); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/model_config.py (+12/-1); python/sglang/srt/configs/step3p5.py (+2/-0); python/sglang/srt/configs/step3p7.py (+97/-0); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+18/-2); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+1/-0); (+7 more)
LABELS: documentation, high priority, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-1c2857b064  (L2, 2026-05-29, sha 1c2857b0646f, PR #25960)
TITLE: bugfix: --decrypted-draft-config-file not applied (#25960)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/speculative_hook.py (+9/-1)
LABELS: speculative-decoding, run-ci
BODY: ## Motivation ⏎  ⏎ Previously, --decrypted-draft-config-file was not applied when getting config.json for speculative draft model. ⏎  ⏎ server_args.speculative_algorithm = _resolve_speculative_algorithm_alias( ⏎                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ⏎   File "/home/wzy/sgl-sglang/python/sglang/srt/arg_groups/speculative_hook.py", line 24, in _resolve_speculative_algorithm_alias ⏎     cfg = get_config( ⏎           ^ …[truncated]

### L2-7619e77b7b  (L2, 2026-05-29, sha 7619e77b7ba5, PR #26466)
TITLE: [NPU] chore: basic software upgrade (#26466)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+21/-9); .github/workflows/nightly-test-npu.yml (+6/-6); .github/workflows/pr-test-npu.yml (+30/-19); .github/workflows/release-docker-npu-nightly.yml (+2/-2); .github/workflows/release-docker-npu.yml (+2/-2); scripts/ci/npu/npu_ci_install_dependency.sh (+14/-21)
LABELS: npu, run-ci
BODY: --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #26567132258](https://github.com/sgl-project/sglang/actions/runs/26567132258) ⏎ Latest PR Test (Extra): :x: [Run #26567132127](https://github.com/sgl-project/sglang/actions/runs/26567132127)

### L2-ff8ed7a302  (L2, 2026-05-29, sha ff8ed7a302d7, PR #26665)
TITLE: [refactor] unify cuda-graph capture/replay across attention backends (#26665)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.flashmla, L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.aiter_mla, L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+17/-23); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+25/-49); python/sglang/srt/layers/attention/flashmla_backend.py (+49/-135); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+50/-53); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+30/-35); python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py (+9/-12); python/sglang/srt/layers/attention/aiter_backend.py (+15/-418); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+31/-34); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+31/-34); python/sglang/srt/layers/attention/dsa_backend.py (+43/-5); (+9 more)
LABELS: deepseek, blackwell, npu, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Reland of #26134 (reverted by #26166), extended to cover four additional backends. Single PR replacing the stacked split series — #26144 (round 2), #26159 (round 3), #26160 (round 4), #26162 (round 5) — which were chained off each other's branches and could not land independently without sequential rebases. ⏎  ⏎ ## Modifications ⏎  ⏎ Unify `init_forward_metadata_{capture,replay}_cuda_graph` across attention backends. Capture either delegat …[truncated]

### L2-6ea69efb7f  (L2, 2026-05-29, sha 6ea69efb7f65, PR #26744)
TITLE: [RL] Forward Kimi K2.5 weight hooks to language model (#26744)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/kimi_k25.py (+18/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Kimi K2.5 needs these wrapper forwards for RL weight updates. RL weight sync code operates on the top-level model wrapper, but the actual language-model weight loading metadata lives on the inner language model. Without forwarding `stacked_params_mapping`, `expert_params_mapping`, `mutate_weight_preload()`, and `custom_scale_remap()`, fused and expert weights can fail to map through the wrapper during RL updates. ⏎  ⏎ The wrapper also  …[truncated]

### L2-716e670d3d  (L2, 2026-05-30, sha 716e670d3dc0, PR #26696)
TITLE: [bugfix]: size CuteDSL MoE allgather buffers for the worst-case forward (#26696)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+8/-18); python/sglang/srt/layers/quantization/modelopt_quant.py (+1/-1); python/sglang/srt/server_args.py (+59/-31); test/registered/unit/server_args/test_server_args.py (+41/-0)
LABELS: documentation, quant, run-ci
BODY: ## Motivation ⏎  ⏎ `--moe-runner-backend flashinfer_cutedsl` on the standard (non-A2A) allgather path crashes at serving time: ⏎  ⏎ ``` ⏎ ValueError: num_tokens (4073) exceeds max_num_tokens (2048) ⏎ ``` ⏎  ⏎ Regression from #22669: `ensure_cutedsl_wrapper` sizes the `CuteDslMoEWrapper` CUDA-graph buffers from the **first forward's** token count (`max(num_tokens, 1)`) — a small startup capture batch. A larger real forward then trips FlashInfer's `run()` guard an …[truncated]

### L2-fe4b29d391  (L2, 2026-05-30, sha fe4b29d3911a, PR #26705)
TITLE: [Bugfix] Fix Ascend NPU CP attention for batch size > 1 (#26705)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+13/-10)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ #23269 generalized attention context-parallel (CP) prefill from `bs == 1` to `bs > 1` by reshaping `ContextParallelMetadata` from single scalars into per-sequence lists/tensors. That change updated the CUDA FlashAttention path and the DSA indexer, but **missed the Ascend NPU FIA CP path** in `do_cp_attn_fia`. As a result, on NPU the CP-extend (prefill) attention now: ⏎  ⏎ 1. **Raises `AttributeError`** — it still reads the removed  …[truncated]

### L2-118465f5b5  (L2, 2026-05-31, sha 118465f5b5e3, PR #26824)
TITLE: [attn backend] Make spec_v2 seq_lens_cpu optional in trtllm_mla backend (#26824)
SOURCES: path_core, path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+21/-0)
LABELS: high priority, blackwell, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ After ⏎ <img width="963" height="781" alt="image" src="https://github.com/user-attachments/assets/6d6b6f40-aef2-4e38-aa7b-a41e69e2e765" /> ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-mer …[truncated]

### L2-524ba10eda  (L2, 2026-06-01, sha 524ba10eda1b, PR #24692)
TITLE: feat: SM120 (Blackwell Desktop) support for DeepSeek-V4 inference (#24692)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.kernel.flash_mla_sm120, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/flash_mla_sm120.py (+252/-0); python/sglang/srt/layers/attention/flash_mla_sm120_triton.py (+370/-0); docs_new/src/snippets/autoregressive/deepseek-v4-deployment.jsx (+47/-1); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+41/-18); python/sglang/srt/layers/attention/dsv4/indexer.py (+73/-1); python/sglang/srt/layers/deep_gemm_wrapper/configurer.py (+3/-0); python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py (+454/-0); python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py (+72/-2); python/sglang/srt/server_args.py (+14/-0); test/registered/kernels/test_sm120_flash_mla.py (+465/-0); (+1 more)
LABELS: documentation, deepseek, run-ci, jit-kernel, run-ci-extra
BODY: ## Summary ⏎  ⏎ Adds full **SM120** (RTX PRO 6000 / RTX 5090 / DGX Spark, compute 12.0) support for DeepSeek-V4/V3 on SGLang. SM120 desktop Blackwell GPUs lack TMEM, tcgen05, and DeepGEMM support — this PR provides Triton-based fallback kernels for all critical paths and enables CUDA graph capture. ⏎  ⏎ ### Key changes ⏎  ⏎ **New kernels (7 files):** ⏎ - `mxfp4_moe_sm120_triton.py` — Triton fused MXFP4 dequant + GEMM for MoE experts (4.1x vs PyTorch per-GEMM) ⏎  …[truncated]

### L2-4151a04d1a  (L2, 2026-06-01, sha 4151a04d1aad, PR #26424)
TITLE: [Perf][Spec Decoding] Skip cat/topk/sort/gather in draft_forward for topk=1 (#26424)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/speculative/eagle_utils.py (+10/-1); python/sglang/srt/speculative/eagle_worker_v2.py (+55/-25); python/sglang/srt/speculative/standalone_worker_v2.py (+5/-0); test/registered/unit/spec/test_eagle_worker_v2_topk1_fastpath.py (+94/-0)
LABELS: high priority, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: When speculative_eagle_topk == 1 the draft tree is a chain, so the post-loop machinery in draft_forward — concat(score_list).flatten → torch.topk(.., num_draft-1) → torch.sort → torch.gather over ss_token_list — is structurally an identity: ⏎  ⏎   parent_list      = [-1, 0, 1, ..., num_draft-2]   (chain parents) ⏎   top_scores_index = [ 0, 1, 2, ..., num_draft-2]   (sorted picks) ⏎   draft_tokens     = cat(token_list, dim=1) ⏎  ⏎ Pre-allocate parent_li …[truncated]

### L2-d8a5a25c36  (L2, 2026-06-01, sha d8a5a25c36e8, PR #25173)
TITLE: Refactor NIXL hicache. Add O_DIRECT support (#25173)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/docs/references/environment_variables.mdx (+10/-0); python/sglang/srt/environ.py (+5/-0); python/sglang/srt/managers/cache_controller.py (+7/-10); python/sglang/srt/mem_cache/hicache_storage.py (+3/-0); python/sglang/srt/mem_cache/memory_pool_host.py (+73/-9); python/sglang/srt/mem_cache/mmap_allocator.py (+127/-0); python/sglang/srt/mem_cache/storage/nixl/README.md (+43/-4); python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py (+331/-402); python/sglang/srt/mem_cache/storage/nixl/nixl_registry.py (+144/-0); python/sglang/srt/mem_cache/storage/nixl/nixl_utils.py (+41/-95); (+2 more)
LABELS: documentation, hicache, run-ci
BODY: ## Motivation ⏎  ⏎ This is a rewrite of the HiCache NIXL connector. It fixes a number of bugs (thread synchronization, object store). It also adds an option of using O_DIRECT for file handles, which can improve I/O perfomance significantly (see below).  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ - Improvement: Register host memory on initialization, not on each transfer ⏎ - Improvement: Register destination descriptors on transfers uses a contextmanager to ensure cor …[truncated]

### L2-931765e23e  (L2, 2026-06-01, sha 931765e23e9b, PR #26607)
TITLE: Do not cap DeepSeek V4 PD prefill by SWA pool size (#26607)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/prefill.py (+3/-8)
LABELS: deepseek, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ DS v4 PD prefill is currently capped by SWA pool size because `DeepSeekV4TokenToKVPool` inherits from generic SWA ⏎  ⏎ Following #24857 which fixes a similar problem for decode by preallocating SWA KV only for sliding-window tail, PD prefill still uses the generic SWA pool size to admit ⏎  ⏎ Before change, hybrid-swa model was treated as if the swa KV pool had to hold the full prompt during disagg prefill: ⏎  ⏎ ```python ⏎ if self.sched …[truncated]

### L2-99da43b900  (L2, 2026-06-02, sha 99da43b900d0, PR #26735)
TITLE: [refactor] init_forward_metadata 3-method ABC + side-channel removal + ForwardMetadata type rename (#26735)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.flashmla, L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.aiter_mla, L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/cutlass_mla_backend.py (+32/-73); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+97/-93); python/sglang/srt/layers/attention/flashmla_backend.py (+48/-80); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+62/-78); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+54/-56); python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py (+15/-47); python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py (+7/-3); python/sglang/srt/layers/attention/aiter_backend.py (+38/-47); python/sglang/srt/layers/attention/base_attn_backend.py (+66/-34); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+179/-195); (+32 more)
LABELS: deepseek, speculative-decoding, blackwell, npu, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Following #26665 (rounds 2-6 capture/replay body unification), this PR finishes the attention-init refactor by shipping the **3-method ABC contract**, removing the **`_replay_forward_batch` side channel** that DSV4 had been relying on, and renaming top-level metadata types to align with the `self.forward_metadata` state field they live in. ⏎  ⏎ The split lets out-of-tree backends consume the unified bodies (already on `main`) witho …[truncated]

### L2-76c9899da7  (L2, 2026-06-02, sha 76c9899da7f9, PR #26623)
TITLE: Fix hybrid linear attention misrouting plain-RadixAttention linear layers to the full backend (Ring-2.5-1T) (#26623)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+8/-13); python/sglang/srt/models/bailing_moe_linear.py (+0/-6); test/registered/8-gpu-models/test_ling_2_6_flash.py (+2/-2)
LABELS: run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 27116 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Problem ⏎  ⏎ Ring-2.5-1T (`inclusionAI/Ring-2.5-1T`) crashes during CUDA-graph capture (nightly TP8): ⏎  ⏎ ``` ⏎ ValueError: layer_id=0 not in full attention layers: dict_keys([7, 15, 23, 31, 39, 47, 55, 63, 71, 79]) ⏎ ``` ⏎  ⏎ A **linear**-attention layer (`BailingMoELinearAttention`, layer 0) is routed to the **full** MLA attention backend, which then calls `set_kv_buffer`/`set_mla_kv_buffer` on the `HybridLinearKVPool` with a non-full layer id and raises. ⏎  …[truncated]

### L2-559581b383  (L2, 2026-06-02, sha 559581b383d3, PR #26000)
TITLE: [codex] Centralize Triton utility kernels (#26000)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.pool.mla_token_kv, L2.kernel.set_mla_kv_buffer
FILES: python/sglang/srt/mem_cache/triton_ops/mla_buffer.py (+377/-0); python/sglang/srt/layers/attention/flashattention_backend.py (+4/-411); python/sglang/srt/layers/attention/triton_backend.py (+4/-54); python/sglang/srt/layers/attention/triton_ops/cache_ops.py (+266/-0); python/sglang/srt/layers/attention/triton_ops/kv_indices.py (+103/-0); python/sglang/srt/layers/attention/triton_ops/metadata.py (+467/-0); python/sglang/srt/layers/attention/triton_ops/pad.py (+162/-0); python/sglang/srt/layers/attention/triton_ops/rope_cache.py (+736/-0); python/sglang/srt/layers/attention/utils.py (+37/-1260); python/sglang/srt/layers/attention/wave_backend.py (+4/-54); (+19 more)
LABELS: run-ci, run-ci-extra
BODY: ## Summary ⏎ - move FlashInfer/FlashMLA KV index, KV split, FlashAttention metadata, cache, padding, RoPE/cache, and position Triton kernels into focused `attention/triton_ops/*` and `model_executor/triton_ops/*` modules ⏎ - move logits/elementwise softcap Triton kernels into `srt/layers/triton_ops/softcap.py` while keeping in-place and out-of-place wrapper semantics separate ⏎ - move speculative decoding cache-location, EAGLE, and multi-layer EAGLE Tr …[truncated]

### L2-ab7c4ab6bb  (L2, 2026-06-02, sha ab7c4ab6bb83, PR #25556)
TITLE: [AMD] Fix correctness for AITER MLA backend with `--page-size > 1` (#25556)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+2/-2)
LABELS: run-ci
BODY: # Fix correctness for AITER MLA backend with `--page-size > 1` ⏎  ⏎ ## Summary ⏎  ⏎ With the AITER attention backend on DeepSeek-R1 + ROCm, `--page-size > 1` ⏎ boots without error but produces broken output: `gsm8k` accuracy ⏎ collapses from **0.975 → 0.005**. ⏎  ⏎ The root cause is small: SGLang's `PagedTokenToKVPoolAllocator` already ⏎ flattens page ids into **per-token absolute slot ids** before anything ⏎ downstream sees them (`out_pages[:, None] * pag …[truncated]

### L2-8e77af1afc  (L2, 2026-06-03, sha 8e77af1afcee, PR #24762)
TITLE: [AMD] fix(triton-mla): cap max_kv_splits at 256 on gfx942 (Kimi-K2.6 hang) (#24762)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/triton_backend.py (+11/-0); .github/workflows/pr-test-amd-rocm720.yml (+2/-2); .github/workflows/pr-test-amd.yml (+1/-1); python/sglang/srt/utils/common.py (+12/-0); test/registered/amd/test_kimi_k2_instruct.py (+1/-1)
LABELS: amd, run-ci
BODY: ## Motivation ⏎  ⏎ Fix the `nightly-8-gpu-kimi-k26` MI325X hang reproduced in https://github.com/sgl-project/sglang/actions/runs/25513282022/job/74877480809 (scheduler watchdog at 300 s, MLA decode kernel page-faulting with `Memory access fault by GPU node ... Reason: Write access to a read-only page`). ⏎  ⏎ Bisect window `d86f2916..7d397ad2` → #20479 / `3fe8bc987`. Its `_mla_decode_kv_splits_cap()` lifts `max_kv_splits` to `next_power_of_2(sm_count) …[truncated]

### L2-93173b27e8  (L2, 2026-06-03, sha 93173b27e8a6, PR #25418)
TITLE: integrate flash_mla_sparse_fwd (#25418)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/deepseek_v4_backend.py (+135/-2); python/sglang/srt/layers/attention/dsv4/dequant_k_cache.py (+226/-0); python/sglang/srt/layers/attention/dsv4/indexer.py (+2/-0); python/sglang/srt/layers/attention/dsv4/metadata.py (+6/-1); python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py (+587/-0); python/sglang/srt/models/deepseek_v4.py (+1/-1); python/sglang/srt/models/deepseek_v4_nextn.py (+1/-1); python/sglang/srt/environ.py (+1/-0)
LABELS: high priority, quant, deepseek, run-ci, run-ci-extra
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎ - `flash_mla_with_kvcache` is slow because of complicated loading logic. Switching to `flash_mla_sparse_fwd` see 1.35x speedup compared to the `flash_mla_with_kvcache` kernel ⏎    - total cuda wall 1.1x speedup ⏎ - Chunk prefill only works for <=8192, fails for 32768. (https://github.com/sgl-project/sglang/pull/25502 has more details on this motivation) ⏎  ⏎  ⏎ ## Modifications ⏎ - Use `flash_mla_sparse_fwd` for prefill instead of `flash …[truncated]

### L2-0ef39784ef  (L2, 2026-06-03, sha 0ef39784ef78, PR #26911)
TITLE: [Bugfix] Gate DP-attention even-token padding to CP-enabled configs (#26911)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/dsa/utils.py (+5/-3); python/sglang/srt/layers/utils/cp_utils.py (+15/-0); python/sglang/srt/model_executor/forward_batch_info.py (+10/-5)
LABELS: documentation, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Fix a NaN crash (`Assertion 'NaN detected! draft_extend_for_prefill' failed`) in EAGLE/MTP speculative decoding with DP attention, introduced by #23269. It surfaces as a server crash on the `nightly-8-gpu-b200` test `TestEagleDPAttnServerLarge.test_a_gsm8k`. ⏎  ⏎ ## Modifications ⏎  ⏎ In `ForwardBatch.prepare_mlp_sync_batch`, only pad `global_num_tokens` to a multiple of `attn_cp_size * 2` when context parallel is enabled (`attn_cp_size >  …[truncated]

### L2-d7013b6537  (L2, 2026-06-03, sha d7013b6537b2, PR #27001)
TITLE: [AMD] [CI] Remove hardcoded model/cache paths from MI35x nightly tests (#27001)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py (+0/-5); test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py (+1/-35); test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py (+1/-35); test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py (+1/-35); test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py (+0/-6); test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py (+0/-5); test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py (+0/-6); test/registered/amd/accuracy/mi35x/test_glm47_fp8_eval_mi35x.py (+0/-6); test/registered/amd/accuracy/mi35x/test_glm51_eval_mi35x.py (+0/-4); test/registered/amd/accuracy/mi35x/test_glm5_eval_mi35x.py (+0/-5); (+17 more)
LABELS: deepseek
BODY: ## Summary ⏎ - Remove the `HF_HOME`/`HF_HUB_CACHE` → `/data2` cache redirect from all MI35x accuracy/perf nightly tests; they now use the default HF cache location instead of a machine-specific mount. ⏎ - Remove every remaining hardcoded local **model** path and collapse the `get_model_path()` resolvers across the MI35x tests: DeepSeek-R1 / GLM-5 MXFP4 (`/data2`), Kimi-K2.5-MXFP4, Qwen3-Coder-Next and Kimi-K2.5 aiter-MLA (`/data`). ⏎ - Since the `*_MOD …[truncated]

### L2-7f706f4cfb  (L2, 2026-06-03, sha 7f706f4cfba0, PR #26854)
TITLE: [Deps] Bump FI to 0.6.12 and cutedsl to 4.5.2 (#26854)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+3/-3); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1)
LABELS: dependencies, run-ci, run-ci-extra
BODY: ## Summary ⏎  ⏎ Bump FI to 0.6.12 and cutedsl to 4.5.2 ⏎  ⏎ ## Test Plan ⏎  ⏎ CI ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #26841042733](https://github.com/sgl-project/sglang/actions/runs/26841042733) ⏎ Latest PR Test (Extra): :x: [Run #26841044109](https://github.com/sgl-project/sglang/actions/runs/26841044109)

### L2-e485ad6ac1  (L2, 2026-06-03, sha e485ad6ac1a4, PR #27120)
TITLE: Fix hybrid linear attention dispatch by layer id with draft-worker awareness (#27120)
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.attention_registry
FILES: python/sglang/srt/layers/attention/attention_registry.py (+5/-1); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+0/-16); python/sglang/srt/models/bailing_moe_linear.py (+0/-6)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `HybridLinearAttnBackend._is_full_attn` dispatched by the layer's runtime type: `RadixLinearAttention` → linear, plain `RadixAttention` → full. This misroutes hybrid models whose linear layers wrap a plain `RadixAttention` (Bailing/Ring, see #26623), which was patched with an `_is_linear_attention` marker attribute — a workaround rather than a fix (see the revert in #27116). ⏎  ⏎ This PR replaces the type/marker-based dispatch with a p …[truncated]

### L2-1dd9432889  (L2, 2026-06-03, sha 1dd9432889b7, PR #26894)
TITLE: [AMD] Fuse compress norm+rope+hadamard into single Triton kernel (#26894)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/dsv4/compressor.py (+19/-8); python/sglang/srt/layers/attention/dsv4/fused_compress_triton.py (+151/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ In the DSV4 compressor decode path on MI355, the post-compression pipeline runs two separate kernels back-to-back on the same `kv_compressed` tensor `[num_tokens, 512]`: ⏎  ⏎ 1. `_compress_norm_rope_kernel` (RMSNorm + RoPE) -- ~4.0 us ⏎ 2. `fast_hadamard_transform_kernel` (Hadamard rotation) -- ~4.1 us ⏎  ⏎ Since both operate on the same data and the full head_dim (512) fits in registers, fusing them eliminates one kernel launch and o …[truncated]

### L2-ac99794e64  (L2, 2026-06-03, sha ac99794e64e0, PR #26997)
TITLE: Reland spec v2 tree drafting (eagle topk>1) with page_size==1 (#26866) (#26997)
SOURCES: release_notes
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.pool.mla_token_kv, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/speculative_hook.py (+13/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+14/-0); python/sglang/srt/layers/attention/triton_backend.py (+8/-2); python/sglang/srt/mem_cache/memory_pool.py (+35/-0); python/sglang/srt/model_executor/forward_batch_info.py (+6/-3); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+6/-0); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+3/-1); python/sglang/srt/speculative/eagle_worker_v2.py (+59/-4); python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py (+2/-1); python/sglang/srt/speculative/triton_ops/eagle.py (+3/-2); (+2 more)
LABELS: high priority, speculative-decoding, run-ci, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study revert record: reland of PR(s) 26866 reason=other
BODY: Relands #26866 (reverted in #26981). ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #26907731073](https://github.com/sgl-project/sglang/actions/runs/26907731073) ⏎ Latest PR Test (Extra): :white_check_mark: [Run #26907730270](https://github.com/sgl-project/sglang/actions/runs/26907730270)

### L2-c9ca56da8c  (L2, 2026-06-03, sha c9ca56da8c5e, PR #27091)
TITLE: Unify full→SWA index translation in init_forward_metadata; drop pool caches (#27091)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.backend.sparse_mla_adapters, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/environ.py (+0/-1); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+78/-4); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+78/-4); python/sglang/srt/layers/attention/dsa_backend.py (+2/-1); python/sglang/srt/layers/attention/dsv4/compressor_v2.py (+4/-1); python/sglang/srt/layers/attention/dsv4/indexer.py (+2/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+8/-7); python/sglang/srt/layers/attention/triton_backend.py (+0/-3); python/sglang/srt/layers/attention/trtllm_mha_backend.py (+6/-3); python/sglang/srt/mem_cache/base_swa_memory_pool.py (+0/-3); (+19 more)
LABELS: deepseek, hicache, blackwell, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ The full→SWA index translation (`translate_loc_from_full_to_swa`) was triggered in two inconsistent styles: ⏎  ⏎ - **Write path (DSV4 KV store):** translated `out_cache_loc` lazily at the *first SWA layer* via `DeepSeekV4TokenToKVPool.get_cached_swa_loc` (env-gated by `SGLANG_OPT_CACHE_SWA_TRANSLATION`), with the result cached across layers. ⏎ - **Read path (attention):** `SWAKVPool` translated on demand and memoized with a `(data_ptr, n …[truncated]

### L2-cfb7fb4fad  (L2, 2026-06-03, sha cfb7fb4fad03, PR #27188)
TITLE: [AMD] Fix TP2 DeepSeek-R1 nhead=64 MLA decode crash and add nightly coverage (#27188)
SOURCES: path_integration+keyword, subject_keyword, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+1/-1); .github/workflows/nightly-test-amd-rocm720.yml (+73/-0); .github/workflows/nightly-test-amd.yml (+73/-0); test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_tp2_mi35x.py (+185/-0); test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_tp4_mi35x.py (+184/-0); test/run_suite.py (+2/-0)
LABELS: amd, deepseek
DEEP_STUDY: deep-study correctness case sglang:cfb7fb4fad: class=shape_alignment_edge; symptom=illegal_memory_access; introducing=unknown
BODY: # Fix TP2 DeepSeek-R1 nhead=64 MLA decode crash and add nightly coverage ⏎  ⏎ ## Summary ⏎  ⏎ Fixes a Memory access fault crash when running DeepSeek-R1-MXFP4 with TP2 and the AITER persistent MLA decode path. ⏎  ⏎ DeepSeek-R1 has 128 attention heads, so TP2 runs with `nhead=64` per rank. AITER added a native qh64 persistent MLA kernel, but SGLang only enabled persistent-mode metadata (`fast_mode=True`, `intra_batch_mode=False`) for `nhead=32` and `nhe …[truncated]

### L2-7aee2ff31b  (L2, 2026-06-04, sha 7aee2ff31b18, PR #26914)
TITLE: [AMD] Remove BF16-to-FP32 elementwise cast from compressor GEMM on HIP (#26914)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/dsv4/compressor.py (+14/-2); python/sglang/srt/layers/attention/dsv4/fused_compress_triton.py (+29/-18)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: Merged on amd/deepseek_v4 (https://github.com/sgl-project/sglang/pull/26272), submit a PR for main as well. ⏎  ⏎ ## Motivation ⏎  ⏎ On MI355/HIP, `Compressor.forward()` calls `linear_bf16_fp32()` which does: ⏎ ```python ⏎ return tgemm.mm(x, y, otype=x.dtype).float() ⏎ ``` ⏎ The `.float()` triggers a separate `bfloat16tofloat32_copy_kernel_cuda` elementwise kernel (~4.2 us each). This fires once per compressor GEMM call: ⏎ - C-type layers (compress_ratio=4 …[truncated]

### L2-5c8a04ac4e  (L2, 2026-06-04, sha 5c8a04ac4ebd, PR #27156)
TITLE: [XPU CI] Expand stage-a and consolidate stage-b tests into stage-a (#27156)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-xpu.yml (+57/-62); test/registered/attention/test_chunk_gated_delta_rule.py (+1/-1); test/registered/lora/test_lora_eviction_policy.py (+2/-0); test/registered/unit/sampling/test_sampling_params.py (+2/-1); test/registered/unit/spec/test_adaptive_spec_params.py (+2/-1); test/registered/xpu/test_deepseek_ocr.py (+1/-1); test/registered/xpu/test_deepseek_ocr_triton.py (+1/-1); test/registered/xpu/test_intel_xpu_backend.py (+1/-1); test/registered/xpu/test_topk.py (+1/-1)
LABELS: lora, deepseek, intel, ci, xpu, run-ci, run-ci-extra
BODY: ## Summary ⏎  ⏎ - Add \`register_xpu_ci(suite="stage-a-test-1-gpu-xpu")\` to 3 new test files: ⏎   - \`test_sampling_params.py\` (66 tests, Sampling) ⏎   - \`test_lora_eviction_policy.py\` (12 tests, LoRA) ⏎   - \`test_adaptive_spec_params.py\` (11 tests, Speculative Decoding) ⏎ - Move all existing \`stage-b-test-1-gpu-xpu\` tests to \`stage-a-test-1-gpu-xpu\`: ⏎   - \`test_chunk_gated_delta_rule.py\` (Attention / Delta Rule) ⏎   - \`test_intel_xpu_backend.py\`  …[truncated]

### L2-0aa72a9e76  (L2, 2026-06-04, sha 0aa72a9e7677, PR #27193)
TITLE: Replace skip_attn_backend_init with a batch-carried attention plan marker (+ staleness re-plan) (#27193)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+8/-6); python/sglang/srt/batch_overlap/two_batch_overlap.py (+6/-0); python/sglang/srt/hardware_backend/mlx/tp_worker.py (+1/-1); python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py (+1/-2); python/sglang/srt/managers/tp_worker.py (+4/-6); python/sglang/srt/model_executor/cpu_graph_runner.py (+0/-1); python/sglang/srt/model_executor/cuda_graph_runner.py (+2/-2); python/sglang/srt/model_executor/forward_batch_info.py (+93/-0); python/sglang/srt/model_executor/model_runner.py (+6/-13); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+3/-0); (+10 more)
LABELS: speculative-decoding, blackwell, npu, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ `skip_attn_backend_init` is a control-coupling boolean threaded through ~6 layers (`TpModelWorker.forward_batch_generation` → `ModelRunner.forward` → `_forward_raw` → `forward_decode` / `forward_extend` / graph-runner `replay`, plus the NPU/CPU/MLX mirrors). It means *"the attention metadata for this batch was already planned elsewhere — don't plan again"*, but the protocol has three structural problems: ⏎  ⏎ 1. **Unverifiable caller a …[truncated]

### L2-7dc7376697  (L2, 2026-06-04, sha 7dc7376697b0, PR #27316)
TITLE: fix(attn): delegate init_mha_chunk_metadata in HybridLinearAttnBackend (#27316)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+10/-0); test/registered/attention/unittests/hybrid_linear/README.md (+64/-0); test/registered/attention/unittests/hybrid_linear/__init__.py (+0/-0); test/registered/attention/unittests/hybrid_linear/test_flashinfer_mla_chunk_metadata.py (+145/-0)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ Serving a hybrid linear-attention MLA model (e.g. `inclusionAI/Ring-2.5-1T`) with the FlashInfer MLA backend crashes during prefill: ⏎  ⏎ ``` ⏎ File ".../flashinfer/prefill.py", line 3318, in run ⏎ ValueError: q.shape[0] (8218) does not match qo_indptr[-1] (800). ⏎ For ragged prefill, q must have shape [total_tokens, num_heads, head_dim] where total_tokens = qo_indptr[-1]. ⏎ ``` ⏎  ⏎ Hybrid linear-attention models (Ring/Ling = `bailing_moe_linear` …[truncated]

### L2-5af02c18ae  (L2, 2026-06-04, sha 5af02c18ae70, PR #25002)
TITLE: [spec_v2] Enable trtllm_mha draft-extend CUDA graph with v2 semantics (#25002)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/trtllm_mha_backend.py (+30/-10); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+3/-2); python/sglang/srt/speculative/eagle_worker_v2.py (+11/-4); python/sglang/test/kits/attention_unittest/runner_modes/speculative_draft_extend_runner.py (+24/-0)
LABELS: speculative-decoding, blackwell, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ `DRAFT_EXTEND_V2` currently cannot use the draft-extend CUDA graph path with `trtllm_mha` because the v2 Eagle draft worker whitelist does not include `TRTLLMHAAttnBackend`. ⏎  ⏎ The previous V2 graph replay computed full-logits `softmax + topk/reduce` work, while `EagleDraftWorker` later recomputed `softmax + fast_topk` on the selected logits it actually consumes. The graph-produced `ret.topk_p` and `ret.topk_index` are therefore  …[truncated]

### L2-bd47869ba4  (L2, 2026-06-04, sha bd47869ba412, PR #27320)
TITLE: [perf] parallelize create_flashmla_kv_indices over page-blocks (#27320)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.flashmla, L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.backend.trtllm_mla, L2.backend.aiter_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+4/-1); python/sglang/srt/layers/attention/cutlass_mla_backend.py (+15/-3); python/sglang/srt/layers/attention/flashmla_backend.py (+18/-4); python/sglang/srt/layers/attention/triton_ops/kv_indices.py (+13/-1); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+15/-2); python/sglang/srt/layers/attention/utils.py (+3/-0)
LABELS: high priority, blackwell, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: Launch one CTA per page-block (grid axis 1) instead of a single CTA looping over every block, so building the kv-index page table scales with context length instead of serializing it (a large win at long context, where bs=1 left one CTA to walk the whole table). ⏎  ⏎ Add get_num_kv_index_blocks_flashmla to size grid axis 1 and update all call sites. Output is unchanged. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## …[truncated]

### L2-c9f582a272  (L2, 2026-06-05, sha c9f582a272dc, PR #27329)
TITLE: [LoRA] Experimental fast LoRA path with `experimental_sgl_trtllm` MoE backend for FP8 and NVFP4 models (#27329)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption
FILES: python/sglang/jit_kernel/csrc/trtllm_lora_temp/kimi_k2_moe_fused_gate.cuh (+453/-0); python/sglang/jit_kernel/csrc/trtllm_lora_temp/moe_lora_merged_align_kernel.cu (+589/-0); python/sglang/jit_kernel/csrc/trtllm_lora_temp/topk_softmax_pack.cuh (+412/-0); python/sglang/jit_kernel/trtllm_lora_temp/SOURCE.md (+20/-0); python/sglang/jit_kernel/trtllm_lora_temp/__init__.py (+17/-0); python/sglang/jit_kernel/trtllm_lora_temp/core.py (+443/-0); python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/fused_activation_quant.cuh (+230/-0); python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu (+1137/-0); python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/fused_permute_quant.cuh (+307/-0); python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/trtllm_fused_moe_kernel_launcher.cu (+3920/-0); (+42 more)
LABELS: documentation, high priority, quant, lora, deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: > **⚠️ Experimental — for early adopters.** This is an opt-in fast LoRA path gated behind a single master switch `SGLANG_EXPERIMENTAL_LORA_OPTI` (default **off** ⇒ upstream behavior is byte-identical) and selected with `--moe-runner-backend experimental_sgl_trtllm`. The new logic is intentionally isolated in `*_temp` packages and is **actively being refactored** toward an upstream-clean form — env flags, the backend name, and file layout may stil …[truncated]

### L2-00fefef16b  (L2, 2026-06-05, sha 00fefef16b02, PR #24880)
TITLE: [PD & HiSparse] Add DeepSeek V4 support for HiSparse direct Prefill-to-Decode DRAM (#24880)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/csrc/deepseek_v4/hisparse_transfer.cuh (+0/-82); python/sglang/jit_kernel/csrc/hisparse.cuh (+61/-4); python/sglang/jit_kernel/dsv4/__init__.py (+0/-2); python/sglang/jit_kernel/dsv4/hisparse.py (+0/-27); python/sglang/jit_kernel/hisparse.py (+34/-1); python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/kvcacheio.cuh (+14/-34); python/sglang/jit_kernel/tests/test_hisparse.py (+128/-1); python/sglang/srt/disaggregation/decode.py (+21/-2); python/sglang/srt/managers/hisparse_coordinator.py (+27/-17); python/sglang/srt/mem_cache/hisparse_memory_pool.py (+37/-129); (+2 more)
LABELS: high priority, deepseek, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ gsm8k 200 examples ⏎ ``` ⏎ 100%|██████████| 200/200 [11:33<00:00,  3.47s/it] ⏎ Total latency: 693.701 s ⏎ Score: 0.975 ⏎ Output throughput: 27.094 token/s ⏎ ``` ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull- …[truncated]

### L2-2e7523ddbe  (L2, 2026-06-05, sha 2e7523ddbefd, PR #26726)
TITLE: fix(spec-dec): treat `num_nextn_predict_layers=0` the same as absent for EAGLE3 drafts (#26726)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/model_runner.py (+7/-1)
LABELS: run-ci
BODY: ## Issue ⏎  ⏎ When using `nvidia/Kimi-K2.5-Thinking-Eagle3` as the EAGLE3 draft for `nvidia/Kimi-K2.5-NVFP4`, sglang crashes during the first draft forward pass with `IndexError: list index out of range` inside `MLATokenToKVPool.set_mla_kv_buffer`. ⏎  ⏎ ### Reproducer (8×B200) ⏎  ⏎ ```bash ⏎ python3 -m sglang.launch_server \ ⏎     --model-path nvidia/Kimi-K2.5-NVFP4 \ ⏎     --served-model-name nvidia/Kimi-K2.5-NVFP4 \ ⏎     --trust-remote-code \ ⏎     --tp  …[truncated]

### L2-8c47b7678a  (L2, 2026-06-05, sha 8c47b7678a8b, PR #27403)
TITLE: [attn backend] clean legacy init_mha_chunk_metadata in trtllm_mla backend (#27403)
SOURCES: path_core, path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+3/-4)
LABELS: high priority, blackwell, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L2-84ca0ffb8c  (L2, 2026-06-06, sha 84ca0ffb8c29, PR #26972)
TITLE: Spec v2 tree drafting (topk>1) with page_size>1 (#26972)
SOURCES: release_notes
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/speculative_hook.py (+14/-7); python/sglang/srt/debug_utils/pr_fix_toggle.py (+21/-0); python/sglang/srt/managers/scheduler.py (+4/-2); python/sglang/srt/managers/utils.py (+8/-3); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+15/-0); python/sglang/srt/speculative/eagle_info_v2.py (+121/-16); test/registered/spec/eagle/test_spec_eagle_page.py (+4/-11); test/registered/spec/eagle/test_spec_eagle_topk_page.py (+41/-0)
LABELS: high priority, speculative-decoding, run-ci, bypass-fastfail, run-ci-extra
BODY: Stacked on #26866. ⏎  ⏎ Extends spec v2 (overlap) EAGLE tree drafting (`topk > 1`) to `page_size > 1`, and sizes + guards the `req_to_token` pool for the per-branch holey draft layout. ⏎  ⏎ ## Summary ⏎ - Lay out each draft branch's region page-aligned ("holey") and duplicate the prefix partial-tail page into every branch so whole-page attention reads stay coherent (`duplicate_prefix_tail_to_draft_branches`). ⏎ - Implement `get_alloc_len_per_decode` for `pag …[truncated]

### L2-393d0e169e  (L2, 2026-06-06, sha 393d0e169e8e, PR #27114)
TITLE: [Bugfix] Restore overridden HF config fields and support index_skip_topk_offset for DSA topk sharing (#27114)
SOURCES: path_core
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+15/-2); python/sglang/srt/batch_overlap/two_batch_overlap.py (+1/-0); python/sglang/srt/model_executor/forward_batch_info.py (+4/-0); python/sglang/srt/models/deepseek_nextn.py (+7/-0); python/sglang/srt/models/deepseek_v2.py (+25/-3); python/sglang/srt/server_args.py (+14/-0); python/sglang/srt/speculative/eagle_worker.py (+14/-0); python/sglang/srt/utils/hf_transformers/config.py (+20/-0)
LABELS: deepseek, run-ci, run-ci-extra
BODY: Restore HF config fields clobbered by GlmMoeDsaConfig and add index_skip_topk_offset handling for DSA topk sharing ⏎  ⏎ And in the PR description, reference the upstream transformers fix: ⏎  ⏎ The GlmMoeDsaConfig drops/clobbers raw checkpoint fields the DSA path needs (qk_rope_head_dim, index_topk_freq), so we re-read them from config.json and restore. Fixed upstream by [this pr](https://github.com/huggingface/transformers/pull/46338) — this workarou …[truncated]

### L2-5160f7914e  (L2, 2026-06-06, sha 5160f7914ebf, PR #27460)
TITLE: Fix MLA EAGLE draft CUDA-graph `kv_indices` under-allocation for `topk > 1` (#27460)
SOURCES: path_core, path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+16/-1); python/sglang/srt/debug_utils/pr_fix_toggle.py (+12/-0)
BODY: ## Summary ⏎ - Widen the multi-step MLA draft CUDA-graph `cuda_graph_kv_indices` buffer by the missing `topk` factor (`max_bs * max_context_len` -> `max_bs * topk * max_context_len`), matching the eager `init_forward_metadata` sizing. ⏎ - Add an always-on size invariant in `common_template` so an undersized row fails fast instead of silently corrupting memory. ⏎  ⏎ This mirrors the non-MLA fix in #27338 for `FlashInferMLAMultiStepDraftBackend`. ⏎  ⏎ ## Bug ⏎ - …[truncated]

### L2-4c8a022f38  (L2, 2026-06-06, sha 4c8a022f38e3, PR #27191)
TITLE: Fix DeepSeek V4 DP reduce scatter when use attention DP + MoE TP (#27191)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v4.py (+10/-2)
LABELS: deepseek, run-ci
BODY: ## Summary ⏎  ⏎ - Use `reduce_scatterv` for DeepSeek V4's TP-MoE gather return path when DP reduce-scatter is enabled. ⏎ - Keep the existing `dp_scatter` path for configurations that do not use DP reduce-scatter. ⏎  ⏎ ## Root Cause ⏎  ⏎ DeepSeek V4 has a hand-written `_use_tp_moe_gather` path. In DP attention with TP MoE, the MLP output can be TP-partial. When `should_use_dp_reduce_scatterv()` is true, scattering that partial tensor directly leaves the local h …[truncated]

### L2-5e2e0d5b49  (L2, 2026-06-07, sha 5e2e0d5b4999, PR #27484)
TITLE: [spec] Make `spec_utils` module-importable: type-only imports under TYPE_CHECKING (#27484)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.aiter_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+1/-2); python/sglang/srt/layers/attention/aiter_backend.py (+1/-2); python/sglang/srt/layers/attention/flashinfer_backend.py (+1/-2); python/sglang/srt/speculative/spec_utils.py (+4/-3)
BODY: ## Summary ⏎ - Move `spec_utils`'s type-only imports (`Req`, `ServerArgs`, `BaseGrammarObject`) under `TYPE_CHECKING` so importing `spec_utils` no longer pulls `schedule_batch` / `server_args` / `constrained` at module load. ⏎ - The 3 attention backends (`flashinfer`, `flashinfer_mla`, `aiter`) now import `generate_draft_decode_kv_indices` from `spec_utils` at module level, dropping the `__init__`-local-import workaround that existed only to dodge a  …[truncated]

### L2-eab2e02fa0  (L2, 2026-06-07, sha eab2e02fa0c4, PR #27475)
TITLE: [spec] Dedup draft `kv_indices` sizing into `spec_utils` helpers (#27475)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.aiter_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+16/-9); python/sglang/srt/layers/attention/aiter_backend.py (+14/-7); python/sglang/srt/layers/attention/flashinfer_backend.py (+16/-9); python/sglang/srt/layers/attention/triton_backend.py (+14/-7); python/sglang/srt/speculative/spec_utils.py (+22/-0)
BODY: ## Summary ⏎  ⏎ Extract the two `kv_indices` sizing formulas that the EAGLE draft-decode backends duplicate inline into shared helpers in `python/sglang/srt/speculative/spec_utils.py`: ⏎  ⏎ - `draft_kv_indices_buffer_width(num_seqs, topk, max_context_len)` = `num_seqs * topk * max_context_len` (per-step row width of the draft `kv_indices` buffer). ⏎ - `draft_kv_indices_used_len(seq_lens_sum, topk, bs, num_steps)` = `seq_lens_sum * topk + bs * num_steps` (l …[truncated]

### L2-10d33bd77e  (L2, 2026-06-07, sha 10d33bd77ea8, PR #22299)
TITLE: [AMD] Enable Piecewise CUDA Graph for AMD GPUs (#22299)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/compilation/cuda_piecewise_backend.py (+21/-4); python/sglang/srt/compilation/piecewise_context_manager.py (+6/-0); python/sglang/srt/layers/moe/hash_topk.py (+11/-2); python/sglang/srt/layers/moe/topk.py (+22/-0); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py (+134/-5); python/sglang/srt/layers/radix_attention.py (+29/-0); python/sglang/srt/model_executor/model_runner.py (+37/-6); python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py (+66/-12); test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py (+7/-2); test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py (+2/-1)
LABELS: amd, deepseek, run-ci, piecewise-cuda-graph, bypass-fastfail
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Enable Piecewise CUDA Graph (PCG) on AMD/ROCm with the aiter backend, while keeping the existing CUDA path unchanged. The PR also fixes a Dynamo recompilation fallback crash when batches exceed `piecewise_cuda_graph_max_tokens`. ⏎  ⏎ ### Tested configurations ⏎  ⏎ - **DeepSeek-R1-MXFP4-Preview** — TP=4, EP=4, EAGLE speculative decoding, FP8 KV cache, aiter backend ⏎ - **DeepSeek-R1-MXFP4-Preview** — TP=8, registered AMD PCG test ⏎ - ** …[truncated]

### L2-f68c79675f  (L2, 2026-06-07, sha f68c79675f16, PR #27463)
TITLE: Support `topk > 1` tree drafting for mamba/hybrid-linear models on spec v2 (#27463)
SOURCES: release_notes
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/speculative_hook.py (+0/-20); python/sglang/srt/layers/attention/linear/lightning_backend.py (+10/-0); python/sglang/srt/mem_cache/memory_pool.py (+2/-0); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+3/-0); python/sglang/srt/speculative/eagle_worker_v2.py (+8/-13); python/sglang/test/kits/attention_unittest/attention_methods/mamba2_attention.py (+9/-13); python/sglang/test/kits/attention_unittest/runner_modes/speculative_target_verify_runner.py (+10/-11); test/registered/models_e2e/test_qwen3_next_models_mtp.py (+3/-0)
LABELS: speculative-decoding, run-ci, run-ci-extra
BODY: ## Summary ⏎ - Mamba / hybrid-linear models (GDN — Qwen3-Next; Mamba2 — Nemotron-H) with `--speculative-eagle-topk > 1` previously fell back to the spec v1 worker. This runs them on spec v2, reusing the tree-drafting machinery from #26972. ⏎  ⏎ ## Changes ⏎ **Spec v2 mamba tree verify** ⏎ - `eagle_worker_v2.py`: compute the tree-aware `last_correct_step_indices` by gathering each request's last accepted node from `accept_index`, replacing the chain-only `a …[truncated]

### L2-303757ccd8  (L2, 2026-06-07, sha 303757ccd8f9, PR #27485)
TITLE: [Attn] Fix aiter MLA verify `kv_indices` under-alloc + shared `assert_buffer_fits` guard (#27485)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.fa3_fa4_mla, L2.backend.aiter_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+20/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+20/-8); python/sglang/srt/layers/attention/flashinfer_backend.py (+10/-9); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+7/-8); python/sglang/srt/layers/attention/utils.py (+13/-0)
LABELS: run-ci, bypass-fastfail, run-ci-extra
BODY: ## Fix (behavior change) ⏎  ⏎ aiter MLA `target_verify` writes `kv_lens = seq_len + num_draft_tokens` per row into the cuda-graph `cuda_graph_kv_indices` buffer, but the buffer was sized only for `seq_len` (no draft slack, unlike `dsa` / `flashmla` / `trtllm_mha`). `create_flashinfer_kv_indices_triton` bounds writes only per-row (`req_to_token` stride), not against the buffer, so a near-full sequence silently overflows into adjacent memory. Reserve t …[truncated]

### L2-61e4132bc2  (L2, 2026-06-08, sha 61e4132bc27f, PR #27537)
TITLE: [MUSA] bump torchada version to 0.1.59 and workaround PCG limitation. (#27537)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: 3rdparty/amd/wheel/sglang/pyproject.toml (+1/-1); python/pyproject_other.toml (+1/-1); sgl-kernel/pyproject_musa.toml (+1/-1)
LABELS: dependencies, sgl-kernel, mthreads
BODY: ## Motivation ⏎  ⏎ Upgrade `torchada` to `0.1.59` to pick up the MUSA CUDA Graph executable rotation fix for the long-standing PCG issue. ⏎  ⏎ SGLang’s piecewise CUDA Graph path can instantiate a large number of MUSA graph executables for deep models. On MUSA, the driver has a per-process limit of roughly 2048 live `musaGraphExec_t` objects, which can cause graph capture failures when PCG is enabled. ⏎  ⏎ `torchada 0.1.59` includes a transparent LRU ro …[truncated]

### L2-b047bb3e92  (L2, 2026-06-08, sha b047bb3e9271, PR #27387)
TITLE: build(sgl-kernel): support configurable mirrors for restricted networks (#27387)
SOURCES: path_core, dependency_pin, body_keyword
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: sgl-kernel/CMakeLists.txt (+27/-18); sgl-kernel/cmake/flashmla.cmake (+11/-3); sgl-kernel/Dockerfile (+25/-5); sgl-kernel/Makefile (+7/-0); sgl-kernel/build.sh (+12/-1)
LABELS: sgl-kernel, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Make `sgl-kernel` Docker builds work in restricted-network or mirrored ⏎ environments (internal artifactory, region-blocked GitHub, etc.) by making ⏎ every external source a configurable mirror. Public defaults are preserved — ⏎ out-of-the-box behavior is unchanged. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Some opt-in env vars, each falling back to its public upstream when unset. ⏎ All threaded through `build.sh` → `docker buildx --build-arg`  …[truncated]

### L2-ea1d190ed0  (L2, 2026-06-08, sha ea1d190ed026, PR #27289)
TITLE: [ROCm] dsv4: remove the redundant fp8 scale transpose-copy on decode (#27289)
SOURCES: path_core
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+3/-0); python/sglang/srt/layers/communicator.py (+3/-0); python/sglang/srt/layers/quantization/fp8_utils.py (+4/-2); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py (+4/-0); python/sglang/srt/models/deepseek_common/utils.py (+2/-0); python/sglang/srt/models/deepseek_v2.py (+2/-1); python/sglang/srt/models/deepseek_v4.py (+2/-0)
LABELS: amd, deepseek, run-ci
BODY: ## TL;DR ⏎  ⏎ On ROCm MI355X DSv4-Pro decode, every fp8 quant was followed by a small copy whose only job was transposing the per-128 block-scale tensor into the layout the aiter bpreshuffle GEMM wants — 183 copies per decode step, ~3% of decode GPU. This PR has the quant producers emit the scale in the layout the *active* GEMM backend expects, so the copy is gone. Median TPOT improves ~2% across concurrencies; the data entering the GEMM is bit-ide …[truncated]

### L2-28c1a3cb45  (L2, 2026-06-08, sha 28c1a3cb450f, PR #25464)
TITLE: [Spec] Deprecate Spec V1 (#25464)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/speculative_hook.py (+5/-5); python/sglang/srt/debug_utils/pr_fix_toggle.py (+0/-19); python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_extend_npu_graph_runner.py (+2/-2); python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py (+2/-2); python/sglang/srt/managers/overlap_utils.py (+3/-2); python/sglang/srt/managers/schedule_batch.py (+3/-4); python/sglang/srt/managers/scheduler.py (+37/-14); python/sglang/srt/managers/scheduler_components/weight_updater.py (+1/-1); python/sglang/srt/mem_cache/kv_cache_builder.py (+3/-4); python/sglang/srt/speculative/adaptive_spec_params.py (+1/-1); (+11 more)
LABELS: high priority, speculative-decoding, npu, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Summary ⏎ - Remove the V1 monolithic speculative workers: `eagle_worker.py`, `multi_layer_eagle_worker.py`, `standalone_worker.py` (~2.3k lines deleted) ⏎ - EAGLE / EAGLE3 / STANDALONE / MULTI_LAYER_EAGLE now always run the V2 worker — overlap scheduling drives it asynchronously; the non-overlap (`--disable-overlap-schedule`) path drives the same V2 worker synchronously ⏎ - NGRAM and FROZEN_KV_MTP are unchanged ⏎ - Drop the v1/v2 branch in `Speculativ …[truncated]

### L2-ca66e6fb5e  (L2, 2026-06-08, sha ca66e6fb5e5d, PR #25195)
TITLE: [BCG] Support breakable CUDA graph for DeepSeek V4 DP attention (#25195)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/base_attn_backend.py (+30/-0); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+251/-26); python/sglang/srt/layers/attention/dsv4/compressor.py (+8/-3); python/sglang/srt/layers/attention/dsv4/indexer.py (+39/-15); python/sglang/srt/model_executor/forward_batch_info.py (+2/-0); python/sglang/srt/models/deepseek_v4.py (+86/-10); python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh (+3/-1); python/sglang/srt/batch_overlap/two_batch_overlap.py (+9/-0); python/sglang/srt/managers/schedule_batch.py (+2/-0); python/sglang/srt/managers/scheduler_components/dp_attn.py (+13/-2); (+3 more)
LABELS: deepseek, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ DeepSeek V4 with DP attention currently cannot use breakable piecewise CUDA graph for mixed/extend batches. This leaves the prefill/mixed path on eager kernel launches and can cause large host-side gaps under high concurrency. This PR adds the DeepSeek V4 and DP-attention plumbing required to make breakable piecewise CUDA graph capture/replay work for DSV4 mixed chunk workloads. ⏎  ⏎ Co-authored by: @Oasis-Git  ⏎  ⏎ ## Modifications …[truncated]

### L2-8ae328e5f0  (L2, 2026-06-09, sha 8ae328e5f042, PR #27647)
TITLE: [sgl] Fix kimi-k2.5 EAGLE3 MLA draft embeds for batched MM prefill (#27647)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/kimi_k25_eagle3.py (+4/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Fixes a silent correctness bug in the kimi-k2.5 EAGLE3 MLA draft model when serving multimodal requests with batch size greater than 1. ⏎  ⏎ The draft model patches multimodal-aware input embeddings by overlaying real `embed_tokens` results on top of the target's `mm_input_embeds`. current code only patched the very last position of the flat extend buffer, which assumed batch size 1. For larger batches, requests 0..N-2 silently kep …[truncated]

### L2-186f1e300a  (L2, 2026-06-09, sha 186f1e300a68, PR #27644)
TITLE: [CI] Move JIT kernel tests + benchmarks to test/registered/jit; add in-package guard (#27644)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.kernel.set_mla_kv_buffer, L2.kernel.concat_mla, L2.kernel.mla_kv_pack_quantize_fp8
FILES: .claude/skills/add-jit-kernel/SKILL.md (+10/-10); .claude/skills/write-sglang-test/SKILL.md (+13/-13); .github/workflows/_pr-test-check-changes.yml (+3/-2); .github/workflows/pr-test-amd-rocm720.yml (+3/-2); .github/workflows/pr-test-amd.yml (+3/-2); .github/workflows/pr-test.yml (+1/-1); .pre-commit-config.yaml (+6/-0); docs/developer_guide/development_jit_kernel_guide.md (+1/-1); docs_new/docs/developer_guide/development_jit_kernel_guide.mdx (+1/-1); python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/existing-fast-paths.md (+15/-15); (+111 more)
LABELS: documentation, quant, amd, lora, hicache, blackwell, run-ci, diffusion, run-ci-extra
BODY: ## Summary ⏎ - Move all JIT kernel correctness tests **and** benchmarks out of the importable `sglang` package (`python/sglang/jit_kernel/{tests,benchmark}/`) into `test/registered/jit/`, so they are no longer shipped in the wheel and are discovered by the standard `test/registered/` glob in `run_suite.py`. ⏎ - Add a pre-commit guard that rejects any CI-registered test placed inside the package. ⏎ - Move the NIXL HiCache storage unit test from `base-a` …[truncated]

### L2-42322947aa  (L2, 2026-06-09, sha 42322947aa32, PR #27645)
TITLE: [BUG FIX]Fix DSA CPU offload mamba indices signature (#27645)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/memory_pool.py (+6/-4); test/registered/unit/mem_cache/test_dsa_pool_host_unit.py (+9/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎   Decode disaggregation may retract running requests under sustained load and offload their KV cache to CPU. The allocator forwards the optional `mamba_indices` argument to the underlying KV pool during ⏎   CPU offload. ⏎  ⏎   `DSATokenToKVPool.get_cpu_copy()` and `load_cpu_copy()` did not accept this argument, which can raise a runtime `TypeError` when DSA KV cache is offloaded during retract. ⏎  ⏎   ## Modifications ⏎  ⏎   - Add optio …[truncated]

### L2-2fef951fe8  (L2, 2026-06-09, sha 2fef951fe8a8, PR #23927)
TITLE: [AMD] Replace fp8 mla with fp8 mha kernel for diffusion model aiter backend (#23927)
SOURCES: subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/multimodal_gen/runtime/layers/attention/backends/aiter.py (+72/-257)
LABELS: run-ci, diffusion
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎ Dependency: https://github.com/ROCm/aiter/pull/2911 (merged) ⏎ The original FP8 prefill path in the AITer backend temporarily used the MLA prefill ASM kernel to get FP8 attention running on MI350 series for Wan 2.2. Switching to the proper FP8 FMHA ASM kernel fmha_fwd_hd128_fp8_gfx950 removes extra padding, the head-divisibility constraint, and the whole metadata-building scaffolding. That FMHA path natively supports MHA with q/k/v  …[truncated]
