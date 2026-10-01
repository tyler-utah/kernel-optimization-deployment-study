### L1-71e8258783  (L1, 2026-06-09, sha 71e825878348, PR #26635)
TITLE: Improve registration in cpu_graph_runner (#26635)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/multimodal_gen/runtime/managers/cpu_worker.py (+0/-4); python/sglang/srt/model_executor/cpu_graph_runner.py (+66/-28); python/sglang/srt/model_executor/model_runner.py (+0/-4)
LABELS: sgl-kernel, intel, cpu, run-ci, diffusion, run-ci-extra
BODY: ## Motivation ⏎  ⏎ During CPU graph compilation, these custom CPU ops are intended to run through Inductor fallback. When a custom op does not have an explicit lowering registered, Inductor can create an implicit fallback, but that path first builds diagnostic ⏎ messages for the missing lowering. For large CPU compile graphs, formatting those diagnostics may be extremely slow. ⏎  ⏎ ## Modifications ⏎  ⏎ * Explicitly registers fallback lowerings with `ma …[truncated]

### L1-186f1e300a  (L1, 2026-06-09, sha 186f1e300a68, PR #27644)
TITLE: [CI] Move JIT kernel tests + benchmarks to test/registered/jit; add in-package guard (#27644)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: .claude/skills/add-jit-kernel/SKILL.md (+10/-10); .claude/skills/write-sglang-test/SKILL.md (+13/-13); .github/workflows/_pr-test-check-changes.yml (+3/-2); .github/workflows/pr-test-amd-rocm720.yml (+3/-2); .github/workflows/pr-test-amd.yml (+3/-2); .github/workflows/pr-test.yml (+1/-1); .pre-commit-config.yaml (+6/-0); docs/developer_guide/development_jit_kernel_guide.md (+1/-1); docs_new/docs/developer_guide/development_jit_kernel_guide.mdx (+1/-1); python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/existing-fast-paths.md (+15/-15); (+111 more)
LABELS: documentation, quant, amd, lora, hicache, blackwell, run-ci, diffusion, run-ci-extra
BODY: ## Summary ⏎ - Move all JIT kernel correctness tests **and** benchmarks out of the importable `sglang` package (`python/sglang/jit_kernel/{tests,benchmark}/`) into `test/registered/jit/`, so they are no longer shipped in the wheel and are discovered by the standard `test/registered/` glob in `run_suite.py`. ⏎ - Add a pre-commit guard that rejects any CI-registered test placed inside the package. ⏎ - Move the NIXL HiCache storage unit test from `base-a` …[truncated]

### L1-a287ab83c0  (L1, 2026-06-09, sha a287ab83c0cd, PR #26791)
TITLE: Fix Gemma4 NVFP4 MoE default attention backend (#26791)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+13/-3)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ I noticed that `nvidia/Gemma-4-26B-A4B-NVFP4` is fully broken with the current SM10X default `trtllm_mha` backend. This is not just a small regression: MMLU drops from 0.622 with `triton` to 0.037 with the default, and the outputs look nonsensical. This PR changes only the affected Gemma4 NVFP4 MoE default to `triton`; users can still explicitly pass `--attention-backend trtllm_mha`. ⏎  ⏎ ## Accuracy Comparison ⏎  ⏎ | Model | Quant | Mo …[truncated]

### L1-c6be251c5b  (L1, 2026-06-09, sha c6be251c5bc6, PR #26717)
TITLE: [NPU] RL update_weights_from_disk/ tensor /distributed (#26717)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+9/-0)
LABELS: run-ci
BODY: ## Motivation ⏎ As for the rl featue update_weights_from_disk / tensor /distributed for moe models like qwen3moe, when the server first loads weights during begining stage, the process_weights_after_loading method of UnquantizedFusedMoEMethod change the param shape for moe weights w13 and w2, to successfully update weight and maintain accuracy, add this. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## C …[truncated]

### L1-2495c02c2c  (L1, 2026-06-09, sha 2495c02c2c57, PR #23906)
TITLE: [Refactor] Cuda Graph Runner/Backend Refactor (#23906)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_cutedsl, L1.runner.marlin, L1.runner.deepgemm_megamoe, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-2); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+7/-7); python/sglang/srt/layers/moe/mega_moe.py (+3/-3); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+17/-14); docs_new/docs/advanced_features/server_arguments.mdx (+105/-21); python/sglang/auto_benchmark_lib.py (+2/-2); python/sglang/bench_one_batch.py (+5/-1); python/sglang/compile_deep_gemm.py (+6/-2); python/sglang/srt/arg_groups/argparse_actions.py (+28/-0); python/sglang/srt/compilation/compile.py (+4/-2); (+150 more)
LABELS: documentation, high priority, lora, deepseek, speculative-decoding, blackwell, npu, run-ci, piecewise-cuda-graph, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/issues/23004  ⏎  ⏎ ## Modifications ⏎  ⏎ ### Runner ↔ CudaGraphBackend Interaction (cg-refactor) ⏎  ⏎ Two layers are in scope: ⏎  ⏎ - **Phase runner** — `PrefillCudaGraphRunner` / `DecodeCudaGraphRunner` ⏎   - Both subclass `BaseCudaGraphRunner` ⏎   - Exactly one runner per phase ⏎   - Fixed implementation ⏎  ⏎ - **CG backend** — `BaseCudaGraphBackend` and its subclasses: ⏎   - `FullCudaGraphBackend` ⏎   -  …[truncated]

### L1-01f10acd06  (L1, 2026-06-10, sha 01f10acd0669, PR #26083)
TITLE: Implement online nvfp4 quantization (#26083)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+2/-1); docs_new/docs/advanced_features/quantization.mdx (+31/-0); docs_new/docs/advanced_features/server_arguments.mdx (+1/-1); docs_new/docs/references/environment_variables.mdx (+8/-3); python/sglang/srt/configs/model_config.py (+3/-0); python/sglang/srt/environ.py (+2/-1); python/sglang/srt/layers/quantization/__init__.py (+2/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+42/-5); python/sglang/srt/layers/quantization/nvfp4_online.py (+583/-0); python/sglang/srt/model_loader/loader.py (+17/-2); (+2 more)
LABELS: documentation, quant, deepseek, blackwell, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎ @humansand ⏎ Add online NVFP4 MoE weight quantization under the `--quantization nvfp4_online` interface. ⏎  ⏎ After #22918, FlashInfer TRTLLM MoE can use runtime per-token activation scaling, so SGLang no longer needs a calibrated static activation FP32 scale for this path. The `nvfp4_online` interface is explicitly a load-time conversion mode: weights still use static NVFP4 block scales plus static per-tensor FP32 scales, while activatio …[truncated]

### L1-2947781ce6  (L1, 2026-06-10, sha 2947781ce6b3, PR #25455)
TITLE: [NPU] MiMo-V2-Flash Adaptation (#25455)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+1/-1); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+186/-37); python/sglang/srt/hardware_backend/npu/graph_runner/multi_layer_eagle_draft_extend_npu_graph_runner.py (+159/-0); python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py (+11/-0); python/sglang/srt/hardware_backend/npu/memory_pool_npu.py (+34/-17); python/sglang/srt/mem_cache/allocator/swa.py (+6/-4); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+0/-3); python/sglang/srt/models/mimo_v2.py (+8/-2); python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py (+14/-3)
LABELS: quant, npu, run-ci
BODY: ## Motivation ⏎  ⏎ Support Multi-Token Prediction (MTP) for MiMo-V2-Flash model on Ascend NPU (requires CANN 9.0). ⏎  ⏎ ## Modifications ⏎  ⏎ - Adapt NPUMHATokenToKVPool and its set_kv_buffer method to support MTP inference, including FIA mode and PagedAttention KV cache writing for NPU. ⏎ - Modify NPUGraphRunner to support NPU Graph capture/replay for MiMo-V2-Flash MTP workflow. ⏎ - Update the forward_mtp attention forward function to match MTP inferenc …[truncated]

### L1-8c6bbe0658  (L1, 2026-06-10, sha 8c6bbe065878, PR #27510)
TITLE: [deepseek] Enable DP attention + TBO + shared experts fusion (#27510)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/batch_overlap/two_batch_overlap.py (+5/-0); python/sglang/srt/models/deepseek_v2.py (+1/-3); python/sglang/srt/server_args.py (+0/-5); test/registered/ep/test_tbo_shared_experts_fusion.py (+73/-0)
LABELS: deepseek
ISSUES: #24690 [Feature] Enabling both TBO and shared experts fusion
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ This PR enables DP attention, TBO, and shared-experts fusion simultaneously. ⏎  ⏎ Closes #24690 ⏎  ⏎ ## Modifications ⏎ * Change 1: `python/sglang/srt/server_args.py` ⏎   *  Remove the guard that rejected `--enable-two-batch-overlap` together with `--enforce-shared-experts-fusion`, so the two flags can now be enabled together.  ⏎  ⏎ Run the test and get error https://github.com/sgl-project/sglang/pull/27510#discussion_r3370366772: ⏎ * [la …[truncated]

### L1-0ae27405d0  (L1, 2026-06-10, sha 0ae27405d08a, PR #22985)
TITLE: [AMD] Support eplb for moriep (#22985)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py, L1.runner.aiter, L1.ep.other_dispatchers, L1.upstream.aiter_moe
FILES: python/sglang/srt/layers/moe/moe_runner/aiter.py (+6/-1); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+27/-5); python/sglang/srt/layers/moe/topk.py (+3/-0); docs_new/docs/references/environment_variables.mdx (+5/-0); python/sglang/srt/eplb/expert_distribution.py (+3/-0); python/sglang/srt/eplb/expert_location_dispatch.py (+9/-1); python/sglang/srt/eplb/expert_location_updater.py (+28/-4); test/manual/ep/test_eplb_mori.py (+242/-0); test/registered/amd/test_moriep_small.py (+68/-0)
LABELS: documentation, amd, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ This patch is to  ⏎ - Support eplb for moriep ⏎ - Fix rccl `batch_isend_irecv` hang during EPLB rebalance by chunking p2p ops ⏎ - Add `SGLANG_EPLB_ROCM_P2P_BATCH_CHUNK_SIZE` env var to control p2p chunking granularity ⏎ - Add unittest `test/manual/ep/test_eplb_mori.py` ⏎  ⏎ cc @Duyi-Wang @HaiShaw  ⏎  ⏎ ## Modifications ⏎  ⏎ **EPLB + MoRI integration** (`moriep.py`, `expert_distribution.py`, `topk.py`): ⏎ - Wire up `local_expert_count` fro …[truncated]

### L1-3c1b0fb226  (L1, 2026-06-10, sha 3c1b0fb22670, PR #27312)
TITLE: [1/n] [CP] Simplify prefill context parallel server args (#27312)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs_new/docs/advanced_features/server_arguments.mdx (+6/-6); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+6/-4); python/sglang/srt/server_args.py (+131/-42); test/manual/test_dsa_alias_cli_registry_env.py (+6/-45); test/registered/unit/server_args/test_server_args.py (+169/-1)
LABELS: documentation, deepseek
BODY: Part of #27252. This PR splits out the first roadmap item: simplify the prefill context parallel server args. ⏎  ⏎ Summary: ⏎ - Adds canonical `--enable-prefill-cp` and `--cp-strategy {zigzag,interleave}`. ⏎ - Routes deprecated CP flags and mode flags into the unified fields. ⏎ - Mirrors unified fields back into legacy attrs so existing runtime paths keep working. ⏎ - Updates CP-related server-args tests and manual alias expectations. ⏎  ⏎ Validation: ⏎ - Pre-comm …[truncated]

### L1-f8b0a120b8  (L1, 2026-06-10, sha f8b0a120b802, PR #27747)
TITLE: fix: DSV4 BCG compress-prefill plan OOB on underfilled (tiny) prefill replay (#27747)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh (+5/-1)
LABELS: high priority, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ Under breakable CUDA graph (BCG) with `--enforce-piecewise-cuda-graph`, a DeepSeek-V4 prefill that is **much smaller than the captured graph bucket** crashes the scheduler with an illegal memory access (IMA) in the DSV4 paged-compressor prefill plan (`plan_compress_prefill`, `c_plan.cuh`). The most reliable trigger is the tiny startup server-warmup request (a few tokens) replayed into e.g. the 2048-token bucket; any short real pref …[truncated]

### L1-b8376aebd0  (L1, 2026-06-11, sha b8376aebd092, PR #27858)
TITLE: [AMD] Fix the dsv4 performance of MoE issue. (#27858)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8.py (+19/-11)
LABELS: amd, deepseek, run-ci
BODY: # [AMD] Fix the dsv4 performance of MoE issue ⏎  ⏎ ## Motivation ⏎  ⏎ On AMD (MI355X / gfx950), the DeepSeek-V4 FP4 MoE path runs the aiter ⏎ `fused_moe` kernels (`mfma_moe1_silu_mul`, `mfma_moe2_afp8_wfp4_bf16_cshuffle`) ⏎ noticeably slower than the reference ATOM engine for the same model/workload. ⏎  ⏎ Root cause: the aiter FP4 MoE kernel requires the per-partition intermediate ⏎ size to be 256-aligned, so during weight loading we pad it ⏎ (`moe_intermediate_size …[truncated]

### L1-c0480a88be  (L1, 2026-06-11, sha c0480a88bee7, PR #27964)
TITLE: [Spec] Retire Spec V1 (#27964)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .claude/skills/env-var-conventions/SKILL.md (+1/-1); docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3.mdx (+1/-1); docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx (+1/-1); docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+1/-1); docs_new/cookbook/autoregressive/GLM/GLM-4.5.mdx (+1/-1); docs_new/cookbook/autoregressive/GLM/GLM-4.6.mdx (+1/-1); docs_new/cookbook/autoregressive/GLM/GLM-4.7-Flash.mdx (+1/-1); docs_new/cookbook/autoregressive/GLM/GLM-4.7.mdx (+1/-1); docs_new/cookbook/autoregressive/OpenAI/GPT-OSS.mdx (+1/-1); docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx (+1/-1); (+36 more)
LABELS: documentation, high priority, deepseek, speculative-decoding, blackwell, npu, run-ci, run-ci-extra
BODY: `SGLANG_ENABLE_SPEC_V2` no longer selects a worker: every speculative algorithm runs the V2 workers, so the flag's only remaining effect was acting as an alias for `--disable-overlap-schedule`. This PR removes the flag entirely -- deleting it is the last user-visible step of retiring spec V1. ⏎  ⏎ What's in here (all traceable to the flag): ⏎  ⏎ - **Runtime**: drop the `Envs` entry, the three `speculative_hook` env-to-overlap mappings, the dsv4 hook forc …[truncated]

### L1-97a0031799  (L1, 2026-06-11, sha 97a00317993e, PR #27984)
TITLE: [lint] Enable Ruff UP037 to drop redundant quoted annotations (#27984)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.routing.topk_py, L1.hardware.cpu_npu_musa, L1.runner.openai_triton_kernels, L1.runner.flashinfer_cutedsl, L1.runner.flashinfer_mxfp4, L1.runner.deepgemm_megamoe
FILES: .pre-commit-config.yaml (+1/-1); python/sglang/_mps_stub.py (+2/-2); python/sglang/jit_kernel/kv_canary/plan/entries_kernel.py (+1/-1); python/sglang/jit_kernel/kv_canary/verify.py (+3/-3); python/sglang/jit_kernel/kv_canary/write.py (+3/-3); python/sglang/jit_kernel/tests/kv_canary/_canary_helpers.py (+1/-1); python/sglang/multimodal_gen/configs/quantization/nunchaku.py (+2/-2); python/sglang/multimodal_gen/runtime/distributed/cfg_parallel_utils.py (+7/-7); python/sglang/multimodal_gen/runtime/distributed/cfg_policy.py (+5/-5); python/sglang/multimodal_gen/runtime/layers/attention/layer.py (+1/-1); (+175 more)
LABELS: quant, lora, deepseek, hicache, blackwell, npu, run-ci, piecewise-cuda-graph, diffusion, jit-kernel
BODY: ## Motivation ⏎  ⏎ With `from __future__ import annotations`, string-quoted type annotations such as `forward_batch: "ForwardBatch"` are unnecessary — the annotation is never evaluated as a real expression, so the quotes are pure noise. This PR enables Ruff's [`UP037` (`quoted-annotation`)](https://docs.astral.sh/ruff/rules/quoted-annotation/) rule so these redundant quotes are caught and removed automatically by the existing pre-commit ruff hook, an …[truncated]

### L1-6ac9f66596  (L1, 2026-06-11, sha 6ac9f6659654, PR #27841)
TITLE: Remove MoE prefill CUDA graph disable guard (#27841)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+0/-16)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ Remove the MoE prefill CUDA graph auto-disable guard for `flashinfer_trtllm` and `flashinfer_mxfp4`, which appears to have been reintroduced accidentally in the CUDA graph runner/backend refactor (#23906). ⏎  ⏎ The blanket disable does not seem to be the right fix for the unclear FlashInfer/BF16 MoE IMA reports such as #27712, so this re-enables PCG for these backends while that issue is investigated separately. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ La …[truncated]

### L1-06e0df5899  (L1, 2026-06-11, sha 06e0df5899aa, PR #26204)
TITLE: Optimize Qwen3 Next FP8 MoE on H200 (#26204)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=513,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json (+162/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=513,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128]_down.json (+182/-0); python/sglang/srt/models/qwen2_moe.py (+31/-18); python/sglang/srt/models/qwen3_next.py (+39/-9); python/sglang/srt/models/qwen3_next_mtp.py (+13/-1)
LABELS: run-ci, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ - Enable guarded CUDA shared-expert fusion for Qwen3-Next FP8 MoE on H200. ⏎ - Remap `mlp.shared_expert.*` checkpoint weights into the fused MoE expert slot for Qwen3-Next. ⏎ - Add H200 FP8 Triton MoE configs for fused `E=513,N=512` normal and down projections. ⏎ - Keep the fusion opt-in for Qwen3-Next so existing Qwen2-MoE paths keep their default behavior. ⏎  ⏎ ## Performance ⏎  ⏎ Environment: `ion8-h200`, `sglang_bbuf`, single NVIDIA H200, TP=1, …[truncated]

### L1-f23f48df98  (L1, 2026-06-12, sha f23f48df98f7, PR #27945)
TITLE: fix(moe): make FlashInfer A2A robust to collapsed global_num_tokens (moe_dense_tp_size NaN) (#27945)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+7/-0); python/sglang/srt/entrypoints/openai/protocol.py (+1/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `--moe-a2a-backend flashinfer` with `--enable-dp-attention` silently produces NaN/inf logits (gsm8k ~0, or a `probability tensor contains inf/nan` device-side assert under sampling) when `--moe-dense-tp-size 1` is set. The FlashInfer CuteDSL/Cutlass MoE kernels are correct on their own; the corruption comes from per-rank A2A dispatch sizing. ⏎  ⏎ Root cause: `moe_dense_tp_size` set makes `require_mlp_tp_gather()` return `False`, so the …[truncated]

### L1-82eedd5bd0  (L1, 2026-06-12, sha 82eedd5bd0dd, PR #27107)
TITLE: [DeepEP] Enable fabric handles automatically when supported (#27107)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+21/-4)
LABELS: run-ci, run-ci-extra
ISSUES: #26196 DeepEP Buffer initialization fails with 'invalid resource handle' on GB200
BODY: ## Motivation ⏎  ⏎  ⏎ Fix https://github.com/sgl-project/sglang/issues/26196 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ Automatically set `use_fabric=True` for the DeepEP comm buffer when FlashInfer reports MNNVL fabric support and the installed DeepEP `Buffer` exposes the `use_fabric` kwarg ⏎  ⏎ See https://github.com/deepseek-ai/DeepEP/blob/e0a5b1d9848ab3e7b4a67842bf06f067bfac67f8/csrc/deep_ep.cpp#L101-L114 ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  …[truncated]

### L1-6c3e429ba1  (L1, 2026-06-12, sha 6c3e429ba131, PR #28107)
TITLE: [Tiny] Cuda Graph Refactor Code Style Follow up (#28107)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/runner_utils/deepep_adapter.py (+13/-0); docs_new/docs/advanced_features/server_arguments.mdx (+2/-14); python/sglang/srt/kv_canary/api.py (+2/-3); python/sglang/srt/model_executor/cuda_graph_buffer_registry.py (+13/-0); python/sglang/srt/model_executor/cuda_graph_config.py (+13/-0); python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py (+17/-0); python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py (+36/-30); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+0/-21); python/sglang/srt/model_executor/runner/shape_key.py (+13/-0); python/sglang/srt/model_executor/runner_backend/base_cuda_graph_backend.py (+13/-0); (+11 more)
LABELS: documentation, deepseek, run-ci
BODY: Follow up of #23906  ⏎  ⏎ 1. Revert necessary comments in cuda graph related files ⏎ 2. Add license for new files ⏎ 3. remove the unused `--prefill/decode-cuda-graph-backend` ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.gi …[truncated]

### L1-d1a39b0c74  (L1, 2026-06-12, sha d1a39b0c74ed, PR #27720)
TITLE: [DeepSeek V3] Defer moe finalize and fused it with main stream add (#27720)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_trtllm
FILES: python/sglang/jit_kernel/csrc/moe/moe_finalize_fuse_shared.cu (+418/-0); python/sglang/jit_kernel/csrc/moe/tvm_ffi_utils.h (+105/-0); python/sglang/jit_kernel/moe_finalize_fuse_shared.py (+60/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+29/-0); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+102/-29); python/sglang/srt/models/deepseek_v2.py (+28/-7); python/sglang/srt/environ.py (+1/-0)
LABELS: deepseek, run-ci, jit-kernel, bypass-fastfail
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Optimizing Kimi K2.5 NVFP4 ⏎ Note: this PR does not change behaviour during prefill phase yet (still 2 individual kernel launches). More work needed to speedup there ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - Swap the execution order. Main stream now execute routed experts, alt stream execute shared experts ⏎ - On main stream, defer finalized (`do_finalize = False`) and fuse finalize with add kernel  ⏎    ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Te …[truncated]

### L1-60d4bd4c70  (L1, 2026-06-13, sha 60d4bd4c70b2, PR #27855)
TITLE: [AMD] fix moriep quant kernel not implemented issue (#27855)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.runner.aiter, L1.upstream.aiter_moe
FILES: python/sglang/srt/layers/moe/moe_runner/aiter.py (+16/-1)
LABELS: amd, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ This patch is to fix the dsv4 server + mori ep8 crash during warmup with the following error message: ⏎  ⏎ server command: ⏎ ```bash ⏎ SGLANG_MORI_DISPATCH_DTYPE=fp4         \ ⏎ SGLANG_MORI_COMBINE_DTYPE=fp8               \ ⏎ SGLANG_DEFAULT_THINKING=1                      \ ⏎ SGLANG_DSV4_REASONING_EFFORT=max               \ ⏎ SGLANG_OPT_DEEPGEMM_HC_PRENORM=false           \ ⏎ SGLANG_USE_AITER=1                             \ ⏎ SGLANG_USE_ …[truncated]

### L1-3f4a338212  (L1, 2026-06-13, sha 3f4a338212b2, PR #18182)
TITLE: [AMD][Quantization] Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs (#18182)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs_new/docs/advanced_features/quantization.mdx (+17/-1); python/sglang/srt/configs/model_config.py (+8/-1); python/sglang/srt/layers/linear.py (+1/-0); python/sglang/srt/layers/quantization/dequantization.py (+44/-0); python/sglang/srt/layers/quantization/fp8.py (+130/-49); python/sglang/srt/layers/quantization/fp8_utils.py (+2/-0); python/sglang/srt/layers/quantization/online_quantization.py (+23/-0); python/sglang/srt/layers/quantization/quark/quark.py (+61/-11); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py (+214/-33); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py (+374/-28); (+5 more)
LABELS: documentation, quant, deepseek, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 28213 (confirmed_revert, reason=ci_or_test_failure) || deep-study: this PR was reverted by PR 28291 (reland, reason=other)
BODY: ## Motivation ⏎  ⏎ Popular recent language models have been released directly as FP8 checkpoints (e.g. https://huggingface.co/moonshotai/Kimi-K2-Instruct-0905 or https://huggingface.co/Qwen/Qwen3-32B-FP8). As part of proposed improvements (https://github.com/sgl-project/sglang/pull/18005) to MXFP4 quantization targeting Instinct MI350X and MI355X GPUs, this PR implements online quantization from FP8 checkpoint to MXFP4 layers. ⏎  ⏎ This PR is a follo …[truncated]

### L1-91c63aeb4d  (L1, 2026-06-13, sha 91c63aeb4dee, PR #28041)
TITLE: Fix stale CUDA graph benchmark and docs refs (#28041)
SOURCES: body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py (+1/-1); docs_new/docs/advanced_features/breakable_cuda_graph.mdx (+6/-6); docs_new/docs/advanced_features/piecewise_cuda_graph.mdx (+2/-2); python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/__init__.py (+2/-2)
LABELS: documentation, run-ci
BODY: ## Summary ⏎  ⏎ Trying to run `benchmark/kernels/fused_moe_triton/benchmark_torch_compile_fused_moe.py` hits a stale import from the old CUDA graph runner path: ⏎  ⏎ ```text ⏎ ModuleNotFoundError: No module named 'sglang.srt.model_executor.cuda_graph_runner' ⏎ ``` ⏎  ⏎ This updates the benchmark to import `set_torch_compile_config` from its current home. It also cleans up the matching stale CUDA graph paths in `docs_new/` and removes an outdated comment  …[truncated]

### L1-0e592395c7  (L1, 2026-06-13, sha 0e592395c70e, PR #26188)
TITLE: [Apple Silicon] [MLX] Fuse SwiGLU activation into gate gather_qmv for SwitchGLU MoE blocks (#26188)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/mlx/moe/__init__.py (+0/-0); python/sglang/srt/hardware_backend/mlx/moe/fused_swiglu.py (+573/-0); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/hardware_backend/mlx/model_runner.py (+16/-0); python/sglang/srt/hardware_backend/mlx/moe/tests/__init__.py (+0/-0); python/sglang/srt/hardware_backend/mlx/moe/tests/test_fused_swiglu.py (+464/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ SwitchGLU MoE blocks on Apple Silicon are dominated by two quantized gather matmuls (`up_proj`, `gate_proj`) plus a separate `silu(gate) x up` kernel. Each Metal dispatch carries command-buffer overhead (~1-4ms of GPU idle, profiled in #22114), so dispatch count drives decode latency. ⏎  ⏎ `FusedSwitchUpGate` (PR #24712) concatenated `up_proj` and `gate_proj` along the output dim into one matmul, but it doubled each kernel's output …[truncated]

### L1-171037c3e7  (L1, 2026-06-13, sha 171037c3e73d, PR #27869)
TITLE: Fix Qwen3.5 deterministic batch-invariant logprobs (#27869)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+7/-2); python/sglang/srt/layers/attention/fla/layernorm_gated.py (+5/-3); test/registered/attention/test_qwen35_deterministic.py (+44/-0)
BODY: ## Motivation ⏎  ⏎ Qwen3.5 deterministic inference had two batch-size-dependent numerical paths that could make exact output logprobs drift between a single request and the same request inside a larger batch. ⏎  ⏎ 1. The FLA gated layernorm Triton wrapper selected `ROWS_PER_BLOCK` from the total row count `M`. In the Qwen3.5 repro, the same row could run with `ROWS_PER_BLOCK=1` when decoded alone and `ROWS_PER_BLOCK=2` in a mixed batch, which changed …[truncated]

### L1-f18d38d040  (L1, 2026-06-14, sha f18d38d04084, PR #28213)
TITLE: Revert "[AMD][Quantization] Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs" (#28213)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs_new/docs/advanced_features/quantization.mdx (+1/-17); python/sglang/srt/configs/model_config.py (+1/-8); python/sglang/srt/layers/linear.py (+0/-1); python/sglang/srt/layers/quantization/dequantization.py (+0/-44); python/sglang/srt/layers/quantization/fp8.py (+49/-130); python/sglang/srt/layers/quantization/fp8_utils.py (+0/-2); python/sglang/srt/layers/quantization/online_quantization.py (+0/-23); python/sglang/srt/layers/quantization/quark/quark.py (+11/-61); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py (+33/-214); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py (+28/-374); (+5 more)
LABELS: documentation, quant, run-ci
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 18182 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#18182 ⏎  ⏎ it breaks the ci ⏎ https://github.com/sgl-project/sglang/actions/runs/27494298180/job/81265388307#step:6:260 ⏎  ⏎ See a correct fix here https://github.com/sgl-project/sglang/pull/28191 ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #27511029332](https://github.com/sgl-project/sglang/actions/runs/27511029332) ⏎ Latest PR Test (Extra): :x: [Run #27511029236](https://github.com/ …[truncated]

### L1-441b75ee69  (L1, 2026-06-14, sha 441b75ee6973, PR #27588)
TITLE: [quantization] NVFP4 MoE: split fused w13 gate/up global scales (#27588)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+25/-10); python/sglang/srt/layers/quantization/modelopt_quant.py (+56/-24); test/registered/unit/layers/quantization/test_modelopt_nvfp4_moe_scales.py (+308/-0)
LABELS: quant, blackwell, run-ci
BODY: ## Motivation ⏎  ⏎ In an NVFP4 MoE layer the `w13` GEMM fuses the gate (`w1`) and up (`w3`) ⏎ projections. Some checkpoints quantize the two halves with separate NVFP4 weight ⏎ global scales, stored as `[num_experts, 2]` (column 0 for gate, column 1 for up). ⏎  ⏎ Today the fused-MoE path drops column 1: it keeps only the gate scale and logs a ⏎ `w1_weight_scale_2 must match w3_weight_scale_2` warning when the columns differ. ⏎ The up half of the GEMM1 output the …[truncated]

### L1-63df86f5e7  (L1, 2026-06-14, sha 63df86f5e7e7, PR #28188)
TITLE: [AMD] Skip eplb bookkeeping and topk remap when EPLB is not in use on mori-ep / HIP (#22985) (#28188)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+21/-13); python/sglang/srt/layers/moe/topk.py (+34/-9)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: CC @billishyahao  ⏎ ## Motivation ⏎  ⏎ #22985 (`[AMD] Support eplb for moriep`) added EPLB support to the mori-ep / HIP path. Two of the added paths do not account for configurations where the EPLB machinery is not actually in use. ⏎  ⏎ ### 1. moriep recorder bookkeeping is not gated on the recorder being active ⏎  ⏎ The mori dispatch path requests the local expert count and feeds it to the expert-distribution recorder unconditionally: ⏎  ⏎ ```python ⏎ ... …[truncated]

### L1-063ab89ac1  (L1, 2026-06-15, sha 063ab89ac168, PR #26471)
TITLE: DeepSeek-V4 Online Compress support MTP (#26471)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/csrc/deepseek_v4/c128_online_v2.cuh (+52/-23); python/sglang/jit_kernel/csrc/deepseek_v4/online_c128_mtp.cuh (+537/-0); python/sglang/jit_kernel/dsv4/compress.py (+16/-7); python/sglang/jit_kernel/dsv4/online_c128_mtp.py (+256/-0); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+159/-11); python/sglang/srt/layers/attention/dsv4/compressor_v2.py (+17/-2); python/sglang/srt/mem_cache/deepseek_v4_compress_state.py (+12/-1); python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py (+28/-1); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+3/-0); (+2 more)
LABELS: documentation, quant, amd, dependencies, lora, deepseek, speculative-decoding, blackwell, npu, run-ci
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:62b3c8e177 (fix PR 28531) || deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ The issue with `online_compress+MTP` has been resolved. Performance is only about 2% lower than `NO_ONLINE+MTP`, but the number of tokens it can process has increased from 2,046,720 to 5,692,160—a roughly 280% improvement. ⏎  ⏎ Command ⏎ ``` ⏎ #prefill ⏎ SGLANG_OPT_USE_ONLINE_COMPRESS=1  SGLANG_DSV4_FP4_EXPERTS=1 SGLANG_DEFAULT_THINKING=1 SGLANG_JIT_DEEPGEMM_PRECOMPILE=1 GLOO_SOCKET_IFNAME=eth0 NCCL_MIN_NCHANNE …[truncated]

### L1-e985422b2b  (L1, 2026-06-15, sha e985422b2b0e, PR #27863)
TITLE: [Fix][MTP][MM] Fix EAGLE v2 chunked-prefill next-token chain crash on multimodal models due to placeholder tokens (#27863)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/schedule_batch.py (+10/-3)
LABELS: run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ An error was triggered when running a multimodal model with multimodal requests and enabling SpecV2. After enabling `ASCEND_LAUNCH_BLOCKING`, I observed that the crash occurred within the embedding layer.Upon debugging, I found that this regression was introduced by PR #26800. Specifically, the `_compute_chunked_req_next_prompt_token` method does not account for multimodal scenarios. In multimodal contexts, `input_ids` can co …[truncated]

### L1-09e9c4fde3  (L1, 2026-06-15, sha 09e9c4fde3b4, PR #28118)
TITLE: 【bugfix】The NPU's forward_dsa_prepare_npu also needs special handling for is_nextn (#28118)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py (+3/-3)
LABELS: deepseek, npu, run-ci, run-ci-extra
BODY: ## Motivation ⏎ Fixed the bug caused by 393d0e169e8e1bab59a85e35392078ce0c684d9b on the NPU platform ⏎  ⏎ The NPU's forward_dsa_prepare_npu also needs special handling for is_nextn ⏎  ⏎ ```shell ⏎ RuntimeError: split_with_sizes expects split_sizes to sum exactly to 512 (input tensor's size at dimension -1), but got split_sizes=[512, 64] ⏎  ⏎ [2026-06-12 08:28:30 DP11 TP22 EP22] Scheduler hit an exception: Traceback (most recent call last): ⏎   File "/sgl- …[truncated]

### L1-102392df5b  (L1, 2026-06-16, sha 102392df5b71, PR #28404)
TITLE: [AMD][Fix] Skip EPLB topk remap when global server args are unset (#28404)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+7/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ PR 28188 added _eplb_remap_enabled() to the HIP branch of _post_process_topk_ids, which calls get_global_server_args() unconditionally. This breaks unit tests that call select_experts directly without a running server runtime: get_global_server_args() raises ValueError("Global server args is not set yet!"). ⏎  ⏎ This regression only affects the AMD/HIP path. The CUDA branch does not probe server args, and the NPU/CPU paths do not e …[truncated]

### L1-92b42c8d8a  (L1, 2026-06-16, sha 92b42c8d8ade, PR #24515)
TITLE: LPLB: linear-programming load balancer for MoE expert parallelism (#24515)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.hash_topk, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-0); python/sglang/srt/eplb/expert_location_dispatch.py (+31/-1); python/sglang/srt/eplb/lplb_solver.py (+280/-0); python/sglang/srt/layers/moe/hash_topk.py (+30/-3); python/sglang/srt/layers/moe/topk.py (+47/-6); python/sglang/srt/model_executor/model_runner.py (+42/-0); python/sglang/srt/models/deepseek_v2.py (+20/-3); python/sglang/srt/server_args.py (+1/-1); python/sglang/jit_kernel/csrc/lplb/dispatch_probability.cuh (+122/-0); python/sglang/jit_kernel/csrc/lplb/ipm.cuh (+275/-0); (+9 more)
LABELS: dependencies, deepseek, run-ci, jit-kernel, release-highlight
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ Adds LPLB (Linear Programming Load Balancer), a new MoE expert-dispatch algorithm that solves a per-layer LP to balance token routing across redundant expert replicas. Opt-in via `--ep-dispatch-algorithm=lp`; default behavior unchanged. ⏎  ⏎ - New `lp` choice for `--ep-dispatch-algorithm` (complements `static`, `dynamic`, `fake`). ⏎ - 4 JIT-compiled CUDA kernels under `python/sglang/jit_kernel/csrc/lplb/`: a single-block IPM (`ipm.cuh`) usi …[truncated]

### L1-4f9b12c5dd  (L1, 2026-06-16, sha 4f9b12c5ddad, PR #28426)
TITLE: [XPU] Guard tvm_ffi import in dsv4 compress modules under TYPE_CHECKING (#28426)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/dsv4/compress.py (+4/-2)
LABELS: run-ci, jit-kernel, run-ci-extra
BODY: ## Summary ⏎  ⏎ - PR #26471 (DeepSeek-V4 Online Compress support MTP) reintroduced an unconditional `from tvm_ffi.module import Module` in `python/sglang/jit_kernel/dsv4/compress.py` and added the same import in the new `python/sglang/jit_kernel/dsv4/online_c128_mtp.py`. ⏎ - The XPU CI image (`intel/sglang-dev:latest`) does not ship `tvm_ffi`, so importing `sglang.jit_kernel.dsv4` (which `__init__.py` does eagerly from `compress`) fails with `ModuleNot …[truncated]

### L1-9c53853ea3  (L1, 2026-06-16, sha 9c53853ea3c0, PR #25702)
TITLE: Use pack topk ids triton kernel for flashinfer_trtllm_routed (#25702)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+6/-17)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ Use fused triton kernel from flashinfer_mxfp4 for the flashinfer_trtllm_routed backend. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ Before ⏎ ``` ⏎ sglang serve --model-path MiniMaxAI/MiniMax-M2.7 --moe-runner-backend flashinfer_trtllm_routed --dtype bfloat16 --tp 8 ⏎ python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1209 --parallel 1209 --platinum                           ⏎ Accuracy: 0.950                                     ⏎ Invalid:  …[truncated]

### L1-0ae4740bd1  (L1, 2026-06-16, sha 0ae4740bd111, PR #27328)
TITLE: fix: add missing clamp_limit for CompressedTensorsWNA16MoE (#27328)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ When deploying DeepSeek V4 W4A16 quantized model, the fused_marlin_moe call in CompressedTensorsWNA16MoE was missing the clamp_limit parameter, causing incorrect outputs. ⏎  ⏎ ## Modifications ⏎  ⏎ Pass clamp_limit=self.moe_runner_config.swiglu_limit to the fused_marlin_moe call. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #26991731789](https://github.com/sgl-project/sglang/actions/runs/26991731789) ⏎ Latest PR Test (Extra): : …[truncated]

### L1-f06e2d3d1f  (L1, 2026-06-16, sha f06e2d3d1ff0, PR #27690)
TITLE: Support asymmetric compressed-tensors MoE (#27690)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-0); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py (+55/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Compressed-tensors MoE quantization should support asymmetric weight quantization with zero-points. ⏎  ⏎ ## Modifications ⏎  ⏎ Relax the symmetric-only detection path and add zero-point parameter creation, shape tracking, Marlin repacking, and runtime forwarding for WNA16 MoE. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See …[truncated]

### L1-66ac385f52  (L1, 2026-06-16, sha 66ac385f520c, PR #28469)
TITLE: fix(moe): MoRI EP init_mori_op missing BF16 dispatch branch (#28469)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+4/-1)
BODY: ## Motivation ⏎  ⏎ BF16 is the default dispatch dtype but init_mori_op has no branch for it, so a BF16 dispatch falls through to the FP8 defaults (data_type=fp8, scale_dim=1). The op then runs FP8+scale dispatch/combine semantics on BF16 data and silently corrupts the result (no crash). Size the op for the real BF16 payload instead (data_type=params_dtype, scale_dim=0). ⏎  ⏎ Surfaced by a model with BF16 (unquantized) expert layers. ⏎  ⏎ ## Modificatio …[truncated]

### L1-c01f62e341  (L1, 2026-06-16, sha c01f62e34128, PR #28486)
TITLE: [bugfix] guard NVIDIA SM-capability checks with is_cuda() for AMD/ROCm (#28486)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+2/-2); python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py (+2/-1); python/sglang/srt/server_args.py (+2/-2)
BODY: ## Motivation ⏎  ⏎ Several capability gates use torch.cuda.is_available() with a torch.cuda.get_device_capability() major comparison to detect a specific NVIDIA architecture. On ROCm, torch.cuda.is_available() is True and AMD gfx9xx GPUs report device capability major == 9 (gfx950/MI355X reports (9, 5)), so these NVIDIA-intended checks misfire on AMD. Most notably the SBO guard hard-raises at startup on MI355X. ⏎  ⏎ Error for SBO run on AMD MI355x: ⏎  …[truncated]

### L1-27291118b9  (L1, 2026-06-16, sha 27291118b92d, PR #28402)
TITLE: Upgrade sgl-deep-gemm to 0.1.3 (#28402)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1)
LABELS: dependencies, run-ci, run-ci-extra
BODY: ## Summary ⏎ - Update `python/pyproject.toml` from `sgl-deep-gemm==0.1.2` to `sgl-deep-gemm==0.1.3`. ⏎ - Update the Docker `SGL_DEEP_GEMM_VERSION` build argument from `0.1.2` to `0.1.3`. ⏎ - Confirm `3rdparty/amd/wheel/sglang/pyproject.toml` already requires `deep-gemm>=0.1.3`. ⏎  ⏎ ## Impact ⏎ - Python package installs now pin `sgl-deep-gemm` to `0.1.3`. ⏎ - Docker builds that install the prebuilt `sgl_deep_gemm` wheel now pull the `0.1.3` release. ⏎ - The AMD  …[truncated]

### L1-72ccfec594  (L1, 2026-06-17, sha 72ccfec5949d, PR #28460)
TITLE: docs(cookbook): verify GLM-5.2 single-node B300 (FP8 + BF16) (#28460)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx (+3/-3); docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx (+61/-8); docs_new/src/snippets/configs/zai-org/glm-5.2.jsx (+12/-12)
LABELS: documentation
BODY: Follow-up to #28448. Benchmarked all 6 single-node **B300** cells on an 8×B300 node (v0.5.13.post1, flush-cache every run) and flipped them to `verified: true`. Coherence clean at every config (`finish_reason=stop`). ⏎  ⏎ ### B300 numbers (per-GPU, TP8) ⏎ | | LL c1 / c16 | balanced c64 / c256 | HT c1024 | ⏎ |---|---|---|---| ⏎ | **FP8** | 34 / 140 | 245 / 265 | 388 | ⏎ | **BF16** | 37 / 146 | 157 / 167 | 168 | ⏎  ⏎ (BF16 balanced/HT run plain TP8 — no DP-Attenti …[truncated]

### L1-b1d18d562b  (L1, 2026-06-17, sha b1d18d562bd0, PR #28572)
TITLE: chore: bump sglang-kernel version to 0.4.4 (#28572)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sglang-kernel` version to `0.4.4` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.4.4` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the `sglang-kernel` dependency are updat …[truncated]

### L1-53318911ca  (L1, 2026-06-17, sha 53318911cab4, PR #28567)
TITLE: Add get_parallel(): a structured accessor for parallel-topology state (#28567)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.ep.other_dispatchers, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+2/-2); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+5/-8); python/sglang/srt/layers/moe/kt_ep_wrapper.py (+2/-2); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+4/-7); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+3/-4); python/sglang/srt/layers/moe/topk.py (+4/-4); python/sglang/srt/layers/moe/utils.py (+4/-5); python/sglang/srt/batch_overlap/two_batch_overlap.py (+2/-2); python/sglang/srt/layers/activation.py (+3/-4); python/sglang/srt/layers/attention/aiter_backend.py (+7/-6); (+174 more)
LABELS: quant, Multi-modal, deepseek, speculative-decoding, blackwell, run-ci
BODY: ## Motivation ⏎  ⏎ Reading the parallel topology today means importing and calling a dozen free ⏎ functions spread across `parallel_state` and `dp_attention` ⏎ (`get_tensor_model_parallel_world_size`, `get_attention_tp_size`, ⏎ `get_moe_expert_parallel_world_size`, ...). This PR adds a single structured ⏎ accessor, `get_parallel()`, so call-sites use one import and one consistent ⏎ naming scheme, and so tests can force a topology without monkeypatching the ⏎ ind …[truncated]

### L1-0e5a66dca4  (L1, 2026-06-17, sha 0e5a66dca400, PR #27837)
TITLE: [AMD] Register 3 JIT kernel unit tests for AMD CI (#27837)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-amd-rocm720.yml (+3/-3); .github/workflows/pr-test-amd.yml (+3/-3); test/registered/jit/test_clamp_position.py (+2/-1); test/registered/jit/test_resolve_future_token_ids.py (+2/-1); test/registered/jit/test_rmsnorm_hf.py (+2/-1)
LABELS: amd, run-ci
BODY: ## Summary ⏎ - Add `register_amd_ci(suite="jit-kernel-unit-test-amd")` to backend-portable `test/registered/jit` tests whose JIT kernels build and pass on MI325 (gfx942), expanding AMD JIT coverage. CUDA registrations are unchanged. ⏎ - Align the `jit-kernel-unit-test-amd` step timeout with NV's 30-min budget (was 10 min) in `pr-test-amd.yml` and `pr-test-amd-rocm720.yml`. ⏎ - Continues the direction of #27644 (which moved JIT kernel tests into `test/r …[truncated]

### L1-6309fb9abb  (L1, 2026-06-17, sha 6309fb9abb65, PR #28378)
TITLE: [AMD] Fix Always mask padded topk_ids on HIP to prevent garbage MoE routing (DeepSeek-R1-MXFP4 accuracy regression) (#28378)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+9/-8)
LABELS: amd, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ `test_deepseek_r1_mxfp4_8gpu.py` (`stage-c-test-large-8-gpu-amd-mi35x`) was failing with a catastrophic GSM8K accuracy drop: ⏎  ⏎ ``` ⏎ AssertionError: 0.1379833206974981 not greater than 0.94 ⏎ ``` ⏎  ⏎ i.e. accuracy collapsed from **>0.94 to ~0.09–0.14** on `amd/DeepSeek-R1-MXFP4-Preview` (MI35x, TP=8). ⏎  ⏎ **Root cause:** commit `63df86f5` (PR #28188, *"Skip eplb bookkeeping and topk remap when EPLB is not in use on mori-ep / HIP"*) accidental …[truncated]

### L1-792cb3a5d0  (L1, 2026-06-18, sha 792cb3a5d069, PR #27377)
TITLE: fix: add missing guard for use_jit_ep_activation (#27377)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.deep_gemm
FILES: python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+1/-1)
LABELS: run-ci
BODY: Qwen 3.5 35B is crashing with DEP2 on the following assert: ⏎  ⏎ https://github.com/sgl-project/sglang/blob/d8487bad06eb305bcb1f1efcd5d89072b15bf0ec/python/sglang/jit_kernel/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh#L319 ⏎  ⏎ 35B has num_experts=256, moe_intermediate_size=512; with ep_size=2, local routed experts are about 128, so: ⏎  ⏎   D / 8 = 512 / 8 = 64 ⏎   E = 128 ⏎   64 < 128 ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: …[truncated]

### L1-e3026ef016  (L1, 2026-06-18, sha e3026ef01622, PR #28421)
TITLE: [3/N][CP] Implement zigzag CP strategy (#28421)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+39/-17); python/sglang/srt/layers/cp/base.py (+4/-8); python/sglang/srt/layers/cp/interleave.py (+3/-1); python/sglang/srt/layers/cp/utils.py (+101/-1); python/sglang/srt/layers/cp/zigzag.py (+264/-15); python/sglang/srt/model_executor/model_runner.py (+64/-3); python/sglang/srt/models/qwen2_moe.py (+3/-0); python/sglang/srt/models/qwen3_moe.py (+2/-1); python/sglang/srt/server_args.py (+11/-0); test/registered/cp/test_cp_strategy_unit.py (+386/-1); (+3 more)
LABELS: run-ci
BODY: ## Summary ⏎ - Implement the CP v2 zigzag strategy for Qwen3/FlashAttention, gated behind `SGLANG_ENABLE_CP_V2=1`. ⏎ - Add model-runner CP v2 metadata preparation, first-layer sequence sharding, final hidden-state gathering, and FlashAttention KV gather/save dispatch without changing legacy `cp_utils.py`. ⏎ - Add CP strategy unit tests plus a Qwen3 30B DeepEP e2e subtest with `--attn-cp-size 4 --ep 4 --moe-a2a-backend deepep`. ⏎  ⏎ ## Notes ⏎ - CP v2 d …[truncated]

### L1-9b10821c8e  (L1, 2026-06-18, sha 9b10821c8e6e, PR #25144)
TITLE: [NPU] Add Ascend NPU support for DeepSeek-V4 (#25144)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.hash_topk, L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+21/-0); python/sglang/srt/layers/moe/hash_topk.py (+5/-3); python/sglang/srt/layers/moe/topk.py (+7/-0); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+21/-10); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+10/-0); python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py (+1488/-0); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py (+625/-0); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_common_hooks.py (+356/-0); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py (+584/-0); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_req_to_token_pool.py (+128/-0); (+18 more)
LABELS: deepseek, npu, run-ci, jit-kernel
BODY: ## Summary ⏎  ⏎ Adds end-to-end Ascend NPU support for the **DeepSeek-V4** architecture, covering both the **V4-Flash** and **V4-Pro** variants (hybrid-SWA with c1 / c4 / c128 compression ratios, 256 experts, EP). ⏎  ⏎ ## What's new ⏎  ⏎ ### New NPU core files (DSV4 backend, pools, allocator) ⏎  ⏎ | File | Change | ⏎ | --- | --- | ⏎ | `hardware_backend/npu/attention/ascend_dsv4_backend.py` | **new (1192)** — DSV4 NPU attention backend. Subclasses the Ascend backend, …[truncated]

### L1-05ee93c44f  (L1, 2026-06-18, sha 05ee93c44f01, PR #28555)
TITLE: Remove redundant cast and copy in calling `trtllm_fp8_block_scale_moe` (#28555)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+4/-12)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ There are 2 elementwise that don't need to exist. ⏎  ⏎ 1 is the downcast of correction bias from FP32 to BF16, the kernel can actually accept fp32 directly, for both fp4_block_scale and fp8_block_scale. ⏎ 2 is the .contiguous when the output of fp8 quantize is not column major, which is required by this API. ⏎ <img width="1677" height="880" alt="Screenshot 2026-06-17 at 12 58 21 PM" src="https://github.com/user-attachments/assets/0dc …[truncated]

### L1-c7397de571  (L1, 2026-06-18, sha c7397de571f7, PR #28231)
TITLE: Use Marlin for SM120 MXFP4 MoE (#28231)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.marlin
FILES: python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+27/-3); python/sglang/srt/layers/moe/fused_moe_triton/mxfp4_moe_sm120_triton.py (+0/-454); python/sglang/srt/layers/moe/moe_runner/marlin.py (+10/-1); python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py (+27/-76); python/sglang/srt/layers/quantization/marlin_utils.py (+18/-8); python/sglang/srt/layers/quantization/marlin_utils_fp4.py (+78/-6); python/sglang/srt/layers/quantization/mxfp4.py (+42/-81); python/sglang/srt/server_args.py (+2/-2)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎  ⏎ Implement MXFP4 Marlin MoE for GPT-OSS and make `marlin` the default GPT-OSS MXFP4 MoE backend on SM120. ⏎  ⏎ With GPT-OSS MXFP4 now covered by `marlin` on SM120, there is no longer a reason to keep the SM120-specific GPT-OSS `triton_kernel` path, especially since `triton_kernels` is mainly developed for data center GPU targets and is not expected to be the right long-term path for this SM120 use case, as mentioned [here](https://gith …[truncated]

### L1-0eded9e208  (L1, 2026-06-18, sha 0eded9e208a4, PR #28661)
TITLE: Add Laguna-M.1 cookbook (#28661)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/Poolside/Laguna-M.1.mdx (+234/-0); docs_new/cookbook/autoregressive/Poolside/Laguna-XS.2.mdx (+0/-1); docs_new/cookbook/autoregressive/intro.mdx (+1/-1); docs_new/docs.json (+1/-0); docs_new/src/snippets/configs/poolside/laguna-m1-benchmarks.jsx (+81/-0); docs_new/src/snippets/configs/poolside/laguna-m1.jsx (+302/-0)
LABELS: documentation
BODY: Adds the config-driven SGLang Cookbook page for **poolside/Laguna-M.1** under `docs_new/`. ⏎  ⏎ ## What's included ⏎ - **Deploy matrix**: H200 + B200 / B300 / GB200 / GB300. Quantizations: **BF16** (all), **FP8** (Hopper-only), **NVFP4** (Blackwell-only). Single-node — `--tp 8` on the 8-GPU HGX platforms (H200/B200/B300), `--tp 4` on the 4-GPU Grace-Blackwell nodes (GB200/GB300). ⏎ - **Single Balanced recipe** per cell, carrying the `poolside_v1` reasoni …[truncated]

### L1-1c6331cbd6  (L1, 2026-06-19, sha 1c6331cbd67f, PR #28384)
TITLE: refactor(runner): rename runner replay/load/can_run for the shared surface (#28384)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py (+2/-2); python/sglang/srt/model_executor/cpu_graph_runner.py (+2/-2); python/sglang/srt/model_executor/cuda_graph_buffer_registry.py (+1/-1); python/sglang/srt/model_executor/forward_batch_info.py (+2/-2); python/sglang/srt/model_executor/model_runner.py (+5/-5); python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py (+9/-9); python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py (+5/-5); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+6/-6); python/sglang/srt/speculative/base_spec_worker.py (+6/-2); python/sglang/srt/speculative/dflash_info.py (+2/-2); (+10 more)
LABELS: speculative-decoding, ready-to-merge, npu
BODY: ## Motivation ⏎  ⏎ Three runner methods are dispatched polymorphically across the cuda-graph runners (and a forthcoming eager runner). Give them a shared, intent-revealing surface; leave the cuda-graph-only methods named as they are. ⏎  ⏎ ## Modifications ⏎  ⏎ - Rename, across `Base`/`Decode`/`Prefill CudaGraphRunner` and the speculative / NPU / CPU graph runners: ⏎   - `replay()` → `execute()` (per-iter run) ⏎   - `replay_prepare()` → `load_batch()` (per-iter f …[truncated]

### L1-856b0dc74b  (L1, 2026-06-19, sha 856b0dc74b77, PR #28739)
TITLE: refactor(runner): move kernel warmup into the shared runner lifecycle (warmup()) (#28739)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py (+8/-3); python/sglang/srt/model_executor/model_runner.py (+51/-467); python/sglang/srt/model_executor/runner/__init__.py (+1/-1); python/sglang/srt/model_executor/runner/base_runner.py (+588/-2); python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py (+12/-133); python/sglang/srt/model_executor/runner/eager_runner.py (+2/-0); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+3/-0); python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py (+5/-0); python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py (+1/-0); python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py (+1/-0); (+6 more)
BODY: ## Motivation ⏎  ⏎ Kernel warmup / flashinfer autotune belongs to the runner lifecycle and should run at init, not as a standalone `ModelRunner` step or in the forward path. ⏎  ⏎ ## Modifications ⏎  ⏎ - `BaseRunner.warmup()` holds the orchestration (run-once guard, device guard, flashinfer allreduce pre-init, flashinfer autotune, PP-DeepGEMM warmup). The cuda-graph runners warm up from `capture()`; the `EagerRunner` warms up in its `__init__` (it has no capt …[truncated]

### L1-420004827c  (L1, 2026-06-19, sha 420004827c4e, PR #28736)
TITLE: [AMD] register 3 tests to stage-b-test-1-gpu-large-amd (batch-6) (#28736)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/attention/test_normal_decode_set_metadata.py (+2/-1); test/registered/lora/test_lora_update.py (+5/-1); test/registered/unit/spec/test_resolve_swa_kv_pool.py (+2/-1)
LABELS: lora, run-ci
BODY: ## Summary ⏎  ⏎ Batch-6 of the NV→AMD CI coverage audit. Registers 3 backend-agnostic / triton-path `base-b-test-1-gpu-large` CUDA tests to the AMD lane (`stage-b-test-1-gpu-large-amd`), AMD est ≈ 1.5× CUDA. All 3 verified green on **both** AMD lanes (mi325 + rocm720) via `continue_on_error` dispatch on the PR head. ⏎  ⏎ | File | AMD est (s) | What it exercises | ⏎ |---|---|---| ⏎ | `lora/test_lora_update.py` | 730 | `csgmv` triton LoRA backend dynamic load/ …[truncated]

### L1-6b945c16f4  (L1, 2026-06-19, sha 6b945c16f496, PR #28091)
TITLE: [LoRA] Fix experimental fast-path multi-adapter correctness + flashinfer 0.6.12 compatibility (#28091)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/trtllm_fused_moe_kernel_launcher.cu (+65/-18); python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/trtllm_fused_moe_runner.cu (+19/-7); python/sglang/jit_kernel/trtllm_lora_temp/data/include/flashinfer/trtllm/fused_moe/runner.h (+1/-0); python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/fused_activation_quant.cuh (+2/-2); python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/fused_permute_quant.cuh (+4/-4); python/sglang/srt/lora/trtllm_lora_temp/lora_dispatch.py (+4/-6); python/sglang/srt/lora/trtllm_lora_temp/moe_overlap.py (+60/-8); python/sglang/srt/lora/trtllm_lora_temp/sgl_fp8_moe.py (+2/-2); python/sglang/srt/lora/trtllm_lora_temp/shared_add_overlap.py (+7/-0); python/sglang/srt/lora/trtllm_lora_temp/triton_ops/gate_up_lora_b.py (+7/-3); (+5 more)
LABELS: quant, lora, run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ Follow-up fixes for the experimental fast LoRA path introduced in #27329 (`--moe-runner-backend experimental_sgl_trtllm` + `SGLANG_EXPERIMENTAL_LORA_OPTI=1`). ⏎  ⏎ Three problems are fixed: ⏎  ⏎ 1. **flashinfer 0.6.12 incompatibility.** The vendored TRT-LLM fused-MoE JIT kernels were forked from flashinfer 0.6.11.post1 and fail to compile against 0.6.12 (`get_sf_out_offset_128x4/_8x4` signature change, new `expertIds` parameter in th …[truncated]

### L1-1115373668  (L1, 2026-06-20, sha 111537366815, PR #28244)
TITLE: [AMD] Fix garbled unquantized Qwen3-30B-A3B output on ROCm/aiter where the aiter CK fused-MoE falls back to Triton with pre-shuffled weights (#28244)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/unquant.py (+35/-18); test/registered/amd/accuracy/mi35x/test_qwen3_moe_eval_mi35x.py (+230/-0)
LABELS: quant, run-ci
BODY: Co-authored-with: @XinyuJiangCMU  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ On ROCm (gfx950, MI350X/MI355X) with `SGLANG_USE_AITER=1`, bf16 `Qwen/Qwen3-30B-A3B` emits garbage whenever tensor parallelism shards the MoE intermediate to a size the aiter CK kernel cannot handle (`tp=8 -> 768 / 8 = 96`). `tp=1` (intermediate 768) is fine. ⏎  ⏎ ``` ⏎ SGLANG_USE_AITER=1 python3 -m sglang.launch_server \ ⏎   --model-path Qwen/Qwen3-30B-A3B \ ⏎   --tp 8 \ ⏎   --mem-fraction-stat …[truncated]

### L1-8a3d6c3403  (L1, 2026-06-20, sha 8a3d6c3403e8, PR #28811)
TITLE: Sort pyproject dependency lists (#28811)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+26/-25); python/pyproject_cpu.toml (+13/-12); python/pyproject_npu.toml (+20/-19); python/pyproject_other.toml (+34/-33); python/pyproject_xpu.toml (+17/-16); python/sglang/launch_server.py (+1/-0)
LABELS: dependencies, npu, run-ci
BODY: ## Summary ⏎ - Sort dependency lists in python pyproject variants alphabetically by package name. ⏎ - Add a comment asking future editors to keep dependency lists sorted. ⏎  ⏎ ## Test Plan ⏎ - Parsed the touched TOML files with python3/tomllib. ⏎ - Verified parsed TOML content changes are reorder-only against HEAD. ⏎ - Verified dependency lists are sorted alphabetically. ⏎ - Ran git diff --check. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #2 …[truncated]

### L1-2552b860a3  (L1, 2026-06-20, sha 2552b860a345, PR #28337)
TITLE: [AMD][bugfix] Place TBO cuda-graph num_token_non_padded buffer on model devices (#28337)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/batch_overlap/two_batch_overlap.py (+3/-1); test/registered/unit/batch_overlap/test_tbo_cuda_graph_num_token_device.py (+82/-0)
LABELS: run-ci
BODY: ## Motivation ⏎ `TboCudaGraphRunnerPlugin` preallocates a persistent `_tbo_children_num_token_non_padded` buffer with a bare `torch.zeros((2,), dtype=torch.int32)`, which leaves it on **CPU**. ⏎ `ForwardBatch.num_token_non_padded` is contractually a tensor on the **model device** (`ForwardBatch.compute` does `.to(device, ...)`), and the eager TBO split path already honors this — `compute_tbo_children_num_token_non_padded_raw` moves its result to `g …[truncated]

### L1-c0bb04b67f  (L1, 2026-06-21, sha c0bb04b67f26, PR #25820)
TITLE: [NVIDIA] Support NVFP4 MoE for DeepSeek-V4 (#25820)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.routing.hash_topk, L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/hash_topk.py (+11/-1); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+5/-15); docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+25/-0); docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx (+50/-0); docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+181/-0); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+11/-0); python/sglang/srt/configs/model_config.py (+22/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+54/-0); python/sglang/srt/model_loader/loader.py (+21/-0); python/sglang/srt/models/deepseek_v4.py (+5/-1)
LABELS: documentation, quant, deepseek, blackwell, run-ci, nvidia, run-ci-extra, release-highlight
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ This PR enables NVFP4 quantization for MoE in DeepSeek-V4. NVFP4 should provide better performance compared to MXFP4. Use `--moe-runner-backend flashinfer_trtllm_routed`. ⏎  ⏎ Needs https://github.com/sgl-project/sglang/pull/25702 to help with perf ⏎  ⏎ ## Modifications ⏎  ⏎ * Read new NVFP4 quant info from model config/hf_quant_config.json. The NVFP4 checkpoint uses FP8 quantization but with `"moe_quant_algo": "NVFP4"`. ⏎ * Handle `app …[truncated]

### L1-be774d0acd  (L1, 2026-06-22, sha be774d0acd6e, PR #28774)
TITLE: [docs][cookbook] Laguna-M.1 playground: add HiCache; refresh EP / DP-Attention notes (#28774)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/src/snippets/configs/poolside/laguna-m1.jsx (+26/-10)
LABELS: documentation
BODY: ## What ⏎  ⏎ Updates the **Laguna-M.1** cookbook Playground (`docs_new/src/snippets/configs/poolside/laguna-m1.jsx`, consumed by the shared `_playground.jsx` engine). Config-only — no engine changes. All claims measured on **8×B200 (BF16)**. ⏎  ⏎ ## Changes ⏎  ⏎ - **HiCache** (new `hicache` axis — host L2 tier + write-policy select). **Verified:** on a zipfian shared-prefix workload, enabling the host L2 tier cut **mean TTFT ~36% / median ~43%** and lifted * …[truncated]

### L1-6779ca8d7f  (L1, 2026-06-22, sha 6779ca8d7f78, PR #28619)
TITLE: Fix Qwen MoE precision issue with PP and all-reduce fusion (#28619)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen2_moe.py (+14/-0)
LABELS: run-ci
ISSUES: #27744 [Bug] PP+TP precision issue
BODY: ## Motivation ⏎  ⏎ Fixes #27744 ⏎  ⏎  ⏎  ⏎   This is a small patch to fix the Qwen MoE precision issue when pipeline parallelism is used together with tensor ⏎   parallelism and FlashInfer all-reduce fusion. ⏎  ⏎   When all-reduce fusion is enabled, the post-expert all-reduce can be deferred and fused into the next layer's ⏎   prepare_attn path. This deferred reduction is tracked by _sglang_needs_allreduce_fusion, and the marker is normally ⏎   consumed by  …[truncated]

### L1-b28e990161  (L1, 2026-06-22, sha b28e990161a3, PR #28919)
TITLE: Migrate all ServerArgs fields to Annotated style, reduce add_cli_args by ~2400 lines (#28919)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/arg_utils.py (+23/-6); python/sglang/srt/server_args.py (+1789/-2836); test/registered/unit/test_server_args_cli_metadata.py (+1/-1); test/registered/unit/test_server_args_migration.py (+126/-0)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ - Migrate ~220 remaining ServerArgs fields from manual `parser.add_argument()`  ⏎   to the `A[type, Arg(...)]` annotated style, completing the migration started  ⏎   in #28830 ⏎ - Reduce `add_cli_args` from ~2400 lines to ~160 lines (only deprecated flags,  ⏎   3 dynamic-choice fields, and `--config` remain manual) ⏎ - Add `dest=field.name` support in `add_cli_args_from_dataclass` so that  ⏎   `from_cli_args` no longer needs manual name remapping …[truncated]

### L1-8dc27f6326  (L1, 2026-06-22, sha 8dc27f632625, PR #28689)
TITLE: [MoE] dedup triton_kernels backend quant-arg asserts and fill weight dtype guard (#28689)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+45/-19)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The `triton_kernel` MoE backend (`python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`) duplicated the same eight unsupported-arg asserts in both `triton_kernel_fused_experts` and `triton_kernel_fused_experts_with_bias`, and left the with-bias per-weight dtype check as a no-op `TODO`/`pass`. ⏎  ⏎ ## Modifications ⏎  ⏎ - Extract the eight shared quant-arg asserts (`use_fp8_w8a8`, `per_channel_quant`, `expert_map`, `w1_scale` …[truncated]

### L1-7c23d2255a  (L1, 2026-06-22, sha 7c23d2255a1f, PR #28712)
TITLE: [minimax-m3] Split 1/4: sparse attention ops + JIT kernels + config foundation (#28712)
SOURCES: path_core, symbol_pickaxe, dependency_pin, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.runner.triton
FILES: docker/rocm.Dockerfile (+2/-1); python/sglang/jit_kernel/csrc/moe/moe_topk_sigmoid.cuh (+544/-0); python/sglang/jit_kernel/moe_topk_sigmoid.py (+105/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-0); python/sglang/srt/layers/moe/moe_runner/base.py (+4/-0); python/sglang/srt/layers/moe/moe_runner/triton.py (+94/-22); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/minimax_m3_gfx950_mxfp8_compact_moe.json (+70/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=128,N=384,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128, 128].json (+164/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=129,N=384,device_name=NVIDIA_H200.json (+146/-0); benchmark/kernels/fused_moe_triton/common_utils.py (+34/-3); (+41 more)
LABELS: quant, amd, run-ci, jit-kernel, run-ci-extra
BODY: Part 1 of a 4-PR split of #27944 (MiniMax-M3). This PR is purely additive on top of `main`: new compute kernels, their tests/benchmarks, and the config helpers / env descriptors they depend on. It does **not** touch any generic forward / quant / attention runtime path, so it is dead-on-main and green without an e2e run. ⏎  ⏎ ## What's in this PR ⏎  ⏎ - `minimax_sparse_ops/**` — decode / prefill / naive sparse attention kernels + unit tests (self-containe …[truncated]

### L1-de3ec2c437  (L1, 2026-06-22, sha de3ec2c43769, PR #28884)
TITLE: [server_args] compute mem_fraction_static after dp chunked-prefill division (#28884)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+4/-1)
LABELS: run-ci, run-ci-extra
DEEP_STUDY: deep-study: this PR was reverted by PR 28991 (confirmed_revert, reason=ci_or_test_failure)
BODY: ## Bug ⏎ When `--enable-dp-attention` is on, `_handle_data_parallelism()` divides `chunked_prefill_size` by `dp_size` (each rank sees 1/dp_size of the activation). But it runs **after** `_handle_gpu_memory_settings()`, which uses `chunked_prefill_size` to reserve prefill activation memory: ⏎  ⏎ ```python ⏎ reserved_mem += chunked_prefill_size * 1.5   # activation reserve ⏎ ... ⏎ mem_fraction_static = (gpu_mem - reserved_mem) / gpu_mem ⏎ ``` ⏎  ⏎ So the auto `mem_f …[truncated]

### L1-4740f23e1f  (L1, 2026-06-22, sha 4740f23e1f6a, PR #28991)
TITLE: Revert "[server_args] compute mem_fraction_static after dp chunked-prefill division" (#28991)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+1/-4)
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 28884 reason=ci_or_test_failure
BODY: Reverts sgl-project/sglang#28884 ⏎  ⏎ This breaks the DeepEP-related tests... ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #28000732886](https://github.com/sgl-project/sglang/actions/runs/28000732886) ⏎ Latest PR Test (Extra): :x: [Run #28000732791](https://github.com/sgl-project/sglang/actions/runs/28000732791)

### L1-b43bd6824f  (L1, 2026-06-22, sha b43bd6824f8b, PR #28786)
TITLE: [B300] Enable FlashInfer allreduce for Qwen3-VL MoE (#28786)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+6/-4)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ This PR enables FlashInfer allreduce fusion by default for Qwen3-VL MoE on the existing SM100 auto-enable path used by other supported MoE models: ⏎  ⏎ - add `Qwen3VLMoeForConditionalGeneration` to the FlashInfer allreduce-fusion allowlist ⏎ - keep the existing guards unchanged: `flashinfer_allreduce_fusion_backend is None`, TP > 1, no DP attention, and `moe_a2a_backend == "none"` ⏎ - no FlashInfer JIT/AOT workaround is included in this PR ⏎  ⏎ Q …[truncated]

### L1-e0dc8b7137  (L1, 2026-06-23, sha e0dc8b7137df, PR #28084)
TITLE: [AMD] Fuse topk padded-token masking into a single Triton kernel (#28084)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+84/-4); test/registered/moe/test_topk_padded_region.py (+194/-0)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: On AMD HIP the aiter MoE kernels cannot use topk_ids=-1 to drop CUDA-graph padding rows, so padded tokens are masked by zeroing their routing weights instead. The eager `arange + (>=) + boolean index_put_` path issues several launch-latency-bound kernels per call, twice per MoE layer. ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Fused masking kernel: Added _fill_padded_rows, a single Triton kernel (one program per row; pad count read from device memory so it is C …[truncated]

### L1-e63b57da0b  (L1, 2026-06-23, sha e63b57da0bd4, PR #28292)
TITLE: [Fix] model init / XPU / transformers-v5 / bench-image fixes (#28292)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/benchmark/datasets/image.py (+10/-1); python/sglang/srt/managers/mm_utils.py (+2/-2); python/sglang/srt/models/afmoe.py (+1/-1); python/sglang/srt/models/baichuan.py (+5/-2); python/sglang/srt/models/gemma3_causal.py (+1/-2); python/sglang/srt/models/gemma3n_causal.py (+8/-3); python/sglang/srt/models/gemma3n_mm.py (+2/-2); python/sglang/srt/models/lightonocr.py (+4/-3); python/sglang/srt/models/phi3_small.py (+5/-2); python/sglang/srt/models/transformers.py (+3/-1)
LABELS: intel, ci, xpu, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Several model initialization and inference bugs causing server crashes across multiple models and platforms. All fixes are correctness-only with no effect on model outputs or forward-pass performance. ⏎  ⏎ ## Modifications ⏎  ⏎ ### Model initialization fixes ⏎  ⏎ **`gemma3_causal.py`** — `KeyError: 'factor'` for `gemma-3-1b-it` ⏎  ⏎ The 1B model's `rope_parameters["full_attention"]` has `rope_type=default` with no `"factor"` key. Old code uncondit …[truncated]

### L1-cedb43d522  (L1, 2026-06-23, sha cedb43d5229b, PR #28942)
TITLE: [DeepEP] Gate DeepEP MNNVL on fabric support (#28942)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+13/-7)
LABELS: run-ci, run-ci-extra
BODY: ## Summary ⏎  ⏎ Make DeepEP only allow MNNVL when the same fabric support check used for `use_fabric` says it is available ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #27956336913](https://github.com/sgl-project/sglang/actions/runs/27956336913) ⏎ Latest PR Test (Extra): :white_check_mark: [Run #27956789780](https://github.com/sgl-project/sglang/actions/runs/27956789780)

### L1-fc27ce0666  (L1, 2026-06-23, sha fc27ce06660f, PR #25665)
TITLE: Add GB10 FP8 fused MoE Triton config (#25665)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=128,N=768,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json (+146/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py (+48/-12)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎ Add a tuned fused MoE Triton FP8 config for NVIDIA GB10 for the Qwen3 MoE shape `E=128, N=768`. ⏎  ⏎ Without a GB10-specific config, SGLang falls back to the default MoE kernel config for this shape. ⏎  ⏎ ## Modifications ⏎  ⏎ Added the Triton 3.6.0 config generated by the fused MoE tuner: ⏎  ⏎ `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=128,N=768,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json` ⏎  ⏎ This PR intentionally  …[truncated]

### L1-d5e9176f65  (L1, 2026-06-24, sha d5e9176f6581, PR #27053)
TITLE: [BCG][GLM5] perf: BCG support and prefill enhancements (#27053)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+1/-0); python/sglang/srt/layers/attention/dsa/dsa_indexer.py (+216/-101); python/sglang/srt/layers/attention/dsa/utils.py (+20/-1); python/sglang/srt/layers/attention/dsa_backend.py (+6/-2); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+287/-104); python/sglang/srt/models/deepseek_v2.py (+89/-16); test/registered/cuda_graph/piecewise/test_pcg_glm5_fp8_tp8.py (+75/-0)
LABELS: documentation, deepseek, run-ci, piecewise-cuda-graph, run-ci-extra, release-highlight
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ https://github.com/sgl-project/sglang/pull/23351 introduced PCG for GLM5 (`nsa_indexer`). As the current implementation runs the full indexer path, which is not fully captured by a CUDA graph, we found that running the fast indexer in eager mode yields better performance. ⏎  ⏎ Moreover, the current PCG split creates a CUDA graph island with only 1 kernel, introducing unnecessary overhead during dispatch. ⏎  ⏎ After `2048` tokens, PCG …[truncated]

### L1-e4bf0043fe  (L1, 2026-06-24, sha e4bf0043fe80, PR #26980)
TITLE: [fix] Skip routed expert capture for draft model under spec v2 (#26980)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+6/-7); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-0); python/sglang/srt/layers/moe/topk.py (+25/-5); python/sglang/srt/model_executor/model_runner.py (+16/-1); python/sglang/srt/state_capturer/routed_experts.py (+17/-1)
LABELS: deepseek, run-ci
DEEP_STUDY: deep-study correctness case sglang:e4bf0043fe: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Summary ⏎ Stop a speculative draft from polluting the target's routed-experts capture. ⏎  ⏎ ## Symptom & Reproduction ⏎ - **Symptom:** With `--enable-return-routed-experts` plus a MoE draft (NextN / MTP / EAGLE-MoE), returned `routed_experts` diverge from the target-only baseline. ⏎ - **Reproduction:** `python -m pytest test/registered/rl/test_return_routed_experts_mtp.py` (Qwen3-30B-A3B target + Qwen3-MoE EAGLE3 draft vs target-only baseline). ⏎  ⏎  …[truncated]

### L1-f82addd4a8  (L1, 2026-06-24, sha f82addd4a80a, PR #27939)
TITLE: Support online MXFP8 quantization for ungated MoE (#27939)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+7/-6); docs_new/docs/advanced_features/quantization.mdx (+3/-1); python/sglang/srt/layers/quantization/fp8.py (+6/-4); python/sglang/srt/layers/quantization/fp8_utils.py (+12/-10)
LABELS: documentation, quant, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ 1. Support online MXFP8 quantization for non-gated MoE (Nemotron) ⏎ 2. Switch defualt backend from Triton to CUTLASS GEMM (on the basis of perf) ⏎ 3. Use Cute-DSL quantizer on the basis of perf and only use 8x4 layout (bitwise exactness is covered by the next release) (https://github.com/flashinfer-ai/flashinfer/pull/3387) ⏎  ⏎ ## Modifications ⏎  ⏎ Profiles are for BS = 128, no DP attention ⏎  ⏎ ### BF16 ⏎  ⏎  ``` ⏎  python3 -m sglang.laun …[truncated]

### L1-8b7a1e908a  (L1, 2026-06-24, sha 8b7a1e908a74, PR #28347)
TITLE: Revert Gemma4 modelopt fp4 MoE backend change (#28347)
SOURCES: path_integration+keyword, subject_keyword, corpus:confirmed-reverts
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+3/-13)
LABELS: run-ci
DEEP_STUDY: deep-study revert record: explicit_rollback of PR(s)  reason=other
BODY: ## Summary ⏎  ⏎ Restore Gemma4 MoE NVFP4 SM10X default `trtllm_mha` and depends on #28144 ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #28086705575](https://github.com/sgl-project/sglang/actions/runs/28086705575) ⏎ Latest PR Test (Extra): :x: [Run #28086705345](https://github.com/sgl-project/sglang/actions/runs/28086705345)

### L1-9fd6d0ec99  (L1, 2026-06-24, sha 9fd6d0ec990f, PR #28953)
TITLE: [LoRA] BF16 support + EP cuda-graph crash fix for experimental_sgl_trtllm MoE-LoRA (#28953)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu (+7/-0); python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/trtllm_fused_moe_kernel_launcher.cu (+413/-0); python/sglang/jit_kernel/trtllm_lora_temp/__init__.py (+2/-0); python/sglang/jit_kernel/trtllm_lora_temp/core.py (+71/-0); python/sglang/srt/lora/trtllm_lora_temp/__init__.py (+20/-5); python/sglang/srt/lora/trtllm_lora_temp/environ.py (+8/-0); python/sglang/srt/lora/trtllm_lora_temp/lora_dispatch.py (+152/-0); python/sglang/srt/lora/trtllm_lora_temp/lora_layer.py (+37/-1); python/sglang/srt/lora/trtllm_lora_temp/moe_overlap.py (+210/-0); python/sglang/srt/lora/trtllm_lora_temp/triton_ops/virtual_experts.py (+26/-5)
LABELS: lora, run-ci, jit-kernel
BODY: ## Summary ⏎  ⏎ Two changes on the `experimental_sgl_trtllm` MoE-LoRA fast path: ⏎  ⏎ 1. **BF16 support** for the path (previously FP8 / NVFP4 only), plus `experts_shared_outer` LoRA. ⏎ 2. **Stability fixes** that make the path correct under expert parallelism (EP) + CUDA graph: ⏎    - **NVFP4 (Kimi-K2.5):** fix an illegal-memory-access that crashed batched serving under CUDA graph with the two-stream overlap enabled. ⏎    - **BF16:** two follow-up corr …[truncated]

### L1-de2d01c8da  (L1, 2026-06-25, sha de2d01c8da49, PR #28450)
TITLE: [AMD] Fuse shared-expert append + DeepEP remap into one Triton kernel (#28450)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py (+86/-0); python/sglang/srt/layers/moe/topk.py (+116/-5); test/registered/moe/test_fused_append_remap_deepep.py (+181/-0)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ On the AMD aiter / DeepEP-class MoE path, every layer runs the shared-expert ⏎ append (`fused_append_shared_experts`) immediately followed by the eager DeepEP ⏎ interleaved remap (`_remap_topk_for_deepep`). The remap is a sequence of small, ⏎ launch-latency-bound elementwise ops (floor-div, add, arange, fill, copy) that ⏎ each cost a kernel launch. With one shared-expert append + remap per MoE layer ⏎ across many layers, this launch o …[truncated]

### L1-ec12a28a87  (L1, 2026-06-25, sha ec12a28a871c, PR #28967)
TITLE: [AMD] Register 7 JIT kernel unit tests for AMD nightly CI (#28967)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/jit/minimax/test_minimax_decode_topk.py (+2/-1); test/registered/jit/minimax/test_minimax_store_kv_index.py (+2/-1); test/registered/jit/test_deepseek_v4_compress_state_runtime_shapes.py (+2/-1); test/registered/jit/test_fused_store_index_cache.py (+2/-1); test/registered/jit/test_fused_verify_triton_gdn.py (+2/-1); test/registered/jit/test_kvcacheio_asymmetric.py (+2/-1); test/registered/jit/test_mla_kv_pack_quantize_fp8.py (+2/-1)
LABELS: quant, lora, deepseek, hicache, run-ci
BODY: ## Summary ⏎  ⏎ Add `register_amd_ci(suite="nightly-amd-kernel-1-gpu", nightly=True)` to **7** JIT-kernel correctness tests (previously CUDA-only) that are **validated to pass on both ROCm 7.0 and ROCm 7.2** (MI35x / gfx950). ⏎  ⏎ **Batch 1 (3 tests):** ⏎ - `test_fused_store_index_cache` ⏎ - `test_mla_kv_pack_quantize_fp8` ⏎ - `test_kvcacheio_asymmetric` ⏎  ⏎ **Batch 2 (4 tests):** ⏎ - `test_fused_verify_triton_gdn` ⏎ - `minimax/test_minimax_store_kv_index` ⏎ - `minimax …[truncated]

### L1-3d3a7ec031  (L1, 2026-06-25, sha 3d3a7ec03159, PR #28237)
TITLE:  [AMD] fix(moe): correct fused shared-expert scaling on aiter/DeepEP path (mori all-to-all) (#28237)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+27/-3); test/registered/unit/layers/moe/test_fused_shared_expert_scaling.py (+117/-0)
LABELS: run-ci
DEEP_STUDY: deep-study correctness case sglang:3d3a7ec031: class=numerical_precision; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: # fix(moe): correct fused shared-expert scaling on aiter/DeepEP path ⏎  ⏎ ## Motivation ⏎  ⏎ On the HIP aiter + DeepEP-class MoE path (e.g. MoRI all-to-all), the fused shared ⏎ expert is under-weighted by `routed_scaling_factor`, corrupting every MoE layer ⏎ and producing degenerate, non-converging generation on long outputs. ⏎  ⏎ Two code paths scale the MoE output differently: ⏎  ⏎ - **Post-MoE scaling (default):** `DeepseekV2MoE.forward_deepep` multipli …[truncated]

### L1-67b2a9ed0c  (L1, 2026-06-25, sha 67b2a9ed0cfb, PR #29042)
TITLE: [NPU] Fix the DeepSeek-V2-Coder model accuracy issue (#29042)
SOURCES: path_core
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+2/-1); python/sglang/srt/models/deepseek_v2.py (+1/-0); python/sglang/srt/models/llada2.py (+1/-0)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ Fix the DeepSeek-V2-Coder model accuracy issue. ⏎  ⏎ ## Modifications ⏎  ⏎ 1. In `fused_topk_npu()`, determine whether `norm_type` is sigmoid or softmax based on the value of `topk_config.scoring_func`. ⏎ 2. In `DeepseekV2MoE::__init__()`, pass `config.scoring_func` into `topk_kwargs` so that `self.topk` correctly determines the scoring function based on its construction-time value. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ #!/bin/bash ⏎  ⏎ export H …[truncated]

### L1-e6efe10072  (L1, 2026-06-25, sha e6efe10072f8, PR #29197)
TITLE: [AMD] Register 5 JIT kernel unit tests for AMD nightly CI (#29197)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/jit/deepseek_v4/test_c128_v2.py (+2/-1); test/registered/jit/deepseek_v4/test_c4_v2.py (+2/-1); test/registered/jit/diffusion/test_group_norm_silu.py (+2/-1); test/registered/jit/diffusion/test_qwen_image_modulation.py (+2/-1); test/registered/jit/diffusion/test_varlen_pack_pad.py (+2/-1)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ Add `register_amd_ci(suite="nightly-amd-kernel-1-gpu", nightly=True)` to **5** JIT-kernel correctness tests (previously CUDA-only) that are **validated to pass on both ROCm 7.0 and ROCm 7.2** (MI35x / gfx950). This is the next batch following #28967. ⏎  ⏎ **Batch 3 (5 tests):** ⏎ - `deepseek_v4/test_c128_v2` ⏎ - `deepseek_v4/test_c4_v2` ⏎ - `diffusion/test_group_norm_silu` ⏎ - `diffusion/test_varlen_pack_pad` ⏎ - `diffusion/test_qwen_image_modulati …[truncated]

### L1-212c30d008  (L1, 2026-06-25, sha 212c30d00899, PR #28211)
TITLE: [MoE Refactor] Centralize FlashInfer CUTLASS MoE runner (#28211)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:production-kernel-provenance, body_keyword
ARTIFACT_HINTS: L1.runner.framework, L1.runner.flashinfer_mxfp4, L1.runner.flashinfer_cutlass
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py (+372/-0); python/sglang/srt/layers/moe/moe_runner/flashinfer_mxfp4.py (+0/-174); python/sglang/srt/layers/moe/moe_runner/runner.py (+2/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+46/-158); python/sglang/srt/layers/quantization/mxfp4.py (+5/-5); python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py (+4/-4); python/sglang/srt/layers/quantization/unquant.py (+19/-38); test/registered/unit/layers/quantization/test_mxfp4_sm90_cutlass.py (+9/-9)
LABELS: quant, run-ci
BODY: ## Motivation ⏎  ⏎ Part of #8715. ⏎  ⏎ CC: @ch-wan ⏎  ⏎ This PR continues the Stage 3 MoE refactor adoption work by moving FlashInfer CUTLASS MoE execution out of quantization `apply()` paths and into a dedicated `MoeRunner` backend file. ⏎  ⏎ Previous work in #26489 migrated the SM90 CUTLASS W4A16 MXFP4 path to `MoeRunner` through `flashinfer_mxfp4.py`. Since that path also calls FlashInfer CUTLASS, this PR folds it into `flashinfer_cutlass.py` and remo …[truncated]

### L1-30ea4c0f4b  (L1, 2026-06-25, sha 30ea4c0f4ba3, PR #29067)
TITLE: build(sgl-kernel): bump FlashMLA pin + fix cccl include for CUDA 13 (#29067)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/cmake/flashmla.cmake (+14/-2)
LABELS: sgl-kernel, run-ci
BODY: ## Summary ⏎  ⏎ Bumps the vendored FlashMLA pin in `sgl-kernel/cmake/flashmla.cmake` to pick up sm90 dense decode `HEAD_DIM_K=512` support, and adds the CUDA 13 `cccl` include-path fix that the pin bump requires. ⏎  ⏎ Two changes, both in `sgl-kernel/cmake/flashmla.cmake`: ⏎  ⏎ 1. **FlashMLA pin bump** `df022eba` → `05e26647` — pulls in sgl-project/FlashMLA#9 (`sm90 dense decode: support HEAD_DIM_K=512 (zero RoPE tail)`, merged into the `rebase` branch), whi …[truncated]

### L1-999199f9ff  (L1, 2026-06-26, sha 999199f9ff4b, PR #29142)
TITLE: [DeepSeek V3] Run routed experts on main stream in dual-stream MoE (#29142)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+48/-46)
LABELS: high priority, deepseek, run-ci, bypass-fastfail
DEEP_STUDY: deep-study: this PR was reverted by PR 29452 (confirmed_revert, reason=unstated) || deep-study: this PR was reverted by PR 29463 (reland, reason=correctness_or_accuracy)
BODY: ## Motivation ⏎  ⏎ Follow-up to #27720. That PR was intended to swap the dual-stream execution order so the routed experts run on the main stream (with the deferred MoE finalize fused into the main-stream residual add), but only the deferred-finalize half landed — `forward_normal_dual_stream` was left with shared experts on the main stream and the routed experts (gate / topk / experts) on the alt stream. ⏎  ⏎ This is the opposite of the tokenspeed ex …[truncated]

### L1-4ce1c180bd  (L1, 2026-06-26, sha 4ce1c180bde5, PR #29270)
TITLE: [sgl-kernel/cpu]: fix arm64 w8a8 moe kernel signature (#29270)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/aarch64/moe.cpp (+4/-0); sgl-kernel/csrc/cpu/torch_extension_cpu.cpp (+4/-27); test/registered/cpu/arm64/test_moe.py (+4/-0)
LABELS: sgl-kernel, run-ci
BODY: ## Motivation ⏎  ⏎ `fused_experts_cpu` signature has changed, arm64 implementation needs update. ⏎  ⏎ ## Modifications ⏎  ⏎ Update arm64 implementation of `fused_experts_cpu`. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ N/A ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎ N/A ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-pr …[truncated]

### L1-7b02eab7a6  (L1, 2026-06-26, sha 7b02eab7a688, PR #29452)
TITLE: Revert "[DeepSeek V3] Run routed experts on main stream in dual-stream MoE" (#29452)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+46/-48)
LABELS: deepseek
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 29142 reason=unstated
BODY: Reverts sgl-project/sglang#29142 ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #28264478296](https://github.com/sgl-project/sglang/actions/runs/28264478296) ⏎ Latest PR Test (Extra): :x: [Run #28264478219](https://github.com/sgl-project/sglang/actions/runs/28264478219)

### L1-5eebb4b2e8  (L1, 2026-06-26, sha 5eebb4b2e820, PR #29377)
TITLE: [AMD] Fix fused append+remap DeepEP equivalence test on aiter path (#29377)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/moe/test_fused_append_remap_deepep.py (+6/-5)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `test/registered/moe/test_fused_append_remap_deepep.py::test_equivalence_with_eager_append_then_remap` ⏎ fails on AMD nightly CI: ⏎ ``` ⏎ AssertionError: False is not true # torch.allclose(fused_w, eager_w) (m=1, k=8, npr=256, ep_rank=0, s=1) ⏎ ``` ⏎ The test was added in #28450 (de2d01c8) and assumed `_remap_topk_for_deepep()` ⏎ always overwrites the fused shared-expert weight with `1/routed_scaling_factor`. ⏎ PR #28237 (3d3a7ec0) late …[truncated]

### L1-43435a2f8e  (L1, 2026-06-26, sha 43435a2f8e34, PR #29097)
TITLE: [mori] Add a combine-kwargs hook and use_external_inp_buf plumbing (#29097)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+14/-2)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ Make the MoRI EP combine path extensible: ⏎  ⏎ - Add a `_combine_kwargs(hidden_states)` hook on `_MoriEPDispatcherImplBase` ⏎   (default returns `{}`), passed through to `mori_op.combine` / `combine_send`, ⏎   so subclasses can supply extra combine arguments (e.g. an external output ⏎   buffer for a zero-copy combine). ⏎ - Plumb `use_external_inp_buf` (default `True`) through `init_mori_op` into the ⏎   mori `EpDispatchCombineConfig`, and include i …[truncated]

### L1-828411e6f1  (L1, 2026-06-28, sha 828411e6f1ad, PR #29461)
TITLE: Fix FlashInfer A2A dispatcher during CUDA graph capture (#29461)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+36/-10)
LABELS: run-ci
BODY: ## Summary ⏎ - Skip the EP>1 runtime token count query branch during CUDA graph capture in the FlashInfer MoE A2A dispatcher. ⏎ - The captured graph uses fixed geometry, so querying runtime token counts via NCCL all-gather during capture is incorrect. ⏎ - Gate the branch on `not is_graph_capture` (combining `get_is_capture_mode()` and `is_in_breakable_cuda_graph()`). ⏎  ⏎ ## Original commits ⏎ - `d3abe7396` ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_c …[truncated]

### L1-38d4ffcd86  (L1, 2026-06-29, sha 38d4ffcd863b, PR #28731)
TITLE: [cookbook] drop redundant serve flags (GLM-5.2) + fix M3 page-size note (#28731)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx (+4/-0); docs_new/cookbook/autoregressive/MiniMax/MiniMax-M3.mdx (+2/-2); docs_new/src/snippets/configs/MiniMaxAI/minimax-m3.jsx (+0/-4); docs_new/src/snippets/configs/zai-org/glm-5.2-benchmarks.jsx (+50/-157); docs_new/src/snippets/configs/zai-org/glm-5.2.jsx (+8/-30)
LABELS: documentation, run-ci, run-ci-extra
BODY: Per Lianmin's feedback (2026-06-19): make the default arguments just work — drop hand-set serve flags where the runtime default is correct or better, instead of overriding in the cookbook. ⏎  ⏎ ## GLM-5.2 (`zai-org/glm-5.2.jsx`) ⏎ - `--mem-fraction-static`: keep it explicit (0.85 / 0.8 / 0.9). Earlier in this PR the flags were dropped to take the runtime auto-mfs, but verification showed the explicit values are the safer choice across the H200/B200/B30 …[truncated]

### L1-6bdecb8206  (L1, 2026-06-29, sha 6bdecb8206b8, PR #29463)
TITLE: [DeepSeek V3] Reland: run routed experts on main stream in dual-stream MoE (#29463)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+49/-46)
LABELS: deepseek
DEEP_STUDY: deep-study revert record: reland of PR(s) 29142 reason=correctness_or_accuracy
BODY: ## Motivation ⏎  ⏎ Reland of #29142, which was reverted in #29452 because it dropped `test_moe_ep_extra.py`'s gsm8k from ~0.7 to **0.005**. ⏎  ⏎ ## Root cause ⏎  ⏎ Not a cross-stream race. The hazard is a **host-side mutation of the input tensor's metadata** between two CUDA kernel launches inside the captured decode graph. ⏎  ⏎ `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:567-606`, `pre_permute_standard_to_deep_gemm` (called as part of the routed `self …[truncated]

### L1-b8c25bfaa7  (L1, 2026-06-29, sha b8c25bfaa792, PR #22394)
TITLE: [NVIDIA] Support flashinfer a2a with flashinfer_trtllm_routed moe (#22394)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+67/-4); python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+13/-29); docs_new/docs/advanced_features/expert_parallelism.mdx (+3/-3); python/sglang/srt/server_args.py (+10/-5); test/registered/ep/test_flashinfer_a2a.py (+125/-0)
LABELS: documentation, run-ci, release-highlight
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Flashinfer a2a can now be used in combination with flashinfer_trtllm_routed moe. ⏎  ⏎ ## Modifications ⏎  ⏎ * Adds "fused_func" for flashinfer dispatcher -> flashinfer_trtllm_routed moe runner ⏎ * Supports fp4 quantize before comm ⏎ * In Flashinfer dispatcher, remove dummy token workaround for zero local tokens - flashinfer a2a now supports this. ⏎ * Update server arg validation ⏎ * Add FP4 and FP8 e2e test ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ FP8 ⏎ ` …[truncated]

### L1-d5133e925b  (L1, 2026-06-29, sha d5133e925bbd, PR #29493)
TITLE: [NPU][Bugfix] Add scoring_func for mimo_v2 (#29493)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/mimo_v2.py (+1/-0)
LABELS: run-ci
BODY: ## Motivation ⏎ PR #29042 updated the NPU topk kernel to respect topk_config.scoring_func for norm_type selection. However, mimo_v2.py did not pass scoring_func to TopK , so it defaulted to "softmax" at the model level. For MiMo-V2.5 models configured with scoring_func="sigmoid" , this caused the NPU kernel to use incorrect softmax normalization. ⏎  ⏎ ## Modifications ⏎ - python/sglang/srt/models/mimo_v2.py : Pass scoring_func=config.scoring_func to  …[truncated]

### L1-a6bc432fd8  (L1, 2026-06-29, sha a6bc432fd8dd, PR #18612)
TITLE: [Perf][Kernel] Fuse SiLU+Mul into NVFP4 Expert Quantization for CUTLASS MoE (#18612)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.framework, L1.cutlass.adapters
FILES: python/sglang/srt/layers/moe/cutlass_moe.py (+5/-8); python/sglang/srt/layers/moe/moe_runner/runner.py (+2/-0); python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_expert_quant.cuh (+107/-13); python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_quant_entry.cuh (+19/-0); python/sglang/jit_kernel/nvfp4.py (+93/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+5/-1); test/registered/jit/test_silu_and_mul_scaled_fp4_experts_quant_packed.py (+337/-0)
LABELS: quant, sgl-kernel, blackwell, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎ In the CUTLASS FP4 MoE pipeline, the path between GEMM1 and GEMM2 previously required **3 separate steps**: allocate intermediate buffer → `silu_and_mul` → `scaled_fp4_experts_quant`. This PR fuses them into a **single CUDA kernel** `silu_and_mul_scaled_fp4_experts_quant_packed`, eliminating one intermediate buffer allocation and one extra kernel launch. Inspired by vllm [#31832](https://github.com/vllm-project/vllm/pull/31832) ⏎  ⏎  ⏎ # …[truncated]

### L1-a5e6dd3767  (L1, 2026-06-30, sha a5e6dd37677f, PR #27204)
TITLE: [AMD] Implement QuarkW4A8MXFp4MoE to support amd/gpt-oss-120b-w-mxfp4-a-fp8 (#27204)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/quark/quark.py (+26/-0); python/sglang/srt/layers/quantization/quark/schemes/__init__.py (+2/-0); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a8_mxfp4_moe.py (+407/-0); python/sglang/srt/layers/quantization/quark/weights.py (+248/-0); python/sglang/srt/models/gpt_oss.py (+14/-3); test/registered/amd/accuracy/mi35x/test_gpt_oss_w4a8_mxfp4_eval_mi35x.py (+251/-0)
LABELS: amd, run-ci
BODY: This PR extends the Quark quantization scheme to support W4A8 MXFP4-FP8 path so SGLang can load and run AMD Quark per-expert MoE checkpoints through the AITER fused-MoE backend. The main motivation is enabling support for amd/gpt-oss-120b-w-mxfp4-a-fp8 (checkpoint carries fp8 scaling for activation - pre-calib). ⏎  ⏎   The change mirrors the existing native MXFP4 GPT-OSS loading flow where possible, while handling the Quark checkpoint layout differ …[truncated]

### L1-89620b9169  (L1, 2026-06-30, sha 89620b9169e6, PR #28980)
TITLE: [NPU] Support DeepSeek V4 Flash MTP on Ascend (#28980)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+1/-8); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+3/-1); python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py (+589/-26); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py (+153/-29); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_common_hooks.py (+53/-5); python/sglang/srt/model_executor/forward_batch_info.py (+1/-0); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+1/-1); python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py (+2/-1); python/sglang/srt/models/deepseek_v4.py (+4/-4); python/sglang/srt/models/deepseek_v4_nextn.py (+6/-3); (+3 more)
LABELS: deepseek, npu, run-ci
BODY: ## Motivation ⏎  ⏎ Enable DeepSeek V4 Flash ModelSlim NEXTN/MTP inference on Ascend NPU. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add DSV4 compressed KV/state allocation and metadata handling for MTP target verify and graph replay. ⏎ - Route DeepSeek V4 speculative draft backends through the Ascend DSV4 backend and adapt the current spec-v2/Frozen-KV flow. ⏎ - Improve ModelSlim/compressed-tensors weight mapping and NEXTN weight loading. ⏎  ⏎ ## Accuracy Tests ⏎ <img wi …[truncated]

### L1-56f22cd520  (L1, 2026-06-30, sha 56f22cd520cb, PR #28612)
TITLE: Optimize C128 state pool allocation using request state pool (#28612)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/csrc/deepseek_v4/c128_online_v2.cuh (+2/-30); python/sglang/jit_kernel/csrc/deepseek_v4/c_plan.cuh (+53/-28); python/sglang/jit_kernel/csrc/deepseek_v4/online_c128_mtp.cuh (+35/-185); python/sglang/jit_kernel/dsv4/__init__.py (+2/-0); python/sglang/jit_kernel/dsv4/attn.py (+15/-12); python/sglang/jit_kernel/dsv4/c128_cleanup.py (+58/-0); python/sglang/jit_kernel/dsv4/compress.py (+4/-13); python/sglang/jit_kernel/dsv4/online_c128_mtp.py (+15/-11); python/sglang/jit_kernel/tests/deepseek_v4/common.py (+2/-2); python/sglang/srt/disaggregation/base/conn.py (+2/-0); (+22 more)
LABELS: high priority, deepseek, run-ci, jit-kernel, run-ci-extra, release-highlight
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This PR optimizes the C128 state slot lookup path. ⏎  ⏎ The issue was introduced when the online C128/MTP path started deriving C128 state slots through SWA mapping. Before this change, `full_to_swa_index_mapping` was only a temporary translation table for SWA attention KV slots. The original radix/SWA cache lifecycle allows radix cache to keep the full KV prefix alive while the SWA sidecar KV for the same prefix may be tombston …[truncated]

### L1-c6a7c98ae4  (L1, 2026-06-30, sha c6a7c98ae429, PR #29509)
TITLE: [NPU]GLM-4.7-Flash optimize with fused kernels (#29509)
SOURCES: path_core
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+1/-1); python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py (+36/-14)
LABELS: deepseek, npu, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Introduce a fused Triton kernel to improve model performance. ⏎  ⏎ ## Modifications ⏎  ⏎ Replace the original split + RMSNorm pipeline with a fused Triton kernel. ⏎  ⏎  ⏎ ## Accuracy Tests ⏎ Before: ⏎ <img width="667" height="86" alt="image" src="https://github.com/user-attachments/assets/68ae017c-fc6a-4484-9178-72478019a71b" /> ⏎ After: ⏎ <img width="662" height="85" alt="image" src="https://github.com/user-attachments/assets/3ab4c852-71fe …[truncated]

### L1-cf36dca6d4  (L1, 2026-06-30, sha cf36dca6d4ca, PR #25835)
TITLE: [JIT Kernel] Triton moe fused gate (#25835)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.fused_gate
FILES: python/sglang/jit_kernel/csrc/moe/moe_fused_gate.cuh (+6/-2); python/sglang/jit_kernel/moe_fused_gate.py (+171/-2); python/sglang/srt/layers/moe/topk.py (+12/-11); test/registered/jit/benchmark/bench_moe_fused_gate.py (+98/-0); test/registered/jit/test_moe_fused_gate.py (+258/-0)
LABELS: run-ci, jit-kernel, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎ [details omitted] ⏎  ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) …[truncated]

### L1-8ee200972e  (L1, 2026-07-01, sha 8ee200972ed1, PR #26255)
TITLE: [fix] Add support for flashinfer MOE A2A to Qwen3 BF16 model path (#26255)
SOURCES: path_core, subject_keyword, symbol_pickaxe, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.routing.topk_softmax
FILES: python/sglang/srt/layers/moe/utils.py (+8/-0); sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu (+11/-0); test/registered/moe/test_flashinfer_a2a_cutlass.py (+93/-0)
LABELS: quant, sgl-kernel, run-ci
DEEP_STUDY: deep-study correctness case sglang:8ee200972e: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Motivation ⏎  ⏎  ⏎ BF16 + DP attn + EP moe + flashinfer A2A + flashinfer MOE cutlass backend is currently not supported in sglang. This MR enables support for this and resolves some related issues that caused crashes ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ 1. Handle non-existent `self.quant_config` in FlashinferDispatcher ⏎ 2. Use communication buffer as moe output in `flashinfer_cutlass_fused_moe` ⏎ 3. Skip duplicated allreduce (relevant for TP attn + EP moe …[truncated]

### L1-a7390b17f8  (L1, 2026-07-01, sha a7390b17f813, PR #29793)
TITLE: [NPU]Modify --lora-backend & --moe-runner-backend description. (#29793)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx (+3/-3)
LABELS: documentation, npu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Update ascend_npu_support_features.md: refine --lora-backend / --moe-runner-backend document. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWN …[truncated]

### L1-779ea4a9b5  (L1, 2026-07-01, sha 779ea4a9b5f4, PR #28676)
TITLE: [RL] fix deepseek v4 MXFP8 flashinfer_trtllm_routed MoE weight update (#28676)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+20/-0); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+8/-0); test/registered/rl/test_update_weights_from_disk_blackwell.py (+1/-1)
BODY: ## Motivation ⏎  ⏎ DeepSeek-V4 (MXFP8) on the `flashinfer_trtllm_routed` MoE path breaks after the ⏎ first RL weight update: `train_rollout_logprob_abs_diff` jumps from ~0.06 to ⏎ **~3.83**. Steady-state is fine — the bug is specific to the weight-reload path. ⏎  ⏎ ## Root cause ⏎  ⏎ `align_mxfp8_moe_weights_for_flashinfer_trtllm` shuffles MoE weights/scales into ⏎ the kernel layout using row-permutation index tensors that depend only on shape, ⏎ so they'r …[truncated]

### L1-c312cdd3a7  (L1, 2026-07-01, sha c312cdd3a7db, PR #29554)
TITLE: Upgrading tvm-ffi/sgl-deep-gemm/tilelang (#29554)
SOURCES: path_core, dependency_pin
ARTIFACT_HINTS: L1.runner.deepgemm_megamoe, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+3/-3); python/sglang/srt/layers/moe/mega_moe.py (+36/-4); python/sglang/srt/layers/attention/dsa/tilelang_kernel.py (+4/-20); python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py (+2/-2); python/sglang/srt/layers/mhc.py (+0/-2)
LABELS: dependencies, deepseek, run-ci, run-ci-extra, release-highlight
BODY: co-authored by @MartinHua  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODE …[truncated]

### L1-8f0d320d31  (L1, 2026-07-01, sha 8f0d320d3162, PR #29595)
TITLE: [Spec] Enable FlashInfer autotune for spec draft (#29595)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/runner/base_runner.py (+11/-153); python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py (+15/-5); python/sglang/srt/model_executor/runner/flashinfer_autotune.py (+224/-0); python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py (+13/-3); python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py (+15/-5); python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py (+13/-3); python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py (+13/-3); python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py (+1/-0); python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py (+1/-0); python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py (+1/-0); (+6 more)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Summary ⏎  ⏎ Enable FlashInfer autotuning for speculative decoding draft graph paths ⏎  ⏎ ## Validation ⏎  ⏎ Checked on GB300 with `auto` picking `flashinfer_trtllm` for both target and draft MoE runner backends: ⏎  ⏎ ```bash ⏎ sglang serve \ ⏎   --model-path nvidia/GLM-5.2-NVFP4 \ ⏎   --tensor-parallel-size 4 \ ⏎   --speculative-algorithm EAGLE \ ⏎   --speculative-num-steps 5 \ ⏎   --speculative-eagle-topk 1 \ ⏎   --speculative-num-draft-tokens 6 ⏎ ``` ⏎  ⏎ In …[truncated]

### L1-df0dfbaa45  (L1, 2026-07-01, sha df0dfbaa4529, PR #29636)
TITLE: [Kernel] Strengthen kernel shape coverage (#29636)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: sgl-kernel/tests/test_dsv3_fused_a_gemm.py (+1/-1); sgl-kernel/tests/test_fp8_gemm.py (+15/-0); sgl-kernel/tests/test_norm.py (+58/-0); sgl-kernel/tests/test_per_token_quant_fp8.py (+11/-5); test/registered/jit/benchmark/bench_sparse_mla_q8kv8_prefill_sm90.py (+10/-4); test/registered/jit/test_activation.py (+6/-2); test/registered/jit/test_cutedsl_dsv3_fused_a_gemm.py (+8/-3); test/registered/jit/test_dsv3_fused_a_gemm.py (+3/-2); test/registered/jit/test_dsv3_router_gemm.py (+23/-5); test/registered/jit/test_fused_add_rmsnorm.py (+14/-5); (+5 more)
LABELS: quant, sgl-kernel, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Summary ⏎ - Replace several broad per-commit JIT kernel test grids with small frozen representative shape tables while keeping broader local/nightly coverage through `get_ci_test_range`. ⏎ - Add representative production-like shapes for RMSNorm, fused-add RMSNorm, activation, FP8 quantization, per-token group quantization, router GEMM, fused-A GEMM, and MoE align. ⏎ - Add dedicated `sgl-kernel` correctness cases for FP8 scaled-mm, per-token FP8 quan …[truncated]

### L1-0543246184  (L1, 2026-07-01, sha 0543246184b2, PR #29458)
TITLE: Enable Breakable Cuda Graph as Default (#29458)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/cuda_graph_config.py (+10/-1); python/sglang/srt/model_executor/runner/__init__.py (+1/-1); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+51/-15); python/sglang/srt/model_executor/runner_backend_utils/__init__.py (+3/-5); python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/context.py (+26/-0); python/sglang/srt/model_executor/runner_backend_utils/tc_piecewise_cuda_graph/__init__.py (+3/-2); python/sglang/srt/model_executor/runner_backend_utils/tc_piecewise_cuda_graph/context_manager.py (+19/-4); python/sglang/srt/server_args.py (+19/-3); test/registered/quant/test_fp8_blockwise_gemm.py (+3/-0); test/registered/quant/test_nvfp4_gemm.py (+2/-1)
LABELS: high priority, blackwell, run-ci, bypass-fastfail, run-ci-extra, release-highlight
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-13dc5f2dc7  (L1, 2026-07-01, sha 13dc5f2dc789, PR #25377)
TITLE: [HiCache][AMD] Add UMBP tiered DRAM + SSD L3 storage backend with hugepage host allocator   (#25377)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/cache_controller.py (+1/-1); python/sglang/srt/mem_cache/pool_host/common.py (+14/-0); python/sglang/srt/mem_cache/storage/backend_factory.py (+8/-0); python/sglang/srt/mem_cache/storage/umbp/__init__.py (+0/-0); python/sglang/srt/mem_cache/storage/umbp/umbp_host_allocator.py (+142/-0); python/sglang/srt/mem_cache/storage/umbp/umbp_store.py (+1444/-0); python/sglang/srt/server_args.py (+1/-0); test/registered/unit/mem_cache/test_umbp_host_allocator.py (+202/-0); test/registered/unit/mem_cache/test_umbp_store.py (+294/-0)
LABELS: amd, run-ci, run-ci-extra
BODY: ## Summary ⏎  ⏎ Adds a new HiCache storage backend, `mori`, that offloads evicted KV pages to a host-side L3 tier managed by the [`mori`](https://github.com/ROCm/mori) library's `umbp` host-memory pool subsystem. (`mori` is AMD's EP / IO backend for MoE serving on MI3xx; `umbp` is its tiered host-memory pool component.) ⏎  ⏎ This PR includes: ⏎  ⏎ - `UMBPStore` as a zero-copy HiCache storage backend. ⏎ - A hugepage-capable host allocator backed by mori' …[truncated]

### L1-bac351d617  (L1, 2026-07-02, sha bac351d6174e, PR #29829)
TITLE: [NPU] Fix block_table batch size mismatch in GLM-4.7-Flash DeepEP + MTP without CUDA Graphs (#29829)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+12/-2)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ When CUDA Graphs are disabled, enabling DeepEP together with MTP causes a batch size mismatch between the block table and the query. ⏎  ⏎ ## Modifications ⏎  ⏎ In forward_mtp, when CUDA Graphs are disabled, slice the block table and actual_seq_lengths_kv to the actual batch size so that their first dimension matches the query batch size. This prevents batch size mismatches when DeepEP is used together with MTP. ⏎  ⏎ ## Accuracy Tests ⏎  …[truncated]

### L1-cb06c4e6ce  (L1, 2026-07-02, sha cb06c4e6ce40, PR #29497)
TITLE: [CPU] Fix model failures on Xeon (#29497)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docker/xeon.Dockerfile (+1/-2); docs_new/docs/hardware-platforms/cpu_server.mdx (+18/-69); python/pyproject_cpu.toml (+2/-1); python/sglang/srt/layers/quantization/mxfp4.py (+58/-25); python/sglang/srt/model_executor/cuda_graph_buffer_registry.py (+4/-1); python/sglang/srt/model_executor/runner/eager_runner.py (+3/-0)
LABELS: documentation, dependencies, intel, cpu, run-ci
BODY: ## Motivation ⏎  ⏎ Fix the model launching failures on Xeon CPU. ⏎  ⏎ ## Modifications ⏎  ⏎ - Fix Qwen3.5/3.6 series: #26924 introduced `fused_sigmoid_mul` triton kernel, but it is not applicable on CPU. Added a gating for this. ⏎ (Update: the gating is deprecated and reverted as it is not needed as CPU `fused_sigmoid_mul` is implemented in #29378 ) ⏎  ⏎ - Fix Llama-3.2-11B-Vision: Fixed the seq_len dtype mismatch of encoder introduced in #27407.The input …[truncated]

### L1-f3904f0293  (L1, 2026-07-02, sha f3904f029341, PR #27835)
TITLE: [bugfix][AMD] Disable aiter allreduce+RMSNorm fusion under DP attention / EP (#27835)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/communicator.py (+2/-0); test/registered/ops/test_aiter_allreduce_fusion_amd.py (+131/-0)
LABELS: amd, deepseek, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ --enable-aiter-allreduce-fusion (the ROCm/aiter fused all-reduce + RMSNorm path) is only meaningful for the dense tensor-parallel path. The attention-side gate (apply_aiter_all_reduce_fusion) already excludes DP attention, but the MLP-side gate in LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer was missing the matching guards. ⏎  ⏎ As a result, enabling the flag together with DP attention and an expert-parallel A2A back …[truncated]

### L1-b276a9acee  (L1, 2026-07-02, sha b276a9acee8a, PR #29770)
TITLE: chore: cleanup garbage code (#29770)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py, L1.runner.triton
FILES: python/sglang/srt/layers/moe/moe_runner/triton.py (+2/-4); python/sglang/srt/layers/moe/topk.py (+1/-1); benchmark/bench_linear_attention/bench_gdn_decode.py (+2/-2); benchmark/mmmu/bench_hf.py (+1/-1); experimental/sgl-router/BENCHMARKS.md (+2/-2); experimental/sgl-router/src/workers/manager.rs (+1/-1); experimental/sgl-router/tests/e2e/chat_completions/test_validation.py (+1/-1); experimental/sgl-router/tests/e2e/conftest.py (+2/-2); experimental/sgl-router/tests/e2e/pyproject.toml (+1/-1); experimental/sgl-router/tests/e2e/requirements.txt (+1/-1); (+45 more)
LABELS: documentation, quant, amd, dependencies, Multi-modal, sgl-kernel, npu, run-ci, diffusion, model-gateway
BODY: ## Summary ⏎ - Remove hard-disabled/dead cleanup paths in DSA index buffer writes, EPLB detail accumulation, Hunyuan forward state plumbing, multimodal hashing, playground router code, and small benchmark/test helpers. ⏎ - Replace low-signal AI-ish wording such as `smoke`/`placeholder`/`dead code` in comments, docstrings, logs, and docs where it is not an API/file/test selector. ⏎ - Apply focused unused-import/unused-local cleanup from ruff on touched  …[truncated]

### L1-860244d4b4  (L1, 2026-07-02, sha 860244d4b4d8, PR #29447)
TITLE: [CI] Add per-stage NVIDIA model inventory tool (#29447)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/ci-model-inventory.yml (+68/-0); scripts/ci/list_stage_models.py (+688/-0); scripts/ci/stage_models_overrides.json (+31/-0); scripts/ci/test_list_stage_models.py (+602/-0)
BODY: ## Motivation ⏎  ⏎ There is currently no authoritative, machine-readable list of which models each NVIDIA (CUDA) CI suite exercises. This makes it hard to pre-warm runner model caches: a model that a suite pulls lazily at test time can cause a cold HuggingFace download (and the occasional rate-limit/timeout) far from where it'd be easy to diagnose. ⏎  ⏎ This PR adds a small, dependency-free tool that statically derives `suite -> [HuggingFace model ids]`  …[truncated]

### L1-bdd3515389  (L1, 2026-07-02, sha bdd35153898b, PR #29503)
TITLE: NPU case rl update weights for tensor load_format == None and flatten bucket (#29503)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+0/-9); python/sglang/srt/layers/quantization/unquant.py (+3/-7)
LABELS: quant, run-ci
BODY: ## Motivation ⏎ For rl demands, with regard to tensor load_format == None and flatten bucket, add post process for npu update weights to keep consistent with the initial weight loading through UnquantizedFusedMoEMethod process_weights_after_loading. ⏎  ⏎  ⏎ ## Modifications ⏎ as follows ⏎  ⏎ I changed caz I consulted the GMM interface colleague, and they said that no additional transpose aclnn ops would be introduced and it would not affect performance, …[truncated]

### L1-c05c48b35e  (L1, 2026-07-02, sha c05c48b35e8a, PR #29853)
TITLE: bugfix for npu Grok2 model --detokenizer without all special ids (#29853)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/utils/patch_tokenizer.py (+4/-2)
LABELS: run-ci
BODY: ## Motivation ⏎ bugfix for error #25309 that 'TiktokenTokenizer' object has no attribute 'all_special_ids' ⏎ [2026-05-29 12:21:03] DetokenizerManager hit an exception: Traceback (most recent call last): ⏎   File "/sgl-workspace/sglang/python/sglang/srt/managers/detokenizer_manager.py", line 440, in run_detokenizer_process ⏎     manager.event_loop() ⏎   File "/sgl-workspace/sglang/python/sglang/srt/managers/detokenizer_manager.py", line 150, in event_l …[truncated]

### L1-a6ee64d237  (L1, 2026-07-02, sha a6ee64d237a2, PR #29619)
TITLE: [DeepSeek-V4] Add an opt-in non-paged indexer for long-context prefill (#29619)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+1/-0); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+11/-2); python/sglang/srt/layers/attention/dsv4/indexer.py (+211/-41); python/sglang/srt/layers/attention/dsv4/metadata.py (+20/-3); python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py (+17/-4); test/registered/unit/layers/test_dsv4_nonpaged_indexer.py (+208/-0)
LABELS: deepseek, run-ci, release-highlight
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ DeepSeek-V4's C4 indexer currently evaluates logits directly from paged KV storage. For long-context, unpadded single-request `EXTEND` batches, gathering the C4 index keys and scales into contiguous storage enables the faster non-paged DeepGEMM `fp8_mqa_logits` path while preserving the existing top-k semantics. ⏎  ⏎ This PR adds that path behind a default-off environment flag. ⏎  ⏎ Related to #23602. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add a d …[truncated]

### L1-e81f05cf4f  (L1, 2026-07-02, sha e81f05cf4f44, PR #29988)
TITLE: [dsv4] Trigger MHC prenorm prewarm at weight-load time with rank sync (#29988)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/mhc.py (+4/-64); python/sglang/srt/models/deepseek_v4.py (+64/-115); test/registered/kernels/test_mhc_kernels.py (+0/-1)
LABELS: deepseek
BODY: ## Motivation ⏎  ⏎ #27986 made the DSV4 MHC prenorm prewarm trigger lazily inside the first forward that carries tokens — per rank, uncoordinated. On wide-EP disaggregated prefill this is fatal: 0-token EP peers still launch `deep_gemm.fp8_fp4_mega_moe` every layer and wait for the compiling rank inside the kernel's NVLink barrier, which device-traps after 180 s. The 23-bucket burst (one DeepGEMM `tf32_hc_prenorm_gemm` + one TileLang big-fuse variant …[truncated]

### L1-d364cd8ead  (L1, 2026-07-02, sha d364cd8ead47, PR #27349)
TITLE: Support DSV4 shared expert fusion for DeepEP and MegaMOE (#27349)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.routing.hash_topk
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+112/-6); python/sglang/srt/layers/moe/hash_topk.py (+53/-8); python/sglang/srt/layers/moe/topk.py (+27/-21); python/sglang/srt/layers/moe/utils.py (+10/-0); python/sglang/srt/layers/quantization/fp8_utils.py (+36/-0); python/sglang/srt/models/deepseek_v2.py (+13/-22); python/sglang/srt/models/deepseek_v4.py (+10/-13); test/registered/moe/test_fused_append_remap_per_rank_shared_slots.py (+12/-7); test/registered/moe/test_hash_topk.py (+145/-0); test/registered/unit/eplb/test_deepep_waterfill_eplb.py (+8/-4); (+3 more)
LABELS: documentation, deepseek, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎ - support DeepSeek V4 shared expert fusion with per-rank shared expert slots for both DeepEP-class and MegaMOE backends ⏎ - extend TopK/HashTopK post-processing and fused MoE weight loading to handle the shared expert physical slot layout ⏎ - add shared expert FP8-to-FP4 load-time quantization for fused FP4 MoE weights ⏎  ⏎ ## Scope ⏎ This PR focuses on DeepSeek V4 shared expert fusion compatibility for DeepEP and MegaMOE backends, including the …[truncated]

### L1-70a813493f  (L1, 2026-07-03, sha 70a813493fd4, PR #29937)
TITLE: [NPU] [DOC] add missing DEEP_NORMAL_MODE_USE_INT8_QUANT for w8a8+deepep scenarios (#29937)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_best_practice.mdx (+0/-6929); docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_environment_variables.mdx (+12/-0); docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_glm5.2_examples.mdx (+2/-0); docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_optimization.mdx (+4/-0); docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx (+1/-1); docs_new/docs/hardware-platforms/ascend-npus/model-tutorials/qwen3_235b_a22b.mdx (+1/-0)
LABELS: documentation, deepseek, npu
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ 1. delete old best practice ⏎ 2. add missing DEEP_NORMAL_MODE_USE_INT8_QUANT for w8a8+deepep scenarios ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ N/A ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ N/A ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ N/A ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). …[truncated]

### L1-a2d7eb303e  (L1, 2026-07-03, sha a2d7eb303eb8, PR #29771)
TITLE: [MoE] Consolidate ungrouped + grouped gate/topk onto one Triton router (#26771) — faster than AOT on B200/H100/H200, at parity with flashinfer (#29771)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.fused_gate
FILES: python/sglang/jit_kernel/csrc/moe/grouped_topk.cuh (+0/-260); python/sglang/jit_kernel/grouped_topk.py (+0/-89); python/sglang/jit_kernel/moe_fused_gate.py (+137/-45); python/sglang/srt/layers/moe/topk.py (+79/-13); python/sglang/srt/environ.py (+5/-0); test/registered/jit/test_grouped_topk.py (+0/-210); test/registered/jit/test_moe_fused_gate.py (+214/-0)
LABELS: run-ci, jit-kernel, run-ci-extra
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ Part of the fused MoE gate/topk kernel **consolidation** (#26771). SGLang has many per-model gate/topk kernels (`grouped_topk.cuh`, AOT `topk_sigmoid`/`topk_softmax`, the CUDA-JIT `moe_fused_gate`, flashinfer `fused_topk_deepseek`, …). #25835 introduced one Triton `moe_fused_gate` router and made it the default for the ungrouped `sqrtsoftplus` path. ⏎  ⏎ This PR converges the remaining paths onto that single router, in three phases: ⏎ -  …[truncated]

### L1-8416544ab0  (L1, 2026-07-03, sha 8416544ab07b, PR #29999)
TITLE: [NPU] bugfix for Base class add mamba_track_indices parameter (#29999)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk
FILES: python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+2/-2); python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py (+2/-0)
LABELS: npu, run-ci
BODY: ## Motivation ⏎ bugfix for 2cases: ⏎ 1.  ⏎ caz #29678 MambaAttnBackendBase class add mamba_track_indices, and affect its subclass ascendmamba--ascendgdnbackend. ⏎ sgl-sglang/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py", line 103, in __init__ ⏎     super().__init__( ⏎   File "/data/wzy/sgl-sglang/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py", line 362, in __init__ ⏎     self.capture() ⏎   File "/data/wzy/ …[truncated]

### L1-e90fec4868  (L1, 2026-07-03, sha e90fec48684f, PR #28048)
TITLE: [Intel GPU] DeepSeek V4 10/N : Add sqrtsoftplus support to fused_topk_torch_native (#28048)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+2/-0)
LABELS: intel, xpu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-c21f6f19cf  (L1, 2026-07-03, sha c21f6f19cfe7, PR #30079)
TITLE: [MoE] Fix moe_fused_gate out-of-range expert id on all-NaN rows (fixes eagle_dp_attention crash) (#30079)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L1.routing.fused_gate
FILES: python/sglang/jit_kernel/moe_fused_gate.py (+3/-0)
LABELS: jit-kernel
BODY: ## Problem ⏎  ⏎ `test/registered/spec/eagle/test_eagle_dp_attention.py` (`TestEAGLE3EngineDPAttention`) crashes the server during decode CUDA-graph capture in the scheduled full run, with a `CUDBG_EXCEPTION_WARP_ILLEGAL_ADDRESS` in `count_and_sort_expert_tokens_kernel` (the `moe_align_block_size` sort). ⏎  ⏎ ## Root cause ⏎  ⏎ PR #29771 (issue #26771) made the unified Triton router the default for ungrouped softmax/sigmoid MoE (`SGLANG_OPT_USE_JIT_KERNEL_FUS …[truncated]

### L1-3836cba9ee  (L1, 2026-07-04, sha 3836cba9eed2, PR #30075)
TITLE: [refactor] Migrate the moe_runner_backend / quantization resolution chains (stack 13/15) (#30075)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+42/-173); python/sglang/srt/arg_groups/overrides.py (+225/-9); python/sglang/srt/runtime_context.py (+6/-0); test/registered/unit/test_model_overrides.py (+137/-3)
BODY: Last writers first: the quantization-driven runner resolutions at the ⏎ head of the moe kernel handler (raises carried into the pass; the ⏎ compatibility asserts and shared-experts-fusion writes stay — that ⏎ field has load-time writers), the deprecated SGLANG_CUTLASS_MOE ⏎ override at its post-fusion slot (the fusion conditions must observe ⏎ the pre-override runner value), and the quantization side of the gguf ⏎ coupling (load_format itself is genuine runt …[truncated]

### L1-7ea2284551  (L1, 2026-07-04, sha 7ea228455101, PR #30076)
TITLE: [refactor] Migrate the DeepSeek family and the parallel-request chains (stack 14/15) (#30076)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/overrides.py (+250/-0); python/sglang/srt/runtime_context.py (+8/-0); python/sglang/srt/server_args.py (+59/-182); test/registered/unit/test_model_overrides.py (+200/-0)
BODY: - Order-safe DeepSeek declarations: the DSA attention fill, the DSA ⏎   page-size selection (aiter preshuffle probe), and the MLA sm100 ⏎   trtllm_mla fill. ⏎ - The dp_size==1 resets and the A2A backend overrides become passes; ⏎   the seven identical ep_size=tp_size adjustments collapse into one ⏎   union-condition declaration. Four parallel-request fields are ⏎   whitelisted with flat leaves (enable_dp_attention, enable_dp_lm_head, ⏎   moe_a2a_backend, ep_si …[truncated]

### L1-754524d8de  (L1, 2026-07-04, sha 754524d8de95, PR #30139)
TITLE: [Fix] Skip cross-node probe in MultimemAllGatherer on single-node runs (fixes mooncake EP segfault) (#30139)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/distributed/device_communicators/triton_symm_mem_ag.py (+11/-2)
LABELS: run-ci
BODY: ## Problem ⏎  ⏎ `test/registered/ep/test_mooncake_ep_small.py` crashes the server at startup with a **segmentation fault** in today's scheduled run: ⏎  ⏎ ``` ⏎ Current thread (most recent call first): ⏎   File ".../torch/distributed/distributed_c10d.py", line 3068 in all_reduce ⏎   ... ⏎   File ".../srt/distributed/parallel_state.py", line 2681 in in_the_same_node_as ⏎   File ".../srt/distributed/device_communicators/triton_symm_mem_ag.py", line 472 in __init__ ⏎    …[truncated]

### L1-fbe3110866  (L1, 2026-07-04, sha fbe3110866cf, PR #29615)
TITLE: Make mem_fraction_static reserve disaggregation-mode aware (#29615)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+30/-7)
LABELS: run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ When `--mem-fraction-static` is not set, SGLang auto-derives it by estimating reserved (non-KV) memory. Today that estimate always includes both prefill activation slack **and** decode/prefill CUDA-graph buffers, regardless of the node's role. ⏎  ⏎ On PD-disaggregated deployments each node runs only one phase, so the other phase's reserve is dead headroom that needlessly shrinks the KV cache: ⏎ - A **prefill** node still reserves decode  …[truncated]

### L1-8fb99bbaf8  (L1, 2026-07-05, sha 8fb99bbaf8dc, PR #30137)
TITLE: [refactor] Config resolution pipeline: full-stack review (10-PR series, review only) (#30137)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: STACK_REVIEW_PLACEHOLDER.md (+5/-0); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+18/-43); python/sglang/srt/arg_groups/hisparse_hook.py (+2/-24); python/sglang/srt/arg_groups/nemotron_h_hook.py (+0/-66); python/sglang/srt/arg_groups/overrides.py (+588/-1); python/sglang/srt/arg_groups/speculative_hook.py (+8/-2); python/sglang/srt/batch_overlap/two_batch_overlap.py (+2/-2); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+2/-1); python/sglang/srt/layers/logits_processor.py (+2/-2); python/sglang/srt/layers/rotary_embedding/mrope.py (+2/-1); (+64 more)
LABELS: documentation, amd, deepseek, speculative-decoding, npu, run-ci, bypass-fastfail, run-ci-extra
BODY: Full-stack review PR for the 10-PR config-resolution series (continues #30063–#30077). This PR carries the complete diff plus a placeholder file and runs full CI — **review and CI only, do not merge**; merging happens through the individual stack PRs below. ⏎  ⏎ | # | PR | Unit | ⏎ |---|----|------| ⏎ | 1 | #30127 | Migrate the disable_overlap_schedule writers and the mamba radix cache resolution to the pipeline | ⏎ | 2 | #30128 | Migrate the NemotronH and …[truncated]

### L1-c016c6f355  (L1, 2026-07-05, sha c016c6f355f7, PR #26788)
TITLE: [JIT Kernel] DeepSeek-V4 DSA indexer: faster top-k + page-table transform (runtime k <= 2048) (#26788)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/dsv4/topk.py (+13/-12); python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh (+296/-331); python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk/cluster.cuh (+0/-257); python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk/common.cuh (+0/-176); python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk/ptx.cuh (+0/-54); python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk/register.cuh (+0/-302); python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk/streaming.cuh (+0/-213); python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk_impl.cuh (+752/-0); python/sglang/jit_kernel/include/sgl_kernel/utils.cuh (+39/-5); test/registered/jit/benchmark/bench_topk.py (+90/-0); (+1 more)
LABELS: high priority, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Note: after this PR, the v1 should be deprecated. v2 is now a superset of v1. (this line is written by human) ⏎   ⏎ (auto-generated by claude) ⏎  ⏎ ## Motivation ⏎  ⏎ The DeepSeek-V4 DSA indexer selects the top-k KV positions per query for sparse ⏎ attention and maps the selected indices through a page table. This fused top-k + ⏎ page-table transform sits on the decode critical path, especially for long context. ⏎ This PR provides an optimized JIT kernel  …[truncated]

### L1-81735ecf80  (L1, 2026-07-05, sha 81735ecf8099, PR #29362)
TITLE: [AMD ]Feat/dsv4 ep tbo prefill (#29362)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/batch_overlap/operations_strategy.py (+76/-0); python/sglang/srt/batch_overlap/two_batch_overlap.py (+22/-2); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+8/-0); python/sglang/srt/layers/attention/tbo_backend.py (+37/-0); python/sglang/srt/layers/dp_attention.py (+106/-0); python/sglang/srt/models/deepseek_v2.py (+9/-0); python/sglang/srt/models/deepseek_v4.py (+375/-24); python/sglang/srt/server_args.py (+18/-5); test/registered/amd/test_deepseek_v4_flash_fp8_tbo.py (+163/-0); test/registered/amd/test_deepseek_v4_pro_fp4_tbo.py (+151/-0); (+1 more)
LABELS: amd, deepseek, run-ci, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation from TBO design of rocm/ATOM (https://github.com/ROCm/ATOM) ⏎  ⏎ Add DeepSeek-V4 **prefill two-batch-overlap (TBO)** together with a set of AMD (HIP,MI355) prefill optimizations. TBO splits a prefill batch into two micro-batches and overlaps one ubatch's MoE **communication** with the other's attention + expert **compute**, reducing prefill latency and raising throughput. ⏎  ⏎ This PR wires DSV4 into sglang's TBO op engine for **both**: …[truncated]

### L1-6f22790943  (L1, 2026-07-06, sha 6f22790943a8, PR #30201)
TITLE: cookbook: add Hunyuan 3 (Hy3) Day-0 page (#30201)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/Tencent/Hunyuan3-Preview.mdx (+0/-1); docs_new/cookbook/autoregressive/Tencent/Hy3.mdx (+370/-0); docs_new/cookbook/autoregressive/intro.mdx (+1/-1); docs_new/docs.json (+1/-0); docs_new/src/snippets/configs/tencent/hy3-benchmarks.jsx (+26/-0); docs_new/src/snippets/configs/tencent/hy3.jsx (+546/-0)
LABELS: documentation
BODY: ## What ⏎  ⏎ Day-0 cookbook page for Hunyuan 3 (Hy3) on the config-driven template (per-model config + benchmarks JSX consumed by the shared `_deployment.jsx` / `_playground.jsx` engines). ⏎  ⏎ - **Variants**: BF16 (`tencent/Hy3`) + FP8 (`tencent/Hy3-FP8`), both ~276B/20B-active MoE (HYV3ForCausalLM, 192 experts, 80 layers, 256K context). ⏎ - **Hardware**: H200 / B200 / B300 / GB200 / GB300, single-node. BF16 needs TP=8 (H200/B200) or TP=4 (B300/GB300); FP …[truncated]

### L1-d8462f4961  (L1, 2026-07-06, sha d8462f496194, PR #29783)
TITLE: Fixes for NVFP4 numerical accuracy for router GEMM output and wrong correction bias cast (#29783)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+5/-13)
LABELS: deepseek, run-ci
BODY: When in quick testing, the accept len go from `3.413 -> 3.894` ⏎  ⏎ When OSL is 2048 using generate, ⏎  ⏎ Before: ⏎ len = 3.459, tps = 304.65 ⏎  ⏎ After: ⏎ accept len = 3.842, tps = 334.12 ⏎  ⏎ when add the second commit of this pr, ⏎ len = 3.879, tps = 335.94 (the same basicaly) ⏎  ⏎ Firstly, it's wrong to cast correction bias from fp32 to bf16 (confirmed when loading checkpoint values the accuracy of these values change), secondly, the router gemm output ne …[truncated]

### L1-c861896721  (L1, 2026-07-06, sha c861896721cf, PR #30297)
TITLE: [refactor] Resolve config declarations onto server_args at the end of __post_init__ (#30297)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/hisparse_hook.py (+22/-7); python/sglang/srt/arg_groups/overrides.py (+218/-57); python/sglang/srt/arg_groups/speculative_hook.py (+31/-17); python/sglang/srt/server_args.py (+273/-231); python/sglang/srt/speculative/adaptive_spec_params.py (+4/-2); test/registered/mock_model/test_self_unit_install.py (+4/-0); test/registered/unit/model_executor/test_pool_configurator.py (+4/-0); test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py (+1/-1); test/registered/unit/server_args/test_server_args.py (+17/-4); test/registered/unit/test_model_overrides.py (+129/-63); (+1 more)
LABELS: deepseek, speculative-decoding, run-ci, bypass-fastfail, run-ci-extra
BODY: Continues the config-resolution pipeline series (#30063–#30077, #30137); supersedes the retirement stack #30228–#30232 (closed unmerged). ⏎  ⏎ The resolution pipeline keeps declaring — registry dispatch, post-process passes, provenance stash — but no longer replays each declaration in place. Instead, `__post_init__` applies the accumulated stash onto the fields once, at its very end (gate order, last writer wins). ⏎  ⏎ **Developer contract:** after `__po …[truncated]

### L1-3abdbab9bb  (L1, 2026-07-06, sha 3abdbab9bbb4, PR #23650)
TITLE: :sparkles: [llm][npu][quant] Add W4A8 MXFP quantization support for Qwen3 Dense on Ascend NPU (#23650)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs_new/docs/advanced_features/quantization.mdx (+7/-0); docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_quantization.mdx (+30/-0); python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py (+311/-1); python/sglang/srt/hardware_backend/npu/utils.py (+24/-4); python/sglang/srt/layers/quantization/__init__.py (+2/-0); python/sglang/srt/layers/quantization/modelslim/modelslim.py (+2/-0); python/sglang/srt/layers/quantization/modelslim/schemes/__init__.py (+2/-0); python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_mxfp4_w4a8.py (+107/-0); python/sglang/srt/layers/quantization/npu_mxfp4.py (+127/-0); python/sglang/srt/server_args.py (+1/-0)
LABELS: documentation, quant, npu, run-ci, diffusion
BODY: # Summary ⏎  ⏎ > **Dependency**: This PR depends on #22352 (W8A8 MXFP8 for Qwen3 Dense on Ascend NPU) and should be merged after that PR lands, as it builds on the same NPU quantization infrastructure (`_NPULinearMethodBase`, `ModelSlimConfig` dispatch, etc.). ⏎  ⏎ This PR adds W4A8 MXFP quantization support for Qwen3 dense LLM models on Ascend NPU. It continues the NPU quantization work tracked in issue #21584. ⏎  ⏎ **Hardware requirement:** Ascend 95 …[truncated]

### L1-9bd02dc5b9  (L1, 2026-07-06, sha 9bd02dc5b90e, PR #29383)
TITLE: feat(sgl-kernel): add InfLLM v2 attention kernels (#29383)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+64/-0); .codespellrc (+1/-1); sgl-kernel/csrc/common_extension.cc (+9/-0); sgl-kernel/csrc/infllm_v2/flash_attn/flash_api.cpp (+485/-0); sgl-kernel/csrc/infllm_v2/flash_attn/src/block_info.h (+90/-0); sgl-kernel/csrc/infllm_v2/flash_attn/src/dropout.h (+103/-0); sgl-kernel/csrc/infllm_v2/flash_attn/src/flash.h (+152/-0); sgl-kernel/csrc/infllm_v2/flash_attn/src/flash_blockmask.h (+108/-0); sgl-kernel/csrc/infllm_v2/flash_attn/src/flash_fwd_kernel.h (+2355/-0); sgl-kernel/csrc/infllm_v2/flash_attn/src/flash_fwd_launch_template.h (+335/-0); (+23 more)
LABELS: sgl-kernel, run-ci
BODY: Add InfLLM v2 sparse attention stage1 and max pooling CUDA kernels with Python bindings under sgl-kernel/python/sgl_kernel/infllm_v2/. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎ This PR adds standalone `sgl-kernel` support for the InfLLM v2 sparse attention primitives. The new kernels expose the stage1 sparse attention score computation and the variable-length max pooling step through `sgl_kernel`, so downstream SGLang model code can call them without depending on  …[truncated]

### L1-9ddea8d9ef  (L1, 2026-07-07, sha 9ddea8d9efb3, PR #30302)
TITLE: [AMD] [MORI-EP] Skip LocalExpertCount kernel in decode graph when not recording (#30302)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+13/-2)
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ `_should_record_expert_distribution()` in the mori EP token dispatcher returned `True` ⏎ whenever the CUDA stream was capturing. This baked mori's `LocalExpertCountKernel` (and ⏎ its buffer memsets) into the decode CUDA graph for **every** run — including normal ⏎ serving with no expert-distribution recorder configured. The kernel then replayed on ⏎ every decode step, producing a `local_expert_count` that nothing consumes when record …[truncated]

### L1-4b5c612257  (L1, 2026-07-07, sha 4b5c61225746, PR #22660)
TITLE: Skip redundant moe_sum_reduce for single-expert routing on XPU (#22660)
SOURCES: path_core, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+8/-5); test/registered/moe/test_fused_moe.py (+37/-4)
LABELS: intel, xpu, run-ci, run-ci-extra
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: When topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0, the second invoke_fused_moe_kernel call already writes its output directly into out_hidden_states, so the subsequent moe_sum_reduce is a no-op reduction over a single element. This adds an early-exit check on the XPU path to skip the unnecessary kernel launch, matching the existing optimization already present in the CUDA path. ⏎  ⏎ This is particularly relevant for Llama-4-Scout models  …[truncated]

### L1-6279805962  (L1, 2026-07-07, sha 627980596254, PR #27867)
TITLE: [DSv4] Loading Time Weight Dequant (#27867)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+6/-1); python/sglang/srt/environ.py (+1/-0); python/sglang/srt/layers/quantization/fp8.py (+95/-0); python/sglang/srt/model_loader/loader.py (+1/-0); test/registered/models_e2e/test_deepseek_v4_flash_fp4_h200.py (+45/-2)
LABELS: documentation, quant, amd, dependencies, Multi-modal, deepseek, npu, run-ci, diffusion, model-gateway
BODY: ## Motivation ⏎ Current available weights for DSv4 Flash (from deepseek-ai or sgl-project) do not support TP8, which performs better in H20. ⏎ Dequanting FP4 to FP8 would be more preferrable during weight loading. Because this process depends on the TP size. ⏎ usage: ⏎ ``` ⏎ SGLANG_DSV4_FP4_DEQUANT=1 \ ⏎ python -m sglang.launch_server \ ⏎   --model deepseek-ai/DeepSeek-V4-Flash/ \ ⏎   --tp 8 \ ⏎   --tool-call-parser deepseekv4 \ ⏎   --reasoning-parser deep …[truncated]

### L1-2ad9a243f5  (L1, 2026-07-07, sha 2ad9a243f560, PR #30157)
TITLE: Size KV pool after CUDA graph capture (opt-in) (#30157)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+3/-0); python/sglang/srt/managers/scheduler.py (+5/-2); python/sglang/srt/mem_cache/allocator/base.py (+6/-0); python/sglang/srt/mem_cache/allocator/swa.py (+14/-0); python/sglang/srt/mem_cache/kv_vmm_backing.py (+424/-0); python/sglang/srt/mem_cache/memory_pool.py (+188/-77); python/sglang/srt/mem_cache/swa_memory_pool.py (+20/-0); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+128/-19); python/sglang/srt/server_args.py (+127/-59); test/registered/mem_cache/test_post_capture_kv_sizing.py (+99/-0)
LABELS: run-ci
BODY: **Problem:** the KV pool is sized *before* CUDA-graph capture, so graph memory is a heuristic guess baked into `mem_fraction_static`. On large configs the guess can be off by ~10 GB/GPU; the padded activation reserve silently absorbs the difference, wasting memory on some configs and risking OOM on others. ⏎  ⏎ **Fix:** add `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING` (opt-in, **default off**). When enabled: ⏎ 1. Reserve the KV pool as **virtual memory on …[truncated]

### L1-bbc537035a  (L1, 2026-07-07, sha bbc537035aa6, PR #30378)
TITLE: [DSA] Re-enable fused top-k v2 for MTP: clamp padded-row seq_lens to >= 0 (#30378)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/dsv4/topk.py (+20/-0); python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py (+17/-11); python/sglang/srt/layers/attention/dsa_backend.py (+0/-16); python/sglang/srt/layers/attention/triton_ops/dsa_metadata.py (+7/-0); python/sglang/srt/layers/attention/triton_ops/pad.py (+6/-1); test/registered/kernels/test_dsa_indexer.py (+7/-0)
LABELS: jit-kernel
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: (generated by claude) ⏎  ⏎ Follow-up to #30274: removes its TEMP `allow_topk_v2` gate and fixes the root cause of the GLM 5.2 MTP illegal memory access ([failing rerun](https://github.com/sgl-project/sglang/actions/runs/28815316756)). ⏎  ⏎ ## Root cause ⏎  ⏎ Under DP attention, the idle-companion / DP-padded rows of a **draft-extend-v2** CUDA graph replay carry the graph's seq_len fill value (**1**), which is smaller than `qo_len` (= `speculative_num_draft_t …[truncated]

### L1-1da7d3a50b  (L1, 2026-07-07, sha 1da7d3a50bfe, PR #29997)
TITLE: [MoE] Retire the AOT moe_fused_gate / kimi_k2_moe_fused_gate gate kernels (#26771) (#29997)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.fused_gate
FILES: python/sglang/srt/layers/moe/topk.py (+23/-62); sgl-kernel/CMakeLists.txt (+0/-2); sgl-kernel/csrc/common_extension.cc (+3/-11); sgl-kernel/csrc/common_extension_musa.cc (+4/-11); sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu (+0/-364); sgl-kernel/csrc/moe/moe_fused_gate.cu (+0/-523); sgl-kernel/csrc/moe/moe_fused_gate_musa.cu (+0/-840); sgl-kernel/include/sgl_kernel_ops.h (+0/-18); sgl-kernel/python/sgl_kernel/__init__.py (+0/-4); sgl-kernel/python/sgl_kernel/moe.py (+3/-68); (+9 more)
LABELS: deepseek, sgl-kernel, run-ci, mthreads, jit-kernel, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Follow-up to the MoE gate/topk **consolidation** (#26771), on top of the now-merged #29771. With #29771 the unified Triton router covers every CUDA gate/topk path (ungrouped softmax/sigmoid/sqrtsoftplus + DeepSeek-V3 grouped). This PR retires the two AOT gate kernels that are now redundant on CUDA, converging toward the issue's goal of one Triton + one CUDA (JIT `moe_fused_gate.cuh`) gate kernel. ⏎  ⏎ ## Modifications ⏎  ⏎ **(1) `moe_fused …[truncated]

### L1-2d9f0b3317  (L1, 2026-07-07, sha 2d9f0b3317a6, PR #30328)
TITLE: [NPU] [DOC] Update arguments detail to NPU support features page (#30328)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx (+1/-1)
LABELS: documentation, npu
BODY: ## Motivation ⏎  ⏎ Update arguments `--deepep-dispatcher-output-dtype` supproted options to NPU support features page ⏎  ⏎ ## Modifications ⏎  ⏎ NA ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ NA ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎ NA ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from …[truncated]

### L1-60f502a4fd  (L1, 2026-07-07, sha 60f502a4fd28, PR #30207)
TITLE: [AMD] Register 2 hardware-agnostic 1-GPU PR tests for AMD CI (#30207)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/lora/test_moe_lora_info.py (+2/-1); test/registered/unit/managers/test_customized_info_streaming.py (+2/-1)
LABELS: lora, run-ci
BODY: ## Motivation ⏎  ⏎ Closes part of the AMD-vs-NVIDIA per-commit (PR-tier) coverage gap tracked on ⏎ the ROCm CI dashboard. Both tests already run on NVIDIA per-commit CI ⏎ (`base-b` 1-GPU) but had no `register_amd_ci(...)`. Both are hardware-agnostic ⏎ (mock-model engine test / pure-Triton kernel) and pass unchanged on ROCm. ⏎  ⏎ ## Modifications ⏎  ⏎ `register_amd_ci(...)` added next to the existing `register_cuda_ci(...)` ⏎ (CUDA `base-b` → AMD `stage-b`): ⏎  ⏎ | File  …[truncated]

### L1-49109d4267  (L1, 2026-07-07, sha 49109d42677d, PR #30426)
TITLE: [Tiny] Fix Import Error for Pure TP config with flashinfer_mxfp4 (#30426)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.framework
FILES: python/sglang/srt/layers/moe/moe_runner/runner.py (+5/-0)
BODY: Pure-TP configs with --moe-runner-backend flashinfer_mxfp4 (a2a backend "none", e.g. DeepSeek-V4 NVFP4 on TP8 without expert parallelism) crash at model init: ⏎  ⏎   NotImplementedError: Runner backend MoeRunnerBackend.FLASHINFER_MXFP4 ⏎   requires a fused func for a2a backend none, but none is registered. ⏎  ⏎ The (none, flashinfer_mxfp4) fused func is registered via @register_fused_func in flashinfer_cutlass.py, but that module — unlike deep_gemm/tr …[truncated]

### L1-b7cca0bf8f  (L1, 2026-07-07, sha b7cca0bf8fa9, PR #30347)
TITLE: [refactor] Collect MoE and DP-attention runtime state into typed flag groups (#30347)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+66/-97); python/sglang/srt/layers/dp_attention.py (+7/-8); python/sglang/srt/runtime_context.py (+39/-1); test/registered/ops/test_aiter_allreduce_fusion_amd.py (+3/-3); test/registered/ops/test_aiter_greedy_sample_amd.py (+3/-1); test/registered/unit/test_module_state_ratchet.py (+72/-0); test/registered/unit/test_runtime_context.py (+102/-0)
LABELS: amd, ready-to-merge
BODY: ## Motivation ⏎  ⏎ `layers/moe/utils.py` kept twelve module-level globals (parsed backend enums, deepep mode/config, overlap switches, and two ACTIVE values swapped by the speculative contexts), and `dp_attention.py` kept the DP-attention enable/max-len flags — per-module runtime state with ad-hoc lifecycle that leaks across unit-test teardowns and can only be forced in tests by rebinding import names. ⏎  ⏎ ## Modifications ⏎  ⏎ - **`flags.moe`**: `initializ …[truncated]

### L1-7709a1f358  (L1, 2026-07-07, sha 7709a1f358d4, PR #30348)
TITLE: [refactor] ctx.resources: named slots, stream leases, and workspace buffer leases (#30348)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/lora/lora_moe_runner_marlin.py (+8/-13); python/sglang/srt/compilation/backend.py (+0/-3); python/sglang/srt/eplb/expert_distribution.py (+10/-8); python/sglang/srt/eplb/expert_location.py (+8/-7); python/sglang/srt/eplb/lplb_solver.py (+9/-4); python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py (+9/-6); python/sglang/srt/layers/attention/aiter_backend.py (+0/-3); python/sglang/srt/layers/attention/dsa_backend.py (+6/-6); python/sglang/srt/layers/attention/flash_mla_sm120.py (+7/-5); python/sglang/srt/layers/attention/flashinfer_backend.py (+8/-8); (+35 more)
LABELS: lora, ready-to-merge, blackwell, piecewise-cuda-graph, mthreads
BODY: ## Motivation ⏎  ⏎ Process-level resource handles were scattered as module singletons, each with its own get/set pair, no reset lifecycle, and monkeypatch-only test injection: the CUDA graph memory pool, the EPLB expert-distribution recorder (101 call sites), the publish-once expert-location metadata, the LPLB solver map, two side streams, a CUDA-event pool, and one lazily-created workspace buffer per attention backend. ~24 model files additionally e …[truncated]

### L1-8a868f8c00  (L1, 2026-07-07, sha 8a868f8c00c7, PR #30443)
TITLE: [NVIDIA] Allow modelopt_mixed quantization with flashinfer_cutedsl MoE runner (#30443)
SOURCES: path_core, path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-1); python/sglang/srt/server_args.py (+3/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ ModelOpt MIXED_PRECISION checkpoints with NVFP4 MoE layers — e.g. [nvidia/Qwen3.5-397B-A17B-NVFP4-V2](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4-V2) (NVFP4 routed experts, FP8 attention/shared experts) — resolve to `quantization=modelopt_mixed` and fail to launch with `--moe-runner-backend flashinfer_cutedsl`: ⏎  ⏎ ``` ⏎ AssertionError: Invalid quantization 'modelopt_mixed'. ⏎ FlashInfer CuteDSL MOE currently supports only: 'mod …[truncated]

### L1-96368a5f77  (L1, 2026-07-08, sha 96368a5f77bc, PR #28658)
TITLE: [AMD] Fuse shared-expert sigmoid + bf16->fp32 cast into the MoE append kernel (3 kernels -> 1) (#28658)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py (+28/-3); python/sglang/srt/models/qwen2_moe.py (+22/-8)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ On the AITER shared-expert-fusion path (`Qwen2MoeSparseMoeBlock._append_shared_to_topk_output`), preparing the fused shared-expert routing weight launched **three** GPU kernels per MoE layer per decode step: ⏎  ⏎ 1. `sigmoid_kernel_cuda` — `w = F.sigmoid(shared_logits)` in `_get_shared_expert_weights` (~5.5 us) ⏎ 2. `bfloat16tofloat32_copy_kernel_cuda` — `shared_weights.to(topk_weights.dtype)` inside the append wrapper (~4.5 us) ⏎ 3. …[truncated]

### L1-ca8f15cd70  (L1, 2026-07-08, sha ca8f15cd70f5, PR #30450)
TITLE: Fix FlashInfer A2A IMA by DP-synchronizing the decode graph bucket (#30242) (#30450)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+44/-33); python/sglang/srt/utils/common.py (+10/-0); test/registered/unit/test_legacy_global_ratchet.py (+1/-1)
LABELS: run-ci
ISSUES: #30242 [Bug] Deepseek V4 NVFP4 Disagg Decode node failure
BODY: ## Motivation ⏎  ⏎ Fixes #30242 — CUDA illegal-memory-access on **DeepSeek-V4 NVFP4 disaggregated decode** (GB300, TP/DP/EP=16, `moe-a2a-backend=flashinfer`). Bisected by the reporter to #29461. ⏎  ⏎ ### Root cause ⏎  ⏎ With FlashInfer MoE A2A + DP attention, `require_mlp_tp_gather` is `False` (standard DeepSeek config: `moe_dense_tp_size=1` + `enable_dp_lm_head`). So each rank picks its decode CUDA-graph bucket from its **local** batch size. Different DP/EP …[truncated]

### L1-b8ca06fdad  (L1, 2026-07-08, sha b8ca06fdad41, PR #30387)
TITLE: Fix zero expert routed ids for MoE backends (#30387)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/kernels.py (+3/-1); test/registered/moe/test_zero_experts.py (+66/-0)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ - Keep zero-expert routed ids valid before entering the routed MoE backend. ⏎ - Use expert id `0` with zero scale instead of `-1` for zero-expert selections. ⏎ - Add a CUDA unit test for the zero-expert preprocessing mutation and identity output. ⏎  ⏎ ## Why ⏎  ⏎ LongCat-Flash zero experts are represented as top-k ids `>= num_experts`. The previous preprocessing converted those ids to `-1` and zeroed their scales before calling the routed MoE bac …[truncated]

### L1-65b14881c5  (L1, 2026-07-09, sha 65b14881c5a6, PR #30489)
TITLE: [refactor] Move the EP dispatcher and fusion-workspace manager state onto ctx.resources (#30489)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+39/-20); python/sglang/srt/layers/moe/token_dispatcher/mooncake.py (+33/-11); python/sglang/srt/layers/moe/token_dispatcher/nixl.py (+39/-22); python/sglang/srt/elastic_ep/elastic_ep.py (+1/-1); python/sglang/srt/layers/flashinfer_comm_fusion.py (+23/-13); test/registered/unit/layers/test_flashinfer_comm_fusion.py (+10/-3); test/registered/unit/test_runtime_context.py (+37/-0)
BODY: ## Motivation ⏎  ⏎ The DeepEP, Mooncake, and NIXL dispatcher buffer managers each held their comm-buffer handle and sizing metadata (and DeepEP's dispatch-mode state machine) as class attributes, and the FlashInfer allreduce-fusion workspace kept two module-level manager instances (attn-TP / MoE-TP groups) — per-process singletons with ad-hoc lifecycle: no reset between unit tests, and test injection only by poking module internals. ⏎  ⏎ ## Modifications …[truncated]

### L1-fef2128e19  (L1, 2026-07-09, sha fef2128e1909, PR #30490)
TITLE: [refactor] Add the per-forward flags tier: ctx.forward (#30490)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.framework, L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/base.py (+5/-14); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+3/-2); python/sglang/srt/layers/communicator.py (+18/-15); python/sglang/srt/layers/dp_attention.py (+9/-11); python/sglang/srt/runtime_context.py (+124/-2); python/sglang/srt/utils/multi_stream_utils.py (+4/-19); test/registered/unit/test_runtime_context.py (+158/-0)
BODY: ## Motivation ⏎  ⏎ Per-forward runtime state lived in per-module holders with inconsistent concurrency choices and lifecycles: a thread-local for the multi-stream switch, a bare contextvar for the MoE output buffer, per-forward slots mixed with init-resolved state on the `AttnTpContext` module singleton, and `is_extend_in_batch` as a bare class attribute on the DP buffer wrapper (process-global, sticky, no default). ⏎  ⏎ ## Modifications ⏎  ⏎ - **`ctx.forwar …[truncated]

### L1-1f15308dca  (L1, 2026-07-09, sha 1f15308dca98, PR #30493)
TITLE: [refactor] Retire the legacy config accessor and the remaining process singletons (#30493)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.routing.hash_topk, L1.runner.flashinfer_cutedsl, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-3); python/sglang/srt/layers/moe/hash_topk.py (+2/-3); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+2/-2); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+2/-2); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py (+3/-3); python/sglang/srt/arg_groups/overrides.py (+1/-1); python/sglang/srt/batch_overlap/two_batch_overlap.py (+4/-5); python/sglang/srt/debug_utils/dumper.py (+2/-2); python/sglang/srt/distributed/device_communicators/pymscclpp.py (+2/-2); python/sglang/srt/distributed/device_communicators/pynccl_allocator.py (+2/-2); (+148 more)
LABELS: amd, Multi-modal, deepseek, speculative-decoding, blackwell, npu, mthreads, apple-silicon
BODY: ## Motivation ⏎  ⏎ The config tier has one blessed accessor (`runtime_context.get_server_args`) but 280 call sites still went through the legacy `get_global_server_args` name; a few process singletons (the indexer/routed-experts state capturers, the shared TCPStore, the trace verbosity level) still lived as module globals outside the context lifecycle; and the attention unit-test kit hand-built and published mock `ServerArgs` objects instead of drivi …[truncated]

### L1-a9e804623e  (L1, 2026-07-09, sha a9e804623ef4, PR #30641)
TITLE: Allow EPLB manual test to use FlashInfer A2A (#30641)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/manual/ep/test_eplb.py (+34/-6)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ - Let manual EPLB tests select the MoE A2A backend through `SGLANG_EPLB_TEST_MOE_A2A_BACKEND`. ⏎ - Add FlashInfer A2A configuration for both server launch args and Engine kwargs. ⏎ - Use distinct ports and short shutdown waits for the static EPLB restart flow. ⏎  ⏎ ## Testing ⏎  ⏎ - `with-proxy uv run pre-commit run --files test/manual/ep/test_eplb.py` ⏎ - `python3 -m py_compile test/manual/ep/test_eplb.py` ⏎  ⏎ ## Original commits ⏎  ⏎ - `aa010a046` ⏎  ⏎ --- ⏎  …[truncated]

### L1-0ffed946f2  (L1, 2026-07-09, sha 0ffed946f202, PR #29480)
TITLE: [NPU] Add extra topk_weights input in deepep ll dispatch (#29480)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L1.ep.deepep_dispatcher
FILES: python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+3/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ To adapt to NPU ROCE networking, corresponding modifications have been implemented in DeepEP. The low_latency_dispatch structure now requires the topk_weights parameter to be passed. Therefore, the SGLang framework injects the topk_weights parameter in NPU scenarios to maintain compatibility with NPU deployments without ROCE networking. ⏎  ⏎ ## Modifications ⏎  ⏎ python/sglang/srt/layers/moe/token_dispatcher/deepep.py ⏎  ⏎ ## Accuracy  …[truncated]

### L1-b717546fab  (L1, 2026-07-09, sha b717546fab88, PR #28982)
TITLE: fix(mtp): avoid mtp perf regression in deepseek when enable eplb (#28982)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+14/-6)
LABELS: deepseek, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Motivation ⏎ When EPLB and MTP are enabled in DeepSeek, the draft model reusing the target model's dispatch info causes a degradation in accept rate. We introduce the is_nextn flag to fix this issue. This approach is consistent with the implementation in Qwen3 MoE (see _forward_deepep in Qwen2MoeSparseMoeBlock). ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎ In deepseek_v2.py, we now leverage the is_nextn flag to differentiate between the draft model and the target  …[truncated]

### L1-87992eeec4  (L1, 2026-07-09, sha 87992eeec407, PR #30460)
TITLE: [DeepSeek V2] Reorder dual-stream MoE to main-first to avoid CUDA graph stream explosion (#30460)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/runner_utils/capture_mode.py (+5/-0); python/sglang/srt/models/deepseek_v2.py (+15/-10); python/sglang/srt/runtime_context.py (+5/-0); python/sglang/srt/utils/common.py (+5/-0)
LABELS: deepseek, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Profiling **Kimi-K2.5-NVFP4** (TP8, EAGLE3 spec decode, `flashinfer_trtllm` MoE) showed the target-verify decode CUDA graph fanning out across **~61 streams** — one per model layer — instead of the intended 2 (main + alt). ⏎  ⏎ Root cause is in `DeepseekV2MoE.forward_normal_dual_stream`: the shared-expert branch was enqueued on the alt stream **before** the main (routed) branch. During CUDA graph capture this alt-first, per-layer o …[truncated]

### L1-2c6cd1ef41  (L1, 2026-07-09, sha 2c6cd1ef41c5, PR #29910)
TITLE: [Dep] Upgrade flashinfer to 0.6.14 (#29910)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+7/-3); python/pyproject.toml (+1/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+14/-1); test/registered/moe/test_cutedsl_moe.py (+24/-17)
LABELS: dependencies, run-ci, bypass-fastfail, run-ci-extra, release-highlight
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-7045e0fdff  (L1, 2026-07-10, sha 7045e0fdff67, PR #30754)
TITLE: Seed sgl-kernel topk sigmoid tests on all backends (#30754)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: sgl-kernel/tests/test_moe_topk_sigmoid.py (+5/-9)
LABELS: sgl-kernel, run-ci
BODY: ## Summary ⏎  ⏎ #25356 seeded these tests on AMD after tie-break flakes showed up. This extends the same fix to every backend now that the same flake has shown up on CUDA too. ⏎  ⏎ https://github.com/sgl-project/sglang/actions/runs/28957635401/job/85934654890 ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #29076001910](https://github.com/sgl-project/sglang/actions/runs/29076001910) ⏎ L …[truncated]

### L1-b2f9a95867  (L1, 2026-07-10, sha b2f9a95867b7, PR #29030)
TITLE: [NPU] use standalone group for moe ep (#29030)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+16/-5); python/sglang/srt/distributed/parallel_state.py (+2/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ On NPU hardware, we require a standalone communication group for expert parallelism. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ - python/sglang/srt/distributed/parallel_state.py: add conditions to create ep group for npu ⏎ - python/sglang/srt/layers/moe/fused_moe_triton/layer.py: use the newly created moe_ep_group if is npu hardware else use tp_group by default ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ Shall be covered by ci ⏎  ⏎ ## Speed Tests and Profiling …[truncated]

### L1-3dc93a12ca  (L1, 2026-07-10, sha 3dc93a12cacf, PR #30646)
TITLE: Improve EPLB dispatch handling and diagnostics (#30646)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+0/-1); python/sglang/srt/eplb/eplb_manager.py (+68/-3); python/sglang/srt/eplb/expert_distribution.py (+9/-3); python/sglang/srt/eplb/expert_location.py (+104/-1); python/sglang/srt/eplb/expert_location_dispatch.py (+2/-5); python/sglang/srt/model_executor/model_runner.py (+5/-1); python/sglang/srt/utils/common.py (+23/-0); test/registered/unit/eplb/test_dispatch_dtype_preservation.py (+11/-21)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ - Improve EPLB single-pass gatherer selection for non-DeepEP A2A backends and custom routing paths. ⏎ - Add expert-location layout formatting and rebalance diagnostics behind the existing logging control. ⏎ - Preserve dispatch index dtype when remapping logical expert ids to physical ids, and update the CPU coverage. ⏎ - Allow selected kernel package version checks to be bypassed through the existing environment control. ⏎  ⏎ ## Testing ⏎  ⏎ - `pyt …[truncated]

### L1-3f1694f5e0  (L1, 2026-07-10, sha 3f1694f5e0fe, PR #30697)
TITLE: Update sgl-deep-gemm to 0.1.4.post1 (#30697)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1)
LABELS: dependencies, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-fc2ef35308  (L1, 2026-07-10, sha fc2ef3530886, PR #30802)
TITLE: [refactor] Move MLP collective flags onto ForwardFlags (#30802)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/__init__.py (+2/-0); python/sglang/srt/layers/moe/utils.py (+21/-14); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+0/-2); python/sglang/srt/layers/attention/mamba/mamba.py (+1/-4); python/sglang/srt/layers/communicator.py (+1/-1); python/sglang/srt/layers/linear.py (+10/-1); python/sglang/srt/lora/layers.py (+2/-0); python/sglang/srt/lora/trtllm_lora_temp/attention.py (+2/-0); python/sglang/srt/models/apertus.py (+1/-5); python/sglang/srt/models/bailing_moe.py (+16/-23); (+27 more)
LABELS: deepseek, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Decoder layers computed `should_allreduce_fusion`, `use_reduce_scatter`, and `use_flashinfer_trtllm_bypass` once per layer and threaded them through dense MLP, MoE, mamba, and hybrid mixer `forward` signatures. That was pure plumbing: the flags are per-layer control-plane state, not tensor data. ⏎  ⏎ SGLang already has a per-forward tier (`get_forward()` / `ForwardFlags`) for this class of state. Registering these flags there removes k …[truncated]

### L1-6ed9843b57  (L1, 2026-07-10, sha 6ed9843b57aa, PR #30044)
TITLE: [Kernel] Introduce sglang.kernels namespace and migrate scattered triton_ops kernels (RFC #29630, Phase 2) (#30044)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.moe_align, L1.routing.topk_py
FILES: benchmark/bench_attention_sink/bench_attention_sink_triton.py (+2/-2); benchmark/kernels/decoding_attention_triton/triton_flashinfer_cudnn.py (+1/-1); benchmark/kernels/lora_csgmv/tune_lora_csgmv.py (+6/-6); benchmark/kernels/sliding_window_attention_triton/bench_triton_swa_kernel.py (+1/-1); benchmark/kernels/verify_splitkv_triton/bench_verify_splitkv.py (+2/-2); python/sglang/jit_kernel/dsa/__init__.py (+15/-10); python/sglang/kernels/README.md (+110/-0); python/sglang/kernels/__init__.py (+72/-0); python/sglang/kernels/fused_op.py (+318/-0); python/sglang/kernels/ops/__init__.py (+40/-0); (+151 more)
LABELS: documentation, amd, lora, Multi-modal, deepseek, sgl-kernel, blackwell, run-ci, jit-kernel, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Phase 2 of RFC #29630. Builds on the Phase-1 test baseline (#29636). Establishes `sglang.kernels.ops.<group>` as the one obvious import surface for callable kernels, migrates the scattered `**/triton_ops` kernels into it, and starts routing call sites through it. ⏎  ⏎ ```python ⏎ from sglang.kernels.ops.layernorm import rmsnorm ⏎ from sglang.kernels.ops.activation import silu_and_mul ⏎ from sglang.kernels.ops.kvcache import reshape_and_cache …[truncated]

### L1-0663ebc783  (L1, 2026-07-11, sha 0663ebc783e6, PR #28715)
TITLE: [minimax-m3] Split 4/4: model + VL + glue + function-call + fp8 quant + generic infra (#28715)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.fused_gate, L1.runner.deep_gemm, L1.ep.layer
FILES: python/sglang/jit_kernel/moe_fused_gate.py (+2/-0); python/sglang/srt/layers/moe/ep_moe/kernels.py (+416/-104); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+145/-22); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+7/-2); python/sglang/srt/layers/moe/topk.py (+43/-7); python/sglang/jit_kernel/csrc/gemm/per_token_group_quant_8bit.cuh (+194/-161); python/sglang/jit_kernel/per_token_group_quant_8bit.py (+17/-5); python/sglang/srt/arg_groups/overrides.py (+102/-15); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/minimax_vl.py (+60/-0); (+36 more)
LABELS: quant, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
BODY: Part 4 (final) of the 4-PR split of #27944 (MiniMax-M3). The integration PR — the only split PR that moves forward / quant / attention behavior and carries the e2e validation. Splits 1-3 (#28712 / #28713 / #28714) have merged to `main`; this PR is rebased on top. ⏎  ⏎ ## What's in this PR ⏎  ⏎ - **Models + VL**: `minimax_m3.py`, `minimax_m3_vl.py`, `minimax_vl_common.py` + VL processor/configs ⏎ - **Glue**: `attention_registry`, `minimax_sparse_backend`, ` …[truncated]

### L1-80856aba85  (L1, 2026-07-12, sha 80856aba85c6, PR #30828)
TITLE: Make the mxfp8 MoE runner backend list extensible (#30828)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+11/-0); python/sglang/srt/arg_groups/overrides.py (+3/-6)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The mxfp8 quantization → MoE runner-backend allowlist is hardcoded inline in `_moe_runner_backend_quant_constraints` (`arg_groups/overrides.py`). Downstream forks that add their own mxfp8-compatible MoE runner backend currently have to patch that constraint logic to avoid the "Overriding ..." warning + fallback. ⏎  ⏎ This mirrors the existing extension hooks (`add_moe_runner_backend_choices`, `add_linear_attn_kernel_backend_choices`, e …[truncated]

### L1-96a04cb13f  (L1, 2026-07-12, sha 96a04cb13f9c, PR #30873)
TITLE: Fix DeepEP CI test registration (#30873)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test-extra.yml (+14/-0); test/registered/cp/test_gqa_prefill_cp.py (+1/-1); test/registered/rl/test_return_routed_experts.py (+1/-1); test/run_suite.py (+1/-0)
LABELS: run-ci, run-ci-extra
BODY: ## Summary ⏎  ⏎ The DeepEP tests from #16859 and #28421 were running on the regular H100 config, which makes them fail on PRs where DeepEP needs to be rebuilt, so this moves them to the existing DeepEP H100 config and adds its `extra-b` job ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #29152362875](https://github.com/sgl-project/sglang/actions/runs/29152362875) ⏎ Latest PR Test (Extra): :x: [Run #29192064506](https://github.com/sgl …[truncated]

### L1-a358abd651  (L1, 2026-07-12, sha a358abd6513c, PR #30866)
TITLE: chore: update vlm moe config and tune scripts (#30866)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=128,N=768,device_name=NVIDIA_H200.json (+146/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=16,N=1408,device_name=NVIDIA_H100_80GB_HBM3.json (+34/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/E=32,N=768,device_name=NVIDIA_H100_80GB_HBM3.json (+30/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py (+20/-3); benchmark/kernels/flashinfer_allreduce_fusion/benchmark_fused_collective.py (+9/-1); benchmark/kernels/fused_moe_triton/common_utils.py (+1/-0); benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (+29/-6); test/registered/unit/layers/moe/test_fused_moe_common_utils.py (+61/-0); test/registered/unit/layers/moe/test_fused_moe_triton_config.py (+59/-0)
LABELS: run-ci
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Summary ⏎  ⏎ - add tuned Triton MoE configurations for Qwen3-VL and Kimi-VL on H100, plus Qwen3-VL on H200 ⏎ - reuse a tuned up-projection configuration when a separate down-projection configuration is unavailable ⏎ - add Kimi-VL architecture recognition to the MoE tuning helper and configurable tuner batch sizes/search spaces ⏎ - align the FlashInfer all-reduce microbenchmark precision with the serving path ⏎  ⏎ ## Root cause ⏎  ⏎ The VLM MoE shapes were missi …[truncated]

### L1-24d59d8d74  (L1, 2026-07-12, sha 24d59d8d748b, PR #30858)
TITLE: Fix CUDA 12 Docker dependency resolution (#30858)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+10/-0)
LABELS: run-ci, post version patch
ISSUES: #30856 [Bug] v0.5.15-cu129 Docker image is missing nvidia-cutlass-dsl, causing Qwen3.6 EAGLE startup failure
BODY: ## Summary ⏎  ⏎ Pulled this fix out of the PyTorch upgrade PR. Not sure how the CUDA 12 dependency mismatch survived for this long, but this fixes #30856. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #29149230279](https://github.com/sgl-project/sglang/actions/runs/29149230279) ⏎ Latest PR Test (Extra): :white_check_mark: [Run #29208350188](https://github.com/sgl-project/sglang/actions/runs/29208350188)

### L1-eb31b5310c  (L1, 2026-07-13, sha eb31b5310c8b, PR #27350)
TITLE: Support Waterfill with MegaMoE backend (#27350)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py, L1.routing.hash_topk
FILES: python/sglang/srt/layers/moe/hash_topk.py (+11/-11); python/sglang/srt/layers/moe/topk.py (+13/-15); python/sglang/srt/layers/moe/waterfill.py (+6/-6); docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+7/-7); docs_new/docs/advanced_features/server_arguments.mdx (+2/-2); docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_features.mdx (+1/-1); docs_new/docs/references/environment_variables.mdx (+1/-1); python/sglang/srt/arg_groups/overrides.py (+6/-6); python/sglang/srt/environ.py (+2/-2); python/sglang/srt/model_executor/model_runner.py (+6/-11); (+4 more)
LABELS: documentation, deepseek, npu, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
BODY: ## Summary ⏎ - allow Waterfill to preserve an explicit MegaMoE backend instead of forcing DeepEP ⏎ - allow the MegaMoE env path to be selected before Waterfill fallback logic runs ⏎ - use a rank-local shared expert slot semantic helper for TopK, FusedMoE, and DeepSeek shared expert fusion call sites ⏎ - add server-args tests for explicit MegaMoE and MegaMoE env selection under Waterfill ⏎  ⏎ ## Dependency ⏎ Depends on #27349, which adds the MegaMoE/shared expe …[truncated]

### L1-08d6d297e5  (L1, 2026-07-13, sha 08d6d297e537, PR #29909)
TITLE: [Bugfix][NPU] Fix Hunyuan3 model where MoE's routing_scaling_ratio is missing on NPU (#29909)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+18/-3)
LABELS: run-ci
BODY: ## Motivation ⏎ ```text ⏎ HY3-Preview Model was not able to reach official GPQA-D accuracy of 87.2 when running on NPU. ⏎ ┌─────────────┬──────────────┬──────────┬──────────┬───────┬─────────┬─────────┐ ⏎ │ Model       │ Dataset      │ Metric   │ Subset   │   Num │   Score │ Cat.0   │ ⏎ ├─────────────┼──────────────┼──────────┼──────────┼───────┼─────────┼─────────┤ ⏎ │ Hy3-preview │ gpqa_diamond │ mean_acc │ default  │    40 │   0.175 │ default │ ⏎ └── …[truncated]
