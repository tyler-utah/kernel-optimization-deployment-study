### L1-f49cbbd67d  (L1, 2026-07-13, sha f49cbbd67dea, PR #31001)
TITLE: Fix GLM/DeepSeek NVFP4 + flashinfer_trtllm long-context "!!!!" collapse (NaN routing) (#31001)
SOURCES: symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v2.py (+8/-0)
LABELS: high priority, deepseek, post version patch
ISSUES: #30989 [Bug] GLM-5.2-NVFP4 + flashinfer_trtllm: long-context outputs collapse to '!!!!' (NaN logits) — fp32 correction_bias regression from #29783
BODY: ## Motivation ⏎  ⏎ Fixes #30989. ⏎  ⏎ Serving `nvidia/GLM-5.2-NVFP4` with `--quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm`, long-context requests return nothing but `!!!!!!...` (every output-token logprob is `null`, i.e. NaN). Short requests and FP8 checkpoints are unaffected. ⏎  ⏎ ### Root cause ⏎  ⏎ `MoEGate.__init__` creates `e_score_correction_bias` as **fp32** for `modelopt_fp4`, and the FP4 moe_runner passes it **unconverted** to flashi …[truncated]

### L1-874fc07d9b  (L1, 2026-07-13, sha 874fc07d9bbb, PR #30784)
TITLE: [Kernel] Migrate scattered quantization kernels to sglang.kernels (RFC #29630, Phase 2.5, 1/7) (#30784)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.runner.deep_gemm, L1.runner.flashinfer_trtllm, L1.runner.flashinfer_cutlass, L1.ep.layer, L1.ep.deepep_dispatcher, L1.ep.other_dispatchers, L1.cutlass.adapters
FILES: python/sglang/kernels/ops/moe/__init__.py (+11/-0); python/sglang/kernels/ops/moe/pack_topk_ids.py (+86/-0); benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py (+1/-1); benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm_blackwell.py (+1/-1); benchmark/kernels/flashinfer_allreduce_fusion/benchmark_fused_collective.py (+2/-2); benchmark/kernels/quantization/bench_int8_quant.py (+1/-1); benchmark/kernels/quantization/tuning_block_wise_kernel.py (+5/-3); python/sglang/jit_kernel/tests/test_minimax_m3_mxfp8.py (+2/-2); python/sglang/jit_kernel/triton_store_cache.py (+1/-1); python/sglang/kernels/ops/quantization/__init__.py (+47/-0); (+242 more)
LABELS: documentation, quant, amd, lora, deepseek, sgl-kernel, blackwell, run-ci, diffusion, jit-kernel
BODY: ## Motivation ⏎  ⏎ Phase 2.5 of RFC #29630 (see the updated migration plan in the issue): a full-tree audit after #30044 found ~280 Triton kernels plus CuTe DSL / TileLang kernels still scattered outside the three canonical locations. This is sweep **1 of 7**: the quantization group. ⏎  ⏎ ## Modifications ⏎  ⏎ Byte-identical file moves + import rewrites only (`git diff -M` shows 100% renames): ⏎  ⏎ | From (srt/layers/quantization) | To | Kernels | ⏎ |---|---|---| ⏎  …[truncated]

### L1-2cf2920d07  (L1, 2026-07-13, sha 2cf2920d070f, PR #28220)
TITLE: [FlashInfer v0.6.13] Use CuTe DSL backend for FlashInfer per-token NVFP4 quantization (#28220)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+1/-0); python/sglang/srt/layers/quantization/nvfp4_online.py (+1/-1); python/sglang/srt/model_loader/loader.py (+1/-4)
LABELS: blackwell, run-ci
BODY: ## Motivation ⏎  ⏎ @humansand ⏎  ⏎ FlashInfer now has merged CuTe DSL support for per-token NVFP4 quantization and NVFP4 4over6 in flashinfer-ai/flashinfer#3448. The SGLang integration originally landed while some CuTe DSL NVFP4 support was still incomplete, so the per-token activation and online NVFP4 load-time conversion paths either relied on FlashInfer defaults or explicitly selected the CUDA backend. ⏎  ⏎ This PR updates those existing integration …[truncated]

### L1-423b8485fb  (L1, 2026-07-14, sha 423b8485fbf4, PR #23754)
TITLE: [Quantization] add humming quantization kernel (#23754)
SOURCES: path_core, symbol_pickaxe, corpus:production-kernel-provenance, corpus:kernel-correctness-cases(introducing), body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.triton.moe_align, L1.align.cuda_aot, L1.runner.framework, L1.ep.layer, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/sglang/jit_kernel/csrc/moe/moe_permute_prepare.cu (+127/-0); python/sglang/jit_kernel/moe_permute_prepare.py (+77/-0); python/sglang/kernels/ops/moe/__init__.py (+15/-1); python/sglang/srt/layers/moe/ep_moe/kernels.py (+76/-2); python/sglang/srt/layers/moe/ep_moe/layer.py (+9/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-0); python/sglang/srt/layers/moe/fused_moe_triton/moe_fused_mul_sum.py (+218/-0); python/sglang/srt/layers/moe/moe_runner/humming.py (+817/-0); python/sglang/srt/layers/moe/moe_runner/runner.py (+4/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py (+13/-6); (+23 more)
LABELS: documentation, amd, dependencies, sgl-kernel, run-ci, diffusion, mthreads, jit-kernel, bypass-fastfail
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:a58fa0388e (fix PR 33669) || deep-study performance PR (precision_format)
BODY: This PR add humming kernels to SGLang. ⏎  ⏎ coauthers: @huangzhilin-hzl @guzekai01  ⏎  ⏎ Humming Kenrels: https://github.com/inclusionAI/humming ⏎  ⏎ vLLM supports: https://github.com/vllm-project/vllm/pull/34556 ⏎  ⏎ Humming is a universal, high-performance quantization kernel (similar to the Marlin kernel), but offers several advantages over Marlin: ⏎  ⏎ * Extensive Quantization Support: Supports all combinations of W{1,2,3,4,5,6,7,8}A{16,8,4} for quanti …[truncated]

### L1-ee464fedc6  (L1, 2026-07-14, sha ee464fedc63e, PR #30786)
TITLE: [Kernel] Migrate scattered MoE kernels to sglang.kernels (RFC #29630, Phase 2.5, 2/7) (#30786)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.routing.topk_py, L1.runner.triton, L1.runner.deep_gemm, L1.runner.aiter, L1.ep.layer, L1.upstream.aiter_moe, L1.routing.router_py, L1.cutlass.adapters
FILES: python/sglang/kernels/ops/moe/__init__.py (+26/-0); python/sglang/kernels/ops/moe/deepep_waterfill_kernels.py (+322/-0); python/sglang/kernels/ops/moe/ep_moe_kernels.py (+0/-0); python/sglang/kernels/ops/moe/fill_padded_rows.py (+78/-0); python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py (+0/-0); python/sglang/kernels/ops/moe/mxfp8_moe_amd_gfx95.py (+0/-0); python/sglang/kernels/ops/moe/rocm_moe_utils.py (+0/-0); python/sglang/kernels/ops/moe/router.py (+0/-0); python/sglang/kernels/ops/moe/trtllm_lora_temp/virtual_experts.py (+1/-1); python/sglang/kernels/ops/moe/virtual_experts.py (+1/-1); (+21 more)
LABELS: quant, amd, lora, sgl-kernel, run-ci, jit-kernel, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Phase 2.5 sweep **2 of 7** (see the updated migration plan in RFC #29630): move the ~42 Triton MoE kernels still living under `srt/layers/moe` into `sglang.kernels.ops.moe`. ⏎  ⏎ ## Modifications ⏎  ⏎ **Wholesale moves** (byte-identical, `git diff -M` shows 100% renames; imports rewritten repo-wide): ⏎  ⏎ | From (srt/layers/moe) | To (sglang/kernels/ops/moe) | Kernels | ⏎ |---|---|---| ⏎ | `ep_moe/kernels.py` | `ep_moe_kernels.py` | 22 Triton | ⏎ |  …[truncated]

### L1-e9ef06c560  (L1, 2026-07-14, sha e9ef06c56084, PR #30787)
TITLE: [Kernel] Migrate top-level srt/layers stray kernels to sglang.kernels (RFC #29630, Phase 2.5, 3/7) (#30787)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmark/kernels/bench_fused_gate_sigmoid_mul_add.py (+1/-1); benchmark/kernels/bench_fused_sigmoid_mul.py (+1/-1); python/sglang/jit_kernel/dsv4/elementwise.py (+3/-1); python/sglang/kernels/ops/attention/__init__.py (+21/-0); python/sglang/kernels/ops/attention/deepseek_v4_rope.py (+0/-0); python/sglang/kernels/ops/attention/fused_qk_norm.py (+0/-0); python/sglang/kernels/ops/attention/fused_qk_norm_rope_store.py (+0/-0); python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py (+0/-0); python/sglang/kernels/ops/attention/mrope.py (+89/-0); python/sglang/kernels/ops/attention/rotary_triton.py (+0/-0); (+27 more)
LABELS: deepseek, npu, run-ci, jit-kernel, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Phase 2.5 sweep **3 of 7** (see the migration plan in RFC #29630): the ~40 kernels living in modules directly under `srt/layers`, plus the `rotary_embedding` and `layers/utils` strays. ⏎  ⏎ ## Modifications ⏎  ⏎ **Wholesale moves** (byte-identical; imports rewritten repo-wide): ⏎  ⏎ | From (srt/layers) | To | ⏎ |---|---| ⏎ | `elementwise.py` (7 Triton) | `ops/layernorm/elementwise.py` | ⏎ | `gemma4_fused_ops.py` (6 Triton) | `ops/layernorm/gemma4_fu …[truncated]

### L1-9756f768a6  (L1, 2026-07-14, sha 9756f768a683, PR #30448)
TITLE: Refactor FP4 quantization and remove deprecated JIT kernels (#30448)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.cutlass.nvfp4, L1.runner.flashinfer_trtllm, L1.cutlass.adapters
FILES: python/sglang/jit_kernel/csrc/moe/nvfp4_blockwise_moe.cuh (+0/-882); python/sglang/srt/layers/moe/cutlass_moe.py (+0/-163); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+0/-3); benchmark/kernels/flashinfer_allreduce_fusion/README.md (+1/-1); benchmark/kernels/flashinfer_allreduce_fusion/benchmark_fused_collective.py (+8/-8); benchmark/kernels/quantization/bench_fp4_quant.py (+0/-137); docs_new/docs/advanced_features/quantization.mdx (+1/-6); docs_new/docs/advanced_features/server_arguments.mdx (+2/-2); python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_expert_quant.cuh (+0/-806); python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_quant.cuh (+0/-160); (+28 more)
LABELS: documentation, quant, sgl-kernel, blackwell, run-ci, diffusion, jit-kernel, bypass-fastfail
ISSUES: #28663 Deprecate and delete `--moe-runner-backend cutlass`, dense FP4 GEMM, FP4 quantization (in sgl-kernel)
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: For MoE, must use Flashinfer CUTLASS on SM120 or trtllm-gen on SM100 (better perf in all cases) (high throughput use Flashinfer Cute-DSL) ⏎ For dense GEMM on SM100, use Flashinfer Cute-DSL, on SM120, use Flashinfer CUTLASS ⏎ For quantize, use backend = cute-dsl of fp4_quantize in Flashinfer ⏎  ⏎ Fix https://github.com/sgl-project/sglang/issues/28663 ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  …[truncated]

### L1-caa85ea022  (L1, 2026-07-14, sha caa85ea022dc, PR #31152)
TITLE: Extract init_torch_distributed and refactor into functions (#31152)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/distributed/bootstrap.py (+291/-0); python/sglang/srt/model_executor/model_runner.py (+24/-157)
BODY: ### mrc-init-torch-distributed(init-dist-prep,non_mechanical_provable): Prep init_torch_distributed for extraction: @staticmethod + 15 kwargs + TorchDistributedResult ⏎  ⏎ De-self in place: the method becomes a kwargs @staticmethod returning a ⏎ frozen TorchDistributedResult; the call site unpacks the result onto the ⏎ runner fields. The body stays at its original position in the class. Stage ⏎ the destination bootstrap module with its header + the result  …[truncated]

### L1-08798dba0d  (L1, 2026-07-14, sha 08798dba0d81, PR #31159)
TITLE: Extract MoE/EP setup into a moe_ep_setup module (#31159)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/eplb/eplb_manager.py (+7/-1); python/sglang/srt/model_executor/model_runner.py (+19/-123); python/sglang/srt/model_executor/model_runner_components/moe_ep_setup.py (+149/-0)
BODY: ### mrc-moe-ep-setup(extract-prepare-moe-topk-prep,non_mechanical_provable): inline prepare_moe_topk into model_runner before move ⏎  ⏎ ### mrc-moe-ep-setup(extract-prepare-moe-topk-move,mechanical_provable): move prepare_moe_topk to moe_ep_setup module ⏎  ⏎ ### mrc-moe-ep-setup(extract-prepare-moe-topk-postpare,non_mechanical_provable): black-collapse two now-shorter Waterfill statements ⏎  ⏎ After the upstream #27350 Waterfill rename shortened two strings  …[truncated]

### L1-1dc48c2c3b  (L1, 2026-07-14, sha 1dc48c2c3bd3, PR #31160)
TITLE: Absorb capturer setup and extract the shared-mooncake gate (#31160)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py (+51/-3); python/sglang/srt/model_executor/model_runner.py (+7/-71); python/sglang/srt/state_capturer/indexer_topk.py (+34/-0); python/sglang/srt/state_capturer/routed_experts.py (+10/-3); test/manual/kv_transfer/test_mooncake_transfer_engine_init.py (+1/-1)
BODY: ### mrc-capturer-absorb(absorb-routed-experts-setup-into-routedexpertsca,non_mechanical_provable): Absorb routed-experts setup into RoutedExpertsCapturer.create ⏎  ⏎ Move the enable resolution (enable_return_routed_experts) and num_fused_shared_experts computation from ModelRunner.init_routed_experts_capturer into RoutedExpertsCapturer.create, which now takes the model + model_config and owns its own setup. ModelRunner's method collapses to a single  …[truncated]

### L1-d6cf2908ce  (L1, 2026-07-14, sha d6cf2908ce5b, PR #31162)
TITLE: Introduce KVCacheConfigurator and migrate KV-cache config logic (#31162)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+1/-1); python/sglang/srt/mem_cache/kv_cache_configurator.py (+1537/-0); python/sglang/srt/model_executor/model_runner.py (+62/-0); python/sglang/srt/model_executor/model_runner_components/kv_pool_runtime.py (+115/-0); python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (+14/-1473); python/sglang/srt/model_executor/pool_configurator.py (+1/-1)
BODY: ### mrc-kv-cache-configurator-core(kvc-introduce-skeleton,non_mechanical_provable): Introduce KVCacheConfigurator + KVCacheConfigResult skeletons ⏎  ⏎ ### mrc-kv-cache-configurator-core(kvc-extract-mla-dim-prep,non_mechanical_provable): Prep calculate_mla_kv_cache_dim for extraction ⏎  ⏎ ### mrc-kv-cache-configurator-core(kvc-extract-mla-dim-move,mechanical_provable): Move calculate_mla_kv_cache_dim to mem_cache.kv_cache_configurator (cut+paste) ⏎  ⏎ ### mrc …[truncated]

### L1-771e386332  (L1, 2026-07-14, sha 771e38633216, PR #31125)
TITLE: Disable flaky DSV4-Flash FP4 BCG determinism test (nondeterminism from #30898 idle-rank dummy extend) (#31125)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py (+11/-0)
LABELS: deepseek
BODY: ## Summary ⏎  ⏎ Disables the flaky `TestDSV4FlashFP4BreakableCudaGraphB200.test_determinism_temp_zero` (`'Paris.' != 'Paris'`, e.g. [this scheduled run](https://github.com/sgl-project/sglang/actions/runs/29292733629/job/86961435737)) to unblock per-commit CI. This is a **stopgap** — a proper fix that keeps breakable CUDA graph (BCG) enabled for sparse-DP prefill is deferred. ⏎  ⏎ ## Root cause (bisected + devbox-confirmed) ⏎  ⏎ Bisected to #30898 ("Enable br …[truncated]

### L1-1a35440c4a  (L1, 2026-07-14, sha 1a35440c4af9, PR #30789)
TITLE: [Kernel] Migrate generic attention kernels to sglang.kernels (RFC #29630, Phase 2.5, 4/7) (#30789)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/__init__.py (+20/-0); python/sglang/kernels/ops/attention/dcp_kernels.py (+0/-0); python/sglang/kernels/ops/attention/flash_mla_sm120.py (+1/-1); python/sglang/kernels/ops/attention/flash_mla_sm120_triton.py (+0/-0); python/sglang/kernels/ops/attention/nsa_triton_decode/__init__.py (+1/-1); python/sglang/kernels/ops/attention/nsa_triton_decode/triton_mla_kernels_decode_fused.py (+0/-0); python/sglang/kernels/ops/attention/nsa_triton_decode/triton_mla_kernels_decode_optimized.py (+0/-0); python/sglang/kernels/ops/attention/pa_page_table.py (+98/-0); python/sglang/kernels/ops/attention/utils.py (+0/-0); python/sglang/srt/layers/attention/aiter_backend.py (+10/-10); (+25 more)
LABELS: deepseek, blackwell, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Phase 2.5 sweep **4 of 7** (migration plan in RFC #29630): the generic attention-family Triton kernels. ⏎  ⏎ ## Modifications ⏎  ⏎ **Wholesale moves** (byte-identical; imports rewritten repo-wide): ⏎  ⏎ | From | To | Kernels | ⏎ |---|---|---| ⏎ | `srt/layers/attention/utils.py` | `ops/attention/utils.py` | 8 (MLA fp8 quantize+rope, reshape_and_cache variants, shuffle gather) | ⏎ | `srt/layers/attention/flash_mla_sm120.py` + `_triton.py` | `ops/atten …[truncated]

### L1-31548781e0  (L1, 2026-07-14, sha 31548781e04a, PR #31110)
TITLE: [CPU] bypass scoring_func argument in topk for cpu device (#31110)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+22/-0); test/registered/cpu/test_spec_eagle_parity_cpu.py (+5/-1)
LABELS: intel, cpu, ci, run-ci
BODY: ## Motivation ⏎  ⏎ fix CI fail: ⏎  ⏎ ``` ⏎ ERROR: test_latency_fp8_moe_model (__main__.TestIntelAMXAttnBackendQuant.test_latency_fp8_moe_model) ⏎ ---------------------------------------------------------------------- ⏎ Traceback (most recent call last): ⏎   File "/opt/.venv/lib/python3.12/site-packages/sglang/srt/utils/common.py", line 3168, in retry ⏎     return fn() ⏎            ^^^^ ⏎   File "/opt/.venv/lib/python3.12/site-packages/sglang/test/test_utils …[truncated]

### L1-241937af87  (L1, 2026-07-15, sha 241937af874d, PR #31107)
TITLE: [NPU] Determine the topk norm_type through scoring_func (#31107)
SOURCES: path_core
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+1/-1); python/sglang/srt/models/glm4_moe_lite.py (+1/-0)
LABELS: npu, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 31388 (partial_revert, reason=crash_or_hang)
BODY: ## Motivation ⏎ pr https://github.com/sgl-project/sglang/pull/29509 ⏎ it affected acc of ds coder v2 lite instruct ⏎ caz this model uses softmax func at topk, refactor topk part for npu. ⏎  ⏎  ⏎ ## Modifications ⏎ pr https://github.com/sgl-project/sglang/pull/29509 ⏎ it affected acc of ds coder v2 lite instruct ⏎ for most basic models with default softmax func, use 0, for special ones, set "sigmoid" at the model side. ⏎ 1. Add scoring_func as a parameter f …[truncated]

### L1-532cd337ed  (L1, 2026-07-15, sha 532cd337ed7f, PR #28428)
TITLE: [Intel GPU] DeepSeek V4 12/N: use sgl-kernel implementation of silu_and_mul_clamp to run on XPU (#28428)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/dsv4/moe.py (+10/-2)
LABELS: intel, xpu, run-ci, jit-kernel
BODY: Add Unit test ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) a …[truncated]

### L1-ba5be86d42  (L1, 2026-07-15, sha ba5be86d42af, PR #30792)
TITLE: [Kernel] Migrate DSA + DSV4 attention kernels to sglang.kernels (RFC #29630, Phase 2.5, 5/7) (#30792)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/__init__.py (+22/-0); python/sglang/kernels/ops/attention/dsa/__init__.py (+1/-0); python/sglang/kernels/ops/attention/dsa/cp_split.py (+29/-0); python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py (+0/-0); python/sglang/kernels/ops/attention/dsa/index_buf_accessor.py (+0/-0); python/sglang/kernels/ops/attention/dsa/quant_k_cache.py (+0/-0); python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py (+0/-0); python/sglang/kernels/ops/attention/dsa/transform_index.py (+0/-0); python/sglang/kernels/ops/attention/dsa/triton_kernel.py (+0/-0); python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py (+0/-0); (+50 more)
LABELS: quant, deepseek, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Phase 2.5 sweep **5 of 7** (migration plan in RFC #29630): the DeepSeek DSA and DSV4 attention kernel subtrees (~48 Triton kernels + two TileLang suites). ⏎  ⏎ ## Modifications ⏎  ⏎ **Wholesale moves** (byte-identical; imports rewritten repo-wide) into new `ops/attention/dsa/` and `ops/attention/dsv4/` subpackages: ⏎  ⏎ - `dsa/`: `dequant_k_cache`, `quant_k_cache`, `triton_kernel`, `triton_sparse_mla`, `transform_index`, `tilelang_kernel` (2.5 …[truncated]

### L1-c00131ebaa  (L1, 2026-07-15, sha c00131ebaaeb, PR #30793)
TITLE: [Kernel] Migrate linear-attention, MiniMax-sparse and diffusion kernels to sglang.kernels (RFC #29630, Phase 2.5, 6/7) (#30793)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .pre-commit-config.yaml (+1/-1); benchmark/bench_linear_attention/bench_gdn_prefill_cutedsl.py (+4/-4); benchmark/bench_linear_attention/bench_kda_prefill_cutedsl.py (+6/-6); python/sglang/kernels/ops/attention/__init__.py (+26/-0); python/sglang/kernels/ops/attention/linear/__init__.py (+1/-0); python/sglang/kernels/ops/attention/linear/gdn_blackwell/__init__.py (+0/-0); python/sglang/kernels/ops/attention/linear/gdn_blackwell/kernel_h.py (+0/-0); python/sglang/kernels/ops/attention/linear/gdn_blackwell/kernel_kkt_inv_uw.py (+0/-0); python/sglang/kernels/ops/attention/linear/gdn_blackwell/kernel_o.py (+0/-0); python/sglang/kernels/ops/attention/linear/kda_blackwell/__init__.py (+0/-0); (+34 more)
LABELS: run-ci, diffusion, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Phase 2.5 sweep **6 of 7** (migration plan in RFC #29630): the linear-attention family, MiniMax sparse ops, and the `multimodal_gen` strays — including the only `.cu` source living outside the canonical kernel trees. ⏎  ⏎ ## Modifications ⏎  ⏎ **Wholesale moves** (byte-identical; imports rewritten repo-wide): ⏎  ⏎ | From | To | ⏎ |---|---| ⏎ | `attention/linear/{seg_la,lightning_attn}.py` (12 Triton) | `ops/attention/linear/` | ⏎ | `attention/linear …[truncated]

### L1-4aadf94146  (L1, 2026-07-15, sha 4aadf94146b1, PR #30795)
TITLE: [Kernel] Relocate vendored fla and mamba kernel trees to sglang.kernels (RFC #29630, Phase 2.5, 7/7) (#30795)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmark/bench_linear_attention/bench_cutedsl_kda_decode.py (+2/-2); benchmark/bench_linear_attention/bench_fused_gate_cumsum.py (+3/-3); benchmark/bench_linear_attention/bench_gdn_decode.py (+2/-2); benchmark/bench_linear_attention/bench_gdn_prefill.py (+2/-2); benchmark/bench_linear_attention/bench_gdn_prefill_cutedsl.py (+4/-4); benchmark/bench_linear_attention/bench_kda_decode.py (+2/-2); benchmark/bench_linear_attention/bench_kda_prefill_cutedsl.py (+1/-1); benchmark/fla/benchmark_layernorm_gated.py (+2/-2); benchmark/kernels/decoding_attention_triton/triton_flashinfer_cudnn.py (+1/-1); python/sglang/kernels/ops/attention/__init__.py (+16/-0); (+78 more)
LABELS: deepseek, blackwell, run-ci, mthreads, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Phase 2.5 sweep **7 of 7** (migration plan in RFC #29630): pure directory relocations of the two vendored linear-state kernel libraries. Large diffstat, zero logic change — kept separate precisely so the trivial-but-huge diff doesn't drown the substantive sweeps. ⏎  ⏎ ## Modifications ⏎  ⏎ | From | To | Content | ⏎ |---|---|---| ⏎ | `srt/layers/attention/fla/` | `ops/attention/fla/` | flash-linear-attention port: 34 Triton kernels / 18 files ( …[truncated]

### L1-ec32590025  (L1, 2026-07-15, sha ec3259002572, PR #30706)
TITLE: feat(moriep): add fp4 combine dtype (SGLANG_MORI_COMBINE_DTYPE=fp4) (#30706)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+6/-1); docker/rocm.Dockerfile (+1/-1); test/manual/ep/test_moriep_combine_dtype.py (+28/-0)
LABELS: amd
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎  ⏎ Exposes MoRI's **blockwise-FP4 (E2M1) combine** via `SGLANG_MORI_COMBINE_DTYPE=fp4`. This maps to ⏎ MoRI's `fp4_blockwise` combine quant type, which transports each combine element as packed FP4 ⏎ (2 values/byte) instead of FP8 (1 byte/elem) — **~half the combine payload**. Combine is a ⏎ transport-bound step in MoE decode, so this improves decode throughput/latency with no accuracy loss. ⏎  ⏎ Requires the companion MoRI change (`fp4_blo …[truncated]

### L1-8ed82afcc8  (L1, 2026-07-15, sha 8ed82afcc8c1, PR #25663)
TITLE: [MoE Refactor] [NPU] Refactor Ascend MoE implementation to reduce code duplication and align with community design (#25663)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework, L1.hardware.cpu_npu_musa, L1.ep.layer, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/hardware_backend/npu/moe/activation.py (+183/-0); python/sglang/srt/hardware_backend/npu/moe/finalize_routing.py (+99/-0); python/sglang/srt/hardware_backend/npu/moe/fuseep.py (+55/-48); python/sglang/srt/hardware_backend/npu/moe/hidden_states_quant.py (+72/-0); python/sglang/srt/hardware_backend/npu/moe/init_routing.py (+129/-0); python/sglang/srt/hardware_backend/npu/moe/matmul.py (+49/-0); python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py (+0/-1217); python/sglang/srt/layers/moe/ep_moe/layer.py (+0/-5); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-1); python/sglang/srt/layers/moe/moe_runner/ascend.py (+310/-0); (+44 more)
LABELS: documentation, quant, amd, Multi-modal, deepseek, speculative-decoding, hicache, npu, run-ci, diffusion
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:d708969f68 (fix PR 31782)
BODY: ## Overview ⏎  ⏎ This PR addresses two major issues in the current Ascend MoE (Mixture of Experts) implementation: ⏎  ⏎ 1. **High code duplication** – Many places required changes whenever the MoE logic was updated, leading to high maintenance overhead and inconsistency risks. ⏎ 2. **Deviation from community design** – dispatching and grouped-GEMM were hard‑coded in a monolithic `apply` flow and scattered across multiple files to handle different quan …[truncated]

### L1-980acd6eca  (L1, 2026-07-15, sha 980acd6eca1a, PR #29007)
TITLE: Fix MoE TP allreduce to use NCCL symmetric memory via in-pool output allocation (#29007)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.deep_gemm
FILES: python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+41/-19); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+13/-1); python/sglang/kernels/ops/layernorm/mhc.py (+24/-9); python/sglang/srt/layers/dp_attention.py (+12/-4); python/sglang/srt/models/deepseek_v4.py (+15/-2); test/registered/kernels/test_mhc_kernels.py (+10/-0)
LABELS: deepseek, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: cc @yizhang2077 @ShangmingCai @nvcastet @ispobock   @Fridge003 @merrymercy PTAL, thx. ⏎  ⏎   ## Motivation ⏎  ⏎ When `--enable-symm-mem` is set, SGLang's MoE layer should ensure the tensor passed to `tensor_model_parallel_all_reduce` resides in the NCCL symmetric memory pool so the fast path is taken. However, two issues prevented this from working: ⏎  ⏎   1. MoE runners allocated their own output buffers outside the symm pool. The forward_impl previou …[truncated]

### L1-5d004a20c5  (L1, 2026-07-15, sha 5d004a20c58a, PR #29929)
TITLE: Fix FlashInfer A2A top-k ID dtype (#29929)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+11/-2); test/registered/ep/test_flashinfer_a2a.py (+65/-0)
LABELS: blackwell, run-ci
BODY: [by Codex] ⏎  ⏎ ## Summary ⏎  ⏎ - Materialize bypassed top-k output before FlashInfer MoE all-to-all dispatch. ⏎ - Convert selected expert IDs to `torch.int32`, as required by FlashInfer's `MoeAlltoAll.dispatch` API. ⏎ - Add a registered four-GPU B200 GLM-5.2 NVFP4 test covering both `DEP4+FlashInferA2A+StaticEP` and the default DEP4 backend. ⏎  ⏎ ## Motivation ⏎  ⏎ SGLang's static expert-parallel dispatch remaps selected experts through an `int64` expert-location t …[truncated]

### L1-edb2059139  (L1, 2026-07-15, sha edb2059139d4, PR #28309)
TITLE: Support Flashinfer one-sided A2A + CuteDSL MoE for Nemotron Ultra (#28309)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py, L1.runner.flashinfer_trtllm, L1.runner.flashinfer_cutedsl
FILES: python/sglang/srt/layers/moe/flashinfer_cutedsl_moe.py (+34/-8); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+14/-3); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+6/-8); python/sglang/srt/layers/moe/topk.py (+1/-1); python/sglang/srt/models/nemotron_h.py (+24/-8); python/sglang/srt/models/nemotron_h_utils.py (+11/-4); python/sglang/srt/server_args.py (+2/-1)
LABELS: run-ci, jit-kernel
BODY: ## Motivation ⏎  ⏎ Support A2A for Nemotron 3 ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048 \ ⏎ SGLANG_FLASHINFER_WORKSPACE_SIZE=1073741824 \ ⏎ python3 -m sglang.launch_server \ ⏎     --dp-size 8 \ ⏎     --enable-dp-attention \ ⏎     --enable-dp-lm-head \ ⏎     --ep-size 8 \ ⏎     --mamba-full-memory-ratio 5.0 \ ⏎     --max-running-requests 1024 \ ⏎     --mem-fraction-static 0.93 \ ⏎     --model-path nvidia/NV …[truncated]

### L1-0b372d03de  (L1, 2026-07-15, sha 0b372d03de14, PR #31416)
TITLE: [CI] Guard partition consumers against failed check-changes and degenerate fits (#31416)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/pr-test.yml (+3/-0); scripts/ci/utils/compute_partitions.py (+5/-0)
LABELS: run-ci
BODY: Two hardenings from today's CI outage (a degenerate partition-model fit failed `check-changes` repo-wide ([example](https://github.com/sgl-project/sglang/actions/runs/29466164026/job/87522759392): `Suite 'base-c-test-deepep-4-gpu-b200': fit bias=1462.2s >= target=1350.0s`), and every `wait-for-*` job then exploded with misleading `fromJson('')` template errors that buried the root cause ([example](https://github.com/sgl-project/sglang/actions/run …[truncated]

### L1-a614821341  (L1, 2026-07-16, sha a61482134186, PR #31373)
TITLE: [Docs] Align B200 DeepSeek-V4-Pro balanced recipe with MegaMoE (#31373)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+4/-4)
LABELS: documentation, deepseek
BODY: ## Motivation ⏎  ⏎ The B200 DeepSeek-V4-Pro FP4 AgentX recipe in [InferenceX #2145](https://github.com/SemiAnalysisAI/InferenceX/pull/2145) validates MegaMoE for the TP8/DP8 path. The cookbook's single-node balanced cell still selected FlashInfer MXFP4, which changes the MoE kernel and serving performance profile. ⏎  ⏎ This update aligns only the major MoE backend and the buffer capacity required by the cell's existing prefill size. The cookbook's indepe …[truncated]

### L1-1b9f228838  (L1, 2026-07-16, sha 1b9f228838b3, PR #31411)
TITLE: [Docs] Playground: migrate CP knob to canonical prefill-CP flags, align gating with runtime semantics (#31411)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .claude/skills/cookbook-add-model/references/authoring-reference.md (+32/-0); docs_new/cookbook/autoregressive/Tencent/Hy3.mdx (+1/-1); docs_new/src/snippets/_playground.jsx (+240/-32); docs_new/src/snippets/configs/MiniMaxAI/minimax-m3.jsx (+4/-1); docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+12/-1); docs_new/src/snippets/configs/poolside/laguna-m1.jsx (+3/-2); docs_new/src/snippets/configs/tencent/hy3.jsx (+4/-1); docs_new/src/snippets/configs/zai-org/glm-5.2.jsx (+30/-5)
LABELS: documentation, deepseek
BODY: ## Motivation ⏎  ⏎ The playground's attention-axis CP knob still emitted the **deprecated** DSA/NSA flag pair (`--enable-nsa-prefill-context-parallel --nsa-prefill-cp-mode round-robin-split`, deprecated by #27312 in favor of `--enable-prefill-cp --cp-strategy`), and its `apply()` only stripped the legacy flag heads. This breaks against new-style recipes: the GLM-5.2 H20 W4AFP8 cell in #31006 bakes `--attn-cp-size 8 --enable-prefill-cp --cp-strategy i …[truncated]

### L1-871c648203  (L1, 2026-07-16, sha 871c6482037e, PR #31388)
TITLE: [NPU]revert add scoring func for GLM 4.7 Flash (#31388)
SOURCES: path_core
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+1/-1); python/sglang/srt/models/glm4_moe_lite.py (+0/-1)
DEEP_STUDY: deep-study revert record: partial_revert of PR(s) 31107 reason=crash_or_hang
BODY: ## Motivation ⏎ My Fault. #31107  ⏎ I revised the wrong place in #31107 Deeply sorry I'll take this as a lesson, ⏎ As TypeError: FusedMoE.__init__() got an unexpected keyword argument 'scoring_func' occcurs, revert scoring func for GLM 4.7 Flash ⏎  ⏎  ⏎ ## Modifications ⏎ As TypeError: FusedMoE.__init__() got an unexpected keyword argument 'scoring_func' occcurs, revert scoring func for GLM 4.7 Flash ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Prof …[truncated]

### L1-3a8ddd3fd0  (L1, 2026-07-16, sha 3a8ddd3fd023, PR #31038)
TITLE: [XPU] Route topk_sigmoid and topk_softmax to AOT sgl-kernel-xpu symbols (#31038)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+11/-6)
LABELS: run-ci, run-ci-extra
BODY: ## Summary ⏎  ⏎ `sgl-kernel-xpu` shipped the pre-#28715 AOT MoE topk kernels, but `sglang.srt.layers.moe.topk` was importing the new CUDA symbols on XPU too, breaking: ⏎  ⏎ - **`topk_sigmoid`** — CUDA JIT variant needs `tvm_ffi` (absent on XPU) → `ModuleNotFoundError` on `_is_xpu` branch. ⏎ - **`topk_softmax`** — CUDA wrapper forwards 6 args, XPU AOT accepts 4 → `TypeError: topk_softmax() takes from 3 to 4 positional arguments but 6 were given`. ⏎  ⏎ Route the …[truncated]

### L1-22453ca63c  (L1, 2026-07-16, sha 22453ca63c46, PR #31390)
TITLE: docker: build HPC-Ops into the GPU image (#31390)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+34/-2)
LABELS: run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Per the discussion with the Hunyuan HPC-Ops team: depend on the `hpc` library directly in our image so that (a) the opt-in `hpc_ops` attention (#30540) and MoE runner (#30541) backends work out of the box (`import hpc` today requires a manual source build), and (b) future kernel-side improvements land by bumping one pinned commit, with no re-porting — the Python API stays stable. ⏎  ⏎ HPC-Ops is MIT-licensed, so bundling it in the imag …[truncated]

### L1-bff489284b  (L1, 2026-07-16, sha bff489284b50, PR #25763)
TITLE: [Feature] Support DeepSeek-V4 Wint4Abf16 and Win4Afp8. (#25763)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.ep.layer, L1.cutlass.adapters
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+62/-0); python/sglang/srt/layers/moe/cutlass_w4a8_moe.py (+20/-4); python/sglang/jit_kernel/csrc/gemm/per_tensor_quant_fp8.cuh (+35/-12); python/sglang/jit_kernel/per_tensor_quant_fp8.py (+32/-0); python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py (+44/-8); python/sglang/srt/layers/quantization/compressed_tensors/schemes/__init__.py (+2/-0); python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py (+323/-0); python/sglang/srt/layers/quantization/compressed_tensors/utils.py (+1/-0); python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py (+3/-1); python/sglang/srt/models/deepseek_common/utils.py (+22/-0); (+2 more)
LABELS: quant, deepseek, run-ci, jit-kernel, run-ci-extra
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ This is a rebase commit from the upstream branch. ⏎ https://github.com/sgl-project/sglang/pull/23724 ⏎ https://github.com/sgl-project/sglang/pull/21741 ⏎  ⏎ We found that the DeepSeek-V4 mxfp4 version adopts quantization with a group size of 32, while Wint4A16 and Wint4AFP8 usually use a group size of 128 for quantization. After multiple tests, we confirmed that adopting either 128 or 32 for Wint4A16 brings no significant accuracy di …[truncated]

### L1-0a64139c94  (L1, 2026-07-16, sha 0a64139c94c4, PR #30975)
TITLE: Fix --moe-a2a-backend silently ignored for LongCat-2.0 (moe_topk missing from gate) (#30975)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/scheduler.py (+3/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ When serving **LongCat-2.0** with `--moe-a2a-backend deepep`, the flag is **silently ignored**: the model runs with `StandardDispatcher` instead of the requested DeepEP expert-parallel all-to-all. ⏎  ⏎ Root cause is in `Scheduler`: `initialize_moe_config()` (the only place that sets the global `MOE_A2A_BACKEND` from `server_args`) is called only when the model config exposes one of a fixed tuple of top-k attribute names (`moe_topk_attr …[truncated]

### L1-e73f323464  (L1, 2026-07-16, sha e73f323464ee, PR #31400)
TITLE: [JIT] Reduce MoE fused gate CI test sweep (#31400)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/jit/test_moe_fused_gate.py (+72/-19)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ PR #30838 surfaced that the JIT kernel unit-test job can spend too much time in the MoE fused gate test. The failed run collected 535 cases for `test_moe_fused_gate.py` before timing out in the JIT kernel unit-test job: https://github.com/sgl-project/sglang/actions/runs/29414558565/job/87401262608. ⏎  ⏎ ## Changes ⏎  ⏎ - Replace broad cartesian parameter sweeps in `test_moe_fused_gate.py` with representative case lists for the reference, s …[truncated]

### L1-77d23a796e  (L1, 2026-07-16, sha 77d23a796e94, PR #30164)
TITLE: [1/N] elastic-ep: Add runtime EP scale-up (#30164)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+13/-3); python/sglang/srt/layers/moe/token_dispatcher/nixl.py (+93/-20); python/sglang/srt/arg_groups/overrides.py (+1/-1); python/sglang/srt/distributed/bootstrap.py (+15/-4); python/sglang/srt/distributed/parallel_state.py (+121/-26); python/sglang/srt/elastic_ep/elastic_ep.py (+288/-57); python/sglang/srt/entrypoints/elastic_ep.py (+87/-0); python/sglang/srt/entrypoints/engine.py (+6/-5); python/sglang/srt/entrypoints/http_server.py (+14/-2); python/sglang/srt/eplb/eplb_manager.py (+16/-0); (+24 more)
LABELS: run-ci, bypass-fastfail
BODY: # [1/N] elastic-ep: Add runtime EP scale-up ⏎  ⏎ ## Summary ⏎  ⏎ This is the first PR in the runtime Expert Parallel (EP) Scaling series. It ⏎ allows a running MoE deployment to grow through one or more append-only scale ⏎ operations up to a reserved `--max-ep-size`, without restarting the primary ⏎ server. Each operation may add any positive number of ranks that fits within ⏎ the reserved capacity. ⏎  ⏎ ## Motivation ⏎  ⏎ SGLang currently fixes the EP topol …[truncated]

### L1-d67aa05697  (L1, 2026-07-17, sha d67aa0569727, PR #31502)
TITLE: Bump FlashInfer to 0.6.15 and revert regressions (#31502)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+15/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+0/-8); python/sglang/srt/utils/common.py (+1/-1); test/registered/models_e2e/test_dsa_glm52_nvfp4_tp_mtp.py (+1/-1)
LABELS: high priority, dependencies, deepseek, blackwell, run-ci, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study revert record: partial_revert of PR(s)  reason=crash_or_hang || deep-study: this PR was reverted by PR 31625 (confirmed_revert, reason=performance_regression)
BODY: --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #29549992666](https://github.com/sgl-project/sglang/actions/runs/29549992666) ⏎ Latest PR Test (Extra): :no_entry_sign: [Run #29549992592](https://github.com/sgl-project/sglang/actions/runs/29549992592)

### L1-bbd2a3fe4a  (L1, 2026-07-17, sha bbd2a3fe4a26, PR #23795)
TITLE: :sparkles: [llm][npu][quant] Add W4A4 MXFP4 quantization support for Qwen3 Dense on Ascend NPU (#23795)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs_new/docs/advanced_features/quantization.mdx (+2/-2); docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_quantization.mdx (+31/-0); python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py (+275/-0); python/sglang/srt/layers/quantization/__init__.py (+5/-0); python/sglang/srt/layers/quantization/modelslim/modelslim.py (+2/-0); python/sglang/srt/layers/quantization/modelslim/schemes/__init__.py (+2/-0); python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_mxfp4.py (+96/-0); python/sglang/srt/layers/quantization/npu_mxfp4_w4a4.py (+140/-0)
LABELS: documentation, quant, npu, run-ci, diffusion
BODY: # Summary ⏎  ⏎ Adds **MXFP4 W4A4** quantization (4-bit weights + 4-bit activations) for Qwen3 / Qwen3.5 **dense** LLM models on Ascend NPU, continuing the NPU quantization work tracked in #21584. The **online** path uses **dual-level** MXFP4; the **offline** msmodelslim path is single-level. ⏎  ⏎ > **Rebased onto latest `main`.** The two prerequisite PRs this built on — #22352 (W8A8 MXFP8) and #23650 (W4A8 MXFP4) — are now **merged**, so this PR contains …[truncated]

### L1-8432eafd3d  (L1, 2026-07-17, sha 8432eafd3d1d, PR #31292)
TITLE: [Kernel] Decouple KernelBackend from device + device-based CapabilityRequirement (RFC #29630) (#31292)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/moe/__init__.py (+7/-7); python/sglang/kernels/README.md (+2/-2); python/sglang/kernels/__init__.py (+4/-0); python/sglang/kernels/fused_op.py (+42/-21); python/sglang/kernels/ops/activation/__init__.py (+62/-15); python/sglang/kernels/ops/diffusion/__init__.py (+10/-10); python/sglang/kernels/ops/gemm/__init__.py (+10/-12); python/sglang/kernels/ops/layernorm/__init__.py (+22/-22); python/sglang/kernels/ops/mamba/__init__.py (+4/-4); python/sglang/kernels/ops/quantization/__init__.py (+8/-10); (+7 more)
LABELS: documentation, run-ci, bypass-fastfail
BODY: ## What ⏎  ⏎ Implements the design feedback from @DarkSharpness on #29630 (Point 1 + the ⏎ activation priority-inversion it flagged): **decouple `KernelBackend` from the ⏎ device**, and make platform support per-`(op, backend)` metadata rather than ⏎ something baked into the backend name. ⏎  ⏎ ## Why ⏎  ⏎ Both kernel sources already run on AMD — the ROCm `sgl_kernel` wheel builds a ⏎ per-op subset, and `jit_kernel`'s `load_jit` compiles under hipcc — so "which ⏎ devic …[truncated]

### L1-7355e0cb87  (L1, 2026-07-17, sha 7355e0cb875f, PR #30731)
TITLE: [NPU] custom-ops adapt (#30731)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+12/-0); .github/workflows/release-docker-npu-nightly.yml (+1/-1); .github/workflows/release-docker-npu.yml (+1/-1); scripts/ci/npu/npu_ci_install_dependency.sh (+14/-1)
LABELS: npu, run-ci
BODY: --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #29400990590](https://github.com/sgl-project/sglang/actions/runs/29400990590) ⏎ Latest PR Test (Extra): :x: [Run #29400990408](https://github.com/sgl-project/sglang/actions/runs/29400990408)

### L1-486a56be56  (L1, 2026-07-17, sha 486a56be561c, PR #31304)
TITLE: [CPU] improve silu performance by replacing fp32 div with rcp14 (#31304)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: sgl-kernel/csrc/cpu/moe.cpp (+17/-107); sgl-kernel/csrc/cpu/moe.h (+41/-82); sgl-kernel/csrc/cpu/moe_fp8.cpp (+2/-2); sgl-kernel/csrc/cpu/moe_int8.cpp (+11/-19); sgl-kernel/csrc/cpu/activation.cpp (+6/-13); sgl-kernel/csrc/cpu/conv3d.cpp (+4/-5); sgl-kernel/csrc/cpu/decode.cpp (+4/-4); sgl-kernel/csrc/cpu/flash_attn.h (+4/-4); sgl-kernel/csrc/cpu/gemm.cpp (+8/-17); sgl-kernel/csrc/cpu/gemm_fp8.cpp (+6/-8); (+5 more)
LABELS: sgl-kernel, intel, cpu, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ replace slow float32 div with faster rcp14 instruction on x86 ⏎  ⏎ ## Modifications ⏎  ⏎ - Replaces repeated paired fVec::loadu(...) patterns with shared helpers (mainly load_float_vec2) across non-ARM CPU kernels. ⏎ - Simplifies MoE/GEMM/decode/flash-attn helper code for readability and consistency. ⏎ - Fixes an accidental regression in fill_val_stub (int4 path) and restores correct vector fill behavior. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ curren …[truncated]

### L1-0ad0ff2e9e  (L1, 2026-07-17, sha 0ad0ff2e9ee4, PR #31618)
TITLE: chore: bump sglang-kernel version to 0.4.5 (#31618)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sglang-kernel` version to `0.4.5` across SGLang files to match the version defined in `sgl-kernel/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.4.5` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The kernel version in `sgl-kernel/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the `sglang-kernel` dependency are updat …[truncated]

### L1-304a529558  (L1, 2026-07-17, sha 304a52955834, PR #31625)
TITLE: Revert "Bump FlashInfer to 0.6.15 and revert regressions" (#31625)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+2/-15); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/models/deepseek_v2.py (+8/-0); python/sglang/srt/utils/common.py (+1/-1); test/registered/models_e2e/test_dsa_glm52_nvfp4_tp_mtp.py (+1/-1)
LABELS: dependencies, deepseek, blackwell
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 31502 reason=performance_regression
BODY: Reverts sgl-project/sglang#31502, since the new flashinfer version caused performance regression ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #29621280754](https://github.com/sgl-project/sglang/actions/runs/29621280754) ⏎ Latest PR Test (Extra): :x: [Run #29621280651](https://github.com/sgl-project/sglang/actions/runs/29621280651)

### L1-7fbe91c6ea  (L1, 2026-07-17, sha 7fbe91c6ea3c, PR #31492)
TITLE: [AMD] register 8 JIT kernel benchmarks to jit-kernel-benchmark-test-amd (#31492)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/jit/benchmark/bench_custom_all_reduce.py (+2/-1); test/registered/jit/benchmark/bench_fp8_blockwise_gemm.py (+2/-1); test/registered/jit/benchmark/bench_post_reorder_deepgemm.py (+2/-1); test/registered/jit/benchmark/bench_symm_mem_all_gather.py (+2/-1); test/registered/jit/benchmark/bench_tp_qknorm.py (+2/-1); test/registered/jit/benchmark/diffusion/bench_causal_conv3d_cat_pad.py (+2/-1); test/registered/jit/benchmark/diffusion/bench_group_norm_silu.py (+2/-1); test/registered/jit/benchmark/diffusion/bench_norm_impls.py (+2/-1)
BODY: ## Summary ⏎ Follow-up to #30307. Extends AMD JIT-kernel benchmark coverage by registering **8 more** device-agnostic kernel benchmarks to the dedicated `jit-kernel-benchmark-test-amd` stage (same 2-line pattern: `register_amd_ci(est_time=…, stage="jit-kernel-benchmark", runner_config="amd")` next to the CUDA registration). ⏎  ⏎ NVIDIA-hardware-specific benches (nvfp4 / mxfp8 / sm90 / dsv4-fp4-indexer) remain CUDA-only. ⏎  ⏎ ## Selection (empirical, both R …[truncated]

### L1-faf6894093  (L1, 2026-07-18, sha faf68940939a, PR #30272)
TITLE: Implement SM120 DeepSeek V4 flashinfer_mxfp4 moe runner backend + TP2 (#30272)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_cutlass, L1.runner.marlin
FILES: python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h (+2/-5); python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py (+2/-1); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py (+45/-22); python/sglang/srt/layers/quantization/fp8.py (+5/-4); python/sglang/srt/layers/quantization/marlin_utils_fp4.py (+5/-0); python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py (+85/-132); python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py (+6/-4); python/sglang/srt/server_args.py (+2/-0); docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+3/-4); docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+3/-4); (+8 more)
LABELS: documentation, dependencies, deepseek, run-ci, jit-kernel, run-ci-extra
BODY: ## Motivation ⏎  ⏎ DeepSeek-V4-Flash (FP8 checkpoint, MXFP4 experts) could not be served on SM120 (Blackwell desktop/workstation, e.g. RTX PRO 6000). The failures encountered while bringing up TP2/TP4 on 2–4 GPUs were: ⏎  ⏎ 1. The first MoE forward crashed because Marlin dereferenced masked `-1` expert blocks under EP. ⏎ 2. TP2 could not allocate a viable KV cache because FP32 loader containers inflated expert scales by about 16 GB/rank at EP2. ⏎ 3. The firs …[truncated]

### L1-359009fa00  (L1, 2026-07-18, sha 359009fa005f, PR #31449)
TITLE: [Bugfix][NPU] Fix/Refactor routed scaling factor application in MoE routing (#31449)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+2/-1); python/sglang/srt/models/llada2.py (+1/-0); test/registered/ascend/basic_function/dllm/test_npu_llada2_mini.py (+1/-0)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ Currently, the `routed_scaling_factor` is applied inconsistently depending on whether `renormalize` is enabled. When `renormalize=True`, the scaling factor is not applied after weight normalization, leading to incorrect router weights and potentially degraded model performance. This PR unifies the application logic to ensure the scaling factor is always correctly applied, regardless of the renormalization setting. ⏎  ⏎ ## Modificat …[truncated]

### L1-216b750c8f  (L1, 2026-07-18, sha 216b750c8f5b, PR #31582)
TITLE: [Kernel] Sweep decoupled scattered kernels into sglang.kernels.ops (RFC #29630) (#31582)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/kernels/ops/moe/__init__.py (+1/-0); python/sglang/kernels/ops/moe/moe_fused_mul_sum.py (+0/-0); python/sglang/srt/layers/moe/moe_runner/humming.py (+1/-1); python/sglang/kernels/ops/attention/__init__.py (+3/-0); python/sglang/kernels/ops/attention/cute_utils/__init__.py (+0/-0); python/sglang/kernels/ops/attention/cute_utils/_tcgen05.py (+0/-0); python/sglang/kernels/ops/attention/cute_utils/cvt.py (+0/-0); python/sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py (+0/-0); python/sglang/kernels/ops/attention/linear/gdn_blackwell/kernel_h.py (+1/-1); python/sglang/kernels/ops/attention/linear/gdn_blackwell/kernel_kkt_inv_uw.py (+1/-1); (+12 more)
LABELS: deepseek, run-ci, bypass-maintenance, bypass-fastfail, run-ci-extra
BODY: ## Summary ⏎  ⏎ Continues the RFC #29630 scattered-kernel sweep. This PR moves the kernels that are **genuinely decoupled** (no srt runtime-type dependency) into `sglang.kernels.ops.*`, byte-identically, following the Phase 2.5 recipe (move + `KernelSpec` registration + in-tree import rewrites only). ⏎  ⏎ ## Moved (byte-identical) ⏎ | from | to | ⏎ |---|---| ⏎ | `srt/speculative/ragged_verify_kernels.py` | `kernels/ops/speculative/` | ⏎ | `srt/speculative/reject …[truncated]

### L1-ece02ffc9c  (L1, 2026-07-18, sha ece02ffc9cc3, PR #31659)
TITLE: [NPU] FIX CMB illusion of garbled characters acc problems, in prefix cache mtp scenarios. (#31659)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py (+16/-7)
LABELS: npu, run-ci
BODY: ## Motivation ⏎ For NPU, when the input and output reach tens of thousands of tokens, enabling radix cache in MTP scenarios can occasionally cause the illusion of garbled characters. Opening images or using PlanStream will exacerbate this issue. ⏎ When close radix cache, the above hallucination problem will not occur. ⏎  ⏎ ``` ⏎ export GDN_ATTN_BACKEND_TRITON=1 ⏎ export ASCEND_USE_FIA=1 ⏎ export STREAMS_PER_DEVICE=32 ⏎ export SGLANG_DEEPEP_NUM_MAX_DISPAT …[truncated]

### L1-688a6d23f1  (L1, 2026-07-19, sha 688a6d23f144, PR #31705)
TITLE: [DeepSeek-V4] Fix idle-rank dummy-extend sparse-prefill crash under DP breakable CUDA graph (#31705)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/deepseek_v4_backend.py (+2/-0)
LABELS: high priority, deepseek, run-ci, bypass-fastfail
DEEP_STUDY: deep-study correctness case sglang:688a6d23f1: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#30898
BODY: ## Problem ⏎  ⏎ `test_deepseek_v4_flash_fp4_b200.py` (base-c b200 deepep, BCG recipe: TP4/DP4/DeepEP, `--enable-mixed-chunk --cuda-graph-backend-prefill breakable`) crashes deterministically on idle DP ranks: ⏎  ⏎ ``` ⏎ _forward_prefill_sparse: assert core_attn_metadata.c4_sparse_raw_indices is not None ⏎ AssertionError: sparse-prefill c4 path requires c4_sparse_raw_indices ⏎ ``` ⏎  ⏎ ## Root cause ⏎  ⏎ Latent defect from #30898, surfaced by #31487. ⏎  ⏎ Under DP attentio …[truncated]

### L1-02236fa38c  (L1, 2026-07-19, sha 02236fa38cb0, PR #31681)
TITLE: Add Inkling model support (#31681)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.runner.flashinfer_trtllm, L1.runner.marlin, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/sglang/jit_kernel/csrc/moe/inkling_gate_topk_renorm.cuh (+1139/-0); python/sglang/jit_kernel/csrc/trtllm_lora_temp/moe_lora_merged_align_kernel.cu (+3/-5); .codespellrc (+1/-1); docker/Dockerfile (+5/-0); docker/rocm.Dockerfile (+33/-12); python/pyproject.toml (+8/-0); python/pyproject_other.toml (+9/-1); python/sglang/jit_kernel/csrc/inkling/causal_conv1d.cuh (+217/-0); python/sglang/jit_kernel/csrc/inkling/draft_extend_sconv.cuh (+147/-0); python/sglang/jit_kernel/csrc/inkling/fused_decode_update.cuh (+187/-0); (+269 more)
LABELS: documentation, high priority, quant, amd, dependencies, lora, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
BODY: Supersedes #31358 — rebased on the latest `main`. ⏎  ⏎ ## Motivation ⏎  ⏎ Add serving support for the **Inkling** model family — a hybrid-attention Mixture-of-Experts architecture that interleaves sliding-window and full softmax attention with Mamba2 linear-attention layers, an NVFP4-quantized MoE, optional vision/audio multimodal towers, and native multi-token-prediction (MTP) speculative decoding. ⏎  ⏎ ## Modifications ⏎  ⏎ - **Model**: `InklingForConditionalG …[truncated]

### L1-3d82dacd58  (L1, 2026-07-20, sha 3d82dacd580a, PR #31714)
TITLE: Bump CuTe DSL to 4.6.0 (#31714)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+3/-3); python/sglang/kernels/ops/attention/cute_utils/_tcgen05.py (+4/-4); python/sglang/multimodal_gen/test/test_utils.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1); scripts/ci/cuda/ci_install_dependency.sh (+0/-29)
LABELS: dependencies, run-ci, diffusion, run-ci-extra, release-highlight
BODY: ## Summary ⏎  ⏎ - CuTe DSL: `4.5.2` → `4.6.0` ⏎ - FlashAttention-4: `4.0.0b15` → `>=4.0.0b16` ⏎ - Quack: `>=0.4.1` → `>=0.6.1` ⏎ - `nvvm.Tcgen05GroupKind` → `nvvm.CTAGroupKind` ⏎ - `nvvm.tcgen05_commit_arrive` → `nvvm.tcgen05_commit` ⏎ - Updates the related CuTe DSL deprecation warning filter ⏎ - Removes the old CU13 wheel force-reinstall workaround, which is no longer needed with 4.6.0 ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run # …[truncated]

### L1-bab1dd0d12  (L1, 2026-07-20, sha bab1dd0d1295, PR #31126)
TITLE: [Intel XPU] Enable (biased) grouped topk for xpu (#31126)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+90/-4); test/registered/xpu/test_topk.py (+141/-0)
LABELS: intel, xpu, run-ci
BODY: ### Motivation ⏎ Added sycl kernel pass for biased_grouped_topk and grouped_topk on xpu device, which has better than previous default triton implementation. ⏎ ### Modifications ⏎ Applied sycl kernel "moe_fused_gate" on xpu device, also added ut test cases. ⏎  ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #29563083843](https://github.com/sgl-project/sglang/actions/runs/29563083843) ⏎ Latest PR Test (Extra): :x: [Run #29563083691](h …[truncated]

### L1-1f637a65b9  (L1, 2026-07-20, sha 1f637a65b933, PR #31707)
TITLE: [NPU] bugfix for W4A8MoE bias 3D dimension mismatch problem (#31707)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py (+11/-0)
LABELS: run-ci
BODY: ## Motivation ⏎ The refactor of fused NPU MoE introduced in PR #25663 caused a regression, as the bias update logic was missing after bias loading. ⏎ In the original implementation, w13_scale_bias was initialized as a 3D tensor. The _update_bias method performed this operation: ⏎ layer.w13_scale_bias.data.transpose(1, 2).contiguous().sum(dim=1) ⏎ This logic converted the 3D bias tensor into a 2D tensor: it first transposed dimensions 1 and 2, then su …[truncated]

### L1-1843384c7a  (L1, 2026-07-20, sha 1843384c7a59, PR #31311)
TITLE: Fix LongCat-2.0 real EP (deepep): double all-reduce + ScMoE RoPE crash (#31311)
SOURCES: subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/longcat_flash.py (+44/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ LongCat-2.0 crashes or emits garbage under a real MoE all-to-all backend ⏎ (`--moe-a2a-backend deepep`, tp16/ep16). Two coupled bugs, both specific to the ⏎ real-EP path (no-ops when `a2a=none`): ⏎  ⏎ 1. **`LongcatFlashMoE.forward` double all-reduce.** `forward()` ends with an ⏎    unconditional post-experts `tensor_model_parallel_all_reduce`. Correct for ⏎    `a2a=none` (EP-over-TP: experts are TP-sharded partial sums), but under a real ⏎    a2 …[truncated]

### L1-91b210f7b0  (L1, 2026-07-20, sha 91b210f7b06c, PR #28416)
TITLE: [GLM5][MoE] perf: Write FlashInfer TRT-LLM MoE output directly (#28416)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/flashinfer_trtllm_moe.py (+28/-18); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+6/-6)
LABELS: deepseek, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ SGLang allocates the FlashInfer MoE result in symmetric memory, but older FlashInfer releases did not let the monolithic `trtllm_fp8_block_scale_moe` API write into that buffer. The result therefore required an additional `copy_`, and the first version of this PR used a dual-stream copy/add workaround to hide it. ⏎  ⏎ SGLang now pins FlashInfer 0.6.14. That release exposes an `output` tensor for both the [non-routed and routed FP8 bloc …[truncated]

### L1-01f558d905  (L1, 2026-07-20, sha 01f558d905a0, PR #31669)
TITLE: Sm120 scatter fallback (#31669)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.openai_triton_kernels, L1.upstream.openai_triton_kernels
FILES: python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+6/-0)
BODY: i was running this model on an rtx pro 6000. ⏎  ⏎ ``` ⏎ python -m sglang.launch_server --model lmsys/gpt-oss-20b-bf16  --moe-runner-backend triton_kernel ⏎ ``` ⏎  ⏎ i was hitting a tile scatter when using triton_kernel backend not supported for sm120 devices, this fixes this issue by using the non persistent fallback. ⏎  ⏎ i also tested for the response and it looks good. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #29644962388](https://github.co …[truncated]

### L1-37a830b667  (L1, 2026-07-20, sha 37a830b66709, PR #29778)
TITLE: [Feature] Add DWDP (Distributed Weight Data Parallelism) for MoE prefill (#29778)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/dwdp/__init__.py (+13/-0); python/sglang/srt/layers/moe/dwdp/dwdp_manager.py (+211/-0); python/sglang/srt/layers/moe/dwdp/layout.py (+287/-0); python/sglang/srt/layers/moe/dwdp/page_pool.py (+116/-0); python/sglang/srt/layers/moe/dwdp/transport.py (+238/-0); python/sglang/srt/layers/moe/dwdp/vmm.py (+258/-0); python/sglang/srt/layers/moe/dwdp/weight_buffer.py (+221/-0); python/sglang/srt/layers/moe/dwdp/weight_manager.py (+142/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+75/-2); python/sglang/srt/layers/moe/utils.py (+2/-0); (+10 more)
LABELS: deepseek, run-ci, run-ci-extra, release-highlight
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: > **WIP**: This PR is in early development. The code is subject to significant changes as we iterate on the implementation and benchmarking. ⏎  ⏎ ## Summary ⏎  ⏎ Add DWDP as a new parallelism strategy for MoE prefill. Instead of EP all-to-all token dispatch, each rank prefetches peer expert weights via NVLink P2P and computes all experts locally, eliminating all cross-rank synchronization. ⏎  ⏎ Based on the DWDP paper (arXiv 2604.01621), ported from TensorRT …[truncated]

### L1-dcd9014f15  (L1, 2026-07-21, sha dcd9014f1503, PR #28291)
TITLE: [AMD][MXFP4] Reland "Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs" (#28291)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs_new/docs/advanced_features/quantization.mdx (+17/-1); python/sglang/srt/configs/model_config.py (+8/-1); python/sglang/srt/layers/linear.py (+1/-0); python/sglang/srt/layers/quantization/dequantization.py (+44/-0); python/sglang/srt/layers/quantization/fp8.py (+127/-51); python/sglang/srt/layers/quantization/fp8_utils.py (+2/-0); python/sglang/srt/layers/quantization/online_quantization.py (+23/-0); python/sglang/srt/layers/quantization/quark/quark.py (+83/-15); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py (+214/-33); python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py (+375/-28); (+4 more)
LABELS: documentation, quant, run-ci, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study revert record: reland of PR(s) 18182 reason=other
BODY: Motivation and description: please refer to https://github.com/sgl-project/sglang/pull/18182 (original PR), the unit tests gsm8k thresholds, as well as https://github.com/sgl-project/sglang/pull/18005#issuecomment-4111707807 and: ⏎  ⏎ <img width="1270" height="586" alt="image" src="https://github.com/user-attachments/assets/47893323-a587-4b3e-a610-8bac58f1c2a7" /> ⏎  ⏎ <img width="1273" height="639" alt="image" src="https://github.com/user-attachment …[truncated]

### L1-d6ef68881e  (L1, 2026-07-21, sha d6ef68881e26, PR #29131)
TITLE: [NPU] Adapt MiMo-V2.5-W8A8 (#29131)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/modelslim/modelslim.py (+4/-0); python/sglang/test/ascend/test_ascend_utils.py (+1/-0); test/manual/ascend/llm_models/test_npu_mimo_v2_5_w8a8.py (+49/-0)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ Adapt MiMo-V2.5 for NPU deployment by making the code more robust against missing optional dependencies and NPU-specific runtime issues. ⏎  ⏎ ## Modifications ⏎  ⏎ - Optional torchcodec import : Changed from hard import to try-except, so the model can load without torchcodec installed. ⏎ - Graceful audio encoder failure : Wrapped build_audio_encoder in try-except, logging a warning and skipping audio instead of crashing. ⏎ - prefix_in_ …[truncated]

### L1-57e5846b90  (L1, 2026-07-21, sha 57e5846b90f5, PR #31608)
TITLE: [LoRA] Guard TMA down path for LoRA hooks (#31608)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+6/-0)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ - Disable the Triton MoE down-projection TMA path when LoRA `after_gate_up` or `after_down` hooks are installed. ⏎ - Preserve the TMA fast path for all unhooked MoE invocations. On GPUs without `USE_TMA` tuned configs the flag is already false, so this guard is a no-op there. ⏎ - Keep full decode CUDA graph enabled. ⏎  ⏎ This is not a new restriction: hooked MoE invocations never ran on the TMA layout until a refactor accidentally made that c …[truncated]

### L1-927979e127  (L1, 2026-07-21, sha 927979e127a6, PR #31838)
TITLE: Fix pad-row top-k masking with custom_routing_function under DP attention (#31838)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+4/-3); python/sglang/srt/model_executor/cuda_graph_buffer_registry.py (+22/-0); python/sglang/srt/model_executor/forward_batch_info.py (+24/-5); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+6/-0); test/registered/moe/test_topk_padded_region.py (+58/-0); test/registered/unit/model_executor/test_cuda_graph_buffer_registry.py (+75/-0)
LABELS: run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ `select_experts` asserts `num_token_non_padded is None` on the `custom_routing_function` branch, so any model that routes through a custom routing function cannot pass it. Under DP attention with CUDA-graph padding this is a correctness problem, not just a missing feature: the padded rows' router logits are garbage, and without the mask every padded row keeps its unmasked top-k expert ids through EP dispatch. Since the padded rows  …[truncated]

### L1-e4eea7ce2f  (L1, 2026-07-21, sha e4eea7ce2ffa, PR #30247)
TITLE: Optimize LongCat-Flash router GEMM with the HPC-Ops bf16xfp32 kernel (#30247)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: python/sglang/jit_kernel/dsv4/gemm.py (+116/-5); python/sglang/srt/models/longcat_flash.py (+29/-2); test/registered/gemm/test_linear_bf16_fp32_hpc.py (+75/-0); test/registered/unit/models/test_longcat_flash_router_hpc_gemm.py (+116/-0)
LABELS: performance, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Optimize the LongCat-Flash router GEMM (bf16 activations x fp32 router weight) with the `gemm_bf16xfp32` kernel from [HPC-Ops](https://github.com/Tencent/hpc-ops): the fp32 weight is decomposed once into two cached bf16 halves (`w = w_high + w_low / 256`) and the kernel computes both bf16 GEMMs fused, preserving fp32-weight accuracy while running on bf16 tensor cores. ⏎  ⏎ Originally this PR carried an in-tree port of the kernel; now t …[truncated]

### L1-11a4c2d057  (L1, 2026-07-22, sha 11a4c2d05771, PR #31814)
TITLE: config: read resolved config via namespace accessors (#31814)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.routing.hash_topk, L1.hardware.cpu_npu_musa, L1.ep.other_dispatchers
FILES: python/sglang/srt/hardware_backend/npu/moe/fuseep.py (+4/-4); python/sglang/srt/batch_overlap/two_batch_overlap.py (+10/-5); python/sglang/srt/configs/inkling.py (+2/-2); python/sglang/srt/constrained/grammar_manager.py (+2/-1); python/sglang/srt/disaggregation/common/conn.py (+3/-6); python/sglang/srt/disaggregation/decode.py (+4/-6); python/sglang/srt/disaggregation/encode_grpc_server.py (+4/-3); python/sglang/srt/disaggregation/encode_server.py (+17/-22); python/sglang/srt/disaggregation/mooncake/conn.py (+4/-7); python/sglang/srt/disaggregation/nixl/conn.py (+6/-10); (+152 more)
LABELS: Multi-modal, deepseek, blackwell, npu, mthreads, apple-silicon
DEEP_STUDY: deep-study: this PR was reverted by PR 32100 (confirmed_revert, reason=crash_or_hang)
BODY: Stacked on https://github.com/sgl-project/sglang/pull/31813. Part of a stacked series introducing a structured RuntimeContext configuration API (resolved config read through domain namespaces; ServerArgs becomes the read-only record). Incremental and behavior-preserving. ⏎  ⏎ Migrate resolved-config reads from the flat get_server_args()/self.server_args ⏎ surface to the domain namespace accessors (get_model()/get_serving()/get_exec()/ ⏎ get_schedule()/ge …[truncated]

### L1-745b2ca45c  (L1, 2026-07-22, sha 745b2ca45c7f, PR #31816)
TITLE: config: read parallel config leaves via get_parallel() (#31816)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/nixl.py (+3/-5); python/sglang/srt/batch_overlap/two_batch_overlap.py (+1/-2); python/sglang/srt/disaggregation/common/conn.py (+2/-2); python/sglang/srt/disaggregation/mooncake/conn.py (+2/-2); python/sglang/srt/distributed/device_communicators/triton_symm_mem_ag.py (+3/-2); python/sglang/srt/entrypoints/engine.py (+3/-2); python/sglang/srt/layers/attention/dsa/utils.py (+4/-4); python/sglang/srt/layers/communicator.py (+4/-5); python/sglang/srt/layers/logits_processor.py (+2/-2); python/sglang/srt/layers/utils/cp_utils.py (+2/-2); (+51 more)
LABELS: deepseek
DEEP_STUDY: deep-study: this PR was reverted by PR 32100 (confirmed_revert, reason=crash_or_hang)
BODY: Stacked on https://github.com/sgl-project/sglang/pull/31815. Part of a stacked series introducing a structured RuntimeContext configuration API (resolved config read through domain namespaces; ServerArgs becomes the read-only record). Incremental and behavior-preserving. ⏎  ⏎ Parallel config-only leaves read through get_parallel(); the live-topology sizes ⏎ (tp/pp/dcp/attn-cp/moe-dp) keep their config-intent reads. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (B …[truncated]

### L1-2f4f2362fb  (L1, 2026-07-22, sha 2f4f2362fbad, PR #31202)
TITLE: Delete sgl-kernel AOT `bmm_fp8`, use `flashinfer.bmm_fp8` (#31202)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/gemm/__init__.py (+28/-1); python/sglang/srt/layers/quantization/fp8_utils.py (+30/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+1/-24); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py (+1/-1); python/sglang/srt/models/minicpm3.py (+1/-26); python/sglang/srt/models/sarvam_moe.py (+2/-1); sgl-kernel/CMakeLists.txt (+0/-1); sgl-kernel/csrc/common_extension.cc (+0/-6); sgl-kernel/csrc/common_extension_musa.cc (+0/-6); sgl-kernel/csrc/gemm/bmm_fp8.cu (+0/-75); (+5 more)
LABELS: amd, sgl-kernel, run-ci, mthreads, bypass-fastfail
BODY: No need to keep it. The supported CC is the same (SM89+) ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #29743907240](https://github.com/sgl-project/sglang/actions/runs/29743907240) ⏎ Latest PR Test (Extra): :x: [Run #29743907047](https://github.com/sgl-project/sglang/actions/runs/29743907047)

### L1-40b2119b23  (L1, 2026-07-22, sha 40b2119b23e4, PR #31889)
TITLE: [AMD] Cache AITER expert mask across decode (#31889)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/token_dispatcher/standard.py (+2/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The AITER MoE runner rebuilds `expert_mask_gpu` from the local expert mapping on every dispatch. The local expert mapping is fixed for the lifetime of the dispatcher, so recomputing the mask (which includes a host→device copy) on every decode step is wasteful. ⏎  ⏎ ## Modifications ⏎  ⏎ - Guard the AITER mask build with `expert_mask_gpu is None` so it is materialized lazily on first use and cached for subsequent steps. ⏎ - Keep the non-AITER …[truncated]

### L1-03342e7732  (L1, 2026-07-22, sha 03342e77325c, PR #30280)
TITLE: Delete sgl-kernel AOT router GEMM and fused A GEMM (#30280)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: sgl-kernel/CMakeLists.txt (+0/-1); sgl-kernel/csrc/common_extension.cc (+0/-3); sgl-kernel/csrc/common_extension_musa.cc (+0/-3); sgl-kernel/include/sgl_kernel_ops.h (+0/-1); sgl-kernel/python/sgl_kernel/__init__.py (+0/-3); benchmark/kernels/deepseek/benchmark_deepgemm_dsv3_router_gemm_blackwell.py (+0/-250); python/sglang/jit_kernel/dsv3_fused_a_gemm.py (+1/-1); python/sglang/jit_kernel/dsv3_router_gemm.py (+1/-1); python/sglang/jit_kernel/fused_a_gemm.py (+2/-6); sgl-kernel/benchmark/bench_dsv3_fused_a_gemm.py (+0/-73); (+9 more)
LABELS: deepseek, sgl-kernel, run-ci, mthreads, jit-kernel, bypass-fastfail
BODY: Since we migrated already ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #29880101887](https://github.com/sgl-project/sglang/actions/runs/29880101887) ⏎ Latest PR Test (Extra): :x: [Run #29880101823](https://github.com/sgl-project/sglang/actions/runs/29880101823)

### L1-a2c38175a4  (L1, 2026-07-22, sha a2c38175a471, PR #31762)
TITLE: fix(marlin_nvfp4): only apply routed_scaling_factor in moe_sum_reduce (#31762)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+6/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `routed_scaling_factor` is used for routed experts weights.  ⏎  ⏎ However, in marlin nvfp4, this param is computed repeatly in topk and moe_sum_reduce, contributing to garbage output text. ⏎  ⏎ topk: ⏎  ⏎ https://github.com/sgl-project/sglang/blob/50c118704a0ec53eee7984dd51ff7dc2be922af5/python/sglang/srt/models/deepseek_v2.py#L631-L674 ⏎  ⏎ moe_sum_reduce: ⏎  ⏎ https://github.com/sgl-project/sglang/blob/9668d9ea72ae9385c96f8dc1202df38222b …[truncated]

### L1-8bb0d8d005  (L1, 2026-07-22, sha 8bb0d8d00500, PR #30924)
TITLE: [JIT] Trait-driven per_token_group_quant: unify the quant kernel family (flat + masked) (#30924)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.deep_gemm, L1.ep.layer
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+64/-66); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+58/-95); python/sglang/jit_kernel/csrc/gemm/per_token_group_quant.cuh (+544/-0); python/sglang/jit_kernel/csrc/gemm/per_token_group_quant_8bit.cuh (+0/-261); python/sglang/jit_kernel/per_token_group_quant.py (+223/-0); python/sglang/jit_kernel/per_token_group_quant_8bit.py (+0/-108); python/sglang/jit_kernel/per_token_group_quant_8bit_v2.py (+6/-0); python/sglang/kernels/ops/quantization/__init__.py (+46/-4); python/sglang/kernels/ops/quantization/fp8_kernel.py (+89/-137); python/sglang/kernels/ops/quantization/int8_kernel.py (+7/-31); (+7 more)
LABELS: documentation, quant, run-ci, jit-kernel, bypass-fastfail
BODY: (generated by claude) ⏎  ⏎ > **Naming**: the kernel ships as plain `per_token_group_quant` (no version suffix) — it is the default; the v2 JIT kernel remains only as the benchmark perf baseline and is marked deprecated. ⏎ > **Rebased onto main** after #30784 (RFC Phase 2.5, quantization kernels moved to `sglang.kernels.ops.quantization`); all touched entry points now live at the new paths. ⏎  ⏎ > **Stacked on #30838** ([JIT] Refactor dtype traits into DTyp …[truncated]

### L1-d708969f68  (L1, 2026-07-22, sha d708969f68b0, PR #31782)
TITLE: [bugfix][NPU] Fix startup bug in olmoe 1b 7b (#31782)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/token_dispatcher/ascend_tp.py (+2/-1)
LABELS: npu, run-ci
DEEP_STUDY: deep-study correctness case sglang:d708969f68: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#25663
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Serving `OLMoE-1B-7B-0924-Instruct` on Ascend NPU crashes on the first forward pass: ⏎  ⏎ ``` ⏎ File ".../srt/hardware_backend/npu/moe/init_routing.py", line 88, in _init_routing ⏎     active_num=num_tokens * top_k, ⏎ TypeError: unsupported operand type(s) for *: 'int' and 'NoneType' ⏎ ``` ⏎  ⏎ `top_k` reaches `npu_moe_init_routing_v2` as `None`. ⏎  ⏎ **Root cause:** The NPU MoE refactor (#25663) moved `top_k` from being derived at run …[truncated]

### L1-246b3c3eaf  (L1, 2026-07-22, sha 246b3c3eafde, PR #31666)
TITLE: [Kernel] Phase 3+4: move JIT infra + operator groups into sglang.kernels (RFC #29630) (#31666)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.fused_gate, L1.runner.flashinfer_trtllm, L1.runner.marlin
FILES: python/sglang/jit_kernel/dsv4/moe.py (+1/-1); python/sglang/jit_kernel/dsv4/topk.py (+1/-1); python/sglang/jit_kernel/__main__.py (+4/-4); python/sglang/jit_kernel/activation.py (+3/-166); python/sglang/jit_kernel/add_constant.py (+1/-1); python/sglang/jit_kernel/all_reduce.py (+2/-2); python/sglang/jit_kernel/awq_dequantize.py (+1/-1); python/sglang/jit_kernel/awq_marlin_repack.py (+1/-1); python/sglang/jit_kernel/benchmark/marker.py (+1/-1); python/sglang/jit_kernel/clamp_position.py (+1/-1); (+145 more)
LABELS: quant, lora, hicache, blackwell, run-ci, jit-kernel, bypass-maintenance, bypass-fastfail, run-ci-extra
BODY: RFC #29630 **Phase 3 + Phase 4 (consolidated)** — the whole JIT-namespace move lands as one PR. (Per request, the five per-group PRs are folded in here; #31693–#31697 are closed in favor of this.) ⏎  ⏎ ### Phase 3 — shared infra ⏎ `sglang/jit_kernel/utils/` → **`sglang/kernels/jit/`** (compile pipeline, arch, deps, common). ~90 import sites rewritten. `KERNEL_PATH` resolves via `find_spec('sglang.jit_kernel')`, so `csrc/`, `include/`, and the build CLI …[truncated]

### L1-f5dcbe8f14  (L1, 2026-07-22, sha f5dcbe8f142f, PR #32100)
TITLE: Revert RuntimeContext config-namespace reads/roles (#31813–#31817) (#32100)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.routing.hash_topk, L1.hardware.cpu_npu_musa, L1.ep.other_dispatchers
FILES: python/sglang/srt/hardware_backend/npu/moe/fuseep.py (+4/-4); python/sglang/srt/arg_groups/overrides.py (+11/-9); python/sglang/srt/batch_overlap/two_batch_overlap.py (+6/-10); python/sglang/srt/configs/inkling.py (+2/-2); python/sglang/srt/constrained/grammar_manager.py (+1/-2); python/sglang/srt/disaggregation/common/conn.py (+8/-5); python/sglang/srt/disaggregation/decode.py (+6/-4); python/sglang/srt/disaggregation/encode_grpc_server.py (+3/-4); python/sglang/srt/disaggregation/encode_server.py (+24/-20); python/sglang/srt/disaggregation/mooncake/conn.py (+8/-5); (+177 more)
LABELS: high priority, Multi-modal, deepseek, blackwell, npu, run-ci, mthreads, apple-silicon, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 31813;31814;31815;31816;31817 reason=crash_or_hang
BODY: Reverts the upper half of the RuntimeContext config-namespace migration — ⏎ **#31813, #31814, #31815, #31816, #31817** — which merged without full ⏎ verification and carries correctness defects: ⏎  ⏎ - **Load-time fusion fallback desync.** `determine_num_fused_shared_experts()` / ⏎   `_maybe_autodisable_shared_experts_fusion()` call `declare_load_time_override()`, ⏎   which updates the published `ServerArgs`, but the migrated construction-time ⏎   readers cons …[truncated]

### L1-0c29c8fece  (L1, 2026-07-22, sha 0c29c8fecee1, PR #31927)
TITLE: Bump FlashInfer to 0.6.15.post1 (#31927)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L1.runner.flashinfer_cutedsl, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+15/-2); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/layers/attention/dsa_backend.py (+22/-0); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+41/-0); python/sglang/srt/models/deepseek_v2.py (+0/-8); python/sglang/srt/utils/common.py (+1/-1); python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py (+1/-0); python/sglang/test/kits/attention_unittest/attention_methods/mla_attention.py (+1/-0); (+1 more)
LABELS: dependencies, deepseek, blackwell, run-ci, bypass-fastfail, release-highlight
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Summary ⏎  ⏎ - bump `flashinfer_python`, `flashinfer-cubin`, and the optional JIT cache from 0.6.14 to 0.6.15.post1 ⏎ - reland the FlashInfer 0.6.15 API compatibility and regression cleanup from #31502 ⏎ - restore the GLM-5.2 NVFP4 performance threshold after removing the obsolete correction-bias workaround ⏎  ⏎ ## Why ⏎  ⏎ The original 0.6.15 bump was reverted in #31625 after host-side regressions reduced long-context serving throughput. FlashInfer 0.6.15.po …[truncated]

### L1-977ea336cd  (L1, 2026-07-22, sha 977ea336cd3e, PR #32015)
TITLE: [Kernel] Phase 4 batch-2: migrate JIT operator groups into kernels.ops (no shims) (RFC #29630) (#32015)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.marlin, L1.ep.layer
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+1/-1); python/sglang/jit_kernel/benchmark/utils.py (+1/-1); python/sglang/jit_kernel/dsv4/attn.py (+3/-1); python/sglang/jit_kernel/tests/utils.py (+1/-1); python/sglang/kernels/ops/communication/all_reduce.py (+0/-0); python/sglang/kernels/ops/communication/mp.py (+0/-0); python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py (+0/-0); python/sglang/kernels/ops/gemm/cutedsl_dsv3_fused_a_gemm.py (+0/-0); python/sglang/kernels/ops/gemm/fp8_blockwise_gemm.py (+0/-0); python/sglang/kernels/ops/gemm/fused_a_gemm.py (+2/-2); (+85 more)
LABELS: quant, deepseek, hicache, blackwell, run-ci, jit-kernel, bypass-maintenance, bypass-fastfail, run-ci-extra
BODY: RFC #29630 Phase 4 — **consolidated** batch-2 (the directly-imported JIT operator groups, in one PR per request): ⏎  ⏎ | group | operators | ⏎ |---|---| ⏎ | gemm | cutedsl_bf16_gemm, cutedsl_dsv3_fused_a_gemm, fp8_blockwise_gemm, fused_a_gemm | ⏎ | communication | all_reduce, mp | ⏎ | layernorm | fused_eh_norm, rmsnorm_hf | ⏎ | speculative | ngram_corpus, resolve_future_token_ids, ngram_embedding | ⏎ | mamba | transfer_mamba, inkling_sconv | ⏎ | quantization | awq …[truncated]

### L1-74338e94f1  (L1, 2026-07-22, sha 74338e94f10e, PR #32045)
TITLE: [Kernel] Phase 4 batch-3: migrate tangled JIT subsystems + new groups into kernels.ops (RFC #29630) (#32045)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.moe_align, L1.routing.topk_py, L1.routing.topk_sigmoid, L1.routing.fused_gate, L1.routing.hash_topk, L1.runner.deep_gemm, L1.runner.flashinfer_trtllm, L1.runner.marlin, L1.runner.deepgemm_megamoe, L1.ep.layer
FILES: python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/trtllm_fused_moe_kernel_launcher.cu (+0/-4380); python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/trtllm_fused_moe_runner.cu (+0/-1116); python/sglang/jit_kernel/trtllm_lora_temp/data/include/flashinfer/trtllm/fused_moe/DevKernel.h (+0/-472); python/sglang/jit_kernel/trtllm_lora_temp/data/include/flashinfer/trtllm/fused_moe/runner.h (+0/-586); python/sglang/jit_kernel/benchmark/kv_canary/utils.py (+1/-1); python/sglang/jit_kernel/dsa/__init__.py (+0/-30); python/sglang/jit_kernel/dsv4/__init__.py (+0/-63); python/sglang/jit_kernel/kv_canary/plan/__init__.py (+0/-1); python/sglang/jit_kernel/minimax_m3/__init__.py (+0/-25); python/sglang/jit_kernel/tests/deepseek_v4/common.py (+4/-1); (+379 more)
LABELS: documentation, quant, amd, dependencies, lora, Multi-modal, deepseek, blackwell, npu, run-ci
BODY: RFC #29630 Phase 4 — **batch-3** (tangled subsystems + new groups + splits, no shims: delete + rewrite call sites). ⏎  ⏎ **Stage 1 — clean moves:** 19 attention ops (flash_attention{,_v3,_v4}, concat_mla, cutedsl_gdn/kda/paged_mqa, rope, fused_qknorm_rope/minimax_qknorm_rope, hadamard, clamp_position, add_constant, mla_kv_pack_quantize_fp8, ...), 8 moe ops, timestep_embedding; `flash_attn/` → attention; **new groups** `lplb/`, `kv_canary/`; `dsv32/`  …[truncated]

### L1-99f636a86f  (L1, 2026-07-23, sha 99f636a86fb8, PR #32072)
TITLE: [Kernel] RFC #29630 finale: retire sglang.jit_kernel into sglang.kernels (#32072)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.routing.topk_sigmoid, L1.routing.fused_gate, L1.routing.hash_topk, L1.align.cuda_jit, L1.runner.deep_gemm, L1.runner.openai_triton_kernels, L1.runner.marlin, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe, L1.upstream.openai_triton_kernels, L1.cutlass.adapters
FILES: .claude/skills/add-jit-kernel/SKILL.md (+35/-35); .claude/skills/add-sgl-kernel/SKILL.md (+1/-1); .claude/skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md (+18/-18); .claude/skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py (+11/-9); .claude/skills/write-sglang-test/SKILL.md (+3/-3); .github/CODEOWNERS (+4/-4); .github/MAINTAINER.md (+1/-1); .github/labeler.yml (+1/-1); .github/workflows/_pr-test-check-changes.yml (+1/-2); .github/workflows/nightly-test-nvidia.yml (+1/-1); (+344 more)
LABELS: documentation, quant, amd, dependencies, lora, Multi-modal, deepseek, hicache, sgl-kernel, blackwell
BODY: ## Summary ⏎  ⏎ Structural **finale** of the `sglang.jit_kernel` → `sglang.kernels` migration (RFC #29630). After batch-1/2/3 (#31666, #32015, #32045) moved every operator into `sglang.kernels.ops.<group>`, this PR removes the `sglang.jit_kernel` package **entirely**. ⏎  ⏎ ## What changed ⏎  ⏎ - **Build infra moved** into `sglang/kernels/jit/`: `csrc/`, `include/`, `__main__.py`, `benchmark/`, `tests/`, `.clang-format` (git-tracked as renames). ⏎ - **`KERNEL_P …[truncated]

### L1-108182cb81  (L1, 2026-07-23, sha 108182cb8199, PR #32113)
TITLE: [Bugfix] [NPU] Fix w4a8 MoE performance degradation (#32113)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py (+4/-3)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The NPU MoE refactor https://github.com/sgl-project/sglang/pull/25663 changed the W4A8 weight-layout conversion order. The transposed weight was passed to `npu_format_cast` without first materializing a contiguous tensor, and packing made the tensor contiguous before reinterpreting it as `int32`. ⏎  ⏎ For Qwen3.5-397B-A17B W4A8 with DeepEP, this increased mean TPOT from 52.53 ms on the known-good commit (`c9b17403e7c2409ac9603a5826 …[truncated]

### L1-2d1a7be8c4  (L1, 2026-07-23, sha 2d1a7be8c453, PR #32128)
TITLE: [Kernel] Reclassify kernel tests by ops group + move helpers out of the package (RFC #29630) (#32128)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/_pr-test-check-changes.yml (+3/-3); .github/workflows/pr-test-amd-rocm720.yml (+3/-3); .github/workflows/pr-test-amd.yml (+3/-3); python/sglang/kernels/README.md (+19/-12); python/sglang/kernels/ops/communication/mp.py (+1/-1); python/sglang/test/kernels/__init__.py (+0/-0); python/sglang/test/kernels/deepseek_v4/__init__.py (+0/-0); python/sglang/test/kernels/deepseek_v4/common.py (+0/-0); python/sglang/test/kernels/kv_canary/__init__.py (+0/-0); python/sglang/test/kernels/kv_canary/_canary_helpers.py (+6/-6); (+195 more)
LABELS: documentation, quant, amd, lora, deepseek, hicache, blackwell, run-ci, jit-kernel, bypass-maintenance
BODY: ## Summary ⏎  ⏎ Follow-up to RFC #29630. Organizes **all** kernel tests + benchmarks to mirror `sglang.kernels.ops.<group>`, keeps **tests and benchmarks separate**, and gets the last test files out of the `sglang` package. ⏎  ⏎ ## Final layout ⏎  ⏎ ``` ⏎ test/registered/kernels/ ⏎   ops/<group>/         # tests            (test_*.py) ⏎   benchmark/<group>/   # benchmarks       (bench_*.py + data fixtures) ⏎ python/sglang/kernels/testing/   # shared test-support hel …[truncated]

### L1-11b0e5c5ad  (L1, 2026-07-23, sha 11b0e5c5add9, PR #32148)
TITLE: [Kernel] Classification cleanup: unify _jit_ naming, drop empty/model groups, add elementwise (RFC #29630) (#32148)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.deep_gemm, L1.runner.openai_triton_kernels, L1.runner.marlin, L1.upstream.openai_triton_kernels, L1.cutlass.adapters
FILES: python/sglang/kernels/ops/moe/inkling_gate_topk_renorm.py (+0/-0); .claude/skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md (+1/-1); .claude/skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py (+1/-1); benchmark/kernels/bench_fused_gate_sigmoid_mul_add.py (+1/-1); benchmark/kernels/bench_fused_sigmoid_mul.py (+1/-1); python/sglang/kernels/README.md (+3/-3); python/sglang/kernels/ops/__init__.py (+1/-2); python/sglang/kernels/ops/activation/__init__.py (+4/-4); python/sglang/kernels/ops/activation/activation.py (+4/-4); python/sglang/kernels/ops/attention/dsv4/compress_old.py (+3/-3); (+92 more)
LABELS: documentation, quant, Multi-modal, deepseek, sgl-kernel, run-ci, diffusion, jit-kernel, bypass-maintenance, bypass-fastfail
BODY: ## Summary ⏎  ⏎ Post-migration review follow-up on `python/sglang/kernels/` — tightens classification and naming consistency. Almost entirely `git mv` renames + import rewrites. ⏎  ⏎ ## Changes ⏎  ⏎ - **Unify JIT-op naming** — drop the leftover `_jit_` prefix (a batch-1 shim-era artifact) from 8 modules so JIT-backed ops read consistently with the rest (`flash_attention.py`, `rope.py`, …): `activation/activation`, `layernorm/norm`, `gemm/dsv3_{fused_a,router …[truncated]

### L1-09071be105  (L1, 2026-07-23, sha 09071be10556, PR #32130)
TITLE: [NPU] [FIX] Fix performance degradation of Qwen3.5-397B-A17B (#32130)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py (+7/-16)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  In ascend_hybrid_linear_attn_backend.py:275 the .item() operation introduces device synchronization and causes service performance degradation. ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎ ```bash ⏎ export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128 ⏎ export HCCL_BUFFSIZE=3000 ⏎ export DEEPEP_NORMAL_LONG_SEQ_ROUND=32 ⏎ export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=3584 ⏎  ⏎ export PYT …[truncated]

### L1-235a488c87  (L1, 2026-07-23, sha 235a488c8783, PR #32040)
TITLE: [NPU] ascend fuseep use moe ep group (#32040)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/fuseep.py (+2/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Switched ascend_fuseep to use the MoE EP group instead of the TP group. This enables independent HCCL buffer configuration for MoE collective communication via the DEEPEP_HCCL_BUFFSIZE environment variable. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/ …[truncated]

### L1-378aea1385  (L1, 2026-07-23, sha 378aea138550, PR #32246)
TITLE: Fix nvfp4 online scale with pcg (#32246)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+7/-1)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-71fe41b6b3  (L1, 2026-07-23, sha 71fe41b6b3c7, PR #29569)
TITLE: [DSV4] Support megamoe for CP (#29569)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.runner.deepgemm_megamoe
FILES: python/sglang/srt/layers/moe/mega_moe.py (+2/-1); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+98/-0); python/sglang/srt/models/deepseek_v4.py (+6/-4); python/sglang/srt/server_args.py (+5/-1); test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py (+56/-1)
LABELS: deepseek, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ In deepseek v4, currently CP and MegaMoE cannot be enabled at the same time. However, theoretically there shouldn't be any conflicts because deepep and cp are compatible. Thus this pr remove the constraints and supports cp enabled together with megamoe. ⏎  ⏎ ## Modifications ⏎  ⏎ 1. Remove the deepep-only validation in server_args and deepseek_v4.py. ⏎ 2. Add arguments validation that chunk-prefill-size should be less than SGLANG_OPT_ …[truncated]

### L1-62aa85d9aa  (L1, 2026-07-23, sha 62aa85d9aa2f, PR #32160)
TITLE: [Kernel] Sweep missed dedicated kernels into kernels.ops (moe/quant siblings + dspark) (RFC #29630) (#32160)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/moe/gate_topk.py (+0/-0); python/sglang/kernels/ops/moe/inkling_moe.py (+0/-0); python/sglang/kernels/ops/moe/sigmoid_gate_topk_renorm.py (+5/-5); python/sglang/kernels/ops/quantization/mxfp8_interleave_sf.py (+0/-0); python/sglang/kernels/ops/quantization/mxfp8_quant.py (+0/-0); python/sglang/kernels/ops/speculative/dspark/__init__.py (+0/-0); python/sglang/kernels/ops/speculative/dspark/dispatch.py (+0/-0); python/sglang/kernels/ops/speculative/dspark/dspark_accept.py (+1/-1); python/sglang/kernels/ops/speculative/dspark/dspark_attn_metadata.py (+1/-1); python/sglang/kernels/ops/speculative/dspark/dspark_draft_model.py (+1/-1); (+15 more)
LABELS: quant, deepseek, run-ci, jit-kernel, bypass-maintenance, bypass-fastfail, run-ci-extra
BODY: ## Summary ⏎  ⏎ Post-migration audit follow-up: move genuinely-missed **dedicated kernel files** into `sglang.kernels.ops`, joining siblings that already migrated. Clean relocations only (git mv + caller import rewrites); no extraction from core/model modules. ⏎  ⏎ ## Changes ⏎  ⏎ - **moe** — `moe_runner/triton_utils/{inkling_moe, gate_topk, sigmoid_gate_topk_renorm}` → `sglang.kernels.ops.moe` (siblings `fused_moe_triton_kernels` / `triton_hash_topk` / `tri …[truncated]

### L1-b8bb1b4e5a  (L1, 2026-07-24, sha b8bb1b4e5a94, PR #31870)
TITLE: [lora] Fix WAR race: never write MoE runner output into hidden_states in place (#31870)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/lora/layers.py (+5/-0); test/registered/unit/lora/test_lora_moe_inplace_unit.py (+116/-0)
LABELS: lora, run-ci
BODY: ## Motivation ⏎  ⏎ - We have a long decode lora eval that improves with this fix. Generates some occasional gibberish without. ⏎  ⏎  ⏎  ⏎ ## Root cause ⏎  ⏎ A write-after-read race between the LoRA MoE runner and dual-stream shared experts: ⏎  ⏎ - With LoRA, the marlin MoE path writes its final reduction **into `hidden_states`**: `output = hidden_states if inplace else torch.empty_like(hidden_states)` with `MoeRunnerConfig.inplace=True` by default. The fus …[truncated]

### L1-b954e9cf3d  (L1, 2026-07-24, sha b954e9cf3dad, PR #30822)
TITLE: [6/6][kimi-deterministic] Use deterministic seeded coins for EAGLE rejection sampling (#30822)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.deep_gemm, L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-2); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+12/-3); python/sglang/kernels/ops/attention/flash_attention.py (+1/-0); python/sglang/kernels/ops/attention/flash_attention_v4.py (+5/-2); python/sglang/srt/batch_overlap/two_batch_overlap.py (+1/-0); python/sglang/srt/layers/attention/flashattention_backend.py (+4/-1); python/sglang/srt/layers/quantization/unquant.py (+0/-2); python/sglang/srt/layers/sampler.py (+15/-0); python/sglang/srt/model_executor/forward_batch_info.py (+13/-0); python/sglang/srt/models/deepseek_common/attention_backend_handler.py (+11/-4); (+9 more)
LABELS: quant, dependencies, deepseek, jit-kernel, bypass-fastfail
BODY: > **Stacked series (6/6, tip)**: stacked on `kimi-deterministic/5-logsoftmax-decode-logprobs` (#30821), so the diff below includes the whole series. This PR's own changes are the last 4 commits (ending 18f2f50bf — incl. the murmur_hash32 relocation to `sglang.kernels`). **CI on this PR covers the combined series.** ⏎  ⏎ ## Motivation ⏎  ⏎ EAGLE verify draws rejection-sampling coins from `torch.rand`, so speculative outputs are not reproducible even with  …[truncated]

### L1-39955d5314  (L1, 2026-07-24, sha 39955d531400, PR #29523)
TITLE: [MoE] Make DeepEP auto serve flashinfer_cutedsl FP4 (coerce to low_latency) + guard (#29523)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+16/-0)
LABELS: run-ci
ISSUES: #29521 [Bug] DeepEP normal (prefill) dispatch crashes flashinfer_cutedsl FP4 MoE: "not enough values to unpack (expected 6, got 5)"
BODY: ## Motivation ⏎  ⏎ Fixes #29521. ⏎  ⏎ Serving an NVFP4 / `modelopt_fp4` MoE model with the default high-throughput DeepEP `auto` mode crashes during the first prefill forward inside the `flashinfer_cutedsl` MoE runner: ⏎  ⏎ ``` ⏎ File ".../layers/moe/moe_runner/flashinfer_cutedsl.py", line 476, ⏎   in fused_experts_deepep_to_flashinfer_cutedsl_fp4 ⏎     hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output ⏎ ValueError: not enough values to unpack  …[truncated]

### L1-d4a0dfbc31  (L1, 2026-07-24, sha d4a0dfbc31ab, PR #32188)
TITLE: [Fix] Two root causes of the H100 deepep TBO CI break: scale-tensor use-after-free + missing non-finite quant sanitization (#32188)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh (+14/-3); python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py (+23/-1); test/registered/kernels/ops/quantization/test_per_token_group_quant.py (+76/-0)
LABELS: quant, jit-kernel
BODY: ## Summary ⏎  ⏎ Root-cause fixes for the H100 `deepep-4-gpu-h100` CI breakage (`TestTBOWithTPAttn`: `RuntimeError: The specified pointer resides on host memory and is not registered with any CUDA device` during CUDA graph capture, and `NaN detected! sampler: next_token_logits`) introduced by #30924. ⏎  ⏎ There are **two independent bugs**. This PR fixes both at their root, keeps the unified JIT quant kernel everywhere, and supersedes #32051 (whose Hopper …[truncated]

### L1-f15b43242b  (L1, 2026-07-24, sha f15b43242b43, PR #32345)
TITLE: Bump sgl-deep-gemm to 0.1.5 (#32345)
SOURCES: dependency_pin
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-8389d79e43  (L1, 2026-07-24, sha 8389d79e43af, PR #32125)
TITLE: ci: add LongCat-Flash-Lite-FP8 8-GPU nightly test + fix NextN rope_theta (#32125)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/longcat_flash_nextn.py (+6/-2); test/registered/8-gpu-models/test_longcat_flash_lite_fp8.py (+83/-0)
LABELS: run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Following the maintainer suggestion to add a CI test that exercises LongCat model support so future LongCat-related PRs are automatically regression-tested. ⏎  ⏎ The obvious candidate, **LongCat-2.0-FP8, is ~2 TB of FP8 weights** and needs **≥16 GPUs** (tp8 OOMs even on H200). No current single-node CUDA runner (`8-gpu-h200` is the largest) can load it, so a full 2.0 e2e is out of reach today. ⏎  ⏎ **LongCat-Flash-Lite-FP8** is the smalles …[truncated]

### L1-4d5917e744  (L1, 2026-07-24, sha 4d5917e74457, PR #31017)
TITLE: Add DeepSeek-reference 1e-20 epsilon to top-k renormalization to prevent 0/0 NaN (#31017)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+41/-12); test/registered/moe/test_topk_renormalize_degenerate.py (+141/-0)
LABELS: run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Fixes a latent 0/0 → NaN in the torch top-k renormalization paths — the same defect class as flashinfer-ai/flashinfer#3803 ("Fix 0/0 NaN in GLM52 Routing Renorm") and part of the fallout analyzed in #30989. ⏎  ⏎ With sigmoid scoring plus a selection bias (DeepSeek noaux_tc-style gates), experts are selected by `sigmoid(logits) + bias` but weighted by the raw sigmoid. For a token whose router logits are all deeply negative (< ~-88), eve …[truncated]

### L1-3d91a569ce  (L1, 2026-07-24, sha 3d91a569ce48, PR #30541)
TITLE: [MoE Backend] Add HPC-Ops FP8 MoE runner backend (#30541)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:production-kernel-provenance, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.framework
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+19/-0); python/sglang/srt/layers/moe/moe_runner/hpc_ops.py (+208/-0); python/sglang/srt/layers/moe/moe_runner/runner.py (+21/-1); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+2/-0); python/sglang/srt/layers/moe/utils.py (+4/-0); python/sglang/srt/layers/quantization/fp8.py (+77/-0); python/sglang/srt/server_args.py (+1/-0); python/sglang/srt/lora/mem_pool.py (+1/-0); test/registered/moe/test_hpc_ops_moe.py (+234/-0); test/registered/unit/layers/moe/test_hpc_ops_runner_guard.py (+63/-0); (+1 more)
LABELS: performance, run-ci, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Add a new opt-in MoE runner backend `hpc_ops` wrapping the monolithic FP8 fused-MoE kernels from [HPC-Ops](https://github.com/Tencent/hpc-ops), the production-grade operator library for LLM inference from the Tencent Hunyuan AI Infra team. ⏎  ⏎ This is the second half of the first-phase HPC-Ops <-> SGLang integration discussed with the Hunyuan team (attention backend: #30540). ⏎  ⏎ Enable with: ⏎  ⏎ ```bash ⏎ python3 -m sglang.launch_server \ ⏎    …[truncated]

### L1-b83041c3cc  (L1, 2026-07-24, sha b83041c3cc28, PR #32248)
TITLE: Migrate CompressedTensorsW4A4Nvfp4MoE TRT-LLM path onto MoeRunner (#32248)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py (+29/-86)
LABELS: blackwell, run-ci
BODY: ## Motivation ⏎  ⏎ The `use_flashinfer_trtllm` branch of `CompressedTensorsW4A4Nvfp4MoE` built a `MoeRunner(TRITON, ...)` that was never used, and hardcoded the FP4 kernel dispatch inline in `apply_weights`, bypassing the runner abstraction entirely (issue #20719). This diverges from `ModelOptNvFp4FusedMoEMethod`, which already routes the identical `trtllm_fp4_block_scale_moe` call through the shared `MoeRunner(FLASHINFER_TRTLLM)` path. ⏎  ⏎ ## Modificat …[truncated]

### L1-9402012f0f  (L1, 2026-07-25, sha 9402012f0f19, PR #32296)
TITLE: [Perf] Halve the non-finite sanitization overhead in per_token_group_quant (#32296)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh (+17/-14); test/registered/kernels/benchmark/quantization/bench_per_token_group_quant.py (+0/-1)
LABELS: quant, jit-kernel
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎ #32188 fixed the H100 deepep TBO CI break by sanitizing non-finite quant inputs with a **two-sided clamp** before the fp8 conversion. That fix is correct, but it costs more than it needs to: the trait-driven quant kernel is latency-bound at decode batch sizes, and the clamp added 16 `HMNMX2` (ue8m0 path) resp. 32 `FMNMX` (fp32-scale path) per bf16 group-of-128 — a 10–17% instruction-count increase on a kernel that runs twice per la …[truncated]

### L1-1054060ef1  (L1, 2026-07-25, sha 1054060ef119, PR #31552)
TITLE: perf: speed up marlin moe with occupancy-aware launch specialization (#31552)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.marlin
FILES: python/sglang/kernels/jit/csrc/gemm/marlin_moe/kernel.h (+3/-1); python/sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h (+48/-31); python/sglang/kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh (+42/-25); python/sglang/kernels/ops/moe/moe_wna16_marlin.py (+5/-3); test/registered/kernels/ops/moe/test_moe_wna16_marlin.py (+77/-0)
LABELS: run-ci, jit-kernel, run-ci-extra
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ - specialize Marlin MoE kernels for expert-parallel and bias launch paths at JIT compile time, removing per-block branches from the hot loop ⏎ - choose large-M Marlin launch configs using register/shared-memory occupancy and available problem parallelism instead of accepting the first valid config ⏎ - cover both bias/no-bias, small/large-M non-EP paths with a numerical reference test ⏎  ⏎ The scheduling and specialization changes are kept tog …[truncated]

### L1-833e1bc601  (L1, 2026-07-25, sha 833e1bc60159, PR #31793)
TITLE: [Fix][AMD] Qwen3.5 MoE: disable global-slot shared-expert fusion under per-rank EP backends (MoRI + dp-attention init crash) (#31793)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen2_moe.py (+15/-0)
LABELS: run-ci
ISSUES: #31336 [Bug] [AMD] Qwen3.5-397B-A17B-FP8 + MoRI + dp-attention crashes at init: FusedMoE shared-expert slot assertion (num_experts - num_shared_slots) % moe_ep_size
BODY: ## Motivation ⏎  ⏎ Fixes #31336. ⏎  ⏎ Serving **Qwen/Qwen3.5-397B-A17B-FP8** (512 routed + 1 fused shared expert) with `--moe-a2a-backend mori --enable-dp-attention` on MI355X/ROCm crashes at model init. Every scheduler rank aborts in `FusedMoE.__init__`: ⏎  ⏎ ``` ⏎ File ".../srt/layers/moe/fused_moe_triton/layer.py", line 227, ... ⏎     assert (num_experts - num_shared_slots) % self.moe_ep_size == 0 ⏎ AssertionError ⏎ ``` ⏎  ⏎ With `--enable-dp-attention`, `moe_ep_size …[truncated]

### L1-c0f47a06fc  (L1, 2026-07-27, sha c0f47a06fc8e, PR #31393)
TITLE: [NPU] Determine the topk norm_type through scoring_func (#31393)
SOURCES: path_core
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+1/-1); python/sglang/srt/models/glm4_moe_lite.py (+3/-0)
LABELS: npu, run-ci
BODY: ## Motivation ⏎ pr https://github.com/sgl-project/sglang/pull/29509 ⏎ it affected acc of ds coder v2 lite instruct ⏎ caz this model uses softmax func at topk, refactor topk part for npu. ⏎  ⏎  ⏎ ## Modifications ⏎ pr https://github.com/sgl-project/sglang/pull/29509 ⏎ it affected acc of ds coder v2 lite instruct ⏎ for most basic models with default softmax func, use 0, for special ones, set "sigmoid" at the model side. ⏎  ⏎ Add scoring_func as a parameter fo …[truncated]

### L1-169fc1e20c  (L1, 2026-07-27, sha 169fc1e20cbf, PR #31280)
TITLE: [NPU] Acc fix for afmoe model introduced by topk refactor. (#31280)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/afmoe.py (+9/-4)
LABELS: run-ci
BODY: ## Motivation ⏎ pr #29909 introduced the acc problems ⏎  ⏎ <img width="706" height="192" alt="image" src="https://github.com/user-attachments/assets/0b02bb1a-5d75-4f3d-a3b8-c37659a79ebc" /> ⏎  ⏎ Before with old version topk judgement, for this model, I set renormalize = False to multiply routed_scaling_factor in ops. ⏎ After this pr's change, trinity model could not * routed_scaling_factor, caz it didn't design the apply_routed_scaling_factor_on_output …[truncated]

### L1-8d6549bc40  (L1, 2026-07-27, sha 8d6549bc4039, PR #32304)
TITLE: [Attention Backend] Extend hpc_ops dynamic-scheduled decode to bf16 (#32304)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.framework
FILES: python/sglang/srt/layers/moe/moe_runner/hpc_ops.py (+4/-4); python/sglang/srt/layers/moe/moe_runner/runner.py (+9/-0); docker/Dockerfile (+1/-1); docs_new/docs/advanced_features/attention_backend.mdx (+1/-1); python/sglang/srt/layers/attention/hpc_ops_backend.py (+51/-25); python/sglang/srt/server_args.py (+2/-2); test/registered/moe/test_hpc_ops_moe.py (+5/-5)
LABELS: documentation, run-ci, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ [Tencent/hpc-ops#73](https://github.com/Tencent/hpc-ops/pull/73) (merged) extends the HPC-Ops SM90 decode kernels: dynamic scheduling is now available for **bf16** decode (previously FP8-only), backed by a new split-K combine shared across both dtypes — uneven KV lengths are packed into per-CTA tasks so long sequences no longer dominate decode latency. This PR wires the new capability into the `hpc_ops` attention backend. ⏎  ⏎ ## Modif …[truncated]

### L1-9cffc2ba52  (L1, 2026-07-28, sha 9cffc2ba526d, PR #32636)
TITLE: [Kernel] Remove unused implementations and stale registry entries (#32636)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.router_py
FILES: python/sglang/kernels/ops/moe/kpool_topk_transform.py (+0/-74); python/sglang/kernels/ops/moe/router.py (+1/-42); python/sglang/kernels/jit/csrc/dsa/kpool_topk_transform.cuh (+0/-440); python/sglang/kernels/jit/csrc/elementwise/resolve_future_token_ids.cuh (+0/-57); python/sglang/kernels/ops/attention/__init__.py (+0/-3); python/sglang/kernels/ops/attention/dsv4/compress_c128_hip.py (+0/-292); python/sglang/kernels/ops/attention/dsv4/fused_scale.py (+0/-51); python/sglang/kernels/ops/attention/dsv4/tilelang_kernel.py (+0/-123); python/sglang/kernels/ops/attention/fla/chunk_scaled_dot_kkt.py (+0/-147); python/sglang/kernels/ops/attention/fla/solve_tril.py (+0/-464); (+17 more)
LABELS: quant, deepseek, speculative-decoding, blackwell, run-ci, diffusion, Refactor, jit-kernel, run-ci-extra, kernel
BODY: ## Summary ⏎  ⏎ Follow-up cleanup for the unified `sglang.kernels` namespace from #29630. ⏎  ⏎ - Remove duplicate attention utility implementations and keep the canonical KV-cache implementations as compatibility re-exports. ⏎ - Delete 12 kernel modules with no production callers: six zero-reference modules, three registry-only modules, and three test/benchmark-only modules. ⏎ - Delete the five tests/benchmarks and two JIT CUDA headers that only supported th …[truncated]

### L1-5558dbad00  (L1, 2026-07-28, sha 5558dbad0037, PR #31931)
TITLE: [NPU] Optimize DeepSeek-V4 performance (#31931)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa, L1.ep.deepep_dispatcher
FILES: python/sglang/srt/hardware_backend/npu/moe/topk.py (+24/-21); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+7/-2); python/sglang/kernels/ops/attention/deepseek_v4_rope.py (+0/-120); python/sglang/srt/disaggregation/ascend/conn.py (+44/-8); python/sglang/srt/disaggregation/decode.py (+75/-19); python/sglang/srt/disaggregation/mooncake/conn.py (+18/-7); python/sglang/srt/disaggregation/prefill.py (+34/-18); python/sglang/srt/disaggregation/utils.py (+16/-1); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+3/-3); python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py (+359/-228); (+11 more)
LABELS: deepseek, npu, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ This PR improves DeepSeek-V4 serving performance on Ascend NPU by adding PD disaggregation and chunked prefill support, together with optimizations for the prefill and decode execution paths. ⏎  ⏎ All hardware-specific behavior is gated to the NPU + DeepSeek-V4 path, leaving CUDA, ROCm, and other model architectures unchanged. ⏎  ⏎ ## Modifications ⏎  ⏎ | Area | Changes | ⏎ | --- | --- | ⏎ | PD disaggregation | Represent the DSV4 SWA, C4, C128, in …[truncated]

### L1-f01a0c7f97  (L1, 2026-07-28, sha f01a0c7f97ec, PR #31510)
TITLE: Fixing MXFP8 online quantization pipeline (#31510)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+4/-2); python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py (+6/-2); python/sglang/srt/layers/quantization/fp8.py (+4/-2); python/sglang/srt/layers/quantization/fp8_utils.py (+8/-1); python/sglang/srt/layers/quantization/mxfp4.py (+7/-2); python/sglang/srt/models/deepseek_v2.py (+18/-12); python/sglang/srt/models/kimi_k25_eagle3.py (+1/-1)
LABELS: deepseek, run-ci
BODY: Otherwise, it will  ⏎  ⏎ Encounter error 1 of replacing the weights inplace when online MXFP8 quantization, which will run into error with fused a gemm dtype checking. Issue 1 is not caught by CI, as it's using Qwen3, not DSV3 (too large). ⏎  ⏎ ``` ⏎   File "/sgl-workspace/sglang/python/sglang/jit_kernel/dsv3_fused_a_gemm.py", line 50, in _dsv3_fused_a_gemm_run ⏎     module.dsv3_fused_a_gemm(mat_a, mat_b, output) ⏎   File "python/tvm_ffi/cython/function …[truncated]

### L1-d12ea3e9ba  (L1, 2026-07-29, sha d12ea3e9ba9f, PR #32760)
TITLE: docker: add Kimi K3 images (#32760)
SOURCES: path_core, dependency_pin, body_keyword
ARTIFACT_HINTS: -
FILES: docker/kimi_k3/apply_deepep_k3_patch.sh (+169/-0); docker/kimi_k3/apply_deepgemm_situ_patch.py (+140/-0); docker/kimi_k3/flashinfer-perkz-dcp-0.6.15.txt (+5639/-0); docker/kimi_k3/kimi_k3_cu12.Dockerfile (+136/-0); docker/kimi_k3/kimi_k3_cu13.Dockerfile (+125/-0); docker/rocm.Dockerfile (+4/-4)
LABELS: amd
BODY: ## Summary ⏎  ⏎ Extract the Docker-only changes from #32541 into a dedicated pull request. ⏎  ⏎ - add CUDA 12 and CUDA 13 Kimi K3 Dockerfiles ⏎ - add the associated DeepEP, DeepGEMM, and FlashInfer patch inputs ⏎ - include the Kimi K3 ROCm Dockerfile update ⏎  ⏎ ## Why ⏎  ⏎ Keeping image build changes separate lets the Kimi K3 runtime support and image publication review independently. ⏎  ⏎ ## Validation ⏎  ⏎ - `git diff --check` ⏎ - repository pre-commit hooks ⏎  ⏎ --- ⏎ ### CI St …[truncated]

### L1-f05c92fb6d  (L1, 2026-07-29, sha f05c92fb6d65, PR #30768)
TITLE: :sparkles: [llm][npu][quant] Add W8A8 MXFP8 quantization for Qwen3 MoE on Ascend NPU (#30768)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.hardware.cpu_npu_musa
FILES: python/sglang/srt/hardware_backend/npu/moe/init_routing.py (+20/-0); python/sglang/srt/hardware_backend/npu/moe/matmul.py (+40/-0); python/sglang/srt/hardware_backend/npu/moe/quant.py (+15/-4); python/sglang/srt/layers/moe/moe_runner/ascend.py (+43/-20); python/sglang/srt/layers/moe/token_dispatcher/ascend_tp.py (+5/-0); python/sglang/srt/layers/moe/utils.py (+2/-0); docs_new/docs/advanced_features/quantization.mdx (+2/-2); docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_quantization.mdx (+36/-3); python/sglang/srt/arg_groups/overrides.py (+7/-1); python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py (+240/-4); (+8 more)
LABELS: documentation, quant, npu, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: # Summary ⏎  ⏎ Adds **W8A8 MXFP8** quantization (8-bit weights + 8-bit activations, block_size = 32) for Qwen3 / Qwen3.5 **MoE** (FusedMoE) LLM models on Ascend NPU, continuing the NPU quantization work tracked in #21584. Both **online** (`--quantization mxfp8`) and **offline** (msmodelslim, flagless — selected by the checkpoint's `quant_model_description.json`) paths are supported and share the same fused-experts kernel. Requires Ascend A5 (Ascend …[truncated]

### L1-8fc54d46ef  (L1, 2026-07-29, sha 8fc54d46eff5, PR #32663)
TITLE: Fix MoE reduce-scatterv eligibility check (#32663)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+6/-0); test/registered/unit/test_runtime_context.py (+17/-0)
LABELS: run-ci, bypass-fastfail, run-ci-extra
BODY: ## Summary ⏎  ⏎   - Enable DP reduce-scatterv only when the TP group size matches the attention-DP split count. ⏎   - Fall back to all-reduce plus DP scatter when the group sizes do not match. ⏎   - Add unit tests covering both valid and fallback paths ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-requ …[truncated]

### L1-e5c46ff07d  (L1, 2026-07-29, sha e5c46ff07d78, PR #32818)
TITLE: [Fix] Route asymmetric-KV models to fa4 on SM100 and pin MiMoV2 FP8 MoE to flashinfer_trtllm (#32818)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+4/-0); docs_new/cookbook/autoregressive/Xiaomi/MiMo-V2.5.mdx (+2/-0); docs_new/src/snippets/autoregressive/mimo-v25-deployment.jsx (+7/-3); python/sglang/srt/arg_groups/overrides.py (+13/-2); python/sglang/srt/configs/model_config.py (+11/-0); test/registered/unit/test_model_overrides.py (+51/-16)
LABELS: documentation
BODY: MiMo-V2.5 cannot be deployed on SM100/SM103 with default flags (#31243). The SM100 auto-dispatch picks `trtllm_mha`, whose paged-KV kernel requires equal K/V row widths, but MiMoV2 has asymmetric KV (`head_dim` 192, `v_head_dim` 128), so decode CUDA-graph capture aborts with `Check failed: key_cache.size(i) == value_cache.size(i) (37074 vs. 24716)`. The reporter had to fall back to triton with CUDA graphs disabled. ⏎  ⏎ Three changes: ⏎  ⏎ 1. `ModelConfi …[truncated]

### L1-3c9efaf3e1  (L1, 2026-07-29, sha 3c9efaf3e192, PR #32834)
TITLE: [docs] Kimi-K3: widen the H200 High-Throughput recipe to 4x8 TP32/EP32 (#32834)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx (+3/-3); docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx (+21/-9)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ The Kimi-K3 cookbook's **H200 / Unified / High-Throughput** cell was a copy of Balanced (2×8 TP16/EP16) with `--mem-fraction-static 0.90` and `extra_buffer_lazy`. The shape actually run for that operating point is **TP32/EP32 across 4 nodes**, so the page shipped a recipe nobody serves. ⏎  ⏎ ## Modifications ⏎  ⏎ `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx` — the H200 Unified High-Throughput cell: ⏎  ⏎ - `nnodes: 2 → 4`, `--tp-size`/ …[truncated]

### L1-c32c4ef79c  (L1, 2026-07-29, sha c32c4ef79cf5, PR #32648)
TITLE: [Kernel] Move sgl-kernel under sglang.kernels.aot (#32648)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.align.cuda_aot, L1.routing.topk_softmax, L1.routing.topk_sigmoid, L1.hardware.cpu_npu_musa, L1.cutlass.fp8_blockwise, L1.cutlass.w4a8, L1.reduce.moe_sum_reduce, L1.reduce.moe_sum, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: docker/arm64.Dockerfile (+1/-1); docker/rocm.Dockerfile (+2/-2); docker/xeon.Dockerfile (+1/-1); python/pyproject.toml (+8/-0); python/sglang/kernels/aot/CMakeLists.txt (+0/-0); .claude/skills/add-jit-kernel/SKILL.md (+1/-1); .claude/skills/add-sgl-kernel/SKILL.md (+38/-38); .claude/skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md (+1/-1); .claude/skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py (+2/-2); .github/CODEOWNERS (+2/-2); (+360 more)
LABELS: documentation, quant, amd, dependencies, Multi-modal, deepseek, speculative-decoding, sgl-kernel, npu, run-ci
BODY: ## Summary ⏎  ⏎ - Move the standalone `sgl-kernel` source tree to `python/sglang/kernels/aot` so SGLang's AOT and JIT kernel implementations live under one kernel namespace. ⏎ - Preserve the published distribution name (`sglang-kernel`), Python import (`sgl_kernel`), operator registrations, versioning, API, and ABI. ⏎ - Update CUDA/ROCm/CPU/ARM build, test, release, Docker, CI, CODEOWNERS, labeler, and developer paths to the new source location. ⏎ - Keep t …[truncated]

### L1-d004a15a3e  (L1, 2026-07-29, sha d004a15a3edf, PR #32371)
TITLE: Fix GLM4-7B-Flash accuracy test configuration, tune Qwen3.6-27B/35B performance test parameters, and harden Ascend NPU multi-node E2E test utilities against pod name format errors. (#32371)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/test/ascend/e2e/run_npu_e2e_test.py (+3/-1); python/sglang/test/ascend/e2e/test_npu_multi_node_utils.py (+20/-3); test/registered/ascend/accuracy/glm4_7_flash/test_npu_glm4_7_flash_1p_aime25.py (+3/-3); test/registered/ascend/accuracy/qwen3_6_27b/test_npu_qwen3_6_27b_1p_gpqa.py (+4/-0); test/registered/ascend/performance/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_2p_in16k_out1k_50ms.py (+25/-8); test/registered/ascend/performance/qwen3_6_35b_a3b/test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms.py (+1/-1)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ Fix GLM4-7B-Flash accuracy test configuration, tune Qwen3.6-27B/35B performance test parameters, and harden Ascend NPU multi-node E2E test utilities against pod name format errors. ⏎  ⏎ ## Modifications ⏎  ⏎ 6 files changed, +56/-16 ⏎  ⏎ ### 1. E2E multi-node test robustness (run_npu_e2e_test.py, test_npu_multi_node_utils.py) ⏎ - run_npu_e2e_test.py : Add label_selector="app=sgl-ascend" to pod listing API call to filter only target pods …[truncated]

### L1-a55e1764a2  (L1, 2026-07-30, sha a55e1764a2e1, PR #32668)
TITLE: Enable GPT-OSS FlashInfer MXFP4 on SM120 (#32668)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/overrides.py (+3/-3); python/sglang/srt/layers/quantization/mxfp4.py (+139/-13); test/registered/unit/layers/quantization/test_mxfp4_sm120_cutlass.py (+187/-1)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ Enable FlashInfer CUTLASS MXFP4 MoE for GPT-OSS on SM120 and make it the default based on the performance results below ⏎  ⏎ ## Accuracy ⏎  ⏎ ```bash ⏎ sglang serve --model-path openai/gpt-oss-20b --trust-remote-code ⏎ ``` ⏎  ⏎ ```bash ⏎ python -m gpt_oss.evals --model openai/gpt-oss-20b --eval gpqa --n-threads 256 --reasoning-effort low --base-url http://127.0.0.1:30000/v1 ⏎ ``` ⏎  ⏎ Result: `0.5751262626262627` ⏎  ⏎ ## Performance ⏎  ⏎ Marlin: ⏎  …[truncated]

### L1-fc007e1f00  (L1, 2026-07-30, sha fc007e1f00fd, PR #29016)
TITLE: Add SM90 FP8 MegaMoE support for DeepSeek-V4 (#29016)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L1.runner.deepgemm_megamoe
FILES: python/sglang/srt/layers/moe/mega_moe.py (+18/-0); python/sglang/srt/layers/moe/mega_moe_sm90.py (+179/-0); docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+31/-2); python/sglang/srt/layers/quantization/fp8.py (+9/-0); test/registered/models_e2e/test_deepseek_v4_flash_fp8_h200.py (+72/-1)
LABELS: documentation, deepseek, run-ci, jit-kernel, run-ci-extra, release-highlight
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎ This PR adds SM90 DeepGEMM MegaMoE adaptation for DeepSeek-V4 FP8 serving. It enables the MegaMoE A2A path with the DeepGEMM MoE runner on SM90, including the pre-dispatch JIT kernel, FP8 expert weight preparation, and DeepSeek-V4 integration. ⏎  ⏎ The goal is to improve long-context / large decode serving throughput for DeepSeek-V4-Flash/Pro-FP8 workloads while keeping the path guarded by environment variables. ⏎  ⏎ ## Modifications ⏎  …[truncated]

### L1-c4af6cf263  (L1, 2026-07-30, sha c4af6cf26397, PR #31220)
TITLE: Qwen3.5-MoE: support modelopt_fp4 checkpoints that quantize attention (+ load baked FP8 KV scales) (#31220)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen3_5.py (+20/-14); test/registered/unit/models/test_qwen3_5_modelopt_fp4.py (+157/-0)
LABELS: bug, run-ci
BODY: ## Motivation ⏎  ⏎ `#18937` ("[Qwen3.5] Enable nvfp4 checkpoint") added SGLang support for ⏎ NVIDIA's released `nvidia/Qwen3.5-397B-A17B-NVFP4` checkpoint. That ⏎ checkpoint quantizes only the MoE experts and excludes attention ⏎ (attention ships in BF16), so the PR hard-forced `quant_config` to ⏎ `None` for the linear-attention and full-attention modules whenever ⏎ `quant_config.get_name() == "modelopt_fp4"`, and never wired FP8 KV ⏎ scale loading for attention …[truncated]

### L1-a6221d776f  (L1, 2026-07-30, sha a6221d776fe1, PR #31989)
TITLE: feat: Support nvidia/MiniMax-M3-NVFP4 (#31989)
SOURCES: path_core
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+6/-4); python/sglang/srt/arg_groups/overrides.py (+5/-0); python/sglang/srt/layers/quantization/modelopt_quant.py (+46/-12); python/sglang/srt/models/minimax_m3.py (+18/-0); python/sglang/srt/models/minimax_m3_vl.py (+18/-0); test/registered/unit/model_loader/test_modelopt_loader.py (+63/-0)
LABELS: bug, quant, run-ci
ISSUES: #31827 [Bug] MiniMax-M3 NVFP4 fails to load: fused MoE falls back to UnquantizedFusedMoEMethod (6144 vs 3072) and MXFP8 linear scales are dropped
BODY: ## Motivation ⏎  ⏎ Fixes https://github.com/sgl-project/sglang/issues/31827 ⏎  ⏎ Similar to `nvidia/DeepSeek-V4-NVFP4`, `nvidia/MiniMax-M3-NVFP4` is a hybrid quantized model with only certain layers in NVFP4. This time the MoE routed experts are in NVFP4 and other layers in MXFP8. We didn't handle the MXFP8 layers properly. ⏎  ⏎ Example command: ⏎ ``` ⏎ sglang serve --model-path nvidia/MiniMax-M3-NVFP4 --trust-remote-code --tp 8 --context-length 2048 --m …[truncated]

### L1-e4a40a71f8  (L1, 2026-07-30, sha e4a40a71f84c, PR #31888)
TITLE: [DSA] Q8KV8 FP8 Sparse Prefill on GLM-5.2 & DeepSeek-V3.2: Q8-Path & Shared-Path Optimizations (#31888)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.triton.helper_kernels, L1.runner.triton, L1.runner.deep_gemm, L1.ep.layer
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+21/-0); python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py (+7/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+30/-4); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+18/-3); python/sglang/srt/layers/moe/moe_runner/triton.py (+11/-1); python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (+28/-1); python/sglang/srt/layers/moe/token_dispatcher/standard.py (+24/-4); benchmark/kernels/deepseek/benchmark_q8kv8_kv_gather.py (+262/-0); benchmark/kernels/deepseek/benchmark_q8kv8_q_prep.py (+486/-0); docs_new/cookbook/autoregressive/GLM/GLM-5.2.mdx (+1/-1); (+20 more)
LABELS: documentation, quant, deepseek, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: > Part of the Q8KV8 FP8 sparse-attention roadmap #25746; follow-up to #25751 (kernel) and ⏎ > #30514 (DSA integration), both now in main. Throughout, ⏎ > **q8** = the Q8KV8 fp8 sparse-prefill backend and **q16** = the bf16-sparse FlashMLA ⏎ > baseline; both arms always run the fp8 KV cache. Scope: **GLM-5.2 + DeepSeek-V3.2** (shared ⏎ > levers measured on both; the q8-only levers apply to any DSA architecture using this backend). ⏎  ⏎ ## TL;DR ⏎  ⏎ - **G …[truncated]

### L1-92b3a51ba6  (L1, 2026-07-30, sha 92b3a51ba619, PR #32884)
TITLE: [LoRA] Fix Marlin MoE kernel import (#32884)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L1.runner.marlin
FILES: python/sglang/srt/lora/marlin_lora_temp/moe_runner.py (+3/-3)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ - Import `moe_sum_reduce_triton` from its current `sglang.kernels.ops.moe` location. ⏎  ⏎ ## Root cause ⏎  ⏎ The fused MoE Triton kernels moved out of `srt.layers.moe.moe_runner.triton_utils`, but the experimental Marlin MoE-LoRA runner retained the old import path. Selecting `experimental_sgl_marlin` therefore raised `ModuleNotFoundError` during server initialization. ⏎  ⏎ ## Validation ⏎  ⏎ - `pre-commit run --all-files` ⏎ - 10 paired base …[truncated]

### L1-3a53c26c27  (L1, 2026-07-30, sha 3a53c26c27c8, PR #32937)
TITLE: [CI] Fix MoE compile and DSA indexer regressions (#32937)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_native.py (+2/-1); test/registered/kernels/ops/attention/test_dsa_indexer.py (+1/-0)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ Fixes two regressions from #31888. Both actually failed in its final base CI run before the PR was force-merged: ⏎  ⏎ - `test_torch_compile_moe.py` failed with `ValueError: too many values to unpack (expected 3)` after `StandardDispatchOutput` gained a fourth field. Read the required fields by name. ⏎   CI: https://github.com/sgl-project/sglang/actions/runs/30411124003/job/90771215458 ⏎  ⏎ - `TestDSAIndexer.test_forward_decode_mode` fail …[truncated]

### L1-a1c30701aa  (L1, 2026-07-30, sha a1c30701aade, PR #30756)
TITLE: Integrate pplx a2a backend (#30756)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.ep.layer, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+1/-0); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+8/-0); python/sglang/srt/layers/moe/token_dispatcher/pplx.py (+527/-0); python/sglang/srt/layers/moe/utils.py (+10/-2); docs_new/docs/advanced_features/expert_parallelism.mdx (+7/-2); docs_new/docs/advanced_features/server_arguments.mdx (+2/-2); docs_new/docs/references/environment_variables.mdx (+5/-0); python/sglang/srt/arg_groups/overrides.py (+10/-1); python/sglang/srt/batch_overlap/two_batch_overlap.py (+5/-0); (+6 more)
LABELS: documentation, quant, deepseek, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Support [`pplx-kernels`](https://github.com/perplexityai/pplx-kernels) as MoE all-to-all backend. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Add `--moe-a2a-backend pplx` ⏎  ⏎ Only support `--moe-runner-backend deepgemm` ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ``` ⏎ SGLANG_PPLX_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 \ ⏎ python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B --trust-remote-code \ ⏎   --tp 4 --dp 4 --enable-dp-attention --enable-dp-lm-hea …[truncated]

### L1-48dbc24cbf  (L1, 2026-07-30, sha 48dbc24cbff1, PR #31382)
TITLE: [Qwen3.5][MTP] Support FlashInfer CuTe DSL for online NVFP4 draft MoE (#31382)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+2/-1); python/sglang/srt/configs/model_config.py (+20/-9); python/sglang/srt/layers/quantization/modelopt_quant.py (+26/-13); python/sglang/srt/layers/quantization/nvfp4_online.py (+10/-4); python/sglang/srt/model_executor/runner/flashinfer_autotune.py (+16/-11)
LABELS: quant, blackwell
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ Qwen3.5 ModelOpt FP4 checkpoints keep the embedded MTP draft expert weights in BF16. `nvfp4_online` converts these weights to NVFP4 during loading, but previously supported only the FlashInfer TRT-LLM MoE runners. ⏎  ⏎ This PR enables the existing FlashInfer CuTe DSL MoE paths for the online-quantized draft model. ⏎  ⏎ ## Modifications ⏎  ⏎ - Preserve explicit `nvfp4_online` selection for the embedded Qwen3.5 MTP draft model. ⏎ - Support `flashi …[truncated]

### L1-b78d3999b5  (L1, 2026-07-30, sha b78d3999b54b, PR #32791)
TITLE: 【NPU】fix decode MTP + eagle shape error (#32791)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+48/-0)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ 1 In Eagle mode, init_forward_metadata is invoked first, followed by _prepare_eager_forward_batch which performs padding. This execution sequence causes q to be padded, while the KV stored in forward_metadata retains values from before padding, resulting in a shape mismatch. ⏎ 2 The topk_indices from draft_extend_for_decode can be reused in the draft step. However, the draft's q may undergo padding, which causes a shape mismatch b …[truncated]

### L1-06ccaef24a  (L1, 2026-07-30, sha 06ccaef24afe, PR #32962)
TITLE: Fix silently wrong EPLB output with --moe-a2a-backend none (rank-invariant dispatch) (#32962)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/eplb/expert_location.py (+5/-2); python/sglang/srt/eplb/expert_location_dispatch.py (+36/-6); python/sglang/srt/server_args.py (+20/-1); test/registered/ep/test_eplb_no_a2a.py (+110/-0); test/registered/unit/eplb/test_compute_logical_to_rank_dispatch_physical_map.py (+12/-3)
LABELS: documentation, deepseek, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ EPLB runs without complaint on the no-a2a MoE path (`--moe-a2a-backend none`, the ⏎ default), but silently corrupts the output as soon as it has redundant experts to ⏎ place. On `lmsys/sglang-ci-dsv3-test` with `--tp 2 --ep-size 2 --enable-eplb ⏎ --ep-num-redundant-experts 48`, GSM8K drops from **~0.63 to 0.42**. No error, no ⏎ warning. ⏎  ⏎ Without an a2a backend every EP rank runs the MoE over the **same** tokens and ⏎ owns only a slice of the …[truncated]

### L1-301ea43f35  (L1, 2026-07-31, sha 301ea43f35b9, PR #32719)
TITLE: [CI] Re-enable GB300 CI jobs (#32719)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/nightly-test-nvidia.yml (+64/-65); .github/workflows/pr-test.yml (+14/-15); .github/workflows/release-whl-deepgemm.yml (+3/-4)
LABELS: run-ci
BODY: Both GB300 runners are online again with the `4-gpu-gb300` label, so the jobs disabled in #31764 can be restored. ⏎  ⏎ `gb300-4-gpu-03` had lost its pod entirely and was recreated on a different GB300 host, this time with `restartPolicy: Always` so a host reboot can't kill it permanently. Smoke-tested on the real install path (`GRACE_BLACKWELL=1 ci_install_deepep.sh`): `test_numa_utils` and `test_flashinfer_comm_fusion` both pass, and all 4 GPUs veri …[truncated]

### L1-55b6769b0e  (L1, 2026-07-31, sha 55b6769b0ede, PR #33013)
TITLE: config: read resolved config via namespace accessors (#33013)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.routing.hash_topk, L1.hardware.cpu_npu_musa, L1.ep.other_dispatchers
FILES: python/sglang/srt/hardware_backend/npu/moe/fuseep.py (+4/-4); python/sglang/srt/arg_groups/overrides.py (+6/-11); python/sglang/srt/batch_overlap/two_batch_overlap.py (+4/-4); python/sglang/srt/configs/inkling.py (+2/-2); python/sglang/srt/debug_utils/dumper.py (+2/-1); python/sglang/srt/disaggregation/base/conn.py (+1/-0); python/sglang/srt/disaggregation/common/conn.py (+6/-5); python/sglang/srt/disaggregation/decode.py (+7/-6); python/sglang/srt/disaggregation/encode_grpc_server.py (+4/-3); python/sglang/srt/disaggregation/encode_server.py (+15/-15); (+177 more)
LABELS: Multi-modal, deepseek, speculative-decoding, ready-to-merge, blackwell, npu, mthreads, apple-silicon
BODY: Part 3 of the 3-PR stack (base: #33012). RFC: #30696. Re-lands the reader migration reverted in #32100, regenerated from scratch against current main with the revert's defects fixed at their origin. ⏎  ⏎ - Mechanical sweep (AST-based, alias-aware): `get_server_args().FIELD` / `self.server_args.FIELD` / local-alias reads flip to the namespace accessors (`get_exec()` / `get_memory()` / …), routed by each field's NS metadata — 628 reads across 160 files …[truncated]

### L1-3e0f7c3f30  (L1, 2026-07-31, sha 3e0f7c3f30e2, PR #31987)
TITLE: [BCG][3/N] Enable bcg on dsa & deepep a2a backend (#31987)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+66/-1); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+98/-42); python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py (+1/-0); python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py (+19/-5); python/sglang/srt/models/deepseek_v2.py (+8/-0); python/sglang/srt/server_args.py (+53/-28); python/sglang/benchmark/one_batch.py (+1/-0); python/sglang/srt/managers/scheduler.py (+9/-0); python/sglang/srt/managers/scheduler_components/dp_attn.py (+42/-19); python/sglang/srt/speculative/eagle_worker_common.py (+5/-5); (+3 more)
LABELS: deepseek, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-1496bfee93  (L1, 2026-07-31, sha 1496bfee93bc, PR #32828)
TITLE: [Kimi] Support DCP + DSpark (ported from kimi-k3 branch) (#32828)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+15/-3); python/sglang/srt/mem_cache/common.py (+6/-13); python/sglang/srt/mem_cache/kv_cache_configurator.py (+10/-0); python/sglang/srt/model_executor/pool_configurator.py (+1/-1); python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py (+64/-0); test/registered/dcp/test_kimi_linear_dcp_dspark4.py (+222/-0); test/registered/dcp/test_tokenspeed_mla_dcp_metadata.py (+87/-0); test/registered/unit/mem_cache/test_paged_free_segment.py (+40/-0)
LABELS: high priority, run-ci, bypass-fastfail, run-ci-extra
BODY: Port the Kimi Linear DCP + DSPARK feature from the `kimi-k3` branch, and add its test coverage. ⏎  ⏎ `main` had the DCP half (#32612) but not the DSPARK half for a hybrid linear-attention target: the target's KDA/mamba recurrent state was never committed to the accepted length after verify, so it drifted from the accepted token sequence every step. ⏎  ⏎ ## Changes ⏎  ⏎ - `speculative/dspark_components/dspark_worker_v2.py` — commit the last accepted ver …[truncated]

### L1-94743f934c  (L1, 2026-07-31, sha 94743f934c98, PR #33083)
TITLE: [Docs] Add DeepSeek-V4 Flash Official (0731) recipe (#33083)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+47/-5); docs_new/src/snippets/_playground.jsx (+47/-14); docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx (+32/-1); docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+431/-2)
LABELS: documentation, deepseek
BODY: ## Summary ⏎  ⏎ - add a separate **Flash Official** option for `deepseek-ai/DeepSeek-V4-Flash-0731` ⏎ - add the measured 4xGB300 FP4 low-latency DSpark recipe without pinning a revision in generated commands ⏎ - expand Flash Official to a 22-cell FP4 hardware/strategy matrix derived from the existing Flash matrix but adapted to the checkpoint: eligible CUDA cells use DSpark, while DP-Attention and AMD cells run target-only ⏎ - add a dedicated DSpark sectio …[truncated]

### L1-0d186f49be  (L1, 2026-07-31, sha 0d186f49be34, PR #33090)
TITLE: [AMD][Fix] Restore aiter-padded MoE weight dims for serialized checkpoints (#33090)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py (+11/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `dcd9014f15` ("[AMD][MXFP4] Reland Online MXFP4 quantization 2/N - FP8 to MXFP4 requantization on AMD GPUs", #28291) silently broke accuracy for **serialized** MXFP4 Quark MoE checkpoints on ROCm whenever the per-rank MoE intermediate size needs aiter padding. ⏎  ⏎ On Qwen3.5-397B-A17B-MXFP4 at TP8 on MI355 this drops GSM8K from **0.99 to 0.54**. The failure is not obvious garbage — output stays fluent English, but the model loses know …[truncated]

### L1-a1344fad4e  (L1, 2026-07-31, sha a1344fad4e59, PR #33143)
TITLE: Replace Kimi K3 DeepGEMM patch with 0.1.5.post1 (#33143)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/kimi_k3/apply_deepgemm_situ_patch.py (+0/-140); docker/kimi_k3/kimi_k3_cu12.Dockerfile (+9/-4); docker/kimi_k3/kimi_k3_cu13.Dockerfile (+9/-5)
BODY: ## Summary ⏎  ⏎ Apply #33066 to `main` ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #30673759902](https://github.com/sgl-project/sglang/actions/runs/30673759902) ⏎ Latest PR Test (Extra): :x: [Run #30673759743](https://github.com/sgl-project/sglang/actions/runs/30673759743)

### L1-47d8b5b749  (L1, 2026-08-01, sha 47d8b5b749fd, PR #33170)
TITLE: config: route parallel config-leaf reads through get_parallel() (#33170)
SOURCES: path_core
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.runner.flashinfer_cutedsl, L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+2/-3); python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py (+4/-1); python/sglang/srt/layers/moe/token_dispatcher/nixl.py (+3/-5); python/sglang/srt/layers/moe/token_dispatcher/pplx.py (+2/-2); python/sglang/srt/layers/moe/utils.py (+1/-1); python/sglang/srt/batch_overlap/two_batch_overlap.py (+1/-1); python/sglang/srt/disaggregation/common/conn.py (+10/-13); python/sglang/srt/disaggregation/mooncake/conn.py (+2/-2); python/sglang/srt/distributed/device_communicators/triton_symm_mem_ag.py (+4/-4); python/sglang/srt/elastic_ep/elastic_ep.py (+2/-4); (+71 more)
LABELS: amd, deepseek, ready-to-merge
BODY: Part 1/4 of the config-namespace follow-up stack (RFC: #30696; follows the merged #33011–#33013). Based on #33168 (the chunked-prefix gate fix) so the stack tests with it; merge #33168 first, then the members in order. ⏎  ⏎ ## Motivation ⏎  ⏎ The parallel namespace was the last reader family left on `get_server_args()`: 106 config-leaf reads (`enable_dp_lm_head`, `enable_dp_attention`, `pp_async_batch_depth`, `dp_size`, `ep_join_rank_offset`, `dwdp_size` …[truncated]

### L1-fb207b72b0  (L1, 2026-08-01, sha fb207b72b02a, PR #32890)
TITLE: feat(kernels): port standalone Kimi K3 kernels (#32890)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L1.routing.fused_gate, L1.reduce.topk_sum_jit
FILES: python/sglang/kernels/jit/csrc/moe/route_quant_fused.cuh (+142/-0); python/sglang/kernels/jit/csrc/moe/route_radix.cuh (+753/-0); python/sglang/kernels/jit/csrc/moe/topk_sum.cuh (+103/-0); python/sglang/kernels/jit/csrc/attention/fixup_zero_kv.cuh (+39/-24); python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh (+1064/-0); python/sglang/kernels/jit/csrc/attention/kda_packed_decode.cuh (+240/-0); python/sglang/kernels/jit/csrc/attention/kda_prefill.cu (+3871/-0); python/sglang/kernels/jit/csrc/elementwise/add3.cuh (+93/-0); python/sglang/kernels/jit/csrc/elementwise/concat_mla.cuh (+20/-13); python/sglang/kernels/jit/csrc/elementwise/set_mla_kv_buffer.cuh (+3/-36); (+74 more)
LABELS: documentation, high priority, quant, Multi-modal, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Port the standalone kernels from the Kimi K3 Day0 work to `main` first. Keeping ⏎ the kernels and their direct tests separate makes the reusable pieces reviewable ⏎ before the more invasive model, scheduler, and serving integration changes. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add the shared JIT/TMA support and generic prerequisite kernels used by Kimi ⏎   K3. ⏎ - Port the standalone KDA kernels, including ReplaySSM, packed decode, and the ⏎   NVIDIA/PTX pr …[truncated]

### L1-e2cf21b9e5  (L1, 2026-08-01, sha e2cf21b9e561, PR #33025)
TITLE: [Kimi K3] Add reasoning, tool-call, and OpenAI serving support (#33025)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+1/-1); python/sglang/srt/constrained/base_grammar_backend.py (+2/-2); python/sglang/srt/constrained/grammar_manager.py (+1/-1); python/sglang/srt/constrained/reasoner_grammar_backend.py (+63/-55); python/sglang/srt/entrypoints/openai/chat_encoding.py (+5/-1); python/sglang/srt/entrypoints/openai/protocol.py (+34/-25); python/sglang/srt/entrypoints/openai/serving_chat.py (+272/-42); python/sglang/srt/entrypoints/openai/serving_responses.py (+26/-10); python/sglang/srt/function_call/base_format_detector.py (+15/-1); python/sglang/srt/function_call/function_call_parser.py (+22/-3); (+24 more)
LABELS: high priority, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Summary ⏎  ⏎ - add Kimi K3 XTML reasoning and native tool-call parsers ⏎ - add XGrammar structural constraints for automatic, required, and named tool choice ⏎ - auto-detect both parsers from Kimi K3 model configuration ⏎ - support Kimi K3 rendering in Chat Completions and Responses, including dynamic message tools, wire-format tool and response schemas, multimodal prompt IDs, and native tool-call IDs ⏎ - preserve the reasoning state resolved during templ …[truncated]

### L1-ae84811666  (L1, 2026-08-01, sha ae848116662e, PR #33109)
TITLE: [Docs] Add verified H200 and B200 DeepSeek-V4 Flash Official results (#33109)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+1/-1); docs_new/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx (+60/-0); docs_new/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+13/-11)
LABELS: documentation, deepseek
BODY: ## Summary ⏎  ⏎ - follow up on #33083 with completed Flash Official FP4 validation on 4xH200 and 8xB200 ⏎ - mark all three H200 and B200 strategies verified and publish twelve serving benchmark records from SGLang v0.5.16 ⏎ - switch the B200 Flash Official recipes to the measured TP8/DP8 configurations ⏎ - add `--mem-fraction-static 0.88` for H200 balanced and `0.90` for B200 low-latency to preserve runtime workspace headroom ⏎  ⏎ ## H200 results ⏎  ⏎ Test environ …[truncated]

### L1-37be4e9247  (L1, 2026-08-01, sha 37be4e924775, PR #32315)
TITLE: [AMD] Speed up DSV4 MoE weight loading from mmap views (#32315)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+26/-0); .github/workflows/nightly-test-amd-rocm720.yml (+9/-0); .github/workflows/nightly-test-amd.yml (+1/-0); .github/workflows/pr-test-amd-rocm720.yml (+2/-0); python/sglang/srt/environ.py (+3/-0); test/registered/unit/layers/moe/test_copy_weight_views_before_h2d.py (+50/-0)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ DeepSeek-V4-Pro TP8 model loading was bottlenecked by 140,544 small MoE H2D copies per rank. Rank-local tensor views retained much larger safetensors mmap-backed storage, causing TP0/TP7 to spend 27-32 minutes in H2D while the other ranks took about 4 minutes. ⏎  ⏎ This PR optionally copies only oversized or non-contiguous CPU weight views into independent contiguous storage immediately before H2D. It is disabled by default via `SGLANG_MO …[truncated]

### L1-00a219f6c9  (L1, 2026-08-02, sha 00a219f6c9e5, PR #32843)
TITLE: [Quant] Keep the flashinfer_deepgemm FP8 GEMM to 1 <= M < 32 (#32843)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8_utils.py (+21/-1)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ `--fp8-gemm-backend flashinfer_deepgemm` routes every dense block-FP8 linear through ⏎ `flashinfer.gemm.fp8_blockscale_gemm_sm90`. That is one Python entry point over **two ⏎ different kernels** — profiling kernel names on H200 shows the dispatch flip at M = 32: ⏎  ⏎ | M | kernel actually launched | ⏎ | --- | --- | ⏎ | M < 32 | `deep_gemm::fp8_gemm_kernel_swapAB` | ⏎ | M >= 32 | `deep_gemm::fp8_gemm_kernel` | ⏎  ⏎ Only the swapAB half is worth takin …[truncated]

### L1-8cc941a672  (L1, 2026-08-02, sha 8cc941a672db, PR #33150)
TITLE: [BCG][4/N] Enable bcg on megamoe & flashinfer a2a backend (#33150)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/server_args.py (+3/-3)
LABELS: run-ci, bypass-fastfail, run-ci-extra
BODY: Extend the BCG a2a allowlist from deepep to also cover megamoe and flashinfer. Both were validated on 4xGB300; the gate previously disabled BCG for every non-DeepEP a2a backend. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/bl …[truncated]

### L1-ec741e4161  (L1, 2026-08-02, sha ec741e4161b8, PR #33276)
TITLE: Fix DSpark loading for hybrid DSV4 NVFP4 (#33276)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_loader/loader.py (+3/-2); test/registered/unit/managers/test_batch_result_processor_hidden_states.py (+1/-1); test/registered/unit/test_model_overrides.py (+1/-0)
LABELS: run-ci
BODY: ## Summary ⏎  ⏎ Add `stages.*` to the hybrid DSV4 NVFP4 exclusions so bundled MXFP4 DSpark experts load correctly instead of skipping their scales. ⏎  ⏎ The issue repeated across the expert scales in all three DSpark stages; for example: ⏎  ⏎ ```text ⏎ DSpark V4 draft: unexpected weight 'mtp.0.ffn.experts.198.w1.scale' -> 'stages.0.mlp.experts.198.gate_proj.weight_scale_inv' ⏎ DSpark V4 draft: unexpected weight 'mtp.0.ffn.experts.198.w2.scale' -> 'stages …[truncated]

### L1-5fe97637df  (L1, 2026-08-02, sha 5fe97637df12, PR #33128)
TITLE: Support DeepGEMM for standard MoE dispatch (#33128)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.runner.deep_gemm, L1.ep.layer
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+14/-5); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+166/-17); python/sglang/srt/layers/quantization/fp8.py (+8/-12); test/registered/kernels/ops/moe/test_minimax_quant_scatter.py (+240/-1); test/registered/unit/layers/quantization/test_deepgemm_ue8m0_requant.py (+56/-1)
LABELS: quant, run-ci, jit-kernel, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ The `deep_gemm` MoE runner can be selected with the standard token dispatcher, but the existing integration assumes the DeepEP weight and activation layouts. Standard MoE layers therefore do not consistently requantize both expert weights to UE8M0, and the standard dispatch path does not provide a compact local-expert layout for DeepGEMM. ⏎  ⏎ This change completes the standard-dispatch integration while preserving the existing DeepEP  …[truncated]

### L1-1a3bea77f2  (L1, 2026-08-02, sha 1a3bea77f2ab, PR #33112)
TITLE: [Feat] DCP + HiCache L2 Support (ported from kimi-k3) (#33112)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/scheduler_components/metrics_reporter.py (+3/-4); python/sglang/srt/mem_cache/hiradix_cache.py (+5/-0); python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py (+9/-0); python/sglang/srt/mem_cache/memory_pool_host.py (+3/-0); python/sglang/srt/mem_cache/pool_host/base.py (+43/-5); python/sglang/srt/mem_cache/pool_host/mla.py (+14/-0); python/sglang/srt/server_args.py (+43/-0); test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dcp.py (+112/-0); test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py (+210/-0)
LABELS: hicache, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers. ⏎  …[truncated]

### L1-48dcadc770  (L1, 2026-08-03, sha 48dcadc7703b, PR #31500)
TITLE: [AMD][DI][CI] 5/N Add DSV4 wide-EP16 4-node 2P1D nightly recipes (#31500)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: scripts/ci/slurm/nightly-configs.yaml (+143/-0); scripts/ci/slurm/recipes/mi355x-fp4/dsv4flash/1k1k/2p1d-ep16-mtp.yaml (+107/-0); scripts/ci/slurm/recipes/mi355x-fp4/dsv4flash/1k1k/2p1d-ep16.yaml (+95/-0); scripts/ci/slurm/recipes/mi355x-fp4/dsv4pro/1k1k/2p1d-ep16-mtp.yaml (+107/-0); scripts/ci/slurm/recipes/mi355x-fp4/dsv4pro/1k1k/2p1d-ep16.yaml (+95/-0); scripts/ci/slurm/recipes/mi355x-fp8/dsv4flash/1k1k/2p1d-ep16-mtp.yaml (+107/-0); scripts/ci/slurm/recipes/mi355x-fp8/dsv4flash/1k1k/2p1d-ep16.yaml (+95/-0); scripts/ci/slurm/recipes/mi355x-fp8/dsv4pro/1k1k/2p1d-ep16-mtp.yaml (+107/-0); scripts/ci/slurm/recipes/mi355x-fp8/dsv4pro/1k1k/2p1d-ep16.yaml (+95/-0)
LABELS: amd
BODY: Coauthor: @yctseng0211 @bingxche @michaelzhang-ai ⏎  ⏎ ## What this PR does ⏎  ⏎ Phase-4 of the AMD disaggregation nightly: add **DSV4 Wide-EP16** legs (no-MTP + MTP). The goal is to run **2P1D EP16 across 4 nodes** of the `mi355x` amd-sglang cluster. ⏎  ⏎ ## Topology (Oren's 2P1D config) ⏎  ⏎ - **2 prefill engines** — EP8 / TP8 / DP8, one node each; the router fans requests across both. ⏎ - **1 decode engine** — EP16 / TP16 / DP16, spanning 2 nodes. ⏎ - 4 …[truncated]

### L1-16d3b118a2  (L1, 2026-08-04, sha 16d3b118a20e, PR #33428)
TITLE: Reduce startup log noise and fix Dynamo / CUDA-graph edge cases (#33428)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe
FILES: python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+14/-6); python/sglang/kernels/jit/utils/compile.py (+6/-0); python/sglang/srt/arg_groups/overrides.py (+6/-0); python/sglang/srt/configs/model_config.py (+0/-14); python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py (+1/-1); python/sglang/srt/server_args.py (+8/-32); python/sglang/srt/utils/hf_transformers/common.py (+7/-6); test/registered/unit/test_model_overrides.py (+3/-1)
LABELS: run-ci, jit-kernel
DEEP_STUDY: deep-study correctness case sglang:16d3b118a2: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Summary ⏎  ⏎ Server startup emits several redundant or misleading log lines, and two small ⏎ correctness issues were found alongside them. ⏎  ⏎ **Log noise:** ⏎ - The eight per-backend `"<X> MoE is enabled. The expert parallel size is adjusted ..."` warnings in `server_args.py` all say the same thing and fire unconditionally, even when `ep_size` already equals `tp_size`. They are replaced by one `logger.info` inside the `_a2a_ep_size` post-process, which  …[truncated]

### L1-0753663b8e  (L1, 2026-08-04, sha 0753663b8ea7, PR #33586)
TITLE: [CI] Trim redundant B200 test registrations (#33586)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/attention/unittests/dense/test_fa3.py (+0/-1); test/registered/backends/test_flashinfer_trtllm_gen_attn_backend.py (+0/-65); test/registered/lora/test_lora_qwen3_5_35b_a3b_logprob_diff.py (+1/-1); test/registered/lora/test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py (+1/-1); test/registered/models_e2e/test_gpt_oss_4gpu_bf16.py (+0/-1); test/registered/models_e2e/test_qwen35_fp4_flashinfer.py (+0/-83); test/registered/models_e2e/test_qwen35_fp4_mtp.py (+1/-29); test/registered/spec/eagle/test_eagle_dp_attention.py (+1/-4); test/registered/spec/eagle/test_eagle_infer_beta_dp_attention.py (+0/-85)
LABELS: lora, run-ci
BODY: ## Summary ⏎ - Remove B200 registrations that duplicate default-path or Hopper coverage, and move hardware-agnostic tests off B200 runners ⏎ - `base-c-test-4-gpu-b200` est total: 90 min -> ~63 min; `nightly-4-gpu-b200` -5 min ⏎  ⏎ ## Removed (pin-equals-default / duplicate coverage) ⏎ - `test_qwen35_fp4_flashinfer.py` + the `TestQwen35FP4MTPFlashInfer` class: on SM100 the linear-attn (GDN) backend already defaults to flashinfer and full attention defaults  …[truncated]

### L1-abddb1c7e9  (L1, 2026-08-04, sha abddb1c7e9d6, PR #32541)
TITLE: [Kimi] Support kimi-k3 (#32541)
SOURCES: path_core, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.align.single_token, L1.runner.deep_gemm, L1.runner.marlin, L1.runner.aiter, L1.ep.layer, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe, L1.upstream.aiter_moe
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/kernels/jit/csrc/moe/align_single_token.cuh (+107/-0); python/sglang/kernels/ops/moe/moe_align_single_token.py (+49/-0); .github/workflows/pr-test.yml (+16/-0); docs/docs/references/environment_variables.mdx (+2/-2); python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh (+3/-1); python/sglang/kernels/ops/sampling/renorm_triton.py (+172/-0); python/sglang/srt/arg_groups/kimi_k3_hook.py (+103/-0); python/sglang/srt/arg_groups/overrides.py (+236/-0); (+129 more)
LABELS: documentation, high priority, quant, amd, dependencies, Multi-modal, hicache, blackwell, npu, run-ci
BODY: Day-0 support for the Kimi K3 model. ⏎  ⏎ #### Nvidia Support ⏎ Day 0 Cuda 13 image: `docker pull lmsysorg/sglang:kimi-k3` ⏎ Day 0 Cuda 12 image: `docker pull lmsysorg/sglang:kimi-k3-cu12` ⏎  ⏎ #### AMD Support ⏎ Day 0 image: `docker pull lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727` ⏎ More details: #32548 ⏎  ⏎ #### Links ⏎  ⏎ Cookbook: https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3 ⏎ Blog: https://www.lmsys.org/blog/2026-07-27-kimi-k3-da …[truncated]

### L1-76dc89f5aa  (L1, 2026-08-04, sha 76dc89f5aa30, PR #33611)
TITLE: [Test] Replace NVFP4 MoE runner backend e2e matrix with a layer-level unit test (#33611)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: test/registered/backends/test_deepseek_v3_fp4_cutedsl_moe.py (+5/-71); test/registered/backends/test_deepseek_v3_fp4_cutlass_moe.py (+0/-76); test/registered/quant/test_deepseek_v3_fp4_4gpu_extra.py (+0/-87); test/registered/unit/layers/quantization/test_nvfp4_moe_backends.py (+228/-0)
LABELS: deepseek, blackwell
BODY: Add a layer-level unit test for the NVFP4 `--moe-runner-backend` choices (flashinfer_cutlass / flashinfer_trtllm / flashinfer_cutedsl): it runs the real `FusedMoE` path (construct -> fill NVFP4 checkpoint-format weights -> `process_weights_after_loading` -> `forward`, tp=ep=1 single GPU) against a dequantized torch MoE reference with the kernels' two NVFP4 activation quantization steps mirrored. This covers the per-backend weight prep (TRTLLM shu …[truncated]

### L1-53804d609c  (L1, 2026-08-04, sha 53804d609cb3, PR #32438)
TITLE: [CI][XPU] Stabilize XPU CI: pin UMD/IGC, retry infra flakes, right-size EAGLE3 (#32438)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/xpu.Dockerfile (+30/-4); .github/workflows/pr-test-xpu.yml (+28/-2); python/sglang/test/ci/ci_utils.py (+24/-0); python/sglang/test/test_utils.py (+3/-0); scripts/ci/xpu/xpu_ci_start_container.sh (+51/-0); test/registered/spec/eagle/test_spec_eagle_parity.py (+1/-1)
LABELS: intel, ci, xpu, run-ci
BODY: ## Summary ⏎  ⏎ Four changes to get B580 XPU CI green: ⏎  ⏎ - **`docker/xpu.Dockerfile`** — pin Level-Zero UMD, IGC, and libigdgmm; `apt-mark hold` to stop the rolling PPA from breaking the pinned host xe KMD. ⏎ - **`python/sglang/test/ci/ci_utils.py`** — retry transient GPU OOMs and server-start timeouts once; also retry per-file timeouts under `--enable-retry`. ⏎ - **`.github/workflows/pr-test-xpu.yml`** — pass `--enable-retry` to both stage-a and stage-b  …[truncated]

### L1-81c7a54ecd  (L1, 2026-08-05, sha 81c7a54ecda1, PR #33433)
TITLE: [NVIDIA] Use sm_100f instead of sm_100a for sgl-kernel and FlashMLA (#33433)
SOURCES: path_core
ARTIFACT_HINTS: L1.cutlass.fp8_blockwise
FILES: python/sglang/kernels/aot/csrc/moe/fp8_blockwise_moe_kernel.cu (+3/-0); python/sglang/kernels/aot/CMakeLists.txt (+10/-2); python/sglang/kernels/aot/cmake/flashmla.cmake (+2/-5)
LABELS: sgl-kernel, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ CUDA 12.9+ supports `sm_100f` which is able to run on all sm_10x-family GPUs, whereas `sm_100a` can only run on `sm_100a` GPUs. ⏎  ⏎ We can use this to reduce wheel size by removing redundant `sm_103` binaries while also being able to run on Rubin `sm_107` when it is available. ⏎  ⏎ See https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/ for more information. ⏎  ⏎ ##  …[truncated]

### L1-a14c870886  (L1, 2026-08-05, sha a14c870886bc, PR #33123)
TITLE: Fix broken Nemotron DP attention (#33123)
SOURCES: path_core
ARTIFACT_HINTS: L1.ep.other_dispatchers
FILES: python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py (+7/-4); python/sglang/srt/environ.py (+3/-0); python/sglang/srt/layers/attention/mamba/mamba2_metadata.py (+1/-1); python/sglang/srt/model_executor/model_runner_components/layer_setup.py (+5/-1); python/sglang/srt/models/nemotron_h_mtp.py (+3/-4); python/sglang/srt/server_args.py (+2/-3); test/registered/4-gpu-models/test_nvidia_nemotron_3_super_nvfp4.py (+53/-1)
LABELS: blackwell, run-ci
BODY: ## Motivation ⏎  ⏎ Launch is broken. Fix broken Nemotron DP attention. ⏎  ⏎ ```bash ⏎ SGLANG_FLASHINFER_WORKSPACE_SIZE=1073741824 \ ⏎ python3 -m sglang.launch_server \ ⏎     --dp-size 8 \ ⏎     --enable-dp-attention \ ⏎     --enable-dp-lm-head \ ⏎     --ep-size 8 \ ⏎     --mamba-full-memory-ratio 5.0 \ ⏎     --max-running-requests 1024 \ ⏎     --mem-fraction-static 0.93 \ ⏎     --model-path nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \ ⏎     --moe-a2a-backen …[truncated]

### L1-b9d572ee02  (L1, 2026-08-05, sha b9d572ee0245, PR #33752)
TITLE: [test] Re-enable a pruned Inkling LoRA unit-test set (68 -> 9 cases) (#33752)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/unit/lora/test_experimental_sgl_marlin_alignment.py (+0/-252); test/registered/unit/lora/test_experimental_sgl_marlin_direct_decode.py (+0/-270); test/registered/unit/lora/test_experimental_sgl_marlin_multi_prefill.py (+14/-228); test/registered/unit/lora/test_experimental_sgl_marlin_policy.py (+33/-185); test/registered/unit/lora/test_experimental_sgl_marlin_runtime_unit.py (+220/-678); test/registered/unit/lora/test_experimental_sgl_marlin_shared_outer_reduce.py (+25/-25); test/registered/unit/lora/test_inkling_linearized_lora_unit.py (+425/-1089); test/registered/unit/lora/test_inkling_lora_normalization_unit.py (+0/-122); test/registered/unit/lora/test_inkling_moe_lora_overlap_unit.py (+0/-313)
LABELS: lora
BODY: ## Motivation ⏎  ⏎ The Inkling support PR (#31681) added 9 LoRA unit-test files / **68 cases** under `test/registered/unit/lora/`, all disabled with a module-level `pytest.mark.skip` ("new inkling LoRA test; disabled on CI"). The underlying CI blocker is fixed, so this re-enables the tests — but 68 cases is far more than the coverage justifies, so they are pruned first against [`.claude/rules/unit-test-admission.md`](../blob/main/.claude/rules/unit-t …[truncated]

### L1-02cd44c59a  (L1, 2026-08-05, sha 02cd44c59a69, PR #33108)
TITLE: feat(dgx-spark): add inkling-small MoE support for sm_121 (#33108)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/moe/inkling_moe.py (+6/-1); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_interleaved_sm_121.json (+39/-0); python/sglang/srt/utils/common.py (+7/-0)
LABELS: run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎ Enable Inkling-small MoE kernels on DGX Spark (GB10, sm_121). The existing Triton configs and grouped-GEMM staging assumptions target larger GPUs; sm_121 has less shared-memory headroom, so default configs can fail or underperform without platform-specific tuning. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add `is_dgx_spark()` in `common.py` (CUDA + compute capability `(12, 1)`), cached via `lru_cache`. ⏎ - In `inkling_moe.grouped_gemm_triton`, red …[truncated]

### L1-7bc90ab394  (L1, 2026-08-05, sha 7bc90ab39444, PR #33474)
TITLE: Select DeepGEMM standard layouts by memory budget (#33474)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.runner.deep_gemm
FILES: python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+86/-5); python/sglang/srt/environ.py (+2/-0); python/sglang/srt/layers/quantization/fp8.py (+16/-8); python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py (+53/-0); test/registered/kernels/ops/moe/test_minimax_quant_scatter.py (+43/-10)
LABELS: quant, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ The standard DeepGEMM dispatcher currently selects the masked layout from expert parallelism alone. However, masked workspace grows with the local expert count and padded forward size, so a fixed expert-count threshold can miss safe performance opportunities and can still select shapes that exceed available memory. ⏎  ⏎ This follows the layout-selection discussion in [#33128](https://github.com/sgl-project/sglang/pull/33128#issueco …[truncated]

### L1-beabc5949b  (L1, 2026-08-05, sha beabc5949b75, PR #33618)
TITLE: Enable MoE deferred finalize by default and drop its expert_weights dtype workaround (#33618)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L1.runner.flashinfer_trtllm
FILES: python/sglang/kernels/jit/csrc/moe/moe_finalize_fuse_shared.cu (+4/-4); python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+0/-7); docs/docs/references/environment_variables.mdx (+1/-1); python/sglang/srt/environ.py (+1/-1)
LABELS: documentation, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ `trtllm_fp4_block_scale_moe` allocated its `expert_weights` output with `routing_logits.dtype` while the trtllm-gen routing kernel always writes bf16 into it, so the `do_finalize=False` path returned bf16 data mislabeled as fp32 for DeepSeekV3-style fp32 routing logits. We worked around it by reinterpreting the bf16 prefix, and kept the fused finalize opt-in because of it. ⏎  ⏎ [flashinfer-ai/flashinfer#3644](https://github.com/flashin …[truncated]

### L1-c11ce7c514  (L1, 2026-08-06, sha c11ce7c514c4, PR #33842)
TITLE: chore: bump sgl-kernel version to 0.4.6.post1 (#33842)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/aot/pyproject.toml (+1/-1); python/sglang/kernels/aot/pyproject_cpu.toml (+1/-1); python/sglang/kernels/aot/pyproject_musa.toml (+1/-1); python/sglang/kernels/aot/pyproject_rocm.toml (+1/-1); python/sglang/kernels/aot/python/sgl_kernel/version.py (+1/-1)
LABELS: amd, dependencies, sgl-kernel, mthreads
BODY: ## Summary ⏎  ⏎ This PR bumps the sgl-kernel version to `0.4.6.post1` across all relevant files. ⏎  ⏎ ## Files Updated ⏎ - python/sglang/kernels/aot/pyproject.toml ⏎ - python/sglang/kernels/aot/pyproject_cpu.toml ⏎ - python/sglang/kernels/aot/pyproject_musa.toml ⏎ - python/sglang/kernels/aot/pyproject_rocm.toml ⏎ - python/sglang/kernels/aot/python/sgl_kernel/version.py ⏎  ⏎ 🤖 Generated with GitHub Actions ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #3108350362 …[truncated]

### L1-ba9074035a  (L1, 2026-08-06, sha ba9074035a98, PR #33498)
TITLE: Build and release sgl-deep-ep wheels (#33498)
SOURCES: path_core, dependency_pin, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/release-whl-deepep.yml (+261/-0); docker/sgl-deep-ep.Dockerfile (+75/-0); scripts/build_sgl_deepep.sh (+197/-0); scripts/update_deepep_whl_index.py (+82/-0)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ Add the release pipeline for the `sgl-deep-ep` binary distribution. Users install CUDA 13 builds with `pip install sgl-deep-ep`; CUDA 12.9 builds are published through the SGLang wheel index. The Python import remains `deep_ep`. ⏎  ⏎ The shared packaging overlay is proposed separately in sgl-project/DeepEP#3 on `sgl-deepep-packaging`. ⏎  ⏎ ## Modifications ⏎  ⏎ - Convert `scripts/build_sgl_deepep.sh` into a Docker orchestrator that takes Pytho …[truncated]

### L1-f8f2870a84  (L1, 2026-08-06, sha f8f2870a84a4, PR #24370)
TITLE: Profiling Enhancements [1/3]: cuda graph profile traces (#24370)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/docs/developer_guide/benchmark_and_profiling.mdx (+31/-0); python/sglang/srt/environ.py (+4/-0); python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py (+73/-9); python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py (+21/-0); python/sglang/srt/utils/profile_utils.py (+17/-6); test/registered/unit/model_executor/runner/test_decode_cuda_graph_runner.py (+294/-0); test/registered/unit/model_executor/runner_backend/test_full_cuda_graph_backend.py (+162/-0)
LABELS: documentation, run-ci
BODY: # Profiling Enhancements [1/3]: CUDA-graph capture profile traces ⏎  ⏎ ## Summary ⏎  ⏎ Adds opt-in profiling of the **CUDA-graph capture phase**. When enabled, each ⏎ captured decode batch size is recorded with a `torch.profiler` pass and exported ⏎ as its own Chrome trace, giving per-shape visibility (kernel identities, input ⏎ shapes, FLOPs, memory) into the graphs that are replayed at serving time. ⏎  ⏎ ## Motivation ⏎  ⏎ During startup the runner captur …[truncated]

### L1-4c0a8940fa  (L1, 2026-08-06, sha 4c0a8940fa0d, PR #33205)
TITLE: [Kernel] Unify BaseFusedOp and MultiPlatformOp dispatch (#33205)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+14/-2); docs/docs/hardware-platforms/plugin.mdx (+1/-1); python/sglang/kernels/README.md (+41/-20); python/sglang/kernels/fused_op.py (+433/-98); python/sglang/kernels/ops/layernorm/__init__.py (+3/-3); python/sglang/srt/compilation/torch_compile_decoration.py (+3/-3); python/sglang/srt/layers/activation.py (+11/-8); python/sglang/srt/layers/attention/dsa/dsa_indexer.py (+20/-2); python/sglang/srt/layers/attention/dsv4/compressor.py (+2/-2); python/sglang/srt/layers/attention/mamba/mixer2_rms_norm_gated.py (+2/-2); (+13 more)
LABELS: documentation, quant, run-ci, piecewise-cuda-graph, jit-kernel, run-ci-extra
BODY: ## Motivation ⏎  ⏎ RFC #29630 introduced `BaseFusedOp` as the per-operator multi-backend contract of the unified `sglang.kernels` namespace, while `sglang.srt.layers.utils.MultiPlatformOp` still owned multi-platform dispatch, OOT platform plugins, and the torch.compile enter/leave protocol. This PR completes the unification discussed in https://github.com/sgl-project/sglang/issues/29630#issuecomment-4920387930 and https://github.com/sgl-project/sglan …[truncated]

### L1-4ad990ba7d  (L1, 2026-08-06, sha 4ad990ba7d75, PR #33115)
TITLE: [ModelOpt FP4] Support online MoE weight quantization (#33115)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L1.ep.layer
FILES: python/sglang/srt/layers/moe/ep_moe/layer.py (+1/-2); docs/docs/advanced_features/quantization.mdx (+5/-5); docs/docs/references/environment_variables.mdx (+2/-2); python/sglang/srt/configs/model_config.py (+9/-2); python/sglang/srt/layers/quantization/modelopt_quant.py (+87/-45); python/sglang/srt/layers/quantization/nvfp4_online.py (+121/-34); python/sglang/srt/model_loader/loader.py (+29/-7); python/sglang/srt/model_loader/weight_utils.py (+36/-3); python/sglang/srt/models/qwen3_5_mtp.py (+9/-4); python/sglang/srt/server_args.py (+10/-0); (+4 more)
LABELS: documentation, quant, speculative-decoding, blackwell, run-ci, run-ci-extra
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ @humansand ⏎  ⏎ - Add online ModelOpt FP4 quantization for eligible MoE expert weights from BF16, FP16, or FP8 checkpoints. ⏎ - Current contract: ⏎   - `modelopt_fp4`: ⏎     - Serialized NVFP4: load packed weights and checkpoint per-tensor FP32 activation scales; missing scales default to `1.0`, and loaded scales overwrite the fallback. ⏎     - BF16/FP16 source: quantize eligible MoE expert weights to NVFP4 at load time; dense and excluded lay …[truncated]

### L1-434e646282  (L1, 2026-08-06, sha 434e646282e5, PR #28836)
TITLE: [Deps] Upgrade CUDA PyTorch stack to 2.13 (#28836)
SOURCES: path_core, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L1.upstream.vllm.fused_topk, L1.triton.fused_moe, L1.routing.topk_py, L1.runner.openai_triton_kernels, L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe, L1.upstream.openai_triton_kernels
FILES: docker/Dockerfile (+21/-27); docker/kimi_k3/kimi_k3_cu12.Dockerfile (+2/-2); docker/kimi_k3/kimi_k3_cu13.Dockerfile (+2/-2); docker/sgl-deep-gemm.Dockerfile (+1/-1); python/pyproject.toml (+4/-4); python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py (+37/-43); python/sglang/srt/layers/moe/moe_runner/triton_kernels.py (+20/-16); python/sglang/srt/layers/moe/topk.py (+26/-17); .github/workflows/_docker-build-and-publish.yml (+4/-4); .github/workflows/_pr-test-stage.yml (+1/-0); (+24 more)
LABELS: documentation, high priority, quant, dependencies, Multi-modal, deepseek, sgl-kernel, ready-to-merge, blackwell, run-ci
BODY: ## Summary ⏎  ⏎ torch: 2.11.0 → 2.13.0 ⏎ torchvision: 0.26.0 → 0.28.0 ⏎ triton: 3.6.0 → 3.7.1 ⏎ torchaudio: stays at 2.11.0 ⏎ triton_kernels: 3.6.0 → 3.7.1 ⏎ torchcodec: 0.11.1 → 0.15.0 ⏎  ⏎ ## Test Plan ⏎  ⏎ CI ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  ⏎  …[truncated]

### L1-e0af47b03e  (L1, 2026-08-06, sha e0af47b03edf, PR #33866)
TITLE: Fix sgl-deep-ep builder dependencies (#33866)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/release-whl-deepep.yml (+1/-1); scripts/build_sgl_deepep.sh (+1/-1); docker/sgl-deep-ep.Dockerfile (+5/-3)
BODY: ## What changed ⏎  ⏎ - explicitly enable the AlmaLinux PowerTools repository so `libfabric-devel` resolves in the x86 manylinux builder ⏎ - register `/usr/local/lib` with `ldconfig` so the freshly installed `libgdrapi.so.2` is discoverable during the image build ⏎ - upgrade the DeepEP release workflow, build-script default, and Docker image to `torch==2.13.0` ⏎ - make the Docker smoke assertion read `TORCH_VERSION` instead of duplicating a hard-coded versi …[truncated]

### L1-295784723a  (L1, 2026-08-06, sha 295784723a6b, PR #33822)
TITLE: [diffusion] Ideogram 4: fuse RMSNorm modulate/gate chains via the Z-Image Triton suite behind quality=high (H200 e2e -2.9%/-3.4%) (#33822)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/diffusion/fused_gate_rmsnorm.py (+135/-0); python/sglang/multimodal_gen/runtime/models/dits/ideogram.py (+60/-7); python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py (+34/-21); test/registered/kernels/ops/diffusion/test_fused_gate_rmsnorm.py (+54/-0)
LABELS: run-ci, diffusion, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Reuses the Z-Image bf16-native norm kernels already in-tree (`kernels/ops/diffusion/triton/zimage_native_norm.py`) — **no new kernels**; adds the mount/unmount site protocol for them (same shape as #33536's `fused_linear_gelu`). 4 files, +283/−28. ⏎  ⏎ ## Motivation ⏎  ⏎ The adaptation-matrix audit (main @ 407a65d3c) flagged Ideogram 4's transformer block as form-identical to Z-Image's modulate/gate structure, for which a fused Triton suite already ships …[truncated]

### L1-4020bc95a7  (L1, 2026-08-07, sha 4020bc95a7b5, PR #33543)
TITLE: Fix Nemotron W4A16 NVFP4 MoE backend (#33543)
SOURCES: path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L1.runner.marlin
FILES: python/sglang/srt/layers/quantization/marlin_utils_fp4.py (+48/-16); python/sglang/srt/arg_groups/overrides.py (+23/-1); test/registered/unit/test_model_overrides.py (+131/-3)
LABELS: run-ci
BODY: Detect Nemotron-H ModelOpt mixed-precision checkpoints with W4A16_NVFP4 MoE expert layers. ⏎  ⏎ - Keep the existing SM100 `flashinfer_trtllm` default for regular NVFP4/W4A4 MoE, but route W4A16 NVFP4 expert MoE to `Marlin`.  ⏎ - Add focused override tests for both W4A16 NVFP4 and regular NVFP4 paths. ⏎  ⏎ Also support TP-sharded W4A16 NVFP4 Marlin layers whose local K/N dimensions do not meet a Marlin thread-tile alignment. Pad weights, scales, and ac …[truncated]

### L1-8a22b8305d  (L1, 2026-08-07, sha 8a22b8305d59, PR #33956)
TITLE: docker: add Kimi K3 artifacts and build hpc-ops with C++20 (#33956)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+49/-3)
BODY: ## Motivation ⏎  ⏎ The generic CUDA image does not include the pinned TRT-LLM generated-MoE cubin pool or the FlashInfer CuTeDSL MLA decode-context-parallel runtime patch currently installed by the Kimi K3 CUDA 12 and CUDA 13 images. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add an independent builder stage that downloads the pinned cubin archive, verifies its SHA-256 digest, extracts it, and requires exactly 1,696 cubin files. ⏎ - Copy the verified pool into both framewor …[truncated]

### L1-eb3cc879e0  (L1, 2026-08-07, sha eb3cc879e0cd, PR #33932)
TITLE: Install DeepEP from release wheels (#33932)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/pyproject.toml (+1/-0); scripts/ci/cuda/ci_install_deepep.sh (+0/-177); .github/workflows/_pr-test-sgl-kernel-build.yml (+0/-1); .github/workflows/_pr-test-stage.yml (+0/-3); .github/workflows/nightly-test-nvidia.yml (+1/-3); .github/workflows/pr-test-extra.yml (+0/-47); .github/workflows/pr-test.yml (+3/-51); .github/workflows/rerun-test.yml (+1/-10); scripts/ci/cuda/ci_install_dependency.sh (+97/-0); scripts/ci/list_stage_models.py (+4/-5); (+21 more)
LABELS: high priority, dependencies, deepseek, run-ci, bypass-fastfail, run-ci-extra, release-highlight
BODY: ## Motivation ⏎  ⏎ SGLang now publishes `sgl-deep-ep` wheels, so CUDA CI should consume the released wheel instead of rebuilding DeepEP from source in dedicated runner configurations and test suites. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add the pinned `sgl-deep-ep==0.1.0` Python dependency. ⏎ - Install GDRCopy and its system dependencies from the common CUDA dependency installer, while keeping GDRCopy optional on runners that do not need it. ⏎ - Use PyPI for CUD …[truncated]

### L1-a42683eb62  (L1, 2026-08-07, sha a42683eb629b, PR #32341)
TITLE: [diffusion] model: support lingbot-video moe 30b t2v (#32341)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/multimodal_gen/configs/models/dits/__init__.py (+4/-0); python/sglang/multimodal_gen/configs/models/dits/lingbot_video_moe.py (+59/-0); python/sglang/multimodal_gen/configs/pipeline_configs/__init__.py (+4/-0); python/sglang/multimodal_gen/configs/pipeline_configs/lingbot_video_moe.py (+86/-0); python/sglang/multimodal_gen/configs/sample/__init__.py (+4/-0); python/sglang/multimodal_gen/configs/sample/lingbot_video_moe.py (+20/-0); python/sglang/multimodal_gen/registry.py (+14/-0); python/sglang/multimodal_gen/runtime/distributed/parallel_state.py (+16/-0); python/sglang/multimodal_gen/runtime/layers/moe.py (+182/-0); python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py (+3/-0); (+10 more)
LABELS: run-ci, diffusion
BODY: ## Motivation ⏎  ⏎ Add native SGLang support for **LingBot-Video MoE 30B** (`robbyant/lingbot-video-moe-30b-a3b`) — a DeepSeek-V3-style MoE text-to-video model (128 experts, 30B total / ~3B active per token, 48 MoE layers). This is the **first MoE DiT** in `multimodal_gen`, reusing SGLang's `fused_experts` Triton kernel for the expert GEMMs. ⏎  ⏎ Related issue: #32336 ⏎  ⏎ ## Modifications ⏎  ⏎ - **`runtime/layers/moe.py`** (NEW): DeepSeek-V3-style MoE F …[truncated]

### L1-eda0ddc260  (L1, 2026-08-07, sha eda0ddc260d4, PR #33888)
TITLE: config: delete the dead get_server_args() bindings across the repo (#33888)
SOURCES: path_core
ARTIFACT_HINTS: L1.routing.topk_py
FILES: python/sglang/srt/layers/moe/topk.py (+1/-1); python/sglang/srt/layers/attention/linear/inkling_sconv_backend.py (+0/-1); python/sglang/srt/managers/schedule_batch.py (+0/-1); python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py (+1/-3); python/sglang/srt/mem_cache/storage/flexkv/flexkv_radix_cache.py (+0/-3); python/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py (+1/-2); python/sglang/srt/model_loader/loader.py (+0/-1); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+1/-2); python/sglang/srt/models/deepseek_v2.py (+0/-3); python/sglang/srt/models/minimax_m3_vl.py (+1/-2); (+4 more)
LABELS: deepseek, ready-to-merge
BODY: ## What ⏎  ⏎ Deletes the 15 dead `server_args = get_server_args()` bindings that earlier bag-migration flips left behind, plus the imports they orphaned: `sampling_batch_info`, `deepseek_v2` (3), `forward_mla`, `minimax_m3_vl`, `routed_experts`, `indexer_topk`, `loader`, `deepseek_v4_memory_pool` (2), `schedule_batch`, `inkling_sconv_backend`, and the flexkv/lmcache commit-prefix paths. ⏎  ⏎ In `layers/moe/topk.py` the call itself is load-bearing — it pr …[truncated]

### L1-b61a06921e  (L1, 2026-08-07, sha b61a06921ef0, PR #33889)
TITLE: moe: the shared-experts-fusion decision is a per-runner value the loader installs (#33889)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/utils.py (+90/-0); python/sglang/srt/model_executor/model_runner.py (+8/-5); python/sglang/srt/model_executor/model_runner_components/load_model_utils.py (+4/-4); python/sglang/srt/models/deepseek_nextn.py (+3/-1); python/sglang/srt/models/deepseek_ocr.py (+21/-5); python/sglang/srt/models/deepseek_v2.py (+37/-44); python/sglang/srt/models/deepseek_v4.py (+18/-27); python/sglang/srt/models/deepseek_vl2.py (+10/-0); python/sglang/srt/models/glm4_moe.py (+20/-35); python/sglang/srt/models/glm4_moe_lite.py (+24/-33); (+30 more)
LABELS: documentation, Multi-modal, deepseek, speculative-decoding, diffusion
BODY: ## What ⏎  ⏎ A draft's construction used to rewrite the process config record. Two writers, two shapes: ⏎  ⏎ **1. The shared-experts-fusion decision becomes a per-runner value the loader installs.** `declare_load_time_override` wrote the fusion decision to the target's bags — and an MTP/nextn draft IS a DeepSeek/GLM/Qwen3.5/MiniMax model, so a draft whose checkpoint differs from the target's (quantization) corrupted the target's record. This is ancestral …[truncated]

### L1-c9444deef4  (L1, 2026-08-07, sha c9444deef4a5, PR #34041)
TITLE: Docker: install DeepEP from release wheels (#34041)
SOURCES: subject_keyword, dependency_pin, body_keyword
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+16/-89); .github/workflows/_docker-build-and-publish.yml (+0/-4); .github/workflows/nightly-72-gpu-gb200.yml (+0/-1); .github/workflows/release-docker-dev.yml (+0/-1)
BODY: ## Motivation ⏎  ⏎ `sgl-deep-ep==0.1.0` is now a released SGLang dependency. The main Dockerfile should consume the published wheels instead of cloning and compiling DeepEP for every image build. ⏎  ⏎ ## Modifications ⏎  ⏎ - Remove the `deepep_builder` stage and the framework-stage DeepEP wheel/source copies. ⏎ - Let CUDA 13 images install the public `sgl-deep-ep==0.1.0` wheel through the existing pyproject dependency. ⏎ - Preinstall `sgl-deep-ep==0.1.0+cu129` f …[truncated]

### L1-55f02e6887  (L1, 2026-08-08, sha 55f02e68875a, PR #33691)
TITLE: Support Intern-S2-Mobius (#33691)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/arg_groups/overrides.py (+10/-0); python/sglang/srt/configs/__init__.py (+8/-0); python/sglang/srt/configs/interns2_mobius.py (+54/-0); python/sglang/srt/configs/model_config.py (+15/-0); python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py (+2/-0); python/sglang/srt/lora/lora_manager.py (+6/-0); python/sglang/srt/models/interns2_mobius.py (+819/-0); python/sglang/srt/multimodal/processors/qwen_vl.py (+7/-0); python/sglang/srt/server_args.py (+13/-1); python/sglang/srt/utils/hf_transformers/common.py (+4/-0)
LABELS: lora, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Support [internlm/Intern-S2-Mobius](https://huggingface.co/internlm/Intern-S2-Mobius) ⏎  ⏎ ```shell ⏎ python -m sglang.launch_server --model-path internlm/Intern-S2-Mobius \ ⏎ --trust-remote-code \ ⏎ --port 8000 \ ⏎ --tp-size 2 \ ⏎ --mem-fraction-static 0.8 \ ⏎ --context-length 262144 \ ⏎ --reasoning-parser qwen3 \ ⏎ --speculative-algo NEXTN \ ⏎ --speculative-num-steps 3 \ ⏎ --speculative-eagle-topk 1 \ ⏎ --speculative-num-draft-tokens 4 ⏎ ``` …[truncated]

### L1-afb4f37ca5  (L1, 2026-08-08, sha afb4f37ca509, PR #33903)
TITLE: [Inkling] silu_and_mul: replace helion kernels with plain Triton (#33903)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L1.upstream.deepep, L1.upstream.deepgemm, L1.upstream.flashinfer_moe
FILES: python/sglang/kernels/ops/moe/inkling_moe.py (+207/-123); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_interleaved_sm_100.json (+0/-39); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_interleaved_sm_121.json (+0/-39); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_interleaved_sm_90.json (+0/-40); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_interleaved_sm_95.json (+0/-40); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_sm_100.json (+0/-39); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_sm_90.json (+0/-40); python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_sm_95.json (+0/-40); python/sglang/srt/layers/moe/moe_runner/triton_utils/helion_utils.py (+0/-296); python/pyproject.toml (+0/-1); (+3 more)
LABELS: dependencies, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Remove the helion dependency from the Inkling MoE activation path: port the two helion `silu_and_mul` kernels to plain Triton (bitwise-identical outputs), delete the helion AOT autotune machinery and its per-arch config tables, and drop `helion==1.4` from the package dependencies. ⏎  ⏎ ## Modifications ⏎  ⏎ - Port the two helion `silu_and_mul` kernels in `python/sglang/kernels/ops/moe/inkling_moe.py` to plain Triton: `silu_and_mul_interlea …[truncated]

### L1-cea16bf229  (L1, 2026-08-08, sha cea16bf22933, PR #34006)
TITLE: Fix Qwen3-MoE producing garbage with the mori a2a backend (#34006)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/qwen3_moe.py (+1/-0)
BODY: MoeA2ABackend.is_deepep() is an exact enum match, so a mori run falls through Qwen3MoeSparseMoeBlock.forward into forward_normal. forward_normal then applies an expert-parallel all-reduce on top of the output mori has already combined. ⏎  ⏎ deepseek_v2, deepseek_v4, glm4_moe and glm4_moe_lite already gate on is_mori() next to is_deepep(); qwen3_moe was missed. ⏎  ⏎ Verified on 8x MI355X (gfx950) driving a full Qwen3-30B-A3B RL run, where Megatron-rec …[truncated]

### L1-5fdf6cd18f  (L1, 2026-08-08, sha 5fdf6cd18f97, PR #32395)
TITLE: [MoE] Single-launch moe_align for tiny batches with many experts (#32395)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L1.triton.moe_align, L1.align.small_numel
FILES: python/sglang/kernels/ops/moe/__init__.py (+17/-0); python/sglang/kernels/ops/moe/moe_align_small_numel.py (+147/-0); python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py (+32/-0); test/registered/kernels/ops/moe/test_moe_align_small_numel.py (+214/-0)
LABELS: run-ci, jit-kernel, run-ci-extra, user-tps
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ **Background**  ⏎ The Triton fused-MoE GEMM consumes tokens in tiles of `block_size` rows, and every row in a tile must belong to the same expert so the tile can load that expert's weight block once. `moe_align_block_size` is what makes that possible: it takes the flattened `num_tokens * topk` (token, slot) pairs, groups them by expert, pads each group up to a multiple of `block_size`, and hands the GEMM three buffers — `sorted_to …[truncated]

### L1-4ad5bb5d9a  (L1, 2026-08-08, sha 4ad5bb5d9ae0, PR #33400)
TITLE: [jit_kernel] Move JIT kernels into namespace sglang (#33400)
SOURCES: path_core, corpus:kernel-correctness-cases(introducing)
ARTIFACT_HINTS: L1.routing.topk_sigmoid, L1.routing.fused_gate, L1.routing.hash_topk, L1.align.cuda_jit, L1.reduce.topk_sum_jit
FILES: python/sglang/kernels/jit/csrc/deepseek_v4/hash_topk.cuh (+2/-2); python/sglang/kernels/jit/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh (+2/-2); .claude/skills/add-jit-kernel/SKILL.md (+3/-2); docs/docs/developer_guide/development_jit_kernel_guide.mdx (+9/-2); python/sglang/kernels/jit/csrc/add_constant.cuh (+2/-2); python/sglang/kernels/jit/csrc/attention/fixup_zero_kv.cuh (+2/-2); python/sglang/kernels/jit/csrc/attention/fused_fp8_qkv_kv_cache.cuh (+2/-2); python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh (+3/-3); python/sglang/kernels/jit/csrc/attention/kda_packed_decode.cuh (+2/-2); python/sglang/kernels/jit/csrc/deepseek_v32/indexer_k.cuh (+2/-2); (+168 more)
LABELS: documentation, quant, lora, hicache, run-ci, jit-kernel
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:ec5199b906 (fix PR 34106)
BODY: > Generated by Claude. ⏎  ⏎ ## Motivation ⏎  ⏎ Every JIT kernel used to sit in the global namespace (or an anonymous one), with a handful of files having already started on `namespace sglang` and reaching back out via `using namespace sglang;` / `sglang::`. This unifies that: all JIT C++ under `python/sglang/kernels/jit/` now lives in `namespace sglang`, and no `sglang::` qualification is needed anywhere inside it. ⏎  ⏎ ## Modifications ⏎  ⏎ - **`namespace sglan …[truncated]

### L1-3fbb5330c7  (L1, 2026-08-08, sha 3fbb5330c7e5, PR #33764)
TITLE: Fix the router GEMM inaccuracy when using _front_w in Kimi-K3 (#33764)
SOURCES: path_core, symbol_pickaxe, corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/moe/route_quant_fused.cuh (+33/-14); python/sglang/kernels/ops/moe/moe_route_quant_fused.py (+1/-1); python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh (+75/-2); python/sglang/kernels/jit/csrc/kimi_k3/situ_and_mul.cuh (+17/-12); python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py (+26/-3); python/sglang/kernels/ops/kimi_k3/activation.py (+10/-8); python/sglang/kernels/ops/quantization/per_token_group_quant.py (+1/-1); python/sglang/srt/models/kimi_k3.py (+53/-18)
LABELS: quant, run-ci, jit-kernel
DEEP_STUDY: deep-study correctness case sglang:3fbb5330c7: class=numerical_precision; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Problem ⏎  ⏎ The Kimi-K3 MoE layer computes the router logits in bf16. The router then ⏎ selects the experts from these bf16 logits. The selection is not accurate. ⏎  ⏎ Other implementations use fp32 for this GEMM: ⏎  ⏎ - vLLM sets `out_dtype=torch.float32` on its gate ⏎ - the SGLang EP front (`moe_front.fused_front`) uses `out_dtype=torch.float32` ⏎  ⏎ Measurement with the layer-1 gate weight and the fp32 correction bias: ⏎  ⏎ - 5.27% of the tokens get a …[truncated]

### L1-ec5199b906  (L1, 2026-08-08, sha ec5199b906f1, PR #34106)
TITLE: [jit_kernel] Fix missing JIT kernel namespaces (#34106)
SOURCES: path_core, corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/moe/align_single_token.cuh (+2/-2); python/sglang/kernels/jit/csrc/diffusion/modulate_scale_shift.cuh (+4/-0)
LABELS: run-ci, jit-kernel, run-ci-extra, mergeable
DEEP_STUDY: deep-study correctness case sglang:ec5199b906: class=integration_backend_cudagraph; symptom=compile_or_build_failure; introducing=#33400
BODY: ## Summary ⏎  ⏎ - Fix the JIT namespace break from #33400 for the single-token MoE kernel added in #32541. ⏎ - Fix the same helper-namespace migration break in the diffusion adaLN modulation kernel. ⏎ - Restore first-use JIT compilation without changing either kernel's computation. ⏎  ⏎ ## Validation ⏎  ⏎ - Targeted pre-commit checks passed, including clang-format and codespell. ⏎ - H200 with a fresh TVM-FFI cache: `test_modulate_scale_shift.py` — **12 passed**. ⏎ - …[truncated]

### L1-dc9624deb2  (L1, 2026-08-09, sha dc9624deb2f0, PR #34085)
TITLE: [diffusion] Clean up kernels and shared fast paths (#34085)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/diffusion/fused_gate_rmsnorm.py (+26/-39); python/sglang/kernels/jit/csrc/diffusion/modulate_scale_shift.cuh (+92/-176); python/sglang/kernels/jit/csrc/diffusion/qknorm_rope.cuh (+1/-0); python/sglang/kernels/jit/csrc/diffusion/residual_gate_add.cuh (+132/-241); python/sglang/kernels/jit/csrc/diffusion/usp_relayout.cuh (+100/-135); python/sglang/kernels/ops/diffusion/__init__.py (+1/-13); python/sglang/kernels/ops/diffusion/cutedsl/scale_residual_norm_scale_shift.py (+5/-5); python/sglang/kernels/ops/diffusion/fused_linear_gelu.py (+18/-28); python/sglang/kernels/ops/diffusion/fused_ln_modulate.py (+12/-16); python/sglang/kernels/ops/diffusion/modulate_scale_shift.py (+37/-2); (+40 more)
LABELS: run-ci, diffusion, apple-silicon, jit-kernel, run-ci-extra, mergeable
BODY: ## Summary ⏎  ⏎ - Centralize diffusion quality-gated fusion selection and shared model residual/modulation fallbacks. ⏎ - Reuse `jit_kernel` launcher utilities in recent CUDA kernels and shared bit-exact numerical primitives in Triton kernels. ⏎ - Move generic native BF16 RMSNorm out of the Z-Image-specific module and tighten shape, dtype, device, and compile guards across recent diffusion fast paths. ⏎ - Remove dead registrations, duplicated model helpers …[truncated]
