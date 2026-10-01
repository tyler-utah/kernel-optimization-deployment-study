### L3-bbbd0a02c9  (L3, 2026-09-14, sha bbbd0a02c9d1, PR #56382)
TITLE: [Bugfix] Carry over queued work when materializing the dedicated stream (#56382)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/utils/torch_utils.py (+4/-1)
LABELS: bug, ready
BODY: # Carry over queued work when materializing the dedicated stream ⏎  ⏎ ## Purpose ⏎  ⏎ `vllm.utils.torch_utils.current_stream()` is a getter that *mutates* the current ⏎ stream: the first call per thread runs ⏎ `torch.cuda.set_stream(torch.cuda.Stream())`. Nothing ties the new stream to the ⏎ one it replaces. Pool streams are created with `cudaStreamNonBlocking` ⏎ ([`c10/cuda/CUDAStream.cpp`](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDAStrea …[truncated]

### L3-9d14fd3095  (L3, 2026-09-15, sha 9d14fd3095eb, PR #52781)
TITLE: [Kernel][MoE] DeepEP v2: async finalize to overlap shared experts with combine (#52781)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_deepep_v2_async_finalize.py (+122/-0); vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py (+28/-8)
LABELS: ready
ISSUES: #52744 [Performance]: DeepEP v2 (ElasticBuffer) finalize_async runs combine synchronously — the shared-expert overlap window is never used
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: > **Status (2026-08-31):** reviewer feedback addressed — overlap is now **default-on** (`VLLM_DEEPEP_V2_COMBINE_OVERLAP=1`, capability-gated fallback, env is a kill-switch), rebased clean onto current main (no conflicts). **Blocked only on a `ready` label** so Buildkite can run — the full suite has never executed on this fork PR. Contract tests included; EP8 A/B in comments. ⏎  ⏎ --- ⏎  ⏎ FIX #52744 ⏎  ⏎ ### What ⏎  ⏎ `prepare_finalize/deepep_v2.py` issues the E …[truncated]

### L3-a7576447b8  (L3, 2026-09-15, sha a7576447b864, PR #56687)
TITLE: [ROCm][Bugfix] Initialize Ray NIXL agents for sharded RDT (#56687)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/distributed/test_sharded_rdt_producer.py (+114/-0); vllm/distributed/nixl_utils.py (+17/-0); vllm/distributed/weight_transfer/sharded_rdt_common.py (+10/-0); vllm/distributed/weight_transfer/sharded_rdt_engine.py (+2/-3); vllm/distributed/weight_transfer/sharded_rdt_trainer.py (+1/-4)
LABELS: bug, rocm, ready, ray, kv-connector
BODY: - Add a shared ROCm initializer that creates Ray's process-local NIXL agent through vLLM's existing `nixl_rocm` loader and populates Ray's cached agent. ⏎ - Call the initializer in both the producer actor and inference worker before NIXL memory registration, keeping transport compatibility in the sharded RDT implementation. ⏎ - Reuse an existing agent, leave non-ROCm initialization unchanged, and report a clear error when the ROCm NIXL package is u …[truncated]

### L3-8263ea12bd  (L3, 2026-09-15, sha 8263ea12bd8f, PR #56876)
TITLE: [Build] Move DeepGEMM pin to the vLLM fork and update the Mega MoE call convention (#56876)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: cmake/external_projects/deepgemm.cmake (+10/-6); docker/Dockerfile (+1/-0); tools/build_deepgemm_C.py (+1/-1); tools/install_deepgemm.sh (+5/-3); vllm/models/kimi_k3/nvidia/model.py (+4/-2); vllm/utils/deep_gemm.py (+5/-1)
LABELS: ci/build, nvidia, kimi, k3
BODY: ## Purpose ⏎  ⏎ Repoint the DeepGEMM pin from `deepseek-ai/DeepGEMM` `nv_dev` (2.6.1, `8b1392b`) to `vllm-project/DeepGEMM` `dev` at [`9a86ae2b`](https://github.com/vllm-project/DeepGEMM/commit/9a86ae2b78991f1c8e6e95945a591e1ca3b19ce7) (2.8.0), and update the Mega MoE call convention that comes with it. ⏎  ⏎ `9a86ae2b` is the tip of the fork's `dev` branch: the upstream 26/09 public release plus five fork-only commits — the SM120 (consumer Blackwell) por …[truncated]

### L3-073f883b56  (L3, 2026-09-15, sha 073f883b56a5, PR #56903)
TITLE: [Perf][DSpark] Collapse DeepSeek-V4.1 draft states before SP all-gather (#56903)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_mhc_kernels.py (+21/-0); vllm/models/deepseek_v41/nvidia/dspark.py (+2/-3)
LABELS: ready, deepseek, dflash, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Under sequence parallelism, the DeepSeek-V4.1 DSpark draft gathers BF16 `[T_local, 4, H]` residual streams and FP32 `[T_local, 4]` pre-mix coefficients before collapsing them into `[T, H]`. The draft returns only the collapsed pre-norm head states; it does not export a full-HC MTP buffer at this boundary. ⏎  ⏎ Collapse each token's streams locally, then gather only `[T_local, H]` and trim SP padding. This reduces the hidden-state gath …[truncated]

### L3-bb55077411  (L3, 2026-09-15, sha bb5507741155, PR #56962)
TITLE: [Perf][Kernel] Integrate Mega-mHC from DeepGEMM for DeepSeek V4.1  (reopen of #56255) (#56962)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/quantization/test_block_fp8.py (+6/-1); tests/kernels/test_mhc_kernels.py (+104/-1); vllm/models/deepseek_v41/nvidia/model.py (+34/-38); vllm/models/deepseek_v41/nvidia/ops/__init__.py (+2/-0); vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py (+156/-0); vllm/utils/deep_gemm.py (+14/-1)
LABELS: ready, ci/build, deepseek, kimi, k3, DSv4.1
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ - Reopen [vLLM #56255](https://github.com/vllm-project/vllm/pull/56255) (closed unmerged), rebased on the [#56876](https://github.com/vllm-project/vllm/pull/56876) base (`1b5a7abb58`) with the model path retargeted to `deepseek_v41`. ⏎ - Keep the original implementation commit separate from the follow-up adaptation; its elfutils dependency is already in the base. ⏎ - Preserve the current TileLang fused path for auxiliary capture and u …[truncated]

### L3-2c2cbd84fd  (L3, 2026-09-15, sha 2c2cbd84fdf1, PR #56408)
TITLE: [Frontend] Use XGrammar schema constraints for DeepSeek V4.1 (#56408)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/tool_parsers/test_structural_tag_registry.py (+77/-3); vllm/tool_parsers/structural_tag_registry.py (+13/-1)
LABELS: ready, tool-calling, deepseek, DSv4.1
BODY: ## Purpose ⏎  ⏎ The Python DeepSeek V4.1 builder currently accepts missing, repeated, undeclared, and incorrectly typed parameters even with `strict=true`. Prefer XGrammar's new V4.1 builtin to enforce parameter schemas and the `string="true|false"` wire types. Serving already owns the reasoning boundary, so the delegated grammar uses `reasoning="disabled"`. ⏎  ⏎ Requires the next XGrammar release. The existing format-only builder remains available for o …[truncated]

### L3-e45bb43984  (L3, 2026-09-15, sha e45bb43984b6, PR #56486)
TITLE: [Bugfix][KV Offload] Ignore pending chunks in invalid sliding windows (#56486)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py (+77/-1); vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py (+9/-3)
LABELS: bug, ready, kv-connector
BODY: ## Purpose ⏎  ⏎ A pending CPU KV write can unnecessarily defer a request even when an earlier sliding window is ready to load. For a two-chunk window, `[HIT, HIT, MISS, HIT_PENDING]` currently returns `None`: the reverse scan keeps the pending flag after the miss has already ruled out that incomplete suffix. The core scheduler therefore submits no load for the ready prefix. ⏎  ⏎ Track pending writes within the current consecutive window and clear that st …[truncated]

### L3-51918e252a  (L3, 2026-09-15, sha 51918e252af4, PR #56825)
TITLE: [Bugfix][SparseMLA] Fix piecewise cudagraph capture crash in index group (#56825)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/index_group.py (+12/-3); tests/v1/attention/test_sparse_mla_backends.py (+77/-1); vllm/forward_context.py (+9/-0); vllm/v1/hisparse/runtime.py (+2/-2)
LABELS: bug, ready, nvidia
BODY: ## What ⏎  ⏎ Fix an engine-startup crash when a sparse-MLA model captures decode batches with **piecewise** cudagraphs (e.g. GLM-5.3-Flash, a hybrid KDA + sparse-MLA model): ⏎  ⏎ ```log ⏎ File "vllm/v1/attention/backends/mla/index_group.py", line 133, in _convert_once ⏎     self.side_stream.wait_event(self.logical_topk_ready) ⏎ torch.AcceleratorError: CUDA error: invalid argument ⏎ ``` ⏎  ⏎ Full startup traceback (4x GB300, TP4, `load_format=dummy`; identical with r …[truncated]

### L3-4bac767695  (L3, 2026-09-15, sha 4bac767695d2, PR #52301)
TITLE: [Perf][Nemotron] Skip redundant latent-MoE all-reduce at TP>1 (~13% decode win) (#52301)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/runner/moe_runner.py (+9/-0); vllm/model_executor/models/nemotron_h.py (+12/-0)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎  ⏎ **Nemotron-3 decode regressed ~14% between v0.26.0 and v0.27.x at TP>1. This PR restores it.** The mechanism is generic but `NemotronH` is the only model that opts in, so it is the only one whose behaviour changes. No existing issue found. ⏎  ⏎ | tp4, concurrency 1, inter-token latency | v0.26.0 | v0.27.1 | + this PR | ⏎ |---|---|---|---| ⏎ | Super 120B-A12B BF16 (n=4/arm) | 4.07 ms | 4.64 ms (**+13.8%**) | **4.08 ms** (+0.15%) | ⏎ | Ultra 550B …[truncated]

### L3-ef5f7cd119  (L3, 2026-09-15, sha ef5f7cd119c6, PR #56560)
TITLE: [ROCm][DSv4.1][Perf] Dequantize the MXFP8 weight once when dot_scaled cannot be used (#56560)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_minimax_m3_amd_ops.py (+63/-0); vllm/model_executor/kernels/linear/mxfp8/rocm_native.py (+13/-5)
LABELS: rocm, ready, quantization, verified, minimax, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ `RocmDotScaledMxfp8LinearKernel` needs `K % 128 == 0` for `tl.dot_scaled`. Where that does not hold, `apply_weights` dequantizes the weight instead: ⏎  ⏎ ```python ⏎ w_bf16 = dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale) ⏎ out = torch.nn.functional.linear(x2d, w_bf16).to(x.dtype) ⏎ ``` ⏎  ⏎ `layer.weight` and `layer.weight_scale` are constant after loading, so that rebuilds the same tensor on every forward. This converts it once in `proce …[truncated]

### L3-000c7df9ff  (L3, 2026-09-15, sha 000c7df9ffd3, PR #53624)
TITLE: [KV Offload] Add KVCR secondary-tier adapter (#53624)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/scripts/install-kv-offload.sh (+18/-0); .buildkite/test_areas/misc.yaml (+1/-0); tests/v1/kv_offload/tiering/test_factory.py (+5/-1); tests/v1/kv_offload/tiering/test_kvcr_tier.py (+544/-0); vllm/v1/kv_offload/tiering/factory.py (+6/-0); vllm/v1/kv_offload/tiering/kvcr/__init__.py (+2/-0); vllm/v1/kv_offload/tiering/kvcr/manager.py (+635/-0)
LABELS: ready, ci/build, kv-connector
BODY: ## Purpose ⏎ Add an in-tree adapter that exposes [KVCR](https://github.com/ai-dynamo/kvcr) as a vLLM secondary tier. ⏎  ⏎ vLLM contains the factory registration and the translation between its secondary-tier interface and KVCR. KVCR remains an optional external package and owns the underlying storage, transfer, and policy implementation. Request-provided routing metadata is forwarded without making vLLM responsible for its semantics. ⏎  ⏎ See the [KVC …[truncated]

### L3-241e9391ed  (L3, 2026-09-15, sha 241e9391ed85, PR #55879)
TITLE: [Bugfix][Attention] Stabilize sparse-MLA DCP for GLM PCP evals (#55879)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer_sparse, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+122/-3); vllm/v1/attention/backends/mla/indexer.py (+39/-7); vllm/v1/attention/backends/utils.py (+7/-1); vllm/v1/attention/ops/dcp.py (+7/-0); tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP1-PCP4-DCP4-EP.yaml (+4/-1); tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP2-PCP2-EP.yaml (+2/-0); tests/v1/attention/test_dcp_a2a_pack_mask.py (+15/-1); tests/v1/attention/test_flashinfer_sparse_mla_workspace.py (+166/-0); tests/v1/attention/test_indexer_dcp_localize.py (+17/-0); tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py (+32/-0)
LABELS: bug, deepseek, nvidia, glm
BODY: ## Purpose ⏎  ⏎ Make the B200 GLM-5.2 PCP evaluation complete reliably across its TP/PCP/DCP configurations. ⏎  ⏎ Exact main build [#89033](https://buildkite.com/vllm/ci/builds/89033#01a0a3bf-3adb-4bec-82f2-da92bed83c19) first exposed two independent memory-budget failures after merged backend repair #56677. Successive literal-head and actual-branch gates then exposed the workspace, empty-rank, warmup-key, packed-pointer, and backend-combination edges de …[truncated]

### L3-35dc273072  (L3, 2026-09-15, sha 35dc273072ff, PR #55966)
TITLE: [ROCm][Spec Decode] Add Aiter MLA decode support non-causal draft block (#55966)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+145/-3); tests/kernels/attention/test_rocm_aiter_mla_fp8_support.py (+116/-1); tests/v1/attention/test_rocm_aiter_mla_fp8_decode_routing.py (+5/-0); tests/v1/attention/test_rocm_aiter_mla_mtp_split.py (+98/-0); vllm/_aiter_ops.py (+26/-1)
LABELS: rocm, ready
BODY: ## Summary ⏎ - Rebase of #53001 onto current `main`. Original PR is conflicting and this account cannot push to `JohnQinAMD/vllm-amd`. ⏎ - Pass the real causal mask into Aiter MLA decode (`get_mla_metadata_v1` / `mla_decode_fwd`) instead of flattening a non-causal DSpark draft query block. ⏎ - Probe `mla_decode_fwd` for a `causal` argument; keep Gluon and DCP verify causal-only; refuse fp8 2-token non-causal blocks until aiter folds that length. ⏎  ⏎  …[truncated]

### L3-18f8aa0465  (L3, 2026-09-15, sha 18f8aa04656d, PR #56845)
TITLE: [Refactor] Remove dead kernel code (#56845)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/triton_fp8_mqa_logits.py (+0/-262); vllm/v1/attention/ops/turboquant_soa/triton_turboquant_decode_v2.py (+0/-572); vllm/v1/attention/ops/turboquant_soa/triton_turboquant_unified_attention.py (+0/-15); vllm/model_executor/layers/mamba/ops/replayssm_config.py (+0/-15); vllm/models/glm5next/amd/ops/kpool_compress.py (+0/-44); vllm/models/inkling/amd/ops/lamport.py (+0/-766); vllm/models/inkling/amd/ops/mm_towers.py (+0/-190); vllm/models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/primitives.py (+0/-31)
LABELS: ready, kimi, k3, inkling
BODY: ## Purpose ⏎  ⏎ Remove dead kernel code

### L3-1257512305  (L3, 2026-09-15, sha 12575123059c, PR #56254)
TITLE: [DSA] Wire DeepGEMM sparse MQA logits into the DeepSeek V4.1 indexer (#56254)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+6/-1); vllm/v1/attention/backends/mla/sparse_indexer.py (+235/-0); tests/kernels/test_fused_indexer_q_rope_quant.py (+52/-0); tests/model_executor/layers/test_mla_short_prefill_indexer.py (+544/-0); vllm/config/attention.py (+9/-0); vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py (+469/-0); vllm/model_executor/layers/sparse_mqa_indexer.py (+209/-0); vllm/models/deepseek_v4/common/ops/fused_indexer_q.py (+22/-2); vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py (+15/-2); vllm/models/deepseek_v41/attention.py (+67/-20); (+3 more)
LABELS: performance, ready, deepseek, DSv4, kv-cache-manager, DSv4.1
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ DeepSeek V4.1 selects attention tokens in two levels: one indexer layer publishes the top candidate blocks of the context, and the indexer layers after it pick their top-k inside those blocks. Today the consumer layers still score the **whole** context with dense MQA logits and then mask everything outside the candidates. DeepGEMM 2.8 ships sparse MQA-logits kernels (deepseek-ai/DeepGEMM#432) that score only a given list of blocks, so …[truncated]

### L3-c6fa1f05d1  (L3, 2026-09-15, sha c6fa1f05d1bc, PR #56346)
TITLE: [Perf][Kernel] Add sampled filtering for persistent top-k (#56346)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_persistent_topk.py (+134/-0); csrc/libtorch_stable/persistent_topk.cuh (+52/-34); csrc/libtorch_stable/sampled_topk.cuh (+253/-0); csrc/libtorch_stable/topk.cu (+17/-1); tests/kernels/test_top_k_per_row.py (+100/-0)
LABELS: performance, ready, nvidia, DSv4
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Speed up long FP32 sparse-indexer decode top-k with sampling inspired by [DeepSelect](https://github.com/deepseek-ai/DeepSelect). A coalesced sample estimates a cutoff, survivors are compacted in shared memory, and exact FP32 selection finishes on those candidates. Too few or too many candidates trigger exact full-row selection. ⏎  ⏎ Enable sampling above 64 rows, with **actual valid length ≥98,304 for k=512 or ≥65,536 for k=1024/2048**,  …[truncated]

### L3-f547c23ec9  (L3, 2026-09-15, sha f547c23ec971, PR #56794)
TITLE: [Bugfix][Kimi-K3] Keep transient checkpoints out of prefix-cache eviction (#56794)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py (+8/-1); tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py (+71/-0); vllm/v1/core/single_type_kv_cache_manager.py (+15/-1)
LABELS: bug, ready, verified, kimi, k3, kv-cache-manager
BODY: ## Summary ⏎  ⏎ Keep transient KDA/Mamba prefill checkpoints out of the ordinary prefix-cache eviction set when sparse retention is explicitly active. ⏎  ⏎ This change targets Kimi-K3's hybrid attention path. ⏎  ⏎ ## Problem ⏎  ⏎ FlashKDA creates internal checkpoints so the current prefill can continue from a recurrent-state boundary without an additional scheduling step. A checkpoint in the middle of a prompt is useful to the current request, but it is not nece …[truncated]

### L3-0136df94b0  (L3, 2026-09-15, sha 0136df94b0d7, PR #56387)
TITLE: [Bugfix][Spec Decode][MoE] Avoid uninitialized EPLB state in DeepSeek V4.1 DSpark drafter (#56387)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_dspark_utils.py (+51/-0); vllm/v1/worker/gpu/spec_decode/dspark/utils.py (+26/-6)
LABELS: bug, speculative-decoding, ready, deepseek, nvidia, mrv2, dflash, DSv4.1
BODY: ## Purpose ⏎  ⏎ Fix DeepSeek V4.1 DSpark startup when EPLB is enabled for the target model. ⏎  ⏎ The target and DSpark draft use different routed-expert topologies (DeepSeek-V4.1-Flash target: 384 routed experts; DSpark drafter: 128). When target-model EPLB is enabled, the draft routers inherit `enable_eplb=True`, but the drafter is not registered with a compatible EPLB model state. Its router-level `EplbLayerState` therefore never receives `expert_load_ …[truncated]

### L3-1e47ec00d2  (L3, 2026-09-15, sha 1e47ec00d2e6, PR #56721)
TITLE: [Bugfix][Gemma 4] Don't read fft_length when profiling unified audio (#56721)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/processing/test_gemma4_unified.py (+43/-1); vllm/model_executor/models/gemma4_mm.py (+24/-1)
LABELS: bug, ready, multi-modality, verified
BODY: ## Purpose ⏎  ⏎ `Gemma4DummyInputsBuilder.get_dummy_mm_data` reads `processor.feature_extractor.fft_length` unconditionally to size the dummy audio item: ⏎  ⏎ ```python ⏎ if num_audios > 0: ⏎     audio_len = processor.feature_extractor.fft_length ⏎ ``` ⏎  ⏎ `Gemma4AudioFeatureExtractor.__init__` sets `self.fft_length`. `Gemma4UnifiedAudioFeatureExtractor` defines no such attribute, so the unified (encoder-free) Gemma 4 variant fails during memory profiling, before …[truncated]

### L3-9446ea1680  (L3, 2026-09-15, sha 9446ea168067, PR #53458)
TITLE: [Bugfix][Spec Decode] Only create draft_id_to_target_id when draft vocab differs (#53458)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/deepseek_eagle3.py (+8/-4); vllm/model_executor/models/llama_eagle3.py (+8/-4); vllm/model_executor/models/qwen3_eagle3.py (+8/-4)
LABELS: bug, speculative-decoding, ready, llama, qwen, deepseek
BODY: ## Purpose ⏎  ⏎ We found a bug that makes full-vocab EAGLE3 drafters (`draft_vocab_size == target vocab_size`, e.g. `Inferact/MiniMax-M3-EAGLE3-GQA` and `nvidia/Qwen3-30B-A3B-Thinking-2507-Eagle3`) run a redundant vocab-remapping step after the lm_head, degrading decode performance with no effect on the output. ⏎  ⏎ `compute_logits` in `llama_eagle3.py` has a fast path and a slow path. Full-vocab drafts were meant to take the fast path, but erroneously t …[truncated]

### L3-836bb3839f  (L3, 2026-09-15, sha 836bb3839ffe, PR #56969)
TITLE: [Bugfix] Prevent out-of-bounds access in FlashInfer SM90 sparse MLA mixed batches (#56969)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer_sparse
FILES: vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py (+8/-0); tests/v1/attention/test_flashinfer_mla_sparse_sm90.py (+19/-7)
LABELS: bug, ready, nvidia
BODY: PLEASE FILL IN THE PR DESCRIPTION HERE ENSURING ALL CHECKLIST ITEMS (AT THE BOTTOM) HAVE BEEN CONSIDERED. ⏎  ⏎ ## Purpose ⏎  ⏎ Prevent out-of-bounds access in FlashInfer SM90 sparse MLA mixed batches ⏎  ⏎  ⏎ ## Test Plan ⏎ On 4 * H20-3e ⏎ ``` ⏎ vllm serve /mnt/nvme/shared/models/ZhipuAI/GLM-5.3-Flash \ ⏎     --host 127.0.0.1 \ ⏎     --port 8000 \ ⏎     --tensor-parallel-size 4 \ ⏎     --load-format instanttensor \ ⏎     --max-model-len 8192 \ ⏎     --max-num-bat …[truncated]

### L3-0fefffc934  (L3, 2026-09-15, sha 0fefffc93466, PR #50494)
TITLE: [KVConnector][NIXL] Support attention-HMA layouts in pipeline-parallel push prefill (#50494)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/design/nixl_kv_push_connector.md (+44/-0); docs/features/nixl_connector_compatibility.md (+18/-0); tests/v1/kv_connector/unit/test_nixl_connector_hma.py (+1/-0); tests/v1/kv_connector/unit/test_nixl_desc_geometry.py (+28/-4); tests/v1/kv_connector/unit/test_nixl_push_connector.py (+398/-2); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+216/-40); vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py (+5/-2); vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py (+3/-1)
LABELS: documentation, ready, kv-connector
BODY: ## Purpose ⏎  ⏎ Enable attention HMA with pipeline-parallel prefill in `NixlPushConnector`, ⏎ following #45880. Producer PP stages and a PP1 decoder may pool the same layers ⏎ into different physical KV-cache regions. This PR matches layers by name and ⏎ builds descriptors using each side's region geometry. ⏎  ⏎ This updates the existing push-mode PR. Pull-mode work in #43368/#48263 and ⏎ the earlier #47981 prototype use different implementations. Packed MLA is  …[truncated]

### L3-676650397f  (L3, 2026-09-15, sha 676650397ff7, PR #53867)
TITLE: [Feature][PCP] Support decode-only FULL CUDA graphs (#53867)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_pcp_manager.py (+155/-4); vllm/v1/worker/gpu/cudagraph_utils.py (+5/-4); vllm/v1/worker/gpu/input_batch.py (+6/-2); vllm/v1/worker/gpu/model_runner.py (+20/-15); vllm/v1/worker/gpu/pcp_manager.py (+97/-17)
LABELS: ready, nvidia, mrv2
BODY: ## Purpose ⏎  ⏎ Enable prefill context parallelism (PCP) to use decode-only FULL CUDA graphs through `FULL_DECODE_ONLY` and `FULL_AND_PIECEWISE`. ⏎  ⏎ This change: ⏎  ⏎ - keeps PCP graph capture and replay on persistent rank-local input buffers; ⏎ - selects PCP persistent capture inputs from the graph mode and typed `PCPManager` in one linear branch; ⏎ - keeps standard block-table, slot-mapping, and DCP preparation together in the common capture path; ⏎ - …[truncated]

### L3-031f5810c1  (L3, 2026-09-15, sha 031f5810c195, PR #56545)
TITLE: [Build][NVIDIA] Update public Rubin dependencies and MSA compatibility (#56545)
SOURCES: dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: cmake/external_projects/fmha_sm100.cmake (+1/-1); docker/Dockerfile (+1/-1); requirements/rubin-prerelease.txt (+16/-6)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ Update public Rubin dependencies and the MSA revision for Quack/CuTe DSL compatibility. ⏎  ⏎ - Pin CUDA Python/bindings to 13.4.1, Quack to 0.6.5, FlashInfer Python/cubin to 0.6.18.post1, its CUDA 13.4 JIT cache to 0.6.18.post1+cu134, and NIXL to 1.4.1. Add the official FlashInfer indexes. ⏎ - Advance the official MSA revision to `f355c37eb4e1413f21ee2ad8bbad25079e6bef9d`, incorporating [MSA#12](https://github.com/vllm-project/MSA/pull/12). …[truncated]

### L3-b6ce714354  (L3, 2026-09-15, sha b6ce7143548a, PR #54098)
TITLE: [Bugfix] Format kernel-import errors eagerly so warning_once does not retain them (#54098)
SOURCES: release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cpu.py (+8/-4); vllm/platforms/cuda.py (+3/-1); vllm/platforms/interface.py (+1/-1)
LABELS: bug, ready, cpu, nvidia
ISSUES: #54096 [Bug]: logger.warning_once with a live exception argument leaks the exception's entire traceback (pins the LLM instance on platforms without vllm._C)`
BODY: FIX #54096 ⏎  ⏎ ## Purpose ⏎  ⏎ `logger.warning_once` caches its arguments forever (`lru_cache`). The kernel import fallbacks passed the live `ImportError` as a lazy format argument, so the cache retained the exception — and through `__traceback__` every frame and local on the raising path. `import_kernels` runs inside `LLM.__init__`, so on platforms without compiled `vllm._C` kernels the pinned locals included the `LLM` instance under construction: it c …[truncated]

### L3-dffbb714e4  (L3, 2026-09-15, sha dffbb714e4e8, PR #56743)
TITLE: [ROCm][Perf] Optimize DSV4.1 K=512 decode top-k on gfx950 (#56743)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+12/-0); csrc/libtorch_stable/sampler.cu (+205/-58); tests/kernels/test_top_k_per_row.py (+189/-0)
LABELS: rocm, ready, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Part of #56506. ⏎  ⏎ DeepSeek-V4.1-Flash uses K=512 for decode indexer top-k, which misses the ⏎ existing gfx950 tuning and falls back to the generic 10-split kernel. Add a ⏎ ROCm/gfx950-only K=512 path for up to 384 rows and 1M columns. It selects ⏎ active splits from device sequence lengths so FULL graph replay remains valid, ⏎ reduces masked `-inf` histogram contention, and uses a smaller final sort when ⏎ possible. Other architectures, K values …[truncated]

### L3-df42d112ee  (L3, 2026-09-15, sha df42d112ee88, PR #57041)
TITLE: [Config] Infer HiSparse attention config from HiSparseConnector (#57041)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/test_config.py (+102/-0); vllm/config/attention.py (+3/-1); vllm/config/vllm.py (+8/-1)
LABELS: ready
BODY: ## What ⏎  ⏎ HiSparse needed to be turned on twice: `HiSparseConnector` in `--kv-transfer-config` *and* `--attention-config '{"hisparse_config":{}}'`. The empty `{}` carries no information — configuring the connector already implies HiSparse, and `hisparse_config` without the connector is rejected at startup anyway (`hisparse/layout.py:78`). ⏎  ⏎ `VllmConfig.__post_init__` now defaults `attention_config.hisparse_config = HiSparseConfig()` when `kv_tr …[truncated]

### L3-f2aad6aa70  (L3, 2026-09-15, sha f2aad6aa7074, PR #56888)
TITLE: [MRV2] Buffer util simplifications (#56888)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.flashmla_sparse
FILES: vllm/model_executor/layers/attention/mla_attention.py (+17/-18); vllm/third_party/flash_linear_attention/ops/index.py (+3/-2); vllm/v1/attention/backends/flashinfer.py (+3/-14); vllm/v1/attention/backends/mla/flashmla_sparse.py (+6/-3); vllm/v1/attention/backends/mla/indexer.py (+3/-4); tests/utils_/test_torch_utils.py (+25/-0); tests/v1/attention/test_indexer_dcp_localize.py (+1/-1); tests/v1/worker/test_gpu_pcp_manager.py (+3/-3); vllm/model_executor/models/diffusion_gemma.py (+3/-3); vllm/models/dots3_note/nvidia/attention.py (+15/-7); (+11 more)
LABELS: speculative-decoding, ready, needs-rebase, cpu, nvidia, mrv2
BODY: - Unify similar `async_copy_to_gpu` and `async_tensor_h2d` methods (remove the former) ⏎ - Adjust `CpuGpuBuffer` for staged+pinned transfers ⏎ - Ensure pinned buffers are used for h2d copy in a few more places

### L3-af1c01499b  (L3, 2026-09-15, sha af1c01499b28, PR #56904)
TITLE: [Bugfix] Make GPU sync checks safe under torch.compile (#56904)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_voxtral_realtime.py (+2/-1); tests/utils_/test_gpu_sync_debug.py (+18/-0); vllm/utils/gpu_sync_debug.py (+12/-2)
LABELS: bug, ready, torch.compile, multi-modality, mistral
BODY: ## Summary ⏎  ⏎ `gpu_sync_debug`'s patched tensor-copy wrappers currently read a `ContextVar` ⏎ before checking whether Dynamo is tracing. A full-graph compile therefore fails ⏎ with an unsupported `ContextVar.get()` instead of bypassing sync checks as ⏎ intended. ⏎  ⏎ This change: ⏎  ⏎ - checks `torch.compiler.is_compiling()` before accessing either `ContextVar`; ⏎ - routes the original C++ `Tensor.to` descriptor through a weak-referenceable ⏎   Python callable regis …[truncated]

### L3-bfd713bf8b  (L3, 2026-09-16, sha bfd713bf8b38, PR #56931)
TITLE: [Rust Frontend] Support HF config overrides `--hf-overrides` (#56931)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: rust/Cargo.lock (+1/-0); rust/src/chat/src/backend/hf.rs (+9/-13); rust/src/chat/src/backend/mod.rs (+3/-0); rust/src/cmd/Cargo.toml (+1/-0); rust/src/cmd/src/cli.rs (+17/-0); rust/src/cmd/src/cli/tests.rs (+85/-0); rust/src/cmd/src/cli/unsupported.rs (+0/-5); rust/src/managed-engine/src/cli.rs (+5/-0); rust/src/server/examples/external_engine_openai_qwen.rs (+1/-0); rust/src/server/src/config.rs (+3/-0); (+6 more)
LABELS: ready, rust
BODY: ## Purpose ⏎  ⏎ Support `--hf-overrides` in Rust serving, Python bootstrap, and render-only mode. Apply JSON Merge Patch to a frontend-owned temporary config file; clones retain the file through backend lifetime. Managed serving forwards the original object to Python, preserving its existing override semantics. ⏎  ⏎ Related open PRs cover Docker/tokenizer or Python-engine changes; this adds Rust config loading and forwarding. AI-assisted implementation a …[truncated]

### L3-d6a1677d55  (L3, 2026-09-16, sha d6a1677d5504, PR #56935)
TITLE: [Model][DSv4.1] FlashMLA mega attention and the NVFP4 compressed KV cache (#56935)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.dispatch.registry
FILES: vllm/models/deepseek_v41/nvidia/flashmla.py (+2/-1); vllm/v1/attention/backends/mla/sparse_swa.py (+4/-1); vllm/v1/attention/backends/registry.py (+3/-0); benchmarks/kernels/benchmark_dsv41_mega_attn.py (+272/-0); csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu (+134/-72); csrc/libtorch_stable/ops.h (+2/-1); csrc/libtorch_stable/torch_bindings.cpp (+8/-1); tests/kernels/test_compressor_kv_cache.py (+133/-6); tests/kernels/test_dsv41_mega_attn_layouts.py (+71/-0); tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py (+183/-0); (+10 more)
LABELS: performance, ready, deepseek, nvidia, DSv4.1
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ Supersedes #56344, which closed when its base branch `dsv41-optimized` was deleted. ⏎  ⏎ FlashMLA's mega-attention kernel does Q RoPE, sparse attention, the output's inverse RoPE and its FP8 cast in one launch, writing straight into the buffer `wo_a` consumes. It arrives as `FlashMLAMegaAttnBackend` (`FLASHMLA_MEGA_ATTN_DSV41`) with its own attention layer, plus the V4.1 NVFP4 compressed KV record that only this kernel can read — and, as  …[truncated]

### L3-b3b13c1292  (L3, 2026-09-16, sha b3b13c12923f, PR #57112)
TITLE: [ROCm][CI] Fix MLA RoPE fused-kernel tests for TheRock image (#57112)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/core/test_rotary_embedding_mla_cache_fused.py (+121/-16)
LABELS: rocm, ready
BODY: The `Core Operations Kernels` test group fails on TheRock image (see [here](https://buildkite.com/vllm/amd-ci/builds/12914/list?sid=01a09eed-c3fc-4155-8ae7-ad55792cdc2c&open=false)). This has been root-caused to the Triton 3.8 upgrade. ⏎ Specifically, in the computation of RoPE, we compute terms similar to  `x * cos - y * sin`. In Triton 3.7, the LLVM backend lowers to `FMA_F16(x, cos, MUL_F16(-y, sin))`, which is the same as the HIP reference fus …[truncated]

### L3-f30a195bbb  (L3, 2026-09-16, sha f30a195bbb15, PR #57050)
TITLE: [Bugfix] Fix incorrect Mamba block allocation estimate that prevents request admission (#57050)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_single_type_kv_cache_manager.py (+57/-0); vllm/v1/core/single_type_kv_cache_manager.py (+4/-4)
LABELS: bug, ready, kv-cache-manager
BODY: ## Purpose ⏎ It is found for Kimi K3, a long request can be rejected admission when loading prefix cache from external KV storage (e.g., Mooncake) and get stuck in the waiting queue forever, causing entire system to stall. Investigation shows the bug is in Mamba's block allocation estimation call `get_num_blocks_to_allocate`, which incorrectly calculates the physical blocks needed for such requests, potentially a regression caused by https://githu …[truncated]

### L3-e97ff80613  (L3, 2026-09-16, sha e97ff8061344, PR #54016)
TITLE: [BugFix][PCP] Handle missing DP metadata in one-sided EP (#54016)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py (+3/-2)
LABELS: bug, ready, nvidia
BODY: Fix FlashInfer NVLink one-sided expert parallelism when running with `DP=1`. ⏎  ⏎ With DP=1, `ForwardContext.dp_metadata` may be None. The one-sided EP path currently asserts that it exists, causing initialization or execution to fail. ⏎  ⏎ This PR: ⏎ - Returns `None` when DP metadata is unavailable. ⏎ - Uses the caller’s existing fallback to the local token count, `a1.shape[0]`. ⏎ - Preserves the existing behavior for `DP>1.` ⏎ - Adds tests for both mis …[truncated]

### L3-651a88c09a  (L3, 2026-09-16, sha 651a88c09ac6, PR #57212)
TITLE: [CI] Ignore ruff D209, rejoin the docstrings it split, and silence incompatible-rule warnings (#57212)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_unified, L3.mla.common_v1, L3.mla.flashmla_sparse, L3.dispatch.abstract_interface
FILES: benchmarks/benchmark_hidden_state_extraction.py (+1/-2); benchmarks/fused_kernels/merge_attn_states_benchmarks.py (+1/-2); benchmarks/kernels/benchmark_moe_defaults.py (+1/-2); benchmarks/kernels/benchmark_vocab_parallel_embedding.py (+1/-2); docs/mkdocs/gen_files/generate_examples.py (+2/-4); examples/features/logits_processor/custom_req.py (+2/-4); examples/features/logits_processor/custom_req_init.py (+2/-4); examples/rl/rlhf_sharded_rdt_small_ep.py (+1/-2); pyproject.toml (+3/-1); setup.py (+1/-2); (+683 more)
LABELS: documentation, performance, rocm, structured-output, frontend, speculative-decoding, ray, torch.compile, ci/build, multi-modality
BODY: ## Purpose ⏎  ⏎ `pre-commit run --all-files` currently fails on `main` (`1085b64425`): three docstrings landed alongside #52136, which selected the whole `pydocstyle` `D` family, so `ruff check` flags them for D209 (`new-line-after-last-paragraph`). ⏎  ⏎ This PR drops D209 rather than reformatting them, and undoes the reformatting the rule already caused. ⏎  ⏎ D209 only moves the closing `"""` of a multi-line docstring onto a line of its own. That costs a li …[truncated]

### L3-975dca5bb5  (L3, 2026-09-16, sha 975dca5bb5db, PR #57140)
TITLE: [Perf][GDN] Scatter mixed speculative outputs into the caller buffer (#57140)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+3/-7)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Mixed speculative/non-speculative Qwen GDN batches currently allocate `merged_out`, scatter both output partitions into it, and copy the complete tensor into `core_attn_out`. This change scatters directly into the caller's buffer, removing one output-sized allocation and the final full-output copy while preserving token order, buffer aliases, and padding rows. ⏎  ⏎ The opportunity was identified while reviewing a submission to a [Pareton  …[truncated]

### L3-1085b64425  (L3, 2026-09-16, sha 1085b64425a9, PR #57083)
TITLE: [Benchmark] Retire stale benchmarks and consolidate RMSNorm (#57083)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/cutlass_benchmarks/utils.py (+0/-39); benchmarks/cutlass_benchmarks/w8a8_benchmarks.py (+0/-372); benchmarks/cutlass_benchmarks/weight_shapes.py (+0/-46); benchmarks/kernels/benchmark_rmsnorm.py (+0/-255); benchmarks/kernels/ir/shapes.py (+3/-2); benchmarks/overheads/benchmark_hashing.py (+0/-64); tools/pre_commit/check_forbidden_imports.py (+0/-2)
LABELS: performance, ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ Retire the legacy CUTLASS benchmark directory and its stale pickle-import exceptions, remove the hashing profiler that still searches for the removed `hash_of_block` function, and consolidate plain/residual RMSNorm benchmarking in the IR harness. ⏎  ⏎ The standalone RMSNorm script is replaced by the shared shape grid for `rms_norm` and `fused_add_rms_norm`; `--ops rms_norm` selects both. This uses the registered IR providers and retires t …[truncated]

### L3-8c1557a79c  (L3, 2026-09-16, sha 8c1557a79c53, PR #52136)
TITLE: Add `pydocstyle` to the `ruff` rules (#52136)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fa4_cutedsl, L3.flash_attn.fa_utils, L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.decode_attention, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashmla_sparse, L3.mla.rocm_aiter, L3.mla.rocm_aiter_sparse, L3.dispatch.selector, L3.dispatch.registry, L3.dispatch.abstract_interface, L3.flex_attention
FILES: .buildkite/lm-eval-harness/test_lm_eval_correctness.py (+1/-2); .buildkite/performance-benchmarks/scripts/compare-json-results.py (+2/-4); .buildkite/performance-benchmarks/scripts/convert-results-json-to-markdown.py (+1/-2); .buildkite/scripts/generate-nightly-index.py (+8/-13); benchmarks/attention_benchmarks/batch_spec.py (+14/-14); benchmarks/attention_benchmarks/benchmark.py (+10/-10); benchmarks/attention_benchmarks/common.py (+7/-8); benchmarks/attention_benchmarks/mla_runner.py (+29/-22); benchmarks/attention_benchmarks/runner.py (+6/-6); benchmarks/benchmark_batch_invariance.py (+2/-4); (+1992 more)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready, ray
BODY: Enables the `D` ruleset in ruff and fixes the resulting violations across the codebase. ⏎  ⏎ ### Config ⏎  ⏎ `D100`–`D107` are ignored, so this does **not** require a docstring on anything that lacks one today. Six more rules are ignored because their remaining violations were either impossible to autofix or actively wrong to autofix: ⏎  ⏎ | Rule | Why skipped | ⏎ |---|---| ⏎ | `D205`, `D400`, `D415` | All treat a summary sentence that merely *wraps* as  …[truncated]

### L3-03f67b3ad1  (L3, 2026-09-16, sha 03f67b3ad1e6, PR #56568)
TITLE: [Perf][DSV4.1] Pad shared experts for native MegaMoE fusion (#56568)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/test_deepseek_v4_mega_moe.py (+80/-18); vllm/models/deepseek_v4/nvidia/model.py (+48/-5)
LABELS: ready, deepseek, verified, DSv4, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ https://github.com/vllm-project/vllm/issues/56217 ⏎  ⏎ DeepSeek-V4.1-Flash's 2304-wide shared expert cannot fuse with MegaMoE's routed experts padded to 2560. Pad shared weights with zeros and unit scales to enable fusion in all 40 layers, preserving checkpoint-shaped parameters and the unsupported-layout fallback. ⏎  ⏎ Extends #53040. Checks of #45861 and related open PRs found no fix for this padding gap; #53567 and #54049 cover diffe …[truncated]

### L3-9f9e1dac26  (L3, 2026-09-16, sha 9f9e1dac26ff, PR #57152)
TITLE: [Bugfix][Model] Restore causal image SWA for DeepSeek V4.1 (#57152)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v41/nvidia/flashmla.py (+2/-19); vllm/v1/attention/backends/mla/sparse_swa.py (+6/-4); tests/v1/attention/test_deepseek_v4_swa_visible.py (+81/-6); vllm/models/deepseek_v41/attention.py (+0/-8); vllm/models/deepseek_v41/common/ops/cache_utils.py (+31/-108); vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py (+2/-17); vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py (+0/-4); vllm/transformers_utils/configs/deepseek_v41.py (+0/-8)
LABELS: bug, ready, deepseek, nvidia, DSv4.1
DEEP_STUDY: deep-study correctness case vllm:9f9e1dac26: class=shape_alignment_edge; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Summary ⏎  ⏎ Restore causal sliding-window attention for image tokens in DeepSeek V4.1 VL. ⏎  ⏎ V4.1 inherited the image visibility rules used by DeepSeek V4 Vision-Exp: tokens inside an image could attend to earlier image tokens outside the sliding window and to future image tokens. This differs from the [official V4.1 reference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/infere …[truncated]

### L3-bc0f47cd03  (L3, 2026-09-16, sha bc0f47cd03d6, PR #57132)
TITLE: [ROCm][Bugfix] Revert #56433 + #51692 to fix accuracy breakdown for DeepSeek-V4 (#57132)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+4/-3); tests/models/test_deepseek_v4_vl_rocm.py (+0/-125); vllm/_aiter_ops.py (+5/-27); vllm/model_executor/kernels/linear/__init__.py (+0/-4); vllm/model_executor/kernels/linear/scaled_mm/aiter.py (+0/-170); vllm/model_executor/layers/linear.py (+3/-3); vllm/models/deepseek_v4/amd/model.py (+6/-11); vllm/models/deepseek_v4/amd/rocm.py (+16/-59)
LABELS: bug, rocm, ready, deepseek, DSv4
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 56433;51692 reason=correctness_or_accuracy
BODY: ## Purpose ⏎  ⏎ When testing `gsm8k` for DeepSeek-V4 on AMD GPUs using latest vLLM `main`, the acc value becomes `0`: ⏎  ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.0053|±  |0.0020| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.0008|±  |0.0008| ⏎  ⏎ After reverting https://githu …[truncated]

### L3-30b847c1fa  (L3, 2026-09-16, sha 30b847c1fa54, PR #57204)
TITLE: [Perf][DSV4.1] Remove MegaMoE padding and shared padding workaround (#57204)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/test_deepseek_v4_mega_moe.py (+37/-92); vllm/models/deepseek_v4/nvidia/model.py (+5/-65)
LABELS: ready, deepseek, DSv4, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Remove unnecessary MegaMoE intermediate-width padding for DeepSeek-V4.1-Flash and revert the production workaround added in #56568. Routed and shared experts both retain the checkpoint width of 2304, so native shared-expert fusion remains enabled without padding either to 2560. ⏎  ⏎ The bundled DeepGEMM uses `layout::Data(..., false)` for activation-scale rows in `include/deep_gemm/layout/mega_moe.cuh`; the old 16-byte TMA-row alignment r …[truncated]

### L3-6a2fdf9ac6  (L3, 2026-09-16, sha 6a2fdf9ac6f9, PR #56034)
TITLE: [Model] Optimize Sarvam MLA routing and preserve FP32 router logits (#56034)
SOURCES: subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/sarvam.py (+21/-10)
LABELS: ready, verified
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Fix Sarvam MLA's missing flat-routing configuration and preserve the configured router-logit dtype through expert selection. On the tested BF16 B200 setup, these changes enable the supported TRTLLM MoE path and reduce router overhead while removing an unnecessary BF16 narrowing of FP32 logits. ⏎  ⏎ The source diff is confined to `vllm/model_executor/models/sarvam.py`: ⏎  ⏎ - Default absent or `None` `n_group` and `topk_group` to `1`. Th …[truncated]

### L3-19b6ff62f2  (L3, 2026-09-17, sha 19b6ff62f24d, PR #56726)
TITLE: [ROCm][Bugfix] Ignore descales for unquantized AITER caches (#56726)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+26/-15); tests/v1/attention/test_attention_backends.py (+59/-7)
LABELS: bug, rocm, ready, quantization, verified
BODY: ## Problem ⏎ `ROCM_AITER_FA` passes the attention layer's K/V scale tensors to AITER decode kernels even when the KV cache is unquantized. If those otherwise-unused tensors contain non-neutral values—for example, stale values left by dummy-weight initialization after loading real weights—AITER treats them as descales and corrupts attention output. In our MI300X reproduction, this reduced the post-reload InfoVQA score from the expected `0.952` to `0 …[truncated]

### L3-9854b580df  (L3, 2026-09-17, sha 9854b580dfe5, PR #56799)
TITLE: [Bugfix][KV Offload] Compact canonical MLA rows (#56799)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/offloading_connector/test_config.py (+40/-0); tests/v1/kv_connector/unit/offloading_connector/test_worker.py (+6/-2); vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py (+14/-0); vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py (+3/-1)
LABELS: bug, kv-connector
ISSUES: #56772 [KV Connector][Offloading] Canonical MLA+DSA secondary transfers use TP-dependent row sizes
BODY: Fixes #56772. ⏎  ⏎ Use a single CPU copy for certified canonical MLA/DSA caches, while retaining canonical writer selection. This gives compatible TP sizes equal P2P row lengths. ⏎  ⏎ No open PR addresses this allocation gap. ⏎  ⏎ ## Tests ⏎  ⏎ - 3× H200, GLM-5.2-FP8 (four layers), TP1 ↔ TP2: baseline rejects unequal row lengths; patched runs reuse 2,432 tokens through P2P in each direction with matching uncached completions. ⏎ - 397 offloading tests passed; pre-c …[truncated]

### L3-4e5ffda19d  (L3, 2026-09-17, sha 4e5ffda19d89, PR #57285)
TITLE: [Bugfix] Isolate supplemental FlashInfer BF16 autotuning (#57285)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+35/-8); vllm/model_executor/warmup/kernel_warmup.py (+40/-3)
LABELS: bug, ready, nvidia, verified
BODY: ## Purpose ⏎  ⏎ The supplemental 32-token FlashInfer BF16 warmup currently applies its bounded `tuning_buckets` to every operation in the dummy run. An MXFP8 drafter GEMM with M=40 can consequently inherit the BF16 bucket cap and reuse a low-M tactic that does not support its input. ⏎  ⏎ Run this supplemental pass after the full-model autotune context exits. Only `flashinfer_bf16_mm` enables tuning and installs the BF16 buckets; other operations execute  …[truncated]

### L3-08633cb5cd  (L3, 2026-09-17, sha 08633cb5cd77, PR #55867)
TITLE: [Qwen3.8-Flash-Next] Enable FP8 TP with FlashInfer TRTLLM MoE (#55867)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_fp8_tp_loading.py (+311/-0); vllm/model_executor/layers/fused_moe/config.py (+2/-0); vllm/model_executor/layers/fused_moe/oracle/fp8.py (+77/-0); vllm/model_executor/layers/fused_moe/routed_experts.py (+33/-2); vllm/model_executor/layers/quantization/fp8.py (+11/-32)
LABELS: ready, qwen, quantization
BODY: It will pad the weight like to avoid the scale mismatching like: ⏎  ⏎ ``` ⏎ rank 0: A | B ⏎ rank 1: C | D ⏎ rank 2: E | pad ⏎ rank 3: pad | pad ⏎ ``` ⏎  ⏎ AI generate: ⏎  ⏎ ## Purpose ⏎  ⏎ Enable FP8 TP2/TP4/TP8 with FlashInfer TRTLLM for Qwen3.8-Flash-Next. Round each rank's allocation to 128 before backend selection, then load weights and scales using the padded allocation as the TP stride, without requantization. This uses the generic MoE loader without a  …[truncated]

### L3-438434b5b5  (L3, 2026-09-17, sha 438434b5b5f8, PR #56849)
TITLE: [ROCm][Perf] Insert MiniMax-M3 sparse-PA K/V without a contiguous copy (#56849)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_minimax_m3_amd_ops.py (+76/-0); vllm/models/minimax_m3/amd/model.py (+28/-2)
LABELS: rocm, ready, minimax
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ - Pass the strided K/V slices straight to AITER `reshape_and_cache` on the MiniMax-M3 sparse-PA path instead of calling `.contiguous()` on them first. ⏎ - Add `_kv_insert_operand`, which returns the tensor as-is when the kernel's layout requirements are met and copies otherwise. ⏎ - No kernel change, no AITER dependency, no new env var, no change to the NVIDIA path. ⏎  ⏎ ## Motivation ⏎  ⏎ `MiniMaxM3SparseAttention._insert_aiter_sparse_pa …[truncated]

### L3-e30bf70c5d  (L3, 2026-09-17, sha e30bf70c5dfb, PR #57380)
TITLE: [ROCm][CI] Fix Entrypoints Integration (Pooling) tests on TheRock image (#57380)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py (+17/-41)
LABELS: rocm
BODY: ## Purpose ⏎ `tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py` has a number of [failures on TheRock image](https://buildkite.com/vllm/amd-ci/builds/13085/list?sid=01a0ae47-79de-4154-ad16-daf19cd84150&tab=output). ⏎  ⏎ This was root-caused to a slight difference in codegen on Triton 3.8 vs Triton 3.7: namely, `chunked_prefill_paged_decode`'s `_fwd_kernel` has a slightly different FP32 accumulation order which leads to a 1-ULP BF …[truncated]

### L3-667b26e50b  (L3, 2026-09-17, sha 667b26e50bc3, PR #57192)
TITLE: [Bugfix][ROCm][GLM-5.3-Flash] Apply deferred tilelang.jit already on attribute access (#57192)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_mhc_tilelang_jit.py (+29/-7); vllm/tilelang_utils/__init__.py (+35/-16)
LABELS: bug, rocm, ready, glm
BODY: ## Purpose ⏎  ⏎ Required to run GLM-5.3-Flash on MI350, otherwise errors during warmup with: ⏎  ⏎ ```bash ⏎ File "vllm/model_executor/warmup/jit_warmup.py", line 969, in compile_many ⏎   self.compile(compile_key) ⏎ File "vllm/model_executor/warmup/jit_warmup_tilelang_helper.py", line 141, in launch ⏎   return compile_tilelang(jit_impl, *args, **call_kwargs) ⏎ File "vllm/model_executor/warmup/jit_warmup_tilelang_helper.py", line 104, in compile_tilelang ⏎   …[truncated]

### L3-41f9104fa6  (L3, 2026-09-17, sha 41f9104fa649, PR #56266)
TITLE: [DSv4.1] Integrate Mega-Gate from DeepGEMM (#56266)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/kernels.yaml (+1/-0); tests/kernels/test_mhc_kernels.py (+3/-2); tests/models/test_deepseek_v4_mega_moe.py (+138/-12); vllm/models/deepseek_v4/nvidia/dspark.py (+14/-0); vllm/models/deepseek_v4/nvidia/model.py (+110/-19); vllm/models/deepseek_v4/nvidia/mtp.py (+20/-1); vllm/models/deepseek_v41/nvidia/dspark.py (+14/-0); vllm/models/deepseek_v41/nvidia/model.py (+15/-1); vllm/utils/deep_gemm.py (+54/-0)
LABELS: new-model, rocm, speculative-decoding, ready, torch.compile, ci/build, tool-calling, deepseek, kv-connector, nvidia
BODY: ## Summary ⏎  ⏎ Integrate DeepGEMM Mega-Gate into the DeepSeek V4/V4.1 DeepGEMM MegaMoE path, fusing the gate GEMM and expert selection. Prepare routing metadata once per model forward and reuse it across layers. Preserve hash routing for V4, image routing bias for V4.1, draft-layer expert counts, MTP warmup, and decoder auxiliary-state capture. ⏎  ⏎ Two integration fixes are included: ⏎  ⏎ - Let CMake FetchContent update an existing DeepGEMM checkout  …[truncated]

### L3-2e50824766  (L3, 2026-09-17, sha 2e50824766ea, PR #48956)
TITLE: [Bugfix] Fall back to native sampling when FlashInfer cannot target the GPU (#48956)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/sample/test_topk_topp_sampler.py (+38/-0); vllm/v1/sample/ops/topk_topp_sampler.py (+38/-1)
LABELS: bug, ready, v1
ISSUES: #42393 [Installation]: RuntimeError: FlashInfer requires GPUs with sm75 or higher when running vllm server
BODY: ## Purpose ⏎  ⏎ Fixes #42393. ⏎  ⏎ On SM 12.x GPUs with a CUDA toolkit older than 12.9 (e.g. RTX 5090 + nvcc 12.8), FlashInfer swallows the real arch-detection error while building its compilation context: `_normalize_cuda_arch` raises `"SM 12.x requires CUDA >= 12.9"`, but the `except Exception` in `CompilationContext.__init__` downgrades it to a warning and leaves `TARGET_CUDA_ARCHS` empty. Every subsequent JIT spec then fails `check_cuda_arch()` with  …[truncated]

### L3-0eae9acd4d  (L3, 2026-09-17, sha 0eae9acd4d01, PR #56590)
TITLE: [Bugfix][ROCm][MoE] Fall back instead of crashing when AITER MoE is requested for a non-gated (is_act_and_mul=False) model (#56590)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_unquantized_backend_selection.py (+60/-0); vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+27/-1)
LABELS: bug, rocm, ready
BODY: ## Purpose ⏎  ⏎ Found during uplift investigation for ⏎ `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`: setting `VLLM_ROCM_USE_AITER_MOE=1` crashes engine init for any MoE model with a non-gated activation (`is_act_and_mul=False`). `AiterExperts` has no kernel for this case, and `select_unquantized_moe_backend()` raised a `ValueError` instead of falling back to another backend. ⏎  ⏎ This PR prevents the crash by falling back to an alternative backend …[truncated]

### L3-a9a7e45f31  (L3, 2026-09-17, sha a9a7e45f3174, PR #57055)
TITLE: [ROCm] Restore `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` and default w4a4 ASM GEMM back to off (#57055)
SOURCES: path_integration+keyword, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+6/-0); tests/kernels/quantization/test_rocm_mxfp4.py (+12/-4); vllm/_aiter_ops.py (+5/-1)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎  ⏎ #53141 removed the `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` environment variable and made ⏎ the aiter w4a4 ASM GEMM path unconditional on gfx950. At the time it was assumed that ⏎ it was only active for dense models like Llama, but change actually regresses MLA-style  ⏎ MoE models (DeepSeek, GLM, Kimi), whose shapes are largely absent from it. ⏎  ⏎ When a shape misses `a4w4_blockscale_tuned_gemm.csv`, aiter falls back to a default ⏎ config, and …[truncated]

### L3-1df336c3ba  (L3, 2026-09-17, sha 1df336c3bab8, PR #48249)
TITLE: [Perf][ROCm] Enable AITER QuickReduce + RMSNorm fusion (#48249)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/distributed/test_quick_all_reduce.py (+172/-0); vllm/_aiter_ops.py (+106/-1)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Route eligible ROCm AITER fused all-reduce + RMSNorm prefill calls through the new AITER QuickReduce + RMSNorm fused kernel added in ROCm/aiter#4104.  ⏎  ⏎ > For reviewers: this lazily creates an AITER QR communicator and caches it on the vLLM device communicator for reuse; it is intentionally not created per call.  ⏎  ⏎ **NB! The fusion only applies until `max_num_batched_tokens<=8192`. The `max_num_batched_tokens>8k` cases need subseq …[truncated]

### L3-80447d2765  (L3, 2026-09-17, sha 80447d276559, PR #57432)
TITLE: [Bugfix][DSv4.1] Fix FlashInfer DSpark non-causal attention (#57432)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/sparse_swa.py (+2/-0); tests/v1/attention/test_dspark_noncausal_sparse_mla.py (+150/-0); vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py (+62/-6)
LABELS: bug, ready, deepseek, nvidia, dflash, DSv4.1
BODY: DSpark's non-causal draft window has 133 valid keys at draft length 5, padded to 256 indices. The DSV4.1 FlashInfer integration passed the padded width as the active sparse length, so padding entered softmax normalization. A constant-value reference expecting 1 returned 0.51953125 (133/256). Short contexts also inherited causal visibility from the launcher's per-request query positions. ⏎  ⏎ Use the valid SWA length, retaining FlashInfer's required 1 …[truncated]

### L3-de24e51908  (L3, 2026-09-17, sha de24e5190820, PR #56922)
TITLE: [MM][V2] Enable encoder-only ViT CUDA graph capture (#56922)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/disaggregated_mooncake.yaml (+10/-0); tests/models/multimodal/generation/test_vit_cudagraph.py (+115/-0); tests/v1/cudagraph/test_encoder_cudagraph.py (+95/-0); tests/v1/ec_connector/integration/run_epd_mooncake_ec_full_pipeline.sh (+35/-1); tests/v1/ec_connector/integration/test_epd_correctness.py (+21/-0); tests/v1/worker/test_encoder_runner.py (+54/-1); vllm/v1/worker/mm_encoder_model_runner.py (+32/-3)
LABELS: documentation, ready, ci/build, multi-modality, nvidia
ISSUES: #57136 [Bug]: NaN vision embeddings with multi-budget encoder CUDA graphs sharing a pool
BODY: ## Purpose ⏎  ⏎ Enable vision encoder CUDA graph capture for ModelRunnerV2 encoder-only instances. #49852 added encoder graph support to the ordinary V2 GPUModelRunner, but MMEncoderModelRunner overrides `capture_model` with an immediate return, so E-only instances do not capture those graphs. ⏎  ⏎ Call the existing EncoderRunner capture path from the E-only runner, with GC handling, capture-memory accounting, and workspace locking. Reuse existing budget …[truncated]

### L3-9612f77077  (L3, 2026-09-17, sha 9612f77077e0, PR #56902)
TITLE: [Bugfix] Release stale FlashMLA workspace views after growth (#56902)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_sparse
FILES: vllm/v1/attention/backends/mla/flashmla_sparse.py (+26/-19)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ Release obsolete FlashMLA scratch storage when another layer grows the shared workspace. ⏎  ⏎ `FlashMLASparseImpl` keeps constructor-time tensor views in `q_concat_buffer`, `prefill_bf16_workspace`, and, with PCP+DCP, `gathered_kv_workspace`. When `WorkspaceManager` grows its allocation, those views keep the old storage alive. Both buffers remain allocated, reducing the memory available for KV cache. ⏎  ⏎ This change stores shapes and d …[truncated]

### L3-ac78445bc9  (L3, 2026-09-17, sha ac78445bc9ee, PR #57405)
TITLE: [MoE] Encapsulate TRT-LLM BF16 weight layout handling (#57405)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer.py (+21/-9); tests/kernels/moe/test_moe_weight_loading_padded.py (+0/-25); tests/kernels/moe/test_trtllm_bf16_moe.py (+49/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py (+13/-4); vllm/model_executor/layers/fused_moe/experts/trtllm_lora_moe.py (+5/-2); vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+26/-1); vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py (+24/-76)
LABELS: ready, nvidia
BODY: ## Purpose ⏎  ⏎ Follow up on #54699 by keeping TRT-LLM BF16 weight-layout handling inside the existing oracle and expert implementations. The generic unquantized method delegates dimension rounding and conversion; the oracle clears padding and packs weights in place while preserving 3D parameter shapes. Modular, monolithic, and LoRA experts share a BlockMajorK view helper at dispatch, including support for legacy 4D IPC weights. ⏎  ⏎ This also addresses  …[truncated]

### L3-ff6b5808c4  (L3, 2026-09-17, sha ff6b5808c457, PR #53792)
TITLE: [ROCm] Resolve the indexer fp8 cache dtype once at import (#53792)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+9/-11)
LABELS: rocm, ready
BODY: ## Summary ⏎  ⏎ `current_platform.fp8_dtype()` cannot change after import, but ⏎ `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` called it on every function ⏎ invocation — once per DSA layer per forward pass, including inside the fused ⏎ QK prologue custom op. ⏎  ⏎ [vllm#46172](https://github.com/vllm-project/vllm/pull/46172) introduced a module-level `FP8_DTYPE` constant but only wired it up ⏎ in two places and left five call sites behind. This PR routes every …[truncated]

### L3-db7a24c230  (L3, 2026-09-17, sha db7a24c230a4, PR #55960)
TITLE: [Perf] Add fused DFlash2 grouped convolution (#55960)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/spec_decode.yaml (+4/-0); tests/v1/e2e/spec_decode/acceptance_rates/dflash/test_dflash.py (+21/-0); tests/v1/spec_decode/test_dflash2.py (+50/-0); vllm/model_executor/models/qwen3_dflash2.py (+124/-0)
LABELS: speculative-decoding, ready, ci/build, qwen, nvidia, dflash
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ - Fuse DFlash2's grouped dynamic convolution in one Triton custom op, keeping the existing Torch CPU fallback. The kernel stays in `qwen3_dflash2.py`. ⏎ - Use row-aligned tiles: 1024 elements for at least 128 rows with hidden width divisible by 1024, otherwise 512. Four warps; no autotuner. ⏎ - Cover non-power-of-two dimensions, strided coefficient views, multiple tap counts, and both tile paths in FP32/BF16. ⏎ - Add real-checkpoint GSM8K a …[truncated]

### L3-78fdf4efb8  (L3, 2026-09-17, sha 78fdf4efb86d, PR #54699)
TITLE: [Bugfix][MoE] Convert FlashInfer BF16 weights in place (#54699)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer.py (+258/-0); tests/kernels/moe/test_moe_weight_loading_padded.py (+126/-0); tests/kernels/moe/test_trtllm_bf16_moe.py (+6/-2); tests/model_executor/model_loader/test_reload.py (+105/-0); vllm/model_executor/layers/fused_moe/routed_experts.py (+18/-0); vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py (+102/-20); vllm/model_executor/layers/quantization/utils/flashinfer_utils.py (+43/-43); vllm/model_executor/model_loader/reload/meta.py (+2/-2); vllm/model_executor/model_loader/reload/utils.py (+7/-1)
LABELS: bug, ready, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ Addresses #52511. ⏎  ⏎ The FlashInfer TRT-LLM BF16 MoE path had transient full-weight allocations at the 120B TP2 dimensions from #52511: ⏎  ⏎ 1. `align_moe_weights_for_fi` padded the already-loaded W13/W2 tensors, requesting another 1.38 GiB after the checkpoint was resident. ⏎ 2. The block-layout conversion allocated complete output tensors for W13 and W2, briefly retaining another 3.9375 GiB weight copy and raising the conversion peak to abo …[truncated]

### L3-e0050f287a  (L3, 2026-09-17, sha e0050f287aae, PR #57252)
TITLE: [Bugfix][ROCm] Add record_logical_topk_ready to ROCMAiterMLASparseImpl (GLM-5.3-Flash boot crash) (#57252)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+5/-0)
LABELS: bug, rocm, ready, verified, glm
BODY: ## Why this is not duplicating #56604 ⏎  ⏎ #56604 (open since 2026-09-12) fixes the same crash class by adding a **default no-op `record_logical_topk_ready` on the `MLAAttentionImpl` base class** in `vllm/v1/attention/backend.py`, so every backend silently inherits it. ⏎  ⏎ This PR instead adds the method **on `ROCMAiterMLASparseImpl` itself**, mirroring the existing `SparseMLACommonImpl` contract (`vllm/model_executor/layers/attention/sparse_mla_attenti …[truncated]

### L3-70df48dc3d  (L3, 2026-09-17, sha 70df48dc3d01, PR #57317)
TITLE: [Bugfix][KV Cache][GLM-5.3-Flash] Disable slot mapping kernel for the kpool tail buffer (#57317)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_kpool_tail_slot_mapping.py (+24/-0); tests/v1/worker/test_gpu_kpool_tail_slot_mapping.py (+98/-0); vllm/v1/kv_cache_interface.py (+4/-0)
LABELS: bug, ready, glm
ISSUES: #57227 [Bug][ROCm/gfx950] GLM-5.3-Flash 16K-chunk prefill: GPU memory-access fault when indexer triton kernels JIT during lazy CUDA-graph capture — clean under --enforce-eager
BODY: ## Purpose ⏎  ⏎ Fixes https://github.com/vllm-project/vllm/issues/57227. ⏎  ⏎ GLM-5.3-Flash crashes with MAF when sending 500k ISL prompts.  This is caused by out of bounds indexing in the default `_compute_slot_mappings_kernel` applied to the Kpool tail buffer. The kpool tail buffer has a fixed size for each request and anyway uses its own slot mapping kernel `compute_kpool_tail_slot_mapping`, and hence should not use the default kernel. ⏎  ⏎ **Altern …[truncated]

### L3-d12c276853  (L3, 2026-09-18, sha d12c2768530a, PR #57425)
TITLE: [Bugfix][ROCm] Alias SparseAttnIndexerKpool.forward_cuda to forward_native (GLM-5.3-Flash boot crash) (#57425)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/glm5next/amd/sparse_indexer.py (+24/-1)
LABELS: bug, rocm, ready, nvidia, verified, glm
ISSUES: #57424 [Bug][ROCm] GLM-5.3-Flash fails to boot on 0.3.1 nightly: SparseAttnIndexerKpool (AMD) missing forward_cuda → NotImplementedError
BODY: ## Purpose ⏎  ⏎ Fixes #57424. `SparseAttnIndexerKpool` on the AMD/ROCm path (`vllm/models/glm5next/amd/sparse_indexer.py`) defines only `forward_native`. `CustomOp.dispatch_forward` routes *enabled* ops to `forward_hip` on ROCm, and `forward_hip` falls back to `forward_cuda` — which doesn't exist → `NotImplementedError` at the first forward of the indexer op. ⏎  ⏎ On the `v0.29.1rc0.dev265+g0bfc7a15d` ROCm nightly, serving GLM-5.3-Flash crashes at `profi …[truncated]

### L3-32636580a6  (L3, 2026-09-18, sha 32636580a6f1, PR #51065)
TITLE: [Bugfix][MLA] TritonMLA: fix illegal memory access on causal multi-token decode (#51065)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/triton_mla.py (+37/-54); tests/kernels/attention/test_triton_mla_causal_verify_flatten.py (+215/-0)
LABELS: bug, rocm, ready, kv-cache-manager
ISSUES: #51848 [Bug]: TRITON_MLA reads past block_table on causal multi-token decode — speculative decoding is broken for MLA models on Ampere
BODY: ## Purpose ⏎ `TritonMLA` declared single-token decode support, so a speculative-decode verify block reached `forward_mqa` unflattened: `query_len * num_reqs` query rows against `num_reqs` rows of `block_table` / `seq_lens`. `decode_attention_fwd` launches one program per query row and indexes `tl.load(B_Seqlen + cur_batch)`, so every row past `num_decodes` is an out-of-bounds read. ⏎  ⏎ Reaching it takes a group whose reorder threshold is raised pas …[truncated]

### L3-48f663c37f  (L3, 2026-09-18, sha 48f663c37f8c, PR #56431)
TITLE: [XPU] Fix incorrect context-key normalization for Qwen DFlash-based models (#56431)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_dflash.py (+10/-0)
LABELS: intel-gpu, qwen, dflash
BODY: ## Purpose ⏎ DFlash stacks per-layer K-norm weights and invokes `rms_norm` with: ⏎  ⏎ ```text ⏎ input:  [num_layers, num_context_tokens, num_kv_heads, head_dim] ⏎ weight: [num_layers, head_dim] ⏎ ``` ⏎ The current XPU kernel used by vLLM treated the 2D weight as a 1D vector and always applied the first layer's weight row. Consequently, context K values for all remaining draft layers were normalized with incorrect weights before being written to the KV c …[truncated]

### L3-50d812b66a  (L3, 2026-09-18, sha 50d812b66a69, PR #57604)
TITLE: [Perf][DSV4.1] Optimize MegaMoE staging and NVFP4 cache gathers (#57604)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_compressor_kv_cache.py (+68/-27); tests/models/test_deepseek_v4_mega_moe.py (+13/-9); vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py (+43/-24); vllm/models/deepseek_v41/common/ops/cache_utils.py (+5/-1); vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py (+1/-1)
LABELS: performance, ready, deepseek, DSv4, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Reduce DeepSeek V4.1 MegaMoE/MegaAttention prefill overhead: tile multiple tokens ⏎ per MegaMoE staging program for sufficiently large inputs, and size the NVFP4 ⏎ gather grid by request count and output capacity. Keep conservative launch ⏎ choices for small inputs. Both transformations preserve the existing arithmetic ⏎ and add no persistent buffers. ⏎  ⏎ Dispatch uses eight-token staging tiles from 64 tokens, otherwise one token. ⏎ On GB200 at H5 …[truncated]

### L3-2bdae2a5a8  (L3, 2026-09-18, sha 2bdae2a5a83f, PR #57563)
TITLE: [fix] Mistral-Large-3 accuracy regression on `main` (#57563)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/transformers_utils/test_config.py (+52/-0); vllm/model_executor/models/AXK1.py (+6/-6); vllm/model_executor/models/deepseek_v2.py (+6/-6); vllm/models/deepseek_v32/attention.py (+3/-3); vllm/models/deepseek_v4/common/rope.py (+3/-3); vllm/models/deepseek_v41/common/rope.py (+3/-3); vllm/models/glm5next/common/attention.py (+3/-3); vllm/models/kimi_k3/nvidia/mla.py (+3/-3)
LABELS: ready, deepseek, mistral, DSv4, kimi, k3, DSv4.1
BODY: ## Purpose ⏎  ⏎ It appears some changes to the config parsing for Mistral Large 3 degraded accuracy. These changes try to remediate. ⏎  ⏎ ### Root cause ⏎  ⏎ Mistral-Large-3 ships a Mistral-format `params.json` with ⏎  ⏎ ```json ⏎ "yarn": {"factor": 36, "beta": 32, "alpha": 1, "original_max_position_embeddings": 8192, "apply_scale": false}, ⏎ "llama_4_scaling": {"beta": 0.1, "original_max_position_embeddings": 8192} ⏎ ``` ⏎  ⏎ i.e. YaRN frequency interpolatio …[truncated]

### L3-1dc2d854c1  (L3, 2026-09-18, sha 1dc2d854c120, PR #56227)
TITLE: [Feat][Model] Support encoder-side SWA-bounded replay for DeepSeek-V4.1-Flash (#56227)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/compressor_utils.py (+26/-5); vllm/v1/attention/backends/mla/indexer.py (+10/-1); vllm/v1/attention/backends/mla/sparse_swa.py (+50/-5); tests/models/test_deepseek_v41_replay_start.py (+112/-0); tests/v1/attention/test_deepseek_v4_swa_visible.py (+174/-10); tests/v1/attention/test_dspark_noncausal_sparse_mla.py (+1/-0); tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py (+23/-0); tests/v1/core/test_contiguous_kv_packing.py (+64/-0); tests/v1/core/test_prefix_caching.py (+3/-1); tests/v1/core/test_prefix_replay.py (+279/-0); (+20 more)
LABELS: new-model, rocm, speculative-decoding, ready, torch.compile, tool-calling, deepseek, kv-connector, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ DeepSeek-V4.1 keeps a 128-token sliding-window (SWA) KV cache per layer beside its prefix-cacheable compressed MLA and indexer caches. Storing that window for prefix caching and KV connectors costs more than it saves, so this implements DeepSeek's encoder-side "SWA bounded replay": the SWA caches leave prefix caching, and after a prefix hit ending at H the scheduler recomputes the hit's last window [H-128, H) to rebuild them. The repl …[truncated]

### L3-71fc70d3ae  (L3, 2026-09-18, sha 71fc70d3ae1d, PR #50455)
TITLE: [ROCm][DSv4] Fix sparse-indexer logits collapse on gfx950/gfx942 (#50455)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+187/-5); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+186/-0)
LABELS: rocm, ready, v1, deepseek, DSv4
BODY: On gfx950 and gfx942, AITER's `deepgemm_fp8_paged_mqa_logits` mis-scores the DSv4 sparse indexer during decode (wrong paged-cache layout), collapsing long-context retrieval. `VLLM_DSV4_LOGITS_FIX=1` (off by default) reroutes decode + MTP through a Triton kernel instead ⏎  ⏎ The replacement kernel (`_fp8_paged_mqa_logits_decode_kernel`) reads the fp8 KV cache in its true block-flat layout and computes `relu(q·k)·weight·scale` per key position, splitti …[truncated]

### L3-b346479065  (L3, 2026-09-18, sha b34647906550, PR #49942)
TITLE: [CPU] Add CPU FP8 W8A8 linear/MoE support (#49942)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+2/-2); vllm/v1/attention/backends/mla/amx_mla.py (+79/-8); cmake/cpu_extension.cmake (+5/-0); csrc/cpu/cpu_attn.cpp (+2/-11); csrc/cpu/cpu_isa.cpp (+20/-0); csrc/cpu/sgl-kernels/common.h (+83/-0); csrc/cpu/sgl-kernels/gemm.h (+53/-1); csrc/cpu/sgl-kernels/gemm_fp8_w8a8.cpp (+1093/-0); csrc/cpu/sgl-kernels/moe.cpp (+93/-7); csrc/cpu/sgl-kernels/moe_fp8_w8a8.cpp (+432/-0); (+10 more)
LABELS: ci/build, cpu, quantization
BODY: ## Purpose ⏎  ⏎ Support FP8 W8A8 Linear and MoE for Intel DMR CPUs. ⏎  ⏎ ## Test Plan ⏎  ⏎ Kernel tests:  ⏎ - `python -m pytest tests/kernels/moe/test_cpu_quant_fused_moe.py -v` ⏎ - `python -m pytest tests/kernels/quantization/test_cpu_fp8_scaled_mm.py -v` ⏎  ⏎  ⏎ Validated models: ⏎  ⏎  ⏎  ⏎ ## Test Result ⏎ `python -m pytest tests/kernels/moe/test_cpu_quant_fused_moe.py -v` : 185 passed ⏎ `python -m pytest tests/kernels/quantization/test_cpu_fp8_scaled_mm.py -v` …[truncated]

### L3-62af3df734  (L3, 2026-09-18, sha 62af3df7347c, PR #57575)
TITLE: [Bugfix][MLA] Reserve sparse prefill buffers before KV cache sizing (#57575)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_sparse
FILES: vllm/models/hy_v4/nvidia/flashmla_sparse.py (+11/-1); vllm/v1/attention/backends/mla/flashmla_sparse.py (+33/-11); tests/v1/attention/test_sparse_mla_backends.py (+15/-0)
LABELS: bug, ready, deepseek
ISSUES: #57449 [Bug]: MLA Workspace OOM At Runtime
BODY: FIX #57449  ⏎  ⏎ Addresses #57449 by reserving sparse-prefill query/output buffers before KV-cache sizing and reusing them through FlashMLA's `out=` argument. Preserves compact output layout and supports the HY V4 override. Native FP32 statistics still allocate. ⏎  ⏎ Validation on B300: ⏎ - `.venv/bin/python -m pytest tests/v1/attention/test_sparse_mla_backends.py -k 'test_sparse_backend_decode_correctness and FlashMLA and fp8_ds_mla' -v`: **42 passed …[truncated]

### L3-38537331d3  (L3, 2026-09-19, sha 38537331d3e8, PR #57667)
TITLE: [Bugfix][DSA] Avoid runtime JIT for offset candidate end buffers (#57667)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/model_executor/layers/test_mla_short_prefill_indexer.py (+6/-3); vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py (+1/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ Mixed prefill batches can pass an offset view of the candidate expansion's `end` buffer after warmup used an aligned pointer. Triton specializes on that alignment and JIT-compiles another kernel during serving. ⏎  ⏎ Disable alignment specialization for `end_ptr`: each program stores one scalar, so alignment does not enable vectorized stores. Extend the existing reference tests to cover both aligned and offset output buffers, including str …[truncated]

### L3-fbe8a157fb  (L3, 2026-09-19, sha fbe8a157fbdd, PR #53623)
TITLE: [ROCm][Perf] Enable the AITER GDN decode fast path for flat qkvz layouts (#53623)
SOURCES: subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+1/-0); .buildkite/test_areas/kernels.yaml (+1/-0); tests/kernels/mamba/test_gdn_rocm_layout_dispatch.py (+124/-0); vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+6/-8)
LABELS: rocm, ready, ci/build
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ `_forward_core_rocm` currently gates the AITER GDN decode fast path on `self.gqa_interleaved_layout`, so only Qwen3-Next takes it and flat-layout models (e.g. Qwen3.5, Qwen3.8) fall through to the generic path. #42880 added that guard because the fused reshape+conv kernel only understood Qwen3-Next's packing and read the wrong columns from a flat tensor. ⏎  ⏎ ROCm/aiter#3251 added a `qkvz_layout` parameter handling both. It selects th …[truncated]

### L3-4cc15f2121  (L3, 2026-09-19, sha 4cc15f2121b3, PR #57316)
TITLE:  [Quantization] Let ModelOpt MXFP8 layers load pre-processed weights  Purpose (#57316)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/quantization/test_modelopt.py (+30/-0); vllm/model_executor/kernels/linear/mxfp8/Mxfp8LinearKernel.py (+4/-0); vllm/model_executor/kernels/linear/mxfp8/emulation.py (+2/-0); vllm/model_executor/kernels/linear/mxfp8/flashinfer.py (+4/-0); vllm/model_executor/kernels/linear/mxfp8/rocm_native.py (+2/-0); vllm/model_executor/layers/quantization/modelopt.py (+15/-4)
LABELS: ready, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ `ModelOptLinearMethod.supports_pre_processed_weights` only returned True for NVFP4, so the weight cache daemon (`--load-format ipc_cache`) rejected every `MXFP8` checkpoint ⏎  ⏎ `Mxfp8LinearKernel` gains a supports_pre_processed_weights flag, set on the kernels whose post-load step only rewrites weight / weight_scale (FlashInfer CUTLASS and CuTe-DSL, ROCm native, emulation). ModelOptLinearMethod reports the kernel's flag for MXFP8. Th …[truncated]

### L3-a7fda4c88b  (L3, 2026-09-19, sha a7fda4c88bfc, PR #49435)
TITLE: [Bugfix] Fix SM100 fp8_ds_mla cache scales (#49435)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/libtorch_stable/cache_kernels.cu (+3/-3); csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu (+10/-9); tests/kernels/attention/test_cache.py (+5/-5); tests/kernels/attention/test_flashmla_sparse.py (+31/-8); tests/kernels/attention/test_kimi_k3_mla_key_concat_kv_cache.py (+16/-2); tests/kernels/test_fused_deepseek_v32_norm_rope.py (+5/-4); vllm/models/deepseek_v32/common/kernels.py (+3/-4); vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py (+3/-3)
LABELS: bug, ready, deepseek, kimi, k3
BODY: ## Summary ⏎  ⏎ Unify the `fp8_ds_mla` cache scale contract used by FlashMLA on SM90 and ⏎ SM100. The physical 656-byte cache entry is unchanged: 512 FP8 NoPE values, ⏎ four FP32 scale fields (one per 128 values), and a 128-byte BF16 RoPE tail. ⏎ The value stored in each FP32 scale field is now always: ⏎  ⏎ ```text ⏎ 2 ** ceil(log2(max(amax / 448, 1e-4))) ⏎ ``` ⏎  ⏎ Power-of-two values survive both reader conversions exactly: SM90 converts ⏎ the scale to BF16, while SM1 …[truncated]

### L3-708c2495c0  (L3, 2026-09-20, sha 708c2495c0c6, PR #57467)
TITLE: [Test][Core] Compute the expected hybrid prefix-cache hit instead of hard-coding it (#57467)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/intel_jobs/test-intel.yaml (+1/-1); tests/v1/core/prefix_cache/test_hybrid_prefix_cache_hit_rate.py (+52/-10)
LABELS: intel-gpu, ready, ci/build
BODY: ## Purpose ⏎ `test_prefix_cache_hit_rate` asserts hard-coded hit-rate floors of 0.90 / 0.75. As https://github.com/vllm-project/vllm/pull/57127 documents, those numbers came from a GB200 run where the hybrid cache resolves to 544-token blocks, giving 3264 cached tokens for base and 2720 for MTP; the floors were set just below the measured minimums. ⏎  ⏎ However, the block size is not portable. It is derived at startup from the mamba state size divid …[truncated]

### L3-4868312128  (L3, 2026-09-20, sha 486831212817, PR #57621)
TITLE: [Refactor] Remove dead kernel code (#57621)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.custom_paged
FILES: csrc/rocm/attention.cu (+0/-33); csrc/cpu/sgl-kernels/common.h (+0/-16); csrc/libtorch_stable/cuda_vec_utils.cuh (+0/-4); csrc/libtorch_stable/moe/moe_wna16_utils.h (+0/-8); csrc/libtorch_stable/quantization/gptq/matrix_view.cuh (+0/-33); csrc/libtorch_stable/quantization/marlin/marlin.cuh (+1/-22); csrc/libtorch_stable/sampler.cu (+0/-5); csrc/quantization/w8a8/fp8/amd/quant_utils.cuh (+1/-241); csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh (+0/-13); csrc/rocm/moe_q_gemm_rdna3.cu (+0/-15); (+2 more)
LABELS: rocm, ready, cpu, nvidia
BODY: ## Purpose ⏎  ⏎ Remove dead kernel code

### L3-1596fa5f88  (L3, 2026-09-20, sha 1596fa5f8825, PR #57434)
TITLE: [ROCm][DSv4.1][Perf] Reuse the decode topk ragged metadata across layers (#57434)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v41/amd/rocm.py (+66/-13)
LABELS: rocm, deepseek, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ `_forward_decode` rebuilds the ragged form of the topk indices on every ⏎ compressed layer. The pack is a pure function of the shared ⏎ `topk_indices_buffer`, which only `index_source_layer_ids` write, plus per-step ⏎ metadata (`token_to_req_indices`, the block table, `is_valid_token`). The one ⏎ per-layer term is `compress_ratio`, which divides the block size. ⏎  ⏎ So the layers between two index sources all rebuild the same thing. On ⏎ DeepSeek-V …[truncated]

### L3-92b40f5a14  (L3, 2026-09-20, sha 92b40f5a1447, PR #56045)
TITLE: [CPU][Perf] refactor paged attention for Arm CPUs (#56045)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+3/-4); csrc/cpu/cpu_attn_impl.hpp (+36/-13); csrc/cpu/cpu_attn_neon.hpp (+2/-2); csrc/cpu/cpu_attn_neon_bfmmla.hpp (+360/-487)
LABELS: cpu
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Previous implementation left performance on the table. ⏎  ⏎ This impl provides: ⏎ - faster GEMM kernels ⏎ - faster kv cache packing ⏎ - prepack Q during copy ⏎ - adds interface allowing attention impl to write softmax probabilities directly in a packed format  ⏎ - enable "NEON" ISA for block_size % 16 ⏎ - up to 25% faster paged attention ⏎ - removes 100 lines of code overall ⏎  ⏎ ## Test Plan ⏎  ⏎ CI ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-e5fce7b56b  (L3, 2026-09-20, sha e5fce7b56b07, PR #57421)
TITLE: [Core][Kernel] Share persistent workspaces for Marlin and Humming (#57421)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_moe.py (+17/-3); tests/kernels/moe/test_moe_permute_unpermute.py (+163/-12); tests/kernels/quantization/test_marlin_gemm.py (+58/-0); tests/kernels/quantization/test_marlin_tile_padding.py (+7/-6); tests/model_executor/model_loader/test_reload.py (+44/-23); tests/v1/worker/test_workspace.py (+63/-0); vllm/model_executor/kernels/linear/mixed_precision/marlin.py (+1/-7); vllm/model_executor/kernels/linear/mxfp4/marlin.py (+1/-1); vllm/model_executor/kernels/linear/mxfp8/marlin.py (+1/-1); vllm/model_executor/kernels/linear/nvfp4/marlin.py (+1/-1); (+11 more)
LABELS: performance, ready, nvidia, quantization
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Co-authored by: ErinYin <liang.yin@daocloud.io> ⏎  ⏎ Humming grouped MoE currently retains permutation scratch in every layer, so its persistent memory grows with the number of layers. Marlin lock buffers are also owned by individual layers or kernels, tying their lifetime to weight preparation and reload. ⏎  ⏎ This PR gives both resources a shared owner in `WorkspaceManager`: ⏎  ⏎ - Add `get_persistent_resource(key, factory)` for objects …[truncated]

### L3-9b49f92344  (L3, 2026-09-20, sha 9b49f9234431, PR #57603)
TITLE: [Perf][DSV4.1] Overlap mHC coefficients for small TP batches (#57603)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_mhc_kernels.py (+129/-12); vllm/model_executor/kernels/mhc/tilelang_kernels.py (+21/-17); vllm/model_executor/kernels/mhc/warmup.py (+1/-0); vllm/models/deepseek_v41/nvidia/model.py (+60/-16); vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py (+29/-1); vllm/models/deepseek_v41/nvidia/ops/mhc.py (+117/-0)
LABELS: ready, deepseek, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Overlap shifted mHC coefficient generation with attention/FFN on small batches. The decoder keeps one post/pre path: the existing mHC dispatcher ⏎ selects overlap or native execution, while the decoder joins the coefficient ⏎ stream before consuming its outputs. ⏎  ⏎ `supports_mhc_overlap` in `.ops.mhc` requires SM100/DeepGEMM, no ubatching, ⏎ and compatible projection dimensions: hidden width divisible by 64, and ⏎ `hc_mult * (hc_mult + 2)` a po …[truncated]

### L3-db1bfdd4fb  (L3, 2026-09-20, sha db1bfdd4fb0d, PR #57477)
TITLE: [Bugfix][GLM-5.3-Flash] Address kpool tail blocks by the padded indexer stride in the NVIDIA prefill seed kernel (#57477)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_kpool_decode_update_batched.py (+10/-3); vllm/models/glm5next/nvidia/ops/kpool_compress.py (+13/-2)
LABELS: bug, ready, nvidia, glm
BODY: ## Purpose ⏎  ⏎ Fix silent indexer-cache corruption for GLM-5.3-Flash on NVIDIA: the kpool prefill tail-seed kernel addresses tail blocks with a dense stride while the tail cache is stored with the indexer's padded block stride. ⏎  ⏎ **How the tail cache is laid out.** `_get_kv_cache_groups_glm5_next` pads the `KpoolTailSpec` page to the indexer page (`page_size_padded = idx_page`; 38016 B for the default `block_size=1152`: 288 pool rows x 132 B), and `g …[truncated]

### L3-1b9fa3eaa8  (L3, 2026-09-20, sha 1b9fa3eaa8ff, PR #57534)
TITLE: [Perf][GLM] Fuse the kpool tail slot mapping into one Triton kernel (#57534)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+57/-2); tests/v1/attention/test_kpool_tail_slot_mapping.py (+63/-0)
LABELS: ready, glm
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ GLM-5.3-Flash sparse-MLA models keep a one-block-per-request circular tail cache (kpool). Every decode step — twice per step with MTP, once for the target model and once for the drafter — `KpoolTailMetadataBuilder.build()` maps each token to its request's tail ring with a chain of small torch ops (`arange` + `searchsorted` + `clamp` + `index_select` + `remainder` + `copy_`): 12 kernel launches and ~250 us of CPU enqueue time per bui …[truncated]

### L3-eb87980585  (L3, 2026-09-20, sha eb87980585c1, PR #57554)
TITLE: [Build] Fix DeepGEMM CUDA 12.9 release builds (#57554)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: cmake/external_projects/deepgemm.cmake (+4/-3); tools/install_deepgemm.sh (+3/-3)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ Restore the six CUDA 12.9 release targets that fail in [release-v2 #6858](https://buildkite.com/vllm/release-v2/builds/6858): x86_64 and aarch64 wheels, plus both architectures' default and Ubuntu 24.04 images. ⏎  ⏎ All six fail while compiling DeepGEMM's host extension because `layout/mega_mhc.cuh` uses `__nv_fp8_e4m3` without including `<cuda_fp8.h>`. CUDA 13's transitive includes mask the missing declaration; CUDA 12.9 does not. The su …[truncated]

### L3-9a70c233cd  (L3, 2026-09-20, sha 9a70c233cd5d, PR #57643)
TITLE: [Perf][DSV4.1] Fuse TP all-reduce with mHC input preparation (#57643)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+2/-0); csrc/libtorch_stable/all_reduce_mhc.cu (+212/-0); tests/distributed/test_custom_all_reduce.py (+111/-0); tests/distributed/test_engram_dp_shard.py (+1/-0); tests/kernels/test_mhc_kernels.py (+45/-1); vllm/models/deepseek_v4/nvidia/model.py (+3/-0); vllm/models/deepseek_v41/nvidia/model.py (+19/-2); vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py (+0/-98); vllm/models/deepseek_v41/nvidia/ops/mhc.py (+156/-4)
LABELS: ready, ci/build, deepseek, nvidia, DSv4, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Fuse TP all-reduce, mHC post mixing, delayed collapse, and RMSNorm at DSV4.1 decoder boundaries. This replaces three launches with one, preserves the BF16 boundaries, and retains coefficient overlap. The kernel reuses the existing MNNVL workspace without an additional persistent communication allocation. ⏎  ⏎ Enabled by default for eligible SM100 TP4 forwards with 1–16 tokens, including speculative decoding. The selector requires hidden s …[truncated]

### L3-27757dde02  (L3, 2026-09-20, sha 27757dde020e, PR #56625)
TITLE: [DSV4.1] Add encoder cuda graph support for deepseek-v4.1-flash (#56625)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_vit_cudagraph.py (+23/-0); tests/models/multimodal/processing/test_tensor_schema.py (+6/-0); vllm/models/deepseek_v4/common/vision.py (+156/-7); vllm/models/deepseek_v41/amd/vl_model.py (+12/-28); vllm/models/deepseek_v41/common/vl_cudagraph.py (+390/-0); vllm/models/deepseek_v41/nvidia/vl_model.py (+12/-28)
LABELS: rocm, ready, multi-modality, deepseek, nvidia, DSv4, DSv4.1
BODY: ## Purpose ⏎  ⏎ Enable CUDA graph capture/replay for the DeepSeek-V4.1 vision encoder (ViT + ⏎ aligner) via the `SupportsEncoderCudaGraph` protocol, removing per-image eager ⏎ encoder overhead (Python loop, per-image kernel launches, RoPE/cu_seqlens ⏎ host computation) from the request critical path. ⏎  ⏎ The captured graph packs a batch of images into one varlen run: ViT blocks ⏎ attend per image via `cu_seqlens`, and the aligner's spatial merge becomes …[truncated]

### L3-97dc6b19d2  (L3, 2026-09-21, sha 97dc6b19d2fe, PR #57885)
TITLE: [Perf][Attention] Avoid redundant sparse attention metadata operations (#57885)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+3/-2); vllm/v1/attention/backends/mla/sparse_swa.py (+1/-1)
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Sparse attention metadata preparation launches unnecessary GPU work on every decode step. Write SWA token validity directly into its persistent buffer, and compute compressed prefill sequence lengths only when the batch contains prefills. ⏎  ⏎ For the tested DSV4.1 TP4 configuration, this eliminates four validity copies and one unused division per decode step. No new kernels, flags, or dispatch thresholds are introduced. ⏎  ⏎ ## Validation ⏎  ⏎ R …[truncated]

### L3-67513c8b67  (L3, 2026-09-21, sha 67513c8b67e2, PR #57874)
TITLE: [Bugfix][DSV4.1] Restrict mHC overlap to full CUDA graphs (#57874)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_mhc_kernels.py (+21/-2); vllm/models/deepseek_v41/nvidia/model.py (+2/-1); vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py (+16/-0); vllm/models/deepseek_v41/nvidia/ops/mhc.py (+6/-8)
LABELS: bug, ready, deepseek, nvidia, mrv2, DSv4.1
DEEP_STUDY: deep-study correctness case vllm:67513c8b67: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=unknown
BODY: ## Problem and fix ⏎  ⏎ DSV4.1's small-batch eager warmup selected overlap, while breakable PIECEWISE capture selected Mega mHC. That could leave Mega mHC's stream-local barriers uninitialized until capture, causing startup to fail. ⏎  ⏎ Use overlap only during FULL CUDA graph capture, still bounded by the existing 16-token cutoff and hardware requirements. Eager execution/warmup and PIECEWISE execution retain native mHC dispatch (Mega mHC where supporte …[truncated]

### L3-e34685dfc0  (L3, 2026-09-21, sha e34685dfc0c9, PR #52362)
TITLE: [ROCm][DSv4] Enable DSpark adaptive verification (#52362)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+28/-1); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+259/-1); tests/models/test_deepseek_v4_dspark_rocm.py (+130/-0); tests/v1/attention/test_deepseek_v4_rocm_adaptive.py (+249/-0); vllm/models/deepseek_v4/amd/dspark.py (+28/-5); vllm/models/deepseek_v4/amd/rocm.py (+31/-0)
LABELS: rocm, ready, deepseek, DSv4, dflash
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Enable DSpark confidence-scheduled adaptive verification for DeepSeek-V4 on ROCm. Developed with contributions from [@larryli2-amd](https://github.com/larryli2-amd), who added explicit missing-confidence-head error handling and provided the MI350X performance validation. ⏎  ⏎ This change: ⏎  ⏎ - adds AMD confidence-head construction, inference, checkpoint remapping, and safe fallback when a checkpoint has no confidence head ⏎ - enables v …[truncated]

### L3-04c1f4a407  (L3, 2026-09-21, sha 04c1f4a40796, PR #57906)
TITLE: [Bugfix][ROCm][DSv4.1] Disable SWA bounded replay on ROCm (#57906)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v41/attention.py (+9/-0)
LABELS: bug, rocm, ready, deepseek, DSv4.1
BODY: ## Purpose ⏎  ⏎ Disable SWA bounded replay on ROCm, where it currently faults the GPU. ⏎  ⏎ SWA bounded replay (#56227, on by default) keeps the sliding-window KV out of prefix caching and rebuilds it after a prefix hit by padding the replayed tokens' slots in the prefix-cacheable groups. Correctness depends on the consuming prefill kernel clamping the window; that clamp landed in the FlashInfer and FlashMLA paths. ⏎  ⏎ The ROCm sparse SWA metadata builders  …[truncated]

### L3-8c0825090d  (L3, 2026-09-21, sha 8c0825090df8, PR #57389)
TITLE: [Bugfix][NIXL] Fix DCP pulls across MLA cache regions (#57389)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector_hma.py (+77/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+36/-8); vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py (+20/-20)
LABELS: bug, kv-connector
BODY: When prefill and decode use different cache-group layouts, as with a HiSparse decoder, NIXL matches their KV blocks by memory region. That receive path currently assumes a single source rank and fails when a DCP prefiller supplies multiple ranks. ⏎  ⏎ Create a read for each source rank using the existing region mapping. For each region, reuse `_apply_dcp_prefix_caching` to select the pages owned by that rank and skip the decoder's cached prefix. Boun …[truncated]

### L3-382970ee6c  (L3, 2026-09-21, sha 382970ee6ca4, PR #50592)
TITLE: [Kimi-K3][AMD] Return KDA and MLA projection outputs directly (#50592)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/models/kimi_k3/test_amd_kda_direct_return.py (+72/-0); tests/models/kimi_k3/test_amd_mla_direct_return.py (+38/-0); tests/models/kimi_k3/test_nvidia_kda_direct_return.py (+31/-0); vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py (+2/-3); vllm/models/kimi_k3/amd/kda.py (+2/-3); vllm/models/kimi_k3/amd/linear.py (+25/-16); vllm/models/kimi_k3/nvidia/model.py (+0/-11)
LABELS: rocm, ready, kimi, k3
BODY: ## Purpose ⏎  ⏎ Kimi-K3's AMD attention paths copied each output projection into a caller-owned buffer even though the projection already returns a tensor with the required shape, dtype, layout, and lifetime. ⏎  ⏎ This PR removes that redundant post-projection allocation and copy from both attention families: ⏎  ⏎ - 69 KDA layers ⏎ - 24 MLA layers ⏎ - 93 post-attention-projection copies per model step in total ⏎  ⏎ The functionality from #50847 is now folded into th …[truncated]

### L3-b8cf275382  (L3, 2026-09-21, sha b8cf2753825d, PR #54535)
TITLE: [AMD][Minimax-M3][perf] Enable packed LBHNC AITER QK-norm fusion for MiniMax-M3 on ROCm (#54535)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_minimax_m3.py (+361/-10); vllm/_aiter_ops.py (+112/-0); vllm/models/minimax_m3/amd/model.py (+163/-81)
LABELS: rocm, ready, minimax
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Keep MiniMax-M3's ROCm sparse path on AITER's fused QK-norm (`fused_qknorm_idxrqknorm`). ⏎  ⏎ Depends on: ⏎ - [ROCm/aiter#5143](https://github.com/ROCm/aiter/pull/5143) (fused QK-norm, including unit-scale e4m3 `index_q`; stacked on [aiter#4787](https://github.com/ROCm/aiter/pull/4787)) ⏎  ⏎ What this PR does: ⏎ - Call AITER fused QK-norm for sparse-PA insert (full and skip-index), with fallback to vLLM's fused kernel. ⏎ - Support packed LBH …[truncated]

### L3-4f14516790  (L3, 2026-09-21, sha 4f1451679088, PR #57931)
TITLE: [ROCm][CI] Use ROCm backend for DeepSeek V4.1 ViT test (#57931)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_vit_cudagraph.py (+5/-1)
LABELS: rocm, ready, multi-modality, deepseek, nvidia, DSv4.1
BODY: ## Summary ⏎  ⏎ PR [#56625](https://github.com/vllm-project/vllm/pull/56625) added the DeepSeek V4.1 ViT cudagraph test with the NVIDIA-only `FLASHMLA_SPARSE_DSV41` backend hardcoded, causing the ROCm job in [Buildkite #90154](https://buildkite.com/vllm/ci/builds/90154/list?sid=01a0c28f-3720-4eb4-a2d2-9c90a8aa45f4&tab=output) to fail during engine initialization. ⏎  ⏎ This change selects `ROCM_FLASHMLA_SPARSE_DSV4` on ROCm while preserving `FLASHMLA_ …[truncated]

### L3-a369a7becc  (L3, 2026-09-21, sha a369a7becc20, PR #57458)
TITLE: [Perf][Attention] Reduce GLM sparse MLA preparation overhead (#57458)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashinfer_sparse
FILES: vllm/model_executor/layers/attention/mla_attention.py (+4/-1); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+19/-15); vllm/v1/attention/backends/mla/sparse_utils.py (+72/-18); tests/kernels/attention/test_flashinfer_mla_decode.py (+103/-49); tests/kernels/test_concat_mla_q.py (+29/-0); tests/v1/attention/test_indexer_dcp_localize.py (+3/-4); tests/v1/attention/test_sparse_mla_backends.py (+18/-7)
LABELS: ready, nvidia, glm
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Reduce query packing and sparse-index preparation overhead around FlashInfer NoPE MLA attention. ⏎  ⏎ The three commits: ⏎  ⏎ 1. Reuse a contiguous NoPE query when its rotary tail is empty, avoiding a redundant `torch.cat` before FP8 quantization. Preserve concatenation for nonempty rotary dimensions and strided queries. ⏎ 2. Use one padded remap program per query for counted sparse capacities up to 4096. Initialize padding and reduce va …[truncated]

### L3-54020c3c3e  (L3, 2026-09-21, sha 54020c3c3ec9, PR #57937)
TITLE: [Engram] Drop redundant VLLM_PLE_CPU_OFFLOAD env var (#57937)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/test_config.py (+2/-10); vllm/config/engram.py (+3/-10); vllm/envs.py (+0/-3)
BODY: TL;DR the functionality of this env var should now be covered by the engram config args (similar process to attention backend and other configs flags) @ZJY0516 ⏎  ⏎ ## Motivation ⏎  ⏎ `VLLM_PLE_CPU_OFFLOAD` predates `EngramConfig`, but today it no longer enables any behavior of its own. Its only consumer in the tree was `_default_cpu_offload()` in `vllm/config/engram.py`, the `default_factory` for `EngramConfig.cpu_offload`. No runtime code read it d …[truncated]

### L3-0961bbae28  (L3, 2026-09-21, sha 0961bbae2894, PR #58065)
TITLE: [Spec Decode] Enable async scheduling for DFlash (#58065)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/test_config.py (+65/-0); vllm/config/vllm.py (+4/-2)
LABELS: dflash
BODY: ## Scope ⏎  ⏎ Adds `dflash` to the async-scheduling allowlist in `VllmConfig`, on both the explicit `--async-scheduling` path (previously a hard `ValueError`) and the default auto-enable path (previously force-disabled with a warning). The V2 model runner's async machinery (async output copies, GPU-side draft-token buffers) is method-agnostic, and the DFlash speculator consumes the same GPU tensors (`num_sampled`, `num_rejected`, `last_sampled`) as …[truncated]

### L3-f92b78f6ef  (L3, 2026-09-22, sha f92b78f6ef9c, PR #57428)
TITLE: [Kernel][DSV4.1] Fuse MXFP8 wo_b GEMM with sequence-parallel reduce-scatter (#57428)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/models/deepseek_v41/nvidia/flashmla.py (+1/-1); .buildkite/test_areas/distributed.yaml (+6/-5); benchmarks/kernels/benchmark_gemm_rs_ar.py (+407/-0); benchmarks/kernels/benchmark_kimi_k3_gemm_rs_ar.py (+0/-469); tests/kernels/test_gemm_rs_ar.py (+87/-26); vllm/cute_utils/_tcgen05.py (+71/-0); vllm/envs.py (+6/-4); vllm/model_executor/kernels/linear/cute_dsl/gemm_rs_ar.py (+340/-57); vllm/model_executor/warmup/kernel_warmup.py (+8/-7); vllm/models/deepseek_v4/nvidia/ops/o_proj.py (+6/-2); (+8 more)
LABELS: performance, ready, ci/build, deepseek, nvidia, DSv4, kimi, k3, DSv4.1
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Under sequence parallel, DeepSeek-V4.1's `wo_b` runs as an MXFP8 row-parallel GEMM followed by a TP reduce-scatter: two kernels and one HBM round trip per layer. This PR fuses them with the SM100 GEMM-RS kernel. It builds on @gau-nernst's MXFP8 support for the Kimi-K3 GEMM-RS/AR kernel (the first five commits), moves that kernel to `vllm/model_executor/kernels/linear/cute_dsl/gemm_rs_ar.py` so both models share one workspace, init gat …[truncated]

### L3-64a48b19b4  (L3, 2026-09-22, sha 64a48b19b410, PR #58097)
TITLE: [CI] Emit a kernel symbol map from the csrc build (opt-in, for test selection) (#58097)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/image_build/image_build.sh (+59/-8); docker/Dockerfile (+18/-0); docs/assets/contributing/dockerfile-stages-dependency.png (+0/-0); tools/ci/kernel_symbol_map.py (+370/-0)
LABELS: documentation, ci/build
BODY: ## Purpose ⏎  ⏎ Part of the CI test-selection work in vllm-project/ci-infra (#555, #610, #611). The nightly CI runs now record which GPU kernels every step launches. To turn that into a decision ("this PR changed `csrc/foo.cu`, which steps launched a kernel from it?") CI also needs to know **which source file produced each kernel symbol**. That has to come from the real build, because the mangled names depend on template instantiations, the arch list …[truncated]

### L3-91d7324cb1  (L3, 2026-09-22, sha 91d7324cb19d, PR #55385)
TITLE: [perf] wire FA and FlashMLA for sm90 GLM5Next NoPE SparseMLA (#55385)
SOURCES: path_core, subject_keyword, dependency_pin, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_build, L3.mla.flashmla_sparse, L3.mla.flashattn_sparse
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/v1/attention/backends/mla/flashattn_mla_sparse.py (+9/-1); vllm/v1/attention/backends/mla/flashmla_sparse.py (+39/-3); tests/v1/attention/test_flashmla_nope_sm90_backend_selection.py (+176/-0)
LABELS: ready, ci/build, glm
DEEP_STUDY: deep-study performance PR ()
BODY: Wire up two additional sparse-MLA attention backends for GLM5Next NoPE (head_size=512, e.g. GLM-5.3-Flash) on SM90: `FLASHMLA_SPARSE` and `FLASH_ATTN_MLA_SPARSE`. FlashInfer SM90 remains the default; the new backends are opt-in via `--attention-backend`. ⏎  ⏎ ## Changes ⏎  ⏎ - `cmake/external_projects/vllm_flash_attn.cmake`: pin vllm-flash-attn to `9cd61de` (vllm-project/flash-attention#172, merged: FA3 OnlyQv forward for head_size==0). The FlashMLA pin  …[truncated]

### L3-e4340e41c9  (L3, 2026-09-22, sha e4340e41c9a3, PR #55881)
TITLE: [Feat][XPU] VLLM_BATCH_INVARIANT support for Dense/MoE models (#55881)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: .buildkite/intel_jobs/misc_intel.yaml (+29/-1); .buildkite/scripts/hardware_ci/run-xpu-batch-invariance.sh (+243/-0); tests/v1/determinism/test_xpu_batch_invariant_ut.py (+228/-0); vllm/distributed/device_communicators/xpu_communicator.py (+35/-3); vllm/model_executor/determinism/batch_invariant.py (+4/-2); vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+7/-1); vllm/model_executor/layers/layernorm.py (+17/-0); vllm/platforms/xpu.py (+40/-0); vllm/sampling_params.py (+10/-0); vllm/v1/sample/ops/topk_topp_sampler.py (+6/-1)
LABELS: intel-gpu, ci/build
BODY: This PR fixed two issues ⏎ 1. auto select triton MoE backend when VLLM_BATCH_INVARIANT=1. This is to align with cuda behavior. (Attn backend auto selection is already in place.) ⏎ 2. Use deterministic, fixed-rank-order reduction when  VLLM_BATCH_INVARIANT=1 , instead of XCCL all-reduce, whose floating-point reduction order can vary with message size and break batch invariance. ⏎  ⏎ test results: ⏎  ⏎ Before this PR: ⏎ 1# issue ⏎ <img width="2107" height= …[truncated]

### L3-066ee91197  (L3, 2026-09-22, sha 066ee9119751, PR #56579)
TITLE: [Bugfix][Attention] Avoid NaN in the Triton softcap for large attention logits (#56579)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/triton_attention_helpers.py (+9/-5); tests/kernels/attention/test_triton_unified_attention.py (+38/-0)
LABELS: bug, ready, verified
ISSUES: #56578 [Bug]: Triton attention softcap returns NaN for large attention logits
BODY: ## Purpose ⏎  ⏎ Fixes #56578  ⏎  ⏎ `apply_softcap` computed `x * tanh(S / x)` as `x * (exp(S/x) - exp(-S/x)) / (exp(S/x) + exp(-S/x))`. Both exponentials overflow to `inf` when `|S / x| > ~88`, so the ratio becomes `inf / inf = NaN` and the whole attention row is poisoned. ⏎  ⏎ This is reachable for Gemma-2 style models that set `attn_logit_softcapping = 50`: a pre-softcap score above 4400 overflows. These models have very large attention logits, which …[truncated]

### L3-c4d424c2e2  (L3, 2026-09-22, sha c4d424c2e27b, PR #56448)
TITLE: [Bugfix][Spec Decode] Cap DFlash/DSpark profiling query batch (#56448)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/spec_decode/test_dflash_profile.py (+98/-0); vllm/v1/worker/gpu/model_runner.py (+5/-0); vllm/v1/worker/gpu/spec_decode/speculator.py (+2/-0)
LABELS: bug, speculative-decoding, ready, nvidia, mrv2, verified, dflash
ISSUES: #56443 [Bug]: DeepSeek-V4.1-Flash + DSpark spec decode hits CUDA device-side assert in `map_draft_to_target` at draft warmup on SM90 (H200) with Marlin MXFP4 MoE backend
BODY: Fixes #56443. ⏎  ⏎ ## Root Cause ⏎ The memory-profiling dummy run can hand-build max_num_seqs requests without applying the scheduler's speculative query-token budget. For DSpark/DFlash, each request can require num_query_per_req draft query rows. ⏎ For example, with max_num_seqs=512, max_num_batched_tokens=2048, and DSpark K=5, the profiling path asks the speculator for 512 * 5 = 2560 query rows while its input buffers are sized for 2048 rows, so DSpark …[truncated]

### L3-74370a0e30  (L3, 2026-09-22, sha 74370a0e3044, PR #43462)
TITLE: [Bugfix] hadacore_transform: respect inplace parameter to fix garbage outputs with QuIP transforms (#43462)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/quantization/hadamard/hadacore/hadamard_transform_cuda.cu (+1/-1); vllm/model_executor/layers/quantization/compressed_tensors/transform/module.py (+2/-2)
LABELS: bug, ready, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ Fix garbage outputs from QuIP transforms on Blackwell GPUs under `torch.compile` + CUDA graph capture. Initially observed on RTX 5090 (SM120); since reproduced on B200 (SM100) with the v0.21.0 prebuilt wheel + torch 2.11.0. Addresses the runtime side of vllm-project/llm-compressor#2006. ⏎  ⏎ ## Background ⏎  ⏎ The `hadacore_transform` op is declared with a mutable schema: ⏎  ⏎ ```cpp ⏎ ops.def("hadacore_transform(Tensor! x, bool inplace) - …[truncated]

### L3-2af05512cf  (L3, 2026-09-22, sha 2af05512cf94, PR #50814)
TITLE: [Kernel] Add opt-in load-time MXFP4 dequantization (#50814)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/kernels/quantization/test_mxfp4_kernel_selection.py (+86/-1); vllm/envs.py (+8/-0); vllm/model_executor/kernels/linear/mxfp4/emulation.py (+18/-2)
LABELS: rocm, quantization, verified, kimi, k3
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎  ⏎ - model: https://huggingface.co/amd/Kimi-K3-Quark-MXFP4-AttnFP8 ⏎ - add an explicit opt-in that dequantizes MXFP4 emulation weights to BF16 once during `process_weights_after_loading`; ⏎ - keep activation QDQ unchanged while avoiding repeated weight dequantization in each linear invocation; ⏎ - implement the optimization in the generic MXFP4 emulation backend so it applies to Quark, compressed-tensors, ModelOpt, online quantization, and oth …[truncated]

### L3-bc162b3f92  (L3, 2026-09-22, sha bc162b3f9280, PR #57834)
TITLE: [MRV2] Release weight offloader on shutdown (#57834)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/offloader/base.py (+5/-5); vllm/v1/worker/gpu/model_runner.py (+1/-0)
LABELS: ready, mrv2
BODY: ## Purpose ⏎  ⏎ MRV2 leaves its process-global `PrefetchOffloader` alive after in-process shutdown, retaining model layers, pinned CPU weights, and GPU staging buffers. Reset it after device synchronization so the existing teardown can reclaim those resources before another engine starts. ⏎  ⏎ ## Reproducer ⏎  ⏎ Save as `/tmp/repro_mrv2_offloader_shutdown.py`: ⏎  ⏎ ```python ⏎ import gc ⏎ import weakref ⏎ from types import SimpleNamespace ⏎ from unittest.moc …[truncated]

### L3-fae5cbfc8b  (L3, 2026-09-22, sha fae5cbfc8b7e, PR #58002)
TITLE: [Refactor] Remove dead code multiple places (#58002)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+0/-42); vllm/entrypoints/generate/base/protocol.py (+0/-50); vllm/logits_process.py (+1/-106); vllm/model_executor/layers/fused_moe/utils.py (+0/-29); vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+0/-72); vllm/model_executor/model_loader/weight_utils.py (+1/-75); vllm/model_executor/models/intern_vit.py (+0/-28); vllm/model_executor/models/mllama4.py (+0/-20); vllm/model_executor/models/teleflm.py (+0/-33); vllm/transformers_utils/processors/funasr.py (+0/-50); (+2 more)
LABELS: rocm, frontend, ready, llama, mrv2
BODY: ## Purpose ⏎  ⏎ Remove dead code multiple places

### L3-ca831d1c55  (L3, 2026-09-22, sha ca831d1c5571, PR #57435)
TITLE: [ROCm][DSv4.1][Perf] Fuse the inverse RoPE into the sparse decode reduce (#57435)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+113/-10); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+49/-0); vllm/models/deepseek_v41/amd/rocm.py (+22/-3)
LABELS: rocm, ready, deepseek, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ The ROCm sparse decode writes the attention output twice. The reduce kernel ⏎ combines the split-K partials and stores `[T, H, 512]`, and then ⏎ `_inverse_rope_gptj_kernel` reads that whole tensor back, rotates the trailing ⏎ 64 rope lanes and writes it out again — a full round trip per layer, for an ⏎ epilogue on data the reduce already held in registers. ⏎  ⏎ From a 10-step decode trace on MI355X (TP4, rank 0): ⏎  ⏎ | kernel | total | cal …[truncated]

### L3-9646f53064  (L3, 2026-09-22, sha 9646f5306475, PR #58153)
TITLE: [Bugfix][ROCm] Use the platform FP8 range in the concat MLA q test (#58153)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_concat_mla_q.py (+9/-3)
LABELS: bug, rocm
BODY: ## Purpose ⏎  ⏎ Three tests in `tests/kernels/test_concat_mla_q.py` fail on MI300X (gfx942): ⏎  ⏎ ``` ⏎ FAILED test_concat_mla_q_fp8_nope_and_rope[False-0-False] ⏎ FAILED test_concat_mla_q_fp8_nope_and_rope[False-0-True] ⏎ FAILED test_concat_mla_q_fp8_nope_and_rope[False-64-False] ⏎  ⏎ AssertionError: Tensor-likes are not equal! ⏎ Mismatched elements: 40769 / 69632 (58.5%) ⏎ Greatest absolute difference: 224.0 ⏎ Greatest relative difference: 0.5 ⏎ ``` ⏎  ⏎ Fail …[truncated]

### L3-1c0eee919d  (L3, 2026-09-22, sha 1c0eee919db3, PR #51915)
TITLE: [ROCm][Model][Bugfix] Enable GLM-5.2-MXFP4 on the deepseek_v32 path and fix sparse attention correctness (#51915)
SOURCES: path_core, path_integration+keyword, subject_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+1/-3); vllm/model_executor/models/deepseek_v2.py (+5/-0); tests/kernels/test_fused_deepseek_v32_norm_rope.py (+187/-151); vllm/_aiter_ops.py (+1/-1); vllm/models/deepseek_v32/__init__.py (+8/-4); vllm/models/deepseek_v32/amd/rocm.py (+16/-25); vllm/models/deepseek_v32/attention.py (+4/-1); vllm/models/deepseek_v32/common/kernels.py (+97/-27)
LABELS: bug, rocm, ready, deepseek, verified, glm
BODY: ## Purpose ⏎  ⏎ Enables GLM-5.2 (`GlmMoeDsaForCausalLM`) end-to-end on `vllm/models/deepseek_v32/amd/` ⏎ for gfx942/gfx950. Routing is opt-in via `--model-class-overrides`; the registry entry ⏎ is unchanged, so the default path for GLM-5.2 and DeepSeek-V3.2 is untouched.  ⏎  ⏎ The following issues were also fixed as a result of this overall enablement since it surfaced dormant bugs from a prior deepseek_v32 porting work: ⏎  ⏎ - **`launch_pdl` forwarded t …[truncated]

### L3-d110c2f19c  (L3, 2026-09-22, sha d110c2f19c4f, PR #52052)
TITLE: [ROCm] Use silu_and_mul_with_clamp's torch._C op (#52052)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/core/test_activation.py (+48/-7); vllm/model_executor/layers/activation.py (+6/-6)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: This is limited to alpha=1.0 and beta=0.0 as a safety consideration due to MiniMax previously avoiding this kernel in other cases. ⏎  ⏎ During DeepSeekV4, day 0 support, forward_cuda was disabled in favor of forward_native for silu_and_mul_with_clamp. This can be reverted for similar accuracy and around a 6% speedup in cases without speculative decode. ⏎  ⏎ Example Command: ⏎ VLLM_ROCM_USE_AITER=1 vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \ ⏎     - …[truncated]

### L3-73c59d365b  (L3, 2026-09-22, sha 73c59d365b9e, PR #55721)
TITLE: [XPU] Wire up SYCL apply_rotary_emb kernel in ApplyRotaryEmb (#55721)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/core/test_apply_rotary_emb.py (+41/-0); vllm/_xpu_ops.py (+25/-0); vllm/model_executor/layers/rotary_embedding/common.py (+12/-0)
LABELS: intel-gpu
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎   `ApplyRotaryEmb.forward_cuda` imports `vllm.vllm_flash_attn.layers.rotary`, a CUDA-only ⏎   module, so there is no `forward_xpu` override and every XPU call falls back to the CustomOp ⏎   default `forward_native` (~8 elementwise kernels). `vllm_xpu_kernels` — already a pinned ⏎   dependency (`requirements/xpu.txt`) — ships a SYCL `apply_rotary_emb` kernel that vLLM never ⏎   calls. This adds an 11-line `forward_xpu` wiring it up, foll …[truncated]

### L3-355d789f0d  (L3, 2026-09-22, sha 355d789f0d5b, PR #58173)
TITLE: [Compilation] Fix QuTLASS compilation with PyTorch 2.13 (#58173)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: cmake/external_projects/qutlass.cmake (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ `uv pip install --editable . --torch-backend=auto` ⏎  ⏎ Current builing from source will fail with error ⏎  ⏎ ```bash ⏎       /home/yewentao256/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/include/python3.12 ⏎       -isystem ⏎       /home/yewentao256/.cache/uv/builds-v0/.tmpcPlAW7/lib/python3.12/site-packages/torch/include ⏎       -isystem ⏎       /home/yewentao256/.cache/uv/builds-v0/.tmpcPlAW7/lib/python3.12/site-packages/torch/inc …[truncated]

### L3-88aa0d287d  (L3, 2026-09-22, sha 88aa0d287dd3, PR #52988)
TITLE: [Spec decode] Support variable-length decode for Kimi-K3 adaptive ver (#52988)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.rocm_aiter
FILES: vllm/model_executor/layers/attention/mla_attention.py (+4/-0); vllm/v1/attention/backends/mla/flashattn_mla.py (+3/-0); vllm/v1/attention/backends/mla/flashinfer_mla.py (+116/-14); vllm/v1/attention/backends/mla/flashmla.py (+1/-0); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+1/-0); tests/v1/attention/test_flashinfer_mla_dcp.py (+10/-0); tests/v1/attention/test_mla_backends.py (+1/-0); tests/v1/attention/test_rocm_aiter_mla_mtp_split.py (+8/-0); tests/v1/worker/test_mamba_hybrid_model_state.py (+1/-0); vllm/models/dots3_note/nvidia/attention.py (+2/-1); (+3 more)
LABELS: rocm, ready, nvidia, mrv2, dflash, kimi, k3
BODY: ## Summary ⏎  ⏎ Make Kimi-K3's MLA and KDA decode paths capturable as FULL varlen CUDA graphs so they work under adaptive DSpark verification (#47808), which schedules ragged per-request draft budgets and requires device-sourced query lengths. Before this change K3's `FLASHINFER_MLA` reported `UNIFORM_BATCH`/`UNIFORM`, and the decode builders derived per-request query length from a measured/averaged count, so a graph captured on a uniform dummy did …[truncated]

### L3-1b3b88ec2b  (L3, 2026-09-22, sha 1b3b88ec2b74, PR #57250)
TITLE: [Core] structured generation mode for DiffusionGemma model (Jev-like) (#57250)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: examples/features/structured_diffusion/README.md (+74/-0); examples/features/structured_diffusion/structured_server.py (+1528/-0); tests/config/test_model_arch_config.py (+25/-0); tests/test_sampling_params.py (+182/-0); tests/v1/core/test_scheduler.py (+130/-0); tests/v1/core/utils.py (+12/-1); tests/v1/engine/test_input_processor_trace_replay.py (+2/-0); tests/v1/sample/test_diffusion_gemma_reads.py (+319/-0); vllm/config/scheduler.py (+11/-7); vllm/config/vllm.py (+10/-0); (+6 more)
LABELS: documentation, speculative-decoding, ready, mrv2, scheduler
BODY: ## Prerequisite PRs ⏎  ⏎ A few PRs that can go in on their own merits were split out: ⏎  ⏎  - https://github.com/vllm-project/vllm/pull/57414: hand out stashed logprobs only on the committing step (bug fix) ⏎  - https://github.com/vllm-project/vllm/pull/57416: give a prefill-only batch the model state's number of logit rows (perf) ⏎  - https://github.com/vllm-project/vllm/pull/57417: honor logprob_token_ids on the converging step (feature parity w/autoregres …[truncated]

### L3-c9b34fdb2d  (L3, 2026-09-22, sha c9b34fdb2d0c, PR #58061)
TITLE: [Bugfix][GLM-5.3-Flash] Run the dense MLP layers on the sequence-parallel shard (#58061)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/glm5next/test_sequence_parallel.py (+114/-0); vllm/models/glm5next/common/model.py (+1/-0)
LABELS: bug, ready, glm
BODY: ## Purpose ⏎  ⏎ GLM-5.3-Flash served with DP > 1, TP > 1 and `--enable-expert-parallel` (the configuration that turns on `use_sequence_parallel_moe`) sporadically emits a wrong first token, stray `</think>` tokens followed by a repeated reasoning block, and occasionally byte garbage such as `ԥԥԥԥ� Canadian French, ԥԥ�1ԥ…`. DP with TP=1 and TP-only serving are clean. ⏎  ⏎ `Glm5NextModel` follows the DeepSeek-V4 sequence-parallel layout: the token dime …[truncated]

### L3-973a3be780  (L3, 2026-09-23, sha 973a3be780bd, PR #57732)
TITLE: [Refactor][Quantization] Make FP8 and MLA weight transforms reusable pure functions (#57732)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+23/-23); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py (+1/-4); vllm/model_executor/layers/quantization/fp8.py (+0/-1); vllm/model_executor/layers/quantization/modelopt.py (+1/-1); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+5/-5); vllm/models/kimi_k3/nvidia/mla.py (+6/-15)
LABELS: ready, quantization, kimi, k3
BODY: ## Summary ⏎  ⏎ vLLM keeps a storage convention for FP8 linear weights: non-block strategies store `layer.weight` as `(K, N)`, block stores it as `(N, K)`. The three `process_fp8_weight_*_strategy` helpers do everything except the final step that establishes it, so each call site ends with a hand-written `weight.t()` — or, for block, deliberately omits one. ⏎  ⏎ That makes a three-way rule (tensor → transpose, channel → transpose, block → don't) invisibl …[truncated]

### L3-f9dce295c9  (L3, 2026-09-23, sha f9dce295c96f, PR #57451)
TITLE: [ROCm][DSv4][Perf] Fuse the inverse RoPE into the sparse decode reduce (#57451)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/amd/rocm.py (+27/-3)
LABELS: rocm, ready, deepseek, DSv4, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ DeepSeek-V4 shares the ROCm sparse decode kernels with V4.1, so the reduce ⏎ epilogue added in #57435 applies here with no kernel work at all — only the ⏎ call site has to opt in. On decode-only steps the standalone ⏎ `_inverse_rope_gptj_kernel` stops launching entirely; on V4.1 that measured ⏎ 201.6 us per step against 8.3 us added to the reduce, for 193 us net. ⏎  ⏎ #57435 has since merged, so this is no longer stacked: the epilogue, the ⏎ `inver …[truncated]

### L3-c843f0cab7  (L3, 2026-09-23, sha c843f0cab78b, PR #55277)
TITLE: [Bugfix][SM120][MLA] Support NoPE sparse MLA (GLM-5.3-Flash) on the FlashInfer SM120 backend (#55277)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: L3.cache.cuda_reshape
FILES: csrc/libtorch_stable/cache_kernels.cu (+11/-6); tests/kernels/attention/test_cache.py (+44/-0)
LABELS: bug, ready, nvidia, glm
BODY: ## Purpose ⏎  ⏎ Enable native NoPE (`qk_rope_head_dim=0`) cache writes for GLM-5.3-Flash on `FLASHINFER_MLA_SPARSE_SM120`, without padding the query or allocating a synthetic RoPE tensor. ⏎  ⏎ Rebased onto `a6131b0e656e18eeb3797b8856aa3dbc69efd8e5`. The former physical top-k capacity patch has been dropped: #53781 already implements it in the shared `_run_mqa_kernel`, including HiSparse paths. This PR now contains the cache-writer fix and regression test …[truncated]

### L3-a9f07d0bc5  (L3, 2026-09-23, sha a9f07d0bc596, PR #58282)
TITLE: [ROCm][CI] Mirror the three TurboQuant evaluation groups on MI355 (#58282)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+78/-0); .buildkite/test_areas/lm_eval.yaml (+84/-0); tests/evals/gsm8k/configs/Qwen3-4B-TQ-k3v4nc-ROCm.yaml (+6/-0); tests/evals/gsm8k/configs/Qwen3-4B-TQ-k8v4-ROCm.yaml (+6/-0); tests/evals/gsm8k/configs/Qwen3-4B-TQ-t4nc-ROCm.yaml (+6/-0); tests/evals/gsm8k/configs/models-turboquant-k3v4nc-rocm.txt (+1/-0); tests/evals/gsm8k/configs/models-turboquant-k8v4-rocm.txt (+1/-0); tests/evals/gsm8k/configs/models-turboquant-t4nc-rocm.txt (+1/-0)
LABELS: rocm, ci/build, quantization
BODY: - Add MI355 mirrors for `TurboQuant k8v4`, `TurboQuant t4nc` and `TurboQuant k3v4nc`, with matching legacy AMD jobs. ⏎ - Add ROCm configuration/list pairs that select `TRITON_ATTN` for native-precision boundary layers while retaining TurboQuant on compressed layers. ⏎ - Preserve the full GSM8K workloads, model settings, acceptance thresholds and tolerance. ⏎ - Set explicit 30-minute AMD timeouts and include shared evaluation helpers and the selected …[truncated]

### L3-26879f3260  (L3, 2026-09-23, sha 26879f326042, PR #58099)
TITLE: [ROCm][Compile] Fuse AITER static FP8 attention output (#58099)
SOURCES: path_integration+keyword, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/compilation/passes/fusion/attn_quant_fusion.py (+54/-7); .buildkite/test-amd.yaml (+23/-0); .buildkite/test_areas/compile.yaml (+25/-0); tests/compile/correctness_e2e/test_attn_quant.py (+88/-0); tests/compile/passes/test_fusion_attn.py (+61/-5)
LABELS: rocm, torch.compile, ci/build
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: - Match the actual static AITER per-tensor quantization operator after attention and preserve its scale output when replacing it with fused attention output quantization. ⏎ - Add BF16/FP16 regressions using non-unit scale `0.125`, checking numerical agreement, an unchanged scale, and removal of exactly the output quantizer while query quantization remains. ⏎ - Add a pinned trained FP8 Llama comparison requiring output-quantization fusion in all 32  …[truncated]

### L3-9127170b88  (L3, 2026-09-23, sha 9127170b880d, PR #57176)
TITLE: [Quantization] Select per-token NVFP4 MoE backends explicitly (#57176)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/quantization/test_online.py (+71/-0); vllm/model_executor/layers/fused_moe/all2all_utils.py (+12/-2); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+8/-17); vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (+1/-0); vllm/model_executor/layers/quantization/online/nvfp4.py (+2/-2); vllm/model_executor/layers/quantization/utils/quant_utils.py (+3/-0)
LABELS: ready, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ Online `nvfp4_per_token` previously advertised the ordinary NVFP4 activation scheme even though execution uses per-token scales. This could select an incompatible backend. One-sided dispatch also sized its buffer for packed FP4 although the per-token path sends unpacked activations and quantizes inside the expert kernel. ⏎  ⏎ - Add `kNvfp4DynamicToken` and use it for online NVFP4 MoE selection. ⏎ - Declare TRTLLM support and size modular ex …[truncated]

### L3-1904ed9402  (L3, 2026-09-23, sha 1904ed9402ce, PR #58225)
TITLE: [Perf][ROCm][Attention] Narrow the Triton prefill-attention KV tile on RDNA3/RDNA4 (#58225)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/triton_prefill_attention.py (+15/-1); tests/kernels/attention/test_triton_prefill_attention.py (+85/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Addresses #58060. On RDNA3/RDNA4, `context_attention_fwd` in `vllm/v1/attention/ops/triton_prefill_attention.py` launches `_fwd_kernel` with a 128x128 tile for bf16/fp16, which is slow on these GPUs. This PR narrows only the KV tile (`BLOCK_N`) to 32 when `on_gfx1x()` is true. `BLOCK_M`, `num_warps` and `num_stages` are unchanged, float32 is untouched (its tile is already 32), and CDNA, gfx1250 and NVIDIA keep the existing configura …[truncated]

### L3-88afb9dcac  (L3, 2026-09-23, sha 88afb9dcac76, PR #56252)
TITLE: [CPU] Adds support for fp32 attention sinks (#56252)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+9/-0); csrc/cpu/cpu_attn.cpp (+13/-1); csrc/cpu/cpu_attn_impl.hpp (+24/-9); tests/kernels/attention/test_cpu_attn.py (+49/-3)
LABELS: ready, cpu
BODY: ## Purpose ⏎  ⏎ This PR adds support for fp32 in attention sinks, to support running the gpt-oss-20B-Bf16 model on fp16. It converts all the tensors other than bf16 to fp32 to compute attention sinks. ⏎ It also updates the unit tests and adds test coverage for fp16.   ⏎  ⏎ ## Test Plan ⏎  ⏎ 1) Run the unit tests for the added support: ⏎    ` python -m pytest tests/kernels/attention/test_cpu_attn.py -q` ⏎ 1) Install the public fork in the validation env an …[truncated]

### L3-b715699df4  (L3, 2026-09-23, sha b715699df45f, PR #50212)
TITLE: [ROCm][Perf] Extend QK-norm/RoPE/KV-cache fusion to MRoPE (#50212)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_unified, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+24/-0); vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+45/-1); tests/compile/fusions_e2e/common.py (+4/-1); tests/compile/fusions_e2e/conftest.py (+15/-2); tests/compile/passes/test_rocm_aiter_qk_norm_rope_kvcache_fusion.py (+393/-31); vllm/_aiter_ops.py (+109/-0); vllm/compilation/passes/fusion/matcher_utils.py (+69/-0); vllm/compilation/passes/fusion/qk_norm_rope_kvcache_fusion.py (+564/-78); vllm/config/compilation.py (+4/-4); vllm/config/vllm.py (+1/-1); (+1 more)
LABELS: rocm, ready, torch.compile, qwen
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Extend the existing `QkNormRopeKvCacheFusionPass` to support Qwen3-VL 3D MRoPE. For eligible decode ranges, it replaces two Q/K RMSNorm launches, MRoPE, and the unified FP8 KV-cache update with one AITER attention-prologue kernel: **three fewer GPU launches per layer**. ⏎  ⏎ This is a model-agnostic compile-pass integration: it removes the earlier Qwen3-specific flag/wrapper and uses the existing O2/O3 pass controls. Unsupported configura …[truncated]

### L3-8033bd08c1  (L3, 2026-09-23, sha 8033bd08c1d3, PR #58095)
TITLE: [ROCm][CI] Validate Mooncake and NIXL prefill/decode accuracy (#58095)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+43/-0); .buildkite/test_areas/disaggregated.yaml (+23/-0); .buildkite/test_areas/disaggregated_mooncake.yaml (+24/-0); tests/v1/kv_connector/mooncake_integration/test_accuracy_rocm.py (+29/-0); tests/v1/kv_connector/nixl_integration/__init__.py (+2/-0); tests/v1/kv_connector/nixl_integration/test_accuracy_rocm.py (+34/-0); tests/v1/kv_connector/rocm_pd_accuracy_utils.py (+254/-0)
LABELS: rocm, ready, ci/build, kv-connector
BODY: - Add two Mooncake HIP and six NIXL/AITER cases, preserving upstream model/TP pairs, all 1,319 GSM8K questions, 100 concurrent requests and the original accuracy bounds. ⏎ - Pin model/tokenizer revisions and require positive evaluation-only external-KV deltas; also require transferred bytes for NIXL, with explicit cache bounds and owned-process cleanup. ⏎ - Add MI355 native mirrors and matching optional legacy jobs, preserving the existing NVIDIA a …[truncated]

### L3-b5ea0c7107  (L3, 2026-09-23, sha b5ea0c71078b, PR #57923)
TITLE: [Bugfix][ROCm] Fix startup OOM in AITER MLA FP8 prefill workspace sizing (#57923)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+10/-1)
LABELS: bug, rocm, ready
BODY: ## Purpose ⏎  ⏎ Fixes a startup OOM on Kimi K2.5 (or similar) when using the AITER ASM MLA prefill kernel, caused by overly pessimistic bounds on the size of the required persistent metadata buffers. These sizes were subsequently passed to the workspace manager which triggered the OOM. ⏎  ⏎ **Triggered by**: The AITER bump to v0.1.22post1 picked up ROCm/aiter#3606 which slightly changed the logic for workspace sizing.  ⏎  ⏎ **Long term fix in AITER**:  …[truncated]

### L3-711fc55c10  (L3, 2026-09-23, sha 711fc55c10a2, PR #57586)
TITLE: [Perf] Use breakable CUDA graphs (no torch.compile) by default under VLLM_BATCH_INVARIANT so the tuned matmul configs see the runtime M (#57586)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/features/batch_invariance.md (+3/-0); tests/test_config.py (+66/-1); vllm/config/vllm.py (+49/-0)
LABELS: documentation, ready, torch.compile, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ `VLLM_BATCH_INVARIANT=1` routes every linear layer through the batch-invariant persistent Triton matmul, which picks its tile config from a per-architecture table keyed by `(N, K)` and an **M bucket** (#53247 Ada/Hopper, #53649 SM100, #57456 SM120). On the default `torch.compile` path that lookup (`_get_matmul_config`, called from `matmul_persistent`) runs while Dynamo traces the model with the compile-time dummy M, and the selected ` …[truncated]

### L3-21ee743f47  (L3, 2026-09-23, sha 21ee743f474a, PR #58465)
TITLE: [CI][Bugfix] Limit MRV2 sampler JIT warmup registration to ROCm (#58465)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/model_runner.py (+9/-7)
LABELS: bug, rocm, mrv2
BODY: ## Purpose ⏎  ⏎ #58092 ("[ROCm][Bugfix] Register MRV2 sampler JIT warmups") calls `register_top_k_top_p_warmups()` in the MRV2 `GPUModelRunner` on **all** platforms, not just ROCm. On CUDA these Triton sampler warmups recompile on most engine starts (about 110s each, against `finished in 0.00s` before), so NVIDIA CI jobs slowed down a lot and many now time out. ⏎  ⏎ This PR keeps the registration for ROCm, which #58092 targeted, and puts CUDA back where  …[truncated]

### L3-6632ed559e  (L3, 2026-09-23, sha 6632ed559e2a, PR #58169)
TITLE: [Perf][Attention] Avoid CPU-GPU sync in DCP sequence lengths (#58169)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+3/-1)
LABELS: ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Passing an integer `dcp_rank` to `get_dcp_local_seq_lens` creates a CUDA scalar tensor, introducing a synchronous host-to-device copy during DCP attention metadata construction. Keep the rank as a Python scalar so PyTorch uses it directly in the existing tensor arithmetic. ⏎  ⏎ Related work: #54790 includes the same scalar fix alongside fused draft metadata updates for sparse MLA. This PR intentionally extracts the shared helper fix i …[truncated]

### L3-e26547adfb  (L3, 2026-09-23, sha e26547adfb49, PR #49845)
TITLE: [Bugfix] Pick a KV block size supported by every attention backend (#49845)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/platforms/cpu.py (+3/-3); vllm/platforms/interface.py (+72/-18); tests/v1/worker/test_gpu_model_runner.py (+57/-0)
LABELS: bug, ready, v1, cpu
ISSUES: #48286 DeepseekV32IndexerBackend requires --block-size 64 — not documented or auto-detected
BODY: ## Purpose ⏎  ⏎ Fixes #48286. ⏎  ⏎ `Platform.update_block_size_for_backend()` picks the KV cache block size from the **first** non-SSM attention backend only. Models that mix attention backends with disjoint block-size support break: GLM-DSA / DeepSeek-V3.2-style models pair a sparse-MLA main backend (`FLASHINFER_MLA_SPARSE`, supports `[32, 64]`, prefers 32) with `DeepseekV32IndexerBackend` (supports only `[64]`). The auto-selected 32 then fails much lat …[truncated]

### L3-4fb767ff0a  (L3, 2026-09-23, sha 4fb767ff0a0e, PR #53175)
TITLE: [Kernel] Resubmit PR 48666 - Gemma4 FP8 KV FA4 head dim 512 backend selection (#53175)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fa4_cutedsl, L3.flash_attn.fa_utils, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashmla_sparse, L3.mla.flashinfer_sparse, L3.mla.rocm_aiter, L3.mla.rocm_aiter_sparse, L3.mla.flashattn_sparse, L3.dispatch.abstract_interface, L3.flex_attention
FILES: vllm/model_executor/layers/attention/attention.py (+14/-5); vllm/v1/attention/backend.py (+3/-1); vllm/v1/attention/backends/b12x.py (+1/-1); vllm/v1/attention/backends/composite.py (+11/-5); vllm/v1/attention/backends/cpu_attn.py (+1/-1); vllm/v1/attention/backends/fa_utils.py (+5/-1); vllm/v1/attention/backends/flash_attn.py (+81/-16); vllm/v1/attention/backends/flashinfer.py (+1/-1); vllm/v1/attention/backends/flex_attention.py (+1/-1); vllm/v1/attention/backends/hpc_attn.py (+1/-1); (+37 more)
LABELS: rocm, speculative-decoding, ready, ci/build, deepseek, cpu, nvidia, mrv2, DSv4, minimax
BODY: ## Purpose ⏎  ⏎ Re-submit https://github.com/vllm-project/vllm/pull/48666 due to https://github.com/vllm-project/vllm/pull/52987. ⏎ Flagged by CI for perf regression, but the regressed perf is resulting from Model emitting special token. ⏎  ⏎ The model emits the special token because plain serving with [RedHatAI/gemma-4-31B-it-FP8-dynamic](https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-dynamic) before https://github.com/vllm-project/vllm/pull/48666 use …[truncated]

### L3-e9f167129c  (L3, 2026-09-23, sha e9f167129c09, PR #58369)
TITLE: [CI][ROCM] Add the Fusion E2E TP2 Quick group on MI355, and the AITER MLA fix it needs (#58369)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+50/-7); .buildkite/test-amd.yaml (+41/-0); .buildkite/test_areas/compile.yaml (+41/-0); tests/compile/fusions_e2e/models.py (+9/-0); tests/compile/fusions_e2e/test_tp2_ar_rms.py (+57/-8); tests/v1/attention/test_rocm_aiter_mla_fp8_decode_routing.py (+19/-0); tests/v1/attention/test_rocm_aiter_mla_mtp_split.py (+3/-1)
LABELS: rocm, torch.compile, ci/build
BODY: ## Purpose ⏎  ⏎ **Add one AMD CI group.** It mirrors `(H100) Fusion E2E TP2 Quick`, keeping only the payloads the two AITER backends support, and is split out of #50519 so it can be reviewed and validated on its own. ⏎  ⏎ **The test could not run on AMD.** `test_tp2_ar_rms_fp8_fusions` was CUDA only and offered three NVIDIA backends, so the models and backends are now parametrized per platform and `ROCM_AITER_MLA` is added as a backend case. ⏎  ⏎ **The …[truncated]

### L3-bcdacfc1ff  (L3, 2026-09-23, sha bcdacfc1ffee, PR #55612)
TITLE: [Test][Determinism] Cover chunked prefill in the batch-invariance suite (#55612)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/determinism/test_batch_invariance.py (+80/-7)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Batch invariance is what stops the same request coming back with different text ⏎ depending on how busy the server was. `tests/v1/determinism/` guards that — but ⏎ not for chunked prefill, which is what a loaded server does constantly: a long ⏎ prompt is split across several scheduler steps, and the later chunks attend over ⏎ KV computed in the earlier ones. ⏎  ⏎ The suite says so itself, in a TODO left by #32561: ⏎  ⏎ ```python ⏎ # TODO: Up …[truncated]

### L3-e78e367c6e  (L3, 2026-09-23, sha e78e367c6e22, PR #54988)
TITLE: [ROCm] Give turboquant boundary layers a layout-compatible backend (#54988)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.dispatch.selector, L3.platform.rocm_selection
FILES: vllm/v1/attention/selector.py (+7/-0); tests/v1/attention/test_rocm_attention_backends_selection.py (+143/-0); vllm/platforms/rocm.py (+82/-16)
LABELS: rocm, ready, quantization
BODY: ## Purpose ⏎  ⏎ The `(MI300) LM Eval TurboQuant KV Cache` group, added by [#50519](https://github.com/vllm-project/vllm/pull/50519), fails all four of its tests in [AMD CI build 12548](https://buildkite.com/vllm/amd-ci/builds/12548/list?sid=01a0609c-72b4-4312-b09a-ec5852db1fc7&tab=output). Each server exits during startup with: ⏎  ⏎ ``` ⏎ ValueError: No KV cache layout satisfies every supported set: [['LHBNC', 'LBHNC'], ['LBNHC']]. ⏎ ``` ⏎  ⏎ A `turboqua …[truncated]

### L3-e6dc16cebd  (L3, 2026-09-23, sha e6dc16cebd57, PR #57075)
TITLE: [PCP] Support prefill context parallelism with data parallelism (#57075)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface, L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/v1/attention/backends/utils.py (+1/-1); vllm/v1/attention/ops/pcp.py (+9/-2); .buildkite/test_areas/misc.yaml (+2/-0); tests/test_pcp_dp.py (+161/-0); tests/v1/worker/test_gpu_autoregressive_speculator.py (+44/-0); tests/v1/worker/test_gpu_pcp_manager.py (+12/-0); vllm/distributed/device_communicators/all2all.py (+7/-0); vllm/distributed/elastic_ep/standby_state.py (+7/-3); vllm/forward_context.py (+14/-3); vllm/model_executor/layers/fused_moe/runner/moe_runner.py (+5/-1); (+5 more)
LABELS: rocm, speculative-decoding, ready, ci/build, deepseek, nvidia, mrv2, glm
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Enable prefill context parallelism (PCP) with data parallelism (DP), including expert-parallel MoE and MTP decoding. ⏎  ⏎ Reuse existing communication groups, add sequence-parallel GLM MoE support, and fix DP padding and idle-rank handling in PCP draft execution. No new process group. ⏎  ⏎ Continues #54131 with the original authorship preserved; supersedes the earlier approach in #49246. ⏎  ⏎ Validation: ⏎  ⏎ - `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest …[truncated]

### L3-44bf011f44  (L3, 2026-09-23, sha 44bf011f447b, PR #58069)
TITLE: [Dependency] Upgrade FlashInfer version to 0.7.0 (#58069)
SOURCES: path_core, path_integration+keyword, subject_keyword, dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.mla.flashinfer_sparse
FILES: docker/Dockerfile (+1/-1); docker/versions.json (+1/-1); requirements/cuda.txt (+2/-2); requirements/rubin-prerelease.txt (+3/-3); vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py (+33/-11); tests/kernels/attention/test_flashmla_sparse.py (+56/-0); tests/samplers/conftest.py (+21/-0); tests/v1/attention/test_flashinfer_mla_sparse_sm90.py (+55/-6); tests/v1/sample/test_topk_topp_sampler.py (+5/-2); vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+1/-0); (+5 more)
LABELS: ready, ci/build, deepseek, nvidia, ready-run-all-tests, DSv4, DSv4.1
BODY: ## Purpose ⏎ Upgrade FlashInfer version to 0.7.0 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-23110a0c9e  (L3, 2026-09-24, sha 23110a0c9ef8, PR #58275)
TITLE: [Bugfix] Capture prefill kernels for mixed FULL graphs (#58275)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/fullgraph/test_full_cudagraph.py (+25/-15); tests/v1/cudagraph/test_cudagraph_manager.py (+86/-0); vllm/v1/worker/gpu/cudagraph_utils.py (+5/-1); vllm/v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py (+1/-0)
LABELS: bug, speculative-decoding, torch.compile, nvidia, mrv2
BODY: - Give mixed FULL graph capture a query-length bound covering the entire token batch so prefill kernels are captured with sufficient coverage. ⏎ - Preserve explicit variable-length and uniform decode bounds in both the target-model and autoregressive speculator capture paths. ⏎ - Add regression coverage for mixed, piecewise, and decode-only capture metadata, including speculative warmup and capture. ⏎ - Pass the selected attention backend to both te …[truncated]

### L3-5963795ec7  (L3, 2026-09-24, sha 5963795ec7d7, PR #54508)
TITLE: [Attention][CPU] Run Zen CPU encoder attention on zentorch SDPA (#54508)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: vllm/v1/attention/backends/cpu_attn.py (+26/-0); vllm/v1/attention/backends/zentorch_sdpa.py (+148/-0); vllm/v1/attention/ops/vit_attn_wrappers.py (+11/-3); setup.py (+1/-1); tests/kernels/attention/test_cpu_attn.py (+133/-0)
LABELS: ready, ci/build, cpu
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ On CPU, encoder / encoder-only attention runs a two-step dance: a reshape-and-cache prologue (`_C::cpu_attn_reshape_and_cache`) stages K/V into a scratch cache, then `_C::cpu_attention_with_kv_cache` reads it straight back on the **same** step — even though an encoder never reuses that cache across steps. ⏎  ⏎ `zentorch::zentorch_sdpa` attends the packed QKV **in place**, so this PR routes encoder / encoder-only attention to it on AMD …[truncated]

### L3-c3f5270270  (L3, 2026-09-24, sha c3f5270270ae, PR #58432)
TITLE: [ROCm][CI] Add the MI355 TurboQuant t3nc mirror (#58432)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+27/-0); .buildkite/test_areas/lm_eval.yaml (+28/-0); tests/evals/gsm8k/configs/Qwen3-4B-TQ-t3nc-ROCm.yaml (+6/-0); tests/evals/gsm8k/configs/models-turboquant-t3nc-rocm.txt (+1/-0)
LABELS: rocm, ci/build, quantization
BODY: - Add a MI355 `mirror.amd` to `lm-eval-turboquant-t3nc` and a matching native AMD job in `.buildkite/test-amd.yaml`. ⏎ - Add the ROCm t3nc model configuration and list, selecting `TRITON_ATTN` for native-precision boundary layers while compressed layers retain TurboQuant. ⏎ - Preserve Qwen3-4B, all 1,319 GSM8K questions, five shots, the 4,096-token context, the 0.75 accuracy threshold, and the existing 0.08 tolerance. ⏎  ⏎ TurboQuant t3nc now has its …[truncated]

### L3-26f49e336a  (L3, 2026-09-24, sha 26f49e336a14, PR #57918)
TITLE: [Perf][Attention] Bound FlashInfer prefill dequantization scratch (#57918)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+23/-1); tests/v1/attention/test_trtllm_attention_integration.py (+71/-1)
LABELS: ready, nvidia, quantization, verified
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ The FlashInfer TRTLLM prefill fallback for BF16/FP16 queries with FP8 KV allocates temporary dequantized pages for every block-table entry, including unused columns. ⏎  ⏎ Bound the prefill block table using host sequence-length bounds before dequantization. This includes cached prefixes and preserves the contiguous strides required by the dequantization kernel, including for a single prefill row. ⏎  ⏎ For the regression test’s single prefill  …[truncated]

### L3-721d0e5c11  (L3, 2026-09-24, sha 721d0e5c112a, PR #58456)
TITLE: [ROCm][DSv4.1][Perf] Emit MXFP8 from the sparse decode reduce and run wo_a as a grouped FP8 GEMM (#58456)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+351/-5); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+177/-2); vllm/models/deepseek_v41/amd/rocm.py (+136/-22)
LABELS: rocm, ready, deepseek, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Follow-up to #57435. That PR moved the inverse RoPE into the sparse-decode ⏎ reduce epilogue, but the reduce still writes the attention output as bf16 ⏎ `[T, H, 512]` and `wo_a` then reads it back through a bf16 hipBLASLt einsum ⏎ (`Cijk_Alik_Bljk_BBS_BH_Bias...`) on a dequantized bf16 copy of the weight. ⏎ `wo_a` is weight-bandwidth bound at decode, so it pays for twice the bytes ⏎ the checkpoint actually stores. ⏎  ⏎ This PR takes the re …[truncated]

### L3-4f72bd0152  (L3, 2026-09-24, sha 4f72bd0152e7, PR #58444)
TITLE: [Bugfix][Quantization] Give LM heads standard linear metadata (#58444)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/quantization/test_lm_head.py (+74/-0); tests/quantization/test_modelopt.py (+0/-43); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py (+0/-2); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa4.py (+0/-2); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8.py (+0/-2); vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa8o8.py (+0/-4); vllm/model_executor/layers/quantization/modelopt.py (+0/-6); vllm/model_executor/layers/vocab_parallel_embedding.py (+19/-11)
LABELS: bug, ready, quantization
BODY: ## Purpose ⏎  ⏎ Checkpoints that explicitly quantize `lm_head` can fail during Humming weight preparation because `ParallelLMHead` lacks the linear dimensions and bias metadata expected by quantization backends. ⏎  ⏎ Retain the linear weight dimensions, partition sizes, and prefix in `VocabParallelEmbedding` before quantization dispatch, and initialize `ParallelLMHead.has_bias` before weight creation. The weight factory consumes the same dimensions expos …[truncated]

### L3-3652f35e7c  (L3, 2026-09-24, sha 3652f35e7c75, PR #57954)
TITLE: [Bugfix][Quantization] Refresh online NVFP4 scales before reload post-processing (#57954)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_trtllm_nvfp4_moe.py (+54/-0); tests/quantization/test_online.py (+29/-8); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+13/-12); vllm/model_executor/layers/quantization/online/nvfp4.py (+8/-0)
LABELS: bug, ready, nvidia, quantization
BODY: ## Problem and change ⏎  ⏎ Online NVFP4 reuses its MoE kernel and quantization config across weight reloads. Conversion installs new weight-scale tensors on the layer, but the reused config still references the previous alpha storage. Expert post-processing can therefore rebuild derived scales (for example TRTLLM's `g1_scale_c`) using the previous weights' scales. Restoring graph-visible parameter storage afterward is too late to correct those derive …[truncated]

### L3-3059155b47  (L3, 2026-09-24, sha 3059155b47ce, PR #58510)
TITLE: [ROCm][Perf] MXFP8 GEMM on native 32x32 block scales for gfx950 (#58510)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+6/-3); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+15/-2); tests/kernels/quantization/test_rocm_mxfp8_linear.py (+141/-0); vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py (+711/-0); vllm/model_executor/kernels/linear/mxfp8/rocm_native.py (+26/-2)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ DeepSeek V4/V4.1 checkpoints store MXFP8 weight scales in 32x32 blocks. The ⏎ ModelOpt loader expands them to one scale row per weight row, and on gfx950 ⏎ `RocmDotScaledMxfp8LinearKernel` then runs one generic `tl.dot_scaled` ⏎ kernel on those per-row scales. On DeepSeek-V4.1-Flash these linears ⏎ (`fused_wqa_wkv`, `wq_b`, the indexer `wq_b`, `wo_b`, and the shared-expert ⏎ `gate_up` and `down`) make up 11.7% of GPU kernel time per decode step …[truncated]

### L3-04730e8270  (L3, 2026-09-24, sha 04730e82700d, PR #57632)
TITLE: [DFlash] Capture the context K/V precompute in the draft CUDA graph (#57632)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py (+16/-8); vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (+46/-22)
LABELS: speculative-decoding, ready, deepseek, nvidia, mrv2, dflash, DSv4.1
BODY: ## Purpose ⏎  ⏎ Even when the query forward replays a FULL CUDA graph, the DFlash/DSpark draft step still runs the context K/V precompute (`precompute_and_store_context_kv`) eagerly outside the graph, because the number of context tokens varies from step to step. ⏎  ⏎ This PR captures the context K/V precompute ahead of the query forward in the draft's FULL decode graph. Each graph stores one full verify of context rows per request; rows past the bat …[truncated]

### L3-5ff3bbfb08  (L3, 2026-09-24, sha 5ff3bbfb089a, PR #58621)
TITLE: [Perf][DSv4] Fuse inverse RoPE + FP8 quant into FlashInfer sparse MLA (#58621)
SOURCES: subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_flashmla_sparse.py (+275/-0); vllm/models/deepseek_v4/attention.py (+37/-20); vllm/models/deepseek_v4/common/ops/cache_utils.py (+72/-8); vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py (+116/-60); vllm/models/deepseek_v4/nvidia/ops/o_proj.py (+83/-16)
LABELS: ready, deepseek, nvidia, quantization, DSv4
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ FlashInfer 0.7.0 (already pinned) ships the TRTLLM-GEN DSv4 sparse-MLA **RopeQuant** epilogue (flashinfer-ai/flashinfer#4918): the kernel applies the inverse RoPE to the attention output and casts it to FP8 with packed UE8M0 scales, in the layout DeepGEMM's `fp8_einsum` consumes. This PR uses it on `FLASHINFER_MLA_SPARSE_DSV4` for DeepSeek V4, so `wo_a` reads the kernel's FP8 output directly instead of a bf16 `[T, 128, 512]` output pl …[truncated]

### L3-d5051abaf1  (L3, 2026-09-24, sha d5051abaf19a, PR #58368)
TITLE: [Bugfix][Mamba] Restore prompt-tail prefix-cache hits with MTP (#58368)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py (+22/-0); vllm/v1/core/single_type_kv_cache_manager.py (+1/-1)
LABELS: bug, ready, verified, kv-cache-manager
BODY: Regression introduced in #55390. ⏎ Affects hybrid Mamba/GDN models with MTP, prefix caching, `--mamba-cache-mode align`, and `hash_block_size` smaller than the Mamba `block_size`. ⏎  ⏎ | Model                                          | Prefix-cache hit (%): before → main → fix | Output TPS/GPU: before → main → fix | ⏎ |------------------------------------------------|-------------------------------------------|-------------------------------------| ⏎  …[truncated]

### L3-609731e8f5  (L3, 2026-09-25, sha 609731e8f56c, PR #57679)
TITLE: [Perf][DSv4.1] Restore the fused query RMSNorm + MXFP8 quantization path (#57679)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v41/common/ops/query_quant.py (+8/-3)
LABELS: ready, deepseek, quantization, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ `can_fuse_query_quant` decides whether the DeepSeek-V4.1 attention block may hand `wq_b` a pre-quantized query, letting `fused_q_kv_rmsnorm_quant` do the q/kv RMSNorm and the MXFP8 quantization in one kernel. It probed `linear.input_quant_key`. #53793 renamed that attribute to the private `_input_quant_key` and introduced `get_input_quant_key` as its reader, but this call site was not updated, and it is the last one in the tree stil …[truncated]

### L3-f55419c43c  (L3, 2026-09-25, sha f55419c43ca6, PR #54223)
TITLE: [Bugfix][Quantization] Fix MXFP8 startup crash on layers below mm_mxfp8 shape limits (#54223)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/core/test_fused_q_kv_rmsnorm.py (+1/-1); tests/kernels/quantization/test_flashinfer_mxfp8_trtllm.py (+9/-3); tests/kernels/quantization/test_marlin_tile_padding.py (+3/-1); tests/kernels/quantization/test_mxfp8_kernel_selection.py (+135/-0); tests/kernels/quantization/test_rocm_mxfp8_linear.py (+2/-2); tests/kernels/test_minimax_m3_amd_ops.py (+4/-2); tests/model_executor/kernels/test_b12x_linear.py (+2/-2); tests/quantization/test_auto_round.py (+2/-2); vllm/model_executor/kernels/linear/__init__.py (+8/-3); vllm/model_executor/kernels/linear/mxfp8/Mxfp8LinearKernel.py (+5/-0); (+6 more)
LABELS: bug, rocm, ready, nvidia, quantization, verified, minimax
BODY: ## Purpose ⏎  ⏎ `vllm serve Qwen/Qwen3.5-0.8B --quantization mxfp8` dies in the profile run on a B200 with `AssertionError: mm_mxfp8 requires N >= 128, got N=32`. The layer is the GDN `in_proj_ba` (weight 32 x 1024). The larger Qwen3.5 sizes and Qwen3-Next have N=64 or N=96 there, so the whole family fails to start with online MXFP8. ⏎  ⏎ `FlashInferCutlassMxfp8LinearKernel` and `FlashInferCutedslMxfp8LinearKernel` return `True` from `can_implement` for  …[truncated]

### L3-442bd00d9f  (L3, 2026-09-25, sha 442bd00d9fef, PR #55072)
TITLE: [Perf][Distributed] Add low-SM multimem reduce-scatter for SM100/SM103 (#55072)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_multimem_reduce_scatter.py (+480/-0); csrc/custom_all_gather_reduce_scatter.cuh (+125/-0); csrc/custom_all_reduce.cuh (+6/-0); csrc/libtorch_stable/custom_all_gather_reduce_scatter.cu (+108/-0); csrc/libtorch_stable/custom_all_gather_reduce_scatter_ops.cpp (+6/-0); csrc/libtorch_stable/ops.h (+4/-0); tests/distributed/test_custom_all_gather_reduce_scatter.py (+68/-0); tests/distributed/test_custom_all_reduce.py (+263/-1); vllm/_custom_ops.py (+20/-0); vllm/distributed/device_communicators/custom_all_reduce.py (+163/-30)
LABELS: performance, ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Add a low-SM MNNVL reduce-scatter backend for TP2, TP4, and TP8 MNNVL groups on SM100 or SM103. The kernel uses `multimem.ld_reduce` to reduce across the NVLS multicast mapping and ordinary local stores for each rank-owned output shard. It launches at most 8 CTAs per GPU with 1024 threads and an 8-way vector unroll. ⏎  ⏎ The dispatcher uses total input message bytes and keeps a single explicit backend policy: ⏎  ⏎ - `<= 16 MiB`: existing MNNV …[truncated]

### L3-29468dde8b  (L3, 2026-09-25, sha 29468dde8b51, PR #55528)
TITLE: [Bugfix][KV Cache][MLA] Align packed block strides for V3.2 sparse MLA (#55528)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+18/-8); tests/v1/attention/test_mla_backends.py (+3/-1); tests/v1/attention/test_sparse_mla_kv_cache_layout.py (+119/-0); tests/v1/core/test_contiguous_kv_packing.py (+27/-0); vllm/v1/core/kv_cache_utils.py (+5/-11); vllm/v1/kv_cache_interface.py (+15/-7)
LABELS: bug, documentation, performance, new-model, rocm, frontend, intel-gpu, speculative-decoding, ready, ray
BODY: ## Purpose ⏎  ⏎ V3.2 sparse MLA did not publish the physical block-stride alignment its selected kernel requires. This PR sets the existing `block_stride_alignment` contract at the producer, lifts the field to `KVCacheSpec`, and lets the allocator consume it without an MLA type check. HiSparse hot-page and stride constraints are combined in one LCM so sequential rounding cannot invalidate the first constraint. ⏎  ⏎ The requirement is selected by backend  …[truncated]

### L3-73a78e6f1f  (L3, 2026-09-25, sha 73a78e6f1f38, PR #57934)
TITLE: [SpecDecode] Add LiLiCorr drafter (#57934)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/features/speculative_decoding/README.md (+1/-0); docs/features/speculative_decoding/lilicorr.md (+77/-0); tests/models/registry.py (+7/-0); tests/test_config.py (+14/-9); tests/v1/spec_decode/test_dflash2.py (+180/-5); tests/v1/spec_decode/test_lilicorr.py (+777/-0); vllm/config/vllm.py (+10/-6); vllm/model_executor/layers/logits_processor.py (+21/-1); vllm/model_executor/models/lilicorr.py (+546/-0); vllm/model_executor/models/nano_nemotron_vl.py (+2/-0); (+11 more)
LABELS: documentation, new-model, speculative-decoding, ready, qwen, mrv2, dflash
BODY: ## What changes ⏎  ⏎ Add `LiLiCorrDraftModel` (https://arxiv.org/abs/2608.20530) to the existing MRV2 DFlash path. The exported SGLang head reranks global candidate lattices using target embeddings, full-vocabulary log probabilities, draft hidden states and the normalized last committed target context. Both plain and grouped-convolution checkpoints are supported. ⏎  ⏎ The port reuses DFlash2 convolution layers and extracts its existing candidate walk …[truncated]

### L3-2edbb32e58  (L3, 2026-09-25, sha 2edbb32e58aa, PR #58704)
TITLE: [Bugfix][GLM-5.3-Flash] SM90 sparse MLA: index_kpool mismatch leads to corruption via unread query token (#58704)
SOURCES: path_core, subject_keyword, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer_sparse
FILES: vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py (+2/-3); tests/v1/attention/test_flashinfer_mla_sparse_sm90.py (+54/-0)
LABELS: bug, ready, nvidia, glm
DEEP_STUDY: deep-study correctness case vllm:2edbb32e58: class=shape_alignment_edge; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: ## Purpose ⏎  ⏎ Another corruption fix found while on the hunt that started in https://github.com/vllm-project/vllm/pull/58454. _Partial_ fix for #56868 and #56605. These both help, but there will likely be one or two more PRs with followups. ⏎  ⏎ Partial fix for GLM-5.3-Flash corruption at medium length context. This is a source for minor corruption: the query token and up to two before it are unread by attention on three of four positions past 2048 …[truncated]

### L3-c88f4824ff  (L3, 2026-09-25, sha c88f4824fff4, PR #58450)
TITLE: [GLM5.3 Perf] Optimize glm 5.3 metadata op, 1.6~4.8x kernel level performance improvement (#58450)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.flashinfer_sparse, L3.mla.flashattn_sparse
FILES: vllm/model_executor/layers/attention/sparse_mla_attention.py (+1/-11); vllm/v1/attention/backends/mla/flashattn_mla_sparse.py (+6/-3); vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+0/-7); tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py (+19/-4)
LABELS: ready, deepseek, nvidia, glm
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ - Reuse `arrange` ⏎ - persistent GPU buffer + Triton kernel for `_build_req_id_per_token` ⏎  ⏎ ## Test ⏎  ⏎ Acc covered by current unit test ⏎  ⏎ Perf can be seen in this ai generated script ⏎  ⏎ ```py ⏎  ⏎ import statistics ⏎ import time ⏎  ⏎ import numpy as np ⏎ import torch ⏎  ⏎ from vllm.utils.torch_utils import np_to_pinned_tensor ⏎ from vllm.v1.attention.backend import CommonAttentionMetadata ⏎  ⏎ TOKENS = [2**i for i in range(16)]  # 1 ... 32768 …[truncated]

### L3-31cc226401  (L3, 2026-09-25, sha 31cc2264011a, PR #58717)
TITLE: [ROCm][CI] Run the MLA attention+quant fusion test on ROCm (#58717)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/passes/test_mla_attn_quant_fusion.py (+16/-1)
LABELS: rocm, torch.compile, quantization
BODY: ## Purpose ⏎  ⏎ **Split out of #50519** and carried over unchanged in intent. Code only. ⏎  ⏎ `tests/compile/passes/test_mla_attn_quant_fusion.py` populates its configuration lists only inside `if current_platform.is_cuda()`. On ROCm they stay empty, so the parametrization yields nothing and pytest skips a placeholder with `got empty parameter set`. The MLA attention plus static FP8 output-quant fusion has never been exercised there. ⏎  ⏎ ## What chang …[truncated]

### L3-378504a544  (L3, 2026-09-25, sha 378504a5442b, PR #58635)
TITLE: [MoE] Defer the TRTLLM-Gen top-k finalize on the modular path (#58635)
SOURCES: subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer.py (+122/-2); tests/kernels/moe/test_ocp_mx_moe.py (+149/-1); tests/kernels/moe/test_trtllm_nvfp4_moe.py (+102/-28); tests/kernels/moe/utils.py (+48/-0); tests/models/kimi_k3/test_latent_moe_tail.py (+6/-6); vllm/model_executor/layers/fused_moe/config.py (+57/-13); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+20/-3); vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py (+36/-9); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+33/-7); vllm/model_executor/layers/fused_moe/fused_moe_method_base.py (+2/-2); (+16 more)
LABELS: ready, nvidia, quantization, kimi, k3
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ #53152 lets the monolithic TRTLLM-Gen experts stop after GEMM2 (`do_finalize=False`) and hand an `UnfinalizedMoEOutput` to a consumer that fuses the top-k reduction into its own kernel. On `main`, a layer asks for this by setting `FusedMoEConfig.defer_moe_finalize`, optionally capped by `defer_moe_finalize_max_num_tokens`. This PR brings the same mechanism to the modular path, which models with their own router use (DeepSeek-V4.1's MX …[truncated]

### L3-b21757555a  (L3, 2026-09-25, sha b21757555ac5, PR #58724)
TITLE: [ROCm][CI] Cover the AITER MQA logits dispatch on gfx950 (#58724)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+59/-0)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ **Split out of #50519** and carried over unchanged in intent. Test only. ⏎  ⏎ `rocm_fp8_mqa_logits` picks between three paths: flydsl on gfx942, the AITER `fp8_mqa_logits` kernel when that module resolves, and a pure-torch fallback. Nothing covered the AITER path. Since the fallback returns the same values, a silent drop to it would still look like a pass, so a value-only test would not have caught it. ⏎  ⏎ ## What changed ⏎  ⏎ One fixtur …[truncated]

### L3-1b922ffcbe  (L3, 2026-09-25, sha 1b922ffcbe75, PR #58748)
TITLE: [ROCm][CI] Add quantized MoE serving test for gfx950 (#58748)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+32/-2); .buildkite/test_areas/quantization.yaml (+28/-2); tests/quantization/test_rocm_moe.py (+146/-0)
LABELS: rocm, ready, ci/build, quantization
BODY: ## Purpose ⏎ Split out of #50519 to keep that PR reviewable. No functional changes outside tests and CI config. ⏎  ⏎ `tests/quantization/test_blackwell_moe.py` guards the startup-and-generation contract for quantized MoE models, but there is no ROCm equivalent, so regressions in the ROCm quantized MoE paths are only caught by much heavier end-to-end jobs. ⏎  ⏎ This adds `tests/quantization/test_rocm_moe.py` as the gfx950 counterpart. It follows the sa …[truncated]

### L3-974cb65155  (L3, 2026-09-25, sha 974cb6515530, PR #48419)
TITLE: [Bugfix] V1: fix allowed_token_ids_mask aliasing in InputBatch.swap_states (#48419)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_input_batch.py (+54/-0); vllm/v1/worker/gpu_input_batch.py (+2/-6)
LABELS: bug, ready, v1, mrv1-only
BODY: ## Purpose ⏎  ⏎ `InputBatch.swap_states()` swaps the two `allowed_token_ids_mask_cpu_tensor` ⏎ rows with a tuple assignment on row **views**: ⏎  ⏎ ```python ⏎ ( ⏎     self.allowed_token_ids_mask_cpu_tensor[i1], ⏎     self.allowed_token_ids_mask_cpu_tensor[i2], ⏎ ) = ( ⏎     self.allowed_token_ids_mask_cpu_tensor[i2], ⏎     self.allowed_token_ids_mask_cpu_tensor[i1], ⏎ ) ⏎ ``` ⏎  ⏎ Integer indexing on a `torch.Tensor` returns a **view**, not a copy. The RHS ⏎ tuple holds two vie …[truncated]

### L3-2f41e00235  (L3, 2026-09-26, sha 2f41e0023511, PR #58609)
TITLE: [CI] Split (B200) Miscellaneous Kernels into mHC, FLA Ops and Misc named jobs (#58609)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/kernels.yaml (+29/-4); tests/kernels/fla/__init__.py (+0/-0); tests/kernels/fla/test_fla_layernorm_guard.py (+0/-0); tests/kernels/fla/test_fused_gdn_post_conv.py (+0/-0); tests/kernels/fla/test_fused_recurrent_packed_decode.py (+0/-0); tests/kernels/fla/test_fused_sigmoid_gating_delta_rule.py (+0/-0); tests/kernels/mhc/__init__.py (+0/-0); tests/kernels/mhc/test_mhc_jit_warmup.py (+0/-0); tests/kernels/mhc/test_mhc_kernels.py (+0/-0); tests/kernels/mhc/test_mhc_tilelang_jit.py (+1/-1); (+1 more)
LABELS: ready, ci/build, cpu
BODY: ## What ⏎  ⏎ `(B200) Miscellaneous Kernels` runs 31–41m of wall time on recent `main`, and two workloads take up most of it: the mHC tests and the `flash_linear_attention` ops tests. This moves each group into its own folder and runs it as a named job, `(B200) mHC Kernels` and `(B200) FLA Ops Kernels`. The root catch-all stays as it is on `main` and adds `--ignore`s for the two new folders. ⏎  ⏎ ## Old ⏎  ⏎ ``` ⏎ kernels-root-misc-test-b200   "(B200) Miscellan …[truncated]

### L3-ad6817b68d  (L3, 2026-09-26, sha ad6817b68d9b, PR #58720)
TITLE: [Perf][MoE] Index expert mapping lookups in RoutedExperts.load_weights (#58720)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_moe_weight_loading_padded.py (+192/-0); vllm/model_executor/layers/fused_moe/routed_experts.py (+32/-2)
LABELS: startup-ux, verified
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ `RoutedExperts.load_weights` scans the whole expert mapping for each checkpoint tensor. Both the mapping size and the tensor count grow with the number of experts, making matching quadratic in expert count. Original profiling of Qwen3.8-Flash-Next-NVFP4 on a DGX Spark attributed about 25 seconds of weight loading to this loop. ⏎  ⏎ Build the index in a small `_index_expert_mapping` helper, grouping entries by the literal logical expert ID …[truncated]

### L3-79ac28c5bd  (L3, 2026-09-26, sha 79ac28c5bd36, PR #58749)
TITLE: [CI] Reduce CUDA graph mode test overhead (#58749)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/cudagraph/test_cudagraph_mode.py (+10/-6)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ The CUDA graph mode matrix took 27m23s in [build 91027](https://buildkite.com/vllm/ci/builds/91027#01a0d4f9-651d-42df-8c11-d2a5ab2675fa). Bound batch and capture sizes for its ten short prompts and remove two duplicate cases. All unique backend/mode combinations, the full model, and automatic memory profiling remain. ⏎  ⏎ Open-PR searches found no duplicate implementation; #58622 addresses the timeout by increasing its limit. ⏎  ⏎ ## Test Pla …[truncated]

### L3-a4eb3f25d6  (L3, 2026-09-26, sha a4eb3f25d6f9, PR #57263)
TITLE: [Spec Decode] Enable Gemma4 DSpark adaptive verification with FlashInfer (#57263)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface
FILES: vllm/config/speculative.py (+2/-0); vllm/model_executor/models/gemma4_dspark.py (+22/-3); vllm/v1/attention/backend.py (+45/-6); vllm/v1/attention/backends/flashinfer.py (+56/-18); vllm/v1/attention/backends/mla/index_group.py (+2/-10); vllm/v1/attention/backends/mla/sparse_swa.py (+4/-7); vllm/v1/attention/ops/dcp.py (+4/-10); vllm/v1/worker/gpu/attn_utils.py (+34/-0); vllm/v1/worker/gpu/cudagraph_utils.py (+2/-1); vllm/v1/worker/gpu/dp_utils.py (+2/-1); (+14 more)
LABELS: documentation, speculative-decoding, ready, nvidia, mrv2, dflash, minimax
BODY: ## Purpose ⏎  ⏎ Enable Gemma4 DSpark adaptive verification with FlashInfer TRTLLM-GEN, building on #52157 (#51871): graph-safe variable-length decode, trained confidence-head loading, graph-cache separation, and RoPE config normalization. ⏎  ⏎ Adaptive verification replays FULL decode CUDA graphs in which each request has 1 to 1 + k query tokens, read from the device `query_start_loc` (k = `num_speculative_tokens`). On `main` it requires every target bui …[truncated]

### L3-927c87b347  (L3, 2026-09-26, sha 927c87b3472a, PR #56723)
TITLE: [PCP][DCP] Support DCP target model with non-DCP Dspark (#56723)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+7/-8); vllm/v1/attention/backends/flash_attn.py (+7/-10); vllm/v1/attention/backends/flashinfer.py (+19/-25); vllm/v1/attention/backends/utils.py (+22/-2); tests/v1/attention/test_attention_backends.py (+1/-2); tests/v1/attention/test_group_head_counts.py (+1/-0); tests/v1/core/test_kv_cache_utils.py (+143/-12); tests/v1/core/test_prefix_caching.py (+12/-7); tests/v1/e2e/spec_decode/draft_model/test_draft_model.py (+33/-0); tests/v1/kv_connector/unit/test_decode_bench_connector.py (+2/-1); (+22 more)
LABELS: bug, speculative-decoding, kv-connector, nvidia, mrv2, dflash, kv-cache-manager
BODY: Run dense DSpark/DFlash drafts at DCP=1 alongside a DCP target, so the draft backend does not need DCP support. ⏎  ⏎ `KVCacheSpec.dcp_sharded` keeps draft allocation, slot mapping, prefix caching, and offloading independent of target sharding. Mixed MLA/dense caches use block-outer packing; FlashInfer gets the resolved layout from attention metadata. Existing MLA-draft PCP restrictions remain unchanged. ⏎  ⏎ Validation: ⏎  ⏎ - CPU regressions: **456 passed,  …[truncated]

### L3-a9eafde59c  (L3, 2026-09-26, sha a9eafde59cbd, PR #58634)
TITLE: [Perf][DSv4.1] Fuse small-batch WO-A with inverse RoPE and MXFP8 quant on SM100/SM103 (#58634)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v41/nvidia/flashmla.py (+7/-19); tests/kernels/test_fused_inv_rope_fp8_quant.py (+48/-0); vllm/models/deepseek_v41/attention.py (+2/-2); vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py (+8/-34); vllm/models/deepseek_v41/nvidia/ops/fused_wo_a.py (+599/-0); vllm/models/deepseek_v41/nvidia/ops/o_proj.py (+95/-0)
LABELS: ready, deepseek, nvidia, quantization, verified, DSv4.1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ On SM100/SM103, the DSv4.1 O-projection runs four kernels per layer. At decode batch sizes the first three are latency-bound and together take 11-12 µs on GB200 (DSv4.1-Flash, TP=4): ⏎  ⏎ ``` ⏎ before: attn_out ─▶ [Triton] inverse RoPE + per-32 FP8 quant ─▶ [DeepGEMM] fp8_einsum wo_a ─BF16─▶ [FlashInfer] MXFP8 quantize ─▶ [FlashInfer CuTe-DSL] wo_b ⏎ after:  attn_out ─▶ [FusedWoAKernel] inverse RoPE + quant + wo_a + MXFP8 quant ──────── …[truncated]

### L3-0c87a197b8  (L3, 2026-09-26, sha 0c87a197b856, PR #56377)
TITLE: [Bugfix] Disable sequence parallelism / async TP under batch invariance and add a TP regression test (#56377)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/features/batch_invariance.md (+1/-1); vllm/config/vllm.py (+20/-0)
LABELS: bug, documentation, ready
ISSUES: #56370 [Bug]: Batch invariance is broken when sequence parallelism / async TP is enabled (`VLLM_BATCH_INVARIANT=1` + `pass_config.enable_sp`)
BODY: ## Purpose ⏎  ⏎ Fixes #56370. ⏎  ⏎ `VLLM_BATCH_INVARIANT=1` did not prevent `pass_config.enable_sp` / `fuse_gemm_comms` from being enabled. The `SequenceParallelismPass` rewrites `all_reduce + rms_norm` into `reduce_scatter + rms_norm + all_gather` for compile ranges above `sp_min_token_num`; the reduce-scatter goes through the plain PyNccl path whose reduction order depends on which rank owns a token's chunk, i.e. on the batch composition. Measured  …[truncated]
