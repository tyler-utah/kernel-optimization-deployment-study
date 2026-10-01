### L3-6472131298  (L3, 2026-07-15, sha 6472131298e7, PR #48379)
TITLE: [Bugfix] Set kv_quant_mode on the generic MLA KV-cache spec (#48379)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+2/-0)
LABELS: bug, ready
ISSUES: #48378 [Bug]: DeepSeek-V3.2 / GLM DSA with --kv-cache-dtype fp8_ds_mla crashes at engine init: KV-cache reshape sizes view by head_size (576) while pages are 656B/token
BODY: FIX #48378 ⏎  ⏎ `MLAAttention.get_kv_cache_spec` left `kv_quant_mode` at its `KVQuantMode.NONE` default, so the KV-cache reshape treated every MLA layer as skipped from quantization and sized the view by `head_size` (576 elements/token) while the raw tensor is allocated by the fp8_ds_mla byte layout (656 B/token): ⏎  ⏎ ``` ⏎ RuntimeError: shape '[512, 64, 576]' is invalid for input of size 21495808 ⏎ ``` ⏎  ⏎ Any DeepSeek-V3.2 / GLM DSA model with `--kv-cache-dt …[truncated]

### L3-7aab6e2684  (L3, 2026-07-15, sha 7aab6e268489, PR #48688)
TITLE: [ROCm][Bugfix] Enable the fp32 head_dtype torch.mm fast path on ROCm (#48688)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/sample/test_head_dtype.py (+31/-0); vllm/model_executor/layers/logits_processor.py (+5/-4)
LABELS: bug, rocm, ready, v1, verified
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ #48390 added a `torch.mm(..., out_dtype=torch.float32)` fast path so the fp32 ⏎ `head_dtype` lm_head projection avoids materializing a full fp32 copy of the ⏎ lm_head weight on every step. That fast path was gated on ⏎ `current_platform.is_cuda()`, so ROCm always fell back to the cast path ⏎ (`F.linear(hidden_states.to(fp32), lm_head.weight.to(fp32), ...)`), which ⏎ allocates and writes a full fp32 copy of the (large) lm_head weight matrix ⏎ ever …[truncated]

### L3-1d99f0f421  (L3, 2026-07-15, sha 1d99f0f42152, PR #47770)
TITLE: [ROCm][BugFix] Triton W4A16 handling for GPTQ/AutoGPTQ qzeros layout  (#47770)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/quantization/test_triton_w4a16.py (+157/-0); vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py (+25/-5)
LABELS: bug, rocm, ready
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:1d3a8b9e22 (fix PR 48998)
BODY: ## Purpose ⏎  ⏎ Fix ROCm Triton W4A16 handling for GPTQ/AutoGPTQ `qzeros` layout specifically issue #47159. ⏎ This addresses two related issues: ⏎ - Symmetric/biased GPTQ models may still have `qzeros` tensors registered by the loader, but the Triton kernel should not use them. For biased types such as `uint4b8`, `apply_weights()` now passes `qzeros=None`, so the kernel uses scalar `zp_bias` and runs with `HAS_ZP=False`. ⏎ - `TritonW4A16LinearKernel.p …[truncated]

### L3-ecf4aa5ce2  (L3, 2026-07-15, sha ecf4aa5ce2cc, PR #48167)
TITLE: [Bugfix] Fix FlashInfer non-causal draft attention (DFlash/DSpark) on Blackwell (#48167)
SOURCES: path_core, path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+6/-1); vllm/v1/attention/backends/flashinfer.py (+2/-1); vllm/v1/worker/gpu/model_runner.py (+4/-3); vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (+35/-10); vllm/v1/worker/gpu/spec_decode/dflash/utils.py (+5/-11); vllm/v1/worker/gpu/spec_decode/dspark/speculator.py (+0/-2); vllm/v1/worker/gpu/spec_decode/dspark/utils.py (+2/-3); vllm/v1/worker/gpu/spec_decode/speculator.py (+8/-2); tests/v1/spec_decode/test_dflash_causality.py (+55/-0); tools/pre_commit/generate_attention_backend_docs.py (+3/-1); (+2 more)
LABELS: bug, speculative-decoding, ready, v1, qwen, nvidia
BODY: ## Purpose ⏎  ⏎ DFlash/DSpark drafts on Blackwell had near-zero acceptance or crashed with an illegal memory access when the drafter used FlashInfer; Hopper was fine. ⏎  ⏎ Root cause: non-causal attention skips trtllm-gen and runs FlashInfer's prefill wrapper, which is only CUDA-graph-replayable when constructed with `use_cuda_graph=True` and persistent buffers — vLLM does not use that mode for draft attention, so replaying a captured `run()` after `plan …[truncated]

### L3-2bd8957627  (L3, 2026-07-15, sha 2bd8957627bf, PR #46880)
TITLE: [Bugfix][NVFP4 MoE] Pad gated intermediate to 64 for FlashInfer TRT-LLM shuffle (M%128) (#46880)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+7/-1)
LABELS: bug, ready, nvidia
ISSUES: #46879 [Bug] `nvidia/Gemma-4-26B-A4B-NVFP4` fails `assert M % 128 == 0` on FlashInfer TRT-LLM NVFP4 MoE at `-tp 4` (Blackwell/B200)
BODY: Closes #46879. ⏎  ⏎ ## Summary ⏎  ⏎ FlashInfer's TRT-LLM NVFP4 MoE weight prep shuffles block-scale rows via ⏎ get_shuffle_matrix_sf_a_row_indices, which asserts the gate/up row dim is a ⏎ multiple of 128 (epilogue_tile_m). align_fp4_moe_weights_for_fi pads the ⏎ intermediate to min_alignment, and the gate/up dim is ⏎ up_mult*padded_intermediate (up_mult=2 when gated). The caller passes ⏎ min_alignment = 16 if is_gated else 128; 16 is the NVFP4 scale-bloc …[truncated]

### L3-3c1bc1fc0d  (L3, 2026-07-15, sha 3c1bc1fc0dd5, PR #48519)
TITLE: [ROCm][Perf] Optimize sparse attention prefill kernel for DeepSeek-V4 (#48519)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+10/-3)
LABELS: rocm, ready, v1, deepseek
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ This PR optimizes DeepSeek-V4's dedicated sparse attention ragged prefill kernel in ROCm. The optimization speeds up the kernel by ~2x, and in E2E benchmark reduces TTFT by ~10%. ⏎  ⏎ In the current upstream ROCm path, this kernel has some slight inefficiencies that can be optimized. Specifically, this PR addresses three points of optimization. ⏎  ⏎ - `num_warps` setting. In DSv4-Pro's architecture, the per-workgroup tile under TP8 is `[B …[truncated]

### L3-66b6c684ab  (L3, 2026-07-15, sha 66b6c684ab5a, PR #48125)
TITLE: [PD][Bugfix] Fix validation of cache shape for attn backends enforcing different `kernel_block_size` (#48125)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+6/-1)
LABELS: bug, ready, kv-connector
BODY: I have started looking into heterogeneous attn backend selection here https://github.com/vllm-project/vllm/pull/48012 (an extension of hybrid ssm setup if you want). ⏎ This exposed a latent bug in the **validation of cache shapes** that we have in the registration phase. ⏎  ⏎ Say we have 2 attn backends enforcing different block sizes: ⏎ - user block_size=128 ⏎ - FA kernel_block_size=128 ⏎ - SWA kernel_block_size=64 ⏎  ⏎ with the setup above, we compute  …[truncated]

### L3-ba47bb5be1  (L3, 2026-07-15, sha ba47bb5be1bf, PR #47669)
TITLE: Bump flashinfer version to 0.6.14 (#47669)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); docker/versions.json (+1/-1); requirements/cuda.txt (+5/-2); setup.py (+5/-0); vllm/utils/jit_monitor.py (+21/-14); tests/test_jit_monitor.py (+20/-0)
LABELS: ready, ci/build, nvidia, ready-run-all-tests, verified
BODY: ## Purpose ⏎  ⏎ Pick up flashinfer-ai/flashinfer#3615, which fixes the multi-CTA radix top-k sampler stream hang (flashinfer-ai/flashinfer#3610): the kernel's epilogue resets the software barrier's arrival counter with no sync against peer CTAs still polling it, so a peer can spin forever and permanently wedge the stream (GPU pinned at 100% util / low power, no Xid, engine frozen). 0.6.14 is the first release containing the fix; current pin (0.6.13)  …[truncated]

### L3-eb33ff34dd  (L3, 2026-07-15, sha eb33ff34dd65, PR #47718)
TITLE: [ROCm][Perf] DSv4 two-stage compressor kernel for HCA prefill (#47718)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_compressor_kv_cache.py (+130/-0); vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py (+355/-0); vllm/models/deepseek_v4/compressor.py (+43/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ This PR adds a two-stage Triton implementation to the DeepSeek-V4 compressor for the prefill path in head_dim=512, cr>=128 to improve TTFT. ⏎  ⏎ On upstream ROCm, the single pass compressor-norm-rope-cache fusion is dispatched one workgroup per token. In HCA where compress ratio is 128, only workgroups that live on the CR boundary (positions % 128 == 0) execute on 128x512 grids, and the others early exit. During perfill, the number of b …[truncated]

### L3-6570c9800c  (L3, 2026-07-15, sha 6570c9800cda, PR #48799)
TITLE: [Model] Add Inkling model support [1/N] (#48799)
SOURCES: dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.flash_attn.fa4_cutedsl
FILES: CMakeLists.txt (+1/-0); cmake/external_projects/tml_fa4.cmake (+50/-0); requirements/cuda.txt (+1/-1); setup.py (+15/-0); .gitignore (+3/-0); rust/Cargo.lock (+2/-1); rust/Cargo.toml (+1/-1); rust/src/chat/src/backend/hf.rs (+2/-0); rust/src/chat/src/lib.rs (+3/-3); rust/src/chat/src/multimodal/audio.rs (+112/-0); (+85 more)
LABELS: new-model, ready, ci/build, tool-calling, nvidia, rust
BODY: ## Purpose ⏎  ⏎ Adds the core vLLM model and frontend support for `thinkingmachines/Inkling`, including the public `thinkingmachines/Inkling-NVFP4` checkpoint. ⏎  ⏎ This is the first independently mergeable slice of #48768. It keeps the model implementation under `vllm/models/inkling` and includes the tokenizer, renderer, reasoning/tool parsers, model registry entries, tests, and the Python-only FA4 dependency needed to serve it. The remaining umbrella w …[truncated]

### L3-8bfd683901  (L3, 2026-07-15, sha 8bfd6839016f, PR #48787)
TITLE: [Spec Decode] Add kv_cache_dtype to speculative_config to control separately from target (#48787)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/speculative.py (+4/-0); vllm/engine/arg_utils.py (+4/-0); vllm/v1/spec_decode/llm_base_proposer.py (+9/-0); vllm/v1/worker/gpu/spec_decode/dflash/utils.py (+8/-0); vllm/v1/worker/gpu/spec_decode/dspark/utils.py (+8/-0); vllm/v1/worker/gpu/spec_decode/eagle/utils.py (+9/-1)
LABELS: speculative-decoding, ready, v1
BODY: ## Purpose ⏎  ⏎ Currently running `vllm serve RedHatAI/GLM-5.2-NVFP4-FP8 --tensor-parallel-size 4 --spec-model GLM-5.2-speculator.dspark --spec-method dspark --spec-tokens 7 --kv-cache-dtype fp8` will fail on Blackwell since the global `--kv-cache-dtype fp8` will apply to both the GLM 5.2 target and the DSpark drafter. We don't currently have an attention backend that simultaneously supports non-causal attention and FP8 kv cache, so the drafter fai …[truncated]

### L3-0becb7486b  (L3, 2026-07-16, sha 0becb7486b3e, PR #47309)
TITLE: [BugFix][MLA] Support kv_cache_dtype_skip_layers for MLA attention (#47309)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+14/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ Fix two gaps in `MLAAttention` that prevent `--kv-cache-dtype-skip-layers` from ⏎ working with MLA models (DeepSeek-V2/V3/V4, GLM and others). ⏎  ⏎ ### Problem 1: `MLAAttention.__init__` ignores `kv_cache_dtype_skip_layers` ⏎  ⏎ The `--kv-cache-dtype-skip-layers` flag (introduced in #33695) allows skipping ⏎ FP8 KV cache quantization for specified layers. However, it was only wired up in ⏎ `Attention.__init__` — `MLAAttention.__init__` **n …[truncated]

### L3-915dffaa5f  (L3, 2026-07-16, sha 915dffaa5f93, PR #47060)
TITLE: [Attention] Mirror Triton KV dtype checks in MLA (#47060)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/triton_mla.py (+26/-0)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Apply the same native BF16/FP8 architecture validation used by the Triton Attention implementation to Triton MLA. ⏎  ⏎ This is the Triton MLA counterpart to architecture validation in PR #43330 (https://github.com/vllm-project/vllm/pull/43330), “Allow native KV cache dtype in Triton cache update.” ⏎  ⏎ The patch prevents unsupported KV-cache dtypes from reaching the Triton MLA cache-update path: ⏎  ⏎ - Native FP8 requires SM89+. ⏎ - BF16 K …[truncated]

### L3-5de1add806  (L3, 2026-07-16, sha 5de1add806ce, PR #48451)
TITLE: [feature]Add int4 quantization support for emulation moe backend (#48451)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/quantization/test_int4_emulation_moe.py (+1189/-0); vllm/model_executor/layers/fused_moe/config.py (+12/-0); vllm/model_executor/layers/fused_moe/experts/int4_emulation_moe.py (+129/-0); vllm/model_executor/layers/fused_moe/oracle/int_wna16.py (+281/-2); vllm/model_executor/layers/quantization/auto_awq.py (+3/-0); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py (+17/-0); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a16_flydsl.py (+2/-1); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py (+2/-1); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py (+18/-6)
LABELS: rocm, ready
BODY: ## Purpose ⏎ This PR is to enable int4 quantization support for emulation moe backend. ROCm platform doesn't have a   ⏎ moe backend that supports asymmetric int4 quantization, so this is necessary to support int4 quantized models like cyankiwi/MiniMax-M3-AWQ-INT4 for now (In the future this support will be added to flydsl/AITER library). ⏎  ⏎ ## Test Plan ⏎ Running cyankiwi/MiniMax-M3-AWQ-INT4 on ROCm platform MI300/350/355 to verify the accuracy. ⏎  ⏎  …[truncated]

### L3-251f7e478e  (L3, 2026-07-16, sha 251f7e478e8e, PR #48822)
TITLE: [Model] Add PW CUDA graph support for Inkling [2/N] (#48822)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/inkling/test_contract_validation.py (+5/-5); tests/v1/cudagraph/test_breakable_cudagraph.py (+50/-0); vllm/config/vllm.py (+2/-0); vllm/models/inkling/nvidia/attention.py (+2/-0); vllm/models/inkling/nvidia/ops/sconv.py (+4/-4); vllm/models/inkling/nvidia/sconv_swa_attn.py (+1/-3); vllm/models/inkling/nvidia/short_conv.py (+2/-2); vllm/v1/worker/gpu/cudagraph_utils.py (+23/-16); vllm/v1/worker/gpu/model_states/default.py (+5/-1); vllm/v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py (+4/-1)
LABELS: ready, v1, nvidia
BODY: ## Summary ⏎  ⏎ This is the second independently mergeable slice carved out of #48768. It adds breakable (piecewise) CUDA graph support for Inkling on Model Runner V2: ⏎  ⏎ - automatically enables breakable CUDA graphs for `InklingForCausalLM` and `InklingForConditionalGeneration` ⏎ - marks the Inkling attention boundary as eager and makes short-convolution metadata safe for uniform batched execution ⏎ - preserves padded request counts and static attention m …[truncated]

### L3-f61163e6c7  (L3, 2026-07-16, sha f61163e6c736, PR #48858)
TITLE: [Model] Add Hopper FA4 relative attention for Inkling (#48858)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/inkling/test_fa4_rel_attention.py (+47/-0); vllm/models/inkling/nvidia/ops/fa4_rel_attention.py (+62/-5)
LABELS: ready
BODY: ## Summary ⏎  ⏎ Inkling's current relative-attention path uses the tml-fa4 sheared-bias interface, which is a Blackwell-specific path. On Hopper, that reaches incompatible SM90 call signatures and split-KV constraints. ⏎  ⏎ This change: ⏎ - uses regular FA4 `score_mod` plus `aux_tensors` for Inkling relative bias on Hopper (SM90); ⏎ - keeps the optimized tml-fa4 sheared-bias path on Blackwell (SM100/SM110); ⏎ - keeps `num_splits=1` on Hopper; ⏎ - adds architectu …[truncated]

### L3-df8a0900df  (L3, 2026-07-16, sha df8a0900df22, PR #48741)
TITLE: [BugFix] Don't apply weight in batch-invariant RMSNorm when has_weight=False (#48741)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/batch_invariant.py (+17/-12); vllm/model_executor/layers/layernorm.py (+4/-1)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ Fix silent output corruption for models using `RMSNorm(has_weight=False)` (Gemma4, Gemma3n, Llama4) when `VLLM_BATCH_INVARIANT=1` is combined with sleep mode - e.g. in standard RL rollout configuration. ⏎  ⏎ `RMSNorm(has_weight=False)` keeps its identity weight as a plain tensor, not an `nn.Parameter` - so `sleep(level=2)` discards it and weight re-sync brings it back zeroed. `VLLM_BATCH_INVARIANT` branch still passed `self.weight.dat …[truncated]

### L3-b8168e33e0  (L3, 2026-07-16, sha b8168e33e000, PR #46275)
TITLE: [ROCm][Perf][DSV4] Enable split sparse decode on gfx942 (#46275)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+1/-1); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+44/-41)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Enable the existing split partial/reduce Triton sparse decode path for DeepSeek-V4 sparse MLA decode on AMD gfx942. ⏎  ⏎ gfx950 already uses this path. gfx942 currently falls back to the monolithic `_sparse_attn_decode_ragged_kernel`, even though the split path runs correctly on gfx942. This PR extends the tuned-architecture guard to include gfx942 and updates the ROCm DeepSeek-V4 sparse attention tests so the split decode path is cov …[truncated]

### L3-fb5ec0dc9e  (L3, 2026-07-16, sha fb5ec0dc9edf, PR #48869)
TITLE: [Model] Add Inkling MTP=1 support [3/N] (#48869)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/config/test_speculative_draft_hf_overrides.py (+28/-0); tests/models/inkling/test_contract_validation.py (+5/-0); tests/models/inkling/test_mtp_input_fusion.py (+106/-0); tests/models/registry.py (+7/-0); vllm/config/speculative.py (+30/-1); vllm/model_executor/models/registry.py (+1/-0); vllm/models/inkling/__init__.py (+6/-0); vllm/models/inkling/configs.py (+6/-0); vllm/models/inkling/nvidia/model.py (+14/-5); vllm/models/inkling/nvidia/mtp.py (+408/-0); (+1 more)
LABELS: new-model, ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Why ⏎  ⏎ This is the third Inkling enablement slice after #48799 and #48822. It adds only the first MTP draft layer so Inkling can use one speculative token with CUDA graphs. ⏎  ⏎ Duplicate-work searches for "Inkling MTP" and "Inkling speculative decoding" found only #48768. This PR intentionally supersedes only the MTP=1 portion of that draft: it excludes LoRA, MTP>1 scheduler/KV-cache work, and unrelated frontend changes. #48858 is separate Hopper r …[truncated]

### L3-7cd1d57b74  (L3, 2026-07-16, sha 7cd1d57b749f, PR #47442)
TITLE: [CI/Build][Docker] Bump nvidia-cutlass-dsl to 4.6.0 and drop packaging workarounds (#47442)
SOURCES: path_core, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_build, L3.flash_attn.fa4_cutedsl
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); docker/Dockerfile (+0/-35); requirements/cuda.txt (+2/-2); requirements/test/cuda.txt (+1/-1); vllm/vllm_flash_attn/flash_attn_interface.py (+1/-1); tests/entrypoints/pooling/scoring/test_cross_encoder_online_vision.py (+1/-0); vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+2/-80)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ In `nvidia-cutlass-dsl` wheel package there was a bug in using the same paths in filesystem by two different sub-wheels that resulted in race conditions during `uv pip install` process in vLLM. Original bug reports https://github.com/NVIDIA/cutlass/issues/3259, https://github.com/NVIDIA/cutlass/issues/3170 in cutlass repository. To overcome this issue vLLM used some temporary workarounds https://github.com/vllm-project/vllm/pull/434 …[truncated]

### L3-d803b44dbe  (L3, 2026-07-16, sha d803b44dbefd, PR #47559)
TITLE: [NIXL] Bump nixl to 1.3.1 (#47559)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/kv_connectors.txt (+1/-1)
LABELS: ready, ci/build, kv-connector
BODY: ## Purpose ⏎  ⏎ Bump NIXL version to 1.3.1. ⏎  ⏎ ## Test Plan ⏎  ⏎ `requirements/kv_connectors.txt` is in `run_all_patterns`, so editing it triggers ⏎ the full NixlConnector P/D sweep; the `*-nixlconnector-pd-accuracy-*` jobs exercise ⏎ the NixlConnector against the new nixl wheel. ⏎  ⏎ ## Test Result ⏎  ⏎ TBD — awaiting the `*-nixlconnector-pd-accuracy-*` sweep on this branch. ⏎  ⏎ --- ⏎ [details omitted]

### L3-c95c663049  (L3, 2026-07-16, sha c95c6630499f, PR #48538)
TITLE: [Quant] Add `nvfp4_per_token` online MoE quantization (#48538)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/quantization/test_online.py (+36/-0); vllm/config/quantization.py (+6/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+63/-5); vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (+3/-0); vllm/model_executor/layers/quantization/__init__.py (+1/-0); vllm/model_executor/layers/quantization/online/base.py (+5/-0); vllm/model_executor/layers/quantization/online/nvfp4.py (+171/-0)
LABELS: ready, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ Adds a new online-quantization shorthand, `--quantization nvfp4_per_token`, that quantizes MoE expert weights to NVFP4 at load time from a bf16 checkpoint and runs them through the FlashInfer TRTLLM fused-MoE kernel with **dynamic per-token global activation scales**. Similar in spirit to https://github.com/sgl-project/sglang/pull/26083, but fits right in vLLM's existing online-quant framework, so it's just a new quant key + online  …[truncated]

### L3-85e296950c  (L3, 2026-07-16, sha 85e296950cff, PR #47973)
TITLE: BF16x3 router GEMM (#47973)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_bf16x3_router_gemm_cutedsl.py (+43/-0); vllm/config/kernel.py (+3/-0); vllm/engine/arg_utils.py (+7/-0); vllm/model_executor/layers/fused_moe/router/bf16x3_router_gemm_cutedsl.py (+449/-0); vllm/model_executor/layers/fused_moe/router/gate_linear.py (+53/-10)
LABELS: ready
BODY: This PR is inspired by CuBLAS BF16x9: https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/ ⏎  ⏎ ## Background ⏎  ⏎ All FP32 numbers, including subnormals, can be represented exactly by a (weighted) sum of 3 BF16 numbers. This is because FP32 has 24 mantissa bits (including the implicit `1.xxx` for normal numbers) and BF16 has 8 mantissa bits. ⏎  ⏎ A simple procedure to get this decompsition ⏎  ⏎ ``` …[truncated]

### L3-d08eebad16  (L3, 2026-07-16, sha d08eebad162b, PR #47156)
TITLE: [Perf][MoE] Write FlashInfer combine into final output (#47156)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: tests/distributed/test_mnnvl_alltoall.py (+30/-0); vllm/distributed/device_communicators/all2all.py (+29/-0); vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py (+2/-2)
LABELS: ready, nvidia, verified
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Pass vLLM's preallocated MoE output tensor to FlashInfer one-sided combine, ⏎ removing the temporary combine output and subsequent `output.copy_()`. ⏎  ⏎ Requires FlashInfer PR: https://github.com/flashinfer-ai/flashinfer/pull/3776 ⏎  ⏎ ## Performance ⏎  ⏎ Nemotron Ultra 550B A55B NVFP4, one GB200 node (4 gpus), TP1/DP4/EP4, FlashInfer ⏎ one-sided, ISL/OSL 50000/2048, concurrency 16: ⏎  ⏎ | | Baseline | Direct output | Delta | ⏎ | --- | ---: | …[truncated]

### L3-2cab53ddee  (L3, 2026-07-16, sha 2cab53ddeea3, PR #42749)
TITLE: [Model][Hardware][AMD]: Part 1/2 -> Enable e2e QK Norm + RoPE + KV Cache runtime fusion for Qwen3-30B-A3B on ROCM_AITER_FA, and ROCM_AITER_UNIFIED_ATTN (#42749)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/compilation/passes/fusion/qk_norm_rope_fusion.py (+22/-1); vllm/compilation/passes/fusion/qk_norm_rope_kvcache_fusion.py (+492/-0); vllm/compilation/passes/pass_manager.py (+6/-0); vllm/config/compilation.py (+36/-2); vllm/config/vllm.py (+32/-0); vllm/model_executor/layers/attention/attention.py (+5/-1); vllm/model_executor/layers/attention/mla_attention.py (+2/-0); vllm/v1/attention/backend.py (+33/-0); vllm/v1/attention/backends/rocm_aiter_fa.py (+41/-0); vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+44/-0); (+5 more)
LABELS: rocm, ready, v1, qwen
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ This is **Part 1** of a stacked PR series that splits the original ⏎ `jhu96/optimize-qwen30b` branch (PR:  https://github.com/vllm-project/vllm/pull/39527) into two reviewable PRs. ⏎  ⏎ - Here, we enabled runtime inductor fusion substitution for both the ROCM_AITER_FA and ROCM_AITER_UNIFIED_ATTN attention backends. As for ROCM_ATTN (coming in part 2), it requires kernel-level changes, thus deferring to another PR. ⏎ - Enable e2e runtime …[truncated]

### L3-ab0a20d151  (L3, 2026-07-16, sha ab0a20d15129, PR #46396)
TITLE: [Docs] Add Phi-3.5-mini-instruct to batch invariance tested models (#46396)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/features/batch_invariance.md (+1/-0)
LABELS: documentation, ready
BODY: ## Description ⏎  ⏎ Adds `microsoft/Phi-3.5-mini-instruct` to the list of tested models for batch invariance. ⏎  ⏎ ## Test Results ⏎  ⏎ **Model:** microsoft/Phi-3.5-mini-instruct (3.8B) ⏎  ⏎ **Results:** ⏎ ``` ⏎ Trial 1: MATCH ⏎ Trial 2: MATCH ⏎ Trial 3: MATCH ⏎ Trial 4: MATCH ⏎ Trial 5: MATCH ⏎  ⏎ [determinism] total=5, passed=5, failed=0, max_batch_size=8 ⏎  ⏎ ✓ TEST PASSED - Batch invariance verified! ⏎ ``` ⏎  ⏎ **Test Configuration:** ⏎ - Hardware: 4x NVIDIA A10G GPUs (23GB each) ⏎ - Ten …[truncated]

### L3-fe784ff22e  (L3, 2026-07-16, sha fe784ff22e63, PR #48582)
TITLE: [M3] Improve indexer for long-context decode (sm100) (#48582)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_minimax_m3.py (+118/-8); vllm/cute_utils/__init__.py (+48/-13); vllm/cute_utils/cvt.py (+26/-0); vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/kernel_kkt_inv_uw.py (+63/-36); vllm/models/minimax_m3/nvidia/indexer_msa.py (+13/-6); vllm/models/minimax_m3/nvidia/ops/__init__.py (+5/-0); vllm/models/minimax_m3/nvidia/ops/index_decode_score.py (+483/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ This PR adds a CuteDSL indexer decode kernel for Minimax M3, optimized for long-context. ⏎ - TMA + `mma.sync` design, instead of tcgen05 as I find saturating memory bandwidth with high occupancy (+ hardware scheduling) is easier than tcgen05 software pipelining, especially when there is context imbalance (different requests have different context lengths) ⏎ - BF16 and FP8 Indexer cache support ⏎ - Speculative decoding support. However, …[truncated]

### L3-f17be06fbe  (L3, 2026-07-16, sha f17be06fbe42, PR #48143)
TITLE: [Perf] Optimize `clamp` to `clamp_` (#48143)
SOURCES: path_core
ARTIFACT_HINTS: L3.rocm.aiter_fa, L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/mla_attention.py (+4/-4); vllm/v1/attention/backends/rocm_aiter_fa.py (+2/-3); vllm/v1/attention/backends/utils.py (+2/-4); vllm/models/deepseek_v4/sparse_mla.py (+1/-1); vllm/v1/attention/backends/mamba_attn.py (+24/-12); vllm/v1/spec_decode/ngram_proposer_gpu.py (+12/-7); vllm/v1/spec_decode/step3p5.py (+3/-1)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready, ci/build
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Optimize `clamp` to `clamp_` to reduce additional memory allocation.

### L3-626c90b2d5  (L3, 2026-07-16, sha 626c90b2d51e, PR #48500)
TITLE: [Refactor] Move fla to third party (#48500)
SOURCES: path_core, corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: vllm/third_party/flash_linear_attention/ops/__init__.py (+0/-0); vllm/third_party/flash_linear_attention/ops/chunk.py (+0/-0); vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py (+0/-0); vllm/third_party/flash_linear_attention/ops/chunk_o.py (+0/-0); vllm/third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py (+0/-0); vllm/third_party/flash_linear_attention/ops/cumsum.py (+0/-0); vllm/third_party/flash_linear_attention/ops/fused_recurrent.py (+0/-0); vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py (+0/-0); vllm/third_party/flash_linear_attention/ops/index.py (+0/-0); vllm/third_party/flash_linear_attention/ops/kda.py (+0/-0); (+32 more)
LABELS: ready, ci/build, v1
BODY: ## Purpose ⏎  ⏎ An alternative for https://github.com/vllm-project/vllm/pull/48424 ⏎  ⏎ CC @ZJY0516 @hmellor

### L3-67f9046e4a  (L3, 2026-07-16, sha 67f9046e4a12, PR #48642)
TITLE: [Bugfix] Sparse MLA: enable fp8_ds_mla dense prefill (#48642)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.mla.common_v1, L3.mla.flashmla_sparse
FILES: vllm/_custom_ops.py (+3/-3); vllm/model_executor/layers/attention/mla_attention.py (+25/-6); vllm/model_executor/layers/attention/sparse_mla_attention.py (+3/-1); vllm/v1/attention/backends/mla/flashmla_sparse.py (+40/-41); vllm/v1/attention/backends/mla/sparse_utils.py (+5/-0); benchmarks/kernels/bench_cp_gather_fp8.py (+3/-4); csrc/cache.h (+1/-2); csrc/libtorch_stable/cache_kernels.cu (+19/-9); csrc/libtorch_stable/ops.h (+2/-2); csrc/libtorch_stable/torch_bindings.cpp (+2/-2); (+3 more)
LABELS: bug, performance, ready, v1
DEEP_STUDY: deep-study correctness case vllm:67f9046e4a: class=memory_safety_oob; symptom=illegal_memory_access; introducing=unknown
BODY: ## Purpose ⏎  ⏎ PR #47327 added dense-MHA prefill routing for sparse MLA, exposing two `fp8_ds_mla` mixed-batch bugs: ⏎  ⏎ - `req_id_per_token` could exceed the top-k tensor length and cause out-of-bounds writes. ⏎ - Dense MHA could not gather the packed 656-byte FP8 cache, and FlashMLA assumed its metadata covered the full batch. ⏎  ⏎ This PR bounds the converter input, adds arbitrary sequence starts to the packed FP8 gather/upconvert kernel, and supports dec …[truncated]

### L3-ce2aecc4dc  (L3, 2026-07-17, sha ce2aecc4dc42, PR #48417)
TITLE: [Performance] Use CuTe-DSL for FlashInfer MXFP4 quantization (#48417)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+3/-1); vllm/model_executor/kernels/linear/mxfp4/flashinfer.py (+3/-1)
LABELS: performance, ready, nvidia, quantization
ISSUES: #48205 [Performance]: The default MXFP4 quant backend of FlashInfer has slow performance, add options for backend "cute-dsl".
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Use the CuTe-DSL backend for activation quantization in ⏎ `FlashInferMxFp4LinearKernel`. ⏎  ⏎ The kernel is already selected only on SM100+ when FlashInfer CuTe-DSL is ⏎ available, and its following MXFP4 GEMM already uses the CuTe-DSL backend. ⏎ This change makes the activation quantization use the same backend while ⏎ keeping backend selection explicit at the kernel call site. ⏎  ⏎ Fixes #48205. ⏎  ⏎ ## Motivation ⏎  ⏎ The default CUDA backen …[truncated]

### L3-d5b1ec2684  (L3, 2026-07-17, sha d5b1ec268431, PR #48828)
TITLE: [XPU] allow forcing flash attn for mm_prefix (#48828)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/platforms/xpu.py (+12/-6)
LABELS: intel-gpu, ready
BODY: ## Purpose ⏎  ⏎ On XPU, multimodal prefix-LM models (e.g. Gemma4-26B-A4B-it) were unconditionally forced to Triton Attention (https://github.com/vllm-project/vllm/pull/47688) because XPU Flash Attention lacks FA4 and cannot apply the bidirectional mask for vision tokens. This prevented users from using Flash Attention even for text-only workloads for better performance where the mask is irrelevant. ⏎  ⏎ This PR allows user to force flash attention fo …[truncated]

### L3-f3e9497e92  (L3, 2026-07-17, sha f3e9497e921a, PR #48884)
TITLE: [Model] Add Inkling LoRA support [4/N] (#48884)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/inkling/test_moe_weight_layout.py (+25/-0); vllm/config/lora.py (+19/-1); vllm/engine/arg_utils.py (+6/-0); vllm/lora/layers/fused_moe.py (+51/-15); vllm/lora/lora_weights.py (+33/-0); vllm/lora/model_manager.py (+27/-25); vllm/lora/utils.py (+14/-2); vllm/model_executor/layers/fused_moe/experts/lora_context.py (+5/-0); vllm/model_executor/layers/fused_moe/experts/lora_experts_mixin.py (+19/-2); vllm/models/inkling/nvidia/logits_processor.py (+57/-1); (+2 more)
LABELS: ready
BODY: ## Summary ⏎  ⏎ - Mark Inkling as LoRA-capable and map adapter names for its packed QKVR projection, shared sink experts, and LM head. ⏎ - Add a LoRA-capable linear implementation of the shared sink experts while retaining the fused implementation when LoRA is disabled. ⏎ - Support Inkling's shared-outer MoE adapter layout, including expert-parallel slicing and stride-zero expansion of shared factors. ⏎ - Apply Inkling's muP width scaling to the complete b …[truncated]

### L3-efed8a1e83  (L3, 2026-07-17, sha efed8a1e8345, PR #48788)
TITLE: [ROCm][Perf][DSV4] Improve sparse decode reduction occupancy on gfx950 (#48788)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+2/-2)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ The split-K sparse decode reducer currently processes 16 heads per workgroup, ⏎ which keeps a `[16, COMB_DIM]` FP32 accumulator in each workgroup and limits ⏎ workgroup-level parallelism. ⏎  ⏎ This change processes one head per reducer workgroup. It lowers per-workgroup ⏎ accumulator/register pressure and exposes up to 16 times more independent ⏎ workgroups while preserving the per-head reduction math and ordering. ⏎  ⏎ On current upstream, this chan …[truncated]

### L3-b5433b6f50  (L3, 2026-07-17, sha b5433b6f5079, PR #48660)
TITLE: [Perf] Optimize dsv4 routing using specialized kernel, 2.94% E2E TPOT improvement (#48660)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu (+78/-0); tests/kernels/moe/test_topk_softplus_sqrt.py (+45/-0); vllm/model_executor/layers/fused_moe/router/dsv4_topk.py (+121/-0); vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py (+20/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Thanks the context from @zyongye ! ⏎  ⏎ ## Test ⏎  ⏎ `vllm serve deepseek-ai/DeepSeek-V4-Flash   -tp 4    -ep   --attention-backend FLASHINFER_MLA_SPARSE_DSV4   --kv-cache-dtype fp8   --tokenizer-mode deepseek_v4   --all2all-backend allgather_reducescatter   --port 8003` ⏎  ⏎ ### Acc ⏎  ⏎ `lm_eval --model  local-completions --model_args "base_url=http://127.0.0.1:8003/v1/completions,model=deepseek-ai/DeepSeek-V4-Flash,num_concurrent=1024" - …[truncated]

### L3-fcd2255d16  (L3, 2026-07-17, sha fcd2255d16bd, PR #37524)
TITLE: [Hardware][GPU] Profiler config additional to increase it scope and annotation details (#37524)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_profiler.py (+63/-1); vllm/config/profiler.py (+17/-0); vllm/v1/worker/gpu_model_runner.py (+62/-12); vllm/v1/worker/gpu_worker.py (+98/-13)
LABELS: ready, v1
BODY: This pull request introduces enhanced configurability and profiling capabilities for CUDA graph capture and trace annotation in the codebase. The main improvements include new options for capturing profiler traces during CUDA graph capture, the ability to generate more detailed trace annotations with KV length information, and the integration of these features into the profiling and model capture workflow.  ⏎  ⏎ **Profiler configuration and validat …[truncated]

### L3-17fdd42100  (L3, 2026-07-17, sha 17fdd4210090, PR #48251)
TITLE: [Bugfix][Attention] Preserve post-load tensors across weight reloads (#48251)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+3/-2); vllm/v1/attention/backends/flashinfer.py (+11/-2); tests/v1/attention/test_attention_backends.py (+27/-0); tests/v1/attention/test_mla_backends.py (+39/-0)
LABELS: bug, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ Preserve attention runtime tensors derived by `process_weights_after_loading()` across layerwise weight reloads. This fixes stale FlashInfer attention sinks and prevents standard MLA CUDA graphs from retaining obsolete `W_UV`/`W_UK_T` addresses. ⏎  ⏎ ## Root cause ⏎  ⏎ Two post-load paths replaced tensors instead of refreshing their existing storage: ⏎  ⏎ - Standard `MLAAttention` reassigned derived `W_UV` and `W_UK_T` views. CUDA graphs captured …[truncated]

### L3-5784507da4  (L3, 2026-07-17, sha 5784507da458, PR #48012)
TITLE: [Attention] Allow selecting a different attention backend per KV-cache group (#48012)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.dispatch.selector
FILES: vllm/config/attention.py (+38/-0); vllm/v1/attention/selector.py (+59/-3); tests/v1/attention/test_backend_per_kind.py (+67/-0); tests/v1/e2e/general/test_attention_backend_per_kind.py (+94/-0)
LABELS: rocm, ready, v1, cpu, nvidia
BODY: vLLM allows picking a **single global** attention backend for the whole model. ⏎ Models that split layers across multiple KV-cache groups are forced to specify one backend, though the runtime (`AttentionGroup`) already supports heterogeneous backends, so all the plumbing is already in place, we just lack the UX to let the user enforce a choice. ⏎ Most notably, hybrid SSM models is one example where different backends are always picked, but the choi …[truncated]

### L3-ce4bdcbda4  (L3, 2026-07-17, sha ce4bdcbda4f3, PR #48855)
TITLE: [Bugfix] Enable FlashAttention MLA prefill for Mistral Small 4 head dims (#48855)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/prefill/flash_attn.py (+7/-2); docs/design/attention_backends.md (+1/-1); tests/v1/attention/test_mla_prefill_selector.py (+2/-2)
LABELS: bug, documentation, ready, v1, mistral
BODY: ## Purpose ⏎  ⏎ Mistral Small 4 uses MLA head dimensions (qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=128). `FlashAttnPrefillBackend.supports_mla_dimensions` did not list these dims, so the model was excluded from the FlashAttention MLA prefill backend. This adds the config to the supported set (FA2/FA3/FA4), extends the selector unit test, and updates the attention backends doc. ⏎  ⏎ ## Test Plan ⏎  ⏎ - A/B probe on H200 (FA3) calling `FlashAttnPre …[truncated]

### L3-02c01f442b  (L3, 2026-07-17, sha 02c01f442b8b, PR #48990)
TITLE: [Model] Use standard ModelOpt config for Inkling NVFP4 (#48990)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/config/test_model_arch_config.py (+17/-0); tests/models/inkling/test_moe_weight_layout.py (+25/-0); vllm/models/inkling/nvfp4.py (+0/-64); vllm/models/inkling/nvidia/model.py (+2/-15); vllm/models/inkling/nvidia/moe.py (+5/-28); vllm/models/inkling/nvidia/mtp.py (+0/-1); vllm/transformers_utils/model_arch_config_convertor.py (+7/-2)
LABELS: ready, quantization
BODY: ## Purpose ⏎  ⏎ Use vLLM’s standard ModelOpt quantization infrastructure for Inkling NVFP4. ⏎  ⏎ - Detect legacy ModelOpt `hf_quant_config.json` files without `producer.name`. ⏎ - Pass the global quantization config to Inkling MoE. ⏎ - Remove the duplicate Inkling-specific NVFP4 config parser and per-layer config construction. ⏎ - Preserve Inkling’s checkpoint-specific expert tensor loading. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ Validated on 4xB300 gsm8k …[truncated]

### L3-425c4eafb0  (L3, 2026-07-17, sha 425c4eafb064, PR #48641)
TITLE: [Sampler] Stop upcasting logits to fp32 in apply_sampling_params (#48641)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/sample/test_topk_topp_sampler.py (+38/-0); vllm/v1/sample/ops/topk_topp_sampler.py (+6/-4); vllm/v1/sample/ops/topk_topp_triton.py (+19/-13); vllm/v1/worker/gpu/sample/logit_bias.py (+3/-1); vllm/v1/worker/gpu/sample/sampler.py (+19/-7)
LABELS: speculative-decoding, ready, v1, mrv2
DEEP_STUDY: deep-study: this PR was reverted by PR 49033 (confirmed_revert, reason=crash_or_hang) || deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ `apply_sampling_params` copies the logits into a new fp32 tensor before applying sampling params. Generation models emit bf16 logits by default (except in RL), so this copy is a pure upcast that costs `num_logits * vocab * 4` bytes. Under spec decode there are `num_reqs * (1 + num_spec_tokens)` logits instead of `num_reqs`, so the copy reaches multiple GiB. On Qwen3-8B + a dflash speculator it tries to allocate 4.12 GiB and OOMs dur …[truncated]

### L3-d4b4562917  (L3, 2026-07-17, sha d4b45629172d, PR #48942)
TITLE: [XPU] Bump vllm_xpu_kernels to v0.1.11.1 (#48942)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/xpu.txt (+1/-1)
LABELS: intel-gpu, ready, ci/build
BODY: ## Summary ⏎  ⏎ Bumps `vllm_xpu_kernels` pin from v0.1.11 to v0.1.11.1. ⏎  ⏎ v0.1.11 registers `moe_sum` with the old 2-arg XPU schema. Commit f7aadae5e5 added 2 extra args (`topk_ids`, `expert_map`) to the CUDA-side `moe_sum` op, but the XPU-side registration was never updated to match — every XPU MoE forward pass now crashes: ⏎  ⏎ ``` ⏎ RuntimeError: _moe_C::moe_sum() expected at most 2 argument(s) but received 4 argument(s) ⏎ ``` ⏎  ⏎ v0.1.11.1 includes [vllm-xpu …[truncated]

### L3-c7ce03bcbd  (L3, 2026-07-18, sha c7ce03bcbd38, PR #48988)
TITLE: [Bugfix] Bump tml-fa4 for cutlass-dsl 4.6 API compatibility (#48988)
SOURCES: subject_keyword, dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fa4_cutedsl
FILES: cmake/external_projects/tml_fa4.cmake (+1/-1)
LABELS: bug, ready, ci/build, nvidia
BODY: Bump tml-fa4 to include the cutlass-dsl 4.6 CuTe API migration (https://github.com/vllm-project/tml-fa4/pull/3) so Inkling Blackwell attention imports cleanly.

### L3-e94243893d  (L3, 2026-07-18, sha e94243893dd3, PR #46832)
TITLE: [ROCm][DSv3.2][Perf] Cap sparse MLA decode KV-splits with a work-per-split heuristic (#46832)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+21/-0)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: # [ROCm][DSv3.2][Perf] Cap sparse MLA decode KV-splits with a work-per-split heuristic ⏎  ⏎ The ROCm aiter sparse-MLA decode backend calls `get_mla_metadata_v1` without `max_split_per_batch`, so aiter defaults to `-1` and splits the decode reduction across every CU (304 on gfx942). The reduce only covers the selected tokens per row (`<= index_topk`), so for `topk=2048` that's ~7 tokens/split — the reduce kernel ends up dominated by launch and combi …[truncated]

### L3-b6ff8a2f50  (L3, 2026-07-19, sha b6ff8a2f509c, PR #46570)
TITLE: [Core] Add MRV2 virtual-batch PCP for MLA (#46570)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.mla.common_v1, L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: vllm/config/parallel.py (+37/-25); vllm/config/vllm.py (+3/-2); vllm/model_executor/layers/attention/mla_attention.py (+60/-12); vllm/model_executor/layers/attention/pcp.py (+92/-0); vllm/model_executor/layers/attention/sparse_mla_attention.py (+2/-2); vllm/v1/attention/backend.py (+14/-2); vllm/v1/attention/backends/mla/indexer.py (+19/-4); vllm/v1/attention/backends/utils.py (+3/-1); vllm/v1/attention/selector.py (+4/-1); vllm/v1/worker/block_table.py (+2/-4); (+29 more)
LABELS: documentation, rocm, ready, ci/build, v1, kv-connector
BODY: ## Summary ⏎  ⏎ Adds an MRV2 virtual-batch implementation for prefill context parallelism (PCP), initially targeting MLA and sparse MLA. ⏎  ⏎ - Add a stateful `PCPManager` that partitions the globally scheduled MRV2 `InputBatch` into rank-local virtual rows while preserving global request state for restore, logits, and postprocessing. ⏎ - Add MRV2 input-batch plumbing for PCP positions, sequence lengths, logits indices, and hidden-state restoration. ⏎  …[truncated]

### L3-1dcbbd9cac  (L3, 2026-07-19, sha 1dcbbd9caca3, PR #43024)
TITLE: [CI] Move compatible 1xL4 jobs to H200 35GB MIG (#43024)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/cuda.yaml (+2/-1); .buildkite/test_areas/entrypoints.yaml (+4/-0); .buildkite/test_areas/kernels.yaml (+2/-0); .buildkite/test_areas/misc.yaml (+2/-1); .buildkite/test_areas/model_executor.yaml (+1/-0); .buildkite/test_areas/models_language.yaml (+18/-3); .buildkite/test_areas/models_multimodal.yaml (+1/-0); .buildkite/test_areas/pytorch.yaml (+38/-2); .buildkite/test_areas/quantization.yaml (+15/-4); .buildkite/test_areas/rust_frontend.yaml (+1/-0); (+7 more)
LABELS: ready, ci/build, nvidia, rust
BODY: ## Summary ⏎  ⏎ - Give 17 compatible implicit single-GPU job groups not covered by #43164 H200 35GB MIG execution. Together, the two PRs give 20 of the 25 audited groups H200 execution; two migrated groups retain narrow L4 compatibility subsets for hardware-sensitive cases. ⏎ - Refresh the original broad migration branch against current `main` and reduce it from the stale 48-job snapshot to the exact compatible remainder. ⏎ - Keep Engine (1 GPU), Kernels …[truncated]

### L3-752bd10647  (L3, 2026-07-19, sha 752bd106477f, PR #49128)
TITLE: [ROCm][CI] Fix sparse MLA metadata sync fixture (#49128)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py (+2/-0)
LABELS: rocm, ready
BODY: The [failing AMD CI run](https://buildkite.com/vllm/amd-ci/builds/11018/list?jid=019f799a-f6b0-4579-9197-1c495ef92071&tab=output) raised `AttributeError: _num_compute_units` in the synthetic sparse-MLA builder fixture. `git bisect` from `c233d90aa826df072872df47b201450059be8e71` to `b6ff8a2f509cc7ac9c58176f5115a836aa1e08bd` identified `e94243893dd30256f58644ad4ecf779be757dff8`, introduced by regression PR [#46832](https://github.com/vllm-project/ …[truncated]

### L3-bd091079cb  (L3, 2026-07-20, sha bd091079cba0, PR #42569)
TITLE: [Attention] FlashAttention 4 SM100 FP8 kv cache support (#42569)
SOURCES: path_core, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_build, L3.flash_attn.fa_utils
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/v1/attention/backends/fa_utils.py (+23/-0); vllm/v1/attention/backends/flash_attn.py (+34/-8); docs/design/attention_backends.md (+2/-2); tests/models/quantization/test_fp8.py (+2/-8); tests/v1/attention/test_attention_backends.py (+64/-7); tests/v1/attention/utils.py (+2/-0); tools/pre_commit/generate_attention_backend_docs.py (+66/-7)
LABELS: documentation, performance, ready, ci/build, v1, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: Depends on https://github.com/vllm-project/flash-attention/pull/166, which resolves an accuracy issue at long contexts. ⏎  ⏎ ## Purpose ⏎ Implement FP8 KV cache support with FlashAttention 4 ⏎  ⏎ ## Test Plan ⏎ GSM8k on Nemotron 3 Nano ⏎  ⏎ ``` ⏎ vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \ ⏎   --attention-backend FLASH_ATTN \ ⏎   --kv-cache-dtype fp8 \ ⏎   --max-model-len 8192 ⏎ ``` ⏎ with ⏎ ``` ⏎ lm_eval run \ ⏎   --model local-chat-completions \ ⏎    …[truncated]

### L3-8ce53a616e  (L3, 2026-07-20, sha 8ce53a616ebf, PR #47574)
TITLE: [Bugfix] Zero new KV blocks for quantized + sliding-window hybrid caches (#47574)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/kv_cache_interface.py (+18/-1)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ A hybrid KV cache that quantizes full-attention layers but keeps sliding-window layers in higher precision (`--kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window`) produces **NaN / all-zero output** once the input exceeds the sliding window. Uniform fp8 and bf16 are fine; only the hybrid combo fails (seen on Blackwell + FlashInfer). ⏎  ⏎ ## Root cause ⏎  ⏎ Attention groups share one block pool. The sliding-window group frees …[truncated]

### L3-7ca017778f  (L3, 2026-07-20, sha 7ca017778fc0, PR #47451)
TITLE: [Feat][Perf] Add new warmup infrastructure for JITs (#47451)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+215/-79); vllm/v1/attention/backends/mla/prefill/flash_attn.py (+246/-50); vllm/v1/attention/backends/mla/sparse_swa.py (+108/-34); .buildkite/test_areas/model_executor.yaml (+4/-0); tests/model_executor/test_jit_warmup.py (+306/-0); vllm/config/kernel.py (+8/-0); vllm/model_executor/warmup/cutedsl_warmup.py (+4/-0); vllm/model_executor/warmup/fa4_cutedsl_config.py (+0/-204); vllm/model_executor/warmup/fa4_cutedsl_warmup.py (+30/-0); vllm/model_executor/warmup/jit_warmup.py (+474/-0); (+4 more)
LABELS: ready, ci/build, v1, deepseek, DSv4
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ This PR introduces a shared warmup infrastructure for JIT kernels in vLLM. ⏎  ⏎ The goal is to provide a standard, extensible way for kernels from different JIT backends, including Triton, CuTeDSL, TileLang, and potential future DSLs, to expose the set of specializations that should be compiled during engine startup. ⏎  ⏎ This is not intended to be a one-off warmup path for a specific kernel. Instead, it defines a kernel-owned contract  …[truncated]

### L3-f007cceb42  (L3, 2026-07-20, sha f007cceb4225, PR #48679)
TITLE: [KV Offload] Support self-describing KV events with TieringOffloadingSpec (#48679)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/features/kv_offloading_usage.md (+1/-1); tests/v1/kv_connector/unit/offloading_connector/test_events.py (+143/-14); tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py (+259/-27); vllm/distributed/kv_transfer/kv_connector/v1/offloading/events.py (+29/-11); vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py (+20/-4); vllm/v1/kv_offload/tiering/spec.py (+0/-8)
LABELS: documentation, ready, v1, kv-connector
BODY: ## Purpose ⏎  ⏎ Related to #38260 and follows up on #43468. ⏎  ⏎ #43468 added opt-in self-describing KV events for GPU-to-CPU stores, but `TieringOffloadingSpec` still rejected the flag. Tier promotions also bypass the normal store path, so their CPU `BlockStored` events had only placeholder payloads. ⏎  ⏎ Those placeholders do not include token IDs, a parent hash, or a block size. Without that data, Dynamo cannot index the promoted copy or route later reque …[truncated]

### L3-97a668152b  (L3, 2026-07-20, sha 97a668152b90, PR #44214)
TITLE: [RL Infra][FlashInfer] Enable router replay output from FlashInfer monolithic MoE kernel (#44214)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_routed_experts_capture_monolithic.py (+880/-0); tests/model_executor/test_routed_experts_capture.py (+55/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py (+13/-1); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+23/-1); vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py (+11/-1); vllm/model_executor/layers/fused_moe/experts/trtllm_mxint4_moe.py (+14/-1); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+13/-1); vllm/model_executor/layers/fused_moe/modular_kernel.py (+51/-0); vllm/model_executor/layers/fused_moe/routed_experts_capturer.py (+11/-1); vllm/model_executor/layers/quantization/utils/flashinfer_mxint4_moe.py (+2/-0); (+1 more)
LABELS: ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ Add `routing_replay_out` support to the FlashInfer monolithic MoE backend, enabling the RoutedExpertsCapturer to capture expert routing decisions from the monolithic kernel. ⏎  ⏎ Currently, routing capture only works through the modular kernel path via `router.set_capture_fn()`. The monolithic path fuses routing into the kernel itself, so routing decisions were lost. This PR enabled routing output by passing a `routing_replay_out` ten …[truncated]

### L3-97a98006b0  (L3, 2026-07-20, sha 97a98006b089, PR #47879)
TITLE: Update qutlass cmake for stable abi (#47879)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: cmake/external_projects/qutlass.cmake (+5/-5); vllm/_custom_ops.py (+6/-4)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ Updates the vLLM QuTLASS CMake integration to build against a stable-ABI QuTLASS fork (see ~~https://github.com/cleonard530/qutlass/pull/1~~ https://github.com/IST-DASLab/qutlass/pull/12), enabling the _qutlass_C library to become torch abi stable. ⏎  ⏎  ⏎ ~~This is just to test the CI and make sure these updates work. We will need to figure out if we want to try and push these updates upstream to [IST-DASLab/qutlass](https://github.com/ …[truncated]

### L3-4ec199b66a  (L3, 2026-07-20, sha 4ec199b66a79, PR #44492)
TITLE: [Bugfix][Spec-Decode] Populate draft seq_lens_cpu_upper_bound for spec-decode attention metadata (#44492)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py (+4/-0); vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (+6/-0); vllm/v1/worker/gpu/spec_decode/speculator.py (+12/-0); tests/v1/spec_decode/test_eagle_draft_attn_metadata.py (+128/-0)
LABELS: bug, speculative-decoding, ready, v1, nvidia, verified
BODY: ## Purpose ⏎ Purpose ⏎ Fixes ROCm CI failures in speculative decoding (EAGLE / EAGLE3) caused by the draft attention metadata being built without seq_lens_cpu_upper_bound. ⏎  ⏎ When the GPU model runner's draft speculators build the draft CommonAttentionMetadata, they did not populate seq_lens_cpu_upper_bound. Several consumers require this field for spec-decode batches — split_decodes_prefills_and_extends and the MLA / indexer / flex-attention backe …[truncated]

### L3-2396a61108  (L3, 2026-07-20, sha 2396a611085d, PR #45964)
TITLE: [Attention][MLA][DCP] Query replication for MLA decode (DeepSeek-V2/R1 + Kimi-K2.5) (#45964)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.common_v1
FILES: vllm/envs.py (+3/-0); vllm/model_executor/layers/attention/mla_attention.py (+46/-5); vllm/model_executor/layers/mla.py (+20/-4); vllm/model_executor/models/deepseek_v2.py (+12/-2); tests/v1/attention/test_mla_backends.py (+1/-0); vllm/model_executor/layers/linear.py (+75/-4)
LABELS: ready, v1, deepseek
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ With Decode Context Parallelism (DCP) the KV cache is sharded across the DCP group, so the standard MLA decode path **all-gathers the query** across the group every step so each rank can attend its KV shard with the full head set, then LSE-reduces the partials. That all-gather sits on the decode critical path. ⏎  ⏎ This PR adds an **opt-in** alternative: replicate the (small) MLA query projection *within each DCP group* at load time,  …[truncated]

### L3-adfbbc1005  (L3, 2026-07-21, sha adfbbc10051f, PR #49177)
TITLE: Propagate Flash Attention cache configuration to Ray workers (#49177)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/ray/ray_env.py (+1/-0)
LABELS: ready
BODY: This change adds `FLASH_ATTENTION_` to the environment-variable prefixes propagated from the vLLM driver to Ray workers. ⏎  ⏎ Flash Attention's compilation cache is configured through environment variables. Without propagating them, Ray workers do not use the configured cache and may repeat expensive compilation during worker startup or scaling operations. ⏎  ⏎ This keeps the cache configuration consistent between the driver and distributed workers, allo …[truncated]

### L3-3e0c887511  (L3, 2026-07-21, sha 3e0c8875118e, PR #47298)
TITLE: [Bugfix] Fix Ovis2_5 special tokens for transformers v5 (#47298)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/ovis2_5.py (+3/-4); vllm/transformers_utils/processors/ovis2_5.py (+9/-15)
LABELS: bug, ready, verified
BODY: ## Purpose ⏎ Fix Ovis2_5 processor  ⏎  ⏎ ## Test Plan ⏎ vllm serve AIDC-AI/Ovis2.5-2B --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 8192 --trust-remote-code 2>&1 | tee "serve.log" ⏎  ⏎ ## Test Result ⏎  ⏎ *Without the fix:* ⏎ (APIServer pid=12810) INFO 07-01 12:59:40 [api_utils.py:339]  ⏎ (APIServer pid=12810) INFO 07-01 12:59:40 [api_utils.py:339]        █     █     █▄   ▄█ ⏎ (APIServer pid=12810) INFO 07-01 12:59:40 [api_utils.py:339]  ▄▄ ▄█ █ …[truncated]

### L3-6700813f86  (L3, 2026-07-21, sha 6700813f8655, PR #44456)
TITLE: [3/N][KV-Cache Layout Refactor] Standardize Mamba cache; drop `get_transfer_cache_regions` (#44456)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention_layer_base.py (+10/-0); tests/v1/kv_connector/unit/offloading_connector/test_worker.py (+1/-2); vllm/distributed/kv_transfer/kv_connector/utils.py (+0/-28); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py (+3/-3); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+51/-61); vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py (+7/-20); vllm/model_executor/layers/mamba/abstract.py (+18/-0); vllm/v1/worker/gpu/attn_utils.py (+11/-87); vllm/v1/worker/gpu_model_runner.py (+9/-22); vllm/v1/worker/utils.py (+5/-2)
LABELS: rocm, ready, ci/build, v1, kv-connector, nvidia
BODY: PR #42374 (first part of RFC #42082) has been split into 4 PRs: ⏎  ⏎ #44454 [1/N][KV-Cache Layout Refactor] Refactor DSV4 KV cache config ⏎ #44455 [2/N][KV-Cache Layout Refactor] Pack K/V into the content dim across attention backends ⏎ -> #44456 [3/N][KV-Cache Layout Refactor] Standardize Mamba cache; drop `get_transfer_cache_regions` ⏎ #44458 [4/N][KV-Cache Layout Refactor] Standardize KV cache layout ⏎  ⏎ ## Summary ⏎  ⏎ Move Mamba (and conv/ssm) resha …[truncated]

### L3-72d16aee15  (L3, 2026-07-21, sha 72d16aee1576, PR #49231)
TITLE: [CI] Exercise FA3 FP8 attention on SM90 (#49231)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/quantization/test_fp8.py (+6/-3)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Follow up on the review feedback in https://github.com/vllm-project/vllm/pull/43024#discussion_r3614823352. ⏎  ⏎ The H200 migration had skipped `test_online_quantization[..., kv_cache_dtype=fp8]` on SM90 with the claim that FlashAttention 3 did not support FP8 attention. FA3 does support FP8 query input. Its implementation [selects BF16 output when Q is FP8 and validates a supplied output tensor against that dtype](https://github.com/vllm …[truncated]

### L3-6e96891ba0  (L3, 2026-07-21, sha 6e96891ba00d, PR #48683)
TITLE: [ROCm] Bump AITER to v0.1.16.post5 (#48683)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm_base (+1/-1)
LABELS: rocm, ready, ci/build
BODY: ## Purpose ⏎  ⏎ Advance the ROCm image build from AITER `v0.1.16.post3` to ⏎ `v0.1.16.post4`. This brings the gfx950 MXFP4 MoE path, top-k and ⏎ MLA metadata/reduction optimizations, and AITER's FlyDSL 0.2.2 dependency. ⏎ FlyDSL does not need a separate vLLM pin because the AITER build stage installs ⏎ AITER's `requirements.txt` before building its wheel. ⏎  ⏎ The tested AITER source commit was ⏎ `72eedddd921d0fb8f274e7b9291dd04306d089ef`. ⏎  ⏎ ### Duplicate-work chec …[truncated]

### L3-61e10f0116  (L3, 2026-07-21, sha 61e10f0116ef, PR #48845)
TITLE: [ROCm][CI] Fix AITER MLA fp8 decode metadata regression test (#48845)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_rocm_aiter_mla_decode_metadata.py (+5/-4)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ The gfx950 AITER MLA persistent-decode regression test ⏎ (`tests/kernels/attention/test_rocm_aiter_mla_decode_metadata.py`) started ⏎ failing on `main` due to a collision between two bugfixes: ⏎  ⏎ - **#46997** added the test *and* the `dtype_q`/`dtype_kv` forwarding to ⏎   `aiter.get_mla_metadata_v1`. At that point the decode **query** dtype was ⏎   `bf16`, so the test pinned `EXPECTED_Q_DTYPE = torch.bfloat16`. ⏎ - **#47276** (a later bu …[truncated]

### L3-96a739289e  (L3, 2026-07-21, sha 96a739289e07, PR #49016)
TITLE: [Bugfix] fix cutalss version upgrade bug, need update MSG new commit (#49016)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: cmake/external_projects/fmha_sm100.cmake (+1/-1); requirements/cuda.txt (+1/-1)
LABELS: bug, ready, ci/build, nvidia
BODY: ## Purpose ⏎  ⏎ Fixs: https://github.com/vllm-project/vllm/issues/49005 ⏎  ⏎ Wait https://github.com/vllm-project/MSA/pull/8 this pr merge after update new commit. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-7bb49be4d1  (L3, 2026-07-21, sha 7bb49be4d116, PR #49306)
TITLE: [Bugfix] Handle MLA fallback during FA4 JIT warmup (#49306)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/warmup/fa4_cutedsl_warmup.py (+5/-1)
LABELS: bug, ready
BODY: Follow-up to #47451. As @mgoin correctly noted in https://github.com/vllm-project/vllm/pull/47451#discussion_r3616102551, JIT warmup must handle models for which no generic MLA prefill backend is available. ⏎  ⏎ Related auto-revert PR: #49268.

### L3-5aab491bc9  (L3, 2026-07-21, sha 5aab491bc98f, PR #49325)
TITLE: [CI] Wire tests/models/inkling into a B200 job (#49325)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/models_basic.yaml (+13/-0)
LABELS: ready, ci/build
BODY: The tests/models/inkling suite (8 files: FA4 kernels, sconv metadata, MoE weight layout, contract validation, MTP input fusion, QKV prep, MM towers) was not referenced by any CI job, so none of it ran in CI - including the contract-validation tests that #48822 itself modified. Add a B200 job (the FA4 tests require SM100; they skip elsewhere) to the Models - Basic group. ⏎  ⏎ Suite runs in ~6.5 min on 1 GPU; validated 194/194 passing at current main …[truncated]

### L3-33178f9006  (L3, 2026-07-21, sha 33178f90063e, PR #49292)
TITLE: Fix Qwen3-VL M-RoPE on the Transformers modeling backend (grids + compile) (#49292)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/transformers/multimodal.py (+3/-7)
LABELS: ready, qwen
BODY: I wanted to serve the [`Qwen/Qwen3-VL-4B-Instruct-FP8`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8) using the transformers modeling backed. I came across two distinct issues, one with *compilation* and other with *eager*. In this PR I will talk about them seperately. ⏎  ⏎ ## Enforcing Eager ⏎  ⏎ ```bash ⏎ vllm serve Qwen/Qwen3-VL-4B-Instruct-FP8 --model-impl transformers --enforce-eager ⏎ ``` ⏎  ⏎ This command was able to get the engine running …[truncated]

### L3-05781e21dd  (L3, 2026-07-21, sha 05781e21dd4a, PR #49329)
TITLE: [ROCm][CI] Fix order-dependent failure in test_flash_attn_accepts_handled_fp8_variants (MI355) (#49329)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_attention_selector.py (+5/-2)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ `tests/kernels/attention/test_attention_selector.py::test_flash_attn_accepts_handled_fp8_variants[fp8|fp8_e4m3]` ⏎ fails on the `Kernels (B200-MI355)` CI group when the file is run in full, while ⏎ passing when the two cases are run in isolation: ⏎  ⏎ ``` ⏎ assert FlashAttentionBackend.supports_kv_cache_dtype('fp8_e4m3') ⏎ E   AssertionError: assert False ⏎ E    +  where False = supports_kv_cache_dtype('fp8_e4m3') ⏎ ``` ⏎  ⏎ This is a **test- …[truncated]

### L3-16aca639b7  (L3, 2026-07-21, sha 16aca639b747, PR #49251)
TITLE: [ROCm] Upgrade NIXL and UCX (#49251)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/scripts/ci-bake-rocm.sh (+28/-28); docker/Dockerfile.rocm (+54/-32); docker/ci-rocm.hcl (+13/-13); docker/docker-bake-rocm.hcl (+2/-2); docs/features/nixl_connector_usage.md (+1/-5); tests/v1/kv_connector/unit/test_nixl_connector.py (+10/-7); tests/v1/kv_connector/unit/test_nixl_rocm_gpu_mem_diag.py (+7/-7); vllm/config/parallel.py (+1/-1); vllm/distributed/eplb/eplb_communicator.py (+2/-2); vllm/distributed/nixl_utils.py (+9/-5)
LABELS: documentation, rocm, intel-gpu, ready, ci/build, v1, kv-connector
BODY: - Upgrade the ROCm image to pinned upstream NIXL and UCX revisions. ⏎ - Build the ROCm NIXL wheel and set `UCX_RMA_PPLN_ENABLE=y`. ⏎ - Prefer `nixl_rocm` while retaining legacy RIXL compatibility.

### L3-0500ca6a58  (L3, 2026-07-22, sha 0500ca6a589b, PR #49380)
TITLE: [CI][Bugfix] Fix ROCm FP8 KV cache dtype in attention backend test (#49380)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_attention_backends.py (+5/-2)
LABELS: bug, rocm, ready, v1
BODY: ## Purpose ⏎  ⏎ `tests/v1/attention/test_attention_backends.py::test_causal_backend_correctness[fp8*]` fails on ROCm (gfx94x / MI300) with `[AttentionBackendEnum.TRITON_ATTN] produced non-finite values`. ⏎  ⏎ Root cause is in the test, not the backend. The test stores the FP8 KV cache using a hardcoded `torch.float8_e4m3fn`, but at runtime the attention backends reinterpret the raw cache bytes with `current_platform.fp8_dtype()`, which is `e4m3fnuz`  …[truncated]

### L3-060b5f61dc  (L3, 2026-07-22, sha 060b5f61dcb1, PR #49294)
TITLE: [Bugfix][Attention] Ignore empty MLA context chunks during merge (#49294)
SOURCES: path_core, subject_keyword, symbol_pickaxe
ARTIFACT_HINTS: L3.merge.triton_lse, L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+19/-0); vllm/v1/attention/ops/triton_merge_attn_states.py (+119/-0); tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP1-PCP4-EP.yaml (+1/-0); tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP2-PCP2-EP.yaml (+1/-0); tests/kernels/attention/test_merge_attn_states.py (+56/-0)
LABELS: bug, ready, v1
BODY: ## Summary ⏎  ⏎ FIX for https://github.com/vllm-project/vllm/issues/49334 ⏎ Alternative to https://github.com/vllm-project/vllm/pull/49196 ⏎  ⏎ - mark zero-context MLA attention states with `-inf` LSE before merging ⏎ - use a single in-place Triton kernel driven by the existing ragged query and context offsets ⏎ - resolve query-block ownership 32 request boundaries at a time with a warp-local reduction, amortized across all attention heads ⏎ - launch the …[truncated]

### L3-7c21548ce3  (L3, 2026-07-22, sha 7c21548ce38f, PR #49297)
TITLE: [PD][Bugfix] Fix NIXL hybrid MLA+mamba heterogeneous TP (#49297)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+5/-3); vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py (+5/-5)
LABELS: bug, ready, kv-connector
BODY: ## Purpose ⏎ Fixes KV transfer for hybrid MLA + SSM (Mamba) models under heterogeneous TP (prefill TP > decode TP) with the NIXL connector. ⏎  ⏎  ⏎ ## Test Plan ⏎  ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-37e370fe93  (L3, 2026-07-22, sha 37e370fe936f, PR #48957)
TITLE: [DSv4 Perf] Skip empty c128 kernel launch, around 2x kernel performance improvement. (#48957)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_compressor_kv_cache.py (+21/-0); vllm/models/deepseek_v4/compressor.py (+32/-2)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ We do compress in `SparseAttnCompressC128Block8Kernel` and store in `SparseAttnNormRopeStoreFullKernel` ⏎  ⏎ We will skip it inside the kernel by taking a look at boundary, but this skip could be move ahead to bypass the full kernel. ⏎  ⏎ ## Test ⏎  ⏎ Acc covered in unit test ⏎  ⏎ Perf through this AI generated script ⏎  ⏎ ```py ⏎ # SPDX-License-Identifier: Apache-2.0 ⏎ # SPDX-FileCopyrightText: Copyright contributors to the vLLM project ⏎ """Mi …[truncated]

### L3-1a659a0c37  (L3, 2026-07-22, sha 1a659a0c3709, PR #49431)
TITLE: Upgrade tpu-inference to v0.25.0 (#49431)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/tpu.txt (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ Upgrade tpu-inference to latest stable release v0.25.0 ⏎  ⏎ ## Test Plan ⏎ Verified on tpu-inference CI.  ⏎  ⏎ ## Test Result ⏎ Success.  ⏎  ⏎ --- ⏎ [details omitted]

### L3-61a09532f2  (L3, 2026-07-22, sha 61a09532f23a, PR #48914)
TITLE: Bump Flashinfer version to 0.6.15 (#48914)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); docker/versions.json (+1/-1); requirements/cuda.txt (+2/-2)
LABELS: ready, ci/build, nvidia, ready-run-all-tests
BODY: ## Purpose ⏎ Bump flashinfer version to 0.6.15 ~~and re-enable persistent cache for autotuning using `set_autotune_process_group`, which supports distributed auto-tuning across ranks with synchronization to avoid timeout caused by straggler.~~ ⏎  ⏎ There has been observation that the current version (0.6.14) may cause hang when using nvfp4 MoE, see https://github.com/flashinfer-ai/flashinfer/issues/3971. This is addresses by https://github.com/flash …[truncated]

### L3-c79ff5f918  (L3, 2026-07-22, sha c79ff5f91854, PR #49326)
TITLE: [Build] Bump vllm-flash-attn to C++20-compatible commit for torch-nightly (#49326)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
BODY: ## Summary ⏎  ⏎ Bumps the `vllm-flash-attn` pin so vLLM builds against **torch nightly**, which now requires **C++20**. ⏎  ⏎ ```diff ⏎ - GIT_TAG 168920233059c48de6199e2cda74003b2ce3d199 ⏎ + GIT_TAG ed4b7342bc8f0489dd9b649d5288867e35fc6a32 ⏎ ``` ⏎  ⏎ `ed4b7342` = flash-attention#168 (*"Require C++20 to match PyTorch ATen headers"*), which sets `CMAKE_CXX_STANDARD 20` in the vllm-flash-attn build. ⏎  ⏎ ## Why ⏎  ⏎ Recent PyTorch added a hard guard in `ATen.h`: ⏎ ``` ⏎ ATen/ATen …[truncated]

### L3-f3a920a076  (L3, 2026-07-22, sha f3a920a07640, PR #48993)
TITLE: [Core][DSV4] Compact MXFP4 indexer KV cache and packed group overlays (#48993)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_contiguous_kv_packing.py (+170/-22); vllm/models/deepseek_v4/attention.py (+11/-5); vllm/v1/core/kv_cache_utils.py (+35/-57)
LABELS: ready, v1
BODY: ## Context ⏎  ⏎ When DeepSeek V4 uses MXFP4 indexer K values, engine reserves the larger FP8 row for them. Its packed KV planner also buckets layers by page size, leaving avoidable holes when different cache groups have different mixtures of page sizes. ⏎  ⏎ This PR: ⏎  ⏎ - sizes MXFP4 indexer K rows from their packed values and UE8M0 scales (`head_dim / 2 + head_dim / 32` bytes), reducing the 128-dimensional row from 132 to 68 bytes; ⏎ - lays out every …[truncated]

### L3-b07ec92faa  (L3, 2026-07-22, sha b07ec92faa2d, PR #49489)
TITLE: [Bugfix] Make shared NVFP4 MoE scales writable (#49489)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/quantization/test_trtllm_nvfp4_hidden_dim_padding.py (+42/-0); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+4/-4)
LABELS: bug, ready, nvidia, quantization
BODY: FlashInfer NVFP4 backends broadcast shared input scales with expand, leaving registered parameters backed by overlapping stride-zero storage. Layerwise reload cannot copy regenerated values into those parameters. ⏎  ⏎ Materialize independent per-expert scale storage and cover the copy-back operation with a focused regression test. ⏎  ⏎ Assisted-by: OpenAI Codex ⏎  ⏎ Without this, NVFP4 online quantization of Qwen3 30B A3B fails with ⏎  ⏎ ```(EngineCore p …[truncated]

### L3-27ffbfde8d  (L3, 2026-07-23, sha 27ffbfde8dec, PR #48044)
TITLE: Fused Shared Expert Support for AMD Quark DeepSeek-V4 Model Checkpoints (#48044)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/amd/model.py (+121/-11); vllm/models/deepseek_v4/amd/mtp.py (+17/-0); vllm/models/deepseek_v4/quant_config.py (+37/-2)
LABELS: rocm, ready, deepseek
BODY: ## Purpose ⏎  ⏎ Enable serving AMD Quark DeepSeek-V4-MXFP4 checkpoints (AMD-Quark quantized, allowing MXFP4 shared expert fusion) in vLLM. The runtime layout implemented in this PR reuses the existing deepseek_v4_fp8 path. ⏎  ⏎ ## Changes ⏎  ⏎ **`vllm/models/deepseek_v4/quant_config.py`** ⏎  ⏎ - `DeepseekV4FP8Config.override_quantization_method` now also claims `quant_method="quark"` for `model_type=deepseek_v4`, guarded by `_is_quark_mxfp4_ocp` so **onl …[truncated]

### L3-76bf55240c  (L3, 2026-07-23, sha 76bf55240cf8, PR #49415)
TITLE: [Bugfix] Fix DeepSeek-V4 DSpark draft shared-expert padding for TP > 8 (#49415)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/nvidia/dspark.py (+12/-4); vllm/models/deepseek_v4/nvidia/model.py (+8/-3); vllm/models/deepseek_v4/nvidia/mtp.py (+11/-4)
LABELS: bug, ready, deepseek
BODY: ## Purpose ⏎  ⏎ Add DeepSeek-V4-Pro DSpark shared-expert padding for built-in MTP loading when tensor parallelism requires block-aligned shared-expert padding to support sharding.  This is needed to serve DSV4-Pro with a 16x H100 sharded configuration using TP=16. ⏎  ⏎ Use in conjunction with https://github.com/vllm-project/vllm/pull/49133 which provides the basic enablement for DSV4-NVFP4 with DSpark-MXFP4 draft enablement. ⏎  ⏎ The target-model loade …[truncated]

### L3-e18f0037a5  (L3, 2026-07-23, sha e18f0037a5d5, PR #48776)
TITLE: [Bugfix][KV cache] Support sparse-MLA targets with SWA drafts (#48776)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_kv_cache_utils.py (+71/-6); vllm/v1/core/kv_cache_utils.py (+110/-55)
LABELS: bug, ready, v1
BODY: ## Purpose ⏎  ⏎ Serving a sparse-MLA target such as `nvidia/GLM-5.2-NVFP4` with a regular sliding-window DSpark draft could fail during KV-cache planning because the target MLA/indexer pages cannot be unified with the draft page. ⏎  ⏎ ## Design ⏎  ⏎ Keep the existing grouping path unchanged when page sizes can be unified. If page-size unification fails for the narrow MLA + regular-SWA case, promote only the draft's cache-allocation spec to `FullAttenti …[truncated]

### L3-b0cb1da1bd  (L3, 2026-07-23, sha b0cb1da1bde6, PR #49486)
TITLE: [DSv4 Perf] Skip topk and router when not needed, 3.4% E2E TTFT improvement for Decode case (#49486)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/attention.py (+43/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Skip topk and router when not needed ⏎  ⏎ Originally: ⏎  ⏎ ```bash ⏎ Input ⏎ -> K compressor (and write K cache) ⏎ -> Query proj wq_b ⏎ -> Query RoPE + quantization ⏎ ->  calculation for Query-K logits ⏎ ->  topk logits ⏎ -> return index ⏎ ``` ⏎  ⏎ Now ⏎  ⏎ ```bash ⏎ -> K compressor (and write K cache) ⏎ -> directly choose all candidates if candidates num <= topk num ⏎ -> return index ⏎ ``` ⏎  ⏎ Part of https://github.com/vllm-project/vllm/issues/45861 ⏎  …[truncated]

### L3-ac36a7a1e7  (L3, 2026-07-23, sha ac36a7a1e7eb, PR #48630)
TITLE: [MRV2][Spec Decode] Avoid rejection sampler OOM by chunking (#48630)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/spec_decode/test_rejection_sampler_utils.py (+62/-0); tests/v1/test_outputs.py (+26/-1); tests/v1/worker/test_gpu_rejection_sampler_chunking.py (+109/-0); vllm/config/model.py (+4/-0); vllm/v1/outputs.py (+27/-0); vllm/v1/sample/ops/topk_topp_sampler.py (+8/-12); vllm/v1/sample/rejection_sampler.py (+2/-4); vllm/v1/worker/gpu/sample/prompt_logprob.py (+1/-9); vllm/v1/worker/gpu/sample/sampler.py (+3/-6); vllm/v1/worker/gpu/spec_decode/rejection_sampler.py (+142/-31); (+2 more)
LABELS: speculative-decoding, ready, v1, mrv2
BODY: ## Purpose ⏎  ⏎ Replacement for https://github.com/vllm-project/vllm/pull/48037 where we cap the intermediate memory required for large batch rejection sampling by using a fixed size scratch buffer and simply go through the sampling process multiple times. The loop is sync-free since chunk bounds come from `cu_num_logits_np`. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ vllm serve Qwen/Qwen3-8B --spec-model RedHatAI/Qwen3-8B-speculator.dflash --spec-m …[truncated]

### L3-0d77325b10  (L3, 2026-07-23, sha 0d77325b10f9, PR #49223)
TITLE: Bump Transformers version to 5.14.1 (#49223)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/test/cpu.txt (+1/-1); requirements/test/cuda.in (+1/-1); requirements/test/cuda.txt (+1/-1); requirements/test/nightly-torch.txt (+1/-1); requirements/test/rocm.in (+1/-1); requirements/test/rocm.txt (+1/-1); requirements/test/xpu.in (+1/-1); requirements/test/xpu.txt (+1/-1)
LABELS: ci/build, cpu, nvidia, ready-run-all-tests
BODY: No fixes needed for this one, it just worked!

### L3-dd72658e7d  (L3, 2026-07-23, sha dd72658e7db0, PR #48597)
TITLE: [Perf][GLM-5.2] Blackwell decode optimizations (#48597)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.mla.flashinfer_sparse
FILES: vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+31/-0); vllm/v1/attention/backends/mla/sparse_utils.py (+56/-4); CMakeLists.txt (+17/-0); csrc/libtorch_stable/bf16_skinny_gemm.cu (+262/-0); csrc/libtorch_stable/bf16_skinny_gemm_entry.cu (+170/-0); csrc/libtorch_stable/dsv3_fused_a_gemm.cu (+112/-36); csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu (+42/-10); csrc/libtorch_stable/torch_bindings.cpp (+4/-0); recipes/glm5.2-ll-b300-tp8-mtp5.md (+41/-0); tests/kernels/test_bf16_skinny_gemm.py (+70/-0); (+19 more)
LABELS: new-model, structured-output, speculative-decoding, ready, ci/build, v1, deepseek, nvidia, verified
DEEP_STUDY: deep-study: this PR was reverted by PR 49768 (confirmed_revert, reason=unstated) || deep-study performance PR ()
BODY: # Blackwell decode optimization tracker ⏎  ⏎ This pull request is now the tracking page for the focused follow-up work. ⏎  ⏎ The original squash merge was reverted by #49768. A checkbox is marked complete only when the corresponding functionality is present on the current `vllm-project/vllm/main` branch. ⏎  ⏎ ## Already on `main` ⏎  ⏎   [`main` source](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/fp32_router_gemm.cu) ⏎  ⏎   [`main` sou …[truncated]

### L3-833483f357  (L3, 2026-07-24, sha 833483f3578a, PR #48218)
TITLE: Encoder cache extension hooks (#48218)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_model_runner_mm_gather.py (+1/-0); vllm/config/__init__.py (+3/-0); vllm/config/ec_manager_config.py (+29/-0); vllm/config/vllm.py (+5/-0); vllm/v1/core/encoder_cache_manager.py (+4/-0); vllm/v1/core/sched/output.py (+4/-1); vllm/v1/core/sched/scheduler.py (+10/-6); vllm/v1/worker/gpu_model_runner.py (+50/-8)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ This PR adds extension hooks for the V1 encoder cache lifecycle so downstream or out-of-tree implementations can customize encoder cache behavior without changing the default vLLM path. ⏎  ⏎ Concretely, this PR: ⏎  ⏎ - Adds `EncoderCacheManagerConfig` and `EncoderCacheManagerMetadata`. ⏎ - Allows `VllmConfig` to provide a custom scheduler-side encoder cache manager class via `ec_manager_config.encoder_cache_manager_cls`. ⏎ - Propagates en …[truncated]

### L3-866fea2b99  (L3, 2026-07-24, sha 866fea2b9900, PR #48018)
TITLE: [Kernel] ReplaySSM: cache SSM inputs for faster Mamba2 standard decode (#48018)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+6/-0); benchmarks/replayssm/e2e_decode_speedup.py (+267/-0); tests/kernels/mamba/test_replayssm_prefill_decode_equivalence_mamba2.py (+244/-0); tests/kernels/mamba/test_replayssm_standard_decode_mamba2.py (+673/-0); tests/kernels/mamba/utils.py (+160/-0); tests/models/test_registry.py (+16/-0); tests/v1/attention/test_mamba_update_block_table.py (+2/-0); tests/v1/attention/test_replayssm_metadata_builder.py (+256/-0); tests/v1/e2e/test_replayssm_decode.py (+138/-0); vllm/config/cache.py (+11/-0); (+13 more)
LABELS: performance, new-model, ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Background ⏎  ⏎ This is the first sub-PR split out from the large draft PR [#47576](https://github.com/vllm-project/vllm/pull/47576), which presents the full ReplaySSM design across Mamba2 and Gated DeltaNet for both standard and speculative decode. To keep review tractable, this PR lands only the first stage, Mamba2 standard decode. ReplaySSM caches recent SSM inputs instead of writing the recurrent state back to HBM on every step. It is a coll …[truncated]

### L3-8c13ee5735  (L3, 2026-07-24, sha 8c13ee5735f5, PR #49387)
TITLE: Add `sm_107` for Rubin (#49387)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.mla.flashmla_build
FILES: cmake/external_projects/flashmla.cmake (+3/-1); CMakeLists.txt (+12/-7); cmake/external_projects/deepgemm.cmake (+3/-0); cmake/external_projects/qutlass.cmake (+5/-1); cmake/utils.cmake (+14/-4); tests/test_cmake_utils.py (+23/-0)
LABELS: ready, ci/build, nvidia
BODY: ## Summary ⏎  ⏎ - enable CUDA 13.4's native SM107 target for Vera Rubin ⏎ - allow SM107 to reuse compatible SM100-family kernels across the CUDA, DeepGEMM, FlashMLA, and QuTLASS builds ⏎ - resolve exact architecture matches before family fallbacks, preventing `10.0f` from masking `10.7f` ⏎  ⏎ This follows [PyTorch #190654](https://github.com/pytorch/pytorch/pull/190654). I also checked its review feedback and mirrored the corrected boundaries here: CUDA 13.4 …[truncated]

### L3-5d8e90a966  (L3, 2026-07-24, sha 5d8e90a96616, PR #45321)
TITLE: [WideEP] Update NCCL to 2.30.7 to enable DeepEPv2 in the vllm/vllm-openai image (#45321)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/test_areas/misc.yaml (+1/-1); .buildkite/test_areas/model_runner_v2.yaml (+1/-1); docker/Dockerfile (+25/-8); docker/versions.json (+4/-1); docs/serving/expert_parallel_deployment.md (+5/-0); tests/distributed/test_mnnvl_alltoall.py (+9/-3); tests/kernels/moe/parallel_utils.py (+36/-12); tests/kernels/moe/test_deepep_v2_moe.py (+101/-50); tools/ep_kernels/README.md (+32/-0); vllm/distributed/device_communicators/all2all.py (+27/-0); (+3 more)
LABELS: documentation, rocm, ready, ci/build
BODY: Update DeepEP to https://github.com/deepseek-ai/DeepEP/commit/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb so that we can pick up DeepEPv2. ⏎  ⏎ Missed this in https://github.com/vllm-project/vllm/pull/41183 ⏎  ⏎ Ideally we'd have a single source of truth for the commit across both the install script and the Dockerfile but I don't see a sane way to do it at the moment ⏎  ⏎ PyTorch ships an earlier version of NCCL. Per feedback from the NCCL team, I think t …[truncated]

### L3-2279575cd9  (L3, 2026-07-24, sha 2279575cd9ca, PR #47206)
TITLE: [AMD][Bugfix][EPLB] Fix elastic EP scaling accuracy on ROCm (#47206)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/eplb/eplb_state.py (+74/-16); vllm/model_executor/layers/fused_moe/routed_experts.py (+15/-5)
LABELS: bug, rocm, ready
BODY: ## Purpose ⏎  ⏎ Two fixes for distributed/test_elastic_ep.py, which collapsed GSM8K accuracy at the initial and scaled checkpoints on ROCm: ⏎  ⏎ Bug 1 Scale-up produces garbage output on ROCm (accuracy=0) ⏎  ⏎ Root cause. New elastic-EP ranks are created with load_dummy_weights=True. On ROCm, initialize_single_dummy_weight() .zero_()s every non-floating-point tensor in model.state_dict(). The FusedMoE expert-topology buffers _expert_map, expert_mask, a …[truncated]

### L3-213f681f81  (L3, 2026-07-24, sha 213f681f8117, PR #49768)
TITLE: Revert "[Perf][GLM-5.2] Blackwell decode optimizations" (#49768)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.mla.flashinfer_sparse
FILES: vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+0/-31); vllm/v1/attention/backends/mla/sparse_utils.py (+4/-56); CMakeLists.txt (+0/-17); csrc/libtorch_stable/bf16_skinny_gemm.cu (+0/-262); csrc/libtorch_stable/bf16_skinny_gemm_entry.cu (+0/-170); csrc/libtorch_stable/dsv3_fused_a_gemm.cu (+36/-112); csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu (+10/-42); csrc/libtorch_stable/torch_bindings.cpp (+0/-4); recipes/glm5.2-ll-b300-tp8-mtp5.md (+0/-41); tests/kernels/test_bf16_skinny_gemm.py (+0/-70); (+19 more)
LABELS: new-model, speculative-decoding, ci/build, v1, deepseek, nvidia
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 48597 reason=unstated
BODY: Reverts vllm-project/vllm#48597

### L3-318b527cc2  (L3, 2026-07-25, sha 318b527cc2d1, PR #49419)
TITLE: [XPU] add warning for xpu graph limitations (#49419)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/platforms/xpu.py (+10/-0)
LABELS: intel-gpu, ready
BODY: ## Purpose ⏎  ⏎ Add a warning when XPU Graph is enabled to highlight its experimental status and known limitations: ⏎ - only single-GPU execution is supported ⏎ - FLASH_ATTN supports PIECEWISE mode only; use TRITON_ATTN for FULL mode ⏎ - graph capture may use significantly more memory than CUDA ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-fe5145765f  (L3, 2026-07-25, sha fe5145765f21, PR #48796)
TITLE: [Core] Keep attention backends eligible for text-only serving of prefix-LM models (#48796)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/model.py (+43/-1); tests/config/test_multimodal_config.py (+100/-0); tests/models/utils.py (+6/-0); vllm/transformers_utils/model_arch_config_convertor.py (+16/-6)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Prefix-LM multimodal models set `is_mm_prefix_lm=True` from static model configuration (e.g. Gemma 4, Gemma 3, Molmo2). This flag is evaluated at **server startup** and tells backend selection that some multimodal tokens may need bidirectional (prefix) attention, so backends without `supports_mm_prefix()` are rejected. ⏎  ⏎ That constraint is **correct while vision inputs can still appear**. The problem is that it is applied unconditional …[truncated]

### L3-b9b6306ebe  (L3, 2026-07-25, sha b9b6306ebe0f, PR #39330)
TITLE: feat[vLLM × v5]: Add audio support for the Transformers backend (#39330)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/models/supported_models.md (+3/-2); tests/models/multimodal/generation/test_transformers_audio.py (+133/-0); tests/models/multimodal/processing/test_common.py (+8/-0); tests/models/multimodal/processing/test_transformers_audio.py (+138/-0); tests/models/multimodal/processing/test_transformers_image.py (+23/-0); tests/models/registry.py (+4/-0); vllm/model_executor/models/registry.py (+4/-0); vllm/model_executor/models/transformers/multimodal.py (+367/-148)
LABELS: documentation, new-model, ready, multi-modality, verified
ISSUES: #32823 [New Model]: microsoft/VibeVoice-ASR support
BODY: ### What does this PR do? ⏎  ⏎ → **This PR adds support for v5 Transformers audio encoder models in the vLLM Transformers backend**. <ins>These changes are deliberate and are blocked by</ins> [this Transformers PR](https://github.com/huggingface/transformers/pull/45326) which adds prerequisite compatibility to the supported models for vLLM. Once that PR is merged, this PR will be marked ready for review! ⏎ → Outlining the design choices of one PR wi …[truncated]

### L3-3e74c60b9c  (L3, 2026-07-25, sha 3e74c60b9c74, PR #49587)
TITLE: [Docs] Use `gen-files` for generated docs content (#49587)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .gitignore (+0/-3); .markdownlint.yaml (+3/-0); .pre-commit-config.yaml (+0/-4); docs/.nav.yml (+3/-1); docs/cli/.nav.yml (+7/-9); docs/cli/bench/latency.md (+0/-9); docs/cli/bench/mm_processor.md (+0/-55); docs/cli/bench/serve.md (+0/-9); docs/cli/bench/sweep/plot.md (+0/-9); docs/cli/bench/sweep/plot_pareto.md (+0/-9); (+24 more)
LABELS: documentation, frontend, ready, build-docs
BODY: Until now we have had a mixture of methods for generating content in the docs using: ⏎  ⏎ - mkdocs hooks to generate hidden `*.inc.md` files and then including them in near empty files using snippets ⏎ - pre-commit hooks to generate a whole page that was tracked in git ⏎  ⏎ This PR ports both of these methods to use `gen-files`, which: ⏎  ⏎ - Generates virtual files for the CLI reference, just like the API reference ⏎ - Moves preamble to the docstring fo …[truncated]

### L3-70009fb934  (L3, 2026-07-25, sha 70009fb9344d, PR #46837)
TITLE: [MM][CG] Support ViT CUDA Graph for Gemma-4 (#46837)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/design/cuda_graphs_multimodal.md (+1/-0); examples/generate/multimodal/vision_language_offline.py (+41/-0); tests/models/multimodal/generation/test_vit_cudagraph.py (+13/-0); vllm/model_executor/models/gemma4_mm.py (+421/-1)
LABELS: documentation, ready, multi-modality, nvidia
BODY: This PR implements static CUDA graph support for the Gemma-4 vision encoder. ⏎  ⏎ ## Purpose ⏎ This PR introduces full `SupportsEncoderCudaGraph` support for `Gemma4ForConditionalGeneration`, enabling 100% static compilation for the vision encoder. The core optimization bypasses dynamic slicing in the pooler. Instead of relying on data-dependent control flow, the host calculates static pointwise mapping indices (`gather_indices`). Inside the compile …[truncated]

### L3-48ebd6f2f1  (L3, 2026-07-26, sha 48ebd6f2f164, PR #49226)
TITLE: [Bugfix][KVConnector] Disable cross-layer KV blocks for per-token-head quant (#49226)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/kv_connector_model_runner_mixin.py (+4/-0)
LABELS: bug, ready, v1, kv-connector, quantization
ISSUES: #48412 [Bug]: OffloadingConnector corrupts outputs with per-token-head quantized KV cache (cross-layer allocation lacks scale packing)
BODY: ## Purpose ⏎  ⏎ Fixes #48412. ⏎  ⏎ Combining `OffloadingConnector` with a per-token-head quantized KV cache ⏎ (`fp8_per_token_head` / `int8_per_token_head` / `int4_per_token_head`) corrupts ⏎ generations on the **first forward pass**, before any offloaded block is ever ⏎ restored. ⏎  ⏎ ### Root cause ⏎  ⏎ `OffloadingConnector.prefer_cross_layer_blocks` is hardcoded `True`. For ⏎ single-group models on backends with `indexes_kv_by_block_stride` (e.g. ⏎ `TRITON …[truncated]

### L3-7eca0e1a64  (L3, 2026-07-26, sha 7eca0e1a649e, PR #48906)
TITLE: [KV Offload] Deduplicate replicated MLA KV in the shared CPU region (#48906)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/lm_eval.yaml (+2/-2); tests/evals/gsm8k/test_gsm8k_offloading.py (+25/-5); tests/v1/kv_connector/unit/offloading_connector/test_config.py (+587/-0); tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py (+47/-0); tests/v1/kv_connector/unit/offloading_connector/test_worker.py (+184/-1); tests/v1/kv_connector/unit/offloading_connector/utils.py (+4/-0); tests/v1/kv_offload/cpu/test_gpu_worker.py (+10/-2); tests/v1/kv_offload/test_factory.py (+215/-476); vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py (+28/-7); vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py (+11/-0); (+4 more)
LABELS: ready, ci/build, v1, kv-connector
BODY: ## Purpose ⏎  ⏎ This implements the first shared CPU-region replica-reduction path from #47929. ⏎  ⏎ For pure MLA tensor parallelism, each TP rank holds a replica of the latent KV payload. This duplicate host-copy path is present in both V1 and V2: both runners hand per-rank canonical MLA tensors to the same offloading connector, which previously reserved and stored one host slot per rank. The runners organize their GPU tensors differently, but replica o …[truncated]

### L3-da3a252fd1  (L3, 2026-07-26, sha da3a252fd13f, PR #48021)
TITLE: [KVOffload][P2P] Generic P2P secondary tier: peer lookup and serving via ParentManager (#48021)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/features/kv_offloading_usage.md (+66/-1); tests/v1/kv_offload/tiering/p2p/p2p_connector_proxy.py (+22/-7); tests/v1/kv_offload/tiering/p2p/test_manager.py (+256/-51); tests/v1/kv_offload/tiering/p2p/test_sessions.py (+966/-83); vllm/v1/kv_offload/base.py (+15/-2); vllm/v1/kv_offload/tiering/p2p/manager.py (+235/-93); vllm/v1/kv_offload/tiering/p2p/session/__init__.py (+2/-0); vllm/v1/kv_offload/tiering/p2p/session/client.py (+342/-36); vllm/v1/kv_offload/tiering/p2p/session/protocol.py (+91/-9); vllm/v1/kv_offload/tiering/p2p/session/server.py (+558/-95); (+1 more)
LABELS: documentation, ready, v1
BODY: ## Summary ⏎  ⏎ Adds a generic peer-to-peer (P2P) secondary tier for KV-cache offloading, ⏎ letting a vLLM engine both **fetch** KV blocks from and **serve** KV blocks to ⏎ remote peers over a NIXL data transport with a ZMQ control channel. It ⏎ generalizes the existing prefill/decode (PD) path into a symmetric P2P model: a ⏎ consumer issues a `LookupMsg` for the blocks it needs; the producer answers ⏎ against its local tiering manager and streams match …[truncated]

### L3-b5b61c622c  (L3, 2026-07-26, sha b5b61c622c94, PR #46877)
TITLE: [Core][Distributed] Add process-checkpoint lifecycle hooks for communicators (starting with Flashinfer) (#46877)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+2/-2); vllm/v1/worker/gpu_worker.py (+8/-0); tests/distributed/test_comm_ops.py (+40/-0); vllm/distributed/device_communicators/all2all.py (+14/-0); vllm/distributed/device_communicators/base_device_communicator.py (+21/-0); vllm/distributed/device_communicators/cuda_communicator.py (+16/-0); vllm/distributed/device_communicators/flashinfer_all_reduce.py (+41/-1); vllm/distributed/parallel_state.py (+36/-0); vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py (+1/-1); vllm/v1/engine/async_llm.py (+6/-0)
LABELS: ready, ci/build, v1, nvidia
BODY: ## Summary ⏎  ⏎ Add a focused process-checkpoint lifecycle for FlashInfer-owned communication resources. ⏎  ⏎ - Forward V1 GPU worker prepare/restore calls through distributed state to registered device communicators, with accelerator synchronization around each transition. ⏎ - Have the CUDA communicator transition FlashInfer all-reduce workspaces owned by its CPU process group independently of the optional standalone FlashInfer all-reduce backend, then tr …[truncated]

### L3-8de50e46d4  (L3, 2026-07-26, sha 8de50e46d4a0, PR #49376)
TITLE: [Docs] Document NVFP4 GEMM kernel selection and Marlin weight-only fallback (#49376)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/features/quantization/modelopt.md (+14/-0)
LABELS: documentation, ready, build-docs
BODY: ## Purpose ⏎  ⏎ Users hitting slow NVFP4 performance have no docs explaining which GEMM kernel vLLM picked or why. This adds a short note to the ModelOpt guide covering: ⏎  ⏎ - Kernel selection happens automatically at load time, based on what the GPU supports. ⏎ - GPUs without a native FP4 GEMM kernel fall back to weight-only (W4A16) Marlin, which logs a warning and can cost throughput. ⏎ - `--linear-backend` overrides the choice, replacing the deprecated ` …[truncated]

### L3-7154856f3d  (L3, 2026-07-26, sha 7154856f3dcb, PR #47791)
TITLE: [Bugfix] Fix handling 5D KV cache in kv_postprocess_layout_on_receive (#47791)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/utils.py (+5/-1)
LABELS: bug, intel-gpu, ready, kv-connector
BODY: ## PR Purpose ⏎  ⏎ Fixes `IndexError` in `kv_postprocess_layout_on_receive()` for 5D blocks-first KV caches. ⏎  ⏎ `kv_postprocess_layout_on_receive()` (added in #30275) permutes a received KV cache from HND to NHD layout. It assumes the cache is 4D. But since #42095 is recently merged, non-MLA backends allocate a 5D cache `(num_blocks, 2, block_size, H, D)`. This causes the following error: ⏎  ⏎ ```bash ⏎ IndexError: index_copy_(): Source dimensionality …[truncated]

### L3-d742856610  (L3, 2026-07-26, sha d74285661064, PR #49502)
TITLE: [3/N][Core][KV Connector] Support reliable partial-tail KV offload for sub-block prompts (#49502)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py (+293/-0); tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py (+55/-0); tests/v1/kv_connector/unit/test_mooncake_store_hma_e2e.py (+244/-3); tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py (+231/-1); tests/v1/kv_connector/unit/test_mooncake_store_worker.py (+122/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py (+53/-13); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py (+27/-8); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py (+53/-2); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py (+193/-16); vllm/v1/core/kv_cache_manager.py (+62/-1); (+3 more)
LABELS: ready, v1, kv-connector
BODY: ## Purpose ⏎  ⏎ see https://github.com/vllm-project/vllm/issues/45702 ⏎  ⏎ Enable fine-grained prefix lookup and MooncakeStore offload when a prompt ends before the shared block boundary. Handle lazy hash sequences and local-only partial hits correctly so sub-block requests can reuse cached state without triggering an invalid remote load. ⏎  ⏎ ### `max_tokens=1` failure mode ⏎  ⏎ A producer's partial-tail marker previously became an offload handoff only when a l …[truncated]

### L3-fdaa0d9e59  (L3, 2026-07-27, sha fdaa0d9e5923, PR #49331)
TITLE: [ModelRunner V2] Support encoder-only attention (#49331)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: tests/models/language/pooling/test_embedding.py (+54/-0); vllm/v1/worker/gpu/attn_utils.py (+12/-0); vllm/v1/worker/gpu/block_table.py (+1/-1); vllm/v1/worker/gpu/model_runner.py (+3/-0); vllm/v1/worker/gpu/model_states/__init__.py (+7/-1); vllm/v1/worker/gpu/model_states/encoder_only.py (+182/-0); vllm/v1/worker/gpu/model_states/interface.py (+10/-0); vllm/v1/worker/gpu/warmup.py (+7/-2)
LABELS: ready, v1, mrv2
BODY: This adds encoder-only attention support to MRV2 in preparation for BERT/RoBERTa model pooling support. ⏎  ⏎ Intentionally isolates the encoder-only related changes to minimize changes to non-pooling files / code paths.

### L3-7f599d7854  (L3, 2026-07-27, sha 7f599d785468, PR #46913)
TITLE: [communication] [bugfix] fix quickreduce acc error in cudagraph mode (#46913)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/quickreduce/quick_reduce.h (+30/-8); tests/distributed/test_rocm_quick_reduce.py (+124/-0)
LABELS: bug, rocm, ready, nvidia
BODY: **1. cause:** ⏎ Once `flag_color` is fixed by `graph`, it remains unchanged for each round  ⏎ → The written flag value repeats in each round and cannot be distinguished from the residual value of the previous round ⏎ → The waiting party is prematurely satisfied by the old value and is immediately granted access  ⏎ → At this point, since the data for the current round has not yet been fully transmitted, the system reads the old data next, resulting in …[truncated]

### L3-50aa830482  (L3, 2026-07-27, sha 50aa83048219, PR #49751)
TITLE: [BugFix][MRV2] Don't create dummy requests longer than `max_model_len` (#49751)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_input_batch_v2.py (+52/-0); vllm/v1/worker/gpu/input_batch.py (+9/-4); vllm/v1/worker/gpu/lora_utils.py (+5/-1); vllm/v1/worker/gpu/model_runner.py (+6/-2)
LABELS: bug, ready, v1, mrv2
BODY: `InputBatch.make_dummy` and `GPUModelRunner._dummy_run` gave every dummy request `num_tokens // num_reqs` tokens and dumped the entire remainder on the last request, so a single dummy request could have `seq_len = query_len` up to `num_tokens - num_reqs + 1` tokens, far exceeding `max_model_len`. Such a request cannot be backed by the block tables (width `cdiv(max_model_len, block_size)`, alignment-padded): any attention kernel that runs on the d …[truncated]

### L3-ffc4f08c8e  (L3, 2026-07-27, sha ffc4f08c8ee1, PR #46116)
TITLE: [Core][KV-transfer] MoRIIO: heterogeneous TP<->DP prefill/decode read routing (#46116)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_moriio_routing_fairness.py (+210/-0); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py (+5/-0); vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py (+252/-7)
LABELS: documentation, structured-output, ready, v1, kv-connector, verified
BODY: Part of RFC #46107. ⏎  ⏎ ## Purpose ⏎  ⏎ MoRIIO today assumes the prefill and decode engines share the same parallelism layout. This PR enables **heterogeneous** disaggregated prefill/decode, where the two phases run different parallelism (TP vs DP+EP). The performance results in the RFC which encompasses this PR shows that heterogeneous parallel PD setups can achieve higher throughput and lower TTFT at high concurrency levels than their homogeneous  …[truncated]

### L3-394beb633b  (L3, 2026-07-27, sha 394beb633b0b, PR #49843)
TITLE: [Bugfix][ROCm] Use batch DMA for CPU KV cache loads (#49843)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_offload/cpu/test_gpu_worker.py (+14/-0); vllm/v1/kv_offload/cpu/gpu_worker.py (+2/-2)
LABELS: bug, rocm, ready, v1
BODY: - Add ROCm to the existing XPU batch-DMA fallback for CPU-to-GPU KV transfers. ⏎ - Keep CUDA, XPU, and every other prior dispatch path unchanged. ⏎  ⏎ The ROCm Triton path directly loaded a raw shared-mmap host pointer. The reproduced fault address matched the mmap base plus the selected block offset in the first case above the Triton descriptor threshold. ⏎  ⏎ Buildkite: [V1 Core + KV + Metrics](https://buildkite.com/vllm/amd-ci/builds/11196/list?jid …[truncated]

### L3-8061dc26bd  (L3, 2026-07-27, sha 8061dc26bd0b, PR #49392)
TITLE: [Bugfix] Normalize sparse MLA warmup compression ratios (#49392)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+1/-1); tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py (+23/-1)
LABELS: bug, ready, v1, deepseek
BODY: ## Summary ⏎  ⏎ Normalize DeepSeek V4 compression ratios before generating sparse-MLA Triton warmup keys, matching the runtime attention path. ⏎  ⏎ DeepSeek V4 configs use `0` to represent uncompressed/SWA-only layers. Runtime attention already converts those values with `max(1, ratio)`, but `BuildPrefillChunkMetadataKernel.get_warmup_keys()` used the raw config values. This generated a `COMPRESS_RATIO=0` specialization containing integer division by …[truncated]

### L3-81962bb699  (L3, 2026-07-27, sha 81962bb6995e, PR #49043)
TITLE: [Bugfix]Reject invalid FlashInfer MNNVL workspaces (#49043)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/distributed/device_communicators/flashinfer_all_reduce.py (+6/-0)
LABELS: bug, performance, ready, nvidia
BODY: ## Purpose ⏎  ⏎ Fixs: https://github.com/vllm-project/vllm/issues/49041 ⏎  ⏎ On topologies without symmetric-memory multicast support, FlashInfer can construct an MNNVL allreduce workspace whose mc_ptr is null. vLLM treated the workspace as initialized and passed it into fused allreduce kernels, which dereferenced the null multicast mapping and failed with a CUDA illegal memory access. ⏎  ⏎ Validate mc_ptr immediately after workspace creation. Destroy  …[truncated]

### L3-f0553889c0  (L3, 2026-07-27, sha f0553889c04a, PR #48366)
TITLE: [Bugfix] Prevent NaN poisoning in xpu_mla_sparse for fully-masked index chunks (#48366)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/xpu_mla_sparse.py (+5/-2); tests/kernels/attention/test_xpu_mla_sparse.py (+52/-0)
LABELS: bug, intel-gpu, ready, v1
ISSUES: #48364 [Bug]: xpu_mla_sparse NaN-poisons attention output when a row's leading topk index chunk is fully masked
BODY: FIX #48364 ⏎  ⏎ ## Purpose ⏎  ⏎ `_bf16_mla_sparse_kernel` (the XPU sparse-MLA kernel behind `XPU_MLA_SPARSE`, DeepSeek-V4 XPU prefill, and the fp8 decode wrapper) NaN-poisons its output whenever the first `BLOCK_N` (=16) topk index entries of a row are all masked, even though valid keys follow later: ⏎  ⏎ - the running max starts at `-inf`, and a fully-masked chunk sets every logit to `-inf`; ⏎ - `re_scale = exp2(-inf - -inf) = NaN` then poisons `acc` a …[truncated]

### L3-0906123953  (L3, 2026-07-27, sha 0906123953a5, PR #48841)
TITLE: [ROCm] [Model] Enable TML inkling (#48841)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_inkling_qkvr_prep.py (+176/-0); tests/models/inkling/rocm/conftest.py (+17/-0); tests/models/inkling/rocm/test_model_alignment.py (+59/-0); tests/models/inkling/rocm/test_mxfp4_load.py (+67/-0); tests/models/inkling/rocm/test_rel_attention.py (+425/-0); tests/models/inkling/rocm/test_rocm_mtp_input_fusion.py (+116/-0); tests/models/inkling/rocm/test_sconv_cache_layout.py (+20/-0); vllm/models/inkling/__init__.py (+30/-9); vllm/models/inkling/amd/__init__.py (+2/-0); vllm/models/inkling/amd/attention.py (+328/-0); (+23 more)
LABELS: performance, new-model, rocm, ready, needs-rebase, ci/build, v1, tool-calling, nvidia, rust
BODY: ## Purpose ⏎  ⏎ This is the first enablement PR for inkling model on ROCm. This PR also targets the more efficient variant of the model which is mxfp4. There are possible path for loading and use nvfp4 weight, however, we could only support it through inefficient emulation code path.  ⏎  ⏎ The weight is generated and distributed by https://huggingface.co/lightseekorg/Inkling-MXFP4 instead.  ⏎  ⏎ There is a continuous effort to try to enable to BF16 wei …[truncated]

### L3-ac87549cbd  (L3, 2026-07-27, sha ac87549cbdff, PR #49916)
TITLE: [CI][ROCm] Reduce V1 attention test runtime (#49916)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+6/-20); tests/v1/attention/test_mla_backends.py (+60/-31)
LABELS: rocm, ready, ci/build, v1
BODY: - Remove V1 attention from MI250 and split its MI300 and MI355 coverage into two pytest shards. ⏎ - Classify unsupported MLA prefill combinations during collection and include the ROCm AITER FA prefill adapter. ⏎  ⏎ https://buildkite.com/vllm/amd-ci/builds/11266/list?sid=019f9c06-99b2-4356-8a8f-9cf9e905a157&tab=output

### L3-92e8518d37  (L3, 2026-07-27, sha 92e8518d376c, PR #49957)
TITLE: Improve Transformers modelling backend `fx` tracer (#49957)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/transformers/fusers/test_linear.py (+3/-4); tests/models/transformers/fusers/test_rms_norm.py (+11/-6); vllm/model_executor/models/transformers/base.py (+4/-5); vllm/model_executor/models/transformers/fuser.py (+17/-9); vllm/model_executor/models/transformers/fusers/base.py (+6/-19); vllm/model_executor/models/transformers/fusers/glu.py (+4/-8); vllm/model_executor/models/transformers/fusers/qkv.py (+17/-19); vllm/model_executor/models/transformers/fusers/rms_norm.py (+7/-12); vllm/model_executor/models/transformers/fx_utils.py (+238/-46)
LABELS: ready
BODY: - Trace with meta tensors: every traced op also runs on meta tensors, so `len`, iteration and `.shape` unpacks come from PyTorch's meta kernels instead of the hand-written rules in `_infer_len`/`_rank`. ⏎ - Record untraceable attention interfaces as opaque leaf calls (flagged in the node's `meta`, with the declared return length). `vllm_attention_forward` used to end the trace early; the graph now reaches the module's output. ⏎ - Trace with `past_key …[truncated]

### L3-a89015c6df  (L3, 2026-07-27, sha a89015c6df8e, PR #48739)
TITLE: [Perf] Make merge attention context count a runtime argument (#48739)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.merge.triton_lse
FILES: vllm/v1/attention/ops/triton_merge_attn_states.py (+2/-2)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Makes `merge_attn_states_kernel`'s batch-varying `prefill_tokens_with_context` a runtime argument instead of a `tl.constexpr`, preventing a new Triton specialization for each distinct value. Partial fix for #48650. ⏎  ⏎ ## Why this is not a duplicate ⏎  ⏎ Rechecked on 2026-07-15 immediately before preparing this draft: ⏎  ⏎ - #48650 is open, unassigned, with no comments. ⏎ - PR #48734 now implements the issue's `count_expert_num_tokens` `BLOCK_SIZE …[truncated]

### L3-77cba0259f  (L3, 2026-07-27, sha 77cba0259f21, PR #48123)
TITLE: [KV Offloading] Per-request tier filtering with TierFilter/TierMatcher (#48123)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/offloading_connector/test_events.py (+11/-13); tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py (+4/-3); tests/v1/kv_offload/cpu/test_manager.py (+2/-2); tests/v1/kv_offload/tiering/test_fs_tier.py (+3/-4); tests/v1/kv_offload/tiering/test_obj_tier.py (+2/-2); tests/v1/kv_offload/tiering/test_tiering_offloading.py (+136/-3); vllm/distributed/kv_events.py (+1/-2); vllm/distributed/kv_transfer/kv_connector/v1/offloading/events.py (+17/-5); vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py (+60/-1); vllm/v1/kv_offload/base.py (+46/-9); (+6 more)
LABELS: ready, v1, kv-connector
BODY: ## Purpose ⏎  ⏎ Adds per-request tier filtering to KV cache offloading. ⏎ Requests can specify which secondary tiers to load from via `kv_transfer_params["kv_load_tiers"]`, filtering by medium and locality. ⏎  ⏎ See detailed design discussion: https://github.com/vllm-project/vllm/pull/48123#issuecomment-4979058299 ⏎  ⏎  ⏎ Key changes: ⏎ - `Medium` enum: `CPU`/`STORAGE` (coarse granularity, both FS and OBJ → STORAGE) ⏎ - `TierMatcher(NamedTuple)` with optio …[truncated]

### L3-59a6b0411d  (L3, 2026-07-27, sha 59a6b0411d18, PR #49204)
TITLE: [Core] Fix internal LB load-balancing (#49204)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: rust/src/engine-core-client/src/client/state.rs (+4/-8); tests/v1/engine/test_engine_core_client.py (+82/-11); vllm/v1/core/sched/interface.py (+4/-0); vllm/v1/core/sched/scheduler.py (+4/-0); vllm/v1/engine/coordinator.py (+40/-29); vllm/v1/engine/core.py (+23/-3); vllm/v1/engine/core_client.py (+31/-8)
LABELS: ready, v1, rust
ISSUES: #48808 [Bug]: DP request distribution becomes imbalanced under long-context workload on H20 GPU
BODY: Previous PR https://github.com/vllm-project/vllm/pull/30739 added support for efficient DP without EP. The coordinator was kept to propagate engine queue stats for loadbalancing, but there was an omission whereby these stats were not actually getting published, which could result in significant imbalance between the ranks. ⏎  ⏎ As well as fixing this (by also calling `_maybe_publish_request_counts` in the `EngineCoreProc` superclass).  I also ran m …[truncated]

### L3-b2f9e4caa4  (L3, 2026-07-27, sha b2f9e4caa494, PR #50004)
TITLE: [DSv4 Perf] Adaptive topk width, 1.0% E2E throughput improvement (#50004)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_flashmla_sparse.py (+37/-0); vllm/models/deepseek_v4/sparse_mla.py (+19/-7)
LABELS: ready
DEEP_STUDY: deep-study: this PR was reverted by PR 51318 (confirmed_revert, reason=correctness_or_accuracy) || deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Let's assume token num = 2 ⏎  ⏎ Originally: Always use maximum length ⏎  ⏎ ```bash ⏎ c128a_max_compressed ⏎ → 1,048,576 / 128 ⏎ → 8,192 ⏎  ⏎ So buffer ⏎  ⏎ global_decode_buffer[:2] ⏎ → shape = [2, 8192] ⏎ → stride = 8192 ⏎  ⏎ kernel loop for 8192 times ⏎ ``` ⏎  ⏎ Now based on current context ⏎  ⏎ ```bash ⏎ 32,768 / 128 ⏎ → 256 ⏎ → active_topk_width = 256 ⏎  ⏎ buffer.view(-1)[: 2 × 256] ⏎ → view(2, 256) ⏎ → shape = [2, 256] ⏎ → stride = 256 ⏎  ⏎ Kernel only loops …[truncated]

### L3-b5bcb3ce88  (L3, 2026-07-27, sha b5bcb3ce881e, PR #49745)
TITLE: [Refactor] Remove dead code in multiple files (#49745)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+0/-31); vllm/model_executor/kernels/linear/cute_dsl/_ll_bf16_dotprod.py (+0/-4); vllm/model_executor/layers/fused_moe/prepare_finalize/nixl_ep.py (+0/-5); vllm/model_executor/layers/quantization/fp8.py (+0/-10); vllm/model_executor/layers/quantization/modelopt.py (+0/-87); vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py (+0/-13); vllm/model_executor/models/funaudiochat.py (+0/-19); vllm/model_executor/models/hyperclovax_vision.py (+0/-32); vllm/model_executor/models/idefics3.py (+0/-24); vllm/model_executor/models/llava_onevision2.py (+0/-4); (+12 more)
LABELS: rocm, tpu, ready, v1, tool-calling, qwen, quantization
BODY: ## Purpose ⏎  ⏎ Remove dead code in multiple files

### L3-eb290ab673  (L3, 2026-07-27, sha eb290ab673c8, PR #49591)
TITLE: [Bugfix][CPU] Zero-pad MoE intermediate size for grouped-gemm TP alignment (#49591)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_cpu_fused_moe.py (+130/-0); vllm/model_executor/layers/fused_moe/cpu_fused_moe.py (+81/-14)
LABELS: bug, ready, cpu
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:ed13deb376 (fix PR 49985)
BODY: ## Purpose ⏎  ⏎ The CPU AMX/vector grouped-gemm MoE kernels (`csrc/cpu/cpu_fused_moe.cpp`) tile the expert intermediate dimension in fixed 32-wide blocks with no tail/remainder handling. `CPUFusedMOE.check_grouped_gemm` only enables the fast kernel when the per-partition `moe_intermediate_size` (i.e. `moe_intermediate_size // tp_size`) is a multiple of 32; otherwise it silently falls back to `cpu_fused_moe_torch`, a per-expert Python loop. ⏎  ⏎ For `goog …[truncated]

### L3-bf2b45b5d6  (L3, 2026-07-27, sha bf2b45b5d6a9, PR #42669)
TITLE: [Attention] Integrate FlashAttention 4 SM100 headdim 256 support (#42669)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fa_utils
FILES: vllm/v1/attention/backends/fa_utils.py (+17/-5); vllm/v1/attention/backends/flash_attn.py (+1/-0); vllm/v1/attention/backends/mla/prefill/flash_attn.py (+17/-20); vllm/v1/attention/backends/mla/prefill/selector.py (+17/-1); benchmarks/attention_benchmarks/benchmark.py (+4/-0); docs/design/attention_backends.md (+3/-1); tests/v1/attention/test_mla_prefill_selector.py (+7/-7)
LABELS: documentation, performance, ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ FA4 added support for headdim 256 in https://github.com/Dao-AILab/flash-attention/pull/2412 (and optimized it in https://github.com/Dao-AILab/flash-attention/pull/2487 and https://github.com/Dao-AILab/flash-attention/pull/2488). This support was integrated into vLLM with #41052. This PR is a follow-up to activate the headdim 256 support. ⏎  ⏎ > [!NOTE] ⏎ > This kernel is currently slower than `TRTLLM_RAGGED`. Regardless, right now on mai …[truncated]

### L3-99de48e98f  (L3, 2026-07-27, sha 99de48e98fe9, PR #49982)
TITLE: Fix MLA padding and grouped topk routing in the Transformers modelling backend (#49982)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/models/transformers/fusers/test_moe.py (+80/-3); vllm/model_executor/models/transformers/__init__.py (+12/-1); vllm/model_executor/models/transformers/fusers/moe.py (+55/-22); vllm/model_executor/models/transformers/moe.py (+73/-13); vllm/model_executor/models/transformers/utils.py (+7/-0)
LABELS: ready
BODY: - Anchor the router's scorer on the *last* `topk` in the gate's graph — DeepSeek's group-limited routers `topk` over expert groups first, so the scorer was sought upstream of the wrong node and the block was declined. ⏎ - Allow `e_score_correction_bias` in the gate's state (DeepSeek-V3 noaux) and carry it onto the rebuilt gate, rather than rejecting any extra state. ⏎ - Pass `use_grouped_topk`, `num_expert_group`, `topk_group`, `scoring_func` and `e_ …[truncated]

### L3-2b465b2c42  (L3, 2026-07-27, sha 2b465b2c42e6, PR #49988)
TITLE: [Misc][PD] Nixl cleanup `get_backend_aware_kv_block_len` and `virtually_split_kv_in_blocks` (#49988)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector.py (+0/-12); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+44/-73)
LABELS: ready, v1, kv-connector
BODY: Following up on https://github.com/vllm-project/vllm/pull/44456/ with some more nixl-specific cleanup. ⏎ Now that KV are packed and we can safely assume KV regions are registered together, a few things can be simplified. ⏎ - `virtually_split_kv_in_blocks` special handling is dead-code, as it resolves to being always False now ⏎ - we can also bid `get_backend_aware_kv_block_len` farewell, as the clumsy logic (that once used to be) can now just be inl …[truncated]

### L3-28158b2fc3  (L3, 2026-07-27, sha 28158b2fc364, PR #48886)
TITLE: [ROCm] [BugFix] Fix Quark GLM-5.2 Checkpoint inference: indexer wk per-channel FP8 dequant + missing sparse-MLA metadata fields (#48886)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.rocm_aiter_sparse, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/mla_attention.py (+26/-19); vllm/model_executor/models/deepseek_v2.py (+7/-2); vllm/v1/attention/backend.py (+4/-0); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+17/-0); tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py (+75/-0)
LABELS: bug, rocm, ready, v1, deepseek, quantization
DEEP_STUDY: deep-study correctness case vllm:28158b2fc3: class=integration_backend_cudagraph; symptom=crash_or_exception; introducing=#47327
BODY: ## Summary ⏎  ⏎ This PR addresses two independent bugs blocking quark quantized GLM-5.2 checkpoints with Attn quantized to PTPC FP8 from running end-to-end on ROCm (MI355X / gfx950).  ⏎  ⏎ ### Per-channel FP8 scale for fused indexer `wk` (`deepseek_v2.py`): ⏎  ⏎ `_try_load_fp8_indexer_wk` dequantizes the FP8 indexer `wk` weight to BF16 at load time so it can be fused with `weights_proj`. However, it unconditionally reads `scale_inv.shape[1]`, assuming  …[truncated]

### L3-ebcef33766  (L3, 2026-07-27, sha ebcef33766f8, PR #49987)
TITLE: Fix MQA with tensor parallelism on transformers modeling backend   (#49987)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/transformers/fusers/test_linear.py (+162/-1); vllm/model_executor/models/transformers/fuser.py (+4/-3); vllm/model_executor/models/transformers/fusers/__init__.py (+8/-1); vllm/model_executor/models/transformers/fusers/base.py (+45/-34); vllm/model_executor/models/transformers/fusers/packed_qkv.py (+163/-0); vllm/model_executor/models/transformers/fusers/qkv.py (+5/-2); vllm/model_executor/models/transformers/fx_utils.py (+9/-3)
LABELS: ready
BODY: ## Purpose ⏎ Fix k and v replication for Multi-query attention on tp>1 by adding a fuser, to enable models like bigcode/starcoder to run with transformers modeling backend on multi gpu setup. This problem was caused by migration of starcoder to transformers backend in this pr: #30966  ⏎  ⏎ ## Test Plan ⏎ `vllm serve bigcode/starcoder --dtype bfloat16 --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.85 --port 8000 --trust-remot …[truncated]

### L3-1e34a13539  (L3, 2026-07-27, sha 1e34a135390b, PR #49096)
TITLE: Fix Humming non-gated MoE (#49096)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_moe.py (+96/-0); vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py (+11/-3); vllm/model_executor/layers/quantization/utils/humming_utils.py (+4/-4)
LABELS: bug, ready, quantization
BODY: Enables Humming MoE for NemotronH models with non-gated squared-ReLU experts.  ⏎ Humming previously assumed every `w13` contained two gated projections, producing incorrect weight and buffer shapes. ⏎  ⏎ ### Results for [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4) ⏎  ⏎ | Backend    | Output tok/s |         TTFT | GSM8K             | ⏎ |------------|-------------:|-------------:|------ …[truncated]

### L3-61ac368021  (L3, 2026-07-28, sha 61ac36802103, PR #50090)
TITLE: [Kimi-K3] Add AttnRes kernels (#50090)
SOURCES: corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: .buildkite/test_areas/models_basic.yaml (+12/-0); CMakeLists.txt (+22/-0); csrc/libtorch_stable/kimi_k3/attn_res_kernel.cu (+954/-0); csrc/libtorch_stable/ops.h (+11/-0); csrc/libtorch_stable/torch_bindings.cpp (+11/-0); tests/models/kimi_k3/test_amd_attn_res.py (+102/-0); tests/models/kimi_k3/test_attn_res.py (+193/-0); vllm/_custom_ops.py (+27/-0); vllm/models/kimi_k3/amd/__init__.py (+2/-0); vllm/models/kimi_k3/amd/ops/__init__.py (+0/-0); (+4 more)
LABELS: ready, ci/build, kimi, k3
BODY: ## Purpose ⏎  ⏎ #50000. Add AttnRes kernels ⏎ - Specialized CUDA C++ kernel for NVIDIA sm100 ⏎ - Triton for NVIDIA fallback and AMD ⏎  ⏎ This PR also adds Buildkite CI for K3 ⏎  ⏎ ## Test Plan ⏎  ⏎ ```bash ⏎ # NVIDIA ⏎ pytest -v tests/models/kimi_k3/test_attn_res.py ⏎  ⏎ # AMD ⏎ pytest -v tests/models/kimi_k3/test_amd_attn_res.py ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-247470f23a  (L3, 2026-07-28, sha 247470f23a4c, PR #48164)
TITLE: [CI] Add PyTorch stable ABI audit check (#48164)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/check-torch-abi.py (+102/-0); .buildkite/ci_config.yaml (+1/-0); .buildkite/test_areas/torch_abi.yaml (+14/-0); requirements/test/cpu.txt (+6/-0); requirements/test/cuda.in (+1/-1); requirements/test/cuda.txt (+6/-0)
LABELS: ready, ci/build, cpu, nvidia
BODY: ## Purpose ⏎ - Add a new Buildkite step, **Torch Stable ABI Audit**, that runs after the CUDA CI image build and uses [`torch-abi-audit`](https://github.com/Quansight/torch-abi-audit) to verify vLLM's compiled extensions comply with the PyTorch stable ABI. ⏎ - Fail CI if any extension links unstable libtorch symbols (`at::` / `c10::` / etc.) unless it is listed in `ALLOWED_UNSTABLE_LIBRARIES` in `.buildkite/check-torch-abi.py`. ⏎ - Fail CI if an all …[truncated]

### L3-f472ab0a4c  (L3, 2026-07-28, sha f472ab0a4c09, PR #49621)
TITLE: Remove triton per group quant [ROCm] [Bugfix] (#49621)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/passes/distributed/test_fusion_all_reduce.py (+6/-25); tests/compile/passes/test_fusion.py (+0/-2); tests/compile/passes/test_silu_mul_quant_fusion.py (+1/-10); vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+1/-2); vllm/model_executor/layers/quantization/input_quant_fp8.py (+0/-5); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+0/-34)
LABELS: bug, rocm, ready, quantization
BODY: ## Purpose ⏎  ⏎   On ROCm, dynamic FP8 **per-group** (block-scale) activation quant was routed through a ⏎   `vllm.triton_per_token_group_quant_fp8` custom op whenever the block-scaled linear kernel ⏎   chose its tuned Triton GEMM path (`AiterFp8BlockScaledMMKernel`, `use_triton=True`). That ⏎   op is a thin wrapper that already dispatches to the C++ `_C.per_token_group_fp8_quant` ⏎   kernel on ROCm (#42758) — so it adds no compute, but it **blocks RMS …[truncated]

### L3-9b9fc4039c  (L3, 2026-07-28, sha 9b9fc4039c25, PR #45841)
TITLE: add epilogue hook to flex attention (#45841)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+9/-0)
LABELS: documentation, ready, v1
BODY: adding optional post-attention epilogue transform to flex attention ([this](https://github.com/pytorch/torchtitan/pull/3721) is an example of how someone could use this)

### L3-601fa9a74e  (L3, 2026-07-28, sha 601fa9a74e8c, PR #49612)
TITLE: [KV Connector] Support NIXL heterogeneous P/D block sizes for hybrid models (#49612)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector.py (+6/-4); tests/v1/kv_connector/unit/test_nixl_connector_hma.py (+114/-0); tests/v1/kv_connector/unit/test_nixl_desc_geometry.py (+619/-0); tests/v1/kv_connector/unit/test_tp_mapping.py (+54/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+207/-64); vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py (+7/-28); vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py (+7/-15)
LABELS: ready, v1, kv-connector
ISSUES: #41037 [Bug] _align_hybrid_block_size produces TP-dependent block sizes, currently unsupported when local and remote kernel block size mismatch
BODY: Hybrid (mamba) models previously asserted out heterogeneous block sizes entirely — yet they are the models where P/D block sizes most readily diverge, since the mamba-padded attention block size varies with TP sharding (#41037). Lift the restriction: ⏎  ⏎ - Mamba state blocks are indivisible, so their descriptors always use local page geometry and their desc ids are never ratio-expanded; attention descriptors remain expanded to remote-block granulari …[truncated]

### L3-a8f296083f  (L3, 2026-07-28, sha a8f296083f43, PR #49858)
TITLE: [KV Offload] Make compact secondary identity TP-independent (#49858)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_offload/test_file_mapper.py (+105/-0); tests/v1/kv_offload/tiering/test_fs_tier.py (+61/-5); tests/v1/kv_offload/tiering/test_obj_tier.py (+50/-5); vllm/v1/kv_offload/file_mapper.py (+10/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ This completes the secondary-tier step scoped in #47929. #48906 made the single-node TP-only MLA primary row compact, but secondary-tier identity still included the TP configuration: FS/OBJ paths remained TP-specific, and P2P derived its compatibility fingerprint from the same fields. ⏎  ⏎ This PR: ⏎  ⏎ - admits `replicated_layout` rows to the existing caller-opted-in parallel-agnostic `FileMapper` path; ⏎ - retains a `replicated_layout=true` i …[truncated]

### L3-0d0504b54c  (L3, 2026-07-28, sha 0d0504b54c73, PR #49903)
TITLE: [Core] Warm up runner-owned Triton kernels before the first request (#49903)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_kv_block_zeroer.py (+65/-1); vllm/model_executor/warmup/kernel_warmup.py (+10/-6); vllm/model_executor/warmup/qwen_triton_warmup.py (+0/-114); vllm/model_executor/warmup/v1_block_table_warmup.py (+14/-28); vllm/v1/worker/gpu/warmup.py (+85/-36); vllm/v1/worker/mamba_utils.py (+3/-3); vllm/v1/worker/utils.py (+6/-1)
LABELS: ready, v1
BODY: Several runner-owned Triton kernels only compiled once a real request ⏎ arrived, spiking first-token latency. Each takes a runtime (non-constexpr) ⏎ integer that Triton specializes into `== 1`, `% 16 == 0` and neither, so ⏎ warming a single shape was not enough. ⏎  ⏎ - `_zero_kv_blocks_kernel` is driven by the scheduler's ⏎   `new_block_ids_to_zero`, which no dummy or warmup step reaches. Add ⏎   `KVBlockZeroer.warmup()` covering all three `n_blocks` variants, …[truncated]

### L3-02b6ecf07c  (L3, 2026-07-28, sha 02b6ecf07cf6, PR #44527)
TITLE: [ROCm][DSv3.2] Eliminate per-decode FillFunctor launches in sparse-MLA hot loop (#44527)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+0/-2)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ On the ROCm sparse-MLA decode path, two `FillFunctor<T>` kernels fire on every indexer-bearing decode step with no observable output — pure launch + bandwidth overhead: ⏎  ⏎ 1. `FillFunctor<float>` — `out_logits.fill_(float("-inf"))` in `rocm_fp8_paged_mqa_logits`. Upstream added a persistent `current_workspace_manager()` workspace for `out_logits` (pre-filled once on allocation), but left the per-call `fill_` in place. The downstream …[truncated]

### L3-05a0814863  (L3, 2026-07-28, sha 05a08148631a, PR #49906)
TITLE: [ROCm] Fix and optimize GPT-J-style MRoPE (#49906)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/core/test_mrope.py (+7/-3); vllm/model_executor/layers/rotary_embedding/mrope.py (+93/-56)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: - Honor the existing adjacent-pair MRoPE setting instead of hardcoding NeoX pairing. ⏎ - Use contiguous ROCm loads and stores with register-only split/interleave, preserving the NeoX and non-ROCm launch paths. ⏎  ⏎ https://buildkite.com/vllm/amd-ci/builds/11254/list?sid=019f966b-1ec6-4bee-8ce6-acf444a98979&tab=output

### L3-6c7e679f04  (L3, 2026-07-28, sha 6c7e679f048d, PR #49714)
TITLE: [ROCm][Bugfix] Sanitize AITER paged-MQA logits before sparse top-k for DeepSeek-V4 (#49714)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+1/-0); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+64/-0)
LABELS: bug, rocm, ready, v1, deepseek
ISSUES: #13 [ROCm][DeepSeek V4] TP8 graph-mode GPU memory fault under 64 concurrent requests
BODY: ## Purpose ⏎  ⏎ Fix: https://github.com/Fangzhou-Ai/vllm/issues/13 ⏎  ⏎ ### Summary ⏎  ⏎ This change prevents a GPU memory access fault when serving DeepSeek V4 with ROCm sparse attention on gfx950 under concurrent mixed prefill/decode workloads. ⏎  ⏎ The ROCm AITER `deepgemm_fp8_paged_mqa_logits` kernel can intermittently return `NaN` or infinite values within the valid logits range. The generic histogram top-k kernel assumes finite, orderable inputs. N …[truncated]

### L3-d18ed2304a  (L3, 2026-07-28, sha d18ed2304a27, PR #49152)
TITLE: [KV-offload][FS] : Batch store/load_block in C  (#49152)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/fs_io.cpp (+283/-1); tests/v1/kv_offload/tiering/test_fs_tier.py (+67/-32); vllm/v1/kv_offload/tiering/fs/io.py (+80/-2); vllm/v1/kv_offload/tiering/fs/manager.py (+22/-23)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ We have pools of python threads to read and write KV files from/to disk. These python threads compete for the GIL and submit bursty read/write commands to the disk. This prevents the disk from reaching 100% utilization.  ⏎  ⏎ ### Changes:  ⏎  1. Add C implementations for store_block and load_block that does the heavy-lifting inside a GIL-free region. Note that we still use the python thread pool, the optimization is just that the work do …[truncated]

### L3-4f56321d7e  (L3, 2026-07-28, sha 4f56321d7ec2, PR #47773)
TITLE: [ROCm] Cache fp32 upcast of static e8m0 weight scale in AITER scaled_mm (#47773)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/scaled_mm/aiter.py (+16/-8)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ The e8m0 weight scale in the AITER block-scaled GEMM path is static, but ⏎ `apply_block_scaled_mm` re-runs the `<<23` bit-shift fp32 upcast plus ⏎ `.contiguous()` on it **every decode step, for every layer**, even though the ⏎ value never changes. ⏎  ⏎ On DeepSeek-V4 FP4 (MI355X / gfx950) this shows up in profiles as ~3.5% of GPU ⏎ time split across `aten::__lshift__` and `direct_copy`. ⏎  ⏎ This PR overrides `process_weights_after_loading` …[truncated]

### L3-98e91a9600  (L3, 2026-07-28, sha 98e91a9600eb, PR #49345)
TITLE: [PD][NixlPush] Skip extra `add_remote_agent` step in D->P handshake (#49345)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+2/-1); vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py (+4/-2)
LABELS: ready, kv-connector
BODY: Address S4 in https://github.com/vllm-project/vllm/issues/48633. ⏎  ⏎ In the workflow described here https://docs.vllm.ai/en/stable/design/nixl_kv_push_connector , when D handshakes with P with the purpose of forwarding block_ids for P to write to, D executes a "full handshake", which includes exchanging kv topology and addresses to register local/remote descs here ⏎ https://github.com/vllm-project/vllm/blob/47f1b47a7396c5ba627001b2f7a1599258a6c856/ …[truncated]

### L3-ba702e978e  (L3, 2026-07-28, sha ba702e978e3b, PR #48407)
TITLE: [Attention] Skip sparse indexer scoring for dense short prefills (#48407)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+6/-7); vllm/model_executor/layers/attention/sparse_mla_attention.py (+4/-0); vllm/model_executor/layers/mla.py (+22/-1); tests/model_executor/layers/test_mla_short_prefill_indexer.py (+165/-0); vllm/model_executor/layers/sparse_attn_indexer.py (+26/-2); vllm/model_executor/models/deepseek_v2.py (+3/-0); vllm/models/deepseek_v32/nvidia/attention.py (+4/-2)
LABELS: documentation, performance, rocm, intel-gpu, ready, v1, deepseek, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Try extend #47327 to see some improvement. ⏎  ⏎ This PR removes dead indexer scoring work before dense attention. ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ MODEL_NAME=zai-org/GLM-5.2-FP8 ⏎ export VLLM_USE_V2_MODEL_RUNNER=1 ⏎ export VLLM_USE_BREAKABLE_CUDAGRAPH=1 ⏎ export VLLM_MOE_SKIP_PADDING=1 ⏎ ``` ⏎ ``` ⏎ vllm serve "$MODEL_NAME" \ ⏎   -tp 8 \ ⏎   --kv-cache-dtype fp8 \ ⏎   --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'  ⏎ ``` ⏎ ``` ⏎ vllm  …[truncated]

### L3-88402a41c4  (L3, 2026-07-28, sha 88402a41c4ab, PR #49945)
TITLE: [Test] Skip ROCm AITER MLA prefill tests on non-ROCm platforms (#49945)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_mla_prefill_selector.py (+6/-0)
LABELS: rocm, intel-gpu, ready, v1
BODY: TestROCmAiterFAPrefillSelection exercises ROCm-specific AITER FlashAttention MLA prefill backend gating. On non-ROCm platforms (e.g. XPU) patching vllm.platforms.rocm forces its module-level import, which runs torch.cuda.get_device_properties and fails when torch is not built with CUDA. Guard the class with a ROCm-only skipif.

### L3-32a423ac0a  (L3, 2026-07-28, sha 32a423ac0aad, PR #49580)
TITLE: Integrate CuTeDSL MoE for ReLU2 NVFP4 (#49580)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer_cutedsl_nvfp4_moe.py (+230/-0); vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py (+6/-2); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+9/-7)
LABELS: ready, nvidia, quantization
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: GSM8K passed at 0.9431 vs 0.9300 for Super NVFP4 model. ⏎  ⏎ Related reference: ⏎ - SGLang FlashInfer one-sided A2A + CuTeDSL MoE integration for Nemotron Ultra: https://github.com/sgl-project/sglang/pull/28309 ⏎  ⏎ ## A2A serving benchmark ⏎  ⏎ Ran an A2A comparison for `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` with CUDA graphs enabled, FlashInfer autotune enabled. ⏎  ⏎ Workload/config: ⏎  ⏎ - GPUs: 4 total, `--data-parallel-size 2`, `--tensor-paral …[truncated]

### L3-56f31af62a  (L3, 2026-07-29, sha 56f31af62afe, PR #41602)
TITLE: [Bugfix] Fix /wake_up crash on hybrid models (Mamba/DeltaNet) (#41602)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_model_runner.py (+56/-0); vllm/v1/worker/gpu_model_runner.py (+10/-3)
LABELS: bug, ready, v1, mrv1-only
ISSUES: #41564 [Bug]: /wake_up fails with "'list' object has no attribute 'zero_'" on hybrid-SWA / Mamba / DeltaNet models (SM120, NVFP4) — only Gemma-4 interleaved-SWA survives
BODY: ## Purpose ⏎  ⏎ Fix `AttributeError: 'list' object has no attribute 'zero_'` when calling `/wake_up` after `/sleep` on hybrid models that use Mamba, DeltaNet, or other `MambaSpec`-based layers with quantized KV cache. ⏎  ⏎ Fixes #41564 ⏎  ⏎ ## Root Cause ⏎  ⏎ `init_fp8_kv_scales()` (called via `post_kv_cache_wake_up()` → `wake_up()`) iterates `self.kv_caches` and calls `.zero_()` on each entry, assuming every entry is a `torch.Tensor`: ⏎  ⏎ ```python ⏎ for cache_tenso …[truncated]

### L3-dc1be79031  (L3, 2026-07-29, sha dc1be79031d9, PR #49114)
TITLE: Add CachePolicyFactory for pluggable/external eviction policies (#49114)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/features/kv_offloading_usage.md (+32/-1); tests/v1/kv_offload/cpu/__init__.py (+0/-0); tests/v1/kv_offload/cpu/policies/__init__.py (+0/-0); tests/v1/kv_offload/cpu/policies/test_factory.py (+111/-0); tests/v1/kv_offload/cpu/test_manager.py (+2/-0); vllm/v1/kv_offload/base.py (+0/-8); vllm/v1/kv_offload/cpu/manager.py (+9/-16); vllm/v1/kv_offload/cpu/policies/arc.py (+1/-1); vllm/v1/kv_offload/cpu/policies/base.py (+2/-2); vllm/v1/kv_offload/cpu/policies/factory.py (+87/-0); (+5 more)
LABELS: documentation, ready, v1
BODY: ## Purpose ⏎  ⏎ `CPUOffloadingManager` resolves `eviction_policy` ("lru"/"arc") through a private, ⏎ closed dict in `manager.py` with no way for an out-of-tree package to plug in ⏎ its own `CachePolicy` implementation by name, the same way `OffloadingSpecFactory` ⏎ and `KVConnectorFactory` already let external specs/connectors register themselves. ⏎  ⏎ Without this, the only way to use a custom `CachePolicy` is to construct `CPUOffloadingManager` with ⏎  …[truncated]

### L3-6370e53f24  (L3, 2026-07-29, sha 6370e53f246c, PR #48145)
TITLE: [Frontend] Reuse prefill token ids on the decode chat path for disaggregated serving (#48145)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+1/-1); .buildkite/test_areas/rust_frontend.yaml (+1/-1); docs/features/disagg_prefill.md (+28/-0); tests/entrypoints/openai/chat_completion/test_chat_completion.py (+89/-0); tests/entrypoints/openai/chat_completion/test_serving_chat.py (+31/-0); vllm/renderers/online_renderer.py (+48/-11)
LABELS: documentation, frontend, ready, ci/build, rust
BODY: ## Why ⏎  ⏎ In prefill and decode disaggregation, the prefill stage renders the prompt from `messages` and tokenizes it. The router forwards the same chat request to the decode stage, which renders and tokenizes it a second time. For long prompts, that repeated render and tokenize adds latency on the decode critical path. ⏎  ⏎ The decode stage does not need to redo it. The prefill response already contains the token ids, and the router already forwards s …[truncated]

### L3-f37f03db4a  (L3, 2026-07-29, sha f37f03db4af6, PR #49762)
TITLE: [KV Connector] Support NIXL P/D for hybrid MLA+SSM models  (#49762)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector_hma.py (+196/-0); tests/v1/kv_connector/unit/test_nixl_desc_geometry.py (+21/-0); tests/v1/kv_connector/unit/test_nixl_push_connector.py (+4/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+25/-5); vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py (+35/-33)
LABELS: ready, v1, kv-connector
BODY: Re-implements the intent of https://github.com/vllm-project/vllm/pull/44848 on top of the reworked NIXL connector, from the current architecture rather than the original patch. KimiLinear pools its KDA (GDN-typed MambaSpec) and MLA layers into shared HMA tensors, making every region dual-purpose. Since the mamba-page unification raises the attention block size until the MLA page equals the unified page, and FlashMLA fixes the kernel block at 64 t …[truncated]

### L3-542a8fad6d  (L3, 2026-07-29, sha 542a8fad6da0, PR #50094)
TITLE: [KV Offload] Move CPUOffloadingSpec onto SharedOffloadRegion (#50094)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_offload/test_factory.py (+111/-4); vllm/v1/kv_offload/cpu/spec.py (+21/-1)
LABELS: ready, v1
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:40b40d1d39 (fix PR 51081)
BODY: ## Purpose ⏎  ⏎ Move the default KV-offload backend `CPUOffloadingSpec`'s worker-side CPU buffer from a per-rank private pinned `torch` tensor onto the existing shared `SharedOffloadRegion` mmap, on CUDA/ROCm. This is the allocation-swap prerequisite for the TP-deduplication feature requested in #47929; it introduces **no** deduplication semantics on its own. ⏎  ⏎ Maintainer authorization (upstream reviewers do not see the issue thread, so the requests a …[truncated]

### L3-6f00a1ae3b  (L3, 2026-07-29, sha 6f00a1ae3bd4, PR #42436)
TITLE: fused_moe: add VLLM_TRITON_USE_TD tensor-descriptor path (#42436)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_moe.py (+24/-7); vllm/model_executor/layers/fused_moe/fused_moe.py (+67/-14); vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+7/-0); vllm/model_executor/layers/fused_moe/utils.py (+81/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: Working on perf optimizations for the Triton MoE kernels. Adds an opt-in `VLLM_TRITON_USE_TD` env var that switches the fused MoE kernel onto a tensor-descriptor based load/store path, mirroring `VLLM_TRITON_ATTN_USE_TD` (PR #40327). Auto-on for XPU; off by default on CUDA/ROCm (opt-in on Blackwell). The changes in this PR are scoped to the fused MoE kernel (`fused_moe_kernel`) only. ⏎  ⏎ The env var name follows the single-flag design proposed in  …[truncated]

### L3-7c6729b769  (L3, 2026-07-29, sha 7c6729b76959, PR #50089)
TITLE: [Model] Add Kimi K3 support: model files and kernels [1/N] (#50089)
SOURCES: path_core, dependency_pin, corpus:production-kernel-provenance
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.merge.triton_lse, L3.merge.cuda_lse
FILES: CMakeLists.txt (+26/-1); cmake/external_projects/flashkda.cmake (+74/-0); csrc/libtorch_stable/attention/merge_attn_states.cu (+32/-7); setup.py (+5/-0); .buildkite/check-torch-abi.py (+1/-0); .buildkite/test_areas/kernels.yaml (+0/-11); .buildkite/test_areas/models_basic.yaml (+4/-2); .pre-commit-config.yaml (+1/-1); benchmarks/kernels/benchmark_k3_cutedsl_residual.py (+367/-0); benchmarks/kernels/benchmark_kimi_k3_latent_moe_tail.py (+806/-0); (+99 more)
LABELS: performance, ready, ci/build, v1, multi-modality, nvidia, kimi, k3
BODY: ## Purpose ⏎  ⏎ split https://github.com/vllm-project/vllm/pull/50000 ⏎  ⏎ Note: it's not runnable now ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-65a1a16594  (L3, 2026-07-29, sha 65a1a1659499, PR #50194)
TITLE: [CPU] Fix FP8 attention scratchpad sizing (#50194)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+3/-2); csrc/cpu/cpu_attn.cpp (+6/-3); csrc/cpu/torch_bindings.cpp (+4/-2); tests/kernels/attention/test_cpu_attn.py (+20/-0); vllm/_custom_ops.py (+2/-0)
LABELS: ready, v1, cpu
BODY: ## Purpose ⏎  ⏎ The CPU attention scheduler computed FP8 KV tile geometry using `sizeof(kv_cache_t)`. With BF16 queries and one-byte FP8 KV cache entries, this selected larger tiles than the BF16-backed scratchpad can hold during large AMX prefills. The resulting out-of-bounds writes cause corrupted output ⏎  ⏎ ## Test Plan ⏎ Unit ⏎ ``` ⏎ pytest tests/kernels/attention/test_cpu_attn.py::test_varlen_with_paged_kv_fp8_large_prefill_amx -q ⏎ ``` ⏎ Model ⏎ ``` …[truncated]

### L3-0a31372e5f  (L3, 2026-07-29, sha 0a31372e5fe7, PR #47301)
TITLE: [Frontend] Add detokenization streaming derender for disaggregated serving (#47301)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/entrypoints/scale_out/derender/test_derender_stream.py (+765/-0); vllm/entrypoints/scale_out/derender/api_router.py (+64/-12); vllm/entrypoints/scale_out/derender/serving.py (+98/-3); vllm/entrypoints/scale_out/token_in_token_out/protocol.py (+165/-11); vllm/entrypoints/serve/engine/typing.py (+4/-0); vllm/renderers/online_derenderer.py (+281/-6)
LABELS: frontend, ready
BODY: ## Purpose ⏎  ⏎ Part 1 of 3 for RFC #47161. Adds a streaming mode to the `derender/` endpoints which mirrors the producer's per chunk SSE shape. Streaming is a requirement for interactive serving on platforms like llm-d and Dynamo. ⏎  ⏎ Points to note: ⏎  ⏎ - State is kept client carried and stateless on the server ⏎ - Responses are returned as plain JSON rather than SSE. This is because the derender can't buffer a whole generation or hold a connection  …[truncated]

### L3-f51193b9ae  (L3, 2026-07-29, sha f51193b9aefe, PR #49291)
TITLE: [Kernel][Mamba] Fused-kernel support for align-mode DS-conv state migration with num_accepted_tokens > 1 (#49291)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/mamba/test_precopy_mamba_align.py (+257/-20); tests/v1/e2e/general/test_mamba_prefix_cache.py (+38/-6); vllm/v1/worker/gpu/model_states/mamba_hybrid.py (+5/-12); vllm/v1/worker/gpu_model_runner.py (+1/-0); vllm/v1/worker/mamba_utils.py (+108/-26)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Hybrid/Mamba models support an align-mode Mamba cache path where the running Mamba state is migrated at scheduler-step / block boundaries. For **DS conv-state layout**, the align path has an intentionally guarded, not-yet-implemented cell: accepted-token tail copies with `num_accepted_tokens > 1` (MTP degree > 1). The scalar ⏎ `get_conv_copy_spec` helper rejects this case: ⏎  ⏎ ```text ⏎ AssertionError: DS conv state with num_accepted_t …[truncated]

### L3-5fa0154448  (L3, 2026-07-29, sha 5fa01544480b, PR #49647)
TITLE: [Rubin] Enable NVLink all-reduce paths on SM107 (#49647)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+10/-0); vllm/distributed/device_communicators/all_reduce_utils.py (+12/-0); vllm/distributed/device_communicators/symm_mem.py (+1/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Part of the ongoing effort to bring up **Rubin (sm_107)** support in ⏎ vLLM (similar to #49387). ⏎  ⏎ sm_107 has no entries in vLLM's collective-communication selection ⏎ tables, so the optimized NVLink all-reduce paths are all skipped and ⏎ traffic silently falls back to NCCL. This PR wires sm_107 into those ⏎ paths: ⏎  ⏎ - **Custom all-reduce** — enables the one-shot/two-shot NVLink kernels ⏎ - **PyTorch symmetric-memory all-reduce** — enables the mu …[truncated]

### L3-a7a204cc6e  (L3, 2026-07-29, sha a7a204cc6ec9, PR #50322)
TITLE: Add FlashMLA H100 tests to CI, fix them after #32810 (#50322)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/kernels.yaml (+18/-0); tests/kernels/attention/test_flashmla.py (+23/-5); tests/kernels/attention/test_mla_cross_layer_kernel_equivalence.py (+3/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ I recently ran the H100 tests locally for flashmla and realized that some were broken after the interface changed in #32810. I fixed them, and to prevent further regressions, I added them to CI. I haven't worked with buildkite before so I'm not sure it's correct, but I did it with the help of Claude. ⏎  ⏎ cc @LucasWilkinson who authored #32810 and @Harry-Chen ⏎  ⏎ ## Test Plan ⏎ CI, specifically these three test files should pass now ⏎    …[truncated]

### L3-b88916617d  (L3, 2026-07-29, sha b88916617d3d, PR #48257)
TITLE: [ROCm] [CI] Support cached K/V (key/value=None) in Triton prefix-prefill (#48257)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/v1/attention/ops/prefix_prefill.py (+189/-40); tests/kernels/attention/test_prefix_prefill.py (+336/-0)
LABELS: rocm, ready, v1
BODY: ## Purpose ⏎  ⏎ On ROCm, some models re-attend an already-cached sequence with the query ⏎ only, calling attention with `key=None` / `value=None`. A concrete example is ⏎ IQuest LoopCoder, whose loop layer calls `global_attn(q, None, None)`: ⏎  ⏎ ```python ⏎ global_attn_output = global_attn(q, None, None)   # key/value already in cache ⏎ local_attn_output  = local_attn(q, k, v) ⏎ ``` ⏎  ⏎ `key=None` / `value=None` is an existing contract in vLLM attention ( …[truncated]

### L3-437e0b7f8e  (L3, 2026-07-30, sha 437e0b7f8eb7, PR #50297)
TITLE: [BugFix] Fix P/D preemption race condition (#50297)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_async_scheduler.py (+73/-0); tests/v1/core/utils.py (+4/-3); vllm/distributed/kv_transfer/kv_connector/v1/base.py (+12/-0); vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py (+4/-0); vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py (+6/-0); vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py (+6/-0); vllm/v1/core/sched/scheduler.py (+12/-2)
LABELS: bug, ready, v1, kv-connector
BODY: There is a small race condition with P/D and async scheduling, where a prefill request can be preempted during it's final (possible only) step. The scheduler logic currently allows such requests to complete normally because for most purposes they are finished (final output token generated and returned). They just get removed from the waiting queue. ⏎  ⏎ However at this point, their kv blocks have been purged and so async kv-consuming connectors (pr …[truncated]

### L3-4e582c5b54  (L3, 2026-07-30, sha 4e582c5b54b8, PR #50326)
TITLE: [PD][Bugfix] Rebase KV lease deadlines onto worker clock (#50326)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_nixl_connector.py (+47/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py (+4/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py (+6/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py (+12/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py (+7/-0)
LABELS: bug, ready, v1, kv-connector
BODY: `pull_scheduler.request_finished` stamps `reqs_to_send` deadlines with the **scheduler process's** `time.perf_counter()`; `base_worker` expires them against **each worker's own** `perf_counter`. perf_counter epochs are process-local — across nodes they differ by boot-time deltas, so on a multi-node prefill deployment the second node's effective TTL is `ttl − (node clock-epoch gap)`, which can be near zero or negative. ⏎  ⏎ Premature release marks b …[truncated]

### L3-aeeb36b1f1  (L3, 2026-07-30, sha aeeb36b1f171, PR #50000)
TITLE: [New model] Kimi K3 (#50000)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.common_v1, L3.mla.flashinfer, L3.mla.rocm_aiter
FILES: vllm/model_executor/layers/attention/mla_attention.py (+52/-8); vllm/model_executor/layers/mla.py (+7/-0); cmake/external_projects/deepgemm.cmake (+3/-3); docs/models/supported_models.md (+1/-0); pyproject.toml (+2/-0); requirements/cuda.txt (+2/-2); requirements/test/cuda.txt (+1/-1); tests/kernels/moe/test_deepgemm.py (+32/-0); tests/models/registry.py (+18/-0); tests/models/test_dspark_mla.py (+143/-0); (+72 more)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1
BODY: ## Purpose ⏎  ⏎ **7.30 Note: after merging, it still needs to install https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.16rc5 to run K3** ⏎  ⏎ add `moonshotai/Kimi-K3` model support ⏎  ⏎ Please use the docker image `vllm/vllm-openai:kimi-k3` to run it, see vllm recipes https://recipes.vllm.ai/moonshotai/Kimi-K3. ⏎  ⏎ This branch has some extra dependencies: ⏎ - SITU trtllmgen MOE: https://github.com/flashinfer-ai/flashinfer/pull/4180 ⏎ - MLA: h …[truncated]

### L3-f1e8fd27ae  (L3, 2026-07-30, sha f1e8fd27aea5, PR #49937)
TITLE: [ROCm] Add AITER FP8 ViT encoder attention (#49937)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/model_executor/layers/attention/mm_encoder_attention.py (+108/-32); vllm/platforms/rocm.py (+1/-1); vllm/v1/attention/ops/vit_attn_wrappers.py (+29/-0); benchmarks/kernels/benchmark_vit_aiter_fp8_attn.py (+124/-0); docs/features/quantization/fp8_vit_attn.md (+27/-8); tests/kernels/attention/test_mha_attn.py (+135/-1); tests/kernels/core/test_vit_fp8_scaling.py (+44/-3); vllm/_aiter_ops.py (+95/-0)
LABELS: documentation, performance, rocm, ready, v1
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎  ⏎ - enable AITER per-tensor FP8 attention for multimodal vision encoders on gfx942 and gfx950 ⏎ - preserve packed variable-length image/video boundaries through AITER's native varlen kernel ⏎ - add compile-safe custom-op wrapping, architecture/symbol validation, correctness tests, a benchmark, and ROCm usage docs ⏎  ⏎ ## Why this is not duplicate work ⏎  ⏎ I searched open vLLM PRs for `AITER FP8 ViT encoder attention` and `ROCM_AITER_FA mm encoder  …[truncated]

### L3-5b95890742  (L3, 2026-07-30, sha 5b958907420e, PR #50339)
TITLE: [FlexAttention] Avoid encoder block-mask compile explosion (#50339)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flex_attention
FILES: vllm/v1/attention/backends/flex_attention.py (+12/-2); tests/entrypoints/pooling/embed/test_online_long_text.py (+5/-3); tests/kernels/test_flex_attention.py (+44/-5); vllm/config/attention.py (+5/-3)
LABELS: ready, v1
BODY: - Default encoder-only FlexAttention masks to 128-token Q/KV blocks, while keeping the existing small-block defaults for paged KV attention and honoring explicit block-size overrides. The long-text fixture also uses a local seeded generator so token counts and compiler shapes are reproducible. ⏎ - On MI355, the exact cold-cache test improved from 300.31 seconds to 37.99 seconds; the full long-text module passed 5/5 in 42.78 seconds, the packed enc …[truncated]

### L3-904fae8be1  (L3, 2026-07-30, sha 904fae8be12f, PR #50312)
TITLE: [DSv4 Perf] Fix redundant memory allocation and copy for dsv4 pp buffer, 448 MiB GPU memory saved (#50312)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/amd/model.py (+8/-10); vllm/models/deepseek_v4/nvidia/model.py (+8/-10); vllm/models/deepseek_v4/xpu/model.py (+8/-8)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ We do `torch.empty` and `copy` in pp last rank even if there is no mtp enabled, this PR fixes the issue, ⏎  ⏎ ## Test ⏎  ⏎ Perf gain can be seen in this AI generated script ⏎  ⏎ ```py ⏎ # SPDX-License-Identifier: Apache-2.0 ⏎  ⏎ import torch ⏎  ⏎ HIDDEN_SIZE = 7168 ⏎ HC_MULT = 4 ⏎ MAX_NUM_TOKENS = 8192 ⏎ NUM_WARMUPS = 20 ⏎ NUM_REPEATS = 1000 ⏎ TOKEN_COUNTS = (1, 8, 32, 128, 512, 2048, 8192) ⏎  ⏎  ⏎ def main() -> None: ⏎     assert torch.cuda.is_availab …[truncated]

### L3-837eae6458  (L3, 2026-07-30, sha 837eae64580c, PR #50298)
TITLE: [DSv4 Perf] Remove redundant full kernel for dsv4, 1.88x kernel performance improvement (#50298)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/nvidia/flashmla.py (+15/-2); tests/kernels/attention/test_flashmla_sparse.py (+16/-12); vllm/models/deepseek_v4/common/ops/cache_utils.py (+13/-9)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Part of https://github.com/vllm-project/vllm/issues/45861 ⏎  ⏎ Passing an out tensor to avoid additional `torch.full` kernel call ⏎  ⏎ ## Test ⏎  ⏎ Covered in unit tests ⏎  ⏎ Perf can be seen in this AI generated script ⏎  ⏎ ```py ⏎ import gc ⏎ import statistics ⏎ import time ⏎ from collections.abc import Callable ⏎  ⏎ import torch ⏎  ⏎ from vllm.models.deepseek_v4.common.ops.cache_utils import ( ⏎     combine_topk_swa_indices, ⏎ ) ⏎  ⏎ # Fixed productio …[truncated]

### L3-70bd10930b  (L3, 2026-07-30, sha 70bd10930bbb, PR #50273)
TITLE: [Quantization] Honor `--linear-backend` for ModelOpt W4A16 (#50273)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/quantization/test_modelopt.py (+23/-4); vllm/model_executor/kernels/linear/__init__.py (+5/-0); vllm/model_executor/layers/quantization/modelopt.py (+14/-17)
LABELS: ready, quantization
BODY: `ModelOptNvFp4W4A16LinearMethod` currently hardcodes usage of the Marlin kernel, ignoring `--linear-backend`. ⏎  ⏎ #### PR ⏎ * `--linear-backend=auto`: Marlin remains the default ⏎ * `--linear-backend=humming`: Explicitly selected compatible backend is now honored. ⏎  ⏎ Related but not duplicate: [#49382](https://github.com/vllm-project/vllm/pull/49382) adds a FlashInfer W4A16 kernel and changes the `auto` selection on SM121.  ⏎ This PR adds no kernel a …[truncated]

### L3-59e831c09a  (L3, 2026-07-30, sha 59e831c09a22, PR #48757)
TITLE: [Compilation]Fuse Transformers Residual Add + RMSNorm (#48757)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/passes/test_rmsnorm_reshape_fusion.py (+105/-0); vllm/compilation/passes/fusion/add_rms_fusion.py (+164/-0); vllm/compilation/passes/pass_manager.py (+22/-0); vllm/kernels/aiter_ops.py (+7/-2)
LABELS: rocm, ready, quantization
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ Transformers models emit residual add and `RMSNorm` separately, preventing the use of FusedAddRMSNorm kernels. This change canonicalizes them into fused_add_rms_norm and handles the intervening reshape. Furthermore, the separate residual add and reshape operations prevented some patterns, like AR + rms, from being discovered and fused. Since this PR hides those offending ops, the fusions are fixed again. ⏎  ⏎ ### Benchmark ⏎ #### Qwen3-0 …[truncated]

### L3-12a34a6bc7  (L3, 2026-07-30, sha 12a34a6bc794, PR #46720)
TITLE: [ROCm][DSV4] B-preshuffle the attention fp8 projections (#46720)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/_aiter_ops.py (+53/-2); vllm/model_executor/model_loader/utils.py (+4/-0); vllm/models/deepseek_v4/amd/model.py (+53/-3); vllm/models/deepseek_v4/amd/rocm.py (+71/-1); vllm/models/deepseek_v4/attention.py (+6/-3)
LABELS: rocm, ready, ci-failure
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎ Speed up the DeepSeek-V4 fp8 attention projections `fused_wqa_wkv` (fused Q/KV ⏎ down-proj) and `wo_b` (output proj) on ROCm (gfx950) by routing them through ⏎ AITER's weight-preshuffled block-scale GEMM instead of the default block-scale GEMM. ⏎  ⏎ ## What it does ⏎ - **Load time:** B-preshuffle the two projections' fp8 weights via the existing ⏎   `shuffle_weight((16, 16))` and cache the block scales (`prepare_attn_preshuffle`). ⏎ - **Run  …[truncated]

### L3-68fb303a41  (L3, 2026-07-30, sha 68fb303a4177, PR #48438)
TITLE: [Bugfix] Preserve Marlin runtime tensor storage across weight reload (#48438)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/model_executor/model_loader/test_reload.py (+277/-0); vllm/model_executor/kernels/linear/mixed_precision/marlin.py (+8/-3); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py (+3/-1); vllm/model_executor/layers/quantization/utils/marlin_utils.py (+21/-4); vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py (+9/-3); vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py (+12/-6)
LABELS: bug, ready, quantization
BODY: The fix is verified at three levels: red/green CPU regression tests (in this PR), live-engine GPU validation on an RTX 4090 (pointer identity across reload, a graph-coupling flip test, and a livelock demonstration on unfixed main), and live bug-side/fixed-side validation of every sibling Marlin path. Method, results, and the validation script for the primary fix are in this description; the per-site sibling matrix is in the validation results com …[truncated]

### L3-3333d7cb63  (L3, 2026-07-30, sha 3333d7cb6391, PR #49361)
TITLE: [ROCm]: bump AITER to 0.1.19 (#49361)
SOURCES: release_notes, corpus:kernel-correctness-cases(introducing)
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm_base (+1/-1)
LABELS: rocm, ready, ci/build
DEEP_STUDY: deep-study: introduced the defect fixed in case vllm:385d4c084e (fix PR 50859)
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-c27b080d26  (L3, 2026-07-30, sha c27b080d262e, PR #50444)
TITLE: [compile] Fix fake kernel return dtype (#50444)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/attention.py (+1/-1); tests/compile/passes/test_rope_kvcache_fusion.py (+32/-0); vllm/compilation/passes/fusion/rope_kvcache_fusion.py (+1/-1)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Align the runtime and fake-kernel metadata for unified KV-cache update operators. This prevents Inductor metadata assertions when quantized KV caches use uint8 while attention activations use bfloat16 or float16. In PyTorch 2.14, we have increased metadata assertions in Inductor that require these to be more exact. ⏎  ⏎ ## Test Plan ⏎  ⏎ Add regression coverage for both regular and fused RoPE update paths. ⏎  ⏎ ## Test Result ⏎  ⏎ Pass ⏎  ⏎ - …[truncated]

### L3-0bff0ce5d3  (L3, 2026-07-30, sha 0bff0ce5d392, PR #50458)
TITLE: [Kimi K3 Bug] Fix deepgemm support for kimi k3 (#50458)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_deepgemm.py (+1/-0); vllm/model_executor/layers/fused_moe/deep_gemm_utils.py (+3/-1)
LABELS: bug, ready, kimi, k3
BODY: ## Purpose ⏎  ⏎ `VLLM_USE_RUST_FRONTEND=0 vllm serve moonshotai/Kimi-K3   --moe-backend auto   --gpu-memory-utilization 0.95   --tensor-parallel-size 8   --load-format fastsafetensors   --no-enable-flashinfer-autotune   --max-model-len 1048576   --kv-cache-dtype fp8   --attention-config '{"use_prefill_query_quantization":true,"mla_prefill_backend":"TRTLLM_RAGGED"}'   --enable-prefix-caching   --enable-auto-tool-choice   --tool-call-parser kimi_k3   …[truncated]

### L3-f1899b2ffd  (L3, 2026-07-30, sha f1899b2ffdf8, PR #45227)
TITLE: [Bugfix][ROCm] AITER MLA: size MTP verification decode metadata for real qlen/dtype (#45227)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+147/-36); tests/v1/attention/test_rocm_aiter_mla_mtp_split.py (+408/-0)
LABELS: bug, documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready
BODY: ## Purpose ⏎  ⏎ On ROCm AITER MLA, MTP verification decode (qlen = `num_speculative_tokens + 1`, so 4 for MTP3) was sized as ordinary qlen=1 decode: `get_mla_metadata_info_v1` received `qo_len=1` and the kernel recomputed metadata internally for qlen>1. The verification logits come out wrong, the accept/reject step drifts from the target distribution, and accuracy drops. Passing q/kv dtypes (#46997) does not fix this on its own. ⏎  ⏎ Fix: size the decode …[truncated]

### L3-6724051c57  (L3, 2026-07-30, sha 6724051c5766, PR #50328)
TITLE: [CI/Build][AMD] Install triton_kernels via CMake (#50328)
SOURCES: dependency_pin
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: cmake/external_projects/triton_kernels.cmake (+11/-6); docker/Dockerfile.rocm_base (+0/-2); vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py (+9/-22)
LABELS: rocm, ready, ci/build, gpt-oss
ISSUES: #49778 [Bug]: gpt-oss-120b fails for ROCm 7.14 on gfx950
BODY: ## Purpose ⏎ Starting with ROCm 7.14 we will install dependencies such as Triton from a [package index](https://repo.amd.com/rocm/whl-multi-arch/) instead of building from source. However, the package index does not include `triton_kernels`. ⏎  ⏎ Running `gpt-oss-120b` on `gfx950` without a ROCm-specific `triton_kernels` ⏎  ⏎ ```bash ⏎ vllm serve openai/gpt-oss-120b \ ⏎   --enable-expert-parallel \ ⏎   --tensor-parallel-size 4 ⏎ ``` ⏎ results in the below  …[truncated]

### L3-34bb795ff3  (L3, 2026-07-31, sha 34bb795ff3ef, PR #49143)
TITLE: [CI] Add M3 MSA tests to CI (#49143)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/kernels.yaml (+4/-0); tests/kernels/attention/test_minimax_m3.py (+24/-17); vllm/models/minimax_m3/common/indexer.py (+6/-6)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Follow up from #47442. Turns out some SM100 kernel tests did not run in CI because they were not included in the buildkite config. ⏎ - Inkling TML FA4: Fixed by #48988 ⏎   - Edit: The tests are added by #49325. This PR now only cover M3 MSA ⏎ - Minimax M3 MSA: To be fixed by https://github.com/vllm-project/MSA/pull/8 and #49016 ⏎  ⏎ cc @arpera  ⏎  ⏎ Edit: ~~This PR also makes RNG inputs for `test_gdn_prefill_cutedsl.py` deterministic to ma …[truncated]
