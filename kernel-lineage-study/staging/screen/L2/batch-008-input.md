### L2-d01812d89e  (L2, 2026-08-17, sha d01812d89e21, PR #34580)
TITLE: [AMD] Optimize KIMI-K3 with Triton MLA decode kernel by tuning the stage-1 geometry for gfx950 (#34580)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm
FILES: python/sglang/kernels/ops/attention/decode_attention.py (+206/-11); python/sglang/srt/environ.py (+3/-0); test/registered/unit/layers/attention/test_mla_decode_forced_splits.py (+259/-0); test/registered/unit/layers/attention/test_mla_decode_geometry.py (+224/-0)
LABELS: amd, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Motivation ⏎  ⏎  ⏎ The Triton MLA decode kernel picks its stage-1 geometry -- the number of KV splits and the block size along the KV axis -- from constants tuned on CDNA3. On gfx950 those constants fit badly at the batch sizes that matter for long-context serving. The split count is derived from a fixed workgroup budget, so at small batch a single split covers the whole sequence, while at large batch the budget is treated as a rounding target ra …[truncated]

### L2-0077f84d37  (L2, 2026-08-18, sha 0077f84d37a0, PR #31180)
TITLE: [mem_cache][8/N] refactor: move MambaPoolHost to pool_host.mamba (#31180)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py (+1/-1); python/sglang/srt/mem_cache/memory_pool_host.py (+3/-582); python/sglang/srt/mem_cache/pool_host/mamba.py (+610/-0); test/registered/kernels/ops/mamba/test_transfer_mamba.py (+1/-1); test/registered/unit/mem_cache/test_hicache_staged_write_back_dispatch.py (+1/-1); test/registered/unit/mem_cache/test_mem_pool_host.py (+1/-1)
LABELS: hicache, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Part of the `pool_host` phase of the mem_cache refactor (issue #25371), continuing from PR #30616 ([7/N]). That PR relocated the **MLA host pool** into its own `pool_host/mla.py` module, leaving the remaining concrete pools in `memory_pool_host.py`. This PR relocates the **Mamba host pool** — `MambaPoolHost` — into its own `pool_host/mamba.py` module, following the exact same mechanical pattern. ⏎  ⏎ ## Modifications ⏎  ⏎ Mechanical reloca …[truncated]

### L2-cfc6dfb364  (L2, 2026-08-18, sha cfc6dfb3642b, PR #34923)
TITLE: Apply latest DeepEP branch (#34923)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+6/-0); python/pyproject.toml (+1/-1); python/sglang/srt/layers/moe/token_dispatcher/deepep.py (+15/-0); scripts/ci/cuda/ci_install_dependency.sh (+12/-0); test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py (+4/-0)
LABELS: dependencies, deepseek, run-ci, run-ci-extra, release-highlight
BODY: ## Summary ⏎  ⏎ - configure `NVSHMEM_QP_DEPTH` before CUDA DeepEP low-latency buffer initialization ⏎ - enforce `max(existing value, 1024, 2 * (num_max_dispatch_tokens_per_rank + 1))` ⏎ - preserve larger user-provided values and leave NPU and normal DeepEP paths unchanged ⏎ - force reinstall `nvidia-nccl-cu13==2.30.7` in CUDA 13 CI and the main CUDA 13 image; CUDA 12 remains unchanged ⏎ - bump the runtime dependency to `sgl-deep-ep==0.1.1` ⏎  ⏎ ## Motivation ⏎  ⏎ De …[truncated]

### L2-3a8f522f65  (L2, 2026-08-18, sha 3a8f522f6547, PR #30612)
TITLE: install sglang in virtual env instead of system path (#30612)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+19/-21)
BODY: ## Motivation ⏎  ⏎ Install `sglang` and related packages in python virtual env instead of system path to avoid conflicts in updating pip packages and debian python packages.  ⏎  ⏎ Detailed error is:  ⏎ ``` ⏎ #26 473.6     Found existing installation: cryptography 41.0.7 ⏎ #26 473.6 error: uninstall-no-record-file ⏎ #26 473.6  ⏎ #26 473.6 × Cannot uninstall cryptography 41.0.7 ⏎ #26 473.6 ╰─> The package's contents are unknown: no RECORD file was found for  …[truncated]

### L2-ce1830c59b  (L2, 2026-08-19, sha ce1830c59b75, PR #33165)
TITLE: [AMD] DeepSeek-V4 MI355X: eliminate bpreshuffle fp8-scale relayout copy in dense w8a8 linear (#33165)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/fp8_utils.py (+12/-3); test/registered/unit/layers/test_fp8_bpreshuffle_dense_linear_mi35x.py (+188/-0); test/registered/unit/layers/test_fp8_bpreshuffle_scale.py (+73/-0)
LABELS: amd, deepseek, run-ci
BODY: ## Summary ⏎  ⏎ On MI355X (gfx950) the CK bpreshuffle w8a8 blockscale GEMM consumes the per-group ⏎ activation scale in **column-major** `[num_groups, tokens]` layout. `aiter_w8a8_block_fp8_linear` ⏎ quantizes the activation row-major and then relays the scale out with ⏎ `materialize_bpreshuffle_fp8_scale` = `.t().contiguous().t()` — a real relayout **copy per dense ⏎ w8a8 GEMM** (the MLA q/kv/o projections and MoE), one of the larger ELEWISE deltas in the ⏎ D …[truncated]

### L2-f446e853e7  (L2, 2026-08-19, sha f446e853e73f, PR #33313)
TITLE: [AMD] DeepSeek-V4: route decode wo_a bf16 batched matmul to aiter batched_gemm_bf16 (#33313)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v4.py (+86/-1); python/sglang/srt/environ.py (+4/-0); test/registered/unit/models/test_deepseek_v4_amd_wo_a_bf16.py (+203/-0)
LABELS: amd, deepseek, run-ci
BODY: # [AMD] DeepSeek-V4: route decode wo_a bf16 batched matmul to aiter `batched_gemm_bf16` ⏎  ⏎ ## Summary ⏎ On the DeepSeek-V4 ROCm decode path, the MLA output-absorb (`wo_a`) bf16 GEMM runs ⏎ `torch.einsum("tgd,grd->tgr", o, wo_a)`, which dispatches to a **rocBLAS/Tensile ⏎ `Cijk_*` batched GEMM** — one of the larger kernels in the DSV4 decode attention ⏎ region. aiter ships a tuned `batched_gemm_bf16` for exactly this shape, and the ⏎ reference ATOM stack uses …[truncated]

### L2-5f12839591  (L2, 2026-08-19, sha 5f128395910d, PR #35077)
TITLE: [Fix] Support Kimi-K3 ModelOpt mixed NVFP4/FP8 checkpoint (#35077)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py (+13/-2); python/sglang/srt/layers/quantization/modelopt_quant.py (+30/-11); python/sglang/srt/models/kimi_k3.py (+82/-23); test/registered/unit/layers/quantization/test_modelopt_nvfp4_moe_scales.py (+20/-0); test/registered/unit/model_loader/test_modelopt_loader.py (+22/-0); test/registered/unit/models/test_kimi_k3_bfa_overlap.py (+38/-2)
LABELS: quant, blackwell, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ The official [nvidia/Kimi-K3-NVFP4](https://huggingface.co/nvidia/Kimi-K3-NVFP4) checkpoint is a ModelOpt mixed-precision checkpoint: ⏎  ⏎ - routed MoE experts use NVFP4 with SiTU (`beta=4`, `linear_beta=25`); ⏎ - supported attention projections use weight-only `FP8_PB_WO` with 128x128 block scales. ⏎  ⏎ Current `main` cannot serve this checkpoint with the FlashInfer TRT-LLM MoE backend. It rejects gated `situ` during startup, and it does no …[truncated]

### L2-e73201e462  (L2, 2026-08-19, sha e73201e46231, PR #35339)
TITLE: [diffusion] feat: support cache-dit, cfg gating, attention backend override as per-request param (#35339)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/docs/sglang-diffusion/attention_backends.mdx (+31/-0); docs/docs/sglang-diffusion/cache_dit.mdx (+44/-4); python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-performance/SKILL.md (+3/-2); python/sglang/multimodal_gen/configs/sample/sampling_params.py (+32/-0); python/sglang/multimodal_gen/runtime/cache/cache_dit_integration.py (+69/-0); python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py (+6/-0); python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py (+6/-0); python/sglang/multimodal_gen/runtime/layers/attention/layer.py (+50/-3); python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py (+249/-46); python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/ltx_2/denoising.py (+8/-8); (+7 more)
LABELS: documentation, run-ci, diffusion
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ Lossy accelerations should be **per-request** switches, not process-wide env vars / server args: one deployment should be able to mix accelerated and lossless requests. SGLang-Diffusion already follows this for TeaCache (`enable_teacache`), Spectrum, progressive resolution, and `quality="high"` — but Cache-DiT was ~20 process-wide `SGLANG_CACHE_DIT_*` env vars, CFG gating was a process-wide `SGLANG_DIFFUSION_CFG_GATE_STEP` float, a …[truncated]

### L2-c7478228dd  (L2, 2026-08-19, sha c7478228dd29, PR #30984)
TITLE: [AMD] [Docker] Upgrade Python 3.12 + torch 2.11 + triton 3.7 in ROCm 7.2.4 (#30984)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/rocm.Dockerfile (+199/-48); .github/workflows/pr-test-amd-extra.yml (+15/-3); .github/workflows/pr-test-amd-rocm720.yml (+46/-20); .github/workflows/release-docker-amd-rocm720-nightly.yml (+18/-3); .github/workflows/release-docker-amd.yml (+17/-4); python/pyproject_other.toml (+17/-0); scripts/ci/amd/amd_ci_install_dependency.sh (+88/-38); scripts/ci/amd/amd_ci_start_container.sh (+2/-2); scripts/ci/amd/amd_ci_start_container_disagg.sh (+26/-14)
LABELS: amd, dependencies, jit-kernel
BODY: ## Motivation ⏎  ⏎ Add ROCm 7.2.4 Docker flavors on Python 3.12 with PyTorch 2.11 and Triton 3.7.  ⏎  ⏎ PyTorch 2.11 for ROCm 7.2 is available from the PyTorch Foundation index. Its dependency initially installs `triton-rocm==3.6.0`, but this PR replaces it at the end of the build with AITER’s pinned Triton 3.7. Installing Triton last prevents later dependency resolution from reverting the validated ROCm stack. ⏎  ⏎ | Component | ROCm 7.2.0 flavors | R …[truncated]

### L2-50dae2d99d  (L2, 2026-08-19, sha 50dae2d99d70, PR #32340)
TITLE: Amd/dsv4 shared experts fusion top6 (#32340)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py (+49/-23); python/sglang/kernels/ops/moe/moe_fused_gate.py (+9/-2); python/sglang/srt/layers/moe/topk.py (+13/-3); test/registered/kernels/ops/moe/test_moe_fused_gate.py (+36/-0); test/registered/moe/test_fused_append_remap_per_rank_shared_slots.py (+69/-3); test/registered/moe/test_fused_append_shared_experts_top6.py (+115/-0)
LABELS: amd, deepseek, run-ci, jit-kernel
BODY: # [AMD] DeepSeek-V4: fix shared-experts fusion for top-6 ⏎  ⏎ ## Summary ⏎  ⏎ Enabling shared-experts fusion (`--enforce-shared-experts-fusion`) for ⏎ DeepSeek-V4 on MI355X (gfx950) crashed at startup. Two independent issues in the ⏎ fused topk / append path assume DeepSeek-V3 conventions (fp32 correction bias, ⏎ power-of-two topk) that DeepSeek-V4 (bf16 correction bias, **top-6** routing) ⏎ violates. This PR fixes both so the fused path runs, and shows  …[truncated]

### L2-9db4ba8da1  (L2, 2026-08-20, sha 9db4ba8da166, PR #32327)
TITLE: [DeepSeek-V4] Add Q8KV8 sparse MLA prefill runtime backend (#32327)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/kernels/ops/attention/dsv4/dequant_k_cache.py (+209/-0); python/sglang/kernels/ops/attention/sparse_mla_q8kv8_prefill_sm90.py (+137/-7); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+246/-2); python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py (+29/-4); python/sglang/srt/server_args.py (+18/-0); test/registered/kernels/ops/attention/test_q8kv8_sparse_prefill_backend.py (+681/-0); test/registered/unit/server_args/test_server_args.py (+17/-0)
LABELS: quant, deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Summary ⏎ This PR is part of the roadmap tracked in https://github.com/sgl-project/sglang/issues/25746. ⏎ This PR ports the Q8KV8 sparse MLA prefill path to the DeepSeek-V4 runtime backend and adds a runtime dispatch path via `--dsv4-prefill-backend flashmla_sparse_q8`. When the KV cache uses `fp8_e4m3`, DeepSeek-V4 can run the FP8 query × FP8 KV sparse MLA kernel during prefill instead of falling back to the existing BF16 sparse prefill path. T …[truncated]

### L2-eac91ac362  (L2, 2026-08-20, sha eac91ac362f1, PR #35412)
TITLE: [Fix] Land the decode mamba checkpoint depth on the tree page under DCP (#35412)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/schedule_batch.py (+2/-2); python/sglang/srt/managers/scheduler_components/batch_result_processor.py (+4/-4); python/sglang/srt/runtime_context.py (+8/-0); python/sglang/srt/speculative/dflash_worker_v2.py (+2/-2); python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py (+8/-2); python/sglang/srt/speculative/spec_utils.py (+4/-4); test/registered/unit/managers/test_batch_result_processor_mamba_boundary.py (+9/-2); test/registered/unit/managers/test_mamba_checkpoint_depth.py (+21/-1); test/registered/unit/spec/test_ngram_mamba_verify_update.py (+2/-2)
LABELS: run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ A donated mamba checkpoint is only reusable at a depth the radix tree can name. `cache_finished_req` / `cache_unfinished_req` floor the insert key with `page_aligned(tree_page)`, so a checkpoint taken off that grid is attached to the *preceding* node while its state already covers tokens past it — a later request matching that node resumes the linear-attention recurrence mid-stream and then re-consumes those tokens. Silent, no asse …[truncated]

### L2-a5a9d66baf  (L2, 2026-08-20, sha a5a9d66bafa9, PR #34546)
TITLE: [XPU] Fix/kimi linear xpu (#34546)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/linear/kernels/kda_triton.py (+5/-2); python/sglang/srt/layers/moe/topk.py (+2/-1); python/sglang/srt/models/kimi_linear.py (+2/-2)
LABELS: intel, xpu, run-ci
BODY: ## Motivation ⏎ Enables KimiLinearForCausalLM (hybrid KDA linear-attention + MLA + MoE) to run on Intel XPU. ⏎  ⏎ ## Modifications ⏎  ⏎ - XPU has no tvm_ffi CUDA JIT kernel for KDA packed decode, so batched decode crashed. Set supports_packed_decode = ... and not is_xpu() so XPU uses the non-packed Triton decode() path (fused_sigmoid_gating_delta_rule_update) — the same fallback CPU/NPU already use ⏎ - grouped-topk kernels (sgl_kernel topk_sigmoid on X …[truncated]

### L2-bda9952377  (L2, 2026-08-20, sha bda995237751, PR #33166)
TITLE: [AMD] DeepSeek-V4 MI355X: eliminate bpreshuffle fp8-scale copies at producer sites (MoE down, MLA o_proj bmm) (#33166)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py (+28/-4); python/sglang/srt/models/deepseek_v2.py (+9/-2); python/sglang/srt/layers/quantization/fp8_utils.py (+37/-4); test/registered/unit/layers/test_fp8_bpreshuffle_producer_mi35x.py (+156/-0); test/registered/unit/layers/test_fp8_bpreshuffle_scale.py (+62/-0)
LABELS: amd, deepseek, run-ci
BODY: ## Summary ⏎  ⏎ Follow-up to the dense-linear bpreshuffle scale no-copy. Several DeepSeek-V4 sites **pre-quantize** ⏎ an activation and hand a `(fp8, scale)` tuple to a downstream Linear; those scales are emitted ⏎ row-major and then relaid out with `materialize_bpreshuffle_fp8_scale` — a relayout **copy per ⏎ site, per layer** on MI355X (gfx950). This PR eliminates those copies for the producer quant ⏎ kernels that honor `transpose_scale`, via a zero-copy ` …[truncated]

### L2-34180a0d35  (L2, 2026-08-20, sha 34180a0d3544, PR #35499)
TITLE: [AMD] Improve K3 dspark draft attn kernel perf (#35499)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/verify_mla.py (+94/-16); python/sglang/srt/configs/model_config.py (+4/-0); python/sglang/srt/layers/attention/triton_backend.py (+10/-1)
LABELS: amd, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ With DSpark on Kimi-K3, every step runs 5 draft attn layers (qwen-style, non-causal).  ⏎ Draft attn dominates their cost. In a trace study, the 5 draft layers took 2.368 ms, of which the 5 attn kernels alone accounted for 1.244 ms (>50%). ⏎  ⏎ <img width="1207" height="344" alt="image" src="https://github.com/user-attachments/assets/3aee2eae-f02e-44cd-a5d1-1c2b82a41f31" /> ⏎  ⏎ The reason is that this falls back to `extend_attenti …[truncated]

### L2-44c90c6282  (L2, 2026-08-20, sha 44c90c628264, PR #34973)
TITLE: [AMD] DSv4: fuse the qk-norm-rope pair on the MTP target-verify path (#34973)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+4/-0); python/sglang/srt/models/deepseek_v4.py (+45/-8)
LABELS: amd, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ On the DeepSeek-V4 unified-KV path, the fused qk-norm-rope kernel is wired to ⏎ decode only. MTP **target-verify** falls back to running the same work as two ⏎ separate launches per layer — `sglang::fused_q_norm_rope` for q and ⏎ `sglang::fused_norm_rope` for kv — which is 122 launches per verify step across ⏎ the 61 target layers. ⏎  ⏎ This PR routes target-verify through the same fused kernel, **with the cache ⏎ store left off**. ⏎  ⏎ The store half  …[truncated]

### L2-d315eb7250  (L2, 2026-08-22, sha d315eb725044, PR #32577)
TITLE: [AMD] DeepSeek-V4: add aiter fused mHC post+pre with cross-layer boundary dispatch (#32577)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py (+259/-0); python/sglang/srt/models/deepseek_v4.py (+144/-51); test/registered/unit/models/test_deepseek_v4_amd_fused_mhc.py (+400/-0); test/registered/unit/models/test_deepseek_v4_fused_mhc_policy.py (+0/-79)
LABELS: amd, deepseek, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ Add a HIP/aiter fused `mhc_post` + next-layer `mhc_pre` path for DeepSeek-V4 on ⏎ gfx95, dispatched across the attention/MoE boundary alongside the existing TileLang ⏎ and Triton fused paths. On MI355X (gfx950) this is a **+0.8%–1.8% output-throughput ⏎ win across concurrency 4–64 with accuracy preserved**. ⏎  ⏎ ## What changed ⏎  ⏎ - **`python/sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py`** ⏎   - `try_aiter_fused_mhc_post_pre()`: w …[truncated]

### L2-af39ad9349  (L2, 2026-08-22, sha af39ad93493c, PR #33829)
TITLE: [Model] Complete dots.note.omni support with native encoders, video preprocessing, and MTP decoding (#33829)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.pool.mla_token_kv, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+21/-0); docs/cookbook/autoregressive/RedNote/Dots3-Note.mdx (+100/-20); docs/src/snippets/configs/rednote/dots3-note.jsx (+112/-60); python/sglang/srt/arg_groups/overrides.py (+3/-0); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/dots3.py (+243/-0); python/sglang/srt/configs/model_config.py (+24/-1); python/sglang/srt/entrypoints/openai/protocol.py (+1/-0); python/sglang/srt/entrypoints/openai/serving_chat.py (+36/-0); python/sglang/srt/function_call/dots_detector.py (+353/-0); (+45 more)
LABELS: documentation, high priority, Multi-modal, deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight, memory-pool
BODY: ## Motivation ⏎  ⏎ Merge dots.note.omni model ⏎  ⏎ ## Modifications ⏎  ⏎ This PR completes the SGLang integration of dots.note.omni, including: ⏎  ⏎   - Native in-process vision and audio encoders ⏎   - Train-consistent native video preprocessing ⏎   - Full-sharing MTP/NextN speculative decoding ⏎   - DP/TP/EP execution support, including overlap scheduling ⏎   - Correct KV-cache sizing for the hybrid sliding-window draft model ⏎  ⏎ ## How dots.note.omni diffe …[truncated]

### L2-b98d472158  (L2, 2026-08-22, sha b98d472158f7, PR #34855)
TITLE: [NPU] [Diffusion] Fix critical Ascend NPU Diffusion regression/bugs & restore 2-NPU CI testcase (#34855)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/npu.Dockerfile (+1/-1); .github/workflows/diffusion-ci-gt-gen-npu.yml (+6/-2); .github/workflows/pr-test-npu.yml (+20/-6); .github/workflows/release-docker-npu-nightly.yml (+1/-1); .github/workflows/release-docker-npu.yml (+1/-1); python/sglang/kernels/ops/attention/flash_attention.py (+0/-1); python/sglang/multimodal_gen/runtime/layers/attention/backends/ascend_fa.py (+205/-1); python/sglang/multimodal_gen/runtime/layers/attention/backends/attention_backend.py (+15/-0); python/sglang/multimodal_gen/runtime/layers/attention/backends/flash_attn.py (+34/-0); python/sglang/multimodal_gen/runtime/layers/attention/layer.py (+1/-1); (+10 more)
LABELS: npu, run-ci, diffusion, jit-kernel
BODY: ## Motivation ⏎  ⏎ Restore Ascend/NPU packed Ring Attention, 2-NPU diffusion CI suite, fix NPU correctness, runtime, and CI regressions exposed by recent Ring, SRT-CLIP, residency, and MOVA changes. ⏎  ⏎ After [#35004](https://github.com/sgl-project/sglang/pull/35004), diffusion can import SRT CLIP code and execute SRT `init_npu_backend()`. Importing `torch_npu.contrib.transfer_to_npu` inside a native NPU diffusion process globally rewrites CUDA-faci …[truncated]

### L2-155aa26c19  (L2, 2026-08-23, sha 155aa26c19cd, PR #36004)
TITLE: [AMD][DSV4] perf: use full 1024-thread block for indexer top-k on ROCm (#36004)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/aot/csrc/elementwise/deepseek_v4_topk.cu (+9/-0); python/sglang/kernels/aot/tests/test_topk.py (+43/-0)
LABELS: amd, deepseek, sgl-kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ `deepseek_v4_topk_transform` launches **one block per row** and is latency-bound on its `O(c4_len)` histogram and emit passes, not throughput-bound. Measured kernel time is essentially flat from batch 8 to batch 256 at 128k context (~36.8 us), confirming the cost is per-block scan latency rather than occupancy. ⏎  ⏎ On CDNA the wavefront is 64 lanes, so a 512-thread block is only 8 wavefronts and uses half the scan width a block ca …[truncated]

### L2-362c2ee849  (L2, 2026-08-23, sha 362c2ee849cf, PR #35908)
TITLE: config: borrowed-record reads follow the config bags (#35908)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.aiter_mla, L2.backend.sparse_mla_adapters, L2.dispatch.attention_registry, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/attention_registry.py (+7/-7); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+7/-2); python/sglang/benchmark/one_batch.py (+6/-6); python/sglang/srt/disaggregation/common/staging_handler.py (+5/-1); python/sglang/srt/disaggregation/decode.py (+11/-11); python/sglang/srt/disaggregation/prefill.py (+2/-3); python/sglang/srt/disaggregation/utils.py (+11/-15); python/sglang/srt/distributed/bootstrap.py (+10/-7); python/sglang/srt/entrypoints/grpc_bridge.py (+6/-4); python/sglang/srt/entrypoints/http_server.py (+5/-4); (+55 more)
LABELS: quant, Multi-modal, deepseek, speculative-decoding, ready-to-merge, diffusion, model-gateway, apple-silicon
BODY: ## Stack ⏎  ⏎ Part of a series that moves `ServerArgs` from "mutate the record at construction" to "resolve once, publish into config bags". Each PR stands alone (builds, passes its own tests); review bottom-up. ⏎  ⏎ | # | PR | base | ⏎ |---|----|------| ⏎ | 0 | #35904 | `main` | ⏎ | 1 | #35905 | #35904 | ⏎ | 2 | #35906 | #35905 | ⏎ | 3 | #35907 | #35906 | ⏎ | 4 | #35908 | #35907 | ⏎ | 5 | #35909 | #35908 | ⏎ | 6 | #35910 | #35909 | ⏎  ⏎ Whole-series CI vehicle (not for mer …[truncated]

### L2-092d85eb87  (L2, 2026-08-24, sha 092d85eb87f6, PR #30360)
TITLE: [Feature] Add MiniCPM-SALA support (#30360)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.pool.mla_token_kv, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/attention_registry.py (+14/-0); python/sglang/kernels/jit/csrc/minicpm_sala/get_block_table.cuh (+212/-0); python/sglang/kernels/jit/minicpm_sala/__init__.py (+3/-0); python/sglang/kernels/jit/minicpm_sala/get_block_table.py (+85/-0); python/sglang/srt/arg_groups/overrides.py (+48/-0); python/sglang/srt/configs/__init__.py (+2/-0); python/sglang/srt/configs/hybrid_arch.py (+3/-0); python/sglang/srt/configs/minicpm.py (+194/-0); python/sglang/srt/disaggregation/decode.py (+15/-0); python/sglang/srt/environ.py (+5/-0); (+34 more)
LABELS: run-ci, jit-kernel, memory-pool
BODY: ## Motivation ⏎  ⏎ Add native serving support for MiniCPM-SALA checkpoints, which interleave MiniCPM sparse softmax-attention layers with constant-decay Lightning linear-attention layers. ⏎  ⏎ The implementation integrates the model with SGLang's existing attention, linear-attention, CUDA graph, and memory-pool infrastructure. It also replaces the serving-time dependency on the original out-of-tree block-table extension with SGLang JIT kernels. ⏎  ⏎ ## …[truncated]

### L2-4c02584773  (L2, 2026-08-24, sha 4c02584773a4, PR #29143)
TITLE: Add intel_xpu to DETERMINISTIC_ATTENTION_BACKEND_CHOICES (#29143)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/xpu_backend.py (+26/-0); python/sglang/srt/server_args.py (+1/-0); test/registered/attention/test_deterministic.py (+40/-2)
LABELS: intel, xpu, run-ci, deterministic
BODY: Add intel_xpu to DETERMINISTIC_ATTENTION_BACKEND_CHOICES ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #32452228416](https://github.com/sgl-project/sglang/actions/runs/32452228416) ⏎ Latest PR Test (Extra): :x: [Run #32452228201](https://github.com/sgl-project/sglang/actions/runs/32452228201) ⏎ Latest PR Test (AMD ROCm 7.2): :x: [Run #32452228442](https://github.com/sgl-project/sglang/actions/runs/32452228442)

### L2-5b5b29d4e2  (L2, 2026-08-24, sha 5b5b29d4e2a6, PR #33354)
TITLE: [XPU] Use a fused GDN kernel from sgl-kernel for Qwen3.5 (#33354)
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+6/-0); python/sglang/srt/hardware_backend/xpu/attention/__init__.py (+0/-0); python/sglang/srt/hardware_backend/xpu/attention/xpu_gdn_backend.py (+138/-0); python/sglang/srt/layers/attention/linear/gdn_backend.py (+14/-0); python/sglang/srt/layers/attention/linear/utils.py (+4/-0); python/sglang/srt/models/qwen3_5.py (+45/-0); python/sglang/srt/server_args.py (+1/-0); test/registered/xpu/llm_models/test_xpu_qwen3_5_9b.py (+23/-3); test/registered/xpu/test_intel_xpu_linear_attn_dispatch.py (+76/-0)
LABELS: intel, xpu, run-ci
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎  ⏎ To use a fused GDN kernel provided by sgl-kernel-xpu for better performance than what the existing triton kernels in SGLang produce. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ - Added a subclass of `GDNAttnBackend` for XPU, which does some checks and calls the fused GDN kernel. ⏎ - Added a simple dispatch mechanism for the fused GDN kernel on XPU with existing paths unchanged. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ Qwen3.5-9B, GSM8K, 200 samples ⏎  ⏎ Fused GD …[truncated]

### L2-0f7ba3d115  (L2, 2026-08-24, sha 0f7ba3d11562, PR #32162)
TITLE: [HiSparse] Support hisparse multi-step swap io kernel (#32162)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh (+0/-0); python/sglang/kernels/jit/csrc/kvcacheio/hisparse_spec.cuh (+1113/-0); python/sglang/kernels/ops/kvcache/hisparse.py (+160/-5); test/registered/jit/benchmark/bench_hisparse_spec.py (+359/-0); test/registered/jit/test_hisparse_spec.py (+550/-0)
LABELS: high priority, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ The existing HiSparse swap path processes speculative decoding steps sequentially. For a four-step MTP request, this requires four independent cache lookups, swap passes, and cache-state updates. It also cannot naturally deduplicate misses shared by multiple speculative steps. ⏎  ⏎ This PR adds a standalone multi-step HiSparse swap kernel. It treats all MTP top-k tensors as one working set and resolves their device locations in one …[truncated]

### L2-91e7e84ee5  (L2, 2026-08-24, sha 91e7e84ee5a0, PR #35116)
TITLE: [SM120] flash_mla: allocate the page-split buffer outside inference mode (#35116)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.kernel.flash_mla_sm120
FILES: python/sglang/kernels/ops/attention/flash_mla_sm120.py (+10/-6)
LABELS: jit-kernel
BODY: ## Purpose ⏎  ⏎ `_split_kv_pages_to_64` keeps two lazily allocated persistent buffers in the same ⏎ `buffers` dict, with the same lifetime: allocated once on first use, reused across ⏎ autotune, CUDA graph capture and steady-state serving. ⏎  ⏎ On current main only one of them is protected: ⏎  ⏎ ```python ⏎     buf = buffers.get(key)                      # page-split destination ⏎     if buf is None or buf.shape[0] < num_dst_pages: ⏎         buf = torch.empty(...)     …[truncated]

### L2-e2b50930b9  (L2, 2026-08-25, sha e2b50930b931, PR #35676)
TITLE: [NPU] DeepSeek-V4 adapt sgl-kernel-npu ops (compressor/sparse-attn/sparse-attn-metadata) (#35676)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py (+0/-6); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+3/-0); python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py (+49/-27); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py (+2/-2); python/sglang/srt/hardware_backend/npu/extra_ops_loader.py (+30/-16); python/sglang/srt/layers/attention/xpu_backend.py (+2/-1); python/sglang/srt/mem_cache/allocation.py (+6/-7); test/registered/unit/npu/attention/test_npu_ascend_backend.py (+1/-0)
LABELS: npu, run-ci, run-ci-extra
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎  ⏎ Adaptation of  DSV4 sparse attention for NPU platforms, with host-side metadata computation and a unified op namespace. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ 1. Compute sparse_attn_sharedkv_metadata on the CPU host instead of the NPU device: device tensors are mirrored to CPU (actual_seq_lengths_q_pa_cpu, seq_lens_cpu_int), and the new host op torch.ops.npu.sparse_attn_sharedkv_metadata_host reads CPU int32 inputs directly, eliminating th …[truncated]

### L2-41e7612dee  (L2, 2026-08-25, sha 41e7612dee44, PR #36186)
TITLE: [Model] Support Nemotron 3.5 Lightning speculative decoding (#36186)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs/cookbook/autoregressive/NVIDIA/Nemotron3.5-Lightning.mdx (+1/-1); python/sglang/srt/arg_groups/overrides.py (+26/-2); python/sglang/srt/arg_groups/speculative_hook.py (+17/-13); python/sglang/srt/layers/quantization/modelopt_quant.py (+13/-4); python/sglang/srt/model_executor/model_runner.py (+9/-0); python/sglang/srt/models/dflash.py (+154/-34); python/sglang/srt/models/dspark.py (+69/-8); python/sglang/srt/models/nemotron_h.py (+42/-1); python/sglang/srt/server_args.py (+4/-9); python/sglang/srt/speculative/dflash_utils.py (+42/-0); (+8 more)
LABELS: documentation, quant, speculative-decoding, run-ci
BODY: ## Motivation ⏎  ⏎ Add clean, minimal-intrusion speculative decoding support for NVIDIA Nemotron 3.5 Lightning. This supersedes the implementation approach explored in #33554 while keeping useful ModelOpt and Nemotron model changes narrowly scoped. ⏎  ⏎ ## Modifications ⏎  ⏎ - Add ModelOpt W4A16 NVFP4 support needed by the published DFlash/DSpark draft checkpoints. ⏎ - Add Nemotron 3.5-specific DFlash and DSpark layouts, including draft embeddings, packed proj …[truncated]

### L2-04c1036bb3  (L2, 2026-08-25, sha 04c1036bb395, PR #36003)
TITLE: [Kernel] Skip reserved writes in MLA KV cache (#36003)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.kernel.set_mla_kv_buffer
FILES: python/sglang/kernels/jit/csrc/elementwise/set_mla_kv_buffer.cuh (+5/-2); python/sglang/kernels/ops/kvcache/mla_buffer.py (+36/-7); python/sglang/kernels/ops/kvcache/set_mla_kv_buffer.py (+13/-1); test/registered/kernels/benchmark/kvcache/bench_set_mla_kv_buffer.py (+1/-0); test/registered/kernels/ops/kvcache/test_set_mla_kv_buffer.py (+199/-4); test/registered/kernels/ops/test_kimi_k3_prerequisite_ops.py (+4/-3)
LABELS: run-ci, jit-kernel, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Related to #36207. ⏎  ⏎ SGLang reserves physical KV-cache slot 0 as a finite padding source for CUDA ⏎ graph and DP-attention padding. Generic KV stores protect that invariant after ⏎ #32477, but MLA-specific writers bypass `store_cache()` and still accept ⏎ `loc == 0` as a write destination. ⏎  ⏎ Undefined graph/DP-padding rows can therefore overwrite slot 0 with NaNs. A ⏎ later padded attention row may read the poisoned slot and propagate NaNs i …[truncated]

### L2-8005df61d3  (L2, 2026-08-26, sha 8005df61d32c, PR #36250)
TITLE: config: spell the parallel config tier at the call site (#36250)
SOURCES: path_core
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla, L2.backend.sparse_mla_adapters
FILES: .claude/skills/sglang-runtime-context/SKILL.md (+59/-46); python/sglang/benchmark/one_batch.py (+1/-1); python/sglang/srt/batch_overlap/two_batch_overlap.py (+1/-1); python/sglang/srt/disaggregation/common/conn.py (+17/-14); python/sglang/srt/disaggregation/encoder/http_server.py (+1/-1); python/sglang/srt/disaggregation/encoder/runtime.py (+10/-9); python/sglang/srt/disaggregation/prefill.py (+1/-1); python/sglang/srt/distributed/bootstrap.py (+2/-2); python/sglang/srt/distributed/device_communicators/triton_symm_mem_ag.py (+1/-1); python/sglang/srt/elastic_ep/elastic_ep.py (+7/-7); (+126 more)
LABELS: documentation, amd, lora, deepseek, hicache, ready-to-merge
BODY: ## Motivation ⏎  ⏎ `get_parallel()` exposes two different things through one spelling. Bare ⏎ `get_parallel().tp_size` answers from the **live** process groups; the ⏎ `configured_*_size()` accessors answered from the **published** `parallel` config. Both ⏎ spellings work everywhere, neither says which one it is, and they are not ⏎ interchangeable — they provably differ in three situations: ⏎  ⏎ 1. before `torch.distributed` is initialised, or in a process that h …[truncated]

### L2-5b7fc61306  (L2, 2026-08-26, sha 5b7fc6130613, PR #36253)
TITLE: config: resolution reads the declarations, not the fields (#36253)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/deepseek_v4_hook.py (+38/-38); python/sglang/srt/arg_groups/expert_pack_hook.py (+15/-14); python/sglang/srt/arg_groups/hisparse_hook.py (+5/-2); python/sglang/srt/arg_groups/kimi_k3_hook.py (+16/-10); python/sglang/srt/arg_groups/mega_moe_hook.py (+10/-5); python/sglang/srt/arg_groups/overrides.py (+179/-135); python/sglang/srt/arg_groups/pd_disaggregation_hook.py (+35/-32); python/sglang/srt/arg_groups/speculative_hook.py (+151/-147); python/sglang/srt/configs/model_config.py (+26/-26); python/sglang/srt/dllm/config.py (+11/-10); (+24 more)
LABELS: Multi-modal, deepseek, speculative-decoding, ready-to-merge, npu
BODY: ## Motivation ⏎  ⏎ Resolution is a chain: one resolver decides a field, the next one reads that decision. ⏎ Today that works only because `declare_resolution` writes the field as a side effect, so ⏎ the record doubles as the scratchpad for a half-finished resolution. That side effect is ⏎ what keeps `ServerArgs` from being what it should be — the raw user input — and it makes ⏎ "who decided this value" unanswerable after the fact. ⏎  ⏎ This PR moves every read t …[truncated]

### L2-413df1f8db  (L2, 2026-08-26, sha 413df1f8db4f, PR #36255)
TITLE: config: ServerArgs holds the raw input (#36255)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: .claude/skills/sglang-runtime-context/SKILL.md (+54/-25); examples/runtime/engine/save_remote_state.py (+2/-1); examples/runtime/engine/save_sharded_state.py (+2/-1); examples/runtime/token_in_token_out/token_in_token_out_vlm_engine.py (+5/-3); python/sglang/benchmark/one_batch.py (+1/-1); python/sglang/srt/arg_groups/arg_utils.py (+4/-4); python/sglang/srt/arg_groups/overrides.py (+43/-62); python/sglang/srt/entrypoints/engine.py (+2/-2); python/sglang/srt/entrypoints/http_server.py (+2/-2); python/sglang/srt/layers/attention/minimax_sparse_backend.py (+4/-5); (+24 more)
LABELS: documentation, Multi-modal, speculative-decoding, ready-to-merge, npu, model-gateway
BODY: ## Motivation ⏎  ⏎ Everything up to here made the record's fields unnecessary as a communication channel: ⏎ resolution reads the declaration stash, and the runtime readers read the published bags. ⏎ What is left is the side effect itself — `declare_resolution` still writes the field it ⏎ declares. While it does, `ServerArgs` is neither the user's input nor the resolved ⏎ configuration but a mutable mixture of both, and no code can ask "what did the user ⏎ actu …[truncated]

### L2-2d8484740d  (L2, 2026-08-26, sha 2d8484740d5e, PR #35314)
TITLE: Support deepseek v4 and kimi k3 on ssd (#35314)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+27/-1); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py (+9/-2); examples/runtime/deepseek_v4/benchmark_deepseek_5090.py (+492/-0); examples/runtime/kimi_k3/benchmark_kimi_k3_5090.py (+573/-0); python/sglang/kernels/jit/csrc/moe/expert_pack_mxfp4.cu (+565/-0); python/sglang/kernels/jit/csrc/ngram_corpus/result.h (+1/-0); python/sglang/kernels/ops/attention/dsv4/compress.py (+7/-0); python/sglang/kernels/ops/kimi_k3/attn_res.py (+4/-3); python/sglang/kernels/ops/moe/expert_pack_mxfp4.py (+113/-0); python/sglang/srt/arg_groups/expert_pack_hook.py (+198/-0); (+36 more)
LABELS: quant, deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: # [Feature] Add SSD-backed Expert Pack inference for DeepSeek-V4-Flash and Kimi-K3 ⏎  ⏎ ## Summary ⏎  ⏎ This PR adds an opt-in SSD-backed inference path for **DeepSeek-V4-Flash** and **Kimi-K3**. It allows these MoE models to run when their complete weights are larger than GPU VRAM and host RAM combined. ⏎  ⏎ Routed expert weights are stored on SSD and loaded only when selected by the router. The implementation uses: ⏎  ⏎ - an expert-major **Expert Pack* …[truncated]

### L2-3ce243da3f  (L2, 2026-08-26, sha 3ce243da3f3d, PR #36233)
TITLE: [NVIDIA] Add CUDA 13.4 container for initial Rubin support (#36233)
SOURCES: path_core, dependency_pin, body_keyword
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: docker/Dockerfile.cu134 (+1167/-0); python/sglang/kernels/aot/CMakeLists.txt (+5/-4); python/sglang/kernels/aot/cmake/flashmla.cmake (+2/-2); .github/workflows/release-docker-cu134-nightly.yml (+104/-0)
LABELS: sgl-kernel, nvidia
BODY: ## Motivation ⏎  ⏎ [CUDA 13.4 Developer Preview](https://docs.nvidia.com/cuda/developer-preview/13.4/index.html) provides initial support for Rubin hardware (sm_107). This PR adds a new docker image with CUDA 13.4 for functional Rubin support, which would be built alongside the current cu12 and cu13 containers. ⏎  ⏎ ## Modifications ⏎  ⏎ Adds `docker/Dockerfile.cu134` which is based on the standard sglang container Dockerfile, with a few key changes. U …[truncated]

### L2-20621aa14b  (L2, 2026-08-26, sha 20621aa14bda, PR #33561)
TITLE: [Model] Support Ling-3.0-flash (BailingMoeV3)  (#33561)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.npu_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+120/-5); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+3/-0); .claude/skills/sglang-runtime-context/SKILL.md (+2/-1); benchmark/kernels/fused_moe_triton/common_utils.py (+1/-0); python/sglang/kernels/aot/csrc/allreduce/custom_all_reduce.cuh (+3/-0); python/sglang/kernels/ops/attention/fla/fused_kda_conv_recurrent_verify.py (+489/-0); python/sglang/kernels/ops/attention/fla/fused_norm_gate.py (+14/-0); python/sglang/kernels/ops/attention/fla/fused_recurrent.py (+4/-2); python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py (+18/-0); python/sglang/kernels/ops/mamba/causal_conv1d_triton.py (+1/-3); (+66 more)
LABELS: documentation, quant, amd, deepseek, sgl-kernel, blackwell, npu, run-ci, jit-kernel, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Day-0 support for [inclusionAI/Ling-3.0-flash](https://huggingface.co/inclusionAI/Ling-3.0-flash) (`BailingMoeV3ForCausalLM`, `model_type=bailing_hybrid`): a hybrid MoE architecture interleaving KDA linear attention (with a safe-gate lower bound) and MLA, with MTP (NEXTN) and DSPARK speculative decoding support. ⏎  ⏎ ## Modifications ⏎  ⏎ - `BailingMoeV3ForCausalLM` model (`bailing_hybrid` config): KDA + MLA hybrid layers, 512-expert MoE,  …[truncated]

### L2-1c8f2b38cb  (L2, 2026-08-26, sha 1c8f2b38cbb3, PR #36396)
TITLE: [AMD][CI] Add DeepSeek-V4-Flash FP8 accuracy coverage on MI30x (#36396)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/workflows/nightly-test-amd-rocm720.yml (+61/-99); test/registered/amd/test_deepseek_v4_flash_fp8_mi30x.py (+139/-0); test/run_suite.py (+1/-0)
LABELS: amd, deepseek
BODY: MI30x only. The V3.x drop was split out of #36388, which now carries only its MI35x half. ⏎  ⏎ ## Motivation ⏎  ⏎ DeepSeek-V4 has no gfx942 coverage: every DSV4 job in the repo runs on `linux-mi35x-gpu-8`, and this workflow's V4 section was literally commented `MI35x only`. Meanwhile the cookbook publishes three `verified: true` MI300X Flash FP8 cells, so the docs call MI300X a supported DSV4 target while CI has never once run DSV4 there. ⏎  ⏎ Separately, `n …[truncated]

### L2-a3ae667d67  (L2, 2026-08-26, sha a3ae667d67a4, PR #35634)
TITLE: [Feature] Add DeepEPv2 (ElasticBuffer) MoE A2A backend  (#35634)
SOURCES: release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: python/sglang/kernels/ops/moe/ep_moe_kernels.py (+363/-0); python/sglang/srt/arg_groups/overrides.py (+5/-1); python/sglang/srt/environ.py (+4/-0); python/sglang/srt/layers/moe/ep_moe/layer.py (+4/-1); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+39/-0); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+192/-1); python/sglang/srt/layers/moe/moe_runner/runner.py (+9/-0); python/sglang/srt/layers/moe/token_dispatcher/__init__.py (+8/-0); python/sglang/srt/layers/moe/token_dispatcher/base.py (+19/-0); python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py (+460/-0); (+10 more)
LABELS: deepseek, run-ci, jit-kernel, release-highlight
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: > Reland of #29525, reverted by #35568 because its e2e test used an ⏎ > unregistered CI runner name. This reland restores the backend and fixes that ⏎ > registration without changing unrelated CI configuration. ⏎  ⏎ ## Motivation ⏎  ⏎ Add DeepEP v2 `ElasticBuffer` as a standalone MoE A2A backend named ⏎ `deepep_v2`, alongside the existing `deepep` backend. Its fixed-capacity ⏎ communication shapes make decode CUDA-graph capturable with both single-node ⏎  …[truncated]

### L2-b8a6adadfe  (L2, 2026-08-27, sha b8a6adadfe8c, PR #35275)
TITLE: [Bug][Spec] fix startup crash and reduce CUDA graph memory usage for speculative adaptive (#35275)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/model_executor/model_runner.py (+24/-3); python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py (+11/-2); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+6/-1); python/sglang/srt/model_executor/runner_utils/__init__.py (+1/-0); python/sglang/srt/model_executor/runner_utils/pool.py (+13/-1); python/sglang/test/kits/attention_unittest/runner_modes/speculative_draft_runner.py (+4/-3); test/registered/unit/model_executor/test_model_runner_decode_rows.py (+51/-0)
LABELS: speculative-decoding, run-ci
BODY: ## Motivation ⏎  ⏎ Three separate problems show up when `--speculative-adaptive` is enabled: ⏎ 1. **Startup crash.** `ModelRunner.max_decode_logits_rows()` sizes the shared logits buffer from the *static* `decode_num_tokens_per_req()`, but adaptive speculative decoding replays runners built for larger draft-token widths. The buffer is too small and the server fails to start. (mentioned in #30549) ⏎ 2. **IMA on the first replay.** `DecodeCudaGraphRunn …[truncated]

### L2-3402265989  (L2, 2026-08-27, sha 3402265989c6, PR #36586)
TITLE: [Core] Refactor server argument choices (#36586)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/dsa_backend.py (+1/-1); python/sglang/srt/server_args.py (+128/-162); test/manual/test_dsa_alias_cli_registry_env.py (+18/-17); test/registered/unit/disaggregation/test_kimi_k3_encoder_mode.py (+2/-2); test/registered/unit/server_args/test_server_args.py (+42/-35)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ Simplify the choice registries in `server_args.py` and keep dynamic CLI-only choices scoped to parser construction. ⏎  ⏎ ## Modifications ⏎  ⏎ - Replace trivial `add_*_choices` wrappers with bound `list.extend`/`list.append` aliases placed next to their registries. ⏎ - Inline choice lists that have no external extension surface. ⏎ - Build the token-oracle sampling choice in `ServerArgs.add_cli_args` so the environment gate is evaluated for eac …[truncated]

### L2-2ded8a6aea  (L2, 2026-08-27, sha 2ded8a6aeaeb, PR #35611)
TITLE: [AMD] Enable moe_a2a_backend=mori for DeepSeek-V4 prefill context parallelism (#35611)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/deepseek_v4.py (+7/-4); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+3/-3)
LABELS: amd, deepseek, run-ci
BODY: Enables DeepSeek-V4 prefill context parallelism to use Mori for MoE all-to-all. ⏎  ⏎ ## Motivation ⏎  ⏎ DeepSeek-V4 prefill context parallelism rejected `moe_a2a_backend=mori`, although Mori already uses the same rank-local EP dispatch/combine path as DeepEP. ⏎  ⏎ ## Changes ⏎  ⏎ - Add `mori` to the DeepSeek-V4 CP validation whitelist. ⏎ - Add `is_mori()` to the matching model-side backend gate. ⏎  ⏎ No dispatcher, collective, or compute kernel is changed. ⏎  ⏎ ## Why thi …[truncated]

### L2-9d07b9e227  (L2, 2026-08-27, sha 9d07b9e2278c, PR #33871)
TITLE: [Performance] Reduce idle DP work in breakable prefill CUDA graphs (#33871)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/batch_overlap/two_batch_overlap.py (+48/-7); python/sglang/srt/layers/radix_attention.py (+27/-0); python/sglang/srt/model_executor/forward_batch_info.py (+25/-6); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+6/-0); python/sglang/srt/models/deepseek_v4.py (+4/-0); test/registered/unit/batch_overlap/test_tbo_children_dummy_token_mask.py (+148/-0); test/registered/unit/layers/test_radix_attention.py (+111/-0)
LABELS: deepseek, run-ci, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎ Under DP attention with breakable prefill CUDA graphs, idle DP ranks may be ⏎ rewritten into fabricated `EXTEND` batches so that all ranks can follow the ⏎ same execution sequence. ⏎  ⏎ Previously, the fabricated rows were counted as real tokens. As a result: ⏎  ⏎ - MoE top-k and dispatch treated dummy rows as valid tokens. ⏎ - Busy ranks performed unnecessary expert work for tokens from idle ranks. ⏎ - Attention was executed even when a …[truncated]

### L2-46a544e0a0  (L2, 2026-08-27, sha 46a544e0a067, PR #36719)
TITLE: [Docs] GLM-5.3-Flash: point at compute-mamba-ratio for the KDA/KV pool split (#36719)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx (+4/-0)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ The GLM-5.3-Flash deployment panel never emits `--mamba-full-memory-ratio` or `--max-mamba-cache-size` (`docs/src/snippets/configs/zai-org/glm-5.3-flash.jsx` has no mamba flag), so every generated command runs at the `ServerArgs` default `0.9`. GLM-5.3-Flash is a hybrid model — a paged MLA/DSA KV pool plus a separate KDA state pool — and that split is a generic default, not a workload-tuned one: too low starves the KDA state pool a …[truncated]

### L2-7c3b5a6732  (L2, 2026-08-27, sha 7c3b5a6732fb, PR #36725)
TITLE: config: every handler declares its cuda-graph decisions (#36725)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/hardware_backend/npu/utils.py (+39/-7); python/sglang/srt/model_executor/cuda_graph_config.py (+20/-1); python/sglang/srt/server_args.py (+242/-48); test/registered/unit/server_args/test_resolution_declarations.py (+98/-0); test/registered/unit/server_args/test_server_args.py (+3/-2)
BODY: ## Motivation ⏎  ⏎ PR 2 of a five-PR series on top of the raw-input `ServerArgs` work (#36250–#36255), based on `d1f14431fdf`. Each builds on the previous one; review them in order. ⏎  ⏎ 1. `cheng/gc-p1` — config: resolution declares, and nothing writes a field ⏎ 2. `cheng/gc-p2` — config: every handler declares its cuda-graph decisions  ← **this PR** ⏎ 3. `cheng/gc-p3` — config: a parallel leaf with no live counterpart is read bare ⏎ 4. `cheng/gc-p4` — config …[truncated]

### L2-ca1d7ed8e6  (L2, 2026-08-27, sha ca1d7ed8e64c, PR #36620)
TITLE: config: a parallel leaf with no live counterpart is read bare (#36620)
SOURCES: path_core
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla, L2.backend.sparse_mla_adapters
FILES: python/sglang/benchmark/one_batch.py (+1/-1); python/sglang/compile_deep_gemm.py (+1/-1); python/sglang/srt/batch_overlap/two_batch_overlap.py (+1/-1); python/sglang/srt/disaggregation/common/conn.py (+12/-16); python/sglang/srt/disaggregation/encoder/http_server.py (+1/-1); python/sglang/srt/disaggregation/encoder/runtime.py (+7/-9); python/sglang/srt/disaggregation/prefill.py (+1/-1); python/sglang/srt/distributed/bootstrap.py (+2/-2); python/sglang/srt/distributed/device_communicators/triton_symm_mem_ag.py (+1/-1); python/sglang/srt/elastic_ep/elastic_ep.py (+6/-7); (+115 more)
LABELS: amd, lora, deepseek, hicache
BODY: ## Motivation ⏎  ⏎ PR 3 of a five-PR series on top of the raw-input `ServerArgs` work (#36250–#36255), based on `f775db03aaa`. Each builds on the previous one; review them in order. ⏎  ⏎ 1. `cheng/gc-p1` — config: resolution declares, and nothing writes a field ⏎ 2. `cheng/gc-p2` — config: every handler declares its cuda-graph decisions ⏎ 3. `cheng/gc-p3` — config: a parallel leaf with no live counterpart is read bare  ← **this PR** ⏎ 4. `cheng/gc-p4` — config …[truncated]

### L2-5640e53cab  (L2, 2026-08-27, sha 5640e53cab65, PR #36640)
TITLE: [NPU] [bugfix] Fix import of ggml_moe_a8_vec and Fix NPU MLA HiCache backup accessing missing data_ptrs (#36640)
SOURCES: subject_keyword, corpus:confirmed-reverts(reverted), body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py (+12/-10); python/sglang/srt/mem_cache/pool_host/mla.py (+8/-3)
LABELS: hicache, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 36747 (confirmed_revert, reason=other)
BODY: ## Motivation ⏎  ⏎ PR #30393 generalized the MLA HiCache backup path to support packed target and draft KV buffers. However,  `NPUMLATokenToKVPool` intentionally does not create `data_ptrs` (it uses contiguous multi-layer K/V/index-K tensors for the NPU transfer kernel).  ⏎ And, Ascend NPU does not support `ggml_moe_a8_vec` which in `sgl_kernel`. ⏎  ⏎ Errors as follows： ⏎ <img width="1103" height="85" alt="error-1" src="https://github.com/user-attachme …[truncated]

### L2-76217f6603  (L2, 2026-08-27, sha 76217f660365, PR #36747)
TITLE: Revert "[NPU] [bugfix] Fix import of ggml_moe_a8_vec and Fix NPU MLA HiCache backup accessing missing data_ptrs" (#36747)
SOURCES: subject_keyword, corpus:confirmed-reverts
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py (+10/-12); python/sglang/srt/mem_cache/pool_host/mla.py (+3/-8)
LABELS: hicache
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 36640 reason=other
BODY: Reverts sgl-project/sglang#36640 ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #33125680932](https://github.com/sgl-project/sglang/actions/runs/33125680932) ⏎ Latest PR Test (Extra): :no_entry_sign: [Run #33125680522](https://github.com/sgl-project/sglang/actions/runs/33125680522) ⏎ Latest PR Test (AMD ROCm 7.2): :no_entry_sign: [Run #33125680741](https://github.com/sgl-project/sglang/actions/runs/33125680741)

### L2-de2fb50120  (L2, 2026-08-27, sha de2fb501202b, PR #36356)
TITLE: [AMD] Enable aiter mla asm path through padding attn heads for Kimi K3 (#36356)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+110/-19); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py (+9/-1)
LABELS: amd, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ The aiter attention backend only accepts an MLA head count of 4, 8, or a ⏎ multiple of 16 in [16, 128]. Kimi-K3 has 96 query heads, so at TP=8 each rank ⏎ gets **12**.  We will see the following errors: ⏎  ⏎ ``` ⏎ [AITER] csrc/kernels/mla/reduce.cu:1264 kn_mla_reduce_v1 doesn't support the ⏎         specified settings: #heads: 12, head dimension: 128. ⏎ Fatal Python error: Aborted ⏎ ``` ⏎ This PR handles both prefill and decode padding  …[truncated]

### L2-aa0a0aa3c3  (L2, 2026-08-27, sha aa0a0aa3c303, PR #36130)
TITLE: [AMD][DSV4] perf: bound the MoRI receive buffer during decode (#36130)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/aiter.py (+100/-0); python/sglang/srt/utils/common.py (+13/-0)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ AITER sizes its quantization grid from the input row count ⏎ (`aiter/csrc/kernels/quant_kernels.cu`, `rows = input.numel() / cols`). Under ⏎ MoRI that input is the **padded** receive buffer ⏎ (`num_max_dispatch_tokens_per_rank * world_size` = 4096 * 8 = 32768 rows), not ⏎ the live tokens. `num_rows` is a device pointer used only for a per-block early ⏎ exit, so it cannot shrink the grid. ⏎  ⏎ In decode the padding dominates: at concurrency 64 a  …[truncated]

### L2-2b209711d8  (L2, 2026-08-27, sha 2b209711d8d4, PR #36119)
TITLE: [AMD][DSV4] perf: MXFP8 MoRI dispatch to match the w4a8 MoE input format (#36119)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/moe_runner/aiter.py (+15/-1); python/sglang/srt/layers/moe/token_dispatcher/moriep.py (+91/-2); test/registered/unit/layers/test_moriep_mxfp8_dispatch.py (+81/-0)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎ DSv4's MoE runs `per_1x32` (MXFP4 weights), so AITER wants fp8 activations ⏎ carrying group-32 e8m0 microscales. None of MoRI's three shipped dispatch dtypes ⏎ produce that: ⏎  ⏎ | dispatch | payload | scales | consequence | ⏎ | -------- | ------- | ------ | ----------- | ⏎ | bf16 | bf16 | none | receiver must quantize | ⏎ | fp8 | fp8 | group-128 fp32 | wrong group size -> fp8->bf16 upscale round trip | ⏎ | fp4 | fp4x2 | group-32 e8m0 | right scal …[truncated]

### L2-43b5a57dbb  (L2, 2026-08-27, sha 43b5a57dbbf6, PR #36784)
TITLE: [Docs] Feature GLM-5.3-Flash in the popular-models banner (#36784)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/intro.mdx (+1/-1); docs/src/snippets/configs/popular-models.jsx (+17/-17)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ Refresh the curated popular-models rotation (docs home hero + Cookbook home strip) now that the GLM-5.3-Flash cookbook is live: ⏎  ⏎ 1. Add **GLM-5.3-Flash** as the second entry, right after Qwen3.8-Flash-Next, and retire the Inkling entry. ⏎ 2. Point the **GLM vendor card** on the autoregressive cookbook overview at `GLM/GLM-5.3-Flash` instead of `GLM/GLM-5.2`. ⏎  ⏎ ## Modifications ⏎  ⏎ - `docs/src/snippets/configs/popular-models.jsx`: new GLM …[truncated]

### L2-1948b61ad4  (L2, 2026-08-27, sha 1948b61ad4d7, PR #36804)
TITLE: [Cookbook] Add the Hy4-Preview model page (Tencent) (#36804)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/Tencent/Hy3.mdx (+0/-1); docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx (+279/-0); docs/cookbook/autoregressive/intro.mdx (+1/-1); docs/docs.json (+1/-0); docs/src/snippets/configs/tencent/hy4-preview-benchmarks.jsx (+33/-0); docs/src/snippets/configs/tencent/hy4-preview.jsx (+574/-0)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ Add the SGLang Cookbook page for **Tencent Hy4-Preview**: a ~760B-total / ~40B-active MoE with MLA + DeepSeek Sparse Attention (DSA) on all 78 layers, Hunyuan's iHC inter-layer residual control, gated MLA with a learned attention sink, and one built-in NEXTN MTP draft layer. Text-only; BF16 (`tencent/Hy4-preview`, ~1.5TB) and MXFP8 ModelOpt (`tencent/Hy4-preview-FP8`, ~760GB) releases. ⏎  ⏎ ## Modifications ⏎  ⏎ Config-driven format only — …[truncated]

### L2-2a7fb511c9  (L2, 2026-08-28, sha 2a7fb511c948, PR #36308)
TITLE: [AMD][CI] Limit HiCache MGSM eval concurrency on ROCm (#36308)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/hicache/test_hicache_variants.py (+2/-0)
LABELS: hicache
BODY: ## Motivation ⏎  ⏎ The ROCm 7.2.4 MI300 job [`stage-b-test-1-gpu-small-amd-rocm720 (4)`](https://github.com/sgl-project/sglang/actions/runs/32367546070/job/96420375920) ⏎ failed `TestHiCacheMLA.test_mgsm_en` with scores of `0.564` and `0.612` on retry, below the `0.8` threshold. ⏎  ⏎ `MGSMEnMixin` defaults to 1024 client threads. Since MGSM English contains 250 examples, this submits the entire evaluation concurrently; the failing job reached 250 runn …[truncated]

### L2-69a49fede8  (L2, 2026-08-28, sha 69a49fede863, PR #36603)
TITLE: fix(kimi-k3): preserve dense ModelSlim MLA weights (#36603)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/quantization/expert_pack.py (+1/-0); python/sglang/srt/models/kimi_k3.py (+10/-3); test/registered/expert_pack/test_kimi_k3_gguf.py (+11/-0)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The Kimi-K3 ModelSlim W4A8 multi-node accuracy job fails after loading its 351 checkpoint shards with: ⏎  ⏎ ```text ⏎ ValueError: Kimi-K3 MLA K projection must remain GGUF Q4_0 ⏎ ``` ⏎  ⏎ Failed job: https://github.com/sgl-project/sglang/actions/runs/32936865342/job/98168349010 ⏎  ⏎ #35314 added split-GGUF K/V support for the Kimi-K3 ExpertPack path, but selected that representation through `supports_kimi_k3_quantized_latent_projections` …[truncated]

### L2-c2928e86d7  (L2, 2026-08-28, sha c2928e86d78e, PR #36789)
TITLE: config: the resolution pipeline moves out of the record (#36789)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/attention_hook.py (+627/-0); python/sglang/srt/arg_groups/cuda_graph_hook.py (+455/-0); python/sglang/srt/arg_groups/dllm_hook.py (+124/-0); python/sglang/srt/arg_groups/hicache_hook.py (+209/-0); python/sglang/srt/arg_groups/kv_cache_hook.py (+425/-0); python/sglang/srt/arg_groups/lora_hook.py (+220/-0); python/sglang/srt/arg_groups/mamba_hook.py (+154/-0); python/sglang/srt/arg_groups/memory_hook.py (+268/-0); python/sglang/srt/arg_groups/model_hook.py (+856/-0); python/sglang/srt/arg_groups/model_path_hook.py (+306/-0); (+20 more)
LABELS: lora, Multi-modal, hicache, npu
BODY: ## Motivation ⏎  ⏎ `ServerArgs` is 11327 lines. About 3000 of those are the 483 field declarations that *are* ⏎ the record; most of the rest is the resolution pipeline living inside the same class. ⏎  ⏎ The series so far made that separable. The record holds the operator's raw input and ⏎ resolution declares rather than writes, so a decision no longer has to live next to the field ⏎ it decides — it can live next to its family. `arg_groups/` already holds seven …[truncated]

### L2-ef20fab38a  (L2, 2026-08-28, sha ef20fab38a03, PR #36792)
TITLE: config: the forwarding slots go; the dispatcher calls the family directly (#36792)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/cuda_graph_hook.py (+7/-7); python/sglang/srt/arg_groups/hicache_hook.py (+4/-4); python/sglang/srt/arg_groups/lora_hook.py (+1/-1); python/sglang/srt/arg_groups/model_hook.py (+17/-5); python/sglang/srt/arg_groups/model_path_hook.py (+5/-3); python/sglang/srt/arg_groups/moe_hook.py (+1/-1); python/sglang/srt/arg_groups/parallel_hook.py (+4/-2); python/sglang/srt/arg_groups/pd_disaggregation_hook.py (+5/-3); python/sglang/srt/arg_groups/serving_hook.py (+4/-2); python/sglang/srt/arg_groups/validation_hook.py (+9/-7); (+13 more)
LABELS: lora, Multi-modal, hicache, npu
BODY: ## Motivation ⏎  ⏎ After PR A, the method left behind for each moved handler is three lines of forwarding: ⏎  ⏎ ```python ⏎ def _handle_cuda_graph_config(self): ⏎     from sglang.srt.arg_groups.cuda_graph_hook import handle_cuda_graph_config ⏎  ⏎     handle_cuda_graph_config(self) ⏎ ``` ⏎  ⏎ Ninety-three of those, 403 lines, and every one of them is also why `arg_groups/` reaches back ⏎ into the record: a moved handler calling a sibling had to go `server_args._disable_x …[truncated]

### L2-2a96ebf648  (L2, 2026-08-28, sha 2a96ebf6486d, PR #36094)
TITLE: [AMD][DSV4] perf: retune decode split-K heuristic for MI355X (#36094)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_decode.py (+28/-7); test/registered/unit/layers/test_dsv4_kv_splits_heuristic.py (+130/-0)
LABELS: run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ `_kv_splits_heuristic` picks how many KV splits DSv4 decode attention launches. ⏎ It over-split by exactly one power of two across the whole decode range: at ⏎ `H=128`/`block_h=64` it chose 8/4/2 splits for `T=32/64/128` where 4/2/1 measure ⏎ faster. ⏎  ⏎ Split-K only pays while the base grid underfills the device. Each extra split ⏎ adds a partial-buffer write plus reduce-kernel work, and once per-split K gets ⏎ short that overhead is no longer …[truncated]

### L2-4944e50e2c  (L2, 2026-08-28, sha 4944e50e2c08, PR #33576)
TITLE:  [AMD] Add Work-Centric (Lean) Attention: a persistent-CTA decode kernel for long-context serving (#33576)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.dispatch.server_args_defaults
FILES: benchmark/lean_kernel_sweep.py (+163/-0); python/sglang/kernels/ops/attention/decode_attention.py (+832/-0); python/sglang/srt/environ.py (+13/-0); python/sglang/srt/layers/attention/triton_backend.py (+119/-0); python/sglang/srt/server_args.py (+5/-0); test/registered/kernels/test_lean_attention.py (+453/-0)
LABELS: documentation, amd, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ The standard flash-decode ("SplitK") attention kernel parallelizes decode by splitting each sequence's KV cache into chunks and assigning chunks to GPU compute units (CUs). While effective at short context, it leaves performance on the table in two common serving regimes: ⏎  ⏎ - **Long-context, low-batch** — std's grid is `batch × head-tiles × num_splits` with a fixed split count, so a single (or few) long sequence cannot generate  …[truncated]

### L2-a16872767f  (L2, 2026-08-28, sha a16872767fbf, PR #36929)
TITLE: Update CUDA 13.4 image to flashinfer 0.6.18rc10, cutedsl 4.8. Fix sgl- wheel unpinning (#36929)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile.cu134 (+27/-13)
BODY: ## Motivation ⏎  ⏎ * Fix bug where locally built sgl-* wheels were replaced by pip installed ones ⏎ * Upgrade to flashinfer 0.6.18 which has the jit cache for CUDA 13.4, and cutedsl 4.8 which supports sm_107 ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/ …[truncated]

### L2-3760296be8  (L2, 2026-08-28, sha 3760296be814, PR #35762)
TITLE: [PD] Pack DCP1→DCP-N PD KV transfers into dest-contiguous RDMA blocks (#35762)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/kvcache/__init__.py (+1/-0); python/sglang/kernels/ops/kvcache/pd_dcp_gather.py (+66/-0); python/sglang/srt/disaggregation/common/conn.py (+20/-0); python/sglang/srt/disaggregation/common/dcp_pack.py (+120/-0); python/sglang/srt/disaggregation/common/staging_buffer.py (+14/-16); python/sglang/srt/disaggregation/mooncake/conn.py (+29/-4); python/sglang/srt/disaggregation/nixl/conn.py (+81/-37); test/registered/kernels/ops/kvcache/test_pd_dcp_gather.py (+41/-0); test/registered/unit/disaggregation/test_dcp_pack.py (+120/-0); test/registered/unit/disaggregation/test_nixl_backend_basic.py (+89/-3)
LABELS: documentation, run-ci, jit-kernel, bypass-fastfail
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Cyclic DCP ownership made the relayout path emit one RDMA per token. Gather owned MLA rows on prefill first so Mooncake/NIXL can send page-sized or larger blocks. DSPARK is left to a follow-up. ⏎  ⏎ ## Motivation ⏎  ⏎ DCP assigns KV ownership round-robin **per token** (`owned_offsets = arange(rank, num_kv_tokens, dcp_size)`), so on a DCP1 -> DCP-N PD transfer the source indices are strided and `group_concurrent_contiguous` cannot merge them: the relayout …[truncated]

### L2-24c9251ac5  (L2, 2026-08-28, sha 24c9251ac52a, PR #36714)
TITLE: [AMD][Spec][PD] Enable the PD DSA fused-TopK seed remap on ROCm (#36714)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/dsa/utils.py (+1/-1); test/registered/unit/disaggregation/test_disaggregation_wire.py (+24/-12)
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ [PR #31477](https://github.com/sgl-project/sglang/pull/31477) added a decode-local remap so a GLM-5.2 MTP IndexShare seed can enter the allocator-local fused TopK domain under PD disaggregation. The remap is gated on `is_cuda()`: ⏎  ⏎ ```python ⏎ def should_remap_pd_dsa_seed_to_local_slots() -> bool: ⏎     return ( ⏎         is_cuda() ⏎         and envs.SGLANG_DSA_FUSE_TOPK.get() ⏎         ... ⏎ ``` ⏎  ⏎ On ROCm that gate is False, so the remap never  …[truncated]

### L2-7f2ee22b70  (L2, 2026-08-28, sha 7f2ee22b70a7, PR #36915)
TITLE: [AMD] Fix eager metadata for AITER EAGLE draft extend (#36915)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+13/-13)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ The AITER unified-attention path for non-MLA EAGLE draft extend can pass stale query sequence metadata to the kernel during eager execution. The regression was introduced by #30105, which enabled this path by default through `SGLANG_AITER_UNIFIED_DRAFT_EXTEND`. ⏎  ⏎ The CUDA Graph metadata path builds the correct query indptr, so tests whose batch sizes remain within the captured graph range do not expose the issue. The failing path is …[truncated]

### L2-48b88e1256  (L2, 2026-08-29, sha 48b88e12562e, PR #36896)
TITLE: config: the resolution pipeline's dispatcher leaves the record (#36896)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/pipeline.py (+361/-0); python/sglang/srt/arg_groups/platform_hook.py (+8/-0); python/sglang/srt/server_args.py (+4/-374); test/registered/unit/server_args/test_model_config_reads_resolved_input.py (+13/-8); test/registered/unit/server_args/test_resolution_declarations.py (+12/-8); test/registered/unit/server_args/test_resolution_is_reproducible.py (+13/-9); test/registered/unit/server_args/test_resolution_reads_the_declarations.py (+21/-57)
LABELS: npu
BODY: ## The series ⏎  ⏎ Stacked, each PR based on the one above it. Read them in order; every boundary is ⏎ self-sufficient and green on its own. ⏎  ⏎ 1. [#36896](https://github.com/sgl-project/sglang/pull/36896) — the resolution pipeline's dispatcher leaves the record ← **you are here** ⏎ 2. [#36972](https://github.com/sgl-project/sglang/pull/36972) — the callbacks into the record go to zero ⏎ 3. [#36973](https://github.com/sgl-project/sglang/pull/36973) — six mor …[truncated]

### L2-b65e677e48  (L2, 2026-08-29, sha b65e677e489d, PR #36972)
TITLE: config: the resolution callbacks into the record go to zero (#36972)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.backend.tokenspeed_mla, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/attention_registry.py (+6/-1); python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+2/-4); python/sglang/srt/arg_groups/attention_hook.py (+15/-6); python/sglang/srt/arg_groups/cuda_graph_hook.py (+132/-15); python/sglang/srt/arg_groups/expert_pack_hook.py (+3/-1); python/sglang/srt/arg_groups/hicache_hook.py (+3/-1); python/sglang/srt/arg_groups/hisparse_hook.py (+5/-2); python/sglang/srt/arg_groups/kv_cache_hook.py (+19/-11); python/sglang/srt/arg_groups/lora_hook.py (+15/-8); python/sglang/srt/arg_groups/memory_hook.py (+125/-15); (+40 more)
LABELS: lora, Multi-modal, speculative-decoding, hicache, npu
BODY: ## The series ⏎  ⏎ Stacked, each PR based on the one above it. Read them in order; every boundary is ⏎ self-sufficient and green on its own. ⏎  ⏎ 1. [#36896](https://github.com/sgl-project/sglang/pull/36896) — the resolution pipeline's dispatcher leaves the record ⏎ 2. [#36972](https://github.com/sgl-project/sglang/pull/36972) — the callbacks into the record go to zero ← **you are here** ⏎ 3. [#36973](https://github.com/sgl-project/sglang/pull/36973) — six mor …[truncated]

### L2-4d53767b09  (L2, 2026-08-29, sha 4d53767b0942, PR #36975)
TITLE: config: the lazy imports that buy nothing become eager (#36975)
SOURCES: path_core
ARTIFACT_HINTS: L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+2/-4); python/sglang/srt/arg_groups/attention_hook.py (+17/-30); python/sglang/srt/arg_groups/cuda_graph_hook.py (+3/-6); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+2/-4); python/sglang/srt/arg_groups/dllm_hook.py (+4/-7); python/sglang/srt/arg_groups/expert_pack_hook.py (+5/-2); python/sglang/srt/arg_groups/hicache_hook.py (+1/-1); python/sglang/srt/arg_groups/hisparse_hook.py (+5/-6); python/sglang/srt/arg_groups/kv_cache_hook.py (+3/-2); python/sglang/srt/arg_groups/memory_hook.py (+3/-6); (+28 more)
LABELS: documentation, quant, lora, Multi-modal, deepseek, speculative-decoding, hicache, npu, diffusion, jit-kernel
BODY: ## The series ⏎  ⏎ Stacked, each PR based on the one above it. Read them in order; every boundary is ⏎ self-sufficient and green on its own. ⏎  ⏎ 1. [#36896](https://github.com/sgl-project/sglang/pull/36896) — the resolution pipeline's dispatcher leaves the record ⏎ 2. [#36972](https://github.com/sgl-project/sglang/pull/36972) — the callbacks into the record go to zero ⏎ 3. [#36973](https://github.com/sgl-project/sglang/pull/36973) — six more runtime readers a …[truncated]

### L2-f60bc73c58  (L2, 2026-08-29, sha f60bc73c5836, PR #33614)
TITLE: [Spec] Fix Dspark and Dflash state divergence across TP rank (#33614)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+5/-0); python/sglang/srt/speculative/dflash_worker_v2.py (+108/-68); python/sglang/srt/speculative/dspark_components/dspark_draft.py (+20/-7); python/sglang/srt/speculative/dspark_components/dspark_draft_sampler.py (+33/-15); python/sglang/srt/speculative/dspark_components/dspark_planner.py (+9/-13); python/sglang/srt/speculative/dspark_components/dspark_verify.py (+17/-0); python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py (+35/-6); python/sglang/srt/speculative/spec_tp_sync.py (+129/-0)
LABELS: run-ci
ISSUES: #33289 [Bug] Multi-node TP rank-divergence deadlock: one rank wedges in NCCL proxy append (logits all-gather), peer idles at request broadcast — DeepSeek-V4 + DSpark on 2× DGX Spark (GB10)
BODY: ## Motivation ⏎  ⏎ Related to: https://github.com/sgl-project/sglang/issues/33289 ⏎  ⏎ To fix https://github.com/sgl-project/sglang/issues/33289 bug, upgrade NCCL to the newest version (2.30.7). ⏎  ⏎ When TP > 1, Dspark makes serveral sampling decisions on reach rank:  ⏎ 1. The draft Markov chain samples proposal tokens step by step (with in-graph philox noise introduced since https://github.com/sgl-project/sglang/pull/33298) ⏎ 2. Target verify derives ` …[truncated]

### L2-fbecd75c83  (L2, 2026-08-29, sha fbecd75c8313, PR #36515)
TITLE: [AMD] fix: do not emit a shared-expert marker twice on the per-rank slot path (#36515)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/moe/topk.py (+14/-2)
LABELS: run-ci
BODY: ## Motivation ⏎  ⏎ On the per-rank fused shared-slot path the shared expert is appended after the ⏎ gate, by `fused_append_remap_shared_experts_deepep`. But `select_experts` also ⏎ passes `num_fused_shared_experts` down into the gate, so both of them emit a ⏎ shared marker. ⏎  ⏎ Two things go wrong, neither of which raises: ⏎  ⏎ **A routed expert is lost.** The gate is asked for ⏎ `K_routed = top_k - num_fused_shared_experts` slots and then spends one of them ⏎ on its …[truncated]

### L2-cdbfe90b4a  (L2, 2026-08-29, sha cdbfe90b4a6c, PR #36798)
TITLE: [HiCache] Align chunked CUDA host registrations (#36798)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.pool.mla_token_kv
FILES: python/sglang/srt/environ.py (+2/-0); python/sglang/srt/mem_cache/memory_pool_host.py (+4/-0); python/sglang/srt/mem_cache/pool_host/common.py (+95/-19); python/sglang/srt/mem_cache/pool_host/dsa.py (+1/-0); python/sglang/srt/mem_cache/pool_host/mamba.py (+3/-0); python/sglang/srt/mem_cache/pool_host/mha.py (+12/-0); python/sglang/srt/mem_cache/pool_host/mla.py (+5/-0); test/registered/unit/mem_cache/test_hicache_host_register.py (+412/-0)
LABELS: hicache, run-ci, run-ci-extra, memory-pool
BODY: ## Motivation ⏎  ⏎ Large HiCache host pools can exceed the practical size of a single `cudaHostRegister` call. Splitting registration at arbitrary byte offsets is also unsafe for page-first copies because one copied page may span two registered ranges, causing `cudaMemcpyBatchAsync` to fail with `invalid argument`. ⏎  ⏎ ## Changes ⏎  ⏎ - Register large host buffers in configurable chunks (`SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB`, default 256 GB). ⏎ - Rou …[truncated]

### L2-7e751153eb  (L2, 2026-08-30, sha 7e751153eb59, PR #37086)
TITLE: [Config] Round 5.1: the published-side readers ask the bags, and a platform fact gets one address (#37086)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.tokenspeed_mla, L2.backend.sparse_mla_adapters, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/attention_registry.py (+6/-4); python/sglang/srt/arg_groups/attention_hook.py (+12/-18); python/sglang/srt/arg_groups/cuda_graph_hook.py (+7/-6); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+2/-2); python/sglang/srt/arg_groups/dllm_hook.py (+2/-2); python/sglang/srt/arg_groups/hisparse_hook.py (+4/-10); python/sglang/srt/arg_groups/kimi_k3_hook.py (+3/-4); python/sglang/srt/arg_groups/kv_cache_hook.py (+5/-13); python/sglang/srt/arg_groups/mamba_hook.py (+10/-14); python/sglang/srt/arg_groups/model_hook.py (+16/-17); (+138 more)
LABELS: quant, amd, Multi-modal, deepseek, speculative-decoding, hicache, blackwell, npu, unified-radix-cache
BODY: ## Motivation ⏎  ⏎ Round 4 got `arg_groups` to the point where resolution never writes a field on ⏎ `ServerArgs`: handlers *declare* what they decide, and `publish()` projects the ⏎ result into read-only config bags. The reader side was still half-converted — ⏎ runtime code read raw `server_args` fields, two production names existed only so ⏎ tests could patch them, and a platform fact such as "this box is SM100" had one ⏎ copy per importing module, so it coul …[truncated]

### L2-4bea51d885  (L2, 2026-08-30, sha 4bea51d88553, PR #34602)
TITLE: feat(unified-memory): dense KV views for uniform-row MHA/SWA models (#34602)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.backend.flashinfer_mla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.pool.mla_token_kv, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+4/-4); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+26/-22); python/sglang/kernels/ops/attention/decode_attention.py (+13/-54); python/sglang/kernels/ops/attention/metadata.py (+2/-2); python/sglang/kernels/ops/kvcache/__init__.py (+0/-1); python/sglang/kernels/ops/kvcache/cache_move.py (+0/-182); python/sglang/kernels/ops/kvcache/kv_indices.py (+3/-3); python/sglang/srt/arg_groups/kv_cache_hook.py (+31/-6); python/sglang/srt/layers/attention/flashattention_backend.py (+4/-4); python/sglang/srt/layers/attention/triton_backend.py (+2/-2); (+20 more)
LABELS: deepseek, blackwell, run-ci, jit-kernel, bypass-fastfail, memory-pool
BODY: > **Note on scope — this PR was split**. It previously contained both the dense-view feature and the ⏎ > write-location refactor that builds on it. Reviewing a new layout and a cross-cutting refactor together ⏎ > made both harder to judge, so the refactor now lives in separate PRs that stack on this one. This PR ⏎ > is now purely the layout change. Stack order: **this PR → #35247 → #35245 → #34613**. ⏎  ⏎ ## Motivation ⏎  ⏎ The unified memory pool stores MHA a …[truncated]

### L2-4f761e8649  (L2, 2026-08-30, sha 4f761e8649c4, PR #36954)
TITLE: [Deps] Bump FlashInfer to 0.6.18 (#36954)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); docker/kimi_k3/apply_deepep_k3_patch.sh (+0/-169); docker/kimi_k3/kimi_k3_cu12.Dockerfile (+0/-113); docker/kimi_k3/kimi_k3_cu13.Dockerfile (+0/-101); python/pyproject.toml (+1/-1); docker/Dockerfile.cu134 (+1/-1); docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1); python/sglang/srt/utils/common.py (+1/-1); test/registered/unit/mem_cache/test_unified_mamba_views.py (+4/-4)
LABELS: documentation, dependencies, run-ci, bypass-fastfail, run-ci-extra, release-highlight
BODY: --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #33289293501](https://github.com/sgl-project/sglang/actions/runs/33289293501) ⏎ Latest PR Test (Extra): :white_check_mark: [Run #33349424388](https://github.com/sgl-project/sglang/actions/runs/33349424388) ⏎ Latest PR Test (AMD ROCm 7.2): :x: [Run #33289293415](https://github.com/sgl-project/sglang/actions/runs/33289293415)

### L2-29578d5578  (L2, 2026-08-30, sha 29578d5578af, PR #35245)
TITLE: refactor(unified-memory): translate the KV write location once, at ForwardBatch construction (#35245)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.backend.sparse_mla_adapters, L2.pool.mla_token_kv, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+10/-24); python/sglang/kernels/ops/kvcache/__init__.py (+1/-0); python/sglang/kernels/ops/kvcache/kv_indices.py (+34/-10); python/sglang/kernels/ops/kvcache/kv_read_table.py (+140/-0); python/sglang/srt/arg_groups/kv_cache_hook.py (+15/-5); python/sglang/srt/layers/attention/triton_backend.py (+91/-142); python/sglang/srt/mem_cache/kv_index_translator.py (+358/-0); python/sglang/srt/mem_cache/memory_pool.py (+35/-26); python/sglang/srt/mem_cache/multi_ended_allocator.py (+10/-0); python/sglang/srt/mem_cache/unified_memory_pool.py (+3/-4); (+18 more)
LABELS: deepseek, blackwell, jit-kernel, memory-pool
BODY: > **Note on scope — this PR was split out.** It was previously part of #34602 alongside the  ⏎ > dense per-layer view feature. Separating them lets the layout change and this refactor be  ⏎ > reviewed on their own terms. The stack was also reordered per review: this PR now sits ON  ⏎ > the read-path choke point (#35247) and keeps its state there, so nothing is stored on the  ⏎ > ForwardBatch. Stack order: **#34602 → #35247 → this PR → #34613**. ⏎ > ⏎ > Only  …[truncated]

### L2-8bb776dc48  (L2, 2026-08-30, sha 8bb776dc48b0, PR #34613)
TITLE: feat(unified-memory): read unified pool from attention backends fa3/flashinfer/trtllm_mha/flashmla (#34613)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.flashmla, L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.backend.cutedsl_mla, L2.pool.mla_token_kv, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/kernels/ops/attention/metadata.py (+57/-57); python/sglang/srt/layers/attention/base_attn_backend.py (+6/-0); python/sglang/srt/layers/attention/cutedsl_mla_backend.py (+1/-1); python/sglang/srt/layers/attention/flashattention_backend.py (+74/-49); python/sglang/srt/layers/attention/flashinfer_backend.py (+92/-27); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+49/-46); python/sglang/srt/layers/attention/flashmla_backend.py (+42/-26); python/sglang/srt/layers/attention/tbo_backend.py (+1/-0); python/sglang/srt/layers/attention/trtllm_mha_backend.py (+67/-18); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+70/-60); (+21 more)
LABELS: deepseek, blackwell, run-ci, jit-kernel, bypass-fastfail, unified-radix-cache, memory-pool
BODY: > **Note on scope — this PR was split.** It previously contained both the read-path refactor  ⏎ > and this backend migration. The refactor is now a separate PR that this one stacks on, so  ⏎ > the mechanism can be reviewed independently of the per-backend work.  ⏎ > Stack order: **#34602 → #35245 → #35247 → this PR**. ⏎ > ⏎ > Only the last 8 commits are for this PR. The other commits are from the stacked PRs. ⏎  ⏎ ## Motivation ⏎  ⏎ With the read path behind a sin …[truncated]

### L2-f61bb7b40a  (L2, 2026-08-31, sha f61bb7b40a4e, PR #37170)
TITLE: [unified-memory] Drop the vacated 'dense' qualifier and the restating comments (#37170)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.pool.mla_token_kv, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+2/-2); python/sglang/kernels/ops/kvcache/kv_read_table.py (+2/-2); python/sglang/srt/mem_cache/memory_pool.py (+3/-3); python/sglang/srt/mem_cache/unified_memory_pool.py (+1/-1); test/registered/models_e2e/test_kimi_linear_unified_memory.py (+4/-3); test/registered/page_major/test_page_major_gpt_oss.py (+2/-2); test/registered/page_major/test_page_major_qwen_hybrid.py (+1/-1); test/registered/unit/layers/attention/test_flashattention_graph_metadata.py (+0/-1); test/registered/unit/mem_cache/test_full_loc_fast_path.py (+4/-4); test/registered/unit/mem_cache/test_kv_index_translator.py (+5/-5); (+8 more)
LABELS: deepseek, blackwell, run-ci, jit-kernel, memory-pool
BODY: Stacked on #34613. ⏎  ⏎ Two cleanups the unified-memory stack left behind. ⏎  ⏎ **1. Drop the vacated `dense` qualifier from the KV vocabulary.** `dense` named a ⏎ per-layer KV view only in contrast to the strided MHA view, which no longer ⏎ exists, so the word marks nothing. Where it qualified a view it is dropped; where ⏎ it named the id space the views index it becomes `kernel-facing`, the term the ⏎ translator already uses. ⏎  ⏎ Deliberately untouched, because t …[truncated]

### L2-3865efc9f7  (L2, 2026-08-31, sha 3865efc9f7e8, PR #36871)
TITLE: [AMD] support gfx1250 on ROCM 10 (#36871)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.kernel.triton_decode_lightllm, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py (+56/-32); .github/workflows/release-docker-amd-rocm7_15-nightly.yml (+0/-109); docker/rocm.Dockerfile (+158/-43); python/pyproject_other.toml (+1/-5); python/sglang/kernels/aot/csrc/allreduce/quick_all_reduce_base.h (+6/-1); python/sglang/kernels/aot/setup_rocm.py (+1/-1); python/sglang/kernels/jit/csrc/moe/moe_fused_gate.cuh (+15/-6); python/sglang/kernels/ops/attention/decode_attention.py (+33/-4); python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_decode.py (+28/-9); python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_prefill.py (+2/-2); (+25 more)
LABELS: high priority, amd, dependencies, deepseek, sgl-kernel, run-ci, jit-kernel, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Based on https://github.com/sgl-project/sglang/pull/32754 (gfx1250 enablement) and https://github.com/sgl-project/sglang/pull/36434 (ROCm 10 release images), see those PRs for the details of each. ⏎  ⏎ On top of them, this adds the gfx1250-rocm1000 Docker target on the ROCm 10.0.0 GA wheel channel, gfx1250 detection in the AMD CI dependency installer, and the gfx1250 kernel/model fixes with the MI45x accuracy tests. ⏎  ⏎ ## Modificat …[truncated]

### L2-bb5e619860  (L2, 2026-08-31, sha bb5e619860f3, PR #37116)
TITLE: [diffusion] perf: absorb Qwen-Image output projection biases (#37116)
SOURCES: subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/diffusion/norm_scale_shift.cuh (+117/-1); python/sglang/kernels/ops/diffusion/__init__.py (+2/-0); python/sglang/kernels/ops/diffusion/norm/norm_scale_shift_jit.py (+70/-0); python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py (+130/-22); test/registered/kernels/ops/diffusion/test_qwen_output_bias_absorption.py (+123/-0)
LABELS: run-ci, diffusion, jit-kernel, run-ci-extra, mergeable
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Qwen-Image's ModelOpt FP8/NVFP4 path materializes output-projection bias before the attention and feed-forward residual updates. In a GB300 profile, the denoiser launched 673 BF16 add kernels per profiled step; 224 of those launches disappear when the four per-block output biases are consumed by the following residual operation. ⏎  ⏎ This implements the Qwen-Image bias-absorption optimization described in [Agentic Kernels in Production …[truncated]

### L2-7700602278  (L2, 2026-08-31, sha 7700602278ec, PR #35281)
TITLE: [PD] Align defensive protocol behavior across Mooncake, NIXL, and Mori (#35281)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/mooncake/conn.py (+2/-2); python/sglang/srt/disaggregation/mori/conn.py (+31/-2); python/sglang/srt/disaggregation/nixl/conn.py (+18/-2)
LABELS: run-ci
BODY: - RFC: [PD disaggregation: single protocol layer, per-backend transport #33861](https://github.com/sgl-project/sglang/issues/33861) ⏎ - Staged implementation plan and PR tracking: [PD shared-protocol implementation plan #34510](https://github.com/sgl-project/sglang/issues/34510) ⏎ - Preceding PR: [NIXL Prefill bootstrap timeout #34692](https://github.com/sgl-project/sglang/pull/34692) ⏎  ⏎ ## Background ⏎  ⏎ #34692 added the missing NIXL Prefill bootst …[truncated]

### L2-4dc7dc8518  (L2, 2026-08-31, sha 4dc7dc851880, PR #35244)
TITLE: [Fix] Transformers-fallback (GPT-NeoX) + KV pool config (DeepSeek-VL2) (#35244)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/configs/model_config.py (+1/-1); python/sglang/srt/models/transformers.py (+5/-1); test/registered/unit/configs/test_model_config_shapes.py (+14/-0); test/registered/unit/model_loader/test_transformers_fallback.py (+77/-0)
LABELS: intel, run-ci
BODY: ## Summary ⏎  ⏎ Three small fixes surfaced by consecutive downstream accuracy runs. All three are one-liners (plus comments); the first two live in the Transformers-fallback layer, the third in `ModelConfig`. ⏎  ⏎ ### 1. `hf_to_sglang_mapper` — map `gpt_neox.` and `embed_out.` ⏎  ⏎ `GPTNeoXForCausalLM` has no native SGLang impl → falls back to `TransformersForCausalLM`. The fallback calls `AutoModel.from_config(config)`, which returns the bare `GPTNeoXModel` …[truncated]

### L2-961beee9e5  (L2, 2026-08-31, sha 961beee9e521, PR #35154)
TITLE: fix(unified-memory): four boot/correctness fixes on the hybrid model paths (#35154)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.backend.aiter_mla
FILES: python/sglang/kernels/jit/csrc/inkling/causal_conv1d.cuh (+1/-1); python/sglang/kernels/jit/csrc/inkling/draft_extend_sconv.cuh (+1/-1); python/sglang/kernels/jit/csrc/inkling/fused_decode_update.cuh (+1/-1); python/sglang/kernels/jit/csrc/inkling/gather_scatter_sconv.cuh (+1/-1); python/sglang/kernels/jit/csrc/inkling/inkling_ar_fused_decode.cuh (+2/-2); python/sglang/kernels/jit/csrc/inkling/update_sconv_cache.cuh (+1/-1); python/sglang/srt/environ.py (+3/-2); python/sglang/srt/layers/attention/aiter_backend.py (+0/-2); python/sglang/srt/layers/attention/flashattention_backend.py (+0/-3); python/sglang/srt/mem_cache/multi_ended_allocator.py (+181/-40); (+7 more)
LABELS: run-ci, jit-kernel, unified-radix-cache, memory-pool
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Four independent bugs on the hybrid model paths, each of which either ⏎ prevents a model from booting or silently degrades behaviour. They are  ⏎ grouped here because they are all small, self-contained fixes to the same  ⏎ subsystem, each with its own regression test. ⏎  ⏎ 1. **A model that is both mambaish and hybrid-SWA cannot boot on ⏎    `--attention-backend triton`.** `TritonAttnBackend.__init__` resolves ⏎    `v_head_dim` from …[truncated]

### L2-98cb3535b7  (L2, 2026-08-31, sha 98cb3535b752, PR #35158)
TITLE: feat(unified-memory): byte-budget sizing, feasibility floor, and a conservation verifier (#35158)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/schedule_batch.py (+8/-4); python/sglang/srt/managers/scheduler.py (+7/-0); python/sglang/srt/mem_cache/allocator/base.py (+33/-0); python/sglang/srt/mem_cache/kv_cache_configurator.py (+29/-1); python/sglang/srt/mem_cache/multi_ended_allocator.py (+69/-0); python/sglang/srt/mem_cache/unified_memory_pool.py (+104/-16); python/sglang/srt/model_executor/pool_configurator.py (+7/-0); test/registered/unit/mem_cache/test_unified_byte_accounting.py (+157/-0); test/registered/unit/mem_cache/test_unified_byte_budget_sizing.py (+244/-0)
LABELS: run-ci, jit-kernel, unified-radix-cache, memory-pool
BODY: > **Stacked PR.** Base is (#35154), not `main`. The ⏎ > review diff here is this PR's own 4 commits; please read #35154 first. ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ Three gaps in how the unified memory pool is sized and checked: ⏎  ⏎ 1. **The buffer was sized by re-summing ratio-derived token counts**, not from ⏎    the profiled byte budget it was given. The SWA split floors the budget by ⏎    the per-token cell size and then page-aligns each side's token count …[truncated]

### L2-ef9e58fd6d  (L2, 2026-08-31, sha ef9e58fd6d01, PR #35177)
TITLE: feat(unified-memory): three sub-pools for mamba + hybrid-SWA models (#35177)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/managers/schedule_policy.py (+10/-7); python/sglang/srt/managers/scheduler_components/invariant_checker.py (+16/-2); python/sglang/srt/managers/scheduler_components/pool_stats_observer.py (+15/-2); python/sglang/srt/mem_cache/common.py (+15/-1); python/sglang/srt/mem_cache/kv_cache_configurator.py (+138/-2); python/sglang/srt/mem_cache/multi_ended_allocator.py (+1406/-130); python/sglang/srt/mem_cache/unified_memory_pool.py (+281/-33); test/registered/models_e2e/test_inkling_unified.py (+240/-0); test/registered/unit/disaggregation/test_unified_memory_move_gate.py (+12/-1); test/registered/unit/mem_cache/test_multi_ended_allocator.py (+479/-2); (+4 more)
LABELS: jit-kernel, bypass-fastfail, unified-radix-cache, memory-pool
BODY: > **Stacked PR.** This builds on #35154 and #35158 and should be reviewed after ⏎ > them. Its own change is the last 13 commits, 7 source + 6 test files, +4359/−179. ⏎ > The rest of the diff shown here is those two PRs.  ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ Unified memory currently supports exactly two sub-pools, so it cannot serve a ⏎ model whose KV state comes in three kinds — full-attention KV, sliding-window ⏎ KV, and recurrent (mamba/conv) state. Inkling …[truncated]

### L2-f50b4ad7ae  (L2, 2026-08-31, sha f50b4ad7ae21, PR #33926)
TITLE: [DCP] Support decode context parallelism on the trtllm_mla decode path (#33926)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.backend.cutedsl_mla, L2.backend.tokenspeed_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/cutedsl_mla_backend.py (+16/-203); python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+6/-188); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+247/-9); test/registered/dcp/test_tokenspeed_mla_dcp_metadata.py (+0/-87); test/registered/dcp/test_trtllm_mla_family_dcp_metadata.py (+250/-0)
LABELS: blackwell, run-ci, release-highlight
BODY: ## Motivation ⏎  ⏎ Draft for the unassigned `trtllm_mla` decode path item in the DCP roadmap #29736 (also relevant to #26432). ⏎  ⏎ `trtllm_mla` has no DCP decode path. `forward_decode` builds its page table and `seq_lens` from the global sequence lengths and returns a bare tensor, while under DCP the model routes decode through `attn_mqa_for_dcp_decode` and expects a rank-local `(out, lse)` pair. The DCP code already in `trtllm_mla_backend.py` cover …[truncated]

### L2-22337e9c56  (L2, 2026-08-31, sha 22337e9c5654, PR #37307)
TITLE: fix(unified-memory): forward the KV-index translator through every wrapper backend (#37307)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/dots_hybrid_backend.py (+2/-0); python/sglang/srt/layers/attention/hybrid_attn_backend.py (+1/-0); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+1/-0); python/sglang/srt/layers/attention/minimax_sparse_backend.py (+1/-0); python/sglang/srt/mem_cache/kv_index_translator.py (+16/-0); python/sglang/srt/model_executor/model_runner.py (+3/-0); test/registered/models_e2e/test_kimi_linear_models.py (+22/-1); test/registered/unit/layers/attention/test_kv_translate_ownership.py (+117/-0)
BODY: ## Motivation ⏎  ⏎ `AttentionBackend.kv_index_translator` is a class attribute that defaults to ⏎ `None`. A backend that **wraps** another and does not re-expose the inner ⏎ backend's copy therefore answers "this backend needs no translation". ⏎  ⏎ Every producer that reaches the translator through the *live backend* rather ⏎ than through a runner then skips translation silently. The MLA ⏎ chunked-prefix-cache path is one: ⏎ `ForwardBatchDeepseekMHAMixin.prepare_c …[truncated]

### L2-8a191554e3  (L2, 2026-08-31, sha 8a191554e379, PR #34647)
TITLE: [AMD] Enable 12-head MLA aiter fp8 Gluon decode (batched bh16bn128). (#34647)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:kernel-correctness-cases(introducing), corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.aiter_mla, L2.kernel.aiter_mla_gluon
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+191/-59); python/sglang/srt/layers/attention/aiter_mla_gluon.py (+250/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py (+10/-1); python/sglang/srt/environ.py (+4/-0); test/registered/attention/test_mla_gluon_h12_fp8.py (+205/-0)
LABELS: amd, run-ci, jit-kernel, amd-aiter-not-ready, run-ci-extra
DEEP_STUDY: deep-study: introduced the defect fixed in case sglang:44a92e54b9 (fix PR 37438) || deep-study performance PR (precision_format)
BODY: Enable 12-head MLA aiter fp8 Gluon decode on gfx950 for Kimi-K3 TP8 (12 local heads). ⏎ If aiter is not updated to the target version, fall back to use the existing zero-pad + mla_decode_fwd. No behavior change on older containers until aiter/Triton are upgraded. below is the dependency: ⏎ **aiter runtime dependencies** (container/image, not pinned in this repo): ⏎ - [ROCm/aiter#4480](https://github.com/ROCm/aiter/pull/4480) (**merged** to `main`, [ …[truncated]

### L2-44a92e54b9  (L2, 2026-09-01, sha 44a92e54b9ef, PR #37438)
TITLE: [AMD] fix aiter cannot get heuristic kernel regression (#37438)
SOURCES: symbol_pickaxe, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+10/-3)
LABELS: amd, run-ci
DEEP_STUDY: deep-study correctness case sglang:44a92e54b9: class=hardware_compiler_specific; symptom=crash_or_exception; introducing=#34647
BODY: ## Motivation ⏎  ⏎  ⏎ https://github.com/sgl-project/sglang/pull/34647 introduced gluon into aiter backend also `SGLANG_AITER_MLA_GLUON` env is enabled by default. This behaviour change also introduces the regression for previous aiter asm path as below: ⏎ ```bash ⏎ SGLANG_AITER_MLA_GLUON=0 \          <--------------- disable gluon to use aiter asm but server crashed ⏎ SGLANG_USE_AITER=1 \ ⏎ SGLANG_AITER_K3_OPT=1 \ ⏎ AITER_FLYDSL_FORCE=1 \ ⏎ AITER_SITUV2_ …[truncated]

### L2-9a05b470fa  (L2, 2026-09-01, sha 9a05b470fa84, PR #36911)
TITLE: [Memory] Size the CUDA graph pool from warmup measurements and fix graph-pool borrowing (#36911)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+3/-0); python/sglang/srt/managers/scheduler.py (+7/-0); python/sglang/srt/model_executor/model_runner.py (+14/-0); python/sglang/srt/model_executor/model_runner_components/kv_pool_runtime.py (+8/-0); python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py (+5/-1); python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py (+5/-1); python/sglang/srt/model_executor/runner_utils/pool.py (+49/-9); python/sglang/srt/speculative/base_spec_worker.py (+7/-1); python/sglang/srt/speculative/dflash_utils.py (+67/-54); python/sglang/srt/speculative/dflash_worker_v2.py (+120/-0); (+7 more)
LABELS: ready-to-merge, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ This PR sizes the CUDA graph pool from measurements taken during graph warmup, and hardens graph-pool borrowing (#35375) against failure modes we hit under speculative-decoding serving loads. ⏎  ⏎ Warmup-based sizing: ⏎  ⏎ - **Pre-carve** (opt-in): the graph pool grows segment by segment during capture, fragmenting free space. Measuring the eager warmup's reserved footprint and minting it as one contiguous span right before capture carves  …[truncated]

### L2-0b1ce3d140  (L2, 2026-09-01, sha 0b1ce3d140c4, PR #36890)
TITLE: [Feature] Unified memory: support decode context parallelism for Kimi-Linear (#36890)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.pool.mla_token_kv, L2.kernel.set_mla_kv_buffer, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/kernels/ops/kvcache/mla_buffer.py (+52/-5); python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+15/-1); python/sglang/kernels/ops/attention/dcp_kernels.py (+1/-5); python/sglang/srt/arg_groups/kv_cache_hook.py (+49/-6); python/sglang/srt/layers/dcp/layout.py (+6/-5); python/sglang/srt/layers/dcp/planner.py (+13/-3); python/sglang/srt/mem_cache/kv_index_translator.py (+44/-12); python/sglang/srt/mem_cache/memory_pool.py (+48/-23); python/sglang/srt/mem_cache/multi_ended_allocator.py (+131/-37); python/sglang/srt/mem_cache/unified_memory_pool.py (+5/-1); (+10 more)
LABELS: amd, deepseek, run-ci, jit-kernel, bypass-fastfail, memory-pool
BODY: ## Motivation ⏎  ⏎ `--enable-unified-memory` asserted `dcp_size == 1`, so the unified memory pool ⏎ and decode context parallelism could not be used together. This enables the ⏎ combination for Kimi-Linear (MLA hybrid Mamba). ⏎  ⏎ The two features each rewrite the KV location space, and they had never been ⏎ composed: ⏎  ⏎ - DCP hands out a **widened** virtual id space — `dcp_size` logical ids share ⏎   one stored row, and a rank owns the ids with `loc % dcp_size == …[truncated]

### L2-ed82bea146  (L2, 2026-09-01, sha ed82bea1464d, PR #37479)
TITLE: [Cookbook] DeepSeek-V4: add DGX Spark (2x GB10) Flash Official FP4 recipe (#37479)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx (+12/-1); docs/src/snippets/configs/deepseek-ai/deepseek-v4.jsx (+54/-0)
LABELS: documentation, deepseek
BODY: ## Motivation ⏎  ⏎ Add **NVIDIA DGX Spark** as a hardware row on the DeepSeek-V4 cookbook page, with the one recipe that is verified on that platform: **Flash Official (0731) · FP4 · Balanced · Multi-Nodes** — TP=2 across two DGX Sparks (GB10, 128 GB unified memory each) over ConnectX-7 RoCE. Every other DGX Spark combination is intentionally absent, so it greys out in the panel: the 284B checkpoint does not fit a single GB10, and the SM12x `b12x` ke …[truncated]

### L2-c16a8fc899  (L2, 2026-09-01, sha c16a8fc89995, PR #36813)
TITLE: [NPU] [bugfix] Fix NPU MLA HiCache backup accessing missing data_ptrs. (#36813)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/pool_host/mla.py (+8/-3)
LABELS: hicache, run-ci
BODY: ## Motivation ⏎  ⏎ PR https://github.com/sgl-project/sglang/pull/30393 generalized the MLA HiCache backup path to support packed target and draft KV buffers. However,  NPUMLATokenToKVPool intentionally does not create data_ptrs (it uses contiguous multi-layer K/V/index-K tensors for the NPU transfer kernel). ⏎  ⏎ <img width="1037" height="82" alt="error-2" src="https://github.com/user-attachments/assets/e33d839e-73b2-4817-bc0b-71ca714e190d" /> ⏎  ⏎ ##  …[truncated]

### L2-c66a285c94  (L2, 2026-09-01, sha c66a285c94bf, PR #37477)
TITLE: [Kernel] GLM 5.3 Flash related kernels (ported from #36507) (#37477)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.kernel.set_mla_kv_buffer
FILES: python/sglang/kernels/ops/kvcache/mla_buffer.py (+94/-5); python/sglang/kernels/jit/csrc/dsa/kpool_topk_transform.cuh (+412/-0); python/sglang/kernels/ops/attention/__init__.py (+1/-0); python/sglang/kernels/ops/attention/dsa/transform_index.py (+52/-0); python/sglang/kernels/ops/attention/fla/kda.py (+5/-0); python/sglang/kernels/ops/attention/helion/kda_prefill.py (+3/-0); python/sglang/kernels/ops/attention/utils.py (+21/-0); python/sglang/kernels/ops/layernorm/mhc.py (+184/-0); python/sglang/kernels/ops/moe/kpool_topk_transform.py (+71/-0); python/sglang/srt/layers/attention/dsa/kpool_fp8_index.py (+1701/-0); (+6 more)
LABELS: run-ci, jit-kernel, run-ci-extra
BODY: Ported from #36507. ⏎  ⏎ No behavior change on main: new files with no callers, plus additive parameters that default to existing behavior. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #33571827221](https://github.com/sgl-project/sglang/actions/runs/33571827221) ⏎ Latest PR Test (Extra): :white_check_mark: [Run #33571826901](https://github.com/sgl-project/sglang/actions/runs/33571826901) ⏎ Latest PR Test (AMD ROCm 7.2): :hourglass_flow …[truncated]

### L2-cb6dd58fbe  (L2, 2026-09-01, sha cb6dd58fbeca, PR #34693)
TITLE: [Kernel] Replace dsv3_router_gemm with the unified tiny GEMM (#34693)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption
FILES: python/sglang/kernels/jit/csrc/gemm/dsv3_router_gemm.cuh (+0/-184); python/sglang/kernels/jit/csrc/gemm/tiny_gemm.cuh (+146/-101); python/sglang/kernels/ops/gemm/__init__.py (+16/-14); python/sglang/kernels/ops/gemm/dsv3_router_gemm.py (+0/-92); python/sglang/kernels/ops/gemm/tiny_gemm.py (+124/-83); python/sglang/kernels/ops/kimi_k3/__init__.py (+7/-11); python/sglang/srt/models/deepseek_common/utils.py (+17/-0); python/sglang/srt/models/deepseek_v2.py (+13/-13); python/sglang/srt/models/dots3_common/modeling.py (+12/-12); python/sglang/srt/models/kimi_k3.py (+1/-1); (+5 more)
LABELS: deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ `dsv3_router_gemm` is a single-purpose kernel: it only accepts exactly 256 or 384 ⏎ experts with a hidden dim that is a multiple of 1024, and it is capped at 16 tokens. ⏎ The tiny GEMM added for Kimi-K3 solves the same problem — a skinny ⏎ `x[m, k] @ w[n, k].T` with a handful of rows — for a strictly larger set of shapes. ⏎ Keeping both means two kernels, two test files and two benchmarks for one job. ⏎  ⏎ This PR deprecates the router kernel  …[truncated]

### L2-26f760d5c0  (L2, 2026-09-02, sha 26f760d5c0e6, PR #32733)
TITLE: [CPU] Support FP8 KV cache (#32733)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/aot/csrc/cpu/decode.cpp (+438/-184); python/sglang/kernels/aot/csrc/cpu/extend.cpp (+36/-18); python/sglang/kernels/aot/csrc/cpu/torch_extension_cpu.cpp (+8/-3); python/sglang/kernels/aot/csrc/cpu/vec_pack.h (+103/-35); python/sglang/srt/layers/attention/intel_amx_backend.py (+46/-7); python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py (+58/-0); python/sglang/srt/mem_cache/kv_cache_configurator.py (+25/-6); python/sglang/srt/mem_cache/kv_cache_dtype.py (+3/-0); test/registered/cpu/test_decode.py (+57/-5); test/registered/cpu/test_extend.py (+24/-3); (+4 more)
LABELS: quant, sgl-kernel, blackwell, intel, cpu, run-ci, memory-pool
BODY: Due to repo permission, rebase https://github.com/sgl-project/sglang/pull/14482 and create a new PR. ⏎ Co-author: @blzheng ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #33477442155](https://github.com/sgl-project/sglang/actions/runs/33477442155) ⏎ Latest PR Test (Extra): :x: [Run #33477441976](https://github.com/sgl-project/sglang/actions/runs/33477441976) ⏎ Latest PR Test (AMD ROCm 7.2): :x: [Run #33477442014](https://github.com/sgl-project/s …[truncated]

### L2-19c30dff56  (L2, 2026-09-02, sha 19c30dff5620, PR #29927)
TITLE: [SM120] DeepSeek-V4: DeepGEMM paged-MQA indexer +FP4 MoE+ page-split  (#29927)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.kernel.flash_mla_sm120, L2.dispatch.server_args_defaults
FILES: python/sglang/kernels/ops/attention/flash_mla_sm120.py (+98/-19); python/sglang/kernels/ops/layernorm/mhc.py (+28/-3); python/sglang/srt/arg_groups/model_hook.py (+12/-5); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+12/-0); python/sglang/srt/layers/attention/dsv4/indexer.py (+101/-77); python/sglang/srt/layers/attention/dsv4/metadata.py (+24/-7); python/sglang/srt/layers/deep_gemm_wrapper/configurer.py (+9/-3); python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (+29/-2); python/sglang/srt/layers/moe/moe_runner/deep_gemm_sm120.py (+198/-0); python/sglang/srt/model_loader/utils.py (+5/-0); (+1 more)
LABELS: quant, deepseek, run-ci, jit-kernel, run-ci-extra, release-highlight
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: **Motivation:** a 4x RTX PRO 6000 Blackwell (SM120) box can hold DeepSeek-V4-Flash, but on ⏎ current main the only configuration that starts there is the slow torch fallback — this PR ⏎ makes the DeepGEMM and FlashInfer paths work on SM120, cutting decode latency up to 3.4x. ⏎  ⏎ It routes the sparse-MLA indexer to DeepGEMM's paged-MQA-logits kernel, enables batched ⏎ sparse-MLA prefill through FlashInfer, and unblocks the DeepGEMM FP4 MoE backend. ⏎  ⏎ ## Spl …[truncated]

### L2-18d5ffb42a  (L2, 2026-09-02, sha 18d5ffb42a5e, PR #37511)
TITLE: Size the unified read-table grid from bs, and fuse the allocator's tombstone scatters (#37511)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/kvcache/kv_read_table.py (+32/-18); python/sglang/kernels/ops/memory/__init__.py (+2/-0); python/sglang/kernels/ops/memory/virtual_slot.py (+95/-0); python/sglang/srt/mem_cache/multi_ended_allocator.py (+27/-23); test/registered/unit/mem_cache/test_unified_free_no_host_sync.py (+89/-3)
LABELS: jit-kernel
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: > **Stack** — first of four. Nothing depends on it upstream; #37512 → #37550 → #37560 sit on top, in that order. ⏎  ⏎ ## Motivation ⏎  ⏎ Two independent costs in the unified memory pool's hot path. ⏎  ⏎ **The read table's cuda-graph grid was sized by `max_context_len`.** `build_kv_read_table` launched `(bs, cdiv(max_pages, BLOCK))` blocks. The eager path passes the batch's live maximum, but the captured path passes `max_context_len` — and a cuda-graph captur …[truncated]

### L2-d9848b9ecd  (L2, 2026-09-02, sha d9848b9ecdf7, PR #37512)
TITLE: Build the unified read stream directly, without the page-table rectangle (#37512)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.backend.fa3_fa4_mla, L2.runner.cuda_graph_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+22/-54); python/sglang/kernels/ops/attention/metadata.py (+50/-57); python/sglang/kernels/ops/kvcache/__init__.py (+1/-0); python/sglang/kernels/ops/kvcache/kv_read_table.py (+215/-59); python/sglang/srt/layers/attention/flashattention_backend.py (+43/-51); python/sglang/srt/layers/attention/flashinfer_backend.py (+30/-74); python/sglang/srt/layers/attention/triton_backend.py (+46/-65); python/sglang/srt/layers/attention/trtllm_mha_backend.py (+10/-13); python/sglang/srt/mem_cache/kv_index_translator.py (+98/-26); test/registered/unit/mem_cache/test_kv_index_translator.py (+40/-5); (+1 more)
LABELS: blackwell, jit-kernel
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: > **Stack** — second of four. **Depends on #37511** (its base branch); review only the commits above `Test the fused tombstone by what it must do`. #37550 → #37560 sit on top. ⏎  ⏎ ## Motivation ⏎  ⏎ Three of the seven backends that read the unified pool never wanted a page table. They built a `[max_bs, max_pages]` rectangle and handed it straight to `create_flashinfer_kv_indices_triton` to repack into the indptr-addressed stream a paged wrapper plans ov …[truncated]

### L2-5a1275a519  (L2, 2026-09-02, sha 5a1275a51951, PR #37550)
TITLE: Converge the two SWA predicates, and stop conditioning the capture sink on the pool (#37550)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.backend.aiter_mla, L2.backend.flashinfer_general_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+7/-4); python/sglang/srt/layers/attention/flashattention_backend.py (+12/-16); python/sglang/srt/layers/attention/flashinfer_backend.py (+1/-4); python/sglang/srt/layers/attention/triton_backend.py (+10/-6); python/sglang/srt/layers/attention/trtllm_mha_backend.py (+14/-6); python/sglang/srt/mem_cache/kv_index_translator.py (+4/-2)
LABELS: blackwell
BODY: > **Stack** — third of four. **Depends on #37511 → #37512** (its base branch). One commit of its own; #37560 sits on top. ⏎  ⏎ ## Motivation ⏎  ⏎ Two findings from auditing the `KVIndexTranslator` series (#35245 → #34613 → #37307). Both predate that series in effect; both were given their current form by it. ⏎  ⏎ **The same question had two answers.** A backend decides "does this pool have a full→swa index mapping" through `_resolve_swa_kv_pool`, which keys  …[truncated]

### L2-5ddca6819e  (L2, 2026-09-02, sha 5ddca6819eba, PR #37560)
TITLE: Fix unified SWA: size a non-owner's v2p by the id space it must address (#37560)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/mem_cache/multi_ended_allocator.py (+13/-6); test/registered/attention/test_gemma4_unified_swa_virtual_ids.py (+100/-0); test/registered/unit/mem_cache/test_unified_swa_shared_virtual_ids.py (+119/-0)
LABELS: blackwell, run-ci, bypass-fastfail
BODY: > **Stack** — fourth of four, the top. **Depends on #37511 → #37512 → #37550** (its base branch). Two commits of its own. It is an independent bug fix that happens to sit here so the whole stack shares one CI run; it can be rebased onto `main` on request. ⏎  ⏎ ## Motivation ⏎  ⏎ `--enable-unified-memory` on a hybrid sliding-window model dies mid-serving: a device-side index assert, the scheduler gone, and the request batch returning nothing. The same mod …[truncated]

### L2-28262c20df  (L2, 2026-09-02, sha 28262c20df6f, PR #37210)
TITLE: [CI][RFC] Replace black-jupyter with ruff-format (#37210)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.flashinfer_mla, L2.backend.flashmla, L2.build.flashmla_sgl_kernel, L2.backend.trtllm_mla, L2.kernel.rocm_mla_decode_rope, L2.kernel.flash_mla_sm120, L2.backend.npu_mla, L2.backend.sparse_mla_adapters, L2.kernel.concat_mla, L2.kernel.mla_kv_pack_quantize_fp8, L2.dispatch.attention_registry, L2.runner.cuda_graph_mla
FILES: .claude/skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py (+4/-6); .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_proof_generator.py (+3/-1); .claude/skills/mechanical-refactor-verify/scripts/mechanical_refactor_reproduction_utils.py (+33/-29); .claude/skills/mechanical-refactor-verify/scripts/tests/proof_generator/test_infer_moves.py (+3/-12); .claude/skills/mechanical-refactor-verify/scripts/tests/reproduction_cli/cli_testlib.py (+2/-6); .claude/skills/mechanical-refactor-verify/scripts/tests/reproduction_utils/test_add_imports.py (+2/-13); .claude/skills/mechanical-refactor-verify/scripts/tests/reproduction_utils/test_extract_symbols_to_new_module.py (+1/-5); .claude/skills/mechanical-refactor-verify/scripts/tests/reproduction_utils/test_move_assign.py (+2/-14); .claude/skills/mechanical-refactor-verify/scripts/tests/reproduction_utils/test_move_symbol.py (+1/-7); .claude/skills/sglang-prod-incident-triage/scripts/incident_artifact_tool.py (+4/-3); (+1401 more)
LABELS: documentation, high priority, quant, amd, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel
BODY: ## Motivation ⏎  ⏎ black-jupyter is now the dominant cost of the lint job: **218s** of the ~6min job (measured on run [33360547828](https://github.com/sgl-project/sglang/actions/runs/33360547828/job/99390908592)), up from 143s in June — it re-formats every file each run, scales with the repo, and swings ±20% with runner CPU (observed 218s→260s across two runs of the same commit). Its on-disk cache can never work in CI (keyed on mtime; every fresh che …[truncated]

### L2-a11dba1a01  (L2, 2026-09-03, sha a11dba1a01ec, PR #37693)
TITLE: [Feature] Unified memory: support decode context parallelism for the trtllm_mla family (#37693)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.backend.cutedsl_mla, L2.backend.tokenspeed_mla, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/kernels/ops/attention/dcp_kernels.py (+18/-2); python/sglang/srt/layers/attention/cutedsl_mla_backend.py (+12/-17); python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+1/-1); python/sglang/srt/layers/attention/trtllm_mla_backend.py (+21/-2); python/sglang/srt/arg_groups/kv_cache_hook.py (+4/-6); python/sglang/srt/mem_cache/kv_index_translator.py (+16/-0); test/registered/dcp/test_trtllm_mla_family_dcp_metadata.py (+219/-0); test/registered/models_e2e/test_kimi_linear_unified_memory_dcp_blackwell.py (+87/-0)
LABELS: blackwell, jit-kernel, bypass-fastfail
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ #36890 added decode context parallelism to the unified memory pool, but gated `--enable-unified-memory --dcp-size > 1` to `--attention-backend flashinfer`. That excludes `cutedsl_mla`, which is the DCP-native MLA decode kernel on Blackwell (flashinfer accepts `enable_dcp=True` for no other backend) and what the B300 hosts run. So the feature was unusable there. ⏎  ⏎ The reason for the gate is that unified memory plus DCP has two read-i …[truncated]

### L2-7825e5ffca  (L2, 2026-09-03, sha 7825e5ffcad2, PR #35092)
TITLE: [AMD] Fix DSV4 unified attention sink TP slice (#35092)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/models/deepseek_v4.py (+1/-1)
LABELS: deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ DeepSeek-V4 stores `attn_sink` as a global per-head parameter. The non-unified path slices it by attention-TP rank, but the unified-KV branch passed the global tensor directly. Unified prefill truncated it to the first `H` entries and unified decode indexed from head zero, so every nonzero attention-TP rank consumed rank 0's sink values. ⏎  ⏎ This affects pure TP configurations such as TP8/DP1. Baseline/candidate token comparisons cann …[truncated]

### L2-1bda9694b7  (L2, 2026-09-03, sha 1bda9694b7bb, PR #37423)
TITLE: [AMD][DSv4] Switch output projection gemm (oproj_a) to fp8 (#37423)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/model_hook.py (+17/-1); python/sglang/srt/layers/quantization/fp8.py (+12/-0); python/sglang/srt/models/deepseek_common/amd/deepseek_v4_wo_a_fp8.py (+223/-0); python/sglang/srt/models/deepseek_v4.py (+60/-1)
LABELS: amd, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎ On DeepSeek-V4, the MLA output projection is split into two GEMMs: `wo_a` ⏎ absorbs the attention output into the low-rank o space, and `wo_b` projects back ⏎ to the hidden size. On ROCm, `wo_a` runs in **bf16** today while `wo_b` already ⏎ runs in fp8 block-scale — so the absorb GEMM reads full-precision weights on a ⏎ decode step that is bound by weight traffic. ⏎  ⏎ sglang already has an fp8 `wo_a` path behind `SGLANG_OPT_FP8_WO_A_G …[truncated]

### L2-65f7957142  (L2, 2026-09-04, sha 65f795714216, PR #38078)
TITLE: fix: gather CP-sharded tokens before TP-sharded dense MLP under prefill CP (#38078)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/communicator.py (+24/-6); test/registered/unit/layers/test_layer_communicator_fusion_gate.py (+16/-4); test/registered/unit/layers/test_layer_scatter_modes_cp_dense_mlp.py (+36/-0)
BODY: ## Summary ⏎  ⏎ - make `LayerScatterModes` mark the dense MLP boundary `MOE_FULL` when strategy prefill CP actually shards tokens (`attn_cp_size > 1`, `--enable-prefill-cp`, generic CP-v2 platform), so the existing moe_cp all-gather runs before a TP-sharded dense MLP and the scatter after it ⏎ - refuse to fuse the MLP all-reduce into the next layer for `MOE_FULL` layers (fusion skips the postprocess scatter) ⏎ - add a unit test for the dense-layer scatte …[truncated]

### L2-3c2724c48d  (L2, 2026-09-04, sha 3c2724c48ddd, PR #35770)
TITLE: [AMD] Optimize Kimi-K3 Triton MLA prefill on gfx950 (#35770)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/extend_attention.py (+729/-60); python/sglang/srt/layers/attention/triton_backend.py (+326/-14); python/sglang/srt/models/deepseek_common/attention_backend_handler.py (+35/-5); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py (+197/-41); python/sglang/srt/environ.py (+9/-0); python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py (+53/-15); test/registered/unit/layers/attention/test_triton_dense_prefill_gfx950.py (+188/-0); test/registered/unit/layers/attention/test_triton_mla_prefill_gfx950.py (+310/-0)
LABELS: amd, deepseek, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ Kimi-K3 uses two attention representations during prefill at TP8 on MI355X: ⏎  ⏎ - Fresh/zero-prefix MHA: 12 heads with QK/V dimensions 192/128. ⏎ - Cached-prefix absorbed MLA: 12 query heads, one KV head, and QK/V ⏎   dimensions 576/512. ⏎  ⏎ Current Triton prefill leaves performance on the table for both shapes: ⏎  ⏎ 1. The exact K3 kernels use one pipeline stage and natural-exponential ⏎    online softmax. ⏎ 2. Absorbed 576/512 prefill  …[truncated]

### L2-55bf3380e0  (L2, 2026-09-04, sha 55bf3380e073, PR #36805)
TITLE: Support Hy4-preview (#36805)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+38/-8); python/sglang/kernels/ops/layernorm/__init__.py (+2/-0); python/sglang/kernels/ops/layernorm/hy4_ihc.py (+443/-0); python/sglang/kernels/ops/moe/ep_moe_kernels.py (+4/-1); python/sglang/kernels/ops/moe/triton_pad_expert_counts.py (+41/-0); python/sglang/kernels/ops/moe/triton_sigmoid_gate_mul.py (+2/-0); python/sglang/srt/arg_groups/model_hook.py (+27/-13); python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py (+44/-2); python/sglang/srt/arg_groups/overrides.py (+32/-1); python/sglang/srt/arg_groups/speculative_hook.py (+1/-0); (+37 more)
LABELS: high priority, quant, deepseek, speculative-decoding, run-ci, jit-kernel, run-ci-extra, release-highlight
BODY: ## Summary ⏎  ⏎ Adds support for the `hy_v4` architecture (Hy4-preview, text-only): MLA with DSA sparse attention, iHC, gated MLA, learned attention sinks, sigmoid-gated MoE, and MTP/NextN speculative drafting, together with MXFP8 quantization support. ⏎  ⏎ ## What is included ⏎  ⏎ ## Validation ⏎  ⏎ ## Limitations ⏎  ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #33934945618](https://github.com/sgl-project/sglang/actions/runs/339349456 …[truncated]

### L2-514b45fd34  (L2, 2026-09-05, sha 514b45fd3447, PR #30315)
TITLE: [AMD][DSV4] Fix unified-KV pool sizing and SWA ring accounting (#30315)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh (+72/-47); python/sglang/kernels/ops/attention/dsv4/attn.py (+5/-0); python/sglang/kernels/ops/attention/dsv4/compress.py (+10/-3); python/sglang/srt/disaggregation/decode.py (+7/-0); python/sglang/srt/layers/attention/dsv4/compress_hip.py (+12/-4); python/sglang/srt/layers/attention/dsv4/compressor.py (+5/-3); python/sglang/srt/layers/attention/dsv4/compressor_v2.py (+3/-0); python/sglang/srt/managers/schedule_batch.py (+47/-5); python/sglang/srt/managers/schedule_policy.py (+41/-11); python/sglang/srt/managers/scheduler_components/invariant_checker.py (+22/-12); (+13 more)
LABELS: amd, deepseek, run-ci, jit-kernel, bypass-fastfail, memory-pool
DEEP_STUDY: deep-study: this PR was reverted by PR 38163 (confirmed_revert, reason=ci_or_test_failure) || deep-study: this PR was reverted by PR 38192 (reland, reason=premature_or_process) || deep-study performance PR (system_performance)
BODY: ## Consolidation note ⏎  ⏎ This PR now also contains the SWA per-request ring runtime accounting and tests previously reviewed in #31040. The two pieces must land together: with only the original pool-sizing commits, the DSV4 Flash and Pro MI35x jobs hit GPU memory access faults during concurrent prefill; the runtime ring reconciliation at the combined tip prevents the allocator from issuing locations beyond the fixed per-request SWA ring. ⏎  ⏎ ## Mo …[truncated]

### L2-09daea94ac  (L2, 2026-09-05, sha 09daea94ac55, PR #38152)
TITLE: Support NoPE layers in the tokenspeed_mla FP8 prefill hook (#38152)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.tokenspeed_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/tokenspeed_mla_backend.py (+16/-4); test/registered/attention/unittests/mla/test_tokenspeed_mla.py (+80/-0)
BODY: ## Motivation ⏎  ⏎ Three `extra-b-test-4-gpu-b200` jobs fail on main ([run 33962519343](https://github.com/sgl-project/sglang/actions/runs/33962519343)): `test_kimi_linear_dcp4.py`, `test_kimi_linear_dcp_dspark4.py`, `test_unified_radix_cache_kl_dcp.py`, all Kimi Linear on `tokenspeed_mla` + `fp8_e4m3` under DCP. Every TP rank crashes during prefill and the server is SIGKILLed: ⏎  ⏎ ``` ⏎ File ".../layers/attention/tokenspeed_mla_backend.py", line 265 …[truncated]

### L2-6cee9285a3  (L2, 2026-09-05, sha 6cee9285a3dd, PR #37591)
TITLE: [ROCm] Make DSA indexer top-k exact with cooperative selection (#37591)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: 3rdparty/amd/wheel/sgl-kernel/rocm_hipify.py (+1/-1); python/sglang/kernels/aot/csrc/elementwise/topk.hip (+325/-0); python/sglang/kernels/aot/include/hip/dsa_topk_coop.cuh (+1182/-0); python/sglang/kernels/aot/setup_rocm.py (+2/-1); python/sglang/kernels/aot/tests/test_topk.py (+162/-0)
LABELS: amd, sgl-kernel, run-ci
BODY: ## Summary ⏎  ⏎ - add a ROCm-specific cooperative top-k implementation for the three DSA indexer entry points ⏎ - use an fp32 coarse histogram, exact radix tie refinement, and an overflow rescan ⏎ - split long rows across blocks when a low batch size would otherwise leave the GPU underutilized ⏎ - preserve the raw, paged, and ragged interfaces and guard launches to the score device ⏎ - keep the CUDA implementation unchanged; ROCm tracks the native `topk.hip` …[truncated]

### L2-97c6978369  (L2, 2026-09-06, sha 97c6978369ac, PR #36507)
TITLE: GLM-5.3-Flash support (#36507)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+3/-0); python/sglang/kernels/ops/attention/dsa/quant_k_cache.py (+69/-5); python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py (+33/-21); python/sglang/kernels/ops/attention/dsa_metadata.py (+10/-6); python/sglang/kernels/ops/attention/fla/kda.py (+4/-5); python/sglang/srt/arg_groups/attention_hook.py (+1/-0); python/sglang/srt/arg_groups/cuda_graph_hook.py (+8/-1); python/sglang/srt/arg_groups/model_hook.py (+1/-0); python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py (+2/-1); python/sglang/srt/arg_groups/overrides.py (+3/-0); (+93 more)
LABELS: quant, amd, Multi-modal, deepseek, hicache, npu, run-ci, apple-silicon, jit-kernel, bypass-fastfail
BODY: Support GLM-5.3-Flash. ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :no_entry_sign: [Run #34024595258](https://github.com/sgl-project/sglang/actions/runs/34024595258) ⏎ Latest PR Test (Extra): :no_entry_sign: [Run #34024595202](https://github.com/sgl-project/sglang/actions/runs/34024595202) ⏎ Latest PR Test (AMD ROCm 7.2): :no_entry_sign: [Run #34024595305](https://github.com/sgl-project/sglang/actions/runs/34024595305)

### L2-15aa2fb843  (L2, 2026-09-06, sha 15aa2fb8433d, PR #37124)
TITLE: [ROCm] Take the fused DSA metadata kernels and drop redundant work from the absorb path (#37124)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/layers/attention/dsa/dsa_backend_mtp_precompute.py (+2/-2); python/sglang/srt/layers/attention/dsa_backend.py (+9/-5); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py (+42/-13)
LABELS: amd, run-ci
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: Three ROCm-only changes on the DSA decode path. All numbers are GLM-5.2-MXFP4 / ⏎ MI355X / TP4, agentic replay at concurrency 4, paired per request against the ⏎ same branch without the change. ⏎  ⏎ ## Fused DSA metadata kernels ⏎  ⏎ The five sites that build DSA's per-step metadata read `is_cuda() and not ⏎ _is_hip`, so HIP falls back to a PyTorch sequence of seven or eight ops. The ⏎ kernels they skip are three `@triton.jit` functions in ⏎ `kernels/ops/ …[truncated]

### L2-6252993afe  (L2, 2026-09-06, sha 6252993afe07, PR #38238)
TITLE: chore: update CI test est_time values (#38238)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: test/registered/4-gpu-models/test_deepseek_v3_cutedsl_4gpu.py (+1/-1); test/registered/8-gpu-models/test_deepseek_v32_indexcache.py (+1/-1); test/registered/attention/test_chunk_gated_delta_rule.py (+1/-1); test/registered/attention/test_create_kvindices.py (+1/-1); test/registered/attention/test_deterministic.py (+1/-1); test/registered/attention/test_flash_attention_4.py (+1/-1); test/registered/attention/test_gdn_fused_split_head_ratios.py (+1/-1); test/registered/attention/test_gdn_noncontiguous_stride.py (+1/-1); test/registered/attention/test_gdn_prefill_layout.py (+1/-1); test/registered/attention/test_gemma4_swa_triton_oob_regression.py (+1/-1); (+1006 more)
LABELS: quant, amd, lora, Multi-modal, deepseek, speculative-decoding, hicache, blackwell, npu, apple-silicon
BODY: ## Summary ⏎  ⏎ Refreshes `est_time` literals from [`sgl-project/sglang-ci-stats`](https://github.com/sgl-project/sglang-ci-stats)'s `model.json` (per-(suite, file) p90 over recent successful CI runs on `main`). ⏎  ⏎ This keeps the LPT load-balancing algorithm accurate for partitioning tests across parallel CI jobs, and serves as the static fallback when `compute_partitions` cannot fetch the live model at PR time. ⏎  ⏎ ### Significant est_time changes (203 o …[truncated]

### L2-1d5d85260c  (L2, 2026-09-06, sha 1d5d85260ccb, PR #37601)
TITLE: [AMD] support qlen>1 for aiter gluon path for Kimi K3 (#37601)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla, L2.kernel.aiter_mla_gluon
FILES: python/sglang/srt/layers/attention/aiter_mla_gluon.py (+70/-197); python/sglang/srt/distributed/parallel_state.py (+10/-4); python/sglang/srt/layers/attention/aiter_backend.py (+30/-18); test/registered/attention/test_aiter_gluon_h12_fp8.py (+321/-0)
LABELS: amd, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ This patch is to add qlen>1 support for aiter gluon path.  https://github.com/sgl-project/sglang/pull/34647 only support qlen=1, for other cases, it will fall back to mla asm ps. If you specify the following configs,  ⏎  ⏎ ```bash ⏎ SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1 \ ⏎ SGLANG_USE_AITER=1 \ ⏎ SGLANG_AITER_K3_OPT=1 \ ⏎ AITER_FLYDSL_FORCE=1 \ ⏎ AITER_SITUV2_A8W4=1 \ ⏎ SGLANG_AITER_MLA_GLUON=1 \ ⏎ python3 -m sglang.launch_server - …[truncated]

### L2-ed82def55f  (L2, 2026-09-06, sha ed82def55fac, PR #38047)
TITLE: [Config] Round 6.2: the field declarations move to their namespaces, and the record is assembled from them (#38047)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/arg_utils.py (+49/-27); python/sglang/srt/arg_groups/choices.py (+289/-0); python/sglang/srt/arg_groups/field_order.py (+511/-0); python/sglang/srt/arg_groups/fields/__init__.py (+70/-0); python/sglang/srt/arg_groups/fields/device.py (+83/-0); python/sglang/srt/arg_groups/fields/disagg.py (+167/-0); python/sglang/srt/arg_groups/fields/exec_.py (+863/-0); python/sglang/srt/arg_groups/fields/lora.py (+119/-0); python/sglang/srt/arg_groups/fields/memory.py (+242/-0); python/sglang/srt/arg_groups/fields/mm.py (+154/-0); (+8 more)
LABELS: lora
BODY: Second of four; stacked on #38046. Mechanical relocation plus one design change ⏎ that the relocation makes possible. **Review by checking the identity proofs at ⏎ the bottom** -- nothing here is meant to change behaviour. ⏎  ⏎ ## The declarations move ⏎  ⏎ `ServerArgs` carried all 487 declarations in one 4,462-line file, each tagged ⏎ with an `NS("...")` marker naming the namespace it belongs to -- structure ⏎ supplied by annotation, in a file a namespace away  …[truncated]

### L2-b6c31b155c  (L2, 2026-09-06, sha b6c31b155c2b, PR #36228)
TITLE: [CP V1 Deprecation 3/5] Remove generic prefill CP v1 runtime (#36228)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla, L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+12/-30); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+1/-1); python/sglang/srt/arg_groups/kv_cache_hook.py (+3/-3); python/sglang/srt/arg_groups/parallel_hook.py (+7/-4); python/sglang/srt/arg_groups/pipeline.py (+1/-2); python/sglang/srt/layers/attention/dsa/dsa_indexer.py (+16/-224); python/sglang/srt/layers/attention/dsa/utils.py (+2/-9); python/sglang/srt/layers/attention/dsa_backend.py (+31/-11); python/sglang/srt/layers/attention/flashattention_backend.py (+32/-73); python/sglang/srt/layers/attention/index_topk_share.py (+13/-2); (+24 more)
LABELS: deepseek, run-ci, mthreads, bypass-fastfail, release-highlight
BODY: ## Summary ⏎  ⏎ - Remove generic prefill CP v1 branches and the DSA v1 in-seq-split indexer/backend path; remove the obsolete single-request scheduling restriction. ⏎ - Keep generic input sharding and token-ID layout handoff at the runner/strategy boundary. ⏎ - Remove unnecessary distributed-layout, projection-cache-write, and communicator-factory wrappers. DeepSeek V4 attention and decoder layers retain their baseline implementation, including HIP KV-sc …[truncated]

### L2-644841c50c  (L2, 2026-09-07, sha 644841c50cb3, PR #37691)
TITLE: [AMD] Support aiter fa mha chunked kv for Kimi-K3 (#37691)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+56/-0); python/sglang/srt/models/deepseek_common/attention_backend_handler.py (+4/-1); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py (+3/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py (+1/-1); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha_rocm.py (+12/-0); python/sglang/srt/models/deepseek_v2.py (+6/-0)
LABELS: amd, deepseek, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ This patch is to fix hip out of memory crash to unblock Semi Analysis's AgentX long sequence benchmark. Kimi-K3 can handle up to 1M context. We will hit OOM during benchmark. ⏎  ⏎ ```bash ⏎ torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.36 GiB. ⏎ GPU 3 has a total capacity of 287.98 GiB of which 10.00 MiB is free. ⏎ Of the allocated memory 268.27 GiB is allocated by PyTorch... ⏎ ``` ⏎  ⏎ How to reprodce this : ⏎ Launc …[truncated]

### L2-b5766336d4  (L2, 2026-09-07, sha b5766336d4fa, PR #37926)
TITLE: [Perf] Unified memory: close the DCP decode gap on Blackwell (#37926)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.pool.mla_token_kv, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+7/-13); python/sglang/kernels/ops/mamba/mamba_state_indices_triton.py (+15/-3); python/sglang/kernels/ops/memory/__init__.py (+1/-0); python/sglang/kernels/ops/memory/virtual_slot.py (+133/-0); python/sglang/srt/arg_groups/kv_cache_hook.py (+6/-0); python/sglang/srt/batch_overlap/two_batch_overlap.py (+5/-0); python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py (+5/-8); python/sglang/srt/layers/attention/triton_backend.py (+8/-14); python/sglang/srt/mem_cache/allocator/unified_hybrid_swa.py (+4/-1); python/sglang/srt/mem_cache/allocator/unified_mamba.py (+4/-1); (+11 more)
LABELS: blackwell, run-ci, jit-kernel, bypass-fastfail, memory-pool
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎ Unified memory (`--enable-unified-memory`) under decode context parallelism was ⏎ measurably slower than the static pool on Blackwell. The Hopper fix (#37511) did ⏎ not carry over: on a B300 with Kimi-Linear at TP2/DCP2 and `cutedsl_mla`, the ⏎ unified pool ran **1.96% behind static** (worst config -4.06%). ⏎  ⏎ Profiling showed the gap is entirely CPU-side. GPU kernel time was slightly ⏎ *lower* for the unified pool, while it issued ~456 more …[truncated]

### L2-c4e52a1051  (L2, 2026-09-07, sha c4e52a105180, PR #24959)
TITLE: XPU: Enable GLM5.1 (GlmMoeDsaForCausalLM) DSA Attention (#24959)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters, L2.pool.mla_token_kv, L2.dispatch.server_args_defaults
FILES: python/sglang/kernels/ops/attention/dsa/index_buf_accessor.py (+7/-1); python/sglang/srt/arg_groups/memory_hook.py (+6/-0); python/sglang/srt/arg_groups/model_hook.py (+15/-0); python/sglang/srt/arg_groups/model_overrides/deepseek_v2.py (+3/-0); python/sglang/srt/arg_groups/overrides.py (+27/-2); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+4/-6); python/sglang/srt/layers/attention/dsa/dsa_indexer.py (+64/-9); python/sglang/srt/layers/attention/dsa_backend.py (+149/-3); python/sglang/srt/layers/rotary_embedding/base.py (+15/-2); python/sglang/srt/mem_cache/memory_pool.py (+6/-0); (+3 more)
LABELS: deepseek, intel, xpu, run-ci, jit-kernel, memory-pool
BODY: ## Motivation ⏎  ⏎  ⏎ GLM5.1 uses Dynamic Sparse Attention (DSA/NSA) with an FP8 indexer that scores KV pages before sparse attention. This PR enables the path on XPU. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ **1. server_args.py** ⏎ When GLM5.1 (or any DSA model) runs on XPU, automatically set: ⏎ - `decode_attention_backend = "dsa"` — puts `DeepseekSparseAttnBackend` as the decode backend inside `HybridAttnBackend`, so it can manage the DSA index K-cache and FP8 logi …[truncated]

### L2-e9e9e37ddc  (L2, 2026-09-07, sha e9e9e37ddcf2, PR #32759)
TITLE: [AMD] Restore SWA reprefill-tail on UnifiedRadixCache when HiCache is off (#32759)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/mem_cache/unified_radix_cache.py (+19/-11)
LABELS: amd, run-ci, unified-radix-cache
BODY: ## Motivation ⏎  ⏎  ⏎ #30339 fixed a stale per-request SWA ring read on radix prefix reuse for DeepSeek-V4 with the unified_kv backend. The mechanism: hold back a trailing sliding window from the radix match so it is re-prefilled into the requesting slot's own ring instead of being inherited from whichever request previously occupied that req_pool_idx. ⏎ It shipped as a `SWARadixCache.swa_reprefill_tail_tokens()` override, gated only on the backend: …[truncated]

### L2-85d39401c8  (L2, 2026-09-07, sha 85d39401c85d, PR #38293)
TITLE: [CP V1 Deprecation 3.5/5]  Deprecate HIP/NPU/MUSA prefill CP and remove legacy implementation (#38293)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.fa3_fa4_mla, L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+0/-7); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py (+0/-30); python/sglang/srt/arg_groups/attention_hook.py (+1/-1); python/sglang/srt/arg_groups/cuda_graph_hook.py (+0/-4); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+0/-18); python/sglang/srt/arg_groups/field_order.py (+0/-4); python/sglang/srt/arg_groups/fields/parallel.py (+0/-4); python/sglang/srt/arg_groups/model_hook.py (+1/-3); python/sglang/srt/arg_groups/parallel_hook.py (+13/-104); python/sglang/srt/arg_groups/pipeline.py (+11/-18); (+37 more)
LABELS: amd, deepseek, npu, run-ci, mthreads, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Deprecate the legacy prefill context-parallel implementations on HIP/ROCm, Ascend NPU, and MUSA ahead of the upcoming CP refactor. CUDA's strategy-based prefill CP, decode context parallelism (DCP), and ordinary non-CP platform inference remain in scope for regression protection, not removal. ⏎  ⏎ ## Modifications ⏎  ⏎ - Reject `--enable-prefill-cp` on HIP/NPU/MUSA in the parallel hooks with: `Prefill CP on HIP/NPU/MUSA is deprecated; CP s …[truncated]

### L2-325ab245a1  (L2, 2026-09-08, sha 325ab245a182, PR #37767)
TITLE: [DCP] Allow fi_a2a on single-node systems Blackwell without MNNVL fabric ( ex B200 B300) (#37767)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/model_overrides/kimi_k3.py (+13/-2); python/sglang/srt/layers/dcp/comm.py (+30/-8)
BODY: ## Motivation ⏎  ⏎ `--dcp-comm-backend fi_a2a` is rejected on every Blackwell system without an IMEX / NVL72 fabric — any single-node 8×B200 or 8×B300 box: ⏎  ⏎ ``` ⏎ RuntimeError: --dcp-comm-backend fi_a2a requires MNNVL fabric memory (e.g. GB200 NVL72); ⏎ is_mnnvl_fabric_supported() returned False. ⏎ ``` ⏎  ⏎ The gate was correct when written: FlashInfer's non-fabric path exchanged CUDA VMM file descriptors with `pidfd_getfd(2)`, which needs `PTRACE_MODE_ATTACH` …[truncated]

### L2-cf35384fe4  (L2, 2026-09-08, sha cf35384fe459, PR #35866)
TITLE: [Intel GPU] Add MLA support to Intel XPU Attention backend for Prefill (#35866)
SOURCES: path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/xpu_backend.py (+51/-64); python/sglang/srt/arg_groups/attention_hook.py (+1/-7); python/sglang/srt/arg_groups/overrides.py (+6/-7); test/registered/xpu/test_intel_xpu_backend.py (+2/-2)
LABELS: intel, xpu, run-ci
BODY: Add Support for MLA models to use `intel_xpu` attention backend for Prefill, decode support already available. ⏎ pure prefill uses `flash_attn_varlen_func` ⏎ incremental prefill uses `flash_mla_prefill` ⏎  ⏎  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.co …[truncated]

### L2-52fecfdf09  (L2, 2026-09-08, sha 52fecfdf0908, PR #37500)
TITLE: support qwen 3.8 flash next (#37500)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+18/-0); benchmark/kernels/fused_moe_triton/common_utils.py (+7/-0); python/sglang/kernels/jit/csrc/attention/qsa_indexer.cuh (+486/-0); python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh (+291/-0); python/sglang/kernels/jit/csrc/elementwise/grouped_gemma_rmsnorm.cuh (+178/-0); python/sglang/kernels/jit/csrc/elementwise/hc_combine.cuh (+381/-0); python/sglang/kernels/kda_kernels/README.md (+1/-0); python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/README.md (+51/-0); python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/__init__.py (+83/-0); python/sglang/kernels/kda_kernels/qwen38_qsa_sm121/kernel.py (+269/-0); (+81 more)
LABELS: documentation, high priority, quant, speculative-decoding, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight, unified-radix-cache
BODY: initial pr: https://github.com/sgl-project/sglang/pull/36497 ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :white_check_mark: [Run #34195586068](https://github.com/sgl-project/sglang/actions/runs/34195586068) ⏎ Latest PR Test (Extra): :white_check_mark: [Run #34273666129](https://github.com/sgl-project/sglang/actions/runs/34273666129) ⏎ Latest PR Test (AMD ROCm 7.2): :x: [Run #34195586146](https://github.com/sgl-project/sglang/actions/runs/34195586146)

### L2-ed183d45ac  (L2, 2026-09-08, sha ed183d45acfb, PR #36229)
TITLE: [CP V1 Deprecation 4/5] Canonicalize prefill CP API names (#36229)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla, L2.backend.sparse_mla_adapters
FILES: python/sglang/kernels/ops/attention/__init__.py (+1/-1); python/sglang/kernels/ops/attention/dsa/cp_split.py (+2/-2); python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py (+1/-1); python/sglang/kernels/ops/layernorm/mhc.py (+2/-2); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+8/-8); python/sglang/srt/layers/attention/dsa/dsa_indexer.py (+3/-3); python/sglang/srt/layers/attention/dsa/utils.py (+11/-9); python/sglang/srt/layers/attention/dsa_backend.py (+5/-5); python/sglang/srt/layers/attention/flashattention_backend.py (+2/-2); python/sglang/srt/layers/attention/index_topk_share.py (+2/-2); (+23 more)
LABELS: deepseek, blackwell, run-ci, jit-kernel
BODY: ## Summary ⏎  ⏎ - rename `is_cp_v2_active` to `is_cp_active` ⏎ - replace the remaining generic CP `v2` fields/methods with versionless names ⏎ - rename CP round-robin implementation APIs and kernels to `interleave` ⏎ - rename MLA CP helpers to `is_mla_cp_enabled` / `is_mla_cp_active` ⏎  ⏎ ## Stack ⏎  ⏎ 1. #36222 — switch CP tests from v1 to v2 ⏎ 2. #36223 — remove CP v1 server-argument compatibility ⏎ 3. #36228 — remove generic CP v1 runtime ⏎ 4. **this PR** — canonical …[truncated]

### L2-db272201a2  (L2, 2026-09-08, sha db272201a2db, PR #38375)
TITLE: [Config] Retire get_global_server_args, and clear the deprecated flags that have a replacement (#38375)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: .claude/skills/cookbook-add-model/references/authoring-reference.md (+1/-1); .claude/skills/cookbook-migrate-model/SKILL.md (+2/-2); .claude/skills/cookbook-migrate-model/references/dimension-mapping.md (+1/-1); .claude/skills/sglang-runtime-context/SKILL.md (+7/-5); benchmark/bench_linear_attention/bench_int8_checkpoint_reuse.py (+2/-2); docs/cookbook/autoregressive/DeepSeek/DeepSeek-V3.mdx (+1/-1); docs/cookbook/autoregressive/Meituan/LongCat-2.0.mdx (+1/-1); docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx (+1/-1); docs/docs/advanced_features/cuda_graph_for_multi_modal_encoder.mdx (+7/-4); docs/docs/advanced_features/pd_disaggregation.mdx (+3/-4); (+203 more)
LABELS: documentation, quant, lora, Multi-modal, deepseek, speculative-decoding, hicache, sgl-kernel, npu, run-ci
BODY: Two cleanups on the config tier, plus the review rounds they went through. ⏎ Separate commits, in order. ⏎  ⏎ ## `get_global_server_args` is retired ⏎  ⏎ It was the last spelling of "reach for the whole record and pick a field off ⏎ it". Nothing in `srt` called it any more -- the two remaining references were ⏎ bare imports, one a deliberate re-export -- so this is where keeping it costs ⏎ more than removing it. The name still worked, still returned a `ServerArg …[truncated]

### L2-8a0863c728  (L2, 2026-09-08, sha 8a0863c72840, PR #36800)
TITLE: [HiCache] Add MLA host-dedup primitives (#36800)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/mla_host_dedup.py (+305/-0); python/sglang/srt/environ.py (+2/-0); python/sglang/srt/mem_cache/pool_host/dsa.py (+25/-2); python/sglang/srt/mem_cache/pool_host/mla.py (+98/-1); test/registered/unit/mem_cache/test_mla_host_dedup_primitives.py (+258/-0)
LABELS: hicache, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ MLA/DSA target KV is replicated across attention-TP ranks. Deduplicating the host copy needs two reusable building blocks before the feature can be wired into Unified HiCache: an allocator-only peer host pool and a dedicated layerwise NCCL broadcaster. ⏎  ⏎ ## Changes ⏎  ⏎ - Add allocator-only MLA and DSA host-pool modes. The default remains `is_dummy=False`, so existing HiCache behavior is unchanged. ⏎ - Add the dedicated attention-T …[truncated]

### L2-1ad3eb09a9  (L2, 2026-09-09, sha 1ad3eb09a91f, PR #38590)
TITLE: [Attention] Size FlashInfer MLA indptr buffers to the padded max batch (#38590)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.backend.flashinfer_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/flashinfer_mla_backend.py (+12/-4)
BODY: ## Motivation ⏎  ⏎ Under MLP sync (DP attention, DeepEP, or `--moe-a2a-backend megamoe`) the eager and cuda-graph runners pad the request count to the attn-tp alignment (`get_eager_max_batch_size` / `get_cuda_graph_max_batch_size`), so the warm-up dummy batch and the largest captured decode batch can be wider than `req_to_token_pool.size`. ⏎  ⏎ `FlashInferMLAAttnBackend` sized `kv_indptr`, `qo_indptr` and `q_indptr_decode` to the raw pool size (and the c …[truncated]

### L2-295132c4a5  (L2, 2026-09-09, sha 295132c4a5d7, PR #33089)
TITLE: [NPU] Add sparsity-driven KV offload for DeepSeek DSA on Ascend (#33089)
SOURCES: symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+1/-0); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+69/-0); python/sglang/srt/hardware_backend/npu/memory_pool_npu.py (+57/-30); python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/attention.py (+318/-0); python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/config.py (+96/-0); python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/host_callback.py (+80/-0); python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/manager.py (+980/-0); python/sglang/srt/model_executor/pool_configurator.py (+15/-0); test/registered/unit/npu/test_sparsity_driven_kv_offload_config.py (+121/-0)
LABELS: npu, run-ci, bypass-fastfail
BODY: # [NPU] Add sparsity-driven KV offload for DeepSeek DSA on Ascend ⏎  ⏎ USTC: Yinhe Chen, Chengru Yang, Chengjie Tang(SXU, intern at USTC-IAI), Youhui Bai, Cheng Li ⏎  ⏎ Ascend NPU Network Lab\*\*: Ning Zhao, Heyuan Li, Biao Wang, Pengcheng Wang, Chen Sun ⏎  ⏎ ## Summary ⏎  ⏎ This PR adds an opt-in sparsity-driven KV offload path for DeepSeek DSA models using the Ascend MLA attention backend. ⏎  ⏎ When enabled, the runtime: ⏎  ⏎ - keeps the DSA index KV cache …[truncated]

### L2-880d6fa64d  (L2, 2026-09-09, sha 880d6fa64d18, PR #30805)
TITLE: [DSv4] Integrate TRT-LLM DSv4 Attention for SM100/103 (#30805)
SOURCES: path_core, symbol_pickaxe, release_notes, corpus:production-kernel-provenance, body_keyword
ARTIFACT_HINTS: L2.dispatch.attention_registry, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/attention_registry.py (+7/-4); python/sglang/srt/arg_groups/deepseek_v4_hook.py (+41/-0); python/sglang/srt/arg_groups/fields/exec_.py (+13/-0); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+112/-8); python/sglang/srt/layers/attention/deepseek_v4_trtllm_backend.py (+532/-0); python/sglang/srt/layers/attention/dsv4/compressor_trtllm.py (+120/-0); python/sglang/srt/layers/attention/dsv4/compressor_v2.py (+16/-0); python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py (+94/-1); python/sglang/srt/speculative/draft_utils.py (+7/-5); test/registered/attention/unittests/dsv4/test_deepseek_v4.py (+28/-0); (+3 more)
LABELS: high priority, deepseek, blackwell, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, release-highlight, memory-pool
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Integrates TRT-LLM attention kernel for DSv4 style attention (CSA, HCA). ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ ``` ⏎ SGLANG_DSV4_ATTN_DECODE_BACKEND=flashmla/trtllm_gen \ ⏎ python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V4-Pro \ ⏎   --trust-remote-code --tp 8 --moe-runner-backend flashinfer_mxfp4 \ ⏎   --chunked-prefill-size 4096 --disable-flashinfer-autotune \ ⏎   --mem-fraction-static 0.88 --max-running-r …[truncated]

### L2-1b77f498a0  (L2, 2026-09-10, sha 1b77f498a0f7, PR #31470)
TITLE: [NVIDIA] Support flashinfer Mega Moe (#31470)
SOURCES: release_notes
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.dispatch.server_args_defaults
FILES: docs/docs/advanced_features/server_arguments.mdx (+7/-1); docs/docs/references/environment_variables.mdx (+15/-0); python/pyproject.toml (+1/-0); python/sglang/srt/arg_groups/choices.py (+2/-0); python/sglang/srt/arg_groups/fields/exec_.py (+6/-0); python/sglang/srt/arg_groups/moe_hook.py (+174/-14); python/sglang/srt/arg_groups/overrides.py (+10/-3); python/sglang/srt/environ.py (+15/-0); python/sglang/srt/layers/moe/flashinfer_megamoe.py (+672/-0); python/sglang/srt/layers/moe/fused_moe_triton/layer.py (+3/-0); (+21 more)
LABELS: documentation, high priority, quant, dependencies, deepseek, run-ci, bypass-fastfail, release-highlight
BODY: Fork from https://github.com/djns99/sglang/tree/djns99/mega_moe_flashinfer ⏎ @djns99 is the main author of this PR.  ⏎  ⏎ ## Motivation ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get …[truncated]

### L2-8a6ab89bf0  (L2, 2026-09-10, sha 8a6ab89bf0b9, PR #30575)
TITLE: [AMD] Enable Fast Triton Sparse MLA backend (#30575)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.sparse_mla_adapters, L2.dispatch.server_args_defaults
FILES: python/sglang/kernels/ops/attention/dsa/triton_sparse_mla.py (+1031/-77); python/sglang/kernels/ops/attention/dsa/triton_sparse_mla_decode.py (+742/-0); python/sglang/srt/layers/attention/dsa_backend.py (+45/-34); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py (+3/-3); python/sglang/srt/arg_groups/fields/exec_.py (+2/-0); python/sglang/srt/arg_groups/hisparse_hook.py (+1/-1); python/sglang/srt/arg_groups/overrides.py (+42/-15); python/sglang/srt/mem_cache/kv_cache_configurator.py (+2/-2); test/registered/unit/test_model_overrides.py (+3/-3)
LABELS: amd, run-ci, jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: # Fast Triton Sparse MLA Kernels for DSA (Prefill + Decode) ⏎  ⏎ ## Summary ⏎  ⏎ This adds `triton` as an explicit DSA prefill/decode backend: ⏎  ⏎ ```bash ⏎ --dsa-prefill-backend triton ⏎ --dsa-decode-backend triton ⏎ ``` ⏎  ⏎ The new backend provides pure Triton sparse MLA kernels for the fp8 DSA path on ROCm, validated on MI355X (gfx950) and MI300X (gfx942). It replaces the previous Triton prefill env-var gate, SGLANG_DSA_TRITON_PREFILL, with a faster ke …[truncated]

### L2-12771786f2  (L2, 2026-09-10, sha 12771786f231, PR #38575)
TITLE: [AMD] Restore AITER verify runtime sizing reverted by #34647 (#38575)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.aiter_mla
FILES: python/sglang/srt/layers/attention/aiter_backend.py (+4/-2)
LABELS: run-ci, run-ci-extra
DEEP_STUDY: deep-study revert record: reland of PR(s) 31221 reason=correctness_or_accuracy
BODY: Restores the two #31221 hunks that #34647 accidentally reverted: ⏎  ⏎ - non-MLA eager verify: size `draft_num` from the runtime input width (`input_ids.shape[0] // bs`) instead of the fixed `spec_info.draft_token_num` ⏎ - unified verify launch: size `seqused_k` from the per-batch metadata width (`forward_metadata.max_q_len`) instead of the fixed `num_draft_tokens` ⏎  ⏎ #31221 (merged Aug 1) derived AITER SBD verify widths from the runtime input. #34647 (me …[truncated]

### L2-3ff226ba8f  (L2, 2026-09-10, sha 3ff226ba8f86, PR #38250)
TITLE: [NPU]Support GLM5.2 and FP8 DSA&Indexer kvcache for 950 (#38250)
SOURCES: path_core
ARTIFACT_HINTS: L2.model.deepseek_v2_mla, L2.optimization.weight_absorption, L2.backend.npu_mla, L2.backend.sparse_mla_adapters
FILES: python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py (+145/-20); python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py (+61/-26); python/sglang/srt/hardware_backend/npu/memory_pool_npu.py (+167/-48); python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py (+56/-24); python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py (+23/-12); python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py (+4/-1); python/sglang/srt/layers/attention/dsa/dsa_npu_indexer.py (+68/-3); python/sglang/srt/mem_cache/kv_cache_configurator.py (+29/-1); python/sglang/srt/model_executor/pool_configurator.py (+12/-0); python/sglang/srt/models/deepseek_v2.py (+1/-1)
LABELS: deepseek, npu, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Enable GLM-5.2 inference on 950PR/DT NPU with FP8 KV cache and MLAProlog, while reducing redundant indexer cache allocation for GLM5.2. ⏎  ⏎ ## Modification ⏎  ⏎ - Add quant sparse attention and quantized DSA indexing for FP8 KV cache. ⏎ - Support packed FP8 DSA&indexer KV cache through the existing --kv-cache-dtype option ⏎ - Add mlaprolog support for GLM5.2 on 950NPU and adapt MLAProlog weight/scale preparation for BF16 and MXFP8 pro …[truncated]

### L2-c0b790cf7f  (L2, 2026-09-10, sha c0b790cf7fe6, PR #32114)
TITLE: Delete cutlass_mla, non-Marlin GPTQ, AWQ AOT kernel, and Dual Chunk Flash Attention (#32114)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L2.kernel.cutlass_mla, L2.backend.cutlass_mla, L2.dispatch.attention_registry, L2.dispatch.server_args_defaults, L2.runner.cuda_graph_mla
FILES: python/sglang/kernels/aot/CMakeLists.txt (+0/-10); python/sglang/kernels/aot/csrc/attention/cutlass_mla_kernel.cu (+0/-274); python/sglang/kernels/aot/csrc/attention/cutlass_sm100_mla/device/sm100_mla.hpp (+0/-358); python/sglang/kernels/aot/csrc/attention/cutlass_sm100_mla/kernel/sm100_fmha_mla_reduction.hpp (+0/-198); python/sglang/kernels/aot/csrc/attention/cutlass_sm100_mla/kernel/sm100_fmha_mla_tma_warpspecialized.hpp (+0/-2018); python/sglang/kernels/aot/csrc/attention/cutlass_sm100_mla/kernel/sm100_mla_tile_scheduler.hpp (+0/-160); .github/labeler.yml (+0/-1); docs/cookbook/autoregressive/DeepSeek/DeepSeek-V3.mdx (+1/-1); docs/docs/advanced_features/attention_backend.mdx (+0/-16); docs/docs/advanced_features/quantization.mdx (+4/-4); (+61 more)
LABELS: documentation, quant, speculative-decoding, sgl-kernel, blackwell, run-ci, mthreads, bypass-fastfail
ISSUES: #32111 Deprecate CUTLASS MLA attention backend | #32112 Deprecate legacy (non-Marlin) GPTQ kernel and Dual Chunk Flash Attention backend
BODY: ## Summary ⏎ - Delete the `cutlass_mla` attention backend (kernel + Python integration): SM10.0-only decode kernel, already disabled on GB300 (SM10.3), falls through to `FlashInferMLABackend` for everything except plain decode, no CI coverage. ⏎ - Delete the non-Marlin GPTQ CUDA kernel (`gptq_gemm`/`gptq_shuffle`) and the `"gptq"` GPU quantization choice: superseded by `gptq_marlin` (fully JIT), which `GPTQMarlinConfig` already auto-upgrades compatib …[truncated]

### L2-d076eec427  (L2, 2026-09-10, sha d076eec42788, PR #36655)
TITLE: [SM120] Use exact query-head widths for DeepSeek-V4 sparse MLA decode (#36655)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.kernel.flash_mla_sm120
FILES: python/sglang/kernels/ops/attention/flash_mla_sm120.py (+30/-1); python/sglang/srt/models/deepseek_v4.py (+48/-23); test/registered/kernels/ops/attention/test_flash_mla_backends.py (+82/-0)
LABELS: deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Motivation ⏎  ⏎ DeepSeek-V4 on SM120 pads TP4 decode queries from 16 real heads to 64 even ⏎ though FlashInfer supports the native width. This makes sparse attention ⏎ process four times as many query heads. This change selects the native width ⏎ for supported FlashInfer decode shapes and supplies a matching attention sink. ⏎  ⏎ The native-width prefill path introduced by #29927 is preserved. This PR adds ⏎ the corresponding optimization for decode-sized batc …[truncated]

### L2-dc5f59c3a2  (L2, 2026-09-10, sha dc5f59c3a2c4, PR #38947)
TITLE: [Refactor] Clarify DeepSeek V4 metadata names for V4.1 (#38947)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/disaggregation/base/conn.py (+2/-2); python/sglang/srt/disaggregation/common/conn.py (+12/-42); python/sglang/srt/disaggregation/decode.py (+2/-2); python/sglang/srt/disaggregation/mooncake/conn.py (+3/-3); python/sglang/srt/disaggregation/nixl/conn.py (+6/-2); python/sglang/srt/disaggregation/prefill.py (+1/-1); python/sglang/srt/disaggregation/utils.py (+5/-14); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_common_hooks.py (+1/-1); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+14/-27); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+18/-49); (+6 more)
LABELS: deepseek, run-ci, memory-pool
BODY: - Clarify ratio-specific names before DeepSeek V4.1 (#38798): SWA uses `c0_flashmla_metadata` and `c0_cnt`; paged indexer metadata uses `compressed_seq_lens`, `compressed_page_size`, and `max_compressed_seq_len`. ⏎ - Rename the request-state buffer accessor to `get_request_state_buf_infos` and the Python enum member to `StateType.DSV4_REQUEST_STATE`, preserving its `"c128_state"` wire value. Keep C128-specific index calculations and HiCache pool id …[truncated]

### L2-52c191da52  (L2, 2026-09-10, sha 52c191da5239, PR #38404)
TITLE: [Deps] Retire the CUDA 12 lane (#38404)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+15/-77); docker/Dockerfile.cu134 (+13/-68); docker/sgl-deep-ep.Dockerfile (+2/-18); docker/sgl-deep-gemm.Dockerfile (+1/-2); .github/workflows/_docker-build-and-publish.yml (+21/-92); .github/workflows/patch-docker-dev.yml (+1/-1); .github/workflows/release-docker-dev.yml (+6/-8); .github/workflows/release-docker-runtime.yml (+2/-3); .github/workflows/release-docker.yml (+2/-3); .github/workflows/release-pypi-nightly.yml (+5/-5); (+28 more)
LABELS: documentation, sgl-kernel, jit-kernel, release-highlight
BODY: ## Summary ⏎  ⏎ Drops the CUDA 12.9 wheels and images, per the [advance notice][notice] — prerequisite for the PyTorch 2.14 upgrade. ⏎  ⏎ Already-published `-cu129` / `-cu12` image tags are unaffected. v0.5.19 is the last release with a CUDA 12 lane. ⏎  ⏎ [notice]: https://sgl-fru7574.slack.com/archives/C064NB2TAP9/p1786520889229779 ⏎  ⏎ --- ⏎ ### CI States ⏎  ⏎ Latest PR Test (Base): :x: [Run #34440714447](https://github.com/sgl-project/sglang/actions/runs/3444 …[truncated]

### L2-b9899b04c1  (L2, 2026-09-10, sha b9899b04c1c6, PR #33939)
TITLE: [AMD] Add gfx1151 (Strix Halo / Ryzen AI MAX+) Docker image (#33939)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/patches/sgl-kernel-gfx1151.sh (+90/-0); docker/rocm-gfx1151.Dockerfile (+156/-0); .github/workflows/release-docker-amd-gfx1151-nightly.yml (+63/-0)
LABELS: amd, run-ci
BODY: Co-author: @hubertlu-tw, @ankith117 ⏎  ⏎ ## Motivation ⏎  ⏎ The existing ROCm image targets CDNA GPUs and includes components that do not support gfx1151. This PR adds a separate image for Strix Halo systems such as the Ryzen AI MAX+ 395. ⏎  ⏎ The scope is limited to building and running SGLang on gfx1151. General RDNA kernel support and performance tuning remain separate work. ⏎  ⏎ ## Changes ⏎  ⏎ - Add `docker/rocm-gfx1151.Dockerfile`, based on AMD's sta …[truncated]

### L2-209654c420  (L2, 2026-09-11, sha 209654c42090, PR #38826)
TITLE: [NPU][Hicache] Optimize HiCache L2 IO with Memfabric acc_offload (#38826)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/hardware_backend/npu/memory_pool_npu.py (+18/-9); python/sglang/srt/hardware_backend/npu/utils.py (+5/-2); python/sglang/srt/managers/cache_controller.py (+18/-0); python/sglang/srt/mem_cache/pool_host/mha.py (+14/-4); python/sglang/srt/mem_cache/pool_host/mla.py (+375/-27); python/sglang/srt/mem_cache/pool_host/npu_memfabric.py (+159/-0)
LABELS: hicache, npu, run-ci, bypass-fastfail
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Motivation ⏎  ⏎ On Ascend NPU, the existing HiCache L2 (host DRAM) IO path uses `transfer_kv_dim_exchange` with default host memory allocation. This works functionally but has two issues: ⏎  ⏎ 1. **Host memory not AIV-de-referencable**: The default `torch.pin_memory` host buffers are only reachable by the SDMA engine; the fused AIV sparse-copy kernel (`offload.kv_exchange_copy`) from the Memfabric `acc_offload` library can only de-reference host V …[truncated]

### L2-833bce9df5  (L2, 2026-09-11, sha 833bce9df57f, PR #34432)
TITLE: [AMD][DCP 1/N] add dcp support for aiter backend (#34432)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.optimization.weight_absorption, L2.backend.aiter_mla, L2.kernel.aiter_mla_gluon, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/layers/attention/aiter_mla_gluon.py (+19/-6); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py (+1/-1); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py (+75/-28); python/sglang/srt/arg_groups/model_overrides/kimi_k3.py (+25/-1); python/sglang/srt/layers/attention/aiter_backend.py (+393/-17); python/sglang/srt/layers/attention/triton_backend.py (+15/-2); python/sglang/srt/layers/dcp/comm.py (+18/-13); python/sglang/srt/models/deepseek_common/attention_backend_handler.py (+2/-0); python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha_rocm.py (+1/-10)
LABELS: amd, run-ci, jit-kernel, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ This patch is 1/N of the original https://github.com/sgl-project/sglang/pull/32796 to enable dcp support for aiter backend. Sample server command is as following: ⏎ ```bash ⏎ python3 -m sglang.launch_server --model-path /data-models/Kimi-K3 --trust-remote-code \ ⏎     --tp-size 8 --dcp-size 8 \ ⏎ 	--prefill-attention-backend aiter \ ⏎ 	--decode-attention-backend aiter \ ⏎ 	--dtype bfloat16 \ ⏎ 	--kv-cache-dtype fp8_e4m3 \ ⏎ 	--context-le …[truncated]

### L2-d7c284b894  (L2, 2026-09-11, sha d7c284b894e9, PR #39106)
TITLE: [AMD] Use the triton DSA backend for GLM-5.2 MXFP4 on MI355X (#39106)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/GLM/GLM-5.2.mdx (+1/-1); docs/src/snippets/configs/zai-org/glm-5.2.jsx (+11/-8)
LABELS: documentation
BODY: ## Summary ⏎ Switch the MI355X MXFP4 (`amd/GLM-5.2-MXFP4`, TP4) recipes from the tilelang DSA ⏎ backend to triton, which is already SGLang's default on ROCm. ⏎  ⏎ Scope is the four MXFP4 cells only. The `tp=8` FP8/BF16 MI355X cells and the ⏎ gfx942 (MI300X/MI325X) cells keep tilelang: their published numbers were ⏎ measured with it, and at `tp=8` GLM-5.2 has 8 query heads per rank, which does ⏎ not meet the gfx950 sparse-MLA tuning shape. ⏎  ⏎ ## Why ⏎ At `tp=4` wit …[truncated]

### L2-747734dce4  (L2, 2026-09-11, sha 747734dce44c, PR #38807)
TITLE: [NPU]glm5.2 fp8 memory opt (#38807)
SOURCES: path_core
ARTIFACT_HINTS: L2.backend.npu_mla
FILES: python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py (+14/-4); python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py (+8/-8)
LABELS: deepseek, npu, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ Certain weights can be released after use to increase the available NPU memory. ⏎  ⏎ ## Modifications ⏎  ⏎ Certain weights can be released after use to increase the available NPU memory. ⏎  ⏎ ## Accuracy Tests ⏎  ⏎ None ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎ None ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAI …[truncated]

### L2-165d8dd177  (L2, 2026-09-11, sha 165d8dd17736, PR #38827)
TITLE: [NPU][Hicache] Add Ascend Memcache Hicache L3 storage backend (#38827)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv, L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/fields/memory.py (+2/-1); python/sglang/srt/arg_groups/hicache_hook.py (+2/-2); python/sglang/srt/environ.py (+6/-0); python/sglang/srt/hardware_backend/npu/memory_pool_npu.py (+18/-9); python/sglang/srt/hardware_backend/npu/utils.py (+5/-2); python/sglang/srt/managers/cache_controller.py (+27/-1); python/sglang/srt/mem_cache/pool_host/group.py (+25/-0); python/sglang/srt/mem_cache/pool_host/mha.py (+14/-4); python/sglang/srt/mem_cache/pool_host/mla.py (+457/-27); python/sglang/srt/mem_cache/pool_host/npu_memfabric.py (+159/-0); (+5 more)
LABELS: documentation, hicache, npu, run-ci, bypass-fastfail
BODY: ## Motivation ⏎  ⏎ SGLang HiCache uses a tiered KV cache architecture: L1 (device/HBM) → L2 (host memory) → L3 (external storage). On CUDA, storage backends such as Mooncake are already supported as HiCache L3, enabling cross-node prefix cache reuse and increasing the effective KV cache capacity. ⏎  ⏎ On Ascend NPU, however, there was previously no corresponding L3 storage backend. As a result, HiCache on Ascend was limited to device + host tiers and …[truncated]

### L2-0d08668821  (L2, 2026-09-11, sha 0d08668821d0, PR #39190)
TITLE: [Cookbook] Kimi-K3: keep DCP under HiCache L1+L2 with DSPARK (#39190)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx (+2/-1); docs/src/snippets/configs/moonshotai/kimi-k3.jsx (+9/-24)
LABELS: documentation
BODY: ## Motivation ⏎  ⏎ HiCache under DCP accepts DSPARK speculative decoding since #35221 (`resolve_hicache_dcp_compatibility` now only rejects L3 storage backends, non-DSPARK draft paths, LMCache, HiSparse, and non-MLA models). The Kimi-K3 cookbook still described the older gate, so on the DCP cells (Blackwell Balanced / High-Throughput) selecting DSPARK + HiCache L1+L2 stripped `--dcp-size` and emitted a "HiCache under DCP rejects speculative decoding" …[truncated]

### L2-6ba96d329f  (L2, 2026-09-11, sha 6ba96d329fb8, PR #39165)
TITLE: [DCP] Resolve --dcp-comm-backend to fi_a2a/a2a by default for every model (#39165)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: docs/docs/advanced_features/dcp.mdx (+5/-5); docs/docs/advanced_features/server_arguments.mdx (+2/-2); python/sglang/srt/arg_groups/fields/parallel.py (+4/-3); python/sglang/srt/arg_groups/model_overrides/kimi_k3.py (+4/-22); python/sglang/srt/arg_groups/overrides.py (+27/-0); python/sglang/srt/arg_groups/parallel_hook.py (+6/-5); python/sglang/srt/arg_groups/pipeline.py (+2/-2); python/sglang/srt/layers/dcp/comm.py (+2/-13); python/sglang/srt/utils/common.py (+12/-0); test/registered/unit/server_args/test_server_args.py (+40/-0)
LABELS: documentation, run-ci
BODY: ## Motivation ⏎  ⏎ #37767 turned `fi_a2a` on by default, but only through the Kimi-K3 model override. Every other DCP model (DeepSeek, GLM-5.3-Flash, Kimi Linear, ...) silently kept `ag_rs` unless the flag was passed explicitly. ⏎  ⏎ ## Modifications ⏎  ⏎ - `--dcp-comm-backend` defaults to unset. A new `handle_dcp_defaults` pass resolves it before the model overrides: `fi_a2a` where `is_fi_a2a_supported` holds (Blackwell, DCP group within one MNNVL domain),  …[truncated]

### L2-21289cfd50  (L2, 2026-09-12, sha 21289cfd50a8, PR #39116)
TITLE: [AMD] Fix Dspark accept length and reduce host bubble on DSV4 (#39116)
SOURCES: corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+94/-84); test/registered/amd/test_deepseek_v4_pro_fp4_dspark.py (+61/-0)
LABELS: amd, deepseek, run-ci
DEEP_STUDY: deep-study correctness case sglang:21289cfd50: class=integration_backend_cudagraph; symptom=wrong_output_or_accuracy; introducing=unknown || deep-study performance PR (system_performance)
BODY: ## Motivation ⏎ Fix accept length for dspark and reduce bubble in draft (labeled as target verify) ⏎  ⏎  ⏎ ## Modifications ⏎ **1. Refresh captured SWA write locations in place** ⏎  ⏎ - UnifiedKvMetadata.copy_ previously rebound swa_loc through assign_fields. For metadata built outside the graph, this updates the Python reference but leaves the buffer referenced by captured kernels unchanged. ⏎ - DSPARK's draft KV-store path consumes swa_loc, so subseque …[truncated]

### L2-a66451c058  (L2, 2026-09-12, sha a66451c058dc, PR #38845)
TITLE: [GLM-5.3 Flash] Restore and enable KPool metadata fusion (#38845)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/kernels/ops/attention/dsa_kpool_metadata/__init__.py (+1/-0); python/sglang/kernels/ops/attention/dsa_kpool_metadata/decode.py (+225/-0); python/sglang/kernels/ops/attention/dsa_kpool_metadata/draft_extend.py (+308/-0); python/sglang/kernels/ops/attention/dsa_kpool_metadata/scan.py (+9/-0); python/sglang/kernels/ops/attention/dsa_kpool_metadata/verify.py (+313/-0); python/sglang/srt/environ.py (+2/-0); python/sglang/srt/layers/attention/dsa/dsa_backend_mtp_precompute.py (+8/-12); python/sglang/srt/layers/attention/dsa/dsa_metadata_manager.py (+320/-0); python/sglang/srt/layers/attention/dsa_backend.py (+32/-133); python/sglang/test/kits/dsa_metadata_kit.py (+147/-0); (+2 more)
LABELS: jit-kernel
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ Restore the KPool metadata fusion removed by #38071 and enable it by default for supported CUDA KPool geometry. GLM-5.3-Flash currently builds pool-aware metadata through the ordinary path; this restores fused decode, target-verify and draft-extend construction, plus reuse of derived metadata between compatible MTP draft backends. ⏎  ⏎ This is the first of three dependent restoration PRs. `SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION` …[truncated]

### L2-6dc7b3421b  (L2, 2026-09-12, sha 6dc7b3421b48, PR #34556)
TITLE: Inference Support Mamba 2 and 1 (#34556)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/docs/supported-models/generative_models.mdx (+5/-0); python/sglang/srt/configs/__init__.py (+5/-0); python/sglang/srt/configs/hybrid_arch.py (+7/-1); python/sglang/srt/configs/mamba.py (+80/-0); python/sglang/srt/configs/mamba2.py (+57/-0); python/sglang/srt/configs/mamba_utils.py (+47/-0); python/sglang/srt/configs/model_config.py (+42/-8); python/sglang/srt/layers/attention/mamba/mamba1.py (+472/-0); python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py (+20/-0); python/sglang/srt/models/falcon_mamba.py (+252/-0); (+6 more)
LABELS: documentation, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Motivation ⏎  ⏎ Add SGLang support for the **Mamba family of pure state-space models**, none of which could be served before — they crashed at startup or fell through to attention code paths they don't fit. This PR covers both generations: ⏎  ⏎ - **Mamba2 (SSD):** `mistralai/Mamba-Codestral-7B-v0.1` ⏎ - **Mamba-1 (selective-scan):** `tiiuae/falcon-mamba-7b` / `-instruct`, `state-spaces/mamba-130m-hf`, and the raw `state-spaces/mamba-{130m,790m,2.8b …[truncated]

### L2-5ebb16005d  (L2, 2026-09-13, sha 5ebb16005d28, PR #39171)
TITLE: [DeepSeek-V4.1] Bump FlashMLA to the fork's rebase head (v4.1 kernels) (#39171)
SOURCES: path_core, subject_keyword, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L2.build.flashmla_sgl_kernel
FILES: python/sglang/kernels/aot/cmake/flashmla.cmake (+57/-43); python/sglang/kernels/aot/csrc/flashmla_extension.cc (+30/-4); python/sglang/test/kits/basic_decode_correctness_kit.py (+4/-1)
LABELS: sgl-kernel, run-ci, run-ci-extra
BODY: ## Motivation ⏎  ⏎ It's just #38942 rebased on main branch ⏎  ⏎  ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ ## Accuracy Tests ⏎  ⏎  ⏎  ⏎ ## Speed Tests and Profiling ⏎  ⏎  ⏎  ⏎ ## Checklist ⏎  ⏎  ⏎ ## Review and Merge Process ⏎  ⏎ 1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process). ⏎ 2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main …[truncated]

### L2-2ec4bbcbd4  (L2, 2026-09-13, sha 2ec4bbcbd47b, PR #37506)
TITLE: [unified-memory] PD disaggregation for every unified pool shape (#37506)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/fields/memory.py (+4/-2); python/sglang/srt/disaggregation/decode.py (+1/-1); python/sglang/srt/disaggregation/mooncake/conn.py (+17/-2); python/sglang/srt/disaggregation/prefill.py (+1/-1); python/sglang/srt/mem_cache/allocator/hisparse.py (+11/-0); python/sglang/srt/mem_cache/allocator/swa.py (+13/-0); python/sglang/srt/mem_cache/allocator/unified_hybrid_swa.py (+157/-29); python/sglang/srt/mem_cache/allocator/unified_mamba.py (+13/-7); python/sglang/srt/mem_cache/allocator/unified_sub_pool.py (+25/-0); python/sglang/srt/mem_cache/kv_cache_builder.py (+15/-5); (+8 more)
LABELS: ready-to-merge, memory-pool
BODY: **Stack 1/3:** main → #37506 → #37418 → #37507. ⏎  ⏎ Enable Mooncake PD transfers for the unified MHA, MLA, SWA, and full/SWA/Mamba pool layouts. Transfers register whole page envelopes, translate virtual IDs to physical page IDs, and prevent relocation while requests have outstanding transfers. ⏎  ⏎ The reviewed implementation also: ⏎ - Prices only the retained SWA tail during paged decode preallocation and rolls back newly allocated pages on failure. ⏎ - P …[truncated]

### L2-96a95171c7  (L2, 2026-09-13, sha 96a95171c714, PR #39346)
TITLE: chore: bump sglang-kernel version to 0.4.7 (#39346)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: docker/Dockerfile (+1/-1); python/pyproject.toml (+1/-1); python/sglang/srt/entrypoints/engine.py (+1/-1)
LABELS: dependencies, run-ci
BODY: ## Summary ⏎  ⏎ This PR bumps the `sglang-kernel` version to `0.4.7` across SGLang files to match the version defined in `python/sglang/kernels/aot/pyproject.toml`. ⏎  ⏎ **Kernel Version:** `0.4.7` ⏎  ⏎ ## Files Updated ⏎ - docker/Dockerfile ⏎ - python/pyproject.toml ⏎ - python/sglang/srt/entrypoints/engine.py ⏎  ⏎ ## Context ⏎  ⏎ The kernel version in `python/sglang/kernels/aot/pyproject.toml` has been updated. This PR ensures that all SGLang files referencing the `sglan …[truncated]

### L2-5a132c061b  (L2, 2026-09-13, sha 5a132c061b3c, PR #39232)
TITLE: [Perf] trtllm_mla: reuse the fused fp8 KV/Q prepare on target verify (#39232)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+25/-7)
LABELS: blackwell, run-ci
ISSUES: #39107 [Feature] TRTLLM MLA target verification misses fused FP8 KV/Q preparation in forward_extend
DEEP_STUDY: deep-study performance PR ()
BODY: ## Motivation ⏎  ⏎ Closes #39107. ⏎  ⏎ `TRTLLMMLABackend.forward_decode` already fuses the fp8 no-RoPE preparation (bf16 -> fp8 quantize + KV scatter + `[q_nope | q_rope]` concat) into one `set_mla_kv_concat_q_fp8` launch. Target verify dispatches through `forward_extend`, which on the same inputs still ran the unfused chain unconditionally: `concat_mla_absorb_q` + three aten fp8 casts + the pool's Triton scatter, i.e. five launches per MLA layer instead …[truncated]

### L2-f2111715cd  (L2, 2026-09-13, sha f2111715cde4, PR #33426)
TITLE: [fa] Make the FlashAttention backend extensible by subclasses (#33426)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.fa3_fa4_mla
FILES: python/sglang/srt/layers/attention/flashattention_backend.py (+32/-11); test/registered/attention/unittests/dense/test_fa4.py (+100/-0)
LABELS: documentation, run-ci
DEEP_STUDY: deep-study: this PR was reverted by PR 39405 (confirmed_revert, reason=premature_or_process)
BODY: ## Motivation ⏎  ⏎ `FlashAttentionBackend` is hard to specialize without copying it. Three things a subclass may need are not exposed today, leaving only the options of duplicating the backend or patching module globals: the logsumexp the kernel already computes, per-layer control of split-KV, and the choice of FA4 build. This adds a seam for each. All three are no-ops for current behavior. ⏎  ⏎ ## Modifications ⏎  ⏎ **`return_lse` on `forward_extend`  …[truncated]

### L2-5200508b0f  (L2, 2026-09-14, sha 5200508b0fd2, PR #32888)
TITLE: [AMD][gfx95] Fill the chunked-prefill compute budget exactly (#32888)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/environ.py (+4/-0); python/sglang/srt/managers/schedule_policy.py (+87/-7)
LABELS: amd, run-ci, bypass-fastfail, run-ci-extra
BODY: ## Summary ⏎  ⏎ A prefill batch never reaches `--chunked-prefill-size`: `_update_prefill_budget` charges the page-ceiled extend length against `rem_chunk_tokens` and `rem_input_tokens`, and `add_one_req` floors the last truncation back to a page. On gfx95 that shortfall is expensive — the aiter MLA absorb bmm picks its `EVEN_MN` specialization from a compile-time `constexpr` on `M % BLOCK_SIZE_M`, so a misaligned M runs a different compiled binary, n …[truncated]

### L2-5aa9b8fb3e  (L2, 2026-09-14, sha 5aa9b8fb3ec9, PR #37413)
TITLE: [AMD][DSV4] feat: enable fp8 two-pool unified_kv on gfx950 (#37413)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh (+145/-33); python/sglang/kernels/ops/attention/dsv4/compress.py (+28/-2); python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/env_gate.py (+24/-1); python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/layout.py (+62/-0); python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/runtime.py (+293/-18); python/sglang/kernels/ops/attention/fused_qk_norm_rope_store.py (+188/-0); python/sglang/srt/environ.py (+6/-0); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+81/-12); python/sglang/srt/layers/attention/dsv4/compressor_v2.py (+16/-1); python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py (+122/-8); (+11 more)
LABELS: quant, deepseek, sgl-kernel, run-ci, jit-kernel, run-ci-extra, memory-pool
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Motivation ⏎  ⏎  ⏎ Under `unified_kv`, DSv4 keeps KV in bf16 across the full latent — `qk_nope_head_dim` 448 plus `qk_rope_head_dim` 64, **1024 B per token per layer**. The nope half actually does not need that precision. Storing nope as fp8 with per-tile E8M0 scales while leaving rope in bf16 cuts the row to **640 B** (512 B of fp8 nope at a fixed stride, plus the same 128 B of bf16 rope), which measures out at **1.51× the KV capacity at identic …[truncated]

### L2-0163f8ff74  (L2, 2026-09-14, sha 0163f8ff74c3, PR #35233)
TITLE: [AMD] Fix registered HiCache host pointer aliases (#35233)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters, L2.pool.mla_token_kv
FILES: python/sglang/kernels/aot/csrc/common_extension.cc (+1/-0); python/sglang/kernels/aot/csrc/common_extension_rocm.cc (+1/-0); python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu (+40/-4); python/sglang/kernels/aot/include/sgl_kernel_ops.h (+2/-0); python/sglang/kernels/aot/python/sgl_kernel/kvcacheio.py (+5/-0); python/sglang/kernels/jit/csrc/kvcacheio/hicache.cuh (+9/-8); python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh (+15/-6); python/sglang/kernels/jit/csrc/kvcacheio/transfer_mamba.cuh (+5/-3); python/sglang/kernels/jit/include/sgl_kernel/runtime.cuh (+16/-0); python/sglang/srt/mem_cache/memory_pool_host.py (+10/-9); (+5 more)
LABELS: amd, hicache, sgl-kernel, run-ci, jit-kernel, memory-pool
BODY: ## Motivation ⏎  ⏎ Registered HiCache host allocations can have different CPU and GPU virtual addresses. This is observable on MI355X, where `hipDeviceAttributeCanUseHostPointerForRegisteredMem` is false. Passing the CPU VA to a custom kernel causes a GPU memory fault even though runtime copy APIs can still accept that address. ⏎  ⏎ The affected paths include AOT/JIT HiCache transfers, Mamba transfers, HiSparse copies, and host-pool pointer tables. CPU-s …[truncated]

### L2-f81fbc749a  (L2, 2026-09-14, sha f81fbc749abc, PR #38941)
TITLE: [Fix] Merge adjacent KV-row frees so a mid-page split under DCP cannot double-free (#38941)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/mem_cache/base_prefix_cache.py (+4/-2); python/sglang/srt/mem_cache/common.py (+11/-0); test/registered/unit/mem_cache/test_free_kv_row_coalesce.py (+92/-0)
LABELS: run-ci, bypass-fastfail, run-ci-extra, unified-radix-cache
BODY: ## Motivation ⏎  ⏎ Radix-cache servers running hybrid Mamba + MLA models (Kimi-K3, Kimi-Linear) with `--dcp-size 8` crash under ordinary load: ⏎  ⏎ ``` ⏎ mem_cache/allocator/base.py, free_segments ⏎ AssertionError: segment at 576 shares a page with the one ending at 576 ⏎ ``` ⏎  ⏎ Every scheduler dies at once. We hit it within a minute of an MMLU-style batch (many prefix-sharing logprob requests finishing together), and after a couple of hours of agentic  …[truncated]

### L2-e4a6b090f4  (L2, 2026-09-15, sha e4a6b090f45f, PR #39553)
TITLE: [CI] Run test_unified_radix_cache_kl_dcp on cutedsl_mla with bf16 KV cache (#39553)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dcp.py (+2/-4)
LABELS: unified-radix-cache
BODY: ## Motivation ⏎  ⏎ `test_unified_radix_cache_kl_dcp.py` has failed ~50% of scheduled `extra-b-test-4-gpu-b200` runs since 09/11 (`avg_kl_div` 0.010-0.013 vs threshold 0.01, both retry attempts). ⏎  ⏎ Bisected to #35770: `resolve_attn_backend()` now unwraps `HybridLinearAttnBackend.full_attn_backend`, so Kimi Linear prefill hits `TokenspeedMLABackend.prepare_prefill_qkv` and quantizes Q/K/V to FP8 before the prefill kernel (#38152 made the hook NoPE-safe  …[truncated]

### L2-99060191e7  (L2, 2026-09-15, sha 99060191e7eb, PR #36821)
TITLE: [KDA] Support ReplaySSM ring-write in the fused chain-verify kernel (#36821)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: benchmark/kernels/bench_kda_verify_sweep.py (+285/-0); python/sglang/kernels/ops/attention/fla/fused_kda_conv_recurrent_verify.py (+127/-13); python/sglang/srt/layers/attention/linear/kda_backend.py (+127/-24); test/registered/kernel/attention/test_kda_fused_verify_backend.py (+322/-0); test/registered/kernels/test_fused_kda_conv_recurrent_verify.py (+74/-13)
LABELS: run-ci, jit-kernel, run-ci-extra
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Motivation ⏎  ⏎ The fused KDA chain-verify kernel (`fused_kda_conv_gating_verify`, gated by `SGLANG_OPT_FUSED_KDA_VERIFY`) and ReplaySSM spec-verify (`--enable-linear-replayssm` + spec ring) were mutually exclusive: `_can_run_fused_chain_verify` returned `False` whenever `replayssm_rawv` was set, so enabling ReplaySSM silently dropped the verify hot path back to the unfused reference pair (transpose-copy + `causal_conv1d_update` + transpose-copy …[truncated]

### L2-2c37b90ad6  (L2, 2026-09-15, sha 2c37b90ad6e9, PR #39487)
TITLE: [Fix][DCP] Localize widened KV ids in MLA retraction CPU backup/restore (#39487)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/mem_cache/memory_pool.py (+7/-0); python/sglang/srt/layers/dcp/layout.py (+13/-0); python/sglang/srt/mem_cache/pool_host/base.py (+0/-13); python/sglang/srt/mem_cache/pool_host/mla.py (+13/-4); test/registered/unit/mem_cache/test_decode_retraction_backup.py (+68/-1); test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py (+5/-19)
LABELS: hicache, memory-pool
ISSUES: #38645 [Bug][DCP][PD Disagg] Decode retraction crashes in get_cpu_copy with CUDA device-side assert
BODY: ## Motivation ⏎  ⏎ Fixes #38645 (related draft: #38790). On a PD decode node with `--dcp-size > 1`, `retract_decode` backs up the retracted request's KV through `MLATokenToKVPool.get_cpu_copy`, which indexes `kv_buffer` with the widened `req_to_token` ids. Ids past this rank's rows trip a CUDA device-side assert; in-range ones back up and restore the wrong token's row. ⏎  ⏎ ## Modifications ⏎  ⏎ - `HostKVCache.maybe_dcp_kernel_indices` moves to `layers/dcp/l …[truncated]

### L2-8a20062652  (L2, 2026-09-15, sha 8a2006265247, PR #39423)
TITLE: [NPU][Bugfix] Disable pinned memory to fix DeepSeek-V2 DP-attention hang (#39423)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: python/sglang/srt/platforms/npu.py (+1/-3); test/registered/unit/platforms/test_platform_interface.py (+6/-3)
LABELS: npu, run-ci
BODY: ## Motivation ⏎  ⏎ #36472 introduced the in-tree `NPUSRTPlatform` and made NPU machines resolve to it instead of the base `SRTPlatform`. Unlike the base default (`is_pin_memory_available=False`), the new class returns `True`, silently switching hot paths such as the next-token-logits copies and the xgrammar bitmask allocation to pinned memory with `non_blocking` H2D copies. ⏎  ⏎ torch_npu's pinned-memory + async-copy semantics are not verified agains …[truncated]

### L2-91f691c490  (L2, 2026-09-15, sha 91f691c49077, PR #39646)
TITLE: dsv4.1: standalone kernels and Python wrappers (#39646)
SOURCES: path_core, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/deepseek_v4/flashmla_sched_meta.cuh (+276/-0); python/sglang/kernels/ops/attention/dsv4/flashmla_sched_meta.py (+69/-0); python/sglang/kernels/jit/csrc/gemm/small_gemm_bf16.cuh (+249/-0); python/sglang/kernels/jit/csrc/gemm/tiny_gemm.cuh (+4/-19); python/sglang/kernels/jit/include/sgl_kernel/gemm/dot_product.cuh (+40/-0); python/sglang/kernels/ops/attention/dsv4/candidate_blocks.py (+207/-0); python/sglang/kernels/ops/attention/dsv4/indexer_postprocess.py (+74/-0); python/sglang/kernels/ops/attention/dsv4/q_rope_store.py (+107/-0); python/sglang/kernels/ops/attention/dsv4/wo_a_bf16.py (+138/-0); python/sglang/kernels/ops/embeddings/__init__.py (+40/-0); (+15 more)
LABELS: run-ci, jit-kernel
BODY: ## Summary ⏎ - Extract standalone DeepSeek V4.1 kernels and Python wrappers from #38798 as the first layer of a linear PR stack. ⏎ - Add small BF16 GEMMs and WO-A projections, Engram hash/gather/gate operations, decode scheduling metadata, candidate selection helpers, Q RoPE/store, FP32 RMSNorm, HC mixing/MXFP8 epilogues, and row-wise argmax. ⏎ - Keep model dispatch, scheduler/cache ownership, and runtime integration in #38798. Existing Top-k implement …[truncated]

### L2-faaff1eca8  (L2, 2026-09-15, sha faaff1eca887, PR #39648)
TITLE: dsv4.1: Top-k kernels and candidate selection (#39648)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: L2.backend.sparse_mla_adapters
FILES: python/sglang/kernels/jit/csrc/deepseek_v4/block_amax.cuh (+155/-0); python/sglang/kernels/jit/csrc/deepseek_v4/candidate_block_table.cuh (+218/-0); python/sglang/kernels/jit/csrc/deepseek_v4/topk_bf16_small.cuh (+440/-0); python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh (+239/-162); python/sglang/kernels/jit/csrc/occupancy/cluster_probe.cuh (+55/-0); python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh (+338/-314); python/sglang/kernels/jit/utils/occupancy.py (+47/-0); python/sglang/kernels/ops/attention/dsv4/candidate_table.py (+93/-0); python/sglang/kernels/ops/attention/dsv4/topk.py (+86/-27); python/sglang/test/kits/dsa_metadata_kit.py (+3/-0); (+2 more)
LABELS: run-ci, jit-kernel
BODY: ## Summary ⏎ - Add the DeepSeek-V4.1 sparse-indexer kernels that sit beside Top-k v2: a bf16 small-row top-k, the two-level indexer's block-max keys and sorted block table, and a cluster-occupancy probe. ⏎ - Rework Top-k v2's edge handling and cluster dispatch (details below); the existing suite still passes and two new regression cases pin the fixed `-inf` tie behaviour. ⏎  ⏎ ## New kernels ⏎ - `topk_bf16_small` (`topk_transform_bf16_small`): bf16 top-k f …[truncated]

### L2-a813224e78  (L2, 2026-09-16, sha a813224e7808, PR #37810)
TITLE: [ROCm][DSV4] Enable breakable CUDA graph prefill (#37810)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py (+5/-4); python/sglang/srt/layers/attention/base_attn_backend.py (+4/-0); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+273/-48); python/sglang/srt/managers/scheduler_components/dp_attn.py (+4/-3); python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py (+20/-0); test/registered/amd/test_dsv4_hip_bcg_metadata.py (+419/-0); test/registered/unit/managers/scheduler_components/test_dp_attn.py (+40/-0); test/registered/unit/model_executor/runner/test_prefill_cuda_graph_padding.py (+13/-2)
LABELS: amd, deepseek, run-ci, jit-kernel, run-ci-extra
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ Enable breakable CUDA graph (BCG) prefill for the ROCm DeepSeek-V4 HIP radix backend. ⏎  ⏎ ## Changes ⏎  ⏎ - Add captured-metadata support for HIP radix prefill capture and replay. ⏎ - Refresh attention, unified-KV, indexer, compressor, and FP4 workspace metadata in place. ⏎ - Preserve the existing CUDA memory-pressure guard while allowing explicit HIP BCG selection. ⏎ - Keep ROCm DSV4 mixed prefill/decode batches eager under DP attention; …[truncated]

### L2-dbd7281b06  (L2, 2026-09-16, sha dbd7281b065e, PR #38402)
TITLE: fix(npu): fix hybrid KV transfer with PP prefill in PD disaggregation (#38402)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/srt/disaggregation/ascend/conn.py (+16/-0); python/sglang/srt/disaggregation/mooncake/conn.py (+9/-0); python/sglang/srt/mem_cache/memory_pool.py (+3/-0); python/sglang/srt/models/kimi_k3.py (+0/-1); test/registered/unit/disaggregation/test_pp_hybrid_kv_transfer.py (+72/-0); test/registered/unit/mem_cache/test_mamba_unittest.py (+22/-0)
LABELS: run-ci, memory-pool
BODY: ## Summary ⏎  ⏎ - route Ascend hybrid-linear PP KV transfer through global layer-id pairing ⏎ - publish one global layer id per NPU KV buffer group ⏎ - validate Mamba state slot sizes after pairing PP-local source layers with decode layers ⏎  ⏎ Ascend already supports PP + PD for dense MLA/MHA layouts. Kimi-K3 uses a HybridLinearKVPool: only sparse full-attention layers own KV buffers, and Ascend exposes multiple buffer groups for each of those layers. …[truncated]

### L2-4678536df6  (L2, 2026-09-16, sha 4678536df6a6, PR #32618)
TITLE: [CPU] Add fp8_per_tensor_scaled_mm_cpu kernel (#32618)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/aot/csrc/cpu/bmm.cpp (+1/-0); python/sglang/kernels/aot/csrc/cpu/gemm.h (+3/-1); python/sglang/kernels/aot/csrc/cpu/gemm_fp8.cpp (+266/-60); python/sglang/kernels/aot/csrc/cpu/torch_extension_cpu.cpp (+12/-0); python/sglang/srt/layers/quantization/fp8.py (+21/-0); python/sglang/srt/model_executor/cpu_graph_runner.py (+13/-0); test/registered/cpu/test_gemm.py (+35/-0)
LABELS: sgl-kernel, intel, cpu, run-ci
BODY: ## Motivation ⏎  ⏎  ⏎ Add a generic CPU implementation of `fp8_per_tensor_scaled_mm` for W8A16 quantization. ⏎  ⏎ The existing CPU FP8 GEMM support in `bmm_cpu` is mainly designed for batched matrix multiplication workloads, such as the DeepSeek MLA path. This PR adds the missing per-tensor FP8 GEMM path. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎ - Add and register the `fp8_per_tensor_scaled_mm_cpu` kernel for W8A16 FP8 GEMM. ⏎ - Reused the existing CPU FP8 scaled GEMM …[truncated]

### L2-13d593b6cf  (L2, 2026-09-16, sha 13d593b6cf88, PR #39652)
TITLE: dsv4.1: compression, KV I/O, and metadata kernels (#39652)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/jit/csrc/deepseek_v4/c1.cuh (+312/-0); python/sglang/kernels/jit/csrc/deepseek_v4/c2.cuh (+398/-0); python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh (+6/-4); python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh (+50/-11); python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh (+47/-10); python/sglang/kernels/jit/csrc/deepseek_v4/store.cuh (+137/-24); python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp4_utils.cuh (+60/-0); python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh (+212/-0); python/sglang/kernels/ops/attention/dsv4/attn.py (+43/-6); python/sglang/kernels/ops/attention/dsv4/compress.py (+14/-2); (+9 more)
LABELS: quant, deepseek, run-ci, jit-kernel
BODY: ## Summary ⏎ - Add `KVLayout` (`kernels/ops/attention/dsv4/kv_layout.py`, `include/sgl_kernel/deepseek_v4/kv_layout.cuh`): the paged FlashMLA main-KV formats as one enum with their per-token data / scale bytes, tile size and page alignment. `V4` is the existing 584-byte layout (448 fp8 nope + 64 bf16 rope + ue8m0 scales); `V41` (528 B, every dim fp8 with one ue8m0 scale per 32) and `V41_FP4` (288 B, e2m1 codes with one e4m3 scale per 16) are the De …[truncated]

### L2-e7f7447333  (L2, 2026-09-16, sha e7f744733333, PR #37615)
TITLE: [kv-shard 2/4] Sharded pools (#37615)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L2.backend.trtllm_mla, L2.backend.fa3_fa4_mla, L2.runner.cuda_graph_mla
FILES: python/sglang/srt/layers/attention/trtllm_mla_backend.py (+13/-0); python/sglang/srt/layers/attention/flashattention_backend.py (+32/-0); python/sglang/srt/layers/attention/kv_shard_hooks.py (+67/-0); python/sglang/srt/mem_cache/page_interleave.py (+29/-2); python/sglang/srt/mem_cache/page_interleave_pool.py (+703/-0); test/registered/unit/layers/attention/test_flashattention_pa_swa_prefill_lens_size.py (+19/-2); test/registered/unit/layers/attention/test_kv_shard_hooks.py (+112/-0); test/registered/unit/mem_cache/test_page_interleave_shard.py (+734/-8)
LABELS: high priority, blackwell, run-ci, bypass-fastfail, run-ci-extra, unified-radix-cache, memory-pool
BODY: ## Motivation ⏎  ⏎  ⏎ This adds the KV pools that actually stripe. ⏎  ⏎ Pool tensors keep their stock shape and size on every rank; only the written rows differ — each rank persists exactly the rows it owns under part 1's bijection. The consequence drives the whole design: during a sharded prefill, extend attention never reads the KV pool, because a rank's pool holds only its stripe of the prefix *and* of the current chunk. It reads a per-layer assemb …[truncated]

### L2-25ce8063f7  (L2, 2026-09-16, sha 25ce8063f7db, PR #30775)
TITLE: Pipeline parallelism x speculative decoding (EAGLE/MTP) compatibility (#30775)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/srt/arg_groups/validation_hook.py (+30/-0); python/sglang/srt/environ.py (+4/-0); python/sglang/srt/managers/scheduler.py (+106/-25); python/sglang/srt/managers/scheduler_pp_mixin.py (+406/-20); python/sglang/srt/managers/tp_worker.py (+25/-0); python/sglang/srt/managers/utils.py (+19/-0); python/sglang/srt/model_executor/cuda_graph_buffer_registry.py (+4/-1); python/sglang/srt/model_executor/model_runner.py (+4/-3); python/sglang/srt/model_executor/model_runner_components/layer_setup.py (+3/-0); python/sglang/srt/model_executor/runner/base_runner.py (+6/-1); (+8 more)
LABELS: hicache, run-ci, bypass-fastfail, run-ci-extra, unified-radix-cache
BODY: ## Motivation ⏎  ⏎ PP and speculative decoding are currently mutually exclusive (`server_args.py` asserts). On PCIe-only multi-GPU boxes (no NVLink), TP-only scaling is bottlenecked by per-layer all-reduces, and PP is the right scaling axis. Combining the two requires PP x spec compatibility, which vLLM has (vllm-project/vllm#16568) and SGLang doesn't. ⏎  ⏎ This PR implements it behind `SGLANG_ENABLE_PP_SPEC=1`; default behavior is unchanged. ⏎  ⏎ **Dependen …[truncated]

### L2-f0bf652534  (L2, 2026-09-17, sha f0bf652534c5, PR #38526)
TITLE: Add Ling-3.0-flash-VL model support (#38526)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.dispatch.server_args_defaults
FILES: python/sglang/kernels/ops/attention/rotary_triton.py (+2/-2); python/sglang/srt/arg_groups/model_overrides/__init__.py (+1/-0); python/sglang/srt/arg_groups/model_overrides/bailing_moe_v3.py (+48/-0); python/sglang/srt/arg_groups/overrides.py (+2/-0); python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py (+3/-0); python/sglang/srt/configs/__init__.py (+4/-1); python/sglang/srt/configs/bailing_hybrid.py (+79/-1); python/sglang/srt/configs/bailing_moe_v2.py (+172/-0); python/sglang/srt/configs/hybrid_arch.py (+4/-0); python/sglang/srt/configs/model_config.py (+36/-5); (+27 more)
LABELS: run-ci, deterministic, jit-kernel, run-ci-extra
BODY: ## Summary ⏎  ⏎ This PR adds native SGLang support for [`inclusionAI/Ling-3.0-flash-VL`](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL), including text, image, and video serving through the OpenAI-compatible API. It supports the official [BF16](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL), [FP8](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL-fp8), [INT4](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL-int4), and [FP4](https …[truncated]

### L2-84d7604b7e  (L2, 2026-09-17, sha 84d7604b7e6b, PR #39439)
TITLE: [XPU] weekly simple model enablement 2026/09/14 (#39439)
SOURCES: body_keyword
ARTIFACT_HINTS: L2.pool.mla_token_kv
FILES: python/sglang/benchmark/one_batch.py (+10/-0); python/sglang/kernels/ops/attention/minimax_sparse/decode/flash_with_topk_idx.py (+7/-1); python/sglang/srt/layers/attention/minimax_sparse_ops/tests/test_flash_with_topk_idx.py (+6/-1); python/sglang/srt/layers/attention/minimax_sparse_ops/tests/test_sparse_gqa.py (+2/-1); python/sglang/srt/layers/attention/xpu_backend.py (+3/-4); python/sglang/srt/layers/quantization/fp8.py (+3/-1); python/sglang/srt/layers/quantization/marlin_utils.py (+2/-2); python/sglang/srt/layers/quantization/mxfp4.py (+11/-14); python/sglang/srt/mem_cache/memory_pool.py (+13/-0); test/registered/moe/test_hash_topk.py (+1/-0); (+1 more)
LABELS: intel, xpu, run-ci, jit-kernel, memory-pool
BODY: ## Motivation ⏎  ⏎   Weekly consolidation of small, straightforward Intel XPU enablement and backend-parity patches. This PR is intentionally draft for final human eligibility review before normal CI consumption. ⏎  ⏎   Base: sgl-project/sglang@869674b3a72ea8de4de63e3e302b1052e6b73392 ⏎  ⏎   ## Included source PRs ⏎  ⏎   - #26594 by @jmunetong — dispatches XPU KV-cache writes to the fused store_cache_xpu kernel. Its kernel implementation and strided-inpu …[truncated]

### L2-6c73368c32  (L2, 2026-09-17, sha 6c73368c329f, PR #37778)
TITLE: [AMD][DSV4] Enable hicache on deepseek-v4 fp8 unified attn (#37778)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py (+33/-21); python/sglang/srt/mem_cache/hicache_storage.py (+4/-0); python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py (+83/-0); python/sglang/srt/mem_cache/memory_pool_host.py (+4/-3); python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py (+2/-0); rust/sglang-radix-tree/src/python_bindings.rs (+4/-0); rust/sglang-radix-tree/src/unified_tree_core.rs (+2/-0); test/registered/unit/mem_cache/test_dsv4_hicache_l2.py (+1/-0); test/registered/unit/mem_cache/test_dsv4_unified_fp8_pool.py (+113/-0)
LABELS: quant, deepseek, hicache, sgl-kernel, run-ci, jit-kernel, run-ci-extra, memory-pool
BODY: ## Motivation ⏎  ⏎  ⏎  ⏎ Based on @amd-danli103's deepseek v4 fp8 unified attn [PR#37413](https://github.com/sgl-project/sglang/pull/37413), enable hicache for long context/large conc situation. ⏎  ⏎ ## Modifications ⏎  ⏎  ⏎  ⏎ Add hicache support to new layout of unified fp8 attn. ⏎ - nope: [512B], fp8 ⏎ - rope: [128B], bf16 ⏎  ⏎ To enable hicache, please use following args. ⏎ ``` ⏎ --enable-hierarchical-cache \ ⏎ --hicache-ratio "${HICACHE_RATIO:-1.5}" \ ⏎ --hic …[truncated]

### L2-1f0c73e9bd  (L2, 2026-09-17, sha 1f0c73e9bd3d, PR #39921)
TITLE: [DSV4] Generalize attention metadata, sparse prefill, and KV pool over compress ratios (#39921)
SOURCES: symbol_pickaxe, body_keyword
ARTIFACT_HINTS: -
FILES: python/sglang/kernels/ops/attention/dsv4/sparse_prefill_kernels.py (+5/-8); python/sglang/srt/disaggregation/decode.py (+8/-15); python/sglang/srt/disaggregation/prefill.py (+5/-25); python/sglang/srt/disaggregation/utils.py (+7/-46); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py (+1/-1); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_common_hooks.py (+4/-2); python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py (+3/-0); python/sglang/srt/layers/attention/deepseek_v4_backend.py (+204/-116); python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (+12/-14); python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py (+148/-116); (+8 more)
LABELS: high priority, deepseek, run-ci, jit-kernel, bypass-fastfail, run-ci-extra, memory-pool
BODY: ## Summary ⏎ - Make the set of compressed-KV ratios data owned by the KV pool (`present_ratios`) and give the shared code one accessor per concept keyed by ratio, instead of hard-coding c4 / c128 in field names, branches and PD payload code ⏎ - No behavior change for V4: every existing path produces the same tensors and indices (see Verification); the new surface is consumed by #38798 ⏎  ⏎ ## Changes ⏎ **Attention metadata (`DSV4AttnMetadata`)** ⏎ - Rename ` …[truncated]
