### L3-58aa1e3d26  (L3, 2026-08-17, sha 58aa1e3d2692, PR #51395)
TITLE: [Bugfix][SM120][MLA] Disable dense prefill for FlashInfer sparse MLA (#51395)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer_sparse
FILES: vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py (+1/-0); tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py (+7/-0)
LABELS: bug, ready, nvidia, verified
BODY: ## Purpose ⏎  ⏎ Fix long-prompt prefill failures in `FlashInferMLASparseSM120Impl`. ⏎  ⏎ The SM120 sparse backend inherits `supports_dense_mha_prefill = True` from `MLAAttentionImpl`, but it implements only the sparse `forward_mqa` path. When a long prompt selects dense or masked MHA prefill, the shared MLA router accesses dense-path state such as `masked_mha_available`, which this implementation does not define. ⏎  ⏎ This change overrides the inherited capa …[truncated]

### L3-0db502c8d8  (L3, 2026-08-17, sha 0db502c8d8a6, PR #50493)
TITLE: [Kimi-K3] support DCP partial prefix cache hit (#50493)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/distributed/test_kimi_linear_context_parallel.py (+111/-0); tests/v1/attention/test_mla_backends.py (+43/-0); tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py (+203/-1); tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py (+23/-21); tests/v1/worker/test_gpu_model_runner_v2.py (+118/-0); vllm/v1/core/kv_cache_coordinator.py (+11/-4); vllm/v1/worker/gpu/block_table.py (+8/-1); vllm/v1/worker/gpu/model_runner.py (+6/-10)
LABELS: ready, ci/build, nvidia, mrv2, kimi, k3
BODY: Builds on the Kimi-K3 DCP support from #50484. ⏎  ⏎ * Enables hash-aligned partial-prefix reuse under DCP for FullAttention plus aligned Mamba while preserving sharded-attention and replicated-Mamba correctness. ⏎ * Fixes MRV2 block-table geometry. MRV2 previously divided every cache group's block-table width by DCP size—correct for attention, but wrong for replicated Mamba. For a 1M context, DCP8, and 16-token Mamba blocks, it allocated 8,192 entries  …[truncated]

### L3-cdb8545a91  (L3, 2026-08-17, sha cdb8545a91be, PR #52539)
TITLE: [Kernel][Perf] Support Qwen head ratios in fused GDN MTP (#52539)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu (+48/-24); tests/kernels/mamba/test_gdn_fused_mtp.py (+42/-0); tests/kernels/test_fused_gdn_post_conv.py (+50/-9); vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+2/-1)
LABELS: qwen
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ #51674 added fused GDN MTP verification for `HV/H=8`. An audit of public Qwen ⏎ GDN configs found production ratios `1/2/3/4/8`, from Qwen3.5 0.8B/2B through ⏎ Qwen3.6 27B/35B and Qwen3.8 2.4T. ⏎  ⏎ This PR: ⏎  ⏎ - makes the value-head ratio a compile-time kernel parameter and dispatches ⏎   exact ratios `1/2/3/4/8`; ⏎ - enables the same ratios in the Qwen GDN guard, with an exact-divisibility ⏎   check; ⏎ - preserves the six original ratio-8 cases and a …[truncated]

### L3-69d3335066  (L3, 2026-08-17, sha 69d333506671, PR #52648)
TITLE: [Bugfix][Quantization] Guard the MXFP8 FlashInfer path on FlashInfer availability (#52648)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/mxfp8/flashinfer.py (+8/-4); vllm/model_executor/layers/quantization/utils/mxfp8_utils.py (+2/-1)
LABELS: bug, ready, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ Two places select or enter the FlashInfer MXFP8 path on device capability alone, ⏎ without checking that FlashInfer is actually importable. ⏎  ⏎ **1. Kernel selection** — `vllm/model_executor/kernels/linear/mxfp8/flashinfer.py` ⏎  ⏎ ```python ⏎ class FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearKernel): ⏎     @classmethod ⏎     def is_supported(cls, compute_capability=None): ⏎         if current_platform.has_device_capability(100): ⏎         …[truncated]

### L3-f27ae25473  (L3, 2026-08-17, sha f27ae25473af, PR #51852)
TITLE: [Bugfix][CPU] Take an attention group's query head count from its layers (#51852)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+8/-8); .buildkite/hardware_tests/cpu.yaml (+2/-0); tests/v1/attention/test_group_head_counts.py (+82/-0)
LABELS: bug, ci/build, cpu, verified
BODY: ## Purpose ⏎ `CPUAttentionMetadataBuilder` sizes the split-KV scratchpad from the model-wide query head count, but the kernel runs with each layer's own count. so models that vary it per layer (e.g. Laguna) index past the end of the scratchpad and ⏎ the decode segfaults or hangs. Attention groups are keyed on `num_heads_q`, so take the count from the group's layers via `get_num_attention_heads_from_layers()`, as `triton_attn` and `flashinfer` alrea …[truncated]

### L3-70afdedc10  (L3, 2026-08-17, sha 70afdedc1081, PR #51855)
TITLE: [K3] support recoverssm for K3 (#51855)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/recoverssm_metadata.py (+26/-0); tests/models/kimi_k3/test_kda.py (+342/-0); tests/models/kimi_k3/test_kda_metadata.py (+145/-4); tests/models/test_registry.py (+3/-1); tests/test_config.py (+47/-0); tests/v1/worker/test_mamba_hybrid_model_state.py (+67/-0); tests/v1/worker/test_mamba_utils.py (+36/-0); vllm/config/cache.py (+4/-3); vllm/config/vllm.py (+33/-8); vllm/model_executor/layers/mamba/abstract.py (+5/-3); (+9 more)
LABELS: performance, ready, mrv2, verified, kimi, k3
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ This PR adds RecoverSSM, a Kimi-K3-specific state recovery path for speculative KDA decoding on NVIDIA Model Runner V2. It is enabled through `--use-replayssm`, is intended for Kimi-K3 serving with DSpark, and supports `mamba_cache_mode=align` prefix caching. ⏎  ⏎ Without RecoverSSM, KDA speculative decoding materializes a full recurrent state for every speculative position. RecoverSSM keeps one checkpoint and compact per-token record …[truncated]

### L3-455edc022b  (L3, 2026-08-17, sha 455edc022b45, PR #42963)
TITLE: [ModelRunnerV2] Support prompt embeds (#42963)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_encoder_runner.py (+90/-5); tests/v1/worker/test_gpu_model_runner.py (+10/-0); tests/v1/worker/test_prompt_embeds_state.py (+149/-0); vllm/config/model.py (+6/-0); vllm/config/vllm.py (+0/-3); vllm/model_executor/models/diffusion_gemma.py (+2/-2); vllm/model_executor/models/longcat_flash_ngram.py (+3/-0); vllm/v1/core/sched/output.py (+8/-0); vllm/v1/request.py (+13/-5); vllm/v1/worker/gpu/mm/encoder_runner.py (+12/-1); (+7 more)
LABELS: ready, v1, mrv2
BODY: ## Purpose ⏎  ⏎ Support prompt embeds for ModelRunnerV2. ⏎  ⏎ ## Test Plan ⏎  ⏎ ```bash ⏎  VLLM_USE_V2_MODEL_RUNNER=1 pytest -sv   tests/basic_correctness/test_basic_correctness.py::test_models   -k "True-uni or True-mp" ⏎ ``` ⏎  ⏎ Before ⏎  ⏎ ```bash ⏎ E       pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig ⏎ E         Value error, VLLM_USE_V2_MODEL_RUNNER does not yet support: prompt embeds [type=value_error, input_value=ArgsK …[truncated]

### L3-c296851a7d  (L3, 2026-08-18, sha c296851a7d17, PR #51924)
TITLE: [MoE] Refine FlashInfer one-sided All2All integration (#51924)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/design/moe_kernel_features.md (+1/-1); tests/distributed/test_mnnvl_alltoall.py (+17/-13); tests/kernels/moe/test_moe_layer.py (+1/-1); vllm/config/parallel.py (+1/-0); vllm/distributed/device_communicators/all2all.py (+4/-8); vllm/model_executor/layers/fused_moe/all2all_utils.py (+52/-22); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+27/-1); vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py (+26/-22)
LABELS: documentation, ready, nvidia, verified
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Refine the FlashInfer NVLink one-sided All2All integration for DeepSeek Blockwise FP8 MoE and sequence parallelism. ⏎  ⏎ - Describe one-sided activation payloads explicitly in bytes for BF16, NVFP4, MXFP8, and DeepSeek Blockwise FP8. ⏎ - Dispatch E4M3 activations with FP32 1x128 scales and feed the received layout directly to the FlashInfer TRT-LLM DeepSeekFp8/BlockMajorK kernel. ⏎ - Validate activation and scale shapes before converting scal …[truncated]

### L3-eab1cff5b0  (L3, 2026-08-18, sha eab1cff5b0ca, PR #52381)
TITLE: Harden DeepSeek V3.2 fused kernel grids (#52381)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_fused_deepseek_v32_norm_rope.py (+73/-0); vllm/models/deepseek_v32/common/kernels.py (+6/-6)
LABELS: ready, deepseek
BODY: ## Purpose ⏎  ⏎ Harden the DeepSeek V3.2 fused Triton kernels against large single-iteration ⏎ token counts. ⏎  ⏎ `fused_norm_rope` and the Triton fallback for `fused_q` placed `num_tokens` on ⏎ CUDA grid-y. A launch with 65,536 tokens therefore exceeded grid-y's 65,535 ⏎ block limit and failed with `Triton Error [CUDA]: invalid argument`. ⏎  ⏎ This change swaps the token and task grid axes so the unbounded token count is ⏎ on grid-x. Token indices are pro …[truncated]

### L3-c296cf8259  (L3, 2026-08-18, sha c296cf8259ed, PR #52174)
TITLE: [Bugfix] Add forward_xpu to XDRotaryEmbedding for HunyuanOCR on XPU (#52174)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rotary_embedding/xdrope.py (+12/-0)
LABELS: bug, intel-gpu, verified
BODY: ## Purpose ⏎  ⏎ `XDRotaryEmbedding` (rope_type `xdrope`, used by `tencent/HunyuanOCR`, arch `HunYuanVLForConditionalGeneration`) overrides `forward_native` and `forward_cuda` but not `forward_xpu`. ⏎  ⏎ `CustomOp.dispatch_forward()` binds the method once at init based on `compilation_config.custom_ops`: ⏎ - Default (Inductor on): `custom_ops=[]` -> `'none'` -> custom op disabled -> `forward_native`. Handles the 2-D `[4, num_tokens]` xdrope positions c …[truncated]

### L3-e8ad2855e7  (L3, 2026-08-18, sha e8ad2855e7f5, PR #52112)
TITLE: [Bugfix][ROCm] Fix a few int4/int8 quantization errors (#52112)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/triton_moe.py (+7/-2); vllm/model_executor/layers/fused_moe/oracle/int_wna16.py (+36/-0); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py (+26/-4); vllm/model_executor/layers/quantization/moe_wna16.py (+0/-1)
LABELS: bug, rocm, ready, quantization
BODY: ## Purpose ⏎ Fix a few quantization bugs on ROCm introduced by #44120 to enable int4/int8 quantized models like cyankiwi/MiniMax-M3-AWQ-INT4, QuantTrio/Qwen3-235B-A22B-GPTQ-Int8.  ⏎ 1. Add asym quantization support for TRITON moe backend; ⏎ 2. Add SWIGLUOAI_UNINTERLEAVE activation for TRITON moe backend; ⏎ 3. Remove wrong assertion (GROUP_SIZE==-1) from MoeWNA16Method; ⏎  ⏎ ## Test Plan ⏎ 1. VLLM_USE_BREAKABLE_CUDAGRAPH=0 vllm serve  cyankiwi/MiniMax-M3 …[truncated]

### L3-01e56caaf2  (L3, 2026-08-18, sha 01e56caaf2b2, PR #52512)
TITLE: [Bugfix][MLA] Do not use Dense MHA for GLM-5.2 (#52512)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+6/-2); tests/model_executor/layers/test_mla_short_prefill_indexer.py (+16/-2); vllm/models/deepseek_v32/attention.py (+2/-0)
LABELS: bug, ready, deepseek
BODY: ## Purpose ⏎  ⏎ Fix incorrect short-prefill dispatch in the NVIDIA DeepSeek-V3.2 attention ⏎ wrapper. ⏎  ⏎ The generic sparse-MLA metadata path may select dense MHA when a prefill's ⏎ sequence length is at most `index_topk`. It then allows the sparse indexer to ⏎ skip top-k scoring. `DeepseekV32Attention`, however, only invokes ⏎ `forward_mqa`; it does not execute the dense-MHA backend. For a pure prefill ⏎ above the backend's decode-like threshold, MQA therefore  …[truncated]

### L3-88b2bff2c6  (L3, 2026-08-18, sha 88b2bff2c63d, PR #51695)
TITLE: [MOE] Standardize and abstract fused shared expert optimization selection (#51695)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/layers/test_fused_shared_expert.py (+700/-0); vllm/model_executor/layers/fused_moe/layer.py (+5/-14); vllm/model_executor/layers/fused_moe/utils.py (+85/-0); vllm/model_executor/layers/quantization/quark/quark.py (+65/-37); vllm/model_executor/layers/quantization/utils/config_utils.py (+155/-0); vllm/model_executor/models/AXK1.py (+22/-10); vllm/model_executor/models/deepseek_mtp.py (+10/-6); vllm/model_executor/models/deepseek_v2.py (+22/-10); vllm/model_executor/models/glm4_moe.py (+20/-12); vllm/model_executor/models/glm4_moe_lite.py (+11/-6); (+10 more)
LABELS: documentation, qwen, deepseek, quantization
BODY: ## Disclosure ⏎  ⏎ AI assistance was used. The changes were reviewed and tested manually. ⏎  ⏎ ## Purpose ⏎  ⏎ Standardize fused shared-expert (FSE) detection so model construction and checkpoint loading use the same quantization-compatible decision **throughout all models implementing FSE**. ⏎  ⏎ This is e.g. useful for: ⏎  ⏎ - `shared_expert` fusion through arbitrary quantization methods/config (e.g. later on with online quantization in https://github.co …[truncated]

### L3-bca7bea240  (L3, 2026-08-18, sha bca7bea24051, PR #52182)
TITLE: Remove VLLM_TEST_FORCE_FP8_MARLIN to replace with linear_backend/moe_backend (#52182)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/compile/passes/test_fusion.py (+4/-0); tests/compile/passes/test_mla_attn_quant_fusion.py (+6/-0); tests/evals/gsm8k/configs/moe-refactor/Llama-4-Scout-Fp8-ModelOpt-marlin.yaml (+1/-3); tests/evals/gsm8k/configs/moe-refactor/Qwen3-30B-A3B-Fp8-AutoFp8-marlin.yaml (+1/-3); tests/evals/gsm8k/configs/moe-refactor/Qwen3-30B-A3B-Fp8-CT-Block-marlin.yaml (+1/-3); tests/evals/gsm8k/configs/moe-refactor/Qwen3-30B-A3B-Fp8-CT-Channel-marlin.yaml (+1/-3); tests/evals/gsm8k/configs/moe-refactor/Qwen3-30B-A3B-NvFp4-CT-marlin.yaml (+1/-3); tests/evals/gsm8k/configs/moe-refactor/Qwen3-30B-A3B-NvFp4-ModelOpt-marlin.yaml (+1/-3); tests/quantization/test_fp8.py (+20/-13); tests/utils.py (+8/-3); (+5 more)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ Remove the test-only `VLLM_TEST_FORCE_FP8_MARLIN` environment variable and use the public `linear_backend` / `moe_backend` configuration to force Marlin. Marlin remains available as the FP8 fallback when earlier compatible kernels cannot be selected. ⏎  ⏎ This also fixes the CI failures exposed by that fallback path: ⏎  ⏎ - use realistic 128 x 128 block-FP8 test shapes instead of synthetic 1 x 1 blocks; ⏎ - give `TestFP8Layer` the partition met …[truncated]

### L3-90984ddbed  (L3, 2026-08-18, sha 90984ddbed27, PR #52797)
TITLE: [CI] Upgrade huggingface-hub to 1.28.0 (#52797)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/common.txt (+1/-1); requirements/test/cpu.txt (+1/-1); requirements/test/cuda.txt (+1/-1); requirements/test/rocm.txt (+1/-1); requirements/test/xpu.txt (+1/-1)
LABELS: ci/build, cpu, nvidia
BODY: Version bump for `huggingface_hub` version from 1.27.0 to 1.28.0

### L3-2687fec6ef  (L3, 2026-08-18, sha 2687fec6ef74, PR #48484)
TITLE: Replicated embedding and norm fusion for DSV3 flat model (#48484)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/kernels/core/test_fused_embed_norm.py (+75/-0); vllm/envs.py (+4/-0); vllm/model_executor/layers/fused_embed_norm.py (+231/-0); vllm/models/deepseek_v32/nvidia/model.py (+32/-6); vllm/models/deepseek_v32/nvidia/mtp.py (+43/-15)
LABELS: ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎  ⏎  ⏎  ⏎ ###  VLLM_REPLICATE_EMBED perf comparison ⏎  ⏎ **Config:** GLM-5.2-NVFP4 / 8×B300 TP8 / MTP(5) ⏎ **Only variable:** `VLLM_REPLICATE_EMBED=1` (replicated embedding + fused gather/norm) ⏎ vs `=0` (vocab-parallel embedding + all-reduce). ⏎ Workload: speed-bench `throughput_16k`, low_entropy, input ≤10240 / output 1536 / `--ignore-eos` / temp 0. ⏎ Numbers below are the mean of 2 independent full repeats. ⏎  ⏎ | conc | ITL ms =1 | ITL ms = …[truncated]

### L3-5f7a20b316  (L3, 2026-08-18, sha 5f7a20b3162e, PR #52046)
TITLE: [nv] add pcp support in dsv3.2 (#52046)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+2/-2); tests/kernels/test_fused_deepseek_v32_norm_rope.py (+90/-6); tests/model_executor/layers/test_mla_short_prefill_indexer.py (+45/-0); vllm/model_executor/layers/sparse_attn_indexer.py (+6/-6); vllm/models/deepseek_v32/attention.py (+103/-20); vllm/models/deepseek_v32/common/kernels.py (+130/-85)
LABELS: new-model, rocm, speculative-decoding, ready, deepseek, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ * seq shard is more efficient than head shard for sparse mla and indexer  ⏎ * wire pcp in dsv3.2 attention w/ fused_q, fused_norm_rope ⏎ * change fused_norm_rope to output kv_c, k_pe, indexer_k instead of direct cache insertion: ⏎   * pcp shards seqlen but keep kv cache replicated like TP, for new cache generated in the forward path it needs to gather kv_c, k_pe, indexer_k before cache insertion ⏎ * disable PCP for mha path ⏎  ⏎ ## Perfor …[truncated]

### L3-5d8a4cf976  (L3, 2026-08-18, sha 5d8a4cf9761e, PR #51647)
TITLE: [ROCm] Pad non-aligned AITER MLA heads (#51647)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.rocm_aiter, L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+79/-31); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+5/-3); tests/kernels/attention/test_rocm_aiter_mla_head_padding.py (+107/-18); tests/v1/attention/test_rocm_aiter_mla_mtp_split.py (+9/-6)
LABELS: rocm, ready, verified
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎ - pad non-16-aligned ROCm AITER MLA query heads to the next supported multiple of 16 and slice padding from the output ⏎ - size dense and sparse persistent metadata for the padded launch shape ⏎ - enable Kimi-K3 TP4's 24 heads/rank to use AITER MLA instead of falling back to Triton MLA ⏎  ⏎ ## Performance ⏎ 8x MI355X, Kimi-K3 TP4/DP2/EP8, 100,000 input tokens (95,911 shared prefix + 4,089 suffix), OSL 1024, one API server, three runs per cohort. …[truncated]

### L3-6066bb3d50  (L3, 2026-08-18, sha 6066bb3d5021, PR #52763)
TITLE: [ROCM][CI] Attention test speedup (#52763)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_attention.py (+2/-0); tests/kernels/attention/test_cache.py (+2/-0); tests/kernels/attention/test_cutlass_mla_decode.py (+2/-0); tests/kernels/attention/test_flashmla.py (+2/-0); tests/kernels/attention/test_merge_attn_states.py (+8/-6); tests/kernels/attention/test_prefix_prefill.py (+2/-0); tests/kernels/attention/test_triton_decode_attention.py (+2/-0); tests/kernels/attention/test_triton_unified_attention.py (+2/-0); tests/kernels/attention/test_triton_unified_attention_diffkv.py (+2/-0)
LABELS: rocm, nvidia
BODY: ## Purpose ⏎  ⏎ Two independent problems make `tests/kernels/attention` far slower than the work it actually performs, and both are addressed here. ⏎  ⏎ **1. `test_merge_attn_states.py` skips its entire matrix on ROCm without reason.** ⏎  ⏎ The test gates all 2592 parametrized cases behind an "only CUDA supports the custom merge_attn_states kernel" check. That is no longer accurate: the kernel source is not CUDA-gated in the build, so it is compiled fo …[truncated]

### L3-f1178f3a06  (L3, 2026-08-18, sha f1178f3a06fa, PR #52836)
TITLE: Revert DSv4 eager workspace reuse (#52836)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/nvidia/flashmla.py (+0/-3); csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu (+6/-24); csrc/libtorch_stable/ops.h (+0/-8); csrc/libtorch_stable/torch_bindings.cpp (+0/-7); tests/kernels/test_compressor_kv_cache.py (+0/-18); tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py (+2/-12); tests/kernels/test_fused_indexer_q_rope_quant.py (+0/-26); vllm/models/deepseek_v4/attention.py (+2/-35); vllm/models/deepseek_v4/common/ops/cache_utils.py (+2/-10); vllm/models/deepseek_v4/common/ops/fused_indexer_q.py (+12/-30); (+5 more)
LABELS: ready, deepseek, nvidia
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 49236 reason=correctness_or_accuracy
BODY: ## Summary ⏎  ⏎ Revert #49236 and restore allocator-backed temporary buffers in the DeepSeek V4 attention input-preparation path. ⏎  ⏎ This removes the model-wide `DeepseekV4EagerScratchPool`, its output-buffer plumbing, and the out-parameter fused op added by #49236. Later changes on `main` are preserved. ⏎  ⏎ ## Why ⏎  ⏎ The model-wide scratch pool can reuse storage across layers and CUDA streams without the caching allocator's stream/event lifetime tracking.  …[truncated]

### L3-ad5e71b276  (L3, 2026-08-18, sha ad5e71b276f6, PR #52293)
TITLE: [ROCm][Perf] Enable fused KDA decode on gfx942 (MI325X) (#52293)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+6/-6); tests/models/kimi_k3/test_amd_kda_decode.py (+5/-5); vllm/models/kimi_k3/amd/ops/kda_decode.py (+4/-3)
LABELS: rocm, ready, ci/build, kimi, k3
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ PR #50654 added the fused Kimi-K3 KDA decode kernel (`csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel_rocm.cu`), but gated it to gfx950 (MI355X) in two places, so it never activates on gfx942 (MI325X): ⏎  ⏎ - **Build gate** — `CMakeLists.txt` filtered the HIP source to `gfx950` only, so the kernel was not compiled and `VLLM_ENABLE_FUSED_KDA_DECODE` (and therefore `torch.ops._C.fused_kda_decode`) was never defined on a gfx942-only …[truncated]

### L3-9842d70145  (L3, 2026-08-18, sha 9842d7014502, PR #48628)
TITLE: [DBO][CI] Increase the coverage of prefill DBO in test_dbo.py (#48628)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/distributed/test_dbo.py (+20/-11)
LABELS: ready, v1
BODY: ## Purpose ⏎ Prefill DBO was broken for a few weeks on main and recently fixed by #46993. Unfortunately, this test didn't catch the original bug in CI despite attempting to exercise the prefill code path. This PR lowers the DBO token threshold for prefills and increases the lm eval concurrency to make it so that more prefills are run with DBO which increases the coverage of this integration test. ⏎  ⏎ ## Test Plan ⏎ This test will run in CI ⏎  ⏎ ## Tes …[truncated]

### L3-8e46accab2  (L3, 2026-08-18, sha 8e46accab22b, PR #52217)
TITLE: [Attention] Vectorize sparse MLA mask loads (#52217)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/sparse_mla_attention.py (+5/-4); vllm/model_executor/layers/attention/sparse_mla_mask.py (+19/-7); tests/v1/attention/test_sparse_mla_mask.py (+22/-0)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ - load four packed sparse-MLA mask words with one aligned 128-bit load ⏎ - retain four-word padding in mask row views so vector loads remain aligned ⏎ - preserve the reserved key-offset word instead of slicing it out of returned mask views ⏎  ⏎ ## Performance ⏎  ⏎ Kernel-only FA4 benchmarks on an NVIDIA B200, using BF16, 8 heads, QK head dimension 192, value head dimension 128, `num_splits=1`, and an all-ones mask: ⏎  ⏎ | Q x KV | Unmasked baseline | …[truncated]

### L3-689be2bcd3  (L3, 2026-08-18, sha 689be2bcd37a, PR #52681)
TITLE: Upgrade Flashinfer version to 0.6.17 (#52681)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); docker/versions.json (+1/-1); requirements/cuda.txt (+2/-2)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎ Upgrade Flashinfer version to 0.6.17 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-ef47a897e2  (L3, 2026-08-18, sha ef47a897e2ad, PR #51875)
TITLE: [Core] Make prefix-cache NONE_HASH deterministic by default (#51875)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/features/kv_offloading_usage.md (+6/-4); docs/features/mooncake_store_connector_usage.md (+5/-3); tests/v1/core/test_kv_cache_utils.py (+63/-5); tests/v1/kv_offload/tiering/p2p/test_manager.py (+35/-16); vllm/v1/core/kv_cache_utils.py (+59/-18); vllm/v1/kv_offload/tiering/fs/manager.py (+8/-7); vllm/v1/kv_offload/tiering/p2p/manager.py (+18/-17); vllm/v1/kv_offload/tiering/p2p/session/client.py (+3/-2); vllm/v1/kv_offload/tiering/p2p/session/protocol.py (+4/-3); vllm/v1/kv_offload/tiering/p2p/session/session.py (+3/-2)
LABELS: documentation, ready
BODY: ## Summary ⏎  ⏎ Prefix-cache block hashes chain from `NONE_HASH`. When `PYTHONHASHSEED` was unset, `NONE_HASH` was seeded with `os.urandom(32)`, so every process produced different block hashes for identical content. This forced distributed KV cache users (shared-FS/object tiers, Mooncake connector) to pin a common `PYTHONHASHSEED` on every node just to get cache hits, and made the P2P tier refuse to start without it. ⏎  ⏎ This makes `NONE_HASH` derive f …[truncated]

### L3-63ff748f65  (L3, 2026-08-19, sha 63ff748f657a, PR #46514)
TITLE: [Attention][MLA] FlashMLA sparse: DCP on the fp8_ds_mla mixed-batch path + MTP (#46514)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_v1_adapter, L3.mla.flashmla_sparse
FILES: vllm/v1/attention/backends/mla/flashmla.py (+12/-0); vllm/v1/attention/backends/mla/flashmla_sparse.py (+129/-34); tests/v1/attention/test_sparse_mla_backends.py (+154/-0)
LABELS: documentation, ready, v1, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: DCP support for the `FLASHMLA_SPARSE` backend on the `fp8_ds_mla` KV path, including MTP spec decode under full cudagraphs. `fp8_ds_mla` is what the backend selector picks on Hopper for DSA models with small per-rank head counts (GLM-5.2 / DeepSeek-V3.2 at TP4/DCP4), and this path previously refused DCP outright. ⏎  ⏎ Earlier revisions stacked on #46076 and carried the sparse-indexer DCP machinery (global top-k merge, `--dcp-sparse-indexer-mode`); al …[truncated]

### L3-b09bd69b5b  (L3, 2026-08-19, sha b09bd69b5bf1, PR #52861)
TITLE: [Model][NVIDIA] Route DSA models to the CUDA non-compiled path (#52861)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/compile/fusions_e2e/conftest.py (+1/-20); tests/compile/fusions_e2e/models.py (+0/-24); tests/compile/fusions_e2e/test_tp1_quant.py (+2/-4); tests/compile/fusions_e2e/test_tp2_ar_rms.py (+2/-4); tests/compile/h100/test_startup.py (+0/-13); tests/kernels/test_fused_deepseek_v32_norm_rope.py (+3/-4); tests/models/registry.py (+5/-0); tests/test_config.py (+140/-4); vllm/config/speculative.py (+7/-5); vllm/config/vllm.py (+67/-39); (+6 more)
LABELS: new-model, speculative-decoding, ready, deepseek, nvidia
BODY: ## Purpose ⏎  ⏎ Route `DeepseekV32ForCausalLM`, `GlmMoeDsaForCausalLM`, and their MTP draft ⏎ model through the CUDA `vllm.models.deepseek_v32` implementation on every ⏎ NVIDIA GPU, while retaining the existing defaults elsewhere. ⏎  ⏎ The optimized NVIDIA classes are currently unreachable from the model registry. ⏎ That leaves DeepSeek V3.2 and GLM-5.2 on the generic runner/graph path and ⏎ prevents their MTP draft model from using the matching implementation. ⏎  …[truncated]

### L3-db92053e97  (L3, 2026-08-19, sha db92053e97b5, PR #52041)
TITLE: [Core] Skip broadcasting mm tensor data to workers for prefix-cache-covered items (#52041)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_output.py (+87/-0); vllm/multimodal/utils.py (+31/-0); vllm/v1/core/sched/output.py (+7/-1); vllm/v1/core/sched/scheduler.py (+5/-1)
LABELS: ready, multi-modality, verified
ISSUES: #52040 [Performance]: MM input tensors are re-broadcast to all TP workers on every request, even when the prefix cache fully covers the image tokens (~19 ms per unique image)
BODY: ## Purpose ⏎  ⏎ Fixes #52040. ⏎  ⏎ ## Background: why this gap exists and who it affects ⏎  ⏎ The EngineCore->TP-worker broadcast was designed when request inputs were token IDs (kilobytes), so shipping them unconditionally was both correct and free. Multimodal grafted large tensors onto the same path, but for the classic workload (a fresh image per chat request) the workers genuinely need every tensor, so unconditional shipping remained necessary. What chan …[truncated]

### L3-be06873198  (L3, 2026-08-19, sha be06873198cd, PR #52002)
TITLE: [Bugfix] compressed-tensors: restore int8 grouped WNA16 MoE support (#52002)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py (+0/-1)
LABELS: bug, ready, quantization
BODY: ## Purpose ⏎  ⏎ Restores the fix from #47154, which was lost in #44570. ⏎  ⏎ #47154 dropped the `group_size == -1` assert from the int8 branch of ⏎ `CompressedTensorsWNA16MarlinMoEMethod` so that checkpoints with group-quantized int8 ⏎ experts could load. #44570 then merged that class into ⏎ `CompressedTensorsWNA16MoEMethod`, deleting the fixed file and keeping the other ⏎ class's version of the branch, which still carried the assert. ⏎  ⏎ Net effect: any  …[truncated]

### L3-160f7f0840  (L3, 2026-08-19, sha 160f7f0840e1, PR #41100)
TITLE: [ROCm][CI] Extended Fused MoE and FP8 MoE test support (#41100)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+58/-1); .buildkite/test_areas/kernels.yaml (+53/-0); tests/kernels/moe/test_modular_oai_triton_moe.py (+27/-2); tests/kernels/moe/test_moe.py (+41/-0); tests/kernels/moe/test_moe_layer.py (+106/-26); tests/kernels/moe/utils.py (+26/-1); tests/quantization/test_fp8.py (+13/-0); vllm/model_executor/layers/fused_moe/fused_moe.py (+6/-0); vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py (+1/-1)
LABELS: rocm, ready, ci/build, quantization
BODY: This PR makes the fused MoE layer test matrix usable on ROCm/MI355 by fixing the real ModelOpt FP8/FP4 failures it exposes and by making distributed subcase failures visible to pytest. ⏎  ⏎ Key changes: ⏎  ⏎ - Propagate fused MoE distributed subcase failures back to the parent pytest process instead of allowing child-rank failures to print as failed subcases while the parent test reports `PASSED`. ⏎ - Allow `modelopt_fp4` MoE test configs on gfx950 al …[truncated]

### L3-f485081e8b  (L3, 2026-08-19, sha f485081e8b76, PR #49532)
TITLE: [XPU] Support EC connector KV Offloading on XPU (#49532)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/intel_jobs/misc_intel.yaml (+4/-2); tests/v1/ec_connector/unit/cpu/worker/test_worker.py (+42/-37); tests/v1/ec_connector/unit/test_ec_cpu_connector.py (+7/-4); vllm/distributed/ec_transfer/ec_connector/cpu/worker/__init__.py (+9/-7); vllm/distributed/ec_transfer/ec_connector/cpu/worker/descriptor_buffers.py (+31/-8)
LABELS: intel-gpu, ci/build, v1, cpu
BODY: ## Purpose ⏎  ⏎ PR #47423 added `ECCPUConnector` for encoder-cache (EC) offloading to host memory, but the swap path assumed CUDA/ROCm-style raw pointers and was never exercised on XPU. This PR adds XPU support so the EC connector's offload path works correctly on Intel GPUs, and extends CI coverage accordingly. ⏎  ⏎ Three issues were fixed: ⏎  ⏎ ### 1. Descriptor pointer dtype on XPU ⏎  ⏎ `swap_blocks_batch` takes `int64` pointer arrays on CUDA/ROCm but …[truncated]

### L3-e4d61d0d22  (L3, 2026-08-19, sha e4d61d0d222a, PR #52616)
TITLE: [CPU] Add AMX-only high-performance MLA backend for DeepSeek V2/V3/R1 (#52616)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.dispatch.registry
FILES: csrc/cpu/sgl-kernels/flash_attn.h (+250/-0); vllm/_custom_ops.py (+103/-0); vllm/model_executor/layers/attention/mla_attention.py (+41/-0); vllm/platforms/cpu.py (+31/-5); vllm/v1/attention/backends/mla/amx_mla.py (+405/-0); vllm/v1/attention/backends/mla/prefill/cpu_native.py (+61/-0); vllm/v1/attention/backends/mla/prefill/registry.py (+3/-0); vllm/v1/attention/backends/mla/prefill/selector.py (+2/-0); vllm/v1/attention/backends/registry.py (+1/-0); .buildkite/hardware_tests/cpu.yaml (+4/-0); (+10 more)
LABELS: ci/build, deepseek, cpu
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Add `AMXMLABackend`/`AMXMLAImpl`, a high-performance MLA attention backend for ⏎ CPU, built on AMX decode/extend/bmm/qkv-proj kernels vendored under ⏎ `csrc/cpu/sgl-kernels/`. It plugs into the same `MLACommonBackend`/ ⏎ `MLACommonImpl` abstraction every other concrete MLA backend uses. ⏎  ⏎ This is a separate, opt-in-by-capability backend alongside the existing ⏎ reference `CPUMLABackend` (#49453): that one targets any CPU/dtype at ⏎ block_size=16 …[truncated]

### L3-f936a267f9  (L3, 2026-08-19, sha f936a267f9a7, PR #48109)
TITLE: [Bugfix][XPU] Fix Mamba state pointer overflow (#48109)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_mamba_utils.py (+89/-0); vllm/v1/worker/mamba_utils.py (+9/-2)
LABELS: bug, intel-gpu, v1, verified
ISSUES: #48059 [Bug][XPU] Mamba align-mode prefix caching crashes: "Overflow when unpacking long long" storing state.data_ptr()
BODY: Fixes #48059 ⏎  ⏎ ## Purpose ⏎  ⏎ XPU USM device pointers can exceed the positive signed int64 range. The Mamba ⏎ align-mode fused copy path stores state tensor and block-table `data_ptr()` ⏎ values in `torch.int64` metadata buffers before passing them to Triton kernels, ⏎ so assigning a high-bit XPU pointer directly can raise: ⏎  ⏎ ```text ⏎ ValueError: Overflow when unpacking long long ⏎ ``` ⏎  ⏎ This preserves the 64-bit pointer pattern by reinterpreting unsigned point …[truncated]

### L3-755492e37d  (L3, 2026-08-19, sha 755492e37d7d, PR #52987)
TITLE: Revert "[Kernel] Gemma-4 FA4 FP8 Kernel" (#52987)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, corpus:confirmed-reverts
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_build, L3.flash_attn.fa4_cutedsl, L3.flash_attn.fa_utils, L3.dispatch.abstract_interface
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/model_executor/layers/attention/attention.py (+3/-11); vllm/platforms/interface.py (+5/-3); vllm/v1/attention/backend.py (+0/-14); vllm/v1/attention/backends/fa_utils.py (+3/-3); vllm/v1/attention/backends/flash_attn.py (+2/-57); vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py (+12/-43); vllm/vllm_flash_attn/flash_attn_interface.py (+3/-20)
LABELS: ci/build, mrv2
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 48666 reason=unstated
BODY: Reverts vllm-project/vllm#48666

### L3-525b7bbb3a  (L3, 2026-08-19, sha 525b7bbb3a74, PR #49688)
TITLE: [Bugfix][CPU] Enable C++ causal_conv1d GDN path and float32 SSM cache on non-AMX AVX-512BF16 CPUs (#49688)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/mamba/cpu/test_cpu_gdn_ops.py (+270/-4); vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py (+8/-6); vllm/model_executor/layers/utils.py (+5/-2); vllm/platforms/cpu.py (+7/-5)
LABELS: bug, cpu
ISSUES: #49640 [Bug]: [CPU] GDN attention falls back to slow torch conv1d on non-AMX AVX-512BF16 CPUs
BODY: ## Purpose ⏎  ⏎ CPU GDN attention (Qwen3.5) selects its causal-conv1d implementation via `torch.cpu._is_amx_tile_supported()`: AMX CPUs (Intel GNR) use the C++ kernels (`causal_conv1d_fwd_cpu` / `causal_conv1d_update_cpu`), everything else falls back to `causal_conv1d_fn_cpu` / `causal_conv1d_update_torch`, a pure-PyTorch per-sequence loop. ⏎  ⏎ The C++ conv kernels (`csrc/cpu/sgl-kernels/conv.cpp`) do not use AMX tiles but `is_amx` gate blocks it fr …[truncated]

### L3-c676232313  (L3, 2026-08-19, sha c676232313fc, PR #52704)
TITLE: [Bugfix][Quantization] Fix OCP MX MoE emulation silently skipping mxfp6 activation QDQ (#52704)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_ocp_mx_moe.py (+148/-0); vllm/model_executor/layers/fused_moe/experts/ocp_mx_emulation_moe.py (+47/-23)
LABELS: bug, quantization
BODY: ## Purpose ⏎  ⏎ `OCP_MXQuantizationEmulationTritonExperts` overrides the `quant_dtype` property with a scheme→dtype map built in `__init__`. For the four mxfp6-activation schemes that map produced `"mxfp6"` — not a key `moe_kernel_quantize_input` dispatches on, which knows `"mxfp6_e3m2"` and `"mxfp6_e2m3"`. Those schemes therefore fall through to the dispatcher's closing `else: return A, A_scale` and run with **unquantized activations**. Nothing ra …[truncated]

### L3-541c6d64c1  (L3, 2026-08-19, sha 541c6d64c19b, PR #52966)
TITLE: [Bugfix][Quantization] Support CT block FP8 with Marlin (#52966)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/quantization/test_compressed_tensors.py (+30/-13); vllm/model_executor/kernels/linear/scaled_mm/marlin.py (+11/-4)
LABELS: bug, ready, cpu, quantization
BODY: ## Summary ⏎  ⏎ - allow the FP8 Marlin block kernel to consume either `weight_scale_inv` or `weight_scale` ⏎ - preserve the existing scale attribute through Marlin post-processing and execution ⏎ - exercise the real Compressed Tensors block-FP8 model with both automatic and explicitly forced Marlin linear selection ⏎  ⏎ The H100 MoE evaluation configuration remains unchanged and continues to force `--linear-backend marlin --moe-backend marlin`. ⏎  ⏎ ## Why ⏎  ⏎ PR # …[truncated]

### L3-58e5ee0158  (L3, 2026-08-19, sha 58e5ee0158b6, PR #52839)
TITLE: [refactor] consolidate cp attn ops (#52839)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/mla_attention.py (+5/-5); vllm/model_executor/layers/attention/sparse_mla_attention.py (+1/-1); vllm/v1/attention/backend.py (+1/-1); vllm/v1/attention/backends/flash_attn.py (+4/-2); vllm/v1/attention/backends/flashinfer.py (+4/-2); vllm/v1/attention/ops/common.py (+0/-299); vllm/v1/attention/ops/cp_common.py (+154/-0); vllm/v1/attention/ops/dcp.py (+1368/-0); vllm/v1/attention/ops/dcp_alltoall.py (+0/-470); vllm/v1/attention/ops/dcp_utils.py (+0/-740); (+8 more)
LABELS: ready, deepseek, nvidia, kimi, k3
BODY: ## Purpose ⏎  ⏎ Cosmetic refactor of cp related attention ops. ⏎  ⏎ * consolidates dcp attn ops across `ops/common.py`, `ops/dcp_utils.py` and `ops/dcp_alltoall.py` into `ops/dcp.py` ⏎ * moves pcp attn ops from `model_executor/layers/attention/pcp.py` to `vllm/v1/attention/ops/pcp.py` ⏎ * extracts symmem related capability into `ops/cp_common.py`, will be used by prefill pcp fused ops in later diffs ⏎  ⏎ End state in attention/ops/ ⏎ - `cp_common.py`: sym …[truncated]

### L3-6a962071bd  (L3, 2026-08-19, sha 6a962071bdad, PR #52998)
TITLE: [Distributed] Enable FlashInfer all-reduce by default (#52998)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/distributed/device_communicators/cuda_communicator.py (+4/-1); vllm/envs.py (+2/-2)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Enable the existing FlashInfer all-reduce backend by default for tensor-parallel ⏎ CUDA groups. The existing platform, world-size, tensor-shape, and workspace ⏎ guards remain in place, so unsupported calls continue through the normal ⏎ all-reduce fallback chain. ⏎  ⏎ `VLLM_ALLREDUCE_USE_FLASHINFER=0` remains available as an explicit opt-out. ⏎ FlashInfer all-reduce is always excluded when `VLLM_BATCH_INVARIANT=1` because ⏎ it does not provide the f …[truncated]

### L3-583a00257d  (L3, 2026-08-20, sha 583a00257d4c, PR #51632)
TITLE: [ROCm] [Bugfix] Fix Triton fused shared expert alignment (#51632)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_moe.py (+44/-0); vllm/model_executor/layers/fused_moe/experts/triton_moe.py (+6/-2)
LABELS: bug, rocm, ready
BODY: ## Motivation ⏎  ⏎ MoE models using fused shared experts with the Triton backend can produce corrupted outputs and suffer from severe accuracy loss. Fused shared experts append extra expert IDs and weight rows, while `global_num_experts` continues to represent only the routed experts. Triton uses this smaller count during token alignment, causing the appended shared expert IDs to be treated as invalid. Since the separate shared expert path is disab …[truncated]

### L3-30e2394c83  (L3, 2026-08-20, sha 30e2394c83af, PR #51703)
TITLE: [Bugfix] Record non-ImportError attention backend probe failures instead of crashing engine init (#51703)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.platform.cuda_selection
FILES: vllm/platforms/cuda.py (+10/-4); tests/v1/attention/test_cuda_backend_probe_errors.py (+104/-0)
LABELS: bug, ready, nvidia
ISSUES: #51658 [Bug]: attention backend probe in cuda.py catches only ImportError; non-ImportError side effects (e.g. cache PermissionError, CUDA runtime   mismatch) crash engine init instead of being recorded as unavailable
BODY: ## Purpose ⏎  ⏎ Fixes #51658. ⏎  ⏎ The backend probe in `CudaPlatform.get_valid_backends` catches only ImportError. The loop exists to mark unavailable backends and keep going, but anything else raised while importing or validating a backend, say a PermissionError from a read-only flashinfer cache dir or an OSError from a broken driver install, escapes and takes down engine init even when a usable backend is next in the priority list. The recorded reason …[truncated]

### L3-d626108b18  (L3, 2026-08-20, sha d626108b1841, PR #52737)
TITLE: [ROCm][Perf] Fuse DeepSeek-V4 mHC post/pre and RMSNorm with AITER (#52737)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/_aiter_ops.py (+96/-0); vllm/model_executor/kernels/mhc/aiter.py (+102/-0); vllm/model_executor/layers/mhc.py (+28/-2); vllm/models/deepseek_v4/amd/dspark.py (+2/-2); vllm/models/deepseek_v4/amd/model.py (+44/-6)
LABELS: rocm, ready, deepseek, DSv4, dflash
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ This PR improves the ROCm DeepSeek-V4 mHC path by wiring the existing AITER fused mHC operators into vLLM. The ROCm path can now replace the previous sequence of separate mHC/RMSNorm kernels with the fused AITER flow: ⏎  ⏎ - `mhc_post` + `mhc_pre_gemm_sqrsum` -> `mhc_fused_post_pre_gemm_sqrsum` ⏎ - `mhc_pre_big_fuse` + `add_rmsnorm_quant` -> `mhc_pre_big_fuse_rmsnorm` ⏎  ⏎ For unsupported fused-RMSNorm hidden sizes, the path still uses f …[truncated]

### L3-c8de519917  (L3, 2026-08-20, sha c8de519917ce, PR #50400)
TITLE: [Kernel][Kimi] fused vision q/k roper kernel (#50400)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/kimi_k25_vit.py (+14/-26)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ By using Triton to fuse Q/K RoPE operations, kernel efficiency is improved by 5x compared to the existing PyTorch implementation. ⏎  ⏎ ``` ⏎ # .venv/bin/python \ benchmark_kimi_vision_rope.py \ ⏎   --dtype bfloat16 \ ⏎   --cudagraph ⏎ tokens heads head_dim dtype mode native_us fused_us speedup ⏎    256     3      128 bfloat16 cudagraph    13.409    1.570   8.542x ⏎    256     6      128 bfloat16 cudagraph    13.775    1.727   7.976x ⏎    256 …[truncated]

### L3-bd8865a299  (L3, 2026-08-20, sha bd8865a299c4, PR #52204)
TITLE: [Kernel] Add FlashInfer TRTLLM MXFP8 linear backend (#52204)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+33/-0); docs/features/quantization/modelopt.md (+5/-0); tests/kernels/quantization/test_flashinfer_mxfp8_trtllm.py (+123/-0); vllm/model_executor/kernels/linear/__init__.py (+4/-0); vllm/model_executor/kernels/linear/mxfp8/flashinfer.py (+100/-0)
LABELS: documentation, ready, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ FlashInfer exposes a TensorRT-LLM backend for dense MXFP8 GEMM, but vLLM's ⏎ `flashinfer_trtllm` linear selector currently covers NVFP4 only. This change ⏎ adds `FlashInferTrtllmMxfp8LinearKernel`, which can be selected with: ⏎  ⏎ ```text ⏎ --linear-backend flashinfer_trtllm ⏎ ``` ⏎  ⏎  ⏎ The kernel prepares TensorRT-LLM weight and scale layouts once after weight ⏎ loading. At runtime it quantizes BF16 activations with the 8x4 scale layout, ⏎  …[truncated]

### L3-01af92e175  (L3, 2026-08-20, sha 01af92e17540, PR #49811)
TITLE: [Feature][Model Runner V2] Support extract_hidden_states speculation (#49811)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/test_config.py (+15/-0); tests/v1/worker/test_gpu_extract_hidden_states_speculator.py (+132/-0); vllm/config/vllm.py (+1/-0); vllm/v1/worker/gpu/model_runner.py (+6/-1); vllm/v1/worker/gpu/spec_decode/__init__.py (+7/-1); vllm/v1/worker/gpu/spec_decode/extract_hidden_states.py (+153/-0)
LABELS: speculative-decoding, ready, v1, mrv2
ISSUES: #49562 [Feature] Support `extract_hidden_states` on Model Runner V2
BODY: ## Purpose ⏎  ⏎ Fixes #49562. ⏎  ⏎ `extract_hidden_states` is supported by the existing model runner, but Model ⏎ Runner V2 currently rejects the configuration and has no speculator for ⏎ forwarding the sampled token while caching the target model's auxiliary hidden ⏎ states. ⏎  ⏎ This PR: ⏎  ⏎ - allows the `extract_hidden_states` speculative method with Model Runner V2; ⏎ - requests auxiliary hidden-state outputs from the target model; ⏎ - adds a Model Runner V2 `Extract …[truncated]

### L3-4f6885fffc  (L3, 2026-08-20, sha 4f6885fffc93, PR #53040)
TITLE: [DSV4][Kernel] Fuse shared experts into MegaMoE (#53040)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/models/test_deepseek_v4_mega_moe.py (+232/-0); vllm/envs.py (+7/-0); vllm/models/deepseek_v4/nvidia/model.py (+274/-42); vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py (+51/-0); vllm/models/kimi_k3/nvidia/model.py (+5/-2)
LABELS: ready, deepseek, DSv4, kimi, k3
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ Part of https://github.com/vllm-project/vllm/issues/45861. ⏎  ⏎ This PR fuses DeepSeek V4's replicated FP8 shared expert into DeepGEMM's persistent SM100 MegaMoE kernel in the NVIDIA-specific model path. ⏎  ⏎ Before this change, every MoE layer launched the routed FP4 MegaMoE kernel and then ran the shared FP8 gate/up and down projections serially, followed by a separate add. The new path lets the native SM100 kernel schedule shared L1, …[truncated]

### L3-cd5035379c  (L3, 2026-08-20, sha cd5035379c4c, PR #51585)
TITLE: [ROCm] [Bugfix] Preserve CPU query offsets during capture (#51585)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.rocm.v1_rocm_attn
FILES: vllm/v1/attention/backends/rocm_attn.py (+2/-4)
LABELS: bug, rocm, ready
BODY: ## Motivation ⏎  ⏎ On ROCm, full CUDA graph capture can fail during server startup for mixed-attention models when a later metadata builder reads a query length of zero. The root cause is that `RocmAttentionMetadataBuilder` clears both device and CPU query offsets to prevent an invalid memory access in the prefix prefill kernel. The device offsets must be zeroed for safe graph capture, but the CPU offsets are not used by that kernel and remain shar …[truncated]

### L3-cb09dd7488  (L3, 2026-08-20, sha cb09dd7488d2, PR #52925)
TITLE: [Core][Multimodal] Skip redundant placeholder scan when token match succeeds (#52925)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/multimodal/test_processing.py (+356/-150); vllm/model_executor/models/gemma3_mm.py (+34/-0); vllm/model_executor/models/gemma3n_mm.py (+34/-0); vllm/multimodal/processing/processor.py (+115/-25)
LABELS: ready, multi-modality, verified
ISSUES: #52924 [Bug]: `find_mm_placeholders` may return an incorrect position for `PromptInsertion`
BODY: ## Purpose ⏎  ⏎ CLOSE #52924 ⏎  ⏎ `_apply_prompt_updates` first calls `_apply_token_matches`. If any token match fails, it falls back to `_apply_text_matches`. The current code always calls `_find_mm_placeholders` after matching, even when all token matches succeed. This call scans for placeholders again through the following path: `_find_mm_placeholders` → `find_mm_placeholders` → `_iter_placeholders` → `prompt[start:end] == content_tokens`. ⏎  ⏎ The  …[truncated]

### L3-7cfb97e337  (L3, 2026-08-20, sha 7cfb97e33791, PR #48915)
TITLE: [Frontend][Core][Spec Decode] Per-request acceptance stats in OpenAI API responses (#48915)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/features/per_request_metrics.md (+9/-0); docs/features/speculative_decoding/README.md (+1/-0); docs/features/speculative_decoding/acceptance_metrics.md (+100/-0); rust/src/engine-core-client/src/protocol/output.rs (+6/-0); rust/src/engine-core-client/src/tests/client.rs (+1/-0); tests/entrypoints/openai/completion/test_completion_error.py (+109/-1); tests/test_config.py (+14/-0); tests/v1/core/test_scheduler.py (+113/-9); tests/v1/core/utils.py (+5/-0); tests/v1/spec_decode/test_request_acceptance.py (+150/-0); (+15 more)
LABELS: documentation, frontend, speculative-decoding, ready, v1, verified, rust, scheduler
BODY: ## Purpose ⏎  ⏎ Speculative decoding guesses a few tokens ahead, then checks them. Some guesses get accepted, some don't. How many get accepted tells you how well spec decode is working. ⏎  ⏎ Today you can only see this as a server-wide average on `/metrics`. You can't tell how any single request did. Tools like AIPerf want that per-request number, and right now they'd have to wrap the engine to get it. ⏎  ⏎ This PR reports it in the response. It's off …[truncated]

### L3-bfb6c13499  (L3, 2026-08-20, sha bfb6c134997a, PR #52989)
TITLE: [Bugfix][MoE] Tune FlashInfer experts to scheduler token limit (#52989)
SOURCES: subject_keyword, corpus:confirmed-reverts(reverted), corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer_moe.py (+2/-0); vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py (+2/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py (+1/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_lora_moe.py (+2/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_mxint4_moe.py (+2/-0); vllm/model_executor/layers/quantization/utils/flashinfer_mxint4_moe.py (+3/-1)
LABELS: bug, nvidia, quantization
DEEP_STUDY: deep-study: this PR was reverted by PR 53186 (confirmed_revert, reason=crash_or_hang) || deep-study performance PR (kernel_tuning_config)
BODY: ## Summary ⏎  ⏎ - pass vLLM's parallel-aware maximum token bucket to FlashInfer CUTLASS experts ⏎ - fix the same omitted or hardcoded bound in modular BF16, BF16 LoRA, and MXINT4 expert paths ⏎ - cover the CUTLASS forwarding behavior with a focused regression assertion ⏎  ⏎ ## Root cause ⏎  ⏎ These expert paths let FlashInfer use its default `tune_max_num_tokens=8192`, even when vLLM's effective MoE input bound was larger. As a result, startup autotuning stopped …[truncated]

### L3-91a893de64  (L3, 2026-08-20, sha 91a893de6472, PR #52809)
TITLE: [Bugfix][Spec Decode] Scope DSpark backend inheritance to DeepSeek V4 (#52809)
SOURCES: symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/spec_decode/dspark/utils.py (+34/-10)
LABELS: bug, speculative-decoding, ready, deepseek, mrv2, DSv4, dflash
BODY: ## Purpose ⏎  ⏎ #52288 made an unspecified DSpark draft backend inherit the target's backend. ⏎ That fallback is required for DeepSeek V4 because its target and draft layers ⏎ share a KV-cache layout, but it is not valid for every DSpark architecture. ⏎  ⏎ For example, a Kimi MLA target can use a Qwen3/SWA DSpark draft. Inheriting ⏎ `FLASHINFER_MLA` makes draft initialization fail because that backend does not ⏎ support sliding-window attention. ⏎  ⏎ Keep the explic …[truncated]

### L3-4b7cb949a9  (L3, 2026-08-20, sha 4b7cb949a906, PR #51777)
TITLE: [Docker] Update to nixl-1.3.2 (#51777)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.xpu (+8/-3)
LABELS: intel-gpu, ci/build, kv-connector, nvidia, verified
BODY: ## Purpose ⏎  ⏎ The KV-connectors layer installs `nixl` from requirements/kv_connectors.txt, then force-reinstalls the CUDA-matched backend wheel so the correct nixl_ep_cpp.so is present. That second install passes `--no-deps` and no version, which bypasses the `nixl` meta package's `nixl-cu*==<version>` constraint and resolves whatever nixl-cu${CUDA_MAJOR} is newest on PyPI. ⏎  ⏎ The image therefore silently ships a different NIXL than it pins: with …[truncated]

### L3-b389ac2946  (L3, 2026-08-20, sha b389ac29465b, PR #52816)
TITLE: [Spec Decode] DFlash2: local convolution + candidate selector (#52816)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/registry.py (+8/-0); tests/test_config.py (+24/-0); tests/v1/spec_decode/test_dflash2.py (+118/-0); tests/v1/spec_decode/test_dflash_causality.py (+25/-1); vllm/config/vllm.py (+17/-0); vllm/model_executor/layers/logits_processor.py (+81/-0); vllm/model_executor/models/qwen3_dflash.py (+11/-4); vllm/model_executor/models/qwen3_dflash2.py (+290/-0); vllm/model_executor/models/registry.py (+1/-0); vllm/v1/worker/gpu/sample/gumbel.py (+52/-34); (+4 more)
LABELS: new-model, speculative-decoding, ready, needs-rebase, qwen, mrv2, dflash
BODY: Two additions to the DFlash drafter, carried by a separate architecture: ⏎ a checkpoint declaring `DFlash2DraftModel` gets them, and every existing ⏎ `DFlashDraftModel` checkpoint resolves to the class it resolves to today, ⏎ untouched by this PR. ⏎  ⏎ **Grouped dynamic depthwise convolution** inside each block, so a proposal ⏎ position can see the ones before it without another backbone pass. ⏎ `out[i,c] = Σ_t (base[t,c] + δ[i,t,g(c)]) · x[i−t,c]`, taps zero  …[truncated]

### L3-2f41c894e3  (L3, 2026-08-20, sha 2f41c894e320, PR #51866)
TITLE: Fix seed loss when batch contains unseeded requests (#51866)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/sample/ops/topk_topp_sampler.py (+1/-1)
LABELS: ready
ISSUES: #51226 [Bug]: seed is ignored on CPU when another request in the batch has no seed
BODY: Fix seed loss when batch has unseeded requests. ⏎  ⏎  ⏎  ⏎  ⏎ ## Purpose ⏎  ⏎ Fixes #51226. ⏎  ⏎ On CPU, `TopKTopPSampler.forward_cpu` decided whether to take the unseeded fast path by checking `len(generators) != logits.shape[0]`. Since `generators` only holds an entry per *seeded* request, this check evaluates true (mismatch) whenever even one request in the batch is unseeded — so the whole batch fell through to the fast path and every per-request `torc …[truncated]

### L3-5df31ea52d  (L3, 2026-08-20, sha 5df31ea52d7d, PR #52795)
TITLE: [Spec Decode] Enable adaptive verification on DSv4 + sm90 (#52795)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+28/-4)
LABELS: ready, DSv4, dflash
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Enable adaptive verification for the DeepSeek V4 indexer on Hopper (SM90). ⏎  ⏎ The native variable-length DeepGEMM paged-MQA path remains restricted to SM100. On SM90, this change instead uses the existing flattened decode path and builds the single-token rows from the device-side decode lengths. This is necessary because adaptive verification can trim each request to a different length while the CPU metadata contains only a uniform plac …[truncated]

### L3-2785c72a14  (L3, 2026-08-21, sha 2785c72a1497, PR #53053)
TITLE: [Kimi-K3] Extend GEMM-RS to GEMM-AR (#53053)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: .buildkite/test_areas/distributed.yaml (+11/-2); benchmarks/kernels/benchmark_kimi_k3_gemm_rs_ar.py (+114/-78); tests/kernels/test_kimi_k3_gemm_rs_ar.py (+114/-47); vllm/envs.py (+4/-0); vllm/model_executor/warmup/kernel_warmup.py (+15/-0); vllm/models/kimi_k3/nvidia/kda.py (+13/-13); vllm/models/kimi_k3/nvidia/mla.py (+13/-13); vllm/models/kimi_k3/nvidia/model.py (+60/-36); vllm/models/kimi_k3/nvidia/ops/cute_dsl/gemm_rs_ar.py (+137/-43)
LABELS: performance, ready, needs-rebase, ci/build, kimi, k3
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Replaces #52687 ⏎  ⏎ Extends GEMM-RS, introduced by #52079, to GEMM-AR by replacing unicast store with `multimem.st`. Compared to #52687, this is better in the following way: ⏎ - Any value of M is supported -> no padding/input copy is required ⏎ - Flag signalling/synchronization is fully done inside the kernel -> no flag reset + entry barrier is needed ⏎  ⏎ However, there is still a limitation. ⏎ - Currently `multimem.st` places the output …[truncated]

### L3-f8e0602713  (L3, 2026-08-21, sha f8e060271381, PR #53132)
TITLE: Support kimi k3 nvfp4 checkpoint (#53132)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+11/-2); tests/kernels/moe/test_trtllm_nvfp4_moe.py (+21/-2); tests/quantization/test_modelopt.py (+59/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+18/-16); vllm/model_executor/layers/quantization/modelopt.py (+31/-12); vllm/models/kimi_k3/nvidia/kda.py (+11/-3)
LABELS: ready, deepseek, nvidia, quantization, mrv2, kimi, k3
BODY: ## Purpose ⏎ This PR supports running kimi k3 nvfp4 checkpoint `nvidia/Kimi-K3-NVFP4` ⏎  ⏎ ## Test Plan ⏎ - Kimi K3 GSM8k on 8 x B300 ⏎  ⏎ ## Test Result ⏎ TP8: ⏎ ``` ⏎ vllm serve nvidia/Kimi-K3-NVFP4 \ ⏎   -tp 8 \ ⏎   --load-format fastsafetensors \ ⏎   --no-enable-flashinfer-autotune \ ⏎   --trust-remote-code \ ⏎   --language-model-only \ ⏎   --attention-config '{"mla_prefill_backend":"TRTLLM_RAGGED","use_prefill_query_quantization":true}' \ ⏎   --kv-cache-dty …[truncated]

### L3-2740c817ff  (L3, 2026-08-21, sha 2740c817ffbe, PR #52018)
TITLE: [Kernel] Add b12x FP4 MoE backend (#52018)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.trtllm_gen
FILES: .buildkite/test_areas/kernels.yaml (+21/-4); docs/features/quantization/b12x.md (+20/-7); setup.py (+1/-1); tests/kernels/moe/test_b12x.py (+1245/-0); tests/kernels/quantization/nvfp4_utils.py (+14/-3); tests/model_executor/test_b12x_warmup.py (+9/-3); tests/quantization/test_auto_round.py (+30/-5); vllm/config/kernel.py (+2/-0); vllm/envs.py (+5/-0); vllm/model_executor/layers/fused_moe/b12x.py (+770/-0); (+13 more)
LABELS: documentation, ready, ci/build, quantization
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ Builds on the optional b12x dependency, shared lazy imports, packed-storage reuse, ⏎ and warmup integration merged in #52016. ⏎  ⏎ This PR adds an explicitly selected ⏎ [b12x](https://github.com/local-inference-lab/b12x) FP4 MoE backend for NVIDIA ⏎ SM120 and SM121 GPUs using vLLM's existing fused-MoE backend interfaces. It ⏎ does not introduce a new MoE abstraction. ⏎  ⏎ Supported paths include: ⏎  ⏎ - Native NVFP4 and MXFP4 W4A4. ⏎ - W4A16 and supported  …[truncated]

### L3-6f74337c47  (L3, 2026-08-21, sha 6f74337c4748, PR #42376)
TITLE: [Bugfix][Spec Decode]Preserve user --speculative-config overrides for speculators-format models (#42376)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/transformers_utils/test_speculators_override.py (+156/-0); vllm/transformers_utils/config.py (+7/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ When the target model is in speculators format, [`maybe_override_with_speculators`](https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/config.py) rebuilds `speculative_config` from the speculators `config_dict` and returns it without merging `vllm_speculative_config`. Any field passed via `--speculative-config` is silently dropped on that path — including `attention_backend` introduced in #39930, `moe_backend`, a …[truncated]

### L3-cd7b7c265a  (L3, 2026-08-21, sha cd7b7c265a30, PR #43018)
TITLE: [ROCm] Cpu offload for ROCm 7.13+ to align the hipMemcpyBatchAsync params and perf in 7.14x (#43018)
SOURCES: release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flashinfer.trtllm_gen
FILES: csrc/libtorch_stable/cache_kernels.cu (+76/-19); tests/v1/simple_kv_offload/test_hip_mem_ops.py (+145/-0); vllm/envs.py (+7/-0); vllm/v1/simple_kv_offload/cuda_mem_ops.py (+110/-40)
LABELS: rocm, ready, v1, nvidia
BODY: ## Purpose ⏎  ⏎ This is related to the issue: https://github.com/vllm-project/vllm/issues/44341 ⏎  ⏎ In a previous enablement PR https://github.com/vllm-project/vllm/pull/40549, we enabled cpu offloading on ROCm, and we noticed the shortcoming of the hipMemcpyBatchAsync API lacked support on numAttrs paramter. ⏎  ⏎ This PR is to fix that situation. This PR is intended to be future compatible and backward compatible. Below is the details: ⏎  ⏎ For the hip …[truncated]

### L3-72aedcc426  (L3, 2026-08-21, sha 72aedcc426c2, PR #50382)
TITLE: [DCP] Default query replication for GLM sparse attention (#50382)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/distributed/test_dcp_a2a.py (+2/-11); vllm/config/parallel.py (+31/-7); vllm/config/vllm.py (+4/-0); vllm/engine/arg_utils.py (+7/-1); vllm/model_executor/models/config.py (+11/-0); vllm/model_executor/models/deepseek_v2.py (+7/-1)
LABELS: ready, deepseek, glm
BODY: ## Summary ⏎  ⏎ - add `--dcp-q-replicate` and `--no-dcp-q-replicate` as explicit CLI options; ⏎ - preserve `VLLM_DCP_Q_REPLICATE` as the highest-precedence override; ⏎ - enable query replication automatically for `glm_moe_dsa` models when decode DCP is enabled and PCP is not enabled; ⏎ - keep query replication opt-in for other MLA model families. ⏎  ⏎ The automatic default removes GLM's decode query all-gather at the cost of replicating the query projec …[truncated]

### L3-fe76112ff2  (L3, 2026-08-21, sha fe76112ff298, PR #52882)
TITLE: [ROCm][Perf] Optimize DeepSeek V4 C4A top-k with AITER (#52882)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+183/-20); csrc/libtorch_stable/sampler.cu (+132/-11); tests/kernels/test_top_k_per_row.py (+293/-0); vllm/model_executor/layers/sparse_attn_indexer.py (+3/-0); vllm/models/deepseek_v4/attention.py (+1/-0)
LABELS: performance, rocm, ready, deepseek, DSv4
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ This replaces the DeepSeek V4 C4A selector's ROCm top-k bottleneck with a ⏎ correctness-adapted AITER path for short/medium contexts and a graph-safe tuned ⏎ native fallback for long contexts on gfx950. ⏎  ⏎ - Route C4A prefill/decode top-k through the public AITER v0.1.19 operations. ⏎ - Convert AITER prefill's packed/global indices back to sequence-local indices, ⏎   preserving the `-1` sentinel. ⏎ - Treat every vLLM MTP query row independently ( …[truncated]

### L3-6feafb8b7d  (L3, 2026-08-21, sha 6feafb8b7d9c, PR #53201)
TITLE: [XPU] follow cuda path for mrope on XPU (#53201)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/rotary_embedding/mrope.py (+9/-0)
LABELS: intel-gpu, ready, nvidia
BODY: ## Purpose ⏎ PR #52005 rewrite interleaved_rope to fix torch.compile break but introduces additional overhead on eager mode on XPU. This PR switches mrope to `forward_cuda` to avoid this overhead and get better performance. ⏎ ## Test Plan ⏎ ``` ⏎ IMAGE_SIZE=512 NUM_WARMUP=2  python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-VL-32B-Instruct --enforce-eager --port 8124 --host 0.0.0.0 --trust-remote-code --gpu-memory-util=0.92 --no-enable …[truncated]

### L3-a60c66e3dc  (L3, 2026-08-21, sha a60c66e3dcd5, PR #53034)
TITLE: [Bugfix] Fix int32 index overflow in LoRA punica kernels at long context (#53034)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/lora/ops/triton_ops/kernel_utils.py (+6/-2)
LABELS: bug, ready, verified
ISSUES: #53028 Gemma 4 E2B + LoRA: illegal memory access in lora_expand at engine init once max_model_len >= 87383
BODY: ## Purpose ⏎  ⏎ Fixes #53028. ⏎  ⏎ `do_expand_kernel` and `do_shrink_kernel` index the token dimension with an int32 `ram`, so the byte offset wraps once a request is long enough. The result is an illegal memory access in `lora_expand`, and because the wrap point depends on the width of the per-token row, the failing length is model-specific rather than a round number. ⏎  ⏎ Gemma 4 E2B has `intermediate_size=6144`, so the merged `gate_up` output row is 12288 …[truncated]

### L3-574e6a00ef  (L3, 2026-08-21, sha 574e6a00efd3, PR #52796)
TITLE: [Bugfix][Attention] Normalize FlashInfer prefill LSE before merging (#52796)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flashinfer.py (+3/-2); vllm/v1/attention/backends/mla/prefill/flashinfer.py (+3/-2); vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py (+3/-2); vllm/v1/attention/backends/utils.py (+8/-0); tests/v1/attention/test_mla_backends.py (+6/-4)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎  ⏎ Prefix-cache hits could produce different logits from a cold prefill and ⏎ eventually lead to repetitive output. This was initially observed with GLM-5.2 ⏎ in a disaggregated prefill/decode deployment. ⏎  ⏎ KV cache digests matched before and after transfer, and the issue also ⏎ reproduced with BF16 KV cache. A local prefix-cache cold/warm comparison then ⏎ reproduced the logits divergence without KV transfer, narrowing the issue to ⏎ the  …[truncated]

### L3-d6c2fec9fd  (L3, 2026-08-21, sha d6c2fec9fd72, PR #53139)
TITLE: [Cleanup][MLA] Remove FlashInfer DSpark DCP support (#53139)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer
FILES: vllm/v1/attention/backends/mla/flashinfer_mla.py (+18/-121); tests/v1/attention/test_mla_backends.py (+37/-62)
LABELS: ready, nvidia, dflash, kimi, k3
BODY: ## Summary ⏎  ⏎ Follow-up to #52188. Preserve the TP-only FlashInfer DSpark support introduced with Kimi K3 in #50000, while removing the FlashInfer-specific extension for DSpark with decode context parallelism. ⏎  ⏎ - keep the original non-causal TP path that flattens a DSpark query block into single-token rows ⏎ - remove the FlashInfer-only decode metadata subclasses and builder override ⏎ - remove the cached flatten helper and causal DCP per-query sequenc …[truncated]

### L3-e6f35d3c69  (L3, 2026-08-21, sha e6f35d3c69b2, PR #52823)
TITLE: [DSv4 Perf] Adaptive topk width for dsv4, making #50004 back (#52823)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_flashmla_sparse.py (+57/-0); vllm/models/deepseek_v4/sparse_mla.py (+21/-3)
LABELS: ready, deepseek, DSv4
BODY: ## Purpose ⏎  ⏎ https://github.com/vllm-project/vllm/pull/50004 was reverted by https://github.com/vllm-project/vllm/pull/51318 ⏎  ⏎ This PR get the similar performance gain but doesn't hurt the accuracy ⏎  ⏎ ## Test ⏎  ⏎ ### Perf ⏎  ⏎ E2E perf see https://github.com/vllm-project/vllm/pull/50004 ⏎  ⏎ Kernel level perf can be seen in this AI generated script ⏎  ⏎ ```bash ⏎ # SPDX-License-Identifier: Apache-2.0 ⏎ # SPDX-FileCopyrightText: Copyright contributors to …[truncated]

### L3-ba53da60bb  (L3, 2026-08-21, sha ba53da60bb1a, PR #53186)
TITLE: Revert "[Bugfix][MoE] Tune FlashInfer experts to scheduler token limit" (#52989) (#53186)
SOURCES: subject_keyword, corpus:confirmed-reverts, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer_moe.py (+0/-2); vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py (+0/-2); vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py (+0/-1); vllm/model_executor/layers/fused_moe/experts/trtllm_lora_moe.py (+0/-2); vllm/model_executor/layers/fused_moe/experts/trtllm_mxint4_moe.py (+0/-2); vllm/model_executor/layers/quantization/utils/flashinfer_mxint4_moe.py (+1/-3)
LABELS: bug, ready, nvidia, quantization
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 52989 reason=crash_or_hang
BODY: Reverts #52989 ("[Bugfix][MoE] Tune FlashInfer experts to scheduler token limit", merged as `bfb6c1349`). ⏎  ⏎ ## Why ⏎  ⏎ Nightly build [#84887](https://buildkite.com/vllm/ci/builds/84887) (`bfb6c1349`, the merge commit of #52989) turned `:nvidia: (B200) LM Eval PCP` red for the first time. That job was green in the previous four nightlies (84473, 84555, 84687, 84753). ⏎  ⏎ `evals/gsm8k/test_gsm8k_correctness.py::test_gsm8k_correctness[GLM-5.2-NVFP4-TP2-PCP …[truncated]

### L3-d9e0ace7a0  (L3, 2026-08-21, sha d9e0ace7a07d, PR #53111)
TITLE: [Bugfix][Attention] Fall back to native FlashInfer decode when XQA cannot serve a KV-cache group's head_dim (#53111)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+17/-0)
LABELS: bug, ready, nvidia
BODY: ## Problem ⏎  ⏎ Serving a hybrid-attention model with the FlashInfer TRTLLM/XQA decode path crashes at engine init: ⏎  ⏎ ``` ⏎ ValueError: Invalid head_dim: 512, must be divisible by 16 and in range [16, 256] ⏎ ``` ⏎  ⏎ raised from `profile_cudagraph_memory()` during `determine_available_memory()`. ⏎  ⏎ The dedicated FlashInfer XQA decode API accepts head dimensions in `[16, 256]` that are divisible by 16. Gemma 4 is a hybrid-attention model that combines sliding-at …[truncated]

### L3-7a2fdbaac4  (L3, 2026-08-21, sha 7a2fdbaac449, PR #53152)
TITLE: [K3 Perf] Fuse MXFP4 top-k finalization into latent-tail, ~5% E2E latency reduction (#53152)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_kimi_k3_latent_moe_tail.py (+89/-17); tests/models/kimi_k3/test_latent_moe_tail.py (+152/-0); vllm/model_executor/layers/fused_moe/config.py (+21/-3); vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py (+10/-15); vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py (+27/-14); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+10/-13); vllm/model_executor/layers/fused_moe/fused_moe_method_base.py (+4/-3); vllm/model_executor/layers/fused_moe/modular_kernel.py (+21/-6); vllm/model_executor/layers/fused_moe/moe_output.py (+90/-0); vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py (+3/-0); (+7 more)
LABELS: performance, ready, nvidia, quantization, kimi, k3
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Part of https://github.com/vllm-project/vllm/issues/50587 ⏎  ⏎ Fuse MXFP4 top-k finalization into the Kimi K3 latent-tail kernel. ⏎  ⏎ Before ⏎  ⏎ ```bash ⏎ MXFP4 MoE kernel ⏎     │ ⏎     ├─ GEMM2 ⏎     │ ⏎     └─finalize kernel ⏎          ├─ unpermute ⏎          ├─ times router weight ⏎          ├─ top-k reduction ⏎          └─ Write [M, 3584] tensor ⏎                     │ ⏎                     ▼ ⏎ latent tail reads the tensor ⏎     └─ AllReduce + R …[truncated]

### L3-9fd750f00d  (L3, 2026-08-21, sha 9fd750f00de5, PR #50501)
TITLE: [XPU][INC] Add int4 w4a8 (dynamic int8 activation) backend for INC linear layers (#50501)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/quantization/test_auto_round.py (+618/-0); vllm/envs.py (+11/-0); vllm/model_executor/layers/quantization/inc/schemes/inc_w4a8_linear.py (+112/-0); vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_scheme.py (+46/-0)
LABELS: intel-gpu, quantization, verified
BODY: INC int4 layers on XPU always dispatch to ARK when it imports, with oneDNN `int4_gemm_w4a16` only as a fallback. `int4_gemm_w4a8` was never wired in at all. ⏎  ⏎ This PR adds `INCXPUW4A8LinearMethod` (int4 weights + dynamic per-token int8 activations, reusing the w4a16 weight layout unchanged, falling back to w4a16 below 512 tokens) and a `VLLM_XPU_INC_WNA16_BACKEND` selector: `auto | ark | w4a16 | w4a8`. ⏎  ⏎ Purely additive: one new test file, no e …[truncated]

### L3-b5f7fcc79d  (L3, 2026-08-21, sha b5f7fcc79df9, PR #53177)
TITLE: [ROCm][CI] Add float16 dtype and unsupported head size tests for paged attention (#53177)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_attention.py (+52/-1)
LABELS: rocm
BODY: - Extended DTYPES in test_attention.py to include torch.float16 (was only torch.bfloat16) ⏎   - Added test_paged_attention_unsupported_head_sizes to verify the ROCm kernel correctly rejects head sizes not supported by the native HIP kernel (only 64 and 128 are supported) ⏎  ⏎ The ROCm paged attention kernel (csrc/rocm/attention.cu) only supports head sizes 64 and 128. This test ensures the kernel raises a clear RuntimeError("Unsupported head size")  …[truncated]

### L3-0a3a4f7f35  (L3, 2026-08-21, sha 0a3a4f7f35b7, PR #52779)
TITLE: [Bugfix][KV Connector][NIXL] Support PCP producers (#52779)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_multi_connector.py (+6/-6); tests/v1/kv_connector/unit/test_nixl_connector.py (+52/-0); tests/v1/kv_connector/unit/test_nixl_push_connector.py (+19/-0); vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py (+12/-3); vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py (+27/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py (+47/-1); vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py (+4/-1); vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py (+3/-0); vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py (+3/-0)
LABELS: bug, kv-connector
BODY: ## Purpose ⏎  ⏎ Support NIXL P/D disaggregation when the producer uses prefill context parallelism (PCP) and decode context parallelism (DCP) is disabled on both sides. ⏎  ⏎ PCP expands the worker world without increasing the KV-cache shard count, so every PCP rank holds an equivalent KV replica. NIXL should publish and wait for one replica instead of treating all PCP workers as distinct KV shards. ⏎  ⏎ This change: ⏎  ⏎ - records each NIXL worker's PCP rank; ⏎ -  …[truncated]

### L3-9ff7041b55  (L3, 2026-08-21, sha 9ff7041b5555, PR #53002)
TITLE: [Bugfix][Spec Decode] Use group geometry for FlashAttention metadata (#53002)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:confirmed-reverts(reverted), body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flash_attn.py (+6/-5); vllm/v1/attention/backends/utils.py (+5/-1); vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (+6/-6); tests/v1/attention/test_group_head_counts.py (+53/-6)
LABELS: bug, speculative-decoding, ready, mrv2, dflash
DEEP_STUDY: deep-study: this PR was reverted by PR 53336 (reland, reason=correctness_or_accuracy)
BODY: ## Summary ⏎  ⏎ Use each FlashAttention metadata builder's own attention group as the source of truth for attention geometry: ⏎  ⏎ - query heads come from the registered attention layers ⏎ - KV heads and head size come from the group's `AttentionSpec` ⏎ - DFlash shallow-copies `VllmConfig` only to override causality, avoiding validator re-entry ⏎  ⏎ This prevents FA3 AOT scheduler metadata from being planned with target-model shapes when a draft model has a diff …[truncated]

### L3-8bdc70ec7b  (L3, 2026-08-21, sha 8bdc70ec7b37, PR #51718)
TITLE: [6/N][KV-Cache Layout Refactor] Standardize KV cache layout (#51718)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.rocm.v1_rocm_attn, L3.rocm.aiter_fa, L3.rocm.aiter_unified, L3.mla.common_v1, L3.mla.flashmla_v1_adapter, L3.mla.cutlass_v1_backend, L3.mla.flashattn, L3.mla.flashinfer, L3.mla.flashmla_sparse, L3.mla.flashinfer_sparse, L3.mla.rocm_aiter_sparse, L3.mla.flashattn_sparse, L3.dispatch.selector, L3.dispatch.abstract_interface, L3.flex_attention
FILES: .buildkite/test_areas/disaggregated.yaml (+3/-0); benchmarks/attention_benchmarks/runner.py (+47/-48); docs/assets/contributing/dockerfile-stages-dependency.png (+0/-0); docs/features/mooncake_store_connector_usage.md (+0/-1); docs/features/nixl_connector_compatibility.md (+4/-4); docs/features/nixl_connector_usage.md (+1/-10); rust/Cargo.lock (+9/-9); tests/compile/passes/test_fusion_attn.py (+32/-24); tests/compile/passes/test_mla_attn_quant_fusion.py (+6/-19); tests/compile/passes/test_mla_rope_kvcache_cat_fusion.py (+4/-19); (+140 more)
LABELS: documentation, performance, rocm, intel-gpu, speculative-decoding, ready, torch.compile, ci/build, multi-modality, deepseek
BODY: ## Purpose ⏎  ⏎ The core of the KV-cache layout standardization series (RFC #42082).  ⏎  ⏎ Standardizes every KV cache allocation on the logical `[L, B, H, N, C]` vocabulary: ⏎  ⏎ - `KVCacheLayout` enumerates the physical stride permutations (`LBHNC`, `LBNHC`, `LHBNC`, `BLHNC`, `BLNHC`, `BHLNC`) ⏎ - Layout resolution has a single writer: attention-backend selection publishes the layout into `CacheConfig` (test override > backend-required > `VLLM_KV_CACH …[truncated]

### L3-b2db227a7c  (L3, 2026-08-21, sha b2db227a7c4c, PR #53240)
TITLE: [Bugfix][R3] Unwrap UniformTypeKVCacheSpecs when selecting the routed-experts KV group (#53240)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+2/-7); tests/model_executor/test_routed_experts_capture.py (+46/-7); tests/v1/core/test_kv_cache_utils.py (+39/-0); vllm/model_executor/layers/fused_moe/routed_experts_capturer.py (+2/-2); vllm/v1/kv_cache_interface.py (+30/-7)
LABELS: bug, ready, nvidia, kv-cache-manager
BODY: ## Purpose ⏎  ⏎ --enable-return-routed-experts` crashes at worker init on DeepSeek-V4: ⏎  ⏎ ``` ⏎ ValueError: Routed-experts capture requires a full-attention KV cache group. ⏎ ``` ⏎  ⏎ `get_routed_experts_attn_gid` scans groups with ⏎ `isinstance(group.kv_cache_spec, FullAttentionSpec)`, but `UniformTypeKVCacheSpecs` ⏎ subclasses `KVCacheSpec` directly, so no group matches. This hits DeepSeek-V4 ⏎ (`group_and_unify_kv_cache_specs` wraps every group) and th …[truncated]

### L3-7f4a1b7e24  (L3, 2026-08-22, sha 7f4a1b7e246c, PR #53110)
TITLE: [Bugfix][ROCM] Fix the MXFP8 block scale exponent (#53110)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/utils/mxfp8_utils.py (+9/-2)
LABELS: bug, rocm, ready, quantization
DEEP_STUDY: deep-study correctness case vllm:7f4a1b7e24: class=numerical_precision; symptom=wrong_output_or_accuracy; introducing=unknown
BODY: The test belongs to the Kernels Root Misc Test group, which AMD CI does not run yet. ⏎ This fix should land before the PR that enables that group (https://github.com/vllm-project/vllm/pull/50519). ⏎  ⏎ ## Purpose ⏎  ⏎ The fused Triton MXFP8 quantizer picks the shared block scale with ⏎ `floor(log2(amax)) + 127`, which maps the block maximum onto 1.0 instead of onto ⏎ the top of the e4m3 range. Small elements of a block then land in the subnormal ⏎ region …[truncated]

### L3-9eb9d9d395  (L3, 2026-08-22, sha 9eb9d9d39539, PR #52789)
TITLE: [Perf] Support internal prefill checkpoints for Mamba prefix caching, 9%~25% TTFT improvement (#52789)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: cmake/external_projects/flashkda.cmake (+1/-1); csrc/flashkda_registration.cpp (+2/-1); tests/models/kimi_k3/test_kda.py (+81/-0); tests/models/kimi_k3/test_kda_metadata.py (+50/-0); tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py (+85/-0); tests/v1/core/test_mamba_align_chunk_split.py (+52/-3); tests/v1/core/test_prefix_caching.py (+1/-0); vllm/models/kimi_k3/nvidia/kda.py (+161/-19); vllm/models/kimi_k3/nvidia/kda_metadata.py (+67/-0); vllm/v1/core/sched/scheduler.py (+17/-2); (+2 more)
LABELS: ready, ci/build, kimi, k3, scheduler, kv-cache-manager
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Mamba prefix caching currently splits a prefill at the last block boundary ⏎  ⏎ Let's use a 8k input as an example ⏎  ⏎ Before ⏎  ⏎ ```bash ⏎ Model forward (7680 tokens) -> ⏎ FlashKDA(7680 tokens) -> ⏎ save checkpoint -> ⏎ Model forward (320 tokens) -> ⏎ FlashKDA(320 tokens) ⏎ ``` ⏎  ⏎ Now ⏎  ⏎ ```bash ⏎ Model forward (8000 tokens) -> ⏎ FlashKDA(7680) -> ⏎ save checkpoint -> ⏎ FlashKDA(320) ⏎ ``` ⏎  ⏎ This PR helps us avoid a second full-model pass throug …[truncated]

### L3-e9d1398d9e  (L3, 2026-08-22, sha e9d1398d9edf, PR #53327)
TITLE: [Bugfix][Kimi K3] Enable deferred MoE finalization before weight loading (#53327)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/kimi_k3/test_latent_moe_tail.py (+79/-0); vllm/models/kimi_k3/nvidia/latent_moe_runner.py (+11/-9)
LABELS: bug, ready, kimi, k3
BODY: ## Summary ⏎  ⏎ - Enable K3 deferred MoE finalization from the selected monolithic expert class during runner construction. ⏎ - Keep the existing hidden-size and parallel-topology guards. ⏎ - Log when deferred top-k finalization is active and add a regression test for the pre-weight-loading lifecycle. ⏎  ⏎ ## Root cause ⏎  ⏎ #53152 checked `quant_method.moe_kernel` in `LatentMoERunner.__init__`. `Mxfp4MoEMethod` initializes that field to `None` and only creates  …[truncated]

### L3-da329cc303  (L3, 2026-08-22, sha da329cc303a5, PR #50272)
TITLE: [Bugfix] Fix speculative decoding for short_conv (LFM2) models (#50272)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/mamba_utils.py (+2/-1); vllm/model_executor/layers/mamba/short_conv.py (+33/-9); vllm/v1/worker/gpu/model_states/mamba_hybrid.py (+6/-1); vllm/v1/worker/gpu_model_runner.py (+2/-0)
LABELS: bug, ready, v1, mrv2
BODY: ## Purpose ⏎  ⏎ Supersedes #44296 (rebased onto main with the original commit and authorship preserved — thanks @Tonoken3); that PR has had failing pre-commit since opening and merge conflicts since July 6 with no author activity, so this revives it to make the merge cheap. ⏎  ⏎ Speculative decoding (ngram, EAGLE-3, any method) on `short_conv` (LFM2 / LFM2.5) targets silently corrupts output at the first rejected token: attention KV rolls back, but the c …[truncated]

### L3-2f55ef254c  (L3, 2026-08-22, sha 2f55ef254c70, PR #52560)
TITLE: [Model] Add Qwen3-Omni DSpark support (#52560)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/test_qwen3_omni.py (+171/-1); tests/models/registry.py (+7/-0); tests/test_config.py (+121/-0); tests/transformers_utils/test_speculators_dspark_config.py (+58/-0); vllm/config/speculative.py (+302/-2); vllm/model_executor/models/qwen3_dflash.py (+8/-8); vllm/model_executor/models/qwen3_dspark.py (+20/-2); vllm/model_executor/models/qwen3_omni_moe_thinker.py (+11/-1); vllm/model_executor/models/registry.py (+1/-0); vllm/transformers_utils/configs/speculators/algos.py (+17/-4); (+1 more)
LABELS: new-model, speculative-decoding, ready, qwen, mrv2, dflash
BODY: ## Purpose ⏎  ⏎ Add framework support for using a `Qwen3OmniDSparkModel` draft architecture with a Qwen3-Omni thinker target. ⏎  ⏎ This change: ⏎ - registers the dedicated Qwen3-Omni DSpark architecture while reusing the shared `Qwen3DSparkForCausalLM` runtime implementation; ⏎ - exposes the target thinker's post-DeepStack auxiliary hidden states required by DSpark; ⏎ - preserves the dedicated architecture and DSpark sampling fields when converting msModelSpec …[truncated]

### L3-30b34171b1  (L3, 2026-08-22, sha 30b34171b113, PR #53351)
TITLE: [ROCm][CI] Restore attention coverage after KV-cache layout refactor (#53351)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_minimax_m3.py (+15/-0); tests/v1/attention/test_attention_backends.py (+20/-9); tests/v1/attention/test_mla_backends.py (+15/-16)
LABELS: rocm, minimax
BODY: This is a follow-up to #51718, which standardized KV-cache layouts but also changed several ROCm test paths. It realigns FlexAttention tests with production and restores coverage for relevant AITER safety and MLA paths. It also avoids generating CUDA-only MLA prefill parameter matrices on ROCm, reducing collection from 2,911 to 591 cases. Together, these changes improve regression coverage while modestly reducing test resource usage. ⏎  ⏎ - Build M …[truncated]

### L3-bbe8b23e1a  (L3, 2026-08-22, sha bbe8b23e1a2b, PR #52557)
TITLE: [Deprecation] Remove dead use_prefill_decode_attention flag (#52557)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/attention.py (+0/-4); tests/engine/test_arg_utils.py (+0/-5); tests/v1/attention/utils.py (+0/-1)
LABELS: ready, cpu
BODY: ## Purpose ⏎  ⏎ `--attention-config.use_prefill_decode_attention` is accepted, documented, and ⏎ has no effect. ⏎  ⏎ It used to select `ROCM_ATTN` on ROCm. #36702 made `ROCM_ATTN` the ⏎ unconditional first entry in the ROCm priority list and removed the branch that ⏎ read the flag, so the field has had no reader in `vllm/` since. Its original ⏎ purpose is now the default behaviour, making this leftover surface rather than ⏎ a regression. ⏎  ⏎ Two tests suggest otherwi …[truncated]

### L3-185cada36b  (L3, 2026-08-23, sha 185cada36bb2, PR #53460)
TITLE: [Model] Fix KV cache layout and optimize Dots3 NOTE Omni encoders (#53460)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/dots3_note/nvidia/attention.py (+8/-0); vllm/models/dots3_note/nvidia/audio.py (+96/-14); vllm/models/dots3_note/nvidia/multimodal.py (+22/-2); vllm/models/dots3_note/nvidia/vision.py (+56/-62); vllm/models/dots3_note/nvidia/vision_attention.py (+29/-12)
LABELS: ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ This PR improves the correctness and efficiency of Dots3 NOTE Omni inference. ⏎  ⏎ The changes include: ⏎  ⏎ - Force the Dots3 NOTE padded sparse attention backend to use the `BLHNC` KV-cache layout so mixed DSA/MLA/SWA cache page sizes are represented correctly. ⏎ - Optimize the vision encoder by: ⏎   - preparing sequence metadata once per forward pass; ⏎   - keeping grid metadata on CPU to avoid unnecessary device synchronization; ⏎   - p …[truncated]

### L3-f94666b60d  (L3, 2026-08-23, sha f94666b60d4c, PR #52389)
TITLE: [Bugfix][XPU] Skip oneCCL warm-up all_reduce when world_size == 1 (#52389)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/xpu_worker.py (+7/-2)
LABELS: bug, intel-gpu, verified
ISSUES: #52386 [Bug]: XPU worker fails to start with world_size=1 when oneCCL cannot initialize
BODY: ## Purpose ⏎  ⏎ Skip the oneCCL warm-up all_reduce in `XPUWorker.init_device()` when ⏎ `world_size == 1`. ⏎  ⏎ The warm-up is guarded only by `is_xccl_available()`, so single-device XPU ⏎ serving still requires a working oneCCL communicator. On platforms where ⏎ oneCCL cannot enumerate GPU topology this aborts engine startup, even though ⏎ no collective is issued during inference. ⏎  ⏎ Fixes #52386 ⏎  ⏎ ## Test Plan ⏎  ⏎ - Single XPU (`--tensor-parallel-size 1 …[truncated]

### L3-a047e2543d  (L3, 2026-08-23, sha a047e2543da5, PR #53000)
TITLE: Fix MNNVL Lamport mailbox publication and cleanup (#53000)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/distributed.yaml (+1/-0); csrc/custom_all_gather_reduce_scatter.cuh (+45/-6); tests/distributed/test_custom_all_gather_reduce_scatter.py (+326/-0)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Fix silent output corruption in vLLM's MNNVL Lamport all-gather and reduce-scatter. CUDA multicast addresses may only be written with `multimem.*`, so mailbox payloads use a relaxed, system-scope multicast store ([PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red)). The payload words are also the readiness signal: volatile readers reje …[truncated]

### L3-e6e1af4ca1  (L3, 2026-08-24, sha e6e1af4ca190, PR #53318)
TITLE: [Perf] Tune FlashInfer all-reduce selection on SM103 (#53318)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/distributed/test_comm_ops.py (+84/-0); vllm/distributed/device_communicators/all_reduce_utils.py (+8/-0); vllm/distributed/device_communicators/cuda_communicator.py (+16/-13); vllm/distributed/device_communicators/flashinfer_all_reduce.py (+56/-8)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Tune standalone FlashInfer all-reduce using conservative, topology-specific cutoffs on SM103. ⏎ - Keep the existing `VLLM_ALLREDUCE_USE_FLASHINFER` gate and its default unchanged. ⏎ - When enabled, try FlashInfer before NCCL symmetric memory for inputs that satisfy FlashInfer's eligibility checks. ⏎  ⏎ ## Performance ⏎  ⏎ Hardware/software: GB300 GPUs in one NVL72 domain, PyTorch 2.13 / CUDA 13.0, FlashInfer 0.6.15, and NCCL 2.30.7. ⏎  ⏎ |  …[truncated]

### L3-4ca856b0b5  (L3, 2026-08-24, sha 4ca856b0b59d, PR #51979)
TITLE: [Bugfix] Release worker RPC payload before next dequeue (#51979)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/executor/test_multiproc_executor.py (+67/-0); vllm/v1/executor/multiproc_executor.py (+25/-20)
LABELS: bug, verified
ISSUES: #43639 [Bug]: OOM killed caused by possible CPU memory leak in vLLM Worker RPC Broadcast Deserialization Path
BODY: ## Summary ⏎  ⏎ Fixes #43639 by executing each worker RPC in a separate stack frame. This ⏎ releases the deserialized request arguments and local output reference before ⏎ the next `MessageQueue.dequeue()` deserializes another request, without adding ⏎ a full `gc.collect()` to the RPC hot path. ⏎  ⏎ The existing exception-to-worker-response behavior is preserved. ⏎  ⏎ ## Why this approach ⏎  ⏎ The previous loop retains `method`, `args`, `kwargs`, and `output` while th …[truncated]

### L3-0ecc284790  (L3, 2026-08-24, sha 0ecc284790e5, PR #51031)
TITLE: [Bugfix][Kernel] Handle kernel block sizes in V2 DCP slot mapping (#51031)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_block_table.py (+44/-0); vllm/v1/worker/gpu/block_table.py (+29/-15)
LABELS: bug, ready, mrv2
BODY: As we know, each backend can force their own supported block_size and override the one specified by the user at runtime. ⏎ Current DCP slot mapping code does not take that into account, and may end up re-using the stale/old block_size value initially set by the user. ⏎  ⏎ Repro with one practical example: set --block-size 128->FlashInfer MLA sets kernel block to 64 ⏎ ``` ⏎   VLLM_USE_V2_MODEL_RUNNER=1 \ ⏎   VLLM_DEEP_GEMM_WARMUP=skip \ ⏎   vllm serve de …[truncated]

### L3-ecfa7bb373  (L3, 2026-08-24, sha ecfa7bb37316, PR #53560)
TITLE: [MM] Cache common token sequences (#53560)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/multimodal/test_processing.py (+2/-6); vllm/model_executor/models/llava_onevision2.py (+2/-1); vllm/model_executor/models/minicpmv.py (+1/-1); vllm/model_executor/models/minicpmv4_6.py (+1/-1); vllm/model_executor/models/moondream3.py (+3/-1); vllm/model_executor/models/moss_audio.py (+7/-2); vllm/model_executor/models/moss_transcribe_diarize.py (+4/-2); vllm/model_executor/models/paligemma.py (+3/-2); vllm/model_executor/models/phi3v.py (+2/-1); vllm/model_executor/models/qwen3_asr.py (+2/-1); (+4 more)
LABELS: ready, multi-modality, qwen
BODY: ## Purpose ⏎  ⏎ Optimization: use `cached_encode` in more places ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-f620499ee3  (L3, 2026-08-24, sha f620499ee3fe, PR #53336)
TITLE: [Bugfix][Spec Decode] Reapply group geometry for FlashAttention metadata (#53336)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:confirmed-reverts, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/flash_attn.py (+6/-5); vllm/v1/attention/backends/utils.py (+5/-1); vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (+6/-6); tests/v1/attention/test_group_head_counts.py (+53/-6)
LABELS: bug, speculative-decoding, ready, mrv2, dflash
DEEP_STUDY: deep-study revert record: reland of PR(s) 53002 reason=correctness_or_accuracy
BODY: ## Summary ⏎  ⏎ Reapply #53002 after it was temporarily reverted in #51718 to unblock the KV-cache layout refactor. ⏎  ⏎ FlashAttention metadata builders use their own attention group as the source of truth for geometry: ⏎  ⏎ - query heads come from the registered attention layers ⏎ - KV heads and head size come from the group's `AttentionSpec` ⏎ - DFlash shallow-copies `VllmConfig` only to override causality, avoiding validator re-entry ⏎  ⏎ This prevents FA3 AOT s …[truncated]

### L3-4f686e182a  (L3, 2026-08-24, sha 4f686e182a34, PR #53559)
TITLE: [MISC] Cleanup deprecated parameters (#53559)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/plugins/bge_m3_sparse_plugin/bge_m3_sparse_processor/sparse_embeddings_processor.py (+4/-6); vllm/config/attention.py (+1/-1); vllm/entrypoints/pooling/base/io_processor.py (+1/-1); vllm/entrypoints/pooling/offline.py (+1/-1); vllm/entrypoints/pooling/pooling/serving.py (+1/-1); vllm/envs.py (+0/-19); vllm/model_executor/models/mistral_large_3.py (+1/-1); vllm/multimodal/utils.py (+0/-14); vllm/plugins/io_processors/interface.py (+0/-44); vllm/pooling_params.py (+1/-1); (+2 more)
LABELS: frontend, ready, multi-modality, mistral
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-6a9c69fa85  (L3, 2026-08-24, sha 6a9c69fa8513, PR #52157)
TITLE: [Attention][Spec Decode] Support varlen trtllm-gen decode for adaptive verification (#52157)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+45/-7); tests/kernels/attention/test_flashinfer_trtllm_attention.py (+93/-0)
LABELS: ready, nvidia
ISSUES: #51871 [Feature][DSpark]: Enable varlen for Gemma4 (FI TRTLLM-GQA)
BODY: ## Purpose ⏎  ⏎ - Resolves #51871: adaptive verification needs decode batches with per-request query lengths ⏎ - Wire FlashInfer trtllm-gen decode to `cum_seq_lens_q`/`max_q_len`; `q_len_per_req` must be `None` for varlen ⏎ - Flip `supports_device_cpu_query_lens_mismatch` on SM100: device `qo_indptr` is the source of truth, CPU lengths only an upper bound ⏎ - Guard: adaptive verification without the trtllm-gen kernel now fails fast at init ⏎ - Not a du …[truncated]

### L3-22099afc64  (L3, 2026-08-24, sha 22099afc6423, PR #52377)
TITLE: [Bugfix][DCP] Handle sparse MLA metadata after DCP Manager refactor (#52377)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+15/-9); vllm/v1/attention/ops/dcp.py (+47/-2)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ DCP Manager refactoring #50484 is focused on DCP for dense MLA, which does not handle sparse MLA properly. ⏎  ⏎ - `MLASparseMetadata` does not inherit `MLACommonMetadata`, thus `attn_metadata.decode` attribute access should be guarded by getattr ⏎ - DCP query-gather assumes a decode query, so its workspace (direct_dcp_q_gather_workspace) is too small for prefill queries (which is passed via forced MQA codepath). ⏎ - Forced MQA codepath  …[truncated]

### L3-07ef21bc69  (L3, 2026-08-24, sha 07ef21bc6984, PR #52676)
TITLE: [Kernel][Perf] Enable fused QK-norm + partial MRoPE + gate for Qwen3.6 (#52676)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_fused_qk_norm_rope_gate.py (+77/-118); vllm/model_executor/layers/fused_qk_norm_rope.py (+87/-10); vllm/model_executor/models/qwen3_next.py (+28/-10)
LABELS: qwen
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ Qwen3.6 vision requests use per-head `[q | gate]`, zero-centered ⏎ GemmaRMSNorm, and partial interleaved MRoPE. This PR extends the existing ⏎ fused Q/K norm + RoPE + gate operator to support `[3, M]` T/H/W positions, ⏎ partial MRoPE, the gated layout, and raw GemmaRMSNorm weights. Unsupported ⏎ inputs keep the existing fallback. ⏎  ⏎ ## Validation ⏎  ⏎ The kernel suite passed **6/6** for plain RoPE and MRoPE at `M=1/4/37`; the ⏎ focused ColQwen regres …[truncated]

### L3-23ab0cfdbc  (L3, 2026-08-24, sha 23ab0cfdbc43, PR #52193)
TITLE: speculative decoding under tensor parallelism (TP>1) , workspace creation select max hidden dim of target and draft model (#52193)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/passes/distributed/test_fusion_all_reduce.py (+44/-0); vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+22/-3)
LABELS: ready, torch.compile, nvidia
ISSUES: #52023 [Bug]: draft_model speculative decoding crashes at init under TP>1 when draft hidden_size > target (TRT-LLM fused allreduce+RMSNorm workspace sized from target only)
BODY: ## Purpose ⏎ Fixes #52023 ⏎ With draft_model speculative decoding under tensor parallelism (TP>1), the engine crashes at init when the draft model's hidden_size is larger than the target model's(TRT-LLM fused allreduce+RMSNorm workspace sized from target only). ⏎  ⏎ I kept logs and [seongyun](https://github.com/seongyun1104) help me run it on H100 and we confirmed workspace creation happens in eager and init does have draft model' config.  ⏎  ⏎ My fix  …[truncated]

### L3-cbe3966f9c  (L3, 2026-08-25, sha cbe3966f9cc6, PR #52066)
TITLE: [XPU] Fix sparse-MLA metadata sync (#52066)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/xpu_mla_sparse.py (+22/-0)
LABELS: intel-gpu, ready, verified
BODY: Fix issue below for sparse-MLA FP8 MoE models (e.g. GLM-5.2, DeepSeek DSA) on Intel GPUs.  ⏎ Sparse-MLA metadata sync (xpu_mla_sparse.py): The shared MLA layer (mla_attention.py::forward_impl) unconditionally reads num_decodes/num_prefills/num_decode_tokens on every MLA metadata; the CUDA sparse backends carry them via SparseMLACommonMetadataBuilder, but the XPU sparse backend built its own metadata without them, so a sparse-MLA run on XPU crashed …[truncated]

### L3-8fe9317f2e  (L3, 2026-08-25, sha 8fe9317f2e40, PR #49636)
TITLE: [Model][MoE] DeepSeek-V4: add opt-in FlashInfer moe_ep expert backend (#49636)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/test_deepseek_v4_fi_moe_ep.py (+259/-0); vllm/config/kernel.py (+56/-1); vllm/models/deepseek_v4/nvidia/fi_moe.py (+397/-0); vllm/models/deepseek_v4/nvidia/model.py (+32/-11); vllm/utils/flashinfer_moe_ep.py (+339/-0)
LABELS: ready, ci/build, deepseek, nvidia, verified, DSv4
BODY: Add a FlashInfer `moe_ep` compute path for the DeepSeek-V4 mega-MoE experts, selectable at runtime alongside the existing native `deep_gemm_mega_moe` backend. ⏎  ⏎ - New module `vllm/utils/flashinfer_moe_ep.py`: model-agnostic plumbing only — backend registry and predicates, flashinfer `moe_ep` runtime bootstrap/finalize (bootstrap passes `device=torch.cuda.current_device()`, see review item on the LOCAL_RANK pin), NVFP4 prequant weight packing + e …[truncated]

### L3-af119619c4  (L3, 2026-08-25, sha af119619c4ca, PR #53530)
TITLE: Exclude the cpu backend from vLLM's active-Triton-driver count (#53530) (#53530)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/test_triton_utils.py (+51/-0); vllm/triton_utils/importing.py (+8/-1)
LABELS: ready, meta-exported, verified
BODY: Summary by human (minjang) ⏎  ⏎ We (Meta) internally have the Triton CPU backend for Triton. So, on a GPU machine, you will see two active drivers as the CPU driver always returns True. This is a quick fix to address a potential additional driver in Triton. ⏎  ⏎  ⏎ --- ⏎  ⏎ ## The issue ⏎ - D116904949 added the Triton CPU backend to `third-party/triton/beta`, and `CPUDriver.is_active()` returns `True` unconditionally (`third_party/cpu/backend/driver.py:4 …[truncated]

### L3-afc91fa0d2  (L3, 2026-08-25, sha afc91fa0d2eb, PR #51839)
TITLE: [Profiler] Fix start_profile permanently no-op after max_iterations auto-stop (#51839)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_profiler.py (+28/-0); vllm/profiler/wrapper.py (+6/-4)
LABELS: ready
BODY: ## Summary ⏎ `WorkerProfiler.step()`'s max_iterations auto-stop calls `_call_stop()` directly instead of the public `stop()`, so `_active` never resets. Any later `start_profile` then silently no-ops forever, since `start()` bails out early whenever `_active` is still True -- only `stop()` resets it. ⏎  ⏎ This breaks the `max_iterations` "fire and forget" use case: profiling only ever runs once per worker process, with no error to explain why a second  …[truncated]

### L3-d5cadcee86  (L3, 2026-08-25, sha d5cadcee8641, PR #52242)
TITLE: [Feature][DSpark]: Logprobs adaptive verification (#52242)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: docs/features/speculative_decoding/adaptive_verification.md (+1/-1); tests/v1/engine/test_output_processor.py (+1/-1); tests/v1/test_outputs.py (+19/-0); tests/v1/worker/test_gpu_rejection_sampler_chunking.py (+1/-0); vllm/sampling_params.py (+0/-14); vllm/v1/engine/logprobs.py (+1/-1); vllm/v1/outputs.py (+21/-4); vllm/v1/worker/gpu/sample/logprob.py (+4/-2); vllm/v1/worker/gpu/spec_decode/rejection_sampler.py (+12/-1); vllm/v1/worker/gpu_model_runner.py (+1/-1)
LABELS: documentation, speculative-decoding, ready, mrv2, dflash
ISSUES: #51873 [Feature][DSpark]: Enable logprobs with adaptive verification
BODY: Solves #51873  ⏎  ⏎ ## Purpose ⏎  ⏎ Lifts the restriction blocking logprobs with enable_adaptive_verification, implementing the TODO from #47808. ⏎  ⏎ Adaptive verification assigns per-request draft counts on device; the CPU-side `cu_num_logits_np` only holds an evenly-distributed stand-in. Using it as `cu_num_generated_tokens` would misassign logprob rows across requests. Instead, the rejection sampler now attaches the device `cu_num_logits`. The tens …[truncated]

### L3-41729fc53b  (L3, 2026-08-25, sha 41729fc53b02, PR #52388)
TITLE: [K3 Perf] Optimize k3 mamba metadata preparation, 6.6~7.6x kernel performance improvement (#52388)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/kimi_k3/test_kda_metadata.py (+37/-0); vllm/models/kimi_k3/nvidia/kda_metadata.py (+18/-6); vllm/v1/worker/gpu/model_states/mamba_hybrid.py (+18/-0); vllm/v1/worker/mamba_utils.py (+114/-0)
LABELS: ready, mrv2, kimi, k3
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Optimize Kimi K3 Mamba `align` metadata preparation by computing aligned state indices for all KV-cache groups in one Triton launch. ⏎  ⏎ Before: ⏎  ⏎ ```text ⏎ prepare_attn ⏎   -> builder group 0 -> allocate buffer -> launch kernel ⏎   -> builder group 1 -> allocate buffer -> launch kernel ⏎   -> builder group N -> allocate buffer -> launch kernel ⏎ ``` ⏎  ⏎ After: ⏎  ⏎ ```text ⏎ prepare_attn ⏎   -> persistent output buffer ⏎   -> one multi-group  …[truncated]

### L3-7de96050c5  (L3, 2026-08-25, sha 7de96050c530, PR #52783)
TITLE: [Spec Decode] Enable adaptive DSpark on SM100 sparse MLA (#52783)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer_sparse
FILES: vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+17/-0); vllm/v1/worker/gpu/attn_utils.py (+27/-7); vllm/v1/worker/gpu/model_runner.py (+16/-4); vllm/v1/worker/gpu/spec_decode/adaptive_verification.py (+24/-5); tests/v1/attention/test_dspark_noncausal_sparse_mla.py (+60/-2); tests/v1/spec_decode/test_adaptive_verification.py (+58/-0); tests/v1/worker/test_attn_utils.py (+91/-0)
LABELS: speculative-decoding, ready, nvidia, mrv2, dflash
ISSUES: #52785 [Feature][DSpark]: Enable varlen for GLM-5.x/DSV3.2 (Sparse MLA)
BODY: ## Purpose ⏎  ⏎ Resolves #52785. This is part of the adaptive DSpark backend expansion tracked by #51303. Related GLM-5.2 DSpark bring-up context is in #50851. ⏎  ⏎ This enables adaptive DSpark verification with `FlashInferMLASparseTRTLLMBackend`. It was validated end to end with `nvidia/GLM-5.2-NVFP4` and `RedHatAI/GLM-5.2-speculator.dspark` on 4× NVIDIA B300. ⏎  ⏎ <img width="3960" height="2520" alt="glm52_dspark_adaptive" src="https://github.com/use …[truncated]

### L3-80771bbbdd  (L3, 2026-08-25, sha 80771bbbddf9, PR #51292)
TITLE: [Core] Disable fuse_allreduce_rms under VLLM_BATCH_INVARIANT (non-deterministic under TP) (#51292)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+4/-0)
LABELS: ready
ISSUES: #51290 [Bug]: VLLM_BATCH_INVARIANT=1 not deterministic under tensor parallelism (TP>1); fuse_allreduce_rms fused all-reduce is the cause
BODY: ## Purpose ⏎  ⏎ Fixes #51290. ⏎  ⏎ `VLLM_BATCH_INVARIANT=1` is bit-stable on a single GPU but **non-deterministic under ⏎ tensor parallelism**: repeating a byte-identical workload against the same engine returns ⏎ different logprobs (and, on near-ties, different tokens) once `tensor_parallel_size > 1`. ⏎  ⏎ The cause is the fused all-reduce + RMSNorm pass (`fuse_allreduce_rms`, a FlashInfer ⏎ kernel), which reduces in a run-to-run-varying order under TP. It is ena …[truncated]

### L3-bc2d63e650  (L3, 2026-08-25, sha bc2d63e650f6, PR #53534)
TITLE: [Kimi K3][Kernel] Enable low-latency decode GEMM dispatch on SM100 (#53534)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_bf16_skinny_gemm.py (+95/-1); vllm/models/kimi_k3/nvidia/low_latency_gemm.py (+228/-6)
LABELS: nvidia, kimi, k3
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## What this PR does / why we need it ⏎  ⏎ The Kimi-K3 BF16 decode GEMM selector (`vllm/models/kimi_k3/nvidia/low_latency_gemm.py`) is gated behind `_is_sm103()` with a single table measured on B300, so on **B200 (SM100)** every decode projection falls back to cuBLAS — even though neither backend is SM103-specific (the CuTe DSL skinny GEMM has no arch gate; `dsv3_fused_a_gemm` only requires SM90+). ⏎  ⏎ This PR adds a separately measured SM100 table (`KI …[truncated]

### L3-7156c63bef  (L3, 2026-08-25, sha 7156c63bef1f, PR #52980)
TITLE: [SM100] Hdim 256 optimized (#52980)
SOURCES: path_core, symbol_pickaxe, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_build, L3.flash_attn.fa_utils
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/v1/attention/backends/fa_utils.py (+79/-14); vllm/v1/attention/backends/flash_attn.py (+83/-6); vllm/v1/attention/backends/flash_attn_diffkv.py (+1/-0); docs/design/attention_backends.md (+8/-0); tests/kernels/attention/test_attention_selector.py (+176/-1); tests/kernels/attention/test_flash_attn.py (+75/-0)
LABELS: documentation, ready, ci/build
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Re-enables FA4 for `head_size=256` on Blackwell, which was temporarily disabled in ⏎ https://github.com/vllm-project/vllm/pull/52050 because the SM100 2-CTA hd256 kernel rejected ⏎ `seqused_q/seqused_k` — something vLLM's decoder attention always supplies. ⏎  ⏎ That gap is now closed upstream in https://github.com/vllm-project/flash-attention/pull/180, ⏎ which adds forward `seqused` support along with hd256 optimizations. This PR reverts …[truncated]

### L3-bc39ded3c4  (L3, 2026-08-25, sha bc39ded3c4e7, PR #53329)
TITLE: [Bugfix][KV Offload] Defer request-level cascade of in-flight primary keys (#53329)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_offload/tiering/test_tiering_offloading.py (+134/-0); vllm/v1/kv_offload/tiering/manager.py (+47/-8)
LABELS: bug, ready
ISSUES: #53062 [Bugfix][KVOffload] Request-level cascade silently drops HIT_PENDING keys
BODY: ## Purpose ⏎  ⏎ Fixes #53062. ⏎  ⏎ `TieringOffloadingManager._cascade_existing_blocks_to_request_level_tiers` keeps only keys whose primary-tier `lookup` is `HIT`. A `HIT_PENDING` key, present in the primary tier with its write still in flight, is dropped permanently: `prepare_store` already excluded it as present so `complete_store` never cascades it, the cascade drops it for not being readable, and the scheduler then advances `next_stored_chunk_idx` pa …[truncated]

### L3-796822d141  (L3, 2026-08-26, sha 796822d14138, PR #53712)
TITLE: [Hardware][AMD][Perf][Bugfix] Update ROCr and clr in base image (#53712)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm_base (+81/-0)
LABELS: bug, rocm, ci/build
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ This PR updates ROCR and CLR in the ROCm 7.2.3 vLLM base image to backport in some bugfixes for graph replay (which was segfaulting under replay of complex graphs under some workloads) and for kernel dispatch latency under some stream dependencies. ⏎  ⏎ In particular, the latter fix improves decode performance measurably on a few workloads (particularly `--async-scheduling` and ModelRunner V2): ⏎ 1. DeepSeek V4, ISL=1024, OSL=512, TP=8,  …[truncated]

### L3-e376d45e82  (L3, 2026-08-26, sha e376d45e82cb, PR #53853)
TITLE: [Config] Delegate PCP compatibility checks to PCP manager (#53853)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+0/-4)
BODY: ## Purpose ⏎  ⏎ The generic Model Runner V2 feature check currently treats PCP as unsupported whenever the model does not use MLA. This duplicates an in-tree implementation restriction and makes it a global configuration restriction. ⏎  ⏎ Some out-of-tree hardware plugins already support GQA with PCP, such as [vLLM Ascend support in vllm-project/vllm-ascend#14023](https://github.com/vllm-project/vllm-ascend/pull/14023). Because the generic validation run …[truncated]

### L3-88e1b11313  (L3, 2026-08-26, sha 88e1b1131319, PR #53606)
TITLE: [Perf] Tune FlashInfer all-reduce thresholds for single-node TP8 on SM103 (#53606)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/distributed/test_comm_ops.py (+1/-1); vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+1/-1); vllm/distributed/device_communicators/all_reduce_utils.py (+1/-0)
LABELS: ready, torch.compile
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ ### Config ⏎  ⏎ K3, TP8, random dataset, 8192 input / 1024 output, `--ignore-eos`. ⏎ Concurrency ~= tokens per decode step, and all-reduce bytes = tokens x 7168 x 2. ⏎  ⏎ | conc | AR MiB | ITL main | ITL tuned | ITL Δ | TPOT main | TPOT tuned | TPOT Δ | tok/s Δ | ⏎ |---:|---:|---:|---:|---:|---:|---:|---:|---:| ⏎ | 128  | 1.75 | 193.54 | 195.81 | -1.17% | 74.30 | 74.41 | -0.14% | -0.14% | ⏎ | 256 | 3.50 | 268.37 | 260.36 | +2.98% | 102.88 | …[truncated]

### L3-044b05220e  (L3, 2026-08-26, sha 044b05220ee9, PR #53819)
TITLE: [Kernel][Perf] Tune fused_moe FP8 config for Qwen3.5 on L40S (+7%) (#53819)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json (+171/-0)
LABELS: ready, qwen
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Add a tuned Triton fused MoE configuration for the E=256,N=512 FP8 W8A8 shape on a single NVIDIA L40S for Qwen3.5. ⏎  ⏎ This is a configuration-only performance change and does not change model numerics or model code. ⏎  ⏎ ## Test Result ⏎  ⏎ Kernel time comparison on NVIDIA L40S (Triton 3.6.0): ⏎  ⏎ | M | Default (us) | Tuned (us) | Speedup | ⏎ |---:|---:|---:|---:| ⏎ | 2 | 31.92 | 29.98 | +6.08% | ⏎ | 4 | 78.61 | 76.57 | +2.60% | ⏎ | 8 | 276.79 | 259.57 |  …[truncated]

### L3-b1fbbc2ade  (L3, 2026-08-26, sha b1fbbc2ade51, PR #53515)
TITLE: BugFix(PCP): use persistent input buffers for PIECEWISE CUDA graphs (#53515)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_pcp_manager.py (+109/-0); vllm/v1/worker/gpu/model_runner.py (+15/-3); vllm/v1/worker/gpu/pcp_manager.py (+71/-22)
LABELS: bug, ready, nvidia, mrv2
BODY: ## Summary ⏎  ⏎ Fix PIECEWISE CUDA graph execution with PCP by using persistent PCP input buffers during graph capture and sizing graph dispatch from the largest real rank-local PCP batch before padding. ⏎  ⏎ ## Implementation ⏎  ⏎ - Decoder graph capture now uses `PCPManager.input_buffers` when PCP is active, ensuring capture and runtime partitioning share persistent `input_ids`, `positions`, and `is_padding` buffers. ⏎ - `GPUModelRunner` obtains the graph di …[truncated]

### L3-2267d3b112  (L3, 2026-08-26, sha 2267d3b1124a, PR #53705)
TITLE: [AttentionBackend][HPC-ops] update hpc rope norm to support stride kv cache (#53705)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/hpc/rope_norm.py (+202/-17)
LABELS: ready
BODY: …ache ⏎  ⏎  ⏎  ⏎  ⏎ ## Purpose ⏎ Fix hpc rope norm not support stride kv cache bug. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-28b484e5d7  (L3, 2026-08-26, sha 28b484e5d741, PR #52185)
TITLE: [Model] Pixtral: use packed multimodal encoder attention (#52185)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_pixtral.py (+34/-0); vllm/model_executor/models/pixtral.py (+123/-71)
LABELS: ready, multi-modality, mistral
ISSUES: #52180 [Performance]: Improve Pixtral vision attention scaling for batched images
BODY: ## Purpose ⏎  ⏎ Fixes #52180 ⏎  ⏎ Pixtral currently concatenates all image patch sequences in an encoder batch. When xFormers is unavailable, it constructs a dense block-diagonal mask and applies SDPA to the combined sequence, causing latency and memory use to scale poorly with the number of images. ⏎  ⏎ This PR routes both Pixtral vision implementations through vLLM's `MMEncoderAttention` with per-image cumulative sequence lengths. It: ⏎  ⏎ - preserves  …[truncated]

### L3-2ab187430b  (L3, 2026-08-26, sha 2ab187430b6b, PR #53884)
TITLE: [Bugfix] Make Gemma4 MTP suppress_tokens masking CUDA-graph-safe (#53884)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/gemma4_mtp.py (+14/-3)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎  ⏎ `Gemma4MTP.compute_logits` masks suppressed tokens by indexing logits with a ⏎ Python list taken from the draft's `generation_config` (`suppress_tokens`). ⏎ List indexing issues an unpinned host->device copy, which is illegal during ⏎ CUDA graph capture. ⏎  ⏎ Since the V2 model runner became the default for non-MoE architectures, the ⏎ speculator captures the drafter prefill routine — including `compute_logits` — ⏎ into FULL CUDA graphs. E …[truncated]

### L3-76cfe1cd88  (L3, 2026-08-26, sha 76cfe1cd88d3, PR #53942)
TITLE: [Kimi K3 Perf] Optimize `eh_proj` linear calculation, 12.9 ~ 25.2% kernel performance improvement (#53942)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_bf16_skinny_gemm.py (+9/-1); vllm/models/kimi_k3/nvidia/low_latency_gemm.py (+8/-0); vllm/models/kimi_k3/nvidia/mtp.py (+9/-1)
LABELS: ready, kimi, k3
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Enabling m=1 and m=2 for low latency gemm ⏎  ⏎ Originally we use `nn.Linear`, it is not `LinearBase`, so won't work in `enable_kimi_k3_low_latency_gemm` ⏎  ⏎ Now we enable it ⏎  ⏎ ## Test ⏎  ⏎ Acc covered in current unit tests ⏎  ⏎ Perf can be seen in this AI generated script ⏎  ⏎ ```bash ⏎ # SPDX-License-Identifier: Apache-2.0 ⏎ # SPDX-FileCopyrightText: Copyright contributors to the vLLM project ⏎ """Sweep cuBLAS vs CuTe for the Kimi-K3 MTP eh_p …[truncated]

### L3-1a085dadf2  (L3, 2026-08-26, sha 1a085dadf254, PR #51040)
TITLE: [ROCm][K3] Extend FP8 asm MLA prefill to non-divisor small head counts (#51040)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+145/-24); tests/kernels/attention/test_rocm_aiter_mla_fp8_prefill.py (+104/-21)
LABELS: rocm, ready, verified, k3
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Context ⏎  ⏎ The AITER FP8 MLA **prefill** path (`mla_prefill_ps_asm_fwd` + `mla_reduce_v1`) is gated on ⏎ `num_heads % 16 == 0`. Kimi-K3 has 96 heads over `kv_lora_rank=512` → **12 heads/rank at TP8**, ⏎ a non-divisor of 16, so its FP8 prefill falls back to the BF16 FMHA decompress path. That fallback ⏎ builds a bf16 working set not covered by the FP8 KV-pool accounting and exhausts the activation ⏎ arena at long context (K3 OOM'd at ~197k tokens,  …[truncated]

### L3-2dc53bae17  (L3, 2026-08-26, sha 2dc53bae1743, PR #53641)
TITLE: [AMD][BugFix] Add gpu_sync_allowed to ROCm AITER FA backend (#53641)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+9/-7)
LABELS: bug, rocm
BODY: ## Purpose ⏎ This https://github.com/vllm-project/vllm/pull/43107  added GPU<->CPU transfer checks  and turned it on in the dockerfile ⏎  ⏎ `VLLM_GPU_SYNC_CHECK=error` ⏎  ⏎ When running a unit test, I encountered the error: ⏎  ⏎ `RuntimeError: GPU<->CPU sync detected - avoid it or wrap with gpu_sync_allowed()` ⏎  ⏎ And when I checked the ROCM `AITER `FA backend, I saw that the context wasn't used even though a lot of the `gpu_sync_allowed` contexts were a …[truncated]

### L3-f18d0ba90d  (L3, 2026-08-26, sha f18d0ba90d97, PR #53650)
TITLE: [Doc] Add Granite 3.1 series to batch invariance tested models (#53650)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/features/batch_invariance.md (+2/-0)
LABELS: documentation, ready
BODY: ## Purpose ⏎  ⏎ Add the Granite 3.1 series to the list of models validated for batch invariance. ⏎  ⏎ This expands model coverage to both `GraniteMoeForCausalLM` (MoE, top-8 routing) and `GraniteForCausalLM` (dense). ⏎  ⏎ Related to #27433. ⏎  ⏎ ## Duplicate check ⏎  ⏎ I searched the open issues and pull requests and found no existing batch-invariance validation for the Granite 3.1 series. ⏎  ⏎ ## Test Plan ⏎  ⏎ - Hardware: NVIDIA H800 (SM90) ⏎ - vLLM commit: `6010c4013` (mai …[truncated]

### L3-61d4f56635  (L3, 2026-08-26, sha 61d4f56635e0, PR #53120)
TITLE: [Offloader] Offload submodules that make_layers never reaches (#53120)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/basic_correctness/test_cpu_offload.py (+59/-0); vllm/model_executor/models/interfaces.py (+15/-0); vllm/model_executor/offloader/base.py (+15/-0); vllm/model_executor/offloader/prefetch.py (+7/-1); vllm/model_executor/offloader/uva.py (+17/-3)
LABELS: ready, qwen
BODY: ## Purpose ⏎  ⏎ `--cpu-offload-params visual` silently offloaded nothing for multimodal models — no warning, no ⏎ error, just no effect. Investigating it turned up a gap in the offloader's common path rather than a ⏎ model-specific bug. ⏎  ⏎ `get_offloader().wrap_modules()` has exactly two callers: ⏎  ⏎ - `vllm/model_executor/models/utils.py:860` — inside `make_layers`, which only ever builds the ⏎   *decoder layer stack*. ⏎ - `vllm/model_executor/models/whisper.py: …[truncated]

### L3-080a66a69c  (L3, 2026-08-26, sha 080a66a69c6f, PR #53818)
TITLE: [Bugfix][ROCm] Capture CUDA graphs on the current stream (#53818)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/spec_decode/gemma4.py (+2/-1); vllm/v1/worker/encoder_cudagraph.py (+5/-1); vllm/v1/worker/gpu/cudagraph_utils.py (+4/-1)
LABELS: bug, rocm, speculative-decoding, ready, nvidia, mrv2
BODY: ## Purpose ⏎  ⏎ `torch.cuda.graph()` lazily creates its own class-level internal capture stream ⏎ when no `stream=` is passed. Three capture sites still rely on that fallback, so ⏎ the graph gets recorded on a stream that never ran warmup, and any backend that ⏎ keeps per-stream state misses its lookup during capture. ⏎  ⏎ For the main site this also contradicts its own enclosing context manager. MRV2 ⏎ captures inside `with graph_capture(device=...)`, w …[truncated]

### L3-657f9b9ce2  (L3, 2026-08-26, sha 657f9b9ce241, PR #53838)
TITLE: [ROCm][DSV4][Perf] Fuse DeepSeek V4 C4 compressor GEMMs (#53838)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/test_deepseek_v4_rocm_compressor_gemm_fusion.py (+141/-0); vllm/models/deepseek_v4/amd/model.py (+7/-0); vllm/models/deepseek_v4/amd/rocm.py (+73/-0)
LABELS: performance, rocm, ready, deepseek, DSv4
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ DeepSeek V4 C4 target-model layers run the main compressor projection and ⏎ indexer-compressor projection with the same input and K dimension. On the ROCm ⏎ path these GEMMs execute serially. This change uses a single-stream path that: ⏎  ⏎ - concatenates the `[2048, 7168]` and `[512, 7168]` BF16 weights once after ⏎   loading; ⏎ - replaces two FP32-output `torch.mm` calls with one ⏎   `[M, 7168] x [7168, 2560]` call; ⏎ - returns zero-copy output vie …[truncated]

### L3-f8d38fbc87  (L3, 2026-08-27, sha f8d38fbc879c, PR #49994)
TITLE: [EC Connector] EC Offloading Connector use events instead of StepTracker (#49994)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/ec_connector/__init__.py (+0/-0); tests/v1/ec_connector/unit/cpu/scheduler/test_embedding_cache.py (+30/-1); tests/v1/ec_connector/unit/cpu/scheduler/test_scheduler.py (+160/-178); tests/v1/ec_connector/unit/cpu/scheduler/test_step_tracker.py (+0/-162); tests/v1/ec_connector/unit/cpu/worker/test_worker.py (+342/-71); tests/v1/ec_connector/unit/test_ec_cpu_connector.py (+4/-3); tests/v1/ec_connector/unit/test_metadata.py (+13/-13); tests/v1/ec_connector/unit/utils.py (+70/-0); vllm/distributed/ec_transfer/ec_connector/cpu/common.py (+36/-5); vllm/distributed/ec_transfer/ec_connector/cpu/connector.py (+16/-0); (+5 more)
LABELS: documentation, structured-output, speculative-decoding, ready, ci/build, v1, cpu, kv-connector, mrv2
BODY: ## Purpose ⏎  ⏎ The CPU encoder-cache offloading connector (`ECCPUConnector`) gated cache-entry transitions on a step-count heuristic (`StepTracker`): a save was marked ready, and a load unpinned, after a fixed number of engine steps — a fragile proxy for DMA completion that can hold blocks pinned too long. ⏎  ⏎ This PR replaces it with a real CUDA-event completion signal: ⏎  ⏎ - The worker brackets each batched GPU copy with start/end events on a CUDA …[truncated]

### L3-b3af042abd  (L3, 2026-08-27, sha b3af042abd8f, PR #53869)
TITLE: Bugfix: use PCP slot mappings for PIECEWISE capture (#53869)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP1-PCP4-EP.yaml (+1/-1); tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP2-PCP2-EP.yaml (+1/-1); tests/v1/cudagraph/test_cudagraph_manager.py (+53/-0); vllm/v1/worker/gpu/cudagraph_utils.py (+8/-1); vllm/v1/worker/gpu/model_runner.py (+1/-0)
LABELS: bug, ready, nvidia, mrv2
BODY: ## Summary ⏎  ⏎ - build PIECEWISE capture slot mappings through `PCPManager` so dummy attention metadata follows the rank-major PCP layout ⏎ - cover the PCP2 mixed-batch capture shape with a focused regression test ⏎ - run both PCP GSM8K end-to-end configurations with PIECEWISE CUDA graphs ⏎  ⏎ ## Root cause ⏎  ⏎ PIECEWISE capture created a global dummy slot mapping with `num_tokens` entries. **The PCP with `TRITON_MLA` preparation path consumes a rank-m …[truncated]

### L3-32ad1400d7  (L3, 2026-08-27, sha 32ad1400d7fa, PR #53540)
TITLE: [ROCm][Perf] Fuse SWA q/kv RMSNorm and q FP8 group quant for DeepSeek-V4 (#53540)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/core/test_rocm_aiter_ops.py (+76/-1); vllm/_aiter_ops.py (+94/-0); vllm/models/deepseek_v4/amd/rocm.py (+96/-0); vllm/models/deepseek_v4/attention.py (+67/-11)
LABELS: rocm, deepseek, quantization, DSv4
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ On the ROCm path of DeepSeek-V4, every decode step's SWA token-insertion pipeline runs: ⏎  ⏎ 1. vLLM's Triton `fused_q_kv_rmsnorm` (norms the q-lora and kv latents to bf16). ⏎ 2. aiter per-1x128 dynamic quant of the bf16 q-lora (input quant of the block-scaled FP8 `wq_b` GEMM). ⏎ 3. the same quant a second time in the indexer's `wq_b`, which shares the q-lora with the attention layer. ⏎  ⏎ This PR replaces 1+2 with a single aiter HIP kern …[truncated]

### L3-07242faa46  (L3, 2026-08-27, sha 07242faa4667, PR #54012)
TITLE: [Attention][DCP] Use FlashInfer native CP for MLA decode (#54012)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer
FILES: vllm/v1/attention/backends/mla/flashinfer_mla.py (+52/-21); tests/v1/attention/test_flashinfer_mla_dcp.py (+64/-1); tests/v1/attention/test_mla_backends.py (+2/-18)
LABELS: ready, nvidia
BODY: ## Summary ⏎  ⏎ FlashInfer 0.6.17 exposes native context-parallel masking through CuTeDSL MLA decode kernel. This PR adds MLA DCP support through FlashInfer (limit to cuteDSL path only) ⏎  ⏎ ### Relationship to existing work ⏎  ⏎ This is not a duplicate of the existing speculative/DCP PRs: ⏎  ⏎ - #52188 added vLLM's DSpark/DCP slot mapping and draft metadata. This PR uses that plumbing; it does not reimplement it. ⏎ - #53139 removed the legacy manual Flas …[truncated]

### L3-478ec3ea41  (L3, 2026-08-27, sha 478ec3ea4135, PR #54021)
TITLE: [Bugfix][KV Offload] Handle padded GPU cache storage (#54021)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/simple_kv_offload/test_worker.py (+32/-2); vllm/v1/simple_kv_offload/worker.py (+7/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ Fix the `nvidia-h100-lm-eval-kv-offload-large` startup regression introduced by #51718. The failure reproduces on upstream `main` in [Build #85765](https://buildkite.com/vllm/ci/builds/85765) and [Build #85757](https://buildkite.com/vllm/ci/builds/85757). ⏎  ⏎ `SimpleCPUOffloadWorker.register_kv_caches()` reshaped the entire 4-KiB-rounded backing storage into logical cache regions. DeepSeek-V4 has 2,816 trailing alignment bytes, so its 33 …[truncated]

### L3-c7b0467c25  (L3, 2026-08-27, sha c7b0467c2524, PR #50932)
TITLE: buffer size insuffient Dspark sd for FlashInfer MNNVL allreduce (#50932)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/distributed/device_communicators/flashinfer_all_reduce.py (+21/-2); vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py (+11/-0)
LABELS: bug, ready, nvidia, dflash
ISSUES: #50877 [Bug]: DSpark speculative decoding triggers FlashInfer MNNVL allreduce "buffer size insufficient" via draft model's embed_input_ids (TP8, GB200 NVL72)
BODY: fixes [#50877](https://github.com/vllm-project/vllm/issues/50877) ⏎  ⏎ ## Purpose ⏎ FlashInferAllReduce.should_use_fi_ar gates on: ⏎  ⏎     self.max_num_tokens = max_workspace_size // (hidden_dim * element_size) ⏎  ⏎ max_workspace_size is the size of the whole MNNVL allocation (2 MB for TP8). But ⏎ the MNNVL backend is Lamport-based and rotates through NUM_LAMPORT_BUFFERS=3 ⏎ buffers, so only ~1/3 of the budget backs any single all-reduce: ⏎  ⏎   budget     …[truncated]

### L3-d21c5b50a1  (L3, 2026-08-27, sha d21c5b50a1d3, PR #53878)
TITLE: [Perf][GLM5.2] Fuse sparse MLA Q concatenation with head padding (#53878)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_sparse
FILES: vllm/v1/attention/backends/mla/flashmla_sparse.py (+22/-4); tests/kernels/test_concat_mla_q.py (+11/-0); tests/v1/attention/test_sparse_mla_backends.py (+4/-2)
LABELS: ready, glm
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ [Perf][GLM5.2] Fuse sparse MLA Q concatenation with head padding ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve zai-org/GLM-5.2-FP8 \ ⏎   --load-format instanttensor \ ⏎   --tensor-parallel-size 8 \ ⏎   --gpu-memory-utilization 0.9 \ ⏎   -ep \ ⏎   --trust-remote-code  --max-num-batched-tokens 16384 \ ⏎   --attention-backend FLASHMLA_SPARSE \ ⏎   --port 8000 --no-enable-prefix-caching --max-model-len 131072 ⏎  ⏎ ``` ⏎ ## Test Result ⏎ ``` ⏎ vllm bench serve \ ⏎    …[truncated]

### L3-4aab2b0ebe  (L3, 2026-08-27, sha 4aab2b0ebed2, PR #53183)
TITLE: [Model Runner V2] Use MRV2 for all models by default (#53183)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+4/-1); tests/model_executor/layers/test_fused_shared_expert.py (+3/-0); tests/test_config.py (+54/-31); vllm/compilation/decorators.py (+9/-0); vllm/config/vllm.py (+52/-85)
LABELS: ready, torch.compile, nvidia, mrv2
BODY: Apart from some specific models in the ROCm case, until MRV2 performance is fixed for those. ⏎  ⏎ MRV1 is still used for certain specific features that aren't yet supported in MRV2.

### L3-75dea9b4ae  (L3, 2026-08-27, sha 75dea9b4ae9e, PR #53755)
TITLE: [Bugfix] Update FlashMLA for sparse decode workspace fix (#53755)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashmla_build
FILES: cmake/external_projects/flashmla.cmake (+1/-1)
LABELS: bug, ready, ci/build
ISSUES: #53413 [Bug]: GLM-5.2 FP8 on 8×H200 dies with runtime CUDA OOM in sparse_decode_fwd
BODY: ## Purpose ⏎  ⏎ Fixes #53413 by integrating the FlashMLA fix from ⏎ vllm-project/FlashMLA#19. ⏎  ⏎ This does not duplicate #49357 or #50668. #49357 bounds the allocation by ⏎ chunking mixed batches in vLLM, with an acknowledged throughput tradeoff, while ⏎ #50668 bounds the separate sparse-prefill path. This PR updates FlashMLA so the ⏎ no-split sparse-decode path does not allocate the unused split-KV accumulators ⏎ at all. ⏎  ⏎ OpenAI Codex assisted with analysis, im …[truncated]

### L3-7d5769b2d9  (L3, 2026-08-27, sha 7d5769b2d9cf, PR #53685)
TITLE: [Perf][DSv4] Use native CUDA SwiGLU clamp kernel for Humming MoE (throughput +1.40%) (#53685)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/core/test_activation.py (+15/-0); vllm/model_executor/layers/fused_moe/utils.py (+2/-0)
LABELS: performance, ready, nvidia, DSv4
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Use native CUDA SwiGLU clamp kernel for Humming MoE ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \ ⏎   --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \ ⏎   --enable-expert-parallel --tensor-parallel-size 4 \ ⏎   --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \ ⏎   --enable-auto-tool-choice --reasoning-parser deepseek_v4 \ ⏎   --no-enable-prefix-caching --max-num-batched-tokens 16384 \ ⏎   --loa …[truncated]

### L3-d262964e8c  (L3, 2026-08-27, sha d262964e8c83, PR #54020)
TITLE: Upgrade tpu-inference to v0.28.0 (#54020)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/tpu.txt (+1/-1)
LABELS: ci/build
BODY: ## Purpose ⏎ Upgrade tpu-inference to latest stable release v0.28.0 ⏎  ⏎ ## Test Plan ⏎  ⏎ Verified on tpu-inference CI.  ⏎  ⏎ ## Test Result ⏎ Success.  ⏎  ⏎ --- ⏎ [details omitted]

### L3-9650dc73e2  (L3, 2026-08-27, sha 9650dc73e249, PR #53443)
TITLE: [CI/Build][Hardware][NVIDIA] Add opt-in Rubin Docker builds (#53443)
SOURCES: dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+255/-22); docker/versions.json (+3/-0); docs/assets/contributing/dockerfile-stages-dependency.png (+0/-0); docs/contributing/dockerfile/dockerfile.md (+4/-0); docs/getting_started/installation/gpu.cuda.inc.md (+83/-2); tools/build_triton_from_source.sh (+43/-0); tools/install_triton_from_source.sh (+41/-0)
LABELS: documentation, ready, ci/build, nvidia
BODY: ## What this PR does and impact ⏎  ⏎ Tracks #49735. ⏎  ⏎ - Adds an opt-in `INSTALL_RUBIN_PRERELEASE=true` Docker path for CUDA 13.4 ⏎   and CUDA 13.5 Rubin builds on ARM64 and AMD64. The switch defaults to ⏎   `false`, so standard Blackwell images retain the public dependency path. ⏎ - Adds generic, opt-in source-Triton build/install scripts controlled by ⏎   `TRITON_INSTALL_FROM_SOURCE_REPO` and ⏎   `TRITON_INSTALL_FROM_SOURCE_REVISION`. This control is independe …[truncated]

### L3-f79a2f5582  (L3, 2026-08-27, sha f79a2f5582cb, PR #53797)
TITLE: Add support for loading dflash2 model in speculators format (#53797)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/transformers_utils/configs/speculators/algos.py (+24/-0); vllm/transformers_utils/configs/speculators/base.py (+2/-0)
LABELS: speculative-decoding, ready, dflash
BODY: ## Purpose ⏎  ⏎ Add speculators format loading logic for dflash2.  ⏎  ⏎ Note: we use `dflash2` in the speculators config to denote the algorithm, but overwrite with `dflash` in vllm to match vllm conventions.  ⏎  ⏎ ## Test Plan ⏎  ⏎ Successfully served a model produced by https://github.com/vllm-project/speculators/pull/1006 in vllm using this change.  ⏎  ⏎ ## Test Result ⏎  ⏎ Speculator served.  ⏎  ⏎ --- ⏎ [details omitted]

### L3-9818bb3db8  (L3, 2026-08-27, sha 9818bb3db8fd, PR #54088)
TITLE: [Kimi Perf] Tune hopper low latency gemm kernel, 4%~97% performance improvement (#54088)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_bf16_skinny_gemm.py (+77/-28); vllm/models/kimi_k3/nvidia/low_latency_gemm.py (+98/-7)
LABELS: ready, kimi, k3
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎  ⏎ Tune hopper low latency gemm kernel ⏎  ⏎ ## Test ⏎  ⏎ Acc covered in current unit tests ⏎  ⏎ Perf can be seen in this AI generated script ⏎  ⏎ ```py ⏎ from statistics import geometric_mean ⏎  ⏎ import torch ⏎ import torch.nn.functional as F ⏎  ⏎ from vllm.models.kimi_k3.nvidia import low_latency_gemm as gemm ⏎ from vllm.triton_utils import triton ⏎  ⏎ COPIES = 8  # Distinct weights approximate different model layers. ⏎ WARMUP = 20 ⏎ REPETITIONS = 100 …[truncated]

### L3-6ec92bcbc8  (L3, 2026-08-27, sha 6ec92bcbc8ef, PR #54015)
TITLE: [Kimi-K3] Merge MLA gate into QKV-A projection (#54015)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/linear.py (+90/-1); vllm/models/kimi_k3/nvidia/mla.py (+167/-94); vllm/models/kimi_k3/nvidia/model.py (+12/-4); vllm/models/kimi_k3/nvidia/mtp.py (+11/-4)
LABELS: kimi, k3
BODY: ## Purpose ⏎  ⏎ This PR merges MLA's `g_proj` (used for output gating) into QKV_a projection to create a larger GEMM. We don't expect this to have a large impact as the original QKV_a gemm is already quite large. The main purpose of this PR is to prepare for the upcoming fused AG-GEMM (used in SP), which benefits significantly from having a larger GEMM. ⏎  ⏎ What makes this PR a bit more complicated is that currently we have a low-latency optimizatio …[truncated]

### L3-674c284dbd  (L3, 2026-08-27, sha 674c284dbd20, PR #54056)
TITLE: Fix Humming MoE activation_output aliasing (#54056)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py (+9/-0)
LABELS: ready
BODY: Restores gsm8k accuracy for Kimi K3 at high lm_eval concurrencies: ⏎  ⏎ ``` ⏎ |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr| ⏎ |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:| ⏎ |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9682|±  |0.0048| ⏎ |     |       |strict-match    |     5|exact_match|↑  |0.9689|±  |0.0048| ⏎ ``` ⏎  ⏎ The drop in accuracy is caused by the activation kernel receiving al …[truncated]

### L3-a18dbe49ad  (L3, 2026-08-27, sha a18dbe49add8, PR #53839)
TITLE: [Doc] Add EXAONE-4.0-1.2B to batch invariance tested models (#53839)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: docs/features/batch_invariance.md (+1/-0)
LABELS: documentation, ready
BODY: ## Purpose ⏎  ⏎ Validate the EXAONE 4.0 series under `VLLM_BATCH_INVARIANT=1` and add it to the ⏎ tested models list in `docs/features/batch_invariance.md`. Part of #27433. ⏎  ⏎ Updated from the original single-model PR to cover the full series, per review ⏎ feedback. This is the first EXAONE-family entry, and the first on the list with ⏎ per-layer hybrid sliding/global attention. ⏎  ⏎ ## Test Plan ⏎  ⏎ - `EXAONE-4.0-1.2B` - RTX 4090 (SM 8.9), driver 590.48 …[truncated]

### L3-5bbc58c0ad  (L3, 2026-08-27, sha 5bbc58c0ad0f, PR #54111)
TITLE: [Bugfix] Remove race in fused groupwise RMSNorm quantization (#54111)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/quantization/fused_kernels/layernorm_utils.cuh (+21/-37); tests/kernels/core/test_fused_quant_layernorm.py (+14/-1)
LABELS: bug, ready, ci-failure, quantization
BODY: ## Purpose ⏎  ⏎ Fix three flaky fused quantized RMSNorm failures observed in the H200 kernel CI job: ⏎ https://buildkite.com/vllm/ci/builds/85850#01a04456-3602-4e62-b405-0c986c85cdc3 ⏎  ⏎ The two group-64 scale failures were caused by a real shared-memory race in `warpReduceMaxSpecialized`. Each reduction stage read values written by other lanes in the preceding stage without synchronization. CUDA racecheck reported thousands of hazards for the failing spe …[truncated]

### L3-e6bfe03ad7  (L3, 2026-08-27, sha e6bfe03ad73a, PR #52227)
TITLE: Count store offers, not lookups, for CPU offload store_threshold (#52227)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_offload/cpu/test_manager.py (+18/-22); vllm/v1/kv_offload/cpu/manager.py (+13/-8); vllm/v1/kv_offload/cpu/spec.py (+4/-3)
LABELS: bug, rocm
BODY: ## Purpose ⏎  ⏎ store_threshold >= 2 asks CPUOffloadingManager to admit a block to the CPU offload tier only after it has been seen that many times. The counter was bumped in lookup(), and the callers make that unreachable for a cold prefix: ⏎  ⏎ - OffloadingConnectorScheduler._maximal_prefix_lookup breaks on the first MISS (offloading/scheduler.py:629), so on any given pass only the first not-yet-cached key of a prefix is ever looked up. ⏎ - _lookup  …[truncated]

### L3-de9250ac9e  (L3, 2026-08-27, sha de9250ac9e9b, PR #53785)
TITLE: [Attention] Enable masked MHA for GLM-5 head dimensions (#53785)
SOURCES: path_core, symbol_pickaxe, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_build, L3.mla.common_v1
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/model_executor/layers/attention/mla_attention.py (+126/-34); vllm/model_executor/layers/attention/sparse_mla_attention.py (+18/-7); benchmarks/attention_benchmarks/benchmark.py (+32/-6); benchmarks/attention_benchmarks/configs/mla_sparse_masked_mha_vs_mqa_glm5.yaml (+59/-0); benchmarks/attention_benchmarks/mla_runner.py (+32/-0); tests/model_executor/layers/test_mla_short_prefill_indexer.py (+75/-7); tests/v1/attention/test_mla_backends.py (+119/-0); tests/v1/attention/test_sparse_mla_backends.py (+42/-5); vllm/models/deepseek_v32/attention.py (+23/-14)
LABELS: performance, ready, ci/build, deepseek, glm
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Depends on https://github.com/vllm-project/flash-attention/pull/179. ⏎  ⏎ This PR enables the sparse-MLA masked-MHA prefill path for the GLM-5 256/256 QK/V head dimensions: ⏎  ⏎ - Temporarily pins FlashAttention to `MatthewBonanni/flash-attention@8f844a0`, which contains head-dim-256 `mask_mod` support. ⏎ - Admits the GLM-5 dimensions `(64 heads, KV rank 512, QK 192+64, V 256)` to the existing masked-MHA route. ⏎ - Keeps the general FA4 head-dim- …[truncated]

### L3-7c877062ac  (L3, 2026-08-27, sha 7c877062acab, PR #50572)
TITLE: [kernel] Integrate FlashInfer BF16 CuTeDSL Low Latency GEMM (#50572)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+76/-0); tests/kernels/test_flashinfer_bf16_gemm.py (+49/-0); vllm/config/kernel.py (+6/-2); vllm/model_executor/kernels/linear/__init__.py (+11/-6); vllm/model_executor/layers/linear.py (+9/-2); vllm/model_executor/layers/utils.py (+172/-2); vllm/model_executor/warmup/kernel_warmup.py (+26/-11)
LABELS: performance, ready, nvidia, quantization, kimi, k3
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: **this is currently blocked by the Flashinfer next 0.6.18 release** ⏎  ⏎ ## Purpose ⏎  ⏎ Integrate the FlashInfer CuTeDSL BF16 low-latency dense GEMM from ⏎ [flashinfer-ai/flashinfer#4266](https://github.com/flashinfer-ai/flashinfer/pull/4266). ⏎  ⏎  ⏎ This PR: ⏎  ⏎ - adds `flashinfer_cutedsl` as an opt-in `--linear-backend`; ⏎ - uses `flashinfer.mm_bf16(..., backend="cute-dsl")` for eligible unquantized ⏎   BF16 GEMMs on SM100-family GPUs; ⏎ - limits the Fla …[truncated]

### L3-06569a8696  (L3, 2026-08-28, sha 06569a869607, PR #53097)
TITLE: [ROCm][Quantization][MOE] Enable fused shared experts for block-quantized FP8 (#53097)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/layers/test_fused_shared_expert.py (+133/-0); vllm/model_executor/layers/quantization/utils/config_utils.py (+44/-0)
LABELS: rocm, quantization
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ **To enable fp8 models such as DeepSeek-R1-0528 ("n_shared_experts": 1, "quant_method": "fp8")** ⏎  ⏎ On top of PR [#51695](https://github.com/vllm-project/vllm/pull/51695). ⏎  ⏎ PR #51695 (`88b2bff2c63`) centralised the "may this model fuse its shared experts into the routed grouped GEMM?" decision into `is_shared_expert_quant_fse_compatible`. That helper implements `None`, `DeepseekV4FP8Config` and `QuarkConfig`, and closes with a TOD …[truncated]

### L3-6f91e3d953  (L3, 2026-08-28, sha 6f91e3d9537e, PR #53412)
TITLE: [Perf] Split xdrope_positions H2D copy into per-row transfers (#53412)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu_model_runner.py (+12/-5)
LABELS: mrv1-only
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ PR #51841 identified a subtle host-run-ahead killer: `copy_(..., non_blocking=True)` on a **strided view of a pinned buffer** silently falls back to a pageable intermediate, and pageable H2D synchronizes the stream regardless of `non_blocking`. That PR fixed `mrope_positions`. Eight lines below the fix, `xdrope_positions` has the identical setup and the identical bug. ⏎  ⏎ Layout is the same: ⏎  ⏎ ```python ⏎ # gpu_model_runner.py, buffer alloc …[truncated]

### L3-6f7df92a8e  (L3, 2026-08-28, sha 6f7df92a8e6c, PR #51471)
TITLE: [CPU][MLA] Fix prefill backend selection so MLA runs end-to-end on CPU (#51471)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/_custom_ops.py (+25/-0); vllm/model_executor/layers/attention/mla_attention.py (+12/-0); vllm/v1/attention/backends/mla/prefill/cpu_native.py (+0/-61); vllm/v1/attention/backends/mla/prefill/cpu_sdpa.py (+137/-0); vllm/v1/attention/backends/mla/prefill/registry.py (+1/-3); vllm/v1/attention/backends/mla/prefill/selector.py (+4/-2); vllm/v1/attention/ops/merge_attn_states.py (+73/-0); tests/v1/attention/test_cpu_mla_backend.py (+164/-2); tests/v1/attention/test_mla_prefill_selector.py (+26/-1)
LABELS: cpu, quantization
BODY: ## Purpose ⏎  ⏎ Make DeepSeek-V2-Lite run end-to-end on CPU when MLA is enabled, including the ⏎ contextful prefill path needed by external KV reloads (for example LMCache). ⏎ This is the missing follow-up to #49453: that PR brought up the CPU MLA decode ⏎ path and the basic CPU MLA backend plumbing, but it still left CPU MLA unable ⏎ to run real external-hit prefill flows. ⏎  ⏎ ## Why #49453 was insufficient ⏎  ⏎ #49453 added the following pieces: ⏎  ⏎ - `C …[truncated]

### L3-06cccf8730  (L3, 2026-08-28, sha 06cccf8730a1, PR #53409)
TITLE: [Bugfix] Fix int32 token offset overflow in fused SiLU block quant (#53409)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu (+1/-1); tests/kernels/core/test_fused_silu_mul_block_quant.py (+25/-0)
LABELS: bug, ready, quantization
ISSUES: #53390 [Bug]: int32 token-offset overflow in silu_and_mul_per_block_quant (act_quant fusion) — illegal memory access above ~2^31/(2*intermediate_size) batched tokens
BODY: ## Purpose ⏎  ⏎ Fixes #53390. ⏎  ⏎ `silu_and_mul_per_block_quant` used a 32-bit `token_idx` when computing ⏎ global input and output row offsets. For sufficiently large token batches, ⏎ `token_idx * (2 * hidden_size)` overflowed `INT32_MAX`, wrapped the input ⏎ pointer, and caused an illegal CUDA memory access. ⏎  ⏎ Widen `token_idx` to `int64_t`, matching the sibling activation kernels. This ⏎ keeps the row-offset arithmetic in 64 bits without widening unrelated ⏎ th …[truncated]

### L3-ffe690eca9  (L3, 2026-08-28, sha ffe690eca956, PR #47625)
TITLE: [MM][CG] Support ViT full CUDA graph for Idefics3 and SmolVLM (#47625)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_vit_cudagraph.py (+14/-0); vllm/model_executor/models/idefics2_vision_model.py (+19/-13); vllm/model_executor/models/idefics3.py (+207/-20)
LABELS: documentation, ready, multi-modality, nvidia, verified
BODY: ## Purpose ⏎  ⏎ This PR adds ViT encoder CUDA Graph support for Idefics3 / SmolVLM as part of #38175. ⏎  ⏎ Idefics3 computes vision position IDs from the patch attention mask during the encoder forward pass. To make the encoder compatible with CUDA Graph capture, this PR precomputes the position IDs and passes them as fixed graph inputs. ⏎  ⏎ Changes: ⏎ - Implement `SupportsEncoderCudaGraph` for `Idefics3ForConditionalGeneration` ⏎ - Add CUDA Graph captu …[truncated]

### L3-2f1cba799e  (L3, 2026-08-28, sha 2f1cba799e02, PR #53141)
TITLE: [ROCm] remove VLLM_ROCM_USE_AITER_FP4_ASM_GEMM environment variable; make w4a4 use the preshuffle triton+asm by default (#53141)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/envs.py (+0/-6); tests/kernels/quantization/test_rocm_mxfp4.py (+6/-14); vllm/_aiter_ops.py (+1/-5); vllm/model_executor/kernels/linear/mxfp4/aiter.py (+2/-2)
LABELS: rocm, ready, verified
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` defaulted to `False`, routing MXFP4 dense linears to aiter's Triton `gemm_afp4wfp4`. That kernel spills (256×256×256 8-warp tile pegged at 256 VGPRs) and is slower than the asm path on every shape measured. ⏎  ⏎ The asm path avoids it entirely — `gemm_a4w4`, or `gemm_afp4wfp4_preshuffled_weight_scales` for M≤64 on tuned (N,K). This makes it unconditional on gfx950 and removes the env var. ⏎  ⏎  ⏎  ⏎   ## Results ⏎  ⏎    …[truncated]

### L3-46a83642f6  (L3, 2026-08-28, sha 46a83642f686, PR #54292)
TITLE: [Perf] Pin CPU tensors before non_blocking H2D in three MM paths (#54292)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/mm_encoder_attention.py (+1/-4); vllm/model_executor/models/kanana_v.py (+2/-4); vllm/model_executor/models/kimi_k25_vit.py (+2/-3)
LABELS: ready, kimi
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Route three multimodal-path `torch.from_numpy(...).to(device, non_blocking=True)` sites through `async_tensor_h2d`, which pins the host tensor first: ⏎  ⏎ - `MMEncoderAttention.maybe_compute_seq_lens` — called per FlashInfer MM encoder invocation to build `sequence_lengths` ⏎ - `CustomQwen2VLVE.forward` `cu_seqlens` in Kanana-V vision tower ⏎ - `MoonViT3dPretrainedModel.prepare_encoder_cudagraph_metadata` `merge_gather_idx` for Kimi-K2.5 visi …[truncated]

### L3-6d4562c59b  (L3, 2026-08-28, sha 6d4562c59b97, PR #54277)
TITLE: [Attention][DCP] Enable FlashInfer MLA for DSpark drafting (#54277)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer
FILES: vllm/v1/attention/backends/mla/flashinfer_mla.py (+7/-2); vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (+1/-11); vllm/v1/worker/gpu/spec_decode/speculator.py (+13/-0); tests/v1/attention/test_flashinfer_mla_dcp.py (+33/-8); tests/v1/attention/test_mla_backends.py (+2/-2); tests/v1/attention/test_mla_noncausal.py (+3/-2); tests/v1/spec_decode/test_eagle_draft_attn_metadata.py (+33/-0); vllm/v1/kv_cache_interface.py (+6/-3)
LABELS: speculative-decoding, ready, nvidia, mrv2, dflash
BODY: ## Summary ⏎  ⏎ - Recompute rank-local DCP sequence lengths from the draft model's current global sequence lengths before every shared draft metadata build. ⏎ - Share that metadata plumbing across EAGLE, MTP, DFlash, and DSpark, removing DFlash's duplicate preparation. ⏎ - Keep causal target MLA layers and non-causal DSpark draft layers in separate KV-cache groups by requiring the group to agree on `non_causal_multi_token_decode`. ⏎ - Advertise FlashInfer  …[truncated]

### L3-df14152ac6  (L3, 2026-08-28, sha df14152ac6b0, PR #54239)
TITLE: [Model] Support speculative decoding method for PLaMo3 (#54239)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/test_plamo3.py (+72/-0); vllm/model_executor/models/plamo3.py (+29/-10)
LABELS: speculative-decoding, ready
BODY: ## Purpose ⏎  ⏎ Add support for speculative decoding methods, such as EAGLE-3 and DFlash, to PLaMo3 by correctly handling auxiliary hidden states. ⏎  ⏎  ## Test Plan ⏎  ⏎  Run the unit test added in this PR. ⏎  ⏎ ```bash ⏎ python -m pytest tests/model_executor/test_plamo3.py -q ⏎ ``` ⏎  ⏎ ## Test Result ⏎  ⏎ ```bash ⏎ /python -m pytest tests/model_executor/test_plamo3.py -q     ⏎ .                                                                                   …[truncated]

### L3-d3d79ffc1e  (L3, 2026-08-28, sha d3d79ffc1e03, PR #50488)
TITLE: [Bugfix][Spec Decode] Capture the widest uniform decode batch by default (#50488)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/test_config.py (+441/-0); tests/v1/cudagraph/test_cudagraph_manager.py (+8/-10); tests/v1/spec_decode/test_dynamic_sd_cug.py (+35/-4); vllm/config/compilation.py (+3/-1); vllm/config/vllm.py (+122/-6); vllm/v1/worker/gpu/cudagraph_utils.py (+12/-9)
LABELS: bug, speculative-decoding, ready, torch.compile, nvidia, mrv2
BODY: # [Bugfix][Spec Decode] Capture the widest uniform decode batch by default ⏎  ⏎ ## Purpose ⏎  ⏎ This PR was four fixes; at a maintainer's request it is now one. The others are ⏎ #50531 (warmup lookahead reservation) and #50532 (uniform-decode dispatch). The ⏎ rejection-sampler argmax fix is dropped from this stack in favour of #50183, ⏎ which covers the same defect. ⏎  ⏎ `max_cudagraph_capture_size` defaults in units of *tokens* ⏎ (`min(max_num_seqs * 2, 512)`) whil …[truncated]

### L3-ae5b8e4a8d  (L3, 2026-08-28, sha ae5b8e4a8d77, PR #52849)
TITLE: [ROCm][PERF] Enable AITER PA gluon decode for MiniMax-M3 MTP and dense layers (#52849)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+280/-62); vllm/models/minimax_m3/amd/model.py (+39/-12); vllm/models/minimax_m3/amd/ops/sparse_pa.py (+155/-118); vllm/models/minimax_m3/amd/sparse_attention_msa.py (+3/-2); vllm/models/minimax_m3/common/sparse_attention.py (+97/-7)
LABELS: rocm, ready, minimax
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ The gluon paged-attention decode kernel handles multi-token query lengths, so ⏎ EAGLE3 speculative decoding no longer has to fall back to native vllm unified_attention.  ⏎  ⏎ ## Test Plan ⏎ Server cmd to run EAGLE3 speculative decoding  ⏎  ⏎ ``` ⏎ export VLLM_ENGINE_READY_TIMEOUT_S=3600 ⏎ export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 ⏎ export VLLM_USE_BREAKABLE_CUDAGRAPH=0 ⏎ export VLLM_ROCM_USE_AITER=1 ⏎ export VLLM_ROCM_USE_AITER_MOE=1 ⏎ expor …[truncated]

### L3-43196f2458  (L3, 2026-08-29, sha 43196f2458be, PR #54295)
TITLE: [Perf][MLA Sparse] Pin req_id_per_token before non_blocking H2D on XPU and ROCm (#54295)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+2/-1); vllm/v1/attention/backends/mla/xpu_mla_sparse.py (+2/-2)
LABELS: rocm, intel-gpu, ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ Route the two `buffer.copy_(torch.from_numpy(...), non_blocking=True)` sites on the sparse-MLA metadata path through `np_to_pinned_tensor` so the copy source is pinned: ⏎  ⏎ - `XPUMLASparseMetadataBuilder.build` `req_id_per_token_buffer` ⏎ - ROCm AITER sparse MLA metadata builder `req_id_per_token_buffer` ⏎  ⏎ ## Context ⏎  ⏎ Same class of stall #53491 adds a stricter check for: `torch.from_numpy(...)` returns a pageable CPU tensor, and `buffer.cop …[truncated]

### L3-99013d77d3  (L3, 2026-08-29, sha 99013d77d332, PR #53253)
TITLE: [Bugfix][Distributed] Gate cross-node MNNVL custom all-reduce by group capability (#53253)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/distributed/test_custom_all_reduce.py (+62/-0); vllm/distributed/device_communicators/custom_all_reduce.py (+57/-0)
LABELS: bug, ready, verified
ISSUES: #52907 Multi-node startup deadlock in in_the_same_node_as() gloo barrier at 2 nodes x TP-16 with the Ray executor (regression between 0.26.1rc1.dev78 and 0.26.1rc1.dev148)
BODY: ## Purpose ⏎  ⏎ Fixes #52907. ⏎  ⏎ A cross-node TP group on pre-Blackwell GPUs can enter ⏎ `torch.distributed._symmetric_memory.rendezvous()` during `CustomAllreduce` ⏎ construction even though the group cannot use MNNVL. The workers then block ⏎ before model-parallel initialization completes. ⏎  ⏎ ## Root cause ⏎  ⏎ PR #50000 removed the previous blanket multi-node CustomAllreduce disable and ⏎ allowed supported world sizes, including TP=16, to attempt the MNNVL path.  …[truncated]

### L3-6c18a54648  (L3, 2026-08-29, sha 6c18a5464852, PR #54299)
TITLE: [Perf] Avoid h2d copies from non-pinned CPU tensors (#54299)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+19/-33); vllm/model_executor/layers/pooler/tokwise/methods.py (+2/-1); vllm/model_executor/model_loader/utils.py (+11/-10); vllm/model_executor/models/ernie45_vl.py (+3/-5); vllm/model_executor/models/glm4_1v.py (+3/-3); vllm/model_executor/models/idefics2_vision_model.py (+4/-1); vllm/model_executor/models/phi4mm_audio.py (+3/-6); vllm/model_executor/models/qwen2_5_vl.py (+5/-7); vllm/model_executor/models/qwen2_vl.py (+3/-3); vllm/model_executor/models/qwen3_asr.py (+3/-2); (+4 more)
LABELS: qwen, nvidia, mrv2, glm
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: This can cause a stall that impacts overlap of gpu and cpu work. ⏎  ⏎ These instances were exposed by https://github.com/vllm-project/vllm/pull/53491.

### L3-b2f685834a  (L3, 2026-08-29, sha b2f685834a64, PR #54160)
TITLE: [Hy4] support Hy4-preview model (#54160)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: vllm/models/hy_v4/nvidia/flashmla_sparse.py (+283/-0); docs/models/supported_models.md (+2/-1); rust/src/chat/src/lib.rs (+2/-2); rust/src/chat/src/parser/reasoning/mod.rs (+3/-0); rust/src/chat/src/parser/tool/mod.rs (+3/-0); rust/src/chat/src/parser/tool/tests.rs (+4/-0); rust/src/chat/src/parser/unified.rs (+34/-2); rust/src/parser/src/reasoning/hy.rs (+7/-7); rust/src/parser/src/reasoning/mod.rs (+2/-2); rust/src/parser/src/tool/hy.rs (+250/-118); (+37 more)
LABELS: documentation, new-model, rocm, speculative-decoding, ready, ci/build, tool-calling, deepseek, cpu, nvidia
BODY: ## Purpose ⏎  ⏎ # Hy4 preview Usage Guide ⏎  ⏎   Hy4 preview is Tencent Hy's new-generation open-source Mixture-of-Experts language ⏎   model. Its backbone has 770B total parameters with 49B activated per token across ⏎   78 layers. The first layer uses a dense FFN; the other 77 layers use 256 routed ⏎   experts (top-8) and 1 shared expert. A native MTP layer adds 10B total parameters ⏎   (0.7B activated) for speculative decoding. ⏎  ⏎   On the architectur …[truncated]

### L3-129087ddab  (L3, 2026-08-29, sha 129087ddab2b, PR #45457)
TITLE: [Perf] Reuse topk SparseMatrix routing metadata in GPT-OSS MoE forward (#45457)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_gpt_oss_triton_kernels.py (+57/-1); vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py (+45/-14)
LABELS: ready, gpt-oss
ISSUES: #28986 [Feature]: Fused Kernel for GPT-OSS Router
DEEP_STUDY: deep-study performance PR ()
BODY: FIX #28986 ⏎  ⏎ Supersedes #35619 — thanks @banparth for the earlier exploration of this idea; the routing code has since been restructured, so this reimplements it against the current `experts/gpt_oss_triton_kernels_moe.py`. ⏎  ⏎ ## Purpose ⏎  ⏎ On the `triton_kernels` v3.6+ path, `topk()` returns a `SparseMatrix` whose construction already computes the bitmatrix and its routing metadata (sorted dispatch/combine indices, per-expert histograms) — the same ma …[truncated]

### L3-7f4793eaa3  (L3, 2026-08-29, sha 7f4793eaa335, PR #50611)
TITLE: [Nixl][PD] DCP support for MLA models   (#50611)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashinfer
FILES: vllm/config/kv_transfer.py (+9/-0); vllm/config/vllm.py (+78/-15); vllm/model_executor/layers/attention/mla_attention.py (+0/-3); vllm/v1/attention/backends/mla/flashinfer_mla.py (+0/-1); vllm/v1/worker/gpu/model_runner.py (+2/-0); vllm/v1/worker/gpu_model_runner.py (+8/-0); tests/test_config.py (+30/-0); tests/v1/kv_connector/unit/test_nixl_connector.py (+1/-0); tests/v1/kv_connector/unit/test_nixl_connector_hma.py (+181/-1); tests/v1/kv_connector/unit/test_tp_mapping.py (+73/-6); (+13 more)
LABELS: ready, kv-connector, nvidia, mrv2
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Alternative to https://github.com/vllm-project/vllm/pull/38433 as we iterate on the design with @pisceskkk . ⏎  ⏎ I feel this version is much closer to the changes we should see to the core files in terms of code structure and modifications to the workflow - it should result closer to injecting DCP login into current abstraction, rather than building something on the side. ⏎ However, it makes some significant assumptions to do that, described below. …[truncated]

### L3-b5707bf994  (L3, 2026-08-29, sha b5707bf994cb, PR #54048)
TITLE: [Bugfix][MoE] Enable cuBLAS out_dtype router GEMM on all CUDA archs (fixes family-120/GB10) (#54048)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/router/gate_linear.py (+10/-11)
LABELS: bug, rocm, nvidia
ISSUES: #49921 [Perf] BF16x3 router GEMM gated off family-120 Blackwell (GB10 / DGX Spark, sm_121) — the only barrier for DeepSeek-V4-Flash's fp32 router
BODY: ## Purpose ⏎  ⏎ Fixes #49921. On family-120 Blackwell (consumer sm_120 / SoC sm_121 — GB10 / DGX Spark) the bf16→fp32 router GEMM falls back to the standalone `bf16→fp32` copy kernel and bf16-rounds the router logits, even though the fused path is available. ⏎  ⏎ That fused path is just `torch.mm`'s out_dtype epilogue (plain cuBLAS / hipBLASLt — no NVVM or CuteDSL codegen), so it's arch-agnostic. But it was gated on `allow_specialized_router_gemm`, which …[truncated]

### L3-dbf662c9e8  (L3, 2026-08-29, sha dbf662c9e810, PR #51171)
TITLE: [ROCm][MLA] Reach FULL cudagraphs for AITER MLA speculative decoding (#51171)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+55/-54); vllm/v1/attention/backends/mla/triton_mla.py (+30/-0); tests/kernels/attention/test_rocm_aiter_mla_causal_verify_mask.py (+48/-71); tests/v1/attention/test_rocm_aiter_mla_mtp_split.py (+48/-2)
LABELS: rocm, ready, nvidia, verified
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Speculative decoding on ROCm cannot reach FULL cudagraphs, for two independent ⏎ reasons. This fixes both. Measured on Kimi-K3 with a DSpark draft, TP8 on MI355X. ⏎  ⏎ **The reported support level.** A DSpark draft group serves its whole ⏎ `1 + num_speculative_tokens` block through the decode path, but ⏎ `TritonMLAMetadataBuilder` reports `UNIFORM_SINGLE_TOKEN_DECODE` from a class ⏎ constant, and the engine takes the minimum over attentio …[truncated]

### L3-1ebff996a1  (L3, 2026-08-30, sha 1ebff996a118, PR #38434)
TITLE: [Fix] Improve ROCm detection in WSL environments (#38434)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/platforms/__init__.py (+15/-1)
LABELS: rocm, ready
BODY: ## Purpose ⏎ Adds fallback detection for ROCm platforms running in WSL` as `amd-smi` and `rocm-smi` is unreachable in WSL now. The fallback now activates ROCm only when vLLM is not a CPU-only build, PyTorch reports HIP support, and `torch.accelerator.is_available()` confirms that an accelerator is visible. This enables proper platform identification for ROCm setups in WSL where traditional detection methods may not work reliably. ⏎  ⏎ The existing ` …[truncated]

### L3-2c7d7dd64a  (L3, 2026-08-30, sha 2c7d7dd64a2e, PR #52033)
TITLE: [Perf][ROCm] Dual-stream decode with hipgraphs (#52033)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/runner/moe_runner.py (+29/-18); vllm/model_executor/layers/fused_moe/runner/shared_experts.py (+41/-39)
LABELS: rocm, ready
ISSUES: #48111 [Feature][ROCm]: Dual-stream decode (`MULTI_STREAM_OVERLAPPED` shared experts)
DEEP_STUDY: deep-study performance PR ()
BODY: ## What's new since reverted in https://github.com/vllm-project/vllm/pull/52024 ⏎  ⏎ TLDR: changes since this PR was reverted: [here](https://github.com/vllm-project/vllm/pull/52033/changes/b25620205d5529e8e36ca143050770ce31969428..1b7ff2c22beb1fddbebaed5c08ef62367e986528). ⏎  ⏎ **Context:** Re-opened https://github.com/vllm-project/vllm/pull/48223 after it failed CI due to poor gsm8k acc for Qwen3.5 with DP2EP. ⏎  ⏎ **Root cause of failure:** Qwen3.5  …[truncated]

### L3-87b9b5b8d9  (L3, 2026-08-30, sha 87b9b5b8d9da, PR #53531)
TITLE: [Test][VLM] Add batch-invariance tests for Qwen3-VL (#53531)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/misc.yaml (+1/-0); docs/features/batch_invariance.md (+1/-0); tests/v1/determinism/test_batch_invariance_vlm.py (+103/-0)
LABELS: documentation, ready, ci/build, qwen
BODY: ## Summary ⏎  ⏎ Adds regression coverage for **batch invariance of vision-language models (VLMs)**, addressing the gap left by issue #27059 ("Batch-invariant Inference Support for VLMs"), which was auto-closed by the stale bot with no associated PRs. ⏎  ⏎ Batch invariance (`VLLM_BATCH_INVARIANT=1`) currently has end-to-end determinism tests only for text models (`tests/v1/determinism/test_batch_invariance.py`). The VLM vision tower is composed almost …[truncated]

### L3-9d0fe9bac8  (L3, 2026-08-30, sha 9d0fe9bac89d, PR #54427)
TITLE: [Bugfix][Quantization][MoE] Route weight only NVFP4 checkpoints through W4A16 (#54427)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/modelopt.py (+16/-0)
LABELS: bug, ready, quantization
ISSUES: #54189 NVFP4 MoE silently folds an uninitialised activation scale for weight-only checkpoints (zeroes every expert)
BODY: ## Purpose ⏎  ⏎ Fixes #54189. ⏎  ⏎ A ModelOpt NVFP4 checkpoint can label itself `quant_algo` NVFP4 while quantizing weights only, leaving `input_activations` null in every config group. There is no activation scale to load, so the W4A4 fused MoE path folds an uninitialized `input_scale` into the dequant alphas: ⏎  ⏎ ``` ⏎ layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale) ⏎ layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale) ⏎ ``` ⏎  ⏎ When that  …[truncated]

### L3-7ab2923489  (L3, 2026-08-30, sha 7ab29234890b, PR #54313)
TITLE: [Flashinfer] Upgrade Flashinfer version to 0.6.18 (#54313)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); docker/versions.json (+1/-1); requirements/cuda.txt (+2/-2); tests/evals/gpt_oss/configs/gpt-oss-20b-flashinfer-mxfp4-bf16-cutlass.yaml (+1/-1); vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+2/-0)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎ Upgrade Flashinfer version to 0.6.18 ⏎  ⏎ ## Test Plan ⏎ Full CI nightly test ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-b2dc864bb6  (L3, 2026-08-30, sha b2dc864bb668, PR #54418)
TITLE: [Bugfix][Spec Decode] Keep default CUDA graph sizes memory-safe (#54418)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/test_config.py (+64/-56); vllm/config/compilation.py (+2/-3); vllm/config/vllm.py (+7/-8)
LABELS: bug, ready, torch.compile, nvidia
BODY: ## Purpose ⏎  ⏎ Fix the default CUDA-graph memory regression introduced by #50488 without reverting its full speculative-decode sizing work. ⏎  ⏎ #50488 appends uniform-decode capture sizes above the platform's established default ceiling (512 tokens on H200, 1024 on data-center Blackwell). On H200 nightly CI this expanded default captures to 2176 and 6400 tokens: ⏎  ⏎ - [Spec Decode N-Gram + Suffix](https://buildkite.com/vllm/ci/builds/86198#01a05142-37be-4 …[truncated]

### L3-56058fd572  (L3, 2026-08-30, sha 56058fd572f6, PR #53877)
TITLE: [Bugfix][Kernel] Keep packed GDN decode beta in FP32 (#53877)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/third_party/flash_linear_attention/ops/fused_recurrent.py (+1/-1); tests/kernels/test_fused_recurrent_packed_decode.py (+32/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ The packed GDN decode kernel loads `b` in FP32 and computes `sigmoid(b)` in ⏎ FP32, but then rounds the result to the input dtype before converting it back to ⏎ FP32: ⏎  ⏎ ```python ⏎ beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32) ⏎ ``` ⏎  ⏎ For BF16 inputs, this perturbs every recurrent state update and the error ⏎ compounds over long decode sequences. The standard fused sigmoid decode path ⏎ and the fused prefill preparation path ke …[truncated]

### L3-da0b2d8b17  (L3, 2026-08-31, sha da0b2d8b17c9, PR #53517)
TITLE: [Performance] Optimize Dots3 NOTE runtime (#53517)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+2/-0); vllm/models/dots3_note/nvidia/attention.py (+7/-1); vllm/models/dots3_note/nvidia/model.py (+25/-22); vllm/models/dots3_note/nvidia/mtp.py (+37/-35); vllm/models/dots3_note/nvidia/vision.py (+7/-21); vllm/models/dots3_note/nvidia/vision_attention.py (+16/-6)
LABELS: ready, torch.compile
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ This PR optimizes the Dots3 NOTE runtime paths for both the language model and vision encoder. ⏎  ⏎ The main changes are: ⏎  ⏎ - Enable Model Runner V2 by default for `Dots3NoteForCausalLM`. ⏎ - Capture the MTP speculator prefill and decode paths with CUDA Graphs under Model Runner V2. ⏎ - Fuse the MTP embedding lookup with the embedding/hidden-state RMSNorm path when the vocabulary is replicated. ⏎ - Enable uniform-batch CUDA Graph suppor …[truncated]
