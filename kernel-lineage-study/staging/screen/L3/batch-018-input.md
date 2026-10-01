### L3-4f1da84eb5  (L3, 2026-07-31, sha 4f1da84eb55d, PR #46516)
TITLE: Enable gfx1250 ROCm architecture (#46516)
SOURCES: path_core, symbol_pickaxe, dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flash_attn.fork_inline_cmake, L3.flash_attn.fa_utils, L3.rocm.custom_paged, L3.rocm.aiter_fa, L3.platform.rocm_selection
FILES: CMakeLists.txt (+40/-2); csrc/rocm/attention.cu (+11/-0); docker/Dockerfile.rocm_base_gfx1250 (+325/-0); docker/Dockerfile.rocm_gfx1250 (+729/-0); vllm/v1/attention/backends/fa_utils.py (+18/-9); vllm/v1/attention/backends/rocm_aiter_fa.py (+3/-3); csrc/quickreduce/base.h (+11/-0); csrc/rocm/torch_bindings.cpp (+5/-0); tests/entrypoints/speech_to_text/transcription/test_transcription_validation_whisper.py (+3/-3); tests/entrypoints/speech_to_text/translation/test_translation_validation.py (+3/-3); (+23 more)
LABELS: performance, rocm, ready, ci/build, v1, multi-modality, gpt-oss, kv-connector, nvidia, quantization
BODY: ## Purpose ⏎ Initial enablement of gfx1250, currently working through known models and UTs.  ⏎  ⏎  ⏎ ## Test Plan ⏎ Testing key models and expanding as hardware becomes available  ⏎  ⏎ ## Test Result ⏎ testing so far: ⏎ | Model | Backend | Decoding | flexible | strict | ⏎ |---|---|---|---:|---:| ⏎ | dsv4f | AITER_MXFP4_BF16 | greedy t=0 | 0.96 ±0.0197 | 0.96 ±0.0197 | ⏎ | dsv4f | AITER_MXFP4_BF16 | sampling t=1.0 | 0.97 ±0.0171 | 0.97 ±0.0171 | ⏎ | gptoss | AI …[truncated]

### L3-4689c7dd61  (L3, 2026-07-31, sha 4689c7dd6154, PR #50006)
TITLE: [ROCm] Add tuned selective_state_update float16 config for AMD Instinct MI325X (#50006)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/ops/configs/selective_state_update/headdim=64,dstate=128,device_name=AMD_Instinct_MI325X,cache_dtype=float16.json (+51/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Summary ⏎  ⏎ Adds a tuned `selective_state_update` (Mamba SSU decode kernel) launch config for the **AMD Instinct MI325X** for the `cache_dtype=float16` variant, mirroring how the MI300X (#47945), MI355 (#47767, #48372) and MI350 (#48159) configs were done. ⏎  ⏎ vLLM bundles per-device SSU configs for NVIDIA parts (B200, GB200, H100, H200, RTX PRO 6000) and, more recently, MI300X / MI350 / MI355. On MI325X `get_ssm_configs()` finds no matching file an …[truncated]

### L3-b49eaf205a  (L3, 2026-07-31, sha b49eaf205a75, PR #48047)
TITLE: [DSv4] Remove sparse-MLA q-head padding for FlashInfer >=0.6.14 (#48047)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py (+16/-19)
LABELS: nvidia
BODY: ## Purpose ⏎  ⏎ Remove the query-head padding in the DeepSeek V4 FlashInfer sparse-MLA ⏎ attention layers. flashinfer-ai/flashinfer#3545 relaxes the ⏎ `trtllm_batch_decode_sparse_mla_dsv4` guard to accept `h_q ∈ {8, 16, 32, 64, 128}` ⏎ on **both** the SM100 and SM120 decode paths (previously only `{64, 128}`). ⏎  ⏎ Before this, vLLM padded the per-rank head count up to the nearest kernel-supported ⏎ value — SM100 (`DeepseekV4FlashInferMLAAttention`) to ` …[truncated]

### L3-82ae4164ee  (L3, 2026-07-31, sha 82ae4164ee01, PR #48770)
TITLE: [2/N][Attention] Enable masked MHA for sparse MLA prefills (#48770)
SOURCES: path_core, subject_keyword, symbol_pickaxe, dependency_pin, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.flash_attn.fork_build, L3.flash_attn.fa4_cutedsl, L3.mla.common_v1
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/model_executor/layers/attention/mla_attention.py (+62/-5); vllm/model_executor/layers/attention/sparse_mla_attention.py (+519/-12); vllm/model_executor/layers/attention/sparse_mla_mask.py (+56/-0); vllm/vllm_flash_attn/flash_attn_interface.py (+13/-0); benchmarks/attention_benchmarks/benchmark.py (+17/-1); benchmarks/attention_benchmarks/common.py (+2/-1); benchmarks/attention_benchmarks/configs/mla_sparse_masked_mha_vs_mqa.yaml (+123/-0); benchmarks/attention_benchmarks/mla_runner.py (+7/-0); tests/v1/attention/test_sparse_mla_backends.py (+26/-6); (+1 more)
LABELS: performance, ready, ci/build, v1
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Depends on https://github.com/vllm-project/flash-attention/pull/155 ⏎  ⏎ ## Purpose ⏎ For pure prefills, there is a threshold of sequence length below which a masked MHA pathway is faster than sparse MQA. The [DeepSeek V3.2 paper mentions](https://arxiv.org/abs/2512.02556) this: ⏎  ⏎ > Note that for short-sequence prefilling, we specially implement a masked MHA mode to ⏎ simulate DSA, which can achieve higher efficiency under short-context conditions. …[truncated]

### L3-541128bdeb  (L3, 2026-07-31, sha 541128bdeb62, PR #50301)
TITLE: [KV Offload] Enable single-copy MLA layout for CPUOffloadingSpec (#50301)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_offload/test_factory.py (+98/-5); vllm/v1/kv_offload/cpu/spec.py (+14/-11); vllm/v1/kv_offload/tiering/spec.py (+7/-1)
LABELS: ready, v1
BODY: ## Purpose ⏎  ⏎ Enable the single-copy (replicated) MLA layout for the default KV-offload backend `CPUOffloadingSpec`, closing out the plan in #47929. After #50094 moved the default backend's worker CPU buffer onto the shared `SharedOffloadRegion` (on CUDA/ROCm), the default backend can now deduplicate the per-TP-rank-replicated MLA KV the same way `TieringOffloadingSpec` already does — but only on deployments that actually allocate on the shared reg …[truncated]

### L3-df71917cf1  (L3, 2026-07-31, sha df71917cf17c, PR #49236)
TITLE: [DSv4 Perf] Optimize workspace reuse for eager break, 3.9% E2E TTFT improvement. (#49236)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/nvidia/flashmla.py (+3/-0); csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu (+24/-6); csrc/libtorch_stable/ops.h (+8/-0); csrc/libtorch_stable/torch_bindings.cpp (+7/-0); tests/kernels/test_compressor_kv_cache.py (+18/-0); tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py (+12/-2); tests/kernels/test_fused_indexer_q_rope_quant.py (+26/-0); vllm/models/deepseek_v4/attention.py (+35/-2); vllm/models/deepseek_v4/common/ops/cache_utils.py (+10/-2); vllm/models/deepseek_v4/common/ops/fused_indexer_q.py (+30/-12); (+5 more)
LABELS: ready, deepseek, nvidia
DEEP_STUDY: deep-study: this PR was reverted by PR 52836 (confirmed_revert, reason=correctness_or_accuracy) || deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Part of https://github.com/vllm-project/vllm/issues/45861 ⏎  ⏎ Optimize workspace reuse for eager break ⏎  ⏎ ## Test ⏎  ⏎ `vllm serve deepseek-ai/DeepSeek-V4-Flash   --tensor-parallel-size 4   --enable-expert-parallel   --attention-backend FLASHMLA_SPARSE_DSV4   --attention-config '{"use_fp4_indexer_cache":true}'   --kv-cache-dtype fp8   --tokenizer-mode deepseek_v4   --all2all-backend allgather_reducescatter   --port 8003` ⏎  ⏎ ### Acc ⏎  ⏎  …[truncated]

### L3-a0cd2b69b3  (L3, 2026-07-31, sha a0cd2b69b3da, PR #50302)
TITLE: [Bugfix] Universally align block table width to 128 tokens (#50302)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+2/-0); vllm/v1/attention/backends/mla/indexer.py (+3/-11); tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py (+4/-0); tests/v1/attention/test_mla_backends.py (+3/-9); tests/v1/worker/test_gpu_model_runner.py (+28/-0); vllm/v1/worker/block_table.py (+32/-4); vllm/v1/worker/gpu/model_runner.py (+6/-5); vllm/v1/worker/utils.py (+12/-1)
LABELS: bug, ready, v1, deepseek, nvidia, mrv2
ISSUES: #46074 [Bug]: GLM-5.2 (DSA sparse MLA) + fp8_ds_mla — sparse indexer off-by-one crashes concurrent decode at max_model_len >= ~325K
BODY: ## Summary ⏎  ⏎ Share one block-table width calculation between MRV1, MRV2, and the DSA indexer. It applies 128-token alignment and kernel-block splitting, preventing indexer buffer mismatches under alignment and DCP. ⏎  ⏎ Fixes #46074. ⏎  ⏎ Supersedes #48404, #50050, and the alignment portion of #43970. ⏎  ⏎ ## Testing ⏎  ⏎ `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/v1/worker/test_gpu_model_runner.py -k get_block_table_width -q` ⏎  ⏎ AI assis …[truncated]

### L3-ef0d084b4b  (L3, 2026-07-31, sha ef0d084b4b11, PR #50349)
TITLE: [XPU] Fix FP8 block scale layout for MLA compatibility (#50349)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/scaled_mm/xpu.py (+34/-25)
LABELS: intel-gpu, ready
BODY: ## Summary ⏎  ⏎ Store block scale as `[n_blocks, k_blocks]` view (matching checkpoint shape) instead of contiguous `[k_blocks, n_blocks]`. This ensures: ⏎ - MLA's `scaled_dequantize` sees the expected shape for dequantization during `process_weights_after_loading` ⏎ - `apply_block_scaled_mm` recovers the contiguous `[k_blocks, n_blocks]` buffer that oneDNN expects via `.t()` ⏎ - `test_can_initialize_large_subset[DeepseekV3ForCausalLM]` passes ⏎  ⏎ Also  …[truncated]

### L3-963a658674  (L3, 2026-07-31, sha 963a65867404, PR #50307)
TITLE: Bump Helion to 1.4.0 (#50307)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/test-amd.yaml (+1/-1); .buildkite/test_areas/kernels.yaml (+1/-1); setup.py (+1/-1)
LABELS: ready, ci/build
BODY: Update Helion commit to 1.4.0. This includes fix https://github.com/pytorch/helion/pull/3081, which fixes numerics issue in the fused_qk_norm_rope Helion kernel

### L3-b2fb83e7ff  (L3, 2026-07-31, sha b2fb83e7ffbc, PR #50148)
TITLE: [Attention]: Use KVCacheSpec for AttentionMetadataBuilder type hints (#50148)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/chunked_local_attention.py (+1/-2); vllm/model_executor/layers/attention/mla_attention.py (+2/-0); vllm/v1/attention/backend.py (+3/-3); vllm/v1/attention/backends/flash_attn.py (+2/-2); vllm/v1/attention/backends/flashinfer.py (+3/-1); vllm/v1/attention/backends/hpc_attn.py (+2/-2); vllm/v1/attention/backends/mla/indexer.py (+2/-2); vllm/v1/attention/backends/turboquant_attn.py (+2/-0); vllm/v1/attention/backends/gdn_attn.py (+3/-3); vllm/v1/attention/backends/linear_attn.py (+5/-5); (+4 more)
LABELS: ready, v1, nvidia, mrv2
BODY: ## Purpose ⏎  ⏎ `AttentionMetadataBuilder.__init__` declares `kv_cache_spec: AttentionSpec` but the mamba backends are handed a `MambaSpec`. Those two are siblings under `KVCacheSpec` not subtypes, so the hint on the base is wrong for every SSM backend. ⏎  ⏎ The base also does `self.kv_cache_spec = kv_cache_spec` which pins the attribute's type for the whole hierarchy. That's why `gdn_attn`, `mamba_attn` and `linear_attn` each carry an `assert isinst …[truncated]

### L3-90329913e9  (L3, 2026-07-31, sha 90329913e945, PR #50590)
TITLE: [UX] Reduce startup log noise (#50590)
SOURCES: path_core
ARTIFACT_HINTS: L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backends/utils.py (+1/-1); vllm/v1/attention/selector.py (+1/-1); vllm/config/vllm.py (+11/-16); vllm/distributed/device_communicators/custom_all_reduce.py (+3/-3); vllm/distributed/device_communicators/flashinfer_all_reduce.py (+2/-2); vllm/model_executor/warmup/cutedsl_warmup.py (+5/-14); vllm/model_executor/warmup/kernel_warmup.py (+11/-11); vllm/utils/system_utils.py (+1/-1); vllm/v1/worker/gpu/model_runner.py (+1/-1)
LABELS: ready, nvidia, mrv2
BODY: ## Summary ⏎  ⏎ - remove the default asynchronous-scheduling status log ⏎ - emit repeated configuration and warmup messages once per process ⏎ - move routine backend-selection, registration, and no-op details to DEBUG ⏎ - preserve warnings whose content may differ by rank ⏎  ⏎ This only changes logging; serving behavior and model outputs are unchanged. ⏎  ⏎ ## Duplicate-work check ⏎  ⏎ Searched open PRs for `startup logging`, `log noise`, and `GPU KV cache size`. ⏎  ⏎ - # …[truncated]

### L3-2c4d348848  (L3, 2026-07-31, sha 2c4d3488483a, PR #48408)
TITLE: [KV Connector] Add per-layer canonical KV page mappings for parallelism-agnostic offload (#48408)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/offloading_connector/test_canonical_mapping.py (+474/-0); tests/v1/kv_connector/unit/offloading_connector/test_worker.py (+31/-6); vllm/distributed/kv_transfer/kv_connector/v1/offloading/canonical_mapping.py (+444/-0); vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py (+20/-0); vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py (+3/-1); vllm/v1/kv_offload/base.py (+44/-0)
LABELS: ready, v1, kv-connector
BODY: Records on `CanonicalKVCacheRef`, per layer, how the worker's KV page maps into a ⏎ canonical (parallelism-free) page: the full offloaded block, all KV heads, all ⏎ `block_size * dcp * pcp` tokens. A cache offloaded under one parallel config can ⏎ later be rearranged or consumed under another. ⏎  ⏎ All parallelism reasoning lives in `vllm/v1/kv_offload/sharding.py`; the connector ⏎ and everything downstream consume byte mappings (`MappedRun` / `CanonicalPage …[truncated]

### L3-10e6b40015  (L3, 2026-07-31, sha 10e6b400150c, PR #50437)
TITLE: [CPU][BugFix] Remove redundant kv cache write (#50437)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+6/-10); .buildkite/scripts/hardware_ci/run-cpu-test-arm.sh (+3/-2); tests/kernels/attention/test_cpu_attn.py (+36/-0)
LABELS: bug, ready, ci/build, v1, cpu
BODY: ## Purpose ⏎  ⏎ While profiling some LLMs, I noticed that `cpu_attn_reshape_and_cache` gets called twice for each invocation of the attention kernel for causal and cross attention. ⏎  ⏎ Since https://github.com/vllm-project/vllm/pull/40470 - the kv write, for causal and cross attention is done by the caller (CPUAttentionBackend.do_kv_cache_update) - `forward_includes_kv_cache_update` is set to false ⏎  ⏎ We now only run `cpu_attn_reshape_and_cache` in  …[truncated]

### L3-4fdd3e0b74  (L3, 2026-07-31, sha 4fdd3e0b74ab, PR #40289)
TITLE: [ROCm][ViT] Detect Triton-AMD kernels at their new aiter location (#40289)
SOURCES: body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+15/-3)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready, ci/build
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎  ⏎ `flash_attn_triton_available()` in `vllm/platforms/rocm.py` gates whether `AttentionBackendEnum.FLASH_ATTN` is offered for the ViT encoder on ROCm. The check looks for `flash_attn.flash_attn_triton_amd`, but ROCm/flash-attention commit [`3f94643`](https://github.com/ROCm/flash-attention/commit/3f94643) (2026-03-12, *\"[AMD] Migrate to Triton Backend to Aiter\"*) removed that subpackage and moved the kernels into aiter at `aiter.ops.tr …[truncated]

### L3-7c08664f5c  (L3, 2026-07-31, sha 7c08664f5c62, PR #50522)
TITLE: Upgrade tpu-inference to v0.26.0 (#50522)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/tpu.txt (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ Upgrade tpu-inference to latest stable release v0.26.0 ⏎  ⏎ ## Test Plan ⏎ Verified on tpu-inference CI.  ⏎  ⏎ ## Test Result ⏎ Success.  ⏎  ⏎ --- ⏎ [details omitted]

### L3-aef85aed5d  (L3, 2026-07-31, sha aef85aed5d89, PR #50533)
TITLE: [Bugfix][TurboQuant] Add KV quant mode for turboquant  (#50533)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/attention.py (+1/-0); tests/quantization/test_turboquant.py (+23/-0); vllm/v1/kv_cache_interface.py (+8/-0); vllm/v1/worker/gpu/attn_utils.py (+0/-2)
LABELS: bug, ready, quantization, mrv2
BODY: ## Purpose ⏎ Adds KV quant mode for turboquant KV cache. This fix prevents the KV cache dtype string being set to auto which causes the following failure in TurboQuant observed in  https://github.com/vllm-project/vllm/pull/47609 https://github.com/vllm-project/vllm/pull/48177/ https://github.com/vllm-project/vllm/pull/48907 ⏎  ⏎ ``` ⏎ vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507  --kv-cache-dtype turboquant_4bit_nc ⏎ ``` ⏎ ``` ⏎ (EngineCore pid=21869)    …[truncated]

### L3-eb6453d95b  (L3, 2026-07-31, sha eb6453d95b6d, PR #50474)
TITLE: [Build] Update pin to build ABI stable FA2 (#50474)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); .buildkite/check-torch-abi.py (+0/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ As a part of #26946, completes making FA2 ABI stable (predecessor #46640). For now, this PR tests the new pin at https://github.com/vllm-project/flash-attention/pull/175 ⏎  ⏎ ## Test Plan ⏎  ⏎ CI should be green. ⏎  ⏎ the .so has no unstable APIs: ⏎ ``` ⏎ (vllm-pt213) ➜  vllm git:(vllm-stable-fa2) ✗ torch-abi-audit  /home/janeyx/repos/vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so  ⏎ Package: _vllm_fa2_C.abi3.so ⏎   Root: /home/janeyx/repos/vl …[truncated]

### L3-77469c9057  (L3, 2026-08-01, sha 77469c9057be, PR #50476)
TITLE: [ROCm][MLA] Mask the AITER MLA small-head verify flatten causally (#50476)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+27/-10); tests/kernels/attention/test_rocm_aiter_mla_causal_verify_mask.py (+257/-0)
LABELS: rocm, ready, v1
BODY: # [ROCm][MLA] Mask the AITER MLA small-head verify flatten causally ⏎  ⏎ ## Purpose ⏎  ⏎ On `ROCM_AITER_MLA`, a multi-token speculative verify block with fewer than 16 query ⏎ heads per rank is flattened into single-token Gluon decodes with no causality check ⏎ (`vllm/v1/attention/backends/mla/rocm_aiter_mla.py:1006-1009`), and every resulting ⏎ row is handed its request's **entire** paged-KV range, `per_req_len[row_req]` ⏎ (`rocm_aiter_mla.py:1035`). Th …[truncated]

### L3-3986b967b2  (L3, 2026-08-01, sha 3986b967b249, PR #50498)
TITLE: (feat): optionally disable lookup on PD decode (#50498)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py (+25/-0); tests/v1/kv_connector/unit/test_mooncake_store_worker.py (+1/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py (+10/-2); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py (+5/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py (+26/-1)
LABELS: ready, kv-connector
BODY: The original implementation is done by @ivanium . ⏎  ⏎ ## Summary ⏎ This PR introduces the `enable_lookup` flag for `MooncakeStoreConnector`, to be added in `kv_connector_extra_config` with `kv_consumer` role. This flag is for a PD setup where decode instance doesn't read from or write to the Mooncake, and only contributes its segment for extending prefill worker KV cache capacity. e.g. ⏎ ``` ⏎ --kv-transfer-config '{"kv_connector":"MultiConnector","k …[truncated]

### L3-38a466e7b6  (L3, 2026-08-01, sha 38a466e7b6e0, PR #46789)
TITLE: [DSV4] Implement Sequence Parallelism (#46789)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/kimi_k3/test_sequence_parallel.py (+19/-1); vllm/models/common/ops/sequence_parallel.py (+4/-9); vllm/models/deepseek_v4/nvidia/dspark.py (+30/-4); vllm/models/deepseek_v4/nvidia/model.py (+53/-4); vllm/models/deepseek_v4/nvidia/mtp.py (+21/-4); vllm/models/kimi_k3/nvidia/model.py (+6/-6); vllm/models/kimi_k3/nvidia/mtp.py (+5/-1)
LABELS: ready, v1, mrv2, kimi, k3
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ Implement DeepSeek V4 sequence parallelism using the collective path introduced for Kimi K3. ⏎  ⏎ - Move the Kimi K3 SP helpers into a shared model-ops module and use the TP device communicator's optimized `custom_all_gather` and `custom_reduce_scatter`, with the existing generic collectives as fallback. ⏎ - Keep token padding, sharding, padding-mask propagation, and output trimming in the DeepSeek V4 model code. This PR does not add an `sp …[truncated]

### L3-e2fa28594f  (L3, 2026-08-02, sha e2fa28594f7b, PR #49934)
TITLE: [1/N] Unify multiple-path encoder cuda graph support (#49934)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/design/cuda_graphs_multimodal.md (+16/-25); tests/v1/cudagraph/test_encoder_cudagraph.py (+2/-0); vllm/model_executor/models/deepseek_ocr.py (+22/-11); vllm/model_executor/models/interfaces.py (+4/-3); vllm/model_executor/models/step3_vl.py (+20/-9); vllm/v1/worker/encoder_cudagraph.py (+103/-322); vllm/v1/worker/encoder_cudagraph_defs.py (+25/-17)
LABELS: documentation, ready, v1, deepseek, nvidia
BODY: PLEASE FILL IN THE PR DESCRIPTION HERE ENSURING ALL CHECKLIST ITEMS (AT THE BOTTOM) HAVE BEEN CONSIDERED. ⏎  ⏎ ## Purpose ⏎ - Separate from #49432 to avoid massive design changes. ⏎ - Currently, for multi-path encoder like deepseek-ocr, we assumed that there's two path (global/local) for encoder cuda graph. However, it's not a good design for encoder cg manager to be aware of the special case global/local path. ⏎ - This PR try to unify the path manage …[truncated]

### L3-0055b8bfa3  (L3, 2026-08-02, sha 0055b8bfa3a2, PR #50032)
TITLE: [Attention][MiniMax-M3] Add MSA speculative decode verification (#50032)
SOURCES: path_core, symbol_pickaxe, dependency_pin
ARTIFACT_HINTS: L3.dispatch.registry
FILES: cmake/external_projects/fmha_sm100.cmake (+1/-1); vllm/v1/attention/backends/registry.py (+8/-0); .buildkite/test_areas/kernels.yaml (+12/-1); csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu (+66/-23); csrc/libtorch_stable/ops.h (+2/-1); csrc/libtorch_stable/torch_bindings.cpp (+2/-1); tests/kernels/attention/test_minimax_m3_msa_cutlass_sparse_decode.py (+561/-0); tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py (+15/-0); vllm/_custom_ops.py (+7/-0); vllm/config/attention.py (+15/-0); (+5 more)
LABELS: ready, ci/build, v1, nvidia
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Add an opt-in MSA/CUTLASS sparse decode path for MiniMax M3 speculative ⏎ verification on SM100 and SM103 GPUs: ⏎  ⏎ ```bash ⏎ VLLM_MINIMAX_M3_MSA_DECODE_BACKEND=cutlass ⏎ ``` ⏎  ⏎ MSA planning and metadata remain NVIDIA-backend-specific. Unsupported ⏎ platforms, KV formats, and graph shapes continue through the existing Triton ⏎ decode path without constructing an MSA plan. ⏎  ⏎ The CUTLASS path is limited to FP8 E4M3 KV cache, page size 128, top-k 16, ⏎ qu …[truncated]

### L3-dd11df04f3  (L3, 2026-08-03, sha dd11df04f3b7, PR #49389)
TITLE: [Misc] Remove deprecated calculate_kv_scales runtime KV scale calculation (#49389)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/attention.py (+0/-62); vllm/model_executor/layers/attention/mla_attention.py (+0/-41); .buildkite/test-amd.yaml (+1/-1); .buildkite/test_areas/compile.yaml (+0/-3); .buildkite/test_areas/pytorch.yaml (+1/-2); docs/design/metrics.md (+1/-1); docs/features/quantization/quantized_kvcache.md (+3/-33); tests/compile/fullgraph/test_full_graph.py (+0/-32); tests/models/quantization/test_per_token_kv_cache.py (+0/-4); tests/quantization/test_fp8.py (+8/-20); (+9 more)
LABELS: documentation, ready, ci/build, v1, quantization
BODY: ## Purpose ⏎ Remove the deprecated `calculate_kv_scales` option (runtime fp8 k/v scale estimation) across config, attention, quantization, runner, and related tests/docs. ⏎  ⏎ fp8 KV cache scales now resolve via a single path: loaded from the checkpoint if present, otherwise defaulting to 1.0. ⏎  ⏎ related change: #37201 ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-2755489a26  (L3, 2026-08-03, sha 2755489a261b, PR #50807)
TITLE: [INC]  fix w4a4 model (#50807)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/quantization/test_auto_round.py (+2/-2); vllm/model_executor/layers/quantization/inc/schemes/inc_mxfp4_linear.py (+2/-1)
LABELS: intel-gpu, ready, quantization
BODY: INCMxfp4LinearMethod called init_mxfp4_linear_kernel() without an activation_quant_key, leaving it as None. This caused platform-specific failures: ⏎  ⏎ **XPU (hard crash)**: XPUMxFp4LinearKernel.can_implement strictly requires activation_quant_key == kMxfp4Dynamic and rejects None, causing model loading to fail with: ⏎ ``` ⏎ ValueError: Failed to find a kernel that can implement the MXFP4 linear layer. ⏎ XPUMxFp4LinearKernel: only supports MXFP4 dyna …[truncated]

### L3-0a6446005d  (L3, 2026-08-03, sha 0a6446005d51, PR #50133)
TITLE: [CPU] Migrate unquantized MoE to the modular-kernel experts structure (#50133)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: cmake/cpu_extension.cmake (+2/-1); csrc/cpu/cpu_fused_moe_activations.hpp (+24/-13); csrc/cpu/cpu_types_riscv_impl.hpp (+22/-0); csrc/cpu/cpu_types_scalar.hpp (+44/-0); csrc/cpu/cpu_types_vsx.hpp (+23/-0); csrc/cpu/cpu_types_vxe.hpp (+65/-1); csrc/cpu/cpu_types_x86.hpp (+35/-0); csrc/cpu/micro_gemm/cpu_micro_gemm_vec.hpp (+1/-2); csrc/cpu/torch_bindings.cpp (+0/-3); csrc/cpu/utils.hpp (+0/-2); (+10 more)
LABELS: documentation, ci/build, cpu, quantization
BODY: ## Purpose ⏎  ⏎ The unquantized (BF16/FP16/FP32) CPU MoE was the last unquantized backend still on the ⏎ legacy monolithic path. `oracle/unquantized.py` short-circuited before backend selection ⏎ with `# TODO: migrate to MK structure`, and `UnquantizedFusedMoEMethod` carried three CPU ⏎ escape hatches (`is_monolithic`, a bespoke `process_weights_after_loading` branch, and a ⏎ bespoke `apply_monolithic` branch) to keep it alive. Every *quantized* CPU MoE path …[truncated]

### L3-f43e1d26e3  (L3, 2026-08-03, sha f43e1d26e3e6, PR #43615)
TITLE: [ROCm] Enable AITER and FP8 inference on GFX120x (#43615)
SOURCES: path_core, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse, L3.platform.rocm_selection
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+2/-2); vllm/_aiter_ops.py (+84/-14); vllm/compilation/passes/fusion/rocm_aiter_fusion.py (+21/-8); vllm/compilation/passes/pass_manager.py (+12/-4); vllm/model_executor/kernels/linear/scaled_mm/aiter.py (+18/-3); vllm/model_executor/kernels/linear/scaled_mm/pytorch.py (+3/-3); vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py (+4/-4); vllm/model_executor/layers/fused_moe/oracle/fp8.py (+8/-2); vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+6/-1); vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+4/-1); (+2 more)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Enable AITER-backed FP8 and attention fast paths on AMD RDNA4 / gfx12x (R9700, RX 9070 XT), while preserving existing MI3xx behavior. ⏎  ⏎ Previously several AITER gates were effectively MI3xx-only, so gfx12 fell back to the generic ROCm/Triton paths even when AITER Triton support was available. This PR separates AITER Triton support from MI3xx-only CK support so gfx12 can use the optimized Triton paths without dispatching into unavai …[truncated]

### L3-11f88260a2  (L3, 2026-08-03, sha 11f88260a240, PR #50818)
TITLE: [Kimi-K3] Migrate FlashKDA to PyTorch stable ABI (#50818)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: cmake/external_projects/flashkda.cmake (+7/-1); .buildkite/check-torch-abi.py (+0/-1); csrc/flashkda_registration.cpp (+12/-5)
LABELS: ready, ci/build, kimi, k3
BODY: ## Purpose ⏎  ⏎ https://github.com/vllm-project/FlashKDA/pull/5 ⏎  ⏎ ## Test Plan ⏎  ⏎ TP8 ⏎ - GSM8K: 0.9674 ⏎ - OCRBench: 89.2 ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-65cf1276a2  (L3, 2026-08-03, sha 65cf1276a25b, PR #50776)
TITLE: [Kernel] Skip fully masked key blocks in windowed Triton prefill (#50776)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/triton_prefill_attention.py (+10/-1)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ `_fwd_kernel` in `triton_prefill_attention.py` walks every key block from 0 to `end_n` and masks out-of-window keys after loading them. The masking is correct, but the work is not skipped, so a sliding-window pass still costs O(seq_len^2). ⏎  ⏎ For a query block covering `[start_m * BLOCK_M, (start_m + 1) * BLOCK_M)`, only keys in `[start - SLIDING_WINDOW_Q, end + SLIDING_WINDOW_K]` can survive the mask, so the loop is clamped to that ran …[truncated]

### L3-385d4c084e  (L3, 2026-08-03, sha 385d4c084ef0, PR #50859)
TITLE: [ROCm][AITER] Hotfix for `memory access fault` errors in AITER triton MOE routing (#50859)
SOURCES: subject_keyword, corpus:kernel-correctness-cases
ARTIFACT_HINTS: -
FILES: tests/models/quantization/test_gpt_oss.py (+212/-1); vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py (+57/-0)
LABELS: rocm, ready, gpt-oss
DEEP_STUDY: deep-study correctness case vllm:385d4c084e: class=memory_safety_oob; symptom=illegal_memory_access; introducing=#49361
BODY: Companion AITER fix: https://github.com/ROCm/aiter/pull/4530 (making this PR fix redundant once the AITER fix is merged, released, and AITER pin in vLLM is updated). ⏎  ⏎ ## Purpose ⏎  ⏎ This PR hot-fixes ⏎  ⏎ ``` ⏎ tests/models/quantization/test_gpt_oss.py::test_gpt_oss_attention_quantization[amd/gpt-oss-20b-MoE-Quant-W-MXFP4-A-FP8-KV-FP8-0.89-1] ⏎ ``` ⏎  ⏎ that started failing in ROCm vLLM following AITER version bump from `0.1.16.post5` to `0.1.19` (htt …[truncated]

### L3-7743486190  (L3, 2026-08-04, sha 77434861904a, PR #50580)
TITLE: [Frontend] DeepSeek V4 0731 reasoning effort prompts & mappings (#50580)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: rust/src/chat/src/renderer/deepseek_v4/encoding.rs (+23/-10); rust/src/chat/src/renderer/deepseek_v4/tests.rs (+100/-4); tests/tokenizers_/test_deepseek_v4.py (+47/-20); vllm/tokenizers/deepseek_v4.py (+10/-6); vllm/tokenizers/deepseek_v4_encoding.py (+25/-11)
LABELS: ready, deepseek, rust
BODY: ## Purpose ⏎  ⏎ Align DeepSeek V4 Flash prompt rendering with DeepSeek-V4-Flash-0731 and the current hosted API's model-specific effort mapping. ⏎  ⏎ The renderer accepts the canonical 0731 levels: `low` emits no prefix, `high` emits the "Absolute maximum" prefix, and `max` emits the "Beyond maximum" prefix. DSML parsing and output syntax retain their existing behavior. ⏎  ⏎ The request normalization follows the public `deepseek-v4-flash actual mapped effort …[truncated]

### L3-52c0e3cb08  (L3, 2026-08-04, sha 52c0e3cb08b8, PR #50157)
TITLE: [Kernel] Add support for Flashinfer Mamba SSU algorithm selection (#50157)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/mamba.py (+28/-1); vllm/engine/arg_utils.py (+8/-1); tests/kernels/mamba/test_ssu_dispatch.py (+45/-1); vllm/model_executor/layers/mamba/ops/ssu_dispatch.py (+7/-2)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎ Allow choosing the Flashinfer Mamba SSU algorithm. ⏎ The new command line argument is `--mamba-ssu-algorithm`. ⏎  ⏎ On some use-cases of Nemotron 3 Nano NVFP4 (like ISL/OSL=1k/8k), using "horizontal" instead of "auto" (which eventually chooses "vertical") improves throughput. ⏎  ⏎ ## Test Plan ⏎ On Nemotron 3 Nano NVFP4 with & without `--mamba-ssu-algorithm horizontal`: ⏎ * Throughput benchmark with aiperf (3 runs to account for variance), o …[truncated]

### L3-199644d410  (L3, 2026-08-04, sha 199644d410db, PR #50906)
TITLE: [Bugfix][Attention] Guard sparse MLA masked MHA workspace (#50906)
SOURCES: path_core, subject_keyword, corpus:kernel-correctness-cases, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+1/-0); vllm/model_executor/layers/attention/sparse_mla_attention.py (+77/-13); tests/v1/attention/test_sparse_mla_backends.py (+37/-0)
LABELS: bug, ready
DEEP_STUDY: deep-study correctness case vllm:199644d410: class=memory_safety_oob; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎  ⏎ Sparse MLA masked-MHA builds a bit-packed top-k mask whose size is approximately: ⏎  ⏎ ```text ⏎ num_prefills × round_up(max_query_len, tile_m) × ceil(max_key_len / 32) ⏎     × sizeof(int32) ⏎ ``` ⏎  ⏎ The existing workspace was fixed at 64 MiB, while the routing matrix allows a ⏎ single 32K prefill to enter masked-MHA. Such a request requires exactly 128 MiB. ⏎ Heterogeneous batches can require even more despite staying within ⏎ `max_num_bat …[truncated]

### L3-59b2fdfc4e  (L3, 2026-08-04, sha 59b2fdfc4e23, PR #48250)
TITLE: Support MLA properly in the Transformers modeling backend (#48250)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/model.py (+7/-1); tests/models/transformers/fusers/test_mla.py (+161/-0); tests/models/transformers/test_backend.py (+42/-11); vllm/model_executor/models/transformers/__init__.py (+34/-3); vllm/model_executor/models/transformers/base.py (+104/-33); vllm/model_executor/models/transformers/fuser.py (+5/-4); vllm/model_executor/models/transformers/fusers/__init__.py (+2/-0); vllm/model_executor/models/transformers/fusers/mla.py (+325/-0); vllm/model_executor/models/transformers/fx_utils.py (+19/-0)
LABELS: ready
ISSUES: #48652 transformers backend: tensor reshape error during profile run with GLM MoE architecture
BODY: Requires the following Transformers side PRs: ⏎ - https://github.com/huggingface/transformers/pull/47435 ⏎ - https://github.com/huggingface/transformers/pull/47451 ⏎ - https://github.com/huggingface/transformers/pull/47460 ⏎  ⏎ Portions of this PR that were split into their own PRs: ⏎  ⏎ - https://github.com/vllm-project/vllm/pull/49957 ⏎ - https://github.com/vllm-project/vllm/pull/49982 ⏎  ⏎ --- ⏎  ⏎ vLLM side changes: ⏎  ⏎ - Add `MLAFuser`: matches a Transfo …[truncated]

### L3-a5149b2fee  (L3, 2026-08-04, sha a5149b2feeb7, PR #51015)
TITLE: [CI] Stabilize GLM-5.2 PCP evaluation (#51015)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/evals/gsm8k/configs/GLM-5.2-NVFP4-TP1-PCP4-EP.yaml (+1/-0)
BODY: ## Summary ⏎  ⏎ - enable PyTorch expandable CUDA allocator segments for the GLM-5.2 TP1/PCP4 GSM8K evaluation; ⏎ - keep `--max-num-batched-tokens 32768`, which is needed to exercise the large-batch PCP path; ⏎ - scope the allocator change to the memory-tight TP1/PCP4 case; TP2/PCP2 is unchanged. ⏎  ⏎ ## Root cause ⏎  ⏎ The pytest-level error in [Buildkite #81913](https://buildkite.com/vllm/ci/builds/81913#019fca13-8131-4f3f-bcfa-6636b1022f66) is only a 20-minute …[truncated]

### L3-f9c74b4b9c  (L3, 2026-08-04, sha f9c74b4b9c25, PR #50991)
TITLE: [Mamba] enable prefix cache by default (#50991)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/language/generation/test_hybrid.py (+24/-17); vllm/config/cache.py (+3/-3); vllm/engine/arg_utils.py (+1/-5); vllm/model_executor/models/config.py (+2/-11)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ enable prefix cache for hybrid model by default and change the default mode to `align` ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-3756bf1e2b  (L3, 2026-08-04, sha 3756bf1e2ba0, PR #49792)
TITLE: [Kernel][SM100] Add a CuTeDSL fused query kernel (#49792)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_fused_q_cutedsl.py (+95/-0); vllm/cute_utils/__init__.py (+1/-0); vllm/cute_utils/cvt.py (+22/-1); vllm/models/deepseek_v32/common/kernels.py (+42/-0); vllm/models/deepseek_v32/nvidia/ops/__init__.py (+2/-0); vllm/models/deepseek_v32/nvidia/ops/fused_q_cutedsl.py (+537/-0)
LABELS: performance, ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## What this does ⏎  ⏎ Add an SM100 CuTeDSL implementation of the fused query preprocessing used by DSA ⏎ sparse attention (`fused_q`): the packed fp8 MQA query, the indexer query ⏎ RoPE + UE8M0 fp8 quant, and the folded index weights — registered as a torch ⏎ custom op. ⏎  ⏎ Selected only for the supported dtype/shape combination ⏎ (`is_fused_q_cutedsl_supported`); the existing Triton implementation stays the ⏎ fallback everywhere else. `vllm/cute_utils` …[truncated]

### L3-8ae8337ffa  (L3, 2026-08-04, sha 8ae8337ffa6e, PR #50911)
TITLE: [Spec Decode] Enable fused non-causal TokenSpeed MLA for DSpark (#50911)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/tokenspeed_mla.py (+8/-0); tests/v1/attention/test_mla_backends.py (+24/-7)
LABELS: ready, verified
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ Enable the TokenSpeed MLA backend for DSpark's non-causal multi-token draft ⏎ blocks. DSpark produces its draft positions in one non-causal forward pass, but ⏎ TokenSpeed did not advertise that capability and vLLM did not forward the ⏎ metadata's causal mode to the kernel. This prevented the draft model from using ⏎ TokenSpeed's fused multi-token path. ⏎  ⏎ This PR: ⏎  ⏎ - advertises non-causal and non-causal multi-token support for TokenSpeed MLA; ⏎ - …[truncated]

### L3-d0ce3dadb6  (L3, 2026-08-04, sha d0ce3dadb674, PR #50607)
TITLE: [ROCm]: Bump torch 2.12, triton 3.7, torchaudio, torchvision (#50607)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm_base (+10/-11); vllm/compilation/caching.py (+11/-3)
LABELS: rocm, ready, ci/build
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-c416f15710  (L3, 2026-08-05, sha c416f15710bb, PR #50404)
TITLE: [Model] Fix Kimi-K3 MLA with disabled context parallelism (#50404)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/kimi_k3/nvidia/mla.py (+5/-0)
LABELS: ready, kimi, k3
BODY: ## Purpose ⏎  ⏎ Fix Kimi-K3 MLA decoding when context parallelism is disabled. ⏎  ⏎ The shared MLA implementation initializes `dcp_world_size` with the internal ⏎ disabled sentinel `-1`. The regular `MLAAttention` wrapper resolves that value ⏎ before invoking the backend, but Kimi-K3 constructs and calls the backend ⏎ implementation directly. ⏎  ⏎ As a result, the Kimi-K3 path can pass invalid context-parallel metadata to the ⏎ FlashAttention backend even  …[truncated]

### L3-cd930c8e78  (L3, 2026-08-05, sha cd930c8e786b, PR #38771)
TITLE: [Bugfix] Fix MLA kv_b_proj activation dtype with Marlin FP8 (#38771)
SOURCES: path_core, subject_keyword, corpus:kernel-correctness-cases
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+24/-25)
LABELS: bug, ready, unstale
ISSUES: #38658 [Bug]: MLA attention casts activations to int32 when using Marlin FP8 on GPUs without native FP8 support (sm < 89)
DEEP_STUDY: deep-study correctness case vllm:cd930c8e78: class=hardware_compiler_specific; symptom=crash_or_exception; introducing=unknown
BODY: ## Purpose ⏎  ⏎ Fixes #38658. ⏎  ⏎ This PR fixes an MLA prefill dtype bug when FP8 weights are served through the Marlin path on GPUs without native FP8 support (`sm < 89`). ⏎  ⏎ On affected GPUs, Marlin repacks FP8 weights into `torch.int32`. In `vllm/model_executor/layers/attention/mla_attention.py`, `_compute_prefill_context()` was using `self.kv_b_proj.weight.dtype` to determine how to cast `kv_c_normed` before passing it to `kv_b_proj`. ⏎  ⏎ After M …[truncated]

### L3-08b8613b7b  (L3, 2026-08-05, sha 08b8613b7b15, PR #48929)
TITLE: [Bugfix][Model] Fix MiniMax-M3 NVFP4 inference correctness (#48929)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer_moe.py (+69/-0); vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py (+31/-31)
LABELS: bug, ready, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ Fix two MiniMax-M3 correctness issues exposed by the NVFP4 checkpoint. ⏎  ⏎ First, the routed experts use packed `SWIGLUOAI_UNINTERLEAVE` with model-specific `alpha`, `beta`, and clamp values. FlashInfer CUTLASS already supports this math, but the vLLM adapter neither advertised the packed activation nor forwarded all three parameters. Marlin similarly replaced missing quant-config alpha/beta values with plain-SiLU defaults instead of fal …[truncated]

### L3-66b3c0e61f  (L3, 2026-08-05, sha 66b3c0e61f1e, PR #50294)
TITLE: [Kernel][Model] Optimize FA4 mm_prefix range lookup (#50294)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/model_executor/models/gemma4_mm.py (+2/-0); vllm/v1/attention/backend.py (+1/-1); vllm/v1/attention/backends/flash_attn.py (+109/-44); vllm/v1/attention/backends/utils.py (+71/-0); tests/v1/attention/test_mm_prefix.py (+636/-0)
LABELS: performance, ready, v1, verified
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: # [Kernel][Model] Gemma4: optimize FA4 mm_prefix range lookup and CuTe JIT stability ⏎  ⏎ ## Motivation ⏎  ⏎ Gemma4 multimodal models can use FA4 as the backend for all layers, so that both ⏎ sliding-attention layers (`head_dim=256`) and global/full layers ⏎ (`global_head_dim=512`) use one attention backend. This is required for the ⏎ vision `mm_prefix` / PrefixLM bidirectional mask path, and avoids mixing ⏎ FlashAttention and Triton attention implementa …[truncated]

### L3-0187f4c88e  (L3, 2026-08-05, sha 0187f4c88e7e, PR #50841)
TITLE: [CPU] Enable tcmalloc for s390x (#50841)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.s390x (+15/-0); docs/getting_started/installation/cpu.s390x.inc.md (+21/-0); setup.py (+1/-1); vllm/platforms/cpu.py (+2/-1)
LABELS: documentation, ci/build, cpu
BODY: ## Purpose ⏎ Enable tcmalloc for s390x ⏎ ## Test Plan ⏎ Run vllm server and run inference ⏎ ## Test Result ⏎ ``` ⏎ [root@b314lp81 vllm]# podman run --rm -it   --shm-size=4g    -p 8000:8000   -v ~/.cache/huggingface:/root/.cache/huggingface   quay.io/r3hankhan/vllm:latest-s390x   --model facebook/opt-125m  --dtype bfloat16   --enforce-eager --gpu-memory-utilization 0.1 ⏎ INFO 08-03 08:58:17 [importing.py:88] Triton not installed or not compatible; certai …[truncated]

### L3-8f158d0ee2  (L3, 2026-08-05, sha 8f158d0ee247, PR #44359)
TITLE: [MoE] Share apply_moe_activation support metadata (#44359)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_cutlass_moe.py (+78/-6); tests/kernels/moe/test_moe.py (+68/-0); tests/kernels/moe/test_triton_moe_no_act_mul.py (+49/-1); vllm/model_executor/layers/fused_moe/__init__.py (+2/-0); vllm/model_executor/layers/fused_moe/activation.py (+91/-16); vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py (+48/-37); vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py (+20/-17); vllm/model_executor/layers/fused_moe/experts/marlin_moe.py (+40/-100); vllm/model_executor/layers/fused_moe/experts/mxfp8_emulation_moe.py (+2/-5); vllm/model_executor/layers/fused_moe/experts/mxfp8_native_moe.py (+4/-6); (+3 more)
LABELS: ready, nvidia
BODY: ## Purpose ⏎  ⏎ Centralize the activation capability and configuration contract for MoE backends that delegate to `apply_moe_activation()`. ⏎  ⏎ This adds `apply_moe_activation_supported()` and uses it from Triton, Marlin, Humming, and the applicable CUTLASS FP8/NVFP4/MXFP4/W4A8 experts instead of copying activation lists into each backend. ⏎  ⏎ It also adds the public, immutable `ApplyMoEActivationConfig`: ⏎  ⏎ - it is resolved once on `FusedMoEExperts` from th …[truncated]

### L3-397094da17  (L3, 2026-08-05, sha 397094da1768, PR #49990)
TITLE: Resolve revision to commit_hash once per model load, via huggingface_hub's `resolve_revision` (#49990)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: requirements/common.txt (+1/-0); requirements/test/cpu.txt (+2/-1); requirements/test/cuda.txt (+3/-1); requirements/test/rocm.txt (+3/-1); requirements/test/xpu.txt (+3/-1); vllm/config/model.py (+27/-0); vllm/model_executor/models/ultravox.py (+4/-2); vllm/transformers_utils/repo_utils.py (+38/-0); vllm/v1/worker/gpu_model_runner.py (+3/-0)
LABELS: ready, ci/build, cpu, nvidia
BODY: ## TL;DR: resolve `revision` and remote `code_revision` commit hashes only once via `huggingface_hub`'s new `resolve_revision`, preventing repeated downstream resolution and avoiding concurrency issues if a repository is updated while a model is loading ⏎  ⏎ **Disclaimer:** AI-assisted PR, heavily human-reviewed/amended. I'm maintainer of the `huggingface_hub` which handles all the download/cache system. ⏎  ⏎ cc @hmellor with whom I've discussed this …[truncated]

### L3-7c77868cdf  (L3, 2026-08-05, sha 7c77868cdf57, PR #51125)
TITLE: [Bugfix] Size and iterate w13 by shard count for non-gated MoE (#51125)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/quantization/test_auto_gptq.py (+1/-0); tests/quantization/test_auto_round.py (+4/-2); vllm/model_executor/layers/fused_moe/config.py (+5/-0); vllm/model_executor/layers/quantization/auto_awq.py (+7/-3); vllm/model_executor/layers/quantization/auto_gptq.py (+6/-4); vllm/model_executor/layers/quantization/bitsandbytes.py (+3/-3); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py (+2/-2); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8_fp8.py (+3/-3); vllm/model_executor/layers/quantization/fp8.py (+12/-5); vllm/model_executor/layers/quantization/humming.py (+2/-2); (+6 more)
LABELS: bug, ready, quantization
BODY: ## Purpose ⏎  ⏎ Non-gated MoE (`is_act_and_mul=False`, e.g. NemotronH's `relu2_no_mul`) fuses ⏎ only the up projection into `w13`, so `w13` holds a **single** ⏎ `intermediate_size_per_partition` shard rather than two gate/up shards. ⏎  ⏎ 13 MoE quantization methods already conditionalize on this — `modelopt.py`, ⏎ `compressed_tensors_moe_*`, `flashinfer`, `unquantized_fused_moe_method.py`, … — ⏎ each with its own local `w13_num_shards = 2 if self.moe.is_act_and_ …[truncated]

### L3-38ebd97bca  (L3, 2026-08-05, sha 38ebd97bcab9, PR #51083)
TITLE: [ROCm] Relax MLA rope+cache test tolerances for bf16 (#51083)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/core/test_rotary_embedding_mla_cache_fused.py (+17/-5)
LABELS: rocm, ready
BODY: ## Purpose ⏎  ⏎ `tests/kernels/core/test_rotary_embedding_mla_cache_fused.py::test_concat_and_cache_mla_rope_fused` ⏎ fails on ROCm for **bfloat16** against the hardcoded `atol=0.001`. On gfx942 this ⏎ is 48 of 3153 parametrizations, all `dtype=bfloat16` (fp16 and fp32 pass 100%). ⏎  ⏎ The failures are ~1 bfloat16 ULP, not a kernel bug. The fixture routes ROCm ⏎ through the AITER Triton rope for fp16-consistent numerics; the fused C++ ⏎ `concat_and_cache_mla_rop …[truncated]

### L3-9f3169960a  (L3, 2026-08-05, sha 9f3169960a37, PR #50942)
TITLE: [MoE] Align TRTLLM MXFP4 autotune buckets (#50942)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py (+6/-5)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎  ⏎ Use the shared FlashInfer MoE bucket helper so MXFP4 follows the same max-token and DP-aware convention as the other TRTLLM backends. ⏎  ⏎ Align with https://github.com/vllm-project/vllm/pull/47427, so that it will autotune up to 8192 tokens. Without this change, MXFP4 TRTLLM sets `tune_max_num_tokens = moe_config.max_capture_size`, which is only 512 -> bad for prefill. ⏎  ⏎ We can verify the behavior in vLLM logs. Before this PR, there …[truncated]

### L3-71f975af43  (L3, 2026-08-05, sha 71f975af437a, PR #50480)
TITLE: [ROCm][CI] Add MLA decode accuracy and determinism tests (#50480)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+36/-0); tests/kernels/attention/test_rocm_aiter_mla_decode.py (+225/-0)
LABELS: rocm, ready, ci/build
BODY: Test the production BF16 decode path through rocm_aiter_ops.mla_decode_fwd which is the hot path for DeepSeek-V3/V4 inference on MI300/MI355. ⏎  ⏎ Covers: ⏎ - Smoke test (shape, dtype, finite, non-zero) ⏎ - Accuracy vs PyTorch reference (absorbed MLA formulation) ⏎ - Parametrized sweep: nhead in {16, 128}, batch in {1, 4, 16}, seq in {16, 256} ⏎ - Bitwise determinism across 4 runs ⏎  ⏎ Uses atol=0.01, rtol=0.0, pass rate >= 99.999% for BF16 attention. Re …[truncated]

### L3-62c5e21621  (L3, 2026-08-05, sha 62c5e2162186, PR #50507)
TITLE: [KV Offloading] Support partial-tail prefix reuse with fine-grained prefix matching (#50507)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py (+114/-7); vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py (+246/-37)
LABELS: ready, kv-connector
BODY: ## Purpose ⏎ see https://github.com/vllm-project/vllm/issues/45702 ⏎  ⏎ In hybrid Attention–Mamba models such as Qwen3.6, physical KV blocks are often very large to accommodate the Mamba state. Even with a smaller `prefix_match_unit`, native offloading previously could only store and restore complete physical blocks. As a result, reusable prefix tokens already computed near the end of a block were lost. ⏎  ⏎ This change enables native offloading to: ⏎  …[truncated]

### L3-2e35c529b8  (L3, 2026-08-05, sha 2e35c529b8de, PR #51038)
TITLE: [Bugfix][Quantization] Fix MXFP4 conversion for FlashInfer CUTLASS (#51038)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_ocp_mx_moe.py (+77/-0); vllm/model_executor/layers/fused_moe/oracle/mxfp4.py (+60/-2)
LABELS: bug, ready, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ The MXFP4 backend selector can choose FlashInfer CUTLASS for standard checkpoints, but convert_weight_to_mxfp4_moe_kernel_format has no corresponding branch and raises ValueError after model loading. ⏎  ⏎ This change: ⏎  ⏎ - reorders standard fused gate/up weights, scales, and bias from [w1; w3] to the [w3; w1] layout expected by FlashInfer CUTLASS; ⏎ - applies the backend-specific MXFP8 scale interleave or BF16 weight/scale interleave; ⏎ - adds  …[truncated]

### L3-2dfb8ba590  (L3, 2026-08-06, sha 2dfb8ba59098, PR #50230)
TITLE: [Perf][CUDA] Programmatic dependent launch for the DSA decode kernels (#50230)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu (+40/-10); vllm/models/deepseek_v32/common/kernels.py (+14/-0)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## What this does ⏎  ⏎ Chain the DSA decode kernels with programmatic dependent launch so back-to-back ⏎ small kernels overlap their launch latency: ⏎  ⏎ - the fused norm/rope and fused-q Triton kernels get a `USE_PDL` constexpr with ⏎   `gdc_wait()` / `gdc_launch_dependents()`, launched with `launch_pdl` ⏎ - the NVFP4 quant kernels switch to `cudaLaunchKernelEx` with programmatic stream ⏎   serialization, guarded on `__CUDA_ARCH__ >= 900` and `sm_version >= 90` …[truncated]

### L3-41e7746b82  (L3, 2026-08-06, sha 41e7746b82b4, PR #49599)
TITLE: Update vllm to point to flash-attention commit that builds FA3 with torch stable API. (Retry) (#49599)
SOURCES: path_core, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); .buildkite/check-torch-abi.py (+1/-4)
LABELS: ready, ci/build
BODY: ## Purpose ⏎ Re-attempt of FA3 stable ABI migration like the one in #46644 ⏎  ⏎ Points to the top commit on the PR https://github.com/vllm-project/flash-attention/pull/165. ⏎  ⏎  ⏎ ## Test Plan ⏎ build with new flash-attention commit and check that only stable symbols are exposed.  ⏎  ⏎  ⏎ ## Test Result ⏎  ⏎ build succeeded and library is torch abi stable (see below) ⏎  ⏎ --- ⏎ [details omitted] ⏎  ⏎ Migration progress of vLLM using the Audit Python extension [t …[truncated]

### L3-2e09247c2d  (L3, 2026-08-06, sha 2e09247c2d7b, PR #51215)
TITLE: [Docs] List Intel XPU attention backends (#51215)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: docs/getting_started/quickstart.md (+1/-0)
LABELS: documentation, intel-gpu, ready, verified, build-docs
BODY: ## Motivation ⏎  ⏎ Document the attention backends available on Intel XPU platforms in the quickstart guide. ⏎  ⏎ ## Modifications ⏎  ⏎ - Added the supported Intel XPU attention backends: ⏎   `FLASH_ATTN`, `TRITON_ATTN`, `TRITON_MLA`, `XPU_MLA_SPARSE`, ⏎   `TORCH_SDPA`, and `TURBOQUANT`. ⏎  ⏎ ## Testing ⏎  ⏎ Not run; documentation-only change. ⏎  ⏎ ## Additional context ⏎  ⏎ This change does not affect runtime behavior or model outputs and does not duplicate an  …[truncated]

### L3-d8eabdbfbe  (L3, 2026-08-06, sha d8eabdbfbe93, PR #50578)
TITLE: [ROCm][MLA] Use asm decode for non-divisor small head counts (#50578)
SOURCES: path_core, path_integration+keyword, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen, L3.mla.rocm_aiter
FILES: vllm/envs.py (+15/-0); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+101/-17); tests/kernels/attention/test_rocm_aiter_mla_causal_verify_mask.py (+12/-8); tests/kernels/attention/test_rocm_aiter_mla_head_padding.py (+265/-0)
LABELS: rocm, ready, quantization
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ On `ROCM_AITER_MLA`, MLA decode with fewer than 16 query heads per rank runs the Gluon ⏎ decode kernel (`AiterMLAHelper.use_gluon_decode`). Gluon parallelizes only over heads, so a ⏎ single workgroup marches through the whole KV cache and per-token latency scales linearly ⏎ with context length. The faster asm *persistent* decode requires `num_heads >= 16`; smaller ⏎ head counts are padded up to 16, but `get_mla_padded_q` / `get_mla_unpadded_o …[truncated]

### L3-470297c143  (L3, 2026-08-06, sha 470297c143c9, PR #50980)
TITLE: [HPC Attention Backend] hpc attention backend support bf16 kv cache with fp8 weight  (#50980)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/hpc_attn.py (+64/-29); vllm/model_executor/layers/fused_moe/hpc_moe.py (+1/-5); vllm/model_executor/layers/hpc/rope_norm.py (+10/-2)
LABELS: ready, verified
BODY: ## Purpose ⏎  ⏎ At previous PR [#46020](https://github.com/vllm-project/vllm/pull/46020) , hpc attention backend (`--attention_backend HPC_ATTN`) only support fp8 kv cache (`--kv-cache-dtype fp8_e4m3`) on fp8 model, for example [Hy3-FP8](https://huggingface.co/tencent/Hy3-FP8). This PR add bfloat16 kv cache (`--kv-cache-dtype bfloat16` or `--kv-cache-dtype auto`) support. ⏎  ⏎ ## Test Plan ⏎  ⏎ You can run FP8 model (not only Hy3-FP8) with bfloat16 kv  …[truncated]

### L3-e7b8d59460  (L3, 2026-08-06, sha e7b8d5946095, PR #51247)
TITLE: Fully generalise input embedding handling in Transformers modelling backend (#51247)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/transformers/test_backend.py (+176/-0); vllm/lora/layers/vocal_parallel_embedding.py (+15/-2); vllm/model_executor/layers/vocab_parallel_embedding.py (+1/-1); vllm/model_executor/models/transformers/base.py (+4/-39); vllm/model_executor/models/transformers/causal.py (+8/-4); vllm/model_executor/models/transformers/utils.py (+96/-1)
LABELS: ready
BODY: Before this PR we replaced the entire module returned by `get_input_embeddings` with `VocabParallelEmbedding` and had a special case for scaled input embeddings. ⏎  ⏎ This does not generalise well, particularly if input embeddings perform additional operations that we have not accounted for. ⏎  ⏎ This PR adds `replace_embedding_class` which: ⏎  ⏎ - Fully replaces with `VocabParallelEmbedding` if the embedding was a bare `nn.Embedding` ⏎ - Rebases the embedding …[truncated]

### L3-b38e111d3e  (L3, 2026-08-06, sha b38e111d3e48, PR #50613)
TITLE: [Attention][MLA] Per-request scheduling for MLA chunked context (#50613)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.merge.triton_lse, L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+454/-272); vllm/model_executor/layers/attention/sparse_mla_attention.py (+50/-54); vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py (+8/-7); vllm/v1/attention/backends/mla/prefill/base.py (+1/-1); vllm/v1/attention/backends/mla/prefill/flash_attn.py (+9/-7); vllm/v1/attention/backends/mla/prefill/flashinfer.py (+7/-10); vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py (+7/-10); vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py (+7/-14); vllm/v1/attention/ops/triton_merge_attn_states.py (+1/-114); tests/kernels/attention/test_merge_attn_states.py (+1/-31); (+5 more)
LABELS: ready, nvidia, kimi, k3
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ Implements #50497. MLA prefill chunks are fit into the available workspace rather than forced to be the same size. This can reduce the overall number of chunks and improve prefill latency. ⏎  ⏎ ## Validation ⏎  ⏎ ### Correctness ⏎  ⏎ `pytest tests/v1/attention/test_mla_context_chunks.py -q` passes ⏎  ⏎ GSM8K with `DeepSeek-V2-Lite-Chat`, TP=4, DCP=2, FlashMLA, prefix caching: ⏎  ⏎ | Revision | Strict exact match | Flexible extract | ⏎ | --- |  …[truncated]

### L3-13726c80fe  (L3, 2026-08-06, sha 13726c80fe57, PR #49453)
TITLE: [CPU] Add MLA backend so DeepSeek-V2/V3 can run on CPU (#49453)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.dispatch.registry
FILES: csrc/cpu/mla_decode.cpp (+9/-0); vllm/platforms/cpu.py (+36/-6); vllm/v1/attention/backends/mla/cpu_mla.py (+220/-0); vllm/v1/attention/backends/registry.py (+1/-0); csrc/cpu/cpu_types_arm.hpp (+13/-8); tests/v1/attention/test_cpu_mla_backend.py (+88/-0)
LABELS: documentation, v1, deepseek, cpu
BODY: ## Purpose ⏎  ⏎ Make DeepSeek-V2/V3 style MLA models runnable on the CPU platform. This is a ⏎ reference-quality path (correctness first, performance later) so people without ⏎ a GPU handy can at least kick the tires on MLA models locally. ⏎  ⏎ The plumbing is: ⏎  ⏎ - New `CPUMLABackend` / `CPUMLAImpl` under `vllm/v1/attention/backends/mla/`. ⏎   It inherits `MLACommonBackend` / `MLACommonImpl` so the shared MLA ⏎   scaffolding (weight-absorbed decode vs. non-absorb …[truncated]

### L3-e08111211b  (L3, 2026-08-06, sha e08111211bc9, PR #47106)
TITLE: [Kernel] Support Nvfp4 Cutedsl Moe Swiglu-oai and Relu2(non-gated) Activation (#47106)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer_cutedsl_layout.py (+43/-0); tests/kernels/moe/test_flashinfer_cutedsl_nvfp4_moe.py (+101/-8); vllm/model_executor/layers/fused_moe/config.py (+4/-0); vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py (+26/-1); vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (+9/-2); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py (+2/-0); vllm/model_executor/layers/quantization/modelopt.py (+2/-0); vllm/model_executor/layers/quantization/quark/quark_moe.py (+3/-0); vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+26/-3)
LABELS: ready, nvidia, quantization, verified
BODY: ## Purpose ⏎  ⏎ FlashInfer modified the CuteDSL NVFP4 MoE kernel to support the SwiGLU-OAI activation (https://github.com/flashinfer-ai/flashinfer/pull/3737). The CuteDSL NVFP4 MoE kernel now also supports ReLU² non-gated activation. This PR achieves compatibility with both new activations by passing the activation type and related parameters into the CuteDSL MoE kernel. ⏎  ⏎ ## Test Plan ⏎  ⏎ Test input layout prepare: `.venv/bin/python -m pytest test …[truncated]

### L3-872fd5973e  (L3, 2026-08-06, sha 872fd5973ede, PR #51002)
TITLE: [Bugfix][LoRA] Guard TrtLlm BF16 MoE LoRA gate on activation type (#51002)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+10/-12)
LABELS: bug, ready
ISSUES: #51001 [Bug]: BF16 MoE + LoRA startup crash on non-gated models (TrtLlmBf16LoRAExperts weight-shape assert)
BODY: Fix BF16 MoE + LoRA backend selection for non-gated models (e.g. NemotronH). ⏎  ⏎ The LoRA path in `select_unquantized_moe_backend()` returns early via ⏎ `_trtllm_bf16_lora_supported()` before the modular-kernel oracle runs, so ⏎ activation / `act_and_mul` checks on `TrtLlmBf16LoRAExperts` were skipped. ⏎ Non-gated experts were routed into the gated-only FlashInfer kernel and ⏎ crashed at startup during `profile_run` with: ⏎  ⏎ ``` ⏎ RuntimeError: Check f …[truncated]

### L3-4d341ca829  (L3, 2026-08-06, sha 4d341ca829d7, PR #49206)
TITLE: fix: resolve silent request skipping in PRIORITY scheduling (#49206)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_priority_preemption_bug.py (+149/-0); vllm/v1/core/sched/scheduler.py (+10/-2)
LABELS: ready, v1
ISSUES: #49097 [Bug]: PRIORITY scheduling can silently skip a running request for a full step when the preemption victim was already deferred earlier in the same schedule() call
BODY: ## Purpose ⏎ Fixes #49097. ⏎  ⏎ This PR resolves a logic bug in the `SchedulingPolicy.PRIORITY` preemption path. When a request is preempted, the scheduler's request bookkeeping (`req_index`) was not correctly adjusted if the preempted request appeared earlier in the `self.running` list than the current iteration cursor. This caused the subsequent request to be silently skipped for the entire scheduling step. ⏎  ⏎ ## Test Plan ⏎ 1. Created a regression …[truncated]

### L3-da788334bc  (L3, 2026-08-07, sha da788334bc06, PR #47972)
TITLE: Support DeepSeek-V4 AMD Quark NVFP4 with emulation kernel  (#47972)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+23/-0); tests/evals/gsm8k/configs/DeepSeek-V4-Flash-NVFP4.yaml (+15/-0); tests/evals/gsm8k/configs/DeepSeek-V4-Pro-NVFP4.yaml (+15/-0); tests/evals/gsm8k/configs/models-gfx950-large.txt (+8/-0); tests/quantization/test_quark.py (+97/-0); vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (+1/-0); vllm/model_executor/layers/quantization/quark/quark.py (+32/-2); vllm/model_executor/layers/quantization/quark/quark_moe.py (+7/-6); vllm/model_executor/layers/quantization/quark/schemes/__init__.py (+2/-1); vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py (+115/-4); (+1 more)
LABELS: rocm, ready, ci/build, deepseek, quantization
BODY: ## Purpose ⏎  ⏎ Add support for DeepSeek-V4 AMD Quark mixed-quantized checkpoints, where different parts of the model may use different quantization layouts, including per-block FP8 linear layers and NVFP4 MoE experts. ⏎  ⏎ This PR focuses on the Quark/DeepSeek-V4 model-side pieces needed to correctly identify and load these mixed-quantized checkpoints. It avoids changing generic weight-loading behavior and keeps DeepSeek-V4-specific mappings out of  …[truncated]

### L3-c84789c40b  (L3, 2026-08-07, sha c84789c40b50, PR #51051)
TITLE: [Refactor] Remove kernel dead code (#51051)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flash_attn.fa_utils
FILES: vllm/v1/attention/backends/fa_utils.py (+0/-70); csrc/cpu/cpu_attn_fp8.hpp (+0/-27); csrc/libtorch_stable/cache_kernels.cu (+0/-51); csrc/libtorch_stable/moe/grouped_topk_kernels.cu (+0/-44); csrc/libtorch_stable/quantization/fused_kernels/quant_conversions.cuh (+0/-11); vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm.py (+0/-11); vllm/models/inkling/amd/ops/gluon/rel_mha_decode_gfx950.py (+0/-113); vllm/models/inkling/amd/ops/gluon/utils.py (+0/-69)
LABELS: ready, cpu
BODY: ## Purpose ⏎  ⏎ Remove kernel dead code

### L3-5ac2684976  (L3, 2026-08-07, sha 5ac2684976ee, PR #51293)
TITLE: [CI] Re-enable FI autotune in GSM8K config for Qwen3.5-35B-A3B (#51293)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-DEP2.yaml (+0/-1)
LABELS: ready, qwen
BODY: ## Purpose ⏎  ⏎ Recently [[Bug]: GPU coredump during FlashInfer trtllm_bf16_moe autotune with Qwen3.5-35B-A3B on B200 (DP=2 + EP) #46083](https://github.com/vllm-project/vllm/issues/46083) found bug in FI MoE kernel autotuning. Because of this bug CI job `LM Eval Qwen3.5 Models (B200)` was failing. To fix CI job's failure [[Bugfix] fix qwen3.5 ep weight loading #45002](https://github.com/vllm-project/vllm/pull/45002) [disabled](https://github.com/v …[truncated]

### L3-ae934ba8a5  (L3, 2026-08-07, sha ae934ba8a557, PR #48355)
TITLE: feat: extended EPLB support for Mistral Large 3 and additional MoE backends (#48355)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/distributed/test_eplb_quant_scale_consistency.py (+299/-0); tests/kernels/moe/test_flashinfer_cutedsl_nvfp4_moe.py (+16/-15); tests/quantization/test_trtllm_nvfp4_hidden_dim_padding.py (+6/-1); tests/v1/worker/test_gpu_model_runner_v2_eplb.py (+6/-4); vllm/model_executor/layers/fused_moe/oracle/fp8.py (+13/-2); vllm/model_executor/layers/fused_moe/oracle/nvfp4.py (+5/-0); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py (+16/-2); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py (+3/-1); vllm/model_executor/layers/quantization/fp8.py (+3/-1); vllm/model_executor/layers/quantization/modelopt.py (+3/-1); (+6 more)
LABELS: ready, v1, nvidia, quantization, mrv2, verified, mistral
BODY: ## Purpose ⏎  ⏎ Enable EPLB for MoE models whose quantization config derives per-expert state at load time, and for multi-modal models that nest the MoE language model: ⏎  ⏎ 1. **Nested MoE models.** Add `get_mixture_of_experts_model()` to resolve the `MixtureOfExperts` interface through VLM wrappers that don't implement it themselves. The model runner resolves it once and reuses it. ⏎ 2. **NVFP4 MoE (compressed-tensors W4A4, FlashInfer CuteDSL / TRTL …[truncated]

### L3-34c1cd20a5  (L3, 2026-08-07, sha 34c1cd20a58e, PR #49373)
TITLE: [Bugfix][ROCm] Fix ROCM_AITER_FA & ROCM_AITER_UNIFIED_ATTN QK-Norm+RoPE+KVCache fusion for the packed KV-cache [BLOCKS, HEADS, BLOCK_SIZE, 2*HEAD_DIM] layout (#49373)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.rocm.aiter_fa, L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_fa.py (+16/-7); vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+0/-3); tests/compile/passes/test_rocm_aiter_qk_norm_rope_kvcache_fusion.py (+11/-4)
LABELS: bug, rocm, ready, v1
BODY: ## Purpose ⏎ - FA fused do_qk_norm_rope_kvcache_update used stale unbind(1) (broke on the (nb, nkvh, bs, 2*hs) packed layout, num_kv_heads != 2); fixed via a shared _split_kv_cache (transpose(1,2).split) matching unified. ⏎ - FA fused_qk_norm_rope_kvcache_supported is now gated and not is_shuffle_kv_cache_enabled() — the aiter fused kernel doesn't support the packed layout in shuffle mode ⏎ - Corrected kv_stride_order in the unit test to use the 4D  …[truncated]

### L3-7e85d3a42c  (L3, 2026-08-07, sha 7e85d3a42cc1, PR #50126)
TITLE: [ROCm] Enable pinned memory on supported WSL2 kernels (#50126)
SOURCES: release_notes
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: .buildkite/scripts/xpu/create-xpu-ecr-manifest.sh (+2/-2); vllm/platforms/rocm.py (+27/-1)
LABELS: rocm, ready, ci/build
BODY: Assisted-by: GitHub Copilot ⏎  ⏎  ⏎  ⏎  ⏎ ## Purpose ⏎  ⏎ `RocmPlatform` did not override `is_pin_memory_available()`, so ROCm ⏎ builds running under WSL always fell back to the conservative base ⏎ `Platform.is_pin_memory_available()`, which unconditionally disables ⏎ pinned memory.  This adds a `RocmPlatform` ⏎ override that mirrors the existing kernel-version gating already ⏎ used by `CudaPlatformBase`. ⏎  ⏎ While touching this code path, the WSL warning in  …[truncated]

### L3-4eccf906ca  (L3, 2026-08-07, sha 4eccf906ca87, PR #44857)
TITLE: [Attention] Mamba attention module refactor - Final part (#44857)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/short_conv.py (+3/-3)
LABELS: documentation, ready, v1, kimi
BODY: ## Purpose ⏎  ⏎ Following #43556. ⏎  ⏎ This is the final part for mamba attention module refactor. ⏎  ⏎ what's done in this PR: ⏎ 1. Change ShortConv from CustomOp to PluggableLayer ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-0df620d429  (L3, 2026-08-07, sha 0df620d42926, PR #51288)
TITLE: [Test] Add packed DeepSeek-V4 KV zeroer geometry regression (#51288)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_dsv4_packed_zeroer_geometry.py (+195/-0)
LABELS: ready, deepseek
BODY: ## Summary ⏎  ⏎ Adds a CPU-only constructor-crossing regression for the packed DeepSeek-V4 KV zeroer geometry fixed by #50276. ⏎  ⏎ The test instantiates the real production seams end to end: ⏎  ⏎ - `_get_packed_kv_cache_layout` ⏎ - `_reshape_attention_kv_cache` ⏎ - `DeepseekV4FlashMLABackend` ⏎ - `AttentionGroup` / `KVBlockZeroer.__init__` ⏎  ⏎ It verifies that the constructor metadata separates the full packed-row block stride from the meaningful per-layer zero span …[truncated]

### L3-1c94e8dc7d  (L3, 2026-08-07, sha 1c94e8dc7da4, PR #45187)
TITLE: Add NVFP4 KV 4-over-6 scale search (#45187)
SOURCES: path_core
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/model_executor/layers/attention/attention.py (+5/-2); vllm/v1/attention/backends/flashinfer.py (+15/-12); csrc/libtorch_stable/cache_kernels.cu (+4/-3); csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu (+167/-21); tests/kernels/attention/test_attention_selector.py (+1/-0); tests/kernels/attention/test_cache.py (+67/-0); tests/utils_/test_torch_utils.py (+14/-0); vllm/config/cache.py (+3/-0); vllm/config/vllm.py (+4/-1); vllm/utils/torch_utils.py (+3/-2); (+1 more)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ Add `nvfp4_4over6` as an optional KV cache dtype for NVFP4 KV cache stores. ⏎  ⏎ The existing `nvfp4` behavior is unchanged. Enable 4-over-6 scale search with: ⏎  ⏎ ```bash ⏎ vllm serve <model> --kv-cache-dtype nvfp4_4over6 ⏎ ``` ⏎  ⏎ For each 16-element NVFP4 group, `nvfp4_4over6` evaluates scales derived from ⏎ `max / 6` and `max / 4`, then stores the candidate with lower reconstruction ⏎ error. The physical cache layout and capacity remain NVFP4; the  …[truncated]

### L3-a0056e103e  (L3, 2026-08-07, sha a0056e103e21, PR #50930)
TITLE: [Test] Add ROCm AITER MLA op registration and env gating tests (#50930)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+4/-0); tests/kernels/attention/test_rocm_aiter_mla_op_registration.py (+79/-0)
LABELS: rocm, ci/build
BODY: ## [Test] ROCm AITER MLA op registration, fake-tensor, and env gating tests ⏎  ⏎ Adds kernel-level tests verifying that `rocm_aiter_mla_decode_fwd` custom op is correctly registered, supports fake tensors for `torch.compile` tracing, and respects `VLLM_ROCM_USE_AITER` / `VLLM_ROCM_USE_AITER_MLA` environment variable gating. ⏎  ⏎ ### Tests added (9 total) ⏎ - Op registration existence and callable check ⏎ - `mutates_args=['o']` schema validation ⏎ - `tor …[truncated]

### L3-a801e71cb8  (L3, 2026-08-07, sha a801e71cb86e, PR #48735)
TITLE: [Perf] Improve `--linear-backend` filtering (#48735)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/kernels/linear/__init__.py (+37/-48)
LABELS: ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ The `--linear-backend <name>` currently raises at startup for any linear layer type the backend doesn't cover, so single-scheme backends are unusable. `flashinfer_b12x` is doubly blocked: its only kernel is absent from the NVFP4 auto-selection list, so the filter comes back empty even for NVFP4 layers and the documented opt-in never worked. ⏎  ⏎ ## Changes ⏎  ⏎ 1. **Replace the five duplicated filter blocks** with one `_resolve_backend_ …[truncated]

### L3-e644c8cd8c  (L3, 2026-08-07, sha e644c8cd8c73, PR #51434)
TITLE: [Perf] Optimize DeepSeek V3.2 sequence parallelism (#51434)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/deepseek_v32/test_sequence_parallel.py (+148/-0); vllm/models/deepseek_v32/nvidia/model.py (+57/-70); vllm/models/deepseek_v32/nvidia/mtp.py (+17/-7)
LABELS: ready, deepseek
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ - keep DeepSeek V3.2 hidden and residual states sequence-sharded across every decoder layer, including the dense prefix ⏎ - use the common optimized sequence-parallel gather/reduce-scatter helpers around attention ⏎ - replicate the three dense MLPs under sequence parallelism, matching the Kimi K3 and DeepSeek V4 dataflow ⏎ - shard MTP inputs before `eh_proj` and restore the full output only once ⏎ - add focused decoder and MTP sequence-parall …[truncated]

### L3-45273b8dcb  (L3, 2026-08-07, sha 45273b8dcbfb, PR #51408)
TITLE: [1/N] Harden Transformers modelling backend multi-modal path (#51408)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/processing/test_common.py (+32/-0); tests/models/multimodal/processing/test_transformers_audio.py (+42/-5); tests/models/multimodal/processing/test_transformers_image.py (+33/-5); vllm/model_executor/models/transformers/multimodal.py (+97/-42)
LABELS: ready, multi-modality
BODY: ## Fixed ⏎  ⏎ - Text-only prompts crashed multimodal models with `KeyError: "Modality 'image' not found"`. `apply()` ran `_apply_vision` whenever the model *supported* images, not when items were present; the `mm_token_type_ids is None` guard never fires because HF returns an all-zero tensor. Repro: `BAAI/Emu3-Chat-hf` + a plain text prompt ⏎ - Token-ID prompts had special tokens added twice on re-tokenisation: Gemma3 `[2]` became `[2, 2]`, so ID and t …[truncated]

### L3-653ebb52df  (L3, 2026-08-08, sha 653ebb52dffd, PR #40116)
TITLE: Add torch compile for qwen3_vl encoder (#40116)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_vl.py (+22/-1)
LABELS: ready, stale, qwen
BODY: ## Purpose ⏎  ⏎ torch.compile can accelerate the computation of the encoder ⏎  ⏎ ## Test Plan ⏎  ⏎ ```shell ⏎ python benchmark_qwen3_vl_encoder.py --model Qwen/Qwen3-VL-2B-Instruct \ ⏎     --grid-configs 1-8-8 1-16-16 1-32-32 1-48-48 4-16-16 8-16-16 \ ⏎     --num-warmup 10 --num-iters 30 ⏎ ``` ⏎  ⏎ [details omitted] ⏎  ⏎  ⏎ ## Test Result ⏎  ⏎ ``` ⏎ ======================================================================== ⏎ Summary ⏎ ================================= …[truncated]

### L3-9e6be4a72b  (L3, 2026-08-08, sha 9e6be4a72bd2, PR #50365)
TITLE: [Perf][Sparse MLA] Drop the atomic contention in the index remap (#50365)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/sparse_utils.py (+61/-16); tests/v1/attention/test_indexer_dcp_localize.py (+13/-8); tests/v1/attention/test_sparse_mla_backends.py (+13/-9)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: The sparse-MLA index remap splits each 2048-wide row into 16 column tiles. On the valid-count path every tile atomic-adds into the same per-row counter, so the row total is built under 16-way contention -- and the DCP compaction path uses that same counter as an atomic slot allocator. ⏎  ⏎ Counting is the only reason the tiles need to talk to each other. Give one program the whole row instead: the count becomes an in-register reduction plus a plain …[truncated]

### L3-9b0afeb4f6  (L3, 2026-08-08, sha 9b0afeb4f6c4, PR #51458)
TITLE: [Perf] Avoid some more unnecessary GPU<->CPU syncs (#51458)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/basic_correctness/test_basic_correctness.py (+2/-1); tests/v1/e2e/general/test_mamba_prefix_cache.py (+7/-7); tests/v1/logits_processors/utils.py (+14/-6); vllm/distributed/kv_transfer/kv_connector/utils.py (+8/-3); vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py (+4/-3); vllm/lora/ops/triton_ops/fused_moe_lora_op.py (+2/-2); vllm/model_executor/models/chameleon.py (+32/-4); vllm/model_executor/models/gemma3n_mm.py (+13/-4); vllm/model_executor/models/glm4_1v.py (+11/-4); vllm/model_executor/models/qwen3_omni_moe_thinker.py (+1/-1); (+3 more)
LABELS: speculative-decoding, qwen, kv-connector, mistral
DEEP_STUDY: deep-study performance PR ()
BODY: Split out from https://github.com/vllm-project/vllm/pull/43107. ⏎  ⏎ Each of these blocks the calling thread on a path that runs per forward pass. Found by running CI with VLLM_GPU_SYNC_CHECK=error; the check itself and the places where a sync is deliberate are not included here. ⏎  ⏎ Replace blocking `torch.tensor(..., device=<gpu>)` construction, which copies from pageable host memory, with `async_tensor_h2d`: ⏎ - lora fused_moe `_get_ptr`, on every …[truncated]

### L3-7581c56c86  (L3, 2026-08-08, sha 7581c56c86ce, PR #51457)
TITLE: [Test] Add ROCm AITER FP8 MLA prefill accuracy test (#51457)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_rocm_aiter_mla_fp8_prefill.py (+161/-0)
LABELS: rocm, ready
BODY: Adds a gfx950-only (MI355) accuracy test for the AITER FP8 MLA prefill path (AiterMLAImpl._mla_fp8_prefill_attn -> mla_prefill_ps_asm_fwd + mla_reduce_v1), which previously had no coverage. It drives the real metadata builder (get_ps_metadata_v1) and impl via object.__new__ so it exercises the actual persistent-scheduling metadata contract plus both kernels, and compares the output against a causal SDPA reference on fp8-cast inputs. Skipped on ev …[truncated]

### L3-f18e10a7e1  (L3, 2026-08-09, sha f18e10a7e1c1, PR #50892)
TITLE: Bump Flashinfer version to 0.6.16.post3 (#50892)
SOURCES: path_integration+keyword, subject_keyword, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile (+1/-1); docker/versions.json (+1/-1); requirements/cuda.txt (+2/-2); vllm/v1/worker/gpu_model_runner.py (+11/-3); vllm/model_executor/warmup/kernel_warmup.py (+36/-51)
LABELS: ready, ci/build, nvidia
BODY: ## Purpose ⏎ Bump flashinfer version to 0.6.16.post1 and re-enable persistent cache for autotuning using set_autotune_process_group, which supports distributed auto-tuning across ranks with synchronization to avoid timeout caused by straggler. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-789c4f905e  (L3, 2026-08-10, sha 789c4f905eb3, PR #51566)
TITLE: [CI] Bump CUTLASS DSL to 4.6.2 (#51566)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: requirements/cuda.txt (+2/-2)
LABELS: ready, ci/build, nvidia
ISSUES: #51286 [Bug]: FA4 SM100 split-KV kernels fail to compile with `TYPE_UNSTABLE_JOIN` (nvidia-cutlass-dsl 4.6.0 pin) — crashes Kimi-K3 startup during CuTeDSL warmup
BODY: ## Summary ⏎  ⏎ - bump `nvidia-cutlass-dsl[cu13]` from 4.6.0 to 4.6.2 ⏎ - bump `quack-kernels` from 0.6.1 to 0.6.4, whose package metadata pins CUTLASS DSL 4.6.2 ⏎ - replace the temporary QuACK downgrade with the CUTLASS DSL fix proposed in #51286 ⏎  ⏎ ## Why ⏎  ⏎ CUTLASS DSL 4.6.0 rejects the FA4 SM100 split-KV kernel with `TYPE_UNSTABLE_JOIN`, which can fail B200/B300 attention tests and Kimi-K3 startup during CuTeDSL warmup. CUTLASS DSL 4.6.2 accepts  …[truncated]

### L3-79c865b838  (L3, 2026-08-10, sha 79c865b838e3, PR #51430)
TITLE: [Perf] Narrow DeepSeek V4 eager CUDA graph region (#51430)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/amd/model.py (+1/-1); vllm/models/deepseek_v4/attention.py (+88/-95); vllm/models/deepseek_v4/nvidia/model.py (+1/-1)
LABELS: ready, deepseek, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Narrow the DeepSeek V4 eager CUDA graph region so attention input preparation remains captured, and organize the flow so the execution stages are visible in one linear `forward` method. ⏎  ⏎ The existing `attention_impl` eager break includes Q up-projection, fused Q normalization/RoPE/KV insertion, indexer input preparation, and MLA/indexer compression. This change moves those producer operations into `forward` and keeps only the sparse i …[truncated]

### L3-405bc86768  (L3, 2026-08-10, sha 405bc86768d1, PR #51507)
TITLE: [Perf] Launch the top-k/top-p Triton sampler kernel with 8 warps (#51507)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/sample/ops/topk_topp_triton.py (+7/-0)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ `_topk_topp_kernel` (the Qrita sampler kernel from #42191) runs one program per logits row, and each program serially sweeps the whole vocab row in `BLOCK_SIZE=8192` tiles. Triton's default `num_warps=4` leaves an 8192-wide fp32 tile at 16 elements per lane, so per-tile latency — which directly bounds kernel latency — is warp-starved. This kernel is on the hot path for seeded / per-request-generator sampling on CUDA (FlashInfer reject …[truncated]

### L3-3dafaef027  (L3, 2026-08-10, sha 3dafaef02735, PR #51573)
TITLE: [Bugfix][Core] Emit --no-{key} for false BooleanOptionalAction flags in YAML config (#51573)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/utils_/test_argparse_utils.py (+22/-0); vllm/utils/argparse_utils.py (+2/-0)
LABELS: bug, ready
ISSUES: #51401 [Bug]: VLLM silently ignores falsey yaml configuration options
BODY: ## Summary ⏎  ⏎ Fixes #51401 ⏎  ⏎ `--config` YAML files silently drop `false` boolean values. For `BooleanOptionalAction` flags (e.g. `--enable-flashinfer-autotune`) whose default is `None` and gets resolved later by optimization-level logic, the user's explicit `false` was lost — causing unexpected behavior (OOM in the reported case, as flashinfer autotune warmup ran despite being explicitly disabled). ⏎  ⏎ ## Root Cause ⏎  ⏎ In `FlexibleArgumentParser.load_con …[truncated]

### L3-b22afe45ac  (L3, 2026-08-10, sha b22afe45ac79, PR #51148)
TITLE: [CPU] Enable GPTQ and AWQ quantization for s390x (#51148)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: cmake/cpu_extension.cmake (+6/-0); csrc/cpu/cpu_types_vxe.hpp (+78/-20); csrc/cpu/torch_bindings.cpp (+1/-1); docs/getting_started/installation/cpu.md (+2/-2); docs/getting_started/installation/cpu.s390x.inc.md (+1/-1)
LABELS: documentation, ci/build, cpu, quantization
BODY: ## Purpose ⏎ Enable GPTQ and AWQ quantization for s390x ⏎ ## Test Plan ⏎ Run inference server with both quantization models and check the result ⏎ ## Test Result ⏎ AWQ Model ⏎ ``` ⏎ [root@b314lp81 ~]# curl -s http://localhost:8000/v1/completions     -H "Content-Type: application/json"     -d '{ ⏎       "model": "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ", ⏎       "prompt": "IBM Z mainframes are", ⏎       "max_tokens": 64 ⏎     }' | jq ⏎ { ⏎   "id": "cmpl-b79bdb7c …[truncated]

### L3-37fbf52084  (L3, 2026-08-10, sha 37fbf5208413, PR #51011)
TITLE: [ROCm][MLA] [K3] Fix fp8 KV cache decode on the AITER MLA backend (#51011)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+64/-35); tests/kernels/attention/test_rocm_aiter_mla_head_padding.py (+13/-12); tests/v1/attention/test_rocm_aiter_mla_fp8_decode_routing.py (+154/-0); tests/v1/attention/test_rocm_aiter_mla_mtp_split.py (+135/-16)
LABELS: rocm, ready, verified, k3
BODY: ## Purpose ⏎  ⏎ Kimi-K3 at TP8 has 12 MLA heads per rank and cannot serve correctly with ⏎ `--kv-cache-dtype fp8` on the ROCm AITER MLA backend. On unmodified main, a full ⏎ GSM8K run at that configuration scores **74.00%** with **285 of 1319 answers ⏎ degenerate**; with this PR it scores **97.19%** with **none**. The backend fails ⏎ three different ways depending on head count and query length, and the third is ⏎ the dangerous one because nothing repor …[truncated]

### L3-63ac04a61e  (L3, 2026-08-10, sha 63ac04a61e62, PR #50484)
TITLE: [Kimi-K3] DCP support (#50484)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake, L3.flashinfer.trtllm_gen, L3.mla.common_v1, L3.mla.flashinfer
FILES: vllm/model_executor/layers/attention/mla_attention.py (+87/-48); vllm/model_executor/layers/attention/sparse_mla_attention.py (+22/-3); vllm/v1/attention/backends/mla/flashinfer_mla.py (+4/-2); vllm/v1/attention/backends/mla/tokenspeed_mla.py (+2/-1); vllm/v1/attention/ops/common.py (+51/-4); vllm/v1/attention/ops/dcp_alltoall.py (+20/-11); vllm/v1/attention/ops/dcp_utils.py (+740/-0); CMakeLists.txt (+5/-0); csrc/libtorch_stable/attention/dcp_utils/dcp_direct_a2a_lse_reduce.cu (+381/-0); csrc/libtorch_stable/attention/dcp_utils/dcp_direct_common.cuh (+92/-0); (+9 more)
LABELS: ready, ci/build, v1, nvidia, kimi, k3
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ Follow-up to #50000 adding decode context parallelism for Kimi-K3: ⏎  ⏎ - direct symmetric-memory DCP A2A output/LSE reduction, including empty-KV-shard masking ⏎ - DCP support for the fused Kimi-K3 MLA layer ⏎ - NVLS-multicast direct query gather and multimem chunked-context KV gather ⏎ - direct publication of query shards into the consumer-final buffer, removing the staging-to-final materialization by @foraxe ⏎  ⏎ This replaces #50055. ⏎  …[truncated]

### L3-cf8f3a3bb2  (L3, 2026-08-10, sha cf8f3a3bb23e, PR #51657)
TITLE: [2/N] Harden Transformers modelling backend multi-modal path (#51657)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/transformers/test_backend.py (+54/-0); vllm/model_executor/models/transformers/base.py (+52/-20); vllm/model_executor/models/transformers/multimodal.py (+170/-116)
LABELS: ready
BODY: Continues from https://github.com/vllm-project/vllm/pull/51408 ⏎  ⏎ ## Added ⏎  ⏎ - `--mm-encoder-only` and `--limit-mm-per-prompt <modality>=0` now skip weights for this backend, via a `_mark_model_components` hook around `AutoModel.from_config` ⏎ - Audio encoders can be torch compiled; `compile_mm_encoder` was hardcoded to the image encoder ⏎  ⏎ ## Fixed ⏎  ⏎ - `_get_prompt_updates` returned `None` against a declared `Sequence[PromptUpdate]` ⏎ - The over …[truncated]

### L3-d40c3e3c00  (L3, 2026-08-10, sha d40c3e3c00ab, PR #50693)
TITLE: Fix DSpark warmup without sparse index buffer (#50693)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/nvidia/flashmla.py (+5/-2)
LABELS: ready
ISSUES: #50615 [Bug]: DSpark spec decode dies in profile_run — forward_mqa warmup asserts topk_indices_buffer, which a drafter never has (regression from #50298)
BODY: ## Purpose ⏎  ⏎ Fixes #50615. ⏎  ⏎ During DSpark startup, `profile_run` calls `forward_mqa` without attention ⏎ metadata to size the warmup workspace. For the SWA-only draft layer ⏎ (`compress_ratio <= 1`), `topk_indices_buffer` is intentionally not allocated, ⏎ but the warmup path asserted that the buffer existed before setting `top_k` to ⏎ zero. This makes the engine fail after loading the full target and draft model. ⏎  ⏎ This change handles the SWA-only case bef …[truncated]

### L3-31cd109f18  (L3, 2026-08-10, sha 31cd109f18f1, PR #40958)
TITLE: [ROCm][CI] Extend ROCm AITER MHA (FA) coverage (#40958)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_aiter_flash_attn.py (+0/-258); tests/kernels/attention/test_rocm_aiter_fa.py (+906/-0)
LABELS: rocm, ready
BODY: This PR consolidates the ROCm AITER flash-attention coverage into one backend-named file: `test_rocm_aiter_fa.py`. It merges the old direct kernel stress file and a new set of tests so the backend and the unique direct-kernel checks live together. That gives the test the same name as the backend it actually exercises, which makes the tree easier to read. There is no intended kernel behavior change here. The goal is to make the test location and b …[truncated]

### L3-8977ea8895  (L3, 2026-08-10, sha 8977ea8895b1, PR #50333)
TITLE: [Perf] Skip detokenization in offline beam search (#50333)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/entrypoints/generate/beam_search/offline.py (+2/-0)
LABELS: frontend, ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## What & why ⏎  ⏎ Follow-up to #46422, which skipped detokenization in online beam search and noted the offline path has the identical pattern. ⏎  ⏎ Offline beam search asks for `logprobs = 2 * beam_width` on every internal per-step request. The engine detokenizes all of those token ids into strings on every step, but beam search never reads them. It ranks beams by `cum_logprob` and decodes the final text itself. This PR sets `detokenize=False` on the t …[truncated]

### L3-243c63baf5  (L3, 2026-08-10, sha 243c63baf523, PR #51603)
TITLE: [V1][Scheduler] Apply Mamba alignment before encoder caps (#51603)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_scheduler.py (+83/-0); vllm/v1/core/sched/scheduler.py (+22/-20)
ISSUES: #47738 [Bug]:  Including two images in a single request with Qwen3.6-27B causes an infinite rollback.
BODY: ## Summary ⏎  ⏎ - Apply Mamba block alignment before multimodal encoder scheduling can cap a prefill chunk. ⏎ - Use EAGLE's shifted encoder window consistently when calculating the overlapping embedding range. ⏎ - Preserve the existing align-mode policy that waits for a fresh scheduler budget when a complete Mamba block normally fits in one step. ⏎ - Add a scheduler-level regression for two images that each fit the encoder cache individually but do not fit …[truncated]

### L3-fac808b36f  (L3, 2026-08-10, sha fac808b36f50, PR #49436)
TITLE: [Perf][Hybrid] 3D-grid tiling of the state-copy Triton kernels (#49436)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/mamba/test_memcpy_u64_tiled.py (+147/-0); tests/kernels/mamba/test_precopy_mamba_align.py (+24/-12); vllm/v1/worker/mamba_utils.py (+176/-56)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Follow-up to https://github.com/vllm-project/vllm/pull/48110. ⏎  ⏎ https://github.com/vllm-project/vllm/pull/48110 optimized the copy of the states for hybrid models using 64bit load/stores. However the state copy is still underutilizing the HBM memory at small batch and the kernel requires 8B-aligned state tensors. ⏎  ⏎ This PR further improves the performance by adding a 3D grid and lift the hard 8B alignment precondition by using a h …[truncated]

### L3-99e62b802c  (L3, 2026-08-10, sha 99e62b802c2a, PR #49519)
TITLE: [Bugfix][Model Loader] Defer post-load attention weight processing (#49519)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/attention/__init__.py (+12/-0); tests/model_executor/model_loader/test_reload.py (+77/-0); vllm/model_executor/model_loader/reload/layerwise.py (+14/-17); vllm/model_executor/model_loader/utils.py (+2/-7)
LABELS: bug, ready
BODY: ## Summary ⏎  ⏎ Align layerwise weight loading and reload with the standard post-load attention ⏎ lifecycle: ⏎  ⏎ - identify deferred attention layers through one shared predicate; ⏎ - cover `AttentionLayerBase` implementations with a post-load hook, including ⏎   pluggable multi-head latent attention implementations; ⏎ - explicitly cover `MMEncoderAttention`, which has the same lifecycle but does ⏎   not inherit `AttentionLayerBase`; ⏎ - use the predicate in both t …[truncated]

### L3-0914ed2e81  (L3, 2026-08-10, sha 0914ed2e816f, PR #51725)
TITLE: [Perf] Adaptive budget for spec scheduled input tokens, ~60% better Kimi K3 DSpark TTFT (#51725)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_scheduler.py (+24/-0); tests/v1/core/utils.py (+2/-0); vllm/config/vllm.py (+8/-19); vllm/v1/core/sched/scheduler.py (+20/-5)
LABELS: ready, kimi, k3
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Let's consider `max_num_seqs=1024`and `max_num_batched_tokens=8192`  (by default)  ⏎  ⏎ Actual Request num | Old logic scheduled tokens | Now ⏎ -- | -- | -- ⏎ 1 | 2048 | 8186 ⏎ 32 | 2048 | 8000 ⏎ 128 | 2048 | 7424 ⏎ 1024 | 2048 | 2048 ⏎  ⏎ This PR make the scheduled tokens much larger when request num is small by using an adaptive strategy ⏎  ⏎ This is **extremely helpful** when request num is small and give very huge perf improvement ⏎  ⏎ ## Te …[truncated]

### L3-c76a425278  (L3, 2026-08-10, sha c76a425278bc, PR #51739)
TITLE: [Kernel] Optimize long-context MLA cache gathers (#51739)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+2/-2); .buildkite/test_areas/kernels.yaml (+1/-1); benchmarks/kernels/benchmark_cp_gather.py (+248/-0); csrc/libtorch_stable/cache_kernels.cu (+326/-232); tests/kernels/attention/test_cache.py (+179/-0); tests/kernels/test_cp_gather_fp8.py (+54/-0)
LABELS: performance, ready, ci/build
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ - schedule MLA cache gathers by logical cache page instead of independently mapping every output token ⏎ - flatten page copies/conversions to generate coalesced vector loads and stores, including partial first/last pages and nonzero `seq_starts` ⏎ - coalesce FP8-to-BF16 stores and tune CTA geometry for long, uneven prefill workloads ⏎ - add long-context correctness coverage and a reusable kernel benchmark for 60K-300K requests ⏎  ⏎ ## Why ⏎  ⏎ The  …[truncated]

### L3-c3cac8c63d  (L3, 2026-08-10, sha c3cac8c63d91, PR #49815)
TITLE: [Bugfix][MiMo] Apply vision attention sinks in the window attention path (#49815)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/triton_prefill_attention.py (+14/-2); tests/models/multimodal/test_mimo_v2_omni.py (+91/-0); vllm/model_executor/models/mimo_v2_omni.py (+27/-10)
LABELS: bug, ready, multi-modality
ISSUES: #47864 [Bug]:MiMo Code mimo_v2_omni.py ERROR
BODY: Fixes #47864. ⏎  ⏎ `MiMoVisionAttention` allocates `self.sinks` only when the block is not in `fullatt_block_indexes`, which is exactly the set of blocks that run `_forward_window_attn`. That path never read the parameter, so the sink weights were loaded from the checkpoint and dropped. `XiaomiMiMo/MiMo-V2.5` ships `visual.blocks.N.attn.sinks` for exactly those blocks, 24 of its 28, and none for the full attention blocks `[0, 9, 18, 27]`. ⏎  ⏎ ## Approac …[truncated]

### L3-2acb055ed7  (L3, 2026-08-10, sha 2acb055ed768, PR #44201)
TITLE: [CPU][Zen] Route BF16 MoE inference through zentorch on AMD (#44201)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: setup.py (+1/-1); tests/kernels/moe/test_zen_cpu_fused_moe.py (+221/-0); vllm/model_executor/kernels/linear/zentorch_utils.py (+63/-1); vllm/model_executor/layers/fused_moe/experts/cpu_moe.py (+31/-1)
LABELS: rocm, ready, ci/build, cpu
BODY: Routes CPU MoE on AMD Zen through a zentorch-backed fused MoE path in `CPUFusedMOE`, ahead of the existing AMX grouped-GEMM / OneDNN / per-expert PyTorch fallbacks: ⏎  ⏎ - **`forward_zentorch`** — MoE FFN via `torch.ops.zentorch.zentorch_fused_moe` (standard `[E, ...]` expert weights; no `cpu_prepack_moe_weight`). ⏎ - **`is_zentorch_moe_supported()`** — capability check at `CPUFusedMOE` init: op registered, `moe_config.is_act_and_mul` when present,  …[truncated]

### L3-07443bea29  (L3, 2026-08-11, sha 07443bea29db, PR #51482)
TITLE: [BugFix][Core] free_blocks: restore prepend (LIFO) reuse order when prefix caching is off (#51482)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/core/block_pool.py (+1/-1)
LABELS: bug, ready
BODY: ## What ⏎  ⏎ #48017 was described as a **pure no-op** that skips the pointless LRU hash-split in `BlockPool.free_blocks` when prefix caching is disabled — but the merged condition (`block.block_hash is None and self.enable_caching`) routes every freed block through `append_n` instead of `prepend_n` on that path. Since allocation pops from the **head** of `FreeKVCacheBlockQueue`, this silently flipped block reuse from LIFO (immediate reuse of a small  …[truncated]

### L3-608c12473f  (L3, 2026-08-11, sha 608c12473f3f, PR #51733)
TITLE: [Attention] Fix MLA prefill workspace allocation size (#51733)
SOURCES: path_core, subject_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+1/-5); vllm/model_executor/layers/attention/sparse_mla_attention.py (+1/-4); tests/distributed/test_dcp_direct_a2a_lse_reduce.py (+5/-2)
LABELS: ready
BODY: ## Purpose ⏎ With [#50613]([Attention][MLA] Per-request scheduling for MLA chunked context), MLA prefill workspace no longer requires to be at least `max-num-seqs` * `block_size`. This requirement is added back in https://github.com/vllm-project/vllm/pull/50484. This PR reverts it. ⏎  ⏎ ## Test Plan ⏎ - Test `test_mla_backends.py` and `test_sparse_mla_backends.py`. ⏎ - Test Kimi K3 DCP 8 on B300 GSM8k ⏎  ⏎ ## Test Result ⏎ ``` ⏎ |Tasks|Version|     Filter …[truncated]

### L3-6c95a641e9  (L3, 2026-08-11, sha 6c95a641e95c, PR #49315)
TITLE: [2/N][Feat][Perf] Add new warmup infrastructure for JITs. Add predicate filtering for JIT warmup, and migrate Inkling FA4 (#49315)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/kernel.py (+1/-1); vllm/v1/attention/backends/mla/prefill/flash_attn.py (+79/-56); tests/model_executor/test_jit_warmup.py (+65/-0); tests/models/inkling/test_fa4_rel_attention.py (+4/-4); tests/models/inkling/test_fa4_warmup.py (+84/-29); vllm/model_executor/warmup/cutedsl_warmup.py (+1/-1); vllm/model_executor/warmup/fa4_cutedsl_warmup.py (+21/-2); vllm/model_executor/warmup/jit_warmup.py (+149/-52); vllm/models/inkling/nvidia/attention.py (+2/-22); vllm/models/inkling/nvidia/ops/__init__.py (+4/-2); (+2 more)
LABELS: ready, v1
DEEP_STUDY: deep-study performance PR ()
BODY: ### Description ⏎ This PR migrates Inkling FA4 attention warmup to the shared JIT warmup contract introduced in #47451, ~and deprecates the legacy CuTeDSL warmup path~ (there is a conflict with the Kimi K3 integration #50089 that prevents this deprecation from being completed. It will be addressed in a future PR). ⏎  ⏎ See https://github.com/vllm-project/vllm/issues/49349 for more context ⏎  ⏎ ### Motivation ⏎ - Respect `enable_jit_warmup` and run Inkl …[truncated]

### L3-5426311d91  (L3, 2026-08-11, sha 5426311d9115, PR #47896)
TITLE: [Kernel][ROCm][Perf] FlyDSL decode-attention kernel for 4-bit TurboQuant KV cache  (#47896)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+23/-8); vllm/v1/attention/backends/turboquant_attn.py (+349/-45); vllm/v1/attention/ops/flydsl_kernels/__init__.py (+2/-0); vllm/v1/attention/ops/flydsl_kernels/tq_decode.py (+823/-0); vllm/v1/attention/ops/flydsl_kernels/tq_decode_gqa6.py (+875/-0); vllm/v1/attention/ops/flydsl_turboquant_decode.py (+874/-0); vllm/v1/attention/ops/turboquant_soa/__init__.py (+3/-0); vllm/v1/attention/ops/turboquant_soa/triton_turboquant_decode.py (+643/-0); vllm/v1/attention/ops/turboquant_soa/triton_turboquant_decode_v2.py (+669/-0); vllm/v1/attention/ops/turboquant_soa/triton_turboquant_store.py (+524/-0); (+3 more)
LABELS: documentation, performance, new-model, rocm, structured-output, frontend, intel-gpu, speculative-decoding, ready, ci/build
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ This PR upstreams the FlyDSL TurboQuant 4-bit KV-cache decode kernel developed and optimized by the AMD team. For background on the TurboQuant algorithm and its role in agentic vLLM serving, see our blog write-up [TurboQuant blog] (https://rocm.blogs.amd.com/artificial-intelligence/turboquant-vllm-agentic/README.html) ⏎  ⏎ This PR adds a custom FlyDSL TurboQuant 4-bit KV-cache decode kernel for ROCm / MI355X (gfx950) as an opt-in altern …[truncated]

### L3-513f83e7ec  (L3, 2026-08-11, sha 513f83e7ecee, PR #51756)
TITLE: [Bugfix] Take the sliding window from the layer, not the KV cache group (#51756)
SOURCES: path_core
ARTIFACT_HINTS: L3.flash_attn.v1_backend
FILES: vllm/v1/attention/backends/cpu_attn.py (+32/-4); vllm/v1/attention/backends/flash_attn.py (+8/-22); tests/v1/attention/test_group_sliding_window.py (+48/-0); tests/v1/e2e/general/test_correctness_sliding_window.py (+28/-0)
LABELS: bug, ready, cpu
BODY: One KV cache group can hold both windowed and global layers — Gemma-3 with `--disable-hybrid-kv-cache-manager` promotes its sliding-window layers to full-attention storage, and the merged group spec still records the window. FlashAttention read that window off the group spec and applied it to every layer in the group, so the global layers silently lost everything older than the window; CPU attention did the same for its group-wide scheduler metad …[truncated]

### L3-52be12cfac  (L3, 2026-08-11, sha 52be12cfac0c, PR #50569)
TITLE: feat: allow shared expert overlapping for FlashInfer one-sided all-to-all (#50569)
SOURCES: subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/runner/shared_experts.py (+9/-2)
LABELS: ready
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Allow shared expert overlap when using EPLB with the `flashinfer_nvlink_one_sided` all2all backend. Previously, PR #28377 disabled shared expert overlap for all non-`allgather_reducescatter` backends under EPLB due to correctness issues observed with `deepep_low_latency`. This PR exempts `flashinfer_nvlink_one_sided` from that restriction after verifying it produces correct results with overlap enabled. ⏎  ⏎ ## Test Plan ⏎  ⏎ Served `de …[truncated]

### L3-e3fe212eaf  (L3, 2026-08-11, sha e3fe212eaf0d, PR #51749)
TITLE: [Bugfix] Generalize KV block zeroing to `AttentionSpec` (#51749)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_single_type_kv_cache_manager.py (+68/-2); tests/v1/worker/test_kv_block_zeroer.py (+62/-1); vllm/v1/core/single_type_kv_cache_manager.py (+3/-5); vllm/v1/worker/utils.py (+1/-2)
LABELS: bug, ready
BODY: ## Summary ⏎  ⏎ - Record newly allocated `AttentionSpec` block IDs when worker-side KV zeroing is required. ⏎ - Include all allocating attention groups in `KVBlockZeroer`. ⏎ - Add focused sliding-window and chunked-local regressions. ⏎  ⏎ ## Root cause ⏎  ⏎ `KVCacheConfig.needs_kv_cache_zeroing` is enabled for hybrid Mamba and mixed-precision caches, but the scheduler only reported an exact allowlist of full/MLA block types and the worker zeroer only pro …[truncated]

### L3-b2ab096017  (L3, 2026-08-11, sha b2ab0960178c, PR #51857)
TITLE: [Docs] Fix broken autorefs cross-reference in TurboQuant v2 docstring (#51857)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/turboquant_soa/triton_turboquant_decode_v2.py (+1/-1)
LABELS: quantization
BODY: ## Purpose ⏎  ⏎ The docs build emits exactly one `WARNING`, from the `triton_turboquant_decode_v2` module docstring: ⏎  ⏎ ``` ⏎ WARNING -  mkdocs_autorefs: api/vllm/v1/attention/ops/turboquant_soa/triton_turboquant_decode_v2.md: from .../triton_turboquant_decode_v2.py:3: (vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode_v2) Could not find cross-reference target 'j' ⏎ ``` ⏎  ⏎ The docstring writes `pair_table[i][j]` unquoted. Markdown parses `[i][j]` …[truncated]

### L3-4f2f31b82e  (L3, 2026-08-11, sha 4f2f31b82e03, PR #51363)
TITLE: [Bugfix][Attention] Forward per-head FP8 descales through FA4 (#51363)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fa4_cutedsl
FILES: vllm/vllm_flash_attn/flash_attn_interface.py (+8/-0)
LABELS: bug, ready
BODY: ## Summary ⏎  ⏎ Forward per-head FP8 Q/K/V descales from vLLM's FlashAttention interface into the FA4 CuTe forward kernel. ⏎  ⏎ ## Root cause ⏎  ⏎ The FA3 path forwarded `q_descale`, `k_descale`, and `v_descale`, but the FA4 path dropped them. Per-head FP8 Q/K/V tensors were therefore consumed without their calibration scales, producing garbage output for llm-compressor attention/KV-quantized checkpoints. ⏎  ⏎ ## Changes ⏎  ⏎ - Pass Q/K/V descales to the F …[truncated]

### L3-a311916a29  (L3, 2026-08-11, sha a311916a291c, PR #46849)
TITLE: [MRV2][Spec] Fuse AR speculator multi-step decodes back into one CUDA graph (#46849)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.triton.v1_backend, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+13/-0); vllm/v1/attention/backends/flash_attn.py (+94/-42); vllm/v1/attention/backends/mla/sparse_swa.py (+35/-0); vllm/v1/attention/backends/mla/triton_mla.py (+5/-0); vllm/v1/attention/backends/triton_attn.py (+5/-0); docs/design/model_runner_v2.md (+8/-0); tests/v1/worker/test_gpu_autoregressive_speculator.py (+166/-0); vllm/models/deepseek_v4/amd/rocm.py (+4/-0); vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py (+151/-7); vllm/v1/worker/utils.py (+11/-0)
LABELS: documentation, ready, v1, nvidia, mrv2
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ This PR restores fused multi-step CUDA graph execution for autoregressive speculative decoding in Model Runner V2. ⏎  ⏎ #41162 fixed stale attention metadata by rebuilding it and replaying a separate CUDA graph for every draft step. While correct, that design reintroduces per-step Python dispatch, metadata construction, and CUDA graph launch overhead. ⏎  ⏎ This PR instead captures the post-prefill draft loop into one CUDA graph. Attenti …[truncated]

### L3-0fb9897df1  (L3, 2026-08-11, sha 0fb9897df129, PR #51819)
TITLE: [Bugfix][MoE] Support GELU tanh in FlashInfer B12x MoE (#51819)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py (+6/-1)
LABELS: bug, ready, nvidia
BODY: ## Summary ⏎  ⏎ Adds `MoEActivation.GELU_TANH` support to the FlashInfer B12x MoE backend so Gemma 4 NVFP4 can use `--moe-backend flashinfer_b12x`. ⏎  ⏎ ## Validation ⏎  ⏎ Validated with `nvidia/Gemma-4-26B-A4B-NVFP4` on DGX Spark using FlashInfer 0.6.16.post3

### L3-1d2d83a07f  (L3, 2026-08-11, sha 1d2d83a07fd3, PR #49718)
TITLE: [Attention] Add FlashInfer XQA decode support on SM12x (#49718)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+16/-9); vllm/v1/attention/backends/flashinfer.py (+212/-31); tests/kernels/attention/test_use_trtllm_attention.py (+14/-0); tests/v1/attention/test_attention_backends.py (+171/-17)
LABELS: ci/build, v1, nvidia
BODY: ## Summary ⏎  ⏎ Enable FlashInfer XQA decode on SM120/SM121 through FlashInfer's dedicated XQA API. ⏎  ⏎ - Select XQA for SM12x decode. ⏎ - Support uniform and ragged speculative decode through `q_len_per_req` and `q_cu_seq_lens`. ⏎ - Build the packed causal or non-causal draft mask required by XQA. ⏎ - Support sliding-window decode, attention sinks, non-causal draft attention, and uniform-batch CUDA graphs on SM12x. ⏎ - Keep the existing SM90 shared-XQA path an …[truncated]

### L3-3e372c5ff2  (L3, 2026-08-11, sha 3e372c5ff234, PR #51837)
TITLE: [Bugfix][ROCm] Give KV-first attention blocks their own page in hybrid models (#51837)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/attn_utils.py (+74/-0); tests/v1/worker/test_attn_utils.py (+103/-1)
LABELS: bug, rocm, ready-run-all-tests, mrv2
BODY: ## Purpose ⏎  ⏎ On MI300, DSpark speculative decoding on `RedHatAI/Qwen3.6-35B-A3B-NVFP4` produces garbage for most requests in a batch. GSM8K accuracy drops to 0.20 against 0.95 for the same target model without speculation, and `test_dspark_correctness_and_acceptance_rate[qwen3.6-speculators]` fails. The same test passes on H200. ⏎  ⏎ Hybrid models share a single page pool between the full-attention layers, the Mamba/GDN state layers, and the draft …[truncated]

### L3-61874f9842  (L3, 2026-08-11, sha 61874f9842bd, PR #51612)
TITLE: [4/N][KV-Cache Layout Refactor] Promote local KV cache specs via a class-changing replace helper (#51612)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/core/kv_cache_utils.py (+20/-35); vllm/v1/kv_cache_interface.py (+15/-1)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ `_promote_local_kv_cache_specs` rebuilds each promoted spec with a hand-written constructor call per class, and the explicit field lists have drifted: the MLA and chunked-local promotions silently drop `kv_quant_mode` (mis-sizing promoted pages for quantized KV caches), and the chunked-local promotion drops `head_size_v`. ⏎  ⏎ This PR adds `replace_as()` — `dataclasses.replace` generalized to rebuild a spec as a different class — and  …[truncated]

### L3-cc668c5b72  (L3, 2026-08-11, sha cc668c5b72e7, PR #51854)
TITLE: [CI][Bugfix][V1] Remove stale FlashAttention metadata arguments (#51854)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_autoregressive_speculator.py (+0/-2)
LABELS: bug, ready, ci-failure
BODY: - Regression: [#51756](https://github.com/vllm-project/vllm/pull/51756) removed `FlashAttentionMetadata.sliding_window`. ⏎ - Detection: [Buildkite main build #83388](https://buildkite.com/vllm/ci/builds/83388) first exposed two stale V1 test constructors; [the V1 Core job in build #83390](https://buildkite.com/vllm/ci/builds/83390/summary?jid=019ff19e-ec7d-4dec-bc99-afb6e0a2c2e2&tab=output) reproduced the same failure. ⏎ - This PR removes the obsol …[truncated]

### L3-f97e502969  (L3, 2026-08-11, sha f97e5029698c, PR #51738)
TITLE: [Perf] Avoid more GPU<->CPU syncs on the model execution path (#51738)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+2/-0); vllm/v1/attention/backends/utils.py (+8/-2); vllm/model_executor/models/audioflamingo3.py (+2/-2); vllm/model_executor/models/diffusion_gemma.py (+20/-12); vllm/model_executor/models/gemma4_mm.py (+15/-1); vllm/model_executor/models/glm4_1v.py (+2/-2); vllm/model_executor/models/interns1.py (+6/-2); vllm/model_executor/models/internvl.py (+3/-1); vllm/model_executor/models/keye.py (+5/-4); vllm/model_executor/models/phi3v.py (+3/-2); (+5 more)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: Follow-on to the earlier sync-avoidance work, splitting further fixes out of the VLLM_GPU_SYNC_CHECK branch. Each of these removes a host roundtrip rather than suppressing it: ⏎  ⏎ - kv-sharing fast prefill: derive the fast-prefill decode metadata from host-side data. `total_num_decode_tokens` is just the number of logits indices, and the max per-request logits count is plumbed down from the runner (which already has it on the host as `num_sampled_ …[truncated]

### L3-7f9173dfa2  (L3, 2026-08-11, sha 7f9173dfa24d, PR #50654)
TITLE: [ROCm][Perf] Kimi-K3 Fused kernel for KDA decode (#50654)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+22/-0); benchmarks/kernels/benchmark_kimi_k3_kda_decode.py (+291/-0); csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel_rocm.cu (+800/-0); tests/models/kimi_k3/test_amd_kda_decode.py (+291/-0); vllm/models/kimi_k3/amd/kda.py (+89/-9); vllm/models/kimi_k3/amd/ops/kda_decode.py (+99/-0)
LABELS: performance, rocm, ready, ci/build, kimi, k3
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Kimi's KDA layers do three things per decode step for every sequence and every value head: ⏎  ⏎ 1. Causal conv1d update over the Q/K/V projections, advancing a small per-sequence convolution state ⏎ 2. Gated delta-rule recurrence, which updates a `[head_dim, head_dim]` recurrent state and produces the attention output ⏎ 3. gated RMSNorm on that output ⏎  ⏎ On ROCm, currently these run as separate Triton kernels. This PR mirrors the CUDA pat …[truncated]

### L3-20727e2841  (L3, 2026-08-12, sha 20727e2841ed, PR #50268)
TITLE:  [Hardware][AMD] Enable fused bf16→fp32 router GEMM on ROCm (#50268)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_gate_linear_rocm_dispatch.py (+98/-0); vllm/model_executor/layers/fused_moe/router/gate_linear.py (+14/-3)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ This pull request addresses issue #50267 by enabling fused bf16→fp32 GEMM operations on AMD ROCm hardware for MoE router gates. ⏎  ⏎ ## Problem Statement ⏎  ⏎ The MoE router gate requires `out_dtype=torch.float32` for the `grouped_topk` operation. However, on ROCm, the fused fp32-output GEMM implementation was gated exclusively to CUDA platforms. This caused a fallback to bf16 GEMM followed by a separate cast operation, creating a stand …[truncated]

### L3-466855a2bf  (L3, 2026-08-12, sha 466855a2bfc5, PR #47017)
TITLE: [ROCm] Enable DeepSeek-V4 on gfx11 (#47017)
SOURCES: release_notes
ARTIFACT_HINTS: L3.platform.rocm_selection
FILES: vllm/_aiter_ops.py (+16/-1); vllm/model_executor/layers/sparse_attn_indexer.py (+7/-1); vllm/platforms/rocm.py (+1/-0)
LABELS: rocm, ready, deepseek, verified
BODY: ## Purpose ⏎  ⏎ This PR enables DeepSeek-V4 checkpoints on ROCm gfx11/RDNA devices. ⏎  ⏎ It removes Python-side blockers in the ROCm sparse-indexer path and allows DeepSeek-V4 checkpoints mapped to `INCConfig` to pass ROCm platform validation: ⏎  ⏎ - Register ROCm sparse-indexer ops on gfx11. ⏎ - Let `SparseAttnIndexer.forward_hip()` delegate to the ROCm sparse-indexer custom op and use the op-level fallback instead of failing early on platform capabili …[truncated]

### L3-7f7a32cfec  (L3, 2026-08-12, sha 7f7a32cfec0f, PR #47808)
TITLE: [Spec Decode] DSpark confidence-scheduled verification (#47808)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: vllm/models/deepseek_v4/nvidia/flashmla.py (+19/-1); vllm/v1/attention/backend.py (+35/-10); vllm/v1/attention/backends/flashinfer.py (+6/-0); vllm/v1/attention/backends/mla/indexer.py (+71/-24); vllm/v1/attention/backends/mla/sparse_swa.py (+27/-1); vllm/v1/attention/selector.py (+15/-0); .buildkite/test_areas/lm_eval.yaml (+20/-0); docs/features/speculative_decoding/README.md (+2/-0); docs/features/speculative_decoding/adaptive_verification.md (+47/-0); tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-confidence-TP4.yaml (+25/-0); (+30 more)
LABELS: documentation, performance, speculative-decoding, ready, ci/build, v1, qwen, nvidia, mrv2
ISSUES: #47839 [RFC]: Packed Variable Length Speculative Decoding
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: Adaptively sizes the DSpark draft-verification budget from per-request confidence instead of always verifying every drafted token. Motivation: fixed-k speculation collapses at high concurrency — once the GPU saturates, verifying 7 drafts per request burns more compute than the accepted tokens return, dropping **below** non-speculative decoding (see table). ⏎  ⏎ ## Design ⏎  ⏎ - A Triton kernel ranks draft slots by survival probability (cumprod of per …[truncated]

### L3-b745d08de1  (L3, 2026-08-12, sha b745d08de14f, PR #51860)
TITLE: [ROCm][K3] Dequantize the fp8 decode query for MLA backends without quant-query support - TRITON_MLA (#51860)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/kimi_k3/nvidia/mla.py (+7/-5)
LABELS: rocm, ready, quantization, kimi, k3
BODY: ## Purpose ⏎ When enabling fp8 kv-cache dtype in DSpark speculative decoding, we got an assert error. THis PR is to put out a minimal fix to unblock this path. ⏎  ⏎  ⏎ On ROCm, the DSpark draft auto-selects TRITON_MLA, the only backend supporting its ⏎ non-causal multi-token blocks atm. TRITON_MLA dequantizes fp8 KV on load and ⏎ takes a bf16 query, so Kimi-K3's `_decode_concat_cache` assert on ⏎ `supports_quant_query_input` makes fp8 KV unusable with D …[truncated]

### L3-025d56a11e  (L3, 2026-08-12, sha 025d56a11e8f, PR #52035)
TITLE: [Build] Update DeepGEMM pin to deepseek-ai nv_dev tip (#52035)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: cmake/external_projects/deepgemm.cmake (+3/-3); tools/install_deepgemm.sh (+3/-3)
LABELS: ready, ci/build, deepseek
BODY: ## Purpose ⏎  ⏎ Update both DeepGEMM pins (`cmake/external_projects/deepgemm.cmake` and `tools/install_deepgemm.sh`, which are documented to stay in sync) from the `vllm-project/DeepGEMM` fork at `e21c821f` to upstream `deepseek-ai/DeepGEMM` at `8b1392b978f5a03c828dd1711090d7fb50958b8a`, the current tip of the `nv_dev` branch. ⏎  ⏎ The fork pin was kept because the plain `nv_dev` branch previously lacked SiTU support (see the removed TODO comment in the  …[truncated]

### L3-34735aceda  (L3, 2026-08-12, sha 34735aceda0b, PR #52028)
TITLE: [Bugfix] Pin DeepEP by its full commit hash (#52028)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tools/ep_kernels/install_python_libraries.sh (+3/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ `tools/ep_kernels/install_python_libraries.sh` pins DeepEP by a 10-character abbreviation: ⏎  ⏎ ```bash ⏎ DEEPEP_COMMIT_HASH=${DEEPEP_COMMIT_HASH:-"d4f41e4e93"} ⏎ ``` ⏎  ⏎ An abbreviated object ID is not a ref, so it cannot be fetched directly. GitHub serves any *complete* commit via `allowAnySHA1InWant`, but an abbreviation is never a valid want: ⏎  ⏎ ```console ⏎ $ git fetch --depth 1 origin d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb ⏎  * branch  d4f41e4 …[truncated]

### L3-23f360edaa  (L3, 2026-08-12, sha 23f360edaaa3, PR #52009)
TITLE: [CI Bug] Fix ci moe test (#52009)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/evals/gsm8k/configs/moe-refactor-dp-ep/Qwen3-30B-A3B-BF16-triton.yaml (+1/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ Fixes https://buildkite.com/vllm/ci/builds/83443#019ff2a1-641e-4d2b-bca9-9eda8a060573 ⏎  ⏎ There are two errors here: ⏎ 1. vLLM side, we name it triton test, but actually running the flashinfer path, this PR fixes the issue ⏎ 2. the root cause of flashinfer is a bug upstream with TRT-LLM BF16 MoE, we may wait for their fix, not related to this PR ⏎  ⏎ ## Test ⏎  ⏎ Covered in CI

### L3-903d2efe7e  (L3, 2026-08-12, sha 903d2efe7eb6, PR #51772)
TITLE: [Attention][MLA] Fuse Kimi-K3 chunked-context K/V packing (#51772)
SOURCES: path_core, subject_keyword, corpus:performance-pr-population
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+32/-6); vllm/v1/attention/backends/mla/prefill/aiter_flash_attn.py (+5/-0); vllm/v1/attention/backends/mla/prefill/base.py (+9/-5); vllm/v1/attention/backends/mla/prefill/flash_attn.py (+2/-0); vllm/v1/attention/backends/mla/prefill/flashinfer.py (+2/-0); vllm/v1/attention/backends/mla/prefill/tokenspeed_mla.py (+4/-1); vllm/v1/attention/backends/mla/prefill/trtllm_ragged.py (+9/-7); csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu (+275/-6); csrc/libtorch_stable/ops.h (+10/-0); csrc/libtorch_stable/torch_bindings.cpp (+11/-0); (+5 more)
LABELS: ready, nvidia, kimi, k3
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ The K3 MLA layer delegated chunked-context prefill to `impl._compute_prefill_context`, which per chunk casts `kv_nope` to fp8, casts `k_pe`, concatenates `[k_nope | k_pe]`, and re-quantizes a query the fused new-token epilogue had already quantized. ⏎  ⏎ This gives the layer its own context loop so that tail collapses into one kernel per chunk: `fused_kimi_k3_mla_kv_concat{,_quant_fp8}` reads the strided `kv_b_proj` output and the gathere …[truncated]

### L3-98f86b9c02  (L3, 2026-08-12, sha 98f86b9c0232, PR #50017)
TITLE: [ROCm] [bugfix] Chunked prefill paged decode masked load perf  (#50017)
SOURCES: path_core
ARTIFACT_HINTS: L3.triton.chunked_prefill_paged_decode
FILES: vllm/v1/attention/ops/chunked_prefill_paged_decode.py (+34/-20)
LABELS: bug, rocm, ready, v1
BODY: ## Purpose ⏎ Observed 10-15% latency performance regression for serving Qwen/Qwen3-30B-A3B-Thinking-2507; trace revealed that `kernel_paged_attention_2d` was the culprit, taking 1.33x as long on average on v0.25.0 vs v0.24.0. #47305 fixed correctness but introduced performance drop because of universally applied masking. ⏎  ⏎ Edited masking so that it is only enacted during last block, when token id can be greater than sequence length.  ⏎  ⏎ ## Test P …[truncated]

### L3-9035151d6c  (L3, 2026-08-12, sha 9035151d6c9f, PR #51255)
TITLE: [Model] Add native Dots3 NOTE multimodal support (#51255)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+28/-6); tests/kernels/attention/test_rocm_aiter_mla_causal_verify_mask.py (+8/-1); tests/models/registry.py (+11/-0); tests/models/test_registry.py (+5/-0); tests/tool_parsers/test_dots_tool_parser.py (+275/-0); tests/v1/attention/test_mla_backends.py (+3/-0); vllm/config/speculative.py (+16/-0); vllm/model_executor/models/registry.py (+5/-0); vllm/models/dots3_note/__init__.py (+8/-0); vllm/models/dots3_note/common/__init__.py (+3/-0); (+18 more)
LABELS: new-model, rocm, ready, tool-calling, verified
BODY: ## Motivation ⏎  ⏎ Add native vLLM support for the complete Dots3 NOTE model, including: ⏎  ⏎ - Text generation ⏎ - Image understanding ⏎ - Audio understanding ⏎ - Native video processing with interleaved visual and audio inputs ⏎ - FP8 MoE inference ⏎ - MTP speculative decoding ⏎ - OpenAI-compatible tool calling ⏎  ⏎ Dots3 NOTE uses one unified Hugging Face model type and architecture: ⏎  ⏎ - `model_type`: `dots3_note` ⏎ - `architectures`: `Dots3NoteForCausalL …[truncated]

### L3-399f97424e  (L3, 2026-08-13, sha 399f97424e61, PR #50874)
TITLE: [Bugfix][R3] Size monolithic routing replay buffer for DP (#50874)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/modular_kernel.py (+8/-1); vllm/model_executor/layers/fused_moe/routed_experts_capturer.py (+22/-4)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ Fix routing-replay capture for the FlashInfer monolithic MoE kernel under naive ⏎ data parallelism, including padded sequence-parallel shards when expert ⏎ parallelism is enabled. ⏎  ⏎ Two related assumptions fail in a TP2/DP2 deployment: ⏎  ⏎ 1. **Replay buffer capacity.** `max_num_tokens` is a per-rank scheduler limit, ⏎    while the naive dispatch path all-gathers rank-local batches before invoking ⏎    the monolithic kernel. This produced an 8192 …[truncated]

### L3-2d24355eb8  (L3, 2026-08-13, sha 2d24355eb87b, PR #52030)
TITLE: [Bugfix] Fix packed GDN decode launch for large batch-head grids (#52030)
SOURCES: path_core
ARTIFACT_HINTS: -
FILES: vllm/third_party/flash_linear_attention/ops/fused_recurrent.py (+10/-3); tests/kernels/test_fused_recurrent_packed_decode.py (+23/-0)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ Avoid a CUDA launch failure in packed GDN decode when `batch_size * num_value_heads` exceeds the maximum CUDA grid Y/Z dimension of 65,535. ⏎  ⏎ The existing launch is preserved for normal sizes. Only overflowing cases use a split `(value_tiles, value_heads, batch)` grid. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ - Verified the failing Qwen shape (`B=1024`, `HV=64`, `K=V=128`) launches successfully. ⏎ - Running `vllm serve mgoin/Qwen3.8-2.4 …[truncated]

### L3-50ba4bc6b2  (L3, 2026-08-13, sha 50ba4bc6b2ca, PR #49577)
TITLE: [Feature] Mask Replay (#49577)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/training/sampling_mask.md (+113/-0); rust/src/engine-core-client/src/protocol/output.rs (+3/-0); rust/src/engine-core-client/src/tests/client.rs (+14/-0); rust/src/engine-core-client/src/tests/python_compat.py (+25/-0); tests/entrypoints/scale_out/token_in_token_out/test_serving_tokens.py (+44/-0); tests/v1/core/test_async_scheduler.py (+1/-0); tests/v1/core/test_scheduler.py (+1/-0); tests/v1/test_outputs.py (+73/-0); vllm/config/model.py (+3/-0); vllm/config/vllm.py (+27/-0); (+14 more)
LABELS: documentation, frontend, ready, v1, mrv2, rust
BODY: ## Summary ⏎  ⏎ This PR adds experimental support for **sampling distribution replay**. ⏎  ⏎ A sampling mask represents the vocabulary support retained after top-k/top-p filtering. It is not an attention mask and does not affect causal attention or KV-cache behavior. ⏎  ⏎ When enabled, vLLM returns the sampling support for each generated token in a CSR-style representation: ⏎  ⏎ ```python ⏎ SamplingMask( ⏎     token_ids=[...], ⏎     offsets=[0, ..., len(tok …[truncated]

### L3-61826c1c6f  (L3, 2026-08-13, sha 61826c1c6fa3, PR #51624)
TITLE: [Hardware][Power] Unqualized MoE Backend for Power (VSX) (#51624)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/cpu/cpu_fused_moe.cpp (+12/-0); csrc/cpu/micro_gemm/cpu_micro_gemm_vsx.hpp (+445/-0); csrc/cpu/utils.hpp (+3/-1); tests/kernels/moe/test_cpu_fused_moe.py (+5/-5); vllm/model_executor/layers/fused_moe/experts/cpu_moe.py (+33/-0); vllm/model_executor/layers/fused_moe/oracle/unquantized.py (+2/-0)
LABELS: cpu
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: This PR adds PowerPC specific unquantized backend support for fused MoE using Power10 VSX MMA instructions. ⏎  ⏎ ## Purpose ⏎ Currently, grouped GEMM is not supported for Power architecture in vLLM. This PR introduces a Power/VSX specific unquantized CPU backend for Fused MoE.  ⏎ Key features include: ⏎ - Addition of `csrc/cpu/micro_gemm/cpu_micro_gemm_vsx.hpp` to support grouped GEMM on Power architectures. ⏎ - Use of Power10 MMA instructions for opti …[truncated]

### L3-b8baa31a28  (L3, 2026-08-13, sha b8baa31a2865, PR #49458)
TITLE: Hardware-agnostic model definition via HF transformer backend (1/N) (#49458)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/models/transformers/test_layer_registry.py (+177/-0); vllm/envs.py (+5/-0); vllm/model_executor/hw_agnostic/__init__.py (+2/-0); vllm/model_executor/hw_agnostic/custom_op.py (+318/-0); vllm/model_executor/hw_agnostic/layers/__init__.py (+2/-0); vllm/model_executor/hw_agnostic/layers/activation.py (+27/-0); vllm/model_executor/hw_agnostic/layers/layernorm.py (+90/-0); vllm/model_executor/models/transformers/fusers/glu.py (+2/-4); vllm/model_executor/models/transformers/fusers/rms_norm.py (+4/-1); vllm/model_executor/models/transformers/layers.py (+64/-0)
LABELS: ready, verified
BODY: ## Purpose ⏎  ⏎ This PR is an alternative approach to realize hardware-agnostic model definitions based on the HF transformer backend.  ⏎  ⏎ In particular, the idea is that the modeling code of tail models resides in HF transformers and can be executed in vLLM through the help of the `transformers` backend, i.e., `--model-impl transformers`. The way this backend currently works is by replacing / patching particular layers from HF transformers with na …[truncated]

### L3-9a276d6375  (L3, 2026-08-13, sha 9a276d63753a, PR #52003)
TITLE: [Mypy Fix] Mypy fix for "vllm/model_executor/models/[cC][dD]" (#52003)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: tools/pre_commit/mypy.py (+0/-3); vllm/model_executor/models/clip.py (+3/-0); vllm/model_executor/models/cohere2_moe.py (+2/-0); vllm/model_executor/models/cohere2_vision.py (+1/-1); vllm/model_executor/models/cohere_asr.py (+28/-13); vllm/model_executor/models/cohere_eagle.py (+3/-1); vllm/model_executor/models/colmodernvbert.py (+2/-0); vllm/model_executor/models/colpali.py (+4/-1); vllm/model_executor/models/commandr.py (+1/-0); vllm/model_executor/models/config.py (+11/-2); (+14 more)
LABELS: speculative-decoding, ready, multi-modality, deepseek
BODY: ## Purpose ⏎  ⏎ Mypy fix for "vllm/model_executor/models/[cC][dD]" ⏎  ⏎ ## Test ⏎  ⏎ ```bash ⏎ pre-commit run --hook-stage manual mypy-3.13 -a ⏎ Run mypy for Python 3.13.................................................Passed ⏎ ```

### L3-11c3fa4adc  (L3, 2026-08-13, sha 11c3fa4adc81, PR #52139)
TITLE: [Bugfix][ROCm][CI] Give the AITER MLA decode metadata stub its MLA dims (#52139)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_rocm_aiter_mla_decode_metadata.py (+13/-3)
LABELS: bug, rocm
BODY: # Purpose ⏎  ⏎ `tests/kernels/attention/test_rocm_aiter_mla_decode_metadata.py::test_persistent_decode_metadata_matches_fp8_golden` ⏎ fails on main with `AttributeError: 'types.SimpleNamespace' object has no ⏎ attribute 'q_lora_rank'`. Two jobs report it, the dedicated AITER MLA job and the sharded `kernels/attention` job, but it is the same test. ⏎  ⏎ "[Model] Add native Dots3 NOTE multimodal support" (#51255), changed ⏎ `MLACommonMetadataBuilder.__ini …[truncated]

### L3-2e2ffd104b  (L3, 2026-08-13, sha 2e2ffd104bd0, PR #52210)
TITLE: [CI Failure] Fix CUDA wheel build for the Kimi K3 fused MLA kernel (#52210)
SOURCES: subject_keyword
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu (+31/-37)
LABELS: bug, ready, ci/build, ci-failure, nvidia, kimi, k3
BODY: ## Purpose ⏎  ⏎ The release pipeline for building wheels has been failing since https://github.com/vllm-project/vllm/pull/51772 landed due to SM75 not supporting bf16 ⏎ Latest failing job on main https://buildkite.com/vllm/release-v2/builds/5155/list?sid=019ffc97-c2e1-4216-87f4-9c55baf6e6fb&tab=output ⏎  ⏎ Use `_typeConvert<scalar_t>::exists` to discard unsupported packed conversion paths during host and pre-Ampere compilation. Supported FP16/BF16 exe …[truncated]

### L3-b652dedd0c  (L3, 2026-08-13, sha b652dedd0c50, PR #52148)
TITLE: [Attention] Fix FlashInfer SM12x prefill with sinks (#52148)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+65/-23); tests/v1/attention/test_attention_backends.py (+1/-1)
LABELS: bug, ready, nvidia
BODY: ## Summary ⏎  ⏎ Use FlashInfer's sink-aware paged prefill wrapper on SM12x when attention sinks are enabled. The generic FA2 prefill path accepts a `sinks` argument but does not apply it, so #49718 can use XQA for decode while producing incorrect prefill output. ⏎  ⏎ The wrapper is specialized with the active dtypes, head dimensions, sliding window, and softmax scale. DCP, NVFP4, SM90/SM100, and sink-free paths are unchanged. ⏎  ⏎ This is not a duplica …[truncated]

### L3-373592ef57  (L3, 2026-08-13, sha 373592ef57d4, PR #52092)
TITLE: [CPU] Ship triton-cpu wheel and fix several hardcoded pin_memory=True (#52092)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/hardware_tests/cpu.yaml (+7/-22); docker/Dockerfile.cpu (+30/-15); vllm/model_executor/models/granite_speech.py (+3/-1); vllm/model_executor/models/qwen3_vl.py (+2/-1); vllm/v1/worker/cpu/shm.py (+1/-0); vllm/v1/worker/cpu_worker.py (+3/-0); vllm/v1/worker/gpu/mm/encoder_runner.py (+2/-1); vllm/v1/worker/gpu/structured_outputs.py (+2/-1)
LABELS: ci/build, qwen, cpu, mrv2
BODY: ## Summary ⏎ - Build and install a pre-built `triton-cpu` wheel in the CPU build/test images instead of `pip install`-ing it from source inside CI, unblocking the Triton topk-topp kernel to run as a normal (non-soft-fail) test. ⏎ - Move the topk-topp Triton kernel test out of the soft-fail `CPU-ModelRunnerV2 Tests` suite into `CPU-Kernel Tests`, and the linear-attention chunked-prefill correctness test into `CPU-Language Generation and Pooling Model  …[truncated]

### L3-b245d8e73f  (L3, 2026-08-13, sha b245d8e73f67, PR #52173)
TITLE: Apply logit softcapping in Transformers modelling backend (#52173)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/transformers/causal.py (+3/-2)
LABELS: ready
BODY: This is mainly used by Gemma models and only affects workloads where the exact logit values are important. Generation is unaffected because it does not reorder anything.

### L3-827a2af806  (L3, 2026-08-13, sha 827a2af806c4, PR #48666)
TITLE: [Kernel] Gemma-4 FA4 FP8 Kernel (#48666)
SOURCES: path_core, path_integration+keyword, subject_keyword, symbol_pickaxe, dependency_pin, corpus:confirmed-reverts(reverted), corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fork_build, L3.flash_attn.fa4_cutedsl, L3.flash_attn.fa_utils, L3.dispatch.abstract_interface
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1); vllm/model_executor/layers/attention/attention.py (+11/-3); vllm/platforms/interface.py (+3/-5); vllm/v1/attention/backend.py (+14/-0); vllm/v1/attention/backends/fa_utils.py (+3/-3); vllm/v1/attention/backends/flash_attn.py (+57/-2); vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py (+43/-12); vllm/vllm_flash_attn/flash_attn_interface.py (+20/-3)
LABELS: documentation, speculative-decoding, ready, ci/build, v1, mrv2, verified
DEEP_STUDY: deep-study: this PR was reverted by PR 52987 (confirmed_revert, reason=unstated) || deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ Gemma-4 uses 256-wide heads in `sliding_attention` and 512-wide heads in ⏎ `full_attention`. On SM90, the full-attention layers upgrade from FA3 to the FA4 ⏎ CuTeDSL kernel. This PR wires the FA4 FP8-KV-dequant path from ⏎ [vllm-project/flash-attention#164](https://github.com/vllm-project/flash-attention/pull/164) ⏎ into vLLM, allowing Gemma-4 to use FP8 KV cache across both FA3 sliding attention ⏎ and FA4 full attention. ⏎  ⏎ It also pres …[truncated]

### L3-3c79b1a8bf  (L3, 2026-08-13, sha 3c79b1a8bf71, PR #52016)
TITLE: [Kernel] Add B12X dense linear backends (#52016)
SOURCES: release_notes, corpus:production-kernel-provenance, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docs/features/quantization/b12x.md (+24/-0); setup.py (+1/-0); tests/kernels/quantization/test_block_fp8.py (+55/-0); tests/model_executor/kernels/test_b12x_mxfp4_linear.py (+199/-0); tests/model_executor/kernels/test_b12x_mxfp8_linear.py (+935/-0); tests/model_executor/kernels/test_b12x_nvfp4_linear.py (+246/-0); tests/model_executor/test_b12x_warmup.py (+100/-0); vllm/config/kernel.py (+2/-0); vllm/model_executor/kernels/linear/__init__.py (+32/-0); vllm/model_executor/kernels/linear/mxfp4/b12x.py (+170/-0); (+7 more)
LABELS: documentation, ready, ci/build
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ This PR integrates [B12X](https://github.com/local-inference-lab/b12x) dense linear kernels for NVIDIA SM120 and SM121 GPUs through the existing vLLM linear backend interfaces. B12X is an optional dependency installed with `vllm[b12x]` and pinned to `b12x==1.2.4`; it is a pure-Python CuTe DSL package and requires no additional vLLM build step. ⏎  ⏎ Supported linear paths are: ⏎  ⏎ - Per-tensor FP8. ⏎ - 128x128 block-scaled FP8. ⏎ - MXFP8. ⏎ - NVFP4 …[truncated]

### L3-bda4c3e8ee  (L3, 2026-08-14, sha bda4c3e8eefe, PR #51583)
TITLE: [CPU] Fold the MXFP4 block scale in 2 instructions instead of 4 (#51583)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/cpu/sgl-kernels/vec.h (+16/-11); tests/kernels/moe/test_cpu_quant_fused_moe.py (+119/-0)
LABELS: ready, cpu, verified
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ The AVX-512 MXFP4 unpack in `csrc/cpu/sgl-kernels/vec.h` applies the E8M0 block ⏎ scale as an integer add on the bf16 exponent field. It has to keep the two zero ⏎ codes at zero, because they have no exponent to shift, and that special case is ⏎ written as `and` + `cmpeq` + `add` + `blend` — four instructions per vector. ⏎  ⏎ `vptestmw` sets a lane's mask bit for exactly the lanes where `(x & 0x7FFF) != 0`, ⏎ which is the complement of the same p …[truncated]

### L3-57bd0ed441  (L3, 2026-08-14, sha 57bd0ed44109, PR #51704)
TITLE: [5/N][KV-Cache Layout Refactor] Backend-published KV packing via customize_spec (#51704)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.triton.v1_backend, L3.mla.common_v1, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/attention.py (+11/-26); vllm/model_executor/layers/attention/mla_attention.py (+20/-1); vllm/v1/attention/backend.py (+15/-1); vllm/v1/attention/backends/flashinfer.py (+17/-1); vllm/v1/attention/backends/mla/sparse_swa.py (+2/-0); vllm/v1/attention/backends/triton_attn.py (+16/-2); vllm/v1/attention/backends/turboquant_attn.py (+16/-1); tests/quantization/test_turboquant.py (+15/-6); tests/v1/core/test_kv_cache_utils.py (+10/-11); tests/v1/test_kv_cache_spec_registry.py (+0/-11); (+8 more)
LABELS: rocm, ready, nvidia, mrv2, kimi, k3
BODY: ## Purpose ⏎  ⏎ Part of the KV-cache layout standardization series (RFC #42082). **Stacked on #51612** — the diff shown includes it until that lands and this retargets `main`. ⏎  ⏎ Attention specs today carry quant-format sizing knowledge inline: `nvfp4` / per-token-head branches in the page-size properties, a `TQFullAttentionSpec` subclass, and fp8_ds_mla constants in MLA spec overrides. This PR makes specs plain data and moves each packed format to …[truncated]

### L3-1f7427bc0a  (L3, 2026-08-14, sha 1f7427bc0ad3, PR #52265)
TITLE: [UT][XPU] fix b12x UT (#52265)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+1/-0); tests/model_executor/kernels/test_b12x_mxfp8_linear.py (+23/-3)
LABELS: intel-gpu, ready, nvidia
BODY: **Intel CI failure**: https://buildkite.com/vllm/intel-ci/builds/8908/canvas?sid=019ffe6c-17b4-443f-bc9f-8bf51457f9e2&tab=output ⏎ **Root cause**: b12x.py does from vllm.utils.flashinfer import flashinfer_mxfp4_quantize, but the function was never registered in that module. The test's monkeypatch.setattr then fails with AttributeError because there's nothing to patch. ⏎  ⏎ **Fix**: Added flashinfer_mxfp4_quantize = _lazy_import_wrapper("flashinfer", …[truncated]

### L3-63a9a5010a  (L3, 2026-08-14, sha 63a9a5010a6d, PR #52164)
TITLE: [Attention][DSA] Take the native decode path for MTP=3 on SM90 (#52164)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+30/-11); tests/kernels/attention/test_deepgemm_attention.py (+96/-91); tests/v1/attention/test_indexer_native_next_n.py (+75/-0); vllm/utils/deep_gemm.py (+31/-5)
LABELS: ready, verified
ISSUES: #35878 [Feature]: Support DeepGEMM MTP3 NV Kernel
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: # [Attention][DSA] Take the native decode path for MTP=3 on SM90 ⏎  ⏎ ## Purpose ⏎  ⏎ Closes #35878. ⏎  ⏎ The DSA indexer flattens a spec-decode batch into one single-token row per query ⏎ whenever `next_n` falls outside `{1, 2}`, so with MTP=3 (`next_n = 4`) each ⏎ request's KV tile is read four times instead of once. DeepGEMM's `nv_dev` branch, ⏎ which vLLM already pins (`cmake/external_projects/deepgemm.cmake`), implements ⏎ `next_n = 4` on SM90 through …[truncated]

### L3-3e3ceb1961  (L3, 2026-08-14, sha 3e3ceb1961d4, PR #52369)
TITLE: [Perf] Avoid more GPU<->CPU syncs in multimodal encoders (#52369)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/third_party/flash_linear_attention/ops/index.py (+5/-2); vllm/model_executor/models/diffusion_gemma.py (+5/-2); vllm/model_executor/models/ernie45_vl.py (+9/-5); vllm/model_executor/models/glm4_1v.py (+18/-37); vllm/model_executor/models/interns1.py (+2/-2); vllm/model_executor/models/keye.py (+25/-17); vllm/model_executor/models/paddleocr_vl.py (+11/-5); vllm/model_executor/models/phi4mm.py (+1/-1); vllm/model_executor/models/ultravox.py (+5/-1)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: Host data staged across blocking, now non-blocking or via `async_tensor_h2d`: ⏎  ⏎ - keye: position ids / cu_seqlens / sample indices on both the image and video paths, and the same pattern in paddleocr_vl, which carries a copy of that encoder. ⏎ - ernie45_vl: the rotary `pos_ids` gather (a device tensor was being indexed with a CPU one), the encoder `cu_seqlens`, and the two numpy-built slice-index tensors in the resampler. ⏎ - flash_linear_attentio …[truncated]

### L3-9b0ab5dd53  (L3, 2026-08-14, sha 9b0ab5dd5358, PR #51216)
TITLE: [ROCm][AMD] Enable preshuffled sparse indexing for 16-token blocks (#51216)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/backends/mla/indexer.py (+1/-1); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+1/-1); tests/v1/worker/test_gpu_model_runner.py (+14/-0)
LABELS: rocm, ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Allow ROCm sparse MLA indexer backends to use KV-cache block sizes aligned to 16 tokens. Previously, `--block-size 16` selected kernel block size 1, disabling AITER’s preshuffled FP8 paged-MQA path. The preshuffled AITER kernel was more performant than the non-shuffled variant so we were leaving some performance on the table. This PR fixes that. ⏎  ⏎ AI assistance was used to prepare this change. ⏎  ⏎ ## Test Plan ⏎  ⏎ ```bash ⏎ pytest tes …[truncated]

### L3-e078a2238a  (L3, 2026-08-14, sha e078a2238a1e, PR #52241)
TITLE: [Bugfix] Widen flashinfer.comm import guard so a failed import doesn't abort engine startup (#52241)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+2/-2)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎  ⏎ The optional `flashinfer.comm` import in `allreduce_rms_fusion.py` is guarded by ⏎ `except ImportError`, so any other exception during import aborts `EngineCore` startup. ⏎ Hit with `flashinfer-python==0.6.16.post3` on Python 3.11, which raises `TypeError` at ⏎ import time — killing startup on a `tp_size=1` run with `fuse_allreduce_rms: False`. ⏎  ⏎ Widen the guard to `except Exception` and log a warning, so a broken flashinfer disables …[truncated]

### L3-615d4cfade  (L3, 2026-08-14, sha 615d4cfadeb3, PR #43107)
TITLE: [Core] Check for GPU<->CPU syncs during CI (#43107)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/third_party/flash_linear_attention/ops/index.py (+5/-6); vllm/v1/attention/backends/flashinfer.py (+10/-7); vllm/v1/attention/ops/vit_attn_wrappers.py (+20/-4); .buildkite/test_areas/plugins.yaml (+6/-0); docker/Dockerfile (+3/-0); docker/Dockerfile.rocm (+3/-0); tests/models/multimodal/generation/test_mm_prefix_lm.py (+4/-1); tests/v1/e2e/general/test_mamba_prefix_cache.py (+28/-17); vllm/distributed/eplb/eplb_communicator.py (+6/-4); vllm/distributed/eplb/eplb_state.py (+21/-18); (+34 more)
LABELS: rocm, speculative-decoding, ready, ci/build, v1, multi-modality, qwen, kv-connector, nvidia, ready-run-all-tests
BODY: vLLM now uses asynchronous scheduling by default and in the majority of cases. Performance relies on the absence of any gpu<->cpu synchronizations on the main cuda stream, but such syncs can be opaque and it is easy for them to creep in accidentally. ⏎  ⏎ This change adds a `VLLM_GPU_SYNC_CHECK` env var which enables `torch.cuda.set_sync_debug_mode` for the model forward pass and sampler, so that we can easily check for such syncs. ⏎  ⏎ A new `gpu_sy …[truncated]

### L3-97388c44f9  (L3, 2026-08-15, sha 97388c44f9c6, PR #51538)
TITLE: [Bugfix] Make DSV4 sparse MLA work end-to-end for plain decode, MTP, and DSpark (#51538)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+16/-0); vllm/v1/attention/backends/mla/compressor_utils.py (+12/-0); vllm/v1/attention/backends/mla/indexer.py (+9/-3); vllm/v1/attention/backends/mla/sparse_swa.py (+16/-4); vllm/v1/worker/gpu/model_runner.py (+48/-40); vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (+33/-9); vllm/v1/worker/gpu_worker.py (+16/-2); vllm/v1/worker/workspace.py (+49/-14); csrc/libtorch_stable/cooperative_topk.cuh (+4/-1); csrc/libtorch_stable/persistent_topk.cuh (+20/-1); (+10 more)
LABELS: bug, speculative-decoding, ready, nvidia, mrv2, verified
BODY: ## Purpose ⏎  ⏎ DeepSeek-V4-Flash-0731 could not run reliably through the SM120 sparse MLA backend. This fixes the seven defects that blocked it across all three decode modes -- plain decode, MTP, and DSpark -- verified end-to-end on 8xRTX PRO 6000 Blackwell across in-flight batching and prefill/decode disaggregation. ⏎  ⏎ Commits 1-5 unblock DSpark. Commits 6-7 fix a hang that is **not** DSpark-specific: it strands any MTP (`next_n > 1`) server on this  …[truncated]

### L3-acb0f1dcdb  (L3, 2026-08-15, sha acb0f1dcdb66, PR #52288)
TITLE: [Bugfix][Spec Decode] DSpark: inherit the target's attention backend when the speculative config names none (#52288)
SOURCES: path_integration+keyword, subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/spec_decode/dspark/utils.py (+7/-1)
LABELS: bug, mrv2
BODY: ## Purpose ⏎  ⏎ `load_dspark_model` passes `backend=speculative_config.attention_backend` when ⏎ building the draft's config. That field is `None` unless the user names a backend ⏎ inside `--speculative-config`, and `None` re-runs backend auto-selection rather ⏎ than meaning "same as target" — so an explicitly pinned target backend is silently ⏎ discarded and the draft can pick a different attention class. ⏎  ⏎ On DeepSeek V4 with `--attention-config.backend FLA …[truncated]

### L3-edd4c8176c  (L3, 2026-08-15, sha edd4c8176cfd, PR #51318)
TITLE: [Bugfix][DSv4] Revert adaptive C128A metadata packing (#51318)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_flashmla_sparse.py (+0/-37); vllm/models/deepseek_v4/sparse_mla.py (+7/-19)
LABELS: bug, ready, verified
ISSUES: #52448 [Bug]: DeepSeek-V4-Flash think-until-cap / empty turn under concurrent DSpark + breakable CUDA graphs
DEEP_STUDY: deep-study revert record: confirmed_revert of PR(s) 50004 reason=correctness_or_accuracy
BODY: ## Summary ⏎  ⏎ Revert the adaptive C128A top-k width and packed persistent-buffer views introduced by #50004. ⏎  ⏎ The C128A metadata builder runs before FULL CUDA graph replay, but the sparse decode consumer is captured and reuses its capture-time row layout. With #50004, runtime metadata writes packed rows using a batch-dependent `active_topk_width`, while the captured consumer retained the capture-time row stride. Row 0 still started at offset 0, but …[truncated]

### L3-8efa13b700  (L3, 2026-08-15, sha 8efa13b700f1, PR #52401)
TITLE: [Bugfix] Pick the DeepSeek V4 eager cudagraph region per model runner (#52401)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/test_config.py (+24/-37); vllm/config/vllm.py (+7/-25); vllm/models/deepseek_v4/attention.py (+65/-4)
LABELS: bug, ready, deepseek, nvidia
BODY: #51430 narrowed the DeepSeek V4 eager cudagraph region, which corrupts MRV1 output, and #51768 responded by defaulting the model to MRV2 and rejecting MRV1 + PIECEWISE. That default costs ROCm, where MRV1 is still the faster runner for this model. ⏎  ⏎ Choose the region from the runner instead: MRV1 wraps the whole attention body in `_prepare_and_attn_eager`, restoring the pre-#51430 region it needs, while MRV2 keeps the narrow region and its short …[truncated]

### L3-84530eb235  (L3, 2026-08-16, sha 84530eb235dc, PR #52441)
TITLE: [Bugfix][Multimodal] Keep Gemma 4 video frame counts on CPU (#52441)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_vit_cudagraph.py (+6/-0); vllm/model_executor/models/gemma4_mm.py (+1/-1)
LABELS: bug, ready, multi-modality, nvidia
BODY: ## Purpose ⏎ FIX https://buildkite.com/vllm/ci/builds/84014#01a0041b-4750-418e-aab2-15f6f61ebfaf ⏎  ⏎ ``` ⏎  ⏎ (EngineCore pid=18850) ERROR 08-15 07:10:27 [dump_input.py:72] Dumping input data for V1 LLM engine (v0.27.2rc1.dev119+ga67d4e226) with config: model='google/gemma-4-E2B-it', speculative_config=None, tokenizer='google/gemma-4-E2B-it', skip_tokenizer_init=False, tokenizer_mode=auto, revision=main, tokenizer_revision=main, trust_remote_code=Tru …[truncated]

### L3-ef43e3101b  (L3, 2026-08-16, sha ef43e3101b8f, PR #52212)
TITLE: [ROCm][DSV4][Perf] Optimize Triton sparse-MLA decode on gfx950 (#52212)
SOURCES: path_core, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+784/-62); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+458/-23); tests/kernels/test_compressor_kv_cache.py (+239/-4); vllm/models/deepseek_v4/amd/rocm.py (+48/-3); vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py (+21/-2)
LABELS: performance, rocm, ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Optimize the existing in-tree Triton DeepSeek-V4 sparse-MLA decode path for ⏎ gfx950/MI355X. ⏎  ⏎ The change adds a dedicated gfx950 partial kernel, workload-aware split ⏎ selection, graph-safe adaptive work selection, and gfx950-only cp-gather ⏎ de-specialization. It adds no AITER/Gluon runtime dependency or implementation ⏎ switch; corrected AITER/Gluon is used only as a performance comparator. ⏎  ⏎ The PR remains draft while refreshed CI runs and  …[truncated]

### L3-fe1c317157  (L3, 2026-08-16, sha fe1c317157d4, PR #52356)
TITLE: [Bugfix][ROCm] Skip FP8 MLA prefill PS-metadata build for chunked-context batches (#52356)
SOURCES: path_core, subject_keyword, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter
FILES: vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+5/-1)
LABELS: bug, rocm, ready
BODY: ## Purpose ⏎  ⏎ FP8 MLA PS ASM kernel is run only on non-chunked (causal) prefills. However, its persistent metadata is always built, even during chunked prefills (unnecessarily). ⏎  ⏎ This PR fixes it so metadata is only computed when it's actually needed. ⏎  ⏎ ## Test Plan ⏎  ⏎ Unit test for the exact guarded call site: ⏎  ⏎ ```bash ⏎ python3 -m pytest -v tests/kernels/attention/test_rocm_aiter_mla_fp8_prefill.py ⏎ ``` ⏎  ⏎ Formatting/lint: ⏎  ⏎ ```bash ⏎ ruff  …[truncated]

### L3-fdab2b10bc  (L3, 2026-08-16, sha fdab2b10bcac, PR #52425)
TITLE: [ModelRunner v2] Support Transformers pooling model  (#52425)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/transformers/test_backend.py (+10/-1); vllm/v1/worker/gpu/model_states/__init__.py (+8/-4)
LABELS: ready, mrv2
BODY: ## Purpose ⏎  ⏎ Support migrating Transformers-backend pooling models to MRV2.  This is a prerequisite for landing [#48290](https://github.com/vllm-project/vllm/pull/48290). ⏎  ⏎ ## Test Plan ⏎  ⏎ ```bash ⏎ CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \ ⏎   tests/models/transformers/test_backend.py::test_pooling -v ⏎ ``` ⏎  ⏎ Tested on NVIDIA H200 with MRV2 and default FlashAttention/FA4: ⏎  ⏎ - `TransformersEmbeddingModel`: passed against the Hugging Fa …[truncated]

### L3-6914d60b1e  (L3, 2026-08-16, sha 6914d60b1e1a, PR #49544)
TITLE: [ROCm][Perf] gfx942: use FlyDSL fp8 MQA logits kernel (ROCm/aiter#3913) (#49544)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+2/-9)
LABELS: rocm, ready, v1
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: On gfx942, replace the vendored Triton `fp8_mqa_logits` with `aiter.ops.flydsl.flydsl_fp8_mqa_logits` from ROCm/aiter#3913. Drop-in replacement (identical args/semantics), gated behind `_ON_GFX942`. ⏎  ⏎ gfx950 and other paths are untouched. ⏎  ⏎ ## Results: GLM-5.2-FP8, 8× MI325X TP8 ⏎  ⏎ ### ISL=128K, OSL=1K, Conc=8  ⏎  ⏎ | Metric | Baseline (Triton) | FlyDSL | Delta | ⏎ |--------|-------------------|--------|-------| ⏎ | Median TTFT (ms) | 42,839 | 21,8 …[truncated]

### L3-1f0e0bf612  (L3, 2026-08-16, sha 1f0e0bf61210, PR #52050)
TITLE: [Bugfix][Attention] Temporarily disable FA4 head-dim 256 (#52050)
SOURCES: path_core, subject_keyword, symbol_pickaxe, body_keyword
ARTIFACT_HINTS: L3.flash_attn.v1_backend, L3.flash_attn.fa_utils
FILES: vllm/v1/attention/backends/fa_utils.py (+10/-13); vllm/v1/attention/backends/flash_attn.py (+0/-1); tests/models/multimodal/pooling/test_colpali.py (+5/-0)
LABELS: bug, ready, multi-modality
BODY: ## Purpose ⏎  ⏎ PR #42669 enabled FA4 head-dim 256 on Blackwell, but the specialized SM100 2-CTA kernel still rejects `seqused_q/k`, which vLLM decoder attention supplies. Temporarily resolve FA2 for head-dim 256 on Blackwell until upstream adds the required support. FA4 remains enabled for head-dim 128 and the supported MLA 192/128 case. ColPali under MRV2 exposed the failure (#48290).  ⏎  ⏎ ## Reproducer ⏎  ⏎ ```bash ⏎ CUDA_VISIBLE_DEVICES=0 .venv/bin …[truncated]

### L3-a18c9b56ff  (L3, 2026-08-16, sha a18c9b56ff9f, PR #52458)
TITLE: [Kimi-K3][Perf] Update FlashKDA for automatic K2 V-split (#52458)
SOURCES: dependency_pin
ARTIFACT_HINTS: -
FILES: cmake/external_projects/flashkda.cmake (+1/-1)
LABELS: ci/build, kimi, k3
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Update the pinned FlashKDA revision from `b5d1101` to `053de1b` in one ⏎ CMake line. ⏎  ⏎ The new revision: ⏎  ⏎ - automatically splits K2's value dimension when ⏎   `2 * local_heads * sequences <= SM count`, increasing CTA-level parallelism ⏎   for low-parallelism workloads; ⏎ - keeps the existing K2 path for shapes rejected by the heuristic; and ⏎ - includes the upstream TMA proxy-fence fixes. ⏎  ⏎ There is no vLLM API or model-code change. I searched op …[truncated]

### L3-7ea4b40954  (L3, 2026-08-16, sha 7ea4b4095408, PR #52502)
TITLE: [Hardware][NVIDIA] Add GB10 fused-MoE fp8 tuning configs (E=256, E=512) (#52502)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json (+147/-0); vllm/model_executor/layers/fused_moe/configs/E=512,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json (+147/-0)
LABELS: ready, nvidia, verified
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎  ⏎ Add tuned fused-MoE Triton configs for NVIDIA GB10 (DGX Spark): `E=256,N=512` and `E=512,N=512`, `fp8_w8a8`, `block_shape=[128,128]` (DeepSeek-family MoE layer shapes). Without device-specific configs, GB10 falls back to heuristic defaults. ⏎  ⏎ Not a duplicate: existing open GB10 config PRs cover different shapes — #52192 is `E=512,N=2688` (Nemotron-3-Super) and #45949 is a Qwen3-Coder-Next shape. No open or merged config covers `device_ …[truncated]

### L3-836aac92ff  (L3, 2026-08-16, sha 836aac92ffda, PR #52084)
TITLE: [Perf][DSV4] Optimize sparse top-k metadata kernels for higher prefill throughput (#52084)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/common/ops/cache_utils.py (+1/-1)
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Optimize sparse top-k metadata kernels for higher prefill throughput ⏎  ⏎ ## Test ⏎ ``` ⏎ vllm bench serve \ ⏎     --backend vllm \ ⏎     --base-url http://localhost:8000 \ ⏎     --model deepseek-ai/DeepSeek-V4-Flash-0731 \ ⏎     --dataset-name random \ ⏎     --random-input-len 1024 \ ⏎     --random-output-len 64 \ ⏎     --num-prompts 128 \ ⏎     --num-warmups 8 \ ⏎     --request-rate inf \ ⏎     --ignore-eos \ ⏎     --temperature 0 \ ⏎     --seed "$ …[truncated]

### L3-311b3513af  (L3, 2026-08-16, sha 311b3513af33, PR #52565)
TITLE: [ROCm][CI] Avoid forcing FlashAttention in the ColPali pooling test (#52565)
SOURCES: subject_keyword, body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/pooling/test_colpali.py (+3/-1)
LABELS: rocm, multi-modality
BODY: PR #52050 added a CUDA-specific forced `FLASH_ATTN` configuration to a cross-platform ColPali test, which then failed on both [MI300](https://buildkite.com/vllm/amd-ci/builds/12112/list?jid=01a00c92-9c7e-4a47-a086-8b869348cac4&tab=output) and [MI355](https://buildkite.com/vllm/amd-ci/builds/12112/list?jid=01a00c92-9ca9-4012-8b7d-a9860d54a4d8&tab=output) because ROCm intentionally does not report a FlashAttention version. This keeps the intended C …[truncated]

### L3-83f591d7f6  (L3, 2026-08-16, sha 83f591d7f694, PR #51967)
TITLE: [Perf][DSV4] Optimize global top-k index kernel with compile-time constants (#51967)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/common/ops/cache_utils.py (+5/-5)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ Optimize global top-k index kernel with compile-time constants ⏎ ## Test Plan ⏎ ``` ⏎ vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 --trust-remote-code --kv-cache-dtype fp8 --block-size 256 --enable-expert-parallel --tensor-parallel-size 8 --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 --no-enable-prefix-caching --max-num-batched-tokens 16384 ⏎ ``` ⏎ ``` ⏎ vllm bench serv …[truncated]

### L3-6664d397bf  (L3, 2026-08-17, sha 6664d397bf09, PR #52329)
TITLE: [Performance][MRV2] Cache logits-processing request state (#52329)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_sampler_flags.py (+90/-0); tests/v1/worker/test_gpu_thinking_budget.py (+39/-0); vllm/v1/worker/gpu/sample/sampler.py (+18/-19)
LABELS: ready, mrv2
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Summary ⏎  ⏎ `Sampler._requires_logits_processing()` runs on every sampling step and currently scans multiple per-request NumPy/Python states. This change computes the combined predicate in `add_request()` and stores one boolean per request slot, so the hot-path gate only scans the active rows of one compact array. ⏎  ⏎ The cache is overwritten on slot reuse, keeps active-subset filtering through the existing `idx_mapping_np`, and does not change  …[truncated]

### L3-a02cfccbc6  (L3, 2026-08-17, sha a02cfccbc618, PR #50729)
TITLE: [Bugfix][Mamba] Fix overlapping state copy race (#50729)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_mamba_utils.py (+152/-37); vllm/v1/worker/mamba_utils.py (+76/-26)
LABELS: bug, ready
BODY: [PR #30877](https://github.com/vllm-project/vllm/pull/30877) introduced generic Mamba state copies, and [PR #40172](https://github.com/vllm-project/vllm/pull/40172) added the fused GPU copy. A speculative-decode convolution-state shift can copy within the same physical block with overlapping source and destination ranges. Parallel memcpy-style loads/stores do not provide memmove ordering, which explains the intermittent [AMD CI failure](https://b …[truncated]

### L3-0ad04cff1b  (L3, 2026-08-17, sha 0ad04cff1b10, PR #52256)
TITLE: [ROCm][CI] Enable ViT CUDA graph tests on AMD gfx950 GPUs (#52256)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+5/-2); docs/design/cuda_graphs_multimodal.md (+54/-24); tests/models/multimodal/generation/test_vit_cudagraph.py (+6/-2); tests/v1/cudagraph/test_encoder_cudagraph.py (+6/-2)
LABELS: documentation, rocm, ci/build, multi-modality, nvidia
BODY: ## Purpose ⏎  ⏎ Enable ViT encoder CUDA graph (encoder CG) test coverage on AMD MI350X/MI355X (gfx950) and document the tested hardware in the design doc. ⏎  ⏎ - **Tests**: Change the skip condition in `tests/v1/cudagraph/test_encoder_cudagraph.py` and `tests/models/multimodal/generation/test_vit_cudagraph.py` from `current_platform.is_cuda()` to `is_cuda_alike()` so the tests run on ROCm. ⏎ - **CI** (`.buildkite/test-amd.yaml`): add two `optional: tr …[truncated]

### L3-49905ad94d  (L3, 2026-08-17, sha 49905ad94dfc, PR #50174)
TITLE: [3/N][Feat][Perf] Add new warmup infrastructure for JITs. Add provider registry and orchestration for JIT warmup (#50174)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/contributing/README.md (+1/-0); docs/contributing/jit_kernel_warmup.md (+308/-0); tests/model_executor/test_jit_warmup.py (+230/-1); tests/v1/worker/test_jit_warmup_migration.py (+52/-0); vllm/model_executor/warmup/jit_warmup.py (+211/-18); vllm/model_executor/warmup/jit_warmup_triton_helper.py (+38/-0); vllm/model_executor/warmup/kernel_warmup.py (+17/-6); vllm/model_executor/warmup/v1_block_table_warmup.py (+0/-29); vllm/v1/worker/block_table.py (+154/-67); vllm/v1/worker/cpu_model_runner.py (+1/-1); (+2 more)
LABELS: documentation, ready, v1, cpu, mrv2
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Description ⏎  ⏎ This PR extends the shared JIT warmup infrastructure with provider registration and centralized orchestration. It builds on #49315 and the contract described in #47456. ⏎  ⏎ For more details, see parent (draft) PR: https://github.com/vllm-project/vllm/pull/49627 and tracking list issue https://github.com/vllm-project/vllm/issues/49349 ⏎  ⏎ ``` ⏎ JIT kernel warmup (5 compile keys): 100%|██████████████████████████████████| 1/1 [00:00<0 …[truncated]

### L3-967e104fad  (L3, 2026-08-17, sha 967e104fad12, PR #52550)
TITLE: [Config] Unify indexer cache dtype under attention_config.indexer_kv_dtype (#52550)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/indexer.py (+24/-13); tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-confidence-TP4.yaml (+1/-1); tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml (+1/-1); vllm/config/attention.py (+34/-7); vllm/models/deepseek_v4/attention.py (+2/-1); vllm/models/minimax_m3/nvidia/model.py (+3/-1)
LABELS: ready
BODY: ## Purpose ⏎  ⏎ The sparse-attention indexer picked its K-cache dtype through two unrelated knobs: ⏎  ⏎ | flag | type | read by | ⏎ | --- | --- | --- | ⏎ | `attention_config.use_fp4_indexer_cache` | `bool` | DeepSeek V3.2/V4 only | ⏎ | `attention_config.indexer_kv_dtype` | enum | MiniMax M3 only | ⏎  ⏎ Because each model read only one of them, passing both was accepted and quietly ⏎ resolved in favor of the bool on the DeepSeek path. A config asking for an fp8 ⏎ index …[truncated]

### L3-3fc2893909  (L3, 2026-08-17, sha 3fc28939099a, PR #52385)
TITLE: [Bugfix] Account for local DP workers in startup thread allocation (#52385)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/distributed/test_multiproc_executor.py (+47/-0); vllm/v1/executor/multiproc_executor.py (+4/-1)
LABELS: bug, ready
ISSUES: #52330 [Bug]: Data-parallel startup ignores DP in startup_omp_num_threads, causing CPU thread oversubscription and 15x slower weight loading (engine-ready timeout)
BODY: ## Purpose ⏎  ⏎ Fixes #52330. ⏎  ⏎ `MultiprocExecutor` sized each worker's startup thread pool using only the ⏎ workers within one engine. With multiple local data-parallel engines, every ⏎ engine therefore claimed the node's full CPU budget, oversubscribing CPU ⏎ threads and severely slowing weight loading. ⏎  ⏎ This change: ⏎  ⏎ - includes `data_parallel_size_local` when counting workers that share the ⏎   node's startup CPU budget; ⏎ - preserves the existing behavior f …[truncated]

### L3-1d3a8b9e22  (L3, 2026-08-17, sha 1d3a8b9e220f, PR #48998)
TITLE: [ROCm][Bugfix] Fix Triton W4A16 bug in determining if transpose is required for GPTQ/AutoGPTQ  (#48998)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/quantization/test_triton_w4a16.py (+2/-0); vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py (+37/-25)
LABELS: bug, rocm, ready
DEEP_STUDY: deep-study correctness case vllm:1d3a8b9e22: class=shape_alignment_edge; symptom=wrong_output_or_accuracy; introducing=#47770
BODY: ## Purpose ⏎ this commit #47770 (for issue #47159) introduced a shape-based method to determine if qzeros need transposing, but the shape check is ambiguous when the two candidate shapes are identical. (cyankiwi-MiniMax-M3-AWQ-INT4 with tp=8 as an example). This PR is to fix this issue: ⏎ 1. Use metadata instead of shape-based method; ⏎ 2. in case output_dim is not set, fall back to shape-based method; ⏎ 3. if 2 candidate shapes are identical, set tr …[truncated]

### L3-7075ddac28  (L3, 2026-08-17, sha 7075ddac28c2, PR #52197)
TITLE: Support DSpark configs with `architectures=DSparkDraftModel` + `model_type=qwen3` (#52197)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/config/speculative.py (+14/-1)
LABELS: speculative-decoding, ready, qwen
BODY: ## Purpose ⏎  ⏎ Generic Qwen `DSparkDraftModel` is now normalized to `Qwen3DSparkModel` so models like https://huggingface.co/RadixArk/Qwen3.8-2.4T-A95B-DSpark can now run on vLLM ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ Tested gsm8k with  ⏎ ``` ⏎ vllm serve mgoin/Qwen3.8-2.4T-A95B-NVFP4-pruned75 -tp=4 --spec-model RadixArk/Qwen3.8-2.4T-A95B-DSpark --spec-method dspark --spec-tokens 7 ⏎ ... ⏎ (APIServer pid=2078258) INFO 08-13 17:48:18 [metrics.py:120] Spe …[truncated]

### L3-d1e3eee6fb  (L3, 2026-08-17, sha d1e3eee6fb8e, PR #52188)
TITLE: [Spec decode] Support Kimi-K3 DCP with DSpark (#52188)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.common_v1, L3.mla.flashinfer, L3.dispatch.selector, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/mla_attention.py (+29/-0); vllm/v1/attention/backend.py (+8/-0); vllm/v1/attention/backends/mla/flashinfer_mla.py (+115/-7); vllm/v1/attention/backends/mla/tokenspeed_mla.py (+1/-0); vllm/v1/attention/selector.py (+4/-1); tests/transformers_utils/test_dspark_mla_config.py (+0/-23); tests/v1/attention/test_flashinfer_mla_dcp.py (+1/-0); tests/v1/attention/test_mla_backends.py (+77/-0); tests/v1/spec_decode/test_dflash_prepare_inputs.py (+25/-1); vllm/config/speculative.py (+0/-10); (+5 more)
LABELS: speculative-decoding, ready, nvidia, mrv2, kimi, k3
BODY: ## Purpose ⏎ This PR adds support for running Kimi-K3 decode context parallel with DSpark with FlashinferMLA and Tokenspeed as target causal attention backend and Tokenspeed as the draft non-causal backend. ⏎  ⏎ ## Test Plan ⏎ Kimi K3 GSM8k with the different backend combination. ⏎  ⏎ ## Test Result ⏎ Default (no backend specified): ⏎ ``` ⏎ vllm serve moonshotai/Kimi-K3 \ ⏎   --tensor-parallel-size 8 \ ⏎   -dcp 8 \ ⏎   --load-format fastsafetensors \ ⏎   --no …[truncated]
