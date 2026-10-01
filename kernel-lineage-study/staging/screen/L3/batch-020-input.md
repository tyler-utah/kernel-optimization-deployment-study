### L3-2cf82bcdd1  (L3, 2026-08-31, sha 2cf82bcdd17f, PR #50005)
TITLE: [Bugfix][DCP] Fix NVIDIA DeepSeek-V3.2 / GLM-5.2 fused attention (#50005)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_fused_deepseek_v32_norm_rope.py (+57/-0); vllm/models/deepseek_v32/attention.py (+29/-13); vllm/models/deepseek_v32/common/kernels.py (+16/-9)
LABELS: bug, ready, deepseek, nvidia, glm
ISSUES: #50095 [Bug][DCP] NVIDIA DeepSeek-V3.2 / GLM-5.2 fused attention bypasses DCP handling
BODY: # [Bugfix][DCP] Fix NVIDIA DeepSeek-V3.2 / GLM-5.2 fused attention ⏎  ⏎ Fixes #50095. ⏎  ⏎ ## Purpose ⏎  ⏎ Fix two DCP correctness defects in the fused DeepSeek-V3.2 / GLM-5.2 ⏎ attention path: ⏎  ⏎ - The fused norm/RoPE kernel skipped query RMSNorm when a DCP rank had no ⏎   owner-local KV-cache slot. KV ownership does not remove that rank's query ⏎   contribution. ⏎ - Pure DCP sent only rank-local query heads into attention over owner-local KV ⏎   and did not combine th …[truncated]

### L3-5707355209  (L3, 2026-08-31, sha 570735520942, PR #54465)
TITLE: [Bugfix][MLA] Fix BLHNC addressing for FlashInfer sparse MLA (#54465)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.flashinfer_sparse
FILES: vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py (+7/-0); vllm/v1/attention/backends/mla/sparse_utils.py (+3/-1); tests/v1/attention/test_indexer_dcp_localize.py (+10/-2); vllm/models/deepseek_v32/common/kernels.py (+5/-1)
LABELS: bug, ready, deepseek, nvidia
BODY: ## Purpose ⏎  ⏎ Fix sparse MLA cache addressing when `FLASHINFER_MLA_SPARSE` uses a block-outermost KV-cache layout such as BLHNC. ⏎  ⏎ BLHNC places other layers' pages between consecutive blocks of a given layer. Two paths incorrectly assumed that a layer's physical block stride was exactly its logical block size: ⏎  ⏎ - FlashInfer sparse index conversion multiplied the physical block ID by `block_size`, causing decode to read another layer's page. ⏎ - …[truncated]

### L3-44fe2a392b  (L3, 2026-08-31, sha 44fe2a392b71, PR #53921)
TITLE: [CPU] add CPU support for Voxtral (#53921)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/whisper_causal.py (+5/-1)
LABELS: ready, verified, mistral
BODY: ## Summary ⏎ co-developed with @NickCao. ⏎  ⏎ - Adds `CPUAttentionBackend` to the supported backends list in `create_whisper_attention_backend_with_block_pooling`  ⏎  ⏎ ``` ⏎ vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --gpu-memory-utilization 0.5 --max-model-len 2048 ⏎ ``` ⏎  ⏎ with this test `python examples/speech_to_text/realtime/openai_realtime_client.py ` ⏎ ``` ⏎ No audio path provided, using default: /root/.cache/vllm/assets/vllm_public_assets/ …[truncated]

### L3-e126687a9a  (L3, 2026-08-31, sha e126687a9a82, PR #53896)
TITLE: [Model] Support Qwen3.8-Flash-Next (#53896)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/lm_eval.yaml (+36/-0); .buildkite/test_areas/models_basic.yaml (+37/-0); .buildkite/test_areas/spec_decode.yaml (+1/-0); csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu (+26/-10); csrc/libtorch_stable/ops.h (+1/-1); csrc/libtorch_stable/torch_bindings.cpp (+2/-1); tests/config/test_config_utils.py (+19/-0); tests/config/test_speculative_draft_hf_overrides.py (+52/-1); tests/distributed/test_custom_all_reduce.py (+23/-0); tests/evals/qwen4_exp/README.md (+14/-0); (+114 more)
LABELS: new-model, speculative-decoding, ready, ci/build, qwen, kv-connector, nvidia, quantization, mrv2, kimi
BODY: ## Purpose ⏎ support https://huggingface.co/Qwen/Qwen3.8-Flash-Next ⏎  ⏎ ## How to run ⏎  ⏎ ``` ⏎ vllm serve Qwen/Qwen3.8-Flash-Next ⏎           --served-model-name qwen3.8-flash-next ⏎           -tp 4 ⏎           --enable-prefix-caching ⏎           --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' ⏎ ``` ⏎ Enable PLE offload(note PLE offload support is in https://github.com/vllm-project/vllm/pull/53899) ⏎ ``` ⏎ VLLM_PLE_CPU_OFFLOAD=1 ⏎ ``` ⏎  …[truncated]

### L3-699e180df4  (L3, 2026-08-31, sha 699e180df48d, PR #53574)
TITLE: [Bugfix][SM120] DSv4: pass contiguous C128A decode topk indices on SM120 (#53574)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_flashmla_sparse.py (+22/-5); vllm/models/deepseek_v4/sparse_mla.py (+9/-2)
LABELS: bug, ready, deepseek, DSv4
BODY: # [Bugfix][SM120] DSv4: keep C128A decode topk indices contiguous on SM120 ⏎  ⏎ ## Purpose ⏎  ⏎ DeepSeek-V4 Flash on SM120 (e.g. RTX 6000 Pro, DGX Spark) with the FlashInfer ⏎ sparse-MLA backend crashes during CUDA-graph capture at server startup: ⏎  ⏎ ``` ⏎ Check failed: (eidx.IsContiguous()) is false: eidx must be contiguous ⏎ ``` ⏎  ⏎ Root cause: since #52823 (adaptive topk width), `build_c128a_topk_metadata` ⏎ returns a width-narrowed slice of the persistent `c128a_ …[truncated]

### L3-d8de4ae322  (L3, 2026-08-31, sha d8de4ae322b5, PR #52912)
TITLE: [Bugfix][KVOffload] P2P tier declares REQUEST_LEVEL on the producer leg (#52912)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py (+235/-4); tests/v1/kv_connector/unit/offloading_connector/utils.py (+1/-1); tests/v1/kv_offload/tiering/p2p/test_manager.py (+28/-1); tests/v1/kv_offload/tiering/test_tiering_offloading.py (+10/-3); vllm/v1/kv_offload/tiering/p2p/manager.py (+8/-0)
LABELS: bug, ready, kv-connector
ISSUES: #52808 [Bug]: PD Multi Tier supplies nothing when the producer already holds the blocks
BODY: A PD producer whose prefix cache is already warm computes no new blocks, so under BLOCK_LEVEL the tiering manager cascades nothing to the P2P secondary tier. The remote decoder's fetch demand then parks until the 30s load timeout and it recomputes the prompt. ⏎  ⏎ Declare OffloadPolicy.REQUEST_LEVEL for requests carrying a remote_decoder kv_request_id. That makes TieringOffloadingManager cascade blocks already resident in the primary tier (read fro …[truncated]

### L3-f9d666f917  (L3, 2026-08-31, sha f9d666f91794, PR #52067)
TITLE: [KV Offload] Forward ownership in KV cache events (#52067)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/offloading_connector/test_events.py (+13/-2); vllm/distributed/kv_events.py (+13/-6); vllm/distributed/kv_transfer/kv_connector/v1/offloading/events.py (+8/-2); vllm/v1/kv_offload/base.py (+2/-0)
LABELS: ready, kv-connector
BODY: ## Purpose ⏎ Add optional ownership information to KV cache events so events emitted by a KV secondary tier can be distinguished from framework-owned events. ⏎  ⏎ No conflicting PRs. ⏎  ⏎ ## Test Plan ⏎ pytest -q tests/v1/kv_connector/unit/offloading_connector/test_events.py

### L3-dbb7fffddb  (L3, 2026-08-31, sha dbb7fffddbca, PR #51705)
TITLE: [ROCm][MLA][DCP] Support causal multi-token verification (#51705)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.mla.rocm_aiter, L3.platform.rocm_selection
FILES: vllm/platforms/rocm.py (+10/-16); vllm/v1/attention/backends/mla/rocm_aiter_mla.py (+530/-82); vllm/v1/attention/ops/rocm_aiter_mla_merge.py (+136/-0); tests/kernels/attention/test_rocm_aiter_mla_causal_verify_mask.py (+20/-5); tests/kernels/attention/test_rocm_aiter_mla_head_padding.py (+2/-0); tests/v1/attention/test_rocm_aiter_mla_fp8_decode_routing.py (+27/-3); tests/v1/attention/test_rocm_aiter_mla_mtp_split.py (+380/-5)
LABELS: rocm, speculative-decoding, ready, deepseek, kv-connector, nvidia, mrv2, dflash, kimi, k3
BODY: ## Summary ⏎  ⏎ Enable ROCm AITER MLA decode context parallelism for causal multi-token target verification. Each verification token gets its correct per-rank causal KV window, and rank-local partial outputs are merged through the existing DCP attention merge. ⏎  ⏎ This PR is scoped to the ROCm AITER target backend. Hybrid prefix-cache geometry and external connector integration remain separate changes. ⏎  ⏎ ## Implementation ⏎  ⏎ - Compute each verification row …[truncated]

### L3-810bc3250c  (L3, 2026-08-31, sha 810bc3250c94, PR #54537)
TITLE: [Frontend][Performance] Resolve async media across modalities concurrently (#54537)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/entrypoints/unit_tests/test_chat_utils.py (+55/-1); vllm/entrypoints/chat_utils.py (+23/-12)
LABELS: frontend, ready, multi-modality
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ `parse_chat_messages_async` records asynchronous media work in ⏎ `AsyncMultiModalItemTracker`, grouped by modality. `resolve_items` currently ⏎ waits for one complete modality group before it starts the next group. For a ⏎ request containing image, audio, and video inputs, this makes independent ⏎ fetch/decode work run one modality group at a time, for example: ⏎  ⏎ ```text ⏎ image group -> audio group -> video group ⏎ ``` ⏎  ⏎ The asynchronous media con …[truncated]

### L3-3593c964de  (L3, 2026-08-31, sha 3593c964de19, PR #49925)
TITLE: [ROCm] Add TheRock preview docker updates, Keep Python 3.12 and Ubuntu 22.04 (#49925)
SOURCES: dependency_pin, release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rock (+992/-0); docker/Dockerfile.rock_base (+360/-0); requirements/build/rock.txt (+19/-0); requirements/build/rocm.txt (+6/-6); requirements/test/rocm.in (+1/-0); requirements/test/rocm.txt (+5/-1); pyproject.toml (+1/-0)
LABELS: rocm, ci/build
BODY: ## Purpose ⏎ This PR adds to using The Rock 7.14 with wheels provided by The Rock as a preview version while keeping Python 3.12 and Ubuntu 22.04 as-is.  ⏎ ## Test Plan ⏎ Full CI runs. ⏎ ## Test Result ⏎ Current persistently failing groups that need to be addressed are: ⏎ ``` ⏎ MI355 ⏎ Language Models Tests (Standard) ⏎ Entrypoints Integration (Pooling) ⏎ ``` ⏎ Keeping track in this BK build:https://buildkite.com/vllm/amd-ci/builds/12222/list ⏎  ⏎  ⏎  ⏎ --- ⏎ [d …[truncated]

### L3-2ba984a5d0  (L3, 2026-08-31, sha 2ba984a5d06d, PR #53598)
TITLE: [ROCm][DSpark][DCP] Serve prefix cache hits under DCP for Kimi-K3 (#53598)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py (+25/-0); tests/v1/core/test_kv_cache_utils.py (+15/-0); tests/v1/core/test_prefix_caching.py (+77/-0); vllm/v1/core/kv_cache_coordinator.py (+18/-10); vllm/v1/core/kv_cache_utils.py (+22/-0); vllm/v1/core/single_type_kv_cache_manager.py (+4/-1)
LABELS: bug, rocm, speculative-decoding, ready, deepseek, kv-connector, nvidia, mrv2, verified, dflash
BODY: ## Summary ⏎  ⏎ Use each KV-cache group's effective DCP geometry when constructing cache managers and performing hybrid prefix-cache lookup. ⏎  ⏎ Kimi-K3 mixes DCP-sharded MLA/full-attention groups with replicated Mamba groups. Applying the process-wide DCP size to every group makes the scheduler, block hashing, and cache managers disagree about block geometry. ⏎  ⏎ This PR is intentionally limited to per-group DCP cache geometry. It does not include: ⏎  ⏎ - DSp …[truncated]

### L3-e2c8eeac40  (L3, 2026-08-31, sha e2c8eeac40d0, PR #53677)
TITLE: [kernel] Fused embedding kernel  (#53677)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-0); benchmarks/kernels/benchmark_vocab_parallel_embedding.py (+197/-0); csrc/libtorch_stable/ops.h (+8/-0); csrc/libtorch_stable/torch_bindings.cpp (+9/-0); csrc/libtorch_stable/vocab_parallel_embedding_kernels.cu (+166/-0); tests/kernels/core/test_vocab_parallel_embedding.py (+126/-0); vllm/_custom_ops.py (+43/-0); vllm/model_executor/layers/vocab_parallel_embedding.py (+30/-10)
LABELS: performance, ready, ci/build
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ before  ⏎ <img width="612" height="548" alt="image" src="https://github.com/user-attachments/assets/22bad978-9119-4053-a9be-6154946c84b0" /> ⏎ after ⏎ <img width="600" height="204" alt="image" src="https://github.com/user-attachments/assets/46f57b6a-a351-4da5-8e00-f7971a6049a0" /> ⏎  ⏎ ## Test Plan ⏎  - Perf comparsion ⏎ <img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/024343b5-3b2e-4142-a116-93301367 …[truncated]

### L3-9acbc5360a  (L3, 2026-08-31, sha 9acbc5360aa5, PR #52068)
TITLE: [KV Offload] Preserve KV event metadata until final residency removal (#52068)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/offloading_connector/test_events.py (+33/-12); vllm/distributed/kv_transfer/kv_connector/v1/offloading/events.py (+20/-7); vllm/v1/kv_offload/base.py (+1/-0)
LABELS: ready, kv-connector
BODY: ## Purpose ⏎ Track offloaded-block residencies by medium and ownership so removing one residency does not discard metadata still needed by another. ⏎  ⏎ Depends on PR #52067, no conflicting PRs. ⏎  ⏎  ⏎ ## Test Plan ⏎ pytest -q tests/v1/kv_connector/unit/offloading_connector/test_events.py

### L3-f5c3cc240b  (L3, 2026-08-31, sha f5c3cc240bc1, PR #53382)
TITLE: [Perf][Kernel] Tune cooperative topk for medium batch-sizes (#53382)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/cooperative_topk.cu (+8/-6); csrc/libtorch_stable/cooperative_topk.cuh (+10/-1); tests/kernels/test_top_k_per_row.py (+14/-2); vllm/model_executor/layers/sparse_attn_indexer.py (+1/-1)
LABELS: performance, ready, deepseek
DEEP_STUDY: deep-study performance PR ()
BODY: Speedups w.r.t. FilteredTopK, which is the backend selected at this moment in MAIN for bs>32  ⏎  ⏎ topK=512 ⏎  ⏎ | Sequence length | BS=33 | BS=40 | BS=48 | BS=64 | ⏎ |---|---:|---:|---:|---:| ⏎ | 64K  | 2.33x | 1.82x | 1.79x | 1.82x | ⏎ | 96K  | 2.24x | 1.66x | 1.65x | 1.64x | ⏎ | 128K | 2.52x | 1.40x | 1.37x | 1.37x | ⏎ | 160K | 2.71x | 1.45x | 1.45x | 1.43x | ⏎ | 192K | 2.99x | 1.52x | 1.52x | 1.53x | ⏎ | 224K | 2.38x | 1.59x | 1.60x | 1.57x | ⏎ | 250K |  …[truncated]

### L3-dafbef15a1  (L3, 2026-08-31, sha dafbef15a1c8, PR #49445)
TITLE: [Core] Add `max_num_queued_reqs` and `max_num_queued_tokens` for queue size management (#49445)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/entrypoints/openai/chat_completion/test_serving_chat.py (+33/-1); tests/entrypoints/speech_to_text/test_speech_to_text_cancellation.py (+2/-0); tests/v1/engine/test_admission_control.py (+384/-0); vllm/config/scheduler.py (+31/-0); vllm/engine/arg_utils.py (+12/-0); vllm/engine/protocol.py (+17/-0); vllm/entrypoints/generate/base/serving.py (+13/-0); vllm/entrypoints/generate/generative_scoring/serving.py (+1/-0); vllm/entrypoints/openai/chat_completion/batch_serving.py (+1/-2); vllm/entrypoints/openai/chat_completion/serving.py (+1/-5); (+8 more)
LABELS: frontend, ready, v1
BODY: vLLM ships with an **unbounded request queue** — a request that gets accepted by the engine in a state where the waiting queue is already saturated, it is silently accepting a (possibly unbounded) TTFT price. ⏎ We currently rely entirely on load balancer to pre-admit only requests that can meet pre-set QoS targets based on a state the LB need to maintain, and offer no tunable knob in-engine. ⏎ There is no mechanism to reject work early and signal u …[truncated]

### L3-bed3280f50  (L3, 2026-08-31, sha bed3280f50dc, PR #50696)
TITLE: [KV offload] Order CPU->GPU loads against the compute stream (#50696)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/kv_offload/cpu/gpu_worker.py (+4/-2)
LABELS: ready
BODY: On models that zero freshly allocated KV blocks (any model with mamba layers, ⏎ i.e. `needs_kv_cache_zeroing`), a CPU->GPU load in the offloading connector can ⏎ be silently wiped. ⏎  ⏎ `SingleDirectionOffloadingHandler.transfer_async` waited on the compute stream ⏎ only for GPU->CPU. Loads ran on their own stream with no ordering against the ⏎ compute stream, which zeroes freshly allocated blocks. When the copy lands ⏎ first, a pending zeroing of its destinat …[truncated]

### L3-e0d27040dd  (L3, 2026-08-31, sha e0d27040ddcc, PR #52571)
TITLE: [Bugfix][KV Offload][P2P] Preserve aborted loads until abort completion (#52571)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_offload/tiering/p2p/test_sessions.py (+88/-0); vllm/v1/kv_offload/tiering/p2p/session/client.py (+9/-11)
LABELS: bug, ready, verified
BODY: ## Motivation ⏎  ⏎ Related to #49829. ⏎  ⏎ Fix the P2P KV-offload request-finish path where `ClientRole.finish()` ⏎ discarded an active load before `AbortAck` or the existing abort timeout ⏎ could produce its failure result. ⏎  ⏎ Previously, dropping the load state also dropped the associated `job_id`, ⏎ preventing the existing failure-completion path from resolving the pending ⏎ primary-tier write. ⏎  ⏎ ## Changes ⏎  ⏎ Retain aborted load state and job identi …[truncated]

### L3-fdbf2ddbd2  (L3, 2026-08-31, sha fdbf2ddbd22a, PR #54042)
TITLE: [Bugfix][CPU] Fix several bugs (#54042)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/cpu_attn.py (+14/-13); csrc/cpu/cpu_fused_moe.cpp (+33/-26); csrc/cpu/sgl-kernels/gemm.cpp (+4/-7); csrc/cpu/torch_bindings.cpp (+6/-5); tests/conftest.py (+9/-1); tests/v1/attention/test_group_head_counts.py (+3/-2); tests/v1/kv_connector/unit/test_bidirectional_kv_transfer.py (+4/-2); vllm/_custom_ops.py (+11/-12); vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+13/-1); vllm/model_executor/layers/utils.py (+7/-5); (+1 more)
LABELS: bug, cpu, kv-connector
BODY: ## Purpose ⏎  ⏎ - Fix CPU fused MoE buffer size overflow issue ⏎ - Fix Qwen3.6 router logits GEMM issue ⏎ - Fix block_size issue on pure SWA models ⏎ - Suppress noise log messages. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-4c58a0c398  (L3, 2026-08-31, sha 4c58a0c398b0, PR #52596)
TITLE: [Bugfix][KV Offload] Unlink /dev/shm region after all workers map it (barrier variant of #51317) (#52596)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/kv_offload/cpu/test_shared_offload_region.py (+155/-0); vllm/v1/kv_offload/cpu/shared_offload_region.py (+74/-34); vllm/v1/kv_offload/cpu/spec.py (+18/-0)
LABELS: bug, ready
ISSUES: #51579 [Bug]: OffloadingConnector CPU tier leaks its /dev/shm mmap file on any unclean exit (including SIGKILL)
BODY: Fixes #51579. `SharedOffloadRegion` backs the CPU offload tier with `/dev/shm/vllm_offload_{engine_id}.mmap`, removed only by `cleanup()`, which never runs on SIGKILL/OOM/crash — every hard exit leaks a `cpu_bytes_to_use`-sized file, and a stale file with a pinned `engine_id` prevents any rank from winning creator election on the next start. ⏎  ⏎ Fix: after every worker `mmap()`s the file, barrier over the worker group (gloo cpu group; `_INNER_DP_W …[truncated]

### L3-c6c33f2b1f  (L3, 2026-08-31, sha c6c33f2b1fac, PR #52191)
TITLE: [CPU] Support FP16/BF16 persisted GDN state on AMX (#52191)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/cpu/sgl-kernels/fla.cpp (+292/-118); tests/kernels/mamba/cpu/test_cpu_gdn_ops.py (+59/-17); tests/platforms/test_cpu.py (+152/-0); vllm/platforms/cpu.py (+72/-6)
LABELS: qwen, cpu
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎ BF16/FP16 GDN state reduces the aligned block size used by hybrid prefix caching and therefore reduce the prefix-caching glassjaw on CPU AMX backends.  ⏎  ⏎ ## Test Plan ⏎ Server ⏎ ```bash ⏎ vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \ ⏎   --kv-cache-dtype {auto|fp8} \ ⏎   --mamba-ssm-cache-dtype {auto|bfloat16} \ ⏎   --language-model-only \ ⏎   --enable-prefix-caching \ ⏎ ``` ⏎ Client ⏎ ```bash ⏎ KV_CACHE_DTYPE=auto ⏎ MAMBA_SSM_CACHE_DTYPE=auto ⏎  ⏎ for PR …[truncated]

### L3-1b9539d37c  (L3, 2026-08-31, sha 1b9539d37c90, PR #51248)
TITLE: [Quantization][Autoround][XPU] Support AutoRound MXFP8 MoE models (#51248)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/quantization/test_auto_round.py (+133/-0); vllm/model_executor/layers/quantization/inc/schemes/inc_mxfp8_moe.py (+196/-0); vllm/model_executor/layers/quantization/inc/schemes/inc_mxfp8_scheme.py (+12/-0)
LABELS: intel-gpu, qwen, quantization, verified
BODY: Adds XPU support for AutoRound-format MXFP8 MoE models. ⏎  ⏎ ### Accuracy ⏎ Evaluated with `Qwen3-30B-A3B-MXFP8-AutoRound` on PIQA. ⏎ |Tasks|Version|Filter|n-shot| Metric |   |Value |   |Stderr| ⏎ |-----|------:|------|-----:|--------|---|-----:|---|-----:| ⏎ |piqa |      1|none  |     0|acc     |↑  |0.7905|±  |0.0094| ⏎ |     |       |none  |     0|acc_norm|↑  |0.7971|±  |0.0094| ⏎  ⏎ #### Evaluation configuration ⏎ ``` text ⏎ vllm ({ ⏎   'pretrained': 'Qwe …[truncated]

### L3-d61b6e1878  (L3, 2026-08-31, sha d61b6e1878a6, PR #54373)
TITLE: [Bugfix][Spec Decode] Take the DFlash draft's RoPE layout from its own config (#54373)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen3_dflash.py (+5/-25); vllm/v1/spec_decode/dflash.py (+0/-15); vllm/v1/worker/gpu/spec_decode/dflash/utils.py (+0/-6)
LABELS: bug, speculative-decoding, ready, qwen, mrv2, dflash
BODY: ### Purpose ⏎  ⏎ The DFlash loader reads `is_neox_style` off the target and stamps it onto the ⏎ draft config, in both the MRV2 and the pre-MRV2 path: ⏎  ⏎ ```python ⏎ is_neox_style = dflash_target_rope_is_neox_style(target_model) ⏎ if is_neox_style is not None: ⏎     draft_model_config.hf_config.is_neox_style = is_neox_style ⏎ ``` ⏎  ⏎ @zzw09773 showed in #53063 why that is wrong: the rotation applies to the ⏎ draft's own Q/K, so the layout that has to hold is the one  …[truncated]

### L3-3a2ed6cbae  (L3, 2026-08-31, sha 3a2ed6cbae16, PR #54636)
TITLE: [Kimi Bug] Fix gdn build_attn_metadata `'KimiK3KDAMetadataBuilder' object has no attribute 'layer_names'` (#54636)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/gdn_attn.py (+1/-2)
LABELS: bug, ready, kimi, k3
BODY: ## Purpose ⏎  ⏎ `vllm serve moonshotai/Kimi-K3   --trust-remote-code   --tensor-parallel-size 8--load-format fastsafetensors   --gpu-memory-utilization 0.85   --enable-prefix-caching   --use-replayssm   --reasoning-parser kimi_k3   --enable-auto-tool-choice   --tool-call-parser kimi_k3   --host 0.0.0.0   --port 30000   --speculative-config '{"model":"RadixArk/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probab …[truncated]

### L3-c28feab989  (L3, 2026-08-31, sha c28feab98919, PR #54646)
TITLE: [Core][MRV2] Freeze gc during V2 CG capture; skip per-descriptor cleanup (#54646)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/compilation/breakable_cudagraph.py (+7/-9); vllm/utils/gc_utils.py (+29/-1); vllm/v1/worker/gpu/model_runner.py (+40/-36); vllm/v1/worker/gpu_model_runner.py (+2/-23)
LABELS: ready, torch.compile, nvidia, mrv2
BODY: The V2 runner's `capture_model` does not freeze/disable gc during CUDA graph capture the way the V1 runner's `_freeze_gc` does, leaving capture exposed to gc cycles that can invalidate an in-progress capture (e.g. a Triton kernel finalized mid-capture unloads its module). It also makes the breakable wrapper's per-descriptor `gc.collect()` + `empty_cache()` both necessary and expensive: ~0.3s per descriptor, ~10x overall capture time (observed on  …[truncated]

### L3-76ff0cdff2  (L3, 2026-08-31, sha 76ff0cdff2de, PR #53821)
TITLE: [Bugfix][ROCm] Preserve AITER unified-attention metadata during graph replay (#53821)
SOURCES: path_core, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.rocm.aiter_unified
FILES: vllm/v1/attention/backends/rocm_aiter_unified_attn.py (+19/-3); tests/v1/attention/test_rocm_attention_backends_selection.py (+40/-0)
LABELS: bug, rocm, nvidia, verified
BODY: ## Purpose ⏎  ⏎ Fix graph-replay correctness for `ROCM_AITER_UNIFIED_ATTN`. ⏎  ⏎ The generic ROCm metadata builder zeros ⏎ `common_attn_metadata.query_start_loc` during graph capture to protect the ⏎ legacy prefix-prefill kernel. AITER unified attention consumes that tensor ⏎ during replay, so sharing the generic capture path corrupts its query ⏎ boundaries. ⏎  ⏎ This PR gives only the AITER unified-attention backend a dedicated metadata ⏎ builder. It retains the inex …[truncated]

### L3-07ea9350ba  (L3, 2026-08-31, sha 07ea9350baf8, PR #53147)
TITLE: [Kernel][Gemma4] Prune Triton sliding-window tiles for multimodal prefixes (#53147)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.triton.unified_attention
FILES: vllm/v1/attention/ops/triton_attention_helpers.py (+49/-9); vllm/v1/attention/ops/triton_unified_attention.py (+6/-1); tests/kernels/attention/test_triton_unified_attention.py (+253/-0)
LABELS: performance, ready
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎ Fix a Triton attention performance regression affecting Gemma 4 multimodal prompts. ⏎ Triton normally prunes KV tiles outside the sliding window. However, when multimodal prefix ranges are present, `compute_tile_loop_bounds` disables that pruning.  ⏎  ⏎ The fix is pretty simple in `vllm/v1/attention/ops/triton_attention_helpers.py` and was already planned as TODO: ⏎ ```python ⏎ # TODO(Isotr0py): sliding window pruning with image bidirectio …[truncated]

### L3-82936c409d  (L3, 2026-08-31, sha 82936c409d17, PR #54172)
TITLE: [Tests][XPU] Limit Qwen2-VL generation length to avoid flaky numerical divergence (#54172)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/generation/test_common.py (+16/-8)
LABELS: intel-gpu, multi-modality, qwen
BODY: ## Summary ⏎  ⏎ This PR stabilizes the Qwen2-VL multimodal generation correctness test on XPU by limiting max_tokens to 64, following the existing CPU workaround for late-generation numerical divergence. ⏎  ⏎ On XPU, the Qwen2-VL multi-image test is flaky when running with BF16 and the ViT FlashAttention backend. HF and vLLM can agree for a long generated prefix and then diverge later in autoregressive decoding, causing check_logprobs_close to fail. …[truncated]

### L3-923949e6e3  (L3, 2026-08-31, sha 923949e6e3b6, PR #49984)
TITLE: [Feat] Add request-level preemption count histogram metric (#49984)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/entrypoints/serve/instrumentator/test_metrics.py (+4/-0); vllm/v1/metrics/loggers.py (+13/-0); vllm/v1/metrics/stats.py (+4/-0)
LABELS: ready, v1, verified
BODY: ## Purpose ⏎  ⏎ This PR introduces request-level preemption tracking by adding a new `vllm:request_num_preemptions` histogram metric. It records the number of times a request is preempted (due to KV cache starvation) before it completes, providing better visibility into system performance and capacity limits under load. ⏎  ⏎ ## Test Plan ⏎  ⏎ End-to-End Metrics integration tests passed successfully (`pytest tests/entrypoints/serve/instrumentator/test_m …[truncated]

### L3-dc9114b201  (L3, 2026-08-31, sha dc9114b20119, PR #50622)
TITLE: [ROCm][MoE] Split AITER CK and Triton MXFP4 W4A16 into separate backends (#50622)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_ocp_mx_moe.py (+25/-21); tests/models/quantization/test_gpt_oss.py (+2/-0); vllm/config/kernel.py (+3/-0); vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a16_moe.py (+403/-0); vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py (+0/-362); vllm/model_executor/layers/fused_moe/oracle/mxfp4.py (+31/-7)
LABELS: rocm, ready, gpt-oss
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: # [ROCm][MoE] Split AITER CK and Triton MXFP4 W4A16 into separate backends ⏎  ⏎ ## Summary ⏎  ⏎ Split the overloaded `AITER_MXFP4_BF16` backend into two: ⏎  ⏎ - **`AITER_MXFP4_BF16`** → CK only (`AiterExperts`), behavior unchanged (gfx950). ⏎ - **`AITER_TRITON_MXFP4_BF16`** (new) → aiter Triton `moe_gemm_a16w4` ⏎   (`AiterW4A16ExpertsMonolithic`), available on gfx942 / gfx950 / gfx1250, ⏎   selectable via `--moe-backend aiter_triton_mxfp4_bf16` and auto-s …[truncated]

### L3-ce2e343be1  (L3, 2026-08-31, sha ce2e343be1f7, PR #53155)
TITLE: [ROCm] Keep GLM-5.2 on MRV1 and disable default breakable cudagraph (#53155)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/test_config.py (+5/-2); vllm/config/vllm.py (+5/-5)
LABELS: rocm, ready, nvidia, glm
BODY: # [ROCm] Keep GLM-5.2 on MRV1 and disable default breakable cudagraph ⏎  ⏎ ## Purpose ⏎  ⏎ #52861 routed the DSA models to the V2 model runner (MRV2) and breakable CUDA ⏎ graphs. On ROCm it excluded `DeepseekV32ForCausalLM` / `DeepseekV4ForCausalLM` ⏎ from MRV2 (the `TODO(rocm)` notes these are "unsupported by MRV2 or slower with ⏎ MRV2 on AMD GPUs") but missed `GlmMoeDsaForCausalLM`. GLM-5.2 (both FP8 and ⏎ MXFP4) thus became the only DSA model defaulting to MR …[truncated]

### L3-d6d6658543  (L3, 2026-08-31, sha d6d665854314, PR #54261)
TITLE: [Kimi-K3][Perf] Make native CUDA AttnRes the SM100 default (#54261)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_kimi_k3_attn_res.py (+235/-0); csrc/libtorch_stable/kimi_k3/attn_res_kernel.cu (+104/-77); csrc/libtorch_stable/ops.h (+5/-4); csrc/libtorch_stable/torch_bindings.cpp (+5/-3); tests/models/kimi_k3/test_attn_res.py (+53/-0); vllm/_custom_ops.py (+4/-2); vllm/models/kimi_k3/nvidia/ops/attn_res.py (+9/-8)
LABELS: performance, ready, nvidia, kimi, k3
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Make the native CUDA Kimi-K3 AttnRes implementation the default for dense ⏎ `hidden_size=7168` inputs on SM100. ⏎  ⏎ This PR: ⏎  ⏎ - extends the existing register/TMEM kernel to cover block counts 0 through 8, ⏎   optional prefix addition, optional output normalization, and block-bank ⏎   writes; ⏎ - uses the existing source-chunk schedule for all supported block counts, ⏎   while preserving a compile-time-folded common path; ⏎ - makes output normalizati …[truncated]

### L3-91752b7a3e  (L3, 2026-08-31, sha 91752b7a3e0c, PR #54634)
TITLE: [K3 Bug] Fix Kimi-K3 RecoverSSM startup failure `'MambaAttentionBackendEnum.GDN_ATTN declares 4 states, but provides 2 state copy funcs'` (#54634)
SOURCES: path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/mamba_utils.py (+3/-2)
LABELS: bug, ready, kimi, k3
BODY: ## Purpose ⏎  ⏎ `vllm serve moonshotai/Kimi-K3   --trust-remote-code   --tensor-parallel-size 8 -- ⏎ load-format fastsafetensors   --gpu-memory-utilization 0.85   --enable-p ⏎ refix-caching   --use-replayssm   --reasoning-parser kimi_k3   --enable- ⏎ auto-tool-choice   --tool-call-parser kimi_k3   --host 0.0.0.0   --port ⏎ 30000   --speculative-config '{"model":"RadixArk/Kimi-K3-DSpark","method ⏎ ":"dspark","num_speculative_tokens":7,"draft_sample_metho …[truncated]

### L3-e16b5e518d  (L3, 2026-08-31, sha e16b5e518db8, PR #50175)
TITLE: [1/N][warmup][DSv4] Migrate generic MLA metadata and indexing kernels (#50175)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+47/-0); vllm/v1/attention/backends/mla/compressor_utils.py (+125/-40); vllm/v1/attention/backends/mla/indexer.py (+163/-101); vllm/v1/attention/backends/mla/sparse_swa.py (+304/-155); vllm/v1/attention/backends/mla/sparse_utils.py (+370/-173); vllm/v1/worker/gpu/model_runner.py (+17/-12); vllm/v1/worker/gpu_model_runner.py (+12/-8); docs/contributing/jit_kernel_warmup.md (+10/-8); tests/model_executor/layers/test_fused_shared_expert.py (+1/-0); tests/model_executor/test_jit_warmup.py (+29/-0); (+7 more)
LABELS: documentation, ready, v1, deepseek, nvidia, mrv2, DSv4, inkling
BODY: Depends on: https://github.com/vllm-project/vllm/pull/49315 ⏎  ⏎ For more details, see parent PR: https://github.com/vllm-project/vllm/pull/49627 and tracking issue https://github.com/vllm-project/vllm/issues/49349 ⏎  ⏎ ## Description ⏎  ⏎ This PR migrates generic MLA metadata and indexing kernels to the shared JIT warmup contract. ⏎  ⏎ ## What Changed ⏎  ⏎ - Migrated sparse-MLA metadata and index-conversion kernels. ⏎ - Migrated prefill chunk metadata and  …[truncated]

### L3-882ca8d696  (L3, 2026-09-01, sha 882ca8d69644, PR #53014)
TITLE: [Kernel] add Flashinfer cutedsl w4a16 linear (#53014)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/config/kernel.py (+1/-1); vllm/utils/flashinfer.py (+49/-0); docs/features/quantization/modelopt.md (+7/-3); tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py (+60/-0); tests/quantization/test_modelopt.py (+11/-1); vllm/model_executor/kernels/linear/__init__.py (+20/-3); vllm/model_executor/kernels/linear/nvfp4/flashinfer.py (+80/-0); vllm/model_executor/layers/quantization/modelopt.py (+2/-2)
LABELS: documentation, ready, nvidia, quantization
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: PLEASE FILL IN THE PR DESCRIPTION HERE ENSURING ALL CHECKLIST ITEMS (AT THE BOTTOM) HAVE BEEN CONSIDERED. ⏎  ⏎ ## Purpose ⏎  ⏎ This PR adds `FlashInferCuteDslNvFp4W4A16LinearKernel`.  Flashinfer already includes a sm12x cute-dsl w4a16 linear kernel. ⏎  ⏎ https://github.com/flashinfer-ai/flashinfer/pull/4466 has added a sm100 cute-dsl w4a16 linear kernel and is included in 0.6.18. vLLM will default to flashinfer cute-dsl instead of Marlin when using sm1 …[truncated]

### L3-446c769482  (L3, 2026-09-01, sha 446c769482dc, PR #53576)
TITLE: [Distributed] Add opt-in FlashInfer PCIe IPC all-reduce backend (#53576)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.trtllm_gen
FILES: tests/distributed/test_comm_ops.py (+1/-0); tests/distributed/test_flashinfer_pcie_ipc_all_reduce.py (+146/-0); vllm/distributed/device_communicators/cuda_communicator.py (+33/-0); vllm/distributed/device_communicators/flashinfer_pcie_ipc_all_reduce.py (+239/-0); vllm/distributed/parallel_state.py (+17/-4); vllm/envs.py (+7/-0); vllm/model_executor/warmup/kernel_warmup.py (+7/-0)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: # [Distributed] Add opt-in FlashInfer PCIe IPC all-reduce backend ⏎  ⏎ ## Purpose ⏎  ⏎ On PCIe-only multi-GPU boxes (e.g. RTX 6000 Pro, SM120 — no NVLink), the default ⏎ NCCL RING_LL all-reduce dominates small-message decode collectives. This PR adds ⏎ an opt-in `flashinfer-ipc` custom all-reduce backend that rides FlashInfer's ⏎ single-node IPC symmetric-memory implementation (flashinfer-ai/flashinfer#4393): ⏎  ⏎ - `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` selec …[truncated]

### L3-ec32f669bb  (L3, 2026-09-01, sha ec32f669bb55, PR #54220)
TITLE: [Feature][MM_UUIDs] Allow empty video URLs when using multi-modal UUIDs (#54220)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/features/multimodal_inputs.md (+7/-0); tests/entrypoints/multimodal/openai/chat_completion/test_video.py (+112/-0); tests/multimodal/test_parse.py (+20/-1); vllm/multimodal/parse.py (+7/-0)
LABELS: documentation, ready, multi-modality
BODY: ## Purpose ⏎ This PR enables omiting the URL when sending a uuid-identifiable video, like for [other modalities](https://docs.vllm.ai/en/latest/features/multimodal_inputs/?h=multimodal+inputs#cached-inputs_1) ⏎  ⏎ ## Test Plan ⏎ ***Unit Tests*** ⏎ ``` ⏎ python -m pytest tests/multimodal/test_parse.py::test_parse_mm_data_accepts_none_ ⏎ cached_item -v ⏎ ``` ⏎  ⏎ ***E2E Tests*** ⏎ ``` ⏎ python -m pytest \ ⏎   tests/entrypoints/multimodal/openai/chat_completion/ …[truncated]

### L3-4707679cd2  (L3, 2026-09-01, sha 4707679cd264, PR #54633)
TITLE: [Bugfix][MiniCPM-V] Route video_embeds to the shared vision parser (#54633)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/test_minicpmv.py (+56/-0); vllm/model_executor/models/minicpmv.py (+17/-1); vllm/model_executor/models/minicpmv4_6.py (+2/-7)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ MiniCPM-V routes video inputs through the image vision parser, stripping the `video_` ⏎ prefix off its multi-modal kwargs on the way: ⏎  ⏎ ```python ⏎ self._parse_and_validate_vision_input( ⏎     "videos", **{k.removeprefix("video_"): v for k, v in kwargs.items()} ⏎ ) ⏎ ``` ⏎  ⏎ That works for `video_pixel_values` and `video_tgt_sizes`. It does not work for ⏎ `video_embeds`, because the image field is named `image_embeds`, not `embeds`. The ⏎  …[truncated]

### L3-58dace61fa  (L3, 2026-09-01, sha 58dace61fa7a, PR #54194)
TITLE: [Kernel] Make prefix-prefill tiling independent of the KV page size (#54194)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.triton.prefix_prefill
FILES: vllm/v1/attention/ops/prefix_prefill.py (+4/-14); tests/kernels/attention/test_prefix_prefill.py (+19/-9)
LABELS: ready, verified, mrv1-only
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Hybrid mamba + full-attention models are given an attention page sized to match ⏎ the mamba page (544, 1040, 1056, ...), a multiple of the kernel alignment rather ⏎ than a power of two. `context_attention_fwd` keyed its tile sizes off that, ⏎ collapsing `BLOCK_M`/`BLOCK_N` from 128/64 down to 32/32. ⏎  ⏎ That collapse is vestigial. It landed in #31380, the same commit that introduced ⏎ `PHYSICAL_BLOCK_SIZE` so page addressing would stop riding on …[truncated]

### L3-8f03625b3d  (L3, 2026-09-01, sha 8f03625b3d14, PR #44834)
TITLE: [CPU][Zen] Route Int8 MoE inference through zentorch on AMD (#44834)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/hardware_tests/cpu.yaml (+3/-0); tests/kernels/moe/test_zen_cpu_int8_moe.py (+253/-0); vllm/model_executor/layers/fused_moe/experts/cpu_moe.py (+162/-0); vllm/model_executor/layers/fused_moe/oracle/int8.py (+2/-1); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_int8.py (+71/-0); vllm/model_executor/models/mixtral.py (+8/-1)
LABELS: rocm, ready, ci/build, v1, cpu, gpt-oss, quantization, mistral
BODY: ## Purpose ⏎ Routes CPU Int8 W8A8 MoE on AMD Zen through zentorch: the expert GEMMs and activation run as a single torch.ops.zentorch.zentorch_fused_moe call, mirroring the existing zentorch W8A8 dense-linear integration. When zentorch's fused-MoE op is unavailable or the deployment config isn't one it supports selection falls through to the existing classes and behaviour elsewhere is unchanged. ⏎  ⏎ - ZenCPUExpertsInt8 (cpu_moe.py): monolithic int8 …[truncated]

### L3-514c7314a0  (L3, 2026-09-01, sha 514c7314a06b, PR #53568)
TITLE: [Perf][Kernel] Initialize NVFP4 padding in quant kernel (#53568)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/quantization/fp4/nvfp4_quant_kernels.cu (+48/-19); tests/kernels/quantization/test_nvfp4_quant.py (+40/-1); vllm/_custom_ops.py (+3/-7)
LABELS: performance, ready, nvidia, quantization
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Follow-up to #49775 and #45739. ⏎  ⏎ #45739 used a separate `torch.zeros` allocation to initialize padded NVFP4 scales. This PR moves the required initialization into the CUDA quantization kernel, allowing the wrapper to safely use `torch.empty` and avoid the extra zero-fill launch. ⏎  ⏎ It also: ⏎  ⏎ - Preserves a specialized fast path for unpadded output. ⏎ - Retains programmatic dependent launch support. ⏎ - Adds sentinel-based regressio …[truncated]

### L3-c866ba9d11  (L3, 2026-09-01, sha c866ba9d1198, PR #53129)
TITLE: [KV Connector] Support heterogeneous TP sharing in Mooncake Store Connector (#53129)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: docs/features/mooncake_store_connector_usage.md (+55/-9); tests/v1/kv_connector/unit/test_mooncake_store_connector.py (+170/-4); tests/v1/kv_connector/unit/test_mooncake_store_layout.py (+180/-0); tests/v1/kv_connector/unit/test_mooncake_store_prepare_values.py (+0/-90); tests/v1/kv_connector/unit/test_mooncake_store_worker.py (+586/-16); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py (+66/-25); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py (+392/-41); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py (+317/-176)
LABELS: documentation, ready, kv-connector
BODY: ## Purpose ⏎  ⏎ This PR allows vLLM instances using the same backend-native LBHNC or LBNHC ⏎ layout, whose TP sizes divide a common Store TP size, to share the same KV ⏎ cache through Mooncake Store. Sharing vLLM instances must use the same PP size ⏎ and satisfy the KV-head and topology requirements described below. ⏎  ⏎ Previously, Mooncake Store keys and physical data layouts were tied to the ⏎ local TP rank. Even when Prefill and Decode instances used the sam …[truncated]

### L3-adebc41b7e  (L3, 2026-09-01, sha adebc41b7e9f, PR #52506)
TITLE: [Mamba] Add FlashInfer ReplaySSM backend (#52506)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/cache.py (+6/-5); vllm/config/vllm.py (+17/-2); vllm/v1/worker/gpu/attn_utils.py (+7/-1); vllm/v1/worker/gpu/model_runner.py (+19/-3); vllm/v1/worker/gpu_model_runner.py (+4/-1); vllm/v1/worker/utils.py (+6/-0); tests/kernels/mamba/test_ssu_dispatch.py (+70/-2); tests/model_executor/test_replayssm_warmup.py (+150/-0); tests/v1/attention/test_replayssm_metadata_builder.py (+46/-12); tests/v1/e2e/test_replayssm_decode.py (+37/-2); (+9 more)
LABELS: ready, needs-rebase, nvidia, mrv2, verified
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Summary ⏎  ⏎ This PR adds FlashInfer as an optional ReplaySSM backend for Mamba2. It calls FlashInfer's native `checkpointing_ssu` API, shares replay trackers per KV-cache group, and autotunes a maximum-batch decode through the existing FlashInfer warmup flow before CUDA-graph capture. FlashInfer ReplaySSM supports Model Runner V1 and V2; Triton ReplaySSM remains V1-only. ⏎  ⏎ ## Blocked on FlashInfer ⏎  ⏎ > [!IMPORTANT] ⏎ > This PR is blocked by [fl …[truncated]

### L3-0d4ad47981  (L3, 2026-09-01, sha 0d4ad47981b6, PR #52017)
TITLE: [Kernel] Add B12X causal paged attention backend (#52017)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.upstream_pip, L3.dispatch.registry
FILES: setup.py (+1/-1); vllm/utils/b12x.py (+5/-0); vllm/v1/attention/backends/b12x.py (+1096/-0); vllm/v1/attention/backends/registry.py (+1/-0); .buildkite/test_areas/kernels.yaml (+1/-1); docs/design/attention_backends.md (+10/-0); tests/v1/attention/test_attention_backends.py (+53/-3); tests/v1/attention/test_b12x.py (+360/-0)
LABELS: documentation, ready, ci/build
BODY: ## Purpose ⏎  ⏎ Builds on the optional B12X dependency and shared lazy-import integration merged in #52016. ⏎  ⏎ This PR adds an explicitly selected ⏎ [B12X](https://github.com/local-inference-lab/b12x) causal paged-attention ⏎ backend for NVIDIA SM120 and SM121 GPUs using vLLM's existing attention backend ⏎ interface. It does not modify generic model-runner behavior or introduce a new ⏎ attention abstraction. ⏎  ⏎ Supported paths include: ⏎  ⏎ - Causal paged MHA, MQA,  …[truncated]

### L3-191cecd51e  (L3, 2026-09-01, sha 191cecd51e25, PR #54560)
TITLE: [Kernel][Qwen] Add Hopper LL-GEMM tuning table for Qwen4Exp (#54560)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_bf16_skinny_gemm.py (+69/-2); vllm/models/qwen4_exp/nvidia/low_latency_gemm.py (+82/-4)
LABELS: ready, qwen
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎  ⏎ Add an independently tuned Hopper (`SM90`) LL-GEMM table for Qwen4Exp / Qwen3.8-Flash-Next decode, using the existing shared CuTe DSL skinny-GEMM implementation. ⏎  ⏎ Following review feedback, the earlier Hopper-only HyperConnection fusion has been removed. The final diff only: ⏎  ⏎ - adds explicit H200 plans for the model's TP=4 local projection shapes; ⏎ - covers selected decode batch sizes from `M={1,2,4,8,16}` instead of reusing the SM103  …[truncated]

### L3-a232e29e9d  (L3, 2026-09-01, sha a232e29e9d71, PR #54306)
TITLE: [Bugfix] Gate sm_100-only kernel tests on the capability family, not >= (#54306)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_cutlass_mla_decode.py (+2/-2); tests/kernels/moe/test_cutedsl_moe.py (+1/-1); tests/kernels/moe/test_routed_experts_capture_monolithic.py (+1/-1); tests/kernels/moe/test_trtllm_bf16_moe.py (+1/-1); tests/kernels/moe/test_trtllm_nvfp4_moe.py (+1/-1)
LABELS: bug, ready, nvidia
BODY: ## Summary ⏎  ⏎ `has_device_capability(100)` is a `>=` test, so a device reporting capability 12.1 satisfies it. Tests that guard sm_100-only kernels with it therefore **run** on consumer and workstation Blackwell, where those kernels do not exist, and fail instead of skipping. ⏎  ⏎ `is_device_capability_family(100)` is the check these guards want. Its docstring states the difference: ⏎  ⏎ > Returns True if the device capability is any `<major>.x`. Mirrors C …[truncated]

### L3-82b7d49a6e  (L3, 2026-09-01, sha 82b7d49a6e30, PR #51217)
TITLE: [MoE] Generalize masked activation for padded layouts (#51217)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/activation_kernels.cu (+297/-48); csrc/libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu (+8/-7); csrc/libtorch_stable/ops.h (+10/-5); csrc/libtorch_stable/torch_bindings.cpp (+8/-2); tests/kernels/core/test_activation.py (+208/-0); tests/kernels/moe/test_batched_moe.py (+145/-3); tests/kernels/moe/test_moe.py (+260/-1); tests/kernels/moe/test_moe_permute_unpermute.py (+36/-0); tests/kernels/moe/test_situ_mul_fp8_quant.py (+7/-7); tests/kernels/moe/utils.py (+3/-1); (+5 more)
LABELS: performance, ready, nvidia
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ - add one shared count-aware MoE activation entry point for a flat valid prefix (`[T, D]`) and per-expert valid prefixes (`[E, T, D]`) ⏎ - support the gated and non-gated activation set explicitly, with a fail-closed `NotImplementedError` when a future activation lacks a masked implementation ⏎ - route Humming, batched Triton, and batched Marlin through `ApplyMoEActivationConfig` instead of backend-specific SITU/clamped-SiLU branches ⏎ - se …[truncated]

### L3-7c5dc571cb  (L3, 2026-09-01, sha 7c5dc571cbd1, PR #51724)
TITLE: [Attention][DSA] Enable W4A16 DSA (#51724)
SOURCES: path_core, dependency_pin, body_keyword
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flash_attn.fork_inline_cmake, L3.mla.common_v1, L3.mla.flashmla_build, L3.mla.flashmla_sparse
FILES: CMakeLists.txt (+2/-1); cmake/external_projects/flashmla.cmake (+2/-1); vllm/model_executor/layers/attention/mla_attention.py (+15/-4); vllm/v1/attention/backends/mla/flashmla_sparse.py (+66/-13); csrc/libtorch_stable/cache_kernels.cu (+104/-0); csrc/libtorch_stable/nvfp4_ds_mla_cache.h (+37/-0); csrc/libtorch_stable/nvfp4_ds_mla_cache_kernels.cu (+281/-0); csrc/libtorch_stable/ops.h (+9/-0); csrc/libtorch_stable/torch_bindings.cpp (+6/-0); tests/kernels/attention/test_cache.py (+151/-0); (+7 more)
LABELS: documentation, ready, ci/build
BODY: Corresponding FlashMLA PR: https://github.com/vllm-project/FlashMLA/pull/18 ⏎  ⏎  ⏎ * concat_and_cache_nvfp4_ds_mla — quantize and store ⏎     - Write path for --kv-cache-dtype nvfp4_fp8_ds_mla, reached through the existing concat_and_cache_mla op. ⏎     - Quantizes the bf16 NoPE latent to e2m1 with one e4m3 scale per 16 elements, and the RoPE to unscaled e4m3. ⏎     - Writes a 352 B entry per token into the paged cache. ⏎ *  cp_gather_and_upconvert_nvf …[truncated]

### L3-1f1f628859  (L3, 2026-09-01, sha 1f1f628859dd, PR #54241)
TITLE: [Feat][MM Hashing]  include media_io_kwargs in multi-modal hashes (#54241)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/multimodal/test_processing.py (+22/-0); tests/renderers/test_multimodal_hashes.py (+92/-0); vllm/inputs/llm.py (+6/-0); vllm/multimodal/processing/inputs.py (+34/-35); vllm/multimodal/processing/processor.py (+1/-0); vllm/renderers/base.py (+8/-0)
LABELS: ready, multi-modality
BODY: ## Purpose ⏎ This PR incorporates `media_io_kwargs` into the cache-key hashing function, identically to `mm_processor_kwargs`, simplifying the [client-side contract](#client-side-contract) for guaranteeing cache correctness with multimodal UUIDs. ⏎  ⏎ ### Hash Derivation ⏎  ⏎ #### UUID and No Media ⏎ **Previously:** `mm_hash = hash(uuid, mm_processor_kwargs)`   ⏎ **After the PR:** `mm_hash = hash(uuid, media_io_kwargs, mm_processor_kwargs)` ⏎  ⏎ #### Medi …[truncated]

### L3-40824284bc  (L3, 2026-09-01, sha 40824284bcb2, PR #49936)
TITLE: [Doc] Document FP8 GEMM kernel selection and Blackwell support (#49936)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .github/CODEOWNERS (+1/-0); docs/features/quantization/llm_compressor/fp8.md (+20/-2)
LABELS: documentation, ci/build, verified, build-docs
BODY: ## Purpose ⏎  ⏎ The FP8 page was out of date and gave users nothing to debug with: ⏎  ⏎ - It said only Hopper and Ada Lovelace support W8A8 — Blackwell was missing. ⏎ - Nothing explained which GEMM kernel vLLM picks, or what to do when the choice goes wrong. ⏎  ⏎ This came up in #46619, where a Blackwell user hit a silent hang with no way to tell what was happening. ⏎  ⏎ This PR adds Blackwell to the supported-hardware lines and a note covering: ⏎  ⏎ - vLLM picks the  …[truncated]

### L3-25efcfa788  (L3, 2026-09-01, sha 25efcfa7887c, PR #52724)
TITLE: [Attention] Enable adaptive verification for FLASHINFER_MLA_SPARSE_DSV4 (#52724)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py (+30/-1)
LABELS: ready, deepseek, nvidia, DSv4
BODY: ## Purpose ⏎  ⏎ Follow-up to #47808. ⏎  ⏎ The FlashInfer sparse MLA decode kernel (`flashinfer_trtllm_batch_decode_sparse_mla_dsv4`) supports variable query lengths via `cum_seq_lens_q` / `max_q_len`, but the backend's metadata builders reported `AttentionCGSupport.UNIFORM_BATCH` instead of `ALWAYS`.  ⏎  ⏎ Adds thin metadata builder subclasses (MLA + SWA) that declare `AttentionCGSupport.ALWAYS`, mirroring the pattern established by the FlashMLA backen …[truncated]

### L3-4ac452ad98  (L3, 2026-09-01, sha 4ac452ad9883, PR #51485)
TITLE: [Core] Release NCCL communicator memory in sleep mode (#51485)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/distributed/test_pynccl.py (+39/-0); tests/v1/worker/test_kv_cache_allocation_scope.py (+4/-1); tests/v1/worker/test_sleep_mode_backend.py (+49/-0); vllm/config/model.py (+4/-0); vllm/device_allocator/sleep_mode_backend.py (+1/-2); vllm/distributed/device_communicators/base_device_communicator.py (+6/-0); vllm/distributed/device_communicators/cuda_communicator.py (+8/-0); vllm/distributed/device_communicators/pynccl.py (+24/-0); vllm/distributed/device_communicators/pynccl_wrapper.py (+21/-0); vllm/distributed/parallel_state.py (+10/-0); (+2 more)
LABELS: ready, v1, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Replacement for #46234 ⏎  ⏎ This recreates #46234, which became unreopenable after its fork was detached. The original PR was not closed for a technical reason; all its review threads were resolved. ⏎  ⏎ ## What ⏎  ⏎ Sleep mode releases weights and KV-cache memory, but NCCL communicators retain their dynamic GPU buffers (~1 GiB/rank at TP4). This change releases that memory during sleep via NCCL's native `ncclCommSuspend(NCCL_SUSPEND_MEM)` / `ncclCommResu …[truncated]

### L3-6c58595c2d  (L3, 2026-09-01, sha 6c58595c2d0d, PR #54794)
TITLE: [Feature] Avoid flashinfer autotune each time when vllm source change (#54794)
SOURCES: path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/config/vllm.py (+7/-3); vllm/model_executor/warmup/flashinfer_autotune_cache.py (+2/-3)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ ```bash ⏎ vllm serve moonshotai/Kimi-K3 \ ⏎   --trust-remote-code \ ⏎   -tp 8 \ ⏎   --load-format fastsafetensors \ ⏎   --enable-prefix-caching \ ⏎   --reasoning-parser kimi_k3 \ ⏎   --enable-auto-tool-choice \ ⏎   --tool-call-parser kimi_k3 \ ⏎   --host 0.0.0.0 \ ⏎   --port 30000 ⏎ ``` ⏎  ⏎ Flashinfer autotune will be enabled each time we pull from main, this costs a lot ⏎  ⏎ ```bash ⏎ [AutoTuner]: Tuning flashinfer::trtllm\_fp4\_block\_scale\_moe …[truncated]

### L3-ab54f5bd83  (L3, 2026-09-01, sha ab54f5bd832d, PR #46872)
TITLE: [Chore] Remove redundant `_pack_topk_ids_weights_kernel` in TrtLLM NvFP4 MoE (#46872)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py (+2/-4); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+2/-4); vllm/model_executor/layers/fused_moe/experts/trtllm_lora_moe.py (+5/-12); vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py (+3/-3); vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py (+5/-8); vllm/model_executor/layers/fused_moe/utils.py (+0/-29)
LABELS: ready, nvidia
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ FlashInfer's unpacked TRTLLM FP4 MoE API now accepts FP32 routing weights. ⏎ Pass `(topk_ids, topk_weights)` directly and remove the explicit ⏎ `trtllm_moe_pack_topk_ids_weights` call, avoiding an unnecessary packing kernel. ⏎  ⏎ ## Test Plan ⏎  ⏎ ```bash ⏎ pre-commit run --all-files ⏎  ⏎ MODEL=/tmp/Mistral-Small-4-119B-2603-NVFP4 ⏎ hf download mistralai/Mistral-Small-4-119B-2603-NVFP4 \ ⏎   --local-dir "$MODEL" --max-workers 8 ⏎  ⏎ CUDA_VISIBLE …[truncated]

### L3-cdefd9d499  (L3, 2026-09-01, sha cdefd9d4997f, PR #54533)
TITLE: [Bugfix] Support Sentence Transformers 5.4+ serialized configs (#54533)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/models/pooling_models/README.md (+3/-3); tests/test_config.py (+108/-0); vllm/transformers_utils/config.py (+39/-18)
LABELS: bug, documentation, ready, verified
ISSUES: #45995 [Bug]: Sentence-Transformers models saved with sentence-transformers>=5.4.0 are silently mis-pooled — compact `1_Pooling/config.json` (`pooling_mode` string) is not parsed, so vLLM falls back to the architecture-default pooler `
BODY: ## Purpose ⏎  ⏎ Fixes #45995. ⏎  ⏎ Sentence Transformers 5.4+ changed both the Pooling configuration schema and ⏎ the fully qualified type paths written to `modules.json`. ⏎  ⏎ In addition to the legacy `sentence_transformers.models.*` paths, serialized ⏎ artifacts can contain: ⏎  ⏎ - `sentence_transformers.sentence_transformer.modules.pooling.Pooling` ⏎ - `sentence_transformers.sentence_transformer.modules.normalize.Normalize` ⏎ - `sentence_transformers.base.modules.no …[truncated]

### L3-7a977c0699  (L3, 2026-09-01, sha 7a977c0699b9, PR #49381)
TITLE: [ModelOpt] Redesign the LinearMethod classes using the generic QuantKey-driven method (#49381)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+8/-2); tests/quantization/test_modelopt.py (+312/-152); vllm/model_executor/kernels/linear/mxfp8/Mxfp8LinearKernel.py (+8/-0); vllm/model_executor/layers/linear.py (+2/-5); vllm/model_executor/layers/quantization/modelopt.py (+899/-858); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+5/-0)
LABELS: ready, needs-rebase, quantization, verified
BODY: ## TL;DR ⏎  ⏎ ModelOpt linear quantization is six near-duplicate `LinearMethod` classes today, one per format (FP8 per-tensor, FP8 per-channel/per-token, FP8 block-weight-only, NVFP4 W4A4, NVFP4 W4A16, MXFP8). This PR replaces all six with **one generic `ModelOptLinearMethod`**, composed from per-`QuantKey` schemes and driven by a `QuantSpec(weight, activation)` pair. Adding a format becomes *data* — a `resolve()` row plus reusable schemes — not a ne …[truncated]

### L3-481839ad9e  (L3, 2026-09-01, sha 481839ad9e5e, PR #53388)
TITLE: [Feature][Spec] Support disabling trailing prefix-cache block dropping (#53388)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/test_config.py (+17/-0); tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py (+3/-3); tests/v1/core/test_kv_cache_utils.py (+1/-0); tests/v1/core/test_mamba_align_chunk_split.py (+14/-1); tests/v1/core/test_prefix_caching.py (+1/-1); tests/v1/core/test_single_type_kv_cache_manager.py (+41/-0); vllm/config/speculative.py (+9/-0); vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py (+5/-4); vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py (+3/-3); vllm/v1/core/kv_cache_utils.py (+1/-1); (+3 more)
LABELS: ready, needs-rebase, kv-connector, scheduler, kv-cache-manager
BODY: ## Purpose ⏎ Add an opt-in `disable_eagle_block_drop` speculative-decoding option for ⏎ EAGLE-family methods, including dSpark. When enabled, vLLM keeps the trailing ⏎ prefix-cache block instead of conservatively dropping it after speculative-model ⏎ prefill. ⏎  ⏎ This does not bypass target-model verification. The option can change which ⏎ draft tokens are proposed and therefore may affect speculative-token acceptance ⏎ rates, but accepted output tokens …[truncated]

### L3-73723b707f  (L3, 2026-09-01, sha 73723b707fe4, PR #54773)
TITLE: [ROCm][MoE] Fix gfx950 block scale swizzle for AITER Triton MXFP4 W4A16 (#54773)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a16_moe.py (+10/-2)
LABELS: rocm, ready
BODY: ## Summary ⏎  ⏎ The AITER Triton MXFP4 W4A16 backend crashes the process on gfx950 (MI355) with a ⏎ GPU memory access fault, because it tells the kernel its block scales are not ⏎ swizzled while weight loading has already swizzled them. This passes the existing ⏎ CDNA4 swizzle gate through to the kernel, which makes the path work on gfx950. ⏎  ⏎ ## The failure ⏎  ⏎ Failing test:  ⏎ `tests/kernels/moe/test_ocp_mx_moe.py::test_rocm_mxfp4_moe_oracle[16-256-25 …[truncated]

### L3-18c53727ce  (L3, 2026-09-01, sha 18c53727cebb, PR #54679)
TITLE: [Bugfix][KV Connector] Fix DecodeBench DCP block selection (#54679)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/kv_connector/unit/test_decode_bench_connector.py (+66/-1); vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py (+24/-12)
LABELS: bug, ready, kv-connector
BODY: # Fix DecodeBenchConnector DCP block selection ⏎  ⏎ ## Purpose ⏎  ⏎ `DecodeBenchConnector` currently converts external-token counts to cache-block ⏎ counts using `cache_config.block_size`. Under decode context parallelism (DCP), ⏎ an attention cache block spans `block_size * dcp_world_size` tokens across the ⏎ DCP ranks. The connector therefore selects too many physical attention blocks ⏎ and can fill the block containing the token that the model must co …[truncated]

### L3-a566ea7e8f  (L3, 2026-09-01, sha a566ea7e8f59, PR #54660)
TITLE: [Perf] Avoid more h2d copies from non-pinned tensors (#54660)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/v1/attention/backends/flashinfer.py (+23/-4); vllm/model_executor/models/qwen2_5_vl.py (+7/-6); vllm/model_executor/models/qwen2_vl.py (+2/-1); vllm/model_executor/models/transformers/multimodal.py (+3/-2); vllm/multimodal/inputs.py (+37/-51)
LABELS: ready, multi-modality, qwen, nvidia
DEEP_STUDY: deep-study performance PR ()
BODY: Follow-on from https://github.com/vllm-project/vllm/pull/54299. ⏎  ⏎ Address more places where async h2d copies are made from paged memory CPU tensors, which could cause a stall harming cpu/gpu overlap, which were exposed via https://github.com/vllm-project/vllm/pull/53491.

### L3-1d8d7a3965  (L3, 2026-09-01, sha 1d8d7a396527, PR #54815)
TITLE: [Bugfix] Fix RoPE construction for deepseek-v4 sparse SWA layers (#54815)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/common/rope.py (+17/-1)
LABELS: bug, ready, deepseek, DSv4
BODY: ## Purpose ⏎ - Refer to https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py#L481-L485 and https://github.com/huggingface/transformers/pull/45892, deepseek-v4 should use plain RoPE without YaRN for sparse SWA layer ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-fc72fc39ac  (L3, 2026-09-01, sha fc72fc39ace2, PR #54781)
TITLE: [Kimi Bug] Fix `cannot access local variable 'active_non_spec_mask_cpu'` (#54781)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/kimi_k3/test_kda_metadata.py (+143/-1); vllm/models/kimi_k3/nvidia/kda_metadata.py (+2/-0)
LABELS: bug, ready, kimi, k3
BODY: ## Purpose ⏎  ⏎  ⏎ ```bash ⏎ vllm serve moonshotai/Kimi-K3 \ ⏎   --trust-remote-code \ ⏎   --tensor-parallel-size 8 \ ⏎   --load-format fastsafetensors \ ⏎   --gpu-memory-utilization 0.92 \ ⏎   --enable-prefix-caching \ ⏎   --use-replayssm \ ⏎   --reasoning-parser kimi_k3 \ ⏎   --enable-auto-tool-choice \ ⏎   --tool-call-parser kimi_k3 \ ⏎   --host 0.0.0.0 \ ⏎   --port 30000 \ ⏎   --speculative-config '{"model":"RedHatAI/Kimi-K3-speculator.dspark","method":"dspa …[truncated]

### L3-259a209bfa  (L3, 2026-09-01, sha 259a209bfa8a, PR #53524)
TITLE: [Kimi-K3][Perf] Prefetch ll_bf16 router weights for M=1 (#53524)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/test_ll_bf16_gemm.py (+23/-0); vllm/model_executor/kernels/linear/cute_dsl/_ll_bf16_dotprod.py (+61/-5); vllm/model_executor/kernels/linear/cute_dsl/ll_bf16.py (+15/-3)
LABELS: ready, verified, kimi, k3
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Reduce the Kimi-K3 single-token AttnRes-to-router latency by loading the first ⏎ `ll_bf16` router-weight tiles before the incoming PDL wait. ⏎  ⏎ The optimized kernel is selected from the runtime `M` shape: ⏎  ⏎ - `M == 1`: use the weight-prefetch PDL specialization; ⏎ - every other `M`: use the original `ll_bf16` kernel unchanged. ⏎  ⏎ The earlier AttnRes and latent-MoE-tail changes have been removed. AttnRes is ⏎ now provided by #54261, which is incl …[truncated]

### L3-ee3c00bbf4  (L3, 2026-09-01, sha ee3c00bbf47e, PR #51453)
TITLE: [Performance] Register Triton W4A16 GEMM as a custom op (#51453)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/quantization/test_triton_w4a16.py (+33/-0); vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py (+32/-1)
LABELS: rocm, ready
ISSUES: #49699 [Performance]: Compile mode 3 degrades triton w4a16 kernel performance in few request scenarios.
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Register the Triton W4A16 GEMM as an opaque custom op with a fake implementation similar to how `rdna_hybrid_w4a16.py` does it. ⏎  ⏎ Without this boundary, `torch.compile` traces the Python tile-selection logic and freezes the prefill configuration for decode workloads. The custom op ensures tile selection runs with the actual runtime shape. ⏎  ⏎ This fixes #49699. ⏎  ⏎ ## Test Plan ⏎  ⏎ ```bash ⏎ pytest -q \ ⏎   tests/kernels/quantization/te …[truncated]

### L3-f4e6136146  (L3, 2026-09-02, sha f4e613614628, PR #54859)
TITLE: [Kimi-K3] Bump FlashKDA to fix unstable inverse (#54859)
SOURCES: dependency_pin, release_notes
ARTIFACT_HINTS: -
FILES: cmake/external_projects/flashkda.cmake (+1/-1); tests/models/kimi_k3/test_kda.py (+50/-0)
LABELS: ready, ci/build, kimi, k3
BODY: ## Purpose ⏎  ⏎ See https://github.com/vllm-project/FlashKDA/pull/10 ⏎  ⏎ We also add a unit test here ⏎  ⏎ We don't expect e2e perf regression since the kernel is only slower by 1-2% in microbenchmarks. ⏎  ⏎ ## Test Plan ⏎  ⏎ TP8 GB300 ⏎ - GSM8K: 0.9651, 2,638 requests, zero errors ⏎ - OCRBench: 0.894, 1,000 requests, zero errors ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-3ba9907a1d  (L3, 2026-09-02, sha 3ba9907a1db2, PR #54697)
TITLE: [Kimi-K3] Overlap low-M TP8 KDA projections (#54697)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_kimi_k3_kda_projection.py (+221/-0); tests/kernels/test_bf16_skinny_gemm.py (+118/-0); vllm/model_executor/warmup/kernel_warmup.py (+7/-0); vllm/models/kimi_k3/nvidia/kda.py (+43/-12); vllm/models/kimi_k3/nvidia/low_latency_gemm.py (+195/-0); vllm/models/kimi_k3/nvidia/model.py (+1/-0); vllm/models/kimi_k3/nvidia/ops/cute_dsl/kda_skinny_gemm.py (+507/-0)
LABELS: performance, ready, kimi, k3
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Reduce Kimi-K3 TP8 decode latency by running the Q/K/V/G projection concurrently with the F_A/beta -> F_B branch during CUDA graph replay. ⏎  ⏎ This change: ⏎  ⏎ - splits the existing contiguous QKVG/F_A/beta packed weight without copying it; ⏎ - uses the shared multi-stream utility to fork and join the projection branches; ⏎ - adds TP8-specific CuTeDSL skinny-N and skinny-K GEMMs for F_A/beta and F_B, using TVM FFI tensor conversion and PDL; ⏎ - u …[truncated]

### L3-396c5a5632  (L3, 2026-09-02, sha 396c5a563223, PR #54869)
TITLE: [Bugfix] Lazy-import FlashInfer PCIe IPC all-reduce in kernel_warmup (#54869)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/warmup/kernel_warmup.py (+8/-4)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ Fix a deterministic L4 V1 Engine regression on main introduced by #53576 (reported by @StevenWang-CY in https://github.com/vllm-project/vllm/pull/53576#issuecomment-5502728637). ⏎  ⏎ `kernel_warmup.py` imported `flashinfer_pcie_ipc_all_reduce` at module scope. That module does `import flashinfer.comm`, which initializes CUDA at import time. In pytest this initializes CUDA in the parent process; `_maybe_force_spawn()` then correctly switch …[truncated]

### L3-584e8f0dda  (L3, 2026-09-02, sha 584e8f0dda42, PR #49869)
TITLE: [Model] Fix GLM-OCR MTP weight loading (#49869)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/registry.py (+0/-1); vllm/model_executor/models/utils.py (+7/-1)
LABELS: ready, glm
ISSUES: #49856 [Bug]: [MTP] ValueError when loading zai-org/GLM-OCR with MTP speculative decoding (model.layers.16.mtp_block uninitialized)
BODY: ### Summary ⏎  ⏎ - recognize `model.language_model.layers.*` as a valid MTP checkpoint prefix ⏎ - exercise GLM-OCR MTP through the online model-initialization registry ⏎ - keep MTP detection shared by model loading and `skip_spec_layers()` ⏎  ⏎ ### Why ⏎  ⏎ Fixes #49856. ⏎  ⏎ The current `zai-org/GLM-OCR` checkpoint stores its appended MTP layer under ⏎ `model.language_model.layers.16.*`. `GlmOcrMTP.load_weights()` calls ⏎ `get_spec_layer_idx_from_weight_name()` before ⏎  …[truncated]

### L3-1c26e57d3c  (L3, 2026-09-02, sha 1c26e57d3c7b, PR #54782)
TITLE: [Bugfix] Raise for unavailable piecewise CUDA graphs (#54782)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/cudagraph_utils.py (+20/-0)
LABELS: bug, ready, nvidia, mrv2
BODY: ## Purpose ⏎ - In https://github.com/vllm-project/vllm/pull/54566, I found module with "no torch.compile + no breakable piecewise CG" will cause silent garbled outputs regression by accident. And it costs me a whole morning to debug it. 😅  ⏎ - This PR adds an error raise there to avoid silent regression ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-3b45d053b4  (L3, 2026-09-02, sha 3b45d053b4bb, PR #53829)
TITLE: [Bugfix][Model] Fix CohereASR streaming audio-token estimate (unit + subsampling) (#53829)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/test_cohere_asr.py (+34/-0); vllm/model_executor/models/cohere_asr.py (+9/-3)
LABELS: bug, ready, multi-modality, verified, cohere
BODY: ## Summary ⏎  ⏎ `CohereAsrForConditionalGeneration.get_num_audio_tokens(audio_duration_s, stt_config, model_config)` — the duration-based estimate used to add audio tokens to `prompt_tokens` for **streaming transcription usage stats** — had two bugs in one small classmethod: ⏎  ⏎ 1. **Unit mismatch.** It used `preprocessor["window_stride"]` (a stride in **seconds**, ~`0.01`) directly as the divisor of `audio_duration_s * sample_rate` (a **sample** count) …[truncated]

### L3-b205750fe0  (L3, 2026-09-02, sha b205750fe0ac, PR #54171)
TITLE: [Bugfix][ROCm][Build] fix profiler hang due to queue interposition bug (#54171)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm_base (+12/-0)
LABELS: bug, rocm, ready, ci/build
ISSUES: #54087 [Bug]: torch profiler hangs on ROCm after rocprofiler-sdk 1.3.2 bump
BODY: ## Purpose ⏎  ⏎ Fixes https://github.com/vllm-project/vllm/issues/54087. ⏎  ⏎ Applies two bug fixes that are not part of the rocprofiler-sdk commit we're on (as they were not part of the therock-7.14 release):  ⏎ - https://github.com/ROCm/rocm-systems/pull/7924 ⏎ - https://github.com/ROCm/rocm-systems/pull/7796 ⏎  ⏎ Alternative: We could also bump to rocprofiler-sdk 1.3.5 but that will be a larger change that ver is on the therock-10.0 line. ⏎  ⏎ ## Test P …[truncated]

### L3-003e34341a  (L3, 2026-09-02, sha 003e34341a82, PR #54513)
TITLE: [Qwen3.8-Flash-Next] Separate prefill and decode paths for QSA indexer (#54513)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/qwen4_exp/test_qsa_reference.py (+252/-83); vllm/model_executor/warmup/kernel_warmup.py (+4/-0); vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py (+66/-0); vllm/models/qwen4_exp/common/qsa_cache.py (+86/-20); vllm/models/qwen4_exp/nvidia/indexer_qsa.py (+79/-15); vllm/models/qwen4_exp/nvidia/ops/qsa.py (+1/-405); vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py (+617/-0)
LABELS: qwen
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ #53896 ships with a simple QSA indexer kernel that is used for both decode and prefill requests. This is not efficient as we can't design kernels specialized for decode and prefill shapes separately. This PR splits a mixed batch into decode and prefill requests, so that efficient decode/prefill kernels can be invoked separately, similar to how other attention backends work in vLLM. ⏎  ⏎ Currently this PR ships 2 decode/prefill-special …[truncated]

### L3-872084fb77  (L3, 2026-09-02, sha 872084fb773b, PR #54817)
TITLE: [CI] Add Kimi-K3-pruned75-DSpark-TP4 gsm8k eval (#54817)
SOURCES: path_core, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/prefill/flashinfer.py (+33/-28); .buildkite/test_areas/lm_eval.yaml (+3/-1); tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-confidence-TEP4.yaml (+0/-0); tests/evals/gsm8k/configs/Kimi-K3-pruned75-DSpark-TP4.yaml (+26/-0); tests/evals/gsm8k/configs/models-spec-decode.txt (+2/-1)
LABELS: ci/build, nvidia, dflash, kimi, k3
BODY: ## Purpose ⏎  ⏎ Adds Kimi K3 coverage by running a 75% expert-pruned model on 4xB200. ⏎  ⏎ The B200 job enables VLLM_GPU_SYNC_CHECK=error. FlashInfer MLA prefill planning currently copies GPU indptr tensors to the CPU, so the synchronization detector raises before evaluation starts. This update temporarily allows that known synchronization around the main and chunked-context FlashInfer planner calls. ⏎  ⏎ This does not duplicate #52657. That PR is the perman …[truncated]

### L3-dbf1a044ea  (L3, 2026-09-02, sha dbf1a044eacd, PR #54847)
TITLE: [Bugfix] Fix ColQwen3.5 pooler projector initialization (#54847)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/colqwen3_5.py (+10/-17)
LABELS: bug, ready, qwen
BODY: **Issue:** ColQwen3.5 model loading failed with missing weight errors (pooler.head.projector.0.weight/bias) and runtime shape mismatches due to double-projection. ⏎  ⏎ **Root cause:** The model created `custom_text_proj` but passed `projector=None` to `pooler_for_token_embed()`, causing `_load_st_projector()` to create an uninitialized ST projector in the pooler head. The model then applied projection in forward(), resulting in incompatible dimensi …[truncated]

### L3-1356635d83  (L3, 2026-09-02, sha 1356635d837c, PR #54566)
TITLE: [New model][Multimodal] Add DeepSeek-V4-Flash-Vision-Exp support (#54566)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/nvidia/flashmla.py (+19/-2); vllm/v1/attention/backends/mla/sparse_swa.py (+280/-74); csrc/libtorch_stable/moe/moe_ops.h (+3/-1); csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu (+172/-28); csrc/libtorch_stable/moe/torch_bindings.cpp (+2/-1); docs/models/supported_models.md (+1/-0); tests/config/test_model_arch_config.py (+31/-0); tests/kernels/moe/test_topk_softplus_sqrt.py (+257/-5); tests/models/multimodal/processing/test_tensor_schema.py (+5/-0); tests/models/registry.py (+3/-0); (+27 more)
LABELS: documentation, intel-gpu, ready, ci/build, multi-modality, deepseek, nvidia, DSv4
BODY: ## Purpose ⏎  ⏎ We have provided the docker image for this model, please refer to https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp ⏎  ⏎  ⏎ ## Test Plan ⏎ Setup: 4x NVIDIA GB200 (TP=4, expert parallel), fp8 KV cache, block size 256, ⏎  ⏎ ```bash ⏎ export VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="trtllm_fp4_block_scale_moe,flashinfer::trtllm_fp4_block_scale_moe" ⏎  ⏎ vllm serve deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \ ⏎     --tensor-parallel-size 4  …[truncated]

### L3-a56654d6de  (L3, 2026-09-02, sha a56654d6de06, PR #54565)
TITLE: [K3 Perf] Enable DSV3 GEMM for inner-contiguous and row-strided tensors, 12%~81% kernel performance improvement (#54565)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/dsv3_fused_a_gemm.cu (+35/-27); tests/kernels/test_bf16_skinny_gemm.py (+6/-6); vllm/models/kimi_k3/nvidia/low_latency_gemm.py (+7/-2)
LABELS: ready, kimi, k3
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ Enable DSV3 GEMM for inner-contiguous and row-strided tensors ⏎  ⏎ They previously fell back to `F.linear`/NVJet because they were not fully contiguous. ⏎  ⏎ This PR fixes the issue ⏎  ⏎ ## Test ⏎  ⏎ Acc covered in unit test ⏎  ⏎ Perf can be seen in this AI generated script ⏎  ⏎ ```py ⏎ import torch ⏎ import torch.nn.functional as F ⏎  ⏎ from vllm.models.kimi_k3.nvidia import low_latency_gemm ⏎ from vllm.triton_utils import triton ⏎  ⏎ # name, M, N, K …[truncated]

### L3-f81eb41934  (L3, 2026-09-02, sha f81eb4193431, PR #54803)
TITLE: [Bugfix] `adjust_dcp_kv_cache_interleave_size` for NixlConnector only (#54803)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/test_config.py (+74/-5); vllm/config/vllm.py (+16/-16); vllm/distributed/kv_transfer/kv_transfer_state.py (+1/-1)
LABELS: bug, kv-connector
BODY: DCP kv block adjustment is not being gated on PD connector being present. ⏎ I like the alternative approach in https://github.com/vllm-project/vllm/pull/54457, but I think it might be slightly overkill: I do not expect the number of connectors to grow here.  ⏎ Basically for PD this is a requirement ow block content gets fractioned and the unit of transfer is the block itself. ⏎  ⏎ This PR simply gates the adjustment behind NixlConnector being present …[truncated]

### L3-f870b92976  (L3, 2026-09-02, sha f870b9297685, PR #54517)
TITLE: [Qwen3.8-Flash-Next] Fuse Qwen4Exp PLE kernels (#54517)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/qwen4_exp/test_ple.py (+983/-65); vllm/models/qwen4_exp/nvidia/model.py (+9/-5); vllm/models/qwen4_exp/nvidia/mtp.py (+2/-2); vllm/models/qwen4_exp/nvidia/ops/ple.py (+669/-0); vllm/models/qwen4_exp/nvidia/ple_layer.py (+173/-516); vllm/v1/attention/backends/short_conv_attn.py (+0/-6)
LABELS: ready, qwen
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎  ⏎ - Optimizations: ⏎   - Fuse n-gram ID generation, dilated short-conv, and the PLE gate on NVIDIA ⏎   - Merge key/value projections ⏎ - n-gram ID generation still has PyTorch fallback for future PLE offload (#53899, not landed in main) ⏎ - Original PyTorch implementations of dilated short-conv (prefill, decode, spec-decode) are moved to `test_ple.py` for correctness testing. Production code only contains Triton kernel now.  ⏎  ⏎ ### Correc …[truncated]

### L3-3e9d364ff7  (L3, 2026-09-02, sha 3e9d364ff727, PR #54984)
TITLE: [CI/Build][ROCm] Guard the two CUDA-only tests in test_bf16_skinny_gemm (#54984)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_bf16_skinny_gemm.py (+9/-2)
LABELS: rocm, nvidia
BODY: ## Purpose ⏎  ⏎ The `(MI355) Miscellaneous Kernels` job fails on two tests in ⏎ `tests/kernels/test_bf16_skinny_gemm.py`. Neither failure indicates a bug in ⏎ the code under test. Both are unstated CUDA assumptions inside the tests ⏎ themselves. ⏎  ⏎ That job is added by [#50519](https://github.com/vllm-project/vllm/pull/50519), which extends ROCm CI coverage to the portable ⏎ parts of upstream CUDA test groups. ⏎  ⏎ Failing run: https://buildkite.com/vllm …[truncated]

### L3-ffe3bb3c72  (L3, 2026-09-02, sha ffe3bb3c723d, PR #49209)
TITLE: [Hardware][XPU] Register matmul and linear batch-invariant kernels for XPU (#49209)
SOURCES: symbol_pickaxe
ARTIFACT_HINTS: -
FILES: docs/features/batch_invariance.md (+23/-1); tests/v1/determinism/test_batch_invariance.py (+0/-11); vllm/engine/arg_utils.py (+13/-0); vllm/model_executor/determinism/batch_invariant.py (+180/-19); vllm/model_executor/determinism/batch_invariant_configs.py (+50/-0); vllm/model_executor/layers/linear.py (+3/-1); vllm/model_executor/layers/vocab_parallel_embedding.py (+3/-1)
LABELS: documentation, intel-gpu, ready, v1
BODY: ## Purpose ⏎ This is the second (2/2) PR introducing batch invariance to Intel XPU devices. You can find the first one [here](https://github.com/vllm-project/vllm/pull/41934). ⏎  ⏎ This PR registers the missing matmul and linear kernels. It also adds a new matmul_kernel_descriptor_persistent Triton kernel for better matmul performance on XPU. ⏎  ⏎ ## Test Plan ⏎ Tested with unit tests covering batch invariance on XPU. ⏎  ⏎ ## Test Result ⏎ All tests pass. …[truncated]

### L3-c6bca6e585  (L3, 2026-09-02, sha c6bca6e58540, PR #54918)
TITLE: [Bugfix][Multimodal] Scope cache hash kwargs by modality (#54918)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/multimodal/test_processing.py (+47/-0); vllm/multimodal/processing/inputs.py (+41/-10)
LABELS: bug, ready, multi-modality
BODY: ## Purpose ⏎  ⏎ Fix unnecessary multimodal processor cache misses caused by unrelated ⏎ modality options being included in every media item's cache hash. ⏎  ⏎ The production trigger chain is: ⏎  ⏎ ```text ⏎ request/config media_io_kwargs or mm_processor_kwargs ⏎   -> renderer._process_multimodal() ⏎   -> ProcessorInputs ⏎   -> ProcessorInputs.get_mm_hashes() ⏎   -> multimodal processor cache hit/miss check ⏎   -> preprocessing of cache-missing media items ⏎ `` …[truncated]

### L3-1945a94575  (L3, 2026-09-02, sha 1945a9457563, PR #54996)
TITLE: [Bugfix][Tests] Stabilize B12X linear kernel checks (#54996)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/quantization/test_block_fp8.py (+4/-2); tests/model_executor/kernels/test_b12x_linear.py (+5/-0)
LABELS: bug, ready
BODY: ## Summary ⏎  ⏎ - Allow the expected normalized error from B12X four-way split-K while tightening the cosine-similarity requirement. ⏎ - Make the B12X W4A16 fallback test independent of the host GPU and unrelated NVFP4 backend registrations. ⏎  ⏎ ## Rationale ⏎  ⏎ B12X block-FP8 split-K CTAs atomically accumulate into a BF16 output. Atomic ordering is nondeterministic, so individual outputs can alternate between adjacent BF16 values one ULP apart. On the 48-SM …[truncated]

### L3-e47356c63e  (L3, 2026-09-02, sha e47356c63e4a, PR #55002)
TITLE: [ROCm][Installation] Add mooncake package to image using public wheels (#55002)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/scripts/ci-bake-rocm.sh (+3/-40); docker/Dockerfile.rocm (+1/-103); docker/ci-rocm.hcl (+1/-25); requirements/rocm.txt (+2/-0)
LABELS: rocm, ci/build
BODY: ## Purpose ⏎  ⏎ Updating the recent [PR#52650](https://github.com/vllm-project/vllm/pull/52650) by removing those changes and simply installing mooncake via the public wheels that are now published. ⏎  ⏎ Same testing was done as original PR, and the results are the same. ⏎  ⏎ Addressed issue #51193 ⏎  ⏎ ## Summary by CodeRabbit ⏎  ⏎ - **Changes** ⏎   - ROCm build requirements now include the Mooncake transfer engine, version 0.3.13 or later, supporting KV-cache  …[truncated]

### L3-1f76efaa21  (L3, 2026-09-03, sha 1f76efaa2195, PR #55063)
TITLE: [Model] Add K2-Horizon model support (#55063)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/models/supported_models.md (+1/-0); tests/models/registry.py (+5/-0); tests/reasoning/test_k2_horizon_reasoning_parser.py (+214/-0); tests/tool_parsers/test_k2_horizon_tool_parser.py (+318/-0); vllm/model_executor/models/k2_horizon.py (+1445/-0); vllm/model_executor/models/registry.py (+1/-0); vllm/reasoning/__init__.py (+4/-0); vllm/reasoning/k2_horizon_reasoning_parser.py (+159/-0); vllm/tool_parsers/__init__.py (+4/-0); vllm/tool_parsers/k2_horizon_tool_parser.py (+462/-0)
LABELS: documentation, new-model, ready, tool-calling, verified
BODY: ## Purpose ⏎  ⏎ Adding K2-Horizon model architecture, including its reasoning parser and tool parser. ⏎  ⏎ The changes include: ⏎  ⏎ - Adding the K2-Horizon model implementation in `vllm/model_executor/models/k2_horizon.py`. ⏎ - Adding the K2-Horizon reasoning parser in `vllm/reasoning/k2_horizon_reasoning_parser.py`. ⏎ - Adding the K2-Horizon tool-call parser in `vllm/tool_parsers/k2_horizon_tool_parser.py`. ⏎  ⏎ ## Test Plan ⏎  ⏎ Test K2-Horizon model load …[truncated]

### L3-0d3ede3e3b  (L3, 2026-09-03, sha 0d3ede3e3bd6, PR #54969)
TITLE: [Bugfix][Model] Enable torch.compile for StableLM (#54969)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/stablelm.py (+2/-0)
LABELS: bug, torch.compile
BODY: ## Purpose ⏎ `StableLM` is missing the `@support_torch_compile` decorator, leaving the model uncompiled. Since `CudaGraphManager.capture()` requires a compiled module to record piecewise CUDA graphs, engine init fails under the default `cudagraph_mode=FULL_AND_PIECEWISE` with: ⏎  ⏎ RuntimeError: StablelmForCausalLM: piecewise CUDA graphs (cudagraph_mode=FULL_AND_PIECEWISE) unavailable, model is not torch-compiled and breakable CUDA graph is off. ⏎  ⏎  …[truncated]

### L3-848ab131bc  (L3, 2026-09-03, sha 848ab131bcdb, PR #55062)
TITLE: [Perf] Accumulate Conformer attention scores with baddbmm (#55062)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/multimodal/test_conformer_encoder.py (+111/-0); vllm/model_executor/models/conformer_encoder.py (+13/-3)
LABELS: ready, multi-modality
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Conformer relative attention currently materializes the content scores, adds ⏎ the shifted positional scores, and then launches a separate in-place scale. ⏎ This PR uses standard `torch.baddbmm` to accumulate the scaled content scores ⏎ directly into the scaled positional scores: ⏎  ⏎ ```text ⏎ before: {bmm: 2, add: 1, mul_: 1} ⏎ after:  {bmm: 1, baddbmm: 1} ⏎ ``` ⏎  ⏎ This removes one score-sized intermediate and the separate add/scale kernels. ⏎ It keeps …[truncated]

### L3-bf95f58d10  (L3, 2026-09-03, sha bf95f58d1089, PR #54651)
TITLE: [Core] Triton kernel for small-batch top-p only masking (#54651)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/sample/test_topk_topp_sampler.py (+71/-0); vllm/v1/sample/ops/topk_topp_sampler.py (+4/-12); vllm/v1/sample/ops/topk_topp_triton.py (+552/-1)
LABELS: ready
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ### Split-row Triton top-p pipeline — dispatch and GB200 results ⏎  ⏎ #### Which cases use the new split-row pipeline ⏎  ⏎ Dispatch in `apply_top_k_top_p_triton` (with the `batch >= 8` gate removed, all CUDA batches reach this): ⏎  ⏎ | Condition | Path | ⏎ |---|---| ⏎ | batch > 64, or p is None, or CPU/XPU | monolithic kernel (unchanged) | ⏎ | batch ≤ 64, pure top-p batch (k is None) | **split pipeline only**; per row, only rows with p < 1.0 are processed …[truncated]

### L3-e55b93f296  (L3, 2026-09-03, sha e55b93f2969d, PR #55041)
TITLE: [Core] Deprecate "all" mamba cache mode (#55041)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/test_routed_experts_capture.py (+4/-1); vllm/config/cache.py (+11/-0); vllm/config/vllm.py (+3/-0)
LABELS: ready
BODY: And fall back to MRV1 if configured since it's not implemented in MRV2.

### L3-98ed0856f3  (L3, 2026-09-03, sha 98ed0856f31f, PR #53906)
TITLE: [Model] add GLM-5.3-Flash support (#53906)
SOURCES: path_core, symbol_pickaxe, release_notes
ARTIFACT_HINTS: L3.cache.cuda_reshape, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1, L3.mla.flashinfer_sparse, L3.mla.rocm_aiter, L3.mla.rocm_aiter_sparse, L3.dispatch.registry, L3.dispatch.abstract_interface
FILES: vllm/model_executor/layers/attention/mla_attention.py (+1/-1); vllm/model_executor/layers/mla.py (+22/-3); .buildkite/test_areas/kernels.yaml (+6/-1); csrc/libtorch_stable/cache_kernels.cu (+9/-4); tests/kernels/attention/test_flashinfer_mla_decode.py (+168/-2); tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py (+2/-0); tests/kernels/mamba/test_gdn_prefill_flashinfer.py (+53/-0); tests/kernels/test_kpool_decode_update_batched.py (+549/-0); tests/kernels/test_mhc_kernels.py (+225/-1); tests/models/registry.py (+6/-0); (+84 more)
LABELS: new-model, rocm, speculative-decoding, ready, torch.compile, ci/build, multi-modality, kv-connector, nvidia, quantization
BODY: ## Purpose ⏎  ⏎ @JaredforReal did most of the work for this day0 support https://github.com/vllm-project/vllm/pull/53906/changes/933876c388fb129ad82590660e6506614559cb86, thanks! ⏎  ⏎ add support for https://huggingface.co/zai-org/GLM-5.3-Flash ⏎  ⏎  ⏎ **Please use docker image to run this model, see https://recipes.vllm.ai/zai-org/GLM-5.3-Flash** ⏎  ⏎ Only the first commit https://github.com/vllm-project/vllm/pull/53906/changes/933876c388fb129ad82590660e …[truncated]

### L3-facd9a74a1  (L3, 2026-09-03, sha facd9a74a1cd, PR #54856)
TITLE: [Model Runner V2][Spec Decode] Skip DP sync for all speculator uniform decodes (#54856)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/dp_utils.py (+13/-1); vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py (+7/-1); vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (+8/-5); vllm/v1/worker/gpu/spec_decode/speculator.py (+20/-0)
LABELS: speculative-decoding, ready, mrv2, dflash
BODY: # Summary ⏎ Continuation of https://github.com/vllm-project/vllm/pull/53694, which removed the DP sync for MTP/EAGLE draft prefill. This PR removes the remaining DP syncs for the rest of the speculators: MTP, EAGLE, DFlash 1 & 2, DSpark. In the MTP/EAGLE case, it removes the DP sync before the multi-step-decode loop to draft the last N-1 tokens. In the DFlash/DSpark case, it removes the DP sync before drafting the B x N tokens in a single model fo …[truncated]

### L3-8bf39632b8  (L3, 2026-09-03, sha 8bf39632b86d, PR #54845)
TITLE: [ROCm][Perf] Add low-M FP32 router GEMM for gfx950 (#54845)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test-amd.yaml (+4/-1); tests/kernels/test_gate_linear_rocm_dispatch.py (+82/-10); tests/kernels/test_rocm_fp32_router_gemm.py (+228/-0); vllm/model_executor/layers/fused_moe/router/gate_linear.py (+37/-7); vllm/model_executor/layers/fused_moe/router/rocm_fp32_router_gemm.py (+148/-0)
LABELS: rocm, ready, ci/build
DEEP_STUDY: deep-study performance PR ()
BODY: # Summary ⏎  ⏎ Add an in-tree Triton GEMM for low-token ROCm gfx950 MoE routers that retain FP32 weights, FP32 accumulation, and FP32 output. ⏎  ⏎ The kernel accepts BF16 or FP32 activations, M=0..32, contiguous tensors, and the tuned (hidden size, expert count) pairs (3072, 256), (4096, 8), (4096, 192), (6144, 128), and (6144, 256). GateLinear stages non-contiguous low-M activations into contiguous storage before dispatch. Unsupported devices, shapes, d …[truncated]

### L3-ee0a4c46ae  (L3, 2026-09-03, sha ee0a4c46ae77, PR #55111)
TITLE: [Bugfix] Account for PCP in multi-node world size validation (#55111)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/engine/test_arg_utils.py (+19/-0); vllm/engine/arg_utils.py (+6/-8)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ `EngineArgs.create_engine_config` recomputes the world size for the `--nnodes` divisibility check, but omits `prefill_context_parallel_size`: ⏎  ⏎ ```python ⏎ world_size = ( ⏎     self.data_parallel_size ⏎     * self.pipeline_parallel_size ⏎     * self.tensor_parallel_size ⏎ ) ⏎ ``` ⏎  ⏎ `ParallelConfig.world_size` already multiplies in PCP, so this local recomputation disagrees with the authoritative value. ⏎  ⏎ With `TP=1 / PP=1 / PCP=2` across 2 nodes, th …[truncated]

### L3-cee0f92c02  (L3, 2026-09-03, sha cee0f92c0211, PR #54682)
TITLE: [ROCm][Perf] Optimize MiniMax-M3 decode indexer and top-k (#54682)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/attention/test_minimax_m3.py (+840/-0); vllm/models/minimax_m3/amd/model.py (+46/-1); vllm/models/minimax_m3/amd/ops/index_topk.py (+986/-306); vllm/models/minimax_m3/amd/ops/sparse_pa.py (+60/-19); vllm/models/minimax_m3/amd/sparse_attention_msa.py (+7/-0); vllm/models/minimax_m3/common/indexer.py (+71/-1)
LABELS: rocm, ready, minimax
ISSUES: #54681 [Performance][ROCm] Optimize MiniMax-M3 fresh decode indexer on gfx950
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ Resolve #54681 by optimizing MiniMax-M3's fresh-per-layer ROCm decode indexer. ⏎ Performance evidence remains from the measured gfx950 TP4 BF16 path, while ⏎ kernel dispatch is independent of tensor-parallel world size and decode query ⏎ length. ⏎  ⏎ This PR: ⏎  ⏎ - maps score work according to each request's real block count and shares an ⏎   index-K tile across the configured decode-query tile; ⏎ - fuses decode top-k selection, page-16 sparse-table c …[truncated]

### L3-b7624069ff  (L3, 2026-09-03, sha b7624069ff02, PR #51415)
TITLE: [Fusion] Manual `ActivationQuantFusionPass` initial application (#51415)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/compile/fusions_e2e/common.py (+39/-0); tests/compile/fusions_e2e/test_tp1_quant.py (+10/-5); tests/compile/fusions_e2e/test_tp2_ar_rms.py (+9/-6); tests/compile/fusions_e2e/test_tp2_async_tp.py (+10/-6); tests/compile/passes/test_silu_mul_quant_manual_fusion.py (+348/-0); tests/fusion/test_quant_activation_contract.py (+4/-0); tests/kernels/test_fused_quant_activation.py (+147/-0); vllm/model_executor/kernels/linear/scaled_mm/pytorch.py (+10/-1); vllm/model_executor/layers/fusion/fused_act_quant.py (+161/-0); vllm/model_executor/models/llama.py (+2/-1)
LABELS: ready, torch.compile, llama, quantization
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Starts the `ActivationQuantFusionPass` manual-fusion migration (RFC #43224, specific tracker https://github.com/vllm-project/vllm/issues/43501) on the producer side of the `QuantizedActivation` contract (#44260).  ⏎  ⏎ Adds `maybe_fused_act_quant`: given an activation and the linear it feeds, it emits a `QuantizedActivation` via the fused `silu_and_mul_quant` kernel when the linear advertises a consumable `input_quant_key`, and falls back to the pl …[truncated]

### L3-9509fc8ae6  (L3, 2026-09-03, sha 9509fc8ae61a, PR #54896)
TITLE: [Perf][Kimi-K3] Cut MLA decode concat/cache epilogue latency (#54896)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu (+98/-36); tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py (+23/-0)
LABELS: ready, kimi, k3
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Reduce the latency of the fused Kimi-K3 MLA decode q-concat + latent cache insert kernel, which sits between the absorbed-q BMM and the decode FMHA on every full-attention layer. ⏎  ⏎ Three changes, all inside `fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu`: ⏎  ⏎ 1. **Warps-per-row split, chosen from the token count.** At decode sizes the kernel had one warp per (token, head+1) row — 9 warps at concurrency 1 — with each lane serially moving …[truncated]

### L3-bc2ee48073  (L3, 2026-09-03, sha bc2ee480738d, PR #55020)
TITLE: [Perf] Prefetch the weight before the PDL wait in fused_q_kv_rmsnorm (#55020)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/common/ops/fused_qk_rmsnorm.py (+12/-5)
LABELS: ready
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Two latency cuts to the fused q/kv RMSNorm triton kernel (used by the Kimi-K3 and DeepSeek-V4 attention front-ends), which runs PDL-launched behind the qkv-a projection GEMM: ⏎  ⏎ 1. The gamma load does not depend on the producer's output, so it now issues **before** `gdc_wait`: the weights stream in while the producing GEMM finishes, leaving one dependent global round trip (the activation row) after the wait. ⏎ 2. `num_warps=8` for rows >= …[truncated]

### L3-579aef4e8d  (L3, 2026-09-03, sha 579aef4e8dfa, PR #52826)
TITLE: [ROCm] Bump AITER to 0.1.21.post1 (#52826)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.rocm_base (+1/-1); tests/kernels/quantization/test_rocm_mxfp4.py (+10/-1); vllm/model_executor/layers/fused_moe/fused_flydsl_moe.py (+117/-138)
LABELS: rocm, ready, ci/build
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-d410fc12f3  (L3, 2026-09-03, sha d410fc12f30b, PR #54606)
TITLE: [Kernel] Enable Kimi-K3 SiTU on the CuteDSL MoE backend and the SM107 low-latency GEMM plan (#54606)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/experts/flashinfer_cutedsl_moe.py (+19/-1); vllm/models/kimi_k3/nvidia/low_latency_gemm.py (+11/-3)
LABELS: ready, nvidia, kimi, k3
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Summary ⏎  ⏎ Two enablement changes for Kimi-K3, ported from a downstream fork where they were validated end-to-end on SM107 (Rubin) hardware: ⏎  ⏎ 1. **SiTU on the FlashInfer CuteDSL NVFP4 MoE backend** (`flashinfer_cutedsl_moe.py`): accept `MoEActivation.SITU` and plumb `situ_beta` / `situ_linear_beta` from `moe_config` into `flashinfer_cute_dsl_fused_moe_nvfp4`. The cute_dsl kernel keys SiTU on `situ_beta` and requires the base `Swiglu` activation  …[truncated]

### L3-6fdee17f84  (L3, 2026-09-03, sha 6fdee17f849b, PR #50314)
TITLE: [CI] Zen5 image build (#50314)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: docker/Dockerfile.cpu (+0/-17); docker/Dockerfile.zen (+98/-0)
LABELS: ready, ci/build, cpu
BODY: ## Purpose ⏎ - Extract the Zen CPU (zentorch) image build from Dockerfile.cpu into a dedicated Dockerfile.zen that layers on top of the existing CPU base image, giving the Zen build its own ⏎   Dockerfile, build script, and CI step ⏎ - The new Dockerfile supports two targets: vllm-openai-zen for serving and vllm-zen-test for CI, with optional zentorch version pinning via ZENTORCH_VERSION build arg ⏎ - Bump torch pin in the Zen test image to 2.13.0 ⏎  …[truncated]

### L3-c21751c90b  (L3, 2026-09-03, sha c21751c90b1b, PR #54251)
TITLE: [Kernel] Warm up Qwen GDN gated RMSNorm (#54251)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/third_party/flash_linear_attention/ops/layernorm_guard.py (+270/-25); tests/kernels/test_fla_layernorm_guard.py (+56/-0); vllm/model_executor/warmup/qwen_triton_warmup.py (+35/-1)
LABELS: ready, qwen
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ Refs #49349. ⏎  ⏎ Qwen3.5's GDN output projection calls the FLA `layer_norm_fwd_kernel` through ⏎ `RMSNormGated`. With cold Triton/vLLM caches, the first real request compiled ⏎ this specialization after the JIT monitor became active: ⏎  ⏎ ```text ⏎ ACTIVATION='silu', BLOCK_N=128, HAS_BIAS=False, HAS_Z=True, ⏎ IS_RMS_NORM=True, N=128, NORM_BEFORE_GATE=True, ROWS_PER_BLOCK=4, ⏎ X/Y/W/Z='*bf16' ⏎ ``` ⏎  ⏎ This PR moves that compilation into startup warmup by: ⏎  …[truncated]

### L3-560ef78bfe  (L3, 2026-09-03, sha 560ef78bfe73, PR #54901)
TITLE: [Perf][Model Runner V2] Compact sampling masks on GPU instead of unpacking the full-vocab bitmask on CPU (#54901)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/entrypoints/scale_out/token_in_token_out/test_serving_tokens.py (+7/-1); tests/test_config.py (+16/-0); tests/v1/core/test_scheduler.py (+47/-0); tests/v1/test_outputs.py (+31/-37); vllm/config/vllm.py (+7/-0); vllm/v1/engine/output_processor.py (+3/-2); vllm/v1/outputs.py (+12/-27); vllm/v1/worker/gpu/async_utils.py (+1/-1); vllm/v1/worker/gpu/sample/output.py (+60/-38); vllm/v1/worker/gpu/sample/sampler.py (+3/-1)
LABELS: ready, mrv2, scheduler
DEEP_STUDY: deep-study performance PR (system_performance)
BODY: ## Purpose ⏎  ⏎ `--return-sampling-mask` (#49577) made decode throughput collapse under load. PrimeIntellect reported ~2x RL step time and higher engine imbalance after integrating it (PrimeIntellect-ai/prime-rl#3431). ⏎  ⏎ Root cause: `SamplingMaskTensors.tolists()` ran `np.unpackbits` + `np.nonzero` over a `[num_reqs, vocab]` bitmask on every step. That is an O(num_reqs × vocab) CPU pass in the worker's async output thread, independent of how many toke …[truncated]

### L3-fc8f10792c  (L3, 2026-09-03, sha fc8f10792c59, PR #45091)
TITLE: Fix DeepSeek V4 FlashMLA auto KV cache dtype (#45091)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/attention.py (+9/-4)
LABELS: deepseek, verified, DSv4
BODY: ## Purpose ⏎  ⏎ DeepSeek V4 FlashMLA uses the `fp8_ds_mla` KV cache layout. When `--kv-cache-dtype` is left at the CLI default `auto`, the current DeepSeek V4 FlashMLA dtype resolver rejects it because `auto` does not start with `fp8`. ⏎  ⏎ This changes the FlashMLA layout path to treat `auto` as `fp8`, allowing the existing normalization to `fp8_ds_mla` to run. Explicit non-fp8 values still fail, but now with an actionable `ValueError` instead of an …[truncated]

### L3-2a336d8239  (L3, 2026-09-03, sha 2a336d8239d0, PR #54557)
TITLE: [warmup] overlap renderer warmup and engine core initialization (#54557)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/renderers/test_warmup.py (+346/-1); vllm/renderers/base.py (+117/-24); vllm/v1/engine/async_llm.py (+8/-0); vllm/v1/engine/core_client.py (+82/-8); vllm/v1/engine/llm_engine.py (+8/-0)
LABELS: ready
BODY: ## Purpose ⏎ This is a follow up pr for https://github.com/vllm-project/vllm/pull/54023. ⏎  ⏎ There is a race condition introduced in previous implementation https://github.com/vllm-project/vllm/pull/52764. I have only tested locally for command cli mode and ignore the sdk mode. The race is in sdk mode, vllm is imported with `VLLM_WORKER_MULTIPROC_METHOD` setting to default value `fork`. `fork` has an already known problem in a multithreaded process …[truncated]

### L3-69cf055936  (L3, 2026-09-03, sha 69cf05593606, PR #55061)
TITLE: [Performance][DSv4] Size dequant gather launch grid by rows (#55061)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py (+6/-1)
LABELS: ready, deepseek, quantization, DSv4
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ `dequantize_and_gather_k_cache_cutedsl` launched `grid = (num_reqs, 1024)`. The SWA call gathers 128 rows per request and the compressed call ~1k rows, so most of the 32k CTAs did no work; measured effective bandwidth was 0.35–2.4 TB/s on GB200 (8 TB/s HBM). `grid.y` is now picked per call from the launch shape. ⏎  ⏎ ## Where this kernel is hot: batch-invariant (deterministic) serving ⏎  ⏎ The kernel came in with vllm-project/vllm#42236 …[truncated]

### L3-8a728663c1  (L3, 2026-09-03, sha 8a728663c1c3, PR #54886)
TITLE: [Bugfix] Reject tokenizer-less Qwen VL processor init (#54886)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/test_qwen3_omni.py (+1/-0); vllm/model_executor/models/terratorch.py (+3/-0); vllm/multimodal/processing/processor.py (+10/-0)
LABELS: bug, ready, multi-modality, qwen
ISSUES: #54670 [Bug]: multimodal convert plus skip-tokenizer-init breaks processor construction
BODY: ## Purpose ⏎  ⏎ Fixes #54670. ⏎  ⏎ Qwen2-VL and Qwen2.5-VL Hugging Face processors require a tokenizer during construction. With `--skip-tokenizer-init`, processor construction instead failed later with an unclear `NoneType.convert_tokens_to_ids` error. ⏎  ⏎ This change validates that requirement before constructing either processor and raises an actionable `ValueError` asking the user to disable `--skip-tokenizer-init`. ⏎  ⏎ I initially considered putting this  …[truncated]

### L3-7dc30f5a66  (L3, 2026-09-03, sha 7dc30f5a663d, PR #55014)
TITLE: [ROCm][CI] Build and publish TheRock nightly docker images (#55014)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/release-pipeline.yaml (+138/-0); .buildkite/scripts/push-nightly-builds-rocm.sh (+81/-34)
LABELS: rocm, ready, ci/build
BODY: ## Purpose ⏎  ⏎ `docker/Dockerfile.rock_base` and `docker/Dockerfile.rock` were added in #49925 but nothing references them. This duplicates ROCm release-pipeline Jobs 1 and 6 (plus the nightly publish) for them, producing on `NIGHTLY=1`, in the existing `vllm/vllm-openai-rocm` repo: ⏎  ⏎ - `vllm/vllm-openai-rocm:base-nightly-rocm714` / `-<commit>` — `Dockerfile.rock_base` ⏎ - `vllm/vllm-openai-rocm:nightly-rocm714` / `-<commit>` — `Dockerfile.rock --targe …[truncated]

### L3-a8693df504  (L3, 2026-09-04, sha a8693df50440, PR #55245)
TITLE: [SpecDecode]Fix spec decode warmup device selection (#55245)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/warmup/spec_decode_rejection_warmup.py (+1/-1)
LABELS: intel-gpu
BODY: ## Purpose ⏎ Fix the speculative decoding rejection sampler warmup on XPU. ⏎  ⏎ The warmup previously hardcoded `torch.device("cuda")`, causing XPU workers to ⏎ attempt CUDA tensor allocations and skip the rejection sampler warmup. Use the ⏎ worker's initialized device instead, preserving both the accelerator type and ⏎ device index. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-19c018ec05  (L3, 2026-09-04, sha 19c018ec05a3, PR #54908)
TITLE: [Bugfix][DCP] Materialize prefill keys on non-owner ranks (#54908)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/test_fused_deepseek_v32_norm_rope.py (+11/-1); vllm/models/deepseek_v32/common/kernels.py (+9/-3)
LABELS: bug, ready, deepseek
ISSUES: #54907 [Bug][DCP] GLM-5.3 dense prefill consumes uninitialized K rows on non-owner ranks
BODY: ## Purpose ⏎  ⏎ Fixes #54907 and follows the GLM-5.3 report in #50095. ⏎  ⏎ NVIDIA DeepSeek-V3.2 / GLM-5.3 dense prefill consumes normalized/rotated K outputs on every DCP rank. The fused kernel treated a negative local cache slot as an unconditional early return, but under DCP it often means that another rank owns the cache row. Non-owner ranks therefore left `kv_c_out` and `k_pe_out` uninitialized. ⏎  ⏎ This patch separates materialization from cache owner …[truncated]

### L3-156050598e  (L3, 2026-09-04, sha 156050598eea, PR #55246)
TITLE: [ROCm][CI] Bump ROCk base to ROCm 10.0 (#55246)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/release-pipeline.yaml (+3/-3); docker/Dockerfile.rock_base (+23/-50)
LABELS: rocm, ready, ci/build
BODY: ## Purpose ⏎  ⏎ Bump the TheRock base (`docker/Dockerfile.rock_base`) from ROCm 7.14.0 to 10.0.0 and switch the ROCk nightly tags `rocm714` → `rocm100`. Based on @rasmith's `ROCm:vllm:rock10` dockerfile; follows #55014. ⏎  ⏎ Scoped for a first landing: ⏎  ⏎ - **Triton:** prebuilt `3.8.0+git4cff872c.rocm10.0.0` wheel (exactly torch's Requires-Dist). Source rebuild from `release/internal/3.8.x` dropped for now — follow-up. ⏎ - **AOTriton:** version/kernel-image  …[truncated]

### L3-8f816a3f66  (L3, 2026-09-04, sha 8f816a3f6654, PR #52358)
TITLE: [MRV2][Metrics] Support `CUDAGraphStat` in MRV2 (#52358)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_model_runner_v2_eplb.py (+1/-0); vllm/v1/worker/gpu/cudagraph_utils.py (+12/-1); vllm/v1/worker/gpu/model_runner.py (+9/-0)
LABELS: ready, nvidia, mrv2
BODY: ## Purpose ⏎ Propagate CUDAGraphStat through ExecuteModelState so generation outputs expose the actual CUDA graph mode and padding counts. Exclude dummy runs from user-visible metrics. ⏎  ⏎ ## Test Plan ⏎ None. ⏎  ⏎ ## Test Result ⏎ None. ⏎  ⏎ --- ⏎ [details omitted]

### L3-f19431e5c8  (L3, 2026-09-04, sha f19431e5c863, PR #55069)
TITLE: [Bugfix][MoE] Allow TRTLLM FP8 block-scale MoE with SwiGLU clamp (#55069)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_flashinfer.py (+70/-0); vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py (+14/-6)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎  ⏎ `TrtLlmFp8ExpertsBase.is_supported_config` rejects any model that requests a SwiGLU clamp (`swiglu_limit` / `swiglu_alpha` / `swiglu_beta`) unless the weights are MXFP8. That was correct when the guard was added in #54160: FlashInfer's FP8 block-scale MoE raised ⏎  ⏎ ``` ⏎ ValueError: gemm1_alpha, gemm1_beta, and gemm1_clamp_limit are only supported ⏎ for Fp8QuantizationType.MxFp8 in FP8 block scale MoE. ⏎ ``` ⏎  ⏎ FlashInfer 0.6.18 — the version p …[truncated]

### L3-1ff5edb023  (L3, 2026-09-04, sha 1ff5edb02327, PR #50220)
TITLE: [Bug-fix] Fix MoE fused sum row offsets (#50220)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py (+1/-1)
LABELS: bug, ready
ISSUES: #47281 [Bug]: Humming backend hits CUDA illegal memory access on Qwen3.5-397B-A17B-GPTQ-Int4 startup, while Marlin works
BODY: Resolves https://github.com/vllm-project/vllm/issues/47281 ⏎  ⏎ `moe_fused_mul_sum` indexes `inputs` by: ⏎  ⏎ ```python ⏎ inputs[offs_m, :, offs_k] ⏎ ``` ⏎  ⏎ where the row stride is: ⏎  ⏎ ```text ⏎ stride_m = top_k * hidden_size ⏎ ``` ⏎  ⏎ The old kernel kept `offs_m` as int32, so `offs_m * stride_m` can overflow before pointer arithmetic. ⏎  ⏎ This only reproduces on the larger Qwen3.5 shape because the max row offset crosses int32: ⏎  ⏎ ```text ⏎ INT32_MAX = 2,1 …[truncated]

### L3-3ff4f02dfe  (L3, 2026-09-04, sha 3ff4f02dfe69, PR #52494)
TITLE: [AMD][kimik3][ROCm][Perf] Fuse MLA q/kv RMSNorm in AMD Kimi-K3 MLA wrapper (#52494)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/kimi_k3/amd/linear.py (+3/-2); vllm/models/kimi_k3/amd/mla.py (+120/-0)
LABELS: rocm, ready, verified, kimi, k3
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ Fuse the MLA layer's `q_a_layernorm` / `kv_a_layernorm` pair into a single ⏎ AITER `fused_qk_rmsnorm` launch for eager Kimi-K3 on ROCm, without putting ⏎ vendor checks in shared `vllm/model_executor/layers/mla.py`. ⏎  ⏎ Kimi-K3 (no `@support_torch_compile`) never hits `MLADualRMSNormFusionPass`, ⏎ so the two RMSNorms stay as separate AITER launches. NVIDIA already owns a ⏎ vendor-local MLA under `vllm/models/kimi_k3/nvidia/`; this mirrors …[truncated]

### L3-8cd95f7de7  (L3, 2026-09-04, sha 8cd95f7de70d, PR #55234)
TITLE: [Bugfix][MLA] Restore DSpark cache-group capability under optimized Python (#55234)
SOURCES: subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/attention/test_mla_noncausal.py (+19/-7); tests/v1/core/test_kv_cache_utils.py (+25/-0); vllm/v1/kv_cache_interface.py (+9/-10)
LABELS: bug, dflash, kv-cache-manager
ISSUES: #54649 [Bug]: Kimi-K3 DSpark/DCP illegal memory access
BODY: ## Summary ⏎  ⏎ - restore `non_causal_multi_token_decode` as an MLA cache-group capability, promoted when any group member needs the non-causal draft path; ⏎ - keep the inherited KV-cache merge incompatibility check active under `python -O` while preserving the `AssertionError` contract used by grouping callers; ⏎ - extend existing tests to cover a causal target and non-causal draft sharing one capability-enabled builder, plus the optimized-Python fallba …[truncated]

### L3-a26b71d868  (L3, 2026-09-04, sha a26b71d86810, PR #55126)
TITLE: [Bugfix][PD] Pad resumed speculative decode requests (#55126)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/v1/core/test_scheduler.py (+16/-0); vllm/v1/core/sched/scheduler.py (+2/-1)
LABELS: bug, ready, kv-connector, scheduler
BODY: ## Purpose ⏎  ⏎ Keep resumed speculative-decode requests on the uniform verifier shape when a data-parallel rank has no already-running requests. ⏎  ⏎ In P/D serving, a synchronous KV connector can resolve a waiting consumer to all but the final prompt token. During that scheduling pass, the matched prefix exists in the local `num_computed_tokens` variable, while `request.num_computed_tokens` is updated only after admission. The current padding condition …[truncated]

### L3-8a0a7ee40a  (L3, 2026-09-04, sha 8a0a7ee40a59, PR #47941)
TITLE: [EC Connector] P2P NIXL + CPU EC Connector (#47941)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py (+16/-6); tests/v1/ec_connector/unit/cpu/scheduler/test_embedding_cache.py (+54/-0); tests/v1/ec_connector/unit/cpu/test_connector.py (+27/-3); tests/v1/ec_connector/unit/cpu/worker/test_worker.py (+19/-0); tests/v1/ec_connector/unit/test_control.py (+285/-0); tests/v1/ec_connector/unit/test_data.py (+184/-0); tests/v1/ec_connector/unit/test_ec_transfer_params.py (+12/-1); tests/v1/ec_connector/unit/test_protocol.py (+41/-0); tests/v1/ec_connector/unit/test_scheduler_nixl_consumer.py (+350/-0); tests/v1/ec_connector/unit/test_scheduler_nixl_ctor.py (+78/-0); (+18 more)
LABELS: documentation, structured-output, frontend, speculative-decoding, ready, ci/build, v1, cpu, kv-connector, mrv2
BODY: ## Purpose ⏎ This PR extends ECCPUConnector to support P2P EC sharing based on NIXL. [The original PR](https://github.com/vllm-project/vllm/pull/42998) was split into two parts; [The first part](https://github.com/vllm-project/vllm/pull/47423) is a standalone CPU-based offloading EC connector, and this PR is an extension of it, allowing CPU-offloaded EC cache to be shared between vLLM instances via NIXL. ⏎ The control plane is ZMQ (consumer DEALER  …[truncated]

### L3-5690b02c03  (L3, 2026-09-04, sha 5690b02c0383, PR #51392)
TITLE: [Quantization] Support online quantization with partially pre-quantized checkpoints (#51392)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/attention.py (+8/-3); vllm/model_executor/layers/attention/mla_attention.py (+5/-2); docs/features/quantization/online.md (+18/-0); tests/model_executor/test_eagle_quantization.py (+1/-0); tests/quantization/test_config_utils.py (+1/-1); tests/quantization/test_online.py (+514/-54); tests/quantization/test_online_mxfp4.py (+1/-1); tests/quantization/test_quantization_config_args.py (+11/-8); vllm/config/quantization.py (+19/-7); vllm/model_executor/layers/fused_moe/routed_experts.py (+2/-1); (+15 more)
LABELS: documentation, speculative-decoding, ready, quantization, kimi, k3
BODY: ## Disclosure ⏎  ⏎ AI assistance was used. The changes were reviewed and tested manually. ⏎  ⏎ ## Purpose ⏎  ⏎ Allow online quantization on the unquantized `linear` or `moe` (or fine-grained `targets`) portions of a partially pre-quantized checkpoint **from ANY `quant_method` (Quark, ModelOpt, compressed-tensors, etc.)**.  ⏎  ⏎ The original `quant_method` remains responsible for its pre-quantized layers, the composed config applies the requested online m …[truncated]

### L3-2524051385  (L3, 2026-09-04, sha 25240513856e, PR #54826)
TITLE: [Bugfix][Spec Decode] Honour the draft's attention_backend on Model Runner V2 (#54826)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/spec_decode/eagle/utils.py (+10/-0); tests/v1/spec_decode/test_draft_attention_backend_override.py (+90/-0)
LABELS: bug, speculative-decoding, ready, mrv2
BODY: ## Purpose ⏎  ⏎ `--speculative-config '{"attention_backend": ...}'` is silently ignored on Model Runner V2. ⏎  ⏎ The three draft overrides have deliberately different contracts, per `SpeculativeConfig`: ⏎  ⏎ | field | when unset | ⏎ |---|---| ⏎ | `moe_backend` | inherits the target's | ⏎ | `kv_cache_dtype` | inherits the target's | ⏎ | **`attention_backend`** | **cleared, so the draft autoselects independently** | ⏎  ⏎ The V1 proposer implements this in `_create_draft_v …[truncated]

### L3-3f41d102c5  (L3, 2026-09-04, sha 3f41d102c5b8, PR #50945)
TITLE: [1/2][Model Runner V2] DBO support, eager mode (#50945)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/test_routed_experts_capture.py (+1/-0); tests/v1/worker/test_gpu_ubatch_slicing.py (+583/-0); vllm/config/parallel.py (+11/-0); vllm/config/vllm.py (+43/-2); vllm/model_executor/models/diffusion_gemma.py (+2/-0); vllm/v1/worker/gpu/attn_utils.py (+15/-8); vllm/v1/worker/gpu/cudagraph_utils.py (+8/-0); vllm/v1/worker/gpu/dp_utils.py (+55/-5); vllm/v1/worker/gpu/model_runner.py (+41/-3); vllm/v1/worker/gpu/model_states/default.py (+2/-0); (+7 more)
LABELS: nvidia, mrv2, verified
BODY: DBO for Model Runner V2 (RFC #50738) is two PRs: ⏎ -> #50945 [1/2][Model Runner V2] DBO support, eager mode (P0–P2) ⏎    #51700 [2/2][Model Runner V2] FULL CUDA graph capture for microbatched steps (P3–P4) ⏎  ⏎ ## Purpose ⏎ Relate to #50738. This PR finishes stage P0, P1, P2 ⏎ ## Test Plan ⏎ Benchmark results are shown in the RFC ⏎  ⏎ smoke.sh ⏎ ```bash ⏎ #!/usr/bin/env bash ⏎ # Baseline (no DBO) V2 eager server. Must stay config-identical to smoke_dbo.sh ⏎ # …[truncated]

### L3-a85d0738da  (L3, 2026-09-04, sha a85d0738da31, PR #55288)
TITLE: [Bugfix] Fix double BOS in LLM.chat() for multimodal models (#55288)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/entrypoints/multimodal/llm/test_mm_processor_kwargs.py (+130/-0); vllm/entrypoints/offline_utils.py (+10/-3)
LABELS: bug, frontend, ready
ISSUES: #55197 [Bug]:  Offline LLM.chat() can add BOS twice for multimodal models
BODY: ## Purpose ⏎  ⏎ Fixes #55197. ⏎  ⏎ `LLM.chat()` renders the chat template to text and tokenizes it with `renderer.default_chat_tok_params`. For multimodal models that returns the processor default `add_special_tokens=True` (right for raw prompts in `LLM.generate()`), so every template that emits `bos_token` gets a second BOS. The online chat API does not have this problem because `ChatCompletionRequest.add_special_tokens` defaults to `False`, so the same …[truncated]

### L3-6a039f465e  (L3, 2026-09-04, sha 6a039f465e37, PR #51826)
TITLE: [feat] add torchcodec as audio loader and implement selective audio backend (#51826)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/features/multimodal_inputs.md (+32/-0); tests/entrypoints/speech_to_text/transcription/test_transcription_validation_whisper.py (+1/-1); tests/multimodal/media/test_audio.py (+100/-1); vllm/multimodal/media/audio.py (+313/-65)
LABELS: documentation, ready, multi-modality
BODY: ## Purpose ⏎ #51354  ⏎ ### Audio Decoding Backend ⏎  ⏎ vLLM decodes audio bytes into waveforms using a selectable decoding backend: ⏎  ⏎ | Backend | Description | ⏎ | --- | --- | ⏎ | `auto` (default) | torchcodec, falling back to soundfile, then PyAV | ⏎ | `soundfile` | libsndfile only, no fallback | ⏎ | `pyav` | PyAV (FFmpeg) only, no fallback | ⏎ | `torchcodec` | TorchCodec (PyTorch-native) only, no fallback | ⏎  ⏎ Select the backend per server via `--media …[truncated]

### L3-5093e4844a  (L3, 2026-09-04, sha 5093e4844a75, PR #54374)
TITLE: [Bugfix][Spec Decode] Drop FlashAttention's AOT schedule for a sliding-window DFlash drafter (#54374)
SOURCES: path_integration+keyword, subject_keyword, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/worker/gpu/spec_decode/dflash/speculator.py (+17/-1)
LABELS: bug, speculative-decoding, mrv2, verified, dflash
BODY: ### Purpose ⏎  ⏎ A DFlash drafter runs sliding-window attention. FlashAttention's AOT split ⏎ schedule is computed for one window configuration shared by every layer, and ⏎ whether the drafter gets a correct one today is decided by accident. ⏎  ⏎ `_get_sliding_window_configs` scans *every* `Attention` layer in the model and ⏎ skips those that are not `FlashAttentionImpl`. Exactly one distinct config ⏎ leaves `aot_schedule` on; more than one turns it off. So: ⏎  ⏎ -  …[truncated]

### L3-8277c42e4c  (L3, 2026-09-04, sha 8277c42e4c74, PR #55202)
TITLE: [Perf] Ensure async h2d copies are pinned in more places (#55202)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.flashinfer.v1_backend, L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.rocm.aiter_fa, L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/mla_attention.py (+8/-2); vllm/utils/flashinfer.py (+26/-0); vllm/v1/attention/backends/flashinfer.py (+34/-1); vllm/v1/attention/backends/hpc_attn.py (+8/-2); vllm/v1/attention/backends/rocm_aiter_fa.py (+10/-5); tests/kernels/attention/test_flashinfer.py (+2/-0); vllm/distributed/eplb/eplb_state.py (+10/-1); vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py (+6/-4); vllm/model_executor/layers/fused_moe/modular_kernel.py (+5/-1); vllm/model_executor/models/cosmos3_edge.py (+2/-4); (+11 more)
LABELS: rocm, ready, qwen, kv-connector, nvidia, glm
DEEP_STUDY: deep-study performance PR ()
BODY: Follow-on from https://github.com/vllm-project/vllm/pull/54660. ⏎  ⏎ Address more places where async h2d copies are made from paged memory CPU tensors, which could cause a stall harming cpu/gpu overlap, which were exposed via https://github.com/vllm-project/vllm/pull/53491.

### L3-701a744490  (L3, 2026-09-04, sha 701a7444909d, PR #55136)
TITLE: [CI] Raise AMD Spec Decode Eagle 1 job timeout to 35min (#55136)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/spec_decode.yaml (+1/-1)
LABELS: rocm, ready, ci/build
BODY: ## Purpose ⏎  ⏎ The `:amd: (MI300) Spec Decode Eagle 1: DeepSeek + Qwen` job (25min limit) consistently takes ~25.5-26min and gets SIGTERM'd mid-way through its 6th test (`test_eagle_correctness_medium[ROCM_AITER_FA-qwen3_eagle3-transformers]`, killed at ~91% of its GSM8K eval; the first 5 tests pass with 96-100/100). Observed on two consecutive attempts of build 87038 (25.8/25.4 min). ⏎  ⏎ One contributor: ROCm 7.2.3's clang rejects aiter's `-amdgpu-coe …[truncated]

### L3-6cbb3c154e  (L3, 2026-09-04, sha 6cbb3c154ef1, PR #55404)
TITLE: [Perf][GDN] Build cudagraph-capture metadata without a device sync (#55404)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/kimi_k3/test_kda_metadata.py (+62/-0); vllm/v1/attention/backends/gdn_attn.py (+2/-1); vllm/v1/attention/backends/short_conv_attn.py (+3/-1)
LABELS: ready, nvidia, kimi, k3
DEEP_STUDY: deep-study performance PR ()
BODY: `build_for_cudagraph_capture` derived the host-side draft counts with `(num_accepted_tokens - 1).cpu()`, a device-to-host copy that drains the whole stream before the metadata can be built. The method is not capture-only: a DP rank with nothing scheduled runs it on every dummy step, because `_dummy_run` re-stages FULL-graph metadata with `for_capture=True`. ⏎  ⏎ Derive `num_decode_draft_tokens_cpu` from `query_start_loc_cpu` instead in the GDN and  …[truncated]

### L3-c81ace1859  (L3, 2026-09-04, sha c81ace18596f, PR #55331)
TITLE: [Perf] Read VidCom2 frame budgets once (#55331)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/multimodal/test_vidcom2.py (+21/-0); vllm/multimodal/video_prune/vidcom2.py (+3/-5)
LABELS: ready, multi-modality
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ VidCom2 currently reads each dynamic per-frame Top-K budget with ⏎ `ks[i].item()`, then reads the selected-token count with another scalar ⏎ reduction. For `T` frames this introduces `T + 1` host-visible scalar reads in ⏎ the token-selection path. ⏎  ⏎ This PR copies the small budget vector to a Python list once, preserves the ⏎ original one-dimensional unsorted `torch.topk` operation for every frame, and ⏎ reuses the budget sum during count reconc …[truncated]

### L3-685074cd61  (L3, 2026-09-04, sha 685074cd61d6, PR #55341)
TITLE: [Bugfix][V2] Warm up kernels before capturing CUDA graphs (#55341)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_model_runner_v2.py (+82/-0); tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py (+9/-8); tests/v1/worker/test_workspace.py (+24/-0); vllm/v1/worker/gpu/cudagraph_utils.py (+1/-1); vllm/v1/worker/gpu/model_runner.py (+7/-2); vllm/v1/worker/gpu_worker.py (+5/-4); vllm/v1/worker/mm_encoder_model_runner.py (+1/-1)
LABELS: bug, ready, nvidia, mrv2
ISSUES: #55336 [Bug]: Model Runner V2 never locks the workspace, so post-capture growth silently invalidates captured CUDA graphs
BODY: ## Purpose ⏎  ⏎ Fixes #55336. ⏎  ⏎ `warmup_kernels` runs a scheduler-realistic prefill plus a decode step at ⏎ `max_num_seqs`. On the V2 path it is called *after* `capture_model()`, so those ⏎ shapes reach the shared workspace arena only once CUDA graphs already hold ⏎ pointers into it. `WorkspaceManager._ensure_workspace_size` grows the arena by ⏎ replacing the tensor, which frees the buffer the captured graphs baked in, and ⏎ the next replay writes into freed me …[truncated]

### L3-874df9373d  (L3, 2026-09-04, sha 874df9373dab, PR #55178)
TITLE: [Bugfix] Preserve Mamba state for padded prompt tails (#55178)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mamba2_attn.py (+1/-0); vllm/v1/attention/backends/mamba_attn.py (+25/-4)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ Fix recurrent-state corruption in hybrid Mamba models when speculative decoding pads the last prompt token to a uniform `K + 1` query length. ⏎  ⏎ This is most visible in prefill/decode-disaggregated serving: ⏎  ⏎ 1. The prefill node transfers the Mamba recurrent state after prompt token `N - 1`. ⏎ 2. The decode node must process the final prompt token `N` to produce `h(N)`. ⏎ 3. If another speculative request is already running, the scheduler pa …[truncated]

### L3-4ee2595512  (L3, 2026-09-04, sha 4ee2595512c8, PR #54819)
TITLE: [Attention] Sync FA with upstream (#54819)
SOURCES: path_core, dependency_pin, release_notes, body_keyword
ARTIFACT_HINTS: L3.flash_attn.fork_build
FILES: cmake/external_projects/vllm_flash_attn.cmake (+1/-1)
LABELS: ready, ci/build
BODY: ## Purpose ⏎  ⏎ Sync vLLM's pinned FlashAttention revision with the landed [vllm-project/flash-attention#188](https://github.com/vllm-project/flash-attention/pull/188). ⏎  ⏎ #188 was approved and squash-merged as `506341a143fcabd4bb79052a7605ada727d6b3f5` (tree `b4c7f467f3874ca7b8fb7471abee0d363b5113ff`). The final tree is byte-identical to the reviewed #188 head `ee391c01f50d48a13d7e3d584f072bb0dba8dc4c`; the excluded `hopper/` subtree remains unchanged …[truncated]

### L3-8369affa54  (L3, 2026-09-05, sha 8369affa5428, PR #55119)
TITLE: [Feat] Add EPLB support for GLM-5.3-Flash (#55119)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/models/glm5next/nvidia/model.py (+68/-9); vllm/models/glm5next/nvidia/mtp.py (+1/-0)
LABELS: ready, nvidia, glm
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎ ``` ⏎ VLLM_ENGINE_READY_TIMEOUT_S=3600 vllm serve  /mnt/nvme/shared/models/ZhipuAI/GLM-5.3-Flash --tensor-parallel-size 4 --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 --load-format instanttensor  -ep --no-enable-flashinfer-autotune --enable-eplb --eplb-config '{"window_size":100,"step_interval":100,"num_redundant_experts":8,"use_async":true} ⏎  ⏎ ``` ⏎ ## Test Result ⏎ before ⏎ ``` ⏎ (EngineCore …[truncated]

### L3-32601ef7a1  (L3, 2026-09-05, sha 32601ef7a1ce, PR #55415)
TITLE: [Perf][Multimodal] Avoid duplicate text embedding in Qwen2.5-Omni (#55415)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/qwen2_5_omni_thinker.py (+5/-7)
LABELS: ready, multi-modality, qwen
DEEP_STUDY: deep-study performance PR ()
BODY: # [Perf][Multimodal] Avoid duplicate text embedding in Qwen2.5-Omni ⏎  ⏎ ## Purpose ⏎  ⏎ `Qwen2_5OmniThinkerForConditionalGeneration.embed_input_ids` embeds the text ⏎ input before it knows which multimodal merge path is required. For a regular ⏎ multimodal request, the method then calls the parent implementation, which ⏎ embeds the same text input again. For an audio-in-video request, the custom ⏎ interleaved path also embeds the text input again. The first emb …[truncated]

### L3-52bc900d93  (L3, 2026-09-05, sha 52bc900d930c, PR #53835)
TITLE: [Bugfix][Kernel] Build fused GDN MTP decode for SM110 (#53835)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.fork_inline_cmake
FILES: CMakeLists.txt (+1/-1)
LABELS: bug, ready, ci/build
ISSUES: #53462 [Bug]: GDN MTP fused decode kernel (fused_gdn_decode_post_conv_mtp) crashes with "no kernel image is available" on SM110a (Jetson Thor) — capability guard checks symbol presence, not cubin arch
BODY: ## Purpose ⏎  ⏎ Fixes #53462. ⏎  ⏎ CUDA 13 advertises SM110 as a supported vLLM build architecture, but ⏎ `FUSED_GDN_DECODE_ARCHS` skipped the SM11x family. In a multi-architecture ⏎ wheel, the fused GDN symbol is still registered because the source is built for ⏎ other architectures, so the runtime capability and symbol checks pass on ⏎ Jetson Thor. The first MTP post-conv launch then fails because the wheel has no ⏎ SM110 kernel image. ⏎  ⏎ Add the CUDA 13 `11.0f` f …[truncated]

### L3-bc96d76aa9  (L3, 2026-09-05, sha bc96d76aa994, PR #54110)
TITLE: [Kernel] Fall back from persistent top-k on low-shared-memory GPUs (#54110)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/topk.cu (+9/-10)
LABELS: verified
BODY: ## Purpose ⏎  ⏎ `persistent_topk` can require more cooperative CTAs than the device can keep ⏎ resident for a long row. The existing dispatcher falls back to `FilteredTopK` ⏎ in that case, but `FilteredTopK` requires at least 128 KiB of opt-in shared ⏎ memory. On GPUs below that limit, the dispatcher raises instead and terminates ⏎ EngineCore. ⏎  ⏎ This change routes only that oversubscribed, sub-128-KiB branch to vLLM's ⏎ existing `top_k_per_row_decode` implement …[truncated]

### L3-385ba6b5b0  (L3, 2026-09-05, sha 385ba6b5b003, PR #54770)
TITLE: [Bugfix][Quantization] Register Quark per-block FP8 scales as weight_scale (#54770)
SOURCES: path_core
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+7/-4); tests/quantization/test_quark.py (+63/-0); vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_fp8.py (+7/-2); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+15/-0); vllm/models/deepseek_v4/amd/model.py (+13/-13); vllm/models/deepseek_v4/amd/mtp.py (+4/-5); vllm/models/deepseek_v4/amd/rocm.py (+2/-1)
LABELS: bug, rocm, deepseek, quantization, verified, DSv4
BODY: ## Summary ⏎ - Register Quark per-block FP8 linear scales as `weight_scale`, matching what Quark exports, instead of `weight_scale_inv` (DeepSeek-V3 `Fp8LinearMethod` convention). ⏎ - Drop the DeepSeek-V4 AMD mapper rewrite of non-expert `.weight_scale` → `.weight_scale_inv`. Loaders already alias to `_inv` only when that parameter is registered (`Fp8LinearMethod` / MXFP4 override). ⏎ - Read block scales through `get_fp8_block_weight_scale()` so DSV4 R …[truncated]

### L3-16328c7a77  (L3, 2026-09-05, sha 16328c7a775f, PR #55242)
TITLE: [Perf] Kimi K3 nvfp4 Align in_proj weights by 128 to avoid elementwise copy (#55242)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/quantization/test_modelopt.py (+6/-2); vllm/model_executor/layers/quantization/modelopt.py (+21/-3); vllm/models/kimi_k3/nvidia/kda.py (+14/-8); vllm/models/kimi_k3/nvidia/mla.py (+1/-4); vllm/models/kimi_k3/nvidia/model.py (+1/-4); vllm/models/kimi_k3/nvidia/ops/cute_dsl/gemm_rs_ar.py (+10/-0)
LABELS: quantization, kimi, k3
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ This PR updates the following: ⏎ - Align Kimi K3 nvfp4 in_proj weights by 128 to avoid elementwise copy from `.contiguous()` to make the output compact ⏎ - Update modelopt Fp8_Pb_Wo to select linear method backend based on padded weight layout (instead of original weight), before this PR it will reject DeepGEMM and chooses Cutlass due to misalignment with DeepGEMM's requirement. ⏎ - Fix GEMM-AR fusion disabled warning to only warn once i …[truncated]

### L3-d87a440f88  (L3, 2026-09-05, sha d87a440f88e2, PR #50514)
TITLE: [Core][MRV2] Support eagle3 spec decode with pipeline parallel (#50514)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/test_qwen3_omni.py (+3/-3); tests/models/kimi_k3/test_aux_attn_res_stream.py (+14/-16); tests/v1/e2e/spec_decode/eagle/test_eagle3_pp.py (+59/-0); tests/v1/worker/test_eagle3_aux_hidden_states_pp.py (+26/-0); tests/v1/worker/test_spec_decode_embed_sharing_pp.py (+110/-0); vllm/config/speculative.py (+1/-1); vllm/config/vllm.py (+0/-6); vllm/model_executor/models/deepseek_eagle3.py (+1/-3); vllm/model_executor/models/interfaces.py (+58/-1); vllm/model_executor/models/laguna_dflash.py (+1/-3); (+17 more)
LABELS: speculative-decoding, ready, ci/build, llama, qwen, deepseek, cpu, nvidia, mrv2, DSv4
BODY: ## Summary ⏎  ⏎ This enables EAGLE3-style external draft models (`eagle3`, `dflash`, and ⏎ `dspark`) with pipeline parallelism for target models that explicitly opt in. ⏎  ⏎ The draft model runs only on the last pipeline stage, but its auxiliary hidden ⏎ states can come from layers on any stage. Each stage adds the states it produces ⏎ to the existing `IntermediateTensors` handoff using globally ordered slots. ⏎ Middle stages forward states received from …[truncated]

### L3-7fbd44cbe0  (L3, 2026-09-05, sha 7fbd44cbe0a9, PR #55299)
TITLE: [Bugfix][DSv4] Seed the -1 sentinel in the prefill sparse index workspace (#55299)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/models/deepseek_v4/common/ops/cache_utils.py (+1/-0)
LABELS: bug, ready, deepseek, DSv4
BODY: ## Purpose ⏎  ⏎ `combine_topk_swa_indices` writes only the valid prefix of each row and leaves ⏎ the rest to the `-1` sentinel that `topk_length` pairs with. The `out=None` path ⏎ gets that sentinel from `torch.full`; the caller-supplied path does not seed it, ⏎ so a workspace buffer reaches the sparse attention kernel with whatever was in ⏎ those columns before. ⏎  ⏎ On main this stays latent: the prefill chunk plan normally yields a single chunk, ⏎ so the same w …[truncated]

### L3-7985444339  (L3, 2026-09-05, sha 7985444339e2, PR #55059)
TITLE: [Kernel][HY V4] Add Triton iHC pre/post fallback (#55059)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: benchmarks/kernels/benchmark_hy_v4_ihc.py (+199/-0); vllm/models/hy_v4/nvidia/hc.py (+19/-3); vllm/models/hy_v4/nvidia/triton_ihc.py (+313/-0)
LABELS: performance, verified
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: Add in-tree Triton implementations for the HY V4 iHC pre and post operations. ⏎  ⏎ The runtime dispatch order is: ⏎  ⏎ 1. Use HPC-Ops when enabled and supported. ⏎ 2. Otherwise use the Triton implementation on NVIDIA CUDA with FP16/BF16 inputs. ⏎ 3. Fall back to the existing eager PyTorch implementation for unsupported configurations. ⏎  ⏎ The Triton implementation is adapted from [sglang#36805](https://github.com/sgl-project/sglang/pull/36805). ⏎  ⏎ ## Pu …[truncated]

### L3-a1541f5742  (L3, 2026-09-06, sha a1541f5742a2, PR #55285)
TITLE: [Perf] Use SDPA for BLIP-2 Q-Former attention (#55285)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/blip2.py (+8/-8)
LABELS: ready, multi-modality
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ BLIP-2 Q-Former attention currently materializes the attention scores, applies ⏎ scaling and softmax separately, and launches a second matrix multiplication: ⏎  ⏎ ```text ⏎ before: {matmul: 2, mul: 1, softmax: 1, dropout: 1} ⏎ after:  {scaled_dot_product_attention: 1} ⏎ ``` ⏎  ⏎ This PR uses standard `torch.nn.functional.scaled_dot_product_attention` for ⏎ both Q-Former self-attention and cross-attention. It preserves the existing ⏎ scale and training dr …[truncated]

### L3-1c344ed41e  (L3, 2026-09-06, sha 1c344ed41e1e, PR #49410)
TITLE: [CPU] [Feat]  Add native AMX-FP8 attention impl for Diamond Rapids (#49410)
SOURCES: release_notes
ARTIFACT_HINTS: L3.flash_attn.upstream_pip
FILES: .buildkite/scripts/hardware_ci/run-cpu-test.sh (+1/-1); benchmarks/kernels/cpu/benchmark_cpu_attn.py (+52/-6); cmake/cpu_extension.cmake (+29/-0); csrc/cpu/cpu_attn.cpp (+30/-2); csrc/cpu/cpu_attn_amx_fp8.hpp (+795/-0); csrc/cpu/cpu_attn_impl.hpp (+54/-36); csrc/cpu/cpu_types_x86.hpp (+58/-0); csrc/cpu/generate_cpu_attn_dispatch.py (+30/-9); docker/Dockerfile.cpu (+14/-6); tests/kernels/attention/test_cpu_attn.py (+148/-1)
LABELS: performance, ci/build, v1, cpu, verified
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ```markdown ⏎ ## Purpose ⏎  ⏎ Add a native AMX-FP8 attention path for Diamond Rapids CPUs. ⏎  ⏎ The existing AMX path (added in 22524f7a92) dequants K/V to BF16 before MMA even when FP8 KV-cache is used. This PR introduces a dedicated `AMX_FP8` ISA that uses hardware FP8 MMA intrinsics (`_tile_dpfp8ps` for QK, `_tile_dpbf16ps` for PV) to avoid the dequant overhead. `_get_attn_isa` automatically selects `amx_fp8` over `amx` when FP8 KV-cache is active  …[truncated]

### L3-6865e67f0b  (L3, 2026-09-06, sha 6865e67f0be0, PR #55455)
TITLE: [Bugfix] Defer adaptive verification until after kernel warmup (#55455)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/worker/test_gpu_warmup_blocks.py (+1/-0); tests/v1/worker/test_mixed_warmup_gate.py (+22/-0); vllm/v1/worker/gpu/warmup.py (+15/-0)
LABELS: bug, mrv2
BODY: The B200 LM Eval Spec Decode lane fails during DeepSeek-V4-Flash DSpark startup. The full log from [main build 87376](https://buildkite.com/vllm/ci/builds/87376#01a07028-57a6-4e1f-9e5e-8d39298b7cb4) reaches `AdaptiveVerificationManager.get_num_tokens` from `warmup_kernels` and asserts because `cost_tables` is still `None`. ⏎  ⏎ `GPUWorker.compile_or_warm_up_model` deliberately runs kernel warmup before `capture_model`. Adaptive cost calibration happe …[truncated]

### L3-144e79c810  (L3, 2026-09-06, sha 144e79c8106d, PR #53614)
TITLE: [Kimi K3] Support internal prefix checkpoints with partial prefix caching and spec-decoding (#53614)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: tests/models/kimi_k3/test_kda_metadata.py (+71/-1); tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py (+104/-37); tests/v1/core/test_mamba_align_chunk_split.py (+92/-13); tests/v1/core/test_prefix_caching.py (+1/-0); vllm/models/kimi_k3/nvidia/kda.py (+3/-0); vllm/models/kimi_k3/nvidia/kda_metadata.py (+35/-15); vllm/v1/core/block_pool.py (+12/-3); vllm/v1/core/kv_cache_coordinator.py (+6/-0); vllm/v1/core/sched/scheduler.py (+39/-13); vllm/v1/core/single_type_kv_cache_manager.py (+80/-24); (+1 more)
LABELS: ready, kv-connector, kimi, k3, scheduler, kv-cache-manager
BODY: ## Purpose ⏎  ⏎ Extend the Kimi-K3 prefill checkpoint optimization from #52789 to support: ⏎  ⏎ - speculative decoding / Eagle block rewind; ⏎ - partial prefix caching (`prefix_match_unit < Mamba block size`); and ⏎ - checkpoint blocks restored through KV connectors such as MooncakeStore. ⏎  ⏎ The cache manager and FlashKDA worker share the same checkpoint-validity rules, so a checkpoint block is allocated and hashed only when the worker can write it. For an ali …[truncated]

### L3-199cb9b964  (L3, 2026-09-06, sha 199cb9b96482, PR #55535)
TITLE: [Kernel] Remove unused fake implementation (#55535)
SOURCES: path_core, symbol_pickaxe
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.mla.common_v1
FILES: vllm/model_executor/layers/attention/attention.py (+0/-14); vllm/model_executor/layers/attention/mla_attention.py (+0/-19); vllm/model_executor/layers/attention/static_sink_attention.py (+0/-8); vllm/utils/flashinfer.py (+0/-8); vllm/_aiter_ops.py (+0/-97); vllm/_custom_ops.py (+0/-56); vllm/_xpu_ops.py (+0/-60); vllm/compilation/passes/fusion/allreduce_rms_fusion.py (+0/-19); vllm/kernels/helion/ops/dynamic_per_token_scaled_fp8_quant.py (+0/-10); vllm/kernels/helion/ops/fused_qk_norm_rope.py (+0/-18); (+29 more)
LABELS: intel-gpu, ready, torch.compile, deepseek, cpu, nvidia, DSv4
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-de69e821b7  (L3, 2026-09-06, sha de69e821b7c8, PR #53161)
TITLE: [ROCm][Perf][DeepSeek V4] Fuse native FP8 shared expert with MXFP4 routed experts (#53161)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/layers/test_fused_shared_expert.py (+340/-1); vllm/_aiter_ops.py (+107/-0); vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py (+15/-0); vllm/models/deepseek_v4/amd/model.py (+377/-5)
LABELS: rocm, ready, deepseek, DSv4
DEEP_STUDY: deep-study performance PR ()
BODY: ## Summary ⏎  ⏎ Enable AITER heterogeneous fused MoE (FHMoE) for the DeepSeek V4 ROCm path. ⏎ The guarded path combines native-FP8 shared-expert weights with MXFP4 routed ⏎ experts for model-visible MoE input rows covered continuously by AITER's ⏎ active FHMoE CSV on gfx950/TP8. The currently shipped table covers ⏎ `1 <= M <= 2048`; unsupported M retains the existing separate routed/shared ⏎ path. ⏎  ⏎ The implementation keeps the native per-rank intermed …[truncated]

### L3-c3ec0d29f5  (L3, 2026-09-07, sha c3ec0d29f54c, PR #55237)
TITLE: [Bugfix] Fix cuda profiler missing bug (#55237)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/engine/test_async_llm.py (+45/-1); vllm/v1/engine/async_llm.py (+3/-2)
LABELS: bug, ready, nvidia
BODY: ## Purpose ⏎ Fix `profiler` not found in `AsyncLLM` for profiler types beside torch profiler. ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-f7f060d253  (L3, 2026-09-07, sha f7f060d253fb, PR #55642)
TITLE: [Bugfix][Audio] Restore soundfile-first automatic decoding (#55642)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/features/multimodal_inputs.md (+9/-7); tests/multimodal/media/test_audio.py (+8/-0); vllm/multimodal/media/audio.py (+21/-22)
LABELS: bug, documentation, ready, multi-modality
BODY: Speech jobs passed in [AMD build 12635](https://buildkite.com/vllm/amd-ci/builds/12635/list) before [PR #51826](https://github.com/vllm-project/vllm/pull/51826) made TorchCodec the preferred automatic audio decoder. A local GPU `git bisect` identified its merge commit, `6a039f465e37`, as the cause of the `test_long_audio_request` failures in build 12653 on [MI300](https://buildkite.com/vllm/amd-ci/builds/12653/list?jid=01a070cd-0522-4363-a01e-2cf …[truncated]

### L3-6fbb00b188  (L3, 2026-09-07, sha 6fbb00b18874, PR #41567)
TITLE: [EPD] Add ECMooncakeConnector for encoder cache over Mooncake TransferEngine (#41567)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/disaggregated_mooncake.yaml (+54/-0); examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py (+336/-95); tests/v1/core/test_encoder_cache_manager.py (+32/-0); tests/v1/core/test_scheduler.py (+89/-3); tests/v1/ec_connector/integration/README.md (+212/-170); tests/v1/ec_connector/integration/run_epd_mooncake_ec_full_pipeline.sh (+289/-0); tests/v1/ec_connector/integration/test_epd_correctness.py (+88/-23); tests/v1/ec_connector/unit/test_epd_proxy_retry.py (+172/-0); tests/v1/ec_connector/unit/test_epd_proxy_round_robin.py (+48/-2); tests/v1/ec_connector/unit/test_worker_ec_connector.py (+26/-0); (+20 more)
LABELS: documentation, ready, ci/build, v1, cpu, kv-connector, mrv2, verified, scheduler
BODY: Wire factory and ec_transfer config; add two-process e2e test, EPD full-pipeline script, and README notes. ⏎  ⏎  ⏎ ## Purpose ⏎  ⏎ - Add **`ECMooncakeConnector`**: encoder-cache (EC) transfer over **Mooncake TransferEngine** (HTTP registry + ZMQ coordination + pull path), for disaggregated setups where consumers load EC tensors without relying on shared filesystem. ⏎ - Register the connector in **`ECConnectorFactory`** and document **`ECTransferConfig. …[truncated]

### L3-5893426b88  (L3, 2026-09-07, sha 5893426b88f7, PR #53586)
TITLE: [Bugfix] DSv4 MXFP4 selector: stop narrowing explicit aliases to their BF16 variant (#53586)
SOURCES: body_keyword
ARTIFACT_HINTS: -
FILES: tests/kernels/moe/test_b12x.py (+29/-0); vllm/model_executor/layers/fused_moe/oracle/mxfp4.py (+7/-1)
LABELS: bug, ready, DSv4
BODY: # [Bugfix] DSv4 MXFP4 selector: stop narrowing explicit aliases to their BF16 variant ⏎  ⏎ ## Purpose ⏎  ⏎ On DeepSeek-V4-class MXFP4 models, passing an explicit non-b12x MoE backend ⏎ (e.g. `--moe-backend flashinfer_cutlass`) can fail outright on SM100+/SM120 ⏎ even though a supported variant of that backend exists. ⏎  ⏎ Root cause: the DSv4 selector routed explicit aliases through ⏎ `_get_requested_backends(alias, None)`. With no declared model activation, ⏎ the B …[truncated]

### L3-70584f69b1  (L3, 2026-09-07, sha 70584f69b1bf, PR #54774)
TITLE: [Model] Add Cohere Compass model (#54774)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/registry.py (+4/-0); vllm/model_executor/models/cohere_compass.py (+2354/-0); vllm/model_executor/models/registry.py (+4/-0); vllm/transformers_utils/config.py (+4/-1)
LABELS: new-model, ready, cohere
BODY: Following the release of https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct and addition to transformers in https://github.com/huggingface/transformers/pull/47878 we add a new model "Cohere Compass" aka North Micro Vision Instruct. ⏎  ⏎ ## Purpose ⏎  ⏎ Adding support for the https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct model. ⏎  ⏎ ## Test Plan ⏎  ⏎ Tested against internal benchmarks and huggingface implemenation ⏎  ⏎ ## Test Res …[truncated]

### L3-e476556189  (L3, 2026-09-07, sha e47655618950, PR #54404)
TITLE: [ROCm][CI] Add attention-sink support to ROCm AITER sparse MLA (#54404)
SOURCES: path_core, subject_keyword, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse, L3.dispatch.abstract_interface
FILES: vllm/v1/attention/backend.py (+9/-0); vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py (+216/-50); vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+2/-2); tests/kernels/attention/test_rocm_aiter_mla_op_registration.py (+43/-15); tests/kernels/attention/test_rocm_aiter_mla_sink.py (+512/-0); tests/kernels/attention/test_rocm_aiter_mla_sparse_metadata_sync.py (+21/-0); tests/models/test_initialization.py (+6/-0); tests/v1/attention/test_rocm_glm5next_sparse.py (+52/-0); vllm/_aiter_ops.py (+142/-0)
LABELS: rocm, glm
BODY: HY-V4 introduced learnable attention sinks in [#54160](https://github.com/vllm-project/vllm/pull/54160), but its ROCm initialization was unsupported in [AMD build 12635](https://buildkite.com/vllm/amd-ci/builds/12635/list?jid=01a06dbb-c994-4dda-9277-8d85dae8eecb&tab=output) and [build 12653](https://buildkite.com/vllm/amd-ci/builds/12653/list?jid=01a070cd-0530-4d60-a5fd-a77dcc439bad&tab=output). This PR supplies reusable sparse-MLA sink support r …[truncated]

### L3-1713b9866a  (L3, 2026-09-07, sha 1713b9866ae5, PR #53564)
TITLE: [2/N][warmup][DSv4] Migrate sequence and DCP kernels (#53564)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/ops/common.py (+271/-115); vllm/v1/attention/ops/dcp.py (+271/-99); .buildkite/test_areas/model_executor.yaml (+4/-0); tests/model_executor/test_jit_warmup.py (+3/-1); tests/model_executor/test_jit_warmup_cutedsl_launcher.py (+76/-0); tests/model_executor/test_jit_warmup_triton_launcher.py (+128/-0); vllm/model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py (+501/-328); vllm/model_executor/layers/sparse_attn_indexer.py (+24/-1); vllm/model_executor/warmup/jit_warmup.py (+37/-7); vllm/model_executor/warmup/jit_warmup_cutedsl_helper.py (+80/-0); (+2 more)
LABELS: ready, ci/build, deepseek, DSv4
BODY: Depends on: https://github.com/vllm-project/vllm/pull/50175 ⏎  ⏎ For more details, see parent (draft) PR: https://github.com/vllm-project/vllm/pull/49627 and tracking list issue https://github.com/vllm-project/vllm/issues/49349 ⏎  ⏎ ## Description ⏎  ⏎ This PR migrates sequence-layout and decode-context-parallel JIT kernels used by the DSv4 attention path to the shared warmup contract. ⏎  ⏎ ## What Changed ⏎  ⏎ - Migrated sequence packing/unpacking and att …[truncated]

### L3-42801b3a6b  (L3, 2026-09-07, sha 42801b3a6b3b, PR #53565)
TITLE: [3/N][warmup][DSv4] Migrate FA4 MLA and shared CuTeDSL kernels (#53565)
SOURCES: path_core, subject_keyword, symbol_pickaxe, release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/v1/attention/backends/mla/prefill/flash_attn.py (+8/-2); tests/models/inkling/test_fa4_rel_attention.py (+3/-3); tests/v1/attention/test_mla_prefill_quant_output.py (+33/-0); vllm/model_executor/warmup/fa4_cutedsl_warmup.py (+0/-53); vllm/model_executor/warmup/kernel_warmup.py (+0/-4); vllm/models/inkling/nvidia/attention.py (+5/-2); vllm/models/inkling/nvidia/ops/__init__.py (+2/-2); vllm/models/inkling/nvidia/ops/fa4_rel_attention.py (+1/-1)
LABELS: ready, DSv4, inkling
BODY: Depends on: https://github.com/vllm-project/vllm/pull/50175 ⏎  ⏎ For more details, see parent (draft) PR: https://github.com/vllm-project/vllm/pull/49627 and tracking list issue https://github.com/vllm-project/vllm/issues/49349 ⏎  ⏎ ## Description ⏎  ⏎ This PR migrates the FA4 MLA prefill and Inkling FA4 relative-attention paths to the shared JIT warmup contract. ⏎  ⏎ ## What Changed ⏎  ⏎ - Migrated FA4 MLA prefill specialization and compile-only warmup. ⏎  …[truncated]

### L3-6748217fb9  (L3, 2026-09-07, sha 6748217fb949, PR #55407)
TITLE: [Bugfix] Fix Kimi K3 NVFP4 MoE weight conversion OOM (#55407)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py (+71/-83)
LABELS: bug, nvidia, quantization, kimi, k3
BODY: ## Purpose ⏎ Currently when running Kimi k3 nvfp4, it throws ~15000 lines of warning like the following due to fragmented memory during quantization post processing. In some cases, it also fatally fails with OOM. ⏎ ``` ⏎ [rank6]:[W904 12:06:06.083917542 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 6 while trying to allocate 20971520 bytes (free: 16252928, total: 287428640768). ⏎ [rank6]:[W904 12:06:06.087318728 CUDACachi …[truncated]

### L3-3dc7a68ce4  (L3, 2026-09-07, sha 3dc7a68ce45c, PR #54797)
TITLE: [Perf] Extend Qwen Triton warmup to avoid first-request latency spikes (#54797)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/model_executor/test_mamba_triton_warmup.py (+14/-0); tests/model_executor/test_qwen_triton_warmup.py (+55/-0); tests/model_executor/test_qwen_vl_triton_warmup.py (+117/-0); vllm/model_executor/warmup/kernel_warmup.py (+4/-0); vllm/model_executor/warmup/mamba_triton_warmup.py (+53/-0); vllm/model_executor/warmup/qwen_triton_warmup.py (+13/-19); vllm/model_executor/warmup/qwen_vl_triton_warmup.py (+112/-0)
LABELS: ready, qwen
DEEP_STUDY: deep-study performance PR ()
BODY: # [Perf] Extend Qwen Triton warmup to avoid first-request latency spikes ⏎  ⏎ ## Purpose ⏎  ⏎ Extend `qwen_triton_warmup` so remaining Qwen3.5 / Qwen3-Next Triton kernels compile **before** the JIT monitor is armed, instead of on the first live request. That first-request compile spike is a large hit for latency-sensitive serving. ⏎  ⏎ The warnings below were captured on **pooling** Qwen3.5 (embedding / classification). **Generate** VL hits most of the …[truncated]

### L3-b339d75a41  (L3, 2026-09-07, sha b339d75a410e, PR #55369)
TITLE: [Bugfix][Spec Decode] Resolve n_predict from text_config for Qwen3.5 multimodal MTP (#55369)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: .buildkite/test_areas/models_basic.yaml (+2/-1); tests/models/test_qwen3_5_mtp_config.py (+96/-8); vllm/config/speculative.py (+3/-0)
LABELS: bug, ready, ci/build, qwen
ISSUES: #55322 [Bug]: Qwen3.5/3.6 MTP resolves n_predict=None for multimodal-wrapper checkpoints (mtp_num_hidden_layers is read from the wrapper, not text_config)
BODY: ### Purpose ⏎ In Qwen3.5/3.6 multimodal wrapper checkpoints (such as `Qwen/Qwen3.6-35B-A3B` and `Qwen/Qwen3.8-27B`), `mtp_num_hidden_layers` is located in `text_config` rather than directly on the outer wrapper config. `SpeculativeConfig.hf_config_override` previously inspected only the top-level `hf_config`, leaving `n_predict` as `None` and causing downstream divisibility checks to be bypassed. ⏎  ⏎ ### Changes ⏎ 1. Extracts `text_config` via `get_hf_t …[truncated]

### L3-f2d45f26bd  (L3, 2026-09-07, sha f2d45f26bd6a, PR #54917)
TITLE: [Bugfix][Gemma] Conditionally create KV projections/norms on KV-shared layers (#54917)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/models/language/generation/test_gemma.py (+66/-0); vllm/model_executor/models/gemma3n.py (+79/-37); vllm/model_executor/models/gemma4.py (+77/-62)
LABELS: bug, ready
BODY: ## Purpose ⏎ transformers only creates `k_proj` / `v_proj` / `k_norm` on the **non-KV-shared** layers of Gemma 4 E2B/E4B and Gemma 3n.  ⏎ Any checkpoint that goes through `save_pretrained` (HF Trainer, TRL, verl, ...) therefore omits those tensors for the last `num_kv_shared_layers` layers. The original Google releases happen to ship them redundantly, so this only surfaces after fine-tuning. ⏎  ⏎ vLLM created those modules unconditionally, so fine-tu …[truncated]

### L3-a69402aaa8  (L3, 2026-09-07, sha a69402aaa817, PR #55364)
TITLE: [Perf] Integrate FlashInfer KDA kernels (#55364)
SOURCES: path_core, path_integration+keyword, subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/engine/arg_utils.py (+14/-2); vllm/utils/flashinfer.py (+34/-0); tests/models/kimi_k3/test_kda.py (+348/-113); tests/models/kimi_k3/test_kda_metadata.py (+4/-1); vllm/model_executor/layers/mamba/mamba_utils.py (+12/-2); vllm/models/kimi_k3/nvidia/kda.py (+284/-36); vllm/models/kimi_k3/nvidia/kda_metadata.py (+23/-0); vllm/models/kimi_k3/nvidia/model.py (+3/-1)
LABELS: nvidia, kimi, k3
DEEP_STUDY: deep-study performance PR (new_kernel_or_fusion)
BODY: ## Purpose ⏎ This PR adds the following: ⏎ - Support bf16 KDA cache state for Kimi K3 ⏎ - Integrate flashinfer KDA prefill and decode backend ⏎ - The default backends will remain the same as before ⏎  ⏎ #### Microbenchmark: FlashInfer fused BF16 vs. Triton BF16 fallback ⏎ | Batch/tokens | FlashInfer | Triton | Speedup | ⏎ |---:|---:|---:|---:| ⏎ | 1 | 4.08 μs | 10.08 μs | 2.47× | ⏎ | 2 | 4.54 μs | 10.07 μs | 2.22× | ⏎ | 4 | 4.63 μs | 10.48 μs | 2.26× | ⏎ | 8 …[truncated]

### L3-9c297e3b8b  (L3, 2026-09-07, sha 9c297e3b8bc8, PR #55755)
TITLE: [Kernel] PDL enablement for fusedQKNormRopeKernel (#55755)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: csrc/libtorch_stable/fused_qknorm_rope_kernel.cu (+112/-48)
LABELS: ready
DEEP_STUDY: deep-study performance PR (kernel_optimization)
BODY: ## Purpose ⏎  ⏎ ## Test Plan ⏎  ⏎ ## Test Result ⏎  ⏎ --- ⏎ [details omitted]

### L3-869f78732b  (L3, 2026-09-07, sha 869f78732b64, PR #55747)
TITLE: [Kimi Bug] Fix kimi k3 AssertionError assert 0 <= checkpoint_idx < len(blocks) (#55747)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py (+1/-1); vllm/v1/core/single_type_kv_cache_manager.py (+16/-11)
LABELS: bug, ready, kimi, k3, kv-cache-manager
BODY: ## Purpose ⏎  ⏎ ```bash ⏎  ⏎ vllm serve moonshotai/Kimi-K3 \ ⏎   --trust-remote-code \ ⏎   --tensor-parallel-size 8 \ ⏎   --load-format fastsafetensors \ ⏎   --gpu-memory-utilization 0.92 \ ⏎   --enable-prefix-caching \ ⏎   --reasoning-parser kimi_k3 \ ⏎   --enable-auto-tool-choice \ ⏎   --tool-call-parser kimi_k3 \ ⏎   --host 0.0.0.0 \ ⏎   --port 30000 \ ⏎   --speculative-config '{"model":"RedHatAI/Kimi-K3-speculator.dspark","method":"dspark","num_speculative_ …[truncated]

### L3-54da70c1eb  (L3, 2026-09-08, sha 54da70c1eb69, PR #53052)
TITLE: [Feature] Support EAGLE3 for Sarvam (#53052)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/sarvam.py (+40/-6)
LABELS: documentation, ready
BODY: ## Purpose ⏎  ⏎ Sarvam MLA (`SarvamMLAForCausalLM`) previously did not expose the EAGLE3 interface or capture the auxiliary hidden states required by the drafter. This adds EAGLE3 support with a compatible draft checkpoint on a single pipeline stage. ⏎  ⏎ - `SarvamMLAModel` uses `EagleModelMixin` to capture embeddings and complete layer outputs (`hidden_states + residual`). ⏎ - `SarvamMLAForCausalLM` declares `SupportsEagle3` and inherits the protocol's au …[truncated]

### L3-7fa2c63796  (L3, 2026-09-08, sha 7fa2c637968e, PR #55377)
TITLE: [Bugfix] Autotune FlashInfer deferred MoE decode kernels before CUDA graph capture (#55377)
SOURCES: path_core, release_notes, body_keyword
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode
FILES: vllm/utils/flashinfer.py (+4/-0); tests/model_executor/test_flashinfer_autotune_warmup.py (+116/-0); vllm/model_executor/warmup/kernel_warmup.py (+39/-8)
LABELS: bug, ready, nvidia, verified
DEEP_STUDY: deep-study performance PR (kernel_tuning_config)
BODY: ## Purpose ⏎  ⏎ Split out the MoE half of #54507 into a standalone fix. This is not duplicate work: #54507 covers both the MLA and MoE decode-autotune cache-miss bugs, while this PR isolates only the MoE part so it can be reviewed and merged independently. ⏎  ⏎ Kimi-K3 decode defers MoE finalization so the consumer can fuse the final top-k reduction. That changes the monolithic FlashInfer MoE output signature from `(tokens, hidden)` to `(tokens, 0)`. The …[truncated]

### L3-782f36cd0c  (L3, 2026-09-08, sha 782f36cd0c79, PR #55779)
TITLE: [Bugfix][InternVL] Stop the video parser consuming image_embeds (#55779)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: vllm/model_executor/models/internvl.py (+1/-1)
LABELS: bug, ready
BODY: ## Purpose ⏎  ⏎ `InternVLChatModel._parse_and_validate_video_input` pops `image_embeds`: ⏎  ⏎ ```python ⏎ video_embeds = kwargs.pop("image_embeds", None) ⏎ ``` ⏎  ⏎ `image_embeds` belongs to the image modality. When a request carries image ⏎ embeddings and a video together, the video parser reads those embeddings, matches ⏎ the `video_embeds is not None` branch, and returns early with ⏎ `InternVLVideoEmbeddingInputs` wrapping the image data. The video is ne …[truncated]

### L3-25047604fe  (L3, 2026-09-08, sha 25047604fe55, PR #55808)
TITLE: [ROCm][Perf] Remove AITER paged-MQA outputs guard for DeepSeek-V4 (#55808)
SOURCES: path_core, release_notes
ARTIFACT_HINTS: L3.mla.rocm_aiter_sparse
FILES: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py (+0/-1); tests/kernels/attention/test_rocm_triton_attn_dsv4.py (+0/-62)
LABELS: rocm, deepseek, DSv4
ISSUES: #13 [ROCm][DeepSeek V4] TP8 graph-mode GPU memory fault under 64 concurrent requests
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎  ⏎ To fix https://github.com/Fangzhou-Ai/vllm/issues/13, https://github.com/vllm-project/vllm/pull/49714 added an additional element-wise kernel after `deepgemm_fp8_paged_mqa_logits`, which could introduce costs. ⏎  ⏎ <img width="1909" height="310" alt="Snipaste_2026-09-08_14-14-51" src="https://github.com/user-attachments/assets/9df30a96-96a5-4536-bb99-4476c4105647" /> ⏎  ⏎ This issue has been fixed in AITER side in https://github.com/ROC …[truncated]

### L3-ce6c241ddc  (L3, 2026-09-08, sha ce6c241ddcc7, PR #54787)
TITLE: [ROCm][Perf][M3] Fused allreduce+GemmaRMSNorm fast path (#54787)
SOURCES: release_notes, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/_aiter_ops.py (+7/-73); vllm/distributed/device_communicators/aiter_custom_all_reduce.py (+34/-0); vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py (+36/-0)
LABELS: rocm, ready, verified
DEEP_STUDY: deep-study performance PR ()
BODY: ## Purpose ⏎ `fused_allreduce_gemma_rms_norm` only has a flashinfer fast path; on ROCm it always **falls back to all_reduce + GemmaRMSNorm** — two kernel launches per layer.  ⏎  ⏎ This PR addes the existing aiter's rocm_aiter_fused_allreduce_rmsnorm custom op into the helper, mirroring the flashinfer branch.  ⏎ before:  two kernels ⏎ <img width="3793" height="976" alt="image" src="https://github.com/user-attachments/assets/eae4f057-75d6-42a6-9c97-a882 …[truncated]

### L3-7c2f1ff495  (L3, 2026-09-08, sha 7c2f1ff4958e, PR #54523)
TITLE: [Core] Scope PCP-DP validation to GPU manager (#54523)
SOURCES: release_notes
ARTIFACT_HINTS: L3.platform.cuda_selection, L3.platform.rocm_selection
FILES: vllm/config/parallel.py (+0/-2); vllm/platforms/cuda.py (+6/-0); vllm/platforms/rocm.py (+6/-0)
LABELS: rocm, ready, nvidia, mrv2
BODY: ## Purpose ⏎  ⏎ Move the current PCP-with-DP capability check from the platform-independent ParallelConfig validator to the GPU MRV2 PCPManager. This keeps the existing GPU behavior while allowing out-of-tree hardware backends to provide their own PCP+DP support. ⏎  ⏎ ## Test Plan ⏎  ⏎ - Run Ruff lint and formatting checks on the changed files. ⏎ - Run the standard PR CI checks. ⏎  ⏎ ## Test Result ⏎  ⏎ - Ruff lint: passed. ⏎ - Ruff format check: passed. ⏎ - Python compil …[truncated]

### L3-263c4ff95f  (L3, 2026-09-08, sha 263c4ff95fad, PR #53945)
TITLE: [Bugfix][Spec Decode] Cache the Mamba state at the block-grid position of EAGLE resume (#53945)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: docs/features/automatic_prefix_caching.md (+22/-0); tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py (+359/-0); tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py (+175/-0); tests/v1/core/test_mamba_align_chunk_split.py (+9/-3); tests/v1/core/test_prefix_caching.py (+8/-6); tests/v1/core/test_single_type_kv_cache_manager.py (+1/-1); vllm/config/cache.py (+7/-0); vllm/engine/arg_utils.py (+10/-0); vllm/v1/core/kv_cache_coordinator.py (+26/-0); vllm/v1/core/kv_cache_manager.py (+17/-0); (+2 more)
LABELS: bug, documentation, ready, kv-connector, scheduler, kv-cache-manager
BODY: ## Purpose ⏎  ⏎ Fixes the second of the two EAGLE + `--mamba-cache-mode align` prefix-cache defects pinned ⏎ by #52371. ⏎  ⏎ Full attention hits at a position it holds a key for. EAGLE prunes one hash unit off that ⏎ candidate and drops it. The Mamba group materializes state only on its own block grid, so ⏎ nothing exists at the resulting position and the hit floors back to the previous block ⏎ boundary. ⏎  ⏎ Commit 1 covers the case where the shared prefi …[truncated]

### L3-6ddbab03de  (L3, 2026-09-08, sha 6ddbab03defe, PR #50176)
TITLE: [4/N][warmup][DSv4] Migrate common attention kernels (#50176)
SOURCES: subject_keyword, release_notes
ARTIFACT_HINTS: -
FILES: tests/kernels/core/test_fused_q_kv_rmsnorm.py (+3/-1); tests/model_executor/test_jit_warmup.py (+34/-0); vllm/model_executor/warmup/jit_warmup.py (+18/-3); vllm/model_executor/warmup/jit_warmup_triton_helper.py (+41/-9); vllm/models/common/ops/__init__.py (+2/-1); vllm/models/common/ops/fused_qk_rmsnorm.py (+196/-90); vllm/models/deepseek_v4/amd/mtp.py (+9/-5); vllm/models/deepseek_v4/attention.py (+58/-1); vllm/models/deepseek_v4/common/ops/__init__.py (+0/-5); vllm/models/deepseek_v4/common/ops/cache_utils.py (+903/-454); (+9 more)
LABELS: ready, v1, deepseek, DSv4, kimi, k3, inkling
BODY: Depends on: https://github.com/vllm-project/vllm/pull/50175 ⏎  ⏎ For more details, see parent PR: https://github.com/vllm-project/vllm/pull/49627 and tracking issue https://github.com/vllm-project/vllm/issues/49349 ⏎  ⏎ ## Description ⏎  ⏎ This PR migrates DSv4 common attention preparation, cache, indexer, and sparse-attention kernels to the shared warmup contract. ⏎  ⏎ ## What Changed ⏎  ⏎ - Migrated fused Q/K normalization and MTP input normalization. ⏎ - …[truncated]

### L3-07950d4734  (L3, 2026-09-08, sha 07950d47347f, PR #55774)
TITLE: [Kimi Bug] Fix kimi k3 startup cuda graph issue with recoverSSM (#55774)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/v1/cudagraph/test_cudagraph_manager.py (+10/-1); vllm/v1/worker/gpu/cudagraph_utils.py (+5/-1)
LABELS: bug, ready, nvidia, mrv2, kimi, k3
BODY: ## Purpose ⏎  ⏎ ```bash ⏎ vllm serve moonshotai/Kimi-K3   --trust-remote-code   --tensor-parallel-size 8   --load-format fastsafetensors   --gpu-memory-utilization 0.9   --enable-prefix-caching   --reasoning-parser kimi_k3   --enable-auto-tool-choice   --tool-call-parser kimi_k3   --host 0.0.0.0   --port 30000   --max-model-len auto   --max-num-seqs 48   --speculative-config '{"model":"RedHatAI/Kimi-K3-speculator.dspark","method":"dspark","num_specu …[truncated]

### L3-e41a17e606  (L3, 2026-09-08, sha e41a17e606c9, PR #52263)
TITLE: [ROCm][Quantization] Support AMD Quark per-block FP8 for fused MoE layers (#52263)
SOURCES: release_notes
ARTIFACT_HINTS: -
FILES: tests/quantization/test_quark.py (+158/-1); vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py (+5/-19); vllm/model_executor/layers/quantization/fp8.py (+12/-22); vllm/model_executor/layers/quantization/quark/quark_moe.py (+79/-14); vllm/model_executor/layers/quantization/utils/fp8_utils.py (+29/-0)
LABELS: rocm, ready, needs-rebase, quantization, verified
DEEP_STUDY: deep-study performance PR (precision_format)
BODY: ## Purpose ⏎  ⏎ #47972 added per-block (128x128) FP8 for Quark **linear** layers, but ⏎ `QuarkW8A8Fp8MoEMethod` still rejects the qscheme, so a Quark checkpoint whose routed ⏎ experts are quantized per-block loads every linear layer and then fails: ⏎  ⏎ ``` ⏎ ValueError: For FP8 Fused MoE layers, only per-tensor and per-channel scales for ⏎ weights and activations are supported. Found per_block, per_group ⏎ ``` ⏎  ⏎ ## Changes ⏎  ⏎ `QuarkW8A8Fp8MoEMethod` now …[truncated]

### L3-6b15bea080  (L3, 2026-09-08, sha 6b15bea080ac, PR #53941)
TITLE: [Refactor] Remove utils dead code (#53941)
SOURCES: path_core
ARTIFACT_HINTS: L3.flashinfer.utils_dependency, L3.flashinfer.trtllm_gen, L3.flashinfer.trtllm_xqa_decode, L3.dispatch.abstract_interface
FILES: vllm/utils/flashinfer.py (+0/-38); vllm/v1/attention/backends/utils.py (+1/-14); tests/models/multimodal/generation/test_memory_leak.py (+2/-1); vllm/compilation/passes/fx_utils.py (+0/-10); vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/utils.py (+0/-74); vllm/model_executor/layers/utils.py (+0/-8); vllm/model_executor/model_loader/weight_utils.py (+0/-15); vllm/utils/hpc.py (+0/-41); vllm/utils/mem_utils.py (+0/-5); vllm/utils/torch_utils.py (+0/-9); (+5 more)
LABELS: ready, ray, torch.compile, multi-modality, kv-connector, nvidia, kv-cache-manager
BODY: ## Purpose ⏎  ⏎ Remove utils dead code

### L3-f6326f53bd  (L3, 2026-09-08, sha f6326f53bda4, PR #55715)
TITLE: [Perf][GDN] Enable the FlashInfer GDN prefill kernel on SM12x (#55715)
SOURCES: subject_keyword, release_notes, corpus:performance-pr-population, body_keyword
ARTIFACT_HINTS: -
FILES: vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py (+10/-2)
LABELS: ready
DEEP_STUDY: deep-study performance PR (perf_regression_fix)
BODY: ## Purpose ⏎  ⏎ FlashInfer's GDN prefill kernel is never used on SM12x, so consumer and SoC Blackwell (RTX PRO 6000, RTX 5090, DGX Spark GB10) silently runs the Triton/FLA fallback for every linear-attention layer. On Qwen3.5/3.6/3.8 that is 3 of every 4 layers. ⏎  ⏎ `_resolve_gdn_prefill_backend()` only sets `supports_flashinfer` for SM90 or the SM10x family, so SM12x falls through to `return backend, "triton"`. ⏎  ⏎ The gate is now stale.. It was wri …[truncated]
