# Kernel lineage study — synthesis

*What survives when production GPU kernels evolve across framework releases?* Three lineages were reconstructed from full Git history, pull requests, releases, tests and the deep-study corpora, through the frozen cutoff **2026-09-28T23:59:59Z**: **L1** SGLang MoE alignment, routing, top-k and fusion; **L2** SGLang MLA, FlashInfer MLA and FlashMLA; **L3** the vLLM attention implementation family. Every number below is generated from `data/*.csv` and carries a hidden claim tag re-verified by `scripts/consistency.py`. Coding was performed by LLM coders under a written codebook (`CODEBOOK.md`); every candidate was inspected in context, and a blind recode of a stratified sample was adjudicated. Agreement figures are model–model consistency, **not** human inter-rater reliability. Repository history cannot establish causality; interpretations are marked. Method: [`METHODS.md`](METHODS.md). Lineage reports: [`sglang-moe-lineage.md`](sglang-moe-lineage.md), [`sglang-mla-lineage.md`](sglang-mla-lineage.md), [`vllm-attention-lineage.md`](vllm-attention-lineage.md).

## Executive summary

- **Scale.** {{M|all|candidates|value|int}} candidate changes were screened in context; {{M|all|verified_events|value|int}} are verified lineage events ({{M|L1|verified_events|value|int}} L1, {{M|L2|verified_events|value|int}} L2, {{M|L3|verified_events|value|int}} L3), linked by {{M|all|edges|value|int}} typed edges over {{M|all|artifacts_registered|value|int}} registered artifacts and {{M|all|moves|value|int}} optimization moves.
- **Maintenance is incremental, and mostly repair.** {{M|all|incremental_events|share|pct1}}% of events are incremental (optimize, retune, repair, adapt, extend, integrate) versus {{M|all|replacement_events|share|pct1}}% replacement-type events (replace, default change, removal, deprecation, reimplementation). Correctness repair alone is {{M|all|event_type:repair_correctness|share|pct1}}% of events, and {{M|all|transitions_with_repair|share|pct1}}% of release transitions carried at least one repair.
- **Code identity is short-lived; mechanisms persist.** Across three subsequent releases, {{C|data/survival-by-releases.csv|kind=artifact&lineage=all&k_releases=3&basis=all_present_releases|share|pct1}}% of artifacts were still present but only {{C|data/survival-by-releases.csv|kind=artifact_unchanged&lineage=all&k_releases=3&basis=all_present_releases|share|pct1}}% were textually unchanged, while move code signatures persisted in {{C|data/survival-by-releases.csv|kind=move&lineage=all&k_releases=3&basis=all_present_releases|share|pct1}}%.
- **The most repaired move is a framework protocol, not a kernel trick.** CUDA-graph static-metadata adaptation in vLLM accounts for {{M|L3|move_repair_events:M-L3-cudagraph-static-metadata|value|int}} repair or revert events among its {{M|L3|move_repair_events:M-L3-cudagraph-static-metadata|denominator|int}} coded events, although the guarantee it rests on is a documented PyTorch/CUDA contract.
- **Assumptions are rarely backed by a specification.** Of {{M|all|audited_assumptions|value|int}} load-bearing assumptions audited against public documentation, {{M|all|assumption_status:specified_guarantee|share|pct1}}% are specified guarantees; the rest are derived in code, tested only, undocumented or unclear.

## 1. Are production kernels maintained incrementally, repeatedly replaced, or both?

**Both — but at different levels.** Individual implementations are overwhelmingly maintained incrementally: {{M|all|incremental_events|value|int}} incremental events against {{M|all|replacement_events|value|int}} replacement-type events. Classifying every present artifact at every release transition ({{M|all|artifact_transition_class:direct_carry|denominator|int}} artifact × transition observations; `CODEBOOK.md` §9, v1.2–v1.3), {{M|all|artifact_transition_class:direct_carry|share|pct1}}% crossed without any verified lineage event (`direct_carry`; in {{M|all|artifact_transition_direct_carry_textually_modified|share|pct1}}% of all observations the files still changed textually, only through commits screened as outside the lineage), {{M|all|artifact_transition_class:derivation_repair|share|pct1}}% needed hand re-derivation of kernel internals (`derivation_repair`: optimizations, new cases and fixes inside kernel bodies), {{M|all|artifact_transition_class:adapter_repair|share|pct1}}% needed adapter or build repair, {{M|all|artifact_transition_class:backend_replacement|share|pct1}}% lost default status to another backend, {{M|all|artifact_transition_class:obsolete|share|pct1}}% were removed or deprecated and {{M|all|artifact_transition_class:manual_reimplementation|share|pct1}}% were reimplemented in the same role.

Replacement is real but lives at the **backend and default level**: {{M|all|events_default_change|value|int}} events change which implementation is selected for some configuration. The clearest case is vLLM's original PagedAttention CUDA kernel — introduced with the project, extended with split-KV partitioning ([#1348](https://github.com/vllm-project/vllm/pull/1348)), displaced as the default by FlashAttention/FlashInfer backends and finally deleted ([#47361](https://github.com/vllm-project/vllm/pull/47361)) — while its KV-cache write kernels and paged layout contract lived on (L3 report §8–9). In SGLang MLA, the attention backend for DeepSeek-style models changed repeatedly by hardware (Triton → FlashInfer MLA → FlashMLA → FA3 on Hopper → TRT-LLM/CUTLASS on Blackwell, with CUTLASS MLA removed in [#32114](https://github.com/sgl-project/sglang/pull/32114)) around an algebraic move — weight absorption — that stayed constant (L2 report).

*Interpretation:* repositories keep kernels alive by continuous local repair; replacement happens when a better external implementation appears for a hardware generation, and it is expressed as dispatch churn rather than as rewriting the surviving kernels.

## 2. How long do kernel artifacts survive compared with optimization ideas?

Artifacts, as named files and adapters, survive long: the Kaplan–Meier median lifetime of L3 artifacts is {{M|L3|km-artifact-lifetime_median_days|value|int}} days, and for L1 and L2 the median is not reached by the cutoff (fewer than half of their artifacts have ended); {{M|all|artifacts_live_at_cutoff|value|int}} of {{M|all|artifacts_registered|value|int}} registered artifacts are live at the cutoff. Many are young, so the lifetime figure is conservative.

Their **code** does not survive: an artifact present at a release was textually unchanged one release later in only {{C|data/survival-by-releases.csv|kind=artifact_unchanged&lineage=all&k_releases=1&basis=all_present_releases|share|pct1}}% of cases and three releases later in {{C|data/survival-by-releases.csv|kind=artifact_unchanged&lineage=all&k_releases=3&basis=all_present_releases|share|pct1}}%. Measured line by line with `git blame` at the cutoff, {{M|all|event_code_survival_share:2024|value|pct1}}% of 2024 events still own at least one line, against {{M|all|event_code_survival_share:2026|value|pct1}}% of 2026 events.

**Ideas** persist: validated code signatures of the catalogued moves were present {{C|data/survival-by-releases.csv|kind=move&lineage=all&k_releases=3&basis=all_present_releases|share|pct1}}% of the time three releases later. Some moves survive only conceptually: PagedAttention V2's split-KV partition-and-merge no longer exists as the original kernel, but the same mechanism — partial softmax statistics per KV split merged with a log-sum-exp reduction — lives in vLLM's merge kernels and in the FlashAttention, FlashInfer and Triton decode paths (L3 biography of `M-L3-pagedattention-v2-split-kv-reduce`).

*Caveat:* a signature detects the mechanism's identifying code (kernel names, metadata fields, reduction buffers), not a proof that the mechanism is intact.

## 3. Which optimization moves transport successfully across releases and backends?

Moves that are **mechanism-level and contract-light** transport best. Of {{M|all|moves|value|int}} moves, {{M|all|moves_cross_repo|value|int}} have verified occurrences in another repository. Examples:

- **MoE block alignment** (L1): vLLM's `moe_align_block_size` padding contract ([#2453](https://github.com/vllm-project/vllm/pull/2453)) moved into SGLang's own kernel ([#2579](https://github.com/sgl-project/sglang/pull/2579)); SGLang's two-phase shared-memory cumsum was then ported *back* to vLLM ([#12574](https://github.com/vllm-project/vllm/pull/12574), [#19572](https://github.com/vllm-project/vllm/pull/19572)), both PRs naming SGLang as the source.
- **Fused gating / top-k softmax** (L1): SGLang cherry-picked vLLM's top-k softmax template ([#4302](https://github.com/sgl-project/sglang/pull/4302)) and then generalised it into a fused biased grouped top-k gate ([#4530](https://github.com/sgl-project/sglang/pull/4530)).
- **MLA weight absorption** (L2 → L3): computing decode attention in the compressed latent space, first in SGLang's Triton path ([#905](https://github.com/sgl-project/sglang/pull/905)), reappears across every MLA backend in both engines.
- **Split-KV with LSE merge** (L3 ↔ L2): from PagedAttention V2 to Triton decode kernels of LightLLM lineage in both engines and to library kernels.

What transports is the *mechanism and its contract*, re-expressed by hand; textual carry across repositories is rare.

## 4. Which moves repeatedly require manual repair?

Repairs concentrate on moves that sit on **framework protocol boundaries**: {{M|all|moves_with_repair_events|value|int}} of {{M|all|moves|value|int}} moves have at least one coded repair or revert. The most repaired is CUDA-graph static-metadata adaptation ({{M|L3|move_repair_events:M-L3-cudagraph-static-metadata|share|pct1}}% of its coded events are repairs or reverts), followed by FP8 KV-cache scaling ({{M|L3|move_repair_events:M-L3-fp8-kv-cache-scales|value|int}} of {{M|L3|move_repair_events:M-L3-fp8-kv-cache-scales|denominator|int}}) and SGLang's routed-scaling fusion ({{M|L1|move_repair_events:M-L1-routing-scale-fusion|value|int}} of {{M|L1|move_repair_events:M-L1-routing-scale-fusion|denominator|int}}). Artifacts reach a first correctness or performance repair after a Kaplan–Meier median of {{M|all|km-first-repair_median_days|value|int}} days; release transitions carry on average {{M|all|repairs_per_transition_mean|value|dec2}} repair events.

*Interpretation:* the fragile part of an optimization is usually not its arithmetic but the invariants that connect it to the framework — persistent buffers for graph replay, scale tensors threaded through cache writes, the order in which routed weights are scaled and reduced.

## 5. Specification vs framework protocol vs hardware/compiler

Primary causes of all verified events: correctness {{M|all|primary_cause:correctness|share|pct1}}%, performance {{M|all|primary_cause:performance|share|pct1}}%, framework integration {{M|all|primary_cause:framework_integration|share|pct1}}%, specification {{M|all|primary_cause:specification|share|pct1}}%, build/dependency {{M|all|primary_cause:build_dependency|share|pct1}}%, hardware/compiler {{M|all|primary_cause:hardware_compiler|share|pct1}}%, maintenance {{M|all|primary_cause:maintenance|share|pct1}}%. Changed operator semantics (true contract changes: {{M|all|spec_change_events|value|int}} events) are outnumbered by newly supported inputs ({{M|all|newly_supported_events|value|int}} events: new dtypes, head dimensions, models) and by framework-protocol work. Hardware is pervasive as *scope* — {{M|all|events_hardware_specific|share|pct1}}% of events are hardware-specific — but is rarely the primary cause.

*Caveat:* primary cause had the weakest blind-recode agreement ({{C|recoding/agreement.csv|field=primary_cause|percent_agreement|pct1}}%, κ = {{C|recoding/agreement.csv|field=primary_cause|cohen_kappa|dec2}}); the adjudicator traced most disputes to the boundary between `specification` and `hardware_compiler` for new dtype/shape support. Treat the cause shares as approximate.

## 6. Do failures and reverts produce durable, reusable knowledge?

Sometimes, locally. Across {{M|all|failure_histories|value|int}} reconstructed failure-to-successor histories, the fix added a test in {{M|all|failure_histories_encoding:test|value|int}} and a guard (disabled path, assertion, capability check) in {{M|all|failure_histories_encoding:guard|value|int}}; {{M|all|failure_histories_no_durable_encoding|value|int}} had no durable encoding found. The event graph holds {{M|all|event_type:revert|value|int}} reverts and {{M|all|event_type:reland|value|int}} relands.

**Repeated assumptions do occur.** In {{M|all|failure_histories_repeat_assumption|value|int}} history the same hazard verifiably resurfaced in a sibling code path: vLLM [#40654](https://github.com/vllm-project/vllm/pull/40654) switched attention metadata from the upper bound `max_model_len` to the actual batch maximum, its follow-up fix [#40772](https://github.com/vllm-project/vllm/pull/40772) repaired an illegal memory access that the change exposed in DSA+MTP cache gathering, and a later follow-up [#43991](https://github.com/vllm-project/vllm/pull/43991) found two model-runner paths that had been missed, where "handing `max_model_len` to FlashInfer makes the TRTLLM attention walk past the valid block-table entries" (L3 history F-L3-01). {{M|all|failure_histories_repeat_possible|value|int}} more history shows a possible recurrence (a second FlashMLA dependency rollback and reland in SGLang, F-L2-03); for the remaining {{M|all|failure_histories_repeat_none_found|value|int}} no repetition was found within the searched scope, which does not show absence of repetition elsewhere.

*Interpretation:* lessons are encoded where the failure happened — as a guard around one backend or a test for one shape — rather than as reusable, backend-independent knowledge. F-L3-01 is the pattern in miniature: the precondition "lengths handed to block-table-walking kernels must be exact, not upper bounds" was repaired path by path, and the same unstated precondition failed again in paths that consumed the same metadata.

## 7. Are optimizations rediscovered? Is prior lineage acknowledged?

Two patterns are visible. **Cross-repository ports are acknowledged**: SGLang and vLLM cite each other when moving MoE alignment and top-k kernels (Q3), and SGLang's Triton attention kernels keep LightLLM source URLs. **Within-repository rewrites re-derive mechanisms silently**: when vLLM rebuilt its backends for the V1 engine, the V1 FlashAttention and FlashInfer backends ([#9289](https://github.com/vllm-project/vllm/pull/9289), [#16684](https://github.com/vllm-project/vllm/pull/16684)), the XQA decode path ([#43232](https://github.com/vllm-project/vllm/pull/43232)) and the Triton unified attention kernel ([#16828](https://github.com/vllm-project/vllm/pull/16828)) re-implemented mechanisms that earlier backends already had, and no acknowledgement of the earlier in-repository implementation was found in their PR text. {{M|all|moves_with_unacknowledged_recurrence|value|int}} moves show such unacknowledged recurrences.

*Interpretation:* repositories preserve provenance when code crosses an organizational boundary (licences, courtesy, review), but not when the same organization re-derives an idea during a framework rewrite. The earlier implementation's failures and preconditions are then not visibly consulted.

## 8. Where do implementations rely on specified, derived, tested-only, undocumented or unclear behaviour?

Of {{M|all|audited_assumptions|value|int}} load-bearing assumptions audited against public specifications: specified guarantees {{M|all|assumption_status:specified_guarantee|value|int}}, derived in code {{M|all|assumption_status:derived|value|int}}, tested only {{M|all|assumption_status:tested_only|value|int}}, unclear {{M|all|assumption_status:unclear|value|int}}, undocumented {{M|all|assumption_status:undocumented_behavior|value|int}}.

- **Specified** (documentation checked and cited): warp width and full-mask shuffle participation for the PagedAttention and alignment warp scans (CUDA C++ Programming Guide, warp shuffle functions); wavefront-64 on AMD for the HIP variants (HIP docs); persistent buffer addresses for CUDA-graph replay (PyTorch CUDA-graphs notes); cross-kernel visibility of partial results in two-kernel split-KV reductions.
- **Derived**: most semantic preconditions — distinct top-k expert ids, padding that makes every tile belong to one expert, one latent KV head for absorbed MLA — are asserted or argued in code.
- **Tested only**: profitability thresholds and regime choices (small-batch alignment fast paths, when to skip absorption for prefill, split counts) are backed only by benchmarks.
- **Undocumented**: bitwise reproducibility of LSE-merged split-KV results across split choices, and AMD MFMA builtins used outside the portable HIP language documentation.

*Interpretation:* correctness preconditions are mostly written down *in the code*, but profitability preconditions are not, and numerical determinism is assumed rather than specified.

## 9. Would a replayable optimization derivation plausibly reduce observed work?

Plausibly, for a bounded part of it. The work that a derivation — a recorded, re-executable account of which mechanism was applied under which preconditions — could absorb is the re-derivation and re-plumbing visible in the data: {{M|all|artifact_transition_class:derivation_repair|share|pct1}}% of artifact × transitions required hand re-derivation and {{M|all|artifact_transition_class:adapter_repair|share|pct1}}% adapter repair; cross-repository ports and V1 reimplementations re-expressed known mechanisms by hand. It would help least with the largest single category, correctness repair of integration defects ({{M|all|event_type:repair_correctness|share|pct1}}% of events), unless the derivation also captured the framework invariants those defects violated. This is an inference from repository history, not a measurement of counterfactual effort.

## 10. What would such a derivation need to represent that current systems do not?

From the observed failures, repairs and silent re-derivations:

1. **The operator contract** — shapes, dtypes, layouts, routing and scaling semantics — separately from the kernel, with versioned changes (true contract changes versus newly supported inputs).
2. **The cache and metadata contracts** the kernel shares with the framework (paged block tables, latent cache layout, FP8 scale tensors, plan/run lifecycles, graph-capture buffers).
3. **Each mechanism as a named transformation** with its semantic, hardware and framework preconditions, each tagged `specified`/`derived`/`tested` with a citation.
4. **Profitability conditions** as first-class data (shape regimes, thresholds, target hardware) with the benchmark evidence that justified them.
5. **Numerical and determinism obligations** (tolerances, reduction-order sensitivity).
6. **Dispatch and default policy** with fallbacks, so that a new backend is a derivation of the same contract rather than a new code path.
7. **Provenance edges** (ported from, reimplementation of, repairs) and **failure knowledge** attached to the precondition that failed, not only to the backend that failed.
8. **Delivery facts** (kernel wheel versions, dependency pins) so that "merged" and "released" can be distinguished.

## Five strongest defensible findings

1. **Incremental repair, not replacement, dominates kernel maintenance.** Incremental events are {{M|all|incremental_events|share|pct1}}% of verified events; correctness repair alone is {{M|all|event_type:repair_correctness|share|pct1}}%; {{M|all|transitions_with_repair|share|pct1}}% of release transitions carry a repair.
2. **Code identity is short-lived while mechanisms persist.** Three releases later, only {{C|data/survival-by-releases.csv|kind=artifact_unchanged&lineage=all&k_releases=3&basis=all_present_releases|share|pct1}}% of artifacts are textually unchanged, yet move signatures persist in {{C|data/survival-by-releases.csv|kind=move&lineage=all&k_releases=3&basis=all_present_releases|share|pct1}}% of cases.
3. **Hand re-derivation is the modal non-trivial rebase.** {{M|all|artifact_transition_class:derivation_repair|share|pct1}}% of artifact × transitions required re-deriving kernel internals versus {{M|all|artifact_transition_class:backend_replacement|share|pct1}}% backend replacement.
4. **The most repaired move is a framework-protocol invariant.** CUDA-graph static-metadata adaptation: {{M|L3|move_repair_events:M-L3-cudagraph-static-metadata|value|int}} repair or revert events among {{M|L3|move_repair_events:M-L3-cudagraph-static-metadata|denominator|int}} coded events, on top of a documented guarantee.
5. **Load-bearing assumptions are rarely specification-backed.** {{M|all|assumption_status:specified_guarantee|share|pct1}}% of audited assumptions are specified guarantees; profitability rests on tests and benchmarks.

## Three best lineage stories for the talk

1. **The MoE alignment round trip (L1).** A padding contract born in vLLM, rewritten and optimised in SGLang's kernel library, then ported back to vLLM with explicit credit — and later re-specialised for decode-sized batches. Knowledge travelled, but only because people copied code and wrote down where it came from. See `figures/l1-move-dag.png`.
2. **The death of PagedAttention's kernel and the survival of its ideas (L3).** The kernel that defined vLLM was displaced as default and deleted, but its block-table layout contract and its split-KV/LSE-merge idea live on in every successor. See `figures/l3-artifact-lineage-p1.png` and `figures/l3-move-dag.png`.
3. **MLA: one algebraic move, many backends (L2).** Weight absorption stayed fixed while SGLang cycled through Triton, FlashInfer MLA, FlashMLA, CUTLASS, FA3 and TRT-LLM backends by hardware generation — including adding and then removing CUTLASS MLA. See `figures/l2-release-timeline.png`.

## Five best figures

1. `figures/synthesis-survival-artifact-vs-move.png` — code identity vs mechanism persistence.
2. `figures/synthesis-rebase-classes.png` — how artifacts cross release boundaries.
3. `figures/l1-move-dag.png` — the MoE alignment round trip.
4. `figures/l3-artifact-lineage-p1.png` — PagedAttention and its successors.
5. `figures/synthesis-causes.png` — why kernel code changes.

## Evidence for and against "regenerative kernels"

**Supporting.** Mechanisms outlive their code (finding 2); maintenance is dominated by re-derivation and adapter repair that a regenerable derivation could absorb (finding 3); engineers already regenerate kernels by hand when they port or rewrite (Q3, Q7); backends are routinely treated as replaceable at the dispatch level (Q1).

**Contradicting.** The largest category of work is correctness repair of integration defects, which regeneration from a kernel-level specification would not remove; the knowledge that breaks lives in framework contracts (graph capture, cache layout, scale plumbing), not in kernels; profitability preconditions are tested only, so a regenerator would need the benchmark oracle as much as the specification; long-lived, heavily repaired artifacts show that production teams value continuity; and the acknowledged cross-repository ports show deliberate, reviewed transfer rather than regeneration.

**Net.** The data support regenerating *mechanisms against explicit contracts*, not regenerating kernels from a formula alone.

## Requirements for a future optimization-derivation representation

- Separate operator contract, framework contract and mechanism; version each.
- Record preconditions with their evidence status (specified with citation, derived with the assertion, tested with the test) and platform scope.
- Make profitability conditions and their benchmarks first-class and re-runnable.
- Attach failures and guards to preconditions so a successor inherits them.
- Keep provenance edges (ported from, reimplementation of, repairs, reverts) machine-readable.
- Model dispatch/default policy and fallbacks as part of the derivation.
- Distinguish merged, released and default-enabled states, including kernel-wheel delivery.

## Threats to validity

- **Construct.** "Event", "artifact" and "move" are analytical constructs; the rebase classes (`CODEBOOK.md` §9, v1.2–v1.3) are a model, not repository facts. Move signatures detect identifying code, not intact semantics. Code survival via `git blame -w -M` follows whole-file renames but not cross-file copies.
- **Coding reliability.** One LLM first pass per event with a blind recode of a 36-event sample: event type {{C|recoding/agreement.csv|field=event_type|percent_agreement|pct1}}% (κ = {{C|recoding/agreement.csv|field=event_type|cohen_kappa|dec2}}), explicit ancestry {{C|recoding/agreement.csv|field=ancestry|percent_agreement|pct1}}%, moves {{C|recoding/agreement.csv|field=moves|percent_agreement|pct1}}%. Disagreements were adjudicated and applied only to the sample; codebook clarifications (v1.2) were not re-applied to the full dataset. Model–model, not human.
- **Scope decisions.** Each lineage's boundary (e.g. sparse MLA as a connected branch, NPU/CPU variants) changes counts; boundaries are documented in `cache/task-screen.md` and in excluded-artifact lists.
- **Discovery recall.** Eight strategies plus a cross-reference pass; PR bodies before 2025-09-29 were scanned only for candidates found by other strategies. Unmerged work and private discussion are invisible.
- **Release mapping.** Branch-point ancestry plus PR-number cherry-pick matching; SGLang kernel-wheel delivery modelled through version pins; tags moved after publication are capped at the publication date.
- **Survival analysis.** Kaplan–Meier estimates treat artifact end and first repair as events with censoring at the cutoff; young artifacts make lifetimes conservative.
- **Two repositories, three lineages**; results need not generalise to other engines or kernel libraries.

## Paths

| Artifact | Path |
|---|---|
| Checkpoint | `experiments/kernel-lineage-study/STATUS.json` |
| Methods / codebook | `METHODS.md`, `CODEBOOK.md` |
| Lineage reports | `sglang-moe-lineage.md`, `sglang-mla-lineage.md`, `vllm-attention-lineage.md` |
| Candidates (all, with labels) | `data/candidate-events.csv` |
| Verified events / edges | `data/lineage-events.csv`, `data/lineage-edges.csv` |
| Optimization moves, assumptions, biographies | `data/optimization-moves.csv`, `data/move-assumptions.csv`, `data/move-biographies.csv` |
| Artifacts and snapshots | `data/artifacts.csv`, `data/artifact-snapshots.csv`, `data/artifact-release-presence.csv` |
| Releases and transitions | `data/release-map.csv`, `data/release-transitions.csv`, `data/release-transition-artifacts.csv` |
| Survival | `data/survival-by-releases.csv`, `data/km-artifact-lifetime.csv`, `data/km-first-repair.csv`, `data/artifact-milestones.csv`, `data/cutoff-line-survival.csv`, `data/move-release-presence.csv` |
| Failures | `data/failure-links.csv`, `staging/failures/L*-failures.json` |
| Metrics used in reports | `data/lineage-metrics.csv` |
| External references | `data/external-references.csv` |
| Recoding | `recoding/agreement.csv`, `recoding/recode-pairs.csv`, `recoding/reconciliation.jsonl`, `recoding/reconciliation-summary.csv` |
| Consistency check | `data/consistency-check.json` |
| Figures (PNG/SVG, DOT, Mermaid) | `figures/` — `l{1,2,3}-artifact-lineage-p*.{png,svg}`, `l{1,2,3}-artifact-lineage.{dot,mmd}`, `l{1,2,3}-move-dag.{png,svg,dot,mmd}`, `l{1,2,3}-release-timeline.{png,svg}`, `synthesis-*.{png,svg}` |
| Registries, move catalogs, narratives, batches | `staging/registry/`, `staging/moves/`, `staging/narratives/`, `staging/screen/`, `staging/code/` |
| Scripts / caches | `scripts/`, `cache/` |

## Resume

```powershell
cd C:\Users\tsorensen\Documents\github\RealOptimizationTalk\experiments\kernel-lineage-study
python scripts\status.py show
python scripts\run_all.py
```

`scripts/run_all.py` re-derives every dataset, figure and report from the cached inputs (no network) and ends with the consistency check.
