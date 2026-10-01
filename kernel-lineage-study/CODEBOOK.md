# Codebook — kernel lineage study

Version 1.0 (written before any adjudication). Changes are logged in §12; no
label is ever silently redefined. All coding is performed by LLM coders (the
orchestrating agent and fresh subagents) reading evidence in context; any
agreement statistic is model–model consistency, **not** human inter-rater
reliability.

## 1. Units of analysis

| Unit | Definition | Stored in |
|---|---|---|
| **Candidate** | A first-parent commit (or, for upstream/other repos, a merged PR) surfaced by at least one discovery strategy (§8). | `data/candidate-events.csv` |
| **Event** | A verified candidate that materially changes a tracked artifact, its integration/dispatch/default status, its specification, or the dependency that delivers it — or reverts/relands such a change. One event = one merged change (a squash commit). | `data/lineage-events.csv` |
| **Artifact** | A durable implementation or integration unit with an identity that can persist across renames, moves, splits and vendoring (§2). | `data/artifacts.csv`, `data/artifact-snapshots.csv` |
| **Edge** | A directed, evidence-backed relation between two events (§4). | `data/lineage-edges.csv` |
| **Optimization move** | A named, structurally verifiable transformation that an implementation applies to improve performance (§5). | `data/optimization-moves.csv` |
| **Release transition** | A pair of consecutive public releases of the host repository within which at least one lineage event lands. | `data/release-transitions.csv` |

## 2. Identity rules

1. **Textual identity (artifact continuity).** An artifact persists through a
   commit when its source is carried forward with the same role: an
   in-place edit, a pure move/rename (Git similarity ≥ 50% or an explicit
   move PR), a split where the kernel body moves intact, or a mechanical
   re-packaging (e.g. AOT → JIT build wrapper around the same kernel body).
   Each such carry is a snapshot of the same artifact ID.
2. **New artifact.** A rewrite in a different language/DSL, a new kernel
   body with a different algorithmic structure, a newly vendored/wrapped
   external kernel, or a new dispatch path gets a new artifact ID even if it
   implements the same operator. Its relation to predecessors is expressed
   only through edges, never by reusing the ID.
3. **Conceptual continuity is separate.** Two artifacts share an
   optimization move only if the move is verified structurally in both
   (§5.3). File or name continuity never implies conceptual continuity, and
   conceptual similarity never implies intellectual lineage without explicit
   evidence (comment, PR text, review, commit message, or copied code).
4. **Upstream artifacts.** Kernels that live in another project (FlashInfer,
   FlashMLA, FlashAttention, TensorRT-LLM, AITER, DeepEP, DeepGEMM, CUTLASS)
   are artifacts with `origin=upstream`. The in-tree **adapter** that calls
   them (backend class, wrapper, build rule) is a separate artifact.
   Upstream-side events are recorded only when they are needed to anchor an
   edge (e.g. the FlashInfer MLA plan API that an SGLang backend adopts).
5. **Vendored copies.** A copied upstream source inside the repository is its
   own artifact with a `forked_from`/`ported_from` edge to the upstream.
6. **Generated sources.** Generated kernel instantiations belong to the
   artifact of their generator.
7. **Optional vs default.** Availability (selectable by flag/env/config) and
   default selection (chosen automatically for some hardware/model) are
   tracked separately; a `change_default` event is required to move an
   artifact between them.
8. **Deleted paths.** Deletion ends a textual artifact only if no
   move/rename successor exists (checked with Git rename detection on the
   blob contents and by searching the same commit for the kernel's symbols).
9. **Name collisions.** A path or symbol that merely shares a word with a
   lineage artifact (e.g. SGLang's request `router`, sampling `top_k`,
   DSA/NSA indexer top-k when unrelated to MoE routing) is not part of the
   lineage.

## 3. Event types (A1)

Exactly one primary `event_type` per event; secondary aspects go in the
boolean/text fields (`spec_change`, `integration_change`,
`optimization_move_ids`).

| Code | Definition | Decision rule / boundary |
|---|---|---|
| `introduce` | First appearance of a new artifact (kernel, adapter, backend, dispatch path) in the observed repository. | New artifact ID per §2.2. If the artifact is copied from elsewhere, prefer `port`. |
| `optimize` | Changes an existing artifact's implementation to improve performance with a structurally new technique (tiling, fusion, vectorization, …). | Must name a verified move. A pure parameter change is `retune`. |
| `retune` | Changes numeric tuning parameters (block sizes, num_warps, stages, heuristics thresholds, tuned configs) without changing kernel structure. | Routine per-GPU config JSON additions are **not** events (`related_context_only`, reason `routine_config`); they are counted separately. |
| `port` | Copies or adapts an implementation from another repository, backend, language or hardware target. | Requires evidence of the source (comment, PR text, identical code). |
| `integrate` | Wires an existing (often upstream) kernel into the framework: new backend class, wrapper, build rule, dependency. | Distinct from `introduce` when the kernel body is external. |
| `adapt_framework` | Changes an artifact or adapter because the framework's protocol changed (API refactor, engine V0→V1, metadata builder, CUDA-graph or compile protocol, file reorganization), without intended performance or semantic change. | Pure moves/renames that carry an artifact are `adapt_framework` events only if they change the call protocol; otherwise they are snapshots recorded in `artifact-snapshots.csv` and the candidate is `related_context_only` (reason `pure_move`). |
| `repair_correctness` | Fixes wrong results, crashes, illegal memory access, hangs, races, NaNs, or unsupported-case failures that should have been supported. | Requires evidence of the incorrect behaviour. |
| `repair_performance` | Fixes a performance regression or pathological slowdown introduced earlier. | Requires evidence of the regression (issue, benchmark, PR text). |
| `repair_build_dependency` | Fixes compilation, packaging, dependency pinning, ABI or import failures of an artifact. | |
| `revert` | Undoes a previous event (full or partial). | Always paired with a `reverts` edge. |
| `reland` | Re-applies previously reverted work, possibly modified. | Always paired with a `relands` edge. |
| `deprecate` | Marks an artifact deprecated/legacy or warns against use, without removing it. | |
| `replace` | Routes a use-case away from one artifact to a different artifact (new default or removal of the old path in the same change). | Prefer `change_default` when both remain available. |
| `change_default` | Changes which artifact is selected by default for some hardware/model/configuration. | Includes auto-selection logic changes. |
| `remove` | Deletes an artifact (and it has no textual successor). | Path deletion with a move successor is not `remove` (§2.8). |
| `extend_support` | Extends an artifact to new dtypes, shapes, head sizes, hardware or model families without a new optimization technique. | Added in v1.0 because many verified changes are neither optimizations nor repairs; see §12. |

### Primary change cause (`primary_cause`)

Every event also receives one primary cause (used for synthesis question 5):

| Code | Meaning |
|---|---|
| `specification` | The operator/semantic contract changed (new dtype, new cache layout, new routing semantics, new scaling rule, new model variant). |
| `framework_integration` | The framework's protocol changed (engine, metadata, CUDA graph, compile, API, file layout, dispatcher). |
| `hardware_compiler` | New hardware, compiler, driver or library version, or architecture-specific behaviour. |
| `performance` | Pursuit of speed (new move, retune, default change for speed). |
| `correctness` | Fixing wrong behaviour. |
| `build_dependency` | Packaging, build system, dependency pinning. |
| `maintenance` | Cleanup, deprecation, removal of dead code. |

## 4. Edge types (A2)

Edges point from the earlier event to the later event unless marked
`retrospective=yes` in the evidence column (e.g. a later PR that explicitly
acknowledges an earlier independent implementation).

| Code | Meaning | Minimum evidence |
|---|---|---|
| `textual_successor` | Later event edits/moves the artifact the earlier event produced. | Same artifact ID; path/blob history. |
| `ported_from` | Later implementation copied/adapted from the earlier one (possibly cross-repo). | Comment, PR text or near-identical code. |
| `optimized_from` | Later event optimizes the artifact produced by the earlier event. | Diff on same artifact + performance claim. |
| `reimplementation_of` | Later artifact reimplements the same operator/role from scratch (different code), explicitly replacing or paralleling the earlier one. | PR text or review naming the predecessor, or same dispatch slot. |
| `integrates` | Later event wires in an artifact produced by the earlier (upstream) event. | Import/call site + dependency evidence. |
| `wraps` | Later adapter wraps the earlier kernel without modifying it. | Call site. |
| `forked_from` | Later artifact is a maintained fork of the earlier (e.g. vllm-flash-attn). | Fork repo/commit evidence. |
| `replaces` | Later event makes the earlier artifact unnecessary for a use-case (default switch, removal). | Dispatch/default diff. |
| `reverts` | Later event reverts the earlier. | Revert commit/PR text. |
| `relands` | Later event re-applies the earlier reverted event. | PR text or diff identity. |
| `repairs` | Later event fixes a defect introduced or exposed by the earlier. | Fix PR names the earlier change, or `git blame`/diff shows the fixed lines came from it. |
| `shares_optimization_move` | Two events apply the same verified move (structural similarity) without verified textual ancestry. | Move verified in both (§5.3). |
| `historical_connection_uncertain` | Some evidence suggests a connection but ancestry cannot be established. | Must state what is missing. |

`repairs` was added to the minimum list because failure-to-successor
histories (WP-G) require an explicit defect link.

## 5. Optimization-move taxonomy (A3)

### 5.1 Categories (fixed)

| Category code | Covers |
|---|---|
| `tiling_blocking` | Block/tile shape choices, block-size padding (e.g. aligning token counts to `BLOCK_M`). |
| `work_partitioning` | Assigning work to threads/warps/CTAs; warp-level reductions; per-expert CTA mapping. |
| `multi_block` | Splitting one logical task across multiple CTAs with inter-CTA coordination (atomics, grid sync, two-phase). |
| `vectorization_coalescing` | Vector loads/stores, packed types, coalesced layouts. |
| `smem_register_staging` | Shared-memory/register staging, cumsum in shared memory, register-resident state. |
| `async_pipelining` | cp.async/TMA, multistage pipelines, warp specialization, producer/consumer. |
| `fusion` | Fusing operators or eliminating intermediates (e.g. gating+top-k, top-k+reduce, scaling into epilogue). |
| `layout_transformation` | Changing data/cache layouts (paged KV layout, HND/NHD, absorbed MLA weights, K/V split). |
| `persistent_scheduling` | Persistent kernels, tile schedulers, stream-K. |
| `split_reduction` | Split-K/split-KV, flash-decoding, two-pass reductions with merge (LSE merge). |
| `sparsity_routing` | Expert skipping, sparse dispatch, masked/padded expert layouts, token dispatch. |
| `quantization_scaling` | Quantization, scale fusion, routed-scaling fusion, FP8/FP4 paths. |
| `specialization` | Shape-, head-dim-, dtype- or architecture-specialized variants and templates. |
| `dispatch_plan` | Runtime plan/metadata selection (FlashInfer plan, scheduler metadata, backend choice heuristics). |
| `graph_compile_adaptation` | Changes so a kernel is capturable/replayable in CUDA graphs or compatible with torch.compile. |
| `algebraic_rewrite` | Mathematically equivalent reformulation (e.g. MLA weight absorption, online softmax). |

`algebraic_rewrite` was added because MLA's weight absorption and online
softmax are central moves that fit none of the required categories.

### 5.2 Moves (grounded)

Specific moves (`move_id`, e.g. `M-L1-align-multiblock`) are induced from
verified diffs during coding and registered in `data/optimization-moves.csv`.
A move record states its semantic, hardware and framework preconditions and
the assumption status of each (§6).

### 5.3 Verification rule

A move may be attached to an event only if at least one of the following was
read and shows the move: the diff of the implementation, the implementation
at that commit, PR body/review text **that describes the mechanism** (not only
a speedup), a linked issue, or documentation. Titles alone never suffice.
Two events share a move only if the same mechanism is visible in both.

## 6. Assumption status (A4)

| Code | Meaning |
|---|---|
| `specified_guarantee` | Supported by a cited public specification (CUDA Programming Guide, PTX ISA, framework/library documentation). The citation must have been checked. |
| `derived` | Justified by an explicit static/dynamic argument in the artifact (comment, assertion, proof sketch, bounds check). |
| `tested_only` | Supported only by tests or benchmarks. |
| `undocumented_behavior` | Relies on behaviour not promised by a checked public specification. |
| `unclear` | Cannot be determined from available evidence. |

Every assumption row records the exact source and a ≤ 25-word excerpt.

## 7. Screening labels (WP-B)

| Label | Meaning |
|---|---|
| `verified_lineage_event` | Material change to a tracked artifact / its dispatch / its spec / its delivering dependency, or a revert/reland of one. **Every candidate with this label appears in `lineage-events.csv`.** |
| `related_context_only` | Touches lineage code or topics but does not materially change a tracked artifact (reason codes below). |
| `name_collision` | Matched only through a shared word/path (e.g. request router, sampling top-k). |
| `insufficient_evidence` | Could be relevant but the public record is too thin to decide. |
| `out_of_scope` | Belongs to a different lineage/operator/hardware scope. |

Rejection/reason codes (`rejection_reason`): `routine_config`, `pure_move`,
`caller_only` (model/layer code calls the artifact without changing it),
`test_or_benchmark_only`, `docs_only`, `ci_only`, `logging_typing_style`,
`refactor_no_protocol_change` (v1.1), `dependency_bump_generic` (v1.1:
version bump not shown to be for a lineage kernel), `other_operator`,
`other_lineage:<Lx>`, `other_hardware_scope`,
`unmerged_or_reverted_before_release` (context only), `duplicate`,
`name_collision:<what>`, `thin_record`, `superseded_candidate` (plus free
text).

`adjudication_status`: `screened` (label assigned in stage 1),
`verified` (stage 2 full coding done, event row exists), `rejected_stage2`
(stage 2 overturned a stage-1 positive; reason recorded).

## 8. Discovery strategies (WP-B)

Each candidate records every strategy that surfaced it (`discovery_source`,
`;`-separated): `path_core`, `path_integration+keyword`, `subject_keyword`,
`body_keyword`, `symbol_pickaxe`, `dependency_pin`, `release_notes`,
`corpus:<deep-study file>`, `pr_search`, `cross_reference`,
`upstream_repo`, `registry_seed`.

## 9. Release-transition rebase classes (WP-F)

An analytical classification of how each lineage artifact crossed a release
boundary (`from_release` → `to_release`). One class per (lineage, transition),
chosen as the most invasive class observed among that transition's events:

| Class | Assigned when the transition's lineage events include … |
|---|---|
| `direct_carry` | no verified lineage event on the artifact (artifacts carried unchanged, or changed textually only by commits screened as outside the lineage — see v1.3) or only pure moves. |
| `parameter_retune` | only `retune` events (and/or routine configs). |
| `adapter_repair` | `adapt_framework`, `integrate` or `repair_build_dependency` changes to adapters, with kernel bodies unchanged. |
| `derivation_repair` | `repair_correctness`/`repair_performance` or `extend_support` changes inside kernel bodies (the optimization's derivation had to be redone for new cases). |
| `manual_reimplementation` | a new in-tree implementation (`introduce`/`port`/`optimize` with a new artifact) for an existing role. |
| `backend_replacement` | `replace` or `change_default` to a different artifact/backend. |
| `obsolete` | `remove`/`deprecate` without replacement in the same role. |
| `unclear` | evidence insufficient. |

Order of invasiveness used for the "most invasive" rule:
`obsolete` > `backend_replacement` > `manual_reimplementation` >
`derivation_repair` > `adapter_repair` > `parameter_retune` > `direct_carry`.
The full multiset of classes is also stored, so the rule is transparent.

## 10. Other controlled fields

- `outcome` (event): `merged_survives`, `merged_modified_later`,
  `merged_reverted`, `merged_replaced`, `merged_removed`, `superseded`.
- `survives_at_cutoff`: `code` (event's code textually present at cutoff),
  `concept_only` (code gone but its move present elsewhere),
  `no`, `unclear`. Checked against the cutoff tree.
- `confidence`: `high` (≥ 2 independent evidence types agree), `medium`
  (one strong source, or two weak), `low` (single weak source/inference).
- `hardware_scope`: free list from {`nvidia_sm70`,`sm75`,`sm80`,`sm86`,`sm89`,
  `sm90`,`sm100`,`sm103`,`sm120`, `amd_mi300`,`mi350`/`gfx942`/`gfx950`,
  `amd_rdna`, `cpu`, `npu`, `xpu`, `musa`, `all_cuda`, `unspecified`}.
- `spec_change` / `integration_change`: `none` or a short description;
  a specification change must say which requirement changed (§ C2 list in
  METHODS).
- Evidence types for triangulation (`evidence_types`): `diff`, `code_at_commit`,
  `pr_body`, `review`, `issue`, `test`, `benchmark`, `release_notes`,
  `docs`, `upstream_source`, `revert_record`, `deep_study_record`.

## 11. Failure-to-successor histories (WP-G)

Each history records: failed implementation or move; failure mechanism;
scope of invalidation (`idea`, `implementation`, `adapter`,
`partial_domain`); durable encoding of the lesson (`test`, `guard`,
`comment`, `documented_requirement`, `abstraction`, or
`no durable encoding found`); whether a later implementation repeats the
assumption (with evidence); whether and how the move returned.

## 12. Change log

- v1.0 (initial): added `extend_support` event type, `repairs` edge type,
  `algebraic_rewrite` move category, and `primary_cause` field to the
  required minimum vocabularies, for the reasons stated inline.
- v1.1 (before any screening output existed): added rejection reasons
  `refactor_no_protocol_change` and `dependency_bump_generic`; `pure_move`
  candidates must still name the moved artifact so that the move is recorded
  as an artifact snapshot. Screening decision rule (a)–(e) is stated verbatim
  in `cache/task-screen.md`.
- v1.2 (after the I1 reconciliation; applied to the 36 reconciled events
  only, not retroactively to the full dataset — see METHODS §9): two boundary
  clarifications surfaced by the adjudicator. (1) *Enabling an existing
  optimized path for a new domain* (new dtype/shape/model/hardware) is
  `extend_support`, not `optimize`, unless the diff adds or changes a
  mechanism; PR titles saying "perf" do not decide it. (2) Primary-cause
  precedence for new dtype/shape/mode support: `specification` when the
  operator contract changes, `hardware_compiler` only when a specific
  hardware or library capability is the decisive reason. `rebase_class`
  derivation rules (§9) were operationalised per artifact × transition in
  `scripts/analysis.py`; `optimize`/`extend_support`/`repair_*` on an existing
  kernel map to `derivation_repair`, `adapt_framework`/`integrate`/
  `repair_build_dependency` to `adapter_repair`, `replace`/`change_default` to
  `backend_replacement`, `remove`/`deprecate` to `obsolete`, and
  `introduce`/`port` of an artifact whose registry predecessor is a
  `reimplementation_of`/`replaces` relation to `manual_reimplementation`.
- v1.3 (final pipeline review; no recoding): operational definition of
  `direct_carry` made explicit. The class is assigned when no verified lineage
  event on the artifact lands in the transition. Every commit touching a
  lineage path was a candidate (source-path strategy), so an artifact whose
  files changed textually without a verified event was changed only by commits
  screened as `related_context_only`/`out_of_scope` (for example shared-file
  refactors). Such observations stay `direct_carry` but are counted separately
  (`artifact_transition_direct_carry_textually_modified` in
  `data/lineage-metrics.csv`); textual identity is measured independently in
  `data/survival-by-releases.csv` (`artifact_unchanged`).
- Upstream-anchor candidates (`upstream_repo`, `staging/code/L1/batch-901-*`)
  are hand-verified in one pass, like `cross_reference` batch-900 candidates;
  their `adjudication_status` is `verified` (or `adjudicated_upstream_anchor`
  if rejected).
