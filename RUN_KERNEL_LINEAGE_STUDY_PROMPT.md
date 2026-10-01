# Kernel lineage study: how optimization knowledge survives framework evolution

You are conducting a long-running empirical software-engineering and programming-
languages study for Tyler Sorensen's "Optimize the Optimization Pipeline" talk.
Work autonomously until every required work package and validation gate below is
complete, or an external blocker is documented with exact recovery state.

This is not a brainstorming task and not a keyword-counting exercise. Reconstruct
verified histories from source code, Git history, pull requests, issues, releases,
tests, documentation, and the existing deep-study datasets. The goal is to learn
whether production repositories preserve optimization knowledge or merely preserve
kernel code until it becomes obsolete.

Project root:

`C:\Users\tsorensen\Documents\github\RealOptimizationTalk`

Frozen study cutoff:

`2026-09-28T23:59:59Z`

Do not include events after the cutoff in quantitative results. Record repository
HEADs and release/tag mappings used by the study.

## Read first

Read these before collecting new data:

1. `README.md`
2. `talk\talk-idea.md`
3. `experiments\system-evolution-report.md`
4. `experiments\deep-study\METHODS.md`
5. `experiments\deep-study\SYNTHESIS.md`
6. `experiments\deep-study\kernel-pr-delivery-report.md`
7. `experiments\deep-study\kernel-failures-report.md`
8. `experiments\deep-study\production-kernel-provenance-report.md`
9. `experiments\deep-study\data\production-kernel-provenance.csv`
10. `experiments\deep-study\data\kernel-correctness-cases.csv`
11. `experiments\deep-study\data\confirmed-reverts.csv`
12. `experiments\deep-study\data\performance-pr-population.csv`
13. `experiments\deep-study\data\review-coding.csv`
14. `experiments\deep-study\STATUS.json`

Reuse the existing blobless clones and cached GitHub data under `experiments\data\`
and `experiments\deep-study\data\`. Fetch additional public data only when needed.
Do not rerun completed broad studies.

# Central research question

When a production GPU kernel evolves across framework releases, what is actually
preserved?

Distinguish:

- textual kernel code;
- the mathematical/operator specification;
- the framework integration protocol;
- optimization decisions and transformations;
- hardware and compiler assumptions;
- correctness and performance evidence;
- lessons learned from failures and reverts.

The main hypothesis to investigate, not assume, is:

> Kernel implementations are frequently replaced or repaired, while reusable
> optimization knowledge remains implicit, is only partially transported to
> successor implementations, and is sometimes rediscovered along with previously
> known failures.

# Required lineages

Reconstruct all three lineages through the frozen cutoff:

## L1. SGLang MoE alignment, routing, top-k, and fusion

Cover at minimum:

- MoE alignment kernels and multi-block execution;
- grouped top-k and routing;
- routed scaling and `topk_reduce` fusion;
- fused MoE implementations and backend changes;
- DeepEP, FlashInfer, CUTLASS, Triton, and other relevant paths;
- quantized/BF16/FP8/FP4 variants where they alter the lineage;
- relevant framework integration, correctness repairs, reverts, and relands.

## L2. SGLang MLA, FlashInfer MLA, and FlashMLA

Cover at minimum:

- initial MLA implementations and backend selection;
- FlashInfer MLA plans and decode/prefill specialization;
- FlashMLA introduction and dependency/build integration;
- CUDA graph and compilation interaction;
- backend API/configuration changes;
- FA3/FA4 or hardware-specific successors where historically connected;
- performance regressions, correctness fixes, reverts, and relands.

## L3. vLLM attention implementation family

Cover at minimum:

- custom PagedAttention;
- FlashAttention integration and major version/specialization changes;
- FlashInfer attention;
- TensorRT-LLM/XQA decode paths;
- Triton paged/flash/FlashInfer-compatible attention;
- relevant ROCm/AITER branches;
- MLA branches when they share or replace optimization ideas;
- changes in default/optional dispatch and framework interfaces.

L3 is expected to be a branching family rather than a single linear kernel.
Represent that honestly.

If investigation proves that one named component is not historically connected
to the claimed lineage, retain it as an explicit independent branch rather than
inventing ancestry.

# Non-negotiable output layout

Create and maintain:

- `experiments\kernel-lineage-study\STATUS.json`
- `experiments\kernel-lineage-study\METHODS.md`
- `experiments\kernel-lineage-study\CODEBOOK.md`
- `experiments\kernel-lineage-study\sglang-moe-lineage.md`
- `experiments\kernel-lineage-study\sglang-mla-lineage.md`
- `experiments\kernel-lineage-study\vllm-attention-lineage.md`
- `experiments\kernel-lineage-study\SYNTHESIS.md`
- `experiments\kernel-lineage-study\data\candidate-events.csv`
- `experiments\kernel-lineage-study\data\lineage-events.csv`
- `experiments\kernel-lineage-study\data\lineage-edges.csv`
- `experiments\kernel-lineage-study\data\optimization-moves.csv`
- `experiments\kernel-lineage-study\data\release-map.csv`
- `experiments\kernel-lineage-study\data\artifact-snapshots.csv`
- `experiments\kernel-lineage-study\data\release-transitions.csv`
- `experiments\kernel-lineage-study\data\consistency-check.json`
- `experiments\kernel-lineage-study\figures\`
- `experiments\kernel-lineage-study\scripts\`
- `experiments\kernel-lineage-study\cache\`

Do not overwrite existing deep-study datasets. Cite and link reused records.

# Checkpointing and recovery

Create `STATUS.json` before any network collection or lengthy analysis. Update it
atomically:

- after every 25 candidate events screened;
- after every 10 verified events;
- after every release transition;
- after every completed report section;
- before and after any network batch.

The status file must include:

- schema version, cutoff, repository HEADs, and current phase;
- every work package and lineage with status, completed count, required count,
  exact next record, and output paths;
- release/tag cursors and Git/GitHub query state;
- cache paths and source URLs already fetched;
- unresolved identity/ancestry questions;
- failures, rate-limit resets, and exact recovery instructions;
- last successful consistency check;
- timestamps.

On startup, resume from `STATUS.json` if it exists. Verify claimed output files
before trusting status. Never discard completed coding merely to restart cleanly.

# Work package A: ontology and codebook

Before adjudicating events, define controlled vocabularies.

## A1. Event types

At minimum:

- `introduce`
- `optimize`
- `retune`
- `port`
- `integrate`
- `adapt_framework`
- `repair_correctness`
- `repair_performance`
- `repair_build_dependency`
- `revert`
- `reland`
- `deprecate`
- `replace`
- `change_default`
- `remove`

## A2. Edge types

At minimum:

- `textual_successor`
- `ported_from`
- `optimized_from`
- `reimplementation_of`
- `integrates`
- `wraps`
- `forked_from`
- `replaces`
- `reverts`
- `relands`
- `shares_optimization_move`
- `historical_connection_uncertain`

## A3. Optimization-move taxonomy

Develop a grounded taxonomy from the data. It should cover at least:

- tiling/blocking;
- thread/warp/CTA work partitioning;
- multi-block execution;
- vectorization and memory coalescing;
- shared-memory/register staging;
- asynchronous copy and software pipelining;
- fusion and intermediate elimination;
- layout transformation;
- persistent scheduling;
- split-K/reduction decomposition;
- sparsity/routing/expert skipping;
- quantization/scaling fusion;
- shape or architecture specialization;
- dispatch and plan selection;
- compilation/graph-capture adaptation.

Do not label a conceptual optimization move from a title alone. Verify it from
the diff, implementation, PR body, review, linked issue, or documentation.

## A4. Assumption status

For each important move, classify every stated or inferred assumption as:

- `specified_guarantee`: supported by a cited public language, CUDA, PTX, framework,
  or library specification;
- `derived`: justified by an explicit static/dynamic argument in the artifact;
- `tested_only`: supported by tests or benchmarks but not a cited guarantee;
- `undocumented_behavior`: appears to rely on behavior not promised by a public
  specification;
- `unclear`.

Preserve the exact source and a short evidence excerpt. Do not claim NVIDIA leaves
behavior unspecified unless the relevant public documentation was actually checked.

# Work package B: exhaustive candidate discovery

For each lineage, build an over-inclusive candidate set using:

- full Git history, including renames and deleted paths;
- commit subjects and bodies;
- source-path history;
- PR titles, bodies, reviews, comments, and linked issues;
- release notes and tags;
- dependency and submodule changes;
- backend registries, dispatch code, tests, benchmarks, and docs;
- the existing provenance, performance-PR, failure, and revert corpora;
- upstream projects when an in-tree kernel is replaced or vendored.

Use multiple discovery strategies. Keyword search alone is inadequate.

Record every candidate in `candidate-events.csv`, including rejected candidates.
Required fields:

`candidate_id,lineage,repo,date,commit_sha,pr_number,title,source_paths,discovery_source,screening_label,rejection_reason,adjudication_status,evidence_urls`

Screening labels:

- `verified_lineage_event`
- `related_context_only`
- `name_collision`
- `insufficient_evidence`
- `out_of_scope`

Inspect every candidate manually in context. A generated shortlist may assist but
may not perform final adjudication.

# Work package C: reconstruct artifact and specification histories

## C1. Identify artifacts

Assign durable artifact IDs to implementations and integration adapters. Track
renames, moves, splits, rewrites, vendoring, generated sources, and deletion.

For each meaningful snapshot, record:

- artifact ID and lineage;
- release/tag and commit;
- source paths;
- implementation language;
- upstream/vendor relationship;
- supported hardware;
- framework-facing call signature and metadata;
- dispatch/default status;
- tests and benchmarks;
- direct predecessor/successor evidence;
- content hash or Git blob identifier where available.

## C2. Reconstruct specification evolution

Do not reduce "specification" to a mathematical formula. Reconstruct observable
requirements from call sites, reference implementations, tests, interfaces, and
documentation:

- mathematical operation;
- accepted shapes, layouts, and dtypes;
- numerical behavior;
- cache and tensor representation;
- quantization/scaling semantics;
- distributed-execution semantics;
- graph-capture/compilation protocol;
- stream, synchronization, mutation, and aliasing behavior;
- error/fallback behavior.

Record when these requirements change and cite evidence. Separate an actual
specification change from a previously unsupported case or an implementation bug.

# Work package D: verified event and lineage graph

Populate `lineage-events.csv` with at least:

`event_id,lineage,date,repo,release,commit_sha,pr_number,event_type,artifact_ids,parent_event_ids,spec_change,integration_change,optimization_move_ids,hardware_scope,performance_claim,correctness_evidence,outcome,survives_at_cutoff,confidence,evidence_excerpt,evidence_urls`

Populate `lineage-edges.csv` with:

`edge_id,lineage,from_event_id,to_event_id,edge_type,evidence,confidence,evidence_urls`

Rules:

- Dates, PRs, SHAs, and release inclusion must be verified.
- Do not infer parentage solely from chronological proximity.
- Represent branches and uncertain edges.
- A revert and later reland are separate events.
- Distinguish code survival from optimization-move survival.
- Distinguish optional availability from default selection.
- Record negative or null findings.

Minimum floor: 20 verified events per lineage and at least 75 verified events
overall. If exhaustive investigation yields fewer for a lineage, document the
complete candidate census and explain why the true population is smaller.

# Work package E: optimization biographies

Populate `optimization-moves.csv` with:

`move_id,name,category,first_observed_event,description,semantic_preconditions,hardware_preconditions,framework_preconditions,assumption_status,performance_evidence,correctness_evidence,later_uses,known_failures,current_status,confidence,evidence_urls`

Trace at least 15 distinct optimization moves overall, with at least four from
each lineage.

For at least 12 moves, write a short biography:

1. where the move first appears in the observed lineage;
2. what problem it solves;
3. its explicit and implicit preconditions;
4. whether it survives textually or conceptually;
5. where it is ported, repeated, replaced, or rediscovered;
6. failures or reverts associated with it;
7. whether later work acknowledges the earlier implementation or lesson.

Do not call two implementations the same move merely because both claim speedups.
Establish structural or semantic similarity.

# Work package F: release-pinned evolution

Build `release-map.csv` from verified Git tags and release dates. Assign every
event to the first containing public release where possible. Mark unreleased,
reverted-before-release, and ambiguous cases explicitly.

For each consecutive release transition that intersects a lineage, populate
`release-transitions.csv`:

`lineage,from_release,to_release,spec_delta,framework_delta,artifacts_carried,artifacts_modified,artifacts_added,artifacts_removed,moves_preserved,moves_repaired,moves_lost,moves_new,rebase_class,evidence,confidence`

Use these `rebase_class` values:

- `direct_carry`
- `parameter_retune`
- `adapter_repair`
- `derivation_repair`
- `manual_reimplementation`
- `backend_replacement`
- `obsolete`
- `unclear`

This classification is an analytical model, not a fact emitted by the repository.
State the inference and supporting evidence.

Calculate:

- artifact survival across 1, 2, and 3 subsequent releases;
- optimization-move survival across 1, 2, and 3 releases;
- number and type of repairs per release transition;
- replacement versus incremental-maintenance frequency;
- time from introduction to first repair, revert, replacement, and removal;
- repeated or apparently rediscovered moves;
- repeated failure assumptions, if verified;
- proportion of changes attributable primarily to specification, framework
  integration, hardware/compiler, performance, or correctness.

Use survival analysis only where event and censoring definitions are defensible.
Otherwise use transparent descriptive statistics.

# Work package G: failures, reverts, and missing durable knowledge

Link all relevant records from:

- `experiments\deep-study\data\kernel-correctness-cases.csv`
- `experiments\deep-study\data\confirmed-reverts.csv`
- linked PRs/issues and subsequent repairs.

For every relevant failure or revert, determine:

- what implementation or move failed;
- whether the failure invalidated the optimization idea, one implementation, an
  integration adapter, or only part of its domain;
- whether the lesson became a test, guard, comment, documented requirement, or
  reusable abstraction;
- whether a later implementation appears to repeat the same assumption;
- whether the move returned and in what form.

Do not infer that a lesson was forgotten merely because no comment is present.
Use `no durable encoding found` rather than claiming absence of organizational
memory.

Produce at least eight detailed failure-to-successor histories, or exhaustively
document that fewer linked histories exist.

# Work package H: visualizations and reports

For each lineage report, include:

1. scope and identity rules;
2. a chronological table with links;
3. an artifact-lineage DAG;
4. an optimization-move DAG;
5. a release-overlaid timeline;
6. specification and framework-boundary changes;
7. optimization biographies;
8. failures, reverts, relands, and replacements;
9. what survives at the cutoff;
10. limitations and uncertain ancestry.

Generate legible PNG and SVG figures where practical. Also include Mermaid source
or Graphviz/DOT source so diagrams can be edited. Avoid unreadable one-page graphs:
split dense lineages into phases or branches.

The reports must narrate mechanisms, not merely enumerate events.

# Work package I: validation

## I1. Independent recoding

Draw a reproducible stratified sample of at least 30 verified events, with at
least 10 per lineage. Recode event type, ancestry edge, optimization move, and
primary change cause in a fresh pass without copying first-pass labels.

Report agreement and reconcile every disagreement. Describe this honestly as
model/model or repeated-agent consistency, not human inter-rater reliability.

## I2. Source triangulation

For every major claim and every optimization biography, require at least two
independent evidence types where available, such as:

- code/diff plus PR discussion;
- code plus test;
- PR plus release/tag;
- upstream source plus downstream integration;
- revert plus later repair.

If only one source exists, lower confidence and say so.

## I3. Exact consistency checks

Write and run a consistency script that verifies:

- unique IDs and valid foreign keys;
- chronological edge ordering, except explicit retrospective edges;
- every cited PR/commit URL corresponds to the recorded repository and ID;
- release dates and first-containing-release assignments;
- all report counts reproduce from CSVs;
- every optimization biography references valid events and moves;
- every executive-summary quantitative claim maps to generated data;
- no event occurs after the frozen cutoff.

Write results to `data\consistency-check.json`. The final run must pass, except
for explicitly enumerated non-fatal warnings.

# Synthesis questions

`SYNTHESIS.md` must answer:

1. Are production kernels maintained incrementally, repeatedly replaced, or both?
2. How long do kernel artifacts survive compared with optimization ideas?
3. Which optimization moves transport successfully across releases and backends?
4. Which moves repeatedly require manual repair?
5. How much maintenance is caused by changed operator semantics versus changed
   framework protocols versus hardware/compiler evolution?
6. Do failures and reverts produce durable, reusable knowledge?
7. Are optimizations rediscovered? If so, is prior lineage acknowledged?
8. Where do implementations rely on specified, derived, tested-only, undocumented,
   or unclear GPU behavior?
9. Would a replayable optimization derivation plausibly reduce observed work?
10. What would such a derivation need to represent that current systems do not?

End with:

- the five strongest defensible findings;
- the three best lineage stories for the talk;
- the five best figures;
- evidence supporting and contradicting "regenerative kernels";
- concrete requirements for a future optimization-derivation representation;
- threats to validity;
- exact paths to every report, figure, and dataset;
- exact resume command.

# Scientific discipline

- Preserve uncertainty and competing interpretations.
- Do not claim causality from repository history.
- Do not treat commit or PR titles as ground truth.
- Do not equate file continuity with conceptual continuity.
- Do not equate similar code with shared intellectual lineage without evidence.
- Do not claim an optimization is specification-tight without checking the cited
  public specification and the implementation.
- Do not claim an implementation was deployed merely because it merged.
- Do not name or rank ordinary contributors in aggregate findings; links to public
  technical discussions are acceptable.
- Quote minimally and preserve URLs.
- Clearly separate repository-observable facts from analytical interpretation.
- Record unsuccessful searches and null results.

# Minimum completion gates

Do not produce a final response until all are true:

- every required file in the output layout exists;
- all three candidate populations have been exhaustively screened;
- at least 20 verified events exist per lineage and 75 overall, unless an
  exhaustive smaller population is documented;
- every verified event has a source URL and concise evidence;
- release mapping and first-containing-release analysis are complete;
- at least 15 optimization moves are traced, with at least four per lineage;
- at least 12 optimization biographies are complete;
- at least eight failure-to-successor histories are complete or an exhaustive
  smaller population is documented;
- release transitions are classified;
- artifact and optimization survival analyses are complete;
- all lineage, optimization, and release diagrams are generated;
- independent recoding and reconciliation are complete;
- exact consistency checks pass;
- all three lineage reports and `SYNTHESIS.md` are complete;
- `STATUS.json` marks every package complete or documents a genuine external
  blocker with exact recovery state.

If GitHub rate limits intervene, checkpoint the exact cursor, continue independent
Git/source/report work, and resume network collection later. A rate limit is not
a reason to end while other required packages remain.

Begin by creating or validating `STATUS.json`, inventorying the existing local
clones and datasets, and writing the codebook. Then proceed autonomously.
