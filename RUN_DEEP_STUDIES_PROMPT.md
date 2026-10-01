# Deep empirical follow-up: kernel delivery, failures, and papers to production

You are continuing a completed preliminary study for Tyler Sorensen's
"Optimize the Optimization Pipeline" talk. The first run collected broad
metadata and produced:

- `experiments\system-evolution-report.md`
- `experiments\METHODS.md`
- `experiments\STUDY_OUTPUTS.md`
- datasets and figures under `experiments\results\`
- a checkpoint at `experiments\results\study-status.json`

Read those files first, plus all plans under `experiments\studies\`. Do not
rerun completed broad collection merely to look busy. This follow-up exists
because the first run stopped at a shallow but valid report. Your job is to
perform the expensive empirical work it deferred.

Work autonomously until ALL required work packages below are complete or an
external blocker is documented with exact recovery state. Do not redefine a
required package as optional. Do not satisfy a manual-adjudication requirement
with keyword counts alone.

Project root:

`C:\Users\tsorensen\Documents\github\RealOptimizationTalk`

# Scientific goals

Produce defensible answers to four questions:

1. What actually happens to performance/kernel PRs on their way into vLLM and
   SGLang, and what downstream evidence and reviewer work do they require?
2. How do optimized kernels fail in these systems, and which validation
   technique would plausibly have caught each confirmed failure?
3. Of the papers at recent ML-systems and architecture conferences, how many
   propose kernel-style optimization, and what observable deployment evidence
   do they achieve?
4. Conversely, where do the kernel families deployed in vLLM and SGLang come
   from: papers, vendor libraries, company engineering, or community work?

# Non-negotiable output layout

Create:

- `experiments\deep-study\STATUS.json`
- `experiments\deep-study\METHODS.md`
- `experiments\deep-study\kernel-pr-delivery-report.md`
- `experiments\deep-study\kernel-failures-report.md`
- `experiments\deep-study\paper-to-production-report.md`
- `experiments\deep-study\production-kernel-provenance-report.md`
- `experiments\deep-study\SYNTHESIS.md`
- `experiments\deep-study\data\` for derived/cached data
- `experiments\deep-study\figures\`
- `experiments\deep-study\scripts\`
- `experiments\deep-study\coding\` for codebooks and adjudication records

Do not overwrite the first study's results. Reuse its cached data and cite it.

# Checkpointing

Create `experiments\deep-study\STATUS.json` before any network work. Update it
atomically after every batch of no more than 100 PRs, 10 confirmed failures, or
10 papers.

The status file must include:

- repository HEAD SHAs and study cutoff
- each work package and subtask: pending/in_progress/complete/blocked
- exact counts completed and required
- API cursors and cache paths
- the exact next record or batch to process
- failures, rate-limit resets, and recovery instructions
- output files produced and last sanity check

On startup, read the status file, verify claimed outputs, and resume. A new
agent with no conversation history must be able to continue.

# Work package A — Deep PR delivery study

Repositories:

- `vllm-project/vllm`
- `sgl-project/sglang`

Window: the most recent complete 12 months ending at the frozen cutoff from the
first study.

## A1. Complete high-precision performance-PR population

Build a high-precision population of kernel/performance PRs from the COMPLETE
recent PR metadata, not only the previous 600-PR sample. Use:

- `[Kernel]`, `[Perf]`, and equivalent title tags
- kernel implementation paths
- explicit optimization claims
- known project conventions

Manually adjudicate every ambiguous candidate. At minimum:

- inspect every candidate currently counted in the open performance queue
  (the first run reported 589 vLLM and 345 SGLang title candidates);
- record `confirmed_performance`, `not_performance`, or `uncertain`;
- include a one-sentence rationale and evidence source;
- never publish the preliminary queue counts without this adjudication.

Required output:

`data\performance-pr-population.csv`

Required fields:

`repo,number,state,title,created,merged,closed,classification,evidence,rationale,confidence`

## A2. Full metadata analysis

For every confirmed performance PR and every available non-performance PR in
the same window, calculate from complete metadata:

- time to merge
- open/closed/merged state
- additions, deletions, changed files
- labels
- author type (human/bot only)
- whether later reverted or superseded

Use full-population statistics for merge time and queue age. Do not use the old
600-PR sample where complete metadata suffices.

Use survival analysis with open PRs censored, not only medians among merged PRs.
Report Kaplan-Meier curves and effect sizes with bootstrap confidence
intervals. Avoid causal language.

## A3. Detailed review corpus

Fetch and cache full review/comment/timeline content for:

- ALL confirmed kernel/performance PRs if the population is at most 2,500 per
  repository; otherwise a reproducible stratified sample of 1,000 per repo;
- a matched comparison corpus of at least 500 non-performance PRs per repo,
  matched on month, PR size, and state.

This is a required floor, not a target that may be reduced for convenience.

For each detailed PR, code:

- microbenchmark evidence
- end-to-end latency/throughput evidence
- accuracy/model-quality evaluation
- numerical/correctness tests
- multi-hardware/backend evidence
- memory-usage evidence
- compile/CUDA-graph/torch.compile compatibility
- reviewer requests for tests
- reviewer requests for another backend/hardware
- integration concerns
- maintenance/complexity concerns
- benchmark-methodology concerns
- documentation concerns
- number of substantive review rounds
- time to first non-author human response

For content coding, use an explicit codebook and preserve short evidence
snippets with URL/PR references. Keyword prefilters are allowed, but final
positive labels require contextual adjudication. Validate at least 200 coded
records per repository against a fresh second pass and report agreement. This
is model-model/adjudication agreement, not human inter-rater agreement; label
it honestly.

## A4. Codified versus undocumented rules

Start from the first study's contribution-rule audit. For the detailed corpus:

- measure compliance with mechanically observable rules;
- identify reviewer requests that recur in at least 10 distinct performance
  PRs but do not appear in contribution docs or templates;
- give exact, anonymized examples and links;
- report how many requirements appear before merge that contributors could not
  have learned from the written rules.

Do not profile or rank individual contributors/reviewers.

## A5. PR case histories

Write 12 carefully verified case histories:

- 4 performance PRs that merged quickly
- 4 that took a long time or many review rounds
- 4 abandoned, reverted, or superseded

For each, reconstruct:

`proposal → evidence → review concerns → revisions → outcome → release/adoption`

These are intended for slides. Verify every fact against the PR timeline and
linked commits/releases.

# Work package B — Confirmed kernel failures and reverts

Window: 24 months ending at the frozen cutoff.

## B1. Revert census

Find every revert or explicit rollback in vLLM and SGLang during the window.
Confirm from commit/PR/issue context whether it reverted:

- kernel/performance work
- kernel correctness
- hardware/backend support
- other work

Do not infer a revert solely from a title if context contradicts it.

Required output:

`data\confirmed-reverts.csv`

## B2. Confirmed kernel-correctness corpus

Build at least 150 confirmed kernel-correctness fixes across the two repos,
unless the complete 24-month population is smaller. A case counts only if there
is a fixing commit/PR or maintainer confirmation.

Code each confirmed case:

- numerical/precision/tolerance
- nondeterminism/race/synchronization
- warp/subgroup participation or mask
- shape/alignment/edge case
- memory safety/OOB/illegal access
- hardware/compiler-specific
- integration/backend selection/CUDA graphs
- performance regression presented as correctness or availability
- other

Also record:

- affected hardware/backend
- time from introducing commit to fix where traceable
- whether a regression test was added
- whether the original PR reported correctness evidence
- whether the failure escaped CI
- whether it caused revert, disablement, or fallback

## B3. Validation counterfactual

For every confirmed case, code which technique plausibly could have detected it:

- more randomized/edge-case tests
- hidden input distributions
- sanitizer/memory checker
- determinism or schedule-perturbation testing
- formal functional equivalence
- WarpDRF/static race analysis
- end-to-end serving tests only
- hardware-matrix CI
- not enough information

This is a counterfactual judgment. Preserve rationale, have a second pass
adjudicate every case, and report agreement rather than pretending certainty.

## B4. Failure case studies

Select and deeply reconstruct at least 10 cases:

- at least 2 concurrency/race cases if the corpus contains them
- at least 2 tolerance/numerical cases
- at least 2 hardware-specific cases
- at least 2 integration-only failures

Tie them carefully to RESOLVE, WarpDRF, SIMT-Step, or SWE-Serve-style E2E
testing. Do not claim a method would certainly have found a bug unless evidence
supports that.

# Work package C — Complete conference paper census and deployment ladder

Conferences:

- MLSys 2025 — all 61 papers
- ASPLOS 2025 — all research papers from official proceedings

Do not stop after title/abstract keyword counts.

## C1. Census

For every paper:

- collect canonical title, authors, venue, URL, abstract, artifact links;
- read at least abstract, introduction, contributions, and evaluation summary;
- classify whether it proposes a kernel-style optimization under the broad
  definition in `experiments\studies\04-paper-to-production.md`;
- record category: kernel, fusion family, precision/format, scheduling/layout,
  compiler/DSL, agentic generation, or not kernel-style;
- preserve a 1–3 sentence rationale.

Adjudicate every positive and every ambiguous case in a second pass.

## C2. Evaluation audit for every positive

Record:

- microbenchmark only versus end-to-end evaluation
- baseline and version
- number and type of hardware targets
- correctness/accuracy validation
- public code
- artifact-evaluation badges
- claimed production deployment

## C3. Deployment evidence

Place every positive on the L0–L5 evidence ladder:

- L0 no public code
- L1 code released
- L2 maintained after publication
- L3 adopted by a kernel library
- L4 merged into a serving/training framework
- L5 default/documented/release-noted
- S self-reported production, tracked separately

Search systematically using:

- exact title
- technique/system name
- arXiv/DOI
- artifact repository
- author handles
- citations or "adapted from" comments
- GitHub PRs/issues/code in vLLM, SGLang, TensorRT-LLM, llama.cpp, PyTorch,
  FlashInfer, xFormers, AITER, Megatron, and DeepSpeed

Every L3–L5 label needs a URL and quoted evidence. Hand-check all positives and
a random sample of at least 20 negatives per conference to estimate missed
adoption. Do not equate "paper code runs" with deployment.

## C4. Adoption latency

For L3–L5 papers, estimate:

- paper-preprint date
- first public code date
- first downstream PR date
- merge date
- first documented/default release where applicable

Produce a Sankey/funnel figure:

`all papers → kernel-style papers → code → maintained → library → framework → default`

and adoption-latency distributions.

# Work package D — Reverse provenance of deployed kernels

Enumerate current kernel/backend families in vLLM and SGLang, at minimum:

- attention backends
- fused MoE
- quantized GEMM/linear paths
- normalization/fused activations
- sampling
- speculative decoding kernels
- custom collectives

For each family, record:

- implementation/backend name
- source path
- supported hardware
- upstream library or vendored source
- cited paper or system, if any
- provenance class: academic paper, vendor library, company engineering,
  community contribution, unclear
- first introduction PR/commit
- current default/selectable/experimental status

Verify from code and history. Do not rely only on arXiv URLs: search names,
comments, git history, documentation, and introducing PRs.

Required output:

`data\production-kernel-provenance.csv`

Include at least 40 distinct families/backends across the two repositories, or
explain with an exhaustive enumeration why fewer exist.

# Analysis quality requirements

- Separate complete-population results from sampled/coded results.
- Report denominators everywhere.
- Bootstrap uncertainty for coded proportions and adoption rates.
- Preserve source URLs and concise evidence snippets.
- Mark self-reported production separately from observable adoption.
- Treat one-agent second-pass agreement as adjudication consistency, not human
  inter-rater reliability.
- Never invent missing review, deployment, or correctness evidence.
- Run an exact consistency check linking every executive-summary number to a
  generated CSV row.
- Label findings preliminary if the evidence does not meet the stated gate.

# Minimum workload gates

Do not produce a final answer until all of these are true:

- every open performance-PR candidate from the first study has been adjudicated;
- detailed review content has been collected for the required performance and
  matched comparison corpora;
- at least 12 PR case histories are complete;
- the 24-month revert census is complete;
- at least 150 confirmed kernel-correctness fixes are coded, or the exhaustive
  population is documented as smaller;
- at least 10 failure case histories are complete;
- all MLSys 2025 papers are classified;
- all ASPLOS 2025 research papers are classified;
- every kernel-style positive has an evaluation audit and deployment score;
- negative adoption checks are complete;
- at least 40 deployed kernel/backend families have provenance records;
- all five reports plus `SYNTHESIS.md` exist;
- every numerical claim passes the consistency check.

If GitHub rate limits intervene, checkpoint the exact cursor, do independent
paper/code work while waiting, then resume. Rate limiting is not a reason to
end the run while other required packages remain.

# Final synthesis

`SYNTHESIS.md` must answer:

1. Is kernel discovery a large or small fraction of real engineering work?
2. What consumes the rest of the work?
3. What evidence and reviewer labor turn an optimization into a merge?
4. What kinds of kernel bugs escape existing validation?
5. How much published kernel research leaves observable traces in production?
6. How much production kernel engineering traces back to papers?
7. Which findings most strongly support or contradict the Optimization Gap?

End with:

- the five strongest defensible findings;
- the five best figures for the talk;
- three case histories suitable for narration;
- open limitations;
- exact paths to all reports and datasets;
- exact resume command.

