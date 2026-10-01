# Methods — kernel lineage study

*How optimization knowledge survives framework evolution in SGLang and vLLM.*
This study extends the deep study (`experiments/deep-study/`) and reuses its
blobless clones, caches and datasets read-only. Every derived record, script,
cache and adjudication artefact is under `experiments/kernel-lineage-study/`
(`$S` below). The resumable checkpoint is `$S/STATUS.json`.

## 1. Frozen inputs

| Item | Value |
|---|---|
| Study cutoff | `2026-09-28T23:59:59Z` — no event after it enters any count |
| SGLang cutoff commit | `89e1316eae9aae634c174c040415289fe4484aee` (committed 2026-09-28T23:41:07Z) |
| vLLM cutoff commit | `c68eb98b69f4863d94318fbe9dedf654be0c5945` (committed 2026-09-28T23:52:29Z) |
| Rule for cutoff commit | `git rev-list -1 --first-parent --before=2026-09-28T23:59:59Z main` |
| Local HEADs of reused clones | SGLang `79cafec013d0…` and vLLM `7230dfea501b…` (both *after* the cutoff; never used as analysis endpoints) |
| Release list | GitHub Releases API (non-draft, `v*` tags, published ≤ cutoff): SGLang 50 (49 final), vLLM 106 (97 final); `data/release-map.csv` |
| Kernel-wheel timeline (SGLang) | 124 `sgl-kernel`/`sglang-kernel` versions, pins at 32 releases; `cache/git/sglang-kernel-wheels.json` |

The study's own bare clones (`$S/cache/git/{sglang,vllm}.git`, `--filter=blob:none`,
`--reference` to the deep study's clones) received a bulk prefetch of every
blob under the lineage and integration pathspecs listed in `scripts/blobs.py`
(SGLang 25,739 blobs; vLLM 18,779), so rename detection, blame and pickaxe
run locally. The deep study's clones were not modified.

First-parent history bounded by the cutoff commits was extracted once
(`scripts/git_extract.py` → `cache/git/<repo>-fp.jsonl`: 19,037 SGLang and
22,101 vLLM commits; name-status without rename detection). Both projects
squash-merge PRs and put `(#N)` in the subject, so a first-parent commit is
the unit of a merged change.

## 2. Lineage scopes and identity

Scopes (L1 SGLang MoE alignment/routing/top-k/fusion; L2 SGLang MLA /
FlashInfer MLA / FlashMLA; L3 the vLLM attention implementation family) are
defined operationally in `scripts/scope.py` (path patterns, integration
patterns, keywords) and in prose in `cache/task-screen.md`. Artifact identity
follows `CODEBOOK.md` §2. Artifact registries were built by one agent per
lineage from the path census (`scripts/paths.py`), rename detection
(`git log -M --follow`, `git show --stat -M`), introducing PRs and dispatch
code, then revised once after an audit for granularity
(`staging/registry/<L>-registry.json`, validated by
`scripts/registry_check.py`: L1 47, L2 36, L3 57 artifacts).

## 3. Candidate discovery (WP-B)

Strategies (`scripts/candidates.py`; each candidate records every strategy
that found it):

1. `path_core` — commit touches a lineage core path;
2. `path_integration+keyword` — integration path plus lineage keyword in the subject;
3. `subject_keyword`;
4. `body_keyword` — commit body (rarely informative: SGLang squash bodies are
   empty) and, for PRs created 2025-09-29..2026-09-28, the PR body from the
   first study's REST cache;
5. `symbol_pickaxe` — `git log --first-parent -G<lineage symbols>` over
   kernel and integration pathspecs (content-based; catches default/dispatch
   edits whose subjects do not name the lineage);
6. `dependency_pin` — `-G` over dependency/build files for the libraries that
   deliver lineage kernels;
7. `release_notes` — release-note lines matching lineage keywords that cite a PR;
8. `corpus:<file>` — deep-study provenance, correctness, revert and
   performance-PR records about lineage kernels.

This produced 7,916 candidate rows (L1 2,376; L2 1,608; L3 3,932; a commit can
be a candidate for L1 and L2). Every candidate PR's title, body, files (+/−),
labels and closing issues were fetched through GraphQL and cached
(`cache/gh/prlight/`).

## 4. Screening and coding protocol

Every candidate was inspected in context by an LLM coder working from a
written protocol (`cache/task-screen.md`, CODEBOOK v1.1) over reproducible
batches (`staging/screen/<L>/batch-NNN-input.{md,jsonl}`); outputs are
validated by `scripts/batches.py check screen` (exactly one schema-valid
record per candidate). Keyword and path signals only surfaced candidates; no
label was assigned mechanically. Verified candidates then received stage-2
coding (`cache/task-code.md`) with diff reading for introductions,
optimizations, ports, replacements, default changes, removals, reverts and
relands. All agreement statistics are model–model consistency, not human
inter-rater reliability.

## 5. Event graph construction (WP-D)

Stage-2 coders (`cache/task-code.md`) produced, per verified candidate, the
event type, primary cause, artifact IDs, specification/integration/default
changes, hardware scope, performance claim, correctness evidence, verified
optimization moves, explicit relations to earlier or later changes,
assumptions, a verbatim evidence excerpt (≤ 25 words), evidence types and
confidence; key event types required reading the diff. Three stage-1
positives were overturned in stage 2.

`scripts/assemble.py` builds the datasets:

- **Event IDs** are `<L>-E-<first 10 hex of the commit>` (stable across reruns).
- **Artifact IDs** proposed as `NEW:*` by coders were reconciled into the
  registries by one agent per lineage (`cache/task-newart.md`,
  `staging/registry/L*-new-map.json`); events whose artifacts were all judged
  out of scope in reconciliation are excluded and marked
  `rejected_reconciliation` in the candidate table.
- **Edges.** `textual_successor` edges link consecutive events on the same
  artifact (a fact of the artifact's path history, not an inference of
  intellectual ancestry). All other edge types come only from coder-recorded
  relations with evidence; relations whose targets are issues, upstream
  repositories or non-lineage PRs are kept in `data/external-references.csv`.
  PRs cited as reverted/relanded/repaired/ported targets that were not yet
  events were fed back as a **`cross_reference`** candidate batch (screened
  and coded in one pass, `staging/code/<L>/batch-900-*`). Three
  cross-repository anchors that the L1 story depends on (vLLM #2453, #12574,
  #19572) were added through the **`upstream_repo`** strategy with
  hand-verified records (`scripts/anchors.py`, `staging/code/L1/batch-901-*`).
- **Outcome** is computed: `merged_reverted` if a later `reverts` edge
  targets it; `merged_removed`/`merged_replaced` if all its artifacts ended;
  `merged_modified_later` if any later event touches one of its artifacts;
  else `merged_survives`.
- **Code survival at the cutoff** (`survives_at_cutoff = code`) means at least
  one line in a live lineage file is attributed to the event's commit by
  `git blame -w -M` at the cutoff commit (`scripts/blame.py`,
  `data/cutoff-line-survival.csv`; whole-file renames are followed, copy
  detection `-C` was disabled because it needs blobs outside the lineage
  pathspecs). `concept_only` means no surviving line but one of the event's
  moves is live at the cutoff; `unclear` is used for events whose artifacts
  are all upstream kernels or pins.

## 6. Optimization moves and assumptions (WP-E)

One agent per lineage induced a move catalog from code and diffs
(`cache/task-moves.md`, `staging/moves/<L>-moves.json`) with a validated code
**signature** per move (pathspec + extended regex, checked to match at one
release and not at another). Stage-2 coders tagged events with catalog move
IDs or proposed new names; a second pass per lineage consolidated names
(`name_map`), corrected first appearances, listed later uses and failures and
wrote biographies (`cache/task-biography.md`, `<L>-moves-final.json`). Because
the catalogs' assumption lists mostly restated design facts, a separate
**spec-checked audit** (`cache/task-assumptions.md`,
`<L>-assumptions-audit.json`) re-derived the *load-bearing* assumptions of
every move and classified them (CODEBOOK §6) only after opening the cited
public documentation (CUDA C++ Programming Guide, PTX ISA, HIP, Triton,
PyTorch CUDA-graph notes, FlashInfer docs). `data/optimization-moves.csv`
reports `first_observed_event` as the earlier of the catalog anchor and the
earliest coded event using the move (`first_observed_basis` says which).
**Move survival across releases** (`scripts/survival.py moves`,
`data/move-release-presence.csv`) runs each move's signature with `git grep`
at every final release commit; moves without a validated signature are not
scanned. A signature detects the mechanism's identifying code, not an intact
mechanism.

## 7. Release mapping (WP-F)

`scripts/release_map.py` → `data/release-map.csv`: GitHub Releases (non-draft,
`v*`, published ≤ cutoff); release date = GitHub publication time, or the
annotated-tag date when earlier (SGLang v0.2.13/v0.3.0 were published weeks
after tagging). For five vLLM tags that were re-pointed after publication
(v0.8.2, v0.9.0, v0.10.1, v0.10.2, v0.11.0) ancestry containment is capped at
the publication date (route suffix `_tag_moved`). An event's **first
containing release** (`scripts/releases.py`) is the earliest final release
whose branch point contains the commit on the first-parent line, or whose
side branch cherry-picks a commit carrying the same PR number. For SGLang
changes under `sgl-kernel/**` or `python/sglang/kernels/aot/**`, the kernel
ships in a separately versioned wheel: the event is released only when a
wheel built at or after the change exists **and** an SGLang release pins a
version at least that high (`cache/git/sglang-kernel-wheels.json`: 124 wheel
versions, pins at 32 releases); the later of the code release and the wheel
release is used (`release_route = kernel_wheel>=<version>`). Events after the
last release are `unreleased_at_cutoff`.

**Release transitions** (`scripts/analysis.py`): consecutive final releases
from each lineage's first appearance. Per artifact present at either end,
presence/change status comes from the release trees (`git ls-tree`,
`git diff --name-only`; `data/artifact-release-presence.csv`) and the
**rebase class** from the events landing in the later release, mapped as in
CODEBOOK v1.2 and resolved per artifact by the most invasive class; the
transition's `rebase_class` is its most invasive non-carry class, with the
full per-artifact multiset in `rebase_class_counts` and
`data/release-transition-artifacts.csv`. An artifact with no verified lineage
event in the transition is `direct_carry` even if its files changed textually
through screened-out commits; those observations are counted separately
(CODEBOOK v1.3), and textual identity is analysed independently in §8.

## 8. Survival analysis

Descriptive: artifact presence, textual identity (present and unchanged
across the next *k* releases) and move-signature presence for *k* = 1–3,
over all (item, release) pairs and over entry cohorts
(`data/survival-by-releases.csv`). Kaplan–Meier estimators
(`data/km-artifact-lifetime.csv`, `data/km-first-repair.csv`): origin =
artifact introduction date from the registry; events = artifact end
(removal/replacement) and first `repair_correctness`/`repair_performance`
event on the artifact; censoring at the cutoff (and, for first repair, at
artifact end). Milestones per artifact (first repair, revert, replacement,
removal) are in `data/artifact-milestones.csv`.

## 9. Failures and validation (WP-G, WP-I)

**Failures.** `scripts/failures.py` links deep-study correctness cases and
reverts to lineage events by PR number (`data/failure-links.csv`). One agent
per lineage (`cache/task-failures.md`) took the census of linked failures,
then reconstructed four failure-to-successor histories each from PRs, issues
and diffs, recording the invalidation scope, the durable encoding verified in
the fix diff (or exactly "no durable encoding found"), repeated assumptions
and the move's return (`staging/failures/<L>-failures.json`).

**I1 — independent recoding.** `scripts/recode.py sample` drew, with seed
20260928, 12 events per lineage (6 key-type, 6 other) = 36. A fresh agent
recoded event type, primary cause, the main explicit ancestry edge, moves and
artifacts **blind** (no access to first-pass outputs, deep-study labels or
artifact hints; `cache/task-recode.md`). `recode.py compare` computes percent
agreement and Cohen's κ (`recoding/agreement.csv`); an adjudicator saw both
codings and the evidence and ruled on every disagreement
(`recoding/reconciliation.jsonl`, `reconciliation-notes.md`,
`reconciliation-summary.csv`); rulings that changed a first-pass value were
applied as overrides (`staging/code-overrides.jsonl`). The two codebook
boundary clarifications the adjudicator identified (CODEBOOK v1.2) were
applied to the reconciled sample only. This is model–model consistency, not
human inter-rater reliability.

**I2 — triangulation.** Every event records its evidence types; event
confidence is `high` only with ≥ 2 independent evidence types. Every
biography and failure history lists its evidence types and states when only
one source existed.

**I3 — consistency.** `scripts/consistency.py` checks unique IDs and foreign
keys (events ↔ candidates ↔ registry artifacts ↔ moves; edges ↔ events),
chronological edges (except edges marked retrospective), URL/ID
correspondence for every event and candidate, that every verified candidate
has an event and every candidate was adjudicated, release existence and dates,
recomputation of every first-containing release, that biographies and
failure histories reference valid events, that no event or candidate is after
the cutoff, that every report number carrying a claim tag reproduces from its
CSV row, and that the SYNTHESIS executive summary and findings contain no
untagged numbers. Results: `data/consistency-check.json`.

## 10. Reproducibility

All analysis steps are re-run offline by `python scripts/run_all.py` (anchors,
name maps, assembly, move CSVs, presence and signature scans, blame,
snapshots, failure links, analysis, figures, reports, consistency).
Agent-produced inputs (registries, batches, catalogs, audits, histories,
narratives, recoding) are versioned under `staging/` and `recoding/`; their
protocols are `cache/task-*.md`. GitHub responses are cached under
`cache/gh/`. Network use: release listings, light PR records (GraphQL, one
point per 25 PRs), a few `gh pr view` calls by agents, and blob prefetches into
the study's own clones.

## 11. Deviations from the requested design

- The candidate unit is a merged first-parent commit (squash-merged PR);
  unmerged PRs appear only as external references.
- `extend_support` (event type), `repairs` (edge type), `algebraic_rewrite`
  (move category) and `primary_cause` were added to the required
  vocabularies (CODEBOOK v1.0) because much verified work fit none of the
  minimum codes.
- PR-body keyword discovery covers PRs created 2025-09-29..2026-09-28 (the
  first study's REST cache); older PR bodies were read only for candidates
  surfaced by other strategies.
- Artifact snapshots (`data/artifact-snapshots.csv`) are taken for in-tree
  artifacts at introduction, at every release where they changed, appeared or
  disappeared, and at the cutoff; upstream kernels are represented through
  their adapters and pins.
