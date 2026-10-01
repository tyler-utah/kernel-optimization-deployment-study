# Phase 01 — contribution-rule audit

## Scope and provenance

This is a repository-state audit, not a pull-request outcome study. It is limited to:

- **vLLM** at `7230dfea501b232ce92d057f1c5ee9cb009c4cb4` (commit date `2026-09-29T09:43:07+08:00`).
- **SGLang** at `79cafec013d0a01c782d7a4bc902ef32a8941ba8` (commit date `2026-09-28T17:21:31-07:00`).

The empty working trees were not restored or modified. Evidence came from the existing Git object stores with commands of these forms:

```powershell
git -C experiments\data\repos\vllm cat-file -t 7230dfea501b232ce92d057f1c5ee9cb009c4cb4
git -C experiments\data\repos\vllm ls-tree -r --name-only 7230dfea501b232ce92d057f1c5ee9cb009c4cb4
git -C experiments\data\repos\vllm show 7230dfea501b232ce92d057f1c5ee9cb009c4cb4:path
git -C experiments\data\repos\vllm grep -n -E 'pattern' 7230dfea501b232ce92d057f1c5ee9cb009c4cb4 -- path

git -C experiments\data\repos\sglang cat-file -t 79cafec013d0a01c782d7a4bc902ef32a8941ba8
git -C experiments\data\repos\sglang ls-tree -r --name-only 79cafec013d0a01c782d7a4bc902ef32a8941ba8
git -C experiments\data\repos\sglang show 79cafec013d0a01c782d7a4bc902ef32a8941ba8:path
git -C experiments\data\repos\sglang grep -n -E 'pattern' 79cafec013d0a01c782d7a4bc902ef32a8941ba8 -- path
```

No live default-branch file was substituted for a frozen object. No GitHub API result was needed. The machine-readable rule inventory, exact quotations, and measurement signals are in `experiments/results/contribution-rules.csv`.

## Documents and controls inspected

### vLLM

The repository has no root `CONTRIBUTING.md` at this SHA. `.github/CONTRIBUTING.md` redirects to the docs site; `docs/contributing/README.md` is the substantive contribution guide. The audit also covered `.github/PULL_REQUEST_TEMPLATE.md`, `.github/CODEOWNERS`, `.github/mergify.yml`, the title, pre-commit, CI-command, CI-authorization, stale, and label workflows/scripts, `DCO`, `docs/governance/{process,committers,collaboration}.md`, and the public kernel-writing/microbenchmark agent guidance.

### SGLang

The repository has no root `CONTRIBUTING.md` at this SHA. `docs/CONTRIBUTING.md` is a generic docs-site editing guide, while `docs/docs/developer_guide/contribution_guide.mdx` is the substantive developer guide. The audit also covered `.github/pull_request_template.md`, `.github/CODEOWNERS`, `.github/MAINTAINER.md`, `.github/CI_PERMISSIONS.json`, `.github/labeler.yml`, CI label scripts, the PR gate, baseline/extra/kernel test workflows, stale and labeler workflows, the diffusion contribution policy, and the path-scoped kernel benchmark rule.

## Automatic enforcement versus social enforcement

| Repository | Automatic or machine-gated | Social, reviewer, or role-gated |
| --- | --- | --- |
| vLLM | A ready, non-draft PR gets a regex title check requiring one or more `[Tag]` prefixes. Mergify reacts to a failing external `dco` status. Pre-commit is withheld unless the PR is `verified`/`ready`/`ready-run-all-tests` or the author has at least four merged PRs. CI comments are authorization-checked; run commands also check branch freshness unless `--allow-stale` is used. Mergify auto-labels by path/title. The stale action marks after 90 inactive days and closes 30 days later, exempting drafts and `keep-open`. | The guide requires DCO signoff, AI disclosure and commit attribution, quality/tests/docs, and an RFC for major architectural changes over 500 non-exempt LOC. Governance requires at least one committer approval and says CODEOWNERS-covered code should receive owner review. Lead maintainers may directly merge trivial/hotfix changes and may force-merge when CI failure is unrelated. |
| SGLang | PR CI uses a runtime gate: non-draft plus `run-ci`; extra CI additionally requires `run-ci-extra`. Slash commands consult `CI_PERMISSIONS.json`, repository permission, command type, and cooldown. Changed kernel paths select dedicated H100/B200 and multi-GPU jobs. The labeler adds `sgl-kernel` or `jit-kernel` from changed paths. The stale workflow closes qualifying inactive/WIP/over-quota PRs, with explicit exemptions. | The template and guide require focused tests and accuracy/benchmark evidence where relevant. Each protected file normally needs a CODEOWNER approval and required CI; a write-role contributor may then merge. A Merge Oncall may bypass branch protection in an exception and assumes revert/hotfix responsibility. CI control labels consume shared GPU capacity and are explicitly non-default. |

Repository files prove that controls are configured; they do not prove which GitHub branch-protection checks were required or whether any historical PR complied.

## Measurable rules and proposed signals

The rules below can be measured later without subjective person ranking:

- **DCO/signoff:** fraction of vLLM PR commits with a successful DCO check and a `Signed-off-by:` trailer.
- **Title format:** vLLM ready PR titles matching `^(\[[^][]+\])+ +[^[:space:]].*$`. No repository-wide SGLang PR-title check or title convention was found; the diffusion guide contains a commit-message convention only.
- **Description completeness:** presence of purpose, test plan, and test result in vLLM; motivation, modifications, accuracy evidence, and speed/profile evidence in SGLang when applicable.
- **Tests/docs:** changed behavior mapped to added/updated tests and user-facing changes mapped to docs.
- **Large-change RFC:** vLLM architectural changes over 500 LOC after excluding kernel/data/config/test lines, linked to an issue/RFC.
- **Review:** at least one vLLM committer approval; applicable owner review; SGLang approval coverage for every CODEOWNERS-protected changed file plus Merge Oncall coordination.
- **CI:** required checks green on the latest commit, label/command path used, actor authorization tier, cooldown, branch freshness, and bypass/force-merge use.
- **Kernel correctness/performance:** correctness tests, `torch.library.opcheck()` where applicable, accuracy evidence for output-changing code, benchmark artifacts for speed-affecting code, cold-L2 handling, and representative device/workload coverage.
- **AI assistance:** disclosure/trailers for vLLM; end-to-end validation for AI-assisted diffusion changes in SGLang.
- **Staleness:** inactivity duration, draft/WIP status, author open-PR count, approval presence, exemption labels, and close/reopen outcome.

All PR-level compliance remains **not measured in Phase 01**.

## Kernel ownership and reviewer coverage

Counts below are configuration counts, not rankings. GitHub applies the last matching CODEOWNERS pattern, so more-specific rows replace rather than add to a broader row.

### vLLM

- Production `/vllm/kernels/` has **2 CODEOWNERS**; `/vllm/kernels/helion` has a more-specific group of **2**.
- `/tests/kernels` has **6 CODEOWNERS**; `/tests/kernels/ir` has a more-specific group of **2**.
- Additional attention, fused-MoE, quantization, ROCm, and third-party-kernel paths have their own groups; a single “all kernel code” count would therefore be misleading.
- Governance treats area owners as committers, and CODEOWNERS-covered changes “should” be reviewed by those owners. It does not state that every listed owner must approve.

### SGLang

- Broad `/python/sglang/kernels` coverage has **6 CODEOWNERS**.
- More-specific groups override it: diffusion ops **6**, FLA attention ops **3**, AOT kernels **6**, and AOT MUSA **1**.
- The maintainer document also defines **one Kernel Merge-Oncall/reviewer group**, containing **1 role holder** at the frozen SHA and covering `python/sglang/kernels` plus the legacy `sgl-kernel` reference.
- The merge rule requires at least one applicable CODEOWNER approval per protected modified file unless a Merge Oncall bypasses.

## Expensive GPU CI: trigger, label, and role requirements

### vLLM

- Upstream Buildkite GPU CI is comment-driven: `/ci run`, `/ci run all`, and `/ci run nightly`; AMD equivalents are also accepted. `all` and `nightly` set broader pipeline environment flags.
- Repository `admin`, `maintain`, or `write` roles and configured trusted contributors may invoke commands directly.
- A non-draft PR author gains access after a `ready`/`ready-run-all-tests` label or approval by a trusted reviewer. Before delegation, unrelated low-permission users cannot trigger it.
- New commits do not automatically start upstream CI. Run commands require the PR head to contain the target branch; `--allow-stale` permits an explicitly warned run, but the guide says the latest-target run is needed before merge.
- Whether Buildkite statuses are truly required is a branch-protection setting outside Git and therefore not observable here.

### SGLang

- Baseline GPU CI requires a non-draft PR and `run-ci`; extra CI requires both `run-ci` and `run-ci-extra`.
- `/tag-run-ci-label` affects future commits only; `/tag-and-rerun-ci` applies the label and reruns. `/run-full-ci` and `/run-extra-ci` require both label and rerun permissions in `CI_PERMISSIONS.json`; PR authors are not automatically allowed to use those commands.
- PR authors may rerun failed CI on their own PR. Selective test/group reruns require a zero cooldown entry or write/admin permission and do not validate a PR-local AOT-kernel wheel.
- Kernel path detection dispatches expensive suites, including H100/B200, 4-GPU B200, and 8-GPU H200 jobs. AOT changes must use normal/full PR CI rather than selective reruns.
- `parallel-stages`, `max-concurrency`, `bypass-fail-fast`, and especially `highest-priority` spend additional shared-runner capacity. These are discretionary control labels, not normal contributor entitlements.

## Who may merge

- **vLLM:** Committers have write and merge rights. Normal PRs require at least one committer approval; owner review is expected when CODEOWNERS applies. Lead Maintainers may directly merge trivial/hotfix changes and may override unrelated CI failures. The frozen repository does not expose the live branch-protection configuration.
- **SGLang:** A person with the Write role may merge after required CI and applicable CODEOWNER approvals. A Merge Oncall may bypass branch protection in exceptional cases and is responsible for resulting reverts/hotfixes. CI Maintenance Mode prohibits non-CI-fix merges even though Merge Oncalls otherwise have bypass authority.

No people are named or ranked here; only documented roles and configured group sizes are reported.

## Undocumented repeated requirements

Phase 01 did not inspect PR histories, so it makes **no claim** that an undocumented requirement is repeated. Candidate signals for **Phase 05 measurement** are:

- recurring reviewer requests for benchmark shapes, hardware, tolerances, before/after tables, or reproduction commands beyond the written guides;
- recurring requests for issue links on changes below vLLM's documented 500-LOC RFC threshold;
- de facto approval counts greater than the documented minimum;
- labels or slash commands repeatedly required in practice but absent from contributor-facing docs;
- recurring SGLang PR-title conventions despite no repository-wide title rule at the frozen SHA;
- recurring AI-disclosure demands outside SGLang Diffusion, where no repository-wide AI contribution policy was found.

Only public PR comments, reviews, checks, labels, and merge metadata should be used to confirm or reject these candidates.

## Limitations

- This is a static snapshot of two repositories and two SHAs; it does not generalize to later policy.
- Git does not contain branch-protection rules, GitHub App configuration, team membership, repository variables/secrets, live labels, or external Buildkite settings. Mergify's `check-failure=dco` proves that the repository reacts to a DCO status, not how the external app is configured.
- CODEOWNERS lines identify requested-review coverage, not actual availability, response time, approval, or permission.
- `CI_PERMISSIONS.json` was inspected only as configuration; identities are deliberately not reproduced or ranked.
- “When applicable,” “sufficient,” “relevant,” and “significant” need a coding rubric before quantitative use.
- SGLang's AI guidance is scoped to Diffusion; vLLM's public `.agents` kernel guidance is agent-oriented auxiliary guidance. Neither was silently generalized beyond its stated scope.
- No PR sample, undocumented-practice inference, or compliance rate belongs to Phase 01; those are deferred to Phase 05.
