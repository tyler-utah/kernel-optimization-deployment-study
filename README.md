# Optimization-Delivery Empirical Artifact

> **This study and all of its code were generated entirely by AI agents under
> human direction.**

This is a standalone reproducibility repository for empirical Git and GitHub
studies of how performance optimizations are proposed, reviewed, integrated,
released, and maintained.

The related-work papers show the optimization gap using benchmarks. These
studies examine it in the public repositories where optimizations are proposed,
reviewed, merged, reverted, released, and maintained.

## Artifact contents

- `config/study.json`: frozen repositories, date windows, seeds, and commit
  SHAs.
- `scripts/fetch_data.py`: resumable source-repository and GitHub collection.
- `scripts/run_agents.ps1`: versioned Copilot CLI entry point for the coding,
  adjudication, and lineage tasks.
- `scripts/run_analysis.py`: deterministic regeneration of derived tables,
  figures, reports, and consistency checks.
- `scripts/capture_environment.py`: machine-readable tool and package versions
  for each reproduction environment.
- `scripts/reproduce.ps1`: top-level verification and reproduction entry point.
- `results/`, `deep-study/data/`, and `kernel-lineage-study/data/`: checked-in
  derived datasets.
- `deep-study/coding/` and `kernel-lineage-study/staging/`: coded records,
  codebooks, and agent outputs needed to reproduce the analyses.

Large API responses, cloned repositories, and extracted source snapshots are
not versioned. They are reproducible caches fetched from the frozen
configuration and are listed in `.gitignore`.

## Quick start

Requirements:

- Python 3.11 or newer;
- Git;
- authenticated GitHub CLI (`gh auth login`);
- GitHub Copilot CLI for the agent-coded stages.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# Validate the frozen configuration.
.\scripts\reproduce.ps1 -Stage check

# Rebuild ignored repository and GitHub caches.
.\scripts\reproduce.ps1 -Stage collect

# Re-run coding/adjudication agents. This is expensive and resumable.
.\scripts\reproduce.ps1 -Stage agents

# Rebuild deterministic outputs from collected and coded records.
.\scripts\reproduce.ps1 -Stage analysis
```

For the checked-in artifact, the fastest verification path is:

```powershell
.\scripts\reproduce.ps1 -Stage check
python deep-study\scripts\consistency.py
python kernel-lineage-study\scripts\consistency.py
```

## Study status

The preliminary, deep delivery/failure/provenance, and kernel-lineage studies
are complete. Their persistent checkpoints document exact recovery state.

## Which frameworks?

SGLang is open source (Apache-2.0, `sgl-project/sglang`). All the candidates below are public on GitHub.

| Tier | Role in the study | Repos | Why |
|---|---|---|---|
| **1. Serving engines** (core) | Where optimizations are *deployed* | **vLLM**, **SGLang**, TensorRT-LLM, **llama.cpp** | vLLM and SGLang are the integration targets named in FlashInfer-Bench and SWE-Serve. llama.cpp ties directly to WarpDRF (3 races found there) and covers many backends (CUDA, Metal, Vulkan, SYCL, ...). |
| **2. Kernel libraries** (supply) | Where optimizations are *discovered and packaged* | **FlashInfer**, FlashAttention, CUTLASS, DeepGEMM, AITER (AMD), Liger-Kernel, TileLang | Upstream suppliers to tier 1. They show the flow from supply to deployment and the duplication between projects. |
| **3. Framework / compiler** | The baseline everyone compares against | PyTorch (Inductor, ATen CUDA), Triton | PyTorch is the "reference" in every benchmark paper. It's huge, so scope it to kernel directories. |

**Recommended core set:** vLLM, SGLang, llama.cpp, FlashInfer (plus PyTorch where feasible). The two big serving engines, one portability-heavy engine, and one kernel supplier.

## Sizing probe (last 90 days, run 2026-09-28)

Script: [scripts/repo_probe.py](scripts/repo_probe.py). The keyword columns are crude title matches, **for sizing only**.

| Repo | Stars | Open PRs | Open issues | Commits / 90d | Merged PRs / 90d | …with "kernel" in title | Issues / 90d w/ correctness keywords |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-project/vllm | 93k | **5,820** | 2,551 | **3,826** (~43/day) | 3,847 | 270 | 63 |
| sgl-project/sglang | 37k | **4,446** | 897 | **4,361** (~48/day) | 4,655 | 319 | 45 |
| pytorch/pytorch | 103k | 3,525 | 14,074 | 4,783 | 51 ⚠️ | 0 ⚠️ | 180 |
| NVIDIA/TensorRT-LLM | 15k | 896 | 606 | 2,412 | 2,371 | 58 | 5 |
| ggml-org/llama.cpp | 130k | 1,679 | 885 | 1,391 | 1,386 | 59 | 37 |
| flashinfer-ai/flashinfer | 6.5k | 702 | 349 | 861 | 900 | 90 | 16 |
| ROCm/aiter | 0.6k | 705 | 351 | 941 | 1,013 | 173 | 19 |
| triton-lang/triton | 20k | 386 | 895 | 628 | 654 | 29 | 13 |
| microsoft/onnxruntime | 22k | 923 | 849 | 703 | 739 | 42 | 27 |
| tile-ai/tilelang | 7.5k | 112 | 213 | 373 | 373 | 15 | 85 |
| linkedin/Liger-Kernel | 6.6k | 106 | 106 | 129 | 129 | 19 | 3 |
| Dao-AILab/flash-attention | 25k | 295 | 1,031 | 66 | 65 | 4 | 2 |
| NVIDIA/cutlass | 10.5k | 306 | 465 | 49 ⚠️ | 49 | 2 | 9 |
| deepseek-ai/DeepGEMM | 7.9k | 79 | 70 | 4 ⚠️ | 10 | 1 | 3 |

**Early observations. These are hypotheses, not results.**
- **The target moves very fast.** vLLM and SGLang each land about 40–50 commits *per day*. Anything tuned against them is aiming at a moving target (→ Study 1).
- **The backlog is huge.** vLLM has about 5.8k open PRs and SGLang about 4.4k. That is roughly *1.5 and 1 quarters* of merge throughput sitting in the queue. It looks like the Optimization Gap as a literal queue (→ Study 2).
- **Kernel-titled PRs are a meaningful slice**: about 7% of merged PRs in both vLLM and SGLang by title alone. That is a lower bound; path-based classification will find more.
- **AMD's AITER is disproportionately kernel-heavy** (17% of merged PRs have "kernel" in the title). Portability work shows up as kernel churn (→ Study 4).

**Methodology traps the probe already exposed:**
- ⚠️ **PyTorch** lands PRs through `pytorchmergebot`, and PRs show up as *closed*, not *merged*. Use commits and linked PR numbers, not GitHub's merged flag.
- ⚠️ **CUTLASS and DeepGEMM** appear to sync from internal repos in bulk drops, so commit counts don't reflect the pace of development. Use release diffs instead.
- ⚠️ Keyword matches are noisy (tilelang's "nan"/"incorrect" hits need checking). Real results need the classification pipeline below.

## Study catalog

Each study has its own file in [studies/](studies/). Scores run from ★ (low) to ★★★ (high).

| # | Study | Research question | Punch | Effort | Priority |
|---|---|---|---|---|---|
| 1 | [The moving target](studies/01-moving-target.md): churn and the half-life of kernel code | How quickly does optimized code decay? | ★★★ | ★★ | **P1** |
| 2 | [The delivery pipeline in the wild](studies/02-pr-lifecycle.md): life of an optimization PR | How do performance changes move through review and integration? | ★★★ | ★★ | **P1** |
| 3 | [What breaks](studies/03-what-breaks.md): kernel correctness bugs, reverts, regressions | Which failures cause deployed kernel changes to be repaired or reverted? | ★★★ | ★★★ | **P1** |
| 4 | [From paper to production](studies/04-paper-to-production.md): do published kernel optimizations get adopted? Scored with a deployment-evidence ladder; pilot on MLSys 2025 (61 papers); reverse trace from production to papers | How often does published optimization work reach production repositories? | ★★★ | ★★★ | **P1** |
| 5 | [Validation practices](studies/05-validation-practices.md): tolerances, tests, CI hardware | What evidence is used to validate performance changes? | ★★ | ★ | P2 |
| 6 | [Portability load](studies/06-portability-load.md): cost of each new GPU and backend | What maintenance burden does hardware portability create? | ★★ | ★★ | P2 |
| 7 | [Reinvention](studies/07-reinvention.md): the same kernel, N copies | How much implementation duplication accumulates across projects? | ★★ | ★★ | P2 |
| 8 | [Agents in the commit log](studies/08-agent-prs.md): AI-generated optimizations reaching upstream | How often do AI-generated optimizations reach upstream repositories? | ★★ | ★ | P3 |

**If we only do three:** Studies 2, 4, and 1. Study 2 measures the gap directly, Study 4 is the most provocative for an academic audience, and Study 1 is the easiest compelling Sustain number. Study 3 is the strongest bridge into RESOLVE, but it's the most labor-intensive; a scoped version (reverts only) is cheap.

## Interpretation principles

- **Use deployment repositories directly.** vLLM, SGLang, and PyTorch provide observable integration and maintenance histories that benchmark results do not.
- **Report comparisons, not isolated counts.** For example, compare the time required to merge a kernel PR with other PRs or measure how often published kernel optimizations become available in deployment repositories.
- **Keep claims tied to generated evidence.** Every reported number should trace to a versioned table, coded record, or consistency check.
- **Be honest about noise.** Report classifier precision on a hand-labeled sample. A skeptical systems audience will ask.

## Shared methodology

1. **Data collection**
   - GitHub GraphQL/REST via `gh` for PRs, issues, reviews, labels, linked issues, and timelines.
   - Local full clones for git history, `git blame` survival, and path analysis.
   - Raw data goes in `data/` (not committed if large).
2. **Kernel-change classification**, in layers:
   - **Path rules** (high precision): `csrc/`, `*.cu`, `*.cuh`, `*.hip`, Triton `@triton.jit` files, `sgl-kernel/`, `ggml-cuda/`, `ggml-metal/`, etc.
   - **Title conventions**: vLLM uses `[Kernel]`, `[Perf]`, `[Bugfix]`, `[Hardware][AMD]`. llama.cpp uses `CUDA:`, `metal:`, `vulkan:`.
   - **LLM classification with a written codebook** (perf-optimization, kernel-correctness-fix, portability, feature, refactor, ...).
   - **Human validation**: hand-label about 100 samples per repo and report precision and recall. *Agents propose, humans decide*, applied to our own methodology.
3. **Time window**: the last 12 months for the headline numbers, and the full history for trends (for example, a kernel-PR share per quarter since 2023).
4. **Reproducibility**: scripts in `scripts/`, parameterized by repository and window.

## Decisions to make

- [ ] Final framework set (recommended: vLLM, SGLang, llama.cpp, FlashInfer, plus PyTorch where feasible)
- [ ] Time window (recommended: 12 months for headline numbers, full history for trends)
- [ ] Which studies to run first (recommended: 2 → 4 → 1, then scoped 3)
- [ ] Whether to pursue an anonymized internal Microsoft comparison for Study 2
- [ ] Budget for LLM classification and the size of the hand-labeled validation sample
