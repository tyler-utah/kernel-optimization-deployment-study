# Study 4: From Paper to Production

**Question:** How many recent systems/ML papers propose kernel-style optimizations, and how many of those optimizations actually reach the frameworks people deploy?

**Talk hook:** *§1: "We publish the top of the optimization pipeline. Industry lives in the rest of it."* This is the most provocative study for an academic audience, because it measures the Optimization Gap for *our own community's* output.

## Scope: what counts as a "kernel-style optimization"
Be broad, as agreed:
- a new or improved kernel (attention variant, GEMM, MoE, sampling, normalization, quantized matmul, collectives)
- fusion (a fused operator family, mega-kernels, persistent kernels)
- a precision or format change that needs kernel support (FP8, FP4, INT4, KV-cache quantization)
- a memory-layout or scheduling change implemented at kernel level (paged/ragged layouts, split-K, stream-K, work partitioning)
- compiler or DSL work whose output is kernels (Triton/TileLang-style, autotuning, superoptimization)
- agentic or LLM-driven kernel generation

## Data
- **Venues, 2023–2026:**
  - MLSys, OSDI, SOSP, NSDI, EuroSys, ATC
  - ASPLOS, ISCA, MICRO, HPCA
  - PPoPP, SC, CGO, PLDI
  - systems-flavored papers at NeurIPS, ICLR, and ICML
- **arXiv:** cs.DC, cs.LG, cs.AR, cs.PL, with keyword filters. Report monthly counts to show the trend.
- **Seed lists:** Awesome-LLM-Kernel-Agent (GitHub) for agentic kernel papers; DBLP for venue proceedings.

## Method
1. **Census:**
   - Collect titles and abstracts per venue and year.
   - Classify papers as *proposes a kernel-style optimization* (yes/no) and by category (list above), using an LLM codebook plus a hand-validated sample.
   - Headline: the share and count per year.
2. **Evaluation-methodology audit** (subsample of about 50–100 papers). Does the paper report:
   - end-to-end serving or training numbers vs. microbenchmarks only
   - which baseline, and how strong (KernelBench-Verified shows this matters)
   - its correctness or accuracy validation method
   - the number of hardware targets
   - public code availability
3. **Adoption tracing** (subsample from 2023–2024, so there has been time to adopt):
   - For each paper, search vLLM, SGLang, TensorRT-LLM, llama.cpp, FlashInfer, and PyTorch for the technique name, paper title, arXiv ID, or author repo, in code, PRs, issues, and docs.
   - Classify as: *upstreamed* (merged, on by default or selectable), *in progress* (open PR or issue), *reimplemented differently*, or *not adopted*.
   - Record the **time from arXiv to merge** for adopted ones.
4. **Trend:** the arXiv count of agentic-kernel-generation papers per month (evidence that "candidates are abundant").

## Making "deployed" measurable: an evidence ladder

"Is it deployed?" can't be observed directly. Instead, score each paper by the **highest level of observable evidence** it reaches. This gives a conservative lower bound, and each level is checkable by a person.

| Level | Evidence | How to observe it |
|---|---|---|
| L0 | No public code | Paper text, artifact links |
| L1 | Code released | Artifact / GitHub link in the paper |
| L2 | Code *maintained* after publication | Commits to the artifact repo more than 3 months after the camera-ready; issues answered; pip package |
| L3 | Picked up by a kernel library | FlashInfer, xformers, Liger-Kernel, AITER, CUTLASS examples, … (code, PRs, docs) |
| L4 | Merged into a serving or training framework | vLLM, SGLang, TensorRT-LLM, llama.cpp, PyTorch, Megatron, DeepSpeed |
| L5 | On by default, documented, or in release notes | Framework docs, release notes, default config |
| (S) | *Self-reported* production deployment | "deployed in production at X" in the paper. Tracked separately because it can't be verified. |

Search signals for L3–L5: paper title, technique name, arXiv ID, author GitHub handles in PRs, and "based on" / "adapted from" comments. Hand-check every positive and a random sample of negatives, and report the false-negative rate.

**The honest framing of the result:** *"At most X% of kernel-optimization papers reach L4 within 12–18 months; Y% never get past L1."* A lower bound is fine for the argument, because the claim is that the pipeline leaks.

## Pilot: one conference, end-to-end

**MLSys 2025** is a good first target:
- 61 papers, with a scrape-friendly list at `proceedings.mlsys.org/paper_files/paper/2025` (checked).
- The papers have had about 16 months to be adopted.
- The venue is squarely systems-for-ML, and it is the same venue as FlashInfer-Bench.

**Steps:**
1. Census the 61 papers: how many propose kernel-style optimizations?
2. Place each kernel paper on the ladder.
3. Report one row: *N papers → K propose kernel optimizations → a reach L2 → b reach L4*.

Repeat for one architecture venue (ASPLOS or ISCA 2025) for contrast.

## The reverse direction: from production back to papers

Start from what's deployed, which is an enumerable set, and ask where it came from. This is easier to measure than forward adoption.

**Pilot (2026-09-28, shallow clones, `arxiv.org/abs|pdf` links in source and docs):**

| Repo | Distinct arXiv papers cited anywhere | …cited from kernel / attention / csrc paths |
|---|---:|---:|
| vLLM | 61 | **7** |
| SGLang | 53 | **2** |

For codebases this size, that's very few. Most adoption doesn't cite papers by URL: it cites by name ("FlashAttention") or not at all. So this is a floor, not a count. It still suggests a slide: *"vLLM's kernel code cites 7 papers."*

**Fuller version:**
- Enumerate the kernels and backends in vLLM and SGLang today: attention backends, MoE kernels, quantization methods, sampling, and so on.
- Attribute each one's origin: academic paper, vendor library, company engineering blog, or community contribution.
- Headline: *"Of the N kernel families in vLLM today, X% trace to an academic paper."*

## Other questions to ask about papers

- **Baseline staleness:** which version of vLLM, SGLang, or PyTorch did the paper compare against, and how old was it at submission? Extract version strings from the evaluation sections. This ties to Study 1 (the moving target) and to KernelBench-Verified (baseline choice erases speedups). Candidate slide: *"The median MLSys 2025 paper benchmarked against a vLLM release that was already N months and K releases old."*
- **Artifact half-life:** for papers with code, when was the last commit relative to publication, and when was the last compatible dependency version? Candidate slide: *"X% of kernel-paper artifacts saw no commits 3 months after the conference."*
- **Industry vs. academic authorship:** does the adoption rate differ? Industry papers often describe systems that are already deployed, so academic and industry papers may be answering different questions.
- **Artifact evaluation badges** (MLSys, ASPLOS, OSDI): the share with Available / Functional / Reproduced. Does "Reproduced" predict adoption?
- **Hardware and correctness validation in the paper:** the number of GPU types evaluated, and whether correctness is checked beyond "matches the reference within tolerance" (overlaps with step 2 above).
- **Author survey (optional, small):** email the authors of kernel papers: "Is this deployed anywhere? What blocked upstreaming?" A handful of quotes about *why* things didn't ship (review burden, API churn, hardware coverage, maintenance) would be very strong talk material and would ground the Establish and Sustain stages in practitioners' own words.

## Candidate slides
- "N papers at top systems/architecture venues proposed kernel-style optimizations in 2025, up from M in 2023."
- "Of the kernel-optimization papers from 2023–24, X% are usable in vLLM or SGLang today; median arXiv-to-merge time is D months." Also name the famous successes (FlashAttention, PagedAttention) *and* note how rare they are.
- "Y% of these papers report only microbenchmarks; Z% validate correctness beyond tolerance testing."
- "Agentic kernel-generation papers per month": a curve that looks like the candidate supply is exploding.

## Pitfalls
- Adoption is hard to detect when techniques are renamed or reimplemented. Use multiple signals and hand-check positives *and* a sample of negatives.
- Survivorship: famous successes dominate memory. The census corrects for this, which is the point.
- Some papers target training, not serving. Track the training-framework subset separately (PyTorch, Megatron, DeepSpeed) or scope to inference.
- Venue access: use DBLP and open proceedings. Some ACM/IEEE abstracts need care with scraping terms.

## Effort
★★★. The census is mechanical with an LLM classifier. Adoption tracing is the labor-intensive part, but it's the payoff.
