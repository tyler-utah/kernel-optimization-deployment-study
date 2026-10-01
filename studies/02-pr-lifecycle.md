# Study 2: The Delivery Pipeline in the Wild

**Question:** When someone contributes a kernel optimization to a real framework, what happens to it? How long does it take to land, how often does it never land, and what evidence has to accompany it?

**Talk hook:** *§1, the Optimization Gap.* "Suppose somebody already produced a kernel that is 20% faster. How long before users see that 20%?" This study answers that question directly, using public data.

## Hypotheses
- H1: Performance and kernel PRs take longer to merge than other PRs, and are abandoned more often.
- H2: Most perf PRs report *microbenchmark* speedups. A minority report *end-to-end* serving impact, and fewer still report accuracy or correctness evidence beyond unit tests.
- H3: The open-PR backlog is disproportionately made up of perf and kernel PRs, meaning discovered optimizations are waiting in a queue.
- H4: Review conversations on perf PRs focus on the downstream stages: correctness on other hardware, integration with CUDA graphs and torch.compile, other backends, and maintenance burden.

## Data
- vLLM, SGLang, llama.cpp, FlashInfer (and PyTorch via linked PRs; see the merge-bot caveat): all PRs from the last 12 months, including still-open ones.
- For each PR: timestamps, state, labels, changed paths, review count and rounds, comments, linked issues, body text, and CI status.

## Method
1. **Classify PRs** as perf/kernel-optimization, kernel-correctness-fix, feature, model-support, portability, or other, using path rules + title tags + an LLM codebook, validated on a hand-labeled sample.
2. **Lifecycle metrics per class:** time to first review, time to merge (median and P90), merge rate, abandonment rate (closed unmerged, or stale for more than 90 days), and number of review rounds.
3. **Evidence content** (LLM extraction from the PR body and comments): does the PR report
   - a microbenchmark speedup,
   - end-to-end throughput or latency,
   - accuracy evals (such as lm-eval or GSM8K),
   - numerical tolerance or correctness tests,
   - results on more than one hardware target?
4. **Reviewer-concern taxonomy:** cluster review comments on perf PRs into correctness, other hardware, integration, maintenance and code complexity, benchmark methodology, and so on.
5. **Backlog composition:** share of open PRs by class and age.

## Candidate slides
- "A kernel-optimization PR in vLLM takes a median of X days to land (vs. Y for other PRs), and Z% never land."
- "Only W% of perf PRs report end-to-end impact; V% report any accuracy validation." (This connects to KernelBench-Verified and FastKernels: the ecosystem reasons from microbenchmarks.)
- "What reviewers ask about: correctness on other GPUs, CUDA-graph compatibility, maintenance." This is the downstream pipeline showing up in human review.
- "Discovered but undelivered: N open optimization PRs claiming a combined M speedups." This is the Optimization Gap as a queue.

## Pitfalls
- SGLang's title conventions are less consistent than vLLM's, so it leans more on path rules and the LLM classifier.
- Some optimizations land as part of big feature PRs. Report the PR-level share, and note this undercounting.
- "Closed unmerged" includes superseded PRs, where the idea landed elsewhere. Detect cross-links ("superseded by #…").
- PyTorch uses ghstack and the merge bot. Derive merge time from the bot's comment or commit time.

## Stretch
- **Anonymized internal comparison:** the same metrics on a Microsoft-internal kernel pipeline would give a public-vs-internal slide, if approved.
- **Follow a cohort:** take 20 recent perf PRs and trace them to the first release that ships them, then to downstream adoption (for example, a FlashInfer kernel reaching vLLM or SGLang). This gives the full "discover → deploy" latency.

## Effort
★★. API collection is straightforward. The classification and evidence extraction is where most of the work goes.
