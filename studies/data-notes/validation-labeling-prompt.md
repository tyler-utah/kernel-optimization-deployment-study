# Validation reference-label provenance

The two `validation-labels-*.csv` files are the raw output of one manual coding
pass performed by a GitHub Copilot CLI general-purpose subagent. The CLI did not
surface the subagent's exact model identifier, so it is recorded as
**Copilot CLI default general-purpose model (model ID unavailable)** rather than
inventing a model name. This is LLM-assisted reference coding, not independent
human double-coding; no inter-rater agreement is claimed.

## Prompt

> Work autonomously in
> `C:\Users\tsorensen\Documents\github\RealOptimizationTalk`. Own ONLY the
> human-reference labeling for Phase 03. Read
> `experiments\studies\data-notes\classification-codebook.md` and the blinded
> files `experiments\results\validation-sample-vllm.csv` and
> `experiments\results\validation-sample-sglang.csv`. Each has 220 records.
> Manually interpret every record using only subject and changed_paths. Fill the
> existing corresponding `experiments\results\validation-labels-vllm.csv` and
> `experiments\results\validation-labels-sglang.csv` fields:
> `human_primary_purpose` must be exactly one of `kernel_performance`,
> `kernel_correctness`, `bugfix`, `hardware_backend`, `model_support`,
> `core_runtime_serving`, `api_frontend_router`,
> `test_ci_build_benchmark_deps`, `documentation`, `maintenance_refactor`,
> `new_workloads`, `other`; `human_touches_kernel_implementation` must be 0 or 1
> following the codebook (tests/docs/benchmarks alone are not implementation);
> `label_notes` should briefly state the decisive evidence. Do not inspect or
> copy classifier predictions from the generated commits CSVs; preserve
> blindness. This is one-coder manual labeling; do not claim independent double
> coding. Validate 220 complete, allowed labels only, no blanks, unique IDs in
> each output.

## Raw outputs

- `experiments/results/validation-labels-vllm.csv`
- `experiments/results/validation-labels-sglang.csv`

The blinded inputs, codebook, prompt, and row-level labels are retained so a
human coder can independently repeat or audit the measurement later.

