# Study 8: Agents in the Commit Log

**Question:** Are AI-generated optimizations already reaching upstream frameworks, how fast is that growing, and do they fare differently in review?

**Talk hook:** *§1 "candidates are abundant" and §4 (conservative path to radical automation).* This is the wild-data counterpart to the benchmark papers.

## Method
1. **Detect agent-authored PRs and commits:**
   - `Co-authored-by:` trailers (Copilot, Claude, Codex, Cursor, ...)
   - bot accounts
   - PR-body disclosures ("generated with", "AI-assisted", KernelAgent/KernelLLM mentions)
   - AI-disclosure checkboxes in PR templates, where repos have them
2. **Trend:** share of PRs per month with agent signals, overall and for kernel paths specifically.
3. **Fate:** compare merge rate, time-to-merge, review rounds, and revert rate for agent-signaled vs. other perf PRs. This reuses Study 2's pipeline.
4. **Policy scan:** do the projects have contribution policies on AI-generated code?

## Candidate slides
- "Agent-attributed PRs in vLLM and SGLang: X% in early 2025 → Y% today."
- "Agent-attributed kernel PRs merge at rate A vs. B for human ones."

## Pitfalls
- Disclosure is voluntary and inconsistent, so this is a lower bound. Say so explicitly.
- Avoid singling out individual contributors. Report aggregates only.

## Effort
★. Cheap once Study 2's data exists.
