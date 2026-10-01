# Reconciliation notes

- Most event-type disagreements came from the optimize vs extend_support boundary: PR titles often said “perf,” but the codebook requires a verified optimization move for `optimize`.
- Hardware-specific support often overlapped with `specification`; when the changed contract was a dtype/shape/mode, I preferred `specification`, except where a specific hardware/library capability was decisive.
- Several ancestry disagreements were caused by blind recoding noticing PR-body references (#13601, #28854, #42095, FlashInfer commit) that first-pass textual edges underweighted.
- Conversely, some blind “extra explicit” ancestry claims lacked PR-body evidence in the sampled record; I retained textual-successor edges there.
- Move disagreements usually came from reusing a predecessor’s optimization move when the disputed PR only enabled, restored, or repaired a path without changing the kernel mechanism.
- Codebook clarification would help: distinguish “enabling an existing optimized path for a new domain” from a new optimization move, and define primary cause precedence for hardware-specific dtype/shape support.
