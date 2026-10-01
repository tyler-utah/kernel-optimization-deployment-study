# Study 7: Reinvention (the same kernel, N copies)

**Question:** How many independent implementations of the same kernel exist across the ecosystem, and how much of it is copied or vendored rather than shared?

**Talk hook:** *Open problems: optimization intent, composability, maintenance.* If the same fused RMSNorm+quant exists in six places, every bug fix, validation, and port happens six times. That is an argument for preserving *intent* and for shared validation infrastructure.

## Method
1. **Pick about 8 kernel families:**
   - fused MoE
   - RMSNorm (+ residual, + quant)
   - rotary embedding
   - top-k/top-p sampling
   - paged/ragged attention decode
   - MLA
   - FP8 blockwise GEMM
   - all-reduce / custom collectives
2. **Locate implementations** across vLLM, SGLang (`sgl-kernel`), TensorRT-LLM, FlashInfer, AITER, llama.cpp, Liger-Kernel, and PyTorch. Use symbol and path search, and write it up as a family × repo table.
3. **Vendoring and copy detection:** use clone detection (such as token-level similarity) and "adapted from" / "copied from" comments and license headers, to measure how much is copied vs. independently written.
4. **Bug-fix propagation:** for a few known bugs in one copy, check whether and when the same fix landed in the others.

## Candidate slides
- "Fused MoE: N implementations across M projects, of which K are near-copies."
- "A bug fixed in X took D days to reach Y, and never reached Z."

## Effort
★★. Finding the implementations is fast. Measuring how the copies relate is more fiddly, but a few worked examples are enough for a slide.
