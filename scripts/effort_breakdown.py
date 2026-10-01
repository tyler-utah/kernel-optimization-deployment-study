"""Where does engineering effort go? Classify every first-parent commit of a repo into a purpose category.

Signals, in order:
  1. Contributor-chosen title tags / prefixes, e.g. "[Bugfix]", "[ROCm]", "fix:" (self-reported purpose)
  2. Fallback: the paths the commit touches
Separately, every commit gets an orthogonal flag: does it touch kernel code?

Input: a log produced by
    git log --no-renames --first-parent --format="@@@%H|%as|%ae|%s" --name-only HEAD > <repo>-log.txt
Usage:
    python effort_breakdown.py <repo>-log.txt <out_prefix>
Outputs: <out_prefix>-commits.csv, <out_prefix>-quarterly.csv, <out_prefix>-effort.png

This is a pilot classifier. Precision/recall must be validated on a hand-labeled sample before quoting numbers.
"""
import collections
import csv
import re
import sys

CATEGORIES = [
    ("kernel_perf", "Kernels, quantization & perf"),
    ("bugfix", "Bug fixes"),
    ("hardware", "Hardware / platform enablement"),
    ("model", "Model support"),
    ("core", "Core runtime & features"),
    ("new_workloads", "New workloads (diffusion, RL, ...)"),
    ("frontend", "API / frontend / router"),
    ("ci_test", "CI, build, tests, benchmarks"),
    ("docs", "Docs"),
    ("maint", "Refactor / maintenance / misc"),
    ("other", "Unclassified"),
]
LABEL = dict(CATEGORIES)

TAG_RULES = [  # (category, regex matched against a lowercase tag or prefix)
    ("bugfix", r"^(bug ?fix|bug|fix|fixes|hotfix|bug-fix)$"),
    ("kernel_perf", r"^(kernel|kernels|perf|performance|optim\w*|cuda|triton|cutlass|moe|fused.?moe|attention|attn|"
                    r"quant|quantization|fp8|fp4|nvfp4|mxfp4|int4|gemm|sgl-kernel|jit.?kernel|flashinfer|deepgemm)$"),
    ("hardware", r"^(rocm|amd|xpu|cpu|tpu|hardware|hpu|neuron|npu|intel|gaudi|ascend|nvidia|blackwell|b200|gb200|arm|"
                 r"mi300|mi355|gfx\w*|aiter|musa|mlu|cambricon|mps|metal|hip|intel.?gpu|sm100|sm90|hardware\]\[\w+)$"),
    ("model", r"^(model|models|new model|vlm|mm|multimodal|multi-modality|dsv\d|deepseek\w*|qwen\w*|llama\w*|gemma\w*|"
              r"mistral\w*|glm\w*|kimi\w*|minimax\w*|gpt-oss|model-specific)$"),
    ("new_workloads", r"^(diffusion|rl|multimodal.?gen|omni|video|image.?gen|embedding|pooling)$"),
    ("core", r"^(core|v1|model runner v2|scheduler|spec ?decode|spec|speculative|eagle|distributed|kv ?offload|nixl|"
             r"torch\.compile|compile|lora|metrics|engine|pd|p/d|disagg\w*|hicache|kv ?cache|eplb|ep|dp|tp|pp|"
             r"structured output|sampling|sampler|memory|cuda ?graph|async|overlap|mem_cache|radix|cache)$"),
    ("frontend", r"^(frontend|rust frontend|api|openai|router|model-gateway|grpc|entrypoints?|tool|tool.?call\w*|"
                 r"server|cli|chat|responses|serve|reasoning|parser)$"),
    ("ci_test", r"^(ci|ci/build|build|test|tests|testing|ci failure|docker|benchmark|benchmarks|bench|auto sync|"
                r"release|ci/cd|nightly|deps?|dependency|dependencies)$"),
    ("docs", r"^(doc|docs|documentation|readme|blog)$"),
    ("maint", r"^(misc|refactor|minor|chore|cleanup|clean ?up|deprecation|v0 deprecation|config|style|lint|typing|"
              r"code ?quality|revert|misc\.|nit|format|rename|log|logging)$"),
]
TAG_RULES = [(c, re.compile(p)) for c, p in TAG_RULES]


def classify_tags(subject):
    tags = [t.strip().lower() for t in re.findall(r"\[([^\]]+)\]", subject[:100])]
    m = re.match(r"^\s*([A-Za-z][\w\-/. ]{0,20}?)(\(.+?\))?!?:", subject)  # conventional commits: "fix(x):"
    if m:
        tags.append(m.group(1).strip().lower())
    if subject.lower().startswith("revert"):
        tags.insert(0, "revert")
    for t in tags:
        for cat, rx in TAG_RULES:
            if rx.match(t):
                return cat
    return None


KEYWORD_RULES = [  # for untagged subjects; checked in order
    ("bugfix", r"^\s*(fix(es|ed)?|hotfix|bug ?fix|correct|resolve)\b"),
    ("docs", r"^\s*(docs?|update (the )?readme|add docs?)\b"),
    ("ci_test", r"^\s*(bump|upgrade|pin|update) .*(version|to v?\d|dependenc|requirements)|^\s*(ci|tests?|add tests?)\b"),
    ("model", r"\b(support|add|enable)\b.*\bmodels?\b"),
    ("kernel_perf", r"\b(kernel|optimi[sz]e|speed ?up|faster|fus(e|ed|ion)|fp8|fp4|int4|gemm|triton|cuda graph)\b"),
    ("maint", r"^\s*(refactor|clean ?up|remove|rename|move|simplify|tidy|chore)\b"),
]
KEYWORD_RULES = [(c, re.compile(p, re.I)) for c, p in KEYWORD_RULES]


def classify_keywords(subject):
    for cat, rx in KEYWORD_RULES:
        if rx.search(subject):
            return cat
    return None


def is_test_or_doc(p):
    pl = p.lower()
    if re.search(r"(^|/)(tests?|testing)/", pl) or "/test_" in pl or pl.startswith("test_"):
        return "ci_test"
    if pl.startswith((".github/", ".buildkite/", "docker", "requirements")) or "dockerfile" in pl or \
            pl in ("setup.py", "pyproject.toml", "cmakelists.txt") or pl.endswith(("/pyproject.toml", "cmakelists.txt")) or \
            pl.startswith(("benchmarks/", "benchmark/", "scripts/", "tools/")):
        return "ci_test"
    if pl.startswith(("docs/", "docs_new/", "doc/")) or pl.endswith((".md", ".rst")):
        return "docs"
    return None


KERNEL_PATH = re.compile(r"(\.cu$|\.cuh$|\.hip$|(^|/)csrc/|sgl-kernel/|/kernels?/|jit_kernel/|/ops/|fused_moe|triton|cutlass)")
HW_PATH = re.compile(r"(platforms/|rocm|/xpu|xpu_|/tpu|tpu_|/hpu|hpu_|neuron|/npu|npu_|ascend|hardware_backend|aiter|"
                     r"_amd|amd_|/cpu/|cpu_worker|cpu_model|musa|mlu)")
MODEL_PATH = re.compile(r"(/models/|model_loader|/multimodal/|transformers_utils/configs|/configs/model)")
FRONTEND_PATH = re.compile(r"(entrypoints/|sgl-router/|sgl-model-gateway/|/router|openai|/serve|tool_parsers|reasoning)")
NEW_WL_PATH = re.compile(r"(multimodal_gen/|/diffusion|/rl/|verl)")


def classify_paths(files):
    votes = collections.Counter()
    aux = collections.Counter()
    for p in files:
        td = is_test_or_doc(p)
        if td:
            aux[td] += 1
            continue
        pl = p.lower()
        if NEW_WL_PATH.search(pl):
            votes["new_workloads"] += 1
        elif KERNEL_PATH.search(pl):
            votes["kernel_perf"] += 1
        elif HW_PATH.search(pl):
            votes["hardware"] += 1
        elif MODEL_PATH.search(pl):
            votes["model"] += 1
        elif FRONTEND_PATH.search(pl):
            votes["frontend"] += 1
        else:
            votes["core"] += 1
    if votes:
        return votes.most_common(1)[0][0]
    if aux:
        return aux.most_common(1)[0][0]
    return "other"


def touches_kernel(files):
    return any(KERNEL_PATH.search(p.lower()) and not is_test_or_doc(p) for p in files)


def parse(path):
    commits, cur = [], None
    for line in open(path, encoding="utf-8-sig"):
        line = line.rstrip("\n")
        if line.startswith("@@@"):
            h, d, a, s = line[3:].split("|", 3)
            cur = {"sha": h, "date": d, "author": a.lower(), "subject": s, "files": []}
            commits.append(cur)
        elif line and cur is not None:
            cur["files"].append(line)
    return commits


def quarter(d):
    y, m = int(d[:4]), int(d[5:7])
    return f"{y}Q{(m - 1) // 3 + 1}"


def main():
    log, out = sys.argv[1], sys.argv[2]
    commits = parse(log)
    for c in commits:
        t = classify_tags(c["subject"])
        k = None if t else classify_keywords(c["subject"])
        c["source"] = "tag" if t else ("keyword" if k else "path")
        c["category"] = t or k or classify_paths(c["files"])
        c["kernel"] = touches_kernel(c["files"])

    with open(f"{out}-commits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sha", "date", "quarter", "author", "category", "source", "touches_kernel", "n_files", "subject"])
        for c in commits:
            w.writerow([c["sha"], c["date"], quarter(c["date"]), c["author"], c["category"], c["source"], int(c["kernel"]),
                        len(c["files"]), c["subject"]])

    by_q = collections.defaultdict(collections.Counter)
    kern_q = collections.Counter()
    for c in commits:
        q = quarter(c["date"])
        by_q[q][c["category"]] += 1
        kern_q[q] += c["kernel"]
    quarters = sorted(by_q)
    cats = [k for k, _ in CATEGORIES]
    with open(f"{out}-quarterly.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quarter", "total", "touches_kernel"] + cats)
        for q in quarters:
            w.writerow([q, sum(by_q[q].values()), kern_q[q]] + [by_q[q][k] for k in cats])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        qs = [q for q in quarters if sum(by_q[q].values()) >= 50]
        shares = [[by_q[q][k] / sum(by_q[q].values()) for q in qs] for k in cats]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 1.2]}, sharex=True)
        a1.stackplot(range(len(qs)), shares, labels=[LABEL[k] for k in cats], alpha=0.9)
        a1.set_ylabel("share of commits")
        a1.set_ylim(0, 1)
        a1.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        a1.set_title(f"{out.split('/')[-1].split(chr(92))[-1]}: where do commits go? (pilot classifier)")
        a2.bar(range(len(qs)), [sum(by_q[q].values()) for q in qs], color="#888", label="all commits")
        a2.bar(range(len(qs)), [kern_q[q] for q in qs], color="#d62728", label="touch kernel code")
        a2.set_ylabel("commits / quarter")
        a2.legend(fontsize=8)
        a2.set_xticks(range(len(qs)))
        a2.set_xticklabels(qs, rotation=45, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{out}-effort.png", dpi=130)
    except ImportError:
        print("matplotlib not available; skipped chart")

    n = len(commits)
    src = collections.Counter(c["source"] for c in commits)
    print(f"{out}: {n} commits | classified by tag {src['tag'] / n:.0%}, keyword {src['keyword'] / n:.0%}, "
          f"path {src['path'] / n:.0%}")
    recent = [c for c in commits if c["date"] >= "2025-10-01"]
    cnt = collections.Counter(c["category"] for c in recent)
    print(f"  last 12 months ({len(recent)} commits):")
    for k in cats:
        print(f"    {LABEL[k]:<38} {cnt[k] / len(recent):6.1%}")
    print(f"    {'(touches kernel code, any category)':<38} {sum(c['kernel'] for c in recent) / len(recent):6.1%}")
    kc = collections.Counter(c["category"] for c in recent if c["kernel"])
    kn = sum(kc.values())
    print("  kernel-touching commits, by purpose: " +
          ", ".join(f"{LABEL[k]} {v / kn:.0%}" for k, v in kc.most_common(6)))


if __name__ == "__main__":
    main()
