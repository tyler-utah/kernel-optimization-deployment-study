"""Reproducible commit classification, validation sampling, analysis, and figures.

The raw inputs are name-only first-parent logs created with:
  git log --no-renames --first-parent --format="@@@%H|%as|%ae|%s" --name-only

Commands:
  python system_evolution.py classify
  python system_evolution.py sample
  python system_evolution.py metrics
  python system_evolution.py figures
  python system_evolution.py all
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import random
import re
from pathlib import Path

from artifact_config import ARTIFACT_ROOT, load_config

CONFIG = load_config()
EXP = ARTIFACT_ROOT
DATA = EXP / "data"
RESULTS = EXP / "results"
FIGURES = RESULTS / "figures"
CUTOFF = CONFIG["studies"]["preliminary"]["cutoff"][:10]
HEADLINE_START = CONFIG["studies"]["preliminary"]["start"][:10]
SEED = CONFIG["studies"]["preliminary"]["seed"]
CLASSIFIER_VERSION = "4.0"

REPOS = {
    repo: {"slug": metadata["slug"], "log": DATA / f"{repo}-log.txt"}
    for repo, metadata in CONFIG["repositories"].items()
}

CATEGORIES = [
    "kernel_performance",
    "kernel_correctness",
    "bugfix",
    "hardware_backend",
    "model_support",
    "core_runtime_serving",
    "api_frontend_router",
    "test_ci_build_benchmark_deps",
    "documentation",
    "maintenance_refactor",
    "new_workloads",
    "other",
]

LABELS = {
    "kernel_performance": "Kernel performance",
    "kernel_correctness": "Kernel correctness",
    "bugfix": "Other bug fix",
    "hardware_backend": "Hardware/backend",
    "model_support": "Model support",
    "core_runtime_serving": "Core runtime/serving",
    "api_frontend_router": "API/frontend/router",
    "test_ci_build_benchmark_deps": "Test/CI/build/bench/deps",
    "documentation": "Documentation",
    "maintenance_refactor": "Maintenance/refactor",
    "new_workloads": "New workloads",
    "other": "Other",
}

COLORS = {
    "kernel_performance": "#d62728",
    "kernel_correctness": "#ff9896",
    "bugfix": "#9467bd",
    "hardware_backend": "#ff7f0e",
    "model_support": "#2ca02c",
    "core_runtime_serving": "#1f77b4",
    "api_frontend_router": "#17becf",
    "test_ci_build_benchmark_deps": "#8c564b",
    "documentation": "#bcbd22",
    "maintenance_refactor": "#7f7f7f",
    "new_workloads": "#e377c2",
    "other": "#c7c7c7",
}

TEST_DOC_RE = re.compile(
    r"(^|/)(tests?|testing|benchmarks?|docs?|examples?|scripts?|tools?)(/|$)|"
    r"(^|/)(test_|conftest)|(^|/)\.github/|\.md$|\.rst$|"
    r"(^|/)\.claude/|(^|/)(bench|benchmark)_[^/]*\.py$|"
    r"(^|/)(dockerfile[^/]*|requirements[^/]*\.txt|pyproject\.toml|setup\.py|"
    r"cmakelists\.txt|package\.json|uv\.lock)$",
    re.I,
)
KERNEL_TEST_RE = re.compile(
    r"kernel|triton|cuda|rocm|fused_moe|attention|gemm|quant|flashinfer|cutlass",
    re.I,
)
KERNEL_IMPL_RE = re.compile(
    r"\.(cu|cuh|hip)$|(^|/)(csrc|kernels?|sgl-kernel)(/|$)|"
    r"(^|/)(triton|cutlass|flashinfer|aiter)(/|_)|jit_kernel|fused_moe|"
    r"/ops/.*\.(cc|cpp|c|py)$|model_executor/layers/quantization|"
    r"(^|/)attention/backends/|compilation/fusion|_aiter_ops\.py$|"
    r"python/sglang/srt/layers/(attention|moe|quantization|triton|linear|dp_attention)|"
    r"python/sglang/srt/hardware_backend/|python/sglang/srt/model_executor/|"
    r"python/sglang/multimodal_gen/runtime/(layers|kernels|models/vaes)/",
    re.I,
)
HARDWARE_RE = re.compile(
    r"\b(rocm|amd|xpu|tpu|hpu|npu|neuron|gaudi|ascend|intel|nvidia|apple silicon|blackwell|h20|b200|"
    r"gb200|mi[0-9]+|gfx[0-9a-z]+|musa|mlu|cambricon|metal|vulkan|sycl|cpu|"
    r"sm[0-9]{2,3})\b",
    re.I,
)
OPT_RE = re.compile(
    r"\b(perf(?:ormance)?|optimi[sz](?:e|ation|ed|ing)?|speed(?:up)?|faster|"
    r"throughput|latency|tun(?:e|ed|ing)|retun(?:e|ed|ing)|"
    r"fast paths?|reduce memory|memory reduction|accelerat(?:e|ion)|overhead|"
    r"remove redundant|affinity|early[- ]exit|hoist|fast paths?|fast regex|vectori[sz]e|"
    r"unnecessary|overlap|swizzle|autotun\w*|stag(?:e|ing) .{0,30}cop(?:y|ies)|pin mem|"
    r"avoid (?:stream sync|materializing)|reduce .{0,30}memory|reduce one|"
    r"improv\w* .{0,30}(?:throughput|latency|performance|memory|ttft|tpot)|"
    r"(?:set|make) .{0,60}(?:by )?default)\b|"
    r"\d+(?:\.\d+)?\s*(?:%|x)\b|\d+(?:\.\d+)?\s*->\s*\d",
    re.I,
)
FIX_RE = re.compile(
    r"\b(fix(?:es|ed)?|bugfix|correct(?:ness|ion)?|wrong|regress(?:ion)?|"
    r"crash|race|hang|deadlock|oom|nan|accuracy|mismatch|illegal memory|"
    r"out[- ]of[- ]bounds|overflow|ensure .*(?:consisten|valid)|unavailable|harden)\b",
    re.I,
)
MAINT_RE = re.compile(
    r"^\s*(?:\[[^\]]+\]\s*)*(refactor|cleanup|clean|remove|drop|rename|move|"
    r"simplify|chore|revert|format|lint|typing|deprecat|begin deprecat)",
    re.I,
)
DOC_RE = re.compile(r"^\s*(?:\[[^\]]+\]\s*)*(docs?|documentation|readme)\b", re.I)
CI_RE = re.compile(
    r"^\s*(?:\[[^\]]+\]\s*)*(ci|tests?|build|benchmark|bench|release|deps?|"
    r"dependenc|docker|packag|bump|upgrade|validating)\b|"
    r"\b(ci|workflow|nightly|tests?|test cases?|pr pipeline|pypi|cmake|"
    r"dockerfile|requirements|build setup|pre-commit|compile warning|benchmark|"
    r"check_env|prebuild)\b|"
    r"\b(bump|upgrade|pin|update)\b.{0,60}\b(version|to v?\d|dependenc|torch==|\.ya?ml)|"
    r"\bimport\b.{0,40}\bfrom\b|aiter\s*->",
    re.I,
)
MODEL_RE = re.compile(
    r"\b(model|llama|qwen|deepseek|gemma|mistral|glm|kimi|minimax|gpt[- ]?oss|"
    r"nemotron|ernie|internvl|phi[- ]?[0-9]|molmo|seed[- ]oss|grok[- ]?[0-9])\b",
    re.I,
)
NEW_WORKLOAD_RE = re.compile(
    r"\b(diffusion|image generation|video generation|reinforcement learning|"
    r"\bRL\b|multimodal generation|verl)\b",
    re.I,
)
FRONTEND_RE = re.compile(
    r"\b(frontend|router|gateway|openai|api|grpc|cli|tool call|chat template|"
    r"reasoning parser|entrypoint|response api|json_schema|http control|"
    r"return (?:400|401|403|404|422|429|500))\b",
    re.I,
)
CORE_RE = re.compile(
    r"\b(scheduler|scheduling|cache|distributed|serving|engine|runtime|"
    r"speculative|spec decode|disaggregat|prefill|decode|cuda graph|"
    r"torch\.compile|lora|metrics|memory manager|data parallel|tensor parallel|"
    r"expert parallel|sampling|load balanc|hicache|nixl|model runner|warmup|"
    r"proxy buffer|shared mem|cuda ipc|expert distribution|custom.?op|offload|"
    r"workers?|blockwise|attention (?:plan|backend)|kernels?|\bpp\b|\bdp\b|\btp\b|\bep\b)\b",
    re.I,
)


def parse_log(path: Path) -> list[dict]:
    commits: list[dict] = []
    cur = None
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line.startswith("@@@"):
                sha, date, author, subject = line[3:].split("|", 3)
                cur = {
                    "sha": sha,
                    "date": date,
                    "author": author.lower(),
                    "subject": subject,
                    "files": [],
                }
                commits.append(cur)
            elif line and cur is not None:
                cur["files"].append(line.replace("\\", "/"))
    return [c for c in commits if c["date"] <= CUTOFF]


def title_tags(subject: str) -> list[str]:
    tags = [x.strip().lower() for x in re.findall(r"\[([^\]]+)\]", subject[:140])]
    conventional = re.match(r"^\s*([a-z][\w./ -]{0,24}?)(?:\([^)]*\))?!?:", subject, re.I)
    if conventional:
        tags.append(conventional.group(1).strip().lower())
        scope = re.match(r"^\s*[a-z][\w./ -]{0,24}?\(([^)]*)\)", subject, re.I)
        if scope:
            tags.append(scope.group(1).strip().lower())
    return tags


def path_flags(files: list[str]) -> dict:
    production = [p for p in files if not TEST_DOC_RE.search(p)]
    test_files = [p for p in files if TEST_DOC_RE.search(p)]
    kernel_impl = [
        p for p in production
        if KERNEL_IMPL_RE.search(p) and not re.search(r"/(patch|utils?)\.py$", p, re.I)
    ]
    kernel_tests = [p for p in test_files if KERNEL_TEST_RE.search(p)]
    joined = " ".join(production)
    return {
        "production": production,
        "test_files": test_files,
        "kernel_impl": kernel_impl,
        "kernel_tests": kernel_tests,
        "hardware_path": bool(HARDWARE_RE.search(joined)),
        "model_path": bool(re.search(r"(^|/)(models?|model_loader|transformers_utils/configs)(/|$)", joined, re.I)),
        "frontend_path": bool(re.search(r"entrypoints|router|gateway|openai|grpc|tool_parser|reasoning_parser", joined, re.I)),
        "new_workload_path": bool(re.search(r"diffusion|multimodal_gen|(^|/)rl(/|$)|verl|image_gen|video_gen", joined, re.I)),
    }


def classify(commit: dict) -> dict:
    subject = commit["subject"]
    low = subject.lower()
    signal_subject = subject.replace("_", " ").replace("-", " ")
    tags = title_tags(subject)
    flags = path_flags(commit["files"])
    optimization = bool(OPT_RE.search(signal_subject) or any(t in {"perf", "performance", "optimization"} for t in tags))
    correctness = bool(FIX_RE.search(signal_subject) or any(t in {"bugfix", "bug fix", "fix", "hotfix"} for t in tags))
    if re.match(r"^\s*feat(?:\([^)]*\))?:", subject, re.I) and re.search(r"\bfixes\s+#\d+\b", subject, re.I):
        correctness = False
    hardware = bool(HARDWARE_RE.search(subject) or flags["hardware_path"])
    kernel_touch = bool(flags["kernel_impl"])
    tests_only = bool(flags["kernel_tests"]) and not kernel_touch
    tag_text = " ".join(tags)
    docs_only = bool(commit["files"]) and all(
        p.lower().endswith((".md", ".rst", ".mdx", ".ipynb"))
        or re.search(r"(^|/)(docs?|cookbook)(/|$)", p, re.I)
        for p in commit["files"]
    )
    aux_only = bool(commit["files"]) and not flags["production"]
    docs_signal = bool(
        re.search(r"\b(doc|docs|documentation|readme)\b", tag_text)
        or DOC_RE.search(subject)
        or bool(re.search(r"^\s*(?:\[[^\]]+\]\s*)*update\b.{0,80}\bdocs?\b", subject, re.I))
    )
    ci_signal = bool(
        re.search(r"\b(ci|ci/build|build|test|tests|testing|benchmark|benchmarks?|"
                  r"release|dependency|dependencies|deps|compile)\b", tag_text)
        or CI_RE.search(subject)
        or bool(re.search(r"\bbuild\b", subject, re.I))
        or bool(re.search(r"\b(profile|metric dumping|performance comparison)\b", subject, re.I))
        or bool(re.search(r"\bchore:\s*(?:bump|update config)\b", subject, re.I))
    )
    maintenance_signal = bool(
        re.search(r"\b(refactor|misc|chore|cleanup|clean up|revert|log|logging|"
                  r"mypy|typing|format|lint|auto sync|tiny|minor|config)\b", tag_text)
        or MAINT_RE.search(subject)
        or bool(re.search(r"\boptimi[sz]\w*\b.{0,40}\b(?:debug|log|logging)\b", signal_subject, re.I))
        or bool(re.search(r"\b(?:silence|suppress)\b.{0,40}\b(?:warning|log)\b", subject, re.I))
        or re.match(r"^\s*(?:\[[^\]]+\]\s*)*(tiny|small refactor|minor|clarify|"
                    r"reorganize|extract|migrate|rework|unify|centralize|lazy import|"
                    r"one read path|log(?:ging)?|update \.gitignore|"
                    r"update .* api|remove duplicate|delete dead|use .*type annotation)\b", subject, re.I)
    )

    category = None
    source = None
    confidence = None

    if (docs_signal or docs_only) and re.search(r"\b(fix|typo|broken|format|cross-reference|docstring)\b", subject, re.I):
        category, source, confidence = "maintenance_refactor", "title", "high"
    elif docs_signal or docs_only:
        category, source, confidence = "documentation", "tag" if tags else "title", "high"
    elif (
        any(t in {"rocm", "amd", "xpu", "tpu", "hpu", "npu", "cpu", "ascend", "intel", "nvidia", "apple silicon", "mlx", "arm"} for t in tags)
        and re.search(r"\b(backend|set .{0,40}default)\b", subject, re.I)
    ):
        category, source, confidence = "hardware_backend", "tag", "high"
    elif ci_signal:
        category, source, confidence = "test_ci_build_benchmark_deps", "tag" if tags else "title", "high"
    elif (
        any(t in {"perf", "performance", "optimization"} for t in tags)
        and not any(t in {"bugfix", "bug fix", "fix", "hotfix"} for t in tags)
    ):
        category, source, confidence = "kernel_performance", "tag", "high"
    elif any(t in {"metrics", "metric"} for t in tags):
        category, source, confidence = "core_runtime_serving", "tag", "high"
    elif aux_only:
        category, source, confidence = "test_ci_build_benchmark_deps", "path", "high"
    elif maintenance_signal or re.search(r"\bfix (?:the )?(?:logging|typo|formatting|grammar|dead links?)\b", subject, re.I):
        category, source, confidence = "maintenance_refactor", "tag" if tags else "title", "high"
    elif optimization and re.search(r"\bperformance regression\b", subject, re.I):
        category, source, confidence = "kernel_performance", "title", "high"
    elif correctness and kernel_touch:
        category, source, confidence = "kernel_correctness", "title+path", "high"
    elif correctness:
        category, source, confidence = "bugfix", "title", "high"
    elif (
        any(MODEL_RE.search(t) for t in tags)
        and not re.search(r"\b(perf|optimi[sz]\w*|tun\w*|speed|faster|reduce|improv\w*|overlap)\b", signal_subject, re.I)
    ):
        category, source, confidence = "model_support", "tag", "high"
    elif any(t in {"model", "models", "new model"} for t in tags) and not optimization:
        category, source, confidence = "model_support", "tag", "high"
    elif any(re.search(r"\b(frontend|router|model-gateway|openai api|parser)\b", t) for t in tags):
        category, source, confidence = "api_frontend_router", "tag", "high"
    elif hardware and (
        (
            re.match(r"^\s*(?:\[[^\]]+\]\s*)*(support|enable|port)\b", subject, re.I)
            and not re.search(r"\b(fus(?:e|ed|ion)|perf|optimi[sz]\w*|tun\w*|reduce|improv\w*)\b", signal_subject, re.I)
        )
        or re.search(r"\b(add|support)\b.{0,40}\bbackend\b", subject, re.I)
        or re.search(r"\badd custom\b.*\b(metal|rocm|xpu|npu|cpu)\b", subject, re.I)
        or re.search(r"\b(npu|xpu|rocm|amd)\b.{0,40}\b(fused op|add fused)\b", signal_subject, re.I)
        or (
            any(t in {"rocm", "amd", "xpu", "tpu", "hpu", "npu", "cpu", "ascend", "intel", "nvidia", "apple silicon"} for t in tags)
            and re.search(r"\badd\b", subject, re.I)
            and not re.search(r"\b(perf|tun|speed|faster|optimi[sz]|reduce|improv)\b", subject, re.I)
        )
    ):
        category, source, confidence = "hardware_backend", "title", "high"
    elif re.match(r"^\s*(?:\[[^\]]+\]\s*)*(fuse|fusing)\b", subject, re.I):
        category, source, confidence = "kernel_performance", "title", "high"
    elif (
        re.search(r"\b(diffusion)\b", subject, re.I)
        and (
            MODEL_RE.search(subject)
            or re.search(r"\blora adapters?\b", subject, re.I)
        )
        and not re.search(r"\b(perf|optimi[sz]\w*|tun\w*|speed|faster|reduce|improv\w*|overlap)\b", signal_subject, re.I)
    ):
        category, source, confidence = "model_support", "title", "high"
    elif optimization:
        category, source, confidence = "kernel_performance", "title", "high"
    elif (
        any(t in {"mm", "multimodal", "vlm"} for t in tags)
        and not re.search(r"\b(llama|qwen|deepseek|gemma|mistral|glm|kimi|minimax|gpt|nemotron|grok)\b", subject, re.I)
    ):
        category, source, confidence = "new_workloads", "tag", "high"
    elif NEW_WORKLOAD_RE.search(subject):
        category, source, confidence = "new_workloads", "title", "medium"
    elif any(t in {"core", "kernel", "attention", "pd", "p/d", "spec", "v1"} for t in tags):
        category, source, confidence = "core_runtime_serving", "tag", "high"
    elif any(t in {"hardware", "rocm", "amd", "xpu", "tpu", "hpu", "npu", "cpu", "ascend", "intel", "nvidia", "apple silicon"} for t in tags):
        category, source, confidence = "hardware_backend", "tag", "high"
    elif (
        MODEL_RE.search(subject)
        and re.search(r"\b(add|support|enable|loading|loader)\b", subject, re.I)
        and not optimization
    ):
        category, source, confidence = "model_support", "title", "high"
    elif MODEL_RE.search(subject) and re.search(r"\b(add|support|enable|implement|update|loader|architecture)\b", subject, re.I):
        category, source, confidence = "model_support", "title", "medium"
    elif CORE_RE.search(subject):
        category, source, confidence = "core_runtime_serving", "title", "medium"
    elif flags["model_path"] or (
        MODEL_RE.search(subject)
        and re.search(r"\b(model|qwen|deepseek|kimi|glm|llama|nemotron)\b", subject, re.I)
    ):
        category, source, confidence = "model_support", "path", "medium"
    elif (
        FRONTEND_RE.search(subject)
        and not re.search(r"router_gemm|\bkernels?\b", subject, re.I)
    ) or flags["frontend_path"]:
        category, source, confidence = "api_frontend_router", "title" if FRONTEND_RE.search(subject) else "path", "medium"
    elif hardware:
        category, source, confidence = "hardware_backend", "path", "medium"
    elif flags["production"]:
        category, source, confidence = "core_runtime_serving", "fallback", "low"
    else:
        category, source, confidence = "other", "fallback", "low"

    if kernel_touch:
        reason_map = {
            "kernel_performance": "optimization",
            "kernel_correctness": "correctness",
            "bugfix": "correctness",
            "hardware_backend": "hardware_port",
            "model_support": "model_integration",
            "test_ci_build_benchmark_deps": "test_ci",
            "maintenance_refactor": "maintenance",
        }
        reason = reason_map.get(category, "other")
    else:
        reason = ""

    return {
        "primary_purpose": category,
        "classifier_source": source,
        "classifier_confidence": confidence,
        "touches_kernel_implementation": int(kernel_touch),
        "touches_kernel_tests_only": int(tests_only),
        "hardware_backend": int(hardware),
        "optimization_claim": int(optimization),
        "correctness_fix_claim": int(correctness),
        "kernel_touch_reason": reason,
    }


def quarter(date: str) -> str:
    d = dt.date.fromisoformat(date)
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def contributor_hash(repo: str, author: str) -> str:
    return hashlib.sha256(f"{repo}:{author}".encode()).hexdigest()[:16]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gini(counts: list[int]) -> float:
    if not counts or sum(counts) == 0:
        return 0.0
    values = sorted(counts)
    n = len(values)
    return sum((2 * i - n - 1) * x for i, x in enumerate(values, 1)) / (n * sum(values))


def concentration(rows: list[dict]) -> dict:
    counts = collections.Counter(r["contributor_id"] for r in rows)
    values = sorted(counts.values(), reverse=True)
    total = sum(values)
    cumulative = 0
    bus50 = 0
    for value in values:
        cumulative += value
        bus50 += 1
        if cumulative >= total * 0.5:
            break
    return {
        "contributors": len(values),
        "top5_share": sum(values[:5]) / total if total else 0,
        "gini": gini(values),
        "contributors_for_50pct": bus50,
    }


def classify_all() -> dict[str, list[dict]]:
    all_rows: dict[str, list[dict]] = {}
    summary = {
        "study_cutoff_date": CUTOFF,
        "headline_start_date": HEADLINE_START,
        "repositories": {},
        "classifier_version": CLASSIFIER_VERSION,
    }
    for repo, info in REPOS.items():
        rows = []
        commits = parse_log(info["log"])
        for i, commit in enumerate(commits, 1):
            measured = classify(commit)
            row = {
                "sha": commit["sha"],
                "date": commit["date"],
                "quarter": quarter(commit["date"]),
                "contributor_id": contributor_hash(repo, commit["author"]),
                **measured,
                "n_files": len(commit["files"]),
                "subject": commit["subject"],
            }
            rows.append(row)
            if i % 5000 == 0:
                print(f"{repo}: classified {i}/{len(commits)}", flush=True)
        all_rows[repo] = rows
        write_csv(RESULTS / f"{repo}-commits.csv", rows)
        src = collections.Counter(r["classifier_source"] for r in rows)
        summary["repositories"][info["slug"]] = {
            "records": len(rows),
            "first_date": min(r["date"] for r in rows),
            "last_date": max(r["date"] for r in rows),
            "signal_counts": dict(src),
            "kernel_implementation_touches": sum(int(r["touches_kernel_implementation"]) for r in rows),
        }
    (RESULTS / "classification-input-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return all_rows


def load_classified(repo: str) -> list[dict]:
    with (RESULTS / f"{repo}-commits.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_samples(iteration: int = 1) -> None:
    rng = random.Random(SEED + iteration - 1)
    for repo, info in REPOS.items():
        commits_by_sha = {c["sha"]: c for c in parse_log(info["log"])}
        rows = load_classified(repo)
        strata: dict[str, list[dict]] = collections.defaultdict(list)
        for row in rows:
            if row["touches_kernel_implementation"] == "1":
                key = "kernel_touch"
            elif row["primary_purpose"] in {"kernel_performance", "kernel_correctness", "hardware_backend", "bugfix"}:
                key = row["primary_purpose"]
            elif row["classifier_confidence"] == "low" or row["classifier_source"] == "path":
                key = "ambiguous_path"
            else:
                key = "other"
            strata[key].append(row)
        allocations = {
            "kernel_touch": 45,
            "kernel_performance": 30,
            "kernel_correctness": 20,
            "hardware_backend": 30,
            "bugfix": 30,
            "ambiguous_path": 35,
            "other": 30,
        }
        selected = []
        seen = set()
        for key, target in allocations.items():
            pool = [r for r in strata.get(key, []) if r["sha"] not in seen]
            take = min(target, len(pool))
            picks = rng.sample(pool, take)
            for row in picks:
                seen.add(row["sha"])
                pop = len(strata.get(key, []))
                item = dict(row)
                item["stratum"] = key
                item["sample_weight"] = pop / take if take else 0
                item["changed_paths"] = ";".join(commits_by_sha[row["sha"]]["files"])
                selected.append(item)
        if len(selected) < 220:
            rest = [r for r in rows if r["sha"] not in seen]
            for row in rng.sample(rest, 220 - len(selected)):
                item = dict(row)
                item["stratum"] = "remainder"
                item["sample_weight"] = len(rest) / (220 - len(selected))
                item["changed_paths"] = ";".join(commits_by_sha[row["sha"]]["files"])
                selected.append(item)
        rng.shuffle(selected)
        blinded = []
        labels = []
        for index, row in enumerate(selected, 1):
            sid = f"{repo}-{index:03d}"
            blinded.append({
                "sample_id": sid,
                "repo": repo,
                "sha": row["sha"],
                "date": row["date"],
                "subject": row["subject"],
                "changed_paths": row["changed_paths"],
                "stratum": row["stratum"],
                "sample_weight": row["sample_weight"],
            })
            labels.append({
                "sample_id": sid,
                "repo": repo,
                "sha": row["sha"],
                "subject": row["subject"],
                "changed_paths": row["changed_paths"],
                "human_primary_purpose": "",
                "human_touches_kernel_implementation": "",
                "label_notes": "",
            })
        write_csv(RESULTS / f"validation-sample-{repo}.csv", blinded)
        write_csv(RESULTS / f"validation-labels-{repo}.csv", labels)
        print(f"{repo}: wrote {len(selected)} stratified validation records", flush=True)


def read_labels(repo: str) -> tuple[list[dict], dict[str, dict]]:
    samples = {r["sample_id"]: r for r in csv.DictReader((RESULTS / f"validation-sample-{repo}.csv").open(encoding="utf-8"))}
    labels = list(csv.DictReader((RESULTS / f"validation-labels-{repo}.csv").open(encoding="utf-8")))
    missing = [r["sample_id"] for r in labels if r["human_primary_purpose"] not in CATEGORIES]
    if missing:
        raise ValueError(f"{repo}: {len(missing)} validation labels are blank or invalid; first: {missing[:5]}")
    return labels, samples


def compute_metrics() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    result = {"classifier_version": CLASSIFIER_VERSION, "repositories": {}, "thresholds": {"macro_f1": 0.8, "kernel_performance_precision": 0.9}}
    for repo in REPOS:
        predicted = {r["sha"]: r for r in load_classified(repo)}
        labels, _ = read_labels(repo)
        matrix = {actual: collections.Counter() for actual in CATEGORIES}
        kernel_binary = collections.Counter()
        for label in labels:
            pred = predicted[label["sha"]]
            actual = label["human_primary_purpose"]
            matrix[actual][pred["primary_purpose"]] += 1
            kernel_binary[(int(label["human_touches_kernel_implementation"]), int(pred["touches_kernel_implementation"]))] += 1
        per_class = {}
        f1s = []
        for cat in CATEGORIES:
            tp = matrix[cat][cat]
            fp = sum(matrix[a][cat] for a in CATEGORIES if a != cat)
            fn = sum(matrix[cat][p] for p in CATEGORIES if p != cat)
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
            per_class[cat] = {"support": sum(matrix[cat].values()), "precision": precision, "recall": recall, "f1": f1}
            if f1 is not None:
                f1s.append(f1)
        macro = sum(f1s) / len(f1s)
        kp = per_class["kernel_performance"]["precision"]
        ktp = kernel_binary[(1, 1)]
        kfp = kernel_binary[(0, 1)]
        kfn = kernel_binary[(1, 0)]
        result["repositories"][repo] = {
            "n": len(labels),
            "macro_f1": macro,
            "kernel_performance_precision": kp,
            "kernel_touch_precision": ktp / (ktp + kfp) if ktp + kfp else None,
            "kernel_touch_recall": ktp / (ktp + kfn) if ktp + kfn else None,
            "per_class": per_class,
            "confusion_matrix": {a: {p: matrix[a][p] for p in CATEGORIES} for a in CATEGORIES},
        }
        fig, ax = plt.subplots(figsize=(10, 8))
        values = [[matrix[a][p] for p in CATEGORIES] for a in CATEGORIES]
        image = ax.imshow(values, cmap="Blues")
        ax.set_xticks(range(len(CATEGORIES)), [LABELS[c] for c in CATEGORIES], rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(len(CATEGORIES)), [LABELS[c] for c in CATEGORIES], fontsize=7)
        ax.set_xlabel("Classifier prediction")
        ax.set_ylabel("Human reference label")
        ax.set_title(f"{repo}: validation confusion matrix (n={len(labels)}, version {CLASSIFIER_VERSION})")
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                if value:
                    ax.text(j, i, str(value), ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(FIGURES / f"confusion-matrix-{repo}.png", dpi=160)
        plt.close(fig)
    macros = [v["macro_f1"] for v in result["repositories"].values()]
    precisions = [v["kernel_performance_precision"] for v in result["repositories"].values() if v["kernel_performance_precision"] is not None]
    result["macro_f1"] = sum(macros) / len(macros)
    result["kernel_performance_precision"] = sum(precisions) / len(precisions)
    result["passed_threshold"] = all(
        v["macro_f1"] >= 0.8 and (v["kernel_performance_precision"] or 0) >= 0.9
        for v in result["repositories"].values()
    )
    (RESULTS / "classification-metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"macro_f1": result["macro_f1"], "kernel_performance_precision": result["kernel_performance_precision"], "passed": result["passed_threshold"]}, indent=2))


def bootstrap_share(rows: list[dict], predicate, iterations: int = 2000) -> tuple[float, float, float]:
    import numpy as np

    n = len(rows)
    estimate = sum(predicate(r) for r in rows) / n
    rng = np.random.default_rng(SEED + len(rows))
    values = np.sort(rng.binomial(n, estimate, size=iterations) / n)
    return estimate, values[int(0.025 * iterations)], values[int(0.975 * iterations)]


def analyze() -> None:
    headline = []
    subsystem_rows = []
    subsystem_patterns = {
        "diffusion_image_video": re.compile(r"\b(diffusion|image generation|video generation|vae decoder|flux)\b", re.I),
        "router_gateway": re.compile(r"\b(router|gateway|routing)\b", re.I),
        "disaggregation_pd": re.compile(r"\b(disaggregat|prefill.?decode|\bP/?D\b|nixl)\b", re.I),
        "multimodal": re.compile(r"\b(multimodal|multi-modal|vision|vlm)\b", re.I),
        "rl_training": re.compile(r"\b(reinforcement learning|\bRL\b|training|verl)\b", re.I),
    }
    for repo in REPOS:
        rows = load_classified(repo)
        by_q: dict[str, list[dict]] = collections.defaultdict(list)
        for row in rows:
            by_q[row["quarter"]].append(row)
        quarterly = []
        for q in sorted(by_q):
            qrows = by_q[q]
            total = len(qrows)
            conc = concentration(qrows)
            qyear, qnum = int(q[:4]), int(q[-1])
            qend_month = qnum * 3
            qend = dt.date(qyear, qend_month, [31, 30, 30, 31][qnum - 1])
            partial = int(qend.isoformat() > CUTOFF)
            for cat in CATEGORIES:
                count = sum(r["primary_purpose"] == cat for r in qrows)
                quarterly.append({
                    "quarter": q,
                    "partial_quarter": partial,
                    "metric": "primary_purpose",
                    "category": cat,
                    "count": count,
                    "total": total,
                    "share": count / total,
                    "kernel_touch_count": sum(int(r["touches_kernel_implementation"]) for r in qrows),
                    "kernel_touch_share": sum(int(r["touches_kernel_implementation"]) for r in qrows) / total,
                    **conc,
                })
        write_csv(RESULTS / f"{repo}-quarterly.csv", quarterly)
        for q in sorted(by_q):
            qrows = by_q[q]
            for subsystem, pattern in subsystem_patterns.items():
                count = sum(bool(pattern.search(row["subject"])) for row in qrows)
                subsystem_rows.append({
                    "repo": repo,
                    "quarter": q,
                    "partial_quarter": int(q == quarter(CUTOFF)),
                    "subsystem": subsystem,
                    "title_match_count": count,
                    "total_commits": len(qrows),
                    "title_match_share": count / len(qrows),
                    "first_observed": int(count > 0 and not any(
                        prior["repo"] == repo
                        and prior["subsystem"] == subsystem
                        and int(prior["title_match_count"]) > 0
                        for prior in subsystem_rows
                    )),
                })

        recent = [r for r in rows if HEADLINE_START <= r["date"] <= CUTOFF]
        metrics = {
            "kernel_performance_share": lambda r: r["primary_purpose"] == "kernel_performance",
            "all_bugfix_share": lambda r: r["primary_purpose"] in {"kernel_correctness", "bugfix"},
            "hardware_backend_share": lambda r: r["primary_purpose"] == "hardware_backend",
            "kernel_implementation_touch_share": lambda r: int(r["touches_kernel_implementation"]) == 1,
            "maintenance_share": lambda r: r["primary_purpose"] == "maintenance_refactor",
            "establish_sustain_share": lambda r: r["primary_purpose"] in {
                "kernel_correctness", "bugfix", "hardware_backend",
                "test_ci_build_benchmark_deps", "maintenance_refactor",
            },
        }
        for category in CATEGORIES:
            metrics[f"primary_purpose_{category}_share"] = (
                lambda row, selected=category: row["primary_purpose"] == selected
            )
        for metric, predicate in metrics.items():
            estimate, low, high = bootstrap_share(recent, predicate)
            headline.append({
                "finding_id": f"{repo}-{metric}",
                "repo": repo,
                "metric": metric,
                "estimate": estimate,
                "denominator": len(recent),
                "start_date": HEADLINE_START,
                "end_date": CUTOFF,
                "source_file": f"experiments/results/{repo}-commits.csv",
                "notes": f"bootstrap_95pct_ci=[{low:.6f},{high:.6f}]; classifier v{CLASSIFIER_VERSION}",
            })
        kernel_recent = [r for r in recent if int(r["touches_kernel_implementation"])]
        for reason in ["optimization", "correctness", "hardware_port", "model_integration", "test_ci", "maintenance", "other"]:
            estimate, low, high = bootstrap_share(
                kernel_recent,
                lambda row, selected=reason: row["kernel_touch_reason"] == selected,
            )
            headline.append({
                "finding_id": f"{repo}-kernel-touch-reason-{reason}",
                "repo": repo,
                "metric": f"kernel_touch_reason_{reason}_share",
                "estimate": estimate,
                "denominator": len(kernel_recent),
                "start_date": HEADLINE_START,
                "end_date": CUTOFF,
                "source_file": f"experiments/results/{repo}-commits.csv",
                "notes": f"Among kernel implementation touches; bootstrap_95pct_ci=[{low:.6f},{high:.6f}]",
            })
        complete_quarters = [
            q for q in sorted(by_q)
            if not any(int(row["partial_quarter"]) for row in quarterly if row["quarter"] == q)
            and len(by_q[q]) >= 50
        ]
        for metric_name, predicate in {
            "kernel_performance": lambda row: row["primary_purpose"] == "kernel_performance",
            "bugfix_maintenance": lambda row: row["primary_purpose"] in {
                "kernel_correctness", "bugfix", "maintenance_refactor"
            },
        }.items():
            early_q = complete_quarters[:4]
            late_q = complete_quarters[-4:]
            early_rows = [row for q in early_q for row in by_q[q]]
            late_rows = [row for q in late_q for row in by_q[q]]
            early_share = sum(predicate(row) for row in early_rows) / len(early_rows)
            late_share = sum(predicate(row) for row in late_rows) / len(late_rows)
            headline.append({
                "finding_id": f"{repo}-{metric_name}-maturation-delta",
                "repo": repo,
                "metric": f"{metric_name}_late4_minus_first4_share",
                "estimate": late_share - early_share,
                "denominator": len(early_rows) + len(late_rows),
                "start_date": min(row["date"] for row in early_rows),
                "end_date": max(row["date"] for row in late_rows),
                "source_file": f"experiments/results/{repo}-quarterly.csv",
                "notes": f"first_complete_quarters={','.join(early_q)} share={early_share:.6f}; last_complete_quarters={','.join(late_q)} share={late_share:.6f}",
            })
        recent_conc = concentration(recent)
        for metric in ("top5_share", "gini", "contributors_for_50pct"):
            headline.append({
                "finding_id": f"{repo}-contributor-{metric}",
                "repo": repo,
                "metric": f"contributor_{metric}",
                "estimate": recent_conc[metric],
                "denominator": len(recent),
                "start_date": HEADLINE_START,
                "end_date": CUTOFF,
                "source_file": f"experiments/results/{repo}-commits.csv",
                "notes": "Aggregate only; contributor IDs are repository-scoped SHA-256 prefixes.",
            })
        for cat in CATEGORIES:
            crows = [r for r in recent if r["primary_purpose"] == cat]
            if not crows:
                continue
            cc = concentration(crows)
            for metric in ("top5_share", "gini", "contributors_for_50pct"):
                headline.append({
                    "finding_id": f"{repo}-{cat}-contributor-{metric}",
                    "repo": repo,
                    "metric": f"{cat}_contributor_{metric}",
                    "estimate": cc[metric],
                    "denominator": len(crows),
                    "start_date": HEADLINE_START,
                    "end_date": CUTOFF,
                    "source_file": f"experiments/results/{repo}-commits.csv",
                    "notes": "Within-category aggregate; no public contributor ranking.",
                })
    write_csv(RESULTS / "headline-numbers.csv", headline)
    write_csv(RESULTS / "subsystem-emergence.csv", subsystem_rows)


def figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    FIGURES.mkdir(parents=True, exist_ok=True)
    repo_rows = {repo: load_classified(repo) for repo in REPOS}
    validation_note = "Classifier validation: see classification-metrics.json"

    for repo, rows in repo_rows.items():
        by_q = collections.defaultdict(list)
        for row in rows:
            by_q[row["quarter"]].append(row)
        quarters = sorted(by_q)
        fig, (ax, volume) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]})
        bottom = [0.0] * len(quarters)
        category_bars = []
        for cat in CATEGORIES:
            shares = [sum(r["primary_purpose"] == cat for r in by_q[q]) / len(by_q[q]) for q in quarters]
            category_bars.append(
                ax.bar(range(len(quarters)), shares, bottom=bottom, label=LABELS[cat], color=COLORS[cat], width=0.85)
            )
            bottom = [a + b for a, b in zip(bottom, shares)]
        totals = [len(by_q[q]) for q in quarters]
        volume.bar(range(len(quarters)), totals, color="#555555", width=0.85)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_ylabel("Share of first-parent commits")
        volume.set_ylabel("Commits")
        volume.set_xticks(range(len(quarters)), quarters, rotation=45, ha="right")
        ax.set_title(f"{repo}: quarterly engineering-purpose mix")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        fig.text(0.01, 0.005, f"Denominator: {len(rows):,} commits, {min(r['date'] for r in rows)}–{CUTOFF}. Hatched last bar is a partial quarter. {validation_note}", fontsize=8)
        for bars in category_bars:
            bars[-1].set_hatch("//")
        volume.patches[-1].set_hatch("//")
        fig.tight_layout(rect=[0, 0.03, 0.83, 1])
        fig.savefig(FIGURES / f"quarterly-purpose-{repo}.png", dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for repo, rows in repo_rows.items():
        by_q = collections.defaultdict(list)
        for row in rows:
            by_q[row["quarter"]].append(row)
        qs = sorted(by_q)
        shares = [sum(int(r["touches_kernel_implementation"]) for r in by_q[q]) / len(by_q[q]) for q in qs]
        ax.plot(qs, shares, marker="o", label=f"{repo} (n={len(rows):,})")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_ylabel("Share touching kernel implementation")
    ax.set_xlabel("Quarter (2026Q3 is partial through Sep 28)")
    ax.set_title("Kernel implementation touch is orthogonal to engineering purpose")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.xticks(rotation=45, ha="right")
    fig.text(0.01, 0.01, f"Denominator: all first-parent commits through {CUTOFF}. Tests/docs do not count as implementation. {validation_note}", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIGURES / "kernel-touch-quarterly.png", dpi=170)
    plt.close(fig)

    reasons = ["optimization", "correctness", "hardware_port", "model_integration", "test_ci", "maintenance", "other"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (repo, rows) in zip(axes, repo_rows.items()):
        recent = [r for r in rows if HEADLINE_START <= r["date"] <= CUTOFF and int(r["touches_kernel_implementation"])]
        counts = collections.Counter(r["kernel_touch_reason"] for r in recent)
        values = [counts[x] for x in reasons]
        ax.barh(reasons, values, color=[COLORS["kernel_performance"], COLORS["bugfix"], COLORS["hardware_backend"], COLORS["model_support"], COLORS["test_ci_build_benchmark_deps"], COLORS["maintenance_refactor"], COLORS["other"]])
        ax.set_title(f"{repo} (kernel-touching n={len(recent):,})")
        ax.set_xlabel("Commits")
    fig.suptitle("Why commits touched kernel implementation, recent complete 12 months")
    fig.text(0.01, 0.01, f"Date range {HEADLINE_START}–{CUTOFF}; mutually exclusive primary-purpose mapping. {validation_note}", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(FIGURES / "kernel-touch-reasons.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for repo, rows in repo_rows.items():
        by_q = collections.defaultdict(list)
        first = min(dt.date.fromisoformat(r["date"]) for r in rows)
        for row in rows:
            by_q[row["quarter"]].append(row)
        points = []
        for q in sorted(by_q):
            qrows = by_q[q]
            qdate = dt.date(int(q[:4]), (int(q[-1]) - 1) * 3 + 1, 1)
            age = (qdate.year - first.year) * 12 + qdate.month - first.month
            share = sum(r["primary_purpose"] in {"kernel_correctness", "bugfix", "maintenance_refactor"} for r in qrows) / len(qrows)
            points.append((age, share))
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=repo)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_xlabel("Project age (months)")
    ax.set_ylabel("Bug-fix + maintenance share")
    ax.set_title("Maturation view: corrective and maintenance work over project age")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.text(0.01, 0.01, f"Denominator: all first-parent commits per quarter through {CUTOFF}; current quarter is partial. Observation, not a causal maturity estimate. {validation_note}", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIGURES / "maturation-bugfix-maintenance.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for repo, rows in repo_rows.items():
        by_q = collections.defaultdict(list)
        for row in rows:
            by_q[row["quarter"]].append(row)
        qs = sorted(by_q)
        ax.plot(qs, [concentration(by_q[q])["top5_share"] for q in qs], marker="o", label=f"{repo}: top-5 share")
        ax.plot(qs, [concentration(by_q[q])["gini"] for q in qs], marker=".", linestyle="--", label=f"{repo}: Gini")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_ylabel("Concentration statistic")
    ax.set_xlabel("Quarter (2026Q3 partial)")
    ax.set_title("Contributor concentration over time (aggregate only)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    plt.xticks(rotation=45, ha="right")
    fig.text(0.01, 0.01, "Repository-scoped hashed author emails; aliases are not merged. No individual names or rankings.", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIGURES / "contributor-concentration.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["classify", "sample", "metrics", "analyze", "figures", "all"])
    parser.add_argument("--iteration", type=int, default=1)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.command in {"classify", "all"}:
        classify_all()
    if args.command in {"sample", "all"}:
        make_samples(args.iteration)
    if args.command in {"metrics", "all"}:
        compute_metrics()
    if args.command in {"analyze", "all"}:
        analyze()
    if args.command in {"figures", "all"}:
        figures()


if __name__ == "__main__":
    main()
