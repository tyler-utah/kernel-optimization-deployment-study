"""Shared paths, constants, atomic I/O, GitHub access, and checkpoint helpers
for the deep empirical follow-up study.

Every writer in this study goes through ``atomic_json``/``atomic_csv`` so that
an interrupted run never leaves a half-written cache or checkpoint.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP / "scripts"))
from artifact_config import load_config, parse_utc  # noqa: E402

CONFIG = load_config()
FIRST = EXP / "results"
FIRST_CACHE = EXP / "data" / "github"
REPOS_DIR = EXP / "data" / "repos"
DEEP = EXP / "deep-study"
DATA = DEEP / "data"
CACHE = DATA / "cache"
FIG = DEEP / "figures"
CODING = DEEP / "coding"
BATCHES = CODING / "batches"
STATUS = DEEP / "STATUS.json"

CUTOFF = parse_utc(CONFIG["studies"]["deep"]["cutoff"])
A_START = parse_utc(CONFIG["studies"]["deep"]["pr_start"])
B_START = parse_utc(CONFIG["studies"]["deep"]["failure_start"])
SEED = CONFIG["studies"]["deep"]["seed"]
REPOS = {repo: metadata["slug"] for repo, metadata in CONFIG["repositories"].items()}
HEADS = CONFIG["studies"]["deep"]["heads"]


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def iso(value) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def rel(path: Path | str) -> str:
    return str(Path(path).resolve().relative_to(EXP)).replace("\\", "/")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    os.replace(tmp, path)


def atomic_json(path: Path, data, indent: int | None = 1) -> None:
    atomic_text(path, json.dumps(data, indent=indent, ensure_ascii=False) + "\n")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# ---------------------------------------------------------------- checkpoint

class status_lock:
    """Cross-process lock (exclusive-create lock file) for STATUS.json updates."""

    def __init__(self, timeout: float = 120):
        self.path = STATUS.with_name("STATUS.json.lock")
        self.timeout = timeout

    def __enter__(self):
        start = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                if time.time() - start > self.timeout:
                    # stale lock from a killed process
                    try:
                        os.remove(self.path)
                    except FileNotFoundError:
                        pass
                    start = time.time()
                time.sleep(0.2)

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


def load_status() -> dict:
    return read_json(STATUS, {})


def save_status(data: dict) -> None:
    data["updated_at_utc"] = now_utc()
    atomic_json(STATUS, data, indent=1)


def update_status(fn) -> dict:
    with status_lock():
        data = load_status()
        fn(data)
        save_status(data)
        return data


def set_subtask(pkg: str, sub: str, **fields) -> dict:
    def apply(data):
        node = data["work_packages"][pkg]["subtasks"][sub]
        node.update(fields)
        node["updated_at_utc"] = now_utc()
        subs = data["work_packages"][pkg]["subtasks"].values()
        states = {s["status"] for s in subs}
        if states == {"complete"}:
            data["work_packages"][pkg]["status"] = "complete"
        elif "blocked" in states:
            data["work_packages"][pkg]["status"] = "blocked"
        elif states & {"in_progress", "complete"}:
            data["work_packages"][pkg]["status"] = "in_progress"
    return update_status(apply)


def set_cursor(key: str, value) -> None:
    update_status(lambda d: d.setdefault("api", {}).setdefault("cursors", {}).__setitem__(key, value))


def set_next(text: str) -> None:
    update_status(lambda d: d.__setitem__("next_action", text))


def log_failure(where: str, message: str, recovery: str) -> None:
    def apply(data):
        data.setdefault("failures", []).append({
            "at_utc": now_utc(), "where": where, "message": message[-600:], "recovery": recovery,
        })
        data["failures"] = data["failures"][-60:]
    update_status(apply)


# ---------------------------------------------------------------- GitHub

def rate_limit() -> dict:
    proc = subprocess.run(["gh", "api", "rate_limit"], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    if proc.returncode != 0:
        return {}
    return json.loads(proc.stdout)["resources"]


def wait_for(resource: str, need: int = 50) -> None:
    res = rate_limit().get(resource, {})
    if res and res.get("remaining", need) < need:
        reset = int(res["reset"])
        wait = max(5, reset - int(time.time()) + 5)
        info = {
            "resource": resource, "remaining": res["remaining"],
            "reset_epoch": reset, "reset_utc": dt.datetime.fromtimestamp(reset, dt.timezone.utc).isoformat(),
            "recorded_at_utc": now_utc(),
        }
        update_status(lambda d: d.setdefault("api", {}).__setitem__("rate_limit_wait", info))
        print(f"rate limit {resource}: sleeping {wait}s until reset", flush=True)
        time.sleep(wait)


def gh_api(args: list[str], attempts: int = 5, timeout: int = 120):
    last = ""
    for attempt in range(attempts):
        try:
            proc = subprocess.run(["gh", "api", *args], capture_output=True, text=True,
                                  encoding="utf-8", timeout=timeout)
        except subprocess.TimeoutExpired:
            last = "timeout"
            time.sleep(3 * (attempt + 1))
            continue
        if proc.returncode == 0:
            out = proc.stdout.strip()
            return json.loads(out) if out else None
        last = proc.stderr[-1500:] + proc.stdout[-500:]
        if "rate limit" in last.lower() or "secondary" in last.lower():
            time.sleep(60 * (attempt + 1))
            wait_for("graphql" if args and args[0] == "graphql" else "core")
        elif "HTTP 404" in last or "Not Found" in last:
            return None
        else:
            time.sleep(3 * (2 ** attempt))
    raise RuntimeError(f"gh api failed after {attempts}: {' '.join(args)[:300]}\n{last}")


PACE = float(os.environ.get("GQL_PACE", "0.6"))


def graphql(query: str, attempts: int = 14, timeout: int = 150) -> dict:
    """Run a GraphQL query; tolerate partial errors (e.g. deleted PRs).

    Self-paces (sleeps PACE x query time) to stay under GitHub's secondary
    CPU-time limit, and backs off for minutes when that limit is reported.
    """
    last = ""
    body = json.dumps({"query": " ".join(query.split())})
    for attempt in range(attempts):
        t0 = time.time()
        try:
            proc = subprocess.run(["gh", "api", "graphql", "--input", "-"], input=body,
                                  capture_output=True, text=True, encoding="utf-8", timeout=timeout)
        except subprocess.TimeoutExpired:
            last = "timeout"
            time.sleep(10 * (attempt + 1))
            continue
        elapsed = time.time() - t0
        out = proc.stdout.strip()
        if out:
            try:
                payload = json.loads(out)
            except json.JSONDecodeError:
                payload = None
            if payload and payload.get("data") is not None:
                time.sleep(PACE * elapsed)
                return payload
            last = out[-1500:]
        else:
            last = proc.stderr[-1500:]
        low = last.lower()
        if "secondary rate limit" in low or "abuse" in low:
            wait = min(900, 90 * (attempt + 1))
            print(f"secondary rate limit; sleeping {wait}s", flush=True)
            time.sleep(wait)
        elif "rate limit" in low:
            wait_for("graphql", need=200)
        elif "timeout" in low or "502" in low or "504" in low or "something went wrong" in low:
            time.sleep(10 * (attempt + 1))
        else:
            time.sleep(min(300, 3 * (2 ** attempt)))
    raise RuntimeError(f"graphql failed: {last}")


def git(repo: str, *args: str, timeout: int = 300) -> str:
    proc = subprocess.run(["git", "-C", str(REPOS_DIR / repo), *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr[-800:]}")
    return proc.stdout


def rng(tag: str) -> random.Random:
    return random.Random(f"{SEED}:{tag}")
