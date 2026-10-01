"""Atomic STATUS.json helper for the kernel-lineage study.

Usage (CLI):
  python status.py show
  python status.py init            # create if missing (never clobbers)
  python status.py set <dotted.path> <json-value>
  python status.py phase <text>
  python status.py log <text>      # append to history
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

from study_config import CUTOFF_ISO

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
STATUS = os.path.join(STUDY, "STATUS.json")

CUTOFF = CUTOFF_ISO


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load():
    with open(STATUS, "r", encoding="utf-8") as f:
        return json.load(f)


def save(st):
    st["updated_at"] = now()
    fd, tmp = tempfile.mkstemp(prefix=".status-", suffix=".json", dir=STUDY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, STATUS)


def set_path(st, dotted, value):
    cur = st
    parts = dotted.split(".")
    for p in parts[:-1]:
        if isinstance(cur, list):
            cur = cur[int(p)]
        else:
            cur = cur.setdefault(p, {})
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def update(mutator, note=None):
    st = load()
    mutator(st)
    if note:
        st.setdefault("history", []).append({"at": now(), "note": note})
    save(st)
    return st


def log(note):
    return update(lambda st: None, note)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "show":
        print(json.dumps(load(), indent=2))
    elif cmd == "set":
        val = json.loads(sys.argv[3])
        update(lambda st: set_path(st, sys.argv[2], val), f"set {sys.argv[2]}")
    elif cmd == "phase":
        update(lambda st: st.__setitem__("current_phase", sys.argv[2]), f"phase -> {sys.argv[2]}")
    elif cmd == "log":
        log(" ".join(sys.argv[2:]))
    else:
        raise SystemExit(f"unknown command {cmd}")
