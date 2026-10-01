"""Print coding-batch input records as readable text for adjudicators.

    python experiments/deep-study/scripts/show.py <task> <batch-number> [start] [count]
"""

from __future__ import annotations

import json
import sys

from common import BATCHES, read_jsonl


def render(value, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(value, dict):
        return "\n".join(f"{pad}{k}: " + (("\n" + render(v, indent + 1)) if isinstance(v, (dict, list)) and v else str(v))
                         for k, v in value.items())
    if isinstance(value, list):
        return "\n".join(f"{pad}- " + (render(v, indent + 1).lstrip() if isinstance(v, (dict, list)) else str(v)) for v in value)
    return pad + str(value)


if __name__ == "__main__":
    task, batch = sys.argv[1], int(sys.argv[2])
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 6
    rows = read_jsonl(BATCHES / task / f"batch-{batch:03d}-input.jsonl")
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"# {task} batch {batch:03d}: records {start}..{min(len(rows), start + count) - 1} of {len(rows)}")
    for r in rows[start:start + count]:
        print("=" * 100)
        print(render(r))
