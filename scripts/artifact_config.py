"""Load and validate the frozen artifact configuration."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ARTIFACT_ROOT / "config" / "study.json"


def load_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def parse_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must include a timezone: {value}")
    return parsed.astimezone(dt.timezone.utc)


def validate_config(config: dict) -> None:
    repos = config.get("repositories", {})
    if set(repos) != {"vllm", "sglang"}:
        raise ValueError("Configuration must define exactly vllm and sglang")

    for study_name in ("preliminary", "deep", "lineage"):
        study = config["studies"][study_name]
        cutoff = parse_utc(study["cutoff"])
        for start_key in ("start", "pr_start", "failure_start"):
            if start_key in study and parse_utc(study[start_key]) > cutoff:
                raise ValueError(f"{study_name}.{start_key} is after its cutoff")
        if set(study["heads"]) != set(repos):
            raise ValueError(f"{study_name}.heads does not match repositories")
        for repo, sha in study["heads"].items():
            if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError(f"Invalid frozen SHA for {study_name}.{repo}")


if __name__ == "__main__":
    cfg = load_config()
    print(f"artifact configuration valid: version {cfg['artifact_version']}")
