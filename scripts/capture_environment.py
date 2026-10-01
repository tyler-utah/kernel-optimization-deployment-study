"""Capture tool and Python-package versions used for a reproduction run."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from artifact_config import ARTIFACT_ROOT


def tool_version(command: str, *args: str) -> str | None:
    if shutil.which(command) is None:
        return None
    result = subprocess.run(
        [command, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip() or result.stderr.strip()
    return output.splitlines()[0] if output else f"exit {result.returncode}"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    record = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "platform": platform.platform(),
        "python": sys.version,
        "tools": {
            "git": tool_version("git", "--version"),
            "gh": tool_version("gh", "--version"),
            "copilot": tool_version("copilot", "--version"),
        },
        "packages": {
            name: package_version(name)
            for name in ("beautifulsoup4", "matplotlib", "numpy", "requests")
        },
    }
    path = ARTIFACT_ROOT / "results" / "environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
