"""Small helpers shared by the BD-HSD benchmark drivers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Any


def token_hash(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def matches_reference(
    token_ids: list[int], reference_token_ids: list[int], max_tokens: int
) -> bool:
    """Accept natural-EOS equality or equality of a fixed-length prefix."""
    return token_ids == reference_token_ids or (
        len(token_ids) == max_tokens
        and token_ids == reference_token_ids[:max_tokens]
    )


def add_counter_delta(
    total: dict[str, int | float], delta: dict[str, int | float]
) -> None:
    for key, value in delta.items():
        total[key] = total.get(key, 0) + value


def gpu_snapshot() -> list[dict[str, str]] | None:
    """Capture coarse GPU state outside measured request intervals."""
    fields = "index,name,temperature.gpu,clocks.sm,power.draw,utilization.gpu"
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    names = fields.split(",")
    return [
        dict(zip(names, (value.strip() for value in line.split(","))))
        for line in output.splitlines()
        if line.strip()
    ]
