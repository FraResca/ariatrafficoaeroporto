#!/usr/bin/env python3
"""Shared runtime helpers for analysis scripts."""

from __future__ import annotations

import os
import re


def parse_first_int(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\d+", value)
    if not match:
        return None
    parsed = int(match.group(0))
    return parsed if parsed > 0 else None


def resolve_workers(requested_workers: int, n_tasks: int) -> int:
    if n_tasks <= 0:
        return 1
    if requested_workers > 0:
        return max(1, min(requested_workers, n_tasks))
    auto = (
        parse_first_int(os.environ.get("SLURM_CPUS_PER_TASK"))
        or parse_first_int(os.environ.get("SLURM_CPUS_ON_NODE"))
        or parse_first_int(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))
        or (os.cpu_count() or 1)
    )
    return max(1, min(auto, n_tasks))
