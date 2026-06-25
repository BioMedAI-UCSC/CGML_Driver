"""Sbatch builder.

A `SbatchSpec` describes the resources a stage needs; `build()` composes
them with the cluster config into runnable sbatch text. Every generated
sbatch is self-sufficient (loads cuda + activates env) so it can be re-run
manually from disk after the orchestrator submits it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SbatchSpec:
    job_name: str
    nodes: int = 1
    ntasks_per_node: int = 1
    gpus_per_node: int = 0
    cpus_per_task: int = 1
    mem: str = "16G"
    walltime: str = "01:00:00"
    output: Optional[Path] = None
    body: str = ""
    extra_directives: list[str] = field(default_factory=list)


def build(spec: SbatchSpec, cluster) -> str:
    """Compose sbatch text. Header from cluster; resources + body from spec."""
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={spec.job_name}",
    ]
    if cluster.account:
        lines += [f"#SBATCH --account={cluster.account}"]
    if cluster.partition:
        lines += [f"#SBATCH --partition={cluster.partition}"]
    lines += [
        f"#SBATCH --time={spec.walltime}",
        f"#SBATCH --nodes={spec.nodes}",
        f"#SBATCH --ntasks-per-node={spec.ntasks_per_node}",
        f"#SBATCH --cpus-per-task={spec.cpus_per_task}",
        f"#SBATCH --mem={spec.mem}",
    ]
    if spec.gpus_per_node > 0:
        lines += [f"#SBATCH --gpus-per-node={spec.gpus_per_node}"]
    if spec.output:
        lines += [f"#SBATCH --output={spec.output}"]
    if cluster.mail_user:
        lines += [
            "#SBATCH --mail-type=END,FAIL",
            f"#SBATCH --mail-user={cluster.mail_user}",
        ]
    for d in spec.extra_directives:
        lines += [f"#SBATCH {d}"]

    lines += ["", "set -euo pipefail", "", "if command -v module >/dev/null; then"]
    for m in cluster.extra_modules:
        lines += [f"    module load {m} 2>/dev/null || true"]
    lines += [
        "fi",
        "source /u/awaghili/micromamba/etc/profile.d/mamba.sh",
        f"micromamba activate {cluster.conda_env}",
        "",
        spec.body,
    ]
    return "\n".join(lines) + "\n"
