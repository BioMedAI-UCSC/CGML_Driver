"""Pydantic config schema for the unified pipeline driver (cgml_run.py).

The same model serializes to JSON (canonical) or YAML (existing west.cfg
style). Pydantic round-trips both. Stages reference fields from this schema
and nothing else — adding a new knob means adding a field here, not threading
arguments through five layers.

Heavy lifting (train, convert, preprocess, benchmark, westpa gen) all lives
in the existing CGML_Driver submodules. This file is just the contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

CURRENT_SCHEMA_VERSION = 1


class StorageConfig(BaseModel):
    """Where stage outputs (WESTPA tars, H5, checkpoints, benchmark reports)
    land. Today: local only. Cloud backends will land later behind the same
    `backend` discriminator without breaking existing configs."""

    backend: Literal["local"] = "local"
    root: Path = Field(
        ...,
        description="Filesystem root for all run artifacts. The orchestrator "
        "writes everything under <root>/<run_id>/<stage>/...",
    )


class ClusterConfig(BaseModel):
    """SLURM cluster the sbatch builder targets."""

    backend: Literal["delta", "local"] = "delta"
    account: Optional[str] = None
    partition: Optional[str] = "ghx4"
    mail_user: Optional[str] = None
    extra_modules: list[str] = Field(
        default_factory=lambda: ["cuda/12.9"],
        description="`module load` lines added to every generated sbatch.",
    )
    conda_env: str = "full_md_env"

    @model_validator(mode="after")
    def _delta_needs_account(self):
        if self.backend == "delta" and not self.account:
            raise ValueError(
                "cluster.account is required when cluster.backend == 'delta'"
            )
        return self


class WestpaGenStage(BaseModel):
    enabled: bool = True
    systems: list[str]
    iterations: int = 100
    nodes_per_system: int = 1
    gpus_per_node: int = 4
    cpus_per_task: int = 32
    mem: str = "64G"
    work_manager: Literal["threads", "mpi", "serial"] = "threads"
    walltime: str = "23:00:00"


class ConvertStage(BaseModel):
    enabled: bool = True
    systems: list[str]
    output_format: Literal["h5"] = "h5"
    num_workers: int = 16


class PreprocessStage(BaseModel):
    enabled: bool = True
    prior: str = "CA_DNA_RNA"
    refit_prior: bool = False
    num_workers: int = 16


class DistributedConfig(BaseModel):
    enabled: bool = False
    nodes: int = 1
    gpus_per_node: int = 4
    backend: Literal["nccl"] = "nccl"


class TrainStage(BaseModel):
    enabled: bool = True
    base_checkpoint: Optional[Path] = None
    epochs: int = 100
    lr: float = 5e-5
    precision: Literal["fp32", "bf16"] = "fp32"
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    extra_args: list[str] = Field(
        default_factory=list,
        description="Forwarded verbatim to the train script. Use for knobs "
        "not yet hoisted into the schema.",
    )


class BenchmarkStage(BaseModel):
    enabled: bool = True
    panel: Literal["5-system", "full"] = "5-system"
    checkpoint_range: tuple[int, int] = (99, 99)
    machine: str = "delta"
    temperature: int = 300


class StagesConfig(BaseModel):
    westpa_gen: Optional[WestpaGenStage] = None
    convert: Optional[ConvertStage] = None
    preprocess: Optional[PreprocessStage] = None
    train: Optional[TrainStage] = None
    benchmark: Optional[BenchmarkStage] = None

    def ordered_enabled(self) -> list[tuple[str, BaseModel]]:
        order = [
            ("westpa_gen", self.westpa_gen),
            ("convert", self.convert),
            ("preprocess", self.preprocess),
            ("train", self.train),
            ("benchmark", self.benchmark),
        ]
        return [(n, s) for n, s in order if s is not None and s.enabled]


class PipelineConfig(BaseModel):
    schema_version: int = CURRENT_SCHEMA_VERSION
    name: str
    run_id: str = Field(
        ...,
        description="Per-sweep directory under storage.root. Keep unique.",
    )
    storage: StorageConfig
    cluster: ClusterConfig
    stages: StagesConfig

    @model_validator(mode="after")
    def _version_check(self):
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"config schema_version={self.schema_version}; this build "
                f"speaks {CURRENT_SCHEMA_VERSION}. Migrate fields manually."
            )
        return self


def load_config(path: str | Path) -> PipelineConfig:
    p = Path(path)
    text = p.read_text()
    if p.suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return PipelineConfig.model_validate(data)


def dump_config(cfg: PipelineConfig, path: str | Path) -> None:
    p = Path(path)
    data = cfg.model_dump(mode="json")
    if p.suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore[import-untyped]

        p.write_text(yaml.safe_dump(data, sort_keys=False))
    else:
        p.write_text(json.dumps(data, indent=2))
