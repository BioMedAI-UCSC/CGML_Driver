"""Stage runners.

Each runner is a *thin* wrapper that maps the validated config into a call to
the relevant existing submodule script (base_model/, benchmark/, westpa_prop/,
tools/setup_tooling/). No business logic lives here — only orchestration glue.

The Stage interface has two methods:
- `plan()` returns an SbatchSpec for SLURM submission, or None for in-process.
- `run_inprocess()` runs the same work synchronously when --foreground is set.

Both methods need to resolve dependencies on prior stages' artifacts. In SLURM
mode the orchestrator injects `CGML_DEP_<stage>_<artifact>` env vars before
the stage body runs; in-process gets the deps dict directly.
"""

from __future__ import annotations

import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cgml_sbatch import SbatchSpec

CGML_DRIVER = Path(__file__).resolve().parent
BASE_MODEL = CGML_DRIVER / "base_model"
WESTPA_TRAINING_DATA = Path("/u/awaghili/FoundationModel/WESTPATrainingData")


@dataclass
class StageResult:
    stage: str
    status: str  # "completed" | "skipped" | "failed" | "submitted"
    artifacts: dict[str, str] = field(default_factory=dict)
    slurm_job_id: Optional[int] = None
    log_path: Optional[Path] = None
    message: Optional[str] = None


class Stage(ABC):
    name: str = ""

    def __init__(self, cfg, storage, run_dir: Path):
        self.cfg = cfg
        self.storage = storage
        self.run_dir = run_dir

    @abstractmethod
    def plan(self) -> Optional[SbatchSpec]: ...

    @abstractmethod
    def run_inprocess(self, deps: dict[str, StageResult]) -> StageResult: ...

    def expected_artifacts(self) -> dict[str, str]:
        """Deterministic outputs this stage will produce. Used by the
        orchestrator to chain dependencies in SLURM mode (where stage N is
        submitted before stage N-1 has actually written anything). Default:
        no declared outputs.
        """
        return {}


# ---------- WESTPA generation ----------


class WestpaGenRunner(Stage):
    name = "westpa_gen"

    def plan(self) -> Optional[SbatchSpec]:
        s = self.cfg.stages.westpa_gen
        list_file = self.run_dir / "systems.txt"
        list_file.write_text("\n".join(s.systems) + "\n")
        body = f"""
LIST_FILE="{list_file}"
pdb=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${{LIST_FILE}}")
WP="{WESTPA_TRAINING_DATA}/${{pdb}}/westpa_prop"
cd "${{WP}}"
export WEST_SIM_ROOT="${{WP}}"
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=0

sed -e 's|^\\(\\s*\\)max_total_iterations:.*|\\1max_total_iterations: {s.iterations}|' \\
    -e 's|^\\(\\s*\\)num_gpus:.*|\\1num_gpus: {s.gpus_per_node}|' \\
    west.cfg > west_cgml.cfg

w_run -r west_cgml.cfg --work-manager={s.work_manager} --n-workers={s.gpus_per_node}
"""
        return SbatchSpec(
            job_name=f"cgml-wgen-{self.cfg.run_id}",
            nodes=s.nodes_per_system,
            ntasks_per_node=1,
            gpus_per_node=s.gpus_per_node,
            cpus_per_task=s.cpus_per_task,
            mem=s.mem,
            walltime=s.walltime,
            output=self.run_dir / "westpa_gen-%A_%a.out",
            body=body,
            extra_directives=[f"--array=0-{len(s.systems) - 1}"],
        )

    def run_inprocess(self, deps: dict[str, StageResult]) -> StageResult:
        return StageResult(
            stage=self.name, status="failed",
            message="WESTPA generation is SLURM-only today; use --submit "
            "(or set stages.westpa_gen.enabled=false to reuse existing data).",
        )


# ---------- Convert WESTPA -> training H5 ----------


class ConvertRunner(Stage):
    name = "convert"
    # TODO(later): expose convert_westpa_to_dataset_parallel.py (cgschnet
    # alex_senior_thesis) as an opt-in MPI-parallel path. Not stubbed.

    def _outdir(self) -> Path:
        d = self.run_dir / "h5"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def expected_artifacts(self) -> dict[str, str]:
        return {"h5_dir": str(self._outdir())}

    def plan(self) -> Optional[SbatchSpec]:
        s = self.cfg.stages.convert
        list_file = self.run_dir / "convert_systems.txt"
        list_file.write_text("\n".join(s.systems) + "\n")
        out = self._outdir()
        body = f"""
LIST_FILE="{list_file}"
pdb=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${{LIST_FILE}}")
WP="{WESTPA_TRAINING_DATA}/${{pdb}}/westpa_prop"
python {BASE_MODEL}/convert_westpa.py \\
    --westpa-dir "${{WP}}" \\
    --base-traj "${{WP}}/simulated_topology.pdb" \\
    --simulated-topology "${{WP}}/simulated_topology.pdb" \\
    --output "{out}/${{pdb}}.h5" \\
    --protein-name "${{pdb}}" \\
    --num-workers {s.num_workers}
"""
        return SbatchSpec(
            job_name=f"cgml-conv-{self.cfg.run_id}",
            # ghx4 / DeltaAI rejects jobs requesting 0 GPUs even when the
            # work is CPU-only. Request 1 GPU; the convert script will not
            # use it.
            nodes=1, ntasks_per_node=1, gpus_per_node=1,
            cpus_per_task=max(s.num_workers, 2),
            mem="32G", walltime="04:00:00",
            output=self.run_dir / "convert-%A_%a.out",
            body=body,
            extra_directives=[f"--array=0-{len(s.systems) - 1}"],
        )

    def run_inprocess(self, deps: dict[str, StageResult]) -> StageResult:
        s = self.cfg.stages.convert
        out = self._outdir()
        for pdb in s.systems:
            wp = WESTPA_TRAINING_DATA / pdb / "westpa_prop"
            r = subprocess.run([
                sys.executable, str(BASE_MODEL / "convert_westpa.py"),
                "--westpa-dir", str(wp),
                "--base-traj", str(wp / "simulated_topology.pdb"),
                "--simulated-topology", str(wp / "simulated_topology.pdb"),
                "--output", str(out / f"{pdb}.h5"),
                "--protein-name", pdb,
                "--num-workers", str(s.num_workers),
            ], capture_output=True, text=True)
            if r.returncode != 0:
                return StageResult(stage=self.name, status="failed",
                                   message=f"{pdb}: {r.stderr[-2000:]}")
        return StageResult(stage=self.name, status="completed",
                           artifacts={"h5_dir": str(out)})


# ---------- Preprocess ----------


class PreprocessRunner(Stage):
    name = "preprocess"

    def _outdir(self) -> Path:
        d = self.run_dir / "preprocessed"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def expected_artifacts(self) -> dict[str, str]:
        return {"preprocessed_dir": str(self._outdir())}

    def plan(self) -> Optional[SbatchSpec]:
        s = self.cfg.stages.preprocess
        out = self._outdir()
        body = (
            'INPUT_DIR="${CGML_DEP_CONVERT_H5_DIR}"\n'
            f'python {BASE_MODEL}/preprocess.py "${{INPUT_DIR}}" '
            f"-o {out} --prior {s.prior} --num-cores {s.num_workers}\n"
        )
        return SbatchSpec(
            job_name=f"cgml-prep-{self.cfg.run_id}",
            # ghx4 / DeltaAI rejects 0-GPU jobs; request 1 even though
            # preprocess is CPU-only.
            nodes=1, ntasks_per_node=1, gpus_per_node=1,
            cpus_per_task=s.num_workers,
            mem="64G", walltime="04:00:00",
            output=self.run_dir / "preprocess-%j.out",
            body=body,
        )

    def run_inprocess(self, deps: dict[str, StageResult]) -> StageResult:
        s = self.cfg.stages.preprocess
        out = self._outdir()
        input_dir = Path(deps["convert"].artifacts["h5_dir"])
        r = subprocess.run([
            sys.executable, str(BASE_MODEL / "preprocess.py"),
            str(input_dir), "-o", str(out),
            "--prior", s.prior, "--num-cores", str(s.num_workers),
        ], capture_output=True, text=True)
        if r.returncode != 0:
            return StageResult(stage=self.name, status="failed",
                               message=r.stderr[-2000:])
        return StageResult(stage=self.name, status="completed",
                           artifacts={"preprocessed_dir": str(out)})


# ---------- Train ----------


class TrainRunner(Stage):
    name = "train"
    # train.py is the single training entry point. `--distributed` activates
    # the DDP code path; default keeps the legacy single-GPU DataParallel path.

    def _outdir(self) -> Path:
        d = self.run_dir / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def expected_artifacts(self) -> dict[str, str]:
        return {"checkpoints_dir": str(self._outdir())}

    def _config_yaml(self) -> Path:
        # Mirrors the user's sbatch_finetune_fm45R_westpa.sh default.
        return CGML_DRIVER / "configs" / "config.yaml"

    def plan(self) -> Optional[SbatchSpec]:
        s = self.cfg.stages.train
        out = self._outdir()
        if s.distributed.enabled:
            world = s.distributed.nodes * s.distributed.gpus_per_node
            cmd = self._train_cmd(s, out, distributed=True)
            body = (
                'export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)\n'
                "export MASTER_PORT=29500\n"
                f"srun -n {world} --gpus-per-task=1 {cmd}\n"
            )
            return SbatchSpec(
                job_name=f"cgml-train-{self.cfg.run_id}",
                nodes=s.distributed.nodes,
                ntasks_per_node=s.distributed.gpus_per_node,
                gpus_per_node=s.distributed.gpus_per_node,
                cpus_per_task=8, mem="64G", walltime="23:00:00",
                output=self.run_dir / "train-%j.out",
                body=body,
                extra_directives=["--gpus-per-task=1"],
            )
        body = self._train_cmd(s, out, distributed=False) + "\n"
        return SbatchSpec(
            job_name=f"cgml-train-{self.cfg.run_id}",
            nodes=1, ntasks_per_node=1, gpus_per_node=1,
            cpus_per_task=8, mem="64G", walltime="23:00:00",
            output=self.run_dir / "train-%j.out",
            body=body,
        )

    def _train_cmd(self, s, out: Path, distributed: bool,
                   input_path: str | None = None) -> str:
        # train.py CLI: positional `input result`, --gpus, --epochs, --lr, etc.
        # In SLURM mode, the input dir comes from a dep env var.
        in_path = input_path or "${CGML_DEP_PREPROCESS_PREPROCESSED_DIR}"
        parts = [
            "python", str(BASE_MODEL / "train.py"),
            in_path, str(out),
            "--config", str(self._config_yaml()),
            "--gpus", "0",
            "--epochs", str(s.epochs),
            "--lr", str(s.lr),
            "--precision", s.precision,
        ]
        if distributed:
            parts.append("--distributed")
        if s.base_checkpoint:
            parts += ["--base-checkpoint", str(s.base_checkpoint)]
        parts += s.extra_args
        return " ".join(parts)

    def run_inprocess(self, deps: dict[str, StageResult]) -> StageResult:
        s = self.cfg.stages.train
        if s.distributed.enabled:
            return StageResult(
                stage=self.name, status="failed",
                message="distributed training requires SLURM (srun launches "
                "the ranks); use --submit or set distributed.enabled=false.",
            )
        out = self._outdir()
        pre = Path(deps["preprocess"].artifacts["preprocessed_dir"])
        cmd_str = self._train_cmd(s, out, distributed=False, input_path=str(pre))
        r = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            return StageResult(stage=self.name, status="failed",
                               message=r.stderr[-2000:])
        return StageResult(stage=self.name, status="completed",
                           artifacts={"checkpoints_dir": str(out)})


# ---------- Benchmark ----------


_5_SYSTEM_PANEL = ["1PGB", "1KU7", "2OOB", "1BQ9", "2L7B"]


class BenchmarkRunner(Stage):
    name = "benchmark"

    def _outdir(self) -> Path:
        d = self.run_dir / "benchmark"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def expected_artifacts(self) -> dict[str, str]:
        return {"benchmark_dir": str(self._outdir())}

    def plan(self) -> Optional[SbatchSpec]:
        s = self.cfg.stages.benchmark
        out = self._outdir()
        parts = [
            "python", str(CGML_DRIVER / "gen_benchmark_allcheckpoints.py"),
            "${CGML_DEP_TRAIN_CHECKPOINTS_DIR}",
            "--machine", s.machine,
            "--temperature", str(s.temperature),
            "--output-dir", str(out),
            "--start", str(s.checkpoint_range[0]),
            "--end", str(s.checkpoint_range[1]),
        ]
        if s.panel != "full":
            parts += ["--proteins", *_5_SYSTEM_PANEL]
        return SbatchSpec(
            job_name=f"cgml-bench-{self.cfg.run_id}",
            nodes=1, ntasks_per_node=1, gpus_per_node=1,
            cpus_per_task=8, mem="64G", walltime="06:00:00",
            output=self.run_dir / "benchmark-%j.out",
            body=" ".join(parts) + "\n",
        )

    def run_inprocess(self, deps: dict[str, StageResult]) -> StageResult:
        s = self.cfg.stages.benchmark
        out = self._outdir()
        ckpt = Path(deps["train"].artifacts["checkpoints_dir"])
        cmd = [
            sys.executable, str(CGML_DRIVER / "gen_benchmark_allcheckpoints.py"),
            str(ckpt), "--machine", s.machine,
            "--temperature", str(s.temperature),
            "--output-dir", str(out),
            "--start", str(s.checkpoint_range[0]),
            "--end", str(s.checkpoint_range[1]),
        ]
        if s.panel != "full":
            cmd += ["--proteins", *_5_SYSTEM_PANEL]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return StageResult(stage=self.name, status="failed",
                               message=r.stderr[-2000:])
        return StageResult(stage=self.name, status="completed",
                           artifacts={"benchmark_dir": str(out)})


REGISTRY: dict[str, type[Stage]] = {
    "westpa_gen": WestpaGenRunner,
    "convert": ConvertRunner,
    "preprocess": PreprocessRunner,
    "train": TrainRunner,
    "benchmark": BenchmarkRunner,
}
