#!/usr/bin/env python3
"""cgml_run — unified pipeline driver.

Usage:
  cgml_run.py <config.json|config.yaml>           # default: --submit
  cgml_run.py --foreground <config>               # run all stages in-process
  cgml_run.py --dry-run <config>                  # write sbatches; submit none
  cgml_run.py --validate <config>                 # parse + validate; exit 0/1

Stages run in declaration order. SLURM mode chains stages with
--dependency=afterok:<prev>. Per-stage artifacts surface to the next stage as
CGML_DEP_<STAGE>_<ARTIFACT> env vars in the generated sbatch body.

Backwards compat: existing scripts (gen_benchmark_allcheckpoints.py,
sbatch_wrun_100iter_4gpu.sh, etc.) keep working. cgml_run is additive.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

import cgml_sbatch as sb
import cgml_storage as storage_mod
from cgml_config import PipelineConfig, load_config
from cgml_sbatch import SbatchSpec
from cgml_stages import REGISTRY, StageResult


def _run_root(cfg: PipelineConfig) -> Path:
    return Path(cfg.storage.root).expanduser().resolve() / cfg.run_id


def _inject_deps(spec: SbatchSpec,
                 prior_artifacts: dict[str, dict[str, str]]) -> SbatchSpec:
    """Inject CGML_DEP_<STAGE>_<ARTIFACT> env exports into the sbatch body
    so stage commands can reference prior artifacts without explicit threading.

    `prior_artifacts` is name -> {artifact_name: uri}. We use the stage's
    DECLARED artifacts (Stage.expected_artifacts()) because in SLURM mode the
    earlier stages haven't actually run yet — the artifact paths are
    deterministic from the run dir layout."""
    if not prior_artifacts:
        return spec
    exports = []
    for name, artifacts in prior_artifacts.items():
        for art_name, art_uri in artifacts.items():
            key = f"CGML_DEP_{name.upper()}_{art_name.upper()}"
            exports.append(f'export {key}="{art_uri}"')
    spec.body = "\n".join(exports) + "\n" + spec.body
    return spec


def run(cfg: PipelineConfig,
        mode: Literal["submit", "foreground", "dry-run"] = "submit",
        ) -> list[StageResult]:
    run_dir = _run_root(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2)
    )
    storage = storage_mod.from_config(cfg.storage)

    results: dict[str, StageResult] = {}
    declared_artifacts: dict[str, dict[str, str]] = {}
    prev_job_id: int | None = None

    for stage_name, _ in cfg.stages.ordered_enabled():
        cls = REGISTRY[stage_name]
        stage_dir = run_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage = cls(cfg, storage, stage_dir)

        if mode == "foreground":
            res = stage.run_inprocess(results)
            results[stage_name] = res
            print(f"[cgml_run] {stage_name}: {res.status} "
                  f"{res.message or res.artifacts}")
            if res.status == "failed":
                return list(results.values())
            continue

        spec = stage.plan()
        if spec is None:
            res = stage.run_inprocess(results)
            results[stage_name] = res
            declared_artifacts[stage_name] = res.artifacts
            continue

        sbatch_text = sb.build(_inject_deps(spec, declared_artifacts), cfg.cluster)
        sb_path = stage_dir / f"{stage_name}.sbatch"
        sb_path.write_text(sbatch_text)

        if mode == "dry-run":
            print(f"[cgml_run] DRY-RUN wrote {sb_path}")
            results[stage_name] = StageResult(
                stage=stage_name, status="skipped", log_path=sb_path)
            declared_artifacts[stage_name] = stage.expected_artifacts()
            continue

        cmd = ["sbatch", "--parsable"]
        if prev_job_id is not None:
            cmd += [f"--dependency=afterok:{prev_job_id}"]
        cmd += [str(sb_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[cgml_run] sbatch failed for {stage_name}: {r.stderr}")
            results[stage_name] = StageResult(
                stage=stage_name, status="failed", message=r.stderr.strip())
            return list(results.values())
        job_id_str = r.stdout.strip().split(";")[0]
        if not re.match(r"^\d+$", job_id_str):
            print(f"[cgml_run] could not parse job id: {r.stdout!r}")
            return list(results.values())
        prev_job_id = int(job_id_str)
        results[stage_name] = StageResult(
            stage=stage_name, status="submitted",
            slurm_job_id=prev_job_id, log_path=sb_path)
        declared_artifacts[stage_name] = stage.expected_artifacts()
        print(f"[cgml_run] {stage_name} submitted as job {prev_job_id}")

    return list(results.values())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cgml_run")
    p.add_argument("config", help="Pipeline config (JSON or YAML).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--submit", action="store_const", const="submit",
                   dest="mode", default="submit",
                   help="Submit each stage as a SLURM job (default).")
    g.add_argument("--foreground", action="store_const", const="foreground",
                   dest="mode",
                   help="Run all stages synchronously in this process.")
    g.add_argument("--dry-run", action="store_const", const="dry-run",
                   dest="mode",
                   help="Write sbatches for inspection; do not submit.")
    g.add_argument("--validate", action="store_true",
                   help="Parse + validate the config; print errors then exit.")
    args = p.parse_args(argv)

    try:
        cfg = load_config(Path(args.config))
    except Exception as e:
        print(f"[cgml_run] config validation failed: {e}", file=sys.stderr)
        return 1

    if args.validate:
        print(json.dumps(cfg.model_dump(mode="json"), indent=2))
        return 0

    results = run(cfg, mode=args.mode)
    failed = [r for r in results if r.status == "failed"]
    if failed:
        for r in failed:
            print(f"FAILED: {r.stage}: {r.message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
