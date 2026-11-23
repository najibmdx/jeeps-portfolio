"""Run the whole project in order, from the downloaded files to results.

    python src/pipeline.py                 real data  (or:  bash run.sh)
    python src/pipeline.py --skip-tune     real data, without the slow tuning re-run
    python src/pipeline.py --sample        MADE-UP data, for CI and quick checks  (or:  bash run.sh --sample)

Steps: check the data files, prepare the piles, leakage audit, train, wobble, tune, column tests,
then verify the frozen choice. Every step uses the TRAIN and VALIDATION piles only.

The one-time test run is NOT in this list, on purpose. It was done once, by hand, and the test pile is
now locked. A test checks that this file never imports the step that opens the test pile.

What "tune" and "verify" mean on real data now that the choices are frozen:
  - tune.py redoes the tuning but writes to outputs/tuned_params_rerun.yaml. The committed
    tuned_params.yaml is never touched.
  - verify redoes the model and cut-off choice from that re-run and COMPARES it with the committed
    tuned_params.yaml and frozen_choice.yaml. A difference stops the run and is named. Nothing is overwritten.

Sample mode is walled off: data, outputs, tuned settings, frozen file and lock all live in a temporary
folder, and it refuses to start if that folder is inside the project folder. Its numbers mean nothing
(the data is random); it only proves that every step runs and hangs together.
"""

import argparse
import copy
import hashlib
import math
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import yaml

import get_data
import leakage_audit
import piles
import prepare
import sample
import select_final
import settings
import train
import tune
import uncertainty
import ablation

TOLERANCE = 1e-9


@dataclass
class Context:
    """Everything a step needs to know about where to read and where to write."""
    config: dict
    data_dir: Path
    out_dir: Path
    params_path: Path          # the tuned settings made by THIS run (a re-run, never the committed file)
    reference_params: Path     # what this run is compared with (real: the committed file)
    reference_frozen: Path
    notes: dict | None         # column notes (sample mode only; real mode reads column_notes.yaml)
    sample_mode: bool
    skip_tune: bool = False


# ---------- sample mode ----------

def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_config(config: dict, data_dir, feature_path, label_path) -> dict:
    """The real config, adjusted for the made-up files: their real sizes and hashes, no real-data facts,
    and smaller forests so a run stays quick."""
    cfg = copy.deepcopy(config)
    cfg["data"]["feature"].update(file=Path(feature_path).name, bytes=Path(feature_path).stat().st_size,
                                  sha256=_sha256(feature_path))
    cfg["data"]["label"].update(file=Path(label_path).name, bytes=Path(label_path).stat().st_size,
                                sha256=_sha256(label_path))
    cfg["verified_on_real_data"] = {}
    cfg["models"]["random_forest"]["n_estimators"] = 40
    cfg["evaluation"]["bootstrap_draws"] = 40
    return settings.validate_config(cfg)


def _inside(path, folder) -> bool:
    path, folder = Path(path).resolve(), Path(folder).resolve()
    return path == folder or folder in path.parents


def build_context(sample_mode: bool, work_dir=None, skip_tune: bool = False) -> Context:
    config = settings.load_config()
    project = settings.PROJECT_DIR
    if not sample_mode:
        out_dir = project / config["evaluation"]["outputs_folder"]
        return Context(config=config, data_dir=settings.DATA_DIR, out_dir=out_dir,
                       params_path=out_dir / "tuned_params_rerun.yaml",
                       reference_params=project / config["tuning"]["tuned_params_file"],
                       reference_frozen=project / config["final"]["frozen_file"],
                       notes=None, sample_mode=False, skip_tune=skip_tune)

    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="p02_sample_"))
    if _inside(work, project):
        raise ValueError(f"sample mode must work outside the project folder, not in {work}")
    data_dir = work / "data"
    feature_path, label_path, notes = sample.write_pipeline_sample(data_dir)
    out_dir = work / "outputs"
    return Context(config=sample_config(config, data_dir, feature_path, label_path), data_dir=data_dir,
                   out_dir=out_dir, params_path=out_dir / "tuned_params_rerun.yaml",
                   reference_params=work / "reference" / "tuned_params.yaml",
                   reference_frozen=work / "reference" / "frozen_choice.yaml",
                   notes=notes, sample_mode=True, skip_tune=skip_tune)


# ---------- comparing with the frozen choice ----------

def differences(new, old, path="", tol=TOLERANCE) -> list:
    """Every place two nested values differ (numbers within `tol` count as equal). Returns readable lines."""
    if isinstance(new, dict) and isinstance(old, dict):
        found = []
        for key in sorted(set(new) | set(old), key=str):
            if key == "note":
                continue
            if key not in new or key not in old:
                found.append(f"{path}{key}: present on only one side")
            else:
                found += differences(new[key], old[key], f"{path}{key}.", tol)
        return found
    if isinstance(new, (list, tuple)) and isinstance(old, (list, tuple)):
        if len(new) != len(old):
            return [f"{path[:-1]}: lengths differ ({len(new)} vs {len(old)})"]
        return [d for i, (a, b) in enumerate(zip(new, old)) for d in differences(a, b, f"{path[:-1]}[{i}].", tol)]
    if isinstance(new, (int, float)) and isinstance(old, (int, float)) and not isinstance(new, bool):
        if new == old or math.isclose(new, old, rel_tol=0, abs_tol=tol):
            return []
        return [f"{path[:-1]}: {new} vs {old}"]
    return [] if new == old else [f"{path[:-1]}: {new!r} vs {old!r}"]


# ---------- the steps (each returns 0 for success) ----------

def step_get_data(ctx: Context) -> int:
    argv = ["--check-only"] if ctx.sample_mode else []
    return get_data.main(argv, config=ctx.config, data_dir=ctx.data_dir)


def step_prepare(ctx: Context) -> int:
    return prepare.main([], config=ctx.config, data_dir=ctx.data_dir)


def step_reference(ctx: Context) -> int:
    """Sample mode only: stand in for the committed tuned_params.yaml and frozen_choice.yaml."""
    ctx.reference_params.parent.mkdir(parents=True, exist_ok=True)
    code = tune.main([], config=ctx.config, data_dir=ctx.data_dir, out_dir=ctx.out_dir, params_path=ctx.reference_params)
    if code:
        return code
    return select_final.main([], config=ctx.config, data_dir=ctx.data_dir, out_dir=ctx.out_dir,
                             params_path=ctx.reference_params, frozen_path=ctx.reference_frozen)


def step_leakage_audit(ctx: Context) -> int:
    return leakage_audit.main([], config=ctx.config, data_dir=ctx.data_dir, out_dir=ctx.out_dir, notes=ctx.notes)


def step_train(ctx: Context) -> int:
    return train.main([], config=ctx.config, data_dir=ctx.data_dir, out_dir=ctx.out_dir)


def step_uncertainty(ctx: Context) -> int:
    return uncertainty.main([], config=ctx.config, data_dir=ctx.data_dir, out_dir=ctx.out_dir)


def step_tune(ctx: Context) -> int:
    return tune.main([], config=ctx.config, data_dir=ctx.data_dir, out_dir=ctx.out_dir, params_path=ctx.params_path)


def step_ablation(ctx: Context) -> int:
    params = ctx.reference_params if ctx.skip_tune else ctx.params_path
    return ablation.main([], config=ctx.config, data_dir=ctx.data_dir, out_dir=ctx.out_dir, params_path=params)


def step_verify(ctx: Context) -> int:
    """Redo the model and cut-off choice from this run's tuning and compare with the reference files."""
    new_params = yaml.safe_load(ctx.params_path.read_text(encoding="utf-8"))
    old_params = yaml.safe_load(ctx.reference_params.read_text(encoding="utf-8"))
    problems = [f"tuned settings: {d}" for d in differences(new_params, old_params)]

    pile_set = piles.load_piles(ctx.data_dir, ctx.config)
    new_frozen, _ = select_final.build_frozen(ctx.config, select_final.tuned_config(ctx.config, ctx.params_path), pile_set)
    old_frozen = yaml.safe_load(ctx.reference_frozen.read_text(encoding="utf-8"))
    problems += [f"frozen choice: {d}" for d in differences(new_frozen, old_frozen)]

    if problems:
        print(f"VERIFY FAILED: the re-run does not reproduce {ctx.reference_frozen.name} / {ctx.reference_params.name}:")
        for line in problems[:20]:
            print("  " + line)
        print("Nothing was overwritten. Find out why before going on (different library versions or data?).")
        return 1
    print(f"verified: the re-run reproduces {ctx.reference_params.name} and {ctx.reference_frozen.name} "
          f"(model {new_frozen['model']}, {len(new_frozen['scenarios'])} cut-offs, tolerance {TOLERANCE}).")
    return 0


def steps_for(ctx: Context) -> list:
    steps = [("get_data", step_get_data), ("prepare", step_prepare)]
    if ctx.sample_mode:
        steps.append(("reference (sample only)", step_reference))
    steps += [("leakage_audit", step_leakage_audit), ("train", step_train), ("uncertainty", step_uncertainty)]
    if not ctx.skip_tune:
        steps.append(("tune", step_tune))
    steps.append(("ablation", step_ablation))
    if not ctx.skip_tune:
        steps.append(("verify", step_verify))
    return steps


def run_steps(ctx: Context, steps=None) -> tuple:
    """Run each step in order and stop at the first one that fails. Returns (all_ok, [(name, seconds)])."""
    steps = steps_for(ctx) if steps is None else steps
    done = []
    for name, step in steps:
        print(f"\n===== {name} =====")
        start = time.time()
        try:
            code = step(ctx)
        except Exception:
            traceback.print_exc()
            print(f"\nSTOPPED: step {name} raised an error (see above).")
            return False, done
        if code:
            print(f"\nSTOPPED: step {name} did not succeed (exit code {code}).")
            return False, done
        done.append((name, time.time() - start))
    return True, done


def main(argv=None) -> int:
    args = argparse.ArgumentParser(description="Run every modelling step (never the one-time test run).")
    args.add_argument("--sample", action="store_true", help="use made-up data in a temporary folder (for CI)")
    args.add_argument("--skip-tune", action="store_true",
                      help="skip the tuning re-run and the verify step (they are the slow part on real data)")
    args.add_argument("--work-dir", help="sample mode only: where to put the made-up files (must be outside the project)")
    args = args.parse_args(argv)
    if args.work_dir and not args.sample:
        print("--work-dir only makes sense with --sample")
        return 2
    ctx = build_context(args.sample, args.work_dir, args.skip_tune)
    if ctx.sample_mode:
        print(f"SAMPLE MODE: made-up data in {ctx.data_dir.parent}. The numbers mean nothing; this only checks that the steps run.")
    ok, done = run_steps(ctx)
    print("\n===== timings =====")
    for name, seconds in done:
        print(f"  {name:26} {seconds:7.1f} s")
    print(f"  {'total':26} {sum(s for _, s in done):7.1f} s")
    if not ok:
        return 1
    print("\nAll steps finished. The test pile was not opened by this run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
