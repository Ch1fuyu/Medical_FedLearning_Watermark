import argparse
import subprocess
import sys
import time
from pathlib import Path


REG_CONFIGS = [
    ("drift_margin", True, True),
    ("drift_only", True, False),
    ("margin_only", False, True),
    ("none", False, False),
]


def build_command(base_args, use_drift_reg, use_margin_reg):
    cmd = [sys.executable, "reg_ablation_experiment.py", *base_args]
    cmd.append("--use_drift_reg" if use_drift_reg else "--no_drift_reg")
    cmd.append("--use_margin_reg" if use_margin_reg else "--no_margin_reg")
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run reg_ablation_experiment.py for all four regularization configurations."
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="stop immediately if any configuration fails",
    )
    args, passthrough_args = parser.parse_known_args()

    project_root = Path(__file__).resolve().parent
    results = []

    print("=" * 80)
    print("Batch running four regularization configurations for reg_ablation_experiment.py")
    print("Project root:", project_root)
    print("Extra args:", " ".join(passthrough_args) if passthrough_args else "<none>")
    print("=" * 80)

    for index, (name, use_drift_reg, use_margin_reg) in enumerate(REG_CONFIGS, start=1):
        cmd = build_command(passthrough_args, use_drift_reg, use_margin_reg)
        print(f"\n[{index}/{len(REG_CONFIGS)}] Running config: {name}")
        print("Command:", " ".join(cmd))
        start_time = time.time()

        completed = subprocess.run(cmd, cwd=project_root)
        elapsed = time.time() - start_time
        success = completed.returncode == 0

        results.append({
            "name": name,
            "use_drift_reg": use_drift_reg,
            "use_margin_reg": use_margin_reg,
            "returncode": completed.returncode,
            "elapsed_sec": elapsed,
            "success": success,
        })

        status = "SUCCESS" if success else "FAILED"
        print(f"[{index}/{len(REG_CONFIGS)}] {name} finished: {status} (exit={completed.returncode}, {elapsed:.1f}s)")

        if not success and args.stop_on_error:
            break

    print("\n" + "=" * 80)
    print("Batch summary")
    print("=" * 80)
    for item in results:
        status = "SUCCESS" if item["success"] else "FAILED"
        print(
            f"- {item['name']}: {status} | drift={item['use_drift_reg']} | "
            f"margin={item['use_margin_reg']} | exit={item['returncode']} | "
            f"time={item['elapsed_sec']:.1f}s"
        )

    failed = [item for item in results if not item["success"]]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
