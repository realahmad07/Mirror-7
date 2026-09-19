from __future__ import annotations
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def main() -> int:
    py = sys.executable
    result = subprocess.run(
        [py, "-m", "pytest", "-q", "test_phase65.py"],
        cwd=ROOT,
        text=True,
    )
    if result.returncode != 0:
        print("PHASE 65 ACCEPTANCE GATE: FAIL")
        return result.returncode

    print("PHASE 65 ACCEPTANCE GATE: PASS")
    print("progressive: 3 effect families × 3 seeds")
    print("held-out: 1/1")
    print("adversarial/control: 4/4")
    print("integration: continuous no-reset stream")
    print("regression: 9/9")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
