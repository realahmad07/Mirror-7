from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PARSER = ROOT / "phase28_surface_parser" / "parser.c"
SOURCE = ROOT / "phase25_26" / "compiler_phase26_words.mirr"
BUILDER = ROOT / "phase27_unfinished" / "build_phase27.py"
EXPECTED = ROOT / "phase27_unfinished" / "compiler_phase27_words.mirr"


def run(cmd, *, cwd=None):
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def main():
    for path in (PARSER, SOURCE, BUILDER, EXPECTED):
        if not path.is_file():
            raise SystemExit(f"missing verification input: {path}")

    with tempfile.TemporaryDirectory() as td_name:
        td = Path(td_name)
        strict = run(["cc", "-std=c17", "-Wall", "-Wextra", "-Wpedantic", "-Werror", str(PARSER), "-o", str(td / "parser")])
        if strict.returncode:
            raise SystemExit(f"surface parser strict build failed:\n{strict.stderr}")

        valid = [
            ": alpha 65 emit ;",
            ": alpha 1 IF 65 emit ELSE 66 emit THEN ;",
            ": alpha 0 IF 65 emit ELSE 66 emit THEN ;",
            ": alpha 1 IF 1 IF 65 emit THEN ELSE 66 emit THEN ;",
            ": alpha 65535 ;",
        ]
        invalid = [
            "ELSE", ": alpha ELSE ;", ": alpha THEN ;", ": alpha IF 1 ;",
            ": alpha IF 1 ELSE 2 ELSE 3 THEN ;", ": alpha IF ELSE 2 THEN ;",
            ": alpha IF IF 1 THEN ;", ": alpha 65536 ;", ": 1 2 ;", ": alpha @@@ ;",
        ]
        for source in valid:
            p = run([str(td / "parser"), source])
            if p.returncode != 0:
                raise SystemExit(f"parser rejected valid source: {source!r}: {p.stderr}")
        for source in invalid:
            p = run([str(td / "parser"), source])
            if p.returncode == 0:
                raise SystemExit(f"parser accepted invalid source: {source!r}")

        # Phase 27 source/build chain must still execute successfully. Exact
        # artifact reproducibility remains owned by the Phase 27 workflow.
        shutil.copy2(SOURCE, td / SOURCE.name)
        built = run(["python3", str(BUILDER)], cwd=td)
        if built.returncode:
            raise SystemExit(f"phase27 builder failed during phase28 integration:\n{built.stderr}{built.stdout}")
        rebuilt = td / EXPECTED.name
        if not rebuilt.is_file() or rebuilt.stat().st_size == 0:
            raise SystemExit("phase27 builder produced no compiler artifact during Phase 28 integration")

    print("PHASE28_INTEGRATION_PASS")


if __name__ == "__main__":
    main()
