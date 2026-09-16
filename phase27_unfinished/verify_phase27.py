from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "phase27_unfinished"
SOURCE = ROOT / "phase25_26" / "compiler_phase26_words.mirr"
BUILDER = PHASE / "build_phase27.py"
EXPECTED = PHASE / "compiler_phase27_words.mirr"
NUCLEUS = ROOT / "phase25_26" / "nucleus.c"


def run(cmd, *, cwd=None, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def main():
    for path in (SOURCE, BUILDER, EXPECTED, NUCLEUS):
        if not path.is_file():
            raise SystemExit(f"missing verification input: {path}")

    with tempfile.TemporaryDirectory() as td_name:
        td = Path(td_name)
        shutil.copy2(SOURCE, td / SOURCE.name)
        r = run(["python3", str(BUILDER)], cwd=td)
        if r.returncode:
            raise SystemExit(f"phase27 builder failed:\n{r.stderr}{r.stdout}")
        rebuilt = td / EXPECTED.name
        if not rebuilt.is_file():
            raise SystemExit("phase27 builder produced no compiler_phase27_words.mirr")
        if rebuilt.read_bytes() != EXPECTED.read_bytes():
            raise SystemExit("phase27 generated artifact is not reproducible from the committed builder")

        cc = run(["cc", "-std=c17", "-Wall", "-Wextra", "-Wpedantic", "-Werror", str(NUCLEUS), "-o", str(td / "nucleus")])
        if cc.returncode:
            raise SystemExit(f"nucleus strict build failed:\n{cc.stderr}")

        cases = [
            (": alpha 65 emit ;\n", "A"),
            (": alpha 1 IF 65 emit ELSE 66 emit THEN ;\n", "A"),
            (": alpha 0 IF 65 emit ELSE 66 emit THEN ;\n", "B"),
            (": alpha 1 IF 1 IF 65 emit THEN ELSE 66 emit THEN ;\n", "A"),
        ]
        for i, (source, want) in enumerate(cases):
            inp = td / f"case_{i}.mirr"
            inp.write_text(source)
            p = run([str(td / "nucleus"), str(rebuilt), "run-alpha", "--input", str(inp)])
            if p.returncode != 0 or p.stdout != want:
                trace = run([str(td / "nucleus"), str(rebuilt), "run-alpha", "--input", str(inp)], env={**__import__('os').environ, "TRACE_VM":"1", "TRACE_CODE":"1"})
                raise SystemExit(f"phase27 case {i} failed: rc={p.returncode}, stdout={p.stdout!r}, stderr={p.stderr!r}\nTRACE:\n{trace.stderr[-20000:]}")

    print("PHASE27_VERIFICATION_PASS")


if __name__ == "__main__":
    main()
