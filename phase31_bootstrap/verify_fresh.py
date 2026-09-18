import subprocess
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'phase27_unfinished' / 'compiler_phase27.mirr'
NUCLEUS = ROOT / 'phase25_26' / 'nucleus.c'

def run(cmd, *, cwd=None):
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

def main():
    with tempfile.TemporaryDirectory() as td_name:
        td = Path(td_name)
        exe = td / 'nucleus'
        cc = run(['C:/zig/zig-windows-x86_64-0.13.0/zig.exe','cc','-std=c17','-Wall','-Wextra','-Wpedantic','-Werror',str(NUCLEUS),'-o',str(exe)])
        if cc.returncode:
            raise SystemExit('strict Nucleus build failed:\n' + cc.stderr)

        p1 = run([str(exe), str(SOURCE), 'run-alpha', '--input', str(SOURCE)])
        if p1.returncode:
            raise SystemExit(f'Fresh-stage bootstrap failed: rc={p1.returncode} stdout={p1.stdout!r} stderr={p1.stderr!r}')

    print('FRESH_STAGE_BOOTSTRAP_PASS')

if __name__ == '__main__':
    main()
