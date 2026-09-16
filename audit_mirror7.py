from pathlib import Path
import difflib
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
PRIMS = {'+','-','dup','drop','swap','@','!','emit','next','nextc','src-pos','src-len','word-count','word-name-len','word-name-char','word-code-len','word-code-byte','word-new','word-append','word-exec','src-set-pos','=','word-code-start','word-patch-u16','u16-add'}


def run(cmd, *, cwd=None):
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def token_size(t):
    if t == 'exit': return 1
    if t.startswith('0branch:') or t.startswith('branch:'): return 3
    if t in PRIMS: return 1
    try:
        n = int(t)
        if 0 <= n <= 255: return 2
    except ValueError:
        pass
    if t.startswith('@L'): return 3
    if t not in (':',';'): return 3
    return 0


def parse_words(path):
    words=[]
    for line in path.read_text().splitlines():
        toks=line.split()
        if not toks: continue
        if toks[0] != ':' or toks[-1] != ';' or len(toks) < 3:
            raise AssertionError(f'malformed definition: {line[:100]}')
        words.append((toks[1], toks[2:-1]))
    return words


def audit_generated(generated):
    words=parse_words(generated)
    names=[n for n,_ in words]
    if len(names)!=len(set(names)):
        raise AssertionError('duplicate word definition')
    if len(PRIMS) != 25:
        raise AssertionError(f'unexpected primitive count: {len(PRIMS)}')
    starts={}
    cur=2*len(PRIMS)
    for name,toks in words:
        starts[name]=cur
        cur += sum(token_size(t) for t in toks) + 1
    if cur > 65535:
        raise AssertionError(f'generated code exceeds u16 branch address space: {cur}')

    boundaries=set()
    for name,toks in words:
        pc=starts[name]
        boundaries.add(pc)
        for t in toks:
            boundaries.add(pc)
            pc += token_size(t)
        boundaries.add(pc)
    for name,toks in words:
        pc=starts[name]
        end=starts[name] + sum(token_size(t) for t in toks) + 1
        for t in toks:
            m=re.match(r'^(?:0branch|branch):(\d+)$',t)
            if m:
                target=int(m.group(1))
                if not (starts[name] <= target < end):
                    raise AssertionError(f'{name}: branch target {target} escapes word [{starts[name]},{end})')
                if target not in boundaries:
                    raise AssertionError(f'{name}: branch target {target} is not an instruction boundary')
            pc += token_size(t)


def main():
    required=[
        ROOT/'README.md', ROOT/'STATUS.md', ROOT/'phase25_26/compiler_phase26_words.mirr',
        ROOT/'phase25_26/nucleus.c', ROOT/'phase27_unfinished/build_phase27.py',
        ROOT/'phase27_unfinished/compiler_phase27_words.mirr', ROOT/'phase28_surface_parser/parser.c',
        ROOT/'phase28_surface_parser/verify_phase28.py'
    ]
    for p in required:
        if not p.is_file(): raise SystemExit(f'missing audit input: {p}')

    with tempfile.TemporaryDirectory() as td_name:
        td=Path(td_name)
        src=ROOT/'phase25_26/compiler_phase26_words.mirr'
        shutil.copy2(src,td/src.name)
        b=run(['python3',str(ROOT/'phase27_unfinished/build_phase27.py')],cwd=td)
        if b.returncode: raise SystemExit(f'builder failed:\n{b.stderr}{b.stdout}')
        rebuilt=td/'compiler_phase27_words.mirr'
        expected=ROOT/'phase27_unfinished/compiler_phase27_words.mirr'
        if rebuilt.read_bytes()!=expected.read_bytes():
            diff=''.join(difflib.unified_diff(expected.read_text().splitlines(True),rebuilt.read_text().splitlines(True),fromfile='committed',tofile='rebuilt',n=2))
            raise SystemExit('generated artifact differs from builder output:\n'+diff[:16000])
        audit_generated(rebuilt)

        for c in [
            ['cc','-std=c17','-Wall','-Wextra','-Wpedantic','-Werror',str(ROOT/'phase25_26/nucleus.c'),'-o',str(td/'nucleus')],
            ['cc','-std=c17','-Wall','-Wextra','-Wpedantic','-Werror','-DTEST',str(ROOT/'phase28_surface_parser/parser.c'),'-o',str(td/'parser')],
        ]:
            r=run(c)
            if r.returncode: raise SystemExit(f'strict build failed:\n{r.stderr}')

    print('MIRROR7_DEEP_AUDIT_STATIC_PASS')
    print('NOTE: absolute branch targets remain a documented architecture debt; this audit proves current targets are internally consistent, not position-independent.')
    print('NOTE: source/target dictionary separation, fresh-stage bootstrap, self-recompile, fixed-point and independent rebuild remain open acceptance gates.')


if __name__=='__main__': main()
