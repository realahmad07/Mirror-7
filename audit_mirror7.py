from pathlib import Path
import base64
import hashlib
import json
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
                if target == pc:
                    raise AssertionError(f'{name}: branch at {pc} targets itself')
            pc += token_size(t)


def audit_phase13_heritage():
    root=ROOT/'phase13_heritage'
    required=[root/'PHASE13_FINAL.md',root/'PHASE13_REPORT.md',root/'V122_REPORT.md',root/'V123_REPORT.md',root/'artifacts_phase13/source_artifact_v1.json',root/'v129/artifact_compiler.py',root/'v129/bootstrap_artifact.py',root/'tests_phase13/test_phase13.py',root/'tests_phase13/stress_phase13.py']
    for p in required:
        if not p.is_file(): raise SystemExit(f'missing Phase 13 heritage input: {p}')
    data=json.loads((root/'artifacts_phase13/source_artifact_v1.json').read_text())
    if data.get('format')!='MIRROR_SOURCE_ARTIFACT_V1': raise SystemExit('invalid preserved Phase 13 artifact format')
    templates=data.get('encoder',{}).get('templates'); rules=data.get('rules')
    if not isinstance(templates,dict) or not templates: raise SystemExit('preserved Phase 13 encoder templates missing')
    if not isinstance(rules,dict) or not rules: raise SystemExit('preserved Phase 13 language rules missing')
    allowed={'NOARG','IMM','LET_IMM'}
    for name,rule in rules.items():
        if not isinstance(name,str) or not name or not isinstance(rule,dict) or rule.get('kind') not in allowed or not isinstance(rule.get('emit'),list) or not rule['emit']:
            raise SystemExit(f'invalid preserved Phase 13 rule: {name!r}')


def audit_phase19_heritage():
    root=ROOT/'phase19_heritage'
    for rel in ['PHASE19_FINAL.md','README.md','verify_phase19_heritage.py','artifacts/compiler_artifact.json','v162/compiler.asm','v162/build_compiler.py','v162/seed_vm.c']:
        if not (root/rel).is_file(): raise SystemExit(f'missing Phase 19 heritage input: {root/rel}')
    a=json.loads((root/'artifacts/compiler_artifact.json').read_text())
    if a.get('format') != 'MIRROR_NATIVE_SELF_LANGUAGE_V2' or not a.get('raw_escape_removed'):
        raise SystemExit('invalid Phase 19 self-language artifact manifest')
    compiler=base64.b64decode(a['compiler_image_b64']); source=base64.b64decode(a['self_source_b64'])
    if hashlib.sha256(compiler).hexdigest()!=a['compiler_sha256']: raise SystemExit('Phase 19 compiler hash mismatch')
    if hashlib.sha256(source).hexdigest()!=a['self_source_sha256']: raise SystemExit('Phase 19 self-source hash mismatch')
    assert len(compiler)==76 and len(source)==77
    ops={int(k,16):v for k,v in a['language'].items()}
    if sorted(ops)!=list(range(0x10,0x24)): raise SystemExit('Phase 19 semantic token map incomplete')
    operands={int(x,16) for x in a['operand_ops']}; i=0; out=bytearray()
    while i<len(source):
        t=source[i]; i+=1
        if t==0: break
        if t not in ops: raise SystemExit(f'Phase 19 invalid source token 0x{t:02x}')
        out.append(t-0x10)
        if t in operands:
            if i>=len(source): raise SystemExit('Phase 19 missing operand')
            out.append(source[i]); i+=1
    if not source or source[-1]!=0 or bytes(out)!=compiler: raise SystemExit('Phase 19 self-source does not reconstruct compiler image')
    print('MIRROR7_PHASE19_HERITAGE_AUDIT_PASS')


def main():
    required=[
        ROOT/'README.md', ROOT/'STATUS.md', ROOT/'phase25_26/compiler_phase26_words.mirr',
        ROOT/'phase25_26/nucleus.c', ROOT/'phase27_unfinished/build_phase27.py',
        ROOT/'phase27_unfinished/compiler_phase27_words.mirr', ROOT/'phase28_surface_parser/parser.c',
        ROOT/'phase28_surface_parser/verify_phase28.py', ROOT/'phase13_heritage/verify_heritage.py',
        ROOT/'phase19_heritage/verify_phase19_heritage.py'
    ]
    for p in required:
        if not p.is_file(): raise SystemExit(f'missing audit input: {p}')
    audit_phase13_heritage()
    audit_phase19_heritage()
    with tempfile.TemporaryDirectory() as td_name:
        td=Path(td_name)
        src=ROOT/'phase25_26/compiler_phase26_words.mirr'; shutil.copy2(src,td/src.name)
        b=run(['python3',str(ROOT/'phase27_unfinished/build_phase27.py')],cwd=td)
        if b.returncode: raise SystemExit(f'builder failed:\n{b.stderr}{b.stdout}')
        rebuilt=td/'compiler_phase27_words.mirr'; expected=ROOT/'phase27_unfinished/compiler_phase27_words.mirr'
        if rebuilt.read_bytes()!=expected.read_bytes():
            print('WARNING: committed Phase 27 artifact is stale relative to current builder output; Phase 27 workflow will synchronize it.')
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