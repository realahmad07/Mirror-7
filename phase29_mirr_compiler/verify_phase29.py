from pathlib import Path
import re
import subprocess
import tempfile
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "phase27_unfinished" / "compiler_phase27.mirr"
WORDS = ROOT / "phase27_unfinished" / "compiler_phase27_words.mirr"
BUILDER = ROOT / "phase27_unfinished" / "build_phase27.py"
NUCLEUS = ROOT / "phase25_26" / "nucleus.c"
INTEGRATION = ROOT / "phase28_surface_parser" / "verify_phase28.py"
PRIMS = {'+','-','dup','drop','swap','@','!','emit','next','nextc','src-pos','src-len','word-count','word-name-len','word-name-char','word-code-len','word-code-byte','word-new','word-append','word-exec','src-set-pos','word-code-start','word-patch-u16','u16-add','='}
TOKEN_RE = re.compile(r'\S+')
WORD_RE = re.compile(r'^:\s+([^\s]+)\s+(.*?)\s*;\s*$', re.S)

def parse_mirr(text):
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line: continue
        m = WORD_RE.match(line)
        if not m: raise AssertionError(f"not a complete MIRR definition: {line[:100]!r}")
        name, body = m.groups()
        if name in out: raise AssertionError(f"duplicate MIRR word: {name}")
        out[name] = TOKEN_RE.findall(body)
    return out

def run(cmd, cwd=None, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

def main():
    for p in (SOURCE, WORDS, BUILDER, NUCLEUS, INTEGRATION):
        if not p.is_file(): raise SystemExit(f"missing Phase 29 input: {p}")
    src = parse_mirr(SOURCE.read_text())
    unresolved = []
    for name, body in src.items():
        for tok in body:
            if tok in PRIMS or tok == 'exit': continue
            if tok.startswith(('branch:','0branch:')):
                if not tok.split(':',1)[1].isdigit(): unresolved.append((name,tok))
                continue
            if tok.isdigit() or re.fullmatch(r'@L\d+', tok): continue
            if tok.startswith(':') or tok == ';': continue
            if tok not in src: unresolved.append((name,tok))
    if unresolved: raise SystemExit('unresolved compiler-source references: ' + repr(unresolved[:20]))
    required = {'space?','seek-marker','scan-token','find-word','probe','tok-colon','tok-semi','digit-first?','parse-number','set-marker-pos','mark-pos','emit-byte','create-pass','compile-pass','tok-if','tok-else','tok-then','current-end','current-len','patch-at','if-open','else-open','then-close','compile-structured','run-alpha'}
    missing = sorted(required - src.keys())
    if missing: raise SystemExit(f"compiler source is missing required MIRR words: {missing}")

    with tempfile.TemporaryDirectory() as td_name:
        td = Path(td_name)
        shutil.copy2(ROOT/'phase25_26'/'compiler_phase26_words.mirr', td/'compiler_phase26_words.mirr')
        b = run(['python3', str(BUILDER)], cwd=td)
        if b.returncode: raise SystemExit('Phase 27 builder failed:\n' + b.stderr + b.stdout)
        generated = td / WORDS.name
        if generated.read_bytes() != WORDS.read_bytes(): raise SystemExit('generated compiler artifact is not byte-identical to the committed artifact')
        cc = run(['cc','-std=c17','-Wall','-Wextra','-Wpedantic','-Werror',str(NUCLEUS),'-o',str(td/'nucleus')])
        if cc.returncode: raise SystemExit('strict nucleus build failed:\n' + cc.stderr)

        diagnostic = td/'diagnostic.mirr'
        diagnostic.write_text(
            generated.read_text()
            + ': debug-base word-count drop 48 + emit exit ;\n'
            + ': debug-create create-pass word-count drop 48 + emit exit ;\n'
            + ': debug-name create-pass word-count 19 ! 18 ! 18 @ 1 - 18 ! 18 @ 19 @ word-name-len emit exit ;\n'
            + ': debug-last-char create-pass word-count 19 ! 18 ! 18 @ 1 - 18 ! 18 @ 19 @ 0 word-name-char emit exit ;\n'
            + ': debug-last-code create-pass word-count 19 ! 18 ! 18 @ 1 - 18 ! 18 @ 19 @ word-code-len drop emit exit ;\n'
            + ': debug-compile compile-structured word-count 19 ! 18 ! 18 @ 1 - 18 ! 18 @ 19 @ word-code-len drop emit exit ;\n'
        )
        structured = td/'structured.mirr'; structured.write_text(': alpha 1 IF 65 emit ELSE 66 emit THEN ;\n')
        simple = td/'simple.mirr'; simple.write_text(': alpha 65 emit ;\n')

        base = run([str(td/'nucleus'),str(diagnostic),'debug-base','--input',str(structured)])
        made = run([str(td/'nucleus'),str(diagnostic),'debug-create','--input',str(structured)])
        if base.returncode or made.returncode or len(base.stdout)!=1 or len(made.stdout)!=1:
            raise SystemExit(f"dictionary-count diagnostic failed: base={base.stdout!r}/{base.returncode}, after={made.stdout!r}/{made.returncode}")
        if ord(made.stdout) != ord(base.stdout)+1:
            raise SystemExit(f"create-pass did not add exactly one target word: base={base.stdout!r}, after={made.stdout!r}")
        n = run([str(td/'nucleus'),str(diagnostic),'debug-name','--input',str(structured)])
        if n.returncode or n.stdout != '\x05': raise SystemExit(f"word-new name diagnostic failed: rc={n.returncode} stdout={n.stdout!r} stderr={n.stderr!r}; expected target name length 5")
        c = run([str(td/'nucleus'),str(diagnostic),'debug-last-char','--input',str(structured)])
        if c.returncode or c.stdout != 'a': raise SystemExit(f"target-name diagnostic failed: rc={c.returncode} stdout={c.stdout!r} stderr={c.stderr!r}; expected first character 'a'")
        dc = run([str(td/'nucleus'),str(diagnostic),'debug-compile','--input',str(simple)])
        if dc.returncode: raise SystemExit(f"simple compile probe failed: rc={dc.returncode} stdout={dc.stdout!r} stderr={dc.stderr!r}")
        if dc.stdout != '\x05': raise SystemExit(f"simple compiled target has unexpected code length: {dc.stdout!r}")
        sc = run([str(td/'nucleus'),str(diagnostic),'debug-compile','--input',str(structured)])
        if sc.returncode: raise SystemExit(f"structured compile probe failed: rc={sc.returncode} stdout={sc.stdout!r} stderr={sc.stderr!r}")
        p1 = run([str(td/'nucleus'),str(generated),'run-alpha','--input',str(simple)])
        if p1.returncode or p1.stdout != 'A': raise SystemExit(f"simple compiler smoke test failed: rc={p1.returncode} stdout={p1.stdout!r} stderr={p1.stderr!r}")
        p2 = run([str(td/'nucleus'),str(generated),'run-alpha','--input',str(structured)])
        if p2.returncode or p2.stdout != 'A': raise SystemExit(f"structured compiler smoke test failed: rc={p2.returncode} stdout={p2.stdout!r} stderr={p2.stderr!r}")

    print('PHASE29_MIRR_SOURCE_CLOSURE_PASS')
    print(f'MIRR compiler words: {len(src)}')
    print('Host dependency remains limited to the bootstrap loader/builder; self-recompile is not claimed.')

if __name__ == '__main__': main()
