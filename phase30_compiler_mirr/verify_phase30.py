from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'phase27_unfinished' / 'compiler_phase27.mirr'
NUCLEUS = ROOT / 'phase25_26' / 'nucleus.c'
PRIMS = {'+','-','dup','drop','swap','@','!','emit','next','nextc','src-pos','src-len','word-count','word-name-len','word-name-char','word-code-len','word-code-byte','word-new','word-append','word-exec','src-set-pos','=','word-code-start','word-patch-u16','u16-add','u16-sub'}
WORD_RE = re.compile(r'^:\s+([^\s]+)\s+(.*?)\s*;\s*$', re.S)
TOKEN_RE = re.compile(r'\S+')
REQUIRED = {'space?','seek-marker','scan-token','find-word','probe','tok-colon','tok-semi','digit-first?','parse-number','set-marker-pos','mark-pos','emit-byte','create-pass','compile-pass','tok-if','tok-else','tok-then','current-end','current-len','patch-at','if-open','else-open','then-close','compile-structured','run-alpha'}


def parse_source(text):
    words = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = WORD_RE.match(line)
        if not m:
            raise SystemExit(f'malformed MIRR definition: {line[:120]!r}')
        name, body = m.groups()
        if name in words:
            raise SystemExit(f'duplicate MIRR word: {name}')
        words[name] = TOKEN_RE.findall(body)
    return words


def run(cmd, *, cwd=None):
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def main():
    if not SOURCE.is_file() or not NUCLEUS.is_file():
        raise SystemExit('missing Phase 30 source or Nucleus')

    src = parse_source(SOURCE.read_text())
    missing = sorted(REQUIRED - src.keys())
    if missing:
        raise SystemExit(f'MIRR compiler source missing required words: {missing}')

    unresolved = []
    for name, body in src.items():
        for tok in body:
            if tok in PRIMS or tok == 'exit':
                continue
            if tok.startswith(('branch:', '0branch:')):
                if not tok.split(':', 1)[1].isdigit():
                    unresolved.append((name, tok))
                continue
            if tok.isdigit() or re.fullmatch(r'@L\d+', tok):
                continue
            if tok not in src:
                unresolved.append((name, tok))
    if unresolved:
        raise SystemExit('unresolved MIRR compiler references: ' + repr(unresolved[:20]))

    with tempfile.TemporaryDirectory() as td_name:
        td = Path(td_name)
        exe = td / 'nucleus'
        cc = run(['cc','-std=c17','-Wall','-Wextra','-Wpedantic','-Werror',str(NUCLEUS),'-o',str(exe)])
        if cc.returncode:
            raise SystemExit('strict Nucleus build failed:\n' + cc.stderr)

        simple = td / 'simple.mirr'
        simple.write_text(': alpha 65 emit ;\n')
        structured = td / 'structured.mirr'
        structured.write_text(': alpha 1 IF 65 emit ELSE 66 emit THEN ;\n')

        p1 = run([str(exe), str(SOURCE), 'run-alpha', '--input', str(simple)])
        if p1.returncode or p1.stdout != 'A':
            raise SystemExit(f'direct MIRR compiler simple smoke failed: rc={p1.returncode} stdout={p1.stdout!r} stderr={p1.stderr!r}')

        p2 = run([str(exe), str(SOURCE), 'run-alpha', '--input', str(structured)])
        if p2.returncode or p2.stdout != 'A':
            raise SystemExit(f'direct MIRR compiler structured smoke failed: rc={p2.returncode} stdout={p2.stdout!r} stderr={p2.stderr!r}')

    print('PHASE30_COMPILER_ENTIRELY_IN_MIRR_PASS')
    print(f'MIRR compiler words: {len(src)}')
    print('Direct Nucleus execution path verified; host builder remains staging-only.')


if __name__ == '__main__':
    main()
