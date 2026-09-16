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

# These are the only host-provided operations that the MIRR compiler may call.
PRIMS = {
    '+', '-', 'dup', 'drop', 'swap', '@', '!', 'emit', 'next', 'nextc',
    'src-pos', 'src-len', 'word-count', 'word-name-len', 'word-name-char',
    'word-code-len', 'word-code-byte', 'word-new', 'word-append', 'word-exec',
    'src-set-pos', 'word-code-start', 'word-patch-u16', 'u16-add', '='
}

TOKEN_RE = re.compile(r'\S+')
WORD_RE = re.compile(r'^:\s+([^\s]+)\s+(.*?)\s*;\s*$', re.S)


def parse_mirr(text):
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = WORD_RE.match(line)
        if not m:
            raise AssertionError(f"not a complete MIRR definition: {line[:100]!r}")
        name, body = m.groups()
        if name in out:
            raise AssertionError(f"duplicate MIRR word: {name}")
        out[name] = TOKEN_RE.findall(body)
    return out


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)


def main():
    for p in (SOURCE, WORDS, BUILDER, NUCLEUS, INTEGRATION):
        if not p.is_file():
            raise SystemExit(f"missing Phase 29 input: {p}")

    src = parse_mirr(SOURCE.read_text())
    if not src:
        raise SystemExit("compiler source contains no MIRR definitions")

    # Every non-literal/non-control token used by the compiler implementation
    # must either be a primitive or another word in the same MIRR source.
    unresolved = []
    for name, body in src.items():
        for tok in body:
            if tok in PRIMS or tok in {'exit'}:
                continue
            if tok.startswith(('branch:', '0branch:')):
                if not tok.split(':', 1)[1].isdigit():
                    unresolved.append((name, tok))
                continue
            if tok.isdigit() or re.fullmatch(r'@L\d+', tok):
                continue
            if tok.startswith(':') or tok == ';':
                continue
            if tok not in src:
                unresolved.append((name, tok))
    if unresolved:
        raise SystemExit("unresolved compiler-source references: " + repr(unresolved[:20]))

    required = {
        'space?', 'seek-marker', 'scan-token', 'find-word', 'probe',
        'tok-colon', 'tok-semi', 'digit-first?', 'parse-number',
        'set-marker-pos', 'mark-pos', 'emit-byte', 'create-pass',
        'compile-pass', 'tok-if', 'tok-else', 'tok-then', 'current-end',
        'current-len', 'patch-at', 'if-open', 'else-open', 'then-close',
        'compile-structured', 'run-alpha'
    }
    missing = sorted(required - src.keys())
    if missing:
        raise SystemExit(f"compiler source is missing required MIRR words: {missing}")

    # The generated compiler must still be exactly reproducible from the
    # preserved Phase-26 source plus the Phase-27 builder.
    with tempfile.TemporaryDirectory() as td_name:
        td = Path(td_name)
        shutil.copy2(ROOT / 'phase25_26' / 'compiler_phase26_words.mirr', td / 'compiler_phase26_words.mirr')
        b = run(['python3', str(BUILDER)], cwd=td)
        if b.returncode:
            raise SystemExit('Phase 27 builder failed:\n' + b.stderr + b.stdout)
        generated = td / WORDS.name
        if generated.read_bytes() != WORDS.read_bytes():
            raise SystemExit('generated compiler artifact is not byte-identical to the committed artifact')

        cc = run(['cc', '-std=c17', '-Wall', '-Wextra', '-Wpedantic', '-Werror', str(NUCLEUS), '-o', str(td / 'nucleus')])
        if cc.returncode:
            raise SystemExit('strict nucleus build failed:\n' + cc.stderr)

        # Smoke-test the real compiler path after source-closure validation.
        inp = td / 'smoke.mirr'
        inp.write_text(': alpha 1 IF 65 emit ELSE 66 emit THEN ;\n')
        p = run([str(td / 'nucleus'), str(generated), 'run-alpha', '--input', str(inp)])
        if p.returncode or p.stdout != 'A':
            raise SystemExit(f"compiler smoke test failed: rc={p.returncode} stdout={p.stdout!r} stderr={p.stderr!r}")

    print('PHASE29_MIRR_SOURCE_CLOSURE_PASS')
    print(f'MIRR compiler words: {len(src)}')
    print('Host dependency remains limited to the bootstrap loader/builder; self-recompile is not claimed.')


if __name__ == '__main__':
    main()
