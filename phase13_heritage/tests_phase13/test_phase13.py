import json, subprocess, tempfile, copy, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'v129'))
from artifact_compiler import compile_from_artifact, validate_artifact
from bootstrap_artifact import build
ART_PATH=ROOT/'artifacts_phase13/source_artifact_v1.json'
if not ART_PATH.exists():
    print('PHASE13 SKIP: legacy V125 payload is not bundled; no false PASS asserted')
    raise SystemExit(0)
ART=json.loads(ART_PATH.read_text())

def run(exe):
    return subprocess.run([str(exe)],capture_output=True,timeout=2).returncode

def test_baseline():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'x'; build(ART_PATH,'let x = 41\nadd 1\nsub 2\nexit',p)
        assert run(p)==40

def test_data_only_extension():
    a=copy.deepcopy(ART); a['rules']['inc']={'kind':'NOARG','emit':['ADD','1']}
    assert compile_from_artifact(a,'let x = 9\ninc\nexit') == 'SET 9\nADD 1\nEXIT'
    with tempfile.TemporaryDirectory() as td:
        ap=Path(td)/'a.json'; ap.write_text(json.dumps(a)); p=Path(td)/'x'
        build(ap,'let x = 9\ninc\nexit',p); assert run(p)==10

def test_no_opcode_names_in_engine():
    text=(ROOT/'v129/artifact_compiler.py').read_text().lower()
    for bad in ('set','add','sub','exit','cmp0','jmp','jz','ret'):
        assert f"'{bad}'" not in text and f'"{bad}"' not in text

def test_missing_machine_semantics_rejects():
    a=copy.deepcopy(ART); a['rules']['inc']={'kind':'NOARG','emit':['NO_SUCH_OP']}
    with tempfile.TemporaryDirectory() as td:
        ap=Path(td)/'a.json'; ap.write_text(json.dumps(a)); p=Path(td)/'x'
        try: build(ap,'inc',p)
        except ValueError: pass
        else: raise AssertionError('unknown machine operation was accepted')

def test_malformed_artifacts():
    cases=[None,{}, {'format':'bad'}, {'format':'MIRROR_SOURCE_ARTIFACT_V1','rules':{}}, {'format':'MIRROR_SOURCE_ARTIFACT_V1','rules':{'x':{}}}]
    for c in cases:
        try: validate_artifact(c)
        except (TypeError,ValueError): continue
        raise AssertionError('malformed artifact accepted')

def main():
    test_baseline(); test_data_only_extension(); test_no_opcode_names_in_engine(); test_missing_machine_semantics_rejects(); test_malformed_artifacts()
    print('PHASE13 PASS: artifact-driven compiler semantics, data-only extension, no opcode literals in engine, malformed artifact rejection')
if __name__=='__main__': main()
