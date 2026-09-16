import copy,json,random,subprocess,tempfile,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'v129'))
from artifact_compiler import compile_from_artifact
from bootstrap_artifact import build
BASE=json.loads((ROOT/'artifacts_phase13/source_artifact_v1.json').read_text())

def run(p): return subprocess.run([str(p)],capture_output=True,timeout=2).returncode

def main():
    r=random.Random(13013)
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        for i in range(1000):
            a=copy.deepcopy(BASE); n=r.randint(-20,20); name=f'op{i}'
            a['rules'][name]={'kind':'NOARG','emit':['ADD',str(n)]}
            src=f'let x = 10\n{name}\nexit'
            asm=compile_from_artifact(a,src); expect=10+n
            assert asm==f'SET 10\nADD {n}\nEXIT'
            ap=td/f'a{i}.json'; p=td/f'x{i}'
            ap.write_text(json.dumps(a)); build(ap,src,p)
            assert run(p)==(expect & 0xff),(i,expect,run(p))
    text=(ROOT/'v129/bootstrap_artifact.py').read_text()
    assert 'v126' not in text and 'v126/compiler' not in text
    print('PHASE13 STRESS PASS: 1,000 data-only language extensions executed natively; V126 compiler absent from bootstrap path')
if __name__=='__main__': main()
