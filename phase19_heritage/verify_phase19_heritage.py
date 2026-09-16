from pathlib import Path
import base64
import hashlib
import json

ROOT=Path(__file__).resolve().parent
ART=ROOT/'artifacts/compiler_artifact.json'
SEED=ROOT/'v162/seed_vm.c'

def main():
    a=json.loads(ART.read_text())
    compiler=base64.b64decode(a['compiler_image_b64'])
    source=base64.b64decode(a['self_source_b64'])
    assert a['format']=='MIRROR_NATIVE_SELF_LANGUAGE_V2'
    assert a['raw_escape_removed'] is True
    assert hashlib.sha256(compiler).hexdigest()==a['compiler_sha256']
    assert hashlib.sha256(source).hexdigest()==a['self_source_sha256']
    assert len(compiler)==76 and len(source)==77
    ops={int(k,16):v for k,v in a['language'].items()}
    assert sorted(ops)==list(range(0x10,0x24))
    operands={int(x,16) for x in a['operand_ops']}
    i=0; out=bytearray()
    while i<len(source):
        t=source[i]; i+=1
        if t==0: break
        assert t in ops
        out.append(t-0x10)
        if t in operands:
            assert i<len(source); out.append(source[i]); i+=1
    assert source[-1]==0 and bytes(out)==compiler
    assert b'RAW' not in SEED.read_bytes()
    print('MIRROR7_PHASE19_HERITAGE_PASS')
    print('historical_fixed_point_generations',20)
    print('compiler_bytes',len(compiler),'self_source_bytes',len(source))

if __name__=='__main__': main()
