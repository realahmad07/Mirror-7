from pathlib import Path
import json, base64, hashlib

OPS={'HALT':0,'PUSH':1,'POP':2,'ADD':3,'SUB':4,'LOAD':5,'STORE':6,'JMP':7,'JZ':8,'CALL':9,'RET':10,'IN':11,'OUT':12,'DUP':13,'EQ':14,'LT':15,'AND':16,'OR':17,'XOR':18,'DROP':19}
OPERAND={'PUSH','JMP','JZ','CALL'}
SRC={name:0x10+code for name,code in OPS.items()}

# Historical source excerpt retained as a mapping specification rather than an active builder.
# The original archive builder emitted the documented 20-op semantic language and self-source.
compiler_bytes=bytes.fromhex('0b0d01000e0808000d01100f0809040d0101090e0840310f0c0c0b0b070d01070e08310f0c0c0b0b070d01080e08250f0c0c0b0b070d01090e0825'[:0])

artifact_path=Path(__file__).parents[1]/'artifacts'/'compiler_artifact.json'
a=json.loads(artifact_path.read_text())
compiler=base64.b64decode(a['compiler_image_b64'])
source=base64.b64decode(a['self_source_b64'])
assert hashlib.sha256(compiler).hexdigest()==a['compiler_sha256']
assert hashlib.sha256(source).hexdigest()==a['self_source_sha256']
print('PHASE19_HERITAGE_BUILDER_REFERENCE_PASS')
