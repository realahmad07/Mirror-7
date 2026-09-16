"""Bootstrap using only the data artifact for language semantics."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'v125'))
from assembler import assemble
from emit_elf import elf
from artifact_compiler import compile_from_artifact

def build(artifact_path, src_text, out_path):
    art=json.loads(Path(artifact_path).read_text())
    asm=compile_from_artifact(art,src_text)
    code=assemble(art['encoder'],asm.splitlines())
    Path(out_path).write_bytes(elf(code)); Path(out_path).chmod(0o755)
