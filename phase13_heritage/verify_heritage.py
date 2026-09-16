import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def main():
    artifact = ROOT / 'artifacts_phase13' / 'source_artifact_v1.json'
    required = [ROOT/'PHASE13_FINAL.md', ROOT/'PHASE13_REPORT.md', ROOT/'V122_REPORT.md', ROOT/'V123_REPORT.md', artifact, ROOT/'v129/artifact_compiler.py', ROOT/'v129/bootstrap_artifact.py', ROOT/'tests_phase13/test_phase13.py', ROOT/'tests_phase13/stress_phase13.py']
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing: raise SystemExit('missing heritage inputs: ' + ', '.join(missing))
    data = json.loads(artifact.read_text())
    if data.get('format') != 'MIRROR_SOURCE_ARTIFACT_V1': raise SystemExit('unexpected source artifact format')
    if not isinstance(data.get('encoder', {}).get('templates'), dict): raise SystemExit('missing encoder templates')
    if not isinstance(data.get('rules'), dict) or not data['rules']: raise SystemExit('missing language rules')
    for name, rule in data['rules'].items():
        if not isinstance(name, str) or not name: raise SystemExit('invalid rule name')
        if rule.get('kind') not in {'NOARG','IMM','LET_IMM'}: raise SystemExit(f'invalid rule kind: {name}')
        if not isinstance(rule.get('emit'), list) or not rule['emit']: raise SystemExit(f'invalid rule emission: {name}')
    print('MIRROR7_PHASE13_HERITAGE_PASS')
    print('preserved: artifact-driven semantics, learned/native encoder evidence, bootstrap-boundary tests')
    print('boundary: historical host-language engine remains reference-only')
if __name__ == '__main__': main()
