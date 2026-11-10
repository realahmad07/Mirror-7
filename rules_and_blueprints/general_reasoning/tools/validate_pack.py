#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from collections import Counter
REQUIRED={'schema_version','id','domain','subdomain','kind','intent','title','trigger','statement','preconditions','steps','postconditions','exceptions','dependencies','example','tests','provenance','authorship','evidence','quality'}
def norm(s:str)->str:
    return re.sub(r'[^\w]+',' ',s.casefold(),flags=re.UNICODE).strip()
def read(path:Path):
    rows=[]
    for n,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        try: obj=json.loads(line)
        except Exception as e: raise ValueError(f'{path}:{n}: invalid JSON: {e}')
        missing=REQUIRED-set(obj)
        if missing: raise ValueError(f'{path}:{n}: missing keys {sorted(missing)}')
        if obj['domain']!='general_reasoning': raise ValueError(f'{path}:{n}: wrong domain {obj["domain"]!r}')
        if obj['evidence']['status'] not in {'unverified','source_supported','curated','execution_tested','formally_verified','independently_reproduced'}:
            raise ValueError(f'{path}:{n}: invalid evidence status')
        if obj['evidence']['status']=='unverified' and any(t.get('status')!='not_run' for t in obj['tests']):
            raise ValueError(f'{path}:{n}: unverified record cannot claim executed tests')
        if not obj['id'] or not obj['title'] or not obj['statement']:
            raise ValueError(f'{path}:{n}: empty id/title/statement')
        if obj['provenance'].get('license_spdx') is None: raise ValueError(f'{path}:{n}: missing license field')
        rows.append(obj)
    return rows
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--existing-jsonl',type=Path,action='append',default=[])
    a=p.parse_args(); here=Path(__file__).resolve().parents[1]
    paths=[here/'rules'/'general_reasoning_candidates_10240.jsonl',here/'rules'/'general_reasoning_source_supported_26.jsonl']
    allrows=[]
    for path in paths:
        rows=read(path); print(f'{path.relative_to(here)}: {len(rows):,} rows')
        allrows.extend(rows)
    ids=[r['id'] for r in allrows]
    if len(ids)!=len(set(ids)): raise SystemExit('FAIL: duplicate IDs across pack')
    prompt_keys=[norm(r['trigger']) for r in allrows]
    if len(prompt_keys)!=len(set(prompt_keys)): raise SystemExit('FAIL: duplicate normalized triggers within pack')
    existing=0; existing_keys=set()
    for path in a.existing_jsonl:
        ex=read(path); existing+=len(ex)
        for r in ex:
            existing_keys.add(norm(r.get('trigger') or r.get('question') or r.get('prompt') or r.get('title') or ''))
    overlap=sum(1 for k in prompt_keys if k and k in existing_keys)
    print(f'TOTAL: {len(allrows):,}; unique IDs: {len(set(ids)):,}; exact normalized trigger duplicates: 0')
    print(f'Evidence: {dict(Counter(r["evidence"]["status"] for r in allrows))}')
    if a.existing_jsonl:
        print(f'Existing rows read: {existing:,}; exact normalized trigger overlaps: {overlap:,}')
        print('NOTE: exact overlap is not semantic deduplication; compare the live vault separately.')
    print('PASS: structural checks completed. This does not prove semantic uniqueness or correctness.')
if __name__=='__main__': main()
