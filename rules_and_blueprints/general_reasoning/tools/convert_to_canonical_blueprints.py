#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path

def convert(row):
    prompt=row['trigger'].strip()
    response=row['statement'].strip()+"\n\nProcedure:\n"+"\n".join(f'{i+1}. {s}' for i,s in enumerate(row['steps']))
    response += "\n\nLimitations:\n"+"\n".join(f'- {x}' for x in row['exceptions'])
    sha=hashlib.sha256((prompt+'\0'+response).encode('utf-8')).hexdigest()
    ev=row['evidence']['status']
    # Mirror's canonical EvidenceLevel does not define source_supported.
    # Keep it as UNVERIFIED in the canonical object and preserve the detailed state in sidecar metadata.
    canonical={
      'blueprint_id':row['id'], 'kind':row['kind'], 'domain':row['domain'], 'intent':row['intent'],
      'prompt_pattern':prompt, 'response_template':response, 'slots':[], 'derived_expr':'',
      'preconditions':row['preconditions'], 'postconditions':row['postconditions'], 'dependencies':[],
      'provenance':{'source':'mirror_general_reasoning_10k_candidate_v1','source_ref':f"{row['id']}|{row['provenance'].get('source_url','')}",'license_spdx':row['provenance'].get('license_spdx','NOASSERTION'),'content_sha256':sha,'captured_at':row['provenance'].get('retrieved_at','')},
      'verification':{'level':'unverified','checks_passed':[],'checks_failed':[],'verifier':'','verified_at':''},
      'legacy_trust_tier':3,'schema_version':1
    }
    sidecar={'blueprint_id':row['id'],'original_evidence_status':ev,'family_id':row.get('family_id',''),'context_id':row.get('context_id',''),'provenance':row['provenance'],'authorship':row['authorship'],'tests':row['tests'],'quality':row['quality'],'candidate_class':row.get('candidate_class','')}
    return canonical,sidecar

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sidecar',type=Path)
    a=p.parse_args(); side=[]; n=0
    with a.input.open(encoding='utf-8') as src, a.output.open('w',encoding='utf-8') as dst:
        for line in src:
            if not line.strip(): continue
            row=json.loads(line); c,s=convert(row)
            dst.write(json.dumps(c,ensure_ascii=False,separators=(',',':'))+'\n'); side.append(s); n+=1
    if a.sidecar:
        with a.sidecar.open('w',encoding='utf-8') as f:
            for s in side: f.write(json.dumps(s,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps({'converted':n,'output':str(a.output),'sidecar':str(a.sidecar) if a.sidecar else None,'warning':'Canonical Blueprint EvidenceLevel has no source_supported value; original evidence/provenance is preserved in sidecar and the canonical verification level remains unverified.'},indent=2))
if __name__=='__main__': main()
