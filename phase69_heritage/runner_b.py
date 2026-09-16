import json, sys

def run(doc):
    a=doc['artifact']; m=a['machine']; trans=m['transitions']; index={}
    for item in trans: index[f"{int(item['control'])}:{int(item['symbol'])}"]=item
    state=int(m['start_control']); halt=int(m['halt_control']); head=0; tape=list(map(int,doc['tape']))
    limit=int(a.get('payload',{}).get('max_steps',4096)); steps=0
    while state!=halt:
        if steps>=limit: raise RuntimeError('step budget exceeded')
        if not (0<=head<len(tape)): raise RuntimeError('head out of range')
        key=f"{state}:{tape[head]}"
        if key not in index: raise RuntimeError('missing transition')
        tr=index[key]; tape[head]=int(tr['write_symbol']); head+=int(tr['move']); state=int(tr['next_control']); steps+=1
    return tape

if __name__=='__main__': print(json.dumps(run(json.load(sys.stdin))))
