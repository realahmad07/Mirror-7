import json, sys

def run(doc):
    a=doc['artifact']; m=a['machine']; tape=list(map(int,doc['tape']))
    table={(int(x['control']),int(x['symbol'])):x for x in m['transitions']}
    c=int(m['start_control']); h=int(m['halt_control']); p=0
    for _ in range(int(a.get('payload',{}).get('max_steps',4096))):
        if c==h: return tape
        if not (0<=p<len(tape)): raise RuntimeError('head out of range')
        tr=table.get((c,tape[p]))
        if tr is None: raise RuntimeError('missing transition')
        tape[p]=int(tr['write_symbol']); p+=int(tr['move']); c=int(tr['next_control'])
    raise RuntimeError('step budget exceeded')

if __name__=='__main__': print(json.dumps(run(json.load(sys.stdin))))
