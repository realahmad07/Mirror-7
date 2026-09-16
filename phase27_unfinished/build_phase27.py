from pathlib import Path
import re

PRIMS={'+','-','dup','drop','swap','@','!','emit','next','nextc','src-pos','src-len','word-count','word-name-len','word-name-char','word-code-len','word-code-byte','word-new','word-append','word-exec','src-set-pos','=','word-code-start','word-patch-u16','u16-add'}
BASE_PRIM_BYTES=48
PRIM_BYTES=2*len(PRIMS)
PRIM_DELTA=PRIM_BYTES-BASE_PRIM_BYTES
base_lines=Path('compiler_phase26_words.mirr').read_text().splitlines()

def token_size(t):
    if t=='exit': return 1
    if t.startswith('0branch:') or t.startswith('branch:'): return 3
    if t in PRIMS: return 1
    try:
        n=int(t)
        if 0<=n<=255:return 2
    except ValueError: pass
    return 3 if t not in (':',';') else 0

def relocate(lines,delta):
    all_toks=[line.split() for line in lines]
    old_addr=PRIM_BYTES-delta
    new_addr=PRIM_BYTES
    addr_map={}
    for toks in all_toks:
        if not toks: continue
        body=toks[2:-1] if toks[0]==':' else toks
        for t in body:
            addr_map[old_addr]=new_addr
            sz=token_size(t)
            old_addr+=sz
            new_addr+=sz
        if toks[0]==':':
            old_addr+=1
            new_addr+=1
    def remap(old_target):
        if old_target in addr_map:
            return addr_map[old_target]
        prior=max((a for a in addr_map if a<=old_target),default=None)
        if prior is None: return old_target+delta
        return addr_map[prior]+(old_target-prior)
    out=[]
    for line in lines:
        toks=[]
        for t in line.split():
            m=re.match(r'^(0branch:|branch:)(\d+)$',t)
            if m: t=m.group(1)+str(remap(int(m.group(2))))
            toks.append(t)
        out.append(' '.join(toks))
    return out

def repair_create_pass_eof_guard(lines):
    out=[]
    for line in lines:
        toks=line.split()
        if len(toks)>=7 and toks[0]==':' and toks[1]=='create-pass':
            for i in range(len(toks)-5):
                if toks[i:i+4] == ['12','@','0','=']:
                    m=re.fullmatch(r'0branch:(\d+)',toks[i+4])
                    if m and toks[i+5]=='exit':
                        toks[i+4]=f'0branch:{int(m.group(1))+1}'
                        break
        out.append(' '.join(toks))
    return out

def repair_find_word_targets(lines):
    """Resolve legacy FIND targets from the emitted instruction layout."""
    out=[]
    addr=PRIM_BYTES
    for line in lines:
        toks=line.split()
        if toks and toks[0]==':' and toks[1]=='find-word':
            body=toks[2:-1]
            pcs=[]; pc=addr
            for t in body:
                pcs.append(pc); pc += token_size(t)
            limit_start=None
            for i in range(len(body)-3):
                if body[i:i+4] == ['13','@','255','=']:
                    limit_start=pcs[i]
                    break
            for i,t in enumerate(body):
                if not t.startswith('0branch:'):
                    continue
                # Legacy 445 lands two bytes into this LIT 13 after relocation.
                if limit_start is not None and int(t.split(':',1)[1]) == limit_start + 2:
                    body[i]=f'0branch:{limit_start}'
                # Legacy 468 lands one byte before the low-byte increment path.
                if i+4 < len(body) and body[i+1:i+4] == ['0','13','!'] and body[i+4] == '1':
                    body[i]=f'0branch:{pcs[i+4]}'
            toks=toks[:2]+body+toks[-1:]
            line=' '.join(toks)
        out.append(line)
        if toks:
            addr += sum(token_size(t) for t in toks[2:-1])+1
    return out

base_lines=relocate(base_lines,PRIM_DELTA)
base_lines=repair_create_pass_eof_guard(base_lines)
base_lines=repair_find_word_targets(base_lines)

def source_size(lines):
    total=PRIM_BYTES
    for line in lines:
        toks=line.split()
        if toks: total += sum(token_size(t) for t in toks[2:-1])+1
    return total

class W:
    def __init__(self,name): self.name=name; self.ins=[]
    def t(self,*xs): self.ins += xs
    def label(self,n): self.ins.append(('LABEL',n))
    def br(self,l): self.ins.append(('BR',l,False))
    def bz(self,l): self.ins.append(('BR',l,True))
words=[]
def Wd(n): w=W(n); words.append(w); return w
w=Wd('tok-if'); w.t('12','@','2','=','64','@','105','=','+','65','@','102','=','+','3','=','exit')
w=Wd('tok-else'); w.t('12','@','4','=','64','@','101','=','+','65','@','108','=','+','66','@','115','=','+','67','@','101','=','+','5','=','exit')
w=Wd('tok-then'); w.t('12','@','4','=','64','@','116','=','+','65','@','104','=','+','66','@','101','=','+','67','@','110','=','+','5','=','exit')
w=Wd('current-end'); w.t('18','@','19','@','word-code-start','swap','40','!','41','!','18','@','19','@','word-code-len','swap','42','!','43','!','40','@','41','@','42','@','43','@','u16-add','swap','40','!','41','!','exit')
w=Wd('current-len'); w.t('18','@','19','@','word-code-len','swap','42','!','43','!','exit')
w=Wd('patch-at'); w.t('18','@','19','@','50','@','51','@','40','@','41','@','word-patch-u16','drop','drop','exit')
w=Wd('if-open'); w.t('30','@','15','=','0','='); w.bz('BAD'); w.t('30','@','30','@','30','@','30','@','+','+','+','192','+','48','!','48','@','1','+','49','!','current-len','42','@','43','@','1','0','u16-add','swap','50','!','51','!','14','10','!','emit-byte','0','10','!','emit-byte','0','10','!','emit-byte','50','@','48','@','!','51','@','49','@','!','0','48','@','2','+','!','0','49','@','2','+','!','1','30','@','+','30','!','exit'); w.label('BAD'); w.t('0','31','!','exit')
w=Wd('else-open'); w.t('30','@','0','=','0','='); w.bz('BAD'); w.t('30','@','1','-','30','@','1','-','30','@','1','-','30','@','1','-','+','+','+','192','+','48','!','48','@','1','+','49','!','48','@','2','+','@','52','!','49','@','2','+','@','53','!','52','@','53','@','+','0','='); w.bz('BAD'); w.t('48','@','@','50','!','49','@','@','51','!','current-len','42','@','43','@','1','0','u16-add','swap','54','!','55','!','15','10','!','emit-byte','0','10','!','emit-byte','0','10','!','emit-byte','current-end','patch-at','54','@','48','@','2','+','!','55','@','49','@','2','+','!','exit'); w.label('BAD'); w.t('0','31','!','exit')
w=Wd('then-close'); w.t('30','@','0','='); w.bz('CONT'); w.t('0','31','!','exit'); w.label('CONT'); w.t('30','@','1','-','30','@','1','-','30','@','1','-','30','@','1','-','+','+','+','192','+','48','!','48','@','1','+','49','!','48','@','2','+','@','52','!','49','@','2','+','@','53','!','52','@','53','@','+','0','='); w.bz('HAS_ELSE'); w.t('48','@','@','50','!','49','@','@','51','@','current-end','patch-at','30','@','1','-','30','!','exit'); w.label('HAS_ELSE'); w.t('52','@','50','!','53','@','51','!','current-end','patch-at','30','@','1','-','30','!','exit')
w=Wd('compile-structured'); w.t('1','31','!','create-pass','set-marker-pos'); w.label('L'); w.t('scan-token','12','!','12','@','0','='); w.bz('HAVE'); w.t('exit'); w.label('HAVE'); w.t('tok-colon'); w.bz('FAIL'); w.t('scan-token','12','!','12','@','0','='); w.bz('GN'); w.bz('FAIL'); w.label('GN'); w.t('find-word','19','!','18','!'); w.label('B'); w.t('scan-token','12','!','12','@','0','='); w.bz('BH'); w.bz('FAIL'); w.label('BH'); w.t('tok-semi'); w.bz('NOTS'); w.t('30','@','0','='); w.bz('FAIL'); w.t('10','10','!','emit-byte'); w.br('L'); w.label('NOTS'); w.t('tok-if'); w.bz('NOIF'); w.t('if-open','31','@','0','=','0','='); w.bz('FAIL'); w.br('B'); w.label('NOIF'); w.t('tok-else'); w.bz('NOELSE'); w.t('else-open','31','@','0','=','0','='); w.bz('FAIL'); w.br('B'); w.label('NOELSE'); w.t('tok-then'); w.bz('NOTHEN'); w.t('then-close','31','@','0','=','0','='); w.bz('FAIL'); w.br('B'); w.label('NOTHEN'); w.t('digit-first?'); w.bz('WORD'); w.t('parse-number','20','!','1','10','!','emit-byte','20','@','10','!','emit-byte'); w.br('B'); w.label('WORD'); w.t('find-word','21','!','20','!','9','10','!','emit-byte','20','@','10','!','emit-byte','21','@','10','!','emit-byte'); w.br('B'); w.label('FAIL'); w.t('exit')
w=Wd('run-alpha'); w.t('compile-structured','word-count','19','!','18','!','18','@','1','-','18','!','18','@','19','@','word-exec','exit')
cur=source_size(base_lines)
for w in words:
    w.start=cur; local=0; labs={}
    for ins in w.ins:
        if isinstance(ins,tuple):
            if ins[0]=='LABEL': labs[ins[1]]=local
            else: local+=3
        else: local+=token_size(ins)
    w.labels={k:w.start+v for k,v in labs.items()}; w.size=local+1; cur+=w.size
lines=list(base_lines)
for w in words:
    out=[f': {w.name}']
    for ins in w.ins:
        if isinstance(ins,tuple):
            if ins[0]=='LABEL': continue
            out.append(('0branch:' if ins[2] else 'branch:')+str(w.labels[ins[1]]))
        else: out.append(ins)
    out.append(';'); lines.append(' '.join(out))
Path('compiler_phase27_words.mirr').write_text('\n'.join(lines)+'\n')
print('base',source_size(base_lines),'total',cur)
for w in words: print(w.name,w.start,w.size,w.labels)