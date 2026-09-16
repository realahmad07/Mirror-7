from pathlib import Path
PRIMS={'+','-','dup','drop','swap','@','!','emit','next','nextc','src-pos','src-len','word-count','word-name-len','word-name-char','word-code-len','word-code-byte','word-new','word-append','word-exec','src-set-pos','=','word-code-start','word-patch-u16','u16-add'}
base_lines=Path('compiler_phase26_words.mirr').read_text().splitlines()

def token_size(t):
    if t in ('exit',): return 1
    if t.startswith('0branch:') or t.startswith('branch:'): return 3
    if t in PRIMS: return 1
    try:
        n=int(t)
        if 0<=n<=255:return 2
    except: pass
    if t.startswith('@L'): return 3
    if t not in (':',';'): return 3
    return 0

def source_size(lines):
    total=PRIM_BYTES
    for line in lines:
        toks=line.split()
        if not toks: continue
        assert toks[0]==':' and toks[-1]==';'
        total += sum(token_size(t) for t in toks[2:-1]) + 1
    return total

PRIM_BYTES=2*len(PRIMS)
BASE_PRIM_BYTES=48
PRIM_DELTA=PRIM_BYTES-BASE_PRIM_BYTES
if PRIM_DELTA:
    import re
    fixed=[]
    for _ln in base_lines:
        _toks=_ln.split(); _out=[]
        for _t in _toks:
            _m=re.match(r'^(0branch:|branch:)(\d+)$',_t)
            if _m: _t=_m.group(1)+str(int(_m.group(2))+PRIM_DELTA)
            _out.append(_t)
        fixed.append(' '.join(_out))
    base_lines=fixed
start=source_size(base_lines)

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
w=Wd('if-open'); w.ins=[]
w.t('30','@','15','=','0','='); w.bz('BAD'); w.t('30','@','30','@','30','@','30','@','+','+','+','192','+','48','!','48','@','1','+','49','!'); w.t('current-len','42','@','43','@','1','0','u16-add','swap','50','!','51','!'); w.t('14','10','!','emit-byte','0','10','!','emit-byte','0','10','!','emit-byte'); w.t('50','@','48','@','!','51','@','49','@','!'); w.t('0','48','@','2','+','!','0','49','@','2','+','!'); w.t('1','30','@','+','30','!','exit'); w.label('BAD'); w.t('0','31','!','exit')
w=Wd('else-open'); w.ins=[]
w.t('30','@','0','=','0','='); w.bz('BAD'); w.t('30','@','1','-','30','@','1','-','30','@','1','-','30','@','1','-','+','+','+','192','+','48','!','48','@','1','+','49','!'); w.t('48','@','2','+','@','52','!','49','@','2','+','@','53','!','52','@','53','@','+','0','='); w.bz('BAD'); w.t('48','@','@','50','!','49','@','@','51','!'); w.t('current-len','42','@','43','@','1','0','u16-add','swap','54','!','55','!'); w.t('15','10','!','emit-byte','0','10','!','emit-byte','0','10','!','emit-byte'); w.t('current-end','patch-at'); w.t('54','@','48','@','2','+','!','55','@','49','@','2','+','!','exit'); w.label('BAD'); w.t('0','31','!','exit')
w=Wd('then-close'); w.ins=[]
w.t('30','@','0','='); w.bz('CONT'); w.t('0','31','!','exit'); w.label('CONT'); w.t('30','@','1','-','30','@','1','-','30','@','1','-','30','@','1','-','+','+','+','192','+','48','!','48','@','1','+','49','!'); w.t('48','@','2','+','@','52','!','49','@','2','+','@','53','!','52','@','53','@','+','0','='); w.bz('HAS_ELSE'); w.t('48','@','@','50','!','49','@','@','51','!','current-end','patch-at','30','@','1','-','30','!','exit'); w.label('HAS_ELSE'); w.t('52','@','50','!','53','@','51','!','current-end','patch-at','30','@','1','-','30','!','exit')
w=Wd('compile-structured'); w.t('1','31','!','create-pass','set-marker-pos'); w.label('L'); w.t('scan-token','12','!','12','@','0','='); w.bz('HAVE'); w.t('exit'); w.label('HAVE'); w.t('tok-colon'); w.bz('FAIL'); w.t('scan-token','12','!','12','@','0','='); w.bz('GN'); w.bz('FAIL'); w.label('GN'); w.t('find-word','19','!','18','!'); w.label('B'); w.t('scan-token','12','!','12','@','0','='); w.bz('BH'); w.bz('FAIL'); w.label('BH'); w.t('tok-semi'); w.bz('NOTS'); w.t('30','@','0','='); w.bz('FAIL'); w.t('10','10','!','emit-byte'); w.br('L'); w.label('NOTS'); w.t('tok-if'); w.bz('NOIF'); w.t('if-open','31','@','0','=','0','='); w.bz('FAIL'); w.br('B'); w.label('NOIF'); w.t('tok-else'); w.bz('NOELSE'); w.t('else-open','31','@','0','=','0','='); w.bz('FAIL'); w.br('B'); w.label('NOELSE'); w.t('tok-then'); w.bz('NOTHEN'); w.t('then-close','31','@','0','=','0','='); w.bz('FAIL'); w.br('B'); w.label('NOTHEN'); w.t('digit-first?'); w.bz('WORD'); w.t('parse-number','20','!','1','10','!','emit-byte','20','@','10','!','emit-byte'); w.br('B'); w.label('WORD'); w.t('find-word','21','!','20','!','9','10','!','emit-byte','20','@','10','!','emit-byte','21','@','10','!','emit-byte'); w.br('B'); w.label('FAIL'); w.t('exit')
w=Wd('run-alpha'); w.t('compile-structured','word-count','19','!','18','!','18','@','1','-','18','!','18','@','19','@','word-exec','exit')
cur=start
for w in words:
    w.start=cur; local=0; labs={}
    for ins in w.ins:
        if isinstance(ins,tuple):
            if ins[0]=='LABEL': labs[ins[1]]=local
            elif ins[0]=='BR': local+=3
        else: local+=token_size(ins)
    w.labels={k:w.start+v for k,v in labs.items()}; w.size=local+1; cur+=w.size
lines=list(base_lines)
for w in words:
    out=[f': {w.name}']
    for ins in w.ins:
        if isinstance(ins,tuple):
            if ins[0]=='LABEL': continue
            if ins[0]=='BR': out.append(('0branch:' if ins[2] else 'branch:')+str(w.labels[ins[1]]))
        else: out.append(ins)
    out.append(';'); lines.append(' '.join(out))
Path('compiler_phase27_words.mirr').write_text('\n'.join(lines)+'\n')
print('base',start,'total',cur)
for w in words: print(w.name,w.start,w.size,w.labels)