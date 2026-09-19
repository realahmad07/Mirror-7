import pathlib, subprocess, random, os
ROOT=pathlib.Path(__file__).resolve().parents[1]
NU=ROOT/'nucleus'; WORDS=ROOT/'compiler_words.mirr'

def ensure_nucleus():
    if NU.exists(): return True
    cc=os.environ.get('CC','cc')
    c=ROOT/'phase24'/'nucleus.c'
    if not c.exists(): return False
    r=subprocess.run([cc,str(c),'-O2','-o',str(NU)],capture_output=True,timeout=20)
    return r.returncode==0 and NU.exists()

def run(src, word='main', timeout=3, exe=NU):
    if not ensure_nucleus():
        raise RuntimeError('no checked-in nucleus executable and no C compiler available')
    p=ROOT/'tmp.mirr'; p.write_text(src)
    try: return subprocess.run([str(exe), str(p), word], capture_output=True, timeout=timeout, env=os.environ.copy())
    finally: p.unlink(missing_ok=True)

s='''\
: emit-char emit ;
: build 102 0 ! 111 1 ! 111 2 ! 0 0 3 0 word-new 1 word-append 65 word-append 11 word-append 10 word-append word-exec ;
: main build ;
'''
r=run(s); assert r.returncode==0 and r.stdout==b'A',(r.returncode,r.stdout,r.stderr)

s='''\
: build 102 0 ! 111 1 ! 111 2 ! 0 0 3 0 word-new 1 word-append 65 word-append 10 word-append ;
: main build 2 0 0 word-name-char emit ;
'''
r=run(s); assert r.returncode==0 and r.stdout==b'f',(r.returncode,r.stdout,r.stderr)

s='''\
: build 102 0 ! 111 1 ! 111 2 ! 0 0 3 0 word-new 1 word-append 65 word-append 10 word-append 2 0 1 word-code-byte ;
: main build emit ;
'''
r=run(s); assert r.returncode==0 and r.stdout==b'A',(r.returncode,r.stdout,r.stderr)

lines=[': main']
for i in range(300):
    name=f'w{i}'
    parts=[]
    for j,ch in enumerate(name.encode()): parts += [str(ch),str(j),'!']
    parts += ['0','0',str(len(name)),'0','word-new','1','word-append',str(i%256),'word-append','10','word-append']
    lines += parts
lines += [';']
r=run('\n'.join(lines)+'\n'); assert r.returncode==0,(r.returncode,r.stderr[:500])

s='''\
: build 102 0 ! 111 1 ! 111 2 ! 0 0 3 0 word-new drop ;
: main build build ;
'''
r=run(s); assert r.returncode!=0

chars=' '.join(f'{65+(i%26)} {i} !' for i in range(64))
s=f': main {chars} 0 0 64 0 word-new ;\n'
r=run(s); assert r.returncode!=0

r=run(': a a ;\n', 'a'); assert r.returncode!=0
print('PHASE24 CORE PASS')
