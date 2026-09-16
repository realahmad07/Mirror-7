import json, subprocess, sys, tempfile, pathlib, hashlib
ROOT=pathlib.Path(__file__).parent
sys.path.insert(0,str(ROOT))
from bootstrap import make_bootstrap_increment, execute_artifact
from runner_a import run as run_a
from runner_b import run as run_b

def main():
    a=make_bootstrap_increment(); doc={'artifact':a.to_dict()}
    assert a.fingerprint()==hashlib.sha256(a.blob()).hexdigest()
    cases=[([0,10],[1,10]),([9,10],[0,1]),([9,9,10],[0,0,1]),([9,9,9,10],[0,0,0,1]),([8,10],[9,10])]
    for tape,expected in cases:
        d={**doc,'tape':tape}
        assert execute_artifact(a,list(tape))==expected
        assert run_a(d)==expected
        assert run_b(d)==expected
    bad={**doc,'tape':[42]}
    for fn in (lambda:execute_artifact(a,[42]), lambda:run_a(bad), lambda:run_b(bad)):
        try: fn(); raise AssertionError('missing transition accepted')
        except RuntimeError: pass
    with tempfile.TemporaryDirectory() as td:
        exe=pathlib.Path(td)/'runner_c'; src=ROOT/'runner_c.c'
        subprocess.run(['cc','-std=c17','-Wall','-Wextra','-Werror',str(src),'-o',str(exe)],check=True)
        for tape,expected in cases[:4]:
            out=subprocess.check_output([str(exe)],input=json.dumps({**doc,'tape':tape}).encode()).decode().strip()
            assert json.loads(out)==expected
    print('PHASE69_HERITAGE_PASS')

if __name__=='__main__': main()
