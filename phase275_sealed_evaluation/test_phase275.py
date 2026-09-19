
import json
from .mirror7_phase275 import SealedEvaluator, write_agent_script

def tasks():
    return [
        {"sequence":[1,3,5,7],"target":9},
        {"sequence":[2,6,10,14],"target":18},
        {"sequence":[-3,0,3,6],"target":9},
    ]

def solver():
    return r'''
import json,sys
for line in sys.stdin:
    t=json.loads(line)
    s=t["sequence"]
    print(json.dumps(s[-1]+(s[-1]-s[-2])))
'''

def bad_solver():
    return "import sys\nfor _ in sys.stdin: print('0')\n"

def test_sealed_process_scores_correct_agent():
    p=write_agent_script(solver())
    good,total=SealedEvaluator(11).evaluate(p,tasks())
    assert (good,total)==(3,3)

def test_bad_agent_rejected():
    p=write_agent_script(bad_solver())
    good,total=SealedEvaluator(11).evaluate(p,tasks())
    assert good<total

def test_held_out_multi_seed_tasks():
    for seed in (3,7,19):
        ts=[{"sequence":[seed,seed+4,seed+8],"target":seed+12}]
        p=write_agent_script(solver())
        assert SealedEvaluator(seed).evaluate(p,ts)==(1,1)

def test_hidden_target_not_in_public_payload():
    # The public stream contains only the sequence; target exists only in the parent evaluator.
    p=write_agent_script("import sys,json\nfor line in sys.stdin:\n t=json.loads(line); print(json.dumps(0))\n")
    good,total=SealedEvaluator().evaluate(p,[{"sequence":[1,2],"target":99}])
    assert total==1 and good==0

def test_process_failure_fails_closed():
    p=write_agent_script("raise SystemExit(2)")
    assert SealedEvaluator().evaluate(p,tasks())==(0,3)

def test_deterministic_score():
    p=write_agent_script(solver())
    assert SealedEvaluator(1).evaluate(p,tasks())==SealedEvaluator(99).evaluate(p,tasks())
