from .mirror7_phase133 import ProgramSandbox
def test_safe_program_runs(): assert ProgramSandbox().run(lambda x:x+1,1).value==2
def test_exception_fails_closed(): assert not ProgramSandbox().run(lambda x:1/0,1).ok
def test_bad_budget_rejected(): assert not ProgramSandbox().run(lambda x:x,1,0).ok
