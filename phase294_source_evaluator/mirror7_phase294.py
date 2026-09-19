from dataclasses import dataclass
import json, os, subprocess, sys, tempfile, textwrap
from phase292_patch_validation import PatchValidator

@dataclass(frozen=True)
class SourceEvaluation:
    train: float
    held_out: float
    regression_ok: bool
    total_cases: int

class SealedSourceEvaluator:
    """Evaluates a validated solve(values) candidate in a separate isolated Python process."""
    HARNESS=textwrap.dedent("""
        import json, runpy, sys
        env=runpy.run_path(sys.argv[1], run_name="__candidate__")
        solve=env.get("solve")
        if not callable(solve):
            raise SystemExit(3)
        for line in sys.stdin:
            payload=json.loads(line)
            answer=solve(payload["values"])
            print(json.dumps(answer,separators=(",",":")))
    """).strip()

    def _score(self,source,tasks):
        cases=list(tasks)
        validation=PatchValidator().validate_source(source)
        if not validation.accepted:
            return 0,len(cases)
        if not cases:
            return 0,0
        with tempfile.TemporaryDirectory() as td:
            candidate=os.path.join(td,"candidate.py")
            with open(candidate,"w",encoding="utf-8") as f:
                f.write(source)
            payload="".join(json.dumps({"values":t["public"]["values"]})+"\n" for t in cases)
            try:
                proc=subprocess.run([sys.executable,"-I","-c",self.HARNESS,candidate],input=payload,text=True,capture_output=True,timeout=5,check=False)
            except (subprocess.TimeoutExpired,OSError):
                return 0,len(cases)
            if proc.returncode!=0:
                return 0,len(cases)
            outputs=[x.strip() for x in proc.stdout.splitlines() if x.strip()]
            if len(outputs)!=len(cases):
                return 0,len(cases)
            good=0
            for raw,task in zip(outputs,cases):
                try:
                    answer=json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if answer==task["target"]:
                    good+=1
            return good,len(cases)

    def evaluate(self,source,pack)->SourceEvaluation:
        tg,tt=self._score(source,pack.train)
        hg,ht=self._score(source,pack.held_out)
        rg,rt=self._score(source,pack.regression)
        return SourceEvaluation(tg/tt if tt else 0.0,hg/ht if ht else 0.0,rt>0 and rg==rt,tt+ht+rt)

    def improves(self,baseline,candidate,min_gain=.01)->bool:
        return candidate.train-baseline.train>=min_gain and candidate.held_out>0 and candidate.held_out>=baseline.held_out and candidate.regression_ok
