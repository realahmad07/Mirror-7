
import json
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Tuple

class SealedEvaluator:
    """Runs an agent in a separate process and retains hidden expected answers locally."""

    def __init__(self, seed:int=0):
        self.seed=seed

    def evaluate(self, agent_script: str, tasks: Iterable[Dict[str,Any]]) -> Tuple[int,int]:
        cases=list(tasks)
        lines=[]
        expected=[]
        for t in cases:
            seq=list(t["sequence"])
            public={"sequence":seq}
            lines.append(json.dumps(public,sort_keys=True))
            expected.append(t["target"])
        proc=subprocess.run(
            [sys.executable, "-I", agent_script],
            input="\n".join(lines)+"\n",
            text=True,capture_output=True,check=False,timeout=10,
        )
        if proc.returncode!=0:
            return 0,len(cases)
        outputs=[line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if len(outputs)!=len(cases):
            return 0,len(cases)
        good=0
        for raw,truth in zip(outputs,expected):
            try:
                answer=json.loads(raw)
            except json.JSONDecodeError:
                continue
            if answer==truth: good+=1
        return good,len(cases)

def write_agent_script(body:str) -> str:
    f=NamedTemporaryFile("w",suffix=".py",delete=False,encoding="utf-8")
    f.write(body); f.close()
    return f.name
