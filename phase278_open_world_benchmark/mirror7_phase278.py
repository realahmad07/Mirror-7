
import random
from typing import Any, Callable, Dict, List, Tuple

class NovelBenchmark:
    """Generates held-out task families with opaque surface identifiers."""

    def __init__(self, seed:int):
        self.rng=random.Random(seed)

    def sequence(self,length:int=5)->Dict[str,Any]:
        start=self.rng.randint(-20,20); step=self.rng.randint(1,7)
        values=[start+i*step for i in range(length)]
        return {"family":"sequence","public":{"values":values[:-1],
                "action_name":f"act_{self.rng.randrange(10**6)}"},"target":values[-1]}

    def grid(self,size:int=5)->Dict[str,Any]:
        start=(0,0); goal=(size-1,size-1)
        actions={"R":f"a{self.rng.randrange(10**6)}","U":f"b{self.rng.randrange(10**6)}"}
        return {"family":"grid","public":{"start":start,"goal":goal,
                "actions":actions,"size":size},"target":["R"]*(size-1)+["U"]*(size-1)}

    def evaluate(self,solver:Callable[[Dict[str,Any]],Any],trials:int=6)->Tuple[int,int]:
        cases=[]
        for i in range(trials):
            cases.append(self.sequence(4+i%3))
            cases.append(self.grid(3+i%2))
        good=0
        for task in cases:
            try:
                out=solver(task["public"])
            except Exception:
                out=None
            if out==task["target"]:
                good+=1
        return good,len(cases)

def perfect_solver(public:Dict[str,Any]):
    if "values" in public:
        v=public["values"]; return v[-1]+(v[-1]-v[-2])
    if "start" in public:
        return ["R"]*(public["size"]-1)+["U"]*(public["size"]-1)
    return None
