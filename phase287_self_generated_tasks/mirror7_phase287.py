from dataclasses import dataclass
import random
from typing import Any, Dict, List, Sequence

@dataclass(frozen=True)
class TaskFamily:
    name: str
    public: Dict[str, Any]
    target: Any

class SelfGeneratedTaskSuite:
    """Creates deterministic hidden-target task families from a small internal grammar."""
    def __init__(self, seed: int):
        self.rng=random.Random(seed)

    def linear_sequence(self, count:int=6)->List[TaskFamily]:
        if count<1: raise ValueError("count must be positive")
        out=[]
        for _ in range(count):
            start=self.rng.randint(-20,20); step=self.rng.randint(1,8); length=self.rng.randint(3,6)
            values=[start+j*step for j in range(length)]
            out.append(TaskFamily("linear",{"values":values[:-1]},values[-1]))
        return out

    def mixed_suite(self, count:int=4)->List[TaskFamily]:
        if count<1: raise ValueError("count must be positive")
        out=self.linear_sequence(count)
        for _ in range(count):
            a=self.rng.randint(-10,10); b=self.rng.randint(-10,10)
            out.append(TaskFamily("difference",{"values":[a,b]},b+(b-a)))
        return out

    @staticmethod
    def as_evaluation_pack(tasks:Sequence[TaskFamily],held_out:Sequence[TaskFamily],regression:Sequence[TaskFamily]):
        from phase280_sealed_candidate_eval import EvaluationPack
        return EvaluationPack(
            [{"public":t.public,"target":t.target} for t in tasks],
            [{"public":t.public,"target":t.target} for t in held_out],
            [{"public":t.public,"target":t.target} for t in regression],
        )
