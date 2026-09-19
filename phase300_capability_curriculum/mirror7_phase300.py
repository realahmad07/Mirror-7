from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

@dataclass(frozen=True)
class CurriculumTask:
    capability:str
    task_id:str
    public:Dict[str,Any]
    target:Any
    difficulty:int

class CapabilityCurriculum:
    """Builds deterministic train/held-out/regression packs for one frontier capability."""
    def __init__(self,generator:Callable[[int],Iterable[Any]]):
        self.generator=generator

    def build(self,capability:str,seed:int,count:int=6)->List[CurriculumTask]:
        if not capability or count<1: raise ValueError("invalid curriculum request")
        raw=list(self.generator(seed))
        if len(raw)<count: raise ValueError("generator returned too few tasks")
        tasks=[]
        for i,item in enumerate(raw[:count]):
            public=dict(item.public)
            tasks.append(CurriculumTask(capability,f"{capability}:{seed}:{i}",public,item.target,i+1))
        return tasks

    def split(self,tasks:List[CurriculumTask])->tuple[list[CurriculumTask],list[CurriculumTask],list[CurriculumTask]]:
        if len(tasks)<3: raise ValueError("at least three tasks required")
        train=tasks[:-2]; held=[tasks[-2]]; regression=[tasks[-1]]
        return train,held,regression
