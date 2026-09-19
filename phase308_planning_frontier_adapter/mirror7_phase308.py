from phase270_long_horizon_planning import LongHorizonPlanner
from phase307_capability_adapter_contract import AdapterResult

class PlanningFrontierAdapter:
    name="planning"

    def __init__(self):
        self.max_depth=2

    @staticmethod
    def _tasks(seed:int):
        size=2+(seed%2)
        return [(tuple([0,0]),(size-1,size-1),size)]

    def _solve(self,size:int,depth:int)->bool:
        goal=(size-1,size-1)
        def transition(state,action):
            x,y=state
            if action=="R" and x<size-1: return (x+1,y)
            if action=="U" and y<size-1: return (x,y+1)
            return None
        p=LongHorizonPlanner(transition,("R","U"),max_expansions=100)
        return p.plan((0,0),lambda s:s==goal,max_depth=depth) is not None

    def evaluate(self,depth:int,seed:int)->float:
        tasks=self._tasks(seed)
        return sum(int(self._solve(size,depth)) for _,_,size in tasks)/len(tasks)

    def improve(self,seed:int,rounds:int,candidates:int)->AdapterResult:
        baseline=self.evaluate(self.max_depth,seed)
        best=self.max_depth
        best_score=baseline
        for candidate in range(self.max_depth+1,self.max_depth+1+max(1,candidates)):
            score=self.evaluate(candidate,seed)
            if score>best_score:
                best,best_score=candidate,score
        if best_score>baseline:
            self.max_depth=best
            return AdapterResult(best_score,True,f"depth={best}")
        return AdapterResult(best_score,False,f"depth={self.max_depth}")
