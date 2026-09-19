from phase307_capability_adapter_contract import AdapterResult

class ReasoningFrontierAdapter:
    name="symbolic_reasoning"

    def __init__(self):
        self.depth=1

    def evaluate(self,depth:int)->float:
        tasks=[
            ((1,2),3),
            ((2,3),5),
            ((3,4),7),
            ((1,2,3),6),
            ((2,3,4),9),
        ]
        good=0
        for values,target in tasks:
            if depth==1:
                pred=sum(values) if len(values)<=2 else sum(values[:2])
            else:
                pred=sum(values)
            good+=int(pred==target)
        return good/len(tasks)

    def improve(self,seed:int,rounds:int,candidates:int)->AdapterResult:
        baseline=self.evaluate(self.depth)
        score=self.evaluate(max(self.depth,2))
        if score>baseline and candidates>0:
            self.depth=2
            return AdapterResult(score,True,"depth=2")
        return AdapterResult(baseline,False,f"depth={self.depth}")
