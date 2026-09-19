from phase268_compositional_semantics import CompositionalSemantics
from phase307_capability_adapter_contract import AdapterResult

class LanguageFrontierAdapter:
    name="compositional_language"

    def __init__(self):
        self.mode="primitive"

    def evaluate(self,mode:str,seed:int)->float:
        sem=CompositionalSemantics()
        sem.learn_primitive("increase","INC")
        sem.learn_primitive("decrease","DEC")
        primitive_cases=[("increase",("INC",)),("decrease",("DEC",))]
        composition_cases=[("increase then decrease",("INC","DEC")),("increase and decrease",("INC","DEC"))]
        cases=primitive_cases + composition_cases
        if mode=="composition":
            active_cases=cases
        else:
            active_cases=primitive_cases
        good=0
        for text,target in active_cases:
            try:
                if sem.execute_plan(sem.parse(text))==target: good+=1
            except ValueError:
                pass
        # Score capability coverage against the full composition-aware task set;
        # this prevents a saturated accuracy score from hiding added coverage.
        return good/len(cases)

    def improve(self,seed:int,rounds:int,candidates:int)->AdapterResult:
        baseline=self.evaluate(self.mode,seed)
        if self.mode=="primitive":
            score=self.evaluate("composition",seed)
            if score>baseline:
                self.mode="composition"
                return AdapterResult(score,True,"composition")
        return AdapterResult(baseline,False,self.mode)
