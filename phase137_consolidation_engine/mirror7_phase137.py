from collections import defaultdict
class ConsolidationEngine:
    def __init__(self,max_rules=128,min_support=2):
        self.max_rules=max_rules; self.min_support=min_support; self.counts=defaultdict(lambda:defaultdict(int))
    def observe(self,key,value): self.counts[key][repr(value)]+=1
    def rules(self):
        out=[]
        for key,vals in self.counts.items():
            value,count=max(vals.items(),key=lambda x:x[1])
            if count>=self.min_support: out.append((key,value,count))
        return tuple(out[:self.max_rules])
