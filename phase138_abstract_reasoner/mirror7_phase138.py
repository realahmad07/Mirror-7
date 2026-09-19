class AbstractReasoner:
    def infer_relation(self,left,right):
        if set(left.keys())!=set(right.keys()): return None
        return {k:right[k] for k in left}
    def compose(self,transformations):
        def apply(x):
            for f in transformations: x=f(x)
            return x
        return apply
