class AdapterRegistry:
    """Deterministic registry for all currently available capability-improvement adapters."""
    def __init__(self,adapters):
        self._adapters={}
        for adapter in adapters:
            name=getattr(adapter,"name","")
            if not name: raise ValueError("adapter requires a name")
            if name in self._adapters: raise ValueError("duplicate adapter: "+name)
            self._adapters[name]=adapter
    @property
    def names(self): return tuple(sorted(self._adapters))
    def get(self,name):
        return self._adapters[name]
