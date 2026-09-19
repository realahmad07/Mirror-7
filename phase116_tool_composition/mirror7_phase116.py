from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class CompositionReport:
    success: bool
    outputs: tuple[Any,...]
    failed_at: int
    reason: str

class ToolComposer:
    """Composes verified tool stages; failed stages stop downstream execution."""
    def run(self,stages:list[Callable[[Any],Any]],initial:Any=None)->CompositionReport:
        value=initial; outs=[]
        for i,stage in enumerate(stages):
            try: value=stage(value)
            except Exception as e: return CompositionReport(False,tuple(outs),i,f"stage error: {type(e).__name__}")
            if value is None: return CompositionReport(False,tuple(outs),i,"stage returned no result")
            outs.append(value)
        return CompositionReport(True,tuple(outs),-1,"verified")
