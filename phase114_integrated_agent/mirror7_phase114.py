from dataclasses import dataclass
from phase108_grounded_dialogue import GroundedDialogue
from phase109_knowledge_ingestion import ValidatedKnowledge
from phase110_tool_guard import GuardedTools
from phase111_task_decomposition import TaskDecomposer,TaskNode
from phase112_self_monitor import SelfMonitor
from phase113_continual_agent_memory import AgentMemory

@dataclass(frozen=True)
class AgentReport:
    needs_context: bool
    knowledge_accepted: bool
    plan: tuple[str,...]
    tool_verified: bool
    ask_context: bool

class IntegratedAgent:
    """Bounded interface joining grounding, learning, planning, tools, monitoring and memory."""
    def __init__(self):
        self.dialogue=GroundedDialogue(["platform"]); self.knowledge=ValidatedKnowledge(); self.tools=GuardedTools(); self.monitor=SelfMonitor(); self.memory=AgentMemory(); self.decomposer=TaskDecomposer()
    def run(self,context:dict, claim:str, value, source:str, tasks:list[TaskNode], tool_name:str|None=None, tool_args:dict|None=None):
        ds=self.dialogue.ingest("",goal=context.get("goal"),slots=context)
        kr=self.knowledge.ingest(claim,value,source,.9)
        plan=self.decomposer.order(tasks, budget=sum(t.cost for t in tasks))
        tr=self.tools.call(tool_name,tool_args or {}) if tool_name else None
        conf=.9 if kr.accepted and (tr is None or tr.verified) else .4
        self.monitor.observe(conf,True if conf>.7 else False)
        mr=self.monitor.report(conf)
        self.memory.remember(claim,value,conf)
        return AgentReport(bool(ds.unresolved or not ds.goal),kr.accepted,plan,True if tr is None else tr.verified,mr.ask_context)
