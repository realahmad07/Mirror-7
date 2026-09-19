from dataclasses import dataclass
from phase122_skill_library import SkillLibrary
from phase123_structural_transfer import StructuralTransfer
from phase124_novelty_detection import NoveltyDetector
from phase125_adaptive_curriculum import AdaptiveCurriculum
from phase126_checkpoint_executor import CheckpointExecutor
from phase127_generalization_gate import GeneralizationGate

@dataclass(frozen=True)
class AgentResult:
    skill_promoted: bool
    transferred: bool
    novel: bool
    curriculum: str|None
    completed: bool
    generalized: bool

class GeneralizationAgent:
    """Bounded integration of skills, transfer, novelty, curriculum, checkpoints and generalization."""
    def __init__(self):
        self.skills=SkillLibrary()
        self.transfer=StructuralTransfer()
        self.novelty=NoveltyDetector()
        self.curriculum=AdaptiveCurriculum()
        self.executor=CheckpointExecutor()
        self.generalizer=GeneralizationGate()
    def run(self,state,skill,skill_verified,source_graph,target_graph,observation,items,steps,scores):
        promoted=self.skills.promote(skill,skill_verified)
        tr=self.transfer.transfer(source_graph,target_graph)
        nv=self.novelty.observe(observation)
        item=self.curriculum.select(items)
        ok,_,_=self.executor.run(steps,state,lambda s:s is not None)
        gr=self.generalizer.evaluate(scores)
        return AgentResult(promoted,tr.transferred,nv.novel,item.name if item else None,ok,gr.passed)