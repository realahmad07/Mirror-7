from dataclasses import dataclass

@dataclass(frozen=True)
class Checkpoint:
    index: int
    state: object

class CheckpointExecutor:
    """Runs bounded steps and can resume from the latest verified checkpoint."""
    def __init__(self): self.checkpoints=[]
    def run(self, steps, state, verify, fail_at=None):
        start=self.checkpoints[-1].index+1 if self.checkpoints else 0
        i=start
        try:
            for i in range(start,len(steps)):
                if fail_at is not None and i==fail_at:
                    raise RuntimeError("injected failure")
                state=steps[i](state)
                if not verify(state):
                    return False,state,"verification failed"
                self.checkpoints.append(Checkpoint(i,state))
            return True,state,"completed"
        except Exception as e:
            return False,state,f"paused at {i}: {type(e).__name__}"
    def latest(self):
        return self.checkpoints[-1] if self.checkpoints else None