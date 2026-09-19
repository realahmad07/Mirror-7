import random
from typing import Any, Dict

class ProceduralTaskGenerator:
    """Generates deterministic unseen tasks from parameterized rules."""
    def __init__(self, seed: int):
        if not isinstance(seed, int): raise TypeError("seed must be an int")
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_causal_environment(self, num_vars: int = 5, noise_level: float = 0.0) -> Dict[str, Any]:
        if num_vars < 2: raise ValueError("num_vars must be >= 2")
        variables=[f"V_{i}" for i in range(num_vars)]
        edges=[]
        for i in range(num_vars):
            for j in range(i+1,num_vars):
                if self.rng.random()>0.5:
                    edges.append((variables[i],variables[j],round(self.rng.uniform(-2.0,2.0),2)))
        return {"type":"causal_dag","variables":variables,"edges":edges,"noise_level":float(noise_level)}

    def generate_planning_world(self, grid_size: int = 10, num_obstacles: int = 15) -> Dict[str, Any]:
        if grid_size < 2: raise ValueError("grid_size must be >= 2")
        if not 0 <= num_obstacles < grid_size*grid_size-1: raise ValueError("num_obstacles leaves insufficient free cells")
        obstacles=set()
        while len(obstacles)<num_obstacles:
            obstacles.add((self.rng.randrange(grid_size),self.rng.randrange(grid_size)))
        free=[(x,y) for x in range(grid_size) for y in range(grid_size) if (x,y) not in obstacles]
        start=self.rng.choice(free); goal=self.rng.choice([p for p in free if p!=start])
        return {"type":"planning_grid","size":grid_size,"start":start,"goal":goal,"obstacles":sorted(obstacles)}

    def generate_sequence_task(self, length: int = 10) -> Dict[str, Any]:
        if length < 2: raise ValueError("length must be >= 2")
        start=self.rng.randint(-10,10); step=self.rng.randint(1,5)
        seq=[start+i*step for i in range(length)]
        return {"type":"sequence_extrapolation","sequence":seq[:-1],"target":seq[-1]}

class IndependentEvaluator:
    """Evaluates supported procedural tasks without exposing hidden evaluator fields."""
    def __init__(self, generator: ProceduralTaskGenerator): self.generator=generator

    def evaluate(self, agent_fn, task_type: str, num_trials: int = 5) -> float:
        if num_trials < 1: raise ValueError("num_trials must be >= 1")
        if task_type not in {"sequence"}:
            raise ValueError("only sequence is an independently solvable task in this bounded evaluator")
        successes=0
        for _ in range(num_trials):
            task=self.generator.generate_sequence_task()
            public={"sequence":list(task["sequence"])}
            result=agent_fn(public["sequence"])
            if result==task["target"]: successes+=1
        return successes/num_trials
