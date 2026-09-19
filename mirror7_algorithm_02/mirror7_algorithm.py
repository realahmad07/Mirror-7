"""
Compositional Program Induction
Infers reusable procedures from input-output examples.
"""

from typing import List, Any, Callable, Dict, Tuple, Optional
import time

class Primitive:
    def __init__(self, name: str, input_types: List[type], output_type: type, func: Callable):
        self.name = name
        self.input_types = input_types
        self.output_type = output_type
        self.func = func
        self.usage_count = 0

class Program:
    def __init__(self, op: Primitive, args: List[Any]):
        self.op = op
        self.args = args
        
    def evaluate(self, inputs: List[Any]) -> Any:
        eval_args = []
        for arg in self.args:
            if isinstance(arg, Program):
                eval_args.append(arg.evaluate(inputs))
            elif isinstance(arg, int) and arg < len(inputs):
                eval_args.append(inputs[arg])
            else:
                eval_args.append(arg)
        try:
            return self.op.func(*eval_args)
        except Exception:
            return None

    def explain(self) -> str:
        arg_strs = []
        for arg in self.args:
            if isinstance(arg, Program):
                arg_strs.append(arg.explain())
            elif isinstance(arg, int):
                arg_strs.append(f"input_{arg}")
            else:
                arg_strs.append(str(arg))
        return f"{self.op.name}({', '.join(arg_strs)})"

class ProgramSynthesis:
    def __init__(self, max_depth: int = 3, beam_width: int = 10, timeout_steps: int = 1000):
        self.library: List[Primitive] = [
            Primitive('add1', [int], int, lambda x: x + 1),
            Primitive('double', [int], int, lambda x: x * 2),
            Primitive('negate', [int], int, lambda x: -x)
        ]
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.timeout_steps = timeout_steps
        
    def synthesize(self, examples: List[Tuple[List[Any], Any]]) -> Optional[Program]:
        if not examples:
            return None
            
        input_types = [type(x) for x in examples[0][0]]
        
        current_beam = []
        
        for prim in self.library:
            if len(prim.input_types) == len(input_types):
                valid = True
                for t1, t2 in zip(prim.input_types, input_types):
                    if t1 != t2:
                        valid = False
                if valid:
                    prog = Program(prim, list(range(len(input_types))))
                    score = self._score(prog, examples)
                    current_beam.append((score, prog))
                    
        current_beam.sort(key=lambda x: x[0], reverse=True)
        current_beam = current_beam[:self.beam_width]
        
        steps = 0
        
        for depth in range(2, self.max_depth + 1):
            next_beam = []
            for score, prog in current_beam:
                if score == 1.0:
                    self._extract_library(prog)
                    return prog
                    
                for prim in self.library:
                    if len(prim.input_types) == 1 and prim.input_types[0] == prog.op.output_type:
                        new_prog = Program(prim, [prog])
                        new_score = self._score(new_prog, examples)
                        next_beam.append((new_score, new_prog))
                        steps += 1
                        if steps >= self.timeout_steps:
                            break
                if steps >= self.timeout_steps:
                    break
            
            if steps >= self.timeout_steps:
                break
                
            current_beam.extend(next_beam)
            current_beam.sort(key=lambda x: x[0], reverse=True)
            seen = set()
            unique_beam = []
            for s, p in current_beam:
                exp = p.explain()
                if exp not in seen:
                    seen.add(exp)
                    unique_beam.append((s, p))
            current_beam = unique_beam[:self.beam_width]
            
        if current_beam and current_beam[0][0] == 1.0:
            self._extract_library(current_beam[0][1])
            return current_beam[0][1]
            
        return current_beam[0][1] if current_beam else None

    def _score(self, prog: Program, examples: List[Tuple[List[Any], Any]]) -> float:
        correct = 0
        for inputs, output in examples:
            if prog.evaluate(inputs) == output:
                correct += 1
        return correct / len(examples)
        
    def _extract_library(self, prog: Program) -> None:
        prog.op.usage_count += 1
        
    def execute(self, prog: Program, inputs: List[Any]) -> Any:
        return prog.evaluate(inputs)
        
    def explain(self, prog: Program) -> str:
        return prog.explain()
