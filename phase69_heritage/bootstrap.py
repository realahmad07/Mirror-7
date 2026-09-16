from dataclasses import dataclass
import hashlib
import json
from typing import Dict, List, Tuple

KERNEL_OPS: Tuple[str, ...] = (
    'CHECK_HALT',
    'FETCH_SYMBOL',
    'LOOKUP_TRANSITION',
    'WRITE_SYMBOL',
    'MOVE_HEAD',
    'SET_CONTROL',
)

@dataclass(frozen=True)
class KernelStep:
    fetch_mode: str
    write_mode: str
    move_mode: str
    next_state_mode: str
    halt_mode: str

@dataclass(frozen=True)
class BootstrapArtifact:
    kernel: KernelStep
    machine: Dict
    payload: Dict
    version: int = 1
    kernel_ops: Tuple[str, ...] = KERNEL_OPS

    def to_dict(self):
        return {
            'version': self.version,
            'kernel': self.kernel.__dict__,
            'kernel_ops': list(self.kernel_ops),
            'machine': self.machine,
            'payload': self.payload,
        }

    @classmethod
    def from_dict(cls, d):
        ops = tuple(d.get('kernel_ops', KERNEL_OPS))
        return cls(
            KernelStep(**d['kernel']),
            dict(d['machine']),
            dict(d['payload']),
            int(d.get('version', 1)),
            ops,
        )

    def blob(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')).encode()

    def fingerprint(self):
        return hashlib.sha256(self.blob()).hexdigest()


def make_bootstrap_increment():
    transitions = []
    for d in range(9):
        transitions.append({'control': 0, 'symbol': d, 'next_control': 2, 'write_symbol': d + 1, 'move': 0})
    transitions.append({'control': 0, 'symbol': 9, 'next_control': 1, 'write_symbol': 0, 'move': 1})
    for d in range(9):
        transitions.append({'control': 1, 'symbol': d, 'next_control': 2, 'write_symbol': d + 1, 'move': 0})
    transitions.append({'control': 1, 'symbol': 9, 'next_control': 1, 'write_symbol': 0, 'move': 1})
    transitions.append({'control': 1, 'symbol': 10, 'next_control': 2, 'write_symbol': 1, 'move': 0})
    kernel = KernelStep('state_symbol_lookup', 'write_symbol', 'relative_head_move', 'next_control', 'halt_control')
    machine = {'start_control': 0, 'halt_control': 2, 'transitions': transitions, 'alphabet': list(range(11))}
    payload = {
        'encoding': 'little_endian_decimal',
        'closed_domain': True,
        'max_steps': 4096,
        'carrier_contract': list(KERNEL_OPS),
    }
    return BootstrapArtifact(kernel, machine, payload)


def execute_artifact(a: BootstrapArtifact, tape: List[int], max_steps=None) -> List[int]:
    missing = [op for op in KERNEL_OPS if op not in a.kernel_ops]
    if missing:
        raise RuntimeError(f'kernel contract incomplete: {missing}')
    d = a.machine
    table = {(int(t['control']), int(t['symbol'])): t for t in d['transitions']}
    control = int(d['start_control'])
    halt = int(d['halt_control'])
    pos = 0
    limit = int(max_steps if max_steps is not None else a.payload.get('max_steps', 4096))
    for _ in range(limit):
        if control == halt:
            return tape
        if not (0 <= pos < len(tape)):
            raise RuntimeError('head out of range')
        t = table.get((control, int(tape[pos])))
        if t is None:
            raise RuntimeError('missing transition')
        tape[pos] = int(t['write_symbol'])
        pos += int(t['move'])
        control = int(t['next_control'])
    raise RuntimeError('step budget exceeded')
