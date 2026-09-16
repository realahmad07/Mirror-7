import json
import sys

REQUIRED_KERNEL_OPS = {
    'CHECK_HALT', 'FETCH_SYMBOL', 'LOOKUP_TRANSITION',
    'WRITE_SYMBOL', 'MOVE_HEAD', 'SET_CONTROL',
}


def run(doc):
    a = doc['artifact']
    if not REQUIRED_KERNEL_OPS.issubset(set(a.get('kernel_ops', []))):
        raise RuntimeError('kernel contract incomplete')
    m = a['machine']
    tape = list(map(int, doc['tape']))
    table = {(int(x['control']), int(x['symbol'])): x for x in m['transitions']}
    control = int(m['start_control'])
    halt = int(m['halt_control'])
    head = 0
    for _ in range(int(a.get('payload', {}).get('max_steps', 4096))):
        if control == halt:
            return tape
        if not (0 <= head < len(tape)):
            raise RuntimeError('head out of range')
        transition = table.get((control, tape[head]))
        if transition is None:
            raise RuntimeError('missing transition')
        tape[head] = int(transition['write_symbol'])
        head += int(transition['move'])
        control = int(transition['next_control'])
    raise RuntimeError('step budget exceeded')


if __name__ == '__main__':
    print(json.dumps(run(json.load(sys.stdin))))
