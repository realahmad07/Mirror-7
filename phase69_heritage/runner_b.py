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
    transitions = a['machine']['transitions']
    index = {}
    for item in transitions:
        index[f"{int(item['control'])}:{int(item['symbol'])}"] = item
    state = int(a['machine']['start_control'])
    halt = int(a['machine']['halt_control'])
    head = 0
    tape = list(map(int, doc['tape']))
    limit = int(a.get('payload', {}).get('max_steps', 4096))
    steps = 0
    while state != halt:
        if steps >= limit:
            raise RuntimeError('step budget exceeded')
        if not (0 <= head < len(tape)):
            raise RuntimeError('head out of range')
        key = f"{state}:{tape[head]}"
        transition = index.get(key)
        if transition is None:
            raise RuntimeError('missing transition')
        tape[head] = int(transition['write_symbol'])
        head += int(transition['move'])
        state = int(transition['next_control'])
        steps += 1
    return tape


if __name__ == '__main__':
    print(json.dumps(run(json.load(sys.stdin))))
