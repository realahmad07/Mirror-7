import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))
from bootstrap import KERNEL_OPS, execute_artifact, make_bootstrap_increment
from runner_a import run as run_a
from runner_b import run as run_b


def main():
    a = make_bootstrap_increment()
    doc = {'artifact': a.to_dict()}
    assert tuple(doc['artifact']['kernel_ops']) == KERNEL_OPS
    assert a.fingerprint() == hashlib.sha256(a.blob()).hexdigest()

    cases = [
        ([0, 10], [1, 10]),
        ([9, 10], [0, 1]),
        ([9, 9, 10], [0, 0, 1]),
        ([9, 9, 9, 10], [0, 0, 0, 1]),
        ([8, 10], [9, 10]),
    ]
    for tape, expected in cases:
        d = {**doc, 'tape': tape}
        assert execute_artifact(a, list(tape)) == expected
        assert run_a(d) == expected
        assert run_b(d) == expected

    bad = {**doc, 'tape': [42]}
    for fn in (
        lambda: execute_artifact(a, [42]),
        lambda: run_a(bad),
        lambda: run_b(bad),
    ):
        try:
            fn()
            raise AssertionError('missing transition accepted')
        except RuntimeError:
            pass

    incomplete = make_bootstrap_increment().to_dict()
    incomplete['kernel_ops'] = list(KERNEL_OPS[:-1])
    try:
        execute_artifact(type(a).from_dict(incomplete), [0, 10])
        raise AssertionError('incomplete kernel contract accepted')
    except RuntimeError as exc:
        assert 'kernel contract incomplete' in str(exc)

    with tempfile.TemporaryDirectory() as td:
        exe = pathlib.Path(td) / 'runner_c'
        src = ROOT / 'runner_c.c'
        subprocess.run(
            ['cc', '-std=c17', '-Wall', '-Wextra', '-Werror', str(src), '-o', str(exe)],
            check=True,
        )
        for tape, expected in cases[:4]:
            out = subprocess.check_output(
                [str(exe)],
                input=json.dumps({**doc, 'tape': tape}).encode(),
            ).decode().strip()
            assert json.loads(out) == expected

    print('PHASE69_HERITAGE_PASS')


if __name__ == '__main__':
    main()
