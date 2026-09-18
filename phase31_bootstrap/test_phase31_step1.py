from mirror7_representation import discover_representation, similarity


def permute_values(data: bytes, mapping: dict[int, int]) -> bytes:
    return bytes(mapping[x] for x in data)


def assert_shape(rep):
    assert rep.length > 0
    assert rep.runs
    assert rep.checksum


def test_seed(seed):
    base = bytes([1,1,2,3,3,2,4,4,1,1,2,3,3,2,4,4])
    maps = [
        {1:10+seed,2:20+seed,3:30+seed,4:40+seed},
        {1:80+seed,2:70+seed,3:60+seed,4:50+seed},
        {1:5+seed,2:35+seed,3:95+seed,4:125+seed},
    ]
    a = discover_representation(base)
    b = discover_representation(permute_values(base, maps[seed-1]))
    assert_shape(a); assert_shape(b)
    assert similarity(a, b) >= 0.80


def test_held_out():
    train = bytes([1,1,2,3,3,2,4,4,1,1,2,3,3,2,4,4])
    held = bytes([9,9,7,8,8,7,6,6,9,9,7,8,8,7,6,6,9,9,7,8,8,7])
    assert similarity(discover_representation(train), discover_representation(held)) >= 0.80


def test_adversarial_negative():
    positive = bytes([1,1,2,3,3,2,4,4,1,1,2,3,3,2,4,4])
    negative = bytes(range(16))
    assert similarity(discover_representation(positive), discover_representation(negative)) < 0.80


def test_invalid_input():
    try:
        discover_representation("not bytes")
    except TypeError:
        return
    raise AssertionError("non-bytes input was accepted")


if __name__ == "__main__":
    for seed in (1,2,3):
        test_seed(seed)
    test_held_out()
    test_adversarial_negative()
    test_invalid_input()
    print("PHASE31_STEP1_PASS: 3 seeds + held-out + adversarial + type rejection")
