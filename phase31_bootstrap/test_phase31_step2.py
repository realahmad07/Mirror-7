from mirror7_representation import discover_representation
from mirror7_state import extract_state


def test_three_seeds():
    samples = [
        bytes([1,1,2,3,3,2,4,4,1,1,2,3,3,2,4,4]),
        bytes([9,9,7,8,8,7,6,6,9,9,7,8,8,7,6,6]),
        bytes([20,20,30,40,40,30,50,50,20,20,30,40,40,30,50,50]),
    ]
    states = [extract_state(discover_representation(x)) for x in samples]
    assert states[0].canonical() == states[1].canonical() == states[2].canonical()


def test_held_out_different_length():
    train = bytes([1,1,2,3,3,2,4,4,1,1,2,3,3,2,4,4])
    held = bytes([100,100,101,102,102,101,103,103,100,100,101,102,102,101,103,103,100,100])
    a = extract_state(discover_representation(train))
    b = extract_state(discover_representation(held))
    assert a.run_lengths == b.run_lengths
    assert a.transition_count == b.transition_count
    assert a.motif_count == b.motif_count


def test_adversarial_negative():
    positive = bytes([1,1,2,3,3,2,4,4,1,1,2,3,3,2,4,4])
    negative = bytes(range(16))
    a = extract_state(discover_representation(positive))
    b = extract_state(discover_representation(negative))
    assert a.canonical() != b.canonical()


def test_invalid_representation():
    try:
        extract_state(object())
    except TypeError:
        return
    raise AssertionError("invalid representation was accepted")


if __name__ == "__main__":
    test_three_seeds()
    test_held_out_different_length()
    test_adversarial_negative()
    test_invalid_representation()
    print("PHASE31_STEP2_PASS: state extraction gate")
