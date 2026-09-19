
from .mirror7_phase267 import GeneralRepresentation

def test_progressive_repeated_objects():
    for raw in (b"red|blue|red", b"AA|BB|AA", b"x0x|y0y|x0x"):
        r = GeneralRepresentation(min_support=2).discover_batch([raw, raw])
        assert r[0].segments

def test_held_out_repetition_is_discovered():
    g = GeneralRepresentation(min_support=2)
    g.discover_batch([b"ABCD-ABCD", b"ABCD-XYZ"])
    held = g.discover(b"ABCD-Q")
    assert b"ABCD" in held.motifs

def test_noise_does_not_destroy_known_motif():
    g = GeneralRepresentation(min_support=2)
    g.discover_batch([b"NODE|NODE", b"NODE|N0DE"])
    r = g.discover(b"NODE|NOISE|NODE")
    assert b"NODE" in r.motifs
    assert len(r.segments) >= 2

def test_adversarial_singleton_is_not_promoted():
    g = GeneralRepresentation(min_support=3)
    r = g.discover_batch([b"RARE1", b"COMMON", b"COMMON"])
    assert b"RARE1" not in r[0].motifs
    assert b"COMMON" in r[0].motifs

def test_deterministic_representation():
    samples=[b"AB|CD|AB", b"CD|AB|CD"]
    a=GeneralRepresentation().discover_batch(samples)
    b=GeneralRepresentation().discover_batch(samples)
    assert a == b

def test_invalid_input_fails_closed():
    try:
        GeneralRepresentation().discover("not-bytes")
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError")
