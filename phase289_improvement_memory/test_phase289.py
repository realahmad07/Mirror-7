from .mirror7_phase289 import ImprovementMemory

def test_fingerprint_deterministic():
    m=ImprovementMemory(); assert m.fingerprint(["last","add_delta"])==m.fingerprint(["last","add_delta"])

def test_different_algorithms_differ():
    m=ImprovementMemory(); assert m.fingerprint(["last"])!=m.fingerprint(["delta"])

def test_seen_after_record():
    m=ImprovementMemory(); m.add(["last"],False,0,"bad"); assert m.seen(["last"])

def test_duplicate_record_suppressed():
    m=ImprovementMemory(); m.add(["last"],False,0,"bad"); m.add(["last"],True,1,"good"); assert len(m.records)==1

def test_three_variants_retained():
    m=ImprovementMemory()
    for op in (["last"],["delta"],["last","add_delta"]): m.add(op,True,.8,"ok")
    assert len(m.successful())==3

def test_records_immutable_view():
    m=ImprovementMemory(); m.add(["last"],True,.9,"ok"); assert isinstance(m.records,tuple)

def test_failed_and_successful_separated():
    m=ImprovementMemory(); m.add(["last"],False,.1,"bad"); m.add(["delta"],True,.8,"ok"); assert len(m.successful())==1

def test_empty_memory_safe():
    m=ImprovementMemory(); assert not m.seen(["last"]) and m.records==()

def test_basic_fingerprints_are_unique():
    m=ImprovementMemory(); fps={m.fingerprint([x]) for x in ["last","delta","add_delta","constant"]}; assert len(fps)==4
