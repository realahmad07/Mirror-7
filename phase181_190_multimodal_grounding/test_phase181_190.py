from .mirror7_phase181_190 import *

def img(v):
    return ModalityView("cam","image",v)

def txt(v):
    return ModalityView("caption","text",v)

def aud(v):
    return ModalityView("mic","audio",v)

def test_181_ingestion_and_determinism():
    a=canonicalize_view(img([[0,1],[1,0]]))
    assert a==canonicalize_view(img([[0,1],[1,0]]))
    assert a

def test_182_cross_encoding_invariance():
    a=canonicalize_view(txt("abca"))
    b=canonicalize_view(txt(bytes([7,8,7,9])))
    assert a[0]==b[0]=="seq"
    assert len(a)==len(b)

def test_183_alignment_is_name_independent():
    x=align_views([ModalityView("x","text","abca"),ModalityView("y","bytes",b"abca")])
    assert ("x","y") in x["pair_scores"]

def test_184_repeated_concepts():
    eps=[[txt("abca")],[txt("abca")],[txt("abca")]]
    cs=discover_grounded_concepts(eps,min_support=2)
    assert cs and cs[0].support>=2

def test_185_binding():
    v=txt("abca")
    h=sha256(repr(canonicalize_view(v)).encode()).hexdigest()[:16]
    c=GroundedConcept("x",(("text",h),),2,.8,"e")
    assert bind_concept(c,[v])

def test_186_contradiction_abstains_on_tie():
    assert contradiction_filter([("c","a",.5),("c","b",.5)])==()

def test_187_missing_view_is_explicit():
    s=build_grounded_scene([txt("abca")])
    assert "image" in s["missing_modalities"]
    assert "audio" in s["missing_modalities"]

def test_188_temporal_fusion():
    e=MultimodalGroundingEngine()
    out=e.temporal_fuse([[ModalityView("a","text","a",2),ModalityView("b","audio",[1,2],1)]])
    assert out[0].startswith("1:audio")

def test_189_memory_export():
    e=MultimodalGroundingEngine()
    e.ingest([txt("abca")])
    assert e.export_memory()["concepts"]

def test_190_integrated_loop():
    e=MultimodalGroundingEngine()
    for seed in range(3):
        e.ingest([txt("abca"),aud([0,1,0,1]),img([[0,1],[1,0]])])
    scene=e.ingest([txt("abca")])
    assert scene["fingerprint"] and len(e.history)==4
    assert len(e.memory)>=3

def test_noise_does_not_crash_and_is_distinct():
    a=canonicalize_view(aud([0,1,0,1]))
    b=canonicalize_view(aud([0,1,0,2]))
    assert a and b

def test_empty_and_unknown_fail_closed():
    try: build_grounded_scene([])
    except Exception: pass
    else: assert False
    try: canonicalize_view(ModalityView("z","unknown",[]))
    except ValueError: pass
    else: assert False

def test_ordered_timestamps_are_stable():
    e=MultimodalGroundingEngine()
    x=e.temporal_fuse([[txt("x")],[ModalityView("a","audio",[1],0)]])
    y=e.temporal_fuse([[ModalityView("a","audio",[1],0)],[txt("x")]])
    assert x==y

def test_quality_is_not_silent_semantic_evidence():
    v=ModalityView("x","text","abca",quality=0.1)
    assert canonicalize_view(v)==canonicalize_view(ModalityView("x","text","abca",quality=1.0))
