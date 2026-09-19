from .mirror7_phase123 import StructuralTransfer

def test_isomorphic_degree_structure_transfers():
    t=StructuralTransfer().transfer({"a":("b",),"b":("a","c"),"c":("b",)},{"x":("y",),"y":("x","z"),"z":("y",)})
    assert t.transferred and set(t.mapping)=={"a","b","c"}

def test_mismatch_is_rejected():
    t=StructuralTransfer().transfer({"a":("b",),"b":("a","c"),"c":("b","d"),"d":("c",)},{"x":("y","z","w"),"y":("x",),"z":("x",),"w":("x",)})
    assert not t.transferred

def test_symmetric_structure_gets_deterministic_mapping():
    t=StructuralTransfer().transfer({"a":("b",),"b":("a",)},{"x":("y",),"y":("x",)})
    assert t.transferred and t.mapping=={"a":"x","b":"y"} and t.confidence<1

def test_mapping_is_bijective():
    t=StructuralTransfer().transfer({"a":("b",),"b":("a","c"),"c":("b",)},{"x":("y",),"y":("x","z"),"z":("y",)})
    assert t.transferred and len(set(t.mapping.values()))==3