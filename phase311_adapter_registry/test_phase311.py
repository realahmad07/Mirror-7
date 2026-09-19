from .mirror7_phase311 import AdapterRegistry
class A: name="a"
class B: name="b"

def test_registry_names_sorted():
    assert AdapterRegistry([B(),A()]).names==("a","b")

def test_registry_get():
    r=AdapterRegistry([A()]); assert r.get("a").name=="a"

def test_duplicate_rejected():
    try: AdapterRegistry([A(),A()])
    except ValueError: pass
    else: assert False

def test_missing_name_rejected():
    class X: pass
    try: AdapterRegistry([X()])
    except ValueError: pass
    else: assert False

def test_three_adapter_registry():
    assert len(AdapterRegistry([A(),B(),type("C",(),{"name":"c"})()]).names)==3
