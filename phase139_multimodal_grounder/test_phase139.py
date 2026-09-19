from .mirror7_phase139 import MultimodalGrounder
def test_text_fingerprint(): assert MultimodalGrounder().fingerprint(" Hello  World ")=="text:hello world"
def test_sequence_fingerprint(): assert MultimodalGrounder().fingerprint((1,2))=="seq:1,2"
def test_shared_grounding_has_all_views():
 g=MultimodalGrounder().ground(["x",(1,2)]); assert g.support==2
