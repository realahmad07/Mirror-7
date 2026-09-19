from phase81_persistent_transfer_memory import PersistentTransferMemory

def test_round_trip(tmp_path):
    p=tmp_path/"memory.json"; m=PersistentTransferMemory(p); m.add("A",(1,2),(0,1),"go",(1,2),.9); m.save()
    n=PersistentTransferMemory(p); assert len(n.retrieve("A",(1,2)))==1

def test_rejects_wrong_environment(tmp_path):
    m=PersistentTransferMemory(tmp_path/"m.json",tolerance=.1); m.add("A",(1,2),(0,1),"go",(1,2),1)
    assert m.retrieve("B",(1,2))==()

def test_rejects_shape_and_distance(tmp_path):
    m=PersistentTransferMemory(tmp_path/"m.json",tolerance=.1); m.add("A",(1,2),(0,1),"go",(1,2),1)
    assert m.retrieve("A",(1,2,3))==(); assert m.retrieve("A",(2,2))==()

def test_transfer_requires_confidence(tmp_path):
    m=PersistentTransferMemory(tmp_path/"m.json"); m.add("A",(1,),(0,),"go",(1,),.4)
    assert m.transfer("A","B",(1,),.5) is None
