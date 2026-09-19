import json
from mirror7_backend.persistence import CheckpointStore, CheckpointError


def test_checkpoint_round_trip(tmp_path):
    store = CheckpointStore(tmp_path)
    payload = {"version": 1, "session_id": "a", "sequence": 4, "state": {"x": 2}}
    path = store.save("a", payload)
    assert path.exists()
    assert store.load("a") == payload


def test_checkpoint_tamper_is_rejected(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save("a", {"x": 1})
    path = store.path_for("a")
    envelope = json.loads(path.read_text())
    envelope["payload"]["x"] = 999
    path.write_text(json.dumps(envelope))
    try:
        store.load("a")
    except CheckpointError:
        pass
    else:
        raise AssertionError("tampered checkpoint was accepted")


def test_path_traversal_is_rejected(tmp_path):
    store = CheckpointStore(tmp_path)
    for bad in ("../x", "..\\x", "", ".", ".."):
        try:
            store.path_for(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid session id: {bad!r}")
